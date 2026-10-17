"""Transcript stage (naryad 007): audio.wav -> transcript.json.

Speech becomes timed text through the external ASR provider (naryad 033
layer); the stage knows only the ASRProvider protocol, never a concrete
API. Long audio is split into parts with overlap: ``[transcript]
part_length_s`` seconds each, consecutive parts share ``overlap_s``
seconds so a phrase cut at a boundary survives in at least one part.
Parts are sliced from audio.wav with the stdlib wave module (ingest
guarantees 16 kHz mono PCM s16le, so slicing by frame index is exact and
lossless) and written to ``audio_parts/part_NNNN.wav`` for auditability.

Merging: every part's segments are shifted by the part's absolute offset
and sorted by time. Joints are deduplicated: a segment of a later part
that starts inside the overlap zone is dropped when an already accepted
segment overlaps it in time and the texts are similar (whitespace- and
case-normalized, then containment for clipped segments or a
SequenceMatcher ratio >= TEXT_SIMILARITY_THRESHOLD). Segments with
different text in the overlap zone are kept - never drop real content
silently; empty-text segments are dropped with a warning (schema
requires non-empty text; adapters normalize provider formats, naryad 007
assumption).

Language: the provider detects it (``lang`` unset) or the config forces
it (``[transcript] lang = "ru"``); the artifact language is the most
frequent part language, first on a tie. A source without an audio track
is not an error (naryad 007): a valid empty transcript (``lang: "und"``,
no segments) is written without touching the provider layer, so no cost
is booked. The same valid empty transcript results when the provider
recognizes no speech at all.

Cache and accounting: the stage cache key is the audio sha256 plus the
[transcript] params and the [asr] provider/model/model_version, so a
provider or config switch re-runs the stage. Every ASR call goes through
the provider layer - cached by audio hash, billed in seconds and
recorded into run.json (units, cost, cache_hit).
"""

from __future__ import annotations

import json
import re
import wave
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from razkadr.atomicio import write_json_atomic
from razkadr.log import get_logger
from razkadr.providers.base import ProviderConfigError
from razkadr.providers.layer import ProviderLayer
from razkadr.providers.types import Transcript
from razkadr.runctx import RunContext, file_sha256, read_cached_json
from razkadr.validation import validate_document

logger = get_logger("transcript")

TRANSCRIPT_JSON = "transcript.json"
AUDIO_NAME = "audio.wav"
SOURCE_JSON = "source.json"
PARTS_DIRNAME = "audio_parts"

# Whitespace/case-normalized texts: containment catches segments clipped at a
# part edge ("hello wor" vs "hello world"); the min length guards short texts
# from false containment hits. Ratio threshold covers minor ASR differences.
TEXT_SIMILARITY_THRESHOLD = 0.8
CONTAINMENT_MIN_CHARS = 6
TIME_EPS = 1e-9

TRANSCRIPT_DEFAULTS: dict = {
    "lang": "",                # force the language; empty = auto-detect by the provider
    "part_length_s": 120.0,    # max length of one ASR call, seconds
    "overlap_s": 2.0,          # shared audio between consecutive parts, seconds
}

LANG_RE = re.compile(r"^[A-Za-z]{2,8}(-[A-Za-z0-9]{1,8})*$")


class TranscriptError(Exception):
    """Transcript stage failure with a clear, user-facing message (CLI exit 2)."""


def transcript_params(config: dict) -> dict:
    """Effective [transcript] section with defaults; goes into the stage cache key."""
    section = {**TRANSCRIPT_DEFAULTS, **(config.get("transcript") or {})}
    lang = str(section["lang"]).strip()
    part_length_s = float(section["part_length_s"])
    overlap_s = float(section["overlap_s"])
    if lang and not LANG_RE.match(lang):
        raise TranscriptError(
            f"[transcript] lang must be a BCP-47 tag (e.g. ru, en-US), got {lang!r}"
        )
    if part_length_s <= 0:
        raise TranscriptError(f"[transcript] part_length_s must be positive, got {part_length_s}")
    if overlap_s < 0:
        raise TranscriptError(f"[transcript] overlap_s must be >= 0, got {overlap_s}")
    if overlap_s >= part_length_s:
        raise TranscriptError(
            f"[transcript] overlap_s ({overlap_s}) must be smaller than part_length_s ({part_length_s})"
        )
    return {"lang": lang, "part_length_s": part_length_s, "overlap_s": overlap_s}


def wav_duration_s(path: Path) -> float:
    """Duration of audio.wav in seconds (ingest guarantees a valid RIFF wav)."""
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        frames = handle.getnframes()
    if rate <= 0:
        raise TranscriptError(f"audio.wav has an unreadable sample rate: {path}")
    return frames / float(rate)


def split_parts(duration: float, params: dict) -> list[tuple[float, float]]:
    """Part spans [start, end] covering the whole audio; the last part takes the tail.

    Audio not longer than one part is a single part; audio exactly on the
    part boundary does not produce an empty tail part.
    """
    part = params["part_length_s"]
    step = part - params["overlap_s"]
    parts: list[tuple[float, float]] = []
    start = 0.0
    while True:
        end = start + part
        if end >= duration - TIME_EPS:
            parts.append((start, duration))
            return parts
        parts.append((start, end))
        start += step


def write_part(audio_path: Path, dst: Path, start_s: float, end_s: float) -> None:
    """Slice audio.wav into dst by frame index (exact for integer-rate PCM)."""
    with wave.open(str(audio_path), "rb") as src:
        rate = src.getframerate()
        total = src.getnframes()
        wav_params = src.getparams()
        first = min(total, max(0, round(start_s * rate)))
        last = min(total, max(first, round(end_s * rate)))
        src.setpos(first)
        frames = src.readframes(last - first)
    with wave.open(str(dst), "wb") as out:
        out.setparams(wav_params)
        out.writeframes(frames)


def _norm_text(text: str) -> str:
    return " ".join(text.split()).casefold()


def _similar_text(a: str, b: str) -> bool:
    na, nb = _norm_text(a), _norm_text(b)
    if not na or not nb:
        return na == nb
    if na == nb:
        return True
    long_enough = len(na) >= CONTAINMENT_MIN_CHARS and len(nb) >= CONTAINMENT_MIN_CHARS
    if long_enough and (na in nb or nb in na):
        return True
    return SequenceMatcher(None, na, nb).ratio() >= TEXT_SIMILARITY_THRESHOLD


def _overlaps(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] < b[1] - TIME_EPS and b[0] < a[1] - TIME_EPS


def merge_parts(
    transcripts: list[Transcript],
    offsets: list[float],
    overlap_s: float,
) -> tuple[list[dict], str]:
    """Shift parts by offsets, drop duplicates at joints, sort by time.

    Returns (segments, doc_lang); doc_lang is the most frequent part
    language, first part's language on a tie.
    """
    segments: list[dict] = []
    for transcript, offset in zip(transcripts, offsets, strict=True):
        overlap_end = offset + overlap_s
        for seg in transcript.segments:
            text = " ".join(seg.text.split())
            if not text:
                logger.warning("empty ASR segment dropped", extra={"fields": {"t_start": seg.t_start}})
                continue
            abs_start = round(offset + seg.t_start, 3)
            abs_end = round(offset + seg.t_end, 3)
            if offset > 0 and abs_start < overlap_end - TIME_EPS:
                # starts inside the overlap zone with the previous part:
                # drop when an accepted segment covers the same audio with similar text
                duplicate = any(
                    _overlaps((abs_start, abs_end), (a["t_start"], a["t_end"]))
                    and _similar_text(a["text"], text)
                    for a in segments
                )
                if duplicate:
                    continue
            segments.append({"t_start": abs_start, "t_end": abs_end, "text": text, "lang": seg.lang})
    segments.sort(key=lambda s: (s["t_start"], s["t_end"], s["text"]))
    counts: Counter[str] = Counter(t.lang for t in transcripts)
    first_seen: dict[str, int] = {}
    for i, t in enumerate(transcripts):
        first_seen.setdefault(t.lang, i)
    doc_lang = max(counts, key=lambda lang: (counts[lang], -first_seen[lang]))
    return segments, doc_lang


def run_transcript(ctx: RunContext, config: dict) -> dict:
    """Transcript stage body; returns the transcript.json content (schema-checked).

    Cache inputs: sha256 of audio.wav (or "none" for a silent source).
    Params carry the [transcript] settings plus the [asr] provider/model/
    model_version, so a provider or config switch invalidates the stage.
    The provider implementation is chosen by config (FACTORIES), the stage
    only knows the ASRProvider protocol (naryad 007 auditor check).
    """
    params = transcript_params(config)

    source_path = ctx.dir / SOURCE_JSON
    if not source_path.is_file():
        raise TranscriptError("transcript stage needs ingest output: source.json is missing in the run directory")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    audio_path = ctx.dir / AUDIO_NAME

    audio_sha: str | None = None
    if any(s.get("type") == "audio" for s in source.get("streams", [])):
        if not audio_path.is_file():
            raise TranscriptError("audio.wav is missing but the source has an audio track (re-run ingest)")
        audio_sha = file_sha256(audio_path)

    asr_section = config.get("asr") or {}
    stage_params = {
        **params,
        "asr_provider": asr_section.get("provider"),
        "asr_model": asr_section.get("model"),
        "asr_model_version": asr_section.get("model_version"),
    }
    inputs = {"audio_sha256": audio_sha} if audio_sha else {"audio": "none"}

    with ctx.stage("transcript", inputs=inputs, params=stage_params) as st:
        if st.cached:
            return read_cached_json(ctx.dir / TRANSCRIPT_JSON, "transcript.json", TranscriptError)

        if audio_sha is None:
            spans: list[tuple[float, float]] = []
            data = {"schema_version": "1.0", "lang": "und", "segments": []}
        else:
            try:
                service = ProviderLayer(config, ctx).asr(stage="transcript")
            except ProviderConfigError as exc:
                raise TranscriptError(f"{exc} (the transcript stage needs the [asr] section; see docs/config.md)") from None
            forced_lang = params["lang"] or None
            duration = wav_duration_s(audio_path)
            spans = split_parts(duration, params)
            parts_dir = ctx.dir / PARTS_DIRNAME
            parts_dir.mkdir(exist_ok=True)
            transcripts: list[Transcript] = []
            for i, (start, end) in enumerate(spans):
                part_path = parts_dir / f"part_{i:04d}.wav"
                write_part(audio_path, part_path, start, end)
                transcripts.append(service.transcribe(part_path, lang=forced_lang))
            segments, doc_lang = merge_parts(transcripts, [start for start, _end in spans], params["overlap_s"])
            data = {"schema_version": "1.0", "lang": doc_lang, "segments": segments}

        issues = validate_document(data, "transcript")
        if issues:
            raise TranscriptError("transcript.json failed schema validation: " + "; ".join(issues))
        write_json_atomic(ctx.dir / TRANSCRIPT_JSON, data)
        st.outputs = {
            "json": TRANSCRIPT_JSON,
            "parts": len(spans),
            "parts_dir": PARTS_DIRNAME if spans else None,
        }
        logger.info(
            "transcript done",
            extra={"fields": {"lang": data["lang"], "segments": len(data["segments"]), "parts": len(spans)}},
        )
        return data

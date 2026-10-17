"""Chunks stage (naryad 008): transcript.json -> chunks.json.

Transcript segments are grouped into thematic chunks for search and
captions. Boundaries are decided by the LLM through the naryad 033
provider layer; the stage knows only the LLMProvider protocol. The LLM
answer is never trusted with text: it only picks boundaries as segment
indexes, and the stage assembles chunk text, timecodes and language
from the transcript itself. Coverage without gaps or overlaps and
"segments are never cut mid-way" therefore hold by construction, and
the concatenation of chunk texts equals the transcript text (naryad
008 auditor check).

The answer must be a JSON object ``{"boundaries": [0, ...]}``: indexes
of the segments that start a new chunk, first is 0, strictly
increasing, each a valid segment index; a multi-segment chunk over
``max_chars`` also invalidates the answer. Anything invalid falls back
to the deterministic split by pauses and length, with the reason logged
as a warning (naryad 008 assumption). The fallback is also the whole
stage when ``[chunks] method = "fallback"`` or when no ``[llm]``
provider is configured at all; a configured-but-broken LLM (unknown
provider name, missing key, auth/quota failure) fails the stage loudly:
explicit configuration must not degrade silently.

A transcript without segments (silent source, naryad 007) yields a
valid empty chunks.json without touching the provider layer, so no cost
is booked.

Cache and accounting: the stage cache key is the transcript.json sha256
plus the [chunks] params and the [llm] provider/model/model_version, so
a provider or config switch re-runs the stage. Every LLM call is billed
in tokens via [llm.tariff] and recorded into run.json (units, cost,
cache_hit).
"""

from __future__ import annotations

import json
from collections import Counter
from importlib import resources
from itertools import pairwise

from razkadr.atomicio import write_json_atomic
from razkadr.log import get_logger
from razkadr.providers.base import ProviderConfigError
from razkadr.providers.layer import ProviderLayer
from razkadr.providers.types import Message
from razkadr.runctx import RunContext, file_sha256, read_cached_json
from razkadr.validation import validate_document

logger = get_logger("chunks")

CHUNKS_JSON = "chunks.json"
TRANSCRIPT_JSON = "transcript.json"
PROMPT_NAME = "chunking.v1.md"
PROMPT_MAX_CHARS_PLACEHOLDER = "{{max_chars}}"

METHODS = ("llm", "fallback")
TIME_EPS = 1e-9

CHUNKS_DEFAULTS: dict = {
    "method": "llm",      # llm = boundaries via provider (fallback on a bad answer); fallback = never calls the provider
    "max_chars": 1200,    # chunk length cap, characters; a multi-segment chunk over the cap invalidates the LLM answer
    "pause_s": 1.5,       # gap between consecutive segments that closes a chunk (fallback path)
}


class ChunksError(Exception):
    """Chunks stage failure with a clear, user-facing message (CLI exit 2)."""


def chunks_params(config: dict) -> dict:
    """Effective [chunks] section with defaults; goes into the stage cache key."""
    section = {**CHUNKS_DEFAULTS, **(config.get("chunks") or {})}
    method = str(section["method"]).strip()
    max_chars = int(section["max_chars"])
    pause_s = float(section["pause_s"])
    if method not in METHODS:
        raise ChunksError(f"[chunks] method must be one of {METHODS}, got {method!r}")
    if max_chars <= 0:
        raise ChunksError(f"[chunks] max_chars must be positive, got {max_chars}")
    if pause_s < 0:
        raise ChunksError(f"[chunks] pause_s must be >= 0, got {pause_s}")
    return {"method": method, "max_chars": max_chars, "pause_s": pause_s}


def load_chunking_prompt() -> str:
    """Versioned prompt packaged with the library (naryad 008 acceptance)."""
    return resources.files("razkadr.prompts").joinpath(PROMPT_NAME).read_text(encoding="utf-8")


def system_prompt(max_chars: int) -> str:
    """Prompt v1 with the chunk length limit substituted."""
    return load_chunking_prompt().replace(PROMPT_MAX_CHARS_PLACEHOLDER, str(max_chars))


def user_prompt(segments: list[dict]) -> str:
    """Numbered transcript segments; the numbers are the boundary vocabulary."""
    return "\n".join(
        f"{i} [{s['t_start']:g}-{s['t_end']:g}] {s['text']}" for i, s in enumerate(segments)
    )


def parse_boundaries(raw: str, n_segments: int) -> list[int] | None:
    """Validate the LLM answer; None means invalid (fallback, reason logged).

    Valid: a JSON object {"boundaries": [...]} where boundaries are
    segment indexes that start a new chunk - first is 0, strictly
    increasing, each within [0, n_segments - 1]. Booleans are rejected
    even though bool is an int subtype in Python.
    """
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    boundaries = data.get("boundaries")
    if not isinstance(boundaries, list) or not boundaries:
        return None
    if any(isinstance(b, bool) or not isinstance(b, int) for b in boundaries):
        return None
    if boundaries[0] != 0 or any(b <= a for a, b in pairwise(boundaries)):
        return None
    if any(b >= n_segments for b in boundaries):
        return None
    return boundaries


def build_chunks(segments: list[dict], boundaries: list[int]) -> list[dict]:
    """Assemble chunks from whole-segment slices; boundaries never cut a segment.

    Chunk language is the most frequent segment language in the chunk,
    the first segment's language on a tie (same rule as the transcript
    document language, naryad 007).
    """
    edges = [*boundaries, len(segments)]
    chunks: list[dict] = []
    for k, (first, last) in enumerate(pairwise(edges)):
        part = segments[first:last]
        counts: Counter[str] = Counter(s["lang"] for s in part)
        first_seen: dict[str, int] = {}
        for offset, s in enumerate(part):
            first_seen.setdefault(s["lang"], offset)
        lang = max(counts, key=lambda lang: (counts[lang], -first_seen[lang]))
        chunks.append({
            "id": f"c{k + 1}",
            "t_start": part[0]["t_start"],
            "t_end": part[-1]["t_end"],
            "text": " ".join(s["text"] for s in part),
            "lang": lang,
            "segment_ids": list(range(first, last)),
        })
    return chunks


def fallback_boundaries(segments: list[dict], max_chars: int, pause_s: float) -> list[int]:
    """Deterministic split without an LLM: by pauses and length (naryad 008).

    The chunk closes before segment i when the pause before it is
    >= pause_s or when adding it would push the chunk over max_chars.
    A lone over-long segment becomes its own chunk: segments are never
    cut mid-way.
    """
    boundaries = [0]
    chars = len(segments[0]["text"])
    for i in range(1, len(segments)):
        gap = segments[i]["t_start"] - segments[i - 1]["t_end"]
        if gap >= pause_s or chars + 1 + len(segments[i]["text"]) > max_chars:
            boundaries.append(i)
            chars = len(segments[i]["text"])
        else:
            chars += 1 + len(segments[i]["text"])
    return boundaries


def _over_limit(chunks: list[dict], max_chars: int) -> bool:
    """A multi-segment chunk over the cap means the LLM ignored the limit.

    Single-segment chunks are exempt: an over-long ASR segment cannot be
    split without cutting it mid-way, the fallback keeps it whole too.
    """
    return any(len(c["segment_ids"]) > 1 and len(c["text"]) > max_chars for c in chunks)


def _coverage_issues(chunks: list[dict], segments: list[dict]) -> list[str]:
    """Internal invariants both paths must satisfy; a violation is a bug.

    Checks the contract "chunks cover the transcript without gaps and
    overlaps": segment_ids partition the transcript segments in order,
    and every chunk's timecodes are its first/last segment's timecodes.
    (Timeline gaps between chunks exist where the transcript itself has
    pauses - the fallback closes chunks on pauses on purpose.)
    """
    issues: list[str] = []
    ids = [i for c in chunks for i in c["segment_ids"]]
    if ids != list(range(len(segments))):
        issues.append("segment_ids do not partition the transcript segments in order")
    for c in chunks:
        part = [segments[i] for i in c["segment_ids"]]
        if abs(c["t_start"] - part[0]["t_start"]) > TIME_EPS or abs(c["t_end"] - part[-1]["t_end"]) > TIME_EPS:
            issues.append(f"{c['id']} timecodes do not match its first/last segment")
    return issues


def run_chunks(ctx: RunContext, config: dict) -> dict:
    """Chunks stage body; returns the chunks.json content (schema-checked).

    Cache inputs: sha256 of transcript.json. Params carry the [chunks]
    settings plus the [llm] provider/model/model_version, so a provider
    or config switch invalidates the stage. The provider implementation
    is chosen by config (FACTORIES), the stage only knows the
    LLMProvider protocol (naryad 008, same split as transcript/ASR).
    """
    params = chunks_params(config)

    transcript_path = ctx.dir / TRANSCRIPT_JSON
    if not transcript_path.is_file():
        raise ChunksError(
            "chunks stage needs the transcript stage output: transcript.json is missing in the run directory"
        )
    try:
        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ChunksError(f"transcript.json is not valid JSON: {exc}") from None
    if validate_document(transcript, "transcript"):
        raise ChunksError("transcript.json is not a valid transcript artifact (re-run the transcript stage)")
    segments = transcript.get("segments", [])

    llm_section = config.get("llm") or {}
    stage_params = {
        **params,
        "llm_provider": llm_section.get("provider"),
        "llm_model": llm_section.get("model"),
        "llm_model_version": llm_section.get("model_version"),
    }

    with ctx.stage("chunks", inputs={"transcript_sha256": file_sha256(transcript_path)},
                   params=stage_params) as st:
        if st.cached:
            return read_cached_json(ctx.dir / CHUNKS_JSON, "chunks.json", ChunksError)

        if not segments:
            data: dict = {"schema_version": "1.0", "chunks": []}
            method_used = "empty"
        else:
            boundaries: list[int] | None = None
            reason: str | None = None
            if params["method"] == "fallback":
                reason = "method = fallback"
            elif not llm_section.get("provider"):
                reason = "no [llm] provider configured"
            else:
                try:
                    service = ProviderLayer(config, ctx).llm(stage="chunks")
                except ProviderConfigError as exc:
                    raise ChunksError(f"{exc} (the [llm] section is configured; see docs/config.md)") from None
                answer = service.complete([
                    Message(role="system", content=system_prompt(params["max_chars"])),
                    Message(role="user", content=user_prompt(segments)),
                ])
                boundaries = parse_boundaries(answer, len(segments))
                if boundaries is None:
                    reason = "invalid LLM answer (expected {\"boundaries\": [0, ...]})"

            chunks: list[dict] = []
            if boundaries is not None:
                chunks = build_chunks(segments, boundaries)
                if _over_limit(chunks, params["max_chars"]):
                    boundaries = None
                    reason = f"LLM chunk exceeds max_chars ({params['max_chars']})"
            if boundaries is None:
                logger.warning("chunking fallback", extra={"fields": {"reason": reason}})
                chunks = build_chunks(
                    segments,
                    fallback_boundaries(segments, params["max_chars"], params["pause_s"]),
                )
                method_used = "fallback"
            else:
                method_used = "llm"
            data = {"schema_version": "1.0", "chunks": chunks}

        issues = validate_document(data, "chunks")
        if issues:
            raise ChunksError("chunks.json failed schema validation: " + "; ".join(issues))
        invariant = _coverage_issues(data["chunks"], segments)
        if invariant:
            raise ChunksError("internal chunking invariant violated: " + "; ".join(invariant))

        write_json_atomic(ctx.dir / CHUNKS_JSON, data)
        st.outputs = {"json": CHUNKS_JSON, "chunks": len(data["chunks"]), "method": method_used}
        logger.info(
            "chunks done",
            extra={"fields": {"chunks": len(data["chunks"]), "method": method_used}},
        )
        return data

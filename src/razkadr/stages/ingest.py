"""Ingest stage (naryad 005): local file or URL -> normalized.mp4, audio.wav, source.json.

Normalization (ffmpeg): mp4 container, mpeg4 video (q:v 3), height capped at
``[ingest] max_height`` (aspect preserved, width kept even), fps capped at
``[ingest] fps``, audio re-encoded to AAC when present; ``audio.wav`` is
always 16 kHz mono PCM s16le, extracted from the normalized video so the
timelines match. A source without an audio track is not an error: no
audio.wav is written and ``streams`` in source.json simply has no audio
entry. URLs are downloaded with yt-dlp (``python -m yt_dlp`` subprocess;
tests replace ``_run_ytdlp``). The fps filter also turns VFR input into CFR.

Clear errors (IngestError subclasses, CLI exit code 2): ffmpeg/ffprobe
missing, input file missing, corrupted media, URL unavailable/deleted/private
or network failure, duration/size limits exceeded (``[ingest]
max_duration_s`` / ``max_file_size_mb``).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path

from razkadr.log import get_logger
from razkadr.runctx import RunContext, file_sha256
from razkadr.validation import validate_document

logger = get_logger("ingest")

NORMALIZED_NAME = "normalized.mp4"
AUDIO_NAME = "audio.wav"
SOURCE_JSON = "source.json"
ORIGINAL_NAME = "original.mp4"

AUDIO_RATE_HZ = 16000
AUDIO_CHANNELS = 1

INGEST_DEFAULTS: dict = {
    "fps": 10,                # fps cap (CFR output)
    "max_height": 1080,       # height cap in px, aspect preserved
    "max_duration_s": 3600,   # reject longer sources
    "max_file_size_mb": 2048,  # reject bigger sources
}

LANG_RE = re.compile(r"^[A-Za-z]{2,8}(-[A-Za-z0-9]{1,8})*$")

STREAM_TYPES = {"video": "video", "audio": "audio", "subtitle": "subtitle"}


class IngestError(Exception):
    """Ingest failure with a clear, user-facing message (CLI exit code 2)."""


class FfmpegMissingError(IngestError):
    """ffmpeg/ffprobe binary not found in PATH."""


class InputNotFoundError(IngestError):
    """Local input file does not exist."""


class MediaInvalidError(IngestError):
    """File is corrupted or is not a readable media file."""


class SourceUnavailableError(IngestError):
    """URL cannot be downloaded: unreachable, deleted, private, no network."""


class LimitExceededError(IngestError):
    """Source exceeds the configured duration or size limit."""


def ingest_params(config: dict) -> dict:
    """Effective [ingest] section with defaults; goes into the stage cache key."""
    section = {**INGEST_DEFAULTS, **(config.get("ingest") or {})}
    return {
        "fps": float(section["fps"]),
        "max_height": int(section["max_height"]),
        "max_duration_s": float(section["max_duration_s"]),
        "max_file_size_mb": float(section["max_file_size_mb"]),
    }


def _binaries() -> tuple[str, str]:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        missing = "ffprobe" if ffprobe is None else "ffmpeg"
        raise FfmpegMissingError(f"{missing} not found in PATH; install ffmpeg (https://ffmpeg.org)")
    return ffmpeg, ffprobe


def _first_line(proc: subprocess.CompletedProcess) -> str:
    return ((proc.stdout or "") or (proc.stderr or "")).splitlines()[0].strip()


def _ffmpeg_version(ffmpeg: str) -> str:
    proc = subprocess.run([ffmpeg, "-version"], capture_output=True, text=True, check=False)
    return _first_line(proc)


def _ytdlp_version() -> str:
    try:
        return package_version("yt-dlp")
    except PackageNotFoundError:
        raise IngestError("yt-dlp is not installed; install dependencies: pip install -e .") from None


def _probe(ffprobe: str, path: Path) -> dict:
    proc = subprocess.run(
        [ffprobe, "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, check=False,
    )
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        data = None
    if proc.returncode != 0 or not isinstance(data, dict) or not data.get("streams"):
        raise MediaInvalidError(f"cannot read media file: {path} (corrupted or not a media file)")
    return data


def _parse_rate(rate: str | None) -> float:
    try:
        num, _, den = (rate or "").partition("/")
        return float(num) / (float(den) if den else 1.0)
    except (ValueError, ZeroDivisionError):
        return 0.0


@dataclass(frozen=True)
class MediaInfo:
    duration_s: float
    fps: float
    width: int
    height: int
    streams: list[dict]  # schema shape: index, type, codec, language?
    has_audio: bool


def _media_info(probe: dict) -> MediaInfo:
    streams = probe.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise MediaInvalidError("no video stream found (audio-only file or an image?)")
    fmt = probe.get("format", {})
    duration = 0.0
    for raw in (fmt.get("duration"), video.get("duration")):
        try:
            duration = float(raw)
            break
        except (TypeError, ValueError):
            continue
    fps = _parse_rate(video.get("avg_frame_rate")) or _parse_rate(video.get("r_frame_rate"))
    if fps <= 0:
        raise MediaInvalidError("video has zero or unreadable frame rate")
    width, height = int(video.get("width") or 0), int(video.get("height") or 0)
    if width <= 0 or height <= 0:
        raise MediaInvalidError("video has unreadable frame dimensions")

    mapped: list[dict] = []
    for s in streams:
        kind = STREAM_TYPES.get(s.get("codec_type", ""))
        if kind is None:
            continue  # data/attachment streams are outside the schema enum
        entry: dict = {"index": int(s.get("index", 0)), "type": kind, "codec": str(s.get("codec_name", "unknown"))}
        lang = (s.get("tags") or {}).get("language")
        if lang and LANG_RE.match(lang):
            entry["language"] = lang
        mapped.append(entry)
    has_audio = any(e["type"] == "audio" for e in mapped)
    return MediaInfo(duration_s=duration, fps=fps, width=width, height=height, streams=mapped, has_audio=has_audio)


def _check_limits(info: MediaInfo, size_bytes: int, params: dict, label: str) -> None:
    if info.duration_s > params["max_duration_s"]:
        raise LimitExceededError(
            f"{label}: duration {info.duration_s:.1f} s exceeds the limit "
            f"{params['max_duration_s']:.0f} s (ingest.max_duration_s)"
        )
    limit = params["max_file_size_mb"] * (1 << 20)
    if size_bytes > limit:
        raise LimitExceededError(
            f"{label}: file size {size_bytes / (1 << 20):.1f} MB exceeds the limit "
            f"{params['max_file_size_mb']:.0f} MB (ingest.max_file_size_mb)"
        )


KNOWN_FAILURES: tuple[tuple[str, str], ...] = (
    ("video unavailable", "video is unavailable (deleted or wrong id)"),
    ("private video", "video is private"),
    ("members-only", "video is members-only"),
    ("sign in to confirm", "provider requires sign-in/age confirmation; try another URL or update yt-dlp"),
    ("unsupported url", "URL is not supported by yt-dlp"),
    ("failed to resolve", "network error: cannot resolve the host (offline?)"),
    ("unable to download", "download failed (network or platform error)"),
)


def _download_failure_reason(stderr: str, url: str) -> str:
    lowered = (stderr or "").lower()
    for marker, reason in KNOWN_FAILURES:
        if marker in lowered:
            return f"download failed: {reason} ({url})"
    tail = " ".join(stderr.split())[-200:]
    return f"download failed for {url}" + (f": {tail}" if tail else "; check the URL, network and yt-dlp version")


def _run_ytdlp(url: str, target: Path) -> None:
    """Download url into target (mp4). Module-level so tests can replace it."""
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--no-playlist", "--no-progress", "--no-warnings",
        "-f", "bv*+ba/b", "--merge-output-format", "mp4",
        "-o", str(target), url,
    ]
    logger.info("downloading", extra={"fields": {"url": url}})
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not target.is_file():
        raise SourceUnavailableError(_download_failure_reason(proc.stderr, url))


def _run_ffmpeg(ffmpeg: str, cmd: list[str], what: str) -> None:
    proc = subprocess.run([ffmpeg, *cmd], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        tail = " ".join((proc.stderr or "").split())[-300:]
        raise IngestError(f"ffmpeg {what} failed: {tail}")


def _normalize(ffmpeg: str, src: Path, dst: Path, info: MediaInfo, params: dict) -> None:
    """Re-encode into the unified format with caps applied (VFR -> CFR)."""
    filters = []
    height = min(info.height, params["max_height"])
    if height < info.height:
        filters.append(f"scale=-2:{height}")  # -2: keep width even, aspect preserved
    fps = min(info.fps, params["fps"])
    if fps < info.fps:
        filters.append(f"fps={fps}")
    cmd = ["-y", "-i", str(src)]
    if filters:
        cmd += ["-vf", ",".join(filters)]
    cmd += ["-c:v", "mpeg4", "-q:v", "3", "-pix_fmt", "yuv420p"]
    cmd += ["-c:a", "aac", "-b:a", "128k"] if info.has_audio else ["-an"]
    cmd.append(str(dst))
    _run_ffmpeg(ffmpeg, cmd, "normalization")


def _extract_audio(ffmpeg: str, src: Path, dst: Path) -> None:
    _run_ffmpeg(
        ffmpeg,
        ["-y", "-i", str(src), "-vn", "-ac", str(AUDIO_CHANNELS), "-ar", str(AUDIO_RATE_HZ),
         "-c:a", "pcm_s16le", str(dst)],
        "audio extraction",
    )


def run_ingest(ctx: RunContext, source_input: str, config: dict) -> dict:
    """Ingest stage body; returns the source.json content (schema-checked).

    Cache inputs: file name + sha256 (local file) or the URL; params carry
    the normalization caps plus ffmpeg/yt-dlp versions, so a tool or config
    change invalidates the stage.
    """
    params = ingest_params(config)
    is_url = "://" in source_input
    ffmpeg, ffprobe = _binaries()
    stage_params: dict = {**params, "ffmpeg": _ffmpeg_version(ffmpeg)}
    ctx.register_tool("ffmpeg", stage_params["ffmpeg"])

    if is_url:
        stage_params["yt_dlp"] = _ytdlp_version()
        ctx.register_tool("yt-dlp", stage_params["yt_dlp"])
        inputs = {"url": source_input}
    else:
        path = Path(source_input).expanduser()
        if not path.is_file():
            raise InputNotFoundError(f"input file not found: {path}")
        try:
            digest = file_sha256(path)
        except OSError as exc:
            raise InputNotFoundError(f"cannot read input file {path}: {exc.strerror or exc}") from None
        inputs = {"file": path.name, "sha256": digest}

    with ctx.stage("ingest", inputs=inputs, params=stage_params) as st:
        if st.cached:
            return json.loads((ctx.dir / SOURCE_JSON).read_text(encoding="utf-8"))

        if is_url:
            src = ctx.dir / ORIGINAL_NAME
            _run_ytdlp(source_input, src)
            label = "downloaded video"
        else:
            src = path
            label = "input file"

        size = src.stat().st_size
        info = _media_info(_probe(ffprobe, src))
        _check_limits(info, size, params, label)

        normalized = ctx.dir / NORMALIZED_NAME
        _normalize(ffmpeg, src, normalized, info, params)
        if info.has_audio:
            _extract_audio(ffmpeg, normalized, ctx.dir / AUDIO_NAME)

        ninfo = _media_info(_probe(ffprobe, normalized))
        data: dict = {
            "schema_version": "1.0",
            "path": NORMALIZED_NAME,
            "sha256": file_sha256(normalized),
            "duration_s": round(ninfo.duration_s, 3),
            "fps": round(ninfo.fps, 3),
            "width": ninfo.width,
            "height": ninfo.height,
            "streams": ninfo.streams,
        }
        if is_url:
            data["url"] = source_input
        issues = validate_document(data, "source")
        if issues:
            raise IngestError("source.json failed schema validation: " + "; ".join(issues))
        (ctx.dir / SOURCE_JSON).write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        st.outputs = {"video": NORMALIZED_NAME, "audio": AUDIO_NAME if info.has_audio else None}
        return data

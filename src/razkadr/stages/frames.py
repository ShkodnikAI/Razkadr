"""Frames stage (naryad 006): normalized.mp4 -> frames/*.jpg + frames.json.

Candidate timestamps come from two sources: the middle of every detected
scene (PySceneDetect ContentDetector, HSV frame difference) and a uniform
grid with step ``[frames] step_s``. Every candidate is decoded once and
scored with three deterministic metrics:

- ``sharpness``: variance of the Laplacian of the grayscale frame,
  normalized to 0..1 as ``var / (var + SHARPNESS_SCALE)`` (higher is sharper);
- ``phash``: 64-bit perceptual hash of the frame (imagehash), 16 hex chars;
- ``exposure``: quality score 0..1 built from mean brightness and the
  fraction of blown-out (>= 250) and crushed (<= 5) pixels:
  ``1 - 2*over - 2*under - 1.5*max(0, |mean - 0.5| - 0.2)``, clamped.

Rejection marks instead of deletes (naryad 006 scope): a candidate whose
sharpness is below ``[frames] sharpness_threshold`` is rejected with the
reason ``blurry``; a candidate whose pHash hamming distance to an already
accepted frame of the same scene or the previous scene is at most
``[frames] dup_threshold`` is rejected with ``duplicate of <t>s``. All
.jpg files stay on disk; frames.json records every candidate with its
metrics, so rejected frames are never lost silently.

The stage is deterministic: the same normalized.mp4 and config produce a
byte-identical frames.json (timestamps are rounded to 3 decimals before
any comparison, candidates are processed in a fixed order).
"""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path

from razkadr.log import get_logger
from razkadr.runctx import RunContext
from razkadr.validation import validate_document

logger = get_logger("frames")

FRAMES_DIRNAME = "frames"
FRAMES_JSON = "frames.json"

SHARPNESS_SCALE = 6.0  # var / (var + SCALE): measured fixture range 1.7-4.6 -> 0.22-0.43
OVER_PIXEL = 250       # gray >= OVER_PIXEL counts as blown out
UNDER_PIXEL = 5        # gray <= UNDER_PIXEL counts as crushed
BRIGHTNESS_MID = 0.5   # target mean brightness
BRIGHTNESS_TOLERANCE = 0.2  # mean brightness may deviate this much without penalty

FRAMES_DEFAULTS: dict = {
    # scene_threshold 12: measured on the fixture (naryad 006) cuts score
    # 21.45-34.05, inside-scene noise <= 0.60 -> >=1.79x margin to cuts;
    # PySceneDetect's 27.0 default misses 3 of 4 fixture cuts.
    "scene_threshold": 12.0,
    "min_scene_len_s": 0.5,       # ignore cuts closer than this (seconds)
    "step_s": 2.5,                # uniform sampling step, seconds
    "dup_threshold": 6,           # pHash hamming distance <= this -> duplicate
    "sharpness_threshold": 0.28,  # normalized sharpness below this -> blurry
    "jpeg_quality": 90,
}


class FramesError(Exception):
    """Frames stage failure with a clear, user-facing message (CLI exit 2)."""


def frames_params(config: dict) -> dict:
    """Effective [frames] section with defaults; goes into the stage cache key."""
    section = {**FRAMES_DEFAULTS, **(config.get("frames") or {})}
    return {
        "scene_threshold": float(section["scene_threshold"]),
        "min_scene_len_s": float(section["min_scene_len_s"]),
        "step_s": float(section["step_s"]),
        "dup_threshold": int(section["dup_threshold"]),
        "sharpness_threshold": float(section["sharpness_threshold"]),
        "jpeg_quality": int(section["jpeg_quality"]),
    }


def _scenedetect_version() -> str:
    try:
        return package_version("scenedetect")
    except PackageNotFoundError:
        raise FramesError("scenedetect is not installed; install dependencies: pip install -e .") from None


def detect_scenes(video_path: Path, fps: float, params: dict) -> list[tuple[float, float]]:
    """Scene intervals in seconds; a video without detected cuts is one scene.

    Deterministic for the same file and parameters: ContentDetector with a
    fixed threshold and min_scene_len derived from the video fps.
    """
    import cv2  # local import: only the frames stage needs OpenCV
    from scenedetect import ContentDetector, SceneManager, open_video

    min_scene_len = max(1, round(fps * params["min_scene_len_s"]))
    video = open_video(str(video_path))
    manager = SceneManager()
    manager.add_detector(ContentDetector(threshold=params["scene_threshold"], min_scene_len=min_scene_len))
    manager.detect_scenes(video, show_progress=False)
    scenes = [(start.seconds, end.seconds) for start, end in manager.get_scene_list()]
    if not scenes:
        # no cuts detected: the whole video is a single scene
        cap = cv2.VideoCapture(str(video_path))
        try:
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        finally:
            cap.release()
        duration = frames / fps if frames > 0 else 0.0
        scenes = [(0.0, duration)]
    return scenes


def _read_frame_at(video_path: Path, fps: float, t: float):
    """Decode the frame closest to time t (frame-accurate index seek)."""
    import cv2

    index = round(t * fps)
    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def _sharpness(gray) -> float:
    """Normalized variance of the Laplacian: 0 = flat, -> 1 = very sharp."""
    import cv2

    var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return var / (var + SHARPNESS_SCALE)


def _phash(frame) -> str:
    """64-bit perceptual hash of a BGR frame as lowercase hex."""
    import cv2
    import imagehash
    from PIL import Image

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return str(imagehash.phash(Image.fromarray(rgb), hash_size=8))


def _exposure(gray) -> float:
    """Exposure quality 0..1: mean brightness distance from mid + over/under shares."""
    total = gray.size
    mean = float(gray.mean()) / 255.0
    over = float((gray >= OVER_PIXEL).sum()) / total
    under = float((gray <= UNDER_PIXEL).sum()) / total
    penalty = 2.0 * over + 2.0 * under
    deviation = max(0.0, abs(mean - BRIGHTNESS_MID) - BRIGHTNESS_TOLERANCE)
    penalty += 1.5 * deviation
    return max(0.0, min(1.0, 1.0 - penalty))


def _hamming(a: str, b: str) -> int:
    return (int(a, 16) ^ int(b, 16)).bit_count()


def _candidates(scenes: list[tuple[float, float]], duration: float, params: dict) -> list[tuple[float, int, int]]:
    """(t, scene_id, kind) with kind 0=scene mid, 1=uniform grid; fixed order."""
    result: list[tuple[float, int, int]] = []
    for scene_id, (start, end) in enumerate(scenes):
        result.append((round((start + end) / 2.0, 3), scene_id, 0))
    step = params["step_s"]
    if step > 0:
        k = 0
        while k * step < duration - 1e-6:
            t = round(k * step, 3)
            scene_id = max(i for i, (s, _e) in enumerate(scenes) if s <= t + 1e-9)
            result.append((t, scene_id, 1))
            k += 1
    result.sort(key=lambda c: (c[0], c[2]))
    return result


def run_frames(ctx: RunContext, config: dict) -> dict:
    """Frames stage body; returns the frames.json content (schema-checked).

    Cache inputs: sha256 of normalized.mp4 from source.json; params carry
    the [frames] thresholds plus the scenedetect version, so a tool or
    config change invalidates the stage.
    """
    import cv2

    params = frames_params(config)
    params["scenedetect"] = _scenedetect_version()
    ctx.register_tool("scenedetect", params["scenedetect"])

    source_path = ctx.dir / "source.json"
    if not source_path.is_file():
        raise FramesError("frames stage needs ingest output: source.json is missing in the run directory")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    video_path = ctx.dir / source["path"]
    if not video_path.is_file():
        raise FramesError(f"normalized video is missing: {video_path.name} (re-run ingest)")

    fps = float(source["fps"])
    duration = float(source["duration_s"])

    with ctx.stage("frames", inputs={"normalized_sha256": source["sha256"]}, params=params) as st:
        if st.cached:
            return json.loads((ctx.dir / FRAMES_JSON).read_text(encoding="utf-8"))

        scenes = detect_scenes(video_path, fps, params)
        candidates = _candidates(scenes, duration, params)
        if not candidates:
            raise FramesError("no frame candidates produced (empty video?)")

        frames_dir = ctx.dir / FRAMES_DIRNAME
        frames_dir.mkdir(exist_ok=True)

        entries: list[dict] = []
        accepted: dict[int, list[tuple[float, str]]] = {}  # scene_id -> [(t, phash)]
        for i, (t, scene_id, _kind) in enumerate(candidates):
            frame = _read_frame_at(video_path, fps, t)
            if frame is None:
                logger.warning("frame decode failed, candidate skipped", extra={"fields": {"t": t}})
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = round(_sharpness(gray), 6)
            phash = _phash(frame)
            exposure = round(_exposure(gray), 6)
            file_rel = f"{FRAMES_DIRNAME}/{i:06d}.jpg"
            ok_quality = cv2.imwrite(str(ctx.dir / file_rel), frame, [cv2.IMWRITE_JPEG_QUALITY, params["jpeg_quality"]])
            if not ok_quality:
                raise FramesError(f"cannot write frame file {file_rel}")

            rejected = False
            reason = None
            if sharpness < params["sharpness_threshold"]:
                rejected, reason = True, "blurry"
            else:
                for other_scene in (scene_id, scene_id - 1):
                    for other_t, other_hash in accepted.get(other_scene, []):
                        if _hamming(phash, other_hash) <= params["dup_threshold"]:
                            rejected, reason = True, f"duplicate of {other_t:.3f}s"
                            break
                    if rejected:
                        break
            if not rejected:
                accepted.setdefault(scene_id, []).append((t, phash))

            entry: dict = {
                "t": t,
                "file": file_rel,
                "scene_id": scene_id,
                "sharpness": sharpness,
                "phash": phash,
                "exposure": exposure,
            }
            if rejected:
                entry["rejected"] = True
                entry["reject_reason"] = reason
            entries.append(entry)

        data = {"schema_version": "1.1", "frames": entries}
        issues = validate_document(data, "frames")
        if issues:
            raise FramesError("frames.json failed schema validation: " + "; ".join(issues))
        (ctx.dir / FRAMES_JSON).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        n_rejected = sum(1 for e in entries if e.get("rejected"))
        logger.info(
            "frames done",
            extra={"fields": {"candidates": len(entries), "accepted": len(entries) - n_rejected,
                              "scenes": len(scenes)}},
        )
        st.outputs = {"json": FRAMES_JSON, "dir": FRAMES_DIRNAME}
        return data

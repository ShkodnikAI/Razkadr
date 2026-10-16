"""Static HTML gallery of a run: label sheet and selection view (naryad 041).

`razkadr gallery runs/<id>` reads the run directory artifacts and writes a
single self-contained `gallery.html` into the run directory:

- frames from ``frames.json`` (naryad 006) as a clickable sheet with
  timecodes (copy on click), quality marks and rejected frames dimmed;
- a selection view from ``storyboard.json`` (naryad 014, wave 2) when the
  file exists: episodes grouped by role with alternatives and scores,
  selection frames highlighted;
- ground-truth labels (``data/eval/labels/<id>.json``, naryad 003) when
  passed with ``--labels``: frames matching a labeled moment within
  ``--tolerance-s`` get a role badge.

The page opens from disk via ``file://`` with no server, no network and no
external assets: CSS and JS are inlined, thumbnails are the run's own
``frames/*.jpg`` referenced by relative paths (the run directory stays
movable, docs/artifacts.md). Full data is embedded as JSON in a
``<script type="application/json">`` block, so tests can check frame counts
and timecodes against frames.json without a browser.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from razkadr.gallery_page import render_html
from razkadr.validation import validate_document

GALLERY_VERSION = "gallery.v1"
DEFAULT_TOLERANCE_S = 2.0  # the contract's matching tolerance (eval-check default)
LABEL_ROLES = ("hero", "ingredients", "steps")
FRAMES_JSON = "frames.json"
SOURCE_JSON = "source.json"
STORYBOARD_JSON = "storyboard.json"


class GalleryError(Exception):
    """The gallery cannot be built (missing run artifacts, bad inputs)."""


def _read_json(path: Path, what: str) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GalleryError(f"{what} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise GalleryError(f"{what} is not valid JSON: {path}: {exc}") from exc


def load_frames(run_dir: Path) -> list[dict]:
    """Read frames.json (naryad 006); every frame needs t and file."""
    doc = _read_json(run_dir / FRAMES_JSON, "frames.json")
    frames = doc.get("frames") if isinstance(doc, dict) else None
    if not isinstance(frames, list) or not frames:
        raise GalleryError(f"frames.json has no frames array: {run_dir / FRAMES_JSON}")
    for i, frame in enumerate(frames):
        if not isinstance(frame, dict) or "t" not in frame or "file" not in frame:
            raise GalleryError(
                f"frames.json frame #{i} lacks 't'/'file': {run_dir / FRAMES_JSON}"
            )
    return frames


def in_sheet_flags(frames: list[dict], every_s: float) -> list[bool]:
    """Label-sheet inclusion flags: with every_s > 0 at most one frame every
    N seconds (greedy: a frame enters when its t is >= last taken t + N).
    every_s <= 0 keeps every frame."""
    flags: list[bool] = []
    last: float | None = None
    for frame in frames:
        t = float(frame["t"])
        if every_s and every_s > 0 and last is not None and t < last + every_s:
            flags.append(False)
            continue
        flags.append(True)
        last = t
    return flags


def match_labels(
    frames: list[dict], labels: dict, tolerance_s: float
) -> list[list[dict]]:
    """For each frame, ground-truth moments within tolerance: a list of
    {role, t, note} per frame index."""
    matched: list[list[dict]] = [[] for _ in frames]
    for role in LABEL_ROLES:
        for moment in labels.get(role) or []:
            if not isinstance(moment, dict) or "t" not in moment:
                continue
            mt = float(moment["t"])
            note = moment.get("note") or ""
            for i, frame in enumerate(frames):
                if abs(float(frame["t"]) - mt) <= tolerance_s:
                    matched[i].append({"role": role, "t": mt, "note": note})
    return matched


def _load_storyboard(run_dir: Path, storyboard: str | None, warnings: list[str]) -> dict | None:
    """Explicit --storyboard must exist and be valid; the default
    run_dir/storyboard.json is optional (pre-selection runs have none) and
    only warns when invalid so the label sheet still works."""
    explicit = storyboard is not None
    path = Path(storyboard) if explicit else run_dir / STORYBOARD_JSON
    if not path.is_file():
        if explicit:
            raise GalleryError(f"storyboard not found: {path}")
        return None
    doc = _read_json(path, "storyboard")
    issues = validate_document(doc, "storyboard")
    if issues:
        text = "; ".join(str(i) for i in issues)
        if explicit:
            raise GalleryError(f"storyboard is invalid: {path}: {text}")
        warnings.append(f"storyboard {path.name} ignored (invalid: {text})")
        return None
    return doc


def _episode_summary(ep: dict) -> dict:
    """Selection-view record for one episode: role, times, frame, alternatives."""
    frame = ep.get("frame") or {}
    alts = [
        {
            "t": alt.get("t"),
            "path": alt.get("path"),
            "score": alt.get("score"),
        }
        for alt in ep.get("alternatives") or []
        if isinstance(alt, dict)
    ]
    return {
        "id": ep.get("id"),
        "role": ep.get("role"),
        "source_id": ep.get("source_id"),
        "t_start": ep.get("t_start"),
        "t_end": ep.get("t_end"),
        "frame": {
            "t": frame.get("t"),
            "path": frame.get("path"),
            "score": frame.get("score"),
            "score_parts": frame.get("score_parts") or {},
        },
        "alternatives": alts,
    }


def build_gallery_data(
    run_dir: str | Path,
    storyboard: str | None = None,
    labels: str | None = None,
    every_s: float = 0.0,
    tolerance_s: float = DEFAULT_TOLERANCE_S,
) -> dict:
    """Collect everything the page needs; pure function, no writes."""
    run = Path(run_dir)
    if not run.is_dir():
        raise GalleryError(f"run directory not found: {run}")
    frames = load_frames(run)

    warnings: list[str] = []
    flags = in_sheet_flags(frames, every_s)
    shown = sum(flags)

    source = None
    source_path = run / SOURCE_JSON
    if source_path.is_file():
        doc = _read_json(source_path, "source.json")
        source = {
            key: doc.get(key)
            for key in ("path", "url", "duration_s", "fps", "width", "height")
        }

    storyboard_doc = _load_storyboard(run, storyboard, warnings)
    selection = None
    if storyboard_doc is not None:
        episodes = [_episode_summary(ep) for ep in storyboard_doc.get("episodes") or []]
        roles: list[str] = []
        for ep in episodes:
            if ep["role"] not in roles:
                roles.append(ep["role"])
        selection = {
            "storyboard_path": str(storyboard or STORYBOARD_JSON),
            "episodes": episodes,
            "roles": roles,
        }
        # label-sheet highlight: a frame chosen by the selection stage is
        # marked with its episode role (matched by the frame file path)
        role_by_file: dict[str, str] = {}
        for ep in episodes:
            frame = ep.get("frame") or {}
            path = frame.get("path")
            if path:
                role_by_file[str(path)] = str(ep.get("role"))
        for frame in frames:
            role = role_by_file.get(str(frame.get("file")))
            if role:
                frame["selected_role"] = role

    labels_doc = None
    if labels is not None:
        path = Path(labels)
        doc = _read_json(path, "labels")
        issues = validate_document(doc, "eval_labels")
        if issues:
            raise GalleryError(
                f"labels are invalid: {path}: " + "; ".join(str(i) for i in issues)
            )
        matched = match_labels(frames, doc, tolerance_s)
        marked = sum(1 for hits in matched if hits)
        labels_doc = {
            "path": str(path),
            "video_id": doc.get("video_id"),
            "tags": doc.get("tags") or [],
            "tolerance_s": tolerance_s,
            "frames_marked": marked,
        }
        for i, hits in enumerate(matched):
            if hits:
                frames[i]["label_hits"] = hits

    data = {
        "schema": GALLERY_VERSION,
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_dir": str(run),
        "every_s": every_s,
        "frames": [dict(frame, in_sheet=flag) for frame, flag in zip(frames, flags, strict=True)],
        "frames_total": len(frames),
        "frames_shown": shown,
        "source": source,
        "selection": selection,
        "labels": labels_doc,
        "warnings": warnings,
    }
    return data


def run_gallery(
    run_dir: str | Path,
    storyboard: str | None = None,
    labels: str | None = None,
    every_s: float = 0.0,
    tolerance_s: float = DEFAULT_TOLERANCE_S,
    out: str | Path | None = None,
) -> Path:
    """Build gallery.html in the run directory (or at `out`); returns path."""
    run = Path(run_dir)
    data = build_gallery_data(
        run, storyboard=storyboard, labels=labels, every_s=every_s, tolerance_s=tolerance_s
    )
    target = Path(out) if out is not None else run / "gallery.html"
    try:
        target.write_text(render_html(data), encoding="utf-8")
    except OSError as exc:
        raise GalleryError(f"cannot write {target}: {exc}") from exc
    return target

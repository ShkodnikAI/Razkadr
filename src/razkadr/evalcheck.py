"""Integrity check of the reference set (``razkadr eval-check``, naryad 003).

The set lives in one directory (``data/eval``):

    data/eval/
      videos/<id>.mp4        the video files themselves (never committed)
      labels/<id>.json       ground-truth markup, schema eval_labels.v1.json
      SOURCES.md             provenance table: id, ссылка, дата, условия, файл, теги
      README.md              owner instructions

``run_eval_check`` reports every violation as one human-readable line and
returns the counts needed for the contract summary
("N видео, N разметок, нарушений нет"). Violations are deterministic:
inputs are iterated in sorted order.

Duration comes from ffprobe; without a media binary the check cannot verify
timecodes, so a set containing videos raises EvalCheckError (exit 2), the
same convention as media tests elsewhere in the repo.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

from razkadr.validation import validate_document

VIDEO_EXT = ".mp4"
LABELS_DIRNAME = "labels"
VIDEOS_DIRNAME = "videos"
SOURCES_FILENAME = "SOURCES.md"

ROLES = ("hero", "ingredients", "steps")
MIN_ROLE_COUNTS = {"hero": 1, "ingredients": 1, "steps": 5}

DURATION_EPSILON_S = 0.001  # float timestamps from ffmpeg routinely overshoot


class EvalCheckError(Exception):
    """The check itself cannot run (missing directory, no ffprobe)."""


@dataclass
class EvalCheckResult:
    """Counts and violations; the CLI turns this into output and exit code."""

    videos: int = 0
    labels: int = 0
    violations: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations


def _probe_duration_s(path: Path) -> float:
    """Video duration in seconds via ffprobe (contract: 'через ffprobe')."""
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise EvalCheckError(
            f"ffprobe failed on {path.name}: {(proc.stderr or '').strip()[:200]}"
        )
    try:
        return float(json.loads(proc.stdout)["format"]["duration"])
    except (KeyError, ValueError) as exc:
        raise EvalCheckError(f"ffprobe returned no duration for {path.name}: {exc}") from exc


def _sources_ids(sources_path: Path) -> set[str] | None:
    """Parse the SOURCES.md table: ids of rows; None when the file is missing.

    The template (data/eval/SOURCES.md) fixes the format: a markdown table
    ``| id | ссылка | дата скачивания | условия использования | локальный файл | теги |``.
    Parser is deliberately forgiving: any table row whose first cell is a
    non-empty string counts as an entry (separator rows are skipped).
    """
    if not sources_path.is_file():
        return None
    ids: set[str] = set()
    for line in sources_path.read_text(encoding="utf-8").splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not cells or all(set(c) <= {":", "-"} and "-" in c for c in cells):
            continue  # header separator |---|---|
        if cells[0] and cells[0].lower() != "id":
            ids.add(cells[0])
    return ids


def _check_label_document(
    label_path: Path,
    label_id: str,
    issues: list[str],
) -> list[dict] | None:
    """Schema + file-level checks; returns moments per role or None if unusable."""
    try:
        data = json.loads(label_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        issues.append(f"labels/{label_path.name}: invalid JSON ({exc})")
        return None
    schema_issues = validate_document(data, kind="eval_labels")
    for issue in schema_issues:
        issues.append(f"labels/{label_path.name}.{issue}" if issue.path else f"labels/{label_path.name}: {issue.reason}")
    if schema_issues:
        return None
    if data["video_id"] != label_id:
        issues.append(
            f"labels/{label_path.name}: video_id ({data['video_id']}) != file name stem ({label_id})"
        )
    return {role: data.get(role, []) for role in ROLES}


def _check_role_counts(moments: dict[str, list[dict]], issues: list[str], label_path: Path) -> None:
    for role in ROLES:
        minimum = MIN_ROLE_COUNTS[role]
        if len(moments[role]) < minimum:
            issues.append(
                f"labels/{label_path.name}: {len(moments[role])} {role} (minimum {minimum})"
            )


def _check_duplicates(
    moments: dict[str, list[dict]], tolerance_s: float, issues: list[str], label_path: Path
) -> None:
    """Two marks of the same role closer than the tolerance cannot be
    distinguished by matching (допуск ±tolerance с) and count as duplicates."""
    for role in ROLES:
        times = sorted(m["t"] for m in moments[role])
        for prev, cur in pairwise(times):
            if cur - prev < tolerance_s:
                issues.append(
                    f"labels/{label_path.name}: duplicate {role} times {prev:g} and {cur:g} "
                    f"(< {tolerance_s:g} s tolerance)"
                )


def _check_timecodes(
    moments: dict[str, list[dict]], duration_s: float, issues: list[str], label_path: Path
) -> None:
    for role in ROLES:
        for i, moment in enumerate(moments[role]):
            t = moment["t"]
            if t > duration_s + DURATION_EPSILON_S:
                issues.append(
                    f"labels/{label_path.name}.{role}[{i}].t: {t:g} s exceeds video duration "
                    f"({duration_s:.3f} s)"
                )


def run_eval_check(eval_dir: Path, tolerance_s: float = 2.0) -> EvalCheckResult:
    """Check the reference set; contract of naryad 003.

    Checks: every video has a labels file and a SOURCES entry; every labels
    file is valid against eval_labels.v1.json, matches its video, has at
    least 1 hero / 1 ingredients / 5 steps, no duplicate times within the
    tolerance, all timecodes within the video duration (ffprobe).
    """
    if tolerance_s <= 0:
        raise EvalCheckError(f"tolerance must be positive, got {tolerance_s:g}")
    eval_dir = Path(eval_dir)
    if not eval_dir.is_dir():
        raise EvalCheckError(f"not a directory: {eval_dir}")

    result = EvalCheckResult()
    videos_dir = eval_dir / VIDEOS_DIRNAME
    labels_dir = eval_dir / LABELS_DIRNAME
    sources_ids = _sources_ids(eval_dir / SOURCES_FILENAME)
    if sources_ids is None:
        result.violations.append(f"{SOURCES_FILENAME}: missing (template: data/eval/SOURCES.md)")
        sources_ids = set()

    video_ids: set[str] = set()
    # An absent videos/ or labels/ directory is an empty set, not a
    # violation: git does not track empty directories, so a fresh clone of
    # this repo has data/eval with SOURCES.md and README.md only. Real
    # gaps are still caught by the pairing checks below.
    if videos_dir.is_dir():
        for path in sorted(videos_dir.iterdir()):
            if not path.is_file():
                continue
            if path.suffix != VIDEO_EXT:
                result.violations.append(
                    f"videos/{path.name}: unexpected file, expected *{VIDEO_EXT}"
                )
                continue
            video_ids.add(path.stem)

    label_ids: set[str] = set()
    if labels_dir.is_dir():
        for path in sorted(labels_dir.iterdir()):
            if not path.is_file() or path.suffix != ".json":
                continue
            label_ids.add(path.stem)

    durations: dict[str, float] = {}

    def duration_of(video_id: str, video_path: Path) -> float | None:
        if video_id not in durations:
            if shutil.which("ffprobe") is None:
                raise EvalCheckError(
                    "ffprobe not found in PATH; cannot verify timecodes of the reference videos"
                )
            durations[video_id] = _probe_duration_s(video_path)
        return durations[video_id]

    for video_id in sorted(video_ids):
        result.videos += 1
        if video_id not in label_ids:
            result.violations.append(f"videos/{video_id}{VIDEO_EXT}: no labels/{video_id}.json")
        if video_id not in sources_ids:
            result.violations.append(
                f"videos/{video_id}{VIDEO_EXT}: no {SOURCES_FILENAME} entry with id '{video_id}'"
            )

    for label_id in sorted(label_ids):
        label_path = labels_dir / f"{label_id}.json"
        if label_id not in video_ids:
            result.violations.append(
                f"labels/{label_path.name}: no videos/{label_id}{VIDEO_EXT}"
            )
        result.labels += 1
        moments = _check_label_document(label_path, label_id, result.violations)
        if moments is None:
            continue  # schema errors already reported; semantic checks need the document
        _check_role_counts(moments, result.violations, label_path)
        _check_duplicates(moments, tolerance_s, result.violations, label_path)
        if label_id in video_ids:
            duration_s = duration_of(label_id, videos_dir / f"{label_id}{VIDEO_EXT}")
            _check_timecodes(moments, duration_s, result.violations, label_path)

    return result


def summarize(result: EvalCheckResult) -> str:
    """Contract summary line: 'N видео, N разметок, нарушений нет'."""
    if result.ok:
        return f"{result.videos} видео, {plural_markup(result.labels)}, нарушений нет"
    return (
        f"{result.videos} видео, {plural_markup(result.labels)}, "
        f"нарушений: {len(result.violations)}"
    )


def plural_markup(n: int) -> str:
    """Russian plural for 'разметка': 1 разметка, 2 разметки, 5 разметок."""
    if n % 100 in (11, 12, 13, 14) or n % 10 in (0, 5, 6, 7, 8, 9):
        word = "разметок"
    elif n % 10 == 1:
        word = "разметка"
    else:
        word = "разметки"
    return f"{n} {word}"

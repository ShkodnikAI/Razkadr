"""Validation of Storyboard and stage artifacts: JSON Schema + semantic rules.

Error format (one issue per line): ``<json-path>: <reason>``,
for example ``episodes[0].frame.path: required``. JSON paths follow the
jsonschema ``json_path`` syntax without the leading ``$``.

Artifact kinds and their schemas (docs/artifacts.md):

======================  =========================
kind                    schema
======================  =========================
storyboard (default)    storyboard.v1.json
source                  source.v1.json
frames                  frames.v1.json
transcript              transcript.v1.json
chunks                  chunks.v1.json
ocr                     ocr.v1.json
index                   index.v1.json
run                     run.v1.json
eval_labels             eval_labels.v1.json
======================  =========================

``eval_labels`` is not a stage artifact: it is the ground-truth markup of a
reference video (``data/eval/labels/<video_id>.json``, naryad 003, see
``data/eval/README.md``); integrity of the whole set is checked by
``razkadr eval-check``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

SCHEMA_DIRNAME = "schemas"
SUPPORTED_MAJOR = 1

# Episodes/chunks whose t_end exceeds the bound by less than this are accepted
# (float timestamps from ffmpeg/ASR routinely overshoot by sub-millisecond).
DURATION_EPSILON_S = 0.001

# kind -> schema file name; storyboard is the default validate target (naryad 002).
KINDS: dict[str, str] = {
    "storyboard": "storyboard.v1.json",
    "source": "source.v1.json",
    "frames": "frames.v1.json",
    "transcript": "transcript.v1.json",
    "chunks": "chunks.v1.json",
    "ocr": "ocr.v1.json",
    "index": "index.v1.json",
    "run": "run.v1.json",
    "eval_labels": "eval_labels.v1.json",
}


@dataclass(frozen=True)
class Issue:
    """One validation problem: JSON path (no leading $) and a reason."""

    path: str
    reason: str

    def __str__(self) -> str:
        return f"{self.path}: {self.reason}"


def _schema_candidates() -> list[Path]:
    candidates: list[Path] = []
    env_dir = os.environ.get("RAZKADR_SCHEMA_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    cwd = Path.cwd()
    candidates += [base / SCHEMA_DIRNAME for base in (cwd, *cwd.parents)]
    return candidates


def load_schema(kind: str = "storyboard") -> dict:
    """Load the schema for kind.

    Resolution order: RAZKADR_SCHEMA_DIR env var, a ``schemas/`` directory found
    walking up from the CWD (git checkout), the copy packaged with the library.
    A test keeps packaged copies identical to the canonical root files.
    """
    filename = KINDS[kind]
    for base in _schema_candidates():
        candidate = base / filename
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))
    packaged = resources.files("razkadr.schemas").joinpath(filename)
    return json.loads(packaged.read_text(encoding="utf-8"))


def _child_path(parent: str, key: str | int) -> str:
    if isinstance(key, int):
        return f"{parent}[{key}]"
    return f"{parent}.{key}" if parent else str(key)


def _format_schema_error(err) -> Issue:
    """Format a jsonschema error as ``<json-path>: <reason>``."""
    path = err.json_path
    path = "" if path == "$" else path.removeprefix("$.").removeprefix("$")
    if err.validator == "required":
        missing = err.message.split("'")[1]
        child = _child_path(path, missing)
        return Issue(child, "required")
    if err.validator == "additionalProperties":
        prop = err.message.split("'")[1]
        return Issue(_child_path(path, prop), "unknown field")
    return Issue(path, err.message)


def _check_version(data, issues: list[Issue]) -> None:
    """Every v1 schema accepts only major version 1."""
    version = data.get("schema_version") if isinstance(data, dict) else None
    if isinstance(version, str):
        major = version.split(".", 1)[0]
        if not major.isdigit() or int(major) != SUPPORTED_MAJOR:
            issues.append(
                Issue(
                    "schema_version",
                    f"unsupported major version {major!r}, this schema validates v{SUPPORTED_MAJOR}.x",
                )
            )


def _check_storyboard(data, issues: list[Issue]) -> None:
    """Storyboard semantic rules (naryad 002); safe on structurally broken input."""
    sources = data.get("sources")
    source_ids: set[str] = set()
    if isinstance(sources, list):
        for i, src in enumerate(sources):
            if not isinstance(src, dict):
                continue
            sid = src.get("id")
            if isinstance(sid, str):
                if sid in source_ids:
                    issues.append(Issue(_child_path("sources", i), f"duplicate id {sid!r}"))
                source_ids.add(sid)

    episodes = data.get("episodes")
    if not isinstance(episodes, list):
        return
    episode_ids: set[str] = set()
    for i, ep in enumerate(episodes):
        if not isinstance(ep, dict):
            continue
        path = _child_path("episodes", i)
        eid = ep.get("id")
        if isinstance(eid, str):
            if eid in episode_ids:
                issues.append(Issue(path, f"duplicate id {eid!r}"))
            episode_ids.add(eid)

        source_id = ep.get("source_id")
        if isinstance(source_id, str) and source_ids and source_id not in source_ids:
            issues.append(Issue(_child_path(path, "source_id"), f"unknown source {source_id!r}"))

        t_start, t_end = ep.get("t_start"), ep.get("t_end")
        if (
            isinstance(t_start, (int, float))
            and isinstance(t_end, (int, float))
            and t_end < t_start
        ):
            issues.append(Issue(_child_path(path, "t_end"), f"t_end ({t_end}) < t_start ({t_start})"))

        duration = None
        if isinstance(source_id, str):
            for src in sources if isinstance(sources, list) else []:
                if isinstance(src, dict) and src.get("id") == source_id:
                    duration = src.get("duration_s")
                    break
        if (
            isinstance(duration, (int, float))
            and isinstance(t_end, (int, float))
            and t_end > duration + DURATION_EPSILON_S
        ):
            issues.append(Issue(_child_path(path, "t_end"), f"exceeds source duration_s ({duration})"))


def _check_timed_items(data, issues: list[Issue], container: str) -> None:
    """Shared rule for transcript segments and chunks: t_end >= t_start, unique ids."""
    items = data.get(container)
    if not isinstance(items, list):
        return
    ids: set[str] = set()
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        path = _child_path(container, i)
        item_id = item.get("id")
        if isinstance(item_id, str):
            if item_id in ids:
                issues.append(Issue(path, f"duplicate id {item_id!r}"))
            ids.add(item_id)
        t_start, t_end = item.get("t_start"), item.get("t_end")
        if (
            isinstance(t_start, (int, float))
            and isinstance(t_end, (int, float))
            and t_end < t_start
        ):
            issues.append(Issue(_child_path(path, "t_end"), f"t_end ({t_end}) < t_start ({t_start})"))


_SEMANTIC_CHECKS = {
    "storyboard": _check_storyboard,
    "transcript": lambda data, issues: _check_timed_items(data, issues, "segments"),
    "chunks": lambda data, issues: _check_timed_items(data, issues, "chunks"),
}


def validate_document(data: object, kind: str = "storyboard") -> list[Issue]:
    """Validate parsed document of kind; return issues (empty list if valid)."""
    if kind not in KINDS:
        raise KeyError(f"unknown kind {kind!r}, expected one of {sorted(KINDS)}")
    validator = Draft202012Validator(load_schema(kind), format_checker=FormatChecker())
    issues = [_format_schema_error(err) for err in validator.iter_errors(data)]
    _check_version(data, issues)
    check = _SEMANTIC_CHECKS.get(kind)
    if check:
        check(data, issues)
    return issues


def validate_storyboard(data: object) -> list[Issue]:
    """Back-compat wrapper: validate a Storyboard document."""
    return validate_document(data, "storyboard")

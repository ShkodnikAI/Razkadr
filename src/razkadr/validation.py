"""Storyboard validation: JSON Schema checks + semantic rules.

Error format (one issue per line): ``<json-path>: <reason>``,
for example ``episodes[0].frame.path: required``. JSON paths follow the
jsonschema ``json_path`` syntax without the leading ``$``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from jsonschema import Draft202012Validator

SCHEMA_FILENAME = "storyboard.v1.json"
SUPPORTED_MAJOR = 1

# Episodes whose t_end exceeds duration_s by less than this are accepted
# (float timestamps from ffmpeg/ASR routinely overshoot by sub-millisecond).
DURATION_EPSILON_S = 0.001


@dataclass(frozen=True)
class Issue:
    """One validation problem: JSON path (no leading $) and a reason."""

    path: str
    reason: str

    def __str__(self) -> str:
        return f"{self.path}: {self.reason}"


def load_schema() -> dict:
    """Load the Storyboard v1 schema.

    Resolution order: RAZKADR_SCHEMA_DIR env var, a ``schemas/`` directory found
    walking up from the CWD (git checkout), the copy packaged with the library.
    A test keeps the packaged copy identical to the canonical root file.
    """
    import os

    candidates: list[Path] = []
    env_dir = os.environ.get("RAZKADR_SCHEMA_DIR")
    if env_dir:
        candidates.append(Path(env_dir) / SCHEMA_FILENAME)
    cwd = Path.cwd()
    for base in (cwd, *cwd.parents):
        candidates.append(base / "schemas" / SCHEMA_FILENAME)
    for candidate in candidates:
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))
    packaged = resources.files("razkadr.schemas").joinpath(SCHEMA_FILENAME)
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


def _check_semantics(data, issues: list[Issue]) -> None:
    """Semantic rules beyond the schema; safe on structurally broken input."""
    version = data.get("schema_version")
    if isinstance(version, str):
        major = version.split(".", 1)[0]
        if not major.isdigit() or int(major) != SUPPORTED_MAJOR:
            issues.append(
                Issue(
                    "schema_version",
                    f"unsupported major version {major!r}, this schema validates v{SUPPORTED_MAJOR}.x",
                )
            )

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


def validate_storyboard(data: object) -> list[Issue]:
    """Validate parsed Storyboard data; return issues (empty list if valid)."""
    schema = load_schema()
    validator = Draft202012Validator(schema)
    issues = [_format_schema_error(err) for err in validator.iter_errors(data)]
    _check_semantics(data, issues)
    return issues

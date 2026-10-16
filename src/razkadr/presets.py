"""Preset loading and validation (naryad 013).

A preset is the domain scenario in data: which frame roles the document
needs, how each role is searched (per-language queries) and scored (signal
weights), episode quotas (min/max), quality constraints and the document
order of roles. Presets are pure YAML: adding a role or a new preset must
not require code changes, so nothing here knows anything about specific
domains -- role names are opaque strings.

`load_preset` reads a YAML file, validates it against schemas/preset.v1.json
(`razkadr validate --kind preset` works on the same document), applies the
semantic checks the schema cannot express (order covers all roles, max >=
min, weights sum to 1, no blank queries) and fills defaults, returning a
normalized dict. Every failure is a PresetError with the file path and all
problems at once.
"""

from __future__ import annotations

import json
from pathlib import Path

from razkadr.validation import validate_document

PRESET_DEFAULTS = {"candidates": 20, "alternatives": 3}
WEIGHTS_SUM_EPSILON = 1e-6


class PresetError(Exception):
    """The preset cannot be loaded (unreadable, invalid, semantically wrong)."""


def _read_yaml(path: Path) -> object:
    try:
        import yaml

        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PresetError(f"preset not found: {path}") from exc
    except UnicodeDecodeError as exc:
        raise PresetError(f"preset must be UTF-8: {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise PresetError(f"preset is not valid YAML: {path}: {exc}") from exc


def _semantic_issues(preset: dict) -> list[str]:
    """Checks beyond JSON Schema: order coverage, max >= min, weights sum,
    blank queries."""
    issues: list[str] = []
    roles = preset["roles"]
    order = preset["order"]

    missing = [role for role in roles if role not in order]
    unknown = [role for role in order if role not in roles]
    if missing:
        issues.append(f"order is missing roles: {', '.join(missing)}")
    if unknown:
        issues.append(f"order lists unknown roles: {', '.join(unknown)}")

    for role, spec in roles.items():
        lo, hi = spec["min"], spec["max"]
        if hi is not None and hi < lo:
            issues.append(f"role '{role}': max ({hi}) < min ({lo})")
        weight_sum = sum(spec["weights"].values())
        if abs(weight_sum - 1.0) > WEIGHTS_SUM_EPSILON:
            issues.append(
                f"role '{role}': weights must sum to 1, got {weight_sum:.6g}"
            )
        for lang, queries in spec["queries"].items():
            for i, query in enumerate(queries):
                if not query.strip():
                    issues.append(
                        f"role '{role}': blank query #{i} in '{lang}'"
                    )
    return issues


def load_preset(path: str | Path) -> dict:
    """Load, validate and normalize a preset YAML; returns a new dict with
    defaults filled in (candidates, alternatives). Raises PresetError with
    the path and all problems at once."""
    file = Path(path)
    data = _read_yaml(file)
    if not isinstance(data, dict):
        raise PresetError(f"preset must be a YAML mapping: {file}")
    issues = [f"{file}: {issue}" for issue in validate_document(data, "preset")]
    if not issues:
        issues = [f"{file}: {text}" for text in _semantic_issues(data)]
    if issues:
        raise PresetError("; ".join(issues))
    normalized = json.loads(json.dumps(data))  # deep copy
    for spec in normalized["roles"].values():
        for key, default in PRESET_DEFAULTS.items():
            spec.setdefault(key, default)
    return normalized

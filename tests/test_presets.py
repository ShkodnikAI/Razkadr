"""Preset tests (naryad 013): schema, loader semantics, YAML-only roles.

The preset is pure data, so the tests are pure file-level: load the shipped
recipe.yaml, feed broken YAML variants (missing role, negative weight, empty
queries, bad order, weights that do not sum to 1) and prove that adding a
new role requires nothing but editing YAML (a synthetic preset with an
invented role loads cleanly and nothing in the code knows recipe roles).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from razkadr.presets import PRESET_DEFAULTS, PresetError, load_preset
from razkadr.validation import validate_document

REPO = Path(__file__).resolve().parent.parent
RECIPE = REPO / "presets" / "recipe.yaml"

EXPECTED_ROLES = ["hero", "ingredients", "prep", "steps", "serving"]


def _write(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "preset.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def _base_preset() -> dict:
    """A minimal valid preset to mutate in the negative tests."""
    return {
        "schema_version": "1.0",
        "name": "base",
        "roles": {
            "hero": {
                "min": 1,
                "max": 1,
                "queries": {"ru": ["готовое блюдо"], "en": ["finished dish"]},
                "weights": {"embedding": 1.0},
            }
        },
        "order": ["hero"],
    }


# --- the shipped recipe preset --------------------------------------------------


def test_recipe_preset_is_valid_and_normalized():
    preset = load_preset(RECIPE)
    assert validate_document(preset, "preset") == []
    assert list(preset["roles"]) == EXPECTED_ROLES or set(preset["roles"]) == set(EXPECTED_ROLES)
    assert preset["order"] == EXPECTED_ROLES
    assert preset["name"] == "recipe"
    # quotas from the contract: hero(1), ingredients(1), prep(0..N), steps(3..12), serving(0..1)
    quotas = {role: (preset["roles"][role]["min"], preset["roles"][role]["max"]) for role in EXPECTED_ROLES}
    assert quotas == {
        "hero": (1, 1),
        "ingredients": (1, 1),
        "prep": (0, None),
        "steps": (3, 12),
        "serving": (0, 1),
    }
    # every role: RU and EN queries, weights normalized, defaults filled
    for role, spec in preset["roles"].items():
        assert set(spec["queries"]) == {"ru", "en"}, role
        assert all(isinstance(q, str) and q for lang in ("ru", "en") for q in spec["queries"][lang])
        assert abs(sum(spec["weights"].values()) - 1.0) < 1e-9, role
        assert spec["candidates"] == preset["roles"][role].get("candidates") or spec["candidates"] >= 1
        for key in PRESET_DEFAULTS:
            assert spec[key] >= (1 if key == "candidates" else 0), role


def test_recipe_preset_passes_validate_cli(tmp_path: Path):
    from razkadr.cli import EXIT_OK, main

    assert main(["validate", "--kind", "preset", str(RECIPE)]) == EXIT_OK


def test_packaged_and_root_preset_schemas_are_identical():
    root = REPO / "schemas" / "preset.v1.json"
    packaged = REPO / "src" / "razkadr" / "schemas" / "preset.v1.json"
    assert root.read_text(encoding="utf-8") == packaged.read_text(encoding="utf-8")


# --- negative: schema-level rejects ----------------------------------------------


def test_missing_role_rejected(tmp_path: Path):
    # a preset without any role is invalid
    data = _base_preset()
    del data["roles"]["hero"]
    with pytest.raises(PresetError, match="should be non-empty"):
        load_preset(_write(tmp_path, data))


def test_negative_weight_rejected(tmp_path: Path):
    data = _base_preset()
    data["roles"]["hero"]["weights"]["embedding"] = -0.5
    with pytest.raises(PresetError, match="minimum"):
        load_preset(_write(tmp_path, data))


def test_empty_queries_rejected(tmp_path: Path):
    data = _base_preset()
    data["roles"]["hero"]["queries"]["ru"] = []
    with pytest.raises(PresetError, match="should be non-empty"):
        load_preset(_write(tmp_path, data))


def test_empty_query_string_rejected(tmp_path: Path):
    data = _base_preset()
    data["roles"]["hero"]["queries"]["ru"] = ["   "]
    with pytest.raises(PresetError, match="blank query #0 in 'ru'"):
        load_preset(_write(tmp_path, data))


def test_unknown_language_rejected(tmp_path: Path):
    data = _base_preset()
    data["roles"]["hero"]["queries"]["de"] = ["fertiges gericht"]
    with pytest.raises(PresetError, match="unknown field"):
        load_preset(_write(tmp_path, data))


def test_missing_max_rejected(tmp_path: Path):
    data = _base_preset()
    del data["roles"]["hero"]["max"]
    with pytest.raises(PresetError, match="required"):
        load_preset(_write(tmp_path, data))


def test_broken_yaml_rejected(tmp_path: Path):
    path = tmp_path / "broken.yaml"
    path.write_text("roles: [unclosed", encoding="utf-8")
    with pytest.raises(PresetError, match="not valid YAML"):
        load_preset(path)


def test_missing_file_rejected(tmp_path: Path):
    with pytest.raises(PresetError, match="not found"):
        load_preset(tmp_path / "nope.yaml")


# --- negative: loader semantics the schema cannot express ------------------------


def test_weights_must_sum_to_one(tmp_path: Path):
    data = _base_preset()
    data["roles"]["hero"]["weights"] = {"embedding": 0.4, "sharpness": 0.2}
    with pytest.raises(PresetError, match="sum to 1, got 0.6"):
        load_preset(_write(tmp_path, data))


def test_max_below_min_rejected(tmp_path: Path):
    data = _base_preset()
    data["roles"]["hero"]["max"] = 0
    with pytest.raises(PresetError, match=r"hero.*max \(0\) < min \(1\)"):
        load_preset(_write(tmp_path, data))


def test_order_must_cover_all_roles(tmp_path: Path):
    data = _base_preset()
    data["roles"]["serving"] = {
        "min": 0,
        "max": 1,
        "queries": {"ru": ["подача"]},
        "weights": {"embedding": 1.0},
    }
    with pytest.raises(PresetError, match="order is missing roles: serving"):
        load_preset(_write(tmp_path, data))


def test_order_must_not_invent_roles(tmp_path: Path):
    data = _base_preset()
    data["order"] = ["hero", "ghost"]
    with pytest.raises(PresetError, match="unknown roles: ghost"):
        load_preset(_write(tmp_path, data))


def test_all_errors_reported_at_once(tmp_path: Path):
    # both problems are loader semantics (schema-clean, checked together)
    data = _base_preset()
    data["roles"]["hero"]["max"] = 0
    data["order"] = ["ghost"]
    with pytest.raises(PresetError) as excinfo:
        load_preset(_write(tmp_path, data))
    message = str(excinfo.value)
    assert "max (0) < min (1)" in message
    assert "unknown roles: ghost" in message


# --- YAML-only extensibility ------------------------------------------------------


def test_new_role_requires_only_yaml(tmp_path: Path):
    """The auditor's criterion: no recipe knowledge in code -- an invented
    role loads, validates and gets defaults without any code change."""
    data = _base_preset()
    data["name"] = "synthetic"
    data["roles"]["dessert"] = {
        "min": 0,
        "max": 2,
        "queries": {"ru": ["десерт"], "en": ["dessert"]},
        "weights": {"embedding": 0.7, "sharpness": 0.3},
    }
    data["order"] = ["hero", "dessert"]
    preset = load_preset(_write(tmp_path, data))
    assert validate_document(preset, "preset") == []
    assert preset["roles"]["dessert"]["candidates"] == PRESET_DEFAULTS["candidates"]
    assert preset["roles"]["dessert"]["alternatives"] == PRESET_DEFAULTS["alternatives"]


def test_recipe_roles_not_hardcoded_in_code():
    """grep-level guarantee: the loader must not know recipe role names."""
    source = (REPO / "src" / "razkadr" / "presets.py").read_text(encoding="utf-8")
    for role in EXPECTED_ROLES:
        assert role not in source, f"recipe role '{role}' leaked into loader code"


# --- loader normalization ---------------------------------------------------------


def test_defaults_filled_and_input_not_mutated(tmp_path: Path):
    data = _base_preset()
    raw = json.dumps(data)
    preset = load_preset(_write(tmp_path, data))
    assert preset["roles"]["hero"]["candidates"] == 20
    assert preset["roles"]["hero"]["alternatives"] == 3
    assert json.dumps(data) == raw  # the caller's dict untouched

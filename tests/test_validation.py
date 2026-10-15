"""Unit tests for storyboard validation (schema + semantic checks)."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from razkadr.validation import validate_storyboard

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples"
SCHEMA_ROOT = REPO / "schemas" / "storyboard.v1.json"
SCHEMA_PACKAGED = REPO / "src" / "razkadr" / "schemas" / "storyboard.v1.json"


def load(name: str) -> dict:
    return json.loads((EXAMPLES / name).read_text(encoding="utf-8"))


def paths_of(issues):
    return {i.path for i in issues}


def test_valid_example_has_no_issues():
    assert validate_storyboard(load("storyboard.valid.json")) == []


def test_valid_edited_example_has_no_issues():
    assert validate_storyboard(load("storyboard.valid-edited.json")) == []


def test_invalid_example_reports_missing_frame_path():
    issues = validate_storyboard(load("storyboard.invalid.json"))
    matching = [i for i in issues if i.path == "episodes[0].frame.path"]
    assert matching
    assert matching[0].reason == "required"


def test_schema_copies_in_sync():
    """Packaged copy must equal the canonical schema at the repo root."""
    assert SCHEMA_ROOT.read_text(encoding="utf-8") == SCHEMA_PACKAGED.read_text(encoding="utf-8")


def test_empty_episodes_is_valid():
    data = load("storyboard.valid.json")
    data["episodes"] = []
    assert validate_storyboard(data) == []


def test_unknown_text_mode():
    data = load("storyboard.valid.json")
    data["episodes"][0]["text"]["mode"] = "improved"
    issues = validate_storyboard(data)
    assert [i for i in issues if i.path == "episodes[0].text.mode"]


def test_t_end_before_t_start():
    data = load("storyboard.valid.json")
    data["episodes"][0]["t_end"] = 100.0
    data["episodes"][0]["t_start"] = 200.0
    issues = validate_storyboard(data)
    assert [i for i in issues if i.path == "episodes[0].t_end" and "< t_start" in i.reason]


def test_unknown_source_id():
    data = load("storyboard.valid.json")
    data["episodes"][0]["source_id"] = "v9"
    issues = validate_storyboard(data)
    assert [i for i in issues if i.path == "episodes[0].source_id" and "unknown source" in i.reason]


def test_duplicate_episode_id():
    data = load("storyboard.valid.json")
    data["episodes"][1]["id"] = data["episodes"][0]["id"]
    issues = validate_storyboard(data)
    assert [i for i in issues if i.path == "episodes[1]" and "duplicate id" in i.reason]


def test_duplicate_source_id():
    data = load("storyboard.valid.json")
    data["sources"].append(deepcopy(data["sources"][0]))
    issues = validate_storyboard(data)
    assert [i for i in issues if i.path == "sources[1]" and "duplicate id" in i.reason]


def test_episode_beyond_source_duration():
    data = load("storyboard.valid.json")
    data["episodes"][0]["t_end"] = 900.0
    issues = validate_storyboard(data)
    assert [i for i in issues if i.path == "episodes[0].t_end" and "duration_s" in i.reason]


def test_duration_within_epsilon_is_accepted():
    data = load("storyboard.valid.json")
    data["episodes"][0]["t_end"] = 800.0005
    assert validate_storyboard(data) == []


def test_unknown_top_level_field():
    data = load("storyboard.valid.json")
    data["extra"] = True
    issues = validate_storyboard(data)
    assert [i for i in issues if i.path == "extra" and "unknown field" in i.reason]


def test_missing_project_title():
    data = load("storyboard.valid.json")
    del data["project"]["title"]
    issues = validate_storyboard(data)
    assert [i for i in issues if i.path == "project.title" and i.reason == "required"]


def test_bad_schema_version_format():
    data = load("storyboard.valid.json")
    data["schema_version"] = "one-dot-zero"
    issues = validate_storyboard(data)
    assert [i for i in issues if i.path == "schema_version"]


def test_future_major_version_rejected():
    data = load("storyboard.valid.json")
    data["schema_version"] = "2.0"
    issues = validate_storyboard(data)
    assert [i for i in issues if i.path == "schema_version" and "major" in i.reason]


def test_negative_t_start_rejected():
    data = load("storyboard.valid.json")
    data["episodes"][0]["t_start"] = -1.0
    issues = validate_storyboard(data)
    assert [i for i in issues if i.path == "episodes[0].t_start"]


def test_score_out_of_range_rejected():
    data = load("storyboard.valid.json")
    data["episodes"][0]["frame"]["score"] = 1.5
    issues = validate_storyboard(data)
    assert [i for i in issues if i.path == "episodes[0].frame.score"]


def test_unknown_field_inside_episode():
    data = load("storyboard.valid.json")
    data["episodes"][0]["mood"] = "happy"
    issues = validate_storyboard(data)
    assert [i for i in issues if i.path == "episodes[0].mood" and "unknown field" in i.reason]


def test_bad_lang_tag_rejected():
    data = load("storyboard.valid.json")
    data["project"]["output_lang"] = "not a lang!"
    issues = validate_storyboard(data)
    assert [i for i in issues if i.path == "project.output_lang"]


@pytest.mark.parametrize(
    "episode_patch",
    [
        {"frame": {"t": 1.0}},  # frame without path
        {"t_end": None},  # wrong type
    ],
    ids=["frame-without-path", "t-end-wrong-type"],
)
def test_structural_violations_report_json_paths(episode_patch: dict):
    data = load("storyboard.valid.json")
    data["episodes"][0].update(episode_patch)
    issues = validate_storyboard(data)
    assert issues, "expected at least one issue"
    assert all(i.path and i.reason for i in issues)

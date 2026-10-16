"""Artifact schema tests (naryad 035): valid example + invalid case per kind."""

import json
from pathlib import Path

import pytest

from razkadr.validation import KINDS, load_schema, validate_document

REPO = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO / "examples" / "artifacts"
SCHEMA_ROOT = REPO / "schemas"
SCHEMA_PACKAGED = REPO / "src" / "razkadr" / "schemas"

EXAMPLE_FILE = {
    "source": "source.json",
    "frames": "frames.json",
    "transcript": "transcript.json",
    "chunks": "chunks.json",
    "ocr": "ocr.json",
    "index": "index.json",
    "run": "run.json",
    "eval_labels": "eval_labels.json",  # naryad 003: not a stage artifact, markup of a reference video
}


def load_example(kind: str) -> dict:
    return json.loads((ARTIFACTS / EXAMPLE_FILE[kind]).read_text(encoding="utf-8"))


def test_every_kind_has_schema_example_and_file():
    for kind, filename in KINDS.items():
        if kind == "storyboard":
            continue  # covered by test_validation.py (naryad 002)
        assert (SCHEMA_ROOT / filename).is_file(), f"missing schema for {kind}"
        assert (SCHEMA_PACKAGED / filename).is_file(), f"missing packaged schema for {kind}"
        assert (ARTIFACTS / EXAMPLE_FILE[kind]).is_file(), f"missing example for {kind}"


@pytest.mark.parametrize("kind", list(EXAMPLE_FILE))
def test_valid_example_passes(kind):
    assert validate_document(load_example(kind), kind) == []


def test_all_schema_copies_in_sync():
    for filename in KINDS.values():
        assert (SCHEMA_ROOT / filename).read_text(encoding="utf-8") == (
            SCHEMA_PACKAGED / filename
        ).read_text(encoding="utf-8"), filename


def test_unknown_kind_raises():
    with pytest.raises(KeyError):
        validate_document({}, "unknown")


@pytest.mark.parametrize("kind", list(EXAMPLE_FILE))
def test_future_major_version_rejected_for_every_kind(kind):
    data = load_example(kind)
    data["schema_version"] = "2.0"
    issues = validate_document(data, kind)
    assert [i for i in issues if i.path == "schema_version" and "major" in i.reason]


# --- invalid cases, one per kind ---------------------------------------------

def test_invalid_source_bad_sha256():
    data = load_example("source")
    data["sha256"] = "not-a-hash"
    issues = validate_document(data, "source")
    assert [i for i in issues if i.path == "sha256"]


def test_invalid_source_zero_fps():
    data = load_example("source")
    data["fps"] = 0
    issues = validate_document(data, "source")
    assert [i for i in issues if i.path == "fps"]


def test_invalid_frames_bad_phash():
    data = load_example("frames")
    data["frames"][0]["phash"] = "ZZZZ"
    issues = validate_document(data, "frames")
    assert [i for i in issues if i.path == "frames[0].phash"]


def test_invalid_frames_sharpness_out_of_range():
    data = load_example("frames")
    data["frames"][0]["sharpness"] = 1.2
    issues = validate_document(data, "frames")
    assert [i for i in issues if i.path == "frames[0].sharpness"]


def test_invalid_transcript_t_end_before_t_start():
    data = load_example("transcript")
    data["segments"][0]["t_start"] = 10.0
    data["segments"][0]["t_end"] = 5.0
    issues = validate_document(data, "transcript")
    assert [i for i in issues if i.path == "segments[0].t_end" and "< t_start" in i.reason]


def test_invalid_transcript_missing_lang():
    data = load_example("transcript")
    del data["segments"][1]["lang"]
    issues = validate_document(data, "transcript")
    assert [i for i in issues if i.path == "segments[1].lang" and i.reason == "required"]


def test_invalid_chunks_duplicate_id():
    data = load_example("chunks")
    data["chunks"][2]["id"] = data["chunks"][1]["id"]
    issues = validate_document(data, "chunks")
    assert [i for i in issues if i.path == "chunks[2]" and "duplicate id" in i.reason]


def test_invalid_chunks_t_end_before_t_start():
    data = load_example("chunks")
    data["chunks"][0]["t_start"] = 50.0
    data["chunks"][0]["t_end"] = 42.5
    issues = validate_document(data, "chunks")
    assert [i for i in issues if i.path == "chunks[0].t_end"]


def test_invalid_ocr_missing_lang():
    data = load_example("ocr")
    del data["results"][0]["lang"]
    issues = validate_document(data, "ocr")
    assert [i for i in issues if i.path == "results[0].lang" and i.reason == "required"]


def test_invalid_ocr_bad_lang_tag():
    data = load_example("ocr")
    data["results"][0]["lang"] = "ru!!"
    issues = validate_document(data, "ocr")
    assert [i for i in issues if i.path == "results[0].lang"]


def test_invalid_index_nonpositive_dim():
    data = load_example("index")
    data["dim"] = 0
    issues = validate_document(data, "index")
    assert [i for i in issues if i.path == "dim"]


def test_invalid_index_unknown_field():
    data = load_example("index")
    data["vectors"] = [0.1, 0.2]
    issues = validate_document(data, "index")
    assert [i for i in issues if i.path == "vectors" and "unknown field" in i.reason]


def test_invalid_run_provider_call_missing_cache_hit():
    data = load_example("run")
    del data["provider_calls"][0]["cache_hit"]
    issues = validate_document(data, "run")
    assert [i for i in issues if i.path == "provider_calls[0].cache_hit" and i.reason == "required"]


def test_invalid_run_provider_call_missing_cost():
    data = load_example("run")
    del data["provider_calls"][1]["cost"]
    issues = validate_document(data, "run")
    assert [i for i in issues if i.path == "provider_calls[1].cost" and i.reason == "required"]


def test_invalid_run_empty_units():
    data = load_example("run")
    data["provider_calls"][0]["units"] = []
    issues = validate_document(data, "run")
    assert [i for i in issues if i.path == "provider_calls[0].units"]


def test_invalid_run_bad_stage_status():
    data = load_example("run")
    data["stages"][0]["status"] = "completed"
    issues = validate_document(data, "run")
    assert [i for i in issues if i.path == "stages[0].status"]


def test_invalid_run_error_must_be_string():
    """error is an optional string; null is not a valid substitute."""
    data = load_example("run")
    data["stages"][3]["error"] = None
    issues = validate_document(data, "run")
    assert [i for i in issues if i.path == "stages[3].error"]


def test_invalid_run_bad_created_utc_format():
    data = load_example("run")
    data["created_utc"] = "15.10.2026 07:30"
    issues = validate_document(data, "run")
    assert [i for i in issues if i.path == "created_utc"]


def test_invalid_run_bad_currency():
    data = load_example("run")
    data["provider_calls"][0]["cost"]["currency"] = "usd"
    issues = validate_document(data, "run")
    assert [i for i in issues if i.path == "provider_calls[0].cost.currency"]


def test_cli_validate_kind_routes_to_schema(tmp_path, capsys):
    from razkadr.cli import main

    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(load_example("frames")), encoding="utf-8")
    assert main(["validate", "--kind", "frames", str(path)]) == 0
    assert capsys.readouterr().out.strip() == "ok"

    broken = load_example("frames")
    broken["frames"][0]["phash"] = "short"
    path.write_text(json.dumps(broken), encoding="utf-8")
    assert main(["validate", "--kind", "frames", str(path)]) == 1
    assert "frames[0].phash" in capsys.readouterr().err


def test_cli_validate_default_kind_is_storyboard(tmp_path, capsys):
    from razkadr.cli import main

    artifact = load_example("frames")  # would fail storyboard schema
    path = tmp_path / "as-storyboard.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    assert main(["validate", str(path)]) == 1
    err = capsys.readouterr().err
    assert "schema_version" in err or "required" in err


def test_load_schema_kinds_match_files():
    for kind, filename in KINDS.items():
        schema = load_schema(kind)
        assert schema["$id"].endswith(filename)


# --- summary (naryad 011): additive optional block of run.v1.1 ----------------

def test_invalid_run_summary_unknown_field():
    data = load_example("run")
    data["summary"]["oops"] = 1
    issues = validate_document(data, "run")
    assert [i for i in issues if i.path == "summary.oops" and i.reason == "unknown field"]


def test_invalid_run_summary_negative_amount():
    data = load_example("run")
    data["summary"]["provider_cost"]["amount"] = -0.5
    issues = validate_document(data, "run")
    assert [i for i in issues if i.path == "summary.provider_cost.amount"]


def test_invalid_run_summary_missing_block():
    data = load_example("run")
    del data["summary"]["stage_times_s"]
    issues = validate_document(data, "run")
    assert [i for i in issues if i.path == "summary.stage_times_s" and i.reason == "required"]


def test_invalid_run_summary_negative_stage_time():
    data = load_example("run")
    data["summary"]["stage_times_s"]["ocr"] = -1.0
    issues = validate_document(data, "run")
    assert [i for i in issues if i.path == "summary.stage_times_s.ocr"]


def test_run_example_11_consistent_numbers():
    """The example documents a successful run: summary matches the log it aggregates."""
    data = load_example("run")
    cost = data["summary"]["provider_cost"]
    assert cost["calls"] == len(data["provider_calls"])
    assert cost["amount"] == round(
        sum(c["cost"]["amount"] for c in data["provider_calls"]), 6
    )
    assert cost["cache_hits"] == sum(1 for c in data["provider_calls"] if c["cache_hit"])
    times = data["summary"]["stage_times_s"]
    for stage in data["stages"]:
        assert times[stage["name"]] == stage["duration_s"]

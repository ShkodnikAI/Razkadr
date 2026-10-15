"""CLI contract tests for `razkadr validate` (naryad 002 contract)."""

import json
from pathlib import Path

from razkadr.cli import main

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples"


def test_valid_file_prints_ok_and_exits_0(capsys):
    code = main(["validate", str(EXAMPLES / "storyboard.valid.json")])
    out = capsys.readouterr()
    assert code == 0
    assert out.out.strip() == "ok"
    assert out.err == ""


def test_valid_edited_file_prints_ok(capsys):
    code = main(["validate", str(EXAMPLES / "storyboard.valid-edited.json")])
    assert code == 0
    assert capsys.readouterr().out.strip() == "ok"


def test_invalid_file_exits_1_with_json_path_on_stderr(capsys):
    code = main(["validate", str(EXAMPLES / "storyboard.invalid.json")])
    out = capsys.readouterr()
    assert code == 1
    assert "episodes[0].frame.path: required" in out.err
    assert out.out == ""


def test_missing_file_exits_2(capsys):
    code = main(["validate", "/nonexistent/storyboard.json"])
    out = capsys.readouterr()
    assert code == 2
    assert "cannot read" in out.err


def test_broken_json_exits_2(tmp_path, capsys):
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    code = main(["validate", str(path)])
    out = capsys.readouterr()
    assert code == 2
    assert "invalid JSON" in out.err


def test_semantic_error_reaches_stderr(tmp_path, capsys):
    data = json.loads((EXAMPLES / "storyboard.valid.json").read_text(encoding="utf-8"))
    data["episodes"][0]["source_id"] = "v9"
    path = tmp_path / "sb.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    code = main(["validate", str(path)])
    err = capsys.readouterr().err
    assert code == 1
    assert "episodes[0].source_id: unknown source 'v9'" in err

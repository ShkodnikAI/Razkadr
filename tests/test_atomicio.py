"""Atomic write tests (naryad 029): no truncated artifacts, no stray tmp files."""

import json
from pathlib import Path

import pytest

from razkadr.atomicio import build_atomic, write_json_atomic, write_text_atomic


def test_write_json_atomic_roundtrip_and_no_tmp_left(tmp_path: Path):
    target = tmp_path / "artifact.json"
    data = {"b": 1, "a": "текст"}
    write_json_atomic(target, data)
    assert json.loads(target.read_text(encoding="utf-8")) == data
    assert target.read_text(encoding="utf-8").endswith("\n")
    assert list(tmp_path.iterdir()) == [target], "no tmp files left behind"


def test_write_json_atomic_newline_flag(tmp_path: Path):
    target = tmp_path / "artifact.json"
    write_json_atomic(target, {"x": 1}, newline=False)
    assert not target.read_text(encoding="utf-8").endswith("\n")


def test_write_over_replaces_whole_file(tmp_path: Path):
    target = tmp_path / "artifact.json"
    write_json_atomic(target, {"old": "x" * 5000})
    write_json_atomic(target, {"new": 1})
    assert json.loads(target.read_text(encoding="utf-8")) == {"new": 1}


def test_write_text_atomic(tmp_path: Path):
    target = tmp_path / "deck.html"
    write_text_atomic(target, "<html></html>")
    assert target.read_text(encoding="utf-8") == "<html></html>"
    assert list(tmp_path.iterdir()) == [target]


def test_build_atomic_failure_leaves_target_untouched(tmp_path: Path):
    target = tmp_path / "deck.pptx"
    target.write_bytes(b"previous deck bytes")
    boom = RuntimeError("crash mid-save")

    def produce(_tmp: Path) -> None:
        raise boom

    with pytest.raises(RuntimeError) as excinfo:
        build_atomic(target, produce, tmp_suffix=".pptx")
    assert excinfo.value is boom
    assert target.read_bytes() == b"previous deck bytes"
    assert list(tmp_path.iterdir()) == [target], "the failed tmp is cleaned up"


def test_build_atomic_moves_complete_result(tmp_path: Path):
    target = tmp_path / "deck.pdf"

    def produce(tmp: Path) -> None:
        tmp.write_bytes(b"%PDF-1.7 complete")

    build_atomic(target, produce, tmp_suffix=".pdf")
    assert target.read_bytes() == b"%PDF-1.7 complete"
    assert list(tmp_path.iterdir()) == [target]

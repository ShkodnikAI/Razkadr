"""Tests for the reference-set tools (naryad 003): eval_labels schema,
eval-check semantics and the CLI command."""

import json
import shutil
from pathlib import Path

import pytest

from razkadr.cli import (
    EXIT_FILE_ERROR,
    EXIT_INVALID,
    EXIT_OK,
    STUB_COMMANDS,
    main,
)
from razkadr.evalcheck import EvalCheckError, plural_markup, run_eval_check

REPO = Path(__file__).resolve().parents[1]
SOURCES_TEMPLATE = REPO / "data" / "eval" / "SOURCES.md"
README = REPO / "data" / "eval" / "README.md"

SOURCES_TABLE = """| id | ссылка | дата скачивания | условия использования | локальный файл | теги |
|----|--------|-----------------|----------------------|----------------|------|
"""


def valid_label(video_id: str = "a") -> dict:
    """A schema-valid label with the role minimums met."""
    return {
        "schema_version": "1.0",
        "video_id": video_id,
        "tags": ["ru", "voice"],
        "hero": [{"t": 5.0}],
        "ingredients": [{"t": 12.0}],
        "steps": [{"t": 20.0 + i * 5.0} for i in range(5)],
    }


def make_evalset(
    tmp_path: Path,
    videos: tuple[str, ...] = (),
    labels: dict[str, object] | None = None,
    source_rows: tuple[str, ...] = (),
    sources: str | None = None,
    extra_video_files: tuple[str, ...] = (),
) -> Path:
    """Build a data/eval-like directory; video files are dummies (no probe
    happens while the video has no labels). An existing video file is kept:
    ffprobe tests copy a real fixture in before calling this."""
    root = tmp_path / "eval"
    (root / "videos").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    for name in videos:
        path = root / "videos" / f"{name}.mp4"
        if not path.exists():
            path.write_bytes(b"dummy video bytes")
    for name in extra_video_files:
        (root / "videos" / name).write_bytes(b"dummy")
    for name, doc in (labels or {}).items():
        text = doc if isinstance(doc, str) else json.dumps(doc, ensure_ascii=False)
        (root / "labels" / f"{name}.json").write_text(text, encoding="utf-8")
    if sources is None:
        sources = SOURCES_TABLE + "".join(r + "\n" for r in source_rows)
    (root / "SOURCES.md").write_text(sources, encoding="utf-8")
    return root


def source_row(video_id: str) -> str:
    return f"| {video_id} | https://example.com/watch?v={video_id} | 2026-10-16 | CC BY 4.0 | videos/{video_id}.mp4 | ru, voice |"


# ---------------------------------------------------------------- empty set


def test_empty_set_is_ok(tmp_path):
    root = make_evalset(tmp_path)
    result = run_eval_check(root)
    assert result.ok
    assert (result.videos, result.labels) == (0, 0)


def test_missing_root_dir_raises(tmp_path):
    with pytest.raises(EvalCheckError):
        run_eval_check(tmp_path / "nowhere")


def test_absent_videos_and_labels_dirs_is_empty_set(tmp_path):
    """git does not track empty directories: a fresh clone has data/eval with
    SOURCES.md and README.md only, and that is a valid empty set."""
    root = tmp_path / "eval"
    root.mkdir()
    (root / "SOURCES.md").write_text(SOURCES_TABLE, encoding="utf-8")
    result = run_eval_check(root)
    assert result.ok
    assert (result.videos, result.labels) == (0, 0)


def test_missing_videos_dir_still_flags_labels(tmp_path):
    """With labels present the pairing check catches the absent videos dir."""
    root = tmp_path / "eval"
    (root / "labels").mkdir(parents=True)
    (root / "labels" / "a.json").write_text(json.dumps(valid_label()), encoding="utf-8")
    (root / "SOURCES.md").write_text(SOURCES_TABLE + source_row("a") + "\n", encoding="utf-8")
    result = run_eval_check(root)
    assert "labels/a.json: no videos/a.mp4" in result.violations


# ------------------------------------------------------- file-level pairing


def test_video_without_labels_and_sources(tmp_path):
    root = make_evalset(tmp_path, videos=("a",))
    result = run_eval_check(root)
    assert "videos/a.mp4: no labels/a.json" in result.violations
    assert "videos/a.mp4: no SOURCES.md entry with id 'a'" in result.violations


def test_labels_without_video(tmp_path):
    root = make_evalset(tmp_path, labels={"a": valid_label()}, source_rows=(source_row("a"),))
    result = run_eval_check(root)
    assert "labels/a.json: no videos/a.mp4" in result.violations


def test_unexpected_video_extension(tmp_path):
    root = make_evalset(tmp_path, extra_video_files=("b.mov",))
    result = run_eval_check(root)
    assert "videos/b.mov: unexpected file, expected *.mp4" in result.violations


def test_sources_missing(tmp_path):
    root = make_evalset(tmp_path, labels={"a": valid_label()})
    (root / "SOURCES.md").unlink()
    result = run_eval_check(root)
    assert result.violations[0] == "SOURCES.md: missing (template: data/eval/SOURCES.md)"


def test_sources_row_without_video_is_not_a_violation(tmp_path):
    """The owner may pre-register a source before downloading; the contract
    requires each video to have a SOURCES entry, not the reverse."""
    root = make_evalset(tmp_path, source_rows=(source_row("planned"),))
    assert run_eval_check(root).ok


# ------------------------------------------------------------ label content


def test_broken_json_label(tmp_path):
    root = make_evalset(tmp_path, labels={"a": "{not json"})
    result = run_eval_check(root)
    assert any(v.startswith("labels/a.json: invalid JSON") for v in result.violations)


def test_schema_violations_reported_with_labels_path(tmp_path):
    bad = valid_label()
    bad["steps"][0]["t"] = -1.0
    bad["hero"][0]["weight"] = 3  # unknown field
    bad["tags"] = ["klingon"]
    root = make_evalset(tmp_path, labels={"a": bad})
    result = run_eval_check(root)
    assert "labels/a.json.steps[0].t: -1.0 is less than the minimum of 0" in result.violations
    assert "labels/a.json.hero[0].weight: unknown field" in result.violations
    assert any("tags[0]" in v and "klingon" in v for v in result.violations)
    # semantic checks stop for a document that failed the schema
    assert not any("minimum" in v and "steps (minimum" in v for v in result.violations)


def test_video_id_mismatch(tmp_path):
    root = make_evalset(tmp_path, labels={"a": valid_label(video_id="other")})
    result = run_eval_check(root)
    assert any("video_id (other) != file name stem (a)" in v for v in result.violations)


def test_role_minimums(tmp_path):
    label = valid_label()
    label["steps"] = label["steps"][:2]
    del label["ingredients"]
    root = make_evalset(tmp_path, labels={"a": label})
    result = run_eval_check(root)
    assert "labels/a.json: 2 steps (minimum 5)" in result.violations
    assert "labels/a.json: 0 ingredients (minimum 1)" in result.violations


def test_duplicate_times_within_tolerance(tmp_path):
    label = valid_label()
    label["steps"].append({"t": 21.5})  # 1.5 s after steps[0] (20.0)
    root = make_evalset(tmp_path, labels={"a": label})
    result = run_eval_check(root, tolerance_s=2.0)
    assert any("duplicate steps times 20 and 21.5" in v for v in result.violations)
    # same marks with a tighter tolerance are distinct
    tighter = run_eval_check(root, tolerance_s=1.0)
    assert not any("duplicate" in v for v in tighter.violations)


def test_non_positive_tolerance_raises(tmp_path):
    root = make_evalset(tmp_path)
    with pytest.raises(EvalCheckError):
        run_eval_check(root, tolerance_s=0.0)


# ------------------------------------------------- ffprobe-dependent checks


def _fixture_video(tmp_path: Path, synthetic_fixtures, video_id: str) -> Path:
    """A data/eval dir whose videos/<id>.mp4 is the 30 s scenes fixture."""
    root = tmp_path / "eval"
    (root / "videos").mkdir(parents=True)
    (root / "labels").mkdir()
    shutil.copyfile(synthetic_fixtures.path("scenes.mp4"), root / "videos" / f"{video_id}.mp4")
    return root


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe not found in PATH")
def test_valid_set_with_fixture(tmp_path, synthetic_fixtures):
    """Acceptance criterion: the contract's example (adapted to the fixture
    duration) passes on the naryad-034 fixture."""
    root = _fixture_video(tmp_path, synthetic_fixtures, "ru_voice_01")
    label = {
        "schema_version": "1.0",
        "video_id": "ru_voice_01",
        "tags": ["ru", "voice"],
        "hero": [{"t": 3.0}],
        "ingredients": [{"t": 6.0}],
        "steps": [
            {"t": 12.0, "note": "нарезка лука"},
            {"t": 15.0, "note": "обжарка"},
            {"t": 21.0, "note": "добавление бульона"},
            {"t": 24.0, "note": "варка"},
            {"t": 27.0, "note": "подача"},
        ],
    }
    (root / "labels" / "ru_voice_01.json").write_text(
        json.dumps(label, ensure_ascii=False), encoding="utf-8"
    )
    (root / "SOURCES.md").write_text(
        SOURCES_TABLE + source_row("ru_voice_01") + "\n", encoding="utf-8"
    )
    result = run_eval_check(root)
    assert result.ok, result.violations
    assert (result.videos, result.labels) == (1, 1)


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe not found in PATH")
def test_timecode_beyond_duration(tmp_path, synthetic_fixtures):
    root = _fixture_video(tmp_path, synthetic_fixtures, "a")
    label = valid_label()
    label["steps"].append({"t": 999.0})
    (root / "labels" / "a.json").write_text(json.dumps(label), encoding="utf-8")
    (root / "SOURCES.md").write_text(SOURCES_TABLE + source_row("a") + "\n", encoding="utf-8")
    result = run_eval_check(root)
    assert any("steps[5].t: 999 s exceeds video duration" in v for v in result.violations)


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe not found in PATH")
def test_timecode_at_duration_edge_is_ok(tmp_path, synthetic_fixtures):
    """scenes.mp4 is 30 s; a mark exactly at the end is within the video."""
    root = _fixture_video(tmp_path, synthetic_fixtures, "a")
    label = valid_label()
    label["steps"] = [{"t": t} for t in (3.0, 6.0, 9.0, 12.0, 30.0)]
    (root / "labels" / "a.json").write_text(json.dumps(label), encoding="utf-8")
    (root / "SOURCES.md").write_text(SOURCES_TABLE + source_row("a") + "\n", encoding="utf-8")
    assert run_eval_check(root).ok


# --------------------------------------------------------------------- CLI


def test_cli_eval_check_ok(tmp_path, capsys):
    root = make_evalset(tmp_path)
    code = main(["eval-check", str(root)])
    out = capsys.readouterr()
    assert code == EXIT_OK
    assert "0 видео, 0 разметок, нарушений нет" in out.out
    assert out.err == ""


def test_cli_eval_check_violations_exit_1(tmp_path, capsys):
    root = make_evalset(tmp_path, videos=("a",))
    code = main(["eval-check", str(root)])
    captured = capsys.readouterr()
    assert code == EXIT_INVALID
    assert "videos/a.mp4: no labels/a.json" in captured.err
    assert "нарушений: 2" in captured.err


def test_cli_eval_check_missing_dir_exit_2(tmp_path, capsys):
    code = main(["eval-check", str(tmp_path / "nowhere")])
    out = capsys.readouterr()
    assert code == EXIT_FILE_ERROR
    assert "razkadr eval-check:" in out.err


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe not found in PATH")
def test_cli_eval_check_tolerance_flag(tmp_path, capsys, synthetic_fixtures):
    """--tolerance-s reaches the check: marks 1.5 s apart are duplicates at
    the default 2 s tolerance and distinct at 1 s (all inside the 30 s video)."""
    root = _fixture_video(tmp_path, synthetic_fixtures, "a")
    label = valid_label()
    label["hero"] = [{"t": 3.0}]
    label["ingredients"] = [{"t": 6.0}]
    label["steps"] = [{"t": t} for t in (9.0, 12.0, 15.0, 18.0, 21.0)] + [{"t": 10.5}]
    (root / "labels" / "a.json").write_text(json.dumps(label), encoding="utf-8")
    (root / "SOURCES.md").write_text(SOURCES_TABLE + source_row("a") + "\n", encoding="utf-8")
    assert main(["eval-check", str(root), "--tolerance-s", "2.0"]) == EXIT_INVALID
    assert main(["eval-check", str(root), "--tolerance-s", "1.0"]) == EXIT_OK


def test_eval_check_no_longer_a_stub():
    assert "eval-check" not in STUB_COMMANDS
    assert "eval" not in STUB_COMMANDS  # naryad 015 made the harness real


# --------------------------------------------------------- shipped template


def test_template_and_readme_exist():
    assert SOURCES_TEMPLATE.is_file()
    assert README.is_file()


def test_template_header_matches_parser():
    """The shipped SOURCES.md parses to an empty id set (fresh repo is valid)."""
    root = SOURCES_TEMPLATE.parent
    result = run_eval_check(root)
    assert result.ok, result.violations


# ----------------------------------------------------------------- plural


@pytest.mark.parametrize(
    "n,expected",
    [(0, "0 разметок"), (1, "1 разметка"), (2, "2 разметки"), (5, "5 разметок"), (11, "11 разметок"), (21, "21 разметка"), (22, "22 разметки")],
)
def test_plural_markup(n, expected):
    assert plural_markup(n) == expected

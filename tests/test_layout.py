"""Layout engine tests (naryad 021): Storyboard -> slides.json.

Hermetic and deterministic: no network, no media fixtures. Examples from
examples/ and synthetic Storyboards (0, 1, 7, 30 episodes) cover the
naryad acceptance list; boundaries (exactly N items) included per the
auditor checklist.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from razkadr.cli import EXIT_FILE_ERROR, EXIT_INVALID, EXIT_OK
from razkadr.cli import main as cli_main
from razkadr.stages.layout import (
    DEFAULT_LIMITS,
    LayoutError,
    build_slides,
    layout_storyboard,
    timecode_s,
    write_slides,
)
from razkadr.validation import validate_document

EXAMPLES = Path(__file__).parent.parent / "examples"


def example(name: str) -> dict:
    return json.loads((EXAMPLES / name).read_text(encoding="utf-8"))


def episode(
    episode_id: str,
    role: str,
    *,
    text: str | None = None,
    t_start: float = 0.0,
    frame: str = "frames/f.jpg",
) -> dict:
    episode = {
        "id": episode_id,
        "role": role,
        "source_id": "v1",
        "t_start": t_start,
        "t_end": t_start + 5,
        "frame": {"path": frame, "t": t_start},
    }
    if text is not None:
        episode["text"] = {"mode": "condensed", "source_lang": "ru", "original": text, "output": text, "output_lang": "ru"}
    return episode


def project(project_id: str = "p1", title: str = "Проект", output_lang: str = "ru") -> dict:
    return {
        "schema_version": "1.0",
        "project": {"id": project_id, "title": title, "preset": "recipe", "output_lang": output_lang},
        "sources": [{"id": "v1", "path": "source.mp4"}],
        "episodes": [],
    }


def assert_valid(doc: dict) -> dict:
    issues = validate_document(doc, "slides")
    assert not issues, [f"{i.path}: {i.message}" for i in issues]
    return doc


# --- contract example: examples/storyboard.valid.json ----------------------------


def test_example_valid_storyboard_produces_valid_slides() -> None:
    doc = assert_valid(layout_storyboard(example("storyboard.valid.json"), {}))
    layouts = [s["layout"] for s in doc["slides"]]
    assert layouts[0] == "title"
    assert "hero_caption" in layouts and "ingredients_list" in layouts and "steps_grid" in layouts
    title = doc["slides"][0]
    assert title["slots"]["heading"] == "Шакшука на сковороде"
    assert title["source_refs"] == ["project"]
    hero_caption = next(s for s in doc["slides"] if s["layout"] == "hero_caption")
    assert hero_caption["slots"]["timecode"] == "13:00"  # episode t_start=780 s (frame.t is 787)
    assert hero_caption["source_refs"] == ["e1"]


def test_example_edited_storyboard_produces_valid_slides() -> None:
    doc = assert_valid(layout_storyboard(example("storyboard.valid-edited.json"), {}))
    assert doc["slides"][0]["layout"] == "title"


def test_lang_from_project_output_lang_and_flag() -> None:
    storyboard = example("storyboard.valid.json")
    doc = layout_storyboard(storyboard, {})
    assert doc["lang"] == "ru"
    forced = layout_storyboard(storyboard, {}, lang="en")
    ingredients = next(s for s in forced["slides"] if s["layout"] == "ingredients_list")
    assert forced["lang"] == "en"
    assert ingredients["slots"]["heading"] == "Ingredients"


# --- boundaries: exactly N and N+1 items / cells ----------------------------------


def twelve_ingredients_storyboard(count: int) -> dict:
    text = ", ".join(f"ингредиент {i + 1}" for i in range(count))
    storyboard = project()
    storyboard["episodes"] = [episode("ing1", "ingredients", text=text, t_start=10)]
    return storyboard


def test_ingredients_exactly_twelve_single_slide() -> None:
    doc = assert_valid(build_slides(twelve_ingredients_storyboard(12), {}))
    lists = [s for s in doc["slides"] if s["layout"] == "ingredients_list"]
    assert len(lists) == 1
    assert len(lists[0]["slots"]["items"]) == 12


def test_ingredients_thirteen_spill_to_second_slide() -> None:
    doc = assert_valid(build_slides(twelve_ingredients_storyboard(13), {}))
    lists = [s for s in doc["slides"] if s["layout"] == "ingredients_list"]
    assert len(lists) == 2
    assert [len(s["slots"]["items"]) for s in lists] == [12, 1]
    assert lists[1]["slots"]["heading"].endswith("(продолжение)")
    assert lists[1]["source_refs"] == ["ing1"]  # continuation repeats the episode


def test_seven_steps_two_grid_slides() -> None:
    storyboard = project()
    storyboard["episodes"] = [
        episode(f"s{i}", "step", text=f"Шаг номер {i}.", t_start=i * 20.0) for i in range(1, 8)
    ]
    doc = assert_valid(build_slides(storyboard, {}))
    grids = [s for s in doc["slides"] if s["layout"] == "steps_grid"]
    assert len(grids) == 2
    assert [len(g["slots"]["cells"]) for g in grids] == [6, 1]
    assert [g["slots"]["columns"] for g in grids] == [3, 2]  # 6 cells -> 3 columns, 1 -> 2x2
    numbers = [c["number"] for g in grids for c in g["slots"]["cells"]]
    assert numbers == list(range(1, 8))  # global numbering across slides


def test_exactly_six_steps_one_slide() -> None:
    storyboard = project()
    storyboard["episodes"] = [episode(f"s{i}", "step", text=f"шаг {i}") for i in range(1, 7)]
    doc = assert_valid(build_slides(storyboard, {}))
    grids = [s for s in doc["slides"] if s["layout"] == "steps_grid"]
    assert len(grids) == 1 and len(grids[0]["slots"]["cells"]) == 6


# --- text limits, truncation, warnings --------------------------------------------


def test_long_caption_truncated_with_warning() -> None:
    long_text = "Очень длинная подпись про готовое блюдо. " * 12  # > caption_max
    storyboard = project()
    storyboard["episodes"] = [episode("srv1", "serving", text=long_text, t_start=30)]
    doc = assert_valid(build_slides(storyboard, {}))
    serving = next(s for s in doc["slides"] if s["layout"] == "serving")
    caption = serving["slots"]["caption"]
    assert len(caption) <= DEFAULT_LIMITS["caption_max"]
    assert caption.endswith((".", "…"))
    assert serving["warnings"] == [
        {"code": "text_truncated", "episode_id": "srv1", "slot": "caption", "limit": DEFAULT_LIMITS["caption_max"]}
    ]


def test_truncation_prefers_sentence_then_word_boundary() -> None:
    from razkadr.stages.layout import _shorten

    text = "Первое предложение про рецепт. Второе предложение про сковороду и соус."
    cut = _shorten(text, 40)
    assert cut == "Первое предложение про рецепт."  # sentence boundary wins
    words = "одно слово два слова три слова четыре слова пять слов"
    cut2 = _shorten(words, 20)
    assert cut2.endswith("…") and " слов" not in cut2.rstrip("…").rsplit(" ", 1)[-1]


def test_item_truncation_warning_lands_on_owning_slide() -> None:
    long_item = "составная часть " * 3  # 48 chars, no commas -> one item over item_max = 40
    storyboard = project()
    storyboard["episodes"] = [episode("ing1", "ingredients", text=f"короткий, {long_item}")]
    doc = assert_valid(build_slides(storyboard, {}))
    lists = [s for s in doc["slides"] if s["layout"] == "ingredients_list"]
    truncated = [item for item in lists[0]["slots"]["items"] if item.endswith("…")]
    assert truncated, "long item must be shortened"
    assert lists[0]["warnings"][0]["episode_id"] == "ing1"
    assert lists[0]["warnings"][0]["slot"] == "items"


# --- episodes without text ---------------------------------------------------------


def test_storyboard_without_text_is_valid_and_quiet() -> None:
    storyboard = example("storyboard.valid.json")
    for ep in storyboard["episodes"]:
        ep.pop("text", None)
    doc = assert_valid(layout_storyboard(storyboard, {}))
    for slide in doc["slides"]:
        if slide["layout"] in ("hero_caption", "serving"):
            assert slide["slots"]["caption"] == ""
        if slide["layout"] == "ingredients_list":
            assert slide["slots"]["items"] == []
        if slide["layout"] == "steps_grid":
            assert all(c["text"] == "" for c in slide["slots"]["cells"])
        assert slide["warnings"] == []


def test_empty_text_string_also_quiet() -> None:
    storyboard = project()
    storyboard["episodes"] = [episode("h1", "hero", text="   ", t_start=5)]
    doc = assert_valid(build_slides(storyboard, {}))
    hero = next(s for s in doc["slides"] if s["layout"] == "hero_caption")
    assert hero["slots"]["caption"] == "" and hero["warnings"] == []


# --- synthetic sizes: 0, 1, 30 episodes ---------------------------------------------


def test_zero_episodes_title_only() -> None:
    doc = assert_valid(build_slides(project(), {}))
    assert [s["layout"] for s in doc["slides"]] == ["title"]
    assert doc["slides"][0]["slots"]["heading"] == "Проект"


def test_single_hero_episode() -> None:
    storyboard = project()
    storyboard["episodes"] = [episode("h1", "hero", text="Финальное блюдо", t_start=60)]
    doc = assert_valid(build_slides(storyboard, {}))
    assert [s["layout"] for s in doc["slides"]] == ["title", "hero_caption"]
    assert doc["slides"][0]["slots"]["subheading"] == "Финальное блюдо"
    assert doc["slides"][0]["slots"]["frame"]["episode_id"] == "h1"


def test_thirty_episodes_no_crash_and_valid() -> None:
    storyboard = project()
    storyboard["episodes"] = (
        [episode("h1", "hero", text="Герой")]
        + [episode("ing", "ingredients", text=", ".join(f"продукт {i}" for i in range(1, 26)))]
        + [episode(f"s{i}", "step", text=f"Шаг {i} описания процесса готовки.") for i in range(1, 28)]
        + [episode("srv", "serving", text="Подача со зеленью")]
    )
    assert len(storyboard["episodes"]) == 30
    doc = assert_valid(build_slides(storyboard, {}))
    assert len(doc["slides"]) > 5
    grids = [s for s in doc["slides"] if s["layout"] == "steps_grid"]
    assert sum(len(g["slots"]["cells"]) for g in grids) == 27


# --- role -> layout mapping from config, never hardcoded ----------------------------


def test_role_mapping_override_from_config() -> None:
    storyboard = project()
    storyboard["episodes"] = [episode("h1", "hero", text="Блюдо", t_start=15)]
    config = {"layout": {"role_layouts": {"hero": "serving"}}}
    doc = assert_valid(build_slides(storyboard, config))
    layouts = [s["layout"] for s in doc["slides"]]
    assert "hero_caption" not in layouts
    assert "serving" in layouts


def test_unmapped_role_skipped_with_document_warning() -> None:
    storyboard = project()
    storyboard["episodes"] = [episode("x1", "finale", text="Финал", t_start=100)]
    doc = assert_valid(build_slides(storyboard, {}))
    assert [s["layout"] for s in doc["slides"]] == ["title"]
    assert doc["warnings"] == [{"code": "unmapped_role", "detail": "role 'finale' has no layout mapping; 1 episode(s) skipped"}]


def test_invalid_layout_in_config_is_clear_error() -> None:
    with pytest.raises(LayoutError, match="not a layout"):
        build_slides(project(), {"layout": {"role_layouts": {"hero": "collage"}}})


def test_invalid_limit_in_config_is_clear_error() -> None:
    with pytest.raises(LayoutError, match="positive integer"):
        build_slides(project(), {"layout": {"caption_max": 0}})


# --- timecodes -----------------------------------------------------------------------


def test_timecode_formats() -> None:
    assert timecode_s(0) == "0:00"
    assert timecode_s(787) == "13:07"
    assert timecode_s(59.9) == "0:59"
    assert timecode_s(3672) == "1:01:12"
    assert timecode_s(7325) == "2:02:05"


# --- determinism ----------------------------------------------------------------------


def test_two_runs_byte_identical(tmp_path: Path) -> None:
    storyboard = example("storyboard.valid.json")
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    write_slides(assert_valid(layout_storyboard(storyboard, {})), first)
    write_slides(assert_valid(layout_storyboard(storyboard, {})), second)
    assert first.read_bytes() == second.read_bytes()


# --- CLI --------------------------------------------------------------------------------


def test_cli_layout_writes_valid_slides(tmp_path: Path) -> None:
    src = tmp_path / "storyboard.json"
    src.write_text(json.dumps(example("storyboard.valid.json"), ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "out" / "slides.json"
    code = cli_main(["layout", str(src), "--out", str(out)])
    assert code == EXIT_OK
    doc = assert_valid(json.loads(out.read_text(encoding="utf-8")))
    assert doc["slides"][0]["layout"] == "title"


def test_cli_layout_default_output_next_to_storyboard(tmp_path: Path) -> None:
    src = tmp_path / "storyboard.json"
    src.write_text(json.dumps(example("storyboard.valid.json"), ensure_ascii=False), encoding="utf-8")
    code = cli_main(["layout", str(src)])
    assert code == EXIT_OK
    assert (tmp_path / "slides.json").is_file()


def test_cli_layout_invalid_storyboard_exit_1(tmp_path: Path) -> None:
    src = tmp_path / "storyboard.json"
    src.write_text('{"schema_version": "9.0"}', encoding="utf-8")
    assert cli_main(["layout", str(src)]) == EXIT_INVALID


def test_cli_layout_missing_file_exit_2(tmp_path: Path) -> None:
    assert cli_main(["layout", str(tmp_path / "nope.json")]) == EXIT_FILE_ERROR

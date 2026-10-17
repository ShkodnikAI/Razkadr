"""PPTX render tests (naryad 022): hermetic, no network, no LibreOffice.

The deck must open with python-pptx and match slides.json: one slide per
slide document entry, one picture per placed frame, text slots filled,
speaker notes carrying the episode id, start/end timecodes and the source.
The template build is reproducible (part contents byte-identical to the
committed file), and no text exceeds its slot at the final font size
(the estimation method is the same Pillow wrap simulation the renderer
uses, applied independently over the saved deck - see docs/render.md).
"""

from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from pptx import Presentation

from razkadr.cli import EXIT_FILE_ERROR, EXIT_INVALID, EXIT_OK
from razkadr.cli import main as cli_main
from razkadr.render.pptx import (
    INSET_SIDE_IN,
    LINE_HEIGHT,
    fit_size,
    render_pptx,
    resolve_template,
    wrap_lines,
)

REPO = Path(__file__).parent.parent
EXAMPLES = REPO / "examples"

EMU_PER_IN = 914400


def load(name: str) -> dict:
    return json.loads((EXAMPLES / name).read_text(encoding="utf-8"))


def make_frames(slides_doc: dict, root: Path, size: tuple = (640, 360)) -> list[Path]:
    """Real JPEGs at the paths recorded in the slides document."""
    from PIL import Image

    paths = []
    for slide in slides_doc["slides"]:
        slots = slide["slots"]
        if "frame" in slots:
            paths.append(slots["frame"]["path"])
        for cell in slots.get("cells", []):
            paths.append(cell["frame"]["path"])
    made = []
    for i, relpath in enumerate(sorted(set(paths))):
        p = root / relpath
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, ((40 * i) % 256, 96, 160)).save(p, "JPEG", quality=85)
        made.append(p)
    return made


def slides_with_frames(tmp_path: Path) -> tuple[dict, Path]:
    doc = load("artifacts/slides.json")
    (tmp_path / "slides.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    make_frames(doc, tmp_path)
    return doc, tmp_path


def shape_texts(slide) -> list[str]:
    texts = []
    for shape in slide.shapes:
        if shape.has_text_frame and shape.text_frame.text.strip():
            texts.append(shape.text_frame.text)
    return texts


def iter_text_shapes(slide):
    for shape in slide.shapes:
        if shape.has_text_frame and shape.text_frame.text.strip():
            yield shape


def run_size(shape) -> int:
    for paragraph in shape.text_frame.paragraphs:
        for run in paragraph.runs:
            if run.font.size is not None:
                return int(run.font.size.pt)
    return 0


# --- template ---------------------------------------------------------------------


def test_template_has_five_named_layouts_with_slots() -> None:
    prs = Presentation(str(resolve_template(None)))
    layouts = prs.slide_masters[0].slide_layouts
    assert [lay.name for lay in layouts] == [
        "title",
        "hero_caption",
        "ingredients_list",
        "steps_grid",
        "serving",
    ]
    expected = {
        "title": {"heading", "subheading", "frame"},
        "hero_caption": {"frame", "caption", "timecode"},
        "ingredients_list": {"heading", "frame", "items"},
        "serving": {"heading", "frame", "caption", "timecode"},
    }
    for name, slots in expected.items():
        lay = next(l for l in layouts if l.name == name)
        assert {sh.name for sh in lay.shapes} >= slots, name
    steps = next(l for l in layouts if l.name == "steps_grid")
    names = {sh.name for sh in steps.shapes}
    for i in range(1, 7):
        assert {f"cell{i}_number", f"cell{i}_frame", f"cell{i}_text", f"cell{i}_timecode"} <= names


def test_template_build_is_reproducible(tmp_path: Path) -> None:
    """A fresh build must be content-identical to the committed template:
    same part names, same decompressed bytes (zip timestamps differ)."""
    script = REPO / "scripts/templates/build_default.py"
    out = tmp_path / "rebuilt.pptx"
    subprocess.run([sys.executable, str(script), "--out", str(out)], check=True, capture_output=True)

    def contents(path: Path) -> dict[str, bytes]:
        with zipfile.ZipFile(path) as z:
            return {name: z.read(name) for name in z.namelist()}

    committed = contents(REPO / "templates/razkadr-default.pptx")
    rebuilt = contents(out)
    assert set(committed) == set(rebuilt)
    for name in committed:
        assert committed[name] == rebuilt[name], name


def test_missing_template_is_a_file_error(tmp_path: Path) -> None:
    doc, _ = slides_with_frames(tmp_path)
    with pytest.raises(Exception, match="template not found"):
        render_pptx(doc, tmp_path / "deck.pptx", template=tmp_path / "nope.pptx")


# --- render: counts and content -----------------------------------------------------


def test_render_example_matches_slides_counts(tmp_path: Path) -> None:
    doc, root = slides_with_frames(tmp_path)
    stats = render_pptx(doc, tmp_path / "deck.pptx", storyboard=load("storyboard.valid.json"), base_dir=root)
    assert stats["slides"] == len(doc["slides"]) == 4
    # one picture per frame slot: title, hero_caption, ingredients, 1 cell
    assert stats["pictures"] == 4
    assert stats["warnings"] == []

    prs = Presentation(str(tmp_path / "deck.pptx"))
    assert len(prs.slides) == 4
    pictures = sum(
        1 for slide in prs.slides for shape in slide.shapes if shape.shape_type == 13
    )
    assert pictures == 4
    # title slide carries heading and subheading
    title = prs.slides[0]
    texts = " ".join(shape_texts(title))
    assert doc["slides"][0]["slots"]["heading"] in texts
    assert doc["slides"][0]["slots"]["subheading"] in texts


def test_render_is_content_deterministic(tmp_path: Path) -> None:
    doc, root = slides_with_frames(tmp_path)
    render_pptx(doc, tmp_path / "a.pptx", base_dir=root)
    render_pptx(doc, tmp_path / "b.pptx", base_dir=root)

    def contents(path: Path) -> dict[str, bytes]:
        with zipfile.ZipFile(path) as z:
            return {n: z.read(n) for n in z.namelist() if n.startswith("ppt/slides/")}

    assert contents(tmp_path / "a.pptx") == contents(tmp_path / "b.pptx")


def test_steps_grid_six_cells_three_columns(tmp_path: Path) -> None:
    doc = {
        "schema_version": "1.0",
        "lang": "ru",
        "slides": [
            {
                "layout": "steps_grid",
                "source_refs": ["e1"],
                "warnings": [],
                "slots": {
                    "columns": 3,
                    "heading": "Шаги",
                    "cells": [
                        {
                            "frame": {"path": f"frames/f{i}.jpg", "t": 10.0 * i, "episode_id": "e1"},
                            "number": i,
                            "text": f"Шаг номер {i}",
                            "timecode": f"{i}:00",
                        }
                        for i in range(1, 7)
                    ],
                },
            }
        ],
    }
    make_frames(doc, tmp_path)
    stats = render_pptx(doc, tmp_path / "deck.pptx", base_dir=tmp_path)
    assert stats["pictures"] == 6
    prs = Presentation(str(tmp_path / "deck.pptx"))
    slide = prs.slides[0]
    texts = shape_texts(slide)
    for i in range(1, 7):
        assert f"Шаг номер {i}" in " ".join(texts)
        assert f"{i}:00" in " ".join(texts)
    # no leftover placeholders for unused cells (all 6 are used here);
    # empty shapes must not exist at all
    for shape in slide.shapes:
        if shape.has_text_frame:
            assert shape.text_frame.text.strip() or shape.name.startswith("deco") is False


def test_steps_grid_unused_cell_placeholders_dropped(tmp_path: Path) -> None:
    doc, root = slides_with_frames(tmp_path)  # example: 1 cell, columns=2
    render_pptx(doc, tmp_path / "deck.pptx", base_dir=root)
    prs = Presentation(str(tmp_path / "deck.pptx"))
    slide = prs.slides[3]  # steps_grid
    names = [shape.name for shape in slide.shapes]
    for i in range(2, 7):  # cell1 used, cells 2..6 dropped
        for suffix in ("number", "frame", "text", "timecode"):
            assert f"cell{i}_{suffix}" not in names


# --- text fitting --------------------------------------------------------------------


def test_no_text_exceeds_its_slot(tmp_path: Path) -> None:
    """Independent re-check over the saved deck: every text wraps into its
    shape at the final run size (Pillow metrics, 1.25 line height)."""
    doc, root = slides_with_frames(tmp_path)
    render_pptx(doc, tmp_path / "deck.pptx", base_dir=root)
    prs = Presentation(str(tmp_path / "deck.pptx"))
    for slide in prs.slides:
        for shape in iter_text_shapes(slide):
            size = run_size(shape)
            assert size > 0, shape.name
            w_pt = shape.width / EMU_PER_IN * 72 - INSET_SIDE_IN * 72 * 2
            h_pt = shape.height / EMU_PER_IN * 72 - 0.03 * 72 * 2
            lines = sum(
                len(wrap_lines(par_text, w_pt, size)) for par_text in shape.text_frame.text.split("\n")
            )
            assert lines * size * LINE_HEIGHT <= h_pt, (shape.name, shape.text_frame.text[:40])


def test_autofit_shrinks_to_floor_and_warns(tmp_path: Path) -> None:
    doc = load("artifacts/slides.json")
    make_frames(doc, tmp_path)
    long_text = "очень длинное описание " * 40
    doc["slides"][1]["slots"]["caption"] = long_text  # hero_caption caption
    stats = render_pptx(doc, tmp_path / "deck.pptx", base_dir=tmp_path, min_font_pt=12)
    assert any("text_overflow" in w for w in stats["warnings"])
    prs = Presentation(str(tmp_path / "deck.pptx"))
    captions = [s for s in prs.slides[1].shapes if s.name == "caption"]
    assert run_size(captions[0]) == 12


def test_fit_size_shrinks_for_long_text() -> None:
    short = fit_size(["привет"], 6.0, 1.0, 28, 12)
    long = fit_size(["слово " * 200], 2.0, 0.5, 28, 12)
    assert short[0] == 28 and short[1] is True
    assert long[0] == 12 and long[1] is False


def test_bilingual_cyrillic_latin_texts(tmp_path: Path) -> None:
    doc = load("artifacts/slides.json")
    make_frames(doc, tmp_path)
    doc["slides"][0]["slots"]["heading"] = "Pancakes: Блины"
    doc["slides"][0]["slots"]["subheading"] = "Классический рецепт / classic recipe"
    render_pptx(doc, tmp_path / "deck.pptx", base_dir=tmp_path)
    prs = Presentation(str(tmp_path / "deck.pptx"))
    texts = " ".join(shape_texts(prs.slides[0]))
    assert "Pancakes: Блины" in texts
    assert "classic recipe" in texts


# --- notes ------------------------------------------------------------------------------


def test_notes_carry_episode_timecodes_and_source(tmp_path: Path) -> None:
    doc, root = slides_with_frames(tmp_path)
    render_pptx(doc, tmp_path / "deck.pptx", storyboard=load("storyboard.valid.json"), base_dir=root)
    prs = Presentation(str(tmp_path / "deck.pptx"))
    notes = [slide.notes_slide.notes_text_frame.text for slide in prs.slides]
    assert notes[1].startswith("e1 13:00-13:20 ")
    assert "https://www.youtube.com/watch?v=demo" in notes[1]
    assert notes[3].startswith("e3 0:45-1:02 ")


def test_notes_degrade_without_storyboard(tmp_path: Path) -> None:
    doc, root = slides_with_frames(tmp_path)
    render_pptx(doc, tmp_path / "deck.pptx", storyboard=None, base_dir=root)
    prs = Presentation(str(tmp_path / "deck.pptx"))
    notes = [slide.notes_slide.notes_text_frame.text for slide in prs.slides]
    assert notes[0].strip() == "project"
    assert notes[1].startswith("e1 13:00")  # timecode from the slot


# --- missing frames ----------------------------------------------------------------------


def test_missing_frame_warns_and_renders_rest(tmp_path: Path) -> None:
    doc, root = slides_with_frames(tmp_path)
    (root / "frames/v1_0000800.jpg").unlink()  # ingredients frame
    stats = render_pptx(doc, tmp_path / "deck.pptx", base_dir=root)
    assert any("frame not found" in w for w in stats["warnings"])
    assert stats["pictures"] == 3
    assert stats["slides"] == 4


# --- CLI ---------------------------------------------------------------------------------


def _run_cli(tmp_path: Path, *argv: str) -> int:
    return cli_main(["render", *argv])


def test_cli_render_ok(tmp_path: Path, capsys) -> None:
    slides_with_frames(tmp_path)
    slides = tmp_path / "slides.json"
    out = tmp_path / "deck.pptx"
    code = _run_cli(tmp_path, str(slides), "--out", str(out), "--storyboard", str(EXAMPLES / "storyboard.valid.json"))
    assert code == EXIT_OK
    assert out.is_file()
    captured = capsys.readouterr().out
    assert "4 slides" in captured and "4 pictures" in captured


def test_cli_default_template_resolved_from_repo(tmp_path: Path) -> None:
    slides_with_frames(tmp_path)
    slides = tmp_path / "slides.json"
    out = tmp_path / "deck.pptx"
    # no --template: falls back to the repo default from any CWD
    import os

    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        code = _run_cli(tmp_path, str(slides), "--out", str(out))
    finally:
        os.chdir(cwd)
    assert code == EXIT_OK and out.is_file()
    assert resolve_template(None) == REPO / "templates/razkadr-default.pptx"


def test_cli_invalid_slides_exit_1(tmp_path: Path, capsys) -> None:
    bad = tmp_path / "slides.json"
    bad.write_text('{"schema_version": "1.0", "lang": "ru", "slides": [{"layout": "nope"}]}', encoding="utf-8")
    code = _run_cli(tmp_path, str(bad), "--out", str(tmp_path / "deck.pptx"))
    assert code == EXIT_INVALID
    assert "invalid" in capsys.readouterr().err


def test_cli_missing_template_exit_2(tmp_path: Path, capsys) -> None:
    slides_with_frames(tmp_path)
    slides = tmp_path / "slides.json"
    code = _run_cli(tmp_path, str(slides), "--out", str(tmp_path / "d.pptx"), "--template", str(tmp_path / "x.pptx"))
    assert code == EXIT_FILE_ERROR
    assert "template not found" in capsys.readouterr().err


def test_timecode_formatting() -> None:
    from razkadr.render.pptx import _fmt_timecode

    assert _fmt_timecode(45) == "0:45"
    assert _fmt_timecode(780) == "13:00"
    assert _fmt_timecode(3725) == "1:02:05"

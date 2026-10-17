# Regen tool: run with .venv/bin/python (naryad 027).
"""Fixture builder for visual regression (naryad 027): slides document with
all five layouts + deterministic frames + baseline PNGs.

Run from the repo root:  .venv/bin/python scripts/regen_visual_baselines.py
Re-renders tests/fixtures/visual/baseline/*.png from the current code after
an accepted layout change; prints per-slide diff metrics against the
previous baselines for the PR description.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from PIL import Image

from razkadr.render.visual import (
    DEFAULT_DPI,
    compare_images,
    deck_to_pngs,
)

FIXTURES = REPO / "tests" / "fixtures" / "visual"
BASELINES = FIXTURES / "baseline"
FRAME_COLORS = {  # deterministic frame art: flat fill + band, same bytes every run
    "f_title.jpg": ((38, 70, 83), (231, 111, 81)),
    "f_hero.jpg": ((42, 157, 143), (38, 70, 83)),
    "f_ing.jpg": ((233, 196, 106), (42, 157, 143)),
    "f_step1.jpg": ((231, 111, 81), (233, 196, 106)),
    "f_step2.jpg": ((38, 70, 83), (231, 111, 81)),
    "f_serving.jpg": ((42, 157, 143), (233, 196, 106)),
}
SLIDES_DOC = {
    "schema_version": "1.0",
    "project": {"title": "Шакшука на сковороде", "output_lang": "ru"},
    "slides": [
        {
            "layout": "title",
            "slots": {
                "heading": "Шакшука на сковороде",
                "subheading": "Готовая шакшука в чугунной сковороде: яйца в томатном соусе, зелень.",
                "frame": {"path": "frames/f_title.jpg", "t": 12.5, "episode_id": "e1"},
            },
            "source_refs": ["project"],
            "warnings": [],
        },
        {
            "layout": "hero_caption",
            "slots": {
                "frame": {"path": "frames/f_hero.jpg", "t": 787.0, "episode_id": "e1"},
                "caption": "Готовая шакшука в чугунной сковороде: яйца в томатном соусе, зелень.",
                "timecode": "13:07",
            },
            "source_refs": ["e1"],
            "warnings": [],
        },
        {
            "layout": "ingredients_list",
            "slots": {
                "heading": "Ингредиенты",
                "frame": {"path": "frames/f_ing.jpg", "t": 16.0, "episode_id": "e2"},
                "items": [
                    "Яйца — 4 шт.",
                    "Помидоры — 500 г",
                    "Перец болгарский — 1 шт.",
                    "Лук — 1 шт.",
                    "Чеснок — 2 зубчика",
                    "Паприка — 1 ч. л.",
                    "Зира — 0,5 ч. л.",
                    "Оливковое масло — 2 ст. л.",
                ],
                "timecode": "0:12",
            },
            "source_refs": ["e2"],
            "warnings": [],
        },
        {
            "layout": "steps_grid",
            "slots": {
                "heading": "Шаги",
                "cells": [
                    {
                        "frame": {"path": "frames/f_step1.jpg", "t": 52.5, "episode_id": "e3"},
                        "text": "Разогрейте масло и обжарьте лук до мягкости.",
                        "number": 1,
                        "timecode": "0:45",
                    },
                    {
                        "frame": {"path": "frames/f_step2.jpg", "t": 130.0, "episode_id": "e4"},
                        "text": "Добавьте помидоры и специи, тушите 10 минут.",
                        "number": 2,
                        "timecode": "2:10",
                    },
                ],
            },
            "source_refs": ["e3", "e4"],
            "warnings": [],
        },
        {
            "layout": "serving",
            "slots": {
                "heading": "Подача",
                "frame": {"path": "frames/f_serving.jpg", "t": 800.0, "episode_id": "e5"},
                "caption": "Serve hot with fresh pita bread and cilantro.",
                "timecode": "13:20",
            },
            "source_refs": ["e5"],
            "warnings": [],
        },
    ],
}


def make_frames() -> None:
    frames_dir = FIXTURES / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for name, (fill, band) in FRAME_COLORS.items():
        image = Image.new("RGB", (640, 360), fill)
        image.paste(band, (0, 240, 640, 360))  # deterministic bottom band
        image.save(frames_dir / name, "JPEG", quality=90)


def main() -> int:
    make_frames()
    # the slides document is a fixture too: tests load exactly what was rendered
    (FIXTURES / "slides.json").write_text(
        json.dumps(SLIDES_DOC, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    BASELINES.mkdir(parents=True, exist_ok=True)
    old = sorted(BASELINES.glob("slide-*.png"))
    tmp_old = None
    if old:
        tmp_old = Path("/tmp/razkadr-visual-old")
        if tmp_old.exists():
            shutil.rmtree(tmp_old)
        tmp_old.mkdir()
        for png in old:
            shutil.copy(png, tmp_old / png.name)
        for png in old:
            png.unlink()

    pngs = deck_to_pngs(
        SLIDES_DOC,
        out_deck=FIXTURES / "deck.pptx",
        png_dir=BASELINES,
        template=REPO / "templates" / "razkadr-default.pptx",
        base_dir=FIXTURES,
        dpi=DEFAULT_DPI,
    )
    print(f"rendered {len(pngs)} slides at {DEFAULT_DPI} dpi -> {BASELINES.relative_to(REPO)}")
    for png in pngs:
        if tmp_old and (tmp_old / png.name).is_file():
            metrics = compare_images(png, tmp_old / png.name)
            print(
                f"  {png.name}: diff vs previous baseline: "
                f"{metrics['mismatches']}/{metrics['pixels']} px "
                f"({metrics['diff_ratio']:.5f}), max_delta={metrics['max_delta']}"
            )
        else:
            print(f"  {png.name}: new baseline")
    # the deck and the pdf are intermediate artifacts: keep the fixtures lean
    (FIXTURES / "deck.pptx").unlink(missing_ok=True)
    (FIXTURES / "deck.pdf").unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

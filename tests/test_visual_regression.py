"""Visual regression tests (naryad 027): baselines, tolerance, determinism.

The full render pipeline (slides document -> PPTX -> PDF -> PNG, naryads
022/024 + poppler) is compared against committed baseline screenshots
(tests/fixtures/visual/baseline): a layout drift beyond the pixel
tolerance fails the test. The comparator itself is pinned by a mutation
test (a real breakage must blow past the tolerance) and by the size
mismatch rule. Determinism is tested end to end: two independent renders
of the same document must be pixel-identical.

The baselines are regenerated deliberately by
``scripts/regen_visual_baselines.py`` after an accepted layout change;
the diff metrics it prints go into the PR description. Skipped when
soffice or pdftoppm is absent (the libreoffice marker), like naryad 022.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

from razkadr.render.visual import (
    DEFAULT_PIXEL_THRESHOLD,
    ImageCompareError,
    compare_images,
    deck_to_pngs,
)

REPO = Path(__file__).parent.parent
FIXTURES = REPO / "tests" / "fixtures" / "visual"
BASELINES = FIXTURES / "baseline"
TEMPLATE = REPO / "templates" / "razkadr-default.pptx"

# CI runs the same LibreOffice/poppler stack as the baseline generation;
# the measured diff on identical stacks is 0.0 (scripts/regen_visual_baselines.py
# prints it). The tolerance exists for minor version drift of the render
# stack (font hinting, antialiasing), not for real layout changes - moved
# text measures tens of percent, not fractions of a percent.
VISUAL_TOLERANCE = 0.01

soffice = shutil.which("soffice")
pdftoppm = shutil.which("pdftoppm")

pytestmark = [
    pytest.mark.libreoffice,
    pytest.mark.skipif(soffice is None or pdftoppm is None, reason="soffice/pdftoppm not installed"),
]


@pytest.fixture(scope="module")
def slides_doc() -> dict:
    return json.loads((FIXTURES / "slides.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rendered(tmp_path_factory, slides_doc) -> list[Path]:
    """One fresh render of the fixture document for the whole module."""
    work = tmp_path_factory.mktemp("visual")
    return deck_to_pngs(
        slides_doc,
        out_deck=work / "deck.pptx",
        png_dir=work / "png",
        template=TEMPLATE,
        base_dir=FIXTURES,
    )


def _baseline_paths() -> list[Path]:
    return sorted(BASELINES.glob("slide-*.png"))


# --- baseline integrity ---------------------------------------------------------------


def test_baselines_match_the_slides_document(slides_doc):
    baselines = _baseline_paths()
    assert len(baselines) == len(slides_doc["slides"]), (
        "stale baselines: regenerate with scripts/regen_visual_baselines.py"
    )
    layouts = [s["layout"] for s in slides_doc["slides"]]
    assert layouts == ["title", "hero_caption", "ingredients_list", "steps_grid", "serving"]


# --- comparison against committed baselines -------------------------------------------


def test_render_matches_committed_baselines(rendered):
    baselines = _baseline_paths()
    assert len(rendered) == len(baselines)
    for actual, baseline in zip(rendered, baselines):
        metrics = compare_images(actual, baseline)
        assert metrics["diff_ratio"] <= VISUAL_TOLERANCE, (
            f"{actual.name} drifted from {baseline.name}: "
            f"{metrics['mismatches']}/{metrics['pixels']} px "
            f"({metrics['diff_ratio']:.5f}), max_delta={metrics['max_delta']}; "
            "regenerate baselines only after an accepted layout change"
        )


def test_render_is_pixel_deterministic_across_runs(tmp_path, slides_doc):
    """Two independent renders of the same document are pixel-identical."""
    second_pngs = deck_to_pngs(
        slides_doc,
        out_deck=tmp_path / "second.pptx",
        png_dir=tmp_path / "second-png",
        template=TEMPLATE,
        base_dir=FIXTURES,
    )
    # compare the second render against the committed baselines with zero tolerance:
    # the baselines were produced by the same deterministic pipeline
    for actual, baseline in zip(second_pngs, _baseline_paths()):
        metrics = compare_images(actual, baseline, pixel_threshold=0)
        assert metrics["diff_ratio"] == 0.0, (
            f"non-deterministic render: {actual.name} differs from baseline "
            f"({metrics['mismatches']} px)"
        )


# --- comparator sensitivity (mutation) -------------------------------------------------


def test_comparator_catches_layout_drift(rendered, tmp_path):
    """A real breakage (a band drawn over the text zone) blows past the tolerance.

    Mutation check: the tolerance must be far below what a moved text block
    or a lost frame produces, otherwise the baseline test would be blind.
    """
    broken = tmp_path / "broken.png"
    image = Image.open(rendered[0]).convert("RGB")  # title slide
    image.paste((255, 0, 0), (0, 60, image.size[0], 140))  # a hard drift band
    image.save(broken)
    metrics = compare_images(broken, _baseline_paths()[0])
    assert metrics["diff_ratio"] > VISUAL_TOLERANCE * 10
    assert metrics["max_delta"] > DEFAULT_PIXEL_THRESHOLD


def test_comparator_tolerates_small_noise(rendered, tmp_path):
    """JPEG-style noise (a few close pixels) stays within the tolerance."""
    noisy = tmp_path / "noisy.png"
    image = Image.open(rendered[0]).convert("RGB")
    pixels = image.load()
    for i in range(0, image.size[0], 7):  # sparse dim pixels, delta = 10 < threshold
        pixels[i, 5] = tuple(max(0, c - 10) for c in pixels[i, 5])
    image.save(noisy)
    metrics = compare_images(noisy, _baseline_paths()[0])
    assert metrics["diff_ratio"] == 0.0  # deltas below pixel_threshold never mismatch


def test_comparator_size_mismatch_is_an_error(rendered, tmp_path):
    other = tmp_path / "other.png"
    Image.new("RGB", (640, 360), "white").save(other)
    with pytest.raises(ImageCompareError, match="sizes differ"):
        compare_images(other, _baseline_paths()[0])

"""Visual regression toolkit (naryad 027): deck -> PNGs, image comparison.

The render pipeline (PPTX via python-pptx, PDF via LibreOffice headless -
naryads 022/024) is turned into per-slide PNGs (pdftoppm from poppler) so
tests can compare a fresh render against committed baseline screenshots:
a layout drift beyond the tolerance fails the test. Comparison is pure
arithmetic over RGB pixels (Pillow): a pixel mismatches when any channel
changes by more than ``pixel_threshold`` (255-scale) - that absorbs PNG
recompression and antialiasing jitter, while moved text, lost frames or
changed fonts blow far past it. Nothing here mutates documents or calls
providers.

The baselines live with the test fixtures; ``scripts/regen_visual_baselines.py``
re-renders them deliberately when a layout change is accepted (the auditor
reviews the baseline diff in the PR). The comparison helpers are importable
without LibreOffice: only the render step needs soffice/pdftoppm on PATH.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from PIL import Image

from razkadr.render.pdf import export_pdf

DEFAULT_DPI = 60
DEFAULT_PIXEL_THRESHOLD = 24  # per-channel delta (0..255) that still counts as the same pixel
INSTALL_HINT = "install poppler-utils (pdftoppm) to render slides to PNG"


class VisualRenderError(Exception):
    """Slide PNG rendering cannot proceed (no pdftoppm, no PDF, empty output)."""


class ImageCompareError(Exception):
    """Images cannot be compared (different sizes)."""


def pdf_to_pngs(pdf: Path, out_dir: Path, dpi: int = DEFAULT_DPI) -> list[Path]:
    """Rasterize every PDF page to ``slide-<N>.png`` via pdftoppm (poppler).

    Pages are ordered like the slides of the deck (pdftoppm's numbering).
    """
    pdftoppm = shutil.which("pdftoppm")
    if pdftoppm is None:
        raise VisualRenderError(INSTALL_HINT)
    pdf = Path(pdf).resolve()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / "slide"
    subprocess.run(
        [pdftoppm, "-png", "-r", str(dpi), str(pdf), str(prefix)],
        check=False,
        capture_output=True,
        timeout=120,
    )
    produced = sorted(out_dir.glob("slide-*.png"))
    if not produced:
        raise VisualRenderError(f"pdftoppm produced no PNGs for {pdf.name} (is the PDF valid?)")
    return produced


def deck_to_pngs(
    slides_doc: dict,
    out_deck: Path,
    png_dir: Path,
    template: str | Path | None = None,
    base_dir: Path | None = None,
    dpi: int = DEFAULT_DPI,
) -> list[Path]:
    """Full pipeline: slides document -> PPTX -> PDF -> per-slide PNGs.

    Needs soffice (LibreOffice) and pdftoppm (poppler) on PATH; both are
    installed in CI. Returns the PNG paths in slide order.
    """
    from razkadr.render.pptx import render_pptx

    out_deck = Path(out_deck)
    out_deck.parent.mkdir(parents=True, exist_ok=True)
    render_pptx(slides_doc, out_deck, template=template, base_dir=base_dir)
    pdf = export_pdf(out_deck, out_deck.with_suffix(".pdf"))
    return pdf_to_pngs(pdf, Path(png_dir), dpi=dpi)


def load_rgb(path: str | Path) -> Image.Image:
    """Load an image as RGB (grayscale/RGBA inputs are converted)."""
    image = Image.open(Path(path)).convert("RGB")
    image.load()
    return image


def compare_images(
    actual: str | Path,
    baseline: str | Path,
    pixel_threshold: int = DEFAULT_PIXEL_THRESHOLD,
) -> dict:
    """Pixel comparison of two same-size images.

    Returns ``{"pixels", "mismatches", "diff_ratio", "max_delta"}``:
    a pixel mismatches when any channel differs by more than
    ``pixel_threshold``; ``diff_ratio`` is mismatches / pixels.
    Different sizes raise ``ImageCompareError`` - a size change is always
    a layout drift, not a tolerance question.
    """
    import numpy as np

    a = load_rgb(actual)
    b = load_rgb(baseline)
    if a.size != b.size:
        raise ImageCompareError(
            f"image sizes differ: {Path(actual).name} {a.size} vs baseline {b.size} "
            "(a size change is a layout drift)"
        )
    if a.size[0] == 0 or a.size[1] == 0:
        raise ImageCompareError("cannot compare empty images")
    delta = np.abs(np.asarray(a, dtype=np.int16) - np.asarray(b, dtype=np.int16)).max(axis=2)
    mismatches = int((delta > pixel_threshold).sum())
    max_delta = int(delta.max())
    return {
        "pixels": int(delta.size),
        "mismatches": mismatches,
        "diff_ratio": mismatches / delta.size,
        "max_delta": max_delta,
    }

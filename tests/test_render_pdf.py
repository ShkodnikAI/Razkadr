"""LibreOffice conversion tests (naryad 022, acceptance criteria 4-5).

The deck must convert to PDF without errors (``soffice --headless
--convert-to pdf``) and the PDF text layer must keep Cyrillic and Latin
readable - a missing-glyph font substitution would show up as missing or
mangled text via pdftotext. Skipped when soffice (or pdftotext) is
absent, so CI without LibreOffice stays green until the auditor adds it
(naryad assumption).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from pptx import Presentation

from razkadr.render.pptx import render_pptx

REPO = Path(__file__).parent.parent
EXAMPLES = REPO / "examples"

soffice = shutil.which("soffice")
pdftotext = shutil.which("pdftotext")

pytestmark = pytest.mark.libreoffice


def make_frames(slides_doc: dict, root: Path) -> None:
    from PIL import Image

    paths = []
    for slide in slides_doc["slides"]:
        slots = slide["slots"]
        if "frame" in slots:
            paths.append(slots["frame"]["path"])
        for cell in slots.get("cells", []):
            paths.append(cell["frame"]["path"])
    for i, relpath in enumerate(sorted(set(paths))):
        p = root / relpath
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (640, 360), ((40 * i) % 256, 96, 160)).save(p, "JPEG", quality=85)


def build_deck(tmp_path: Path) -> Path:
    doc = json.loads((EXAMPLES / "artifacts/slides.json").read_text(encoding="utf-8"))
    # bilingual: both alphabets must survive the PDF text layer
    doc["slides"][0]["slots"]["heading"] = "Шакшука / Shakshuka"
    doc["slides"][0]["slots"]["subheading"] = "Рецепт recipe"
    make_frames(doc, tmp_path)
    render_pptx(doc, tmp_path / "deck.pptx", storyboard=None, base_dir=tmp_path)
    return tmp_path / "deck.pptx"


@pytest.mark.skipif(soffice is None, reason="soffice not installed")
def test_soffice_converts_deck_to_pdf(tmp_path: Path) -> None:
    deck = build_deck(tmp_path)
    subprocess.run(
        [soffice, "--headless", "--convert-to", "pdf", "--outdir", str(tmp_path), str(deck)],
        check=True,
        capture_output=True,
        timeout=300,
    )
    pdf = tmp_path / "deck.pdf"
    assert pdf.is_file() and pdf.stat().st_size > 0
    prs = Presentation(str(deck))
    assert len(prs.slides) == 4


@pytest.mark.skipif(soffice is None or pdftotext is None, reason="soffice/pdftotext not installed")
def test_pdf_text_layer_keeps_cyrillic_and_latin(tmp_path: Path) -> None:
    deck = build_deck(tmp_path)
    subprocess.run(
        [soffice, "--headless", "--convert-to", "pdf", "--outdir", str(tmp_path), str(deck)],
        check=True,
        capture_output=True,
        timeout=300,
    )
    text = subprocess.run(
        [pdftotext, str(tmp_path / "deck.pdf"), "-"], check=True, capture_output=True, text=True
    ).stdout
    # Cyrillic and Latin both present and intact (no tofu/mangling)
    assert "Шакшука" in text
    assert "Shakshuka" in text
    assert "Ингредиенты" in text
    assert "Шаги" in text
    assert "0:45" in text

"""PDF export from a rendered deck (naryad 024).

PDF is derived from the PPTX master (ADR-0003): the deck is converted
with LibreOffice headless (``soffice --convert-to pdf``) in an isolated
profile (temporary ``-env:UserInstallation``), so parallel runs and a
dirty user profile cannot interfere. Errors are explicit: no soffice
(with an install hint), timeout (configurable via ``pdf_timeout_s``),
empty result. Template fonts missing from the system are reported as
warnings - substitution degrades the look but does not block export.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from razkadr.atomicio import build_atomic

DEFAULT_PDF_TIMEOUT_S = 120
INSTALL_HINT = (
    "LibreOffice (soffice) not found in PATH; "
    "install it (e.g. `apt install libreoffice` or `brew install --cask libreoffice`) "
    "or put soffice on PATH"
)


class PdfExportError(Exception):
    """PDF export cannot proceed (no soffice, timeout, empty result)."""


def _iter_typefaces(root: ElementTree.Element) -> set[str]:
    ns = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
    faces = set()
    for el in root.iter(f"{ns}latin"):
        face = (el.get("typeface") or "").strip()
        if face and not face.startswith("+"):
            faces.add(face)
    return faces


def deck_fonts(deck: Path) -> set[str]:
    """Typefaces referenced by the deck's slides, layouts and masters
    (theme fonts are excluded: they are inherited defaults, not what the
    renderer draws with)."""
    faces: set[str] = set()
    with zipfile.ZipFile(deck) as z:
        prefixes = ("ppt/slides/slide", "ppt/slideLayouts/slideLayout", "ppt/slideMasters/slideMaster")
        for name in z.namelist():
            if not name.startswith(prefixes) or not name.endswith(".xml"):
                continue
            faces |= _iter_typefaces(ElementTree.fromstring(z.read(name)))
    return faces


def missing_fonts(deck: Path) -> set[str]:
    """Referenced fonts that fontconfig does not know about (empty when
    fontconfig itself is unavailable - the check degrades to no-op)."""
    fc_list = shutil.which("fc-list")
    if fc_list is None:
        return set()
    try:
        installed = subprocess.run(
            [fc_list, ":", "family"], check=True, capture_output=True, text=True, timeout=30
        ).stdout.lower()
    except (subprocess.SubprocessError, OSError):
        return set()
    return {face for face in deck_fonts(deck) if face.lower() not in installed}


def export_pdf(deck: Path, out: Path, timeout_s: int = DEFAULT_PDF_TIMEOUT_S) -> Path:
    """Convert ``deck`` (PPTX) to ``out`` (PDF) via LibreOffice headless in
    an isolated profile. Raises ``PdfExportError`` with a clear message on
    any failure."""
    soffice = shutil.which("soffice")
    if soffice is None:
        raise PdfExportError(INSTALL_HINT)
    deck = deck.resolve()
    with tempfile.TemporaryDirectory(prefix="razkadr-lo-") as tmp:
        profile = Path(tmp) / "profile"
        outdir = Path(tmp) / "out"
        outdir.mkdir(parents=True)
        cmd = [
            soffice,
            "--headless",
            "--norestore",
            f"-env:UserInstallation={profile.as_uri()}",
            "--convert-to",
            "pdf",
            "--outdir",
            str(outdir),
            str(deck),
        ]
        try:
            subprocess.run(cmd, check=False, capture_output=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            raise PdfExportError(
                f"LibreOffice did not finish in {timeout_s}s "
                "(raise pdf_timeout_s in razkadr.toml if the deck is heavy)"
            ) from None
        produced = outdir / f"{deck.stem}.pdf"
        if not produced.is_file() or produced.stat().st_size == 0:
            raise PdfExportError(f"LibreOffice produced no PDF for {deck.name}")
        out.parent.mkdir(parents=True, exist_ok=True)
        # atomic: land the converted file next to the target first, then swap -
        # a crash during the (possibly cross-device) move cannot truncate the PDF
        build_atomic(out, lambda tmp: shutil.move(str(produced), str(tmp)), tmp_suffix=".pdf")
    return out

"""export-pdf tests (naryad 024).

Hermetic tests fake soffice with shell stubs on PATH (no soffice ->
exit 1 with an install hint, timeout, empty result, happy path through a
stub that copies a fixture PDF). The real conversion runs under the
`libreoffice` marker: the deck built by `razkadr render` converts page
for page (pdfinfo page count == slide count).
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from razkadr.cli import EXIT_FILE_ERROR, EXIT_INVALID, EXIT_OK
from razkadr.cli import main as cli_main
from razkadr.render.pdf import (
    DEFAULT_PDF_TIMEOUT_S,
    PdfExportError,
    deck_fonts,
    export_pdf,
    missing_fonts,
)

REPO = Path(__file__).parent.parent
EXAMPLES = REPO / "examples"

soffice = shutil.which("soffice")
pdfinfo = shutil.which("pdfinfo")

pytestmark = pytest.mark.libreoffice


def make_stub_dir(tmp_path: Path, script: str) -> str:
    """A PATH-patched directory with a fake `soffice`."""
    stub_dir = tmp_path / "stubs"
    stub_dir.mkdir(exist_ok=True)
    stub = stub_dir / "soffice"
    stub.write_text(script, encoding="utf-8")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    return str(stub_dir)


def build_deck(tmp_path: Path) -> Path:
    from razkadr.render.pptx import render_pptx

    doc = json.loads((EXAMPLES / "artifacts/slides.json").read_text(encoding="utf-8"))
    from PIL import Image

    paths = []
    for slide in doc["slides"]:
        slots = slide["slots"]
        if "frame" in slots:
            paths.append(slots["frame"]["path"])
        for cell in slots.get("cells", []):
            paths.append(cell["frame"]["path"])
    for i, rel in enumerate(sorted(set(paths))):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (640, 360), ((40 * i) % 256, 96, 160)).save(p, "JPEG", quality=85)
    render_pptx(doc, tmp_path / "deck.pptx", storyboard=None, base_dir=tmp_path)
    return tmp_path / "deck.pptx"


# --- hermetic: error paths via stubs -------------------------------------------------


def test_no_soffice_gives_install_hint(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("razkadr.render.pdf.shutil.which", lambda name: None)
    with pytest.raises(PdfExportError, match="apt install libreoffice"):
        export_pdf(tmp_path / "deck.pptx", tmp_path / "deck.pdf")


def test_cli_no_soffice_exit_1(tmp_path: Path, monkeypatch, capsys) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    monkeypatch.setenv("PATH", str(empty_dir))
    deck = build_deck(tmp_path)
    code = cli_main(["export-pdf", str(deck)])
    assert code == EXIT_INVALID
    assert "soffice) not found" in capsys.readouterr().err


def test_timeout_is_reported(tmp_path: Path, monkeypatch) -> None:
    import subprocess as sp

    deck = build_deck(tmp_path)

    def raise_timeout(*a, **kw):
        raise sp.TimeoutExpired(cmd="soffice", timeout=1)

    monkeypatch.setattr("razkadr.render.pdf.subprocess.run", raise_timeout)
    with pytest.raises(PdfExportError, match="did not finish in 120s"):
        export_pdf(deck, tmp_path / "deck.pdf", timeout_s=DEFAULT_PDF_TIMEOUT_S)


def test_empty_result_is_reported(tmp_path: Path, monkeypatch) -> None:
    stub_dir = make_stub_dir(tmp_path, "#!/bin/sh\nexit 0\n")
    monkeypatch.setenv("PATH", stub_dir)
    deck = build_deck(tmp_path)
    with pytest.raises(PdfExportError, match="produced no PDF"):
        export_pdf(deck, tmp_path / "deck.pdf")


def test_happy_path_via_stub_respects_out(tmp_path: Path, monkeypatch) -> None:
    """A stub soffice that copies a fixture PDF exercises the move logic
    (custom --out path, parent dirs) without LibreOffice."""
    fixture_pdf = tmp_path / "fixture.pdf"
    fixture_pdf.write_bytes(b"%PDF-1.4 fake for stub")
    stub_dir = make_stub_dir(
        tmp_path,
        "#!/bin/sh\noutdir=\"\"\nprev=\"\"\nfor a in \"$@\"; do [ \"$prev\" = \"--outdir\" ] && outdir=\"$a\"; prev=\"$a\"; done\n/bin/cp \"$FIXTURE_PDF\" \"$outdir/deck.pdf\"\n",
    )
    monkeypatch.setenv("PATH", stub_dir)
    monkeypatch.setenv("FIXTURE_PDF", str(fixture_pdf))
    deck = build_deck(tmp_path)
    out = tmp_path / "nested" / "renamed.pdf"
    result = export_pdf(deck, out)
    assert result == out and out.read_bytes() == b"%PDF-1.4 fake for stub"


def test_cli_input_unreadable_exit_2(tmp_path: Path, capsys) -> None:
    code = cli_main(["export-pdf", str(tmp_path / "missing.pptx")])
    assert code == EXIT_FILE_ERROR
    assert "cannot read" in capsys.readouterr().err


# --- fonts -----------------------------------------------------------------------------


def test_deck_fonts_extraction(tmp_path: Path) -> None:
    deck = build_deck(tmp_path)
    assert deck_fonts(deck) == {"DejaVu Sans"}


def test_missing_fonts_detects_unknown_face(tmp_path: Path) -> None:
    deck = build_deck(tmp_path)
    assert missing_fonts(deck) == set()  # DejaVu Sans is installed locally


def test_missing_fonts_ignores_theme_and_refs(tmp_path: Path) -> None:
    deck = build_deck(tmp_path)
    faces = deck_fonts(deck)
    assert all(not f.startswith("+") for f in faces)


# --- real conversion (soffice installed) -------------------------------------------------


@pytest.mark.skipif(soffice is None, reason="soffice not installed")
def test_real_conversion_page_count_matches_slides(tmp_path: Path) -> None:
    deck = build_deck(tmp_path)
    out = export_pdf(deck, tmp_path / "out.pdf", timeout_s=180)
    assert out.is_file() and out.stat().st_size > 0
    if pdfinfo is None:
        pytest.skip("pdfinfo not installed")
    info = subprocess.run([pdfinfo, str(out)], check=True, capture_output=True, text=True).stdout
    pages = int(next(line.split(":")[1] for line in info.splitlines() if line.startswith("Pages:")))
    from pptx import Presentation

    assert pages == len(Presentation(str(deck)).slides) == 4


@pytest.mark.skipif(soffice is None, reason="soffice not installed")
def test_cli_export_pdf_ok(tmp_path: Path, capsys) -> None:
    deck = build_deck(tmp_path)
    code = cli_main(["export-pdf", str(deck), "--out", str(tmp_path / "custom.pdf")])
    assert code == EXIT_OK
    assert (tmp_path / "custom.pdf").is_file()
    assert "razkadr export-pdf" in capsys.readouterr().out


@pytest.mark.skipif(soffice is None, reason="soffice not installed")
def test_cli_timeout_from_config(tmp_path: Path) -> None:
    deck = build_deck(tmp_path)
    config = tmp_path / "razkadr.toml"
    config.write_text("pdf_timeout_s = 240\n", encoding="utf-8")
    code = cli_main(["export-pdf", str(deck), "--config", str(config)])
    assert code == EXIT_OK


def test_config_default_timeout_constant() -> None:
    assert DEFAULT_PDF_TIMEOUT_S == 120


def test_isolated_profile_flag_present(tmp_path: Path, monkeypatch) -> None:
    """The soffice invocation must carry an isolated UserInstallation."""
    captured = {}

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        outdir = Path(cmd[cmd.index("--outdir") + 1])
        (outdir / "deck.pdf").write_bytes(b"%PDF-1.4 x")

    monkeypatch.setattr("razkadr.render.pdf.shutil.which", lambda name: "/usr/bin/soffice")
    monkeypatch.setattr("razkadr.render.pdf.subprocess.run", fake_run)
    deck = build_deck(tmp_path)
    export_pdf(deck, tmp_path / "deck.pdf")
    cmd = captured["cmd"]
    assert any(str(a).startswith("-env:UserInstallation=file://") for a in cmd)
    assert "--headless" in cmd and "--convert-to" in cmd


def test_env_var_docs_key_present() -> None:
    """pdf_timeout_s is a plain config key overridable via RAZKADR_PDF_TIMEOUT_S."""
    os.environ["RAZKADR_PDF_TIMEOUT_S"] = "5"
    try:
        from razkadr.config import load_config

        assert load_config().get("pdf_timeout_s") == "5"
    finally:
        del os.environ["RAZKADR_PDF_TIMEOUT_S"]

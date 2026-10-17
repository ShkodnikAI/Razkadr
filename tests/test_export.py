"""HTML/JSON export tests (naryad 026): hermetic, no network.

The deck must be fully self-contained (no external scripts/styles), keep
slide and timecode counts in sync with slides.json, build YouTube t=
links and file:// fragments from the Storyboard, and degrade to text
timecodes when the Storyboard is absent.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from razkadr.cli import EXIT_FILE_ERROR, EXIT_INVALID, EXIT_OK
from razkadr.cli import main as cli_main
from razkadr.render.html import export_html, export_json, link_moment
from razkadr.validation import validate_document

EXAMPLES = Path(__file__).parent.parent / "examples"


def load(name: str) -> dict:
    return json.loads((EXAMPLES / name).read_text(encoding="utf-8"))


def make_frame(path: Path, content: bytes = b"fake jpeg") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


# --- link rules -------------------------------------------------------------------


def test_youtube_watch_link_gets_t_seconds() -> None:
    href = link_moment({"url": "https://www.youtube.com/watch?v=abc123"}, 780.0)
    assert href == "https://www.youtube.com/watch?v=abc123&t=780s"


def test_youtube_existing_t_param_replaced() -> None:
    href = link_moment({"url": "https://www.youtube.com/watch?v=abc123&t=10s"}, 95.0)
    assert href == "https://www.youtube.com/watch?v=abc123&t=95s"


def test_youtu_be_link() -> None:
    href = link_moment({"url": "https://youtu.be/xyz987"}, 3725.0)
    assert href == "https://youtu.be/xyz987?t=3725s"


def test_tiktok_and_unknown_urls_unchanged() -> None:
    assert link_moment({"url": "https://www.tiktok.com/@u/video/123"}, 12.0) == "https://www.tiktok.com/@u/video/123"
    assert link_moment({"url": "https://vimeo.com/42"}, 60.0) == "https://vimeo.com/42"


def test_local_file_gets_media_fragment(tmp_path: Path) -> None:
    href = link_moment({"path": str(tmp_path / "source.mp4")}, 130.0)
    assert href.startswith("file://") and href.endswith("#t=130")
    assert "source.mp4" in href


def test_no_source_no_link() -> None:
    assert link_moment({}, 10.0) is None


# --- HTML deck ---------------------------------------------------------------------


def build_deck(tmp_path: Path, with_frames: bool = True) -> tuple[Path, dict]:
    slides_doc = load("artifacts/slides.json")
    storyboard = load("storyboard.valid.json")
    frames_dir = tmp_path / "frames"
    if with_frames:
        for episode in storyboard["episodes"]:
            make_frame(frames_dir / Path(episode["frame"]["path"]).name)
    out = tmp_path / "deck.html"
    export_html(slides_doc, out, storyboard=storyboard, base_dir=tmp_path)
    return out, slides_doc


def test_deck_self_contained_no_external_requests(tmp_path: Path) -> None:
    out, _ = build_deck(tmp_path)
    html = out.read_text(encoding="utf-8")
    assert "<style>" in html and "<script>" in html  # inline, not files
    for tag in re.findall(r"<(?:script|link|img)[^>]+>", html):
        src = re.search(r'(?:src|href)="([^"]+)"', tag)
        if src:
            assert not src.group(1).startswith(("http://", "https://")), tag


def test_deck_slide_and_timecode_counts_match_slides(tmp_path: Path) -> None:
    out, slides_doc = build_deck(tmp_path)
    html = out.read_text(encoding="utf-8")
    assert html.count('<section class="slide') == len(slides_doc["slides"])
    expected_timecodes = 0
    for slide in slides_doc["slides"]:
        if "timecode" in slide["slots"]:
            expected_timecodes += 1
        expected_timecodes += len(slide["slots"].get("cells", []))
    assert html.count('class="tc') == expected_timecodes


def test_deck_youtube_links_carry_correct_t(tmp_path: Path) -> None:
    out, slides_doc = build_deck(tmp_path)
    html = out.read_text(encoding="utf-8")
    storyboard = load("storyboard.valid.json")
    # only slots with a timecode are linked: hero_caption and grid cells
    # (the ingredients list has no timecode slot by design)
    linked_ids = set()
    for slide in slides_doc["slides"]:
        if "timecode" in slide["slots"]:
            linked_ids.add(slide["slots"]["frame"]["episode_id"])
        for cell in slide["slots"].get("cells", []):
            linked_ids.add(cell["frame"]["episode_id"])
    for episode in storyboard["episodes"]:
        if episode["id"] in linked_ids:
            assert f"t={int(episode['t_start'])}s" in html, episode["id"]


def test_deck_without_storyboard_keeps_text_timecodes(tmp_path: Path) -> None:
    slides_doc = load("artifacts/slides.json")
    out = tmp_path / "deck.html"
    export_html(slides_doc, out, storyboard=None)
    html = out.read_text(encoding="utf-8")
    assert '<a class="tc"' not in html
    assert '<span class="tc">' in html


def test_deck_copies_existing_frames_and_marks_missing(tmp_path: Path) -> None:
    out, _ = build_deck(tmp_path, with_frames=True)
    assets = list((tmp_path / "deck_assets").iterdir())
    assert assets, "existing frames are copied"
    html_with = out.read_text(encoding="utf-8")
    assert "deck_assets/" in html_with
    out2, _ = build_deck(tmp_path / "noframes", with_frames=False)
    html_without = out2.read_text(encoding="utf-8")
    assert "кадр недоступен" in html_without


def test_deck_theme_print_navigation_markers(tmp_path: Path) -> None:
    out, _ = build_deck(tmp_path)
    html = out.read_text(encoding="utf-8")
    assert "@media print" in html
    assert "prefers-color-scheme" in html
    assert "keydown" in html  # keyboard navigation
    assert "ArrowRight" in html


def test_deck_escapes_html_in_texts(tmp_path: Path) -> None:
    slides_doc = load("artifacts/slides.json")
    slides_doc["slides"][0]["slots"]["heading"] = "<script>alert(1)</script> &Co"
    out = export_html(slides_doc, tmp_path / "deck.html", storyboard=None)
    html = out.read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in html.replace('<script>', '[[S]]') or "<h1>&lt;script&gt;" in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt; &amp;Co" in html


def test_export_not_a_slides_doc(tmp_path: Path) -> None:
    from razkadr.render.html import ExportError

    with __import__("pytest").raises(ExportError, match="not a slides document"):
        export_html({"foo": 1}, tmp_path / "deck.html")


# --- JSON export --------------------------------------------------------------------


def test_export_json_roundtrip_valid(tmp_path: Path) -> None:
    storyboard = load("storyboard.valid.json")
    out = export_json(storyboard, tmp_path / "storyboard.out.json")
    reloaded = json.loads(out.read_text(encoding="utf-8"))
    assert reloaded == storyboard
    assert validate_document(reloaded, "storyboard") == []


# --- CLI -----------------------------------------------------------------------------


def test_cli_export_html(tmp_path: Path) -> None:
    slides_path = tmp_path / "slides.json"
    slides_path.write_text(json.dumps(load("artifacts/slides.json"), ensure_ascii=False), encoding="utf-8")
    storyboard_path = tmp_path / "storyboard.json"
    storyboard_path.write_text(json.dumps(load("storyboard.valid.json"), ensure_ascii=False), encoding="utf-8")
    for episode in load("storyboard.valid.json")["episodes"]:
        make_frame(tmp_path / Path(episode["frame"]["path"]))
    code = cli_main(["export-html", str(slides_path), "--out", str(tmp_path / "deck.html")])
    assert code == EXIT_OK
    assert (tmp_path / "deck.html").is_file()
    assert list((tmp_path / "deck_assets").iterdir())


def test_cli_export_json(tmp_path: Path) -> None:
    src = tmp_path / "storyboard.json"
    src.write_text(json.dumps(load("storyboard.valid.json"), ensure_ascii=False), encoding="utf-8")
    code = cli_main(["export-json", str(src), "--out", str(tmp_path / "out.json")])
    assert code == EXIT_OK
    assert validate_document(json.loads((tmp_path / "out.json").read_text(encoding="utf-8")), "storyboard") == []


def test_cli_export_json_invalid_storyboard_exit_1(tmp_path: Path) -> None:
    src = tmp_path / "storyboard.json"
    src.write_text('{"schema_version": "9.9"}', encoding="utf-8")
    assert cli_main(["export-json", str(src), "--out", str(tmp_path / "o.json")]) == EXIT_INVALID


def test_cli_export_html_missing_slides_exit_2(tmp_path: Path) -> None:
    assert cli_main(["export-html", str(tmp_path / "no.json"), "--out", str(tmp_path / "d.html")]) == EXIT_FILE_ERROR

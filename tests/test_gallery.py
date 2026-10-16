"""Gallery tests (naryad 041): label sheet, selection view, labels marking.

The gallery is a pure file-to-file transform (frames.json + optional
storyboard.json / labels JSON -> one self-contained gallery.html), so tests
build a tiny synthetic run directory on disk -- no ffmpeg, no network, no
browser. The page embeds its full data payload as JSON in
`<script id="gallery-data" type="application/json">`, which tests parse to
check frame counts and timecodes against frames.json; DOM structure is
checked with the stdlib HTML parser (no external deps).

Acceptance criteria covered:
- the page is self-contained: no external http(s) references at all
- frame count and timecodes on the page match frames.json
- label-sheet and selection-view modes work on the fixture run
"""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

from razkadr.cli import EXIT_FILE_ERROR, EXIT_OK, main
from razkadr.gallery import (
    GalleryError,
    build_gallery_data,
    in_sheet_flags,
    match_labels,
    run_gallery,
)
from razkadr.gallery_page import render_html
from razkadr.validation import validate_document

# a 1x1 gray JPEG so <img> files are real images when a human opens the page
_JPEG_1PX = bytes.fromhex(
    "ffd8ffe000104a46494600010100000100010000ffdb004300080606070605080707070909"
    "080a0c140d0c0b0b0c1912130f141d1a1f1e1d1a1c1c20242e2720222c231c1c2837292c31"
    "3434341f27393d38323c2e333432ffc0000b080001000101011100ffc4001f000001050101"
    "0101010100000000000000000102030405060708090a0bffc400b510000201030302040305"
    "0504040000017d01020300041105122131410613516107227114328191a1082342b1c11552"
    "d1f02433627282090a161718191a25262728292a3435363738393a434445464748494a5354"
    "55565758595a636465666768696a737475767778797a838485868788898a92939495969798"
    "999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9"
    "dae1e2e3e4e5e6e7e8e9eaf1f2f3f4f5f6f7f8f9faffda0008010100003f00fbfe8a28a280"
    "0a28a2800a28a2ffd9"
)


class _Dom(HTMLParser):
    """Collects elements with data-* attributes for structural assertions."""

    def __init__(self) -> None:
        super().__init__()
        self.elements: list[tuple[str, dict[str, str | None]]] = []
        self._depth_open = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.elements.append((tag, dict(attrs)))


def _parse(html: str) -> _Dom:
    parser = _Dom()
    parser.feed(html)
    return parser


def _data_block(html: str) -> dict:
    match = re.search(
        r'<script id="gallery-data" type="application/json">(.*?)</script>',
        html,
        re.DOTALL,
    )
    assert match, "gallery-data JSON block missing"
    raw = match.group(1)
    assert "</script>" not in raw, "embedded JSON must not close the script tag"
    return json.loads(raw)


def _cards(dom: _Dom) -> list[tuple[str, dict[str, str | None]]]:
    return [(tag, attrs) for tag, attrs in dom.elements if tag == "figure" and "data-t" in attrs]


# --- fixture run directory -----------------------------------------------------


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    """A minimal but complete run: source.json + frames.json + real jpg files."""
    run = tmp_path / "runs" / "demo-001"
    (run / "frames").mkdir(parents=True)
    source = {
        "schema_version": "1.1",
        "path": "normalized.mp4",
        "sha256": "ab" * 32,
        "duration_s": 30.0,
        "fps": 25.0,
        "width": 640,
        "height": 360,
    }
    frames = []
    for i in range(10):
        t = round(0.5 + i * 2.7, 3)
        name = f"{i:06d}.jpg"
        (run / "frames" / name).write_bytes(_JPEG_1PX)
        frames.append(
            {
                "t": t,
                "file": f"frames/{name}",
                "scene_id": i // 3,
                "sharpness": round(0.5 + 0.05 * i, 3),
                "phash": f"{i:016x}",
                "exposure": 0.9,
                **({"rejected": True, "reject_reason": "duplicate of 3.200s"} if i == 4 else {}),
            }
        )
    (run / "source.json").write_text(json.dumps(source), encoding="utf-8")
    (run / "frames.json").write_text(
        json.dumps({"schema_version": "1.1", "frames": frames}), encoding="utf-8"
    )
    return run


def _storyboard(run_dir: Path) -> dict:
    """Draft storyboard over the fixture frames (hero at frame 1, step at 6)."""
    frames = json.loads((run_dir / "frames.json").read_text(encoding="utf-8"))["frames"]
    f1, f6 = frames[1], frames[6]
    return {
        "schema_version": "1.0",
        "project": {
            "id": "demo",
            "title": "Demo storyboard",
            "preset": "recipe",
            "output_lang": "ru",
        },
        "sources": [{"id": "src", "path": "normalized.mp4", "duration_s": 30.0}],
        "episodes": [
            {
                "id": "ep-hero",
                "role": "hero",
                "source_id": "src",
                "t_start": f1["t"] - 0.5,
                "t_end": f1["t"] + 1.5,
                "frame": {"path": f1["file"], "t": f1["t"], "score": 0.87,
                          "score_parts": {"sharpness": 0.4, "exposure": 0.47}},
                "alternatives": [
                    {"path": frames[0]["file"], "t": frames[0]["t"], "score": 0.71},
                    {"path": frames[2]["file"], "t": frames[2]["t"], "score": 0.65},
                ],
            },
            {
                "id": "ep-step-1",
                "role": "step",
                "source_id": "src",
                "t_start": f6["t"] - 0.5,
                "t_end": f6["t"] + 1.5,
                "frame": {"path": f6["file"], "t": f6["t"], "score": 0.8},
            },
        ],
    }


def _labels_file(tmp_path: Path) -> Path:
    """Ground-truth labels: hero lands exactly on frame 1 (3.2 s), steps on
    frame 0 (0.5 s, within tolerance), ingredients miss entirely."""
    frames = json.loads((tmp_path / "runs/demo-001/frames.json").read_text(encoding="utf-8"))["frames"]
    t1 = frames[1]["t"]
    labels = {
        "schema_version": "1.0",
        "video_id": "demo-001",
        "tags": ["ru", "voice"],
        "hero": [{"t": t1, "note": "готовое блюдо"}],
        "ingredients": [{"t": 999.0}],
        "steps": [{"t": 1.0}],
    }
    path = tmp_path / "labels_demo.json"
    path.write_text(json.dumps(labels), encoding="utf-8")
    assert validate_document(labels, "eval_labels") == []
    return path


# --- data building -------------------------------------------------------------


def test_build_data_matches_frames(run_dir: Path):
    data = build_gallery_data(run_dir)
    src = json.loads((run_dir / "frames.json").read_text(encoding="utf-8"))["frames"]
    assert data["frames_total"] == len(src)
    for got, want in zip(data["frames"], src, strict=True):
        assert got["t"] == want["t"]
        assert got["file"] == want["file"]
        assert got["in_sheet"] is True
    assert data["frames_shown"] == len(src)
    assert data["selection"] is None and data["labels"] is None


def test_every_flag_densifies_the_sheet(run_dir: Path):
    frames = json.loads((run_dir / "frames.json").read_text(encoding="utf-8"))["frames"]
    data = build_gallery_data(run_dir, every_s=5.0)
    flags = in_sheet_flags(frames, 5.0)
    assert [f["in_sheet"] for f in data["frames"]] == flags
    assert data["frames_shown"] == sum(flags)
    # greedy: 0.5 taken, next allowed >= 5.5 -> 5.9 (i=2), then >= 10.9 -> 11.3
    assert [f["t"] for f in data["frames"] if f["in_sheet"]] == [0.5, 5.9, 11.3, 16.7, 22.1]


def test_labels_marked_within_tolerance(run_dir: Path, tmp_path: Path):
    labels_path = _labels_file(tmp_path)
    data = build_gallery_data(run_dir, labels=str(labels_path))
    frames = data["frames"]
    # hero sits exactly on frame 1; steps@1.0 s only reaches frame 0 (0.5 s);
    # ingredients@999 s reaches nothing
    assert [h["role"] for h in frames[1]["label_hits"]] == ["hero"]
    assert frames[1]["label_hits"][0]["note"] == "готовое блюдо"
    assert [h["role"] for h in frames[0]["label_hits"]] == ["steps"]
    assert data["labels"]["frames_marked"] == 2
    assert data["labels"]["video_id"] == "demo-001"
    for i, frame in enumerate(frames):
        if i not in (0, 1):
            assert "label_hits" not in frame


def test_match_labels_unit():
    frames = [{"t": 1.0}, {"t": 3.5}, {"t": 100.0}]
    labels = {"hero": [{"t": 3.0}], "steps": [{"t": 99.0, "note": "x"}]}
    matched = match_labels(frames, labels, tolerance_s=1.0)
    assert matched[0] == []
    assert [m["role"] for m in matched[1]] == ["hero"]
    assert [m["role"] for m in matched[2]] == ["steps"]
    assert matched[2][0]["note"] == "x"


# --- selection view ------------------------------------------------------------


def test_selection_view_and_sheet_highlight(run_dir: Path, tmp_path: Path):
    sb = _storyboard(run_dir)
    sb_path = tmp_path / "storyboard.json"
    sb_path.write_text(json.dumps(sb), encoding="utf-8")
    assert validate_document(sb, "storyboard") == []

    data = build_gallery_data(run_dir, storyboard=str(sb_path))
    assert data["selection"] is not None
    assert data["selection"]["roles"] == ["hero", "step"]
    episodes = data["selection"]["episodes"]
    assert [ep["id"] for ep in episodes] == ["ep-hero", "ep-step-1"]
    assert episodes[0]["frame"]["score"] == 0.87
    assert len(episodes[0]["alternatives"]) == 2

    html = render_html(data)
    dom = _parse(html)
    # default tab is the label sheet; selection frames highlighted by role
    cards = _cards(dom)
    assert len(cards) == data["frames_total"]
    selected = [attrs for _t, attrs in cards if attrs.get("data-role")]
    assert {selected[0]["data-file"], selected[1]["data-file"]} == {
        "frames/000001.jpg",
        "frames/000006.jpg",
    }
    assert selected[0]["data-role"] == "hero"
    # the selection tab is enabled (no boolean 'disabled' attribute)
    tab = [attrs for _t, attrs in dom.elements if attrs.get("id") == "tab-select"]
    assert tab and "disabled" not in tab[0]
    groups = [attrs for tag, attrs in dom.elements if tag == "div" and "select-group" in (attrs.get("class") or "")]
    assert [g["data-role"] for g in groups] == ["hero", "step"]
    eps = [attrs for tag, attrs in dom.elements if tag == "article" and attrs.get("data-episode")]
    assert [e["data-episode"] for e in eps] == ["ep-hero", "ep-step-1"]


def test_default_storyboard_from_run_dir(run_dir: Path):
    (run_dir / "storyboard.json").write_text(json.dumps(_storyboard(run_dir)), encoding="utf-8")
    data = build_gallery_data(run_dir)
    assert data["selection"] is not None
    assert data["selection"]["storyboard_path"] == "storyboard.json"


def test_missing_storyboard_disables_select_tab(run_dir: Path):
    html = render_html(build_gallery_data(run_dir))
    dom = _parse(html)
    tab = [attrs for _tag, attrs in dom.elements if attrs.get("id") == "tab-select"]
    assert tab and "disabled" in tab[0]
    assert "storyboard.json" in html  # the user is told what is missing


def test_explicit_storyboard_must_exist(run_dir: Path):
    with pytest.raises(GalleryError, match="storyboard not found"):
        build_gallery_data(run_dir, storyboard="nope.json")


def test_invalid_default_storyboard_warns_but_sheet_works(run_dir: Path):
    (run_dir / "storyboard.json").write_text('{"broken": true}', encoding="utf-8")
    data = build_gallery_data(run_dir)
    assert data["selection"] is None
    assert data["warnings"] and "storyboard" in data["warnings"][0]
    assert data["frames_total"] == 10


# --- html page properties ------------------------------------------------------


def test_page_has_no_external_references(run_dir: Path):
    html = render_html(build_gallery_data(run_dir))
    assert not re.search(r'(?:src|href)="https?://', html)
    assert not re.search(r"url\(https?://", html)
    assert "@import" not in html
    assert "cdn" not in html.lower()


def test_page_embeds_lazy_images_with_relative_paths(run_dir: Path):
    html = render_html(build_gallery_data(run_dir))
    dom = _parse(html)
    imgs = [attrs for tag, attrs in dom.elements if tag == "img"]
    assert imgs, "thumbnails expected"
    for attrs in imgs:
        src = attrs.get("src") or ""
        assert src.startswith("frames/"), f"thumbnail path must be relative: {src}"
        assert attrs.get("loading") == "lazy"


def test_frame_count_and_timecodes_match_frames_json(run_dir: Path):
    html = render_html(build_gallery_data(run_dir))
    data = _data_block(html)
    frames = json.loads((run_dir / "frames.json").read_text(encoding="utf-8"))["frames"]
    assert data["frames_total"] == len(frames)
    assert [f["t"] for f in data["frames"]] == [f["t"] for f in frames]
    assert [f["file"] for f in data["frames"]] == [f["file"] for f in frames]
    # and the DOM grid itself carries the same timecodes
    dom = _parse(html)
    dom_ts = [attrs["data-t"] for _t, attrs in _cards(dom)]
    assert len(dom_ts) == len(frames)
    assert [f"{f['t']:.3f}" for f in frames] == dom_ts


# --- run_gallery (file output) + CLI --------------------------------------------


def test_run_gallery_writes_default_file(run_dir: Path, tmp_path: Path, capsys):
    labels_path = _labels_file(tmp_path)
    out = run_gallery(run_dir, labels=str(labels_path))
    assert out == run_dir / "gallery.html"
    assert out.is_file()
    html = out.read_text(encoding="utf-8")
    assert 'id="gallery-data"' in html
    data = _data_block(html)
    assert data["labels"]["frames_marked"] == 2


def test_run_gallery_out_and_missing_run(tmp_path: Path):
    with pytest.raises(GalleryError, match="run directory not found"):
        run_gallery(tmp_path / "nope")
    run = tmp_path / "runs" / "x"
    run.mkdir(parents=True)
    (run / "frames.json").write_text("{}", encoding="utf-8")
    with pytest.raises(GalleryError, match="no frames array"):
        run_gallery(run, out=tmp_path / "g.html")


def test_cli_gallery(run_dir: Path, tmp_path: Path):
    sb_path = tmp_path / "storyboard.json"
    sb_path.write_text(json.dumps(_storyboard(run_dir)), encoding="utf-8")
    labels_path = _labels_file(tmp_path)
    rc = main(
        [
            "gallery",
            str(run_dir),
            "--storyboard",
            str(sb_path),
            "--labels",
            str(labels_path),
            "--every",
            "4",
        ]
    )
    assert rc == EXIT_OK
    html = (run_dir / "gallery.html").read_text(encoding="utf-8")
    data = _data_block(html)
    assert data["frames_shown"] < data["frames_total"]
    assert data["selection"] is not None and data["labels"] is not None


def test_cli_gallery_errors(run_dir: Path, capsys):
    rc = main(["gallery", str(run_dir / "no-such")])
    assert rc == EXIT_FILE_ERROR
    assert "razkadr gallery" in capsys.readouterr().err
    empty = run_dir.parent / "empty"
    empty.mkdir()
    rc = main(["gallery", str(empty)])
    assert rc == EXIT_FILE_ERROR
    assert "frames.json not found" in capsys.readouterr().err


def test_cli_gallery_help():
    from razkadr.cli import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["gallery", "--help"])
    assert exc.value.code == 0

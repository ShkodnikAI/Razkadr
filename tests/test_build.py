"""End-to-end build tests (naryad 028): one command, the whole package.

Runs the real CLI (``razkadr build``) on the naryad 034 synthetic fixtures
with the fake providers of the naryad 033 layer: the package (deck.pptx,
deck.pdf, deck.html, storyboard.json) lands in the run directory, every
artifact is schema-valid, the run.json summary matches the artifacts, a
resumed build re-renders nothing (cached layout/render/html/pdf stages),
--lang overrides the output language end to end, and a machine without
LibreOffice gets a package without the PDF plus a warning.
"""

import contextlib
import io
import json
import shutil
from pathlib import Path

import pytest

from razkadr.cli import EXIT_FILE_ERROR, EXIT_OK, main
from razkadr.validation import validate_document

REPO = Path(__file__).resolve().parent.parent
PRESET = REPO / "presets" / "recipe.yaml"
PACKAGE = ("deck.pptx", "deck.pdf", "deck.html", "storyboard.json")


def _write_toml(tmp_path: Path) -> Path:
    """Full pipeline config: fake providers for every ML task (naryad 033)."""
    lines = [
        f'runs_dir = "{tmp_path / "runs"}"',
        "",
        "[cache]",
        f'dir = "{tmp_path / "calls"}"',
        "",
        "[asr]",
        'provider = "fake"',
        'model = "fake-nova-2"',
        "",
        "[vlm]",
        'provider = "fake"',
        'model = "fake-vision-1"',
        "",
        "[llm]",
        'provider = "fake"',
        'model = "fake-gpt-4o-mini"',
        "",
        "[embeddings]",
        'provider = "fake"',
        'model = "fake-embed-v2"',
        "dim = 8",
        "",
    ]
    path = tmp_path / "razkadr.toml"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _run_cli(argv: list[str]) -> tuple[int, str, str]:
    """main() with stdout/stderr captured (module fixtures have no capsys)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


def _single_run_dir(tmp_path: Path) -> Path:
    runs = [d for d in sorted((tmp_path / "runs").iterdir()) if d.is_dir()]
    assert len(runs) == 1, f"expected one run directory, got {[r.name for r in runs]}"
    return runs[0]


def _load(run_dir: Path, name: str) -> dict:
    return json.loads((run_dir / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def video_with_audio(synthetic_fixtures, tmp_path_factory):
    """scenes.mp4 (video only) muxed with a sine track: video + audio fixture."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not found in PATH")
    dst = tmp_path_factory.mktemp("build_with_audio") / "with_audio.mp4"
    import subprocess

    subprocess.run(
        ["ffmpeg", "-y", "-i", str(synthetic_fixtures.path("scenes.mp4")),
         "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=30",
         "-shortest", "-c:v", "copy", "-c:a", "aac", "-b:a", "96k", str(dst)],
        check=True, capture_output=True,
    )
    return dst


# --- the full package --------------------------------------------------------------


def test_build_full_package_in_run_dir(video_with_audio, tmp_path):
    """One command: video -> deck.pptx + deck.pdf + deck.html + storyboard.json."""
    config_path = _write_toml(tmp_path)
    code, out, err = _run_cli([
        "build", str(video_with_audio), "--config", str(config_path), "--preset", str(PRESET),
    ])
    assert code == EXIT_OK, err
    run_dir = _single_run_dir(tmp_path)
    for name in PACKAGE:
        assert (run_dir / name).is_file(), f"missing {name} in the package"
    assert (run_dir / "deck_assets").is_dir(), "the HTML deck carries its assets"

    storyboard = _load(run_dir, "storyboard.json")
    assert validate_document(storyboard, "storyboard") == []
    assert storyboard["episodes"], "the fixture must select at least one episode"
    slides = _load(run_dir, "slides.json")
    assert validate_document(slides, "slides") == []
    assert len(slides["slides"]) >= 1

    run_json = _load(run_dir, "run.json")
    stages = {name: status for name, status in ((s["name"], s["status"]) for s in run_json["stages"])}
    for name in ("layout", "render", "html", "pdf"):
        assert stages.get(name) == "ok", f"{name} must run as a stage"
    summary = run_json["summary"]
    assert summary["artifacts"]["episodes"] == len(storyboard["episodes"])
    assert summary["artifacts"]["slides"] == len(slides["slides"])
    assert summary["artifacts"]["package_files"] == 4

    # the console names the package and the per-artifact lines
    assert "package:" in out and "pptx" in out and "json" in out
    assert "episodes" in out


def test_build_preset_by_name_from_repo_root(video_with_audio, tmp_path, monkeypatch):
    """--preset recipe (a bare name) resolves to presets/recipe.yaml from CWD."""
    config_path = _write_toml(tmp_path)
    monkeypatch.chdir(REPO)
    code, _out, err = _run_cli([
        "build", str(video_with_audio), "--config", str(config_path), "--preset", "recipe",
        "--runs-dir", str(tmp_path / "runs"),
    ])
    assert code == EXIT_OK, err
    run_dir = _single_run_dir(tmp_path)
    storyboard = _load(run_dir, "storyboard.json")
    assert storyboard["project"]["preset"] == "recipe"


def test_build_lang_en_overrides_end_to_end(video_with_audio, tmp_path):
    """--lang en: storyboard output_lang and slide captions follow it;
    described (silence) episodes come out in en; a condensed episode whose
    translation loses numbers falls back to the source language - the
    naryad 018 number invariant, exercised live through the fake LLM."""
    config_path = _write_toml(tmp_path)
    code, _out, err = _run_cli([
        "build", str(video_with_audio), "--config", str(config_path), "--preset", str(PRESET),
        "--lang", "en",
    ])
    assert code == EXIT_OK, err
    run_dir = _single_run_dir(tmp_path)
    storyboard = _load(run_dir, "storyboard.json")
    assert storyboard["project"]["output_lang"] == "en"
    slides = _load(run_dir, "slides.json")
    assert slides["lang"] == "en", "captions must follow --lang"
    texts = [ep["text"] for ep in storyboard["episodes"] if ep.get("text")]
    assert texts, "the fixture episodes must get text"
    described = [t for t in texts if t.get("mode") == "described"]
    assert described, "silence episodes must be described"
    assert all(t.get("output_lang") == "en" for t in described), (
        "described episodes must come out in the requested language"
    )
    fallback = [t for t in texts if t.get("mode") == "verbatim" and t.get("output_lang") == "ru"]
    assert fallback, (
        "a translation that loses numbers must fall back to the source language"
    )


# --- resume / cache ------------------------------------------------------------------


def test_build_resume_renders_nothing_and_adds_no_cost(video_with_audio, tmp_path):
    """--resume: layout/render/html/pdf are cached, the deck files are untouched,
    and the provider cost stays where the first run left it."""
    config_path = _write_toml(tmp_path)
    argv = ["build", str(video_with_audio), "--config", str(config_path), "--preset", str(PRESET)]
    assert _run_cli(argv)[0] == EXIT_OK
    run_dir = _single_run_dir(tmp_path)
    first = _load(run_dir, "run.json")
    mtimes = {
        name: (run_dir / name).stat().st_mtime_ns
        for name in ("slides.json", "deck.pptx", "deck.html", "deck.pdf")
    }

    code, out, err = _run_cli([*argv, "--resume", run_dir.name])
    assert code == EXIT_OK, err
    run_json = _load(run_dir, "run.json")
    statuses = {s["name"]: s["status"] for s in run_json["stages"][-6:]}
    for name in ("layout", "render", "html", "pdf"):
        assert statuses[name] == "cached", f"{name} must be a cache hit on resume"
    for name, mtime in mtimes.items():
        assert (run_dir / name).stat().st_mtime_ns == mtime, f"{name} was rewritten on resume"
    assert run_json["summary"]["provider_cost"]["amount"] == (
        first["summary"]["provider_cost"]["amount"]
    ), "resume must not add provider cost (call-cache hits bill nothing)"
    assert "cached" in out


# --- package placement and degradation ----------------------------------------------


def test_build_out_dir_moves_the_deck_files(video_with_audio, tmp_path):
    """--out-dir: deck files + the json copy land there, run dir keeps the intermediates."""
    config_path = _write_toml(tmp_path)
    out_dir = tmp_path / "package"
    code, _out, err = _run_cli([
        "build", str(video_with_audio), "--config", str(config_path), "--preset", str(PRESET),
        "--out-dir", str(out_dir),
    ])
    assert code == EXIT_OK, err
    for name in PACKAGE:
        assert (out_dir / name).is_file(), f"missing {name} in --out-dir"
    run_dir = _single_run_dir(tmp_path)
    assert (run_dir / "storyboard.json").is_file(), "the run keeps its own storyboard"
    assert (run_dir / "slides.json").is_file()
    assert (out_dir / "storyboard.json").read_text(encoding="utf-8") == (
        run_dir / "storyboard.json").read_text(encoding="utf-8"
    ), "the package json is the validated Storyboard"


def test_build_without_soffice_degrades_with_warning(video_with_audio, tmp_path, monkeypatch):
    """No LibreOffice: the package ships without the PDF, warning to stderr, exit 0."""
    real_which = shutil.which
    monkeypatch.setattr(
        shutil, "which", lambda name: None if name == "soffice" else real_which(name)
    )
    config_path = _write_toml(tmp_path)
    code, out, err = _run_cli([
        "build", str(video_with_audio), "--config", str(config_path), "--preset", str(PRESET),
    ])
    assert code == EXIT_OK, err
    run_dir = _single_run_dir(tmp_path)
    assert (run_dir / "deck.pptx").is_file()
    assert (run_dir / "deck.html").is_file()
    assert (run_dir / "storyboard.json").is_file()
    assert not (run_dir / "deck.pdf").is_file(), "no soffice - no PDF in the package"
    assert "pdf skipped" in err
    assert "- (skipped)" in out
    run_json = _load(run_dir, "run.json")
    assert all(s["name"] != "pdf" for s in run_json["stages"]), "a skipped export keeps no stage entry"


# --- failures ------------------------------------------------------------------------


def test_build_missing_video_fails_cleanly(tmp_path):
    """A missing input fails with the command prefix before any stage runs
    (ingest validates the file first, so there is no failed stage to resume)."""
    config_path = _write_toml(tmp_path)
    code, _out, err = _run_cli([
        "build", str(tmp_path / "no-such-video.mp4"), "--config", str(config_path),
        "--preset", str(PRESET),
    ])
    assert code == EXIT_FILE_ERROR
    assert err.startswith("razkadr build: input file not found")


def test_build_unknown_preset_fails_before_stages(tmp_path):
    config_path = _write_toml(tmp_path)
    for value in ("no-such-preset", str(tmp_path / "no-such.yaml")):
        code, _out, err = _run_cli([
            "build", str(tmp_path / "whatever.mp4"), "--config", str(config_path),
            "--preset", value,
        ])
        assert code == EXIT_FILE_ERROR, value
        assert "preset not found" in err, value
    assert not (tmp_path / "runs").exists(), "no run directory without a preset"

"""Ingest stage tests (naryad 005): file and URL branches, limits, clear errors.

All tests are offline: the URL branch replaces/monkeys the yt-dlp runner
or its subprocess. Media work needs ffmpeg/ffprobe (skipped when absent).
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from razkadr.cli import EXIT_FILE_ERROR, EXIT_OK, main
from razkadr.config import load_config
from razkadr.runctx import RunContext
from razkadr.stages import ingest as ingest_mod
from razkadr.stages.ingest import (
    IngestError,
    InputNotFoundError,
    LimitExceededError,
    MediaInvalidError,
    SourceUnavailableError,
    _run_ytdlp,
    ingest_params,
    run_ingest,
)
from razkadr.validation import validate_document

pytestmark = pytest.mark.ffmpeg


@pytest.fixture()
def ffmpeg_bin():
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not found in PATH")
    return True


@pytest.fixture()
def env(tmp_path, ffmpeg_bin):
    """RunContext with default config; runs isolated in tmp_path."""
    config = load_config(None)
    config["runs_dir"] = str(tmp_path / "runs")
    ctx = RunContext(config["runs_dir"], config=config)
    return tmp_path, config, ctx


@pytest.fixture(scope="session")
def video_with_audio(synthetic_fixtures, tmp_path_factory):
    """scenes.mp4 (video only) muxed with a sine track: video + audio fixture."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not found in PATH")
    dst = tmp_path_factory.mktemp("with_audio") / "with_audio.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(synthetic_fixtures.path("scenes.mp4")),
         "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=30",
         "-shortest", "-c:v", "copy", "-c:a", "aac", "-b:a", "96k", str(dst)],
        check=True, capture_output=True,
    )
    return dst


def _probe(path: Path) -> dict:
    return json.loads(
        subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(path)],
            capture_output=True, text=True, check=True,
        ).stdout
    )


# --- contract example: file input, all three artifacts ------------------------

def test_contract_example_file_input(env, video_with_audio):
    _tmp_path, config, ctx = env
    data = run_ingest(ctx, str(video_with_audio), config)

    assert validate_document(data, "source") == []
    assert data["path"] == "normalized.mp4"

    normalized = ctx.dir / "normalized.mp4"
    audio = ctx.dir / "audio.wav"
    assert normalized.is_file() and audio.is_file()
    from razkadr.runctx import file_sha256

    assert data["sha256"] == file_sha256(normalized)
    assert abs(data["duration_s"] - 30.0) < 1.5
    assert data["width"] == 320 and data["height"] == 240
    assert data["fps"] <= 10.0 + 1e-6
    types = {s["type"]: s["codec"] for s in data["streams"]}
    assert types["video"] == "mpeg4"
    assert types["audio"] == "aac"

    wav = _probe(audio)
    astream = next(s for s in wav["streams"] if s["codec_type"] == "audio")
    assert astream["codec_name"] == "pcm_s16le"
    assert astream["sample_rate"] == "16000" and astream["channels"] == 1

    run = json.loads((ctx.dir / "run.json").read_text(encoding="utf-8"))
    assert validate_document(run, "run") == []
    assert run["stages"][-1]["name"] == "ingest" and run["stages"][-1]["status"] == "ok"
    assert run["tool_versions"]["ffmpeg"]
    assert "ffmpeg" in run["tool_versions"]["ffmpeg"].lower()


# --- no audio track: flag, not an error ---------------------------------------

def test_no_audio_video_processed_without_failure(env, synthetic_fixtures):
    _tmp_path, config, ctx = env
    data = run_ingest(ctx, str(synthetic_fixtures.path("scenes.mp4")), config)
    assert validate_document(data, "source") == []
    assert not (ctx.dir / "audio.wav").is_file()
    assert all(s["type"] != "audio" for s in data["streams"])


# --- URL branch: yt-dlp mocked, no network ------------------------------------

def test_url_branch_success_with_mocked_ytdlp(env, video_with_audio, monkeypatch):
    _tmp_path, config, ctx = env
    url = "https://www.youtube.com/watch?v=fake"

    def fake_download(url_, target):
        shutil.copyfile(video_with_audio, target)

    monkeypatch.setattr(ingest_mod, "_run_ytdlp", fake_download)
    data = run_ingest(ctx, url, config)
    assert validate_document(data, "source") == []
    assert data["url"] == url
    assert (ctx.dir / "original.mp4").is_file()


def test_url_branch_unavailable_clear_error(env, monkeypatch):
    _tmp_path, _config, ctx = env

    def fail_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="ERROR: Video unavailable")

    monkeypatch.setattr(ingest_mod.subprocess, "run", fail_run)
    with pytest.raises(SourceUnavailableError, match="video is unavailable"):
        _run_ytdlp("https://www.youtube.com/watch?v=gone", ctx.dir / "original.mp4")


def test_url_branch_private_and_network_errors(env, monkeypatch):
    _tmp_path, _config, ctx = env

    def private_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="ERROR: Private video. Sign in")

    monkeypatch.setattr(ingest_mod.subprocess, "run", private_run)
    with pytest.raises(SourceUnavailableError, match="video is private"):
        _run_ytdlp("https://v.example/1", ctx.dir / "x.mp4")

    def net_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="ERROR: unable to download video data")

    monkeypatch.setattr(ingest_mod.subprocess, "run", net_run)
    with pytest.raises(SourceUnavailableError, match="download failed"):
        _run_ytdlp("https://v.example/2", ctx.dir / "x.mp4")


def test_ytdlp_command_shape(monkeypatch, tmp_path):
    """python -m yt_dlp, fixed output template, no playlist; target must exist."""
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"video")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(ingest_mod.subprocess, "run", fake_run)
    target = tmp_path / "original.mp4"
    _run_ytdlp("https://v.example/3", target)
    cmd = captured["cmd"]
    assert cmd[:3] == [sys.executable, "-m", "yt_dlp"]
    assert "--no-playlist" in cmd and "-o" in cmd and cmd[-1] == "https://v.example/3"
    assert target.is_file()


def test_url_without_ytdlp_installed_clear_error(env, monkeypatch):
    _tmp_path, config, ctx = env

    def no_pkg(name):
        raise ingest_mod.PackageNotFoundError(name)

    monkeypatch.setattr(ingest_mod.package_version, "__call__", no_pkg, raising=False)
    monkeypatch.setattr(ingest_mod, "package_version", no_pkg)
    with pytest.raises(IngestError, match="yt-dlp is not installed"):
        run_ingest(ctx, "https://v.example/4", config)


# --- limits --------------------------------------------------------------------

def test_duration_limit(env, synthetic_fixtures):
    _tmp_path, config, ctx = env
    config["ingest"] = {"max_duration_s": 5}
    with pytest.raises(LimitExceededError, match=r"duration .* exceeds the limit 5 s"):
        run_ingest(ctx, str(synthetic_fixtures.path("scenes.mp4")), config)
    assert not (ctx.dir / "normalized.mp4").is_file()


def test_size_limit(env, synthetic_fixtures):
    _tmp_path, config, ctx = env
    config["ingest"] = {"max_file_size_mb": 0.000001}  # ~1 byte
    with pytest.raises(LimitExceededError, match="file size"):
        run_ingest(ctx, str(synthetic_fixtures.path("scenes.mp4")), config)


def test_limits_surface_in_run_json(env, synthetic_fixtures):
    _tmp_path, config, ctx = env
    config["ingest"] = {"max_duration_s": 5}
    with pytest.raises(LimitExceededError):
        run_ingest(ctx, str(synthetic_fixtures.path("scenes.mp4")), config)
    run = json.loads((ctx.dir / "run.json").read_text(encoding="utf-8"))
    assert run["stages"][-1]["status"] == "failed"
    assert "exceeds the limit" in run["stages"][-1]["error"]


# --- invalid inputs --------------------------------------------------------------

def test_corrupted_file_clear_error(env, tmp_path):
    _tmp_path, config, ctx = env
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"this is not a video file" * 10)
    with pytest.raises(MediaInvalidError, match="corrupted or not a media file"):
        run_ingest(ctx, str(broken), config)


def test_missing_file_clear_error(env, tmp_path):
    _tmp_path, config, ctx = env
    with pytest.raises(InputNotFoundError, match="input file not found"):
        run_ingest(ctx, str(tmp_path / "nope.mp4"), config)


# --- auditor's edge cases --------------------------------------------------------

def test_spaces_in_filename(env, synthetic_fixtures, tmp_path):
    _tmp_path, config, ctx = env
    spaced = tmp_path / "my recipe video.mp4"
    shutil.copyfile(synthetic_fixtures.path("scenes.mp4"), spaced)
    data = run_ingest(ctx, str(spaced), config)
    assert validate_document(data, "source") == []


def test_very_short_video(env, tmp_path):
    _tmp_path, config, ctx = env
    tiny = tmp_path / "tiny.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=red:s=64x64:d=0.5:r=10",
         "-c:v", "mpeg4", "-pix_fmt", "yuv420p", str(tiny)],
        check=True, capture_output=True,
    )
    data = run_ingest(ctx, str(tiny), config)
    assert 0 < data["duration_s"] <= 1.0


def test_normalization_caps_resolution_and_fps(env, synthetic_fixtures):
    _tmp_path, config, ctx = env
    config["ingest"] = {"fps": 5, "max_height": 120}
    data = run_ingest(ctx, str(synthetic_fixtures.path("scenes.mp4")), config)
    assert data["height"] == 120
    assert data["width"] == 160  # 320x240 is 4:3 -> 160x120, aspect preserved
    assert data["fps"] <= 5.0 + 1e-6


# --- cache / resume --------------------------------------------------------------

def test_resume_second_run_is_cached(env, video_with_audio):
    _tmp_path, config, ctx = env
    first = run_ingest(ctx, str(video_with_audio), config)

    normalized = ctx.dir / "normalized.mp4"
    stamp = normalized.stat().st_mtime_ns

    ctx2 = RunContext(ctx.runs_dir, run_id=ctx.run_id, config=config)
    second = run_ingest(ctx2, str(video_with_audio), config)
    assert second == first
    assert normalized.stat().st_mtime_ns == stamp, "cached run must not rewrite artifacts"

    run = json.loads((ctx.dir / "run.json").read_text(encoding="utf-8"))
    assert run["stages"][-1]["status"] == "cached"


def test_changed_config_invalidates_stage_cache(env, video_with_audio):
    _tmp_path, config, ctx = env
    run_ingest(ctx, str(video_with_audio), config)
    config["ingest"] = {"fps": 7}
    ctx2 = RunContext(ctx.runs_dir, run_id=ctx.run_id, config=config)
    run_ingest(ctx2, str(video_with_audio), config)
    assert (ctx.dir / "normalized.mp4").is_file()
    run = json.loads((ctx.dir / "run.json").read_text(encoding="utf-8"))
    assert run["stages"][-1]["status"] == "ok"  # params changed -> re-ran


def test_ingest_params_defaults_and_overrides():
    defaults = ingest_params({})
    assert defaults == {
        "fps": 10.0, "max_height": 1080, "max_duration_s": 3600.0, "max_file_size_mb": 2048.0,
    }
    partial = ingest_params({"ingest": {"fps": 6}})
    assert partial["fps"] == 6.0 and partial["max_height"] == 1080


# --- CLI -------------------------------------------------------------------------

def test_cli_index_end_to_end(synthetic_fixtures, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    code = main(["index", "--runs-dir", str(tmp_path / "runs"), str(synthetic_fixtures.path("scenes.mp4"))])
    assert code == EXIT_OK
    out = capsys.readouterr().out
    assert "ingest ok" in out
    run_dirs = list((tmp_path / "runs").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "source.json").is_file()
    assert not (run_dir / "audio.wav").is_file()  # scenes.mp4 has no audio
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert validate_document(run, "run") == []


def test_cli_index_missing_file_exit_2(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    code = main(["index", "--runs-dir", str(tmp_path / "runs"), str(tmp_path / "nope.mp4")])
    assert code == EXIT_FILE_ERROR
    err = capsys.readouterr().err
    assert "razkadr index:" in err and "not found" in err

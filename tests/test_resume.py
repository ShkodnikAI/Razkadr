"""Crash-safe resume tests (naryad 029): обрыв посреди стадии -> повторный
запуск продолжает с места сбоя.

Three crash shapes are covered:
1. a real SIGKILL in the middle of a running stage (the ocr stage of the
   fake pipeline runs 3 s - a reliable kill window): the killed stage
   leaves no cache marker and no partial JSON (atomic writes), the
   resume re-runs exactly that stage and completes the run;
2. a kill modeled after the fact - a half-written frames.json with the
   stage marker deleted (what a mid-write kill leaves behind): the
   resume recomputes the stage and overwrites the truncated file;
3. a corrupt artifact with a fresh cache marker (disk damage / manual
   edit, not a crash): a clean error naming both remedies, no traceback.
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from razkadr.cli import EXIT_FILE_ERROR, EXIT_OK, main
from razkadr.validation import validate_document

REPO = Path(__file__).resolve().parent.parent
FULL_KINDS = {
    "source.json": "source",
    "frames.json": "frames",
    "transcript.json": "transcript",
    "chunks.json": "chunks",
    "ocr.json": "ocr",
    "index/manifest.json": "index",
    "run.json": "run",
}


def _write_toml(tmp_path: Path) -> Path:
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


def _run_json(run_dir: Path) -> dict:
    return json.loads((run_dir / "run.json").read_text(encoding="utf-8"))


def _stage_names(run_dir: Path) -> list[str]:
    return [s["name"] for s in _run_json(run_dir).get("stages", [])]


@pytest.fixture(scope="module")
def video_with_audio(synthetic_fixtures, tmp_path_factory):
    """scenes.mp4 (video only) muxed with a sine track: video + audio fixture."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not found in PATH")
    dst = tmp_path_factory.mktemp("resume_with_audio") / "with_audio.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(synthetic_fixtures.path("scenes.mp4")),
         "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=30",
         "-shortest", "-c:v", "copy", "-c:a", "aac", "-b:a", "96k", str(dst)],
        check=True, capture_output=True,
    )
    return dst


def _single_run_dir(tmp_path: Path) -> Path:
    runs = [d for d in sorted((tmp_path / "runs").iterdir()) if d.is_dir()]
    assert len(runs) == 1, f"expected one run directory, got {[r.name for r in runs]}"
    return runs[0]


# --- 1: the literal contract - SIGKILL in the middle of a stage ----------------------


def test_sigkill_mid_stage_resume_completes(video_with_audio, tmp_path):
    config_path = _write_toml(tmp_path)
    runs_dir = tmp_path / "runs"
    env = {**os.environ, "PYTHONPATH": str(REPO / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}
    proc = subprocess.Popen(
        [sys.executable, "-m", "razkadr.cli", "index", str(video_with_audio),
         "--config", str(config_path)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env,
    )
    # the fake ocr stage takes ~3 s: wait until ingest..chunks are done, then kill
    deadline = time.monotonic() + 120
    killed = False
    run_dir = None
    while time.monotonic() < deadline:
        run_files = sorted(runs_dir.glob("*/run.json")) if runs_dir.is_dir() else []
        if run_files:
            try:
                names = _stage_names(run_files[0].parent)
            except json.JSONDecodeError:
                names = []
            if names == ["ingest", "frames", "transcript", "chunks"]:
                proc.send_signal(signal.SIGKILL)
                killed = True
                run_dir = run_files[0].parent
                break
        if proc.poll() is not None:
            break  # finished before the window (should not happen with the fake ocr)
        time.sleep(0.05)
    assert killed, "the kill window (during the ocr stage) was missed"
    proc.wait()

    # the crash left: completed stages ok, no failed entry; ocr.json is
    # either absent or complete (atomic writes forbid a partial one)
    names = _stage_names(run_dir)
    assert names == ["ingest", "frames", "transcript", "chunks"]
    statuses = [s["status"] for s in _run_json(run_dir)["stages"]]
    assert statuses == ["ok"] * 4, "a SIGKILL leaves no failed stage entry"
    if (run_dir / "ocr.json").exists():
        json.loads((run_dir / "ocr.json").read_text(encoding="utf-8"))
    assert "summary" not in _run_json(run_dir), "a crashed run has no final summary"

    # the resume continues from the crash point and completes the run
    code = main(["index", str(video_with_audio), "--config", str(config_path),
                 "--resume", run_dir.name])
    assert code == EXIT_OK
    run_json = _run_json(run_dir)
    statuses = [s["status"] for s in run_json["stages"]]
    assert statuses[:4] == ["ok"] * 4, "completed stages stay untouched"
    assert statuses[4:10] == ["cached"] * 4 + ["ok"] * 2, (
        "ingest..chunks are cache hits, ocr and index re-run"
    )
    assert "summary" in run_json
    for name, kind in FULL_KINDS.items():
        data = json.loads((run_dir / name).read_text(encoding="utf-8"))
        assert validate_document(data, kind) == [], f"invalid {name} after resume"


# --- 2: kill mid-write, modeled: truncated artifact, no marker -----------------------


def test_truncated_artifact_without_marker_recomputes(video_with_audio, tmp_path):
    config_path = _write_toml(tmp_path)
    assert main(["index", str(video_with_audio), "--config", str(config_path)]) == EXIT_OK
    run_dir = _single_run_dir(tmp_path)

    # what a kill during the frames write leaves: half a file, no marker yet
    frames_path = run_dir / "frames.json"
    raw = frames_path.read_text(encoding="utf-8")
    frames_path.write_text(raw[: len(raw) // 2], encoding="utf-8")
    (run_dir / ".cache" / "frames.json").unlink()

    code = main(["index", str(video_with_audio), "--config", str(config_path),
                 "--resume", run_dir.name])
    assert code == EXIT_OK
    statuses = [s["status"] for s in _run_json(run_dir)["stages"]]
    assert statuses[:6] == ["ok"] * 6, "the first full run"
    assert statuses[6] == "cached" and statuses[8] == "cached" and statuses[9] == "cached", (
        "ingest and transcript/chunks (unchanged inputs) are cache hits"
    )
    assert statuses[7] == "ok", "frames re-ran: no marker, truncated output"
    assert statuses[8:12] == ["cached"] * 4, (
        "transcript/chunks/ocr/index are cache hits: the recomputed frames.json "
        "is byte-identical (the pipeline is deterministic)"
    )
    frames = json.loads(frames_path.read_text(encoding="utf-8"))
    assert validate_document(frames, "frames") == [], "the truncated file is replaced whole"


# --- 3: corrupt artifact with a fresh marker = a clean error, not a traceback --------


def test_corrupt_artifact_with_marker_is_a_clean_error(video_with_audio, tmp_path, capsys):
    config_path = _write_toml(tmp_path)
    assert main(["index", str(video_with_audio), "--config", str(config_path)]) == EXIT_OK
    run_dir = _single_run_dir(tmp_path)

    frames_path = run_dir / "frames.json"
    raw = frames_path.read_text(encoding="utf-8")
    frames_path.write_text(raw[: len(raw) // 2], encoding="utf-8")
    # the marker stays: the cache believes frames.json is current

    code = main(["index", str(video_with_audio), "--config", str(config_path),
                 "--resume", run_dir.name])
    assert code == EXIT_FILE_ERROR
    err = capsys.readouterr().err
    assert "frames.json is not valid JSON" in err
    assert "--force" in err, "the message must name the remedies"
    assert "Traceback" not in err, "a clean error, never a stack trace"

    # remedy 2: delete the file, resume recomputes the stage
    frames_path.unlink()
    (run_dir / ".cache" / "frames.json").unlink()
    assert main(["index", str(video_with_audio), "--config", str(config_path),
                 "--resume", run_dir.name]) == EXIT_OK

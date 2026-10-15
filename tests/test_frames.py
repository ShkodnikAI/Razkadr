"""Frames stage tests (naryad 006): scene detection, dedup, quality marks, determinism.

All tests run on the synthetic fixtures (naryad 034) or additional
ffmpeg-generated synthetic cases; no network, no external APIs. Media
work needs ffmpeg/ffprobe (skipped when absent).

Measured reference points (fixture scenes.mp4, ffmpeg 7.1.5):
- ContentDetector frame scores at cuts 21.45-34.05, inside-scene noise <= 0.60
- normalized sharpness: blurry scene 0.206-0.212, sharp scenes 0.909-0.977
- pHash distance: within one scene 0, between neighboring scenes >= 8
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from razkadr.cli import EXIT_FILE_ERROR, EXIT_OK, main
from razkadr.config import load_config
from razkadr.runctx import RunContext
from razkadr.stages.frames import (
    FRAMES_DEFAULTS,
    FramesError,
    _candidates,
    detect_scenes,
    frames_params,
    run_frames,
)
from razkadr.stages.ingest import run_ingest
from razkadr.validation import validate_document


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


@pytest.fixture()
def ingested(env, synthetic_fixtures):
    """env with the ingest stage already run on scenes.mp4."""
    _tmp_path, config, ctx = env
    run_ingest(ctx, str(synthetic_fixtures.path("scenes.mp4")), config)
    return config, ctx


def _read_frames_json(ctx: RunContext) -> dict:
    return json.loads((ctx.dir / "frames.json").read_text(encoding="utf-8"))


def _accepted(doc: dict) -> list[dict]:
    return [f for f in doc["frames"] if not f.get("rejected")]


# --- contract: all known cuts detected with +-0.5 s tolerance ------------------

def test_all_cuts_detected_within_tolerance(ingested, synthetic_fixtures):
    _config, ctx = ingested
    gt = synthetic_fixtures.ground_truth["scenes"]
    scenes = detect_scenes(ctx.dir / "normalized.mp4", 10.0, frames_params({}))
    cuts = [start for start, _end in scenes[1:]]
    expected = gt["cuts_s"]
    assert len(cuts) == len(expected), f"expected {len(expected)} cuts, detected {len(cuts)}"
    for got, want in zip(cuts, expected, strict=True):
        assert abs(got - want) <= 0.5, f"cut at {got:.2f} s, expected {want:.2f} s (+-0.5)"


def test_contract_frames_artifact(ingested):
    _config, ctx = ingested
    doc = run_frames(ctx, _config)

    assert validate_document(doc, "frames") == []
    assert doc["schema_version"] == "1.1"
    frames = doc["frames"]
    assert 15 <= len(frames) <= 19  # measured 17: 5 scene mids + 12 uniform points
    assert len(_accepted(doc)) == 4  # measured: one frame per scene except the blurry one
    assert [f["t"] for f in frames] == sorted(f["t"] for f in frames)
    scene_ids = [f["scene_id"] for f in frames]
    assert scene_ids == sorted(scene_ids)
    for f in frames:
        assert (ctx.dir / f["file"]).is_file(), f"frame file missing: {f['file']}"
        assert f["file"].startswith("frames/") and f["file"].endswith(".jpg")
        assert 0.0 <= f["sharpness"] <= 1.0
        assert 0.0 <= f["exposure"] <= 1.0
        assert len(f["phash"]) == 16 and all(c in "0123456789abcdef" for c in f["phash"])

    run = json.loads((ctx.dir / "run.json").read_text(encoding="utf-8"))
    assert validate_document(run, "run") == []
    assert run["stages"][-1]["name"] == "frames" and run["stages"][-1]["status"] == "ok"
    assert run["tool_versions"]["scenedetect"]


# --- blurry fragment: below threshold and marked --------------------------------

def test_blur_fragment_marked(ingested, synthetic_fixtures):
    _config, ctx = ingested
    doc = run_frames(ctx, _config)
    blur_start, blur_end = synthetic_fixtures.ground_truth["scenes"]["blur_interval_s"]
    threshold = FRAMES_DEFAULTS["sharpness_threshold"]

    inside = [f for f in doc["frames"] if blur_start + 0.5 < f["t"] < blur_end - 0.5]
    assert inside, "expected candidates inside the blurred interval"
    for f in inside:
        assert f["sharpness"] < threshold, f"blurry frame {f['t']} s above threshold"
        assert f.get("rejected") is True and f["reject_reason"] == "blurry"

    for f in _accepted(doc):
        assert f["sharpness"] >= threshold


# --- near-duplicate fragment: collapsed to one frame ----------------------------

def test_duplicate_fragment_collapsed(ingested, synthetic_fixtures):
    _config, ctx = ingested
    doc = run_frames(ctx, _config)
    dup_scene = synthetic_fixtures.ground_truth["scenes"]["duplicate_pair"][0]

    accepted_in_dup = [f for f in _accepted(doc) if f["scene_id"] == dup_scene]
    assert len(accepted_in_dup) == 1, "near-duplicate fragment must collapse to one frame"

    rejected_in_dup = [f for f in doc["frames"] if f["scene_id"] == dup_scene and f.get("rejected")]
    for f in rejected_in_dup:
        assert f["reject_reason"].startswith("duplicate of ")


# --- rejected frames are kept, with a reason (auditor criterion) -----------------

def test_rejected_frames_not_lost(ingested):
    _config, ctx = ingested
    doc = run_frames(ctx, _config)
    rejected = [f for f in doc["frames"] if f.get("rejected")]
    assert rejected, "fixture must produce rejected candidates"
    for f in rejected:
        assert f["reject_reason"]
        assert (ctx.dir / f["file"]).is_file(), "rejected frame file must stay on disk"


# --- determinism: re-run produces byte-identical frames.json ---------------------

def test_rerun_identical_frames_json(ingested):
    _config, ctx = ingested
    doc = run_frames(ctx, _config)
    first = (ctx.dir / "frames.json").read_bytes()

    rerun_ctx = RunContext(ctx.runs_dir, run_id=ctx.run_id, config=_config, force=True)
    rerun_doc = run_frames(rerun_ctx, _config)
    assert rerun_doc == doc
    assert (ctx.dir / "frames.json").read_bytes() == first


# --- stage cache and parameter changes -------------------------------------------

def test_frames_cached_on_resume(ingested):
    _config, ctx = ingested
    run_frames(ctx, _config)
    resumed = RunContext(ctx.runs_dir, run_id=ctx.run_id, config=_config)
    doc = run_frames(resumed, _config)
    assert validate_document(doc, "frames") == []
    run = json.loads((ctx.dir / "run.json").read_text(encoding="utf-8"))
    frames_statuses = [s["status"] for s in run["stages"] if s["name"] == "frames"]
    assert frames_statuses == ["ok", "cached"]


def test_param_change_reruns_frames(ingested):
    config, ctx = ingested
    run_frames(ctx, config)
    changed = dict(config)
    changed["frames"] = {"step_s": 1.0}
    resumed = RunContext(ctx.runs_dir, run_id=ctx.run_id, config=changed)
    run_frames(resumed, changed)
    run = json.loads((ctx.dir / "run.json").read_text(encoding="utf-8"))
    frames_statuses = [s["status"] for s in run["stages"] if s["name"] == "frames"]
    assert frames_statuses == ["ok", "ok"]


# --- clear error without ingest output -------------------------------------------

def test_missing_source_json_is_a_clear_error(env):
    _tmp_path, config, ctx = env
    with pytest.raises(FramesError, match="source.json is missing"):
        run_frames(ctx, config)


# --- auditor cases: long static scene and frequent cuts ---------------------------

def _ffmpeg_concat_video(out: Path, colors: list[str], part_s: float, fps: int = 10,
                         box_xy: list[tuple[int, int]] | None = None) -> Path:
    """Solid-color video with a white box (edges for sharpness); hard cuts between parts.

    box_xy: per-part box positions; different positions change frame structure,
    so neighboring scenes differ in pHash the way real content differs.
    """
    if box_xy is None:
        box_xy = [(32, 32)] * len(colors)
    cmd = ["ffmpeg", "-y"]
    for i, color in enumerate(colors):
        cmd += ["-f", "lavfi", "-i", f"color=c={color}:s=320x240:d={part_s:g}:r={fps}"]
    graph = ";".join(
        f"[{i}:v]drawbox=x={x}:y={y}:w=100:h=80:color=white:t=fill,format=yuv420p,setsar=1[v{i}]"
        for i, (x, y) in enumerate(box_xy)
    )
    graph += ";" + "".join(f"[v{i}]" for i in range(len(colors))) + f"concat=n={len(colors)}:v=1:a=0[out]"
    cmd += ["-filter_complex", graph, "-map", "[out]", "-c:v", "mpeg4", "-q:v", "2", str(out)]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


def test_long_static_scene(env):
    _tmp_path, config, ctx = env
    src = _ffmpeg_concat_video(ctx.dir / "static.mp4", ["#306090"], 20.0)
    run_ingest(ctx, str(src), config)
    doc = run_frames(ctx, config)

    assert validate_document(doc, "frames") == []
    assert {f["scene_id"] for f in doc["frames"]} == {0}, "static video is one scene"
    accepted = len(_accepted(doc))
    assert 1 <= accepted <= 3, f"static scene should dedup to ~1 frame, got {accepted}"


def test_frequent_cuts(env):
    _tmp_path, config, ctx = env
    colors = ["#CC4444", "#44CC44", "#4444CC", "#CCCC44", "#CC44CC",
              "#44CCCC", "#C0C0C0", "#404040", "#F0A000", "#00A0F0"]
    # each scene gets its own box position: structure changes at every cut
    box_xy = [(20 + 28 * i, 30 + 40 * (i % 3)) for i in range(len(colors))]
    src = _ffmpeg_concat_video(ctx.dir / "fast.mp4", colors, 1.0, box_xy=box_xy)
    run_ingest(ctx, str(src), config)
    doc = run_frames(ctx, config)

    assert validate_document(doc, "frames") == []
    n_scenes = len({f["scene_id"] for f in doc["frames"]})
    assert n_scenes >= 9, f"expected >= 9 scenes for cuts every 1 s, got {n_scenes}"
    assert len(_accepted(doc)) >= 9, "every 1-second scene must keep at least one frame"


# --- uniform sampling grid ---------------------------------------------------------

def test_uniform_grid_covers_duration():
    scenes = [(0.0, 30.0)]
    params = {**FRAMES_DEFAULTS, "step_s": 2.5}
    cands = _candidates(scenes, 30.0, params)
    times = sorted(t for t, _s, kind in cands if kind == 1)
    assert times[0] == 0.0
    assert times[-1] == 27.5  # 30.0 itself is excluded (beyond the last frame)
    assert len(times) == 12
    mids = [t for t, _s, kind in cands if kind == 0]
    assert mids == [15.0]


# --- CLI: index runs ingest + frames ------------------------------------------------

def test_cli_index_runs_frames(env, synthetic_fixtures, capsys):
    _tmp_path, _config, _ctx = env
    code = main(["index", str(synthetic_fixtures.path("scenes.mp4")),
                 "--runs-dir", str(_tmp_path / "runs")])
    assert code == EXIT_OK
    out = capsys.readouterr().out
    assert "ingest ok, frames ok" in out
    assert "accepted" in out
    # the env fixture pre-creates an empty run dir; take the run the CLI reports
    run_id = out.split("run ")[1].split(":")[0]
    run_dir = _tmp_path / "runs" / run_id
    assert (run_dir / "frames.json").is_file()
    doc = json.loads((run_dir / "frames.json").read_text(encoding="utf-8"))
    assert validate_document(doc, "frames") == []


def test_cli_frames_failure_exit_2(env, synthetic_fixtures, capsys, monkeypatch):
    _tmp_path, _config, _ctx = env
    import razkadr.stages.frames as frames_mod

    def boom(_ctx, _config):
        raise FramesError("boom")

    monkeypatch.setattr(frames_mod, "run_frames", boom)
    code = main(["index", str(synthetic_fixtures.path("scenes.mp4")),
                 "--runs-dir", str(_tmp_path / "runs")])
    assert code == EXIT_FILE_ERROR
    assert "boom" in capsys.readouterr().err

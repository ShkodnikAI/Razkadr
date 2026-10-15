"""End-to-end pipeline tests (naryad 011): one command, a full index.

Runs the real CLI (``razkadr index``) on the naryad 034 synthetic
fixtures with the fake providers of the naryad 033 layer: every artifact
is created and schema-valid, the run.json summary matches the artifacts
it aggregates, a resumed run is fully cached with zero new provider
calls, and an injected transcript failure resumes from that stage
without recomputing the earlier ones.
"""

import contextlib
import io
import json
from pathlib import Path

import pytest

from razkadr.cli import EXIT_FILE_ERROR, EXIT_OK, main
from razkadr.validation import validate_document

ARTIFACT_KINDS = {
    "source.json": "source",
    "frames.json": "frames",
    "transcript.json": "transcript",
    "chunks.json": "chunks",
    "ocr.json": "ocr",
    "index/manifest.json": "index",
    "run.json": "run",
}
STAGES = ("ingest", "frames", "transcript", "chunks", "ocr", "index")


def load_json(run_dir: Path, name: str) -> dict:
    return json.loads((run_dir / name).read_text(encoding="utf-8"))


def _write_toml(tmp_path: Path, asr_extra: str = "") -> Path:
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
    ]
    if asr_extra:
        lines.append(asr_extra)
    lines += [
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


@pytest.fixture(scope="module")
def video_with_audio(synthetic_fixtures, tmp_path_factory):
    """scenes.mp4 (video only) muxed with a sine track: video + audio fixture."""
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not found in PATH")
    dst = tmp_path_factory.mktemp("pipeline_with_audio") / "with_audio.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(synthetic_fixtures.path("scenes.mp4")),
         "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=30",
         "-shortest", "-c:v", "copy", "-c:a", "aac", "-b:a", "96k", str(dst)],
        check=True, capture_output=True,
    )
    return dst


def test_full_run_all_artifacts_valid_and_summary_matches(video_with_audio, tmp_path):
    """AC1: one command creates every artifact, all valid; summary = the artifacts."""
    config_path = _write_toml(tmp_path)
    code, out, err = _run_cli(["index", str(video_with_audio), "--config", str(config_path)])
    assert code == EXIT_OK, err
    run_dir = _single_run_dir(tmp_path)

    for name, kind in ARTIFACT_KINDS.items():
        assert (run_dir / name).is_file(), f"missing artifact {name}"
        data = load_json(run_dir, name)
        assert validate_document(data, kind) == [], f"invalid {name}"

    # the summary in run.json must match the artifacts it aggregates (auditor check)
    run_json = load_json(run_dir, "run.json")
    summary = run_json["summary"]
    assert set(summary["stage_times_s"]) == set(STAGES)
    assert all(t >= 0 for t in summary["stage_times_s"].values())

    cost = summary["provider_cost"]
    assert cost["calls"] == len(run_json["provider_calls"]) > 0
    assert cost["amount"] == round(sum(c["cost"]["amount"] for c in run_json["provider_calls"]), 6)
    assert cost["currency"] == "USD"
    assert cost["cache_hits"] == sum(1 for c in run_json["provider_calls"] if c["cache_hit"])

    frames_doc = load_json(run_dir, "frames.json")
    accepted = [f for f in frames_doc["frames"] if not f.get("rejected")]
    artifacts = summary["artifacts"]
    assert artifacts["frames_candidates"] == len(frames_doc["frames"])
    assert artifacts["frames_accepted"] == len(accepted)
    assert artifacts["frames_rejected"] == len(frames_doc["frames"]) - len(accepted)
    assert artifacts["ocr_results"] == len(load_json(run_dir, "ocr.json")["results"])
    transcript_doc = load_json(run_dir, "transcript.json")
    assert artifacts["transcript_segments"] == len(transcript_doc["segments"])
    assert artifacts["transcript_s"] == round(
        sum(s["t_end"] - s["t_start"] for s in transcript_doc["segments"]), 3
    )
    assert artifacts["chunks"] == len(load_json(run_dir, "chunks.json")["chunks"])
    manifest = load_json(run_dir, "index/manifest.json")
    assert artifacts["index_vectors"] == manifest["num_vectors"]

    # the console prints the contract summary: frames, transcript length, ocr, cost, time
    assert "frames:" in out and "ocr:" in out and "transcript:" in out
    assert "cost:" in out and "USD" in out
    assert "time:" in out and "ingest" in out and "index" in out


def test_rerun_all_cached_zero_provider_calls(video_with_audio, tmp_path):
    """AC2: --resume re-run is fully cached, adds no provider calls, artifacts untouched."""
    config_path = _write_toml(tmp_path)
    assert _run_cli(["index", str(video_with_audio), "--config", str(config_path)])[0] == EXIT_OK
    run_dir = _single_run_dir(tmp_path)
    run_json = load_json(run_dir, "run.json")
    calls_before = len(run_json["provider_calls"])
    assert calls_before > 0
    mtimes = {
        name: (run_dir / name).stat().st_mtime_ns
        for name in ("source.json", "frames.json", "transcript.json", "chunks.json", "ocr.json")
    }

    code, _out, err = _run_cli(
        ["index", str(video_with_audio), "--config", str(config_path), "--resume", run_dir.name]
    )
    assert code == EXIT_OK, err
    run_json = load_json(run_dir, "run.json")
    statuses = [s["status"] for s in run_json["stages"]]
    assert statuses[:6] == ["ok"] * 6
    assert statuses[6:] == ["cached"] * 6, "every stage must be a cache hit on resume"
    assert len(run_json["provider_calls"]) == calls_before, "resume must make zero provider calls"
    for name, mtime in mtimes.items():
        assert (run_dir / name).stat().st_mtime_ns == mtime, f"{name} was rewritten"
    assert run_json["summary"]["provider_cost"]["calls"] == calls_before

    # the resumed run still ends with a valid, consistent summary
    assert validate_document(run_json, "run") == []


def test_transcript_failure_resume_continues(video_with_audio, tmp_path, monkeypatch):
    """AC3: an ASR outage fails the transcript stage with a resume hint;
    --resume re-runs only that stage, earlier ones stay cached."""
    from razkadr.providers.base import TransientError
    from razkadr.providers.fakes import FakeASR
    from razkadr.providers.layer import FACTORIES

    config_path = _write_toml(tmp_path, asr_extra="base_delay_s = 0")  # retries instant
    state = {"fail": True}

    class ExplodingASR:
        name = "fake"
        model = "fake-nova-2"

        def transcribe(self, audio_path, lang=None):
            raise TransientError("injected ASR outage")

    def flaky_factory(section):
        if state["fail"]:
            return ExplodingASR()
        return FakeASR(model=section.get("model", "fake-nova-2"))

    monkeypatch.setitem(FACTORIES["asr"], "fake", flaky_factory)

    code, _out, err = _run_cli(["index", str(video_with_audio), "--config", str(config_path)])
    assert code == EXIT_FILE_ERROR
    assert "injected ASR outage" in err
    assert "--resume" in err, "the failure must print the resume hint"
    run_dir = _single_run_dir(tmp_path)
    run_json = load_json(run_dir, "run.json")
    assert [(s["name"], s["status"]) for s in run_json["stages"]] == [
        ("ingest", "ok"), ("frames", "ok"), ("transcript", "failed"),
    ]
    assert "injected ASR outage" in run_json["stages"][-1]["error"]
    assert "summary" not in run_json, "a failed run gets no final summary"
    assert not (run_dir / ".cache" / "transcript.json").is_file(), "failed stage leaves no marker"
    frames_marker = (run_dir / ".cache" / "frames.json").read_text(encoding="utf-8")

    state["fail"] = False
    code, _out, err = _run_cli(
        ["index", str(video_with_audio), "--config", str(config_path), "--resume", run_dir.name]
    )
    assert code == EXIT_OK, err
    run_json = load_json(run_dir, "run.json")
    assert [(s["name"], s["status"]) for s in run_json["stages"]] == [
        ("ingest", "ok"), ("frames", "ok"), ("transcript", "failed"),
        ("ingest", "cached"), ("frames", "cached"), ("transcript", "ok"),
        ("chunks", "ok"), ("ocr", "ok"), ("index", "ok"),
    ], "earlier stages must not recompute; the failed stage re-runs"
    assert (run_dir / ".cache" / "frames.json").read_text(encoding="utf-8") == frames_marker, (
        "the frames cache marker is untouched: the stage did not recompute"
    )
    assert len(run_json["provider_calls"]) > 0
    assert validate_document(run_json, "run") == []
    assert run_json["summary"]["artifacts"]["transcript_segments"] == len(
        load_json(run_dir, "transcript.json")["segments"]
    )


def test_resume_unknown_run_id_fails_clearly(tmp_path):
    """A typo in --resume must not silently start a new run under that id."""
    config_path = _write_toml(tmp_path)
    code, _out, err = _run_cli(
        ["index", "whatever.mp4", "--config", str(config_path), "--resume", "no-such-run"]
    )
    assert code == EXIT_FILE_ERROR
    assert "nothing to resume" in err
    assert not (tmp_path / "runs" / "no-such-run").exists()

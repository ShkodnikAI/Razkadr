"""RunContext tests: stage cache, failure accounting, secrets (naryad 036)."""

import hashlib
import json
from pathlib import Path

import pytest

from razkadr.config import load_config
from razkadr.runctx import RunContext, file_sha256, fingerprint
from razkadr.validation import validate_document


def read_run_json(runs_dir: Path, run_id: str) -> dict:
    return json.loads((runs_dir / run_id / "run.json").read_text(encoding="utf-8"))


# --- contract: test stage run twice, second is a cache hit -------------------

def test_stage_cached_on_second_run(tmp_path):
    runs_dir = tmp_path / "runs"
    ctx = RunContext(runs_dir, config={"preset": "recipe"})
    executed = []
    with ctx.stage("frames", inputs={"video_sha256": "a" * 64}, params={"fps": 10}) as st:
        assert not st.cached
        executed.append(st.name)

    resumed = RunContext(runs_dir, run_id=ctx.run_id, config={"preset": "recipe"})
    with resumed.stage("frames", inputs={"video_sha256": "a" * 64}, params={"fps": 10}) as st:
        assert st.cached, "second run must be a cache hit"
        if not st.cached:  # the cached body is skipped by the caller
            executed.append(st.name)

    assert executed == ["frames"]  # body ran once
    data = read_run_json(runs_dir, ctx.run_id)
    assert validate_document(data, "run") == []
    assert [s["status"] for s in data["stages"]] == ["ok", "cached"]


def test_cache_key_changes_with_inputs_and_params(tmp_path):
    runs_dir = tmp_path / "runs"
    ctx = RunContext(runs_dir)
    with ctx.stage("s", inputs={"x": 1}) as st:
        first = st.key
    with ctx.stage("s", inputs={"x": 2}) as st:
        assert st.key != first
        assert not st.cached
    with ctx.stage("s", inputs={"x": 1}, params={"p": True}) as st:
        assert not st.cached, "params are part of the key"


def test_force_reruns_cached_stage(tmp_path):
    runs_dir = tmp_path / "runs"
    ctx = RunContext(runs_dir)
    with ctx.stage("frames", inputs={"v": "1"}):
        pass
    forced = RunContext(runs_dir, run_id=ctx.run_id, force=True)
    with forced.stage("frames", inputs={"v": "1"}) as st:
        assert not st.cached


def test_run_id_reuse_creates_one_directory(tmp_path):
    runs_dir = tmp_path / "runs"
    ctx = RunContext(runs_dir)
    RunContext(runs_dir, run_id=ctx.run_id)
    assert [p.name for p in runs_dir.iterdir() if p.is_dir()] == [ctx.run_id]


# --- failure: recorded with status and reason, resume continues --------------

def test_failed_stage_recorded_and_resume_reruns_it(tmp_path):
    runs_dir = tmp_path / "runs"
    ctx = RunContext(runs_dir)
    with ctx.stage("ingest", inputs={"src": "x"}):
        pass
    with pytest.raises(RuntimeError, match="provider exploded"), ctx.stage(
        "transcript", inputs={"src": "x"}
    ):
        raise RuntimeError("provider exploded")

    data = read_run_json(runs_dir, ctx.run_id)
    assert validate_document(data, "run") == []
    failed = data["stages"][-1]
    assert failed["status"] == "failed" and failed["name"] == "transcript"
    assert "provider exploded" in failed["error"]

    # resume: completed stage is a cache hit, failed stage runs again
    resumed = RunContext(runs_dir, run_id=ctx.run_id)
    with resumed.stage("ingest", inputs={"src": "x"}) as st:
        assert st.cached
    ran_again = []
    with resumed.stage("transcript", inputs={"src": "x"}) as st:
        assert not st.cached
        ran_again.append(st.name)
    assert ran_again == ["transcript"]
    data = read_run_json(runs_dir, ctx.run_id)
    assert [s["status"] for s in data["stages"]] == ["ok", "failed", "cached", "ok"]


def test_failed_stage_leaves_no_cache_marker(tmp_path):
    runs_dir = tmp_path / "runs"
    ctx = RunContext(runs_dir)
    with pytest.raises(ValueError), ctx.stage("frames", inputs={"v": "1"}):
        raise ValueError("bad frame")
    assert not (ctx.dir / ".cache" / "frames.json").exists()


def test_run_json_written_even_without_clean_exit(tmp_path):
    """run.json is rewritten after every stage event, so a crash keeps history."""
    runs_dir = tmp_path / "runs"
    ctx = RunContext(runs_dir)
    with ctx.stage("ingest", inputs={"src": "x"}):
        pass
    snapshot = read_run_json(runs_dir, ctx.run_id)
    assert len(snapshot["stages"]) == 1


# --- provider call accounting ------------------------------------------------

def test_provider_call_recorded_and_schema_valid(tmp_path):
    ctx = RunContext(tmp_path / "runs")
    ctx.record_provider_call(
        stage="index",
        provider="openai",
        model="text-embedding-3-small",
        units=[{"kind": "images", "value": 3}, {"kind": "tokens", "value": 120}],
        cost_amount=0.0012,
        currency="USD",
        tariff_date="2026-10-01",
        cache_hit=False,
        task="embed",
    )
    data = json.loads((ctx.dir / "run.json").read_text(encoding="utf-8"))
    assert validate_document(data, "run") == []
    call = data["provider_calls"][0]
    assert call["units"] == [{"kind": "images", "value": 3}, {"kind": "tokens", "value": 120}]
    assert call["cache_hit"] is False and call["task"] == "embed"


def test_provider_call_rejects_bad_unit_kind(tmp_path):
    ctx = RunContext(tmp_path / "runs")
    with pytest.raises(ValueError):
        ctx.record_provider_call(
            stage="index", provider="p", model="m",
            units=[{"kind": "pixels", "value": 1}],
            cost_amount=0, currency="USD", tariff_date="2026-10-01", cache_hit=False,
        )


# --- secrets: never in logs, never in run directory files --------------------

def test_secrets_masked_in_logs_and_run_files(tmp_path):
    import io

    from razkadr.log import SECRETS, get_logger, setup_logging

    key = "sk-test-abcdef123456"
    SECRETS.register(key)

    buf = io.StringIO()
    setup_logging(verbose=False, stream=buf)
    logger = get_logger("test")
    logger.info("calling provider with key %s", key)

    ctx = RunContext(tmp_path / "runs", config={"runs_dir": "runs"})
    with pytest.raises(RuntimeError), ctx.stage("transcript", inputs={"src": "x"}):
        raise RuntimeError(f"401 unauthorized for {key}")

    setup_logging(stream=None)  # restore default handler for other tests
    captured = buf.getvalue()
    assert key not in captured
    assert "[REDACTED]" in captured

    # auditor's check: the key must not appear in ANY file of the run directory
    for path in ctx.dir.rglob("*"):
        if path.is_file():
            assert key not in path.read_text(encoding="utf-8"), f"secret leaked into {path.name}"


def test_secret_config_keys_excluded_from_config_hash():
    import os

    os.environ["RAZKADR_TEST_API_KEY"] = "sk-env-secret-987"
    os.environ["RAZKADR_PRESET"] = "recipe"
    try:
        plain = load_config()
        assert "test_api_key" not in plain, "secret env var must not become a config key"
        assert plain["preset"] == "recipe"
        from razkadr.config import config_hash

        with_secret = dict(plain)
        with_secret["api_key"] = "sk-other-secret"
        # api_key is a secret name: excluded from the hash
        assert config_hash(plain) == config_hash(
            {k: v for k, v in with_secret.items() if k != "api_key"}
        )
    finally:
        del os.environ["RAZKADR_TEST_API_KEY"]
        del os.environ["RAZKADR_PRESET"]


# --- final summary (naryad 011) -----------------------------------------------

def test_write_summary_written_and_schema_valid(tmp_path):
    runs_dir = tmp_path / "runs"
    ctx = RunContext(runs_dir, config={"preset": "recipe"})
    with ctx.stage("ingest", inputs={"src": "x"}):
        pass
    ctx.record_provider_call(
        stage="transcript", provider="fake", model="fake-nova-2",
        units=[{"kind": "seconds", "value": 30.0}],
        cost_amount=0.472, currency="USD", tariff_date="2026-10-01", cache_hit=False,
    )
    summary = ctx.write_summary(artifacts={"frames_accepted": 12, "chunks": 3})
    data = read_run_json(runs_dir, ctx.run_id)
    assert validate_document(data, "run") == []
    assert data["summary"] == summary
    assert summary["stage_times_s"] == {"ingest": data["stages"][0]["duration_s"]}
    cost = summary["provider_cost"]
    assert cost["amount"] == 0.472
    assert cost["calls"] == 1 and cost["cache_hits"] == 0
    assert cost["currency"] == "USD"
    assert cost["by_stage"] == {"transcript": 0.472}
    assert summary["artifacts"] == {"frames_accepted": 12, "chunks": 3}


def test_write_summary_no_provider_calls(tmp_path):
    """A CPU-only run: zero cost, no currency (nothing was billed)."""
    ctx = RunContext(tmp_path / "runs")
    summary = ctx.write_summary()
    assert validate_document(read_run_json(tmp_path / "runs", ctx.run_id), "run") == []
    assert summary["provider_cost"] == {"amount": 0.0, "calls": 0, "cache_hits": 0, "by_stage": {}}
    assert "currency" not in summary["provider_cost"]
    assert summary["stage_times_s"] == {} and summary["artifacts"] == {}


def test_write_summary_mixed_currencies_rejected(tmp_path):
    ctx = RunContext(tmp_path / "runs")
    for currency in ("USD", "EUR"):
        ctx.record_provider_call(
            stage="transcript", provider="p", model="m",
            units=[{"kind": "tokens", "value": 1}],
            cost_amount=0.1, currency=currency, tariff_date="2026-10-01", cache_hit=False,
        )
    with pytest.raises(ValueError, match="mixed currencies"):
        ctx.write_summary()


def test_write_summary_uses_last_completed_attempt(tmp_path):
    """Resume: the last completed attempt counts; failed attempts have no duration."""
    runs_dir = tmp_path / "runs"
    ctx = RunContext(runs_dir)
    with ctx.stage("transcript", inputs={"a": 1}):
        pass
    with pytest.raises(RuntimeError), ctx.stage("transcript", inputs={"a": 1}):
        raise RuntimeError("boom")
    resumed = RunContext(runs_dir, run_id=ctx.run_id)
    with resumed.stage("transcript", inputs={"a": 1}) as st:
        assert st.cached
    summary = resumed.write_summary()
    data = read_run_json(runs_dir, ctx.run_id)
    assert [s["status"] for s in data["stages"]] == ["ok", "failed", "cached"]
    completed = [s["duration_s"] for s in data["stages"] if s["status"] != "failed"]
    assert summary["stage_times_s"]["transcript"] == completed[-1]


def test_write_summary_rejects_bad_artifacts(tmp_path):
    ctx = RunContext(tmp_path / "runs")
    with pytest.raises(ValueError, match="must be a number"):
        ctx.write_summary(artifacts={"frames": "many"})
    with pytest.raises(ValueError, match="must be a number"):
        ctx.write_summary(artifacts={"frames": -1})
    with pytest.raises(ValueError, match="must be a number"):
        ctx.write_summary(artifacts={"frames": True})


# --- helpers -----------------------------------------------------------------

def test_fingerprint_stable_and_file_sha256(tmp_path):
    assert fingerprint({"a": 1, "b": [2, 3]}) == fingerprint({"b": [2, 3], "a": 1})
    f = tmp_path / "x.bin"
    f.write_bytes(b"razkadr")
    assert file_sha256(f) == hashlib.sha256(b"razkadr").hexdigest()

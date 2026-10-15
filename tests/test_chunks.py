"""Chunks stage tests (naryad 008): coverage, boundaries, LLM answers, fallback.

All tests run on hand-written transcript artifacts through the fake
providers of the naryad 033 layer; no network, no external APIs. The
LLM never writes chunk text: answers are boundary indexes only, so
"chunks do not cut an ASR segment mid-way" and "chunk text equals
transcript text" hold by construction - the tests pin both invariants
plus the auditor's no-loss check (concatenation of chunks equals the
transcript).
"""

import json
import logging
import shutil
from pathlib import Path

import pytest

from razkadr.cli import EXIT_OK, main
from razkadr.config import load_config
from razkadr.providers.base import AuthError, UsageReporter
from razkadr.providers.layer import FACTORIES
from razkadr.providers.types import Message
from razkadr.runctx import RunContext
from razkadr.stages.chunks import (
    CHUNKS_JSON,
    ChunksError,
    build_chunks,
    chunks_params,
    fallback_boundaries,
    parse_boundaries,
    run_chunks,
    system_prompt,
)
from razkadr.validation import validate_document


class ScriptLLM(UsageReporter):
    """LLM fake with a scripted answer; counts calls, can raise auth errors."""

    name = "script"
    model = "script-1"
    task = "llm"

    def __init__(self, answer: str, auth_error: str | None = None) -> None:
        self.answer = answer
        self.auth_error = auth_error
        self.calls = 0
        self.last_usage = []
        self.last_messages: list[Message] = []

    def complete(self, messages: list[Message]) -> str:
        self.calls += 1
        self.last_messages = list(messages)
        if self.auth_error:
            raise AuthError(self.auth_error)
        self._report([{"kind": "tokens", "value": float(sum(len(m.content.split()) for m in messages) + 8)}])
        return self.answer


def seg(t_start: float, t_end: float, text: str, lang: str = "ru") -> dict:
    return {"t_start": t_start, "t_end": t_end, "text": text, "lang": lang}


def write_transcript(ctx: RunContext, segments: list[dict], lang: str = "ru") -> dict:
    data = {"schema_version": "1.0", "lang": lang, "segments": segments}
    (ctx.dir / "transcript.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return data


@pytest.fixture()
def env(tmp_path):
    """RunContext + config with isolated runs and call cache dirs."""
    config = load_config(None)
    config["runs_dir"] = str(tmp_path / "runs")
    config["cache"] = {"dir": str(tmp_path / "calls")}
    ctx = RunContext(config["runs_dir"], config=config)
    return tmp_path, config, ctx


@pytest.fixture()
def fake_llm_calls(monkeypatch):
    """Replace the fake LLM factory, capturing every created instance.

    The scripted answer comes from the [llm] answer key (test-only
    section key; the factory reads it, the stage never does).
    """
    created: list[ScriptLLM] = []

    def factory(section: dict) -> ScriptLLM:
        provider = ScriptLLM(
            str(section.get("answer", '{"boundaries": [0]}')),
            auth_error=section.get("auth_error"),
        )
        created.append(provider)
        return provider

    monkeypatch.setitem(FACTORIES["llm"], "fake", factory)
    return created


def _read_json(ctx: RunContext, name: str) -> dict:
    return json.loads((ctx.dir / name).read_text(encoding="utf-8"))


def _warning_reasons(caplog) -> list[str]:
    """The stage logs the fallback reason in extra.fields.reason (JSON logging)."""
    return [
        str(getattr(r, "fields", {}).get("reason", ""))
        for r in caplog.records
        if r.levelno == logging.WARNING
    ]


# --- unit: params -------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"method": "magic"},     # unknown method
        {"max_chars": 0},        # zero cap
        {"max_chars": -5},       # negative cap
        {"pause_s": -1.0},       # negative pause
    ],
)
def test_params_validation_errors(overrides):
    with pytest.raises(ChunksError, match="method|max_chars|pause_s"):
        chunks_params({"chunks": overrides})


def test_params_defaults():
    assert chunks_params({}) == {"method": "llm", "max_chars": 1200, "pause_s": 1.5}


# --- unit: parse_boundaries ----------------------------------------------------------


def test_parse_boundaries_valid_single_chunk():
    assert parse_boundaries('{"boundaries": [0]}', 5) == [0]


def test_parse_boundaries_valid_multi_chunk():
    assert parse_boundaries('{"boundaries": [0, 3]}', 5) == [0, 3]


@pytest.mark.parametrize(
    "raw",
    [
        "not json at all",
        "null",
        "[0, 2]",                     # not an object
        '{"chunks": [0, 2]}',         # wrong key
        '{"boundaries": []}',         # empty
        '{"boundaries": [1]}',        # does not start at 0
        '{"boundaries": [0, 1, 1]}',  # not strictly increasing
        '{"boundaries": [0, 2, 1]}',  # decreases
        '{"boundaries": [0, 9]}',     # out of range
        '{"boundaries": [0, "2"]}',   # string index
        '{"boundaries": [0, 2.0]}',   # float index
        '{"boundaries": [0, true]}',  # bool is not an index
        '{"boundaries": 0}',          # not a list
    ],
)
def test_parse_boundaries_invalid(raw):
    assert parse_boundaries(raw, 3) is None


# --- unit: fallback split (pauses and length) -----------------------------------------


def test_fallback_closes_chunk_on_pause():
    segments = [seg(0.0, 2.0, "тема один"), seg(5.0, 7.0, "тема два")]
    assert fallback_boundaries(segments, max_chars=1200, pause_s=1.5) == [0, 1]


def test_fallback_closes_chunk_on_length():
    segments = [seg(0.0, 1.0, "а" * 10), seg(1.0, 2.0, "б" * 10), seg(2.0, 3.0, "в" * 10)]
    # max_chars 25: seg0 + seg1 = 21 chars, adding seg2 would give 32 > 25
    assert fallback_boundaries(segments, max_chars=25, pause_s=1.5) == [0, 2]


def test_fallback_touching_segments_without_pause_stay_together():
    segments = [seg(0.0, 2.0, "привет"), seg(2.0, 4.0, "мир")]
    assert fallback_boundaries(segments, max_chars=1200, pause_s=1.5) == [0]


def test_fallback_single_over_long_segment_is_its_own_chunk():
    # a segment longer than max_chars cannot be split without cutting it mid-way
    segments = [seg(0.0, 1.0, "длинный " * 100), seg(1.0, 2.0, "короткий")]
    boundaries = fallback_boundaries(segments, max_chars=50, pause_s=1.5)
    chunks = build_chunks(segments, boundaries)
    assert len(chunks[0]["segment_ids"]) == 1
    assert len(chunks[0]["text"]) > 50  # kept whole, never cut


# --- unit: build_chunks (both paths share it) ------------------------------------------


def test_build_chunks_never_cuts_segment_and_matches_timecodes():
    segments = [seg(0.0, 2.0, "один"), seg(2.0, 4.0, "два"), seg(6.0, 8.0, "три")]
    chunks = build_chunks(segments, [0, 2])
    assert [c["segment_ids"] for c in chunks] == [[0, 1], [2]]
    assert chunks[0]["t_start"] == 0.0 and chunks[0]["t_end"] == 4.0
    assert chunks[1]["t_start"] == 6.0 and chunks[1]["t_end"] == 8.0
    assert chunks[0]["text"] == "один два"
    assert [c["id"] for c in chunks] == ["c1", "c2"]


def test_build_chunks_lang_most_frequent_tie_first():
    segments = [seg(0, 1, "en first", "en"), seg(1, 2, "ru"), seg(2, 3, "ru")]
    assert build_chunks(segments, [0])[0]["lang"] == "ru"  # 2 ru vs 1 en
    tie = [seg(0, 1, "en first", "en"), seg(1, 2, "ru")]
    assert build_chunks(tie, [0])[0]["lang"] == "en"  # 1-1: the first segment wins


# --- stage: fallback path (no LLM) ------------------------------------------------------


def test_fallback_stage_coverage_no_loss(env, caplog):
    _tmp_path, config, ctx = env
    config["chunks"] = {"method": "fallback"}
    segments = [
        seg(0.0, 2.0, "привет мир"),
        seg(2.0, 4.0, "готовим тесто"),
        seg(7.0, 9.0, "тема после паузы"),
        seg(9.0, 12.0, "ещё немного текста"),
    ]
    write_transcript(ctx, segments)
    with caplog.at_level(logging.WARNING, logger="razkadr.chunks"):
        doc = run_chunks(ctx, config)

    assert validate_document(doc, "chunks") == []
    ids = [i for c in doc["chunks"] for i in c["segment_ids"]]
    assert ids == list(range(len(segments)))  # partition: every segment exactly once
    assert " ".join(c["text"] for c in doc["chunks"]) == " ".join(s["text"] for s in segments)
    for c in doc["chunks"]:
        part = [segments[i] for i in c["segment_ids"]]
        assert c["t_start"] == part[0]["t_start"] and c["t_end"] == part[-1]["t_end"]
    assert "chunking fallback" in caplog.text
    assert "method = fallback" in _warning_reasons(caplog)
    run = _read_json(ctx, "run.json")
    assert run["provider_calls"] == []  # no LLM, no cost


def test_fallback_result_identical_with_and_without_llm_section(env):
    _tmp_path, config, ctx = env
    segments = [seg(0.0, 2.0, "а"), seg(5.0, 7.0, "б")]
    write_transcript(ctx, segments)

    config["chunks"] = {"method": "fallback"}
    with_llm = {**config, "llm": {"provider": "fake", "model": "script-1"}}
    doc1 = run_chunks(ctx, config)

    ctx2 = RunContext(config["runs_dir"], run_id=ctx.run_id, config=with_llm, force=True)
    doc2 = run_chunks(ctx2, with_llm)
    assert doc1 == doc2
    assert (ctx.dir / CHUNKS_JSON).read_bytes() == (ctx2.dir / CHUNKS_JSON).read_bytes()


def test_method_fallback_never_calls_configured_llm(env, fake_llm_calls):
    _tmp_path, config, ctx = env
    config["chunks"] = {"method": "fallback"}
    config["llm"] = {"provider": "fake", "model": "script-1"}
    write_transcript(ctx, [seg(0.0, 2.0, "текст")])
    run_chunks(ctx, config)
    assert fake_llm_calls == []  # the factory was never used


# --- stage: LLM path ---------------------------------------------------------------------


def test_llm_boundaries_respected_no_segment_cut(env, fake_llm_calls):
    _tmp_path, config, ctx = env
    config["llm"] = {"provider": "fake", "model": "script-1", "answer": '{"boundaries": [0, 3]}'}
    segments = [seg(float(i), float(i + 2), f"сегмент {i}") for i in range(5)]
    write_transcript(ctx, segments)

    doc = run_chunks(ctx, config)

    assert [c["segment_ids"] for c in doc["chunks"]] == [[0, 1, 2], [3, 4]]
    assert doc["chunks"][1]["t_start"] == segments[3]["t_start"]
    for c in doc["chunks"]:
        part = [segments[i] for i in c["segment_ids"]]
        assert c["text"] == " ".join(s["text"] for s in part)  # whole segments, no cuts


def test_llm_path_no_text_loss_and_valid_schema(env, fake_llm_calls):
    _tmp_path, config, ctx = env
    config["llm"] = {"provider": "fake", "model": "script-1", "answer": '{"boundaries": [0, 2]}'}
    segments = [seg(0.0, 2.0, "раз"), seg(2.0, 4.0, "два"), seg(4.0, 6.0, "три")]
    write_transcript(ctx, segments)

    doc = run_chunks(ctx, config)

    assert validate_document(doc, "chunks") == []
    assert " ".join(c["text"] for c in doc["chunks"]) == "раз два три"
    run = _read_json(ctx, "run.json")
    llm_calls = [c for c in run["provider_calls"] if c["task"] == "llm"]
    assert len(llm_calls) == 1 and llm_calls[0]["cache_hit"] is False
    assert llm_calls[0]["stage"] == "chunks"
    assert llm_calls[0]["units"] == [{"kind": "tokens", "value": llm_calls[0]["units"][0]["value"]}]


def test_llm_receives_system_prompt_and_numbered_segments(env, fake_llm_calls):
    _tmp_path, config, ctx = env
    config["llm"] = {"provider": "fake", "model": "script-1", "answer": '{"boundaries": [0]}'}
    segments = [seg(0.0, 2.0, "тема один"), seg(5.0, 7.0, "тема два")]
    write_transcript(ctx, segments)
    run_chunks(ctx, config)

    assert len(fake_llm_calls) == 1 and fake_llm_calls[0].calls == 1
    provider = fake_llm_calls[0]
    assert [m.role for m in provider.last_messages] == ["system", "user"]
    assert "boundaries" in provider.last_messages[0].content  # the versioned prompt
    assert "0 [0-2] тема один" in provider.last_messages[1].content
    assert "1 [5-7] тема два" in provider.last_messages[1].content


def test_llm_system_prompt_carries_versioned_template_and_limit(env, fake_llm_calls):
    _tmp_path, config, ctx = env
    config["llm"] = {"provider": "fake", "model": "script-1", "answer": '{"boundaries": [0]}'}
    config["chunks"] = {"max_chars": 777}
    write_transcript(ctx, [seg(0.0, 2.0, "текст")])
    run_chunks(ctx, config)
    assert "boundaries" in system_prompt(777)  # versioned prompt packaged with the library
    assert "{{max_chars}}" not in system_prompt(777) and "777" in system_prompt(777)


# --- stage: invalid LLM answers fall back with a logged reason ----------------------------


@pytest.mark.parametrize(
    "answer",
    [
        "fake completion not-json",
        '{"boundaries": [1]}',
        '{"boundaries": []}',
        '{"boundaries": [0, 2, 2]}',
        '{"boundaries": [0, 99]}',
    ],
)
def test_invalid_llm_answer_falls_back(env, fake_llm_calls, caplog, answer):
    _tmp_path, config, ctx = env
    config["llm"] = {"provider": "fake", "model": "script-1", "answer": answer}
    segments = [seg(0.0, 2.0, "а"), seg(2.0, 4.0, "б"), seg(7.0, 9.0, "в")]
    write_transcript(ctx, segments)

    with caplog.at_level(logging.WARNING, logger="razkadr.chunks"):
        doc = run_chunks(ctx, config)

    expected = build_chunks(segments, fallback_boundaries(segments, 1200, 1.5))
    assert doc["chunks"] == expected  # exactly the deterministic fallback result
    assert "chunking fallback" in caplog.text
    assert any("invalid LLM answer" in r for r in _warning_reasons(caplog))


def test_llm_multi_segment_chunk_over_max_chars_falls_back(env, fake_llm_calls, caplog):
    _tmp_path, config, ctx = env
    # the answer keeps segments 0-1 together: 40 chars > max_chars 20 -> invalid
    config["llm"] = {"provider": "fake", "model": "script-1", "answer": '{"boundaries": [0, 2]}'}
    config["chunks"] = {"max_chars": 20}
    segments = [seg(0.0, 2.0, "х" * 20), seg(2.0, 4.0, "у" * 20), seg(6.0, 8.0, "з" * 5)]
    write_transcript(ctx, segments)

    with caplog.at_level(logging.WARNING, logger="razkadr.chunks"):
        doc = run_chunks(ctx, config)

    expected = build_chunks(segments, fallback_boundaries(segments, 20, 1.5))
    assert doc["chunks"] == expected
    assert any("max_chars" in r for r in _warning_reasons(caplog))


def test_llm_single_segment_chunk_over_max_chars_is_kept(env, fake_llm_calls):
    _tmp_path, config, ctx = env
    # both chunks are single-segment: an over-long ASR segment cannot be split
    config["llm"] = {"provider": "fake", "model": "script-1", "answer": '{"boundaries": [0, 1]}'}
    config["chunks"] = {"max_chars": 5}
    segments = [seg(0.0, 2.0, "длинный сегмент номер один"), seg(2.0, 4.0, "и второй тоже длинный")]
    write_transcript(ctx, segments)

    doc = run_chunks(ctx, config)
    assert [c["segment_ids"] for c in doc["chunks"]] == [[0], [1]]  # LLM answer kept


def test_stock_fake_llm_answer_is_invalid_and_falls_back(env):
    """The naryad 033 FakeLLM returns prose, not JSON: fallback must catch it."""
    _tmp_path, config, ctx = env
    config["llm"] = {"provider": "fake", "model": "fake-gpt-4o-mini"}
    segments = [seg(0.0, 2.0, "а"), seg(5.0, 7.0, "б")]
    write_transcript(ctx, segments)

    doc = run_chunks(ctx, config)

    expected = build_chunks(segments, fallback_boundaries(segments, 1200, 1.5))
    assert doc["chunks"] == expected
    run = _read_json(ctx, "run.json")
    assert len([c for c in run["provider_calls"] if c["task"] == "llm"]) == 1  # the call still happened


def test_no_llm_section_falls_back_without_provider_calls(env, caplog):
    _tmp_path, config, ctx = env
    write_transcript(ctx, [seg(0.0, 2.0, "а"), seg(5.0, 7.0, "б")])
    with caplog.at_level(logging.WARNING, logger="razkadr.chunks"):
        doc = run_chunks(ctx, config)
    assert doc["chunks"][0]["segment_ids"] == [0]  # pause 3 s >= 1.5 s closed the chunk
    assert any("no [llm] provider configured" in r for r in _warning_reasons(caplog))
    assert _read_json(ctx, "run.json")["provider_calls"] == []


# --- stage: configured-but-broken LLM fails loudly ------------------------------------------


def test_unknown_llm_provider_fails_the_stage(env):
    _tmp_path, config, ctx = env
    config["llm"] = {"provider": "nope"}
    write_transcript(ctx, [seg(0.0, 2.0, "текст")])
    with pytest.raises(ChunksError, match="unknown llm provider"):
        run_chunks(ctx, config)


def test_llm_auth_error_fails_the_stage(env, fake_llm_calls):
    _tmp_path, config, ctx = env
    config["llm"] = {"provider": "fake", "model": "script-1", "auth_error": "bad key"}
    write_transcript(ctx, [seg(0.0, 2.0, "текст")])
    with pytest.raises(AuthError):
        run_chunks(ctx, config)


# --- stage: empty and broken inputs -----------------------------------------------------------


def test_empty_transcript_gives_empty_chunks_without_llm_calls(env, fake_llm_calls):
    _tmp_path, config, ctx = env
    config["llm"] = {"provider": "fake", "model": "script-1"}
    write_transcript(ctx, [])

    doc = run_chunks(ctx, config)

    assert doc == {"schema_version": "1.0", "chunks": []}
    assert validate_document(doc, "chunks") == []
    assert fake_llm_calls == []  # no segments -> no provider work, no cost
    assert (ctx.dir / CHUNKS_JSON).is_file()


def test_missing_transcript_is_a_clear_error(env):
    _tmp_path, config, ctx = env
    with pytest.raises(ChunksError, match="transcript.json is missing"):
        run_chunks(ctx, config)


def test_corrupted_transcript_json_is_a_clear_error(env):
    _tmp_path, config, ctx = env
    (ctx.dir / "transcript.json").write_text("{oops", encoding="utf-8")
    with pytest.raises(ChunksError, match="not valid JSON"):
        run_chunks(ctx, config)


def test_invalid_transcript_document_is_a_clear_error(env):
    _tmp_path, config, ctx = env
    (ctx.dir / "transcript.json").write_text('{"schema_version": "1.0"}', encoding="utf-8")
    with pytest.raises(ChunksError, match="not a valid transcript"):
        run_chunks(ctx, config)


# --- stage: cost, cache, resume, determinism ---------------------------------------------------


def test_llm_cost_recorded_in_run_json(env, fake_llm_calls):
    _tmp_path, config, ctx = env
    config["llm"] = {
        "provider": "fake",
        "model": "script-1",
        "answer": '{"boundaries": [0]}',
        "tariff": {"units": "tokens", "price": 0.002, "currency": "USD", "tariff_date": "2026-10-01"},
    }
    write_transcript(ctx, [seg(0.0, 2.0, "текст для чанка")])

    run_chunks(ctx, config)

    run = _read_json(ctx, "run.json")
    calls = [c for c in run["provider_calls"] if c["task"] == "llm"]
    assert len(calls) == 1 and calls[0]["cache_hit"] is False
    assert calls[0]["cost"] == {
        "amount": round(calls[0]["units"][0]["value"] * 0.002, 6),
        "currency": "USD",
        "tariff_date": "2026-10-01",
    }


def test_resume_returns_cached_artifact_without_new_llm_calls(env, fake_llm_calls):
    _tmp_path, config, ctx = env
    config["llm"] = {"provider": "fake", "model": "script-1", "answer": '{"boundaries": [0, 2]}'}
    write_transcript(ctx, [seg(0.0, 2.0, "а"), seg(2.0, 4.0, "б"), seg(4.0, 6.0, "в")])
    first = run_chunks(ctx, config)

    ctx2 = RunContext(config["runs_dir"], run_id=ctx.run_id, config=config)
    second = run_chunks(ctx2, config)

    assert second == first
    run = _read_json(ctx2, "run.json")
    entries = [s for s in run["stages"] if s["name"] == "chunks"]
    assert entries[-1]["status"] == "cached"
    assert len(fake_llm_calls) == 1  # the factory built one provider for the first run only


def test_force_rerun_serves_llm_call_from_call_cache(env, fake_llm_calls):
    _tmp_path, config, ctx = env
    config["llm"] = {
        "provider": "fake",
        "model": "script-1",
        "answer": '{"boundaries": [0]}',
        "tariff": {"units": "tokens", "price": 0.002, "currency": "USD", "tariff_date": "2026-10-01"},
    }
    write_transcript(ctx, [seg(0.0, 2.0, "текст")])
    run_chunks(ctx, config)

    ctx2 = RunContext(config["runs_dir"], run_id=ctx.run_id, config=config, force=True)
    run_chunks(ctx2, config)

    run = _read_json(ctx2, "run.json")
    calls = [c for c in run["provider_calls"] if c["task"] == "llm"]
    assert len(calls) == 2
    assert calls[0]["cache_hit"] is False and calls[0]["cost"]["amount"] > 0
    assert calls[1]["cache_hit"] is True and calls[1]["cost"]["amount"] == 0.0


def test_chunks_config_change_invalidates_stage(env, fake_llm_calls):
    _tmp_path, config, ctx = env
    config["llm"] = {"provider": "fake", "model": "script-1", "answer": '{"boundaries": [0, 2]}'}
    write_transcript(ctx, [seg(0.0, 2.0, "а"), seg(2.0, 4.0, "б"), seg(4.0, 6.0, "в")])
    run_chunks(ctx, config)

    config["chunks"] = {"max_chars": 10}
    ctx2 = RunContext(config["runs_dir"], run_id=ctx.run_id, config=config)
    run_chunks(ctx2, config)
    run = _read_json(ctx2, "run.json")
    entries = [s for s in run["stages"] if s["name"] == "chunks"]
    assert entries[-1]["status"] == "ok"  # re-ran, not cached


def test_llm_provider_change_invalidates_stage(env, fake_llm_calls):
    _tmp_path, config, ctx = env
    config["llm"] = {"provider": "fake", "model": "script-1", "answer": '{"boundaries": [0]}'}
    write_transcript(ctx, [seg(0.0, 2.0, "текст")])
    run_chunks(ctx, config)

    config["llm"]["model_version"] = "2"
    ctx2 = RunContext(config["runs_dir"], run_id=ctx.run_id, config=config)
    run_chunks(ctx2, config)
    run = _read_json(ctx2, "run.json")
    entries = [s for s in run["stages"] if s["name"] == "chunks"]
    assert entries[-1]["status"] == "ok"


def test_deterministic_byte_identical_across_runs(env, fake_llm_calls):
    _tmp_path, config, ctx1 = env
    config["llm"] = {"provider": "fake", "model": "script-1", "answer": '{"boundaries": [0, 2]}'}
    segments = [seg(0.0, 2.0, "а"), seg(2.0, 4.0, "б"), seg(7.0, 9.0, "в")]
    write_transcript(ctx1, segments)
    doc1 = run_chunks(ctx1, config)

    config["runs_dir"] = str(_tmp_path / "runs2")
    ctx2 = RunContext(config["runs_dir"], config=config)
    write_transcript(ctx2, segments)
    doc2 = run_chunks(ctx2, config)

    assert doc1 == doc2
    assert (ctx1.dir / CHUNKS_JSON).read_bytes() == (ctx2.dir / CHUNKS_JSON).read_bytes()


# --- CLI: index runs ingest -> frames -> transcript -> chunks ------------------------------------


def _write_toml(tmp_path: Path, with_llm: bool = True) -> Path:
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
        "[transcript]",
        "part_length_s = 6",
        "overlap_s = 2",
        "",
    ]
    if with_llm:
        lines += ["[llm]", 'provider = "fake"', 'model = "fake-gpt-4o-mini"', ""]
    path = tmp_path / "razkadr.toml"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def video_with_audio(synthetic_fixtures, tmp_path_factory):
    """scenes.mp4 (video only) muxed with a sine track: video + audio fixture."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not found in PATH")
    dst = tmp_path_factory.mktemp("chunks_with_audio") / "with_audio.mp4"
    import subprocess

    subprocess.run(
        ["ffmpeg", "-y", "-i", str(synthetic_fixtures.path("scenes.mp4")),
         "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=30",
         "-shortest", "-c:v", "copy", "-c:a", "aac", "-b:a", "96k", str(dst)],
        check=True, capture_output=True,
    )
    return dst


def test_cli_index_runs_chunks(tmp_path, video_with_audio, monkeypatch, capsys):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not found in PATH")
    monkeypatch.chdir(tmp_path)
    config_path = _write_toml(tmp_path)

    code = main(["index", str(video_with_audio), "--config", str(config_path)])

    assert code == EXIT_OK
    out = capsys.readouterr().out
    assert "ingest ok, frames ok, transcript ok, chunks ok" in out
    assert "chunks:" in out
    runs_dir = tmp_path / "runs"
    run_dir = next(d for d in sorted(runs_dir.iterdir()) if d.is_dir())
    doc = json.loads((run_dir / CHUNKS_JSON).read_text(encoding="utf-8"))
    assert validate_document(doc, "chunks") == []
    assert doc["chunks"], "the fake ASR produces segments, so chunks are expected"


def test_cli_index_fallback_without_llm_section(tmp_path, video_with_audio, monkeypatch, capsys):
    """No [llm] in the config: chunks still arrive via the deterministic fallback."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not found in PATH")
    monkeypatch.chdir(tmp_path)
    config_path = _write_toml(tmp_path, with_llm=False)

    code = main(["index", str(video_with_audio), "--config", str(config_path)])

    assert code == EXIT_OK
    runs_dir = tmp_path / "runs"
    run_dir = next(d for d in sorted(runs_dir.iterdir()) if d.is_dir())
    doc = json.loads((run_dir / CHUNKS_JSON).read_text(encoding="utf-8"))
    assert validate_document(doc, "chunks") == []
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert [c for c in run["provider_calls"] if c["task"] == "llm"] == []


def test_cli_index_no_audio_empty_chunks(tmp_path, synthetic_fixtures, monkeypatch, capsys):
    """A silent source: valid empty chunks.json, no transcript/chunks lines."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not found in PATH")
    monkeypatch.chdir(tmp_path)
    config_path = _write_toml(tmp_path)

    code = main(["index", str(synthetic_fixtures.path("scenes.mp4")), "--config", str(config_path)])

    assert code == EXIT_OK
    out = capsys.readouterr().out
    assert "transcript:" not in out and "chunks:" not in out
    runs_dir = tmp_path / "runs"
    run_dir = next(d for d in sorted(runs_dir.iterdir()) if d.is_dir())
    doc = json.loads((run_dir / CHUNKS_JSON).read_text(encoding="utf-8"))
    assert doc == {"schema_version": "1.0", "chunks": []}


def test_cli_validate_chunks_kind(tmp_path, video_with_audio, monkeypatch, capsys):
    """The written artifact passes `razkadr validate --kind chunks`."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not found in PATH")
    monkeypatch.chdir(tmp_path)
    config_path = _write_toml(tmp_path)
    assert main(["index", str(video_with_audio), "--config", str(config_path)]) == EXIT_OK
    capsys.readouterr()  # discard the index output; only the validate run is asserted

    runs_dir = tmp_path / "runs"
    run_dir = next(d for d in sorted(runs_dir.iterdir()) if d.is_dir())
    code = main(["validate", "--kind", "chunks", str(run_dir / CHUNKS_JSON)])

    assert code == EXIT_OK
    assert capsys.readouterr().out.strip() == "ok"


def test_example_artifact_chunks_is_valid():
    from razkadr.validation import validate_document

    root = Path(__file__).resolve().parents[1]
    doc = json.loads((root / "examples" / "artifacts" / "chunks.json").read_text(encoding="utf-8"))
    assert validate_document(doc, "chunks") == []

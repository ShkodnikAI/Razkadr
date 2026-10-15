"""Transcript stage tests (naryad 007): part splitting, joint dedup, empty speech.

All tests run on the synthetic fixtures (naryad 034) through the fake
providers of the naryad 033 layer; no network, no external APIs.

The SampledFakeASR test provider derives each segment text from the hash
of the segment's audio samples: the same absolute audio region yields the
same text in any part, so consecutive parts duplicate content in the
overlap zone - exactly what the joint deduplication must remove. With
part_length_s = 6 and overlap_s = 2 on the 30 s fixture, part windows are
aligned to the 2 s segment grid and every joint produces exactly one
duplicated segment (21 raw segments -> 15 unique).
"""

import hashlib
import json
import math
import shutil
import wave
from pathlib import Path

import pytest

from razkadr.cli import EXIT_FILE_ERROR, EXIT_OK, main
from razkadr.config import load_config
from razkadr.providers.base import UsageReporter
from razkadr.providers.fakes import FakeASR
from razkadr.providers.layer import FACTORIES
from razkadr.providers.types import Transcript, TranscriptSegment
from razkadr.runctx import RunContext
from razkadr.stages.ingest import run_ingest
from razkadr.stages.transcript import (
    PARTS_DIRNAME,
    TRANSCRIPT_JSON,
    TranscriptError,
    _overlaps,
    _similar_text,
    merge_parts,
    run_transcript,
    split_parts,
    transcript_params,
    wav_duration_s,
    write_part,
)
from razkadr.validation import validate_document


class SampledFakeASR(UsageReporter):
    """Deterministic ASR: text = hash of the segment's audio samples."""

    name = "sampled"
    model = "sampled-1"
    task = "asr"

    def __init__(self, segment_s: float = 2.0, usage_seconds: float = 6.0, lang: str = "ru") -> None:
        self.segment_s = segment_s
        self.usage_seconds = usage_seconds
        self.lang = lang
        self.calls = 0
        self.last_usage = []

    def transcribe(self, audio_path: Path, lang: str | None = None) -> Transcript:
        self.calls += 1
        self._report([{"kind": "seconds", "value": self.usage_seconds}])
        use_lang = lang or self.lang
        with wave.open(str(audio_path), "rb") as handle:
            rate, total = handle.getframerate(), handle.getnframes()
        duration = total / float(rate)
        segments: list[TranscriptSegment] = []
        t = 0.0
        while t < duration - 1e-9:
            t2 = min(t + self.segment_s, duration)
            first, last = round(t * rate), round(t2 * rate)
            with wave.open(str(audio_path), "rb") as handle:
                handle.setpos(first)
                payload = handle.readframes(last - first)
            digest = hashlib.sha256(payload).hexdigest()[:10]
            segments.append(
                TranscriptSegment(t_start=round(t, 3), t_end=round(t2, 3), text=f"seg-{digest}", lang=use_lang)
            )
            t = t2
        return Transcript(lang=use_lang, segments=segments)


class MuteASR(UsageReporter):
    """Recognizes nothing: valid empty transcript, still bills seconds."""

    name = "mute"
    model = "mute-1"
    task = "asr"

    def __init__(self, usage_seconds: float = 30.0) -> None:
        self.usage_seconds = usage_seconds
        self.calls = 0
        self.last_usage = []

    def transcribe(self, audio_path: Path, lang: str | None = None) -> Transcript:
        self.calls += 1
        self._report([{"kind": "seconds", "value": self.usage_seconds}])
        return Transcript(lang=lang or "und", segments=[])


@pytest.fixture()
def ffmpeg_bin():
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not found in PATH")
    return True


@pytest.fixture()
def fake_asr_calls(monkeypatch):
    """Replace the fake ASR factory, capturing every created instance."""
    created: list[FakeASR] = []

    def factory(section: dict) -> FakeASR:
        provider = FakeASR(
            model=section.get("model", "fake-nova-2"),
            lang=section.get("lang", "ru"),
            usage_seconds=float(section.get("usage_seconds", 10.0)),
        )
        created.append(provider)
        return provider

    monkeypatch.setitem(FACTORIES["asr"], "fake", factory)
    return created


@pytest.fixture()
def env(tmp_path, ffmpeg_bin, fake_asr_calls):
    """RunContext + config with isolated runs and call cache dirs."""
    config = load_config(None)
    config["runs_dir"] = str(tmp_path / "runs")
    config["cache"] = {"dir": str(tmp_path / "calls")}
    config["asr"] = {"provider": "fake", "model": "fake-nova-2"}
    ctx = RunContext(config["runs_dir"], config=config)
    return tmp_path, config, ctx


@pytest.fixture()
def ingested(env, video_with_audio):
    """env with the ingest stage already run on the video+audio fixture."""
    _tmp_path, config, ctx = env
    run_ingest(ctx, str(video_with_audio), config)
    return config, ctx


@pytest.fixture(scope="session")
def video_with_audio(synthetic_fixtures, tmp_path_factory):
    """scenes.mp4 (video only) muxed with a sine track: video + audio fixture."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not found in PATH")
    dst = tmp_path_factory.mktemp("with_audio") / "with_audio.mp4"
    import subprocess

    subprocess.run(
        ["ffmpeg", "-y", "-i", str(synthetic_fixtures.path("scenes.mp4")),
         "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=30",
         "-shortest", "-c:v", "copy", "-c:a", "aac", "-b:a", "96k", str(dst)],
        check=True, capture_output=True,
    )
    return dst


def _read_json(ctx: RunContext, name: str) -> dict:
    return json.loads((ctx.dir / name).read_text(encoding="utf-8"))


# --- unit: part splitting edge cases (naryad 007 auditor list) ------------------


def test_split_parts_audio_shorter_than_one_part():
    params = transcript_params({"transcript": {"part_length_s": 120, "overlap_s": 2}})
    assert split_parts(30.0, params) == [(0.0, 30.0)]


def test_split_parts_exactly_on_boundary_no_tiny_tail():
    params = transcript_params({"transcript": {"part_length_s": 10, "overlap_s": 0}})
    assert split_parts(30.0, params) == [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0)]
    assert split_parts(10.0, params) == [(0.0, 10.0)]


def test_split_parts_overlap_between_parts():
    params = transcript_params({"transcript": {"part_length_s": 120, "overlap_s": 2}})
    assert split_parts(300.0, params) == [(0.0, 120.0), (118.0, 238.0), (236.0, 300.0)]


def test_split_parts_tail_just_over_boundary_becomes_small_last_part():
    params = transcript_params({"transcript": {"part_length_s": 10, "overlap_s": 2}})
    assert split_parts(10.001, params) == [(0.0, 10.0), (8.0, 10.001)]


@pytest.mark.parametrize(
    "overrides",
    [
        {"overlap_s": 10, "part_length_s": 5},   # overlap > length
        {"overlap_s": 5, "part_length_s": 5},    # overlap == length: step 0, would hang
        {"overlap_s": -1},                        # negative overlap
        {"part_length_s": 0},                     # zero part
        {"part_length_s": -3},                    # negative part
    ],
)
def test_params_overlap_bigger_than_part_length_is_an_error(overrides):
    with pytest.raises(TranscriptError, match="part_length_s|overlap_s"):
        transcript_params({"transcript": overrides})


def test_params_invalid_forced_lang_is_an_error():
    with pytest.raises(TranscriptError, match="BCP-47"):
        transcript_params({"transcript": {"lang": "русский"}})


# --- unit: joint dedup text similarity ------------------------------------------

def test_similar_text_exact_and_normalized():
    assert _similar_text("Hello world", "hello  world!")
    assert _similar_text("один два три", "  один   два   три ")
    assert not _similar_text("совсем другой текст", "привет мир друзья")


def test_similar_text_containment_only_for_long_enough_texts():
    assert _similar_text("hello world segment", "hello world")  # clipped at a part edge
    assert not _similar_text("да", "да, и ещё")  # short texts: no containment


def test_similar_text_ratio_threshold():
    a = "как приготовить тесто для пиццы быстро"
    b = "как приготовить тесто для пиццы медленно"
    assert _similar_text(a, b)
    assert not _similar_text(a, "завтра будет дождь и снег")


def test_overlaps_intervals():
    assert _overlaps((0.0, 2.0), (1.0, 3.0))
    assert not _overlaps((0.0, 2.0), (2.0, 4.0))  # touching is not overlapping


def test_merge_parts_drops_duplicates_and_keeps_different_text():
    # part 2 starts at 2.0 with overlap 2.0, so its overlap zone is [2, 4);
    # the first segment of part 2 covers the same audio as part 1's (2, 4)
    tr1 = Transcript(lang="ru", segments=[
        TranscriptSegment(0.0, 2.0, "общий кусок текста", "ru"),
        TranscriptSegment(2.0, 4.0, "только в первой", "ru"),
    ])
    tr2 = Transcript(lang="ru", segments=[
        TranscriptSegment(0.0, 2.0, "только в первой", "ru"),
        TranscriptSegment(2.0, 3.5, "новый кусок текста", "ru"),
    ])
    segments, doc_lang = merge_parts([tr1, tr2], [0.0, 2.0], overlap_s=2.0)
    assert [s["text"] for s in segments] == ["общий кусок текста", "только в первой", "новый кусок текста"]
    assert [s["t_start"] for s in segments] == [0.0, 2.0, 4.0]
    assert doc_lang == "ru"


def test_merge_parts_tie_on_language_keeps_first_part_lang():
    tr1 = Transcript(lang="ru", segments=[TranscriptSegment(0.0, 1.0, "раз", "ru")])
    tr2 = Transcript(lang="en", segments=[TranscriptSegment(0.0, 1.0, "one", "en")])
    _segments, doc_lang = merge_parts([tr1, tr2], [0.0, 1.0], overlap_s=0.0)
    assert doc_lang == "ru"


def test_merge_parts_majority_language_wins():
    tr1 = Transcript(lang="ru", segments=[TranscriptSegment(0.0, 1.0, "раз", "ru")])
    tr2 = Transcript(lang="en", segments=[TranscriptSegment(0.0, 1.0, "one", "en")])
    tr3 = Transcript(lang="en", segments=[TranscriptSegment(0.0, 1.0, "two", "en")])
    _segments, doc_lang = merge_parts([tr1, tr2, tr3], [0.0, 1.0, 2.0], overlap_s=0.0)
    assert doc_lang == "en"


# --- unit: wav duration and part slicing ----------------------------------------

def test_wav_duration_and_write_part_roundtrip(ingested):
    _config, ctx = ingested
    audio = ctx.dir / "audio.wav"
    duration = wav_duration_s(audio)
    assert 29.0 <= duration <= 31.0  # 30 s fixture, ffmpeg rounding

    params = transcript_params({})
    spans = split_parts(duration, params)
    assert len(spans) == 1
    write_part(audio, ctx.dir / "copy.wav", 0.0, duration / 2)
    assert abs(wav_duration_s(ctx.dir / "copy.wav") - duration / 2) < 0.01


# --- stage: contract on the fake ASR ---------------------------------------------


def test_contract_transcript_artifact(ingested):
    config, ctx = ingested
    doc = run_transcript(ctx, config)

    assert validate_document(doc, "transcript") == []
    assert doc["schema_version"] == "1.0"
    assert doc["lang"] == "ru"
    assert len(doc["segments"]) == 2  # FakeASR: two deterministic segments
    duration = wav_duration_s(ctx.dir / "audio.wav")
    for seg in doc["segments"]:
        assert 0.0 <= seg["t_start"] <= seg["t_end"] <= duration + 0.5
        assert seg["lang"] == "ru" and seg["text"]
    starts = [s["t_start"] for s in doc["segments"]]
    assert starts == sorted(starts)  # monotonic timecodes

    parts_dir = ctx.dir / PARTS_DIRNAME
    assert parts_dir.is_dir() and len(list(parts_dir.glob("part_*.wav"))) == 1

    run = _read_json(ctx, "run.json")
    assert validate_document(run, "run") == []
    assert run["stages"][-1]["name"] == "transcript" and run["stages"][-1]["status"] == "ok"
    calls = run["provider_calls"]
    assert len(calls) == 1
    assert calls[0]["stage"] == "transcript" and calls[0]["task"] == "asr"
    assert calls[0]["cache_hit"] is False
    assert calls[0]["units"] == [{"kind": "seconds", "value": 10.0}]


def test_transcript_file_written_and_validated_by_cli(ingested):
    config, ctx = ingested
    run_transcript(ctx, config)
    assert (ctx.dir / TRANSCRIPT_JSON).is_file()
    assert validate_document(_read_json(ctx, TRANSCRIPT_JSON), "transcript") == []


# --- stage: multi-part merge gives monotonic timecodes without joint dupes -------


def test_multi_part_merge_monotonic_no_dupes_at_joints(env, video_with_audio, monkeypatch):
    _tmp_path, config, ctx = env
    config["transcript"] = {"part_length_s": 6, "overlap_s": 2}
    sampled = SampledFakeASR()
    monkeypatch.setitem(FACTORIES["asr"], "sampled", lambda _section: sampled)
    config["asr"] = {"provider": "sampled", "model": "sampled-1"}
    run_ingest(ctx, str(video_with_audio), config)

    doc = run_transcript(ctx, config)

    assert validate_document(doc, "transcript") == []
    duration = wav_duration_s(ctx.dir / "audio.wav")
    spans = split_parts(duration, transcript_params(config))
    assert sampled.calls == len(spans) >= 2  # the 30 s fixture is split into parts

    segments = doc["segments"]
    # unique 2 s windows over the audio: 0..duration; every joint duplicated
    # the window shared by consecutive parts and the merge removed it
    expected = math.ceil(duration / 2.0 - 1e-9)
    assert len(segments) == expected
    texts = [s["text"] for s in segments]
    assert len(set(texts)) == len(texts), "duplicated texts at joints must be removed"
    starts = [s["t_start"] for s in segments]
    assert starts == [2.0 * i for i in range(len(segments))], f"non-monotonic grid: {starts}"
    ends = [s["t_end"] for s in segments]
    assert ends == sorted(ends) and ends[-1] <= duration + 0.5

    run = _read_json(ctx, "run.json")
    asr_calls = [c for c in run["provider_calls"] if c["task"] == "asr"]
    assert len(asr_calls) == len(spans) and all(c["cache_hit"] is False for c in asr_calls)


def test_multi_part_content_matches_single_part_content(env, video_with_audio, monkeypatch):
    """Merged multi-part text set equals the single-part text set for the same audio."""
    _tmp_path, config, ctx = env
    run_ingest(ctx, str(video_with_audio), config)

    single = SampledFakeASR()
    monkeypatch.setitem(FACTORIES["asr"], "sampled", lambda _s: single)
    config["asr"] = {"provider": "sampled", "model": "sampled-1"}
    doc_single = run_transcript(ctx, config)

    # force a re-run of the stage with the multi-part config; the provider
    # call cache still holds the single-part result, so use a fresh provider
    # name to keep the two runs independent
    multi = SampledFakeASR()
    monkeypatch.setitem(FACTORIES["asr"], "sampled", lambda _s: multi)
    config["asr"] = {"provider": "sampled", "model": "sampled-2"}
    config["transcript"] = {"part_length_s": 6, "overlap_s": 2}
    doc_multi = run_transcript(ctx, config)

    assert sorted(s["text"] for s in doc_single["segments"]) == sorted(
        s["text"] for s in doc_multi["segments"]
    )


# --- stage: no speech / no audio -> valid empty transcript, not an error ----------


def test_no_audio_track_empty_transcript_without_provider(env, synthetic_fixtures):
    _tmp_path, config, ctx = env
    run_ingest(ctx, str(synthetic_fixtures.path("scenes.mp4")), config)  # video has no audio
    doc = run_transcript(ctx, config)

    assert validate_document(doc, "transcript") == []
    assert doc["lang"] == "und" and doc["segments"] == []
    assert not (ctx.dir / PARTS_DIRNAME).exists()
    run = _read_json(ctx, "run.json")
    assert run["provider_calls"] == []  # no cost booked without audio
    assert run["stages"][-1]["name"] == "transcript" and run["stages"][-1]["status"] == "ok"


def test_no_speech_provider_returns_valid_empty_transcript(env, video_with_audio, monkeypatch):
    _tmp_path, config, ctx = env
    mute = MuteASR()
    monkeypatch.setitem(FACTORIES["asr"], "mute", lambda _section: mute)
    config["asr"] = {"provider": "mute", "model": "mute-1"}
    run_ingest(ctx, str(video_with_audio), config)

    doc = run_transcript(ctx, config)

    assert validate_document(doc, "transcript") == []
    assert doc["lang"] == "und" and doc["segments"] == []
    assert mute.calls == 1
    run = _read_json(ctx, "run.json")
    assert len(run["provider_calls"]) == 1  # the call still happened and is accounted


# --- stage: forced language -------------------------------------------------------


def test_forced_lang_passed_to_provider_and_recorded(ingested):
    config, ctx = ingested
    config["transcript"] = {"lang": "en"}
    doc = run_transcript(ctx, config)
    assert doc["lang"] == "en"
    assert all(s["lang"] == "en" for s in doc["segments"])


def test_invalid_forced_lang_fails_before_any_provider_call(ingested, fake_asr_calls):
    config, ctx = ingested
    config["transcript"] = {"lang": "not a lang!"}
    with pytest.raises(TranscriptError, match="BCP-47"):
        run_transcript(ctx, config)
    assert fake_asr_calls == []


# --- stage: cost and cache_hit in run.json ----------------------------------------


def test_cost_and_cache_hit_recorded_in_run_json(env, video_with_audio):
    _tmp_path, config, ctx = env
    config["asr"]["tariff"] = {"units": "seconds", "price": 0.0059, "currency": "USD",
                               "tariff_date": "2026-10-01"}
    run_ingest(ctx, str(video_with_audio), config)

    run_transcript(ctx, config)
    run = _read_json(ctx, "run.json")
    first = [c for c in run["provider_calls"] if c["task"] == "asr"]
    assert len(first) == 1
    assert first[0]["cache_hit"] is False
    assert first[0]["cost"] == {"amount": 0.059, "currency": "USD", "tariff_date": "2026-10-01"}

    # forced re-run: the stage body executes again, but the provider layer
    # serves the calls from the call cache -> cache_hit=true, zero cost
    ctx2 = RunContext(config["runs_dir"], run_id=ctx.run_id, config=config, force=True)
    run_transcript(ctx2, config)
    run = _read_json(ctx2, "run.json")
    calls = [c for c in run["provider_calls"] if c["task"] == "asr"]
    assert len(calls) == 2
    assert calls[0]["cache_hit"] is False and calls[0]["cost"]["amount"] == 0.059
    assert calls[1]["cache_hit"] is True and calls[1]["cost"]["amount"] == 0.0


# --- stage: cache, resume and invalidation ----------------------------------------


def test_resume_cached_returns_artifact_without_new_provider_calls(ingested):
    config, ctx = ingested
    first = run_transcript(ctx, config)

    ctx2 = RunContext(config["runs_dir"], run_id=ctx.run_id, config=config)
    second = run_transcript(ctx2, config)

    assert second == first
    run = _read_json(ctx2, "run.json")
    transcript_entries = [s for s in run["stages"] if s["name"] == "transcript"]
    assert transcript_entries[-1]["status"] == "cached"
    assert len([c for c in run["provider_calls"] if c["task"] == "asr"]) == 1  # no new calls


def test_transcript_config_change_invalidates_stage(ingested):
    config, ctx = ingested
    run_transcript(ctx, config)
    config["transcript"] = {"part_length_s": 10, "overlap_s": 1}

    ctx2 = RunContext(config["runs_dir"], run_id=ctx.run_id, config=config)
    run_transcript(ctx2, config)
    run = _read_json(ctx2, "run.json")
    transcript_entries = [s for s in run["stages"] if s["name"] == "transcript"]
    assert transcript_entries[-1]["status"] == "ok"  # re-ran, not cached
    assert len([c for c in run["provider_calls"] if c["task"] == "asr"]) > 1


def test_asr_provider_change_invalidates_stage(ingested):
    config, ctx = ingested
    run_transcript(ctx, config)
    config["asr"]["model"] = "fake-nova-alt"

    ctx2 = RunContext(config["runs_dir"], run_id=ctx.run_id, config=config)
    run_transcript(ctx2, config)
    run = _read_json(ctx2, "run.json")
    transcript_entries = [s for s in run["stages"] if s["name"] == "transcript"]
    assert transcript_entries[-1]["status"] == "ok"


def test_transcript_deterministic_across_runs(env, video_with_audio):
    _tmp_path, config, ctx1 = env
    config["transcript"] = {"part_length_s": 6, "overlap_s": 2}
    run_ingest(ctx1, str(video_with_audio), config)
    doc1 = run_transcript(ctx1, config)

    config["runs_dir"] = str(_tmp_path / "runs2")
    ctx2 = RunContext(config["runs_dir"], config=config)
    run_ingest(ctx2, str(video_with_audio), config)
    doc2 = run_transcript(ctx2, config)

    assert doc1 == doc2
    assert (ctx1.dir / TRANSCRIPT_JSON).read_bytes() == (ctx2.dir / TRANSCRIPT_JSON).read_bytes()


# --- stage: clear errors -----------------------------------------------------------


def test_missing_source_json_is_a_clear_error(env):
    _tmp_path, config, ctx = env
    with pytest.raises(TranscriptError, match="source.json is missing"):
        run_transcript(ctx, config)


def test_missing_audio_wav_with_audio_track_is_a_clear_error(env, tmp_path):
    _tmp_path, config, ctx = env
    (ctx.dir / "source.json").write_text(
        json.dumps({
            "schema_version": "1.0", "path": "normalized.mp4", "sha256": "0" * 64,
            "duration_s": 1.0, "fps": 10.0, "width": 320, "height": 240,
            "streams": [{"index": 0, "type": "video", "codec": "mpeg4"},
                        {"index": 1, "type": "audio", "codec": "aac"}],
        }),
        encoding="utf-8",
    )
    with pytest.raises(TranscriptError, match="re-run ingest"):
        run_transcript(ctx, config)


def test_missing_asr_config_is_a_clear_error(env, video_with_audio):
    _tmp_path, config, ctx = env
    config.pop("asr")
    run_ingest(ctx, str(video_with_audio), config)
    with pytest.raises(TranscriptError, match=r"\[asr\].*docs/config.md"):
        run_transcript(ctx, config)


# --- CLI: index runs ingest -> frames -> transcript --------------------------------


def _write_toml(tmp_path: Path, extra_asr: bool = True) -> Path:
    lines = [
        f'runs_dir = "{tmp_path / "runs"}"',
        "",
        "[cache]",
        f'dir = "{tmp_path / "calls"}"',
        "",
    ]
    if extra_asr:
        lines += [
            "[asr]",
            'provider = "fake"',
            'model = "fake-nova-2"',
            "",
        ]
    lines += [
        "[transcript]",
        "part_length_s = 6",
        "overlap_s = 2",
        "",
    ]
    path = tmp_path / "razkadr.toml"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_cli_index_runs_transcript(tmp_path, video_with_audio, monkeypatch, capsys):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not found in PATH")
    monkeypatch.chdir(tmp_path)
    config_path = _write_toml(tmp_path)

    code = main(["index", str(video_with_audio), "--config", str(config_path)])

    assert code == EXIT_OK
    out = capsys.readouterr().out
    assert "ingest ok, frames ok, transcript ok" in out
    assert "lang=ru" in out
    runs_dir = tmp_path / "runs"
    run_dir = next(d for d in sorted(runs_dir.iterdir()) if d.is_dir())
    doc = json.loads((run_dir / TRANSCRIPT_JSON).read_text(encoding="utf-8"))
    assert validate_document(doc, "transcript") == []
    assert doc["lang"] == "ru"


def test_cli_index_without_asr_config_exit_2(tmp_path, video_with_audio, monkeypatch, capsys):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not found in PATH")
    monkeypatch.chdir(tmp_path)
    config_path = _write_toml(tmp_path, extra_asr=False)

    code = main(["index", str(video_with_audio), "--config", str(config_path)])

    assert code == EXIT_FILE_ERROR
    err = capsys.readouterr().err
    assert "razkadr index:" in err and "[asr]" in err


def test_cli_index_no_audio_transcript_is_und(tmp_path, synthetic_fixtures, monkeypatch, capsys):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not found in PATH")
    monkeypatch.chdir(tmp_path)
    config_path = _write_toml(tmp_path, extra_asr=True)

    code = main(["index", str(synthetic_fixtures.path("scenes.mp4")), "--config", str(config_path)])

    assert code == EXIT_OK
    out = capsys.readouterr().out
    assert "none (source has no audio track)" in out
    assert "transcript:" not in out  # no speech line for a silent source
    runs_dir = tmp_path / "runs"
    run_dir = next(d for d in sorted(runs_dir.iterdir()) if d.is_dir())
    doc = json.loads((run_dir / TRANSCRIPT_JSON).read_text(encoding="utf-8"))
    assert doc == {"schema_version": "1.0", "lang": "und", "segments": []}

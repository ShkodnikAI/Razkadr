"""OCR stage tests (naryad 009): candidate filter, VLM calls, collapsing.

Unit tests run on hand-written frames artifacts and tiny synthetic JPEGs
(openCV, no ffmpeg); the acceptance tests run the real ingest + frames
stages on the titles.mp4 fixture (naryad 034) with a scripted VLM that
returns the ground-truth strings. No network, no external APIs.

Measured reference points (titles.mp4 fixture, ffmpeg 7.1.5): Canny
edge density of blank frames 0.000000, caption frames 0.0150-0.0178 ->
default edge_density_threshold 0.005 separates them with a >= 3x margin.
"""

import json
import logging
from pathlib import Path

import pytest

from razkadr.cli import EXIT_OK, main
from razkadr.config import load_config
from razkadr.providers.base import AuthError, TransientError, UsageReporter
from razkadr.providers.layer import FACTORIES
from razkadr.runctx import RunContext
from razkadr.stages.frames import FRAMES_JSON
from razkadr.stages.ocr import (
    OCR_JSON,
    OcrError,
    collapse_runs,
    edge_density,
    load_ocr_prompt,
    ocr_params,
    parse_ocr_answer,
    plan_ocr_candidates,
    run_ocr,
    scene_key_frames,
)
from razkadr.validation import validate_document


class ScriptVLM(UsageReporter):
    """VLM fake with per-frame scripted answers; counts calls, can fail."""

    name = "script"
    task = "vlm"

    def __init__(
        self,
        answers: dict[str, str] | None = None,
        default: str = "",
        fail_first: int = 0,
        auth_error: str | None = None,
        model: str = "script-1",
    ) -> None:
        self.answers = dict(answers or {})
        self.default = default
        self.fail_first = fail_first
        self.auth_error = auth_error
        self.model = model
        self.calls = 0
        self.last_usage = []
        self.last_paths: list[Path] = []
        self.last_prompt: str | None = None

    def describe_image(self, path: Path, prompt: str) -> str:
        self.calls += 1
        self.last_paths.append(Path(path))
        self.last_prompt = prompt
        if self.auth_error:
            raise AuthError(self.auth_error)
        if self.calls <= self.fail_first:
            raise TransientError(f"fake transient failure #{self.calls}")
        # realistic usage report: one image + prompt tokens (like FakeVLM)
        self._report(
            [
                {"kind": "images", "value": 1},
                {"kind": "tokens", "value": float(len(prompt.split()))},
            ]
        )
        rel = "/".join(Path(path).parts[-2:])  # answer keys may use the run-relative path
        return self.answers.get(rel, self.answers.get(Path(path).name, self.default))


@pytest.fixture()
def fake_vlm_calls(monkeypatch):
    """Replace the fake VLM factory, capturing every created instance.

    The scripted answers come from the [vlm] section keys (test-only:
    answers/default/fail_first/auth_error; the factory reads them, the
    stage never does).
    """
    created: list[ScriptVLM] = []

    def factory(section: dict) -> ScriptVLM:
        provider = ScriptVLM(
            answers=section.get("answers"),
            default=str(section.get("default", "")),
            fail_first=int(section.get("fail_first", 0)),
            auth_error=section.get("auth_error"),
            model=str(section.get("model", "script-1")),
        )
        created.append(provider)
        return provider

    monkeypatch.setitem(FACTORIES["vlm"], "fake", factory)
    return created


@pytest.fixture()
def env(tmp_path):
    """RunContext + config with isolated runs and call cache dirs."""
    config = load_config(None)
    config["runs_dir"] = str(tmp_path / "runs")
    config["cache"] = {"dir": str(tmp_path / "calls")}
    ctx = RunContext(config["runs_dir"], config=config)
    return tmp_path, config, ctx


TEXT_VARIANTS = {"text": (16, 20), "text2": (8, 36)}  # (row, col) of the white block


def make_jpg(path: Path, kind: str) -> None:
    """Tiny deterministic JPEG: 'blank' (flat dark) or a white block variant.

    Measured densities on 64x48: blank 0.0, text blocks ~0.02 - the same
    >= 3x separation as the real titles.mp4 fixture. Variants differ in
    position (and therefore bytes), so identical-looking captions on two
    frames are still distinct provider inputs.
    """
    import cv2
    import numpy as np

    img = np.full((48, 64), 32, dtype=np.uint8)
    if kind in TEXT_VARIANTS:
        row, col = TEXT_VARIANTS[kind]
        img[row : row + 16, col : col + 24] = 230
    assert cv2.imwrite(str(path), img)


def frame_entry(i: int, t: float, scene_id: int, sharpness: float, kind: str, **extra) -> dict:
    return {
        "t": t,
        "file": f"frames/{i:06d}.jpg",
        "scene_id": scene_id,
        "sharpness": sharpness,
        "phash": f"{i:016x}",
        "exposure": 0.9,
        **extra,
    }


def write_frames(ctx: RunContext, entries: list[dict], kinds: dict[str, str]) -> dict:
    """Valid frames.json + the jpg files it references."""
    (ctx.dir / "frames").mkdir(exist_ok=True)
    for name, kind in kinds.items():
        make_jpg(ctx.dir / name, kind)
    data = {"schema_version": "1.1", "frames": entries}
    (ctx.dir / FRAMES_JSON).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return data


def kinds_for(entries: list[dict], text_files: dict[str, str]) -> dict[str, str]:
    """file -> jpg kind; frames not named in text_files are blank."""
    return {e["file"]: text_files.get(e["file"], "blank") for e in entries}


def default_entries() -> list[dict]:
    """4 accepted frames: 2 blank, 2 with text; scene 1's key frame is blank.

    f0 (t=0, blank, scene 0), f1 (t=2.5, text, scene 0, sharpest of 0),
    f2 (t=5.0, text, scene 0), f3 (t=7.5, blank, scene 1 -> key frame).
    """
    return [
        frame_entry(0, 0.0, 0, 0.900, "blank"),
        frame_entry(1, 2.5, 0, 0.990, "text"),
        frame_entry(2, 5.0, 0, 0.980, "text"),
        frame_entry(3, 7.5, 1, 0.970, "blank"),
    ]


def _read_json(ctx: RunContext, name: str) -> dict:
    return json.loads((ctx.dir / name).read_text(encoding="utf-8"))


# --- unit: params -------------------------------------------------------------------


def test_params_defaults():
    assert ocr_params({}) == {"max_ocr_frames": 60, "edge_density_threshold": 0.005, "lang": ""}


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_ocr_frames": -1},  # negative cap
        {"edge_density_threshold": -0.1},  # below 0
        {"edge_density_threshold": 1.5},  # above 1
        {"lang": "ru_RU!"},  # not a BCP-47 tag
    ],
)
def test_params_validation_errors(overrides):
    with pytest.raises(OcrError, match="max_ocr_frames|edge_density_threshold|lang"):
        ocr_params({"ocr": overrides})


def test_params_zero_limit_and_forced_lang_are_valid():
    params = ocr_params({"ocr": {"max_ocr_frames": 0, "lang": "ru"}})
    assert params == {"max_ocr_frames": 0, "edge_density_threshold": 0.005, "lang": "ru"}


# --- unit: versioned prompt ---------------------------------------------------------


def test_prompt_is_packaged_and_placeholder_free():
    prompt = load_ocr_prompt()
    assert "ocr.v1" in prompt
    assert "{{" not in prompt  # nothing to substitute: one prompt for every frame


# --- unit: edge density ---------------------------------------------------------------


def test_edge_density_blank_vs_text(tmp_path):
    make_jpg(tmp_path / "blank.jpg", "blank")
    make_jpg(tmp_path / "text.jpg", "text")
    blank = edge_density(tmp_path / "blank.jpg")
    text = edge_density(tmp_path / "text.jpg")
    assert blank < 0.001
    assert text > 0.01
    assert text >= 3 * max(blank, 1e-9)  # the separation the default threshold relies on


def test_edge_density_missing_file_is_a_clear_error(tmp_path):
    with pytest.raises(OcrError, match="cannot read frame file"):
        edge_density(tmp_path / "missing.jpg")


# --- unit: scene key frames -----------------------------------------------------------


def test_scene_key_frames_sharpest_per_scene_tie_earliest():
    entries = [
        frame_entry(0, 0.0, 0, 0.90, "blank"),
        frame_entry(1, 2.5, 0, 0.99, "text"),
        frame_entry(2, 5.0, 0, 0.99, "text"),  # sharpness tie with f1: earlier t wins
        frame_entry(3, 7.5, 1, 0.50, "blank"),
    ]
    assert scene_key_frames(entries) == {"frames/000001.jpg", "frames/000003.jpg"}


# --- unit: candidate plan ---------------------------------------------------------------


def test_plan_dense_first_then_key_frames_capped(tmp_path, env):
    _tmp, config, ctx = env
    entries = default_entries()
    write_frames(
        ctx,
        entries,
        kinds_for(entries, {"frames/000001.jpg": "text", "frames/000002.jpg": "text2"}),
    )
    params = ocr_params(config)

    plan = plan_ocr_candidates(_read_json(ctx, FRAMES_JSON), ctx.dir, params)
    # dense frames first (tie on density: t asc), then scene 1's blank key frame
    assert [p["file"] for p in plan] == [
        "frames/000001.jpg",
        "frames/000002.jpg",
        "frames/000003.jpg",
    ]

    capped = plan_ocr_candidates(
        _read_json(ctx, FRAMES_JSON), ctx.dir, {**params, "max_ocr_frames": 2}
    )
    assert [p["file"] for p in capped] == ["frames/000001.jpg", "frames/000002.jpg"]


def test_plan_skips_rejected_frames(tmp_path, env):
    _tmp, config, ctx = env
    entries = default_entries()
    entries[1] = frame_entry(1, 2.5, 0, 0.990, "text", rejected=True, reject_reason="blurry")
    write_frames(ctx, entries, kinds_for(entries, {"frames/000002.jpg": "text"}))
    params = ocr_params(config)

    plan = plan_ocr_candidates(_read_json(ctx, FRAMES_JSON), ctx.dir, params)
    # f1 rejected: f2 (text) is scene 0's sharpest accepted frame and dense,
    # f3 is scene 1's key frame; f0 is neither dense nor a key frame
    assert [p["file"] for p in plan] == ["frames/000002.jpg", "frames/000003.jpg"]


def test_plan_respects_zero_limit(tmp_path, env):
    _tmp, config, ctx = env
    entries = default_entries()
    write_frames(ctx, entries, kinds_for(entries, {"frames/000001.jpg": "text"}))
    params = ocr_params(config)
    assert (
        plan_ocr_candidates(_read_json(ctx, FRAMES_JSON), ctx.dir, {**params, "max_ocr_frames": 0})
        == []
    )


def test_plan_empty_without_accepted_frames(tmp_path, env):
    _tmp, config, ctx = env
    entries = [frame_entry(0, 0.0, 0, 0.1, "blank", rejected=True, reject_reason="blurry")]
    write_frames(ctx, entries, {"frames/000000.jpg": "blank"})
    assert plan_ocr_candidates(_read_json(ctx, FRAMES_JSON), ctx.dir, ocr_params(config)) == []


# --- unit: answer parsing and collapsing --------------------------------------------------


def test_parse_ocr_answer_strips_and_drops_empty_lines():
    assert parse_ocr_answer("  FLOUR 500 G \n\n\n МУКА 500 Г \n") == "FLOUR 500 G\nМУКА 500 Г"
    assert parse_ocr_answer("") == ""
    assert parse_ocr_answer("   \n  \n") == ""


def test_collapse_consecutive_same_text_keeps_first_frame():
    cands = [
        {"file": "frames/a.jpg", "t": 2.5, "density": 0.02},
        {"file": "frames/b.jpg", "t": 5.0, "density": 0.02},
    ]
    results = collapse_runs(
        cands, {"frames/a.jpg": "FLOUR 500 G", "frames/b.jpg": "FLOUR 500 G"}, ""
    )
    assert len(results) == 1  # duplicates collapsed
    assert results[0]["frame"] == "frames/a.jpg" and results[0]["t"] == 2.5  # interval start kept


def test_collapse_non_consecutive_same_text_stays_two_results():
    cands = [
        {"file": "frames/a.jpg", "t": 2.5, "density": 0.02},
        {"file": "frames/b.jpg", "t": 5.0, "density": 0.02},
        {"file": "frames/c.jpg", "t": 7.5, "density": 0.02},
    ]
    texts = {"frames/a.jpg": "FLOUR", "frames/b.jpg": "BAKE", "frames/c.jpg": "FLOUR"}
    results = collapse_runs(cands, texts, "")
    assert [r["t"] for r in results] == [2.5, 5.0, 7.5]  # other text in between breaks the run
    assert [r["text"] for r in results] == ["FLOUR", "BAKE", "FLOUR"]


def test_collapse_whitespace_normalized_case_sensitive():
    cands = [
        {"file": "frames/a.jpg", "t": 2.5, "density": 0.02},
        {"file": "frames/b.jpg", "t": 5.0, "density": 0.02},
    ]
    merged = collapse_runs(
        cands, {"frames/a.jpg": "FLOUR  500 G", "frames/b.jpg": "FLOUR 500 G\n"}, ""
    )
    assert len(merged) == 1 and merged[0]["text"] == "FLOUR  500 G"  # first answer kept verbatim
    split = collapse_runs(cands, {"frames/a.jpg": "flour 500 g", "frames/b.jpg": "FLOUR 500 G"}, "")
    assert len(split) == 2  # case differs: never merge different text


def test_collapse_empty_answers_break_runs_and_produce_nothing():
    cands = [
        {"file": "frames/a.jpg", "t": 2.5, "density": 0.02},
        {"file": "frames/b.jpg", "t": 5.0, "density": 0.02},
        {"file": "frames/c.jpg", "t": 7.5, "density": 0.02},
    ]
    texts = {"frames/a.jpg": "FLOUR", "frames/b.jpg": "", "frames/c.jpg": "FLOUR"}
    results = collapse_runs(cands, texts, "")
    assert [r["frame"] for r in results] == ["frames/a.jpg", "frames/c.jpg"]


def test_collapse_lang_forced_or_und():
    cands = [{"file": "frames/a.jpg", "t": 2.5, "density": 0.02}]
    assert collapse_runs(cands, {"frames/a.jpg": "МУКА"}, "")[0]["lang"] == "und"
    assert collapse_runs(cands, {"frames/a.jpg": "МУКА"}, "ru")[0]["lang"] == "ru"


# --- stage: happy path ---------------------------------------------------------------------


def test_stage_recognizes_collapses_and_accounts(env, fake_vlm_calls):
    _tmp, config, ctx = env
    config["vlm"] = {
        "provider": "fake",
        "model": "script-1",
        "answers": {"frames/000001.jpg": "FLOUR 500 G", "frames/000002.jpg": "FLOUR 500 G"},
    }
    entries = default_entries()
    write_frames(
        ctx,
        entries,
        kinds_for(entries, {"frames/000001.jpg": "text", "frames/000002.jpg": "text2"}),
    )

    doc = run_ocr(ctx, config)

    assert validate_document(doc, "ocr") == []
    # f1/f2 same text collapsed; f3 (blank key frame) answered empty -> nothing
    assert doc["results"] == [
        {"frame": "frames/000001.jpg", "t": 2.5, "text": "FLOUR 500 G", "lang": "und"}
    ]
    assert (ctx.dir / OCR_JSON).is_file()
    assert len(fake_vlm_calls) == 1 and fake_vlm_calls[0].calls == 3  # f1, f2, f3 (key frame)
    provider = fake_vlm_calls[0]
    assert provider.last_prompt == load_ocr_prompt()  # the versioned prompt goes to the VLM
    run = _read_json(ctx, "run.json")
    vlm_calls = [c for c in run["provider_calls"] if c["task"] == "vlm"]
    assert len(vlm_calls) == 3 and all(
        c["stage"] == "ocr" and c["cache_hit"] is False for c in vlm_calls
    )


def test_stage_summary_fields_and_stage_cache(env, fake_vlm_calls, caplog):
    _tmp, config, ctx = env
    config["vlm"] = {
        "provider": "fake",
        "model": "script-1",
        "answers": {"frames/000001.jpg": "FLOUR 500 G"},
    }
    entries = default_entries()
    write_frames(ctx, entries, kinds_for(entries, {"frames/000001.jpg": "text"}))

    with caplog.at_level(logging.INFO, logger="razkadr.ocr"):
        run_ocr(ctx, config)
    ocr_logs = [getattr(r, "fields", {}) for r in caplog.records if r.msg == "ocr done"]
    # sent: f1 (dense) + f3 (scene 1 key frame); f0 is neither
    assert ocr_logs == [{"sent": 2, "accepted": 4, "share": 0.5, "results": 1}]

    # second run on the same run dir: stage cached, no new provider calls
    ctx2 = RunContext(config["runs_dir"], run_id=ctx.run_id, config=config)
    doc2 = run_ocr(ctx2, config)
    assert doc2 == _read_json(ctx, OCR_JSON)
    assert fake_vlm_calls[0].calls == 2  # nothing re-sent, nothing re-billed
    statuses = {s["name"]: s["status"] for s in _read_json(ctx2, "run.json")["stages"]}
    assert statuses["ocr"] == "cached"


def test_stage_provider_switch_invalidates_cache(env, fake_vlm_calls):
    _tmp, config, ctx = env
    config["vlm"] = {
        "provider": "fake",
        "model": "script-1",
        "answers": {"frames/000001.jpg": "FLOUR 500 G"},
    }
    entries = default_entries()
    write_frames(ctx, entries, kinds_for(entries, {"frames/000001.jpg": "text"}))
    run_ocr(ctx, config)

    config["vlm"]["model"] = "script-2"
    run_ocr(ctx, config)
    assert len(fake_vlm_calls) == 2  # new service built and called again
    assert fake_vlm_calls[1].calls == 2


def test_stage_forced_lang(env, fake_vlm_calls):
    _tmp, config, ctx = env
    config["ocr"] = {"lang": "ru"}
    config["vlm"] = {
        "provider": "fake",
        "model": "script-1",
        "answers": {"frames/000001.jpg": "МУКА 500 Г"},
    }
    entries = default_entries()
    write_frames(ctx, entries, kinds_for(entries, {"frames/000001.jpg": "text"}))
    doc = run_ocr(ctx, config)
    assert doc["results"][0]["lang"] == "ru"


def test_stage_transient_error_is_retried(env, fake_vlm_calls):
    _tmp, config, ctx = env
    config["vlm"] = {"provider": "fake", "model": "script-1", "fail_first": 1, "base_delay_s": 0}
    entries = default_entries()
    write_frames(ctx, entries, kinds_for(entries, {"frames/000001.jpg": "text"}))
    doc = run_ocr(ctx, config)
    assert doc["results"] == []  # empty answers, but the retry happened
    # 2 frames sent (f1 dense, f3 key frame) + 1 retry of the first call
    assert fake_vlm_calls[0].calls == 3


# --- stage: empty results never touch the provider layer --------------------------------------


def test_zero_limit_empty_result_no_provider_calls(env, fake_vlm_calls):
    _tmp, config, ctx = env
    config["ocr"] = {"max_ocr_frames": 0}
    config["vlm"] = {"provider": "fake", "model": "script-1"}
    entries = default_entries()
    write_frames(ctx, entries, kinds_for(entries, {"frames/000001.jpg": "text"}))

    doc = run_ocr(ctx, config)

    assert doc == {"schema_version": "1.0", "results": []}
    assert validate_document(doc, "ocr") == []
    assert (ctx.dir / OCR_JSON).is_file()
    assert fake_vlm_calls == []  # the factory was never used: no cost at zero limit
    assert _read_json(ctx, "run.json")["provider_calls"] == []


def test_no_accepted_frames_empty_result_no_provider_calls(env, fake_vlm_calls):
    _tmp, config, ctx = env
    config["vlm"] = {"provider": "fake", "model": "script-1"}
    entries = [frame_entry(0, 0.0, 0, 0.1, "blank", rejected=True, reject_reason="blurry")]
    write_frames(ctx, entries, {"frames/000000.jpg": "blank"})
    doc = run_ocr(ctx, config)
    assert doc == {"schema_version": "1.0", "results": []}
    assert fake_vlm_calls == []


# --- stage: broken inputs and configuration fail loudly ----------------------------------------


def test_missing_frames_json_is_a_clear_error(env):
    _tmp, config, ctx = env
    with pytest.raises(OcrError, match="frames.json is missing"):
        run_ocr(ctx, config)


def test_corrupted_frames_json_is_a_clear_error(env):
    _tmp, config, ctx = env
    (ctx.dir / FRAMES_JSON).write_text("{oops", encoding="utf-8")
    with pytest.raises(OcrError, match="not valid JSON"):
        run_ocr(ctx, config)


def test_invalid_frames_artifact_is_a_clear_error(env):
    _tmp, config, ctx = env
    data = {"schema_version": "1.1", "frames": [{"t": 0.0}]}  # missing required fields
    (ctx.dir / FRAMES_JSON).write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(OcrError, match="not a valid frames artifact"):
        run_ocr(ctx, config)


def test_missing_frame_file_is_a_clear_error(env):
    _tmp, config, ctx = env
    config["vlm"] = {"provider": "fake", "model": "script-1"}
    entries = default_entries()
    (ctx.dir / FRAMES_JSON).write_text(
        json.dumps({"schema_version": "1.1", "frames": entries}, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(OcrError, match="cannot read frame file"):
        run_ocr(ctx, config)


def test_no_vlm_section_skips_with_empty_result_and_warning(env, fake_vlm_calls, caplog):
    """No [vlm] provider at all: the stage is skipped (chunks-fallback rule),
    a valid empty ocr.json is written, the reason is logged, no cost."""
    _tmp, config, ctx = env
    entries = default_entries()
    write_frames(ctx, entries, kinds_for(entries, {"frames/000001.jpg": "text"}))

    with caplog.at_level(logging.WARNING, logger="razkadr.ocr"):
        doc = run_ocr(ctx, config)

    assert doc == {"schema_version": "1.0", "results": []}
    assert validate_document(doc, "ocr") == []
    assert (ctx.dir / OCR_JSON).is_file()
    assert fake_vlm_calls == []  # the factory was never used
    assert _read_json(ctx, "run.json")["provider_calls"] == []
    reasons = [
        str(getattr(r, "fields", {}).get("reason", ""))
        for r in caplog.records
        if r.levelno == logging.WARNING
    ]
    assert "no [vlm] provider configured" in reasons


def test_unknown_vlm_provider_fails_the_stage(env):
    _tmp, config, ctx = env
    config["vlm"] = {"provider": "nope"}
    entries = default_entries()
    write_frames(ctx, entries, kinds_for(entries, {"frames/000001.jpg": "text"}))
    with pytest.raises(OcrError, match="unknown vlm provider"):
        run_ocr(ctx, config)


def test_vlm_auth_error_fails_the_stage(env, fake_vlm_calls):
    _tmp, config, ctx = env
    config["vlm"] = {"provider": "fake", "model": "script-1", "auth_error": "bad key"}
    entries = default_entries()
    write_frames(ctx, entries, kinds_for(entries, {"frames/000001.jpg": "text"}))
    with pytest.raises(AuthError):
        run_ocr(ctx, config)


# --- acceptance: titles.mp4 with a ground-truth VLM (naryad 009) --------------------------------


@pytest.fixture()
def titles_env(tmp_path, synthetic_fixtures):
    """ingest + frames already run on titles.mp4; ground-truth VLM answers.

    The scripted answer for every frame is the fixture's caption text when
    the frame time falls inside the caption's on-screen window, else "" -
    the fake returns exactly what a perfect OCR provider would see.
    """
    from razkadr.stages.frames import run_frames
    from razkadr.stages.ingest import run_ingest

    config = load_config(None)
    config["runs_dir"] = str(tmp_path / "runs")
    config["cache"] = {"dir": str(tmp_path / "calls")}
    ctx = RunContext(config["runs_dir"], config=config)
    run_ingest(ctx, str(synthetic_fixtures.path("titles.mp4")), config)
    run_frames(ctx, config)

    gt = synthetic_fixtures.ground_truth["titles"]["texts"]
    frames = json.loads((ctx.dir / FRAMES_JSON).read_text(encoding="utf-8"))["frames"]
    eps = 1e-6
    answers: dict[str, str] = {}
    for entry in frames:
        shown = [t["text"] for t in gt if t["t_start"] - eps <= entry["t"] <= t["t_end"] + eps]
        answers[Path(entry["file"]).name] = "\n".join(shown)
    return config, ctx, answers, gt


def test_titles_acceptance_ground_truth_texts_found_in_time(titles_env, fake_vlm_calls, caplog):
    """Acceptance: every ground-truth line found, time within +-1 s of its
    on-screen window; provider calls within max_ocr_frames; the share of
    frames sent to OCR is in the stage summary."""
    config, ctx, answers, gt = titles_env
    config["vlm"] = {"provider": "fake", "model": "gt-1", "answers": answers}

    with caplog.at_level(logging.INFO, logger="razkadr.ocr"):
        doc = run_ocr(ctx, config)

    assert validate_document(doc, "ocr") == []
    frames = _read_json(ctx, FRAMES_JSON)["frames"]
    accepted = [f for f in frames if not f.get("rejected")]
    n_calls = fake_vlm_calls[0].calls

    for item in gt:
        text, t0, t1 = item["text"], item["t_start"], item["t_end"]
        ts = [r["t"] for r in doc["results"] if r["text"] == text]
        assert ts, f"{text!r} not found in ocr.json"
        assert len(ts) == 1  # windows are disjoint: one collapsed result per caption
        assert t0 - 1.0 <= ts[0] <= t1 + 1.0, f"{text!r} at {ts[0]} outside [{t0}, {t1}] +-1 s"

    # call budget and the share of frames sent to OCR (stage summary)
    assert n_calls <= ocr_params(config)["max_ocr_frames"]
    ocr_logs = [getattr(r, "fields", {}) for r in caplog.records if r.msg == "ocr done"]
    assert ocr_logs == [
        {
            "sent": n_calls,
            "accepted": len(accepted),
            "share": round(n_calls / len(accepted), 3),
            "results": len(doc["results"]),
        }
    ]


def test_titles_cli_index_prints_ocr_share(tmp_path, synthetic_fixtures, fake_vlm_calls, capsys):
    """razkadr index runs the ocr stage and prints the sent-frames share."""
    config, _ctx, _answers, _gt = None, None, None, None  # noqa: F841 - fixtures self-contained
    toml = tmp_path / "razkadr.toml"
    toml.write_text('[vlm]\nprovider = "fake"\nmodel = "script-1"\n', encoding="utf-8")

    rc = main(
        [
            "index",
            str(synthetic_fixtures.path("titles.mp4")),
            "--config",
            str(toml),
            "--runs-dir",
            str(tmp_path / "runs"),
        ]
    )

    assert rc == EXIT_OK
    out = capsys.readouterr().out
    assert "ocr ok" in out
    ocr_line = next(line for line in out.splitlines() if line.strip().startswith("ocr:"))
    assert "frames sent" in ocr_line and "%" in ocr_line

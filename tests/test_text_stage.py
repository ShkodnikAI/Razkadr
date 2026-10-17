"""Text stage tests (naryad 017): verbatim / condensed / described.

All tests run on hand-written storyboard drafts and transcript artifacts
through the fake providers of the naryad 033 layer; no network, no
external APIs. The stage never trusts the LLM answer with the source
text: an answer that is empty, loses numbers or exceeds the limit is
truncated at a word boundary or falls back to verbatim with a warning -
the tests pin every branch, the user_edited preservation and the
degradation policy (no provider configured -> deterministic fallback,
configured-but-broken provider -> loud failure).
"""

import json
from pathlib import Path

import pytest

from razkadr.cli import EXIT_FILE_ERROR, EXIT_OK, main
from razkadr.config import load_config
from razkadr.providers.base import AuthError, UsageReporter
from razkadr.providers.layer import FACTORIES
from razkadr.providers.types import Message
from razkadr.runctx import RunContext
from razkadr.stages.text import (
    STORYBOARD_JSON,
    TEXT_JSON,
    TRANSCRIPT_JSON,
    TextError,
    condense_system_prompt,
    describe_system_prompt,
    dominant_lang,
    normalize_answer,
    numbers,
    numbers_preserved,
    resolve_mode,
    run_text,
    segments_in_span,
    text_params,
    truncate_at_word,
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


class ScriptVLM(UsageReporter):
    """VLM fake with a scripted answer; counts calls, can raise auth errors."""

    name = "script"
    model = "script-vision-1"
    task = "vlm"

    def __init__(self, answer: str, auth_error: str | None = None) -> None:
        self.answer = answer
        self.auth_error = auth_error
        self.calls = 0
        self.last_usage = []
        self.last_paths: list[Path] = []

    def describe_image(self, path: Path, prompt: str) -> str:
        self.calls += 1
        self.last_paths.append(path)
        if self.auth_error:
            raise AuthError(self.auth_error)
        self._report([{"kind": "images", "value": 1}, {"kind": "tokens", "value": float(len(prompt.split()))}])
        return self.answer


def seg(t_start: float, t_end: float, text: str, lang: str = "ru") -> dict:
    return {"t_start": t_start, "t_end": t_end, "text": text, "lang": lang}


def episode(ep_id: str, role: str, t_start: float, t_end: float, frame: str, **extra) -> dict:
    base = {
        "id": ep_id,
        "role": role,
        "source_id": "v1",
        "t_start": t_start,
        "t_end": t_end,
        "frame": {"path": frame, "t": (t_start + t_end) / 2},
        "alternatives": [],
    }
    base.update(extra)
    return base


def write_draft(ctx: RunContext, episodes: list[dict], output_lang: str = "ru") -> dict:
    storyboard = {
        "schema_version": "1.0",
        "project": {"id": "p", "title": "Блины", "preset": "recipe", "output_lang": output_lang},
        "sources": [{"id": "v1", "path": "normalized.mp4", "duration_s": 60.0, "detected_lang": "ru"}],
        "episodes": episodes,
    }
    (ctx.dir / STORYBOARD_JSON).write_text(json.dumps(storyboard, ensure_ascii=False), encoding="utf-8")
    return storyboard


def write_transcript(ctx: RunContext, segments: list[dict], lang: str = "ru") -> dict:
    data = {"schema_version": "1.0", "lang": lang, "segments": segments}
    (ctx.dir / TRANSCRIPT_JSON).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def make_frame(ctx: RunContext, name: str) -> str:
    path = ctx.dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\xff\xd8\xff\xe0fakejpeg")  # existence is what the stage checks
    return name


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
    """Replace the fake LLM factory, capturing every created instance."""
    created: list[ScriptLLM] = []

    def factory(section: dict) -> ScriptLLM:
        provider = ScriptLLM(str(section.get("answer", "сжатый текст")), auth_error=section.get("auth_error"))
        created.append(provider)
        return provider

    monkeypatch.setitem(FACTORIES["llm"], "fake", factory)
    return created


@pytest.fixture()
def fake_vlm_calls(monkeypatch):
    """Replace the fake VLM factory, capturing every created instance."""
    created: list[ScriptVLM] = []

    def factory(section: dict) -> ScriptVLM:
        provider = ScriptVLM(str(section.get("answer", "описание кадра")), auth_error=section.get("auth_error"))
        created.append(provider)
        return provider

    monkeypatch.setitem(FACTORIES["vlm"], "fake", factory)
    return created


def _read_json(ctx: RunContext, name: str) -> dict:
    return json.loads((ctx.dir / name).read_text(encoding="utf-8"))


# --- unit: params --------------------------------------------------------------------


def test_params_defaults():
    params = text_params({})
    assert params["mode"] == "auto"
    assert params["role_limits"]["hero"] == 120
    assert params["role_limits"]["ingredients"] == 480
    assert params["role_limits"]["step"] == 120
    assert params["default_limit"] == 180


def test_params_role_limit_override_keeps_other_defaults():
    params = text_params({"text": {"role_limits": {"hero": 60}}})
    assert params["role_limits"]["hero"] == 60
    assert params["role_limits"]["serving"] == 180


@pytest.mark.parametrize(
    "overrides",
    [
        {"mode": "magic"},
        {"default_limit": 0},
        {"default_limit": -1},
        {"role_limits": {"hero": 0}},
        {"role_limits": {"hero": -5}},
        {"role_limits": {"hero": True}},
        {"role_limits": {"hero": "60"}},
    ],
)
def test_params_validation_errors(overrides):
    with pytest.raises(TextError, match=r"mode|default_limit|role_limits"):
        text_params({"text": overrides})


# --- unit: prompts -------------------------------------------------------------------


def test_condense_prompt_carries_limit():
    prompt = condense_system_prompt(150)
    assert "150" in prompt
    assert "наряд №017" in prompt  # versioned file, packaged with the library


def test_describe_prompt_carries_limit_and_lang():
    prompt = describe_system_prompt(90, "ru")
    assert "90" in prompt and "{{max_chars}}" not in prompt
    assert "ru" in prompt and "{{lang}}" not in prompt


# --- unit: span, language, numbers, normalization ------------------------------------


def test_segments_in_span_overlaps_whole_segments():
    segments = [seg(0, 10, "a"), seg(10, 20, "b"), seg(20, 30, "c"), seg(30, 40, "d")]
    assert [s["text"] for s in segments_in_span(segments, 12, 22)] == ["b", "c"]
    assert [s["text"] for s in segments_in_span(segments, 0, 5)] == ["a"]
    assert segments_in_span(segments, 50, 60) == []


def test_segments_in_span_touching_segments_do_not_overlap():
    # a segment that ends exactly at t_start does not count (TIME_EPS guard)
    segments = [seg(0, 10, "a"), seg(10, 20, "b")]
    assert [s["text"] for s in segments_in_span(segments, 10, 20)] == ["b"]


def test_dominant_lang_most_frequent_first_on_tie():
    assert dominant_lang([seg(0, 1, "a"), seg(1, 2, "b", "en"), seg(2, 3, "c", "en")]) == "en"
    # tie (1-1): the first seen language wins
    assert dominant_lang([seg(0, 1, "a", "en"), seg(1, 2, "b")]) == "en"
    assert dominant_lang([seg(0, 1, "a"), seg(1, 2, "b", "en")]) == "ru"
    assert dominant_lang([]) is None


def test_numbers_extracts_digit_tokens():
    assert numbers("Варите 200 г риса 15 минут при 180°C") == ["200", "15", "180"]
    assert numbers("0,5 ч. л. соли и 1.5 стакана") == ["0,5", "1.5"]
    assert numbers("без чисел") == []


def test_numbers_preserved_with_decimal_separator_swap():
    assert numbers_preserved("0,5 ч. л.", "пол чайной ложки (0.5)") is True
    assert numbers_preserved("1.5 стакана", "полтора стакана") is False
    assert numbers_preserved("200 г муки", "мука") is False
    assert numbers_preserved("", "любой текст") is True


def test_normalize_answer_strips_quotes_and_whitespace():
    assert normalize_answer('  "сжатый   текст"\n') == "сжатый текст"
    assert normalize_answer("«ответ»") == "ответ"


def test_truncate_at_word_boundary():
    text = "одно два три четыре пять"
    assert truncate_at_word(text, 100) == text
    cut = truncate_at_word(text, 14)
    assert cut.startswith("одно два") and cut.endswith("…") and len(cut) <= 14


# --- unit: mode resolution -----------------------------------------------------------


@pytest.mark.parametrize(
    "wanted,has_speech,expected",
    [
        ("auto", True, "condensed"),
        ("auto", False, "described"),
        ("verbatim", True, "verbatim"),
        ("verbatim", False, "described"),
        ("condensed", True, "condensed"),
        ("condensed", False, "described"),
        ("described", True, "described"),
        ("described", False, "described"),
    ],
)
def test_resolve_mode_matrix(wanted, has_speech, expected):
    assert resolve_mode(wanted, has_speech) == expected


# --- stage: auto mode end to end -----------------------------------------------------


def test_auto_mode_speech_condensed_silent_described(env, fake_llm_calls, fake_vlm_calls):
    _, config, ctx = env
    config["llm"] = {"provider": "fake", "answer": "Вылейте тесто на сковороду."}
    config["vlm"] = {"provider": "fake", "answer": "Готовые блины на тарелке со сметаной."}
    make_frame(ctx, "frames/f1.jpg")
    make_frame(ctx, "frames/f2.jpg")
    write_draft(ctx, [
        episode("ep-steps", "steps", 10.0, 20.0, "frames/f1.jpg"),
        episode("ep-serving", "serving", 40.0, 50.0, "frames/f2.jpg"),
    ])
    write_transcript(ctx, [seg(0, 15, "Выливаем тесто на горячую сковороду."), seg(15, 30, "Жарим до золотистого цвета.")])

    storyboard, report = run_text(ctx, config)

    by_id = {e["id"]: e for e in storyboard["episodes"]}
    # speech in span -> condensed through the LLM
    text = by_id["ep-steps"]["text"]
    assert text["mode"] == "condensed"
    assert text["original"] == "Выливаем тесто на горячую сковороду. Жарим до золотистого цвета."
    assert text["output"] == "Вылейте тесто на сковороду."
    assert text["source_lang"] == "ru" and text["output_lang"] == "ru"
    # silent span -> described through the VLM
    text = by_id["ep-serving"]["text"]
    assert text["mode"] == "described"
    assert text["original"] == ""
    assert text["output"] == "Готовые блины на тарелке со сметаной."
    assert text["output_lang"] == "ru"
    assert validate_document(storyboard, "storyboard") == []
    modes = {e["id"]: e.get("text_mode") for e in report["episodes"]}
    assert modes == {"ep-steps": "condensed", "ep-serving": "described"}
    assert (ctx.dir / TEXT_JSON).is_file()


def test_report_carries_input_fingerprints_and_providers(env, fake_llm_calls, fake_vlm_calls):
    _, config, ctx = env
    config["llm"] = {"provider": "fake", "answer": "сжатие", "model": "script-model-llm"}
    config["vlm"] = {"provider": "fake", "answer": "кадр", "model": "script-model-vlm"}
    make_frame(ctx, "frames/f1.jpg")
    write_draft(ctx, [episode("ep-hero", "hero", 0.0, 10.0, "frames/f1.jpg")])
    write_transcript(ctx, [seg(0, 10, "текст")])

    import hashlib

    # the report fingerprints the INPUTS: capture them before the stage rewrites the draft
    sb_sha = hashlib.sha256((ctx.dir / STORYBOARD_JSON).read_bytes()).hexdigest()
    tr_sha = hashlib.sha256((ctx.dir / TRANSCRIPT_JSON).read_bytes()).hexdigest()

    _, report = run_text(ctx, config)

    assert report["inputs"]["storyboard_sha256"] == sb_sha
    assert report["inputs"]["transcript_sha256"] == tr_sha
    assert report["providers"]["llm"]["provider"] == "fake"
    assert report["providers"]["vlm"]["model"] == "script-model-vlm"


# --- stage: verbatim -----------------------------------------------------------------


def test_verbatim_mode_no_llm_calls(env, fake_llm_calls, fake_vlm_calls):
    _, config, ctx = env
    config["text"] = {"mode": "verbatim"}
    config["llm"] = {"provider": "fake", "answer": "LLM не должен зваться"}
    make_frame(ctx, "frames/f1.jpg")
    write_draft(ctx, [episode("ep-steps", "steps", 0.0, 20.0, "frames/f1.jpg")])
    write_transcript(ctx, [seg(0, 10, "Берем 2 яйца."), seg(10, 20, "Взбиваем с 200 г муки.")])

    storyboard, _ = run_text(ctx, config)

    text = storyboard["episodes"][0]["text"]
    assert text["mode"] == "verbatim"
    assert text["original"] == text["output"] == "Берем 2 яйца. Взбиваем с 200 г муки."
    assert fake_llm_calls == []  # verbatim never calls the LLM


def test_verbatim_over_limit_warns_but_keeps_speech(env, fake_llm_calls):
    _, config, ctx = env
    config["text"] = {"mode": "verbatim", "role_limits": {"steps": 10}}
    config["llm"] = {"provider": "fake"}
    make_frame(ctx, "frames/f1.jpg")
    write_draft(ctx, [episode("ep-steps", "steps", 0.0, 10.0, "frames/f1.jpg")])
    write_transcript(ctx, [seg(0, 10, "очень длинный дословный текст шага рецепта")])

    storyboard, report = run_text(ctx, config)

    assert storyboard["episodes"][0]["text"]["output"] == "очень длинный дословный текст шага рецепта"
    codes = [w["code"] for w in report["warnings"]]
    assert "verbatim_over_limit" in codes


# --- stage: condensed guards ----------------------------------------------------------


def test_condensed_answer_over_limit_truncated_with_warning(env, fake_llm_calls):
    _, config, ctx = env
    long_answer = "слово " * 60  # 360 chars, limit 120: valid text, over limit
    config["llm"] = {"provider": "fake", "answer": long_answer}
    make_frame(ctx, "frames/f1.jpg")
    write_draft(ctx, [episode("ep-steps", "steps", 0.0, 10.0, "frames/f1.jpg")])
    write_transcript(ctx, [seg(0, 10, "Жарим блины на сковороде до готовности.")])

    storyboard, report = run_text(ctx, config)

    text = storyboard["episodes"][0]["text"]
    assert text["mode"] == "condensed"
    assert len(text["output"]) <= 120 and text["output"].endswith("…")
    assert any(w["code"] == "text_over_limit_truncated" for w in report["warnings"])


def test_condensed_losing_numbers_falls_back_to_verbatim(env, fake_llm_calls):
    _, config, ctx = env
    # the answer drops "200" and "15" the original carries
    config["llm"] = {"provider": "fake", "answer": "Жарьте блины до готовности."}
    make_frame(ctx, "frames/f1.jpg")
    write_draft(ctx, [episode("ep-steps", "steps", 0.0, 10.0, "frames/f1.jpg")])
    write_transcript(ctx, [seg(0, 10, "Жарим блины: 200 г теста 15 минут.")])

    storyboard, report = run_text(ctx, config)

    text = storyboard["episodes"][0]["text"]
    assert text["mode"] == "verbatim"  # numbers hold by construction
    assert "200" in text["output"] and "15" in text["output"]
    codes = [w["code"] for w in report["warnings"]]
    assert "condensed_fallback_verbatim" in codes
    entry = next(e for e in report["episodes"] if e["id"] == "ep-steps")
    assert "numbers" in entry["fallback"]


def test_condensed_empty_answer_falls_back_to_verbatim(env, fake_llm_calls):
    _, config, ctx = env
    config["llm"] = {"provider": "fake", "answer": "   "}
    make_frame(ctx, "frames/f1.jpg")
    write_draft(ctx, [episode("ep-steps", "steps", 0.0, 10.0, "frames/f1.jpg")])
    write_transcript(ctx, [seg(0, 10, "текст шага")])

    storyboard = run_text(ctx, config)[0]

    text = storyboard["episodes"][0]["text"]
    assert text["mode"] == "verbatim" and text["output"] == "текст шага"
    assert any(w["code"] == "condensed_fallback_verbatim" for w in _read_json(ctx, TEXT_JSON)["warnings"])


def test_condensed_no_llm_provider_falls_back_to_verbatim(env, fake_llm_calls):
    _, config, ctx = env  # no [llm] section at all
    make_frame(ctx, "frames/f1.jpg")
    write_draft(ctx, [episode("ep-steps", "steps", 0.0, 10.0, "frames/f1.jpg")])
    write_transcript(ctx, [seg(0, 10, "текст шага с числом 5")])

    storyboard, report = run_text(ctx, config)

    text = storyboard["episodes"][0]["text"]
    assert text["mode"] == "verbatim" and text["output"] == "текст шага с числом 5"
    assert any(w["code"] == "condensed_fallback_verbatim" for w in report["warnings"])
    assert fake_llm_calls == []


def test_condensed_llm_receives_limit_in_system_prompt(env, fake_llm_calls):
    _, config, ctx = env
    config["llm"] = {"provider": "fake", "answer": "ответ"}
    make_frame(ctx, "frames/f1.jpg")
    write_draft(ctx, [episode("ep-ingredients", "ingredients", 0.0, 10.0, "frames/f1.jpg")])
    write_transcript(ctx, [seg(0, 10, "ингредиенты")])

    run_text(ctx, config)

    provider = fake_llm_calls[0]
    assert provider.calls == 1
    system = provider.last_messages[0].content
    assert "480" in system  # the ingredients role limit
    assert provider.last_messages[1].content == "ингредиенты"


# --- stage: described -----------------------------------------------------------------


def test_described_forced_even_with_speech(env, fake_llm_calls, fake_vlm_calls):
    _, config, ctx = env
    config["text"] = {"mode": "described"}
    config["vlm"] = {"provider": "fake", "answer": "Кадр: тесто на сковороде."}
    config["llm"] = {"provider": "fake"}
    make_frame(ctx, "frames/f1.jpg")
    write_draft(ctx, [episode("ep-steps", "steps", 0.0, 10.0, "frames/f1.jpg")])
    write_transcript(ctx, [seg(0, 10, "текст шага")])

    storyboard, _ = run_text(ctx, config)

    text = storyboard["episodes"][0]["text"]
    assert text["mode"] == "described"
    assert text["output"] == "Кадр: тесто на сковороде."
    assert text["original"] == ""
    assert fake_llm_calls == [] and len(fake_vlm_calls) == 1


def test_described_no_vlm_provider_episode_keeps_no_text(env, fake_llm_calls):
    _, config, ctx = env
    config["llm"] = {"provider": "fake", "answer": "сжатие"}
    make_frame(ctx, "frames/f1.jpg")
    write_draft(ctx, [
        episode("ep-steps", "steps", 10.0, 20.0, "frames/f1.jpg"),
        episode("ep-serving", "serving", 40.0, 50.0, "frames/f1.jpg"),  # silent span
    ])
    write_transcript(ctx, [seg(10, 20, "текст шага")])

    storyboard, report = run_text(ctx, config)

    by_id = {e["id"]: e for e in storyboard["episodes"]}
    assert "text" in by_id["ep-steps"]  # speech episode got its text
    assert "text" not in by_id["ep-serving"]  # silent one keeps no text, no crash
    assert any(w["code"] == "no_vlm_provider" for w in report["warnings"])
    assert validate_document(storyboard, "storyboard") == []


def test_described_missing_frame_file_skips_episode_with_warning(env, fake_vlm_calls):
    _, config, ctx = env
    config["vlm"] = {"provider": "fake", "answer": "кадр"}
    write_draft(ctx, [episode("ep-serving", "serving", 40.0, 50.0, "frames/nope.jpg")])
    write_transcript(ctx, [])

    storyboard, report = run_text(ctx, config)

    assert "text" not in storyboard["episodes"][0]
    assert any(w["code"] == "frame_missing" for w in report["warnings"])
    assert fake_vlm_calls == []


def test_described_output_over_limit_truncated(env, fake_vlm_calls):
    _, config, ctx = env
    config["vlm"] = {"provider": "fake", "answer": "слово " * 40}
    make_frame(ctx, "frames/f1.jpg")
    write_draft(ctx, [episode("ep-serving", "serving", 40.0, 50.0, "frames/f1.jpg")])
    write_transcript(ctx, [])

    storyboard = run_text(ctx, config)[0]

    text = storyboard["episodes"][0]["text"]
    assert len(text["output"]) <= 180 and text["output"].endswith("…")

# --- stage: user edits and input contract ---------------------------------------------


def test_user_edited_episode_text_is_preserved(env, fake_llm_calls, fake_vlm_calls):
    _, config, ctx = env
    config["llm"] = {"provider": "fake", "answer": "сжатие"}
    config["vlm"] = {"provider": "fake", "answer": "кадр"}
    make_frame(ctx, "frames/f1.jpg")
    edited_text = {"mode": "verbatim", "source_lang": "ru", "original": "моя правка", "output": "моя правка", "output_lang": "ru"}
    write_draft(ctx, [
        episode("ep-steps", "steps", 0.0, 10.0, "frames/f1.jpg", user_edited=True, text=edited_text),
        episode("ep-serving", "serving", 40.0, 50.0, "frames/f1.jpg"),
    ])
    write_transcript(ctx, [seg(0, 10, "текст")])

    storyboard, report = run_text(ctx, config)

    by_id = {e["id"]: e for e in storyboard["episodes"]}
    assert by_id["ep-steps"]["text"] == edited_text
    assert by_id["ep-serving"]["text"]["mode"] == "described"
    assert fake_llm_calls == []  # the edited episode never reaches the LLM
    assert any(e.get("kept") for e in report["episodes"])


def test_user_edited_without_text_still_gets_text(env, fake_llm_calls):
    _, config, ctx = env
    config["llm"] = {"provider": "fake", "answer": "сжатый ответ"}
    make_frame(ctx, "frames/f1.jpg")
    write_draft(ctx, [episode("ep-steps", "steps", 0.0, 10.0, "frames/f1.jpg", user_edited=True)])
    write_transcript(ctx, [seg(0, 10, "текст")])

    storyboard = run_text(ctx, config)[0]

    # the user edited the frame, the text did not exist yet: it is produced
    assert storyboard["episodes"][0]["text"]["output"] == "сжатый ответ"


def test_missing_storyboard_fails(env):
    _, config, ctx = env
    write_transcript(ctx, [])
    with pytest.raises(TextError, match="storyboard.json is missing"):
        run_text(ctx, config)


def test_missing_transcript_fails(env):
    _, config, ctx = env
    write_draft(ctx, [])
    with pytest.raises(TextError, match="transcript.json is missing"):
        run_text(ctx, config)


def test_invalid_storyboard_fails(env):
    _, config, ctx = env
    (ctx.dir / STORYBOARD_JSON).write_text(json.dumps({"schema_version": "1.0"}), encoding="utf-8")
    write_transcript(ctx, [])
    with pytest.raises(TextError, match="not a valid Storyboard"):
        run_text(ctx, config)


# --- stage: broken provider is loud ---------------------------------------------------


def test_auth_error_from_llm_fails_the_stage(env, monkeypatch):
    _, config, ctx = env
    config["llm"] = {"provider": "fake", "answer": "не будет", "auth_error": "401 no key"}

    def factory(section):
        return ScriptLLM(section.get("answer", ""), auth_error=section.get("auth_error"))

    monkeypatch.setitem(FACTORIES["llm"], "fake", factory)
    make_frame(ctx, "frames/f1.jpg")
    write_draft(ctx, [episode("ep-steps", "steps", 0.0, 10.0, "frames/f1.jpg")])
    write_transcript(ctx, [seg(0, 10, "текст шага")])

    with pytest.raises(AuthError):
        run_text(ctx, config)
    # nothing was written: the storyboard keeps its draft shape
    assert "text" not in json.loads((ctx.dir / STORYBOARD_JSON).read_text(encoding="utf-8"))["episodes"][0]


def test_unknown_llm_provider_name_fails_loudly(env):
    _, config, ctx = env
    config["llm"] = {"provider": "no-such-provider"}
    make_frame(ctx, "frames/f1.jpg")
    write_draft(ctx, [episode("ep-steps", "steps", 0.0, 10.0, "frames/f1.jpg")])
    write_transcript(ctx, [seg(0, 10, "текст шага")])

    with pytest.raises(TextError, match=r"\[llm\]"):
        run_text(ctx, config)


# --- CLI ------------------------------------------------------------------------------


def test_cli_text_end_to_end(env, fake_llm_calls, fake_vlm_calls, capsys):
    _, _, ctx = env
    config_path = ctx.runs_dir.parent / "razkadr.toml"
    config_path.write_text(
        '[llm]\nprovider = "fake"\nanswer = "Вылейте тесто."\n'
        '[vlm]\nprovider = "fake"\nanswer = "Блины на тарелке."\n',
        encoding="utf-8",
    )
    make_frame(ctx, "frames/f1.jpg")
    make_frame(ctx, "frames/f2.jpg")
    write_draft(ctx, [
        episode("ep-steps", "steps", 10.0, 20.0, "frames/f1.jpg"),
        episode("ep-serving", "serving", 40.0, 50.0, "frames/f2.jpg"),
    ])
    write_transcript(ctx, [seg(10, 20, "Выливаем тесто на сковороду.")])

    code = main(["text", str(ctx.dir), "--config", str(config_path)])
    assert code == EXIT_OK
    out = capsys.readouterr().out
    assert "text:" in out and "condensed 1" in out and "described 1" in out

    doc = _read_json(ctx, STORYBOARD_JSON)
    by_id = {e["id"]: e for e in doc["episodes"]}
    assert by_id["ep-steps"]["text"]["output"] == "Вылейте тесто."
    assert by_id["ep-serving"]["text"]["output"] == "Блины на тарелке."
    assert _read_json(ctx, TEXT_JSON)["episodes"]


def test_cli_text_mode_override(env, fake_llm_calls, capsys):
    _, _, ctx = env
    config_path = ctx.runs_dir.parent / "razkadr.toml"
    config_path.write_text('[llm]\nprovider = "fake"\nanswer = "не зваться"\n', encoding="utf-8")
    make_frame(ctx, "frames/f1.jpg")
    write_draft(ctx, [episode("ep-steps", "steps", 0.0, 10.0, "frames/f1.jpg")])
    write_transcript(ctx, [seg(0, 10, "дословный текст шага")])

    code = main(["text", str(ctx.dir), "--config", str(config_path), "--mode", "verbatim"])
    assert code == EXIT_OK
    doc = _read_json(ctx, STORYBOARD_JSON)
    assert doc["episodes"][0]["text"]["mode"] == "verbatim"
    assert doc["episodes"][0]["text"]["output"] == "дословный текст шага"


def test_cli_text_missing_run_dir_exits_2(capsys):
    assert main(["text", "/nonexistent/run"]) == EXIT_FILE_ERROR
    assert "run directory not found" in capsys.readouterr().err


def test_cli_text_stage_error_exits_2(env, capsys):
    _, _, ctx = env
    config_path = ctx.runs_dir.parent / "empty.toml"
    config_path.write_text("", encoding="utf-8")
    write_transcript(ctx, [])  # storyboard.json missing -> stage error

    assert main(["text", str(ctx.dir), "--config", str(config_path)]) == EXIT_FILE_ERROR
    assert "razkadr text:" in capsys.readouterr().err

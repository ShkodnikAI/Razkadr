"""Provider layer tests (naryad 033): fakes, cache, tariffs, keys, errors.

All tests run without network; the "unusual" key in masking tests mixes
regex metacharacters, quotes and non-ASCII to try to break the masker.
"""

import io
import json
from pathlib import Path

import pytest

from razkadr.config import load_config
from razkadr.log import SECRETS, get_logger, setup_logging
from razkadr.providers import (
    AuthError,
    Message,
    ProviderConfigError,
    ProviderError,
    ProviderLayer,
    QuotaError,
    TransientError,
)
from razkadr.providers.cache import call_key
from razkadr.providers.fakes import FakeASR
from razkadr.providers.layer import Throttle
from razkadr.providers.tariffs import Tariff
from razkadr.runctx import RunContext
from razkadr.validation import validate_document

UNUSUAL_KEY = "sk-\"'`$();drop--αβγ🤖[a-z].*"


def write_config(tmp_path: Path, **overrides: str) -> Path:
    lines = [
        "[keys]",
        'mode = "user"',
        "",
        "[cache]",
        f'dir = "{tmp_path / "cache"}"',
        "",
        "[asr]",
        'provider = "fake"',
        'model = "fake-nova-2"',
        "[asr.tariff]",
        'units = "seconds"',
        "price = 0.0059",
        'currency = "USD"',
        'tariff_date = "2026-10-01"',
        "",
        "[embeddings]",
        'provider = "fake"',
        'model = "fake-embed-v2"',
        "[embeddings.tariff]",
        'units = "images"',
        "price = 0.0001",
        'currency = "USD"',
        'tariff_date = "2026-10-01"',
        "",
        "[vlm]",
        'provider = "fake"',
        'model = "fake-vision-1"',
        "",
        "[llm]",
        'provider = "fake"',
        'model = "fake-gpt-4o-mini"',
    ]
    lines += ["", *overrides.get("extra_lines", [])]
    path = tmp_path / "razkadr.toml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """RunContext + layer built from a temp config; isolated cache and runs."""
    config_path = write_config(tmp_path)
    config = load_config(config_path)
    ctx = RunContext(tmp_path / "runs", config=config)
    layer = ProviderLayer(config, ctx)
    return tmp_path, config, ctx, layer


# --- contract: all four tasks on fakes, no network ----------------------------

def test_contract_example_all_tasks_on_fakes(env):
    tmp_path, _config, ctx, layer = env
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fake audio bytes")
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"fake image bytes")

    transcript = layer.asr().transcribe(audio, lang="ru")
    assert transcript.lang == "ru" and len(transcript.segments) == 2

    vectors = layer.embeddings().embed_images([image])
    assert len(vectors) == 1 and len(vectors[0]) == 8
    vectors2 = layer.embeddings().embed_texts(["hello world"])
    assert len(vectors2[0]) == 8

    description = layer.vlm().describe_image(image, "what is on the screen?")
    assert description.startswith("fake description")

    answer = layer.llm().complete([Message(role="user", content="hi")])
    assert answer.startswith("fake completion")

    data = json.loads((ctx.dir / "run.json").read_text(encoding="utf-8"))
    assert validate_document(data, "run") == []
    calls = data["provider_calls"]
    assert [c["task"] for c in calls] == ["asr", "embed", "embed", "vlm", "llm"]
    for call in calls:
        assert set(call) >= {"stage", "provider", "model", "units", "cost", "cache_hit"}
        assert call["cache_hit"] is False
    assert calls[0]["units"] == [{"kind": "seconds", "value": 10.0}]
    assert calls[0]["cost"] == {"amount": 0.059, "currency": "USD", "tariff_date": "2026-10-01"}


# --- provider switching via config only ---------------------------------------

def test_provider_switch_by_config(tmp_path):
    config_path = write_config(tmp_path)
    text = config_path.read_text(encoding="utf-8").replace(
        "[asr]\nprovider = \"fake\"", "[asr]\nprovider = \"fake-alt\""
    )
    config_path.write_text(text, encoding="utf-8")

    config = load_config(config_path)
    ctx = RunContext(tmp_path / "runs", config=config)
    layer = ProviderLayer(config, ctx)

    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    assert layer.asr().transcribe(audio).lang == "ru"  # same stage code, other provider

    data = json.loads((ctx.dir / "run.json").read_text(encoding="utf-8"))
    assert data["provider_calls"][0]["provider"] == "fake-alt"
    assert data["provider_calls"][0]["model"] == "fake-nova-2"  # model from config unchanged


def test_unknown_provider_clear_error(tmp_path):
    config_path = write_config(tmp_path)
    text = config_path.read_text(encoding="utf-8").replace(
        "[asr]\nprovider = \"fake\"", "[asr]\nprovider = \"mystery\""
    )
    config_path.write_text(text, encoding="utf-8")
    config = load_config(config_path)
    ctx = RunContext(tmp_path / "runs", config=config)
    with pytest.raises(ProviderConfigError, match="unknown asr provider 'mystery'"):
        ProviderLayer(config, ctx).asr()


# --- cache: second call does not reach the provider ---------------------------

def test_repeat_call_served_from_cache(env):
    _tmp_path, _config, ctx, layer = env
    service = layer.embeddings()
    provider = service.provider
    texts = ["one", "two"]

    first = service.embed_texts(texts)
    assert provider.calls == 1
    second = service.embed_texts(texts)
    assert provider.calls == 1, "repeated call must not reach the provider"
    assert second == first

    data = json.loads((ctx.dir / "run.json").read_text(encoding="utf-8"))
    assert validate_document(data, "run") == []
    hit = data["provider_calls"][1]
    assert hit["cache_hit"] is True
    assert hit["cost"]["amount"] == 0.0
    assert hit["units"] == [{"kind": "tokens", "value": 2.0}]


def test_cache_key_includes_model_version_and_params(tmp_path):
    a = call_key("embed", "fake", "m", {"texts": ["x"]}, {})
    b = call_key("embed", "fake", "m", {"texts": ["x"]}, {"model_version": "2"})
    c = call_key("embed", "fake", "other-model", {"texts": ["x"]}, {})
    d = call_key("embed", "fake", "m", {"texts": ["y"]}, {})
    assert len({a, b, c, d}) == 4


def test_model_version_change_bypasses_cache(env):
    _tmp_path, config, ctx, layer = env
    service = layer.embeddings()
    service.embed_texts(["same text"])
    assert service.provider.calls == 1

    config["embeddings"]["model_version"] = "3"
    layer2 = ProviderLayer(config, ctx)
    service2 = layer2.embeddings()
    service2.embed_texts(["same text"])
    assert service2.provider.calls == 1, "model_version bump must invalidate the cache"


# --- secrets: never in logs, artifacts, exceptions ----------------------------

def test_key_never_leaks_anywhere(tmp_path, monkeypatch):
    config_path = write_config(tmp_path)
    key_toml = json.dumps(UNUSUAL_KEY, ensure_ascii=False)  # TOML basic string with escapes
    text = config_path.read_text(encoding="utf-8").replace(
        "[asr]\n", f"[asr]\napi_key = {key_toml}\n"
    )
    config_path.write_text(text, encoding="utf-8")

    monkeypatch.setenv("RAZKADR_TEST_TOKEN", UNUSUAL_KEY)
    config = load_config(config_path)
    assert UNUSUAL_KEY not in json.dumps(config), "secret must not stay in the effective config"

    ctx = RunContext(tmp_path / "runs", config=config)
    layer = ProviderLayer(config, ctx)

    buf = io.StringIO()
    setup_logging(stream=buf)
    log = get_logger("probe")
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    try:
        try:
            layer.asr().transcribe(audio)
        except ProviderError as exc:
            assert UNUSUAL_KEY not in str(exc), "exception text must be masked"
        log.info("probe with %s and env %s", UNUSUAL_KEY, UNUSUAL_KEY)
    finally:
        setup_logging(stream=None)

    assert UNUSUAL_KEY not in buf.getvalue()
    assert "[REDACTED]" in buf.getvalue()
    for path in ctx.dir.rglob("*"):
        if path.is_file():
            content = path.read_text(encoding="utf-8")
            assert UNUSUAL_KEY not in content, f"secret leaked into {path.name}"


def test_key_from_env_missing_is_clear_error(tmp_path, monkeypatch):
    config_path = write_config(tmp_path)
    text = config_path.read_text(encoding="utf-8").replace(
        "[asr]\n", '[asr]\napi_key_env = "ASR_API_KEY"\n'
    )
    config_path.write_text(text, encoding="utf-8")
    monkeypatch.delenv("ASR_API_KEY", raising=False)
    config = load_config(config_path)
    ctx = RunContext(tmp_path / "runs", config=config)
    with pytest.raises(ProviderConfigError, match="export ASR_API_KEY"):
        ProviderLayer(config, ctx).asr()


def test_key_from_env_used(tmp_path, monkeypatch):
    config_path = write_config(tmp_path)
    text = config_path.read_text(encoding="utf-8").replace(
        "[asr]\n", '[asr]\napi_key_env = "ASR_API_KEY"\n'
    )
    config_path.write_text(text, encoding="utf-8")
    monkeypatch.setenv("ASR_API_KEY", UNUSUAL_KEY)
    config = load_config(config_path)
    ctx = RunContext(tmp_path / "runs", config=config)
    ProviderLayer(config, ctx).asr()  # builds without error; key registered for masking


def test_keys_mode_validated(tmp_path):
    config_path = write_config(tmp_path)
    text = config_path.read_text(encoding="utf-8").replace('mode = "user"', 'mode = "weird"')
    config_path.write_text(text, encoding="utf-8")
    config = load_config(config_path)
    ctx = RunContext(tmp_path / "runs", config=config)
    with pytest.raises(ProviderConfigError, match="mode"):
        ProviderLayer(config, ctx)


# --- cost accounting: table test ----------------------------------------------

@pytest.mark.parametrize(
    "units_kind,units_value,price,expected",
    [
        ("seconds", 42.0, 0.0059, 0.2478),
        ("seconds", 0.0, 0.0059, 0.0),
        ("tokens", 15360, 0.0000002, 0.003072),
        ("images", 96, 0.0001, 0.0096),
        ("images", 3, 0.0, 0.0),
    ],
)
def test_tariff_cost_table(units_kind, units_value, price, expected):
    tariff = Tariff({"units": units_kind, "price": price, "currency": "USD", "tariff_date": "2026-10-01"})
    units = [{"kind": units_kind, "value": units_value}]
    assert tariff.cost(units) == expected


def test_tariff_ignores_other_unit_kinds():
    tariff = Tariff({"units": "seconds", "price": 0.01, "currency": "USD", "tariff_date": "2026-10-01"})
    units = [{"kind": "images", "value": 100}]
    assert tariff.cost(units) == 0.0


def test_no_tariff_section_means_zero_cost_with_marker(env):
    tmp_path, _config, ctx, layer = env  # [vlm] has no tariff section by design
    layer2 = ProviderLayer(config=layer.config, runctx=ctx)
    image = tmp_path / "f.jpg"
    image.write_bytes(b"img")
    layer2.vlm().describe_image(image, "p")
    data = json.loads((ctx.dir / "run.json").read_text(encoding="utf-8"))
    vlm_call = next(c for c in data["provider_calls"] if c["task"] == "vlm")
    assert vlm_call["cost"] == {"amount": 0.0, "currency": "USD", "tariff_date": "1970-01-01"}


# --- retries, auth, quota ------------------------------------------------------

def test_transient_errors_retried_then_success(tmp_path, monkeypatch):
    config_path = write_config(tmp_path)
    config = load_config(config_path)
    ctx = RunContext(tmp_path / "runs", config=config)
    sleeps = []
    provider = FakeASR(fail_first=2)
    service = ProviderLayer(config, ctx).asr()
    service.sleeper = lambda s: sleeps.append(s)  # patch service retry sleeper
    service.provider = provider
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")

    transcript = service.transcribe(audio)
    assert provider.calls == 3
    assert len(transcript.segments) == 2
    assert sleeps == [1.0, 2.0]  # base_delay * 2^(attempt-1)


def test_transient_errors_exhausted_clear_message(tmp_path):
    config_path = write_config(tmp_path)
    config = load_config(config_path)
    ctx = RunContext(tmp_path / "runs", config=config)
    service = ProviderLayer(config, ctx).asr()
    service.provider = FakeASR(fail_first=10)
    service.sleeper = lambda s: None
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    with pytest.raises(TransientError, match="failed after 3 attempts"):
        service.transcribe(audio)


def test_auth_and_quota_not_retried_and_masked(tmp_path):
    SECRETS.register(UNUSUAL_KEY)  # the layer knows the key it authenticated with
    config_path = write_config(tmp_path)
    config = load_config(config_path)
    ctx = RunContext(tmp_path / "runs", config=config)
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")

    auth_service = ProviderLayer(config, ctx).asr(stage="auth-probe")
    auth_service.provider = FakeASR(auth_error=f"401 for key {UNUSUAL_KEY}")
    auth_service.sleeper = lambda s: None
    with pytest.raises(AuthError) as excinfo:
        auth_service.transcribe(audio)
    assert UNUSUAL_KEY not in str(excinfo.value)
    assert "[REDACTED]" in str(excinfo.value)
    assert auth_service.provider.calls == 1, "auth errors must not be retried"

    quota_service = ProviderLayer(config, ctx).asr(stage="quota-probe")
    quota_service.provider = FakeASR(quota_error="quota exhausted, pay at ...")
    with pytest.raises(QuotaError, match="quota exhausted"):
        quota_service.transcribe(audio)
    assert quota_service.provider.calls == 1


# --- throttle -------------------------------------------------------------------

def test_throttle_respects_requests_per_minute():
    now = {"t": 100.0}
    slept = []
    throttle = Throttle(requests_per_minute=60, clock=lambda: now["t"], sleeper=slept.append)
    throttle.wait()
    now["t"] = 100.2  # 0.2 s later; min interval is 1 s
    throttle.wait()
    assert slept == pytest.approx([0.8])


def test_throttle_positive_rate_required():
    with pytest.raises(ValueError):
        Throttle(requests_per_minute=0, clock=lambda: 0.0, sleeper=lambda s: None)

"""Real ASR adapter tests (naryad 037): hermetic, no network, no keys.

Fixtures under tests/fixtures/adapters/ are SYNTHETIC responses modeled on
the official API documentation formats (Gemini generateContent, Deepgram
/v1/listen); they contain no personal data and no keys. Live tests are
marked ``live`` and skipped unless the provider key is set in the
environment; they must be run manually with real keys (owner-side).
"""

from __future__ import annotations

import json
import os
import wave
from pathlib import Path

import pytest

from razkadr.providers.base import (
    AuthError,
    ProviderConfigError,
    ProviderError,
    QuotaError,
    TransientError,
)
from razkadr.providers.deepgram import DeepgramASR
from razkadr.providers.gemini import GeminiASR, normalize_lang
from razkadr.providers.layer import FACTORIES, ProviderLayer, resolve_key
from razkadr.providers.rest import Request, Response
from razkadr.providers.types import Transcript
from razkadr.runctx import RunContext

FIXTURES = Path(__file__).parent / "fixtures" / "adapters"


def fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


class StubTransport:
    """Hermetic transport: returns canned responses, records requests."""

    def __init__(self, status: int = 200, body: bytes = b"{}", calls_before_error: int = 0) -> None:
        self.status = status
        self.body = body
        self.requests: list[Request] = []
        self.calls_before_error = calls_before_error

    def __call__(self, request: Request) -> Response:
        self.requests.append(request)
        if self.calls_before_error and len(self.requests) <= self.calls_before_error:
            return Response(status=500, body=b'{"error": {"code": 500, "message": "internal"}}')
        return Response(status=self.status, body=self.body)


def make_wav(path: Path, seconds: float = 0.5, rate: int = 16000) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * int(rate * seconds))
    return path


def layer_with_asr(
    tmp_path: Path, provider: str, model: str, monkeypatch: pytest.MonkeyPatch, transport=None
) -> tuple[ProviderLayer, Path]:
    """Build a ProviderLayer with a real adapter name over a stub transport."""
    monkeypatch.setenv("ASR_TEST_KEY", "k-test-0000000000")
    config = {
        "keys": {"mode": "user"},
        "cache": {"dir": str(tmp_path / "cache")},
        "asr": {
            "provider": provider,
            "model": model,
            "api_key_env": "ASR_TEST_KEY",
            "max_attempts": 1,
            "base_delay_s": 0.0,
            "requests_per_minute": 10000,
            "tariff": {"units": "seconds", "price": 0.003, "currency": "USD", "tariff_date": "2026-10-16"},
        },
    }
    if transport is not None:
        monkeypatch.setattr(
            "razkadr.providers.layer.FACTORIES",
            {
                **FACTORIES,
                "asr": {
                    **FACTORIES["asr"],
                    provider: lambda s, p=provider, t=transport: (
                        GeminiASR(model=s.get("model", "gemini-3.5-transcribe"), api_key=resolve_key(s, "asr") or "", transport=t)
                        if p == "gemini"
                        else DeepgramASR(model=s.get("model", "nova-3"), api_key=resolve_key(s, "asr") or "", transport=t)
                    ),
                },
            },
        )
    runctx = RunContext(runs_dir=str(tmp_path))
    return ProviderLayer(config, runctx), runctx.dir


# --- shared: lang normalization -------------------------------------------------


def test_normalize_lang_primary_subtag() -> None:
    assert normalize_lang("ru-RU") == "ru"
    assert normalize_lang("en-US") == "en"
    assert normalize_lang("ru") == "ru"
    assert normalize_lang("") == "und"
    assert normalize_lang(None) == "und"
    assert normalize_lang("   ") == "und"
    assert normalize_lang("!!!", fallback="qaa") == "qaa"


# --- gemini ---------------------------------------------------------------------


def test_gemini_happy_path_parses_transcript(tmp_path: Path) -> None:
    transport = StubTransport(body=fixture("gemini_transcribe_ok.json"))
    adapter = GeminiASR(api_key="k", transport=transport)
    transcript = adapter.transcribe(make_wav(tmp_path / "a.wav"), lang=None)
    assert isinstance(transcript, Transcript)
    assert transcript.lang == "ru"
    assert [s.text for s in transcript.segments] == [
        "Две ложки муки в миску.",
        "Разогрейте сковороду на среднем огне.",
    ]
    assert transcript.segments[1].t_start == pytest.approx(2.4)
    # usage billed in seconds of audio (WAV header: 0.5 s)
    assert adapter.last_usage == [{"kind": "seconds", "value": 0.5}]


def test_gemini_request_shape_key_in_header_not_url(tmp_path: Path) -> None:
    transport = StubTransport(body=fixture("gemini_transcribe_ok.json"))
    adapter = GeminiASR(api_key="secret-key-123", transport=transport)
    adapter.transcribe(make_wav(tmp_path / "a.wav"), lang="ru")
    request = transport.requests[0]
    assert request.headers["x-goog-api-key"] == "secret-key-123"
    assert "secret-key-123" not in request.url
    assert request.url.endswith("/gemini-3.5-transcribe:generateContent")
    payload = json.loads(request.body)
    assert payload["generationConfig"]["response_mime_type"] == "application/json"
    assert payload["generationConfig"]["temperature"] == 0
    inline = payload["contents"][0]["parts"][1]["inline_data"]
    assert inline["mime_type"] == "audio/wav"


def test_gemini_lang_from_param_when_response_lang_empty(tmp_path: Path) -> None:
    body = json.dumps({"candidates": [{"content": {"parts": [{"text": '{"lang": "", "segments": [{"t_start": 0.0, "t_end": 1.0, "text": "x"}]}'}]}}]}).encode()
    adapter = GeminiASR(api_key="k", transport=StubTransport(body=body))
    transcript = adapter.transcribe(make_wav(tmp_path / "a.wav"), lang="de")
    assert transcript.lang == "de"
    assert transcript.segments[0].lang == "de"


def test_gemini_malformed_segments_dropped_not_invented(tmp_path: Path) -> None:
    body = json.dumps({
        "candidates": [{"content": {"parts": [{"text": json.dumps({
            "lang": "ru",
            "segments": [
                {"t_start": 0.0, "t_end": 1.0, "text": "ok"},
                {"t_start": 2.0, "t_end": 1.0, "text": "reverted timecodes"},
                {"t_start": 3.0, "t_end": 4.0, "text": "   "},
                {"t_start": "x", "t_end": 4.0, "text": "bad types"},
            ],
        })}]}}]
    }).encode()
    adapter = GeminiASR(api_key="k", transport=StubTransport(body=body))
    transcript = adapter.transcribe(make_wav(tmp_path / "a.wav"))
    assert [s.text for s in transcript.segments] == ["ok"]


def test_gemini_error_mapping(tmp_path: Path) -> None:
    audio = make_wav(tmp_path / "a.wav")
    cases = [
        (401, "gemini_auth_error.json", AuthError),
        (429, "gemini_quota_error.json", QuotaError),
        (500, "gemini_server_error.json", TransientError),
    ]
    for status, fixture_name, expected in cases:
        adapter = GeminiASR(api_key="k", transport=StubTransport(status=status, body=fixture(fixture_name)))
        with pytest.raises(expected):
            adapter.transcribe(audio)


def test_gemini_malformed_response_shape(tmp_path: Path) -> None:
    adapter = GeminiASR(api_key="k", transport=StubTransport(body=fixture("gemini_malformed_text.json")))
    with pytest.raises(ProviderError, match="unexpected response shape"):
        adapter.transcribe(make_wav(tmp_path / "a.wav"))


def test_gemini_transport_network_failure_is_transient(tmp_path: Path) -> None:
    def boom(request: Request) -> Response:
        raise TimeoutError("connect timeout")

    adapter = GeminiASR(api_key="k", transport=boom)
    with pytest.raises(TransientError, match="transport failure"):
        adapter.transcribe(make_wav(tmp_path / "a.wav"))


# --- deepgram -------------------------------------------------------------------


def test_deepgram_happy_path_from_paragraphs(tmp_path: Path) -> None:
    transport = StubTransport(body=fixture("deepgram_listen_ok.json"))
    adapter = DeepgramASR(api_key="k", transport=transport)
    transcript = adapter.transcribe(make_wav(tmp_path / "a.wav"), lang=None)
    assert isinstance(transcript, Transcript)
    assert transcript.lang == "ru"
    assert [s.text for s in transcript.segments] == [
        "Две ложки муки в миску.",
        "Разогрейте сковороду на среднем огне.",
    ]
    assert transcript.segments[0].t_start == pytest.approx(0.0)
    assert transcript.segments[1].t_end == pytest.approx(5.1)
    # billing: provider metadata.duration (5.2), not the WAV header (0.5)
    assert adapter.last_usage == [{"kind": "seconds", "value": 5.2}]


def test_deepgram_request_shape_token_header(tmp_path: Path) -> None:
    transport = StubTransport(body=fixture("deepgram_listen_ok.json"))
    adapter = DeepgramASR(api_key="dg-key-456", transport=transport)
    adapter.transcribe(make_wav(tmp_path / "a.wav"), lang=None)
    request = transport.requests[0]
    assert request.headers["Authorization"] == "Token dg-key-456"
    assert "dg-key-456" not in request.url
    assert "model=nova-3" in request.url
    assert "language=multi" in request.url
    assert request.headers["Content-Type"] == "audio/wav"


def test_deepgram_word_grouping_fallback(tmp_path: Path) -> None:
    """No paragraphs -> words grouped by pauses >= 0.8 s."""
    transport = StubTransport(body=fixture("deepgram_words_gap.json"))
    adapter = DeepgramASR(api_key="k", transport=transport)
    transcript = adapter.transcribe(make_wav(tmp_path / "a.wav"))
    assert transcript.lang == "en"
    assert [s.text for s in transcript.segments] == [
        "Add two cups of flour.",
        "Stir until smooth.",
    ]
    assert transcript.segments[1].t_start == pytest.approx(2.6)


def test_deepgram_error_mapping(tmp_path: Path) -> None:
    audio = make_wav(tmp_path / "a.wav")
    cases = [
        (401, "deepgram_auth_error.json", AuthError),
        (429, "deepgram_quota_error.json", QuotaError),
        (500, b'{"errCode": "ServerError", "errMsg": "boom"}', TransientError),
    ]
    for status, body, expected in cases:
        payload = body if isinstance(body, bytes) else fixture(body)
        adapter = DeepgramASR(api_key="k", transport=StubTransport(status=status, body=payload))
        with pytest.raises(expected):
            adapter.transcribe(audio)


def test_deepgram_unexpected_shape(tmp_path: Path) -> None:
    adapter = DeepgramASR(api_key="k", transport=StubTransport(body=b'{"results": {}}'))
    with pytest.raises(ProviderError, match="unexpected response shape"):
        adapter.transcribe(make_wav(tmp_path / "a.wav"))


# --- layer integration: config-driven build, cache, cost accounting --------------


def test_layer_builds_gemini_and_records_cost(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    transport = StubTransport(body=fixture("gemini_transcribe_ok.json"))
    layer, run_dir = layer_with_asr(tmp_path, "gemini", "gemini-3.5-transcribe", monkeypatch, transport)
    audio = make_wav(tmp_path / "a.wav")
    transcript = layer.asr().transcribe(audio, lang=None)
    assert isinstance(transcript, Transcript)
    # cost: 0.5 s * 0.003 USD/s = 0.0015, rounded to 6 decimals
    calls = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["provider_calls"]
    assert len(calls) == 1
    assert calls[0]["task"] == "asr" and calls[0]["provider"] == "gemini"
    assert calls[0]["cache_hit"] is False
    assert calls[0]["units"] == [{"kind": "seconds", "value": 0.5}]
    assert calls[0]["cost"]["amount"] == pytest.approx(0.0015, abs=1e-9)


def test_layer_gemini_cached_call_cost_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    transport = StubTransport(body=fixture("gemini_transcribe_ok.json"))
    layer, run_dir = layer_with_asr(tmp_path, "gemini", "gemini-3.5-transcribe", monkeypatch, transport)
    audio = make_wav(tmp_path / "a.wav")
    layer.asr().transcribe(audio)
    layer.asr().transcribe(audio)
    calls = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["provider_calls"]
    assert [c["cache_hit"] for c in calls] == [False, True]
    assert len(transport.requests) == 1  # second call served from cache


def test_layer_missing_key_is_config_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ASR_MISSING_KEY", raising=False)
    config = {
        "cache": {"dir": str(tmp_path / "cache")},
        "asr": {"provider": "gemini", "model": "gemini-3.5-transcribe", "api_key_env": "ASR_MISSING_KEY"},
    }
    runctx = RunContext(runs_dir=str(tmp_path))
    with pytest.raises(ProviderConfigError, match="ASR_MISSING_KEY"):
        ProviderLayer(config, runctx).asr()


def test_factories_registered() -> None:
    assert "gemini" in FACTORIES["asr"]
    assert "deepgram" in FACTORIES["asr"]


# --- live tests (manual, owner-side with real keys) ------------------------------


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="GEMINI_API_KEY is not set")
def test_live_gemini_transcribe(tmp_path: Path) -> None:
    adapter = GeminiASR(api_key=os.environ["GEMINI_API_KEY"])
    transcript = adapter.transcribe(make_wav(tmp_path / "live.wav", seconds=1.0), lang="ru")
    assert isinstance(transcript, Transcript)
    assert adapter.last_usage[0]["kind"] == "seconds"


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("DEEPGRAM_API_KEY"), reason="DEEPGRAM_API_KEY is not set")
def test_live_deepgram_transcribe(tmp_path: Path) -> None:
    adapter = DeepgramASR(api_key=os.environ["DEEPGRAM_API_KEY"])
    transcript = adapter.transcribe(make_wav(tmp_path / "live.wav", seconds=1.0), lang="ru")
    assert isinstance(transcript, Transcript)
    assert adapter.last_usage[0]["kind"] == "seconds"

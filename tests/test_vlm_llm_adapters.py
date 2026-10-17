"""Real VLM and LLM adapter tests (naryad 039): hermetic, no network.

Fixtures are SYNTHETIC responses modeled on the official API formats
(Gemini generateContent, OpenAI /v1/chat/completions). Live tests are
marked ``live`` and skipped unless the provider key is set.
Prompt-version recording in run.json is stage-owned (naryads 008/009):
tests reference the existing stage behavior; adapters themselves never
see the prompt catalog.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from razkadr.providers.base import AuthError, ProviderConfigError, ProviderError, TransientError
from razkadr.providers.gemini_vlm import GeminiVLM
from razkadr.providers.layer import FACTORIES, ProviderLayer, resolve_key
from razkadr.providers.openai_llm import OpenAILLM
from razkadr.providers.rest import Request, Response
from razkadr.providers.types import Message
from razkadr.runctx import RunContext

FIXTURES = Path(__file__).parent / "fixtures" / "adapters"


def fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


class StubTransport:
    """Canned responses in order; when exhausted, repeats the last one."""

    def __init__(self, responses: list[Response]) -> None:
        self.responses = responses
        self.requests: list[Request] = []

    def __call__(self, request: Request) -> Response:
        self.requests.append(request)
        return self.responses[min(len(self.requests) - 1, len(self.responses) - 1)]


def ok_response(body: bytes) -> Response:
    return Response(status=200, body=body)


def make_jpg(tmp_path: Path, width: int = 800, height: int = 600) -> Path:
    rng = np.random.default_rng(7)
    frame = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
    path = tmp_path / "frame.jpg"
    cv2.imwrite(str(path), frame)
    return path


def layer_with_task(
    tmp_path: Path, task: str, section: str, monkeypatch: pytest.MonkeyPatch, transport=None, extra=None
) -> tuple[ProviderLayer, Path]:
    monkeypatch.setenv("TASK_TEST_KEY", "k-test-0000000000")
    section_cfg = {
        "provider": "gemini" if task == "vlm" else "openai",
        "api_key_env": "TASK_TEST_KEY",
        "max_attempts": 1,
        "base_delay_s": 0.0,
        "requests_per_minute": 10000,
        "tariff": {"units": "tokens", "price": 0.001, "currency": "USD", "tariff_date": "2026-10-16"},
    }
    section_cfg.update(extra or {})
    config = {
        "keys": {"mode": "user"},
        "cache": {"dir": str(tmp_path / "cache")},
        section: section_cfg,
    }
    if transport is not None:
        monkeypatch.setattr(
            "razkadr.providers.layer.FACTORIES",
            {
                **FACTORIES,
                task: {
                    **FACTORIES[task],
                    section_cfg["provider"]: lambda s, t=task, tr=transport: (
                        GeminiVLM(
                            model=s.get("model", "gemini-3.5-flash-lite"),
                            api_key=resolve_key(s, "vlm") or "",
                            transport=tr,
                            json_mode=bool(s.get("json_mode", False)),
                        )
                        if t == "vlm"
                        else OpenAILLM(
                            model=s.get("model", "gpt-6-luna"),
                            api_key=resolve_key(s, "llm") or "",
                            transport=tr,
                            json_mode=bool(s.get("json_mode", False)),
                        )
                    ),
                },
            },
        )
    runctx = RunContext(runs_dir=str(tmp_path))
    return ProviderLayer(config, runctx), runctx.dir


# --- gemini VLM -------------------------------------------------------------------


def test_gemini_vlm_describe(tmp_path: Path) -> None:
    transport = StubTransport([ok_response(fixture("gemini_vlm_ok.json"))])
    adapter = GeminiVLM(api_key="k", transport=transport)
    image = make_jpg(tmp_path)
    text = adapter.describe_image(image, "Что на экране?")
    assert text == "Ингредиенты: 2 яйца\nСоль по вкусу"
    assert adapter.last_usage == [{"kind": "images", "value": 1}, {"kind": "tokens", "value": 338.0}]


def test_gemini_vlm_image_sent_downscaled(tmp_path: Path) -> None:
    transport = StubTransport([ok_response(fixture("gemini_vlm_ok.json"))])
    adapter = GeminiVLM(api_key="k", transport=transport, max_image_dim=256, max_image_bytes=200_000)
    import base64

    adapter.describe_image(make_jpg(tmp_path, width=3000, height=2000), "prompt")
    parts = json.loads(transport.requests[0].body)["contents"][0]["parts"]
    raw = base64.b64decode(parts[1]["inline_data"]["data"])
    decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    assert max(decoded.shape[0], decoded.shape[1]) <= 256
    assert len(raw) <= 200_000


def test_gemini_vlm_key_in_header_not_url(tmp_path: Path) -> None:
    transport = StubTransport([ok_response(fixture("gemini_vlm_ok.json"))])
    GeminiVLM(api_key="gk-secret-1", transport=transport).describe_image(make_jpg(tmp_path), "p")
    request = transport.requests[0]
    assert request.headers["x-goog-api-key"] == "gk-secret-1"
    assert "gk-secret-1" not in request.url
    assert request.url.endswith("/gemini-3.5-flash-lite:generateContent")


def test_gemini_vlm_temperature_zero_and_auth_error(tmp_path: Path) -> None:
    transport = StubTransport([ok_response(fixture("gemini_vlm_ok.json"))])
    GeminiVLM(api_key="k", transport=transport).describe_image(make_jpg(tmp_path), "p")
    assert json.loads(transport.requests[0].body)["generationConfig"]["temperature"] == 0
    auth = GeminiVLM(api_key="k", transport=StubTransport([Response(status=401, body=b"{}")]))
    with pytest.raises(AuthError):
        auth.describe_image(make_jpg(tmp_path), "p")


def test_gemini_vlm_malformed_shape(tmp_path: Path) -> None:
    adapter = GeminiVLM(api_key="k", transport=StubTransport([ok_response(b'{"candidates": []}')]))
    with pytest.raises(ProviderError, match="unexpected response shape"):
        adapter.describe_image(make_jpg(tmp_path), "p")


# --- openai LLM -------------------------------------------------------------------


def test_openai_llm_complete() -> None:
    transport = StubTransport([ok_response(fixture("openai_chat_ok.json"))])
    adapter = OpenAILLM(api_key="k", transport=transport)
    reply = adapter.complete([Message(role="system", content="Return JSON"), Message(role="user", content="текст")])
    assert reply == '{"boundaries": [0, 2, 5]}'
    assert adapter.last_usage == [{"kind": "tokens", "value": 222.0}]
    body = json.loads(transport.requests[0].body)
    assert body["temperature"] == 0
    assert body["messages"][0]["role"] == "system"


def test_openai_llm_key_in_header_not_url() -> None:
    transport = StubTransport([ok_response(fixture("openai_chat_ok.json"))])
    OpenAILLM(api_key="sk-llm-1", transport=transport).complete([Message(role="user", content="q")])
    request = transport.requests[0]
    assert request.headers["Authorization"] == "Bearer sk-llm-1"
    assert "sk-llm-1" not in request.url


def test_openai_llm_error_mapping() -> None:
    for status, expected in ((401, AuthError), (500, TransientError)):
        adapter = OpenAILLM(api_key="k", transport=StubTransport([Response(status=status, body=b"{}")]))
        with pytest.raises(expected):
            adapter.complete([Message(role="user", content="q")])


# --- JSON mode: invalid reply -> retry -> clear error (naryad 039 acceptance) ------


def test_json_mode_retries_then_succeeds() -> None:
    bad = ok_response(json.dumps({
        "choices": [{"message": {"content": "not json at all"}}], "usage": {"prompt_tokens": 5, "completion_tokens": 2}
    }).encode())
    good = ok_response(fixture("openai_chat_ok.json"))
    transport = StubTransport([bad, bad, good])
    adapter = OpenAILLM(api_key="k", transport=transport, json_mode=True, json_retries=2)
    reply = adapter.complete([Message(role="user", content="JSON please")])
    assert reply == '{"boundaries": [0, 2, 5]}'
    assert len(transport.requests) == 3  # two invalid replies retried


def test_json_mode_fails_clearly_after_retries() -> None:
    bad = ok_response(json.dumps({"choices": [{"message": {"content": "still not json"}}]}).encode())
    transport = StubTransport([bad])
    adapter = OpenAILLM(api_key="k", transport=transport, json_mode=True, json_retries=2)
    with pytest.raises(ProviderError, match="not valid JSON after 3 attempts"):
        adapter.complete([Message(role="user", content="JSON please")])
    assert len(transport.requests) == 3


def test_json_mode_valid_json_not_retried() -> None:
    transport = StubTransport([ok_response(fixture("openai_chat_ok.json"))])
    adapter = OpenAILLM(api_key="k", transport=transport, json_mode=True)
    adapter.complete([Message(role="user", content="JSON please")])
    assert len(transport.requests) == 1


def test_gemini_json_mode_forces_mime_and_retries() -> None:
    bad = ok_response(json.dumps({"candidates": [{"content": {"parts": [{"text": "oops"}]}}]}).encode())
    good = ok_response(json.dumps({"candidates": [{"content": {"parts": [{"text": '{"ok": 1}'}]}}], "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 3}}).encode())
    transport = StubTransport([bad, good])
    adapter = GeminiVLM(api_key="k", transport=transport, json_mode=True, json_retries=1)
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        reply = adapter.describe_image(make_jpg(Path(td), width=100, height=80), "p")
    assert reply == '{"ok": 1}'
    assert adapter.last_usage == [{"kind": "images", "value": 1}, {"kind": "tokens", "value": 13.0}]
    assert len(transport.requests) == 2
    assert json.loads(transport.requests[0].body)["generationConfig"]["response_mime_type"] == "application/json"


def test_json_mode_off_does_not_validate() -> None:
    transport = StubTransport([ok_response(json.dumps({"choices": [{"message": {"content": "plain text"}}]}).encode())])
    adapter = OpenAILLM(api_key="k", transport=transport, json_mode=False)
    assert adapter.complete([Message(role="user", content="q")]) == "plain text"
    assert len(transport.requests) == 1


# --- layer integration ------------------------------------------------------------


def test_layer_builds_gemini_vlm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    transport = StubTransport([ok_response(fixture("gemini_vlm_ok.json"))])
    layer, run_dir = layer_with_task(tmp_path, "vlm", "vlm", monkeypatch, transport)
    text = layer.vlm().describe_image(make_jpg(tmp_path), "Что на экране?")
    assert "Ингредиенты" in text
    calls = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["provider_calls"]
    assert calls[0]["task"] == "vlm" and calls[0]["provider"] == "gemini"
    assert calls[0]["cache_hit"] is False


def test_layer_builds_openai_llm_with_json_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    transport = StubTransport([ok_response(fixture("openai_chat_ok.json"))])
    layer, run_dir = layer_with_task(
        tmp_path, "llm", "llm", monkeypatch, transport, extra={"json_mode": True, "model": "gpt-6-luna"}
    )
    reply = layer.llm().complete([Message(role="user", content="JSON please")])
    assert json.loads(reply) == {"boundaries": [0, 2, 5]}
    calls = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["provider_calls"]
    assert calls[0]["task"] == "llm" and calls[0]["provider"] == "openai"


def test_layer_missing_llm_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_MISSING_KEY", raising=False)
    config = {
        "cache": {"dir": str(tmp_path / "cache")},
        "llm": {"provider": "openai", "model": "gpt-6-luna", "api_key_env": "LLM_MISSING_KEY"},
    }
    runctx = RunContext(runs_dir=str(tmp_path))
    with pytest.raises(ProviderConfigError, match="LLM_MISSING_KEY"):
        ProviderLayer(config, runctx).llm()


def test_factories_registered() -> None:
    assert "gemini" in FACTORIES["vlm"]
    assert "openai" in FACTORIES["llm"]


# --- prompt versioning is stage-owned; sanity-check the packaged catalog ----------


def test_prompt_catalog_versions_and_no_secrets() -> None:
    from importlib import resources

    root = resources.files("razkadr.prompts")
    names = sorted(p.name for p in root.iterdir() if p.name.endswith(".md"))
    assert names, "prompt catalog is empty"
    assert all(".v" in name and name.endswith(".md") for name in names)
    for name in names:
        text = root.joinpath(name).read_text(encoding="utf-8")
        assert "sk-" not in text and "api_key" not in text  # auditor: prompts hold no secrets


# --- live tests (manual, owner-side with real keys) ---------------------------------


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="GEMINI_API_KEY is not set")
def test_live_gemini_vlm(tmp_path: Path) -> None:
    adapter = GeminiVLM(api_key=os.environ["GEMINI_API_KEY"])
    text = adapter.describe_image(make_jpg(tmp_path, width=320, height=240), "Опиши изображение одной строкой.")
    assert isinstance(text, str) and len(text) > 0


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY is not set")
def test_live_openai_llm() -> None:
    adapter = OpenAILLM(api_key=os.environ["OPENAI_API_KEY"], json_mode=True)
    reply = adapter.complete([Message(role="user", content='Return JSON {"ok": true} exactly.')])
    assert json.loads(reply)["ok"] is True

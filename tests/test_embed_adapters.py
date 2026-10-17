"""Real embeddings adapter tests (naryad 038): hermetic, no network.

Fixtures are SYNTHETIC responses modeled on the official API formats
(OpenAI /v1/embeddings, Cohere /v2/embed). Live tests are marked ``live``
and skipped unless the provider key is set; they run owner-side with
real keys.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from razkadr.providers.base import (
    AuthError,
    ProviderConfigError,
    ProviderError,
    QuotaError,
    TransientError,
)
from razkadr.providers.cohere import CohereEmbeddings
from razkadr.providers.images import prepare_image_jpeg
from razkadr.providers.layer import FACTORIES, ProviderLayer, resolve_key
from razkadr.providers.openai import OpenAIEmbeddings
from razkadr.providers.rest import Request, Response
from razkadr.runctx import RunContext

FIXTURES = Path(__file__).parent / "fixtures" / "adapters"


def fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


class StubTransport:
    def __init__(self, status: int = 200, body: bytes = b"{}") -> None:
        self.status = status
        self.body = body
        self.requests: list[Request] = []

    def __call__(self, request: Request) -> Response:
        self.requests.append(request)
        return Response(status=self.status, body=self.body)


def make_png(tmp_path: Path, width: int = 640, height: int = 480, noise: bool = True) -> Path:
    rng = np.random.default_rng(42)
    frame = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8) if noise else np.zeros((height, width, 3), np.uint8)
    path = tmp_path / f"img_{width}x{height}.png"
    cv2.imwrite(str(path), frame)
    return path


def layer_with_embed(
    tmp_path: Path, provider: str, monkeypatch: pytest.MonkeyPatch, transport=None
) -> tuple[ProviderLayer, Path]:
    monkeypatch.setenv("EMBED_TEST_KEY", "k-test-0000000000")
    config = {
        "keys": {"mode": "user"},
        "cache": {"dir": str(tmp_path / "cache")},
        "embeddings": {
            "provider": provider,
            "model": "text-embedding-3-small" if provider == "openai" else "embed-v4.0",
            "api_key_env": "EMBED_TEST_KEY",
            "max_attempts": 1,
            "base_delay_s": 0.0,
            "requests_per_minute": 10000,
            "batch_size": 16,
            "tariff": {"units": "tokens", "price": 0.001, "currency": "USD", "tariff_date": "2026-10-16"},
        },
    }
    if transport is not None:
        monkeypatch.setattr(
            "razkadr.providers.layer.FACTORIES",
            {
                **FACTORIES,
                "embed": {
                    **FACTORIES["embed"],
                    provider: lambda s, p=provider, t=transport: (
                        OpenAIEmbeddings(
                            model=s.get("model", "text-embedding-3-small"),
                            api_key=resolve_key(s, "embeddings") or "",
                            transport=t,
                        )
                        if p == "openai"
                        else CohereEmbeddings(
                            model=s.get("model", "embed-v4.0"),
                            api_key=resolve_key(s, "embeddings") or "",
                            transport=t,
                        )
                    ),
                },
            },
        )
    runctx = RunContext(runs_dir=str(tmp_path))
    return ProviderLayer(config, runctx), runctx.dir


# --- image preparation (acceptance: downscale verified by a test) ---------------


def test_prepare_image_downscales_long_side(tmp_path: Path) -> None:
    path = make_png(tmp_path, width=3000, height=1000)
    data = prepare_image_jpeg(path, max_dim=1024, max_bytes=4_000_000)
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert max(decoded.shape[0], decoded.shape[1]) <= 1024


def test_prepare_image_enforces_byte_limit(tmp_path: Path) -> None:
    path = make_png(tmp_path, width=4000, height=4000)
    data = prepare_image_jpeg(path, max_dim=1024, max_bytes=120_000)
    assert len(data) <= 120_000
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert max(decoded.shape[0], decoded.shape[1]) <= 1024


def test_prepare_image_deterministic(tmp_path: Path) -> None:
    path = make_png(tmp_path, width=800, height=600)
    assert prepare_image_jpeg(path) == prepare_image_jpeg(path)


def test_prepare_image_unreadable_file(tmp_path: Path) -> None:
    path = tmp_path / "not-an-image.png"
    path.write_bytes(b"garbage")
    with pytest.raises(ProviderError, match="cannot read image"):
        prepare_image_jpeg(path)


def test_prepare_image_rejects_tiny_limits(tmp_path: Path) -> None:
    path = make_png(tmp_path, width=64, height=64)
    with pytest.raises(ProviderError, match="unreasonably small"):
        prepare_image_jpeg(path, max_dim=8, max_bytes=512)


# --- openai (text-only, ADR-0006 default path) -----------------------------------


def test_openai_embed_texts_orders_by_index_and_bills_tokens() -> None:
    transport = StubTransport(body=fixture("openai_embeddings_ok.json"))
    adapter = OpenAIEmbeddings(api_key="k", transport=transport)
    vectors = adapter.embed_texts(["первый", "второй"])
    assert vectors == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]  # ordered by index, not response order
    assert adapter.last_usage == [{"kind": "tokens", "value": 24.0}]


def test_openai_request_shape_key_in_header(tmp_path: Path) -> None:
    transport = StubTransport(body=fixture("openai_embeddings_ok.json"))
    adapter = OpenAIEmbeddings(api_key="sk-test-789", transport=transport)
    adapter.embed_texts(["q"])
    request = transport.requests[0]
    assert request.headers["Authorization"] == "Bearer sk-test-789"
    assert "sk-test-789" not in request.url
    assert json.loads(request.body) == {"input": ["q"]}


def test_openai_batches_requests() -> None:
    transport = StubTransport(body=fixture("openai_embeddings_ok.json"))
    adapter = OpenAIEmbeddings(api_key="k", transport=transport, max_batch=2)
    adapter.embed_texts([f"t{i}" for i in range(5)])
    assert len(transport.requests) == 3  # 2 + 2 + 1
    assert adapter.last_usage == [{"kind": "tokens", "value": 72.0}]  # 24 per request x 3


def test_openai_embed_images_is_clear_config_error() -> None:
    adapter = OpenAIEmbeddings(api_key="k", transport=StubTransport())
    with pytest.raises(ProviderConfigError, match="text-only"):
        adapter.embed_images([Path("frame.jpg")])


def test_openai_error_mapping() -> None:
    cases = [
        (401, AuthError),
        (402, QuotaError),
        (429, TransientError),
        (500, TransientError),
    ]
    for status, expected in cases:
        adapter = OpenAIEmbeddings(
            api_key="k", transport=StubTransport(status=status, body=b'{"error": {"message": "x"}}')
        )
        with pytest.raises(expected):
            adapter.embed_texts(["q"])


def test_openai_malformed_shape() -> None:
    adapter = OpenAIEmbeddings(api_key="k", transport=StubTransport(body=b'{"data": []}'))
    with pytest.raises(ProviderError, match="empty embeddings response"):
        adapter.embed_texts(["q"])


# --- cohere (multimodal fallback) -------------------------------------------------


def test_cohere_embed_texts() -> None:
    transport = StubTransport(body=fixture("cohere_embed_ok.json"))
    adapter = CohereEmbeddings(api_key="k", transport=transport)
    vectors = adapter.embed_texts(["текст один", "текст два"])
    assert vectors == [[0.11, 0.22, 0.33], [0.44, 0.55, 0.66]]
    assert adapter.last_usage == [{"kind": "tokens", "value": 12.0}]
    body = json.loads(transport.requests[0].body)
    assert body["texts"] == ["текст один", "текст два"]
    assert body["input_type"] == "search_document"
    assert body["embedding_types"] == ["float"]


def test_cohere_embed_images_sends_data_uri_within_limits(tmp_path: Path) -> None:
    transport = StubTransport(body=fixture("cohere_embed_ok.json"))
    adapter = CohereEmbeddings(api_key="k", transport=transport, max_image_dim=512, max_image_bytes=200_000)
    vectors = adapter.embed_images([make_png(tmp_path, width=2000, height=1200)])
    assert vectors == [[0.11, 0.22, 0.33], [0.44, 0.55, 0.66]]
    body = json.loads(transport.requests[0].body)
    assert len(body["images"]) == 1
    uri = body["images"][0]
    assert uri.startswith("data:image/jpeg;base64,")
    raw = base64.b64decode(uri.split(",", 1)[1])
    assert len(raw) <= 200_000  # acceptance: sent size within the limit
    decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    assert max(decoded.shape[0], decoded.shape[1]) <= 512


def test_cohere_batches() -> None:
    transport = StubTransport(body=fixture("cohere_embed_ok.json"))
    adapter = CohereEmbeddings(api_key="k", transport=transport, max_batch=2)
    adapter.embed_texts([f"t{i}" for i in range(5)])
    assert len(transport.requests) == 3
    assert adapter.last_usage == [{"kind": "tokens", "value": 36.0}]


def test_cohere_error_mapping() -> None:
    adapter = CohereEmbeddings(
        api_key="k",
        transport=StubTransport(status=429, body=b'{"message": "quota exceeded for billing period"}'),
    )
    with pytest.raises(QuotaError, match="quota"):
        adapter.embed_texts(["q"])


def test_cohere_malformed_shape() -> None:
    adapter = CohereEmbeddings(api_key="k", transport=StubTransport(body=b'{"embeddings": {"int8": []}}'))
    with pytest.raises(ProviderError, match="unexpected embeddings response shape"):
        adapter.embed_texts(["q"])


def test_vector_type_contract() -> None:
    transport = StubTransport(body=fixture("openai_embeddings_ok.json"))
    vectors = OpenAIEmbeddings(api_key="k", transport=transport).embed_texts(["q"])
    assert all(isinstance(v, list) and all(isinstance(x, float) for x in v) for v in vectors)


# --- layer integration ------------------------------------------------------------


def test_layer_builds_openai_and_records_cost(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    transport = StubTransport(body=fixture("openai_embeddings_ok.json"))
    layer, run_dir = layer_with_embed(tmp_path, "openai", monkeypatch, transport)
    vectors = layer.embeddings().embed_texts(["a", "b"])
    assert len(vectors) == 2
    calls = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["provider_calls"]
    assert calls[0]["task"] == "embed" and calls[0]["provider"] == "openai"
    assert calls[0]["units"] == [{"kind": "tokens", "value": 24.0}]
    # 24 tokens * 0.001 = 0.024 (test tariff; live per-minute check is Unresolved)
    assert calls[0]["cost"]["amount"] == pytest.approx(0.024, abs=1e-9)


def test_layer_embed_cached_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    transport = StubTransport(body=fixture("openai_embeddings_ok.json"))
    layer, run_dir = layer_with_embed(tmp_path, "openai", monkeypatch, transport)
    layer.embeddings().embed_texts(["a"])
    layer.embeddings().embed_texts(["a"])
    calls = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["provider_calls"]
    assert [c["cache_hit"] for c in calls] == [False, True]
    assert len(transport.requests) == 1


def test_layer_missing_embed_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EMBED_MISSING_KEY", raising=False)
    config = {
        "cache": {"dir": str(tmp_path / "cache")},
        "embeddings": {"provider": "openai", "model": "text-embedding-3-small", "api_key_env": "EMBED_MISSING_KEY"},
    }
    runctx = RunContext(runs_dir=str(tmp_path))
    with pytest.raises(ProviderConfigError, match="EMBED_MISSING_KEY"):
        ProviderLayer(config, runctx).embeddings()


def test_factories_registered() -> None:
    assert "openai" in FACTORIES["embed"]
    assert "cohere" in FACTORIES["embed"]


# --- live tests (manual, owner-side with real keys) --------------------------------


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY is not set")
def test_live_openai_embeddings() -> None:
    adapter = OpenAIEmbeddings(api_key=os.environ["OPENAI_API_KEY"])
    vectors = adapter.embed_texts(["проверка размерности"])
    assert len(vectors) == 1 and len(vectors[0]) > 0
    assert adapter.last_usage[0]["kind"] == "tokens"


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("COHERE_API_KEY"), reason="COHERE_API_KEY is not set")
def test_live_cohere_embeddings(tmp_path: Path) -> None:
    adapter = CohereEmbeddings(api_key=os.environ["COHERE_API_KEY"])
    text_vectors = adapter.embed_texts(["проверка"])
    image_vectors = adapter.embed_images([make_png(tmp_path, width=640, height=480)])
    assert len(text_vectors[0]) == len(image_vectors[0])  # one space, one dim
    assert adapter.last_usage[0]["kind"] == "tokens"

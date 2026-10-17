"""OpenAI adapter: text embeddings via /v1/embeddings.

Naryad 038, ADR-0006: the default embeddings path is "VLM descriptions +
text" with ``text-embedding-3-small``; this adapter implements the text
half only (the composition with a VLM is naryad 039's ADR-0002 matter).
``embed_images`` therefore fails with a clear configuration error instead
of silently degrading. Usage is billed in tokens (the response's
``usage.prompt_tokens``); the key travels in the Authorization header.
"""

from __future__ import annotations

import json

from razkadr.providers.base import ProviderConfigError, ProviderError, UsageReporter
from razkadr.providers.rest import (
    DEFAULT_TIMEOUT_S,
    Request,
    Transport,
    call_json,
    urllib_transport,
)
from razkadr.providers.types import Vector

OPENAI_EMBED_URL = "https://api.openai.com/v1/embeddings"

DEFAULT_MAX_BATCH = 2048  # provider limit: inputs per embeddings request


class OpenAIEmbeddings(UsageReporter):
    """Text-only embeddings (ADR-0006 default); bills tokens per call."""

    name = "openai"
    task = "embed"

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str = "",
        transport: Transport = urllib_transport,
        base_url: str = OPENAI_EMBED_URL,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_batch: int = DEFAULT_MAX_BATCH,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self._transport = transport
        self._base_url = base_url
        self._timeout_s = timeout_s
        self._max_batch = max(1, int(max_batch))
        self.last_usage: list[dict] = []

    def embed_texts(self, texts: list[str]) -> list[Vector]:
        vectors: list[Vector] = []
        billed = 0
        for start in range(0, len(texts), self._max_batch):
            chunk = texts[start:start + self._max_batch]
            data = call_json(self._transport, self.name, self._request({"input": chunk}))
            vectors.extend(self._parse(data))
            billed += self._billed_tokens(data)
        self._report([{"kind": "tokens", "value": float(billed)}])
        return vectors

    def embed_images(self, paths: list) -> list[Vector]:
        raise ProviderConfigError(
            "openai embeddings are text-only: use the multimodal 'cohere' provider "
            "or the VLM-description composition (naryad 039)"
        )

    def _request(self, body: dict) -> Request:
        return Request(
            method="POST",
            url=self._base_url,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            body=json.dumps(body).encode("utf-8"),
            timeout_s=self._timeout_s,
        )

    def _parse(self, data: dict) -> list[Vector]:
        try:
            items = data["data"]
            pairs = sorted((int(item["index"]), [float(v) for v in item["embedding"]]) for item in items)
        except (KeyError, TypeError, ValueError):
            raise ProviderError("openai: unexpected embeddings response shape") from None
        if not pairs:
            raise ProviderError("openai: empty embeddings response")
        return [vector for _, vector in pairs]

    def _billed_tokens(self, data: dict) -> int:
        try:
            return int(data["usage"]["prompt_tokens"])
        except (KeyError, TypeError, ValueError):
            return 0

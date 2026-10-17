"""Cohere adapter: multimodal embeddings via /v2/embed.

Naryad 038, ADR-0006: the multimodal fallback — ``embed-v4.0`` embeds
frames and texts into one vector space. Images are downscaled and
JPEG-re-encoded before sending (providers/images.py) and travel as
base64 data URIs; the key travels in the Authorization header. Batches
are capped at ``max_batch`` inputs per request (provider limit: 96).
Usage is billed in tokens (``meta.billed_units.input_tokens``).
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

from razkadr.providers.base import ProviderError, UsageReporter
from razkadr.providers.images import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_DIM,
    QUALITY_LADDER,
    prepare_image_jpeg,
)
from razkadr.providers.rest import (
    DEFAULT_TIMEOUT_S,
    Request,
    Transport,
    call_json,
    urllib_transport,
)
from razkadr.providers.types import Vector

COHERE_EMBED_URL = "https://api.cohere.com/v2/embed"

DEFAULT_MAX_BATCH = 96  # provider limit: inputs per /v2/embed request


class CohereEmbeddings(UsageReporter):
    """Multimodal embeddings (ADR-0006 fallback); bills tokens per call."""

    name = "cohere"
    task = "embed"

    def __init__(
        self,
        model: str = "embed-v4.0",
        api_key: str = "",
        transport: Transport = urllib_transport,
        base_url: str = COHERE_EMBED_URL,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_batch: int = DEFAULT_MAX_BATCH,
        input_type: str = "search_document",
        max_image_dim: int = DEFAULT_MAX_DIM,
        max_image_bytes: int = DEFAULT_MAX_BYTES,
        jpeg_quality: int = QUALITY_LADDER[0],
    ) -> None:
        self.model = model
        self.api_key = api_key
        self._transport = transport
        self._base_url = base_url
        self._timeout_s = timeout_s
        self._max_batch = max(1, int(max_batch))
        self._input_type = input_type
        self._max_image_dim = max_image_dim
        self._max_image_bytes = max_image_bytes
        self._jpeg_quality = jpeg_quality
        self.last_usage: list[dict] = []

    def embed_texts(self, texts: list[str]) -> list[Vector]:
        vectors: list[Vector] = []
        billed = 0
        for start in range(0, len(texts), self._max_batch):
            chunk = texts[start:start + self._max_batch]
            data = call_json(self._transport, self.name, self._request({"texts": chunk}))
            vectors.extend(self._parse(data))
            billed += self._billed_tokens(data)
        self._report([{"kind": "tokens", "value": float(billed)}])
        return vectors

    def embed_images(self, paths: list[Path]) -> list[Vector]:
        vectors: list[Vector] = []
        billed = 0
        uris = [self._image_uri(p) for p in paths]
        for start in range(0, len(uris), self._max_batch):
            chunk = uris[start:start + self._max_batch]
            data = call_json(self._transport, self.name, self._request({"images": chunk}))
            vectors.extend(self._parse(data))
            billed += self._billed_tokens(data)
        self._report([{"kind": "tokens", "value": float(billed)}])
        return vectors

    def _image_uri(self, path: Path) -> str:
        data = prepare_image_jpeg(
            path,
            max_dim=self._max_image_dim,
            max_bytes=self._max_image_bytes,
            quality=self._jpeg_quality,
        )
        return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")

    def _request(self, inputs: dict) -> Request:
        body = {"model": self.model, "embedding_types": ["float"], "input_type": self._input_type, **inputs}
        return Request(
            method="POST",
            url=self._base_url,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            body=json.dumps(body).encode("utf-8"),
            timeout_s=self._timeout_s,
        )

    def _parse(self, data: dict) -> list[Vector]:
        try:
            rows = data["embeddings"]["float"]
            vectors = [[float(v) for v in row] for row in rows]
        except (KeyError, TypeError, ValueError):
            raise ProviderError("cohere: unexpected embeddings response shape") from None
        if not vectors:
            raise ProviderError("cohere: empty embeddings response")
        return vectors

    def _billed_tokens(self, data: dict) -> int:
        try:
            return int(data["meta"]["billed_units"]["input_tokens"])
        except (KeyError, TypeError, ValueError):
            return 0

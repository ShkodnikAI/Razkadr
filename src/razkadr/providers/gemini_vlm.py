"""Google Gemini adapter: VLM image descriptions via generateContent.

Naryad 039, ADR-0006: default VLM ``gemini-3.5-flash-lite`` for OCR and
frame descriptions. The image is downscaled/re-encoded before sending
(providers/images.py) and travels as inline base64; the key travels in
the ``x-goog-api-key`` header. Temperature 0 by default (naryad
assumption: extraction tasks). Optional JSON mode: the model is forced
to answer JSON, unparseable replies are retried and then rejected with
a clear error. Usage is billed in tokens from ``usageMetadata``.
"""

from __future__ import annotations

import base64
import json

from razkadr.providers.base import ProviderError, UsageReporter
from razkadr.providers.gemini import GEMINI_BASE_URL
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

TEMPERATURE = 0  # naryad 039 assumption: extraction tasks are deterministic


def reply_text(data: dict) -> str:
    """The first candidate's text of a Gemini generateContent response."""
    try:
        return str(data["candidates"][0]["content"]["parts"][0]["text"])
    except (KeyError, IndexError, TypeError):
        raise ProviderError("gemini: unexpected response shape; no reply text") from None


class GeminiVLM(UsageReporter):
    """Default VLM adapter (ADR-0006); bills 1 image + response tokens."""

    name = "gemini"
    task = "vlm"

    def __init__(
        self,
        model: str = "gemini-3.5-flash-lite",
        api_key: str = "",
        transport: Transport = urllib_transport,
        base_url: str = GEMINI_BASE_URL,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        json_mode: bool = False,
        json_retries: int = 2,
        max_image_dim: int = DEFAULT_MAX_DIM,
        max_image_bytes: int = DEFAULT_MAX_BYTES,
        jpeg_quality: int = QUALITY_LADDER[0],
    ) -> None:
        self.model = model
        self.api_key = api_key
        self._transport = transport
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._json_mode = json_mode
        self._json_retries = max(0, int(json_retries))
        self._max_image_dim = max_image_dim
        self._max_image_bytes = max_image_bytes
        self._jpeg_quality = jpeg_quality
        self.last_usage: list[dict] = []

    def describe_image(self, path, prompt: str) -> str:
        image = prepare_image_jpeg(
            path, max_dim=self._max_image_dim, max_bytes=self._max_image_bytes, quality=self._jpeg_quality
        )
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(image).decode("ascii")}},
                    ]
                }
            ],
            "generationConfig": {"temperature": TEMPERATURE},
        }
        if self._json_mode:
            payload["generationConfig"]["response_mime_type"] = "application/json"
        request = Request(
            method="POST",
            url=f"{self._base_url}/{self.model}:generateContent",
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
            body=json.dumps(payload).encode("utf-8"),
            timeout_s=self._timeout_s,
        )
        attempts = 1 + self._json_retries
        for attempt in range(1, attempts + 1):
            data = call_json(self._transport, self.name, request)
            text = reply_text(data)
            if not self._json_mode or _is_json(text):
                self._report_usage(data)
                return text
            if attempt < attempts:
                continue  # invalid JSON: retry the same request
        raise ProviderError(
            f"gemini: reply is not valid JSON after {attempts} attempts"
        )

    def _report_usage(self, data: dict) -> None:
        try:
            meta = data["usageMetadata"]
            tokens = int(meta.get("promptTokenCount", 0)) + int(meta.get("candidatesTokenCount", 0))
        except (AttributeError, KeyError, IndexError, TypeError, ValueError):
            tokens = 0
        self._report([{"kind": "images", "value": 1}, {"kind": "tokens", "value": float(tokens)}])


def _is_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, TypeError):
        return False

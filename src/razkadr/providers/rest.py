"""Shared plumbing for real provider adapters (naryad 037).

Stdlib-only (urllib): adapters add no runtime dependencies. Every adapter
takes an injectable ``transport`` callable so unit tests run hermetically
against recorded/synthetic responses; the default transport hits the
network. HTTP status to layer-error mapping is shared:

- 401/403 -> AuthError (check the key, do not retry)
- 402 -> QuotaError
- 408/425 -> TransientError (timeout, too early; safe to retry)
- 429 -> QuotaError when the body indicates exhausted quota/RESOURCE_EXHAUSTED,
  else TransientError (rate limit, safe to retry)
- 5xx -> TransientError
- anything else -> ProviderError

Error messages never contain the request body, URLs or the API key;
the layer masks provider errors through SECRETS on top of that.
"""

from __future__ import annotations

import http.client
import json
import urllib.error
import urllib.request
import wave
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from razkadr.providers.base import AuthError, ProviderError, QuotaError, TransientError

DEFAULT_TIMEOUT_S = 120.0

ERROR_BODY_LIMIT = 500  # chars of an error body kept in exception messages


@dataclass(frozen=True)
class Request:
    """One HTTP call to a provider API; the key travels in headers only."""

    method: str
    url: str
    headers: dict
    body: bytes
    timeout_s: float = DEFAULT_TIMEOUT_S


@dataclass(frozen=True)
class Response:
    status: int
    body: bytes


Transport = Callable[[Request], Response]


def urllib_transport(request: Request) -> Response:
    """Default transport: one urllib call; failures are mapped in call_json."""

    req = urllib.request.Request(
        request.url,
        data=request.body if request.method != "GET" else None,
        method=request.method,
    )
    for name, value in request.headers.items():
        req.add_header(name, value)
    try:
        with urllib.request.urlopen(req, timeout=request.timeout_s) as resp:
            return Response(status=resp.status, body=resp.read())
    except urllib.error.HTTPError as exc:
        return Response(status=exc.code, body=exc.read())


def raise_for_status(provider: str, response: Response) -> None:
    """Map a non-2xx provider response to the layer error taxonomy."""
    if 200 <= response.status < 300:
        return
    text = response.body.decode("utf-8", errors="replace")[:ERROR_BODY_LIMIT]
    if response.status in (401, 403):
        raise AuthError(f"{provider}: auth failed (HTTP {response.status}); check the API key")
    if response.status == 402:
        raise QuotaError(f"{provider}: payment required (HTTP 402); check billing")
    if response.status in (408, 425):
        raise TransientError(f"{provider}: temporary failure (HTTP {response.status})")
    if response.status == 429:
        lowered = text.lower()
        if "quota" in lowered or "exhausted" in lowered or "resource_exhausted" in lowered:
            raise QuotaError(f"{provider}: quota exhausted (HTTP 429); check billing/limits")
        raise TransientError(f"{provider}: rate limited (HTTP 429)")
    if response.status >= 500:
        raise TransientError(f"{provider}: server error (HTTP {response.status})")
    raise ProviderError(f"{provider}: unexpected HTTP {response.status}: {text[:200]}")


def call_json(transport: Transport, provider: str, request: Request) -> dict:
    """One JSON call: transport -> error mapping -> parsed JSON object.

    Any non-provider exception raised by the transport (timeout, DNS,
    connection reset, ...) is transient by definition; only the exception
    type name goes into the message.
    """
    try:
        response = transport(request)
    except ProviderError:
        raise
    except (OSError, TimeoutError, http.client.HTTPException) as exc:
        # timeout, DNS, connection reset, TLS — all transient; only the
        # exception type name goes into the message (no URL, no body, no key)
        raise TransientError(f"{provider}: transport failure: {type(exc).__name__}") from None
    raise_for_status(provider, response)
    try:
        data = json.loads(response.body)
    except json.JSONDecodeError:
        raise ProviderError(f"{provider}: response is not valid JSON") from None
    if not isinstance(data, dict):
        raise ProviderError(f"{provider}: response JSON must be an object")
    return data


def wav_duration_s(path: Path) -> float:
    """Duration of a PCM WAV file in seconds.

    Mirrors ``stages.transcript.wav_duration_s``; provider adapters must
    not import stage code (stages depend on providers, not vice versa).
    """
    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
    if rate <= 0:
        raise ProviderError(f"audio file {Path(path).name} has invalid sample rate {rate}")
    return frames / float(rate)

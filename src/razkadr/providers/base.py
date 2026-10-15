"""Provider interfaces (Protocols) and errors.

Stages depend only on these Protocols and result types; a provider is
switched by config, never by stage code (naryad 033). Concrete providers
implement the Protocol methods and report the units of the last call via
``last_usage`` (see UsageReporter) - calls of one service are serialized
by the layer's concurrency limit, so the attribute is race-free.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from razkadr.providers.types import Message, Transcript, Vector

TASKS = ("asr", "embed", "ocr", "vlm", "llm")
UNIT_KINDS = ("seconds", "tokens", "images")
KEY_MODES = ("platform", "user")


class ProviderError(Exception):
    """Base provider-layer error; messages are safe to log (masked)."""


class ProviderConfigError(ProviderError):
    """Misconfiguration: unknown provider, missing key, bad section."""


class AuthError(ProviderError):
    """Invalid or unauthorized API key; check the key, do not retry."""


class QuotaError(ProviderError):
    """Quota exhausted or payment required; check billing, do not retry."""


class TransientError(ProviderError):
    """Temporary failure (rate limit, 5xx, timeout); safe to retry."""


class UsageReporter:
    """Mixin for providers: report units of the last call (naryad 033)."""

    last_usage: list[dict]

    def _report(self, units: list[dict]) -> None:
        for unit in units:
            if unit.get("kind") not in UNIT_KINDS:
                raise ValueError(f"unit kind must be one of {UNIT_KINDS}, got {unit.get('kind')!r}")
        self.last_usage = [{"kind": u["kind"], "value": u["value"]} for u in units]


@runtime_checkable
class ASRProvider(Protocol):
    """Speech to text."""

    name: str
    model: str

    def transcribe(self, audio_path: Path, lang: str | None = None) -> Transcript: ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Text and image embeddings into one vector space."""

    name: str
    model: str

    def embed_images(self, paths: list[Path]) -> list[Vector]: ...

    def embed_texts(self, texts: list[str]) -> list[Vector]: ...


@runtime_checkable
class VLMProvider(Protocol):
    """Image description (screen text, frames)."""

    name: str
    model: str

    def describe_image(self, path: Path, prompt: str) -> str: ...


@runtime_checkable
class LLMProvider(Protocol):
    """Chat completion."""

    name: str
    model: str

    def complete(self, messages: list[Message]) -> str: ...

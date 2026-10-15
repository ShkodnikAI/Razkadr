"""Deterministic fake providers: full Protocol coverage, no network.

Fakes are the reference implementations for stage tests (naryad 033):
same input -> same output, units reported like real adapters would bill.
Failure injection (``fail_first`` transient errors, ``auth_error``,
``quota_error``) lets tests exercise retries and error paths without
network or real keys.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from razkadr.providers.base import AuthError, QuotaError, TransientError, UsageReporter
from razkadr.providers.types import Message, Transcript, TranscriptSegment, Vector


def _seed(*parts: object) -> bytes:
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return hashlib.sha256(payload).digest()


class FakeASR(UsageReporter):
    """Two deterministic segments; bills usage_seconds per call."""

    name = "fake"
    task = "asr"

    def __init__(
        self,
        model: str = "fake-nova-2",
        lang: str = "ru",
        usage_seconds: float = 10.0,
        fail_first: int = 0,
        auth_error: str | None = None,
        quota_error: str | None = None,
        name: str = "fake",
    ) -> None:
        self.name = name
        self.model = model
        self.lang = lang
        self.usage_seconds = usage_seconds
        self.fail_first = fail_first
        self.auth_error = auth_error
        self.quota_error = quota_error
        self.calls = 0
        self.last_usage = []

    def _gate(self) -> None:
        self.calls += 1
        if self.auth_error:
            raise AuthError(self.auth_error)
        if self.quota_error:
            raise QuotaError(self.quota_error)
        if self.calls <= self.fail_first:
            raise TransientError(f"fake transient failure #{self.calls}")

    def transcribe(self, audio_path: Path, lang: str | None = None) -> Transcript:
        self._gate()
        digest = _seed(Path(audio_path).name, lang or self.lang)
        self._report([{"kind": "seconds", "value": self.usage_seconds}])
        text = f"fake transcript {digest.hex()[:12]}"
        return Transcript(
            lang=lang or self.lang,
            segments=[
                TranscriptSegment(t_start=0.0, t_end=2.5, text=f"{text} part 1", lang=lang or self.lang),
                TranscriptSegment(t_start=2.5, t_end=5.0, text=f"{text} part 2", lang=lang or self.lang),
            ],
        )


class FakeEmbeddings(UsageReporter):
    """L2-normalized pseudo-random vectors; bills images/tokens per call."""

    name = "fake"
    task = "embed"

    def __init__(
        self,
        model: str = "fake-embed-v2",
        dim: int = 8,
        fail_first: int = 0,
        auth_error: str | None = None,
        quota_error: str | None = None,
        name: str = "fake",
    ) -> None:
        self.name = name
        self.model = model
        self.dim = dim
        self.fail_first = fail_first
        self.auth_error = auth_error
        self.quota_error = quota_error
        self.calls = 0
        self.last_usage = []

    def _gate(self) -> None:
        self.calls += 1
        if self.auth_error:
            raise AuthError(self.auth_error)
        if self.quota_error:
            raise QuotaError(self.quota_error)
        if self.calls <= self.fail_first:
            raise TransientError(f"fake transient failure #{self.calls}")

    def _vector(self, content: bytes) -> Vector:
        digest = hashlib.sha256(b"%s|%d" % (content, self.dim)).digest()
        raw = [int(digest[i * 4:(i + 1) * 4].hex(), 16) / 2**32 for i in range(self.dim)]
        shifted = [v * 2 - 1 for v in raw]
        norm = sum(v * v for v in shifted) ** 0.5 or 1.0
        return [round(v / norm, 6) for v in shifted]

    def embed_images(self, paths: list[Path]) -> list[Vector]:
        self._gate()
        self._report([{"kind": "images", "value": len(paths)}])
        return [self._vector(Path(p).read_bytes()) for p in paths]

    def embed_texts(self, texts: list[str]) -> list[Vector]:
        self._gate()
        self._report([{"kind": "tokens", "value": float(sum(len(t.split()) for t in texts))}])
        return [self._vector(t.encode("utf-8")) for t in texts]


class FakeVLM(UsageReporter):
    """Deterministic image description; bills 1 image + prompt tokens."""

    name = "fake"
    task = "vlm"

    def __init__(
        self,
        model: str = "fake-vision-1",
        fail_first: int = 0,
        auth_error: str | None = None,
        quota_error: str | None = None,
        name: str = "fake",
    ) -> None:
        self.name = name
        self.model = model
        self.fail_first = fail_first
        self.auth_error = auth_error
        self.quota_error = quota_error
        self.calls = 0
        self.last_usage = []

    def _gate(self) -> None:
        self.calls += 1
        if self.auth_error:
            raise AuthError(self.auth_error)
        if self.quota_error:
            raise QuotaError(self.quota_error)
        if self.calls <= self.fail_first:
            raise TransientError(f"fake transient failure #{self.calls}")

    def describe_image(self, path: Path, prompt: str) -> str:
        self._gate()
        digest = _seed(Path(path).name, prompt)
        self._report([
            {"kind": "images", "value": 1},
            {"kind": "tokens", "value": float(len(prompt.split()))},
        ])
        return f"fake description {digest.hex()[:12]}: {prompt[:32]}"


class FakeLLM(UsageReporter):
    """Deterministic completion; bills prompt+response tokens."""

    name = "fake"
    task = "llm"

    def __init__(
        self,
        model: str = "fake-gpt-4o-mini",
        fail_first: int = 0,
        auth_error: str | None = None,
        quota_error: str | None = None,
        name: str = "fake",
    ) -> None:
        self.name = name
        self.model = model
        self.fail_first = fail_first
        self.auth_error = auth_error
        self.quota_error = quota_error
        self.calls = 0
        self.last_usage = []

    def _gate(self) -> None:
        self.calls += 1
        if self.auth_error:
            raise AuthError(self.auth_error)
        if self.quota_error:
            raise QuotaError(self.quota_error)
        if self.calls <= self.fail_first:
            raise TransientError(f"fake transient failure #{self.calls}")

    def complete(self, messages: list[Message]) -> str:
        self._gate()
        digest = _seed(*(f"{m.role}:{m.content}" for m in messages))
        self._report([{"kind": "tokens", "value": float(sum(len(m.content.split()) for m in messages) + 8)}])
        return f"fake completion {digest.hex()[:12]}"

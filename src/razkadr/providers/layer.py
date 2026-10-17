"""Provider layer: config-driven services over provider implementations.

Stages get services with clean methods (transcribe, embed_images,
embed_texts, describe_image, complete); the implementation behind them
is chosen by razkadr.toml and can be switched without touching stage
code (naryad 033, ADR-0004). Every call is cached by input+params and
accounted into run.json (units, cost, cache_hit).

    [asr]
    provider = "fake"          # implementation, see FACTORIES
    model = "fake-nova-2"
    api_key_env = "ASR_API_KEY"  # or api_key = "..." (discouraged, masked)
    max_attempts = 3           # retries on transient errors
    base_delay_s = 1.0         # backoff: base_delay * 2^(attempt-1)
    requests_per_minute = 60   # throttle
    max_concurrency = 1        # parallel calls to one provider
    model_version = "2"        # part of the cache key
    [asr.tariff]
    units = "seconds"
    price = 0.0059
    currency = "USD"
    tariff_date = "2026-10-01"
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path

from razkadr.log import SECRETS, get_logger
from razkadr.providers.base import (
    KEY_MODES,
    ProviderConfigError,
    ProviderError,
    TransientError,
)
from razkadr.providers.cache import CallCache, call_key
from razkadr.providers.cohere import CohereEmbeddings
from razkadr.providers.deepgram import DeepgramASR
from razkadr.providers.fakes import FakeASR, FakeEmbeddings, FakeLLM, FakeVLM
from razkadr.providers.gemini import GeminiASR
from razkadr.providers.gemini_vlm import GeminiVLM
from razkadr.providers.openai import OpenAIEmbeddings
from razkadr.providers.openai_llm import OpenAILLM
from razkadr.providers.tariffs import tariff_from_config
from razkadr.providers.types import Message, Transcript, Vector
from razkadr.runctx import RunContext, file_sha256

logger = get_logger("providers")

TASK_SECTIONS = {"asr": "asr", "embed": "embeddings", "vlm": "vlm", "llm": "llm"}
DEFAULT_STAGES = {"asr": "transcript", "embed": "index", "vlm": "ocr", "llm": "chunks"}


def resolve_key(section: dict, section_name: str) -> str | None:
    """Key from env (api_key_env) or from the user config (api_key, masked).

    Missing env value is a clear configuration error; direct config keys
    are supported for BYOK (mode = user) but discouraged.
    """
    import os

    env_name = section.get("api_key_env")
    if env_name:
        value = os.environ.get(env_name) or SECRETS.get(env_name)
        if not value:
            raise ProviderConfigError(
                f"[{section_name}] api_key_env={env_name!r} is set but the variable is empty; "
                f"export {env_name}=<your key> (keys are never logged)"
            )
        SECRETS.register(value)
        return value
    return SECRETS.get(f"{section_name}.api_key")


class Throttle:
    """Minimal-interval rate limiter; clock/sleeper are injectable for tests."""

    def __init__(self, requests_per_minute: float, clock=time.monotonic, sleeper=time.sleep) -> None:
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        self.min_interval = 60.0 / requests_per_minute
        self._clock = clock
        self._sleep = sleeper
        self._last: float | None = None

    def wait(self) -> None:
        now = self._clock()
        if self._last is not None:
            remaining = self.min_interval - (now - self._last)
            if remaining > 0:
                self._sleep(remaining)
                now = self._clock()
        self._last = now


class _BaseService:
    """Cache + throttle + retries + accounting around a provider instance."""

    task = ""

    def __init__(
        self,
        provider: object,
        section: dict,
        runctx: RunContext,
        stage: str,
        cache: CallCache,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.provider = provider
        self.runctx = runctx
        self.stage = stage
        self.cache = cache
        self.sleeper = sleeper
        self.model_version = section.get("model_version")
        self.max_attempts = max(1, int(section.get("max_attempts", 3)))
        self.base_delay = float(section.get("base_delay_s", 1.0))
        self.tariff = tariff_from_config(section)
        self._semaphore = threading.Semaphore(max(1, int(section.get("max_concurrency", 1))))
        self._throttle = Throttle(float(section.get("requests_per_minute", 60)), clock=clock, sleeper=sleeper)

    # -- plumbing ------------------------------------------------------------

    def _record(self, units: list[dict], cache_hit: bool, cost: float) -> None:
        self.runctx.record_provider_call(
            stage=self.stage,
            provider=self.provider.name,
            model=self.provider.model,
            units=units,
            cost_amount=cost,
            currency=self.tariff.currency,
            tariff_date=self.tariff.tariff_date,
            cache_hit=cache_hit,
            task=self.task,
        )

    def _invoke_with_retry(self, invoke: Callable[[], object]) -> object:
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                return invoke()
            except TransientError as exc:
                last_error = exc
                if attempt < self.max_attempts:
                    delay = self.base_delay * 2 ** (attempt - 1)
                    logger.warning(
                        "transient failure, retrying",
                        extra={"fields": {"stage": self.stage, "attempt": attempt, "delay_s": delay}},
                    )
                    self.sleeper(delay)
            except ProviderError as exc:
                raise type(exc)(SECRETS.mask(str(exc))) from None
        raise TransientError(
            SECRETS.mask(f"{self.task} failed after {self.max_attempts} attempts: {last_error}")
        ) from last_error

    def _call(
        self,
        inputs: dict,
        params: dict,
        invoke: Callable[[], object],
        to_json: Callable[[object], object],
        from_json: Callable[[object], object],
    ) -> object:
        key = call_key(
            self.task,
            self.provider.name,
            self.provider.model,
            inputs,
            {**params, "model_version": self.model_version},
        )
        entry = self.cache.get(key)
        if entry is not None:
            units = entry.get("units", [])
            self._record(units=units, cache_hit=True, cost=0.0)
            return from_json(entry["result"])
        with self._semaphore:
            self._throttle.wait()
            result = self._invoke_with_retry(invoke)
        units = list(getattr(self.provider, "last_usage", []))
        cost = self.tariff.cost(units)
        self._record(units=units, cache_hit=False, cost=cost)
        self.cache.put(key, {
            "task": self.task,
            "provider": self.provider.name,
            "model": self.provider.model,
            "units": units,
            "result": to_json(result),
        })
        return result


class ASRService(_BaseService):
    task = "asr"

    def transcribe(self, audio_path: Path, lang: str | None = None) -> Transcript:
        inputs = {"audio_sha256": file_sha256(audio_path)}
        params = {"lang": lang}
        result = self._call(
            inputs,
            params,
            invoke=lambda: self.provider.transcribe(audio_path, lang),
            to_json=lambda r: r.to_json(),
            from_json=Transcript.from_json,
        )
        return result  # type: ignore[return-value]


class EmbeddingsService(_BaseService):
    task = "embed"

    def embed_images(self, paths: list[Path]) -> list[Vector]:
        inputs = {"images_sha256": [file_sha256(p) for p in paths]}
        result = self._call(
            inputs,
            {},
            invoke=lambda: self.provider.embed_images(paths),
            to_json=lambda r: r,
            from_json=lambda r: r,
        )
        return result  # type: ignore[return-value]

    def embed_texts(self, texts: list[str]) -> list[Vector]:
        result = self._call(
            {"texts": texts},
            {},
            invoke=lambda: self.provider.embed_texts(texts),
            to_json=lambda r: r,
            from_json=lambda r: r,
        )
        return result  # type: ignore[return-value]


class VLMService(_BaseService):
    task = "vlm"

    def describe_image(self, path: Path, prompt: str) -> str:
        inputs = {"image_sha256": file_sha256(path), "prompt": prompt}
        result = self._call(
            inputs,
            {},
            invoke=lambda: self.provider.describe_image(path, prompt),
            to_json=lambda r: r,
            from_json=lambda r: r,
        )
        return result  # type: ignore[return-value]


class LLMService(_BaseService):
    task = "llm"

    def complete(self, messages: list[Message]) -> str:
        inputs = {"messages": [{"role": m.role, "content": m.content} for m in messages]}
        result = self._call(
            inputs,
            {},
            invoke=lambda: self.provider.complete(messages),
            to_json=lambda r: r,
            from_json=lambda r: r,
        )
        return result  # type: ignore[return-value]


SERVICES = {"asr": ASRService, "embed": EmbeddingsService, "vlm": VLMService, "llm": LLMService}

FACTORIES: dict[str, dict[str, Callable[[dict], object]]] = {
    "asr": {
        "fake": lambda s: FakeASR(model=s.get("model", "fake-nova-2"), lang=s.get("lang", "ru")),
        "fake-alt": lambda s: FakeASR(name="fake-alt", model=s.get("model", "fake-nova-alt")),
        # real adapters (naryad 037, ADR-0006): default + fallback
        "gemini": lambda s: GeminiASR(
            model=s.get("model", "gemini-3.5-transcribe"), api_key=resolve_key(s, "asr") or ""
        ),
        "deepgram": lambda s: DeepgramASR(
            model=s.get("model", "nova-3"), api_key=resolve_key(s, "asr") or ""
        ),
    },
    "embed": {
        "fake": lambda s: FakeEmbeddings(model=s.get("model", "fake-embed-v2"), dim=int(s.get("dim", 8))),
        "fake-alt": lambda s: FakeEmbeddings(name="fake-alt", model=s.get("model", "fake-embed-alt")),
        # real adapters (naryad 038, ADR-0006): default text-only + multimodal fallback
        "openai": lambda s: OpenAIEmbeddings(
            model=s.get("model", "text-embedding-3-small"),
            api_key=resolve_key(s, "embeddings") or "",
            max_batch=int(s.get("max_batch", 2048)),
        ),
        "cohere": lambda s: CohereEmbeddings(
            model=s.get("model", "embed-v4.0"),
            api_key=resolve_key(s, "embeddings") or "",
            max_batch=int(s.get("max_batch", 96)),
            input_type=s.get("input_type", "search_document"),
            max_image_dim=int(s.get("max_image_dim", 1024)),
            max_image_bytes=int(s.get("max_image_bytes", 4_000_000)),
            jpeg_quality=int(s.get("jpeg_quality", 85)),
        ),
    },
    "vlm": {
        "fake": lambda s: FakeVLM(model=s.get("model", "fake-vision-1")),
        "fake-alt": lambda s: FakeVLM(name="fake-alt", model=s.get("model", "fake-vision-alt")),
        # real adapter (naryad 039, ADR-0006): default VLM
        "gemini": lambda s: GeminiVLM(
            model=s.get("model", "gemini-3.5-flash-lite"),
            api_key=resolve_key(s, "vlm") or "",
            json_mode=bool(s.get("json_mode", False)),
            max_image_dim=int(s.get("max_image_dim", 1024)),
            max_image_bytes=int(s.get("max_image_bytes", 4_000_000)),
            jpeg_quality=int(s.get("jpeg_quality", 85)),
        ),
    },
    "llm": {
        "fake": lambda s: FakeLLM(model=s.get("model", "fake-gpt-4o-mini")),
        "fake-alt": lambda s: FakeLLM(name="fake-alt", model=s.get("model", "fake-llm-alt")),
        # real adapter (naryad 039, ADR-0006): default LLM
        "openai": lambda s: OpenAILLM(
            model=s.get("model", "gpt-6-luna"),
            api_key=resolve_key(s, "llm") or "",
            temperature=float(s.get("temperature", 0)),
            json_mode=bool(s.get("json_mode", False)),
        ),
    },
}


class ProviderLayer:
    """Facade: layer.asr(stage=...).transcribe(...) etc.; built from config."""

    def __init__(self, config: dict, runctx: RunContext, cache: CallCache | None = None) -> None:
        self.config = config
        self.runctx = runctx
        mode = config.get("keys", {}).get("mode", "platform")
        if mode not in KEY_MODES:
            raise ProviderConfigError(f"[keys] mode must be one of {KEY_MODES}, got {mode!r}")
        self._cache = cache or CallCache(config.get("cache", {}).get("dir"))
        self._services: dict[tuple[str, str], object] = {}

    def asr(self, stage: str | None = None) -> ASRService:
        return self._service("asr", stage)  # type: ignore[return-value]

    def embeddings(self, stage: str | None = None) -> EmbeddingsService:
        return self._service("embed", stage)  # type: ignore[return-value]

    def vlm(self, stage: str | None = None) -> VLMService:
        return self._service("vlm", stage)  # type: ignore[return-value]

    def llm(self, stage: str | None = None) -> LLMService:
        return self._service("llm", stage)  # type: ignore[return-value]

    def _service(self, task: str, stage: str | None):
        stage = stage or DEFAULT_STAGES[task]
        key = (task, stage)
        if key not in self._services:
            section_name = TASK_SECTIONS[task]
            section = self.config.get(section_name, {})
            self._services[key] = SERVICES[task](
                provider=self._build_provider(task, section_name, section),
                section=section,
                runctx=self.runctx,
                stage=stage,
                cache=self._cache,
            )
        return self._services[key]

    def _build_provider(self, task: str, section_name: str, section: dict):
        provider_name = section.get("provider")
        if not provider_name:
            raise ProviderConfigError(f"[{section_name}] provider is not set in the configuration")
        factories = FACTORIES[task]
        if provider_name not in factories:
            raise ProviderConfigError(
                f"unknown {section_name} provider {provider_name!r} (available: {sorted(factories)})"
            )
        resolve_key(section, section_name)
        return factories[provider_name](section)

"""OpenAI adapter: chat completions via /v1/chat/completions.

Naryad 039, ADR-0006: default LLM ``gpt-6-luna`` for chunking, text
condensation and translation. Temperature 0 by default (naryad
assumption). Optional JSON mode: ``response_format: json_object`` plus
local validation - an unparseable reply is retried up to ``json_retries``
times and then rejected with a clear error (the chunking stage treats
that as its documented deterministic fallback). Usage is billed in
tokens (``usage.prompt_tokens + completion_tokens``); the key travels
in the Authorization header. Note: OpenAI's JSON mode requires the word
"JSON" to appear in the messages (provider contract, see docs/config.md).
"""

from __future__ import annotations

import json

from razkadr.providers.base import ProviderError, UsageReporter
from razkadr.providers.rest import (
    DEFAULT_TIMEOUT_S,
    Request,
    Transport,
    call_json,
    urllib_transport,
)
from razkadr.providers.types import Message

OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"

TEMPERATURE = 0  # naryad 039 assumption: extraction tasks are deterministic


class OpenAILLM(UsageReporter):
    """Default LLM adapter (ADR-0006); bills prompt+completion tokens."""

    name = "openai"
    task = "llm"

    def __init__(
        self,
        model: str = "gpt-6-luna",
        api_key: str = "",
        transport: Transport = urllib_transport,
        base_url: str = OPENAI_CHAT_URL,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        temperature: float = TEMPERATURE,
        json_mode: bool = False,
        json_retries: int = 2,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self._transport = transport
        self._base_url = base_url
        self._timeout_s = timeout_s
        self._temperature = float(temperature)
        self._json_mode = json_mode
        self._json_retries = max(0, int(json_retries))
        self.last_usage: list[dict] = []

    def complete(self, messages: list[Message]) -> str:
        body = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": self._temperature,
        }
        if self._json_mode:
            body["response_format"] = {"type": "json_object"}
        request = Request(
            method="POST",
            url=self._base_url,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            body=json.dumps(body).encode("utf-8"),
            timeout_s=self._timeout_s,
        )
        attempts = 1 + self._json_retries
        for attempt in range(1, attempts + 1):
            data = call_json(self._transport, self.name, request)
            text = self._reply(data)
            if not self._json_mode or _is_json(text):
                self._report_usage(data)
                return text
            if attempt < attempts:
                continue  # invalid JSON: retry the same request
        raise ProviderError(f"openai: reply is not valid JSON after {attempts} attempts")

    def _reply(self, data: dict) -> str:
        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError):
            raise ProviderError("openai: unexpected completion response shape") from None

    def _report_usage(self, data: dict) -> None:
        try:
            usage = data["usage"]
            tokens = int(usage.get("prompt_tokens", 0)) + int(usage.get("completion_tokens", 0))
        except (AttributeError, KeyError, IndexError, TypeError, ValueError):
            tokens = 0
        self._report([{"kind": "tokens", "value": float(tokens)}])


def _is_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, TypeError):
        return False

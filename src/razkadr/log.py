"""Secret registry and structured JSON logging with masking.

Secrets (API keys, tokens) are registered by the config loader and by
callers; the masker replaces them in every log line and in every JSON
file the RunContext writes, so keys never reach logs or artifacts
(naryad 036, AGENTS.md).
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime

MASK = "[REDACTED]"

# Substrings (lowercase) that mark a config key or env var name as secret.
SECRET_MARKERS = ("key", "token", "secret", "password")


def is_secret_name(name: str) -> bool:
    """Secret-valued config key names.

    ``*_env`` keys are references to environment variables (e.g.
    ``api_key_env = "ASR_API_KEY"``) - the value is a variable name, not a
    secret, so such keys stay in the config.
    """
    lowered = name.lower()
    if lowered.endswith("_env"):
        return False
    return any(marker in lowered for marker in SECRET_MARKERS)


class SecretRegistry:
    def __init__(self) -> None:
        self._secrets: set[str] = set()
        self._named: dict[str, str] = {}

    def register(self, value: object) -> None:
        """Register an anonymous secret value for masking; empty values ignored."""
        if isinstance(value, str) and value.strip():
            self._secrets.add(value)

    def store(self, name: str, value: object) -> None:
        """Store a named secret (retrievable) and register it for masking."""
        if isinstance(value, str) and value.strip():
            self._named[name] = value
            self._secrets.add(value)

    def get(self, name: str) -> str | None:
        return self._named.get(name)

    def mask(self, text: str) -> str:
        for secret in self._secrets:
            if secret in text:
                text = text.replace(secret, MASK)
        return text

    @property
    def registered(self) -> int:
        return len(self._secrets)


SECRETS = SecretRegistry()


def deep_mask(obj: object) -> object:
    """Mask every string inside a nested structure BEFORE serialization:
    JSON escaping changes quotes/backslashes, so masking the final line
    alone would miss secrets in escaped form."""
    if isinstance(obj, str):
        return SECRETS.mask(obj)
    if isinstance(obj, dict):
        return {k: deep_mask(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [deep_mask(v) for v in obj]
    return obj


class _MaskingJsonFormatter(logging.Formatter):
    """One JSON object per line; values masked before escaping, line after."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in getattr(record, "fields", {}).items():
            entry[key] = value
        line = json.dumps(deep_mask(entry), ensure_ascii=False, default=str)
        return SECRETS.mask(line)


def setup_logging(verbose: bool = False, stream=None) -> None:
    """Configure the razkadr logger: JSON lines to stderr, INFO or DEBUG."""
    logger = logging.getLogger("razkadr")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False
    for old in list(logger.handlers):
        logger.removeHandler(old)
    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(_MaskingJsonFormatter())
    logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"razkadr.{name}")

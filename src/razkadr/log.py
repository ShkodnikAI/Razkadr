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
    lowered = name.lower()
    return any(marker in lowered for marker in SECRET_MARKERS)


class SecretRegistry:
    def __init__(self) -> None:
        self._secrets: set[str] = set()

    def register(self, value: object) -> None:
        """Register a secret value for masking; empty values are ignored."""
        if isinstance(value, str) and value.strip():
            self._secrets.add(value)

    def mask(self, text: str) -> str:
        for secret in self._secrets:
            if secret in text:
                text = text.replace(secret, MASK)
        return text

    @property
    def registered(self) -> int:
        return len(self._secrets)


SECRETS = SecretRegistry()


class _MaskingJsonFormatter(logging.Formatter):
    """One JSON object per line; the whole line is masked before output."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in getattr(record, "fields", {}).items():
            entry[key] = value
        line = json.dumps(entry, ensure_ascii=False, default=str)
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

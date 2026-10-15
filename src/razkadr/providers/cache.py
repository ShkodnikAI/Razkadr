"""Disk cache of provider calls: repeat call with the same inputs never
reaches the provider (naryad 033).

The key is sha256 over the canonical JSON of task, provider name, model,
model_version and all inputs/params - the auditor's requirement that the
cache key include model and parameters, not only the input. Entries are
JSON files: {key, task, provider, model, units, result, created_utc}.
Result values must be JSON-serializable (dataclasses convert first).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from razkadr.log import SECRETS, deep_mask, get_logger
from razkadr.runctx import canonical_json

logger = get_logger("providers.cache")

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "razkadr" / "calls"


def call_key(task: str, provider: str, model: str, inputs: dict, params: dict) -> str:
    payload = {
        "task": task,
        "provider": provider,
        "model": model,
        "inputs": inputs,
        "params": params,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


class CallCache:
    def __init__(self, directory: str | Path | None = None) -> None:
        self.dir = Path(directory) if directory is not None else DEFAULT_CACHE_DIR
        self.dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> dict | None:
        path = self.dir / f"{key}.json"
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("corrupt cache entry removed: %s", key[:12])
            path.unlink(missing_ok=True)
            return None

    def put(self, key: str, entry: dict) -> None:
        path = self.dir / f"{key}.json"
        data = {"key": key, **entry, "created_utc": entry.get("created_utc")}
        text = SECRETS.mask(json.dumps(deep_mask(data), ensure_ascii=False, sort_keys=True))
        tmp = path.with_suffix(".tmp")
        tmp.write_text(text + "\n", encoding="utf-8")
        tmp.replace(path)

"""Configuration loading: razkadr.toml, env overrides, secret handling.

Search order for the config file: explicit --config path, ./razkadr.toml,
~/.config/razkadr/razkadr.toml; otherwise defaults. Environment variables
RAZKADR_<NAME> override config keys (name lowercased). Names that look
secret (contain key/token/secret/password) never enter the effective
config or its hash: their values are registered for masking instead.
"""

from __future__ import annotations

import hashlib
import json
import os
import tomllib
from pathlib import Path

from razkadr.log import SECRETS, is_secret_name

DEFAULT_CONFIG: dict = {"runs_dir": "runs"}
CONFIG_FILENAME = "razkadr.toml"
ENV_PREFIX = "RAZKADR_"


def _read_toml(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def find_config_file(explicit: str | Path | None = None) -> Path | None:
    if explicit is not None:
        path = Path(explicit)
        return path if path.is_file() else None
    for candidate in (Path.cwd() / CONFIG_FILENAME, Path.home() / ".config" / "razkadr" / CONFIG_FILENAME):
        if candidate.is_file():
            return candidate
    return None


def _absorb_secrets(node: dict, prefix: str = "") -> dict:
    """Return a copy of node with secret-named values moved to the registry."""
    result: dict = {}
    for key, value in node.items():
        if isinstance(value, dict):
            result[key] = _absorb_secrets(value, prefix=f"{prefix}{key}.")
        elif is_secret_name(key):
            SECRETS.store(f"{prefix}{key}", value)
        else:
            result[key] = value
    return result


def load_config(explicit: str | Path | None = None) -> dict:
    """Effective config: defaults <- toml file <- RAZKADR_* env vars.

    Secret-valued keys (any nesting level) are removed from the result and
    stored in the secret registry under a dotted name (e.g. ``asr.api_key``):
    they never reach the config hash, logs or artifacts.
    """
    effective = dict(DEFAULT_CONFIG)
    config_file = find_config_file(explicit)
    if config_file is not None:
        effective.update(_absorb_secrets(_read_toml(config_file)))
    for env_name, env_value in os.environ.items():
        if not env_name.startswith(ENV_PREFIX) or len(env_name) <= len(ENV_PREFIX):
            continue
        key = env_name[len(ENV_PREFIX):].lower()
        if is_secret_name(env_name):
            SECRETS.store(env_name, env_value)
        else:
            effective[key] = env_value
    return effective


def config_hash(config: dict) -> str:
    """sha256 of the canonical JSON of the effective (non-secret) config."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

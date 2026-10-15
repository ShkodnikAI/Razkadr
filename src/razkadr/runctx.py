"""RunContext: run directory, stage execution, cache, run.json accounting.

Every stage runs in the same shape (naryad 036):

    ctx = RunContext(runs_dir)
    with ctx.stage("frames", inputs={"video": sha}, params={"fps": 10}) as st:
        if st.cached:
            return          # artifact already exists in this run
        ... write artifacts ...

The context manager measures wall time, records status (ok/cached/failed)
and writes run.json after every stage event, so a stage that fails midway
leaves a failed entry with the reason and the run can be resumed: stages
with unchanged cache keys are reported cached, the failed one re-runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import secrets as pysecrets
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from razkadr import __version__
from razkadr.config import config_hash
from razkadr.log import SECRETS, deep_mask, get_logger

logger = get_logger("runctx")

CACHE_DIRNAME = ".cache"
RUN_JSON = "run.json"
UNIT_KINDS = ("seconds", "tokens", "images")
STAGE_STATUSES = ("ok", "failed", "skipped", "cached")


def utcnow_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(obj: object) -> str:
    """Stable fingerprint of arbitrary JSON-able stage inputs/params."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


@dataclass
class StageHandle:
    """Passed into the stage body; cached=True means the body must be skipped."""

    name: str
    key: str
    cached: bool
    outputs: dict = field(default_factory=dict)


class RunContext:
    def __init__(
        self,
        runs_dir: str | Path,
        run_id: str | None = None,
        config: dict | None = None,
        force: bool = False,
    ) -> None:
        self.runs_dir = Path(runs_dir)
        self.config = dict(config or {})
        self.force = force
        self.run_id = run_id or self._new_run_id()
        self.dir = self.runs_dir / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / CACHE_DIRNAME).mkdir(exist_ok=True)
        self.created_utc = utcnow_iso()
        self.stages: list[dict] = []
        self.provider_calls: list[dict] = []
        self._load_existing_run()
        self._write_run_json()

    # -- public API ----------------------------------------------------------

    @contextmanager
    def stage(self, name: str, inputs: dict | None = None, params: dict | None = None) -> Iterator[StageHandle]:
        """Run a stage with cache and accounting; see module docstring."""
        key = self._stage_key(name, inputs, params)
        cached = not self.force and self._cached_key(name) == key
        handle = StageHandle(name=name, key=key, cached=cached)
        started = datetime.now(UTC)
        logger.info("stage start", extra={"fields": {"stage": name, "key": key[:12], "cached": cached}})
        try:
            yield handle
        except Exception as exc:
            self._record_stage(name, status="failed", started=started, error=f"{type(exc).__name__}: {exc}")
            logger.error("stage failed", extra={"fields": {"stage": name, "error": str(exc)}})
            raise
        duration = (datetime.now(UTC) - started).total_seconds()
        status = "cached" if handle.cached else "ok"
        if not handle.cached:
            self._write_marker(name, key)
        self._record_stage(name, status=status, started=started, duration_s=round(duration, 3))
        logger.info("stage done", extra={"fields": {"stage": name, "status": status, "duration_s": round(duration, 3)}})

    def record_provider_call(
        self,
        stage: str,
        provider: str,
        model: str,
        units: list[dict],
        cost_amount: float,
        currency: str,
        tariff_date: str,
        cache_hit: bool,
        task: str | None = None,
    ) -> None:
        """Append one provider-call accounting record (schema run.v1.json)."""
        for unit in units:
            if unit.get("kind") not in UNIT_KINDS:
                raise ValueError(f"unit kind must be one of {UNIT_KINDS}, got {unit.get('kind')!r}")
        call: dict = {
            "stage": stage,
            "provider": provider,
            "model": model,
            "units": [{"kind": u["kind"], "value": u["value"]} for u in units],
            "cost": {"amount": cost_amount, "currency": currency, "tariff_date": tariff_date},
            "cache_hit": cache_hit,
        }
        if task is not None:
            call["task"] = task
        self.provider_calls.append(call)
        self._write_run_json()

    # -- internals -----------------------------------------------------------

    def _new_run_id(self) -> str:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        return f"{stamp}_{pysecrets.token_hex(3)}"

    def _load_existing_run(self) -> None:
        """Resume: keep original created_utc and accumulate stages/calls."""
        path = self.dir / RUN_JSON
        if not path.is_file():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("run.json unreadable, starting fresh accounting")
            return
        self.created_utc = data.get("created_utc", self.created_utc)
        self.stages = list(data.get("stages", []))
        self.provider_calls = list(data.get("provider_calls", []))

    def _stage_key(self, name: str, inputs: dict | None, params: dict | None) -> str:
        versions = {"razkadr": __version__, "python": platform.python_version()}
        payload = {"stage": name, "inputs": inputs or {}, "params": params or {}, "versions": versions}
        return fingerprint(payload)

    def _marker_path(self, name: str) -> Path:
        return self.dir / CACHE_DIRNAME / f"{name}.json"

    def _cached_key(self, name: str) -> str | None:
        path = self._marker_path(name)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8")).get("key")
        except json.JSONDecodeError:
            return None

    def _write_marker(self, name: str, key: str) -> None:
        marker = {"stage": name, "key": key, "finished_utc": utcnow_iso()}
        self._write_json(self._marker_path(name), marker)

    def _record_stage(
        self,
        name: str,
        status: str,
        started: datetime,
        duration_s: float | None = None,
        error: str | None = None,
    ) -> None:
        entry: dict = {"name": name, "status": status, "started_utc": started.strftime("%Y-%m-%dT%H:%M:%SZ")}
        if duration_s is not None:
            entry["duration_s"] = duration_s
        if error is not None:
            entry["error"] = error
        self.stages.append(entry)
        self._write_run_json()

    def _write_run_json(self) -> None:
        data = {
            "schema_version": "1.0",
            "run_id": self.run_id,
            "created_utc": self.created_utc,
            "razkadr_version": __version__,
            "config_hash": config_hash(self.config),
            "tool_versions": {"python": platform.python_version()},
            "stages": self.stages,
            "provider_calls": self.provider_calls,
        }
        self._write_json(self.dir / RUN_JSON, data)

    def _write_json(self, path: Path, data: object) -> None:
        """Atomic write; values masked before serialization (and the line after)."""
        text = SECRETS.mask(canonical_json(deep_mask(data)))
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text + "\n", encoding="utf-8")
        os.replace(tmp, path)

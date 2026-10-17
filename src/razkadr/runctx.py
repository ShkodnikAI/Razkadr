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


def read_cached_json(path: Path, what: str, error_cls: type[Exception]) -> dict:
    """Read the stage's own artifact on a cache hit (naryad 029).

    The cache marker says the stage is current, so an unreadable file
    means disk damage or manual editing - not a crash mid-write (atomic
    writes, also naryad 029, make that impossible for our own output).
    The message names both remedies: recompute with --force, or delete
    the file and resume.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        detail = getattr(exc, "strerror", None) or exc
        raise error_cls(
            f"{what} is not valid JSON: {path}: {detail} - the stage cache says it is "
            "current; re-run with --force to recompute the stage, or delete the file"
        ) from exc


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
        self.tool_versions: dict[str, str] = {"python": platform.python_version()}
        self.stages: list[dict] = []
        self.provider_calls: list[dict] = []
        self._summary: dict | None = None
        self._load_existing_run()
        self._write_run_json()

    def register_tool(self, name: str, version: str) -> None:
        """Register an external tool version for run.json tool_versions.

        Stages call this for the tools they invoke (ingest: ffmpeg, yt-dlp;
        naryad 005). The run schema (run.v1.json) keeps tool_versions as a
        free string map for exactly this purpose.
        """
        if not name or not version:
            raise ValueError("tool name and version must be non-empty")
        self.tool_versions[name] = str(version)

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

    def write_summary(self, artifacts: dict[str, float] | None = None) -> dict:
        """Write the final run summary into run.json (naryad 011) and return it.

        Called once by the command after every stage has finished. Three
        blocks: ``stage_times_s`` (last completed attempt per stage),
        ``provider_cost`` (provider_calls aggregated) and ``artifacts``
        (key artifact numbers passed by the caller). A failed run gets no
        summary; the stage log and call accounting above are enough to
        diagnose it.
        """
        stage_times: dict[str, float] = {}
        for entry in self.stages:
            if entry.get("status") == "failed" or entry.get("duration_s") is None:
                continue
            stage_times[entry["name"]] = entry["duration_s"]  # later entries overwrite

        currencies = {c["cost"]["currency"] for c in self.provider_calls}
        if len(currencies) > 1:
            raise ValueError(f"mixed currencies in provider calls: {sorted(currencies)}")
        by_stage: dict[str, float] = {}
        total = 0.0
        for call in self.provider_calls:
            amount = call["cost"]["amount"]
            total += amount
            by_stage[call["stage"]] = round(by_stage.get(call["stage"], 0.0) + amount, 6)
        provider_cost: dict = {
            "amount": round(total, 6),
            "calls": len(self.provider_calls),
            "cache_hits": sum(1 for c in self.provider_calls if c["cache_hit"]),
            "by_stage": by_stage,
        }
        if currencies:
            provider_cost["currency"] = next(iter(currencies))

        artifact_counts: dict[str, float] = {}
        for name, value in (artifacts or {}).items():
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
                raise ValueError(f"summary artifact {name!r} must be a number >= 0, got {value!r}")
            artifact_counts[name] = value

        self._summary = {
            "stage_times_s": stage_times,
            "provider_cost": provider_cost,
            "artifacts": artifact_counts,
        }
        self._write_run_json()
        return self._summary

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
            "schema_version": "1.1",
            "run_id": self.run_id,
            "created_utc": self.created_utc,
            "razkadr_version": __version__,
            "config_hash": config_hash(self.config),
            "tool_versions": dict(self.tool_versions),
            "stages": self.stages,
            "provider_calls": self.provider_calls,
        }
        if self._summary is not None:
            data["summary"] = self._summary
        self._write_json(self.dir / RUN_JSON, data)

    def _write_json(self, path: Path, data: object) -> None:
        """Atomic write; values masked before serialization (and the line after)."""
        text = SECRETS.mask(canonical_json(deep_mask(data)))
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text + "\n", encoding="utf-8")
        os.replace(tmp, path)

"""Eval harness: hit@k over a labeled reference set (naryad 015).

`razkadr eval data/eval --config A.toml --config B.toml` measures how well
the hybrid search (naryad 012) finds the labeled moments of the reference
set (naryad 003 labels: hero / ingredients / steps) for every configuration:

- each reference video is indexed with the configuration's pipeline
  (ingest -> frames -> transcript -> chunks -> ocr -> index), the cost and
  time of that run are recorded from run.json accounting;
- each role's queries (preset naryad 013) run through the search with the
  configuration's [search] settings (signals on/off: text-only vs
  frames-only vs hybrid, different embedding providers);
- a label moment counts as HIT@k when one of the top-k candidate frames of
  the role falls within `tolerance` seconds (default 2.0, the contract's
  matching tolerance) of the moment's timecode;
- a video joins a role's denominator only when the labels mark that role
  (auditor rule); hit@k per role = covered moments / all labeled moments of
  the videos that mark the role.

Output: <out-dir>/report.json (reproducible: tool versions, config hashes,
label hashes, per-video results) and report.md (the gateway-friendly
table). Misses are listed per video with the candidate frames and their
timecodes. Reference videos live outside git (naryad 003); reports go to
the out directory, never committed by the harness.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from razkadr import __version__
from razkadr.log import get_logger
from razkadr.presets import load_preset
from razkadr.search import SearchConfig, SearchError, role_candidates
from razkadr.validation import validate_document

logger = get_logger("eval")

LABEL_ROLES = ("hero", "ingredients", "steps")
DEFAULT_TOLERANCE_S = 2.0
DEFAULT_K_LIST = (1, 3)
REPORT_JSON = "report.json"
REPORT_MD = "report.md"


class EvalError(Exception):
    """The evaluation cannot run (bad set, bad config, pipeline failure)."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ffmpeg_version() -> str:
    try:
        out = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, timeout=15, check=True
        )
        return out.stdout.splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def load_eval_set(eval_dir: str | Path) -> dict:
    """Videos + labels of the reference set; labels validated (naryad 003).

    Returns {"dir", "videos": [(video_id, path)], "labels": {id: doc}};
    videos without a labels file are reported in "missing_labels"."""
    root = Path(eval_dir)
    videos_dir = root / "videos"
    labels_dir = root / "labels"
    if not videos_dir.is_dir() or not labels_dir.is_dir():
        raise EvalError(
            f"reference set layout expected at {root}: videos/ and labels/ (see data/eval/README.md)"
        )
    videos: list[tuple[str, Path]] = []
    missing: list[str] = []
    labels: dict[str, dict] = {}
    for video in sorted(videos_dir.iterdir()):
        if video.suffix.lower() not in (".mp4", ".mkv", ".mov", ".webm"):
            continue
        video_id = video.stem
        videos.append((video_id, video))
        label_path = labels_dir / f"{video_id}.json"
        if not label_path.is_file():
            missing.append(video_id)
            continue
        doc = json.loads(label_path.read_text(encoding="utf-8"))
        issues = validate_document(doc, "eval_labels")
        if issues:
            raise EvalError(
                f"labels are invalid for {video_id}: " + "; ".join(str(i) for i in issues)
            )
        labels[video_id] = doc
    if not videos:
        raise EvalError(f"no reference videos in {videos_dir}")
    return {"dir": root, "videos": videos, "labels": labels, "missing_labels": missing}


def index_video(video_path: Path, config: dict) -> tuple[Path, dict]:
    """Run the full indexing pipeline on one video; returns (run_dir, summary)."""
    from razkadr.providers.base import ProviderError
    from razkadr.runctx import RunContext
    from razkadr.stages.chunks import ChunksError, run_chunks
    from razkadr.stages.frames import FramesError, run_frames
    from razkadr.stages.index import IndexStageError, run_index
    from razkadr.stages.ingest import IngestError, run_ingest
    from razkadr.stages.ocr import OcrError, run_ocr
    from razkadr.stages.transcript import TranscriptError, run_transcript

    ctx = RunContext(config.get("runs_dir", "runs"), config=config)
    try:
        run_ingest(ctx, str(video_path), config)
        run_frames(ctx, config)
        run_transcript(ctx, config)
        run_chunks(ctx, config)
        run_ocr(ctx, config)
        run_index(ctx, config)
    except (
        IngestError, FramesError, TranscriptError, ChunksError, OcrError, IndexStageError, ProviderError
    ) as exc:
        raise EvalError(f"indexing {video_path.name} failed: {exc}") from exc
    summary = ctx.write_summary()
    return ctx.dir, summary


def hit_counts(
    candidates_t: list[float], moments_t: list[float], k: int, tolerance: float
) -> tuple[int, int]:
    """(covered moments, all moments) for one video-role: a moment is
    covered when one of the top-k candidates is within tolerance of it."""
    top = candidates_t[:k]
    covered = 0
    for moment in moments_t:
        if any(abs(t - moment) <= tolerance for t in top):
            covered += 1
    return covered, len(moments_t)


def _role_moments(labels: dict, role: str) -> list[float]:
    return [float(m["t"]) for m in (labels.get(role) or []) if "t" in m]


def evaluate_configuration(
    eval_set: dict,
    name: str,
    config: dict,
    config_path: Path,
    preset: dict,
    k_list: list[int],
    tolerance: float,
    keep_runs: bool,
) -> dict:
    """Index every video with the config and measure hit@k per role."""
    search_cfg = SearchConfig.from_config(config)
    runs_root = Path(config.get("runs_dir", "runs"))
    results: dict = {
        "name": name,
        "config_path": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "search": {"signals": dict(search_cfg.signals), "weights": dict(search_cfg.weights),
                   "rrf_k": search_cfg.rrf_k, "k": search_cfg.k},
        "cost": {"amount": 0.0, "currency": "USD", "calls": 0, "cache_hits": 0},
        "index_time_s": 0.0,
        "video_seconds": 0.0,
        "roles": {
            role: {"moments": 0, "videos": 0, "hits_by_k": {str(k): 0 for k in k_list}, "misses": []}
            for role in LABEL_ROLES
        },
        "per_video": {},
    }
    report_warnings: list[str] = list(eval_set["missing_labels"])

    for video_id, video_path in eval_set["videos"]:
        labels = eval_set["labels"].get(video_id)
        if labels is None:
            logger.warning("no labels for %s; skipped", video_id)
            continue
        run_dir, summary = index_video(video_path, config)
        source = json.loads((run_dir / "source.json").read_text(encoding="utf-8"))
        duration = float(source["duration_s"])
        results["cost"]["amount"] += float(summary["provider_cost"].get("amount", 0.0))
        results["cost"]["currency"] = summary["provider_cost"].get("currency", "USD")
        results["cost"]["calls"] += int(summary["provider_cost"].get("calls", 0))
        results["cost"]["cache_hits"] += int(summary["provider_cost"].get("cache_hits", 0))
        results["index_time_s"] += float(sum(summary.get("stage_times_s", {}).values()))
        results["video_seconds"] += duration

        # the service set is built per run: the provider layer needs the run ctx
        from razkadr.providers.layer import ProviderLayer
        from razkadr.runctx import RunContext

        ctx = RunContext(runs_root, run_id=run_dir.name, config=config)
        layer = ProviderLayer(config, ctx)
        embeddings = layer.embeddings(stage="eval")
        llm = layer.llm(stage="eval") if (config.get("llm") or {}).get("provider") else None

        per_video: dict = {"duration_s": duration, "roles": {}}
        for role in LABEL_ROLES:
            moments = _role_moments(labels, role)
            if not moments:
                continue  # auditor rule: not in the denominator
            results["roles"][role]["videos"] += 1
            spec = preset["roles"].get(role)
            if spec is None:
                raise EvalError(f"preset has no role '{role}' for labels of {video_id}")
            # the label tags carry the language (ru/en) first when present
            query_lang = next((tag for tag in labels.get("tags") or [] if tag in spec["queries"]), None)
            queries = spec["queries"].get(query_lang) if query_lang else None
            if not queries:
                queries = [q for qs in spec["queries"].values() for q in qs]
            try:
                candidates = role_candidates(
                    run_dir, queries, [query_lang] * len(queries) if query_lang else None,
                    embeddings=embeddings, llm=llm,
                    config={"search": search_cfg.__dict__ | {"signals": dict(search_cfg.signals)}},
                )
            except SearchError as exc:
                raise EvalError(f"{video_id}/{role}: {exc}") from exc
            candidates_t = [hit.t for hit in candidates]
            role_result = {"moments": len(moments), "candidates_top": []}
            results["roles"][role]["moments"] += len(moments)
            for k in k_list:
                covered, _total = hit_counts(candidates_t, moments, k, tolerance)
                results["roles"][role]["hits_by_k"][str(k)] += covered
            for hit in candidates[: max(k_list)]:
                role_result["candidates_top"].append({"file": hit.file, "t": hit.t})
            missed = [
                m for m in moments
                if not any(abs(t - m) <= tolerance for t in candidates_t[: max(k_list)])
            ]
            if missed:
                results["roles"][role]["misses"].append(
                    {"video": video_id, "missed_moments_s": missed,
                     "top_candidates": role_result["candidates_top"]}
                )
            per_video["roles"][role] = role_result
        results["per_video"][video_id] = per_video
        if not keep_runs:
            shutil.rmtree(run_dir, ignore_errors=True)
    return results, report_warnings


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _render_markdown(report: dict, k_list: list[int], tolerance: float) -> str:
    header = (
        f"Пресет: `{report['preset']}` · допуск: ±{tolerance:g} с · k: {', '.join(map(str, k_list))} · "
        f"razkadr {report['versions']['razkadr']} · python {report['versions']['python']}"
    )
    lines = [
        f"# Eval report — {report['generated_utc']}",
        "",
        header,
        f"ffmpeg: {report['versions']['ffmpeg']}",
        "",
    ]
    for name, result in report["results"].items():
        lines.append(f"## Конфигурация `{name}`")
        lines.append("")
        cost = result["cost"]
        minutes = result["video_seconds"] / 60.0
        per_min = cost["amount"] / minutes if minutes else 0.0
        lines.append(
            f"Стоимость индексации: {cost['amount']:.6f} {cost['currency']} "
            f"({cost['calls']} вызовов, {cost['cache_hits']} из кэша), "
            f"{per_min:.6f} {cost['currency']}/мин видео, время индексации {result['index_time_s']:.3f} с"
        )
        lines.append("")
        lines.append("| Роль | Видео | Моменты | " + " | ".join(f"hit@{k}" for k in k_list) + " |")
        lines.append("|---|---|---|" + "---|" * len(k_list))
        for role, data in result["roles"].items():
            if not data["videos"]:
                continue
            cells = [
                _fmt_pct(data["hits_by_k"][str(k)] / data["moments"] if data["moments"] else None)
                for k in k_list
            ]
            lines.append(
                f"| {role} | {data['videos']} | {data['moments']} | " + " | ".join(cells) + " |"
            )
        lines.append("")
        misses = [(role, m) for role, data in result["roles"].items() for m in data["misses"]]
        if misses:
            lines.append("### Промахи")
            lines.append("")
            for role, miss in misses:
                tops = ", ".join(f"{c['file']}@{c['t']:g}с" for c in miss["top_candidates"])
                lines.append(
                    f"- `{miss['video']}` роль **{role}**: моменты "
                    f"{', '.join(f'{m:g}с' for m in miss['missed_moments_s'])}; топ кандидатов: {tops}"
                )
            lines.append("")
    return "\n".join(lines)


def run_eval(
    eval_dir: str | Path,
    config_paths: list[str | Path],
    k_list: list[int] | None = None,
    tolerance: float = DEFAULT_TOLERANCE_S,
    preset_path: str | Path = "presets/recipe.yaml",
    out_dir: str | Path | None = None,
    keep_runs: bool = False,
) -> tuple[Path, Path]:
    """Run the whole harness; writes report.json + report.md, returns paths."""
    if not config_paths:
        raise EvalError("at least one --config is required")
    k_list = sorted(set(k_list or list(DEFAULT_K_LIST)))
    if any(k < 1 for k in k_list):
        raise EvalError(f"k values must be >= 1, got {k_list}")
    if tolerance < 0:
        raise EvalError(f"tolerance must be >= 0, got {tolerance}")
    root = Path(out_dir) if out_dir else Path("eval")
    root.mkdir(parents=True, exist_ok=True)
    eval_set = load_eval_set(eval_dir)
    preset = load_preset(preset_path)

    results = {}
    warnings: list[str] = []
    for config_path in config_paths:
        path = Path(config_path)
        if not path.is_file():
            raise EvalError(f"config not found: {path}")
        from razkadr.config import load_config

        name = path.stem
        config = load_config(path)
        # isolate runs per config inside the eval out dir
        config["runs_dir"] = str(root / "runs" / name)
        result, config_warnings = evaluate_configuration(
            eval_set, name, config, path, preset, k_list, tolerance, keep_runs
        )
        results[name] = result
        warnings.extend(f"[{name}] {w}" for w in config_warnings)

    report = {
        "schema": "eval-report.v1",
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "preset": preset.get("name"),
        "tolerance_s": tolerance,
        "k_list": k_list,
        "eval_dir": str(eval_dir),
        "labels_sha256": {
            video_id: _sha256_file(Path(eval_dir) / "labels" / f"{video_id}.json")
            for video_id in eval_set["labels"]
        },
        "versions": {
            "razkadr": __version__,
            "python": platform.python_version(),
            "ffmpeg": _ffmpeg_version(),
        },
        "results": results,
        "warnings": warnings,
    }
    json_path = root / REPORT_JSON
    md_path = root / REPORT_MD
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(report, k_list, tolerance), encoding="utf-8")
    return json_path, md_path

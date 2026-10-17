"""Cost report aggregator (naryad 030): run.json files -> markdown tables.

Package module; the command-line entry point is the thin wrapper
``scripts/cost_report.py``.

Reads one or more run directories (razkadr index / razkadr build output),
sums the provider calls and stage times recorded by the naryad 033 layer,
normalizes per video minute and emits a markdown fragment for
docs/cost.md. Deterministic arithmetic over recorded runs - no providers,
no network, nothing billed here:

    python scripts/cost_report.py runs/<id> --label "cold" \
        [--label ...] [--out cost-tables.md]

Every run directory must contain run.json and source.json (the source
duration drives the per-minute normalization). Mixed currencies across
runs are rejected the same way runctx rejects them inside one run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

UNIT_LABELS = {"seconds": "s", "tokens": "tok", "images": "img"}


def load_run(run_dir: str | Path) -> dict:
    """Read run.json + source.json; verify the run finished with a summary."""
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if "summary" not in run:
        raise SystemExit(f"cost-report: {run_dir}: run.json has no summary (the run did not finish)")
    source = json.loads((run_dir / "source.json").read_text(encoding="utf-8"))
    duration_s = float(source["duration_s"])
    if duration_s <= 0:
        raise SystemExit(f"cost-report: {run_dir}: source duration is {duration_s}")
    return {"dir": run_dir, "run": run, "duration_s": duration_s}


def _fmt_units(units: list[dict]) -> str:
    parts = []
    for unit in units:
        label = UNIT_LABELS.get(unit["kind"], unit["kind"])
        parts.append(f"{unit['value']:g} {label}")
    return ", ".join(parts) if parts else "-"


def stage_rows(entries: dict) -> list[list[str]]:
    """stage -> [stage, time s, calls, cache hits, units, cost]."""
    rows: dict[str, dict] = {}
    for call in entries["run"]["provider_calls"]:
        row = rows.setdefault(
            call["stage"],
            {"time": entries["run"]["summary"]["stage_times_s"].get(call["stage"], 0.0),
             "calls": 0, "cache_hits": 0, "units": {}, "cost": 0.0},
        )
        row["calls"] += 1
        row["cache_hits"] += 1 if call["cache_hit"] else 0
        row["cost"] = round(row["cost"] + call["cost"]["amount"], 6)
        for unit in call["units"]:
            key = unit["kind"]
            row["units"][key] = row["units"].get(key, 0.0) + float(unit["value"])
    ordered = sorted(rows.items(), key=lambda kv: kv[1]["time"], reverse=True)
    return [
        [
            stage,
            f"{row['time']:g}",
            str(row["calls"]),
            str(row["cache_hits"]),
            _fmt_units([{"kind": k, "value": v} for k, v in sorted(row["units"].items())]),
            f"{row['cost']:.6f}".rstrip("0").rstrip(".") or "0",
        ]
        for stage, row in ordered
    ]


def provider_rows(entries: dict) -> list[list[str]]:
    """provider+model -> [provider, model, calls, cache hits, cost]."""
    rows: dict[tuple[str, str], dict] = {}
    for call in entries["run"]["provider_calls"]:
        key = (call["provider"], call["model"])
        row = rows.setdefault(key, {"calls": 0, "cache_hits": 0, "cost": 0.0})
        row["calls"] += 1
        row["cache_hits"] += 1 if call["cache_hit"] else 0
        row["cost"] = round(row["cost"] + call["cost"]["amount"], 6)
    return [
        [provider, model, str(row["calls"]), str(row["cache_hits"]),
         f"{row['cost']:.6f}".rstrip("0").rstrip(".") or "0"]
        for (provider, model), row in sorted(rows.items())
    ]


def _currencies(entries: list[dict]) -> set[str]:
    found = {
        call["cost"]["currency"]
        for entry in entries
        for call in entry["run"]["provider_calls"]
        if call["cost"].get("currency")
    }
    return found or {"USD"}


def render(entries: list[dict], labels: list[str]) -> str:
    """The whole markdown fragment: per-run tables + totals + per-minute."""
    if len(entries) != len(labels):
        raise SystemExit("cost-report: one --label per run directory is required")
    currencies = _currencies(entries)
    if len(currencies) > 1:
        raise SystemExit(f"cost-report: mixed currencies across runs: {sorted(currencies)}")
    currency = next(iter(currencies))
    lines: list[str] = []

    grand_calls = 0
    grand_cost = 0.0
    for entry, label in zip(entries, labels):
        run = entry["run"]
        total_calls = len(run["provider_calls"])
        cache_hits = sum(1 for c in run["provider_calls"] if c["cache_hit"])
        total_cost = run["summary"]["provider_cost"]["amount"]
        grand_calls += total_calls
        grand_cost = round(grand_cost + total_cost, 6)
        per_min = total_cost * 60.0 / entry["duration_s"]
        lines.append(f"### Прогон «{label}» — `{entry['dir']}` ({entry['duration_s']:g} с)")
        lines.append("")
        lines.append("| стадия | время, с | вызовы | из кэша | единицы | стоимость |")
        lines.append("|---|---:|---:|---:|---|---:|")
        for row in stage_rows(entry):
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        lines.append("| провайдер | модель | вызовы | из кэша | стоимость |")
        lines.append("|---|---|---:|---:|---:|")
        for row in provider_rows(entry):
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        lines.append(
            f"Итого: {total_calls} вызовов ({cache_hits} из кэша), "
            f"**{total_cost:.6f} {currency}**; "
            f"себестоимость минуты: **{per_min:.4f} {currency}/мин** "
            f"(источник {entry['duration_s']:g} с)."
        )
        lines.append("")

    lines.append("### Сводка серии")
    lines.append("")
    lines.append("| прогон | длительность, с | вызовы | стоимость | за минуту |")
    lines.append("|---|---:|---:|---:|---:|")
    for entry, label in zip(entries, labels):
        total_cost = entry["run"]["summary"]["provider_cost"]["amount"]
        per_min = total_cost * 60.0 / entry["duration_s"]
        lines.append(
            f"| {label} | {entry['duration_s']:g} | "
            f"{len(entry['run']['provider_calls'])} | {total_cost:.6f} | {per_min:.4f} |"
        )
    lines.append("")
    lines.append(
        f"Всего по серии: {grand_calls} вызовов, {grand_cost:.6f} {currency}. "
        "Все числа — арифметика над записанными `run.json` (слоя провайдеров, наряд №033); "
        "ничего не тарифицируется здесь."
    )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate run.json files into cost tables (naryad 030)")
    parser.add_argument("runs", nargs="+", help="Run directories (contain run.json + source.json)")
    parser.add_argument(
        "--label", action="append", default=[],
        help="Human label for the run (repeat in the same order as the directories)",
    )
    parser.add_argument("--out", default=None, help="Write the markdown fragment to PATH (default: stdout)")
    args = parser.parse_args(argv)

    entries = [load_run(run_dir) for run_dir in args.runs]
    labels = args.label or [Path(entry["dir"]).name for entry in entries]
    text = render(entries, labels)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

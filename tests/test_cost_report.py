"""Cost report aggregator tests (naryad 030): run.json -> markdown tables."""

import json
from pathlib import Path

import pytest

from razkadr.cost_report import load_run, render, stage_rows


def make_run(
    base: Path,
    *,
    name: str = "run-1",
    duration_s: float = 60.0,
    calls: list[dict] | None = None,
) -> Path:
    """A minimal finished run: source.json + run.json with a summary."""
    run_dir = base / name
    run_dir.mkdir(parents=True)
    (run_dir / "source.json").write_text(
        json.dumps({"schema_version": "1.0", "duration_s": duration_s}), encoding="utf-8"
    )
    if calls is None:
        calls = [
            {
                "stage": "transcript", "provider": "fake", "model": "fake-asr",
                "units": [{"kind": "seconds", "value": 30.0}],
                "cost": {"amount": 0.177, "currency": "USD", "tariff_date": "2026-10-01"},
                "cache_hit": False,
            },
            {
                "stage": "ocr", "provider": "fake", "model": "fake-vision",
                "units": [{"kind": "images", "value": 2.0}, {"kind": "tokens", "value": 100.0}],
                "cost": {"amount": 0.003, "currency": "USD", "tariff_date": "2026-10-01"},
                "cache_hit": True,
            },
        ]
    (run_dir / "run.json").write_text(
        json.dumps({
            "run_id": name,
            "stages": [
                {"name": "transcript", "status": "ok", "duration_s": 2.5},
                {"name": "ocr", "status": "cached", "duration_s": 0.1},
            ],
            "provider_calls": calls,
            "summary": {
                "stage_times_s": {"transcript": 2.5, "ocr": 0.1},
                "provider_cost": {
                    "amount": round(sum(c["cost"]["amount"] for c in calls), 6),
                    "calls": len(calls),
                    "cache_hits": sum(1 for c in calls if c["cache_hit"]),
                },
                "artifacts": {},
            },
        }),
        encoding="utf-8",
    )
    return run_dir


def test_load_run_requires_summary(tmp_path: Path):
    entry = load_run(make_run(tmp_path))
    assert entry["duration_s"] == 60.0

    unfinished = tmp_path / "run-2"
    unfinished.mkdir()
    (unfinished / "source.json").write_text(json.dumps({"duration_s": 60.0}), encoding="utf-8")
    (unfinished / "run.json").write_text(json.dumps({"stages": []}), encoding="utf-8")
    with pytest.raises(SystemExit, match="no summary"):
        load_run(unfinished)


def test_stage_rows_aggregate_calls_units_cost(tmp_path: Path):
    entry = load_run(make_run(tmp_path))
    by_stage = {row[0]: row for row in stage_rows(entry)}
    assert by_stage["transcript"][2] == "1" and by_stage["transcript"][3] == "0"
    assert by_stage["transcript"][4] == "30 s"
    assert by_stage["transcript"][5] == "0.177"
    assert by_stage["ocr"][3] == "1", "cache-hit calls are counted separately"
    assert "2 img" in by_stage["ocr"][4] and "100 tok" in by_stage["ocr"][4]


def test_render_per_minute_and_totals(tmp_path: Path):
    entry = load_run(make_run(tmp_path))
    text = render([entry], ["холодный"])
    assert "0.1800 USD/мин" in text, "60 s source: the per-minute rate equals the run cost"
    assert "**0.180000 USD**" in text
    assert "0.177" in text and "0.003" in text
    assert "арифметика над записанными" in text


def test_render_rejects_label_mismatch(tmp_path: Path):
    entry = load_run(make_run(tmp_path))
    with pytest.raises(SystemExit, match="one --label per run"):
        render([entry], [])
    with pytest.raises(SystemExit, match="one --label per run"):
        render([entry], ["a", "b"])


def test_render_rejects_mixed_currencies(tmp_path: Path):
    eur_calls = [
        {
            "stage": "transcript", "provider": "fake", "model": "fake-asr",
            "units": [{"kind": "seconds", "value": 10.0}],
            "cost": {"amount": 0.05, "currency": "EUR", "tariff_date": "2026-10-01"},
            "cache_hit": False,
        },
    ]
    entries = [
        load_run(make_run(tmp_path, name="usd")),
        load_run(make_run(tmp_path, name="eur", duration_s=30.0, calls=eur_calls)),
    ]
    with pytest.raises(SystemExit, match="mixed currencies"):
        render(entries, ["x", "y"])

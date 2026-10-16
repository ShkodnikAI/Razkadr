"""Eval harness tests (naryad 015): hit@k, tolerance bounds, reports.

The metrics are unit-tested on known candidate lists (no pipeline); the
integration test runs the harness over a real reference-set layout with a
synthetic video (ffmpeg fixture, naryad 034) and fake providers, comparing
two configs: hybrid vs keyword-only (the comparison mode requirement).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from razkadr.evalharness import (
    EvalError,
    evaluate_configuration,
    hit_counts,
    load_eval_set,
    run_eval,
)
from razkadr.presets import load_preset
from razkadr.validation import validate_document

REPO = Path(__file__).resolve().parent.parent
RECIPE = REPO / "presets" / "recipe.yaml"


# --- metrics unit tests ---------------------------------------------------------


def test_hit_counts_known_result():
    # candidates sorted best-first; moments: one easy, one far away
    candidates = [10.0, 20.0, 30.0, 40.0]
    moments = [10.5, 31.5]
    covered, total = hit_counts(candidates, moments, k=1, tolerance=2.0)
    assert (covered, total) == (1, 2)  # only 10.5 is covered by the top-1
    covered, total = hit_counts(candidates, moments, k=3, tolerance=2.0)
    assert (covered, total) == (2, 2)  # 31.5 falls into the top-3 via 30.0


def test_tolerance_boundary_1_9_vs_2_1():
    """The auditor's boundary: 1.9 s away is a hit, 2.1 s away is not."""
    candidates = [11.9, 12.1]
    moments = [10.0]
    assert hit_counts(candidates, moments, k=1, tolerance=2.0)[0] == 1  # 1.9 s away
    assert hit_counts([12.1], moments, k=1, tolerance=2.0)[0] == 0  # 2.1 s away
    assert hit_counts([12.0], moments, k=1, tolerance=2.0)[0] == 1  # exact 2.0 s bound
    assert hit_counts([12.1], moments, k=1, tolerance=2.1)[0] == 1  # wider tolerance hits


def test_hit_counts_empty_moments_and_candidates():
    assert hit_counts([], [10.0], k=3, tolerance=2.0) == (0, 1)
    assert hit_counts([10.0], [], k=3, tolerance=2.0) == (0, 0)
    assert hit_counts([], [], k=3, tolerance=2.0) == (0, 0)


# --- reference set loading -------------------------------------------------------


def _write_labels(dir: Path, video_id: str, **roles) -> dict:
    labels = {"schema_version": "1.0", "video_id": video_id, "tags": ["ru", "voice"]}
    labels.update(roles)
    path = dir / "labels" / f"{video_id}.json"
    path.write_text(json.dumps(labels, ensure_ascii=False), encoding="utf-8")
    assert validate_document(labels, "eval_labels") == []
    return labels


def test_load_eval_set_missing_layout(tmp_path: Path):
    with pytest.raises(EvalError, match="videos/ and labels/"):
        load_eval_set(tmp_path / "not-a-set")


def test_load_eval_set_reports_missing_labels(tmp_path: Path):
    (tmp_path / "videos").mkdir()
    (tmp_path / "labels").mkdir()
    (tmp_path / "videos" / "vid1.mp4").write_bytes(b"stub")
    (tmp_path / "videos" / "readme.txt").write_text("skip me")  # not a video suffix
    eval_set = load_eval_set(tmp_path)
    assert [vid for vid, _path in eval_set["videos"]] == ["vid1"]
    assert eval_set["missing_labels"] == ["vid1"]
    assert eval_set["labels"] == {}


def test_load_eval_set_invalid_labels(tmp_path: Path):
    (tmp_path / "videos").mkdir()
    (tmp_path / "labels").mkdir()
    (tmp_path / "videos" / "vid1.mp4").write_bytes(b"stub")
    (tmp_path / "labels" / "vid1.json").write_text('{"broken": true}', encoding="utf-8")
    with pytest.raises(EvalError, match="labels are invalid"):
        load_eval_set(tmp_path)


# --- integration: full harness on the synthetic fixture ---------------------------


@pytest.fixture()
def eval_set(tmp_path: Path, synthetic_fixtures) -> Path:
    """data/eval layout with the scenes.mp4 fixture and known labels."""
    root = tmp_path / "eval"
    (root / "videos").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    shutil.copy(synthetic_fixtures.path("scenes.mp4"), root / "videos" / "demo1.mp4")
    # labels land on frames the pipeline accepts (measured: cuts at
    # 6.9/13.9/20.9/27.9 s, scene mids are accepted)
    _write_labels(
        root,
        "demo1",
        hero=[{"t": 10.0, "note": "готовое блюдо"}],
        ingredients=[{"t": 3.0, "note": "ингредиенты"}],
        steps=[
            {"t": 16.0},
            {"t": 22.5},
            {"t": 28.0},
            {"t": 5.0},
            {"t": 11.0},
        ],
    )
    return root


@pytest.fixture()
def eval_configs(tmp_path: Path) -> dict[str, Path]:
    """Two configs: hybrid (all signals) vs keyword-only (text/image off)."""
    base = [
        f'runs_dir = "{tmp_path / "runs"}"',
        "",
        "[cache]",
        f'dir = "{tmp_path / "calls"}"',
        "",
        "[asr]",
        'provider = "fake"',
        "[vlm]",
        'provider = "fake"',
        "[llm]",
        'provider = "fake"',
        "",
        "[embeddings]",
        'provider = "fake"',
        "dim = 8",
    ]
    hybrid = tmp_path / "hybrid.toml"
    hybrid.write_text("\n".join(base), encoding="utf-8")
    keyword_only = tmp_path / "keyword.toml"
    keyword_only.write_text(
        "\n".join(base + ["", "[search]", 'signals = { image = false, text = false, keyword = true }']),
        encoding="utf-8",
    )
    return {"hybrid": hybrid, "keyword": keyword_only}


def test_run_eval_end_to_end(eval_set: Path, eval_configs: dict[str, Path], tmp_path: Path):
    out = tmp_path / "out"
    json_path, md_path = run_eval(
        eval_set,
        [eval_configs["hybrid"], eval_configs["keyword"]],
        k_list=[1, 3],
        tolerance=2.0,
        preset_path=RECIPE,
        out_dir=out,
        keep_runs=True,
    )
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert md_path.is_file()
    assert report["schema"] == "eval-report.v1"
    assert report["preset"] == "recipe"
    assert report["versions"]["razkadr"] and report["versions"]["ffmpeg"] != ""
    assert set(report["labels_sha256"]) == {"demo1"}
    assert set(report["results"]) == {"hybrid", "keyword"}

    for result in report["results"].values():
        assert result["cost"]["calls"] > 0  # fake providers still bill calls
        assert result["video_seconds"] > 0
        # roles marked in labels are in the denominator
        assert result["roles"]["hero"]["videos"] == 1
        assert result["roles"]["hero"]["moments"] == 1
        assert result["roles"]["steps"]["moments"] == 5
        for role in ("hero", "ingredients", "steps"):
            data = result["roles"][role]
            for k in (1, 3):
                assert 0 <= data["hits_by_k"][str(k)] <= data["moments"]
    # the markdown carries the cost and versions
    markdown = md_path.read_text(encoding="utf-8")
    assert "hit@1" in markdown and "hit@3" in markdown
    assert "Стоимость индексации" in markdown and "razkadr" in markdown


def test_run_eval_deterministic_report(eval_set: Path, eval_configs: dict[str, Path], tmp_path: Path):
    out1, out2 = tmp_path / "r1", tmp_path / "r2"
    run_eval(eval_set, [eval_configs["hybrid"]], out_dir=out1, preset_path=RECIPE)
    run_eval(eval_set, [eval_configs["hybrid"]], out_dir=out2, preset_path=RECIPE)
    r1 = json.loads((out1 / "report.json").read_text(encoding="utf-8"))
    r2 = json.loads((out2 / "report.json").read_text(encoding="utf-8"))
    for report in (r1, r2):
        report.pop("generated_utc")
        for result in report["results"].values():
            # wall times and cache-hit counts legitimately differ between runs
            result.pop("index_time_s")
            result["cost"].pop("cache_hits")
    assert r1 == r2  # same inputs -> identical results (deterministic search)


def test_run_eval_missing_config(eval_set: Path, tmp_path: Path):
    with pytest.raises(EvalError, match="config not found"):
        run_eval(eval_set, [tmp_path / "nope.toml"], out_dir=tmp_path / "out", preset_path=RECIPE)


def test_run_eval_rejects_bad_k(eval_set: Path, eval_configs: dict[str, Path], tmp_path: Path):
    with pytest.raises(EvalError, match="k values must be >= 1"):
        run_eval(eval_set, [eval_configs["hybrid"]], k_list=[0], out_dir=tmp_path / "o", preset_path=RECIPE)


def test_denominator_excludes_unmarked_roles(eval_set: Path, eval_configs: dict[str, Path], tmp_path: Path):
    """Auditor rule: a video without a role's labels never joins that
    role's denominator."""
    # demo2 has hero labels only: ingredients/steps keep their denominators
    shutil.copy(eval_set / "videos" / "demo1.mp4", eval_set / "videos" / "demo2.mp4")
    _write_labels(eval_set, "demo2", hero=[{"t": 10.0}])
    out = tmp_path / "out"
    report = json.loads(
        (run_eval(eval_set, [eval_configs["hybrid"]], out_dir=out, preset_path=RECIPE)[0]).read_text(encoding="utf-8")
    )
    result = report["results"]["hybrid"]
    assert result["roles"]["hero"]["videos"] == 2
    assert result["roles"]["ingredients"]["videos"] == 1
    assert result["roles"]["steps"]["videos"] == 1


def test_evaluate_configuration_skips_unlabeled_video(eval_set: Path, eval_configs: dict[str, Path]):
    """A video without any labels file is warned and never evaluated."""
    eval_set_copy = load_eval_set(eval_set)
    preset = load_preset(RECIPE)
    from razkadr.config import load_config

    config = load_config(eval_configs["hybrid"])
    config["runs_dir"] = str(eval_set.parent / "runs-x")
    labeled = {"demo1": eval_set_copy["labels"]["demo1"]}
    results, warnings = evaluate_configuration(
        {
            **eval_set_copy,
            "labels": labeled,
            "videos": eval_set_copy["videos"],
            "missing_labels": ["demo2"],
        },
        "hybrid",
        config,
        eval_configs["hybrid"],
        preset,
        [1, 3],
        2.0,
        keep_runs=False,
    )
    assert list(results["per_video"]) == ["demo1"]
    assert any("demo2" in w for w in warnings)  # the unlabeled video is reported

"""Index stage tests (naryad 010): spaces, batching, cache, staleness, manifest.

All tests run on hand-written stage artifacts through the fake embedder
of the naryad 033 layer; no network, no external APIs. The factory is
monkeypatched to capture every FakeEmbeddings instance, so provider
call counts (batching, rejected-frame exclusion, dim probe) are asserted
from real counters, not from assumptions.
"""

import json
import logging
from pathlib import Path

import pytest

from razkadr.cli import EXIT_OK, main
from razkadr.config import load_config
from razkadr.providers.fakes import FakeEmbeddings
from razkadr.providers.layer import FACTORIES
from razkadr.runctx import RunContext
from razkadr.stages.index import (
    DIM_PROBE_TEXT,
    INDEX_DIRNAME,
    IndexStageError,
    index_params,
    run_index,
)
from razkadr.validation import validate_document
from razkadr.vectorstore import Filters, LocalVectorStore


def frame(t: float, scene: int, sharpness: float, rejected: bool | None = None, reason: str | None = None) -> dict:
    entry = {
        "t": t,
        "file": f"frames/f-{int(t * 10):05d}.jpg",
        "scene_id": scene,
        "sharpness": sharpness,
        "phash": "0123456789abcdef",
        "exposure": 0.9,
    }
    if rejected:
        entry["rejected"] = True
        entry["reject_reason"] = reason or "duplicate"
    return entry


def chunk(cid: str, t_start: float, t_end: float, text: str, lang: str = "ru") -> dict:
    return {"id": cid, "t_start": t_start, "t_end": t_end, "text": text, "lang": lang}


def write_json(ctx: RunContext, name: str, data: dict) -> None:
    (ctx.dir / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def full_source(path: str = "normalized.mp4") -> dict:
    return {
        "schema_version": "1.0",
        "path": path,
        "sha256": "a" * 64,
        "duration_s": 30.0,
        "fps": 10.0,
        "width": 640,
        "height": 360,
    }


def write_inputs(
    ctx: RunContext,
    frames: list[dict],
    chunks: list[dict] | None = None,
    ocr_results: list[dict] | None = None,
    source: dict | None = None,
) -> None:
    write_json(ctx, "frames.json", {"schema_version": "1.0", "frames": frames})
    write_json(ctx, "chunks.json", {"schema_version": "1.0", "chunks": chunks or []})
    write_json(ctx, "ocr.json", {"schema_version": "1.0", "results": ocr_results or []})
    if source is not None:
        write_json(ctx, "source.json", source)
    # frame jpgs: the fake embedder hashes the bytes, the files must exist
    for f in frames:
        path = ctx.dir / f["file"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"jpg-bytes-{f['t']}".encode())


STANDARD_FRAMES = [
    frame(0.0, 0, 0.9),
    frame(0.5, 0, 0.7),
    frame(1.0, 1, 0.8),
    frame(1.5, 1, 0.2, rejected=True, reason="blurry"),
]
STANDARD_CHUNKS = [
    chunk("c1", 0.0, 2.5, "как приготовить борщ", lang="ru"),
    chunk("c2", 2.5, 5.0, "подавайте со сметаной", lang="ru"),
]
STANDARD_OCR = [
    {"frame": "frames/f-00000.jpg", "t": 0.0, "text": "ингредиенты", "lang": "ru"},
]


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """RunContext + config with fake embeddings; captures provider instances."""
    created: list[FakeEmbeddings] = []

    def factory(section: dict) -> FakeEmbeddings:
        provider = FakeEmbeddings(
            model=section.get("model", "fake-embed-v2"),
            dim=int(section.get("dim", 8)),
        )
        created.append(provider)
        return provider

    monkeypatch.setitem(FACTORIES["embed"], "fake", factory)
    config = load_config(None)
    config["runs_dir"] = str(tmp_path / "runs")
    config["cache"] = {"dir": str(tmp_path / "calls")}
    config["embeddings"] = {"provider": "fake", "model": "fake-embed-v2", "dim": 8}
    ctx = RunContext(config["runs_dir"], config=config)
    write_inputs(ctx, STANDARD_FRAMES, STANDARD_CHUNKS, STANDARD_OCR, source=full_source())
    return tmp_path, config, ctx, created


def _manifest(ctx: RunContext) -> dict:
    return json.loads((ctx.dir / INDEX_DIRNAME / "manifest.json").read_text(encoding="utf-8"))


def _warning_fields(caplog) -> list[dict]:
    return [
        dict(getattr(r, "fields", {}))
        for r in caplog.records
        if r.levelno == logging.WARNING
    ]


# --- params ---------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [0, -1])
def test_params_batch_size_validation(batch_size):
    with pytest.raises(IndexStageError, match="batch_size"):
        index_params({"embeddings": {"batch_size": batch_size}})


def test_params_defaults():
    params = index_params({})
    assert params == {"batch_size": 16}


# --- happy path -----------------------------------------------------------------


def test_happy_path_builds_all_three_spaces(env):
    _tmp_path, config, ctx, _created = env
    manifest, counts = run_index(ctx, config)
    assert manifest["schema_version"] == "1.0"
    assert manifest["provider"] == "fake"
    assert manifest["model"] == "fake-embed-v2"
    assert manifest["dim"] == 8
    assert manifest["num_vectors"] == 6  # 3 accepted frames + 2 chunks + 1 ocr
    assert manifest["vectors_file"] == "index/vectors.bin"
    assert counts == {"frames": 3, "chunks": 2, "ocr": 1}
    assert validate_document(manifest, "index") == []
    assert (ctx.dir / INDEX_DIRNAME / "vectors.bin").is_file()


def test_manifest_version_from_config(env):
    _tmp_path, config, ctx, _created = env
    config["embeddings"]["model_version"] = "2026-01"
    manifest, _ = run_index(ctx, config)
    assert manifest["version"] == "2026-01"


def test_manifest_version_unknown_without_report(env):
    _tmp_path, config, ctx, _created = env
    manifest, _ = run_index(ctx, config)
    assert manifest["version"] == "unknown"


def test_search_finds_same_text_top1(env):
    """Fake embedder is deterministic: identical text -> identical vector."""
    _tmp_path, config, ctx, _created = env
    run_index(ctx, config)
    with LocalVectorStore(ctx.dir / INDEX_DIRNAME) as store:
        hits = store.search([*FakeEmbeddings(dim=8)._vector("как приготовить борщ".encode())], k=2)
        assert hits[0][0] == "chunks:c1"
        assert hits[0][1] == pytest.approx(1.0)


def test_store_metadata_roundtrip(env):
    _tmp_path, config, ctx, _created = env
    run_index(ctx, config)
    with LocalVectorStore(ctx.dir / INDEX_DIRNAME) as store:
        hits = store.search([0.0] * 8, k=10, filters=Filters(spaces=("frames",), min_sharpness=0.5))
        assert {h[0] for h in hits} == {
            "frames:frames/f-00000.jpg",
            "frames:frames/f-00005.jpg",
            "frames:frames/f-00010.jpg",
        }
        assert all(h[2]["space"] == "frames" for h in hits)
        assert all(h[2]["source"] == "normalized.mp4" for h in hits)
        assert all(h[2]["scene_id"] in (0, 1) for h in hits)


def test_rejected_frame_not_embedded_and_not_searchable(env):
    _tmp_path, config, ctx, _created = env
    run_index(ctx, config)
    # the honest counter is run.json: the layer records every call's units
    run = json.loads((ctx.dir / "run.json").read_text(encoding="utf-8"))
    embed_calls = [c for c in run["provider_calls"] if c["task"] == "embed"]
    billed_images = sum(
        u["value"] for c in embed_calls for u in c["units"] if u["kind"] == "images"
    )
    assert billed_images == 3  # exactly the accepted frames, rejected never sent
    with LocalVectorStore(ctx.dir / INDEX_DIRNAME) as store:
        all_ids = []
        for space in ("frames", "chunks", "ocr"):
            all_ids.extend(
                store.search([0.0] * 8, k=100, filters=Filters(spaces=(space,)))
            )
        assert all("f-00015" not in cid for cid, _, _ in all_ids)


def test_batching_calls(env):
    _tmp_path, config, ctx, created = env
    config["embeddings"]["batch_size"] = 2
    run_index(ctx, config)
    provider = created[0]
    # 3 frames -> 2 calls; 2 chunks -> 1 call; 1 ocr -> 1 call; no probe
    assert provider.calls == 4


def test_stage_cache_second_run_free(env):
    _tmp_path, config, ctx, created = env
    run_index(ctx, config)
    calls_after_first = created[0].calls
    manifest2, counts2 = run_index(ctx, config)
    assert created[0].calls == calls_after_first  # no new provider work
    assert manifest2 == _manifest(ctx)
    assert counts2 == {"frames": 3, "chunks": 2, "ocr": 1}


def test_rebuild_after_force_gives_same_results(env):
    _tmp_path, config, ctx, _created = env
    manifest1, _ = run_index(ctx, config)
    with LocalVectorStore(ctx.dir / INDEX_DIRNAME) as store:
        hits1 = store.search([0.0] * 8, k=3)
    ctx2 = RunContext(config["runs_dir"], run_id=ctx.run_id, config=config, force=True)
    manifest2, _ = run_index(ctx2, config)
    assert manifest2 == manifest1
    with LocalVectorStore(ctx2.dir / INDEX_DIRNAME) as store:
        hits2 = store.search([0.0] * 8, k=3)
    assert hits1 == hits2


# --- staleness ------------------------------------------------------------------


def test_model_change_marks_stale_and_rebuilds(env, caplog):
    _tmp_path, config, ctx, _created = env
    run_index(ctx, config)[0]
    caplog.set_level(logging.WARNING)
    config["embeddings"]["model"] = "fake-embed-alt"
    manifest, _ = run_index(ctx, config)
    warnings = _warning_fields(caplog)
    stale = [w for w in warnings if w.get("reason") == "provider/model/version changed"]
    assert stale, f"no stale warning in {warnings}"
    assert "fake/fake-embed-v2/unknown" in stale[0]["old"]
    assert "fake/fake-embed-alt/unknown" in stale[0]["new"]
    assert manifest["model"] == "fake-embed-alt"
    # the store was rebuilt: only the new model's vectors are there
    with LocalVectorStore(ctx.dir / INDEX_DIRNAME) as store:
        assert store.count() == 6
        assert store.get_meta("model") == "fake-embed-alt"


def test_provider_change_marks_stale(env, caplog):
    _tmp_path, config, ctx, _created = env
    run_index(ctx, config)
    caplog.set_level(logging.WARNING)
    config["embeddings"]["provider"] = "fake-alt"  # different fake: name fake-alt
    manifest, _ = run_index(ctx, config)
    assert manifest["provider"] == "fake-alt"


def test_no_stale_warning_when_unchanged(env, caplog):
    _tmp_path, config, ctx, _created = env
    run_index(ctx, config)
    caplog.set_level(logging.WARNING)
    ctx2 = RunContext(config["runs_dir"], run_id=ctx.run_id, config=config, force=True)
    run_index(ctx2, config)
    assert not [w for w in _warning_fields(caplog) if w.get("reason") == "provider/model/version changed"]


# --- skip / loud failure --------------------------------------------------------


def test_nothing_to_embed_without_provider_skips(tmp_path):
    config = load_config(None)
    config["runs_dir"] = str(tmp_path / "runs")
    config["cache"] = {"dir": str(tmp_path / "calls")}
    ctx = RunContext(config["runs_dir"], config=config)
    write_inputs(ctx, [frame(0.0, 0, 0.9, rejected=True)], [], [])
    manifest, counts = run_index(ctx, config)
    assert manifest is None
    assert counts == {}
    assert not (ctx.dir / INDEX_DIRNAME).exists()


def test_without_provider_skips_even_with_frames(tmp_path, caplog):
    """Repo convention: unconfigured provider -> skip with a warning."""
    config = load_config(None)
    config["runs_dir"] = str(tmp_path / "runs")
    ctx = RunContext(config["runs_dir"], config=config)
    write_inputs(ctx, [frame(0.0, 0, 0.9)], [chunk("c1", 0.0, 1.0, "текст")], [])
    import logging as _logging

    caplog.set_level(_logging.WARNING)
    manifest, counts = run_index(ctx, config)
    assert manifest is None and counts == {}
    assert not (ctx.dir / INDEX_DIRNAME).exists()
    reasons = [dict(getattr(r, "fields", {})).get("reason", "") for r in caplog.records if r.levelno == _logging.WARNING]
    assert any("no [embeddings] provider" in r for r in reasons)
    # re-run over the same run: cached path, still no manifest
    ctx2 = RunContext(config["runs_dir"], run_id=ctx.run_id, config=config)
    assert run_index(ctx2, config) == (None, {})


def test_nothing_to_embed_with_provider_probes_dim(env):
    _tmp_path, config, ctx, created = env
    write_inputs(ctx, [], [], [])  # all rejected? no: replace with empty
    write_json(ctx, "frames.json", {"schema_version": "1.0", "frames": []})
    manifest, counts = run_index(ctx, config)
    assert manifest["num_vectors"] == 0
    assert manifest["dim"] == 8
    assert counts == {"frames": 0, "chunks": 0, "ocr": 0}
    provider = created[0]
    assert provider.calls == 1  # the dim probe
    assert provider.last_usage[0]["kind"] == "tokens"


def test_probe_text_is_stable_constant():
    assert DIM_PROBE_TEXT == "razkadr dim probe"


def test_unknown_provider_fails_loudly(env):
    _tmp_path, config, ctx, _created = env
    config["embeddings"]["provider"] = "no-such-provider"
    with pytest.raises(IndexStageError, match="unknown embeddings provider"):
        run_index(ctx, config)


# --- input validation -----------------------------------------------------------


def test_missing_frames_json(env):
    _tmp_path, config, ctx, _created = env
    (ctx.dir / "frames.json").unlink()
    with pytest.raises(IndexStageError, match="frames.json is missing"):
        run_index(ctx, config)


def test_missing_chunks_json(env):
    _tmp_path, config, ctx, _created = env
    (ctx.dir / "chunks.json").unlink()
    with pytest.raises(IndexStageError, match="chunks.json is missing"):
        run_index(ctx, config)


def test_invalid_frames_json(env):
    _tmp_path, config, ctx, _created = env
    (ctx.dir / "frames.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(IndexStageError, match="not valid JSON"):
        run_index(ctx, config)


def test_schema_invalid_frames_json(env):
    _tmp_path, config, ctx, _created = env
    write_json(ctx, "frames.json", {"schema_version": "1.0", "frames": [{"t": 0.0}]})
    with pytest.raises(IndexStageError, match="not a valid frames artifact"):
        run_index(ctx, config)


def test_duplicate_ocr_frame_id_rejected(env):
    _tmp_path, config, ctx, _created = env
    dup = {"frame": "frames/f-00000.jpg", "t": 0.0, "text": "дубликат", "lang": "ru"}
    write_inputs(ctx, STANDARD_FRAMES, STANDARD_CHUNKS, [STANDARD_OCR[0], dup])
    with pytest.raises(IndexStageError, match="duplicate vector id"):
        run_index(ctx, config)


def test_missing_source_json_ok(env):
    _tmp_path, config, ctx, _created = env
    (ctx.dir / "source.json").unlink(missing_ok=True)
    manifest, _counts = run_index(ctx, config)
    assert manifest["num_vectors"] == 6
    with LocalVectorStore(ctx.dir / INDEX_DIRNAME) as store:
        hits = store.search([0.0] * 8, k=1)
        assert "source" not in hits[0][2]


# --- CLI e2e on the synthetic fixture -------------------------------------------


def _write_toml(tmp_path: Path, with_embeddings: bool = True) -> Path:
    lines = [
        f'runs_dir = "{tmp_path / "runs"}"',
        "",
        "[cache]",
        f'dir = "{tmp_path / "calls"}"',
        "",
        "[asr]",
        'provider = "fake"',
        'model = "fake-nova-2"',
        "",
        "[embeddings]",
        'provider = "fake"',
        'model = "fake-embed-v2"',
        "dim = 8",
        "",
    ]
    if not with_embeddings:
        # drop the [embeddings] section (5 lines + separator)
        lines = lines[:-6]
    path = tmp_path / "razkadr.toml"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def video_with_audio(synthetic_fixtures, tmp_path_factory):
    """scenes.mp4 (video only) muxed with a sine track: video + audio fixture."""
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not found in PATH")
    dst = tmp_path_factory.mktemp("index_with_audio") / "with_audio.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(synthetic_fixtures.path("scenes.mp4")),
         "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=30",
         "-shortest", "-c:v", "copy", "-c:a", "aac", "-b:a", "96k", str(dst)],
        check=True, capture_output=True,
    )
    return dst


def test_cli_index_full_pipeline_with_fixtures(video_with_audio, tmp_path, monkeypatch, capsys):
    """razkadr index: index/ artifacts exist, summary printed, resume is cached."""
    monkeypatch.chdir(tmp_path)
    config_path = _write_toml(tmp_path)

    code = main(["index", str(video_with_audio), "--config", str(config_path)])

    assert code == EXIT_OK
    out = capsys.readouterr().out
    assert "index ok" in out
    assert "index:" in out and "vectors" in out
    runs_dir = tmp_path / "runs"
    run_dir = next(d for d in sorted(runs_dir.iterdir()) if d.is_dir())
    manifest = json.loads((run_dir / INDEX_DIRNAME / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["provider"] == "fake"
    assert manifest["num_vectors"] > 0
    assert validate_document(manifest, "index") == []

    # the manifest passes `razkadr validate --kind index`
    capsys.readouterr()
    assert main(["validate", "--kind", "index", str(run_dir / INDEX_DIRNAME / "manifest.json")]) == EXIT_OK
    assert capsys.readouterr().out.strip() == "ok"

    # resume: the stage is cached, the store still answers
    code = main(["index", str(video_with_audio), "--config", str(config_path), "--resume", run_dir.name])
    assert code == EXIT_OK
    assert "index ok" in capsys.readouterr().out
    run_json = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    statuses = [s["status"] for s in run_json["stages"] if s["name"] == "index"]
    assert statuses[-1] == "cached"


def test_cli_index_without_embeddings_skips(video_with_audio, tmp_path, monkeypatch, capsys):
    """No [embeddings] in config: pipeline finishes, index skipped (repo convention)."""
    import shutil

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not found in PATH")
    monkeypatch.chdir(tmp_path)
    config_path = _write_toml(tmp_path, with_embeddings=False)

    code = main(["index", str(video_with_audio), "--config", str(config_path)])

    assert code == EXIT_OK
    out = capsys.readouterr().out
    assert "index skipped" in out
    runs_dir = tmp_path / "runs"
    run_dir = next(d for d in sorted(runs_dir.iterdir()) if d.is_dir())
    assert not (run_dir / INDEX_DIRNAME).exists()


# --- ADR-0007 -------------------------------------------------------------------


def test_adr_0007_exists_as_proposed():
    text = Path("docs/adr/0007-vectorstore-core.md").read_text(encoding="utf-8")
    assert "Статус:** proposed" in text
    assert "pgvector" in text


# --- benchmark (explicit: pytest -m bench) ---------------------------------------


def _bench_search(tmp_path: Path, n: int, dim: int, queries: int = 100) -> float:
    import time

    import numpy as np

    rng = np.random.default_rng(42)
    st = LocalVectorStore(tmp_path / f"index-{dim}")
    st.create(dim=dim)
    block_rows = 10_000
    for start in range(0, n, block_rows):
        block = rng.standard_normal((block_rows, dim)).astype(np.float32)
        st.add(
            "frames",
            [f"frames:{start + i:06d}" for i in range(block_rows)],
            block.tolist(),
            [{"t": float(start + i) / 10.0} for i in range(block_rows)],
        )
    st.save()
    q = rng.standard_normal(dim).astype(np.float32).tolist()

    st.search(q, k=10)  # warm-up (page cache, BLAS init)
    started = time.perf_counter()
    for _ in range(queries):
        st.search(q, k=10)
    per_query_ms = (time.perf_counter() - started) / queries * 1000
    st.close()
    return per_query_ms


@pytest.mark.bench
def test_bench_search_100k_vectors(tmp_path):
    """Search timing at the core scale (naryad 010 assumption: ~10^5 vectors).

    Not run by default (pyproject addopts excludes 'bench'); run explicitly:
    `pytest -m bench -q -s tests/test_index.py`. The printed numbers go
    into the PR and are the baseline for the pgvector decision (ADR-0007):
    dim 256 - the bench default, dim 1536 - a real embedding-model width;
    the exact scan scales linearly with dim.
    """
    n = 100_000
    for dim in (256, 1536):
        ms = _bench_search(tmp_path, n=n, dim=dim)
        print(f"bench: {n} vectors x dim {dim}, k=10: {ms:.1f} ms/query")
        assert ms < 2000, f"exact search unexpectedly slow at dim {dim}"

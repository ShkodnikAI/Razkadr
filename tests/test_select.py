"""Episode selection tests (naryad 014): preset + hybrid search -> Storyboard.

The toy run mirrors tests/test_search.py (hand-filled vector store,
scripted embedder); a synthetic preset with hero/ingredients/steps roles
exercises quotas, order, diversity and user-edit preservation without
touching the shipped recipe preset. No network, no real providers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from razkadr.stages.select import SelectError, run_select, select_params
from razkadr.validation import validate_document

# the scripted embedder machinery is shared with the search tests
from test_search import (
    FRAME_VECTORS,
    OCR_TEXT,
    VIDEO_LANG,
    ScriptedEmbedder,
)

# the toy run has 8 frames; the search tests define vectors for 5
FRAME_VECTORS = {
    **FRAME_VECTORS,
    "frames/000005.jpg": [-0.1, 0.9],
    "frames/000006.jpg": [0.35, 0.6],
    "frames/000007.jpg": [-0.5, 0.75],
}


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    run = tmp_path / "runs" / "toy"
    run.mkdir(parents=True)
    frames = []
    for i in range(8):
        t = 1.0 + i * 4.0  # 1, 5, 9, 13, 17, 21, 25, 29 s
        frames.append(
            {
                "t": t,
                "file": f"frames/{i:06d}.jpg",
                "scene_id": i // 2,  # 4 scenes, 2 frames each
                "sharpness": 0.9 - 0.05 * i,
                "phash": f"{0x1111111111111111 * (i + 1) % (1 << 64):016x}",  # pairwise >= 8 bits apart
                "exposure": 0.9,
            }
        )
    (run / "frames.json").write_text(
        json.dumps({"schema_version": "1.1", "frames": frames}), encoding="utf-8"
    )
    chunks = [
        {"t_start": 0.0, "t_end": 10.0, "text": "готовим тесто для блинов", "lang": VIDEO_LANG},
        {"t_start": 11.0, "t_end": 20.0, "text": "жарим блины на сковороде", "lang": VIDEO_LANG},
        {"t_start": 21.0, "t_end": 30.0, "text": "готовое блюдо: блины со сметаной", "lang": VIDEO_LANG},
    ]
    (run / "chunks.json").write_text(
        json.dumps({"schema_version": "1.1", "lang": VIDEO_LANG, "chunks": chunks}), encoding="utf-8"
    )
    (run / "ocr.json").write_text(
        json.dumps({"schema_version": "1.1", "results": [{"t": 25.0, "text": OCR_TEXT}]}),
        encoding="utf-8",
    )
    (run / "transcript.json").write_text(
        json.dumps({"schema_version": "1.1", "lang": VIDEO_LANG, "segments": []}), encoding="utf-8"
    )
    (run / "source.json").write_text(
        json.dumps({"schema_version": "1.1", "path": "normalized.mp4", "duration_s": 30.0}),
        encoding="utf-8",
    )
    from razkadr.vectorstore import LocalVectorStore

    store = LocalVectorStore(run / "index")
    store.create(dim=2)
    # frames: hero frame (000001) is closest to the query vector
    accepted = [f for f in frames]
    store.add(
        "frames",
        [f"frame-{f['t']}" for f in accepted],
        [FRAME_VECTORS[f["file"]] for f in accepted],
        [
            {"file": f["file"], "t": f["t"], "scene_id": f["scene_id"], "sharpness": f["sharpness"]}
            for f in accepted
        ],
    )
    store.add(
        "chunks",
        ["chunk-0", "chunk-1", "chunk-2"],
        [[0.0, 1.0], [0.8, 0.6], [0.9, 0.44]],
        [
            {"t": c["t_start"], "t_end": c["t_end"], "text": c["text"], "lang": VIDEO_LANG}
            for c in chunks
        ],
    )
    store.add("ocr", ["ocr-0"], [[0.95, 0.31]], [{"t": 25.0, "t_end": 25.0, "text": OCR_TEXT, "lang": VIDEO_LANG}])
    store.save()
    store.close()
    return run


@pytest.fixture()
def preset_path(tmp_path: Path) -> Path:
    preset = {
        "schema_version": "1.0",
        "name": "test",
        "min_step_gap_s": 6.0,
        "order": ["hero", "ingredients", "steps"],
        "roles": {
            "hero": {
                "min": 1,
                "max": 1,
                "candidates": 10,
                "alternatives": 2,
                "queries": {"ru": ["готовое блюдо"]},
                "weights": {"embedding": 0.6, "keyword": 0.2, "sharpness": 0.1, "exposure": 0.1},
                "constraints": {"min_sharpness": 0.3},
            },
            "ingredients": {
                "min": 1,
                "max": 1,
                "candidates": 10,
                "alternatives": 2,
                "queries": {"ru": ["тесто"]},
                "weights": {"embedding": 0.5, "keyword": 0.3, "sharpness": 0.1, "exposure": 0.1},
            },
            "steps": {
                "min": 2,
                "max": 4,
                "candidates": 10,
                "alternatives": 1,
                "queries": {"ru": ["блины", "сковорода"]},
                "weights": {"embedding": 0.5, "keyword": 0.2, "sharpness": 0.2, "exposure": 0.1},
            },
        },
    }
    path = tmp_path / "preset.yaml"
    path.write_text(yaml.safe_dump(preset, allow_unicode=True), encoding="utf-8")
    return path


def _select(run_dir: Path, preset_path: Path, **kwargs):
    return run_select(run_dir, preset_path, config=kwargs.pop("config", {}), **kwargs)


# --- validity ---------------------------------------------------------------------


def test_valid_storyboard_with_roles_and_bounds(run_dir: Path, preset_path: Path):
    storyboard, report = _select(run_dir, preset_path, embeddings=ScriptedEmbedder())
    assert validate_document(storyboard, "storyboard") == []
    roles = {ep["role"] for ep in storyboard["episodes"]}
    assert roles == {"hero", "ingredients", "steps"}
    duration = 30.0
    for ep in storyboard["episodes"]:
        assert 0.0 <= ep["t_start"] <= ep["frame"]["t"] <= ep["t_end"] <= duration
        assert ep["t_end"] > ep["t_start"]
        assert ep["source_id"] == "src"
    assert report["preset"] == "test"
    assert report["queries"]  # every role searched something


def test_hero_quota_and_order(run_dir: Path, preset_path: Path):
    storyboard, _report = _select(run_dir, preset_path, embeddings=ScriptedEmbedder())
    heroes = [ep for ep in storyboard["episodes"] if ep["role"] == "hero"]
    assert len(heroes) == 1
    order = [ep["role"] for ep in storyboard["episodes"]]
    # roles follow the preset order, steps are time-ordered within the role
    first_of = []
    for role in order:
        if role not in first_of:
            first_of.append(role)
    assert first_of == ["hero", "ingredients", "steps"]
    steps_t = [ep["frame"]["t"] for ep in storyboard["episodes"] if ep["role"] == "steps"]
    assert steps_t == sorted(steps_t)


def test_ids_are_unique(run_dir: Path, preset_path: Path):
    storyboard, _report = _select(run_dir, preset_path, embeddings=ScriptedEmbedder())
    ids = [ep["id"] for ep in storyboard["episodes"]]
    assert len(ids) == len(set(ids))


# --- alternatives ------------------------------------------------------------------


def test_alternatives_differ_from_selection_and_each_other(run_dir: Path, preset_path: Path):
    storyboard, _report = _select(run_dir, preset_path, embeddings=ScriptedEmbedder())
    for ep in storyboard["episodes"]:
        alt_paths = [alt["path"] for alt in ep["alternatives"]]
        assert ep["frame"]["path"] not in alt_paths, "an alternative must not be the chosen frame"
        assert len(alt_paths) == len(set(alt_paths)), "alternatives must be pairwise distinct"
        assert len(alt_paths) <= 2  # preset alternatives: 2 (hero/ingredients), 1 (steps)


# --- diversity ----------------------------------------------------------------------


def test_diversity_rejects_near_duplicate_shots(tmp_path: Path, run_dir: Path, preset_path: Path):
    """Two near-identical frames (pHash distance 2, adjacent in time) must
    not both become steps (the auditors' duplicate-shots case)."""
    frames = json.loads((run_dir / "frames.json").read_text(encoding="utf-8"))["frames"]
    # clone frame 1 as a near-duplicate: pHash distance 2, close in time
    clone = dict(frames[1])
    clone["t"] = frames[1]["t"] + 1.0
    clone["file"] = "frames/000100.jpg"
    clone["phash"] = f"{int(frames[1]['phash'], 16) ^ 0b11:016x}"  # distance 2 bits
    frames.append(clone)
    (run_dir / "frames.json").write_text(
        json.dumps({"schema_version": "1.1", "frames": frames}), encoding="utf-8"
    )
    from razkadr.vectorstore import LocalVectorStore

    store = LocalVectorStore(run_dir / "index")
    store.open()
    store.add(
        "frames",
        ["frame-clone"],
        [FRAME_VECTORS[frames[1]["file"]]],  # the same vector: same search rank
        [{"file": clone["file"], "t": clone["t"], "scene_id": clone["scene_id"], "sharpness": clone["sharpness"]}],
    )
    store.save()
    store.close()

    storyboard, _report = _select(run_dir, preset_path, embeddings=ScriptedEmbedder())
    chosen = {ep["frame"]["path"] for ep in storyboard["episodes"] if ep["role"] == "steps"}
    both = {frames[1]["file"], clone["file"]}
    assert not (both <= chosen), "near-duplicate shots both selected as steps"


# --- user edits -----------------------------------------------------------------------


def test_user_edited_episodes_survive(run_dir: Path, preset_path: Path):
    # first selection
    storyboard, _ = _select(run_dir, preset_path, embeddings=ScriptedEmbedder())
    hero = next(ep for ep in storyboard["episodes"] if ep["role"] == "hero")
    # the user swaps the hero frame and marks the episode
    other = next(
        f["file"]
        for f in json.loads((run_dir / "frames.json").read_text(encoding="utf-8"))["frames"]
        if f["file"] != hero["frame"]["path"]
    )
    hero["frame"] = {"path": other, "t": 9.0, "score": 1.0, "score_parts": {}}
    hero["user_edited"] = True
    (run_dir / "storyboard.json").write_text(
        json.dumps(storyboard, ensure_ascii=False), encoding="utf-8"
    )
    # re-selection keeps the user's hero and does not reuse its frame
    second, report = _select(run_dir, preset_path, embeddings=ScriptedEmbedder())
    hero2 = [ep for ep in second["episodes"] if ep["role"] == "hero"]
    assert len(hero2) == 1
    assert hero2[0]["user_edited"] is True
    assert hero2[0]["frame"]["path"] == other
    # the kept role is reported as fully user-edited (nothing re-selected)
    assert any("user-edited" in w and "hero" in w for w in report["warnings"])
    # no other episode sits on the user's frame
    for ep in second["episodes"]:
        if ep["id"] != hero2[0]["id"]:
            assert ep["frame"]["path"] != other


# --- quotas, score parts, short video ---------------------------------------------------


def test_score_parts_normalized_and_sum_to_score(run_dir: Path, preset_path: Path):
    storyboard, _ = _select(run_dir, preset_path, embeddings=ScriptedEmbedder())
    for ep in storyboard["episodes"]:
        parts = ep["frame"]["score_parts"]
        assert parts, "score_parts must be recorded"
        for value in parts.values():
            assert 0.0 <= value <= 1.0
        assert ep["frame"]["score"] == pytest.approx(sum(parts.values()), abs=1e-5)
        assert 0.0 <= ep["frame"]["score"] <= 1.0


def test_short_video_fewer_frames_than_roles(tmp_path: Path, preset_path: Path):
    """Auditor case: roles exceed frames -- fewer episodes, warnings, still valid."""
    run = tmp_path / "runs" / "tiny"
    run.mkdir(parents=True)
    frames = [{"t": 2.0, "file": "frames/000000.jpg", "scene_id": 0, "sharpness": 0.8,
               "phash": f"{0x5555_5555_0000:016x}", "exposure": 0.9}]
    (run / "frames.json").write_text(json.dumps({"schema_version": "1.1", "frames": frames}), encoding="utf-8")
    (run / "chunks.json").write_text(
        json.dumps({"schema_version": "1.1", "lang": VIDEO_LANG, "chunks": []}), encoding="utf-8"
    )
    (run / "ocr.json").write_text(json.dumps({"schema_version": "1.1", "results": []}), encoding="utf-8")
    (run / "transcript.json").write_text(
        json.dumps({"schema_version": "1.1", "lang": VIDEO_LANG, "segments": []}), encoding="utf-8"
    )
    (run / "source.json").write_text(
        json.dumps({"schema_version": "1.1", "path": "normalized.mp4", "duration_s": 4.0}), encoding="utf-8"
    )
    from razkadr.vectorstore import LocalVectorStore

    store = LocalVectorStore(run / "index")
    store.create(dim=2)
    store.add("frames", ["frame-2"], [[0.9, 0.1]], [{"file": "frames/000000.jpg", "t": 2.0, "scene_id": 0, "sharpness": 0.8}])
    store.save()
    store.close()

    storyboard, report = _select(run, preset_path, embeddings=ScriptedEmbedder())
    assert validate_document(storyboard, "storyboard") == []
    assert len(storyboard["episodes"]) <= 1  # one frame cannot fill three roles
    assert any("minimum" in w for w in report["warnings"])  # steps min=2 not met


def test_min_sharpness_constraint_from_preset(tmp_path: Path, preset_path: Path):
    run = tmp_path / "runs" / "blurry"
    run.mkdir(parents=True)
    frames = [
        {"t": 2.0, "file": "frames/000000.jpg", "scene_id": 0, "sharpness": 0.1,
         "phash": f"{0x5555_5555_0000:016x}", "exposure": 0.9},
        {"t": 10.0, "file": "frames/000001.jpg", "scene_id": 1, "sharpness": 0.8,
         "phash": f"{0x5555_5555_00FF:016x}", "exposure": 0.9},
    ]
    (run / "frames.json").write_text(json.dumps({"schema_version": "1.1", "frames": frames}), encoding="utf-8")
    (run / "chunks.json").write_text(
        json.dumps({"schema_version": "1.1", "lang": VIDEO_LANG, "chunks": []}), encoding="utf-8"
    )
    (run / "ocr.json").write_text(json.dumps({"schema_version": "1.1", "results": []}), encoding="utf-8")
    (run / "transcript.json").write_text(
        json.dumps({"schema_version": "1.1", "lang": VIDEO_LANG, "segments": []}), encoding="utf-8"
    )
    (run / "source.json").write_text(
        json.dumps({"schema_version": "1.1", "path": "normalized.mp4", "duration_s": 14.0}), encoding="utf-8"
    )
    from razkadr.vectorstore import LocalVectorStore

    store = LocalVectorStore(run / "index")
    store.create(dim=2)
    store.add(
        "frames",
        ["frame-2", "frame-10"],
        [[0.9, 0.1], [0.5, 0.5]],
        [
            {"file": "frames/000000.jpg", "t": 2.0, "scene_id": 0, "sharpness": 0.1},
            {"file": "frames/000001.jpg", "t": 10.0, "scene_id": 1, "sharpness": 0.8},
        ],
    )
    store.save()
    store.close()
    storyboard, _report = _select(run, preset_path, embeddings=ScriptedEmbedder())
    hero = next(ep for ep in storyboard["episodes"] if ep["role"] == "hero")
    # hero has min_sharpness 0.3: the blurry frame (0.1) is not eligible
    assert hero["frame"]["path"] == "frames/000001.jpg"


def test_report_records_reasons(run_dir: Path, preset_path: Path):
    storyboard, report = _select(run_dir, preset_path, embeddings=ScriptedEmbedder())
    hero_files = {ep["frame"]["path"] for ep in storyboard["episodes"] if ep["role"] == "hero"}
    hero_report = [r for r in report["roles"]["hero"] if r["file"] in hero_files]
    assert hero_report
    for record in hero_report:
        assert record["query"] and record["query_lang"]
        assert set(record["rank_parts"]) == {"image", "text", "keyword"}


# --- errors -------------------------------------------------------------------------------


def test_missing_index_is_a_clear_error(run_dir: Path, preset_path: Path, tmp_path: Path):
    import shutil

    moved = tmp_path / "no-index"
    shutil.copytree(run_dir, moved)
    shutil.rmtree(moved / "index")
    with pytest.raises(SelectError, match="meta.db is missing"):
        _select(moved, preset_path, embeddings=ScriptedEmbedder())


def test_missing_run_and_preset(tmp_path: Path, preset_path: Path):
    with pytest.raises(SelectError, match="run directory not found"):
        _select(tmp_path / "nope", preset_path, embeddings=ScriptedEmbedder())
    empty_run = tmp_path / "runs" / "empty"
    empty_run.mkdir(parents=True)
    with pytest.raises(SelectError, match="frames is missing"):
        _select(empty_run, preset_path, embeddings=ScriptedEmbedder())
    with pytest.raises(Exception, match="preset not found"):
        _select(empty_run, tmp_path / "nope.yaml", embeddings=ScriptedEmbedder())


def test_select_params_validation():
    with pytest.raises(SelectError, match="episode_half_s must be > 0"):
        select_params({"select": {"episode_half_s": 0}})
    with pytest.raises(SelectError, match="phash_min_hamming must be >= 0"):
        select_params({"select": {"phash_min_hamming": -1}})
    assert select_params({})["episode_half_s"] == 2.0
    assert select_params({})["phash_min_hamming"] == 8

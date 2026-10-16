"""Hybrid search tests (naryad 012): signals, RRF formula, determinism.

Tests build a toy run directory (frames.json + chunks.json + ocr.json +
transcript.json + a hand-filled vector store) and drive the search with a
scripted fake embedder (fixed vectors per text, journal of calls) -- no
network, no real providers, fully deterministic. The RRF score of a hit is
recomputed by hand in the tests from the ranks, proving the fusion uses
ranks, not raw scores (auditor check).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from razkadr.providers.types import Message
from razkadr.search import (
    BM25_B,
    BM25_K1,
    SearchConfig,
    SearchError,
    bm25_scores,
    hybrid_search,
)
from razkadr.vectorstore import LocalVectorStore

# toy world: 30 s video, 5 frames (t = 1, 6, 11, 16, 21 s), two transcript
# chunks and one OCR result; frame vectors are placed so that frame 6.0 s is
# the closest to the query vector along dim 0.
QUERY_VECTOR = [1.0, 0.0]
FRAME_VECTORS = {
    "frames/000000.jpg": [0.9, 0.1],
    "frames/000001.jpg": [0.99, 0.05],  # the hero frame: closest to the query
    "frames/000002.jpg": [0.5, 0.5],
    "frames/000003.jpg": [-0.8, 0.2],
    "frames/000004.jpg": [0.2, -0.9],
}
CHUNK_TEXTS = [
    "Смешайте муку с водой и замесите тесто",      # t 0..8  (covers 1.0, 6.0)
    "Обжарьте котлеты на сковороде до готовности",  # t 9..17 (covers 11.0, 16.0)
]
OCR_TEXT = "Готовое блюдо: подать горячим"           # t 19.0 (covers 21.0 via window)
VIDEO_LANG = "ru"
FRAMES_META = [
    {"t": 1.0 + i * 5.0, "file": f"frames/{i:06d}.jpg", "scene_id": i,
     "sharpness": 0.9 - 0.1 * i, "exposure": 0.9, "rejected": i == 3}
    for i in range(5)
]


class ScriptedEmbedder:
    """Fake EmbeddingProvider: known text -> fixed vector, unknown text ->
    deterministic pseudo-vector (sha-free, simple and stable); records every
    embed_texts call for translation assertions."""

    name = "scripted"
    model = "scripted-1"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.extend(texts)
        return [KNOWN_VECTORS.get(text, [0.0, 1.0]) for text in texts]

    def embed_images(self, paths) -> list[list[float]]:  # pragma: no cover
        raise AssertionError("search must not embed images")


KNOWN_VECTORS: dict[str, list[float]] = {
    "готовое блюдо": QUERY_VECTOR,
    "finished dish": [0.0, 1.0],
}


class YesLLM:
    """Fake LLMProvider: deterministic 'translation'."""

    name = "yes-llm"
    model = "yes-llm-1"

    def complete(self, messages: list[Message]) -> str:
        assert messages[0].role == "system"
        assert messages[-1].role == "user"
        return f"[перевод] {messages[-1].content}"


# --- toy run fixture ------------------------------------------------------------


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    run = tmp_path / "runs" / "toy"
    run.mkdir(parents=True)
    frames = []
    vectors = []
    for i in range(5):
        file = f"frames/{i:06d}.jpg"
        t = 1.0 + i * 5.0
        frame = {
            "t": t,
            "file": file,
            "scene_id": i,
            "sharpness": 0.9 - 0.1 * i,
            "phash": f"{i:016x}",
            "exposure": 0.9,
        }
        if i == 3:
            frame["rejected"] = True
            frame["reject_reason"] = "duplicate of 6.000s"
        frames.append(frame)
        vectors.append(FRAME_VECTORS[file])
    (run / "frames.json").write_text(
        json.dumps({"schema_version": "1.1", "frames": frames}), encoding="utf-8"
    )
    chunks = [
        {"t_start": 0.0, "t_end": 8.0, "text": CHUNK_TEXTS[0], "lang": VIDEO_LANG},
        {"t_start": 9.0, "t_end": 17.0, "text": CHUNK_TEXTS[1], "lang": VIDEO_LANG},
    ]
    (run / "chunks.json").write_text(
        json.dumps({"schema_version": "1.1", "lang": VIDEO_LANG, "chunks": chunks}),
        encoding="utf-8",
    )
    ocr = [{"t": 19.0, "text": OCR_TEXT, "sharpness": 0.8}]
    (run / "ocr.json").write_text(
        json.dumps({"schema_version": "1.1", "results": ocr}), encoding="utf-8"
    )
    transcript = {"schema_version": "1.1", "lang": VIDEO_LANG, "segments": []}
    (run / "transcript.json").write_text(json.dumps(transcript), encoding="utf-8")

    store = LocalVectorStore(run / "index")
    store.create(dim=2)
    accepted = [f for f in frames if not f.get("rejected")]
    accepted_vectors = [FRAME_VECTORS[f["file"]] for f in accepted]
    store.add(
        "frames",
        [f"frame-{f['t']}" for f in accepted],
        accepted_vectors,
        [
            {"file": f["file"], "t": f["t"], "scene_id": f["scene_id"], "sharpness": f["sharpness"]}
            for f in accepted
        ],
    )
    store.add(
        "chunks",
        ["chunk-0", "chunk-1"],
        [[0.0, 1.0], [0.7, 0.7]],
        [
            {"t": c["t_start"], "t_end": c["t_end"], "text": c["text"], "lang": VIDEO_LANG}
            for c in chunks
        ],
    )
    store.add(
        "ocr",
        ["ocr-0"],
        [[0.6, 0.8]],
        [{"t": 19.0, "t_end": 19.0, "text": OCR_TEXT, "lang": VIDEO_LANG}],
    )
    store.save()
    store.close()
    return run


def _config(**overrides) -> dict:
    base = {"rrf_k": 1, "k": 10, "candidates": 50}  # small k makes hand math easy
    base.update(overrides)
    return {"search": base}


# --- single signals --------------------------------------------------------------


def test_image_signal_alone_ranks_by_cosine(run_dir: Path):
    hits = hybrid_search(
        run_dir, "готовое блюдо", embeddings=ScriptedEmbedder(),
        config=_config(signals={"image": True, "text": False, "keyword": False}),
    )
    files = [hit.file for hit in hits]
    assert files[0] == "frames/000001.jpg"  # closest to the query vector
    assert "frames/000003.jpg" not in files  # rejected frame never returned
    top = hits[0]
    assert top.rank_parts["image"] == 1
    assert top.rank_parts["text"] is None and top.rank_parts["keyword"] is None
    assert top.score_parts["text"] == 0.0 and top.score_parts["keyword"] == 0.0
    assert top.score == pytest.approx(top.score_parts["image"])
    assert top.score == pytest.approx(1.0 / (1 + 1))  # rrf_k=1, rank 1


def test_text_signal_inherits_chunk_ranks_by_time(run_dir: Path):
    embedder = ScriptedEmbedder()
    # make the query sit exactly on the chunk-0 vector so chunk-0 ranks 1st
    KNOWN_VECTORS["запрос чанка"] = [0.0, 1.0]
    hits = hybrid_search(
        run_dir, "запрос чанка", embeddings=embedder,
        config=_config(signals={"image": False, "text": True, "keyword": False}),
    )
    by_file = {hit.file: hit for hit in hits}
    # chunk-0 spans 0..8 s: covers frames at 1.0 and 6.0 (+window 5.0)
    assert by_file["frames/000000.jpg"].rank_parts["text"] == 1
    assert by_file["frames/000001.jpg"].rank_parts["text"] == 1
    # frame at 21.0 s is covered only by the OCR text at 19.0 s (rank 2),
    # not by the best chunk: inheritance must take ranks, not coverage alone
    assert by_file["frames/000004.jpg"].rank_parts["text"] == 2
    assert "frames/000003.jpg" not in by_file  # rejected: not a hit
    # frame at 11.0 s: chunk-1 covers it directly, but chunk-0 (rank 1)
    # covers it through the +-5 s window (0..8 + 5 = 13 >= 11) -> best wins
    frame11 = by_file["frames/000002.jpg"]
    assert frame11.rank_parts["text"] == 1


def test_keyword_signal_bm25(run_dir: Path):
    hits = hybrid_search(
        run_dir, "готовое блюдо", embeddings=ScriptedEmbedder(),
        config=_config(signals={"image": False, "text": False, "keyword": True}),
    )
    by_file = {hit.file: hit for hit in hits}
    # OCR text "Готовое блюдо: ..." matches the query words -> rank 1, covers 21.0
    assert by_file["frames/000004.jpg"].rank_parts["keyword"] == 1
    # frames without a covering matching text get no keyword contribution
    assert by_file["frames/000000.jpg"].rank_parts["keyword"] is None
    assert by_file["frames/000000.jpg"].score == 0.0


def test_bm25_unit_math():
    corpus = [
        type("D", (), {"key": "a", "t_start": 0, "t_end": 1, "text": "готовое блюдо горячим"})(),
        type("D", (), {"key": "b", "t_start": 2, "t_end": 3, "text": "тесто месить"})(),
    ]
    scores = bm25_scores(corpus, "блюдо")
    assert scores[0] > 0.0 and scores[1] == 0.0
    # the classic Okapi formula, checked by hand for doc a (avg len is 2.5)
    tf, df, n, length, avg = 1, 1, 2, 3, 2.5
    idf = __import__("math").log((n - df + 0.5) / (df + 0.5) + 1.0)
    expected = idf * (tf * (BM25_K1 + 1)) / (tf + BM25_K1 * (1 - BM25_B + BM25_B * length / avg))
    assert scores[0] == pytest.approx(expected)


# --- RRF fusion (auditor: ranks, not raw scores) ----------------------------------


def test_rrf_is_the_hand_computed_rank_formula(run_dir: Path):
    """Every signal's rank is known from the single-signal runs; the fused
    score must be exactly sum(weight / (rrf_k + rank))."""
    embedder = ScriptedEmbedder()
    all_on = _config(signals={"image": True, "text": True, "keyword": True})
    fused = {hit.file: hit for hit in hybrid_search(run_dir, "готовое блюдо", embeddings=embedder, config=all_on)}
    image_only = {hit.file: hit for hit in hybrid_search(
        run_dir, "готовое блюдо", embeddings=ScriptedEmbedder(),
        config=_config(signals={"image": True, "text": False, "keyword": False}),
    )}

    hero = fused["frames/000001.jpg"]
    assert hero.rank_parts["image"] == image_only["frames/000001.jpg"].rank_parts["image"]
    hero_expected = 0.0
    for signal in ("image", "text", "keyword"):
        rank = hero.rank_parts[signal]
        if rank is not None:
            hero_expected += 1.0 / (1 + rank)  # rrf_k=1, weight=1
    assert hero.score == pytest.approx(hero_expected, abs=1e-12)

    # frame 000004: keyword rank 1 (OCR match) but a weak image match
    tail = fused["frames/000004.jpg"]
    tail_expected = 0.0
    for signal in ("image", "text", "keyword"):
        rank = tail.rank_parts[signal]
        if rank is not None:
            tail_expected += 1.0 / (1 + rank)
    assert tail.score == pytest.approx(tail_expected, abs=1e-12)
    assert tail.score_parts["keyword"] == pytest.approx(1.0 / (1 + tail.rank_parts["keyword"]))
    # ranks, not raw cosine scores: a huge cosine gap cannot beat one rank step
    assert tail.score_parts["keyword"] > tail.score_parts["image"]


def test_disabling_a_signal_changes_results_as_expected(run_dir: Path):
    """The text signal is the only one covering frame 000000 with rank 1
    (chunk-0); turning it off must drop that frame's score by exactly its
    text contribution."""
    embedder = ScriptedEmbedder()
    KNOWN_VECTORS["запрос чанка"] = [0.0, 1.0]
    both = {hit.file: hit for hit in hybrid_search(
        run_dir, "запрос чанка", embeddings=embedder,
        config=_config(signals={"image": True, "text": True, "keyword": False}),
    )}
    image_only = {hit.file: hit for hit in hybrid_search(
        run_dir, "запрос чанка", embeddings=embedder,
        config=_config(signals={"image": True, "text": False, "keyword": False}),
    )}
    frame = both["frames/000000.jpg"]
    assert frame.score_parts["text"] > 0.0
    diff = frame.score - image_only["frames/000000.jpg"].score
    assert diff == pytest.approx(frame.score_parts["text"], abs=1e-12)
    # and the disabled signal reports zero, not None
    assert image_only["frames/000000.jpg"].score_parts["text"] == 0.0


def test_weights_scale_the_contributions(run_dir: Path):
    hits = hybrid_search(
        run_dir, "готовое блюдо", embeddings=ScriptedEmbedder(),
        config=_config(
            signals={"image": True, "text": False, "keyword": False}, weights={"image": 0.5}
        ),
    )
    assert hits[0].score == pytest.approx(0.5 / (1 + 1))


# --- determinism, filters, edges ---------------------------------------------------


def test_search_is_deterministic(run_dir: Path):
    kwargs = {"embeddings": ScriptedEmbedder(), "config": _config()}
    first = hybrid_search(run_dir, "готовое блюдо", **kwargs)
    second = hybrid_search(run_dir, "готовое блюдо", **kwargs)
    assert [hit.to_dict() for hit in first] == [hit.to_dict() for hit in second]


def test_ties_break_by_file(run_dir: Path):
    """Frames sharing the same inherited rank (both covered by chunk-0) tie
    on the text signal; the one with the smaller file path wins."""
    hits = hybrid_search(
        run_dir, "запрос чанка", embeddings=ScriptedEmbedder(),
        config=_config(signals={"image": False, "text": True, "keyword": False}),
    )
    covered = [hit for hit in hits if hit.rank_parts["text"] == 1]
    assert [hit.file for hit in covered] == sorted(hit.file for hit in covered)


def test_min_sharpness_and_interval_filters(run_dir: Path):
    hits = hybrid_search(
        run_dir, "готовое блюдо", embeddings=ScriptedEmbedder(),
        config=_config(signals={"image": True, "text": False, "keyword": False}, min_sharpness=0.85),
    )
    assert all(hit.sharpness >= 0.85 for hit in hits)
    assert "frames/000004.jpg" not in [hit.file for hit in hits]  # sharpness 0.5

    hits = hybrid_search(
        run_dir, "готовое блюдо", embeddings=ScriptedEmbedder(),
        config=_config(signals={"image": True, "text": False, "keyword": False},
                       t_start=9.0, t_end=13.0),
    )
    assert {hit.file for hit in hits} == {"frames/000002.jpg"}  # t=11.0 only


def test_score_parts_present_for_every_hit(run_dir: Path):
    hits = hybrid_search(run_dir, "готовое блюдо", embeddings=ScriptedEmbedder(), config=_config())
    assert hits
    for hit in hits:
        assert set(hit.score_parts) == {"image", "text", "keyword"}
        assert set(hit.rank_parts) == {"image", "text", "keyword"}
        assert hit.query_used == "готовое блюдо"
        assert hit.translated_from is None


def test_empty_texts_do_not_break_text_signals(run_dir: Path):
    """Auditor check: silent video, no captions -- only the image signal
    works, text/keyword contributions are zero, nothing crashes. The store
    is rebuilt without text spaces to match the empty artifacts."""
    (run_dir / "chunks.json").write_text(
        json.dumps({"schema_version": "1.1", "lang": VIDEO_LANG, "chunks": []}), encoding="utf-8"
    )
    (run_dir / "ocr.json").write_text(
        json.dumps({"schema_version": "1.1", "results": []}), encoding="utf-8"
    )
    store = LocalVectorStore(run_dir / "index")
    store.create(dim=2)
    accepted = [f for f in FRAMES_META if not f["rejected"]]
    store.add(
        "frames",
        [f"frame-{f['t']}" for f in accepted],
        [FRAME_VECTORS[f["file"]] for f in accepted],
        [{"file": f["file"], "t": f["t"], "scene_id": f["scene_id"], "sharpness": f["sharpness"]} for f in accepted],
    )
    store.save()
    store.close()
    hits = hybrid_search(run_dir, "готовое блюдо", embeddings=ScriptedEmbedder(), config=_config())
    assert hits
    for hit in hits:
        assert hit.score_parts["text"] == 0.0 and hit.score_parts["keyword"] == 0.0
        assert hit.rank_parts["text"] is None and hit.rank_parts["keyword"] is None
    assert hits[0].file == "frames/000001.jpg"  # image signal still ranks


def test_translation_through_llm(run_dir: Path):
    embedder = ScriptedEmbedder()
    hits = hybrid_search(
        run_dir, "finished dish", query_lang="en", embeddings=embedder, llm=YesLLM(),
        config=_config(signals={"image": True, "text": False, "keyword": False}),
    )
    assert embedder.calls[-1].startswith("[перевод]")  # the translation was embedded
    for hit in hits:
        assert hit.query_used.startswith("[перевод]")
        assert hit.translated_from == "finished dish"  # the original query


def test_same_language_skips_translation(run_dir: Path):
    embedder = ScriptedEmbedder()
    hybrid_search(
        run_dir, "готовое блюдо", query_lang="ru", embeddings=embedder, llm=YesLLM(), config=_config()
    )
    assert embedder.calls[-1] == "готовое блюдо"  # embedded as-is


def test_translation_without_llm_is_a_clear_error(run_dir: Path):
    with pytest.raises(SearchError, match="translation needs an LLM provider"):
        hybrid_search(
            run_dir, "finished dish", query_lang="en", embeddings=ScriptedEmbedder(), llm=None,
            config=_config(signals={"image": True, "text": False, "keyword": False}),
        )


def test_missing_embedder_is_a_clear_error(run_dir: Path):
    with pytest.raises(SearchError, match="need an embeddings provider"):
        hybrid_search(
            run_dir, "готовое блюдо", embeddings=None,
            config=_config(signals={"image": True, "text": False, "keyword": False}),
        )
    # keyword-only works without any provider
    hits = hybrid_search(
        run_dir, "готовое блюдо", embeddings=None,
        config=_config(signals={"image": False, "text": False, "keyword": True}),
    )
    assert hits


def test_empty_query_and_missing_run(run_dir: Path, tmp_path: Path):
    with pytest.raises(SearchError, match="query is empty"):
        hybrid_search(run_dir, "   ", embeddings=ScriptedEmbedder(), config=_config())
    with pytest.raises(SearchError, match="run directory not found"):
        hybrid_search(tmp_path / "nope", "запрос", embeddings=ScriptedEmbedder(), config=_config())


def test_missing_index_fails_clearly(run_dir: Path, tmp_path: Path):
    import shutil

    moved = tmp_path / "no-index"
    shutil.copytree(run_dir, moved)
    shutil.rmtree(moved / "index")
    with pytest.raises(SearchError, match="meta.db is missing"):
        hybrid_search(moved, "готовое блюдо", embeddings=ScriptedEmbedder(), config=_config())


def test_config_validation():
    with pytest.raises(SearchError, match="unknown signals"):
        SearchConfig.from_config({"search": {"signals": {"video": True}}})
    with pytest.raises(SearchError, match="weight image must be in"):
        SearchConfig.from_config({"search": {"weights": {"image": 1.5}}})
    with pytest.raises(SearchError, match="rrf_k must be >= 1"):
        SearchConfig.from_config({"search": {"rrf_k": 0}})
    with pytest.raises(SearchError, match="k must be >= 1"):
        SearchConfig.from_config({"search": {"k": 0}})
    cfg = SearchConfig.from_config(None)
    assert cfg.rrf_k == 60 and cfg.k == 20 and cfg.text_window_s == 5.0

"""VectorStore tests (naryad 010): cosine math, filters, persistence, corruption.

The store is exercised directly with hand-written vectors: orthogonal
and collinear directions give provable cosine scores, so the search
order and scores are checked against hand-computed expectations rather
than against the same code under test.
"""

import math

import pytest

from razkadr.vectorstore import (
    Filters,
    LocalVectorStore,
    VectorStoreError,
)


@pytest.fixture()
def store(tmp_path):
    """An open store of dim 2 with three hand-written vectors."""
    st = LocalVectorStore(tmp_path / "index")
    st.create(dim=2)
    st.add(
        "frames",
        ["frames:a.jpg", "frames:b.jpg", "frames:c.jpg"],
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        [
            {"file": "frames/a.jpg", "t": 0.0, "scene_id": 0, "sharpness": 0.9, "source": "video.mp4"},
            {"file": "frames/b.jpg", "t": 1.0, "scene_id": 0, "sharpness": 0.5, "source": "video.mp4"},
            {"file": "frames/c.jpg", "t": 2.0, "scene_id": 1, "sharpness": 0.1, "source": "other.mp4"},
        ],
    )
    yield st
    st.close()


def test_exact_match_scores_one(store):
    hits = store.search([1.0, 0.0], k=3)
    assert hits[0][0] == "frames:a.jpg"
    assert hits[0][1] == pytest.approx(1.0)
    assert hits[0][2]["file"] == "frames/a.jpg"


def test_cosine_order_and_orthogonality(store):
    hits = store.search([1.0, 0.0], k=3)
    # [1,0] -> a=1.0; [1,1]/sqrt(2)=0.7071; [0,1]=0.0
    assert [h[0] for h in hits] == ["frames:a.jpg", "frames:c.jpg", "frames:b.jpg"]
    assert hits[1][1] == pytest.approx(math.sqrt(0.5))
    assert hits[2][1] == pytest.approx(0.0, abs=1e-6)


def test_k_limits_result_count(store):
    assert len(store.search([1.0, 0.0], k=1)) == 1
    assert len(store.search([1.0, 0.0], k=99)) == 3
    assert store.search([1.0, 0.0], k=0) == []
    assert store.search([1.0, 0.0], k=-5) == []


def test_ties_broken_by_id_asc(tmp_path):
    """Two collinear vectors score equally; the order is deterministic."""
    st = LocalVectorStore(tmp_path / "index")
    st.create(dim=1)
    st.add(
        "frames",
        ["frames:z.jpg", "frames:a.jpg"],
        [[1.0], [2.0]],  # same direction, scale does not matter for cosine
        [{}, {}],
    )
    hits = st.search([1.0], k=2)
    assert [h[0] for h in hits] == ["frames:a.jpg", "frames:z.jpg"]
    assert hits[0][1] == pytest.approx(1.0)
    st.close()


def test_zero_query_vector_no_nan(store):
    hits = store.search([0.0, 0.0], k=3)
    assert all(math.isfinite(h[1]) and h[1] == 0.0 for h in hits)
    # deterministic tie order
    assert [h[0] for h in hits] == ["frames:a.jpg", "frames:b.jpg", "frames:c.jpg"]


def test_zero_stored_vector_scores_zero(tmp_path):
    st = LocalVectorStore(tmp_path / "index")
    st.create(dim=2)
    st.add("chunks", ["chunks:zero"], [[0.0, 0.0]], [{"text": "empty"}])
    hits = st.search([1.0, 0.0], k=1)
    assert hits[0][1] == 0.0
    st.close()


# --- filters ------------------------------------------------------------------


def test_filter_spaces(store):
    st = store
    st.add("chunks", ["chunks:1"], [[1.0, 0.0]], [{"text": "текст"}])
    hits = st.search([1.0, 0.0], k=10, filters=Filters(spaces=("chunks",)))
    assert [h[0] for h in hits] == ["chunks:1"]
    hits = st.search([1.0, 0.0], k=10, filters=Filters(spaces=("frames", "ocr")))
    assert all(h[2]["space"] == "frames" for h in hits)


def test_filter_unknown_space_is_empty(store):
    assert store.search([1.0, 0.0], k=10, filters=Filters(spaces=("descriptions",))) == []


def test_filter_source(store):
    hits = store.search([1.0, 1.0], k=10, filters=Filters(source="other.mp4"))
    assert [h[0] for h in hits] == ["frames:c.jpg"]


def test_filter_time_interval_overlap(store):
    # [0, 0] and [1, 1] instants; [2, 2]; interval (0.5, 1.5) overlaps only t=1
    hits = store.search([1.0, 0.0], k=10, filters=Filters(t_start=0.5, t_end=1.5))
    assert [h[0] for h in hits] == ["frames:b.jpg"]
    # an open interval from 1.5 keeps the frame at t=2
    hits = store.search([1.0, 0.0], k=10, filters=Filters(t_start=1.5))
    assert [h[0] for h in hits] == ["frames:c.jpg"]
    # an open interval until 1.5 keeps t=0 and t=1
    hits = store.search([1.0, 0.0], k=10, filters=Filters(t_end=1.5))
    assert [h[0] for h in hits] == ["frames:a.jpg", "frames:b.jpg"]


def test_filter_min_sharpness_excludes_texts(store):
    store.add("chunks", ["chunks:sharp-text"], [[1.0, 0.0]], [{"text": "no sharpness"}])
    hits = store.search([1.0, 0.0], k=10, filters=Filters(min_sharpness=0.3))
    # text vectors carry no sharpness: excluded even though they match best
    assert [h[0] for h in hits] == ["frames:a.jpg", "frames:b.jpg"]


def test_filters_combined(store):
    hits = store.search(
        [1.0, 0.0], k=10, filters=Filters(source="video.mp4", min_sharpness=0.6)
    )
    assert [h[0] for h in hits] == ["frames:a.jpg"]


# --- persistence --------------------------------------------------------------


def test_reload_from_disk_same_results(tmp_path):
    st = LocalVectorStore(tmp_path / "index")
    st.create(dim=2)
    st.add(
        "chunks",
        ["chunks:1", "chunks:2"],
        [[3.0, 4.0], [-1.0, 0.5]],
        [{"text": "борщ с пампушками", "lang": "ru", "t": 0.0, "t_end": 5.0}, {"text": "и чай"}],
    )
    st.save()
    st.close()

    reopened = LocalVectorStore(tmp_path / "index").open()
    hits = reopened.search([3.0, 4.0], k=2)
    assert hits[0][0] == "chunks:1"
    assert hits[0][1] == pytest.approx(1.0)
    assert hits[0][2]["text"] == "борщ с пампушками"
    assert hits[0][2]["lang"] == "ru"
    assert reopened.count() == 2
    reopened.close()


def test_save_creates_vectors_file(tmp_path):
    st = LocalVectorStore(tmp_path / "index")
    st.create(dim=2)
    st.add("frames", ["frames:x"], [[1.0, 2.0]], [{"file": "x"}])
    st.save()
    assert (tmp_path / "index" / "vectors.bin").stat().st_size == 1 * 2 * 4
    st.close()


def test_add_then_search_without_save_uses_in_memory_matrix(tmp_path):
    st = LocalVectorStore(tmp_path / "index")
    st.create(dim=1)
    st.add("frames", ["frames:x"], [[1.0]], [{}])
    assert st.search([1.0], k=1)[0][0] == "frames:x"
    st.close()


# --- errors and corruption ----------------------------------------------------


def test_dim_mismatch_on_add_rejected(store):
    with pytest.raises(VectorStoreError, match="dim 3 != store dim 2"):
        store.add("frames", ["frames:bad"], [[1.0, 0.0, 0.0]], [{}])


def test_duplicate_id_rejected(store):
    with pytest.raises(VectorStoreError, match="duplicate vector id"):
        store.add("frames", ["frames:a.jpg"], [[0.5, 0.5]], [{}])


def test_unknown_metadata_key_rejected(store):
    with pytest.raises(VectorStoreError, match="not storable"):
        store.add("frames", ["frames:k"], [[1.0, 0.0]], [{"api_key": "sk-secret"}])


def test_unknown_space_rejected(store):
    with pytest.raises(VectorStoreError, match="space must be one of"):
        store.add("descriptions", ["d:1"], [[1.0, 0.0]], [{}])


def test_length_mismatch_rejected(store):
    with pytest.raises(VectorStoreError, match="equal lengths"):
        store.add("frames", ["frames:a", "frames:b"], [[1.0, 0.0]], [{}])


def test_search_before_open_rejected(tmp_path):
    st = LocalVectorStore(tmp_path / "index")
    with pytest.raises(VectorStoreError, match="not open"):
        st.search([1.0], k=1)


def test_open_missing_store_rejected(tmp_path):
    with pytest.raises(VectorStoreError, match="no vector store"):
        LocalVectorStore(tmp_path / "absent").open()


def test_open_corrupt_meta_db_missing_dim(tmp_path):
    d = tmp_path / "index"
    d.mkdir()
    import sqlite3

    conn = sqlite3.connect(d / "meta.db")
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.commit()
    conn.close()
    with pytest.raises(VectorStoreError, match="meta.dim is missing"):
        LocalVectorStore(d).open()


def test_open_corrupt_bin_size_mismatch(tmp_path):
    st = LocalVectorStore(tmp_path / "index")
    st.create(dim=2)
    st.add("frames", ["frames:x"], [[1.0, 0.0]], [{}])
    st.save()
    st.close()
    (tmp_path / "index" / "vectors.bin").write_bytes(b"\x00" * 7)
    with pytest.raises(VectorStoreError, match="vectors.bin holds"):
        LocalVectorStore(tmp_path / "index").open()


def test_open_db_rows_without_bin_rejected(tmp_path):
    st = LocalVectorStore(tmp_path / "index")
    st.create(dim=2)
    st.add("frames", ["frames:x"], [[1.0, 0.0]], [{}])
    st._conn.commit()  # rows committed, vectors.bin never written
    st.close()
    with pytest.raises(VectorStoreError, match="vectors.bin is missing"):
        LocalVectorStore(tmp_path / "index").open()


def test_create_zero_dim_rejected(tmp_path):
    with pytest.raises(VectorStoreError, match="dim must be positive"):
        LocalVectorStore(tmp_path / "index").create(dim=0)


def test_query_dim_mismatch_rejected(store):
    with pytest.raises(VectorStoreError, match="query dim"):
        store.search([1.0, 0.0, 0.0], k=1)


def test_protocol_satisfied(store):
    from razkadr.vectorstore import VectorStore as VectorStoreProtocol

    assert isinstance(store, VectorStoreProtocol)


def test_public_filter_fields():
    """Guards the public filter contract: exactly these filters exist."""
    fields = {f for f in Filters.__dataclass_fields__}
    assert fields == {"spaces", "source", "t_start", "t_end", "min_sharpness"}


def test_meta_roundtrip(store):
    store.set_meta("provider", "fake")
    store.set_meta("model", "fake-embed-v2")
    assert store.get_meta("provider") == "fake"
    assert store.get_meta("model") == "fake-embed-v2"
    assert store.get_meta("absent") is None

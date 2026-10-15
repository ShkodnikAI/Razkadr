"""VectorStore interface and the core implementation (naryad 010).

The index stage (stages/index.py) writes vectors through this module;
hybrid search (naryad 012) and the select stage (naryad 014) read them.
The interface is intentionally provider-agnostic: it stores (id, vector,
metadata) tuples and answers nearest-neighbour queries - nothing in it
knows which embedding API produced the vectors.

Core storage (ADR-0007, proposed): one SQLite database for ids and
metadata plus one raw float32 file for the matrix, searched exactly by
cosine similarity with numpy. This holds to the order of 10^5 vectors
per run; pgvector (or another ANN store) is deferred to the web wave,
when indexes outgrow a single machine - the interface does not change.

Layout inside the store directory (the run's ``index/``):

    meta.db        SQLite: table ``vectors`` (one row per vector),
                   table ``meta`` (dim, provider, model, version, ...)
    vectors.bin    N x dim float32 matrix, row order == vectors.row

Search filters (``Filters``, all optional, AND-combined):
- ``spaces``    provenance of the vector: frames | chunks | ocr
- ``source``    exact match on the video the vector came from
- ``t_start``/``t_end``  keep vectors whose time span intersects the
                interval (frames/ocr span the instant t..t, chunks span
                t_start..t_end, so a chunk overlapping the interval is kept)
- ``min_sharpness``      frames only; vectors without a sharpness metric
                (texts) are excluded

Cosine similarity: vectors are stored exactly as the provider returned
them; both sides are L2-normalized at query time (a zero vector scores
0.0 against everything instead of producing NaN). Deterministic order:
score desc, then id asc - same data, same query, same result list.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Self, runtime_checkable

import numpy as np

from razkadr.log import get_logger

logger = get_logger("vectorstore")

VECTORS_BIN = "vectors.bin"
META_DB = "meta.db"

# Vector provenance; stages/index.py writes one space per input artifact.
SPACE_KINDS = ("frames", "chunks", "ocr")


class VectorStoreError(Exception):
    """Vector store failure with a clear, user-facing message."""


@dataclass(frozen=True)
class Filters:
    """Search filters; None fields are ignored (see module docstring)."""

    spaces: tuple[str, ...] | None = None
    source: str | None = None
    t_start: float | None = None
    t_end: float | None = None
    min_sharpness: float | None = None


@runtime_checkable
class VectorStore(Protocol):
    """Storage of (id, vector, metadata) with exact nearest-neighbour search."""

    def add(
        self,
        space: str,
        ids: list[str],
        vectors: list[list[float]],
        metadatas: list[dict],
    ) -> None: ...

    def search(
        self,
        query: list[float] | object,
        k: int,
        filters: Filters | None = None,
    ) -> list[tuple[str, float, dict]]: ...

    def count(self) -> int: ...

    def close(self) -> None: ...


_SCHEMA = """
CREATE TABLE IF NOT EXISTS vectors (
    id TEXT PRIMARY KEY,
    space TEXT NOT NULL,
    row INTEGER NOT NULL UNIQUE,
    source TEXT,
    file TEXT,
    t REAL,
    t_end REAL,
    scene_id INTEGER,
    sharpness REAL,
    lang TEXT,
    text TEXT
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_vectors_space ON vectors(space);
"""

# Metadata dict keys -> columns; every value must be JSON-able and scalar.
_META_COLUMNS = ("source", "file", "t", "t_end", "scene_id", "sharpness", "lang", "text")


def _normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize rows; zero rows stay zero (they score 0, never NaN)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0.0, 1.0, norms)


class LocalVectorStore:
    """SQLite + numpy exact-search store (ADR-0007); one directory per index."""

    def __init__(self, dir: str | Path) -> None:
        self.dir = Path(dir)
        self.db_path = self.dir / META_DB
        self.bin_path = self.dir / VECTORS_BIN
        self._conn: sqlite3.Connection | None = None
        self._dim: int | None = None
        self._matrix: np.ndarray | None = None
        self._row_norms_cache: np.ndarray | None = None

    # -- lifecycle -----------------------------------------------------------

    def create(self, dim: int) -> None:
        """Fresh empty store; wipes any previous content of the directory."""
        if dim <= 0:
            raise VectorStoreError(f"dim must be positive, got {dim}")
        self.dir.mkdir(parents=True, exist_ok=True)
        for suffix in ("", "-wal", "-shm"):
            path = self.dir / (META_DB + suffix)
            if path.exists():
                path.unlink()
        if self.bin_path.exists():
            self.bin_path.unlink()
        self._connect()
        self._conn.executescript(_SCHEMA)
        self._set_meta("dim", str(dim))
        self._dim = dim
        self._matrix = np.zeros((0, dim), dtype=np.float32)

    def open(self) -> LocalVectorStore:
        """Load an existing store; the matrix is memory-mapped, not copied."""
        if not self.db_path.is_file():
            raise VectorStoreError(
                f"no vector store at {self.dir} (meta.db is missing; run the index stage first)"
            )
        self._connect()
        dim = self._get_meta("dim")
        if dim is None:
            raise VectorStoreError(f"corrupt store at {self.dir}: meta.dim is missing")
        self._dim = int(dim)
        num = self._count_rows()
        if self.bin_path.is_file():
            expected = num * self._dim * 4
            actual = self.bin_path.stat().st_size
            if actual != expected:
                raise VectorStoreError(
                    f"corrupt store at {self.dir}: vectors.bin holds {actual} bytes, "
                    f"expected {expected} for {num} vectors of dim {self._dim}"
                )
            self._matrix = np.memmap(self.bin_path, dtype=np.float32, mode="r", shape=(num, self._dim))
        elif num:
            raise VectorStoreError(
                f"corrupt store at {self.dir}: {num} vectors in meta.db but vectors.bin is missing"
            )
        else:
            self._matrix = np.zeros((0, self._dim), dtype=np.float32)
        return self

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        self._matrix = None

    def __enter__(self) -> Self:
        if self._conn is None:
            self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- writes --------------------------------------------------------------

    def add(
        self,
        space: str,
        ids: list[str],
        vectors: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        """Append vectors (upsert by id); call save() to persist the matrix."""
        if self._conn is None or self._dim is None:
            raise VectorStoreError("store is not open: call create() or open() first")
        if space not in SPACE_KINDS:
            raise VectorStoreError(f"space must be one of {SPACE_KINDS}, got {space!r}")
        if not (len(ids) == len(vectors) == len(metadatas)):
            raise VectorStoreError(
                f"ids, vectors and metadatas must have equal lengths, got "
                f"{len(ids)}, {len(vectors)}, {len(metadatas)}"
            )
        for vector in vectors:
            if len(vector) != self._dim:
                raise VectorStoreError(
                    f"vector dim {len(vector)} != store dim {self._dim} (wrong model or stale index)"
                )
        rows = self._count_rows()
        for offset, (vid, vector, meta) in enumerate(zip(ids, vectors, metadatas)):
            unknown = set(meta) - set(_META_COLUMNS)
            if unknown:
                raise VectorStoreError(
                    f"metadata keys {sorted(unknown)} are not storable; known: {list(_META_COLUMNS)}"
                )
            values = {key: meta.get(key) for key in _META_COLUMNS}
            try:
                self._conn.execute(
                    "INSERT INTO vectors (id, space, row, source, file, t, t_end, scene_id, "
                    "sharpness, lang, text) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (vid, space, rows + offset, *values.values()),
                )
            except sqlite3.IntegrityError:
                raise VectorStoreError(
                    f"duplicate vector id {vid!r}: ids must be unique within a store"
                ) from None
        new_rows = np.asarray(vectors, dtype=np.float32).reshape(len(vectors), self._dim)
        if self._matrix is not None and self._matrix.size:
            self._matrix = np.concatenate([np.asarray(self._matrix), new_rows])
        else:
            self._matrix = new_rows
        self._row_norms_cache = None

    def save(self) -> None:
        """Persist the matrix atomically and commit the metadata."""
        if self._conn is None or self._dim is None:
            raise VectorStoreError("store is not open: call create() or open() first")
        rows = self._count_rows()
        matrix = np.ascontiguousarray(
            (self._matrix if self._matrix is not None else np.zeros((0, self._dim), np.float32)),
            dtype=np.float32,
        )
        if matrix.shape[0] != rows:
            raise VectorStoreError(
                f"matrix rows {matrix.shape[0]} != db rows {rows} (internal error)"
            )
        tmp = self.bin_path.with_suffix(self.bin_path.suffix + ".tmp")
        matrix.tofile(tmp)
        os.replace(tmp, self.bin_path)
        self._conn.commit()
        self._matrix = matrix
        self._row_norms_cache = None

    # -- reads ---------------------------------------------------------------

    def search(
        self,
        query: list[float] | object,
        k: int,
        filters: Filters | None = None,
    ) -> list[tuple[str, float, dict]]:
        """Exact cosine nearest neighbours; [(id, score, metadata)], best first.

        Two-phase read: phase 1 fetches (id, row) of the candidates
        (SQL WHERE for filters), numpy scores them against the matrix
        (gemv over the memory-mapped vectors, row norms cached), and
        phase 2 fetches the metadata columns of the top-k rows only -
        the per-query cost is one matrix pass instead of materializing
        every row's metadata. Deterministic order: score desc, then id
        asc (the k-boundary is resolved the same way: a score tie at
        the cut is won by the smaller id). k <= 0 returns an empty
        list; a zero query vector returns zero scores.
        """
        if self._conn is None or self._matrix is None:
            raise VectorStoreError("store is not open: call create() or open() first")
        if k <= 0:
            return []
        q = np.asarray(query, dtype=np.float32)
        if q.shape != (self._dim,):
            raise VectorStoreError(
                f"query dim {q.shape[0]} != store dim {self._dim} (wrong model or stale index)"
            )
        q_norm = _normalize(q.reshape(1, -1))[0]

        where, args = self._where(filters)
        if where:
            sql = f"SELECT id, row FROM vectors WHERE {' AND '.join(where)}"
            candidates = self._conn.execute(sql, args).fetchall()
            if not candidates:
                return []
            ids = [c[0] for c in candidates]
            rows = np.fromiter((c[1] for c in candidates), dtype=np.int64, count=len(candidates))
            subset = np.asarray(self._matrix)[rows]
            norms = np.linalg.norm(subset, axis=1)
            scores = (subset @ q_norm) / np.where(norms == 0.0, 1.0, norms)
            row_of = rows
        else:
            num = self._count_rows()
            if num == 0:
                return []
            ids = [r[0] for r in self._conn.execute("SELECT id FROM vectors ORDER BY row")]
            norms = self._row_norms()
            # gemv over the memmap: one streaming pass, no matrix copy
            scores = (self._matrix @ q_norm) / np.where(norms == 0.0, 1.0, norms)
            row_of = np.arange(num, dtype=np.int64)

        # top-k superset by score (argpartition), then the id tie-break
        n = len(ids)
        take = min(k, n)
        if n > take:
            threshold = scores[np.argpartition(scores, n - take)[n - take]]
            superset = np.nonzero(scores >= threshold)[0]
        else:
            superset = range(n)
        order = sorted((int(i) for i in superset), key=lambda i: (-float(scores[i]), ids[i]))[:take]

        # phase 2: metadata of the chosen rows only
        wanted_rows = [int(row_of[i]) for i in order]
        by_row: dict[int, tuple] = {}
        for start in range(0, len(wanted_rows), 500):
            chunk = wanted_rows[start : start + 500]
            marks = ",".join("?" for _ in chunk)
            sql = ("SELECT row, space, source, file, t, t_end, scene_id, sharpness, lang, text "
                   f"FROM vectors WHERE row IN ({marks})")
            for record in self._conn.execute(sql, chunk):
                by_row[record[0]] = record[1:]

        results: list[tuple[str, float, dict]] = []
        for i in order:
            space, source, file, t, t_end, scene_id, sharpness, lang, text = by_row[int(row_of[i])]
            meta = {
                "space": space,
                "source": source,
                "file": file,
                "t": t,
                "t_end": t_end,
                "scene_id": scene_id,
                "sharpness": sharpness,
                "lang": lang,
                "text": text,
            }
            results.append(
                (ids[i], float(scores[i]), {key: value for key, value in meta.items() if value is not None})
            )
        return results

    def count(self, space: str | None = None) -> int:
        """Number of vectors, optionally within one space."""
        if self._conn is None:
            raise VectorStoreError("store is not open: call create() or open() first")
        if space is None:
            return self._count_rows()
        return int(self._conn.execute("SELECT COUNT(*) FROM vectors WHERE space = ?", (space,)).fetchone()[0])

    # -- meta ----------------------------------------------------------------

    def set_meta(self, key: str, value: str) -> None:
        if self._conn is None:
            raise VectorStoreError("store is not open: call create() or open() first")
        self._set_meta(key, value)
        self._conn.commit()

    def get_meta(self, key: str) -> str | None:
        if self._conn is None:
            raise VectorStoreError("store is not open: call create() or open() first")
        return self._get_meta(key)

    @property
    def dim(self) -> int:
        if self._dim is None:
            raise VectorStoreError("store is not open: call create() or open() first")
        return self._dim

    # -- internals -----------------------------------------------------------

    def _connect(self) -> None:
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")

    def _set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def _get_meta(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def _count_rows(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0])

    def _row_norms(self) -> np.ndarray:
        """L2 norms of all matrix rows, cached (blockwise: no huge temporaries)."""
        num = self._matrix.shape[0]
        if self._row_norms_cache is None or len(self._row_norms_cache) != num:
            norms = np.empty(num, dtype=np.float64)
            block = 8192
            for start in range(0, num, block):
                chunk = np.asarray(self._matrix[start : start + block], dtype=np.float64)
                norms[start : start + block] = np.sqrt(np.einsum("ij,ij->i", chunk, chunk))
            self._row_norms_cache = norms
        return self._row_norms_cache

    @staticmethod
    def _where(filters: Filters | None) -> tuple[list[str], list[object]]:
        if filters is None:
            return [], []
        where: list[str] = []
        args: list[object] = []
        if filters.spaces:
            where.append("space IN ({})".format(",".join("?" for _ in filters.spaces)))
            args.extend(filters.spaces)
        if filters.source is not None:
            where.append("source = ?")
            args.append(filters.source)
        if filters.t_start is not None and filters.t_end is not None:
            # overlap: [t, t_end] intersects [t_start, t_end]; a vector without
            # an explicit t_end spans the instant COALESCE(t_end, t) = t
            where.append("t <= ? AND COALESCE(t_end, t) >= ?")
            args.extend([filters.t_end, filters.t_start])
        elif filters.t_start is not None:
            where.append("COALESCE(t_end, t) >= ?")
            args.append(filters.t_start)
        elif filters.t_end is not None:
            where.append("t <= ?")
            args.append(filters.t_end)
        if filters.min_sharpness is not None:
            where.append("sharpness >= ?")  # NULL (texts) compares false: excluded
            args.append(filters.min_sharpness)
        return where, args

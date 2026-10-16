"""Hybrid frame search: image + text + keyword signals fused by RRF (naryad 012).

Given a run directory (the artifacts of the indexing pipeline) and a text
query, returns the best frames: three signals are computed independently and
fused with Reciprocal Rank Fusion, so the score is built from RANKS in each
signal's list, never from raw (incomparable) scores:

- ``image``: cosine similarity between the query embedding and the frame
  vectors of the index (space ``frames``, naryad 010);
- ``text``: cosine similarity between the query embedding and the text
  vectors (spaces ``chunks`` + ``ocr``); a frame INHERITS the rank of every
  text whose time span (chunk t_start..t_end, OCR instant) covers the frame
  within ``text_window_s`` -- the frame gets the best (lowest) inherited
  rank;
- ``keyword``: BM25 over the chunk + OCR texts (own implementation, no
  external services), inherited by frames through the same time window.

    score(frame) = sum over enabled signals s of weight_s / (rrf_k + rank_s)

Every hit carries ``score_parts`` ({signal: contribution}) and
``rank_parts`` ({signal: rank or None}) -- the auditor and eval must be able
to tell which signal fired. When the query language differs from the video
language (transcript.json), the query is translated through the LLM provider
layer first; the translation is recorded on every hit.

The search is deterministic: no randomness, stable sorts, and ties are
broken by the frame file path (and by the text key inside a signal).
Rejected frames are never returned (the index holds accepted frames only;
the candidate projection filters frames.json as a second guard).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from razkadr.log import get_logger
from razkadr.vectorstore import Filters, LocalVectorStore, VectorStoreError

logger = get_logger("search")

TRANSLATE_PROMPT_FILE = "translate.v1.md"
TRANSLATE_PROMPT = (Path(__file__).parent / "prompts" / TRANSLATE_PROMPT_FILE).read_text(encoding="utf-8")

SIGNALS = ("image", "text", "keyword")
DEFAULT_RRF_K = 60
DEFAULT_K = 20
DEFAULT_TEXT_WINDOW_S = 5.0
DEFAULT_CANDIDATES = 200
# classic BM25 parameters (Robertson & Zaragoza); not tuned measurements
BM25_K1 = 1.2
BM25_B = 0.75


class SearchError(Exception):
    """Hybrid search cannot run (bad config, missing index, no embedder)."""


@dataclass(frozen=True)
class SearchConfig:
    """All knobs of the hybrid search; every stage of naryad 015 compares
    configurations by flipping these."""

    signals: dict[str, bool] = field(
        default_factory=lambda: {signal: True for signal in SIGNALS}
    )
    weights: dict[str, float] = field(
        default_factory=lambda: {signal: 1.0 for signal in SIGNALS}
    )
    rrf_k: int = DEFAULT_RRF_K
    k: int = DEFAULT_K
    text_window_s: float = DEFAULT_TEXT_WINDOW_S
    candidates: int = DEFAULT_CANDIDATES
    min_sharpness: float | None = None
    t_start: float | None = None
    t_end: float | None = None
    exclude_rejected: bool = True

    @staticmethod
    def from_config(config: dict | None) -> SearchConfig:
        """Build from the [search] section of razkadr.toml (None = defaults)."""
        section = dict((config or {}).get("search") or {})
        signals = {**{s: True for s in SIGNALS}, **(section.get("signals") or {})}
        weights = {**{s: 1.0 for s in SIGNALS}, **(section.get("weights") or {})}
        unknown = (set(signals) | set(weights)) - set(SIGNALS)
        if unknown:
            raise SearchError(f"[search] unknown signals: {sorted(unknown)} (known: {list(SIGNALS)})")
        for name, weight in weights.items():
            if not 0.0 <= weight <= 1.0:
                raise SearchError(f"[search] weight {name} must be in [0, 1], got {weight}")
        cfg = SearchConfig(
            signals={s: bool(signals.get(s, True)) for s in SIGNALS},
            weights={s: float(weights.get(s, 1.0)) for s in SIGNALS},
            rrf_k=int(section.get("rrf_k", DEFAULT_RRF_K)),
            k=int(section.get("k", DEFAULT_K)),
            text_window_s=float(section.get("text_window_s", DEFAULT_TEXT_WINDOW_S)),
            candidates=int(section.get("candidates", DEFAULT_CANDIDATES)),
            min_sharpness=section.get("min_sharpness"),
            t_start=section.get("t_start"),
            t_end=section.get("t_end"),
            exclude_rejected=bool(section.get("exclude_rejected", True)),
        )
        if cfg.rrf_k < 1:
            raise SearchError(f"[search] rrf_k must be >= 1, got {cfg.rrf_k}")
        if cfg.k < 1:
            raise SearchError(f"[search] k must be >= 1, got {cfg.k}")
        if cfg.text_window_s < 0:
            raise SearchError(f"[search] text_window_s must be >= 0, got {cfg.text_window_s}")
        if cfg.candidates < 1:
            raise SearchError(f"[search] candidates must be >= 1, got {cfg.candidates}")
        return cfg


@dataclass(frozen=True)
class SearchHit:
    """One search result; score is the RRF sum, parts explain it."""

    file: str
    t: float
    score: float
    score_parts: dict[str, float]
    rank_parts: dict[str, int | None]
    scene_id: int | None = None
    sharpness: float | None = None
    query_used: str = ""
    translated_from: str | None = None  # the original query when translated

    def to_dict(self) -> dict:
        return {
            "file": self.file,
            "t": self.t,
            "score": self.score,
            "score_parts": dict(self.score_parts),
            "rank_parts": dict(self.rank_parts),
            "scene_id": self.scene_id,
            "sharpness": self.sharpness,
            "query_used": self.query_used,
            "translated_from": self.translated_from,
        }


@dataclass(frozen=True)
class _TextDoc:
    """A text of the run for the keyword corpus and time inheritance."""

    key: str  # "chunk-0" / "ocr-0" (artifact order, deterministic)
    t_start: float
    t_end: float
    text: str


def _load_json(path: Path, what: str) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SearchError(f"{what} is missing: {path} (run razkadr index first)") from exc
    except json.JSONDecodeError as exc:
        raise SearchError(f"{what} is not valid JSON: {path}: {exc}") from exc


def _texts_from_artifacts(run_dir: Path) -> list[_TextDoc]:
    """chunk + OCR texts in artifact order (the BM25 corpus)."""
    docs: list[_TextDoc] = []
    chunks = _load_json(run_dir / "chunks.json", "chunks").get("chunks") or []
    for i, chunk in enumerate(chunks):
        if chunk.get("text"):
            docs.append(
                _TextDoc(f"chunk-{i}", float(chunk["t_start"]), float(chunk["t_end"]), chunk["text"])
            )
    ocr = _load_json(run_dir / "ocr.json", "ocr").get("results") or []
    for i, result in enumerate(ocr):
        if result.get("text"):
            docs.append(_TextDoc(f"ocr-{i}", float(result["t"]), float(result["t"]), result["text"]))
    return docs


def _candidates_from_frames(run_dir: Path, cfg: SearchConfig) -> list[dict]:
    """frames.json candidates after the run-level filters (rejected out)."""
    frames = _load_json(run_dir / "frames.json", "frames").get("frames") or []
    out = []
    for frame in frames:
        if cfg.exclude_rejected and frame.get("rejected"):
            continue
        if cfg.min_sharpness is not None and (frame.get("sharpness") or 0.0) < cfg.min_sharpness:
            continue
        t = float(frame["t"])
        if cfg.t_start is not None and t < cfg.t_start:
            continue
        if cfg.t_end is not None and t > cfg.t_end:
            continue
        out.append(frame)
    return out


def _video_lang(run_dir: Path) -> str | None:
    path = run_dir / "transcript.json"
    if not path.is_file():
        return None
    try:
        return _load_json(path, "transcript").get("lang")
    except SearchError:
        return None


def _translate_query(
    query: str, query_lang: str, video_lang: str, llm
) -> str:
    """Translate the query to the video language through the LLM provider
    layer; an empty completion falls back to the original query."""
    if llm is None:
        raise SearchError(
            f"query language {query_lang!r} differs from the video language {video_lang!r}: "
            "translation needs an LLM provider (configure [llm] in razkadr.toml)"
        )
    from razkadr.providers.types import Message

    answer = llm.complete(
        [
            Message(role="system", content=TRANSLATE_PROMPT),
            Message(
                role="user",
                content=f"Запрос: {query}\nЯзык перевода: {video_lang}",
            ),
        ]
    ).strip()
    if not answer:
        logger.warning("translation returned an empty completion; using the original query")
        return query
    return answer


def bm25_scores(corpus: list[_TextDoc], query: str) -> list[float]:
    """BM25 (Okapi) of the query against the corpus; returns scores aligned
    with `corpus`. Tokenization: unicode word characters, lowercased."""
    tokenized = [[tok for tok in re.findall(r"\w+", doc.text.lower())] for doc in corpus]
    query_tokens = re.findall(r"\w+", query.lower())
    n = len(corpus)
    if not n or not query_tokens:
        return [0.0] * n
    avg_len = sum(map(len, tokenized)) / n or 1.0
    df: dict[str, int] = {}
    for tokens in tokenized:
        for token in set(tokens):
            df[token] = df.get(token, 0) + 1
    scores = []
    for tokens in tokenized:
        length = len(tokens) or 1
        tf: dict[str, int] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        score = 0.0
        for token in query_tokens:
            if token not in tf:
                continue
            idf = math.log((n - df[token] + 0.5) / (df[token] + 0.5) + 1.0)
            score += idf * (tf[token] * (BM25_K1 + 1.0)) / (
                tf[token] + BM25_K1 * (1.0 - BM25_B + BM25_B * length / avg_len)
            )
        scores.append(score)
    return scores


def _ranks(scores: list[float], keys: list[str]) -> list[tuple[int, str]]:
    """Ranks (1 = best) of scored items: score desc, then key asc; items
    with score <= 0 are ranked but naturally sink to the tail."""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], keys[i]))
    return [(rank + 1, keys[i]) for rank, i in enumerate(order)]


def _inherit_rank(
    frame_times: list[float], frame_files: list[str], docs: list[_TextDoc], ranks: dict[str, int], window: float
) -> dict[str, int]:
    """frame file -> the best rank among covering texts (None = no cover)."""
    best: dict[str, int] = {}
    for t, file in zip(frame_times, frame_files, strict=True):
        covering = [ranks[doc.key] for doc in docs if doc.t_start - window <= t <= doc.t_end + window and doc.key in ranks]
        if covering:
            best[file] = min(covering)
    return best


def hybrid_search(
    run_dir: str | Path,
    query: str,
    query_lang: str | None = None,
    embeddings=None,
    llm=None,
    config: dict | None = None,
) -> list[SearchHit]:
    """Run the hybrid search over a run's index; returns up to
    ``SearchConfig.k`` hits, best first. ``embeddings``/``llm`` are provider
    layer services (naryad 033) or any object speaking the same protocol;
    tests may pass scripted fakes."""
    cfg = SearchConfig.from_config(config or {})
    run = Path(run_dir)
    if not run.is_dir():
        raise SearchError(f"run directory not found: {run}")
    if not query or not query.strip():
        raise SearchError("query is empty")
    query = query.strip()

    frames = _candidates_from_frames(run, cfg)
    if not frames:
        return []

    video_lang = _video_lang(run)
    working_query = query
    translated_from: str | None = None
    if query_lang and video_lang and query_lang != video_lang:
        working_query = _translate_query(query, query_lang, video_lang, llm)
        translated_from = query

    needs_vectors = cfg.signals["image"] or cfg.signals["text"]
    if needs_vectors and embeddings is None:
        raise SearchError(
            "signals 'image'/'text' need an embeddings provider (configure [embeddings] in razkadr.toml "
            "or pass a service); disable them in [search] signals to search keywords only"
        )

    frame_files = [str(frame["file"]) for frame in frames]
    frame_times = [float(frame["t"]) for frame in frames]
    rank_parts: dict[str, dict[str, int | None]] = {file: {} for file in frame_files}
    for file in frame_files:
        for signal in SIGNALS:
            rank_parts[file][signal] = None

    def _open_store() -> LocalVectorStore:
        try:
            return LocalVectorStore(run / "index").open()
        except VectorStoreError as exc:
            raise SearchError(str(exc)) from exc

    if cfg.signals["image"] or cfg.signals["text"]:
        store = _open_store()
        try:
            query_vector = embeddings.embed_texts([working_query])[0]
            def _store_filters(spaces: tuple[str, ...], min_sharpness: float | None = None) -> Filters:
                kwargs = {"spaces": spaces, "t_start": cfg.t_start, "t_end": cfg.t_end}
                if min_sharpness is not None:
                    kwargs["min_sharpness"] = min_sharpness
                return Filters(**kwargs)

            if cfg.signals["image"]:
                results = store.search(query_vector, cfg.candidates, _store_filters(("frames",), cfg.min_sharpness))
                for rank, (_id, _score, meta) in enumerate(
                    sorted(results, key=lambda r: (-r[1], str(r[2].get("file")))), start=1
                ):
                    file = meta.get("file")
                    if file in rank_parts:
                        rank_parts[file]["image"] = rank
            if cfg.signals["text"]:
                results = store.search(query_vector, cfg.candidates, _store_filters(("chunks", "ocr")))
                text_docs: list[_TextDoc] = []
                text_ranks: dict[str, int] = {}
                for rank, (vid, _score, meta) in enumerate(
                    sorted(results, key=lambda r: (-r[1], r[0])), start=1
                ):
                    text = meta.get("text")
                    if not text:
                        continue
                    t = float(meta.get("t") or 0.0)
                    t_end = float(meta.get("t_end") if meta.get("t_end") is not None else t)
                    text_docs.append(_TextDoc(str(vid), t, t_end, str(text)))
                    text_ranks[str(vid)] = rank
                inherited = _inherit_rank(frame_times, frame_files, text_docs, text_ranks, cfg.text_window_s)
                for file, rank in inherited.items():
                    rank_parts[file]["text"] = rank
        finally:
            store.close()

    if cfg.signals["keyword"]:
        docs = _texts_from_artifacts(run)
        scores = bm25_scores(docs, working_query)
        # only texts the query actually matches (score > 0) join the ranking
        matched = [doc for doc, score in zip(docs, scores, strict=True) if score > 0.0]
        ranks: dict[str, int] = {}
        for rank, key in _ranks(
            [score for score in scores if score > 0.0],
            [doc.key for doc in matched],
        ):
            ranks[key] = rank
        inherited = _inherit_rank(frame_times, frame_files, matched, ranks, cfg.text_window_s)
        for file, rank in inherited.items():
            rank_parts[file]["keyword"] = rank

    hits: list[SearchHit] = []
    for frame, file in zip(frames, frame_files, strict=True):
        score = 0.0
        score_parts: dict[str, float] = {}
        for signal in SIGNALS:
            if not cfg.signals[signal]:
                score_parts[signal] = 0.0
                continue
            rank = rank_parts[file][signal]
            contribution = cfg.weights[signal] / (cfg.rrf_k + rank) if rank is not None else 0.0
            score_parts[signal] = contribution
            score += contribution
        hits.append(
            SearchHit(
                file=file,
                t=float(frame["t"]),
                score=score,
                score_parts=score_parts,
                rank_parts=dict(rank_parts[file]),
                scene_id=frame.get("scene_id"),
                sharpness=frame.get("sharpness"),
                query_used=working_query,
                translated_from=translated_from,
            )
        )
    hits.sort(key=lambda hit: (-hit.score, hit.file))
    return hits[: cfg.k]

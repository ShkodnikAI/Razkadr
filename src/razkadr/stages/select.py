"""Episode selection: preset + hybrid search -> draft Storyboard (naryad 014).

For every role of the preset (naryad 013) the best frames are picked from
the hybrid search results (naryad 012) and turned into episodes of a
Storyboard draft (naryad 002) without text:

- role queries (per language) run through ``hybrid_search``; a frame's role
  rank is the best rank across the role's queries;
- the final frame score fuses the search contributions with the frame
  quality metrics, weighted by the preset: embedding (vector search:
  image+text), keyword (BM25), sharpness and exposure (frames stage);
  everything is normalized to [0, 1] so ``score_parts`` sums to ``score``;
- diversity: a candidate too close in time or in pHash to an already
  selected frame is skipped, so steps do not repeat the same shot;
- episode bounds come from the frame's scene (frames.json scene ids) and
  the covering transcript chunk, clamped to the video duration;
- order: roles in preset order, episodes of a multi-episode role by time;
- re-selection: episodes marked ``user_edited`` in the existing storyboard
  are kept verbatim, their frames are not reused by other roles.

Constraints the data does not support yet (face/hand penalties from the
preset) are recorded as warnings, never silently ignored: see
``runs/<id>/select.json`` — the selection report with queries, ranks and
warnings per role.
"""

from __future__ import annotations

import json
from pathlib import Path

from razkadr.presets import load_preset
from razkadr.search import SIGNALS, SearchConfig, SearchError, hybrid_search
from razkadr.validation import validate_document

SELECT_JSON = "select.json"
STORYBOARD_JSON = "storyboard.json"

# [select] defaults: episode_half_s -- the half-width of an episode around
# the chosen frame; episode_min_s -- the shortest episode allowed;
# phash_min_hamming -- the pHash distance below which two frames count as
# the same shot (frames stage measured >= 8 between neighboring scenes);
# max_min_sharpness is not a select setting: it lives in the preset roles.
SELECT_DEFAULTS = {
    "episode_half_s": 2.0,
    "episode_min_s": 0.5,
    "phash_min_hamming": 8,
}
EPISODE_DEFAULT_HALF_FALLBACK_S = 0.25


class SelectError(Exception):
    """Episode selection cannot run (bad preset, missing artifacts)."""


def hamming(a: str, b: str) -> int:
    """Bitwise Hamming distance between two 64-bit hex pHash strings
    (the same metric the frames stage uses for duplicate detection)."""
    return (int(a, 16) ^ int(b, 16)).bit_count()


def select_params(config: dict) -> dict:
    """Effective [select] settings."""
    params = {**SELECT_DEFAULTS, **(config.get("select") or {})}
    if params["episode_half_s"] <= 0:
        raise SelectError(f"[select] episode_half_s must be > 0, got {params['episode_half_s']}")
    if params["episode_min_s"] <= 0:
        raise SelectError(f"[select] episode_min_s must be > 0, got {params['episode_min_s']}")
    if params["phash_min_hamming"] < 0:
        raise SelectError(f"[select] phash_min_hamming must be >= 0, got {params['phash_min_hamming']}")
    return params


def _load_json(path: Path, what: str, required: bool = True) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        if required:
            raise SelectError(f"{what} is missing: {path} (run razkadr index first)") from exc
        return None
    except json.JSONDecodeError as exc:
        raise SelectError(f"{what} is not valid JSON: {path}: {exc}") from exc


def _scene_spans(frames: list[dict], duration: float) -> dict[int, tuple[float, float]]:
    """Scene id -> (t_start, t_end) from frames.json scene ids: a scene
    starts at the last frame of the previous scene and ends where the next
    scene's first frame appears."""
    ordered = sorted(frames, key=lambda f: float(f["t"]))
    spans: dict[int, tuple[float, float]] = {}
    seen: set[int] = set()
    for i, frame in enumerate(ordered):
        scene = int(frame["scene_id"])
        if scene not in seen:
            start = float(ordered[i - 1]["t"]) if i else 0.0
            spans[scene] = (start, duration)
            seen.add(scene)
    last_scene = None
    for frame in ordered:
        scene = int(frame["scene_id"])
        if last_scene is not None and scene != last_scene:
            prev_start, _ = spans[last_scene]
            spans[last_scene] = (prev_start, float(frame["t"]))
        last_scene = scene
    return spans


def _covering_chunk(chunks: list[dict], t: float) -> dict | None:
    """The first chunk whose span contains t (artifact order)."""
    for chunk in chunks:
        if float(chunk.get("t_start", 0.0)) <= t <= float(chunk.get("t_end", 0.0)):
            return chunk
    return None


def _episode_span(
    t: float, scene_span: tuple[float, float], chunk: dict | None, duration: float, params: dict
) -> tuple[float, float]:
    """Episode bounds: scene ∩ chunk around the frame, at least min long."""
    half = params["episode_half_s"]
    lo = max(scene_span[0], t - half)
    hi = min(scene_span[1], t + half)
    if chunk is not None:
        lo = max(lo, float(chunk.get("t_start", lo)))
        hi = min(hi, float(chunk.get("t_end", hi)))
    if hi - lo < params["episode_min_s"]:
        half_min = params["episode_min_s"] / 2.0
        lo, hi = max(0.0, t - half_min), min(duration, t + half_min)
    return round(max(0.0, lo), 3), round(min(duration, hi), 3)


def _role_queries(spec: dict, video_lang: str | None) -> list[tuple[str, str]]:
    """(query, language) pairs for a role: the video language when the
    preset has it, else every preset language."""
    queries = spec["queries"]
    if video_lang in queries:
        return [(q, video_lang) for q in queries[video_lang]]
    pairs: list[tuple[str, str]] = []
    for lang in queries:
        pairs.extend((q, lang) for q in queries[lang])
    return pairs


def _max_search_score(cfg: SearchConfig) -> float:
    """Theoretical RRF maximum: every enabled signal at rank 1."""
    return sum(
        cfg.weights[s] / (cfg.rrf_k + 1) for s in SIGNALS if cfg.signals[s]
    )


def _role_candidates(
    run_dir: Path,
    spec: dict,
    role: str,
    queries: list[tuple[str, str]],
    embeddings,
    llm,
    params: dict,
    duration: float,
    report: dict,
) -> dict[str, dict]:
    """frame file -> candidate record for one role (best query wins)."""
    search_cfg = SearchConfig(
        k=int(spec.get("candidates", 20)),
        min_sharpness=(spec.get("constraints") or {}).get("min_sharpness"),
        t_start=0.0,
        t_end=duration or None,
    )
    max_score = _max_search_score(search_cfg)
    candidates: dict[str, dict] = {}
    for query, lang in queries:
        try:
            hits = hybrid_search(
                run_dir, query, query_lang=lang, embeddings=embeddings, llm=llm,
                config={"search": search_cfg.__dict__ | {"signals": dict(search_cfg.signals)}},
            )
        except SearchError as exc:
            raise SelectError(f"role '{role}', query {query!r}: {exc}") from exc
        report["queries"].append({"role": role, "query": query, "lang": lang, "hits": len(hits)})
        for hit in hits:
            if hit.file in candidates:
                continue  # a frame keeps its best query result (first list wins on ties)
            search_norm = hit.score / max_score if max_score else 0.0
            parts = {signal: value / max_score for signal, value in hit.score_parts.items()}
            candidates[hit.file] = {
                "file": hit.file,
                "t": hit.t,
                "scene_id": hit.scene_id,
                "sharpness": hit.sharpness,
                "search_norm": search_norm,
                "search_parts": parts,
                "rank_parts": hit.rank_parts,
                "query": query,
                "query_lang": lang,
            }
    return candidates


def _final_score(candidate: dict, spec: dict) -> tuple[float, dict[str, float]]:
    """Preset-weighted score in [0, 1]; score_parts sums to the score."""
    weights = spec["weights"]
    parts_search = candidate["search_parts"]
    embedding_norm = parts_search.get("image", 0.0) + parts_search.get("text", 0.0)
    keyword_norm = parts_search.get("keyword", 0.0)
    sharpness = candidate.get("sharpness")
    exposure = candidate.get("exposure")
    parts = {
        "embedding": weights.get("embedding", 0.0) * embedding_norm,
        "keyword": weights.get("keyword", 0.0) * keyword_norm,
        "sharpness": weights.get("sharpness", 0.0) * (sharpness if sharpness is not None else 0.0),
        "exposure": weights.get("exposure", 0.0) * (exposure if exposure is not None else 0.0),
    }
    score = sum(parts.values())
    return min(score, 1.0), parts


def _too_close(a: dict, b: dict, params: dict, min_gap_s: float) -> bool:
    """Same shot: pHash near-identical or time-gapped (per preset)."""
    if min_gap_s and abs(float(a["t"]) - float(b["t"])) < min_gap_s:
        return True
    a_hash, b_hash = a.get("phash"), b.get("phash")
    return bool(a_hash and b_hash and hamming(str(a_hash), str(b_hash)) < params["phash_min_hamming"])


def _pick_diverse(
    ranked: list[dict], spec: dict, min_gap_s: float, params: dict, quota: int | None
) -> list[dict]:
    """Greedy diverse pick: best score first, close shots skipped."""
    limit = quota if quota is not None else int(spec.get("max") or 10**9)
    selected: list[dict] = []
    for candidate in ranked:
        if len(selected) >= limit:
            break
        if any(_too_close(candidate, chosen, params, min_gap_s) for chosen in selected):
            continue
        selected.append(candidate)
    return selected


def run_select(
    run_dir: str | Path,
    preset_path: str | Path,
    config: dict | None = None,
    embeddings=None,
    llm=None,
) -> tuple[dict, dict]:
    """Build the draft Storyboard for a run; returns (storyboard, report).

    Writes nothing: the caller (CLI) saves storyboard.json (+ select.json).
    """
    config = dict(config or {})
    params = select_params(config)
    run = Path(run_dir)
    if not run.is_dir():
        raise SelectError(f"run directory not found: {run}")
    preset = load_preset(preset_path)

    frames_doc = _load_json(run / "frames.json", "frames")
    frames = list(frames_doc.get("frames") or [])
    if not frames:
        raise SelectError(f"frames.json has no frames: {run / 'frames.json'}")
    source_doc = _load_json(run / "source.json", "source", required=False) or {}
    duration = float(source_doc.get("duration_s") or max(float(f["t"]) for f in frames))
    chunks_doc = _load_json(run / "chunks.json", "chunks", required=False) or {}
    chunks = list(chunks_doc.get("chunks") or [])
    transcript_doc = _load_json(run / "transcript.json", "transcript", required=False) or {}
    video_lang = transcript_doc.get("lang")
    scene_spans = _scene_spans(frames, duration)
    frames_by_file = {str(f["file"]): f for f in frames}

    report: dict = {"preset": preset.get("name"), "queries": [], "roles": {}, "warnings": []}

    # user-edited episodes survive re-selection; their frames stay reserved
    existing = _load_json(run / STORYBOARD_JSON, "storyboard", required=False)
    kept: list[dict] = []
    reserved_files: set[str] = set()
    if existing:
        for ep in existing.get("episodes") or []:
            if ep.get("user_edited"):
                kept.append(ep)
                reserved_files.add(str((ep.get("frame") or {}).get("path")))

    episodes: list[dict] = []
    # frames chosen for earlier roles are not re-selectable by later roles
    taken_files: set[str] = set(reserved_files)
    for role in preset["order"]:
        spec = preset["roles"][role]
        if spec["min"] == 0 and spec["max"] == 0:
            report["warnings"].append(f"role '{role}': quota is 0, skipped")
            continue
        kept_in_role = [ep for ep in kept if ep.get("role") == role]
        quota_left = spec["max"] - len(kept_in_role) if spec["max"] is not None else None
        if quota_left is not None and quota_left <= 0:
            report["warnings"].append(
                f"role '{role}': all {len(kept_in_role)} episode(s) are user-edited, nothing to select"
            )
            continue
        queries = _role_queries(spec, video_lang)
        if not queries:
            report["warnings"].append(f"role '{role}': no queries for language {video_lang!r}")
            continue
        candidates = _role_candidates(
            run, spec, role, queries, embeddings, llm, params, duration, report
        )
        # user-reserved frames and frames taken by earlier roles are out
        candidates = {f: c for f, c in candidates.items() if f not in taken_files}
        for file, candidate in candidates.items():
            frame = frames_by_file.get(file)
            if frame is not None:
                candidate["phash"] = frame.get("phash")
                candidate["exposure"] = frame.get("exposure")
                candidate["scene_id"] = frame.get("scene_id", candidate["scene_id"])
        ranked: list[dict] = []
        for candidate in candidates.values():
            score, parts = _final_score(candidate, spec)
            candidate["score"], candidate["score_parts"] = score, parts
            ranked.append(candidate)
        ranked.sort(key=lambda c: (-c["score"], c["file"]))
        # the preset gap rule protects repeated roles from clustering
        min_gap_s = float(preset.get("min_step_gap_s") or 0.0)
        if int(spec.get("max") or 2) <= 1:
            min_gap_s = 0.0  # single-episode roles need no gap
        picked = _pick_diverse(ranked, spec, min_gap_s, params, quota_left)
        if len(picked) < int(spec["min"]):
            report["warnings"].append(
                f"role '{role}': {len(picked)} episode(s) selected, preset minimum is {spec['min']} "
                "(short video or weak matches)"
            )
        taken_files.update(c["file"] for c in picked)
        picked.sort(key=lambda c: float(c["t"]))
        role_episodes: list[dict] = []
        for i, candidate in enumerate(picked, start=len(kept_in_role) + 1):
            frame = frames_by_file[candidate["file"]]
            scene_span = scene_spans.get(int(frame["scene_id"]), (0.0, duration))
            chunk = _covering_chunk(chunks, float(frame["t"]))
            t_start, t_end = _episode_span(float(frame["t"]), scene_span, chunk, duration, params)
            role_episodes.append(
                {
                    "id": f"ep-{role}-{i}" if len(picked) + len(kept_in_role) > 1 else f"ep-{role}",
                    "role": role,
                    "source_id": "src",
                    "t_start": t_start,
                    "t_end": t_end,
                    "frame": {
                        "path": candidate["file"],
                        "t": float(frame["t"]),
                        "score": round(candidate["score"], 6),
                        "score_parts": {k: round(v, 6) for k, v in candidate["score_parts"].items()},
                    },
                    "alternatives": [],
                }
            )
            report["roles"].setdefault(role, []).append(
                {
                    "file": candidate["file"],
                    "t": candidate["t"],
                    "score": round(candidate["score"], 6),
                    "score_parts": {k: round(v, 6) for k, v in candidate["score_parts"].items()},
                    "rank_parts": candidate["rank_parts"],
                    "query": candidate["query"],
                    "query_lang": candidate["query_lang"],
                }
            )
        # alternatives: next-best distinct frames that are not selected shots
        chosen_files = {ep["frame"]["path"] for ep in role_episodes}
        chosen_ranked = list(picked)
        alt_pool = [
            c for c in ranked
            if c["file"] not in chosen_files
            and not any(_too_close(c, chosen, params, min_gap_s) for chosen in chosen_ranked)
        ]
        want = int(spec.get("alternatives", 3))
        for ep in role_episodes:
            alts: list[dict] = []
            used_paths: set[str] = set()
            used_hashes: set[str] = set()
            for candidate in alt_pool:
                if len(alts) >= want:
                    break
                file = candidate["file"]
                if file in used_paths:
                    continue
                cand_hash = str(candidate.get("phash") or "")
                if cand_hash and any(
                    hamming(cand_hash, other) < params["phash_min_hamming"]
                    for other in used_hashes if other
                ):
                    continue
                used_paths.add(file)
                if cand_hash:
                    used_hashes.add(cand_hash)
                alts.append(
                    {
                        "path": file,
                        "t": float(candidate["t"]),
                        "score": round(candidate["score"], 6),
                    }
                )
            ep["alternatives"] = alts
        episodes.extend(role_episodes)

    for ep in kept:
        role = ep.get("role")
        if role in preset["order"]:
            episodes.append(ep)
        else:
            report["warnings"].append(
                f"user-edited episode {ep.get('id')!r} has role {role!r} outside the preset; kept anyway"
            )
            episodes.append(ep)

    episodes.sort(key=lambda e: (preset["order"].index(e["role"]), float(e["frame"]["t"])))
    storyboard = {
        "schema_version": "1.0",
        "project": {
            "id": f"run-{run.name}",
            "title": source_doc.get("path") or run.name,
            "preset": preset["name"],
            "output_lang": video_lang or "ru",
        },
        "sources": [
            {
                "id": "src",
                "path": source_doc.get("path") or "normalized.mp4",
                "url": source_doc.get("url"),
                "duration_s": duration,
                "detected_lang": video_lang,
            }
        ],
        "episodes": episodes,
    }
    storyboard["sources"][0] = {
        key: value for key, value in storyboard["sources"][0].items() if value is not None
    }
    issues = validate_document(storyboard, "storyboard")
    if issues:
        raise SelectError("selection produced an invalid storyboard: " + "; ".join(str(i) for i in issues))
    report["warnings"] = list(dict.fromkeys(report["warnings"]))
    return storyboard, report

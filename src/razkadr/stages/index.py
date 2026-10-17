"""Index stage (naryad 010): frames.json + chunks.json + ocr.json -> index/.

Embeddings through the naryad 033 provider layer; the stage knows only
the ``EmbeddingProvider`` protocol (``embed_images`` / ``embed_texts``)
behind the layer's ``EmbeddingsService`` - the implementation is chosen
by ``[embeddings]`` in razkadr.toml and never appears in stage code.
Vectors are stored through the ``VectorStore`` interface
(vectorstore.py): the core implementation is SQLite + numpy with exact
cosine search (ADR-0007, proposed); pgvector is deferred to the web wave.

What is embedded (three provenance spaces in one store):
- ``frames``: every ACCEPTED frame of frames.json - rejected frames are
  never sent to the provider (they are duplicates or blur, embedding
  them would burn money on noise; acceptance criterion checks this by
  the provider call counter);
- ``chunks``: chunk texts of chunks.json (thematic transcript chunks);
- ``ocr``: recognized screen texts of ocr.json.

Calls are batched (``[embeddings] batch_size``) so one API request
carries several inputs; every batch is cached by the provider layer
(sha256 of its inputs), so an interrupted stage re-runs pay only for
the batches that never completed. Batch boundaries follow the input
order, so unchanged prefixes hit the cache.

Rebuild semantics: the stage always rebuilds ``index/`` from scratch
(dir wiped before a run) - partial stores from a failed attempt are
never reused, consistency comes from the run being atomic at stage
level while provider-level caching keeps the re-run cheap. A provider,
model, model_version or batch_size change alters the stage cache key,
so the stage re-runs and rebuilds; when an existing manifest names a
different provider/model/version, the rebuild is logged as
"index is stale" (the acceptance criterion "smena modeli pomechaet
indeks ustarevshim"). A dim mismatch between the provider output and
an existing store is impossible after the wipe - and the store itself
rejects mismatched vectors defensively.

Manifest: ``index/manifest.json`` validates against
schemas/index.v1.json (naryad 035): provider, model, version, dim,
num_vectors and the vectors_file path relative to the run directory.
``space`` is left unset: the store deliberately mixes the three spaces
(one provider/model per run), per-space vectors are distinguished by
the metadata ``space`` column and the ``Filters.spaces`` search filter.

Dim when there is nothing to embed: frames.json may hold zero accepted
frames, chunks/ocr may be empty (silent video, no captions). The
manifest still needs a positive dim, so the stage makes one probe call
(``embed_texts([DIM_PROBE_TEXT])``, cached like any other call) and
records the returned dimensionality without storing the vector.

Skip semantics: with no ``[embeddings]`` provider configured the stage
is skipped with a warning and no ``index/`` artifacts (same rule as ocr
without a VLM and chunks without an LLM: an unconfigured provider keeps
the pipeline usable; the CLI summary prints "index skipped"). A
CONFIGURED-but-broken provider (unknown name, missing key, auth/quota)
fails the stage loudly - explicit configuration must not degrade
silently.

Version pinning: the manifest ``version`` is the provider-reported
``version`` attribute when the adapter provides one, else the
``[embeddings] model_version`` config value, else ``unknown`` - fake
providers report none, real adapters (naryad 038) will.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from razkadr.atomicio import write_json_atomic
from razkadr.log import get_logger
from razkadr.providers.base import ProviderConfigError
from razkadr.providers.layer import ProviderLayer
from razkadr.runctx import RunContext, file_sha256
from razkadr.validation import validate_document
from razkadr.vectorstore import SPACE_KINDS, LocalVectorStore, VectorStoreError

logger = get_logger("index")

INDEX_DIRNAME = "index"
MANIFEST_JSON = "manifest.json"
VECTORS_BIN = "vectors.bin"
FRAMES_JSON = "frames.json"
CHUNKS_JSON = "chunks.json"
OCR_JSON = "ocr.json"
SOURCE_JSON = "source.json"

INDEX_DEFAULTS: dict = {
    "batch_size": 16,  # inputs per provider call; 1 = one call per input
}

# Probe text for the dim of an all-empty index; never stored as a vector.
DIM_PROBE_TEXT = "razkadr dim probe"


class IndexStageError(Exception):
    """Index stage failure with a clear, user-facing message (CLI exit 2)."""


def index_params(config: dict) -> dict:
    """Effective [embeddings] stage settings; goes into the stage cache key."""
    section = {**INDEX_DEFAULTS, **(config.get("embeddings") or {})}
    batch_size = int(section["batch_size"])
    if batch_size < 1:
        raise IndexStageError(f"[embeddings] batch_size must be >= 1, got {batch_size}")
    return {"batch_size": batch_size}


def _load_json(path: Path, kind: str) -> dict:
    """Read a stage input artifact; missing or invalid input is a stage error."""
    if not path.is_file():
        raise IndexStageError(f"index stage needs the {kind} stage output: {path.name} is missing in the run directory")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise IndexStageError(f"{path.name} is not valid JSON: {exc}") from None
    if validate_document(data, kind):
        raise IndexStageError(f"{path.name} is not a valid {kind} artifact (re-run the {kind} stage)")
    return data


def _source_identity(source_doc: dict | None) -> str | None:
    """What the video came from: URL when present, else the local path."""
    if not source_doc:
        return None
    return source_doc.get("url") or source_doc.get("path")


def _batches(items: list, batch_size: int) -> list[list]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def run_index(ctx: RunContext, config: dict) -> tuple[dict | None, dict[str, int]]:
    """Index stage body; returns (manifest, per-space vector counts).

    manifest is None when the stage is skipped (no ``[embeddings]``
    provider configured). Cache inputs: sha256 of the three input
    artifacts plus source.json (its identity lands in the metadata).
    Params carry the [embeddings] settings and the provider/model/
    model_version, so a provider or config switch re-runs the stage.
    """
    params = index_params(config)

    frames_doc = _load_json(ctx.dir / FRAMES_JSON, "frames")
    chunks_doc = _load_json(ctx.dir / CHUNKS_JSON, "chunks")
    ocr_doc = _load_json(ctx.dir / OCR_JSON, "ocr")
    source_path = ctx.dir / SOURCE_JSON
    source_doc = None
    if source_path.is_file():
        source_doc = _load_json(source_path, "source")
    source = _source_identity(source_doc)

    accepted = [f for f in frames_doc.get("frames", []) if not f.get("rejected")]
    chunks = list(chunks_doc.get("chunks", []))
    ocr_results = list(ocr_doc.get("results", []))

    embeddings_section = config.get("embeddings") or {}
    stage_params = {
        **params,
        "embeddings_provider": embeddings_section.get("provider"),
        "embeddings_model": embeddings_section.get("model"),
        "embeddings_model_version": embeddings_section.get("model_version"),
    }

    with ctx.stage(
        "index",
        inputs={
            "frames_sha256": file_sha256(ctx.dir / FRAMES_JSON),
            "chunks_sha256": file_sha256(ctx.dir / CHUNKS_JSON),
            "ocr_sha256": file_sha256(ctx.dir / OCR_JSON),
            **({"source_sha256": file_sha256(source_path)} if source_doc else {}),
        },
        params=stage_params,
    ) as st:
        manifest_path = ctx.dir / INDEX_DIRNAME / MANIFEST_JSON
        if st.cached:
            # the skip case leaves no manifest behind
            if not manifest_path.is_file():
                return None, {}
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            return manifest, _space_counts(ctx.dir)

        if not embeddings_section.get("provider"):
            logger.warning(
                "index skipped",
                extra={"fields": {"reason": "no [embeddings] provider configured"}},
            )
            st.outputs = {"skipped": True}
            return None, {}

        # Staleness: an existing manifest from another provider/model/version
        # means the vectors on disk do not match the current configuration.
        old_manifest = None
        if manifest_path.is_file():
            try:
                old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                old_manifest = None

        # Rebuild from scratch: no partial stores, no stale artifacts.
        index_dir = ctx.dir / INDEX_DIRNAME
        if index_dir.exists():
            shutil.rmtree(index_dir)

        try:
            layer = ProviderLayer(config, ctx)
            service = layer.embeddings(stage="index")
        except ProviderConfigError as exc:
            raise IndexStageError(f"{exc} (the [embeddings] section is configured; see docs/config.md)") from None

        provider = service.provider
        model_version = getattr(provider, "version", None) or embeddings_section.get("model_version")
        version = str(model_version) if model_version else "unknown"

        if old_manifest and (
            old_manifest.get("provider") != provider.name
            or old_manifest.get("model") != provider.model
            or old_manifest.get("version") != version
        ):
            logger.warning(
                "index is stale, rebuilding",
                extra={
                    "fields": {
                        "reason": "provider/model/version changed",
                        "old": f"{old_manifest.get('provider')}/{old_manifest.get('model')}/{old_manifest.get('version')}",
                        "new": f"{provider.name}/{provider.model}/{version}",
                    }
                },
            )

        store = LocalVectorStore(index_dir)
        dim: int | None = None
        counts: dict[str, int] = {}
        try:
            # -- frames (images) -----------------------------------------
            for batch in _batches(accepted, params["batch_size"]):
                paths = [ctx.dir / f["file"] for f in batch]
                vectors = service.embed_images(paths)
                if dim is None:
                    dim = len(vectors[0])
                    store.create(dim)
                _check_dims(vectors, dim)
                store.add(
                    "frames",
                    [f"frames:{f['file']}" for f in batch],
                    vectors,
                    [
                        {
                            "file": f["file"],
                            "t": f["t"],
                            "t_end": f["t"],  # frames span the instant t..t (Filters overlap)
                            "scene_id": f["scene_id"],
                            "sharpness": f["sharpness"],
                            "source": source,
                        }
                        for f in batch
                    ],
                )

            # -- chunks (texts) ------------------------------------------
            for batch in _batches(chunks, params["batch_size"]):
                vectors = service.embed_texts([c["text"] for c in batch])
                if dim is None:
                    dim = len(vectors[0])
                    store.create(dim)
                _check_dims(vectors, dim)
                store.add(
                    "chunks",
                    [f"chunks:{c['id']}" for c in batch],
                    vectors,
                    [
                        {
                            "t": c["t_start"],
                            "t_end": c["t_end"],
                            "lang": c.get("lang"),
                            "text": c["text"],
                            "source": source,
                        }
                        for c in batch
                    ],
                )

            # -- ocr (texts) ---------------------------------------------
            for batch in _batches(ocr_results, params["batch_size"]):
                vectors = service.embed_texts([r["text"] for r in batch])
                if dim is None:
                    dim = len(vectors[0])
                    store.create(dim)
                _check_dims(vectors, dim)
                store.add(
                    "ocr",
                    [f"ocr:{r['frame']}" for r in batch],
                    vectors,
                    [
                        {
                            "file": r["frame"],
                            "t": r["t"],
                            "t_end": r["t"],  # ocr results span the instant t..t
                            "lang": r.get("lang"),
                            "text": r["text"],
                            "source": source,
                        }
                        for r in batch
                    ],
                )

            # -- all-empty: probe the dim, store stays empty -------------
            if dim is None:
                probe = service.embed_texts([DIM_PROBE_TEXT])
                dim = len(probe[0])
                store.create(dim)

            for space in SPACE_KINDS:
                counts[space] = store.count(space)
            store.set_meta("provider", provider.name)
            store.set_meta("model", provider.model)
            store.set_meta("version", version)
            store.save()
        except VectorStoreError as exc:
            raise IndexStageError(str(exc)) from None
        finally:
            store.close()

        manifest = {
            "schema_version": "1.0",
            "provider": provider.name,
            "model": provider.model,
            "version": version,
            "dim": dim,
            "num_vectors": sum(counts.values()),
            "vectors_file": f"{INDEX_DIRNAME}/{VECTORS_BIN}",
        }
        issues = validate_document(manifest, "index")
        if issues:
            raise IndexStageError("manifest.json failed schema validation: " + "; ".join(issues))
        write_json_atomic(index_dir / MANIFEST_JSON, manifest)
        st.outputs = {
            "dir": INDEX_DIRNAME,
            "num_vectors": manifest["num_vectors"],
            "dim": dim,
            **{f"vectors_{space}": counts[space] for space in SPACE_KINDS},
        }
        logger.info(
            "index done",
            extra={
                "fields": {
                    "num_vectors": manifest["num_vectors"],
                    "dim": dim,
                    "frames": counts["frames"],
                    "chunks": counts["chunks"],
                    "ocr": counts["ocr"],
                }
            },
        )
        return manifest, counts


def _check_dims(vectors: list[list[float]], dim: int) -> None:
    """Provider returned a wrong dim mid-run: loud failure, not a broken store."""
    bad = next((v for v in vectors if len(v) != dim), None)
    if bad is not None:
        raise IndexStageError(
            f"embedding provider returned inconsistent dim {len(bad)} (expected {dim}); "
            "check [embeddings] model and model_version"
        )


def _space_counts(run_dir: Path) -> dict[str, int]:
    """Per-space counts of an already-built index (cached stage path)."""
    try:
        with LocalVectorStore(run_dir / INDEX_DIRNAME) as store:
            return {space: store.count(space) for space in SPACE_KINDS}
    except VectorStoreError:
        return {}

"""OCR stage (naryad 009): frames.json + frames/*.jpg -> ocr.json.

Screen text (ingredient lists, step captions, timers) is recognized by
the VLM through the naryad 033 provider layer; the stage knows only the
VLMProvider protocol (``describe_image(path, prompt)``). OCR does not
run on every frame - that would burn provider money on empty shots -
so a cheap CPU filter picks the candidates first.

Candidate filter (all CPU, no provider calls): every accepted (not
``rejected``) frame of frames.json is scored with an edge density - the
share of Canny edge pixels of the grayscale frame. Frames with density
>= ``[ocr] edge_density_threshold`` are "probably has text" candidates;
in addition every scene contributes its key frame - the sharpest
accepted frame of the scene (tie: earliest timestamp), derived from the
sharpness metric frames.json already carries, so the frames.json schema
stays untouched. Candidates are ordered dense-first (density desc, then
t asc) and capped at ``[ocr] max_ocr_frames``: the cap bounds provider
spend, so the frames most likely to carry text are sent first. The
measured fixture separation (naryad 034 titles.mp4, ffmpeg 7.1.5):
blank frames 0.0000, caption frames 0.0150-0.0178 -> default threshold
0.005 keeps a >= 3x margin; the threshold is revisited on the reference
set (gate B).

Recognition: one VLM call per candidate frame with the versioned prompt
``prompts/ocr.v1.md`` (asked to return only the on-screen text, line by
line, empty answer when there is none). The answer is never trusted
beyond whitespace cleanup: lines are stripped, empty lines dropped.
Empty answers produce no result. The provider layer caches every call
by the frame file hash + prompt and bills it via ``[vlm.tariff]`` into
run.json, so a re-run never pays twice for the same frame.

Collapsing: identical text on consecutive OCR-processed frames (time
order, whitespace-normalized comparison, case-sensitive) is collapsed
into a single result that keeps the FIRST frame of the run - ``t`` is
the interval start, and the interval runs until the next result's ``t``
(or until the last processed frame). ocr.v1.json has a single ``t`` per
result, so the interval end is not stored explicitly; keeping the first
frame preserves the interval start, dropping it would shift the text in
time (naryad 009 auditor check). A schema extension with explicit
t_start/t_end is out of scope (PR Assumptions).

Language: results carry ``[ocr] lang`` when set (BCP-47) or ``und`` -
the VLM is not asked for the language, real detection arrives with the
adapters (naryad 039) / reference set.

Failure modes: no candidates (zero limit, no accepted frames or nothing
passes the filter) yields a valid empty ocr.json without touching the
provider layer - no cost. With candidates but no ``[vlm]`` provider
configured at all the stage is skipped the same way: a valid empty
ocr.json is written and the reason is logged as a warning (the pipeline
stays usable without a VLM, same rule as the chunks fallback). A
configured-but-broken ``[vlm]`` (unknown provider name, missing key,
auth/quota failure) fails the stage loudly: explicit configuration must
not degrade silently. There is no deterministic OCR fallback: a bad VLM
answer would silently invent or hide screen text, so provider errors
propagate.

Cache: the stage cache key is the sha256 of frames.json plus the [ocr]
params, the prompt sha256 and the [vlm] provider/model/model_version,
so a provider, prompt or config switch re-runs the stage.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from razkadr.atomicio import write_json_atomic
from razkadr.log import get_logger
from razkadr.providers.base import ProviderConfigError
from razkadr.providers.layer import ProviderLayer
from razkadr.runctx import RunContext, file_sha256, read_cached_json
from razkadr.validation import validate_document

logger = get_logger("ocr")

OCR_JSON = "ocr.json"
FRAMES_JSON = "frames.json"
PROMPT_NAME = "ocr.v1.md"

OCR_DEFAULTS: dict = {
    "max_ocr_frames": 60,  # cap on provider calls; 0 disables OCR (empty result)
    "edge_density_threshold": 0.005,  # share of Canny edge pixels; fixture margins in the docstring
    "lang": "",  # force the result language (BCP-47); empty = "und"
}

# Same BCP-47 shape as the transcript stage and the ocr.v1.json schema.
LANG_RE = re.compile(r"^[A-Za-z]{2,8}(-[A-Za-z0-9]{1,8})*$")


class OcrError(Exception):
    """OCR stage failure with a clear, user-facing message (CLI exit 2)."""


def ocr_params(config: dict) -> dict:
    """Effective [ocr] section with defaults; goes into the stage cache key."""
    section = {**OCR_DEFAULTS, **(config.get("ocr") or {})}
    max_ocr_frames = int(section["max_ocr_frames"])
    threshold = float(section["edge_density_threshold"])
    lang = str(section["lang"]).strip()
    if max_ocr_frames < 0:
        raise OcrError(f"[ocr] max_ocr_frames must be >= 0, got {max_ocr_frames}")
    if not 0.0 <= threshold <= 1.0:
        raise OcrError(f"[ocr] edge_density_threshold must be within [0, 1], got {threshold}")
    if lang and not LANG_RE.match(lang):
        raise OcrError(f"[ocr] lang must be a BCP-47 tag (e.g. ru, en-US), got {lang!r}")
    return {"max_ocr_frames": max_ocr_frames, "edge_density_threshold": threshold, "lang": lang}


def load_ocr_prompt() -> str:
    """Versioned prompt packaged with the library (naryad 009)."""
    from importlib import resources

    return resources.files("razkadr.prompts").joinpath(PROMPT_NAME).read_text(encoding="utf-8")


def prompt_sha256() -> str:
    """Prompt digest: a prompt change invalidates the stage cache key."""
    return hashlib.sha256(load_ocr_prompt().encode("utf-8")).hexdigest()


def edge_density(path: Path) -> float:
    """Share of Canny edge pixels of the grayscale frame, 0..1 (rounded)."""
    import cv2

    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise OcrError(f"cannot read frame file: {Path(path).name} (re-run the frames stage)")
    edges = cv2.Canny(gray, 100, 200)
    return round(float((edges > 0).sum()) / edges.size, 6)


def scene_key_frames(accepted: list[dict]) -> set[str]:
    """Per scene: the sharpest accepted frame (tie: earliest t) is the key frame.

    frames.json does not mark scene middles (naryad 006 schema is frozen),
    so the scene key frame is derived from the recorded sharpness metric
    instead of re-running scene detection.
    """
    best: dict[int, dict] = {}
    for entry in accepted:
        scene = entry["scene_id"]
        current = best.get(scene)
        if current is None or (entry["sharpness"], -entry["t"]) > (
            current["sharpness"],
            -current["t"],
        ):
            best[scene] = entry
    return {entry["file"] for entry in best.values()}


def plan_ocr_candidates(frames_doc: dict, run_dir: Path, params: dict) -> list[dict]:
    """Cheap CPU filter: which frames go to the VLM (dense-first, capped).

    Returns [{file, t, density}] in call order; empty when nothing should
    be sent (no accepted frames, nothing passes the filter or the limit
    is 0) - the caller then skips the provider layer entirely.
    """
    accepted = [f for f in frames_doc.get("frames", []) if not f.get("rejected")]
    if not accepted or params["max_ocr_frames"] == 0:
        return []
    keys = scene_key_frames(accepted)
    scored = []
    for entry in accepted:
        density = edge_density(run_dir / entry["file"])
        scored.append({"file": entry["file"], "t": entry["t"], "density": density})
    candidates = [
        s for s in scored if s["density"] >= params["edge_density_threshold"] or s["file"] in keys
    ]
    candidates.sort(key=lambda s: (-s["density"], s["t"]))
    return candidates[: params["max_ocr_frames"]]


def parse_ocr_answer(raw: str) -> str:
    """VLM answer -> screen text: stripped non-empty lines joined by newline."""
    lines = [line.strip() for line in (raw or "").splitlines()]
    return "\n".join(line for line in lines if line)


def collapse_runs(sent: list[dict], texts: dict[str, str], lang: str) -> list[dict]:
    """Collapse identical text on consecutive sent frames (time order).

    ``sent`` is the candidate list (any order); grouping runs in time
    order over frames with non-empty recognized text. A run keeps its
    first frame: ``t`` is the interval start, the end is bounded by the
    next result (module docstring). Comparison is whitespace-normalized
    and case-sensitive: never merge different text, only re-flowed same
    text.
    """
    results: list[dict] = []
    prev_norm: str | None = None
    for cand in sorted(sent, key=lambda s: (s["t"], s["file"])):
        text = texts.get(cand["file"], "")
        if not text:
            prev_norm = None  # a gap of empty frames breaks the run
            continue
        norm = " ".join(text.split())
        if norm == prev_norm:
            continue  # same text on the next frame: the run keeps its first frame
        results.append({"frame": cand["file"], "t": cand["t"], "text": text, "lang": lang or "und"})
        prev_norm = norm
    return results


def run_ocr(ctx: RunContext, config: dict) -> dict:
    """OCR stage body; returns the ocr.json content (schema-checked).

    Cache inputs: sha256 of frames.json (the frames stage writes the
    jpgs before frames.json, so the artifact covers its inputs). Params
    carry the [ocr] settings, the prompt sha256 and the [vlm] provider/
    model/model_version, so a provider, prompt or config switch
    invalidates the stage. The provider implementation is chosen by
    config (FACTORIES), the stage only knows the VLMProvider protocol
    (same split as transcript/ASR and chunks/LLM).
    """
    params = ocr_params(config)

    frames_path = ctx.dir / FRAMES_JSON
    if not frames_path.is_file():
        raise OcrError(
            "ocr stage needs the frames stage output: frames.json is missing in the run directory"
        )
    try:
        frames_doc = json.loads(frames_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise OcrError(f"frames.json is not valid JSON: {exc}") from None
    if validate_document(frames_doc, "frames"):
        raise OcrError("frames.json is not a valid frames artifact (re-run the frames stage)")

    vlm_section = config.get("vlm") or {}
    stage_params = {
        **params,
        "prompt_sha256": prompt_sha256(),
        "vlm_provider": vlm_section.get("provider"),
        "vlm_model": vlm_section.get("model"),
        "vlm_model_version": vlm_section.get("model_version"),
    }

    with ctx.stage(
        "ocr", inputs={"frames_sha256": file_sha256(frames_path)}, params=stage_params
    ) as st:
        if st.cached:
            return read_cached_json(ctx.dir / OCR_JSON, "ocr.json", OcrError)

        accepted = [f for f in frames_doc.get("frames", []) if not f.get("rejected")]
        results: list[dict] = []
        sent: list[dict] = []
        if accepted:
            sent = plan_ocr_candidates(frames_doc, ctx.dir, params)
        if sent:
            skip_reason: str | None = None
            if not vlm_section.get("provider"):
                skip_reason = "no [vlm] provider configured"
            else:
                try:
                    service = ProviderLayer(config, ctx).vlm(stage="ocr")
                except ProviderConfigError as exc:
                    raise OcrError(
                        f"{exc} (the [vlm] section is configured; see docs/config.md)"
                    ) from None
                prompt = load_ocr_prompt()
                texts = {
                    cand["file"]: parse_ocr_answer(
                        service.describe_image(ctx.dir / cand["file"], prompt)
                    )
                    for cand in sent
                }
                results = collapse_runs(sent, texts, params["lang"])
            if skip_reason:
                logger.warning("ocr skipped", extra={"fields": {"reason": skip_reason}})
                sent, results = [], []

        data = {"schema_version": "1.0", "results": results}
        issues = validate_document(data, "ocr")
        if issues:
            raise OcrError("ocr.json failed schema validation: " + "; ".join(issues))
        write_json_atomic(ctx.dir / OCR_JSON, data)
        share = (len(sent) / len(accepted)) if accepted else 0.0
        st.outputs = {
            "json": OCR_JSON,
            "sent": len(sent),
            "accepted": len(accepted),
            "share": round(share, 3),
            "results": len(results),
        }
        logger.info(
            "ocr done",
            extra={
                "fields": {
                    "sent": len(sent),
                    "accepted": len(accepted),
                    "share": round(share, 3),
                    "results": len(results),
                }
            },
        )
        return data

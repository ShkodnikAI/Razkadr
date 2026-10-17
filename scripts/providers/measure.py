"""Provider замеры for naryad 004: latency, cost, simple quality.

Runs N calls against a provider endpoint, records wall latency and billed
usage, converts usage to USD via scripts/providers/tariffs.json, prints a
markdown table and writes results.json. All numbers come from these real
runs (AGENTS.md: no invented numbers).

Keys come strictly from environment variables and are registered in the
razkadr secret registry; every printed/written string passes through
SECRETS.mask, so a leaked key cannot appear in the output.

Usage (see scripts/providers/README.md for the full matrix):
    .venv/bin/python scripts/providers/measure.py --task asr \\
        --provider deepgram-nova3 --audio /path/minute.wav --repeat 3 \\
        --out runs/providers/asr_deepgram.json
    .venv/bin/python scripts/providers/measure.py --dry-run  # no network

Exit codes: 0 ok, 2 usage error, 1 at least one measured call failed.
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from razkadr.log import SECRETS, deep_mask

TARIFFS = Path(__file__).resolve().parent / "tariffs.json"

TASK_PROVIDERS: dict[str, list[str]] = {
    "asr": ["gemini-transcribe", "elevenlabs-scribe", "deepgram-nova3", "openai-transcribe"],
    "embed": ["openai-text-embedding-3-small", "cohere-embed-v4", "jina-embeddings-v4", "gemini-embedding"],
    "vlm": ["gemini-flashlite", "openai-luna"],
    "llm": ["openai-luna", "gemini-flashlite", "gemini-flash"],
}

PROVIDER_ENV: dict[str, str] = {
    "gemini-transcribe": "GEMINI_API_KEY",
    "gemini-flashlite": "GEMINI_API_KEY",
    "gemini-flash": "GEMINI_API_KEY",
    "gemini-embedding": "GEMINI_API_KEY",
    "elevenlabs-scribe": "ELEVENLABS_API_KEY",
    "deepgram-nova3": "DEEPGRAM_API_KEY",
    "openai-transcribe": "OPENAI_API_KEY",
    "openai-luna": "OPENAI_API_KEY",
    "openai-text-embedding-3-small": "OPENAI_API_KEY",
    "cohere-embed-v4": "COHERE_API_KEY",
    "jina-embeddings-v4": "JINA_API_KEY",
}

DEFAULT_MODELS = {
    "gemini-transcribe": "gemini-3.5-transcribe",
    "elevenlabs-scribe": "scribe_v2",
    "deepgram-nova3": "nova-3",
    "openai-transcribe": "gpt-transcribe",
    "openai-4o-mini-transcribe": "gpt-4o-mini-transcribe",
    "gemini-flashlite": "gemini-3.5-flash-lite",
    "gemini-flash": "gemini-3.7-flash",
    "openai-luna": "gpt-6-luna",
    "openai-text-embedding-3-small": "text-embedding-3-small",
    "cohere-embed-v4": "embed-v4.0",
    "jina-embeddings-v4": "jina-embeddings-v4",
    "gemini-embedding": "gemini-embedding-001",
}


class MeasureError(Exception):
    """Script-level failure; message is masked before printing."""


# ---------------------------------------------------------------- http utils
def http_json(url: str, *, method: str = "POST", headers: dict | None = None,
              data: bytes | None = None, timeout: float = 120.0) -> tuple[dict, float]:
    """One HTTP request returning (parsed json, latency seconds)."""
    request = urllib.request.Request(url, data=data, method=method)
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise MeasureError(f"HTTP {exc.code} from {url.split('?')[0]}: {SECRETS.mask(detail)}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise MeasureError(f"network error calling {url.split('?')[0]}: {exc}") from None
    latency = time.perf_counter() - started
    try:
        return json.loads(body), latency
    except json.JSONDecodeError as exc:
        raise MeasureError(f"non-JSON response from {url.split('?')[0]}: {exc}") from None


def multipart(fields: dict[str, str], file_field: str, file_path: Path,
              file_type: str) -> tuple[bytes, str]:
    """Build a multipart/form-data body; returns (body, content_type)."""
    boundary = "razkadr-measure-boundary-7f3a"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode()
        )
    chunks.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"{file_field}\"; "
        f"filename=\"{file_path.name}\"\r\nContent-Type: {file_type}\r\n\r\n".encode()
    )
    chunks.append(file_path.read_bytes())
    chunks.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


# ------------------------------------------------------------- input helpers
def prepare_audio(audio: Path, workdir: Path) -> Path:
    """Normalize to 16 kHz mono wav (API-friendly, ~1.9 MB per minute)."""
    workdir.mkdir(parents=True, exist_ok=True)
    prepared = workdir / f"{audio.stem}_16k.wav"
    proc = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(audio), "-ac", "1", "-ar", "16000", str(prepared)],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0 or not prepared.exists():
        raise MeasureError(f"ffmpeg failed on {audio.name}: {proc.stderr[-300:]}")
    return prepared


def audio_duration_s(path: Path) -> float:
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(json.loads(probe.stdout)["format"]["duration"])


def norm_words(text: str) -> list[str]:
    return "".join(ch if ch.isalnum() else " " for ch in text.casefold()).split()


def levenshtein(a: str, b: str) -> int:
    if not a or not b:
        return max(len(a), len(b))
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def wer(reference: str, hypothesis: str) -> float:
    ref, hyp = norm_words(reference), norm_words(hypothesis)
    if not ref:
        return 0.0
    rows = list(range(len(hyp) + 1))
    for i, rw in enumerate(ref, 1):
        cur = [i]
        for j, hw in enumerate(hyp, 1):
            cur.append(min(rows[j] + 1, cur[j - 1] + 1, rows[j - 1] + (rw != hw)))
        rows = cur
    return rows[-1] / len(ref)


# ---------------------------------------------------------------- cost utils
def load_tariffs() -> dict:
    with TARIFFS.open(encoding="utf-8") as fh:
        return json.load(fh)


def cost_usd(task: str, provider: str, usage: list[dict], tariffs: dict) -> tuple[float | None, str]:
    """Convert measured usage to USD; returns (cost, billing note)."""
    entry = tariffs.get(task, {}).get(provider) or {}
    price_mtok = entry.get("price_per_mtok")
    price_mtok_out = entry.get("output_price_per_mtok")
    price_minute = entry.get("price_per_minute")
    notes: list[str] = []
    total = 0.0
    known = False
    for unit in usage:
        kind, value = unit["kind"], float(unit["value"])
        if kind == "out_tokens" and price_mtok_out is not None:
            total += value / 1_000_000 * price_mtok_out
            known = True
        elif kind == "tokens" and price_mtok is not None:
            total += value / 1_000_000 * price_mtok
            known = True
        elif kind == "seconds" and price_minute is not None:
            total += value / 60 * price_minute
            known = True
        elif kind == "minutes" and price_minute is not None:
            total += value * price_minute
            known = True
        else:
            notes.append(f"unpriced:{kind}")
    if not known:
        return None, "; ".join(notes) or "no tariff"
    return round(total, 6), entry.get("unit_note", "")


# --------------------------------------------------------------- task runners
# Each runner returns a list of per-repeat records:
# {repeat, latency_s, usage:[{kind,value}], output|error, quality:{...}}
# Usage kinds match the provider layer: seconds | tokens | images (base.py).


def run_asr(provider: str, model: str, audio: Path, args: argparse.Namespace) -> list[dict]:
    prepared = prepare_audio(audio, Path(args.workdir))
    duration = audio_duration_s(prepared)
    usage_base = [{"kind": "seconds", "value": round(duration, 3)}]
    key = os.environ[PROVIDER_ENV[provider]]
    records = []
    for i in range(args.repeat):
        try:
            if provider == "gemini-transcribe":
                body = json.dumps({
                    "contents": [{"parts": [
                        {"text": "Transcribe the audio. Return JSON: "
                                 "{\"segments\":[{\"t_start\":number,\"t_end\":number,\"text\":str}]}. "
                                 "Times in seconds from the start."},
                        {"inline_data": {"mime_type": "audio/wav",
                                         "data": base64.b64encode(prepared.read_bytes()).decode()}},
                    ]}],
                    "generationConfig": {"responseMimeType": "application/json", "temperature": 0},
                }).encode()
                payload, latency = http_json(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                    headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                    data=body, timeout=args.timeout,
                )
                text = "".join(p.get("text", "") for p in payload["candidates"][0]["content"]["parts"])
                used = payload.get("usageMetadata", {})
                tokens = used.get("promptTokenCount", 0) + used.get("candidatesTokenCount", 0)
                usage = [{"kind": "tokens", "value": tokens}]  # billed per audio token, not per minute
            elif provider == "deepgram-nova3":
                body, ctype = multipart({"model": model, "smart_format": "true", "language": "multi"},
                                        "file", prepared, "audio/wav")
                payload, latency = http_json(
                    "https://api.deepgram.com/v1/listen",
                    headers={"Authorization": f"Token {key}", "Content-Type": ctype},
                    data=body, timeout=args.timeout,
                )
                alt = payload["results"]["channels"][0]["alternatives"][0]
                text = alt.get("transcript", "")
                usage = usage_base  # billed per minute of audio
            elif provider == "elevenlabs-scribe":
                body, ctype = multipart({"model_id": model}, "file", prepared, "audio/wav")
                payload, latency = http_json(
                    "https://api.elevenlabs.io/v1/speech-to-text",
                    headers={"xi-api-key": key, "Content-Type": ctype},
                    data=body, timeout=args.timeout,
                )
                text = payload.get("text", "")
                usage = usage_base  # billed per minute of audio
            else:  # openai-transcribe / openai-4o-mini-transcribe
                body, ctype = multipart({"model": model, "response_format": "json"},
                                        "file", prepared, "audio/wav")
                payload, latency = http_json(
                    "https://api.openai.com/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {key}", "Content-Type": ctype},
                    data=body, timeout=args.timeout,
                )
                text = payload.get("text", "")
                used = payload.get("usage", {})
                usage = [{"kind": "tokens", "value": used.get("input_tokens", 0)}]
                if used.get("output_tokens"):
                    usage.append({"kind": "out_tokens", "value": used["output_tokens"]})
            quality: dict = {"segments": None, "lang": None, "wer": None}
            if args.reference_text:
                quality["wer"] = round(wer(args.reference_text, text), 4)
            records.append({"repeat": i, "latency_s": round(latency, 3), "usage": usage,
                            "output": {"text_sample": SECRETS.mask(text[:200])}, "quality": quality,
                            "error": None, "audio_s": round(duration, 3)})
        except MeasureError as exc:
            records.append({"repeat": i, "latency_s": None, "usage": [], "output": None,
                            "quality": {}, "error": str(exc), "audio_s": round(duration, 3)})
    return records


def run_embed(provider: str, model: str, args: argparse.Namespace) -> list[dict]:
    key = os.environ[PROVIDER_ENV[provider]]
    texts = list(args.texts or [])
    images = list(args.images or [])
    if not texts and not images:
        raise MeasureError("--task embed needs --text and/or --image inputs")
    if provider.startswith("openai") and images and not texts:
        # /v1/embeddings would reject an empty text list; fail fast without network.
        return [{"repeat": i, "latency_s": None, "usage": [], "output": None, "quality": {},
                 "error": "openai embeddings do not accept images; use the VLM-description "
                          "+ text path (ADR-0002)", "dims_txt": None, "dims_img": None}
                for i in range(args.repeat)]
    records = []
    for i in range(args.repeat):
        try:
            n_tokens, dims_img, dims_txt = 0, None, None
            if provider.startswith("openai"):
                payload, latency = http_json(
                    "https://api.openai.com/v1/embeddings",
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    data=json.dumps({"model": model, "input": texts}).encode(), timeout=args.timeout,
                )
                dims_txt = len(payload["data"][0]["embedding"])
                n_tokens = payload.get("usage", {}).get("prompt_tokens", 0)
                if images:
                    records.append({"repeat": i, "latency_s": None, "usage": [], "output": None,
                                    "quality": {}, "error": "openai embeddings do not accept images; "
                                    "use the VLM-description + text path (ADR-0002)", "dims_txt": dims_txt})
                    continue
            elif provider == "cohere-embed-v4":
                body: dict = {"model": model, "embedding_types": ["float"], "texts": texts}
                if images:
                    body["images"] = [data_uri(p) for p in images]
                payload, latency = http_json(
                    "https://api.cohere.com/v2/embed",
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    data=json.dumps(body).encode(), timeout=args.timeout,
                )
                emb = payload["embeddings"]["float"]
                dims_txt = len(emb[0]) if texts else None
                dims_img = len(emb[len(texts)]) if images and len(emb) > len(texts) else dims_img
                n_tokens = payload.get("meta", {}).get("billed_units", {}).get("input_tokens", 0)
            elif provider == "jina-embeddings-v4":
                items = [{"text": t} for t in texts] + [{"image": data_uri(p)} for p in images]
                payload, latency = http_json(
                    "https://api.jina.ai/v1/embeddings",
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    data=json.dumps({"model": model, "input": items}).encode(), timeout=args.timeout,
                )
                embs = payload["data"]
                dims_txt = len(embs[0]["embedding"]) if embs else None
                dims_img = dims_txt
                n_tokens = payload.get("usage", {}).get("total_tokens", 0)
            else:  # gemini-embedding (text only)
                payload, latency = http_json(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents",
                    headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                    data=json.dumps({"requests": [{"model": f"models/{model}",
                                                  "content": {"parts": [{"text": t}]}} for t in texts]}).encode(),
                    timeout=args.timeout,
                )
                dims_txt = len(payload["embeddings"][0]["values"])
                n_tokens = 0  # usageMetadata not returned by batchEmbedContents
            usage = [{"kind": "tokens", "value": n_tokens}] if n_tokens else []
            records.append({"repeat": i, "latency_s": round(latency, 3), "usage": usage,
                            "output": None, "quality": {}, "error": None,
                            "dims_txt": dims_txt, "dims_img": dims_img,
                            "n_texts": len(texts), "n_images": len(images)})
        except MeasureError as exc:
            records.append({"repeat": i, "latency_s": None, "usage": [], "output": None,
                            "quality": {}, "error": str(exc)})
    return records


def _openai_chat(key: str, model: str, messages: list[dict], timeout: float) -> tuple[dict, float]:
    return http_json(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        data=json.dumps({"model": model, "messages": messages, "temperature": 0}).encode(),
        timeout=timeout,
    )


def _gemini_generate(key: str, model: str, parts: list[dict], timeout: float) -> tuple[dict, float]:
    return http_json(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": key, "Content-Type": "application/json"},
        data=json.dumps({"contents": [{"parts": parts}],
                         "generationConfig": {"temperature": 0}}).encode(),
        timeout=timeout,
    )


def run_vlm(provider: str, model: str, args: argparse.Namespace) -> list[dict]:
    key = os.environ[PROVIDER_ENV[provider]]
    images = list(args.images or [])
    if not images:
        raise MeasureError("--task vlm needs --image inputs")
    prompt = args.prompt or "Describe the image and transcribe any visible text verbatim."
    records = []
    for i in range(args.repeat):
        for image in images:
            try:
                if provider.startswith("openai"):
                    payload, latency = _openai_chat(key, model, [{
                        "role": "user",
                        "content": [{"type": "text", "text": prompt},
                                    {"type": "image_url", "image_url": {"url": data_uri(image)}}],
                    }], args.timeout)
                    text = payload["choices"][0]["message"]["content"]
                    used = payload.get("usage", {})
                    usage = [{"kind": "tokens", "value": used.get("prompt_tokens", 0)},
                             {"kind": "tokens", "value": used.get("completion_tokens", 0)}]
                else:  # gemini
                    payload, latency = _gemini_generate(key, model, [
                        {"text": prompt},
                        {"inline_data": {"mime_type": mimetypes.guess_type(image.name)[0] or "image/png",
                                         "data": base64.b64encode(image.read_bytes()).decode()}},
                    ], args.timeout)
                    text = "".join(p.get("text", "") for p in payload["candidates"][0]["content"]["parts"])
                    used = payload.get("usageMetadata", {})
                    usage = [{"kind": "tokens", "value": used.get("promptTokenCount", 0)},
                             {"kind": "tokens", "value": used.get("candidatesTokenCount", 0)}]
                quality: dict = {}
                if args.expected_text:
                    quality["expected_hit"] = args.expected_text.casefold() in text.casefold()
                    quality["levenshtein_ratio"] = round(
                        1 - levenshtein("".join(norm_words(args.expected_text)),
                                        "".join(norm_words(text[:len(args.expected_text) + 20])))
                        / max(len(norm_words(args.expected_text)), 1), 4)
                records.append({"repeat": i, "image": image.name, "latency_s": round(latency, 3),
                                "usage": usage, "quality": quality, "error": None,
                                "output": {"text_sample": SECRETS.mask(text[:200])}})
            except MeasureError as exc:
                records.append({"repeat": i, "image": image.name, "latency_s": None, "usage": [],
                                "quality": {}, "error": str(exc), "output": None})
    return records


def run_llm(provider: str, model: str, args: argparse.Namespace) -> list[dict]:
    key = os.environ[PROVIDER_ENV[provider]]
    if not args.texts:
        raise MeasureError("--task llm needs --text inputs")
    prompt = args.prompt or "Translate the following text to English. Output only the translation."
    records = []
    for i in range(args.repeat):
        for text in args.texts:
            try:
                if provider.startswith("openai"):
                    payload, latency = _openai_chat(
                        key, model, [{"role": "user", "content": f"{prompt}\n\n{text}"}], args.timeout)
                    out = payload["choices"][0]["message"]["content"]
                    used = payload.get("usage", {})
                    usage = [{"kind": "tokens", "value": used.get("prompt_tokens", 0)},
                             {"kind": "tokens", "value": used.get("completion_tokens", 0)}]
                else:  # gemini
                    payload, latency = _gemini_generate(
                        key, model, [{"text": f"{prompt}\n\n{text}"}], args.timeout)
                    out = "".join(p.get("text", "") for p in payload["candidates"][0]["content"]["parts"])
                    used = payload.get("usageMetadata", {})
                    usage = [{"kind": "tokens", "value": used.get("promptTokenCount", 0)},
                             {"kind": "tokens", "value": used.get("candidatesTokenCount", 0)}]
                quality = {}
                if args.reference_text:
                    quality["wer"] = round(wer(args.reference_text, out), 4)
                records.append({"repeat": i, "latency_s": round(latency, 3), "usage": usage,
                                "quality": quality, "error": None,
                                "output": {"text_sample": SECRETS.mask(out[:200])}})
            except MeasureError as exc:
                records.append({"repeat": i, "latency_s": None, "usage": [], "quality": {},
                                "error": str(exc), "output": None})
    return records


# ---------------------------------------------------------------- aggregation
def aggregate(task: str, provider: str, model: str, records: list[dict],
              tariffs: dict) -> dict:
    latencies = [r["latency_s"] for r in records if r["latency_s"] is not None]
    total_cost, note = None, ""
    billed = 0.0
    priced_records = 0
    for record in records:
        cost, cost_note = cost_usd(task, provider, record.get("usage", []), tariffs)
        if cost is not None:
            total_cost = round((total_cost or 0.0) + cost, 6)
            billed += cost
            priced_records += 1
        if cost_note and not note:
            note = cost_note
    errors = [r for r in records if r.get("error")]
    agg: dict = {
        "task": task, "provider": provider, "model": model,
        "calls": len(records), "errors": len(errors),
        "latency_s_min": min(latencies) if latencies else None,
        "latency_s_median": round(statistics.median(latencies), 3) if latencies else None,
        "latency_s_max": max(latencies) if latencies else None,
        "cost_usd_total": total_cost, "cost_note": note,
    }
    if task == "asr" and billed and priced_records:
        audio_s = statistics.mean(r["audio_s"] for r in records if r.get("audio_s"))
        agg["cost_usd_per_minute_audio"] = round(billed / priced_records / (audio_s / 60), 6)
    if errors:
        agg["error_sample"] = SECRETS.mask(errors[0]["error"])[:300]
    return agg


def main() -> int:
    parser = argparse.ArgumentParser(description="Provider latency/cost/quality замеры (naryad 004)")
    parser.add_argument("--task", choices=sorted(TASK_PROVIDERS))
    parser.add_argument("--provider", action="append", choices=sorted(PROVIDER_ENV))
    parser.add_argument("--model", help="override the default model for a single --provider")
    parser.add_argument("--audio", type=Path, help="input audio/video for ASR (any ffmpeg-readable)")
    parser.add_argument("--image", dest="images", action="append", type=Path, default=[])
    parser.add_argument("--text", dest="texts", action="append", default=[])
    parser.add_argument("--prompt", help="task prompt for vlm/llm")
    parser.add_argument("--reference-text", help="ASR: reference for WER; llm: reference output")
    parser.add_argument("--expected-text", help="vlm: text expected on the image (OCR check)")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--out", type=Path, default=Path("runs/providers/results.json"))
    parser.add_argument("--workdir", type=Path, default=Path("runs/providers/work"))
    parser.add_argument("--dry-run", action="store_true", help="validate inputs and env, no network")
    args = parser.parse_args()

    for value in [*args.texts, args.reference_text or "", args.expected_text or ""]:
        SECRETS.register(value)  # user text may contain private data; mask it too

    plan_providers = args.provider or (TASK_PROVIDERS.get(args.task, []) if args.task else [])
    if args.dry_run:
        print("dry-run: no network calls will be made")
        print(f"task: {args.task} providers: {plan_providers or '(none)'}")
        missing_env = sorted({PROVIDER_ENV[p] for p in plan_providers
                              if not os.environ.get(PROVIDER_ENV[p])})
        print("missing env keys:", ", ".join(missing_env) if missing_env else "none")
        inputs = [("--audio", args.audio), * [("--image", i) for i in args.images]]
        for name, path in inputs:
            print(f"input {name}: {path} -> {'ok' if path and path.exists() else 'MISSING'}")
        if args.audio:
            prepared = prepare_audio(args.audio, args.workdir)
            print(f"audio prepared: {prepared.name} {audio_duration_s(prepared):.1f}s "
                  f"{prepared.stat().st_size} bytes")
        print("tariffs verified_date:", load_tariffs()["verified_date"])
        return 0

    if not args.task or not plan_providers:
        parser.error("--task and at least one --provider are required (or use --dry-run)")
    for provider in plan_providers:
        env = PROVIDER_ENV[provider]
        if not os.environ.get(env):
            print(f"SKIP {provider}: {env} not set", file=sys.stderr)
            plan_providers = [p for p in plan_providers if p != provider]
    if not plan_providers:
        print("no providers with keys; nothing to measure", file=sys.stderr)
        return 1

    tariffs = load_tariffs()
    results: dict = {"task": args.task, "repeat": args.repeat,
                     "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     "tariffs_verified_date": tariffs["verified_date"], "providers": []}
    failed = 0
    for provider in plan_providers:
        model = args.model or DEFAULT_MODELS[provider]
        if args.task == "asr":
            if not args.audio:
                parser.error("--task asr needs --audio")
            records = run_asr(provider, model, args.audio, args)
        elif args.task == "embed":
            records = run_embed(provider, model, args)
        elif args.task == "vlm":
            records = run_vlm(provider, model, args)
        else:
            records = run_llm(provider, model, args)
        failed += bool(any(r.get("error") for r in records))
        block = aggregate(args.task, provider, model, records, tariffs)
        block["records"] = records
        results["providers"].append(block)
        print(json.dumps(deep_mask(block), ensure_ascii=False, indent=2))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(deep_mask(results), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {args.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

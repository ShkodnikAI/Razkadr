"""Episode text stage (naryad 017): draft Storyboard -> Storyboard with text.

Fills ``episodes[].text`` of the draft storyboard (naryad 014) from the run
artifacts, in three modes fixed by the storyboard schema (naryad 002):

- **verbatim** - the transcript as spoken: transcript segments overlapping
  the episode span [t_start, t_end] joined in time order; the text is never
  rewritten (numbers hold by construction);
- **condensed** - the verbatim text shortened by the LLM through the naryad
  033 provider layer; the stage knows only the LLMProvider protocol;
- **described** - a VLM description of the episode frame: the only source of
  text for a silent episode (no transcript segments in the span).

Mode selection: ``[text] mode`` = auto (default; speech -> condensed, silent
-> described), or an explicit verbatim / condensed / described. An explicit
verbatim/condensed on a silent episode falls back to described; an explicit
described is used even where speech exists. The LLM answer is never trusted
with the source text: it is accepted only when it is non-empty, preserves
every number of the original (gateway D invariant, deterministic check) and
fits the role length limit; otherwise the stage falls back to verbatim and
records the reason as a warning. An answer that is valid but too long is
shortened at a word boundary with an ellipsis and a warning. VLM answers
(described mode) are normalized and length-capped the same way: there is no
source text to verify them against, the mode is the storyboard schema's
explicit "frame description".

Degradation follows the chunks stage policy (naryad 008): when the mode
needs an LLM but no ``[llm]`` provider is configured, the stage falls back
to verbatim with a warning; when described mode needs a VLM but no ``[vlm]``
provider is configured, the episode keeps no text (the storyboard schema
allows text to be absent) with a warning. A configured-but-broken provider
(unknown name, missing key, auth/quota failure) fails the stage loudly:
explicit configuration must not degrade silently.

User edits are preserved (storyboard contract): episodes with
``user_edited: true`` that already carry text are copied as-is; their text
is never regenerated. Episodes without text are produced for none of the
modes silently - every decision lands in the report ``text.json``.

Length limits per role live in ``[text.role_limits]`` (defaults mirror the
layout engine's ``[layout]`` limits, naryad 021: hero 120, ingredients
12 items x 40, steps/prep 120, serving 180); unknown roles take
``[text] default_limit``. The limit is the generation target for condensed
and the hard cap for described; verbatim is capped only by the layout
engine's truncation, "verbatim is the transcript as spoken" is its point.

Re-run semantics: the stage rewrites ``storyboard.json`` in place (the run
dir's draft gains text); a repeat run regenerates non-edited episodes -
provider calls are cached by the naryad 033 call cache, so repeats are
deterministic and free, and user edits survive both.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from importlib import resources
from pathlib import Path

from razkadr.log import get_logger
from razkadr.providers.base import ProviderConfigError
from razkadr.providers.layer import ProviderLayer
from razkadr.providers.types import Message
from razkadr.runctx import RunContext, file_sha256
from razkadr.validation import validate_document

logger = get_logger("text")

STORYBOARD_JSON = "storyboard.json"
TEXT_JSON = "text.json"
TRANSCRIPT_JSON = "transcript.json"

PROMPT_CONDENSE = "condense.v1.md"
PROMPT_DESCRIBE = "describe.v1.md"
PROMPT_TRANSLATE = "translate_episode.v1.md"
PROMPT_MAX_CHARS_PLACEHOLDER = "{{max_chars}}"
PROMPT_LANG_PLACEHOLDER = "{{lang}}"

MODES = ("auto", "verbatim", "condensed", "described")
EPISODE_TEXT_MODES = ("verbatim", "condensed", "described")
TIME_EPS = 1e-9
ELLIPSIS = "…"
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
ANSWER_EDGES_RE = re.compile(r"^[\s\"'«»“”»]+|[\s\"'«»“”»]+$")
# unit normalization (naryad 018): applied to the source text BEFORE the
# provider step, so the number invariant compares against normalized original
FAHRENHEIT_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*[°º]\s*[Ff]\b")
CUP_RE = re.compile(r"\b(\d+(?:[.,]\d+)?)\s*cups?\b", re.IGNORECASE)

# [text] defaults: mode = auto (speech -> condensed, silent -> described);
# role_limits mirror the layout engine defaults (naryad 021): hero is the
# hero_caption subheading, ingredients is 12 list items x 40 chars, prep and
# steps are the steps_grid cell, serving is a caption.
# output_lang = "" -> project.output_lang (naryad 018): when it differs from
# the episode's source_lang the output is translated through [llm].
# units: unit normalization is off by default - numbers must not change
# without an explicit owner decision (naryad 018 "по настройке").
DEFAULT_ROLE_LIMITS: dict[str, int] = {
    "hero": 120,
    "ingredients": 480,
    "prep": 120,
    "steps": 120,
    "step": 120,  # the canonical example (examples/storyboard.*) uses the singular
    "serving": 180,
}
TEXT_DEFAULTS: dict = {
    "mode": "auto",
    "role_limits": {},
    "default_limit": 180,
    "output_lang": "",
    "units": {},
}
UNIT_DEFAULTS: dict = {
    "fahrenheit_to_celsius": False,  # 350 °F -> 177 °C (arithmetic, rounded)
    "cup_to_ml": 0,                  # 1 cup -> N ml (US cup ~ 240 ml); 0 = off
}


class TextError(Exception):
    """Text stage failure with a clear, user-facing message (CLI exit 2)."""


def text_params(config: dict) -> dict:
    """Effective [text] settings; goes into the stage cache key."""
    section = {**TEXT_DEFAULTS, **(config.get("text") or {})}
    mode = str(section["mode"]).strip()
    if mode not in MODES:
        raise TextError(f"[text] mode must be one of {MODES}, got {mode!r}")
    default_limit = int(section["default_limit"])
    if default_limit <= 0:
        raise TextError(f"[text] default_limit must be positive, got {default_limit}")
    role_limits = dict(DEFAULT_ROLE_LIMITS)
    for role, limit in (section.get("role_limits") or {}).items():
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise TextError(f"[text] role_limits.{role} must be a positive integer, got {limit!r}")
        role_limits[str(role)] = int(limit)
    output_lang = str(section.get("output_lang") or "").strip()
    units_section = {**UNIT_DEFAULTS, **(section.get("units") or {})}
    f_to_c = bool(units_section["fahrenheit_to_celsius"])
    cup_ml = int(units_section["cup_to_ml"])
    if cup_ml < 0:
        raise TextError(f"[text] units.cup_to_ml must be >= 0, got {cup_ml}")
    return {
        "mode": mode,
        "role_limits": role_limits,
        "default_limit": default_limit,
        "output_lang": output_lang,
        "units": {"fahrenheit_to_celsius": f_to_c, "cup_to_ml": cup_ml},
    }


def load_prompt(name: str) -> str:
    """Versioned prompt packaged with the library (same contract as naryad 008)."""
    return resources.files("razkadr.prompts").joinpath(name).read_text(encoding="utf-8")


def condense_system_prompt(max_chars: int) -> str:
    """Prompt condense.v1 with the character limit substituted."""
    return load_prompt(PROMPT_CONDENSE).replace(PROMPT_MAX_CHARS_PLACEHOLDER, str(max_chars))


def describe_system_prompt(max_chars: int, lang: str) -> str:
    """Prompt describe.v1 with the character limit and output language substituted."""
    return (
        load_prompt(PROMPT_DESCRIBE)
        .replace(PROMPT_MAX_CHARS_PLACEHOLDER, str(max_chars))
        .replace(PROMPT_LANG_PLACEHOLDER, lang)
    )


def translate_system_prompt(lang: str) -> str:
    """Prompt translate_episode.v1 with the target language substituted."""
    return load_prompt(PROMPT_TRANSLATE).replace(PROMPT_LANG_PLACEHOLDER, lang)


def normalize_units(text: str, units: dict) -> tuple[str, list[str]]:
    """Deterministic unit normalization of the source text (naryad 018).

    Applied BEFORE the provider step: the LLM then receives already
    normalized text, the number invariant compares against the normalized
    original, and no provider can silently misconvert. Off by default -
    numbers must not change without an explicit owner decision. Returns
    (text, applied conversions) with human-readable records for the report.
    """
    applied: list[str] = []

    def _f_to_c(match: re.Match) -> str:
        f_value = float(match.group(1).replace(",", "."))
        c_value = round((f_value - 32.0) * 5.0 / 9.0)
        applied.append(f"{match.group(0).strip()} -> {c_value:g} °C")
        return f"{c_value:g} °C"

    def _cup_to_ml(match: re.Match) -> str:
        cups = float(match.group(1).replace(",", "."))
        ml = cups * units["cup_to_ml"]
        applied.append(f"{match.group(0).strip()} -> {ml:g} мл")
        return f"{ml:g} мл"

    if units.get("fahrenheit_to_celsius"):
        text = FAHRENHEIT_RE.sub(_f_to_c, text)
    if int(units.get("cup_to_ml") or 0) > 0:
        text = CUP_RE.sub(_cup_to_ml, text)
    return text, applied


def segments_in_span(segments: list[dict], t_start: float, t_end: float) -> list[dict]:
    """Transcript segments overlapping the episode span, in artifact order.

    A segment counts when it intersects [t_start, t_end] at all: segments
    are short, cutting one mid-way would drop words - the verbatim mode is
    "the transcript as spoken", so whole segments are joined.
    """
    return [
        s
        for s in segments
        if float(s["t_start"]) < t_end - TIME_EPS and float(s["t_end"]) > t_start + TIME_EPS
    ]


def dominant_lang(segments: list[dict]) -> str | None:
    """Most frequent segment language, first seen on a tie (naryad 007 rule)."""
    counts: Counter[str] = Counter(str(s["lang"]) for s in segments)
    if not counts:
        return None
    first_seen: dict[str, int] = {}
    for i, s in enumerate(segments):
        first_seen.setdefault(str(s["lang"]), i)
    return max(counts, key=lambda lang: (counts[lang], -first_seen[lang]))


def numbers(text: str) -> list[str]:
    """Number tokens of the text: gateway D's "quantities survive" invariant."""
    return NUMBER_RE.findall(text)


def numbers_preserved(original: str, output: str) -> bool:
    """Every number of the original appears in the output.

    Decimal separators are interchangeable ("0,5" == "0.5"): the check
    accepts either spelling. Units and words are the prompt's contract and
    the gateway D review's job; digits are enforced here, deterministically.
    """
    for token in numbers(original):
        if token in output or token.replace(",", ".") in output or token.replace(".", ",") in output:
            continue
        return False
    return True


def normalize_answer(raw: str) -> str:
    """Provider answer -> slide text: unquoted, single-spaced, trimmed."""
    text = " ".join(str(raw).split())
    return ANSWER_EDGES_RE.sub("", text)


def truncate_at_word(text: str, limit: int) -> str:
    """Hard cap at a word boundary with an ellipsis (text already fits -> as is)."""
    if len(text) <= limit:
        return text
    head = text[:limit]
    space = head.rfind(" ")
    if space >= limit // 2:
        return text[:space].rstrip(" ,;:-") + ELLIPSIS
    return text[:limit].rstrip(" ,;:-") + ELLIPSIS


def resolve_mode(wanted: str, has_speech: bool) -> str:
    """[text] mode + episode speech -> the episode's text mode."""
    if wanted == "auto":
        return "condensed" if has_speech else "described"
    if wanted in ("verbatim", "condensed") and not has_speech:
        return "described"
    return wanted


def _load_json(path: Path, what: str, required: bool = True) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        if required:
            raise TextError(f"{what} is missing: {path}") from exc
        return None
    except json.JSONDecodeError as exc:
        raise TextError(f"{what} is not valid JSON: {path}: {exc}") from exc


def _build_text_block(
    mode: str,
    original: str,
    output: str,
    source_lang: str | None,
    output_lang: str | None,
) -> dict:
    """Episode text dict per the storyboard schema (naryad 002); empty fields omitted.

    ``output_lang`` carries the language of ``output``: the source language
    by default, the target language after a successful translation (naryad
    018), the project language for described mode.
    """
    block: dict = {"mode": mode}
    if source_lang:
        block["source_lang"] = source_lang
    if mode == "described":
        block["original"] = ""
    else:
        block["original"] = original
    out_lang = output_lang or source_lang
    if out_lang:
        block["output_lang"] = out_lang
    block["output"] = output
    return block


def run_text(ctx: RunContext, config: dict) -> tuple[dict, dict]:
    """Text stage body; returns (storyboard with text, report).

    Like select (naryad 014), the stage records no RunContext stage cache
    entry: its output overwrites its own input (the draft gains text), so a
    stage-level cache key cannot be stable. Repeats are safe by design -
    regeneration is idempotent and every LLM/VLM call is cached and billed
    through the naryad 033 provider layer into run.json. The report carries
    the input fingerprints, so the auditor can see which storyboard and
    transcript the text was produced from.
    """
    params = text_params(config)

    storyboard_path = ctx.dir / STORYBOARD_JSON
    if not storyboard_path.is_file():
        raise TextError(f"storyboard.json is missing in the run directory: {ctx.dir} (run razkadr select first)")
    storyboard = _load_json(storyboard_path, "storyboard")
    if validate_document(storyboard, "storyboard"):
        raise TextError("storyboard.json is not a valid Storyboard (re-run razkadr select)")
    transcript_path = ctx.dir / TRANSCRIPT_JSON
    if not transcript_path.is_file():
        raise TextError(
            "text stage needs the transcript stage output: transcript.json is missing in the run directory"
        )
    transcript = _load_json(transcript_path, "transcript")
    if validate_document(transcript, "transcript"):
        raise TextError("transcript.json is not a valid transcript artifact")

    segments = list(transcript.get("segments") or [])
    sources = {
        str(s.get("id")): str(s.get("detected_lang") or "")
        for s in (storyboard.get("sources") or [])
    }
    output_lang_project = str((storyboard.get("project") or {}).get("output_lang") or "")

    llm_section = config.get("llm") or {}
    vlm_section = config.get("vlm") or {}

    # services are built lazily and memoized: a run whose episodes never
    # need the LLM/VLM must not construct the provider (and must not trip
    # resolve_key on an api_key_env that is set but empty), and episodes
    # that do must share one service (one call cache, one accounting stream)
    def make_service(section_name: str, task: str, stage: str):
        holder: dict = {"service": None}

        def get():
            if holder["service"] is None:
                try:
                    holder["service"] = ProviderLayer(config, ctx)._service(task, stage)
                except ProviderConfigError as exc:
                    raise TextError(f"{exc} (the [{section_name}] section is configured; see docs/config.md)") from None
            return holder["service"]

        return get

    llm_service = make_service("llm", "llm", "text")
    vlm_service = make_service("vlm", "vlm", "text")

    has_llm = bool(llm_section.get("provider"))
    has_vlm = bool(vlm_section.get("provider"))

    report: dict = {
        "mode": params["mode"],
        "inputs": {
            "storyboard_sha256": file_sha256(storyboard_path),
            "transcript_sha256": file_sha256(transcript_path),
        },
        "providers": {
            "llm": {"provider": llm_section.get("provider"), "model": llm_section.get("model")} if has_llm else None,
            "vlm": {"provider": vlm_section.get("provider"), "model": vlm_section.get("model")} if has_vlm else None,
        },
        "episodes": [],
        "warnings": [],
    }
    mode_counts: dict[str, int] = {}
    for episode in storyboard.get("episodes") or []:
        entry = _episode_text(
            ctx,
            episode,
            segments,
            sources,
            output_lang_project,
            params,
            llm_service if has_llm else None,
            vlm_service if has_vlm else None,
            report["warnings"],
        )
        report["episodes"].append(entry)
        if entry.get("kept"):
            mode_counts["kept_user_edited"] = mode_counts.get("kept_user_edited", 0) + 1
        elif entry.get("text_mode"):
            mode_counts[entry["text_mode"]] = mode_counts.get(entry["text_mode"], 0) + 1
        elif entry.get("skipped"):
            mode_counts["skipped_no_text"] = mode_counts.get("skipped_no_text", 0) + 1

    issues = validate_document(storyboard, "storyboard")
    if issues:
        raise TextError("storyboard.json failed schema validation after the text stage: " + "; ".join(issues))

    storyboard_path.write_text(
        json.dumps(storyboard, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (ctx.dir / TEXT_JSON).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    logger.info(
        "text done",
        extra={"fields": {"episodes": len(storyboard.get("episodes") or []), "modes": mode_counts}},
    )
    return storyboard, report


def _episode_text(
    ctx: RunContext,
    episode: dict,
    segments: list[dict],
    sources: dict[str, str],
    output_lang_project: str,
    params: dict,
    llm,
    vlm,
    warnings: list[dict],
) -> dict:
    """Resolve, generate and record one episode's text (mutates the episode).

    ``llm`` and ``vlm`` are zero-arg service getters (or None when the
    section is not configured): the provider is built on first use only."""
    episode_id = str(episode.get("id") or "")
    role = str(episode.get("role") or "")
    limit = int(params["role_limits"].get(role, params["default_limit"]))
    existing_text = episode.get("text") or {}
    if episode.get("user_edited") and existing_text:
        return {"id": episode_id, "role": role, "kept": True, "text_mode": existing_text.get("mode")}

    t_start = float(episode.get("t_start") or 0.0)
    t_end = float(episode.get("t_end") or 0.0)
    span_segments = segments_in_span(segments, t_start, t_end)
    wanted_mode = resolve_mode(params["mode"], bool(span_segments))

    entry: dict = {"id": episode_id, "role": role, "limit": limit}
    if wanted_mode == "described":
        if vlm is None:
            warnings.append(
                {"episode_id": episode_id, "code": "no_vlm_provider",
                 "detail": "described mode needs a [vlm] provider; episode keeps no text"}
            )
            entry["skipped"] = True
            return entry
        frame = episode.get("frame") or {}
        frame_path = ctx.dir / str(frame.get("path") or "")
        if not frame_path.is_file():
            warnings.append(
                {"episode_id": episode_id, "code": "frame_missing",
                 "detail": f"frame file not found: {frame.get('path')}"}
            )
            entry["skipped"] = True
            return entry
        lang = output_lang_project or sources.get(str(episode.get("source_id")), "") or "ru"
        answer = vlm().describe_image(frame_path, describe_system_prompt(limit, lang))
        output = truncate_at_word(normalize_answer(answer), limit)
        if not output:
            warnings.append(
                {"episode_id": episode_id, "code": "empty_described_answer",
                 "detail": "VLM returned an empty description; episode keeps no text"}
            )
            entry["skipped"] = True
            return entry
        source_lang = sources.get(str(episode.get("source_id"))) or None
        episode["text"] = _build_text_block("described", "", output, source_lang, lang or None)
        entry.update({"text_mode": "described", "reason": "no speech in the episode span" if not span_segments else "mode = described",
                      "chars": {"output": len(output)}, "warnings": []})
        return entry

    original = " ".join(str(s["text"]) for s in span_segments).strip()
    if not original:
        # resolve_mode sends silent episodes to described; this branch is a
        # defensive stop for whitespace-only speech.
        warnings.append(
            {"episode_id": episode_id, "code": "empty_original",
             "detail": "transcript segments in the span carry no text; episode keeps no text"}
        )
        entry["skipped"] = True
        return entry
    source_lang = dominant_lang(span_segments)
    entry["chars"] = {"original": len(original)}

    # unit normalization (naryad 018) runs BEFORE any provider call: the
    # provider sees already-converted numbers, the number invariant compares
    # against the normalized original, conversions land in the report.
    original, unit_ops = normalize_units(original, params["units"])
    if unit_ops:
        entry["units_converted"] = unit_ops

    target_lang = params["output_lang"] or output_lang_project or ""
    needs_translation = bool(target_lang and source_lang and target_lang != source_lang)

    if wanted_mode == "verbatim":
        output = normalize_answer(original)
        entry.update({"text_mode": "verbatim", "chars": {"original": len(original), "output": len(output)}})
        if len(output) > limit:
            entry["over_limit"] = True
            warnings.append(
                {"episode_id": episode_id, "code": "verbatim_over_limit",
                 "detail": f"verbatim is {len(output)} chars, the layout limit is {limit}; the layout engine truncates"}
            )
    else:  # condensed
        fallback_reason: str | None = None
        if llm is None:
            fallback_reason = "no [llm] provider configured"
        else:
            answer = llm().complete([
                Message(role="system", content=condense_system_prompt(limit)),
                Message(role="user", content=original),
            ])
            output = normalize_answer(answer)
            if not output:
                fallback_reason = "LLM returned an empty answer"
            elif not numbers_preserved(original, output):
                fallback_reason = "LLM answer lost numbers of the original"
        if fallback_reason is not None:
            output = normalize_answer(original)
            entry.update({"text_mode": "verbatim", "fallback": fallback_reason})
            warnings.append(
                {"episode_id": episode_id, "code": "condensed_fallback_verbatim", "detail": fallback_reason}
            )
        else:
            if len(output) > limit:
                output = truncate_at_word(output, limit)
                warnings.append(
                    {"episode_id": episode_id, "code": "text_over_limit_truncated",
                     "detail": f"condensed answer was over the limit ({limit}), shortened at a word boundary"}
                )
            entry["text_mode"] = "condensed"
        entry["chars"]["output"] = len(output)
        if entry.get("text_mode") == "condensed" and not numbers_preserved(original, output):
            # the truncation itself cannot drop a number that sat before the cut,
            # but the check is cheap and the invariant is contractual
            entry["numbers_preserved"] = False
            warnings.append(
                {"episode_id": episode_id, "code": "numbers_lost",
                 "detail": "output does not carry every number of the original; gateway D will review"}
            )

    # translation (naryad 018): when the target output language differs from
    # the source language, the provider-translated output replaces the text;
    # a translation that loses numbers or comes back empty falls back to the
    # source-language text with a warning (the mode itself never changes:
    # verbatim/condensed describe how the text was derived, not its language).
    output_lang_used = source_lang
    if needs_translation:
        if llm is None:
            warnings.append(
                {"episode_id": episode_id, "code": "translate_no_llm",
                 "detail": f"output language {target_lang} differs from {source_lang}, but no [llm] provider is configured"}
            )
        else:
            translated = normalize_answer(llm().complete([
                Message(role="system", content=translate_system_prompt(target_lang)),
                Message(role="user", content=output),
            ]))
            if translated and numbers_preserved(output, translated):
                output = translated
                entry["chars"]["output"] = len(output)
                output_lang_used = target_lang
                entry["translated_to"] = target_lang
            else:
                warnings.append(
                    {"episode_id": episode_id, "code": "translate_fallback_source",
                     "detail": "translation lost numbers of the text" if translated else "translation came back empty"}
                )

    episode["text"] = _build_text_block(
        entry["text_mode"], original, output, source_lang, output_lang_used
    )
    return entry

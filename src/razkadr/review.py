"""Text review toolkit (naryad 020): fact checklist + comparison tool.

Gateway D checks text quality "on 6 videos", but the auditor and the owner
need a deterministic first pass: which quantities, times and temperatures of
the source text survived into the slide text, and which words may be lost
ingredient names. This module compares ``text.original`` with
``text.output`` of every episode of a Storyboard and produces a per-episode
checklist report; nothing here calls providers or rewrites documents.

Facts are extracted deterministically from both texts and compared as
multisets (counts matter: "2 eggs" and "2 tsp" are two distinct occurrences
of "2"):

- **numbers** - every digit token; the gateway D invariant "quantities,
  time, temperatures transfer from original to output unchanged" is exactly
  the "numbers: no lost" check;
- **temperatures** - "350 °F / 180 °C / 200 градусов" values with scale;
- **times** - "15 минут / 1,5 часа / 30 сек" values with normalized unit;
- **measures** - values with mass/volume units ("200 г", "1 cup", "2 ст. л.").

Words of the original that are absent from the output (not stop words, not
units) are listed as ``ingredient_candidates``: deterministic heuristics
cannot tell a lost ingredient from a paraphrase, so they are marked for
human review, never auto-failed.

Episode status: ``fail`` when a number/temperature/time/measure of the
original is missing from the output (the gateway D invariant broken);
``review`` when there are ingredient candidates or added numbers (a unit
conversion, a paraphrase rounding - a human decides); ``ok`` otherwise;
``n/a`` for episodes without an original (described mode has no source
text to compare against). CLI exit codes follow template-check: 0 - all
reviewable episodes ok, 1 - there is something to review, 2 - file error.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
TEMP_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*[°º]\s*([CFcf])|(\d+(?:[.,]\d+)?)\s*градус\w*", re.IGNORECASE)
TIME_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*"
    r"(час(?:ов|а|ам|ах|ами|ы)?|мин(?:ут(?:ы|у|е|ой|ах|ам|ами)?)?|сек(?:унд(?:ы|у|е|ой|ах|ам|ами)?)?|hours?|hr|h|minutes?|min|seconds?|sec|s)\b",
    re.IGNORECASE,
)
MEASURE_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*("
    r"ст\.?\s?л|ч\.?\s?л|"
    r"гр|кг|мг|мл|г|л|"
    r"cups|cup|tbsp|tsp|grams|gram|g|kgs|kg|mls|ml|liters|liter|lbs|lb|oz|l)\b",
    re.IGNORECASE,
)
WORD_RE = re.compile(r"[а-яёa-z]{3,}", re.IGNORECASE)

TIME_UNITS = {
    "час": "h", "часа": "h", "часов": "h", "часам": "h", "часами": "h", "часах": "h", "часы": "h",
    "мин": "min", "минут": "min", "минута": "min", "минуты": "min", "минуту": "min",
    "минуте": "min", "минутой": "min", "минутам": "min", "минутами": "min", "минутах": "min",
    "сек": "sec", "секунд": "sec", "секунда": "sec", "секунды": "sec", "секунду": "sec",
    "секунде": "sec", "секундой": "sec", "секундам": "sec", "секундами": "sec", "секундах": "sec",
    "h": "h", "hr": "h", "hour": "h", "hours": "h",
    "min": "min", "minute": "min", "minutes": "min",
    "sec": "sec", "second": "sec", "seconds": "sec", "s": "sec",
}
MEASURE_UNITS = {
    "г": "g", "гр": "g", "g": "g", "gram": "g", "grams": "g",
    "кг": "kg", "kg": "kg", "kgs": "kg", "мг": "mg",
    "мл": "ml", "ml": "ml", "mls": "ml",
    "л": "l", "l": "l", "liter": "l", "liters": "l",
    "стл": "tbsp", "tbsp": "tbsp", "чл": "tsp", "tsp": "tsp",
    "cup": "cup", "cups": "cup", "oz": "oz", "lb": "lb", "lbs": "lb",
}

# stop words are excluded from ingredient candidates; the lists are a
# heuristic seed (pronouns, prepositions, common cooking verbs), review
# quality is the auditor's call, not a contract number
STOP_WORDS_RU = {
        "для", "затем", "когда", "чтобы", "этого", "который", "которая", "очень", "более", "менее",
        "добавить", "добавьте", "добавляем", "положить", "положите", "смешать", "смешайте",
        "разогрейте", "разогреть", "варить", "варите", "жарить", "жарьте", "запекать", "запекайте",
        "готовьте", "готовить", "перемешайте", "перемешать", "нарезать", "нарежьте", "нагрейте",
        "пока", "также", "можно", "нужно", "надо", "после", "перед", "около", "примерно",
        "при", "как", "ещё", "еще", "нам", "уже", "всё", "все", "это", "тот", "там", "тогда",
        "разогреваем", "обжариваем", "перемешиваем", "наливаем", "выливаем", "посыпаем", "ставим",
        "убираем", "снимаем", "подаем", "подаём", "нарезаем", "заливаем", "посолите", "поперчите",
}
STOP_WORDS_EN = {
        "for", "then", "when", "that", "this", "with", "from", "into", "onto", "until", "about",
        "add", "adding", "mix", "mixing", "heat", "heating", "cook", "cooking", "bake", "baking",
        "fry", "frying", "boil", "boiling", "stir", "stirring", "cut", "chop", "chopping",
        "while", "also", "some", "need", "after", "before", "over", "under", "and", "the",
}
STOP_WORDS = STOP_WORDS_RU | STOP_WORDS_EN
UNIT_WORDS = set(TIME_UNITS) | set(MEASURE_UNITS) | {"cups", "градусов", "градуса", "градус", "минуты", "минут", "секунд"}

EPISODE_STATUSES = ("ok", "review", "fail", "n/a")


class ReviewError(Exception):
    """Review tool failure with a clear, user-facing message (CLI exit 2)."""


def _num(token: str) -> float:
    return float(token.replace(",", "."))


def _fmt(value: float) -> str:
    return f"{value:g}"


def extract_numbers(text: str) -> Counter:
    """Multiset of digit tokens as canonical strings ("0,5" == "0.5" -> "0.5")."""
    return Counter(_fmt(_num(t)) for t in NUMBER_RE.findall(text))


def extract_temperatures(text: str) -> Counter:
    """Multiset of "<value> <scale>" strings: "350 °F" -> "350 °F".

    A bare "градус" carries no scale in the text; the key keeps that
    spelling ("200 градус") so original and output must agree.
    """
    pairs: Counter = Counter()
    for match in TEMP_RE.finditer(text):
        if match.group(1):
            pairs[f"{_fmt(_num(match.group(1)))} °{match.group(2).upper()}"] += 1
        else:
            pairs[f"{_fmt(_num(match.group(3)))} градус"] += 1
    return pairs


def extract_times(text: str) -> Counter:
    """Multiset of "<value> <unit>" strings with normalized unit: "1,5 часа" -> "1.5 h"."""
    pairs: Counter = Counter()
    for match in TIME_RE.finditer(text):
        unit = TIME_UNITS.get(match.group(2).lower(), match.group(2).lower())
        pairs[f"{_fmt(_num(match.group(1)))} {unit}"] += 1
    return pairs


def _canon_measure(raw: str) -> str:
    """"ст. л" / "СТ.Л" / "стл" -> "tbsp": dots and spaces are spelling noise."""
    unit = raw.lower().replace(" ", "").replace(".", "")
    return MEASURE_UNITS.get(unit, unit)


def extract_measures(text: str) -> Counter:
    """Multiset of "<value> <unit>" strings with normalized unit: "200 г" -> "200 g"."""
    pairs: Counter = Counter()
    for match in MEASURE_RE.finditer(text):
        pairs[f"{_fmt(_num(match.group(1)))} {_canon_measure(match.group(2))}"] += 1
    return pairs


def _diff(original: Counter, output: Counter) -> dict:
    """lost: in original but not (or fewer) in output; added: the reverse."""
    lost = {key: original[key] - output[key] for key in original if output[key] < original[key]}
    added = {key: output[key] - original[key] for key in output if original[key] < output[key]}
    return {"lost": lost, "added": added}


def fact_diff(original: str, output: str) -> dict:
    """Checklist comparison of one episode: categories with lost/added facts."""
    return {
        "numbers": _diff(extract_numbers(original), extract_numbers(output)),
        "temperatures": _diff(extract_temperatures(original), extract_temperatures(output)),
        "times": _diff(extract_times(original), extract_times(output)),
        "measures": _diff(extract_measures(original), extract_measures(output)),
    }


def _stem(word: str) -> str:
    """Naive case ending strip (RU/EN plural), deterministic, no dictionaries.

    Enough to match "риса" against "рис", "муки" against "мука",
    "eggs" against "egg": candidates are for human review, and a word that
    only inflects differently must not be flagged as lost.
    """
    low = word.lower()
    if len(low) > 4:
        for suffix in ("ами", "ями", "ого", "его", "ому", "ему", "ыми", "ими", "ов",
                       "ев", "ам", "ям", "ах", "ях", "ой", "ей", "ый", "ий", "ая",
                       "ое", "ые", "ых", "ут", "ют", "ат", "ят", "es"):
            if low.endswith(suffix) and len(low) - len(suffix) >= 3:
                return low[: len(low) - len(suffix)]
    if len(low) > 3 and low.endswith(("а", "я", "ы", "и", "е", "у", "ю", "ь", "s")):
        return low[:-1]
    return low


def ingredient_candidates(original: str, output: str) -> list[str]:
    """Words of the original missing from the output: possible lost ingredients.

    Deterministic heuristic (no lemmatization, no dictionaries beyond stop
    words): a word is a candidate when neither its form nor its stem appears
    in the output. Candidates are marked for human review, never auto-failed.
    """
    low_output = output.lower()
    output_stems = {_stem(w) for w in WORD_RE.findall(output)}
    candidates: list[str] = []
    seen: set[str] = set()
    for word in WORD_RE.findall(original):
        low = word.lower()
        if low in seen or low in UNIT_WORDS or low in STOP_WORDS:
            continue
        seen.add(low)
        if low in low_output or _stem(low) in output_stems:
            continue
        candidates.append(word)
    return candidates


def episode_verdict(facts: dict, candidates: list[str]) -> str:
    """Episode status from the checklist: fail / review / ok."""
    if any(category["lost"] for category in facts.values()):
        return "fail"
    if candidates or any(category["added"] for category in facts.values()):
        return "review"
    return "ok"


def review_storyboard(storyboard: dict) -> dict:
    """Fact checklist over every episode with text; returns the report dict.

    Episodes without text (silent and no VLM) or without an original
    (described mode) are listed with status "n/a": there is nothing to
    compare. The summary counts reviewable episodes only.
    """
    episodes: list[dict] = []
    for episode in storyboard.get("episodes") or []:
        text = episode.get("text") or {}
        entry: dict = {
            "id": episode.get("id"),
            "role": episode.get("role"),
            "mode": text.get("mode"),
            "lang": {"source": text.get("source_lang"), "output": text.get("output_lang")},
        }
        original = str(text.get("original") or "")
        output = str(text.get("output") or "")
        if not text or not original.strip():
            entry.update({"status": "n/a", "reason": "no source text to compare (described or no text)"})
            episodes.append(entry)
            continue
        facts = fact_diff(original, output)
        candidates = ingredient_candidates(original, output)
        status = episode_verdict(facts, candidates)
        entry.update({
            "status": status,
            "facts": facts,
            "ingredient_candidates": candidates,
            "chars": {"original": len(original), "output": len(output)},
        })
        episodes.append(entry)
    reviewable = [e for e in episodes if e["status"] != "n/a"]
    summary = {
        "episodes": len(episodes),
        "reviewable": len(reviewable),
        "ok": sum(1 for e in reviewable if e["status"] == "ok"),
        "review": sum(1 for e in reviewable if e["status"] == "review"),
        "fail": sum(1 for e in reviewable if e["status"] == "fail"),
    }
    return {"episodes": episodes, "summary": summary}


def review_file(path: str | Path) -> tuple[dict, dict]:
    """Load a Storyboard file and review it; returns (report, storyboard)."""
    try:
        storyboard = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReviewError(f"file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ReviewError(f"not valid JSON: {path}: {exc}") from None
    if not isinstance(storyboard, dict) or "episodes" not in storyboard:
        raise ReviewError(f"not a Storyboard (no 'episodes'): {path}")
    return review_storyboard(storyboard), storyboard

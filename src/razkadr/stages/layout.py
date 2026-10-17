"""Layout engine (naryad 021, ADR-0003): Storyboard -> slides.json.

Turns episodes into format-neutral slides (slots + source_refs +
warnings); PPTX, HTML and PDF renderers build from that (naryads 022+).
The role-to-layout mapping is configuration, never hardcoded
(``[layout.role_layouts]`` in razkadr.toml; defaults below only seed it).
Text limits live in ``[layout]``. Overflow rules: lists and grids first
spill onto a new slide, then any remaining too-long text is shortened at
a sentence or word boundary with an ellipsis - every shortening is
recorded in the warnings of the slide that holds the text, nothing is
cut silently. Episodes without text produce slides with empty text
slots and no warnings.
"""

from __future__ import annotations

import json
from pathlib import Path

SLIDES_SCHEMA_VERSION = "1.0"
ELLIPSIS = "…"

DEFAULT_ROLE_LAYOUTS = {
    "hero": "hero_caption",
    "ingredients": "ingredients_list",
    "prep": "steps_grid",
    "steps": "steps_grid",
    "step": "steps_grid",  # the canonical example (examples/storyboard.*) uses the singular
    "serving": "serving",
}

DEFAULT_LIMITS = {
    "heading_max": 60,
    "subheading_max": 120,
    "caption_max": 180,
    "item_max": 40,
    "step_text_max": 120,
    "ingredients_max_items": 12,
    "steps_max_cells": 6,
}

CAPTIONS = {
    "ru": {"ingredients": "Ингредиенты", "steps": "Шаги", "serving": "Подача", "continued": "(продолжение)"},
    "en": {"ingredients": "Ingredients", "steps": "Steps", "serving": "Serving", "continued": "(continued)"},
}

KNOWN_LAYOUTS = ("title", "hero_caption", "ingredients_list", "steps_grid", "serving")


class LayoutError(Exception):
    """Layout cannot proceed (bad storyboard or bad [layout] config)."""


def timecode_s(seconds: float) -> str:
    """mm:ss (or h:mm:ss from one hour on), floor-rounded."""
    total = int(seconds)
    if total < 3600:
        return f"{total // 60}:{total % 60:02d}"
    return f"{total // 3600}:{total % 3600 // 60:02d}:{total % 60:02d}"


def _shorten(text: str, limit: int) -> str | None:
    """Cut at a sentence boundary, then a word boundary; None if it fits."""
    if len(text) <= limit:
        return None
    head = text[:limit]
    dot = head.rfind(". ")
    if dot >= limit // 2:
        return text[: dot + 1]
    space = head.rfind(" ")
    if space >= limit // 2:
        return text[:space].rstrip(" ,;:-") + ELLIPSIS
    return text[:limit].rstrip(" ,;:-") + ELLIPSIS


def _truncate(text: str, limit: int, episode_id: str, slot: str, warnings: list[dict]) -> str:
    if not text:
        return ""
    cut = _shorten(text, limit)
    if cut is None:
        return text
    warnings.append({"code": "text_truncated", "episode_id": episode_id, "slot": slot, "limit": limit})
    return cut


def _split_items(text: str) -> list[str]:
    """Episode text -> ingredient rows: lines first, then comma groups."""
    items: list[str] = []
    for line in text.splitlines():
        line = line.strip().strip("•-–— ")
        if not line:
            continue
        if "," in line:
            items.extend(part.strip() for part in line.split(","))
        else:
            items.append(line)
    return [item for item in (i.strip() for i in items) if item]


def _episode_text(episode: dict) -> str:
    text = episode.get("text") or {}
    return str(text.get("output") or "").strip()


def _frame_slot(episode: dict) -> dict:
    frame = episode.get("frame") or {}
    return {"path": frame.get("path", ""), "t": frame.get("t", 0), "episode_id": episode["id"]}


def _config_values(config: dict) -> tuple[dict, dict]:
    section = config.get("layout", {}) if isinstance(config, dict) else {}
    limits = dict(DEFAULT_LIMITS)
    for key in DEFAULT_LIMITS:
        if key in section:
            value = section[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise LayoutError(f"[layout] {key} must be a positive integer, got {value!r}")
            limits[key] = int(value)
    role_layouts = dict(DEFAULT_ROLE_LAYOUTS)
    for role, layout in (section.get("role_layouts") or {}).items():
        if layout not in KNOWN_LAYOUTS:
            raise LayoutError(
                f"[layout] role_layouts.{role} = {layout!r} is not a layout (known: {list(KNOWN_LAYOUTS)})"
            )
        role_layouts[str(role)] = layout
    return limits, role_layouts


def build_slides(storyboard: dict, config: dict, lang: str | None = None) -> dict:
    """Pure, deterministic Storyboard -> slides document (schema slides.v1)."""
    if "episodes" not in storyboard or "project" not in storyboard:
        raise LayoutError("not a Storyboard: 'project' or 'episodes' missing")
    limits, role_layouts = _config_values(config)
    project = storyboard["project"]
    captions_lang = (
        lang or (project.get("output_lang") if project.get("output_lang") in CAPTIONS else None) or "ru"
    )
    captions = CAPTIONS[captions_lang]
    doc_warnings: list[dict] = []
    slides: list[dict] = []

    # consecutive same-role episodes form groups; each group maps to slides
    groups: list[tuple[str, list[dict]]] = []
    for episode in storyboard["episodes"]:
        role = episode["role"]
        if groups and groups[-1][0] == role:
            groups[-1][1].append(episode)
        else:
            groups.append((role, [episode]))

    hero_group = next((eps for role, eps in groups if role_layouts.get(role) == "hero_caption"), None)

    # title slide: project heading, hero text as subheading, hero frame as background
    title_warnings: list[dict] = []
    title_slots: dict = {
        "heading": _truncate(
            str(project.get("title") or ""), limits["heading_max"], "project", "heading", title_warnings
        ),
        "subheading": "",
    }
    if hero_group:
        hero = hero_group[0]
        title_slots["subheading"] = _truncate(
            _episode_text(hero), limits["subheading_max"], hero["id"], "subheading", title_warnings
        )
        title_slots["frame"] = _frame_slot(hero)
    slides.append(
        {"layout": "title", "slots": title_slots, "source_refs": ["project"], "warnings": title_warnings}
    )

    step_number = 0
    for role, eps in groups:
        layout = role_layouts.get(role)
        if layout is None:
            doc_warnings.append({
                "code": "unmapped_role",
                "detail": f"role {role!r} has no layout mapping; {len(eps)} episode(s) skipped",
            })
            continue

        if layout == "hero_caption":
            for episode in eps:  # the first one also feeds the title slide
                warnings: list[dict] = []
                caption = _truncate(
                    _episode_text(episode), limits["caption_max"], episode["id"], "caption", warnings
                )
                slides.append({
                    "layout": "hero_caption",
                    "slots": {
                        "frame": _frame_slot(episode),
                        "caption": caption,
                        "timecode": timecode_s(episode["t_start"]),
                    },
                    "source_refs": [episode["id"]],
                    "warnings": warnings,
                })

        elif layout == "ingredients_list":
            # (item, warning) pairs: a truncated item carries its own warning
            pairs: list[tuple[str, list[dict]]] = []
            for episode in eps:
                for item in _split_items(_episode_text(episode)):
                    item_warnings: list[dict] = []
                    pairs.append((
                        _truncate(item, limits["item_max"], episode["id"], "items", item_warnings),
                        item_warnings,
                    ))
            max_items = limits["ingredients_max_items"]
            for start in range(0, len(pairs), max_items):
                chunk = pairs[start:start + max_items]
                warnings = [w for _, ws in chunk for w in ws]
                slides.append({
                    "layout": "ingredients_list",
                    "slots": {
                        "frame": _frame_slot(eps[0]),
                        "heading": captions["ingredients"] + (f" {captions['continued']}" if start else ""),
                        "items": [item for item, _ in chunk],
                    },
                    "source_refs": [episode["id"] for episode in eps],
                    "warnings": warnings,
                })

        elif layout == "steps_grid":
            cells: list[dict] = []
            cell_warnings: dict[int, list[dict]] = {}
            for episode in eps:
                step_number += 1
                warnings = []
                cells.append({
                    "frame": _frame_slot(episode),
                    "number": step_number,
                    "text": _truncate(
                        _episode_text(episode), limits["step_text_max"], episode["id"], "cells", warnings
                    ),
                    "timecode": timecode_s(episode["t_start"]),
                })
                if warnings:
                    cell_warnings[len(cells) - 1] = warnings
            max_cells = limits["steps_max_cells"]
            for start in range(0, len(cells), max_cells):
                chunk = cells[start:start + max_cells]
                warnings = [w for i, ws in cell_warnings.items() if start <= i < start + len(chunk) for w in ws]
                slides.append({
                    "layout": "steps_grid",
                    "slots": {
                        "columns": 2 if len(chunk) <= 4 else 3,
                        "heading": captions["steps"] + (f" {captions['continued']}" if start else ""),
                        "cells": chunk,
                    },
                    "source_refs": [c["frame"]["episode_id"] for c in chunk],
                    "warnings": warnings,
                })

        elif layout == "serving":
            for episode in eps:
                warnings = []
                caption = _truncate(
                    _episode_text(episode), limits["caption_max"], episode["id"], "caption", warnings
                )
                slides.append({
                    "layout": "serving",
                    "slots": {
                        "frame": _frame_slot(episode),
                        "caption": caption,
                        "timecode": timecode_s(episode["t_start"]),
                        "heading": captions["serving"],
                    },
                    "source_refs": [episode["id"]],
                    "warnings": warnings,
                })

        else:  # defensive: _config_values already validates the mapping
            raise LayoutError(f"role {role!r} maps to unknown layout {layout!r}")

    return {
        "schema_version": SLIDES_SCHEMA_VERSION,
        "lang": captions_lang,
        "warnings": doc_warnings,
        "slides": slides,
    }


def layout_storyboard(storyboard: dict, config: dict, lang: str | None = None) -> dict:
    """Public entry: build + validate against slides.v1.json."""
    from razkadr.validation import validate_document

    result = build_slides(storyboard, config, lang)
    issues = validate_document(result, "slides")
    if issues:
        detail = "; ".join(f"{issue.path}: {issue.reason}" for issue in issues[:5])
        raise LayoutError(f"internal error: generated slides failed the schema: {detail}")
    return result


def write_slides(slides: dict, out_path: Path) -> None:
    """Deterministic serialization: fixed key order, 2-space indent."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(slides, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

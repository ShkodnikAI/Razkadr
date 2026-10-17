"""PPTX render: slides.json -> deck.pptx on the reference template (naryad 022).

PPTX is the master format (ADR-0003): the renderer fills the template's
named placeholders with the slides.json data, places frames inside their
placeholder boxes with proportions preserved (contain-fit, no crop and no
distortion), auto-fits text within slots down to a floor size (below the
floor it warns instead of shrinking further) and writes speaker notes with
the episode id, start/end timecodes and the source - so every slide points
back to the video moment it came from.

Text metrics: word wrap is simulated with Pillow against the DejaVu Sans
metrics (the template font) at 72 dpi, so 1 pt == 1 px; line height is
estimated at 1.25 x font size (DejaVu is ~1.17, PowerPoint single spacing
~1.2 - the margin keeps the estimate conservative). The same functions are
used by the tests to re-check that no text exceeds its slot.

Determinism: the renderer never embeds timestamps, so two runs on the same
inputs produce content-identical decks (zip entry timestamps differ; parts
do not - same convention as the template build).
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import ImageFont
from pptx import Presentation
from pptx.util import Emu, Pt

TEMPLATE_RELPATH = "templates/razkadr-default.pptx"
LINE_HEIGHT = 1.25  # conservative single-spacing estimate
INSET_SIDE_IN = 0.06  # template bodyPr insets (scripts/templates/build_default.py)
INSET_TB_IN = 0.03
BULLET_PREFIX = "• "

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu-sans/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
)
_FONT_BOLD_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu-sans/DejaVuSans-Bold.ttf",
)

# start font sizes (pt) per slot kind when the template placeholder does not
# declare one in its lstStyle
START_SIZES = {
    "heading": 28,
    "subheading": 20,
    "caption": 20,
    "items": 16,
    "cell_text": 13,
    "cell_number": 13,
    "timecode": 14,
    "cell_timecode": 11,
}


class RenderError(Exception):
    """Render cannot proceed (missing template, unknown layout)."""


# --- text metrics (shared with the tests) ---------------------------------------


def _font(size_pt: int, bold: bool = False):
    """DejaVu Sans at ``size_pt`` px (1 pt == 1 px at 72 dpi). Falls back to
    a character-count estimate when the font file is unavailable."""
    candidates = _FONT_BOLD_CANDIDATES if bold else _FONT_CANDIDATES
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size_pt)
    return None


def text_width_pt(text: str, size_pt: float, bold: bool = False) -> float:
    """Width of a single line in points."""
    font = _font(round(size_pt), bold)
    if font is None:
        return len(text) * size_pt * 0.62
    return font.getlength(text)


def wrap_lines(text: str, width_pt: float, size_pt: float, bold: bool = False) -> list[str]:
    """Greedy word wrap; long words hard-split at the character level."""
    lines: list[str] = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if text_width_pt(candidate, size_pt, bold) <= width_pt:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def fit_size(
    texts: list[str],
    width_in: float,
    height_in: float,
    start_sz: int,
    floor_sz: int,
    bold: bool = False,
) -> tuple[int, bool]:
    """Largest size in ``[floor_sz, start_sz]`` at which all ``texts``
    (separate paragraphs) wrap into the box. Returns ``(size, fits)``;
    ``fits`` is False when even the floor size overflows."""
    w_pt = width_in * 72 - INSET_SIDE_IN * 72 * 2
    h_pt = height_in * 72 - INSET_TB_IN * 72 * 2
    size = start_sz
    while size > floor_sz:
        total = sum(len(wrap_lines(t, w_pt, size, bold)) for t in texts if t)
        if total * size * LINE_HEIGHT <= h_pt:
            return size, True
        size -= 1
    total = sum(len(wrap_lines(t, w_pt, floor_sz, bold)) for t in texts if t)
    return floor_sz, total * floor_sz * LINE_HEIGHT <= h_pt


# --- helpers ---------------------------------------------------------------------


def resolve_template(template: str | Path | None) -> Path:
    """--template path, or the repo default (CWD then source tree)."""
    if template is not None:
        path = Path(template)
        if not path.is_file():
            raise RenderError(f"template not found: {path}")
        return path
    candidates = [Path.cwd() / TEMPLATE_RELPATH]
    # source checkout (also editable installs): src/razkadr/render/pptx.py
    candidates.append(Path(__file__).resolve().parents[3] / TEMPLATE_RELPATH)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RenderError(
        f"default template not found ({TEMPLATE_RELPATH}); pass --template"
    )


def _layout_by_name(prs: Presentation, name: str):
    for layout in prs.slide_masters[0].slide_layouts:
        if layout.name == name:
            return layout
    raise RenderError(f"template has no layout '{name}'")


def _slot_kind(name: str) -> str:
    """Slot kind for START_SIZES lookup: cellN_text -> cell_text."""
    if name.startswith("cell") and "_" in name:
        return "cell_" + name.split("_", 1)[1]
    return name


def _make_inheritance_explicit(slide_ph, layout_ph) -> None:
    """Copy the layout placeholder's geometry (xfrm) into the cloned slide
    placeholder.

    Clones carry no xfrm, and renderers resolve position through
    placeholder inheritance; copying removes any dependence on how a
    particular renderer matches ph type/idx up the chain. List styles are
    synced separately in ``_fill_text`` - python-pptx ``clear()`` inserts
    a fresh empty <a:lstStyle/>, which would end up as a second, winning
    element if the style was copied before filling."""
    from copy import deepcopy

    from pptx.oxml.ns import qn

    xfrm = layout_ph._element.spPr.find(qn("a:xfrm"))
    if xfrm is not None:
        slide_ph._element.spPr.append(deepcopy(xfrm))


def _slot_shapes(slide, layout) -> tuple[dict, dict[str, int]]:
    """Map template slot names to the cloned slide placeholders.

    ``add_slide`` rebuilds placeholder clones with generic names, so the
    name->shape mapping goes through the layout: the layout placeholder's
    name and ph idx give the slide placeholder (clones keep the ph idx).
    Clones are renamed to the slot names and get explicit geometry+style
    (see ``_make_inheritance_explicit``); the declared start font size
    comes from the layout placeholder."""
    by_idx = {}
    for shape in slide.shapes:
        if shape.is_placeholder:
            by_idx[shape.placeholder_format.idx] = shape
    shapes, start_sizes, styles = {}, {}, {}
    for layout_ph in layout.placeholders:
        name = layout_ph.name
        slide_ph = by_idx.get(layout_ph.placeholder_format.idx)
        if slide_ph is None:
            continue
        _make_inheritance_explicit(slide_ph, layout_ph)
        slide_ph.name = name
        shapes[name] = slide_ph
        layout_tx = layout_ph._element.txBody
        if layout_tx is not None:
            start_sizes[name] = _start_size(layout_ph, _slot_kind(name))
            from copy import deepcopy

            from pptx.oxml.ns import qn

            lst = layout_tx.find(qn("a:lstStyle"))
            body = layout_tx.find(qn("a:bodyPr"))
            styles[name] = (
                deepcopy(lst) if lst is not None else None,
                deepcopy(body) if body is not None else None,
            )
    return shapes, start_sizes, styles


def _start_size(layout_placeholder, kind: str) -> int:
    """Declared lstStyle size of the layout placeholder, else START_SIZES."""
    try:
        lst_style = layout_placeholder.text_frame._txBody.find_qn("a:lstStyle")
        lvl1 = lst_style.find_qn("a:lvl1pPr")
        def_rpr = lvl1.find_qn("a:defRPr")
        sz = def_rpr.get("sz")
        if sz:
            return max(6, int(sz) // 100)
    except (AttributeError, ValueError):
        pass
    return START_SIZES.get(kind, 14)


def _sync_style(placeholder, style) -> None:
    """Make the placeholder's bodyPr and lstStyle exactly the template's.

    Runs after ``text_frame.clear()`` because clear() inserts a fresh
    empty <a:lstStyle/>; leaving it in place would override the copied
    style (two lstStyle elements are also a schema violation)."""
    from copy import deepcopy

    from pptx.oxml.ns import qn

    if style is None:
        return
    lst_el, body_el = style
    tx_body = placeholder._element.txBody
    if tx_body is None:
        return
    for stale in tx_body.findall(qn("a:lstStyle")):
        tx_body.remove(stale)
    body_pr = tx_body.find(qn("a:bodyPr"))
    if body_el is not None and body_pr is not None:
        body_pr.getparent().replace(body_pr, deepcopy(body_el))
    body_pr = tx_body.find(qn("a:bodyPr"))
    if lst_el is not None and body_pr is not None:
        body_pr.addnext(deepcopy(lst_el))


def _fill_text(
    shapes,
    name: str,
    text: str,
    *,
    kind: str,
    start_sz: int,
    min_sz: int,
    slide_no: int,
    warnings: list[str],
    bullet: bool = False,
    bold: bool = False,
    style=None,
) -> None:
    """Fill one text placeholder and auto-fit its size within the slot."""
    placeholder = shapes.get(name)
    if placeholder is None:
        warnings.append(f"placeholder missing: {name} (slide {slide_no})")
        return
    lines = text.split("\n") if text else []
    if bullet:
        lines = [f"{BULLET_PREFIX}{line}" for line in lines]
    frame = placeholder.text_frame
    frame.clear()
    _sync_style(placeholder, style)
    if lines:
        size, fits = fit_size(
            lines,
            placeholder.width / 914400,
            placeholder.height / 914400,
            start_sz,
            min_sz,
            bold=bold,
        )
        if not fits:
            warnings.append(
                f"text_overflow: {name} on slide {slide_no} stays at the {min_sz}pt floor"
            )
        for i, line in enumerate(lines):
            paragraph = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
            run = paragraph.add_run()
            run.text = line
            run.font.size = Pt(size)


def _add_picture_contain(shapes, image: Path, box) -> int:
    """Insert the frame image inside ``box`` (l, t, w, h EMU), proportions
    preserved, centred; returns the picture count delta (0 or 1)."""
    left, top, width, height = box
    from PIL import Image

    with Image.open(image) as img:
        iw, ih = img.size
    scale = min(width / iw, height / ih)
    w, h = int(iw * scale), int(ih * scale)
    pic = shapes.add_picture(
        str(image), Emu(int(left + (width - w) / 2)), Emu(int(top + (height - h) / 2)), Emu(w), Emu(h)
    )
    return pic


def _drop(placeholder) -> None:
    placeholder._element.getparent().remove(placeholder._element)


def _place_frame(
    slide,
    shapes,
    name: str,
    frame_slot: dict,
    box: tuple,
    base_dir: Path | None,
    slide_no: int,
    warnings: list[str],
) -> bool:
    """Replace an empty picture placeholder with the real frame image."""
    placeholder = shapes.get(name)
    if placeholder is not None:
        _drop(placeholder)
    path = frame_slot.get("path", "")
    image = (base_dir / path) if base_dir else Path(path)
    if not image.is_file():
        warnings.append(f"frame not found: {path} (slide {slide_no}, slot {name})")
        return False
    pic = _add_picture_contain(slide.shapes, image, box)
    pic._element.nvPicPr.cNvPr.set(
        "descr", f"{frame_slot.get('episode_id', '')} t={frame_slot.get('t', 0)}"
    )
    return True


def _fmt_timecode(seconds: float) -> str:
    total = max(0, int(seconds))
    if total < 3600:
        return f"{total // 60}:{total % 60:02d}"
    return f"{total // 3600}:{total % 3600 // 60:02d}:{total % 60:02d}"


def _notes_text(slide_doc: dict, storyboard: dict | None) -> str:
    """One line per source_ref: id, start-end, source (url or path)."""
    episodes = {}
    sources = {}
    if storyboard:
        episodes = {e["id"]: e for e in storyboard.get("episodes", [])}
        sources = {s["id"]: s for s in storyboard.get("sources", [])}
    lines = []
    slots = slide_doc["slots"]
    for ref in slide_doc["source_refs"]:
        episode = episodes.get(ref)
        if episode is not None:
            source = sources.get(episode.get("source_id", ""), {})
            where = source.get("url") or source.get("path") or "-"
            lines.append(
                f"{ref} {_fmt_timecode(episode['t_start'])}-{_fmt_timecode(episode['t_end'])} {where}"
            )
            continue
        # graceful degradation without an episode record: the slot timecode
        # when this slide's frame belongs to the ref, else the bare id
        fallback = ""
        if slots.get("frame", {}).get("episode_id") == ref:
            fallback = slots.get("timecode") or _fmt_timecode(slots["frame"].get("t", 0))
        lines.append(f"{ref} {fallback}".strip())
    return "\n".join(lines)


# --- layout fillers ----------------------------------------------------------------


def _fill_simple(slide, slide_doc, shapes, start_sizes, styles, min_sz, slide_no, warnings, base_dir):
    """title / hero_caption / ingredients_list / serving share the same
    slot semantics: optional heading/subheading/caption/timecode/items and
    a single frame."""
    slots = slide_doc["slots"]
    mapping = (
        ("heading", "heading", False),
        ("subheading", "subheading", False),
        ("caption", "caption", False),
        ("timecode", "timecode", True),
    )
    for slot, kind, bold in mapping:
        if slot in slots:
            _fill_text(
                shapes, kind, str(slots[slot]), kind=kind,
                start_sz=start_sizes.get(kind, START_SIZES.get(kind, 14)),
                min_sz=min_sz, slide_no=slide_no, warnings=warnings, bold=bold,
                style=styles.get(kind),
            )
    if "items" in slots:
        items = [str(i) for i in slots["items"]]
        _fill_text(
            shapes, "items", "\n".join(items), kind="items",
            start_sz=start_sizes.get("items", START_SIZES["items"]),
            min_sz=min_sz, slide_no=slide_no, warnings=warnings, bullet=True,
            style=styles.get("items"),
        )
    if "frame" in slots:
        placeholder = shapes.get("frame")
        if placeholder is not None:
            box = (placeholder.left, placeholder.top, placeholder.width, placeholder.height)
            ok = _place_frame(
                slide, shapes, "frame", slots["frame"], box, base_dir, slide_no, warnings
            )
            if ok:
                return 1
    return 0


def _fill_steps_grid(slide, slide_doc, shapes, start_sizes, styles, min_sz, slide_no, warnings, base_dir):
    """Reposition the used cell groups per slides.json geometry
    (columns=2 -> 2x2 grid, columns=3 -> 3x2), drop unused ones."""
    slots = slide_doc["slots"]
    heading = slots.get("heading")
    if heading:
        _fill_text(
            shapes, "heading", str(heading), kind="heading",
            start_sz=start_sizes.get("heading", START_SIZES["heading"]),
            min_sz=min_sz, slide_no=slide_no, warnings=warnings, bold=True,
            style=styles.get("heading"),
        )
    columns = slots.get("columns", 2)
    cells = slots.get("cells", [])
    cols = 2 if columns == 2 else 3
    if len(cells) > cols * 2:
        warnings.append(
            f"cells_overflow: {len(cells)} cells for columns={columns} on slide {slide_no}"
        )

    # grid area: below the heading, to the slide margins
    heading_ph = shapes.get("heading")
    area_top = ((heading_ph.top + heading_ph.height) if heading_ph is not None else 914400) + 114300
    area_left, area_right = 457200, 12192000 - 457200
    area_bottom = 6858000 - 320040
    gap_x, gap_y = 219456, 182880
    cw = (area_right - area_left - (cols - 1) * gap_x) / cols
    ch = (area_bottom - area_top - gap_y) / 2

    pictures = 0
    for i in range(1, 7):
        names = (f"cell{i}_number", f"cell{i}_timecode", f"cell{i}_frame", f"cell{i}_text")
        cell_index = i - 1
        if cell_index >= len(cells):
            for name in names:  # unused group: drop the cloned placeholders
                placeholder = shapes.get(name)
                if placeholder is not None:
                    _drop(placeholder)
            continue
        cell = cells[cell_index]
        col, row = cell_index % cols, cell_index // cols
        cx = area_left + col * (cw + gap_x)
        cy = area_top + row * (ch + gap_y)
        badge_h, frame_w, pad = 347472, 0.46, 109728

        number = shapes.get(names[0])
        if number is not None:
            number.left, number.top = Emu(int(cx)), Emu(int(cy))
            number.width, number.height = Emu(502920), Emu(badge_h)
            _fill_text(
                shapes, names[0], str(cell["number"]), kind="cell_number",
                start_sz=start_sizes.get(names[0], START_SIZES["cell_number"]),
                min_sz=min_sz, slide_no=slide_no, warnings=warnings, bold=True,
                style=styles.get(names[0]),
            )
        timecode = shapes.get(names[1])
        if timecode is not None:
            timecode.left, timecode.top = Emu(int(cx + cw - 1188720)), Emu(int(cy))
            timecode.width, timecode.height = Emu(1188720), Emu(badge_h)
            _fill_text(
                shapes, names[1], str(cell["timecode"]), kind="cell_timecode",
                start_sz=start_sizes.get(names[1], START_SIZES["cell_timecode"]),
                min_sz=min_sz, slide_no=slide_no, warnings=warnings, bold=True,
                style=styles.get(names[1]),
            )
        frame = shapes.get(names[2])
        frame_box = (
            int(cx), int(cy + badge_h),
            int(cw * frame_w), int(ch - badge_h - pad),
        )
        if frame is not None:
            frame.left, frame.top = Emu(frame_box[0]), Emu(frame_box[1])
            frame.width, frame.height = Emu(frame_box[2]), Emu(frame_box[3])
            ok = _place_frame(
                slide, shapes, names[2], cell["frame"], frame_box, base_dir, slide_no, warnings
            )
            if ok:
                pictures += 1
        text = shapes.get(names[3])
        if text is not None:
            text.left = Emu(int(cx + cw * frame_w + pad))
            text.top = Emu(int(cy + badge_h))
            text.width = Emu(int(cw * (1 - frame_w) - pad))
            text.height = Emu(int(ch - badge_h - pad))
            _fill_text(
                shapes, names[3], str(cell["text"]), kind="cell_text",
                start_sz=start_sizes.get(names[3], START_SIZES["cell_text"]),
                min_sz=min_sz, slide_no=slide_no, warnings=warnings,
                style=styles.get(names[3]),
            )
    return pictures


# --- entry point --------------------------------------------------------------------


def render_pptx(
    slides_doc: dict,
    out: Path,
    template: str | Path | None = None,
    storyboard: dict | None = None,
    base_dir: Path | None = None,
    min_font_pt: int = 12,
) -> dict:
    """Render ``slides_doc`` into ``out`` on the template. Returns stats:
    ``{"slides", "pictures", "warnings"}``. Warnings are also printed to
    stderr (frame not found, text at the floor, missing placeholders)."""
    template_path = resolve_template(template)
    prs = Presentation(str(template_path))
    stats = {"slides": 0, "pictures": 0, "warnings": []}
    for slide_no, slide_doc in enumerate(slides_doc["slides"], 1):
        layout = _layout_by_name(prs, slide_doc["layout"])
        slide = prs.slides.add_slide(layout)
        shapes, start_sizes, styles = _slot_shapes(slide, layout)
        if slide_doc["layout"] == "steps_grid":
            pictures = _fill_steps_grid(
                slide, slide_doc, shapes, start_sizes, styles, min_font_pt, slide_no, stats["warnings"], base_dir
            )
        else:
            pictures = _fill_simple(
                slide, slide_doc, shapes, start_sizes, styles, min_font_pt, slide_no, stats["warnings"], base_dir
            )
        stats["pictures"] += pictures
        slide.notes_slide.notes_text_frame.text = _notes_text(slide_doc, storyboard)
        stats["slides"] += 1
    prs.save(str(out))
    for warning in stats["warnings"]:
        print(f"razkadr render: {warning}", file=sys.stderr)
    return stats

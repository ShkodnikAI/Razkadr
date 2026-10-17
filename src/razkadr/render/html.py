"""Self-contained HTML export of a slides document (naryad 026).

One page, no CDN and no external requests: CSS and JS are inline, frame
images are copied into ``deck_assets/`` next to the HTML. Every timecode
becomes a link to the source moment when the source is known (the
Storyboards sources are joined by episode id):

- YouTube (watch/youtu.be) -> ``...&t=<seconds>s`` time parameter
- TikTok and other URLs -> the URL as is (no reliable time parameter),
  the timecode stays as text on the slide
- local file -> ``file://`` URI with a ``#t=<seconds>`` media fragment
  (absolute path; portability is not guaranteed - naryad assumption)
- unknown/absent source -> timecode as plain text

Keyboard and click navigation, print layout (@media print) and a
dark/light theme (prefers-color-scheme plus a toggle) are built in.
"""

from __future__ import annotations

import hashlib
import html
import shutil  # noqa: F401 (kept for future stream-copy of large assets)
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

from razkadr.atomicio import write_json_atomic, write_text_atomic

ASSETS_DIRNAME = "deck_assets"
YOUTUBE_HOSTS = ("youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be")


class ExportError(Exception):
    """Export cannot proceed (bad inputs)."""


def link_moment(source: dict, seconds: float) -> str | None:
    """URL for the source moment per the naryad rules; None = no link."""
    url = (source.get("url") or "").strip()
    if url:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        seconds = max(0, int(seconds))
        if host == "youtu.be":
            return f"https://youtu.be{parsed.path}?t={seconds}s"
        if host in YOUTUBE_HOSTS:
            query = parse_qs(parsed.query)
            query.pop("t", None)
            query.pop("start", None)
            query["t"] = [f"{seconds}s"]
            return parsed._replace(query=urlencode(query, doseq=True)).geturl()
        return url  # tiktok and everything else: no reliable time parameter
    path = source.get("path")
    if path:
        absolute = Path(path).expanduser().resolve()
        return f"file://{absolute}#t={max(0, int(seconds))}"
    return None


def _timecode_html(tc: str, href: str | None) -> str:
    label = html.escape(tc)
    if href:
        return f'<a class="tc" href="{html.escape(href, quote=True)}" target="_blank" rel="noopener">{label}</a>'
    return f'<span class="tc">{label}</span>'


def _frame_img(slot: dict, asset_names: dict[str, str]) -> str:
    path = slot.get("path", "")
    name = asset_names.get(path)
    if name:
        return f'<img class="frame" src="{ASSETS_DIRNAME}/{html.escape(name)}" alt="кадр">'
    return '<div class="frame missing">кадр недоступен</div>'


def _render_slide(index: int, total: int, slide: dict, links: dict[str, str | None], asset_names: dict[str, str]) -> str:
    layout = html.escape(slide["layout"])
    slots = slide["slots"]
    parts: list[str] = [f'<section class="slide {layout}" data-index="{index}">']
    if layout == "title":
        parts.append('<div class="title-block">')
        parts.append(f'<h1>{html.escape(slots.get("heading", ""))}</h1>')
        if slots.get("subheading"):
            parts.append(f'<p class="subheading">{html.escape(slots["subheading"])}</p>')
        parts.append("</div>")
        if "frame" in slots:
            parts.append(_frame_img(slots["frame"], asset_names))
    elif layout == "hero_caption":
        parts.append(_frame_img(slots["frame"], asset_names))
        parts.append(f'<p class="caption">{html.escape(slots.get("caption", ""))}</p>')
        parts.append(_timecode_html(slots.get("timecode", ""), links.get(slots["frame"]["episode_id"])))
    elif layout == "ingredients_list":
        parts.append(f'<h2>{html.escape(slots.get("heading", ""))}</h2>')
        parts.append('<div class="split">')
        parts.append(_frame_img(slots["frame"], asset_names))
        parts.append('<ul class="items">')
        for item in slots.get("items", []):
            parts.append(f"<li>{html.escape(item)}</li>")
        parts.append("</ul></div>")
    elif layout == "steps_grid":
        parts.append(f'<h2>{html.escape(slots.get("heading", ""))}</h2>')
        columns = slots.get("columns", 2)
        parts.append(f'<div class="grid columns-{columns}">')
        for cell in slots.get("cells", []):
            parts.append('<div class="cell">')
            parts.append(f'<span class="number">{cell["number"]}</span>')
            parts.append(_frame_img(cell["frame"], asset_names))
            parts.append(f'<p class="cell-text">{html.escape(cell.get("text", ""))}</p>')
            parts.append(_timecode_html(cell.get("timecode", ""), links.get(cell["frame"]["episode_id"])))
            parts.append("</div>")
        parts.append("</div>")
    elif layout == "serving":
        parts.append(f'<h2>{html.escape(slots.get("heading", ""))}</h2>')
        parts.append(_frame_img(slots["frame"], asset_names))
        parts.append(f'<p class="caption">{html.escape(slots.get("caption", ""))}</p>')
        parts.append(_timecode_html(slots.get("timecode", ""), links.get(slots["frame"]["episode_id"])))
    warnings = slide.get("warnings") or []
    if warnings:
        notes = "; ".join(
            w.get("code", "") + (f" ({w.get('episode_id')})" if w.get("episode_id") else "") for w in warnings
        )
        parts.append(f'<div class="warnings" title="{html.escape(notes)}">⚠</div>')
    parts.append(f'<div class="pager">{index + 1} / {total}</div>')
    parts.append("</section>")
    return "\n".join(parts)


CSS = """
:root { --bg:#fafafa; --fg:#1a1a1a; --muted:#666; --card:#fff; --accent:#0a7; }
@media (prefers-color-scheme: dark) { :root { --bg:#111; --fg:#eee; --muted:#999; --card:#1b1b1b; --accent:#0f8; } }
html.light { --bg:#fafafa; --fg:#1a1a1a; --muted:#666; --card:#fff; --accent:#0a7; }
html.dark { --bg:#111; --fg:#eee; --muted:#999; --card:#1b1b1b; --accent:#0f8; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:16px/1.5 system-ui, sans-serif; }
main { display:flex; flex-direction:column; align-items:center; }
.slide { width:min(960px, 96vw); aspect-ratio:16/9; background:var(--card); margin:16px 0; padding:28px;
  border-radius:12px; display:none; flex-direction:column; gap:12px; overflow:hidden; position:relative; }
.slide.active { display:flex; }
.slide img.frame, .slide .frame { width:100%; border-radius:8px; object-fit:cover; }
.frame.missing { background:var(--bg); color:var(--muted); display:flex; align-items:center; justify-content:center; min-height:120px; }
h1 { font-size:2em; margin:0 0 8px; } h2 { font-size:1.3em; margin:0; }
.subheading, .caption, .cell-text { margin:0; }
.split { display:flex; gap:16px; align-items:flex-start; } .split img.frame { width:45%; }
.items { margin:0; padding-left:1.2em; columns:2; } .items li { break-inside:avoid; }
.grid { display:grid; gap:12px; } .grid.columns-2 { grid-template-columns:1fr 1fr; }
.grid.columns-3 { grid-template-columns:1fr 1fr 1fr; }
.cell { border:1px solid var(--muted); border-radius:8px; padding:8px; }
.cell .number { font-weight:700; color:var(--accent); }
a.tc { color:var(--accent); text-decoration:none; border-bottom:1px dotted var(--accent); }
.pager { position:absolute; bottom:10px; right:14px; color:var(--muted); font-size:.85em; }
.warnings { position:absolute; top:10px; right:12px; color:var(--muted); cursor:help; }
nav { display:flex; gap:12px; padding:12px; position:fixed; bottom:0; left:0; right:0;
  justify-content:center; background:var(--bg); }
button { background:var(--card); color:var(--fg); border:1px solid var(--muted); border-radius:8px; padding:6px 14px; cursor:pointer; }
@media print { body { background:#fff; } nav { display:none; }
  .slide { display:flex !important; page-break-after:always; width:100%; aspect-ratio:auto; min-height:90vh; margin:0; border-radius:0; } }
"""

JS = """
(function () {
  var slides = Array.prototype.slice.call(document.querySelectorAll('.slide'));
  var current = 0;
  function show(i) {
    current = Math.max(0, Math.min(slides.length - 1, i));
    slides.forEach(function (s, k) { s.classList.toggle('active', k === current); });
    var p = document.getElementById('pos'); if (p) p.textContent = (current + 1) + ' / ' + slides.length;
  }
  document.addEventListener('keydown', function (e) {
    if (e.key === 'ArrowRight' || e.key === 'PageDown' || e.key === ' ') show(current + 1);
    if (e.key === 'ArrowLeft' || e.key === 'PageUp') show(current - 1);
    if (e.key === 'Home') show(0);
    if (e.key === 'End') show(slides.length - 1);
  });
  document.addEventListener('click', function (e) {
    if (e.target.closest('a') || e.target.closest('button')) return;
    show(e.clientX > window.innerWidth / 2 ? current + 1 : current - 1);
  });
  var theme = document.documentElement;
  document.getElementById('theme').addEventListener('click', function () {
    theme.classList.toggle('dark');
    theme.classList.toggle('light');
  });
  document.getElementById('print').addEventListener('click', function () { window.print(); });
  show(0);
})();
"""


def _collect_frames(slides: list[dict], base_dir: Path | None) -> dict[str, Path]:
    paths: list[str] = []
    for slide in slides:
        slots = slide["slots"]
        if "frame" in slots:
            paths.append(slots["frame"].get("path", ""))
        for cell in slots.get("cells", []):
            paths.append(cell["frame"].get("path", ""))
    resolved = {}
    for p in paths:
        if not p:
            continue
        path = Path(p)
        if not path.is_absolute() and base_dir is not None:
            path = base_dir / path
        resolved[p] = path
    return resolved


def _copy_assets(slides: list[dict], out_html: Path, base_dir: Path | None) -> dict[str, str]:
    """Copy referenced frames into deck_assets/; map path -> asset name."""
    assets = out_html.parent / ASSETS_DIRNAME
    assets.mkdir(parents=True, exist_ok=True)
    names: dict[str, str] = {}
    used: dict[str, bytes] = {}
    for source_path, path in _collect_frames(slides, base_dir).items():
        try:
            data = path.read_bytes()
        except OSError:
            continue  # missing frame: the slide shows a placeholder
        name = path.name
        if name in used and used[name] != data:  # collision: suffix with a digest
            name = f"{path.stem}_{hashlib.sha256(data).hexdigest()[:8]}{path.suffix}"
        if name not in used:
            (assets / name).write_bytes(data)
            used[name] = data
        names[source_path] = name
    return names


def export_html(
    slides_doc: dict, out_path: Path, storyboard: dict | None = None, base_dir: Path | None = None
) -> Path:
    """Write the self-contained deck; returns the HTML path.

    ``base_dir`` resolves relative frame paths (the slides document's own
    directory by convention).
    """
    try:
        slides = slides_doc["slides"]
    except (TypeError, KeyError):
        raise ExportError("not a slides document: 'slides' missing") from None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    asset_names = _copy_assets(slides, out_path, base_dir)

    # episode -> source -> moment link (from the Storyboard, when provided)
    sources_by_id = {s["id"]: s for s in (storyboard or {}).get("sources", []) if "id" in s}
    episodes = {e["id"]: e for e in (storyboard or {}).get("episodes", []) if "id" in e}
    links: dict[str, str | None] = {}
    for episode_id, episode in episodes.items():
        source = sources_by_id.get(episode.get("source_id", ""), {})
        links[episode_id] = link_moment(source, episode.get("t_start", 0.0))

    body = "\n".join(
        _render_slide(i, len(slides), slide, links, asset_names) for i, slide in enumerate(slides)
    )
    title = html.escape(str((storyboard or {}).get("project", {}).get("title") or "Razkadr deck"))
    document = f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{CSS}</style>
</head>
<body>
<main id="deck">
{body}
</main>
<nav>
<button id="prev" aria-label="Назад">←</button>
<span id="pos"></span>
<button id="next" aria-label="Вперёд">→</button>
<button id="theme" aria-label="Тема">◐</button>
<button id="print" aria-label="Печать">⎙</button>
</nav>
<script>{JS}</script>
</body>
</html>
"""
    write_text_atomic(out_path, document)
    return out_path


def export_json(storyboard: dict, out_path: Path) -> Path:
    """Canonical JSON rewrite of a (pre-validated) Storyboard."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out_path, storyboard)
    return out_path

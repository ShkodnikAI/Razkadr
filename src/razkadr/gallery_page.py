"""HTML page template for the gallery (naryad 041).

`render_html(data)` returns one self-contained document: inlined CSS and JS,
no external assets, no CDN, no network requests. Thumbnails point at the
run's own frames by relative paths, so the page works from disk via
``file://``. The full data payload is embedded as JSON in a
``<script type="application/json">`` block (``#gallery-data``), which keeps
the page inspectable and lets tests verify frame counts and timecodes
against frames.json without a browser.
"""

from __future__ import annotations

import json

_HEAD = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  --bg: #f5f5f2; --card: #ffffff; --ink: #1f2430; --muted: #6b7280;
  --line: #e3e3dd; --accent: #0f6b5c;
  --role-hero: #b4530a; --role-ingredients: #0a5ab4; --role-step: #0f6b5c;
  --rejected: #9ca3af;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
header { position: sticky; top: 0; z-index: 5; background: var(--card);
  border-bottom: 1px solid var(--line); padding: 10px 16px; }
header h1 { margin: 0 0 4px; font-size: 16px; }
.meta { color: var(--muted); font-size: 12px; margin-bottom: 6px; }
.meta b { color: var(--ink); font-weight: 600; }
.tabs { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.tab { border: 1px solid var(--line); background: #fff; color: var(--ink);
  padding: 5px 12px; border-radius: 6px; cursor: pointer; font: inherit; }
.tab.active { background: var(--accent); border-color: var(--accent); color: #fff; }
.tab:disabled { color: var(--muted); cursor: not-allowed; background: #f0f0ec; }
.filters { margin-left: auto; display: flex; gap: 10px; color: var(--muted);
  font-size: 12px; align-items: center; }
.filters label { display: inline-flex; gap: 4px; align-items: center; cursor: pointer; }
.warn { color: #92400e; font-size: 12px; }
main { padding: 14px 16px 60px; }
.view { display: none; }
.view.active { display: block; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(168px, 1fr)); gap: 10px; }
.card { margin: 0; background: var(--card); border: 1px solid var(--line);
  border-radius: 8px; overflow: hidden; }
.card.rejected { opacity: 0.55; }
.card.hidden { display: none; }
.card.selected { outline: 2px solid var(--role-step); outline-offset: -2px; }
.card[data-role="hero"].selected { outline-color: var(--role-hero); }
.card[data-role="ingredients"].selected { outline-color: var(--role-ingredients); }
.thumb { position: relative; aspect-ratio: 16/9; background: #ddd; }
.thumb img { width: 100%; height: 100%; object-fit: cover; display: block; }
.thumb.missing::after { content: "нет файла"; position: absolute; inset: 0;
  display: flex; align-items: center; justify-content: center;
  color: var(--muted); font-size: 12px; background: repeating-linear-gradient(45deg, #eee, #eee 8px, #e2e2e2 8px, #e2e2e2 16px); }
figcaption { padding: 6px 8px; display: flex; flex-wrap: wrap; gap: 4px 8px;
  align-items: center; font-size: 12px; }
.tcode { font: 600 12px/1 ui-monospace, Menlo, Consolas, monospace;
  border: 1px solid var(--line); background: #fafaf8; border-radius: 4px;
  padding: 3px 6px; cursor: copy; color: var(--ink); }
.tcode:hover { border-color: var(--accent); color: var(--accent); }
.dim { color: var(--muted); }
.badge { border-radius: 4px; padding: 2px 6px; font-size: 11px; color: #fff; }
.badge.rejected { background: var(--rejected); }
.badge.role { background: var(--role-step); }
.badge.role.hero { background: var(--role-hero); }
.badge.role.ingredients { background: var(--role-ingredients); }
.badge.role.steps { background: var(--role-step); }
.badge.label { background: #4b5563; }
.select-group { margin-bottom: 22px; }
.select-group h2 { font-size: 14px; margin: 0 0 8px; padding: 4px 0;
  border-bottom: 1px solid var(--line); }
.select-group h2 .dot { display: inline-block; width: 10px; height: 10px;
  border-radius: 50%; margin-right: 6px; background: var(--role-step); }
.select-group[data-role="hero"] h2 .dot { background: var(--role-hero); }
.select-group[data-role="ingredients"] h2 .dot { background: var(--role-ingredients); }
.episodes { display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 12px; }
.episode { background: var(--card); border: 1px solid var(--line);
  border-left: 4px solid var(--role-step); border-radius: 8px; padding: 10px; }
.episode[data-role="hero"] { border-left-color: var(--role-hero); }
.episode[data-role="ingredients"] { border-left-color: var(--role-ingredients); }
.episode .head { display: flex; justify-content: space-between; gap: 8px;
  font-size: 12px; color: var(--muted); margin-bottom: 8px; }
.episode img.main { width: 100%; aspect-ratio: 16/9; object-fit: cover;
  border-radius: 4px; background: #ddd; display: block; }
.score { font-size: 12px; color: var(--muted); margin-top: 6px; }
.alts { display: flex; gap: 6px; margin-top: 8px; flex-wrap: wrap; }
.alts .alt { width: 100px; }
.alts .alt img { width: 100%; aspect-ratio: 16/9; object-fit: cover;
  border-radius: 3px; background: #ddd; display: block; }
.alts .alt .tcode { margin-top: 3px; }
.empty { color: var(--muted); padding: 20px 0; }
#toast { position: fixed; bottom: 18px; left: 50%; transform: translateX(-50%);
  background: var(--ink); color: #fff; border-radius: 6px; padding: 7px 14px;
  font-size: 13px; opacity: 0; pointer-events: none; transition: opacity .2s; z-index: 9; }
#toast.show { opacity: 1; }
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <div class="meta">__META__</div>
  <div class="meta">Кадров в run: <b>__TOTAL__</b>__SHOWN__ · сгенерировано __GENERATED__ UTC</div>
  <div class="meta warn">__WARNINGS__</div>
  <nav class="tabs">
    <button id="tab-sheet" class="tab active" data-mode="sheet" type="button">Лист разметки</button>
    <button id="tab-select" class="tab" data-mode="select" type="button">Просмотр выбора</button>
    <span class="filters">
      <label><input type="checkbox" id="f-rejected" checked> отбракованные</label>
      <label><input type="checkbox" id="f-insheet"> только лист (каждые __EVERY__ с)</label>
    </span>
  </nav>
</header>
<main>
<section id="view-sheet" class="view active">
__SHEET_GRID__
</section>
<section id="view-select" class="view">
__SELECT_VIEW__
</section>
</main>
<div id="toast" role="status"></div>
"""

_BODY_END = """
<script id="gallery-data" type="application/json">__DATA__</script>
<script>
(function () {
  "use strict";
  var dataEl = document.getElementById("gallery-data");
  var data = JSON.parse(dataEl.textContent);
  var grid = document.getElementById("sheet-grid");
  var toast = document.getElementById("toast");
  var toastTimer = null;

  function showToast(msg) {
    toast.textContent = msg;
    toast.classList.add("show");
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(function () { toast.classList.remove("show"); }, 1600);
  }

  function copyText(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(function () {
        showToast("Скопировано: " + text);
      }, function () { fallbackCopy(text); });
    } else { fallbackCopy(text); }
  }
  function fallbackCopy(text) {
    var ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand("copy"); showToast("Скопировано: " + text); }
    catch (e) { showToast("Не удалось скопировать: " + text); }
    document.body.removeChild(ta);
  }

  function fmtTc(t) {
    var s = Number(t).toFixed(3);
    return s;
  }

  grid.addEventListener("click", function (ev) {
    var btn = ev.target.closest(".tcode");
    if (!btn) return;
    copyText(btn.getAttribute("data-t"));
  });

  var tabSheet = document.getElementById("tab-sheet");
  var tabSelect = document.getElementById("tab-select");
  var viewSheet = document.getElementById("view-sheet");
  var viewSelect = document.getElementById("view-select");
  function activate(mode) {
    tabSheet.classList.toggle("active", mode === "sheet");
    tabSelect.classList.toggle("active", mode === "select");
    viewSheet.classList.toggle("active", mode === "sheet");
    viewSelect.classList.toggle("active", mode === "select");
  }
  if (tabSelect.disabled) {
    tabSelect.title = "storyboard.json не найден в каталоге запуска";
  } else {
    tabSelect.title = "Кадры выбора из Storyboard по ролям";
  }
  tabSheet.addEventListener("click", function () { activate("sheet"); });
  tabSelect.addEventListener("click", function () {
    if (!tabSelect.disabled) activate("select");
  });

  var fRejected = document.getElementById("f-rejected");
  var fInSheet = document.getElementById("f-insheet");
  function applyFilters() {
    var cards = grid.querySelectorAll(".card");
    for (var i = 0; i < cards.length; i++) {
      var card = cards[i];
      var hide = false;
      if (card.getAttribute("data-rejected") === "1" && !fRejected.checked) hide = true;
      if (fInSheet.checked && card.getAttribute("data-in-sheet") !== "1") hide = true;
      card.classList.toggle("hidden", hide);
    }
  }
  fRejected.addEventListener("change", applyFilters);
  fInSheet.addEventListener("change", applyFilters);
})();
</script>
</body>
</html>
"""

_EMPTY_SELECT = """<p class="empty">Storyboard не найден: режим «Просмотр выбора» появится, \
когда в каталоге запуска появится storyboard.json (наряд №014, волна 2) или будет передан \
--storyboard.</p>"""


def _safe_json(obj: object) -> str:
    """JSON for embedding inside a <script> block: "</" would close it."""
    return json.dumps(obj, ensure_ascii=False).replace("</", "<\\/")


def _esc(text: object) -> str:
    """HTML-escape a small header value."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _fmt_num(value: object, nd: int = 3) -> str:
    try:
        return f"{float(value):.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def _meta_line(data: dict) -> str:
    src = data.get("source")
    if not src:
        return "источник: <i>source.json отсутствует</i>"
    parts = []
    if src.get("width") and src.get("height"):
        parts.append(f"{src['width']}×{src['height']}")
    if src.get("fps"):
        parts.append(f"{_fmt_num(src['fps'], 2)} fps")
    if src.get("duration_s") is not None:
        parts.append(f"{_fmt_num(src['duration_s'], 3)} с")
    what = src.get("url") or src.get("path") or ""
    return f"источник: <b>{_esc(what)}</b> · {' · '.join(parts)}"


def _card(frame: dict, index: int) -> str:
    t = _fmt_num(frame.get("t"))
    file_rel = _esc(frame.get("file"))
    rejected = bool(frame.get("rejected"))
    cls = ["card"]
    if rejected:
        cls.append("rejected")
    role = frame.get("selected_role")
    if role:
        cls.append("selected")
    attrs = [
        f'class="{" ".join(cls)}"',
        f'data-index="{index}"',
        f'data-t="{t}"',
        f'data-file="{file_rel}"',
        f'data-rejected="{"1" if rejected else "0"}"',
        f'data-in-sheet="{"1" if frame.get("in_sheet") else "0"}"',
    ]
    if role:
        attrs.append(f'data-role="{_esc(role)}"')
    badges = []
    if rejected:
        reason = frame.get("reject_reason") or "rejected"
        badges.append(f'<span class="badge rejected">{_esc(reason)}</span>')
    if role:
        badges.append(f'<span class="badge role {_esc(role)}">роль: {_esc(role)}</span>')
    for hit in frame.get("label_hits") or []:
        title = _esc(hit.get("note")) if hit.get("note") else ""
        badges.append(
            f'<span class="badge label" title="{title}">эталон: {_esc(hit.get("role"))}'
            f" @{_fmt_num(hit.get('t'))}с</span>"
        )
    scene = frame.get("scene_id")
    quality = (
        f'<span class="dim">резк. {_fmt_num(frame.get("sharpness"), 2)} · '
        f"эксп. {_fmt_num(frame.get('exposure'), 2)}</span>"
    )
    scene_html = f'<span class="dim">scene {scene}</span>' if scene is not None else ""
    return (
        f'<figure {" ".join(attrs)}>'
        f'<div class="thumb"><img src="{file_rel}" loading="lazy" decoding="async" '
        f'alt="кадр {t} с" onerror="this.closest(\'.thumb\').classList.add(\'missing\')"></div>'
        f"<figcaption>"
        f'<button class="tcode" type="button" data-t="{t}" title="копировать таймкод">{t} с</button>'
        f"{scene_html}"
        f"{quality}{''.join(badges)}"
        f"</figcaption></figure>"
    )


def _alt(alt: dict) -> str:
    t = _fmt_num(alt.get("t"))
    path = _esc(alt.get("path") or "")
    score = alt.get("score")
    score_html = f' <span class="dim">score {_fmt_num(score, 3)}</span>' if score is not None else ""
    return (
        f'<div class="alt">'
        f'<img src="{path}" loading="lazy" decoding="async" alt="альтернатива {t} с" '
        f"onerror=\"this.closest('.alt').classList.add('missing')\">"
        f'<button class="tcode" type="button" data-t="{t}" title="копировать таймкод">{t} с</button>'
        f"{score_html}</div>"
    )


def _episode(ep: dict) -> str:
    role = _esc(ep.get("role"))
    frame = ep.get("frame") or {}
    path = _esc(frame.get("path") or "")
    t = _fmt_num(frame.get("t"))
    parts = frame.get("score_parts") or {}
    parts_html = ""
    if parts:
        joined = ", ".join(f"{k} {_fmt_num(v, 3)}" for k, v in sorted(parts.items()))
        parts_html = f' <span class="dim">({joined})</span>'
    score_html = ""
    if frame.get("score") is not None:
        score_html = f'score <b>{_fmt_num(frame.get("score"), 3)}</b>{parts_html}'
    alts = ep.get("alternatives") or []
    alts_html = ""
    if alts:
        alts_html = '<div class="alts">' + "".join(_alt(a) for a in alts) + "</div>"
    head = (
        f'<span><b>{_esc(ep.get("id"))}</b> · {_esc(ep.get("source_id"))}</span>'
        f"<span>{_fmt_num(ep.get('t_start'), 2)}–{_fmt_num(ep.get('t_end'), 2)} с</span>"
    )
    return (
        f'<article class="episode" data-role="{role}" '
        f'data-episode="{_esc(ep.get("id"))}" data-t="{t}">'
        f'<div class="head">{head}</div>'
        f'<img class="main" src="{path}" loading="lazy" decoding="async" '
        f'alt="кадр эпизода {_esc(ep.get("id"))}" '
        f"onerror=\"this.classList.add('missing')\">"
        f'<div class="score">t {t} с · {score_html}</div>'
        f"{alts_html}</article>"
    )


def _select_view(data: dict) -> str:
    selection = data.get("selection")
    if not selection:
        return _EMPTY_SELECT
    by_role: dict[str, list[dict]] = {}
    for ep in selection.get("episodes") or []:
        by_role.setdefault(ep.get("role") or "?", []).append(ep)
    out = []
    for role in selection.get("roles") or list(by_role):
        episodes = by_role.get(role) or []
        out.append(f'<div class="select-group" data-role="{_esc(role)}">')
        out.append(f"<h2><span class=\"dot\"></span>роль: {_esc(role)} · эпизодов: {len(episodes)}</h2>")
        out.append('<div class="episodes">')
        out.extend(_episode(ep) for ep in episodes)
        out.append("</div></div>")
    return "".join(out)


def render_html(data: dict) -> str:
    """The whole page: header, label sheet grid, selection view, data+app JS."""
    run_id = _esc(data.get("run_dir"))
    total = data.get("frames_total", 0)
    shown = data.get("frames_shown", 0)
    every_s = float(data.get("every_s") or 0)
    shown_html = ""
    if every_s > 0:
        shown_html = f", в листе: <b>{shown}</b> (каждые {every_s:g} с)"
    warnings = data.get("warnings") or []
    warnings_html = " · ".join(_esc(w) for w in warnings) if warnings else ""
    sheet_cards = "".join(
        _card(frame, i) for i, frame in enumerate(data.get("frames") or [])
    )
    sheet = f'<div class="grid" id="sheet-grid">{sheet_cards}</div>' if sheet_cards else '<p class="empty">Кадров нет.</p>'

    html = _HEAD
    html = html.replace("__TITLE__", f"Razkadr gallery — {run_id}")
    html = html.replace("__META__", _meta_line(data))
    html = html.replace("__TOTAL__", str(total))
    html = html.replace("__SHOWN__", shown_html)
    html = html.replace("__GENERATED__", _esc(data.get("generated_utc", "")))
    html = html.replace("__WARNINGS__", warnings_html)
    html = html.replace("__EVERY__", f"{every_s:g}")
    html = html.replace("__SHEET_GRID__", sheet)
    html = html.replace("__SELECT_VIEW__", _select_view(data))
    select_tab_disabled = "" if data.get("selection") else " disabled"
    html = html.replace(
        '<button id="tab-select" class="tab" data-mode="select" type="button">',
        f'<button id="tab-select" class="tab" data-mode="select" type="button"{select_tab_disabled}>',
    )
    html += _BODY_END.replace("__DATA__", _safe_json(data))
    return html

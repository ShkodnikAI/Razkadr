"""Assemble final Razkadr logo set: python build_final.py <font.woff> <out_dir>"""
import re, sys
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.transformPen import TransformPen

FONT, OUT = sys.argv[1], sys.argv[2]
SLATE, INK, CREAM = "#2B2B2B", "#1A1A1A", "#F8F4E9"
TEXT = "RAZKADR"; TRACK = 0.06

def inner(path):
    return re.search(r"<svg[^>]*>(.*)</svg>", open(path).read(), re.S).group(1)

font = TTFont(FONT); gs = font.getGlyphSet(); cmap = font.getBestCmap(); upm = font["head"].unitsPerEm
def word(cap, x0, cy, color, anchor="start"):
    bp = BoundsPen(gs); gs[cmap[ord("D")]].draw(bp)
    fs = cap / (bp.bounds[3]/upm); s = fs/upm
    adv = [gs[cmap[ord(c)]].width*s + TRACK*fs for c in TEXT]
    width = sum(adv) - TRACK*fs
    x = x0 - (width/2 if anchor == "middle" else 0)
    ymax = ymin = None
    for c in TEXT:
        b = BoundsPen(gs); gs[cmap[ord(c)]].draw(b)
        ymax = b.bounds[3]*s if ymax is None else max(ymax, b.bounds[3]*s)
        ymin = b.bounds[1]*s if ymin is None else min(ymin, b.bounds[1]*s)
    base = cy + (ymax+ymin)/2
    d = ""
    for c, a in zip(TEXT, adv):
        pen = SVGPathPen(gs); gs[cmap[ord(c)]].draw(TransformPen(pen, (s, 0, 0, -s, x, base))); d += pen.getCommands(); x += a
    return f'<path d="{d}" fill="{color}"/>', width

def svg(w, h, body, title="Razkadr"):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}" width="{w:.0f}" height="{h:.0f}" '
            f'role="img" aria-label="{title}"><title>{title}</title>{body}</svg>\n')
def save(name, w, h, body): open(f"{OUT}/{name}", "w").write(svg(w, h, body))

M = {"": inner("aperture-strip-mark.svg"), "-on-dark": inner("aperture-strip-mark-on-dark.svg"), "-mono": inner("aperture-strip-mark-mono.svg")}
TC = {"": SLATE, "-on-dark": CREAM, "-mono": INK}
for k in M:
    save(f"logo-mark{k}.svg", 300, 300, M[k])
    # horizontal: mark 300, gap 40, wordmark cap height 104
    wp, ww = word(104, 340, 150, TC[k])
    save(f"logo-horizontal{k}.svg", 340 + ww + 28, 300, M[k] + wp)
    # vertical: mark centred, wordmark below
    W = 560
    wp, ww = word(66, W/2, 300 + 56 + 33, TC[k], "middle")
    save(f"logo-vertical{k}.svg", W, 300 + 56 + 66 + 36, f'<g transform="translate({(W-300)/2:.0f} 0)">{M[k]}</g>' + wp)
# simple versions for small sizes
for src, dst in (("aperture-mark.svg", "mark-simple.svg"), ("aperture-mark-on-dark.svg", "mark-simple-on-dark.svg"),
                 ("aperture-mark-mono.svg", "mark-simple-mono.svg"), ("aperture-icon.svg", "icon.svg")):
    open(f"{OUT}/{dst}", "w").write(open(src).read())
print("done")

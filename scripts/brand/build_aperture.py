import math
from shapely.geometry import Point, Polygon, LineString, MultiPolygon
from shapely.ops import unary_union

BURG, GOLD, CREAM, SLATE, INK = "#6B0F1A", "#D4AF7F", "#F8F4E9", "#2B2B2B", "#1A1A1A"
CX = CY = 128.0

def blades_path(R=108.0, r0=40.0, gap=9.0):
    verts = [(r0*math.cos(math.radians(-90+60*i)), r0*math.sin(math.radians(-90+60*i))) for i in range(6)]
    hexa = Polygon(verts)
    cuts = [hexa]
    for i in range(6):
        a, b = verts[i], verts[(i+1) % 6]
        dx, dy = b[0]-a[0], b[1]-a[1]
        n = math.hypot(dx, dy); dx, dy = dx/n, dy/n
        cuts.append(LineString([a, (a[0]+dx*400, a[1]+dy*400)]).buffer(gap/2, cap_style=2))
    shape = Point(0, 0).buffer(R, resolution=64).difference(unary_union(cuts))
    shape = shape.buffer(-1.6, join_style=1).buffer(1.6, join_style=1)   # soften corners
    polys = list(shape.geoms) if isinstance(shape, MultiPolygon) else [shape]
    d = []
    for p in polys:
        if p.area < 20: continue
        pts = list(p.exterior.coords)[:-1]
        d.append("M" + " L".join(f"{CX+x:.2f} {CY+y:.2f}" for x, y in pts) + "Z")
    return "".join(d)

BL = blades_path()

def mark_group(blade, ring=None, ring_w=8, ring_r=119):
    g = f'<path d="{BL}" fill="{blade}"/>'
    if ring:
        g += f'<circle cx="{CX}" cy="{CY}" r="{ring_r}" fill="none" stroke="{ring}" stroke-width="{ring_w}"/>'
    return g

def svg(w, h, body, label="Razkadr"):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'role="img" aria-label="{label}">{body}</svg>\n')

def write(name, w, h, body):
    open(name, "w").write(svg(w, h, body))

write("aperture-mark.svg", 256, 256, mark_group(BURG, GOLD))
write("aperture-mark-on-dark.svg", 256, 256, mark_group(CREAM, GOLD))
write("aperture-mark-mono.svg", 256, 256, mark_group(INK, INK))
# app icon / favicon: no ring, blades larger, burgundy tile
big = blades_path(R=112, r0=42, gap=10)
write("aperture-icon.svg", 256, 256,
      f'<rect width="256" height="256" rx="56" fill="{BURG}"/>'
      f'<path d="{big.replace("M","M")}" fill="{CREAM}"/>')

print("ok")

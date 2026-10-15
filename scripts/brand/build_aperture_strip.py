import math
from shapely import affinity
from shapely.geometry import Point, Polygon, LineString, MultiPolygon, box
from shapely.ops import unary_union

BURG, BURG2, GOLD, CREAM, CREAM2, SLATE, INK = "#6B0F1A", "#4A0B14", "#D4AF7F", "#F8F4E9", "#DCCFB4", "#2B2B2B", "#1A1A1A"
S = 300.0; CX = CY = S/2

def pth(geom, rnd=2):
    out = ""
    for p in (geom.geoms if isinstance(geom, MultiPolygon) else [geom]):
        if p.area < 3: continue
        for ring in [p.exterior] + list(p.interiors):
            pts = list(ring.coords)[:-1]
            out += "M" + " L".join(f"{CX+x:.{rnd}f} {CY+y:.{rnd}f}" for x, y in pts) + "Z"
    return out

def blades(R=88.0, r0=36.0, gap=7.5):
    verts = [(r0*math.cos(math.radians(-90+60*i)), r0*math.sin(math.radians(-90+60*i))) for i in range(6)]
    cuts = [Polygon(verts)]
    for i in range(6):
        a, b = verts[i], verts[(i+1) % 6]
        dx, dy = b[0]-a[0], b[1]-a[1]; n = math.hypot(dx, dy); dx, dy = dx/n, dy/n
        cuts.append(LineString([a, (a[0]+dx*400, a[1]+dy*400)]).buffer(gap/2, cap_style=2))
    shape = Point(0, 0).buffer(R, resolution=64).difference(unary_union(cuts))
    shape = shape.buffer(-1.3, join_style=1).buffer(1.3, join_style=1)
    polys = [p for p in (shape.geoms if isinstance(shape, MultiPolygon) else [shape]) if p.area > 20]
    polys.sort(key=lambda p: math.atan2(p.centroid.y, p.centroid.x))
    return [pth(p) for p in polys]

# film strip geometry (radii in viewBox units)
R_IN, R_OUT = 98.0, 138.0
FR_IN, FR_OUT = 107.0, 129.0
N_FR = 20; PITCH = 360/N_FR; FR_ANG = 12.0
HOLE_R = (102.5, 133.5); HOLE_RAD, HOLE_TAN = 4.6, 7.5

def build_strip():
    band = Point(0, 0).buffer(R_OUT, resolution=96).difference(Point(0, 0).buffer(R_IN, resolution=96))
    holes = []
    for rc in HOLE_R:
        for j in range(N_FR*2):
            h = box(rc-HOLE_RAD/2, -HOLE_TAN/2, rc+HOLE_RAD/2, HOLE_TAN/2).buffer(-1.0, join_style=2).buffer(1.0, join_style=1)
            holes.append(affinity.rotate(h, j*PITCH/2 + PITCH/4, origin=(0, 0)))
    ring_fr = Point(0, 0).buffer(FR_OUT, resolution=96).difference(Point(0, 0).buffer(FR_IN, resolution=96))
    frames = []
    for k in range(N_FR):
        a1, a2 = math.radians(k*PITCH + PITCH/2 - FR_ANG/2), math.radians(k*PITCH + PITCH/2 + FR_ANG/2)
        wedge = Polygon([(0, 0), (400*math.cos(a1), 400*math.sin(a1)), (400*math.cos(a2), 400*math.sin(a2))])
        c = ring_fr.intersection(wedge).buffer(-1.8, join_style=1).buffer(1.8, join_style=1)
        frames.append(c)
    frames_u = unary_union(frames)
    band_cut = band.difference(unary_union(holes)).difference(frames_u)
    return pth(band_cut), pth(frames_u)
BAND, FRAMES = build_strip()
BL = blades()

def play(color, r=19, dx=3):
    pts = [(CX+dx+r*math.cos(math.radians(a)), CY+r*math.sin(math.radians(a))) for a in (0, 120, 240)]
    return (f'<path d="M{pts[0][0]:.2f} {pts[0][1]:.2f}L{pts[1][0]:.2f} {pts[1][1]:.2f}L{pts[2][0]:.2f} {pts[2][1]:.2f}Z" '
            f'fill="{color}" stroke="{color}" stroke-width="4.5" stroke-linejoin="round"/>')

def mark(b1, b2, band, frame, ring, pl):
    g = "".join(f'<path d="{d}" fill="{b1 if i % 2 == 0 else b2}"/>' for i, d in enumerate(BL))
    g += f'<circle cx="{CX}" cy="{CY}" r="92.5" fill="none" stroke="{ring}" stroke-width="2.5"/>'
    g += f'<path d="{BAND}" fill="{band}" fill-rule="evenodd"/>'
    if frame: g += f'<path d="{FRAMES}" fill="{frame}" fill-rule="evenodd"/>'
    g += play(pl)
    return g

def svg(w, h, body):
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="Razkadr">{body}</svg>\n'
def write(name, w, h, body): open(name, "w").write(svg(w, h, body))

M_LIGHT = mark(BURG, BURG2, BURG2, GOLD, GOLD, GOLD)
M_DARK = mark(CREAM, CREAM2, CREAM, GOLD, GOLD, GOLD)
M_MONO = mark(INK, INK, INK, None, INK, INK)
write("aperture-strip-mark.svg", 300, 300, M_LIGHT)
write("aperture-strip-mark-on-dark.svg", 300, 300, M_DARK)
write("aperture-strip-mark-mono.svg", 300, 300, M_MONO)

print("ok")

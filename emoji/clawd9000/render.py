#!/usr/bin/env python3
"""'Clawd 9000' (single-lens cut) -- Clawd IS the faceplate. His whole body is
machined into a charcoal HAL panel: seams, rivets, a top-lit bevel, dark struts
where his legs are -- and one enormous fisheye lens set into the middle of him,
black housing, brushed bezel, crimson iris, hot yellow core.

Design notes:
  * ONE lens, as big as the silhouette can hold (r=28 on a 120x80 sprite, i.e.
    nearly half his height). HAL is a single unblinking eye; two eyes read as a
    face, one reads as a machine. It is centred at (row 3, col 6) -- the widest
    part of him -- and every point of the circle lands inside his own pixels, so
    nothing bulges out of the silhouette.
  * His eye cells are not deleted, they are ABSORBED: painted as dark recessed
    vents in the plate, then all but swallowed by the lens housing. He still has his
    hands, his legs, his proportions and his sacred 2 px white outline -- the
    creature is intact, the paint job is a machine.
  * The 128-grid is used raw (no CELL blockiness) so the circle and its bezel
    stay round at 32 px, where this reads as a red dot in a dark slab.
  * The glow is alpha-composited through a precomputed palette LUT (mad-Clawd's
    trick), so the plate and its seams read *through* the crimson wash instead of
    being erased, and every lit pixel is clipped to Clawd -- the background stays
    genuinely transparent for any Slack theme.

Motion, all seamless:
  pulse   : 0.60 + 0.40*sin(2*pi*f/F)      -- the slow breath of the iris.
  focus   : 1 + 0.35*cos(4*pi*f/F)         -- the core stops down twice a loop,
                                              a mechanical focus hunt (2nd
                                              harmonic: still equal at f=0/f=F).
  scan    : a bright bar creeps down the glass once per loop, entering above the
            housing and leaving below it, so the wrap happens off-glass.

Outputs the still + gif.
"""
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.clawd import ART, WHITE_RGB, border_mask, pen_disk

OUT = Path(__file__).resolve().parent
NAME = "clawd_9000"

N = 128
F = 20                            # 2.0 s loop
DUR = 100
SCALE = 10                        # 12x8 art -> 120x80: full canvas width

# ---- lens geometry ---------------------------------------------------------
R_HOUS = 28                       # outer housing radius: the biggest circle that
                                  # fits inside his silhouette at this centre
R_BEZ = 4                         # brushed bezel ring thickness
R_GLASS = 22                      # dark glass inside the bezel
R_IRIS = 16.0                     # glowing crimson aperture
R_SPILL = 40.0                    # crimson wash thrown across the plate

# ---- palette ---------------------------------------------------------------
BASE = [
    (0, 0, 0),                    # 0 transparent slot
    WHITE_RGB,                    # 1 outline (sacred)
    (44, 47, 54),                 # 2 plate, top-lit
    (32, 34, 40),                 # 3 plate, body
    (20, 21, 26),                 # 4 plate, shadowed / struts / seams
    (72, 77, 86),                 # 5 rivets + bezel highlight
    (52, 56, 63),                 # 6 bezel shadow
    (10, 10, 13),                 # 7 lens glass / housing
]
T, OUTLINE, PLATE_L, PLATE_M, PLATE_D, RIVET, BEZ_D, GLASS = range(8)

STOPS = ((0.00, (46, 2, 6)), (0.20, (124, 8, 12)), (0.44, (204, 22, 20)),
         (0.66, (248, 70, 36)), (0.85, (255, 172, 74)), (1.00, (255, 250, 230)))
LEVELS = 14


def ramp(i):
    for k in range(len(STOPS) - 1):
        p0, c0 = STOPS[k]
        p1, c1 = STOPS[k + 1]
        if i <= p1:
            t = 0.0 if p1 == p0 else (i - p0) / (p1 - p0)
            return tuple(round(c0[j] + (c1[j] - c0[j]) * t) for j in range(3))
    return STOPS[-1][1]


def alpha(i):
    return min(1.0, (i ** 0.7) * 1.25)


COLORS = list(BASE)
LUT = {}
for _b in range(1, len(BASE)):                 # everything but the transparent slot
    row = [_b]
    for _k in range(1, LEVELS + 1):
        _i = _k / LEVELS
        _a = alpha(_i)
        _c = ramp(_i)
        COLORS.append(tuple(round(BASE[_b][j] * (1 - _a) + _c[j] * _a)
                            for j in range(3)))
        row.append(len(COLORS) - 1)
    LUT[_b] = row
assert len(COLORS) <= 256, len(COLORS)
PAL = bytes([c for rgb in COLORS for c in rgb] + [0] * (768 - 3 * len(COLORS)))

BASE_OF, LEVEL_OF = {}, {}
for _b, _row in LUT.items():
    for _lv, _idx in enumerate(_row):
        if _lv:
            BASE_OF[_idx], LEVEL_OF[_idx] = _b, _lv

# ---- layout ----------------------------------------------------------------
SH, SW = 8 * SCALE, 12 * SCALE
X0, Y0 = (N - SW) // 2, (N - SH) // 2          # 4, 24
CY, CX = Y0 + 3 * SCALE, X0 + 6 * SCALE        # 54, 64: lens centre (row 3, col 6)

RIVETS = [(Y0 + 22, X0 + 8), (Y0 + 22, X0 + SW - 9),          # the wide band's
          (Y0 + 38, X0 + 8), (Y0 + 38, X0 + SW - 9)]          # four corners


def build_plate():
    """Clawd, machined: the ART grid painted as a bevelled charcoal panel with
    seams, rivets, dark struts for the legs, recessed vents where his eyes are,
    and his 2 px white outline drawn last so it stays exactly what it always is."""
    g = np.zeros((N, N), dtype=np.uint8)
    vents = []
    for r, row in enumerate(ART):
        for c, ch in enumerate(row):
            if ch == ".":
                continue
            y, x = Y0 + r * SCALE, X0 + c * SCALE
            g[y:y + SCALE, x:x + SCALE] = PLATE_D if r >= 6 else PLATE_M
            if ch == "O":
                vents.append((y, x))

    body = g != T
    # top-lit bevel: two bright rows along every upward-facing edge
    for y in range(N - 1, 1, -1):
        lit = body[y] & ~body[y - 2]
        g[y][lit] = PLATE_L
        g[y - 1][lit & body[y - 1]] = PLATE_L
    # panel seams: two horizontals across the torso, one vertical down the middle
    for y in (Y0 + 2 * SCALE, Y0 + 5 * SCALE):
        g[y][body[y]] = PLATE_D
        g[y + 1][body[y + 1]] = PLATE_D
    for x in (X0 + 2 * SCALE - 1, X0 + 10 * SCALE):
        col = body[:, x]
        g[col, x] = PLATE_D
    # recessed vents where his eyes were: three dark slits each
    for (y, x) in vents:
        for k in range(3):
            g[y + 2 + k * 3:y + 4 + k * 3, x + 1:x + SCALE - 1] = PLATE_D
    for (y, x) in RIVETS:
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if abs(dy) + abs(dx) < 2:
                    g[y + dy, x + dx] = RIVET

    # the lens: brushed bezel ring, then the dark glass, with faint concentric
    # machining rings so the glass reads as a fisheye and not a hole
    for y in range(CY - R_HOUS, CY + R_HOUS + 1):
        for x in range(CX - R_HOUS, CX + R_HOUS + 1):
            d = math.hypot(y - CY, x - CX)
            if d > R_HOUS:
                continue
            if d > R_GLASS + R_BEZ:                     # black outer housing
                g[y, x] = GLASS
            elif d > R_GLASS:                           # bezel: lit up-left
                g[y, x] = RIVET if (y - CY) + (x - CX) < 0 else BEZ_D
            else:
                g[y, x] = PLATE_D if int(d) % 6 == 3 else GLASS

    g[border_mask(g != T, pen_disk(2))] = OUTLINE
    return g


PLATE = build_plate()


OUTLINE_LIGHT = 0.40              # the outline only ever catches a warm hint of
                                  # the glow -- it has to stay a white outline


def light(g, y, x, inten):
    """Add light to one of Clawd's pixels, keeping the brighter of two lights."""
    if inten <= 0.02 or not (0 <= y < N and 0 <= x < N):
        return
    if g[y, x] == OUTLINE or LEVEL_OF.get(g[y, x], 0) and BASE_OF.get(g[y, x]) == OUTLINE:
        inten *= OUTLINE_LIGHT
    lv = min(LEVELS, int(round(inten * LEVELS)))
    if lv == 0:
        return
    under = g[y, x]
    if under == T:                                # never light the empty canvas
        return
    if under not in LUT:
        if lv <= LEVEL_OF.get(under, 0):
            return
        under = BASE_OF[under]
    g[y, x] = LUT[under][lv]


def compose(f):
    g = PLATE.copy()
    ph = 2 * math.pi * f / F
    p = 0.60 + 0.40 * math.sin(ph)                # the breath
    focus = 1 + 0.35 * math.cos(2 * ph)           # the mechanical focus hunt
    core_r = R_IRIS * 0.26 * focus
    # scan bar: enters above the housing, leaves below it -> the wrap is hidden
    scan_y = CY - R_HOUS - 3 + (2 * R_HOUS + 6) * f / F

    r = int(R_SPILL)
    for y in range(CY - r, CY + r + 1):
        for x in range(CX - r, CX + r + 1):
            d = math.hypot(y - CY, x - CX)
            if d <= R_IRIS:                       # the aperture
                u = d / R_IRIS
                core = math.exp(-(d / core_r) ** 2)
                iris = (1 - u) ** 1.4
                inten = 0.18 + p * (0.80 * core + 0.66 * iris)
                if abs(y - scan_y) <= 1.5:        # scan bar crossing the glass
                    inten += 0.34
                light(g, y, x, min(inten, 1.0))
            elif d <= R_GLASS:                    # glow bleeding into the glass
                u = (d - R_IRIS) / (R_GLASS - R_IRIS)
                inten = p * 0.5 * (1 - u) ** 1.6
                if abs(y - scan_y) <= 1.5:
                    inten += 0.20
                light(g, y, x, inten)
            elif d <= R_SPILL and d > R_HOUS:     # crimson wash over the plate
                u = (d - R_HOUS) / (R_SPILL - R_HOUS)
                light(g, y, x, p * 0.60 * (1 - u) ** 2)
    # specular catchlight: a solid 2 px arc on the up-left rim of the glass
    for a in range(-152, -84):
        ang = math.radians(a)
        for rr in (R_GLASS - 2, R_GLASS - 3):
            light(g, int(round(CY + rr * math.sin(ang))),
                  int(round(CX + rr * math.cos(ang))), 0.45 + 0.35 * p)
    return g


def save():
    frames = []
    for f in range(F):
        im = Image.frombytes("P", (N, N), compose(f).tobytes())
        im.putpalette(PAL)
        frames.append(im)

    frames[F // 4].convert("RGBA").save(OUT / f"{NAME}_still.png")    # peak pulse
    gif = OUT / f"{NAME}.gif"
    frames[0].save(
        gif, save_all=True, append_images=frames[1:], duration=DUR, loop=0,
        transparency=T, disposal=2, optimize=False,
    )
    kb = gif.stat().st_size / 1024
    assert kb <= 128, f"{gif.name} is {kb:.0f} KB - over Slack's 128 KB cap"
    print(f"{NAME}: {F} frames @ {DUR}ms, gif={kb:.0f} KB")


if __name__ == "__main__":
    save()

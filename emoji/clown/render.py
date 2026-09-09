#!/usr/bin/env python3
"""'Clown' Clawd -- rainbow afro, big red ball nose, spotty bow tie, and a honk.

Rendered on the full 128 grid (CELL=1) so the wig's puffs and the round nose
stay curved rather than blocky.

  wig   : seven overlapping two-tone puffs domed over his head, red -> purple,
          the outer ones dropping past his temples.
  nose  : a glossy sphere centred on his face, drawn as an ellipse so it can
          squash.
  tie   : a spotty yellow bow across his legs, built as its own layer and
          rotated about its knot so it can waggle.
  honk  : twice a loop the nose swells, flattens and flashes bright; the wig
          ripples and the tie waggles on the same beat, and he recoils a pixel.

Seamless by construction: the bob is sin(2*pi*f/F) and every honk-driven term
is a sin/cos of 2*pi*HONKS*f/F, so frame 0 and frame F match exactly.
"""
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.clawd import ART, CLAWD_RGB, EYE_RGB, WHITE_RGB, border_mask, pen_disk

OUT = Path(__file__).resolve().parent

N = 128                 # canvas (Slack emoji size)
F = 16                  # frames per loop
DUR = 80                # ms per frame

# ---- palette (P-mode GIF: index 0 is transparent) --------------------------
COLORS = [
    (0, 0, 0),          # 0  transparent slot
    WHITE_RGB,          # 1  outline
    CLAWD_RGB,          # 2  body
    EYE_RGB,            # 3  eyes
    (214, 45, 45),      # 4  wig red          } six hues, light...
    (238, 124, 30),     # 5  wig orange
    (250, 199, 32),     # 6  wig yellow
    (47, 158, 68),      # 7  wig green
    (36, 104, 205),     # 8  wig blue
    (124, 60, 178),     # 9  wig purple
    (168, 30, 32),      # 10 wig red shade    } ...and their shaded twins
    (196, 92, 18),      # 11 wig orange shade
    (216, 158, 20),     # 12 wig yellow shade
    (30, 118, 48),      # 13 wig green shade
    (22, 74, 158),      # 14 wig blue shade
    (92, 40, 138),      # 15 wig purple shade
    (226, 22, 30),      # 16 nose
    (154, 12, 20),      # 17 nose shade
    (255, 168, 178),    # 18 nose sheen
    (255, 62, 54),      # 19 nose, honking (a real jump in luminance --
                        #    a near-identical red reads as no flash at all)
    (250, 214, 60),     # 20 tie yellow
    (206, 162, 24),     # 21 tie shade
]
(T, OUTLINE, BODY, EYE, W_RED, W_ORG, W_YEL, W_GRN, W_BLU, W_PUR,
 S_RED, S_ORG, S_YEL, S_GRN, S_BLU, S_PUR,
 NOSE, NOSE_SHADE, NOSE_SHEEN, NOSE_HOT, TIE, TIE_SHADE) = range(22)
PAL = bytes([c for rgb in COLORS for c in rgb] + [0] * (768 - 3 * len(COLORS)))

# ---- layout ----------------------------------------------------------------
# SCALE 10 is the widest Clawd fits: 12 cells * 10 = 120 px, leaving exactly the
# 4 px his outline needs. The wig then sets the height, so everything else is
# measured off the body's top-left corner (BY, BX) in a padded work canvas.
SCALE = 10
AH, AW = 180, 180                       # work canvas (blitted down to 128 later)
BY, BX = 70, 30                         # body top-left
SX = 12 * SCALE                         # 120 px body width

# wig: puff centres ride an ellipse over the head. The high ones also grow a
# second disk beneath them, so the seven fuse into one bumpy mass that scallops
# onto his forehead instead of reading as a chain of balls. WIG_ROOT is the
# limit of that reach and is deliberately 4 px clear of the eye row -- an afro
# over his eyes stops being Clawd.
WIG_CY, WIG_CX = BY - 1, BX + SX // 2   # arc centre, level with his head top
WIG_RX, WIG_RY = 46, 19                 # arc radii (RX capped: the wig has to
                                        # stay inside his hands, or it crops)
WIG_ROOT = BY + 6                       # how far the mass reaches onto his head
PUFF_R = 13                             # main puff radius
HUES = [(W_RED, S_RED), (W_ORG, S_ORG), (W_YEL, S_YEL), (W_YEL, S_YEL),
        (W_GRN, S_GRN), (W_BLU, S_BLU), (W_PUR, S_PUR)]
DRAW_ORDER = [3, 2, 4, 1, 5, 0, 6]      # crown first, working outwards, so
                                        # every band overlaps the one inside it
                                        # and all six hues keep their share

NOSE_Y, NOSE_X = BY + 38, BX + SX // 2  # ball nose, low and centred
NOSE_R = 12

TIE_Y, TIE_X = BY + 58, BX + SX // 2    # bow tie, across his legs
WING_W, WING_HW = 22, 12                # wing length and outer half-height
KNOT_HW, KNOT_HH = 4, 6                 # knot half-width / half-height

# ---- motion ----------------------------------------------------------------
HONKS = 2               # honks per loop
BOB = 1.5               # gentle up-down (px)
RECOIL = 1.2            # how far he flinches back on the honk (px)
SWELL = 4.0             # nose growth on the honk (px)
SQUASH = 0.45           # share of the swell taken back off the height
RIPPLE = 2.0            # wig puff travel on the honk (px)
WAGGLE = 7.0            # bow tie rotation on the honk (degrees)


def fill_ellipse(a, cy, cx, ry, rx, color):
    h, w = a.shape
    for dy in range(-ry, ry + 1):
        y = cy + dy
        if not 0 <= y < h:
            continue
        span = rx * math.sqrt(max(0.0, 1 - (dy / ry) ** 2))
        for dx in range(-int(span), int(span) + 1):
            x = cx + dx
            if 0 <= x < w:
                a[y, x] = color


def puff(a, cy, cx, base, shade):
    """One cauliflower puff: a fat disk, two smaller bumps for the ragged top,
    a second disk beneath if it needs to reach down to the mass, and a shaded
    lower-right crescent left behind by redrawing a smaller disk over a
    shifted one."""
    fill_ellipse(a, cy, cx, PUFF_R, PUFF_R, base)
    if cy + PUFF_R < WIG_ROOT:
        fill_ellipse(a, WIG_ROOT - PUFF_R, cx, PUFF_R, PUFF_R, base)
    fill_ellipse(a, cy - PUFF_R + 5, cx - 6, 7, 7, base)
    fill_ellipse(a, cy - PUFF_R + 6, cx + 7, 6, 6, base)
    fill_ellipse(a, cy + 4, cx + 4, PUFF_R - 3, PUFF_R - 3, shade)
    fill_ellipse(a, cy - 1, cx - 1, PUFF_R - 4, PUFF_R - 4, base)


def wig_centres(ripple):
    """Puff centres along the dome. Stepped evenly across in x rather than by
    angle -- an even angle step bunches the outer puffs almost on top of each
    other and the hues at that end lose their band. `ripple` (px) lifts
    alternating puffs so the whole wig shivers on the honk."""
    out = []
    for i in range(len(HUES)):
        t = i / (len(HUES) - 1)
        u = 2 * t - 1                                 # -1 .. +1 across the head
        cy = WIG_CY - WIG_RY * math.sqrt(1 - u * u) + ripple * math.cos(2 * math.pi * t)
        cx = WIG_CX + WIG_RX * u
        out.append((int(round(cy)), int(round(cx))))
    return out


def build_body(ripple):
    """Clawd wearing the wig, as one silhouette under a single white outline."""
    a = np.zeros((AH, AW), dtype=np.uint8)
    for r, row in enumerate(ART):
        for c, ch in enumerate(row):
            if ch == ".":
                continue
            a[BY + r * SCALE:BY + (r + 1) * SCALE,
              BX + c * SCALE:BX + (c + 1) * SCALE] = EYE if ch == "O" else BODY

    centres = wig_centres(ripple)
    for i in DRAW_ORDER:                              # wig sits over his brow
        puff(a, *centres[i], *HUES[i])

    a[border_mask(a > 0, pen_disk(2))] = OUTLINE
    return a


def draw_nose(a, honk):
    """Squash-and-stretch ball nose: it widens and flattens as it flashes."""
    grow = SWELL * honk
    rx = int(round(NOSE_R + grow))
    ry = int(round(NOSE_R - grow * SQUASH))
    body = NOSE_HOT if honk > 0.55 else NOSE
    fill_ellipse(a, NOSE_Y, NOSE_X, ry + 1, rx + 1, NOSE_SHADE)
    fill_ellipse(a, NOSE_Y, NOSE_X, ry, rx, body)
    fill_ellipse(a, NOSE_Y + ry // 3, NOSE_X + rx // 3, ry // 2, rx // 2, NOSE_SHADE)
    fill_ellipse(a, NOSE_Y - 1, NOSE_X - 1, ry - 3, rx - 3, body)
    fill_ellipse(a, NOSE_Y - ry // 2, NOSE_X - rx // 3, ry // 4 + 1, rx // 4 + 1,
                 NOSE_SHEEN)


# dots are (wing side, along-wing fraction, height fraction, hue). Three a wing
# at DOT_R, not five smaller ones: 9 px across here is ~2 px in Slack, and
# anything under that stops being a spot and turns into speckle.
DOT_R = 4
DOTS = [(-1, 0.34, -0.30, W_RED), (-1, 0.66, 0.34, W_BLU), (-1, 0.62, -0.60, W_GRN),
        (1, 0.34, 0.30, W_GRN), (1, 0.66, -0.34, W_RED), (1, 0.62, 0.62, W_BLU)]


def build_tie():
    """The bow on its own layer, knot at the centre, so it can be rotated."""
    h = w = 73                                        # odd, so PIL's rotation
    a = np.zeros((h, w), dtype=np.uint8)              # centre is exactly the
    cy, cx = h // 2, w // 2                           # knot -- an even size
                                                      # wobbles it half a pixel

    for side in (-1, 1):
        for step in range(WING_W + 1):
            u = step / WING_W
            x = cx + side * (KNOT_HW + step)
            hh = KNOT_HH + (WING_HW - KNOT_HH) * math.sqrt(u)
            notch = 0.0
            if u > 0.80:                              # concave bite in the outer edge
                notch = (u - 0.80) / 0.20 * (WING_HW - 2)
            for dy in range(-int(hh), int(hh) + 1):
                if abs(dy) < notch:
                    continue
                a[cy + dy, x] = TIE_SHADE if dy > hh - 3 else TIE

    for side, along, high, hue in DOTS:
        x = cx + side * (KNOT_HW + along * WING_W)
        hh = KNOT_HH + (WING_HW - KNOT_HH) * math.sqrt(along)
        fill_ellipse(a, int(round(cy + high * hh)), int(round(x)), DOT_R, DOT_R, hue)

    fill_ellipse(a, cy, cx, KNOT_HH, KNOT_HW + 1, NOSE)
    a[border_mask(a > 0, pen_disk(2))] = OUTLINE
    return a


TIE_ART = build_tie()


def rotate(a, angle):
    im = Image.new("P", (a.shape[1], a.shape[0]))
    im.putpalette(PAL)
    im.frombytes(a.tobytes())
    return np.asarray(im.rotate(angle, resample=Image.NEAREST, expand=False,
                                fillcolor=0))


def blit(dst, a, y0, x0):
    """Paint a's non-zero pixels onto dst at (y0, x0), clipped to dst."""
    h, w = dst.shape
    ys, xs = np.nonzero(a)
    keep = (ys + y0 >= 0) & (ys + y0 < h) & (xs + x0 >= 0) & (xs + x0 < w)
    dst[ys[keep] + y0, xs[keep] + x0] = a[ys[keep], xs[keep]]


def build_frame(f):
    """The whole clown, posed for frame f, on the roomy work canvas."""
    ph = 2 * math.pi * f / F
    beat = 2 * math.pi * HONKS * f / F
    honk = (1 - math.cos(beat)) / 2                   # 0 -> 1 -> 0, twice a loop
    bob = int(round(BOB * math.sin(ph)))

    a = np.zeros((AH, AW), dtype=np.uint8)
    # the bow is pinned to his chest, so body and tie share one offset -- give
    # the tie its own and it slides against him every time he flinches
    drop = bob + int(round(RECOIL * honk))
    body = build_body(RIPPLE * math.sin(beat))
    draw_nose(body, honk)
    blit(a, body, drop, 0)

    tie = rotate(TIE_ART, WAGGLE * math.sin(beat))
    blit(a, tie, drop + TIE_Y - tie.shape[0] // 2, TIE_X - tie.shape[1] // 2)
    return a


# The whole loop is measured once and centred as a unit: the union bounding box
# over every frame is what has to fit, since the extremes of the bob and the
# honk are what would crop.
_ext = [np.nonzero(build_frame(f)) for f in range(F)]
_ys = np.concatenate([p[0] for p in _ext])
_xs = np.concatenate([p[1] for p in _ext])
OY = (N - (_ys.max() - _ys.min() + 1)) // 2 - _ys.min()
OX = (N - (_xs.max() - _xs.min() + 1)) // 2 - _xs.min()


def compose(f):
    g = np.zeros((N, N), dtype=np.uint8)
    blit(g, build_frame(f), OY, OX)
    return g


def save():
    frames = []
    for f in range(F):
        im = Image.frombytes("P", (N, N), compose(f).tobytes())
        im.putpalette(PAL)
        frames.append(im)

    frames[0].convert("RGBA").save(OUT / "clawd_clown_still.png")
    gif = OUT / "clawd_clown.gif"
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=DUR,
                   loop=0, transparency=T, disposal=2, optimize=False)
    kb = os.path.getsize(gif) / 1024
    assert kb <= 128, f"{gif.name} is {kb:.0f} KB — over Slack's 128 KB cap"
    print(f"clown: {F} frames @ {DUR}ms, gif={kb:.0f} KB")


if __name__ == "__main__":
    save()

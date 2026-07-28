#!/usr/bin/env python3
"""'Mad Clawd' -- heat vision. His eyes narrow to white-hot slits that streak
sideways off the edge of the frame.

Built to a reference (Superman heat-vision art), and two things there drive the
whole design:

  * The glow is a thin HORIZONTAL SLIT, not a round dot. A disc reads as a
    glowing eyeball; a slit reads as something being emitted.
  * The BODY DOES NOT CHANGE COLOUR. Only the eyes light up. Washing the whole
    figure to sell "light is happening" is exactly the mistake this replaces --
    it recolours the character instead of lighting him.

The slits deliberately run off the canvas: at full power each beam is wider
than the frame, so it is cut off at the edge rather than tidily contained.
That crop is the point (see NOTE below).

  power  : hand-authored, because the shape is the gag -- it snaps up in two
           frames, ripples through a shallow V at the top, and then bleeds
           down over eleven. Fast up, slow down; a sine cannot do that.
  beam   : an edge-on galaxy per eye -- a compact bright bulge with a long thin
           disc through it, the same shape on both, free to overlap. Genuinely
           ALPHA-BLENDED over whatever is beneath it (see the palette note), so
           his face and outline read through the glow instead of being erased.

NOTE: like walkin/walkout this deliberately breaks CONTRIBUTING's "never crop"
rule. The beams leaving frame IS the effect; contained beams read as glowing
eyes rather than as emission. Clawd himself is fully inside the canvas.

Seamless by construction: the power schedule is a hand-authored list of length
F that starts and ends at zero, so f=0 and f=F match exactly.
Outputs the still + gif.
"""
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.clawd import ART, CLAWD_RGB, EYE_RGB, WHITE_RGB, border_mask, pen_disk

OUT = Path(__file__).resolve().parent
NAME = "clawd_mad"

N = 128
F = 22
DUR = 80
SCALE = 10                       # 12x8 art -> 120x80: full canvas width

# ---- palette --------------------------------------------------------------
# The beam is genuinely ALPHA-BLENDED, not stippled. GIF has no alpha channel,
# but it does have 256 palette slots -- so every (underlying colour x beam
# level) blend is precomputed and given its own entry. Quantising the beam to
# LEVELS steps bounds that at 4 + 4*LEVELS entries, far inside the limit.
#
# The earlier version faked transparency with an ordered dither, which leaves
# literal HOLES in the glow. Blending the underlying pixel toward red instead
# gives a solid, actually-translucent beam.
BASE = [
    (0, 0, 0),                   # 0 transparent slot
    WHITE_RGB,                   # 1 outline
    CLAWD_RGB,                   # 2 body -- never replaced, only tinted
    EYE_RGB,                     # 3 eye socket, unlit
]
T, OUTLINE, BODY, SOCKET = range(4)

# Crimson ramp: (position, rgb), dimmest first. Interpolated per level.
STOPS = ((0.00, (88, 8, 10)), (0.30, (168, 16, 18)), (0.55, (228, 30, 30)),
         (0.80, (255, 96, 84)), (1.00, (255, 255, 255)))
LEVELS = 16                              # beam quantisation steps
AIR_MIN = 0.16                           # below this the beam over empty canvas
                                         # stays transparent, so it fades out


def _ramp(i):
    """Beam colour at intensity i, linearly interpolated between STOPS."""
    for k in range(len(STOPS) - 1):
        p0, c0 = STOPS[k]
        p1, c1 = STOPS[k + 1]
        if i <= p1:
            t = 0.0 if p1 == p0 else (i - p0) / (p1 - p0)
            return tuple(round(c0[j] + (c1[j] - c0[j]) * t) for j in range(3))
    return STOPS[-1][1]


def _alpha(i):
    """Opacity at intensity i. Slightly gained so the core goes fully solid."""
    return min(1.0, (i ** 0.8) * 1.15)


def _blend(under, over, a):
    return tuple(round(under[j] * (1 - a) + over[j] * a) for j in range(3))


# LUT[base_index][level] -> palette index of that base tinted to that level
COLORS = list(BASE)
LUT = {}
for _b in (T, OUTLINE, BODY, SOCKET):
    row = [_b]                                       # level 0 = untouched
    for _k in range(1, LEVELS + 1):
        _i = _k / LEVELS
        _c = _ramp(_i)
        _a = _alpha(_i)
        if _b == T:                                  # nothing behind it: the
            if _a < AIR_MIN:                         # glow just thins into air
                row.append(T)
                continue
            _rgb = _c
        else:
            _rgb = _blend(BASE[_b], _c, _a)
        COLORS.append(_rgb)
        row.append(len(COLORS) - 1)
    LUT[_b] = row
assert len(COLORS) <= 256, f"{len(COLORS)} palette entries"
PAL = bytes([c for rgb in COLORS for c in rgb] + [0] * (768 - 3 * len(COLORS)))

# Where the two eyes' discs overlap, the brighter level must win -- otherwise
# the second eye paints its own dim tail over the first eye's bright core.
BASE_OF, LEVEL_OF = {}, {}
for _b, _row in LUT.items():
    for _lv, _idx in enumerate(_row):
        if _lv:
            BASE_OF[_idx], LEVEL_OF[_idx] = _b, _lv

# ---- layout ----------------------------------------------------------------
SH, SW = 8 * SCALE, 12 * SCALE           # sprite 80 x 120
X0, Y0 = (N - SW) // 2, (N - SH) // 2    # centred: 4, 24

EYE_CY = Y0 + 1 * SCALE + SCALE // 2     # ART row 1 -> the eye row
EYE_CX = [X0 + c * SCALE + SCALE // 2 for c in (3, 8)]

# Each eye carries the SAME shape -- an edge-on galaxy: a compact bright bulge
# with a long thin disc running through it. Both eyes get it identically, not
# mirrored to fire away from each other, and the two discs are free to overlap
# across the middle of his face.
EYE_HALF = SCALE / 2                     # 5.0 -- what the bulge has to swallow
BULGE_W0, BULGE_W1 = 5.0, 13.0           # bulge half-width,  idle -> full
BULGE_H0, BULGE_H1 = 4.0, 8.0            # bulge half-height, idle -> full
DISC_W0, DISC_W1 = 12.0, 58.0            # disc half-length: overruns the canvas
DISC_H0, DISC_H1 = 1.4, 3.2              # disc half-height: stays thin

# The power curve IS the gag, so it is written out by hand rather than eased
# from a formula. Read left to right:
#   f0-f1    dead
#   f2-f3    charge -- 160ms, two frames, near enough a snap
#   f4       full
#   f5-f9    a shallow V: dips to 85% and recovers. The bright core only drops
#            to 88% of peak, so it reads as a ripple, not a stutter.
#   f10      full again
#   f11-f21  the long bleed down -- 880ms, five and a half times the rise
# Rising fast and falling slow is the whole character of it; an earlier version
# had the rise SLOWER than the fall and the curve leaned the wrong way.
POWER = [0.00, 0.00, 0.23, 0.58,
         1.00, 0.95, 0.90, 0.85, 0.90, 0.95, 1.00,
         0.89, 0.78, 0.67, 0.57, 0.47, 0.37, 0.28, 0.20, 0.12, 0.05, 0.00]
assert len(POWER) == F and POWER[0] == 0.0 and POWER[-1] == 0.0


def sprite():
    g = np.zeros((SH, SW), dtype=np.uint8)
    for r, row in enumerate(ART):
        for c, ch in enumerate(row):
            if ch != ".":
                g[r * SCALE:(r + 1) * SCALE,
                  c * SCALE:(c + 1) * SCALE] = SOCKET if ch == "O" else BODY
    return g


SPRITE = sprite()


def paint_beam(g, cy, cx, lvl):
    """One edge-on galaxy centred on an eye: a compact bulge plus a long thin
    disc, combined by taking whichever is brighter. Composited transparently
    (see the palette note) so it never erases what it covers."""
    bw = BULGE_W0 + (BULGE_W1 - BULGE_W0) * lvl
    bh = BULGE_H0 + (BULGE_H1 - BULGE_H0) * lvl
    dw = DISC_W0 + (DISC_W1 - DISC_W0) * lvl
    dh = DISC_H0 + (DISC_H1 - DISC_H0) * lvl

    y0, y1 = max(0, int(cy - max(bh, dh)) - 2), min(N, int(cy + max(bh, dh)) + 3)
    x0, x1 = max(0, int(cx - dw) - 2), min(N, int(cx + dw) + 3)
    for y in range(y0, y1):
        dv = y - cy
        for x in range(x0, x1):
            du = x - cx
            bulge = math.exp(-((du / bw) ** 2 + (dv / bh) ** 2))
            disc = math.exp(-((du / dw) ** 2 + (dv / dh) ** 2))
            inten = bulge if bulge > disc else disc
            if inten < 0.05:
                continue
            lv = min(LEVELS, int(round(inten * LEVELS)))
            if lv == 0:
                continue
            under = g[y, x]
            # already lit by the other eye? keep whichever level is brighter
            if under not in LUT:
                prev = LEVEL_OF.get(under, 0)
                if lv <= prev:
                    continue
                under = BASE_OF[under]
            g[y, x] = LUT[under][lv]


def compose(f):
    g = np.zeros((N, N), dtype=np.uint8)
    g[Y0:Y0 + SH, X0:X0 + SW] = SPRITE
    g[border_mask(g != 0, pen_disk(2))] = OUTLINE      # sacred 2 px, drawn first

    lvl = POWER[f]
    if lvl > 0.0:
        for cx in EYE_CX:                              # same shape on both eyes
            paint_beam(g, EYE_CY, cx, lvl)
    return g


def save():
    frames = []
    for f in range(F):
        im = Image.frombytes("P", (N, N), compose(f).tobytes())
        im.putpalette(PAL)
        frames.append(im)

    frames[POWER.index(1.00)].convert("RGBA").save(OUT / f"{NAME}_still.png")
    gif = OUT / f"{NAME}.gif"
    frames[0].save(
        gif, save_all=True, append_images=frames[1:], duration=DUR, loop=0,
        transparency=T, disposal=2, optimize=False,
    )
    kb = gif.stat().st_size / 1024
    assert kb <= 128, f"{gif.name} is {kb:.0f} KB — over Slack's 128 KB cap"
    print(f"{NAME}: {F} frames @ {DUR}ms, gif={kb:.0f} KB")


if __name__ == "__main__":
    save()

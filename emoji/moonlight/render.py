#!/usr/bin/env python3
"""'Moonlit Clawd' -- Clawd dozed off outside: slumped on a dark hill under a
big crescent moon, stars turning over him, Zs drifting off toward the light.

A full opaque scene (like bugcatcher and downunder) so the night reads on any
Slack theme, composited back-to-front:

  sky   : a banded night gradient, dither-blended at the seam, with a soft glow
          pooling around the moon.
  moon  : a fat crescent -- a disc with a bite taken out of it -- plus stars
          that twinkle on their own phases.
  Clawd : the authentic sprite (full 2px white outline) with his eyes shut,
          leaning as he sleeps. The lean is a static tilt plus a slow sway of a
          couple of degrees, rotated about his feet so he rocks rather than
          slides.
  hill  : a dark ridge across the bottom that his leg-tips sink into, with a
          few blades of grass catching the moonlight.
  Zs    : three Zs rising off him toward the moon.

Seamless by construction: the sway is sin(2*pi*f/F), every star's twinkle is
sin(2*pi*f/F + phase), and each Z's life is ((f/F) + phase) mod 1 -- all equal
at f=0 and f=F. Outputs the still + gif.
"""
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.clawd import ART, CLAWD_RGB, EYE_RGB, WHITE_RGB, shut_eye, border_mask, pen_disk

OUT = Path(__file__).resolve().parent
NAME = "clawd_moonlight"

N = 128
F = 24
DUR = 110
# SCALE is capped by the LEAN, not by the sprite: rotating about his feet gives
# him an effective half-width of (SW/2)*cos(t) + SH*sin(t), so at SCALE=9 any
# lean at all pushes his top corner off the canvas. SCALE=8 buys the tilt.
SCALE = 8                        # 12x8 art -> 96x64

# ---- palette ---------------------------------------------------------------
COLORS = [
    (0, 0, 0),                   # 0  unused (index 0 stays "nothing" while drawing)
    CLAWD_RGB,                   # 1  body
    EYE_RGB,                     # 2  (reserved: the open eye colour)
    WHITE_RGB,                   # 3  outline
    (12, 16, 44),                # 4  sky, high and dark
    (24, 32, 70),                # 5  sky, lower
    (44, 56, 104),               # 6  sky glow around the moon
    (248, 244, 206),             # 7  moon / bright star
    (186, 196, 232),             # 8  mid star
    (104, 116, 168),             # 9  dim star
    (22, 44, 34),                # 10 hill, dark
    (14, 30, 24),                # 11 hill, darker foot
    (56, 92, 62),                # 12 moonlit grass
    (18, 16, 30),                # 13 shut-lid arc
    (170, 190, 240),             # 14 Z blue
    (100, 116, 176),             # 15 Z blue, dimmed
]
(T, BODY, EYE, OUTLINE, SKY_D, SKY, GLOW, MOON, STAR_M, STAR_D,
 HILL, HILL_D, GRASS, LID, Z_NEAR, Z_FAR) = range(16)
pal_bytes = bytes([c for rgb in COLORS for c in rgb] + [0] * (768 - 3 * len(COLORS)))

OPAQUE = True                            # full-frame scene: no transparency
SUBJECT = (BODY, LID, OUTLINE)           # what the fit check measures

# ---- layout ----------------------------------------------------------------
SH, SW = 8 * SCALE, 12 * SCALE           # sprite 72 x 108
FEET_Y, FEET_X = 120, 63                 # where he sits on the hill (the pivot)

SKY_SPLIT = 46                           # band seam, dither-blended
MOON_CY, MOON_CX, MOON_R = 28, 30, 18
HILL_Y = 110                             # mean top of the ridge

LEAN = -7.0                              # static slump (degrees)
SWAY = 1.5                               # slow rock about his feet (degrees)

STARS = [                                # (y, x, size, phase)
    (12, 74, 2, 0.0), (22, 100, 2, 1.3), (8, 112, 1, 2.6), (40, 88, 1, 3.9),
    (34, 118, 2, 5.2), (56, 12, 1, 0.7), (52, 104, 1, 2.1), (66, 122, 1, 4.4),
    (18, 58, 1, 3.3), (44, 66, 1, 5.9),
]

Z_PHASES = (0.0, 1 / 3, 2 / 3)
Z_X0, Z_Y0 = 82, 46
Z_DX, Z_DY = 22, 40
Z_MIN, Z_MAX = 5, 13
Z_FADE, Z_GONE = 0.62, 0.92


def fill_disk(A, cy, cx, r, color):
    H, W = A.shape
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy * dy + dx * dx <= r * r:
                yy, xx = cy + dy, cx + dx
                if 0 <= yy < H and 0 <= xx < W:
                    A[yy, xx] = color


def blit(g, arr, y0, x0):
    ys, xs = np.nonzero(arr)
    for ry, rx in zip(ys, xs):
        wy, wx = y0 + ry, x0 + rx
        if 0 <= wy < N and 0 <= wx < N:
            g[wy, wx] = arr[ry, rx]




def build_body():
    """Clawd asleep on a padded layer, outline baked in, positioned so his feet
    sit at the bottom-centre -- that point is the rotation pivot, so the lean
    rocks him instead of sliding him off the hill."""
    pad = 6
    A = np.zeros((SH + 2 * pad, SW + 2 * pad), dtype=np.uint8)
    eyes = []
    for r, row in enumerate(ART):
        for c, ch in enumerate(row):
            if ch == ".":
                continue
            A[pad + r * SCALE:pad + (r + 1) * SCALE,
              pad + c * SCALE:pad + (c + 1) * SCALE] = BODY
            if ch == "O":
                eyes.append((r, c))
    for r, c in eyes:
        shut_eye(A, pad + r * SCALE, pad + c * SCALE, SCALE, LID)
    A[border_mask(A > 0, pen_disk(2))] = OUTLINE
    return A, (pad + SH, pad + SW // 2)      # pivot: between the leg tips


BODY_SPR, PIVOT = build_body()


def _center_on(A, cy, cx):
    """Pad A so (cy, cx) sits at the exact centre -> rotation pivots there."""
    H, W = A.shape
    py, px = max(cy, H - 1 - cy), max(cx, W - 1 - cx)
    out = np.zeros((2 * py + 1, 2 * px + 1), dtype=np.uint8)
    out[py - cy:py - cy + H, px - cx:px - cx + W] = A
    return out


CEN_BODY = _center_on(BODY_SPR, *PIVOT)


def rot(arr, angle):
    im = Image.new("P", (arr.shape[1], arr.shape[0]))
    im.putpalette(pal_bytes)
    im.frombytes(arr.tobytes())
    r = im.rotate(angle, resample=Image.NEAREST, expand=True, fillcolor=0)
    return np.asarray(r)


# a static ragged profile for the ridge line
_ridge = np.array([
    round(3.0 * math.sin(x * 0.045) + 1.6 * math.sin(x * 0.13 + 0.8))
    for x in range(N)])


def draw_sky(g):
    g[:, :] = SKY
    g[:SKY_SPLIT, :] = SKY_D
    for k, y in enumerate(range(SKY_SPLIT, SKY_SPLIT + 4)):   # dither the seam
        g[y, (k % 2)::2] = SKY_D if k < 2 else SKY
    for r in range(MOON_R + 16, MOON_R, -4):                  # glow pooling
        fill_disk(g, MOON_CY, MOON_CX, r, GLOW if r <= MOON_R + 8 else SKY_D)


def draw_moon(g):
    fill_disk(g, MOON_CY, MOON_CX, MOON_R, MOON)
    fill_disk(g, MOON_CY - 5, MOON_CX + 8, MOON_R - 1, GLOW)  # the bite


def draw_stars(g, f):
    ph = 2 * math.pi * f / F
    for (y, x, size, phase) in STARS:
        tw = math.sin(ph + phase)
        color = MOON if tw > 0.45 else (STAR_M if tw > -0.4 else STAR_D)
        arm = size + (1 if tw > 0.6 else 0)
        for d in range(-arm, arm + 1):
            for (yy, xx) in ((y + d, x), (y, x + d)):
                if 0 <= yy < N and 0 <= xx < N:
                    g[yy, xx] = color


BLADES = [(10, 7, 0.4), (26, 5, 1.9), (48, 6, 3.2), (86, 5, 0.9),
          (104, 7, 2.4), (118, 6, 4.1), (68, 4, 5.3)]


def draw_hill(g, f):
    """The ridge he's sleeping on, drawn AFTER Clawd so his leg-tips sink in."""
    for x in range(N):
        top = HILL_Y + _ridge[x]
        g[max(0, top):, x] = HILL
        g[max(0, top + 10):, x] = HILL_D
    ph = 2 * math.pi * f / F
    for (bx, h, phase) in BLADES:                         # blades catching light
        sway = 1.6 * math.sin(ph + phase)
        base = HILL_Y + _ridge[min(bx, N - 1)]
        for t in range(h):
            frac = t / h
            y = base - t
            x = int(round(bx + sway * frac * frac))
            if 0 <= y < N and 0 <= x < N:
                g[y, x] = GRASS


def z_glyph(size):
    t = max(1, size // 4)
    pad = 2
    A = np.zeros((size + 2 * pad, size + 2 * pad), dtype=np.uint8)
    x0, y0 = pad, pad
    A[y0:y0 + t, x0:x0 + size] = Z_NEAR
    A[y0 + size - t:y0 + size, x0:x0 + size] = Z_NEAR
    for s in range(size):
        u = s / max(size - 1, 1)
        yy = int(round(y0 + (size - t) * (1 - u)))
        xx = int(round(x0 + (size - t) * u))
        A[yy:yy + t, xx:xx + t] = Z_NEAR
    A[border_mask(A > 0, pen_disk(1))] = SKY_D           # dark rim against the sky
    return A


GLYPHS = {s: z_glyph(s) for s in range(Z_MIN, Z_MAX + 1)}


def paint_zs(g, f):
    for phase in Z_PHASES:
        life = ((f / F) + phase) % 1.0
        if life > Z_GONE:
            continue
        size = int(round(Z_MIN + (Z_MAX - Z_MIN) * life))
        A = GLYPHS[size].copy()
        if life > Z_FADE:
            A[A == Z_NEAR] = Z_FAR
        y = int(round(Z_Y0 - Z_DY * life))
        x = int(round(Z_X0 + Z_DX * life + 3 * math.sin(6 * life)))
        blit(g, A, y, x)


def compose(f):
    g = np.zeros((N, N), dtype=np.uint8)
    ph = 2 * math.pi * f / F

    draw_sky(g)
    draw_stars(g, f)
    draw_moon(g)

    R = rot(CEN_BODY, LEAN + SWAY * math.sin(ph))        # slumped, rocking gently
    blit(g, R, FEET_Y - R.shape[0] // 2, FEET_X - R.shape[1] // 2)

    draw_hill(g, f)                                      # his leg-tips sink in
    paint_zs(g, f)
    return g


def save():
    frames = []
    for f in range(F):
        im = Image.frombytes("P", (N, N), compose(f).tobytes())
        im.putpalette(pal_bytes)
        frames.append(im)

    frames[0].convert("RGBA").save(OUT / f"{NAME}_still.png")
    gif = OUT / f"{NAME}.gif"
    # the night fills the whole square, so write full opaque frames (see
    # bugcatcher): a transparency index would fight disposal and flicker.
    frames[0].save(
        gif, save_all=True, append_images=frames[1:], duration=DUR, loop=0,
        disposal=1, optimize=False,
    )
    kb = gif.stat().st_size / 1024
    assert kb <= 128, f"{gif.name} is {kb:.0f} KB — over Slack's 128 KB cap"
    print(f"{NAME}: {F} frames @ {DUR}ms, gif={kb:.0f} KB")


if __name__ == "__main__":
    save()

#!/usr/bin/env python3
"""Clawdlock Holmes — Clawd in a deerstalker, peering through a magnifying glass.

The glass pivots on his own right hand. It comes up over his right eye, he
squints the other one, sways side to side while he studies you, then lowers it.
The lens really magnifies: it resamples the frame underneath it, so what you
see in the glass is his eye, blown up.
"""
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.clawd import ART, CLAWD_RGB, EYE_RGB, WHITE_RGB, border_mask, pen_disk

OUT = Path(__file__).resolve().parent
NAME = "clawd_detective"

N = 128                 # canvas (Slack emoji size)
F = 36                  # frames in the loop
DUR = 60                # ms per frame (multiple of 10: GIFs store centiseconds)
SCALE = 9               # 108x72 sprite: as wide as the sway allows
STILL = 18              # mid-peer frame for the gallery still

RAISE = (4, 11)         # frames over which the glass comes up
LOWER = (27, 33)        # ... and goes back down; frame 0 == frame F at rest
SWAY = 4                # px he leans either way while peering
SCAN = 2                # px the lens drifts across his eye while he leans
ZOOM = 1.7              # lens magnification at the start of the peer
ZOOM_PULSE = 0.3        # extra magnification at the middle of it

LENS_R = 16             # outer radius of the rim
RIM = 3                 # rim thickness
REST_ANG = 120          # degrees: lens hanging down in front of him
SQUINT_AT = 0.55        # how far up the glass is before the other eye squints

COLORS = [
    (0, 0, 0),          # 0: transparent slot
    WHITE_RGB,          # 1: outline
    CLAWD_RGB,          # 2: body
    EYE_RGB,            # 3: eyes
    (196, 160, 118),    # 4: hat tweed
    (150, 108, 70),     # 5: hat check
    (110, 72, 42),      # 6: hat band, bills, bow
    (64, 64, 72),       # 7: rim
    (170, 170, 182),    # 8: rim highlight
    (128, 80, 44),      # 9: handle
    (205, 232, 245),    # 10: glass
]
T, OUTLINE, BODY, EYE, TWEED, CHECK, BAND, RIM_C, RIM_HI, HANDLE, GLASS = range(11)
PAL = bytes([c for rgb in COLORS for c in rgb] + [0] * (768 - 3 * len(COLORS)))

GY, GX = len(ART), len(ART[0])
SH, SW = GY * SCALE, GX * SCALE
HAT_H = round(2.95 * SCALE)
Y0 = (N - SH - HAT_H) // 2 + HAT_H
X0 = (N - SW) // 2

EYE_CELLS = [(r, c) for r, row in enumerate(ART) for c, ch in enumerate(row) if ch == "O"]
SQUINT_EYE = min(EYE_CELLS, key=lambda rc: rc[1])
LOOK_EYE = max(EYE_CELLS, key=lambda rc: rc[1])
HAND = (2 * SCALE + SCALE, (GX - 1) * SCALE + SCALE // 4)
EYE_C = (LOOK_EYE[0] * SCALE + SCALE / 2, LOOK_EYE[1] * SCALE + SCALE / 2)
UP_ANG = math.degrees(math.atan2(EYE_C[0] - HAND[0], EYE_C[1] - HAND[1])) % 360
REACH = math.hypot(EYE_C[0] - HAND[0], EYE_C[1] - HAND[1])


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def raised(f):
    """0 at rest, 1 with the glass at his eye."""
    if f < RAISE[0] or f >= LOWER[1]:
        return 0.0
    if f < RAISE[1]:
        return ease((f - RAISE[0]) / (RAISE[1] - RAISE[0] - 1))
    if f < LOWER[0]:
        return 1.0
    return 1 - ease((f - LOWER[0]) / (LOWER[1] - LOWER[0] - 1))


def peer(f):
    """0 -> 1 across the frames the glass is fully up, else None."""
    if RAISE[1] <= f < LOWER[0]:
        return (f - RAISE[1]) / (LOWER[0] - RAISE[1])
    return None


def disk(g, cy, cx, r, color):
    yy, xx = np.ogrid[:N, :N]
    g[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = color


def thick_line(g, y0, x0, y1, x1, r, color):
    n = int(max(abs(y1 - y0), abs(x1 - x0))) + 1
    for i in range(n + 1):
        t = i / n
        disk(g, y0 + (y1 - y0) * t, x0 + (x1 - x0) * t, r, color)


def draw_clawd(g, x0, squint):
    for r, row in enumerate(ART):
        for c, ch in enumerate(row):
            if ch == ".":
                continue
            ty, tx = Y0 + r * SCALE, x0 + c * SCALE
            g[ty:ty + SCALE, tx:tx + SCALE] = EYE if ch == "O" else BODY
    if squint:
        r, c = SQUINT_EYE
        ty, tx = Y0 + r * SCALE, x0 + c * SCALE
        g[ty:ty + SCALE, tx:tx + SCALE] = BODY
        lid = max(3, SCALE // 3)
        mid = ty + SCALE // 2 + 1
        g[mid - lid // 2:mid - lid // 2 + lid, tx - 1:tx + SCALE + 1] = EYE


def draw_hat(g, x0):
    s = SCALE
    yy, xx = np.mgrid[:N, :N]
    cy, cx = Y0 - 0.35 * s, x0 + 6 * s
    dome = (((yy - cy) / (2.3 * s)) ** 2 + ((xx - cx) / (4.3 * s)) ** 2 <= 1) & (yy <= cy)
    check = ((yy - Y0) % 6 < 2) | ((xx - x0) % 6 < 2)
    g[dome] = TWEED
    g[dome & check] = CHECK

    def box(y_a, y_b, x_a, x_b):
        g[round(Y0 + y_a * s):round(Y0 + y_b * s), round(x0 + x_a * s):round(x0 + x_b * s)] = BAND

    box(-0.6, 0.35, 1.5, 10.5)
    box(-0.2, 0.45, 0.4, 1.5)
    box(-0.2, 0.45, 10.5, 11.6)
    box(-2.95, -2.45, 5.5, 6.5)


def compose(f):
    r = raised(f)
    p = peer(f)
    sway = round(SWAY * math.sin(2 * math.pi * p)) if p is not None else 0
    scan = SCAN * math.cos(2 * math.pi * p) if p is not None else 0.0
    zoom = ZOOM + (ZOOM_PULSE * math.sin(math.pi * p) if p is not None else 0.0)
    x0 = X0 + sway

    base = np.zeros((N, N), dtype=np.uint8)
    draw_clawd(base, x0, squint=r > SQUINT_AT)
    draw_hat(base, x0)
    base[border_mask(base != 0, pen_disk(2))] = OUTLINE

    ang = math.radians(REST_ANG + (UP_ANG - REST_ANG) * r)
    hy, hx = Y0 + HAND[0], x0 + HAND[1]
    ly = hy + REACH * math.sin(ang)
    lx = hx + REACH * math.cos(ang) + scan

    g = base.copy()
    ey, ex = ly - LENS_R * math.sin(ang), lx - LENS_R * math.cos(ang)
    thick_line(g, ey, ex, hy, hx, 2, HANDLE)

    yy, xx = np.mgrid[:N, :N]
    d = np.hypot(yy - ly, xx - lx)
    inner = d < LENS_R - RIM
    sy = np.clip(np.round(ly + (yy - ly) / zoom).astype(int), 0, N - 1)
    sx = np.clip(np.round(lx + (xx - lx) / zoom).astype(int), 0, N - 1)
    seen = base[sy, sx]
    g[inner] = np.where(seen[inner] == 0, GLASS, seen[inner])

    rim = (d >= LENS_R - RIM) & (d <= LENS_R)
    g[rim] = RIM_C
    g[rim & (yy < ly) & (xx < lx)] = RIM_HI
    gy, gx = round(ly - LENS_R * 0.5), round(lx - LENS_R * 0.5)
    g[gy:gy + 3, gx:gx + 2] = OUTLINE
    g[gy:gy + 2, gx:gx + 3] = OUTLINE

    g[border_mask(g != 0, pen_disk(2))] = OUTLINE
    return g


def save():
    frames = []
    for f in range(F):
        im = Image.frombytes("P", (N, N), compose(f).tobytes())
        im.putpalette(PAL)
        frames.append(im)

    assert np.array_equal(compose(0), compose(F)), "loop seam: frame F must equal frame 0"

    frames[STILL].convert("RGBA").save(OUT / f"{NAME}_still.png")

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

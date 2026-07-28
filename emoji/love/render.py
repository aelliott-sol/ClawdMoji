#!/usr/bin/env python3
"""'Clawd Love' -- Clawd stands there feeling something, and a heart swells up
off his head as he throws his hands up, floats away, and bursts.

ORIGIN: found as `:Clawd_Love:` in a Claude Discord server, then rebuilt here
from scratch and improved. No pixels are copied from it -- the original is not
even pixel art (100+ anti-aliased colours, so it was drawn or scaled in an
editor rather than generated). What was taken from it is the STAGING, recovered
by measuring the file frame by frame. What measurement did recover is the staging:
Clawd sits along the bottom edge at roughly cell 7, dips very slightly, then a
heart appears at his crown around frame 7, rises the full height of the canvas
while growing, and then BURSTS.

The burst is the whole point, and it was easy to get wrong: a single frame of
it looks like an outline heart, which reads as a hollow-out. Isolating the red
pixels frame by frame says otherwise -- past the peak the bounding box keeps
GROWING (56 -> 75 px wide) while the pixel count collapses (1579 -> 21) and
fill drops 56% -> 1%. Extent up, mass down, over four frames: that is a shell
flying apart, slowly. Nothing there fades.

  Clawd : the authentic sprite, breathing with a squash-and-stretch (his bottom
          edge pinned) rather than a bob.
  arms  : his hand-bumps are their own rotatable layers, so he can crouch and
          THROW them overhead to launch the heart, then lower them and land.
          Measured, not invented -- see the note by RAISE below.
  heart : a real heart curve -- ((u^2+v^2-1)^3 - u^2*v^3 <= 0) -- rasterised
          per size, so it holds its shape at every size as it swells and rises.
  pop   : the shell expands past the peak, its rim thins, and it breaks into
          arcs that shrink toward their own centres as they travel.

Seamless by construction: the breath is sin(2*pi*f/F) and the heart's life is a
pure function of f that is absent at both f=0 and f=F-1.
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
NAME = "clawd_love"

N = 128
F = 24
DUR = 90
SCALE = 8                        # 12x8 art -> 96x64, leaving the heart its sky

# ---- palette (P-mode GIF: index 0 is transparent) --------------------------
COLORS = [
    (0, 0, 0),                   # 0 transparent slot
    WHITE_RGB,                   # 1 outline
    CLAWD_RGB,                   # 2 body
    EYE_RGB,                     # 3 eyes
    (231, 69, 69),               # 4 heart red   (sampled from the original)
    (255, 138, 138),             # 5 heart highlight
]
T, OUTLINE, BODY, EYE, HEART, HEART_HI = range(6)
PAL = bytes([c for rgb in COLORS for c in rgb] + [0] * (768 - 3 * len(COLORS)))

# ---- layout ----------------------------------------------------------------
SH, SW = 8 * SCALE, 12 * SCALE           # sprite 64 x 96
X0 = (N - SW) // 2                       # 16
Y0 = N - SH - 4                          # 60 -- he stands on the bottom edge

BREATH = 2                               # px of stretch at the top of the inhale

# ---- the heart -------------------------------------------------------------
# The rise is bounded by the heart's own size, not by the canvas: the glyph is
# blitted about its centre, so the topmost centre it can reach is (H_R1 + pad +
# outline) below the edge. Push H_Y1 past that and it silently loses its lobes.
# The heart does NOT inflate smoothly. Measured off the original, it is small
# for exactly ONE frame, then holds a regular size for FOUR while it rises, and
# only then starts growing:
#     f7        14 px wide   small, one frame
#     f8-f11    32 px wide   held -- rising, not growing
#     f12-f14   39 -> 56     now it swells
# A single continuous ramp (what this had) skips the hold entirely and reads as
# one long inflate instead of "thrown, floats a beat, then swells".
HEART_R = {7: 6, 8: 12, 9: 12, 10: 12, 11: 12, 12: 15, 13: 19, 14: 22}
HEART_Y = {7: 48, 8: 38, 9: 36, 10: 35, 11: 34, 12: 34, 13: 34, 14: 34}
POP = range(15, 20)                      # frames it bursts and scatters
H_R0, H_R1 = 5, 22                       # half-width at birth / at the peak
H_Y1 = 34                                # where it is when it goes

# The burst, measured off the original rather than guessed: from the peak its
# bounding box keeps GROWING (56 -> 75 px wide) while the pixel count collapses
# (1579 -> 21) and fill falls 56% -> 1%. That is a shell flying apart, not a
# fade and not a hollow-out -- so the shell expands, its rim thins, and it
# breaks into arcs that shrink toward their own centres as they travel.
POP_SPREAD = 0.40                        # how far the OUTER tips reach by the end
POP_RAYS = 34                            # how many streaks the shell throws
POP_THIN = 0.80                          # fraction of them gone by the last frame


# ---- the throw ------------------------------------------------------------
# Measuring the ORIGINAL's body pixels (heart excluded) shows a real gesture,
# not a static pose: 86 px wide at rest -> 56 px while the heart launches (his
# two side-bumps leave the silhouette entirely) -> 72 -> back to 86, with the
# body reaching 28 px above his head at the peak and squashing on the landing.
# He crouches, throws both arms overhead, and settles.
#
# So his hand-bumps are built as their own rotatable layers -- the mariachi /
# bugcatcher pattern -- and the ART core is rasterised WITHOUT them. At angle 0
# and rest length the layer covers the bump's exact footprint, so the resting
# silhouette is still the authentic Clawd.
BUMP_ROWS = (2, 3)                       # the wide ART rows: his arms
BUMP_COLS = (0, 1, 10, 11)               # ...and the cells that are his hands
# Read frame by frame, the original's raised hands are SHORT nubs parked at the
# top corners of his head and angled outward -- they never elongate into limbs
# and they never pass across his face. His eyes stay visible in every frame of
# the throw. Rotating a long arm about the shoulder sweeps it straight over the
# eyes, which is wrong; the hands travel UP and slightly INBOARD instead.
ARM_UP = 90.0                            # raised angle: hands go fully UP
ARM_L0, ARM_L1 = 2.0, 4.5                # cells: the ART bump -> a raised hand
# The raised pivot must sit INSIDE the head, not level with its crown. Parked on
# the crown the rotated bar meets the body at a single corner and reads as a
# detached block floating beside him. Rooting it in the forehead -- below the
# skull line, above the eye row -- keeps the join solid while the hand itself
# still clears his face entirely.
RAISE_Y = Y0 + 0.5 * SCALE               # rooted in the forehead, above the eyes
RAISE_DX = 1.4 * SCALE                   # inboard, but wide enough to read

# The throw and the heart must be COUPLED: the release happens on the peak of
# the raise, so the hands are what launch it. Peaking after the heart already
# exists reads as two unrelated animations sharing a canvas -- the heart drifts
# off on its own and the arms flap somewhere behind it.
# The hands SNAP. In the original they are down at f6 and fully up at f7 -- one
# frame -- then hold for four and drop over two. A gradual ramp reads as a
# stretch, not a throw.
RAISE = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.92,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#                        crouch --^  ^-- snap        drop --^
# Positive squashes, negative stretches; feet pinned either way.
CROUCH = [0, 0, 0, 0, 0, 5, 5, -3, -3, -3, -3, 0,
          3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
assert len(RAISE) == F and len(CROUCH) == F
assert RAISE[0] == 0.0 and CROUCH[0] == 0          # wraps to frame 0 exactly


def sprite():
    """The ART core WITHOUT the hand-bumps -- those are drawn as arm layers."""
    g = np.zeros((SH, SW), dtype=np.uint8)
    for r, row in enumerate(ART):
        for c, ch in enumerate(row):
            if ch == "." or (r in BUMP_ROWS and c in BUMP_COLS):
                continue
            g[r * SCALE:(r + 1) * SCALE,
              c * SCALE:(c + 1) * SCALE] = EYE if ch == "O" else BODY
    return g


SPRITE = sprite()

# shoulders in world coords: the inner edge of each bump, mid-way down the band
SHO_Y = Y0 + 3 * SCALE
SHO_XR = X0 + 10 * SCALE                 # right arm grows +x from here
SHO_XL = X0 + 2 * SCALE                  # left arm grows -x from here


def build_arm(cells):
    """One hand-bump extending +x from a pivot at its inner edge. At ARM_L0 it
    is exactly the two ART cells it replaces; the throw stretches it."""
    L = max(1, int(round(cells * SCALE)))
    th = 2 * SCALE                                   # the band is two cells tall
    pad = 4
    A = np.zeros((th + 2 * pad, L + 2 * pad), dtype=np.uint8)
    A[pad:pad + th, pad:pad + L] = BODY
    return A, (pad + th // 2, pad)                   # pivot: inner edge, centred


def _center_on(A, cy, cx):
    """Pad A so (cy, cx) is the exact centre -> rotation pivots there."""
    H, W = A.shape
    py, px = max(cy, H - 1 - cy), max(cx, W - 1 - cx)
    out = np.zeros((2 * py + 1, 2 * px + 1), dtype=np.uint8)
    out[py - cy:py - cy + H, px - cx:px - cx + W] = A
    return out


def rot(arr, angle):
    im = Image.new("P", (arr.shape[1], arr.shape[0]))
    im.putpalette(PAL)
    im.frombytes(arr.tobytes())
    return np.asarray(im.rotate(angle, resample=Image.NEAREST, expand=True,
                                fillcolor=0))


def place_arms(g, lift, u):
    """Both arms at throw fraction u: they lengthen and swing overhead, mirrored
    so they rise together."""
    # Length trails the lift (u^1.6): extended early it swings out sideways on
    # the single descent frame and reads as a Y-pose rather than hands dropping.
    A, piv = build_arm(ARM_L0 + (ARM_L1 - ARM_L0) * u ** 1.6)
    cen = _center_on(A, *piv)
    ang = ARM_UP * u
    # The pivot itself travels: shoulder -> just above the crown, tucked inboard.
    # That is what keeps the hands clear of the eyes and keeps him inside his 8
    # core cells while they are up, exactly as the original does.
    # The two axes are eased DIFFERENTLY on purpose. Interpolated together the
    # hand cuts the corner and passes straight over an eye mid-travel (13 eye
    # pixels survived at u=0.45). Lifting fast while tucking late routes it up
    # the outside of his head first, then inboard once it is clear of the eyes.
    py = SHO_Y + (RAISE_Y - SHO_Y) * u ** 0.55
    for sx, sign in ((SHO_XR, +1), (SHO_XL, -1)):
        px = sx - sign * RAISE_DX * u ** 3
        layer = cen if sign > 0 else cen[:, ::-1].copy()
        R = rot(layer, ang * sign)                   # both tilt outward, not across
        blit(g, R, int(round(py)) - lift - R.shape[0] // 2,
             int(round(px)) - R.shape[1] // 2)


def stretched(A, dh):
    """Vertically rescale a layer by dh px, nearest-neighbour."""
    if dh == 0:
        return A
    h, w = A.shape
    im = Image.new("P", (w, h))
    im.frombytes(A.tobytes())
    return np.asarray(im.resize((w, h + dh), Image.NEAREST))


def heart_glyph(r):
    """A solid heart of half-width r, on its own layer with a 2 px white outline.

    Uses the actual heart curve rather than a hand-plotted sprite, so it stays
    the same shape at every size as it grows:
        (u^2 + v^2 - 1)^3 - u^2 * v^3 <= 0
    """
    pad = 3
    size = 2 * r + 1 + 2 * pad
    A = np.zeros((size, size), dtype=np.uint8)
    cy, cx = size // 2, size // 2
    # Map the cell to u,v in [-1.2, +1.2] SYMMETRICALLY. The curve's lobes sit
    # at v ~ +0.5..+1.0, so any mapping that tops out below +1 shears them off
    # and the heart renders as a plain triangle.
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            u = dx / r * 1.2
            v = -dy / r * 1.2
            if (u * u + v * v - 1) ** 3 - u * u * v ** 3 <= 0:
                A[cy + dy, cx + dx] = HEART
    A[cy - r // 2, cx - r // 2] = HEART_HI            # a little shine
    A[cy - r // 2, cx - r // 2 + 1] = HEART_HI
    A[border_mask(A > 0, pen_disk(2))] = OUTLINE
    return A


# pre-render the solid heart at every size the rise ever asks for
GLYPHS = {r: heart_glyph(r) for r in range(H_R0, H_R1 + 1)}


def heart_mask(cy, cx, r):
    """The heart shape at radius r, as a canvas-sized boolean mask."""
    m = np.zeros((N, N), dtype=bool)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            u, v = dx / r * 1.2, -dy / r * 1.2
            if (u * u + v * v - 1) ** 3 - u * u * v ** 3 <= 0:
                yy, xx = cy + dy, cx + dx
                if 0 <= yy < N and 0 <= xx < N:
                    m[yy, xx] = True
    return m


def _heart_radius(theta):
    """Normalised radius of the heart curve along a direction (binary search)."""
    lo, hi = 0.0, 1.6
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        u, v = mid * math.cos(theta), mid * math.sin(theta)
        if (u * u + v * v - 1) ** 3 - u * u * v ** 3 <= 0:
            lo = mid
        else:
            hi = mid
    return lo


# one ray per direction, each launched from the heart's own outline, so the
# first burst frame still carries the heart's shape as a ring of spikes
RAY_DIRS = [(th, _heart_radius(th))
            for th in (2 * math.pi * (k + 0.5) / POP_RAYS for k in range(POP_RAYS))]


def paint_pop(g, cy, cx, p):
    """The burst drawn as radial STREAKS. Each ray runs along one of the heart's
    own rim directions; the inner end pulls away from the centre while the ray
    shortens, so the shape flies apart outward instead of dissolving in place.

    Rays start at the CENTRE, not at the rim: launched from the rim the first
    burst frame is a hollow ring carrying 7% fill, where the original still
    holds 36% -- it has barely come apart yet. Rays are also retired
    progressively (golden-ratio order, so the survivors stay evenly spread)
    which is what makes the tail thin out instead of stopping dead."""
    for k, (th, rn) in enumerate(RAY_DIRS):
        if (k * 0.6180339887) % 1.0 >= 1.0 - POP_THIN * p:
            continue                                  # this streak is spent
        rim = rn * H_R1 / 1.2
        L = max(1, int(round(rim * (1 - p) ** 2)))
        # Drive the OUTER tip, then place the inner end behind it. Driving the
        # inner end instead makes the streaks shorten faster than they travel,
        # so the burst's extent SHRINKS (31 px -> 19 px) where the original's
        # grows (56 -> 75). Extent up, mass down is the whole signature.
        r0 = max(0.0, rim * (1 + POP_SPREAD * p) - L)
        dy, dx = -math.sin(th), math.cos(th)
        py, px = -dx, dy                              # unit perpendicular
        for t in range(L):
            d = r0 + t
            for w in (0, 1):                          # 2 px wide: reads at 32 px
                y = int(round(cy + dy * d + py * w))
                x = int(round(cx + dx * d + px * w))
                if 0 <= y < N and 0 <= x < N:
                    g[y, x] = HEART


def blit(g, arr, y0, x0):
    h, w = arr.shape
    ys, xs = max(0, -y0), max(0, -x0)
    ye, xe = min(h, N - y0), min(w, N - x0)
    if ys >= ye or xs >= xe:
        return
    src = arr[ys:ye, xs:xe]
    view = g[y0 + ys:y0 + ye, x0 + xs:x0 + xe]
    view[src != 0] = src[src != 0]


def compose(f):
    g = np.zeros((N, N), dtype=np.uint8)
    ph = 2 * math.pi * f / F
    swell = int(round(BREATH * 0.5 * (1 - math.cos(ph)))) - CROUCH[f]

    chest = stretched(SPRITE, swell)                  # breath, minus the crouch
    region = g[Y0 - swell:Y0 - swell + SH + swell, X0:X0 + SW]   # bottom pinned
    region[chest != 0] = chest[chest != 0]

    # the shoulders sit 5/8 of the way up from his feet, so they carry that much
    # of whatever the body is doing vertically
    place_arms(g, int(round(swell * 0.625)), RAISE[f])

    g[border_mask(g != 0, pen_disk(2))] = OUTLINE     # sacred 2 px, whole shape

    if f in HEART_R:                                  # small -> held -> swelling
        A = GLYPHS[HEART_R[f]]
        blit(g, A, HEART_Y[f] - A.shape[0] // 2, (N - A.shape[1]) // 2)
    elif f in POP:                                    # ...and then it goes
        p = (f - POP.start + 1) / len(POP)
        paint_pop(g, H_Y1, N // 2, p)
    return g


def save():
    frames = []
    for f in range(F):
        im = Image.frombytes("P", (N, N), compose(f).tobytes())
        im.putpalette(PAL)
        frames.append(im)

    frames[14].convert("RGBA").save(OUT / f"{NAME}_still.png")   # heart mid-rise
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

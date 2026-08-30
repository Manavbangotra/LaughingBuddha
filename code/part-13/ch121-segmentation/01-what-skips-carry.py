# -*- coding: utf-8 -*-
# Extracted from: Chapter 121 — Segmentation and the Segment Anything Model
# Source: src/.../ch121-segmentation.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""What skip connections are actually for, and why mean IoU hides it.

ch:mm-cv-fundamentals stated the tension every dense-prediction task has: context
wants downsampling and localisation wants resolution (eq:resolution-tension).
cite:ronneberger2015unet's answer is to do both -- downsample for context, then
upsample, and carry the high-resolution detail across on skip connections.

This listing measures what that carrying is worth. Nothing is trained: the
question is what an encoder-decoder CAN represent, and the answer without skips
is bounded by the information surviving the bottleneck
(eq:bottleneck-iou-bound). An upper bound is a stronger statement than a trained
result, because no amount of training beats it.

The measurement is split into interior pixels and boundary pixels, because
reporting one number for both is exactly how this effect stays invisible.
"""
import numpy as np

rng = np.random.default_rng(29)

H = 128
N_SHAPE = 200


def make_mask(radius, thin=False):
    """A blob, optionally with a thin protrusion -- the structure that dense
    prediction is asked for and that a coarse grid cannot hold."""
    yy, xx = np.mgrid[0:H, 0:H]
    cy, cx = rng.uniform(0.35 * H, 0.65 * H, size=2)
    # A wobbly blob: radius modulated by a few Fourier components.
    ang = np.arctan2(yy - cy, xx - cx)
    r = radius * (1.0 + 0.18 * np.sin(3 * ang + rng.uniform(0, 6))
                  + 0.12 * np.sin(5 * ang + rng.uniform(0, 6)))
    m = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r ** 2
    if thin:
        w = max(int(radius * 0.12), 1)
        y0 = int(np.clip(cy, 0, H - 1))
        m[max(y0 - w, 0):y0 + w, :] = True
    return m


def downsample(mask, s):
    """Area-average to a coarse grid: what the encoder's bottleneck retains."""
    h = H // s
    return mask.reshape(h, s, h, s).mean(axis=(1, 3))


def upsample(coarse, s):
    """Nearest-neighbour expansion, thresholded -- the best a decoder can do
    from the bottleneck alone."""
    return np.repeat(np.repeat(coarse, s, axis=0), s, axis=1) >= 0.5


def boundary_band(mask, width=2):
    """Pixels within `width` of the mask edge, found by comparing the mask with
    shifted copies of itself."""
    edge = np.zeros_like(mask)
    for dy in range(-width, width + 1):
        for dx in range(-width, width + 1):
            edge |= (np.roll(np.roll(mask, dy, 0), dx, 1) != mask)
    return edge


def iou(a, b, where=None):
    if where is not None:
        a, b = a & where, b & where
    inter = float((a & b).sum())
    union = float((a | b).sum())
    return inter / union if union else 1.0


print(f"{H}x{H} masks. Every row is the BEST a decoder could do from the")
print("bottleneck alone. Nothing is trained, so these are upper bounds that no")
print("amount of training beats. WITH a skip connection the full-resolution "
      "evidence is available directly, so that column would read 1.000 in "
      "every row and is omitted -- the question is what the "
      "bottleneck-only path has already thrown away." + chr(10))
print(f"{'object':<16}{'stride':>8}{'overall IoU':>14}{'boundary IoU':>15}"
      f"{'band/object':>14}{'overall hides':>15}")
print("-" * 82)

for label, radius, thin in (("large blob", 34, False),
                            ("small blob", 12, False),
                            ("blob + thin bar", 30, True)):
    for s in (4, 8, 16):
        ov, bd, frac = [], [], []
        for _ in range(N_SHAPE):
            m = make_mask(radius, thin)
            rec = upsample(downsample(m, s), s)
            band = boundary_band(m)
            ov.append(iou(m, rec))
            bd.append(iou(m, rec, where=band))
            frac.append(float(band.sum()) / max(float(m.sum()), 1.0))
        o, b = float(np.mean(ov)), float(np.mean(bd))
        print(f"{label:<16}{s:>8}{o:>14.3f}{b:>15.3f}{np.mean(frac):>14.3f}"
              f"{o - b:>15.3f}")

print("""
Read the last column first, because it is the reason this measurement is split.
Overall IoU and boundary IoU are not close, and the gap is the amount of error
that a single reported mean IoU is concealing. For a large blob at stride 8, the
overall number stays high while the boundary number is far lower -- the
reconstruction is correct almost everywhere and wrong exactly where the answer is
decided.

The mechanism is a counting argument, not a modelling one (eq:boundary-dilution).
A large blob is mostly interior: the boundary band is a third of its pixels, so
getting every boundary pixel wrong still leaves overall IoU respectable. Mean IoU
averages over pixels, and pixels are dominated by the easy part of the problem.

Now read down each block. Increasing the stride makes both numbers worse, and it
makes the boundary number worse much faster, because the reconstruction can only
place an edge on a multiple of the stride. At stride 16 the decoder is choosing
between edges 16 pixels apart -- for a blob of radius 12 that is larger than the
object.

Compare the three blocks and the pattern sharpens. The band/object column is the
explanation: it is 0.334 for the large blob and 0.938 for the small one, so the
small blob is almost ENTIRELY boundary and there is no easy interior to dilute
its errors. Its overall IoU therefore falls to 0.464 at stride 16 where the large
blob still reads 0.773. And the thin bar is the worst case: a structure narrower
than the stride is erased by area-averaging, so no decoder recovers it, however
deep.

That is what the skip connection buys, and it explains why the U-shape is drawn
the way it is. The bottleneck answers WHAT -- it has the context, the receptive
field, the semantics. It cannot answer WHERE to a precision finer than its own
grid, and eq:resolution-tension says making it finer costs the context that made
it useful. The skip does not improve the bottleneck. It routes around it, carrying
the high-resolution evidence directly to the layer that has to draw the edge.

The practical consequence is a reporting rule. Report boundary IoU separately, or
a boundary F-score, because the aggregate is dominated by interior pixels that
were never in doubt -- and it will tell you a segmentation model is fine while it
is systematically unable to place an edge.""")

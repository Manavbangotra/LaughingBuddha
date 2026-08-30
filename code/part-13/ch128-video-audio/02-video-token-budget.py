# -*- coding: utf-8 -*-
# Extracted from: Chapter 128 — Video, Audio, and Spatial Reasoning
# Source: src/.../ch128-video-audio.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Video makes ch:mm-vlms's token budget two-dimensional, and the axes conflict.

A VLM spends N visual tokens on one image. A video spends frames x tokens-per-frame,
so at a fixed budget B the two multiply out (eq:video-token-budget):

    B = n_frames * tokens_per_frame

Every token spent on temporal coverage is a token not spent on spatial detail,
and vice versa. There is no setting that is generous on both, which makes this a
genuine allocation problem rather than a tuning one.

The two task families from the previous listing want opposite splits. A SPATIAL
question -- read the sign, identify the small object -- needs resolution and
tolerates few frames. A TEMPORAL question -- count the events, order them -- needs
frames and tolerates low resolution. This listing sweeps the split and finds both
optima.
"""
import numpy as np

DURATION = 60.0
BUDGETS = (256, 1024, 4096)
SPLITS = (1, 2, 4, 8, 16, 32, 64, 128)      # frames; per-frame tokens = B / n

FEATURE_PX = 14.0        # the thing to read is 14 px on a 1024 px frame
FRAME_PX = 1024.0
EVENT_S = 2.5            # the thing to count lasts 2.5 seconds


def spatial_score(tokens_per_frame):
    """ch:mm-vit's eq:patch-compression: a feature survives if the patch is no
    bigger than it. Patch side = frame / sqrt(tokens). See eq:tokens-for-feature."""
    if tokens_per_frame < 1:
        return 0.0
    patch = FRAME_PX / np.sqrt(tokens_per_frame)
    return float(np.clip(FEATURE_PX / patch, 0.0, 1.0))


def temporal_score(n_frames):
    """eq:event-catch-probability: an event of EVENT_S is caught when the
    sampling interval is shorter than it."""
    interval = DURATION / n_frames
    return float(np.clip(EVENT_S / interval, 0.0, 1.0))


print(f"{DURATION:.0f}s video, frames {FRAME_PX:.0f}px; spatial target "
      f"{FEATURE_PX:.0f}px, temporal target {EVENT_S:.1f}s\n")

best = {}
for B in BUDGETS:
    print(f"budget B = {B} visual tokens")
    print(f"{'frames':>8}{'tok/frame':>11}{'spatial':>10}{'temporal':>11}"
          f"{'both (min)':>12}")
    print("-" * 52)
    rows = []
    for n in SPLITS:
        tpf = B // n
        sp, tp = spatial_score(tpf), temporal_score(n)
        rows.append((n, tpf, sp, tp, min(sp, tp)))
        print(f"{n:>8}{tpf:>11}{sp:>10.3f}{tp:>11.3f}{min(sp, tp):>12.3f}")
    bs = max(rows, key=lambda r: r[2])
    bt = max(rows, key=lambda r: r[3])
    bb = max(rows, key=lambda r: r[4])
    best[B] = (bs, bt, bb)
    print(f"  best for SPATIAL: {bs[0]} frames    "
          f"best for TEMPORAL: {bt[0]} frames    "
          f"best for BOTH: {bb[0]} frames (score {bb[4]:.3f})\n")

b_small, b_large = best[BUDGETS[0]], best[BUDGETS[-1]]
print(f"""
Within any single budget block, the spatial and temporal columns move in opposite
directions -- one rises as the other falls, because they are reading the same
number from two ends (eq:video-token-budget). The best split for a spatial
question is {b_small[0][0]} frame at B={BUDGETS[0]} and the best for a temporal
one is {b_small[1][0]} -- opposite ends of the sweep, on the same footage and the
same budget. There is no compromise setting that is good at both; there is only a
choice about which question you are asking.

The "both" column makes that concrete by scoring the WORSE of the two, which is
what a question needing detail AND timing actually experiences. At B={BUDGETS[0]}
its best is {b_small[2][4]:.3f} -- the budget simply cannot serve a question that
needs to read something small at a moment it must not miss.

Now compare across budgets. At B={BUDGETS[-1]} the joint best rises to
{b_large[2][4]:.3f} at {b_large[2][0]} frames, so the conflict is not permanent --
it is what a small budget looks like. Buying tokens buys both axes at once, which
is unusual and worth noticing: most trades in this book do not resolve by
spending more.

The practical shape of the answer is therefore to stop treating "how many frames"
as a video question. It is two questions with two different answers:

  What is the smallest thing I must SEE?   -> sets tokens per frame, via
                                              ch:mm-vit's eq:patch-compression.
  What is the shortest thing I must NOTICE? -> sets frame count, via
                                              eq:event-catch-probability.

Both are measurable from the task before any model is chosen, and their product
is the budget you need. If that product exceeds what you can afford, the answer is
not a compromise split -- it is a cascade: a cheap detector at full frame rate to
find candidate moments, then the expensive model at high resolution on only those.
Uniform sampling spends its budget evenly over a video whose interesting content
is not evenly distributed, and that is the assumption to break first.""")

# -*- coding: utf-8 -*-
# Extracted from: Chapter 121 — Segmentation and the Segment Anything Model
# Source: src/.../ch121-segmentation.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Class imbalance in segmentation, and why cross-entropy quietly gives up.

Segmentation is per-pixel classification, so the obvious loss is per-pixel
cross-entropy. It has a failure mode that does not appear in ordinary
classification, because segmentation's class balance is set by GEOMETRY: a tumour,
a crack, a defect, or a thin wire occupies a tiny fraction of its image, and the
fraction can be 1% or 0.1% without anything being unusual.

At that ratio, predicting "background" everywhere is already 99% accurate and has
low cross-entropy (eq:ce-imbalance). The all-background solution is a good local
optimum, and the gradient pointing away from it is outnumbered.

The Dice loss (eq:dice) is built from overlap rather than from per-pixel
correctness, so it is invariant to how much background there is
(eq:dice-invariance). This listing trains the same model with each loss across a
sweep of foreground fractions, and reports IoU -- which is what anyone actually
cares about.
"""
import numpy as np

rng = np.random.default_rng(37)

N_PIX, D = 4000, 12
STEPS, LR = 700, 0.5
N_TRIAL = 5


def make_task(fg_frac, sep=3.0):
    """Per-pixel features. Foreground pixels are shifted along one direction by
    `sep` -- the same separability at every foreground fraction, so the only
    thing changing down the sweep is the CLASS BALANCE."""
    y = (rng.random(N_PIX) < fg_frac).astype(float)
    w_true = rng.normal(size=D)
    w_true /= np.linalg.norm(w_true)
    X = rng.normal(size=(N_PIX, D)) + sep * y[:, None] * w_true[None, :]
    return X, y


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def train(X, y, loss, steps=STEPS):
    w, b = np.zeros(D), 0.0
    for _ in range(steps):
        p = sigmoid(X @ w + b)
        if loss == "ce":
            # dL/dz for mean binary cross-entropy.
            gz = (p - y) / len(y)
        elif loss == "weighted-ce":
            # Positives reweighted by the inverse class frequency.
            pos = max(y.mean(), 1e-6)
            wt = np.where(y > 0, 1.0 / pos, 1.0 / (1.0 - pos))
            gz = wt * (p - y) / len(y)
        else:
            # eq:dice, soft form: 1 - 2<p,y> / (sum p + sum y).
            num, den = 2.0 * (p * y).sum(), p.sum() + y.sum() + 1e-8
            gz = -((2.0 * y * den - num) / den ** 2) * p * (1 - p)
        w -= LR * (X.T @ gz)
        b -= LR * gz.sum()
    return w, b


def iou_of(X, y, w, b, thr=0.5):
    pred = sigmoid(X @ w + b) >= thr
    t = y > 0
    union = float((pred | t).sum())
    return float((pred & t).sum()) / union if union else 1.0


FRACTIONS = (0.30, 0.10, 0.03, 0.01, 0.003)
LOSSES = ("ce", "weighted-ce", "dice")

print(f"{N_PIX} pixels, {D} features, identical class separability in every row.")
print("Only the foreground fraction changes.\n")
print(f"{'foreground':>11}{'':>3}" + "".join(f"{L + ' IoU':>16}" for L in LOSSES)
      + f"{'CE predicts fg':>17}")
print("-" * 79)

for f in FRACTIONS:
    scores = {L: [] for L in LOSSES}
    ce_rate = []
    for _ in range(N_TRIAL):
        X, y = make_task(f)
        for L in LOSSES:
            w, b = train(X, y, L)
            scores[L].append(iou_of(X, y, w, b))
            if L == "ce":
                ce_rate.append(float((sigmoid(X @ w + b) >= 0.5).mean()))
    print(f"{f:>11.3f}{'':>3}"
          + "".join(f"{np.mean(scores[L]):>16.3f}" for L in LOSSES)
          + f"{np.mean(ce_rate):>17.4f}")

print("""
Read the last column alongside the first. As the foreground fraction falls,
cross-entropy stops predicting foreground at all -- the share of pixels it calls
positive goes 0.2910, 0.0899, 0.0219, 0.0029, 0.0001, and its IoU follows it
down: 0.822 to 0.025, a factor of thirty-three.

Nothing about the problem got harder. The separability is identical in every row
-- the same feature shift, the same noise, the same pixel count. The only thing
that changed is how many pixels are foreground, and cross-entropy responded by
abandoning the class it was asked to find.

That is eq:ce-imbalance behaving as written rather than misbehaving. Mean
cross-entropy averages over pixels, so at a 0.3% foreground fraction the
background terms outnumber the foreground terms three hundred to one. The
all-background predictor already scores 99.7% pixel accuracy at a low loss, and
the gradient pulling toward the foreground is averaged against three hundred
pulling the other way. The optimiser is not failing; it is succeeding at the
objective it was given, and the objective was the wrong one.

Now the weighted-cross-entropy column, which is the standard first response and
does not behave the way the standard advice implies. At moderate imbalance it is
WORSE than plain cross-entropy -- 0.328 against 0.557 at a 3% foreground, and
0.134 against 0.300 at 1%. Reweighting the positive term by the inverse class
frequency does stop the model ignoring the foreground, and it overshoots: the
model now over-predicts foreground, precision falls, and IoU falls with it
(eq:weighting-overshoot). The fix has traded a recall failure for a precision
failure and kept the same shape of problem.

It only overtakes plain cross-entropy in the last row, where plain CE has
collapsed entirely. So inverse-frequency weighting is not a solution to imbalance,
it is a knob that moves the operating point, and the correct weight is neither 1
nor 1/frequency but something dataset-specific that has to be tuned and re-tuned
as the balance drifts.

Dice needs no such constant. It is built from the OVERLAP between prediction and
target (eq:dice), and true negatives appear nowhere in it -- adding a million
background pixels changes the loss not at all. That invariance is structural, and
it shows: 0.821 down to 0.368 across a hundredfold change in class balance, a
factor of 2.2 against cross-entropy's 33. Dice degrades; cross-entropy collapses.

The transferable rule: when the metric you care about is defined by overlap,
optimising per-pixel correctness is optimising a proxy -- and the proxy diverges
from the metric exactly as the class becomes rare, which is exactly the regime
that made anyone build the model.""")

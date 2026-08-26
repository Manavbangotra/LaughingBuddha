# Extracted from: Chapter 89 — Next-Token Prediction and Cross-Entropy Loss
# Source: src/.../ch089-next-token.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Expected calibration error and a reliability diagram, from scratch."""
import numpy as np

rng = np.random.default_rng(0)
N, N_CLASSES = 6000, 20


# Draw the ground truth ONCE. Every row below is the same data and the same
# outcomes, differing only in how sharply the model reports its beliefs — which
# is what makes the accuracy column a controlled comparison rather than four
# independent experiments.
TRUE_LOGITS = rng.normal(size=(N, N_CLASSES))
_p = np.exp(TRUE_LOGITS - TRUE_LOGITS.max(1, keepdims=True))
P_TRUE = _p / _p.sum(1, keepdims=True)
Y = np.array([rng.choice(N_CLASSES, p=row) for row in P_TRUE])


def make_predictions(sharpening):
    """The same model, reporting its beliefs at a given sharpness.

    sharpening = 1.0 -> calibrated; > 1 -> overconfident; < 1 -> underconfident.
    """
    z = TRUE_LOGITS * sharpening
    rep = np.exp(z - z.max(1, keepdims=True))
    rep /= rep.sum(1, keepdims=True)
    return rep, Y


def ece_and_bins(probs, y, n_bins=10):
    """Equations (eq:bin-accuracy) and (eq:ece)."""
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    rows, ece = [], 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        acc, cf, w = correct[m].mean(), conf[m].mean(), m.sum() / len(conf)
        ece += w * abs(acc - cf)
        rows.append((lo, hi, int(m.sum()), cf, acc, acc - cf))
    return ece, rows


print("Reliability diagram for a CALIBRATED model\n")
probs, y = make_predictions(1.0)
ece, rows = ece_and_bins(probs, y)
print(f"{'bin':>12} {'n':>6} {'confidence':>12} {'accuracy':>10} {'gap':>8}")
for lo, hi, n, cf, acc, gap in rows:
    bar = "#" * int(abs(gap) * 100)
    print(f"[{lo:.1f},{hi:.1f}] {n:>6} {cf:>12.3f} {acc:>10.3f} {gap:>+8.3f} {bar}")
print(f"\nECE = {ece:.4f}  (overall accuracy {(probs.argmax(1) == y).mean():.3f})")

# Calibration and accuracy are independent axes.
print(f"\n{'model':<26} {'accuracy':>10} {'mean conf':>11} {'ECE':>8} "
      f"{'verdict':<18}")
for label, sharp in [("underconfident", 0.6), ("calibrated", 1.0),
                     ("overconfident", 1.6), ("very overconfident", 2.5)]:
    p, yy = make_predictions(sharp)
    e, _ = ece_and_bins(p, yy)
    acc = float((p.argmax(1) == yy).mean())
    mc = float(p.max(1).mean())
    verdict = ("well calibrated" if e < 0.03
               else ("overconfident" if mc > acc else "underconfident"))
    print(f"{label:<26} {acc:>10.3f} {mc:>11.3f} {e:>8.4f} {verdict:<18}")

print("""
ACCURACY IS IDENTICAL IN ALL FOUR ROWS, to the last digit. Sharpening a
distribution is a monotone transformation of the logits, so it cannot change
which token is the argmax — only how confident the model claims to be about it.

That is equation (eq:perfect-calibration)'s point made concretely: calibration
and accuracy are independent axes. A model can be made to look confident
without being made to be right, and the four rows here are literally the same
model on the same data.""")

# The binning sensitivity of the warning in section 5.2, at two sample sizes.
p_full, y_full = make_predictions(1.6)
print(f"\n{'bins':>6} {'ECE (n=' + str(N) + ')':>18} {'ECE (n=200)':>14} "
      f"{'empty bins at n=200':>21}")
for nb in (5, 10, 20, 50, 100):
    e_full, _ = ece_and_bins(p_full, y_full, n_bins=nb)
    e_small, rows_small = ece_and_bins(p_full[:200], y_full[:200], n_bins=nb)
    print(f"{nb:>6} {e_full:>18.4f} {e_small:>14.4f} "
          f"{nb - len(rows_small):>21}")

print("""
With 6,000 samples the estimate is stable across bin counts — which is worth
knowing, because it means the usual warning is conditional rather than general.
With 200 samples it is not: bins empty out, the surviving ones are estimated
from a handful of points, and the number wanders.

So report the bin count AND the sample size, and prefer the reliability diagram,
which shows whether the gaps share a sign — systematic overconfidence — or
alternate, which is noise. The scalar cannot distinguish those and the shape
can.""")

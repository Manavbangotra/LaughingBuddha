# Extracted from: Chapter 42 — Anomaly Detection Methods
# Source: src/.../ch042-anomaly.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Evaluating a detector honestly under extreme imbalance.
"""
import numpy as np

rng = np.random.default_rng(4)


def pr_auc(y, s):
    o = np.argsort(-s, kind="mergesort")
    ys = y[o]
    prec = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    return float(np.sum(prec * ys) / max(1, int(y.sum())))


def roc_auc(y, s):
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s))
    r[o] = np.arange(1, len(s) + 1)
    npos = int(y.sum())
    return float((r[y == 1].sum() - npos * (npos + 1) / 2)
                 / (npos * (len(y) - npos)))


def precision_at_k(y, s, k):
    return float(y[np.argsort(-s)[:k]].mean())


def recall_at_k(y, s, k):
    return float(y[np.argsort(-s)[:k]].sum() / max(1, y.sum()))


# --- section 6.4: the same detector, four anomaly rates ---------------------
print("=" * 72)
print("ROC-AUC hides what PR-AUC shows (section 6.4)")
print("=" * 72)
print("The detector's QUALITY is held fixed throughout — anomalies always")
print("score from the same shifted distribution. Only the rate changes.\n")
print(f"{'anomaly rate':>13} {'ROC-AUC':>9} {'PR-AUC':>8} {'baseline':>10} "
      f"{'lift':>7} {'precision@100':>15} {'alerts per hit':>16}")
for rate in (0.20, 0.05, 0.01, 0.001):
    n = 200000
    y = (rng.random(n) < rate).astype(float)
    s = rng.normal(np.where(y == 1, 2.2, 0.0), 1.0)
    p100 = precision_at_k(y, s, 100)
    print(f"{rate:>13.3f} {roc_auc(y, s):>9.4f} {pr_auc(y, s):>8.4f} "
          f"{y.mean():>10.4f} {pr_auc(y, s) / y.mean():>7.1f}x "
          f"{p100:>15.4f} "
          f"{(1 / p100 if p100 > 0 else float('inf')):>15.1f}")

print("\nROC-AUC barely moves across a 200-fold change in the anomaly rate,")
print("because its false-positive rate divides by the number of NEGATIVES,")
print("which is enormous. PR-AUC collapses, because precision divides by the")
print("number of ALERTS — the quantity a human has to work through.")
print("\nThe last column is the operational translation: at a 0.1% rate the")
print("same detector produces several false alarms for every real hit, even")
print("at the very top of its ranking. Both statements are true — it is")
print("hundreds of times better than chance AND it may be unusable — and")
print("only one of them is visible in the ROC number.")

# --- the concrete example from section 6.4 ----------------------------------
print("\n" + "=" * 72)
print("the arithmetic, spelled out")
print("=" * 72)
P, N = 1000, 1000000
TP, FP = 900, 10000
print(f"  {P:,} anomalies, {N:,} normal points")
print(f"  a detector catches {TP} anomalies with {FP:,} false positives\n")
print(f"  recall (TPR)     = {TP}/{P}       = {TP / P:.4f}")
print(f"  false-pos. rate  = {FP:,}/{N:,} = {FP / N:.4f}   <- looks excellent")
print(f"  PRECISION        = {TP}/{TP + FP:,}   = {TP / (TP + FP):.4f}   "
      f"<- {FP / TP:.1f} false alarms per real one")
print("\nThe ROC curve reports the middle number. The operations team")
print("experiences the last one.")

# --- choosing the operating point from cost ---------------------------------
print("\n" + "=" * 72)
print("choosing k from what an analyst can actually review")
print("=" * 72)
n = 100000
rate = 0.004
y = (rng.random(n) < rate).astype(float)
s = rng.normal(np.where(y == 1, 2.6, 0.0), 1.0)
print(f"{int(y.sum())} anomalies in {n:,} records "
      f"({y.mean() * 100:.2f}%)\n")

COST_MISS = 3000.0          # an anomaly we failed to flag
COST_REVIEW = 25.0          # an analyst-hour spent on any alert, real or not
print(f"cost of a missed anomaly     : GBP {COST_MISS:,.0f}")
print(f"cost of reviewing one alert  : GBP {COST_REVIEW:,.0f}\n")
print(f"{'alerts/day (k)':>15} {'precision@k':>12} {'recall@k':>10} "
      f"{'missed':>8} {'total cost':>13}")
best = (None, np.inf)
for k in (50, 100, 200, 400, 800, 1600, 3200):
    p, r = precision_at_k(y, s, k), recall_at_k(y, s, k)
    missed = int(y.sum() - round(r * y.sum()))
    cost = k * COST_REVIEW + missed * COST_MISS
    if cost < best[1]:
        best = (k, cost)
    print(f"{k:>15} {p:>12.4f} {r:>10.4f} {missed:>8} {cost:>13,.0f}")
print(f"\ncheapest operating point: k = {best[0]} alerts, "
      f"GBP {best[1]:,.0f}")
print("\nThe threshold is a business decision, exactly as in Chapter 33 — and")
print("here it is expressed in the unit that actually constrains the system:")
print("how many alerts a human can work through. `contamination` is a guess")
print("at this number; the cost table is a derivation of it.")

# --- contamination does not do what people think ----------------------------
print("\n" + "=" * 72)
print("what `contamination` actually does (section 4.3)")
print("=" * 72)
clean = rng.normal(0, 1, (5000, 4))
scores_clean = np.abs(clean).max(1)
print("5,000 points drawn from ONE clean Gaussian. There are no anomalies.\n")
print(f"{'contamination':>15} {'points flagged':>16} {'true anomalies':>16}")
for c in (0.01, 0.05, 0.10):
    thr = np.quantile(scores_clean, 1 - c)
    print(f"{c:>15.2f} {int((scores_clean > thr).sum()):>16} {0:>16}")
print("\nIt is a quantile of the score, nothing more. It carries no")
print("information about what an anomaly looks like, and on clean data it")
print("flags exactly the fraction you asked for. Prefer the raw scores and a")
print("threshold you derived, as above.")

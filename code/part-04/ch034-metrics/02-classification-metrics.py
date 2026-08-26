# -*- coding: utf-8 -*-
# Extracted from: Chapter 34 — Evaluation Metrics and the Bias–Variance Tradeoff
# Source: src/.../ch034-metrics.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Classification metrics: where each one misleads, measured.
"""
import numpy as np

rng = np.random.default_rng(4)


def confusion(y, pred):
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    return tp, fp, fn, tn


def prf(y, pred):
    tp, fp, fn, _ = confusion(y, pred)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return prec, rec, f1


def roc_auc(y, score):
    """AUC as the probability a positive outranks a negative (eq. 34.9).

    Computed via ranks, which handles ties correctly and is O(n log n).
    """
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0     # average rank for ties
        i = j + 1
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def pr_auc(y, score):
    """Average precision: the step-wise area under the PR curve."""
    order = np.argsort(-score, kind="mergesort")
    y_s = y[order]
    tp = np.cumsum(y_s)
    prec = tp / np.arange(1, len(y_s) + 1)
    n_pos = max(1, int(y.sum()))
    return float(np.sum(prec * y_s) / n_pos)


# --- the AUC identity, checked by brute force -------------------------------
y_small = np.array([1, 0, 1, 1, 0, 0, 1, 0.])
s_small = np.array([0.9, 0.8, 0.7, 0.4, 0.4, 0.3, 0.2, 0.1])
pos, neg = s_small[y_small == 1], s_small[y_small == 0]
pairs = [(1.0 if p > n else 0.5 if p == n else 0.0) for p in pos for n in neg]
print(f"AUC by rank formula   : {roc_auc(y_small, s_small):.6f}")
print(f"AUC by pair counting  : {np.mean(pairs):.6f}")
print("Identical — AUC IS the probability that a positive outranks a")
print("negative, ties counting a half (eq. 34.9).\n")

# --- ROC-AUC's blind spot under imbalance -----------------------------------
print("=" * 72)
print("the same ranker, four different base rates")
print("=" * 72)
print(f"{'positive rate':>14} {'ROC-AUC':>9} {'PR-AUC':>8} "
      f"{'PR baseline':>12} {'lift over baseline':>19}")

for rate in (0.50, 0.10, 0.01, 0.001):
    n = 200000
    y = (rng.random(n) < rate).astype(float)
    # score quality held FIXED: same two Gaussians regardless of base rate
    score = rng.normal(np.where(y == 1, 1.4, 0.0), 1.0)
    print(f"{rate:>14.3f} {roc_auc(y, score):>9.4f} {pr_auc(y, score):>8.4f} "
          f"{y.mean():>12.4f} {pr_auc(y, score) / y.mean():>19.1f}x")

print("\nROC-AUC is essentially constant: it is a property of the ranker and")
print("is invariant to class balance. PR-AUC collapses, because precision")
print("depends on how many negatives are competing for the top of the list.")
print("At a 0.1% base rate the ranker is unchanged and the product built on")
print("it is unusable. Report PR-AUC against its baseline under imbalance.")

# --- accuracy under imbalance -----------------------------------------------
print("\n" + "=" * 72)
print("accuracy is uninformative under imbalance")
print("=" * 72)
n = 20000
y = (rng.random(n) < 0.01).astype(float)
score = rng.normal(np.where(y == 1, 1.6, 0.0), 1.0)
print(f"{'model':<26} {'accuracy':>9} {'precision':>10} {'recall':>8} "
      f"{'F1':>7} {'ROC-AUC':>9}")
always0 = np.zeros(n)
p, r, f = prf(y, always0)
print(f"{'always predict negative':<26} {(always0 == y).mean():>9.4f} "
      f"{p:>10.4f} {r:>8.4f} {f:>7.4f} {0.5:>9.4f}")
for t in (0.5, 1.0, 2.0):
    pred = (score >= t).astype(float)
    p, r, f = prf(y, pred)
    print(f"{'threshold ' + str(t):<26} {(pred == y).mean():>9.4f} "
          f"{p:>10.4f} {r:>8.4f} {f:>7.4f} {roc_auc(y, score):>9.4f}")
print("\nThe useless model wins on accuracy and scores zero on everything")
print("that measures whether it found anything.")

# --- calibration and discrimination are independent -------------------------
print("\n" + "=" * 72)
print("calibration and discrimination are independent (section 5.3)")
print("=" * 72)


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def ece(y, p, n_bins=10):
    """Expected calibration error (eq. 34.7), equal-count bins."""
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    total = 0.0
    for i in range(n_bins):
        m = (p >= edges[i]) & (p <= edges[i + 1])
        if m.sum():
            total += m.sum() / len(p) * abs(y[m].mean() - p[m].mean())
    return total


n = 40000
z = rng.normal(0, 1.5, n)
p_true = 1 / (1 + np.exp(-z))
y = (rng.random(n) < p_true).astype(float)

variants = {
    "perfectly calibrated": p_true,
    "halved (same ranking)": p_true * 0.5,
    "over-confident": np.clip(1 / (1 + np.exp(-2.5 * z)), 1e-6, 1 - 1e-6),
    "always the base rate": np.full(n, y.mean()),
}
print(f"{'model':<24} {'ROC-AUC':>9} {'Brier':>9} {'ECE':>8} {'log loss':>10}")
for name, p in variants.items():
    ll = -np.mean(y * np.log(np.clip(p, 1e-12, 1))
                  + (1 - y) * np.log(np.clip(1 - p, 1e-12, 1)))
    print(f"{name:<24} {roc_auc(y, p):>9.4f} {brier(y, p):>9.4f} "
          f"{ece(y, p):>8.4f} {ll:>10.4f}")

print("\nThe first three have IDENTICAL ROC-AUC — halving or sharpening every")
print("probability preserves the order — while Brier, ECE and log loss")
print("separate them completely. The last has perfect calibration and zero")
print("resolution (eq. 34.10): honest and useless. No single number covers")
print("both failure modes, which is why you report at least two.")

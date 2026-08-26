# -*- coding: utf-8 -*-
# Extracted from: Chapter 28 — Data Leakage, Imbalanced Data, and Dataset Pathologies
# Source: src/.../ch028-leakage.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Class imbalance: why accuracy lies, and which remedy actually helps.
"""
import numpy as np

rng = np.random.default_rng(3)


def auc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos, neg = labels.sum(), (1 - labels).sum()
    return (ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)


def fit_logistic(X, y, weights=None, steps=600, lr=0.5):
    X = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(X.shape[1])
    sw = np.ones(len(y)) if weights is None else weights
    sw = sw / sw.mean()
    for _ in range(steps):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        w -= lr * (X.T @ (sw * (p - y)) / len(y))
    return w


def predict(X, w):
    X = np.column_stack([np.ones(len(X)), X])
    return 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))


def metrics(scores, y, thresh):
    pred = (scores >= thresh).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"accuracy": (tp + tn) / len(y), "precision": prec,
            "recall": rec, "f1": f1}


# --- eq. 28.4: precision collapses with prevalence, model unchanged ---------
print("=" * 72)
print("the same model at different prevalences (eq. 28.4)")
print("=" * 72)
s, f = 0.90, 0.05
print(f"sensitivity {s}, false-positive rate {f} — FIXED\n")
print(f"{'prevalence':>12} {'precision':>11} {'accuracy':>10}")
for pi in (0.5, 0.1, 0.01, 0.001):
    prec = s * pi / (s * pi + f * (1 - pi))
    acc = s * pi + (1 - f) * (1 - pi)
    print(f"{pi:>12.1%} {prec:>11.1%} {acc:>10.1%}")
print("\nAccuracy RISES as the problem gets harder, because the majority")
print("baseline improves. Precision collapses. Only one of these is useful.")

# --- a realistically imbalanced problem --------------------------------------
n = 30_000
x1 = rng.normal(0, 1, n)
x2 = rng.normal(0, 1, n)
logit = -5.2 + 1.4 * x1 + 0.9 * x2
y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
X = np.column_stack([x1, x2])

split = int(0.7 * n)
Xtr, ytr, Xte, yte = X[:split], y[:split], X[split:], y[split:]
print(f"\npositive rate: {y.mean():.2%}  "
      f"({y.sum():,} positives in {n:,} rows)")

print("\n" + "=" * 72)
print("remedies compared")
print("=" * 72)
results = {}

# baseline
w = fit_logistic(Xtr, ytr)
results["none (0.5 threshold)"] = (predict(Xte, w), 0.5)

# threshold tuned on TRAIN to maximise F1 — no data change, cannot leak
tr_scores = predict(Xtr, w)
grid = np.quantile(tr_scores, np.linspace(0.5, 0.9999, 300))
best_t = max(grid, key=lambda t: metrics(tr_scores, ytr, t)["f1"])
results["threshold tuned"] = (predict(Xte, w), best_t)

# class weighting
weights = np.where(ytr == 1, (ytr == 0).sum() / max((ytr == 1).sum(), 1), 1.0)
w_cw = fit_logistic(Xtr, ytr, weights=weights)
results["class weighting"] = (predict(Xte, w_cw), 0.5)

# oversampling INSIDE the training fold — correct
pos_idx = np.where(ytr == 1)[0]
extra = rng.choice(pos_idx, size=(ytr == 0).sum() - len(pos_idx), replace=True)
idx_bal = np.concatenate([np.arange(len(ytr)), extra])
w_os = fit_logistic(Xtr[idx_bal], ytr[idx_bal])
results["oversample (in-fold)"] = (predict(Xte, w_os), 0.5)

print(f"{'remedy':<24} {'accuracy':>10} {'precision':>11} {'recall':>9} "
      f"{'F1':>8} {'AUC':>8}")
for name, (scores, t) in results.items():
    m = metrics(scores, yte, t)
    print(f"{name:<24} {m['accuracy']:>10.4f} {m['precision']:>11.3f} "
          f"{m['recall']:>9.3f} {m['f1']:>8.3f} {auc(scores, yte):>8.4f}")

print(f"\n{'always predict 0':<24} {1-yte.mean():>10.4f} "
      f"{0.0:>11.3f} {0.0:>9.3f} {0.0:>8.3f} {0.5:>8.4f}")
print("\nThe do-nothing baseline has the highest accuracy and is useless.")
print("Note the AUC barely moves across remedies — they change the operating")
print("point, not the ranking. Threshold tuning gets most of the benefit")
print("without touching the data.")

# --- eq. 28.3: oversampling BEFORE the split leaks --------------------------
print("\n" + "=" * 72)
print("oversampling before the split (eq. 28.3)")
print("=" * 72)


def knn_scores(Xtr, ytr, Xte, k=5):
    """A memorising model. With a linear model this leak is invisible, because
    duplicated rows only reweight the loss — they cannot be looked up."""
    out = np.empty(len(Xte))
    for start in range(0, len(Xte), 2000):
        block = Xte[start:start + 2000]
        d = ((block[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)
        nn = np.argpartition(d, k, axis=1)[:, :k]
        out[start:start + 2000] = ytr[nn].mean(axis=1)
    return out


# Keep k modest so the demonstration is quick and the arithmetic is clear.
sub = rng.choice(n, 6000, replace=False)
Xs, ys = X[sub], y[sub]
pos_all = np.where(ys == 1)[0]
k_factor = 6
extra_all = rng.choice(pos_all, size=len(pos_all) * (k_factor - 1), replace=True)
idx_all = np.concatenate([np.arange(len(ys)), extra_all])
rng.shuffle(idx_all)

cut = int(0.7 * len(idx_all))
tr_idx, te_idx = idx_all[:cut], idx_all[cut:]
overlap = len(set(tr_idx.tolist()) & set(te_idx.tolist())) / len(set(te_idx.tolist()))

leaky = knn_scores(Xs[tr_idx], ys[tr_idx], Xs[te_idx])
honest = knn_scores(Xs[tr_idx], ys[tr_idx], Xte)

print(f"oversampling factor for the minority class : {k_factor}")
print(f"eq. 28.3 predicts {1 - 0.7**k_factor - 0.3**k_factor:.1%} of duplicated "
      f"rows appear on both sides")
print(f"measured overlap of distinct rows          : {overlap:.1%}")
print(f"\n{'evaluation':<34} {'AUC':>8}")
print(f"{'on the OVERSAMPLED test split':<34} {auc(leaky, ys[te_idx]):>8.4f}"
      f"   <- inflated")
print(f"{'on the untouched real test set':<34} {auc(honest, yte):>8.4f}"
      f"   <- the truth")
print(f"\noptimism: {auc(leaky, ys[te_idx]) - auc(honest, yte):+.4f}")
print("\nThe same model, evaluated two ways. Resampling before the split")
print("evaluates the model on rows it has literally already seen.")
print("\nNote this leak is invisible to a linear model, which cannot memorise")
print("a row — it appears with k-NN, trees and boosting, which can.")

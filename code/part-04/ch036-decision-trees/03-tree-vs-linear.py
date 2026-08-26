# -*- coding: utf-8 -*-
# Extracted from: Chapter 36 — Decision Trees
# Source: src/.../ch036-decision-trees.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Where trees win, where they lose, and the failure that matters in
production: extrapolation.
"""
import numpy as np

rng = np.random.default_rng(17)


def fit_tree_reg(X, y, max_depth=6, min_leaf=5):
    """Regression CART: impurity is variance (eq. 36.2)."""
    def grow(idx, depth):
        yy = y[idx]
        node = {"value": float(yy.mean()), "n": len(idx)}
        if depth >= max_depth or len(idx) < 2 * min_leaf or yy.std() < 1e-12:
            return node
        best = (0.0, None, None)
        parent_sse = float(((yy - yy.mean()) ** 2).sum())
        for j in range(X.shape[1]):
            o = idx[np.argsort(X[idx, j], kind="mergesort")]
            xs, ys = X[o, j], y[o]
            cs, cs2 = np.cumsum(ys), np.cumsum(ys ** 2)
            tot, tot2 = cs[-1], cs2[-1]
            n = len(o)
            for i in range(min_leaf, n - min_leaf + 1):
                if xs[i] == xs[i - 1]:
                    continue
                sse_l = cs2[i - 1] - cs[i - 1] ** 2 / i
                sse_r = (tot2 - cs2[i - 1]) - (tot - cs[i - 1]) ** 2 / (n - i)
                gain = parent_sse - sse_l - sse_r
                if gain > best[0]:
                    best = (gain, j, 0.5 * (xs[i] + xs[i - 1]))
        gain, j, thr = best
        if j is None:
            return node
        m = X[idx, j] <= thr
        node["f"], node["t"] = j, thr
        node["l"] = grow(idx[m], depth + 1)
        node["r"] = grow(idx[~m], depth + 1)
        return node

    return grow(np.arange(len(y)), 0)


def predict_tree_reg(node, X):
    out = np.empty(len(X))
    for i, x in enumerate(X):
        nd = node
        while "f" in nd:
            nd = nd["l"] if x[nd["f"]] <= nd["t"] else nd["r"]
        out[i] = nd["value"]
    return out


def fit_linear(X, y):
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return beta


def predict_linear(beta, X):
    return np.column_stack([np.ones(len(X)), X]) @ beta


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def count_leaves(node):
    return 1 if "f" not in node else (count_leaves(node["l"])
                                      + count_leaves(node["r"]))


# --- three target shapes ----------------------------------------------------
n = 1200
X = rng.uniform(-3, 3, (n, 2))
targets = {
    "linear (a plane)":      2.0 * X[:, 0] - 1.0 * X[:, 1],
    "diagonal step":         np.where(X[:, 0] + X[:, 1] > 0, 4.0, -1.0),
    "axis-aligned boxes":    np.where((np.abs(X[:, 0]) < 1) &
                                      (np.abs(X[:, 1]) < 1), 5.0, 0.0),
    "interaction (x1 * x2)": X[:, 0] * X[:, 1],
}
Xte = rng.uniform(-3, 3, (4000, 2))

print("=" * 72)
print("test RMSE: the inductive bias decides")
print("=" * 72)
print(f"{'target':<24} {'linear model':>14} {'depth-6 tree':>14} {'winner':>10}")
for name, ytr in targets.items():
    yv = ytr + rng.normal(0, 0.4, n)
    if "linear" in name:
        yte_true = 2.0 * Xte[:, 0] - 1.0 * Xte[:, 1]
    elif "diagonal" in name:
        yte_true = np.where(Xte[:, 0] + Xte[:, 1] > 0, 4.0, -1.0)
    elif "boxes" in name:
        yte_true = np.where((np.abs(Xte[:, 0]) < 1) &
                            (np.abs(Xte[:, 1]) < 1), 5.0, 0.0)
    else:
        yte_true = Xte[:, 0] * Xte[:, 1]
    lin = rmse(predict_linear(fit_linear(X, yv), Xte), yte_true)
    tre = rmse(predict_tree_reg(fit_tree_reg(X, yv), Xte), yte_true)
    print(f"{name:<24} {lin:>14.4f} {tre:>14.4f} "
          f"{('linear' if lin < tre else 'tree'):>10}")

print("\nThe linear model owns the plane and is helpless on the boxes and the")
print("interaction; the tree is the reverse. That is inductive bias, not")
print("quality.")

print("\nThe diagonal step deserves a closer look, because the tree wins")
print("there and still pays for its bias. The true boundary x1 + x2 = 0 is a")
print("SINGLE straight cut, and no axis-aligned split can be it — so the")
print("tree builds a staircase. Rotating the feature space by 45 degrees")
print("puts that boundary on an axis and hands the tree the same target in a")
print("form it can express exactly:\n")

ystep = np.where(X[:, 0] + X[:, 1] > 0, 4.0, -1.0) + rng.normal(0, 0.4, n)
yte_step = np.where(Xte[:, 0] + Xte[:, 1] > 0, 4.0, -1.0)
Xrot = np.column_stack([X[:, 0] + X[:, 1], X[:, 0] - X[:, 1]])
Xte_rot = np.column_stack([Xte[:, 0] + Xte[:, 1], Xte[:, 0] - Xte[:, 1]])

print(f"{'representation':<28} {'depth':>6} {'leaves':>8} {'test RMSE':>11}")
for d in (2, 4, 6, 10):
    t_raw = fit_tree_reg(X, ystep, max_depth=d)
    t_rot = fit_tree_reg(Xrot, ystep, max_depth=d)
    print(f"{'axis-aligned features':<28} {d:>6} "
          f"{count_leaves(t_raw):>8} "
          f"{rmse(predict_tree_reg(t_raw, Xte), yte_step):>11.4f}")
    print(f"{'  same data, rotated 45 deg':<28} {d:>6} "
          f"{count_leaves(t_rot):>8} "
          f"{rmse(predict_tree_reg(t_rot, Xte_rot), yte_step):>11.4f}")

print("\nThe rotated tree reaches RMSE 0.05 at depth 2 with four leaves. The")
print("axis-aligned tree never gets below 0.88, even at depth 10 with 137")
print("leaves — a seventeen-fold worse error using thirty-four times the")
print("model. The information is identical; only the coordinate system")
print("changed.")
print("This is the precise sense in which trees are NOT rotation-invariant,")
print("and it is one of the three mechanisms Grinsztajn et al. identify —")
print("except that on real tabular data the axes are meaningful (age,")
print("income, count) rather than arbitrary, so axis-alignment is usually")
print("an advantage rather than the handicap it is here.")

# --- the failure that matters in production ---------------------------------
print("\n" + "=" * 72)
print("extrapolation: a tree cannot leave its training range")
print("=" * 72)
xs = rng.uniform(0, 10, 900).reshape(-1, 1)
ys = 3.0 * xs[:, 0] + 5.0 + rng.normal(0, 1.0, 900)
tree = fit_tree_reg(xs, ys, max_depth=8)
beta = fit_linear(xs, ys)

print(f"training range of x: [{xs.min():.2f}, {xs.max():.2f}]")
print(f"\n{'x':>7} {'truth':>9} {'linear':>9} {'tree':>9} {'tree error':>12}")
for xq in (2.0, 5.0, 9.5, 12.0, 20.0, 50.0):
    q = np.array([[xq]])
    truth = 3.0 * xq + 5.0
    lp = float(predict_linear(beta, q)[0])
    tp = float(predict_tree_reg(tree, q)[0])
    flag = "  <-- outside training range" if xq > xs.max() else ""
    print(f"{xq:>7.1f} {truth:>9.2f} {lp:>9.2f} {tp:>9.2f} "
          f"{tp - truth:>12.2f}{flag}")

print("\nInside the range the tree is fine. Outside it the tree returns the")
print("same constant forever, because there is no leaf beyond the last")
print("split — so its error grows without bound while the linear model's")
print("stays near zero. Any tabular model on a drifting or growing feature")
print("(prices, counts, timestamps) has this failure mode, and it is silent:")
print("nothing in the model's output signals that the input was")
print("out-of-range. Monitoring for that is Chapter 179's job.")

# --- and the instability that motivates Chapter 37 --------------------------
print("\n" + "=" * 72)
print("instability: resampling the rows rebuilds the tree")
print("=" * 72)
Xs, ys2 = X[:300], (targets["interaction (x1 * x2)"][:300]
                    + rng.normal(0, 0.4, 300))
base = fit_tree_reg(Xs, ys2, max_depth=4)
base_pred = predict_tree_reg(base, Xte)
print(f"reference tree, all 300 points: root split is "
      f"x{base['f']} <= {base['t']:.3f}\n")

print("A. drop 5% of the training data at random:")
print(f"{'':>6} {'root split':>18} {'corr with reference':>21}")
for trial in range(5):
    keep = rng.permutation(300)[:285]
    t2 = fit_tree_reg(Xs[keep], ys2[keep], max_depth=4)
    p2 = predict_tree_reg(t2, Xte)
    split = f"x{t2['f']} <= {t2['t']:.3f}"
    print(f"{trial + 1:>6} {split:>18} "
          f"{np.corrcoef(base_pred, p2)[0, 1]:>21.4f}")

print("\nB. bootstrap resamples — the perturbation bagging actually uses:")
print(f"{'':>6} {'root split':>18} {'corr with reference':>21}")
boot_preds = []
for trial in range(5):
    b = rng.integers(0, 300, 300)
    t2 = fit_tree_reg(Xs[b], ys2[b], max_depth=4)
    p2 = predict_tree_reg(t2, Xte)
    boot_preds.append(p2)
    split = f"x{t2['f']} <= {t2['t']:.3f}"
    print(f"{trial + 1:>6} {split:>18} "
          f"{np.corrcoef(base_pred, p2)[0, 1]:>21.4f}")

# many more resamples, to measure the variance and what averaging does to it
many = []
for _ in range(40):
    b = rng.integers(0, 300, 300)
    many.append(predict_tree_reg(fit_tree_reg(Xs[b], ys2[b], max_depth=4), Xte))
many = np.array(many)

y_true_te = Xte[:, 0] * Xte[:, 1]
per_tree = np.mean([rmse(p, y_true_te) for p in many])
averaged = rmse(many.mean(axis=0), y_true_te)
print(f"\npointwise sd across 40 resampled trees : "
      f"{many.std(axis=0).mean():.4f}")
print(f"mean RMSE of a SINGLE resampled tree   : {per_tree:.4f}")
print(f"RMSE of the AVERAGE of all 40          : {averaged:.4f}")
print(f"improvement from averaging alone       : "
      f"{per_tree - averaged:+.4f}")

print("\nDropping 5% of the rows already moves the prediction correlation")
print("as low as 0.47. Bootstrap resampling — drawing 300 rows WITH")
print("replacement from the same 300 — changes even the root FEATURE, from")
print("x0 to x1, and drives the correlation with the reference tree down to")
print("0.16. No new information entered; only which rows were drawn.")
print("\nAcross 40 resamples the trees disagree by 1.38 at a typical test")
print("point, on a target whose own spread is about 3. And the average of")
print("the 40 beats the typical individual tree by 0.49 RMSE.")
print("\nThat last number is the whole idea of Chapter 37. Averaging")
print("high-variance, low-bias models leaves the bias alone and divides the")
print("variance down — and a deep tree is about as high-variance and")
print("low-bias a model as exists, which is why it is the base learner of")
print("choice for bagging.")

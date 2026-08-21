# Extracted from: Chapter 38 — Gradient Boosting: Theory, XGBoost, LightGBM, CatBoost
# Source: src/.../ch038-gradient-boosting.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Gradient boosting from scratch: squared error, then log loss, then the
second-order version — each derived rather than described.
"""
import numpy as np

rng = np.random.default_rng(0)


# --- a depth-limited regression tree, vectorised split search ---------------
def _best_split(X, y, w, j, min_leaf):
    """Weighted SSE reduction on feature j, via cumulative sums."""
    n = len(y)
    o = np.argsort(X[:, j], kind="mergesort")
    xs, ys, ws = X[o, j], y[o], w[o]
    swy, swy2, sw = np.cumsum(ws * ys), np.cumsum(ws * ys ** 2), np.cumsum(ws)
    TY, TY2, TW = swy[-1], swy2[-1], sw[-1]
    k = np.arange(1, n)
    ok = (xs[1:] != xs[:-1]) & (k >= min_leaf) & (n - k >= min_leaf)
    if not ok.any():
        return (-np.inf, None, None)
    wl, wr = sw[:-1], TW - sw[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sse_l = swy2[:-1] - swy[:-1] ** 2 / np.where(wl > 0, wl, 1)
        sse_r = (TY2 - swy2[:-1]) - (TY - swy[:-1]) ** 2 / np.where(wr > 0, wr, 1)
    parent = TY2 - TY ** 2 / TW
    gain = np.where(ok & (wl > 0) & (wr > 0), parent - sse_l - sse_r, -np.inf)
    i = int(gain.argmax())
    if not np.isfinite(gain[i]) or gain[i] <= 0:
        return (-np.inf, None, None)
    return (float(gain[i]), j, 0.5 * (xs[i] + xs[i + 1]))


def grow(X, y, w, depth, max_depth, min_leaf=1):
    """Fit a tree to targets y with per-sample weights w."""
    tw = w.sum()
    node = {"v": float((w * y).sum() / tw) if tw > 0 else 0.0}
    if depth >= max_depth or len(y) < 2 * min_leaf or np.ptp(y) < 1e-12:
        return node
    best = (-np.inf, None, None)
    for j in range(X.shape[1]):
        c = _best_split(X, y, w, j, min_leaf)
        if c[1] is not None and c[0] > best[0]:
            best = c
    _, j, thr = best
    if j is None:
        return node
    m = X[:, j] <= thr
    node["f"], node["t"] = j, thr
    node["l"] = grow(X[m], y[m], w[m], depth + 1, max_depth, min_leaf)
    node["r"] = grow(X[~m], y[~m], w[~m], depth + 1, max_depth, min_leaf)
    return node


def apply_tree(node, X):
    out = np.empty(len(X))

    def walk(nd, idx):
        if "f" not in nd:
            out[idx] = nd["v"]
            return
        m = X[idx, nd["f"]] <= nd["t"]
        walk(nd["l"], idx[m])
        walk(nd["r"], idx[~m])

    walk(node, np.arange(len(X)))
    return out


def leaf_index(node, X):
    """Which leaf each row falls into — needed to re-optimise leaf values."""
    out = np.empty(len(X), dtype=int)
    counter = [0]

    def walk(nd, idx):
        if "f" not in nd:
            out[idx] = counter[0]
            counter[0] += 1
            return
        m = X[idx, nd["f"]] <= nd["t"]
        walk(nd["l"], idx[m])
        walk(nd["r"], idx[~m])

    walk(node, np.arange(len(X)))
    return out, counter[0]


# --- section 5.1: gradient boosting for squared error -----------------------
class GBRegressor:
    def __init__(self, n_trees=300, eta=0.1, max_depth=3, subsample=1.0,
                 seed=0):
        self.M, self.eta, self.depth = n_trees, eta, max_depth
        self.subsample, self.seed = subsample, seed

    def fit(self, X, y, X_val=None, y_val=None):
        rs = np.random.default_rng(self.seed)
        self.F0 = float(y.mean())              # best constant for squared error
        F = np.full(len(y), self.F0)
        self.trees, self.train_curve, self.val_curve = [], [], []
        for m in range(self.M):
            r = y - F                          # pseudo-residual (table 38.1)
            if self.subsample < 1.0:
                k = max(2, int(self.subsample * len(y)))
                idx = rs.choice(len(y), k, replace=False)
            else:
                idx = np.arange(len(y))
            t = grow(X[idx], r[idx], np.ones(len(idx)), 0, self.depth)
            self.trees.append(t)
            F += self.eta * apply_tree(t, X)
            self.train_curve.append(float(np.mean((y - F) ** 2)))
            if X_val is not None:
                self.val_curve.append(float(np.mean(
                    (y_val - self.predict(X_val)) ** 2)))
        return self

    def predict(self, X, n_trees=None):
        M = len(self.trees) if n_trees is None else n_trees
        F = np.full(len(X), self.F0)
        for t in self.trees[:M]:
            F += self.eta * apply_tree(t, X)
        return F


def make_reg(n, rs):
    X = rs.uniform(-3, 3, (n, 6))
    f = (2.0 * np.sin(1.3 * X[:, 0]) + 0.9 * X[:, 1]
         - 0.7 * X[:, 0] * X[:, 2] + 1.1 * np.abs(X[:, 3]))
    return X, f, f + rs.normal(0, 1.0, n)


rs = np.random.default_rng(1)
Xtr, _, ytr = make_reg(600, rs)
Xva, _, yva = make_reg(600, rs)
Xte, f_te, yte = make_reg(2500, rs)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


print("=" * 72)
print("boosting reduces BIAS, so the base learner must be WEAK")
print("=" * 72)
print("(Chapter 37 needed the opposite: deep trees, because bagging reduces")
print(" only variance and cannot touch bias.)\n")
print(f"{'tree depth':>11} {'trees':>7} {'train RMSE':>12} {'test RMSE':>11}")
for d in (1, 2, 3, 6, 10):
    g = GBRegressor(n_trees=150, eta=0.1, max_depth=d).fit(Xtr, ytr)
    print(f"{d:>11} {150:>7} {rmse(g.predict(Xtr), ytr):>12.4f} "
          f"{rmse(g.predict(Xte), f_te):>11.4f}")
print("\nDepth 10 drives training error towards zero and test error up: the")
print("first few trees already fit the data, so the residuals are noise and")
print("every later tree fits noise. Compare a random forest, where deeper")
print("is essentially free.")

# --- section 4.3: the learning rate / number of trees trade -----------------
print("\n" + "=" * 72)
print("shrinkage: many small steps beat few large ones")
print("=" * 72)
print(f"{'eta':>7} {'best round':>12} {'best val RMSE':>15} "
      f"{'test RMSE there':>17}")
for eta in (1.0, 0.5, 0.2, 0.1):
    g = GBRegressor(n_trees=300, eta=eta, max_depth=3).fit(Xtr, ytr, Xva, yva)
    best_m = int(np.argmin(g.val_curve)) + 1
    print(f"{eta:>7} {best_m:>12} {np.sqrt(min(g.val_curve)):>15.4f} "
          f"{rmse(g.predict(Xte, best_m), f_te):>17.4f}")
print("\nHalving eta roughly doubles the round at which the optimum occurs —")
print("the two trade off almost exactly — and smaller eta reaches a better")
print("optimum, because each tree's own overfit is diluted before it enters")
print("the sum.")

# --- section 6.3: boosting CAN overfit, and a forest cannot -----------------
print("\n" + "=" * 72)
print("boosting overfits with more trees; Chapter 37's forest does not")
print("=" * 72)
# A high learning rate, deep trees and a small noisy sample: the conditions
# under which boosting overfits fastest.
Xn, _, yn_ = make_reg(250, np.random.default_rng(11))
Xnv, _, ynv = make_reg(1500, np.random.default_rng(12))
g = GBRegressor(n_trees=400, eta=0.4, max_depth=8).fit(Xn, yn_, Xnv, ynv)
print(f"{'trees':>7} {'train RMSE':>12} {'val RMSE':>10} {'test RMSE':>11}")
for m in (1, 3, 5, 10, 20, 50, 100, 200, 400):
    print(f"{m:>7} {np.sqrt(g.train_curve[m - 1]):>12.4f} "
          f"{np.sqrt(g.val_curve[m - 1]):>10.4f} "
          f"{rmse(g.predict(Xte, m), f_te):>11.4f}")
best_m = int(np.argmin(g.val_curve)) + 1
print(f"\nvalidation minimum at round {best_m} "
      f"(RMSE {np.sqrt(min(g.val_curve)):.4f})")
print(f"at 400 rounds                : RMSE "
      f"{np.sqrt(g.val_curve[-1]):.4f}")
print(f"degradation from not stopping: "
      f"{np.sqrt(g.val_curve[-1]) - np.sqrt(min(g.val_curve)):+.4f}")
print("\nTraining error falls forever; validation error turns and rises.")
print("That U-shape is why n_estimators is set by EARLY STOPPING and never")
print("fixed in advance — the exact opposite of the advice in Chapter 37.")

# --- section 4.2: depth controls INTERACTION ORDER --------------------------
print("\n" + "=" * 72)
print("tree depth controls interaction order, not just capacity")
print("=" * 72)
Xa = rs.uniform(-3, 3, (700, 4))
add = 2.0 * np.sin(1.5 * Xa[:, 0]) + 1.5 * Xa[:, 1] - 1.0 * np.abs(Xa[:, 2])
inter = 3.0 * Xa[:, 0] * Xa[:, 1]
Xa_te = rs.uniform(-3, 3, (3000, 4))
add_te = (2.0 * np.sin(1.5 * Xa_te[:, 0]) + 1.5 * Xa_te[:, 1]
          - 1.0 * np.abs(Xa_te[:, 2]))
inter_te = 3.0 * Xa_te[:, 0] * Xa_te[:, 1]

print("500 rounds at eta=0.1 for every depth, so what differs is what each")
print("CAN represent, not how far it got.\n")
print(f"{'target':<26} {'depth 1 (stumps)':>18} {'depth 2':>10} "
      f"{'depth 4':>10}")
for name, ytr_a, yte_a in (("purely additive", add, add_te),
                           ("pure interaction x0*x1", inter, inter_te)):
    ynz = ytr_a + rs.normal(0, 0.5, len(ytr_a))
    row = []
    for d in (1, 2, 4):
        gg = GBRegressor(n_trees=500, eta=0.1, max_depth=d).fit(Xa, ynz)
        row.append(rmse(gg.predict(Xa_te), yte_a))
    print(f"{name:<26} {row[0]:>18.4f} {row[1]:>10.4f} {row[2]:>10.4f}")

print("\nOn the additive target the depths are close, as they should be: a")
print("sum of single-feature functions is exactly what stumps produce, so")
print("extra depth buys capacity the target does not need.")
print("\nOn the pure interaction the stump model is hopeless — no number of")
print("single-feature terms can represent x0*x1 — and every increase in")
print("depth helps substantially. Note that depth 2 is far from sufficient")
print("either: two-way splits can EXPRESS an interaction, but approximating")
print("a smooth product surface with axis-aligned rectangles still needs")
print("many of them (Chapter 36's staircase problem, in three dimensions).")
print("\nThe usable rule: max_depth is a statement about how many features")
print("may interact, and it is a lower bound rather than a setting.")

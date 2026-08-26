# -*- coding: utf-8 -*-
# Extracted from: Chapter 37 — Random Forests and Bagging
# Source: src/.../ch037-random-forests.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A forest end to end: classification, OOB tuning, importance, calibration,
and the limits worth knowing before you ship one.
"""
import numpy as np

rng = np.random.default_rng(31)


def grow(X, y, depth=0, max_depth=12, min_leaf=1, m_features=None, rs=None):
    rs = rs if rs is not None else rng
    node = {"v": float(y.mean()), "n": len(y)}
    if depth >= max_depth or len(y) < 2 * min_leaf or y.std() < 1e-12:
        return node
    D = X.shape[1]
    feats = (np.arange(D) if m_features is None
             else rs.choice(D, min(m_features, D), replace=False))
    n, best = len(y), (1e-12, None, None)
    p = y.mean()
    parent = n * p * (1 - p)                 # Gini impurity times n
    for j in feats:
        o = np.argsort(X[:, j], kind="mergesort")
        xs, ys = X[o, j], y[o]
        cs = np.cumsum(ys)
        tot = cs[-1]
        for i in range(min_leaf, n - min_leaf + 1):
            if xs[i] == xs[i - 1]:
                continue
            pl, pr = cs[i - 1] / i, (tot - cs[i - 1]) / (n - i)
            child = i * pl * (1 - pl) + (n - i) * pr * (1 - pr)
            if parent - child > best[0]:
                best = (parent - child, int(j), 0.5 * (xs[i] + xs[i - 1]))
    gain, j, thr = best
    if j is None:
        return node
    m_ = X[:, j] <= thr
    node["f"], node["t"] = j, thr
    node["l"] = grow(X[m_], y[m_], depth + 1, max_depth, min_leaf, m_features, rs)
    node["r"] = grow(X[~m_], y[~m_], depth + 1, max_depth, min_leaf, m_features, rs)
    return node


def tree_proba(node, X):
    out = np.empty(len(X))
    for i, x in enumerate(X):
        nd = node
        while "f" in nd:
            nd = nd["l"] if x[nd["f"]] <= nd["t"] else nd["r"]
        out[i] = nd["v"]
    return out


class RF:
    def __init__(self, B=200, m=3, max_depth=12, min_leaf=1, seed=0):
        self.B, self.m, self.d, self.leaf, self.seed = B, m, max_depth, min_leaf, seed

    def fit(self, X, y):
        rs = np.random.default_rng(self.seed)
        self.trees, self.bags = [], []
        for _ in range(self.B):
            idx = rs.integers(0, len(y), len(y))
            self.trees.append(grow(X[idx], y[idx], max_depth=self.d,
                                   min_leaf=self.leaf, m_features=self.m, rs=rs))
            self.bags.append(np.bincount(idx, minlength=len(y)) == 0)
        self.Xt, self.yt = X, y
        return self

    def predict_proba(self, X):
        return np.mean([tree_proba(t, X) for t in self.trees], axis=0)

    def oob_proba(self):
        tot = np.zeros(len(self.yt))
        cnt = np.zeros(len(self.yt))
        for t, oob in zip(self.trees, self.bags):
            if oob.any():
                tot[oob] += tree_proba(t, self.Xt[oob])
                cnt[oob] += 1
        ok = cnt > 0
        return tot[ok] / cnt[ok], ok


# --- a credit-risk-shaped problem -------------------------------------------
def make_data(n):
    income = rng.lognormal(10.4, 0.55, n)
    util = np.clip(rng.beta(2, 5, n), 0, 1)
    late = rng.poisson(0.4, n)
    tenure = rng.uniform(0, 25, n)
    # a genuine interaction: high utilisation only matters at low income
    z = (-2.4 - 0.8 * (np.log(income) - 10.4) + 1.4 * util
         + 2.2 * util * (np.log(income) < 10.2) + 0.5 * late - 0.03 * tenure)
    noise = rng.normal(size=(n, 6))            # six pure-noise columns
    X = np.column_stack([np.log(income), util, late, tenure, noise])
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(float)
    return X, y


NAMES = ["log_income", "utilisation", "late_payments", "tenure"] + \
        [f"noise_{i}" for i in range(6)]
Xtr, ytr = make_data(1500)
Xte, yte = make_data(6000)
print(f"positive rate: {ytr.mean():.4f}")


def roc_auc(y, s):
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s))
    r[o] = np.arange(1, len(s) + 1)
    npos = int(y.sum())
    return float((r[y == 1].sum() - npos * (npos + 1) / 2)
                 / (npos * (len(y) - npos)))


# --- 1. tune on OOB ---------------------------------------------------------
print("\n" + "=" * 72)
print("1. tuning on out-of-bag AUC (no validation split needed)")
print("=" * 72)
print(f"{'max_features':>13} {'min_leaf':>10} {'OOB AUC':>9} {'test AUC':>10}")
best = (None, -1)
for m in (2, 3, 5, 10):
    for leaf in (1, 5):
        f = RF(B=120, m=m, min_leaf=leaf, seed=5).fit(Xtr, ytr)
        o, ok = f.oob_proba()
        oa = roc_auc(ytr[ok], o)
        ta = roc_auc(yte, f.predict_proba(Xte))
        if oa > best[1]:
            best = ((m, leaf), oa)
        print(f"{m:>13} {leaf:>10} {oa:>9.4f} {ta:>10.4f}")
(m_star, leaf_star), _ = best
print(f"\nOOB chooses max_features={m_star}, min_samples_leaf={leaf_star}")

rf = RF(B=300, m=m_star, min_leaf=leaf_star, seed=9).fit(Xtr, ytr)
p_te = rf.predict_proba(Xte)
print(f"final forest test AUC: {roc_auc(yte, p_te):.4f}")

# --- 2. against the alternatives from earlier chapters ----------------------
print("\n" + "=" * 72)
print("2. against the models of Chapters 33 and 36")
print("=" * 72)


def fit_logistic(A, b, lam=1e-3, n_iter=60):
    mu, sd = A.mean(0), A.std(0)
    A1 = np.column_stack([np.ones(len(A)), (A - mu) / sd])
    w = np.zeros(A1.shape[1])
    for _ in range(n_iter):
        p = 1 / (1 + np.exp(-np.clip(A1 @ w, -30, 30)))
        g = A1.T @ (p - b) / len(b)
        g[1:] += 2 * lam * w[1:]
        S = np.maximum(p * (1 - p), 1e-7)
        H = (A1 * S[:, None]).T @ A1 / len(b) + (2 * lam + 1e-6) * np.eye(len(w))
        w -= np.linalg.solve(H, g)
    return w, mu, sd


w, mu, sd = fit_logistic(Xtr, ytr)
p_log = 1 / (1 + np.exp(-np.clip(
    np.column_stack([np.ones(len(Xte)), (Xte - mu) / sd]) @ w, -30, 30)))
one_tree = grow(Xtr, ytr, max_depth=12)
p_tree = tree_proba(one_tree, Xte)

print(f"{'model':<32} {'test AUC':>10} {'test accuracy':>15}")
for name, p in (("logistic regression", p_log),
                ("single depth-12 tree", p_tree),
                (f"random forest ({rf.B} trees)", p_te)):
    print(f"{name:<32} {roc_auc(yte, p):>10.4f} "
          f"{((p >= 0.5) == (yte == 1)).mean():>15.4f}")
print("\nThe forest wins because the target contains a genuine interaction")
print("(high utilisation matters only at low income) that the linear model")
print("cannot express without being told about it, and the single tree can")
print("express but estimates far too noisily.")

# --- 3. permutation importance on held-out data -----------------------------
print("\n" + "=" * 72)
print("3. permutation importance (Chapter 36: never use the built-in MDI)")
print("=" * 72)
base = roc_auc(yte, p_te)
imps = []
for j in range(Xte.shape[1]):
    drops = []
    for _ in range(5):
        Xp = Xte.copy()
        rng.shuffle(Xp[:, j])
        drops.append(base - roc_auc(yte, rf.predict_proba(Xp)))
    imps.append((NAMES[j], float(np.mean(drops)), float(np.std(drops))))
for name, mean, sd_ in sorted(imps, key=lambda t: -t[1]):
    bar = "#" * max(0, int(mean * 200))
    print(f"  {name:<15} {mean:>+8.4f} +-{sd_:.4f}  {bar}")
print("\nAll six noise columns land at approximately zero, and the four real")
print("features are ranked sensibly. This is measured on held-out data with")
print("the metric we actually care about.")

# --- 4. the limits ----------------------------------------------------------
print("\n" + "=" * 72)
print("4. two limits to know before shipping")
print("=" * 72)

# calibration
print("calibration of the averaged vote:")
print(f"{'predicted band':>18} {'n':>6} {'mean p':>9} {'observed':>10}")
edges = np.quantile(p_te, np.linspace(0, 1, 7))
ece = 0.0
for i in range(6):
    msk = (p_te >= edges[i]) & (p_te <= edges[i + 1])
    ece += msk.sum() / len(p_te) * abs(yte[msk].mean() - p_te[msk].mean())
    print(f"  [{edges[i]:.3f}, {edges[i+1]:.3f}] {msk.sum():>6} "
          f"{p_te[msk].mean():>9.4f} {yte[msk].mean():>10.4f}")
print(f"ECE = {ece:.4f}")
print("Averaging many trees' leaf frequencies gives probabilities that are")
print("usually decent but pulled towards the middle — a forest rarely says")
print("0.99, because that needs every tree to agree. Check, do not assume.")

# extrapolation
print("\nextrapolation (unchanged from Chapter 36):")
lo, hi = Xtr[:, 0].min(), Xtr[:, 0].max()
print(f"  training range of log_income: [{lo:.2f}, {hi:.2f}]")
probe = np.tile(np.median(Xtr, axis=0), (5, 1))
for k, v in enumerate((lo, (lo + hi) / 2, hi, hi + 2, hi + 6)):
    probe[k, 0] = v
out = rf.predict_proba(probe)
print(f"\n  {'log_income':>12} {'P(default)':>12}")
for v, p in zip((lo, (lo + hi) / 2, hi, hi + 2, hi + 6), out):
    tag = "  <-- beyond training range" if v > hi else ""
    print(f"  {v:>12.2f} {p:>12.4f}{tag}")
print("\n  The prediction is flat beyond the training range: an average of")
print("  constants is a constant. A forest is no better at extrapolation")
print("  than the trees it averages, and it will not tell you it is")
print("  guessing.")

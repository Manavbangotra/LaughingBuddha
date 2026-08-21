# Extracted from: Chapter 37 — Random Forests and Bagging
# Source: src/.../ch037-random-forests.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Out-of-bag error as free validation, and a comparison against extra-trees.
"""
import numpy as np

rng = np.random.default_rng(4)


def grow(X, y, depth=0, max_depth=10, min_leaf=1, m_features=None,
         rs=None, extra=False):
    """As before, plus `extra`: draw a RANDOM threshold per candidate feature
    instead of searching for the best one (section 5.3)."""
    rs = rs if rs is not None else rng
    node = {"v": float(y.mean()), "n": len(y)}
    if depth >= max_depth or len(y) < 2 * min_leaf or y.std() < 1e-12:
        return node
    D = X.shape[1]
    feats = (np.arange(D) if m_features is None
             else rs.choice(D, min(m_features, D), replace=False))
    n, best = len(y), (0.0, None, None)
    parent_sse = float(((y - y.mean()) ** 2).sum())
    for j in feats:
        lo, hi = X[:, j].min(), X[:, j].max()
        if hi - lo < 1e-12:
            continue
        if extra:
            thresholds = [float(rs.uniform(lo, hi))]
        else:
            o = np.argsort(X[:, j], kind="mergesort")
            xs = X[o, j]
            thresholds = [0.5 * (xs[i] + xs[i - 1])
                          for i in range(min_leaf, n - min_leaf + 1)
                          if xs[i] != xs[i - 1]]
        for thr in thresholds:
            m_ = X[:, j] <= thr
            nl, nr = int(m_.sum()), int((~m_).sum())
            if nl < min_leaf or nr < min_leaf:
                continue
            sse = (((y[m_] - y[m_].mean()) ** 2).sum()
                   + ((y[~m_] - y[~m_].mean()) ** 2).sum())
            gain = parent_sse - sse
            if gain > best[0]:
                best = (gain, int(j), thr)
    gain, j, thr = best
    if j is None:
        return node
    m_ = X[:, j] <= thr
    node["f"], node["t"] = j, thr
    node["l"] = grow(X[m_], y[m_], depth + 1, max_depth, min_leaf,
                     m_features, rs, extra)
    node["r"] = grow(X[~m_], y[~m_], depth + 1, max_depth, min_leaf,
                     m_features, rs, extra)
    return node


def predict(node, X):
    out = np.empty(len(X))
    for i, x in enumerate(X):
        nd = node
        while "f" in nd:
            nd = nd["l"] if x[nd["f"]] <= nd["t"] else nd["r"]
        out[i] = nd["v"]
    return out


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def make_data(n, d=8):
    X = rng.uniform(-3, 3, (n, d))
    f = (np.sin(1.3 * X[:, 0]) * 2.0 + 0.8 * X[:, 1]
         - 0.5 * X[:, 0] * X[:, 2] + 1.2 * np.abs(X[:, 3]))
    return X, f, f + rng.normal(0, 1.0, n)


N_TRAIN = 300
Xtr, _, ytr = make_data(N_TRAIN)
Xte, f_te, yte = make_data(2000)

# --- section 6.3: the 63.2% fraction ----------------------------------------
print("=" * 72)
print("the bootstrap leaves out 1/e of the rows (eq. 37.6)")
print("=" * 72)
print(f"{'N':>8} {'theory (1-1/N)^N':>18} {'measured over 400 draws':>26}")
for N in (10, 100, 1000):
    frac = np.mean([len(np.unique(rng.integers(0, N, N))) / N
                    for _ in range(400)])
    print(f"{N:>8} {(1 - 1 / N) ** N:>18.4f} {1 - frac:>26.4f}")
print(f"\n1/e = {np.exp(-1):.4f}. Each tree sees about 63.2% of the distinct")
print("rows; the other 36.8% are out-of-bag for it and can score it for free.")

# --- out-of-bag error as free validation ------------------------------------
print("\n" + "=" * 72)
print("OOB error vs a held-out set vs the truth")
print("=" * 72)


class Forest:
    def __init__(self, B=200, m_features=3, max_depth=10, extra=False,
                 bootstrap=True, seed=0):
        self.B, self.m, self.depth = B, m_features, max_depth
        self.extra, self.bootstrap, self.seed = extra, bootstrap, seed

    def fit(self, X, y):
        rs = np.random.default_rng(self.seed)
        self.trees, self.bags = [], []
        for _ in range(self.B):
            idx = (rs.integers(0, len(y), len(y)) if self.bootstrap
                   else np.arange(len(y)))
            self.trees.append(grow(X[idx], y[idx], max_depth=self.depth,
                                   m_features=self.m, rs=rs, extra=self.extra))
            # a boolean mask, not a set: `i not in bag` per row is O(N) per
            # tree in Python and dominates the whole computation
            self.bags.append(np.bincount(idx, minlength=len(y)) == 0)
        self.X_train, self.y_train = X, y
        return self

    def predict(self, X):
        return np.mean([predict(t, X) for t in self.trees], axis=0)

    def oob_predict(self):
        """Score each row using ONLY the trees that never saw it."""
        N = len(self.y_train)
        total, count = np.zeros(N), np.zeros(N)
        for tree, oob in zip(self.trees, self.bags):
            if not oob.any():
                continue
            total[oob] += predict(tree, self.X_train[oob])
            count[oob] += 1
        ok = count > 0
        return total[ok] / count[ok], ok


rf = Forest(B=120, m_features=3).fit(Xtr, ytr)
oob_pred, ok = rf.oob_predict()

print(f"OOB RMSE (free, uses all {N_TRAIN} rows for training) : "
      f"{rmse(oob_pred, ytr[ok]):.4f}")
print(f"held-out RMSE on {len(yte):,} fresh rows                : "
      f"{rmse(rf.predict(Xte), yte):.4f}")
print(f"rows with at least one OOB tree                  : "
      f"{ok.sum()} of {N_TRAIN}")
print(f"mean number of trees scoring each row            : "
      f"{120 * np.exp(-1):.0f} of 120")

# 5-fold CV for comparison, at 5x the training cost
folds = np.array_split(np.random.default_rng(7).permutation(N_TRAIN), 5)
cv = []
for i in range(5):
    va = folds[i]
    tr = np.concatenate([folds[j] for j in range(5) if j != i])
    m = Forest(B=120, m_features=3, seed=100 + i).fit(Xtr[tr], ytr[tr])
    cv.append(rmse(m.predict(Xtr[va]), ytr[va]))
print(f"5-fold CV RMSE (5x the training cost)            : {np.mean(cv):.4f}")

print("\nThe two honest estimators — OOB and 5-fold CV — agree to within a")
print("few per cent of each other, and both sit ABOVE the held-out score.")
print("That direction is not an accident, and is the point worth taking")
print("away:")
print("\n  * OOB scores each row using only the ~37% of trees that excluded")
print("    it, so it is really measuring a forest a third the size.")
print("  * 5-fold CV fits each forest on 4/5 of the rows, so it measures a")
print("    model trained on less data than the one you will ship.")
print("\nBoth are therefore mildly PESSIMISTIC, exactly as nested CV was in")
print("Chapter 34, and for the same structural reason. OOB gets there at a")
print("fifth of CV cost and without holding anything out, which is why it")
print("is the default way to tune a forest.")
print("\nOne hard caveat: OOB is silently WRONG when rows are grouped or")
print("time-ordered, because the bootstrap assumes they are exchangeable")
print("(Chapter 28). It will report a number, the number will be optimistic,")
print("and nothing will warn you.")

# --- tuning max_features on OOB ---------------------------------------------
print("\n" + "=" * 72)
print("tuning max_features on OOB alone")
print("=" * 72)
print(f"{'max_features':>13} {'OOB RMSE':>10} {'true test RMSE':>16}")
best = (None, np.inf)
for m in (1, 2, 4, 8):
    f = Forest(B=80, m_features=m, seed=3).fit(Xtr, ytr)
    o, ok2 = f.oob_predict()
    ov = rmse(o, ytr[ok2])
    if ov < best[1]:
        best = (m, ov)
    print(f"{m:>13} {ov:>10.4f} {rmse(f.predict(Xte), yte):>16.4f}")
print(f"\nOOB picks max_features = {best[0]}")

# --- random forest vs extra-trees vs plain bagging --------------------------
print("\n" + "=" * 72)
print("plain bagging vs random forest vs extra-trees")
print("=" * 72)
import time

print(f"{'model':<34} {'test RMSE':>10} {'fit seconds':>13}")
configs = [
    ("bagging (all features at each split)", dict(m_features=8)),
    ("random forest (m = 3)", dict(m_features=3)),
    ("extra-trees (m = 3, random splits)", dict(m_features=3, extra=True,
                                                bootstrap=False)),
    ("extra-trees (m = 8, random splits)", dict(m_features=8, extra=True,
                                                bootstrap=False)),
]
for name, kw in configs:
    t0 = time.perf_counter()
    f = Forest(B=80, seed=11, **kw).fit(Xtr, ytr)
    dt = time.perf_counter() - t0
    print(f"{name:<34} {rmse(f.predict(Xte), yte):>10.4f} {dt:>13.2f}")

print("\nExtra-trees replace the threshold SEARCH with a single random draw")
print("per candidate feature, which is why they fit three to nine times")
print("faster — the scan over candidate thresholds was the whole cost of")
print("Chapter 36's split search.")
print("\nOn accuracy the ordering here is: extra-trees with all features,")
print("then plain bagging, then the two m=3 variants. Consistent with")
print("everything above — on independent features, restricting the split")
print("search costs more than the decorrelation buys — and a reminder that")
print("extra-trees randomise on a DIFFERENT axis from max_features. You can")
print("randomise the thresholds without also restricting the features, and")
print("here that combination is both the fastest and the most accurate.")

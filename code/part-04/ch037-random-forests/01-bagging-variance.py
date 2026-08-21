# Extracted from: Chapter 37 — Random Forests and Bagging
# Source: src/.../ch037-random-forests.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Bagging from scratch, and the variance algebra of eq. 37.3 measured.
"""
import numpy as np

rng = np.random.default_rng(0)


# --- a deliberately literal tree, so the ensemble is the only new idea ------
def grow(X, y, depth=0, max_depth=8, min_leaf=2, m_features=None, rs=None):
    """Regression CART. If m_features is set, only that many randomly chosen
    features are considered AT EACH NODE — the random-forest modification."""
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
        o = np.argsort(X[:, j], kind="mergesort")
        xs, ys = X[o, j], y[o]
        cs, cs2 = np.cumsum(ys), np.cumsum(ys ** 2)
        tot, tot2 = cs[-1], cs2[-1]
        for i in range(min_leaf, n - min_leaf + 1):
            if xs[i] == xs[i - 1]:
                continue
            sse_l = cs2[i - 1] - cs[i - 1] ** 2 / i
            sse_r = (tot2 - cs2[i - 1]) - (tot - cs[i - 1]) ** 2 / (n - i)
            gain = parent_sse - sse_l - sse_r
            if gain > best[0]:
                best = (gain, int(j), 0.5 * (xs[i] + xs[i - 1]))
    gain, j, thr = best
    if j is None:
        return node
    msk = X[:, j] <= thr
    node["f"], node["t"] = j, thr
    node["l"] = grow(X[msk], y[msk], depth + 1, max_depth, min_leaf,
                     m_features, rs)
    node["r"] = grow(X[~msk], y[~msk], depth + 1, max_depth, min_leaf,
                     m_features, rs)
    return node


def predict(node, X):
    out = np.empty(len(X))
    for i, x in enumerate(X):
        nd = node
        while "f" in nd:
            nd = nd["l"] if x[nd["f"]] <= nd["t"] else nd["r"]
        out[i] = nd["v"]
    return out


# --- two datasets that differ ONLY in how the features are related ---------
def make_independent(n, rs):
    """Eight independent features; four carry signal, four are noise."""
    X = rs.uniform(-3, 3, (n, 8))
    f = (np.sin(1.3 * X[:, 0]) * 2.0 + 0.8 * X[:, 1]
         - 0.5 * X[:, 0] * X[:, 2] + 1.2 * np.abs(X[:, 3]))
    return X, f, f + rs.normal(0, 1.0, n)


def make_correlated(n, rs):
    """Five latent drivers, each observed through four noisy copies. Every
    informative feature therefore has near-substitutes — which is what real
    tabular data usually looks like."""
    Z = rs.uniform(-3, 3, (n, 5))
    X = np.column_stack([Z[:, k] + rs.normal(0, 0.35, n)
                         for k in range(5) for _ in range(4)])
    f = (2.0 * np.sin(1.2 * Z[:, 0]) + 1.5 * Z[:, 1]
         - 1.0 * Z[:, 2] * Z[:, 3] + 1.2 * np.abs(Z[:, 4]))
    return X, f, f + rs.normal(0, 1.0, n)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def decompose(gen, ms, n=400, B=60, depth=8, label=""):
    """Fit B bootstrapped trees at each max_features and report the two terms
    of eq. 37.3 separately."""
    rs = np.random.default_rng(0)
    Xtr, _, ytr = gen(n, rs)
    Xte, f_te, _ = gen(3000, rs)
    print(f"\n{label}")
    print(f"{'max_features':>22} {'sigma^2':>9} {'rho':>8} {'floor':>9} "
          f"{'single tree':>13} {'ensemble':>10}")
    rows = []
    for m in ms:
        rs2 = np.random.default_rng(1)
        P = []
        for _ in range(B):
            i = rs2.integers(0, n, n)
            P.append(predict(grow(Xtr[i], ytr[i], max_depth=depth,
                                  m_features=m, rs=rs2), Xte))
        P = np.array(P)
        C = np.corrcoef(P)
        rho = float((C.sum() - len(C)) / (len(C) * (len(C) - 1)))
        s2 = float(P.var(axis=0, ddof=1).mean())
        single = float(np.mean([rmse(p, f_te) for p in P]))
        ens = rmse(P.mean(axis=0), f_te)
        rows.append((m, ens))
        tag = f"{m} (all: plain bagging)" if m == ms[0] else str(m)
        print(f"{tag:>22} {s2:>9.4f} {rho:>8.4f} {rho * s2:>9.4f} "
              f"{single:>13.4f} {ens:>10.4f}")
    best = min(rows, key=lambda r: r[1])
    print(f"  --> best ensemble at max_features = {best[0]} "
          f"(RMSE {best[1]:.4f}); plain bagging gives {rows[0][1]:.4f}")
    return best


print("=" * 72)
print("eq. 37.3:  Var[average] = rho*sigma^2 + (1-rho)/B * sigma^2")
print("=" * 72)

decompose(make_independent, [8, 4, 3, 2, 1],
          label="A. eight INDEPENDENT features (four signal, four noise)")
decompose(make_correlated, [20, 10, 6, 4, 2, 1],
          label="B. twenty features in five CORRELATED groups of four")

print("\nThe mechanism is identical in both tables and exactly what eq. 37.3")
print("describes: as max_features falls, the correlation rho falls — that is")
print("the term no number of trees can remove — while per-tree variance")
print("sigma^2 rises, because each tree is now built from a restricted")
print("split search.")
print("\nWhat differs between A and B is the PRICE of that decorrelation,")
print("and the single-tree column is what tells you.")
print("\nWith independent features (A) there is no substitute for an")
print("excluded feature, so single-tree error climbs steeply — 2.16 to 2.72")
print("— and plain bagging wins outright. With correlated groups (B),")
print("excluding one copy of a latent driver leaves three others, so")
print("single-tree error is nearly FLAT across the whole range and")
print("decorrelation is close to free; subsampling then improves the")
print("ensemble.")
print("\nThis is why max_features is the one hyperparameter worth tuning,")
print("and why sqrt(D) and D/3 are starting points rather than answers: the")
print("right value depends on how much redundancy the features carry, which")
print("a default cannot know. Real tabular data usually looks more like B")
print("than A — measurements repeated, derived, lagged and re-expressed —")
print("which is why the defaults subsample at all.")

# --- more trees cannot overfit ----------------------------------------------
print("\n" + "=" * 72)
print("adding trees cannot overfit (section 5.4)")
print("=" * 72)
rs = np.random.default_rng(0)
Xtr, _, ytr = make_correlated(400, rs)
Xte, f_te, _ = make_correlated(3000, rs)
rs2 = np.random.default_rng(2)
P_all = []
for _ in range(400):
    i = rs2.integers(0, 400, 400)
    P_all.append(predict(grow(Xtr[i], ytr[i], max_depth=8, m_features=4,
                              rs=rs2), Xte))
P_all = np.array(P_all)

print(f"{'trees':>7} {'test RMSE':>11}")
for B in (1, 2, 5, 10, 25, 50, 100, 200, 400):
    print(f"{B:>7} {rmse(P_all[:B].mean(axis=0), f_te):>11.4f}")
print("\nTest error falls steeply, flattens, and thereafter moves only in")
print("the fourth decimal place — never upward in any sustained way.")
print("The limit is the expectation over the bootstrap and")
print("feature-subset randomness (section 5.4) — a fixed function of the")
print("training data — and more trees only reduce the Monte Carlo error of")
print("estimating it. B is the only hyperparameter in this book you can set")
print("by budget alone.")

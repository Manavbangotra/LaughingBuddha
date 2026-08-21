# Extracted from: Chapter 36 — Decision Trees
# Source: src/.../ch036-decision-trees.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Cost-complexity pruning, and why the default feature importance lies.

The tree is a nested dict here rather than a class, because pruning means
producing modified copies of a tree and dicts make that a one-liner.
"""
import copy

import numpy as np

rng = np.random.default_rng(5)


def gini(y):
    if len(y) == 0:
        return 0.0
    p = np.bincount(y, minlength=2) / len(y)
    return float(1.0 - np.sum(p ** 2))


def grow(X, y, depth=0, max_depth=20, min_leaf=1):
    """A node is {'value','n','imp'} plus, if internal, 'f','t','l','r'."""
    node = {"value": float(y.mean()), "n": len(y), "imp": gini(y)}
    if depth >= max_depth or node["imp"] == 0.0 or len(y) < 2 * min_leaf:
        return node
    n, best = len(y), (1e-12, None, None)
    for j in range(X.shape[1]):
        o = np.argsort(X[:, j], kind="mergesort")
        xs, ys = X[o, j], y[o]
        for i in range(min_leaf, n - min_leaf + 1):
            if xs[i] == xs[i - 1]:
                continue
            dec = node["imp"] - (i / n) * gini(ys[:i]) \
                - ((n - i) / n) * gini(ys[i:])
            if dec > best[0]:
                best = (dec, j, 0.5 * (xs[i] + xs[i - 1]))
    dec, j, thr = best
    if j is None:
        return node
    m = X[:, j] <= thr
    node["f"], node["t"] = j, thr
    node["l"] = grow(X[m], y[m], depth + 1, max_depth, min_leaf)
    node["r"] = grow(X[~m], y[~m], depth + 1, max_depth, min_leaf)
    return node


def predict(node, X):
    out = np.empty(len(X))
    for i, x in enumerate(X):
        nd = node
        while "f" in nd:
            nd = nd["l"] if x[nd["f"]] <= nd["t"] else nd["r"]
        out[i] = nd["value"]
    return (out >= 0.5).astype(int)


def n_leaves(node):
    return 1 if "f" not in node else n_leaves(node["l"]) + n_leaves(node["r"])


def collapse(node):
    """Turn an internal node into a leaf, in place."""
    for k in ("f", "t", "l", "r"):
        node.pop(k, None)


# --- cost-complexity pruning, eqs. 36.7 and 36.8 ----------------------------
def errors_here(y):
    """R(t): training misclassifications if this node were a leaf."""
    return 0 if len(y) == 0 else int(min((y == 0).sum(), (y == 1).sum()))


def subtree_errors(node, X, y):
    """R(T_t) and |leaves(T_t)| for the subtree rooted at node."""
    if "f" not in node:
        return errors_here(y), 1
    m = X[:, node["f"]] <= node["t"]
    el, nl = subtree_errors(node["l"], X[m], y[m])
    er, nr = subtree_errors(node["r"], X[~m], y[~m])
    return el + er, nl + nr


def weakest_link(node, X, y):
    """Find the internal node minimising g(t) = (R(t)-R(T_t))/(|T_t|-1)."""
    best = (np.inf, None)
    if "f" not in node:
        return best
    R_t = errors_here(y)
    R_T, nl = subtree_errors(node, X, y)
    if nl > 1:
        best = ((R_t - R_T) / (nl - 1), node)
    m = X[:, node["f"]] <= node["t"]
    for child, Xc, yc in ((node["l"], X[m], y[m]),
                          (node["r"], X[~m], y[~m])):
        g, nd = weakest_link(child, Xc, yc)
        if nd is not None and g < best[0]:
            best = (g, nd)
    return best


def prune_path(root, X, y):
    """The nested sequence of subtrees, from full to a single leaf."""
    tree = copy.deepcopy(root)
    path = [copy.deepcopy(tree)]
    while "f" in tree:
        g, node = weakest_link(tree, X, y)
        if node is None:
            break
        collapse(node)                 # node is a reference INTO tree
        path.append(copy.deepcopy(tree))
    return path


def make_data(n, noise=0.12):
    X = rng.uniform(-3, 3, (n, 2))
    z = np.sin(1.2 * X[:, 0]) + 0.6 * X[:, 1] - 0.25 * X[:, 0] * X[:, 1]
    y = (rng.random(n) < 1 / (1 + np.exp(-2.0 * z))).astype(int)
    flip = rng.random(n) < noise       # label noise, so a full tree must hurt
    y[flip] = 1 - y[flip]
    return X, y


Xtr, ytr = make_data(400)
Xva, yva = make_data(1500)
Xte, yte = make_data(4000)

full = grow(Xtr, ytr, max_depth=20)
print("=" * 72)
print("cost-complexity pruning (eqs. 36.7, 36.8)")
print("=" * 72)
print(f"fully grown: {n_leaves(full)} leaves, "
      f"train {(predict(full, Xtr) == ytr).mean():.4f}, "
      f"test {(predict(full, Xte) == yte).mean():.4f}")

path = prune_path(full, Xtr, ytr)
rows = [(n_leaves(t),
         (predict(t, Xtr) == ytr).mean(),
         (predict(t, Xva) == yva).mean(),
         (predict(t, Xte) == yte).mean()) for t in path]

print(f"\nthe nested sequence has {len(rows)} subtrees, "
      f"from {rows[0][0]} leaves down to {rows[-1][0]}\n")
print(f"{'leaves':>8} {'train acc':>11} {'val acc':>9} {'test acc':>10}")
step = max(1, len(rows) // 14)
shown = rows[::step]
if rows[-1] not in shown:
    shown = shown + [rows[-1]]
for r in shown:
    print(f"{r[0]:>8} {r[1]:>11.4f} {r[2]:>9.4f} {r[3]:>10.4f}")

best_va = max(rows, key=lambda r: r[2])
print(f"\nchosen by validation : {best_va[0]:>4} leaves, "
      f"test accuracy {best_va[3]:.4f}")
print(f"fully grown          : {rows[0][0]:>4} leaves, "
      f"test accuracy {rows[0][3]:.4f}")
print(f"single leaf (no model): {rows[-1][0]:>3} leaf,   "
      f"test accuracy {rows[-1][3]:.4f}")
print(f"\ngain from pruning    : {best_va[3] - rows[0][3]:+.4f}")
print("\nTraining accuracy falls monotonically as leaves are removed — that")
print("is what pruning does, by construction — while validation and test")
print("accuracy rise to a peak and then collapse. The peak is the")
print("bias-variance minimum of Chapter 34, located by search. Note that")
print("validation and test agree on roughly where it is, which is the whole")
print("reason a validation set is worth holding out.")

# --- section 5.5: MDI is biased towards high-cardinality features -----------
def mdi(root, N, D):
    """Mean decrease in impurity, eq. 36.6."""
    imp = np.zeros(D)

    def walk(node):
        if "f" not in node:
            return
        dec = node["imp"] - (node["l"]["n"] / node["n"]) * node["l"]["imp"] \
            - (node["r"]["n"] / node["n"]) * node["r"]["imp"]
        imp[node["f"]] += (node["n"] / N) * dec
        walk(node["l"])
        walk(node["r"])

    walk(root)
    return imp / max(imp.sum(), 1e-12)


def permutation_importance(root, X, y, n_repeats=10):
    """Shuffle one column of HELD-OUT data; measure the accuracy drop."""
    base = (predict(root, X) == y).mean()
    out = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            rng.shuffle(Xp[:, j])
            drops.append(base - (predict(root, Xp) == y).mean())
        out[j] = float(np.mean(drops))
    return out


print("\n" + "=" * 72)
print("mean decrease in impurity is biased (section 5.5)")
print("=" * 72)
print("Test 1. The target depends on NOTHING — it is a coin flip. Every")
print("honest importance measure should report zero for all four features.\n")


def cardinality_features(n):
    return np.column_stack([
        rng.normal(size=n),                        # ~n distinct values
        rng.integers(0, 20, n).astype(float),      # 20 levels
        rng.integers(0, 4, n).astype(float),       # 4 levels
        rng.integers(0, 2, n).astype(float),       # binary
    ])


names = ["continuous (~800 values)", "20 levels", "4 levels", "binary"]
Xb, yb = cardinality_features(800), (rng.random(800) < 0.5).astype(int)
Xb_te, yb_te = cardinality_features(3000), (rng.random(3000) < 0.5).astype(int)
tb = grow(Xb, yb, max_depth=8)

print(f"{'feature':<26} {'MDI (train)':>13} {'permutation (test)':>20} "
      f"{'truth':>7}")
for nm, a, b in zip(names, mdi(tb, len(yb), 4),
                    permutation_importance(tb, Xb_te, yb_te)):
    print(f"{nm:<26} {a:>13.4f} {b:>20.4f} {0.0:>7.1f}")

print("\nMDI hands most of the credit to the continuous feature and almost")
print("none to the binary one, purely because a continuous feature offers")
print("799 candidate thresholds and a binary feature offers one. There is no")
print("signal in this data at all. Permutation importance on held-out data")
print("reports approximately zero for everything, which is correct.")

print("\nTest 2. Now the binary feature is the ONLY real predictor, and the")
print("other three remain pure noise.\n")

Xg = cardinality_features(800)
yg = np.where(rng.random(800) < np.where(Xg[:, 3] > 0.5, 0.85, 0.15), 1, 0)
Xg_te = cardinality_features(3000)
yg_te = np.where(rng.random(3000) < np.where(Xg_te[:, 3] > 0.5, 0.85, 0.15),
                 1, 0)
tg = grow(Xg, yg, max_depth=8)

print(f"{'feature':<26} {'MDI (train)':>13} {'permutation (test)':>20} "
      f"{'truth':>7}")
truth = ["noise", "noise", "noise", "THE signal"]
for nm, a, b, t in zip(names, mdi(tg, len(yg), 4),
                       permutation_importance(tg, Xg_te, yg_te), truth):
    print(f"{nm:<26} {a:>13.4f} {b:>20.4f} {t:>7}")

print("\nMDI does find the real feature — and still awards 28% of the credit")
print("to three columns that contain nothing, because below the one real")
print("split a deep tree keeps carving up noise and every such split counts")
print("towards its feature's total. Permutation importance on held-out data")
print("gives them approximately zero.")
print("\nThis is not a subtle bias. Ranking features by")
print("`feature_importances_` partly ranks them by cardinality, and it is")
print("measured on the data the tree was fitted to.")

# Extracted from: Chapter 36 — Decision Trees
# Source: src/.../ch036-decision-trees.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""CART from scratch: impurity, exhaustive split search, recursion, pruning.
"""
import numpy as np

rng = np.random.default_rng(0)


def gini(y):
    """1 - sum p_c^2  (eq. 36.1)."""
    if len(y) == 0:
        return 0.0
    p = np.bincount(y, minlength=2) / len(y)
    return float(1.0 - np.sum(p ** 2))


def entropy(y):
    if len(y) == 0:
        return 0.0
    p = np.bincount(y, minlength=2) / len(y)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


class Node:
    __slots__ = ("feature", "threshold", "left", "right", "value", "n", "imp")

    def __init__(self, value, n, imp):
        self.feature = self.threshold = self.left = self.right = None
        self.value, self.n, self.imp = value, n, imp

    @property
    def is_leaf(self):
        return self.left is None


class Tree:
    """Binary classification CART. Deliberately literal: the split search is
    the exhaustive scan of section 5.2, not a clever one."""

    def __init__(self, criterion="gini", max_depth=None, min_samples_leaf=1,
                 min_impurity_decrease=0.0):
        self.imp = {"gini": gini, "entropy": entropy}[criterion]
        self.max_depth = max_depth if max_depth is not None else 10 ** 9
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.n_splits_evaluated = 0

    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.root = self._grow(X, y, 0)
        return self

    def _best_split(self, X, y):
        """Exhaustive over every feature and every midpoint (eq. 36.3)."""
        n = len(y)
        parent = self.imp(y)
        best = (0.0, None, None)
        for j in range(X.shape[1]):
            order = np.argsort(X[:, j], kind="mergesort")
            xs, ys = X[order, j], y[order]
            for i in range(self.min_samples_leaf,
                           n - self.min_samples_leaf + 1):
                if xs[i] == xs[i - 1]:
                    continue               # no threshold separates equal values
                self.n_splits_evaluated += 1
                dec = parent - (i / n) * self.imp(ys[:i]) \
                             - ((n - i) / n) * self.imp(ys[i:])
                if dec > best[0]:
                    best = (dec, j, 0.5 * (xs[i] + xs[i - 1]))
        return best

    def _grow(self, X, y, depth):
        node = Node(float(y.mean()), len(y), self.imp(y))
        if (depth >= self.max_depth or len(y) < 2 * self.min_samples_leaf
                or node.imp == 0.0):
            return node
        dec, j, thr = self._best_split(X, y)
        if j is None or dec <= self.min_impurity_decrease:
            return node
        m = X[:, j] <= thr
        node.feature, node.threshold = j, thr
        node.left = self._grow(X[m], y[m], depth + 1)
        node.right = self._grow(X[~m], y[~m], depth + 1)
        return node

    def _predict_one(self, x, node):
        while not node.is_leaf:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value

    def predict_proba(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def n_leaves(self, node=None):
        node = self.root if node is None else node
        return 1 if node.is_leaf else (self.n_leaves(node.left)
                                       + self.n_leaves(node.right))

    def depth(self, node=None):
        node = self.root if node is None else node
        return 0 if node.is_leaf else 1 + max(self.depth(node.left),
                                              self.depth(node.right))


# --- data: two informative features, a nonlinear boundary -------------------
def make_data(n):
    X = rng.uniform(-3, 3, (n, 2))
    z = np.sin(1.2 * X[:, 0]) + 0.6 * X[:, 1] - 0.25 * X[:, 0] * X[:, 1]
    p = 1 / (1 + np.exp(-2.0 * z))
    return X, (rng.random(n) < p).astype(int)


Xtr, ytr = make_data(600)
Xte, yte = make_data(4000)

print("=" * 72)
print("depth is the complexity knob (Chapter 34)")
print("=" * 72)
print(f"{'max_depth':>10} {'leaves':>8} {'train acc':>11} {'test acc':>10} "
      f"{'gap':>8}")
for d in (1, 2, 3, 5, 8, 12, None):
    t = Tree(max_depth=d).fit(Xtr, ytr)
    tr = (t.predict(Xtr) == ytr).mean()
    te = (t.predict(Xte) == yte).mean()
    label = "full" if d is None else str(d)
    print(f"{label:>10} {t.n_leaves():>8} {tr:>11.4f} {te:>10.4f} "
          f"{tr - te:>8.4f}")
print("\nThe unrestricted tree reaches training accuracy 1.0000 with one leaf")
print("per distinct region — it has memorised (Chapter 31) — and its test")
print("accuracy is well below the best shallow tree. This is the sharpest")
print("overfitting demonstration in Part IV because nothing is regularising")
print("it at all.")

# --- section 6.1: Gini vs entropy rarely matters ----------------------------
print("\n" + "=" * 72)
print("Gini vs entropy (section 6.2: they agree to first order)")
print("=" * 72)
print(f"{'max_depth':>10} {'gini test':>11} {'entropy test':>14} "
      f"{'same prediction?':>18}")
for d in (2, 3, 5, 8):
    g = Tree("gini", max_depth=d).fit(Xtr, ytr)
    e = Tree("entropy", max_depth=d).fit(Xtr, ytr)
    agree = (g.predict(Xte) == e.predict(Xte)).mean()
    print(f"{d:>10} {(g.predict(Xte) == yte).mean():>11.4f} "
          f"{(e.predict(Xte) == yte).mean():>14.4f} {agree:>18.4f}")
print("\nThey agree on nearly every prediction, as eq. 36.9 says they must.")
print("This is not a hyperparameter worth searching over.")

# --- section 6.3: greed, and the XOR trap -----------------------------------
print("\n" + "=" * 72)
print("what greed costs: the XOR trap (section 6.3)")
print("=" * 72)
n = 2000
Xx = rng.normal(size=(n, 2))
yx = ((Xx[:, 0] > 0) ^ (Xx[:, 1] > 0)).astype(int)
Xx_te = rng.normal(size=(4000, 2))
yx_te = ((Xx_te[:, 0] > 0) ^ (Xx_te[:, 1] > 0)).astype(int)

print("the best single split at the root, on pure XOR:")
probe = Tree(max_depth=1)
probe.n_features = 2
dec, j, thr = probe._best_split(Xx, yx)
print(f"  best impurity decrease available = {dec:.6f} "
      f"(feature {j}, threshold {thr:.4f})")
print("  Essentially zero: every split leaves both children at 50/50.\n")

print(f"{'min_impurity_decrease':>23} {'leaves':>8} {'depth':>7} "
      f"{'test accuracy':>15}")
for mid in (0.0, 0.001, 0.01, 0.05):
    t = Tree(max_depth=6, min_impurity_decrease=mid).fit(Xx, yx)
    print(f"{mid:>23} {t.n_leaves():>8} {t.depth():>7} "
          f"{(t.predict(Xx_te) == yx_te).mean():>15.4f}")

print("\nA threshold of 0.01 — a value that looks conservative — stops the")
print("tree at the root and produces a coin flip. Two splits in sequence")
print("solve XOR perfectly; no single split makes any progress at all.")
print("This is why the standard advice is GROW FIRST, PRUNE AFTER: pruning")
print("can see what a subtree eventually achieved, and early stopping")
print("cannot.")

# --- section 6.4: invariance to monotone transforms -------------------------
print("\n" + "=" * 72)
print("trees are invariant to any monotone transform (section 6.4)")
print("=" * 72)
base = Tree(max_depth=4).fit(Xtr, ytr)
base_pred = base.predict(Xte)
_sorted_x1 = np.sort(Xtr[:, 0])          # the rank map must be FITTED on
                                         # train and then applied to test,
                                         # exactly like any other transform
transforms = {
    "raw": lambda A: A,
    "x1 * 10000": lambda A: A * np.array([10000.0, 1.0]),
    "standardised": lambda A: (A - Xtr.mean(0)) / Xtr.std(0),
    "exp(x1)": lambda A: np.column_stack([np.exp(A[:, 0]), A[:, 1]]),
    "log1p(x1 + 4)": lambda A: np.column_stack(
        [np.log1p(A[:, 0] + 4.0), A[:, 1]]),
    "rank of x1": lambda A: np.column_stack(
        [np.searchsorted(_sorted_x1, A[:, 0]).astype(float), A[:, 1]]),
}
print(f"{'transform':<16} {'test accuracy':>14} {'identical to raw?':>19}")
for name, f in transforms.items():
    t = Tree(max_depth=4).fit(f(Xtr), ytr)
    pred = t.predict(f(Xte))
    print(f"{name:<16} {(pred == yte).mean():>14.4f} "
          f"{str(np.array_equal(pred, base_pred)):>19}")
print("\nMultiplying a feature by 10,000, exponentiating it, taking a log, or")
print("replacing it by its rank produces BITWISE IDENTICAL predictions — the")
print("same samples fall on the same side of the same split, because the")
print("split only ever asks 'is x_j below a constant?' (section 6.4).")
print("Compare Chapter 35, where multiplying one feature by 10,000 cost k-NN")
print("17 accuracy points. Note that the rank map must be fitted on the")
print("training data and applied to the test data, like any other transform;")
print("computing ranks separately per array is a different function on each")
print("and would break the invariance for an unrelated reason.")

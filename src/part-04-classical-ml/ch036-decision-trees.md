---
id: ml-trees
number: 36
part: IV
tier: focused
status: reviewed
requires: [ml-metrics, ml-knn-nb, math-probability]
provides: [decision-tree, gini-impurity, entropy-split, information-gain,
           cart, pruning, feature-importance, axis-aligned-splits,
           greedy-recursive-partitioning]
citations: [breiman2001cultures, grinsztajn2022, pedregosa2011]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Implement CART from scratch: split search, recursion, and prediction.
2. Derive Gini impurity and entropy and explain when the choice matters.
3. Explain why the split search is greedy and what that costs.
4. Prune a tree by cost-complexity and explain why pruning beats early
   stopping.
5. Explain the inductive bias of axis-aligned splits and what it cannot
   represent.
6. Explain why impurity-based feature importance is biased, and what to use
   instead.
7. State why trees need no scaling and handle mixed types natively.

## 2. Why This Matters

The single tree is a mediocre model and the foundation of the best tabular
models in existence.

**Everything in the next two chapters is trees.** Random forests average
hundreds of them; gradient boosting stacks thousands. If the split criterion and
the greedy recursion are not clear here, {{ch:ml-boosting}} — which is the model
you will most often reach for on tabular data in 2026 — will be a hyperparameter
list rather than an algorithm.

**Trees have an unusual set of properties, and they explain the tabular
result.** No scaling, no imputation strategy required, native handling of
categorical and numeric features together, invariance to any monotone transform
of a feature, and automatic discovery of interactions. {{cite:grinsztajn2022}}
attributes tree dominance on tabular data to three specific properties, all of
which are visible in this chapter's mechanics rather than mysterious.

**A shallow tree is the only genuinely interpretable model in this part.** Not
"interpretable" in the sense of a coefficient you have to reason about, but a
flowchart a domain expert can read, disagree with, and correct. That is worth a
great deal in a regulated or high-stakes setting, and it is why depth-3 trees
still appear in clinical decision rules.

## 3. Prerequisites

{{ch:ml-metrics}} for bias-variance and the overfitting the tree will
demonstrate more dramatically than any other model here.
{{ch:math-probability}} for entropy. {{ch:ml-knn-nb}} for the contrast: trees are
immune to the scaling and irrelevant-feature problems k-NN suffers from.

## 4. Intuitive Explanation

### 4.1 Twenty questions

A tree is a sequence of yes/no questions about single features, arranged so each
answer narrows the possibilities.

```text
                    ┌─────────────────────┐
                    │  income < 42,000 ?  │
                    └──────────┬──────────┘
                    yes ┌──────┴──────┐ no
                        ▼             ▼
              ┌──────────────┐   ┌──────────────────┐
              │ n_late < 2 ? │   │ utilisation<0.6? │
              └──────┬───────┘   └────────┬─────────┘
              yes ┌──┴──┐ no        yes ┌─┴──┐ no
                  ▼     ▼               ▼    ▼
               p=0.04  p=0.31        p=0.02 p=0.19
```

Each internal node tests one feature against one threshold; each leaf holds a
prediction. Prediction is a walk from the root to a leaf, costing $O(\text{depth})$
comparisons — which is why trees are among the fastest models to serve.

The **inductive bias** is right there in the picture: every boundary is
perpendicular to an axis. A tree carves the feature space into rectangles.

### 4.2 How the questions are chosen

Greedily. At each node, consider every feature and every candidate threshold,
score each split by how much it reduces impurity, take the best, and recurse.

**Impurity** measures how mixed a node's labels are. A node that is all one
class has impurity zero. A 50/50 node has maximum impurity. A good split
produces children that are purer than the parent, and the improvement is
weighted by how many samples land in each child — a split isolating three
samples perfectly is worth less than one that cleanly separates half the data.

The search is exhaustive within a node and greedy across nodes: it never
reconsiders an earlier split in light of a later one. Finding the globally
optimal tree is NP-hard, so greed is not laziness — it is the only tractable
option, and {{sec:6-mathematical-foundation}} shows exactly what it costs.

### 4.3 Why a full tree is useless

Grow a tree until every leaf is pure and you have a memoriser. Every training
point sits in its own leaf and training error is zero — exactly the
{{ch:ml-what-it-is}} memoriser, reached by a route that looks like learning.

Two remedies, and the ordering between them is not obvious:

**Early stopping** (pre-pruning) refuses to split when the improvement is too
small. It is cheap and it is myopic: a split that looks worthless can enable an
excellent one below it. XOR is the standard example — neither feature alone
reduces impurity at all, so a greedy criterion with a minimum-gain threshold
stops at the root and never finds the structure that a second split would
reveal.

**Cost-complexity pruning** (post-pruning) grows the tree fully and then removes
subtrees that do not earn their complexity. It costs more and it can see what
the subtree eventually achieved. This is the standard advice, and
{{sec:7-implementation}} measures the XOR case where the difference is total.

### 4.4 What trees are bad at

**Smooth relationships.** Approximating a straight line with axis-aligned steps
takes many splits and still looks like a staircase. A linear model needs two
parameters.

**Diagonal boundaries.** The boundary $x_1 + x_2 = 0$ is one linear split and an
unbounded number of axis-aligned ones.

**Extrapolation.** A tree's prediction outside the training range is the nearest
leaf's constant, forever. Feed a tree trained on prices from £100k to £900k an
input implying £5M and it returns roughly £900k. Linear models extrapolate —
sometimes wrongly, but they respond. This is a genuine and underrated hazard for
any tabular model on a drifting distribution ({{ch:mle-drift}}).

**Instability.** Change one training point and the root split can change, which
changes everything below it. That instability is high variance in the sense of
{{ch:ml-metrics}} — and averaging it away is precisely what {{ch:ml-forests}}
does.

## 5. Formal Explanation

### 5.1 Impurity measures

For a node with class proportions $p_1, \dots, p_C$:

$$
\text{Gini}(t) = 1 - \sum_{c=1}^{C} p_c^{2}
\qquad
H(t) = -\sum_{c=1}^{C} p_c \log_2 p_c
$$ (eq:impurity)

Gini is the probability that two independently drawn samples from the node have
different labels. Entropy is the expected bits needed to encode a label. Both are
zero for a pure node and maximised at the uniform distribution — $1 - 1/C$ for
Gini, $\log_2 C$ for entropy.

For regression, impurity is the variance:

$$
\text{MSE}(t) = \frac{1}{n_t}\sum_{i \in t}(y_i - \bar{y}_t)^{2}
$$ (eq:regression-impurity)

### 5.2 The split criterion

A split $s$ sends $n_L$ samples left and $n_R$ right. Its **impurity decrease**
is

$$
\Delta I(s, t) = I(t) - \frac{n_L}{n_t}I(t_L) - \frac{n_R}{n_t}I(t_R)
$$ (eq:impurity-decrease)

With entropy this is called **information gain**. The weighting by child size is
essential; without it, a split peeling off one perfectly-classified sample would
score best.

CART chooses $\argmax_s \Delta I(s, t)$ over all features and all thresholds. For
a numeric feature with $n_t$ distinct values there are $n_t - 1$ candidate
thresholds, conventionally taken at midpoints between adjacent sorted values.

> NOTE: **Gini or entropy?** They agree on the vast majority of splits and
> produce measurably different trees only rarely. Gini is marginally cheaper
> (no logarithm) and is scikit-learn's default {{cite:pedregosa2011}}. Entropy
> penalises impurity slightly more sharply and can favour more balanced splits.
> If your model selection is sensitive to this choice, the difference is noise
> and you are overfitting the validation set ({{ch:ml-metrics}}).

### 5.3 Categorical features

For an unordered categorical feature with $q$ levels there are $2^{q-1}-1$
possible binary partitions — intractable beyond about 15 levels.

Two escapes. For **binary classification and regression** there is an exact
shortcut: sort the levels by their mean target value and consider only the $q-1$
splits along that order; this provably contains the optimal partition. For
**multiclass**, no such shortcut exists, and implementations either one-hot
encode (which fragments the feature across many weak binary splits) or use
target statistics with the leakage protection {{ch:ds-feature-eng}} described.
CatBoost's ordered target statistics ({{ch:ml-boosting}}) are a systematic
solution to exactly this.

### 5.4 Cost-complexity pruning

Grow $T_{\max}$ fully, then define for a subtree $T$

$$
R_{\alpha}(T) = R(T) + \alpha |\tilde{T}|
$$ (eq:cost-complexity)

where $R(T)$ is training error, $|\tilde{T}|$ is the number of leaves, and
$\alpha \ge 0$ prices complexity. For each $\alpha$ there is a unique smallest
minimising subtree, and as $\alpha$ increases from 0 the sequence of optimal
subtrees is **nested**: each is obtained from the previous by collapsing one
more internal node. That nesting is what makes the whole path computable in one
pass, and $\alpha$ is then chosen by cross-validation.

The **weakest link** at each stage is the internal node $t$ minimising

$$
g(t) = \frac{R(t) - R(T_t)}{|\tilde{T_t}| - 1}
$$ (eq:weakest-link)

the extra training error per leaf saved by collapsing the subtree $T_t$ rooted
at $t$. Collapsing it and repeating generates the whole nested sequence.

### 5.5 Feature importance, and why the default is wrong

The conventional measure sums, over all nodes splitting on feature $j$, the
impurity decrease weighted by the samples reaching the node:

$$
\text{Imp}(j) = \sum_{t \,:\, \text{split on } j}
   \frac{n_t}{N}\,\Delta I(s_t, t)
$$ (eq:mdi)

This is **mean decrease in impurity** (MDI), and it is biased in a specific and
serious way: **it favours high-cardinality features.** A continuous feature
offers $n-1$ candidate thresholds and a binary feature offers one, so the
continuous feature gets many more chances to find a lucky split. A pure random
noise column with many distinct values will out-rank a genuinely predictive
binary column. {{sec:7-implementation}} measures this on a target that depends
on nothing at all.

**Permutation importance** avoids the bias: shuffle one feature in the
*validation* set and measure the drop in score. It is model-agnostic, measures
what you actually care about, and costs one extra scoring pass per feature. It
has its own flaw — with correlated features, permuting one leaves the
information available through the other, so both look unimportant — and
{{ch:rai-interpretability}} covers the alternatives.

> WARNING: `feature_importances_` on a fitted tree or forest is MDI. It is
> computed on training data, it is biased towards high-cardinality features, and
> it is reported in a great many analyses as though it meant something. Use
> permutation importance on held-out data.

## 6. Mathematical Foundation

### 6.1 Why impurity must be concave

Any sensible impurity measure $I(p)$ must be **strictly concave** in the class
proportions. The reason: concavity is exactly the condition guaranteeing that
splitting never *increases* weighted impurity.

By Jensen's inequality, for a concave $I$ and a split with weights $w_L = n_L/n_t$
and $w_R = n_R/n_t$,

$$
w_L I(p_L) + w_R I(p_R) \le I(w_L p_L + w_R p_R) = I(p_t)
$$ (eq:jensen-impurity)

since the parent's proportions are exactly the weighted average of the
children's. So $\Delta I \ge 0$ always, with equality only when $p_L = p_R$ —
that is, when the split carries no information about the label.

Both Gini ($1 - \sum p_c^{2}$) and entropy are strictly concave; classification
*error rate* $1 - \max_c p_c$ is concave but **not strictly**, which is why it is
a poor split criterion. A split can leave the error rate unchanged while
substantially improving purity, so error rate is blind to progress that Gini and
entropy can see. It is nonetheless the right criterion for *pruning*, where the
question is final performance rather than progress.

### 6.2 Gini as a first-order approximation to entropy

Expand $\log_2 p$ about $p = 1$ using $\ln p \approx p - 1$:

$$
H = -\sum_c p_c \log_2 p_c
  \approx \frac{1}{\ln 2}\sum_c p_c(1 - p_c)
  = \frac{1}{\ln 2}\Big(1 - \sum_c p_c^{2}\Big)
  = \frac{\text{Gini}}{\ln 2}
$$ (eq:gini-entropy)

The two criteria are the same quantity up to a constant, to first order. That is
the formal reason they so rarely disagree, and it means the choice between them
almost never deserves a hyperparameter search.

### 6.3 What greed costs

Finding the smallest tree consistent with a training set is NP-complete, so CART
uses a greedy heuristic. The failure mode is precise: **a split whose immediate
gain is zero can be a prerequisite for a large gain below it.**

XOR is the canonical case. With $y = \Ind[x_1 > 0] \oplus \Ind[x_2 > 0]$ and both
features independent and symmetric, splitting on either at zero gives two
children each still exactly 50/50. So

$$
\Delta I = I(t) - \tfrac{1}{2}I(t_L) - \tfrac{1}{2}I(t_R) = 0
$$

for *every* split at the root. The greedy criterion sees no reason to split at
all — and yet two splits in sequence classify the data perfectly.

Three consequences, all practical:

**Do not set `min_impurity_decrease` above zero without thinking.** A positive
threshold makes the tree stop at the root on XOR-like structure. Grow first,
prune after.

**Depth-1 trees cannot find interactions, by construction.** This is why boosting
with stumps fits an additive model and boosting with depth-3 trees can represent
three-way interactions — a distinction {{ch:ml-boosting}} makes precise.

**Randomised split selection can escape it.** Choosing among a random subset of
features, as {{ch:ml-forests}} does, sometimes takes a zero-gain split for
reasons unrelated to its gain, and can therefore stumble into structure that a
purely greedy search will not.

### 6.4 Why a tree needs no scaling

A split is $x_j < \tau$. For any strictly increasing $\phi$, the split
$\phi(x_j) < \phi(\tau)$ selects **exactly the same samples**. Trees are
therefore invariant to every monotone transformation of any feature: log,
square root, standardisation, rank, min-max, all of it.

The contrast with {{ch:ml-knn-nb}} is total and is worth stating as the general
principle. Distance-based methods and gradient-based methods on raw inputs are
sensitive to scale because they *combine* features numerically; trees only ever
*compare* a feature against a constant, one at a time. This is also why a tree is
unbothered by an irrelevant feature — it simply never splits on it — where k-NN
must include it in every distance.

## 7. Implementation

```python {tier=A name=cart-from-scratch}
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
```

```python {tier=A name=pruning-and-importance}
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
```

## 8. Practical Example

```python {tier=A name=tree-vs-linear}
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
```

## 9. Common Mistakes

**Using a fully grown tree.** It memorises; the measured gap is the largest in
this part.

**Setting `min_impurity_decrease` above zero.** The XOR measurement shows a
conservative-looking 0.01 reducing the model to a coin flip.

**Trusting `feature_importances_`.** It is MDI, computed on training data, and
biased towards high-cardinality features — measured here on a target with no
signal at all.

**Searching over Gini versus entropy.** They agree to first order; the search is
fitting noise.

**One-hot encoding a high-cardinality categorical for a tree.** It fragments one
feature into many weak binary splits, each of which the tree must find
separately.

**Expecting a tree to extrapolate.** It returns the nearest leaf's constant
forever, and says nothing about being out of range.

**Comparing trees to linear models on a smooth target.** The staircase is the
inductive bias, not a bug.

**Treating a single tree's structure as a finding.** Change one point and the
root can change.

**Interpreting a deep tree.** A depth-3 tree is a flowchart; a depth-25 tree is
a black box with worse accuracy than a forest.

## 10. Connection to Previous Chapters

{{ch:ml-metrics}} supplied the bias-variance frame that depth traverses and
that pruning searches over. {{ch:ml-knn-nb}} is the exact contrast: k-NN is
destroyed by rescaling one feature and degraded by irrelevant ones, and the same
measurements here show a tree is completely unaffected by both.
{{ch:math-probability}} supplied entropy. {{ch:ds-feature-eng}} supplied the
target-encoding machinery that {{sec:5-formal-explanation}} needs for
high-cardinality categoricals.

Forward: {{ch:ml-forests}} averages the instability measured at the end of
{{sec:8-practical-example}} — the whole method is a response to that one
property. {{ch:ml-boosting}} keeps the greedy tree and changes what it is fitted
to, and uses the depth limit of {{sec:6-mathematical-foundation}} to control
interaction order. {{ch:rai-interpretability}} returns to feature importance with
SHAP and the correlated-feature problem permutation importance does not solve.
{{ch:mle-drift}} monitors for the extrapolation failure.

## 11. Exercises

**Beginner**

1. Compute the Gini impurity of a node with 30 positives and 10 negatives.
2. Compute the entropy of the same node.
3. Why is the impurity decrease weighted by child size?
4. Why do trees not need feature scaling?
5. What does a tree predict for an input far outside the training range?

**Intermediate**

6. Compute {{eq:impurity-decrease}} for a split of $(40,40)$ into $(30,10)$ and
   $(10,30)$, using Gini.
7. Explain why classification error is a poor split criterion but a reasonable
   pruning criterion.
8. Why is early stopping worse than pruning? Give the XOR argument.
9. Explain the MDI bias towards high-cardinality features.
10. How many binary partitions does a 10-level categorical feature admit, and
    what shortcut applies for binary classification?
11. Why does a depth-1 tree fit an additive model?

**Advanced**

12. Prove {{eq:jensen-impurity}} and state where strict concavity is needed.
13. Derive {{eq:gini-entropy}} and quantify the approximation error at
    $p = 0.1$.
14. Prove the cost-complexity subtree sequence is nested.
15. Prove the sorted-by-mean-target shortcut for categorical splits in the
    binary case.
16. Construct a target where a greedy tree provably needs exponentially more
    leaves than the optimal tree.

**Implementation**

17. Extend the CART implementation to regression using the cumulative-sum trick
    from {{sec:8-practical-example}}, and explain why it makes the split search
    $O(n\log n)$ rather than $O(n^{2})$.
18. Add native categorical support using the sorted-by-mean-target shortcut.
19. Implement full cost-complexity pruning with cross-validated $\alpha$ and
    compare against depth limiting.
20. Reproduce the MDI bias experiment with a genuinely predictive binary feature
    added, and see whether MDI ranks it above the noise.

**Reasoning**

21. A depth-3 tree and a random forest give 0.82 and 0.89 accuracy on a clinical
    risk task. Which do you deploy?
22. Your tree's top feature by MDI is `customer_id`. What has happened?

## 12. Chapter Summary

A decision tree recursively partitions the feature space with axis-aligned
splits, chosen greedily to maximise the size-weighted impurity decrease. Gini
and entropy are the same quantity to first order and essentially never disagree,
so the choice between them is not worth a search.

Impurity must be strictly concave, which is what guarantees by Jensen that a
split can never increase weighted impurity — and why classification error, being
concave but not strictly, is a poor split criterion and a fine pruning
criterion.

Greed is necessary, since optimal trees are NP-hard, and its cost is precise: a
zero-gain split can be the prerequisite for a large gain below it. The measured
XOR case shows a `min_impurity_decrease` of 0.01 reducing the model to chance.
Grow first, prune after.

Cost-complexity pruning generates a nested sequence of subtrees by repeatedly
collapsing the weakest link, and cross-validation picks the point on it. The
measurement shows training accuracy falling monotonically while validation
accuracy rises to a peak — the bias-variance minimum located by search.

Trees are invariant to every monotone transformation of every feature, because a
split only ever compares one feature to a constant. This is the exact
complement of k-NN's fragility, and it is a large part of why trees dominate
tabular data.

Mean decrease in impurity is biased towards high-cardinality features. On a
target with no signal at all, MDI awards most of the importance to the
continuous column purely because it offers more candidate thresholds.
Permutation importance on held-out data reports approximately zero, correctly.

Two failures motivate the next chapter. A tree cannot extrapolate: outside the
training range it returns a constant forever, silently. And a tree is unstable:
moving one training point out of three hundred can change the root split and
everything beneath it.

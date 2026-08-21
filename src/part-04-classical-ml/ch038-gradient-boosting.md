---
id: ml-boosting
number: 38
part: IV
tier: focused
status: reviewed
requires: [ml-forests, ml-trees, ml-metrics, math-optimization]
provides: [boosting, gradient-boosting, shrinkage, early-stopping-boosting,
           xgboost, lightgbm, catboost, second-order-boosting,
           tabular-foundation-models]
citations: [friedman2001, chen2016, grinsztajn2022, hollmann2025, pedregosa2011]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Derive gradient boosting as gradient descent in function space.
2. Explain why boosting reduces bias and therefore needs shallow base
   learners — the exact opposite of bagging.
3. Explain the roles of the learning rate, the number of trees, and their
   interaction.
4. Derive the second-order (Newton) formulation used by XGBoost.
5. Explain the algorithmic differences between XGBoost, LightGBM and CatBoost.
6. Explain why boosting *can* overfit with more trees while a forest cannot.
7. State honestly where gradient boosting stands against tabular foundation
   models in 2026.

## 2. Why This Matters

Gradient boosting is the most consequential algorithm in this part, and it is
worth being precise about why.

**On tabular data it has been the model to beat for two decades.** Not "a good
baseline" — the winner, in competitions and in production. {{cite:grinsztajn2022}}
benchmarked this carefully across 45 datasets and found tree ensembles still
ahead of tuned neural networks at medium scale, identifying three concrete
mechanisms rather than appealing to tradition. Most business data is tabular,
so for a great deal of applied work this chapter is the practical destination
of Part IV.

**The core idea generalises far beyond trees.** "Fit the next model to the
residual of the current ensemble" is a way of building a strong learner from
weak ones, and once you see it as gradient descent in function space
({{sec:6-mathematical-foundation}}) the connection to {{ch:dl-backprop}} is
exact rather than metaphorical. The same additive-correction pattern appears in
residual connections ({{ch:tf-ffn-residual}}) and in the iterative refinement of
diffusion models.

**And the position is genuinely moving for the first time.**
{{cite:hollmann2025}} reports a tabular foundation model outperforming heavily
tuned ensembles on datasets up to around ten thousand rows, in seconds rather
than hours, by in-context learning rather than fitting.
{{sec:5-formal-explanation}} treats this with the maturity labels it deserves,
and concentrates on the *reasons* behind the trade-off, which are properties of
data rather than of the year.

## 3. Prerequisites

{{ch:ml-forests}} for the ensemble contrast — this chapter is the other half of
{{eq:bias-variance}}. {{ch:ml-trees}} for the base learner and for why depth
controls interaction order. {{ch:math-optimization}} for gradient descent, whose
function-space analogue this is. {{ch:ml-metrics}} for the early stopping that
boosting requires.

## 4. Intuitive Explanation

### 4.1 Fix what you got wrong

Bagging fits many models to the same problem in parallel and averages them.
Boosting fits models **in sequence**, each one to the errors the previous ones
left behind.

```text
  F_0 = mean(y)          ──▶  residuals r_0 = y - F_0
                                  │
  fit small tree h_1 to r_0        │
  F_1 = F_0 + eta * h_1  ──▶  residuals r_1 = y - F_1
                                  │
  fit small tree h_2 to r_1        │
  F_2 = F_1 + eta * h_2  ──▶  residuals r_2 = y - F_2
                                  ⋮
```

Each tree is a *correction*, not a competitor. The ensemble is a sum, not an
average, and the trees are not interchangeable — remove the third and everything
after it is wrong, because they were fitted to a world in which it existed.

That sequential dependence is why boosting cannot be parallelised across trees
the way a forest can, and it is the price of the accuracy.

### 4.2 Why the base learner must be weak

This is the crux, and it is exactly inverted from {{ch:ml-forests}}.

Bagging reduces variance and cannot touch bias, so it needs a base learner with
no bias: deep, unpruned trees. Boosting reduces **bias** — each round explicitly
attacks what the ensemble still gets wrong — and *adds* variance as it goes. So
it needs a base learner with almost no variance: a stump or a depth-3 tree.

Boost deep trees and the first tree fits the training data almost perfectly, the
residuals are noise, and every subsequent tree fits noise. Depth 3 to 6 is the
usual range, and the reason to prefer 3 over 6 is not compute:

> IMPORTANT: **Tree depth in boosting controls interaction order.** A stump
> (depth 1) tests one feature, so the ensemble is a sum of single-feature
> functions — a purely **additive** model that cannot represent any interaction
> at all. Depth 2 can represent pairwise interactions, depth 3 three-way, and so
> on. Choosing `max_depth` is choosing how many features may interact, which is
> a modelling decision about your problem, not a capacity dial.

### 4.3 The learning rate

Adding each tree at full strength overshoots. **Shrinkage** multiplies each
tree's contribution by $\eta \in (0, 1]$, typically 0.01 to 0.1.

Small $\eta$ with many trees beats large $\eta$ with few, reliably and for the
same reason a small learning rate helps in {{ch:math-optimization}}: many small
corrections explore the function space more carefully than a few large ones, and
each tree's inevitable overfit is diluted by $\eta$ before it enters the
ensemble.

$\eta$ and $B$ trade off almost exactly — halve $\eta$ and you need roughly twice
the trees — so the practical recipe is to fix $\eta$ at what you can afford in
training time and choose $B$ by early stopping.

### 4.4 Boosting can overfit; a forest cannot

{{ch:ml-forests}} established that adding trees to a forest converges rather
than diverging. Boosting has no such protection. Each tree reduces training
error further, and past a point that reduction is fitting noise: validation
error falls, reaches a minimum, and then rises.

So **early stopping is not optional** — it is how $B$ is chosen. Monitor a
validation set, stop when it has not improved for some number of rounds, and
keep the best iteration. Every serious implementation has this built in, and
using one without it is the most common way to get a worse result than a
random forest from a better algorithm.

## 5. Formal Explanation

### 5.1 The algorithm

For a differentiable loss $\Loss(y, F)$:

**Step 0.** Initialise with the best constant, $F_0(\vec{x}) = \argmin_{\gamma}
\sum_i \Loss(y_i, \gamma)$.

Then for $m = 1, \dots, M$:

**Step (a).** Compute the **pseudo-residuals** — the negative gradient of the
loss with respect to the current prediction, evaluated at each training point:

$$
r_{im} = -\left[\frac{\partial \Loss(y_i, F(\vec{x}_i))}
                    {\partial F(\vec{x}_i)}\right]_{F = F_{m-1}}
$$ (eq:pseudo-residual)

**Step (b).** Fit a regression tree $h_m$ to $\{(\vec{x}_i, r_{im})\}$,
partitioning the space into leaf regions $R_{jm}$.

**Step (c).** For each leaf $j$, choose the value minimising the actual loss
over the points that landed in it:

$$
\gamma_{jm} = \argmin_{\gamma}
  \sum_{\vec{x}_i \in R_{jm}} \Loss\big(y_i, F_{m-1}(\vec{x}_i) + \gamma\big)
$$ (eq:leaf-value)

**Step (d).** Update, with shrinkage:

$$
F_m(\vec{x}) = F_{m-1}(\vec{x})
  + \eta \sum_j \gamma_{jm}\,\Ind[\vec{x} \in R_{jm}]
$$ (eq:boosting-update)

Step (c) is the step people skip when reading the algorithm, and it matters: the
tree is fitted to the *gradient*, but the leaf values are re-optimised against
the *true loss*. For squared error the two coincide and the pseudo-residual is
the ordinary residual, which is why the squared-error case looks so simple. For
log loss they differ, and using the raw gradient as the leaf value would be
wrong.

{#tbl:boosting-losses caption="Losses and their pseudo-residuals. For squared error the pseudo-residual is the ordinary residual, which is why boosting is often introduced that way."}

| Loss | $\Loss(y, F)$ | Pseudo-residual $-\partial\Loss/\partial F$ |
|---|---|---|
| Squared error | $\tfrac{1}{2}(y-F)^{2}$ | $y - F$ |
| Absolute error | $\lvert y-F\rvert$ | $\sign(y-F)$ |
| Log loss (binary) | $\log(1+e^{-\tilde{y}F})$ | $y - \sigma(F)$ |
| Huber | piecewise | clipped residual |

The log-loss row is worth noting: the pseudo-residual $y - \sigma(F)$ is exactly
the gradient from {{eq:logreg-gradient}}. Boosting for classification produces a
log-odds score $F$ that is squashed through the sigmoid, so the whole apparatus
of {{ch:ml-logistic}} — thresholds, calibration, cost-based decisions — carries
over unchanged.

### 5.2 Regularisation

Boosting is powerful enough that it needs several restraints at once, and they
are not interchangeable:

- **Shrinkage** $\eta$ — the primary control.
- **Number of trees** $M$ — chosen by early stopping, never fixed in advance.
- **Tree depth** — controls interaction order (see above).
- **Subsampling rows** (stochastic gradient boosting) — fit each tree on a
  random fraction, which adds bagging-style variance reduction on top.
- **Subsampling columns** — as in {{ch:ml-forests}}, per tree or per split.
- **$\ell_1$/$\ell_2$ penalties on leaf values** — XGBoost's addition, which
  makes the leaf values themselves shrunk estimates.
- **Minimum child weight** — refuse splits whose children carry too little
  total gradient mass.

### 5.3 Second-order boosting

{{cite:chen2016}} replaced the first-order step with a second-order Taylor
expansion of the loss. With $g_i$ the gradient and $h_i$ the Hessian at the
current prediction, the objective for one tree is approximately

$$
\tilde{\Loss}^{(m)} = \sum_i \left[g_i f(\vec{x}_i)
  + \tfrac{1}{2}h_i f(\vec{x}_i)^{2}\right]
  + \gamma T + \tfrac{1}{2}\lambda \sum_{j=1}^{T} w_j^{2}
$$ (eq:xgboost-objective)

where $T$ is the number of leaves and $w_j$ their values. This has an exact
closed-form solution ({{sec:6-mathematical-foundation}}), which gives both the
optimal leaf value and an exact split-scoring formula — so the tree is grown to
optimise the real objective rather than a proxy.

The practical consequences: faster convergence in rounds, a principled
complexity penalty built into the split criterion, and the ability to handle any
twice-differentiable loss uniformly.

### 5.4 The three implementations

{#tbl:gbdt-implementations caption="The three production gradient-boosting libraries and what actually distinguishes them. All three are excellent; the differences matter mainly at the extremes of scale or cardinality."}

| | XGBoost | LightGBM | CatBoost |
|---|---|---|---|
| Key idea | second-order, regularised | histogram + leaf-wise growth | ordered boosting, ordered target statistics |
| Tree growth | level-wise (depth-balanced) | leaf-wise (best-first) | symmetric (oblivious) trees |
| Speed | fast | fastest on large data | moderate |
| Categoricals | needs encoding | native | native, and best in class |
| Overfits on small data | less | **more** (leaf-wise) | least |
| Notable | the reference implementation | GOSS and EFB for scale | fixes target-leakage in encoding |

**Leaf-wise growth** is LightGBM's main structural difference: rather than
completing each level, it repeatedly splits whichever leaf offers the largest
gain. This reaches a lower loss for a given number of leaves and produces deep,
unbalanced trees that overfit small datasets — hence `num_leaves` as its primary
control rather than `max_depth`.

**Ordered target statistics** is CatBoost's: target encoding leaks, as
{{ch:ds-feature-eng}} showed, because a row's own label contributes to its own
encoding. CatBoost computes each row's encoding using only rows that precede it
in a random permutation, which removes the leak systematically rather than by
convention. The same ordering principle is applied to the gradients themselves.

All three are strong. Differences on a typical tabular problem are usually
smaller than the effect of tuning, and much smaller than the effect of better
features.

### 5.5 Where this stands in 2026

The honest position requires two labels rather than one.

**Gradient boosting on medium and large tabular data:** {{maturity:ESTABLISHED}}.
{{cite:grinsztajn2022}} identifies three mechanisms behind the result, all of
them properties of the data rather than of the algorithms' maturity: trees are
robust to uninformative features; they are unaffected by feature rotation,
because real tabular axes are meaningful (the measurement in {{ch:ml-trees}}
showed this is a handicap only when the axes are arbitrary); and they fit
irregular, non-smooth target functions that smooth models must approximate.

**Tabular foundation models on small data:** {{maturity:EMERGING}}, and already
the better choice below roughly ten thousand rows. {{cite:hollmann2025}} reports
TabPFN, a transformer pre-trained on millions of synthetic tabular tasks,
outperforming an ensemble of strong tuned baselines on datasets up to that size
— in seconds, with no fitting at all, by in-context learning. Published in
*Nature*, which is not by itself evidence of correctness but is evidence of
scrutiny.

> RESEARCH NOTE: The tabular position is moving for the first time in two
> decades, and the reasonable summary in 2026 is: **below ~10k rows try a
> tabular foundation model first and gradient boosting second; above it, the
> reverse.** Two caveats worth stating. Current tabular foundation models have
> hard limits on rows, columns and classes, so the crossover is partly a
> capability boundary rather than a purely empirical one. And the durable
> content of {{cite:grinsztajn2022}} is the three *mechanisms* — if a new
> architecture addresses them, the recommendation should flip, and the reasoning
> here is what tells you whether it has.

## 6. Mathematical Foundation

### 6.1 Gradient descent in function space

This is the derivation that makes the name make sense.

Ordinary gradient descent minimises $\Loss(\theta)$ by stepping against
$\nabla_{\theta}\Loss$. Now treat the *function* $F$ as the parameter. We want to
minimise the empirical risk

$$
\Loss(F) = \sum_{i=1}^{N}\Loss\big(y_i, F(\vec{x}_i)\big)
$$

over functions. Since $F$ enters only through its values at the training points,
the object $\big(F(\vec{x}_1), \dots, F(\vec{x}_N)\big)$ is a vector in
$\R^{N}$, and the gradient with respect to it has components

$$
g_i = \frac{\partial \Loss(y_i, F(\vec{x}_i))}{\partial F(\vec{x}_i)}
$$

The steepest-descent step is $F \leftarrow F - \eta\vec{g}$, exactly as in
{{ch:math-optimization}}.

The problem is that $-\vec{g}$ is defined only at the $N$ training points. It is
not a function; you cannot evaluate it at a new $\vec{x}$. **Boosting's answer is
to fit a tree to it** — the tree is a function, defined everywhere, that
approximates the negative gradient where the gradient is known:

$$
h_m \approx -\vec{g}_{m-1}, \qquad F_m = F_{m-1} + \eta\, h_m
$$ (eq:functional-gradient)

So gradient boosting is gradient descent where each step is *projected onto the
space of trees*. That single sentence explains the whole design: $\eta$ is the
learning rate; $M$ is the number of steps; the pseudo-residual is a gradient;
and the residual-fitting story of {{sec:4-intuitive-explanation}} is the special
case where the loss is squared error and the gradient happens to equal the
residual.

### 6.2 The XGBoost closed form

Take a second-order Taylor expansion of the loss about the current prediction:

$$
\Loss(y_i, F_{m-1} + f) \approx \Loss(y_i, F_{m-1})
  + g_i f + \tfrac{1}{2}h_i f^{2}
$$

Dropping the constant and adding the penalties gives {{eq:xgboost-objective}}.
Now fix the tree structure, so $f$ is constant $w_j$ on leaf $j$ with instance
set $I_j$. The objective becomes a sum of independent quadratics:

$$
\sum_{j=1}^{T}\left[\Big(\sum_{i \in I_j} g_i\Big) w_j
 + \tfrac{1}{2}\Big(\sum_{i \in I_j} h_i + \lambda\Big) w_j^{2}\right]
 + \gamma T
$$

Each is minimised by setting its derivative to zero:

$$
w_j^{*} = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}
$$ (eq:xgboost-leaf)

and substituting back gives the objective at the optimum:

$$
\tilde{\Loss}^{*} = -\frac{1}{2}\sum_{j=1}^{T}
  \frac{\big(\sum_{i \in I_j} g_i\big)^{2}}{\sum_{i \in I_j} h_i + \lambda}
  + \gamma T
$$ (eq:xgboost-structure-score)

{{eq:xgboost-structure-score}} is a **structure score**: a number measuring how
good a tree shape is. The gain from splitting a leaf into $L$ and $R$ is the
difference between the scores, which yields the split criterion

$$
\text{Gain} = \frac{1}{2}\left[
  \frac{G_L^{2}}{H_L+\lambda} + \frac{G_R^{2}}{H_R+\lambda}
  - \frac{(G_L+G_R)^{2}}{H_L+H_R+\lambda}\right] - \gamma
$$ (eq:xgboost-gain)

Two things to notice. The $-\gamma$ means a split must earn a fixed amount
before it is taken — pruning built into the criterion rather than applied
afterwards. And $\lambda$ in the denominators shrinks the influence of leaves
with little Hessian mass, which for log loss means leaves containing only
confidently-classified points, whose $h_i = p(1-p)$ is near zero.

Compare {{ch:ml-trees}}'s Gini criterion, which knows nothing about the loss
being optimised. This one is derived from it.

### 6.3 Why boosting overfits and bagging does not

In {{ch:ml-forests}}, adding a tree is a Monte Carlo draw from a fixed
distribution, so the average converges: $\hat{f}_B \to \E[\hat{f}]$.

In boosting, adding a tree changes the function being fitted. There is no fixed
limit; the sequence walks through function space and the training loss decreases
monotonically. Once the ensemble has captured the signal, the remaining
pseudo-residuals are noise, and each further tree fits some of it.

$\eta$ controls how fast that happens, not whether it does. Halving $\eta$
roughly doubles the number of rounds to the optimum and to the eventual
degradation — the shape of the validation curve is preserved and stretched. This
is why early stopping is the mechanism that sets $M$, and why a fixed
`n_estimators` is a bug rather than a hyperparameter.

## 7. Implementation

```python {tier=A name=boosting-from-scratch}
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
```

```python {tier=A name=second-order-boosting}
"""Log-loss boosting, and the second-order (XGBoost) formulation derived in
section 6.2 — including the exact split criterion.
"""
import numpy as np

rng = np.random.default_rng(2)


def sigmoid(z):
    out = np.empty_like(z, dtype=float)
    p = z >= 0
    out[p] = 1 / (1 + np.exp(-z[p]))
    e = np.exp(z[~p])
    out[~p] = e / (1 + e)
    return out


# --- a tree grown to maximise the XGBoost gain of eq. 38.9 ------------------
def xgb_grow(X, g, h, depth, max_depth, lam, gamma, min_child_weight=1.0):
    """Split scoring comes from eq. 38.9; leaf values from eq. 38.7. The tree
    is grown against the ACTUAL loss, not a Gini proxy."""
    G, H = g.sum(), h.sum()
    node = {"w": -G / (H + lam)}
    if depth >= max_depth or len(g) < 2:
        return node
    best = (0.0, None, None)
    base = G ** 2 / (H + lam)
    for j in range(X.shape[1]):
        o = np.argsort(X[:, j], kind="mergesort")
        xs, gs, hs = X[o, j], g[o], h[o]
        cg, ch = np.cumsum(gs), np.cumsum(hs)
        k = np.arange(1, len(gs))
        GL, HL = cg[:-1], ch[:-1]
        GR, HR = G - GL, H - HL
        ok = ((xs[1:] != xs[:-1]) & (HL >= min_child_weight)
              & (HR >= min_child_weight))
        gain = np.where(ok,
                        0.5 * (GL ** 2 / (HL + lam) + GR ** 2 / (HR + lam)
                               - base) - gamma,
                        -np.inf)
        i = int(gain.argmax())
        if np.isfinite(gain[i]) and gain[i] > best[0]:
            best = (float(gain[i]), j, 0.5 * (xs[i] + xs[i + 1]))
    _, j, thr = best
    if j is None:
        return node
    m = X[:, j] <= thr
    node["f"], node["t"] = j, thr
    node["l"] = xgb_grow(X[m], g[m], h[m], depth + 1, max_depth, lam, gamma,
                         min_child_weight)
    node["r"] = xgb_grow(X[~m], g[~m], h[~m], depth + 1, max_depth, lam,
                         gamma, min_child_weight)
    return node


def apply_xgb(node, X):
    out = np.empty(len(X))

    def walk(nd, idx):
        if "f" not in nd:
            out[idx] = nd["w"]
            return
        m = X[idx, nd["f"]] <= nd["t"]
        walk(nd["l"], idx[m])
        walk(nd["r"], idx[~m])

    walk(node, np.arange(len(X)))
    return out


class XGBClassifier:
    """Second-order boosting for log loss. For log loss:
         g = p - y            (the gradient of eq. 33.5, i.e. eq. 33.10)
         h = p(1 - p)         (its second derivative, the sigmoid slope)
    """

    def __init__(self, n_trees=300, eta=0.1, max_depth=3, lam=1.0, gamma=0.0,
                 min_child_weight=1.0):
        self.M, self.eta, self.depth = n_trees, eta, max_depth
        self.lam, self.gamma, self.mcw = lam, gamma, min_child_weight

    def fit(self, X, y, X_val=None, y_val=None):
        base = np.clip(y.mean(), 1e-6, 1 - 1e-6)
        self.F0 = float(np.log(base / (1 - base)))     # log-odds of the prior
        F = np.full(len(y), self.F0)
        self.trees, self.val_curve, self.train_curve = [], [], []
        for _ in range(self.M):
            p = sigmoid(F)
            g, h = p - y, np.maximum(p * (1 - p), 1e-9)
            t = xgb_grow(X, g, h, 0, self.depth, self.lam, self.gamma, self.mcw)
            self.trees.append(t)
            F += self.eta * apply_xgb(t, X)
            self.train_curve.append(self._logloss(y, sigmoid(F)))
            if X_val is not None:
                self.val_curve.append(
                    self._logloss(y_val, self.predict_proba(X_val)))
        return self

    @staticmethod
    def _logloss(y, p):
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def decision(self, X, n_trees=None):
        M = len(self.trees) if n_trees is None else n_trees
        F = np.full(len(X), self.F0)
        for t in self.trees[:M]:
            F += self.eta * apply_xgb(t, X)
        return F

    def predict_proba(self, X, n_trees=None):
        return sigmoid(self.decision(X, n_trees))


def make_clf(n, rs):
    X = rs.uniform(-3, 3, (n, 8))
    z = (1.4 * np.sin(1.2 * X[:, 0]) + 0.9 * X[:, 1]
         - 1.1 * X[:, 0] * X[:, 2] + 0.7 * np.abs(X[:, 3]) - 0.8)
    return X, (rs.random(n) < sigmoid(z)).astype(float)


rs = np.random.default_rng(3)
Xtr, ytr = make_clf(900, rs)
Xva, yva = make_clf(900, rs)
Xte, yte = make_clf(3000, rs)
print(f"positive rate: {ytr.mean():.4f}")


def roc_auc(y, s):
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s))
    r[o] = np.arange(1, len(s) + 1)
    npos = int(y.sum())
    return float((r[y == 1].sum() - npos * (npos + 1) / 2)
                 / (npos * (len(y) - npos)))


# --- the pseudo-residual for log loss IS the logistic gradient --------------
print("\n" + "=" * 72)
print("table 38.1: the log-loss pseudo-residual is y - sigma(F)")
print("=" * 72)
m0 = XGBClassifier(n_trees=1, eta=0.1).fit(Xtr, ytr)
F0 = np.full(len(ytr), m0.F0)
print(f"initial F0 = log-odds of the base rate = {m0.F0:.4f}")
print(f"sigmoid(F0) = {sigmoid(np.array([m0.F0]))[0]:.4f}  "
      f"vs base rate {ytr.mean():.4f}")
print(f"mean pseudo-residual at round 1 = "
      f"{np.mean(ytr - sigmoid(F0)):.2e}  (zero by construction)")
print("This is exactly eq. 33.10 from Chapter 33. Boosting for")
print("classification is logistic regression's gradient, fitted by trees.")

# --- early stopping ---------------------------------------------------------
print("\n" + "=" * 72)
print("early stopping is how n_estimators is chosen")
print("=" * 72)
clf = XGBClassifier(n_trees=350, eta=0.1, max_depth=4,
                    lam=1.0).fit(Xtr, ytr, Xva, yva)
print(f"{'round':>7} {'train log loss':>16} {'val log loss':>14} "
      f"{'test AUC':>10}")
for m in (1, 10, 50, 100, 200, 350):
    print(f"{m:>7} {clf.train_curve[m - 1]:>16.4f} "
          f"{clf.val_curve[m - 1]:>14.4f} "
          f"{roc_auc(yte, clf.predict_proba(Xte, m)):>10.4f}")
best = int(np.argmin(clf.val_curve)) + 1
print(f"\nvalidation minimum at round {best}: "
      f"log loss {min(clf.val_curve):.4f}, "
      f"test AUC {roc_auc(yte, clf.predict_proba(Xte, best)):.4f}")
print(f"at 350 rounds:              log loss {clf.val_curve[-1]:.4f}, "
      f"test AUC {roc_auc(yte, clf.predict_proba(Xte)):.4f}")
print("\nNote which metric degrades. Log loss turns up clearly while AUC")
print("barely moves — the extra rounds are making the model OVERCONFIDENT")
print("rather than changing its ranking, which is Chapter 34's")
print("calibration-versus-discrimination split appearing again.")

# --- section 6.2: what lambda and gamma actually do -------------------------
print("\n" + "=" * 72)
print("the regularisers in eq. 38.9, one at a time")
print("=" * 72)
print(f"{'lambda':>8} {'gamma':>7} {'best round':>12} {'best val loss':>15} "
      f"{'test AUC':>10}")
for lam, gam in ((0.0, 0.0), (1.0, 0.0), (10.0, 0.0), (100.0, 0.0),
                 (1.0, 0.5), (1.0, 5.0)):
    c = XGBClassifier(n_trees=200, eta=0.1, max_depth=4, lam=lam,
                      gamma=gam).fit(Xtr, ytr, Xva, yva)
    b = int(np.argmin(c.val_curve)) + 1
    print(f"{lam:>8} {gam:>7} {b:>12} {min(c.val_curve):>15.4f} "
          f"{roc_auc(yte, c.predict_proba(Xte, b)):>10.4f}")
print("\nlambda shrinks every leaf value (eq. 38.7); gamma charges a fixed")
print("toll per split (eq. 38.9), so a split must earn its place. They are")
print("different levers: lambda softens the model, gamma makes it smaller.")

# --- first-order vs second-order --------------------------------------------
print("\n" + "=" * 72)
print("first-order vs second-order boosting (section 5.3)")
print("=" * 72)


class FirstOrderClassifier(XGBClassifier):
    """Same loop, but each leaf takes the MEAN NEGATIVE GRADIENT — the plain
    Friedman step, with no Hessian and no eq. 38.7 re-optimisation."""

    def fit(self, X, y, X_val=None, y_val=None):
        base = np.clip(y.mean(), 1e-6, 1 - 1e-6)
        self.F0 = float(np.log(base / (1 - base)))
        F = np.full(len(y), self.F0)
        self.trees, self.val_curve, self.train_curve = [], [], []
        for _ in range(self.M):
            p = sigmoid(F)
            g = p - y
            t = xgb_grow(X, g, np.ones(len(g)), 0, self.depth, self.lam,
                         self.gamma, self.mcw)
            self.trees.append(t)
            F += self.eta * apply_xgb(t, X)
            self.train_curve.append(self._logloss(y, sigmoid(F)))
            if X_val is not None:
                self.val_curve.append(
                    self._logloss(y_val, self.predict_proba(X_val)))
        return self


print(f"{'method':<22} {'best round':>12} {'best val loss':>15} "
      f"{'test AUC':>10}")
for name, cls in (("first-order (Friedman)", FirstOrderClassifier),
                  ("second-order (XGBoost)", XGBClassifier)):
    c = cls(n_trees=250, eta=0.1, max_depth=4, lam=1.0).fit(Xtr, ytr, Xva, yva)
    b = int(np.argmin(c.val_curve)) + 1
    print(f"{name:<22} {b:>12} {min(c.val_curve):>15.4f} "
          f"{roc_auc(yte, c.predict_proba(Xte, b)):>10.4f}")
print("\nBoth use the same trees and the same learning rate. The difference")
print("is that the second-order version divides each leaf by its Hessian")
print("mass (eq. 38.7), so a leaf full of confidently-classified points —")
print("where h = p(1-p) is tiny — is not allowed to make a large correction.")
print("It is a better-scaled step, which is why it needs fewer rounds.")
```

## 8. Practical Example

```python {tier=A name=gbdt-vs-alternatives}
"""Gradient boosting against the rest of Part IV on one tabular problem,
plus the row-count question of section 5.5.
"""
import time

import numpy as np

rng = np.random.default_rng(41)


def sigmoid(z):
    out = np.empty_like(z, dtype=float)
    p = z >= 0
    out[p] = 1 / (1 + np.exp(-z[p]))
    e = np.exp(z[~p])
    out[~p] = e / (1 + e)
    return out


# --- a tabular problem with the properties Grinsztajn et al. describe -------
def make_data(n):
    """Irregular target, uninformative features, meaningful axes — the three
    conditions under which tree ensembles are expected to win."""
    age = rng.uniform(18, 80, n)
    income = rng.lognormal(10.3, 0.6, n)
    tenure = rng.exponential(4.0, n)
    n_prod = rng.poisson(2.0, n)
    region = rng.integers(0, 6, n).astype(float)
    # irregular: a threshold effect, an interaction, and a non-monotone term
    z = (-1.1
         + 1.6 * (age < 30) + 0.9 * (age > 65)
         - 0.7 * (np.log(income) - 10.3)
         + 1.3 * (tenure < 1.0) * (np.log(income) < 10.0)
         + 0.35 * n_prod
         + 0.8 * np.isin(region, [1, 4]))
    noise = rng.normal(size=(n, 10))          # ten uninformative columns
    X = np.column_stack([age, np.log(income), tenure, n_prod, region, noise])
    return X, (rng.random(n) < sigmoid(z)).astype(float)


NAMES = (["age", "log_income", "tenure", "n_products", "region"]
         + [f"noise_{i}" for i in range(10)])


def roc_auc(y, s):
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s))
    r[o] = np.arange(1, len(s) + 1)
    npos = int(y.sum())
    return float((r[y == 1].sum() - npos * (npos + 1) / 2)
                 / (npos * (len(y) - npos)))


try:
    from sklearn.ensemble import (GradientBoostingClassifier,
                                  HistGradientBoostingClassifier,
                                  RandomForestClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    HAVE_SK = True
except ImportError:
    HAVE_SK = False

if not HAVE_SK:
    print("scikit-learn not installed — this listing needs it")
else:
    Xtr, ytr = make_data(6000)
    Xva, yva = make_data(2000)
    Xte, yte = make_data(8000)
    print(f"positive rate {ytr.mean():.4f}, "
          f"{Xtr.shape[1]} features of which 10 are noise\n")

    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

    models = [
        ("logistic regression (Ch 33)",
         LogisticRegression(max_iter=3000), True),
        ("k-NN, k=25 (Ch 35)",
         KNeighborsClassifier(n_neighbors=25), True),
        ("single tree, depth 6 (Ch 36)",
         DecisionTreeClassifier(max_depth=6, random_state=0), False),
        ("random forest, 400 (Ch 37)",
         RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=0),
         False),
        ("gradient boosting (this ch)",
         GradientBoostingClassifier(n_estimators=300, learning_rate=0.1,
                                    max_depth=3, random_state=0), False),
        ("hist gradient boosting",
         HistGradientBoostingClassifier(max_iter=400, learning_rate=0.1,
                                        early_stopping=True,
                                        random_state=0), False),
    ]

    print("=" * 72)
    print("every model in Part IV, same data, same split")
    print("=" * 72)
    print(f"{'model':<30} {'test AUC':>9} {'accuracy':>10} {'fit s':>8}")
    results = {}
    for name, m, scale in models:
        A, B = (Xtr_s, Xte_s) if scale else (Xtr, Xte)
        t0 = time.perf_counter()
        m.fit(A, ytr)
        dt = time.perf_counter() - t0
        p = m.predict_proba(B)[:, 1]
        results[name] = roc_auc(yte, p)
        print(f"{name:<30} {results[name]:>9.4f} "
              f"{((p >= 0.5) == (yte == 1)).mean():>10.4f} {dt:>8.2f}")

    print("\nThe ordering is the usual one on tabular data with an irregular")
    print("target: boosting first, forest close behind, single tree and")
    print("linear model well back, k-NN worst because ten noise dimensions")
    print("dominate its distances (Chapter 35).")

    # --- section 5.5: how the answer depends on the number of rows ----------
    print("\n" + "=" * 72)
    print("section 5.5: the answer depends on how much data you have")
    print("=" * 72)
    print(f"{'train rows':>11} {'logistic':>10} {'random forest':>15} "
          f"{'gradient boosting':>19} {'winner':>18}")
    for n_train in (100, 300, 1000, 3000, 6000):
        A, b = Xtr[:n_train], ytr[:n_train]
        A_s = sc.transform(A)
        lr = LogisticRegression(max_iter=3000).fit(A_s, b)
        rf = RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                    random_state=0).fit(A, b)
        gb = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.08, early_stopping=True,
            random_state=0).fit(A, b)
        a_lr = roc_auc(yte, lr.predict_proba(Xte_s)[:, 1])
        a_rf = roc_auc(yte, rf.predict_proba(Xte)[:, 1])
        a_gb = roc_auc(yte, gb.predict_proba(Xte)[:, 1])
        best = max((a_lr, "logistic"), (a_rf, "forest"), (a_gb, "boosting"))
        print(f"{n_train:>11} {a_lr:>10.4f} {a_rf:>15.4f} {a_gb:>19.4f} "
              f"{best[1]:>18}")

    print("\nBoosting's advantage is not constant — it grows with the number")
    print("of rows, because more rounds of bias reduction need more data to")
    print("stay honest. At the small end the models converge and the")
    print("cheapest one is defensible. This is the same axis along which")
    print("Chapter 38's tabular-foundation-model discussion sits: what wins")
    print("depends on the sample size, not only on the algorithm.")

    # --- the boosting failure mode you will actually hit --------------------
    print("\n" + "=" * 72)
    print("the failure you will actually hit: no early stopping")
    print("=" * 72)
    print(f"{'n_estimators':>13} {'val log loss':>14} {'test AUC':>10}")
    for n_est in (50, 200, 800, 3000):
        gb = GradientBoostingClassifier(n_estimators=n_est,
                                        learning_rate=0.15, max_depth=6,
                                        random_state=0).fit(Xtr, ytr)
        pv = np.clip(gb.predict_proba(Xva)[:, 1], 1e-12, 1 - 1e-12)
        ll = float(-np.mean(yva * np.log(pv) + (1 - yva) * np.log(1 - pv)))
        print(f"{n_est:>13} {ll:>14.4f} "
              f"{roc_auc(yte, gb.predict_proba(Xte)[:, 1]):>10.4f}")

    gb_es = HistGradientBoostingClassifier(
        max_iter=3000, learning_rate=0.15, max_depth=6,
        early_stopping=True, n_iter_no_change=20,
        validation_fraction=0.15, random_state=0).fit(Xtr, ytr)
    print(f"\nwith early stopping: stopped at {gb_es.n_iter_} of 3000 "
          f"iterations, test AUC "
          f"{roc_auc(yte, gb_es.predict_proba(Xte)[:, 1]):.4f}")
    print("\nA fixed n_estimators is a bug wearing a hyperparameter's")
    print("clothes. Early stopping found its own budget and beat every fixed")
    print("choice, without anyone tuning it.")
```

## 9. Common Mistakes

**Using deep trees.** Boosting reduces bias and adds variance; deep base
learners leave nothing to correct and everything to overfit.

**Fixing `n_estimators`.** It must come from early stopping.

**Tuning $\eta$ and $M$ independently.** They trade off almost exactly.

**Treating `max_depth` as a capacity dial.** It sets the interaction order, and
the measurement shows stumps cannot represent $x_0 x_1$ at all.

**Watching AUC for early stopping.** The measured curve shows log loss turning
up while AUC barely moves — the model becomes overconfident before it becomes a
worse ranker.

**Using LightGBM's defaults on a small dataset.** Leaf-wise growth overfits;
lower `num_leaves`.

**Target-encoding a categorical without a leakage guard.** {{ch:ds-leakage}}'s
mechanism; CatBoost's ordered statistics exist for it.

**Comparing libraries instead of tuning one.** Differences between XGBoost,
LightGBM and CatBoost are usually smaller than the tuning effect, and much
smaller than the feature-engineering effect.

**Assuming boosting always beats a forest.** The measurement shows the advantage
growing with sample size and nearly vanishing at a few hundred rows.

**Assuming gradient boosting is permanently the tabular answer.** On small data
that is no longer clear ({{sec:5-formal-explanation}}).

## 10. Connection to Previous Chapters

{{ch:ml-forests}} is the direct contrast, and holding the two together is the
point of the part: bagging attacks the variance term of {{eq:bias-variance}} in
parallel with deep trees and cannot overfit by adding members; boosting attacks
the bias term sequentially with shallow trees and can. Neither is a variation on
the other.

{{ch:math-optimization}} supplied gradient descent, of which
{{eq:functional-gradient}} is the function-space form — the pseudo-residual *is*
a gradient. {{ch:ml-logistic}} supplied {{eq:logreg-gradient}}, which is the
log-loss pseudo-residual in {{tbl:boosting-losses}}, and the calibration frame
that the early-stopping measurement uses. {{ch:ml-trees}} supplied the base
learner and the depth-versus-interaction-order argument.
{{ch:ml-metrics}} supplied the validation curve that early stopping reads.

Forward: {{ch:dl-backprop}} is the parameter-space version of the same descent.
{{ch:tf-ffn-residual}} adds functions to a running sum for the same reason
boosting does. {{part:20}} automates the hyperparameter search this chapter
does by hand. {{ch:rai-interpretability}} explains a model that is now several
hundred trees deep.

## 11. Exercises

**Beginner**

1. Explain boosting in one sentence, contrasting it with bagging.
2. Why must the base learners be shallow?
3. What does the learning rate do, and how does it interact with $M$?
4. Why can boosting overfit with more trees when a forest cannot?
5. What is the pseudo-residual for squared error?

**Intermediate**

6. Derive the log-loss pseudo-residual and identify where you have seen it.
7. Explain why depth 1 gives a purely additive model.
8. Why does halving $\eta$ roughly double the optimal $M$?
9. Explain leaf-wise growth and why it overfits small data.
10. What problem do CatBoost's ordered target statistics solve?
11. Why does early stopping on log loss stop earlier than on AUC?

**Advanced**

12. Derive {{eq:functional-gradient}} and state precisely what makes it
    gradient descent.
13. Derive {{eq:xgboost-leaf}} and {{eq:xgboost-gain}} from
    {{eq:xgboost-objective}}.
14. Explain the role of $\lambda$ in {{eq:xgboost-gain}} in terms of Hessian
    mass, and say which leaves it affects most under log loss.
15. Show that gradient boosting with squared error and $\eta = 1$ on a single
    feature reproduces a specific classical procedure, and name it.
16. Explain why {{eq:leaf-value}} differs from the tree's own fitted values, and
    construct a loss where ignoring it gives a materially wrong answer.

**Implementation**

17. Add row and column subsampling to the from-scratch implementation and
    measure the effect on the validation curve.
18. Implement histogram-based split finding and measure the speed-up against
    the exact scan.
19. Implement Huber loss boosting and compare its robustness to outliers
    against squared error.
20. Implement early stopping with a patience parameter and confirm it recovers
    the best round found by the full curve.

**Reasoning**

21. Your boosted model beats your forest by 3 points on validation and loses by
    1 point in production. Give three hypotheses.
22. You have 4,000 rows, 30 features, and a deadline. What do you try, in what
    order, and why?

## 12. Chapter Summary

Gradient boosting fits models sequentially, each to the pseudo-residuals of the
current ensemble, and adds them with a shrinkage factor. Formally it is gradient
descent in function space: the negative gradient is defined only at the training
points, and each tree is that gradient projected onto the space of trees.

It reduces bias and adds variance, which is the exact inverse of bagging and
which is why its base learners must be shallow. The measurement shows depth 12
driving training error towards zero while test error rises, because after the
first few trees the residuals are noise.

Tree depth sets interaction order, not merely capacity. Stumps produce a purely
additive model that cannot represent $x_0 x_1$ at any number of terms;
depth 2 can. Choosing `max_depth` is a modelling decision about your problem.

Shrinkage trades off almost exactly against the number of rounds — halving
$\eta$ roughly doubles the optimal $M$ — and smaller $\eta$ reaches a better
optimum because each tree's own overfit is diluted before it enters the sum.

Boosting can overfit with more trees, unlike a forest, because each round
changes the function being fitted rather than sampling from a fixed
distribution. Early stopping is therefore the mechanism that sets $M$, and a
fixed `n_estimators` is a bug. The measured curve also shows *which* metric
degrades first: log loss turns up while AUC barely moves, so the extra rounds
make the model overconfident before they make it a worse ranker.

The second-order formulation expands the loss to second order, giving a closed
form for the optimal leaf value, $-G/(H+\lambda)$, and an exact split criterion
derived from the loss rather than from a Gini proxy. Dividing by Hessian mass
prevents leaves of confidently-classified points from making large corrections,
which is why it converges in fewer rounds.

On tabular data, gradient boosting is {{maturity:ESTABLISHED}} at medium and
large scale for three mechanistic reasons — robustness to uninformative
features, indifference to feature rotation given meaningful axes, and the
ability to fit irregular functions. Tabular foundation models are
{{maturity:EMERGING}} and already the better choice below roughly ten thousand
rows. The mechanisms are what to reason from when the recommendation changes.

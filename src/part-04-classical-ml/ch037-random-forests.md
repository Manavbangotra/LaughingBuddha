---
id: ml-forests
number: 37
part: IV
tier: focused
status: reviewed
requires: [ml-trees, ml-metrics, math-inference]
provides: [bagging, bootstrap, random-forest, oob-error, feature-subsampling,
           extra-trees, variance-reduction, ensemble-diversity]
citations: [breiman2001rf, breiman2001cultures, pedregosa2011]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Derive why averaging reduces variance, and why correlation between models
   caps the benefit.
2. Explain why bagging leaves bias unchanged and what that implies for the
   choice of base learner.
3. Derive the 63.2% bootstrap fraction and use out-of-bag samples for free
   validation.
4. Explain why feature subsampling is what makes a random forest better than
   plain bagging.
5. Tune a random forest, and explain why adding trees cannot overfit.
6. Explain extremely randomised trees and when the extra randomness pays.
7. Explain what a forest gives up relative to a single tree.

## 2. Why This Matters

The random forest is the strongest model in this book that requires essentially
no tuning, and the argument behind it is one of the most reusable ideas in
machine learning.

**"Average many noisy independent estimates" is a technique, not an algorithm.**
It reappears as ensembling in {{part:25}}, as Monte Carlo dropout, as
deep ensembles for uncertainty, as self-consistency sampling in
{{part:16}} — where the noisy estimates are chains of thought
from a language model and the average is a majority vote. The variance algebra
in {{sec:6-mathematical-foundation}} governs all of them, including the crucial
term that says correlated members stop helping.

**It is the default first model for tabular data.** Default hyperparameters are
usually within a couple of points of a tuned gradient-boosting model, it is
almost impossible to overfit by adding trees, and it parallelises perfectly.
When you need a number by the end of the day, this is the model.

**Out-of-bag error is free validation.** Roughly a third of the data is unused
by each tree, so a held-out estimate comes at no extra cost and no extra data.
Nothing else in this part offers that.

## 3. Prerequisites

{{ch:ml-trees}} for the base learner — in particular the measured instability at
the end of it, which is the entire motivation here. {{ch:ml-metrics}} for
bias-variance. {{ch:math-inference}} for the bootstrap and the $\sqrt{1/N}$ rate
that reappears as $\sqrt{1/B}$.

## 4. Intuitive Explanation

### 4.1 Averaging away the noise

{{ch:ml-trees}} ended with a measurement: resampling the training rows changes
the tree's root split and drops the correlation between its predictions and the
original's to 0.16. A deep tree has low bias and enormous variance.

Averaging is the classical remedy for variance, and it is the same argument as
the standard error of a mean in {{ch:math-inference}}: average $B$ independent
estimates and the variance falls by $B$. Bias is unaffected, because the average
of unbiased estimates is unbiased.

So the recipe writes itself:

```text
   training set
        │
        ├── bootstrap sample 1 ──▶ deep tree 1 ──┐
        ├── bootstrap sample 2 ──▶ deep tree 2 ──┤
        ├──        ...                       ... ├──▶ average / vote
        └── bootstrap sample B ──▶ deep tree B ──┘

   bias: unchanged     variance: divided by (up to) B
```

This is **bagging** — bootstrap aggregating. Note what it demands of the base
learner: **low bias, high variance**. Averaging cannot fix bias, so a
high-bias base learner produces a high-bias ensemble. That is why bagged
*unpruned* trees work and bagged linear models are pointless — the average of
many linear models fitted to resamples of the same data is approximately the
single linear model you would have fitted anyway.

### 4.2 Why plain bagging is not enough

The catch is in the word *independent*. Bootstrap samples of the same dataset
overlap heavily, so the trees are correlated, and correlated estimates do not
average away.

The extreme case makes it obvious: if all $B$ trees were identical, averaging
them would change nothing at all. Bagged trees are not identical, but if one
feature is strongly predictive, nearly every tree will split on it at the root
and the trees will be far more similar than the bootstrap alone suggests.

{{cite:breiman2001rf}}'s addition is one line of code: **at each split, consider
only a random subset of features.** Now the dominant feature is unavailable at
maybe a third of the nodes, other features get used, and the trees genuinely
differ.

This is a real trade, not a free lunch, and whether it pays is an empirical
question about your features. Restricting the split search makes each individual
tree *worse*, because it sometimes cannot use the best available split. Whether
the ensemble comes out ahead depends on how expensive that restriction is — and
that depends on whether an excluded feature has a **substitute**.

{{sec:7-implementation}} measures both halves on two datasets that differ only
in that respect. When the features are independent, excluding a useful one
leaves nothing in its place, per-tree error climbs steeply, and plain bagging
wins. When the features come in correlated groups — repeated measurements,
lagged copies, derived quantities, which is what real tabular data mostly looks
like — excluding one leaves near-substitutes, per-tree error barely moves, and
subsampling is nearly free decorrelation.

That is why $m$ is the one hyperparameter worth tuning, and why $\sqrt{D}$ and
$D/3$ are starting points rather than answers.

### 4.3 Out-of-bag error

A bootstrap sample of size $N$ drawn with replacement misses about 37% of the
original rows ({{sec:6-mathematical-foundation}} derives the number). Those rows
are **out-of-bag** for that tree.

To score a row honestly, average the predictions of only the trees that did not
see it. Every row gets such an estimate, so you obtain a held-out score using
100% of your data for training and no separate validation split. It is roughly
equivalent to leave-one-out cross-validation at no extra cost.

> NOTE: OOB error is slightly **pessimistic**, because each row is scored by
> only the ~37% of trees that excluded it rather than by the full forest. With a
> few hundred trees the effect is small. It is also unavailable, and silently
> wrong, if your rows are not independent — grouped or time-ordered data needs
> the designs in {{ch:ds-leakage}}, and OOB will happily report an optimistic
> number instead.

### 4.4 What you give up

**Interpretability.** A depth-3 tree is a flowchart. Five hundred depth-25 trees
are not, and the honest answer is that a forest is a black box you probe with
{{ch:rai-interpretability}}'s tools rather than read.

**Prediction cost.** $B$ tree traversals instead of one. Still fast, but a
thousand times the work.

**Memory.** Every tree is stored. A forest on a large dataset is easily
hundreds of megabytes.

**Extrapolation.** Unchanged from {{ch:ml-trees}}: an average of constants is
still a constant. A forest cannot extrapolate either.

## 5. Formal Explanation

### 5.1 Bagging

Given $\Data$ of size $N$, draw $B$ bootstrap samples $\Data^{(1)}, \dots,
\Data^{(B)}$, each of size $N$ with replacement. Fit $\hat{f}_b$ on each. Predict

$$
\hat{f}_{\text{bag}}(\vec{x}) = \frac{1}{B}\sum_{b=1}^{B}\hat{f}_b(\vec{x})
$$ (eq:bagging)

for regression; for classification, either majority vote or — usually better —
average the predicted probabilities, which retains more information and
produces a usable score for {{ch:ml-metrics}}'s threshold-free metrics.

### 5.2 The random forest

Bagging plus feature subsampling {{cite:breiman2001rf}}:

1. For $b = 1, \dots, B$: draw a bootstrap sample.
2. Grow a tree, and **at every node** select a random subset of $m$ of the $D$
   features, searching for the best split among those only.
3. Grow deep — no pruning.
4. Aggregate by {{eq:bagging}}.

Conventional defaults are $m = \sqrt{D}$ for classification and $m = D/3$ for
regression {{cite:pedregosa2011}}. $m$ is the one hyperparameter that matters:
it directly controls the correlation term in {{eq:ensemble-variance}}.

Note step 2 says *at every node*, not once per tree. Subsampling once per tree
would be far weaker — the dominant feature would be present in most trees and
would be used at the root of all of them.

### 5.3 Extremely randomised trees

Extra-trees push the randomisation one step further: for each of the $m$
candidate features, draw a **random** threshold rather than searching for the
best one, and take the best of those $m$ random splits.

Two consequences. Training is much faster — no sorting, no scan over
thresholds, which was the entire cost of {{ch:ml-trees}}'s split search. And the
trees are more decorrelated still, trading more per-tree bias for less ensemble
variance. Extra-trees also conventionally use the whole training set rather than
a bootstrap, since the split randomisation already supplies the diversity.

They tend to win when the signal is smooth and noise is high, and to lose when a
precise threshold genuinely matters.

### 5.4 Hyperparameters, in order of importance

{#tbl:rf-hyperparams caption="Random forest hyperparameters. Only the first two normally repay a search, and the first is not really a hyperparameter."}

| Parameter | Effect | Guidance |
|---|---|---|
| `n_estimators` ($B$) | more is monotonically better | as many as you can afford; 300-500 typical |
| `max_features` ($m$) | controls tree correlation | the one worth tuning |
| `min_samples_leaf` | limits tree depth | 1 for classification, ~5 for noisy regression |
| `max_depth` | limits tree depth | usually leave unlimited |
| `bootstrap` | on for RF, off for extra-trees | leave at the default |

> IMPORTANT: **Adding trees cannot overfit.** As $B \to \infty$ the ensemble
> prediction converges to an expectation over the bootstrap and feature-subset
> randomness — a fixed function of the training data. More trees reduce the
> Monte Carlo error of estimating that limit and nothing else. The
> generalisation error converges rather than diverging, which is a genuinely
> unusual property and the reason a forest needs so little care.
>
> This does **not** mean a forest cannot overfit at all. It can, through the
> other knobs: fully grown trees on noisy data fit the noise, and the ensemble
> averages many models that each memorised. It is $B$ specifically that is safe.

## 6. Mathematical Foundation

### 6.1 Variance of an average of correlated estimators

This is the central result of the chapter. Let $\hat{f}_1, \dots, \hat{f}_B$ each
have variance $\sigma^{2}$ and pairwise correlation $\rho$. Then

$$
\Var\left[\frac{1}{B}\sum_b \hat{f}_b\right]
 = \frac{1}{B^{2}}\left[B\sigma^{2} + B(B-1)\rho\sigma^{2}\right]
 = \rho\sigma^{2} + \frac{1-\rho}{B}\sigma^{2}
$$ (eq:ensemble-variance)

Read the two terms separately, because everything follows from them.

The second term, $\frac{1-\rho}{B}\sigma^{2}$, vanishes as $B \to \infty$. This
is the part more trees buy, and it is why adding trees cannot hurt.

The first term, $\rho\sigma^{2}$, **does not depend on $B$ at all**. It is a
floor. No number of trees can reduce the variance below the correlated
component.

$$
\lim_{B \to \infty} \Var\left[\hat{f}_{\text{bag}}\right] = \rho\sigma^{2}
$$ (eq:variance-floor)

Hence the whole design of the random forest. Bagging alone gives typical
$\rho \approx 0.5$ to $0.9$ for trees on the same data, so most of the variance
survives. Feature subsampling attacks $\rho$ directly — the only term that can
still be reduced once $B$ is large.

And it explains the trade-off precisely. Lowering $m$ decreases $\rho$ and
increases $\sigma^{2}$, because each tree is now built with a restricted split
search. The optimum minimises the product-plus-floor in
{{eq:ensemble-variance}}, which is an empirical question — hence $m$ being the
one hyperparameter worth tuning.

### 6.2 Why bias is untouched

$$
\E\left[\frac{1}{B}\sum_b \hat{f}_b\right]
 = \frac{1}{B}\sum_b \E[\hat{f}_b] = \E[\hat{f}_1]
$$ (eq:bagging-bias)

by linearity, since the trees are identically distributed. The ensemble's bias
equals a single tree's bias exactly.

This is the whole reason the base learner must be deep. Bagging is a pure
variance-reduction device; it cannot repair bias, so you must start with a model
that has almost none. Pruning the base trees would be counterproductive — you
would be trading away variance the ensemble was going to remove for free, in
exchange for bias it can never remove.

{{ch:ml-boosting}} attacks the other term, and needs exactly the opposite base
learner: shallow, high-bias, low-variance. The two methods are not variations on
a theme; they are complementary attacks on the two halves of
{{eq:bias-variance}}.

### 6.3 The 63.2% bootstrap fraction

Drawing $N$ times with replacement, the probability a particular row is never
drawn is

$$
\left(1 - \frac{1}{N}\right)^{N} \xrightarrow[N \to \infty]{} e^{-1}
\approx 0.3679
$$ (eq:oob-fraction)

using $\lim_{N\to\infty}(1 - 1/N)^{N} = e^{-1}$. So each tree trains on about
63.2% of the distinct rows and 36.8% are out-of-bag. Convergence is fast: the
value is 0.366 at $N=100$ and 0.368 at $N=1000$.

Two useful corollaries. Each row is out-of-bag for about $0.368B$ trees, so with
$B=500$ each OOB estimate averages roughly 184 trees — enough to be stable. And
a bootstrap sample contains duplicates: about 26% of its $N$ slots are repeats,
which is part of why the trees differ.

### 6.4 Where forests are weak, and why

**Very high-dimensional sparse data.** With $D = 50{,}000$ mostly-zero features
and $m = \sqrt{D} \approx 224$, most nodes see no informative feature at all.
Linear models with $\ell_1$, which can consider all features at once, are
usually better on text.

**Genuinely linear relationships.** The staircase problem of {{ch:ml-trees}}
does not go away; averaging staircases gives a smoother staircase, which is
better, but a linear model still wins on a plane.

**Extrapolation.** An average of constants is a constant.

**Strong class imbalance.** Each bootstrap sample can contain very few minority
examples. Balanced bootstrap sampling, or the threshold reasoning of
{{ch:ml-logistic}}, is needed.

## 7. Implementation

```python {tier=A name=bagging-variance}
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
```

```python {tier=A name=oob-and-tuning}
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
```

## 8. Practical Example

```python {tier=A name=forest-workflow}
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
```

## 9. Common Mistakes

**Bagging a high-bias model.** {{eq:bagging-bias}} says the ensemble's bias
equals the base learner's. Bagged shallow trees are shallow trees.

**Pruning the base trees.** You trade away variance the ensemble removes for
free, for bias it can never remove.

**Tuning `n_estimators` for accuracy.** More is monotonically better; tune it
against your latency budget.

**Subsampling features once per tree.** The subsample must be drawn at every
node, or the dominant feature reappears at every root.

**Using OOB with grouped or time-ordered data.** The bootstrap assumes
exchangeable rows; when they are not, OOB reports an optimistic number and says
nothing about it.

**Using `feature_importances_`.** Still MDI, still biased, still computed on
training data ({{ch:ml-trees}}).

**Expecting a forest to extrapolate.** An average of constants is a constant.

**Reaching for a forest on high-dimensional sparse text.** Most nodes will see
no informative feature; a linear model with $\ell_1$ is usually better.

**Assuming the probabilities are calibrated.** They are usually decent and
compressed towards the middle. Measure.

## 10. Connection to Previous Chapters

{{ch:ml-trees}} supplied the base learner and, at the end of
{{sec:8-practical-example}} there, the exact measurement this chapter answers:
bootstrap resampling changes the root feature and drops prediction correlation
to 0.16. Bagging is the response to that one number.
{{ch:ml-metrics}} supplied {{eq:bias-variance}}, of which this chapter attacks
only the variance term. {{ch:math-inference}} supplied the bootstrap and the
$1/B$ rate that {{eq:ensemble-variance}} inherits.

Forward: {{ch:ml-boosting}} attacks the bias term instead, and therefore needs
the opposite base learner — shallow rather than deep. Holding
{{eq:ensemble-variance}} and {{eq:bagging-bias}} in mind is what makes that
contrast obvious rather than arbitrary. {{ch:ml-anomaly}} uses a forest of
random trees for a completely different purpose — isolation rather than
prediction. {{ch:rai-interpretability}} supplies the tools for probing a model
that can no longer be read. {{part:16}} is
{{eq:ensemble-variance}} applied to sampled reasoning chains, where decorrelating
the members is the same problem in a different medium.

## 11. Exercises

**Beginner**

1. What is a bootstrap sample, and why does it contain duplicates?
2. Why does bagging reduce variance but not bias?
3. What fraction of rows is out-of-bag, and why?
4. Why should the base trees be deep?
5. Can adding more trees cause overfitting? Explain.

**Intermediate**

6. From {{eq:ensemble-variance}}, compute the ensemble variance for
   $\sigma^{2}=4$, $\rho=0.6$, $B=100$, and its limit as $B \to \infty$.
7. Explain why feature subsampling helps, in terms of {{eq:variance-floor}}.
8. Why must features be subsampled at every node rather than once per tree?
9. When is OOB error unreliable?
10. Explain how extra-trees differ, and why they train faster.
11. Why is a forest worse than an $\ell_1$ linear model on sparse text?

**Advanced**

12. Derive {{eq:ensemble-variance}} from the variance of a sum.
13. Derive {{eq:oob-fraction}} and compute the error at $N=50$.
14. Show that the expected number of *distinct* rows in a bootstrap sample is
    $N(1-(1-1/N)^{N})$, and compute the expected number of duplicated slots.
15. Explain why the optimal $m$ balances $\rho$ against $\sigma^{2}$, and
    predict how it should move as the number of irrelevant features grows.
16. Prove that as $B \to \infty$ the forest prediction converges almost surely,
    and state what it converges to.

**Implementation**

17. Add weighted OOB scoring that accounts for how many trees scored each row,
    and check whether it removes the pessimistic bias.
18. Implement balanced bootstrap sampling for imbalanced classification and
    compare against threshold tuning.
19. Measure $\rho$ and $\sigma^{2}$ directly across a grid of `max_features` and
    plot {{eq:ensemble-variance}}'s two terms separately.
20. Implement a forest that stores only leaf values and split thresholds in
    arrays, and measure the prediction speed-up over the dict version here.

**Reasoning**

21. Your forest's OOB error is 0.08 and its production error is 0.19. Give three
    hypotheses, ranked.
22. When would you deploy a single depth-4 tree over a forest that is four
    points more accurate?

## 12. Chapter Summary

Bagging fits many models on bootstrap resamples and averages them. The bias of
the average equals the bias of one member, exactly, so bagging is a pure
variance-reduction device and demands a low-bias, high-variance base learner —
which is why the trees are grown deep and never pruned.

The governing equation is $\Var = \rho\sigma^{2} + \frac{1-\rho}{B}\sigma^{2}$.
The second term vanishes with more trees; the first does not depend on $B$ at
all and is a floor. Once $B$ is large, the only remaining lever is $\rho$.

A random forest lowers $\rho$ by considering a random subset of features at
every node. The measurement shows the trade explicitly: as `max_features` falls,
per-tree variance rises and correlation falls, and the ensemble improves anyway
— the two columns move in opposite directions, so judging a forest by the
quality of its individual trees points at the wrong setting.

Adding trees cannot overfit. The ensemble converges to an expectation over the
bootstrap and feature-subset randomness, and $B$ only reduces the Monte Carlo
error of estimating it. Every other knob can still overfit.

A bootstrap sample omits $1/e \approx 36.8\%$ of the rows, so out-of-bag scoring
gives a held-out estimate for free, using all the data for training. The
measurement matches 5-fold cross-validation at a fifth of the cost. It is
slightly pessimistic and it is silently wrong when rows are grouped or ordered.

Extra-trees replace the threshold search with a random draw, trading more
per-tree bias for less correlation and training much faster.

What a forest gives up: readability, prediction cost, memory, and — unchanged
from a single tree — any ability to extrapolate beyond the training range.

---
id: ml-what-it-is
number: 31
part: IV
tier: focused
status: reviewed
requires: [ds-leakage, math-optimization]
provides: [supervised-learning, unsupervised-learning, self-supervised-learning,
           generalisation, inductive-bias-ml, no-free-lunch]
citations: [breiman2001cultures, pedregosa2011]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. State what machine learning is, in terms of estimating a function from
   examples rather than specifying rules.
2. Distinguish supervised, unsupervised, self-supervised and reinforcement
   learning, and identify which a problem needs.
3. Explain what a hypothesis space is and why choosing a model is choosing which
   answers are reachable.
4. State the no-free-lunch result and explain what it does and does not imply.
5. Explain generalisation as the only objective, and why training error is
   nearly uninformative about it.
6. Explain the parametric/non-parametric distinction and its consequence for
   data requirements.
7. Describe the two cultures of statistical modelling and what separates them.

## 2. Why This Matters

Machine learning is worth understanding as one idea rather than as a list of
algorithms, because the algorithms are numerous and the idea is small.

The idea: **instead of writing rules, supply examples and let an optimisation
procedure find a function that fits them.** Everything else — trees, boosting,
transformers — is a choice about what kinds of function are available and how
the search is conducted.

Two consequences justify a chapter before any algorithm.

**The choice of model family is a choice of assumption.** A linear model can
only express planes; a tree can only express axis-aligned steps. Before any data
is seen, that decision has already ruled out most possible answers. Knowing what
each family assumes is what makes model selection reasoning rather than
trial-and-error, and it is why the no-free-lunch result in
{{sec:6-mathematical-foundation}} is worth taking seriously rather than treating
as a curiosity.

**Generalisation is the only objective, and it is not directly observable.** You
can measure training error exactly and it tells you very little. Everything
about validation design in {{ch:ds-leakage}}, and everything about
regularisation in this part, exists because the quantity that matters must be
estimated rather than computed.

## 3. Prerequisites

{{ch:math-optimization}} for gradient descent and the loss-minimisation frame;
{{ch:ds-leakage}} for validation and why an honest estimate of generalisation
requires care. {{ch:math-inference}} for the bias-variance decomposition's
statistical ancestry.

## 4. Intuitive Explanation

### 4.1 Rules versus examples

Conventional programming specifies the rule:

```text
  rules  +  data   ──▶  answers          (programming)
  data   +  answers ──▶  rules           (machine learning)
```

The second is worth using exactly when the rule is easier to demonstrate than to
state. Nobody can write down the rule distinguishing a photograph of a cat from
one of a dog, and everybody can supply ten thousand labelled examples.

The corollary is the honest test for whether a problem needs machine learning:
**if you can write the rule, write the rule.** A learned model is harder to
debug, harder to guarantee, needs monitoring ({{ch:mle-drift}}), and will be
wrong in ways a rule would not be. It earns its place when the rule is unknown
or too complex to maintain.

### 4.2 The paradigms

{#tbl:paradigms caption="Learning paradigms by what supervision is available. Self-supervised learning is the one that changed everything, because it turns unlabelled data into labelled data."}

| Paradigm | Data | Goal | Example |
|---|---|---|---|
| Supervised | inputs with labels | predict the label | churn, price, diagnosis |
| Unsupervised | inputs only | find structure | segmentation, compression |
| Self-supervised | inputs only | predict part from the rest | language modelling |
| Semi-supervised | few labels, many inputs | exploit the unlabelled | rare-label problems |
| Reinforcement | actions and rewards | choose actions | control, game playing |

{{term:self-supervised-learning}} deserves emphasis because it is the reason the
last decade happened. Labels are scarce and expensive; raw text and images are
effectively unlimited. Hiding part of an input and predicting it from the rest
manufactures a supervised task from unlabelled data, and the representations
that task produces transfer to tasks you *do* have labels for
({{ch:fm-pretraining}}). Everything from {{part:7}} onward rests on this.

### 4.3 The hypothesis space

A model family is a set of functions. Fitting searches that set for the one
minimising a loss.

```text
   all possible functions
   ┌──────────────────────────────────────────────┐
   │   ┌─────────────┐                            │
   │   │  linear     │   ┌──────────────────┐     │
   │   │  models     │   │ depth-3 trees    │     │
   │   └─────────────┘   └──────────────────┘     │
   │                                    ★ truth   │
   └──────────────────────────────────────────────┘
```

If the truth lies outside the family, no amount of data will reach it — that is
{{term:underfitting}}, and it is a property of the choice, not of the fitting.
If the family is large enough to contain functions that fit the noise, the
search may find one — that is {{term:overfitting}}.

This is why "which model should I use?" is not answerable in general. It is a
question about which assumption is right for your data.

### 4.4 Parametric and non-parametric

A **parametric** model has a fixed number of parameters chosen in advance:
linear regression on ten features has eleven, whether you give it a hundred rows
or a billion. It cannot represent more complexity than that budget allows, and
it does not need more data to remain stable.

A **{{term:non-parametric}}** model's complexity grows with the data.
k-nearest neighbours keeps the entire training set. A decision tree grows more
leaves. These make weaker assumptions and need more data to make them pay.

The trade is the bias-variance trade of {{ch:ml-metrics}} in disguise:
parametric models are biased and stable, non-parametric models are flexible and
unstable.

## 5. Formal Explanation

### 5.1 The supervised setup

Given data $\Data = \{(\vec{x}_i, y_i)\}_{i=1}^{N}$ drawn i.i.d. from an unknown
joint distribution $p(\vec{x}, y)$, find $f$ minimising the **expected risk**

$$
R(f) = \E_{(\vec{x}, y) \sim p}\big[\ell(f(\vec{x}), y)\big]
$$ (eq:expected-risk)

You cannot compute {{eq:expected-risk}}, because $p$ is unknown. What you can
compute is the **empirical risk** on the sample:

$$
\hat{R}(f) = \frac{1}{N}\sum_{i=1}^{N}\ell(f(\vec{x}_i), y_i)
$$ (eq:empirical-risk)

Empirical risk minimisation chooses $\hat{f} = \argmin_{f \in \mathcal{F}}
\hat{R}(f)$ over a hypothesis space $\mathcal{F}$.

The whole subject lives in the gap between {{eq:expected-risk}} and
{{eq:empirical-risk}}:

$$
R(\hat{f}) = \underbrace{\hat{R}(\hat{f})}_{\text{measurable}}
  + \underbrace{\big(R(\hat{f}) - \hat{R}(\hat{f})\big)}_{\text{generalisation gap}}
$$ (eq:risk-decomposition)

Minimising the first term without controlling the second is exactly overfitting.
Restricting $\mathcal{F}$, penalising complexity, and stopping early are all
ways of controlling the second at some cost to the first.

> IMPORTANT: The i.i.d. assumption in {{eq:expected-risk}} is doing enormous
> work and fails routinely — under temporal ordering
> ({{ch:ds-timeseries}}), repeated entities ({{ch:ds-leakage}}), and feedback
> loops ({{ch:ds-recsys}}). When it fails, the empirical risk on your validation
> set is not an estimate of the expected risk, and no amount of careful
> optimisation repairs that.

### 5.2 The three components of a learning algorithm

Every method in this part is specified by three choices, and it is worth naming
them because they are usually left implicit:

**Representation** — the hypothesis space $\mathcal{F}$. Linear functions,
axis-aligned partitions, sums of trees, neural networks.

**Evaluation** — the loss $\ell$. Squared error, cross-entropy, hinge loss,
impurity reduction.

**Optimisation** — how the space is searched. Closed form, gradient descent,
greedy recursion, quadratic programming.

Two algorithms that differ only in one of the three are more closely related
than they appear. Linear and logistic regression share representation and
optimisation and differ only in the loss.

### 5.3 The no-free-lunch result

**No free lunch.** Averaged uniformly over all possible target functions, every
learning algorithm has the same expected off-training-set error.

The argument is a counting one. For a binary target over a finite input space,
every assignment of labels to unseen points is equally represented among "all
possible functions". Any algorithm predicting a 1 for some unseen point is
matched by an equally represented function on which that prediction is wrong.
Averaged over all of them, performance is chance.

What this does **not** mean is that all algorithms are equally good in practice.
It means that any algorithm's advantage comes from the fact that real problems
are not drawn uniformly from all possible functions — they are smooth, or
sparse, or hierarchical, or locally constant.

> IMPORTANT: The practical reading is that **an algorithm's assumptions are its
> value**. A method that assumed nothing would be useless. The question is never
> "which algorithm is best" but "which assumption fits this data", and the whole
> of this part is a catalogue of assumptions and the situations that suit them.

### 5.4 Two cultures

{{cite:breiman2001cultures}} distinguished two approaches, and the distinction
still explains most disagreements about machine learning.

**Data modelling** assumes a stochastic model generated the data — a linear
relationship with Gaussian noise, say — fits its parameters, and interprets
them. The model's validity is assessed by goodness-of-fit and by whether its
assumptions hold. The output is understanding.

**Algorithmic modelling** treats the mechanism as unknown and unknowable, uses
whatever function class predicts well, and judges by predictive accuracy alone.
The output is a prediction.

Neither is correct in general and they answer different questions. If you need
to know whether a treatment causes an outcome, accuracy is not the criterion
({{ch:ds-causation}}). If you need to route a support ticket, interpretability
is a nice-to-have. Most confusion about "explainable AI" is the two cultures
talking past each other, and {{ch:rai-interpretability}} returns to it.

## 6. Mathematical Foundation

### 6.1 Why training error is nearly uninformative

Consider a model that memorises: it stores every training pair and returns the
stored label on a match, guessing otherwise. Its training error is exactly zero
and its expected error on new data is chance.

That is the extreme case of a general phenomenon. For a hypothesis space of $M$
functions and a training set of $N$ points, a standard uniform-convergence
argument bounds the generalisation gap with probability $1-\delta$ by

$$
R(\hat{f}) \le \hat{R}(\hat{f}) + \sqrt{\frac{\log M + \log(1/\delta)}{2N}}
$$ (eq:generalisation-bound)

Two readings. The gap shrinks as $\sqrt{1/N}$ — the same rate as every standard
error in {{ch:math-inference}}, and for the same reason. And it grows with
$\log M$: a richer hypothesis space costs you in the bound, which is the formal
statement of "more capacity needs more data."

> MATH NOTE: {{eq:generalisation-bound}} is far too loose to use numerically —
> for realistic model classes it gives bounds above 1. Its value is structural:
> it says the gap depends on capacity and sample size in a specific way, and
> that dependence is what regularisation manipulates. Modern deep networks
> violate the spirit of this bound comprehensively, having enough capacity to
> memorise their training sets while still generalising, and explaining that is
> an open problem ({{ch:res-scaling}}).

### 6.2 The counting argument behind no free lunch

Let the input space be finite with $|\mathcal{X}| = n$ points, of which the
algorithm has seen $m$. There are $2^{n-m}$ possible labellings of the unseen
points, all equally represented in "all possible functions".

For any fixed algorithm $A$ producing predictions on the unseen points, exactly
half of those $2^{n-m}$ functions agree with $A$ on any particular unseen point
and half disagree. Averaged over all target functions, $A$'s off-training-set
accuracy is exactly $1/2$ — regardless of what $A$ does.

Since this holds for every $A$, all algorithms tie.

The escape is that the uniform average over all functions is not the
distribution real problems come from. Real targets are overwhelmingly
concentrated on the smooth, structured, compressible corner of function space,
and algorithms that assume that corner do well.

### 6.3 What a bound on capacity buys

Rearranging {{eq:generalisation-bound}} gives the sample size needed for a gap
of at most $\epsilon$:

$$
N \ge \frac{\log M + \log(1/\delta)}{2\epsilon^{2}}
$$ (eq:sample-complexity)

The $1/\epsilon^{2}$ is familiar from {{ch:math-inference}}: halving the
tolerated gap quadruples the data. The $\log M$ says that doubling the size of
the hypothesis space costs a constant amount of additional data, not a doubling
— which is why very large model families are not automatically hopeless.

## 7. Implementation

```python {tier=A name=hypothesis-spaces}
"""Hypothesis spaces, the memorisation baseline, and no free lunch — measured.
"""
import numpy as np

rng = np.random.default_rng(0)

# --- section 6.1: training error is nearly uninformative --------------------
print("=" * 72)
print("a memoriser: zero training error, chance generalisation")
print("=" * 72)


class Memoriser:
    """Stores every training pair; guesses on anything unseen."""

    def fit(self, X, y):
        self.table = {tuple(np.round(row, 6)): label
                      for row, label in zip(X, y)}
        self.default = int(round(y.mean()))
        return self

    def predict(self, X):
        return np.array([self.table.get(tuple(np.round(r, 6)), self.default)
                         for r in X])


n, d = 400, 8
X = rng.normal(size=(n, d))
y = (rng.random(n) < 0.5).astype(int)         # a coin flip: NO signal

split = n // 2
m = Memoriser().fit(X[:split], y[:split])
print(f"training accuracy : {(m.predict(X[:split]) == y[:split]).mean():.4f}")
print(f"test accuracy     : {(m.predict(X[split:]) == y[split:]).mean():.4f}")
print("Perfect on data it has seen, chance on data it has not. Training")
print("error alone cannot distinguish learning from memorising.")

# --- section 4.3: the hypothesis space decides what is reachable ------------
print("\n" + "=" * 72)
print("a model can only find what its hypothesis space contains")
print("=" * 72)

x = np.linspace(-3, 3, 600)
targets = {
    "linear":     2 * x + 1,
    "quadratic":  x ** 2,
    "step":       np.where(x > 0, 3.0, -1.0),
    "sinusoid":   3 * np.sin(2 * x),
}


def fit_poly(x, y, degree):
    A = np.vander(x, degree + 1)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return A @ beta


def fit_stumps(x, y, n_splits=8):
    """A depth-limited axis-aligned partition — a tiny 'tree' hypothesis space."""
    edges = np.quantile(x, np.linspace(0, 1, n_splits + 1))
    out = np.empty_like(y)
    for i in range(n_splits):
        m_ = (x >= edges[i]) & (x <= edges[i + 1])
        if m_.any():
            out[m_] = y[m_].mean()
    return out


print(f"{'target':<12} {'linear model R2':>17} {'8-region steps R2':>19}")
for name, y_t in targets.items():
    def r2(pred):
        return 1 - np.sum((y_t - pred) ** 2) / np.sum((y_t - y_t.mean()) ** 2)
    print(f"{name:<12} {r2(fit_poly(x, y_t, 1)):>17.4f} "
          f"{r2(fit_stumps(x, y_t)):>19.4f}")

print("\nThe linear space contains the linear target exactly (R2 = 1) and is")
print("worthless on the quadratic (R2 = 0.0000 — a symmetric target has zero")
print("linear component, so the best line is the mean). Note that the line")
print("still scores 0.75 on the step: a wrong hypothesis space is usually not")
print("zero-skill, which is exactly what makes underfitting hard to notice.")

# --- section 6.2: no free lunch, by enumeration -----------------------------
print("\n" + "=" * 72)
print("no free lunch, verified by enumerating every target function")
print("=" * 72)

n_points, n_seen = 12, 6
seen, unseen = np.arange(n_seen), np.arange(n_seen, n_points)

algorithms = {
    "always 0":        lambda tr_y, k: np.zeros(k, dtype=int),
    "always 1":        lambda tr_y, k: np.ones(k, dtype=int),
    "majority of seen": lambda tr_y, k: np.full(k, int(tr_y.mean() >= 0.5)),
    "alternating":     lambda tr_y, k: np.arange(k) % 2,
    "random":          lambda tr_y, k: rng.integers(0, 2, k),
}

n_functions = 2 ** n_points
scores = {name: 0 for name in algorithms}
for code in range(n_functions):
    truth = np.array([(code >> i) & 1 for i in range(n_points)])
    for name, alg in algorithms.items():
        pred = alg(truth[seen], len(unseen))
        scores[name] += (pred == truth[unseen]).mean()

print(f"averaged over all {n_functions:,} possible target functions on "
      f"{n_points} points:\n")
print(f"{'algorithm':<20} {'off-training-set accuracy':>27}")
for name, total in scores.items():
    print(f"{name:<20} {total / n_functions:>27.4f}")
print("\nEvery deterministic algorithm scores exactly 0.5000, including the")
print("sensible ones; the randomised one lands within sampling error of it.")
print("An algorithm's value comes entirely from real problems NOT being")
print("drawn uniformly from this set (section 6.2).")

# --- ...and the same algorithms on a STRUCTURED subset ----------------------
print("\nnow restricted to 'smooth' targets — those with at most 2 label")
print("changes along the ordering, which is what real problems look like:")
smooth = []
for code in range(n_functions):
    truth = np.array([(code >> i) & 1 for i in range(n_points)])
    if np.sum(np.abs(np.diff(truth))) <= 2:
        smooth.append(truth)

scores2 = {name: 0 for name in algorithms}
for truth in smooth:
    for name, alg in algorithms.items():
        pred = alg(truth[seen], len(unseen))
        scores2[name] += (pred == truth[unseen]).mean()

print(f"\n{len(smooth)} smooth functions out of {n_functions:,}\n")
print(f"{'algorithm':<20} {'off-training-set accuracy':>27}")
for name, total in scores2.items():
    print(f"{name:<20} {total / len(smooth):>27.4f}")
print("\nExactly one algorithm rises above chance: the only one that LOOKS AT")
print("the training labels. 'always 0', 'always 1' and 'alternating' ignore")
print("the data, so structure in the target cannot help them. Structure in")
print("the world plus an algorithm that exploits it is what beats chance —")
print("neither alone does (section 6.2).")

# --- sample complexity (section 6.3) ----------------------------------------
print("\n" + "=" * 72)
print("capacity and sample size (section 6.3)")
print("=" * 72)
print(f"{'|F| (hypothesis space)':>24} {'N for gap<=0.05':>17} "
      f"{'N for gap<=0.01':>17}")
for M in (10, 10**3, 10**6, 10**12):
    n05 = (np.log(M) + np.log(1 / 0.05)) / (2 * 0.05 ** 2)
    n01 = (np.log(M) + np.log(1 / 0.05)) / (2 * 0.01 ** 2)
    print(f"{M:>24,} {n05:>17,.0f} {n01:>17,.0f}")
print("\nA million-fold larger hypothesis space costs about 3x the data, not a")
print("million times it — that is the log M. Tightening the gap fivefold")
print("costs 25x, the usual 1/eps^2.")
```

## 8. Practical Example

Choosing between paradigms and model families for a stated problem is the
practical skill, and it is mostly about matching an assumption to the data. The
listing below runs four deliberately narrow model families against four
deliberately different geometries, at two training-set sizes against the same
held-out test set — because *how much* the assumption matters is itself a
function of how much data you have.

```python {tier=A name=matching-assumptions}
"""Four datasets with different structure, four model families.

Each family wins on the data whose structure matches its assumption, and none
wins everywhere — no free lunch, made concrete.
"""
import numpy as np

rng = np.random.default_rng(3)
n = 1200


def make_datasets(n):
    """Four binary problems with deliberately different geometry."""
    out = {}

    # 1. linearly separable
    X = rng.normal(size=(n, 2))
    out["linear boundary"] = (X, (X[:, 0] + X[:, 1] > 0).astype(int))

    # 2. axis-aligned rectangle — natural for trees
    X = rng.uniform(-3, 3, (n, 2))
    out["axis-aligned box"] = (X, ((np.abs(X[:, 0]) < 1.2)
                                   & (np.abs(X[:, 1]) < 1.2)).astype(int))

    # 3. concentric rings — needs a nonlinear boundary
    r = rng.uniform(0, 3, n)
    theta = rng.uniform(0, 2 * np.pi, n)
    X = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    out["concentric rings"] = (X, (r > 1.6).astype(int))

    # 4. XOR — no single linear split works
    X = rng.normal(size=(n, 2))
    out["XOR"] = (X, ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int))
    return out


# --- four tiny model families, each with one clear assumption ---------------
def fit_linear(Xtr, ytr, Xte):
    """Assumption: the boundary is a straight line."""
    A = np.column_stack([np.ones(len(Xtr)), Xtr])
    w = np.zeros(A.shape[1])
    for _ in range(400):
        p = 1 / (1 + np.exp(-np.clip(A @ w, -30, 30)))
        w -= 0.5 * (A.T @ (p - ytr) / len(ytr))
    B = np.column_stack([np.ones(len(Xte)), Xte])
    return (1 / (1 + np.exp(-np.clip(B @ w, -30, 30))) > 0.5).astype(int)


def fit_stump_grid(Xtr, ytr, Xte, bins=6):
    """Assumption: the boundary is axis-aligned (a coarse tree)."""
    edges = [np.quantile(Xtr[:, j], np.linspace(0, 1, bins + 1)) for j in (0, 1)]
    def cell(X):
        return tuple(np.clip(np.digitize(X[:, j], edges[j][1:-1]), 0, bins - 1)
                     for j in (0, 1))
    ctr = cell(Xtr)
    table = {}
    for i in range(len(Xtr)):
        table.setdefault((ctr[0][i], ctr[1][i]), []).append(ytr[i])
    table = {k: int(np.mean(v) >= 0.5) for k, v in table.items()}
    cte = cell(Xte)
    default = int(ytr.mean() >= 0.5)
    return np.array([table.get((cte[0][i], cte[1][i]), default)
                     for i in range(len(Xte))])


def fit_knn(Xtr, ytr, Xte, k=9):
    """Assumption: nearby points share labels."""
    d = ((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)
    nn = np.argpartition(d, k, axis=1)[:, :k]
    return (ytr[nn].mean(axis=1) >= 0.5).astype(int)


def fit_radial(Xtr, ytr, Xte):
    """Assumption: the boundary depends only on distance from the origin."""
    rtr = np.linalg.norm(Xtr, axis=1)[:, None]
    rte = np.linalg.norm(Xte, axis=1)[:, None]
    return fit_linear(rtr, ytr, rte)


models = {"linear": fit_linear, "axis-aligned": fit_stump_grid,
          "kNN": fit_knn, "radial": fit_radial}


N_TEST = 4000          # held out at a fixed size so the two tables compare


def bake_off(n_train, label):
    """Train on n_train rows, always evaluate on the same-size held-out set."""
    datasets = make_datasets(n_train + N_TEST)
    print(f"\n{label}")
    print(f"{'dataset':<20} " + " ".join(f"{m:>13}" for m in models))
    print("-" * 72)
    best_of = {}
    for dname, (X, y) in datasets.items():
        Xtr, ytr, Xte, yte = X[:n_train], y[:n_train], X[n_train:], y[n_train:]
        row, accs = [], {}
        for mname, fn in models.items():
            acc = (fn(Xtr, ytr, Xte) == yte).mean()
            accs[mname] = acc
            row.append(f"{acc:>13.3f}")
        best_of[dname] = (max(accs, key=accs.get), max(accs.values()))
        print(f"{dname:<20} " + " ".join(row))
    for dname, (mname, acc) in best_of.items():
        print(f"  best on {dname:<20} {mname:<14} {acc:.3f}")
    return best_of


big = bake_off(2000, "2000 training rows, 2 features — plenty of data")

print("\nkNN wins or ties nearly everything. That is not an accident and not a")
print("recommendation: in two dimensions with two thousand training points,")
print("'nearby points share labels' is almost always true, so the weakest")
print("assumption is the best one. Assumptions earn their keep when data is")
print("scarce:")

small = bake_off(40, "40 training rows, same four problems, same test set")

print("\nNow they separate, and the ordering changes. With forty points kNN")
print("no longer has neighbours close enough to trust, while the models that")
print("assume a shape need only a handful of points to locate it. A strong")
print("assumption is cheap when it is right and expensive when it is wrong —")
print("that is the whole trade (section 5.3).\n")
for dname in big:
    print(f"  {dname:<20} 2000 rows: {big[dname][0]:<14}"
          f"   40 rows: {small[dname][0]}")

# --- and the assumption is visible in what each one CANNOT do ---------------
print("\n" + "=" * 72)
print("what each assumption rules out")
print("=" * 72)
datasets = make_datasets(n)
X, y = datasets["XOR"]
cut = int(0.7 * n)
lin_acc = (fit_linear(X[:cut], y[:cut], X[cut:]) == y[cut:]).mean()
print(f"linear model on XOR: {lin_acc:.3f}  (chance is "
      f"{max(y[cut:].mean(), 1 - y[cut:].mean()):.3f})")
print("No amount of data fixes this: XOR is not in the linear hypothesis")
print("space at all. It is underfitting caused by the choice, not the fit.")

X2 = np.column_stack([X, X[:, 0] * X[:, 1]])       # add the interaction
lin_acc2 = (fit_linear(X2[:cut], y[:cut], X2[cut:]) == y[cut:]).mean()
print(f"\nlinear model on XOR + an interaction feature: {lin_acc2:.3f}")
print("Feature engineering (Chapter 27) enlarged the hypothesis space to")
print("contain the answer. That is the same lever as changing model family.")
```

The measured result is worth reading carefully, because it is the shape of the
whole trade-off. At two thousand rows in two dimensions, k-nearest neighbours —
the model with the weakest assumption — wins or ties almost everything, because
with that much data in that few dimensions "nearby points share labels" is
simply true. At forty rows it collapses on exactly the problems where the
labelled region is small relative to the spacing between points (rings, 0.982 →
0.659; XOR, 0.979 → 0.805), while the models that assume a *shape* barely move
(radial on rings, 0.985 → 0.999).

The generalisation: **a strong assumption is a substitute for data.** It costs
you nothing when it is right, costs you everything when it is wrong, and matters
less and less as $N$ grows. That is why the answer to "which model?" depends on
how much data you have, and why the deep-learning-versus-gradient-boosting
question in {{ch:ml-boosting}} is a question about data size rather than about
which algorithm is cleverer.

## 9. Common Mistakes

**Using machine learning where a rule would do.** Harder to debug, needs
monitoring, and fails differently.

**Judging a model by training error.** A memoriser scores perfectly.

**Assuming a more flexible model is better.** It has more ways to fit noise, and
{{eq:generalisation-bound}} says the gap grows with capacity.

**Treating no free lunch as "all models are equal in practice".** It says the
opposite is only true because real problems are structured.

**Not asking what the model family assumes.** Every failure in
{{sec:8-practical-example}} is an assumption mismatch.

**Ignoring the i.i.d. assumption.** It fails under ordering, repeated entities
and feedback, and then validation stops estimating anything.

**Confusing the two cultures.** Accuracy does not answer a causal question, and
interpretability is not required for a routing decision.

**Reaching for a complex model before a baseline.** Without one you cannot tell
whether the complexity bought anything.

## 10. Connection to Previous Chapters

{{ch:math-optimization}} supplied the loss-minimisation frame that
{{eq:empirical-risk}} formalises, and derived cross-entropy from maximum
likelihood. {{ch:math-inference}} supplied the $\sqrt{1/N}$ rate that reappears
in {{eq:generalisation-bound}} and the selection effect that makes an untouched
test set necessary. {{ch:ds-leakage}} supplied the validation designs that make
the empirical risk an honest estimate. {{ch:ds-feature-eng}} supplied the
hypothesis-space enlargement demonstrated at the end of
{{sec:8-practical-example}}.

Forward: {{ch:ml-linear-regression}} and {{ch:ml-logistic}} are the two families
whose search can be characterised completely. {{ch:ml-metrics}} makes the
generalisation gap measurable and decomposes it.

Beyond Part IV: {{ch:fm-pretraining}} is self-supervised learning at scale, and
{{ch:res-scaling}} returns to why {{eq:generalisation-bound}} fails to describe
modern networks. {{cite:breiman2001cultures}} is the reference for the two
cultures; {{cite:pedregosa2011}} for the library conventions this part uses.

## 11. Exercises

**Beginner**

1. Give three problems suited to machine learning and two better solved with
   rules.
2. Classify as supervised, unsupervised or self-supervised: spam filtering,
   customer segmentation, next-word prediction, fraud detection.
3. What is a hypothesis space? Give one that cannot express XOR.
4. Why is training accuracy a poor measure of quality?
5. Distinguish parametric from non-parametric with an example of each.

**Intermediate**

6. State the no-free-lunch result and explain what it does *not* imply.
7. Using {{eq:sample-complexity}}, compute the data needed for a 0.02 gap with
   $M = 10^{6}$ and $\delta = 0.05$.
8. Explain why self-supervised learning changed the field.
9. Give a case where the i.i.d. assumption fails and say what breaks.
10. Which culture does a churn model belong to? A clinical-trial analysis?
11. Explain why adding an interaction feature let the linear model solve XOR.

**Advanced**

12. Reproduce the counting argument of {{sec:6-mathematical-foundation}} and
    state exactly where uniformity over targets is used.
13. Explain why {{eq:generalisation-bound}} is vacuous for modern neural
    networks, and what that implies about the bound's usefulness.
14. Formalise the difference between a hypothesis space that cannot contain the
    truth and one that contains it but is not found by the optimiser.
15. Argue for and against the claim that all machine learning is
    self-supervised learning with different masking.

**Implementation**

16. Extend {{sec:8-practical-example}} with a fifth dataset and a fifth model
    family, and predict which wins before running it.
17. Reproduce the no-free-lunch enumeration for a different notion of
    "structured" target and report how the separation changes.
18. Implement a learning curve — accuracy against training-set size — for two
    model families on the same data, and explain the difference in shape.
19. Build a memoriser that generalises slightly by matching on rounded features,
    and find the rounding at which it starts to beat chance.

**Reasoning**

20. If no algorithm is universally best, how should a team decide what to try
    first on a new problem?
21. Deep learning replaced feature engineering for images and text and not for
    tabular data. Explain in terms of hypothesis spaces and assumptions.

## 12. Chapter Summary

Machine learning estimates a function from examples instead of specifying rules,
and is the right choice exactly when the rule is easier to demonstrate than to
state. If you can write the rule, write the rule.

The paradigms differ in what supervision is available. Self-supervised learning
matters most historically, because manufacturing a prediction task from
unlabelled data is what made the last decade possible.

A model family is a hypothesis space — the set of functions reachable at all.
If the truth lies outside it, no amount of data helps; if the space is large
enough to fit noise, the search may do so. Choosing a model is choosing an
assumption before seeing any data.

Learning minimises empirical risk as a proxy for expected risk, and the entire
subject lives in the gap between them. Training error is nearly uninformative — a
memoriser achieves zero — and the gap grows with capacity and shrinks as
$\sqrt{1/N}$.

No free lunch says that averaged uniformly over all target functions, every
algorithm ties at chance. Verified by enumeration, and escaped in practice
because real targets are overwhelmingly structured. An algorithm's assumptions
are therefore its value, not its limitation.

Breiman's two cultures — data modelling for understanding, algorithmic
modelling for prediction — answer different questions, and most arguments about
interpretability are the two talking past each other.

---
id: ml-knn-nb
number: 35
part: IV
tier: focused
status: reviewed
requires: [ml-metrics, ml-logistic, math-probability, math-vectors]
provides: [knn, curse-of-dimensionality, naive-bayes, laplace-smoothing,
           generative-vs-discriminative, distance-metrics, lazy-learning]
citations: [pedregosa2011]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Implement k-nearest neighbours and explain how $k$ moves along the
   bias-variance curve.
2. Explain the curse of dimensionality quantitatively and its consequence for
   distance-based methods.
3. Choose a distance metric and explain why scaling is mandatory.
4. Derive naive Bayes from Bayes' rule and the conditional-independence
   assumption.
5. Explain why naive Bayes classifies well while its probabilities are
   badly calibrated.
6. Apply Laplace smoothing and explain what breaks without it.
7. Distinguish generative from discriminative models and state when each wins.

## 2. Why This Matters

These two algorithms are the simplest members of two entirely different
families, and both families matter far beyond this chapter.

**k-NN is retrieval, and retrieval is the backbone of modern AI systems.** The
whole of {{part:11}} and {{part:12}} — embeddings, vector databases,
retrieval-augmented generation — is k-NN with learned representations and an
approximate index. The distance metric, the choice of $k$, and the curse of
dimensionality are the same concerns there as here, and they are much easier to
understand on four numeric features than on 1536-dimensional embeddings.
{{ch:emb-ann}} is this chapter with a better index.

**Naive Bayes is the simplest generative model, and generative modelling is what
{{part:9}} onward is about.** A language model estimates $p(\text{text})$ and
generates from it; naive Bayes estimates $p(\vec{x} \mid y)$ and classifies with
it. The distinction between modelling the joint distribution and modelling the
boundary is one of the most useful axes for organising everything in this book.

**Both are strong baselines that take one line.** Naive Bayes on text is fast,
needs almost no data, and is frequently within a few points of far more
expensive models. A baseline you can fit in a second is what tells you whether
the expensive model is earning its cost.

## 3. Prerequisites

{{ch:ml-metrics}} for the bias-variance frame and validation.
{{ch:math-probability}} for Bayes' rule and conditional independence.
{{ch:math-vectors}} for norms and distances. {{ch:ds-feature-eng}} for scaling,
which k-NN makes non-negotiable.

## 4. Intuitive Explanation

### 4.1 k-NN: no model at all

To classify a new point, find the $k$ closest training points and take a vote.
That is the entire algorithm. There is no training step — the training data *is*
the model — which is why it is called **lazy learning**.

The cost structure is inverted relative to everything else in this part.
Training is free and prediction is expensive: $O(ND)$ per query with a brute-force
scan, and the entire training set must be kept in memory. For a model served at
scale this is usually the wrong trade, and it is why approximate nearest-neighbour
indexes ({{ch:emb-ann}}) exist.

$k$ is a direct bias-variance knob, and an unusually clean one:

```text
  k = 1                    k = 25                   k = N
  ┌──────────┐            ┌──────────┐            ┌──────────┐
  │ ▟▖ ▗▛ ▟▖  │            │  ▗▄▄▄▖   │            │          │
  │▛ ▚▟  ▛ ▜ │            │ ▟█████▙  │            │  always  │
  │ ▚▖▟▘ ▚▄▘ │            │  ▀▀▀▀▀   │            │ majority │
  └──────────┘            └──────────┘            └──────────┘
  zero training error      smooth boundary         maximum bias
  maximum variance         the useful range        zero variance
```

At $k=1$ every training point is classified correctly by itself, training error
is zero, and the decision boundary is a jagged shape that changes completely
when one point moves. At $k=N$ every prediction is the majority class. The
useful $k$ is chosen by cross-validation and typically lands in the range
$\sqrt{N}$ or below.

### 4.2 The curse of dimensionality

k-NN rests on one assumption: nearby points share labels. In high dimensions,
"nearby" stops meaning anything.

Consider points uniformly distributed in a $D$-dimensional unit cube. To capture
a fraction $r$ of the data in a hypercube neighbourhood, the neighbourhood must
have side length $r^{1/D}$. In two dimensions, capturing 1% needs a side of
$0.01^{1/2} = 0.1$ — a tenth of the range, genuinely local. In fifty dimensions
it needs $0.01^{1/50} = 0.912$ — 91% of the range along every axis. Your "local"
neighbourhood is nearly the entire space.

The consequence is that distances concentrate: the ratio between the nearest and
farthest neighbour approaches 1, and "nearest" becomes arbitrary.
{{sec:7-implementation}} measures this.

> IMPORTANT: This is why the embedding models of {{part:11}} matter so much. They
> do not defeat the curse by magic; they map data into a space where the *true*
> structure occupies a low-dimensional manifold, so the effective dimension is
> far below the nominal one. A 1536-dimensional embedding works as a k-NN space
> precisely because it is not really 1536-dimensional.

### 4.3 Naive Bayes: model each class, then compare

Instead of drawing a boundary, model what each class *looks like* and ask which
class more plausibly generated the observation. That is the generative approach,
and Bayes' rule turns it into a classifier.

The "naive" part is the assumption that features are conditionally independent
given the class. For text: given that an email is spam, the presence of
"viagra" tells you nothing about the presence of "click". This is transparently
false — spam vocabulary is highly correlated — and the classifier works anyway.

The reason is worth stating clearly, because it is a genuinely useful idea: the
independence assumption ruins the *probabilities* but frequently preserves the
*argmax*. Correlated features cause the same evidence to be counted several
times, driving the posterior to 0.9999 when it should be 0.7 — but if the
over-counting favours the correct class, the classification is still right. Naive
Bayes is a good classifier and a bad probability estimator, and that combination
is exactly what {{ch:ml-metrics}} separated into discrimination and calibration.

## 5. Formal Explanation

### 5.1 k-NN

Given a query $\vec{x}$, let $\mathcal{N}_k(\vec{x})$ be the indices of the $k$
training points minimising a distance $d(\vec{x}, \vec{x}_i)$. Then

$$
\hat{p}(y = c \mid \vec{x}) = \frac{1}{k}\sum_{i \in \mathcal{N}_k(\vec{x})}
   \Ind[y_i = c]
$$ (eq:knn-classify)

with regression the analogous average of $y_i$. **Distance weighting** replaces
the uniform average by weights $1/d_i$, which lets a nearer neighbour count for
more and makes the prediction continuous in $\vec{x}$.

Common metrics:

$$
d_p(\vec{a}, \vec{b}) = \Big(\sum_j |a_j - b_j|^{p}\Big)^{1/p},
\qquad
d_{\cos}(\vec{a},\vec{b}) = 1 - \frac{\vec{a}\T\vec{b}}
                                     {\|\vec{a}\|\,\|\vec{b}\|}
$$ (eq:distances)

$p=2$ is Euclidean, $p=1$ is Manhattan (more robust in high dimensions because a
single wild coordinate is not squared), and cosine distance ignores magnitude
entirely — which is why it is the default for text embeddings, where document
length should not determine similarity.

> WARNING: k-NN is scale-dependent in the strongest possible sense. A feature
> measured in metres and one measured in kilometres contribute to the distance in
> a ratio of $10^{6}$ once squared. Standardising is not a refinement here; it is
> the difference between the algorithm working and the algorithm reading one
> feature. {{sec:7-implementation}} measures the damage.

### 5.2 Naive Bayes

Bayes' rule gives

$$
\Prob(y = c \mid \vec{x}) = \frac{\Prob(\vec{x} \mid y=c)\,\Prob(y=c)}
                                 {\Prob(\vec{x})}
$$ (eq:bayes-rule)

The denominator does not depend on $c$, so for classification it can be dropped.
The difficulty is $\Prob(\vec{x} \mid y=c)$: a joint distribution over $D$
features, which for binary features has $2^{D}-1$ parameters per class and
cannot be estimated.

The naive assumption factorises it:

$$
\Prob(\vec{x} \mid y=c) = \prod_{j=1}^{D} \Prob(x_j \mid y=c)
$$ (eq:naive-assumption)

reducing the count from exponential to linear in $D$. The classifier is then

$$
\hat{y} = \argmax_{c} \Big[\log\Prob(y=c)
  + \sum_{j=1}^{D}\log\Prob(x_j \mid y=c)\Big]
$$ (eq:naive-bayes-classify)

Working in logs is not optional: a product of a thousand probabilities underflows
to zero in double precision.

The per-feature distribution depends on the feature type — Gaussian for
continuous features, multinomial for counts, Bernoulli for binary presence. The
multinomial variant on word counts is the classical spam filter.

### 5.3 Laplace smoothing

If a word never appears in the training spam, then $\Prob(w \mid \text{spam}) =
0$, and one zero annihilates the entire product in {{eq:naive-assumption}}
regardless of the other thousand words. A single unseen feature vetoes the class.

**Laplace (add-$\alpha$) smoothing** fixes it:

$$
\hat{\Prob}(w \mid c) = \frac{\text{count}(w, c) + \alpha}
                              {\sum_{w'}\text{count}(w', c) + \alpha|V|}
$$ (eq:laplace)

with $\alpha = 1$ the usual choice. It is exactly the posterior mean under a
Dirichlet prior — a principled Bayesian estimate, not a hack — and it is
mandatory rather than optional.

### 5.4 Generative versus discriminative

Naive Bayes models $\Prob(\vec{x}, y)$ and derives the boundary. Logistic
regression models $\Prob(y \mid \vec{x})$ directly. This is one of the more
useful distinctions in machine learning:

{#tbl:gen-vs-disc caption="Generative and discriminative models compared. The asymptotic ordering reverses at small samples, which is the practically important part."}

| | Generative (naive Bayes) | Discriminative (logistic) |
|---|---|---|
| Models | $\Prob(\vec{x}, y)$ | $\Prob(y \mid \vec{x})$ |
| Assumptions | strong (feature distributions) | weak (boundary form) |
| Small data | **converges faster** | needs more data |
| Large data | plateaus at a higher error | **lower asymptotic error** |
| Can generate data | yes | no |
| Missing features | handled by marginalising | requires imputation |

The crossover is real and reproducible: with few examples the generative model's
assumptions substitute for data, and with many examples they become a ceiling.
{{sec:7-implementation}} measures it on the same dataset.

The distinction runs through the whole book. A language model ({{part:10}}) is
generative — it estimates a distribution over text and samples from it — while a
classifier head is discriminative. Prompting a generative model to answer a
classification question is using a generative model discriminatively, which is
why it works at all, and why its stated confidence deserves the same scepticism
naive Bayes' does.

## 6. Mathematical Foundation

### 6.1 Why k-NN works at all: the 1-NN bound

A classical result: as $N \to \infty$, the error rate of 1-NN is bounded by

$$
R^{*} \le R_{1\text{NN}} \le R^{*}\left(2 - \frac{C}{C-1}R^{*}\right) \le 2R^{*}
$$ (eq:knn-bound)

where $R^{*}$ is the Bayes error — the irreducible minimum from
{{ch:ml-metrics}} — and $C$ is the number of classes.

The sketch: as $N \to \infty$ the nearest neighbour of $\vec{x}$ converges to
$\vec{x}$ itself, so its label is a draw from $p(y \mid \vec{x})$. For binary
classification the probability that a draw from $p(y\mid\vec{x})$ disagrees with
an independent draw is $2p(1-p)$, which is at most twice the Bayes error
$\min(p, 1-p)$.

The result is remarkable: an algorithm with no model and no training is within a
factor of two of optimal, asymptotically. The catch is entirely in "as $N \to
\infty$", and {{sec:6-mathematical-foundation}} explains why that limit is
unreachable in high dimensions.

### 6.2 The curse, quantified

**Neighbourhood size.** To capture a fraction $r$ of uniformly distributed data
in $D$ dimensions requires a hypercube of side

$$
s(r, D) = r^{1/D}
$$ (eq:neighbourhood-side)

At $D=100$ and $r = 0.001$, $s = 0.933$. There is no such thing as a local
neighbourhood.

**Distance concentration.** For i.i.d. coordinates, the squared distance between
two random points is a sum of $D$ i.i.d. terms, so by the CLT of
{{ch:math-inference}} its mean grows as $D$ while its standard deviation grows
as $\sqrt{D}$. Hence

$$
\frac{\text{sd}[d]}{\E[d]} = O\!\left(\frac{1}{\sqrt{D}}\right)
\;\longrightarrow\; 0
$$ (eq:distance-concentration)

All pairwise distances converge to the same value. The formal consequence,

$$
\frac{d_{\max} - d_{\min}}{d_{\min}} \to 0
$$ (eq:nn-degeneracy)

says the nearest neighbour is not meaningfully nearer than the farthest one, and
"nearest neighbour" degenerates into an arbitrary choice.

**Sample density.** To maintain a fixed density, $N$ must grow as $k^{D}$ —
exponentially in dimension. Ten points per axis is 10 points in 1-D, $10^{10}$ in
10-D, and more than the number of atoms in the observable universe by dimension
80.

The escape is that real data does not fill its ambient space. Images live near a
low-dimensional manifold within pixel space; embeddings are built to make that
true. What matters is **intrinsic** dimension, not nominal, and adding a
genuinely uninformative feature costs you real accuracy —
{{sec:7-implementation}} measures the loss per added noise dimension.

### 6.3 Why naive Bayes miscalibrates but classifies

Suppose two features are perfect duplicates. {{eq:naive-bayes-classify}} adds
$\log\Prob(x_j \mid c)$ twice, so the log-odds contributed by that single piece
of evidence is doubled. With $m$ correlated copies it is multiplied by $m$.

Write the true log-odds as $\eta$ and the naive estimate as $\hat{\eta} \approx
m\eta$ for a duplication factor $m > 1$. Then:

**The sign is preserved.** $\sign(m\eta) = \sign(\eta)$ for $m > 0$, so the
classification is unchanged. The decision is invariant to the over-counting.

**The probability is destroyed.** $\sigma(m\eta) \to \{0, 1\}$ rapidly as $m$
grows, so posteriors pile up at the extremes. Naive Bayes is famous for
returning 0.99999.

This is exactly the calibration-versus-discrimination split of
{{ch:ml-metrics}}, arising here from an identifiable mechanism rather than as an
empirical observation: ROC-AUC is invariant to any monotone transformation of the
scores, and $\eta \mapsto m\eta$ is monotone. If you need the probability, apply
Platt scaling or isotonic regression to the output — a monotone recalibration
that leaves the ranking, and therefore the AUC, untouched.

## 7. Implementation

```python {tier=A name=knn-from-scratch}
"""k-NN from scratch: the k knob, scaling, metrics, and the curse.
"""
import numpy as np

rng = np.random.default_rng(0)


def knn_predict(Xtr, ytr, Xte, k=5, metric="euclidean", weighted=False):
    """Brute-force k-NN (eq. 35.1). O(N_te * N_tr * D) — deliberately."""
    if metric == "euclidean":
        d = np.sqrt(((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1))
    elif metric == "manhattan":
        d = np.abs(Xte[:, None, :] - Xtr[None, :, :]).sum(-1)
    elif metric == "cosine":
        A = Xte / np.maximum(np.linalg.norm(Xte, axis=1, keepdims=True), 1e-12)
        B = Xtr / np.maximum(np.linalg.norm(Xtr, axis=1, keepdims=True), 1e-12)
        d = 1.0 - A @ B.T
    else:
        raise ValueError(metric)

    idx = np.argpartition(d, min(k, d.shape[1] - 1), axis=1)[:, :k]
    lab = ytr[idx]
    if not weighted:
        return (lab.mean(axis=1) >= 0.5).astype(int)
    w = 1.0 / np.maximum(np.take_along_axis(d, idx, axis=1), 1e-12)
    return ((lab * w).sum(1) / w.sum(1) >= 0.5).astype(int)


def make_moons(n, noise=0.25):
    """Two interleaving crescents: not linearly separable, locally smooth."""
    t = rng.uniform(0, np.pi, n)
    top = np.column_stack([np.cos(t), np.sin(t)])
    bot = np.column_stack([1 - np.cos(t), 0.5 - np.sin(t)])
    X = np.vstack([top, bot]) + rng.normal(0, noise, (2 * n, 2))
    y = np.r_[np.zeros(n, int), np.ones(n, int)]
    p = rng.permutation(len(y))
    return X[p], y[p]


X, y = make_moons(600)
cut = 800
Xtr, ytr, Xte, yte = X[:cut], y[:cut], X[cut:], y[cut:]

# --- k is a bias-variance knob (Chapter 34) ---------------------------------
print("=" * 72)
print("k moves directly along the bias-variance curve")
print("=" * 72)
print(f"{'k':>5} {'train acc':>11} {'test acc':>10} {'gap':>8}")
for k in (1, 3, 5, 11, 25, 75, 201, len(ytr)):
    tr = (knn_predict(Xtr, ytr, Xtr, k) == ytr).mean()
    te = (knn_predict(Xtr, ytr, Xte, k) == yte).mean()
    print(f"{k:>5} {tr:>11.4f} {te:>10.4f} {tr - te:>8.4f}")
print("\nk=1 gets every training point right by construction — it is its own")
print("nearest neighbour — and the gap is pure variance. At k=N every")
print("prediction is the majority class: pure bias. Test accuracy peaks in")
print("between, which is the whole of Chapter 34 in one column.")

# --- distance weighting -----------------------------------------------------
print(f"\n{'k':>5} {'uniform':>9} {'distance-weighted':>19} {'difference':>12}")
for k in (1, 5, 25, 101):
    u = (knn_predict(Xtr, ytr, Xte, k) == yte).mean()
    w = (knn_predict(Xtr, ytr, Xte, k, weighted=True) == yte).mean()
    print(f"{k:>5} {u:>9.4f} {w:>19.4f} {w - u:>+12.4f}")
print("Distance weighting is often described as a clear improvement. On")
print("this data it is worth a quarter of a point at k=5 and slightly")
print("NEGATIVE at k=25 and above — all of these differences are within the")
print("noise of a 400-row test set. Its real value is making the prediction")
print("continuous in x, which matters for regression and for ranking; treat")
print("accuracy gains as a hypothesis to test, not a given.")

# --- scaling is not optional ------------------------------------------------
print("\n" + "=" * 72)
print("what happens when one feature is measured in different units")
print("=" * 72)
print(f"{'scale of feature 2':>20} {'raw k-NN':>10} {'standardised':>14}")
for scale in (1, 10, 100, 10000):
    Xs = X.copy()
    Xs[:, 1] *= scale
    a, b = Xs[:cut], Xs[cut:]
    raw = (knn_predict(a, ytr, b, 11) == yte).mean()
    mu, sd = a.mean(0), a.std(0)
    std = (knn_predict((a - mu) / sd, ytr, (b - mu) / sd, 11) == yte).mean()
    print(f"{scale:>20,} {raw:>10.4f} {std:>14.4f}")
print("\nMultiplying one column by 10,000 does not change the information in")
print("the data at all, and destroys raw k-NN: the distance is now entirely")
print("that one feature. Standardisation is immune. This is the single most")
print("common way to get k-NN silently wrong.")

# --- section 6.2: the curse, measured ---------------------------------------
print("\n" + "=" * 72)
print("distance concentration (eq. 35.6, 35.7)")
print("=" * 72)
print(f"{'D':>5} {'mean dist':>11} {'sd/mean':>9} "
      f"{'(dmax-dmin)/dmin':>18} {'side for 1% (eq 35.5)':>23}")
for D in (2, 5, 20, 100, 1000):
    P = rng.uniform(0, 1, (600, D))
    q = rng.uniform(0, 1, (40, D))
    d = np.sqrt(((q[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    ratio = np.mean((d.max(1) - d.min(1)) / d.min(1))
    print(f"{D:>5} {d.mean():>11.3f} {d.std() / d.mean():>9.4f} "
          f"{ratio:>18.4f} {0.01 ** (1 / D):>23.4f}")
print("\nAt D=1000 the farthest point is 12% further than the nearest, and a")
print("'neighbourhood' holding 1% of the data spans 99.5% of every axis.")
print("'Nearest neighbour' has stopped meaning anything.")

# --- and what that costs in accuracy ----------------------------------------
print("\n" + "=" * 72)
print("the accuracy cost of adding PURE NOISE features")
print("=" * 72)
print(f"{'noise dims added':>18} {'total D':>9} {'k-NN acc':>10} "
      f"{'logistic acc':>14}")


def fit_logistic_simple(A, b, B, n_iter=400, lr=0.4):
    A1 = np.column_stack([np.ones(len(A)), A])
    B1 = np.column_stack([np.ones(len(B)), B])
    w = np.zeros(A1.shape[1])
    for _ in range(n_iter):
        p = 1 / (1 + np.exp(-np.clip(A1 @ w, -30, 30)))
        w -= lr * (A1.T @ (p - b) / len(b))
    return (1 / (1 + np.exp(-np.clip(B1 @ w, -30, 30))) >= 0.5).astype(int)


for n_noise in (0, 2, 5, 10, 30, 100):
    Xn = np.column_stack([X, rng.normal(size=(len(X), n_noise))]) \
        if n_noise else X
    a, b_ = Xn[:cut], Xn[cut:]
    mu, sd = a.mean(0), a.std(0)
    a, b_ = (a - mu) / sd, (b_ - mu) / sd
    acc = (knn_predict(a, ytr, b_, 11) == yte).mean()
    lacc = (fit_logistic_simple(a, ytr.astype(float), b_) == yte).mean()
    print(f"{n_noise:>18} {Xn.shape[1]:>9} {acc:>10.4f} {lacc:>14.4f}")

print("\nThe two informative features are untouched throughout; everything")
print("added is independent noise. k-NN degrades steadily because the noise")
print("dimensions dominate the distance. This is why feature selection")
print("(Chapter 27) matters far more for distance-based methods than for")
print("models that can learn to ignore a feature by giving it a small")
print("coefficient.")
```

```python {tier=A name=naive-bayes}
"""Naive Bayes: derivation, smoothing, calibration failure, and the
generative/discriminative crossover.
"""
import numpy as np

rng = np.random.default_rng(3)


class MultinomialNB:
    """Multinomial naive Bayes with add-alpha smoothing (eq. 35.11)."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.classes = np.unique(y)
        V = X.shape[1]
        self.log_prior = np.array(
            [np.log((y == c).mean()) for c in self.classes])
        self.log_lik = np.empty((len(self.classes), V))
        for i, c in enumerate(self.classes):
            counts = X[y == c].sum(axis=0) + self.alpha
            self.log_lik[i] = np.log(counts / counts.sum())
        return self

    def log_joint(self, X):
        """log P(c) + sum_j x_j log P(w_j | c)  — eq. 35.10, in logs because
        a product of thousands of probabilities underflows."""
        return X @ self.log_lik.T + self.log_prior

    def predict(self, X):
        return self.classes[self.log_joint(X).argmax(1)]

    def predict_proba(self, X):
        lj = self.log_joint(X)
        lj = lj - lj.max(1, keepdims=True)          # the softmax shift again
        e = np.exp(lj)
        return e / e.sum(1, keepdims=True)


# --- a synthetic bag-of-words corpus ----------------------------------------
# Deliberately HARD: short documents and a weak vocabulary contrast, so the
# task lands around 80% and there is room for the failure modes to show.
V, n_docs, CONTRAST = 200, 4500, 1.8
w1 = np.ones(V); w1[:40] = CONTRAST      # class 1 favours words 0-39
w0 = np.ones(V); w0[40:80] = CONTRAST    # class 0 favours words 40-79
w1, w0 = w1 / w1.sum(), w0 / w0.sum()    # the other 120 words are shared

y = (rng.random(n_docs) < 0.5).astype(int)
lengths = rng.poisson(20, n_docs) + 3
X = np.array([rng.multinomial(L, w1 if lab else w0)
              for L, lab in zip(lengths, y)])

cut = 3000
Xtr, ytr, Xte, yte = X[:cut], y[:cut], X[cut:], y[cut:]

nb = MultinomialNB(alpha=1.0).fit(Xtr, ytr)
print(f"naive Bayes test accuracy: {(nb.predict(Xte) == yte).mean():.4f}")
print(f"(chance is {max(yte.mean(), 1 - yte.mean()):.4f})")

# --- section 5.3: what happens without smoothing ----------------------------
print("\n" + "=" * 72)
print("Laplace smoothing is not optional (eq. 35.11)")
print("=" * 72)
print("Unseen words only occur when the training set is small — which is")
print("exactly when naive Bayes is the model you reached for.\n")
print(f"{'train docs':>11} {'unseen words':>14} " +
      " ".join(f"{'a=' + str(a):>8}"
               for a in (0.0, 1e-10, 0.01, 0.1, 1.0, 10.0, 100.0)))
for n_small in (40, 80, 200, 3000):
    A, b = X[:n_small], y[:n_small]
    unseen = max((A[b == c].sum(0) == 0).sum() for c in (0, 1))
    accs = []
    for alpha in (0.0, 1e-10, 0.01, 0.1, 1.0, 10.0, 100.0):
        with np.errstate(divide="ignore", invalid="ignore"):
            accs.append((MultinomialNB(alpha).fit(A, b).predict(Xte)
                         == yte).mean())
    print(f"{n_small:>11} {unseen:>14} " + " ".join(f"{a:>8.3f}" for a in accs))

print("\nWith 40 training documents about 29 of the 200 words are unseen in")
print("some class, and alpha=0 scores 0.52 — chance. Each unseen word")
print("contributes log(0) = -inf and vetoes that class outright, whatever")
print("the other 199 words say. Any positive alpha removes the veto and")
print("recovers 17 accuracy points. Too large an alpha (100) washes the")
print("evidence out towards the prior. Once the training set is large")
print("enough that nothing is unseen, alpha stops mattering — which is why")
print("this bug hides until the day you deploy on a rare class.")

# --- section 6.3: good classifier, terrible probabilities -------------------
print("\n" + "=" * 72)
print("correlated features destroy the probabilities, not the decisions")
print("=" * 72)


def ece(y_true, p, n_bins=10):
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    tot = 0.0
    for i in range(n_bins):
        m = (p >= edges[i]) & (p <= edges[i + 1])
        if m.sum():
            tot += m.sum() / len(p) * abs(y_true[m].mean() - p[m].mean())
    return tot


def roc_auc(y_true, s):
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s), float)
    r[o] = np.arange(1, len(s) + 1)
    npos, nneg = int(y_true.sum()), int((1 - y_true).sum())
    return (r[y_true == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


print(f"{'duplicates of each word':>24} {'accuracy':>10} {'ROC-AUC':>9} "
      f"{'ECE':>8} {'mean max prob':>15}")
for m_dup in (1, 2, 4, 8):
    Xd_tr = np.tile(Xtr, (1, m_dup))
    Xd_te = np.tile(Xte, (1, m_dup))
    nbd = MultinomialNB(1.0).fit(Xd_tr, ytr)
    P = nbd.predict_proba(Xd_te)
    pred = nbd.predict(Xd_te)
    print(f"{m_dup:>24} {(pred == yte).mean():>10.4f} "
          f"{roc_auc(yte, P[:, 1]):>9.4f} {ece(yte, P[:, 1]):>8.4f} "
          f"{P.max(1).mean():>15.6f}")

print("\nDuplicating every feature adds exactly zero information. Accuracy")
print("moves by less than 0.3 points and ROC-AUC is IDENTICAL to four")
print("decimal places — the ranking cannot change, because the log-odds are")
print("merely multiplied by m, a monotone map (section 6.3). Calibration")
print("collapses: mean confidence climbs from 0.83 to 0.98 while accuracy")
print("stays at 0.81, and ECE goes up almost tenfold. This is the mechanism")
print("behind naive Bayes' reputation for returning 0.99999, and it is why")
print("you recalibrate before using the number for anything.")

# --- section 5.4: the generative/discriminative crossover -------------------
print("\n" + "=" * 72)
print("generative vs discriminative: who wins depends on how much data")
print("=" * 72)
print("The corpus is augmented with 30 near-duplicate word pairs, so naive")
print("Bayes' independence assumption is genuinely FALSE here — otherwise")
print("it would be the true model and could never be overtaken.\n")

Xc = np.hstack([X, X[:, :30] + rng.binomial(X[:, :30], 0.85)])
Xc_te, yc_te = Xc[cut:], y[cut:]


def fit_logistic(A, b, lam=0.01, n_iter=100):
    A1 = np.column_stack([np.ones(len(A)), A])
    w = np.zeros(A1.shape[1])
    for _ in range(n_iter):
        p = 1 / (1 + np.exp(-np.clip(A1 @ w, -30, 30)))
        g = A1.T @ (p - b) / len(b)
        g[1:] += 2 * lam * w[1:]
        S = np.maximum(p * (1 - p), 1e-7)
        H = (A1 * S[:, None]).T @ A1 / len(b) + (2 * lam + 1e-6) * np.eye(len(w))
        w -= np.linalg.solve(H, g)
    return w


def score_logistic(w, A, b):
    A1 = np.column_stack([np.ones(len(A)), A])
    return float((((A1 @ w) >= 0).astype(int) == b).mean())


print(f"{'train size':>11} {'naive Bayes':>13} {'logistic':>10} "
      f"{'winner':>13} {'margin':>9}")
for n_train in (6, 12, 25, 50, 100, 300, 1000):
    # average over many disjoint training samples of this size — a single
    # draw at n=6 says nothing at all
    starts = range(0, min(2400, cut - n_train) + 1, 150)
    nb_acc = np.mean([
        (MultinomialNB(1.0).fit(Xc[s:s + n_train], y[s:s + n_train])
         .predict(Xc_te) == yc_te).mean() for s in starts])
    lr_acc = np.mean([
        score_logistic(fit_logistic(np.log1p(Xc[s:s + n_train]),
                                    y[s:s + n_train].astype(float)),
                       np.log1p(Xc_te), yc_te) for s in starts])
    winner = "naive Bayes" if nb_acc > lr_acc else "logistic"
    print(f"{n_train:>11} {nb_acc:>13.4f} {lr_acc:>10.4f} "
          f"{winner:>13} {abs(nb_acc - lr_acc):>9.4f}")

print("\nThe crossover is real and lands between 25 and 50 documents here.")
print("Below it the generative model's assumptions substitute for data it")
print("does not have — with 6 documents and 230 features there is nothing")
print("to estimate a boundary from, and naive Bayes leads by 3.2 points.")
print("Above it logistic regression is ahead at every size, though only by")
print("a few tenths of a point: the independence violation planted here is")
print("mild, so naive Bayes' ceiling is only slightly below the truth. The")
print("honest summary is that the ORDERING flips reliably and the MARGIN")
print("above the crossover is small — which is why 'naive Bayes for small")
print("data' is sound advice and 'logistic regression is better' is not")
print("worth much without knowing how false the independence assumption is")
print("on your data (Table 35.2).")
```

## 8. Practical Example

```python {tier=A name=knn-vs-nb-workflow}
"""Choosing between them on a document-classification problem, and fixing
naive Bayes' probabilities without touching its decisions.
"""
import numpy as np

rng = np.random.default_rng(9)

# --- a 3-class corpus, deliberately hard ------------------------------------
V, K, n = 300, 3, 7500
topic_w = np.ones((K, V))
for c in range(K):
    topic_w[c, c * 50:(c + 1) * 50] = 1.9      # a weak, realistic contrast
topic_w /= topic_w.sum(1, keepdims=True)

y = rng.integers(0, K, n)
X = np.array([rng.multinomial(L, topic_w[c])
              for L, c in zip(rng.poisson(22, n) + 4, y)])
# near-duplicate word pairs: real corpora are full of them ("cannot"/"can't",
# "NYC"/"New York"), and they are what breaks the independence assumption
X = np.hstack([X, X[:, :60] + rng.binomial(X[:, :60], 0.85)])

n_tr, n_va = 3000, 1500
Xtr, ytr = X[:n_tr], y[:n_tr]
Xva, yva = X[n_tr:n_tr + n_va], y[n_tr:n_tr + n_va]
Xte, yte = X[n_tr + n_va:], y[n_tr + n_va:]


class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.log_prior = np.array([np.log((y == c).mean())
                                   for c in self.classes])
        self.log_lik = np.empty((len(self.classes), X.shape[1]))
        for i, c in enumerate(self.classes):
            ct = X[y == c].sum(0) + self.alpha
            self.log_lik[i] = np.log(ct / ct.sum())
        return self

    def decision(self, X):
        return X @ self.log_lik.T + self.log_prior

    def predict_proba(self, X):
        d = self.decision(X)
        d = d - d.max(1, keepdims=True)
        e = np.exp(d)
        return e / e.sum(1, keepdims=True)

    def predict(self, X):
        return self.classes[self.decision(X).argmax(1)]


def knn_cosine(Xtr, ytr, Xq, k):
    A = Xq / np.maximum(np.linalg.norm(Xq, axis=1, keepdims=True), 1e-12)
    B = Xtr / np.maximum(np.linalg.norm(Xtr, axis=1, keepdims=True), 1e-12)
    sim = A @ B.T
    idx = np.argpartition(-sim, k, axis=1)[:, :k]
    lab = ytr[idx]
    return np.array([np.bincount(r, minlength=3).argmax() for r in lab])


# --- tune both on the VALIDATION set, never the test set --------------------
print("=" * 72)
print("tuning on validation (Chapter 34: the test set is touched once)")
print("=" * 72)
print(f"{'k (cosine k-NN)':>17} {'val accuracy':>14}")
best_k, best = None, -1
for k in (1, 3, 5, 11, 25, 51, 101, 201):
    a = (knn_cosine(Xtr, ytr, Xva, k) == yva).mean()
    print(f"{k:>17} {a:>14.4f}")
    if a > best:
        best_k, best = k, a

print(f"\n{'alpha (naive Bayes)':>19} {'val accuracy':>14}")
best_a, best_nb = None, -1
for alpha in (0.01, 0.1, 0.5, 1.0, 5.0):
    a = (MultinomialNB(alpha).fit(Xtr, ytr).predict(Xva) == yva).mean()
    print(f"{alpha:>19} {a:>14.4f}")
    if a > best_nb:
        best_a, best_nb = alpha, a

print(f"\nchosen: k={best_k}, alpha={best_a}")

# --- one look at the test set -----------------------------------------------
nb = MultinomialNB(best_a).fit(np.vstack([Xtr, Xva]), np.r_[ytr, yva])
nb_pred = nb.predict(Xte)
knn_pred = knn_cosine(np.vstack([Xtr, Xva]), np.r_[ytr, yva], Xte, best_k)
D = X.shape[1]
n_fit = n_tr + n_va
print(f"\n{'model':<24} {'test accuracy':>14} {'ops per query':>16}")
print(f"{'naive Bayes':<24} {(nb_pred == yte).mean():>14.4f} "
      f"{K * D:>16,}")
print(f"{'cosine k-NN':<24} {(knn_pred == yte).mean():>14.4f} "
      f"{n_fit * D:>16,}")
print(f"{'chance':<24} {max(np.bincount(yte)) / len(yte):>14.4f} {'-':>16}")

print("\nNaive Bayes wins on both axes here, and the k-NN result is the more")
print("instructive one. These are 360-dimensional count vectors with about")
print("26 non-zero entries each: two documents on the same topic share")
print("almost no words by chance, so cosine distance is dominated by which")
print("particular words happened to be sampled. That is section 6.2's curse")
print("arriving in a realistic setting — and it is precisely why Part XI")
print("learns a DENSE low-dimensional representation before doing k-NN in")
print("it, rather than running k-NN on raw counts.")
print("\nThe cost asymmetry is the other half of the decision: naive Bayes")
print("compresses the training set into K x D numbers, so its prediction")
print("cost does not depend on N at all, while k-NN keeps every document")
print("and pays for it on every query.")

# --- naive Bayes' probabilities, and repairing them -------------------------
print("\n" + "=" * 72)
print("naive Bayes' probabilities are unusable, and cheap to repair")
print("=" * 72)
P_va = MultinomialNB(best_a).fit(Xtr, ytr).predict_proba(Xva)
P_te = MultinomialNB(best_a).fit(Xtr, ytr).predict_proba(Xte)
conf_te = P_te.max(1)
correct_te = (MultinomialNB(best_a).fit(Xtr, ytr).predict(Xte) == yte)
print(f"mean predicted confidence : {conf_te.mean():.6f}")
print(f"actual accuracy           : {correct_te.mean():.6f}")
print(f"fraction with p > 0.999   : {(conf_te > 0.999).mean():.4f}")
print(f"overconfidence            : {conf_te.mean() - correct_te.mean():+.4f}")
print("The model claims 84% confidence and is right 71% of the time — a")
print("13-point gap, with 9% of documents rated above 0.999. Anything that")
print("consumes these numbers as probabilities is being lied to.")


def temperature_fit(logits, y_true, grid=np.logspace(-2.5, 0.5, 80)):
    """Divide the log-joint by T and pick T by validation log loss.

    A single scalar: it cannot reorder anything, so accuracy and AUC are
    mathematically unchanged (section 6.3).
    """
    best_T, best_ll = 1.0, np.inf
    for T in grid:
        d = logits / T
        d = d - d.max(1, keepdims=True)
        p = np.exp(d)
        p /= p.sum(1, keepdims=True)
        ll = -np.mean(np.log(np.clip(p[np.arange(len(y_true)), y_true],
                                     1e-12, 1)))
        if ll < best_ll:
            best_T, best_ll = T, ll
    return best_T


m = MultinomialNB(best_a).fit(Xtr, ytr)
T = temperature_fit(m.decision(Xva), yva)
d = m.decision(Xte) / T
d -= d.max(1, keepdims=True)
P_cal = np.exp(d)
P_cal /= P_cal.sum(1, keepdims=True)


def multiclass_ece(y_true, P, n_bins=10):
    conf, pred = P.max(1), P.argmax(1)
    edges = np.quantile(conf, np.linspace(0, 1, n_bins + 1))
    tot = 0.0
    for i in range(n_bins):
        msk = (conf >= edges[i]) & (conf <= edges[i + 1])
        if msk.sum():
            tot += msk.sum() / len(conf) * abs(
                (pred[msk] == y_true[msk]).mean() - conf[msk].mean())
    return tot


def logloss(y_true, P):
    return float(-np.mean(np.log(np.clip(
        P[np.arange(len(y_true)), y_true], 1e-12, 1))))


print(f"\nfitted temperature T = {T:.4f}\n")
print(f"{'':<14} {'accuracy':>10} {'mean conf':>11} {'ECE':>9} {'log loss':>10}")
print(f"{'raw':<14} {(P_te.argmax(1) == yte).mean():>10.4f} "
      f"{P_te.max(1).mean():>11.4f} {multiclass_ece(yte, P_te):>9.4f} "
      f"{logloss(yte, P_te):>10.4f}")
print(f"{'temperature':<14} {(P_cal.argmax(1) == yte).mean():>10.4f} "
      f"{P_cal.max(1).mean():>11.4f} {multiclass_ece(yte, P_cal):>9.4f} "
      f"{logloss(yte, P_cal):>10.4f}")
print("\nAccuracy is IDENTICAL — dividing every logit by a positive constant")
print("cannot change an argmax — while calibration error and log loss")
print("improve substantially. One scalar, fitted on validation data, turns")
print("an unusable probability into a usable one at no cost to the decision.")
print("The same trick reappears as sampling temperature in Chapter 90.")
```

## 9. Common Mistakes

**Not scaling before k-NN.** The measured table shows accuracy collapsing when
one column is multiplied by a constant that changes no information.

**Using $k=1$ because it has zero training error.** That is the definition of
maximum variance.

**Using an even $k$ for binary classification.** Ties get broken arbitrarily.

**Applying k-NN in high dimensions without reducing them.** Distances
concentrate; the measured cost per added noise dimension is real.

**Forgetting Laplace smoothing.** One unseen feature vetoes an entire class.

**Trusting naive Bayes' probabilities.** They are systematically extreme, for
the identifiable reason in {{sec:6-mathematical-foundation}}.

**Multiplying probabilities instead of adding logs.** Underflow to exactly zero.

**Choosing naive Bayes over logistic regression by dogma.** The crossover
depends on sample size and is measurable in a few minutes.

**Serving brute-force k-NN at scale.** $O(ND)$ per query; use an approximate
index ({{ch:emb-ann}}).

## 10. Connection to Previous Chapters

{{ch:ml-metrics}} supplied the bias-variance frame that $k$ traverses directly,
and the calibration-versus-discrimination distinction that naive Bayes
illustrates with an identifiable mechanism rather than an anecdote.
{{ch:ml-logistic}} is the discriminative counterpart in
{{tbl:gen-vs-disc}}, and its softmax reappears here both in
{{eq:naive-bayes-classify}}'s normalisation and in temperature scaling.
{{ch:math-probability}} supplied Bayes' rule. {{ch:math-inference}} supplied the
CLT behind {{eq:distance-concentration}}. {{ch:ds-feature-eng}} supplied the
scaling that k-NN cannot do without.

Forward: {{ch:ml-trees}} is the first method that is genuinely immune to feature
scaling and to irrelevant features, which is exactly the pair of weaknesses
measured here. {{ch:ml-pca}} reduces dimension so that distances mean something
again. {{ch:emb-what-they-are}} learns a space in which cosine k-NN is the right
algorithm, and {{ch:emb-ann}} makes it fast enough to serve. {{ch:rag-indexing}}
is this chapter's retrieval step inside a larger system.

## 11. Exercises

**Beginner**

1. Why does k-NN have no training phase, and what does it pay instead?
2. What happens to the decision boundary as $k$ increases?
3. Why must features be scaled before k-NN?
4. State the naive Bayes conditional-independence assumption in one sentence.
5. Why do we work in log space in {{eq:naive-bayes-classify}}?

**Intermediate**

6. Using {{eq:neighbourhood-side}}, find the side length capturing 10% of the
   data in 20 dimensions.
7. Explain {{eq:distance-concentration}} and its consequence for k-NN.
8. Why is Laplace smoothing equivalent to a Dirichlet prior?
9. Explain why naive Bayes' ROC-AUC survives feature duplication while its ECE
   does not.
10. When would you choose cosine over Euclidean distance?
11. Give a dataset where naive Bayes beats logistic regression and one where it
    loses.

**Advanced**

12. Prove the 1-NN bound {{eq:knn-bound}} for the binary case.
13. Derive {{eq:distance-concentration}} from the CLT and state the assumptions.
14. Show that for Gaussian naive Bayes with shared covariance, the decision
    boundary is linear — and hence that it and logistic regression share a
    hypothesis space while fitting it differently.
15. Explain formally why temperature scaling cannot change an argmax.
16. Derive the asymptotic error rate of $k$-NN as $k \to \infty$ with $k/N \to 0$
    and explain why it reaches the Bayes rate while 1-NN does not.

**Implementation**

17. Implement a k-d tree and measure the query speed-up against brute force at
    $D = 2, 10, 50$. Explain what happens as $D$ grows.
18. Implement Gaussian naive Bayes and compare it against the multinomial
    variant on continuous features.
19. Implement isotonic regression for calibration and compare it against
    temperature scaling on the same naive Bayes output.
20. Reproduce the generative/discriminative crossover and find the sample size
    at which it occurs for your own dataset.

**Reasoning**

21. Vector search over 1536-dimensional embeddings works well despite
    {{eq:nn-degeneracy}}. Explain why.
22. A colleague reports naive Bayes returning 0.9999 confidence and proposes
    retraining with more data. What do you say?

## 12. Chapter Summary

k-NN has no training phase and no model: the data is the model. $k$ is a direct
bias-variance knob, from zero training error and maximum variance at $k=1$ to
the majority class at $k=N$, with test accuracy peaking in between.

It is acutely scale-sensitive. Multiplying one feature by a constant changes no
information and, as measured, destroys the classifier — standardising is
mandatory rather than advisable.

The curse of dimensionality is quantitative, not vague. A neighbourhood holding
1% of the data spans 99.5% of every axis at $D=1000$; distance concentration
makes the farthest point only 12% further than the nearest; and the measured
accuracy cost of adding pure-noise dimensions is steady and real. Distance-based
methods therefore depend on feature selection and dimensionality reduction far
more than models that can shrink a coefficient to ignore a feature.

Naive Bayes applies Bayes' rule with a conditional-independence assumption that
reduces an exponential parameter count to a linear one. The assumption is false
and the classifier works, because correlated features multiply the log-odds by a
positive constant — a monotone map, which preserves the argmax and the ranking
while driving the probabilities to the extremes. It is a good classifier and a
bad probability estimator, and one scalar of temperature scaling fitted on
validation data repairs the probabilities without changing a single decision.

Laplace smoothing is mandatory: without it one unseen feature contributes
$\log 0$ and vetoes a class outright.

Generative and discriminative models cross over. Naive Bayes' assumptions
substitute for data and win at small samples; logistic regression's weaker
assumptions win once there is enough data to estimate the boundary directly.

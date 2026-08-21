---
id: math-covariance
number: 9
part: I
tier: focused
status: reviewed
requires: [math-random-vars, math-vectors]
provides: [variance, standard-deviation, covariance, correlation,
           covariance-matrix, standardisation]
citations: [deisenroth2020]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Compute variance and standard deviation, and explain why variance is defined
   with a square.
2. Derive the computational identity $\Var(X) = \E[X^{2}] - \E[X]^{2}$ and state
   when it is numerically unsafe.
3. Explain why variances add for independent variables but expectations add
   always.
4. Compute covariance and correlation and state precisely what each measures.
5. Explain why zero correlation does not imply independence, and construct a
   counterexample.
6. Build and interpret a covariance matrix, and relate its eigenvectors to
   directions of variation.
7. Derive $\Var(\vec{q}\T\vec{k}) = d_k$ for independent unit-variance
   components, and explain its two consequences later in the book.
8. Standardise features and explain when it is necessary.

## 2. Why This Matters

Expectation tells you where a distribution sits. Variance tells you how much it
moves, and covariance tells you how quantities move together. Almost every
practical question in machine learning is about the second and third of these.

**Variance is the second half of the bias-variance decomposition**, which is the
central framework for reasoning about generalisation
({{ch:ml-metrics}}). A model that memorises its training data has low bias and
ruinous variance; understanding that trade-off requires understanding what
variance is.

**Covariance structure is what dimensionality reduction exploits.** PCA finds
the eigenvectors of the covariance matrix, and the reason it works is that real
data has features that move together, so the covariance matrix is far from
diagonal ({{ch:ml-pca}}).

**Correlation is the most misused statistic in the field.** Zero correlation is
routinely read as "no relationship", which is false; and nonzero correlation is
routinely read as causation, which is worse. {{ch:ds-causation}} takes up the
second error; this chapter deals with the first.

And one specific result in {{sec:6-mathematical-foundation}} — that the variance
of a dot product of $d$-dimensional random vectors is $d$ — is used twice later
in this book, in places that look unrelated. It is why attention divides by
$\sqrt{d_k}$ ({{ch:tf-scaled-dot-product}}) and why weight initialisation scales
by $1/\sqrt{\text{fan-in}}$ ({{ch:dl-initialization}}). It is stated here as a
theorem rather than left to be rediscovered twice.

## 3. Prerequisites

{{ch:math-random-vars}} for random variables, expectation and its linearity.
{{ch:math-vectors}} for dot products, and {{ch:math-matrices}} for matrices —
the covariance matrix needs both.

## 4. Intuitive Explanation

### 4.1 Variance measures spread

Two datasets can share a mean and be nothing alike. $\{50, 50, 50\}$ and
$\{0, 50, 100\}$ both average 50. The second is far more variable, and
{{term:variance}} is how that difference is quantified.

The construction: take each value's deviation from the mean, square it, and
average.

$$
\Var(X) = \E\big[(X - \E[X])^{2}\big]
$$ (eq:variance-def)

Why square rather than take absolute values? Three reasons, in ascending order
of importance.

Squaring makes deviations positive, so they cannot cancel — but so does the
absolute value.

Squaring penalises large deviations disproportionately, which is often what you
want.

The real reason is that squares are differentiable everywhere and absolute
values are not, and that variances of independent quantities **add**. The
mean absolute deviation is a perfectly reasonable measure of spread with none of
that algebraic structure, which is why it is rarely used despite being more
robust to outliers.

The {{term:standard-deviation}} is $\sigma = \sqrt{\Var(X)}$. It exists because
variance has squared units — a variance of house prices is in
pounds-squared, which means nothing — and taking the root restores
interpretability.

### 4.2 Covariance measures moving together

{{term:covariance}} extends variance to two quantities. Where variance asks "how
far does $X$ stray from its mean", covariance asks "when $X$ is above its mean,
is $Y$ also?"

$$
\Cov(X, Y) = \E\big[(X - \E[X])(Y - \E[Y])\big]
$$ (eq:covariance-def)

The product inside is positive when both deviations share a sign — both above
their means, or both below — and negative when they disagree. Averaging over the
distribution gives a positive number when the variables tend to move together
and a negative one when they move oppositely.

Note immediately that $\Cov(X, X) = \Var(X)$: variance is the covariance of a
variable with itself.

### 4.3 Correlation is covariance made comparable

Covariance has a serious practical flaw: its units are the product of the two
variables' units, so its magnitude is uninterpretable. Covariance of height in
metres with weight in kilograms is in metre-kilograms, and switching to
centimetres multiplies it by 100 without anything real having changed.

{{term:correlation}} fixes this by dividing out both standard deviations:

$$
\rho_{XY} = \frac{\Cov(X, Y)}{\sigma_X \sigma_Y}
$$ (eq:correlation-def)

The result always lies in $[-1, 1]$, is dimensionless, and is unaffected by the
units. It is exactly the cosine similarity of {{ch:math-norms}} applied to
mean-centred data, a correspondence made precise in
{{sec:6-mathematical-foundation}}.

> IMPORTANT: Correlation measures **linear** association and nothing else. Two
> variables can be perfectly, deterministically related and have correlation
> exactly zero. {{sec:6-mathematical-foundation}} constructs the standard
> example. Reading $\rho = 0$ as "unrelated" is one of the most common and most
> consequential errors in applied statistics.

### 4.4 The covariance matrix holds all the pairs

With many variables, all the pairwise covariances go in a matrix:

$$
\Sigma_{ij} = \Cov(X_i, X_j)
$$ (eq:covariance-matrix-def)

The diagonal holds the variances, since $\Cov(X_i, X_i) = \Var(X_i)$. The matrix
is symmetric, because covariance is.

The {{term:covariance-matrix}} is where {{ch:math-eigen}} pays off. Because it is
symmetric and positive semi-definite, the spectral theorem applies: it has
orthogonal eigenvectors, and those eigenvectors are the directions along which
the data actually varies, with the eigenvalues giving the variance along each.
That is PCA, in one sentence ({{ch:ml-pca}}).

## 5. Formal Explanation

### 5.1 Variance

$$
\Var(X) = \E\big[(X - \mu)^{2}\big], \qquad \mu = \E[X]
$$ (eq:variance)

The **computational form**, derived in {{sec:6-mathematical-foundation}}:

$$
\Var(X) = \E[X^{2}] - \E[X]^{2}
$$ (eq:variance-computational)

Properties, for constants $a$ and $b$:

$$
\Var(aX + b) = a^{2}\Var(X)
$$ (eq:variance-scaling)

The $b$ vanishes — shifting a distribution does not change its spread — and the
$a$ is squared, because variance is in squared units. Consequently
$\sigma_{aX+b} = \lvert a \rvert \sigma_X$.

For a **sum**:

$$
\Var(X + Y) = \Var(X) + \Var(Y) + 2\Cov(X, Y)
$$ (eq:variance-sum)

and therefore, **only when $X$ and $Y$ are uncorrelated**:

$$
\Var(X + Y) = \Var(X) + \Var(Y)
$$ (eq:variance-sum-independent)

> IMPORTANT: Compare this with {{eq:linearity-expectation}}. Expectation adds
> unconditionally; variance adds only when the covariance term vanishes. That
> asymmetry is the reason correlated errors are so damaging — averaging $n$
> independent estimates cuts the variance by $n$, but averaging $n$ correlated
> ones does not. It is why ensembles help ({{ch:ml-forests}}), why correlated
> validation folds give overconfident estimates ({{ch:mle-splits}}), and why
> A/B tests on non-independent units need larger samples
> ({{ch:ds-experiments}}).

### 5.2 Covariance and correlation

$$
\Cov(X, Y) = \E[(X - \mu_X)(Y - \mu_Y)] = \E[XY] - \E[X]\E[Y]
$$ (eq:covariance)

The second form follows by expanding, exactly as {{eq:variance-computational}}
does.

Properties: covariance is symmetric, bilinear, and satisfies
$\Cov(X, X) = \Var(X)$. If $X$ and $Y$ are independent then
$\E[XY] = \E[X]\E[Y]$ by {{eq:expectation-product}}, so $\Cov(X, Y) = 0$.

**The converse is false.** Zero covariance does not imply independence, for the
reason that covariance detects only linear association.

Correlation is defined by {{eq:correlation-def}}, and $\lvert\rho\rvert \le 1$
follows from Cauchy-Schwarz ({{ch:math-vectors}}). Values of $\pm 1$ occur
exactly when $Y$ is an exact linear function of $X$.

### 5.3 The covariance matrix

For a random vector $\vec{X} \in \R^{d}$ with mean $\vec{\mu}$:

$$
\mat{\Sigma} = \E\big[(\vec{X} - \vec{\mu})(\vec{X} - \vec{\mu})\T\big]
$$ (eq:covariance-matrix)

Note this is an *outer* product, giving a $d \times d$ matrix, not the inner
product that would give a scalar.

$\mat{\Sigma}$ is symmetric and positive semi-definite. The latter has a
concrete meaning: for any direction $\vec{w}$,

$$
\Var(\vec{w}\T\vec{X}) = \vec{w}\T\mat{\Sigma}\vec{w} \ge 0
$$ (eq:variance-projection)

The variance of the data projected onto any direction is non-negative, which it
must be. {{eq:variance-projection}} is also the key to PCA: maximising it over
unit $\vec{w}$ gives the top eigenvector of $\mat{\Sigma}$, which is the
direction of greatest variance.

For a data matrix $\mat{X} \in \R^{n \times d}$ with rows as observations, the
sample covariance is

$$
\hat{\mat{\Sigma}} = \frac{1}{n-1}\mat{X}_c\T\mat{X}_c
$$ (eq:sample-covariance)

where $\mat{X}_c$ is $\mat{X}$ with the column means subtracted. The $n-1$ is
**Bessel's correction**: dividing by $n$ underestimates the variance, because
the deviations are measured from the *sample* mean rather than the true mean,
and the sample mean is by construction the point that minimises them.
{{ch:math-inference}} makes this precise.

### 5.4 Standardisation

{{term:standardisation}} rescales a variable to mean 0 and variance 1:

$$
Z = \frac{X - \mu}{\sigma}
$$ (eq:standardise)

By {{eq:variance-scaling}}, $\E[Z] = 0$ and $\Var(Z) = 1$.

This matters whenever an algorithm compares features against each other. A
distance measure ({{ch:math-norms}}) treats a feature measured in millimetres as
a thousand times more important than the same feature in metres. Gradient
descent on unstandardised features gives an ill-conditioned problem and zigzags
({{ch:math-optimization}}). Regularisation penalises large coefficients, which
punishes features that happen to be measured in small units.

Tree-based methods are the notable exception: they split on thresholds within
one feature at a time and never compare across features, so scaling has no
effect on them ({{ch:ml-trees}}).

## 6. Mathematical Foundation

### 6.1 The computational form of variance

Expand {{eq:variance}} and use linearity:

$$
\Var(X) = \E[(X - \mu)^{2}] = \E[X^{2} - 2\mu X + \mu^{2}]
$$

Linearity of expectation splits this into three terms. Since $\mu$ is a
constant, $\E[2\mu X] = 2\mu\E[X] = 2\mu^{2}$ and $\E[\mu^{2}] = \mu^{2}$:

$$
\Var(X) = \E[X^{2}] - 2\mu^{2} + \mu^{2} = \E[X^{2}] - \mu^{2}
$$ (eq:variance-derivation)

which is {{eq:variance-computational}}.

> WARNING: {{eq:variance-computational}} is convenient — one pass through the
> data, accumulating $\sum x$ and $\sum x^{2}$ — and numerically dangerous. When
> the mean is large relative to the spread, $\E[X^{2}]$ and $\E[X]^{2}$ are
> nearly equal, and subtracting them is catastrophic cancellation: you lose most
> of your significant digits, and can even get a negative variance. For values
> around $10^{9}$ with a spread of 1, the naive formula fails in single
> precision. Use a two-pass algorithm or Welford's online method.
> {{sec:7-implementation}} demonstrates the failure.

### 6.2 Why variance needs independence to add

Expand $\Var(X + Y)$ using {{eq:variance}} with mean $\mu_X + \mu_Y$:

$$
\Var(X+Y) = \E\big[((X - \mu_X) + (Y - \mu_Y))^{2}\big]
$$

Expanding the square inside gives three terms:

$$
= \E[(X-\mu_X)^{2}] + \E[(Y-\mu_Y)^{2}] + 2\,\E[(X-\mu_X)(Y-\mu_Y)]
$$

The first two are the variances; the third is twice the covariance by
{{eq:covariance-def}}. That gives {{eq:variance-sum}}. The cross term vanishes
exactly when the covariance is zero, which is what independence guarantees.

The reason expectation had no such term is that expectation involves no product
of the two variables — the algebra never generates a cross term to begin with.

### 6.3 Zero correlation without independence

Let $X$ be uniform on $\{-2, -1, 0, 1, 2\}$ and let $Y = X^{2}$.

$Y$ is a deterministic function of $X$: knowing $X$ tells you $Y$ exactly. They
could not be more dependent.

Compute the covariance. By symmetry $\E[X] = 0$. Then

$$
\Cov(X, Y) = \E[XY] - \E[X]\E[Y] = \E[X^{3}] - 0 = \E[X^{3}]
$$

and $\E[X^{3}] = \frac{1}{5}((-8) + (-1) + 0 + 1 + 8) = 0$ by symmetry.

So $\Cov(X, Y) = 0$ and $\rho = 0$, despite a perfect deterministic
relationship.

The reason is visible in the geometry: $Y = X^{2}$ is a parabola, symmetric
about the vertical axis. For every point on the rising branch there is a
mirror-image point on the falling branch, and their contributions to the
covariance cancel exactly. Covariance measures the *linear* tendency, and a
symmetric parabola has none.

> IMPORTANT: This is not an artificial construction. Any symmetric nonlinear
> relationship produces it — a feature that helps at moderate values and hurts
> at extremes, a latency that rises at both very low and very high load. If you
> screen features by correlation with the target, you will silently discard
> every such feature. Plot the data.

### 6.4 The variance of a dot product

This is the result the chapter exists for.

Let $\vec{q}, \vec{k} \in \R^{d}$ have components that are independent with mean
0 and variance 1. What is the distribution of $\vec{q}\T\vec{k}$?

For the mean, use linearity and then independence within each term:

$$
\E[\vec{q}\T\vec{k}] = \E\left[\sum_{i=1}^{d} q_i k_i\right]
  = \sum_{i=1}^{d}\E[q_i]\E[k_i] = 0
$$ (eq:dot-mean-derived)

For the variance, the terms $q_i k_i$ are independent across $i$, so by
{{eq:variance-sum-independent}} the variances add:

$$
\Var(\vec{q}\T\vec{k}) = \sum_{i=1}^{d}\Var(q_i k_i)
$$

Each term, using {{eq:variance-computational}} and the fact that the product has
mean zero:

$$
\Var(q_i k_i) = \E[q_i^{2}k_i^{2}] - 0 = \E[q_i^{2}]\,\E[k_i^{2}] = 1 \cdot 1 = 1
$$

Hence

$$
\Var(\vec{q}\T\vec{k}) = d, \qquad \text{sd}(\vec{q}\T\vec{k}) = \sqrt{d}
$$ (eq:dot-variance-d)

**The spread of a dot product grows as the square root of the dimension.**

This one result has two consequences that look unrelated:

**Attention.** Raw attention scores at $d_k = 64$ have standard deviation 8.
Exponentiating scores spread over $\pm 8$ gives a softmax that is effectively
one-hot before training begins, whose gradient is zero. Dividing by $\sqrt{d_k}$
restores unit variance. That is the entire justification for the scaling factor
in {{ch:tf-scaled-dot-product}}.

**Initialisation.** A neuron's pre-activation is a dot product of its input with
its weight vector. If weights are drawn with variance 1, the pre-activation has
variance equal to the fan-in, and signals amplify by $\sqrt{\text{fan-in}}$ at
every layer — exploding through depth. Scaling the initialisation variance by
$1/\text{fan-in}$ keeps the signal stationary, which is exactly what Xavier and
He initialisation do ({{ch:dl-initialization}}).

Two apparently distinct pieces of deep-learning folklore, one theorem.

### 6.5 Correlation is cosine similarity of centred data

Let $\vec{x}_c$ and $\vec{y}_c$ be the mean-centred data vectors. Then the
sample covariance is $\vec{x}_c\T\vec{y}_c/(n-1)$ and each sample standard
deviation is $\norm{\vec{x}_c}/\sqrt{n-1}$. Substituting into
{{eq:correlation-def}}, every $\sqrt{n-1}$ cancels:

$$
\hat{\rho} = \frac{\vec{x}_c\T\vec{y}_c}{\norm{\vec{x}_c}\,\norm{\vec{y}_c}}
= \cos\theta
$$ (eq:correlation-cosine)

Correlation *is* the cosine of the angle between the centred data vectors. This
is why $\lvert\rho\rvert \le 1$ — it is Cauchy-Schwarz — and it also explains
why correlation is scale-invariant but not shift-invariant in the way cosine
similarity is: the centring step is what handles the shift.

## 7. Implementation

```python {tier=A name=variance-covariance}
"""Variance, covariance, correlation, and the dot-product variance result.

Includes the catastrophic-cancellation failure of the naive variance formula
and the zero-correlation-with-perfect-dependence counterexample.
"""
import numpy as np

rng = np.random.default_rng(0)

# --- variance and standard deviation ----------------------------------------
a = np.array([50.0, 50.0, 50.0])
b = np.array([0.0, 50.0, 100.0])
print(f"{'data':<20} {'mean':>7} {'var':>10} {'sd':>8}")
for name, d in (("[50, 50, 50]", a), ("[0, 50, 100]", b)):
    print(f"{name:<20} {d.mean():>7.1f} {d.var():>10.1f} {d.std():>8.2f}")

# --- eq. 9.11: the computational form, and where it breaks ------------------
x = rng.normal(100.0, 5.0, 10_000)
print(f"\ntwo-pass  : {np.mean((x - x.mean())**2):.10f}")
print(f"E[X^2]-E[X]^2: {np.mean(x**2) - x.mean()**2:.10f}   (agree here)")

# Now shift the data far from zero, in single precision.
big = (x + 1e8).astype(np.float32)
naive = np.mean(big**2) - np.mean(big)**2
twopass = np.mean((big - big.mean())**2)
print(f"\nsame data shifted by 1e8, in float32:")
print(f"  naive E[X^2]-E[X]^2 : {naive:>14.4f}   <- catastrophic cancellation")
print(f"  two-pass            : {twopass:>14.4f}")
print(f"  true variance       : {25.0:>14.4f}")
print("The naive form is one pass and numerically unusable at scale.")

# --- eq. 9.14: variance adds only without covariance ------------------------
n = 200_000
u = rng.normal(0, 1, n)
v = rng.normal(0, 1, n)                 # independent of u
w = 0.8 * u + 0.6 * rng.normal(0, 1, n)  # correlated with u

print(f"\n{'pair':<22} {'Var(X)+Var(Y)':>14} {'Var(X+Y)':>10} {'2Cov':>8}")
for name, (p, q) in (("independent", (u, v)), ("correlated", (u, w))):
    print(f"{name:<22} {p.var()+q.var():>14.4f} {(p+q).var():>10.4f} "
          f"{2*np.cov(p, q)[0,1]:>8.4f}")
print("The gap is exactly 2Cov(X, Y) — eq. 9.14.")

# --- section 6.3: zero correlation, perfect dependence ----------------------
X = np.array([-2, -1, 0, 1, 2], dtype=float)
Y = X ** 2
print(f"\nX = {X}\nY = X^2 = {Y}")
print(f"correlation: {np.corrcoef(X, Y)[0,1]:.10f}")
print("Y is a deterministic function of X, yet the correlation is zero.")
print("Correlation measures LINEAR association only.")

# A mutual-information-style check confirms they are far from independent:
# knowing |X| determines Y exactly.
print(f"knowing X determines Y: {np.array_equal(Y, X**2)}")

# --- eq. 9.20: correlation is cosine similarity of centred data -------------
p = rng.normal(size=500)
q = 0.5 * p + rng.normal(size=500)
pc, qc = p - p.mean(), q - q.mean()
cosine_centred = (pc @ qc) / (np.linalg.norm(pc) * np.linalg.norm(qc))
print(f"\ncorrelation        : {np.corrcoef(p, q)[0,1]:.10f}")
print(f"cosine of centred  : {cosine_centred:.10f}   <- identical (eq. 9.20)")
assert np.isclose(np.corrcoef(p, q)[0, 1], cosine_centred)

# --- eq. 9.19: the variance of a dot product is d ---------------------------
print(f"\n{'d':>7} {'Var(q.k) sim':>14} {'predicted = d':>15} {'sd':>9} "
      f"{'sqrt(d)':>9}")
for d in (2, 8, 64, 256, 1024):
    qs = rng.normal(size=(60_000, d))
    ks = rng.normal(size=(60_000, d))
    dots = (qs * ks).sum(axis=1)
    print(f"{d:>7} {dots.var():>14.2f} {d:>15} {dots.std():>9.3f} "
          f"{np.sqrt(d):>9.3f}")
print("\nThe spread grows as sqrt(d). This single fact justifies BOTH the")
print("1/sqrt(d_k) in attention and the 1/sqrt(fan_in) in initialisation.")

# --- the covariance matrix and its eigenvectors -----------------------------
true_cov = np.array([[4.0, 3.0], [3.0, 9.0]])
L = np.linalg.cholesky(true_cov)
data = rng.normal(size=(50_000, 2)) @ L.T

S = np.cov(data, rowvar=False)
print(f"\nsample covariance matrix:\n{np.round(S, 3)}")
print(f"true:\n{true_cov}")

vals, vecs = np.linalg.eigh(S)
order = np.argsort(-vals)
vals, vecs = vals[order], vecs[:, order]
print(f"\neigenvalues (variance along each principal direction): "
      f"{np.round(vals, 3)}")
print(f"top eigenvector (direction of greatest spread): {np.round(vecs[:,0], 3)}")

# eq. 9.17: variance along a direction is w^T Sigma w
w = vecs[:, 0]
print(f"variance of the projection onto it : "
      f"{np.var(data @ w):.4f}")
print(f"w^T Sigma w                        : {w @ S @ w:.4f}   <- eq. 9.17")

# No direction has more variance than the top eigenvector — that is PCA.
best = max((np.var(data @ np.array([np.cos(t), np.sin(t)])), t)
           for t in np.linspace(0, np.pi, 2000))
print(f"best variance over 2000 random directions: {best[0]:.4f} "
      f"(top eigenvalue {vals[0]:.4f})")

# --- eq. 9.18: standardisation ----------------------------------------------
raw = np.column_stack([rng.normal(1000, 200, 5000),    # e.g. price
                       rng.normal(3, 0.5, 5000)])       # e.g. rating
z = (raw - raw.mean(axis=0)) / raw.std(axis=0)
print(f"\nraw column sds : {np.round(raw.std(axis=0), 3)}")
print(f"standardised   : {np.round(z.std(axis=0), 6)}")
print(f"raw covariance matrix condition number : "
      f"{np.linalg.cond(np.cov(raw, rowvar=False)):.1f}")
print(f"standardised                           : "
      f"{np.linalg.cond(np.cov(z, rowvar=False)):.1f}")
print("Standardising fixes the conditioning — which is why gradient descent")
print("on unscaled features zigzags (Chapter 12).")
```

## 8. Practical Example

Feature correlation analysis is the first thing done to any tabular dataset, and
it is where the limits of correlation become concrete.

```python {tier=A name=feature-correlation}
"""Screening features by correlation — and the features it silently misses.

A realistic mixture: linearly related features, redundant features, a
nonlinearly related one, and pure noise.
"""
import numpy as np

rng = np.random.default_rng(4)
n = 4000

age = rng.uniform(18, 80, n)
income = 20_000 + 900 * age + rng.normal(0, 12_000, n)     # linear in age
income_k = income / 1000                                    # redundant, rescaled
distance = rng.uniform(-1, 1, n)                            # symmetric
noise = rng.normal(0, 1, n)

# The target depends on income linearly AND on distance quadratically.
target = 0.4 * (income / 1000) + 30 * distance**2 + rng.normal(0, 4, n)

features = {"age": age, "income": income, "income_k": income_k,
            "distance": distance, "noise": noise}

print(f"{'feature':<12} {'corr with target':>18} {'|corr|':>8}")
for name, f in features.items():
    r = np.corrcoef(f, target)[0, 1]
    print(f"{name:<12} {r:>18.4f} {abs(r):>8.4f}")

print("\n'distance' has near-zero correlation but drives 30*distance^2.")
print("A correlation screen would discard the second-strongest predictor.")

# Squaring it first reveals the relationship immediately.
r2 = np.corrcoef(distance**2, target)[0, 1]
print(f"corr(distance^2, target) = {r2:.4f}  <- there it is")

# --- multicollinearity: income and income_k are the same feature ------------
names = list(features)
M = np.corrcoef(np.stack([features[k] for k in names]))
print(f"\nfeature-feature correlation matrix:")
print(f"{'':<11}" + "".join(f"{k:>10}" for k in names))
for i, k in enumerate(names):
    print(f"{k:<11}" + "".join(f"{M[i,j]:>10.3f}" for j in range(len(names))))

print("\nincome and income_k correlate at 1.000 — perfectly redundant.")
print("A linear model cannot separate their coefficients; the design matrix")
print("is rank-deficient (Chapter 4) and the fit is unstable (Chapter 32).")

# A quick diagnostic: the condition number of the feature covariance matrix.
X = np.stack([features[k] for k in names], axis=1)
Xz = (X - X.mean(0)) / X.std(0)
print(f"\ncondition number of the standardised covariance matrix: "
      f"{np.linalg.cond(np.cov(Xz, rowvar=False)):.3e}")
print("Very large — the signature of multicollinearity (Chapter 6).")

X2 = np.stack([features[k] for k in names if k != "income_k"], axis=1)
X2z = (X2 - X2.mean(0)) / X2.std(0)
print(f"after dropping the redundant column: "
      f"{np.linalg.cond(np.cov(X2z, rowvar=False)):.2f}")
```

> PRODUCTION TIP: A correlation matrix is a useful first look and a terrible
> last word. Use it to spot redundant features — pairs near $\pm 1$ — and to
> catch obvious leakage, where a feature correlates suspiciously highly with the
> target. Do not use it to select features, because it is blind to every
> nonlinear relationship and every interaction. Mutual information, or simply
> fitting a flexible model and reading its feature importances, catches what
> correlation cannot ({{ch:ds-feature-eng}}).

## 9. Common Mistakes

**Reading zero correlation as no relationship.** It means no *linear*
relationship. Plot the data.

**Reading correlation as causation.** {{ch:ds-causation}} is devoted to this.

**Using the naive variance formula on large-magnitude data.** Catastrophic
cancellation, demonstrated in {{sec:7-implementation}}. Use two-pass or
Welford's algorithm.

**Adding variances of correlated quantities.** {{eq:variance-sum}} has a cross
term. Ignoring it is why correlated validation folds and clustered A/B
experimental units produce confidence intervals that are far too narrow.

**Comparing covariances across different units.** Covariance is not
scale-invariant. Use correlation.

**Forgetting Bessel's correction.** `np.var` defaults to `ddof=0`, dividing by
$n$; `np.cov` defaults to `ddof=1`, dividing by $n-1$. The two NumPy functions
disagree by default, which is a genuine trap.

**Standardising before splitting the data.** Computing the mean and standard
deviation on the full dataset and then splitting leaks test-set information into
training. Fit the scaler on the training set only ({{ch:ds-leakage}}).

**Standardising when the algorithm does not need it.** Tree-based models are
scale-invariant. Standardising them is harmless but pointless, and it destroys
the interpretability of the split thresholds.

## 10. Connection to Previous Chapters

{{ch:math-random-vars}} defined expectation, and this chapter defines variance
and covariance as expectations of particular functions via
{{eq:lotus}}. The contrast between {{eq:linearity-expectation}} and
{{eq:variance-sum}} — expectation always adds, variance only sometimes — is the
central structural fact.

{{ch:math-vectors}} supplied the dot product and cosine similarity, and
{{eq:correlation-cosine}} shows correlation is exactly the latter on centred
data; Cauchy-Schwarz from that chapter is what bounds $\rho$.
{{ch:math-matrices}} supplied the outer product that builds
{{eq:covariance-matrix}}, and {{ch:math-eigen}} supplied the spectral theorem
that makes its eigendecomposition well behaved.

Forward: {{ch:math-inference}} treats the sample mean as a random variable and
needs {{eq:variance-sum-independent}} to derive its standard error.
{{ch:math-optimization}} uses the conditioning improvement from standardisation.

Beyond Part I: {{ch:ml-pca}} is the eigendecomposition of
{{eq:covariance-matrix}}; {{ch:ml-metrics}} decomposes error into bias and
variance; {{ch:tf-scaled-dot-product}} and {{ch:dl-initialization}} both rest on
{{eq:dot-variance-d}}.

## 11. Exercises

**Beginner**

1. Compute the mean, variance and standard deviation of $\{2, 4, 4, 4, 5, 5, 7, 9\}$.
2. If $\Var(X) = 9$, what is $\Var(3X)$? What is $\Var(X + 100)$?
3. Two variables have $\Cov = 12$, $\sigma_X = 3$, $\sigma_Y = 8$. Compute the
   correlation.
4. Standardise the values $\{10, 20, 30\}$.
5. Give the covariance matrix of two uncorrelated variables with variances 4 and
   9.

**Intermediate**

6. Derive {{eq:variance-computational}} from {{eq:variance}}.
7. Show $\Var(aX + b) = a^{2}\Var(X)$ directly from the definition.
8. Two independent random variables each have variance 5. What is the variance
   of their sum? Of their difference?
9. Verify the counterexample of {{sec:6-mathematical-foundation}} by computing
   $\Cov(X, X^{2})$ by hand for $X$ uniform on $\{-2,-1,0,1,2\}$.
10. Explain why `np.var` and `np.cov` give different answers on the same data by
    default.
11. For $d = 512$, what is the standard deviation of a dot product of two
    standard-normal vectors? What does that imply for an unscaled softmax over
    such scores?

**Advanced**

12. Prove {{eq:variance-sum}} from the definitions.
13. Prove that $\lvert\rho\rvert \le 1$ using Cauchy-Schwarz.
14. Prove that the covariance matrix is positive semi-definite, using
    {{eq:variance-projection}}.
15. Derive {{eq:dot-variance-d}} for components with variance $\sigma^{2}$
    rather than 1. What scaling factor would attention need in that case, and
    why is the fixed $1/\sqrt{d_k}$ nonetheless the right engineering choice?
16. Show that Bessel's correction gives an unbiased variance estimator, by
    computing $\E[\sum_i (x_i - \bar{x})^{2}]$.

**Implementation**

17. Implement Welford's online variance algorithm and compare it against the
    naive one-pass formula on data with mean $10^{8}$.
18. Reproduce the dot-product variance experiment and additionally verify that
    the distribution of $\vec{q}\T\vec{k}/\sqrt{d}$ approaches a standard normal
    as $d$ grows. Name the theorem responsible.
19. Generate data with a known covariance matrix using a Cholesky factor,
    recover it from the sample, and measure how the estimation error shrinks
    with $n$.
20. Build a dataset with a purely quadratic relationship and show that
    correlation-based feature selection discards the strongest predictor while
    mutual information retains it.

**Reasoning**

21. An ensemble averages $n$ models. Using {{eq:variance-sum}}, explain how the
    variance of the average depends on the correlation between the models, and
    what that implies about how ensembles should be built.
22. A colleague standardises features using statistics computed over the whole
    dataset, then splits into train and test. What exactly leaks, and how large
    could the effect be?

## 12. Chapter Summary

Variance is the expected squared deviation from the mean, and standard deviation
is its square root, restored to the original units. The square is used not for
robustness — absolute deviation is more robust — but because it is
differentiable and because variances of uncorrelated quantities add.

The computational form $\Var(X) = \E[X^{2}] - \E[X]^{2}$ allows a single pass
but suffers catastrophic cancellation when the mean is large relative to the
spread. Use a two-pass or online algorithm.

Expectation adds unconditionally; variance adds only when the covariance term
vanishes. That asymmetry explains why correlated errors are so damaging, why
ensembles of diverse models help more than ensembles of similar ones, and why
correlated validation folds produce overconfident estimates.

Covariance measures joint movement but has uninterpretable units; correlation
divides out both standard deviations, lands in $[-1,1]$, and is exactly the
cosine similarity of the mean-centred data vectors. Correlation captures linear
association only: a variable and its own square can be perfectly dependent and
have correlation exactly zero.

The covariance matrix collects all pairwise covariances, is symmetric and
positive semi-definite, and its eigenvectors are the directions of greatest
variation — which is PCA.

The variance of a dot product of independent unit-variance $d$-dimensional
vectors is exactly $d$, so its spread grows as $\sqrt{d}$. This single result
justifies both the $1/\sqrt{d_k}$ scaling in attention and the
$1/\sqrt{\text{fan-in}}$ scaling in weight initialisation — two pieces of
apparently unrelated deep-learning practice, one theorem.

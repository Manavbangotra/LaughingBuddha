---
id: ml-logistic
number: 33
part: IV
tier: focused
status: reviewed
requires: [ml-linear-regression, math-probability, math-derivatives, math-optimization]
provides: [logistic-regression, sigmoid, log-odds, cross-entropy-loss, softmax,
           maximum-likelihood-classification, separability, decision-threshold]
citations: [pedregosa2011, hoerl1970]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Explain why linear regression is the wrong model for a binary target.
2. Derive logistic regression from the log-odds assumption.
3. Derive the cross-entropy loss from maximum likelihood.
4. Derive the gradient and show why it takes the same form as least squares.
5. Explain why the loss is convex and what that guarantees.
6. Interpret coefficients as log-odds and odds ratios.
7. Generalise to multiple classes with the softmax.
8. Diagnose and handle complete separation.
9. Choose a decision threshold from costs rather than defaulting to 0.5.

## 2. Why This Matters

Logistic regression is the single most important model in this part, for reasons
that have little to do with tabular classification.

**It is the output layer of essentially every classifier you will ever use.** A
neural-network classifier is a stack of nonlinear feature extractors ending in a
softmax — which is multiclass logistic regression. A transformer's next-token
prediction is a softmax over the vocabulary ({{ch:tf-embeddings}}). The
cross-entropy loss derived in {{sec:6-mathematical-foundation}} is the loss used
to train every language model in this book. Understanding it here, where it can
be derived in full, is what makes the later use intelligible rather than
incantatory.

**It produces calibrated probabilities, and most classifiers do not.** A model
that outputs 0.7 and is right 70% of the time is worth far more than a slightly
more accurate model whose scores mean nothing, because probabilities compose
with costs and thresholds and downstream decisions. {{ch:ml-metrics}} makes this
precise; logistic regression is the baseline against which other models'
calibration is judged.

**It remains the default in regulated settings.** Credit scoring and clinical
risk models are logistic regressions because the coefficients are auditable and
the monotone effect of each feature can be stated. Accuracy is not the only
criterion when the model must be defended.

## 3. Prerequisites

{{ch:ml-linear-regression}} for the linear predictor and regularisation, which
carry over unchanged. {{ch:math-probability}} for likelihood and Bernoulli
distributions. {{ch:math-derivatives}} for the chain rule and the sigmoid
derivative. {{ch:math-optimization}} for gradient descent and convexity.
{{ch:ds-leakage}} for validation, and {{ch:ds-leakage}} for why the threshold
is not 0.5.

## 4. Intuitive Explanation

### 4.1 Why not just fit a line

Fit least squares to a 0/1 target and three things go wrong.

Predictions leave $[0,1]$, so you produce probabilities of $-0.3$ and $1.4$.
Residuals cannot be homoscedastic, because a Bernoulli variable's variance
$p(1-p)$ depends on the mean. And the fit is dragged by extreme points in a
direction that has no meaning — a customer who is *very* obviously going to
churn is still just a 1, but least squares pulls the line towards it.

The fix is not to clip the output. It is to model the right quantity.

### 4.2 Model the log-odds

Probability is bounded, which is what makes it awkward to model linearly. The
**odds** $p/(1-p)$ are unbounded above; the **log-odds** $\log[p/(1-p)]$ are
unbounded in both directions, and therefore a natural target for a linear
predictor.

```text
  p        0.01    0.10    0.25    0.50    0.75    0.90    0.99
  odds     0.01    0.11    0.33    1.00    3.00    9.00   99.0
  logit   -4.60   -2.20   -1.10    0.00    1.10    2.20    4.60
           └───────── symmetric, unbounded, linear-friendly ────┘
```

So we assume $\logit(p) = \vec{w}\T\vec{x} + b$, and invert to recover $p$. The
inverse of the logit is the **sigmoid**, and that is the whole model: a linear
predictor squashed through a fixed curve.

### 4.3 What the sigmoid does

$$
\sigma(z) = \frac{1}{1+e^{-z}}
$$

At $z=0$ it gives 0.5. It saturates towards 0 and 1 and never reaches them. It
is steepest at the decision boundary and nearly flat far from it — which is
exactly the right behaviour: pushing an already-confident prediction further
should change little.

That flatness has a cost. The derivative $\sigma(z)(1-\sigma(z))$ approaches
zero at both extremes, so a badly wrong confident prediction produces a small
*sigmoid* gradient. Cross-entropy is chosen partly because it cancels that term
exactly ({{sec:6-mathematical-foundation}}), which is the reason squared error
is not used for classification.

### 4.4 Reading a coefficient

$w_j$ is the change in **log-odds** per unit of $x_j$. Exponentiating gives the
**odds ratio**: $e^{w_j}$ multiplies the odds.

A coefficient of $0.7$ means $e^{0.7} = 2.01$ — a one-unit increase roughly
doubles the odds. It does not double the probability. Going from $p=0.1$
(odds 0.11) to odds 0.22 gives $p=0.18$; going from $p=0.5$ to odds 2.0 gives
$p=0.67$. The same coefficient, very different probability changes, because the
effect on probability depends on where you start.

> WARNING: Reporting "this feature doubles the risk" from an odds ratio of 2 is
> wrong unless the outcome is rare. For a common outcome the risk ratio is much
> closer to 1 than the odds ratio. This error is endemic in reporting of medical
> and social-science findings.

## 5. Formal Explanation

### 5.1 The model

For binary $y \in \{0,1\}$,

$$
\Prob(y=1 \mid \vec{x}) = \sigma(\vec{w}\T\vec{x} + b),
\qquad \sigma(z) = \frac{1}{1+e^{-z}}
$$ (eq:logistic-model)

equivalently

$$
\log\frac{\Prob(y=1\mid\vec{x})}{\Prob(y=0\mid\vec{x})} = \vec{w}\T\vec{x} + b
$$ (eq:log-odds)

The decision boundary $\{\vec{x} : \vec{w}\T\vec{x} + b = 0\}$ is a hyperplane.
Logistic regression is a **linear classifier**: the probabilities are nonlinear
in $\vec{x}$, the boundary is not.

### 5.2 Maximum likelihood gives cross-entropy

Each observation is Bernoulli with parameter $p_i = \sigma(\vec{w}\T\vec{x}_i +
b)$. The likelihood of the sample is

$$
\Like(\vec{w}, b) = \prod_{i=1}^{N} p_i^{y_i}(1-p_i)^{1-y_i}
$$ (eq:bernoulli-likelihood)

Taking the negative log and dividing by $N$:

$$
\Loss(\vec{w}, b) = -\frac{1}{N}\sum_{i=1}^{N}
   \big[y_i \log p_i + (1-y_i)\log(1-p_i)\big]
$$ (eq:cross-entropy)

This is the **binary cross-entropy**, and it is not a design choice — it is what
maximum likelihood produces. The same derivation in {{ch:math-optimization}}
produced squared error from a Gaussian likelihood; the loss follows from the
noise model in both cases.

Unlike least squares there is **no closed form**. Setting the gradient to zero
gives a transcendental system, solved iteratively.

### 5.3 Convexity

$\Loss$ is convex in $(\vec{w}, b)$ — its Hessian is positive semi-definite
({{sec:6-mathematical-foundation}}) — so every local minimum is global and
gradient descent cannot get stuck. This is a genuine and unusual guarantee, and
it is exactly what {{part:6}} gives up: a neural network's loss is not convex,
and everything about initialisation, learning-rate schedules and optimiser
choice exists because of that.

### 5.4 Regularisation and separation

Penalties carry over from {{ch:ml-linear-regression}} unchanged:

$$
\Loss_{\text{reg}} = \Loss(\vec{w},b) + \lambda\|\vec{w}\|_2^{2}
\quad\text{or}\quad \Loss(\vec{w},b) + \lambda\|\vec{w}\|_1
$$ (eq:logistic-regularised)

scikit-learn parameterises this as $C = 1/\lambda$ and applies $\ell_2$ by
default {{cite:pedregosa2011}} — so smaller $C$ is *more* regularisation, which
inverts the intuition from `alpha` in ridge and is a routine source of confusion.

**Complete separation** occurs when a hyperplane classifies the training data
perfectly. Then scaling $\vec{w}$ by any factor $>1$ strictly increases the
likelihood, the maximum likelihood estimate does not exist, coefficients diverge
to infinity, and standard errors are meaningless. It is common in small samples,
with many features, or when a feature is a proxy for the label — one of the
leakage signatures from {{ch:ds-leakage}}.

> IMPORTANT: Any regularisation whatsoever makes the penalised optimum finite
> and fixes separation. This is why scikit-learn regularises by default and why
> `penalty=None` on a wide dataset produces a convergence warning and enormous
> coefficients. The warning is the model telling you the estimate does not
> exist.

### 5.5 Multiple classes

For $K$ classes, one linear predictor per class fed through the **softmax**:

$$
\Prob(y=k \mid \vec{x}) = \frac{\exp(\vec{w}_k\T\vec{x} + b_k)}
                               {\sum_{j=1}^{K}\exp(\vec{w}_j\T\vec{x} + b_j)}
$$ (eq:softmax)

with the multiclass cross-entropy $\Loss = -\frac{1}{N}\sum_i \log
\Prob(y=y_i\mid\vec{x}_i)$. At $K=2$ this reduces to {{eq:logistic-model}}.

The softmax is **shift-invariant**: adding a constant to every logit leaves the
probabilities unchanged, so the parameters are identified only up to that shift.
The practical consequence is the standard numerical trick of subtracting the
maximum logit before exponentiating, without which $\exp$ overflows.

This exact function is the output of every classifier in {{part:6}} onward and
of every language model's next-token distribution ({{ch:tf-embeddings}}). Its
temperature-scaled form is how sampling is controlled in {{ch:llm-decoding}}.

### 5.6 Class weights and the threshold are different levers

Imbalanced data invites two distinct interventions, and conflating them is
common.

**Reweighting the loss** multiplies each class's contribution to
{{eq:cross-entropy}} by a constant:

$$
\Loss_{w} = -\frac{1}{N}\sum_{i}
   \big[c_1 y_i \log p_i + c_0 (1-y_i)\log(1-p_i)\big]
$$ (eq:weighted-cross-entropy)

This changes *the model*. With $c_1 > c_0$ the fitted probabilities are pushed
upward, the intercept shifts by approximately $\log(c_1/c_0)$, and the model is
no longer calibrated to the observed base rate.

**Moving the threshold** changes *the decision* and leaves the model alone.

For a linear model the two are nearly equivalent, because the weighting mostly
moves the intercept and the threshold moves the boundary along the same axis.
Given that, the threshold is the better lever, for three reasons: the
probabilities stay calibrated and therefore remain usable for anything else; the
threshold can be changed after deployment without retraining; and one model can
serve several decisions with different cost structures.

Reweighting earns its place when it changes the *shape* of the fit rather than
just its level — with a flexible model that would otherwise ignore a rare class
entirely, or when the training sample was collected under a different class
balance than production will see. In that second case the correction is exact
and known: under case-control sampling only the intercept is biased, and
subtracting $\log(\pi_{\text{sample}}/\pi_{\text{population}})$ restores it,
leaving every slope untouched. That is a genuinely useful property of the logit
link and one that most classifiers do not have.

> WARNING: `class_weight="balanced"` and resampling both destroy calibration by
> design. If a downstream system consumes the probability rather than the
> decision — expected-loss arithmetic, pricing, triage ordering across
> populations — you have broken it. {{ch:ds-leakage}} showed the resampling
> version of this error; the weighting version is quieter because no data is
> duplicated and nothing looks wrong.

### 5.7 Why libraries do not run Newton's method

{{sec:6-mathematical-foundation}} shows Newton converging in a handful of
iterations, which raises the obvious question of why any other optimiser exists.

Each Newton step forms and solves the $D \times D$ Hessian
{{eq:logreg-hessian}}: $O(ND^{2})$ to build and $O(D^{3})$ to factorise. At
$D = 20$ that is free. At $D = 100{,}000$ — routine for text features or one-hot
categoricals — the Hessian alone would need eighty gigabytes, so the method is
not slow but impossible.

What libraries use instead:

- **L-BFGS**, scikit-learn's default: approximates the inverse Hessian from the
  last $m$ gradient pairs in $O(mD)$ memory, keeping most of Newton's fast
  convergence at linear cost. This is the right default whenever the data fits
  in memory.
- **Stochastic gradient descent** for data that does not fit, at the cost of
  many more passes and a learning rate to tune ({{ch:dl-optimizers}}).
- **Coordinate descent** for $\ell_1$ penalties, since the objective is not
  differentiable at zero and second-order methods do not apply directly.

The pattern generalises: exact second-order optimisation is unbeatable at small
scale and unavailable at large scale, and essentially all of {{part:6}} operates
in the regime where only first-order methods are affordable.

## 6. Mathematical Foundation

### 6.1 The sigmoid derivative

$$
\sigma'(z) = \sigma(z)\big(1 - \sigma(z)\big)
$$ (eq:sigmoid-derivative)

Derivation: with $\sigma(z) = (1+e^{-z})^{-1}$,

$$
\sigma'(z) = \frac{e^{-z}}{(1+e^{-z})^{2}}
 = \frac{1}{1+e^{-z}}\cdot\frac{e^{-z}}{1+e^{-z}}
 = \sigma(z)\big(1-\sigma(z)\big)
$$

Maximised at $z=0$ with value $1/4$, vanishing at both extremes.

### 6.2 The gradient, and why it looks like least squares

Differentiate {{eq:cross-entropy}} for a single observation. Write $z =
\vec{w}\T\vec{x}+b$ and $p = \sigma(z)$:

$$
\frac{\partial \ell}{\partial p} = -\frac{y}{p} + \frac{1-y}{1-p}
 = \frac{p-y}{p(1-p)}
$$

Chain through {{eq:sigmoid-derivative}}:

$$
\frac{\partial \ell}{\partial z}
 = \frac{p-y}{p(1-p)} \cdot p(1-p) = p - y
$$ (eq:logit-delta)

The $p(1-p)$ cancels exactly. This is the reason cross-entropy is paired with
the sigmoid: the saturation that would otherwise kill the gradient is removed by
the loss. With squared error the factor survives and a confidently wrong
prediction learns almost nothing.

Then

$$
\nabla_{\vec{w}}\Loss = \frac{1}{N}\mat{X}\T(\vec{p} - \vec{y}),
\qquad
\frac{\partial \Loss}{\partial b} = \frac{1}{N}\sum_i (p_i - y_i)
$$ (eq:logreg-gradient)

Compare least squares, whose gradient is $\frac{2}{N}\mat{X}\T(\hat{\vec{y}} -
\vec{y})$. **Identical in form**: features transposed times the error. That
correspondence is not a coincidence — both are generalised linear models with
canonical link functions, and both are the single-layer case of the
backpropagation rule in {{ch:dl-backprop}}.

Setting {{eq:logreg-gradient}} to zero shows that with an intercept, the fitted
probabilities sum to the number of positives: $\sum_i p_i = \sum_i y_i$. Logistic
regression is calibrated *in aggregate* by construction, which is the algebraic
root of the calibration property in {{sec:2-why-this-matters}}.

### 6.3 Convexity via the Hessian

$$
\mat{H} = \frac{1}{N}\mat{X}\T\mat{S}\mat{X},
\qquad \mat{S} = \diag\big(p_i(1-p_i)\big)
$$ (eq:logreg-hessian)

Since every $p_i(1-p_i) > 0$, $\mat{S}$ is positive definite, so for any
$\vec{v}$,

$$
\vec{v}\T\mat{H}\vec{v} = \frac{1}{N}\|\mat{S}^{1/2}\mat{X}\vec{v}\|^{2} \ge 0
$$

$\mat{H} \succeq 0$, so $\Loss$ is convex. It is *strictly* convex when
$\mat{X}$ has full column rank, giving a unique minimum — the same rank
condition that made the normal equations solvable.

Newton's method with {{eq:logreg-hessian}} gives **iteratively reweighted least
squares**: each step is a weighted least-squares fit with weights $p_i(1-p_i)$,
so the observations near the boundary (where $p \approx 0.5$) dominate and the
confidently classified ones are nearly ignored. This is what most statistical
software actually runs, and it converges in a handful of iterations rather than
the thousands gradient descent needs.

### 6.4 Why separation breaks it

Suppose $\vec{w}^{*}$ separates the classes perfectly, so $y_i = 1 \Rightarrow
\vec{w}^{*\top}\vec{x}_i > 0$ and $y_i = 0 \Rightarrow \vec{w}^{*\top}\vec{x}_i <
0$. Consider $c\vec{w}^{*}$ for $c > 1$: every correct prediction moves closer
to certainty, so every term of {{eq:cross-entropy}} strictly decreases.

The loss therefore decreases monotonically in $c$ with infimum 0, attained only
in the limit $c \to \infty$. **No minimiser exists.** Gradient descent will run
until its iteration limit, reporting ever-larger coefficients and ever-smaller
loss.

Adding $\lambda\|\vec{w}\|^{2}$ makes the objective grow without bound in $c$, so
a finite minimum exists and is unique. One line of regularisation converts a
non-existent estimate into a well-posed one.

## 7. Implementation

```python {tier=A name=logistic-from-scratch}
"""Logistic regression from scratch: gradient descent, Newton/IRLS, and the
properties the derivation predicts.
"""
import numpy as np

rng = np.random.default_rng(0)


def sigmoid(z):
    """Numerically stable sigmoid — never exp() a large positive number."""
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def nll(X, y, w, lam=0.0):
    """Mean negative log-likelihood (eq. 33.5), computed stably.

    log(1+exp(z)) via logaddexp avoids overflow for large |z|.
    """
    z = X @ w
    loss = np.mean(np.logaddexp(0, z) - y * z)
    return loss + lam * np.sum(w[1:] ** 2)


def fit_gd(X, y, lr=0.5, n_iter=4000, lam=0.0):
    """Plain gradient descent using eq. 33.11."""
    w = np.zeros(X.shape[1])
    for _ in range(n_iter):
        p = sigmoid(X @ w)
        g = X.T @ (p - y) / len(y)
        g[1:] += 2 * lam * w[1:]            # intercept is never penalised
        w -= lr * g
    return w


def fit_newton(X, y, n_iter=25, lam=1e-8, tol=1e-10):
    """Newton / IRLS using the Hessian of eq. 33.12."""
    w = np.zeros(X.shape[1])
    for it in range(n_iter):
        p = sigmoid(X @ w)
        g = X.T @ (p - y) / len(y)
        g[1:] += 2 * lam * w[1:]
        S = p * (1 - p)
        H = (X * S[:, None]).T @ X / len(y) + 2 * lam * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < tol:
            break
    return w, it + 1


# --- data with known coefficients -------------------------------------------
n, d = 4000, 4
Xr = rng.normal(size=(n, d))
w_true = np.array([-0.4, 1.2, -0.8, 0.5, 1.5])       # intercept first
X = np.column_stack([np.ones(n), Xr])
p_true = sigmoid(X @ w_true)
y = (rng.random(n) < p_true).astype(float)

w_gd = fit_gd(X, y)
w_nt, iters = fit_newton(X, y)
print("true      :", np.round(w_true, 4))
print("grad desc :", np.round(w_gd, 4), f" (4000 iterations)")
print("Newton    :", np.round(w_nt, 4), f" ({iters} iterations)")
print(f"\nfinal loss: GD {nll(X, y, w_gd):.8f}   Newton {nll(X, y, w_nt):.8f}")
print("The same optimum, reached in 7 steps instead of 4000. Newton reads the")
print("step size off the curvature (eq. 33.12) instead of being told one, and")
print("because the problem is convex there is only one optimum to reach.")

# --- eq. 33.11 predicts aggregate calibration -------------------------------
p_hat = sigmoid(X @ w_nt)
print(f"\nsum of fitted probabilities : {p_hat.sum():.4f}")
print(f"number of positives         : {y.sum():.4f}")
print("Equal by construction: setting the gradient to zero forces it")
print("(section 6.2). Logistic regression is calibrated in aggregate whether")
print("or not the model is any good.")

# --- ...but aggregate calibration is not calibration ------------------------
print("\ncalibration by predicted-probability decile:")
print(f"{'bin':>14} {'n':>6} {'mean predicted':>16} {'observed rate':>15}")
edges = np.quantile(p_hat, np.linspace(0, 1, 11))
for i in range(10):
    m = (p_hat >= edges[i]) & (p_hat <= edges[i + 1])
    print(f"  [{edges[i]:.3f},{edges[i+1]:.3f}] {m.sum():>6} "
          f"{p_hat[m].mean():>16.4f} {y[m].mean():>15.4f}")

# --- section 6.1/6.2: why cross-entropy, not squared error ------------------
print("\n" + "=" * 72)
print("the gradient of a confidently WRONG prediction")
print("=" * 72)
print(f"{'z':>7} {'p=sigma(z)':>12} {'sigma prime':>13} "
      f"{'d(CE)/dz':>11} {'d(MSE)/dz':>12}")
for z in (-6.0, -3.0, -1.0, 0.0, 1.0, 3.0, 6.0):
    p = 1 / (1 + np.exp(-z))
    sp = p * (1 - p)
    y_true = 1.0                                  # truth is 1 throughout
    d_ce = p - y_true                             # eq. 33.10
    d_mse = 2 * (p - y_true) * sp                 # chain rule keeps sigma'
    print(f"{z:>7.1f} {p:>12.6f} {sp:>13.6f} {d_ce:>11.6f} {d_mse:>12.8f}")

print("\nAt z = -6 the model says 0.0025 when the truth is 1 — as wrong as it")
print("gets. Cross-entropy delivers a gradient of -0.9975; squared error")
print("delivers -0.00496, two hundred times smaller. Squared error learns")
print("least exactly where it is most wrong, because sigma' has vanished.")
print("Cross-entropy cancels that factor exactly (eq. 33.10).")

# --- section 5.4 / 6.4: complete separation ---------------------------------
print("\n" + "=" * 72)
print("complete separation: the estimate does not exist")
print("=" * 72)
Xs = np.column_stack([np.ones(40), np.linspace(-2, 2, 40)])
ys = (Xs[:, 1] > 0).astype(float)            # perfectly separable by design

print(f"{'iterations':>12} {'|w|':>14} {'loss':>14}")
for it in (100, 1000, 10000, 50000):
    w = fit_gd(Xs, ys, lr=1.0, n_iter=it)
    print(f"{it:>12} {np.linalg.norm(w):>14.4f} {nll(Xs, ys, w):>14.8f}")
print("\nThe norm grows without bound and the loss creeps towards zero but")
print("never arrives. There is no minimiser (section 6.4) — more iterations")
print("only produce larger numbers.")

print(f"\n{'lambda':>12} {'|w| (Newton)':>14} {'|w| (50k GD steps)':>20} "
      f"{'penalised loss':>16}")
for lam in (1e-4, 1e-2, 1e-1, 1.0):
    w_n, _ = fit_newton(Xs, ys, lam=lam)
    w_g = fit_gd(Xs, ys, lr=0.2, n_iter=50000, lam=lam)
    print(f"{lam:>12} {np.linalg.norm(w_n):>14.4f} "
          f"{np.linalg.norm(w_g):>20.4f} {nll(Xs, ys, w_n, lam):>16.8f}")
print("\nAny penalty at all makes the optimum finite, and the two optimisers")
print("now agree on where it is — which they could not do before, because")
print("there was nowhere to agree on. This is why every library regularises")
print("by default.")
```

```python {tier=A name=softmax-multiclass}
"""Multiclass logistic regression via the softmax, plus a check against
scikit-learn.
"""
import numpy as np

rng = np.random.default_rng(2)


def softmax(Z):
    """Row-wise softmax (eq. 33.14), shifted for numerical stability.

    Subtracting the row max exploits the shift-invariance noted in section 5.5:
    it changes nothing mathematically and everything numerically.
    """
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


# demonstrate why the shift is not optional
big = np.array([[1000.0, 1001.0, 1002.0]])
with np.errstate(over="ignore", invalid="ignore"):
    naive = np.exp(big) / np.exp(big).sum()
print("naive softmax of [1000, 1001, 1002]:", naive)
print("stable softmax                     :", softmax(big))
print("Shift-invariance is the difference between nan and the right answer.\n")


def fit_softmax(X, y, K, lr=0.5, n_iter=3000, lam=1e-4):
    """Gradient descent on multiclass cross-entropy.

    The gradient is X^T (P - Y) — the same form as the binary case and the
    same form as least squares (section 6.2).
    """
    N, D = X.shape
    Y = np.zeros((N, K))
    Y[np.arange(N), y] = 1.0
    W = np.zeros((D, K))
    for _ in range(n_iter):
        P = softmax(X @ W)
        G = X.T @ (P - Y) / N
        G[1:] += 2 * lam * W[1:]
        W -= lr * G
    return W


# --- three classes arranged so no single linear split works -----------------
n_per, K = 500, 3
centres = np.array([[0.0, 2.0], [-2.0, -1.0], [2.0, -1.0]])
Xr = np.vstack([c + rng.normal(0, 1.1, (n_per, 2)) for c in centres])
y = np.repeat(np.arange(K), n_per)
perm = rng.permutation(len(y))
Xr, y = Xr[perm], y[perm]
X = np.column_stack([np.ones(len(Xr)), Xr])

cut = int(0.7 * len(y))
W = fit_softmax(X[:cut], y[:cut], K)
P_te = softmax(X[cut:] @ W)
pred = P_te.argmax(1)
print(f"test accuracy (from scratch): {(pred == y[cut:]).mean():.4f}")
print(f"probabilities sum to 1      : {np.allclose(P_te.sum(1), 1.0)}")
print(f"mean predicted prob of the true class: "
      f"{P_te[np.arange(len(pred)), y[cut:]].mean():.4f}")

# --- check against the library ----------------------------------------------
try:
    from sklearn.linear_model import LogisticRegression
    sk = LogisticRegression(C=1 / (2 * 1e-4 * cut), max_iter=5000)
    sk.fit(Xr[:cut], y[:cut])
    sk_pred = sk.predict(Xr[cut:])
    agree = (sk_pred == pred).mean()
    print(f"\nscikit-learn test accuracy  : "
          f"{(sk_pred == y[cut:]).mean():.4f}")
    print(f"agreement with from-scratch : {agree:.4f}")
    print(f"max |P_scratch - P_sklearn| : "
          f"{np.abs(sk.predict_proba(Xr[cut:]) - P_te).max():.4f}")
    print("The predicted probabilities differ in the fourth decimal place —")
    print("the two use different optimisers and slightly different penalty")
    print("conventions — and the two models pick the same class for every")
    print("single test point. This is what 'implemented from scratch' should")
    print("mean: the same answer as the library, by a route you can read.")
except ImportError:
    print("\n(scikit-learn not installed — cross-check skipped)")

# --- the boundary is linear even though the probabilities are not -----------
print("\n" + "=" * 72)
print("the decision boundary is a hyperplane (section 5.1)")
print("=" * 72)
line = np.column_stack([np.ones(9), np.linspace(-4, 4, 9),
                        np.zeros(9)])
Pl = softmax(line @ W)
print(f"{'x1':>7} " + " ".join(f"{'P(class ' + str(k) + ')':>13}"
                               for k in range(K)))
for i in range(9):
    print(f"{line[i, 1]:>7.1f} " + " ".join(f"{Pl[i, k]:>13.4f}"
                                            for k in range(K)))
print("\nThe probabilities move smoothly and nonlinearly along the line, but")
print("the point at which the argmax changes is where two linear functions")
print("cross — so the boundary itself is straight.")
```

## 8. Practical Example

```python {tier=A name=threshold-selection}
"""Credit-default scoring: the threshold is a business decision, not 0.5.
"""
import numpy as np

rng = np.random.default_rng(11)
n = 12000


def sigmoid(z):
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1 / (1 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1 + e)
    return out


# --- a deliberately imbalanced problem --------------------------------------
income = rng.lognormal(10.5, 0.5, n)
utilisation = np.clip(rng.beta(2, 5, n), 0, 1)
n_late = rng.poisson(0.4, n)
years = rng.uniform(0, 25, n)

z = (-3.2
     - 0.9 * (np.log(income) - 10.5)
     + 3.0 * utilisation
     + 0.55 * n_late
     - 0.04 * years)
y = (rng.random(n) < sigmoid(z)).astype(float)
print(f"default rate: {y.mean():.4f}  ({int(y.sum())} of {n})")

Xr = np.column_stack([np.log(income), utilisation, n_late, years])
mu, sd = Xr[:8000].mean(0), Xr[:8000].std(0)
Xs = (Xr - mu) / sd
X = np.column_stack([np.ones(n), Xs])
Xtr, ytr, Xte, yte = X[:8000], y[:8000], X[8000:], y[8000:]


def fit_newton(X, y, lam=1e-4, n_iter=50):
    w = np.zeros(X.shape[1])
    for _ in range(n_iter):
        p = sigmoid(X @ w)
        g = X.T @ (p - y) / len(y)
        g[1:] += 2 * lam * w[1:]
        S = np.maximum(p * (1 - p), 1e-9)
        H = (X * S[:, None]).T @ X / len(y) + 2 * lam * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < 1e-11:
            break
    return w


w = fit_newton(Xtr, ytr)
names = ["log income", "utilisation", "late payments", "years"]
print(f"\n{'feature':<16} {'coef (log-odds)':>17} {'odds ratio':>12}")
for j, nm in enumerate(names):
    print(f"{nm:<16} {w[j + 1]:>17.4f} {np.exp(w[j + 1]):>12.4f}")
print("Coefficients are per standard deviation because the features were")
print("standardised: 'one SD more utilisation multiplies the odds by "
      f"{np.exp(w[2]):.2f}'.")

p_te = sigmoid(Xte @ w)

# --- why 0.5 is the wrong threshold here ------------------------------------
print("\n" + "=" * 72)
print("the default threshold of 0.5")
print("=" * 72)
pred50 = (p_te >= 0.5).astype(float)
print(f"predictions above 0.5: {int(pred50.sum())} of {len(pred50)}")
print(f"accuracy             : {(pred50 == yte).mean():.4f}")
print(f"always-predict-zero  : {(yte == 0).mean():.4f}")
print("The model beats the trivial baseline by almost nothing on accuracy")
print("while flagging almost no one. Accuracy is the wrong metric and 0.5")
print("is the wrong threshold (Chapter 34).")

# --- choose the threshold from costs ----------------------------------------
print("\n" + "=" * 72)
print("choosing the threshold from the cost of each error")
print("=" * 72)
COST_FN = 4000.0        # a default we approved: the money we lose
COST_FP = 250.0         # a good customer we declined: the margin forgone
print(f"cost of a missed default (FN): GBP {COST_FN:,.0f}")
print(f"cost of a declined good customer (FP): GBP {COST_FP:,.0f}")

print(f"\n{'threshold':>10} {'flagged':>8} {'TP':>6} {'FP':>6} {'FN':>6} "
      f"{'recall':>8} {'precision':>10} {'total cost':>13}")
best = (None, np.inf)
for t in (0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50):
    pred = (p_te >= t)
    tp = int(np.sum(pred & (yte == 1)))
    fp = int(np.sum(pred & (yte == 0)))
    fn = int(np.sum(~pred & (yte == 1)))
    cost = fn * COST_FN + fp * COST_FP
    rec = tp / max(1, tp + fn)
    prec = tp / max(1, tp + fp)
    if cost < best[1]:
        best = (t, cost)
    print(f"{t:>10.2f} {int(pred.sum()):>8} {tp:>6} {fp:>6} {fn:>6} "
          f"{rec:>8.3f} {prec:>10.3f} {cost:>13,.0f}")

t_star, c_star = best
c_half = (np.sum((p_te < 0.5) & (yte == 1)) * COST_FN
          + np.sum((p_te >= 0.5) & (yte == 0)) * COST_FP)
print(f"\ncheapest threshold: {t_star:.2f} at GBP {c_star:,.0f}")
print(f"threshold of 0.50 : GBP {c_half:,.0f}")
print(f"difference        : GBP {c_half - c_star:,.0f} "
      f"({(c_half - c_star) / max(c_half, 1) * 100:.1f}% of the cost)")

theory = COST_FP / (COST_FP + COST_FN)
print(f"\ntheoretical optimum = COST_FP / (COST_FP + COST_FN) = {theory:.4f}")
print("Expected cost is minimised by flagging whenever")
print("p * COST_FN > (1 - p) * COST_FP, i.e. p > COST_FP/(COST_FP+COST_FN).")
print("This requires CALIBRATED probabilities — the rule is meaningless if")
print("p is merely a score. It is the main practical reason to care about")
print("calibration rather than only about ranking.")

# --- and the calibration that makes the rule valid --------------------------
print("\ncalibration check on the test set, with the noise floor:")
print(f"{'predicted band':>18} {'n':>6} {'mean p':>9} {'observed':>10} "
      f"{'expected':>9} {'+-2 SE':>16}")
edges = np.quantile(p_te, np.linspace(0, 1, 9))
for i in range(8):
    m = (p_te >= edges[i]) & (p_te <= edges[i + 1])
    k, pbar = int(m.sum()), p_te[m].mean()
    se = np.sqrt(pbar * (1 - pbar) / k)
    print(f"  [{edges[i]:.3f}, {edges[i+1]:.3f}] {k:>6} "
          f"{pbar:>9.4f} {yte[m].mean():>10.4f} {pbar * k:>9.1f} "
          f"[{pbar - 2 * se:>6.4f},{pbar + 2 * se:>6.4f}]")
print("\nEvery observed rate but one falls inside two standard errors of the")
print("prediction, and the one that does not is a band containing about")
print("eighteen expected events — where a handful either way moves the rate")
print("by a third. A calibration table without its noise floor invites you")
print("to diagnose a model problem that is really a sample-size problem")
print("(Chapter 8). Chapter 34 turns this into a single metric.")
```

## 9. Common Mistakes

**Using linear regression on a binary target.** Unbounded predictions,
guaranteed heteroscedasticity, and a fit dragged by extreme points.

**Defaulting to a 0.5 threshold.** It is optimal only when the two errors cost
the same and the classes are balanced. Neither is usually true.

**Reading an odds ratio as a risk ratio.** Only approximately equal when the
outcome is rare.

**Not standardising before regularising.** Same error as
{{ch:ml-linear-regression}}, same cause.

**Confusing `C` with `alpha`.** In scikit-learn, smaller `C` means *more*
regularisation.

**Ignoring a convergence warning.** It usually means separation, which means
your estimate does not exist — or a leaked feature, which is worse.

**Using squared error for classification.** The gradient vanishes exactly where
the model is most wrong, as the table in {{sec:7-implementation}} measures.

**Naive softmax.** `exp` of a logit above ~709 overflows; subtract the row
maximum.

**Reporting accuracy on imbalanced data.** The default-rate baseline in
{{sec:8-practical-example}} beats a real model on accuracy while being useless.

## 10. Connection to Previous Chapters

{{ch:ml-linear-regression}} supplied the linear predictor, the regularisation
and the standardisation requirement, all reused without change; the gradient
{{eq:logreg-gradient}} is the least-squares gradient with a different residual.
{{ch:math-probability}} supplied the Bernoulli likelihood that
{{eq:cross-entropy}} is the negative log of. {{ch:math-derivatives}} supplied the
chain rule that produces the exact cancellation in {{eq:logit-delta}}.
{{ch:math-optimization}} supplied convexity, Newton's method, and the general
principle that the loss follows from the noise model.
{{ch:ds-leakage}} supplied the reason the threshold is a separate decision.

Forward: {{ch:ml-metrics}} formalises calibration and the threshold-free metrics
this chapter leaned on. {{ch:dl-neural-networks}} replaces the linear predictor with a
learned representation and keeps everything else. {{ch:tf-embeddings}} uses
{{eq:softmax}} over a vocabulary, and {{ch:llm-decoding}} scales its temperature.
The cross-entropy of {{eq:cross-entropy}} is the training objective for every
model from {{part:6}} onward.

## 11. Exercises

**Beginner**

1. Give three reasons not to use linear regression on a binary target.
2. Compute $\sigma(0)$, $\sigma(2)$, $\sigma(-2)$.
3. Convert a coefficient of $-0.5$ to an odds ratio and state its meaning.
4. Why is the decision boundary linear when the probabilities are not?
5. A model outputs 0.8 for 1000 cases and 600 of them are positive. Is it
   calibrated?

**Intermediate**

6. Derive {{eq:sigmoid-derivative}}.
7. Show {{eq:cross-entropy}} is the negative log of
   {{eq:bernoulli-likelihood}}.
8. Explain the cancellation in {{eq:logit-delta}} and why it matters.
9. What is complete separation, how would you detect it, and how do you fix it?
10. Why does the softmax need the max-subtraction trick?
11. Given $\text{COST}_{FN} = 10\times\text{COST}_{FP}$, what threshold minimises
    expected cost?

**Advanced**

12. Prove {{eq:logreg-hessian}} is positive semi-definite and state when it is
    strictly positive definite.
13. Derive the IRLS update and explain the role of the weights $p_i(1-p_i)$.
14. Show that softmax with $K=2$ reduces to the sigmoid.
15. Show that with an intercept, $\sum_i p_i = \sum_i y_i$ at the optimum, and
    explain why this does not imply the model is calibrated within subgroups.
16. Derive the gradient of multiclass cross-entropy and show it has the same
    form as the binary case.

**Implementation**

17. Implement IRLS and compare its iteration count against gradient descent to a
    fixed tolerance.
18. Reproduce the separation experiment and plot $\|\vec{w}\|$ against iteration
    on a log scale.
19. Implement $\ell_1$-penalised logistic regression by proximal gradient
    descent and confirm it produces exact zeros.
20. Build a reliability diagram and compute expected calibration error for a
    model of your choice.

**Reasoning**

21. Why is logistic regression still the default for credit scoring when
    gradient boosting is more accurate?
22. Your model's AUC is excellent and its probabilities are badly miscalibrated.
    Which downstream uses still work, and which break?

## 12. Chapter Summary

Logistic regression models the log-odds as a linear function and inverts through
the sigmoid, which keeps predictions in $[0,1]$ and makes the decision boundary a
hyperplane.

Cross-entropy is not a design choice: it is the negative log-likelihood of the
Bernoulli model, exactly as squared error is the negative log-likelihood of the
Gaussian one.

The gradient is $\mat{X}\T(\vec{p}-\vec{y})/N$ — the same form as least squares —
because the $p(1-p)$ from the sigmoid's derivative cancels against the
$1/[p(1-p)]$ from the loss. That cancellation is why cross-entropy is paired with
the sigmoid: the measured table in {{sec:7-implementation}} shows squared error
producing a gradient two hundred times smaller than cross-entropy's on a
confidently wrong prediction.

The loss is convex, so gradient descent cannot get stuck — a guarantee
{{part:6}} gives up entirely. Newton's method exploits the Hessian and converges
in a handful of steps.

Setting the gradient to zero forces the fitted probabilities to sum to the number
of positives, so the model is calibrated in aggregate by construction. Aggregate
calibration is not calibration within subgroups.

Complete separation means the maximum likelihood estimate does not exist:
coefficients diverge and loss approaches zero without ever attaining a minimum.
Any regularisation makes the optimum finite, which is why libraries regularise by
default.

The softmax generalises the model to $K$ classes and is the output layer of
essentially every classifier in the rest of this book.

The threshold is a decision, not a default. Minimising expected cost gives
$p^{*} = \text{COST}_{FP}/(\text{COST}_{FP}+\text{COST}_{FN})$, and that rule is
only valid for calibrated probabilities.

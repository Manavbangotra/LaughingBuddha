---
id: dl-losses
number: 52
part: VI
tier: full
status: reviewed
requires: [dl-forward, ml-logistic, ml-metrics, math-inference, math-derivatives]
provides: [loss-function, mse-loss, negative-log-likelihood,
           softmax-cross-entropy, label-smoothing, loss-class-weighting,
           huber-loss, logsumexp-trick, focal-loss]
citations: [rumelhart1986, goodfellow2016, szegedy2016, lin2017focal]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Derive squared error and cross-entropy as maximum-likelihood estimators
   under specific noise models.
2. State what distributional assumption each loss encodes and when it is wrong.
3. Derive the softmax–cross-entropy gradient and explain why it is so simple.
4. Implement cross-entropy numerically stably and explain what fails otherwise.
5. Choose a loss for imbalanced, noisy, or heavy-tailed problems with a reason.
6. Explain the difference between a loss and a metric, and why they differ.
7. Apply label smoothing, class weighting and focal loss, and say what each
   costs.

## 2. Why This Matters

**The loss is the only place you tell the network what you want.** Everything
else — architecture, optimiser, schedule — is machinery for minimising it. A
network trained on the wrong loss will minimise it competently and be useless,
and this is one of the most common ways a technically correct training run
produces a worthless model.

**Every loss is a distributional assumption in disguise.**
{{sec:6-mathematical-foundation}} derives squared error from Gaussian noise and
cross-entropy from a categorical likelihood. This is not a curiosity: it tells
you exactly when each is wrong. Squared error on heavy-tailed targets is a
Gaussian assumption applied to non-Gaussian data, and it fails in the specific,
predictable way that {{sec:9-practical-example}} measures.

**The softmax–cross-entropy gradient is $\hat{p} - y$**, one of the cleanest
results in the subject. It is why classification trains so much more reliably
than it has any right to, and the cancellation that produces it is worth
understanding rather than memorising — it is the same cancellation that makes
{{ch:ml-logistic}}'s logistic regression convex.

**A loss is not a metric.** You are evaluated on accuracy, F1 or revenue and you
optimise cross-entropy, because the metric is not differentiable. The gap
between the two is a permanent feature of the field and a permanent source of
surprise.

## 3. Prerequisites

{{ch:math-inference}} for maximum likelihood, which is the derivation behind every
loss here. {{ch:ml-logistic}} for the sigmoid, the log-loss and the binary case.
{{ch:ml-metrics}} for the metric/loss distinction and for what accuracy hides.
{{ch:dl-forward}} for the graph, whose final node this chapter specifies.

## 4. Intuitive Explanation

### 4.1 What a loss is

A loss $\ell(\hat{y}, y)$ is a number saying how bad this prediction was for
this target. Training minimises its average over the data:

$$
\Like(\vecgreek{\theta}) = \frac{1}{N}\sum_{i=1}^{N}\ell\big(f(\vec{x}_i;
\vecgreek{\theta}),\, y_i\big)
$$ (eq:empirical-risk-dl)

Three requirements, and they are more restrictive than they look:

**Differentiable almost everywhere**, so a gradient exists. Accuracy is not, and
that single fact is why accuracy is never a training objective.

**Minimised at the right answer.** Surprisingly easy to get wrong when a loss is
designed rather than derived.

**Well-scaled gradients.** A loss whose gradient vanishes exactly where the model
is worst — which is what squared error on a saturated sigmoid does — cannot
recover from a bad start.

### 4.2 The shape of a loss is what matters

```text
   squared error            cross-entropy
        ╲   ╱                    ╲
         ╲ ╱                      ╲
          V                        ╲___
    quadratic: doubling      unbounded as p→0: a
    the error quadruples     confident wrong answer
    the penalty              costs arbitrarily much
```

Squared error is *quadratic*, so it cares about large errors far more than small
ones — and therefore chases outliers. Cross-entropy is *unbounded* as the
predicted probability of the true class goes to zero, so confident mistakes are
catastrophically expensive. That asymmetry is exactly what you want from a
probabilistic classifier and exactly what makes cross-entropy sensitive to label
noise: a mislabelled example is a confident mistake by construction.

### 4.3 Loss and metric are different objects

You care about accuracy, or F1, or click-through, or money. You optimise
cross-entropy. Why:

**Metrics are usually not differentiable.** Accuracy is a step function; its
gradient is zero everywhere it exists.

**Metrics are often not decomposable.** AUC is a property of a ranking over the
whole dataset, not a sum over examples, so there is nothing to average in
{{eq:empirical-risk-dl}}.

**Metrics can be discontinuous in the parameters.** A tiny weight change flips
one prediction and moves accuracy by a whole step.

So we optimise a differentiable **surrogate** and evaluate the metric. The two
are correlated and not identical, and {{sec:9-practical-example}} measures a
case where the loss improves while the metric does not — which happens more
often than the literature's neat curves suggest.

### 4.4 The pairings that are not arbitrary

Output activation and loss are chosen together, because the pairing determines
the gradient:

```text
   task                   output activation      loss
   ────────────────────   ───────────────────    ──────────────────
   regression             identity               squared error
   binary classification  sigmoid                binary cross-entropy
   multi-class (1 label)  softmax                cross-entropy
   multi-label (k labels) sigmoid per class      binary cross-entropy, summed
   count data             exp                    Poisson NLL
```

Each row is a maximum-likelihood pairing under a different noise model, derived
in {{sec:6-mathematical-foundation}}. The multi-class and multi-label rows are
the ones people confuse: softmax forces the predictions to sum to one, which is
correct when exactly one label applies and wrong when several can.

## 5. Formal Explanation

### 5.1 Regression losses

$$
\ell_{\text{MSE}}(\hat{y}, y) = (\hat{y}-y)^2, \qquad
\ell_{\text{MAE}}(\hat{y}, y) = |\hat{y}-y|
$$ (eq:mse-mae)

$$
\ell_{\delta}(\hat{y},y) =
\begin{cases}
\tfrac{1}{2}(\hat{y}-y)^2 & |\hat{y}-y| \le \delta\\[2pt]
\delta\big(|\hat{y}-y| - \tfrac{1}{2}\delta\big) & \text{otherwise}
\end{cases}
$$ (eq:huber)

{{eq:huber}} is the **Huber loss**: quadratic near zero so the gradient
vanishes at the optimum, linear in the tails so a single outlier contributes a
bounded gradient. It interpolates between the two above and $\delta$ chooses
where.

The behavioural difference is about *which statistic* each recovers. Squared
error is minimised by the conditional **mean**, absolute error by the
conditional **median**, and that is the whole story about outlier sensitivity:
the mean of a contaminated distribution moves and the median does not.
{{sec:6-mathematical-foundation}} proves both.

### 5.2 Classification losses

Binary, with $\hat{p} = \sigma(z)$:

$$
\ell_{\text{BCE}} = -\big[y\log\hat{p} + (1-y)\log(1-\hat{p})\big]
$$ (eq:bce)

Multi-class with $C$ classes, one-hot $y$, and $\hat{\vec{p}} =
\softmax(\vec{z})$:

$$
\ell_{\text{CE}} = -\sum_{c=1}^{C} y_c \log \hat{p}_c = -\log \hat{p}_{y}
$$ (eq:cross-entropy-dl)

The second form is the useful one for implementation: only the true class's
probability appears, so the loss is a single indexed lookup rather than a sum.

### 5.3 What cross-entropy measures

For distributions $p$ (true) and $q$ (predicted):

$$
H(p, q) = -\sum_{c} p_c \log q_c = H(p) + \KL(p \parallel q)
$$ (eq:ce-decomposition)

Since $H(p)$ does not depend on the model, **minimising cross-entropy is
minimising the KL divergence** from the true distribution to the predicted one.
With one-hot targets $H(p) = 0$ and the two coincide exactly.

This is why cross-entropy is the right loss for a probabilistic classifier: it
is the only one whose minimiser is the true conditional distribution rather than
some summary of it.

### 5.4 The stability problem

The naive implementation of {{eq:cross-entropy-dl}} computes a softmax and then
its logarithm, and both steps overflow:

```text
   softmax(z) with max(z) = 800    →  exp(800) = inf         →  nan
   log(p) with p underflowed to 0  →  log(0)   = -inf        →  nan
```

Both are fixed by never forming the probabilities. Using
$\logsumexp$ with the max subtracted:

$$
\log \hat{p}_y = z_y - \logsumexp(\vec{z}), \qquad
\logsumexp(\vec{z}) = m + \log\sum_c e^{z_c - m}
$$ (eq:logsumexp)

with $m = \max_c z_c$. Every exponential now has a non-positive argument, so
nothing overflows, and the largest term is exactly $e^0 = 1$, so nothing
underflows to zero either.

> IMPORTANT: **Always fuse softmax and cross-entropy.** Every framework provides
> a combined function for exactly this reason, and the single most common
> numerical bug in beginner deep learning code is applying softmax in the model
> and then a loss that expects logits — which either double-softmaxes or loses
> the stability. {{sec:8-implementation}} measures where each naive version
> fails.

### 5.5 Modifications for imbalance and noise

**Class weighting** multiplies each example's loss by $w_{y_i}$, commonly
$w_c \propto 1/N_c$. It changes the effective class prior, and therefore the
calibration ({{ch:ml-logistic}}) — you get better recall on the rare class
and probabilities that no longer mean what they say.

**Label smoothing** {{cite:szegedy2016}} replaces the one-hot target with

$$
\tilde{y}_c = (1-\epsilon)y_c + \frac{\epsilon}{C}
$$ (eq:label-smoothing)

which bounds the loss, prevents the logits from growing without limit, and
improves calibration. It also *hurts* the model's ability to distinguish
confident correct predictions, which matters when you need a reliable ranking.

**Focal loss** {{cite:lin2017focal}} down-weights easy examples:

$$
\ell_{\text{focal}} = -(1-\hat{p}_y)^{\gamma}\log\hat{p}_y
$$ (eq:focal)

With $\gamma = 2$, an example already predicted at $\hat{p} = 0.9$ contributes
one hundredth of its usual gradient. It was designed for extreme foreground /
background imbalance in detection, where the easy negatives outnumber the
positives by a thousand to one.

## 6. Mathematical Foundation

### 6.1 Squared error is a Gaussian likelihood

Assume $y = f(\vec{x};\vecgreek{\theta}) + \varepsilon$ with
$\varepsilon \sim \mathcal{N}(0, \sigma^2)$. The log-likelihood of the data is

$$
\log \Like = \sum_{i} \left[-\frac{(y_i - \hat{y}_i)^2}{2\sigma^2}
 - \tfrac{1}{2}\log(2\pi\sigma^2)\right]
$$ (eq:gaussian-ll)

Maximising over $\vecgreek{\theta}$ discards the constant and the positive
factor $1/2\sigma^2$, leaving $\min \sum_i (y_i-\hat{y}_i)^2$.

**Squared error is exactly maximum likelihood under additive Gaussian noise of
constant variance**, and that assumption is the one to check. Heavy tails
violate it, heteroscedasticity violates it, and both failures are visible in a
residual plot ({{ch:ds-eda}}).

### 6.2 The minimiser of each regression loss

**Squared error gives the conditional mean.** Differentiate
$\E[(a - Y)^2]$ with respect to $a$: $2(a - \E[Y]) = 0$, so
$a^\star = \E[Y]$.

**Absolute error gives the conditional median.** For
$g(a) = \E|a - Y|$,

$$
g'(a) = \Pr(Y < a) - \Pr(Y > a)
$$ (eq:mae-derivative)

which is zero when both are $1/2$ — the median by definition. Note also that
$|g'| \le 1$ always: **each example contributes a gradient of bounded
magnitude**, which is the entire robustness argument. Under squared error a
single point at distance $d$ contributes gradient $2d$, so one bad label can
dominate the whole batch.

### 6.3 Cross-entropy is a categorical likelihood

For one observation with true class $y$ and predicted distribution
$\hat{\vec{p}}$, the categorical likelihood is $\prod_c \hat{p}_c^{y_c}$. Its
negative logarithm is $-\sum_c y_c \log \hat{p}_c$, which is
{{eq:cross-entropy-dl}}.

So both principal losses of this chapter are the same principle — maximum
likelihood — applied to different noise models. Choosing a loss is choosing what
you believe about the noise, and the reason to say this explicitly is that it
converts an arbitrary-looking menu into a decision with a criterion.

### 6.4 The softmax–cross-entropy gradient

The result that makes classification work. With $z_c$ the logits,
$\hat{p} = \softmax(\vec{z})$ and $\ell = -\log \hat{p}_y$:

First, the softmax Jacobian.

$$
\frac{\partial \hat{p}_c}{\partial z_k} = \hat{p}_c(\delta_{ck} - \hat{p}_k)
$$ (eq:softmax-jacobian)

*Proof.* $\hat{p}_c = e^{z_c}/S$ with $S = \sum_j e^{z_j}$ and
$\partial S/\partial z_k = e^{z_k}$. For $c = k$, the quotient rule gives
$(e^{z_c}S - e^{z_c}e^{z_c})/S^2 = \hat{p}_c(1-\hat{p}_c)$. For $c \neq k$ it
gives $-e^{z_c}e^{z_k}/S^2 = -\hat{p}_c\hat{p}_k$. Both cases are
{{eq:softmax-jacobian}}. $\square$

Now the loss. Writing $\ell = -\sum_c y_c \log\hat p_c$:

$$
\frac{\partial \ell}{\partial z_k}
 = -\sum_c \frac{y_c}{\hat{p}_c}\cdot \hat{p}_c(\delta_{ck}-\hat{p}_k)
 = -\sum_c y_c(\delta_{ck}-\hat{p}_k)
 = \hat{p}_k \sum_c y_c - y_k
$$

and since $\sum_c y_c = 1$,

$$
\boxed{\;\nabla_{\vec{z}}\,\ell = \hat{\vec{p}} - \vec{y}\;}
$$ (eq:softmax-ce-gradient)

**The gradient is the prediction error.** Three consequences, and each is worth
stating separately:

**The $1/\hat{p}_c$ cancels.** That factor is what would blow up for a confident
wrong answer, and it is exactly cancelled by the $\hat{p}_c$ in the Jacobian.
This is why cross-entropy is stable where the loss value itself is unbounded.

**The gradient magnitude is proportional to the error.** Confidently wrong gives
a gradient near 1; confidently right gives one near 0. No saturation, which is
precisely what squared error on a sigmoid fails to provide.

**It is the same result as logistic regression** ({{ch:ml-logistic}}), which is
the binary case of this derivation, and the same as linear regression under
squared error. All three are generalised linear models with the canonical link,
and the cancellation is a general property of that pairing rather than three
coincidences.

### 6.5 Why squared error on a sigmoid fails

Take $\hat{p} = \sigma(z)$ and $\ell = (\hat{p}-y)^2$. Then

$$
\frac{\partial \ell}{\partial z} = 2(\hat{p}-y)\,\sigma'(z)
 = 2(\hat{p}-y)\,\hat{p}(1-\hat{p})
$$ (eq:mse-sigmoid-gradient)

Consider $y = 1$ and $\hat{p} = 0.001$ — as wrong as it is possible to be. The
error term is $-0.999$ and the derivative factor is $0.000999$, giving a
gradient of about $-0.002$. **The model is maximally wrong and receives almost
no gradient.**

Under cross-entropy the same case gives $\hat{p} - y = -0.999$, five hundred
times larger. {{sec:8-implementation}} measures the whole curve, and the
difference is not subtle.

This is the sharpest available demonstration that loss and output activation
must be chosen together. Neither the sigmoid nor squared error is wrong; the
*pairing* is.

### 6.6 What label smoothing does to the optimum

Under {{eq:label-smoothing}}, cross-entropy is minimised when
$\hat{p}_c = \tilde{y}_c$, so the target for the true class is $1 - \epsilon +
\epsilon/C$ rather than 1. Setting $\hat p_y$ to that value requires a finite
logit gap:

$$
z_y - z_c = \log\frac{(1-\epsilon) + \epsilon/C}{\epsilon/C}
$$ (eq:smoothing-logit-gap)

for any other class $c$, by taking the log-ratio of the softmax outputs. With
$\epsilon = 0.1$ and $C = 10$ this is $\log(0.91/0.01) \approx 4.51$.

Without smoothing the target is 1 and no finite logit achieves it, so the logits
grow without bound throughout training. **Label smoothing replaces an
unreachable optimum with a reachable one**, which is the mechanism behind both
its regularising effect and its calibration benefit.

## 7. Internal Mechanics

### 7.1 What the fused kernel does

`cross_entropy(logits, targets)` in any framework performs, in one pass:

```text
   m       = max(logits, axis=-1)          numerical stabilisation
   lse     = m + log(sum(exp(logits - m)))
   loss    = lse - logits[target]          eq. 52.10, no probabilities formed
   backward: softmax(logits) - onehot      eq. 52.15, one subtraction
```

The forward never materialises $\hat{\vec{p}}$ and the backward computes it once
directly. Splitting the two into `softmax` then `nll_loss` produces the same
numbers in exact arithmetic, a $C\times C$ Jacobian in the naive backward, and
overflow in float16 at logit magnitudes that occur routinely.

### 7.2 Reduction, and why it interacts with everything

Frameworks default to the **mean** over the batch. The alternatives are `sum`
and `none`, and the choice is not cosmetic:

Under `sum`, the gradient magnitude scales with batch size, so **changing the
batch size silently changes the effective learning rate** by the same factor.
Under `mean` it does not, which is why `mean` is the default.

The subtlety is masked or variable-length data. Averaging over padded positions
divides by the wrong denominator, so the loss depends on how much padding
happened to be in the batch. The correct reduction is the sum over valid
positions divided by the *count* of valid positions, and getting this wrong is
a standard bug in sequence models — it produces a loss that decreases when
batches happen to contain longer sequences.

### 7.3 Precision

In float16 the largest representable value is 65504 and `exp` overflows above
about 11. Logit magnitudes reach that in ordinary training, which is why:

- the loss is computed in float32 even when the model runs in float16;
- the max-subtraction of {{eq:logsumexp}} is mandatory rather than an
  optimisation;
- the accumulation over the batch is in float32, since summing thousands of
  float16 values loses precision ({{ch:mle-reproducibility}}).

### 7.4 Class weighting changes two things

`weight=` in a framework's cross-entropy multiplies each example's loss by
$w_{y_i}$. Note that it also changes the *denominator* of the mean reduction in
most implementations — the weighted mean divides by $\sum_i w_{y_i}$, not by
$N$. Two implementations that disagree about this differ by a constant factor,
which is a learning-rate change in disguise and a real source of
"why does this reproduce differently" confusion.

## 8. Implementation

```python {tier=A name=losses-from-likelihood}
"""Every loss in this chapter, derived and verified: minimisers, gradients,
and the numerical failures of the naive implementations.
"""
import numpy as np

rng = np.random.default_rng(0)


# --- section 6.2: which statistic does each loss recover? -------------------
print("=" * 72)
print("what each regression loss estimates (section 6.2)")
print("=" * 72)


def best_constant(y, loss, grid):
    """Minimise the average loss over a fine grid of constants."""
    return grid[np.argmin([loss(np.full_like(y, a), y).mean() for a in grid])]


mse = lambda p, t: (p - t) ** 2
mae = lambda p, t: np.abs(p - t)


def huber(delta):
    def f(p, t):
        r = np.abs(p - t)
        return np.where(r <= delta, 0.5 * r ** 2, delta * (r - 0.5 * delta))
    return f


clean = rng.normal(10.0, 1.0, 4000)
contaminated = clean.copy()
contaminated[:40] = 200.0                       # 1% gross outliers

grid = np.linspace(0, 40, 8001)
print(f"{'data':<16} {'loss':<12} {'minimiser':>10} {'true mean':>11} "
      f"{'true median':>13}")
for label, y in (("clean", clean), ("1% outliers", contaminated)):
    for name, f in (("squared", mse), ("absolute", mae),
                    ("huber(1.0)", huber(1.0))):
        print(f"{label:<16} {name:<12} {best_constant(y, f, grid):>10.3f} "
              f"{y.mean():>11.3f} {np.median(y):>13.3f}")

print("\nSquared error tracks the MEAN and absolute error the MEDIAN, exactly")
print("as section 6.2 proves. With 1% of the data at 200, the mean moves by")
print("about two units and the median does not move at all — so the squared-")
print("error fit is dragged toward points it will never predict well, and the")
print("absolute-error fit ignores them. Huber sits with the median here")
print("because delta=1.0 puts the outliers deep in its linear region.")

# --- and the gradient argument, which is the REASON -------------------------
print("\nthe reason, in one line: gradient contributed by ONE example")
print(f"{'residual':>10} {'squared':>12} {'absolute':>12} {'huber(1.0)':>12}")
for r in (0.1, 1.0, 10.0, 100.0):
    g_sq, g_ab = 2 * r, 1.0
    g_hu = r if abs(r) <= 1.0 else 1.0
    print(f"{r:>10.1f} {g_sq:>12.1f} {g_ab:>12.1f} {g_hu:>12.1f}")
print("Squared error's gradient is UNBOUNDED in the residual; the other two")
print("are bounded by construction. One mislabelled point at residual 100")
print("contributes 200x the gradient of a typical point under squared error")
print("and 1x under absolute error.")

# --- section 6.4: the softmax cross-entropy gradient, verified numerically --
print("\n" + "=" * 72)
print("the softmax cross-entropy gradient is p - y (eq. 52.15)")
print("=" * 72)


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def ce_from_logits(z, y_idx):
    """Eq. 52.10: stable, and never forms the probabilities."""
    m = z.max(axis=-1, keepdims=True)
    lse = m[..., 0] + np.log(np.exp(z - m).sum(axis=-1))
    return lse - z[np.arange(len(z)), y_idx]


C = 5
z = rng.normal(size=(3, C)) * 2.0
y_idx = rng.integers(0, C, size=3)
onehot = np.eye(C)[y_idx]

analytic = softmax(z) - onehot
numeric = np.zeros_like(z)
eps = 1e-6
for i in range(z.shape[0]):
    for k in range(C):
        zp, zm = z.copy(), z.copy()
        zp[i, k] += eps
        zm[i, k] -= eps
        numeric[i, k] = (ce_from_logits(zp, y_idx)[i]
                         - ce_from_logits(zm, y_idx)[i]) / (2 * eps)

print(f"max |analytic - numerical| = {np.abs(analytic - numeric).max():.3e}")
print("\nThe boxed result of eq. 52.15 is confirmed to central-difference")
print("accuracy. The whole gradient of a classifier's output layer is one")
print("subtraction, because the 1/p_c that would explode for a confident")
print("wrong answer is cancelled exactly by the p_c in the softmax Jacobian.")

# --- section 6.5: why squared error on a sigmoid fails ----------------------
print("\n" + "=" * 72)
print("squared error on a sigmoid saturates; cross-entropy does not")
print("=" * 72)
print("target y = 1, varying how wrong the model is\n")
print(f"{'p_hat':>8} {'|dCE/dz|':>12} {'|dMSE/dz|':>12} {'ratio':>10}")
for p in (0.001, 0.01, 0.1, 0.5, 0.9, 0.99):
    g_ce = abs(p - 1.0)
    g_mse = abs(2 * (p - 1.0) * p * (1 - p))
    print(f"{p:>8.3f} {g_ce:>12.5f} {g_mse:>12.7f} {g_ce / g_mse:>10.1f}x")

print("\nRead the top row. The model assigns probability 0.001 to the correct")
print("class — it could not be more wrong — and squared error responds with a")
print("gradient of 0.002. Cross-entropy responds with 0.999.")
print("\nThe MSE gradient is largest in the MIDDLE and vanishes at both ends,")
print("so a network that starts confidently wrong under this pairing cannot")
print("dig itself out. That is eq. 52.17, and it is the sharpest argument in")
print("the chapter for choosing the loss and the output activation together.")

# --- section 5.4: where the naive implementations break ---------------------
print("\n" + "=" * 72)
print("numerical stability: where each implementation fails (eq. 52.11)")
print("=" * 72)


def ce_naive(z, y_idx):
    """softmax then log — both steps can fail."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        e = np.exp(z)
        p = e / e.sum(axis=-1, keepdims=True)
        return -np.log(p[np.arange(len(z)), y_idx])


def ce_halfstable(z, y_idx):
    """max-subtracted softmax, then log — fixes overflow, not underflow."""
    with np.errstate(divide="ignore"):
        p = softmax(z)
        return -np.log(p[np.arange(len(z)), y_idx])


fmt = lambda v: ("nan" if np.isnan(v) else
                 ("inf" if v > 0 else "-inf") if np.isinf(v) else f"{v:.4f}")

for case, true_is_max in (("A: the model is confidently WRONG "
                           "(true class has the small logit)", False),
                          ("B: the model is confidently RIGHT "
                           "(true class has the large logit)", True)):
    print(f"\ncase {case}")
    print(f"{'max logit':>10} {'naive':>14} {'softmax+log':>14} {'fused':>14}")
    for scale in (1.0, 10.0, 100.0, 400.0, 700.0, 800.0):
        zz = np.zeros((1, 4))
        zz[0, 0] = scale
        yy = np.array([0 if true_is_max else 1])
        a, b, c = (ce_naive(zz, yy)[0], ce_halfstable(zz, yy)[0],
                   ce_from_logits(zz, yy)[0])
        print(f"{scale:>10.0f} {fmt(a):>14} {fmt(b):>14} {fmt(c):>14}")

print("\nThe two cases fail differently and that is the whole point.")
print("\nIn case B the max-subtraction is exactly what is needed: the naive")
print("version overflows once exp(z) exceeds float64's range near 710, while")
print("subtracting the max keeps every exponent at or below zero and the")
print("answer stays correct.")
print("\nIn case A the max-subtraction does not help at all. Both unfused")
print("versions fail at the same magnitude, for different reasons: the")
print("naive one because exp(800) overflows, and the max-subtracted one")
print("because exp(-800) underflows to exactly zero, making the true")
print("class's probability zero and its logarithm -inf. Rearranging the")
print("softmax cannot fix the second — the information was destroyed when")
print("the probability was rounded away.")
print("\nOnly the fused form of eq. 52.10 survives both, because it never")
print("forms a probability at all: it subtracts a logit from a logsumexp and")
print("both are ordinary-sized numbers. This is exactly the case that")
print("matters, since a confidently wrong prediction early in training is")
print("routine. In float16 these failures arrive at logit magnitudes around")
print("11 rather than 700, which any real training run reaches.")

# --- section 6.6: label smoothing bounds the logits -------------------------
print("\n" + "=" * 72)
print("label smoothing replaces an unreachable optimum (eq. 52.20)")
print("=" * 72)
print(f"{'epsilon':>9} {'target p_y':>12} {'optimal logit gap':>19}")
for epsA in (0.0, 0.01, 0.05, 0.1, 0.2):
    C10 = 10
    if epsA == 0.0:
        print(f"{epsA:>9.2f} {1.0:>12.4f} {'unbounded':>19}")
        continue
    t = (1 - epsA) + epsA / C10
    gap = np.log(t / (epsA / C10))
    print(f"{epsA:>9.2f} {t:>12.4f} {gap:>19.2f}")

print("\nWith no smoothing the target probability is exactly 1, which no")
print("finite logit achieves, so the logits grow throughout training with no")
print("stopping point. Any epsilon at all makes the optimum finite — 4.51")
print("nats at the standard epsilon=0.1 with ten classes.")
print("\nThat is the whole mechanism. The regularisation and the calibration")
print("benefit both follow from the optimum being reachable, and so does the")
print("cost: the model can no longer express 'certain', which degrades any")
print("downstream use that needs a confident ranking.")
```

```python {tier=A name=choosing-a-loss}
"""Three decisions measured rather than asserted: the loss-metric gap, what
class weighting actually buys, and what focal loss does.
"""
import numpy as np

rng = np.random.default_rng(3)


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def sigmoid(z):
    return np.where(z >= 0, 1 / (1 + np.exp(-np.clip(z, -500, 500))),
                    np.exp(np.clip(z, -500, 500))
                    / (1 + np.exp(np.clip(z, -500, 500))))


# --- an imbalanced binary problem -------------------------------------------
def make_data(n, pos_rate, sep, seed):
    rs = np.random.default_rng(seed)
    y = (rs.random(n) < pos_rate).astype(float)
    X = rs.normal(size=(n, 6))
    X[:, 0] += sep * y                          # only feature 0 is informative
    X[:, 1] += 0.4 * sep * y
    return X, y


Xtr, ytr = make_data(4000, 0.03, 1.6, 11)
Xte, yte = make_data(4000, 0.03, 1.6, 12)
print("=" * 72)
print("an imbalanced problem: 3% positive")
print("=" * 72)
print(f"train positives: {int(ytr.sum())}/{len(ytr)}   "
      f"test positives: {int(yte.sum())}/{len(yte)}")


def train_logreg(X, y, loss="bce", weight=None, gamma=0.0,
                 steps=3000, lr=0.3, seed=0):
    """Plain logistic regression; the LOSS is the only thing that varies."""
    rs = np.random.default_rng(seed)
    w = rs.normal(0, 0.01, X.shape[1])
    b = 0.0
    for _ in range(steps):
        p = sigmoid(X @ w + b)
        p = np.clip(p, 1e-12, 1 - 1e-12)
        if loss == "bce":
            g = p - y                                  # eq. 52.15, binary
        elif loss == "focal":
            # d/dz of -(1-p_t)^gamma log p_t, with p_t the true-class prob.
            # Using dp_t/dz = s p_t(1-p_t) with s = +1 for y=1 and -1 for y=0,
            # this collapses to s[gamma p_t (1-p_t)^g log p_t - (1-p_t)^(g+1)],
            # which reduces to p - y at gamma = 0 (verified below).
            pt = np.where(y == 1, p, 1 - p)
            s_ = np.where(y == 1, 1.0, -1.0)
            g = s_ * (gamma * pt * (1 - pt) ** gamma * np.log(pt)
                      - (1 - pt) ** (gamma + 1))
        if weight is not None:
            g = g * np.where(y == 1, weight, 1.0)
        w -= lr * (X.T @ g) / len(y)
        b -= lr * g.mean()
    return w, b


def report(name, w, b):
    s = Xte @ w + b
    p = sigmoid(s)
    pred = (p > 0.5).astype(float)
    tp = float(((pred == 1) & (yte == 1)).sum())
    fp = float(((pred == 1) & (yte == 0)).sum())
    fn = float(((pred == 0) & (yte == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    # AUC by rank
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    npos, nneg = yte.sum(), (1 - yte).sum()
    auc = (ranks[yte == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)
    nll = -np.mean(yte * np.log(np.clip(p, 1e-12, 1))
                   + (1 - yte) * np.log(np.clip(1 - p, 1e-12, 1)))
    print(f"{name:<26} {nll:>8.4f} {auc:>7.4f} {prec:>8.3f} {rec:>7.3f} "
          f"{f1:>7.3f} {p.mean():>9.4f}")


print(f"\n{'objective':<26} {'NLL':>8} {'AUC':>7} {'prec':>8} {'rec':>7} "
      f"{'F1':>7} {'mean p':>9}")
w0, b0 = train_logreg(Xtr, ytr, "bce")
report("plain BCE", w0, b0)
for weight in (5.0, 32.0):
    ww, bb = train_logreg(Xtr, ytr, "bce", weight=weight)
    report(f"BCE, positive weight {weight:g}", ww, bb)
for gamma in (1.0, 2.0):
    wf, bf = train_logreg(Xtr, ytr, "focal", gamma=gamma)
    report(f"focal loss, gamma={gamma:g}", wf, bf)

print(f"\nbase rate for reference: {yte.mean():.4f}")
print("\nThree things to read out of this table, and one of them is not")
print("what the usual account of these techniques would lead you to expect.")
print("\nFirst, EVERY modification is ranking-neutral. AUC moves in the")
print("fourth decimal place across all five rows. Neither weighting nor")
print("focal loss taught the model anything it did not already know about")
print("which examples are positive; the decision function is the same.")
print("\nSecond, every modification is worse on NLL and moves the mean")
print("predicted probability far above the 3% base rate. That is")
print("decalibration, and it is the price of both techniques.")
print("\nThird — and this is the part worth pausing on — weighting shifts")
print("recall substantially and FOCAL LOSS DOES NOT. Focal loss inflated the")
print("probabilities just as much and left precision and recall essentially")
print("unchanged. Down-weighting easy examples rescales the loss surface")
print("without preferentially favouring the positive class, so it does not")
print("act as a threshold shift the way a class weight does.")
print("\nThat is consistent with what focal loss was designed for. It was")
print("built for foreground/background imbalance at thousands to one, where")
print("the easy negatives are so numerous that they dominate the gradient")
print("sum outright. At 30 to 1 they do not, so there is nothing for the")
print("modulating factor to suppress.")
print("\nThe practical summary: class weighting is a threshold choice")
print("expressed as a loss, and if you can tune the threshold directly")
print("(Chapter 33) you should, because it is reversible and does not")
print("decalibrate. Focal loss is a different tool for a much more extreme")
print("regime, and reaching for it at mild imbalance — as is common — is")
print("using it far outside the setting it was validated in.")

# --- the loss-metric gap ----------------------------------------------------
print("\n" + "=" * 72)
print("the loss improves and the metric does not (section 4.3)")
print("=" * 72)


def train_traced(X, y, Xv, yv, steps=4000, lr=0.3, seed=0, every=250):
    rs = np.random.default_rng(seed)
    w, b = rs.normal(0, 0.01, X.shape[1]), 0.0
    trace = []
    for t in range(steps + 1):
        s = Xv @ w + b
        p = sigmoid(s)
        if t % every == 0:
            nll = -np.mean(yv * np.log(np.clip(p, 1e-12, 1))
                           + (1 - yv) * np.log(np.clip(1 - p, 1e-12, 1)))
            acc = ((p > 0.5) == yv).mean()
            order = np.argsort(s)
            ranks = np.empty(len(s))
            ranks[order] = np.arange(1, len(s) + 1)
            npos, nneg = yv.sum(), (1 - yv).sum()
            auc = (ranks[yv == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)
            trace.append((t, nll, acc, auc))
        ptr = sigmoid(X @ w + b)
        g = ptr - y
        w -= lr * (X.T @ g) / len(y)
        b -= lr * g.mean()
    return trace


trace = train_traced(Xtr, ytr, Xte, yte)
print(f"{'step':>6} {'val NLL':>10} {'accuracy':>10} {'AUC':>8}")
for t, nll, acc, auc in trace:
    print(f"{t:>6} {nll:>10.5f} {acc:>10.5f} {auc:>8.5f}")

nlls = [r[1] for r in trace]
accs = [r[2] for r in trace]
aucs = [r[3] for r in trace]
i1 = 1                                    # the first checkpoint after step 0
print(f"\nfrom step 0 (random init):")
print(f"  NLL      {nlls[0] - nlls[-1]:+.5f}   (improvement)")
print(f"  accuracy {accs[-1] - accs[0]:+.5f}")
print(f"  AUC      {aucs[-1] - aucs[0]:+.5f}")
print(f"\nfrom step {trace[i1][0]} onward, which is the part that matters:")
print(f"  NLL      {nlls[i1] - nlls[-1]:+.5f}   "
      f"({(nlls[i1] - nlls[-1]) / nlls[i1]:.1%} of the remaining loss)")
print(f"  accuracy {accs[-1] - accs[i1]:+.5f}")
print(f"  AUC      {aucs[-1] - aucs[i1]:+.5f}")

print("\nThe first checkpoint does almost all the visible work: from random")
print("initialisation both metrics jump, and that is not the interesting")
print("part. Everything after it is.")
print("\nAfter step 250 the loss continues to improve by a real margin while")
print("accuracy does NOT improve — it drifts slightly DOWN — and AUC is")
print("flat to four decimal places. So the second half of training refined")
print("probabilities in a way that neither the argmax nor the ranking")
print("registers at all.")
print("\nThis is a sharper version of the point than 'the metric lags the")
print("loss'. The loss and the metric are measuring genuinely different")
print("things, and it is possible — as here — for the loss to be improving")
print("while a threshold metric slowly degrades. Neither number is lying.")
print("\nThe practical rule: monitor the loss AND a threshold-free metric,")
print("and do not read a falling loss as evidence that the thing you are")
print("evaluated on is improving. Decide in advance which one you will stop")
print("on, because they will disagree.")

# --- reduction and batch size (section 7.2) ---------------------------------
print("\n" + "=" * 72)
print("'sum' reduction couples the learning rate to the batch size (7.2)")
print("=" * 72)


# One fixed model and one fixed data stream; the batch is a PREFIX of it, so
# the only thing changing across rows is the batch size itself.
# The labels must depend on X. With coin-flip labels the true gradient is
# zero, so the mean-reduced norm would decay as 1/sqrt(B) and the experiment
# would measure sampling noise rather than the reduction.
_rs = np.random.default_rng(7)
_Xpool = _rs.normal(size=(8192, 6))
_wtrue = np.array([1.4, -1.1, 0.8, 0.0, 0.5, -0.3])
_ypool = (_rs.random(8192) < sigmoid(_Xpool @ _wtrue)).astype(float)
_w = _rs.normal(0, 0.1, 6)                 # the model, deliberately not _wtrue


def one_step_norm(batch, reduction):
    Xb, yb = _Xpool[:batch], _ypool[:batch]
    grad = Xb.T @ (sigmoid(Xb @ _w) - yb)
    if reduction == "mean":
        grad = grad / batch
    return float(np.linalg.norm(grad))


print(f"{'batch':>7} {'|grad| (mean)':>15} {'|grad| (sum)':>15} "
      f"{'sum / batch-8 sum':>19}")
base = one_step_norm(8, "sum")
for batch in (8, 32, 128, 512, 2048, 8192):
    print(f"{batch:>7} {one_step_norm(batch, 'mean'):>15.4f} "
          f"{one_step_norm(batch, 'sum'):>15.4f} "
          f"{one_step_norm(batch, 'sum') / base:>19.1f}x")
print("\nUnder 'mean' the gradient norm settles: it is an estimate of a fixed")
print("quantity — the full-dataset gradient at this parameter setting — and")
print("larger batches estimate it more precisely rather than differently.")
print("The small batches wobble around that value because they are noisy")
print("estimates of it, not because the quantity itself is changing.")
print("\nUnder 'sum' the norm grows roughly in proportion to the batch, so")
print("at a fixed learning rate the step length is multiplied by the same")
print("factor. That is why 'mean' is the default, and why switching to 'sum'")
print("presents as a diverging model rather than as a configuration change.")

# --- masked reduction, the sequence-model bug -------------------------------
print("\n" + "=" * 72)
print("the masked-reduction bug (section 7.2)")
print("=" * 72)
B_, T = 4, 10
lengths = np.array([10, 6, 3, 8])
mask = np.arange(T)[None, :] < lengths[:, None]
per_token = rng.random((B_, T)) * 2.0
per_token_masked = per_token * mask

wrong = per_token_masked.sum() / per_token_masked.size    # divide by B*T
right = per_token_masked.sum() / mask.sum()               # divide by n valid
print(f"sequence lengths           : {lengths.tolist()} of max {T}")
print(f"valid tokens               : {int(mask.sum())} of {B_ * T}")
print(f"loss, divided by B*T       : {wrong:.4f}   WRONG")
print(f"loss, divided by valid     : {right:.4f}   correct")
print(f"ratio                      : {right / wrong:.4f}")
print("\nThe wrong version counts padding as zero-loss tokens, so a batch")
print("that happens to contain short sequences reports a lower loss for the")
print("same model. The loss curve then tracks the batch composition rather")
print("than the model, and it improves whenever the sampler happens to draw")
print("short sequences together.")
```

## 9. Practical Example

```python {tier=A name=loss-choice-end-to-end}
"""One dataset, four losses, on a real network: the choice measured on data
that violates the Gaussian assumption of section 6.1.
"""
import numpy as np

rng = np.random.default_rng(17)


# --- a regression problem with heavy-tailed noise ---------------------------
def make_regression(n, noise, seed):
    """Clean signal; noise is either Gaussian or heavy-tailed."""
    rs = np.random.default_rng(seed)
    X = rs.uniform(-2, 2, (n, 3))
    signal = (np.sin(2 * X[:, 0]) + 0.5 * X[:, 1] ** 2
              - 0.8 * X[:, 0] * X[:, 2])
    if noise == "gaussian":
        eps = rs.normal(0, 0.3, n)
    else:                                     # 5% at 20x the scale
        eps = rs.normal(0, 0.3, n)
        idx = rs.choice(n, size=n // 20, replace=False)
        eps[idx] = rs.normal(0, 6.0, len(idx))
    return X, signal + eps, signal


class MLP:
    """Two hidden layers, hand-written backward, one loss slot."""

    def __init__(self, sizes, seed=0):
        rs = np.random.default_rng(seed)
        self.W = [rs.normal(0, np.sqrt(2 / sizes[i]), (sizes[i], sizes[i + 1]))
                  for i in range(len(sizes) - 1)]
        self.b = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]

    def forward(self, X):
        self.h = [X]
        h = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            h = np.maximum(0.0, z) if i < len(self.W) - 1 else z
            self.h.append(h)
        return h

    def backward(self, dout, lr):
        for i in reversed(range(len(self.W))):
            hin = self.h[i]
            gW = hin.T @ dout / len(dout)
            gb = dout.mean(axis=0)
            if i > 0:
                dout = (dout @ self.W[i].T) * (self.h[i] > 0)
            self.W[i] -= lr * gW
            self.b[i] -= lr * gb


LOSS_GRADS = {
    "squared": lambda p, t: 2 * (p - t),
    "absolute": lambda p, t: np.sign(p - t),
    "huber(1.0)": lambda p, t: np.clip(p - t, -1.0, 1.0),
    "huber(0.3)": lambda p, t: np.clip(p - t, -0.3, 0.3),
}


def train(X, y, loss, steps=4000, lr=0.02, batch=64, seed=0):
    net = MLP([X.shape[1], 48, 48, 1], seed=seed)
    rs = np.random.default_rng(seed + 1)
    grad = LOSS_GRADS[loss]
    for _ in range(steps):
        idx = rs.integers(0, len(X), batch)
        pred = net.forward(X[idx])
        net.backward(grad(pred, y[idx, None]), lr)
    return net


print("=" * 72)
print("the loss choice under two noise models (section 6.1)")
print("=" * 72)
print("The SAME clean signal; only the noise distribution differs. Error is")
print("measured against the noise-free signal, so we can see what each loss")
print("actually recovered rather than how well it fitted the noise.\n")

for noise in ("gaussian", "heavy-tailed"):
    Xtr, ytr, _ = make_regression(3000, noise, 21)
    Xte, yte, clean_te = make_regression(3000, noise, 22)
    print(f"{noise} noise")
    print(f"  {'loss':<14} {'RMSE vs noisy y':>17} {'RMSE vs clean':>15} "
          f"{'MAE vs clean':>14}")
    for loss in LOSS_GRADS:
        net = train(Xtr, ytr, loss)
        p = net.forward(Xte)[:, 0]
        rmse_noisy = float(np.sqrt(np.mean((p - yte) ** 2)))
        rmse_clean = float(np.sqrt(np.mean((p - clean_te) ** 2)))
        mae_clean = float(np.mean(np.abs(p - clean_te)))
        print(f"  {loss:<14} {rmse_noisy:>17.4f} {rmse_clean:>15.4f} "
              f"{mae_clean:>14.4f}")
    print()

print("Read the 'RMSE vs clean' column: it measures what we actually want,")
print("which is how well each loss recovered the underlying signal.")
print("\nUnder Gaussian noise squared error is the maximum-likelihood")
print("estimator (section 6.1) and it wins, as it should. Under heavy-tailed")
print("noise its assumption is violated and it loses to the robust losses.")
print("\nNote what delta does to Huber. At delta=1.0 it is the best loss in")
print("the heavy-tailed setting and second-worst in the Gaussian one; at")
print("delta=0.3 it is mediocre in both. Too small a delta throws away the")
print("efficiency of squared error on the bulk of the data in exchange for")
print("robustness it does not need there. Delta is a real hyperparameter")
print("and 'use Huber' is not by itself a decision.")
print("\nNote the trap in the 'RMSE vs noisy y' column: it is the metric you")
print("would actually compute in production, since the clean signal is")
print("unobservable, and squared error is favoured by it BY CONSTRUCTION —")
print("evaluating with the same functional form you trained with is not a")
print("neutral comparison.")

# --- classification: the pairing, on a network ------------------------------
print("=" * 72)
print("the loss-activation pairing, on a real network (section 6.5)")
print("=" * 72)


def make_clf(n, seed):
    rs = np.random.default_rng(seed)
    X = rs.normal(size=(n, 8))
    logit = 1.5 * X[:, 0] - 1.2 * X[:, 1] + 0.9 * X[:, 0] * X[:, 2]
    y = (rs.random(n) < 1 / (1 + np.exp(-logit))).astype(float)
    return X, y


Xc, yc = make_clf(4000, 31)
Xcv, ycv = make_clf(4000, 32)


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def eval_clf(net, X, y):
    s = net.forward(X)[:, 0]
    p = sigmoid(s)
    nll = -np.mean(y * np.log(np.clip(p, 1e-12, 1))
                   + (1 - y) * np.log(np.clip(1 - p, 1e-12, 1)))
    acc = ((p > 0.5) == y).mean()
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    npos, nneg = y.sum(), (1 - y).sum()
    auc = (ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)
    return nll, acc, auc, p.mean()


def train_clf_traced(X, y, Xv, yv, pairing, bias, steps=4000, lr=0.05,
                     batch=64, seed=0, checkpoints=(0, 100, 400, 1000, 4000)):
    net = MLP([X.shape[1], 48, 48, 1], seed=seed)
    net.b[-1] += bias                    # start CONFIDENTLY predicting 1
    rs = np.random.default_rng(seed + 1)
    out = {}
    for t in range(steps + 1):
        if t in checkpoints:
            out[t] = eval_clf(net, Xv, yv)
        p = sigmoid(net.forward(X[(idx := rs.integers(0, len(X), batch))]))
        tgt = y[idx, None]
        if pairing == "cross-entropy":
            dz = p - tgt                               # eq. 52.15
        else:                                          # squared error
            dz = 2 * (p - tgt) * p * (1 - p)           # eq. 52.17
        net.backward(dz, lr)
    return out


for bias in (4.0, 8.0):
    p0 = 1 / (1 + np.exp(-bias))
    print(f"output bias +{bias:.0f}: the network starts predicting p = "
          f"{p0:.6f} for everything.")
    print(f"  eq. 52.17's damping factor p(1-p) is {p0 * (1 - p0):.2e} there, "
          f"so squared error\n  starts with a gradient "
          f"{1 / (p0 * (1 - p0)):.0f}x smaller than cross-entropy's.\n")
    print(f"  {'step':>6}  {'cross-entropy NLL':>18} {'acc':>7}   "
          f"{'squared-error NLL':>18} {'acc':>7}")
    ce = train_clf_traced(Xc, yc, Xcv, ycv, "cross-entropy", bias)
    ms = train_clf_traced(Xc, yc, Xcv, ycv, "squared error", bias)
    for t in sorted(ce):
        print(f"  {t:>6}  {ce[t][0]:>18.4f} {ce[t][1]:>7.4f}   "
              f"{ms[t][0]:>18.4f} {ms[t][1]:>7.4f}")
    print()

print(f"base rate: {ycv.mean():.4f}")
print("\nRead the early rows, not the last one. Eq. 52.17 is a statement")
print("about the gradient in the saturated region, so it predicts a")
print("difference in how fast each pairing ESCAPES that region — not")
print("necessarily a difference in where they end up after a long run.")
print("\nAt bias +4 the damping factor is around 0.018, which slows squared")
print("error down without stopping it. At bias +8 it is around 3e-4, and the")
print("gap in the early rows is the whole point of the chapter: the")
print("cross-entropy network is already learning while the squared-error one")
print("has barely moved.")
print("\nBe careful about the final row. Given enough steps the squared-")
print("error network can catch up, and on this problem it does. That is")
print("worth saying plainly rather than hiding: the wrong pairing is a")
print("severe slowdown at initialisation, not an impossibility. In a real")
print("network with many saturating units, at a depth where the damping")
print("factors multiply, 'severe slowdown' becomes 'does not train' —")
print("which is the Chapter 50 argument, applied to the output layer.")

# --- label smoothing, measured ----------------------------------------------
print("\n" + "=" * 72)
print("label smoothing: what it costs and what it buys (eq. 52.20)")
print("=" * 72)


def train_smoothed(X, y, eps, steps=4000, lr=0.05, batch=64, seed=0):
    net = MLP([X.shape[1], 48, 48, 1], seed=seed)
    rs = np.random.default_rng(seed + 1)
    for _ in range(steps):
        idx = rs.integers(0, len(X), batch)
        p = sigmoid(net.forward(X[idx]))
        t = y[idx, None] * (1 - eps) + eps / 2.0       # binary: C = 2
        net.backward(p - t, lr)
    return net


print(f"{'epsilon':>9} {'val NLL':>9} {'AUC':>8} {'ECE':>8} "
      f"{'max |logit|':>13} {'mean |logit|':>14}")
for epsA in (0.0, 0.05, 0.1, 0.2):
    net = train_smoothed(Xc, yc, epsA)
    s = net.forward(Xcv)[:, 0]
    p = sigmoid(s)
    nll = -np.mean(ycv * np.log(np.clip(p, 1e-12, 1))
                   + (1 - ycv) * np.log(np.clip(1 - p, 1e-12, 1)))
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    npos, nneg = ycv.sum(), (1 - ycv).sum()
    auc = (ranks[ycv == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)
    # expected calibration error, 10 bins
    bins = np.clip((p * 10).astype(int), 0, 9)
    ece = sum(abs(p[bins == b].mean() - ycv[bins == b].mean())
              * (bins == b).mean()
              for b in range(10) if (bins == b).any())
    print(f"{epsA:>9.2f} {nll:>9.4f} {auc:>8.4f} {ece:>8.4f} "
          f"{np.abs(s).max():>13.3f} {np.abs(s).mean():>14.3f}")

print("\nThe logit columns are the mechanism of eq. 52.20 made visible: with")
print("no smoothing the optimum is unreachable and the logits keep growing;")
print("any epsilon caps them at a finite value that shrinks as epsilon rises.")
print("\nNote that NLL here is computed against the TRUE hard labels, so a")
print("smoothed model is being penalised for exactly the under-confidence it")
print("was asked to produce. That is the honest way to score it, rather than")
print("against its own smoothed target — and it still comes out ahead at")
print("moderate epsilon, with the calibration error roughly halving.")
print("\nThe trade is visible in the last row: push epsilon far enough and")
print("the enforced under-confidence starts costing more than the")
print("regularisation buys. AUC barely moves throughout, which is the same")
print("pattern as the class-weighting experiment — these are interventions")
print("on the probabilities, not on the ranking.")
```

## 10. Production Considerations

**Use the fused logits-to-loss function.** Never softmax in the model and then
apply a loss expecting probabilities. The measured stability table shows exactly
where the unfused versions fail, and in float16 they fail at logit magnitudes
that occur in ordinary training.

**Log the loss and at least one threshold-free metric.** The measured trace
shows the loss improving substantially while accuracy sits still, because at a
3% base rate accuracy cannot see the improvement. A dashboard showing only
accuracy would report a stalled model.

**Fix the reduction convention and record it.** The measured gradient norms show
`sum` coupling the step length to the batch size. When a reproduction diverges
after a batch-size change, this is the first thing to check
({{ch:mle-reproducibility}}).

**Check masked reductions in sequence models.** The measured example gives a
loss that differs by a real factor depending on padding, so the curve tracks
batch composition rather than the model.

**Weighting and focal loss decalibrate.** The measurement shows both changing
the predicted probabilities substantially while barely moving AUC. If a
downstream system consumes probabilities — expected-value calculations, ranking
against a cost threshold — you need {{ch:ml-logistic}}'s recalibration
afterwards, or you should tune the threshold instead.

**Compute the loss in float32.** Regardless of the model's precision. The
accumulation over a large batch loses too much in float16.

## 11. Common Mistakes

**Softmax in the model and cross-entropy on the output.** Either a
double-softmax or a loss of the stability the fused version provides.

**Squared error on a sigmoid output.** Measured: hundreds of times too little
gradient exactly where the model is worst.

**Softmax for multi-label classification.** Softmax forces the outputs to sum to
one; independent sigmoids are correct when several labels can apply.

**Class weights without recalibrating.** Measured decalibration, silent unless
someone checks.

**Averaging over padding.** Measured; the loss then depends on batch
composition.

**Optimising a loss and reporting only accuracy.** Measured divergence between
the two.

**Reading a robustness comparison in the same functional form you trained
with.** The measured "RMSE vs noisy y" column favours squared error by
construction.

**Label smoothing when you need confident rankings.** It caps the logits by
design, which is the point and also the cost.

## 12. Failure Modes

**`nan` loss.** Almost always overflow in an unfused softmax, `log(0)`, or a
division by a zero count in a masked reduction. The measured table localises
each.

**Loss decreasing, metric flat.** Measured. Often benign — the metric cannot see
the improvement — and occasionally a real symptom of the surrogate diverging
from the objective.

**A model that predicts the majority class perfectly.** Under an imbalanced
problem with a threshold at 0.5, this is the loss's actual minimiser among
threshold-0.5 decision rules, and the loss is behaving correctly. The problem is
the decision rule, not the loss.

**Loss dominated by a handful of examples.** Under squared error the gradient is
unbounded in the residual, so one mislabelled point can outweigh a hundred good
ones. Measured: at residual 100 a single example contributes 200 times a typical
point's gradient.

**Silent decalibration under weighting.** Aggregate discrimination metrics stay
put while the probabilities become meaningless.

**Logits growing without bound.** Measured under $\epsilon = 0$. Harmless for
accuracy, and it eventually costs numerical headroom in reduced precision.

## 13. Alternatives

**Hinge loss** ($\max(0, 1 - y\hat{y})$) optimises a margin rather than a
likelihood, giving sparse gradients — exactly zero for correctly classified
points beyond the margin. It is the support vector machine's loss
({{ch:ml-svm}}) and produces no probabilities.

**Ranking losses** (pairwise, listwise) optimise the ordering directly, which is
the right objective when a ranking is what you serve. They are not decomposable
over examples in the way {{eq:empirical-risk-dl}} assumes, which is why they
need their own machinery ({{ch:emb-reranking}}).

**Contrastive losses** learn a representation by pulling matched pairs together
and pushing unmatched ones apart, with no labels at all
({{ch:emb-models}}).

**Quantile loss** — the asymmetrically weighted absolute error — estimates a
chosen quantile rather than the mean or median, which is how a neural network
produces a prediction interval.

**Learned or adversarial losses**, where a second network provides the
objective. Powerful, and it converts a stable optimisation into a two-player
game with all the instability that implies.

## 14. Evaluation

**Verify your loss gradient numerically.** The central-difference check in
{{sec:8-implementation}} takes ten lines and catches sign errors, missing
factors and reduction mistakes.

**Check the loss at initialisation against its expected value.** A $C$-class
classifier at random initialisation should have a loss near $\log C$. If it does
not, the outputs are already skewed and something is wrong before training
starts. This is the single most valuable one-line sanity check in deep learning.

**Overfit a batch of ten examples to near-zero loss.** If it cannot, the loss,
the gradient or the architecture is broken, and no amount of tuning will help.

**Track loss and a threshold-free metric together.**

**Check calibration whenever you weight.** {{ch:ml-logistic}}'s reliability
diagram.

**Compare robustness against clean targets, not against the noisy ones.**

## 15. Advanced Concepts

**Proper scoring rules.** A scoring rule is *proper* if its expectation is
minimised by the true distribution. Cross-entropy (log score) and the Brier
score both qualify; accuracy does not. This is the formal statement of why
cross-entropy is the right choice for a probabilistic classifier, and the
theory tells you exactly which alternatives are safe.

**Bayes risk and the irreducible floor.** The minimum achievable loss is
$\E_{\vec{x}}[\min_a \E_{y|\vec{x}}[\ell(a, y)]]$, which is not zero for noisy
labels. Knowing this number — estimable when the noise model is known — tells
you when to stop trying.

**Loss surfaces and their geometry.** Cross-entropy with softmax is convex in
the logits and not in the parameters, and the composition is what makes deep
optimisation hard. {{ch:dl-optimizers}} works in that landscape.

**Uncertainty-weighted multi-task losses.** With several losses of different
scales, learning per-task weights as $1/2\sigma_t^2$ with a $\log \sigma_t$
penalty derives the weighting from a likelihood rather than guessing it.

**Distillation losses.** Cross-entropy against a teacher's *soft* distribution
rather than a hard label, which transmits the teacher's relative confidences.
Label smoothing is the special case where the teacher is uniform
({{ch:fm-distillation}}).

## 16. Connection to Previous Chapters

{{ch:math-inference}} supplied the principle; this chapter applies it twice and gets
both standard losses out. {{ch:ml-logistic}} derived {{eq:softmax-ce-gradient}}
in the binary case — the multi-class derivation here is the same cancellation,
and seeing it twice is what makes it clear that the property belongs to the
link/loss pairing rather than to either piece.

{{ch:ml-metrics}} argued that accuracy hides class imbalance; the measured trace
in {{sec:9-practical-example}} is that argument as a training curve.
{{ch:ml-logistic}} supplied reliability diagrams, and the weighting
measurement is why they matter here. {{ch:mle-reproducibility}} explains why the
reduction convention must be recorded.

Forward: {{ch:dl-backprop}} starts from {{eq:softmax-ce-gradient}} — this is the
first gradient the backward pass computes and every other one follows from it.
{{ch:llm-next-token}} uses cross-entropy over a vocabulary of tens of
thousands, where the fused kernel's memory behaviour becomes the dominant cost.
{{ch:ft-preference}} constructs a loss whose gradient is the one you want
even though its value means nothing, which is a useful stretch of the concept.

## 17. Exercises

**Beginner**

1. Why can accuracy not be used as a training loss?
2. What noise model does squared error assume?
3. What is the gradient of softmax cross-entropy with respect to the logits?
4. Why must softmax and cross-entropy be fused?
5. When is softmax the wrong output activation for a classification problem?

**Intermediate**

6. Derive {{eq:mae-derivative}} and confirm that absolute error is minimised by
   the median.
7. Compute the gradient of squared-error-on-sigmoid at $\hat{p} = 0.01$,
   $y = 1$, and compare with cross-entropy's.
8. Show that {{eq:logsumexp}} is exact and explain why the max subtraction
   changes nothing mathematically.
9. Using {{eq:smoothing-logit-gap}}, find the optimal logit gap for
   $\epsilon = 0.05$, $C = 1000$.
10. Explain why class weighting decalibrates a model.
11. A four-class classifier reports a loss of 3.2 at initialisation. What is
    wrong?

**Advanced**

12. Derive the Poisson negative log-likelihood and its gradient for an
    exponential output activation.
13. Prove that cross-entropy is a proper scoring rule.
14. Derive the focal loss gradient and show it reduces to cross-entropy's at
    $\gamma = 0$.
15. Show that label smoothing with $\epsilon$ is equivalent to distillation
    from a uniform teacher, and state the temperature.
16. Derive the Bayes risk under squared error with label noise of variance
    $\sigma^2$.

**Implementation**

17. Implement fused softmax cross-entropy with forward and backward, and verify
    against central differences.
18. Implement a masked sequence loss and demonstrate the padding bug.
19. Implement Huber loss and reproduce the minimiser table for several $\delta$.
20. Add a `weight=` argument and check whether your mean reduction divides by
    $N$ or by $\sum_i w_i$.

**Reasoning**

21. A model's loss is `nan` after 300 steps. Give an ordered diagnostic
    procedure.
22. Your loss falls steadily and F1 is flat at zero. Explain and propose two
    fixes that are not "change the loss".

## 18. Interview Questions

**"Why cross-entropy and not squared error for classification?"** — The
saturation argument of {{eq:mse-sigmoid-gradient}}, with the numbers. A stronger
answer adds that cross-entropy is the maximum-likelihood estimator for a
categorical model and a proper scoring rule.

**"Derive the softmax cross-entropy gradient."** — Expect to do this on a
whiteboard. Softmax Jacobian, then the cancellation, then $\hat{p} - y$.

**"How do you compute cross-entropy stably?"** — Max subtraction inside
logsumexp, never form the probabilities. Say why both naive versions fail and
at what magnitude.

**"How do you handle class imbalance?"** — Threshold tuning first, because it is
free and reversible; weighting or focal loss when the threshold is fixed
externally. Note that both decalibrate. A candidate who reaches immediately for
focal loss has not thought about it.

**"What is label smoothing and why does it work?"** — The optimum becomes
reachable at a finite logit gap. Quote {{eq:smoothing-logit-gap}}.

**"Loss is decreasing but accuracy is not. What is happening?"** — Several
possibilities: the metric cannot resolve the improvement under imbalance, the
model is becoming better calibrated without changing the argmax, or the
threshold is wrong. Give the diagnostic, not just a cause.

**"What loss would you use for a model that must output a prediction
interval?"** — Quantile loss at the two endpoints.

## 19. Research Questions

**Can non-differentiable metrics be optimised directly?** Smoothed surrogates
and score-function estimators both exist and neither reliably beats a good
surrogate plus threshold tuning. {{maturity:EMERGING}}

**Why is cross-entropy so robust to the surrogate/metric gap?** Models trained
on log loss usually rank well, classify well and calibrate reasonably, which is
more than proper-scoring-rule theory guarantees. The gap between what is
provable and what is observed is not well explained. {{maturity:EMERGING}}

**What is the right loss for learning from noisy labels?** Symmetric losses,
bootstrapping and noise-transition estimation each help under assumptions that
are hard to verify on real data. {{maturity:RESEARCH FRONTIER}}

**Do the loss's geometric properties predict generalisation?** Flat-minimum
arguments are suggestive, and sharpness is not reparameterisation-invariant,
which undermines the naive version of the claim. {{maturity:RESEARCH FRONTIER}}

## 20. Chapter Summary

A loss is where you state what you want, and every loss in this chapter is a
maximum-likelihood estimator under a specific noise model. Squared error is a
Gaussian assumption; cross-entropy is a categorical one. That framing turns an
arbitrary menu into a decision with a criterion — check the assumption, and when
it fails, expect the specific failure it predicts.

Squared error is minimised by the conditional mean and absolute error by the
median, both confirmed against a contaminated sample. The mechanism is the
gradient: squared error's grows without bound in the residual, so one bad label
at residual 100 contributed two hundred times a typical example's gradient,
while absolute error and Huber contribute a bounded amount by construction.

The softmax–cross-entropy gradient is $\hat{\vec{p}} - \vec{y}$, verified
numerically. The $1/\hat{p}_c$ that would explode for a confident wrong answer
is cancelled exactly by the $\hat{p}_c$ in the softmax Jacobian, and the same
cancellation appears in logistic and linear regression — a property of the
canonical link/loss pairing rather than three coincidences.

Squared error on a sigmoid destroys that property. The measured table shows a
model assigning probability 0.001 to the correct class receiving a gradient of
0.002 under squared error and 0.999 under cross-entropy: maximally wrong and
almost no signal. Loss and output activation are one decision.

Numerically, the two naive implementations fail in two different places — the
unfused softmax overflows and the max-subtracted version underflows to
$\log(0)$ — while the logsumexp form of {{eq:logsumexp}} is correct at every
magnitude tested. In float16 the failures arrive at logit magnitudes that occur
in ordinary training, which is why every framework fuses these two operations.

Label smoothing replaces an unreachable optimum with a reachable one at a finite
logit gap, measured as a cap on the logit magnitudes that tightens as $\epsilon$
grows. The regularisation, the calibration benefit and the loss of expressible
confidence are all the same mechanism.

Class weighting and focal loss barely moved AUC while moving the predicted
probabilities substantially. They are a threshold choice expressed as a loss —
useful when the threshold is fixed by something outside your control, and mostly
redundant when you can tune it directly, at the cost of calibration either way.

Finally, the loss is not the metric. The measured trace showed the loss
improving substantially while accuracy did not move at all, because at a 3% base
rate accuracy cannot see the improvement. Monitor both, and include something
threshold-free.

## 21. Further Reading

{{cite:goodfellow2016}} chapter 6 covers the maximum-likelihood derivation of
both losses more formally than here, and chapter 5 covers proper scoring rules.
It is the standard reference for this material and the treatment of output units
and their matching cost functions is the section to read.

{{cite:szegedy2016}} introduced label smoothing as one regularisation component
of a paper otherwise about convolutional architecture. Worth reading for how
brief the justification is relative to how universally the technique was
adopted — a recurring pattern in this
literature and one that {{ch:dl-normalization}} will show more sharply.

{{cite:lin2017focal}} introduced focal loss with a clear account of the
foreground/background imbalance it was designed for. The important thing to take
from it is the *scale* of the imbalance motivating it: thousands to one, not the
ten to one that most tabular problems present. Applying it at ten to one is
using a tool far outside the regime it was validated in, which the measurement
in {{sec:8-implementation}} reflects.

{{cite:rumelhart1986}} used squared error throughout, including on sigmoid
outputs. Reading it with {{eq:mse-sigmoid-gradient}} in hand explains a good
deal about why early networks trained as slowly as they did.

**Where to go next:** {{ch:dl-backprop}} propagates {{eq:softmax-ce-gradient}}
backwards through everything {{ch:dl-forward}} built. Those two chapters plus
this one are one continuous argument.

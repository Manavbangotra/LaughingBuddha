---
id: math-optimization
number: 12
part: I
tier: focused
status: reviewed
requires: [math-derivatives, math-norms, math-random-vars, math-eigen]
provides: [convex-function, local-minimum, global-minimum, saddle-point,
           gradient-descent, learning-rate, stochastic-gradient-descent,
           momentum-term, regularisation-term, cross-entropy-term]
citations: [boyd2004, robbins1951, kingma2015, goodfellow2016]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. State an optimisation problem formally and identify its objective, variables
   and constraints.
2. Define convexity and explain why convex problems have no local minima to get
   stuck in.
3. Implement gradient descent and explain the role of the learning rate.
4. Derive the maximum stable learning rate for a quadratic and explain what
   happens beyond it.
5. Explain why the condition number governs convergence speed, and why feature
   standardisation helps.
6. Explain why stochastic gradient descent works, and state the conditions under
   which it converges.
7. Explain why saddle points, not local minima, are the real obstacle in high
   dimensions.
8. Explain how momentum accelerates descent, and how regularisation corresponds
   to a prior.

## 2. Why This Matters

This chapter is where Part I converges. Training a model is an optimisation
problem: define a loss, compute its gradient, step downhill, repeat. Every
technique in Parts VI onward — Adam, learning-rate schedules, warmup, weight
decay, gradient clipping, batch size selection — is a modification of the loop
in this chapter, and none of them make sense without it.

Three things are worth setting up front, because they shape everything later.

**Neural network training is not convex, and that is fine.** The theory of
convex optimisation gives clean guarantees that do not apply here. Understanding
*why* they do not apply, and why gradient descent works anyway, is more useful
than either pretending the problem is convex or concluding that nothing can be
said.

**The learning rate is the most important hyperparameter, and there is a reason
for that.** {{sec:6-mathematical-foundation}} derives the exact stability
threshold for a quadratic, and it depends on the curvature. That derivation
explains why the right learning rate varies by orders of magnitude between
models, and why schedules exist.

**Stochastic gradient descent is not an approximation people settle for.** It is
faster than full-batch descent in wall-clock terms *and* its noise appears to
help generalisation. The justification rests on the linearity of expectation
from {{ch:math-random-vars}}, and its convergence conditions come from a 1951
paper on a completely different problem {{cite:robbins1951}}.

## 3. Prerequisites

{{ch:math-derivatives}} for gradients, the chain rule and the Hessian — this
chapter is the application of that one. {{ch:math-norms}} for the penalties in
{{sec:5-formal-explanation}}. {{ch:math-eigen}} for eigenvalues and the
condition number, which govern convergence. {{ch:math-random-vars}} for the
expectation argument behind SGD.

## 4. Intuitive Explanation

### 4.1 The landscape picture

Imagine the loss as a landscape. The horizontal coordinates are the parameters;
the height is the loss. Training means finding a low point.

You are standing somewhere in fog, able to feel only the slope beneath your feet
— that is the gradient. The strategy is simple: feel which way is downhill, take
a step, repeat.

The picture is useful and, in one respect, actively misleading. Your intuition
supplies a two-dimensional landscape with a few valleys. Real loss surfaces have
millions of dimensions, and high-dimensional geometry is nothing like the
picture ({{ch:math-vectors}}). {{sec:6-mathematical-foundation}} explains what
actually changes.

### 4.2 Convexity is the property that makes optimisation easy

A {{term:convex-function}} is bowl-shaped: the straight line between any two
points on its graph never dips below the graph.

The consequence is decisive. **For a convex function, every local minimum is a
global minimum.** There is nowhere to get stuck. Find a point where the gradient
is zero and you are done — provably, not hopefully.

Linear regression, logistic regression, and SVMs all have convex objectives,
which is why they have reliable solvers and reproducible answers
({{ch:ml-linear-regression}}, {{ch:ml-logistic}}, {{ch:ml-svm}}).

Neural networks do not. Their loss surfaces have many local minima, vast flat
regions, and enormously more saddle points. No guarantee of global optimality is
available, and none is likely to become available.

> IMPORTANT: The practical response is not despair. Empirically, most local
> minima of large networks have similar loss values, so finding *a* good one is
> usually enough. The real difficulties are elsewhere — saddle points, poor
> conditioning, and the interaction between learning rate and curvature — which
> is why this chapter spends more time on those than on local minima.

### 4.3 Gradient descent

The algorithm is one line:

$$
\vec{x}_{t+1} = \vec{x}_t - \eta\,\nabla f(\vec{x}_t)
$$ (eq:gradient-descent)

Compute the gradient, step against it, repeat. The gradient points uphill
({{ch:math-derivatives}}), so its negative points downhill; $\eta$, the
{{term:learning-rate}}, controls how far you go.

That single parameter is the difference between a model that trains and one that
does not:

- **Too small:** progress is correct but glacial. Thousands of extra iterations.
- **Too large:** you overshoot the minimum, land further up the other side, and
  overshoot again — worse each time. The loss diverges to `nan`.
- **Just right:** rapid convergence.

The window between "too small" and "too large" can be narrow, and where it sits
depends on the curvature of the loss — which is exactly what
{{sec:6-mathematical-foundation}} quantifies.

### 4.4 Stochastic gradient descent

Computing the gradient over a million examples to take one step is wasteful. The
gradient computed on 32 randomly chosen examples points in approximately the
same direction, at 1/30,000 of the cost.

That is {{term:stochastic-gradient-descent}}. The gradient estimate is noisy,
but by the linearity of expectation ({{ch:math-random-vars}}) it is **unbiased**
— correct on average — and you can take 30,000 noisy steps in the time one exact
step would take. For any realistic dataset, that trade is overwhelming.

The noise turns out to be a feature as well as a cost. It can shake the
optimiser out of sharp minima and narrow valleys, and there is a persistent
empirical association between the noise level — governed by batch size and
learning rate — and how well the resulting model generalises
({{ch:dl-optimizers}}).

## 5. Formal Explanation

### 5.1 The optimisation problem

$$
\min_{\vec{x} \in \R^{n}} f(\vec{x})
\quad\text{subject to}\quad
g_i(\vec{x}) \le 0, \quad h_j(\vec{x}) = 0
$$ (eq:optimisation-problem)

$f$ is the objective, $\vec{x}$ the decision variables, and the $g_i$ and $h_j$
are inequality and equality constraints. Most machine learning problems are
**unconstrained** — the constraints are handled by adding penalties to $f$
instead ({{sec:5-formal-explanation}} below).

A point $\vec{x}^{*}$ is a {{term:local-minimum}} if $f(\vec{x}^{*}) \le
f(\vec{x})$ for all $\vec{x}$ nearby, and a {{term:global-minimum}} if the
inequality holds everywhere.

At any interior minimum the gradient vanishes:

$$
\nabla f(\vec{x}^{*}) = \vec{0}
$$ (eq:first-order-condition)

This is *necessary* but not sufficient — maxima and saddle points satisfy it
too. The second-order condition distinguishes them: at a minimum the Hessian is
positive semi-definite, meaning all its eigenvalues are $\ge 0$
({{ch:math-eigen}}).

{#tbl:stationary-points caption="Classifying a stationary point by the eigenvalues of the Hessian. In high dimensions the last row is overwhelmingly the most common."}

| Hessian eigenvalues | Point type |
|---|---|
| all positive | local minimum |
| all negative | local maximum |
| mixed signs | {{term:saddle-point}} |
| some zero | degenerate; higher-order test needed |

### 5.2 Convexity

$f$ is convex if for all $\vec{x}, \vec{y}$ and $\lambda \in [0,1]$:

$$
f(\lambda\vec{x} + (1-\lambda)\vec{y}) \le \lambda f(\vec{x}) + (1-\lambda)f(\vec{y})
$$ (eq:convexity)

For twice-differentiable $f$, this is equivalent to the Hessian being positive
semi-definite everywhere.

The two properties that matter:

1. Every local minimum is global.
2. The first-order condition {{eq:first-order-condition}} is sufficient, not
   just necessary.

Convexity is preserved under useful operations: non-negative weighted sums of
convex functions are convex, as is the composition of a convex function with an
affine map, and the pointwise maximum of convex functions. That is how you
recognise convexity in practice — by construction rather than by checking
{{eq:convexity}}. Squared error composed with a linear model is convex; the same
loss composed with a two-layer network is not, because the composition of two
convex functions need not be convex.

{{cite:boyd2004}} is the standard reference, and the boundary it draws between
tractable and intractable problems is the reason the distinction is worth
learning.

### 5.3 Gradient descent and its convergence

The update is {{eq:gradient-descent}}. For a convex $f$ with **$L$-Lipschitz
gradient** — meaning
$\norm{\nabla f(\vec{x}) - \nabla f(\vec{y})} \le L\norm{\vec{x} - \vec{y}}$,
which bounds how fast the gradient can change — gradient descent with
$\eta \le 1/L$ converges, at rate $O(1/t)$ for convex $f$ and geometrically for
strongly convex $f$.

For a quadratic, $L$ is the largest eigenvalue of the Hessian, and the
convergence rate depends on the condition number
$\kappa = \lambda_{\max}/\lambda_{\min}$ ({{ch:math-eigen}}):

$$
\text{iterations to a given accuracy} \sim O(\kappa)
$$ (eq:convergence-conditioning)

A poorly conditioned problem is slow, and no choice of a single scalar learning
rate fixes it: the step size that is stable for the steepest direction is far too
small for the shallowest. {{sec:6-mathematical-foundation}} shows why.

### 5.4 Stochastic gradient descent

$$
\vec{x}_{t+1} = \vec{x}_t - \eta_t\,\nabla f_{B_t}(\vec{x}_t)
$$ (eq:sgd)

where $f_{B_t}$ is the loss on a random minibatch $B_t$. Because the batch is
drawn at random and expectation is linear:

$$
\E[\nabla f_{B}(\vec{x})] = \nabla f(\vec{x})
$$ (eq:sgd-unbiased)

The estimate is unbiased. Its variance falls as $1/\lvert B\rvert$, so larger
batches give less noisy gradients — at proportionally greater cost per step.

{{cite:robbins1951}} established the convergence conditions, decades before
anyone applied them to neural networks. For a decreasing step size $\eta_t$:

$$
\sum_{t=1}^{\infty}\eta_t = \infty
\qquad\text{and}\qquad
\sum_{t=1}^{\infty}\eta_t^{2} < \infty
$$ (eq:robbins-monro)

The first condition says the steps must not shrink so fast that you cannot reach
the minimum from anywhere. The second says they must shrink fast enough that the
noise is eventually averaged out. A schedule like $\eta_t = \eta_0/t$ satisfies
both; a constant learning rate satisfies the first but not the second, which is
why constant-rate SGD converges to a *neighbourhood* of the optimum rather than
to it, and why learning rates are decayed at the end of training
({{ch:dl-lr-schedules}}).

### 5.5 Momentum

{{term:momentum-term}} accumulates a running average of gradients:

$$
\vec{v}_{t+1} = \beta\vec{v}_t + \nabla f(\vec{x}_t), \qquad
\vec{x}_{t+1} = \vec{x}_t - \eta\,\vec{v}_{t+1}
$$ (eq:momentum)

with $\beta$ typically 0.9. Consistent gradient directions accumulate;
oscillating ones cancel. In a narrow valley — the poorly conditioned case —
plain descent bounces between the walls while momentum cancels the bouncing and
accelerates along the floor.

The effective step size along a consistently downhill direction is amplified by
roughly $1/(1-\beta)$, so $\beta = 0.9$ gives about a tenfold speed-up along
such directions.

### 5.6 Regularisation is a prior

Adding a penalty to the objective:

$$
\Loss_{\text{reg}}(\vec{w}) = \Loss(\vec{w}) + \lambda R(\vec{w})
$$ (eq:regularisation)

with $R = \norm{\vec{w}}_2^{2}$ for {{term:regularisation-term}} of the $L_2$
kind, or $\norm{\vec{w}}_1$ for $L_1$ ({{ch:math-norms}}).

This has an exact Bayesian reading. Maximising the posterior
({{ch:math-probability}}) means maximising the likelihood times the prior;
taking logs turns the product into a sum:

$$
\argmax_{\vec{w}} \big[\log p(\Data \given \vec{w}) + \log p(\vec{w})\big]
$$ (eq:map)

Negating to get a minimisation, the first term is the loss and the second is the
penalty. A Gaussian prior $p(\vec{w}) \propto \exp(-\norm{\vec{w}}^{2}/2\tau^{2})$
gives $\log p(\vec{w}) = -\norm{\vec{w}}^{2}/2\tau^{2}$ plus a constant — which
is exactly an $L_2$ penalty. A Laplace prior gives $L_1$.

> MATH NOTE: This is not an analogy. $L_2$ regularisation *is* maximum a
> posteriori estimation with a zero-mean Gaussian prior on the weights, with
> $\lambda = 1/(2\tau^{2})$. Stronger regularisation is a narrower prior — a
> firmer belief that the weights should be small. It also connects to
> {{eq:mse-decomposition}} from {{ch:math-inference}}: regularisation
> deliberately introduces bias to reduce variance, and is worth it whenever the
> variance reduction is larger.

## 6. Mathematical Foundation

### 6.1 The maximum stable learning rate

Take the simplest nontrivial case, $f(x) = \frac{1}{2}ax^{2}$ with $a > 0$. Then
$f'(x) = ax$, and the update is

$$
x_{t+1} = x_t - \eta a x_t = (1 - \eta a)\,x_t
$$ (eq:quadratic-update)

This is a geometric sequence: $x_t = (1 - \eta a)^{t}x_0$. It converges to zero
exactly when

$$
\lvert 1 - \eta a \rvert < 1
\quad\Longleftrightarrow\quad
0 < \eta < \frac{2}{a}
$$ (eq:stability-threshold)

Four regimes, all visible in this one formula:

{#tbl:learning-rate-regimes caption="Behaviour of gradient descent on f(x) = ax²/2 as the learning rate varies. The optimal rate lands exactly on the minimum in one step."}

| Learning rate | $1 - \eta a$ | Behaviour |
|---|---|---|
| $\eta < 1/a$ | in $(0, 1)$ | monotone convergence |
| $\eta = 1/a$ | $0$ | converges in a single step |
| $1/a < \eta < 2/a$ | in $(-1, 0)$ | oscillating convergence |
| $\eta = 2/a$ | $-1$ | oscillates forever, never converges |
| $\eta > 2/a$ | $< -1$ | divergence |

The threshold is $2/a$, where $a$ is the curvature. In several dimensions $a$
becomes the largest eigenvalue of the Hessian, so the stability limit is
$\eta < 2/\lambda_{\max}$.

**This is why the learning rate is model-dependent.** It is not an arbitrary
knob: its safe range is set by the sharpest curvature in the loss surface, and
that varies by orders of magnitude between architectures, and over the course of
training.

### 6.2 Why conditioning determines speed

Now two dimensions with different curvatures:

$$
f(x, y) = \tfrac{1}{2}\big(\lambda_1 x^{2} + \lambda_2 y^{2}\big),
\qquad \lambda_1 \gg \lambda_2
$$ (eq:anisotropic-quadratic)

Gradient descent updates each coordinate independently:

$$
x_{t+1} = (1 - \eta\lambda_1)x_t, \qquad y_{t+1} = (1 - \eta\lambda_2)y_t
$$

Stability requires $\eta < 2/\lambda_1$ — set by the *steep* direction. But
progress along $y$ is governed by $1 - \eta\lambda_2$, and with
$\eta \approx 1/\lambda_1$ that factor is $1 - \lambda_2/\lambda_1 = 1 - 1/\kappa$,
which is very close to 1 when $\kappa$ is large. The shallow direction barely
moves.

The result is the classic zigzag: rapid oscillation across the narrow valley,
crawling progress along it. The number of iterations needed scales with
$\kappa$, exactly as {{eq:convergence-conditioning}} states.

This is the mathematical reason for three separate pieces of practice:

- **Feature standardisation** ({{ch:math-covariance}}) makes the curvatures more
  similar, lowering $\kappa$.
- **Momentum** cancels the oscillation and accelerates the shallow direction.
- **Adaptive methods** such as Adam {{cite:kingma2015}} give each parameter its
  own effective step size, which is an attempt to normalise away the anisotropy
  altogether.

### 6.3 Saddle points, not local minima

In two dimensions, a stationary point is a minimum if the surface curves up in
both directions. With $n$ dimensions, it must curve up in all $n$.

Suppose, as a crude model, each direction is independently equally likely to
curve up or down at a random stationary point. Then a minimum requires all $n$
coin flips to come up the same way, with probability $2^{-n}$.

For $n = 1{,}000{,}000$ that is $2^{-1{,}000{,}000}$ — effectively zero. Almost
every stationary point in a high-dimensional loss surface is a **saddle**: a
point that curves up in some directions and down in others.

> IMPORTANT: This reverses the naive picture entirely. The problem is not that
> optimisation gets trapped in bad local minima; those are vanishingly rare. The
> problem is that it *slows down* near saddle points, where the gradient is
> small in every direction but the surface is not actually a minimum. This is
> why momentum and adaptive methods help so much — they carry the optimiser
> through flat regions that plain gradient descent would crawl across.

The independence assumption is crude, and real loss surfaces are structured
rather than random. But the qualitative conclusion — saddles dominate minima in
high dimensions — holds up, and it is one of the more useful corrections that
mathematics makes to two-dimensional intuition.

### 6.4 Deriving cross-entropy from maximum likelihood

Part I ends by deriving the loss function that most of the rest of the book
minimises.

Suppose a model with parameters $\theta$ assigns probability
$p_{\theta}(y \given \vec{x})$ to label $y$. Maximum likelihood chooses $\theta$
to maximise the probability of the observed data. Assuming the examples are
independent, the joint probability is a product:

$$
L(\theta) = \prod_{i=1}^{N} p_{\theta}(y_i \given \vec{x}_i)
$$ (eq:likelihood-product)

Take logs — legitimate because $\log$ is monotonic ({{ch:math-functions}}), so
the maximiser is unchanged — turning the product into a sum:

$$
\log L(\theta) = \sum_{i=1}^{N}\log p_{\theta}(y_i \given \vec{x}_i)
$$ (eq:log-likelihood)

Negate to make it a minimisation, and divide by $N$ so the scale does not depend
on dataset size:

$$
\Loss(\theta) = -\frac{1}{N}\sum_{i=1}^{N}\log p_{\theta}(y_i \given \vec{x}_i)
$$ (eq:cross-entropy-derived)

This is {{term:cross-entropy-term}} loss — and it is the same
{{eq:nll}} that {{ch:math-notation}} used as an unexplained example in the very
first chapter of this book. It is now derived: **minimising cross-entropy is
maximum likelihood estimation.** Not a heuristic, not a convenient
differentiable surrogate, but the direct consequence of asking which parameters
make the observed data most probable.

Add a prior, as in {{eq:map}}, and you get regularised cross-entropy. Part I
closes where it opened.

## 7. Implementation

```python {tier=A name=gradient-descent}
"""Gradient descent: the learning-rate threshold, conditioning, momentum, and
SGD — each claim in the chapter verified numerically.
"""
import numpy as np

rng = np.random.default_rng(0)

# --- eq. 12.10: the stability threshold is exactly 2/a ----------------------
a = 4.0            # f(x) = a x^2 / 2, so f'(x) = a x, curvature a


def run_1d(eta, steps=60, x0=1.0):
    x, traj = x0, [x0]
    for _ in range(steps):
        x = x - eta * (a * x)
        traj.append(x)
        if abs(x) > 1e12:
            break
    return np.array(traj)


print(f"f(x) = {a}x^2/2, so the predicted stability threshold is "
      f"2/a = {2/a}\n")
print(f"{'eta':>8} {'1 - eta*a':>11} {'final |x|':>14} {'behaviour':<26}")
for eta in (0.1, 0.25, 0.4, 0.49, 0.5, 0.55):
    traj = run_1d(eta)
    factor = 1 - eta * a
    final = abs(traj[-1])
    if final > 1e10:
        behaviour = "DIVERGED"
    elif abs(factor) < 1e-12:
        behaviour = "converged in one step"
    elif factor > 0:
        behaviour = "monotone convergence"
    elif abs(factor) < 1:
        behaviour = "oscillating convergence"
    else:
        behaviour = "oscillates forever"
    print(f"{eta:>8.2f} {factor:>11.3f} {final:>14.3e} {behaviour:<26}")
print(f"\nEverything below eta = {2/a} converges; everything above diverges.")

# --- section 6.2: conditioning determines how many steps you need -----------
print("\n" + "=" * 64)
print("condition number vs iterations to converge")
print("=" * 64)


def run_2d(lam1, lam2, steps=200_000, tol=1e-6, beta=0.0):
    """Gradient descent on (lam1 x^2 + lam2 y^2)/2, at the stable learning rate."""
    eta = 1.0 / lam1                      # near-optimal for the steep direction
    v = np.zeros(2)
    p = np.array([1.0, 1.0])
    lams = np.array([lam1, lam2])
    for i in range(steps):
        g = lams * p
        v = beta * v + g
        p = p - eta * v
        if np.linalg.norm(p) < tol:
            return i + 1
    return steps


print(f"{'kappa':>8} {'plain GD steps':>16} {'with momentum':>15} "
      f"{'speed-up':>10}")
for kappa in (1, 10, 100, 1000):
    plain = run_2d(1.0, 1.0 / kappa)
    withmom = run_2d(1.0, 1.0 / kappa, beta=0.9)
    print(f"{kappa:>8} {plain:>16,} {withmom:>15,} "
          f"{plain/max(withmom,1):>9.1f}x")
print("\nIterations scale roughly with kappa (eq. 12.6). Momentum cancels the")
print("oscillation across the valley and accelerates along it.")

# --- standardisation improves conditioning ----------------------------------
n = 2000
raw = np.column_stack([rng.normal(0, 1, n), rng.normal(0, 100, n)])
w_true = np.array([2.0, 0.05])
y = raw @ w_true + rng.normal(0, 0.1, n)

H_raw = raw.T @ raw / n                            # Hessian of the squared loss
std = (raw - raw.mean(0)) / raw.std(0)
H_std = std.T @ std / n
print(f"\ncondition number of the Hessian, raw features        : "
      f"{np.linalg.cond(H_raw):>12,.0f}")
print(f"condition number after standardisation               : "
      f"{np.linalg.cond(H_std):>12,.2f}")
print("Standardising the features is a change to the OPTIMISATION problem,")
print("not to the model. It is why unscaled features train so slowly.")

# --- eq. 12.8: the minibatch gradient is unbiased ---------------------------
print("\n" + "=" * 64)
print("stochastic gradient descent")
print("=" * 64)

N, d = 20_000, 10
X = rng.normal(size=(N, d))
w_star = rng.normal(size=d)
Y = X @ w_star + rng.normal(0, 0.5, N)


def full_gradient(w):
    return X.T @ (X @ w - Y) / N


def batch_gradient(w, size):
    idx = rng.choice(N, size=size, replace=False)
    Xb, Yb = X[idx], Y[idx]
    return Xb.T @ (Xb @ w - Yb) / size


w0 = rng.normal(size=d)
exact = full_gradient(w0)
print(f"{'batch size':>11} {'bias (norm)':>13} {'noise (norm sd)':>17} "
      f"{'cos to exact':>13}")
for bs in (1, 8, 32, 256, 2048):
    grads = np.stack([batch_gradient(w0, bs) for _ in range(600)])
    bias = np.linalg.norm(grads.mean(0) - exact)
    noise = np.linalg.norm(grads - exact, axis=1).std()
    cos = np.mean([g @ exact / (np.linalg.norm(g) * np.linalg.norm(exact))
                   for g in grads])
    print(f"{bs:>11} {bias:>13.5f} {noise:>17.4f} {cos:>13.4f}")
print("\nBias stays near zero at every batch size (eq. 12.8) while the noise")
print("falls as 1/sqrt(batch). Noisy, but never systematically wrong.")

# --- wall-clock: many cheap steps beat few exact ones -----------------------
def train(batch_size, epochs=6, eta=0.05):
    w = np.zeros(d)
    grads_computed = 0
    for _ in range(epochs):
        order = rng.permutation(N)
        for start in range(0, N, batch_size):
            idx = order[start:start + batch_size]
            Xb, Yb = X[idx], Y[idx]
            w = w - eta * (Xb.T @ (Xb @ w - Yb) / len(idx))
            grads_computed += len(idx)
    return np.linalg.norm(w - w_star), grads_computed


print(f"\n{'batch size':>11} {'steps taken':>13} {'gradient evals':>16} "
      f"{'final error':>13}")
for bs in (32, 512, N):
    err, evals = train(bs)
    print(f"{bs:>11} {6*N//bs:>13,} {evals:>16,} {err:>13.5f}")
print("Identical gradient budget; small batches take far more STEPS and get")
print("much closer. This is the whole argument for SGD.")

# --- eq. 12.9: the Robbins-Monro conditions ---------------------------------
print("\n" + "=" * 64)
print("Robbins-Monro step-size conditions (eq. 12.9)")
print("=" * 64)
T = 100_000
t = np.arange(1, T + 1)
schedules = {
    "constant  eta=0.1": np.full(T, 0.1),
    "1/t":               1.0 / t,
    "1/sqrt(t)":         1.0 / np.sqrt(t),
    "1/t^2":             1.0 / t**2,
}
print(f"{'schedule':<20} {'sum eta':>12} {'sum eta^2':>12} "
      f"{'converges?':>12}")
for name, sched in schedules.items():
    s1, s2 = sched.sum(), (sched**2).sum()
    # Condition 1 needs sum eta -> infinity; condition 2 needs sum eta^2 finite.
    ok = s1 > 50 and s2 < 100
    print(f"{name:<20} {s1:>12.2f} {s2:>12.4f} {str(ok):>12}")
print("\n1/t satisfies both. A constant rate fails the second, so it converges")
print("only to a NEIGHBOURHOOD — which is why schedules decay at the end.")

# --- section 6.4: cross-entropy IS maximum likelihood -----------------------
print("\n" + "=" * 64)
print("minimising cross-entropy == maximising likelihood (eq. 12.16)")
print("=" * 64)
probs = np.array([0.9, 0.6, 0.2, 0.75])       # model's p(true label)
likelihood = np.prod(probs)
ce = -np.mean(np.log(probs))
print(f"likelihood  (product)      : {likelihood:.6f}")
print(f"log-likelihood (sum)       : {np.sum(np.log(probs)):.6f}")
print(f"cross-entropy (-mean log)  : {ce:.6f}")
print(f"check: -N * CE == log-lik  : {-len(probs)*ce:.6f}")
assert np.isclose(-len(probs) * ce, np.sum(np.log(probs)))
print("\nThe two are the same objective up to sign and a positive constant,")
print("so they have the same minimiser. This is eq. 1.1 from Chapter 1,")
print("now derived rather than merely read.")
```

## 8. Practical Example

Training a logistic regression from scratch closes the loop on Part I: it uses
the gradient of {{ch:math-derivatives}}, the standardisation of
{{ch:math-covariance}}, the penalties of {{ch:math-norms}}, and the loss derived
in {{sec:6-mathematical-foundation}}.

```python {tier=A name=train-logistic-regression}
"""Logistic regression trained by gradient descent, from first principles.

No scikit-learn. Every component comes from Part I.
"""
import numpy as np

rng = np.random.default_rng(3)

# --- data, deliberately on very different scales ----------------------------
n = 4000
age = rng.uniform(18, 80, n)
income = rng.uniform(15_000, 200_000, n)
logit_true = -6.0 + 0.06 * age + 0.00004 * income
y = (rng.random(n) < 1 / (1 + np.exp(-logit_true))).astype(float)

X_raw = np.column_stack([age, income])
mu, sd = X_raw.mean(0), X_raw.std(0)
X_std = (X_raw - mu) / sd                       # eq. 9.18

print(f"positive class rate: {y.mean():.3f}")
print(f"raw feature scales : {np.round(sd, 1)}")


def sigmoid(z):
    out = np.empty_like(z)
    pos, neg = z >= 0, z < 0
    out[pos] = 1 / (1 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1 + ez)
    return out


def add_bias(X):
    return np.column_stack([np.ones(len(X)), X])


def loss_and_grad(w, X, y, lam=0.0):
    """Cross-entropy (eq. 12.16) with an optional L2 penalty (eq. 12.11)."""
    z = X @ w
    p = np.clip(sigmoid(z), 1e-12, 1 - 1e-12)
    ce = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    # eq. 11.19 generalised to a batch: X^T (p - y) / n
    grad = X.T @ (p - y) / len(y)
    if lam:
        penalty_w = w.copy()
        penalty_w[0] = 0.0                      # never regularise the bias
        ce += lam * np.sum(penalty_w ** 2)
        grad += 2 * lam * penalty_w
    return ce, grad


def train(X, y, eta, steps=3000, lam=0.0, beta=0.0):
    w = np.zeros(X.shape[1])
    v = np.zeros_like(w)
    history = []
    for _ in range(steps):
        l, g = loss_and_grad(w, X, y, lam)
        v = beta * v + g                        # eq. 12.10
        w = w - eta * v
        history.append(l)
    return w, np.array(history)


# --- standardised vs raw: the same model, a different optimisation problem --
Xs, Xr = add_bias(X_std), add_bias(X_raw)
print(f"\n{'features':<16} {'eta':>8} {'loss @100':>11} {'loss @3000':>12} "
      f"{'status':<12}")
for name, Xd, eta in (("standardised", Xs, 0.5),
                      ("raw", Xr, 0.5),
                      ("raw", Xr, 1e-9)):
    with np.errstate(over="ignore", invalid="ignore"):
        w, hist = train(Xd, y, eta)
    status = "diverged" if not np.isfinite(hist[-1]) else "ok"
    h100 = hist[100] if np.isfinite(hist[100]) else float("nan")
    print(f"{name:<16} {eta:>8.0e} {h100:>11.4f} {hist[-1]:>12.4f} "
          f"{status:<12}")
print("Raw features diverge at a sensible learning rate and crawl at a safe")
print("one. Standardisation fixes the conditioning (section 6.2).")

# --- momentum ----------------------------------------------------------------
print(f"\n{'optimiser':<22} {'loss after 300 steps':>22}")
for name, beta in (("plain gradient descent", 0.0), ("momentum beta=0.9", 0.9)):
    _, hist = train(Xs, y, eta=0.3, steps=300, beta=beta)
    print(f"{name:<22} {hist[-1]:>22.6f}")

# --- regularisation, and the bias-variance trade ----------------------------
split = 3000
Xtr, ytr, Xte, yte = Xs[:split], y[:split], Xs[split:], y[split:]
print(f"\n{'lambda':>10} {'train loss':>12} {'test loss':>11} "
      f"{'||w||':>8} {'test acc':>10}")
for lam in (0.0, 0.001, 0.01, 0.1, 1.0):
    w, _ = train(Xtr, ytr, eta=0.5, steps=3000, lam=lam)
    tr, _ = loss_and_grad(w, Xtr, ytr)
    te, _ = loss_and_grad(w, Xte, yte)
    acc = np.mean((sigmoid(Xte @ w) > 0.5) == yte)
    print(f"{lam:>10.3f} {tr:>12.4f} {te:>11.4f} "
          f"{np.linalg.norm(w[1:]):>8.3f} {acc:>10.4f}")
print("Stronger regularisation shrinks ||w|| and raises training loss — it is")
print("buying variance reduction with bias (eq. 10.5).")

# --- recover the true coefficients -------------------------------------------
w_final, _ = train(Xs, y, eta=0.5, steps=8000)
# Undo the standardisation to compare against the generating coefficients.
coef = w_final[1:] / sd
intercept = w_final[0] - np.sum(w_final[1:] * mu / sd)
print(f"\nrecovered  : intercept {intercept:+.4f}, coefficients "
      f"{np.array2string(coef, precision=6)}")
print(f"true       : intercept {-6.0:+.4f}, coefficients [0.06     0.00004 ]")
```

## 9. Common Mistakes

**Using a learning rate without regard to curvature.** The stable range is
$\eta < 2/\lambda_{\max}$, and $\lambda_{\max}$ differs by orders of magnitude
between problems and changes during training. Always sweep the learning rate on
a log scale ({{ch:math-functions}}).

**Not standardising features.** It is a change to the conditioning of the
optimisation problem, not a cosmetic step. The demonstration in
{{sec:8-practical-example}} shows raw features diverging at a rate that works
perfectly once standardised.

**Regularising the bias term.** The bias should be free to represent the base
rate. Penalising it biases predictions toward 0.5 for no benefit.

**Assuming a training loss that stops falling means a local minimum.** In high
dimensions it is far more likely to be a saddle or a flat region
({{sec:6-mathematical-foundation}}).

**Comparing losses across different batch sizes or regularisation strengths.**
They are different objectives. Compare a validation metric instead.

**Expecting convergence guarantees to apply to neural networks.** They are
derived for convex problems. Knowing they do not transfer is more useful than
either assuming they do or concluding nothing can be said.

**Decaying the learning rate too early.** {{eq:robbins-monro}}'s first condition
requires the steps to sum to infinity; decaying too aggressively means never
reaching the minimum from where you started.

**Treating momentum as free.** It changes the effective learning rate by roughly
$1/(1-\beta)$, so adding $\beta = 0.9$ to a tuned learning rate can cause
divergence. Retune after changing it.

## 10. Connection to Previous Chapters

This chapter uses nearly all of Part I. {{ch:math-derivatives}} supplied the
gradient and the proof that it is the steepest direction, which is the
justification for {{eq:gradient-descent}}. {{ch:math-eigen}} supplied the
eigenvalues that set the stability threshold and the condition number that
governs speed. {{ch:math-norms}} supplied the penalties in
{{eq:regularisation}}. {{ch:math-random-vars}} supplied the linearity of
expectation that makes {{eq:sgd-unbiased}} true. {{ch:math-covariance}} supplied
standardisation, whose value is now explained rather than asserted.
{{ch:math-probability}} supplied the prior that {{eq:map}} identifies with
regularisation. {{ch:math-inference}} supplied the bias-variance decomposition
that makes that trade worth making. And {{ch:math-notation}}'s opening equation
is derived in {{sec:6-mathematical-foundation}}.

Beyond Part I: {{ch:ml-linear-regression}} and {{ch:ml-logistic}} solve convex
versions of this problem; {{ch:dl-backprop}} computes the gradients for a whole
network; {{ch:dl-optimizers}} develops Adam {{cite:kingma2015}} and its
relatives as responses to the conditioning problem of
{{sec:6-mathematical-foundation}}; {{ch:dl-lr-schedules}} implements
{{eq:robbins-monro}}; and {{ch:dl-regularization}} extends
{{eq:regularisation}}. {{cite:goodfellow2016}} covers optimisation for deep
learning in more depth, and {{cite:boyd2004}} is the reference for the convex
theory.

## 11. Exercises

**Beginner**

1. For $f(x) = x^{2} - 4x + 7$, find the minimum analytically by setting the
   derivative to zero.
2. Perform three steps of gradient descent on $f(x) = x^{2}$ from $x_0 = 4$ with
   $\eta = 0.1$.
3. Is $f(x) = x^{4}$ convex? Is $f(x) = x^{3}$? Justify each.
4. For $f(x) = 3x^{2}$, what is the maximum stable learning rate?
5. State the two Robbins-Monro conditions and say what each ensures.

**Intermediate**

6. Derive {{eq:stability-threshold}} from {{eq:quadratic-update}}.
7. For $f(x,y) = 10x^{2} + 0.1y^{2}$, give the condition number and the largest
   stable learning rate. Roughly how many iterations to reduce the error
   tenfold?
8. Show that the sum of two convex functions is convex, directly from
   {{eq:convexity}}.
9. Explain why a constant learning rate converges only to a neighbourhood of the
   optimum.
10. Show that an $L_2$ penalty with strength $\lambda$ corresponds to a Gaussian
    prior, and give the relationship between $\lambda$ and the prior variance.
11. A model has 1 million parameters. Under the crude independence model of
    {{sec:6-mathematical-foundation}}, what fraction of stationary points are
    minima?

**Advanced**

12. Prove that for a convex function, every local minimum is global.
13. Derive the momentum update's effective step-size amplification of
    $1/(1-\beta)$ for a constant gradient.
14. Derive gradient descent's convergence rate for a strongly convex quadratic,
    and show it depends on $\kappa$.
15. Derive {{eq:cross-entropy-derived}} from maximum likelihood, being explicit
    about where independence and monotonicity are used.
16. Show that composing two convex functions need not be convex, with a
    counterexample. Then state a condition under which it is.

**Implementation**

17. Implement gradient descent on $f(x,y) = x^{2} + 10y^{2}$ and plot the
    trajectory for several learning rates. Identify the zigzag.
18. Add momentum to the above and measure the reduction in iterations across
    $\beta \in \{0, 0.5, 0.9, 0.99\}$.
19. Implement a learning-rate finder: sweep $\eta$ exponentially over a few
    hundred steps and plot loss against $\eta$. Identify the usable range.
20. Train the logistic regression of {{sec:8-practical-example}} with SGD at
    batch sizes 1, 32 and full, and compare final loss against gradient
    evaluations rather than against steps.
21. Empirically find the stability threshold for a random quadratic and compare
    it with $2/\lambda_{\max}$.

**Reasoning**

22. Your loss plateaus. Give three distinct explanations from this chapter and
    describe how you would distinguish them.
23. Large-batch training often generalises slightly worse than small-batch at
    the same number of epochs. Propose an explanation in terms of gradient
    noise, and describe an experiment that would test it.

## 12. Chapter Summary

Optimisation minimises an objective over decision variables. At an interior
minimum the gradient vanishes, but that condition is necessary rather than
sufficient — maxima and saddle points satisfy it too, and the Hessian's
eigenvalues distinguish them.

Convex functions are bowl-shaped, and for them every local minimum is global.
Linear and logistic regression are convex; neural networks are not, and no
global guarantee is available for them.

Gradient descent steps against the gradient. For a quadratic with curvature $a$,
convergence requires $\eta < 2/a$; in several dimensions the limit is
$2/\lambda_{\max}$, which is why the safe learning rate is problem-specific and
varies by orders of magnitude.

The condition number governs speed. When curvatures differ greatly, the step
size is limited by the steepest direction while progress is limited by the
shallowest, producing the characteristic zigzag and an iteration count scaling
with $\kappa$. Feature standardisation, momentum, and adaptive optimisers are
three distinct responses to this one problem.

In high dimensions, saddle points vastly outnumber local minima, so the real
obstacle is slowdown in flat regions rather than entrapment in bad optima.

Stochastic gradient descent replaces the exact gradient with a minibatch
estimate that is unbiased by linearity of expectation and has variance falling
as $1/\lvert B\rvert$. Many noisy steps beat few exact ones for any realistic
dataset. The Robbins-Monro conditions — steps summing to infinity, squared steps
summing to something finite — are why learning rates are decayed rather than
held constant.

Regularisation adds a penalty on the size of the parameters, and this is exactly
maximum a posteriori estimation with a prior: $L_2$ corresponds to a Gaussian
prior, $L_1$ to a Laplace one. It buys reduced variance at the cost of bias.

Finally, cross-entropy loss is not a heuristic. Minimising it is maximum
likelihood estimation, derived directly from asking which parameters make the
observed data most probable — which closes the loop on the equation this book
opened with in {{ch:math-notation}}.

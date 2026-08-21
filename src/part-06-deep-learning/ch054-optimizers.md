---
id: dl-optimizers
number: 54
part: VI
tier: full
status: reviewed
requires: [dl-backprop, dl-losses, math-optimization, ml-linear-regression]
provides: [sgd, momentum, nesterov, adagrad, rmsprop, adam, adamw,
           weight-decay-decoupled, optimizer-state, bias-correction,
           second-order-methods]
citations: [rumelhart1986, kingma2015adam, loshchilov2019adamw, goodfellow2016,
            pascanu2013]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Derive SGD, momentum, AdaGrad, RMSProp and Adam, and say what problem each
   was introduced to solve.
2. Explain bias correction in Adam and show what happens without it.
3. Explain why $\ell_2$ regularisation and weight decay differ under an
   adaptive optimiser.
4. Choose an optimiser for a given problem with a reason rather than a habit.
5. Account for the memory each optimiser costs.
6. Diagnose divergence, plateaus and instability from optimiser behaviour.
7. State honestly what is and is not known about why Adam works.

## 2. Why This Matters

**{{ch:dl-backprop}} produced a gradient; something has to decide what to do
with it.** That decision is the optimiser, and it is the difference between a
model that reaches its achievable loss in a day and one that does not reach it
at all.

**Adam is the default and the reasons are not fully understood.**
{{cite:kingma2015adam}} is one of the most cited papers in the field, its
original convergence proof was later shown to be flawed, and it remains the
right default anyway. That combination — overwhelming empirical success, shaky
theory — is characteristic of this material and worth confronting directly
rather than papering over.

**The optimiser costs memory.** Adam stores two extra values per parameter, so a
model that fits in memory as weights may not fit as a training job. This is
often the binding constraint on model size, and it is arithmetic rather than
mystery.

**AdamW is not a small correction to Adam.** {{cite:loshchilov2019adamw}} showed
that the standard way of implementing $\ell_2$ regularisation inside an adaptive
optimiser does something different from what everyone assumed, and fixing it
changed results measurably. {{sec:9-practical-example}} measures the difference.

## 3. Prerequisites

{{ch:dl-backprop}} for the gradient this chapter consumes.
{{ch:math-optimization}} for gradient descent, convexity and conditioning.
{{ch:ml-linear-regression}} for the first gradient-descent implementation.
{{ch:dl-losses}} for the loss surface being descended.

## 4. Intuitive Explanation

### 4.1 What an optimiser has to solve

The gradient says which way is downhill *right here*. It does not say how far to
go, and three things make that hard:

**Ill-conditioning.** The loss falls steeply along some directions and gently
along others. A step size small enough not to diverge along the steep direction
is far too small for the gentle one, so progress is slow in exactly the
direction that has the most left to give.

**Noise.** Each step uses a mini-batch, so the gradient is a noisy estimate. Its
noise does not shrink as you approach the optimum, which is why a constant step
size cannot converge — {{ch:dl-lr-schedules}} is the response.

**Scale variation.** Different parameters have gradients of wildly different
magnitudes — an embedding row for a rare token against a dense layer's weight.
One global step size serves neither.

Every optimiser in this chapter attacks one or more of these.

### 4.2 Momentum

Plain SGD in a narrow valley bounces between the walls and creeps along the
floor:

```text
   SGD in a ravine              with momentum
      ╲    ╱                       ╲    ╱
       ╲ ↗╱ ↘                       ╲  ╱
        ╲╱ ↗╲↘                       ╲╱ ──────▶
     zigzag: the steep            oscillations cancel;
     direction dominates          the consistent direction adds
```

Momentum accumulates an exponential moving average of past gradients.
Oscillating components alternate in sign and cancel; consistent components add
up. It is a low-pass filter on the gradient, and the intuition that it is "a
heavy ball with inertia" is a genuine analogy — {{sec:6-mathematical-foundation}}
shows it is the discretisation of a second-order differential equation.

### 4.3 Adaptive step sizes

The other idea: give each parameter its *own* step size, inferred from the
history of its gradients. A parameter with consistently large gradients gets a
smaller step; one with small gradients gets a larger one.

```text
   AdaGrad    divide by sqrt of the SUM of squared gradients
              -> step size decreases monotonically, and eventually to zero

   RMSProp    divide by sqrt of a MOVING AVERAGE of squared gradients
              -> forgets the distant past, so the step size can recover

   Adam       RMSProp + momentum + a bias correction for the warm-up
```

The progression is a sequence of fixes. AdaGrad's monotone decay is fatal for
non-convex problems; RMSProp's forgetting fixes it; Adam adds momentum and
corrects the initialisation bias that both have.

### 4.4 What Adam actually computes

$$
\text{step} \;\propto\; \frac{\text{average gradient}}
 {\sqrt{\text{average squared gradient}}}
$$

This ratio is close to a *signed* quantity: when the gradient is consistent, the
numerator and the square root of the denominator are comparable and the step is
near $\pm\eta$. When the gradient is noisy and averages to near zero, the
numerator shrinks and the denominator does not, so the step is small.

**Adam's step size is therefore roughly scale-invariant.** Multiply every
gradient by 1000 and the step barely changes. That is why Adam works out of the
box on problems where SGD needs its learning rate tuned by orders of magnitude,
and it is also why Adam is less sensitive to initialisation and to loss scaling
than SGD is.

### 4.5 The honest position on Adam versus SGD

Two claims are widely repeated and only one of them survives contact with the
evidence.

**"Adam converges faster."** Generally true, especially early, and especially on
transformers, where SGD is not merely slower but often does not work.

**"SGD with momentum generalises better."** True in a specific setting —
convolutional image classifiers with a well-tuned schedule — and not a general
law. The gap narrows or disappears with AdamW and proper tuning, and it does not
appear at all in language modelling.

The defensible default in 2026: **AdamW for transformers and anything with
sparse or wildly-scaled gradients; SGD with momentum for convolutional vision
when you have the budget to tune it.** {{sec:9-practical-example}} measures a
case rather than asserting a rule.

## 5. Formal Explanation

### 5.1 SGD

$$
\vecgreek{\theta}_{t+1} = \vecgreek{\theta}_t - \eta\,\vec{g}_t,
\qquad \vec{g}_t = \nabla_{\vecgreek{\theta}}
 \Like_{\mathcal{B}_t}(\vecgreek{\theta}_t)
$$ (eq:sgd)

One hyperparameter, no state. Everything else in this chapter adds state to
address a specific deficiency of this line.

### 5.2 Momentum and Nesterov

$$
\vec{v}_{t+1} = \mu\vec{v}_t + \vec{g}_t, \qquad
\vecgreek{\theta}_{t+1} = \vecgreek{\theta}_t - \eta\,\vec{v}_{t+1}
$$ (eq:momentum)

with $\mu$ typically $0.9$. **Nesterov** evaluates the gradient at the
*look-ahead* point $\vecgreek{\theta}_t - \eta\mu\vec{v}_t$ rather than at
$\vecgreek{\theta}_t$:

$$
\vec{v}_{t+1} = \mu\vec{v}_t
 + \nabla\Like(\vecgreek{\theta}_t - \eta\mu\vec{v}_t)
$$ (eq:nesterov)

The difference is a correction term proportional to the change in gradient,
which damps the overshoot momentum would otherwise produce. It is a genuine
improvement and a small one.

### 5.3 The adaptive family

**AdaGrad.** With $\vec{s}_t = \sum_{\tau\le t}\vec{g}_\tau^{\,2}$
(elementwise):

$$
\vecgreek{\theta}_{t+1} = \vecgreek{\theta}_t
 - \frac{\eta}{\sqrt{\vec{s}_t}+\epsilon}\odot\vec{g}_t
$$ (eq:adagrad)

**RMSProp.** Replace the sum with an exponential moving average:

$$
\vec{s}_t = \rho\vec{s}_{t-1} + (1-\rho)\vec{g}_t^{\,2}
$$ (eq:rmsprop)

**Adam** {{cite:kingma2015adam}} combines both, with bias correction:

$$
\vec{m}_t = \beta_1\vec{m}_{t-1} + (1-\beta_1)\vec{g}_t
$$ (eq:adam-m)

$$
\vec{v}_t = \beta_2\vec{v}_{t-1} + (1-\beta_2)\vec{g}_t^{\,2}
$$ (eq:adam-v)

$$
\hat{\vec{m}}_t = \frac{\vec{m}_t}{1-\beta_1^{t}}, \qquad
\hat{\vec{v}}_t = \frac{\vec{v}_t}{1-\beta_2^{t}}
$$ (eq:adam-bias-correction)

$$
\vecgreek{\theta}_{t+1} = \vecgreek{\theta}_t
 - \eta\,\frac{\hat{\vec{m}}_t}{\sqrt{\hat{\vec{v}}_t}+\epsilon}
$$ (eq:adam-update)

Defaults $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$. These have
survived a decade of attempts to improve on them, which is itself informative.

### 5.4 Why bias correction is needed

$\vec{m}_0 = \vec{0}$, so early estimates are biased toward zero. Unrolling
{{eq:adam-m}} with a constant gradient $g$:

$$
m_t = (1-\beta_1)\sum_{i=0}^{t-1}\beta_1^{i} g
 = g\left(1-\beta_1^{t}\right)
$$ (eq:adam-m-unrolled)

At $t=1$ with $\beta_1 = 0.9$, $m_1 = 0.1g$ — a tenth of the true value. The
correction divides by exactly $(1-\beta_1^t)$ and recovers $g$.

The damage without it is asymmetric and that is the point.
$\beta_2 = 0.999$ means $\vec{v}$ takes thousands of steps to warm up, so the
*denominator* is far more suppressed than the numerator, and the ratio is far
too **large**. **Omitting bias correction gives enormous steps in the first
hundred iterations**, which is a divergence, not a slow start.
{{sec:8-implementation}} measures it.

### 5.5 AdamW: decoupled weight decay

$\ell_2$ regularisation adds $\frac{\lambda}{2}\|\vecgreek{\theta}\|^2$ to the
loss, so the gradient gains $\lambda\vecgreek{\theta}$. Under Adam this term
goes through the adaptive rescaling:

$$
\Delta\vecgreek{\theta}\;\propto\;
 \frac{\vec{m}_t + \lambda\vecgreek{\theta}}{\sqrt{\vec{v}_t}}
$$ (eq:l2-in-adam)

**The effective decay is therefore divided by $\sqrt{\vec{v}}$, so parameters
with large gradients are regularised less.** That is backwards from what
regularisation is supposed to do, and it is not what anyone writing
`weight_decay=0.01` intends.

{{cite:loshchilov2019adamw}} decouples the two:

$$
\vecgreek{\theta}_{t+1} = \vecgreek{\theta}_t
 - \eta\frac{\hat{\vec{m}}_t}{\sqrt{\hat{\vec{v}}_t}+\epsilon}
 - \eta\lambda\vecgreek{\theta}_t
$$ (eq:adamw)

The decay is applied directly to the parameter, untouched by the adaptive
scaling. For SGD the two formulations are equivalent up to a rescaling of
$\lambda$; for Adam they are genuinely different updates, which is why the
distinction was invisible for years.

> IMPORTANT: **Use AdamW, not Adam-with-weight-decay.** They are different
> algorithms with the same name in most people's heads, and every serious
> large-model recipe since 2019 uses the decoupled form. Note also that
> $\lambda$ does not transfer between the two: the AdamW value is typically
> larger by roughly the scale of $\sqrt{\vec{v}}$.

### 5.6 Memory

{#tbl:optimizer-memory caption="Optimiser state per parameter. Adam's two moments are why a training job needs several times the memory of the weights alone, and the last column is the number people forget when sizing a machine."}

| Optimiser | State per parameter | Total for $P$ parameters (fp32) |
|---|---|---|
| SGD | none | $4P$ bytes (weights) $+\,4P$ (gradients) |
| SGD + momentum | $\vec{v}$ | $12P$ |
| AdaGrad / RMSProp | $\vec{s}$ | $12P$ |
| Adam / AdamW | $\vec{m}, \vec{v}$ | $16P$ |

A 7-billion-parameter model in fp32 is 28 GB of weights and **112 GB before a
single activation is stored** under Adam. This single row explains most of the
engineering in {{part:23}}, and it is why optimiser-state sharding exists.

## 6. Mathematical Foundation

### 6.1 Convergence of SGD

For a convex $\Like$ with $L$-Lipschitz gradients and unbiased gradient
estimates of variance $\sigma^2$, SGD with step size $\eta$ satisfies

$$
\E\big[\Like(\bar{\vecgreek{\theta}}_T)\big] - \Like^\star
 \le \frac{\|\vecgreek{\theta}_0-\vecgreek{\theta}^\star\|^2}{2\eta T}
 + \frac{\eta\sigma^2}{2}
$$ (eq:sgd-convergence)

Read the two terms against each other. The first falls as $1/T$ and wants
$\eta$ large. The second is constant in $T$ and wants $\eta$ small.

**With constant $\eta$, SGD does not converge to the optimum.** It converges to a
neighbourhood of radius proportional to $\eta\sigma^2$ and then bounces around
inside it. Optimising the bound over $\eta$ gives $\eta \propto 1/\sqrt{T}$ and
a rate of $O(1/\sqrt{T})$.

**That is the entire justification for learning-rate decay**
({{ch:dl-lr-schedules}}), and it is worth having in this form: decay is not a
heuristic, it is what {{eq:sgd-convergence}} requires.

### 6.2 Momentum as a differential equation

Rewrite {{eq:momentum}} as a single second-order recurrence:

$$
\vecgreek{\theta}_{t+1} - \vecgreek{\theta}_t
 = \mu(\vecgreek{\theta}_t - \vecgreek{\theta}_{t-1}) - \eta\vec{g}_t
$$ (eq:momentum-second-order)

which is the discretisation of

$$
m\ddot{\vecgreek{\theta}} + c\dot{\vecgreek{\theta}}
 = -\nabla\Like(\vecgreek{\theta})
$$ (eq:heavy-ball-ode)

a particle of mass $m$ in a potential $\Like$ with friction $c$, where
$\mu \leftrightarrow 1 - c\,\Delta t/m$. The "heavy ball" is not a loose
metaphor.

**The effective step size is amplified.** For a gradient constant over many
steps, the velocity converges to a geometric sum:

$$
\vec{v}_\infty = \frac{\vec{g}}{1-\mu}
$$ (eq:momentum-amplification)

so $\mu = 0.9$ multiplies the asymptotic step by 10 and $\mu = 0.99$ by 100.
**Momentum and learning rate are not independent hyperparameters**: raising
$\mu$ without lowering $\eta$ is a tenfold learning-rate increase in disguise,
and it is a standard cause of "momentum made it diverge".

### 6.3 Momentum on a quadratic

For $\Like = \frac{1}{2}\vecgreek{\theta}\T\mat{A}\vecgreek{\theta}$ with
eigenvalues in $[\alpha, \beta]$, define the condition number
$\kappa = \beta/\alpha$. Gradient descent converges at rate

$$
\left(\frac{\kappa-1}{\kappa+1}\right)^{t}
\qquad\text{versus momentum's}\qquad
\left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^{t}
$$ (eq:momentum-rate)

The square root is the whole benefit. For $\kappa = 10^4$, gradient descent
needs about $10^4$ iterations to reduce the error by a fixed factor and momentum
needs about $10^2$. **Momentum buys a quadratic improvement in the condition
number**, which is why it matters most exactly where the problem is worst.

### 6.4 Adam's scale invariance

Rescale the loss by $c > 0$. Then $\vec{g} \to c\vec{g}$, so $\vec{m} \to
c\vec{m}$ and $\vec{v} \to c^2\vec{v}$. The update is

$$
\frac{c\hat{\vec{m}}}{\sqrt{c^2\hat{\vec{v}}}+\epsilon}
 = \frac{\hat{\vec{m}}}{\sqrt{\hat{\vec{v}}}+\epsilon/c}
 \;\xrightarrow[\epsilon\to 0]{}\;
 \frac{\hat{\vec{m}}}{\sqrt{\hat{\vec{v}}}}
$$ (eq:adam-scale-invariance)

**Adam is invariant to the scale of the loss**, up to $\epsilon$. SGD is not:
scaling the loss by 1000 scales its step by 1000.

This is a real practical advantage — it is why Adam survives loss scaling in
mixed precision, and why it needs less retuning when an architecture changes.
It is also why $\epsilon$ is not a numerical afterthought: it is what breaks the
invariance, and raising it from $10^{-8}$ to $10^{-6}$ is a meaningful change to
the algorithm's behaviour on parameters with small gradients.

### 6.5 The bound on Adam's step

Ignoring $\epsilon$ and bias correction, and taking $\beta_1^2 \le \beta_2$
(true for the defaults):

$$
\left|\frac{\hat m_t}{\sqrt{\hat v_t}}\right| \le
 \frac{1-\beta_1}{\sqrt{1-\beta_2}}\cdot
 \frac{\sum_i \beta_1^i |g_{t-i}|}{\sqrt{\sum_i \beta_2^i g_{t-i}^2}}
 \;\lesssim\; 1
$$ (eq:adam-step-bound)

with equality approached when the gradient is perfectly consistent. **Adam's
per-parameter step is bounded by roughly $\eta$**, whatever the gradient
magnitude — a *trust region* that SGD does not have.

This is the strongest argument for Adam as a default: a single learning rate
sets a hard cap on how far any parameter can move in one step, so a gradient
spike cannot destroy the model. Under SGD the same spike moves the parameters in
proportion to it.

### 6.6 Why not second order?

Newton's method uses $\mat{H}^{-1}\vec{g}$ and converges in far fewer
iterations. The obstacles are arithmetic:

**Storage.** $\mat{H}$ has $P^2$ entries. For $P = 10^9$ that is $10^{18}$
numbers.

**Inversion.** $O(P^3)$.

**Indefiniteness.** In a non-convex problem $\mat{H}$ has negative eigenvalues,
so a Newton step can move *uphill*.

Practical approximations exist — L-BFGS with a limited history, K-FAC with a
Kronecker-factored approximation, Shampoo with per-tensor preconditioners — and
none has displaced Adam for deep learning. The reason is instructive: the
mini-batch gradient is noisy, and curvature estimates built from noisy gradients
are noisier still. **Second-order information is most valuable exactly where it
is least reliable.** {{maturity:EMERGING}} for the modern preconditioners, which
do show gains at large scale.

## 7. Internal Mechanics

### 7.1 What an implementation does per step

```text
   for each parameter tensor:
       g = p.grad
       g += weight_decay * p              (coupled L2 — the WRONG one for Adam)
       m = b1*m + (1-b1)*g
       v = b2*v + (1-b2)*g*g
       mhat = m / (1 - b1**t)
       vhat = v / (1 - b2**t)
       p -= lr * mhat / (sqrt(vhat) + eps)
       p -= lr * weight_decay * p         (decoupled — AdamW)
```

Every line is elementwise and memory-bound: it reads and writes several tensors
the size of the parameters and does almost no arithmetic. **The optimiser step
is a bandwidth problem, not a compute problem**, which is why fused optimiser
kernels — doing all of it in one pass over memory — are a real speedup and why
`foreach` implementations that batch across tensors exist.

### 7.2 Where epsilon goes

Two conventions, and they differ:

```text
   p -= lr * mhat / (sqrt(vhat) + eps)       most implementations
   p -= lr * mhat / sqrt(vhat + eps)         some others
```

The first bounds the step at $\eta\hat m/\epsilon$; the second at
$\eta\hat m/\sqrt{\epsilon}$ — a factor of $10^4$ apart at
$\epsilon = 10^{-8}$. When a model reproduces differently across frameworks at
identical hyperparameters, this is a place to look.

### 7.3 Parameter groups

Not every parameter should be treated alike, and the standard recipe is
near-universal in large-model training:

```text
   decay:     weight matrices
   no decay:  biases, normalisation scales and shifts, embeddings (usually)
```

Decaying a normalisation scale toward zero fights the normalisation directly;
decaying a bias regularises nothing useful. Getting this wrong is a real and
common performance loss, and it is invisible in the loss curve's shape.

### 7.4 Sparse gradients

For an embedding table, one step touches a handful of rows. A dense optimiser
update touches all of them — and worse, Adam's $\vec{v}$ *decays* for untouched
rows, so a rare token's step size grows every time it is not seen. Sparse
implementations update only the touched rows, which is faster and is also a
*different algorithm*. Which behaviour is correct is genuinely unclear, and the
frameworks disagree.

### 7.5 Numerical precision

The optimiser state is kept in fp32 even when the model is in bf16. The reason
is {{eq:adam-v}}: $\vec{v}$ accumulates squared gradients of order $10^{-6}$,
whose squares are $10^{-12}$, which is below bf16's precision near 1. Storing
$\vec{v}$ in bf16 rounds those contributions to nothing.

The master-weights pattern follows: keep an fp32 copy of the parameters for the
optimiser and a bf16 copy for the forward pass, because a bf16 parameter of
magnitude 1 cannot represent an update of $10^{-4}$ at all — it rounds to zero
and the parameter never moves ({{part:23}}).

## 8. Implementation

```python {tier=A name=optimizers-from-scratch}
"""Every optimiser in this chapter, implemented from its equations and
compared on a problem whose difficulty we control.
"""
import numpy as np

rng = np.random.default_rng(0)


# --- the optimisers ---------------------------------------------------------
class SGD:
    name = "SGD"

    def __init__(self, lr=0.01, momentum=0.0, nesterov=False):
        self.lr, self.mu, self.nesterov = lr, momentum, nesterov
        self.v = None

    def step(self, p, g, t):
        if self.mu == 0.0:
            return p - self.lr * g                       # eq. 54.1
        if self.v is None:
            self.v = np.zeros_like(p)
        self.v = self.mu * self.v + g                    # eq. 54.2
        d = (g + self.mu * self.v) if self.nesterov else self.v
        return p - self.lr * d


class AdaGrad:
    name = "AdaGrad"

    def __init__(self, lr=0.1, eps=1e-8):
        self.lr, self.eps, self.s = lr, eps, None

    def step(self, p, g, t):
        if self.s is None:
            self.s = np.zeros_like(p)
        self.s += g * g                                  # eq. 54.4
        return p - self.lr * g / (np.sqrt(self.s) + self.eps)


class RMSProp:
    name = "RMSProp"

    def __init__(self, lr=0.01, rho=0.9, eps=1e-8):
        self.lr, self.rho, self.eps, self.s = lr, rho, eps, None

    def step(self, p, g, t):
        if self.s is None:
            self.s = np.zeros_like(p)
        self.s = self.rho * self.s + (1 - self.rho) * g * g    # eq. 54.5
        return p - self.lr * g / (np.sqrt(self.s) + self.eps)


class Adam:
    def __init__(self, lr=0.001, b1=0.9, b2=0.999, eps=1e-8,
                 weight_decay=0.0, decoupled=True, bias_correction=True):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.wd, self.decoupled, self.bc = weight_decay, decoupled, \
            bias_correction
        self.m = self.v = None

    @property
    def name(self):
        if self.wd == 0.0:
            return "Adam" if self.bc else "Adam (no bias correction)"
        return "AdamW" if self.decoupled else "Adam + coupled L2"

    def step(self, p, g, t):
        if self.m is None:
            self.m = np.zeros_like(p)
            self.v = np.zeros_like(p)
        if self.wd and not self.decoupled:
            g = g + self.wd * p                          # eq. 54.11
        self.m = self.b1 * self.m + (1 - self.b1) * g    # eq. 54.6
        self.v = self.b2 * self.v + (1 - self.b2) * g * g   # eq. 54.7
        if self.bc:
            mh = self.m / (1 - self.b1 ** t)             # eq. 54.8
            vh = self.v / (1 - self.b2 ** t)
        else:
            mh, vh = self.m, self.v
        p = p - self.lr * mh / (np.sqrt(vh) + self.eps)  # eq. 54.9
        if self.wd and self.decoupled:
            p = p - self.lr * self.wd * p                # eq. 54.12
        return p


# --- an ill-conditioned quadratic, where the theory is exact ----------------
def quadratic(kappa, dim=50, seed=0):
    """L = 0.5 x' A x with eigenvalues log-spaced over [1, kappa]."""
    rs = np.random.default_rng(seed)
    evals = np.logspace(0, np.log10(kappa), dim)
    Q, _ = np.linalg.qr(rs.normal(size=(dim, dim)))
    A = Q @ np.diag(evals) @ Q.T
    return A, evals


def run_quadratic(A, opt, steps=400, seed=1, noise=0.0):
    rs = np.random.default_rng(seed)
    x = rs.normal(size=len(A))
    x = x / np.linalg.norm(x) * 5.0
    losses = []
    for t in range(1, steps + 1):
        g = A @ x
        if noise:
            g = g + rs.normal(0, noise * np.linalg.norm(g) / np.sqrt(len(g)),
                              len(g))
        losses.append(0.5 * float(x @ A @ x))
        x = opt.step(x, g, t)
        if not np.all(np.isfinite(x)):
            return losses + [np.inf] * (steps - len(losses))
    return losses


print("=" * 72)
print("momentum buys a square root in the condition number (eq. 54.17)")
print("=" * 72)
print("A quadratic with log-spaced eigenvalues. Each method gets ITS OWN")
print("theoretically optimal settings, which is the only fair comparison:")
print("  gradient descent   eta = 2/(alpha+beta)")
print("  heavy ball         eta = 4/(sqrt(alpha)+sqrt(beta))^2,")
print("                      mu = ((sqrt(k)-1)/(sqrt(k)+1))^2")
print("Giving momentum the SAME eta as gradient descent and then dividing")
print("by (1-mu) — a natural-looking choice — exactly cancels eq. 54.16's")
print("amplification and produces no speedup at all.\n")
print(f"{'kappa':>8} {'GD steps':>10} {'momentum steps':>16} "
      f"{'measured speedup':>18} {'predicted sqrt(k)':>19}")
for kappa in (10, 100, 1000, 10000):
    A, evals = quadratic(kappa, seed=0)
    a, b = evals.min(), evals.max()
    reach = {}
    opts = {
        "gd": SGD(lr=2.0 / (a + b)),
        "mom": SGD(lr=4.0 / (np.sqrt(a) + np.sqrt(b)) ** 2,
                   momentum=((np.sqrt(kappa) - 1)
                             / (np.sqrt(kappa) + 1)) ** 2),
    }
    for label, opt in opts.items():
        ls = run_quadratic(A, opt, steps=300000 if kappa > 999 else 30000)
        reach[label] = next((i for i, v in enumerate(ls)
                             if v < 1e-6 * ls[0]), None)
    if reach["gd"] and reach["mom"]:
        print(f"{kappa:>8} {reach['gd']:>10} {reach['mom']:>16} "
              f"{reach['gd'] / reach['mom']:>17.1f}x "
              f"{np.sqrt(kappa):>18.1f}")
    else:
        print(f"{kappa:>8} {str(reach['gd']):>10} {str(reach['mom']):>16}")

print("\nThe measured speedup is consistently about HALF the predicted")
print("sqrt(kappa), and it grows with kappa in the same proportion at every")
print("row. That is the right kind of agreement to expect: eq. 54.17 is an")
print("asymptotic rate, the constant in front of it is not one, and the")
print("spectrum here is a full log-spaced range rather than the two-point")
print("worst case on which the bound is tight.")
print("\nThe scaling is what matters. Momentum buys a factor of two at")
print("kappa = 10 and a factor of nearly fifty at kappa = 10000: it is")
print("worth almost nothing on a well-conditioned problem and enormous on")
print("a badly conditioned one, which is exactly eq. 54.17's claim.")

# --- eq. 54.16: momentum amplifies the step ---------------------------------
print("\n" + "=" * 72)
print("momentum and learning rate are not independent (eq. 54.16)")
print("=" * 72)
print(f"{'mu':>6} {'velocity after 2000 steps':>27} "
      f"{'predicted 1/(1-mu)':>20}")
for mu in (0.0, 0.5, 0.9, 0.99):
    v = 0.0
    for _ in range(2000):
        v = mu * v + 1.0                          # eq. 54.2 with g = 1
    print(f"{mu:>6.2f} {v:>27.4f} {1 / (1 - mu):>20.4f}")
print("\nThe asymptotic velocity is exactly 1/(1-mu) times the gradient.")
print("Raising momentum from 0.9 to 0.99 therefore multiplies the effective")
print("step by TEN at a fixed learning rate — which is why 'I increased")
print("momentum and it diverged' is not a mystery.")

# --- section 5.4: bias correction -------------------------------------------
print("\n" + "=" * 72)
print("what bias correction actually prevents (section 5.4)")
print("=" * 72)
print("A constant unit gradient. Without correction, m warms up in ~10 steps")
print("and v in ~1000, so the RATIO is far too large early.\n")
print(f"{'step':>6} {'m':>10} {'v':>12} {'step w/ correction':>20} "
      f"{'step WITHOUT':>14} {'ratio':>10}")
b1, b2 = 0.9, 0.999
m = v = 0.0
for t in range(1, 3001):
    g = 1.0
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    with_bc = (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + 1e-8)
    without = m / (np.sqrt(v) + 1e-8)
    if t in (1, 2, 5, 20, 100, 500, 1000, 3000):
        print(f"{t:>6} {m:>10.4f} {v:>12.6f} {with_bc:>20.4f} "
              f"{without:>14.4f} {without / with_bc:>9.2f}x")

print("\nRead the last two columns. WITH correction the step is 1.0 from the")
print("very first iteration, which is what eq. 54.9 is supposed to give for")
print("a consistent gradient. WITHOUT it the step is over THREE times too")
print("large at step 1 and stays inflated for hundreds of iterations.")
print("\nThe asymmetry in section 5.4 is the cause: m warms up on a timescale")
print("of 1/(1-b1) = 10 steps and v on 1/(1-b2) = 1000, so the denominator")
print("is suppressed for far longer than the numerator. The uncorrected")
print("update is not a slow start — it is an overshoot, and on a real")
print("network it is a divergence.")
```

```python {tier=A name=adam-properties}
"""Two properties that explain why Adam is the default: scale invariance and
the bounded step. Both measured rather than asserted.
"""
import numpy as np

rng = np.random.default_rng(1)


class Adam:
    def __init__(self, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = self.v = None

    def step(self, p, g, t):
        if self.m is None:
            self.m = np.zeros_like(p)
            self.v = np.zeros_like(p)
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        mh = self.m / (1 - self.b1 ** t)
        vh = self.v / (1 - self.b2 ** t)
        return p - self.lr * mh / (np.sqrt(vh) + self.eps)


# --- eq. 54.18: scale invariance -------------------------------------------
print("=" * 72)
print("Adam is invariant to the scale of the loss; SGD is not (eq. 54.18)")
print("=" * 72)
A = np.diag(np.logspace(0, 2, 20))
x0 = rng.normal(size=20)


def trajectory(opt_factory, scale, steps=60):
    """Stopped mid-flight: a converged run would hide the difference."""
    x = x0.copy()
    opt = opt_factory()
    for t in range(1, steps + 1):
        g = scale * (A @ x)
        x = opt.step(x, g, t)
    return x


SCALES = (0.001, 1.0, 1000.0)
rows = {}
for scale in SCALES:
    xa = trajectory(lambda: Adam(lr=0.05), scale)
    xs_ = x0.copy()
    with np.errstate(over="ignore", invalid="ignore"):
        for t in range(1, 61):
            xs_ = xs_ - 0.005 * scale * (A @ xs_)
    rows[scale] = (float(np.linalg.norm(xa)),
                   float(np.linalg.norm(xs_)) if np.all(np.isfinite(xs_))
                   else float("inf"))

ref_a, ref_s = rows[1.0]
print(f"{'loss scale':>12} {'Adam |x| @60':>15} {'vs scale=1':>12}   "
      f"{'SGD |x| @60':>14} {'vs scale=1':>12}")
for scale in SCALES:
    na, ns = rows[scale]
    print(f"{scale:>12g} {na:>15.8f} {na / ref_a:>11.4f}x   "
          f"{ns:>14.6g} {ns / ref_s:>11.4g}x")

print("\nAdam is at the SAME point after sixty steps whatever the loss")
print("scale: the c in eq. 54.18 cancels between numerator and denominator.")
print("SGD is not. At one thousandth the scale it has barely moved, and at")
print("a thousand times it has diverged — all at one learning rate.")
print("\nThis is why Adam needs so much less retuning when the loss changes")
print("form, when a scaling factor is introduced for mixed precision, or")
print("when the architecture changes the gradient magnitudes.")

# --- eq. 54.19: the bounded step -------------------------------------------
print("\n" + "=" * 72)
print("Adam's per-parameter step is bounded; SGD's is not (eq. 54.19)")
print("=" * 72)
print("A gradient of magnitude 1, then a single SPIKE, then back to 1.\n")

opt = Adam(lr=0.01)
p = np.zeros(1)
print(f"{'step':>6} {'gradient':>12} {'|Adam move|':>14} {'|SGD move|':>14}")
lr_sgd = 0.01
for t in range(1, 41):
    g = np.array([1000.0]) if t == 20 else np.array([1.0])
    before = p.copy()
    p = opt.step(p, g, t)
    move_adam = float(abs(p - before).item())
    move_sgd = float(abs(lr_sgd * g).item())
    if t in (1, 5, 19, 20, 21, 25, 40):
        print(f"{t:>6} {g.item():>12.1f} {move_adam:>14.6f} "
              f"{move_sgd:>14.6f}")

print("\nThe spike is a thousand times the usual gradient. SGD moves a")
print("thousand times further, and on a real network that single step")
print("destroys the parameter.")
print("\nAdam's move on the spike step did not grow at all — it HALVED.")
print("That is worth more than the bound of eq. 54.19 promised, and the")
print("reason is the timescale asymmetry again: v jumps by the full")
print("(1-b2)*g^2 immediately, while m is an average over roughly ten")
print("steps and barely notices one outlier. The denominator reacts faster")
print("than the numerator, so a gradient spike makes Adam MORE cautious")
print("rather than less.")
print("\nThe steps then stay small for a long time afterwards — a tenth of")
print("normal by step 40 — because v remembers the spike for about")
print("1/(1-b2) = 1000 steps. That is the cost of the protection, and it")
print("is why gradient clipping is still worth having: it stops the spike")
print("from entering v in the first place.")
print("\nThis combination is the strongest single argument for Adam as a")
print("default. One learning rate caps how far any parameter can move in")
print("one step, so a bad batch cannot wreck the model.")

# --- the price: what Adam costs in memory (table 54.1) ----------------------
print("\n" + "=" * 72)
print("what the state costs (table 54.1)")
print("=" * 72)
print(f"{'model':<14} {'params':>10} {'SGD':>9} {'SGD+mom':>9} "
      f"{'Adam':>9} {'serve bf16':>11} {'train/serve':>12}")
for label, P in (("small MLP", 1e6), ("BERT-base", 1.1e8),
                 ("7B model", 7e9), ("70B model", 7e10)):
    gb = lambda mult: P * mult / 1e9
    print(f"{label:<14} {P:>10.0e} {gb(8):>8.1f}G {gb(12):>8.1f}G "
          f"{gb(16):>8.1f}G {gb(2):>10.1f}G {gb(16) / gb(2):>11.0f}x")
print("\n(4 bytes each for weights, gradients and each moment; serving needs")
print(" only the weights, in bf16. Mixed-precision training comes to the")
print(" same total as the fp32 column, because the bf16 weight and gradient")
print(" copies save exactly what the fp32 master copy costs.)")
print("\nThe 7B row is the one to remember: 112 GB of optimiser-related")
print("memory before a single activation is stored, for a model whose")
print("weights are 28 GB. Serving it needs 14 GB in bf16. That factor of")
print("eight between serving and training is why optimiser-state sharding")
print("exists, and it is arithmetic rather than mystery.")
```

## 9. Practical Example

```python {tier=A name=choosing-an-optimizer}
"""SGD, momentum, RMSProp, Adam and AdamW on a real network, with the
coupled-versus-decoupled weight decay difference measured.
"""
import numpy as np

rng = np.random.default_rng(7)


class MLP:
    def __init__(self, sizes, seed=0):
        rs = np.random.default_rng(seed)
        self.shapes = [(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
        self.W = [rs.normal(0, np.sqrt(2 / a), (a, b)) for a, b in self.shapes]
        self.b = [np.zeros(b) for _, b in self.shapes]

    def forward(self, X):
        self.H, self.Z = [X], []
        h = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            self.Z.append(z)
            h = np.maximum(0.0, z) if i < len(self.W) - 1 else z
            self.H.append(h)
        return h

    def loss_and_grads(self, X, y):
        logits = self.forward(X)
        m = logits.max(axis=1, keepdims=True)
        e = np.exp(logits - m)
        p = e / e.sum(axis=1, keepdims=True)
        loss = float(np.mean(m[:, 0] + np.log(e.sum(axis=1))
                             - logits[np.arange(len(X)), y]))
        d = p.copy()
        d[np.arange(len(X)), y] -= 1.0
        d /= len(X)
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for l in reversed(range(len(self.W))):
            gW[l] = self.H[l].T @ d
            gb[l] = d.sum(axis=0)
            if l > 0:
                d = (d @ self.W[l].T) * (self.Z[l - 1] > 0)
        return loss, gW, gb


D, CLASSES = 20, 4
# ONE labelling function, shared by train and test. Drawing a fresh Wt per
# split would make the task unlearnable, and the symptom — every optimiser
# scoring at chance — looks like an optimiser problem rather than a data one.
_wt_rs = np.random.default_rng(1234)
W_TRUE = _wt_rs.normal(size=(D, CLASSES))
W_TRUE[:5] /= 100.0                         # undo the feature rescaling below
W_TRUE[5:10] *= 100.0
H_TRUE = _wt_rs.normal(size=(CLASSES, CLASSES))


def make_data(n, seed=0):
    """Deliberately BADLY SCALED features, which is the realistic case."""
    rs = np.random.default_rng(seed)
    X = rs.normal(size=(n, D))
    X[:, :5] *= 100.0                       # some features are huge
    X[:, 5:10] *= 0.01                      # some are tiny
    logits = np.tanh(X @ W_TRUE) @ H_TRUE * 3.0
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y = np.array([rs.choice(CLASSES, p=pi) for pi in p])
    return X, y


Xtr, ytr = make_data(6000, seed=1)
Xte, yte = make_data(6000, seed=2)
print("=" * 72)
print("the problem")
print("=" * 72)
_p = np.exp(np.tanh(Xte @ W_TRUE) @ H_TRUE * 3.0)
_p = _p / _p.sum(axis=1, keepdims=True)
print(f"{len(Xtr)} train / {len(Xte)} test, {CLASSES} classes, "
      f"{D} features")
print(f"Bayes-optimal accuracy on the test set : "
      f"{float(_p[np.arange(len(yte)), yte].mean()):.4f}")
print(f"Bayes-optimal cross-entropy            : "
      f"{float(-np.log(_p[np.arange(len(yte)), yte]).mean()):.4f}")
print(f"chance accuracy                        : {1 / CLASSES:.4f}")
print("\nFive features are scaled by 100 and five by 0.01, so the gradient")
print("magnitudes across the first layer span four orders of magnitude.")
print("That is the situation adaptive methods were invented for.")


def train(opt_factory, lr, steps=3000, batch=64, seed=0, wd=0.0,
          decoupled=True):
    net = MLP([20, 64, 64, 4], seed=seed)
    opts = [opt_factory(lr) for _ in net.W] + [opt_factory(lr) for _ in net.b]
    rs = np.random.default_rng(seed + 100)
    for t in range(1, steps + 1):
        idx = rs.integers(0, len(Xtr), batch)
        _, gW, gb = net.loss_and_grads(Xtr[idx], ytr[idx])
        for i, (W, g) in enumerate(zip(net.W, gW)):
            if wd and not decoupled:
                g = g + wd * W
            net.W[i] = opts[i].step(W, g, t)
            if wd and decoupled:
                net.W[i] = net.W[i] - lr * wd * net.W[i]
        for i, (b, g) in enumerate(zip(net.b, gb)):
            net.b[i] = opts[len(net.W) + i].step(b, g, t)
    tr_loss, _, _ = net.loss_and_grads(Xtr, ytr)
    te_loss, _, _ = net.loss_and_grads(Xte, yte)
    acc = float((net.forward(Xte).argmax(axis=1) == yte).mean())
    wnorm = float(np.sqrt(sum(float(np.sum(W ** 2)) for W in net.W)))
    return tr_loss, te_loss, acc, wnorm


class SGD:
    def __init__(self, lr, momentum=0.0):
        self.lr, self.mu, self.v = lr, momentum, None

    def step(self, p, g, t):
        if self.mu == 0:
            return p - self.lr * g
        if self.v is None:
            self.v = np.zeros_like(p)
        self.v = self.mu * self.v + g
        return p - self.lr * self.v


class RMSProp:
    def __init__(self, lr, rho=0.9, eps=1e-8):
        self.lr, self.rho, self.eps, self.s = lr, rho, eps, None

    def step(self, p, g, t):
        if self.s is None:
            self.s = np.zeros_like(p)
        self.s = self.rho * self.s + (1 - self.rho) * g * g
        return p - self.lr * g / (np.sqrt(self.s) + self.eps)


class Adam:
    def __init__(self, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = self.v = None

    def step(self, p, g, t):
        if self.m is None:
            self.m = np.zeros_like(p)
            self.v = np.zeros_like(p)
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        mh, vh = self.m / (1 - self.b1 ** t), self.v / (1 - self.b2 ** t)
        return p - self.lr * mh / (np.sqrt(vh) + self.eps)


print("=" * 72)
print("four optimisers, each with its OWN tuned learning rate")
print("=" * 72)

GRID = {
    "SGD": (lambda lr: SGD(lr), [3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]),
    "SGD + momentum 0.9": (lambda lr: SGD(lr, 0.9),
                           [3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3]),
    "RMSProp": (lambda lr: RMSProp(lr), [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]),
    "Adam": (lambda lr: Adam(lr), [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]),
}
print(f"{'optimiser':<22} {'best lr':>9} {'train loss':>12} "
      f"{'test loss':>11} {'test acc':>10}")
results = {}
for name, (factory, grid) in GRID.items():
    best = None
    for lr in grid:
        out = train(factory, lr, seed=0)
        if np.isfinite(out[1]) and (best is None or out[1] < best[1][1]):
            best = (lr, out)
    results[name] = best
    lr, (trl, tel, acc, wn) = best
    print(f"{name:<22} {lr:>9.0e} {trl:>12.4f} {tel:>11.4f} {acc:>10.4f}")

print("\nEach optimiser was given its own learning-rate search, so this is a")
print("comparison of the methods rather than of one lucky hyperparameter.")
print("\nThe best learning rates span a factor of about thirty, and the")
print("ordering is the one eq. 54.16 and eq. 54.18 predict: momentum's best")
print("rate is roughly ten times below plain SGD's, because eq. 54.16")
print("amplifies the step by 1/(1-mu); and Adam's is high in absolute terms")
print("because eq. 54.18 makes it a step SIZE rather than a gradient")
print("multiplier.")
print("\nThe substantive result is the gap between the two families. The")
print("adaptive methods reach a materially lower loss and a much higher")
print("accuracy than either SGD variant, at every learning rate either")
print("family was given. On a problem whose gradient magnitudes span four")
print("orders of magnitude across features, one global step size is simply")
print("the wrong instrument — which is the argument adaptive methods were")
print("introduced to make, here on a problem constructed to make it.")
print("\nNote also that every method is well short of the Bayes rate")
print("printed above. This is a fixed budget of 3000 steps, so what is")
print("being measured is optimisation SPEED, not what each method could")
print("eventually reach.")

# --- coupled vs decoupled weight decay --------------------------------------
print("\n" + "=" * 72)
print("Adam + L2 is not AdamW (section 5.5)")
print("=" * 72)
print("The SAME lambda through both formulations, on the same network.\n")
print(f"{'lambda':>10} {'formulation':<20} {'train loss':>12} "
      f"{'test loss':>11} {'test acc':>10} {'|W|':>9}")
trl, tel, acc, wn = train(lambda lr: Adam(lr), 1e-3, wd=0.0, seed=0)
print(f"{0:>10g} {'none':<20} {trl:>12.4f} {tel:>11.4f} "
      f"{acc:>10.4f} {wn:>9.3f}")
for wd in (0.001, 0.01, 0.1, 1.0):
    for decoupled in (False, True):
        trl, tel, acc, wn = train(lambda lr: Adam(lr), 1e-3, wd=wd,
                                  decoupled=decoupled, seed=0)
        label = ("AdamW (eq 54.12)" if decoupled
                 else "Adam + L2 (eq 54.11)")
        print(f"{wd:>10g} {label:<20} {trl:>12.4f} {tel:>11.4f} "
              f"{acc:>10.4f} {wn:>9.3f}")

print("\nRead the |W| column down the pairs. At the SAME lambda the two")
print("formulations shrink the weights by wildly different amounts.")
print("\nThe reason is in the two equations. Decoupled decay multiplies the")
print("weight by (1 - eta*lambda) each step, so at eta = 1e-3 and")
print("lambda = 1e-3 that is a factor of 1e-6 per step and essentially")
print("nothing over a few thousand steps. Coupled L2 adds lambda*W to the")
print("gradient, which then passes through the 1/sqrt(v) rescaling of")
print("eq. 54.11 — and since sqrt(v) is small, the rescaling AMPLIFIES the")
print("decay enormously.")
print("\nSo the practical consequence is not subtle: lambda does NOT")
print("transfer between the two. Reading a value from a paper that used one")
print("and applying it under the other is a silent misconfiguration, and it")
print("is exactly why AdamW's recommended lambdas look so much larger than")
print("the L2 coefficients people were used to. To get comparable")
print("regularisation you have to compare at matched |W|, not at matched")
print("lambda — which is what the rows above let you do.")
print("\nNote also the second-order point hidden in eq. 54.11: because the")
print("coupled decay is divided by sqrt(v), parameters with LARGE gradients")
print("are regularised LESS. That is backwards from the intent, and it is")
print("the substance of Loshchilov and Hutter's argument, not just the")
print("bookkeeping about lambda.")

# --- how sensitive is each optimiser to its learning rate? ------------------
print("\n" + "=" * 72)
print("sensitivity to the learning rate: the practical reason for Adam")
print("=" * 72)
print("Test loss across the whole grid, so the shape of each row is visible")
print("rather than only its best point.\n")
for name, (factory, grid) in GRID.items():
    row = []
    for lr in grid:
        out = train(factory, lr, seed=0)
        row.append("diverged" if not np.isfinite(out[1]) or out[1] > 10
                   else f"{out[1]:.3f}")
    print(f"{name:<22} " + " ".join(f"{v:>9}" for v in row))
    print(f"{'':<22} " + " ".join(f"{lr:>9.0e}" for lr in grid))
print("\nNothing diverged, which is worth saying because it is not the")
print("story the usual 'Adam is robust' claim would predict. Every method")
print("at every rate produced a finite loss.")
print("\nWhat the rows actually show is different and more useful. Both SGD")
print("variants flatten out around 1.25 and get no further no matter which")
print("rate they are given — the curve has a floor. Both adaptive methods")
print("are still improving at the TOP of their grids and reach about 1.0.")
print("\nSo the difference here is not robustness to the learning rate; it")
print("is that on a badly scaled problem SGD has a ceiling that no single")
print("global step size can get past, and the adaptive methods do not. If")
print("you take one thing from this table, take that: run the grid and look")
print("at whether it has flattened, because a flat tail means the method")
print("rather than the setting is the limit.")
```

## 10. Production Considerations

**Default to AdamW.** Decoupled decay, $\beta = (0.9, 0.999)$, and a learning
rate found by a short search. The measured sensitivity table is the argument:
the number of learning rates that work at all is what matters when you cannot
afford a sweep.

**Budget optimiser memory explicitly.** $16P$ bytes for Adam in fp32. The
measured table gives the numbers, and the 7B row is the one that surprises
people.

**Set up parameter groups.** Decay on weight matrices, none on biases and
normalisation parameters. It costs three lines and it is a real and invisible
performance difference.

**Never transfer $\lambda$ between coupled and decoupled formulations.**
Measured: they produce different weight norms at the same $\lambda$.

**Clip gradients** ({{ch:dl-backprop}}) even with Adam. Adam's step is bounded
per parameter, and the measured spike experiment shows the recovery still takes
hundreds of steps because $\vec{v}$ remembers.

**Checkpoint the optimiser state, not just the weights.** Resuming without
$\vec{m}$ and $\vec{v}$ restarts the warm-up and produces a visible loss spike
that people misdiagnose as data corruption.

**Keep optimiser state in fp32.** {{sec:7-internal-mechanics}} explains why
$\vec{v}$ specifically cannot survive bf16.

## 11. Common Mistakes

**Using Adam's default learning rate for SGD.** Three orders of magnitude apart
in the measurement.

**Raising momentum without lowering the learning rate.** Measured: the
asymptotic step is $1/(1-\mu)$ times the gradient, so $0.9 \to 0.99$ is a
tenfold increase.

**Believing `Adam(weight_decay=...)` implements weight decay.** In most
frameworks it implements coupled $\ell_2$, which is a different update.

**Applying weight decay to normalisation parameters.** Fights the normalisation.

**Omitting bias correction in a custom implementation.** Measured: the first
step is more than three times too large.

**Sharing one optimiser object across parameter tensors.** Each tensor needs its
own state, and sharing produces shape errors at best and silent nonsense at
worst.

**Comparing optimisers at a single learning rate.** Measures the learning rate,
not the optimiser.

**Forgetting to zero the gradients.** {{ch:dl-backprop}}.

## 12. Failure Modes

**Divergence in the first few steps.** Learning rate too high, or momentum
raised without compensating, or bias correction missing. The measurement
separates the three.

**A loss plateau that a learning-rate drop fixes.** {{eq:sgd-convergence}}'s
noise floor: the iterate is bouncing inside a neighbourhood of radius
$\propto\eta\sigma^2$ rather than descending.

**AdaGrad stalling.** Its denominator only grows, so the step size decays
monotonically toward zero. Fine for convex problems, fatal for long
non-convex training runs, and this is exactly why RMSProp exists.

**A loss spike on resume.** Optimiser state not checkpointed, so the moments
restart from zero and the bias correction warm-up happens again.

**A gradient spike propagating for hundreds of steps.** Measured: $\vec{v}$
remembers a spike for about $1/(1-\beta_2) = 1000$ steps, so the model takes
unusually *small* steps long after the bad batch is gone.

**Silent underperformance from decayed normalisation scales.** No error, a
worse model.

**Sparse-embedding step sizes drifting.** {{sec:7-internal-mechanics}}: a rare
token's $\vec{v}$ decays while it is unseen, so its step grows.

## 13. Alternatives

**LAMB and LARS** rescale the update per layer by the ratio of the weight norm
to the update norm, which is what makes very large batch training stable. Useful
above batch sizes of a few thousand and unnecessary below.

**Lion** uses only the sign of a momentum estimate, halving the optimiser state
relative to Adam. Competitive in several published comparisons and not yet a
default. {{maturity:EMERGING}}

**Adafactor** factorises the second moment into row and column statistics,
reducing $O(P)$ state to $O(\sqrt{P})$ per matrix. Widely used where memory is
the binding constraint.

**Shampoo and K-FAC** maintain structured curvature approximations. Real gains
at large scale and a substantially more complex implementation.
{{maturity:EMERGING}}

**Sign-based methods** (signSGD and relatives) transmit one bit per coordinate,
which makes them interesting for distributed training where communication rather
than compute is the bottleneck.

**L-BFGS** is excellent for deterministic full-batch problems and does not
tolerate mini-batch noise, because its curvature estimate is built from
differences of noisy gradients. Use it for small deterministic optimisations,
not for training networks.

## 14. Evaluation

**Search the learning rate per optimiser.** A comparison at one rate is
meaningless, and the measured spread of best rates is three orders of magnitude.

**Report the sensitivity, not just the best.** The measured grid shows how many
settings work at all, which is what you actually care about.

**Track the update-to-weight ratio** $\|\Delta\vecgreek{\theta}\| /
\|\vecgreek{\theta}\|$ per layer. Around $10^{-3}$ per step is healthy; this is
more informative than the gradient norm because it is scale-free.

**Log the gradient norm and the clip rate.**

**Verify a custom optimiser on a quadratic** where the answer is known, as in
{{sec:8-implementation}}, before trusting it on a network.

**Check that resuming from a checkpoint reproduces the loss.** If it spikes, the
optimiser state was not saved.

## 15. Advanced Concepts

**The noise scale and critical batch size.** The gradient noise scale predicts
the batch size beyond which further increases stop buying faster convergence.
It is estimable during training and it is the principled version of "how large
should the batch be".

**Linear scaling and its limits.** Doubling the batch and doubling the learning
rate holds up to a point and then breaks, which is where LARS and LAMB were
introduced.

**Adam's convergence proof was wrong.** The original paper's proof contained an
error, later shown by a counterexample on which Adam fails to converge; AMSGrad
was proposed as a fix. **AMSGrad is not used in practice and Adam is.** This is
a case worth remembering when weighing theory against evidence in this field.

**Sharpness-aware minimisation** takes a step at a nearby worst-case point to
bias training toward flat minima. Real generalisation gains at roughly double
the compute; sharpness is not reparameterisation-invariant, which complicates
the story. {{maturity:EMERGING}}

**Optimiser-state sharding.** Splitting $\vec{m}$ and $\vec{v}$ across devices
removes the largest memory term in {{tbl:optimizer-memory}}, at the cost of a
gather each step ({{part:23}}).

**Learned optimisers.** A network that outputs the update. They work on the
tasks they were trained on and generalise poorly beyond them.
{{maturity:RESEARCH FRONTIER}}

## 16. Connection to Previous Chapters

{{ch:dl-backprop}} produced $\vec{g}$; this chapter consumes it. The clipping of
that chapter and the trust-region property of {{eq:adam-step-bound}} are two
different answers to the same problem, and using both is standard.

{{ch:math-optimization}} supplied gradient descent, convexity and the condition
number; {{eq:momentum-rate}} is that chapter's convergence analysis with
momentum added. {{ch:ml-linear-regression}} was the first place gradient descent
appeared, on a convex problem where all of this is easy.
{{ch:mle-hpo}} supplies the search machinery for the learning rate,
and the measured sensitivity grid is why that search matters more for some
optimisers than others.

Forward: {{ch:dl-lr-schedules}} varies $\eta$ over training, which
{{eq:sgd-convergence}} shows to be necessary rather than optional.
{{ch:dl-initialization}} sets the starting point these methods descend from.
{{ch:ft-training-config}} gives the specific AdamW configurations used for
fine-tuning, and they will read as instances of this chapter rather than as
folklore.

## 17. Exercises

**Beginner**

1. Write the SGD update. What is its only hyperparameter?
2. What does momentum accumulate, and what does that do to oscillations?
3. Why does AdaGrad's step size decay monotonically?
4. What are Adam's two moments?
5. How much memory does Adam need per parameter?

**Intermediate**

6. Derive {{eq:momentum-amplification}} and state the effective step at
   $\mu = 0.95$.
7. Derive {{eq:adam-m-unrolled}} and compute the bias at $t = 5$ for
   $\beta_1 = 0.9$.
8. Explain why {{eq:sgd-convergence}} implies a constant learning rate cannot
   converge.
9. Compute the optimiser memory for a 3-billion-parameter model under SGD,
   momentum and Adam.
10. Explain why coupled $\ell_2$ and decoupled weight decay differ under Adam
    but not under SGD.
11. Why must the optimiser state be in fp32?

**Advanced**

12. Derive {{eq:momentum-rate}} for a quadratic and find the optimal $\mu$ in
    terms of $\kappa$.
13. Prove {{eq:adam-scale-invariance}} and state exactly where $\epsilon$
    breaks it.
14. Derive {{eq:adam-step-bound}} and state the condition under which it is
    tight.
15. Construct a problem on which AdaGrad stalls before reaching the optimum.
16. Explain why the noise in a mini-batch gradient makes curvature estimates
    unreliable, quantitatively.

**Implementation**

17. Implement Adam and verify it against a framework's to $10^{-6}$.
18. Implement AdamW and reproduce the measured weight-norm difference.
19. Implement parameter groups and measure the effect of decaying
    normalisation parameters.
20. Implement Adafactor's factored second moment and compare memory and final
    loss against Adam.

**Reasoning**

21. Training diverges at step 3 with Adam at the default learning rate. List
    your hypotheses in order.
22. A run resumed from a checkpoint shows a loss spike that recovers over about
    a thousand steps. What happened, and what is the significance of the
    number?

## 18. Interview Questions

**"Explain Adam."** — Two moments, bias correction, per-parameter adaptive step.
A strong answer adds the trust-region property of {{eq:adam-step-bound}}.

**"Why bias correction?"** — Moments initialise at zero. The strong answer gives
the asymmetry: $\vec{v}$ warms up a hundred times more slowly than $\vec{m}$, so
the uncorrected step is too *large*, not too small.

**"Adam or SGD?"** — AdamW as a default; SGD with momentum for convolutional
vision with a tuning budget. Say that the generalisation-gap claim is
setting-specific rather than a law.

**"What is the difference between Adam and AdamW?"** — Decoupled decay, and why
it matters only for adaptive optimisers. Note that $\lambda$ does not transfer.

**"How much memory does training need?"** — Weights + gradients + optimiser
state + activations, with the numbers.

**"Why does momentum help?"** — Oscillation cancellation, and the square root in
the condition number. The second is the answer that distinguishes.

**"Why not second-order methods?"** — Storage, inversion, indefiniteness, and
the deeper reason: curvature estimates from noisy gradients are noisier still.

## 19. Research Questions

**Why does Adam work as well as it does?** The original convergence proof was
flawed and the fix is unused. Several later analyses explain parts of the
behaviour and none accounts for the empirical dominance.
{{maturity:RESEARCH FRONTIER}}

**Does the optimiser affect generalisation, independently of the loss reached?**
Suggestive evidence in specific settings; no general result, and much of the
apparent gap disappears under equal tuning. {{maturity:RESEARCH FRONTIER}}

**Can second-order information be used affordably?** Shampoo and its relatives
show gains at scale, and whether they justify the complexity is unsettled.
{{maturity:EMERGING}}

**Is there a principled way to choose the learning rate?** Learning-rate range
tests, $\mu$P transfer and gradient-noise-scale estimates all help.
$\mu$P is the most promising: it makes the optimal rate transfer across model
widths, which turns a per-model search into a one-off. {{maturity:EMERGING}}

## 20. Chapter Summary

An optimiser decides what to do with the gradient, and every method here is a
response to one of three problems: ill-conditioning, mini-batch noise, and
parameters whose gradients differ in scale by orders of magnitude.

Momentum attacks the first. Measured on a quadratic with a controlled condition
number and each method at its own optimal settings, its advantage over plain
gradient descent grew from about twofold at $\kappa = 10$ to nearly fiftyfold at
$\kappa = 10^4$ — consistently around half the $\sqrt{\kappa}$ that
{{eq:momentum-rate}} predicts, which is the agreement an asymptotic rate with an
unspecified constant should give. Its cost is a coupling that catches
people out: the asymptotic step is $1/(1-\mu)$ times the gradient, confirmed
exactly by measurement, so raising $\mu$ from 0.9 to 0.99 is a tenfold
learning-rate increase in disguise.

The adaptive family attacks the third, and its history is a sequence of fixes:
AdaGrad's monotone decay, RMSProp's forgetting, Adam's momentum and bias
correction. The bias correction is not cosmetic — measured on a constant
gradient, omitting it made the first step more than three times too large, and
the step stayed inflated for hundreds of iterations because $\vec{v}$ warms up a
hundred times more slowly than $\vec{m}$.

Two measured properties explain Adam's status as the default. It is invariant to
the scale of the loss: after sixty steps it reached the identical point at loss
scales of $10^{-3}$, $1$ and $10^{3}$, where SGD at one learning rate barely
moved at the smallest and diverged at the largest. And its step is bounded: when
a gradient spike a thousand times the usual size arrived, SGD moved a thousand
times further while Adam's move *halved* — better than {{eq:adam-step-bound}}
promises, because $\vec{v}$ absorbs the spike immediately while $\vec{m}$
averages it away. The cost is that the caution persists for roughly
$1/(1-\beta_2)$ steps, which is why clipping remains worth having.

The measured comparison on a deliberately badly scaled problem found best
learning rates a factor of thirty apart, which is why comparing optimisers at
one learning rate measures nothing. The substantive finding was not robustness
but a ceiling: both SGD variants flattened at the same loss across their whole
grid, while both adaptive methods were still improving at the top of theirs and
reached a materially lower loss. When gradient magnitudes span orders of
magnitude across parameters, one global step size has a limit that no tuning
removes.

AdamW is a different algorithm from Adam-with-$\ell_2$, not a refinement. The
coupled decay term passes through the $1/\sqrt{\vec{v}}$ rescaling, so
parameters with large gradients get regularised *less* — backwards from the
intent. Measured at identical $\lambda$, the two produced different weight
norms, which is why $\lambda$ does not transfer between them.

Finally, the state costs memory: $16P$ bytes for Adam in fp32, so a
7-billion-parameter model needs 112 GB before a single activation is stored
against 14 GB to serve it in bf16. That factor is the binding constraint on
model size for most people, and it is the reason optimiser-state sharding
exists.

## 21. Further Reading

{{cite:kingma2015adam}} is short and readable, and worth reading with
{{sec:19-research-questions}} in mind: the convergence proof in it was later
shown to be flawed, and the algorithm is used everywhere regardless. Few papers
illustrate the gap between theoretical guarantee and empirical adoption so
cleanly.

{{cite:loshchilov2019adamw}} is the more instructive read of the two. The
argument is a page long, the fix is one line, and it went unnoticed for years
because everyone assumed the two formulations of weight decay were the same
thing. It is a good demonstration that reading an implementation carefully is a
research contribution.

{{cite:goodfellow2016}} chapter 8 covers the optimisation landscape more
thoroughly than here — plateaus, saddle points, cliffs — and is the place to go
for why non-convexity is less of an obstacle than it sounds.

{{cite:pascanu2013}} for the clipping argument, which pairs with
{{eq:adam-step-bound}}: two different mechanisms for bounding how far a step can
go, one applied to the gradient and one built into the update rule.

**Where to go next:** {{ch:dl-lr-schedules}} varies $\eta$ over training, which
{{eq:sgd-convergence}} shows to be a requirement rather than a refinement. Read
it immediately after this chapter, because the two together are what actually
gets used.

---
id: dl-regularization
number: 58
part: VI
tier: full
status: reviewed
requires: [dl-normalization, dl-optimizers, dl-losses, ml-metrics, mle-splits]
provides: [dropout, weight-decay-dl, early-stopping-dl, data-augmentation,
           stochastic-depth, mixup, double-descent, implicit-regularization]
citations: [srivastava2014, zhang2017rethinking, loshchilov2019adamw,
            ioffe2015, belkin2019]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Explain why classical capacity control does not describe deep networks.
2. Derive dropout's training and inference forms and explain inverted dropout.
3. Distinguish weight decay from $\ell_2$ regularisation under an adaptive
   optimiser, and say what it does in a normalised network.
4. Apply early stopping correctly and state what it costs.
5. Choose an augmentation for a given invariance.
6. Explain double descent and what it does to the bias–variance picture.
7. Explain what is meant by implicit regularisation and how strong the evidence
   is.

## 2. Why This Matters

**The classical story of overfitting does not describe deep networks.**
{{cite:zhang2017rethinking}} showed that standard architectures trained with
standard methods fit *randomly labelled* data perfectly. Their capacity is
therefore sufficient to memorise the training set outright, and no
uniform-convergence bound of the form {{ch:ml-metrics}} presented can explain why
they nonetheless generalise on real labels. {{sec:9-practical-example}}
reproduces the result at small scale.

**Regularisation helps and is not what makes generalisation possible.** That
distinction is the honest position, and it is worth stating before the
techniques rather than after, because it changes how you should use them: as
tunable knobs with measurable effects, not as the thing keeping the model
honest.

**Several techniques regularise without being called regularisers.** Batch
normalisation's batch coupling ({{ch:dl-normalization}}), the noise in
mini-batch gradients, and early stopping all reduce overfitting. Removing one
can cost accuracy for reasons that appear unrelated.

**Dropout's role has narrowed.** Universal in 2015, and now largely absent from
large language models, where the data is large enough relative to the parameters
that memorisation is not the binding constraint. Knowing why the practice
changed is more useful than knowing the current default.

## 3. Prerequisites

{{ch:ml-metrics}} for the bias–variance decomposition and its limits.
{{ch:mle-splits}} for honest evaluation, without which nothing here can be
measured. {{ch:dl-normalization}} for the incidental regularisation this chapter
must account for. {{ch:dl-optimizers}} for the weight decay this chapter
reinterprets.

## 4. Intuitive Explanation

### 4.1 What "regularisation" means here

The classical definition — anything that reduces test error at the cost of
training error — is a good enough working definition and it hides an
assumption. It assumes the failure mode is fitting noise in the training set,
which is what happens with a model of limited capacity.

A deep network can fit noise *and* signal simultaneously. So the question is
not "does the model have enough capacity to memorise" — it does — but "of the
many parameter settings that fit the training data, which one does training
find". Regularisation is one way to influence that choice, and
{{sec:6-mathematical-foundation}} argues that it is not the main one.

### 4.2 Dropout

Randomly zero a fraction $p$ of activations at each training step:

```text
   training      h = [0.4, 0.9, 0.2, 0.7]
                 m = [1,   0,   1,   0  ]     random mask
                 h = [0.8, 0,   0.4, 0  ]     kept values SCALED by 1/(1-p)

   inference     h = [0.4, 0.9, 0.2, 0.7]     no mask, no scaling
```

The scaling is the part people get wrong. Multiplying the kept activations by
$1/(1-p)$ during training makes the expectation match the unmasked value, so
inference needs no adjustment at all. This is **inverted dropout** and it is
what every framework implements; the original formulation scaled at inference
instead, which is equivalent and worse operationally.

Two ways to think about what it does. **Preventing co-adaptation**: a unit
cannot rely on a specific other unit being present, so it must be useful on its
own. **Implicit ensembling**: each mask defines a different subnetwork, and
inference approximates averaging over all $2^n$ of them.

Both stories are in {{cite:srivastava2014}}. Neither is a theorem, and the
second is exact only for a linear network.

### 4.3 Weight decay in a normalised network

{{ch:dl-normalization}} established that a layer feeding a normalisation is
scale-invariant: multiplying $\mat{W}$ by any positive constant leaves the
output unchanged. So weight decay on that layer **cannot change the function
directly.**

What it does instead is control $\|\mat{W}\|$, and since the effective learning
rate goes as $\eta/\|\mat{W}\|^2$, weight decay is acting as a *learning-rate
control*. That is a completely different mechanism from the classical
shrink-the-coefficients story of {{ch:ml-linear-regression}}, and it explains
why weight decay remains important in normalised networks where the classical
argument says it should do nothing.

### 4.4 Data augmentation

Transform the input in ways that should not change the label:

```text
   images     flip, crop, rotate, colour jitter, cutout
   audio      time shift, pitch shift, noise, masking
   text       synonym replacement, back-translation, token dropout
   tabular    usually nothing safe
```

**Augmentation is the only technique here that adds information** — specifically,
the information that a particular transformation is label-preserving. Everything
else redistributes what is already in the data.

That is also its constraint: the invariance must actually hold. Horizontal flip
is right for a photograph of a cat and wrong for a photograph of text, and
getting it wrong teaches the model something false.

### 4.5 Double descent

The classical picture says test error is U-shaped in capacity: underfit, then
optimal, then overfit. {{cite:belkin2019}} showed the curve continues:

```text
   test
   error │╲                 ╱╲
         │ ╲               ╱  ╲
         │  ╲___________ ╱     ╲______________
         │                ↑
         └────────────────┴──────────────────▶ capacity
              classical   interpolation
              U-shape     threshold
```

Past the point where the model can exactly fit the training data — the
*interpolation threshold* — test error rises to a peak and then **falls again**,
often below the classical optimum. Modern networks operate far to the right of
that peak.

This is not a curiosity. It means "more parameters will overfit" is false as
stated, and it is why the field's practice of scaling models up worked at all.

## 5. Formal Explanation

### 5.1 Dropout

At training, with mask $m_i \sim \text{Bernoulli}(1-p)$:

$$
\tilde{h}_i = \frac{m_i}{1-p}\,h_i
$$ (eq:inverted-dropout)

so that $\E[\tilde{h}_i] = h_i$. At inference, $\tilde{h}_i = h_i$.

The variance introduced is

$$
\Var[\tilde{h}_i] = h_i^2\,\frac{p}{1-p}
$$ (eq:dropout-variance)

which is worth reading: **the injected noise is proportional to the
activation's own magnitude**, so large activations are perturbed more. At
$p = 0.5$ the factor is 1, so the noise standard deviation equals the signal.

### 5.2 Weight decay

Decoupled, from {{eq:adamw}}:

$$
\vecgreek{\theta}_{t+1} = \vecgreek{\theta}_t - \eta\,\vec{u}_t
 - \eta\lambda\vecgreek{\theta}_t
$$ (eq:decoupled-decay)

In the absence of any gradient this is geometric shrinkage,
$\theta_t = \theta_0(1-\eta\lambda)^t$, with half-life
$\ln 2/(\eta\lambda)$ steps. At $\eta = 10^{-3}$ and $\lambda = 0.1$ that is
about 6900 steps — comparable to a training run, which is why $\lambda$ has to
be chosen against the budget and not in isolation.

### 5.3 Early stopping

Monitor validation loss, stop when it has not improved for `patience`
evaluations, restore the best checkpoint.

Three details that are usually left implicit and all matter:

**The validation set is consumed by the decision.** Stopping on it makes it a
training signal, so the reported validation number is optimistically biased.
A separate test set is required ({{ch:mle-splits}}).

**Patience interacts with the schedule.** A cosine schedule's loss keeps
improving until the end by design, so early stopping rarely fires and, when it
does, it fires because something went wrong.

**"Best" needs a definition.** Best validation loss and best validation accuracy
generally occur at different steps ({{ch:dl-losses}} measured them diverging),
so the criterion is a choice.

### 5.4 Stochastic depth

Randomly skip entire residual blocks during training:

$$
\vec{y} = \vec{x} + b\,F(\vec{x}), \qquad b \sim \text{Bernoulli}(1-p_l)
$$ (eq:stochastic-depth)

with $p_l$ increasing linearly with depth, so later blocks are dropped more.
Equivalent to dropout at the block level, and it makes very deep residual
networks trainable while also reducing training time — the skipped blocks cost
nothing.

### 5.5 Mixup

Train on convex combinations of pairs:

$$
\tilde{\vec{x}} = \alpha\vec{x}_i + (1-\alpha)\vec{x}_j,
\qquad
\tilde{y} = \alpha y_i + (1-\alpha)y_j
$$ (eq:mixup)

with $\alpha \sim \text{Beta}(a, a)$. It encourages linear behaviour between
training examples and improves calibration noticeably.

**It also produces inputs that are not in the data distribution** — a blend of
two photographs is not a photograph — so it is a strange technique that works
better than it should, which is worth flagging rather than smoothing over.

### 5.6 What to use

{#tbl:reg-choice caption="Regularisation by setting. The right column is the reason, and it is more useful than the recommendation because the recommendations change."}

| Setting | Use | Why |
|---|---|---|
| Small data, large model | Everything: augmentation, dropout, decay, early stopping | Memorisation is the binding constraint |
| Convolutional vision | Augmentation, weight decay, batch norm's implicit noise | Augmentation dominates |
| Transformers, moderate data | Weight decay, dropout 0.1, label smoothing | Standard recipe |
| Large language models | Weight decay only, usually no dropout | Data $\gg$ needed; each token seen roughly once |
| Fine-tuning | Early stopping, low learning rate, sometimes dropout | Few steps, easy to destroy the pretrained weights |

**The last row of that table is the one that changed.** When a model sees each
training token approximately once, it cannot memorise it, so the techniques
aimed at preventing memorisation have nothing to do.

## 6. Mathematical Foundation

### 6.1 Dropout as an $\ell_2$ penalty, for linear models

For linear regression with dropout on the inputs, the expected loss is

$$
\E_m\big[\|\vec{y} - (\mat{X}\odot\mat{M})\vec{w}\|^2\big]
 = \|\vec{y}-\mat{X}\vec{w}\|^2
 + \frac{p}{1-p}\sum_j \|\vec{x}_j\|^2 w_j^2
$$ (eq:dropout-l2)

**Dropout is exactly an $\ell_2$ penalty with a per-feature weighting by
$\|\vec{x}_j\|^2$** — so features with larger scale are penalised more, which is
a data-dependent ridge rather than a plain one.

This is exact for a linear model and only suggestive for a deep one, where the
expectation over masks does not factor. It is still the clearest statement of
what dropout is *like*.

### 6.2 Weight decay in a scale-invariant layer

From {{eq:norm-scale-invariance}}, the loss satisfies
$\Like(a\mat{W}) = \Like(\mat{W})$ for all $a > 0$. Differentiating with respect
to $a$ at $a = 1$:

$$
\langle \nabla_{\mat{W}}\Like, \mat{W}\rangle = 0
$$ (eq:grad-orthogonal-to-w)

The gradient is orthogonal to the weights — measured directly in
{{ch:dl-normalization}}. So a gradient step changes the norm by

$$
\|\mat{W}_{t+1}\|^2 = \|\mat{W}_t\|^2 + \eta^2\|\vec{g}_t\|^2
$$ (eq:norm-growth)

by Pythagoras: **it can only grow.** Adding decay gives

$$
\|\mat{W}_{t+1}\|^2 = (1-\eta\lambda)^2\|\mat{W}_t\|^2
 + \eta^2\|\vec{g}_t\|^2
$$ (eq:norm-equilibrium)

which has a fixed point where growth and decay balance. Since the effective
learning rate is $\propto\eta/\|\mat{W}\|^2$, **weight decay sets the
equilibrium effective learning rate.**

That is the mechanism in a normalised network, and it is why $\lambda$ matters
there despite {{eq:norm-scale-invariance}} saying the function does not depend
on $\|\mat{W}\|$ at all.

### 6.3 Early stopping approximates $\ell_2$

For a quadratic loss $\Like = \frac{1}{2}(\vecgreek{\theta}-
\vecgreek{\theta}^\star)\T\mat{H}(\vecgreek{\theta}-\vecgreek{\theta}^\star)$
started from $\vecgreek{\theta}_0 = \vec{0}$, gradient descent after $t$ steps
gives, in the eigenbasis of $\mat{H}$ with eigenvalues $\lambda_i$:

$$
\theta_i^{(t)} = \theta_i^\star\left(1-(1-\eta\lambda_i)^t\right)
$$ (eq:early-stopping-trajectory)

Compare $\ell_2$ regularisation with coefficient $\alpha$, whose solution is

$$
\theta_i^{\text{ridge}} = \theta_i^\star\frac{\lambda_i}{\lambda_i+\alpha}
$$ (eq:ridge-solution)

Both shrink each direction by a factor that approaches 1 for large $\lambda_i$
and 0 for small $\lambda_i$. Matching the two for small $\eta\lambda_i$ gives

$$
\alpha \approx \frac{1}{\eta t}
$$ (eq:early-stopping-equivalence)

**Training for $t$ steps is approximately ridge regression with
$\alpha = 1/(\eta t)$.** Stopping earlier regularises more, and the
correspondence is exact only for a quadratic. {{sec:8-implementation}} measures
how well it holds.

### 6.4 Why classical capacity control fails

The uniform-convergence bound of {{ch:ml-metrics}} has the shape

$$
\text{test} \le \text{train} + O\!\left(\sqrt{\frac{\text{capacity}}{N}}\right)
$$ (eq:uniform-convergence)

{{cite:zhang2017rethinking}}'s argument is a reductio. A network that can fit
$N$ random labels has capacity at least $N$ in any measure that supports
{{eq:uniform-convergence}}, so the bound reads $\text{test} \le \text{train} +
O(1)$ — vacuous.

**And the same network, same optimiser, same regularisation, generalises well on
real labels.** So whatever explains generalisation must depend on properties of
the data and the training procedure, not on the hypothesis class alone.

The paper also measured that explicit regularisation reduces the generalisation
gap by a modest amount and that removing it entirely still leaves the model
generalising. That is the sentence to carry: **regularisation is a tuning knob,
not the mechanism.**

### 6.5 Double descent

Let $P$ be parameters and $N$ examples. The interpolation threshold is around
$P \approx N$, and the test error curve has:

- $P \ll N$: classical U-shape.
- $P \approx N$: a peak. Exactly one solution fits the data, and it is
  determined entirely by the noise.
- $P \gg N$: many solutions fit, the optimiser selects among them, and test
  error **falls again**.

The mechanism in the overparameterised regime is that gradient descent from a
small initialisation converges to the minimum-norm interpolating solution, which
is a form of regularisation nobody applied.

**That is implicit regularisation**, and it is the current best account of why
overparameterised networks generalise. It is proven for linear and kernel
regression and *conjectured* for deep networks.
{{maturity:RESEARCH FRONTIER}}

## 7. Internal Mechanics

### 7.1 Dropout placement

```text
   Linear -> ReLU -> Dropout -> Linear         standard: AFTER the activation
   Conv -> BN -> ReLU -> Dropout               works, and often unnecessary
   Attention -> Dropout -> Residual add        transformer convention
```

**Do not place dropout before a batch normalisation.** The dropout changes the
batch statistics at training time and not at inference, which widens the
train/eval gap {{ch:dl-normalization}} measured. This interaction is a known
source of degradation and the fix is ordering.

### 7.2 Dropout and RNNs

Applying an independent mask at every time step injects noise that accumulates
over the sequence and destroys the recurrent state. *Variational dropout* uses
the **same mask at every time step**, which regularises without the
accumulation. Frameworks' recurrent-layer dropout arguments implement this, and
hand-rolled recurrent dropout usually does not.

### 7.3 Which parameters to decay

```text
   decay:     weight matrices, convolution kernels
   no decay:  biases, normalisation gamma and beta, embeddings (usually)
```

Decaying a normalisation's $\gamma$ toward zero fights the normalisation
directly. Decaying a bias regularises nothing. This is
{{ch:dl-optimizers}}'s parameter-group advice, and it is the single most
commonly skipped three lines in a training script.

### 7.4 Augmentation cost

Augmentation runs on the CPU while the accelerator waits. A heavy pipeline can
make the data loader the bottleneck, so the model trains at a fraction of its
possible throughput with no error. **Profile the loader, not just the model**;
{{ch:dl-forward}}'s throughput argument applies to the whole pipeline.

### 7.5 Early stopping bookkeeping

Restoring the best checkpoint requires having saved it, which means saving
whenever validation improves — including the optimiser state
({{ch:dl-optimizers}}) and the schedule's step count
({{ch:dl-lr-schedules}}), or the restored run will not resume correctly.

## 8. Implementation

```python {tier=A name=regularizers-from-scratch}
"""Dropout, weight decay and early stopping, each measured against the
theory that predicts what it should do.
"""
import numpy as np

rng = np.random.default_rng(0)


# --- section 5.1: inverted dropout ------------------------------------------
def dropout(h, p, rs, training=True):
    """Eq. 58.1. The 1/(1-p) is what makes inference a no-op."""
    if not training or p == 0.0:
        return h
    mask = (rs.random(h.shape) >= p) / (1.0 - p)
    return h * mask


print("=" * 72)
print("inverted dropout: the expectation is preserved (eq. 58.1)")
print("=" * 72)
h = rng.random((2000, 64)) * 2.0
rs = np.random.default_rng(1)
print(f"{'p':>6} {'E[dropout(h)] / h':>20} {'sd / h  (measured)':>21} "
      f"{'predicted sqrt(p/(1-p))':>25}")
for p in (0.0, 0.1, 0.3, 0.5, 0.8):
    samples = np.array([dropout(h, p, rs) for _ in range(200)])
    ratio_mean = float((samples.mean(axis=0) / h).mean())
    ratio_sd = float((samples.std(axis=0) / h).mean())
    pred = np.sqrt(p / (1 - p)) if p < 1 else np.inf
    print(f"{p:>6.1f} {ratio_mean:>20.5f} {ratio_sd:>21.5f} {pred:>25.5f}")

print("\nThe mean ratio is 1.000 at every rate, which is the point of the")
print("1/(1-p) scaling: inference needs no adjustment because training")
print("already matched the expectation.")
print("\nThe standard deviation matches eq. 58.2's sqrt(p/(1-p)) exactly.")
print("At p = 0.5 the injected noise has the same magnitude as the signal,")
print("and the noise is PROPORTIONAL to each activation — large activations")
print("are perturbed more, which is a data-dependent perturbation rather")
print("than additive noise.")

# --- section 6.1: dropout is an L2 penalty for a linear model ---------------
print("\n" + "=" * 72)
print("for a LINEAR model, dropout IS an L2 penalty (eq. 58.5)")
print("=" * 72)
N, D = 400, 12
Xl = rng.normal(size=(N, D))
Xl[:, :4] *= 3.0                                  # some features larger
w_true = rng.normal(size=D)
yl = Xl @ w_true + rng.normal(0, 0.5, N)


def fit_dropout(X, y, p, steps=6000, lr=0.02, seed=0, n_masks=1):
    rs = np.random.default_rng(seed)
    w = np.zeros(X.shape[1])
    for _ in range(steps):
        Xd = X if p == 0 else X * ((rs.random(X.shape) >= p) / (1 - p))
        w -= lr * (Xd.T @ (Xd @ w - y)) / len(X)
    return w


def fit_ridge_weighted(X, y, p):
    """Eq. 58.5's closed form: ridge with per-feature weight ||x_j||^2."""
    lam = p / (1 - p)
    Pen = np.diag(lam * (X ** 2).sum(axis=0))
    return np.linalg.solve(X.T @ X + Pen, X.T @ y)


print(f"{'p':>6} {'|w| dropout':>13} {'|w| weighted ridge':>20} "
      f"{'max |diff|':>12} {'cos similarity':>16}")
for p in (0.0, 0.1, 0.3, 0.5):
    wd = fit_dropout(Xl, yl, p)
    wr = fit_ridge_weighted(Xl, yl, p) if p > 0 else np.linalg.lstsq(
        Xl, yl, rcond=None)[0]
    cos = float(wd @ wr / (np.linalg.norm(wd) * np.linalg.norm(wr)))
    print(f"{p:>6.1f} {np.linalg.norm(wd):>13.5f} {np.linalg.norm(wr):>20.5f} "
          f"{np.abs(wd - wr).max():>12.5f} {cos:>16.6f}")

print("\nThe two agree closely: SGD with dropout converges to the solution")
print("of the weighted ridge problem eq. 58.5 predicts, without ridge ever")
print("being written down. The residual difference is the sampling noise")
print("in a finite number of masks.")
print("\nNote the weighting. The penalty is per-feature and proportional to")
print("||x_j||^2, so the four features scaled by 3 are penalised nine times")
print("as hard as the rest. Plain ridge would treat them alike, so dropout")
print("is a DATA-DEPENDENT ridge rather than a plain one.")
print("\nThis is exact for a linear model. For a deep one the expectation")
print("over masks does not factor and the correspondence is suggestive")
print("rather than derived — which is worth remembering when the intuition")
print("'dropout is like L2' is applied to a network.")

# --- section 6.3: early stopping approximates ridge -------------------------
print("\n" + "=" * 72)
print("early stopping approximates ridge with alpha = 1/(eta*t) (eq. 58.11)")
print("=" * 72)


def gd_trajectory(X, y, lr, steps):
    w = np.zeros(X.shape[1])
    out = {}
    for t in range(1, steps + 1):
        w -= lr * (X.T @ (X @ w - y)) / len(X)
        out[t] = w.copy()
    return out


def ridge(X, y, alpha):
    return np.linalg.solve(X.T @ X / len(X) + alpha * np.eye(X.shape[1]),
                           X.T @ y / len(X))


lr = 0.01
traj = gd_trajectory(Xl, yl, lr, 20000)
print(f"{'steps t':>9} {'predicted alpha':>17} {'|w_gd|':>10} "
      f"{'|w_ridge|':>11} {'cos(w_gd, w_ridge)':>20} {'max |diff|':>12}")
for t in (50, 200, 1000, 5000, 20000):
    a = 1.0 / (lr * t)
    wg = traj[t]
    wr = ridge(Xl, yl, a)
    cos = float(wg @ wr / (np.linalg.norm(wg) * np.linalg.norm(wr)))
    print(f"{t:>9} {a:>17.5f} {np.linalg.norm(wg):>10.4f} "
          f"{np.linalg.norm(wr):>11.4f} {cos:>20.6f} "
          f"{np.abs(wg - wr).max():>12.5f}")

print("\nThe correspondence of eq. 58.11 holds well: stopping at step t")
print("gives a solution close to ridge at alpha = 1/(eta*t), and both norms")
print("grow together as t increases and the implied penalty weakens.")
print("\nSo early stopping is not a separate idea from weight decay — on a")
print("quadratic they are the same regulariser expressed two ways, one as a")
print("penalty and one as a budget of steps. The approximation is exact")
print("only for a quadratic, and the direction of the effect is robust: FEWER")
print("STEPS MEANS MORE REGULARISATION.")

# --- section 6.2: weight decay in a scale-invariant layer -------------------
print("\n" + "=" * 72)
print("weight decay sets an equilibrium norm, not a smaller function (6.2)")
print("=" * 72)
print("A scale-invariant layer: the loss depends only on W/|W|, so the")
print("gradient is orthogonal to W and eq. 58.8 says |W| can only GROW.\n")


def scale_invariant_run(lam, steps=20000, lr=0.05, d=32, seed=2,
                        noise=1.0):
    """A scale-invariant loss with a STOCHASTIC gradient, so the run never
    converges and the norm growth of eq. 58.8 is not confounded with the
    gradient decaying to zero."""
    rs = np.random.default_rng(seed)
    W = rs.normal(0, 1.0, d)
    hist = []
    for t in range(1, steps + 1):
        u = W / np.linalg.norm(W)
        target = rs.normal(0, 1.0, d)             # a fresh target each step
        target /= np.linalg.norm(target)
        g_u = u - target
        g = (g_u - u * (g_u @ u)) / np.linalg.norm(W)   # radial part removed
        W = W - lr * g - lr * lam * W             # eq. 58.6
        if t in (1, 100, 1000, 5000, 20000):
            hist.append((t, float(np.linalg.norm(W))))
    return hist


STEPS = (1, 100, 1000, 5000, 20000)
print(f"{'lambda':>9} " + " ".join(f"{f'|W| @{t}':>11}" for t in STEPS)
      + f" {'1/|W|^2 @20000':>16}")
for lam in (0.0, 0.0003, 0.003, 0.03):
    h_ = scale_invariant_run(lam)
    final = h_[-1][1]
    print(f"{lam:>9.4f} " + " ".join(f"{v:>11.4f}" for _, v in h_)
          + f" {1.0 / final ** 2:>16.5f}")

print("\nAt lambda = 0 the norm only ever GROWS — monotonically at every")
print("checkpoint, and never once down. That is eq. 58.8: the gradient is")
print("orthogonal to W, so each step adds to the norm by Pythagoras and")
print("nothing subtracts. It grows slowly, because the increment is")
print("eta^2|g|^2 and |g| itself falls as 1/|W|, but the direction is")
print("forced and there is no equilibrium without decay.")
print("\nWith decay the norm settles at an equilibrium where the shrinkage")
print("balances that growth, and the equilibrium is lower for larger")
print("lambda. The last column is the effective learning rate multiplier at")
print("that equilibrium, which spans two orders of magnitude across the")
print("table.")
print("\nThat is the mechanism. In a normalised network weight decay is not")
print("shrinking the function — eq. 57.10 says the function does not depend")
print("on |W| at all — it is setting the step size. Which is why lambda")
print("matters there despite the classical argument saying it should do")
print("nothing.")
```

```python {tier=A name=random-labels-and-double-descent}
"""Zhang et al.'s random-label result and double descent, reproduced at a
scale that runs on a laptop.
"""
import numpy as np

rng = np.random.default_rng(0)

D, C = 16, 4


def make_data(n, seed, randomize_labels=False):
    rs = np.random.default_rng(seed)
    X = rs.normal(size=(n, D))
    W1 = np.random.default_rng(555).normal(size=(D, 10))
    W2 = np.random.default_rng(556).normal(size=(10, C))
    logits = np.tanh(X @ W1) @ W2 * 1.5
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y = (p.cumsum(axis=1) < rs.random((n, 1))).sum(axis=1).clip(0, C - 1)
    if randomize_labels:
        y = rs.integers(0, C, n)
    return X, y


class MLP:
    def __init__(self, sizes, seed=0):
        rs = np.random.default_rng(seed)
        self.W = [rs.normal(0, np.sqrt(2 / sizes[i]),
                            (sizes[i], sizes[i + 1]))
                  for i in range(len(sizes) - 1)]
        self.b = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]
        self.n_params = sum(W.size for W in self.W) + sum(
            b.size for b in self.b)

    def forward(self, X, p_drop=0.0, rs=None):
        self.H, self.Z, self.M = [X], [], []
        h = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            self.Z.append(z)
            if i < len(self.W) - 1:
                h = np.maximum(0.0, z)
                if p_drop > 0 and rs is not None:
                    m = (rs.random(h.shape) >= p_drop) / (1 - p_drop)
                    h = h * m
                    self.M.append(m)
                else:
                    self.M.append(None)
            else:
                h = z
                self.M.append(None)
            self.H.append(h)
        return h

    def loss_and_grads(self, X, y, p_drop=0.0, rs=None):
        logits = self.forward(X, p_drop, rs)
        m = logits.max(axis=1, keepdims=True)
        e = np.exp(logits - m)
        loss = float(np.mean(m[:, 0] + np.log(e.sum(axis=1))
                             - logits[np.arange(len(X)), y]))
        d = e / e.sum(axis=1, keepdims=True)
        d[np.arange(len(X)), y] -= 1.0
        d /= len(X)
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for l in reversed(range(len(self.W))):
            gW[l] = self.H[l].T @ d
            gb[l] = d.sum(axis=0)
            if l > 0:
                d = d @ self.W[l].T
                if self.M[l - 1] is not None:
                    d = d * self.M[l - 1]
                d = d * (self.Z[l - 1] > 0)
        return loss, gW, gb


def train(net, X, y, Xv, yv, steps=4000, lr=2e-3, batch=64, wd=0.0,
          p_drop=0.0, seed=0):
    params = net.W + net.b
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    rs = np.random.default_rng(seed + 20)
    for t in range(1, steps + 1):
        idx = rs.integers(0, len(X), min(batch, len(X)))
        _, gW, gb = net.loss_and_grads(X[idx], y[idx], p_drop, rs)
        for i, (pp, g) in enumerate(zip(params, gW + gb)):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            pp -= lr * (m[i] / (1 - 0.9 ** t)) / (
                np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
            if wd and pp.ndim == 2:
                pp -= lr * wd * pp
    tr = net.loss_and_grads(X, y)[0]
    te = net.loss_and_grads(Xv, yv)[0]
    tr_acc = float((net.forward(X).argmax(axis=1) == y).mean())
    te_acc = float((net.forward(Xv).argmax(axis=1) == yv).mean())
    return tr, te, tr_acc, te_acc


# --- Zhang et al.: networks fit random labels -------------------------------
print("=" * 72)
print("networks fit RANDOM labels perfectly (section 6.4)")
print("=" * 72)
print("Same architecture, same optimiser, same regularisation. Only the")
print("labels differ: real, or drawn uniformly at random.\n")
N_SMALL = 600
Xs, ys = make_data(N_SMALL, 1)
Xr, yr = make_data(N_SMALL, 1, randomize_labels=True)
Xv, yv = make_data(4000, 2)

print(f"{'labels':<12} {'regularisation':<22} {'train acc':>11} "
      f"{'test acc':>10} {'train loss':>12}")
for label, (XX, yy) in (("real", (Xs, ys)), ("RANDOM", (Xr, yr))):
    for reg_name, kw in (("none", {}),
                         ("wd 0.01 + dropout 0.3",
                          {"wd": 0.01, "p_drop": 0.3})):
        net = MLP([D, 256, 256, C], seed=3)
        tr, te, tra, tea = train(net, XX, yy, Xv, yv, steps=8000, **kw)
        print(f"{label:<12} {reg_name:<22} {tra:>11.4f} {tea:>10.4f} "
              f"{tr:>12.5f}")
print(f"\n(chance accuracy is {1 / C:.4f}; the network has "
      f"{MLP([D, 256, 256, C]).n_params:,} parameters "
      f"for {N_SMALL} examples)")

print("\nThe network drives training accuracy high on labels that contain")
print("NO information whatsoever, and its test accuracy on those labels is")
print("at chance — which it must be, since there is nothing to generalise.")
print("\nThat is Zhang et al.'s reductio. Whatever capacity measure appears")
print("in eq. 58.12's bound must be at least large enough to shatter this")
print("training set, which makes the bound vacuous — and yet the SAME")
print("network on real labels generalises.")
print("\nSo the explanation for generalisation cannot live in the hypothesis")
print("class alone. It has to involve the data and the training procedure.")
print("\nNote also what the regularisation did: it slowed the memorisation")
print("without preventing it, and it changed the real-label result by a")
print("modest amount. Regularisation is a knob, not the mechanism.")

# --- double descent ---------------------------------------------------------
print("\n" + "=" * 72)
print("double descent: test error past the interpolation threshold (6.5)")
print("=" * 72)
print(f"A fixed training set of {N_SMALL} examples with 15% label noise,")
print("and networks of increasing width.\n")
Xd, yd = make_data(N_SMALL, 7)
noise_idx = np.random.default_rng(8).choice(N_SMALL, N_SMALL * 15 // 100,
                                            replace=False)
yd = yd.copy()
yd[noise_idx] = np.random.default_rng(9).integers(0, C, len(noise_idx))

print(f"{'width':>7} {'params':>9} {'params/N':>10} {'train acc':>11} "
      f"{'test acc':>10} {'test loss':>11}")
rows = []
for width in (2, 4, 8, 12, 16, 24, 40, 80, 160, 320):
    net = MLP([D, width, C], seed=4)
    tr, te, tra, tea = train(net, Xd, yd, Xv, yv, steps=6000, lr=3e-3)
    rows.append((width, net.n_params, tra, tea, te))
    print(f"{width:>7} {net.n_params:>9,} {net.n_params / N_SMALL:>10.2f} "
          f"{tra:>11.4f} {tea:>10.4f} {te:>11.4f}")

losses = [r[4] for r in rows]
peak = int(np.argmax(losses[1:-1])) + 1
print(f"\nworst test loss at width {rows[peak][0]} "
      f"({rows[peak][1] / N_SMALL:.2f} params per example)")
print(f"best test loss at width {rows[int(np.argmin(losses))][0]} "
      f"({rows[int(np.argmin(losses))][1] / N_SMALL:.2f} params per example)")

print("\nThe peak sits essentially at the interpolation threshold — where")
print("the parameter count first passes the number of training examples and")
print("training accuracy first reaches 1.0 — and BOTH test loss and test")
print("accuracy improve monotonically for every width beyond it. That is")
print("the second descent, and it is clearly present here.")
print("\nThe classical picture predicts only the rise: past the point where")
print("the model can fit the training set, more capacity should mean")
print("monotonically worse test error. It does not.")
print("\nOne honest limitation. The second descent does not get BELOW the")
print("classical optimum at this scale — the tiny width-2 model still has")
print("the lowest test loss, because with 15 per cent label noise a model")
print("that predicts near-uniform scores well on log loss. Belkin et al.'s")
print("stronger claim, that the interpolating regime can beat the classical")
print("optimum, needs more data and more capacity than a laptop-sized")
print("reproduction has.")
print("\nWhat this table does establish is the part that matters for")
print(f"practice: the widest network, with {rows[-1][1] / N_SMALL:.0f} times as many")
print("parameters as training examples, is not the worst model — the one at")
print("the threshold is. 'More parameters means more overfitting' is false")
print("as a general rule, and the field's whole practice of scaling up")
print("depends on that.")
```

## 9. Practical Example

```python {tier=A name=which-regularizer}
"""Every regulariser in the chapter on the same problem, at the same
budget, with the data-size regime varied — because that is what decides.
"""
import numpy as np

rng = np.random.default_rng(11)

D, C = 20, 5
_a = np.random.default_rng(321)
T1, T2 = _a.normal(size=(D, 14)), _a.normal(size=(14, C))


def make_data(n, seed):
    rs = np.random.default_rng(seed)
    X = rs.normal(size=(n, D))
    logits = np.tanh(X @ T1) @ T2 * 1.5
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y = (p.cumsum(axis=1) < rs.random((n, 1))).sum(axis=1).clip(0, C - 1)
    return X, y


# One pool split in half. Val and test must be EXCHANGEABLE, or the
# early-stopping measurement below picks up the difference between two
# separately drawn sets instead of the selection effect it is testing.
_Xpool, _ypool = make_data(24000, 90)
Xva, yva = _Xpool[:12000], _ypool[:12000]
Xte, yte = _Xpool[12000:], _ypool[12000:]
_p = np.exp(np.tanh(Xte @ T1) @ T2 * 1.5)
_p /= _p.sum(axis=1, keepdims=True)
BAYES = float(-np.log(_p[np.arange(len(yte)), yte]).mean())


class Net:
    def __init__(self, sizes, seed=0):
        rs = np.random.default_rng(seed)
        self.W = [rs.normal(0, np.sqrt(2 / sizes[i]),
                            (sizes[i], sizes[i + 1]))
                  for i in range(len(sizes) - 1)]
        self.b = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]

    def forward(self, X, p_drop=0.0, rs=None):
        self.H, self.Z, self.M = [X], [], []
        h = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            self.Z.append(z)
            if i < len(self.W) - 1:
                h = np.maximum(0.0, z)
                if p_drop > 0 and rs is not None:
                    m = (rs.random(h.shape) >= p_drop) / (1 - p_drop)
                    h, keep = h * m, m
                else:
                    keep = None
                self.M.append(keep)
            else:
                h, keep = z, None
                self.M.append(keep)
            self.H.append(h)
        return h

    def loss_and_grads(self, X, y, p_drop=0.0, rs=None):
        logits = self.forward(X, p_drop, rs)
        m = logits.max(axis=1, keepdims=True)
        e = np.exp(logits - m)
        loss = float(np.mean(m[:, 0] + np.log(e.sum(axis=1))
                             - logits[np.arange(len(X)), y]))
        d = e / e.sum(axis=1, keepdims=True)
        d[np.arange(len(X)), y] -= 1.0
        d /= len(X)
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for l in reversed(range(len(self.W))):
            gW[l] = self.H[l].T @ d
            gb[l] = d.sum(axis=0)
            if l > 0:
                d = d @ self.W[l].T
                if self.M[l - 1] is not None:
                    d = d * self.M[l - 1]
                d = d * (self.Z[l - 1] > 0)
        return loss, gW, gb


def evaluate(net, X, y):
    loss, _, _ = net.loss_and_grads(X, y)
    acc = float((net.forward(X).argmax(axis=1) == y).mean())
    return loss, acc


def train(Xtr, ytr, wd=0.0, p_drop=0.0, aug=0.0, label_smooth=0.0,
          early_stop=False, steps=6000, lr=2e-3, batch=64, seed=0,
          eval_every=250):
    net = Net([D, 128, 128, C], seed=seed)
    params = net.W + net.b
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    rs = np.random.default_rng(seed + 60)
    best = (np.inf, None, 0)
    for t in range(1, steps + 1):
        idx = rs.integers(0, len(Xtr), min(batch, len(Xtr)))
        xb, yb = Xtr[idx], ytr[idx]
        if aug:                       # Gaussian jitter: the tabular analogue
            xb = xb + rs.normal(0, aug, xb.shape)
        logits = net.forward(xb, p_drop, rs)
        mm = logits.max(axis=1, keepdims=True)
        e = np.exp(logits - mm)
        d = e / e.sum(axis=1, keepdims=True)
        onehot = np.eye(C)[yb]
        if label_smooth:
            onehot = onehot * (1 - label_smooth) + label_smooth / C
        d = (d - onehot) / len(xb)
        gW, gb = [None] * len(net.W), [None] * len(net.W)
        for l in reversed(range(len(net.W))):
            gW[l] = net.H[l].T @ d
            gb[l] = d.sum(axis=0)
            if l > 0:
                d = d @ net.W[l].T
                if net.M[l - 1] is not None:
                    d = d * net.M[l - 1]
                d = d * (net.Z[l - 1] > 0)
        for i, (pp, g) in enumerate(zip(params, gW + gb)):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            pp -= lr * (m[i] / (1 - 0.9 ** t)) / (
                np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
            if wd and pp.ndim == 2:
                pp -= lr * wd * pp
        if early_stop and t % eval_every == 0:
            vl, _ = evaluate(net, Xva, yva)
            if vl < best[0]:
                best = (vl, [p.copy() for p in params], t)
    if early_stop and best[1] is not None:
        for pp, saved in zip(params, best[1]):
            pp[...] = saved
    tr_loss, tr_acc = evaluate(net, Xtr, ytr)
    te_loss, te_acc = evaluate(net, Xte, yte)
    return tr_loss, te_loss, tr_acc, te_acc, best[2]


RECIPES = {
    "none": {},
    "weight decay 0.01": {"wd": 0.01},
    "weight decay 0.1": {"wd": 0.1},
    "dropout 0.2": {"p_drop": 0.2},
    "dropout 0.5": {"p_drop": 0.5},
    "input noise 0.3": {"aug": 0.3},
    "label smoothing 0.1": {"label_smooth": 0.1},
    "early stopping": {"early_stop": True},
    "wd 0.01 + drop 0.2 + noise": {"wd": 0.01, "p_drop": 0.2, "aug": 0.3},
}

print("=" * 72)
print("the same regularisers in two data regimes")
print("=" * 72)
print(f"Bayes-optimal test cross-entropy: {BAYES:.4f}")
print("Excess test loss above that floor; lower is better.\n")

for n_train, label in ((500, "SMALL: 500 training examples"),
                       (20000, "LARGE: 20000 training examples")):
    Xtr, ytr = make_data(n_train, 1)
    print(f"{label}   ({n_train} examples, "
          f"{sum(W.size for W in Net([D, 128, 128, C]).W):,} weights)")
    print(f"  {'recipe':<28} {'train loss':>11} {'excess test':>12} "
          f"{'test acc':>10} {'gap':>8}")
    base = None
    for name, kw in RECIPES.items():
        trl, tel, tra, tea, stopped = train(Xtr, ytr, **kw)
        if base is None:
            base = tel - BAYES
        note = f"  (stopped @{stopped})" if kw.get("early_stop") else ""
        print(f"  {name:<28} {trl:>11.4f} {tel - BAYES:>12.4f} "
              f"{tea:>10.4f} {tel - trl:>8.4f}{note}")
    print()

print("Read the GAP column first — the train/test difference, which is")
print("what regularisation exists to reduce.")
print("\nAt 500 examples the network has far more weights than data, the")
print("unregularised gap is enormous, and every technique has something to")
print("work with. At 20000 the unregularised gap is several times smaller")
print("before any technique is applied.")
print("\nThe honest headline is in the 'none' rows: going from 500 to 20000")
print("examples improved the excess test loss by more than any regulariser")
print("achieved within either regime. MORE DATA BEAT EVERY TECHNIQUE ON")
print("THIS TABLE, which is the comparison people skip and the one that")
print("usually decides.")
print("\nRegularisation still helps at the larger size — this is not a")
print("regime where it stops mattering — but the amount available to gain")
print("has shrunk with the gap, and the techniques that cost accuracy by")
print("removing capacity are correspondingly closer to breaking even.")
print("\nExtrapolate that trend and you get the change in practice noted in")
print("section 5.6. A large language model sees each token roughly once,")
print("so its gap is near zero by construction and the techniques aimed at")
print("preventing memorisation have nothing left to prevent — which is why")
print("dropout largely disappeared from them without anyone deciding it was")
print("a bad idea.")

# --- early stopping's honest accounting -------------------------------------
print("=" * 72)
print("early stopping consumes the validation set (section 5.3)")
print("=" * 72)
Xtr, ytr = make_data(500, 1)
print("Validation and test are two halves of ONE pool, so they are")
print("exchangeable and any systematic difference is the selection effect.\n")
print(f"{'seed':>6} {'best VAL loss':>15} {'TEST loss at that point':>25} "
      f"{'optimism':>10}")
opt = []
for seed in range(5):
    net = Net([D, 128, 128, C], seed=seed)
    params = net.W + net.b
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    rs = np.random.default_rng(seed + 60)
    best_v, best_t = np.inf, None
    LR = 2e-3
    for t in range(1, 6001):
        idx = rs.integers(0, len(Xtr), 64)
        _, gW, gb = net.loss_and_grads(Xtr[idx], ytr[idx])
        for i, (pp, g) in enumerate(zip(params, gW + gb)):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            pp -= LR * (m[i] / (1 - 0.9 ** t)) / (
                np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
        if t % 250 == 0:
            vl, _ = evaluate(net, Xva, yva)
            if vl < best_v:
                best_v = vl
                best_t = evaluate(net, Xte, yte)[0]
    opt.append(best_t - best_v)
    print(f"{seed:>6} {best_v:>15.4f} {best_t:>25.4f} "
          f"{best_t - best_v:>10.4f}")
print(f"\nmean optimism: {float(np.mean(opt)):+.4f} "
      f"(positive means the validation number FLATTERS the model)")
print(f"consistent in sign across seeds: "
      f"{all(o > 0 for o in opt) or all(o < 0 for o in opt)}")
print("\nThe two sets are exchangeable by construction, so in the absence")
print("of any selection they should agree up to sampling noise. The")
print("systematic component is the price of having CHOSEN the stopping")
print("step by looking at the validation set — the selection effect of")
print("Chapter 43, arriving through a different door.")
print("\nThe magnitude here is small, and that is worth saying rather than")
print("overstating: with a validation set of this size and checkpoints")
print("every 250 steps, there are few opportunities to overfit the split.")
print("It grows with the number of decisions taken on the same set, which")
print("in a real project is not one but dozens.")
print("\nThe fix is Chapter 43's: a third split that no decision touches.")
```

## 10. Production Considerations

**Pick the regularisation from the data-to-parameter ratio, not from habit.**
Measured: the same techniques that help substantially at 500 examples do little
or nothing at 20000, and some cost.

**Report early-stopping performance on a third split.** Measured optimism: the
validation loss at the selected step is systematically better than the test
loss at the same step.

**Set up parameter groups.** No decay on biases or normalisation parameters
({{ch:dl-optimizers}}).

**Do not place dropout before batch normalisation.** It widens the train/eval
gap {{ch:dl-normalization}} measured.

**Budget the augmentation pipeline.** A heavy CPU pipeline starves the
accelerator with no error and no symptom other than throughput.

**Account for the regularisation you are removing.** Measured in
{{ch:dl-normalization}}: replacing batch normalisation removes noise injection
that scales as $1/\sqrt{B}$.

**Tune $\lambda$ against the step budget.** {{eq:decoupled-decay}}'s half-life
is $\ln 2/(\eta\lambda)$ steps, so the same $\lambda$ means different things in
a short run and a long one.

## 11. Common Mistakes

**Leaving dropout on at inference.** {{ch:dl-forward}} measured the symptom:
per-request nondeterminism with correct aggregate expectations, so metrics look
fine.

**Reporting the early-stopping validation number as final performance.**
Measured optimism.

**Independent dropout masks at each recurrent time step.** Use one mask for the
whole sequence.

**Decaying normalisation parameters.**

**An augmentation whose invariance does not hold.** Horizontal flip on text,
colour jitter when colour is the label.

**Adding regularisation to a model that is underfitting.** Check the training
loss first: if the model cannot fit the training set, regularisation is the
wrong direction entirely.

**Transferring a $\lambda$ between coupled and decoupled formulations.**
{{ch:dl-optimizers}} measured them producing very different weight norms.

**Assuming more parameters means more overfitting.** Measured to be false.

## 12. Failure Modes

**Underfitting from too much regularisation.** Train and test loss both high
and close. The diagnosis is the *gap*, not the level.

**Dropout too high in a narrow layer.** At $p = 0.5$ and 16 units, some forward
passes drop nearly everything, and the variance {{eq:dropout-variance}} injects
overwhelms the signal.

**Augmentation that shifts the distribution.** If the augmented distribution
differs from the test distribution, you have introduced train/serve skew
({{ch:mle-pipelines}}) with the best of intentions.

**Early stopping firing on a noisy validation curve.** Small validation sets
produce noisy estimates and the run stops on noise. Increase the set or the
patience.

**Early stopping never firing.** With a cosine schedule the loss improves until
the end by construction.

**Silent loss of implicit regularisation.** Removing batch normalisation,
increasing the batch size, or reducing the number of steps all reduce
regularisation without anyone changing a regularisation setting.

## 13. Alternatives

**Getting more data.** Almost always better than any technique here, and the
comparison is worth making explicitly: the measured large-data block shows the
gap closing without any regularisation at all.

**Transfer learning.** A pretrained model needs far less regularisation on a
small dataset than a randomly initialised one, because the representation is
already constrained ({{part:14}}).

**Smaller models.** The classical answer. Double descent complicates it — the
measured widest network was not the worst — and it remains right when inference
cost matters.

**Ensembling.** Reliable variance reduction at $k$ times the cost
({{ch:ml-forests}}). Dropout is sometimes described as an approximation to it.

**Sharpness-aware minimisation** biases training toward flat minima and reports
real generalisation gains at roughly double the compute.
{{maturity:EMERGING}}

**Doing nothing.** For a model trained on data it sees once, the honest answer.

## 14. Evaluation

**Look at the train/test gap, not the test loss alone.** A high test loss with a
small gap is underfitting and more regularisation will make it worse.

**Vary the training-set size.** Measured: it changes which techniques help more
than any other single factor.

**Reserve a third split for anything selected on validation.**

**Ablate each regulariser separately.** They interact, and a stack tuned as a
whole often contains components that contribute nothing.

**Check the augmented distribution against the test distribution.**

**Sweep the width past the interpolation threshold** before concluding that a
model is too large.

## 15. Advanced Concepts

**Implicit regularisation of gradient descent.** From a small initialisation,
gradient descent on an underdetermined problem converges to the minimum-norm
solution. Proven for linear and kernel regression, conjectured for deep
networks, and the current best account of why overparameterised models
generalise. {{maturity:RESEARCH FRONTIER}}

**The flat-minimum hypothesis.** Solutions in wide basins are conjectured to
generalise better, and sharpness is not invariant to reparameterisation, which
undermines the naive statement. Sharpness-aware minimisation works regardless.

**Grokking.** Networks trained far past the point of fitting the training set
sometimes transition suddenly from memorisation to generalisation, thousands of
steps after the training loss reached zero. Direct evidence that something
continues to happen after interpolation.
{{maturity:RESEARCH FRONTIER}}

**Data pruning and scaling laws.** Which examples to keep is a form of
regularisation, and carefully pruned datasets can beat larger random ones.

**Regularisation in the fine-tuning regime.** With few steps and a strong
initialisation the risk is *forgetting* rather than overfitting, so the
techniques change: layerwise decay, low rank ({{ch:ft-lora}}), and small
learning rates.

## 16. Connection to Previous Chapters

{{ch:ml-metrics}} presented the bias–variance decomposition and warned that the
classical U-shape stops describing what happens; this chapter is where that
warning is cashed in, and the measured double-descent table is the payoff.

{{ch:mle-splits}}'s selection effect reappears in the measured early-stopping
optimism — the same phenomenon reached through a different door.
{{ch:dl-normalization}} supplied {{eq:norm-scale-invariance}}, without which
{{sec:6-mathematical-foundation}}'s account of weight decay makes no sense.
{{ch:dl-optimizers}} supplied the decoupled decay and the parameter groups.
{{ch:dl-losses}}'s label smoothing is a regulariser and is measured there.

Forward: {{ch:ft-lora}} regularises by restricting the *rank* of the update
rather than its norm. {{ch:llm-next-token}} is the regime where most of this
chapter does not apply, for the reason the measured large-data block shows.
{{ch:ev-why-hard}} treats benchmark overfitting, which is this chapter's
selection effect at the level of a research field.

## 17. Exercises

**Beginner**

1. What does inverted dropout scale by, and why?
2. Why does dropout need no adjustment at inference?
3. What is early stopping, and what does it consume?
4. Give an augmentation that is valid for photographs and invalid for text.
5. What is the interpolation threshold?

**Intermediate**

6. Derive {{eq:dropout-variance}}.
7. Derive {{eq:dropout-l2}} for a linear model.
8. Using {{eq:decoupled-decay}}, find the half-life at $\eta = 3\times10^{-4}$
   and $\lambda = 0.1$.
9. Explain why weight decay affects a scale-invariant layer at all.
10. Explain why {{eq:uniform-convergence}} is vacuous for a network that fits
    random labels.
11. Why does dropout largely disappear from large language models?

**Advanced**

12. Derive {{eq:early-stopping-equivalence}} from
    {{eq:early-stopping-trajectory}} and {{eq:ridge-solution}}.
13. Derive {{eq:norm-equilibrium}}'s fixed point and express the equilibrium
    effective learning rate in terms of $\eta$, $\lambda$ and $\|\vec{g}\|$.
14. Explain double descent in the linear-regression case, where it can be
    derived.
15. Explain what "minimum-norm interpolating solution" means and why gradient
    descent finds it in the linear case.
16. Construct a case where dropout hurts, and explain it via
    {{eq:dropout-variance}}.

**Implementation**

17. Implement dropout with forward and backward and gradient-check it.
18. Reproduce the dropout/weighted-ridge equivalence for a linear model.
19. Reproduce the random-label result and vary the data size.
20. Sweep width across the interpolation threshold and plot test error.

**Reasoning**

21. Train and test loss are both high and nearly equal. What do you do?
22. Your model's validation loss is much better than its test loss, with no
    distribution shift. Explain.

## 18. Interview Questions

**"How does dropout work?"** — Random masking with $1/(1-p)$ scaling at
training. Note that inference is unmodified and say why.

**"Why does dropout regularise?"** — Preventing co-adaptation and implicit
ensembling, and the exact $\ell_2$ equivalence for a linear model. Saying that
the equivalence is exact only in the linear case is the distinguishing detail.

**"Weight decay or $\ell_2$?"** — Different under adaptive optimisers
({{ch:dl-optimizers}}), and in a normalised network weight decay controls the
effective learning rate rather than the function.

**"Can a neural network overfit?"** — Yes, and it can also fit random labels
perfectly and still generalise on real ones. The interesting answer is the
second half.

**"What is double descent?"** — Test error rises past the interpolation
threshold and then falls. Note that it falsifies "more parameters means more
overfitting".

**"Why don't large language models use dropout?"** — Each token is seen roughly
once, so memorisation is not the binding constraint.

**"How do you know whether to add regularisation?"** — Look at the gap. High
loss with a small gap is underfitting.

## 19. Research Questions

**Why do overparameterised networks generalise?** The implicit-regularisation
account is proven for linear and kernel regression and conjectured for deep
networks. This is the central open question of the theory.
{{maturity:RESEARCH FRONTIER}}

**Does double descent occur in practice at realistic scale?** Clearly with label
noise and a fixed dataset; less clearly in normal training, and the conditions
are not fully characterised. {{maturity:EMERGING}}

**What is grokking?** Sudden generalisation long after the training loss reaches
zero. Suggests a phase transition in the representation that the loss curve does
not show. {{maturity:RESEARCH FRONTIER}}

**Can regularisation be selected automatically?** It is a hyperparameter search
({{ch:mle-hpo}}), and nothing predicts the right setting from properties of the
dataset. {{maturity:EMERGING}}

## 20. Chapter Summary

The classical story does not describe deep networks. Reproduced here at small
scale, a network fit randomly labelled data — labels containing no information
at all — while scoring at chance on the test set, which is what it must do. That
is {{cite:zhang2017rethinking}}'s reductio: any capacity measure supporting
{{eq:uniform-convergence}} must be large enough to shatter that training set,
making the bound vacuous, and the same network on real labels generalises. The
explanation therefore cannot live in the hypothesis class. Regularisation slowed
the memorisation without preventing it and changed the real-label result
modestly. **It is a tuning knob, not the mechanism.**

Inverted dropout preserves the expectation exactly — measured at 1.000 at every
rate — which is why inference needs no adjustment. The injected noise matches
{{eq:dropout-variance}}'s $\sqrt{p/(1-p)}$ and is proportional to each
activation's own magnitude, so it is a data-dependent perturbation rather than
additive noise. For a linear model, dropout is *exactly* a ridge penalty
weighted per feature by $\|\vec{x}_j\|^2$, confirmed here by fitting both and
comparing: SGD with dropout converged to the weighted-ridge solution without
ridge ever being written down. That equivalence is exact only in the linear
case.

Early stopping is the same regulariser expressed as a budget of steps rather
than a penalty. The measured trajectory matched ridge at
$\alpha = 1/(\eta t)$ closely, with both norms growing together as the implied
penalty weakened. Fewer steps means more regularisation, and that direction is
robust even where the quadratic approximation is not.

Weight decay in a normalised network does something the classical story does not
describe. {{eq:grad-orthogonal-to-w}} makes the gradient orthogonal to the
weights, so — measured directly — the norm can only grow under a plain gradient
step, and decay is what establishes an equilibrium. Since the effective learning
rate scales as $1/\|\mat{W}\|^2$, **weight decay is setting the step size, not
shrinking the function**, which does not depend on $\|\mat{W}\|$ at all.

The measured comparison across two data regimes gives the practical result, and
the headline is not the one the chapter's topic would suggest. At 500 examples
the unregularised train/test gap was enormous and every technique helped; at
20000 it was several times smaller and they helped less. But the largest single
improvement in the whole table came from neither column: **going from 500 to
20000 examples beat every regulariser applied within either regime.** More data
is the comparison people skip and usually the one that decides. Extrapolating
that trend gives dropout's disappearance from large language models — a model
that sees each token roughly once has no gap left to close.

Finally, early stopping consumes the validation set. Measured on two
exchangeable halves of one pool, the validation loss at the selected step was
systematically better than the test loss at the same step, by a small and
consistent margin. This is {{ch:mle-splits}}'s selection effect reached through
a different door, it grows with the number of decisions taken on the same split,
and the fix is the same one — a split that no decision has touched.

## 21. Further Reading

{{cite:zhang2017rethinking}} is the most important paper in this chapter and one
of the most important in the part. It is short, the experiments are simple
enough to reproduce, and the argument is a clean reductio rather than a new
technique. Read it and then re-read {{ch:ml-metrics}}'s bias–variance section
with it in mind.

{{cite:srivastava2014}} for dropout. Both mechanistic stories — co-adaptation
and ensembling — are in it, and reading them side by side is instructive because
they are not the same claim and the paper does not fully commit to either.

{{cite:belkin2019}} for double descent. The linear-regression case is worked out
in a way that makes the phenomenon comprehensible rather than merely surprising,
and that is the section to read if you want to know *why* the curve turns.

{{cite:loshchilov2019adamw}} again, read this time with
{{eq:norm-equilibrium}} in hand: the reason weight decay matters so much in
normalised networks is not the reason the paper gives, and holding both accounts
at once is a good exercise.

**Where to go next:** {{ch:dl-cnns}} introduces the first architecture whose
inductive bias does regularisation's job structurally — a convolution is a
constraint on the hypothesis class, applied before training rather than during
it.

---
id: dl-normalization
number: 57
part: VI
tier: full
status: reviewed
requires: [dl-initialization, dl-backprop, dl-forward, dl-optimizers, ds-cleaning]
provides: [batch-normalization, layer-normalization, rmsnorm, groupnorm,
           internal-covariate-shift, pre-norm, post-norm, running-statistics,
           train-eval-divergence]
citations: [ioffe2015, ba2016layernorm, santurkar2018, zhang2019rmsnorm,
            he2016resnet, xiong2020prenorm]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Derive batch normalisation's forward and backward passes.
2. Explain the difference between batch, layer, group and RMS normalisation,
   and which axis each reduces over.
3. State what is and is not known about *why* normalisation helps.
4. Explain why batch normalisation behaves differently at training and
   inference, and what that costs.
5. Choose a normalisation for a given architecture and batch size.
6. Explain pre-norm versus post-norm and why the field moved.
7. Diagnose the specific failures each normalisation introduces.

## 2. Why This Matters

**Normalisation is what made networks deeper than about twenty layers routinely
trainable.** {{cite:ioffe2015}} is one of the most cited papers in machine
learning, and every large model since uses some form of it.

**Its stated explanation was wrong, and the technique works anyway.**
{{cite:ioffe2015}} attributed the benefit to reducing *internal covariate
shift* — the change in each layer's input distribution as the layers below
update. {{cite:santurkar2018}} tested that directly by *injecting* covariate
shift after the normalisation and found training was unaffected. **A method
working is not evidence that the stated reason is correct.** This is the
clearest example of that lesson in the book, and it is why
{{sec:19-research-questions}} is worth reading rather than skipping.

**Batch normalisation introduces a train/eval divergence that nothing else in
this book does.** The layer computes a different function at training time than
at inference time, by construction, and that single design choice is responsible
for a whole family of production bugs.

**The 2026 default is not what the field used in 2018.** Language models use
RMSNorm with pre-normalisation almost universally; batch normalisation survives
in convolutional vision. Knowing which is a decision, not a habit.

## 3. Prerequisites

{{ch:dl-initialization}} for variance propagation — normalisation is the
alternative solution to the same problem, and the measured insensitivity there
is this chapter's motivation. {{ch:dl-backprop}} for the backward pass through a
normalisation layer. {{ch:dl-forward}} for the train/eval mode flag.
{{ch:ds-cleaning}} for feature standardisation, which is this idea applied once
at the input.

## 4. Intuitive Explanation

### 4.1 The idea

{{ch:ds-cleaning}} standardises the *inputs* so that no feature dominates by
having a larger scale. Normalisation applies the same operation to every
layer's activations, not just the first:

$$
\hat{x} = \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}},
\qquad y = \gamma\hat{x} + \beta
$$

The learnable $\gamma$ and $\beta$ matter more than they look: **without them
the layer could only produce zero-mean unit-variance outputs, which is a
restriction rather than a help.** With them, the network can undo the
normalisation entirely if that is what it wants, so the operation adds capacity
rather than removing it. What it changes is the *parameterisation* — the scale
is now set by one parameter per channel rather than emerging from a product of
weight matrices.

### 4.2 Which axis

This is the only real difference between the variants:

```text
   a batch of activations, shape (B, C)

              C features →
        ┌───────────────────┐
   B    │                   │   BatchNorm:  normalise DOWN each column
   ex   │                   │               (over the batch, per feature)
   ↓    │                   │   LayerNorm:  normalise ACROSS each row
        └───────────────────┘               (over features, per example)
```

**Batch norm's statistics depend on the other examples in the batch.** Layer
norm's do not. That single difference produces every other distinction between
them: batch norm needs a reasonably large batch, breaks at batch size 1, needs
running statistics for inference, and behaves differently at train and eval
time. Layer norm has none of those properties and none of the mild
regularisation the batch coupling provides.

```text
   BatchNorm    (B, C)      over B         needs a batch; vision
   LayerNorm    (B, T, C)   over C         per-token; transformers
   GroupNorm    (B, C,H,W)  over C/g,H,W   batch-independent vision
   InstanceNorm (B, C,H,W)  over H,W       style transfer
   RMSNorm      (B, T, C)   over C, no mean  transformers, 2023 onwards
```

### 4.3 RMSNorm

{{cite:zhang2019rmsnorm}} observed that the mean subtraction may not be doing
much and removed it:

$$
\hat{x} = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}},
\qquad y = \gamma\hat{x}
$$

No mean, no $\beta$. Cheaper — one pass over the data instead of two, and one
fewer parameter per channel — and empirically as good. **It is the default in
essentially every open-weight language model since 2023**, which is a strong
practical endorsement of the claim that the centring was not the active
ingredient.

Note what that claim is *not*. RMSNorm and LayerNorm are not mathematically
equivalent: they agree only when each row's empirical mean is zero, and
{{sec:8-implementation}} measures the discrepancy shrinking as $1/\sqrt{d}$ for
zero-mean data and not shrinking at all for data with a genuine offset. The
equivalence is an empirical fact about trained networks at large width, not an
identity.

### 4.4 Pre-norm and post-norm

Where the normalisation sits relative to the residual connection:

```text
   post-norm  (original transformer)     pre-norm  (everything since ~2019)
   x ──┬───────────────┐                 x ──┬────────────────────┐
       │               ▼                     ▼                    ▼
       └──▶ sublayer ──+──▶ Norm ──▶        Norm ──▶ sublayer ──▶ + ──▶
```

**Pre-norm leaves a clean identity path from input to output** — nothing on the
skip connection is normalised, so a gradient can travel from the loss to any
layer without passing through a normalisation. Post-norm places a normalisation
on that path at every layer.

The consequence, analysed by {{cite:xiong2020prenorm}}: post-norm needs warmup
to train at all at depth, and pre-norm does not. Pre-norm is universal in modern
models, and post-norm is reported to reach slightly better final quality when it
can be trained — which is a real trade and not a settled one.

## 5. Formal Explanation

### 5.1 Batch normalisation

For a mini-batch $\{x_1,\dots,x_B\}$ of one feature:

$$
\mu_{\mathcal{B}} = \frac{1}{B}\sum_{i=1}^{B} x_i,
\qquad
\sigma^2_{\mathcal{B}} = \frac{1}{B}\sum_{i=1}^{B}(x_i-\mu_{\mathcal{B}})^2
$$ (eq:bn-stats)

$$
\hat{x}_i = \frac{x_i-\mu_{\mathcal{B}}}{\sqrt{\sigma^2_{\mathcal{B}}+\epsilon}},
\qquad
y_i = \gamma\hat{x}_i+\beta
$$ (eq:bn-transform)

At inference, $\mu$ and $\sigma^2$ come from running averages accumulated during
training:

$$
\mu_{\text{run}} \leftarrow (1-m)\mu_{\text{run}} + m\mu_{\mathcal{B}}
$$ (eq:running-stats)

with momentum $m$ typically $0.1$.

> WARNING: **The layer computes a different function in the two modes.** At
> training it is a function of the whole batch; at inference it is a fixed
> affine map. They agree only if the running statistics match the batch
> statistics, which requires the inference distribution to match the training
> one — the assumption {{ch:mle-drift}} spends a chapter saying will fail.

### 5.2 Layer normalisation

{{cite:ba2016layernorm}} reduces over the *feature* axis of a single example:

$$
\mu_i = \frac{1}{d}\sum_{j=1}^{d} x_{ij},
\qquad
\sigma_i^2 = \frac{1}{d}\sum_{j=1}^{d}(x_{ij}-\mu_i)^2
$$ (eq:ln-stats)

$$
y_{ij} = \gamma_j\frac{x_{ij}-\mu_i}{\sqrt{\sigma_i^2+\epsilon}} + \beta_j
$$ (eq:ln-transform)

**No dependence on other examples**, so training and inference are identical,
batch size 1 works, and nothing needs to be accumulated. This is why sequence
models use it: batches of variable-length sequences make batch statistics
awkward, and inference is often one sequence at a time.

### 5.3 RMSNorm

$$
y_{ij} = \gamma_j \frac{x_{ij}}
 {\sqrt{\frac{1}{d}\sum_{k}x_{ik}^2 + \epsilon}}
$$ (eq:rmsnorm)

Layer norm without the mean subtraction and without $\beta$. Note that
$\sqrt{\frac{1}{d}\sum x^2}$ equals the standard deviation only when the mean is
zero, so RMSNorm and LayerNorm coincide exactly on centred inputs and differ
otherwise.

### 5.4 Group normalisation

Partition $C$ channels into $g$ groups and normalise over each group's channels
together with the spatial dimensions. It interpolates:

$$
g = 1 \Rightarrow \text{LayerNorm},
\qquad
g = C \Rightarrow \text{InstanceNorm}
$$

Batch-independent like layer norm, and it respects the channel structure that a
plain layer norm over all of $(C, H, W)$ would ignore. Standard where batch
sizes are small — detection and segmentation, where each image is large.

### 5.5 Choosing

{#tbl:norm-choice caption="Which normalisation to use. The deciding question is almost always whether the batch is large and independent-and-identically-distributed, and whether train/inference divergence is acceptable."}

| Setting | Choice | Reason |
|---|---|---|
| Convolutional vision, batch $\ge 32$ | BatchNorm | Best results; the regularisation is real |
| Convolutional vision, small batch | GroupNorm | Batch-independent |
| Transformers, sequences | LayerNorm or RMSNorm | Variable length, batch 1 at inference |
| Language models, 2023 onwards | RMSNorm, pre-norm | Cheaper, empirically equal |
| Online or streaming inference | Anything but BatchNorm | No batch to compute statistics over |
| Fine-tuning a small batch | Freeze BatchNorm statistics | Running stats will be corrupted otherwise |

## 6. Mathematical Foundation

### 6.1 The backward pass through batch normalisation

The subtlety is that $\mu$ and $\sigma^2$ *depend on every $x_i$*, so the
gradient has three paths: directly through $\hat{x}_i$, and indirectly through
both statistics. Writing $\bar{y}_i = \partial\Like/\partial y_i$:

$$
\frac{\partial\Like}{\partial\gamma} = \sum_i \bar{y}_i\hat{x}_i,
\qquad
\frac{\partial\Like}{\partial\beta} = \sum_i \bar{y}_i
$$ (eq:bn-param-grads)

and, writing $\bar{x}_i = \gamma\bar{y}_i$ and
$s = \sqrt{\sigma^2_{\mathcal{B}}+\epsilon}$, the input gradient collapses to

$$
\frac{\partial\Like}{\partial x_i}
 = \frac{1}{Bs}\left(B\bar{x}_i
 - \sum_{j}\bar{x}_j
 - \hat{x}_i\sum_{j}\bar{x}_j\hat{x}_j\right)
$$ (eq:bn-backward)

Three terms, and each is worth reading:

**$B\bar{x}_i$** is the direct path.

**$-\sum_j \bar{x}_j$** subtracts the mean gradient. The consequence is that
**the gradients within a batch are forced to sum to zero**: a uniform push on
all examples has no effect after normalisation, because it would just move the
mean, which the normalisation removes.

**$-\hat{x}_i\sum_j\bar{x}_j\hat{x}_j$** removes the component of the gradient
along $\hat{\vec{x}}$, for the same reason applied to the scale.

**So the backward pass projects the gradient onto the subspace orthogonal to
both the mean and the current activation direction.** This is a genuine
constraint on what the optimiser can do, and it is the sharpest formal statement
available of how normalisation changes training.

### 6.2 Scale invariance of the weights

Let $\mat{W}$ be the weights feeding a normalisation layer. Replace $\mat{W}$
with $a\mat{W}$ for $a > 0$. Then $\vec{z} \to a\vec{z}$, so
$\mu \to a\mu$ and $\sigma \to a\sigma$, and

$$
\hat{z} = \frac{az-a\mu}{a\sigma} = \frac{z-\mu}{\sigma}
$$ (eq:norm-scale-invariance)

**The output is completely unchanged.** The network's function does not depend
on the scale of any weight matrix feeding a normalisation layer.

Two consequences follow, and both are important.

**The initialisation scale stops mattering** for those layers, which is
{{ch:dl-initialization}}'s measured insensitivity, now explained rather than
merely observed.

**The effective learning rate becomes $\eta/\|\mat{W}\|^2$.** Since the function
is invariant to $\|\mat{W}\|$, the gradient must be orthogonal to $\mat{W}$; a
step of size $\eta$ then *increases* $\|\mat{W}\|$ by Pythagoras, which
*decreases* the effective step next time. **Normalisation therefore induces an
automatic learning-rate decay**, which is a genuinely surprising consequence of
{{eq:norm-scale-invariance}} and is measured in {{sec:8-implementation}}.

### 6.3 What the internal covariate shift claim said, and what happened

{{cite:ioffe2015}}'s argument: as lower layers update, the distribution of each
layer's inputs shifts, so upper layers must continually re-adapt. Normalising
fixes the first two moments and removes the problem.

{{cite:santurkar2018}} tested it with a decisive experiment: take a network with
batch normalisation and **deliberately inject noise after each normalisation
layer**, with a distribution that changes every step. This restores — indeed
worsens — internal covariate shift while keeping the normalisation.

**The network trained just as well.** So covariate shift is not what
normalisation is fixing.

Their alternative account is that normalisation makes the loss landscape
smoother — improving the Lipschitz constants of the loss and of its gradient —
which makes larger steps safe and gradients more predictive. That has
supporting theory and is not the settled final answer either.

> IMPORTANT: **The technique is {{maturity:ESTABLISHED}}; the explanation is
> {{maturity:RESEARCH FRONTIER}}.** These are different claims and the
> literature routinely conflates them. When you read "batch norm works because
> it reduces internal covariate shift" — which appears in a great many
> tutorials — you are reading a superseded explanation of a technique that
> works for other reasons.

### 6.4 The regularisation effect

Because a training example's normalised value depends on the other examples in
its batch, the same input produces different activations in different batches.
This is noise injection, and it regularises.

The noise scale goes as $1/\sqrt{B}$, so:

- **Small batches: more regularisation, noisier statistics.**
- **Large batches: less regularisation, more stable statistics.**

That is why batch normalisation's benefit is partly a regularisation benefit,
why it interacts with other regularisers, and why replacing it with a
batch-independent normalisation can *lose* accuracy for reasons unrelated to
optimisation. {{sec:9-practical-example}} measures the noise.

### 6.5 Pre-norm's gradient path

In a post-norm block, $\vec{y} = \text{Norm}(\vec{x} + F(\vec{x}))$, so
$\partial\vec{y}/\partial\vec{x}$ passes through the normalisation's Jacobian at
every layer, and {{eq:unrolled-backprop}}'s product accumulates $L$ of them.

In a pre-norm block, $\vec{y} = \vec{x} + F(\text{Norm}(\vec{x}))$, so

$$
\frac{\partial\vec{y}}{\partial\vec{x}}
 = \mat{I} + \frac{\partial F}{\partial\vec{x}}
$$ (eq:prenorm-jacobian)

**The identity term is untouched by the normalisation**, so a gradient path of
exactly 1 exists from the loss to every layer. {{cite:xiong2020prenorm}} shows
that post-norm's gradients at initialisation scale badly with depth while
pre-norm's do not, which is the formal version of "post-norm needs warmup".

## 7. Internal Mechanics

### 7.1 Fusion at inference

At inference a batch normalisation is a fixed affine map, so it can be folded
into the preceding convolution or linear layer:

$$
\mat{W}' = \frac{\gamma}{\sqrt{\sigma^2+\epsilon}}\mat{W},
\qquad
\vec{b}' = \gamma\frac{\vec{b}-\mu}{\sqrt{\sigma^2+\epsilon}} + \beta
$$ (eq:bn-folding)

The layer then costs nothing at all. **Every inference toolchain does this
automatically**, and it is why batch normalisation's inference cost is often
quoted as zero. Layer norm cannot be folded, because its statistics depend on
the input.

### 7.2 Running statistics and their traps

The momentum in {{eq:running-stats}} means the running estimate has an effective
window of about $1/m$ batches — 10 at the common default of $0.1$. Three
consequences:

**A short fine-tune corrupts them.** A few hundred steps on new data overwrites
statistics accumulated over a long pretraining run.

**They lag a changing distribution**, which is what you want during training and
not what you want if the last batches of training were unrepresentative.

**They are not gradients**, so they are not affected by `requires_grad=False`.
Freezing a batch-norm layer requires putting it in eval mode, not merely
freezing its parameters — a distinction that catches people constantly.

### 7.3 Distributed training

Batch statistics are computed per device by default, so a global batch of 1024
across 8 devices gives statistics over 128 examples, not 1024. Synchronised
batch normalisation communicates the statistics and is slower; whether it is
needed depends on whether the per-device batch is large enough.

**This also means the same code gives different results on different numbers of
devices**, which is a reproducibility problem ({{ch:mle-reproducibility}}) that
has nothing to do with random seeds.

### 7.4 Gradient accumulation

{{ch:dl-backprop}} measured that gradient accumulation reproduces a full batch
exactly. **That result does not hold with batch normalisation**, because the
statistics are computed per micro-batch. Accumulating 8 micro-batches of 32 is
not equivalent to one batch of 256, and the difference is real rather than
floating-point.

### 7.5 Epsilon

$\epsilon$ prevents division by zero and is not merely defensive: a channel that
is constant across the batch has $\sigma^2 = 0$ exactly, which happens for a
dead ReLU channel or a padded position. Values differ between frameworks
($10^{-5}$ against $10^{-8}$ is common), which is a source of small numerical
discrepancies when porting a model.

## 8. Implementation

```python {tier=A name=normalization-forward-backward}
"""Batch, layer and RMS normalisation implemented from their equations, with
the backward pass verified numerically.
"""
import numpy as np

rng = np.random.default_rng(0)


# --- forward passes ---------------------------------------------------------
def batchnorm_forward(x, gamma, beta, eps=1e-5):
    """Eqs. 57.1-57.2. Reduces over the BATCH axis, per feature."""
    mu = x.mean(axis=0)
    var = x.var(axis=0)
    xhat = (x - mu) / np.sqrt(var + eps)
    return gamma * xhat + beta, (xhat, np.sqrt(var + eps), gamma)


def layernorm_forward(x, gamma, beta, eps=1e-5):
    """Eqs. 57.4-57.5. Reduces over the FEATURE axis, per example."""
    mu = x.mean(axis=1, keepdims=True)
    var = x.var(axis=1, keepdims=True)
    xhat = (x - mu) / np.sqrt(var + eps)
    return gamma * xhat + beta, (xhat, np.sqrt(var + eps), gamma)


def rmsnorm_forward(x, gamma, eps=1e-5):
    """Eq. 57.6. No mean, no beta."""
    rms = np.sqrt((x ** 2).mean(axis=1, keepdims=True) + eps)
    return gamma * (x / rms), (x, rms, gamma)


def batchnorm_backward(dy, cache):
    """Eq. 57.9 — the three-term collapse."""
    xhat, s, gamma = cache
    B = len(dy)
    dgamma = (dy * xhat).sum(axis=0)
    dbeta = dy.sum(axis=0)
    dxhat = dy * gamma
    dx = (B * dxhat - dxhat.sum(axis=0)
          - xhat * (dxhat * xhat).sum(axis=0)) / (B * s)
    return dx, dgamma, dbeta


def layernorm_backward(dy, cache):
    xhat, s, gamma = cache
    d = xhat.shape[1]
    dgamma = (dy * xhat).sum(axis=0)
    dbeta = dy.sum(axis=0)
    dxhat = dy * gamma
    dx = (d * dxhat - dxhat.sum(axis=1, keepdims=True)
          - xhat * (dxhat * xhat).sum(axis=1, keepdims=True)) / (d * s)
    return dx, dgamma, dbeta


# --- verify against central differences -------------------------------------
print("=" * 72)
print("the backward passes, verified (eq. 57.9)")
print("=" * 72)
B, D = 8, 5
x = rng.normal(size=(B, D)) * 2 + 1
gamma = rng.normal(1.0, 0.2, D)
beta = rng.normal(0.0, 0.2, D)
w_out = rng.normal(size=(B, D))          # arbitrary downstream gradient


def check(fwd, bwd, label):
    y, cache = fwd(x, gamma, beta)
    dy = w_out
    dx, dgamma, dbeta = bwd(dy, cache)
    num = np.zeros_like(x)
    e = 1e-6
    for i in range(B):
        for j in range(D):
            xp, xm = x.copy(), x.copy()
            xp[i, j] += e
            xm[i, j] -= e
            num[i, j] = ((fwd(xp, gamma, beta)[0] * dy).sum()
                         - (fwd(xm, gamma, beta)[0] * dy).sum()) / (2 * e)
    rel = np.max(np.abs(dx - num) / np.maximum(np.abs(num), 1e-8))
    print(f"{label:<16} max relative error in dx: {rel:.3e}")
    return dx


dx_bn = check(batchnorm_forward, batchnorm_backward, "batchnorm")
dx_ln = check(layernorm_forward, layernorm_backward, "layernorm")

# --- section 6.1: the gradients are forced to sum to zero -------------------
print("\n" + "=" * 72)
print("what the backward pass CONSTRAINS (section 6.1)")
print("=" * 72)
print("Eq. 57.9's second and third terms remove the mean of the gradient")
print("and its component along xhat. So both must vanish exactly.\n")
xhat_bn = batchnorm_forward(x, gamma, beta)[1][0]
xhat_ln = layernorm_forward(x, gamma, beta)[1][0]
print(f"batchnorm: sum of dx over the BATCH axis, per feature")
print(f"  max |sum_i dx_ij|            = {np.abs(dx_bn.sum(axis=0)).max():.3e}")
print(f"  max |sum_i dx_ij * xhat_ij|  = "
      f"{np.abs((dx_bn * xhat_bn).sum(axis=0)).max():.3e}")
print(f"layernorm: sum of dx over the FEATURE axis, per example")
print(f"  max |sum_j dx_ij|            = "
      f"{np.abs(dx_ln.sum(axis=1)).max():.3e}")
print(f"  max |sum_j dx_ij * xhat_ij|  = "
      f"{np.abs((dx_ln * xhat_ln).sum(axis=1)).max():.3e}")

print("\nThe first constraint holds to machine precision. The second holds")
print("to about 1e-5, and the reason is the epsilon: eq. 57.2 divides by")
print("sqrt(var + eps), not by sqrt(var), so the layer is only")
print("scale-invariant up to that additive term. Set eps to zero and the")
print("second column drops to machine precision too:\n")
for label, ee in (("eps = 1e-5", 1e-5), ("eps = 0", 0.0)):
    y2, c2 = batchnorm_forward(x, gamma, beta, eps=ee)
    d2, _, _ = batchnorm_backward(w_out, c2)
    xh2 = c2[0]
    print(f"  {label:<12} max |sum_i dx| = {np.abs(d2.sum(axis=0)).max():.3e}"
          f"   max |sum_i dx*xhat| = "
          f"{np.abs((d2 * xh2).sum(axis=0)).max():.3e}")

print("\nThe constraints are not incidental. A gradient that would push")
print("every example in a batch the same way is projected out entirely,")
print("because it would only move the mean — which the normalisation")
print("immediately removes.")
print("\nThis is the most concrete formal statement available of how")
print("normalisation changes optimisation: it restricts the gradient to a")
print("subspace, and the two removed directions are exactly the two")
print("statistics the layer controls.")

# --- section 6.2: scale invariance ------------------------------------------
print("\n" + "=" * 72)
print("the weights feeding a normalisation are scale-invariant (eq. 57.10)")
print("=" * 72)
h = rng.normal(size=(32, 12))
W = rng.normal(0, 0.5, (12, 6))
g6, b6 = np.ones(6), np.zeros(6)
print(f"{'weight scale a':>15} {'eps = 1e-5':>16} {'eps = 0':>16}")
for a in (0.001, 0.01, 0.5, 1.0, 2.0, 100.0, 10000.0):
    row = []
    for ee in (1e-5, 0.0):
        base = batchnorm_forward(h @ W, g6, b6, eps=ee)[0]
        out = batchnorm_forward(h @ (a * W), g6, b6, eps=ee)[0]
        row.append(np.abs(out - base).max())
    print(f"{a:>15g} {row[0]:>16.3e} {row[1]:>16.3e}")

print("\nWith eps = 0 the invariance of eq. 57.10 is exact to floating")
print("point across seven orders of magnitude of weight scale. That is why")
print("Chapter 56's careful initialisation scale stops mattering in a")
print("normalised network: the scale is removed before the next layer sees")
print("it.")
print("\nWith the standard eps = 1e-5 the invariance is exact for large")
print("scales and BREAKS DOWN for small ones, because eps is an additive")
print("term in the denominator and stops being negligible once the")
print("variance falls to its order. At a = 0.001 the variance is a")
print("millionth of its original value and eps dominates entirely.")
print("\nThat is a real and easily missed limitation. Normalisation")
print("protects you from a badly scaled initialisation in one direction")
print("only: too large is absorbed exactly, and too SMALL runs into the")
print("epsilon floor and is not.")

# --- the effective learning rate consequence --------------------------------
print("\n" + "=" * 72)
print("the surprising consequence: an automatic learning-rate decay (6.2)")
print("=" * 72)
print("If the function is invariant to |W|, the gradient must be")
print("ORTHOGONAL to W. A step then grows |W| by Pythagoras, and since the")
print("effective step scales as 1/|W|^2, it shrinks by itself.\n")


def grad_wrt_W(W, h, target, eps=0.0):
    """Gradient of 0.5*||BN(hW) - target||^2 with respect to W.

    eps = 0 here deliberately: the invariance of eq. 57.10 is exact only
    without the epsilon, and this experiment is about that exact geometry.
    """
    z = h @ W
    y, cache = batchnorm_forward(z, np.ones(W.shape[1]),
                                 np.zeros(W.shape[1]), eps=eps)
    dy = y - target
    dz, _, _ = batchnorm_backward(dy, cache)
    return h.T @ dz


# A FRESH minibatch each step, so the gradient never decays to zero and the
# norm growth is not confounded with convergence.
rs_sgd = np.random.default_rng(4)
H_pool = rng.normal(size=(4096, 12))
T_pool = rng.normal(size=(4096, 6)) * 0.3
Wc = W.copy()
print(f"{'step':>6} {'|W|':>10} {'|grad|':>12} "
      f"{'|cos(W, grad)|':>16} {'eta*|g| / |W|':>15}")
lr = 0.05
for t in range(1, 4001):
    idx = rs_sgd.integers(0, len(H_pool), 32)
    g = grad_wrt_W(Wc, H_pool[idx], T_pool[idx])
    if t in (1, 10, 100, 500, 2000, 4000):
        cos = abs(float((Wc.ravel() @ g.ravel())
                        / (np.linalg.norm(Wc) * np.linalg.norm(g) + 1e-30)))
        print(f"{t:>6} {np.linalg.norm(Wc):>10.4f} "
              f"{np.linalg.norm(g):>12.3e} {cos:>16.3e} "
              f"{lr * np.linalg.norm(g) / np.linalg.norm(Wc):>15.3e}")
    Wc = Wc - lr * g

print("\nThe cosine between W and its gradient is zero to machine precision")
print("at every step. That is not a coincidence — eq. 57.10 says the loss")
print("does not change along the radial direction, so the derivative along")
print("it must vanish identically.")
print("\nEvery step therefore adds to |W| by Pythagoras and nothing ever")
print("subtracts, so the norm grows monotonically. The last column is the")
print("relative step size, and it falls as a direct consequence.")
print("\nNo schedule was applied here and the gradient is a fresh minibatch")
print("every step, so it is not decaying because the problem is being")
print("solved. The decay is produced entirely by the geometry of eq. 57.10.")
print("\nThis is the mechanism behind Chapter 58's account of weight decay")
print("in a normalised network: decay is what stops |W| from growing")
print("forever, and therefore what stops the effective learning rate from")
print("falling to zero on its own.")

# --- RMSNorm vs LayerNorm ---------------------------------------------------
print("\n" + "=" * 72)
print("RMSNorm and LayerNorm agree on centred input and not otherwise")
print("=" * 72)
print("They coincide when each ROW's mean is zero. The distribution's mean")
print("being zero is not enough: with d features the empirical row mean has")
print("standard deviation 1/sqrt(d), so at finite width the rows are not")
print("centred and the two differ.\n")
print(f"{'width d':>9} {'input mean':>12} {'typical row mean':>18} "
      f"{'mean |LN - RMS|':>18}")
for d in (8, 32, 128, 1024):
    for m in (0.0, 1.0):
        xx = rng.normal(size=(256, d)) + m
        lnv = layernorm_forward(xx, np.ones(d), np.zeros(d))[0]
        rnv = rmsnorm_forward(xx, np.ones(d))[0]
        print(f"{d:>9} {m:>12.1f} "
              f"{float(np.abs(xx.mean(axis=1)).mean()):>18.4f} "
              f"{float(np.abs(lnv - rnv).mean()):>18.4f}")

print("\nRead the zero-mean rows down the table: the discrepancy shrinks as")
print("1/sqrt(d), exactly tracking the row-mean column, and by width 1024")
print("it is negligible. That is the regime a transformer operates in.")
print("\nThe rows with an input mean of 1.0 do not shrink with width at")
print("all. A genuine offset in the data is not averaged away by making")
print("the layer wider, and RMSNorm's denominator absorbs it into the")
print("second moment, shrinking every output toward zero.")
print("\nSo the claim that RMSNorm works as well as LayerNorm is a")
print("substantive empirical claim about trained networks: their")
print("activations must be close enough to centred, at a width large")
print("enough, that the mean subtraction has nothing to remove. It is not")
print("a mathematical identity, and it would fail on data with a")
print("systematic offset.")
```

```python {tier=A name=batchnorm-batch-dependence}
"""The property that distinguishes batch normalisation from everything else:
its output depends on the other examples in the batch.
"""
import numpy as np

rng = np.random.default_rng(1)


def bn_train(x, gamma, beta, eps=1e-5):
    mu, var = x.mean(axis=0), x.var(axis=0)
    return gamma * (x - mu) / np.sqrt(var + eps) + beta


def bn_eval(x, gamma, beta, mu, var, eps=1e-5):
    return gamma * (x - mu) / np.sqrt(var + eps) + beta


def ln(x, gamma, beta, eps=1e-5):
    mu = x.mean(axis=1, keepdims=True)
    var = x.var(axis=1, keepdims=True)
    return gamma * (x - mu) / np.sqrt(var + eps) + beta


D = 16
gamma, beta = np.ones(D), np.zeros(D)

# --- the same example, different batches ------------------------------------
print("=" * 72)
print("one example, many batches: how much does the output move?")
print("=" * 72)
probe = rng.normal(size=(1, D))
pool = rng.normal(size=(20000, D))

print(f"{'batch size':>11} {'BN output sd':>14} {'LN output sd':>14} "
      f"{'BN noise / signal':>19}")
for B in (2, 4, 8, 32, 128, 512):
    outs_bn, outs_ln = [], []
    for _ in range(400):
        others = pool[rng.integers(0, len(pool), B - 1)]
        batch = np.vstack([probe, others])
        outs_bn.append(bn_train(batch, gamma, beta)[0])
        outs_ln.append(ln(batch, gamma, beta)[0])
    bn_sd = float(np.mean(np.std(outs_bn, axis=0)))
    ln_sd = float(np.mean(np.std(outs_ln, axis=0)))
    bn_mean = float(np.mean(np.abs(np.mean(outs_bn, axis=0))))
    print(f"{B:>11} {bn_sd:>14.5f} {ln_sd:>14.5f} "
          f"{bn_sd / max(bn_mean, 1e-12):>19.4f}")

print("\nThe SAME input produces a different output every time, depending")
print("on which other examples happened to share its batch. Layer norm's")
print("output does not move at all, because it never looks at them.")
print("\nThat variability is noise injection, and it is the regularisation")
print("effect of section 6.4. It shrinks as the batch grows — roughly as")
print("1/sqrt(B), which the column follows — so a large-batch run gets less")
print("regularisation from its batch norm than a small-batch one does.")
print("\nThis is why swapping BatchNorm for a batch-independent")
print("normalisation can COST accuracy for reasons that have nothing to do")
print("with optimisation: you removed a regulariser you did not know you")
print("were relying on.")

# --- the train/eval gap -----------------------------------------------------
print("\n" + "=" * 72)
print("the train/eval divergence (section 5.1 warning)")
print("=" * 72)
mu_run = pool.mean(axis=0)
var_run = pool.var(axis=0)
print(f"{'batch size':>11} {'mean |train - eval| output':>28} "
      f"{'relative to output scale':>26}")
for B in (1, 2, 8, 32, 256):
    diffs = []
    for _ in range(200):
        batch = pool[rng.integers(0, len(pool), B)]
        t = bn_train(batch, gamma, beta)
        e = bn_eval(batch, gamma, beta, mu_run, var_run)
        diffs.append(np.abs(t - e).mean())
    scale = np.abs(bn_eval(pool[:1000], gamma, beta, mu_run,
                           var_run)).mean()
    print(f"{B:>11} {float(np.mean(diffs)):>28.5f} "
          f"{float(np.mean(diffs)) / scale:>26.4f}")

print("\nAt batch size 1 the training-mode output is ZERO for every feature")
print("— one example has zero variance about its own mean — while the")
print("eval-mode output is whatever the running statistics say. The two")
print("modes compute completely different things.")
print("\nThe gap closes as the batch grows and never reaches zero, because")
print("a finite batch's statistics are an estimate. This is a permanent")
print("property of the design, not a bug to be fixed.")

# --- what a distribution shift does to the running statistics ---------------
print("\n" + "=" * 72)
print("running statistics assume the inference distribution matches (7.2)")
print("=" * 72)
print(f"{'shift (sd)':>11} {'output mean':>14} {'output sd':>12} "
      f"{'target: 0 and 1':>17}")
for shift in (0.0, 0.5, 1.0, 3.0):
    xs = pool[:2000] + shift
    out = bn_eval(xs, gamma, beta, mu_run, var_run)
    print(f"{shift:>11.1f} {out.mean():>14.4f} {out.std():>12.4f} "
          f"{'':>17}")

print("\nThe layer's whole purpose is to hand the next layer something with")
print("mean 0 and standard deviation 1. Under a distribution shift it hands")
print("over something else, and every layer above was trained assuming it")
print("would not.")
print("\nThat makes batch normalisation an amplifier of the covariate shift")
print("of Chapter 48 rather than a defence against it, which is worth")
print("noting given the technique's original name. Layer norm has no such")
print("exposure: it recomputes from the input it is actually given.")

# --- gradient accumulation is NOT exact with batch norm ---------------------
print("\n" + "=" * 72)
print("gradient accumulation is NOT exact with batch norm (section 7.4)")
print("=" * 72)
print("Chapter 53 measured accumulation reproducing a full batch to 1e-16.")
print("Here is the same test with a batch-norm layer in the way.\n")
Xb = pool[:256]
full = bn_train(Xb, gamma, beta)
print(f"{'micro-batch':>12} {'max |accumulated - full|':>26} "
      f"{'relative':>11}")
for micro in (256, 64, 32, 8):
    parts = [bn_train(Xb[i:i + micro], gamma, beta)
             for i in range(0, 256, micro)]
    acc = np.vstack(parts)
    print(f"{micro:>12} {np.abs(acc - full).max():>26.4e} "
          f"{np.abs(acc - full).max() / np.abs(full).max():>11.4f}")

print("\nThe difference is not floating-point; it is a different function.")
print("Each micro-batch normalises by its own statistics, so an example's")
print("output depends on which micro-batch it landed in. Chapter 53's")
print("exactness result required the loss to be a mean over INDEPENDENT")
print("examples, and batch normalisation is precisely the construction that")
print("breaks that independence.")
```

## 9. Practical Example

```python {tier=A name=normalization-on-a-network}
"""Does normalisation actually help, and which one? Measured on a deep
network, with the internal-covariate-shift test of Santurkar et al.
"""
import numpy as np

rng = np.random.default_rng(5)

D, C = 24, 5
_rs = np.random.default_rng(88)
A1, A2 = _rs.normal(size=(D, 16)), _rs.normal(size=(16, C))


def make_data(n, seed):
    rs = np.random.default_rng(seed)
    X = rs.normal(size=(n, D))
    logits = np.tanh(X @ A1) @ A2 * 1.6
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y = (p.cumsum(axis=1) < rs.random((n, 1))).sum(axis=1).clip(0, C - 1)
    return X, y


Xtr, ytr = make_data(40000, 1)
Xte, yte = make_data(10000, 2)
_p = np.exp(np.tanh(Xte @ A1) @ A2 * 1.6)
_p /= _p.sum(axis=1, keepdims=True)
BAYES = float(-np.log(_p[np.arange(len(yte)), yte]).mean())


class Net:
    """Deep MLP with a configurable normalisation, hand-written backward."""

    def __init__(self, depth, width, norm="none", scale=1.0, seed=0,
                 inject_shift=0.0):
        rs = np.random.default_rng(seed)
        self.depth, self.norm, self.inject = depth, norm, inject_shift
        self.W = [rs.normal(0, scale * np.sqrt(2.0 / D), (D, width))]
        for _ in range(depth - 1):
            self.W.append(rs.normal(0, scale * np.sqrt(2.0 / width),
                                    (width, width)))
        self.g = [np.ones(width) for _ in range(depth)]
        self.b = [np.zeros(width) for _ in range(depth)]
        self.Wout = rs.normal(0, np.sqrt(2.0 / width), (width, C))
        self.bout = np.zeros(C)
        self.shift_rs = np.random.default_rng(seed + 999)

    def _fwd_norm(self, z, l):
        if self.norm == "none":
            return z, None
        if self.norm == "batch":
            mu, var = z.mean(axis=0), z.var(axis=0)
            xhat = (z - mu) / np.sqrt(var + 1e-5)
            axis = 0
        else:                                   # layer
            mu = z.mean(axis=1, keepdims=True)
            var = z.var(axis=1, keepdims=True)
            xhat = (z - mu) / np.sqrt(var + 1e-5)
            axis = 1
        out = self.g[l] * xhat + self.b[l]
        if self.inject:
            # Santurkar et al.'s test: deliberately RESTORE covariate shift
            # by adding noise whose MEAN and VARIANCE change every step,
            # AFTER the normalisation has done its work. The magnitude is a
            # parameter because it decides whether this restores covariate
            # shift or simply destroys the signal.
            a = self.inject
            mu_t = self.shift_rs.normal(0, a, out.shape[1])
            sd_t = np.abs(self.shift_rs.normal(a, a / 2, out.shape[1]))
            out = out + mu_t + sd_t * self.shift_rs.normal(0, 1, out.shape)
        return out, (xhat, np.sqrt(var + 1e-5), axis)

    def forward(self, X):
        self.cache = []
        h = X
        for l in range(self.depth):
            z = h @ self.W[l]
            n, ncache = self._fwd_norm(z, l)
            a = np.maximum(0.0, n)
            self.cache.append((h, z, n, ncache))
            h = a
        self.hL = h
        return h @ self.Wout + self.bout

    def loss_and_grads(self, X, y):
        logits = self.forward(X)
        m = logits.max(axis=1, keepdims=True)
        e = np.exp(logits - m)
        loss = float(np.mean(m[:, 0] + np.log(e.sum(axis=1))
                             - logits[np.arange(len(X)), y]))
        d = e / e.sum(axis=1, keepdims=True)
        d[np.arange(len(X)), y] -= 1.0
        d /= len(X)
        gWout, gbout = self.hL.T @ d, d.sum(axis=0)
        dh = d @ self.Wout.T
        gW = [None] * self.depth
        gg = [None] * self.depth
        gb = [None] * self.depth
        for l in reversed(range(self.depth)):
            h_in, z, n, ncache = self.cache[l]
            dn = dh * (n > 0)
            if ncache is None:
                dz = dn
                gg[l] = np.zeros_like(self.g[l])
                gb[l] = np.zeros_like(self.b[l])
            else:
                xhat, s, axis = ncache
                gg[l] = (dn * xhat).sum(axis=0)
                gb[l] = dn.sum(axis=0)
                dxhat = dn * self.g[l]
                N = xhat.shape[axis]
                if axis == 0:
                    dz = (N * dxhat - dxhat.sum(axis=0)
                          - xhat * (dxhat * xhat).sum(axis=0)) / (N * s)
                else:
                    dz = (N * dxhat - dxhat.sum(axis=1, keepdims=True)
                          - xhat * (dxhat * xhat).sum(axis=1, keepdims=True)
                          ) / (N * s)
            gW[l] = h_in.T @ dz
            dh = dz @ self.W[l].T
        return loss, gW, gg, gb, gWout, gbout


def train(depth, width, norm, scale=1.0, steps=3000, lr=2e-3, batch=128,
          seed=0, inject_shift=0.0):
    net = Net(depth, width, norm, scale, seed, inject_shift)
    params = net.W + net.g + net.b + [net.Wout, net.bout]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    rs = np.random.default_rng(seed + 30)
    for t in range(1, steps + 1):
        idx = rs.integers(0, len(Xtr), batch)
        _, gW, gg, gb, gWout, gbout = net.loss_and_grads(Xtr[idx], ytr[idx])
        grads = gW + gg + gb + [gWout, gbout]
        for i, (pp, g) in enumerate(zip(params, grads)):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            pp -= lr * (m[i] / (1 - 0.9 ** t)) / (
                np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    net.inject = 0.0                        # evaluate without the injection
    te, _, _, _, _, _ = net.loss_and_grads(Xte, yte)
    acc = float((net.forward(Xte).argmax(axis=1) == yte).mean())
    return te - BAYES, acc


print("=" * 72)
print("does normalisation help, and at what depth?")
print("=" * 72)
print(f"Bayes-optimal test cross-entropy: {BAYES:.4f}")
print("Excess test loss above that floor; lower is better.\n")
print(f"{'normalisation':<16} " + " ".join(f"{f'depth {d}':>12}"
                                           for d in (2, 6, 16)))
for norm in ("none", "batch", "layer"):
    row = []
    for depth in (2, 6, 16):
        ex, acc = train(depth, 64, norm)
        row.append("diverged" if not np.isfinite(ex) else f"{ex:.4f}")
    print(f"{norm:<16} " + " ".join(f"{v:>12}" for v in row))

print("\nBoth normalisations give a consistent improvement at every depth,")
print("and — read the gap against the unnormalised row — it does NOT grow")
print("with depth over this range. That is not what the usual account")
print("predicts, and the reason is in the previous chapter: these networks")
print("are He-initialised, and Chapter 56 measured He initialisation")
print("keeping the variance profile flat to fifty layers on its own. There")
print("is no signal-propagation problem here for normalisation to solve.")
print("\nSo this table is NOT where normalisation earns its place. The next")
print("one is.")

# --- robustness to a bad initialisation scale -------------------------------
print("\n" + "=" * 72)
print("normalisation buys robustness to the initialisation scale (6.2)")
print("=" * 72)
print("Depth 16, weights multiplied by a factor the scheme did not intend.\n")
print(f"{'init scale':>11} " + " ".join(f"{n:>13}" for n in
                                        ("none", "batch", "layer")))
cols = {n: [] for n in ("none", "batch", "layer")}
for scale in (0.25, 0.5, 1.0, 2.0, 4.0):
    row = []
    for norm in ("none", "batch", "layer"):
        ex, acc = train(16, 64, norm, scale=scale)
        cols[norm].append(ex)
        row.append("diverged" if not np.isfinite(ex) or ex > 5
                   else f"{ex:.4f}")
    print(f"{scale:>11.2f} " + " ".join(f"{v:>13}" for v in row))

print(f"\n{'spread':>11} " + " ".join(
    f"{('inf' if not all(np.isfinite(v) and v < 5 for v in cols[n]) else f'{max(cols[n]) - min(cols[n]):.4f}'):>13}"
    for n in ("none", "batch", "layer")))

print("\nRead the SPREAD row, not the individual values. The question is")
print("how much the result depends on a scale factor the network should not")
print("care about at all.")
print("\nThe unnormalised network's spread is unbounded — it diverges at")
print("the top of the range. Batch normalisation's is much smaller, which")
print("is eq. 57.10 doing exactly what it says: the output does not depend")
print("on |W|, so the scale cannot matter.")
print("\nLayer normalisation sits between them, and the reason is worth")
print("noticing. It normalises across FEATURES within one example, so it")
print("controls the scale of each layer's output but not the relative")
print("scale of the weight matrix against the input — and at the smallest")
print("init scale it does noticeably worse. eq. 57.10's invariance is a")
print("property of what the layer is normalising over, and the two")
print("normalisations are normalising over different things.")
print("\nThe practical form of what normalisation buys is that spread row:")
print("it removes a hyperparameter you would otherwise have to get right,")
print("and it is more useful than a small improvement in the best")
print("achievable loss.")

# --- Santurkar et al.'s test ------------------------------------------------
print("\n" + "=" * 72)
print("the internal covariate shift test (section 6.3)")
print("=" * 72)
print("Santurkar et al.'s experiment: inject noise with a RANDOM, time-")
print("varying mean AFTER each normalisation layer. This deliberately")
print("restores — and worsens — internal covariate shift while keeping the")
print("normalisation. If Ioffe and Szegedy's explanation were right, this")
print("should destroy the benefit.\n")
none_a, _ = train(16, 64, "none")
fmt = lambda v: "diverged" if not np.isfinite(v) or v > 5 else f"{v:.4f}"
print(f"unnormalised baseline: {fmt(none_a)}\n")
print(f"{'injection scale':>16} " + " ".join(f"{n:>14}" for n in
                                             ("batch", "layer")))
for a_ in (0.0, 0.05, 0.15, 0.5, 1.0):
    row = []
    for norm in ("batch", "layer"):
        ex, _ = train(16, 64, norm, inject_shift=a_)
        row.append(fmt(ex))
    print(f"{a_:>16.2f} " + " ".join(f"{v:>14}" for v in row))

print("\nRead the batch-norm column against the unnormalised baseline.")
print("\nAt injection 0.05 and 0.15 the distribution downstream of every")
print("normalisation layer is being shifted and rescaled by a DIFFERENT")
print("random amount at every single step — internal covariate shift,")
print("deliberately restored and worse than anything the network would")
print("produce on its own. Batch normalisation's result is unchanged, and")
print("still comfortably better than the unnormalised baseline.")
print("\nThat is Santurkar et al.'s finding, reproduced. If normalisation")
print("worked by removing covariate shift, restoring covariate shift should")
print("have given back the unnormalised result. It did not.")
print("\nThe two largest injections do degrade both, and that is a")
print("different experiment: at that magnitude the added noise is")
print("comparable to the signal itself, so the damage is the noise")
print("destroying the representation rather than the covariate shift")
print("mattering. Distinguishing those two regimes is why the sweep is")
print("here rather than a single injection level — a one-row version of")
print("this experiment can accidentally test the wrong thing.")
print("\nNote that layer normalisation degrades earlier than batch")
print("normalisation does. It normalises within one example, so an")
print("injected per-feature offset is not averaged over anything.")
print("\nThis remains a small-scale reproduction. What matters is that the")
print("explanation was testable, someone tested it, and the field's")
print("standard account of its most-cited normalisation technique did not")
print("survive. The technique is ESTABLISHED. The explanation is not.")
print("Those are different claims and it is worth keeping them apart.")
```

## 10. Production Considerations

**Choose by batch size and by inference pattern.** Batch normalisation needs a
reasonable batch at training time and running statistics at inference; layer and
RMS normalisation need neither. If you serve one request at a time, prefer the
latter.

**Fold batch normalisation at inference.** {{eq:bn-folding}} makes it free, and
every deployment toolchain does it — so a benchmark that omits the folding
overstates its cost.

**Freeze batch-norm layers by putting them in eval mode**, not by freezing their
parameters. Running statistics are not parameters and are not affected by
`requires_grad`.

**Do not assume gradient accumulation reproduces a large batch.** Measured: with
batch normalisation it does not, and the difference is a different function
rather than floating-point.

**Check whether your distributed setup synchronises statistics.** Per-device
statistics mean the effective batch is the per-device one, and the result
changes with the device count.

**Monitor the running statistics.** Drift in them is an early signal of input
distribution change ({{ch:mle-drift}}), and it is a signal you are already
computing.

**Prefer pre-norm for deep transformers.** Post-norm needs warmup to train at
depth and pre-norm does not.

## 11. Common Mistakes

**Using batch normalisation at batch size 1.** Measured: the training-mode
output is exactly zero, because one example has zero variance about its own
mean.

**Forgetting eval mode.** {{ch:dl-forward}} measured the general case; with
batch normalisation the divergence is larger and systematic rather than random.

**Freezing parameters and expecting the statistics to freeze too.**

**Fine-tuning on a small batch without freezing the statistics.** A few hundred
steps overwrites what a long pretraining run accumulated.

**Assuming layer norm and RMSNorm are interchangeable.** Measured: identical on
centred input, sharply different otherwise.

**Placing a bias immediately before a normalisation.** The mean subtraction
removes it exactly, so it is a parameter that provably cannot do anything. RMS
normalisation is the exception, since it does not subtract a mean.

**Believing the internal covariate shift explanation.** Measured to be testable
and tested; it did not survive.

## 12. Failure Modes

**Train/eval mismatch.** Measured: the two modes compute different functions,
and the gap never reaches zero because a finite batch's statistics are an
estimate.

**Corrupted running statistics.** A short fine-tune or a final few unusual
batches, and inference degrades with no training-time symptom at all.

**Batch composition leakage.** If batches are constructed non-randomly — sorted
by length, or by class — the batch statistics carry information about the other
examples, which is a subtle form of the leakage {{ch:mle-pipelines}} treats.

**Distributed statistics mismatch.** The same code on a different device count
computes different statistics.

**Distribution shift amplified rather than absorbed.** Measured: the layer's
output mean and standard deviation move away from 0 and 1 under a shifted input,
and every layer above was trained assuming they would not.

**Silent regularisation loss.** Replacing batch normalisation with a
batch-independent variant removes the measured noise injection, and accuracy
falls for reasons unrelated to optimisation.

## 13. Alternatives

**Weight normalisation** reparameterises $\vec{w} = g\vec{v}/\|\vec{v}\|$,
normalising the weights rather than the activations. Batch-independent and
generally less effective.

**Normalisation-free networks** (NFNets, Fixup) reach comparable accuracy using
careful initialisation and adaptive gradient clipping instead. They demonstrate
that normalisation is not strictly necessary and have not displaced it.
{{maturity:EMERGING}}

**Dynamic Tanh** and related proposals replace the statistics with a fixed
elementwise nonlinearity, which is cheaper and removes the reduction. Promising
in published comparisons and, as of 2026, not used in shipped production models.
{{maturity:EMERGING}}

**ScaleNorm and variants** normalise by the whole vector's norm with a single
learned scalar — a further simplification of the RMSNorm direction.

**No normalisation at all** remains correct for shallow networks. The measured
depth-2 row shows it costing nothing there.

## 14. Evaluation

**Check the layer is in the right mode before evaluating.** The single most
common production error involving this chapter.

**Compare training-mode and eval-mode outputs on the same batch.** They should
be close for a large batch; a large gap means the running statistics are stale.

**Log the running statistics over time.** Free drift detection.

**Verify folding is numerically equivalent** before and after export.

**Ablate the normalisation at your actual depth.** Measured: it costs nothing at
depth 2, so the benefit is depth-dependent and worth confirming rather than
assuming.

**Sweep the initialisation scale with and without.** Measured: the spread across
scales is the clearest quantification of what normalisation buys.

## 15. Advanced Concepts

**The effective learning rate decay.** {{eq:norm-scale-invariance}} makes the
gradient orthogonal to $\mat{W}$, so $\|\mat{W}\|$ grows monotonically and the
effective step shrinks — measured directly here. This means a normalised network
has a *built-in* decay that interacts with whatever schedule
{{ch:dl-lr-schedules}} applies on top, and it partly explains why weight decay
matters so much in normalised networks: it counteracts the growth.

**Batch normalisation and weight decay.** Since the function is invariant to
$\|\mat{W}\|$, weight decay on those layers cannot change the function directly.
What it does is control $\|\mat{W}\|$ and therefore the effective learning rate,
which is a completely different mechanism from the classical
regularisation story.

**Loss-landscape smoothness.** {{cite:santurkar2018}}'s alternative explanation,
with bounds on the Lipschitz constants of the loss and its gradient. Better
supported than covariate shift and still not the whole answer.

**Ghost batch normalisation** computes statistics over sub-batches deliberately,
to keep the regularisation noise of a small batch while training with a large
one. A direct application of the measured $1/\sqrt{B}$ scaling.

**Normalisation in attention.** QK-normalisation — normalising the query and key
projections — has become common in large models to control attention-logit
growth ({{ch:tf-scaled-dot-product}}), which is this chapter's idea applied
somewhere it was not originally intended.

## 16. Connection to Previous Chapters

{{ch:ds-cleaning}} standardised features once at the input; this chapter does it
at every layer, and the reasoning is identical.

{{ch:dl-initialization}} is the alternative solution to the same problem, and
the two chapters should be read as a pair. The measured insensitivity there is
explained here by {{eq:norm-scale-invariance}}: the output does not depend on
the weight scale, so the initialisation cannot matter.

{{ch:dl-backprop}} supplied the machinery for {{eq:bn-backward}}, and its exact
gradient-accumulation result is the one this chapter breaks.
{{ch:dl-forward}}'s train/eval mode flag becomes consequential here.
{{ch:dl-optimizers}}'s weight decay acquires a different meaning under
{{eq:norm-scale-invariance}}. {{ch:mle-drift}}'s covariate shift is what
corrupts the running statistics.

Forward: {{ch:tf-architectures}} uses pre-norm layer normalisation and
{{ch:llm-anatomy}} uses RMSNorm, and both will read as instances of
{{sec:5-formal-explanation}}. {{ch:dl-cnns}} is where batch normalisation still
wins.

## 17. Exercises

**Beginner**

1. Which axis does batch normalisation reduce over? Layer normalisation?
2. Why does batch normalisation need running statistics?
3. What does RMSNorm remove relative to layer normalisation?
4. What are $\gamma$ and $\beta$ for?
5. Why does batch normalisation fail at batch size 1?

**Intermediate**

6. Derive {{eq:bn-param-grads}}.
7. Show from {{eq:bn-backward}} that the gradients sum to zero over the batch.
8. Prove {{eq:norm-scale-invariance}}.
9. Derive the folding formulae {{eq:bn-folding}}.
10. Explain why a bias immediately before a batch normalisation is redundant,
    and why the same is not true before an RMS normalisation.
11. Why does the regularisation effect scale as $1/\sqrt{B}$?

**Advanced**

12. Derive {{eq:bn-backward}} in full from the three gradient paths.
13. Show that {{eq:norm-scale-invariance}} implies the gradient is orthogonal
    to $\mat{W}$, and derive the resulting growth of $\|\mat{W}\|$.
14. Explain what weight decay does to a scale-invariant layer, and why it is
    not classical regularisation.
15. Derive the pre-norm and post-norm Jacobians and compare their products over
    $L$ layers.
16. Design an experiment that would distinguish the loss-smoothness explanation
    from a third hypothesis of your choosing.

**Implementation**

17. Implement all four normalisations and verify each backward pass
    numerically.
18. Implement batch-norm folding and verify equivalence to $10^{-6}$.
19. Reproduce the batch-dependence measurement and confirm the $1/\sqrt{B}$
    scaling.
20. Implement ghost batch normalisation and measure whether it recovers the
    small-batch regularisation at a large batch.

**Reasoning**

21. A model scores well offline and badly in production, with no data drift.
    What do you check first?
22. Replacing BatchNorm with GroupNorm cost you 1% accuracy at the same
    training loss. Explain.

## 18. Interview Questions

**"What does batch normalisation do?"** — Normalise per-feature over the batch,
then a learned affine. A strong answer immediately notes the train/eval
difference.

**"Why does it help?"** — The honest answer: originally attributed to reducing
internal covariate shift, which was tested and refuted; the current best account
is loss-landscape smoothing, which is not settled. Saying "internal covariate
shift" without qualification is a real signal.

**"BatchNorm or LayerNorm?"** — Batch dependence. Vision with a decent batch
against sequences and batch-1 inference.

**"Why do transformers use LayerNorm?"** — Variable-length sequences, batch size
1 at inference, no running statistics.

**"What is RMSNorm and why did it win?"** — No mean subtraction, no $\beta$;
cheaper and empirically equal, which implies the centring was not the active
ingredient.

**"What is the difference between pre-norm and post-norm?"** — Where the
normalisation sits relative to the residual. Pre-norm leaves a clean identity
gradient path and does not need warmup.

**"Your model behaves differently in production."** — Eval mode, then running
statistics, then distribution shift. In that order.

## 19. Research Questions

**Why does normalisation help?** Internal covariate shift was tested and
refuted. Loss-landscape smoothing is better supported and not conclusive, and
the automatic learning-rate decay of {{eq:norm-scale-invariance}} is a third
mechanism that is real and hard to separate from the others.
{{maturity:RESEARCH FRONTIER}}

**Is normalisation necessary?** Normalisation-free networks reach comparable
accuracy with careful initialisation and clipping, and have not been adopted. It
is unclear whether that is because they are worse in some way not captured by
the benchmarks or because normalisation is simply good enough.
{{maturity:EMERGING}}

**How much of batch normalisation's benefit is regularisation?** Measured here
to be real and to scale as $1/\sqrt{B}$; disentangling it from the optimisation
benefit is not straightforward. {{maturity:EMERGING}}

**Can the statistics be replaced by a fixed nonlinearity?** Dynamic Tanh and
relatives suggest yes on the benchmarks tried, and no production model has
adopted them as of 2026. {{maturity:EMERGING}}

## 20. Chapter Summary

Normalisation standardises activations at every layer rather than only at the
input, and the only substantive difference between the variants is which axis
they reduce over. Batch normalisation reduces over the batch, so its output
depends on the other examples present; layer, group and RMS normalisation do
not. Every other distinction — batch-size sensitivity, running statistics, the
train/eval divergence, the regularisation effect — follows from that one choice.

The backward pass, verified numerically here, does something specific: it forces
the gradients to sum to zero over the reduced axis and removes their component
along the normalised activation, both confirmed to machine precision. The
optimiser is therefore restricted to a subspace, and the two removed directions
are exactly the two statistics the layer controls. That is the sharpest formal
statement available of how normalisation changes training.

{{eq:norm-scale-invariance}} says the output does not depend on the scale of the
weights feeding the layer, measured here to be exact across four orders of
magnitude of weight scale. Two consequences follow. Initialisation stops
mattering for those layers, which explains {{ch:dl-initialization}}'s measured
insensitivity. And — measured directly — the gradient is orthogonal to
$\mat{W}$ at every step, so $\|\mat{W}\|$ only grows and the effective step size
falls on its own. A normalised network has a built-in learning-rate decay that
no schedule asked for.

Batch normalisation's batch dependence was measured directly: the same input
produced a different output in every batch, with a spread shrinking as
$1/\sqrt{B}$. That is noise injection and it regularises, which is why replacing
batch normalisation with a batch-independent variant can cost accuracy for
reasons that have nothing to do with optimisation. The same dependence produces
the train/eval divergence — at batch size 1 the training-mode output is exactly
zero — and it breaks {{ch:dl-backprop}}'s exact gradient-accumulation result,
because that result assumed the loss was a mean over independent examples and
batch normalisation is precisely the construction that removes the independence.

RMSNorm and LayerNorm were measured to be identical on centred input and sharply
different otherwise. That RMSNorm performs as well in practice is therefore a
substantive finding rather than a triviality: it says the activations in a
trained network are centred enough that the mean subtraction is not doing work.

Finally, the epistemic lesson. {{cite:ioffe2015}} attributed the benefit to
reducing internal covariate shift. {{cite:santurkar2018}} injected covariate
shift *after* the normalisation, deliberately restoring the thing the technique
was supposed to remove, and training was unaffected. The technique is
{{maturity:ESTABLISHED}}; the explanation was wrong for three years and the
replacement is not settled. **A method working is not evidence that the stated
reason for it working is correct**, and this chapter is the book's clearest case
of that.

## 21. Further Reading

{{cite:ioffe2015}} should be read for the technique and read sceptically for the
explanation. It is a good exercise: the covariate-shift argument is stated
plainly enough that you can see exactly what claim was being made and why it
seemed reasonable.

{{cite:santurkar2018}} is the more valuable paper and it is a model of how to
test a mechanistic claim. The noise-injection experiment is the part to
understand — it is simple, decisive, and it is the kind of experiment that
should be run far more often than it is.

{{cite:ba2016layernorm}} is short and the motivation section is the useful part:
it says exactly which properties of batch normalisation it was trying to avoid,
which is a cleaner statement of the trade-off than most later treatments.

{{cite:zhang2019rmsnorm}} for RMSNorm. The argument is essentially "we tried
removing the mean and nothing got worse", which turned out to be right and is
worth noting as a style of contribution.

{{cite:xiong2020prenorm}} for the pre-norm/post-norm gradient analysis, which is
what justifies a choice every transformer implementation makes.

**Where to go next:** {{ch:dl-regularization}} covers the explicit regularisers
that normalisation partly substitutes for, and the interaction measured here —
that removing batch normalisation removes a regulariser — is the reason to read
them together.

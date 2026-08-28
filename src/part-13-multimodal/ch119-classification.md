---
id: mm-classification
number: 119
part: XIII
tier: full
status: draft
requires: [mm-cv-fundamentals, dl-cnns, dl-normalization, dl-initialization,
           dl-optimizers]
provides: [degradation-problem, identity-embedding, residual-connection-mechanism,
           residual-variance-growth, bottleneck-block, unrolled-ensemble-view,
           recipe-versus-architecture, transfer-learning-baseline]
citations: [he2016resnet, krizhevsky2012, simonyan2015vgg, russakovsky2015ilsvrc,
            liu2022convnext, ioffe2015, dosovitskiy2021vit]
---

## 1. Learning Objectives

By the end of this chapter you will be able to state the **degradation problem**
precisely — and say why it is a statement about optimisation rather than about
capacity or overfitting; write the identity-embedding argument that makes
degradation surprising; explain the residual connection's two distinct effects and
show, by measurement, that the one usually quoted is the smaller; derive why a
residual stream's variance grows linearly in depth and therefore why
normalisation is not optional; and say what {{cite:liu2022convnext}} established
about how much of the transformer-versus-convnet gap was architecture at all.

## 2. Why This Matters

Classification is the task nobody deploys and everybody depends on. What you
actually use is a **backbone** — a stack of layers trained on classification and
then reused, as the feature extractor under a detector, the encoder under a
segmenter, or the vision tower inside a VLM. Its architecture is the one you
inherit.

And the architecture is what it is because of one experiment that came out the
wrong way. Adding layers to a plain deep network made it **worse on training
error** — not test error, which would be ordinary overfitting, but training error,
on the objective being directly minimised. {{cite:he2016resnet}} named this
degradation, and the residual connection is the response.

**The reason this matters beyond history is that the response turned out to be
general.** Transformers are residual. Diffusion denoisers are residual. Nothing
you use in the rest of this book is convolutional and everything you use is
residual, so understanding *why* the wire works is worth more than the
architecture it was introduced in.

{{sec:9-practical-example}} reproduces degradation at a scale you can check, and
then tests the standard explanation. Skip connections do improve gradient flow —
but only by a factor of **1.8** at depth 32, which is being asked to explain the
difference between a network that learned the task and one that mostly did not.
The second listing isolates what else changed: at initialisation a plain stack has
already destroyed its input, and asked to learn the identity — a function it can
represent exactly — it lands at **0.7173** relative error where the residual stack
reaches **0.0947**.

{{maturity:ESTABLISHED}} Residual networks. {{maturity:MATURE}}
{{cite:liu2022convnext}}'s finding that much of the apparent
architecture gap was training recipe.

## 3. Prerequisites

{{ch:mm-cv-fundamentals}} for the stack this chapter deepens, and for
{{eq:resolution-tension}}; {{ch:dl-cnns}} for the convolutional block;
{{ch:dl-normalization}} for batch normalisation, which {{eq:residual-variance}}
shows is load-bearing rather than incidental; {{ch:dl-initialization}} for why the
starting point is a design decision; {{ch:dl-optimizers}} for what "the optimiser
cannot find it" means.

## 4. Intuitive Explanation

### The experiment that did not behave

Depth had been the reliable lever. {{cite:krizhevsky2012}} was 8 layers,
{{cite:simonyan2015vgg}} showed 16–19 was better, and the obvious next step was
more.

It stopped working, and it stopped working in a way that ruled out the two
comfortable explanations:

> **Not overfitting.** The *training* error was higher. Overfitting is low
> training error and high test error; this is high training error.
>
> **Not capacity.** A deeper network contains a shallower one. Take the 20-layer
> solution, add 36 layers set to the identity, and you have a 56-layer network
> computing exactly the same function. The deeper model's hypothesis space
> *contains* the shallower model's solution.

So the deeper network is doing worse at a job it demonstrably could do. **The
solution exists, is reachable, and the optimiser does not find it.** That is the
degradation problem, and it is an interesting statement because it is about
search rather than about representation.

### The fix is a wire

$$ \text{plain: } h_{\ell+1} = F(h_\ell) \qquad\longrightarrow\qquad \text{residual: } h_{\ell+1} = h_\ell + F(h_\ell) $$

No parameters. And the effect is best understood by asking what each architecture
must do **to do nothing**:

| To compute the identity | plain block | residual block |
|---|---|---|
| what the weights must be | a specific structured matrix | **zero** |
| where initialisation starts | far from it | near it |
| where weight decay pulls | away from it | **toward it** |

**In a plain stack, doing nothing is a skill that has to be learned. In a residual
stack, doing nothing is the default.** That reframes what depth costs: extra
residual layers start out harmless, so the optimiser can spend its effort on
whether they should do something, rather than on stopping them doing damage.

{{sec:9-practical-example}} tests this directly by asking both architectures to
learn the identity, and the plain stack fails at it **even with only two layers**
(0.3123 relative error) while the residual stack reaches 0.0001.

### The explanation everyone gives, and why it is only half

The standard account is vanishing gradients: the skip gives the gradient a
shortcut, so early layers get a usable signal.

That is true, and it is not enough. {{sec:9-practical-example}} measures the
first layer's gradient with and without skips and finds a ratio of **1.8 at depth
32** — under a factor of two, at the depth where the plain network's training
accuracy has collapsed from 0.751 to 0.553. A modest change in gradient magnitude
is being asked to account for a large change in outcome.

**What else changed is the starting function.** At initialisation a residual
network is approximately the identity plus noise, and a plain network is
approximately noise. The second listing measures it: the correlation between an
input coordinate and its matching output coordinate is **0.0544** for a 32-layer
plain stack and **0.3735** for the residual one. A plain deep network has thrown
its input away before training begins, and every useful solution must start by
rebuilding a path back to it.

### What the skip costs

Nothing is free, and {{sec:9-practical-example}} shows the price. At depths 2 and
4 the *plain* network wins — 0.6193 training loss against 0.7760. A residual
stream accumulates one branch's worth of variance per layer, so it must be scaled
or normalised, and at shallow depth that is insurance you are paying for and do
not need. **The crossover in the measurement is around depth 8.**

## 5. Formal Explanation

### 5.1 Degradation, stated

Let $\mathcal{H}_d$ be the functions representable by a depth-$d$ network of a
given width, and $\hat{L}_d$ the *training* loss actually achieved by the
optimiser. The identity-embedding argument gives

$$ \mathcal{H}_{d} \subseteq \mathcal{H}_{d'} \quad \text{for } d' > d $$ (eq:identity-embedding)

by setting the extra layers to the identity. So $\min_{f \in \mathcal{H}_{d'}} L(f)
\le \min_{f \in \mathcal{H}_{d}} L(f)$: a deeper network's *best achievable*
training loss cannot be worse.

Degradation is the observation that

$$ \hat{L}_{d'} > \hat{L}_{d} \quad \text{for } d' > d $$ (eq:degradation)

**{{eq:identity-embedding}} and {{eq:degradation}} together are the whole
puzzle.** The gap between them is the optimiser's failure, and nothing else.

> **One caveat on {{eq:identity-embedding}} that is usually skipped.** A ReLU
> layer computes $\max(Wh, 0)$, which equals $h$ at $W = I$ only when $h \ge 0$.
> Inside a network that holds — activations after the first ReLU are non-negative
> — so the argument is sound, but it is sound *conditionally*, and
> {{sec:9-practical-example}}'s second listing is built on non-negative inputs
> precisely so the comparison stays fair.

### 5.2 The residual block

$$ h_{\ell+1} = h_\ell + F_\ell(h_\ell; \theta_\ell) $$ (eq:residual-block)

and the property that matters:

$$ \theta_\ell = 0 \;\Longrightarrow\; F_\ell \equiv 0 \;\Longrightarrow\; h_{\ell+1} = h_\ell $$ (eq:identity-is-default)

**{{eq:identity-is-default}} is the mechanism.** The identity is not a point the
optimiser must search for; it is where the optimiser starts and where
regularisation pushes.

### 5.3 The gradient, and what it does and does not explain

Differentiating {{eq:residual-block}}:

$$ \frac{\partial h_{\ell+1}}{\partial h_\ell} = I + \frac{\partial F_\ell}{\partial h_\ell} \quad\Longrightarrow\quad \frac{\partial h_L}{\partial h_\ell} = \prod_{j=\ell}^{L-1}\left(I + \frac{\partial F_j}{\partial h_j}\right) $$ (eq:residual-gradient)

Expanding the product gives an additive term equal to $I$ — a path from the loss
to layer $\ell$ that passes through *no* Jacobians. For a plain network the
corresponding quantity is a bare product of Jacobians, which shrinks or explodes
geometrically.

**This is the standard argument and it is correct.** What
{{sec:9-practical-example}} adds is a magnitude: the measured first-layer
gradient ratio at depth 32 is 1.8, so {{eq:residual-gradient}}'s benefit is real
and modest, and cannot by itself account for the outcome gap.

### 5.4 The unrolled ensemble

Expanding {{eq:residual-gradient}}'s forward analogue over $L$ blocks gives
$2^L$ terms, one per subset of blocks:

$$ h_L = h_0 + \sum_{\ell} F_\ell(\cdot) + \sum_{\ell < m} F_m(F_\ell(\cdot)) + \cdots $$ (eq:unrolled-ensemble)

so a residual network behaves like an **ensemble of exponentially many paths of
varying depth**, most of them short. This explains an otherwise odd empirical
fact: deleting a single block from a trained ResNet barely changes its output,
while deleting a layer from a plain network destroys it. The short paths carry
most of the signal, and no single block is on the critical path.

### 5.5 Variance, and why normalisation is not optional

Take $\operatorname{Var}[F_\ell(h)] \approx \sigma_F^2$, roughly independent
across blocks. Then from {{eq:residual-block}}:

$$ \operatorname{Var}[h_L] \approx \operatorname{Var}[h_0] + L\,\sigma_F^2 \quad\Longrightarrow\quad \|h_L\| \sim \sqrt{L} $$ (eq:residual-variance)

**The residual stream grows without bound in depth.** At depth 32 that is a
factor of ~6 in scale, and in {{sec:9-practical-example}}'s first draft it
overflowed outright. The fixes are equivalent up to constants: normalisation
inside the branch ({{cite:ioffe2015}}, which is what a real ResNet does), or
scaling the branch by $1/\sqrt{L}$, which is what the listing does in one line.

So the residual connection and normalisation are **not two independent tricks**.
The first creates a problem that the second solves, and a residual network without
normalisation is not a simplification, it is a divergence.

### 5.6 The bottleneck block

Depth is cheap only if each block is. For width $C$, a $3\times3$ block costs
$9C^2$ per position; the bottleneck factorises it as $1\times1$ down, $3\times3$
at reduced width $C/r$, $1\times1$ up:

$$ \text{cost} = \frac{C^2}{r} + \frac{9C^2}{r^2} + \frac{C^2}{r} \;\approx\; C^2\left(\frac{2}{r} + \frac{9}{r^2}\right) $$ (eq:bottleneck-cost)

At $r = 4$: $0.5 + 0.56 = 1.06\,C^2$ against $9C^2$ — an **8.5× saving**, which is
what buys depths of 50, 101, and 152.

## 6. Mathematical Foundation

### 6.1 Why the plain stack loses its input

Model each plain layer as a random projection followed by rectification. ReLU
zeroes about half its inputs, so the surviving correlation between an input
coordinate and its matching output coordinate is multiplied by some $\rho < 1$
per layer:

$$ \text{corr}_L \approx \rho^{L} \quad\text{(geometric decay)} $$ (eq:signal-decay)

For a residual stack, {{eq:residual-block}} keeps $h_0$ as an explicit additive
term, so the correlation is bounded below by the identity path's share of the
variance:

$$ \text{corr}^{\text{res}}_L \gtrsim \sqrt{\frac{\operatorname{Var}[h_0]}{\operatorname{Var}[h_0] + L\sigma_F^2}} \sim \frac{1}{\sqrt{L}} $$ (eq:signal-survival)

**Geometric decay against $1/\sqrt{L}$ decay.** Both fall; only one falls fast
enough to matter. {{sec:9-practical-example}} measures plain going 0.0870 →
0.0544 over depths 2→32 and residual going 0.7308 → 0.3735 — the residual stack
retains **seven times more** signal at depth 32.

### 6.2 The identity-learning task, worked

At depth $d$, a plain stack must find $W_\ell \approx I$ at every layer
*simultaneously*: the error in the composed map is first-order the sum of the
per-layer errors, so a fixed per-layer accuracy $\epsilon$ gives composed error
$\approx d\,\epsilon$. Errors accumulate linearly in depth.

A residual stack must drive $\theta_\ell \to 0$, and its composed error under the
$1/\sqrt{d}$ branch scaling is

$$ \left\|\sum_{\ell} \tfrac{1}{\sqrt{d}} F_\ell\right\| \approx \sqrt{d}\cdot\tfrac{1}{\sqrt{d}}\,\|F\| = \|F\| $$ (eq:residual-identity-error)

— *independent of depth*, provided each $\|F_\ell\|$ shrinks equally. The
measurement is not quite that clean (0.0001 at depth 2 rising to 0.0947 at 32),
because a fixed optimisation budget is being spread over more parameters, but the
contrast with the plain stack's 0.3123 → 0.7173 is the predicted one.

> **MATH NOTE:** {{eq:residual-identity-error}} explains why the residual column
> degrades *gently* rather than not at all. The budget, not the architecture, is
> what makes depth 32 worse than depth 2 there — and that is a different and much
> more benign failure than the plain stack's, where the error grows because the
> problem itself got harder.

### 6.3 Reading {{cite:liu2022convnext}}'s result correctly

When ViTs beat ResNets, the comparison changed several things at once:
architecture, optimiser, augmentation, regularisation, epochs.
{{cite:liu2022convnext}} held the recipe fixed and modernised a ResNet one change
at a time until it matched Swin Transformers.

$$ \Delta_{\text{observed}} = \Delta_{\text{architecture}} + \Delta_{\text{recipe}} $$ (eq:recipe-versus-architecture)

and the ablation attributes much of $\Delta_{\text{observed}}$ to the second term.
**The general lesson is about experimental hygiene**, not about convolutions: a
comparison that varies the training recipe alongside the architecture measures
their sum, and reports it as the first.

## 7. Internal Mechanics

```mermaid {#fig:residual-block caption="The block, and the two things the wire changes. The identity path makes doing nothing free (eq:identity-is-default) and gives the gradient a Jacobian-free route back (eq:residual-gradient). It also makes the stream's variance grow with depth (eq:residual-variance), which is why the normalisation inside the branch is structural rather than decorative."}
flowchart LR
    H["h"] --> N1["norm"] --> C1["1x1 conv<br/>C -> C/4"] --> A1["ReLU"]
    A1 --> C2["3x3 conv<br/>C/4"] --> A2["ReLU"] --> C3["1x1 conv<br/>C/4 -> C"]
    C3 --> ADD(("+"))
    H -->|"identity path:<br/>no weights, no Jacobian"| ADD
    ADD --> OUT["h + F(h)"]
    C3 -.->|"branch variance adds<br/>every block"| ADD
```

### 7.1 The stages, and where the backbone is cut

A ResNet is four stages, each a run of blocks at one resolution, halving spatial
size and doubling channels between them ({{eq:feature-map-cost}}). What matters
downstream is **which stage you tap**:

| Tap | Resolution | Receptive field | Used by |
|---|---|---|---|
| stage 2 | /8 | small | small-object detection, segmentation detail |
| stage 3 | /16 | medium | the usual detection default |
| stage 4 | /32 | large | classification, global context |
| after pool | 1×1 | whole image | a classifier head only |

**{{eq:pooling-invariance}}'s global pool is where the *where* is discarded**, and
every task in the rest of this part either taps before it or reconstructs what it
threw away.

### 7.2 Where the projection shortcut goes

{{eq:residual-block}} requires $h_\ell$ and $F(h_\ell)$ to have the same shape,
which fails at stage boundaries where channels double and resolution halves. The
standard answer is a $1\times1$ convolution with stride 2 on the shortcut — and it
is worth noting that this **breaks {{eq:identity-is-default}} at exactly those
blocks**. Downsampling blocks are the ones that must learn something; the rest are
free to do nothing.

### 7.3 Transfer learning, and the honest version

The reason anyone trains on ImageNet is to reuse the result. Two facts about that
which are usually asserted rather than measured:

- **Early layers transfer almost universally.** Edges and textures are not
  ImageNet-specific.
- **Late layers transfer only to similar domains.** A stage-4 feature is a
  1000-class discriminative summary; on X-rays or satellite imagery it may be
  worse than random initialisation plus enough data.

The practical rule follows: **fine-tune from the top down**, and treat "how many
stages to unfreeze" as a hyperparameter set by how far your domain is from
{{cite:russakovsky2015ilsvrc}}'s.

## 8. Implementation

```python {tier=A name=degradation-not-vanishing}
"""The degradation problem, and why "vanishing gradients" is the wrong diagnosis.

cite:he2016resnet's motivating observation is easy to state and easy to
misremember. Adding layers to a plain deep network made it worse -- not on test
error, which would be overfitting, but on TRAINING error, which cannot be.

That rules out capacity. A deeper network can express everything a shallower one
can: set the extra layers to the identity and you have the shallower network
exactly (eq:identity-embedding). So the deeper model's higher training loss is a
statement about OPTIMISATION, not about what the architecture can represent.

This listing reproduces degradation on a task small enough to check, and measures
the gradient at the same time -- because if vanishing gradients were the whole
story, the gradient measurement would show it.
"""
import numpy as np

rng = np.random.default_rng(11)

D_IN, WIDTH, N_CLASS = 24, 32, 4
N_TRAIN = 4000
EPOCHS, BATCH, LR = 40, 64, 0.05


def teacher_data(n):
    """A fixed random two-layer teacher, so the task is definitely learnable by
    a small network and any failure is the optimiser's."""
    g = np.random.default_rng(0)
    A = g.normal(size=(D_IN, 16)) / np.sqrt(D_IN)
    B = g.normal(size=(16, N_CLASS)) / 4
    X = rng.normal(size=(n, D_IN))
    y = (np.maximum(X @ A, 0) @ B).argmax(axis=1)
    return X, y


class Net:
    def __init__(self, depth, residual):
        self.depth, self.residual = depth, residual
        # A residual stream accumulates one branch's variance per layer, so its
        # scale grows like sqrt(depth) and a deep stack overflows
        # (eq:residual-variance). A real ResNet controls this with normalisation
        # inside the branch; scaling the branch by 1/sqrt(depth) does the same
        # job in one line. It is not optional -- without it, depth 32 diverges.
        self.scale = 1.0 / np.sqrt(depth) if residual else 1.0
        self.Win = rng.normal(scale=np.sqrt(2 / D_IN), size=(D_IN, WIDTH))
        self.W = [rng.normal(scale=np.sqrt(2 / WIDTH), size=(WIDTH, WIDTH))
                  for _ in range(depth)]
        self.b = [np.zeros(WIDTH) for _ in range(depth)]
        self.Wout = rng.normal(scale=np.sqrt(2 / WIDTH), size=(WIDTH, N_CLASS))
        self.bout = np.zeros(N_CLASS)

    def forward(self, X):
        self.hs, self.zs = [], []
        h = np.maximum(X @ self.Win, 0)
        self.x = X
        for W, b in zip(self.W, self.b):
            self.hs.append(h)
            z = h @ W + b
            self.zs.append(z)
            a = np.maximum(z, 0)
            # The ONLY difference between the two architectures.
            h = h + self.scale * a if self.residual else a
        self.hlast = h
        return h @ self.Wout + self.bout

    def backward(self, g, lr):
        gWout, gbout = self.hlast.T @ g, g.sum(axis=0)
        gh = g @ self.Wout.T
        first_grad = None
        for i in reversed(range(self.depth)):
            ga = gh * self.scale                # residual: identity path carries gh
            gz = ga * (self.zs[i] > 0)
            gW = self.hs[i].T @ gz
            gb = gz.sum(axis=0)
            gh = gz @ self.W[i].T + (gh if self.residual else 0)
            self.W[i] -= lr * gW
            self.b[i] -= lr * gb
            if i == 0:
                first_grad = float(np.linalg.norm(gW))
        self.Wout -= lr * gWout
        self.bout -= lr * gbout
        return first_grad


def softmax_ce(logits, y):
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
    loss = -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
    g = p.copy(); g[np.arange(len(y)), y] -= 1
    return loss, g / len(y)


X, y = teacher_data(N_TRAIN)

print(f"{N_TRAIN} training points, width {WIDTH}, {EPOCHS} epochs. Every number "
      f"below is\nTRAINING performance -- no test set is involved, so nothing "
      f"here is overfitting.\n")
print(f"{'depth':>7}{'plain loss':>13}{'plain acc':>12}{'':>4}"
      f"{'residual loss':>15}{'residual acc':>14}{'':>4}{'grad ratio':>12}")
print("-" * 82)

rows = {}
for depth in (2, 4, 8, 16, 32):
    out = {}
    for residual in (False, True):
        rng2 = np.random.default_rng(11)
        globals()['rng'] = rng2
        net = Net(depth, residual)
        gnorm = []
        for ep in range(EPOCHS):
            order = rng2.permutation(N_TRAIN)
            for s in range(0, N_TRAIN, BATCH):
                b = order[s:s + BATCH]
                logits = net.forward(X[b])
                _, g = softmax_ce(logits, y[b])
                fg = net.backward(g, LR)
                if ep == 0:
                    gnorm.append(fg)
        logits = net.forward(X)
        loss, _ = softmax_ce(logits, y)
        acc = float((logits.argmax(axis=1) == y).mean())
        out[residual] = (loss, acc, float(np.mean(gnorm)))
    rows[depth] = out
    ratio = out[True][2] / max(out[False][2], 1e-12)
    print(f"{depth:>7}{out[False][0]:>13.4f}{out[False][1]:>12.3f}{'':>4}"
          f"{out[True][0]:>15.4f}{out[True][1]:>14.3f}{'':>4}{ratio:>12.1f}")

p2, p32 = rows[2][False], rows[32][False]
r2, r32 = rows[2][True], rows[32][True]
print(f"""
Read the plain columns downward first, and remember these are TRAINING numbers.
Loss falls from {p2[0]:.4f} at depth 2 to {rows[4][False][0]:.4f} at depth 4 --
depth is helping -- and then reverses, rising to {p32[0]:.4f} at depth 32 with
training accuracy down to {p32[1]:.3f}. The deep model is not overfitting. It
never fitted. It is worse at the task it was directly optimised on.

That single observation is what makes degradation interesting, and it is why
"deeper is harder to train" is an incomplete explanation. Consider what the
depth-32 network could do: set twenty-four of its layers to the identity and it
becomes the depth-8 network exactly (eq:identity-embedding). The solution is
inside the hypothesis space, it is reachable, and the optimiser does not find it.
The failure is in the SEARCH, not in the space.

The residual columns are the same task, width, optimiser and epoch count, with
one line different. Loss at depth 32 is {r32[0]:.4f} against the plain network's
{p32[0]:.4f}, and accuracy holds at {r32[1]:.3f} against {p32[1]:.3f}. The
residual column is nearly FLAT in depth, which is the real claim: it is not that
skips make deep networks better, it is that they stop depth from making things
worse.

Now read the top two rows, because they are the part an enthusiastic account
would omit. At depth 2 and depth 4 the plain network WINS -- {rows[4][False][0]:.4f}
against {rows[4][True][0]:.4f}. Residual connections are not free: the branch is
scaled down to keep the stream stable, so at shallow depth the architecture is
paying for insurance it does not need. The crossover here is around depth 8, and
below it the skip is a small cost rather than a benefit.

Finally the gradient ratio column, where the usual explanation gets tested rather
than repeated. It reports how much larger the first layer's gradient is with skips
than without, early in training. It rises with depth, so skips do improve gradient
flow and the vanishing-gradient story is not wrong.

It is just too small to be the whole story. At depth 32 the ratio is only
{rows[32][True][2] / rows[32][False][2]:.1f} -- less than a factor of two -- while
the outcome gap is the difference between a network that learned the task and one
that mostly did not. A modest change in gradient magnitude is being asked to
explain a large change in result, and it cannot.

What the skip also changes is WHICH FUNCTION IS EASY. In a plain layer the
identity is a particular setting of the weights that has to be found. In a
residual block the identity is what you get when the weights are ZERO -- which is
where initialisation starts and where weight decay pulls (eq:identity-is-default).
The optimiser no longer has to discover how to do nothing.

So the residual connection fixes two things at once, and the one usually quoted
is the smaller one.""")
```

The first listing shows *that* the skip helps and that the usual reason is too
small. The second isolates *what else* it changed.

```python {tier=A name=identity-is-the-default}
"""Why the identity is hard for a plain stack and free for a residual one.

The previous listing showed degradation and showed that gradient magnitude does
not fully explain it. This one isolates the mechanism by giving both
architectures the easiest possible task: reproduce your input.

The identity is the function eq:identity-embedding says a deep network must be
able to express in order for depth to be harmless. Measuring how hard each
architecture finds it separates "can represent" from "can find", which is the
distinction the degradation result turns on.

One detail makes the comparison fair. A ReLU layer computes max(Wh, 0), which
equals h at W = I only when h is non-negative -- so the inputs here are
non-negative, exactly as they would be inside a network after the first
activation. Both architectures can therefore represent the identity EXACTLY, and
any difference in the result is the optimiser's.
"""
import numpy as np

WIDTH = 48
DEPTHS = (2, 4, 8, 16, 32)
N = 384
STEPS, LR = 600, 0.01


def stack_forward(X, Ws, residual, scale):
    h, hs, zs = X, [], []
    for W in Ws:
        hs.append(h)
        z = h @ W
        zs.append(z)
        a = np.maximum(z, 0)
        h = h + scale * a if residual else a
    return h, hs, zs


def init(depth, g):
    return [g.normal(scale=np.sqrt(2 / WIDTH), size=(WIDTH, WIDTH))
            for _ in range(depth)]


def signal_survival(residual, depth, trials=10):
    """At INITIALISATION, how much of the input is still present in the output?

    Mean absolute correlation between an input coordinate and the matching
    output coordinate -- a direct measure of whether the stack has destroyed its
    input before training even starts (eq:signal-decay, eq:signal-survival).
    """
    out = []
    for t in range(trials):
        g = np.random.default_rng(100 + t)
        X = np.abs(g.normal(size=(N, WIDTH)))
        scale = 1.0 / np.sqrt(depth) if residual else 1.0
        Y, _, _ = stack_forward(X, init(depth, g), residual, scale)
        Xc, Yc = X - X.mean(0), Y - Y.mean(0)
        den = np.sqrt((Xc ** 2).sum(0) * (Yc ** 2).sum(0)) + 1e-12
        out.append(float(np.abs((Xc * Yc).sum(0) / den).mean()))
    return float(np.mean(out))


def learn_identity(residual, depth, seed=5):
    """Train the stack to output its input, under an identical budget for both.
    Reported as relative error, so 1.0 means 'no better than predicting zero'.

    Adam rather than plain SGD, deliberately. Gradient magnitudes differ by
    orders of magnitude across these depths, so a single fixed learning rate
    would be testing step-size tuning rather than architecture. An adaptive
    optimiser removes that confound -- and makes the result stronger, since the
    plain stack fails even when the optimiser is choosing its own scale.
    """
    g = np.random.default_rng(seed)
    X = np.abs(g.normal(size=(N, WIDTH)))
    Ws = init(depth, g)
    scale = 1.0 / np.sqrt(depth) if residual else 1.0
    m = [np.zeros_like(W) for W in Ws]
    v = [np.zeros_like(W) for W in Ws]
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, STEPS + 1):
        Y, hs, zs = stack_forward(X, Ws, residual, scale)
        gh = 2.0 * (Y - X) / N
        grads = [None] * depth
        for i in reversed(range(depth)):
            ga = gh * scale if residual else gh
            gz = ga * (zs[i] > 0)
            grads[i] = hs[i].T @ gz
            # Backpropagate through the ORIGINAL weights before updating them.
            gh = gz @ Ws[i].T + (gh if residual else 0)
        for i in range(depth):
            m[i] = b1 * m[i] + (1 - b1) * grads[i]
            v[i] = b2 * v[i] + (1 - b2) * grads[i] ** 2
            mh = m[i] / (1 - b1 ** t)
            vh = v[i] / (1 - b2 ** t)
            Ws[i] -= LR * mh / (np.sqrt(vh) + eps)
    Y, _, _ = stack_forward(X, Ws, residual, scale)
    return float(np.linalg.norm(Y - X) / np.linalg.norm(X))


print(f"width {WIDTH}, {STEPS} gradient steps, identical budget for both.")
print("Inputs are non-negative, so BOTH architectures can represent the "
      "identity exactly.\n")
print(f"{'depth':>7}{'signal at init':>28}{'':>4}"
      f"{'identity error after training':>32}")
print(f"{'':>7}{'plain':>14}{'residual':>14}{'':>4}{'plain':>16}{'residual':>16}")
print("-" * 76)

res = {}
for d in DEPTHS:
    sp, sr = signal_survival(False, d), signal_survival(True, d)
    ip, ir = learn_identity(False, d), learn_identity(True, d)
    res[d] = (sp, sr, ip, ir)
    print(f"{d:>7}{sp:>14.4f}{sr:>14.4f}{'':>4}{ip:>16.4f}{ir:>16.4f}")

d0, dN = DEPTHS[0], DEPTHS[-1]
print(f"""
The left pair is measured before a single gradient step, and it is the cleaner of
the two results. In a plain stack the correlation between an input coordinate and
the matching output coordinate decays with depth -- {res[d0][0]:.4f} at depth
{d0} down to {res[dN][0]:.4f} at depth {dN}. Each layer mixes and rectifies, and
after enough layers the output retains almost no coordinate-wise trace of the
input.

The residual stack starts far higher and decays far more slowly:
{res[d0][1]:.4f} at depth {d0}, {res[dN][1]:.4f} at depth {dN}. The reason is
structural rather than statistical -- the identity path is a TERM IN THE SUM, so
whatever the branches compute, the input is still literally present in the
output. A residual network at initialisation is the identity plus noise; a plain
network at initialisation is noise.

That reframes what depth costs. A plain deep network has to LEARN to preserve
information that its own architecture destroyed by construction. Its optimiser
starts from a function that has thrown the input away, and every useful solution
begins by rebuilding a path back to it.

The right pair puts a number on the consequence, and the comparison is a fair one
because the inputs are non-negative: max(Ih, 0) = h, so a plain stack CAN be the
identity, with W = I at every layer. Asked to find it under a fixed budget, the
plain stack sits at {res[dN][2]:.4f} at depth {dN}, having got worse at every
step down the table, while the residual stack reaches {res[dN][3]:.4f} -- a
factor of {res[dN][2] / res[dN][3]:.1f} between them. Note the plain stack's
depth-2 figure, {res[2][2]:.4f}: it does not find the identity even when there
are only two layers to coordinate.

Nothing separates them but where they start. The residual stack reaches the
identity by driving its branch weights toward ZERO, and zero is where
initialisation already is and where weight decay pulls
(eq:identity-is-default). The plain stack has to find a specific, structured
matrix at every one of {dN} layers simultaneously, through {dN} layers of
rectification, with a gradient that has crossed all of them.

So the two architectures are not competing on capacity -- eq:identity-embedding
holds for both here, by construction. They are competing on where they start and
on what the optimiser must undo. That is why the fix is a wire rather than a
layer, why it costs no parameters, and why it turned out to matter for
transformers and diffusion models too, none of which are convolutional and all of
which are residual.""")
```

## 9. Practical Example

**Degradation, reproduced.** Plain training loss falls from 0.7441 at depth 2 to
**0.6193** at depth 4 — depth helping — then reverses, reaching **1.0308** at
depth 32 with training accuracy down from 0.751 to **0.553**.

These are training numbers. **Nothing here is overfitting**, and by
{{eq:identity-embedding}} nothing here is capacity: the depth-32 network could
become the depth-4 network by setting twenty-eight layers to the identity. The
solution is in the space and the optimiser does not find it.

The residual column is nearly **flat** — 0.8186, 0.7760, 0.7363, 0.7352, 0.7563
across depths 2 to 32. Which is the accurate claim: **skips do not make deep
networks better, they stop depth from making things worse.**

> **IMPORTANT:** The top two rows are what an enthusiastic account omits. At depth
> 4 the *plain* network wins, 0.6193 against 0.7760. {{eq:residual-variance}} says
> the residual stream must be scaled or normalised, and at shallow depth that is
> insurance being paid for and not needed. **The crossover is around depth 8.**
> A residual connection is not universally correct; it is correct when you are
> going deep.

**And the standard explanation is too small.** The gradient ratio — first-layer
gradient with skips over without — rises with depth to **1.8 at depth 32**. Real,
directionally right, and under a factor of two, being asked to explain the
difference between 0.553 and 0.685 training accuracy. {{eq:residual-gradient}} is
correct and insufficient.

**What else changed, measured.** At initialisation, before any training, the
correlation between an input coordinate and its matching output coordinate is
**0.0544** for a 32-layer plain stack and **0.3735** for the residual one — seven
times more signal retained, exactly the geometric-versus-$1/\sqrt{L}$ contrast of
{{eq:signal-decay}} and {{eq:signal-survival}}.

**A plain deep network has thrown its input away before training begins.** Every
useful solution must start by rebuilding a path back to it.

Asked to learn the identity — with non-negative inputs, so
{{eq:identity-embedding}} holds exactly for both and the comparison is fair — the
plain stack reaches **0.7173** relative error at depth 32 against the residual
stack's **0.0947**, a factor of **7.6**. And the plain stack scores **0.3123 at
depth two**: it fails to find the identity with only two layers to coordinate,
using Adam, which is choosing its own step size.

The two architectures are not competing on capacity. **They are competing on where
they start and on what the optimiser has to undo** — which is why the fix is a
wire, costs no parameters, and turned out to matter for transformers and diffusion
models, none of which are convolutional and all of which are residual.

## 10. Production Considerations

**Do not train a backbone.** Start from pretrained weights unless your domain is
genuinely far from natural images, and even then measure the pretrained baseline
first.

**Fine-tune from the top down.** Unfreeze stage 4, then 3, and treat the depth as
a hyperparameter chosen by domain distance.

**Match the preprocessing exactly** ({{ch:mm-cv-fundamentals}}). Silent, and worth
a few points.

**Never use a residual network without normalisation.**
{{eq:residual-variance}} is not a detail; the listing overflowed at depth 32
before the branch scaling was added.

**Choose the tap point deliberately**, not by convention. Stage 3 for detection,
stage 2 as well if small objects matter, and never after the global pool if you
need position.

**Watch for train/eval batch-norm skew.** Batch statistics at train time and
running statistics at eval time diverge under small batches or distribution shift,
and it presents as a mysterious train/test gap that is not overfitting.

**Report the training recipe with every architecture comparison**
({{eq:recipe-versus-architecture}}), or you are reporting their sum.

## 11. Common Mistakes

**Calling degradation overfitting.** The training error went up.

**Believing residual connections are only about vanishing gradients.** The
measured ratio is 1.8 at depth 32.

**Adding skips to a shallow network** and expecting a gain — the measurement says
they cost slightly at depth 4.

**Removing normalisation from a residual stack** "to simplify".

**Comparing architectures across different training recipes.**

**Fine-tuning every layer on a small dataset**, destroying the general early
features that were the reason to transfer.

**Reading ImageNet accuracy as general visual competence.**
{{cite:russakovsky2015ilsvrc}} measured one distribution.

## 12. Failure Modes

**Degradation.** Symptom: deeper is worse *on training loss*. Cause: no residual
path, or one broken by projections at every block.

**Residual explosion.** Symptom: NaNs after a few steps in a deep stack. Cause:
{{eq:residual-variance}} without normalisation.

**Dead stage.** Symptom: a stage's branches all collapse to near-zero output.
Under {{eq:identity-is-default}} the network still works — it has become
shallower. Detect by logging branch output norms per stage.

**Batch-norm distribution skew.** Symptom: excellent training accuracy, poor eval,
and it does not respond to regularisation.

**Transfer failure on distant domains.** Symptom: pretrained initialisation is no
better than random. Diagnose by comparing frozen-feature linear probes against
full fine-tuning.

**Resolution mismatch at fine-tune time.** Symptom: a backbone trained at 224
underperforms at 384 or vice versa; the receptive field relative to object size
changed ({{ch:mm-cv-fundamentals}}).

## 13. Alternatives

| Alternative | What it trades | When it wins |
|---|---|---|
| plain deep CNN | trainability past ~20 layers | never, now |
| DenseNet-style concatenation | memory for feature reuse | small-data regimes |
| ConvNeXt ({{cite:liu2022convnext}}) | nothing much — a modernised ResNet | when you want a convnet with transformer-era recipe |
| ViT ({{ch:mm-vit}}) | the convolutional prior | large data or strong pretraining |
| self-supervised backbone | label supervision | dense tasks; see {{ch:mm-clip}} |
| train from scratch | pretraining's head start | domains far from natural images, with enough data |

**The third row is the one to take seriously as a default.**
{{cite:liu2022convnext}} showed a ResNet with a modern recipe is competitive, so
"use a transformer" is a data-scale decision rather than a correctness one — which
{{ch:mm-vit}} makes precise.

## 14. Evaluation

**Report training loss when investigating depth.** Degradation is invisible in
test metrics.

**Report the recipe.** {{eq:recipe-versus-architecture}}.

**Use linear probes on frozen features** to measure representation quality
separately from fine-tuning capacity — they answer different questions.

**Evaluate on your own distribution, not ImageNet.** Accuracy on
{{cite:russakovsky2015ilsvrc}} predicts transfer imperfectly and predicts your
domain worse.

**Measure per-class and per-object-size accuracy.** Aggregate top-1 hides both the
long tail and {{ch:mm-cv-fundamentals}}'s resolution effects.

## 15. Advanced Concepts

**The unrolled-ensemble view.** {{maturity:MATURE}} {{eq:unrolled-ensemble}}
explains why deleting a block from a trained ResNet barely matters and deleting a
layer from a plain network is fatal — the short paths carry most of the signal.
Stochastic depth is the training-time exploitation of exactly this.

**Normalisation and the residual connection are one design.**
{{maturity:ESTABLISHED}} {{eq:residual-variance}} means the first creates the
problem the second solves. Pre-norm versus post-norm — the same argument that
governs transformer stability — is this equation with the normalisation moved.

**Identity is the right default beyond vision.** {{maturity:ESTABLISHED}}
{{eq:identity-is-default}} is why transformers, diffusion U-Nets and state-space
models are all residual. **The wire generalised; the convolution did not.**

**Recipe as a confound.** {{maturity:MATURE}} {{cite:liu2022convnext}}'s
methodology — change one thing at a time and report the table — is worth more than
its conclusion, and applies to every architecture comparison in this book.

**Scaling depth versus width.** {{maturity:EMERGING}} {{eq:bottleneck-cost}}
makes depth affordable, but the returns are not equal: past a point, width and
resolution buy more than depth. The optimal trade depends on the task's
{{eq:erf-worked}} requirement, and is usually decided by convention rather than
measurement.

## 16. Connection to Previous Chapters

{{ch:mm-cv-fundamentals}}'s stack is what this chapter makes deep, and
{{eq:erf-worked}} is why depth was wanted in the first place.
{{ch:dl-normalization}}'s batch norm is revealed by {{eq:residual-variance}} to be
structurally required rather than a training aid; {{ch:dl-initialization}}
supplies the starting point that {{eq:identity-is-default}} exploits; and
{{ch:dl-optimizers}} is the search whose failure {{eq:degradation}} is about.
Forward: {{ch:mm-detection}} and {{ch:mm-segmentation}} consume this backbone and
spend their architectures recovering the position it discarded;
{{ch:mm-vit}} replaces it while keeping the residual wire; and every transformer
in {{part:07}} was already using {{eq:residual-block}} without calling it that.

## 17. Exercises

1. Prove {{eq:identity-embedding}} for a ReLU network and state the condition on
   the activations that makes it valid.
2. Derive {{eq:residual-gradient}} and identify the term that is absent in a plain
   network.
3. Derive {{eq:residual-variance}} and compute the stream's scale at depth 100.
   What normalisation placement keeps it bounded?
4. In `degradation-not-vanishing`, remove the branch scaling. At what depth does
   it diverge, and does that match {{eq:residual-variance}}?
5. In the same listing, sweep the learning rate for the plain depth-32 network.
   Can tuning alone eliminate the degradation?
6. In `identity-is-the-default`, initialise the plain stack's weights at $I$
   rather than randomly. What does that do to the identity error, and what does
   it tell you about {{eq:identity-embedding}}?
7. Verify {{eq:bottleneck-cost}} for $r = 2$ and $r = 8$. Why is 4 the usual
   choice?
8. Take a pretrained ResNet. Measure linear-probe accuracy on your own task from
   each of the four stages, and relate the pattern to
   {{sec:7-internal-mechanics}}'s transfer discussion.

## 18. Interview Questions

1. What is the degradation problem, and why is it not overfitting?
2. Why does {{eq:identity-embedding}} make degradation surprising?
3. Explain a residual connection in one sentence, then in two mechanisms.
4. Why is "it fixes vanishing gradients" incomplete?
5. Why do residual networks need normalisation?
6. What is a bottleneck block and what does it save?
7. Why does deleting one block from a trained ResNet barely change the output?
8. Which stage of a backbone would you tap for small-object detection, and why?
9. A comparison shows a ViT beating a ResNet. What do you ask first?
10. When is a residual connection not worth adding?

## 19. Research Questions

1. {{sec:9-practical-example}} finds a gradient ratio of 1.8 explaining a large
   outcome gap. Can the two effects — gradient flow and initialisation-near-identity
   — be separated experimentally, for example by initialising a plain network near
   the identity?
2. {{eq:residual-identity-error}} predicts depth-independent error under a fixed
   budget. What is the right budget scaling to make the residual column exactly
   flat?
3. {{eq:unrolled-ensemble}} says short paths dominate. Is there a training
   objective that deliberately lengthens the effective path distribution, and does
   it help?
4. {{cite:liu2022convnext}} attributed much of the gap to recipe. What is the
   residual architectural difference once recipe is fully controlled, and how does
   it scale with data?
5. Transfer degrades with domain distance. Is there a cheap predictor of how many
   stages to unfreeze, computable before fine-tuning?

## 20. Chapter Summary

**Degradation is an optimisation result, not a capacity or overfitting result.**
{{eq:identity-embedding}} says a deeper network contains a shallower one, so its
best achievable training loss cannot be worse — and {{eq:degradation}} says the
achieved loss is. Measured: plain training loss 0.6193 at depth 4 rising to
**1.0308** at depth 32, training accuracy 0.751 falling to **0.553**. The solution
is in the space and the search does not reach it.

**The residual connection is a wire, and it does two things.** The quoted one is
gradient flow ({{eq:residual-gradient}}), and the measurement puts it at **1.8×**
at depth 32 — real, and too small to carry the explanation. The other is that
{{eq:identity-is-default}} makes *doing nothing* the default: at initialisation a
residual stack retains **0.3735** correlation with its input against a plain
stack's **0.0544**, and asked to learn the identity it reaches **0.0947** against
**0.7173**, a factor of 7.6, on a task both can represent exactly.

**A plain deep network destroys its input by construction and must learn to
rebuild it.** That, rather than gradient magnitude, is what depth costs.

**The residual column is flat, not high** — 0.8186 to 0.7563 across depths 2 to
32. Skips do not make deep networks better; they stop depth from making things
worse. And **they cost something**: at depth 4 the plain network wins, because
{{eq:residual-variance}} forces the branch to be scaled and shallow networks do
not need the protection. The crossover measured here is around depth 8.

**Residual and normalisation are one design**, not two tricks:
{{eq:residual-variance}} says the stream grows as $\sqrt{L}$, and the listing
overflowed before the scaling was added.

**And the wire generalised where the convolution did not.** Transformers,
diffusion denoisers, and state-space models are all residual and none are
convolutional — which is why this chapter is worth reading even by someone who
will never train a CNN. {{cite:liu2022convnext}} adds the methodological coda:
much of the apparent gap between architectures was training recipe
({{eq:recipe-versus-architecture}}), so a comparison that varies both measures
their sum and reports it as the first.

## 21. Further Reading

{{cite:he2016resnet}} for the original, and read Section 3.1 for the degradation
experiment itself, which is more interesting than the architecture it motivated.
{{cite:simonyan2015vgg}} and {{cite:krizhevsky2012}} for what depth had bought
before it stopped working.
{{cite:ioffe2015}} for normalisation, which {{eq:residual-variance}} shows is
structurally required in a residual stack rather than merely helpful.
{{cite:liu2022convnext}} for the ablation that separates architecture from recipe,
and as a methodological model.
{{cite:dosovitskiy2021vit}} for the architecture that discards the convolutional
prior and keeps the wire — developed in {{ch:mm-vit}}.
{{cite:russakovsky2015ilsvrc}} for what the benchmark all of this was optimised
against actually measured.

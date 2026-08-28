---
id: mm-vit
number: 122
part: XIII
tier: full
status: draft
requires: [mm-cv-fundamentals, mm-classification, tf-scaled-dot-product,
           tf-positional, tf-complexity]
provides: [patchification, patch-compression-ratio, vit-cost-model,
           position-embeddings-for-images, prior-as-data-equivalence,
           invariance-forbids, hybrid-architectures]
citations: [dosovitskiy2021vit, liu2022convnext, oquab2023dinov2,
            carion2020detr, he2016resnet, vaswani2017]
---

## 1. Learning Objectives

By the end of this chapter you will be able to describe patchification as the
*only* image-specific decision a ViT makes, and derive from patch size the token
count, the compute, and what the model is structurally unable to resolve; explain
why the standard $16\times16$ patch with $d = 768$ is the configuration where the
patch embedding is **dimensionally lossless**, and what changes when it is not;
state when "attention is quadratic" is true and when it is a minority of the cost;
quantify the convolutional prior as an equivalent number of training examples;
and — the half usually omitted — demonstrate what that prior **forbids**.

## 2. Why This Matters

A vision transformer does almost nothing image-specific. It cuts the image into
patches, projects each into a vector, adds a position embedding, and hands the
result to the architecture from {{part:07}} unchanged.

**That one decision — the patch size — sets everything else.** It fixes the token
count, and therefore the compute; it fixes the compression ratio at the input, and
therefore the finest detail that can survive to the model at all; and it fixes the
resolution at which the model can localise anything.

{{sec:9-practical-example}} makes those exact rather than approximate. At the
standard configuration a patch holds $16 \times 16 \times 3 = 768$ raw values and
the embedding width is 768 — a compression ratio of exactly **1.00**. The
canonical ViT is the one whose input projection is a change of basis rather than a
summary, and that is not a coincidence. At patch 32 the ratio is **4.00**, and
three quarters of what is inside each patch is discarded before any attention
runs.

The second measurement settles a claim that is quoted without its condition.
{{cite:dosovitskiy2021vit}} says a ViT beats a convnet *given enough data*, and
below that it does not. Measured on a translation-invariant task: at 100 training
examples a small convnet scores **1.000** and the transformer **0.334** — chance.
At 6400 the transformer has reached 0.672 and is still climbing.

**And the half that gets left out**: on a task about *position*, the convnet
scores **0.495, 0.528, 0.503, 0.514** across a 64-fold increase in data. That is
chance at every size, and it is not a data problem — {{eq:pooling-invariance}}
discarded the information the task depends on. The transformer reaches **1.000**.

**A prior is a constraint. It buys sample efficiency inside its scope and costs
everything outside it.**

{{maturity:ESTABLISHED}} ViTs as backbones. {{maturity:MATURE}} Hybrid
convolution-plus-attention designs, which keep reappearing for the reason this
chapter measures.

## 3. Prerequisites

{{ch:tf-scaled-dot-product}} and {{ch:tf-positional}} for attention and position
encoding — this chapter adds no new transformer machinery, only a tokeniser;
{{ch:tf-complexity}} for the quadratic term, whose *regime* this chapter pins
down; {{ch:mm-cv-fundamentals}} for {{eq:translation-equivariance}} and
{{eq:pooling-invariance}}, which is what a ViT gives up;
{{ch:mm-classification}} for the baseline it is compared against.

## 4. Intuitive Explanation

### An image is a sequence if you cut it up

Split a $224 \times 224$ image into $16 \times 16$ patches: 196 of them. Flatten
each to a 768-vector, project, add a position embedding, and you have a sequence
of 196 tokens. **Every remaining component is the transformer you already know.**

The elegance is real and so is the consequence: **the model has no idea that
patch 5 is next to patch 6.** A convolution has adjacency built into its
structure; a transformer must learn it from the position embeddings, from data.

That is the trade in one sentence, and both directions of it matter.

### The patch embedding is where the detail goes

A patch of $p \times p \times C$ raw values becomes one $d$-dimensional vector. If
$p^2 C > d$, information is destroyed — permanently, before the first attention
layer.

$$ p = 16,\; C = 3,\; d = 768 \;\Longrightarrow\; \tfrac{768}{768} = 1.00 $$

**The standard configuration sits exactly at the lossless point.** At patch 32 the
ratio is 4.00 and three quarters of each patch is gone.

This is the single most useful fact for predicting VLM failures. Below the patch
grid, the model does not see pixels — it sees whatever survived one linear
projection. **Small text on a document, tick labels on a chart, a distant sign: if
the stroke is thin relative to the patch, the evidence is gone before the model
starts**, and no amount of language model behind it recovers what was discarded.

### Why not just use smaller patches

Because {{eq:vit-attention-cost}} is quadratic in the token count and the token
count is quadratic in $1/p$. Halving the patch **quadruples** tokens and
**sixteen-folds** attention cost. {{sec:9-practical-example}} measures 0.06 → 0.94
GFLOPs for that one change.

### When attention is and is not the bottleneck

Here is a claim that is repeated as though it were unconditional, and
{{sec:9-practical-example}} shows the condition:

> At $224$ pixels, attention is **9.3%** of a ViT's FLOPs. At $1024$ pixels, it is
> **68.1%**.

Same architecture, same patch size. Profile a ViT at benchmark resolution and you
will correctly conclude attention is not the bottleneck — and generalising that
is wrong, because attention is the only term growing quadratically while
everything else grows linearly in token count.

**"Attention is quadratic" is a statement about a regime.** It is false where ViTs
are usually measured and dominant where documents live, which is why
{{ch:mm-vlms}} has a token-budget problem at all.

### What the prior is worth, and what it costs

The convolutional prior says two things: features are local, and a feature means
the same thing everywhere. A transformer assumes neither and must learn both.

{{sec:9-practical-example}} prices that in examples. On a task the prior fits, the
convnet is at 1.000 with 100 examples while the transformer is at chance, and the
transformer needs roughly 64× more data to become competitive.

**And then the second task.** Asked which half of the image the shape is in — a
question *about position* — the convnet sits at chance across every training size,
because its global pool discarded position by construction. More data does not
help. More depth does not help. **The architecture cannot express a function of
something it threw away.**

That is what "prior" means in both directions, and it is why hybrids keep
reappearing: a convolutional stem for cheap local structure, attention above for
the relations a convolution cannot represent.

## 5. Formal Explanation

### 5.1 Patchification

$$ x \in \mathbb{R}^{H \times W \times C} \;\longrightarrow\; \{x_i\}_{i=1}^{N},\; x_i \in \mathbb{R}^{p^2 C}, \qquad N = \frac{HW}{p^2} $$ (eq:patchify)

$$ z_i = x_i E + e_{\text{pos}}(i), \qquad E \in \mathbb{R}^{p^2C \times d} $$ (eq:patch-embedding)

{{eq:patch-embedding}} is a convolution with kernel $p$ and stride $p$ — the
tokeniser is a single, non-overlapping convolutional layer, and that is the entire
convolutional content of a ViT.

### 5.2 The compression ratio

$$ \rho = \frac{p^2 C}{d} $$ (eq:patch-compression)

- $\rho \le 1$: the projection can be injective. No information is lost by
  dimension count.
- $\rho > 1$: information is destroyed, unrecoverably, before layer one.

At the canonical $p=16$, $C=3$, $d=768$: $\rho = 1.00$ exactly. **The standard
configuration is the largest patch that is still dimensionally lossless at that
width**, which is a design choice rather than a coincidence.

### 5.3 Cost

$$ \text{FLOPs} \approx \underbrace{N p^2 C d}_{\text{patch embed} \,=\, HWCd,\ \text{constant in } p} + \underbrace{4Nd^2}_{\text{projections}} + \underbrace{2N^2 d}_{\text{attention}} $$ (eq:vit-attention-cost)

Three readings, all confirmed by measurement:

- **The patch-embedding term does not depend on $p$**, since $Np^2 = HW$. It is
  fixed by image size.
- **Projections are linear in $N$**, attention is **quadratic**.
- The attention share is therefore $\Theta(N)$ relative to the rest:

$$ \frac{\text{attention}}{\text{total}} \approx \frac{2N^2d}{HWCd + 4Nd^2 + 2N^2d} \;\xrightarrow[N \to \infty]{}\; 1 $$ (eq:attention-share)

{{eq:attention-share}} is why the regime matters: at $N=196$ it evaluates to
**9.3%** and at $N=4096$ to **68.1%**.

### 5.4 Position embeddings are not optional

Self-attention is **permutation-equivariant**: for any permutation $\pi$,

$$ \text{Attn}(\pi Z) = \pi\,\text{Attn}(Z) $$ (eq:attention-permutation)

so without position information a ViT computes a function of the *bag* of
patches. Shuffle the patches of an image and the prediction is unchanged.

**Compare {{eq:translation-equivariance}}.** A convolution's equivariance is a
structural guarantee about *shifts*; attention's is an unwanted invariance to
*arbitrary permutations*, which the position embedding exists to break. Two very
different facts that both get called "equivariance".

Note the asymmetry: convolution's prior cannot be turned off, and attention's
position embedding is *learned*, so a ViT can learn to be translation-invariant
where that helps and not where it does not. That flexibility is exactly what
{{sec:9-practical-example}}'s second task rewards.

### 5.5 Prior as data

Write $n_{\text{eff}}$ for the training-set size at which an unconstrained model
matches a constrained one:

$$ \text{prior} \;\equiv\; n_{\text{eff}} \text{ examples}, \qquad \text{gap}(n) \searrow 0 \text{ as } n \to n_{\text{eff}} $$ (eq:prior-as-data)

**{{eq:prior-as-data}} is the honest form of {{cite:dosovitskiy2021vit}}'s claim.**
A prior is not better or worse than data; it is a substitute for it, at an
exchange rate set by how well the prior matches the task.

### 5.6 What a prior forbids

If a model's representation $\phi$ is invariant under a group $G$, then for any
$g \in G$:

$$ \phi(gx) = \phi(x) \;\Longrightarrow\; f(gx) = f(x) \quad \text{for every head } f $$ (eq:invariance-forbids)

So **no task whose label distinguishes $x$ from $gx$ is learnable**, at any data
scale, with any head. {{eq:pooling-invariance}} makes a globally pooled convnet
invariant to translation, so "which half is the object in" is outside its
hypothesis space entirely.

{{sec:9-practical-example}} confirms this is not a soft effect: **0.495, 0.528,
0.503, 0.514** across a 64× data increase. Flat, at chance, exactly as
{{eq:invariance-forbids}} requires.

## 6. Mathematical Foundation

### 6.1 The cost of halving the patch

From {{eq:vit-attention-cost}}, going $p \to p/2$ gives $N \to 4N$:

$$ \text{embed} \to \text{embed}, \qquad \text{proj} \to 4 \times, \qquad \text{attn} \to 16 \times $$ (eq:halving-patch)

Measured at $224$: attention 0.06 → **0.94** GFLOPs (16×, exactly), projections
0.46 → **1.85** (4×, exactly), embedding 0.12 → **0.12** (unchanged, exactly), and
total 0.64 → **2.91**, a factor of **4.5**.

**Four times the spatial detail for 4.5 times the compute** — which sounds
tolerable until you notice the exponent is on the term that grows.

### 6.2 Resolution versus patch size: the same token count, different costs

Compare $224/8$ against $448/16$. Both give $N = 784$:

| | $224$, $p{=}8$ | $448$, $p{=}16$ |
|---|---|---|
| tokens | 784 | 784 |
| compression $\rho$ | **0.25** | **1.00** |
| embed GFLOPs | 0.12 | 0.46 |
| total GFLOPs | 2.91 | 3.26 |

**Same attention cost, and very different inputs.** The $p=8$ configuration
oversamples a small image — $\rho = 0.25$ means the projection has four times the
dimensions it needs. The $448$ configuration feeds four times as many *pixels* at
the lossless ratio.

For reading a document, the second is what you want, and it is why the answer to
"the model cannot read the small print" is almost always *increase the input
resolution*, not *decrease the patch size*.

> **MATH NOTE:** {{eq:patch-compression}} is a *dimension* count, not an
> information-theoretic guarantee. $\rho \le 1$ says the projection *can* be
> injective; whether the trained $E$ actually preserves the detail that matters is
> a separate question, and a learned $E$ may discard high-frequency content
> regardless. So $\rho \le 1$ is necessary and not sufficient — which is the safe
> direction for a design rule, since $\rho > 1$ is *sufficient for loss*.

### 6.3 The exchange rate, from the measurement

On task A, the convnet is at 1.000 from 100 examples. The transformer goes
$0.334 \to 0.328 \to 0.328 \to 0.672$ across 100, 400, 1600, 6400 — flat at chance
until somewhere past 1600, then rising steeply.

That shape is characteristic and worth recognising: **the transformer is not
gradually approximating the prior, it is failing to learn anything until it has
enough data to discover the regularity, then learning quickly.**
{{eq:prior-as-data}}'s $n_{\text{eff}}$ here is well above 6400 for a task a
convnet solves with 100 examples — an exchange rate of at least 64×, on a task
whose entire content is the prior.

That ratio is why {{cite:dosovitskiy2021vit}} needed JFT-300M to make its point,
and why the same architecture underperforms on ImageNet alone.

## 7. Internal Mechanics

```mermaid {#fig:vit-pipeline caption="A ViT is a tokeniser followed by the transformer of Part VII. The only image-specific decisions are in the dashed box, and both are consequences of the patch size: how many tokens (hence eq:vit-attention-cost) and how much of each patch survives (eq:patch-compression). Everything below the box is unchanged from text."}
flowchart TB
    IM["image H x W x C"] --> PT["cut into p x p patches<br/>N = HW/p^2"]
    PT --> PE["linear projection to d<br/>compression p^2C/d"]
    PE --> POS["+ learned position embedding<br/>(required: attention is<br/>permutation-equivariant)"]
    POS --> TR["transformer blocks<br/>(unchanged from Part VII)"]
    TR --> HD["pool or CLS token"] --> OUT["prediction"]
    PT -.-> DEC["THE decision:<br/>patch size p"]
    PE -.-> DEC
```

### 7.1 What the patch size decides, in one table

| Choice | Tokens at 224 | $\rho$ at $d{=}768$ | Attention share | Finest detail |
|---|---|---|---|---|
| $p = 32$ | 49 | 4.00 | 1.6% | quarter of the patch survives |
| $p = 16$ | 196 | 1.00 | 9.3% | lossless projection |
| $p = 14$ | 256 | 0.77 | 12.3% | oversampled |
| $p = 8$ | 784 | 0.25 | 32.5% | heavily oversampled |

**Read the $\rho$ column as a warning about $p = 32$ specifically.** It is the
cheapest configuration and it throws away three quarters of every patch, which is
why $32$-patch models are poor at anything requiring fine detail regardless of how
large the model behind them is.

### 7.2 CLS token or mean pool

Two ways to get one vector from $N$: prepend a learned token and read it out, or
average. Both work; the CLS token is the transformer convention inherited from
BERT, mean pooling is simpler and often equivalent.

**What matters more is that either discards position** — the same
{{eq:pooling-invariance}} choice as a convnet's global pool, and the same
consequence. Dense tasks read the patch tokens directly, which is exactly what
{{cite:carion2020detr}} does.

### 7.3 Why hybrids keep coming back

A convolutional stem for the first stage and attention above is a recurring
design, and {{sec:9-practical-example}} explains why in one line: **the prior is
right about low-level vision and wrong about long-range relations.** Edges and
textures genuinely are local and translation-invariant; "which of these is left of
the other" genuinely is not.

Using the cheap constrained operator where its assumption holds and the expensive
unconstrained one where it does not is not a compromise. It is matching each part
of the problem to an operator whose assumptions fit it.

### 7.4 Interpolating position embeddings

A practical detail that causes real bugs. Position embeddings are learned per
grid position, so a model trained at $224$ ($14 \times 14$ patches) has 196 of
them. Run it at $448$ and you need 784.

The standard fix is to **bicubically interpolate the embedding grid**, which works
because position embeddings are spatially smooth. Forgetting it produces a model
that silently degrades at a resolution it appears to accept.

## 8. Implementation

```python {tier=A name=patch-size-economics}
"""Patch size: the one hyperparameter that sets everything else about a ViT.

A vision transformer's only real decision about images is how to cut them into
tokens. Everything downstream -- how much it can see, what it costs, what it is
structurally unable to resolve -- follows from the patch size, and the
relationships are exact rather than empirical.

Three quantities matter and they pull in opposite directions:
  - token count grows as (H/p)^2, and attention costs its square
    (eq:vit-attention-cost);
  - the patch embedding compresses p*p*C numbers into d, so the compression
    ratio is p^2 C / d (eq:patch-compression);
  - anything finer than a patch has to survive that compression to be
    representable at all.

This listing computes all three across the configurations people actually use.
"""
import numpy as np

C, D = 3, 768                   # input channels, embedding width
HEADS = 12


def config(img, patch):
    n = (img // patch) ** 2                       # tokens (ignoring CLS)
    # Patch embedding: one matmul of (n x p^2 C) by (p^2 C x d).
    embed_flops = n * (patch ** 2 * C) * D
    # Attention: QKV projections, the n x n score matrix, and the value mix.
    proj = 4 * n * D * D
    attn = 2 * n * n * D
    return {
        "tokens": n,
        "compress": (patch ** 2 * C) / D,
        "embed_gf": embed_flops / 1e9,
        "attn_gf": attn / 1e9,
        "proj_gf": proj / 1e9,
        "total_gf": (embed_flops + proj + attn) / 1e9,
        "attn_share": attn / (embed_flops + proj + attn),
    }


print(f"embedding width d = {D}, {C} input channels\n")
print(f"{'image':>7}{'patch':>7}{'tokens':>8}{'p^2C/d':>9}{'embed GF':>10}"
      f"{'proj GF':>9}{'attn GF':>9}{'total GF':>10}{'attn share':>12}")
print("-" * 81)

rows = {}
for img, patch in ((224, 32), (224, 16), (224, 14), (224, 8),
                   (448, 16), (896, 16), (1024, 16)):
    r = config(img, patch)
    rows[(img, patch)] = r
    print(f"{img:>7}{patch:>7}{r['tokens']:>8}{r['compress']:>9.2f}"
          f"{r['embed_gf']:>10.2f}{r['proj_gf']:>9.2f}{r['attn_gf']:>9.2f}"
          f"{r['total_gf']:>10.2f}{r['attn_share']:>12.1%}")

a, b = rows[(224, 16)], rows[(224, 8)]
c, e = rows[(224, 16)], rows[(1024, 16)]
print(f"""
Start with the compression column, because it explains a number everyone treats
as arbitrary. At patch 16 with three channels, a patch holds 16*16*3 = 768 raw
values and the embedding has width 768 -- a ratio of exactly
{rows[(224, 16)]['compress']:.2f}. The standard configuration is the one where
the patch embedding is dimensionally lossless: it is a change of basis, not a
summary. At patch 32 the ratio is {rows[(224, 32)]['compress']:.2f}, so three
quarters of the information inside each patch has to be discarded before a single
attention operation runs.

That is the first thing to know about a vision tower: below the patch grid, it
does not see pixels, it sees whatever survived one linear projection. Text on a
document, a distant road sign, the tick labels on a chart -- if the stroke is thin
relative to the patch and the projection is compressive, the evidence is gone
before the model starts, and no amount of attention recovers it.

Now the cost columns, and the reason patch 16 is not simply replaced by patch 8.
Halving the patch quadruples the token count, {a['tokens']} to {b['tokens']}, and
attention cost goes as the SQUARE of that: {a['attn_gf']:.2f} GFLOPs to
{b['attn_gf']:.2f}, a factor of {b['attn_gf'] / a['attn_gf']:.0f}
(eq:vit-attention-cost). Total cost rises by
{b['total_gf'] / a['total_gf']:.1f}x. Four times the detail for
{b['total_gf'] / a['total_gf']:.0f} times the compute, and the exponent is what
makes fine patches unaffordable rather than merely expensive.

Read the attention-share column down the last three rows, because it settles a
common misconception. At 224 pixels attention is a MINORITY of the cost --
{c['attn_share']:.0%} -- and the projections and patch embedding dominate. People
who profile a ViT at standard resolution correctly conclude that attention is not
the bottleneck, and then generalise it. At 1024 pixels the same architecture
spends {e['attn_share']:.0%} of its FLOPs inside attention, because that term is
the only one growing quadratically while everything else grows linearly in token
count.

So "attention is quadratic" is a statement about a REGIME, not about a model. It
is false at the resolution ViTs are usually benchmarked at and dominant at the
resolution documents need -- which is exactly why high-resolution vision is where
the efficient-attention literature (ch:tf-efficient) actually earns its keep, and
why ch:mm-vlms has to solve the token-budget problem before it can read a page.""")
```

The first listing is about what the architecture costs. The second is about what
its assumptions are worth, and what they cost.

```python {tier=A name=prior-versus-data}
"""What the convolutional prior buys, and what it forbids.

cite:dosovitskiy2021vit's claim is routinely quoted without its condition. A
transformer over image patches beats a convolutional network GIVEN ENOUGH DATA,
and underperforms it below that, because the convolution's locality and
translation-equivariance are assumptions the transformer has to learn from
examples instead (eq:prior-as-data).

An assumption is not free in either direction. This listing runs two tasks:

  TASK A  which shape is present?   -- translation-invariant, exactly what
                                       eq:pooling-invariance was built for
  TASK B  which half is it in?      -- a question ABOUT position

A conv net with global pooling is invariant to translation by construction, so on
task B it is not merely worse, it is structurally incapable: eq:invariance-forbids
says its output cannot depend on something its representation discarded. That is
the cost of a prior, and it is usually left out of the comparison.
"""
import numpy as np

rng = np.random.default_rng(41)

H = 24
P = 4                     # patch side for the transformer
T = (H // P) ** 2         # tokens
DP = P * P                # raw patch dimension
DM = 24                   # model width
K = 5                     # conv kernel / shape template size
HID = 24

TPL = np.zeros((3, K, K))
TPL[0, 2, :] = 1; TPL[0, :, 2] = 1            # cross
TPL[1, 0, :] = 1; TPL[1, :, 0] = 1            # corner
TPL[2, 1:4, 1:4] = 1                          # block


def place(img, t, r, c):
    img[r:r + K, c:c + K] = np.maximum(img[r:r + K, c:c + K], TPL[t])


def task_a(n):
    """Which shape is present? One shape, anywhere."""
    X = np.zeros((n, H, H)); y = rng.integers(0, 3, size=n)
    for i in range(n):
        r, c = rng.integers(0, H - K + 1, size=2)
        place(X[i], y[i], r, c)
    return X + 0.06 * rng.normal(size=X.shape), y


def task_b(n):
    """Which half of the image is the shape in?

    One shape, its TYPE chosen at random and independent of the label, so the
    two classes contain exactly the same distribution of image CONTENT and
    differ only in where that content sits. A model whose features are
    invariant to translation therefore cannot separate them at all -- not
    poorly, but provably at chance, because eq:pooling-invariance says its
    representation is identical for the two classes.
    """
    X = np.zeros((n, H, H)); y = np.zeros(n, dtype=int)
    for i in range(n):
        t = int(rng.integers(0, 3))
        left = int(rng.integers(0, 2))
        # Both bands sit well inside the frame. That matters: a shape near an
        # edge produces a different set of PARTIAL window views than the same
        # shape near the opposite edge, so translation equivariance is only
        # exact away from the borders (ch:mm-cv-fundamentals). Keeping both
        # bands interior removes that leak, so the only remaining difference
        # between the classes really is position.
        c = int(rng.integers(4, 7)) if left else int(rng.integers(H - K - 6, H - K - 3))
        r = int(rng.integers(4, H - K - 3))
        place(X[i], t, r, c)
        y[i] = left
    return X + 0.06 * rng.normal(size=X.shape), y


def softmax_ce(logits, y):
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
    return -np.log(p[np.arange(len(y)), y] + 1e-12).mean(), \
        (p - np.eye(p.shape[1])[y]) / len(y)


class Conv:
    """Shared filters at every position, then a global max over positions --
    translation-equivariant then translation-INVARIANT (eq:pooling-invariance)."""

    def __init__(self, ncls):
        self.F = rng.normal(scale=np.sqrt(2 / (K * K)), size=(K * K, HID))
        self.bf = np.zeros(HID)
        self.W = rng.normal(scale=np.sqrt(2 / HID), size=(HID, ncls))
        self.b = np.zeros(ncls)

    def patches(self, X):
        n, S = len(X), H - K + 1
        out = np.empty((n, S * S, K * K))
        for i in range(S):
            for j in range(S):
                out[:, i * S + j] = X[:, i:i + K, j:j + K].reshape(n, -1)
        return out

    def forward(self, X):
        self.p = self.patches(X)
        self.a = np.maximum(self.p @ self.F + self.bf, 0)
        self.arg = self.a.argmax(axis=1)
        self.h = np.take_along_axis(self.a, self.arg[:, None, :], axis=1)[:, 0]
        return self.h @ self.W + self.b

    def step(self, g, lr):
        gW, gb = self.h.T @ g, g.sum(0)
        gh = (g @ self.W.T) * (self.h > 0)
        gF = np.zeros_like(self.F)
        for c in range(HID):
            gF[:, c] = self.p[np.arange(len(gh)), self.arg[:, c]].T @ gh[:, c]
        self.F -= lr * gF; self.bf -= lr * gh.sum(0)
        self.W -= lr * gW; self.b -= lr * gb


class ViT:
    """Patch embed, learned position embeddings, one self-attention block, mean
    pool. No locality prior and no translation equivariance -- position is an
    explicit learned input rather than an architectural assumption."""

    def __init__(self, ncls):
        s = np.sqrt(2 / DP)
        self.We = rng.normal(scale=s, size=(DP, DM)); self.be = np.zeros(DM)
        self.Pe = rng.normal(scale=0.1, size=(T, DM))
        sm = np.sqrt(1 / DM)
        self.Wq = rng.normal(scale=sm, size=(DM, DM))
        self.Wk = rng.normal(scale=sm, size=(DM, DM))
        self.Wv = rng.normal(scale=sm, size=(DM, DM))
        self.Wo = rng.normal(scale=np.sqrt(2 / DM), size=(DM, ncls))
        self.bo = np.zeros(ncls)

    def tokens(self, X):
        n = len(X)
        g = H // P
        out = np.empty((n, T, DP))
        for i in range(g):
            for j in range(g):
                out[:, i * g + j] = X[:, i*P:(i+1)*P, j*P:(j+1)*P].reshape(n, -1)
        return out

    def forward(self, X):
        self.tk = self.tokens(X)
        self.Z = self.tk @ self.We + self.be + self.Pe
        self.Q, self.Kx, self.V = self.Z @ self.Wq, self.Z @ self.Wk, self.Z @ self.Wv
        S = self.Q @ self.Kx.transpose(0, 2, 1) / np.sqrt(DM)
        S -= S.max(axis=-1, keepdims=True)
        A = np.exp(S); self.A = A / A.sum(axis=-1, keepdims=True)
        self.O = self.A @ self.V
        self.Hh = self.Z + self.O
        self.pool = self.Hh.mean(axis=1)
        return self.pool @ self.Wo + self.bo

    def step(self, g, lr):
        gWo, gbo = self.pool.T @ g, g.sum(0)
        gH = np.repeat((g @ self.Wo.T)[:, None, :], T, axis=1) / T
        gZ, gO = gH.copy(), gH.copy()
        gA = gO @ self.V.transpose(0, 2, 1)
        gV = self.A.transpose(0, 2, 1) @ gO
        gS = self.A * (gA - (gA * self.A).sum(axis=-1, keepdims=True))
        gS /= np.sqrt(DM)
        gQ = gS @ self.Kx
        gK = gS.transpose(0, 2, 1) @ self.Q
        gWq = (self.Z.transpose(0, 2, 1) @ gQ).sum(0)
        gWk = (self.Z.transpose(0, 2, 1) @ gK).sum(0)
        gWv = (self.Z.transpose(0, 2, 1) @ gV).sum(0)
        gZ += gQ @ self.Wq.T + gK @ self.Wk.T + gV @ self.Wv.T
        gWe = (self.tk.transpose(0, 2, 1) @ gZ).sum(0)
        gbe = gZ.sum(axis=(0, 1))
        gPe = gZ.sum(axis=0)
        for p, gp in ((self.Wo, gWo), (self.bo, gbo), (self.Wq, gWq),
                      (self.Wk, gWk), (self.Wv, gWv), (self.We, gWe),
                      (self.be, gbe), (self.Pe, gPe)):
            p -= lr * gp


def run(model_cls, task, n_train, ncls, epochs=40, lr=0.05):
    Xtr, ytr = task(n_train)
    Xte, yte = task(1200)
    m = model_cls(ncls)
    for _ in range(epochs):
        order = rng.permutation(n_train)
        for s in range(0, n_train, 32):
            b = order[s:s + 32]
            _, g = softmax_ce(m.forward(Xtr[b]), ytr[b])
            m.step(g, lr)
    return float((m.forward(Xte).argmax(1) == yte).mean())


SIZES = (100, 400, 1600, 6400)
print(f"{H}x{H} images. Conv: shared {K}x{K} filters + global max pool.")
print(f"ViT: {P}x{P} patches, learned positions, 1 attention block, mean pool.\n")
print(f"{'train size':>11}{'':>3}{'TASK A: which shape':>26}{'':>4}"
      f"{'TASK B: which half':>26}")
print(f"{'':>11}{'':>3}{'conv':>12}{'ViT':>14}{'':>4}{'conv':>12}{'ViT':>14}")
print("-" * 74)

res = {}
for n in SIZES:
    a_c = run(Conv, task_a, n, 3)
    a_v = run(ViT, task_a, n, 3)
    b_c = run(Conv, task_b, n, 2)
    b_v = run(ViT, task_b, n, 2)
    res[n] = (a_c, a_v, b_c, b_v)
    print(f"{n:>11}{'':>3}{a_c:>12.3f}{a_v:>14.3f}{'':>4}{b_c:>12.3f}{b_v:>14.3f}")

lo, hi = SIZES[0], SIZES[-1]
print(f"""
Task A is the case the convolutional prior was designed for, and the sample
efficiency shows. At {lo} training examples the conv model reaches
{res[lo][0]:.3f} while the transformer manages {res[lo][1]:.3f} -- chance for a
three-way choice. The conv model already knows that a shape means the same thing
wherever it appears; the transformer is still learning that from examples, and by
{hi} examples it has reached {res[hi][1]:.3f} and is still climbing.

Look at the SHAPE of the transformer's column, because it is characteristic. It
does not improve gradually from 100 to 1600 -- it sits at chance, learning
nothing, and then rises steeply. It is not slowly approximating the prior; it is
failing until it has enough data to discover the regularity, then learning
quickly. eq:prior-as-data's exchange rate here is at least 64x, on a task whose
entire content IS the prior.

That is cite:dosovitskiy2021vit's claim with its condition attached, and the
condition is the part that decides which architecture to use. Quoting
"transformers beat convnets" without the data scale drops the only operative
detail.

Task B is what the prior FORBIDS, and it is the half of the comparison usually
missing. The question is about position -- which half of the frame the shape is
in -- and the shape's TYPE is random and independent of the label, so the two
classes contain identical image content and differ only in where it sits. The
conv model's global max pool discarded position by construction, so
eq:invariance-forbids says the two classes have identical representations.

The measurement is flat at chance -- {res[lo][2]:.3f}, {res[400][2]:.3f},
{res[1600][2]:.3f}, {res[hi][2]:.3f} -- across a 64-fold increase in training
data. That is not a model that needs more examples. More data cannot help, more
depth cannot help, and a better optimiser cannot help, because the function is
outside the hypothesis space. The transformer reaches {res[hi][3]:.3f} on the
same task with the same budget, because its position embeddings are an INPUT
rather than an assumption: it can learn to be translation-invariant when that is
right, and learn not to be when it is not.

So the trade is not "prior versus no prior, with data deciding". It is: a prior is
a constraint; constraints buy sample efficiency inside their scope and cost
everything outside it; and the transformer's advantage at scale is partly that it
has fewer constraints to be wrong about. That also explains why hybrid designs
keep reappearing -- a convolutional stem where the assumption holds, attention
above for the relations a convolution cannot represent.""")
```

## 9. Practical Example

**The patch size decides everything.** At the canonical $224/16$ configuration the
compression ratio is exactly **1.00** — a patch holds $16\times16\times3 = 768$
values and the embedding is 768 wide. **The standard ViT's input projection is a
change of basis, not a summary**, and that is a design choice. At patch 32 the
ratio is **4.00** and three quarters of every patch is destroyed before attention
runs.

That single number predicts most VLM detail failures. Below the patch grid the
model sees whatever survived one linear projection — so thin strokes, small text
and distant signs are gone before the model starts.

**Halving the patch is exactly as expensive as {{eq:halving-patch}} says.** Tokens
196 → 784; attention **0.06 → 0.94 GFLOPs (16×)**; projections **0.46 → 1.85
(4×)**; patch embedding **0.12 → 0.12 (unchanged)**; total **0.64 → 2.91**.

> **IMPORTANT:** The attention share settles a claim that is usually stated
> unconditionally. At 224 pixels attention is **9.3%** of FLOPs; at 1024 pixels,
> same architecture and patch size, it is **68.1%**. Anyone profiling a ViT at
> benchmark resolution will correctly find attention is not the bottleneck and
> will be wrong to generalise it. **"Attention is quadratic" is a claim about a
> regime**, false where ViTs are measured and dominant where documents live.

And {{sec:6-mathematical-foundation}}'s comparison of $224/p{=}8$ against
$448/p{=}16$ — identical token counts, $\rho = 0.25$ against $1.00$ — is why the
fix for "it cannot read the small print" is more input resolution rather than
smaller patches.

**What the prior is worth.** On the translation-invariant task, the convnet scores
**1.000 at 100 training examples**; the transformer scores **0.334**, which is
chance for three classes. The transformer's column is **0.334, 0.328, 0.328,
0.672** — flat at chance until past 1600 examples, then rising steeply.

**That shape matters.** The transformer is not gradually approximating the prior;
it learns nothing until it has enough data to discover the regularity.
{{eq:prior-as-data}}'s exchange rate is **at least 64×** on a task whose entire
content is the prior — which is why {{cite:dosovitskiy2021vit}} needed a
300-million-image dataset to make its point.

**And what the prior forbids.** On the position task — where the shape's *type* is
random and independent of the label, so both classes contain identical content —
the convnet scores **0.495, 0.528, 0.503, 0.514** across a 64-fold data increase.
Flat, at chance, exactly as {{eq:invariance-forbids}} requires.

**This is not a model that needs more data.** More data cannot help, more depth
cannot help, and a better optimiser cannot help, because
{{eq:pooling-invariance}} makes the two classes' representations *identical*. The
transformer reaches **1.000** on the same task with the same budget.

So the trade is not "prior versus no prior, with data deciding". **A prior is a
constraint: it buys sample efficiency inside its scope and costs everything
outside it**, and the transformer's advantage at scale is partly that it has fewer
constraints to be wrong about.

## 10. Production Considerations

**Compute $\rho$ before choosing a vision tower.** {{eq:patch-compression}} is one
division and it predicts whether fine detail can survive at all.

**Raise input resolution, do not shrink the patch**, when detail is the problem —
{{sec:6-mathematical-foundation}}'s table shows why.

**Know your regime before optimising attention.** At 224 pixels the win is in the
projections; at 1024 it is in attention.

**Interpolate position embeddings when changing resolution**, or the model
degrades silently.

**Prefer a convnet or a hybrid below roughly $10^5$ labelled images** unless you
are starting from strong pretrained weights — which changes the calculation
entirely, because the pretraining supplied the data
{{eq:prior-as-data}} was asking for.

**Do not pool if you need position.** {{eq:invariance-forbids}} applies to a ViT's
mean pool exactly as it does to a convnet's.

**Check what your tower was trained at.** A tower trained at 224 and run at 896
is outside its distribution in a way that no error message reports.

## 11. Common Mistakes

**Quoting "ViTs beat convnets" without the data condition.**

**Choosing patch 32 for speed** and then wondering about small text —
$\rho = 4.00$.

**Optimising attention at 224 pixels**, where it is 9.3% of the cost.

**Forgetting position-embedding interpolation.**

**Comparing architectures across different training recipes** —
{{cite:liu2022convnext}}'s point, and it applies to every ViT/convnet comparison.

**Assuming a pooled representation can support a positional task.**

**Treating patchification as lossless** when $\rho > 1$.

## 12. Failure Modes

**Fine-detail blindness.** Symptom: the model cannot read small text or resolve
distant objects at any prompt. Cause: {{eq:patch-compression}} at the tokeniser.
Not fixable downstream.

**Resolution-change degradation.** Symptom: accuracy drops when input size
changes. Cause: position embeddings not interpolated, or a distribution shift the
tower never saw.

**Data-starved ViT.** Symptom: a transformer plateaus at chance on a small
dataset. Cause: {{eq:prior-as-data}}. Fix: pretrain, or use a convnet.

**Quadratic blowup at high resolution.** Symptom: memory explodes
super-linearly with input size. Cause: {{eq:attention-share}} entering its
dominant regime.

**Positional task on a pooled model.** Symptom: chance accuracy that does not
improve with data — the signature of {{eq:invariance-forbids}} rather than of
under-training.

**Patch-boundary artefacts.** Symptom: predictions change discontinuously when an
object crosses a patch boundary. Cause: non-overlapping tokenisation, which has no
equivalent of {{eq:translation-equivariance}}.

## 13. Alternatives

| Alternative | Trades away | When it wins |
|---|---|---|
| ConvNeXt ({{cite:liu2022convnext}}) | global receptive field from layer 1 | small/medium data; strong baseline |
| hierarchical ViT (Swin-style) | full global attention | dense prediction, high resolution |
| convolutional stem + attention | architectural purity | almost always, in practice |
| self-supervised ViT ({{cite:oquab2023dinov2}}) | label supervision | frozen features, dense tasks |
| efficient attention ({{ch:tf-efficient}}) | exactness | only once {{eq:attention-share}} says attention dominates |
| smaller patches | compute, quadratically | when $\rho > 1$ is provably the limit |

**The third row is the honest default.** Pure architectures are cleaner to write
about; hybrids match the structure of the problem, which is why they keep winning
and keep being reinvented.

## 14. Evaluation

**Report the training data scale** with any ViT/convnet comparison — without it
the comparison is unreadable ({{eq:prior-as-data}}).

**Report input resolution and patch size**, always. They determine $\rho$ and the
token count, and a result without them is not reproducible.

**Evaluate at multiple resolutions** if you will deploy at more than one.

**Test positional sensitivity explicitly** if your task needs position —
{{eq:invariance-forbids}} failures look like under-training and are not.

**Control the training recipe** ({{cite:liu2022convnext}}).

## 15. Advanced Concepts

**The tokeniser is the whole prior.** {{maturity:ESTABLISHED}}
{{eq:patch-embedding}} is a stride-$p$ convolution, so a ViT is not
"convolution-free" — it has exactly one convolutional layer, and every remaining
image-specific assumption lives in the position embeddings. Seeing it that way
makes hybrids obvious rather than eclectic.

**Learned versus fixed position embeddings.** {{maturity:MATURE}} Learned
embeddings need interpolation to change resolution; relative and rotary schemes
({{ch:tf-positional}}) extrapolate better and are increasingly standard in vision
for exactly that reason.

**Register tokens and attention sinks.** {{maturity:EMERGING}} ViTs allocate high
attention to a few uninformative background patches, apparently using them as
scratch space. Adding dedicated register tokens removes the artefact and improves
dense-prediction features — an architectural fix for a pathology nobody designed
in.

**Self-supervision as the alternative to scale.** {{maturity:EMERGING}}
{{cite:oquab2023dinov2}} shows that curating the pretraining data can substitute
for the labels {{eq:prior-as-data}} demands, and that the resulting features beat
language-supervised ones on dense tasks — the counterweight to {{ch:mm-clip}}.

**Invariance as a design variable.** {{maturity:MATURE}}
{{eq:invariance-forbids}} says every invariance is a commitment. The right
question for any architecture is not "how much prior" but "which invariances, and
does my task respect them" — and it is answerable from the label definition alone,
before any training.

## 16. Connection to Previous Chapters

{{ch:tf-scaled-dot-product}} and {{ch:tf-positional}} supply everything below the
tokeniser; this chapter adds only {{eq:patchify}}.
{{ch:tf-complexity}}'s quadratic term becomes {{eq:attention-share}}, and this
chapter pins down the regime in which it bites.
{{ch:mm-cv-fundamentals}}'s {{eq:translation-equivariance}} is what is being given
up and {{eq:pooling-invariance}} is what {{eq:invariance-forbids}} turns into a
hard limit. {{ch:mm-classification}}'s residual wire survives unchanged into the
transformer — the architecture that replaced convolutions kept the connection that
made deep convolutions possible. Forward: {{ch:mm-clip}} trains this tower with
language supervision, {{ch:mm-vlms}} makes its token count a budget, and
{{ch:mm-ocr}} is where {{eq:patch-compression}} decides whether text is legible at
all.

## 17. Exercises

1. Compute $\rho$ for $p = 16$, $C = 3$, $d = 384$. Is the projection lossless,
   and what does that predict?
2. Derive {{eq:attention-share}} and find the token count at which attention
   reaches 50% of FLOPs for $d = 768$, $C = 3$, $p = 16$.
3. In `patch-size-economics`, add $1536 \times 1536$ at patch 16. What is the
   attention share, and what does that imply for document VLMs?
4. Verify {{eq:halving-patch}}'s three factors against the measured table.
5. In `prior-versus-data`, add a training size of 25600 for task A. Does the
   transformer overtake the convnet, and what does {{eq:prior-as-data}} predict?
6. Replace the ViT's mean pool with a CLS token. Does task B still reach 1.000?
7. Give the conv model a positional feature channel (the column index as an extra
   input plane). Does task B become learnable, and what does that say about
   {{eq:invariance-forbids}}?
8. For a vision tower you use: find its patch size, $d$, and training resolution,
   compute $\rho$, and predict the smallest text it can read.

## 18. Interview Questions

1. How does a ViT turn an image into tokens?
2. Why is $16\times16$ with $d = 768$ the standard configuration?
3. Why not use smaller patches?
4. Is attention the bottleneck in a ViT? Answer carefully.
5. Why do ViTs need position embeddings when convnets do not?
6. When would you choose a convnet over a ViT?
7. What does a prior cost, as opposed to what it buys?
8. Your model cannot read small text. What do you change, and what will not help?
9. What happens when you run a ViT at a different resolution than it was trained
   at?
10. Why do hybrid architectures keep reappearing?

## 19. Research Questions

1. {{eq:prior-as-data}} treats a prior as an exchange rate for examples. Can
   $n_{\text{eff}}$ be predicted from a measure of how well the prior matches the
   task distribution?
2. Register tokens fix an artefact nobody designed in. What causes the attention
   sink, and does the same mechanism appear in text transformers?
3. {{eq:patch-compression}} is a dimension count. What is the right
   information-theoretic version, and does a trained $E$ approach it?
4. Non-overlapping patches have no translation equivariance at all. How much of
   the ViT/convnet gap at small data is attributable to that alone, as opposed to
   locality?
5. {{cite:oquab2023dinov2}} substitutes data curation for labels. Is there a
   principled account of how much curation is worth how many labels?

## 20. Chapter Summary

A ViT does one image-specific thing: it cuts the image into patches. **That
decision fixes everything else.**

**The compression ratio {{eq:patch-compression}} decides what can be seen at all.**
At the canonical $p=16$, $d=768$, $C=3$ it is exactly **1.00** — the standard
configuration is the largest patch that is still dimensionally lossless, which is
a choice rather than a coincidence. At $p = 32$ it is **4.00**, and three quarters
of every patch is destroyed before attention runs. **Below the patch grid a vision
tower does not see pixels; it sees what survived one linear projection.**

**The cost model {{eq:vit-attention-cost}} decides what is affordable.** Halving
the patch multiplies attention by exactly **16** (0.06 → 0.94 GFLOPs), projections
by **4**, and leaves the patch embedding **unchanged** — four times the detail for
4.5 times the compute.

**And the attention share is a regime, not a property.** **9.3%** of FLOPs at 224
pixels, **68.1%** at 1024. "Attention is quadratic" is false where ViTs are
benchmarked and dominant where documents are read.

**The convolutional prior is worth data.** Measured on a translation-invariant
task, the convnet is at **1.000 with 100 examples** and the transformer at
**0.334** — chance — rising only past 1600 and reaching 0.672 at 6400. An exchange
rate of at least **64×**, and the transformer's column is flat-then-steep rather
than gradual: it learns nothing until it has enough data to find the regularity.
That is {{cite:dosovitskiy2021vit}}'s claim *with* its condition, and the condition
is the operative part.

**And a prior forbids as well as buys.** On a task about position, with image
content identical between classes, the convnet scored **0.495, 0.528, 0.503,
0.514** across a 64-fold data increase — flat, at chance, exactly as
{{eq:invariance-forbids}} requires, because {{eq:pooling-invariance}} makes the two
classes' representations identical. **More data cannot help. The function is
outside the hypothesis space.** The transformer reached **1.000**.

So the framing "prior versus no prior, and data decides" is incomplete. **A prior
is a constraint: it buys sample efficiency inside its scope and costs everything
outside it**, and the transformer's advantage at scale is partly that it has fewer
constraints to be wrong about. Which is why the right question about any
architecture is not *how much* prior but **which invariances, and does my task
respect them** — answerable from the label definition alone, before training.

## 21. Further Reading

{{cite:dosovitskiy2021vit}} for the architecture, and read the data-scale ablation
rather than the headline — it is the paper's actual claim.
{{cite:vaswani2017}} for everything below the tokeniser, unchanged.
{{cite:liu2022convnext}} as the corrective: much of the apparent gap was training
recipe, and the methodology matters more than the conclusion.
{{cite:oquab2023dinov2}} for the self-supervised route, and for evidence that
language supervision is neither required nor best for dense features.
{{cite:carion2020detr}} for what happens when you keep the patch tokens instead of
pooling them.
{{cite:he2016resnet}} once more, because the residual connection survived the
architecture that replaced convolutions — which is the better lesson about which
ideas generalise.

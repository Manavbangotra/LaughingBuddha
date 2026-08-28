---
id: mm-segmentation
number: 121
part: XIII
tier: full
status: draft
requires: [mm-cv-fundamentals, mm-classification, mm-detection, dl-losses]
provides: [semantic-instance-panoptic, encoder-decoder-skip, boundary-versus-region,
           dice-loss, class-imbalance-geometry, promptable-segmentation,
           mask-quality-metrics]
citations: [ronneberger2015unet, he2017maskrcnn, kirillov2023sam, ravi2024sam2,
            lin2014coco, carion2020detr, lin2017focal]
---

## 1. Learning Objectives

By the end of this chapter you will be able to distinguish semantic, instance and
panoptic segmentation by what question each answers; explain the encoder–decoder
with skips as the direct architectural response to
{{eq:resolution-tension}}, and **measure what the skip carries** — separating
boundary accuracy from region accuracy, because a single mean IoU conceals the
entire effect; derive why per-pixel cross-entropy fails under the class imbalance
that segmentation's *geometry* creates, and why inverse-frequency weighting is a
knob rather than a fix; and state what promptable segmentation
({{cite:kirillov2023sam}}) changed about the task's shape.

## 2. Why This Matters

A box is a poor description of most things. A road is not rectangular, a tumour is
not rectangular, and a person holding a bag is two boxes that overlap almost
entirely. Segmentation replaces the box with a per-pixel answer, and in doing so
it inherits {{ch:mm-cv-fundamentals}}'s central tension in its sharpest form:
**the network must produce an output at full resolution, using features computed
at low resolution.**

{{sec:9-practical-example}} measures what that costs, and the result is about
*where* the cost lands rather than how large it is. Reconstructing a mask from a
stride-8 bottleneck gives an overall IoU of **0.881** — respectable, shippable,
apparently fine. The IoU **at the boundary** is **0.498**. The reconstruction is
correct almost everywhere and wrong precisely where the answer is decided, and the
aggregate metric hides it because interior pixels outnumber boundary pixels and
were never in doubt.

The second measurement is about the loss. Segmentation's class balance is set by
geometry, not by dataset curation: a crack, a defect, a wire, a lesion occupies a
fraction of a per cent of its image. At a 0.3% foreground fraction, per-pixel
cross-entropy learns to output *background everywhere* — IoU falls from **0.822 to
0.025** with the task's difficulty held exactly constant. And the standard fix
turns out not to be one: **inverse-frequency weighting is worse than plain
cross-entropy at moderate imbalance**, 0.328 against 0.557.

{{maturity:ESTABLISHED}} Encoder–decoder segmentation.
{{maturity:MATURE}} Promptable, class-agnostic segmentation
({{cite:kirillov2023sam}}, {{cite:ravi2024sam2}}), which has largely replaced
training a segmenter for a new object type.

## 3. Prerequisites

{{ch:mm-cv-fundamentals}} for {{eq:resolution-tension}} and the jump — this
chapter is that equation's architectural resolution; {{ch:mm-classification}} for
the encoder; {{ch:mm-detection}} for {{eq:iou}} and for the assignment problem,
which returns here in instance form; {{ch:dl-losses}} for what a loss is allowed
to be.

## 4. Intuitive Explanation

### Three tasks that sound like one

| Task | Question | "Two adjacent cars" |
|---|---|---|
| **semantic** | what class is each pixel? | one region labelled *car* |
| **instance** | which object is each pixel? | two regions, *car 1* and *car 2* |
| **panoptic** | both, for every pixel | two car instances plus labelled road, sky |

**The distinction is not cosmetic — it decides the architecture.** Semantic
segmentation is per-pixel classification and needs no notion of objects. Instance
segmentation must separate touching objects of the same class, which is a
*grouping* problem that per-pixel classification cannot express: no labelling of
individual pixels distinguishes one car from two touching cars.

That is why instance segmentation is usually detection plus a mask
({{cite:he2017maskrcnn}}) — it borrows {{ch:mm-detection}}'s machinery to answer
"how many", then segments within each answer.

### The U-shape, and what it is fixing

To label a pixel you need two incompatible things:

> **Context** — what is this region? Requires a large receptive field, which
> requires downsampling ({{eq:receptive-field}}).
>
> **Precision** — exactly which pixels? Requires full resolution, which
> downsampling destroys.

{{cite:ronneberger2015unet}}'s answer is to do both in sequence and then
*reconnect*: downsample to get context, upsample to get resolution back, and carry
the pre-downsampling features across on **skip connections** so the decoder has
the fine detail available when it draws the edge.

The critical thing to understand is what the skip is *not* doing. It is not
improving the bottleneck. **It is routing around it** — carrying high-resolution
evidence directly to the layer that needs it, because the bottleneck cannot
localise more precisely than its own grid however good it becomes at semantics.

{{sec:9-practical-example}} makes this an information argument rather than a
training result: it measures the *best possible* reconstruction from a bottleneck,
which no amount of training can beat. At stride 16, a decoder is choosing between
edges 16 pixels apart. For a small object that is wider than the object.

### Why mean IoU is the wrong number to watch

A large blob is mostly interior. Get every boundary pixel wrong and overall IoU
stays high, because the metric averages over pixels and pixels are dominated by
the part of the problem that was never hard.

{{sec:9-practical-example}} measures the gap directly: **0.881 overall against
0.498 at the boundary**. If you report one number, report the wrong one, and the
model will look fine while being systematically unable to place an edge.

### The imbalance is geometric, and it is not a data problem

In classification, class imbalance is a property of how the dataset was collected
and can often be fixed by collecting differently. In segmentation it is a property
of the *world*: a hairline crack in a bridge is 0.1% of its photograph no matter
how many photographs you take.

At that ratio, "background everywhere" is 99.9% pixel-accurate. It is a good
solution to the loss you wrote and a useless solution to the problem you have —
and {{sec:9-practical-example}} shows cross-entropy converging on it.

**Dice attacks this structurally.** It is built from the overlap between
prediction and target, and true negatives appear nowhere in it, so adding a
million background pixels changes the loss not at all. That invariance is why it
degrades where cross-entropy collapses.

### Promptable segmentation changed the task

{{cite:kirillov2023sam}}'s contribution is worth separating into two parts,
because the less-discussed one is larger. The model is promptable — click a point,
draw a box, get a mask — and **class-agnostic**: it segments *things*, without
knowing what they are.

That converts segmentation from something you train per object type into something
you *call*, and pairs naturally with an open-vocabulary classifier
({{ch:mm-clip}}) that says what the returned region is. The second part is the
**data engine**: SA-1B's billion masks were produced by bootstrapping annotation
with the model being trained, which is arguably a bigger contribution than the
architecture.

## 5. Formal Explanation

### 5.1 The three tasks, written down

$$ \text{semantic: } f: \mathbb{R}^{H\times W\times 3} \to \{1..C\}^{H \times W} $$ (eq:semantic-seg)

$$ \text{instance: } f \to \big\{(m_i, c_i)\big\}_{i=1}^{n},\; m_i \in \{0,1\}^{H\times W},\; n \text{ unknown} $$ (eq:instance-seg)

{{eq:instance-seg}} has the variable-length output of {{ch:mm-detection}}'s
{{eq:detector-families}}, so it inherits the same three difficulties and the same
family of solutions. **Semantic segmentation is a dense classification problem;
instance segmentation is a set-prediction problem that happens to emit masks.**

### 5.2 The resolution bound

Let the encoder reduce resolution by stride $s$. A decoder reconstructing from the
bottleneck alone can only place a boundary on the coarse grid, so its expected
boundary error is

$$ \mathbb{E}[\text{edge error}] \approx \tfrac{s}{4} \text{ pixels}, \qquad \text{IoU}_{\max} \approx 1 - \frac{\kappa\, s\, P}{A} $$ (eq:bottleneck-iou-bound)

for perimeter $P$, area $A$ and a shape constant $\kappa$. Two consequences follow
and both are visible in the measurement:

- **The penalty scales with $P/A$** — the perimeter-to-area ratio. Large compact
  objects are barely affected; small or thin ones are destroyed. At radius $r$,
  $P/A \sim 1/r$.
- **Structures thinner than $s$ can vanish entirely** under area-averaging, and no
  decoder recovers them, because the information is not in the bottleneck at all.

The skip connection removes {{eq:bottleneck-iou-bound}} by making the
full-resolution feature available:

$$ \text{IoU}_{\max}^{\text{skip}} = 1 \quad\text{(information-theoretically)} $$ (eq:skip-removes-bound)

### 5.3 Why the aggregate metric hides it

Split the pixels into a boundary band $B$ of width $w$ and an interior $I$:

$$ \text{IoU}_{\text{overall}} \approx \frac{|I| \cdot \text{acc}_I + |B| \cdot \text{acc}_B}{|I| + |B|}, \qquad \frac{|B|}{|I|} \approx \frac{wP}{A} $$ (eq:boundary-dilution)

For a large blob $|B|/|I|$ is small, so $\text{acc}_B$ barely moves the aggregate.
**{{eq:boundary-dilution}} is the same dilution argument as
{{ch:rag-chunking}}'s {{eq:chunk-dilution}}**, in a different domain: an average
over many easy items conceals systematic failure on a few hard ones.

The fix is to report $\text{acc}_B$ separately — boundary IoU, or a boundary
F-score at a tolerance.

### 5.4 Cross-entropy under geometric imbalance

For foreground fraction $\phi$, mean binary cross-entropy is

$$ \mathcal{L}_{\text{CE}} = -\phi\,\mathbb{E}_{+}[\log p] - (1-\phi)\,\mathbb{E}_{-}[\log(1-p)] $$ (eq:ce-imbalance)

The gradient contributions are in the ratio $\phi : (1-\phi)$, so at $\phi =
0.003$ the background outvotes the foreground **332 to 1**. The all-background
predictor achieves loss $-\phi \log \epsilon$ bounded and pixel accuracy
$1 - \phi$, which is an excellent local optimum by every quantity the loss can
see.

**Nothing is malfunctioning.** {{eq:ce-imbalance}} is being minimised correctly;
it simply is not the objective anyone wanted.

### 5.5 Dice

$$ \mathcal{L}_{\text{Dice}} = 1 - \frac{2\sum_i p_i y_i}{\sum_i p_i + \sum_i y_i} $$ (eq:dice)

Note what is absent: $\sum_i (1-p_i)(1-y_i)$, the true negatives. So

$$ \frac{\partial \mathcal{L}_{\text{Dice}}}{\partial(\text{number of background pixels})} = 0 $$ (eq:dice-invariance)

**{{eq:dice-invariance}} is the whole reason Dice works here**, and it is
structural rather than a tuned correction. Dice is a soft relaxation of the
F1 score, and F1 — like IoU — ignores true negatives by construction.

### 5.6 Weighted cross-entropy is a knob, not a fix

Reweighting positives by $1/\phi$ equalises the gradient contributions. But
equalising is not optimising: it moves the model's operating point toward
predicting foreground, and

$$ \text{recall} \nearrow, \qquad \text{precision} \searrow, \qquad \text{IoU} = \frac{\text{TP}}{\text{TP}+\text{FP}+\text{FN}} \text{ can fall} $$ (eq:weighting-overshoot)

{{sec:9-practical-example}} measures exactly this: weighted CE scores **0.328
against plain CE's 0.557** at a 3% foreground fraction — *worse*, because it has
traded a recall failure for a precision failure. The correct weight is neither 1
nor $1/\phi$ but a dataset-specific value that has to be tuned, and re-tuned when
the balance drifts.

## 6. Mathematical Foundation

### 6.1 The perimeter-to-area argument, worked

For a disc of radius $r$: $A = \pi r^2$, $P = 2\pi r$, so $P/A = 2/r$. With a
boundary band of half-width $w$:

$$ \frac{|B|}{A} \approx \frac{2wP}{A} = \frac{4w}{r} $$ (eq:band-share)

At $w = 2$: a radius-34 blob has band share $8/34 = 0.24$, and a radius-12 blob
has $8/12 = 0.67$. The measured values — computed on wobbly rather than perfect
discs, so a little larger — are **0.334** and **0.938**.

**The small object is almost entirely boundary.** Which is why
{{eq:boundary-dilution}}'s concealment works in reverse there: for a small object
the aggregate metric *is* the boundary metric, and it falls to 0.464 at stride 16
where the large blob still reads 0.773.

### 6.2 What stride 16 means for a small object

A radius-12 blob is 24 pixels across. At stride 16 the decoder places edges on a
16-pixel grid, so the entire object spans between one and two grid cells. The best
available reconstruction is a square of 16 or 32 pixels against a roughly circular
target of diameter 24:

$$ \text{IoU} \approx \frac{\min(A_{\text{sq}}, A_{\text{circ}})}{\max(A_{\text{sq}}, A_{\text{circ}})} \approx \frac{256}{452} \approx 0.57 $$ (eq:small-object-stride)

against a measured 0.464 — the difference being alignment, since the grid is not
centred on the object. **The order of magnitude is set by geometry alone**, before
any network exists.

> **MATH NOTE:** This is why "just train longer" and "use a bigger model" do not
> help small-object segmentation, and why the effective fixes are all about
> resolution: keep a higher-resolution branch, use skips, use dilation instead of
> stride ({{ch:mm-cv-fundamentals}}), or run at a larger input size.
> {{eq:bottleneck-iou-bound}} is a bound on the architecture, not on the training.

### 6.3 The imbalance ratio, worked

At $\phi = 0.003$ the background/foreground gradient ratio is $(1-\phi)/\phi =
332$. For a foreground gradient to move the parameters against the background
consensus, the per-pixel foreground gradient must exceed the *average* background
gradient by that factor.

Early in training, $p \approx 0.5$ everywhere, so per-pixel gradients are
comparable in magnitude and the background wins by 332 to 1. The model drives $b$
strongly negative, $p \to 0$ everywhere, and the foreground gradient — now
$|p - y| \to 1$ per foreground pixel — is still outnumbered.

The measured escape rate confirms it: the fraction of pixels cross-entropy calls
foreground goes **0.2910, 0.0899, 0.0219, 0.0029, 0.0001** across the sweep, which
is the model progressively deciding the class does not exist.

## 7. Internal Mechanics

```mermaid {#fig:unet caption="The U-shape, drawn to show what each path carries. The bottom of the U has the context and cannot localise below its own grid (eq:bottleneck-iou-bound); the skips carry the resolution the encoder discarded straight to the decoder layer that needs it (eq:skip-removes-bound). The skip is not improving the bottleneck — it is routing around it."}
flowchart TB
    I["image H x W"] --> E1["enc /2"] --> E2["enc /4"] --> E3["enc /8"] --> B["bottleneck /16<br/>large receptive field,<br/>coarse grid"]
    B --> D3["dec /8"] --> D2["dec /4"] --> D1["dec /2"] --> O["mask H x W"]
    E3 -.->|"skip: full detail<br/>at this resolution"| D3
    E2 -.->|"skip"| D2
    E1 -.->|"skip"| D1
    B -.->|"answers WHAT"| B
    E1 -.->|"answers WHERE"| E1
```

### 7.1 Getting resolution back: three mechanisms

| Mechanism | What it does | Cost |
|---|---|---|
| skip connections | carry fine features across | memory for stored activations |
| dilated convolution | field without stride ({{ch:mm-cv-fundamentals}}) | compute at full resolution |
| learned upsampling | transposed conv | checkerboard artefacts if stride and kernel mismatch |
| higher input resolution | avoid the problem | quadratic compute |

**All four are spending something to avoid {{eq:bottleneck-iou-bound}}**, and the
right mix depends on your $P/A$ distribution — that is, on how thin and small your
objects are, which is measurable from your labels before you choose.

### 7.2 Instance segmentation, and where the assignment problem returns

Two families, and they are {{ch:mm-detection}}'s families again:

**Detect then segment** ({{cite:he2017maskrcnn}}). Get boxes, segment inside each.
Simple, and it inherits detection's crowd failure — {{eq:crowd-ambiguity}} deletes
an object, and the mask goes with it.

**Mask-level set prediction.** Predict $N$ masks with one-to-one matching, exactly
{{eq:set-prediction}}. No NMS, and the matching cost uses mask overlap instead of
box IoU. This is where {{cite:carion2020detr}}'s contribution ended up mattering
most.

### 7.3 SAM's shape, and what it is not

Three parts: a heavy image encoder run **once** per image, a light prompt encoder,
and a light mask decoder run **per prompt**. That split is what makes interactive
use practical — the expensive part is amortised across every click.

Two limits worth being explicit about, because they are frequently missed:

- **It does not know what things are.** Class-agnostic by design; pairing it with
  a classifier is on you.
- **Ambiguity is inherent to a point prompt.** A click on a shirt could mean the
  shirt, the person, or the group, so the model returns *multiple* masks with
  confidence scores. Consuming only the top one silently discards the disambiguation
  the model was built to provide.

## 8. Implementation

```python {tier=A name=what-skips-carry}
"""What skip connections are actually for, and why mean IoU hides it.

ch:mm-cv-fundamentals stated the tension every dense-prediction task has: context
wants downsampling and localisation wants resolution (eq:resolution-tension).
cite:ronneberger2015unet's answer is to do both -- downsample for context, then
upsample, and carry the high-resolution detail across on skip connections.

This listing measures what that carrying is worth. Nothing is trained: the
question is what an encoder-decoder CAN represent, and the answer without skips
is bounded by the information surviving the bottleneck
(eq:bottleneck-iou-bound). An upper bound is a stronger statement than a trained
result, because no amount of training beats it.

The measurement is split into interior pixels and boundary pixels, because
reporting one number for both is exactly how this effect stays invisible.
"""
import numpy as np

rng = np.random.default_rng(29)

H = 128
N_SHAPE = 200


def make_mask(radius, thin=False):
    """A blob, optionally with a thin protrusion -- the structure that dense
    prediction is asked for and that a coarse grid cannot hold."""
    yy, xx = np.mgrid[0:H, 0:H]
    cy, cx = rng.uniform(0.35 * H, 0.65 * H, size=2)
    # A wobbly blob: radius modulated by a few Fourier components.
    ang = np.arctan2(yy - cy, xx - cx)
    r = radius * (1.0 + 0.18 * np.sin(3 * ang + rng.uniform(0, 6))
                  + 0.12 * np.sin(5 * ang + rng.uniform(0, 6)))
    m = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r ** 2
    if thin:
        w = max(int(radius * 0.12), 1)
        y0 = int(np.clip(cy, 0, H - 1))
        m[max(y0 - w, 0):y0 + w, :] = True
    return m


def downsample(mask, s):
    """Area-average to a coarse grid: what the encoder's bottleneck retains."""
    h = H // s
    return mask.reshape(h, s, h, s).mean(axis=(1, 3))


def upsample(coarse, s):
    """Nearest-neighbour expansion, thresholded -- the best a decoder can do
    from the bottleneck alone."""
    return np.repeat(np.repeat(coarse, s, axis=0), s, axis=1) >= 0.5


def boundary_band(mask, width=2):
    """Pixels within `width` of the mask edge, found by comparing the mask with
    shifted copies of itself."""
    edge = np.zeros_like(mask)
    for dy in range(-width, width + 1):
        for dx in range(-width, width + 1):
            edge |= (np.roll(np.roll(mask, dy, 0), dx, 1) != mask)
    return edge


def iou(a, b, where=None):
    if where is not None:
        a, b = a & where, b & where
    inter = float((a & b).sum())
    union = float((a | b).sum())
    return inter / union if union else 1.0


print(f"{H}x{H} masks. Every row is the BEST a decoder could do from the")
print("bottleneck alone. Nothing is trained, so these are upper bounds that no")
print("amount of training beats. WITH a skip connection the full-resolution "
      "evidence is available directly, so that column would read 1.000 in "
      "every row and is omitted -- the question is what the "
      "bottleneck-only path has already thrown away." + chr(10))
print(f"{'object':<16}{'stride':>8}{'overall IoU':>14}{'boundary IoU':>15}"
      f"{'band/object':>14}{'overall hides':>15}")
print("-" * 82)

for label, radius, thin in (("large blob", 34, False),
                            ("small blob", 12, False),
                            ("blob + thin bar", 30, True)):
    for s in (4, 8, 16):
        ov, bd, frac = [], [], []
        for _ in range(N_SHAPE):
            m = make_mask(radius, thin)
            rec = upsample(downsample(m, s), s)
            band = boundary_band(m)
            ov.append(iou(m, rec))
            bd.append(iou(m, rec, where=band))
            frac.append(float(band.sum()) / max(float(m.sum()), 1.0))
        o, b = float(np.mean(ov)), float(np.mean(bd))
        print(f"{label:<16}{s:>8}{o:>14.3f}{b:>15.3f}{np.mean(frac):>14.3f}"
              f"{o - b:>15.3f}")

print("""
Read the last column first, because it is the reason this measurement is split.
Overall IoU and boundary IoU are not close, and the gap is the amount of error
that a single reported mean IoU is concealing. For a large blob at stride 8, the
overall number stays high while the boundary number is far lower -- the
reconstruction is correct almost everywhere and wrong exactly where the answer is
decided.

The mechanism is a counting argument, not a modelling one (eq:boundary-dilution).
A large blob is mostly interior: the boundary band is a third of its pixels, so
getting every boundary pixel wrong still leaves overall IoU respectable. Mean IoU
averages over pixels, and pixels are dominated by the easy part of the problem.

Now read down each block. Increasing the stride makes both numbers worse, and it
makes the boundary number worse much faster, because the reconstruction can only
place an edge on a multiple of the stride. At stride 16 the decoder is choosing
between edges 16 pixels apart -- for a blob of radius 12 that is larger than the
object.

Compare the three blocks and the pattern sharpens. The band/object column is the
explanation: it is 0.334 for the large blob and 0.938 for the small one, so the
small blob is almost ENTIRELY boundary and there is no easy interior to dilute
its errors. Its overall IoU therefore falls to 0.464 at stride 16 where the large
blob still reads 0.773. And the thin bar is the worst case: a structure narrower
than the stride is erased by area-averaging, so no decoder recovers it, however
deep.

That is what the skip connection buys, and it explains why the U-shape is drawn
the way it is. The bottleneck answers WHAT -- it has the context, the receptive
field, the semantics. It cannot answer WHERE to a precision finer than its own
grid, and eq:resolution-tension says making it finer costs the context that made
it useful. The skip does not improve the bottleneck. It routes around it, carrying
the high-resolution evidence directly to the layer that has to draw the edge.

The practical consequence is a reporting rule. Report boundary IoU separately, or
a boundary F-score, because the aggregate is dominated by interior pixels that
were never in doubt -- and it will tell you a segmentation model is fine while it
is systematically unable to place an edge.""")
```

The first listing is about what the architecture can represent. The second is
about whether the loss will ask for it.

```python {tier=A name=dice-versus-cross-entropy}
"""Class imbalance in segmentation, and why cross-entropy quietly gives up.

Segmentation is per-pixel classification, so the obvious loss is per-pixel
cross-entropy. It has a failure mode that does not appear in ordinary
classification, because segmentation's class balance is set by GEOMETRY: a tumour,
a crack, a defect, or a thin wire occupies a tiny fraction of its image, and the
fraction can be 1% or 0.1% without anything being unusual.

At that ratio, predicting "background" everywhere is already 99% accurate and has
low cross-entropy (eq:ce-imbalance). The all-background solution is a good local
optimum, and the gradient pointing away from it is outnumbered.

The Dice loss (eq:dice) is built from overlap rather than from per-pixel
correctness, so it is invariant to how much background there is
(eq:dice-invariance). This listing trains the same model with each loss across a
sweep of foreground fractions, and reports IoU -- which is what anyone actually
cares about.
"""
import numpy as np

rng = np.random.default_rng(37)

N_PIX, D = 4000, 12
STEPS, LR = 700, 0.5
N_TRIAL = 5


def make_task(fg_frac, sep=3.0):
    """Per-pixel features. Foreground pixels are shifted along one direction by
    `sep` -- the same separability at every foreground fraction, so the only
    thing changing down the sweep is the CLASS BALANCE."""
    y = (rng.random(N_PIX) < fg_frac).astype(float)
    w_true = rng.normal(size=D)
    w_true /= np.linalg.norm(w_true)
    X = rng.normal(size=(N_PIX, D)) + sep * y[:, None] * w_true[None, :]
    return X, y


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def train(X, y, loss, steps=STEPS):
    w, b = np.zeros(D), 0.0
    for _ in range(steps):
        p = sigmoid(X @ w + b)
        if loss == "ce":
            # dL/dz for mean binary cross-entropy.
            gz = (p - y) / len(y)
        elif loss == "weighted-ce":
            # Positives reweighted by the inverse class frequency.
            pos = max(y.mean(), 1e-6)
            wt = np.where(y > 0, 1.0 / pos, 1.0 / (1.0 - pos))
            gz = wt * (p - y) / len(y)
        else:
            # eq:dice, soft form: 1 - 2<p,y> / (sum p + sum y).
            num, den = 2.0 * (p * y).sum(), p.sum() + y.sum() + 1e-8
            gz = -((2.0 * y * den - num) / den ** 2) * p * (1 - p)
        w -= LR * (X.T @ gz)
        b -= LR * gz.sum()
    return w, b


def iou_of(X, y, w, b, thr=0.5):
    pred = sigmoid(X @ w + b) >= thr
    t = y > 0
    union = float((pred | t).sum())
    return float((pred & t).sum()) / union if union else 1.0


FRACTIONS = (0.30, 0.10, 0.03, 0.01, 0.003)
LOSSES = ("ce", "weighted-ce", "dice")

print(f"{N_PIX} pixels, {D} features, identical class separability in every row.")
print("Only the foreground fraction changes.\n")
print(f"{'foreground':>11}{'':>3}" + "".join(f"{L + ' IoU':>16}" for L in LOSSES)
      + f"{'CE predicts fg':>17}")
print("-" * 79)

for f in FRACTIONS:
    scores = {L: [] for L in LOSSES}
    ce_rate = []
    for _ in range(N_TRIAL):
        X, y = make_task(f)
        for L in LOSSES:
            w, b = train(X, y, L)
            scores[L].append(iou_of(X, y, w, b))
            if L == "ce":
                ce_rate.append(float((sigmoid(X @ w + b) >= 0.5).mean()))
    print(f"{f:>11.3f}{'':>3}"
          + "".join(f"{np.mean(scores[L]):>16.3f}" for L in LOSSES)
          + f"{np.mean(ce_rate):>17.4f}")

print("""
Read the last column alongside the first. As the foreground fraction falls,
cross-entropy stops predicting foreground at all -- the share of pixels it calls
positive goes 0.2910, 0.0899, 0.0219, 0.0029, 0.0001, and its IoU follows it
down: 0.822 to 0.025, a factor of thirty-three.

Nothing about the problem got harder. The separability is identical in every row
-- the same feature shift, the same noise, the same pixel count. The only thing
that changed is how many pixels are foreground, and cross-entropy responded by
abandoning the class it was asked to find.

That is eq:ce-imbalance behaving as written rather than misbehaving. Mean
cross-entropy averages over pixels, so at a 0.3% foreground fraction the
background terms outnumber the foreground terms three hundred to one. The
all-background predictor already scores 99.7% pixel accuracy at a low loss, and
the gradient pulling toward the foreground is averaged against three hundred
pulling the other way. The optimiser is not failing; it is succeeding at the
objective it was given, and the objective was the wrong one.

Now the weighted-cross-entropy column, which is the standard first response and
does not behave the way the standard advice implies. At moderate imbalance it is
WORSE than plain cross-entropy -- 0.328 against 0.557 at a 3% foreground, and
0.134 against 0.300 at 1%. Reweighting the positive term by the inverse class
frequency does stop the model ignoring the foreground, and it overshoots: the
model now over-predicts foreground, precision falls, and IoU falls with it
(eq:weighting-overshoot). The fix has traded a recall failure for a precision
failure and kept the same shape of problem.

It only overtakes plain cross-entropy in the last row, where plain CE has
collapsed entirely. So inverse-frequency weighting is not a solution to imbalance,
it is a knob that moves the operating point, and the correct weight is neither 1
nor 1/frequency but something dataset-specific that has to be tuned and re-tuned
as the balance drifts.

Dice needs no such constant. It is built from the OVERLAP between prediction and
target (eq:dice), and true negatives appear nowhere in it -- adding a million
background pixels changes the loss not at all. That invariance is structural, and
it shows: 0.821 down to 0.368 across a hundredfold change in class balance, a
factor of 2.2 against cross-entropy's 33. Dice degrades; cross-entropy collapses.

The transferable rule: when the metric you care about is defined by overlap,
optimising per-pixel correctness is optimising a proxy -- and the proxy diverges
from the metric exactly as the class becomes rare, which is exactly the regime
that made anyone build the model.""")
```

## 9. Practical Example

**What the skip carries.** Reconstructing from a stride-8 bottleneck gives overall
IoU **0.881** and boundary IoU **0.498** — a gap of **0.383** that a single
reported mean IoU conceals entirely. Nothing here is trained, so this is an upper
bound: {{eq:bottleneck-iou-bound}} is a fact about the architecture that no
training beats.

The mechanism is {{eq:boundary-dilution}}, and the band/object column makes it
explicit: **0.334** for the large blob and **0.938** for the small one.

**The small object is almost entirely boundary**, so it has no easy interior to
dilute its errors — which is why its overall IoU falls to **0.464** at stride 16
where the large blob still reads **0.773**. {{eq:band-share}} predicted 0.24 and
0.67 for perfect discs; the measured 0.334 and 0.938 are larger because the shapes
are wobbly, and the ordering is what matters.

**And the thin bar is unrecoverable.** A structure narrower than the stride is
erased by area-averaging, so the information is not in the bottleneck at all.
Depth does not help; only resolution does.

> **IMPORTANT:** This is why "train longer" and "use a bigger model" do not fix
> small-object segmentation. {{eq:small-object-stride}} puts the achievable IoU at
> roughly 0.57 for a 24-pixel object on a 16-pixel grid **from geometry alone**,
> before any network exists. The effective interventions are all about resolution:
> skips, dilation instead of stride, a higher-resolution branch, or a larger input.

**What the loss will ask for.** With separability held exactly constant and only
the class balance changing, cross-entropy's IoU falls **0.822 → 0.025**, a factor
of **33**, and the share of pixels it calls foreground goes **0.2910 → 0.0001**.
The model has learned that the class does not exist.

{{eq:ce-imbalance}} is being minimised correctly — at $\phi = 0.003$ the
background outvotes the foreground **332 to 1**, and "background everywhere" is
99.7% pixel-accurate. **The optimiser is succeeding at an objective nobody
wanted.**

**And the standard fix is not one.** Inverse-frequency weighting scores **0.328
against plain cross-entropy's 0.557** at a 3% foreground fraction, and **0.134
against 0.300** at 1% — *worse in both*. {{eq:weighting-overshoot}}: it stops the
model ignoring the foreground and overshoots into over-predicting it, trading a
recall failure for a precision failure. It only wins in the last row, where plain
CE has already collapsed.

**Dice needs no constant.** {{eq:dice-invariance}} means true negatives are absent
from the loss, so background quantity is irrelevant by construction: **0.821 →
0.368** across a hundredfold change in balance, a factor of **2.2** against
cross-entropy's 33. **Dice degrades; cross-entropy collapses.**

## 10. Production Considerations

**Report boundary IoU separately** from overall IoU, always. The aggregate is
dominated by pixels that were never in doubt.

**Measure your objects' $P/A$ distribution from the labels** before choosing a
stride. {{eq:bottleneck-iou-bound}} is computable in advance.

**Use Dice, or Dice combined with cross-entropy**, whenever foreground is below
roughly 10%. The combination is common because CE gives better-calibrated
probabilities and Dice gives the invariance.

**Do not reach for inverse-frequency weighting as a default.** If you use
weighting, tune the weight and re-check it when the data changes.

**Never resize away thin structures.** A wire two pixels wide at native resolution
does not exist at half resolution, and no model recovers it.

**Prefer SAM plus a classifier over training a segmenter** for a new object type,
unless you have a specific reason. It is usually faster and better.

**Consume SAM's multiple masks.** A point prompt is ambiguous by design; taking
only the top mask discards the disambiguation.

**Check the train/eval resize path matches.** Mask misalignment from
interpolation mismatch is a common, silent few-point loss.

## 11. Common Mistakes

**Reporting only mean IoU.** The chapter's headline.

**Using per-pixel cross-entropy on a 1% foreground class.**

**Assuming inverse-frequency weighting fixes imbalance** — measured worse than
plain CE at moderate imbalance.

**Downsampling before segmenting**, then wondering about thin structures.

**Using semantic segmentation where instances are needed.** No pixel labelling
separates two touching cars.

**Interpolating masks with bilinear and thresholding at 0.5** without checking the
boundary shift it introduces.

**Ignoring SAM's ambiguity outputs.**

## 12. Failure Modes

**Boundary blur.** Symptom: masks look correct at a glance and are consistently a
few pixels off. Cause: {{eq:bottleneck-iou-bound}}. Detect with boundary IoU.

**Thin-structure erasure.** Symptom: wires, cracks, catheters missing entirely.
Cause: stride wider than the structure. Not fixable downstream.

**All-background collapse.** Symptom: training loss looks excellent, IoU is zero.
Cause: {{eq:ce-imbalance}}. Detect by logging predicted foreground fraction — the
single most useful segmentation training metric.

**Over-prediction after reweighting.** Symptom: recall high, precision poor, IoU
still bad. Cause: {{eq:weighting-overshoot}}.

**Instance merging.** Symptom: touching objects of the same class become one.
Cause: semantic architecture used for an instance task.

**Checkerboard artefacts.** Symptom: regular grid patterning in masks. Cause:
transposed convolution with stride not dividing kernel size.

**Class leakage at boundaries.** Symptom: a thin halo of the wrong class around
every object — the boundary band failing systematically, visible only if you look
at it separately.

## 13. Alternatives

| Approach | Trades away | When it wins |
|---|---|---|
| U-Net and descendants | nothing much | the default for semantic segmentation |
| dilated/atrous networks | compute at full resolution | when memory allows and boundaries matter |
| Mask R-CNN ({{cite:he2017maskrcnn}}) | crowd robustness | instances, when detection already works |
| mask set prediction | convergence speed | crowded instance scenes |
| SAM + classifier ({{cite:kirillov2023sam}}) | class knowledge, latency | novel object types, no labels |
| SAM 2 ({{cite:ravi2024sam2}}) | — | video, and faster than SAM on images |
| classical (watershed, graph cuts) | semantics | high-contrast, well-specified domains |

**The last row is not a joke.** For a controlled imaging setup — microscopy,
industrial inspection with fixed lighting — a thresholding pipeline can beat a
neural network, run in microseconds, and be debuggable by a human.

## 14. Evaluation

**Boundary IoU or boundary F-score, reported separately.** Non-negotiable given
{{eq:boundary-dilution}}.

**IoU per class, and per object size.** The mean over classes hides rare ones, and
the mean over sizes hides {{eq:bottleneck-iou-bound}}.

**Predicted foreground fraction during training.** The earliest detector of
{{eq:ce-imbalance}} collapse.

**For instances: use panoptic quality or mask AP**, not semantic IoU. They measure
different things and semantic IoU cannot see a merge.

**Evaluate at native resolution.** Evaluating on downsampled masks measures a
different, easier problem.

**Count connected components** against the expected instance count. A cheap check
that catches merging and fragmentation.

## 15. Advanced Concepts

**Boundary-aware losses.** {{maturity:MATURE}} Weight the loss by distance to the
boundary, attacking {{eq:boundary-dilution}} in the objective rather than only in
the metric. It works, and it needs the same care as any reweighting
({{eq:weighting-overshoot}}).

**Panoptic segmentation as unified set prediction.** {{maturity:MATURE}}
{{eq:instance-seg}} and {{eq:semantic-seg}} become one problem once both are
mask-plus-label set prediction, which is why architectures descended from
{{cite:carion2020detr}} now do all three tasks with one head.

**The data engine.** {{maturity:EMERGING}} SA-1B's billion masks came from
bootstrapping annotation with the model being trained.
**That loop, not the architecture, is the transferable idea**, and it applies
wherever labels are the bottleneck.

**Segmentation as a general interface.** {{maturity:EMERGING}} A class-agnostic
segmenter is a *region proposer* for anything downstream — a VLM
({{ch:mm-vlms}}), a retrieval system ({{ch:mm-multimodal-rag}}), an editor. The
mask is becoming a primitive rather than an output.

**Temporal propagation.** {{maturity:EMERGING}} {{cite:ravi2024sam2}}'s streaming
memory makes tracking a segmentation prompt rather than a separate task —
{{ch:mm-video-audio}} takes this up.

## 16. Connection to Previous Chapters

{{ch:mm-cv-fundamentals}}'s {{eq:resolution-tension}} is the problem this chapter
exists to resolve, and its jump is what {{eq:bottleneck-iou-bound}} turns into a
quality bound. {{ch:mm-classification}}'s encoder is the U's left arm and
{{eq:residual-block}}'s wire is the same idea as the skip at a different scale —
both route information around a lossy transformation.
{{ch:mm-detection}}'s {{eq:iou}} defines correctness here too, its
{{eq:set-prediction}} solves instance assignment, and its
{{eq:crowd-ambiguity}} is inherited wholesale by detect-then-segment.
{{eq:boundary-dilution}} is {{ch:rag-chunking}}'s {{eq:chunk-dilution}} in a new
domain. Forward: {{ch:mm-clip}} supplies the class knowledge SAM lacks, and
{{ch:mm-video-audio}} extends masks through time.

## 17. Exercises

1. Derive {{eq:band-share}} for a disc and check it against the measured
   band/object values for radius 34 and 12.
2. Use {{eq:bottleneck-iou-bound}} to predict overall IoU for a radius-20 blob at
   stride 8, then verify by adding it to `what-skips-carry`.
3. In the same listing, replace nearest-neighbour upsampling with bilinear. How
   much of the boundary gap closes, and why not all of it?
4. Add a structure two pixels wide. At what stride does it disappear entirely?
5. Derive {{eq:dice}}'s gradient and show that it does not depend on the number of
   true negatives.
6. In `dice-versus-cross-entropy`, sweep the positive weight from 1 to $1/\phi$ at
   a 3% foreground fraction. Where is the IoU-optimal weight, and is it either
   endpoint?
7. Combine Dice and cross-entropy with a mixing coefficient and sweep it. Does the
   combination beat both?
8. Take a segmentation model you use. Report boundary IoU and overall IoU
   separately, and the predicted foreground fraction. Which of
   {{sec:12-failure-modes}}'s modes do you have?

## 18. Interview Questions

1. Distinguish semantic, instance and panoptic segmentation.
2. Why can semantic segmentation not separate two touching cars?
3. What problem do skip connections solve, and what would happen without them?
4. Why is mean IoU a misleading metric, and what would you report instead?
5. Why does cross-entropy fail on a 1% foreground class?
6. Why is Dice invariant to background quantity?
7. Is inverse-frequency weighting a fix for imbalance? Justify.
8. Your model misses thin structures entirely. Diagnose, and say what will not
   help.
9. What did SAM change about how segmentation is built?
10. Why does a point prompt produce multiple masks?

## 19. Research Questions

1. {{eq:bottleneck-iou-bound}} bounds a bottleneck-only decoder. What is the
   analogous bound with skips at strides $\{2,4,8\}$, and does it explain the
   observed benefit of each level?
2. {{eq:weighting-overshoot}} says the optimal positive weight is interior. Is
   there a principled estimator for it from the foreground fraction and the
   separability?
3. Dice degrades from 0.821 to 0.368 across the sweep. What loss, if any, is
   genuinely invariant to class balance at fixed separability?
4. SA-1B's data engine produced a billion masks. What is the general condition
   under which model-in-the-loop annotation converges rather than amplifying its
   own errors?
5. Boundary metrics need a tolerance. Is there a tolerance-free measure of mask
   quality that correlates with downstream task performance?

## 20. Chapter Summary

Segmentation replaces the box with a per-pixel answer and inherits
{{eq:resolution-tension}} in its sharpest form: **full-resolution output from
low-resolution features.**

**The encoder–decoder with skips is the direct response**, and what the skip
carries is measurable. Reconstructing from a stride-8 bottleneck gives overall IoU
**0.881** and boundary IoU **0.498** — and nothing is trained, so
{{eq:bottleneck-iou-bound}} is a bound on the architecture that training cannot
beat. **The skip does not improve the bottleneck; it routes around it.**

**The aggregate metric conceals the entire effect.** {{eq:boundary-dilution}}:
interior pixels outnumber boundary pixels and were never in doubt. The band/object
share is **0.334** for a large blob and **0.938** for a small one, which is why
the small object's overall IoU falls to **0.464** at stride 16 where the large
one still reads 0.773 — for a small object there is no interior to hide behind.
**Report boundary IoU separately or you will not see this.**

**Structures thinner than the stride are unrecoverable.** The information is not
in the bottleneck, so depth does not help and
{{eq:small-object-stride}} sets the ceiling from geometry alone. Only resolution
interventions work.

**And segmentation's class imbalance is geometric, not curatorial.** With
separability held constant, cross-entropy's IoU fell **0.822 → 0.025** as
foreground went 30% → 0.3%, and the fraction of pixels it called foreground fell
to **0.0001**. {{eq:ce-imbalance}} is being minimised correctly — the background
outvotes the foreground 332 to 1 — and it is the wrong objective.

**The standard fix underperforms.** Inverse-frequency weighting measured **worse
than plain cross-entropy** at moderate imbalance (0.328 against 0.557), because
{{eq:weighting-overshoot}} trades a recall failure for a precision failure. It is
a knob, not a solution.

**Dice's invariance is structural** ({{eq:dice-invariance}}) — true negatives are
absent from the loss — and it degrades by a factor of **2.2** where cross-entropy
collapses by **33**.

**And {{cite:kirillov2023sam}} changed the task's shape**: segmentation became
something you *prompt* rather than something you train, class-agnostic by design
and paired with a classifier for semantics. The data engine that produced its
billion masks is arguably the larger contribution, and it generalises anywhere
labels are the bottleneck.

## 21. Further Reading

{{cite:ronneberger2015unet}} for the architecture, and read it noticing how much
of the paper is about data augmentation — the skips are the famous part and were
not the only contribution.
{{cite:he2017maskrcnn}} for instance segmentation and for RoIAlign, which is a
lesson in the cost of a rounding operation.
{{cite:kirillov2023sam}} for promptable segmentation — Section 4 for the data
engine, which is the part to steal.
{{cite:ravi2024sam2}} for the video extension, and note it is faster on single
images than its predecessor.
{{cite:carion2020detr}} again, because mask set prediction is where its
contribution mattered most.
{{cite:lin2017focal}} for the other standard response to
{{eq:ce-imbalance}}, and {{cite:lin2014coco}} for the mask conventions every
number in this chapter inherits.

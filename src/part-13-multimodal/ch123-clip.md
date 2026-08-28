---
id: mm-clip
number: 123
part: XIII
tier: full
status: draft
requires: [mm-vit, mm-classification, emb-what-they-are, emb-similarity,
           emb-models, nlp-contextual]
provides: [contrastive-vision-language, shared-embedding-space,
           zero-shot-as-retrieval, modality-gap, prompt-ensembling,
           open-vocabulary-recognition, sigmoid-contrastive-loss]
citations: [radford2021clip, zhai2023siglip, oquab2023dinov2, dosovitskiy2021vit,
            russakovsky2015ilsvrc, oord2018cpc, faysse2025colpali]
---

## 1. Learning Objectives

By the end of this chapter you will be able to write the contrastive image–text
objective and say what it constrains and — crucially — what it leaves free;
explain zero-shot classification as **retrieval against text embeddings** and
derive from that why the "classifier" is a set of sentences; measure the
**modality gap** and state precisely why the loss does not remove it; show that
prompt *ensembling* is worth an order of magnitude more than prompt *selection*,
and explain the $1/\sqrt{N}$ mechanism; and say what {{cite:zhai2023siglip}}
changed about the relationship between batch size and the loss.

## 2. Why This Matters

Every vision model before this one had a fixed list of classes. Train on 1000
ImageNet categories and you have a model that can say 1000 things; a 1001st
requires new labels and new training.

{{cite:radford2021clip}} removed the list. Train an image encoder and a text
encoder together so that matching pairs are close, and classification becomes
*retrieval*: embed the image, embed a sentence describing each candidate class,
take the nearest. **The label set becomes a runtime argument.** That is the
property that made CLIP the default vision front-end for almost everything in
{{ch:mm-vlms}} and {{ch:mm-multimodal-rag}}.

**And the shared space is not what people assume it is.** The phrase suggests
image and text embeddings are now the same kind of object, interchangeable in one
index, comparable with one threshold. {{sec:9-practical-example}} measures what
they actually are: two random images score **0.009** on average, an image and its
own caption score **0.454**. A fiftyfold difference in typical value — and the
distributions overlap at the tail, with the 99th percentile of image-image
similarity at **0.444**, essentially the mean image-caption score.

**There is no number in this space that means "similar" for both kinds of
comparison.** That is not a training defect; {{eq:clip-objective}} never asked for
one, and {{sec:5-formal-explanation}} shows why it structurally cannot.

The second measurement changes a common practice. Because the classifier is a set
of sentences, wording matters — and the useful finding is *which* intervention
pays. Choosing the best of eight prompt templates instead of the worst is worth
**0.021** accuracy. **Averaging all eight is worth 0.240** — more than ten times
as much.

{{maturity:ESTABLISHED}} Contrastive vision-language pretraining.
{{maturity:MATURE}} Sigmoid-loss variants ({{cite:zhai2023siglip}}), now the more
common choice for new vision towers.

## 3. Prerequisites

{{ch:emb-what-they-are}} for contrastive learning and InfoNCE — this chapter
applies that theory across modalities and does not rederive it;
{{ch:emb-similarity}} for "the score is a rank, not a measurement", which becomes
a hard constraint here; {{ch:emb-models}} for what training an encoder involves;
{{ch:mm-vit}} for the image tower; {{ch:nlp-contextual}} for the text tower.

## 4. Intuitive Explanation

### One space, two encoders, one dot product

Take 400 million (image, caption) pairs from the web. Encode each image and each
caption into the same-dimensional space. Train so that **an image is closer to its
own caption than to any other caption in the batch**, and symmetrically.

That is the entire idea. What it buys is that a dot product between an image
vector and a text vector now means something — and therefore that anything you can
write down, you can search for.

**The supervision is the interesting part.** Nobody labelled these images. The
caption was already there, written by someone for their own reasons, and the
model's supervision is the *pairing*. That is why it scaled to 400 million when
ImageNet stopped at one million: the labels already existed.

### Classification without a classifier

To classify into $\{$cat, dog, car$\}$: embed "a photo of a cat", "a photo of a
dog", "a photo of a car", embed the image, take the nearest.

There is no trained head. **The classifier is three sentences**, and you can
change them at runtime — which is why this is called *open-vocabulary*
recognition and why {{ch:mm-segmentation}}'s class-agnostic SAM pairs with it so
naturally.

The consequence is immediate and often missed: **if the classifier is a set of
sentences, then its decision boundaries move when you rewrite the sentences.**

### The modality gap

Here is the misconception worth dismantling, because it produces real bugs.

"Shared space" suggests one cloud of points, images and texts interleaved. What
you actually get is **two clouds**, and the loss is entirely satisfied by that.

Think about what {{eq:clip-objective}} compares. An image against *texts*. A text
against *images*. **It never once compares an image against another image**, so
nothing in the objective has any opinion about where the image cloud sits relative
to the text cloud. Two well-separated regions with matching internal ordering
minimise the loss perfectly.

{{sec:9-practical-example}} measures the consequence, and the practical form is
the scale mismatch: within-modality similarities cluster near zero while matched
cross-modality pairs sit near 0.45. **A threshold calibrated on one is meaningless
applied to the other.**

### What the temperature does

The loss divides similarities by a temperature $\tau$ before the softmax, and
$\tau$ is *learned*. Small $\tau$ sharpens: the loss concentrates on the hardest
negative. Large $\tau$ flattens: all negatives contribute.

**This matters more than it looks**, because it is also what sets the scale of the
final similarities — and therefore where any threshold you pick will land. Two
CLIP models trained identically except for $\tau$ produce similarity scores on
different scales, which is another reason {{ch:emb-similarity}}'s rule holds here.

### Why the batch is the loss

CLIP's negatives are *the other items in the batch*. So batch size is not a
throughput setting — it is the number of negatives, and therefore part of the
objective. Doubling the batch changes what is being optimised.

{{cite:zhai2023siglip}} breaks that coupling by replacing the softmax over the
batch with an independent sigmoid on each pair. The loss becomes a sum over pairs
with no normalisation across the batch, so batch size returns to being an
engineering choice. **That is why SigLIP encoders largely displaced CLIP encoders
as vision towers** — not better scores so much as a better-behaved objective.

## 5. Formal Explanation

### 5.1 The objective

For a batch of $N$ pairs, with image embeddings $u_i$ and text embeddings $v_j$
both L2-normalised, and learned temperature $\tau$:

$$ \mathcal{L} = -\frac{1}{2N}\sum_{i=1}^{N}\left[ \log \frac{e^{u_i \cdot v_i/\tau}}{\sum_j e^{u_i \cdot v_j/\tau}} + \log \frac{e^{u_i \cdot v_i/\tau}}{\sum_j e^{u_j \cdot v_i/\tau}} \right] $$ (eq:clip-objective)

This is {{ch:emb-what-they-are}}'s InfoNCE ({{cite:oord2018cpc}}) applied in both
directions across modalities. Two structural facts follow, and the second is the
one this chapter is about.

**Fact one: only rankings are constrained.** {{eq:clip-objective}} depends on the
similarities only through their *differences* inside a softmax, so any
transformation preserving the ordering of $\{u_i \cdot v_j\}_j$ leaves the loss
unchanged.

**Fact two: no term compares within a modality.** Every dot product in
{{eq:clip-objective}} is $u \cdot v$ — image against text. There is no $u_i \cdot
u_j$ anywhere:

$$ \frac{\partial \mathcal{L}}{\partial (u_i \cdot u_j)} = 0 \qquad \forall\, i, j $$ (eq:no-within-modality-term)

### 5.2 The modality gap, derived

Let $\mu_I$ and $\mu_T$ be the modality centroids and consider translating the
whole image cloud: $u_i \mapsto \tilde{u}_i = \Pi(u_i + \delta)$ for a shared
$\delta$, where $\Pi$ renormalises. To first order in small $\delta$, every
similarity changes by approximately the same amount:

$$ \tilde{u}_i \cdot v_j \approx u_i \cdot v_j + \delta \cdot v_j - (u_i\cdot\delta)(u_i \cdot v_j) $$ (eq:gap-shift)

The first correction term $\delta \cdot v_j$ is **independent of $i$** — it shifts
an entire column of the similarity matrix — and a softmax is invariant to adding a
constant across... no: it is invariant to constants added across the *summed*
index. So a shared shift is *nearly* free, and

$$ \text{a separation } \|\mu_I - \mu_T\| > 0 \text{ costs the loss almost nothing} $$ (eq:modality-gap)

**{{eq:modality-gap}} is the point.** The gap is not learned *against*; it is
simply unconstrained, so it takes whatever value initialisation and optimisation
happen to leave it at. It varies with temperature, batch size, and seed — which is
why it is a property of the training run rather than of the content.

> Note the direction of the argument. This does not say a gap is *required*; it
> says the loss is indifferent, and an indifferent loss leaves whatever it
> started with. Empirically the gap is large in real CLIP models;
> {{sec:9-practical-example}}'s toy shows a smaller but clearly detectable version.

### 5.3 Zero-shot classification

For classes $c$ with prompt template $t$:

$$ \hat{y}(x) = \arg\max_{c}\; f_I(x) \cdot f_T\big(t(c)\big) $$ (eq:zero-shot-as-retrieval)

**There is no trained head**, so the boundary between classes $c$ and $c'$ is

$$ \{u : u \cdot (v_c - v_{c'}) = 0\} $$ (eq:zero-shot-boundary)

— the hyperplane perpendicular to the *difference of two text embeddings*. Change
the template and $v_c$ moves, so {{eq:zero-shot-boundary}} moves. **A zero-shot
classifier's errors are sentence errors.**

### 5.4 Why prompt ensembling works, and how much

Model a template as adding two nuisance terms to the true class direction $c$: a
**shared** offset $s_t$ (the same for every class, so carrying no class
information) and a **per-class** wording perturbation $w_{t,c}$:

$$ v_{t,c} = \Pi\big(c + s_t + w_{t,c}\big) $$ (eq:template-model)

Averaging over $T$ templates:

$$ \bar{v}_c = \Pi\!\left(c + \frac{1}{T}\sum_t s_t + \frac{1}{T}\sum_t w_{t,c}\right) $$ (eq:prompt-ensembling)

Both nuisance terms are independent across templates, so their means shrink as
$1/\sqrt{T}$ while $c$ — common to every template — is untouched:

$$ \|\text{nuisance}\| \;\propto\; \frac{1}{\sqrt{T}} $$ (eq:ensembling-rate)

**Prompt ensembling is not a trick; it is averaging out a nuisance variable**, and
{{eq:ensembling-rate}} says the returns diminish without stopping.
{{sec:9-practical-example}} measures the ensemble at **0.445** against the best
single template's **0.205**.

### 5.5 The shared offset changes the answer even though it carries no information

This is worth isolating because it is counter-intuitive. $s_t$ is added to *every*
class, so it cannot distinguish classes. Yet from {{eq:template-model}}, after
renormalisation:

$$ \Pi(c + s) \cdot u \;\ne\; \alpha\,(c \cdot u) + \beta \quad \text{for any } \alpha, \beta \text{ independent of } c $$ (eq:renormalisation-not-rank-preserving)

because $\|c + s\|$ depends on the angle between $c$ and $s$, which differs by
class. **Renormalisation makes a class-independent shift class-dependent**, and
the argmax moves. A completely uninformative change to the prompt changes
predictions.

### 5.6 Sigmoid contrastive loss

$$ \mathcal{L}_{\text{sig}} = \frac{1}{N}\sum_{i}\sum_{j} \log\Big(1 + e^{\,z_{ij}(-\alpha\, u_i \cdot v_j + b)}\Big), \qquad z_{ij} = \begin{cases} +1 & i = j \\ -1 & i \ne j\end{cases} $$ (eq:siglip)

No softmax, so no normalisation across the batch, so

$$ \frac{\partial \mathcal{L}_{\text{sig}}}{\partial(\text{batch size})} \text{ is a variance effect only, not an objective change} $$ (eq:siglip-batch-decoupled)

{{eq:siglip-batch-decoupled}} is {{cite:zhai2023siglip}}'s contribution: batch size
returns to being an engineering decision. Under {{eq:clip-objective}} it is a
*hyperparameter of the loss*, which makes small-scale replication of CLIP results
genuinely difficult.

## 6. Mathematical Foundation

### 6.1 The two similarity scales, worked

From the measurement: image–image mean **0.009**, image–own-caption mean
**0.454**, image–random-caption mean **0.001**.

The within-modality mean near zero is expected — two random unit vectors in $d$
dimensions have expected cosine 0 with standard deviation $1/\sqrt{d}$, and at
$d = 32$ that is $0.177$, against a measured spread (p5 to p95) of $-0.308$ to
$0.325$, i.e. about $\pm 1.8$ standard deviations. **The image cloud is
essentially isotropic.**

The cross-modal *matched* mean of 0.454 is what training produced. So:

$$ \frac{\text{matched cross-modal}}{\text{typical within-modal}} = \frac{0.454}{0.009} \approx 50 $$ (eq:scale-ratio)

**But the distributions overlap.** The 99th percentile of image–image similarity
is **0.444**, essentially the mean matched-caption score. A threshold at 0.45
admits a typical caption match *and* the top 1% of image pairs; a threshold at
0.05 admits half the image pairs and rejects nothing cross-modal.

$$ \nexists\, \theta \text{ separating "similar" from "not" for both comparisons} $$ (eq:no-universal-threshold)

### 6.2 Retrieval works while the gap persists

Top-1 retrieval was **0.173** against 2000 candidates, where chance is
**0.0005** — a factor of **346**. The alignment is real.

That is the whole tension in one pair of numbers: **retrieval is excellent and the
clouds are still separable.** A linear probe recovers the modality at
**0.677** (chance 0.5), and a typical image embedding sits closer to the image
centroid (**0.089**) than to the text centroid (**0.006**).

{{eq:no-within-modality-term}} explains why there is no contradiction. The loss
optimised ranking across the gap and had no term that could have closed it.

### 6.3 Ensembling against selection, quantified

From the measurement: eight templates spanning $[0.184, 0.205]$, ensemble
$0.445$, oracle $0.560$.

$$ \text{gain from selecting best} = 0.205 - 0.184 = 0.021 $$
$$ \text{gain from averaging} = 0.445 - 0.205 = 0.240 $$ (eq:ensemble-vs-select)

**A ratio of about 11.** And the ensemble recovers $0.445/0.560 = 79\%$ of the gap
to the oracle, with the residual being the part of $w_{t,c}$ that eight samples
cannot average away — consistent with {{eq:ensembling-rate}}'s $1/\sqrt{8} =
0.354$.

> **MATH NOTE:** The single-template accuracies are far below the oracle (0.20
> against 0.56), which is more degradation than real CLIP shows from prompt
> choice alone. The toy's per-class wording perturbation is deliberately large so
> the ensembling mechanism is visible in eight samples rather than eighty. **The
> ordering and the $1/\sqrt{T}$ mechanism are the transferable parts**; the
> magnitude is a modelling choice, and in practice ensembling buys a few points
> rather than twenty.

## 7. Internal Mechanics

```mermaid {#fig:clip-training caption="Training and inference share one space and use it differently. The loss (dashed) only ever compares across the diagonal — image against text — which is why eq:no-within-modality-term holds and why the two clouds are free to separate. At inference the class list is text, so it is an argument rather than a trained head."}
flowchart TB
    IMG["images"] --> IE["image tower<br/>(ch:mm-vit)"] --> U["u, normalised"]
    TXT["captions"] --> TE["text tower"] --> V["v, normalised"]
    U --> S["N x N similarity matrix<br/>u . v / tau"]
    V --> S
    S -.->|"softmax over ROWS<br/>and over COLUMNS;<br/>never within a modality"| L["contrastive loss"]
    U2["new image"] --> IE
    C["'a photo of a {class}'<br/>for each candidate class"] --> TE
    IE --> ZS["nearest text = prediction"]
    TE --> ZS
```

### 7.1 What the data actually was

400 million web (image, alt-text) pairs. Two properties of that corpus decide
what the model can and cannot do:

- **The caption describes what a person thought worth mentioning.** Common objects
  are named; fine-grained species, technical readings, exact counts and spatial
  relations mostly are not. So CLIP is excellent at "is there a dog" and poor at
  counting, at fine-grained distinctions, and at spatial relations.
- **Web text is not uniform.** It concentrates on the popular, the photographed,
  and the English-speaking, and every downstream system inherits that.

**Zero-shot performance is a property of the pretraining distribution**, and it
does not transfer to a domain the web does not caption — medical imaging,
industrial inspection, satellite imagery. Checking that before adopting a zero-shot
pipeline saves a great deal of time.

### 7.2 Prompt engineering as classifier design

Since {{eq:zero-shot-boundary}} depends on the text embeddings, wording is
classifier design:

| Choice | Why it matters |
|---|---|
| `"cat"` vs `"a photo of a cat"` | captions are sentences; bare nouns are off-distribution |
| domain framing (`"a satellite photo of..."`) | moves the whole class set toward the right region |
| disambiguation (`"a crane (bird)"`) | the text tower has no idea which sense you meant |
| ensembling over templates | {{eq:prompt-ensembling}} — worth ~11× selection |

**The third row is a failure mode people hit and misdiagnose.** A polysemous class
name embeds as a blend of its senses, and the fix is in the sentence, not the
model.

### 7.3 What CLIP features are and are not good for

Two distinct uses, and they have different quality profiles:

- **As a shared space** for retrieval and zero-shot classification — what it was
  trained for, and it is excellent.
- **As a frozen image backbone** for dense tasks — segmentation, depth,
  correspondence. Here {{cite:oquab2023dinov2}} is the counterweight: purely
  self-supervised features, trained with no language at all, are *better* at
  pixel-level tasks.

The reason is in the objective. {{eq:clip-objective}} rewards whatever
distinguishes captions, and captions describe *what is in the image*, not *where*
or *how much*. **Language supervision optimises for the information language
carries**, which is a subset of what an image contains.

## 8. Implementation

```python {tier=A name=modality-gap}
"""The modality gap: CLIP's shared space is shared, and it is not one region.

Contrastive image-text training puts both modalities in one vector space so a
dot product can compare them (eq:clip-objective). The natural conclusion -- that
an image embedding and a text embedding are now the same kind of object -- is
false, and the way it is false breaks thresholds, clustering, and any index that
mixes modalities.

Image embeddings and text embeddings occupy SEPARATE CONES. The contrastive loss
only ever compares an image against texts and a text against images, so nothing in
it ever asks the two clouds to coincide -- it asks for the right RANKING across
the gap, which a pair of well-separated cones satisfies perfectly
(eq:modality-gap).

This listing trains a small two-tower model and measures the gap, then measures
what the gap does to a similarity threshold.
"""
import numpy as np

rng = np.random.default_rng(53)

N_CONCEPT, DIM_RAW, DIM = 20000, 64, 32
STEPS, BATCH, LR = 4000, 256, 0.5
TAU = 0.07


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


# A latent concept per training pair. The two encoders see the SAME concept
# through different, modality-specific transformations -- which is the situation
# contrastive alignment is meant to handle.
concept = unit(rng.normal(size=(N_CONCEPT, DIM_RAW)))
A_img = rng.normal(size=(DIM_RAW, DIM_RAW)) / np.sqrt(DIM_RAW)
A_txt = rng.normal(size=(DIM_RAW, DIM_RAW)) / np.sqrt(DIM_RAW)
b_img = rng.normal(size=DIM_RAW) * 0.6          # modality-specific offset
b_txt = rng.normal(size=DIM_RAW) * 0.6


def sample(n, distinct=False):
    c = (rng.choice(N_CONCEPT, size=n, replace=False) if distinct
         else rng.integers(0, N_CONCEPT, size=n))
    xi = unit(concept[c] @ A_img + b_img + 0.15 * rng.normal(size=(n, DIM_RAW)))
    xt = unit(concept[c] @ A_txt + b_txt + 0.15 * rng.normal(size=(n, DIM_RAW)))
    return xi, xt, c


Wi = rng.normal(size=(DIM_RAW, DIM)) / np.sqrt(DIM_RAW)
Wt = rng.normal(size=(DIM_RAW, DIM)) / np.sqrt(DIM_RAW)

for step in range(STEPS):
    xi, xt, _ = sample(BATCH)
    zi, zt = unit(xi @ Wi), unit(xt @ Wt)
    S = zi @ zt.T / TAU
    Pi = np.exp(S - S.max(1, keepdims=True)); Pi /= Pi.sum(1, keepdims=True)
    Pt = np.exp(S - S.max(0, keepdims=True)); Pt /= Pt.sum(0, keepdims=True)
    tgt = np.eye(BATCH)
    gS = ((Pi - tgt) + (Pt - tgt).T) / (2 * BATCH * TAU)
    gzi, gzt = gS @ zt, gS.T @ zi
    # Backprop through the L2 normalisation.
    def dnorm(g, z, x):
        n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
        return (g - (g * z).sum(1, keepdims=True) * z) / n
    Wi -= LR * (xi.T @ dnorm(gzi, zi, xi @ Wi))
    Wt -= LR * (xt.T @ dnorm(gzt, zt, xt @ Wt))

xi, xt, c = sample(2000, distinct=True)
zi, zt = unit(xi @ Wi), unit(xt @ Wt)

# --- does retrieval work? ---
S = zi @ zt.T
top1 = float((S.argmax(1) == np.arange(len(S))).mean())

# --- the gap, measured as SEPARABILITY rather than as centroid distance ---
# Centroid distance is misleading on a sphere: both means sit near the origin.
# The question that matters is whether the two clouds are distinguishable at
# all, so fit a linear probe to predict which modality an embedding came from.
mi, mt = zi.mean(0), zt.mean(0)
gap = float(np.linalg.norm(mi - mt))
Z = np.vstack([zi, zt])
lab = np.concatenate([np.zeros(len(zi)), np.ones(len(zt))])
Zc = np.hstack([Z, np.ones((len(Z), 1))])
w = np.linalg.lstsq(Zc, 2 * lab - 1, rcond=None)[0]
separability = float(((Zc @ w > 0) == (lab > 0)).mean())
# How far is a typical embedding from its OWN modality's centroid, versus from
# the other modality's? If the clouds were interleaved these would be equal.
own = float(np.mean([unit(mi[None])[0] @ z for z in zi[:800]]))
other = float(np.mean([unit(mt[None])[0] @ z for z in zi[:800]]))

# --- similarity distributions ---
def offdiag(M):
    return M[~np.eye(len(M), dtype=bool)]

ii = offdiag(zi[:600] @ zi[:600].T)
tt = offdiag(zt[:600] @ zt[:600].T)
it_pos = np.diag(zi[:600] @ zt[:600].T)
it_neg = offdiag(zi[:600] @ zt[:600].T)

print(f"trained {STEPS} steps, batch {BATCH}, temperature {TAU}\n")
print(f"image->text retrieval, top-1 of 2000:      {top1:.3f}")
print(f"difference of modality means (norm):       {gap:.3f}")
print(f"chance retrieval rate would be:             {1/len(S):.4f}")
print(f"linear probe -- which modality is this?     {separability:.3f}")
print(f"mean cosine to OWN modality centroid:       {own:.3f}")
print(f"mean cosine to OTHER modality centroid:     {other:.3f}")
print()
print(f"{'similarity between':<34}{'mean':>9}{'p5':>9}{'p95':>9}")
print("-" * 61)
for name, v in (("two images", ii), ("two texts", tt),
                ("an image and ITS text", it_pos),
                ("an image and a random text", it_neg)):
    print(f"{name:<34}{v.mean():>9.3f}{np.percentile(v, 5):>9.3f}"
          f"{np.percentile(v, 95):>9.3f}")

ii_hi = float(np.percentile(ii, 99))
print(f"""
Retrieval works, and by a wide margin: {top1:.3f} top-1 against 2000 candidates
where chance is {1/len(S):.4f}. The alignment succeeded. By the measure the
objective optimised, there is nothing wrong with this space.

Now the two similarity scales, which is the result. Two random images score
{ii.mean():.3f} on average; an image and its own caption score {it_pos.mean():.3f}.
Those are not the same scale, and the gap is not a quality difference -- it is
where each kind of comparison LIVES. Cross-modal matched pairs sit far above
everything, and within-modality pairs cluster near zero however related their
content is.

So a threshold means different things depending on what it is comparing, and the
two distributions are not merely offset -- they overlap at the tail while
differing fiftyfold in the middle. The 99th percentile of image-image similarity
is {ii_hi:.3f}, which is essentially the MEAN image-caption score
({it_pos.mean():.3f}). So a cutoff at 0.45 admits a typical matched caption and
also the top one per cent of image pairs, while a cutoff at 0.05 admits half the
image pairs and rejects nothing cross-modal. There is no value that separates
"similar" from "not similar" for both kinds of comparison at once.

The linear probe puts the structural version of this at {separability:.3f}: an
embedding carries enough information about WHICH MODALITY produced it that a
linear classifier beats chance at recovering it, and a typical image embedding
sits closer to the image centroid ({own:.3f}) than to the text centroid
({other:.3f}). Be aware that this toy UNDERSTATES the effect -- with real
encoders, deeper towers and web-scale data, the two clouds are close to perfectly
separable, and the reported gap is much larger than the one measured here. The
direction is right and the magnitude is a lower bound.

The reason is in eq:clip-objective and it is not a training failure. The loss only
ever compares an image against texts and a text against images. It therefore
constrains the RANKING across the gap and says nothing whatsoever about where
either cloud sits, so two separated cones with matching internal ordering minimise
it perfectly. Nothing was ever asked to bring them together.

This is ch:emb-similarity's rule with a second modality attached: the score is a
rank, not a measurement. Within one modality that is a caution. Across two it is a
hard constraint, because the offset between the clouds is a property of the
training run -- it moves with temperature, batch size and initialisation -- rather
than a property of the content.

The practical response is the one ch:emb-what-they-are used for anisotropy:
centre each modality separately before comparing within it, calibrate any
threshold on the specific comparison it will be applied to, and never compare a
within-modality score against a cross-modality one. What you must not do is
assume the shared space made the two interchangeable. It made them comparable by
ranking, which is strictly weaker, and is the only thing the loss requested.""")
```

The first listing is about the space. The second is about the classifier built
from it.

```python {tier=A name=prompt-sensitivity}
"""Zero-shot classification is retrieval, and that changes what it depends on.

CLIP classifies without a classifier: embed the image, embed a text description of
each candidate class, and take the nearest (eq:zero-shot-as-retrieval). No head is
trained, no class list is fixed at training time, and the label set can change at
runtime -- which is the property that made the technique matter.

It also means the "classifier" is a set of text embeddings, so its decision
boundary is determined by how the classes were WORDED (eq:zero-shot-boundary).
This listing measures that dependence, and measures the standard mitigation.

The class-name geometry here is deliberately not uniform: some classes are near
neighbours in the shared space and some are isolated, because that is what a real
label set looks like and it is what makes prompt wording matter unevenly.
"""
import numpy as np

rng = np.random.default_rng(61)

N_CLASS, DIM = 40, 48
N_IMG = 4000
N_TEMPLATE = 8


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


# True class directions in the shared space. Half the classes are grouped into
# tight clusters of near-synonyms, which is where prompt wording does damage.
base = unit(rng.normal(size=(N_CLASS // 2, DIM)))
cls_dir = np.zeros((N_CLASS, DIM))
cls_dir[:N_CLASS // 2] = base
for i in range(N_CLASS // 2):
    cls_dir[N_CLASS // 2 + i] = unit(base[i] + 0.45 * rng.normal(size=DIM))
cls_dir = unit(cls_dir)

# Images: the class direction plus content the caption never mentions.
y = rng.integers(0, N_CLASS, size=N_IMG)
img = unit(cls_dir[y] + 0.42 * rng.normal(size=(N_IMG, DIM)))

# A prompt template shifts every class embedding by a shared, template-specific
# direction ("a photo of a {}", "a blurry photo of a {}", ...) and adds a small
# per-class wording effect (eq:template-model). The shared component is the
# interesting one: it is the same for every class, so it cannot carry class
# information, and it still moves the decision boundary.
TEMPLATES = []
for t in range(N_TEMPLATE):
    shared = unit(rng.normal(size=DIM)) * rng.uniform(0.15, 0.75)
    per_class = 0.22 * rng.normal(size=(N_CLASS, DIM))
    TEMPLATES.append(unit(cls_dir + shared + per_class))


def accuracy(text_emb):
    return float((img @ text_emb.T).argmax(1).__eq__(y).mean())


print(f"{N_CLASS} classes ({N_CLASS // 2} of them near-synonym pairs), "
      f"{N_IMG} images\n")
print(f"{'prompt template':<22}{'accuracy':>11}")
print("-" * 34)
accs = []
for t in range(N_TEMPLATE):
    a = accuracy(TEMPLATES[t])
    accs.append(a)
    print(f"{'template ' + str(t + 1):<22}{a:>11.3f}")

oracle = accuracy(cls_dir)
ens = accuracy(unit(np.mean(TEMPLATES, axis=0)))
best_single = max(accs)

print("-" * 34)
print(f"{'worst template':<22}{min(accs):>11.3f}")
print(f"{'best template':<22}{best_single:>11.3f}")
print(f"{'ENSEMBLE of all 8':<22}{ens:>11.3f}")
print(f"{'oracle class direction':<22}{oracle:>11.3f}")

print(f"""
Read the single-template rows first. They span {min(accs):.3f} to
{best_single:.3f} -- a spread of {best_single - min(accs):.3f} from nothing but
how the classes were phrased. Identical images, identical class set, identical
model. Only the sentences standing in for the classifier changed.

That is eq:zero-shot-as-retrieval's direct consequence and it is easy to miss.
There is no trained head, so the boundary between two classes is the
perpendicular bisector of their two TEXT embeddings, and a template moves those
embeddings. A zero-shot classifier IS a set of sentences, so its errors are
sentence errors.

Now compare every single template against the oracle row, {oracle:.3f}, which
uses the true class directions with no wording at all. The best template reaches
{best_single:.3f}. Wording is not costing a couple of points here -- it is costing
most of the available accuracy, because each template adds a direction to every
class embedding that has nothing to do with the class.

And now the row that changes what you should DO about it. The ensemble -- the
mean of all eight templates' embeddings, re-normalised -- scores {ens:.3f}. That
is {ens - best_single:.3f} above the best single template, while CHOOSING the
best template rather than the worst was worth only
{best_single - min(accs):.3f}. Averaging is worth {(ens - best_single) / max(best_single - min(accs), 1e-9):.0f} times as much as
selecting.

The mechanism is why this generalises. Each template contributes two nuisance
terms: a shared direction added to every class, and a per-class wording effect.
Neither carries class information, and both are independent across templates, so
their average shrinks like 1/sqrt(N) while the class direction -- common to every
template -- survives untouched (eq:ensembling-rate). Prompt ensembling is not a
trick; it is averaging out a nuisance variable, and the square-root law says the
returns diminish but do not stop.

The shared component deserves one note because it is counter-intuitive. It is the
SAME vector added to every class, so it cannot possibly carry information that
distinguishes classes -- and it still changes the answer, because adding a
constant vector to points on a sphere and re-normalising does not preserve their
ranking against a query (eq:renormalisation-not-rank-preserving). A perfectly
uninformative change to the prompt moves the decision boundary.

Note also where the damage concentrates. Half of these classes are near-synonym
pairs sitting close together in the space, and those are the pairs a template
shift can reorder; well-separated classes survive any wording. That is why prompt
sensitivity shows up in practice as confusion between specific confusable pairs
rather than as a uniform drop, and why a per-class error breakdown is the right
way to look for it.

The practical conclusion: a zero-shot classifier needs a validation set as much as
a trained one, not to fit weights but to choose sentences -- and the first thing
to do with it is not to pick the best prompt, it is to stop picking one.""")
```

## 9. Practical Example

**Retrieval works and the clouds stay apart.** Top-1 retrieval is **0.173** against
2000 candidates where chance is **0.0005** — a factor of **346**. The alignment
succeeded by the measure the objective optimised.

**And the space is still two spaces.** A linear probe recovers which modality
produced an embedding at **0.677**, and a typical image embedding sits closer to
the image centroid (**0.089**) than to the text centroid (**0.006**).
{{eq:no-within-modality-term}} says there was never a term that could have closed
that.

**The practical form is the scale mismatch.** Two random images: **0.009**. An
image and its own caption: **0.454** — a fiftyfold difference in typical value.

> **IMPORTANT:** The distributions do not merely sit at different offsets; they
> **overlap at the tail**. The 99th percentile of image-image similarity is
> **0.444**, essentially the mean image-caption score. So a cutoff at 0.45 admits
> a typical matched caption *and* the top 1% of image pairs, while a cutoff at
> 0.05 admits half the image pairs and rejects nothing cross-modal.
> {{eq:no-universal-threshold}}: **no single value means "similar" for both kinds
> of comparison.** Any mixed-modality index with one threshold is comparing
> quantities that are not commensurable, and its errors will be systematic by
> modality rather than random.

This toy **understates** the effect. With real encoders and web-scale data the
clouds are close to perfectly separable; 0.677 is a lower bound, and the direction
is what transfers.

**Prompt sensitivity, and the intervention that actually pays.** Eight templates
span **0.184 to 0.205** — a spread of 0.021 from wording alone, with identical
images, classes, and model. {{eq:zero-shot-boundary}}: the boundary is the
perpendicular bisector of two *text* embeddings, so a zero-shot classifier's
errors are sentence errors.

Every single template sits far below the oracle's **0.560**. **And the ensemble
reaches 0.445** — **0.240 above the best single template**, where *choosing* the
best template instead of the worst was worth only **0.021**.

**Averaging is worth about eleven times as much as selecting.** The mechanism is
{{eq:ensembling-rate}}: both nuisance terms are independent across templates and
shrink as $1/\sqrt{T}$, while the class direction is common to all of them and
survives. The ensemble recovers 79% of the gap to the oracle, consistent with
$1/\sqrt{8} = 0.354$ of the nuisance remaining.

**And the shared component changes the answer while carrying no information.** It
is added to every class identically, so it cannot distinguish them — yet
{{eq:renormalisation-not-rank-preserving}} means renormalising after a shared
shift is not rank-preserving, so the argmax moves anyway.

## 10. Production Considerations

**Never apply one similarity threshold across modalities**
({{eq:no-universal-threshold}}). Calibrate per comparison type.

**Centre each modality separately** before any within-modality operation —
clustering, deduplication, nearest-neighbour. Same fix as
{{ch:emb-what-they-are}}'s anisotropy correction.

**Ensemble prompts; do not hunt for the best one.** Eleven times the return, and
it is three lines.

**Build a validation set for a zero-shot classifier.** Not to fit weights — to
choose sentences, and to find the confusable pairs where wording bites.

**Check your domain is one the web captions.** Zero-shot performance is a property
of the pretraining distribution, and medical, industrial and satellite imagery are
mostly outside it.

**Use CLIP/SigLIP features for retrieval and semantics; use a self-supervised
backbone for dense tasks** ({{cite:oquab2023dinov2}}).

**Pin the model version with the index.** The gap, the temperature and the
similarity scale are all training-run properties, so embeddings from two CLIP
versions are not comparable — {{ch:emb-what-they-are}}'s versioned-schema rule,
with an extra reason.

**Disambiguate polysemous class names in the prompt.** The text tower cannot know
which sense you meant.

## 11. Common Mistakes

**Treating "shared space" as "interchangeable vectors".**

**One threshold for image-image and image-text similarity.**

**Reporting a raw CLIP similarity as an absolute quality score.**

**Picking the best prompt template instead of averaging.**

**Using bare class names** rather than sentences — off-distribution for a
caption-trained text tower.

**Using CLIP features for segmentation or depth** and concluding vision models are
bad at it.

**Assuming zero-shot transfers to a specialist domain.**

**Comparing embeddings across model versions.**

## 12. Failure Modes

**Threshold nonsense in a mixed index.** Symptom: retrieval quality differs
systematically by modality. Cause: {{eq:no-universal-threshold}}.

**Confusable-pair collapse.** Symptom: two similar classes are consistently
confused while everything else is fine. Cause: {{eq:zero-shot-boundary}} with
close text embeddings. Fix in the prompt.

**Polysemy.** Symptom: one class behaves as though it means something else.
Cause: the class name has two senses and the embedding is a blend.

**Domain mismatch.** Symptom: zero-shot accuracy near chance on a specialist
corpus. Not fixable by prompting.

**Counting and spatial failures.** Symptom: "three cats" and "a cat left of a
dog" fail. Cause: captions rarely state these, so the objective never rewarded
them.

**Batch-size dependence.** Symptom: a small-scale CLIP reproduction underperforms
inexplicably. Cause: {{eq:clip-objective}} makes batch size part of the loss —
{{cite:zhai2023siglip}} exists for this.

**Version skew.** Symptom: retrieval degrades after a model upgrade with no code
change. Cause: mixed-version embeddings in one index.

## 13. Alternatives

| Alternative | Trades away | When it wins |
|---|---|---|
| SigLIP ({{cite:zhai2023siglip}}) | nothing much | new vision towers; batch size decoupled |
| DINOv2 ({{cite:oquab2023dinov2}}) | language alignment | dense tasks, frozen features |
| supervised classifier | open vocabulary | fixed class set, labels available |
| linear probe on CLIP features | zero-shot flexibility | some labels available; usually much better |
| fine-tuned CLIP | zero-shot generality | in-domain accuracy matters most |
| late-interaction VLM ({{cite:faysse2025colpali}}) | single-vector efficiency | document retrieval |

**The fourth row is the most under-used option.** If you have even a hundred
labelled examples per class, a linear probe on frozen CLIP features usually beats
zero-shot by a wide margin — and people reach for prompt engineering instead
because zero-shot is the advertised feature.

## 14. Evaluation

**Report the prompt template**, or the result is not reproducible. This is as
important as reporting the model.

**Report per-class accuracy**, because prompt sensitivity concentrates on
confusable pairs and an aggregate hides it.

**Calibrate thresholds per comparison type**, and report which.

**Compare against a linear probe** before claiming a zero-shot result is good.

**Evaluate on your domain, not on {{cite:russakovsky2015ilsvrc}}.** Zero-shot
ImageNet accuracy says little about a specialist corpus.

**For retrieval, report both directions.** Image→text and text→image can differ,
and the asymmetry is informative about which tower is weaker.

## 15. Advanced Concepts

**The gap is a free parameter, and it can be closed after the fact.**
{{maturity:EMERGING}} Because {{eq:modality-gap}} shows the loss is indifferent,
the gap can be reduced post-hoc by translating one cloud — and doing so changes
cross-modal similarity scales without changing rankings. Whether that helps
depends entirely on whether anything downstream uses absolute scores.

**Temperature sets the score scale.** {{maturity:MATURE}} The learned $\tau$
controls how sharply the loss weights hard negatives *and* the spread of final
similarities. Two models with different $\tau$ produce incomparable scores, which
is a second reason to pin versions.

**Sigmoid loss decouples batch from objective.**
{{maturity:MATURE}} {{eq:siglip-batch-decoupled}} is a bigger practical
contribution than its benchmark numbers, because it makes small-scale work
meaningful.

**Language supervision optimises for what language carries.**
{{maturity:ESTABLISHED}} {{cite:oquab2023dinov2}}'s result — self-supervised
features beating language-supervised ones on dense tasks — follows from
{{eq:clip-objective}} rewarding only caption-distinguishing information. **Choose
the supervision to match the downstream task, not the popularity of the method.**

**Compositionality is the standing weakness.** {{maturity:EMERGING}} CLIP-style
models behave substantially like bags of concepts: "a red cube on a blue sphere"
and "a blue cube on a red sphere" embed similarly. The contrastive objective
rarely needs to distinguish them, because the negatives in a random batch rarely
differ only by composition — a data problem as much as an architectural one.

## 16. Connection to Previous Chapters

{{ch:emb-what-they-are}}'s InfoNCE is {{eq:clip-objective}} with two towers, and
its anisotropy correction is the fix for the modality gap.
{{ch:emb-similarity}}'s "the score is a rank, not a measurement" becomes
{{eq:no-universal-threshold}} — a caution within one modality and a hard
constraint across two. {{ch:mm-vit}} is the image tower, and its
{{eq:patch-compression}} bounds what detail can reach this space at all.
{{ch:mm-classification}}'s fixed 1000-way head is what
{{eq:zero-shot-as-retrieval}} replaces. Forward: {{ch:mm-vlms}} uses this tower as
its vision front-end, {{ch:mm-multimodal-rag}} builds an index in this space and
inherits the gap, and {{ch:mm-segmentation}}'s class-agnostic SAM is the natural
partner for an open-vocabulary classifier.

## 17. Exercises

1. Show from {{eq:clip-objective}} that no term involves $u_i \cdot u_j$, and
   state what that implies about within-modality geometry.
2. Derive {{eq:renormalisation-not-rank-preserving}} and give a two-dimensional
   example where a shared shift changes the argmax.
3. In `modality-gap`, set `b_img` and `b_txt` to zero. What happens to the linear
   probe, and what does that tell you about the gap's origin?
4. In the same listing, sweep `TAU` over $\{0.02, 0.07, 0.2\}$. How do the
   similarity scales move, and what does that imply for a fixed threshold?
5. Implement per-modality centring and re-measure the probe and the similarity
   tables. What improves and what does not?
6. In `prompt-sensitivity`, sweep the number of ensembled templates from 1 to 32.
   Does the gain follow {{eq:ensembling-rate}}'s $1/\sqrt{T}$?
7. Set the per-class wording term to zero, leaving only the shared shift. Does
   ensembling still help, and by how much?
8. Take a real CLIP model. Measure its modality gap and its image-image and
   image-text similarity distributions on your own data, then check whether any
   threshold in your system is applied across both.

## 18. Interview Questions

1. How does CLIP classify without a classifier?
2. What supervision does CLIP use, and why did that let it scale?
3. What is the modality gap, and why does the loss not remove it?
4. Can you use one similarity threshold for image-image and image-text? Justify.
5. Why does prompt wording change a zero-shot prediction?
6. Is it better to pick the best prompt or to average several? By how much?
7. Why is batch size part of CLIP's loss, and what does SigLIP change?
8. When would you use DINOv2 instead of CLIP?
9. Your zero-shot classifier confuses two specific classes. What do you try?
10. You have 100 labelled examples per class. What should you do?

## 19. Research Questions

1. {{eq:modality-gap}} says the gap is unconstrained. Does closing it post-hoc
   ever improve a downstream task, or is it purely cosmetic?
2. {{eq:ensembling-rate}} predicts $1/\sqrt{T}$. Is there a template-selection
   strategy that beats random averaging by choosing *diverse* nuisance
   directions?
3. Compositional failures follow from batch negatives rarely differing by
   composition. Would hard compositional negatives fix it, and at what cost to
   general performance?
4. Language supervision optimises for what captions carry. Can the missing
   information — counts, spatial relations — be added by a targeted auxiliary
   objective without diluting the main one?
5. The learned temperature sets the score scale. Is there a parameterisation
   under which similarities are comparable across training runs?

## 20. Chapter Summary

{{cite:radford2021clip}} removed the fixed class list. Train two encoders so
matching image–text pairs are close, and classification becomes **retrieval
against sentences** ({{eq:zero-shot-as-retrieval}}) — the label set is a runtime
argument, and the supervision was already on the web.

**The shared space is shared and it is not one region.** Retrieval works — **0.173
top-1 against 2000 candidates where chance is 0.0005** — while a linear probe
still recovers the modality at **0.677**. There is no contradiction:
{{eq:no-within-modality-term}} shows the loss contains no term comparing an image
to an image, so {{eq:modality-gap}} leaves the separation unconstrained. **It is
not a training defect; it is an indifference.**

**The practical form is that no threshold means one thing.** Two random images
score **0.009**, an image and its caption **0.454** — fiftyfold apart in the
middle and *overlapping at the tail*, with image-image p99 at **0.444** against a
matched-caption mean of **0.454**. {{eq:no-universal-threshold}}. Centre each
modality separately, calibrate per comparison, and never compare a within-modality
score to a cross-modality one.

**A zero-shot classifier is a set of sentences**, so {{eq:zero-shot-boundary}}
makes its errors sentence errors — concentrated on confusable pairs rather than
spread uniformly.

**And the intervention that pays is not the one people reach for.** Choosing the
best of eight templates instead of the worst was worth **0.021**; averaging all
eight was worth **0.240** — about **eleven times** as much. {{eq:ensembling-rate}}
explains it: the nuisance terms are independent across templates and shrink as
$1/\sqrt{T}$ while the class direction survives. **Prompt ensembling is averaging
out a nuisance variable, not a hack.** More strikingly, a shared shift that
carries *no class information whatsoever* still moves predictions, because
{{eq:renormalisation-not-rank-preserving}} makes renormalisation
non-rank-preserving.

**Finally, choose supervision to fit the task.** {{eq:clip-objective}} rewards
whatever distinguishes *captions*, and captions describe what is in an image
rather than where or how much — which is why {{cite:oquab2023dinov2}}'s
purely self-supervised features beat language-supervised ones on dense tasks, and
why CLIP is weak at counting and spatial relations. And why
{{cite:zhai2023siglip}}'s sigmoid loss matters: it removes batch size from the
objective, which is what makes this work reproducible outside a large lab.

## 21. Further Reading

{{cite:radford2021clip}} for the method, and read Section 3.1.4 on prompt
engineering — it is the part that turns out to matter most in practice, and it is
usually skipped.
{{cite:zhai2023siglip}} for the sigmoid loss and for why decoupling batch size
from the objective is a bigger deal than the benchmark delta.
{{cite:oquab2023dinov2}} as the counterweight: language supervision is neither
necessary for strong visual features nor best for dense tasks.
{{cite:oord2018cpc}} for InfoNCE itself, developed properly in
{{ch:emb-what-they-are}}.
{{cite:dosovitskiy2021vit}} for the tower, and {{cite:faysse2025colpali}} for what
happens when you keep the patch tokens instead of pooling to one vector.
{{cite:russakovsky2015ilsvrc}} for the benchmark zero-shot numbers are quoted
against, and for why that number says less about your domain than it appears to.

---
id: mm-detection
number: 120
part: XIII
tier: full
status: draft
requires: [mm-cv-fundamentals, mm-classification, ml-metrics, dl-losses]
provides: [detection-as-set-prediction, intersection-over-union,
           non-maximum-suppression, anchor-assignment, average-precision-mechanics,
           bipartite-matching-loss, crowd-ambiguity, confidence-threshold-gap]
citations: [ren2015fasterrcnn, redmon2016yolo, carion2020detr, he2017maskrcnn,
            lin2014coco, lin2017focal, he2016resnet]
---

## 1. Learning Objectives

By the end of this chapter you will be able to state detection as **set
prediction** and explain why that framing, not the architecture, is what
distinguishes the three detector families; compute IoU, NMS, and average
precision from their definitions; demonstrate that NMS embeds a false assumption
about crowded scenes and measure what it costs; explain why **average precision
barely punishes duplicates and never punishes low-scored junk**, and what that
means for anything you deploy; and say precisely what bipartite matching replaces
and what it costs to replace it.

## 2. Why This Matters

Classification answers *what*. Detection answers *what and where, and how many* —
and that last part is what makes it structurally different from everything in
{{ch:mm-classification}}.

A classifier's output has a fixed shape: one distribution over classes. A
detector's output is a **set of variable size**, and neither the loss functions
nor the metrics of ordinary supervised learning handle sets. Everything strange
about detection follows from that: anchors exist because a network cannot emit a
variable-length list; non-maximum suppression exists because anchors produce
duplicates; and mAP exists because accuracy is undefined for a set.

**Two of those three are hand-designed components sitting outside the model**, and
{{sec:9-practical-example}} shows what they cost. NMS assumes that two heavily
overlapping boxes are two detections of one object — which is exactly wrong in a
crowd. Measured: a badly chosen threshold costs 0.064 AP on well-separated
objects and **0.282 on crowded ones**, a 4.4× amplification of the same mistake,
with the damage concentrated in the images that contain the most objects.

And the metric has its own surprises. **Appending five hundred junk boxes per
image does not reduce mAP** — it slightly raises it. A detector optimised for the
reported number therefore emits hundreds of predictions per image, and the
threshold you actually ship is a number the metric never told you.

{{maturity:ESTABLISHED}} Two-stage and single-stage detectors.
{{maturity:MATURE}} Set-prediction detectors ({{cite:carion2020detr}}), now the
basis of most new work.

## 3. Prerequisites

{{ch:mm-cv-fundamentals}} for the receptive field and the jump, which decide what
a detector can localise — {{eq:erf-worked}} in particular;
{{ch:mm-classification}} for the backbone being consumed here;
{{ch:ml-metrics}} for precision and recall, which mAP composes in a specific and
consequential way; {{ch:dl-losses}} for what a loss over a *set* has to do
differently.

## 4. Intuitive Explanation

### The problem is that the answer is a set

$$ \text{image} \;\longrightarrow\; \big\{(b_1, c_1), \dots, (b_n, c_n)\big\}, \quad n \text{ unknown} $$

Three difficulties come free with that, and every detector is a set of answers to
them:

1. **Variable length.** A network emits fixed-size tensors. How do you produce a
   variable number of things?
2. **No canonical order.** $\{A, B\}$ and $\{B, A\}$ are the same answer, so a
   loss comparing them position-by-position is measuring something meaningless.
3. **Matching.** To compute *any* loss you must first decide which prediction
   corresponds to which ground truth — and that decision is itself part of the
   design.

### The classical answer, and the bill it leaves

Fix the length: predict a box for every one of a large fixed set of **anchors**
tiled over the image, and let most of them say "nothing here". Length problem
solved. Ordering problem solved, because anchors have a canonical order. Matching
solved by a rule: an anchor is responsible for a ground truth if it overlaps it
enough.

**And that rule creates the duplicate problem.** Several anchors overlap the same
object, all are trained to fire, so one object produces several boxes. Something
must remove the extras, and that something is NMS: sort by score, keep the
highest, delete anything overlapping it by more than $t$.

Now look at the assumption NMS is making:

> **"These two boxes overlap a lot, therefore they are two detections of one
> object."**

In a photograph of one dog that is true. In a photograph of a crowd, two people
standing shoulder to shoulder genuinely have overlapping boxes, and NMS deletes
one of them. **The assumption fails exactly where object density is highest**,
which is where detection is most valuable, and no amount of training fixes it
because NMS runs after the model.

{{sec:9-practical-example}} measures the shape of that failure and it is not the
shape people expect. The best threshold does not move much — what changes is how
much a wrong threshold *costs*, and that grows 4.4× from sparse scenes to crowded
ones. **A default validated on sparse images looks fine and is quietly
catastrophic on the dense ones.**

### The modern answer: make the matching part of the loss

{{cite:carion2020detr}} removes anchors and NMS together by changing the
*assignment*. The model emits a fixed number of predictions (say 100), and the
loss finds the **optimal one-to-one matching** between predictions and ground
truths, then penalises each prediction against its match — with unmatched
predictions trained to say "nothing".

One-to-one is the whole trick. If exactly one prediction may claim each object,
then **producing a duplicate is itself a training error**, so the model learns not
to. There is nothing left for NMS to remove, and no threshold to tune.

It is not free: the matching is a global assignment problem solved per image
during training, the model must learn to coordinate its predictions with each
other, and that coordination is slow to learn — original DETR needed far more
epochs than the detector it replaced.

### The metric shapes the model more than you would like

**mAP does not punish junk.** A prediction ranked below every real detection
cannot reduce precision at any recall already achieved, and if it happens to land
on a missed object it *adds* recall. {{sec:9-practical-example}} adds 500 noise
boxes per image and mAP goes **up**.

So a detector tuned for the leaderboard emits everything it can. What you show a
user is governed by a confidence threshold — and **the metric contains no
information about where to set it.** That gap between the reported number and the
deployed system is the single most common surprise for people shipping their first
detector.

## 5. Formal Explanation

### 5.1 IoU

$$ \text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|} \in [0, 1] $$ (eq:iou)

The universal currency of detection: it defines correctness, drives NMS, and
appears in the loss. Note what it is *not* — a distance. It saturates: two boxes
that do not overlap have IoU 0 whether they are adjacent or on opposite sides of
the image, so it gives no gradient toward a distant target. That is why box
regression is trained on coordinates and only *evaluated* with IoU.

### 5.2 Anchor assignment

For anchors $\{a_i\}$ and ground truths $\{g_j\}$, the standard rule is

$$ \text{match}(a_i) = \begin{cases} g_j & \text{if } \text{IoU}(a_i, g_j) \ge \tau_+ \\ \varnothing & \text{if } \max_j \text{IoU}(a_i, g_j) < \tau_- \\ \text{ignored} & \text{otherwise} \end{cases} $$ (eq:anchor-assignment)

**This is one-to-many**: many anchors may match one ground truth. That is the
design decision that makes duplicates inevitable and NMS necessary — the two are
the same choice seen at training and inference time.

It also creates a severe class imbalance: with tens of thousands of anchors and a
handful of objects, negatives outnumber positives by a thousand to one, which is
what {{cite:lin2017focal}}'s focal loss exists to address.

### 5.3 Non-maximum suppression

$$ \text{keep } b_i \iff \nexists\, b_j : s_j > s_i \wedge \text{IoU}(b_i, b_j) > t \wedge b_j \text{ kept} $$ (eq:nms)

A greedy, discrete, non-differentiable procedure with a hyperparameter, applied
after the model. Its implicit claim is

$$ \text{IoU}(b_i, b_j) > t \;\Longrightarrow\; b_i \text{ and } b_j \text{ describe the same object} $$ (eq:crowd-ambiguity)

and {{eq:crowd-ambiguity}} is **false whenever two distinct objects overlap by
more than $t$**. In {{sec:9-practical-example}}'s densest scenes neighbouring
objects have a true IoU of 0.36, so any $t \le 0.36$ is guaranteed to delete real
detections.

### 5.4 Average precision, and what it rewards

Rank all predictions by score, greedily match each to an unclaimed ground truth
above the IoU threshold, and accumulate:

$$ P(k) = \frac{\text{TP}(k)}{k}, \qquad R(k) = \frac{\text{TP}(k)}{|G|}, \qquad \text{AP} = \frac{1}{101}\sum_{r \in \{0, 0.01, \dots, 1\}} \max_{k : R(k) \ge r} P(k) $$ (eq:average-precision)

Two properties follow directly, and both matter more than their obscurity
suggests.

**Low-ranked predictions are nearly free.** A prediction at rank $k$ affects only
$P(k')$ for $k' \ge k$, and the interpolated maximum in
{{eq:average-precision}} takes the *best* precision at each recall level, so
appending predictions below every true positive cannot reduce any term already
achieved:

$$ \text{AP}(\text{preds} \cup \text{low-scored junk}) \;\gtrsim\; \text{AP}(\text{preds}) $$ (eq:junk-is-free)

**Duplicates are cheap too.** A duplicate ranks just below the box it duplicates,
so it damages precision only at high recall, where most of the area has already
been accumulated. {{sec:9-practical-example}} measures switching NMS off entirely
costing only **0.011** AP on sparse scenes — which means **mAP is not the reason
to run NMS.** The reason is that a user seeing three boxes on one object sees a
broken product.

### 5.5 Set prediction

Let the model emit a fixed $N$ predictions, padded with a "no object" class. Find

$$ \hat{\sigma} = \arg\min_{\sigma \in \mathfrak{S}_N} \sum_{i=1}^{N} \mathcal{L}_{\text{match}}\big(y_i,\, \hat{y}_{\sigma(i)}\big) $$ (eq:set-prediction)

by the Hungarian algorithm, then train against $\hat{\sigma}$. Because $\sigma$ is
a **permutation**, the assignment is one-to-one, and

$$ \text{one-to-one assignment} \;\Longrightarrow\; \text{a duplicate is a training error} \;\Longrightarrow\; \text{no NMS} $$ (eq:one-to-one-removes-nms)

{{eq:one-to-one-removes-nms}} is the chapter's structural point. **NMS was never
an inference detail; it was the inference-time consequence of a training-time
assignment choice.** Change the assignment and the inference step disappears.

The costs are real: $O(N^3)$ matching per image (small in practice), and — the
serious one — the model must learn *global coordination* between its predictions,
because whether prediction 7 should fire depends on whether prediction 12 already
claimed that object. That is a much harder credit-assignment problem than "does
this anchor overlap something", and it is why the original converged slowly.

### 5.6 The three families, in one equation each

$$ \text{two-stage: } \text{propose} \to \text{refine}, \qquad \text{one-stage: } \text{dense predict} \to \text{NMS}, \qquad \text{set: } \text{predict } N \to \text{match} $$ (eq:detector-families)

{{cite:ren2015fasterrcnn}} made the proposal stage learned and shared; the
two-stage family buys accuracy with a second pass. {{cite:redmon2016yolo}}
removed the proposal stage and made detection one forward pass, buying latency
with accuracy. {{cite:carion2020detr}} removed the hand-designed components
instead.

## 6. Mathematical Foundation

### 6.1 Why the NMS threshold cannot be right

Let neighbouring objects in a scene have true pairwise IoU distributed with mean
$\mu_d$, increasing in density $d$. NMS with threshold $t$ makes two errors:

$$ \Prob[\text{delete a real object}] = \Prob[\text{IoU}_{\text{neighbours}} > t] \nearrow d, \qquad \Prob[\text{keep a duplicate}] = \Prob[\text{IoU}_{\text{dup}} < t] \nearrow t $$ (eq:nms-two-errors)

Minimising the sum gives a $t^*(d)$ that depends on **density**, a property of the
scene. A single deployed threshold serves the whole distribution of densities, so
its excess loss is

$$ \mathbb{E}_d\big[L(t^*_{\text{global}}, d) - L(t^*(d), d)\big] > 0 $$ (eq:threshold-regret)

with the gap growing as the density distribution widens.
{{sec:9-practical-example}} measures the second-derivative version of this: at
$t = 0.3$ the penalty relative to the best available threshold is 0.064 when
sparse and **0.282 when crowded**.

> **MATH NOTE:** The measurement did not show $t^*$ *moving* — 0.5 was best in
> every row. {{eq:nms-two-errors}} predicts movement, and the reason it is not
> visible here is that the duplicate-IoU distribution and the neighbour-IoU
> distribution are well separated until the densest setting. The honest reading
> is that density sharpens the *curvature* of the loss around $t^*$ before it
> moves $t^*$ itself — which is the more dangerous failure, because a wrong
> threshold stays wrong in the same direction and simply costs more.

### 6.2 Why junk is free, worked

Take 8 ground truths, 7 correct detections ranked 1–7, and AP $\approx 0.875$.
Append 500 junk boxes at scores below all of them. The precision at every $k \le
7$ is unchanged, so every term $\max_{k: R(k) \ge r} P(k)$ for $r \le 7/8$ is
unchanged. The only new terms are at $r > 7/8$, where precision was previously
**zero** because that recall was never reached.

So the junk can only add. And if one of the 500 lands on the missed object with
IoU $\ge 0.5$ — with 500 tries, not unlikely — recall reaches 1 and a positive
term appears where there was none:

$$ \Delta\text{AP} \;\ge\; 0, \qquad \mathbb{E}[\Delta\text{AP}] > 0 \text{ when junk can hit} $$ (eq:junk-expected-gain)

{{sec:9-practical-example}} measures AP going **0.8542 → 0.8622** as boxes per
image go from 6.9 to 506.8.

### 6.3 The threshold the metric does not give you

Deployment needs a score cutoff $\theta$. The metric integrates over *all*
$\theta$, so it says nothing about any particular one. What determines $\theta$ is
an external requirement — "no more than one false alarm per hundred images", or
"recall at least 0.9" — which maps to a point on the precision–recall curve:

$$ \theta^* = \min\{\theta : P(\theta) \ge P_{\text{required}}\} \quad\text{or}\quad \max\{\theta : R(\theta) \ge R_{\text{required}}\} $$ (eq:operating-point)

**{{eq:operating-point}} is the number you ship, and mAP is the number you
report.** They are different numbers derived from different requirements, and
confusing them is why a detector with excellent mAP can be unusable out of the
box.

## 7. Internal Mechanics

```mermaid {#fig:detector-families caption="Three answers to the same set-prediction problem. The dashed box is the part that is not learned: anchor assignment at training time and NMS at inference are the same design decision seen twice (eq:one-to-one-removes-nms). The set-prediction path replaces both with a matching inside the loss, and pays for it in convergence speed."}
flowchart TB
    IM["image"] --> BB["backbone<br/>(ch:mm-classification)"]
    BB --> TWO["two-stage:<br/>propose, then refine"]
    BB --> ONE["one-stage:<br/>dense anchor grid"]
    BB --> SET["set prediction:<br/>N learned queries"]
    TWO --> NMS{"NMS<br/>threshold t"}
    ONE --> NMS
    NMS --> OUT["boxes"]
    SET --> HUN["Hungarian matching<br/>INSIDE the loss"]
    HUN --> OUT
    NMS -.->|"not learned;<br/>assumes overlap = duplicate<br/>(eq:crowd-ambiguity)"| NMS
    OUT --> TH{"confidence threshold<br/>-- NOT given by mAP"}
    TH --> SHIP["what the user sees"]
```

### 7.1 Where the backbone is tapped, and why it matters here

{{ch:mm-cv-fundamentals}}'s jump is the localisation resolution. At stage 4 the
jump is 32 pixels, so a detector predicting only from there cannot place a box
more precisely than that. Feature pyramids exist to fix this: predict small
objects from high-resolution early stages with small receptive fields, and large
objects from deep stages with large ones.

**Match the effective receptive field to the object size at each level**
({{eq:erf-worked}}), not the theoretical one. A pyramid level whose effective
field is 40 pixels is not going to classify a 120-pixel object well, however
generous its theoretical field.

### 7.2 What actually breaks in production detection

In rough order of frequency:

| Failure | Where it comes from |
|---|---|
| small objects missed | input resize, and {{eq:erf-worked}} at the prediction layer |
| crowded scenes | {{eq:crowd-ambiguity}} |
| too many boxes shown | {{eq:junk-is-free}} plus no {{eq:operating-point}} |
| duplicate boxes | NMS threshold too high, or class-agnostic NMS not applied |
| good mAP, unhappy users | the operating point was never chosen |

**Three of those five are not model problems.** They are consequences of
components and metrics that sit around the model, which is the argument for
knowing this chapter even when you are calling someone else's detector.

### 7.3 Class-wise versus class-agnostic NMS

A detail that causes real bugs. Run NMS per class and two different objects that
overlap survive — correct. But two *predictions of different classes* for the
**same** object also survive, and the user sees a box labelled "dog" inside a box
labelled "cat". Run it class-agnostically and that is fixed, at the cost of
deleting genuinely overlapping objects of different classes, which is common
(a person holding a phone).

There is no setting that is right in both cases, which is
{{eq:crowd-ambiguity}} appearing a second time.

## 8. Implementation

```python {tier=A name=nms-and-crowds}
"""Non-maximum suppression: a hand-designed piece of the model, run at inference.

A detector trained with anchors produces many boxes per object, because many
anchors overlap the same thing and all of them are trained to fire. NMS removes
the duplicates by deleting any box that overlaps a higher-scoring one by more
than a threshold (eq:nms).

That step is not learned, has a hyperparameter nobody tunes per deployment, and
makes an assumption that is false in exactly the scenes people care about:
that two boxes overlapping a lot are two detections of ONE object rather than
detections of two objects that happen to be close (eq:crowd-ambiguity).

This listing sweeps the threshold against crowd density and measures average
precision, with IoU, NMS and AP all implemented from scratch so what the metric
rewards is visible.
"""
import numpy as np

rng = np.random.default_rng(19)

IMG, SIZE = 120.0, 24.0        # canvas, object side length
N_SCENE = 80
DETS_PER_OBJ = 4               # anchors that fire on one object
N_FALSE = 12                   # false positives per scene


def iou(a, b):
    """a: (n,4), b: (m,4), boxes as x1,y1,x2,y2. Returns (n,m). eq:iou."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ar_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (ar_a[:, None] + ar_b[None, :] - inter + 1e-9)


def nms(boxes, scores, thresh):
    """eq:nms -- keep the highest scorer, delete everything overlapping it."""
    order = np.argsort(-scores)
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        ious = iou(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= thresh]
    return np.array(keep, dtype=int)


def scene(spacing):
    """Objects on a jittered grid. Small spacing means a crowded scene."""
    per_side = max(int(IMG // spacing), 1)
    gt = []
    for i in range(per_side):
        for j in range(per_side):
            cx = (i + 0.5) * spacing + rng.normal(scale=spacing * 0.06)
            cy = (j + 0.5) * spacing + rng.normal(scale=spacing * 0.06)
            gt.append([cx - SIZE / 2, cy - SIZE / 2, cx + SIZE / 2, cy + SIZE / 2])
    gt = np.array(gt)

    boxes, scores = [], []
    for g in gt:
        for d in range(DETS_PER_OBJ):
            jit = rng.normal(scale=SIZE * 0.16, size=4)
            boxes.append(g + jit)
            # The best-aligned proposal usually scores highest, but not always.
            scores.append(0.95 - 0.10 * d + rng.normal(scale=0.06))
    for _ in range(N_FALSE):
        cx, cy = rng.uniform(0, IMG, size=2)
        boxes.append([cx - SIZE / 2, cy - SIZE / 2, cx + SIZE / 2, cy + SIZE / 2])
        scores.append(rng.uniform(0.30, 0.80))
    return gt, np.array(boxes), np.array(scores)


def average_precision(gt, boxes, scores, iou_thresh=0.5):
    """eq:average-precision with greedy highest-score-first matching, each
    ground truth claimed once -- the COCO convention, written out so the
    behaviour is visible."""
    if len(boxes) == 0:
        return 0.0
    order = np.argsort(-scores)
    boxes, scores = boxes[order], scores[order]
    ious = iou(boxes, gt)
    claimed = np.zeros(len(gt), dtype=bool)
    tp = np.zeros(len(boxes))
    for i in range(len(boxes)):
        j = int(np.argmax(np.where(claimed, -1.0, ious[i])))
        if ious[i, j] >= iou_thresh and not claimed[j]:
            tp[i], claimed[j] = 1.0, True
    ctp = np.cumsum(tp)
    prec = ctp / np.arange(1, len(tp) + 1)
    rec = ctp / len(gt)
    # 101-point interpolated AP, as COCO computes it.
    ap = 0.0
    for r in np.linspace(0, 1, 101):
        p = prec[rec >= r].max() if (rec >= r).any() else 0.0
        ap += p / 101
    return float(ap)


SPACINGS = (60.0, 30.0, 20.0, 15.0, 12.0)
THRESHOLDS = (0.3, 0.5, 0.7, 0.9, 1.0)

print(f"objects are {SIZE:.0f} px wide on a {IMG:.0f} px canvas; "
      f"{DETS_PER_OBJ} proposals per object\n")
print(f"{'spacing':>9}{'neighbour IoU':>15}{'objects':>9}   "
      + "".join(f"{'t=' + str(t):>9}" for t in THRESHOLDS) + f"{'best t':>9}")
print("-" * 92)

table = {}
for sp in SPACINGS:
    aps = {t: [] for t in THRESHOLDS}
    nbr, nobj = [], 0
    for _ in range(N_SCENE):
        gt, boxes, scores = scene(sp)
        nobj = len(gt)
        m = iou(gt, gt)
        np.fill_diagonal(m, 0)
        nbr.append(float(m.max(axis=1).mean()))
        for t in THRESHOLDS:
            k = nms(boxes, scores, t)
            aps[t].append(average_precision(gt, boxes[k], scores[k]))
    row = {t: float(np.mean(v)) for t, v in aps.items()}
    best = max(row, key=row.get)
    table[sp] = (row, best, float(np.mean(nbr)))
    print(f"{sp:>9.0f}{np.mean(nbr):>15.3f}{nobj:>9}   "
          + "".join(f"{row[t]:>9.3f}" for t in THRESHOLDS) + f"{best:>9.1f}")

sparse, dense = SPACINGS[0], SPACINGS[-1]
gap_sparse = table[sparse][0][0.5] - table[sparse][0][0.3]
gap_dense = table[dense][0][0.5] - table[dense][0][0.3]
print(f"""
Start with the t=1.0 column, which is NMS switched off. In the sparse scene it
scores {table[sparse][0][1.0]:.3f} against the best setting's
{table[sparse][0][0.5]:.3f} -- a gap of only
{table[sparse][0][0.5] - table[sparse][0][1.0]:.3f}, even though every object is
contributing {DETS_PER_OBJ} boxes and three of them are duplicates.

That is worth pausing on, because it is not what the metric is supposed to do.
Average precision barely punishes duplicates: a duplicate scores just below the
box it duplicates, so it damages precision only at the high-recall end of the
curve, where the interpolation has already collected most of the area. So mAP is
not the reason to run NMS. The reason is that a user looking at three boxes
around one object sees a broken detector, which is a fact about the product and
not about the number.

Now the sensitivity, which is the real finding. Compare t=0.3 against the best
setting in each row. In the sparse scene it costs {gap_sparse:.3f}. In the
crowded scene it costs {gap_dense:.3f} -- {gap_dense / gap_sparse:.1f} times as
much, from the same hyperparameter error.

The mechanism is visible in the neighbour-IoU column. At spacing {dense:.0f} two
adjacent objects genuinely overlap by {table[dense][2]:.2f}, so a threshold of
0.3 cannot tell "second detection of one object" from "detection of a second
object" -- and it deletes the second object. NMS assumes high overlap means
duplication, and in a crowd that assumption is simply false.

Note what did NOT happen: the best threshold stayed at 0.5 in every row. Density
does not move the optimum here so much as sharpen the penalty for missing it,
which is the more dangerous shape. A default chosen on sparse validation images
looks entirely fine, and then costs {gap_dense:.3f} of AP on the crowded images
-- the ones with the most objects in them, and usually the ones that matter.

And none of it is trainable. The threshold is applied after the model, so no
amount of training adapts it, and it cannot vary per image because nothing
measures density at inference. That is the argument cite:carion2020detr acts on:
NMS is a piece of the model that was moved out of the model -- a discrete,
non-differentiable, hand-tuned decision about which predictions are duplicates.
Predict a SET directly, with a loss that already forbids two predictions claiming
one object (eq:set-prediction), and the step has nothing left to do.""")
```

The first listing measured what NMS costs. The second measures what the metric
that judged it actually rewards.

```python {tier=A name=what-map-rewards}
"""What mAP rewards, and why a detector tuned for it is not deployable as-is.

Detection is reported in mAP, and mAP has two properties that shape every
detector built to maximise it. Both are consequences of eq:average-precision
rather than accidents, and both are invisible if you only ever read the number.

First: adding low-scoring predictions can only help (eq:junk-is-free). A
prediction ranked below everything else cannot lower precision at any recall
level already achieved, and it might add recall. So the metric actively rewards
emitting junk.

Second: the IoU threshold is not a detail. mAP@0.5 asks "did you find it" and
mAP@[.5:.95] asks "did you find it and draw the box accurately", and those are
different questions that can rank two detectors in opposite orders.

This listing measures both on the same detections.
"""
import numpy as np

rng = np.random.default_rng(23)

IMG, SIZE = 120.0, 24.0
N_SCENE, N_OBJ = 200, 8


def iou(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ar_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (ar_a[:, None] + ar_b[None, :] - inter + 1e-9)


def average_precision(gt, boxes, scores, thr):
    if len(boxes) == 0:
        return 0.0
    order = np.argsort(-scores)
    boxes = boxes[order]
    ious = iou(boxes, gt)
    claimed = np.zeros(len(gt), dtype=bool)
    tp = np.zeros(len(boxes))
    for i in range(len(boxes)):
        j = int(np.argmax(np.where(claimed, -1.0, ious[i])))
        if ious[i, j] >= thr and not claimed[j]:
            tp[i], claimed[j] = 1.0, True
    ctp = np.cumsum(tp)
    prec, rec = ctp / np.arange(1, len(tp) + 1), ctp / len(gt)
    return float(sum((prec[rec >= r].max() if (rec >= r).any() else 0.0)
                     for r in np.linspace(0, 1, 101)) / 101)


def coco_map(gt, boxes, scores):
    """mAP@[.5:.95] -- the COCO primary metric, averaged over ten thresholds."""
    return float(np.mean([average_precision(gt, boxes, scores, t)
                          for t in np.arange(0.5, 1.0, 0.05)]))


def make_scene(loc_noise, miss_rate):
    """A detector with two independent weaknesses: how precisely it localises
    (loc_noise) and how often it misses an object entirely (miss_rate)."""
    gt = []
    for _ in range(N_OBJ):
        cx, cy = rng.uniform(SIZE, IMG - SIZE, size=2)
        gt.append([cx - SIZE / 2, cy - SIZE / 2, cx + SIZE / 2, cy + SIZE / 2])
    gt = np.array(gt)
    boxes, scores = [], []
    for g in gt:
        if rng.random() < miss_rate:
            continue
        boxes.append(g + rng.normal(scale=SIZE * loc_noise, size=4))
        scores.append(rng.uniform(0.55, 0.99))
    return gt, np.array(boxes).reshape(-1, 4), np.array(scores)


def junk(n):
    """Confident-looking nonsense, scored below everything real."""
    b, s = [], []
    for _ in range(n):
        cx, cy = rng.uniform(0, IMG, size=2)
        b.append([cx - SIZE / 2, cy - SIZE / 2, cx + SIZE / 2, cy + SIZE / 2])
        s.append(rng.uniform(0.001, 0.05))
    return np.array(b).reshape(-1, 4), np.array(s)


print("PART 1 -- what happens when you append low-scoring junk predictions\n")
print(f"{'junk boxes added':>18}{'mAP@0.5':>12}{'mAP@[.5:.95]':>16}"
      f"{'boxes per image':>18}")
print("-" * 64)
base = None
for n_junk in (0, 5, 20, 100, 500):
    a5, a95, nb = [], [], []
    for _ in range(N_SCENE):
        gt, boxes, scores = make_scene(0.10, 0.15)
        if n_junk:
            jb, js = junk(n_junk)
            boxes, scores = np.vstack([boxes, jb]), np.concatenate([scores, js])
        a5.append(average_precision(gt, boxes, scores, 0.5))
        a95.append(coco_map(gt, boxes, scores))
        nb.append(len(boxes))
    if base is None:
        base = (float(np.mean(a5)), float(np.mean(a95)))
    print(f"{n_junk:>18}{np.mean(a5):>12.4f}{np.mean(a95):>16.4f}"
          f"{np.mean(nb):>18.1f}")

print("\n\nPART 2 -- two detectors, and which one is better depends on the metric\n")
print(f"{'detector':<34}{'mAP@0.5':>12}{'mAP@[.5:.95]':>16}")
print("-" * 62)
DETECTORS = {
    "A: finds everything, sloppy boxes": (0.11, 0.02),
    "B: precise boxes, misses more":     (0.035, 0.28),
}
res = {}
for name, (loc, miss) in DETECTORS.items():
    a5, a95 = [], []
    for _ in range(N_SCENE):
        gt, boxes, scores = make_scene(loc, miss)
        a5.append(average_precision(gt, boxes, scores, 0.5))
        a95.append(coco_map(gt, boxes, scores))
    res[name] = (float(np.mean(a5)), float(np.mean(a95)))
    print(f"{name:<34}{res[name][0]:>12.4f}{res[name][1]:>16.4f}")

A, B = list(DETECTORS)
print(f"""
Part 1 is the property that shapes every detector's output. Appending 500 junk
boxes per image -- scored near zero, overlapping nothing, pure noise -- moves
mAP@0.5 from {base[0]:.4f} to the last row's value, and the movement is upward or
negligible. It is never a real penalty.

The reason is in eq:average-precision. Precision is only ever evaluated at recall
levels the ranking has already reached, and a box ranked below every real
detection cannot displace anything above it. If one of those junk boxes happens
to land on a missed object, recall goes UP. The metric offers a free lottery
ticket with no cost per ticket.

So a detector tuned to maximise mAP emits hundreds of boxes per image, and the
right number to show a user is not in the metric anywhere. That confidence
threshold has to come from somewhere else -- a precision/recall requirement
someone states in product terms (eq:operating-point) -- and teams that report mAP
and ship the raw output are shipping the lottery tickets.

Part 2 is the other property, and it is why COCO averages over ten IoU
thresholds instead of using one. Detector {A[0]} finds nearly every object and
draws loose boxes. Detector {B[0]} draws tight boxes and misses more of them. At
mAP@0.5 -- which asks only "is there a box roughly here" -- A scores
{res[A][0]:.4f} against B's {res[B][0]:.4f}. Under the stricter average, which
keeps raising the bar for how well the box must fit, the ordering
{'REVERSES' if (res[A][0] > res[B][0]) != (res[A][1] > res[B][1]) else 'holds but narrows sharply'}:
{res[A][1]:.4f} against {res[B][1]:.4f}.

Neither detector is better. They are better at different things, and the metric
chooses which. That is a decision about the application rather than about the
model: a counting system wants A, a robot that has to grasp the object wants B,
and quoting a single mAP hides which question was asked.""")
```

## 9. Practical Example

**What NMS costs, and where.** Switching NMS off entirely costs only **0.011** AP
on sparse scenes — 0.832 against 0.843 — despite every object contributing four
boxes. {{eq:junk-is-free}}'s cousin: duplicates rank just below the box they
duplicate and damage precision only at high recall.

**So mAP is not the reason to run NMS.** The reason is that a user seeing three
boxes on one object sees a broken detector, which is a product fact rather than a
metric fact. That is worth knowing before optimising the threshold against a
leaderboard.

**The real finding is the sensitivity.** Choosing $t = 0.3$ instead of the best
available costs **0.064** AP on well-separated objects and **0.282** on crowded
ones — the same hyperparameter error, **4.4× more expensive**.

The mechanism is in the neighbour-IoU column: at the densest spacing, adjacent
objects genuinely overlap by **0.36**, so any threshold below that is *guaranteed*
to delete real detections. {{eq:crowd-ambiguity}} is not approximately false
there; it is definitionally false.

> **IMPORTANT:** The best threshold stayed at 0.5 in every row, so
> {{eq:nms-two-errors}}'s predicted movement of $t^*$ did not appear at these
> densities. **What density changed was the curvature, not the location** — and
> that is the more dangerous shape, because a default validated on sparse images
> looks entirely correct and then costs 0.282 AP on exactly the images with the
> most objects in them. A moving optimum would at least be visible in validation.

And none of it is trainable: the threshold runs after the model, and nothing
measures density at inference. That is the argument
{{cite:carion2020detr}} acts on, and {{eq:one-to-one-removes-nms}} is the
resolution — change the *assignment* and the inference step has nothing to do.

**What the metric rewards.** Appending **500 junk boxes per image** moves mAP@0.5
from 0.8542 to **0.8622** — upward — while boxes per image go from 6.9 to
**506.8**. {{eq:junk-expected-gain}} exactly: junk below every true positive
cannot displace anything, and with 500 tries some of it lands on a missed object.

**A detector tuned for mAP therefore emits everything it can**, and the
confidence threshold you actually ship is nowhere in the metric.
{{eq:operating-point}} has to come from a product requirement.

**And the IoU threshold chooses the winner.** Detector A finds nearly everything
with loose boxes; detector B draws tight boxes and misses more. At mAP@0.5, A wins
**0.9557 to 0.7209**. Under mAP@[.5:.95] the ordering **reverses**: **0.4180
against 0.5948**.

Neither is better. **They are better at different things, and the metric picks
which** — a counting system wants A, a robot that must grasp the object wants B,
and a single reported mAP conceals which question was asked.

## 10. Production Considerations

**Choose the operating point from a stated requirement**
({{eq:operating-point}}), not from the metric. Write down the acceptable false
alarm rate before looking at any curve.

**Tune the NMS threshold on your own density distribution**, and check the
crowded tail separately. The sparse average will not reveal
{{eq:threshold-regret}}.

**Report mAP@0.5 and mAP@[.5:.95] separately.** They can rank systems in opposite
orders, as measured.

**Evaluate by object size.** COCO's small/medium/large split exists because
{{eq:erf-worked}} makes them different problems.

**Decide class-wise versus class-agnostic NMS deliberately**, and know which
failure you chose.

**Log boxes-per-image after thresholding.** A sudden rise is the earliest signal
of a score-calibration drift.

**Do not resize away small objects.** The most common detection failure is
upstream of the detector.

**Prefer a set-prediction detector for crowded domains** unless latency forbids
it — the crowd failure is structural in the anchor-based family.

## 11. Common Mistakes

**Shipping the mAP-optimal output.** Hundreds of boxes per image.

**Treating the NMS threshold as a constant.** It is a scene-density parameter
being held constant.

**Reporting one mAP number** and hiding the localisation/recall trade.

**Comparing detectors at different input resolutions.**

**Using IoU as a regression loss without care** — {{eq:iou}} is flat at zero
overlap and provides no gradient toward a distant box.

**Ignoring the anchor/ground-truth size mismatch.** If your objects are all
smaller than your smallest anchor, {{eq:anchor-assignment}} assigns nothing and
the model trains on pure background.

**Assuming DETR-style models are drop-in.** They converge slowly and want
different schedules.

## 12. Failure Modes

**Crowd collapse.** Symptom: recall falls sharply as objects per image rises.
Cause: {{eq:crowd-ambiguity}}. Diagnose by plotting recall against scene density.

**Duplicate boxes in production.** Symptom: users report double-counting. Cause:
threshold too high, or per-class NMS with overlapping classes.

**Box flood.** Symptom: hundreds of low-confidence boxes. Cause:
{{eq:junk-is-free}} and no operating point.

**Small-object blindness.** Symptom: recall collapses below some pixel area.
Cause: resize, pyramid level, or {{eq:erf-worked}}.

**Score drift.** Symptom: a fixed confidence threshold silently changes behaviour
after a retrain, because scores are not calibrated across model versions.

**Anchor mismatch.** Symptom: training loss looks fine and detection recall is
near zero. Cause: {{eq:anchor-assignment}} never firing.

**Good mAP, bad product.** The most common outcome of all, and it is
{{eq:operating-point}} never having been chosen.

## 13. Alternatives

| Family | Trades away | When it wins |
|---|---|---|
| two-stage ({{cite:ren2015fasterrcnn}}) | latency | accuracy-first, offline batch |
| one-stage ({{cite:redmon2016yolo}}) | some accuracy | real time, embedded, video |
| set prediction ({{cite:carion2020detr}}) | convergence speed | crowded scenes; no threshold to tune |
| detection via segmentation ({{cite:he2017maskrcnn}}) | compute | when you need masks anyway |
| open-vocabulary detection ({{ch:mm-clip}}) | closed-set accuracy | classes not known at training time |
| a VLM asked to point | precision, latency, throughput | few images, novel classes, no training budget |

**The last row is now a real option** and deserves stating plainly: for a
low-volume task with no labelled data, prompting a vision-language model
({{ch:mm-vlms}}) beats training a detector, and it stops being competitive as
soon as throughput or localisation precision matters.

## 14. Evaluation

**Report mAP@0.5 and mAP@[.5:.95] separately**, plus per-size breakdowns.

**Report the operating point** — precision and recall at the shipped threshold —
alongside mAP. That pair is what the system actually does.

**Stratify by scene density.** The aggregate hides the crowd failure entirely.

**Count boxes per image** before and after thresholding.

**Evaluate localisation and classification separately.** A box in the right place
with the wrong label and a right label in the wrong place are different bugs, and
mAP merges them.

**Use a fixed evaluation resolution**, and state it.

## 15. Advanced Concepts

**Soft-NMS and learned NMS.** {{maturity:MATURE}} Decay a neighbour's score
rather than deleting it, so a genuine second object survives with a reduced score.
It softens {{eq:crowd-ambiguity}} without resolving it, and adds a second
hyperparameter.

**One-to-one assignment as the real contribution.**
{{maturity:MATURE}} {{eq:one-to-one-removes-nms}} is what removes NMS, not the
transformer. Later work applies one-to-one matching to convolutional detectors
and removes NMS there too — which is the clean statement of what
{{cite:carion2020detr}} actually established.

**IoU-based regression losses.** {{maturity:MATURE}} GIoU and its successors
extend {{eq:iou}} with a term that remains informative at zero overlap, fixing the
flat-gradient problem directly rather than routing around it.

**The metric as a design constraint.** {{maturity:ESTABLISHED}}
{{eq:junk-is-free}} means the community optimised for a metric that rewards
verbosity, and every deployed detector then needs a threshold the metric never
specified. **A metric that cannot be shipped shapes models that cannot be
shipped**, and this is worth remembering well beyond detection —
{{ch:ev-why-hard}} generalises it.

**Detection as the wrong abstraction.** {{maturity:EMERGING}} A box is a poor
description of a non-rectangular object, and for many downstream uses
segmentation ({{ch:mm-segmentation}}) or keypoints are the better output. The box
persists partly because {{cite:lin2014coco}} made it the measured thing.

## 16. Connection to Previous Chapters

{{ch:mm-cv-fundamentals}}'s jump and {{eq:erf-worked}} decide what a detector can
localise and at what size, which is why feature pyramids exist.
{{ch:mm-classification}} supplies the backbone and
{{eq:pooling-invariance}} is precisely what a detector must *not* do — the
position it discards is half the answer. {{ch:ml-metrics}}'s precision and recall
are composed by {{eq:average-precision}} in a way that has consequences neither
component has alone. Forward: {{ch:mm-segmentation}} replaces the box with a mask
and inherits the assignment problem; {{ch:mm-vit}} supplies the architecture
{{cite:carion2020detr}} needed; and {{ch:mm-clip}} makes the class set open,
which changes {{eq:anchor-assignment}}'s premise entirely.

## 17. Exercises

1. Compute {{eq:iou}} for two 24-pixel boxes offset by 12 pixels along one axis.
   Verify against the listing's neighbour-IoU column.
2. Prove {{eq:junk-is-free}} from {{eq:average-precision}}, and state the
   condition under which appending a prediction *can* reduce AP.
3. In `nms-and-crowds`, add a spacing of 8. Does the best threshold finally move,
   and does that match {{eq:nms-two-errors}}?
4. Modify the same listing to implement soft-NMS. Does it reduce the crowded-scene
   penalty, and what does it cost in the sparse rows?
5. In `what-map-rewards`, make the junk boxes score *above* the real detections.
   What happens, and why is that the condition in exercise 2?
6. Implement Hungarian matching for {{eq:set-prediction}} on a small example and
   verify it produces no duplicates.
7. Using {{eq:operating-point}}, choose a threshold for "at most one false
   positive per ten images" from a precision–recall curve you compute.
8. Take a detector you use. Plot recall against object pixel area and against
   scene density, and say which of {{sec:12-failure-modes}}'s modes you have.

## 18. Interview Questions

1. Why is detection structurally harder than classification?
2. What is NMS and what assumption does it make?
3. When does that assumption fail, and what does it cost?
4. Why does adding low-confidence boxes not hurt mAP?
5. What threshold do you ship, and where does it come from?
6. Explain the difference between mAP@0.5 and mAP@[.5:.95] and when they disagree.
7. What does bipartite matching replace, and what does it cost?
8. Why is IoU a poor regression loss?
9. Your detector's recall collapses on busy images. Diagnose.
10. When would you use a VLM instead of a detector?

## 19. Research Questions

1. {{eq:threshold-regret}} says a global threshold is suboptimal. Can density be
   estimated cheaply at inference and the threshold adapted per image?
2. {{eq:nms-two-errors}} predicts $t^*$ moves with density and the measurement
   shows curvature changing first. At what density does the optimum actually move,
   and does that depend on the score distribution rather than the geometry?
3. One-to-one assignment removes NMS and slows convergence. What is the minimal
   change to the matching that recovers convergence speed without reintroducing
   duplicates?
4. {{eq:junk-is-free}} shaped a decade of detectors. What metric would reward a
   *deployable* output set, and would optimising it change architectures?
5. Boxes persist because they are what is measured. What would a benchmark
   measuring a downstream task, rather than boxes, select for?

## 20. Chapter Summary

**Detection is set prediction**, and the three difficulties that creates —
variable length, no canonical order, and matching — generate everything unusual
about the field. Anchors fix the length, {{eq:anchor-assignment}}'s one-to-many
rule fixes the matching, and NMS cleans up the duplicates the rule guarantees.

**NMS is a hand-designed component running outside the model**, and its claim —
{{eq:crowd-ambiguity}}, that high overlap means duplication — is false whenever
two real objects overlap. Measured: at the densest spacing, neighbouring objects
genuinely overlap by **0.36**, so any threshold below that deletes real
detections by construction.

**The failure's shape is worse than expected.** The best threshold did not move
across densities; the *penalty for missing it* grew — 0.064 AP on sparse scenes
and **0.282 on crowded ones**, a 4.4× amplification. A default validated on
sparse images therefore looks correct and is quietly catastrophic on the images
with the most objects, which are usually the ones that matter.

**And the metric will not warn you**, because switching NMS off entirely costs
only **0.011** AP. mAP is not the reason to run NMS; the user is.

**mAP has two properties that shaped the field.** {{eq:junk-is-free}}: appending
**500 junk boxes per image** raised mAP@0.5 from 0.8542 to **0.8622** while boxes
per image went from 6.9 to 506.8 — the metric rewards verbosity, so detectors are
verbose, and the confidence threshold you ship
({{eq:operating-point}}) is nowhere in the reported number. And the IoU threshold
selects the winner: two detectors ranked **0.9557 to 0.7209** at mAP@0.5 reverse
to **0.4180 against 0.5948** under mAP@[.5:.95].

**Bipartite matching removes NMS by changing the assignment, not the
architecture.** {{eq:one-to-one-removes-nms}}: if only one prediction may claim
each object, a duplicate is a training error and there is nothing left to
suppress. The cost is convergence — the model must learn to coordinate its
predictions with one another.

The general lesson is worth more than the techniques: **a hand-designed
inference step is usually a training-time assignment decision in disguise**, and
a metric that cannot be shipped shapes models that cannot be shipped.

## 21. Further Reading

{{cite:ren2015fasterrcnn}} for the two-stage architecture, and read it for how
completely the proposal stage was absorbed into the network.
{{cite:redmon2016yolo}} for the opposite trade, and for the clearest statement of
detection as a single regression.
{{cite:carion2020detr}} for set prediction — Section 3 for the matching loss, and
note that the contribution is the assignment rather than the transformer.
{{cite:he2017maskrcnn}} for RoIAlign, which is a lesson in how much a rounding
operation can cost.
{{cite:lin2017focal}} for the class imbalance that {{eq:anchor-assignment}}
creates.
{{cite:lin2014coco}} for the benchmark whose conventions — the IoU sweep, the
size splits, the 101-point interpolation — are baked into every number in this
chapter.

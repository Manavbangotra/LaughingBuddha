# -*- coding: utf-8 -*-
# Extracted from: Chapter 120 — Object Detection: Faster R-CNN, YOLO, and DETR
# Source: src/.../ch120-detection.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

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

# -*- coding: utf-8 -*-
# Extracted from: Chapter 120 — Object Detection: Faster R-CNN, YOLO, and DETR
# Source: src/.../ch120-detection.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

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

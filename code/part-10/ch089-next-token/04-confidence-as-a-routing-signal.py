# -*- coding: utf-8 -*-
# Extracted from: Chapter 89 — Next-Token Prediction and Cross-Entropy Loss
# Source: src/.../ch089-next-token.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Can confidence decide when to escalate? Only if it is calibrated."""
import numpy as np

rng = np.random.default_rng(4)
N, K = 5000, 20

# Ground truth difficulty: some questions the small model can answer, some not.
difficulty = rng.random(N)
small_correct = rng.random(N) > difficulty          # harder -> less likely right
LARGE_ACC = 0.93
large_correct = rng.random(N) < LARGE_ACC

COST_SMALL, COST_LARGE = 1.0, 12.0


def confidence(sharpening):
    """The small model's reported confidence, at a given miscalibration."""
    # True confidence tracks difficulty; sharpening distorts the reported value.
    true_conf = np.clip(1 - difficulty + rng.normal(0, 0.08, N), 0.02, 0.98)
    logit = np.log(true_conf / (1 - true_conf))
    return 1 / (1 + np.exp(-logit * sharpening))


def evaluate(conf, threshold):
    escalate = conf < threshold
    correct = np.where(escalate, large_correct, small_correct)
    cost = np.where(escalate, COST_SMALL + COST_LARGE, COST_SMALL)
    return float(correct.mean()), float(cost.mean()), float(escalate.mean())


print(f"small model alone : accuracy {small_correct.mean():.3f}, "
      f"cost {COST_SMALL:.1f}")
print(f"large model alone : accuracy {large_correct.mean():.3f}, "
      f"cost {COST_LARGE:.1f}\n")

for label, sharp in [("calibrated", 1.0), ("overconfident (aligned)", 2.6)]:
    conf = confidence(sharp)
    print(f"--- {label} confidence ---")
    print(f"{'threshold':>10} {'escalated':>11} {'accuracy':>10} {'cost':>8} "
          f"{'acc per cost':>13}")
    best = None
    for thr in (0.0, 0.3, 0.5, 0.7, 0.85, 1.0):
        acc, cost, esc = evaluate(conf, thr)
        eff = acc / cost
        if best is None or eff > best[1]:
            best = (thr, eff, acc, cost, esc)
        print(f"{thr:>10.2f} {esc:>11.1%} {acc:>10.3f} {cost:>8.2f} "
              f"{eff:>13.4f}")
    print(f"  best efficiency at threshold {best[0]:.2f}: "
          f"accuracy {best[2]:.3f} at cost {best[3]:.2f}\n")

# The measurement that decides whether the signal is usable at all.
for label, sharp in [("calibrated", 1.0), ("overconfident", 2.6)]:
    conf = confidence(sharp)
    # Does confidence actually separate correct from incorrect?
    auc = float(np.mean([
        (conf[i] > conf[j]) for i in rng.choice(np.flatnonzero(small_correct), 2000)
        for j in rng.choice(np.flatnonzero(~small_correct), 1)]))
    print(f"{label:<16} mean conf when right {conf[small_correct].mean():.3f}, "
          f"when wrong {conf[~small_correct].mean():.3f}, "
          f"separation AUC {auc:.3f}")

print("""
The separation AUC is the number that matters and it barely moves: sharpening a
distribution is monotone, so it preserves the ORDERING of confidences and
therefore the ranking quality of the signal.

What miscalibration destroys is the meaning of the THRESHOLD. On the calibrated
model, 0.7 means roughly 70% and a threshold can be chosen from a target error
rate. On the overconfident model, 0.7 means something else entirely, and the
threshold has to be found empirically and refitted whenever the model changes.

So confidence remains usable for routing after alignment — as a rank, not as a
probability. Anything that needs the number to mean what it says (abstention at
a stated error rate, expected-value calculations, cost-sensitive decisions)
needs calibration first.""")

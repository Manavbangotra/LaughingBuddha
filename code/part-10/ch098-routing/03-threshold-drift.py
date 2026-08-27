# -*- coding: utf-8 -*-
# Extracted from: Chapter 98 — Model Routing and Model Selection
# Source: src/.../ch098-routing.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A threshold is a quantile of a distribution that moves. Costs move with it."""
import numpy as np

rng = np.random.default_rng(4)
N = 20_000
SMALL_COST, LARGE_COST = 1.0, 12.0

# Month 1: confidence distribution under the current checkpoint.
conf_v1 = rng.normal(0.0, 1.0, N)
TARGET_ESCALATION = 0.30
tau = float(np.quantile(conf_v1, TARGET_ESCALATION))
print(f"threshold set for {TARGET_ESCALATION:.0%} escalation: tau = {tau:.3f}\n")

# Month 2: the provider updates the model. ch:llm-next-token — alignment shifts
# the confidence distribution, and the threshold is a fixed number.
SHIFTS = {
    "no change":                    (0.00, 1.00),
    "slightly more confident":      (0.30, 1.00),
    "much more confident":          (0.80, 1.00),
    "less confident":              (-0.40, 1.00),
    "more spread (less certain)":   (0.00, 1.40),
}

print(f"{'checkpoint':<30} {'escalation':>12} {'cost/request':>14} "
      f"{'vs planned':>12}")
planned_cost = SMALL_COST + TARGET_ESCALATION * LARGE_COST
for name, (shift, scale) in SHIFTS.items():
    conf = rng.normal(shift, scale, N)
    e = float((conf < tau).mean())
    cost = SMALL_COST + e * LARGE_COST
    print(f"{name:<30} {e:>12.1%} {cost:>14.2f} {cost / planned_cost:>11.0%}")

print(f"""
The threshold is a fixed number; the distribution it was fitted to is not. A
model that becomes more confident escalates LESS and quietly loses quality; one
that becomes less confident escalates more and quietly costs more. Neither
change is visible in any code diff.

'Much more confident' here escalates almost nothing — the cascade silently
degenerates into always-small, which is the dangerous direction because the cost
metric IMPROVES while quality falls.

The fix is mechanical: store the threshold as a target escalation RATE and
re-derive tau from a recent traffic sample, rather than storing tau. That makes
the invariant the thing you care about — how much traffic escalates — instead of
an implementation detail of a checkpoint that will change.""")

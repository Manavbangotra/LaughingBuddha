# -*- coding: utf-8 -*-
# Extracted from: Chapter 83 — Emergent Capabilities and What Emergence Means
# Source: src/.../ch083-emergence.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Forecasting an all-or-nothing product requirement from continuous measurements."""
import numpy as np

FIELDS = 7                       # every one must be correct
SIZES = np.array([1e9, 7e9, 70e9])
NEXT_SIZE = 400e9

# What the team measured: per-field correctness, which is continuous and
# therefore forecastable — this is the metric to instrument.
per_field = np.array([0.72, 0.89, 0.968])
observed_valid = per_field ** FIELDS

print(f"{'params':>9} {'per-field':>11} {'valid JSON':>12} "
      f"{'usable?':>9}")
for s, pf, v in zip(SIZES, per_field, observed_valid):
    print(f"{s:>9.0e} {pf:>11.3f} {v:>12.3f} {str(v > 0.95):>9}")

# Forecast the CONTINUOUS quantity, then compose the metric. Fit in logit
# space, where the smooth improvement is close to linear in log scale.
def logit(x):
    return np.log(x / (1 - x))


slope, intercept = np.polyfit(np.log10(SIZES), logit(per_field), 1)
pred_logit = slope * np.log10(NEXT_SIZE) + intercept
pred_field = 1 / (1 + np.exp(-pred_logit))
pred_valid = pred_field ** FIELDS

print(f"\nforecast at {NEXT_SIZE:.0e} parameters:")
print(f"  per-field correctness : {pred_field:.4f}  (extrapolated)")
print(f"  valid-JSON rate       : {pred_valid:.4f}  (composed)")

# The naive alternative: extrapolate the discontinuous metric directly.
naive_slope, naive_int = np.polyfit(np.log10(SIZES), observed_valid, 1)
naive_pred = naive_slope * np.log10(NEXT_SIZE) + naive_int
print(f"  naive extrapolation of valid-JSON rate: {naive_pred:.4f}")
print(f"  (composing the smooth forecast gives {pred_valid:.4f} — the naive "
      f"line is fitted to a curve that is not straight)")

# How much per-field accuracy does the requirement actually need?
for target in (0.90, 0.95, 0.99):
    needed = target ** (1 / FIELDS)
    print(f"\nto reach {target:.0%} valid JSON, per-field must reach "
          f"{needed:.4f}")
    print(f"  currently {per_field[-1]:.4f} at 70B -> the gap is "
          f"{needed - per_field[-1]:+.4f}")

print("""
Two lessons, and the second is the one teams get wrong.

Forecast the CONTINUOUS quantity and compose the metric afterwards. Per-field
accuracy extrapolates sensibly in logit space; the valid-JSON rate does not
extrapolate at all, because it is a seventh power and fitting a straight line
to it is meaningless.

But the REQUIREMENT is still all-or-nothing. Knowing that the sharpness is a
metric artefact does not soften the product constraint one bit — it just tells
you which number to measure and forecast. The discontinuity is real for the
deployment even though it is not real in the model.""")

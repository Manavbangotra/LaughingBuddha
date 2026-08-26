# -*- coding: utf-8 -*-
# Extracted from: Chapter 83 — Emergent Capabilities and What Emergence Means
# Source: src/.../ch083-emergence.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Can the continuous metric tell a real jump from a metric artefact? Yes."""
import numpy as np

scales = np.logspace(7, 11, 40)
log_s = np.log10(scales)
K = 12

# Case A: smooth competence, discontinuous METRIC (the Schaeffer scenario).
p_smooth = 1 / (1 + np.exp(-(log_s - 9.0) * 1.6))
metric_A = p_smooth ** K

# Case B: genuinely discontinuous competence — the model acquires the ability
# at a threshold — scored with the SAME exact-match metric.
p_step = np.where(log_s < 9.0, 0.05, 0.95)
metric_B = p_step ** K

# Case C: the same genuine discontinuity, scored continuously.
metric_C = p_step


def transition_decades(y, x_log):
    """How many decades of scale the curve needs to go from 10% to 90% of its
    range. Narrow means sharp. This is the right measure: unlike a per-step
    jump, it does not depend on how densely the scale axis was sampled."""
    y = (y - y.min()) / (y.max() - y.min() + 1e-12)
    lo = x_log[np.argmax(y >= 0.1)]
    hi = x_log[np.argmax(y >= 0.9)]
    return float(hi - lo)


print(f"{'case':<44} {'metric':<8} {'10%->90% width (decades)':>26}")
rows = [
    ("A: smooth ability, exact-match metric", "p^12", metric_A),
    ("A: smooth ability, continuous metric", "p", p_smooth),
    ("B: real discontinuity, exact-match metric", "p^12", metric_B),
    ("C: real discontinuity, continuous metric", "p", metric_C),
]
for label, m, y in rows:
    print(f"{label:<44} {m:<8} {transition_decades(y, log_s):>26.3f}")

print("""
Compare row 1 against row 2, then row 3 against row 4.

The exact-match metric more than halves the apparent transition width of a
perfectly smooth ability (2.46 decades -> 1.03). Sample that at four model
sizes an order of magnitude apart and it reads as a step. The sharpness was
added by the exponent.

The real discontinuity measures 0.00 decades under BOTH metrics. No rescoring
smooths a competence that genuinely stepped, which is what makes the continuous
metric a test rather than a way of explaining emergence away: row 2 and row 4
differ by everything, while row 1 and row 3 are both narrow enough to be
mistaken for each other on a sparse sweep.""")

# The exact-match metric compresses the smooth ability's transition...
assert transition_decades(metric_A, log_s) < 0.5 * transition_decades(p_smooth, log_s), \
    "p^k must narrow the transition of a smooth ability"
# ...but a genuine step stays a step however it is scored.
assert transition_decades(metric_C, log_s) < 0.2, \
    "a real discontinuity survives rescoring"
assert transition_decades(p_smooth, log_s) > 1.0, \
    "the smooth ability must remain wide under a continuous metric"

# -*- coding: utf-8 -*-
# Extracted from: Chapter 84 — Instruction Tuning
# Source: src/.../ch084-instruction-tuning.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Instruction data: does quality or quantity move the needle?"""
import numpy as np

rng = np.random.default_rng(3)


def simulate(n_examples, n_templates, quality, noise_floor=0.02):
    """A stand-in for downstream instruction-following quality.

    Three inputs, deliberately not symmetric:
      - n_examples: strong diminishing returns (log-shaped)
      - n_templates: teaches the INVARIANCE of eq:template-invariance
      - quality: bounds what can be learned at all — a ceiling, not a term
    """
    from_count = 0.30 * (1 - np.exp(-n_examples / 2000))
    from_templates = 0.25 * (1 - np.exp(-n_templates / 6))
    ceiling = quality
    raw = 0.25 + from_count + from_templates
    return min(raw, ceiling) - rng.normal(0, noise_floor)


print(f"{'option':<38} {'examples':>9} {'templates':>10} {'quality':>8} "
      f"{'result':>8}")
options = [
    ("as-is", 3_000, 2, 0.72),
    ("10x more examples, same templates", 30_000, 2, 0.72),
    ("same examples, 12 templates", 3_000, 12, 0.72),
    ("same examples, clean to high quality", 3_000, 2, 0.88),
    ("clean + diversify (one month)", 3_000, 12, 0.88),
    ("10x examples AND clean + diversify", 30_000, 12, 0.88),
]
results = {}
for name, n, k, q in options:
    r = simulate(n, k, q)
    results[name] = r
    print(f"{name:<38} {n:>9,} {k:>10} {q:>8.2f} {r:>8.3f}")

base = results["as-is"]
print(f"\n{'intervention':<38} {'gain over as-is':>17}")
for name, r in results.items():
    if name != "as-is":
        print(f"{name:<38} {r - base:>+17.3f}")

print("""
Three readings, and the third is the one that decides the month.

Diversifying templates on the data already collected beats collecting ten times
as much data with the same two templates. Example count has diminishing returns
because the model is re-weighting a mixture (eq:continuation-mixture), and
re-weighting does not need many samples; template diversity teaches an
INVARIANCE (eq:template-invariance), which is a different kind of thing.

But cleaning ALONE barely moves anything — less than the extra data does. That
is not a contradiction, it is what a ceiling means. At two templates the
configuration is nowhere near 0.72, so raising the cap to 0.88 has nothing to
bind on. Quality is not a term you add, it is a limit you eventually hit.

Hence the ordering. Diversify first, because it raises the achieved score; then
clean, because cleaning is what lets the raised score keep going. Doing them in
the other order looks like quality work that did not pay, and teams conclude
from that experience that data quality does not matter.""")

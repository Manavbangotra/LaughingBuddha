# -*- coding: utf-8 -*-
# Extracted from: Chapter 97 — Long-Context Behavior and Its Limits
# Source: src/.../ch097-long-context.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The position curve, measured. Equations (eq:needle-test) and (eq:u-shape)."""
import numpy as np

rng = np.random.default_rng(0)


def retrieval_accuracy(position, total_tokens, trials=3000):
    """A model of the measured effect: high at both ends, degraded in the
    middle, with the depression DEEPENING as context grows.

    The functional form is a stand-in; the shape is what liu2023lost measured.
    """
    # Depth grows with length — attention mass per position falls as T rises.
    depth = 0.45 * (1 - np.exp(-total_tokens / 40_000))
    quality = 1.0 - depth * np.sin(np.pi * position) ** 0.7
    base = 0.97
    p = np.clip(base * quality, 0.0, 1.0)
    return float((rng.random(trials) < p).mean())


LENGTHS = [2_000, 8_000, 32_000, 128_000]
POSITIONS = [0.0, 0.15, 0.3, 0.5, 0.7, 0.85, 1.0]

print("Needle retrieval accuracy by position and context length\n")
print(f"{'context':>9} " + " ".join(f"{p:>7.0%}" for p in POSITIONS) +
      f" {'depth':>8} {'usable?':>8}")
for T in LENGTHS:
    row = [retrieval_accuracy(p, T) for p in POSITIONS]
    depth = max(row) - min(row)
    usable = "yes" if min(row) >= 0.90 else "NO"
    print(f"{T:>9,} " + " ".join(f"{a:>7.3f}" for a in row) +
          f" {depth:>8.3f} {usable:>8}")

print("""
Read down the columns. The two edge columns barely move with context length —
a fact at the very start or the very end is retrieved reliably at 128k. The
middle column falls steadily, and the depth of the U grows with T.

That is the supported/usable distinction (eq:usable-context) as a measurement:
every one of these lengths is "supported", and by an accuracy floor of 0.90 only
the shortest two are usable for evidence at an arbitrary position.""")

# Equation (eq:usable-context): find the usable length for a required floor.
print(f"\n{'accuracy floor':>16} {'usable length':>16}")
for alpha in (0.95, 0.90, 0.80, 0.70):
    usable = 0
    for T in (1_000, 2_000, 4_000, 8_000, 16_000, 32_000, 64_000, 128_000):
        worst = min(retrieval_accuracy(p, T) for p in POSITIONS)
        if worst >= alpha:
            usable = T
    print(f"{alpha:>16.0%} {usable:>16,}")

# Equation (eq:multi-fact-degradation): what multi-fact tasks cost.
print(f"\n{'context':>9} {'1 fact':>9} {'3 facts':>9} {'5 facts':>9} "
      f"{'10 facts':>10}")
for T in LENGTHS:
    worst = min(retrieval_accuracy(p, T) for p in POSITIONS)
    row = " ".join(f"{worst ** m:>9.3f}" for m in (1, 3, 5))
    print(f"{T:>9,} {row} {worst ** 10:>10.3f}")

print("""
This is the gap between benchmark and production. A needle test places ONE fact
and reports the first column; users ask questions needing several, and
equation (eq:multi-fact-degradation) compounds the per-fact rate. At 128k a
single fact at the worst position is found 56% of the time, five facts 5% of the
time, and ten facts essentially never.

A model can pass a needle benchmark convincingly and be unusable for the task
people bought it for.""")

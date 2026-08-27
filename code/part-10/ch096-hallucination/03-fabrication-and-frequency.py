# -*- coding: utf-8 -*-
# Extracted from: Chapter 96 — Hallucination: Causes, Taxonomy, and Mitigation
# Source: src/.../ch096-hallucination.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Fabrication concentrates where the corpus was thin. Eq (eq:fabrication-vs-frequency)."""
import numpy as np

rng = np.random.default_rng(3)

# Entity frequency in a corpus is heavy-tailed (ch:nlp-preprocessing's Zipf).
N_ENTITIES = 6000
ranks = np.arange(1, N_ENTITIES + 1)
frequency = 1e7 / ranks ** 1.1


def fabrication_rate(n):
    """Decreasing in corpus frequency: below n*, the model fills in the
    typical pattern for entities of this kind rather than recalling."""
    n_star = 800.0
    return float(1 / (1 + (n / n_star) ** 0.8))


print(f"{'rank':>8} {'corpus mentions':>17} {'P(fabricate)':>14} "
      f"{'entity class'}")
CLASSES = [(1, "the most-discussed entities"), (10, ""), (100, ""),
           (500, ""), (2000, ""), (5000, "long tail")]
for r, note in CLASSES:
    n = frequency[r - 1]
    print(f"{r:>8,} {n:>17,.0f} {fabrication_rate(n):>14.3f}  {note}")

# What that means for an evaluation set's composition.
print(f"\n{'evaluation set':<32} {'mean P(fabricate)':>19}")
SETS = {
    "famous entities only (top 100)":    ranks[:100],
    "uniform over all entities":         ranks,
    "sampled by corpus frequency":       None,
    "long tail only (rank > 3000)":      ranks[3000:],
}
for name, sel in SETS.items():
    if sel is None:
        p = frequency / frequency.sum()
        sel = rng.choice(ranks, size=4000, p=p)
    rates = [fabrication_rate(frequency[r - 1]) for r in sel]
    print(f"{name:<32} {np.mean(rates):>19.3f}")

print("""
The same model has a fabrication rate varying by more than an order of magnitude
depending on which entities you evaluate it on — and 'sampled by corpus
frequency' looks excellent because it is dominated by entities the model has
seen thousands of times.

That is the measurement trap. A benchmark built from prominent entities
understates fabrication on exactly the queries where users encounter it, because
users ask about the things THEY care about, not the things the corpus discussed
most. Evaluation sets must be sampled from the traffic distribution, not from a
convenient one.""")

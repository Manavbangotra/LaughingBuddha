# -*- coding: utf-8 -*-
# Extracted from: Chapter 97 — Long-Context Behavior and Its Limits
# Source: src/.../ch097-long-context.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Ordering retrieved passages to exploit the U. Equation (eq:fold-ordering)."""
import numpy as np

N_PASSAGES = 20


def position_quality(p, depth=0.35):
    """The U from eq:u-shape, as a quality weight in [0, 1]."""
    return 1.0 - depth * np.sin(np.pi * p)


def utility(order, scores):
    """Equation (eq:position-ordering): relevance x position quality."""
    n = len(order)
    return float(sum(scores[idx] * position_quality((i + 0.5) / n)
                     for i, idx in enumerate(order)))


scores = np.array([0.9 ** i for i in range(N_PASSAGES)])
ranked = list(range(N_PASSAGES))          # already sorted by relevance


def fold(ranked):
    """Equation (eq:fold-ordering): best passages to the two ends, worst to
    the middle. Odd ranks ascend from the start, even ranks descend to the end."""
    front, back = [], []
    for i, idx in enumerate(ranked):
        (front if i % 2 == 0 else back).append(idx)
    return front + back[::-1]


ARRANGEMENTS = {
    "descending relevance": ranked,
    "ascending relevance":  ranked[::-1],
    "random":               list(np.random.default_rng(0).permutation(N_PASSAGES)),
    "folded (eq:fold-ordering)": fold(ranked),
}

print(f"{N_PASSAGES} passages, relevance decaying as 0.9^i\n")
print(f"{'arrangement':<28} {'utility':>9} {'vs descending':>15} "
      f"{'best passage at':>16}")
base = utility(ranked, scores)
for name, order in ARRANGEMENTS.items():
    u = utility(order, scores)
    pos = order.index(0) / N_PASSAGES
    print(f"{name:<28} {u:>9.3f} {u / base - 1:>+14.1%} {pos:>15.0%}")

print(f"\nfolded order: {fold(ranked)[:6]} ... {fold(ranked)[-4:]}")
print("""
The folded arrangement puts the two best passages at the two ends — the premium
positions — and buries the weakest in the middle where the model reads poorly.
It is a reordering of a list you already have and it costs nothing.

Note that descending and ASCENDING relevance score identically. That is not a
coincidence: the quality function is symmetric, so reversing a list maps every
passage to a position of equal quality. Reversal is not an intervention;
folding is, because it is the only arrangement that treats BOTH ends as
premium.

Most systems concatenate in descending relevance, which places the second-best
passage at position 2 and the mid-ranked ones squarely in the middle — an
arrangement optimised for a uniform-quality context that does not exist.""")

# Dropping the tail: what do the weakest passages actually contribute?
print(f"\n{'passages kept':>14} {'utility':>9} {'tokens':>9} "
      f"{'utility per 1k tokens':>23}")
TOKENS_EACH = 500
for k in (20, 15, 10, 6, 4, 2):
    order = fold(list(range(k)))
    u = utility(order, scores[:k])
    toks = k * TOKENS_EACH
    print(f"{k:>14} {u:>9.3f} {toks:>9,} {u / toks * 1000:>23.3f}")

print("""
Utility per token rises sharply as the tail is dropped. Passages 11-20
contribute little raw relevance AND sit at the worst positions, so removing them
costs almost nothing and saves 5,000 tokens of prefill.

That is the same conclusion ch:llm-inference's capacity planning reached from
the cost side: retrieve fewer, better passages. Here it arrives from the QUALITY
side, which makes it a rare case where the cheap option is also the good one.""")

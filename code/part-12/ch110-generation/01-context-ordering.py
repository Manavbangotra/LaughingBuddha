# -*- coding: utf-8 -*-
# Extracted from: Chapter 110 — Prompt Construction, Generation, and Citation
# Source: src/.../ch110-generation.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Reordering the same chunks, for free.

A model does not use context uniformly by position (ch:llm-long-context). Given
a position-utilisation profile rho, eq:optimal-ordering says pair the most
relevant chunk with the best position -- which under a U-shaped rho means an
outside-in interleave, NOT descending rank.

Nothing here changes which chunks are sent or how many tokens are spent. It is a
permutation of text you were already going to send.

The rho profiles below are INPUTS, stated explicitly rather than fitted: a flat
control, a U-shape of the kind reported for long-context models, and a
monotone-decreasing alternative. The conclusion is a function of rho, and the
listing shows how it changes when rho does.
"""
import numpy as np

rng = np.random.default_rng(31)

N_CHUNK, N_TRIAL = 12, 4000

PROFILES = {
    "flat": lambda n: np.ones(n),
    "U-shaped": lambda n: 0.45 + 0.55 * np.abs(np.linspace(-1, 1, n)) ** 1.6,
    "monotone decreasing": lambda n: np.linspace(1.0, 0.35, n),
}


def descending(order_by_relevance):
    """The usual: best chunk first, then descending."""
    return list(order_by_relevance)


def outside_in(order_by_relevance):
    """eq:u-shape-ordering: 1st, last, 2nd, second-last, ... -- the most
    relevant chunks at the ends, the least in the middle."""
    n = len(order_by_relevance)
    slots = [None] * n
    lo, hi = 0, n - 1
    for j, c in enumerate(order_by_relevance):
        if j % 2 == 0:
            slots[lo] = c
            lo += 1
        else:
            slots[hi] = c
            hi -= 1
    return slots


def shuffled(order_by_relevance):
    return list(rng.permutation(order_by_relevance))


STRATEGIES = {"descending rank": descending,
              "outside-in": outside_in,
              "random": shuffled}


# One fixed set of relevance draws, shared by every strategy and profile, so
# that any difference between strategies is attributable to placement alone.
DRAWS = np.sort(rng.random((N_TRIAL, N_CHUNK)), axis=1)[:, ::-1]


def expected_utilisation(strategy, rho):
    """Sum of relevance x position-utilisation over the shared relevance draws --
    the objective of eq:context-assembly with the subset held fixed."""
    totals = []
    for rel in DRAWS:
        placed = strategy(list(range(N_CHUNK)))       # placed[p] = chunk index
        totals.append(sum(rel[c] * rho[p] for p, c in enumerate(placed)))
    return float(np.mean(totals))


for pname, pfn in PROFILES.items():
    rho = pfn(N_CHUNK)
    print(f"\n{pname} profile   rho = "
          + " ".join(f"{v:.2f}" for v in rho))
    base = None
    for sname, sfn in STRATEGIES.items():
        val = expected_utilisation(sfn, rho)
        if base is None:
            base = val
        print(f"   {sname:<20}{val:>9.4f}{(val / base - 1) * 100:>+9.2f}%")

print("""
Under the FLAT profile the deterministic strategies score IDENTICALLY -- not
approximately, exactly -- because with rho constant the objective is just the sum
of relevances and placement cannot change it. That is the control: if position
does not matter, ordering does not matter, and every difference in the blocks
below is attributable to rho alone.

Under the U-SHAPED profile the outside-in order beats descending rank. The
mechanism is eq:optimal-ordering, a rearrangement inequality: descending rank
puts the SECOND most relevant chunk in position two, which under a U-shape is
among the worst places in the context. Outside-in puts it last instead -- one of
the best -- and pushes the least relevant chunks into the middle where their low
utilisation costs least.

Note that random ordering scores close to descending rank. That is worth
knowing on its own: descending rank is not a good order that outside-in improves
slightly, it is a MEDIOCRE order that happens to be the obvious one. Sorting by
relevance feels right and, under a U-shape, buys almost nothing over shuffling.

And the gain is free. No extra tokens, no extra calls, no model change -- a
permutation of text already being sent.

The MONOTONE DECREASING block is the honest control on the recommendation. If
your model's rho falls with position rather than U-shaping, descending rank IS
optimal and outside-in is a mistake. eq:optimal-ordering depends on rho being
non-uniform, not on its shape, so measure rho for your model (ch:llm-long-context)
rather than adopting a conclusion from a paper about a different one.""")

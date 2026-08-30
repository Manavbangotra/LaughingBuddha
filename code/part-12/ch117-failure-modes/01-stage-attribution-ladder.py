# -*- coding: utf-8 -*-
# Extracted from: Chapter 117 — RAG Failure Modes and How to Debug Them
# Source: src/.../ch117-failure-modes.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Which stage to fix, and why the obvious way of deciding is biased.

A RAG pipeline is a chain of stages, and end-to-end accuracy tells you nothing
about which one is losing the queries. The standard localisation move is ORACLE
SUBSTITUTION: replace one stage with a perfect version and see how much the score
rises (eq:oracle-substitution).

Done one stage at a time, that measurement is biased in a direction that matters:
a downstream stage's headroom is HIDDEN while an upstream stage is broken, so the
procedure systematically under-rates exactly the stages you will need to fix
second (eq:downstream-underrating). Substituting a cumulative PREFIX of stages
instead gives increments that telescope, and therefore decompose the total gap
exactly (eq:prefix-decomposition).

This listing measures both on the same pipeline and compares the rankings.
"""
import numpy as np

rng = np.random.default_rng(97)

N = 200_000
STAGES = ("ingestion", "indexing", "retrieval", "generation")

# Per-stage conditional success. Ingestion is the badly broken one here, which is
# common and rarely where teams look first (ch:rag-ingestion).
BASE = {"ingestion": 0.55, "indexing": 0.90, "retrieval": 0.80, "generation": 0.75}

# The generator sometimes answers correctly from its own weights even when
# retrieval failed. That leak is what makes the pipeline non-multiplicative, and
# it is why this has to be simulated rather than multiplied out.
PARAMETRIC_LEAK = 0.12


def score(p):
    """End-to-end accuracy of the pipeline with per-stage success rates p."""
    reached = np.ones(N, dtype=bool)
    for s in ("ingestion", "indexing", "retrieval"):
        reached &= rng.random(N) < p[s]
    gen_ok = rng.random(N) < p["generation"]
    leak = (~reached) & (rng.random(N) < PARAMETRIC_LEAK) & gen_ok
    return float(((reached & gen_ok) | leak).mean())


def with_oracle(subset):
    p = dict(BASE)
    for s in subset:
        p[s] = 1.0
    return p


base = score(BASE)
ceiling = score(with_oracle(STAGES))

print(f"pipeline: " + ", ".join(f"{s} {BASE[s]:.2f}" for s in STAGES))
print(f"end-to-end accuracy {base:.3f}; perfect-pipeline ceiling {ceiling:.3f}; "
      f"gap {ceiling - base:.3f}\n")

print(f"{'stage':<14}{'fix it alone':>15}{'fix it in order':>18}{'':>4}"
      f"{'naive rank':>12}{'true rank':>11}")
print("-" * 74)

single = {s: score(with_oracle([s])) - base for s in STAGES}

prefix, prev = {}, base
for i, s in enumerate(STAGES):
    cur = score(with_oracle(STAGES[:i + 1]))
    prefix[s] = cur - prev
    prev = cur

naive_rank = {s: i + 1 for i, s in enumerate(sorted(STAGES, key=lambda x: -single[x]))}
true_rank = {s: i + 1 for i, s in enumerate(sorted(STAGES, key=lambda x: -prefix[x]))}

for s in STAGES:
    print(f"{s:<14}{single[s]:>15.3f}{prefix[s]:>18.3f}{'':>4}"
          f"{naive_rank[s]:>12}{true_rank[s]:>11}")

print("-" * 74)
print(f"{'sum':<14}{sum(single.values()):>15.3f}{sum(prefix.values()):>18.3f}"
      f"     (gap is {ceiling - base:.3f})")

print(f"""
Look at the two sums before anything else. The prefix increments add to
{sum(prefix.values()):.3f}, which is the gap exactly -- they telescope, because
each one is measured against a pipeline in which every earlier stage is already
perfect. The one-at-a-time jumps add to {sum(single.values()):.3f}, which is not
the gap and is not meant to be: those measurements overlap, and reporting them as
a breakdown implies an additivity they do not have.

The bias has a direction, and it is the direction that costs you. Fixing
generation alone is worth {single['generation']:.3f}, which looks like the
smallest project on the board. Fixed IN ORDER -- after the upstream stages are
working -- the same stage is worth {prefix['generation']:.3f}. The headroom was
always there; it was hidden, because a query the ingester dropped fails whether
or not the generator is any good.

So the one-at-a-time procedure systematically under-rates downstream stages while
anything upstream is broken. That is not a small distortion: here it changes the
ranking, and a team following the naive numbers would work through the pipeline
in an order that keeps under-delivering, because each fix unmasks the next.

One honesty note about the right-hand column before using it. Prefix increments
telescope for ANY ordering of the stages, so they always sum to the gap -- but
how much of the gap each stage is credited with DOES depend on the order chosen.
Pipeline order is used here because it is the order in which repairs actually
have to happen: you cannot fix retrieval for a document that was never ingested.
A fully order-free attribution is the Shapley value over stage subsets, which
costs 2^n evaluations and, for four stages, mostly confirms the same ranking.

The rule that follows is simple and worth stating plainly. LOCALISE IN PIPELINE
ORDER. Fix the earliest broken stage first, then re-measure everything -- the
measurements downstream of a repair are stale the moment the repair lands. A
diagnosis that took a day is worth repeating after every fix, and a dashboard
that reports one-at-a-time headroom for four stages simultaneously is reporting
four numbers that were never simultaneously true.""")

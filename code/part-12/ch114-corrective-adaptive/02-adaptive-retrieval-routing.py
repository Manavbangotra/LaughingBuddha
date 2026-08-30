# -*- coding: utf-8 -*-
# Extracted from: Chapter 114 — Corrective and Adaptive RAG
# Source: src/.../ch114-corrective-adaptive.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Should retrieval happen at all? The break-even accuracy of a router.

ch:rag-why treated retrieval as the architecture. cite:jeong2024adaptiverag treats
it as a DECISION -- always-retrieve is a policy, and like any policy it can be
wrong. Retrieval costs latency and context budget, and on a query the model
already answers well it costs accuracy too, because irrelevant retrieved text
displaces attention (ch:llm-long-context) and invites the model to ground on it.

A router that predicts the query type recovers the best of both fixed policies,
minus its own error rate. eq:router-breakeven says a router must clear a
computable accuracy before it beats simply picking the better fixed policy, and
that threshold depends on the query MIX. This listing computes it.
"""
import numpy as np

rng = np.random.default_rng(31)

N = 120_000

A_PARAM_KNOWN = 0.86    # accuracy answering from weights, on what the model knows
A_PARAM_UNKNOWN = 0.12  # accuracy answering from weights, on corpus-specific facts
A_RETR_CORPUS = 0.79    # accuracy with retrieval, when the corpus has the answer
DISTRACTION = 0.14      # accuracy lost when irrelevant context is retrieved anyway


def evaluate(known, route_retrieve):
    """known: query is answerable from the model's own weights.
    route_retrieve: whether this query is sent to the retriever."""
    a = np.where(
        route_retrieve,
        np.where(known, A_PARAM_KNOWN - DISTRACTION, A_RETR_CORPUS),
        np.where(known, A_PARAM_KNOWN, A_PARAM_UNKNOWN),
    )
    return a.mean(), route_retrieve.mean()


def router(known, accuracy):
    """A classifier that predicts 'needs retrieval' (i.e. not known) with the
    given per-query accuracy. At 0.5 it is a coin; at 1.0 it is the oracle."""
    correct = rng.random(len(known)) < accuracy
    return np.where(correct, ~known, known)


print(f"{N:,} queries. Answering from weights: {A_PARAM_KNOWN:.2f} on what the "
      f"model knows,\n{A_PARAM_UNKNOWN:.2f} on corpus-specific facts. With "
      f"retrieval: {A_RETR_CORPUS:.2f} on corpus facts,\nand a {DISTRACTION:.2f} "
      f"distraction penalty when retrieval fires on a query it was not needed for.\n")

print(f"{'known share':>12}{'never':>9}{'always':>9}{'oracle':>9}"
      f"{'r=0.70':>9}{'r=0.85':>9}{'r=0.95':>9}{'break-even r':>15}")
print("-" * 81)

for share_known in (0.05, 0.20, 0.40, 0.60, 0.80, 0.90):
    known = rng.random(N) < share_known
    never, _ = evaluate(known, np.zeros(N, dtype=bool))
    always, _ = evaluate(known, np.ones(N, dtype=bool))
    oracle, _ = evaluate(known, ~known)
    best_fixed = max(never, always)

    routed = {r: evaluate(known, router(known, r))[0] for r in (0.70, 0.85, 0.95)}

    # The router accuracy at which routing first beats the better fixed policy
    # (eq:breakeven-solved), found by bisection rather than by the formula, so
    # the two can be checked against each other.
    lo, hi = 0.50, 1.00
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if evaluate(known, router(known, mid))[0] >= best_fixed:
            hi = mid
        else:
            lo = mid
    breakeven = hi

    print(f"{share_known:>12.2f}{never:>9.3f}{always:>9.3f}{oracle:>9.3f}"
          f"{routed[0.70]:>9.3f}{routed[0.85]:>9.3f}{routed[0.95]:>9.3f}"
          f"{breakeven:>15.3f}")

print("""
Start with the two fixed policies. At a 5% known share always-retrieve wins by a
mile, 0.787 against 0.156. By a 90% known share the ordering has REVERSED --
never-retrieve scores 0.786 against always-retrieve's 0.727 -- because on a query
the model already answers, firing the retriever spends latency and budget to lose
accuracy. Neither policy is a default. Both are bets on a mix, and the bet flips
at a known share of about 0.83 for these numbers.

The distraction penalty is what makes always-retrieve loseable at all, and it is
the term most pipelines assume is zero. It is not: retrieved text that does not
answer the question still occupies the context window, still carries the
authority of having been retrieved, and still invites the model to ground on it
(ch:llm-hallucination). Set DISTRACTION to 0 and always-retrieve wins at every
mix -- which is precisely why teams who have never measured it believe it does.

Now read the break-even column together with the oracle column, because the two
tell one story and it is not the story routing is usually sold with.

Where one fixed policy dominates, routing is BOTH hard and pointless. At a 5%
known share the router must be right 98.9% of the time merely to match
always-retrieve -- and a PERFECT router would score 0.793 against
always-retrieve's 0.787, so the entire prize is six thousandths of accuracy. You
would be building a classifier that must be near-flawless in order to win almost
nothing.

Where the fixed policies are close, routing is both easy and valuable. At an 80%
known share the two are nearly tied (0.734 and 0.712), break-even falls to 0.542
-- barely better than a coin -- and the oracle scores 0.846, so perfect routing is
worth 0.112 of accuracy, nearly twenty times the prize at the other end.

The difficulty of routing and the value of routing move in OPPOSITE directions,
and both are determined by one measurable quantity: the query mix. So the
adaptive-RAG decision does not start with a classifier. It starts with an
afternoon spent labelling two hundred real queries for whether the corpus was
needed. If that sample comes back 95% corpus-dependent, the correct adaptive
policy is to always retrieve, and a router is a component that can only cost
you.""")

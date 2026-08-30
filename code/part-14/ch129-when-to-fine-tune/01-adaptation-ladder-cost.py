# -*- coding: utf-8 -*-
# Extracted from: Chapter 129 — When to Fine-Tune and When Not To
# Source: src/.../ch129-when-to-fine-tune.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The adaptation ladder, priced -- including the term everyone leaves out.

Four ways to make a model do what you want, in increasing order of commitment:

  PROMPT      write instructions into every request. Free to change, and you pay
              for those tokens on every call forever.
  FEW-SHOT    add examples to every request. Better behaviour, more tokens.
  RETRIEVE    fetch relevant context per query (Part XII). Handles facts, adds a
              retrieval call.
  FINE-TUNE   move the weights. No per-request overhead at all, one large
              up-front cost -- and a MAINTENANCE cost nobody budgets for
              (eq:adaptation-tco).

The first three are priced per query and the fourth is priced per CHANGE, so the
comparison depends on two numbers: how many queries you serve, and how often the
requirement moves. This listing computes the crossover in both.
"""
import numpy as np

# Token prices in arbitrary but consistent units.
P_IN = 1.0                     # cost per 1k input tokens
BASE_TOKENS = 250              # the actual question
PROMPT_TOKENS = 900            # instructions repeated every call
FEWSHOT_TOKENS = 2600          # instructions + eight examples
RAG_TOKENS = 2200              # instructions + retrieved chunks
RAG_RETRIEVAL = 0.15           # the retrieval call itself, per query

TRAIN_COST = 40_000.0          # one fine-tuning run
EVAL_COST = 15_000.0           # building and running the eval that says it worked
CHANGE_REWORK = 0.6            # fraction of train+eval repaid on each change


def per_query(mode):
    if mode == "prompt":
        return (BASE_TOKENS + PROMPT_TOKENS) / 1000 * P_IN
    if mode == "few-shot":
        return (BASE_TOKENS + FEWSHOT_TOKENS) / 1000 * P_IN
    if mode == "retrieve":
        return (BASE_TOKENS + RAG_TOKENS) / 1000 * P_IN + RAG_RETRIEVAL
    return BASE_TOKENS / 1000 * P_IN          # fine-tuned: no overhead


def total(mode, queries, changes):
    """eq:adaptation-tco: per-query cost times volume, plus per-change cost
    times churn. Only fine-tuning has a meaningful second term."""
    variable = per_query(mode) * queries
    if mode == "fine-tune":
        fixed = TRAIN_COST + EVAL_COST + changes * CHANGE_REWORK * (TRAIN_COST + EVAL_COST)
    else:
        # Changing a prompt is not free either -- someone edits and re-tests it.
        fixed = changes * 400.0
    return variable + fixed


MODES = ("prompt", "few-shot", "retrieve", "fine-tune")
VOLUMES = (10_000, 100_000, 1_000_000, 10_000_000)
CHANGES = (0, 2, 12, 52)

print("cost per query, before any fixed cost:")
for m in MODES:
    print(f"   {m:<12}{per_query(m):>8.3f}")
print(f"\none fine-tuning run costs {TRAIN_COST + EVAL_COST:,.0f} "
      f"(training + the eval that proves it worked)")
print(f"each requirement change repays {CHANGE_REWORK:.0%} of that\n")

for changes in CHANGES:
    label = {0: "requirements never change", 2: "2 changes/year",
             12: "monthly changes", 52: "weekly changes"}[changes]
    print(f"--- {label} ---")
    print(f"{'queries/year':>14}" + "".join(f"{m:>14}" for m in MODES)
          + f"{'winner':>12}")
    for v in VOLUMES:
        costs = {m: total(m, v, changes) for m in MODES}
        win = min(costs, key=costs.get)
        print(f"{v:>14,}" + "".join(f"{costs[m]:>14,.0f}" for m in MODES)
              + f"{win:>12}")
    print()

# Where does fine-tuning start winning, as a function of churn?
# (eq:breakeven-volume, found by bisection rather than the formula so the two
# can be checked against each other.)
print(f"{'changes/year':>14}{'break-even query volume':>26}")
print("-" * 41)
be = {}
for changes in (0, 1, 4, 12, 26, 52):
    lo, hi = 1.0, 1e12
    for _ in range(80):
        mid = (lo + hi) / 2
        best_other = min(total(m, mid, changes) for m in MODES if m != "fine-tune")
        if total("fine-tune", mid, changes) < best_other:
            hi = mid
        else:
            lo = mid
    be[changes] = hi
    print(f"{changes:>14}{hi:>26,.0f}")

print(f"""
The per-query table at the top is where the intuition for fine-tuning comes from,
and it is correct as far as it goes: a fine-tuned model carries no instructions,
no examples and no retrieved context, so its marginal cost is the question alone.
Against few-shot prompting that is a factor of about ten per call, and at high
volume a factor of ten is the whole argument.

Read down the volume rows in the first block and that argument holds. With
requirements that never change, fine-tuning overtakes the alternatives somewhere
in the hundreds of thousands of queries and wins decisively above that. If you
serve a stable, high-volume workload, the case is not close.

Now read across the blocks, because the churn dimension is the one that is
usually missing from the decision. Every requirement change repays a large
fraction of the training and evaluation cost -- not because retraining is
technically hard, but because the EVAL has to be rebuilt and rerun to establish
that the new model did not break the previous behaviour. A prompt change costs an
edit and a test run. A weight change costs a release.

The break-even table makes that concrete. With no churn the crossover sits at
{be[0]:,.0f} queries a year -- a modest system clears it. At weekly changes it is
{be[52]:,.0f}, a factor of {be[52]/be[0]:.0f} higher, and for many real systems
that is past any volume they will ever see. Same technique, same model, same cost
per token, and the recommendation inverts on a variable that has nothing to do
with machine learning.

Which is the practical point of this listing. The fine-tuning decision is usually
argued on per-query cost, where fine-tuning wins, and it is usually DECIDED by
churn, where it often loses. Before running a training job, write down how many
times last year someone changed what the system was supposed to do. That number,
not the token price, is the one that decides.

Two things this model deliberately understates, both in fine-tuning's disfavour.
It does not price the risk that a fine-tune regresses behaviour nobody was
testing for (ch:ft-training-config), and it does not price the fact that a
prompt can be rolled back in seconds while a model cannot. Include those and the
break-even volumes rise further.""")

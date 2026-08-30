# -*- coding: utf-8 -*-
# Extracted from: Chapter 115 — Agentic RAG
# Source: src/.../ch115-agentic-rag.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A loop is not a chain -- but only if its failures are visible.

ch:llm-function-calling's eq:tool-chain-success says h steps at per-step success p
succeed together with probability p^h, and at h=5, p=0.92 that is 0.66. The usual
conclusion is that multi-step retrieval cannot work.

The conclusion is wrong, and the reason matters more than the arithmetic. A CHAIN
compounds because a failed step is passed downstream unnoticed. A LOOP can
observe its own step and retry it. So the compounding penalty is not a property
of having many steps -- it is a property of not being able to SEE a bad step,
which is exactly what ch:rag-corrective's grader provides.

This listing sweeps observability against per-step accuracy and asks which one is
worth buying.
"""
import numpy as np

rng = np.random.default_rng(17)

N_TASK = 40_000
BUDGET = 10                     # total retrieval steps the agent may spend


def run(p_step, observability, budget=BUDGET, depths=(1, 2, 3, 4)):
    """Simulate an agentic retrieval loop.

    Each hop succeeds with probability p_step. A failed hop is DETECTED with
    probability `observability` -- detected failures are retried (costing a step
    from the budget); undetected failures are carried forward silently and the
    task is lost, which is the chain behaviour (eq:loop-degenerates).
    """
    depth = rng.choice(depths, size=N_TASK)
    solved = np.zeros(N_TASK, dtype=bool)
    steps_used = np.zeros(N_TASK, dtype=int)

    for t in range(N_TASK):
        hops_done, steps, alive = 0, 0, True
        while alive and hops_done < depth[t] and steps < budget:
            steps += 1
            if rng.random() < p_step:
                hops_done += 1
            elif rng.random() >= observability:
                alive = False           # silent bad hop: the loop became a chain
        solved[t] = alive and hops_done == depth[t]
        steps_used[t] = steps
    return solved.mean(), steps_used.mean()


print(f"{N_TASK:,} multi-hop tasks of depth 1-4; budget {BUDGET} retrieval steps\n")
print(f"{'p_step':>8}{'chain (o=0)':>14}{'o=0.25':>10}{'o=0.50':>10}"
      f"{'o=0.75':>10}{'o=0.95':>10}{'steps @o=0.95':>16}")
print("-" * 78)

table = {}
for p_step in (0.75, 0.85, 0.92):
    row = {}
    for o in (0.0, 0.25, 0.50, 0.75, 0.95):
        row[o] = run(p_step, o)
    table[p_step] = row
    print(f"{p_step:>8.2f}" + "".join(f"{row[o][0]:>10.3f}" if o else
                                      f"{row[o][0]:>14.3f}"
                                      for o in (0.0, 0.25, 0.50, 0.75, 0.95))
          + f"{row[0.95][1]:>16.2f}")

better_p = table[0.92][0.25][0]
better_o = table[0.75][0.95][0]
print(f"""
Read the first column as the pessimistic claim, and it is correct as far as it
goes: with no observability the loop IS a chain, and success is the product of
per-step successes over the task's depth. At p_step = 0.75 that is
{table[0.75][0.0][0]:.3f}.

Now read across a row rather than down a column. Holding per-step accuracy fixed
at the WORST value in the table, raising observability from 0 to 0.95 takes
success from {table[0.75][0.0][0]:.3f} to {better_o:.3f}. Holding observability
near the bottom and raising per-step accuracy from 0.75 all the way to 0.92 --
a large, expensive improvement in retrieval quality -- reaches only
{better_p:.3f}.

Being able to SEE a bad step is worth more than making bad steps rarer. That is
not a general law, it is a consequence of the loop structure: a detected failure
costs one step from a budget, while an undetected failure costs the task. The
same grader ch:rag-corrective built for a single retry is what converts a
compounding chain into a self-correcting loop, and without it, agentic retrieval
inherits eq:tool-chain-success in full.

Which reverses the usual build order. Teams add iteration first and observability
later, because iteration is the visible feature. The table says the observability
is the load-bearing part, and iteration without it makes things worse -- more
steps, each a fresh chance to go silently wrong.""")

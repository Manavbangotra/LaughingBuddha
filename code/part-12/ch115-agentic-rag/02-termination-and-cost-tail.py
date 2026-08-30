# -*- coding: utf-8 -*-
# Extracted from: Chapter 115 — Agentic RAG
# Source: src/.../ch115-agentic-rag.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Termination, and where an agentic loop actually spends its money.

ch:rag-corrective's loop ran exactly once. Removing that bound is what makes
retrieval agentic, and it introduces a problem a fixed pipeline never had: the
loop has to decide when to stop.

The cost is not evenly distributed. A query whose answer exists is finished when
it is found; a query whose answer is ABSENT never finishes, so it runs to
whatever limit exists. eq:adverse-selection says the queries that consume the
most compute are systematically the ones that will not produce an answer, and
eq:quadratic-context says the token bill grows faster than the step count because
each iteration re-reads everything the previous ones retrieved.

This listing measures both, and prices three termination policies.
"""
import numpy as np

rng = np.random.default_rng(53)

N_QUERY = 40_000
UNANSWERABLE = 0.15         # share of queries whose answer is not in the corpus
P_STEP = 0.85               # a retrieval step makes progress
OBSERVABILITY = 0.85        # a bad step is noticed (ch:rag-corrective's grader)
BUDGET = 12
NO_PROGRESS_LIMIT = 3       # consecutive steps with no new information
P_DECLARE = 0.30            # chance the model calls itself finished, per step

BASE_TOKENS = 900           # instructions, question, scratchpad
STEP_TOKENS = 1100          # what each retrieval adds, and never removes


def tokens_for(steps):
    """eq:quadratic-context: iteration i re-reads every earlier retrieval, so
    the bill is the SUM of growing prompts, not steps x prompt."""
    return steps * BASE_TOKENS + STEP_TOKENS * steps * (steps + 1) // 2


def simulate(policy):
    depth = rng.choice((1, 2, 3, 4), size=N_QUERY)
    dead_end = rng.random(N_QUERY) < UNANSWERABLE

    correct = np.zeros(N_QUERY, dtype=bool)
    harmful = np.zeros(N_QUERY, dtype=bool)     # answered, and wrong
    steps_used = np.zeros(N_QUERY, dtype=int)

    for t in range(N_QUERY):
        # `hops` is real progress; `felt` is what the agent BELIEVES it has,
        # and the two diverge exactly when a bad step goes unnoticed
        # (eq:felt-versus-real).
        hops, felt, steps, stale, alive, answered = 0, 0, 0, 0, True, False
        while steps < BUDGET:
            steps += 1
            if (not dead_end[t]) and rng.random() < P_STEP:
                hops, felt, stale = hops + 1, felt + 1, 0
            elif rng.random() < OBSERVABILITY:
                stale += 1                      # noticed: nothing was learned
            else:
                felt, stale = felt + 1, 0       # NOT noticed: false progress
                if not dead_end[t]:
                    alive = False               # and wrong information carried on
            if alive and not dead_end[t] and hops >= depth[t]:
                answered = True
                break
            if policy == "self-report" and felt >= 1 and rng.random() < P_DECLARE:
                # The model declares itself finished, on `felt` rather than on
                # `hops` -- it cannot see the difference, which is the point.
                answered = True
                break
            if policy == "no-progress" and stale >= NO_PROGRESS_LIMIT:
                break
        steps_used[t] = steps
        if answered:
            ok = alive and (not dead_end[t]) and hops >= depth[t]
            correct[t] = ok
            harmful[t] = not ok
    return correct, harmful, steps_used, dead_end


print(f"{N_QUERY:,} queries, {UNANSWERABLE:.0%} with no answer in the corpus. "
      f"Budget {BUDGET} steps,\np_step {P_STEP}, observability {OBSERVABILITY}. "
      f"Prompt grows by {STEP_TOKENS} tokens per iteration.\n")
print(f"{'termination':<16}{'correct':>9}{'harm':>8}{'abstain':>9}"
      f"{'mean':>8}{'p95':>7}{'Mtok':>9}{'% tok on dead ends':>21}")
print("-" * 87)

for policy in ("budget only", "self-report", "no-progress"):
    correct, harmful, steps, dead = simulate(policy)
    tok = tokens_for(steps)
    print(f"{policy:<16}{correct.mean():>9.3f}{harmful.mean():>8.3f}"
          f"{1 - correct.mean() - harmful.mean():>9.3f}"
          f"{steps.mean():>8.2f}{np.percentile(steps, 95):>7.0f}"
          f"{tok.sum() / 1e6:>9.1f}{tok[dead].sum() / tok.sum():>20.1%}")

print(f"""
Start with the last column, because it is the one that shows up on an invoice.
Dead ends are {UNANSWERABLE:.0%} of the traffic and, under a budget-only policy,
53% of the token spend -- for output that cannot exist, because the answer is not
in the corpus. That is eq:adverse-selection stated as a bill: a loop runs until
it succeeds or until it is stopped, so the queries that never succeed are exactly
the queries that run longest. Fifteen per cent of the traffic, more than half the
compute, none of the answers.

eq:quadratic-context is what turns a bad ratio into a worse one. Each iteration
re-reads every earlier retrieval, so a {BUDGET}-step query does not cost
{BUDGET} times a 1-step query -- it costs {tokens_for(BUDGET) / tokens_for(1):.0f}
times as much. The dead ends' share of TOKENS is therefore far above their share
of STEPS, and a capacity plan built on mean step count is wrong in the expensive
direction.

Self-report is the policy most agent frameworks ship, and it is the worst row in
the table. Letting the model declare itself finished halves the bill and destroys
the system: correct answers fall from 0.796 to 0.487 and harm rises from 0.000 to
0.477. The mechanism is in the simulation: the model decides on `felt` progress,
which includes the bad steps it failed to notice, so it stops early on real
questions and fabricates answers on dead ends. Note that it does not even fix the
cost problem it was reached for -- the dead ends' share is 56.5%, HIGHER than
under a plain budget. This is ch:rag-corrective's terminal-handler problem wearing
a different hat: an unreliable decider handed an irreversible action.

The no-progress detector is the cheap fix, and cheap is an understatement.
Stopping after {NO_PROGRESS_LIMIT} consecutive steps that retrieved nothing new
holds correctness at 0.793 against budget-only's 0.796 -- a difference of three
thousandths -- while cutting the token bill by 45% and the dead ends' share from
53.0% to 17.6%. It costs nothing because a query that is making progress never
triggers it.

Look at what the detector measures, because that is the transferable part. Not
whether the agent is CONFIDENT, which it always is and which the self-report row
shows is worthless. Whether it is LEARNING ANYTHING -- which is a property of the
retrieved set, visible from outside the model, and cheap to check. Terminate on
the checkable signal, never on the model's own account of itself.""")

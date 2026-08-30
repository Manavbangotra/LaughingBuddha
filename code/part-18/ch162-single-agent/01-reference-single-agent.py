# -*- coding: utf-8 -*-
# Extracted from: Chapter 162 — Single-Agent Architectures
# Source: src/.../ch162-single-agent.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""What part:17's findings are worth stacked together.

Every chapter of part:17 measured one intervention against one baseline. This
listing puts them in one agent and measures the cumulative effect, because a
system is what you get when you apply all of them and the interactions are not
obvious (eq:reference-single-agent).

The components, in the order the part introduced them:

  errors      informative tool errors (ch:ag-tool-calling)
  stopbias    bias the stopping classifier against stopping (ch:ag-loop)
  dedupe      refuse to re-issue an action that already failed (ch:ag-loop)
  checkpoint  verified segment boundaries to resume from (ch:ag-planning)
  scratchpad  record derived values instead of recomposing (ch:ag-memory)
  pooled      a shared step budget rather than a per-task cap (ch:ag-termination)

This is also the baseline part:18 has to beat. Every multi-agent claim should be
measured against a single agent with all of this switched on, and usually is not.
"""
import numpy as np

rng = np.random.default_rng(2749)

M = 20000               # tasks
NEED = 10               # productive steps required
SEGMENTS = 5            # when checkpoints are on
BUDGET_PER = 26

P_ACT = 0.88            # a fresh action makes progress
STICK = 0.70            # chance of repeating a failed action without dedupe
P_FIX_OPAQUE = 0.03     # a retry after an opaque error
P_FIX_GOOD = 0.75       # a retry after an informative error
FPR_LOOSE = 0.01        # false stop rate, biased against stopping
FPR_TIGHT = 0.06        # false stop rate, tuned for promptness
TPR = 0.85
P_COMPOSE = 0.90        # recomposing a derived value inside one pass
P_LOOKUP = 0.985


def run(cfg, m=M, need=NEED, budget=BUDGET_PER):
    """cfg is a set of enabled component names."""
    errors = "errors" in cfg
    stopbias = "stopbias" in cfg
    dedupe = "dedupe" in cfg
    checkpoint = "checkpoint" in cfg
    scratch = "scratchpad" in cfg
    pooled = "pooled" in cfg

    fpr = FPR_LOOSE if stopbias else FPR_TIGHT
    seg_len = max(1, need // SEGMENTS) if checkpoint else need
    p_step = P_ACT * (P_LOOKUP if scratch else P_COMPOSE)

    prog = np.zeros(m, dtype=np.int64)
    anchor = np.zeros(m, dtype=np.int64)     # last verified progress
    used = np.zeros(m, dtype=np.int64)
    failed_last = np.zeros(m, dtype=bool)
    alive = np.ones(m, dtype=bool)
    early = np.zeros(m, dtype=bool)
    done = np.zeros(m, dtype=bool)

    total = m * budget
    spent = 0
    for _ in range(budget * 3):
        live = alive & ~done & ~early
        if pooled:
            live &= (spent < total)
        else:
            live &= (used < budget)
        idx = np.flatnonzero(live)
        if not len(idx):
            break
        used[idx] += 1
        spent += len(idx)

        # A repeated action reproduces its failure unless dedupe forbids it.
        rep = failed_last[idx] & (rng.random(len(idx)) < STICK) if not dedupe \
            else np.zeros(len(idx), dtype=bool)
        # A retry after a failure is conditioned only if the error said something.
        cond = failed_last[idx] & ~rep
        p = np.where(rep, 0.0,
                     np.where(cond, P_FIX_GOOD if errors else P_FIX_OPAQUE,
                              p_step))
        ok = rng.random(len(idx)) < p
        prog[idx[ok]] += 1
        failed_last[idx] = ~ok

        # Checkpoints: a verified boundary becomes the anchor; a failure past it
        # rolls back only to the anchor rather than losing everything.
        if checkpoint:
            at_boundary = ok & (prog[idx] % seg_len == 0)
            anchor[idx[at_boundary]] = prog[idx[at_boundary]]
            # Without a checkpoint a run that stalls restarts from zero.
            stalled = (~ok) & (rng.random(len(idx)) < 0.04)
            prog[idx[stalled]] = anchor[idx[stalled]]
        else:
            stalled = (~ok) & (rng.random(len(idx)) < 0.04)
            prog[idx[stalled]] = 0

        finished = prog[idx] >= need
        u = rng.random(len(idx))
        stop = np.where(finished, u < TPR, u < fpr)
        done[idx[stop & finished]] = True
        early[idx[stop & ~finished]] = True
        alive[idx[stop]] = False

    return (float(done.mean()), float(early.mean()), float(used.mean()))


ORDER = ["errors", "stopbias", "dedupe", "checkpoint", "scratchpad", "pooled"]

print(f"{M:,} tasks needing {NEED} productive steps, {P_ACT:.0%} per action,")
print(f"a budget of {BUDGET_PER} steps per task. Components are added one at a")
print("time, in the order part:17 introduced them.")
print()
print(f"{'configuration':>36}{'completed':>12}{'stopped early':>15}"
      f"{'steps':>9}{'gain':>9}")
print("-" * 81)
cum = {}
prev = None
cfg = set()
r = run(cfg)
cum["baseline (none)"] = r
print(f"{'baseline (none)':>36}{r[0]:>12.1%}{r[1]:>15.1%}{r[2]:>9.1f}"
      f"{'--':>9}")
prev = r[0]
for c in ORDER:
    cfg = cfg | {c}
    r = run(cfg)
    name = "+ " + c
    cum[name] = r
    print(f"{name:>36}{r[0]:>12.1%}{r[1]:>15.1%}{r[2]:>9.1f}"
          f"{r[0] - prev:>+9.1%}")
    prev = r[0]

print()
print()
print("Each component ALONE, against the same baseline -- so the additions above")
print("can be compared with what each is worth on its own.")
print()
print(f"{'component alone':>36}{'completed':>12}{'vs baseline':>13}")
print("-" * 61)
base = cum["baseline (none)"][0]
alone = {}
for c in ORDER:
    r = run({c})
    alone[c] = r[0]
    print(f"{c:>36}{r[0]:>12.1%}{r[0] - base:>+13.1%}")

print()
print()
print("And each component REMOVED from the full configuration -- what you lose")
print("by leaving it out of a system that has everything else.")
print()
print(f"{'component removed':>36}{'completed':>12}{'loss':>10}")
print("-" * 58)
full = set(ORDER)
full_score = run(full)[0]
drop = {}
for c in ORDER:
    r = run(full - {c})
    drop[c] = r[0]
    print(f"{c:>36}{r[0]:>12.1%}{r[0] - full_score:>+10.1%}")

print()
print()
print("How the full configuration and the bare one respond to more budget.")
print()
print(f"{'budget/task':>13}{'baseline':>11}{'full':>10}{'gap':>9}")
print("-" * 43)
bd = {}
for b in (14, 20, 26, 40, 60):
    a = run(set(), budget=b)[0]
    f = run(full, budget=b)[0]
    bd[b] = (a, f)
    print(f"{b:>13}{a:>11.1%}{f:>10.1%}{f - a:>+9.1%}")

print()
print()
print("And how they respond to task length, at a budget of 2.6x the task.")
print()
print(f"{'steps needed':>14}{'baseline':>11}{'full':>10}{'gap':>9}")
print("-" * 44)
kl = {}
for k in (4, 10, 20, 30):
    a = run(set(), need=k, budget=int(2.6 * k))[0]
    f = run(full, need=k, budget=int(2.6 * k))[0]
    kl[k] = (a, f)
    print(f"{k:>14}{a:>11.1%}{f:>10.1%}{f - a:>+9.1%}")

print(f"""
The first table is part:17 assembled, and the two ends of the column are the
argument for having read it.

A bare loop -- a model, tools, and a stopping decision tuned for promptness --
completes {cum['baseline (none)'][0]:.1%} of tasks and stops early on
{cum['baseline (none)'][1]:.1%} of them. Almost every failure is a confident
partial answer.

The same model with every component from part:17 completes
{cum['+ pooled'][0]:.1%}. **Nothing about the model changed.** The action accuracy
is {P_ACT:.0%} in both rows.

The three big steps are informative errors ({cum['+ errors'][0] - cum['baseline (none)'][0]:+.1%}),
biasing the stopping classifier against stopping
({cum['+ stopbias'][0] - cum['+ errors'][0]:+.1%}), and refusing to re-issue a
failed action ({cum['+ dedupe'][0] - cum['+ stopbias'][0]:+.1%}). None of the
three is a model change, an architecture change, or a framework choice. Two are
policy constants and one is a set membership test.

The second and third tables together contain the finding this listing exists for,
and it only appears when you compare them.

Informative errors ALONE buy {alone['errors'] - base:+.1%}. Removing them from a
system that has everything else costs {drop['errors'] - full_score:+.1%}.

Stop-bias alone buys {alone['stopbias'] - base:+.1%}. Removing it from the full
system costs {drop['stopbias'] - full_score:+.1%}.

**The components are worth several times more together than apart**, and the
mechanism is specific rather than mysterious. An informative error conditions a
retry -- but only if a retry happens, which requires the stopping classifier not
to have declared victory, and only if the retry is a different action, which
requires deduplication. Each one removes a blocker on the others.

That has a practical consequence that is easy to get backwards. **Evaluating these
interventions one at a time UNDERSTATES all of them**, and a team that A/B tests
each in isolation against a bare baseline will conclude that most of them are
marginal and ship none. The measurement that matters is the ablation from the full
system, not the addition to the empty one.

Note also the components that look weak in both tables. Checkpoints buy
{cum['+ checkpoint'][0] - cum['+ dedupe'][0]:+.1%} here and cost
{drop['checkpoint'] - full_score:+.1%} when removed, which is far less than
ch:ag-planning measured. That is not a contradiction: at {NEED} steps with
{BUDGET_PER} of budget and dedupe already working, the run rarely reaches the
state where a rollback matters. **A component's value depends on which failures
are still available for it to prevent**, and the ones added earlier have already
taken most of them.

The fourth table is the one to remember when someone proposes buying more budget.

The bare loop completes {bd[14][0]:.1%} at a budget of {14} steps per task and
{bd[60][0]:.1%} at {60} -- a fourfold increase in spend for
{bd[60][0] - bd[14][0]:+.1%}. It cannot use the budget, because it stops early
before exhausting it. The full configuration is flat too, at about
{bd[60][1]:.1%}, because it does not need the extra.

**Neither system is budget-limited, and their gap of about
{bd[26][1] - bd[26][0]:.0f} points is entirely structural.** Buying compute is the
most common response to an underperforming agent and it is the one this table
rules out first.

The last table is why the gap matters more as tasks get longer. At {4} steps the
bare loop reaches {kl[4][0]:.1%} and the full one {kl[4][1]:.1%}. At {30} steps it
is {kl[30][0]:.1%} against {kl[30][1]:.1%}.

The bare loop's completion falls off a cliff because every mechanism that would
have contained a failure is missing, and part:17's arithmetic -- p^k on the way
down, checkpoints and retries on the way back up -- all bites hardest at length.
**A design that looks acceptable on three-step demos is not a design that scales
to twenty**, and the difference is not the model.

That is the baseline part:18 has to beat. Every multi-agent claim in the next
eight chapters is measured against THIS number -- a single agent with informative
errors, a conservative stopping threshold, action deduplication, checkpoints, a
scratchpad and a pooled budget -- rather than against the {cum['baseline (none)'][0]:.1%}
that a naive loop achieves. cite:cemri2025mast's finding that multi-agent gains on
popular benchmarks are often minimal is much easier to understand once you notice
which baseline they are usually compared against.""")

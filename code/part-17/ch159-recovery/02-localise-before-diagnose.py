# -*- coding: utf-8 -*-
# Extracted from: Chapter 159 — Reflection, Replanning, and Error Recovery
# Source: src/.../ch159-recovery.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Detection, localisation, diagnosis: three different things a failure can tell you.

"The agent noticed it failed and tried again" collapses three separate pieces of
information into one sentence, and they are worth very different amounts
(eq:localise-before-diagnose).

  DETECTION    -- something went wrong
  LOCALISATION -- it went wrong at step j
  DIAGNOSIS    -- it went wrong at step j BECAUSE of x

Detection alone forces a restart from the beginning, which throws away every step
that was fine. Localisation lets the retry resume from the last good state, which
is exactly ch:ag-planning's checkpoint. Diagnosis conditions the retry, which is
ch:ag-tool-calling's error message.

This listing gives an agent each in turn and measures what each one buys, under a
fixed step budget.
"""
import numpy as np

rng = np.random.default_rng(2357)

N = 40000
K = 8                   # steps in the task
P_STEP = 0.88
BUDGET = 26
P_INFORMED = 0.97       # a retry that knows the cause


def run(mode, budget=BUDGET, k=K, p_step=P_STEP, loc_acc=1.0):
    """mode: 'none' | 'detect' | 'localise' | 'diagnose'.
    loc_acc is how often localisation identifies the right step."""
    done = np.zeros(N, dtype=bool)
    spent = np.zeros(N, dtype=np.int64)
    resume = np.zeros(N, dtype=np.int64)     # step to restart from
    for _ in range(budget):
        live = ~done & (spent < budget)
        idx = np.flatnonzero(live)
        if not len(idx):
            break
        start = resume[idx]
        # Run from `start` to the end, or until a step fails.
        fail_at = np.full(len(idx), k)
        for j in range(k):
            active = (start <= j) & (fail_at == k)
            if not active.any():
                break
            p = np.where((mode == "diagnose") & (resume[idx] == j),
                         P_INFORMED, p_step)
            bad = active & (rng.random(len(idx)) >= p)
            fail_at = np.where(bad, j, fail_at)
        spent[idx] += (k - start).clip(min=1)
        won = fail_at == k
        done[idx[won]] = True
        lost = ~won
        if mode == "none" or mode == "detect":
            resume[idx[lost]] = 0                       # restart from scratch
        else:
            guessed = fail_at
            if loc_acc < 1.0:
                miss = rng.random(len(idx)) >= loc_acc
                guessed = np.where(miss, 0, fail_at)    # a bad guess restarts
            resume[idx[lost]] = guessed[lost]
        if mode == "none":
            break                                       # a single attempt
    return float(done.mean()), float(spent.mean())


print(f"A {K}-step task at {P_STEP:.0%} per step, budget {BUDGET} steps. A retry")
print(f"that knows the cause succeeds at {P_INFORMED:.0%} on the failed step.")
print()
print(f"{'what the failure tells you':>32}{'completed':>12}{'steps':>9}"
      f"{'per step':>11}")
print("-" * 64)
tab = {}
for name, mode in [("nothing (one attempt)", "none"),
                   ("detection: restart", "detect"),
                   ("localisation: resume at j", "localise"),
                   ("diagnosis: resume, informed", "diagnose")]:
    s, c = run(mode)
    tab[name] = (s, c)
    print(f"{name:>32}{s:>12.1%}{c:>9.1f}{s / max(c, 1e-9):>11.3f}")

print()
print()
print("Localisation is a guess. Sweep how often it names the right step;")
print("a wrong guess sends the agent back to the beginning.")
print()
print(f"{'localisation accuracy':>23}{'localise':>11}{'diagnose':>11}"
      f"{'detect only':>14}")
print("-" * 59)
la = {}
det = run("detect")[0]
for a in (1.0, 0.85, 0.65, 0.45, 0.20):
    x = run("localise", loc_acc=a)[0]
    y = run("diagnose", loc_acc=a)[0]
    la[a] = (x, y)
    print(f"{a:>23.0%}{x:>11.1%}{y:>11.1%}{det:>14.1%}")

print()
print()
print("How the three scale with the budget -- the resource localisation saves.")
print()
print(f"{'budget':>8}{'detect':>10}{'localise':>11}{'diagnose':>11}")
print("-" * 40)
bd = {}
for b in (10, 16, 26, 40, 60):
    bd[b] = (run("detect", budget=b)[0], run("localise", budget=b)[0],
             run("diagnose", budget=b)[0])
    print(f"{b:>8}{bd[b][0]:>10.1%}{bd[b][1]:>11.1%}{bd[b][2]:>11.1%}")

print()
print()
print("And how it scales with task length, at a budget of 3x the task.")
print()
print(f"{'steps k':>9}{'detect':>10}{'localise':>11}{'gain':>9}")
print("-" * 39)
kl = {}
for k in (4, 8, 14, 22):
    a = run("detect", budget=3 * k, k=k)[0]
    b = run("localise", budget=3 * k, k=k)[0]
    kl[k] = (a, b)
    print(f"{k:>9}{a:>10.1%}{b:>11.1%}{b - a:>+9.1%}")

print(f"""
The first table is the ordering, and the size of the steps between rows is the
result.

Detection alone -- knowing that something went wrong and starting over -- takes
completion from {tab['nothing (one attempt)'][0]:.1%} to
{tab['detection: restart'][0]:.1%}. That is a large gain and it is the one most
systems have.

Localisation takes it to {tab['localisation: resume at j'][0]:.1%}, and it does so
while spending FEWER steps: {tab['localisation: resume at j'][1]:.1f} against
{tab['detection: restart'][1]:.1f}. Better outcome, cheaper.

Diagnosis -- additionally knowing why -- adds
{tab['diagnosis: resume, informed'][0] - tab['localisation: resume at j'][0]:+.1%}.

**Knowing WHERE is worth roughly thirteen times knowing WHY**, at these
parameters, and that ordering is the opposite of where engineering attention
usually goes. Teams write elaborate error taxonomies and reflection prompts to
extract the cause, on top of a retry that starts from the beginning.

The reason is arithmetic rather than psychology. A restart re-runs every step that
was already fine, so it pays the whole task's failure probability again -- and
those steps fail at the same rate they did the first time. Resuming from the last
good state pays only the remainder. Diagnosis improves ONE step's odds; localisation
removes all the steps before it from the retry.

The second table is the objection: localisation is a guess, and a wrong guess sends
you back to the start. So how good does the guess have to be?

At {1.0:.0%} accuracy localisation scores {la[1.0][0]:.1%}. At {0.2:.0%} -- a guess
that is wrong four times in five -- it still scores {la[0.2][0]:.1%}, against
detection-only's {det:.1%}.

**Localisation beats restarting even when it is usually wrong**, because a wrong
guess costs exactly what detection-only always costs, and a right guess saves a
great deal. The downside is bounded by the alternative and the upside is not, which
makes it a free option in the strict sense.

That is a strong practical statement: you do not need a reliable failure localiser
to benefit from having one, so the usual reason for not building it -- "we could
not identify the failing step reliably" -- is not a reason.

The third table shows what localisation is actually buying, which is budget. At a
budget of {16} steps -- twice the task length -- detection reaches
{bd[16][0]:.1%} and localisation {bd[16][1]:.1%}. Detection needs a budget of
{60} to reach {bd[60][0]:.1%}, which localisation reaches at {16}.

**Localisation converts a budget problem into a non-problem**, and budget is the
scarce resource in every chapter of this part. Restarting a partially-completed task
is the single largest source of wasted steps an agent has.

The fourth table is why this grows in importance rather than staying constant. The
gain from localisation is {kl[4][1] - kl[4][0]:+.1%} on a {4}-step task and
{kl[22][1] - kl[22][0]:+.1%} on a {22}-step one, where detection-only completes
{kl[22][0]:.1%} and localisation {kl[22][1]:.1%}.

The mechanism is the same exponent as everywhere else: a restart re-runs $k$ steps
and resumption re-runs the remainder, so the gap widens with $k$. **On long tasks,
the difference between restarting and resuming is most of the outcome.**

Which connects this chapter to two others and makes the recommendation concrete.

Localisation is what ch:ag-planning's checkpoints provide. A verified segment
boundary IS a known-good state to resume from, so a system with checkpoints has
localisation for free and a system without one cannot have it at any price. That is
a second, independent argument for checkpoints, arrived at from recovery rather
than from the success exponent.

Diagnosis is what ch:ag-tool-calling's error messages provide, and this listing
puts it in its place: worth having, worth much less than localisation, and worth
almost nothing if the retry restarts from the beginning anyway.

So the build order is: resume before you diagnose. Get a known-good state to return
to, and only then invest in working out what went wrong.""")

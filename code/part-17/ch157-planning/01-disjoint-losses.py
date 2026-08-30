# -*- coding: utf-8 -*-
# Extracted from: Chapter 157 — Planning and Plan-and-Execute
# Source: src/.../ch157-planning.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Is a better plan worth more than a more frequent one?

cite:liu2024agentbench identifies long-horizon reasoning as a primary obstacle for
agents across eight environments. A plan is long-horizon reasoning in its purest
form: a prediction, made before any observation, about a sequence of states the
agent has not seen.

So planning asks models to do the thing they are worst at, and then commits to the
answer. This listing separates the two things that could rescue that -- making the
plan better, and rewriting it more often -- and asks which is worth more
(eq:replan-rate-dominates-plan-quality).

The environment drifts: at each step there is some chance the world is not what
the plan assumed. Once that has happened, every subsequent planned step is
addressing a state that no longer exists, until somebody replans.
"""
import numpy as np

rng = np.random.default_rng(2003)

N = 60000
K = 12                  # steps in the task
DRIFT = 0.08            # chance per step that the state departs from the plan


def run(quality, replan_every, k=K, drift=DRIFT):
    """quality: chance a planned step is correct GIVEN the assumed state holds.
    replan_every: rewrite the plan every r steps. r=1 is fully reactive,
    r=k is plan-once-and-execute."""
    ok = np.ones(N, dtype=bool)
    stale = np.zeros(N, dtype=bool)      # the plan no longer matches reality
    calls = np.ones(N)                   # the initial plan
    for i in range(k):
        if i > 0 and i % replan_every == 0:
            stale[:] = False             # a rewrite re-syncs with the world
            calls += 1
        # A step succeeds if the plan is right AND the plan still applies.
        good = (rng.random(N) < quality) & ~stale
        ok &= good
        stale |= rng.random(N) < drift
        calls += 1
    return float(ok.mean()), float(calls.mean())


QUALITIES = [0.90, 0.94, 0.97, 0.99]
RATES = [12, 6, 4, 2, 1]

print(f"A {K}-step task. The world departs from the plan's assumptions")
print(f"{DRIFT:.0%} of the time per step. `replan every r` rewrites the plan")
print("every r steps; r=12 is plan-once, r=1 is fully reactive.")
print()
print(f"{'plan quality':>14}" + "".join(f"{'r=' + str(r):>10}" for r in RATES))
print("-" * 64)
tab = {}
for q in QUALITIES:
    row = {}
    for r in RATES:
        row[r] = run(q, r)
        tab[(q, r)] = row[r]
    print(f"{q:>14.0%}" + "".join(f"{row[r][0]:>10.1%}" for r in RATES))

print()
print()
print("The same grid, in model calls -- replanning is not free.")
print()
print(f"{'plan quality':>14}" + "".join(f"{'r=' + str(r):>10}" for r in RATES))
print("-" * 64)
for q in QUALITIES:
    print(f"{q:>14.0%}" + "".join(f"{tab[(q, r)][1]:>10.1f}" for r in RATES))

print()
print()
print("Two ways to spend, from a common baseline of quality 90%, r=12.")
print()
base_s, base_c = tab[(0.90, 12)]
print(f"{'change':>38}{'success':>11}{'gain':>9}{'calls':>9}")
print("-" * 67)
moves = [("baseline: quality 90%, plan once", (0.90, 12)),
         ("quality 90% -> 99% (plan once)", (0.99, 12)),
         ("keep 90%, replan every 4 steps", (0.90, 4)),
         ("keep 90%, replan every 2 steps", (0.90, 2)),
         ("keep 90%, replan every step", (0.90, 1)),
         ("quality 99% AND replan every 2", (0.99, 2))]
mv = {}
for name, key in moves:
    s, c = tab[key]
    mv[name] = (s, c)
    print(f"{name:>38}{s:>11.1%}{s - base_s:>+9.1%}{c:>9.1f}")

print()
print()
print("How much of the loss is drift, and how much is the plan being wrong?")
print("Hold one at zero and vary the other, at r=12 and r=2.")
print()
print(f"{'condition':>34}{'r=12':>10}{'r=2':>10}")
print("-" * 54)
iso = {}
for name, q, d in [("perfect plan, no drift", 1.0, 0.0),
                   ("perfect plan, drift 8%", 1.0, DRIFT),
                   ("quality 90%, no drift", 0.90, 0.0),
                   ("quality 90%, drift 8%", 0.90, DRIFT)]:
    a = run(q, 12, drift=d)[0]
    b = run(q, 2, drift=d)[0]
    iso[name] = (a, b)
    print(f"{name:>34}{a:>10.1%}{b:>10.1%}")

print()
print()
print("And how the answer moves with how volatile the environment is.")
print()
print(f"{'drift':>8}{'plan once':>12}{'replan /4':>12}{'replan /1':>12}"
      f"{'best':>12}")
print("-" * 56)
dr = {}
for d in (0.0, 0.02, 0.05, 0.10, 0.20):
    a = run(0.94, 12, drift=d)[0]
    b = run(0.94, 4, drift=d)[0]
    c = run(0.94, 1, drift=d)[0]
    dr[d] = (a, b, c)
    best = ["plan once", "replan /4", "replan /1"][int(np.argmax([a, b, c]))]
    print(f"{d:>8.0%}{a:>12.1%}{b:>12.1%}{c:>12.1%}{best:>12}")

print(f"""
The first table is the comparison, and it did not come out the way the chapter
was outlined to expect.

Along the top row -- plan quality {QUALITIES[0]:.0%} -- going from planning once
to replanning every step takes success from {tab[(0.90, 12)][0]:.1%} to
{tab[(0.90, 1)][0]:.1%}, a gain of {tab[(0.90, 1)][0] - tab[(0.90, 12)][0]:.1%}.

Down the left column -- plan once -- going from quality {QUALITIES[0]:.0%} to
{QUALITIES[-1]:.0%} takes it from {tab[(0.90, 12)][0]:.1%} to
{tab[(0.99, 12)][0]:.1%}, a gain of
{tab[(0.99, 12)][0] - tab[(0.90, 12)][0]:.1%}.

**Plan quality moves the number further than replanning does**, at these
parameters, and the intuition that a frequently-rewritten mediocre plan beats a
good one is simply wrong here. The interesting question is why, and the third
table answers it.

The second table prices replanning first, because it is cheap: every {4} steps
costs {tab[(0.90, 4)][1] - tab[(0.90, 12)][1]:.1f} extra calls on a {K}-step
task, every step costs {tab[(0.90, 1)][1] - tab[(0.90, 12)][1]:.1f}. Quality is
free in call terms and expensive in every other sense -- it is a better model, a
better prompt, or a better planner, none of which is a configuration change.

Note the last row: quality {0.99:.0%} AND replanning every {2} steps reaches
{tab[(0.99, 2)][0]:.1%}, against {tab[(0.99, 12)][0]:.1%} for quality alone and
{tab[(0.90, 2)][0]:.1%} for replanning alone. **The two compose**, which is the
first clue that they are not competing.

The third table is the one to take away, because it separates the two losses
instead of comparing them.

With a perfect plan and no drift, success is
{iso['perfect plan, no drift'][0]:.1%}. Introduce {DRIFT:.0%} drift and a perfect
plan falls to {iso['perfect plan, drift 8%'][0]:.1%}, recovering to
{iso['perfect plan, drift 8%'][1]:.1%} when replanned every {2} steps. That is
the DRIFT loss: the plan was correct when written and the world moved. Replanning
addresses it and plan quality cannot -- there is no plan skilful enough to predict
an unobserved change.

Now the other axis. Quality {0.9:.0%} with NO drift scores
{iso['quality 90%, no drift'][0]:.1%} at r={12} and
{iso['quality 90%, no drift'][1]:.1%} at r={2}. Replanning changes nothing,
because a rewrite re-syncs the plan with a world that never moved. That is the
QUALITY loss, and only a better planner touches it.

**The two interventions fix disjoint losses**, which is why they compose in the
second table and why neither dominates in general. Which one is worth more is
decided by which loss is bigger, and that is arithmetic you can do in advance:
the quality loss is $1 - q^{{k}}$ and the drift loss is governed by
$1 - (1-\delta)^{{r}}$ per planning segment. At {0.9:.0%} quality over {K} steps
the first is {1 - 0.9 ** K:.0%}; at {DRIFT:.0%} drift over {12} steps the second
is {1 - (1 - DRIFT) ** 12:.0%}. The quality term is larger, so quality wins --
and at {0.99:.0%} quality it would not be.

That is the chapter's actual finding, and it is more useful than a
recommendation: **compute the two losses before choosing which to attack**, and
expect the answer to flip as either parameter moves.

The fourth table sweeps the drift side to show the flip. At {0:.0%} drift,
planning once scores {dr[0.0][0]:.1%} against reactive's {dr[0.0][2]:.1%} --
identical, and the extra {tab[(0.94, 1)][1] - tab[(0.94, 12)][1]:.0f} calls
bought nothing whatsoever. At {0.2:.0%} drift it is {dr[0.2][0]:.1%} against
{dr[0.2][2]:.1%}.

That agrees with ch:ag-react's informativeness sweep from a different direction,
and the two together are the durable version of this part's architecture advice:
**the volatility of the environment decides how often to replan, and the
capability of the model decides how good the plan is, and neither substitutes for
the other.**

Two honest caveats on what this listing does not show.

It gives replanning a free, perfect re-sync: a rewrite always restores the plan to
match the world. A real replan is a fresh long-horizon prediction with the same
quality {QUALITIES[0]:.0%}--{QUALITIES[-1]:.0%} problem as the first one, and it
also needs to DETECT that drift occurred, which ch:ag-react measured as the hard
part. Both make the replanning column optimistic.

And it treats plan quality as a free parameter. cite:liu2024agentbench identifies
long-horizon reasoning as a primary agent bottleneck, which is precisely the
capability the quality column represents -- so the column that wins here is the
one that is hardest to move. **Planning's problem is not that better plans are not
worth having. It is that a plan is the most long-horizon thing an agent does, made
with the least information it will ever have.**

Which leaves the justification this listing cannot score, and it is not accuracy.
A plan is a STRUCTURE: inspectable before execution, checkable during it, and a
place for a human to intervene. ch:ag-loop needed exactly such a structure for a
completion condition that is not the agent's own judgement, and ch:ag-termination
will need one for a budget. **The plan earns its place as an artefact other
components can use, rather than as a prediction to be followed.**""")

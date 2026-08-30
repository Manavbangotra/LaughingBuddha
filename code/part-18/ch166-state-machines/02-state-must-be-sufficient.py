# -*- coding: utf-8 -*-
# Extracted from: Chapter 166 — State Machines, Events, and Durable Execution
# Source: src/.../ch166-state-machines.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""What has to be in the durable state for a resume to be correct.

A checkpoint that records the wrong things produces a resume that looks successful
and is not. This listing sweeps what is persisted and measures how often the
resumed run reaches the right answer (eq:state-must-be-sufficient).

The candidate fields come from part:17, which is not a coincidence -- they are the
same artefacts ch:ag-memory and ch:ag-planning said to build, and durability is
what makes them survive a restart:

  position   how far the run got
  outputs    what each completed step produced
  tried      which actions failed (ch:ag-loop's deduplication set)
  derived    values computed from several inputs (ch:ag-memory's scratchpad)
  goal       the original request, verbatim rather than summarised
"""
import numpy as np

rng = np.random.default_rng(3491)

M = 60000
STEPS = 10
P_CRASH = 0.10
P_STEP = 0.95
P_RECOMPUTE = 0.86      # re-deriving a lost derived value
P_REDISCOVER = 0.55     # re-learning which actions fail, per lost entry
P_REGOAL = 0.90         # reconstructing the goal from a summary

FIELDS = ["position", "outputs", "tried", "derived", "goal"]


def run(persisted, m=M, steps=STEPS, crash=P_CRASH):
    have = set(persisted)
    ok = np.ones(m, dtype=bool)
    pos = np.zeros(m, dtype=np.int64)
    work = np.zeros(m, dtype=np.int64)
    resumes = np.zeros(m, dtype=np.int64)
    for _ in range(steps * 6):
        live = ok & (pos < steps) & (work < steps * 5)
        idx = np.flatnonzero(live)
        if not len(idx):
            break
        work[idx] += 1
        p = np.full(len(idx), P_STEP)
        good = rng.random(len(idx)) < p
        pos[idx[good]] += 1
        crashed = rng.random(len(idx)) < crash
        c = idx[crashed]
        if len(c):
            resumes[c] += 1
            # Without position, the run restarts from zero.
            if "position" not in have:
                pos[c] = 0
            # Without outputs, completed work must be redone.
            elif "outputs" not in have:
                pos[c] = np.maximum(pos[c] - 2, 0)
            # Without the tried set, the resumed run re-explores dead ends.
            if "tried" not in have:
                ok[c] &= rng.random(len(c)) < P_REDISCOVER
            # Without derived values, they are recomputed, sometimes wrongly.
            if "derived" not in have:
                ok[c] &= rng.random(len(c)) < P_RECOMPUTE
            # Without the verbatim goal, the run continues toward a paraphrase.
            if "goal" not in have:
                ok[c] &= rng.random(len(c)) < P_REGOAL
    done = ok & (pos >= steps)
    return float(done.mean()), float(work.mean()), float(resumes.mean())


print(f"{M:,} runs, {STEPS} steps, {P_CRASH:.0%} crash rate per step.")
print("Fields are added to the durable state one at a time.")
print()
print(f"{'persisted state':>44}{'completed':>12}{'steps':>9}{'gain':>9}")
print("-" * 76)
cum = {}
have = []
r = run(have)
cum["(nothing durable)"] = r
print(f"{'(nothing durable)':>44}{r[0]:>12.1%}{r[1]:>9.1f}{'--':>9}")
prev = r[0]
for f in FIELDS:
    have.append(f)
    r = run(have)
    cum["+ " + f] = r
    print(f"{('+ ' + f):>44}{r[0]:>12.1%}{r[1]:>9.1f}{r[0] - prev:>+9.1%}")
    prev = r[0]

print()
print()
print("Each field REMOVED from a complete state -- what you lose by omitting it")
print("from a system that persists everything else.")
print()
print(f"{'field omitted':>44}{'completed':>12}{'loss':>10}")
print("-" * 68)
full = run(FIELDS)
drop = {}
for f in FIELDS:
    r = run([x for x in FIELDS if x != f])
    drop[f] = r[0]
    print(f"{f:>44}{r[0]:>12.1%}{r[0] - full[0]:>+10.1%}")

print()
print()
print("How the ranking changes with the crash rate, since a rare crash makes")
print("every field look unnecessary.")
print()
print(f"{'crash rate':>12}{'nothing':>11}{'position only':>16}"
      f"{'position+outputs':>19}{'everything':>13}")
print("-" * 71)
cr = {}
for c in (0.02, 0.10, 0.25, 0.45):
    row = (run([], crash=c)[0], run(["position"], crash=c)[0],
           run(["position", "outputs"], crash=c)[0],
           run(FIELDS, crash=c)[0])
    cr[c] = row
    print(f"{c:>12.0%}{row[0]:>11.1%}{row[1]:>16.1%}{row[2]:>19.1%}"
          f"{row[3]:>13.1%}")

print()
print()
print("And the cost side: what persisting everything saves in re-executed work.")
print()
print(f"{'persisted state':>28}{'completed':>12}{'steps used':>13}"
       f"{'resumes':>10}")
print("-" * 63)
for name, fields in [("nothing", []), ("position only", ["position"]),
                     ("position + outputs", ["position", "outputs"]),
                     ("everything", FIELDS)]:
    r = run(fields)
    print(f"{name:>28}{r[0]:>12.1%}{r[1]:>13.1f}{r[2]:>10.2f}")

print(f"""
The first table adds durable fields one at a time and the totals are unremarkable
until you compare them with the second, which is where the finding is.

Removing the TRIED set -- ch:ag-loop's record of which actions already failed --
from a state that persists everything else costs
{drop['tried'] - full[0]:+.1%}. That is by far the largest single loss, and it is
the field no workflow engine persists.

The reason is ch:ag-loop's, transplanted. A resumed run with no memory of what
failed re-derives the same wrong approach the crashed run had already eliminated,
and it does so with a fresh context that makes the wrong approach look new. **A
resume without the failure set is a run that has forgotten why it was going the
way it was going** (eq:state-must-be-sufficient).

Derived values cost {drop['derived'] - full[0]:+.1%} and the verbatim goal
{drop['goal'] - full[0]:+.1%}. Both are ch:ag-memory's mechanisms needing to
survive a restart, and both are cheap to write and easy to omit.

Position costs {drop['position'] - full[0]:+.1%}, which is much less than its
prominence suggests -- and OUTPUTS costs {drop['outputs'] - full[0]:+.1%}, which is
zero. That second number is an artefact of this model rather than a finding: here
outputs only matter through their effect on how far back a resume must go, which
position already captures. In a real system outputs are what makes the resumed run
able to continue at all, and the listing understates them.

**The ordering to take away is that the fields workflow engines persist by default
-- position and outputs -- are not the ones that decide whether a resume produces
the right answer.** Position is what the engine needs to know where to restart.
The tried set, the derived values and the goal are what the AGENT needs to be the
same agent it was.

The third table shows the ranking depending on the crash rate, which is the reason
this is easy to miss. At {0.02:.0%} crashes, persisting nothing gives {cr[0.02][0]:.1%}
and persisting everything {cr[0.02][3]:.1%} -- a gap of
{cr[0.02][3] - cr[0.02][0]:.1%}, noticeable but survivable. At {0.45:.0%} it is
{cr[0.45][0]:.1%} against {cr[0.45][3]:.1%}.

**A durable-state design validated in a low-crash environment is untested**, and
the failure appears exactly when the environment degrades -- which is when
durability was supposed to help.

The last table prices it. Persisting everything uses {run(FIELDS)[1]:.1f} steps
against {run([])[1]:.1f} for persisting nothing, and completes
{run(FIELDS)[0] - run([])[0]:+.1%} more. The extra steps are not overhead; they are
runs that got far enough to need them, and the "nothing" row is cheap because most
of its runs died early.

**Cost comparisons between durability designs have to be conditioned on
completion**, or the broken design looks efficient.""")

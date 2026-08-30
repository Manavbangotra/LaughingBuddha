# -*- coding: utf-8 -*-
# Extracted from: Chapter 205 — The ML Lifecycle
# Source: src/.../ch205-lifecycle.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The lifecycle is a loop, and the loop's period is what determines throughput.

Diagrams of the ML lifecycle are drawn as a cycle and reasoned about as a line: scope,
build, deploy, monitor, repeat. That framing makes the natural optimisation "make each
stage faster", which is what a team does when it is told to ship more often.

But a loop's throughput is set by its PERIOD -- the time to get all the way round -- and
the period is dominated by whichever stage has the longest wait, not the longest work
(eq:lifecycle-period-is-wait-not-work).

This listing separates work from wait across the lifecycle and finds that the stages
consuming the most calendar time are not the ones consuming the most effort.
"""
# (stage, engineer-hours of work, calendar hours of wait, who waits on whom)
STAGES = [
    ("scope and design",       22.0,    4.0,  "nobody"),
    ("data preparation",       36.0,   12.0,  "access approvals"),
    ("build and iterate",      58.0,    2.0,  "nobody"),
    ("offline evaluation",     14.0,   30.0,  "eval-set labelling"),
    ("review and approval",     3.0,  110.0,  "a person with a calendar"),
    ("deploy to staging",       6.0,    9.0,  "environment queue"),
    ("shadow and canary",       4.0,  168.0,  "statistical significance"),
    ("full rollout",            2.0,   16.0,  "staged ramp"),
    ("monitor and attribute",  11.0,  340.0,  "a signal that takes weeks"),
]

total_work = sum(s[1] for s in STAGES)
total_wait = sum(s[2] for s in STAGES)
period = total_work + total_wait

print("One trip round the lifecycle, separating effort from elapsed time.")
print()
print(f"{'stage':>24}{'work hrs':>11}{'wait hrs':>11}{'elapsed':>10}"
      f"{'work share':>13}{'wait share':>13}")
print("-" * 84)
tab = {}
for name, work, wait, why in STAGES:
    tab[name] = (work, wait, work + wait)
    print(f"{name:>24}{work:>11.0f}{wait:>11.0f}{work + wait:>10.0f}"
          f"{work / total_work:>13.0%}{wait / total_wait:>13.0%}")
print("-" * 84)
print(f"{'TOTAL':>24}{total_work:>11.0f}{total_wait:>11.0f}{period:>10.0f}")
print()
print(f"period: {period:.0f} hours = {period / 24.0:.1f} days")
print(f"work is {total_work / period:.0%} of it")

print()
print()
print("Ranked by calendar time, which is what sets how often you can go round.")
print()
order = sorted(STAGES, key=lambda s: -(s[2]))
print(f"{'rank':>6}{'stage':>24}{'wait hrs':>11}{'share of wait':>16}"
      f"{'waiting on':>28}")
print("-" * 86)
for i, (name, work, wait, why) in enumerate(order, 1):
    print(f"{i:>6}{name:>24}{wait:>11.0f}{wait / total_wait:>16.0%}{why:>28}")

print()
print()
print("What halving each stage's WORK buys, against halving its WAIT.")
print()
print(f"{'stage':>24}{'halve work':>13}{'halve wait':>13}"
      f"{'wait is worth':>16}")
print("-" * 68)
lever = {}
for name, work, wait, why in STAGES:
    w_gain = work / 2.0
    t_gain = wait / 2.0
    lever[name] = (w_gain, t_gain)
    ratio = (t_gain / w_gain) if w_gain > 0 else float("inf")
    print(f"{name:>24}{w_gain:>12.0f}h{t_gain:>12.0f}h"
          f"{ratio:>15.1f}x")

print()
print()
print("The three biggest interventions, priced against each other.")
print()
INTERVENTIONS = [
    ("hire another engineer (work -25%)",      "work", 0.25),
    ("automate the build pipeline (work -40%)", "work", 0.40),
    ("pre-approve low-risk changes",           "approval", 0.85),
    ("shrink the canary window",               "canary", 0.60),
    ("faster attribution signal",              "monitor", 0.70),
]
print(f"{'intervention':>36}{'hours saved':>14}{'new period':>13}"
      f"{'trips per year':>17}")
print("-" * 82)
base_trips = 365.0 * 24.0 / period
res = {}
for label, kind, frac in INTERVENTIONS:
    saved = 0.0
    if kind == "work":
        saved = total_work * frac
    elif kind == "approval":
        saved = tab["review and approval"][1] * frac
    elif kind == "canary":
        saved = tab["shadow and canary"][1] * frac
    elif kind == "monitor":
        saved = tab["monitor and attribute"][1] * frac
    newp = period - saved
    res[label] = (saved, newp, 365.0 * 24.0 / newp)
    print(f"{label:>36}{saved:>14.0f}{newp:>13.0f}{365.0 * 24.0 / newp:>17.1f}")
print()
print(f"baseline: {base_trips:.1f} trips per year")

print()
print()
print("And what the period costs, because a slow loop is not merely slow.")
print("A regression discovered on trip N was introduced on trip N-1.")
print()
print(f"{'period days':>13}{'trips/year':>12}{'mean age of a live bug':>25}"
      f"{'changes in flight':>19}")
print("-" * 72)
CHANGE_RATE = 3.0        # changes a team wants to make per week
age = {}
for p_days in (period / 24.0, 21.0, 14.0, 7.0, 3.0):
    trips = 365.0 / p_days
    mean_age = p_days / 2.0 + p_days     # half a period to introduce, one to detect
    inflight = CHANGE_RATE * p_days / 7.0
    age[round(p_days, 1)] = (trips, mean_age, inflight)
    print(f"{p_days:>13.1f}{trips:>12.1f}{mean_age:>25.1f}{inflight:>19.1f}")

print(f"""
The work-versus-wait split is the first thing to look at, and it is stark. Of a
{period:.0f}-hour lifecycle period, **{total_work:.0f} hours are work and
{total_wait:.0f} hours are waiting** -- work is
{total_work / period:.0%} of the elapsed time
(eq:lifecycle-period-is-wait-not-work).

That number is why the natural optimisation misfires. A team told to ship faster
optimises what it can see itself doing: building, iterating, writing code. Those stages
are real effort and they are {total_work / period:.0%} of the calendar.

The ranked table shows where the calendar actually goes.
`{order[0][0]}` is {order[0][2] / total_wait:.0%} of all waiting,
`{order[1][0]}` is {order[1][2] / total_wait:.0%}, and
`{order[2][0]}` is {order[2][2] / total_wait:.0%}. Together those three are
{(order[0][2] + order[1][2] + order[2][2]) / total_wait:.0%} of the wait and
{(order[0][1] + order[1][1] + order[2][1]) / total_work:.0%} of the work.

**The stages that consume the calendar consume almost none of the effort**, which is
exactly why they are invisible in any accounting based on where people spend their time.
Nobody is working during them. That is the point.

The lever table prices the two kinds of intervention per stage. For
`{order[0][0]}`, halving the wait saves {lever[order[0][0]][1]:.0f} hours against
{lever[order[0][0]][0]:.0f} for halving the work --
{lever[order[0][0]][1] / max(lever[order[0][0]][0], 0.01):.0f} times more.

The intervention table makes the comparison concrete against realistic options. Hiring
an engineer and cutting total work by a quarter saves
{res['hire another engineer (work -25%)'][0]:.0f} hours and takes the loop from
{base_trips:.1f} to {res['hire another engineer (work -25%)'][2]:.1f} trips a year.
Pre-approving low-risk changes saves
{res['pre-approve low-risk changes'][0]:.0f} hours and reaches
{res['pre-approve low-risk changes'][2]:.1f}. A faster attribution signal saves
{res['faster attribution signal'][0]:.0f} and reaches
{res['faster attribution signal'][2]:.1f}.

**The two best interventions are a policy change and a measurement change**, and neither
is engineering capacity. That is an uncomfortable result for a team whose instinct under
schedule pressure is to add people, and it is the practical form of
cite:paleyes2020deployment's finding that the obstacles are distributed across the whole
workflow rather than concentrated in modelling.

The last table is why the period matters beyond throughput. A regression introduced on
one trip is detected on the next, so **the mean age of a live defect is about one and a
half periods** -- {age[round(period / 24.0, 1)][1]:.0f} days at the baseline. And with a
team wanting {CHANGE_RATE:.0f} changes a week, {age[round(period / 24.0, 1)][2]:.0f}
changes are in flight simultaneously at any moment.

That second number is the one that compounds. Changes in flight are changes whose
effects overlap, and ch:sd-routing-caching's attribution problem arrives here: when a
metric moves and eleven changes are outstanding, **the period has destroyed your ability
to attribute** -- not merely slowed you down.

Cutting the period from {period / 24.0:.0f} days to {7.0:.0f} takes changes in flight
from {age[round(period / 24.0, 1)][2]:.0f} to {age[7.0][2]:.0f} and mean defect age from
{age[round(period / 24.0, 1)][1]:.0f} days to {age[7.0][1]:.0f}. **A short loop is a
diagnostic instrument**, and that is the argument for shortening it that survives when
someone asks why shipping faster is worth the risk.""")

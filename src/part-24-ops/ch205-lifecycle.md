---
id: ops-lifecycle
number: 205
part: XXIV
tier: full
status: draft
requires: [semantic-failure-has-no-instrument, semantic-breaker-is-affordable,
           detection-time-sets-the-blast-radius, tail-attribution-differs-from-mean]
provides: [lifecycle-period-is-wait-not-work, period-destroys-attribution,
           rework-cost-is-set-by-detection-lateness, shift-to-shorter-return-not-earlier]
citations: [paleyes2020deployment, sculley2015, breck2017, cemri2025mast]
---

## 1. Learning Objectives

By the end of this chapter you will be able to separate work from wait across a
lifecycle and show which stages consume the calendar; explain why the stages that
dominate elapsed time consume almost no effort, and why that makes them invisible to
time-tracking; compute the number of changes in flight from the loop period and say why a
long period destroys attribution rather than merely slowing delivery; calculate a stage's
true cost as work times expected visits under rework; and state the correction to "shift
left" — that what matters is not how early a detector fires but how far back it sends the
work.

## 2. Why This Matters

{{part:23}} finished making the system fast. This part is about operating it over months,
and the first thing to establish is where the time actually goes — because the answer
determines what every subsequent chapter is optimising.

{{sec:9-practical-example}} separates effort from elapsed time across one trip round a
realistic lifecycle and finds a **847-hour period of which 156 hours are work**. Work is
**18%** of the calendar ({{eq:lifecycle-period-is-wait-not-work}}). The three
longest-waiting stages — monitoring, canary, and approval — are **89%** of the waiting
and **12%** of the effort.

**Nobody is working during the stages that consume the calendar.** That is precisely why
they are invisible to any accounting based on where people spend their time, and why a
team told to ship faster optimises the 18%.

The second half concerns rework, and it contradicts a piece of standard advice. Expected
effort including rework is **2.25×** a single clean pass
({{eq:rework-cost-is-set-by-detection-lateness}}) — and moving detection *earlier*, from
canary into offline evaluation, changes it by **nothing**, because both send the work
back to the same expensive stage ({{eq:shift-to-shorter-return-not-earlier}}).

## 3. Prerequisites

You need {{eq:semantic-failure-has-no-instrument}} from {{ch:sd-architecture}} — the
reason `monitor and attribute` takes weeks is that the signal it waits for does not exist
by default.

{{eq:semantic-breaker-is-affordable}} from {{ch:sd-fault-tolerance}} priced the
instrument that would shorten it, and this chapter supplies a second, independent
argument for building it.

{{eq:detection-time-sets-the-blast-radius}} is the same quantity from the damage side;
here it appears from the effort side.

{{eq:tail-attribution-differs-from-mean}} from {{ch:sd-latency}} is the method:
rank interventions by measured effect per unit of work, not by which number is largest.

## 4. Intuitive Explanation

Every ML lifecycle diagram is drawn as a circle: scope, build, evaluate, deploy, monitor,
back to scope. Everyone nods. And then everyone reasons about it as though it were a
line, because a line is what you can put on a schedule.

The difference matters, and it matters in a specific way. A line has a length; a loop has
a *period*. If you want to ship more often, you are not shortening a line — you are
raising the frequency of a loop, and a loop's frequency is set by how long it takes to
get all the way round.

Now here is the part that misdirects effort. Ask a team where the time goes and they will
tell you honestly: building, iterating, preparing data. That is where they are. It is
real work and it is skilled work.

It is also about a fifth of the calendar.

The rest is waiting. Waiting for a review from someone whose calendar is full. Waiting
for a canary to accumulate enough traffic to be statistically meaningful. Waiting for a
monitoring signal that, because it measures something semantic, takes weeks to move
enough to be believed. Nobody is working during any of that — which is exactly why it
does not show up when you ask where the time goes.

So the natural optimisation is to make the work faster, and the natural optimisation is
aimed at the smaller number.

There is a second consequence of a long period that matters more than throughput. If a
change takes five weeks to get round the loop, and your team wants to make three changes
a week, then at any moment there are about fifteen changes in flight — deployed,
undeployed, being canaried, being monitored. When a metric moves, which of the fifteen
caused it?

**A long loop does not merely slow you down. It destroys your ability to attribute.**
That is a stronger argument for shortening it than velocity, and it is the one that
survives a conversation about risk.

The second half of the chapter is about what happens when things go wrong, which they do
constantly. A problem found in evaluation sends you back to building. A problem found in
canary sends you back further. Every return re-does everything in between, so a stage's
real cost is not what it takes once — it is what it takes times how many times you end up
there.

That produces a counterintuitive accounting: the build stage in
{{sec:9-practical-example}} consumes **43%** of all effort and has a rework probability of
*zero*. Nothing goes wrong there. It is simply where everything gets sent back to.

Which leads to the advice everyone knows — shift left, find problems earlier — and to the
finding that it does not work here. Moving defects from canary detection into evaluation
detection changes nothing, because evaluation also sends you back to build. You have
found the problem sooner and you re-do the same expensive work.

What helps is finding problems at a stage whose *return trip* is short. Catching a data
problem during data preparation costs you data preparation. That is a different
instruction from "earlier", and in a looping process the two come apart.

## 5. Formal Explanation

**Period.** Let stage $i$ require work $w_i$ and wait $\tau_i$. The period of one trip is

$$ P \;=\; \sum_i (w_i + \tau_i), \qquad \text{work share} \;=\; \frac{\sum_i w_i}{P} $$ (eq:lifecycle-period-is-wait-not-work)

Since $\tau_i$ is elapsed time during which no effort is expended, it is invisible to
effort-based accounting. **Reducing $w_i$ reduces $P$ by at most the work share**, which
{{sec:9-practical-example}} measures at 18%.

Trips per year are $8760/P$, and with a change arrival rate $\lambda$ the number of
changes in flight is $\lambda P$. A defect introduced uniformly within a period and
detected at the end of the next has mean age $1.5P$:

$$ \text{changes in flight} = \lambda P, \qquad \mathbb{E}[\text{defect age}] = 1.5P $$ (eq:period-destroys-attribution)

**Attribution difficulty scales with $\lambda P$**, not with $P$ alone, so a fast-moving
team suffers more from a long period than a slow one does.

**Rework.** Model the lifecycle as a Markov chain over stages. At stage $i$ a problem is
found with probability $p_i$, sending the process back $b_i$ stages; otherwise it
advances. Let $v_i$ be the expected number of visits to stage $i$ before one clean pass.
Total expected effort is

$$ W \;=\; \sum_i v_i w_i \;\ge\; \sum_i w_i $$ (eq:rework-cost-is-set-by-detection-lateness)

with the inequality strict whenever any $p_i > 0$. The visit counts satisfy a linear
system determined by the $(p_i, b_i)$ pairs, and crucially $v_i$ depends on the rework
parameters of stages **downstream** of $i$ — so a stage's cost is set by what happens
after it.

The expected rework caused by detection at stage $i$ is

$$ R_i \;=\; p_i \sum_{j=i-b_i}^{i} w_j $$

which factors into a detection rate and a **return-trip cost**. Two detectors with the
same $p_i$ but different $b_i$ have different costs, and two with the same $b_i$ landing
on different stages also differ, because the summed work differs.

## 6. Mathematical Foundation

The correction to "shift left" follows directly from that factorisation.

Moving a defect from detection at stage $i$ to detection at stage $k < i$ changes the
expected rework from $p\sum_{j=i-b_i}^{i} w_j$ to $p\sum_{j=k-b_k}^{k} w_j$. The change
is beneficial only if

$$ \sum_{j=k-b_k}^{k} w_j \;<\; \sum_{j=i-b_i}^{i} w_j $$ (eq:shift-to-shorter-return-not-earlier)

**which is a statement about return-trip cost, not about position.** Detecting earlier
helps if and only if the earlier detector's return trip is cheaper, and in a lifecycle
where several detectors all return to the same expensive stage, they are not.

{{sec:9-practical-example}} measures exactly that degenerate case: offline evaluation
returns 2 stages costing 72 hours and canary returns 4 stages costing 79 — nearly
identical, because both land on `build and iterate` which dominates the sum. Shifting
between them is worth **nothing** (1.03× and 0.99×).

The stages worth shifting *to* are those with $b_i = 0$, which re-do only themselves.
Data preparation costs 36 hours to re-do against the 108 of a build-return, and moving
defects there is worth **23%** of total effort.

There is a second-order effect worth noting. Raising $p_k$ at an early stage increases
$v_k$, and if $w_k$ is large that partially offsets the gain — which is why moving
defects into offline evaluation, whose own visits are already high, does not help even
before considering the return trip. **The intervention must satisfy both conditions:
short return trip and low own-cost.**

## 7. Internal Mechanics

**Why `monitor and attribute` waits weeks.** The signal it is waiting for is a semantic
error rate, and {{ch:sd-fault-tolerance}} showed detection time scales with the inverse
square of the effect size. A regression that is not enormous takes weeks to distinguish
from noise at typical sampling rates. **The wait is a statistical property of the
measurement, not a queue** — which is why it does not respond to prioritisation, and why
the fix is a better instrument rather than more urgency.

**Why the canary wait is irreducible for the same reason.** Statistical significance on a
semantic signal needs samples, and samples arrive at traffic rate.
{{ch:sd-fault-tolerance}}'s arithmetic applies unchanged, and
{{ch:ops-deployment}} takes up how to size the window against blast radius.

**Why approval waits are calendar-shaped.** They depend on a person's availability rather
than on any property of the work, which makes them the one long wait that responds to a
policy change rather than an engineering one. Pre-approving a class of low-risk changes
removes it entirely for that class, and {{sec:9-practical-example}} prices it at 94 hours.

**Where {{cite:sculley2015}}'s debt fits.** Configuration debt, glue code, and pipeline
jungles all raise $w_i$ for the build and data stages — the stages whose visit counts are
highest. So technical debt is not merely a tax on each visit; it is a tax multiplied by
the rework factor, which {{sec:9-practical-example}} puts at **2.25×**. That
multiplication is why debt in a high-rework pipeline compounds faster than the same debt
elsewhere.

**Why the two listings disagree about the rework factor.** The first listing's period
counts one trip; the second finds expected effort is 2.08 times a clean pass. Those are
consistent and they measure different things -- the period is calendar time for a trip
that completes, and the rework factor is effort across all trips including the ones that
send you back. A team planning on the period alone will under-resource by roughly the
rework factor, and a team planning on effort alone will promise a date the loop cannot
meet. **Both numbers are needed and they answer different questions**, which is why
schedules built from one of them are reliably wrong in one direction.

**Why the obstacles are distributed.** {{cite:paleyes2020deployment}}'s survey of case
studies found practitioners hitting problems at every stage rather than one, which is the
qualitative form of this chapter's tables. A single-bottleneck model of the lifecycle
predicts the wrong intervention, and the evidence has never supported one.

**{{cite:breck2017}}'s rubric as a return-trip shortener.** Production-readiness tests
that run early — data schema checks, feature distribution checks, model staleness checks
— are precisely detectors with $b_i = 0$ or small. Read through
{{eq:shift-to-shorter-return-not-earlier}}, the rubric's value is not that it catches
problems but *where* it catches them.

## 8. Implementation

The first listing separates work from wait and prices the interventions.

```python {tier=A name=ea1}
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
```

## 9. Practical Example

One trip round the lifecycle:

```
                   stage   work hrs   wait hrs   elapsed   work share   wait share
------------------------------------------------------------------------------------
        scope and design         22          4        26          14%           1%
        data preparation         36         12        48          23%           2%
       build and iterate         58          2        60          37%           0%
      offline evaluation         14         30        44           9%           4%
     review and approval          3        110       113           2%          16%
       deploy to staging          6          9        15           4%           1%
       shadow and canary          4        168       172           3%          24%
            full rollout          2         16        18           1%           2%
   monitor and attribute         11        340       351           7%          49%
------------------------------------------------------------------------------------
                   TOTAL        156        691       847
```

**A 847-hour period of which work is 18%**
({{eq:lifecycle-period-is-wait-not-work}}). Ranked by wait:

```
  rank                   stage   wait hrs   share of wait                  waiting on
--------------------------------------------------------------------------------------
     1   monitor and attribute        340             49%   a signal that takes weeks
     2       shadow and canary        168             24%    statistical significance
     3     review and approval        110             16%    a person with a calendar
```

Those three are **89%** of the waiting and **12%** of the effort. Nobody is working
during them.

The interventions, priced:

```
                        intervention   hours saved   new period   trips per year
----------------------------------------------------------------------------------
   hire another engineer (work -25%)            39          808             10.8
automate the build pipeline (work -40%)            62          785             11.2
        pre-approve low-risk changes            94          754             11.6
            shrink the canary window           101          746             11.7
           faster attribution signal           238          609             14.4
```

**A faster attribution signal beats hiring an engineer by six times**, and the two best
interventions are a policy change and a measurement change rather than engineering
capacity.

```mermaid {#fig:period caption="The lifecycle is a loop whose period is dominated by waiting. Effort-based accounting sees only the work, which is 18% of the calendar, so the natural optimisation is aimed at the smaller term."}
flowchart LR
  A["work: 156h<br/>18% of period"] --> C["period 847h"]
  B["wait: 691h<br/>82% of period"] --> C
  B --> D["monitor 340h<br/>canary 168h<br/>approval 110h"]
  C --> E["10.3 trips/year"]
  C --> F["15 changes in flight"]
```

And what the period costs beyond throughput:

```
  period days  trips/year   mean age of a live bug  changes in flight
------------------------------------------------------------------------
         35.3        10.3                     52.9               15.1
         21.0        17.4                     31.5                9.0
         14.0        26.1                     21.0                6.0
          7.0        52.1                     10.5                3.0
          3.0       121.7                      4.5                1.3
```

At the baseline, **15 changes are in flight simultaneously** and a live defect is
**53 days old** on average ({{eq:period-destroys-attribution}}). When a metric moves,
fifteen candidates moved with it. **A short loop is a diagnostic instrument.**

The second listing turns to rework.

```python {tier=A name=ea2}
"""Rework is where the effort goes, and it is priced by how late you find things.

The previous listing counted one trip round the lifecycle. Real projects do not make one
trip: a problem found in evaluation sends you back to building, a problem found in
canary sends you back further, and each return re-does everything in between.

So a stage's true cost is its per-visit cost times how many times it is VISITED, and
the visit count is driven by the rework probabilities of every stage downstream of it
(eq:rework-cost-is-set-by-detection-lateness).

This listing computes the expected visits and finds that the cheapest change to make is
not doing less work -- it is finding problems earlier.
"""
# (stage, work hours per visit, P(a problem is found here), how far back it sends you)
STAGES = [
    ("scope and design",       22.0, 0.00, 0),
    ("data preparation",       36.0, 0.10, 0),
    ("build and iterate",      58.0, 0.00, 0),
    ("offline evaluation",     14.0, 0.34, 1),   # back to build
    ("review and approval",     3.0, 0.12, 2),   # back to build
    ("shadow and canary",       4.0, 0.22, 3),   # back to build
    ("full rollout",            2.0, 0.05, 6),   # back to scope
    ("monitor and attribute",  11.0, 0.18, 7),   # back to scope
]
N = len(STAGES)
NAMES = [s[0] for s in STAGES]


def expected_visits(stages, trials=200000, seed=20260829):
    """Simulate trips until one completes without rework; count stage visits."""
    import random
    rng = random.Random(seed)
    visits = [0.0] * N
    completed = 0
    runs = 0
    while completed < trials:
        i = 0
        guard = 0
        while i < N and guard < 500:
            visits[i] += 1
            name, work, p_fail, back = stages[i]
            if p_fail > 0 and rng.random() < p_fail:
                i = max(0, i - back)
            else:
                i += 1
            guard += 1
        completed += 1
        runs += 1
    return [v / runs for v in visits]


vis = expected_visits(STAGES)
print("Expected visits to each stage before one clean pass, and what that")
print("does to effort.")
print()
print(f"{'stage':>24}{'work/visit':>12}{'P(rework)':>12}{'visits':>9}"
      f"{'total work':>13}{'share':>9}")
print("-" * 80)
tot = sum(vis[i] * STAGES[i][1] for i in range(N))
naive = sum(s[1] for s in STAGES)
tab = {}
for i, (name, work, p, back) in enumerate(STAGES):
    tw = vis[i] * work
    tab[name] = (work, p, vis[i], tw)
    print(f"{name:>24}{work:>12.0f}{p:>12.0%}{vis[i]:>9.2f}"
          f"{tw:>13.0f}{tw / tot:>9.0%}")
print("-" * 80)
print(f"{'TOTAL':>24}{naive:>12.0f}{'':>12}{'':>9}{tot:>13.0f}")
print()
print(f"one clean pass would be {naive:.0f} hours; expected is {tot:.0f} "
      f"({tot / naive:.2f}x)")

print()
print()
print("Where the rework comes FROM, as opposed to where it is paid.")
print("A stage that detects a problem sends the work back; the cost lands upstream.")
print()
print(f"{'detected at':>24}{'P':>7}{'sends back':>12}{'stages re-done':>17}"
      f"{'hours re-done':>16}")
print("-" * 78)
cause = {}
for i, (name, work, p, back) in enumerate(STAGES):
    if p == 0:
        continue
    start = max(0, i - back)
    redone = sum(STAGES[j][1] for j in range(start, i + 1))
    cause[name] = (p, back, i - start + 1, redone, p * redone)
    print(f"{name:>24}{p:>7.0%}{back:>12}{i - start + 1:>17}{redone:>16.0f}")

print()
print()
print("Expected rework cost per trip, by where the problem is detected.")
print("This is P(detected here) times what has to be re-done.")
print()
order = sorted(cause, key=lambda k: -cause[k][4])
print(f"{'rank':>6}{'detected at':>24}{'expected rework hrs':>22}"
      f"{'share of rework':>18}")
print("-" * 72)
tot_rework = sum(cause[k][4] for k in cause)
for i, k in enumerate(order, 1):
    print(f"{i:>6}{k:>24}{cause[k][4]:>22.1f}{cause[k][4] / tot_rework:>18.0%}")

print()
print()
print("What moving detection EARLIER buys. Same total defect rate, discovered")
print("sooner -- so the same problems cost less because less is re-done.")
print()
SHIFTS = [
    ("as built",                                 {}),
    ("canary defects moved to eval",             {"shadow and canary": 0.11,
                                                  "offline evaluation": 0.45}),
    ("monitor defects moved to canary",          {"monitor and attribute": 0.09,
                                                  "shadow and canary": 0.31}),
    ("eval defects moved to data preparation",   {"offline evaluation": 0.19,
                                                  "data preparation": 0.25}),
    ("all defects moved to data preparation",    {"offline evaluation": 0.10,
                                                  "shadow and canary": 0.06,
                                                  "monitor and attribute": 0.05,
                                                  "data preparation": 0.53}),
]
print(f"{'scenario':>42}{'sends back':>12}{'expected hrs':>14}"
      f"{'vs as-built':>14}")
print("-" * 82)
shifted = {}
for label, changes in SHIFTS:
    st = []
    for name, work, p, back in STAGES:
        st.append((name, work, changes.get(name, p), back))
    v = expected_visits(st, trials=60000)
    t = sum(v[i] * st[i][1] for i in range(N))
    shifted[label] = t
    moved_to = ("-" if not changes else
                max(changes, key=lambda k: changes[k] -
                    dict((n, p) for n, w, p, b in STAGES)[k]))
    back_of = dict((n, b) for n, w, p, b in STAGES)
    print(f"{label:>42}{(str(back_of[moved_to]) if moved_to != '-' else '-'):>12}"
          f"{t:>14.0f}{t / shifted['as built']:>13.2f}x")

print()
print()
print("And what reducing the defect rate buys instead, for comparison.")
print()
print(f"{'defect rates scaled by':>26}{'expected hrs':>15}{'vs as-built':>14}")
print("-" * 56)
scaled = {}
for f in (1.0, 0.8, 0.6, 0.4, 0.2):
    st = [(n, w, p * f, b) for n, w, p, b in STAGES]
    v = expected_visits(st, trials=60000)
    t = sum(v[i] * st[i][1] for i in range(N))
    scaled[f] = t
    print(f"{f:>26.1f}{t:>15.0f}{t / scaled[1.0]:>13.2f}x")

print(f"""
The visits table is the correction to any plan built on a single pass. One clean trip
through this lifecycle is {naive:.0f} hours of work. The expected cost, accounting for
rework, is **{tot:.0f} hours -- {tot / naive:.2f} times more**
(eq:rework-cost-is-set-by-detection-lateness).

Look at where that lands. `build and iterate` costs {tab['build and iterate'][0]:.0f}
hours per visit and is visited {tab['build and iterate'][2]:.2f} times, so it consumes
{tab['build and iterate'][3]:.0f} hours -- {tab['build and iterate'][3] / tot:.0%} of
total effort.

**And it has a rework probability of zero.** Nothing goes wrong in the build stage; it is
simply where everything gets sent back to. A stage's cost is determined by what happens
*downstream* of it, which means the team spending most of its time there is not the team
causing the expense.

The cause table separates the two. `{order[0]}` detects
{cause[order[0]][0]:.0%} of problems and sends work back {cause[order[0]][1]} stages,
re-doing {cause[order[0]][3]:.0f} hours each time.
`{order[1]}` detects {cause[order[1]][0]:.0%} and re-does
{cause[order[1]][3]:.0f}.

Ranked by expected rework, `{order[0]}` is {cause[order[0]][4] / tot_rework:.0%} of it
and `{order[1]}` is {cause[order[1]][4] / tot_rework:.0%}.

**The expensive detector is not the one that finds the most problems.** It is the one
that finds them furthest from where they were introduced, and the two rank differently.

The shift table is where the standard advice fails, and the failure is instructive.

"Shift left" says find problems earlier. Moving half the canary-detected defects into
offline evaluation does exactly that -- same problems, found three stages sooner -- and
expected effort goes from {shifted['as built']:.0f} hours to
{shifted['canary defects moved to eval']:.0f}. **It got slightly worse.**

Moving monitor-detected defects into canary reaches
{shifted['monitor defects moved to canary']:.0f}, essentially unchanged.

The reason is in the middle column. Offline evaluation sends work back
{1} stage and canary sends it back {3}, but both land on `build and iterate` -- so both
re-do the expensive thing. **Detecting earlier did not shorten the return trip**, and
the return trip is the cost.

Now look at the rows that do work. Moving evaluation-detected defects into data
preparation reaches {shifted['eval defects moved to data preparation']:.0f} hours, and
moving as many as possible there reaches
{shifted['all defects moved to data preparation']:.0f} --
{shifted['all defects moved to data preparation'] / shifted['as built']:.2f} times
as-built.

Data preparation sends work back **{0}** stages. It re-does only itself.

**So the rule is not "detect earlier". It is "detect at a stage that sends work back
less far"** (eq:rework-cost-is-set-by-detection-lateness), and those are different
instructions that happen to coincide in a linear process and come apart in a looping
one.

Practically, that redirects the effort. Building a better offline evaluation set catches
problems sooner and still sends you back to rebuild the model. Building better data
validation catches a different class of problem at a stage that costs
{[w for n, w, p, b in STAGES if n == 'data preparation'][0]:.0f} hours to redo rather
than {sum(w for n, w, p, b in STAGES[1:4]):.0f}. The second is worth more per defect
caught, and it is a smaller piece of work.

The comparison table prices the alternative of simply making fewer mistakes. Cutting
every defect rate to {0.6:.0%} of current reaches {scaled[0.6]:.0f} hours and
{0.4:.0%} reaches {scaled[0.4]:.0f}.

So a substantial across-the-board quality improvement is worth
{1 - scaled[0.6] / scaled[1.0]:.0%}, and relocating detection to a zero-return-trip
stage is worth {1 - shifted['all defects moved to data preparation'] / shifted['as built']:.0%}.
**They are comparable**, and only one of them is a bounded engineering task.

This composes with the previous listing in a way worth stating. There, the period was
dominated by waiting, and the longest wait was `monitor and attribute`. Here that same
stage re-does {cause['monitor and attribute'][3]:.0f} hours per defect --
the most of any detector -- because it is furthest from the cause.

**Late detection costs calendar time and effort simultaneously**, and
ch:sd-fault-tolerance already priced the instrument that moves it: a sampled semantic
monitor at {0.005:.1%} of traffic, detecting in hours rather than weeks. This listing
is the second argument for it. The monitor is cheaper than the damage it prevents, and
it is also cheaper than the rework -- and the rework case is easier to make to a team
that has not yet had the incident.""")
```

```
                   stage  work/visit   P(rework)   visits   total work    share
--------------------------------------------------------------------------------
        scope and design          22          0%     1.22           27       8%
        data preparation          36         10%     2.75           99      29%
       build and iterate          58          0%     2.48          144      43%
      offline evaluation          14         34%     2.84           40      12%
     review and approval           3         12%     1.87            6       2%
       shadow and canary           4         22%     1.65            7       2%
            full rollout           2          5%     1.28            3       1%
   monitor and attribute          11         18%     1.22           13       4%
```

One clean pass is 150 hours; expected is **312 — 2.08×**
({{eq:rework-cost-is-set-by-detection-lateness}}). `build and iterate` is **43%** of
effort with a rework probability of **zero**: nothing goes wrong there, it is where
everything is sent back to.

```
             detected at      P  sends back   stages re-done   hours re-done
------------------------------------------------------------------------------
        data preparation    10%           0                1              36
      offline evaluation    34%           1                2              72
     review and approval    12%           2                3              75
       shadow and canary    22%           3                4              79
            full rollout     5%           6                7             139
   monitor and attribute    18%           7                8             150
```

Note the middle rows. Evaluation returns 72 hours and canary returns 79 — **nearly
identical**, because both land on the same expensive stage.

Which is why shifting left does nothing:

```
                                  scenario  sends back  expected hrs   vs as-built
----------------------------------------------------------------------------------
                                  as built           -           312         1.00x
              canary defects moved to eval           1           320         1.03x
           monitor defects moved to canary           3           308         0.99x
    eval defects moved to data preparation           0           285         0.91x
     all defects moved to data preparation           0           239         0.77x
```

**Detecting three stages earlier is worth nothing; detecting at a stage that returns zero
stages is worth 23%** ({{eq:shift-to-shorter-return-not-earlier}}). The rule is not
"earlier" — it is "shorter return trip", and the two coincide in a line and come apart in
a loop.

For comparison, an across-the-board quality improvement:

```
    defect rates scaled by   expected hrs   vs as-built
--------------------------------------------------------
                       0.6            224         0.72x
                       0.4            194         0.62x
```

Cutting every defect rate to 60% is worth **28%**; relocating detection is worth
**23%**. **Comparable — and only one of them is a bounded engineering task.**

## 10. Production Considerations

Measure wait separately from work. Most time-tracking captures only effort, which is 18%
of the calendar and the part least worth optimising.

Rank interventions by hours removed from the *period*, not by effort saved. The two
rankings differ, and the period ranking puts a measurement change and a policy change
above hiring.

Count changes in flight and publish the number. It is $\lambda P$, it predicts how hard
attribution will be, and it makes the case for shortening the loop in terms a risk-averse
reviewer accepts.

Classify detectors by return-trip cost, not by how early they run. A detector that
returns to an expensive stage is expensive regardless of its position.

Invest in detectors with zero return trip — data validation, schema checks, contract
tests. {{cite:breck2017}}'s rubric is largely a list of these, and its value is where it
catches things rather than that it catches them.

Pre-approve a class of low-risk changes. It is the only long wait that responds to policy
rather than engineering, and it is worth 94 hours in the worked example.

Build the semantic monitor. It is the largest single intervention on the period, and
{{ch:sd-fault-tolerance}} showed it is affordable; this chapter is the second,
independent argument.

## 11. Common Mistakes

**Optimising the work.** It is 18% of the calendar, and it is the only part anyone can
see themselves doing, which is why it attracts the attention.

**Adding people to shorten a loop dominated by waiting.** They shorten the smaller term.

**Reading "shift left" as "detect earlier".** It should be read as "shorten the return
trip", and the two differ.

**Costing a stage by what it takes once.** Multiply by expected visits.

**Blaming the stage that consumes the most effort.** Its cost is caused downstream, by
the detectors that keep sending work back to it.

**Treating a statistical wait as a queue.** Prioritisation does not shorten a
significance test, and escalating it produces frustration rather than samples. The
only levers are a larger sample rate, a larger effect to detect, or a different
statistic.

## 12. Failure Modes

**Attribution collapse.** Enough changes in flight that no metric movement can be
assigned, so every regression becomes a bisect over weeks of history.

**Rework spiral.** A high-rework stage raises visit counts on an expensive upstream
stage, raising the cost of every subsequent defect and lengthening the period further.

**Invisible approval debt.** The approval wait grows as the organisation grows and never
appears in any engineering metric, so the period lengthens with no identifiable cause.

**Shift-left theatre.** Effort spent moving detection earlier with no change in return
trip, producing a better-looking process and identical cost.

**Monitoring that never concludes.** A semantic signal too weak to reach significance
means `monitor and attribute` never completes, changes accumulate indefinitely, and
the loop degenerates into a queue nobody drains.

**Serial-wait creep.** A stage that used to run in parallel becomes blocking after a
reorganisation, and the team's cadence collapses with no change to any stage's own
duration.

## 13. Alternatives

**Ship less often, deliberately.** Reduces changes in flight per unit time and is
sometimes correct, especially where the return trip is long and attribution matters
more than cadence. It is the honest response to a period that cannot be shortened,
and it is strictly better than shipping fast into a loop that cannot tell you what
happened.

**Batch changes into releases.** Trades attribution granularity for fewer trips; the
opposite of the recommendation here and defensible when the period genuinely cannot be
shortened, since fewer larger trips at least make the candidate set explicit rather
than continuous.

**Feature flags to decouple deploy from release.** Shortens the deploy portion of the
period without touching the monitoring portion, so it addresses the smaller term. Its
real value here is different: a flag makes rollback instant, which shortens the
return trip for defects caught after release — and by
{{eq:shift-to-shorter-return-not-earlier}} that is the axis that matters.

**Offline replay against recorded traffic.** Converts part of the canary wait into
compute, which is the only intervention that attacks a statistical wait directly:
instead of waiting for traffic to accumulate, replay traffic you already have. The
cost is a replay harness and the fidelity gap between recorded and live conditions,
and {{ch:ops-deployment}} takes up how large that gap is.

**Accept the period and invest in bisection.** If attribution is the real problem and
the period genuinely cannot move, tooling that narrows fifteen candidates to one is a
legitimate substitute for shortening the loop. It treats the symptom rather than the
cause, and treating the symptom is sometimes the correct engineering decision when the
cause sits outside the team's control.

## 14. Evaluation

Report the period and its work/wait split as a standing metric. Both move, and only the
split explains why an intervention did or did not help.

Track expected visits per stage from your own project history. The rework factor is
measurable from ticket transitions and almost nobody measures it.

Separate serial waits from parallel ones before optimising either. A wait that blocks
only its own change costs cadence very differently from one that blocks the pipeline, and
the same number of hours in the two places calls for different fixes.

Report return-trip cost per detector. It is the number that ranks detectors correctly and
it is derivable from the same history.

Measure changes in flight directly rather than inferring it. It is the attribution
difficulty, and it is what a risk conversation should be about.

Re-measure after each intervention. The period's composition shifts, and the next
best intervention changes with it — {{ch:sd-latency}}'s method applies.

## 15. Advanced Concepts

The Markov model treats rework probabilities as independent of history, which understates
the cost. In practice a change that failed canary once is more likely to fail again — the
underlying difficulty has not gone away — so visit counts are heavier-tailed than the
geometric model predicts. That makes the rework factor worse than **2.08×** for hard
changes and better for easy ones, and it means an average is a poor planning figure. The
practical response is to model rework per change class rather than per stage.

There is an interaction between the two listings that neither models. Shortening the
period reduces changes in flight, which reduces the chance that two changes interact —
and interactions are a source of defects that the per-stage rework probabilities treat as
exogenous. So shortening the period lowers $p_i$ as well as $P$, and the benefit compounds
in a way the linear model misses. {{cite:cemri2025mast}}'s finding that failures correlate
suggests the effect is not small.

The wait figures also hide a distinction that matters for which intervention applies.
Some waits are *serial* -- the canary must run before rollout, and nothing else can
proceed. Others are *parallel* -- a review can happen while other work continues, so its
wait costs period only for the change being reviewed. The model here treats all waiting
as serial, which is right for a single change and wrong for a team with several in
flight. For a team, the binding quantity is the throughput of the narrowest stage rather
than the sum of all waits, which is a queueing question rather than an arithmetic one.
**The single-change period and the team's sustainable cadence are different numbers**,
and a team that has parallelised its pipeline well can have a long period and a short
cadence simultaneously.

The return-trip framing also suggests a design principle the chapter does not develop: a
lifecycle can be **restructured** so that expensive stages come later. If `build and
iterate` ran after the checks that most often send work back to it, those checks would
return to something cheap. That is what a specification-first or test-first process
attempts, and it is a stronger argument for those practices than the usual one — it
changes $b_i$ rather than $p_i$.

## 16. Connection to Previous Chapters

{{eq:semantic-failure-has-no-instrument}} from {{ch:sd-architecture}} explains the largest
wait in the lifecycle: the monitoring signal does not exist by default and takes weeks to
accumulate when it does.

{{eq:semantic-breaker-is-affordable}} from {{ch:sd-fault-tolerance}} priced the fix at
0.5% sampling. This chapter shows it is also the largest single reduction in the period.

{{eq:detection-time-sets-the-blast-radius}} measured late detection in damage; this
chapter measures it in effort and calendar. Same lateness, three costs.

{{eq:tail-attribution-differs-from-mean}} from {{ch:sd-latency}} supplied the method:
rank by measured effect per unit of work rather than by which number looks largest.

## 17. Exercises

1. For your own team, estimate work and wait per stage. What is your work share, and
   which three stages dominate the wait?

2. Compute changes in flight from your period and change rate. How many candidates does a
   metric movement have?

3. Derive the condition under which moving a detector earlier reduces expected rework,
   and check it for two detectors in your process.

4. Extend the second listing so rework probability depends on how many times a change has
   already failed. How much worse is the tail?

5. Design a zero-return-trip detector for a defect class your team currently catches in
   canary. What would it cost to build?

## 18. Interview Questions

1. Our team wants to ship weekly and ships monthly. Where would you look first?

2. Why does hiring an engineer barely change the release cadence here?

3. A metric moved and fifteen changes are outstanding. What went wrong before the metric
   moved?

4. "We should shift testing left." Under what condition is that wrong?

5. Which stage consumes the most effort in the worked example, and what is its rework
   probability? Explain the apparent contradiction.

6. Our period is five weeks and our cadence is weekly. Are those consistent, and what
   would have to be true for both to hold?

## 19. Research Questions

1. How heavy-tailed are real rework distributions, and what does that do to planning
   figures based on means?

2. How much does shortening the period reduce defect rates through fewer interacting
   changes, as distinct from detecting them faster?

3. Can return-trip cost be measured automatically from issue-tracker transitions with
   enough fidelity to rank detectors?

4. Does restructuring a lifecycle so expensive stages come later actually reduce total
   effort, or does it move the cost elsewhere?

## 20. Chapter Summary

The lifecycle is a loop and its period is dominated by waiting. Across a realistic trip,
**847 hours** of which **156 are work** — work is **18%** of the calendar
({{eq:lifecycle-period-is-wait-not-work}}). The three longest waits are **89%** of the
waiting and **12%** of the effort, and nobody is working during them.

Ranked by effect on the period, a **faster attribution signal saves 238 hours** against
hiring an engineer's **39** — the two best interventions are a measurement change and a
policy change rather than capacity.

A long period costs more than throughput: at the baseline, **15 changes are in flight**
and a live defect is **53 days old** ({{eq:period-destroys-attribution}}). When a metric
moves there are fifteen candidates. **A short loop is a diagnostic instrument.**

Rework makes expected effort **2.08×** a clean pass, and it lands on `build and iterate` —
**43%** of all effort at a rework probability of **zero**, because it is where everything
returns to ({{eq:rework-cost-is-set-by-detection-lateness}}).

And the correction to standard advice: moving detection from canary to evaluation is
worth **nothing** (1.03×) because both return to the same expensive stage, while moving
it to a zero-return stage is worth **23%**
({{eq:shift-to-shorter-return-not-earlier}}). **The rule is shorter return trip, not
earlier detection.**

Both findings share a shape with the rest of this book: the quantity that is easy to
measure is not the quantity that binds. Effort is measurable because people record it;
waiting is not, because nobody is doing anything. Detection timing is visible on a
process diagram; return-trip cost is not, because the diagram draws every arrow the
same length. In both cases the correction requires writing down a number the existing
instrumentation does not produce, which is the recurring theme of Parts XXII through
XXIV.

Carry forward: **optimise the wait, not the work**, and **classify detectors by where
they send you back to**.

## 21. Further Reading

- {{cite:paleyes2020deployment}} — obstacles distributed across every workflow stage,
  which is this chapter's tables in qualitative form.
- {{cite:sculley2015}} — the debt that raises per-visit cost on the highest-visit stages.
- {{cite:breck2017}} — a rubric that is largely a list of zero-return-trip detectors.
- {{cite:cemri2025mast}} — correlated failures, which suggest the interaction effect
  between in-flight changes is not small.

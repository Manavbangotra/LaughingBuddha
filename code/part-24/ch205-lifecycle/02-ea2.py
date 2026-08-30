# -*- coding: utf-8 -*-
# Extracted from: Chapter 205 — The ML Lifecycle
# Source: src/.../ch205-lifecycle.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

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

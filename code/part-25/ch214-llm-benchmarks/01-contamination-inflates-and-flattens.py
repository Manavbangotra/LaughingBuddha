# -*- coding: utf-8 -*-
# Extracted from: Chapter 214 — LLM Evaluation: Benchmarks and Their Limits
# Source: src/.../ch214-llm-benchmarks.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A public benchmark has a lifespan, and it is shorter than anyone plans for.

Two things happen to a benchmark after publication. Its items leak into training corpora,
so reported scores rise faster than capability does. And the frontier moves toward the
ceiling, so the difference between the models anyone cares about gets smaller.

Both effects push the same way: **the gap between two models shrinks while the noise stays
put** (eq:contamination-inflates-and-flattens), so the number of items needed to tell them
apart grows until it exceeds the number of items the benchmark has
(eq:headroom-sets-benchmark-lifespan).

This listing computes that crossing point, which is the honest definition of when a
benchmark is finished.
"""
import math

N_ITEMS = 4000
LEAK_RATE = 0.145             # per year; contamination = 1 - exp(-rate * years)
POWER_Z = 2.80                # z(0.80 power) + z(0.05 two-sided), roughly


def contamination(years):
    return 1.0 - math.exp(-LEAK_RATE * years)


def reported(true_q, years):
    """Contaminated items are answered correctly whether or not the model can."""
    c = contamination(years)
    return true_q + (1.0 - true_q) * c


print(f"A {N_ITEMS}-item benchmark. Items leak into training corpora over time.")
print()
print(f"{'years':>7}{'contaminated':>15}", end="")
for q in (0.40, 0.55, 0.70):
    print(f"{('true ' + format(q, '.2f')):>13}", end="")
print(f"{'A vs C gap':>13}")
print("-" * 74)
tab = {}
for y in (0, 1, 2, 3, 5, 8):
    c = contamination(y)
    row = [reported(q, y) for q in (0.40, 0.55, 0.70)]
    tab[y] = (c, row, row[2] - row[0])
    print(f"{y:>7}{c:>15.1%}", end="")
    for v in row:
        print(f"{v:>13.3f}", end="")
    print(f"{row[2] - row[0]:>13.3f}")

print()
print("Every model looks better every year without improving, and the space")
print("between them closes.")

print()
print()
print("What that does to the sample size needed to separate two models.")
print()


def n_needed(p1, p2):
    """Two-proportion comparison, 80% power, alpha 0.05."""
    d = abs(p2 - p1)
    if d < 1e-9:
        return float("inf")
    p = (p1 + p2) / 2.0
    return (POWER_Z ** 2) * 2.0 * p * (1 - p) / (d ** 2)


print(f"{'years':>7}{'reported gap':>15}{'items needed':>15}"
      f"{'benchmark has':>16}{'usable?':>10}")
print("-" * 63)
life = {}
for y in (0, 1, 2, 3, 5, 8, 12, 18):
    c = contamination(y)
    a, b = reported(0.55, y), reported(0.70, y)
    n = n_needed(a, b)
    life[y] = (b - a, n)
    print(f"{y:>7}{b - a:>15.3f}{n:>15.0f}{N_ITEMS:>16}"
          f"{('yes' if n <= N_ITEMS else 'no'):>10}")

dead = min((y for y in (0, 1, 2, 3, 5, 8, 12, 18) if life[y][1] > N_ITEMS),
           default=None)
print()
print("for a gap this coarse the benchmark survives "
      + (f"until year {dead}" if dead else "past year 18"))

print()
print()
print("Saturation does the same thing independently, with no contamination at all.")
print()
print(f"{'frontier score':>16}{'next model':>13}{'true gap':>11}"
      f"{'items needed':>15}{'vs at 0.50':>13}")
print("-" * 68)
STEP = 0.045
sat = {}
base = None
for f in (0.50, 0.65, 0.78, 0.87, 0.93, 0.965, 0.985):
    nxt = min(0.999, f + STEP * (1 - f) / 0.5)
    n = n_needed(f, nxt)
    if base is None:
        base = n
    sat[f] = (nxt, nxt - f, n)
    print(f"{f:>16.3f}{nxt:>13.3f}{nxt - f:>11.4f}{n:>15.0f}{n / base:>12.1f}x")

print()
print("A fixed fraction of remaining headroom is a shrinking absolute gain,")
print("and sample size goes as one over the gap squared.")

print()
print()
print("Both effects together: benchmark lifespan against the frontier it tracks.")
print()
print(f"{'year':>6}{'contamination':>16}{'frontier':>11}{'true step':>12}"
      f"{'observed step':>16}{'items needed':>15}{'usable?':>10}")
print("-" * 86)
front = 0.52
both = {}
for y in range(0, 9):
    c = contamination(y)
    nxt = front + STEP * (1 - front) / 0.5
    obs_a, obs_b = reported(front, y), reported(min(nxt, 0.999), y)
    n = n_needed(obs_a, obs_b)
    both[y] = (c, front, nxt - front, obs_b - obs_a, n)
    print(f"{y:>6}{c:>16.1%}{front:>11.3f}{nxt - front:>12.4f}"
          f"{obs_b - obs_a:>16.4f}{n:>15.0f}"
          f"{('yes' if n <= N_ITEMS else 'no'):>10}")
    front = nxt
dead_gen = min((y for y in both if both[y][4] > N_ITEMS), default=None)
print()
print(f"for one generation of progress it survives until year {dead_gen}")

print()
print()
print("What each remedy buys, at year 5.")
print()
Y = 5
c5 = contamination(Y)
f5 = both[Y][1]
n5 = both[Y][4]
print(f"{'remedy':>32}{'items needed':>15}{'vs nothing':>13}{'cost':>22}")
print("-" * 82)
REMEDIES = [
    ("nothing", n5, "zero"),
    ("grow the benchmark 4x", n5, "4x labelling"),
    ("hold out a private split", n_needed(f5, f5 + both[Y][2]), "one-time"),
    ("regenerate items annually",
     n_needed(reported(f5, 1), reported(f5 + both[Y][2], 1)), "annual labelling"),
    ("raise difficulty (reset headroom)",
     n_needed(0.39, 0.39 + STEP * (1 - 0.39) / 0.5), "a new benchmark"),
]
rem = {}
for name, n, cost in REMEDIES:
    rem[name] = n
    print(f"{name:>32}{n:>15.0f}{n5 / n:>12.1f}x{cost:>22}")

print()
print(f"note: growing the benchmark changes what {N_ITEMS} means, not what is needed.")

print(f"""
The contamination table is the mechanism and the last column is the finding. A model with
true capability {0.40:.2f} reports {tab[5][1][0]:.3f} after five years without having
learned anything, and one at {0.70:.2f} reports {tab[5][1][2]:.3f}. The gap between them
falls from {tab[0][2]:.3f} to {tab[5][2]:.3f}.

That compression is the part that matters. Score inflation on its own is annoying and
correctable -- everyone knows the numbers are optimistic. **The gap shrinking is not
correctable**, because it is the signal (eq:contamination-inflates-and-flattens), and it
shrinks by exactly the contaminated fraction.

The sample-size table converts that into a date. Separating a {0.55:.2f} model from a
{0.70:.2f} model needs {life[0][1]:.0f} items at release and {life[18][1]:.0f} in year 18,
because the required count goes as one over the gap squared. For a gap that coarse, a
{N_ITEMS}-item benchmark survives past year 18.

That is the reassuring version, and it is why benchmarks feel durable. It is also the wrong
question, because nobody is choosing between a {0.55:.2f} model and a {0.70:.2f} one -- the
comparisons that matter are between this generation and the last.

The saturation table shows the same failure arriving without any contamination at all.
Suppose each model generation closes a fixed fraction of the remaining headroom -- a
reasonable model of progress. At a frontier of {0.50:.2f} that is a
{sat[0.50][1]:.4f} gain needing {sat[0.50][2]:.0f} items; at {0.965:.3f} it is
{sat[0.965][1]:.4f} needing {sat[0.965][2]:.0f} --
**{sat[0.965][2] / sat[0.50][2]:.0f} times as many** (eq:headroom-sets-benchmark-lifespan).

The two mechanisms are independent and they compound. The combined table walks a frontier
from {0.52:.2f} upward while contamination accumulates, and the items needed to detect one
generation's progress goes from {both[0][4]:.0f} in year 0 to {both[8][4]:.0f} in year 8.

Against {N_ITEMS} items, the benchmark can do it in years 0 and 1 and cannot from
**year {dead_gen}** onward.

So the useful definition is conditional on the question: **a benchmark is finished when the
sample size it would need exceeds the sample size it has**, and the year that happens is
{dead_gen} for tracking progress and past 18 for sorting coarse tiers. Both numbers are
computable in advance from a leak-rate estimate; the second is the one that gets quoted and
the first is the one that governs whether a leaderboard means anything.

The remedies table is where this becomes a decision rather than an observation, and the
first two rows are the important ones. **Growing the benchmark does not help.** Quadrupling
the item count leaves the required count exactly where it was -- it changes what you have,
not what you need, and it is the intervention teams reach for because it is the one that
feels like effort.

A private held-out split removes contamination entirely and takes the requirement to
{rem['hold out a private split']:.0f} items -- a
{n5 / rem['hold out a private split']:.1f}x improvement for a one-time cost, and the best
return per unit of effort on the list. Annual regeneration reaches only
{rem['regenerate items annually']:.0f} for a *recurring* cost, because a set regenerated
last year has already absorbed a year of leakage.

And the last row is the honest one. Raising difficulty so the frontier sits back at
{0.39:.2f} needs {rem['raise difficulty (reset headroom)']:.0f} items --
{n5 / rem['raise difficulty (reset headroom)']:.0f} times better than anything else -- and
it is not a repair. It is a new benchmark, which is why the field keeps building them, and
why each new one is harder than the last by construction rather than by ambition.

Two cautions. The leak rate here is a parameter and nobody publishes theirs, so the dates
above are arithmetic on an estimate; the *shape* is robust and the year is not. And this
listing measures only whether two models can be *separated*, which is a lower bar than
whether the separation means anything -- ch:ev-llm-benchmarks' second listing takes up what
the score is measuring in the first place.""")

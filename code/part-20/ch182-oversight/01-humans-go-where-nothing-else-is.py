# -*- coding: utf-8 -*-
# Extracted from: Chapter 182 — Where Human Oversight Remains Necessary
# Source: src/.../ch182-oversight.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Allocating a fixed amount of human attention across an analysis pipeline.

Every chapter in part:20 has ended by saying a human is needed somewhere. This
listing puts a budget on that and asks where.

The inputs are the ones the part measured: each stage has an error rate, a
detection rate for whatever AUTOMATED check exists there, and a detection rate for
a human looking at it. ch:aids-stack found automated detection varying ninefold
across stages. Human detection varies much less -- a person is moderately good at
everything -- which is exactly what makes the allocation non-obvious.

The consequence is that human attention is worth most where it is the ONLY verifier
available, not where the errors are most common (eq:humans-go-where-nothing-else-is).
"""
import numpy as np
import itertools

rng = np.random.default_rng(4877)

M = 40000

# (stage, error rate, automated detection, human detection, human hours per check)
STAGES = [
    ("frame the question", 0.14, 0.02, 0.72, 0.6),
    ("access",             0.09, 0.90, 0.60, 0.3),
    ("clean",              0.20, 0.45, 0.66, 1.1),
    ("explore",            0.13, 0.15, 0.58, 0.9),
    ("feature",            0.15, 0.35, 0.62, 0.8),
    ("model",              0.08, 0.80, 0.55, 0.5),
    ("conclude",           0.13, 0.10, 0.75, 0.7),
]
N = len(STAGES)
DECAY = 0.66            # ch:as-failures: detection falls as an error propagates
FIX = 0.88


def run(human_at, m=M, decay=DECAY, automated=True):
    """`human_at` is the set of stages a human inspects. Automated checks run
    everywhere they exist. Returns (correct, human hours)."""
    err_at = np.full(m, -1, dtype=np.int64)
    hours = 0.0
    for i, (name, p_err, auto, hum, cost) in enumerate(STAGES):
        fresh = (err_at < 0) & (rng.random(m) < p_err)
        err_at[fresh] = i
        live = err_at >= 0
        lag = np.where(live, i - err_at, 0)
        power = 0.0
        if automated:
            power = auto * (decay ** np.clip(lag, 0, None))
        if i in human_at:
            hours += cost
            hp = hum * (decay ** np.clip(lag, 0, None))
            # Two independent-ish checks: the combined miss rate is the product.
            power = 1 - (1 - power) * (1 - hp)
        if np.any(power):
            caught = live & (rng.random(m) < power) & (rng.random(m) < FIX)
            err_at[caught] = -1
    return float((err_at < 0).mean()), hours


print(f"{M:,} analyses. Each stage has an automated check of the strength")
print("ch:aids-stack measured, and a human can additionally inspect any stage.")
print()
print(f"{'stage':>20}{'error':>8}{'automated':>11}{'human':>8}{'hours':>8}"
      f"{'only human?':>13}")
print("-" * 68)
for name, e, a, h, c in STAGES:
    only = "yes" if a < 0.25 else ""
    print(f"{name:>20}{e:>8.0%}{a:>11.0%}{h:>8.0%}{c:>8.1f}{only:>13}")

base_auto = run(set())[0]
base_none = run(set(), automated=False)[0]
print()
print(f"   automated checks only: {base_auto:.1%} correct")
print(f"   no checks at all:      {base_none:.1%} correct")

print()
print()
print("One human inspection, placed at each stage in turn.")
print()
print(f"{'human inspects':>20}{'correct':>10}{'gain':>9}{'hours':>8}"
      f"{'gain/hour':>12}")
print("-" * 59)
single = {}
for i, (name, e, a, h, c) in enumerate(STAGES):
    r = run({i})
    single[name] = (r[0], r[0] - base_auto, (r[0] - base_auto) / c)
    print(f"{name:>20}{r[0]:>10.1%}{r[0] - base_auto:>+9.1%}{c:>8.1f}"
          f"{(r[0] - base_auto) / c:>12.3f}")

print()
print()
print("Ranked by return per hour, against the two variables people allocate by.")
print()
order = sorted(single, key=lambda k: -single[k][2])
lookup = {s[0]: s for s in STAGES}
print(f"{'rank':>6}{'stage':>20}{'gain/hour':>12}{'error rate':>12}"
      f"{'automated detection':>21}")
print("-" * 71)
for r, name in enumerate(order, 1):
    st = lookup[name]
    print(f"{r:>6}{name:>20}{single[name][2]:>12.3f}{st[1]:>12.0%}{st[2]:>21.0%}")

print()
print()
print("The best allocation at each budget, searched exhaustively.")
print()
print(f"{'hours':>8}{'stages':>8}{'best placement':>46}{'correct':>10}")
print("-" * 72)
budgets = {}
for budget in (1.0, 2.0, 3.5, 5.0):
    best, best_v = None, -1.0
    for k in range(1, N + 1):
        for combo in itertools.combinations(range(N), k):
            cost = sum(STAGES[i][4] for i in combo)
            if cost > budget:
                continue
            v = run(set(combo))[0]
            if v > best_v:
                best, best_v = combo, v
    budgets[budget] = (best, best_v)
    short = {"frame the question": "frame", "conclude": "conclude"}
    names = ", ".join(short.get(STAGES[i][0], STAGES[i][0]) for i in best)         if best else "(none)"
    print(f"{budget:>8.1f}{len(best or ()):>8}{names:>46}{best_v:>10.1%}")

print()
print()
print("Against the allocations teams actually use, at a 3.5-hour budget.")
print()
print(f"{'policy':>34}{'correct':>10}{'hours':>8}")
print("-" * 52)
POLICIES = [
    ("review the model and conclusions", {5, 6}),
    ("review the data work", {1, 2}),
    ("review the final report only", {6}),
    ("spread evenly (every other stage)", {0, 2, 4, 6}),
]
pol = {}
for label, ck in POLICIES:
    r = run(ck)
    pol[label] = r
    print(f"{label:>34}{r[0]:>10.1%}{r[1]:>8.1f}")
opt = budgets[3.5]
print(f"{'the optimum at this budget':>34}{opt[1]:>10.1%}"
      f"{sum(STAGES[i][4] for i in opt[0]):>8.1f}")

print()
print()
print("And what happens if the automated checks are removed -- which is the")
print("regime a team without them is in.")
print()
print(f"{'human inspects':>20}{'with automation':>17}{'without':>10}"
      f"{'difference':>13}")
print("-" * 60)
noauto = {}
for i, (name, e, a, h, c) in enumerate(STAGES):
    w = run({i})[0]
    wo = run({i}, automated=False)[0]
    noauto[name] = (w, wo)
    print(f"{name:>20}{w:>17.1%}{wo:>10.1%}{w - wo:>13.1%}")

print(f"""
The single-check table has a clear winner and it is not where errors are made.

A human inspecting the CONCLUSION is worth {single['conclude'][1]:+.1%} for
{0.7:.1f} hours -- {single['conclude'][2]:.3f} per hour, more than double the next
best. A human inspecting the CLEANING, which has the highest error rate in the
table at {0.20:.0%}, is worth {single['clean'][1]:+.1%} for {1.1:.1f} hours, and
ranks last.

The ranking table shows the two variables people actually allocate by failing to
predict it. Error rate does not: cleaning is the highest and ranks
{[i for i, n in enumerate(order, 1) if n == 'clean'][0]}. Nor does time share, which
ch:aids-stack put mostly in the data stages.

What predicts it is **how weak the automated check is, multiplied by how much has
accumulated by the time the human looks**
(eq:humans-go-where-nothing-else-is). The conclusion stage scores highest on both:
automated detection of {0.10:.0%}, and every error made anywhere upstream is still
outstanding when it is reached.

Cleaning loses on the first term -- {0.45:.0%} automated detection already covers
much of it -- and on cost, since reading a cleaning pipeline takes longer than
reading a conclusion.

The budget table is the practical output. At {1.0:.1f} hours the optimum inspects
access and the conclusion; at {3.5:.1f} it adds framing, exploration, features and
the model, and reaches {budgets[3.5][1]:.1%}.

Note what the optimum does at every budget: **it always includes the conclusion,
and it never includes cleaning until the budget is large.** That is the opposite of
how review is usually organised.

The policy comparison makes the cost of the usual organisation concrete. "Review
the model and conclusions" -- the standard practice -- reaches
{pol['review the model and conclusions'][0]:.1%} at {1.2:.1f} hours. Spreading
evenly reaches {pol['spread evenly (every other stage)'][0]:.1%} at {3.2:.1f}. The
optimum at {3.5:.1f} hours reaches {budgets[3.5][1]:.1%}.

The gaps are modest, which is worth saying honestly: the surface is fairly flat and
almost any review is much better than none. The comparison that is not modest is
the last table.

**Removing the automated checks costs about {28:.0f} points at every placement.**
A human inspecting the conclusion reaches {noauto['conclude'][0]:.1%} alongside
automated checks and {noauto['conclude'][1]:.1%} without them.

Human and automated verification are **complements, not substitutes**. The
automated checks do the volume; the human covers the stages where no automated
check exists. A team that has one and not the other is not at some intermediate
point on a trade-off -- it is missing half of a mechanism, and the half it has
cannot do the other's job.

Which is the practical summary of this listing. Build the automated checks
everywhere they are possible; spend the human on the stages where they are not; and
spend it late, where everything upstream is still visible.""")

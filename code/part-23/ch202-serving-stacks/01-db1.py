# -*- coding: utf-8 -*-
# Extracted from: Chapter 202 — Serving Stacks: vLLM, TensorRT-LLM, and Triton
# Source: src/.../ch202-serving-stacks.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""What a serving stack buys, decomposed -- and why the decomposition is ambiguous.

A serving stack is a bundle of techniques this part has measured separately: continuous
batching, paged cache, chunked prefill, quantised kernels, graph capture. Vendors report
the bundle's total, which is real and tells you nothing about which parts you need.

This listing adds the features one at a time and measures each one's marginal gain --
then adds them in a different order and gets different numbers for the same features
(eq:feature-credit-depends-on-order).

That is not a measurement error. It is what happens when techniques multiply rather than
add, and it is why "feature X gives 3x" is a claim that requires a baseline to mean
anything.
"""
import itertools

# Each feature's effect is a multiplier on achieved throughput, but several act
# on the SAME inefficiency, so applying one reduces what the next can recover.
# (name, inefficiency it removes, share of that inefficiency it removes)
FEATURES = [
    ("continuous batching", "slot idleness",   0.92),
    ("paged KV cache",      "slot idleness",   0.55),
    ("chunked prefill",     "phase stalls",    0.88),
    ("fp8 weights",         "weight traffic",  0.50),
    ("graph capture",       "launch overhead", 0.85),
]

# How much of a naive loop's time each inefficiency accounts for.
BUDGET = {
    "slot idleness":   0.46,
    "phase stalls":    0.21,
    "weight traffic":  0.18,
    "launch overhead": 0.09,
    "irreducible":     0.06,
}


def throughput(active):
    """Relative throughput with `active` features enabled.

    Time is the sum of each inefficiency's remaining share plus the irreducible
    part. Two features acting on one inefficiency cannot each remove all of it.
    """
    remaining = 0.0
    for cause, share in BUDGET.items():
        if cause == "irreducible":
            remaining += share
            continue
        left = 1.0
        for name, target, frac in FEATURES:
            if name in active and target == cause:
                left *= (1.0 - frac)
        remaining += share * left
    return 1.0 / remaining


print("A naive serving loop's time, by what it is wasted on.")
print()
print(f"{'inefficiency':>18}{'share of time':>16}   {'addressed by':<38}")
print("-" * 76)
for cause, share in BUDGET.items():
    who = ", ".join(n for n, t, _ in FEATURES if t == cause) or "nothing"
    print(f"{cause:>18}{share:>16.0%}   {who:<38}")

base = throughput(set())
full = throughput(set(n for n, _, _ in FEATURES))
print()
print(f"naive loop: {base:.2f}x   everything on: {full:.2f}x")

print()
print()
print("Adding features in the order a vendor's changelog lists them.")
print()
ORDER_A = ["continuous batching", "paged KV cache", "chunked prefill",
           "fp8 weights", "graph capture"]
print(f"{'added':>22}{'cumulative':>13}{'marginal gain':>16}{'credit':>10}")
print("-" * 62)
active = set()
prev = base
creditA = {}
for f in ORDER_A:
    active.add(f)
    now = throughput(active)
    creditA[f] = (now / prev, (now - prev) / (full - base))
    print(f"{f:>22}{now:>12.2f}x{now / prev:>15.2f}x"
          f"{(now - prev) / (full - base):>10.0%}")
    prev = now

print()
print()
print("The same five features, added in reverse order.")
print()
ORDER_B = list(reversed(ORDER_A))
print(f"{'added':>22}{'cumulative':>13}{'marginal gain':>16}{'credit':>10}")
print("-" * 62)
active = set()
prev = base
creditB = {}
for f in ORDER_B:
    active.add(f)
    now = throughput(active)
    creditB[f] = (now / prev, (now - prev) / (full - base))
    print(f"{f:>22}{now:>12.2f}x{now / prev:>15.2f}x"
          f"{(now - prev) / (full - base):>10.0%}")
    prev = now

print()
print()
print("Same features, same total, different attribution.")
print()
print("-" * 71)
print(f"{'feature':>22}{'credit in order A':>20}{'credit in order B':>20}"
      f"{'ratio':>9}")
for f in ORDER_A:
    a = creditA[f][1]
    b = creditB[f][1]
    print(f"{f:>22}{a:>20.0%}{b:>20.0%}"
          f"{max(a, b) / max(min(a, b), 1e-9):>9.1f}x")

print()
print()
print("The order-independent measure: what each feature is worth on its own,")
print("and what removing it costs from the full stack.")
print()
print(f"{'feature':>22}{'alone':>10}{'removing from full':>21}"
      f"{'Shapley share':>16}")
print("-" * 70)
ALL = [n for n, _, _ in FEATURES]
shap = {}
for f in ALL:
    alone = throughput({f}) / base
    without = full / throughput(set(ALL) - {f})
    # Shapley value: average marginal contribution over all orderings.
    total = 0.0
    count = 0
    for perm in itertools.permutations(ALL):
        act = set()
        for g in perm:
            before = throughput(act)
            act.add(g)
            if g == f:
                total += throughput(act) - before
                break
        count += 1
    shap[f] = total / count
    print(f"{f:>22}{alone:>9.2f}x{without:>20.2f}x"
          f"{(total / count) / (full - base):>16.0%}")

print()
print()
print("And the question that actually matters: which subset is worth building?")
print()
print(f"{'subset':>52}{'throughput':>13}{'features':>11}")
print("-" * 78)
best_by_size = {}
for k in range(1, len(ALL) + 1):
    best, bestv = None, 0.0
    for combo in itertools.combinations(ALL, k):
        v = throughput(set(combo))
        if v > bestv:
            best, bestv = combo, v
    best_by_size[k] = (best, bestv)
    label = ", ".join(b.split()[0] for b in best)
    print(f"{label:>52}{bestv:>12.2f}x{k:>11}")

print(f"""
The budget table is where the naive loop's time goes, and it is worth noticing that no
single cause dominates. Slot idleness is {BUDGET['slot idleness']:.0%}, phase stalls
{BUDGET['phase stalls']:.0%}, weight traffic {BUDGET['weight traffic']:.0%}, launch
overhead {BUDGET['launch overhead']:.0%}, and {BUDGET['irreducible']:.0%} is
irreducible.

Turning everything on takes throughput from {base:.2f}x to **{full:.2f}x**. That is the
number a serving stack advertises, and it is honest.

The two ordering tables are where honesty gets complicated. Read the credit column --
each feature's share of the total improvement.

In the first ordering, continuous batching is credited with
{creditA['continuous batching'][1]:.0%} of the gain and graph capture with
{creditA['graph capture'][1]:.0%}. In the reverse ordering, continuous batching gets
{creditB['continuous batching'][1]:.0%} and graph capture gets
{creditB['graph capture'][1]:.0%}.

**Same features, same workload, same total. The attributed credit differs by a factor of
{creditB['continuous batching'][1] / creditA['continuous batching'][1]:.1f} for one
feature and {creditA['graph capture'][1] / creditB['graph capture'][1]:.1f} for another,
purely from the order they were switched on**
(eq:feature-credit-depends-on-order).

The attribution table shows this is not confined to one feature. Whichever feature is
switched on first gets credit for the largest share of time, because it is the only one
operating against the full naive baseline -- and every feature after it works on what is
left.

That has a direct consequence for reading benchmarks. **"Feature X gives Nx" is not a
property of feature X**; it is a property of X and the baseline it was measured
against. A vendor comparing against a naive loop and a vendor comparing against their
previous release are reporting different quantities, and neither is wrong.

The Shapley column is the order-independent answer: average each feature's marginal
contribution over every possible ordering. By that measure
`{max(shap, key=lambda k: shap[k])}` is worth
{shap[max(shap, key=lambda k: shap[k])] / (full - base):.0%} of the total gain and
`{min(shap, key=lambda k: shap[k])}` is worth
{shap[min(shap, key=lambda k: shap[k])] / (full - base):.0%}.

It is also the wrong question for a build decision, which is why the last table exists.
A team is not choosing an attribution; it is choosing a subset to implement. The best
single feature gives {best_by_size[1][1]:.2f}x, the best two give
{best_by_size[2][1]:.2f}x, and the best three give {best_by_size[3][1]:.2f}x against the
full five's {full:.2f}x.

**Three of the five features capture
{(best_by_size[3][1] - base) / (full - base):.0%} of the available gain**, and the two
left out are the ones that overlap with something already chosen. That is the useful
form of the result: not which feature is best, but which ones stop being worth building
once you have the others.

The general lesson is worth stating outside this table. **Techniques that address the
same inefficiency are substitutes, not complements**, and a roadmap that budgets them
additively will overpromise by the amount they overlap. The way to catch it before
building is to name the inefficiency each item addresses -- which is what the first
column of the budget table does, and which almost no roadmap records.""")

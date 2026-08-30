# -*- coding: utf-8 -*-
# Extracted from: Chapter 233 — Scaling Laws Revisited
# Source: src/.../ch233-scaling-revisited.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A scaling law is an allocation rule, and the allocation changes when you count inference.

The parametric form is `L(N, D) = E + A/N^a + B/D^b` -- an irreducible floor plus a parameter
term plus a data term. Everything interesting follows from three facts about it.

First, the exponents `a` and `b` decide how a fixed training budget splits between parameters
and tokens, and they do not decide where the curve ends
(eq:scaling-exponents-set-allocation-not-the-ceiling). Refitting them on a wider range changed
the recommended allocation by an order of magnitude without changing the shape of the surface
(cite:kaplan2020scaling, cite:hoffmann2022chinchilla), which is why the same budget has been
spent two very different ways within a few years.

Second, the split that minimises training loss per training FLOP is not the split that
minimises cost per served token, and for any real serving volume the two are far apart
(eq:the-training-optimum-is-not-the-deployment-optimum).

Third, `E` is a floor. As the budget grows, the terms the exponents govern shrink and the floor
does not.

The constants below are illustrative and chosen to make the arithmetic legible; the structure
is what transfers.
"""
E_FLOOR = 1.69       # irreducible loss (nats/token), illustrative
N_REF, D_REF = 1e10, 2e11    # anchor point both fits are made to agree on
TERM_REF = 0.30              # each of the two reducible terms, at the anchor


def constants(a, b):
    """Constants that make a given exponent pair agree at the anchor point."""
    return TERM_REF * N_REF ** a, TERM_REF * D_REF ** b


def loss(n, d, a, b):
    an, bd = constants(a, b)
    return E_FLOOR + an / n ** a + bd / d ** b


def optimal_split(c, a, b, steps=4000):
    """Cheapest (N, D) with 6ND = C, by scanning log N."""
    best = None
    lo, hi = 1e6, 1e13
    for k in range(steps):
        n = lo * (hi / lo) ** (k / (steps - 1))
        d = c / (6 * n)
        if d < 1e7:
            continue
        v = loss(n, d, a, b)
        if best is None or v < best[0]:
            best = (v, n, d)
    return best


# exponent pairs, not attributions: the point is that the pair decides the split
REGIMES = [
    ("shallow exponents", 0.076, 0.095),
    ("steeper exponents", 0.340, 0.280),
]
SHALLOW, STEEP = REGIMES[0][0], REGIMES[1][0]

print("The same budget, split two ways.")
print()
print(f"{'training FLOPs':>17}{'exponent pair':>20}{'parameters':>15}{'tokens':>15}"
      f"{'tokens/param':>15}{'loss':>9}")
print("-" * 91)
ratios = {}
for c in (1e19, 1e21, 1e23, 1e25):
    for label, a, b in REGIMES:
        v, n, d = optimal_split(c, a, b)
        ratios.setdefault(label, []).append(d / n)
        print(f"{c:>17.0e}{label:>20}{n:>15.3e}{d:>15.3e}{d / n:>15.1f}{v:>9.3f}")
    print()

mean_shallow = sum(ratios[SHALLOW]) / len(ratios[SHALLOW])
mean_steep = sum(ratios[STEEP]) / len(ratios[STEEP])
SPLIT_FACTOR = mean_shallow / mean_steep
print(f"mean tokens per parameter: {mean_shallow:.1f} under the shallow pair,"
      f" {mean_steep:.1f} under the steeper one")
print(f"a factor of {SPLIT_FACTOR:.0f} in the same budget")

print()
print()
print("Both pairs agree on what the budget buys, and on where it stops.")
print()
A, B = REGIMES[1][1], REGIMES[1][2]
AN, BD = constants(A, B)
print(f"{'training FLOPs':>17}{'best loss':>12}{'above the floor':>18}"
      f"{'floor share of loss':>22}")
print("-" * 69)
gaps = {}
for c in (1e19, 1e21, 1e23, 1e25, 1e27):
    v, n, d = optimal_split(c, A, B)
    gaps[c] = v - E_FLOOR
    print(f"{c:>17.0e}{v:>12.3f}{v - E_FLOOR:>18.3f}{E_FLOOR / v:>22.1%}")

print()
print(f"from 1e19 to 1e27 -- eight orders of magnitude -- the reducible part falls")
print(f"from {gaps[1e19]:.3f} to {gaps[1e27]:.3f}, a factor of {gaps[1e19] / gaps[1e27]:.1f}")
print(f"and the floor's share of the loss rises to"
      f" {E_FLOOR / (E_FLOOR + gaps[1e27]):.1%}")

print()
print()
print("What another halving of the reducible loss costs.")
print()
print(f"{'reducible loss target':>24}{'training FLOPs needed':>24}"
      f"{'multiple of the last':>23}")
print("-" * 71)
prev_c = None
targets = [1.0, 0.5, 0.25, 0.125]
needs = {}
for t in targets:
    c = 1e17
    while optimal_split(c, A, B)[0] - E_FLOOR > t and c < 1e34:
        c *= 1.5
    needs[t] = c
    mult = f"{c / prev_c:>22.0f}x" if prev_c else f"{'--':>23}"
    print(f"{t:>24.3f}{c:>24.2e}{mult}")
    prev_c = c

print()
print("The exponent sets that multiple and nothing sets the floor.")

print()
print()
print("Now count inference, which the training-optimal split does not.")
print()
SERVE = [
    ("a research artefact",      1e9),
    ("an internal tool",         1e12),
    ("a product feature",        1e14),
    ("a consumer product",       1e16),
]
TARGET = 2.10
print(f"{'deployment':>22}{'tokens served':>16}{'best N':>13}{'best D':>13}"
      f"{'train FLOPs':>14}{'serve FLOPs':>14}{'total':>13}")
print("-" * 105)


def cheapest_for_target(tokens_served, target, steps=600):
    """Smallest total FLOPs reaching a loss target, over choices of N."""
    best = None
    for k in range(steps):
        n = 1e7 * (1e13 / 1e7) ** (k / (steps - 1))
        # tokens needed at this N to hit the target
        rem = target - E_FLOOR - AN / n ** A
        if rem <= 0:
            continue
        d = (BD / rem) ** (1 / B)
        total = 6 * n * d + 2 * n * tokens_served
        if best is None or total < best[0]:
            best = (total, n, d, 6 * n * d, 2 * n * tokens_served)
    return best


serve_best = {}
for label, served in SERVE:
    total, n, d, tr, inf = cheapest_for_target(served, TARGET)
    serve_best[label] = (n, d, tr, inf, total)
    print(f"{label:>22}{served:>16.0e}{n:>13.3e}{d:>13.3e}"
          f"{tr:>14.2e}{inf:>14.2e}{total:>13.2e}")

n_small = serve_best["a research artefact"][0]
n_large = serve_best["a consumer product"][0]
print()
print(f"at a loss target of {TARGET:.2f}, the best model shrinks from"
      f" {n_small:.2e} to {n_large:.2e} parameters")
print(f"a factor of {n_small / n_large:.0f}, driven entirely by serving volume")

print()
print()
print("What using the training-optimal model instead costs.")
print()
tr_opt_n, tr_opt_d = None, None
v, tr_opt_n, tr_opt_d = optimal_split(6e23, A, B)
print(f"training-optimal at 6e23 FLOPs: {tr_opt_n:.3e} parameters,"
      f" loss {v:.3f}")
print()
overspend = {}
print(f"{'deployment':>22}{'training-optimal total':>25}{'inference-aware total':>24}"
      f"{'overspend':>13}")
print("-" * 84)
for label, served in SERVE:
    # match the loss target by adding tokens to the training-optimal N
    rem = TARGET - E_FLOOR - AN / tr_opt_n ** A
    d_match = (BD / rem) ** (1 / B) if rem > 0 else float("inf")
    naive = 6 * tr_opt_n * d_match + 2 * tr_opt_n * served
    aware = serve_best[label][4]
    overspend[label] = naive / aware
    print(f"{label:>22}{naive:>25.2e}{aware:>24.2e}{naive / aware:>12.1f}x")

print(f"""
The first table is the result that made scaling laws an engineering subject rather than a
curiosity. The same training budget, split according to two different exponent pairs, produces
models that differ by a factor of **{SPLIT_FACTOR:.0f}** in tokens per parameter --
{mean_shallow:.1f} under the shallow pair against {mean_steep:.1f} under the steeper one.

Both pairs describe a surface of the same shape and both are anchored to agree at one point.
They disagree about the *slope in two directions*, and that is what decides the split: at the
optimum the two reducible terms stand in the ratio `b/a`, so the exponents alone fix how much of
a budget becomes parameters and how much becomes tokens.

**The exponents are an allocation rule**
(eq:scaling-exponents-set-allocation-not-the-ceiling). Refitting them on a wider range is what
changed the industry's recommended model size for a given budget
(cite:kaplan2020scaling, cite:hoffmann2022chinchilla) -- not a new capability, a new division of
the same money.

The second table is the part that gets less attention and matters more over time. From 1e19 to
1e27 training FLOPs -- eight orders of magnitude -- the reducible part of the loss falls from
{gaps[1e19]:.3f} to {gaps[1e27]:.3f}, a factor of {gaps[1e19] / gaps[1e27]:.1f}, and the
irreducible floor's share of the total rises to
{E_FLOOR / (E_FLOOR + gaps[1e27]):.1%}.

**The exponents govern a shrinking share of the number being reported.** A curve fitted where
the reducible term dominates is being extrapolated into a region where it does not, which is the
second listing's problem.

The third table prices the exponent directly. Each halving of the reducible loss costs roughly
{needs[0.25] / needs[0.5]:.0f} times the compute of the previous one. That multiple is set by
`a` and `b` and by nothing else -- it is the same whether the constants are large or small --
and it is why an order-of-magnitude compute increase is a routine expectation rather than a
breakthrough.

The fourth table is the one to act on, and it is not in the original framing at all.

The compute-optimal split minimises training loss per *training* FLOP. A deployed model also
costs about `2N` FLOPs per served token, and that term grows with the product's success rather
than with the training run. Holding the loss target at {TARGET:.2f} and varying serving volume
from {1e9:.0e} to {1e16:.0e} tokens, the cheapest model shrinks from {n_small:.2e} to
{n_large:.2e} parameters -- **a factor of {n_small / n_large:.0f}, driven entirely by how much
the thing is used** (eq:the-training-optimum-is-not-the-deployment-optimum).

The last table prices the mistake. Training a compute-optimal model and then serving it costs
{overspend['a research artefact']:.1f} times an inference-aware design for a research artefact
and **{overspend['a consumer product']:.0f} times** for a consumer product, at identical
quality, and the gap widens with every user.

**"Compute-optimal" is a claim about a training run, not about a system**, and the two answers
diverge in exactly the direction a successful product moves.

There is a corollary worth stating plainly: cite:hoffmann2022chinchilla's result made models
smaller for a given budget, and inference-awareness pushes further in the same direction --
smaller models trained past the training-optimal point, deliberately. The overspend column is
the argument, and it grows monotonically with serving volume.""")

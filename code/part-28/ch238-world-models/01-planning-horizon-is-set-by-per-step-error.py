# -*- coding: utf-8 -*-
# Extracted from: Chapter 238 — World Models and Embodied AI
# Source: src/.../ch238-world-models.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A learned world model is a rollout budget, and per-step error sets it.

The appeal of a world model is that it lets a system plan: imagine a sequence of actions, predict
where they lead, choose (cite:ha2018worldmodels). The cost is that predictions are made by a
learned model whose errors compound step by step.

So the usable planning horizon is not a design choice. It falls out of the per-step error and the
tolerance the task allows (eq:planning-horizon-is-set-by-per-step-error).

And the value of planning at all is bounded by that horizon: if the decisions that matter lie
beyond it, a better planner buys nothing
(eq:model-based-gain-is-bounded-by-the-usable-horizon).
"""
import math

TOLERANCE = 0.35        # state error beyond which a plan is not actionable
CHAOS = 1.06            # per-step amplification of an existing error


def divergence(eps, steps):
    """Accumulated state error after a rollout, with amplification."""
    if abs(CHAOS - 1.0) < 1e-9:
        return eps * steps
    return eps * (CHAOS ** steps - 1.0) / (CHAOS - 1.0)


def horizon(eps, tol=TOLERANCE, cap=400):
    h = 0
    while h < cap and divergence(eps, h + 1) <= tol:
        h += 1
    return h


print("How far a rollout stays usable.")
print()
print(f"{'per-step error':>17}{'error at 5 steps':>19}{'at 20':>10}{'at 50':>10}"
      f"{'usable horizon':>17}")
print("-" * 73)
hz = {}
for eps in (0.001, 0.003, 0.010, 0.030, 0.060, 0.100):
    h = horizon(eps)
    hz[eps] = h
    print(f"{eps:>17.3f}{divergence(eps, 5):>19.4f}{divergence(eps, 20):>10.3f}"
          f"{divergence(eps, 50):>10.2f}{h:>17}")

print()
print(f"a 10x reduction in per-step error, from {0.030:.3f} to {0.003:.3f},")
print(f"buys {hz[0.003] - hz[0.030]} extra steps -- {hz[0.003] / hz[0.030]:.1f}x the horizon")
print("(eq:planning-horizon-is-set-by-per-step-error)")

print()
print()
print("Which is a poor exchange rate, because error is expensive to reduce.")
print()
print(f"{'per-step error':>17}{'training data needed':>23}{'relative cost':>16}"
      f"{'horizon':>10}{'cost per extra step':>22}")
print("-" * 88)
BASE_EPS, BASE_DATA = 0.100, 1.0
prev_h, prev_c = None, None
for eps in (0.100, 0.060, 0.030, 0.010, 0.003, 0.001):
    data = BASE_DATA * (BASE_EPS / eps) ** 2.1
    h = hz[eps]
    if prev_h is None:
        print(f"{eps:>17.3f}{data:>23.1f}{data:>16.1f}{h:>10}{'--':>22}")
    else:
        per = (data - prev_c) / max(h - prev_h, 1)
        print(f"{eps:>17.3f}{data:>23.1f}{data:>16.1f}{h:>10}{per:>22.1f}")
    prev_h, prev_c = h, data

print()
print("The last column is what a longer horizon costs, and it rises fast.")

print()
print()
print("Now what a horizon is worth: how far ahead the decisions actually are.")
print()
TASKS = [
    ("grasp a rigid object",       3,  0.030),
    ("pour a liquid",             11,  0.045),
    ("assemble two parts",        24,  0.055),
    ("navigate a cluttered room",  40, 0.020),
    ("load a dishwasher",        130,  0.060),
    ("cook a meal",              900,  0.080),
]
print(f"{'task':>28}{'decision depth':>17}{'per-step error':>17}"
      f"{'usable horizon':>17}{'covered?':>11}{'model-based gain':>19}")
print("-" * 109)
gains = {}
for name, depth, eps in TASKS:
    h = horizon(eps)
    frac = min(1.0, h / depth)
    gain = 0.42 * frac ** 0.7
    gains[name] = (h, frac, gain)
    print(f"{name:>28}{depth:>17}{eps:>17.3f}{h:>17}"
          f"{frac:>11.1%}{gain:>19.4f}")

print()
print(f"planning helps most where the horizon covers the decision:")
print(f"`{max(gains, key=lambda n: gains[n][2])}` at {max(g[2] for g in gains.values()):.4f}")
print(f"and least where it does not: `{min(gains, key=lambda n: gains[n][2])}`"
      f" at {min(g[2] for g in gains.values()):.4f}")
print("(eq:model-based-gain-is-bounded-by-the-usable-horizon)")

print()
print()
print("What buys horizon other than a better model.")
print()
FIXES = [
    ("10x less per-step error",       hz[0.003] / hz[0.030], 128.0, "data and compute"),
    ("replan every step",             2.30,                   4.0, "latency budget"),
    ("coarser state, looser tolerance", 1.85,                 1.1, "less precise plans"),
    ("hierarchical actions",          3.60,                   1.4, "an action abstraction"),
    ("closed-loop sensing",           4.10,                   1.9, "sensors and bandwidth"),
]
print(f"{'approach':>34}{'horizon multiple':>19}{'cost multiple':>16}"
      f"{'horizon per unit cost':>24}{'what it needs':>24}")
print("-" * 117)
for name, mult, cost, needs in FIXES:
    print(f"{name:>34}{mult:>19.2f}x{cost:>15.1f}x{mult / cost:>24.2f}{needs:>24}")

best_fix = max(FIXES, key=lambda f: f[1] / f[2])
print()
print(f"best horizon per unit cost: {best_fix[0]} at {best_fix[1] / best_fix[2]:.2f}")
print(f"a better model is {(hz[0.003] / hz[0.030]) / 128.0:.3f} -- the worst row by far")

print()
print()
print("And the asymmetry that makes embodiment different from prediction.")
print()
OUTCOMES = [
    ("the plan was right",                 0.71, 0.0,     "nothing"),
    ("the plan was wrong, recoverable",    0.24, 12.0,    "retry"),
    ("the plan was wrong, object damaged", 0.04, 1_400.0, "replace it"),
    ("the plan was wrong, unsafe",         0.01, 90_000.0, "an incident"),
]
print(f"{'outcome':>38}{'probability':>14}{'cost':>12}{'expected':>13}{'response':>14}")
print("-" * 91)
exp = 0.0
for name, p, cost, resp in OUTCOMES:
    exp += p * cost
    print(f"{name:>38}{p:>14.2f}{cost:>12,.1f}{p * cost:>13,.1f}{resp:>14}")
print("-" * 91)
print(f"{'EXPECTED COST PER ATTEMPT':>38}{'':>14}{'':>12}{exp:>13,.1f}")

tail = sum(p * c for n, p, c, r in OUTCOMES if c >= 1_000)
print()
print(f"{tail / exp:.0%} of the expected cost sits in the"
      f" {sum(p for n, p, c, r in OUTCOMES if c >= 1000):.0%} of attempts")
print("that damage something")

print(f"""
The first table is the constraint that organises the whole subject. A rollout's error accumulates
and amplifies, so a model with per-step error {0.030:.3f} stays inside a {TOLERANCE:.2f}
tolerance for {hz[0.030]} steps and one with {0.003:.3f} lasts {hz[0.003]}.

**The usable horizon is not a design parameter; it falls out of the per-step error**
(eq:planning-horizon-is-set-by-per-step-error). This is ch:rsn-cot's
`per-step-error-compounding` in a state space rather than a token sequence, with an amplification
term because physical systems diverge as well as accumulate.

The exchange-rate table is why this is hard. Reducing per-step error is roughly quadratic in
data, so a 10x reduction costs on the order of {(0.100 / 0.010) ** 2.1:.0f}x the data and buys
{hz[0.003] - hz[0.030]} extra steps. The cost per additional step of horizon rises at every row.

**Horizon is the expensive axis**, and a research result that improves per-step error by a
factor of two is buying a handful of steps.

The task table asks what those steps are worth, and the answer depends entirely on the task's
decision depth (eq:model-based-gain-is-bounded-by-the-usable-horizon). `grasp a rigid object`
needs {3} steps of lookahead and gets {gains['grasp a rigid object'][0]}: fully covered, and
planning delivers {gains['grasp a rigid object'][2]:.4f}. `cook a meal` needs {900} and gets
{gains['cook a meal'][0]}: **{gains['cook a meal'][1]:.1%} covered**, and planning delivers
{gains['cook a meal'][2]:.4f}.

That is the honest summary of where model-based methods work. **They work where the horizon
covers the decision**, and the tasks people find most impressive are exactly the ones where it
does not -- which is why demonstrations cluster at short-horizon manipulation and long-horizon
tasks are done by decomposition rather than by planning.

The fixes table is the practical consequence, and it is unusually clear-cut.
`{best_fix[0]}` buys {best_fix[1]:.2f}x the horizon for {best_fix[2]:.1f}x the cost --
{best_fix[1] / best_fix[2]:.2f} per unit. A better model buys
{hz[0.003] / hz[0.030]:.1f}x for {128.0:.0f}x, which is
{(hz[0.003] / hz[0.030]) / 128.0:.3f} per unit and the worst row in the table by two orders of
magnitude.

**Every cheap way to extend the horizon works by not needing it**: replanning discards the tail
of the rollout, hierarchy shortens the sequence, closed-loop sensing replaces prediction with
measurement. The expensive way is the one that makes the model better at predicting, and it is
the one the research literature optimises.

The last table is what separates embodiment from prediction. A wrong plan in a token sequence
costs a retry. A wrong plan in the physical world costs a retry
{0.24:.0%} of the time, an object {0.04:.0%} of the time, and an incident {0.01:.0%} of the time
-- and **{tail / exp:.0%} of the expected cost sits in the
{sum(p for n, p, c, r in OUTCOMES if c >= 1000):.0%} of attempts that damage something.**

That is ch:ops-deployment's `reversibility-is-a-design-property` and ch:sec-tool-abuse's
permanence result arriving in a domain where they are not design choices at all. **You cannot
make a dropped object recoverable by changing the architecture**, which is why the data
economics of the second listing look nothing like the data economics of anything else in this
book.""")

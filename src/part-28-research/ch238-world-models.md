---
id: res-world-models
number: 238
part: XXVIII
tier: full
status: draft
requires: [per-step-error-compounding, reversibility-is-a-design-property,
           collapse-rate-is-set-by-the-real-data-fraction, coverage-is-a-union-not-a-sum]
provides: [planning-horizon-is-set-by-per-step-error,
           model-based-gain-is-bounded-by-the-usable-horizon,
           embodied-data-is-rate-limited-not-cost-limited,
           transfer-discount-sets-the-sim-real-mixture]
citations: [ha2018worldmodels, driess2023palme, tong2022videomae, ravi2024sam2]
---

## 1. Learning Objectives

By the end of this chapter you will be able to compute a learned dynamics model's usable planning
horizon from its per-step error and the task's tolerance; show that the value of model-based
planning is bounded by whether that horizon covers the task's decision depth; rank
horizon-extending techniques by horizon per unit cost; explain why embodied data is limited by
wall-clock rather than budget; and compute how a simulator's transfer coefficient sets the size
of the robot fleet you need.

## 2. Why This Matters

A world model earns its keep by letting a system plan: imagine actions, predict where they lead,
choose ({{cite:ha2018worldmodels}}). The predictions come from a learned model, and its errors
compound.

At per-step error 0.030 a rollout stays inside tolerance for **9** steps; at 0.003, **35**
({{eq:planning-horizon-is-set-by-per-step-error}}). That 10× error reduction costs roughly
**126×** the training data and buys **3.9×** the horizon.

Whether the horizon is worth anything depends on the task. Grasping needs 3 steps of lookahead
and gets 9 — **100% covered**, planning worth **0.4200**. Cooking a meal needs 900 and gets 4 —
**0.4% covered**, worth **0.0095**
({{eq:model-based-gain-is-bounded-by-the-usable-horizon}}).

And every cheap way to extend the horizon works by *not needing it*: hierarchical actions buy
**2.57** horizon per unit cost, closed-loop sensing **2.16**, a better model **0.03**.

The other half is data. Embodied experience is bought with wall-clock, not money: a robot
produces **140** trajectories a day, so 50 robots need **51 months** for a residual of 10.8
million ({{eq:embodied-data-is-rate-limited-not-cost-limited}}). And that residual is set by the
simulator's transfer coefficient — **59%** of the requirement at transfer 0.05, **14%** at 0.60
({{eq:transfer-discount-sets-the-sim-real-mixture}}).

## 3. Prerequisites

{{eq:chain-accuracy-compounds}} from {{ch:rsn-cot}} is this chapter's rollout result in a token
sequence rather than a state space; the addition here is an amplification term, because physical
systems diverge as well as accumulate.

{{eq:reversibility-is-a-design-property}} from {{ch:ops-deployment}} stops being a design
property when the actuator is physical, which is what makes the failure-cost table in
{{sec:9-practical-example}} different in kind.

{{eq:collapse-rate-is-set-by-the-real-data-fraction}} from {{ch:res-continual}} is the same
structure as the simulator's coverage cap: a source that is not the world eventually stops
supplying information about it.

{{eq:coverage-is-a-union-not-a-sum}} from {{ch:ev-framework}} is why the data sources in the
second listing have caps that do not add to one.

## 4. Intuitive Explanation

A world model is a learned function that answers "if I do this, what happens?" Given one, a
system can plan: roll a candidate action sequence forward, look at where it ends up, pick the
best one ({{cite:ha2018worldmodels}}).

The catch is that each prediction is made from the previous prediction, so errors accumulate.
Worse, physical systems amplify: a small error in a contact estimate becomes a large error in
where the object ends up. So the error after $h$ steps is not $h\epsilon$; it is $\epsilon$
compounded.

Put a tolerance on it — the state error beyond which a plan is not actionable — and you get a
horizon.

At per-step error 0.100 the model stays inside a 0.35 tolerance for **3** steps. At 0.030, **9**.
At 0.010, **19**. At 0.003, **35**. At 0.001, **53**.

**The usable horizon is not a design choice; it falls out of the per-step error**
({{eq:planning-horizon-is-set-by-per-step-error}}). That is {{ch:rsn-cot}}'s
{{eq:chain-accuracy-compounds}} in a state space, with amplification added.

Now the exchange rate, because per-step error is what research improves. Model error falls
roughly with the square root of data, so a 10× reduction costs about **126×** the data — and
buys **26 extra steps**, from 9 to 35.

The cost per additional step of horizon rises at every row: 1.0 at the first improvement, 2.4,
11.3, 90.7, **792.8**. **Horizon is the expensive axis**, and a research result that halves
per-step error is buying a handful of steps.

So what is a step worth? That depends entirely on how far ahead the task's decisions actually
are.

Grasping a rigid object needs about 3 steps of lookahead, and a model with 0.030 per-step error
gives 9 — **fully covered**, and planning is worth **0.4200**. Pouring a liquid needs 11 and
gets 6: 54.5% covered, worth 0.2748. Assembling two parts needs 24 and gets 5: 20.8%, worth
0.1401. Loading a dishwasher needs 130 and gets 5: 3.8%, worth 0.0429. Cooking a meal needs 900
and gets 4: **0.4% covered**, worth **0.0095**
({{eq:model-based-gain-is-bounded-by-the-usable-horizon}}).

That is the honest summary of where model-based methods work: **where the horizon covers the
decision.** And it explains a pattern in the literature that otherwise looks like fashion —
demonstrations cluster at short-horizon manipulation, and long-horizon tasks are done by
decomposition rather than by planning, because planning has nothing to offer at 0.4% coverage.

Which raises the obvious question: how do you get more horizon?

Rank the options by horizon per unit cost. A 10× better model: 3.89× the horizon for 128× the
cost — **0.03 per unit**. Replanning every step: 2.30× for 4× — 0.57. Coarser state with looser
tolerance: 1.85× for 1.1× — 1.68. Hierarchical actions: 3.60× for 1.4× — **2.57**. Closed-loop
sensing: 4.10× for 1.9× — 2.16.

Look at what the winning rows have in common. **Every cheap way to extend the horizon works by
not needing it.** Replanning discards the tail of the rollout and never relies on it. Hierarchy
shortens the sequence by making each step bigger. Closed-loop sensing replaces prediction with
measurement.

The expensive way — 0.03 per unit, two orders of magnitude worse than the best row — is the one
that makes the model better at predicting, and it is the one the research literature optimises.

There is one more thing that separates embodiment from every other subject in this book, and it
is not about horizons.

A wrong plan in a token sequence costs a retry. A wrong plan in the physical world costs a retry
24% of the time, a damaged object **4%** of the time, and a safety incident **1%** of the time.
Expected cost per attempt: 65.9 units — and **99% of it sits in the 5% of attempts that damage
something.**

That is {{ch:ops-deployment}}'s {{eq:reversibility-is-a-design-property}} and
{{ch:sec-tool-abuse}}'s permanence result arriving in a domain where they are not design choices.
**You cannot make a dropped object recoverable by changing the architecture.** Which is why the
data economics of embodiment look nothing like the data economics of anything else here.

Everything else in this book buys data with money. Physical interaction is bought with
wall-clock.

Compare the sources. A real robot: 14 trajectories per unit-hour, $9.40 each, transfer 1.00.
Simulation: 4,200 per hour, $0.011, transfer **0.24**. Internet video: 90,000 per hour, $0.0004,
transfer **0.06**. Human demonstration video: 30 per hour, $4.20, transfer 0.11.

Per real-equivalent trajectory per dollar: internet video 150.0, simulation 21.8, a real robot
0.11, human demonstration video 0.03.

On cost alone the answer is obvious, and it is wrong — because of the coverage cap.

**Every non-real source has a share of the requirement beyond which its residual gap is
systematic and more of it adds nothing.** Simulation caps at 55%, internet video at 18%. Those
caps exist because the mismatch between a simulator and the world is not noise; it is a set of
specific unmodelled effects, and drawing more samples from the same simulator reproduces them
exactly. That is {{ch:res-continual}}'s
{{eq:collapse-rate-is-set-by-the-real-data-fraction}} in a different costume.

So buy in order of real-equivalent per dollar: internet video for 18%, simulation for 55%, and
the remaining **27%** — **10,800,000 trajectories** — has to be real. That residual is **99% of
the money**.

Note which source never gets used. Human demonstration video *feels* like cheap real-world data
and is the most expensive source in the table per real-equivalent trajectory: **0.026** against
a real robot's **0.106**. A human demonstration answers a different question than a robot
attempt does, and the transfer coefficient charges for the difference.

Now the constraint that actually binds. A robot produces **140** trajectories a day and no
budget changes that number. Ten robots need **254 months**. Fifty need **51**. Two hundred need
**13**. A thousand need **2.5** — and cost $78,000,000 in hardware before a single trajectory is
collected ({{eq:embodied-data-is-rate-limited-not-cost-limited}}).

**This is the only chapter in this book where the binding constraint is wall-clock**, and it
changes what optimisation means. Every other domain's answer to "we need more data" is a budget
line. Here it is a fleet, a building, and a year.

Which puts the leverage in an unexpected place. Sweep the simulator's transfer coefficient. At
0.05, the residual real share is **59%** and a thousand robots need 5.5 months. At 0.12, 44% and
4.2 months. At 0.24, 27% and 2.5. At 0.40 and above, **14%** and 1.3 —
**a factor of 4.2 from one coefficient**, and the residual bottoms out at 5% because some
experience has to be real however good the simulator is
({{eq:transfer-discount-sets-the-sim-real-mixture}}).

So the highest-leverage work in embodied learning is not the policy and not the world model. It
is whatever raises the fraction of simulated experience that transfers, **because that fraction
divides the fleet size.**

What raises it? System identification: 0.11 of transfer for 1.4× the cost — **0.079 per unit**,
the best row in the table — and what it needs is *measurements* of the real system rather than
trajectories from it. Measuring friction and inertia is cheap in exactly the currency that is
scarce.

Contact and friction modelling: 0.16 for 2.6×. Domain randomisation: 0.09 for 1.2×. A better
renderer: 0.03 for 1.9× — the worst row, and the one that looks most like progress.

And then the row that is not cheap. Fine-tuning on real data gives 0.14, the second-largest
single gain, at 0.045 per unit — and it consumes exactly the resource the exercise is short of.
**The best-known way to close the reality gap requires the thing the reality gap is preventing
you from getting.** That circularity is at the centre of the field.

End to end: all real with a thousand robots takes 9.4 months and $454,000,000. Sim-heavy with
two hundred robots takes 12.7 months and $118,176,333. And the same two hundred robots with
transfer raised to 0.62 takes **6.6 months and $68,770,581** — half the time of the same fleet
without the gap work, and the cheapest strategy on the page.

**A better simulator substitutes for a bigger fleet, at a fraction of the cost**, through a
single coefficient most teams never measure.

## 5. Formal Explanation

**Rollout divergence.** With per-step error $\epsilon$ and amplification $\gamma > 1$,
accumulated error after $h$ steps is
$D(h) = \epsilon\,(\gamma^h - 1)/(\gamma - 1)$. The usable horizon is
$H = \max\{h : D(h) \le \tau\}$, which grows like $\log_\gamma(1 + \tau(\gamma-1)/\epsilon)$ —
**logarithmically in $1/\epsilon$**. That is the formal reason the exchange rate is poor: linear
reductions in error buy logarithmic gains in horizon.

**Planning value.** With decision depth $d$ and usable horizon $H$, the covered fraction is
$\min(1, H/d)$ and the value of planning is increasing and concave in it. Value is therefore
bounded above by the coverage, regardless of planner quality: a perfect planner over a 4-step
horizon on a 900-step task is still planning over 0.4% of the decision.

**Rate limiting.** A source supplies $r$ trajectories per unit-hour at cost $c$ with transfer
$\tau$ and coverage cap $\kappa$. Cost-limited sources are ordered by $\tau/c$; the *time* to
acquire $N$ real trajectories with $F$ units is $N/(rFh)$, independent of budget except through
$F$. When $\kappa < 1$ for every cheap source, the residual is real and its acquisition time is
the schedule.

**The fleet identity.** With residual share $\rho(\tau)$, target $N$, fleet $F$ and daily rate
$q$: $T = N\rho(\tau)/(Fq)$. So $F$ and $\rho$ are substitutes at a fixed $T$, and since
$\rho$ falls with $\tau$ while $F$ costs hardware, raising $\tau$ is a direct substitute for
capital.

## 6. Mathematical Foundation

The horizon falls out of the error:

$$H = \max\left\{h : \epsilon\,\frac{\gamma^h - 1}{\gamma - 1} \le \tau\right\} = 9 \ (\epsilon = 0.030) \ \to \ 35 \ (\epsilon = 0.003)$$ (eq:planning-horizon-is-set-by-per-step-error)

a **3.9×** horizon for **126×** the data.

Planning is worth what the horizon covers:

$$V = v_{\max}\left(\min(1, H/d)\right)^{0.7} = 0.4200 \ (d = 3) \ \to \ 0.0095 \ (d = 900)$$ (eq:model-based-gain-is-bounded-by-the-usable-horizon)

Embodied data is bought with time:

$$T = \frac{N\rho}{Fq}, \qquad q = 140 \ \text{trajectories/robot-day} \implies 51 \ \text{months at } F = 50$$ (eq:embodied-data-is-rate-limited-not-cost-limited)

And the transfer coefficient sets the residual:

$$\rho(\tau) = \max\left(0.05,\ 1 - \kappa(\tau) - 0.18\right) = 59\% \ (\tau = 0.05) \ \to \ 14\% \ (\tau = 0.60)$$ (eq:transfer-discount-sets-the-sim-real-mixture)

## 7. Internal Mechanics

The logarithmic horizon is the mechanism worth internalising, because it explains why the field
looks the way it does. Error compounds geometrically, so horizon grows with the *logarithm* of
error reduction. A team that halves per-step error — a serious research result — gains a few
steps. A team that restructures the task so each step is ten times larger gains an order of
magnitude. The second is engineering and the first is science, and the arithmetic strongly
favours the engineering.

The amplification term is what distinguishes physical rollouts from token rollouts. In a token
sequence errors accumulate but do not generally amplify: a wrong token makes the next token
harder, not exponentially harder. In contact-rich physics they do amplify, because small state
differences change *which* contacts occur, and a different contact is a discontinuity rather
than a perturbation. That is why the horizon numbers here are single or double digits where
{{ch:rsn-cot}}'s were larger.

The data-source cap has a mechanism worth separating from the transfer coefficient it sits
beside. Transfer says how much one simulated trajectory is worth; the cap says how much
simulated experience *in total* can substitute. They are different because the simulator's error
is systematic — it gets the same things wrong every time — so beyond some point additional
samples are re-drawing from a distribution whose difference from reality is fixed. This is the
same shape as {{ch:res-continual}}'s diversity contraction and as
{{eq:coverage-is-a-union-not-a-sum}}: overlapping sources do not sum.

The rate limit deserves its own statement because it inverts a habit. In every other chapter,
"more data" is a budget question, and the useful reflex is to ask what a marginal dollar buys.
Here the useful reflex is to ask what a marginal *day* buys, and the answer is fixed by the
fleet. Teams that carry the budget reflex into embodiment consistently over-invest in data
acquisition capacity and under-invest in the things that reduce how much acquisition is needed.

Finally, the substitution between simulator quality and fleet size is exact enough to be a
planning identity. $T = N\rho(\tau)/(Fq)$ — so a transfer improvement that halves $\rho$ has
precisely the effect of doubling the fleet, at a small fraction of the capital. The reason this
is not the default strategy is that $\tau$ is rarely measured, so the substitution is not visible
to anyone deciding a budget.

## 8. Implementation

The first listing prices the horizon.

```python {tier=A name=C:/Users/MANAVB~1/AppData/Local/Temp/claude/C--Github-LaughingBuddha/30a87753-43a8-48c3-8378-261faf976dbb/scratchpad/kf1}
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
```

## 9. Practical Example

How far a rollout stays usable:

```
   per-step error   error at 5 steps     at 20     at 50   usable horizon
-------------------------------------------------------------------------
            0.001             0.0056     0.037      0.29               53
            0.003             0.0169     0.110      0.87               35
            0.010             0.0564     0.368      2.90               19
            0.030             0.1691     1.104      8.71                9
            0.100             0.5637     3.679     29.03                3
```

**A 10× error reduction buys 3.9× the horizon**
({{eq:planning-horizon-is-set-by-per-step-error}}).

```
   per-step error   training data needed   relative cost   horizon   cost per extra step
----------------------------------------------------------------------------------------
            0.100                    1.0             1.0         3                    --
            0.030                   12.5            12.5         9                   2.4
            0.010                  125.9           125.9        19                  11.3
            0.003                 1577.8          1577.8        35                  90.7
            0.001                15848.9         15848.9        53                 792.8
```

```
                        task   decision depth   per-step error   usable horizon   covered?   model-based gain
-------------------------------------------------------------------------------------------------------------
        grasp a rigid object                3            0.030                9     100.0%             0.4200
               pour a liquid               11            0.045                6      54.5%             0.2748
          assemble two parts               24            0.055                5      20.8%             0.1401
           load a dishwasher              130            0.060                5       3.8%             0.0429
                 cook a meal              900            0.080                4       0.4%             0.0095
```

**Planning is worth what the horizon covers**
({{eq:model-based-gain-is-bounded-by-the-usable-horizon}}).

```
                          approach   horizon multiple   cost multiple   horizon per unit cost           what it needs
---------------------------------------------------------------------------------------------------------------------
           10x less per-step error               3.89x          128.0x                    0.03        data and compute
                 replan every step               2.30x            4.0x                    0.57          latency budget
   coarser state, looser tolerance               1.85x            1.1x                    1.68      less precise plans
              hierarchical actions               3.60x            1.4x                    2.57   an action abstraction
              closed-loop sensing                4.10x            1.9x                    2.16   sensors and bandwidth
```

**Every cheap way to extend the horizon works by not needing it.**

```
                               outcome   probability        cost     expected      response
-------------------------------------------------------------------------------------------
                    the plan was right          0.71         0.0          0.0       nothing
       the plan was wrong, recoverable          0.24        12.0          2.9         retry
   the plan was wrong, object damaged           0.04     1,400.0         56.0    replace it
             the plan was wrong, unsafe         0.01    90,000.0        900.0    an incident
```

**99% of the expected cost sits in the 5% of attempts that damage something.**

The second listing prices the data.

```python {tier=A name=C:/Users/MANAVB~1/AppData/Local/Temp/claude/C--Github-LaughingBuddha/30a87753-43a8-48c3-8378-261faf976dbb/scratchpad/kf2}
"""Embodied data is rate-limited, not cost-limited, and the transfer coefficient sets the fleet.

Everything else in this book buys data with money. Physical interaction is bought with
wall-clock: a robot performs a bounded number of attempts per hour, and no budget changes that
number (eq:embodied-data-is-rate-limited-not-cost-limited).

Simulation and video escape the rate limit and pay a transfer discount instead -- a simulated
trajectory is worth some fraction of a real one, and that fraction has a ceiling because the
residual gap is systematic rather than random.

Which makes the transfer coefficient the parameter that decides how large a robot fleet has to
be (eq:transfer-discount-sets-the-sim-real-mixture).
"""
import math

TARGET = 40_000_000        # real-equivalent trajectories for the capability
HOURS_PER_DAY = 10.0

# (source, trajectories per hour per unit, $ per trajectory, transfer coefficient,
#  share of the requirement it can cover before the residual gap binds)
SOURCES = [
    ("a real robot",             14.0,   9.40,  1.00, 1.00),
    ("teleoperation",            22.0,  31.00,  1.00, 1.00),
    ("simulation",            4_200.0,   0.011, 0.24, 0.55),
    ("internet video",       90_000.0,   0.0004, 0.06, 0.18),
    ("human demonstration video", 30.0,  4.20,  0.11, 0.09),
]

print("What each source of experience actually supplies.")
print()
print(f"{'source':>28}{'traj / unit-hour':>19}{'$ / trajectory':>17}"
      f"{'transfer':>11}{'real-equiv / $':>17}{'coverage cap':>15}")
print("-" * 107)
for name, rate, cost, tau, cap in SOURCES:
    print(f"{name:>28}{rate:>19,.0f}{cost:>17.4f}{tau:>11.2f}"
          f"{tau / cost:>17,.1f}{cap:>15.0%}")

print()
print(f"a simulated trajectory is worth {0.24:.2f} of a real one and costs"
      f" {9.40 / 0.011:,.0f}x less")
print(f"which is {(0.24 / 0.011) / (1.00 / 9.40):,.0f}x the real-equivalent per dollar")

print()
print()
print("So buy real-equivalent experience in order of what it costs.")
print()
print(f"target: {TARGET:,} real-equivalent trajectories")
print()
print(f"{'source':>28}{'real-equiv / $':>17}{'coverage taken':>17}"
      f"{'real-equiv supplied':>22}{'trajectories':>17}{'cost':>16}")
print("-" * 117)
remaining = 1.0
plan, spend = {}, 0.0
for name, rate, cost, tau, cap in sorted(SOURCES, key=lambda s: -s[3] / s[2]):
    take = min(cap, remaining)
    remaining -= take
    re = TARGET * take
    traj = re / tau
    c = traj * cost
    spend += c
    plan[name] = (take, re, traj, c)
    print(f"{name:>28}{tau / cost:>17,.2f}{take:>17.0%}{re:>22,.0f}"
          f"{traj:>17,.0f}{c:>16,.0f}")
print("-" * 117)
print(f"{'TOTAL':>28}{'':>17}{1.0 - remaining:>17.0%}{TARGET * (1.0 - remaining):>22,.0f}"
      f"{'':>17}{spend:>16,.0f}")

REAL_SHARE = plan["a real robot"][0]
REAL_TRAJ = plan["a real robot"][2]
real_cost = plan["a real robot"][3]
print()
print(f"{REAL_SHARE:.0%} of the requirement -- {REAL_TRAJ:,.0f} trajectories -- has to be real")
print(f"and that share is {real_cost / spend:.0%} of the money")
print(f"`human demonstration video` supplies {plan['human demonstration video'][0]:.0%}:"
      f" at {0.11 / 4.20:.3f} real-equivalent per dollar it is dominated")

print()
print()
print("But money is not what those trajectories cost. Wall-clock is.")
print()
PER_ROBOT_DAY = 14.0 * HOURS_PER_DAY
print(f"{'fleet size':>13}{'trajectories / day':>21}{'days to the residual':>23}"
      f"{'months':>10}{'hardware $':>15}")
print("-" * 82)
ROBOT_COST = 78_000.0
fleet_months = {}
for robots in (10, 50, 200, 1_000, 5_000, 20_000):
    per_day = robots * PER_ROBOT_DAY
    days = REAL_TRAJ / per_day
    fleet_months[robots] = days / 30.4
    print(f"{robots:>13,}{per_day:>21,.0f}{days:>23,.0f}"
          f"{days / 30.4:>10,.1f}{robots * ROBOT_COST:>15,.0f}")

print()
print(f"{50:,} robots need {fleet_months[50]:,.0f} months;"
      f" {5_000:,} need {fleet_months[5000]:,.1f}")
print("(eq:embodied-data-is-rate-limited-not-cost-limited)")

print()
print()
print("And the transfer coefficient decides how big that residual is.")
print()
print(f"{'simulation transfer':>21}{'sim coverage cap':>19}{'residual real share':>22}"
      f"{'real trajectories':>20}{'months at 1,000 robots':>25}")
print("-" * 107)
tau_rows = {}
for tau in (0.05, 0.12, 0.24, 0.40, 0.60, 0.80):
    cap = min(0.68, 0.55 * (tau / 0.24) ** 0.55)
    residual = max(0.05, 1.0 - cap - 0.18)
    traj = TARGET * residual
    months = traj / (1_000 * PER_ROBOT_DAY) / 30.4
    tau_rows[tau] = (cap, residual, traj, months)
    print(f"{tau:>21.2f}{cap:>19.0%}{residual:>22.0%}{traj:>20,.0f}"
          f"{months:>25,.1f}")

print()
print(f"transfer {0.12:.2f} needs {tau_rows[0.12][3]:,.0f} months at 1,000 robots;")
print(f"transfer {0.60:.2f} needs {tau_rows[0.60][3]:,.1f}")
print(f"a factor of {tau_rows[0.12][3] / max(tau_rows[0.60][3], 1e-9):,.1f}"
      f" from one coefficient")
print("(eq:transfer-discount-sets-the-sim-real-mixture)")

print()
print()
print("What actually moves the transfer coefficient.")
print()
GAP = [
    ("better renderer",              0.03, 1.9,  "graphics work"),
    ("domain randomisation",         0.09, 1.2,  "more sim compute"),
    ("system identification",        0.11, 1.4,  "real measurements"),
    ("fine-tune on real data",       0.14, 3.1,  "real trajectories"),
    ("contact and friction modelling", 0.16, 2.6, "physics engineering"),
    ("all of them",                  0.38, 10.2, "everything above"),
]
print(f"{'intervention':>34}{'transfer gain':>16}{'cost multiple':>16}"
      f"{'gain per unit cost':>22}{'what it needs':>22}")
print("-" * 110)
for name, gain, cost, needs in GAP:
    print(f"{name:>34}{gain:>16.2f}{cost:>15.1f}x{gain / cost:>22.3f}{needs:>22}")

best_gap = max(GAP[:-1], key=lambda g: g[1] / g[2])
print()
print(f"best gain per unit cost: {best_gap[0]} at {best_gap[1] / best_gap[2]:.3f}")
print(f"note that `fine-tune on real data` needs the thing that is rate-limited")

print()
print()
print("Three strategies, end to end.")
print()
STRATS = [
    ("all real, 1,000 robots",   0.00, 1_000),
    ("sim-heavy, 200 robots",    0.24,   200),
    ("sim-heavy, 1,000 robots",  0.24, 1_000),
    ("gap closed, 200 robots",   0.62,   200),
]
print(f"{'strategy':>28}{'sim transfer':>15}{'robots':>10}"
      f"{'real trajectories':>20}{'months':>10}{'total $':>16}")
print("-" * 99)
for name, tau, robots in STRATS:
    if tau == 0.0:
        residual, sim_cost = 1.0, 0.0
    else:
        cap = min(0.68, 0.55 * (tau / 0.24) ** 0.55)
        residual = max(0.05, 1.0 - cap - 0.18)
        sim_cost = TARGET * cap / tau * 0.011 + TARGET * 0.18 / 0.06 * 0.0004
    traj = TARGET * residual
    months = traj / (robots * PER_ROBOT_DAY) / 30.4
    total = traj * 9.40 + sim_cost + robots * ROBOT_COST
    print(f"{name:>28}{tau:>15.2f}{robots:>10,}{traj:>20,.0f}"
          f"{months:>10,.1f}{total:>16,.0f}")

print(f"""
The source table is the fact that makes embodied learning a different subject. A simulated
trajectory is worth {0.24:.2f} of a real one and costs {9.40 / 0.011:,.0f} times less, which is
{(0.24 / 0.011) / (1.00 / 9.40):,.0f} times the real-equivalent per dollar. On cost alone the
answer is obvious and it is also wrong, because of the last column.

Every non-real source has a **coverage cap**: a share of the requirement beyond which its
residual gap is systematic and more of it adds nothing. Simulation caps at {0.55:.0%}, internet
video at {0.18:.0%}. Those caps exist because the mismatch between a simulator and the world is
not noise -- it is a set of specific unmodelled effects, and drawing more samples from the same
simulator reproduces them exactly.

The plan table buys in order of real-equivalent per dollar and reports what is left.
**{REAL_SHARE:.0%} of the requirement -- {REAL_TRAJ:,.0f} trajectories -- has to be real**, and
that residual is {real_cost / spend:.0%} of the money.

`human demonstration video` is worth a note. It feels like cheap real-world data and it is the
most expensive source in the table per real-equivalent trajectory --
{0.11 / 4.20:.3f} against a real robot's {1.00 / 9.40:.3f} -- because a human demonstration
answers a different question than a robot attempt does, and the transfer coefficient charges for
the difference.

The fleet table is why the residual matters more than the money
(eq:embodied-data-is-rate-limited-not-cost-limited). A robot produces
{PER_ROBOT_DAY:,.0f} trajectories a day and no budget changes that. Fifty robots need
**{fleet_months[50]:,.0f} months**. A thousand need {fleet_months[1000]:,.1f}. Five thousand need
{fleet_months[5000]:,.1f} -- and cost {5_000 * ROBOT_COST:,.0f} in hardware before a single
trajectory is collected.

**This is the only chapter in this book where the binding constraint is wall-clock**, and it
changes what an optimisation looks like. Every other domain's answer to "we need more data" is a
budget line. Here it is a fleet, a building, and a year.

The transfer table is where the leverage is
(eq:transfer-discount-sets-the-sim-real-mixture). At transfer {0.12:.2f} the residual is
{tau_rows[0.12][1]:.0%} and a thousand robots need {tau_rows[0.12][3]:,.0f} months. At
{0.60:.2f} the residual is {tau_rows[0.60][1]:.0%} and the same fleet needs
{tau_rows[0.60][3]:,.1f} -- **a factor of
{tau_rows[0.12][3] / max(tau_rows[0.60][3], 1e-9):,.0f} from one coefficient.**

So the highest-leverage work in embodied learning is not the policy and not the world model. It
is whatever raises the fraction of simulated experience that transfers, because that fraction
divides the fleet size.

The gap table says what does, and the winner is instructive. `{best_gap[0]}` gives
{best_gap[1]:.2f} of transfer for {best_gap[2]:.1f}x the cost -- {best_gap[1] / best_gap[2]:.3f}
per unit, the best row in the table -- and what it needs is *measurements* of the real system
rather than trajectories from it. Measuring a robot's friction and inertia is cheap in exactly
the currency that is scarce here.

Now the row that is not. `fine-tune on real data` is the second-largest single gain at
{0.14:.2f} and one of the worst per unit cost, at {0.14 / 3.1:.3f} -- and it consumes exactly the
resource the whole exercise is short of. **The best-known way to close the reality gap requires
the thing the reality gap is preventing you from getting**, which is the circularity at the
centre of this field.

The strategy table puts it together. `all real, 1,000 robots` needs {TARGET:,} real trajectories
and {TARGET / (1_000 * PER_ROBOT_DAY) / 30.4:,.1f} months at {1_000 * ROBOT_COST + TARGET * 9.40:,.0f}.
`sim-heavy, 200 robots` cuts the real requirement to {REAL_SHARE:.0%} of it and finishes in
{TARGET * REAL_SHARE / (200 * PER_ROBOT_DAY) / 30.4:,.1f} months.

The row to read twice is `gap closed, 200 robots`: the same small fleet with transfer raised to
{0.62:.2f} finishes in {TARGET * 0.14 / (200 * PER_ROBOT_DAY) / 30.4:,.1f} months -- **half the time of the
same fleet without the gap work, and the cheapest strategy on the page.**

The one thing that beats it on time is `sim-heavy, 1,000 robots` at
{TARGET * REAL_SHARE / (1_000 * PER_ROBOT_DAY) / 30.4:,.1f} months, and it costs
{(TARGET * REAL_SHARE * 9.40 + 1_000 * ROBOT_COST) / (TARGET * 0.14 * 9.40 + 200 * ROBOT_COST):.1f}
times as much to get there.

**A better simulator substitutes for a bigger fleet, at a fraction of the cost**, and it does so
through a single coefficient that most teams never measure. That is an unusual conclusion for a
data-hungry field, and it is the practical content of this listing.""")
```

```
                      source   traj / unit-hour   $ / trajectory   transfer   real-equiv / $   coverage cap
-----------------------------------------------------------------------------------------------------------
                a real robot                 14           9.4000       1.00              0.1           100%
                  simulation              4,200           0.0110       0.24             21.8            55%
              internet video             90,000           0.0004       0.06            150.0            18%
   human demonstration video                 30           4.2000       0.11              0.0             9%

                      source   real-equiv / $   coverage taken   real-equiv supplied     trajectories            cost
---------------------------------------------------------------------------------------------------------------------
              internet video           150.00              18%             7,200,000      120,000,000          48,000
                  simulation            21.82              55%            22,000,000       91,666,667       1,008,333
                a real robot             0.11              27%            10,800,000       10,800,000     101,520,000
```

**27% has to be real, and that residual is 99% of the money.**

```
   fleet size   trajectories / day   days to the residual    months     hardware $
----------------------------------------------------------------------------------
           10                1,400                  7,714     253.8        780,000
           50                7,000                  1,543      50.8      3,900,000
          200               28,000                    386      12.7     15,600,000
        1,000              140,000                     77       2.5     78,000,000
        5,000              700,000                     15       0.5    390,000,000
```

**Wall-clock, not budget** ({{eq:embodied-data-is-rate-limited-not-cost-limited}}).

```
  simulation transfer   sim coverage cap   residual real share   real trajectories   months at 1,000 robots
-----------------------------------------------------------------------------------------------------------
                 0.05                23%                   59%          23,515,896                      5.5
                 0.12                38%                   44%          17,773,557                      4.2
                 0.24                55%                   27%          10,800,000                      2.5
                 0.60                68%                   14%           5,600,000                      1.3

                      intervention   transfer gain   cost multiple    gain per unit cost         what it needs
--------------------------------------------------------------------------------------------------------------
                   better renderer            0.03            1.9x                 0.016         graphics work
             system identification            0.11            1.4x                 0.079     real measurements
            fine-tune on real data            0.14            3.1x                 0.045     real trajectories
    contact and friction modelling            0.16            2.6x                 0.062   physics engineering

                    strategy   sim transfer    robots   real trajectories    months         total $
---------------------------------------------------------------------------------------------------
      all real, 1,000 robots           0.00     1,000          40,000,000       9.4     454,000,000
       sim-heavy, 200 robots           0.24       200          10,800,000      12.7     118,176,333
     sim-heavy, 1,000 robots           0.24     1,000          10,800,000       2.5     180,576,333
      gap closed, 200 robots           0.62       200           5,600,000       6.6      68,770,581
```

**A better simulator substitutes for a bigger fleet**
({{eq:transfer-discount-sets-the-sim-real-mixture}}).

## 10. Production Considerations

Measure your per-step error and your task's decision depth, and compute coverage before choosing
a model-based approach. Below about 20% coverage, planning buys little.

Extend the horizon by not needing it. Hierarchy at 2.57 and closed-loop sensing at 2.16 beat a
better model at 0.03 by two orders of magnitude.

Replan every step wherever latency allows. It discards the part of the rollout that is wrong.

Measure your simulator's transfer coefficient. It divides your fleet size and almost nobody has
the number.

Spend on system identification before rendering. 0.079 per unit against 0.016, and it needs
measurements rather than trajectories.

Plan embodied data acquisition in months, not dollars. The fleet sets the schedule and the
schedule is the project plan.

Price the damage tail explicitly. 99% of the expected cost per attempt is in 5% of attempts, and
a safety envelope is worth more than an accuracy improvement.

Do not buy human demonstration video expecting cheap real-world data. It is the most expensive
source per real-equivalent trajectory in the table.

## 11. Common Mistakes

**Treating the planning horizon as a hyperparameter.** It falls out of the per-step error.

**Improving the dynamics model to get more horizon.** 0.03 horizon per unit cost, the worst
option available.

**Applying model-based planning to long-horizon tasks.** 0.4% coverage on a 900-step task.

**Comparing data sources on cost per trajectory.** The transfer coefficient and the coverage cap
both matter more.

**Budgeting embodied data in dollars.** The constraint is robot-days.

**Assuming more simulation always helps.** It caps at 55% because the gap is systematic.

**Treating a physical failure like a retry.** 5% of attempts hold 99% of the expected cost.

## 12. Failure Modes

**A planner that works in demo and fails on the real task.** Decision depth 24, horizon 5.

**A research programme that halves per-step error.** Two extra steps, at 4× the data.

**A fleet sized from a budget.** Fifty robots and a 51-month schedule discovered afterwards.

**A simulator improved by rendering quality.** 0.03 of transfer for 1.9× the cost, and the
contact model unchanged.

**A sim-to-real programme with no measured transfer coefficient.** The one number that divides
the fleet, and it is not instrumented.

**An unsafe outcome at 1%.** 900 of 65.9 expected units, and it is the row nobody prices until
it happens.

## 13. Alternatives

**Model-free control with closed-loop sensing.** Skips the rollout entirely; the 2.16 row, and
the right answer whenever sensing is cheaper than prediction.

**Hierarchical decomposition.** 2.57 horizon per unit cost, and the reason long-horizon tasks
are done by decomposition rather than planning.

**Teleoperation for the residual.** Transfer 1.00 at 22 trajectories per hour, and it swaps
robot-hours for human-hours — the only substitution that beats the rate limit, at $31.00 each.

**Vision-language-action models trained on passive data**
({{cite:driess2023palme}}, {{cite:tong2022videomae}}). Escapes the rate limit and pays the
transfer discount; the 18% coverage row, and it is genuinely free relative to everything else.

**Better perception rather than better dynamics** ({{cite:ravi2024sam2}}). Reduces the state
error that feeds the rollout, which multiplies through the horizon rather than adding to it.

## 14. Evaluation

Measure per-step prediction error on held-out rollouts at several horizons, and fit the
amplification term. Both parameters are needed and most teams report only the first.

Measure your task's decision depth by counting the steps between a choice and its consequence.
Then compute coverage.

A/B planning against a reactive baseline on the same task. If coverage is under 20%, expect no
difference — and if you see one, something else is going on.

Measure the transfer coefficient directly: train on simulation only, evaluate on real, and
compare against a real-data-only control at matched trajectory counts.

Instrument the outcome distribution of physical attempts, including the damage tail. The expected
cost is dominated by rows that are rare enough to be missing from a small sample.

## 15. Advanced Concepts

The rollout model treats per-step error as a constant, and it is not: error is much larger near
contacts and discontinuities than in free motion. That means the effective horizon is not a
number but a distribution over trajectories, and the relevant statistic for planning is
something like the horizon at the 10th percentile rather than the mean. **A planner that uses
the mean horizon will be over-confident exactly on the trajectories that involve contact**,
which are the ones the task is about. Nothing here models that, and it likely makes the
{{sec:9-practical-example}} coverage figures optimistic.

The transfer coefficient is treated as a scalar property of a simulator, and it is really a
property of a (simulator, task, policy) triple. A simulator that transfers well for reaching may
transfer badly for insertion, and — more awkwardly — transfer degrades as the policy improves,
because a better policy operates closer to the boundaries where the simulator is wrong. That
predicts $\tau$ falling over the course of a project, which would make the fleet-size estimate
in {{sec:9-practical-example}} optimistic in the same direction as everything else here.

There is an interaction between this chapter's two halves that neither listing models. A world
model is itself trained on trajectories, so its per-step error depends on the same rate-limited
data that the second listing budgets — and simulated trajectories train the world model with the
same transfer discount they carry everywhere else. **The horizon and the fleet size are coupled
through the data budget**, and a joint optimisation would allocate between "collect more
trajectories to reduce per-step error" and "collect more trajectories to train the policy", which
nobody does explicitly.

Finally, the damage tail interacts with data collection in a way that has no analogue elsewhere
in this book. Exploration is how a system learns, and in the physical world exploration is
exactly what produces the 5% of attempts that carry 99% of the cost. **A safety envelope that
prevents damage also prevents the data collection that would make damage less likely**, which is
a genuine trade rather than a solvable problem, and it is the reason so much of this field
happens in simulation despite the transfer discount.

## 16. Connection to Previous Chapters

{{eq:chain-accuracy-compounds}} from {{ch:rsn-cot}} is the same accumulation in a state space,
with an amplification term that makes physical horizons shorter than token ones.

{{eq:reversibility-is-a-design-property}} from {{ch:ops-deployment}} is not a design property
here — 99% of the expected cost is in outcomes no architecture can undo.

{{eq:collapse-rate-is-set-by-the-real-data-fraction}} from {{ch:res-continual}} is the
simulator's coverage cap: a source that is not the world stops supplying information about it.

{{eq:coverage-is-a-union-not-a-sum}} from {{ch:ev-framework}} is why the data sources' caps do
not add to one, and why the residual is real.

## 17. Exercises

1. Measure per-step error and amplification on your dynamics model and compute the usable
   horizon at your tolerance.

2. Count decision depth for three tasks in your domain and compute coverage for each.

3. Rank horizon-extending interventions by horizon per unit cost for your system. Where does a
   better model rank?

4. Measure your simulator's transfer coefficient against a real-data control.

5. Compute the fleet size and schedule implied by your residual real requirement.

6. Model per-step error as trajectory-dependent per {{sec:15-advanced-concepts}} and recompute
   coverage at the 10th percentile rather than the mean.

## 18. Interview Questions

1. How far ahead can this world model usefully plan, and how do you know?

2. Our dynamics model got twice as accurate. What did that buy?

3. Why do long-horizon robotic tasks get decomposed rather than planned?

4. We have budget for more data. How much faster does that make us?

5. What is your simulator's transfer coefficient, and what does it cost you?

6. Why is a dropped object a different kind of error than a wrong token?

## 19. Research Questions

1. How does per-step error vary along a trajectory, and what is the right horizon statistic for
   planning?

2. Does the sim-to-real transfer coefficient degrade as policies improve, and by how much?

3. What is the joint optimum between collecting data for the world model and for the policy?

4. Can safety envelopes be designed that preserve the exploratory value of risky attempts?

## 20. Chapter Summary

A world model is a rollout budget, and per-step error sets it.

At per-step error 0.030 a rollout stays usable for **9** steps; at 0.003, **35** — so a 10×
error reduction costs **126×** the data and buys **3.9×** the horizon
({{eq:planning-horizon-is-set-by-per-step-error}}), with the cost per additional step rising to
**792.8** at the far end. Horizon is the expensive axis.

What it is worth depends on the task's decision depth. Grasping needs 3 and gets 9 — **100%
covered**, planning worth **0.4200**. Cooking needs 900 and gets 4 — **0.4%**, worth **0.0095**
({{eq:model-based-gain-is-bounded-by-the-usable-horizon}}). And every cheap way to extend the
horizon works by not needing it: hierarchical actions **2.57** per unit cost, closed-loop sensing
**2.16**, a better model **0.03**.

Then the data, where embodiment stops resembling anything else in this book. A robot produces
**140** trajectories a day and no budget changes that: 50 robots need **51 months**, a thousand
need **2.5** and $78,000,000 of hardware
({{eq:embodied-data-is-rate-limited-not-cost-limited}}). Simulation escapes the rate limit and
pays a transfer discount — **0.24**, capped at **55%** coverage — and after the cheap sources
are exhausted, **27%** of the requirement must be real, which is **99%** of the money.

That makes the transfer coefficient the lever: residual **59%** at 0.05 against **14%** at 0.60
({{eq:transfer-discount-sets-the-sim-real-mixture}}). **System identification** raises it at
**0.079** per unit cost — the best row available, and it needs measurements rather than
trajectories. Two hundred robots with the gap closed finish in **6.6 months for $68,770,581**,
against $454,000,000 for an all-real programme.

And under all of it, the asymmetry that makes the field cautious: **99% of the expected cost per
attempt sits in the 5% of attempts that damage something**, and no architecture makes a dropped
object recoverable.

What runs through the chapter is that both halves reward not needing the expensive thing. The
horizon is extended by shortening the sequence, not by predicting better. The fleet is shrunk by
measuring the robot, not by running it. In each case the direct approach — a better model, more
robots — is available, expensive, and roughly two orders of magnitude worse per unit than the
indirect one.

Carry forward: **coverage decides whether planning helps**, and **the transfer coefficient
divides the fleet**.

## 21. Further Reading

- {{cite:ha2018worldmodels}} — learning a compact dynamics model and planning inside it.
- {{cite:driess2023palme}} — grounding a language model in embodied observations and actions.
- {{cite:tong2022videomae}} — learning representations from passive video, the source with the
  best real-equivalent-per-dollar and the lowest transfer.
- {{cite:ravi2024sam2}} — segmentation and tracking, which reduce the state error that feeds
  every rollout.

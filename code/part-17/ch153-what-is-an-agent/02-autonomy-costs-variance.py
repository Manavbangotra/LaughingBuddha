# -*- coding: utf-8 -*-
# Extracted from: Chapter 153 — What an AI Agent Is: LLM versus Workflow versus Agent
# Source: src/.../ch153-what-is-an-agent.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The two things autonomy costs that a mean does not show.

The previous listing compared architectures on average success and average cost.
Averages are the wrong summary for an agent, and this listing measures the two
reasons (eq:autonomy-costs-variance).

The first is that an agent's cost is a DISTRIBUTION, not a number. A router
spends exactly two calls on every request. An agent spends however many steps it
takes, and "however many it takes" has a tail: retries, wandering, and the case
where it does not converge at all. Capacity planning against a mean under-
provisions by whatever the tail is.

The second is that an agent has more distinct execution paths than you can
enumerate, which changes what "tested" can mean. A pipeline has one path. A
router has one per branch. An agent's path count grows exponentially in the
horizon, and this listing computes what fraction of it a test suite covers.
"""
import numpy as np

rng = np.random.default_rng(1523)

N_RUNS = 200000
P_STEP = 0.93           # a step does the right thing
P_STUCK = 0.06          # a step enters a non-productive cycle
STEPS_NEEDED = 5        # productive steps required to finish
P_ESCAPE = 0.35         # chance of leaving a non-productive cycle each step


def simulate(budget):
    """One agent run. Each step either makes progress, gets stuck, or is a wasted
    retry. Returns (completed, steps_used)."""
    done = np.zeros(N_RUNS, dtype=bool)
    steps = np.zeros(N_RUNS, dtype=np.int64)
    progress = np.zeros(N_RUNS, dtype=np.int64)
    stuck = np.zeros(N_RUNS, dtype=bool)
    alive = np.ones(N_RUNS, dtype=bool)
    for _ in range(budget):
        idx = np.flatnonzero(alive)
        if not len(idx):
            break
        steps[idx] += 1
        u = rng.random(len(idx))
        # Runs in a non-productive cycle either escape or burn a step.
        st = stuck[idx]
        esc = st & (u < P_ESCAPE)
        stuck[idx[esc]] = False
        # Runs not stuck make progress, get stuck, or waste the step.
        free = ~st
        v = rng.random(len(idx))
        adv = free & (v < P_STEP)
        newstuck = free & (v >= P_STEP) & (v < P_STEP + P_STUCK)
        progress[idx[adv]] += 1
        stuck[idx[newstuck]] = True
        fin = progress[idx] >= STEPS_NEEDED
        done[idx[fin]] = True
        alive[idx[fin]] = False
    return done, steps


BUDGETS = [5, 8, 12, 20, 40, 100]

print(f"An agent needs {STEPS_NEEDED} productive steps. Each step makes progress")
print(f"{P_STEP:.0%} of the time, enters a non-productive cycle {P_STUCK:.0%} of")
print(f"the time, and otherwise is wasted; a cycle is escaped {P_ESCAPE:.0%} of")
print("the time per step.")
print()
print(f"{'step budget':>13}{'completed':>12}{'mean':>8}{'p50':>7}{'p90':>7}"
      f"{'p99':>7}{'max':>7}")
print(f"{'':>13}{'':>12}{'steps':>8}{'':>7}{'':>7}{'':>7}{'':>7}")
print("-" * 61)

tab = {}
for b in BUDGETS:
    done, steps = simulate(b)
    q = np.percentile(steps, [50, 90, 99])
    tab[b] = (float(done.mean()), float(steps.mean()), q[0], q[1], q[2],
              int(steps.max()))
    print(f"{b:>13}{tab[b][0]:>12.1%}{tab[b][1]:>8.2f}{q[0]:>7.0f}{q[1]:>7.0f}"
          f"{q[2]:>7.0f}{tab[b][5]:>7}")

print()
print()
print("What does the budget cost in completions, and what does it buy in")
print("predictability? Compare each budget against the largest.")
print()
ref = BUDGETS[-1]
print(f"{'step budget':>13}{'completions':>14}{'p99 cost':>11}"
      f"{'worst case':>13}")
print(f"{'':>13}{'lost':>14}{'vs b=' + str(ref):>11}{'cost':>13}")
print("-" * 51)
for b in BUDGETS:
    print(f"{b:>13}{tab[b][0] - tab[ref][0]:>+14.1%}"
          f"{tab[b][4] - tab[ref][4]:>+11.0f}{b:>13}")

print()
print()
print("How many distinct execution paths are there? A path is a sequence of")
print("per-step outcomes, and a test case exercises one of them.")
print()
K_OUT = 3               # outcomes per step: progress, stuck, wasted
N_TESTS = 500
print(f"{'architecture':>22}{'distinct paths':>17}{'covered by':>13}"
      f"{'coverage':>11}")
print(f"{'':>22}{'':>17}{str(N_TESTS) + ' tests':>13}{'':>11}")
print("-" * 63)
arch = [("pipeline", 1), ("router, 6 branches", 6)]
for h in (3, 5, 8, 12):
    arch.append((f"agent, horizon {h}", K_OUT ** h))
for name, paths in arch:
    cov = min(N_TESTS, paths)
    print(f"{name:>22}{paths:>17,}{cov:>13,}{cov / paths:>11.2%}")

print()
print()
print("Paths are not equally likely. What share of PROBABILITY MASS do the")
print("commonest paths cover, for an agent at horizon 8?")
print()
H = 8
probs = np.array([P_STEP, P_STUCK, 1 - P_STEP - P_STUCK])
mass = np.array([1.0])
for _ in range(H):
    mass = np.outer(mass, probs).ravel()
mass = np.sort(mass)[::-1]
cum = np.cumsum(mass)
print(f"{'commonest N paths':>20}{'share of runs':>16}"
      f"{'share of paths':>17}")
print("-" * 53)
for n in (1, 10, 50, 200, 1000, len(mass)):
    print(f"{n:>20,}{cum[n - 1]:>16.1%}{n / len(mass):>17.2%}")

hit_90 = int(np.searchsorted(cum, 0.90) + 1)
hit_99 = int(np.searchsorted(cum, 0.99) + 1)
print(f"""
The first table is the shape of an agent's cost, and the columns to compare are
the mean and the p99.

At a budget of {BUDGETS[-1]} steps the agent completes {tab[ref][0]:.1%} of tasks
using {tab[ref][1]:.2f} steps on average -- and a p99 of {tab[ref][4]:.0f} and a
worst case of {tab[ref][5]}. **The tail is {tab[ref][4] / tab[ref][1]:.1f} times
the mean**, and capacity planned against the mean under-provisions by that factor
for the requests that matter most, which are the slow ones.

A router's equivalent row would read: mean 2, p50 2, p90 2, p99 2, max 2. That is
the difference autonomy makes to a capacity plan, and it is invisible in any
comparison of averages.

The second table is what a step budget buys, and it is the most useful thing in
this listing because the trade is so lopsided.

Cutting the budget from {BUDGETS[-1]} to {20} costs
{tab[20][0] - tab[ref][0]:+.1%} in completions and takes the p99 from
{tab[ref][4]:.0f} steps to {tab[20][4]:.0f}. Cutting to {12} costs
{tab[12][0] - tab[ref][0]:+.1%} and caps the worst case at {12}.

**A budget converts a worst case you do not control into one you do, at a small
cost in completions**, and that is the argument for having one.

Be precise about what the uncapped case is, because the usual telling
overstates it. The run-length distribution here has a geometric tail rather than
an infinite one: escape probability is positive, so every run terminates with
probability 1, and across {N_RUNS:,} runs the longest observed was
{tab[ref][5]} steps -- {tab[ref][5] / tab[ref][1]:.1f} times the mean. The
expected cost is finite. What is not bounded is the MAXIMUM, which grows
logarithmically with how many requests you serve: the worst run you will see this
year is a function of your traffic volume, not of your system.

That is the honest form of the argument, and it is still decisive for a capacity
plan. You cannot state a p100 without stating a request count, and a budget lets
you state one without either.

The third table is the other thing autonomy costs, and it is about what testing
can mean.

A pipeline has one execution path and {N_TESTS} tests cover it. A six-branch
router has six, and {N_TESTS} tests cover them all. An agent at horizon
{8} has {K_OUT ** 8:,} distinct paths and {N_TESTS} tests cover
{N_TESTS / K_OUT ** 8:.2%} of them; at horizon {12} it is
{N_TESTS / K_OUT ** 12:.4%}.

**A workflow can be verified by enumerating its paths and an agent cannot.** That
is not a matter of testing harder. It is a statement about the size of the set,
and it means the two architectures need different correctness arguments: a
workflow's is coverage, and an agent's has to be something else.

The fourth table says what that something else is, and it is the reason the
situation is not hopeless.

Paths are wildly unequal in probability. At horizon {H}, the commonest
{hit_90:,} paths -- {hit_90 / len(mass):.2%} of the total -- account for
{0.9:.0%} of runs, and {hit_99:,} paths account for {0.99:.0%}. The exponential
blow-up is real and almost all of it is in outcomes that essentially never
happen.

So the achievable correctness argument for an agent is **statistical rather than
exhaustive**: sample from the real distribution of runs, cover the mass rather
than the paths, and accept that the remaining {1 - 0.99:.0%} contains behaviours
nobody has ever seen. That is a weaker guarantee than a workflow's and it is not
nothing, and stating it in those terms is more honest than either "we tested it"
or "it cannot be tested".

One consequence worth drawing out. Because the uncovered mass is where the
unusual outcomes live, and because unusual outcomes are exactly where an agent
does something surprising with its tools, **the untested fraction of an agent's
behaviour is systematically the dangerous fraction**. That is
ch:ag-security's subject, and it is why the design response there is about
limiting consequences rather than about improving coverage.""")

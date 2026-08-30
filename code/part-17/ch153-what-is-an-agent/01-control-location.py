# -*- coding: utf-8 -*-
# Extracted from: Chapter 153 — What an AI Agent Is: LLM versus Workflow versus Agent
# Source: src/.../ch153-what-is-an-agent.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Pipeline, router, agent: the choice is a measurement about your traffic.

The three architectures differ in ONE thing -- who chooses the next step. A
pipeline's control flow is written by a human. A router picks among human-written
flows. An agent chooses each action itself, including when to stop.

That difference has a cost and a benefit, and both are computable
(eq:control-location). The cost is that an agent's per-step choices compound
(ch:rsn-cot's eq:chain-accuracy-compounds arriving in a new place). The benefit is
that it can handle task shapes nobody anticipated.

So the question "should this be an agent" is not an architecture preference. It
is a question about how much of your traffic falls outside the shapes you
enumerated, and this listing puts a number on the crossover.
"""
import numpy as np

rng = np.random.default_rng(1451)

N_TASKS = 40000
N_SHAPES = 6            # task shapes the team enumerated and built flows for
STEPS_HEAD = 3          # steps a head task needs
STEPS_TAIL = 6          # steps a tail task needs (they are the awkward ones)
P_STEP = 0.93           # agent's per-step action accuracy
P_ROUTE = 0.94          # router's classification accuracy
COST_STEP = 1.0         # one model call
COST_FLOW = 1.0         # a hand-written flow is one call, no deliberation


def run(tail_mass, n=N_TASKS):
    """One draw of the task distribution, run through all three architectures."""
    is_tail = rng.random(n) < tail_mass
    shape = rng.integers(0, N_SHAPES, size=n)          # for head tasks
    steps = np.where(is_tail, STEPS_TAIL, STEPS_HEAD)

    # PIPELINE: one hand-written flow for the commonest shape. It is correct on
    # that shape and wrong on everything else, and it costs one call either way.
    pipe_ok = (~is_tail) & (shape == 0)
    pipe_cost = np.full(n, COST_FLOW)

    # ROUTER: classify into one of the known shapes, then run that flow. Correct
    # when the task is a known shape AND the classification is right. Costs one
    # call to route plus one to run.
    routed_right = rng.random(n) < P_ROUTE
    rout_ok = (~is_tail) & routed_right
    rout_cost = np.full(n, 2 * COST_FLOW)

    # AGENT: chooses each action. It can address any shape, and every step is a
    # chance to go wrong. Cost is one call per step.
    agent_ok = (rng.random((n, STEPS_TAIL)) < P_STEP)
    agent_ok = np.array([agent_ok[i, :steps[i]].all() for i in range(n)])
    agent_cost = steps * COST_STEP

    return {
        "pipeline": (float(pipe_ok.mean()), float(pipe_cost.mean())),
        "router": (float(rout_ok.mean()), float(rout_cost.mean())),
        "agent": (float(agent_ok.mean()), float(agent_cost.mean())),
    }


TAILS = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70]

print(f"{N_TASKS} tasks. {N_SHAPES} task shapes were enumerated and given")
print(f"hand-written flows; the rest is tail. Head tasks take {STEPS_HEAD}")
print(f"steps, tail tasks {STEPS_TAIL}. The agent is {P_STEP:.0%} accurate per")
print(f"step; the router is {P_ROUTE:.0%} accurate at classifying.")
print()
print(f"{'tail mass':>11}{'pipeline':>22}{'router':>20}{'agent':>20}")
print(f"{'':>11}{'success':>11}{'cost':>11}{'success':>10}{'cost':>10}"
      f"{'success':>10}{'cost':>10}")
print("-" * 73)

res = {}
for t in TAILS:
    r = run(t)
    res[t] = r
    print(f"{t:>11.0%}{r['pipeline'][0]:>11.1%}{r['pipeline'][1]:>11.2f}"
          f"{r['router'][0]:>10.1%}{r['router'][1]:>10.2f}"
          f"{r['agent'][0]:>10.1%}{r['agent'][1]:>10.2f}")

print()
print()
print("Success per unit of cost -- what you get for each model call.")
print()
print(f"{'tail mass':>11}{'pipeline':>12}{'router':>10}{'agent':>10}"
      f"{'best':>12}")
print("-" * 55)
eff = {}
for t in TAILS:
    r = res[t]
    e = {k: r[k][0] / r[k][1] for k in r}
    eff[t] = e
    best = max(e, key=e.get)
    print(f"{t:>11.0%}{e['pipeline']:>12.3f}{e['router']:>10.3f}"
          f"{e['agent']:>10.3f}{best:>12}")

print()
print()
print("Where does the agent overtake on raw success? Sweep finely.")
print()
print(f"{'tail mass':>11}{'router':>10}{'agent':>10}{'gap':>10}")
print("-" * 41)
fine = {}
for t in (0.08, 0.12, 0.16, 0.20, 0.24, 0.30):
    r = run(t, n=60000)
    fine[t] = (r["router"][0], r["agent"][0])
    print(f"{t:>11.0%}{r['router'][0]:>10.1%}{r['agent'][0]:>10.1%}"
          f"{r['agent'][0] - r['router'][0]:>+10.1%}")

cross = [t for t in sorted(fine) if fine[t][1] > fine[t][0]]
head_step = P_STEP ** STEPS_HEAD
tail_step = P_STEP ** STEPS_TAIL
print(f"""
The first table is the trade in its simplest form, and the two ends of the tail
column are the whole argument.

At {0:.0%} tail mass the pipeline succeeds {res[0.0]['pipeline'][0]:.1%} of the
time -- it only handles one of the {N_SHAPES} shapes -- the router
{res[0.0]['router'][0]:.1%}, and the agent {res[0.0]['agent'][0]:.1%}. The router
wins comfortably, and it wins because every task IS one of the shapes somebody
enumerated. There is nothing for autonomy to buy.

At {0.5:.0%} tail mass the same three are {res[0.5]['pipeline'][0]:.1%},
{res[0.5]['router'][0]:.1%} and {res[0.5]['agent'][0]:.1%}. The router's ceiling
is the head mass, by construction: it can only run flows that exist. The agent
has no such ceiling and pays for it per step.

{'The agent overtakes the router at a tail mass of about ' + format(cross[0], '.0%') + '.' if cross else 'The agent does not overtake the router over the range swept.'}

That crossover is the number to compute for your own traffic, and it moves with
two things you can measure. It moves LEFT as your agent's per-step accuracy
rises: at {P_STEP:.0%} per step a {STEPS_HEAD}-step task completes
{head_step:.1%} of the time and a {STEPS_TAIL}-step task {tail_step:.1%}. And it
moves RIGHT as you enumerate more shapes, because every shape you add converts
tail mass into head mass.

**So "should this be an agent" is a question with a numeric answer, and the
answer is mostly about your traffic rather than about your model.** A team whose
requests fall into six shapes should write six flows. A team whose requests have
a long tail of one-off combinations cannot enumerate their way out, and the
per-step tax is what they pay to avoid trying.

The second table adds cost, and it does not merely shift the crossover -- it
removes it over the whole range swept.

Per model call the router is the most efficient of the three at every tail mass
here: {eff[0.0]['router']:.3f} against the agent's {eff[0.0]['agent']:.3f} at
zero tail, and {eff[0.7]['router']:.3f} against {eff[0.7]['agent']:.3f} at
{0.7:.0%} -- where the agent is winning on raw success by
{res[0.7]['agent'][0] - res[0.7]['router'][0]:+.1%}. The agent spends
{res[0.0]['agent'][1]:.1f} calls at zero tail and {res[0.7]['agent'][1]:.1f} at
{0.7:.0%}, because tail tasks are longer as well as rarer, so its cost rises
exactly where its advantage does.

**The success crossover and the cost-efficiency crossover are in completely
different places, and on this cost model the second one never arrives.** Which
you should use depends on whether model calls or failed tasks are the expensive
thing, and those differ by orders of magnitude between products. For a
high-volume assistant the calls dominate and the router is correct well past the
success crossover. For an agent that files support tickets or writes code, a
failed task costs a human's time and the calls are a rounding error.

That is a more useful framing than "agents are expensive", because it says what
to measure: **the ratio of the cost of a failed task to the cost of a model
call.** Below roughly {res[0.2]['agent'][1] / res[0.2]['router'][1]:.1f} -- the
call-count ratio at the success crossover -- the router wins on both counts.

One thing this listing deliberately does not model, and it is the largest
omission: the agent's cost here is deterministic given the task length. Real
agent loops retry, wander, and occasionally do not stop at all, which turns cost
into a heavy-tailed distribution and makes the mean a poor summary. That is the
next listing's subject, and it moves the recommendation further toward the
router than this table suggests.""")

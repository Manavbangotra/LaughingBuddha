# -*- coding: utf-8 -*-
# Extracted from: Chapter 155 — The Agent Loop: Perception, Decision, Action
# Source: src/.../ch155-agent-loop.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The stopping decision is a separate classifier, and its errors dominate.

An agent loop is an absorbing Markov chain: states, transitions, and a "done"
state you hope to reach. ch:ag-what-is-an-agent measured the cost distribution.
This listing measures the thing that determines whether the chain absorbs at all,
and it is not the quality of the actions (eq:stopping-is-a-classifier).

An agent decides to stop by judging that the task is complete. That judgement is a
binary classifier with two error rates, and they cause opposite failures: stopping
early when the work is not done, and never stopping when it is. Systems report
"task success" and tune the actions, and the actions may not be where the loss is.
"""
import numpy as np

rng = np.random.default_rng(1721)

N = 60000
NEED = 6              # productive steps a task requires
P_ACT = 0.90          # a step makes progress
HORIZON = 25


def run(p_act, tpr, fpr, horizon=HORIZON, need=NEED):
    """tpr: chance of correctly recognising the task IS done.
    fpr: chance of wrongly declaring it done when it is not.
    Returns (succeeded, stopped_early, ran_out, steps)."""
    prog = np.zeros(N, dtype=np.int64)
    steps = np.zeros(N, dtype=np.int64)
    alive = np.ones(N, dtype=bool)
    early = np.zeros(N, dtype=bool)
    good = np.zeros(N, dtype=bool)
    for _ in range(horizon):
        idx = np.flatnonzero(alive)
        if not len(idx):
            break
        steps[idx] += 1
        prog[idx] += (rng.random(len(idx)) < p_act)
        done = prog[idx] >= need
        # The stopping classifier fires on every step, on both kinds of state.
        u = rng.random(len(idx))
        stop = np.where(done, u < tpr, u < fpr)
        good[idx[stop & done]] = True
        early[idx[stop & ~done]] = True
        alive[idx[stop]] = False
    return good, early, ~good & ~early, steps


print(f"A task needs {NEED} productive steps; a step makes progress")
print(f"{P_ACT:.0%} of the time. The agent stops when it JUDGES the task done.")
print(f"Horizon {HORIZON}.")
print()
print("First: a perfect stopping judgement, to isolate the action quality.")
print()
print(f"{'p(action)':>11}{'succeeded':>12}{'ran out':>10}{'mean steps':>13}")
print("-" * 46)
act_tab = {}
for pa in (0.75, 0.85, 0.90, 0.95, 0.99):
    g, e, r, s = run(pa, 1.0, 0.0)
    act_tab[pa] = (float(g.mean()), float(r.mean()), float(s.mean()))
    print(f"{pa:>11.0%}{act_tab[pa][0]:>12.1%}{act_tab[pa][1]:>10.1%}"
          f"{act_tab[pa][2]:>13.2f}")

print()
print()
print("Now hold the actions fixed and vary the stopping judgement instead.")
print(f"Actions are {P_ACT:.0%} accurate throughout.")
print()
print(f"{'recognises':>12}{'false':>9}{'succeeded':>12}{'stopped':>10}"
      f"{'ran':>8}{'mean':>8}")
print(f"{'done':>12}{'stop':>9}{'':>12}{'early':>10}{'out':>8}{'steps':>8}")
print("-" * 59)
stop_tab = {}
CASES = [(1.00, 0.00), (0.95, 0.02), (0.85, 0.05), (0.70, 0.10),
         (0.50, 0.02), (0.95, 0.15)]
# (0.95, 0.02) and (0.50, 0.02) differ only in recognition, which is the
# comparison the narrative turns on.
for tpr, fpr in CASES:
    g, e, r, s = run(P_ACT, tpr, fpr)
    stop_tab[(tpr, fpr)] = (float(g.mean()), float(e.mean()), float(r.mean()),
                            float(s.mean()))
    v = stop_tab[(tpr, fpr)]
    print(f"{tpr:>12.0%}{fpr:>9.0%}{v[0]:>12.1%}{v[1]:>10.1%}{v[2]:>8.1%}"
          f"{v[3]:>8.2f}")

print()
print()
print("Which is the better place to spend? Equal-sized improvements to the")
print("action quality and to the stopping judgement, from a common baseline.")
print()
BASE = (P_ACT, 0.85, 0.05)
g, e, r, s = run(*BASE)
base_succ = float(g.mean())
print(f"{'intervention':>40}{'succeeded':>12}{'change':>10}")
print("-" * 62)
print(f"{'baseline (act 90%, tpr 85%, fpr 5%)':>40}{base_succ:>12.1%}"
      f"{0.0:>+10.1%}")
spend = {}
for name, args in [
        ("actions 90% -> 95%", (0.95, 0.85, 0.05)),
        ("actions 90% -> 99%", (0.99, 0.85, 0.05)),
        ("recognises done 85% -> 95%", (P_ACT, 0.95, 0.05)),
        ("false stops 5% -> 1%", (P_ACT, 0.85, 0.01)),
        ("both stopping fixes", (P_ACT, 0.95, 0.01))]:
    g, e, r, s = run(*args)
    spend[name] = float(g.mean())
    print(f"{name:>40}{spend[name]:>12.1%}{spend[name] - base_succ:>+10.1%}")

print()
print()
print("A false stop is not a failure to finish -- it is a WRONG ANSWER returned")
print("confidently. Split the outcomes by what the user actually receives.")
print()
print(f"{'false stop rate':>17}{'correct':>10}{'confidently':>14}{'visibly':>10}")
print(f"{'':>17}{'answer':>10}{'wrong':>14}{'failed':>10}")
print("-" * 51)
fs_tab = {}
for fpr in (0.0, 0.01, 0.03, 0.05, 0.10, 0.20):
    g, e, r, s = run(P_ACT, 0.90, fpr)
    fs_tab[fpr] = (float(g.mean()), float(e.mean()), float(r.mean()))
    print(f"{fpr:>17.0%}{fs_tab[fpr][0]:>10.1%}{fs_tab[fpr][1]:>14.1%}"
          f"{fs_tab[fpr][2]:>10.1%}")

print(f"""
The first table is the loop working as advertised, and it is the comparison every
agent system reports. With a perfect stopping judgement, EVERY action quality
reaches {act_tab[0.75][0]:.1%}. Not approximately -- exactly, at
{0.75:.0%} per action and at {0.99:.0%}. The only thing that changes is the number
of steps taken: {act_tab[0.75][2]:.2f} against {act_tab[0.99][2]:.2f}.

That is worth sitting with, because it contradicts the intuition carried over from
ch:rsn-cot. A chain has to get every step right and its accuracy is $p^k$. **A
loop only has to get ENOUGH steps right eventually**, so with slack in the horizon
it converts a reliability problem into a cost problem. Per-action accuracy buys
speed, not success.

Which means the loop's success has to be decided somewhere else, and the second
table finds where.

Holding actions at {P_ACT:.0%}, a stopping judgement that recognises completion
{0.85:.0%} of the time with a {0.05:.0%} false-stop rate scores
{stop_tab[(0.85, 0.05)][0]:.1%} against a perfect judgement's
{stop_tab[(1.0, 0.0)][0]:.1%}. **The actions did not change, and
{stop_tab[(1.0, 0.0)][0] - stop_tab[(0.85, 0.05)][0]:.1%} of the outcome was
decided by a classifier nobody was measuring.**

The third table prices the interventions, and the result was not the one I
expected.

Improving actions from {P_ACT:.0%} to {0.99:.0%} -- about the most you could hope
for, and a large investment -- buys
{spend['actions 90% -> 99%'] - base_succ:+.1%}. Improving completion RECOGNITION
from {0.85:.0%} to {0.95:.0%} buys
{spend['recognises done 85% -> 95%'] - base_succ:+.1%}, which is nothing. Cutting
FALSE STOPS from {0.05:.0%} to {0.01:.0%} buys
{spend['false stops 5% -> 1%'] - base_succ:+.1%}.

One parameter is worth roughly eight times the other two combined, and the reason
is structural rather than numerical. **The two stopping errors are not
symmetric.**

A missed completion is recoverable. The agent does not notice it is done, takes
another step, and gets another chance to notice -- and another, every step until
the horizon. Over a run with slack, a recognition rate of {0.5:.0%} and one of
{0.95:.0%} produce almost the same outcome
({stop_tab[(0.5, 0.02)][0]:.1%} against {stop_tab[(0.95, 0.02)][0]:.1%}); the
low-recognition agent simply takes longer ({stop_tab[(0.5, 0.02)][3]:.2f} steps
against {stop_tab[(0.95, 0.02)][3]:.2f}).

A false stop is terminal. It ends the run, and there is no next step in which to
correct it. So one error gets retried at every opportunity and the other gets one
shot at ruining the task, and their per-step rates should never be compared
directly (eq:stopping-is-a-classifier).

This also explains the {0.5:.0%}-recognition row, which looks anomalous until you
see it: {stop_tab[(0.5, 0.02)][0]:.1%} success from an agent that recognises
completion only half the time. Half of a lot of chances is still enough chances.

The fourth table is why the false-stop direction deserves separate treatment
beyond its size, and it is about what the user receives rather than what the
metric records.

At a {0.05:.0%} false-stop rate, {fs_tab[0.05][1]:.1%} of runs end early and
{fs_tab[0.05][2]:.1%} exhaust the horizon. At {0.20:.0%} it is
{fs_tab[0.2][1]:.1%} against {fs_tab[0.2][2]:.1%}.

Those two failures are usually summed into one "did not succeed" number, and they
are not comparable. Exhausting the horizon is a VISIBLE failure: the budget was
hit, the system knows it, it can retry or escalate. Stopping early is INVISIBLE:
the agent returns an answer, confidently, having done part of the work. **A
visible failure costs a retry; a confident wrong answer costs whatever the wrong
answer causes**, and for an agent with write access that is ch:ag-security's
subject.

So the design conclusion is a threshold, and it points against the intuitive
setting. **Bias the stopping classifier heavily toward not stopping, and let the
budget end the run.** Missed completions cost steps, which are cheap and bounded
by the horizon. False stops cost correctness, and they are not recoverable. Most
systems tune this the other way, because an agent that stops promptly feels
better than one that keeps checking.

Two caveats, and the second is the larger.

The horizon is doing a great deal of work here. All of the "a missed stop is
recoverable" argument depends on there being steps left, so at a horizon close to
{NEED} the two error directions become comparable and the argument weakens. The
right reading is that the asymmetry is a function of slack, and slack is a design
parameter.

And this models completion detection as a classifier with a fixed operating point.
In a real agent that judgement comes from the same model that took the actions,
which is ch:rsn-self-consistency's correlated critic: the false-stop rate will be
highest exactly on the tasks the agent handled badly, because that is where its
own judgement is least reliable. **The numbers here are optimistic in precisely
the direction that matters**, and the fix is the same one that chapter reached --
a completion check that is not the agent.""")

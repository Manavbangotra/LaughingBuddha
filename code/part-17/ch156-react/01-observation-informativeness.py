# -*- coding: utf-8 -*-
# Extracted from: Chapter 156 — ReAct and Interleaved Reasoning and Acting
# Source: src/.../ch156-react.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""When is it worth looking before you act?

cite:yao2023react interleaves reasoning and acting: think, act, observe, repeat.
ch:rsn-tool-assisted measured the cost of that shape and found it losing to a
single up-front program at every chain length, because each boundary crossing is
paid twice -- once to construct the call and once to read the result back.

Both are right, and the variable that separates them is how much an observation
tells you that you could not have predicted (eq:observation-informativeness).

A plan written before any observation is a prediction about the environment. Where
the environment is predictable the prediction is accurate and the plan is free.
Where it is not, every step of the plan after the first surprise is executing
against a state that no longer exists.

This listing sweeps that one variable and finds the crossover.
"""
import numpy as np

rng = np.random.default_rng(1877)

N = 60000
K = 7                    # steps in the task
P_ACT = 0.94             # executing a correctly-chosen action
P_TRANS = 0.96           # composing a call for an interleaved step
P_PARSE = 0.97           # reading one observation back correctly
P_PLAN_STEP = 0.97       # getting one step of an up-front plan right


def run(info, k=K):
    """`info` is observation informativeness: the chance that a step's correct
    action depends on something only observable at that step.

    plan-then-execute: commits to k actions up front. A step whose action
    depended on an unobserved fact is wrong, and everything after a wrong step
    in a dependent plan is executing from a bad state.

    interleaved: observes before each action, so informativeness costs it
    nothing -- but it constructs a call and parses a result at every step.
    """
    # PLAN: each step is right if the plan got it right AND it did not depend on
    # something unobservable at planning time.
    surprises = rng.random((N, k)) < info
    plan_ok = (rng.random((N, k)) < P_PLAN_STEP) & ~surprises
    plan_done = plan_ok.all(1)
    plan_calls = np.full(N, 1.0 + 1.0)          # write the plan, execute it

    # INTERLEAVED: sees the state before choosing, so surprises are handled.
    inter_ok = ((rng.random((N, k)) < P_ACT) &
                (rng.random((N, k)) < P_TRANS) &
                (rng.random((N, k)) < P_PARSE)).all(1)
    inter_calls = np.full(N, float(k))

    # REPLAN ON SURPRISE: execute the plan, and rewrite it when surprised.
    # Costs one extra call per surprise; the rewritten steps are then informed.
    n_sur = surprises.sum(1)
    # Every step still has to be planned correctly; a surprise
    # additionally costs one observation to read, and only surprises do.
    parse_ok = (rng.random((N, k)) < P_PARSE) | ~surprises
    replan_ok = (rng.random((N, k)) < P_PLAN_STEP).all(1) & parse_ok.all(1)
    replan_calls = 2.0 + n_sur

    return {
        "plan": (float(plan_done.mean()), float(plan_calls.mean())),
        "interleaved": (float(inter_ok.mean()), float(inter_calls.mean())),
        "replan": (float(replan_ok.mean()), float(replan_calls.mean())),
    }


INFOS = [0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.60]

print(f"A {K}-step task. `informativeness` is the chance that a step's correct")
print("action depends on something only visible at that step. Plan-then-execute")
print(f"commits up front ({P_PLAN_STEP:.0%} per planned step); interleaving")
print(f"observes first but pays {P_TRANS:.0%} to compose each call and")
print(f"{P_PARSE:.0%} to read each result.")
print()
print(f"{'info':>7}{'plan-then-execute':>24}{'interleaved':>20}"
      f"{'replan on surprise':>24}")
print(f"{'':>7}{'success':>13}{'calls':>11}{'success':>11}{'calls':>9}"
      f"{'success':>14}{'calls':>10}")
print("-" * 75)

res = {}
for i in INFOS:
    r = run(i)
    res[i] = r
    print(f"{i:>7.0%}{r['plan'][0]:>13.1%}{r['plan'][1]:>11.1f}"
          f"{r['interleaved'][0]:>11.1%}{r['interleaved'][1]:>9.1f}"
          f"{r['replan'][0]:>14.1%}{r['replan'][1]:>10.1f}")

cross = [i for i in INFOS if res[i]["interleaved"][0] > res[i]["plan"][0]]

print()
print()
print("Success per model call -- what each shape returns for its cost.")
print()
print(f"{'info':>7}{'plan':>10}{'interleaved':>14}{'replan':>10}{'best':>14}")
print("-" * 55)
eff = {}
for i in INFOS:
    e = {k2: res[i][k2][0] / res[i][k2][1] for k2 in res[i]}
    eff[i] = e
    print(f"{i:>7.0%}{e['plan']:>10.3f}{e['interleaved']:>14.3f}"
          f"{e['replan']:>10.3f}{max(e, key=e.get):>14}")

print()
print()
print("Does the crossover move with task length? Sweep k at fixed")
print(f"informativeness of {0.05:.0%} and {0.20:.0%}.")
print()
print(f"{'steps k':>9}{'info 5%':>24}{'info 20%':>24}")
print(f"{'':>9}{'plan':>12}{'interleaved':>12}{'plan':>12}{'interleaved':>12}")
print("-" * 57)
len_tab = {}
for k in (2, 4, 7, 12, 20):
    a, b = run(0.05, k), run(0.20, k)
    len_tab[k] = (a, b)
    print(f"{k:>9}{a['plan'][0]:>12.1%}{a['interleaved'][0]:>12.1%}"
          f"{b['plan'][0]:>12.1%}{b['interleaved'][0]:>12.1%}")

print()
print()
print("What if the boundary crossing gets cheaper? Interleaving's cost is two")
print("multiplications per step; sweep them together.")
print()
print(f"{'per-step overhead':>19}{'interleaved success':>22}{'vs plan at 5%':>16}")
print("-" * 57)
P_T_SAVE, P_P_SAVE = P_TRANS, P_PARSE
ov = {}
plan5 = res[0.05]["plan"][0]
for q in (0.90, 0.94, 0.96, 0.98, 0.995):
    P_TRANS = P_PARSE = q
    v = run(0.05)["interleaved"][0]
    ov[q] = v
    print(f"{q:>19.1%}{v:>22.1%}{v - plan5:>+16.1%}")
P_TRANS, P_PARSE = P_T_SAVE, P_P_SAVE

print(f"""
The first table is the reconciliation, and the two ends of the informativeness
column are the two papers.

At {0:.0%} informativeness -- a fully predictable environment -- planning up front
scores {res[0.0]['plan'][0]:.1%} in {res[0.0]['plan'][1]:.0f} calls and
interleaving scores {res[0.0]['interleaved'][0]:.1%} in
{res[0.0]['interleaved'][1]:.0f}. Interleaving loses on both axes, which is
ch:rsn-tool-assisted's result: it is paying {K} boundary crossings for
information it could have predicted.

At {0.2:.0%} informativeness the same comparison is
{res[0.2]['plan'][0]:.1%} against {res[0.2]['interleaved'][0]:.1%}. The plan is
now a prediction about an environment that keeps surprising it, and each surprise
invalidates a step.

{'The crossover is at about ' + format(cross[0], '.0%') + ' informativeness.' if cross else 'Interleaving does not overtake over the range swept.'}

**So "should I interleave" is a question about the environment, not about the
architecture.** The number to estimate is how often a step's correct action
depends on something you could not have known before taking the previous one, and
that is measurable on a task distribution you already have.

The replan column is the result, and it is neither of the two shapes the
literature argues about. It writes a plan, executes it, and rewrites when
surprised -- so it pays for informativeness only when informativeness occurs.

It beats BOTH pure strategies at every informativeness level swept.
{res[0.0]['replan'][0]:.1%} at {0:.0%}, where it matches planning because there is
nothing to replan; {res[0.6]['replan'][0]:.1%} at {0.6:.0%}, where planning has
collapsed to {res[0.6]['plan'][0]:.1%} and interleaving sits at
{res[0.6]['interleaved'][0]:.1%}. Its call count rises from
{res[0.0]['replan'][1]:.1f} to {res[0.6]['replan'][1]:.1f} -- **a call per
surprise rather than a call per step**, which is the whole difference.

The second table says the same thing in cost terms: replanning is the best of the
three at all {len(INFOS)} informativeness levels swept.

That is a stronger result than the sweep was built to produce, and the reason is
worth extracting. Interleaving pays the observation cost unconditionally, on the
assumption that every step might be surprising. Planning pays it never, on the
assumption that none is. **Replanning pays it exactly when a surprise occurs**,
which is the only one of the three policies whose cost is proportional to the
thing it is buying.

The catch is in the word "when". Replanning requires DETECTING that the plan no
longer applies, which is a classifier -- ch:ag-loop's stopping decision wearing a
different hat, with the same correlated-critic ceiling. This listing gives it a
perfect detector. A real one is not, and the gap between these numbers and a
deployed system is almost entirely that detector, which is ch:ag-planning's
subject and its hardest problem.

The third table checks whether the conclusion depends on task length, and the
answer is cleaner than I expected: it does not.

At {0.05:.0%} informativeness the plan leads at every length -- {2} steps
({len_tab[2][0]['plan'][0]:.1%} against
{len_tab[2][0]['interleaved'][0]:.1%}) through {20}
({len_tab[20][0]['plan'][0]:.1%} against
{len_tab[20][0]['interleaved'][0]:.1%}). At {0.2:.0%} interleaving leads at every
length. The winner never changes.

The reason is that both shapes are products of a per-step base raised to k, so
the comparison is between the two bases and k cancels out. Planning's base is
{P_PLAN_STEP:.2f}(1 - info); interleaving's is
{P_ACT * P_TRANS * P_PARSE:.3f}. They cross where
{P_PLAN_STEP:.2f}(1 - info) = {P_ACT * P_TRANS * P_PARSE:.3f}, at
info = {1 - (P_ACT * P_TRANS * P_PARSE) / P_PLAN_STEP:.1%}, and that expression
contains no k.

**So the architecture choice is decided entirely by a comparison of two per-step
reliabilities, and task length only amplifies whichever one is already winning.**
That is a much simpler rule than "use ReAct for complex tasks": complexity does
not enter. What enters is how predictable the environment is and how lossy your
tool boundary is.

The last table asks what would change the answer, and it is the actionable one.
Interleaving's disadvantage is entirely the per-step overhead: composing a call
and parsing a result, {K} times. Take that overhead from {0.90:.0%} to
{0.995:.0%} and interleaved success goes {ov[0.90]:.1%} to {ov[0.995]:.1%},
against the plan's {plan5:.1%} at the same informativeness.

**Every point of per-step overhead is multiplied by the horizon**, which is
ch:ag-tool-calling's finding stated as an architectural argument: constrained
decoding, enumerated arguments and unambiguous response formats do not merely
improve tool calls, they change which agent architecture is correct. A team with
a rigorous tool interface should interleave; a team without one should plan and
replan, because it cannot afford {K} round trips through a lossy boundary.""")

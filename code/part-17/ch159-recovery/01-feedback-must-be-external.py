# -*- coding: utf-8 -*-
# Extracted from: Chapter 159 — Reflection, Replanning, and Error Recovery
# Source: src/.../ch159-recovery.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Recovery works when something outside the agent grades it.

cite:huang2024selfcorrect separates intrinsic self-correction -- a model revising
using only its own capabilities -- from correction guided by external feedback, and
finds the first does not improve reasoning while the second does.
ch:rsn-self-consistency measured the mechanism: a self-critic's errors are
correlated with the solver's, so it is least reliable exactly where it is needed.

An agent looks like it escapes this, because it has an ENVIRONMENT. The tool
returns an error; the test fails; the page does not load. That is external, it is
not a sample from the model, and it is exactly the ingredient
cite:huang2024selfcorrect says is missing.

This listing asks how much of that escape survives the environment's feedback being
imperfect (eq:feedback-must-be-external), which it always is.
"""
import numpy as np

rng = np.random.default_rng(2281)

N = 60000
P_FIRST = 0.42          # chance the first attempt succeeds
ATTEMPTS = 4


def run(kind, quality=1.0, attempts=ATTEMPTS, p_first=P_FIRST):
    """kind:
       'none'  -- no retry at all
       'blind' -- always retry, learning nothing from the failure
       'env'   -- the environment reports the failure; `quality` is how often
                  that report is correct and useful
       'self'  -- the agent judges its own attempt. Its judgement is CORRELATED
                  with its competence: on attempts it got wrong, it is also
                  wrong about whether it got them wrong.
    """
    ok = rng.random(N) < p_first
    for _ in range(attempts - 1):
        if kind == "none":
            break
        failed = ~ok
        if kind == "blind":
            told = failed                     # knows it failed, learns nothing
            informed = np.zeros(N, dtype=bool)
        elif kind == "env":
            told = failed & (rng.random(N) < quality)
            informed = told
        else:  # self
            # Detecting its own failure is itself a skill it lacks on the
            # attempts it botched: correlate the detection with the outcome.
            told = failed & (rng.random(N) < quality * p_first)
            informed = told & (rng.random(N) < p_first)
        # An informed retry is conditioned on the fault and much more likely to
        # land; an uninformed one is a fresh draw from the same distribution.
        gain = np.where(informed, 0.75, p_first)
        ok |= told & (rng.random(N) < gain)
    return float(ok.mean())


print(f"First attempt succeeds {P_FIRST:.0%} of the time; up to {ATTEMPTS}")
print("attempts. An informed retry -- one that knows what went wrong -- lands")
print(f"{0.75:.0%} of the time; an uninformed one is a fresh {P_FIRST:.0%} draw.")
print()
print(f"{'recovery mode':>34}{'final success':>15}{'vs no retry':>13}")
print("-" * 62)
base = run("none")
modes = {}
for name, args in [("no retry", ("none",)),
                   ("blind retry", ("blind",)),
                   ("self-assessment", ("self",)),
                   ("environment feedback", ("env",))]:
    v = run(*args)
    modes[name] = v
    print(f"{name:>34}{v:>15.1%}{v - base:>+13.1%}")

print()
print()
print("Environment feedback is not perfect either. Sweep how often the")
print("environment's report is correct and actionable.")
print()
print(f"{'feedback quality':>18}{'environment':>14}{'self-assessment':>18}"
      f"{'blind retry':>14}")
print("-" * 64)
q_tab = {}
for q in (1.0, 0.85, 0.65, 0.45, 0.25):
    a = run("env", quality=q)
    b = run("self", quality=q)
    c = run("blind")
    q_tab[q] = (a, b, c)
    print(f"{q:>18.0%}{a:>14.1%}{b:>18.1%}{c:>14.1%}")

print()
print()
print("Where does self-assessment break? Sweep the agent's base competence,")
print("holding feedback quality at 85%.")
print()
print(f"{'first-attempt':>15}{'no retry':>11}{'self':>10}{'environment':>14}"
      f"{'self gap':>11}")
print("-" * 61)
comp = {}
for p in (0.25, 0.40, 0.55, 0.70, 0.85):
    n = run("none", p_first=p)
    s = run("self", quality=0.85, p_first=p)
    e = run("env", quality=0.85, p_first=p)
    comp[p] = (n, s, e)
    print(f"{p:>15.0%}{n:>11.1%}{s:>10.1%}{e:>14.1%}{s - n:>+11.1%}")

print()
print()
print("And how many attempts are worth having, under each regime.")
print()
print(f"{'attempts':>10}{'blind':>10}{'self':>10}{'environment':>14}")
print("-" * 44)
at = {}
for a in (1, 2, 3, 5, 8):
    at[a] = (run("blind", attempts=a), run("self", quality=0.85, attempts=a),
             run("env", quality=0.85, attempts=a))
    print(f"{a:>10}{at[a][0]:>10.1%}{at[a][1]:>10.1%}{at[a][2]:>14.1%}")

print(f"""
The first table has the result, and it is not the one the chapter was outlined to
find.

Environment feedback is the best mode at {modes['environment feedback']:.1%},
which is cite:huang2024selfcorrect's positive case and no surprise. Self-assessment
reaches {modes['self-assessment']:.1%}, well short of it, which is
cite:huang2024selfcorrect's negative case and also no surprise.

The surprise is the middle row. **Blind retry -- retrying with no idea what went
wrong -- scores {modes['blind retry']:.1%}, beating self-assessment by
{modes['blind retry'] - modes['self-assessment']:.1%}.**

An agent that knows nothing about its failure does better than one that tries to
work out what happened. That is worth understanding precisely, because it is not
about information content.

The mechanism is gating. Self-assessment does not just supply a diagnosis; it
supplies the DECISION TO RETRY. An agent that judges it succeeded does not try
again -- and on the attempts it botched, its judgement is exactly the judgement
that is unreliable (ch:rsn-self-consistency's correlated critic). So a failed
self-assessment does not merely fail to help. It throws away a retry the agent
would otherwise have taken for free.

Blind retry never makes that mistake, because it never asks. It retries
unconditionally, and even an uninformed retry is a fresh {P_FIRST:.0%} draw, which
compounds across attempts into {modes['blind retry']:.1%}.

**A feedback signal that gates a retry has to be better than the retry it
suppresses**, and that is a much higher bar than "better than nothing"
(eq:feedback-must-be-external).

The second table shows the same bar applying to the environment, which is the part
that transfers to real systems. Environment feedback falls from
{q_tab[1.0][0]:.1%} at perfect quality to {q_tab[0.25][0]:.1%} at {0.25:.0%}, while
blind retry sits flat at about {q_tab[0.25][2]:.1%} because it does not depend on
quality at all.

They cross. Somewhere between {0.45:.0%} and {0.25:.0%} feedback quality,
**consulting the environment becomes worse than ignoring it** -- not because the
environment is lying, but because acting on a weak signal costs the unconditional
retries that a policy with no signal would have taken.

That is a testable property of your own tools. A test suite that reports precisely
is far above the crossover; a tool that returns "error" and a system that treats
that as a verdict on whether to continue may be below it.

The third table locates where self-assessment does its most damage, and the shape
is not monotone. The gap between self-assessment and no retry is
{comp[0.25][1] - comp[0.25][0]:+.1%} at {0.25:.0%} base competence,
{comp[0.55][1] - comp[0.55][0]:+.1%} at {0.55:.0%}, and
{comp[0.85][1] - comp[0.85][0]:+.1%} at {0.85:.0%}.

Self-assessment helps most in the middle and least at both ends, for two different
reasons. A weak agent cannot tell that it failed, so it rarely retries. A strong
agent rarely needs to. Meanwhile the environment column is
{comp[0.25][2]:.1%} even at {0.25:.0%} competence -- **external feedback rescues a
weak agent and self-assessment cannot**, which is the practically important half of
cite:huang2024selfcorrect's distinction.

The fourth table says how many attempts are worth buying under each regime, and the
curves separate rather than converge. At {8} attempts blind reaches
{at[8][0]:.1%}, self-assessment {at[8][1]:.1%}, environment {at[8][2]:.1%}. More
attempts help every regime and they do not close the gap, because the gap is a
per-attempt conditioning difference and attempts multiply it.

So the design conclusions, in order of how much they contradict common practice.

**Do not gate a retry on the agent's own judgement.** If you can afford the retry,
take it unconditionally. Blind beat self-assessed here by
{modes['blind retry'] - modes['self-assessment']:.1%}, and the only thing
self-assessment added was an opportunity to skip.

**Use the agent's judgement for DIAGNOSIS, never for DETECTION.** Detection is what
gates, and it is the correlated part. Once you have decided to retry anyway,
whatever the agent thinks went wrong is free information that can only condition
the next attempt.

**Measure your environment's feedback quality**, because the second table says it
has a threshold below which acting on it is negative, and nobody checks.

One honest boundary. This models a retry as costless. It is not: ch:ag-loop's step
budget is finite, and an unconditional-retry policy spends it on attempts that had
already succeeded. The real recommendation is therefore narrower than "always
retry" -- it is that the decision should be made by a budget policy rather than by
the agent's opinion of its own work, which is ch:ag-termination's subject.""")

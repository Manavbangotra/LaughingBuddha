# -*- coding: utf-8 -*-
# Extracted from: Chapter 210 — Agent Tracing and Tool-Call Monitoring
# Source: src/.../ch210-agent-tracing.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Agent traces arrive faster than anything can triage them, and nothing can triage them.

A single-turn request produces one span worth reading. An agent request produces a
sequence of steps, each with a tool call, a result, and a model decision -- and the
failure is somewhere in the sequence rather than at a point.

cite:deshpande2025trail built a benchmark for exactly this task, localising the issue
inside an annotated agent trace, and the best model tested reached **11%**. Humans do
better and not by enough to matter at volume
(eq:triage-capacity-is-the-binding-constraint).

This listing measures the arrival rate against both triage channels and finds neither
scales, which makes trace STRUCTURE the only lever left.
"""
REQUESTS_PER_DAY = 42000.0
FAIL_RATE = 0.09              # agent requests that end badly
STEPS_MEAN = 7.4
HUMAN_MINUTES_PER_TRACE = 26.0
HUMAN_HOURS_PER_DAY = 6.0     # productive triage hours per engineer
MODEL_ACCURACY = 0.11         # cite:deshpande2025trail, best model on TRAIL
MODEL_COST_PER_TRACE = 0.34

failing = REQUESTS_PER_DAY * FAIL_RATE
print("An agent service at %.0f requests a day, %.0f%% ending badly."
      % (REQUESTS_PER_DAY, FAIL_RATE * 100))
print("That is %.0f failing traces a day, averaging %.1f steps each."
      % (failing, STEPS_MEAN))
print()
print("Human triage capacity, at %.0f minutes a trace." % HUMAN_MINUTES_PER_TRACE)
print()
print(f"{'engineers':>11}{'traces/day':>13}{'share of failures':>20}"
      f"{'days to clear one day':>24}")
print("-" * 70)
cap = {}
for n in (1, 2, 5, 10, 25, 100):
    per_day = n * HUMAN_HOURS_PER_DAY * 60.0 / HUMAN_MINUTES_PER_TRACE
    cap[n] = (per_day, per_day / failing)
    print(f"{n:>11}{per_day:>13.0f}{per_day / failing:>20.1%}"
          f"{failing / per_day:>24.1f}")

print()
print()
print("Automated triage: cheap per trace, and it is right %.0f%% of the time."
      % (MODEL_ACCURACY * 100))
print()
print(f"{'coverage':>11}{'traces/day':>13}{'cost/day':>11}"
      f"{'correctly localised':>22}{'wrong or unlocalised':>23}")
print("-" * 82)
auto = {}
for cov in (0.05, 0.25, 0.50, 1.00):
    n = failing * cov
    correct = n * MODEL_ACCURACY
    auto[cov] = (n, n * MODEL_COST_PER_TRACE, correct, n - correct)
    print(f"{cov:>11.0%}{n:>13.0f}{n * MODEL_COST_PER_TRACE:>11.0f}"
          f"{correct:>22.0f}{n - correct:>23.0f}")

print()
print()
print("Both channels together, and what is left untriaged.")
print()
print(f"{'engineers':>11}{'auto coverage':>15}{'localised/day':>16}"
      f"{'untriaged/day':>16}{'share untriaged':>18}")
print("-" * 78)
both = {}
for n in (2, 10, 25):
    for cov in (0.0, 0.5, 1.0):
        auto_correct = failing * cov * MODEL_ACCURACY
        # Humans triage what automation could not localise, up to capacity.
        remaining = failing - auto_correct
        human = min(cap[n][0], remaining)
        left = remaining - human
        both[(n, cov)] = (auto_correct + human, left)
        print(f"{n:>11}{cov:>15.0%}{auto_correct + human:>16.0f}"
              f"{left:>16.0f}{left / failing:>18.1%}")

print()
print()
print("The arithmetic that matters: how much triage effort a single percentage")
print("point of failure rate costs.")
print()
print(f"{'failure rate':>14}{'traces/day':>13}{'engineers to clear':>21}"
      f"{'annual cost':>14}")
print("-" * 64)
ENG_COST = 195000.0
for fr in (0.01, 0.03, 0.05, 0.09, 0.15):
    f = REQUESTS_PER_DAY * fr
    n_eng = f / (HUMAN_HOURS_PER_DAY * 60.0 / HUMAN_MINUTES_PER_TRACE)
    print(f"{fr:>14.0%}{f:>13.0f}{n_eng:>21.0f}{n_eng * ENG_COST:>14,.0f}")

print()
print("Clearing every failing trace is not a staffing plan. It is a different")
print("company.")

print()
print()
print("So the lever is not capacity. It is minutes per trace -- which structure")
print("controls.")
print()
print(f"{'minutes/trace':>15}{'traces/engineer/day':>22}"
      f"{'engineers for 10%':>20}{'engineers for 50%':>20}")
print("-" * 78)
mins = {}
for m in (26.0, 14.0, 8.0, 4.0, 1.5):
    per_eng = HUMAN_HOURS_PER_DAY * 60.0 / m
    mins[m] = per_eng
    print(f"{m:>15.1f}{per_eng:>22.0f}"
          f"{failing * 0.10 / per_eng:>20.1f}{failing * 0.50 / per_eng:>20.1f}")

print()
print()
print("And the same lever applied to the automated channel: a model localises")
print("better on a trace that is structured for it.")
print()
print(f"{'trace quality':>32}{'model accuracy':>17}{'localised/day at 100%':>24}"
      f"{'vs baseline':>13}")
print("-" * 88)
QUAL = [
    ("raw log lines",           0.11),
    ("+ explicit step boundaries", 0.19),
    ("+ tool inputs and outputs", 0.28),
    ("+ recorded intermediate state", 0.37),
    ("+ causal links between steps", 0.44),
]
qual = {}
for label, acc in QUAL:
    n = failing * acc
    qual[label] = (acc, n)
    print(f"{label:>32}{acc:>17.0%}{n:>24.0f}"
          f"{acc / MODEL_ACCURACY:>12.1f}x")

print(f"""
The capacity table is the first thing to look at and it settles the staffing question
immediately. At {HUMAN_MINUTES_PER_TRACE:.0f} minutes a trace, one engineer triages
{cap[1][0]:.0f} traces a day against {failing:.0f} failing ones -- **{cap[1][1]:.1%} of
them**. Twenty-five engineers reach {cap[25][1]:.1%}.

Clearing a single day's failures would take {failing / cap[1][0]:.0f} engineer-days
(eq:triage-capacity-is-the-binding-constraint). **Human triage is not a partial solution
here. It is a sampling strategy**, and it samples at whatever rate the headcount
happens to allow.

The automated table is the obvious alternative and cite:deshpande2025trail measured its
ceiling. At {MODEL_ACCURACY:.0%} accuracy, running automated triage over every failing
trace costs {auto[1.0][1]:.0f} a day and correctly localises {auto[1.0][2]:.0f} of
{failing:.0f} -- leaving {auto[1.0][3]:.0f} either wrong or unlocalised.

That is not a system anyone can act on. **An eleven percent localisation rate means
nearly nine in ten automated diagnoses are wrong**, and a wrong diagnosis is worse than
no diagnosis because someone acts on it.

The combined table shows the two channels do not rescue each other. With
{25} engineers and full automated coverage, {both[(25, 1.0)][1] / failing:.0%} of failing
traces are still untriaged. The automation removes {MODEL_ACCURACY:.0%} and the humans
remove what they can reach, and the sum is far short.

The failure-rate table is where this becomes a design constraint rather than an
operations problem. Every percentage point of failure rate costs
{REQUESTS_PER_DAY * 0.01 / (HUMAN_HOURS_PER_DAY * 60.0 / HUMAN_MINUTES_PER_TRACE):.0f}
engineers to triage exhaustively. At {0.09:.0%}, exhaustive triage is
{failing / (HUMAN_HOURS_PER_DAY * 60.0 / HUMAN_MINUTES_PER_TRACE):.0f} engineers and
{failing / (HUMAN_HOURS_PER_DAY * 60.0 / HUMAN_MINUTES_PER_TRACE) * ENG_COST / 1e6:.1f}
million a year.

**Nobody is going to staff that**, which means the question is never "how do we triage
everything" but "what do we do with the tiny share we can look at" -- and
ch:ops-observability's sampling result then decides which share that is.

The minutes-per-trace table is the lever that is actually available. Cutting triage from
{26.0:.0f} minutes to {8.0:.0f} takes one engineer from {mins[26.0]:.0f} traces a day to
{mins[8.0]:.0f}, so the engineers needed to cover a tenth of failures falls from
{failing * 0.10 / mins[26.0]:.1f} to {failing * 0.10 / mins[8.0]:.1f}.

**That is a {mins[8.0] / mins[26.0]:.1f}x capacity gain from making traces easier to
read**, and unlike headcount it compounds with the automated channel.

Which the last table shows. The same structural properties that make a trace fast for a
human to read make it tractable for a model: explicit step boundaries, recorded tool
inputs and outputs, intermediate state, causal links. Adding them takes automated
localisation from {MODEL_ACCURACY:.0%} to
{qual['+ causal links between steps'][0]:.0%} --
{qual['+ causal links between steps'][0] / MODEL_ACCURACY:.1f} times more traces
correctly localised, for the same model.

**Trace structure is the only lever that improves both channels at once**, and it is
the one nobody budgets for, because a trace format does not look like a reliability
investment. ch:ops-agent-tracing's second listing takes up what the structure has to
contain.""")

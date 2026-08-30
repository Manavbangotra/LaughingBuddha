# -*- coding: utf-8 -*-
# Extracted from: Chapter 186 — Test Generation, Debugging, and Refactoring Agents
# Source: src/.../ch186-testing-debugging.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Tests written by the model that wrote the code.

ch:aise-swe-agents found a test suite to be a partial specification, and everything
in this part rests on it being an INDEPENDENT one. A suite written by the developer
encodes what they intended; a patch that satisfies it is evidence because the
intention and the implementation came from different acts.

Generated tests break that. A model asked to write tests for code it just wrote
produces tests of what the code DOES, because that is what it is looking at. The
tests pass. They would also have passed on the bug (eq:same-model-tests-codify-the-bug).

This is ch:aids-autonomous's self-judging result in the domain with the strongest
verifier, and the mechanism is identical: a generator and a checker from one model
share the blind spot that produced the defect.
"""
import numpy as np

rng = np.random.default_rng(5147)

M = 60000
P_BUG = 0.22                # the code under test contains a defect
SPEC_COVERAGE = 0.75        # share of intended behaviour any good suite reaches


def run(author, m=M, p_bug=P_BUG, coverage=SPEC_COVERAGE, shown_code=True):
    """`author` is who wrote the tests:
       'human'      wrote them from the specification
       'other'      a different model, given the specification
       'same'       the model that wrote the code, given the code
       'same-spec'  the same model, given the SPECIFICATION and not the code
    Returns (catches the bug, tests pass on buggy code, coverage achieved).
    """
    buggy = rng.random(m) < p_bug

    if author == "human":
        # Written from intent, so a bug is caught wherever the suite reaches.
        reach, independence = coverage, 0.92
    elif author == "other":
        reach, independence = coverage * 0.97, 0.74
    elif author == "same-spec":
        # Same model, but writing from the specification rather than the code.
        reach, independence = coverage * 0.99, 0.46
    elif author == "same":
        # Generated from the code: high coverage of what the code does, and
        # almost no independence from it.
        reach, independence = min(coverage * 1.18, 0.97), 0.11
    else:
        raise ValueError(author)

    # A bug is caught only if the suite reaches that behaviour AND the test
    # encodes the INTENDED result rather than the observed one.
    caught = buggy & (rng.random(m) < reach) & (rng.random(m) < independence)
    passes_buggy = buggy & ~caught
    return (float(caught.mean() / max(p_bug, 1e-9)),
            float(passes_buggy.mean() / max(p_bug, 1e-9)),
            reach)


AUTHORS = [("human, from the spec", "human"),
           ("a different model, from the spec", "other"),
           ("the same model, from the spec", "same-spec"),
           ("the same model, from the code", "same")]

print(f"{M:,} modules, {P_BUG:.0%} of them containing a defect. A suite catches a")
print("defect only if it reaches that behaviour AND asserts what was INTENDED")
print("rather than what the code happens to do.")
print()
print(f"{'tests written by':>34}{'coverage':>11}{'catches the bug':>17}"
      f"{'passes on it':>14}")
print("-" * 76)
tab = {}
for label, a in AUTHORS:
    r = run(a)
    tab[label] = r
    print(f"{label:>34}{r[2]:>11.0%}{r[0]:>17.1%}{r[1]:>14.1%}")

print()
print()
print("Coverage and catch rate move in opposite directions here, which is the")
print("whole problem: generated tests achieve the HIGHEST coverage and the")
print("LOWEST defect detection.")
print()
print(f"{'tests written by':>34}{'coverage':>11}{'catch rate':>13}"
      f"{'coverage per catch':>20}")
print("-" * 78)
for label, a in AUTHORS:
    r = tab[label]
    print(f"{label:>34}{r[2]:>11.0%}{r[0]:>13.1%}"
          f"{r[2] / max(r[0], 1e-9):>20.2f}")

print()
print()
print("Writing tests from the SPECIFICATION rather than from the code recovers")
print("most of the independence, at no cost in coverage.")
print()
print(f"{'':>34}{'from the code':>15}{'from the spec':>15}{'gain':>9}")
print("-" * 73)
a_code = run("same")
a_spec = run("same-spec")
print(f"{'catch rate, same model':>34}{a_code[0]:>15.1%}{a_spec[0]:>15.1%}"
      f"{a_spec[0] - a_code[0]:>+9.1%}")
print(f"{'coverage':>34}{a_code[2]:>15.0%}{a_spec[2]:>15.0%}"
      f"{a_spec[2] - a_code[2]:>+9.0%}")

print()
print()
print("What a generated suite is worth as the verifier in ch:aise-swe-agents'")
print("loop -- the number that decides whether the agent's iteration means")
print("anything.")
print()
print(f"{'suite author':>34}{'usable as a verifier':>22}")
print("-" * 58)
for label, a in AUTHORS:
    r = tab[label]
    print(f"{label:>34}{r[0]:>22.1%}")

print()
print()
print("And what raising the generated suite's coverage buys, which is the")
print("counterintuitive part: 37 points of coverage move the catch rate by 4.")
print()
print(f"{'generated coverage':>20}{'catch rate':>13}{'passes on bug':>16}")
print("-" * 49)
gc = {}
for c in (0.60, 0.75, 0.90, 0.97):
    buggy = rng.random(M) < P_BUG
    caught = buggy & (rng.random(M) < c) & (rng.random(M) < 0.11)
    gc[c] = (float(caught.mean() / P_BUG),
             float((buggy & ~caught).mean() / P_BUG))
    print(f"{c:>20.0%}{gc[c][0]:>13.1%}{gc[c][1]:>16.1%}")

print()
print()
print("The independence term is what matters, and it is the one nobody reports.")
print()
print(f"{'independence':>14}{'catch rate':>13}{'equivalent human coverage':>27}")
print("-" * 54)
ind = {}
for i in (0.11, 0.30, 0.55, 0.92):
    buggy = rng.random(M) < P_BUG
    caught = buggy & (rng.random(M) < 0.90) & (rng.random(M) < i)
    cr = float(caught.mean() / P_BUG)
    ind[i] = cr
    print(f"{i:>14.0%}{cr:>13.1%}{cr / 0.92:>27.0%}")

print(f"""
The first table has the coverage and catch-rate columns moving in opposite
directions, which is the finding.

Tests generated from the code achieve the HIGHEST coverage in the table --
{tab['the same model, from the code'][2]:.0%} -- and catch
{tab['the same model, from the code'][0]:.1%} of defects. Human tests written from
the specification cover {tab['human, from the spec'][2]:.0%} and catch
{tab['human, from the spec'][0]:.1%}.

**A generated suite tests what the code does**, which is why its coverage is
excellent and its detection is not: if the code has a bug, the tests assert the
buggy behaviour and pass ({{eq:same-model-tests-codify-the-bug}}).

That is ch:aids-autonomous's self-judging result in the domain with the strongest
verifier, and the mechanism is identical -- a generator and a checker from one model
share the blind spot that produced the defect. Here it is worse than there, because
the checker is not merely correlated with the generator: it is looking at the
generator's output and describing it.

The coverage-per-catch column makes the trap concrete. Generated tests need
{tab['the same model, from the code'][2] / max(tab['the same model, from the code'][0], 1e-9):.1f}
points of coverage per point of catch rate; human tests need
{tab['human, from the spec'][2] / max(tab['human, from the spec'][0], 1e-9):.1f}.
**Coverage is the metric these tools raise and the metric that stops meaning
anything when they raise it.**

The third table is the intervention, and it is nearly free. Asking the SAME model to
write tests from the SPECIFICATION rather than from the code takes the catch rate
from {a_code[0]:.1%} to {a_spec[0]:.1%} -- {a_spec[0] - a_code[0]:+.1%} -- at a cost
of {a_spec[2] - a_code[2]:+.0%} in coverage.

**Do not show the model the implementation when asking it for tests.** That single
change recovers most of the independence, and it is a prompt-level decision rather
than an architecture.

A different model from the specification does better still at
{tab['a different model, from the spec'][0]:.1%}, which is
ch:as-failures' decorrelation result: independence is the input, and different
sources of it compose.

The fifth table is why "our generated tests reach 95% coverage" is not the
reassurance it sounds like. Moving generated coverage from {0.60:.0%} to
{0.97:.0%} moves the catch rate from {gc[0.60][0]:.1%} to {gc[0.97][0]:.1%} --
{37} points of coverage buying {(gc[0.97][0] - gc[0.60][0]) * 100:.0f} points of
detection.

**Independence is the binding term and coverage is the reported one.** The last
table prices it directly: at {0.11:.0%} independence a suite is worth what a
{ind[0.11] / 0.92:.0%}-coverage human suite would be; at {0.55:.0%} independence,
{ind[0.55] / 0.92:.0%}.

Which gives the rule for anyone using generated tests as the verifier in
ch:aise-swe-agents' loop: **a suite generated from the code is not a verifier**, and
an agent iterating against one is doing ch:ag-recovery's resample with extra steps.""")

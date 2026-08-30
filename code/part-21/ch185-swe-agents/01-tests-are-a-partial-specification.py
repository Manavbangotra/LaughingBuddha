# -*- coding: utf-8 -*-
# Extracted from: Chapter 185 — SWE Agents and Automated Issue Resolution
# Source: src/.../ch185-swe-agents.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Passing the tests is not the same as fixing the bug.

Software engineering has the strongest verifier in this book: tests execute, and
cite:jimenez2023swebench grades a patch by running the repository's own suite. That
is a real verifier, which is why coding agents are further along than agents in any
other domain.

cite:wang2025solvedcorrectly measured where it stops. Applying differential testing
to patches from three state-of-the-art tools on SWE-bench Verified, they found 7.8%
of patches counted correct while FAILING the developer-written test suite, 29.6% of
plausible patches behaving differently from the ground-truth human patch, 28.6% of
those divergent patches confirmed incorrect by manual inspection, and reported
resolution rates inflated by about 6.2 percentage points.

The mechanism is that a test suite is a PARTIAL SPECIFICATION. Passing it is
evidence proportional to its coverage, and coverage is never complete
(eq:tests-are-a-partial-specification).
"""
import numpy as np

rng = np.random.default_rng(5077)

M = 80000
P_PATCH_PLAUSIBLE = 0.42    # the agent produces something that compiles and runs
COVERAGE = 0.71             # share of the intended behaviour the suite pins down
P_RIGHT_GIVEN_PLAUSIBLE = 0.63


def run(m=M, coverage=COVERAGE, p_plaus=P_PATCH_PLAUSIBLE,
        p_right=P_RIGHT_GIVEN_PLAUSIBLE, differential=0.0, extra_tests=0.0):
    """Returns (passes the suite, actually correct, precision among passing).

    `differential` is the share of divergent-but-passing patches caught by
    comparing behaviour against the reference patch. `extra_tests` raises
    coverage toward 1.
    """
    cov = coverage + (1.0 - coverage) * extra_tests
    plausible = rng.random(m) < p_plaus
    correct = plausible & (rng.random(m) < p_right)
    wrong = plausible & ~correct
    # A wrong patch fails the suite only where the suite covers the divergence.
    caught_by_tests = wrong & (rng.random(m) < cov)
    passes = plausible & ~caught_by_tests
    # Differential testing against the reference catches divergences the suite
    # does not cover.
    survivors = passes & ~correct
    caught_diff = survivors & (rng.random(m) < differential)
    passes = passes & ~caught_diff

    p = float(passes.mean())
    c = float((passes & correct).mean())
    return p, c, (c / p if p else 0.0)


print(f"{M:,} issues. The agent produces a plausible patch {P_PATCH_PLAUSIBLE:.0%}")
print(f"of the time; {P_RIGHT_GIVEN_PLAUSIBLE:.0%} of plausible patches are right.")
print(f"The repository's test suite pins down {COVERAGE:.0%} of the intended")
print("behaviour.")
print()
print(f"{'':>34}{'share of issues':>17}")
print("-" * 53)
p, c, prec = run()
print(f"{'reported resolved (tests pass)':>34}{p:>17.1%}")
print(f"{'actually correct':>34}{c:>17.1%}")
print(f"{'inflation':>34}{p - c:>17.1%}")
print()
print(f"   Of patches that pass, {1 - prec:.1%} are wrong.")

print()
print()
print("Test coverage is what decides it, because the suite is the specification")
print("and an uncovered behaviour is unspecified.")
print()
print(f"{'coverage':>10}{'reported':>11}{'correct':>10}{'inflation':>12}"
      f"{'precision':>11}")
print("-" * 54)
cv = {}
for k in (0.40, 0.55, 0.71, 0.88, 0.98):
    r = run(coverage=k)
    cv[k] = r
    print(f"{k:>10.0%}{r[0]:>11.1%}{r[1]:>10.1%}{r[0] - r[1]:>12.1%}"
          f"{r[2]:>11.1%}")

print()
print()
print("Differential testing against the reference patch -- run both, compare")
print("behaviour on generated inputs -- catches divergences the suite misses.")
print()
print(f"{'differential catch':>20}{'reported':>11}{'correct':>10}"
      f"{'inflation':>12}{'precision':>11}")
print("-" * 64)
df = {}
for d in (0.0, 0.3, 0.6, 0.85):
    r = run(differential=d)
    df[d] = r
    print(f"{d:>20.0%}{r[0]:>11.1%}{r[1]:>10.1%}{r[0] - r[1]:>12.1%}"
          f"{r[2]:>11.1%}")

print()
print()
print("Which is worth more: more tests, or differential testing against the")
print("reference? Both attack the same gap from different sides.")
print()
print(f"{'intervention':>34}{'reported':>11}{'correct':>10}{'inflation':>12}")
print("-" * 67)
iv = {}
for label, kw in (("as is", {}),
                  ("coverage 71% -> 88%", {"extra_tests": 0.59}),
                  ("differential testing at 60%", {"differential": 0.60}),
                  ("both", {"extra_tests": 0.59, "differential": 0.60})):
    r = run(**kw)
    iv[label] = r
    print(f"{label:>34}{r[0]:>11.1%}{r[1]:>10.1%}{r[0] - r[1]:>12.1%}")

print()
print()
print("Note what differential testing does to the REPORTED number, which is")
print("why it is not adopted.")
print()
print(f"{'':>34}{'reported':>11}{'correct':>10}")
print("-" * 55)
print(f"{'without differential testing':>34}{iv['as is'][0]:>11.1%}"
      f"{iv['as is'][1]:>10.1%}")
print(f"{'with it':>34}{iv['differential testing at 60%'][0]:>11.1%}"
      f"{iv['differential testing at 60%'][1]:>10.1%}")
print()
print(f"   Reported falls {(iv['as is'][0] - iv['differential testing at 60%'][0]) * 100:.1f} points.")
print(f"   Correct is unchanged at {iv['as is'][1]:.1%} -- it was always that.")

print()
print()
print("And how the gap scales with how much the agent attempts. A tool that")
print("produces more plausible patches produces more passing-but-wrong ones.")
print()
print(f"{'plausible patch rate':>22}{'reported':>11}{'correct':>10}"
      f"{'inflation':>12}{'precision':>11}")
print("-" * 66)
pl = {}
for pp in (0.20, 0.42, 0.65, 0.85):
    r = run(p_plaus=pp)
    pl[pp] = r
    print(f"{pp:>22.0%}{r[0]:>11.1%}{r[1]:>10.1%}{r[0] - r[1]:>12.1%}"
          f"{r[2]:>11.1%}")

print(f"""
The first block is the shape of the problem. The pipeline reports {p:.1%} resolved
and {c:.1%} are correct, so **{1 - prec:.1%} of the patches that pass the tests are
wrong**.

cite:wang2025solvedcorrectly measured the real version of this on SWE-bench
Verified: 29.6% of plausible patches behaved differently from the ground-truth human
patch, 28.6% of those were confirmed incorrect on inspection, and reported resolution
rates were inflated by about 6.2 percentage points. This listing produces
{(p - c) * 100:.1f} points of inflation from an independent set of assumptions, which is the
same phenomenon at the same order.

The coverage table says why it happens, and it is not a defect in the grading. A
test suite is a **partial specification**: it pins down the behaviour someone thought
to write a test for, and a patch that satisfies it may do anything at all on the
rest. At {0.40:.0%} coverage the inflation is {cv[0.40][0] - cv[0.40][1]:.1%}; at
{0.98:.0%} it is {cv[0.98][0] - cv[0.98][1]:.1%}
(eq:tests-are-a-partial-specification).

**Passing a test suite is evidence proportional to its coverage**, and every
statement about coding agents' capability inherits whatever coverage the graded
repositories happened to have.

The differential-testing table is the fix, and it works: comparing the agent's patch
against the reference patch on generated inputs -- which is what
cite:wang2025solvedcorrectly's PatchDiff does -- takes inflation from
{df[0.0][0] - df[0.0][1]:.1%} to {df[0.85][0] - df[0.85][1]:.1%} and precision from
{df[0.0][2]:.1%} to {df[0.85][2]:.1%}.

Then the table that explains why it is not standard.

Adding differential testing takes the REPORTED number from {iv['as is'][0]:.1%} to
{iv['differential testing at 60%'][0]:.1%} -- a fall of
{(iv['as is'][0] - iv['differential testing at 60%'][0]) * 100:.1f} points -- while
the number of actually-correct patches does not move. It was always
{iv['as is'][1]:.1%}.

**A better verifier makes your headline worse and your knowledge better**, which is
exactly ch:aids-automl's leakage guard in a new setting. Any team or benchmark
reporting resolution rate is measured against the honest measurement, and the
incentive points away from it.

That is the second independent arrival of this structure in this book, and it is
worth naming as a general hazard: **whenever the metric is produced by the thing
being measured, improving the measurement is penalised.**

The last table is the one to carry forward. As the plausible-patch rate rises from
{0.20:.0%} to {0.85:.0%} -- which is what a better agent looks like -- inflation
rises from {pl[0.20][0] - pl[0.20][1]:.1%} to {pl[0.85][0] - pl[0.85][1]:.1%}
while precision stays flat at about {pl[0.85][2]:.0%}.

**A better agent produces more passing-but-wrong patches in absolute terms.** The
proportion holds and the volume grows, so the amount of plausible-wrong code entering
repositories scales with capability rather than shrinking with it -- unless coverage
or differential testing scales too.""")

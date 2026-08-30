# -*- coding: utf-8 -*-
# Extracted from: Chapter 179 — Agentic EDA, Cleaning, and Visualization
# Source: src/.../ch179-agentic-eda.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Exploration at machine speed, and what it does to what you find.

ch:aids-stack found the exploration stage to have almost no verifier: there is no
reference answer for "was this exploration adequate". This listing measures a
consequence that is worse than the absence of a check.

Exploration is a search over comparisons. Each comparison can turn up a real
effect or a spurious one, and the spurious rate is a property of the DATA rather
than of the analyst -- with enough comparisons on finite data, something always
looks significant. That is the multiple-comparisons problem, and it has been known
for a century.

What changes when an agent does the exploring is the count. A human runs twenty
comparisons in an afternoon; an agent runs eight hundred in a minute. The false
discovery rate is a function of that count (eq:exploration-is-a-search), so
automating exploration multiplies the false findings rather than the findings.
"""
import numpy as np

rng = np.random.default_rng(4591)

M = 3000                # datasets simulated
N_REAL = 6              # genuine effects present
N_NULL = 900            # comparisons available with no real effect
POWER = 0.72            # chance a real effect is detected when tested
ALPHA = 0.05


def effective_alpha(n_tests, correction, alpha, n_real):
    """The per-test threshold each method actually applies."""
    if correction == "bonferroni":
        return alpha / max(n_tests, 1)
    if correction == "fdr":
        # Benjamini-Hochberg sits between alpha and alpha/n, closer to alpha
        # when there are many true effects among the tests.
        return alpha * (1.0 + min(n_real, n_tests)) / max(n_tests, 1)
    return alpha


def explore(n_tests, m=M, n_real=N_REAL, n_null=N_NULL, power=POWER,
            alpha=ALPHA, correction=None, holdout=False, assume_tests=None):
    """Run `n_tests` comparisons. The analyst tests the hypotheses they came
    with FIRST and then explores, so the real effects are always among the
    tested set and every additional comparison is a null. That ordering is what
    makes exploration a forking path rather than a wider search.

    `assume_tests` corrects as if that many tests had been run, which is what
    happens when the count is not reported.
    """
    n_tests = min(n_tests, n_real + n_null)
    tested_real = min(n_tests, n_real)
    tested_null = n_tests - tested_real

    a = effective_alpha(assume_tests or n_tests, correction, alpha, n_real)
    # Tightening the threshold costs power.
    pw = power * min((a / alpha) ** 0.25, 1.0)

    tp = rng.binomial(tested_real, pw, m)
    fp = rng.binomial(tested_null, min(a, 1.0), m)

    if holdout:
        # Re-test every candidate on held-out data: real effects mostly
        # survive, spurious ones survive at the base rate.
        tp = rng.binomial(tp, 0.85)
        fp = rng.binomial(fp, alpha)

    reported = tp + fp
    prec = np.where(reported > 0, tp / np.maximum(reported, 1), np.nan)
    return (float(tp.mean()), float(fp.mean()),
            float(np.nanmean(prec)) if np.isfinite(prec).any() else 0.0)


print(f"A dataset with {N_REAL} genuine effects and {N_NULL} comparisons that")
print(f"have none. Each real effect is detected {POWER:.0%} of the time it is")
print(f"tested; each null comparison looks significant {ALPHA:.0%} of the time.")
print()
print(f"{'comparisons run':>17}{'true found':>12}{'false found':>13}"
      f"{'precision':>11}")
print("-" * 53)
tab = {}
for n in (10, 25, 100, 400, 900):
    r = explore(n)
    tab[n] = r
    print(f"{n:>17}{r[0]:>12.2f}{r[1]:>13.2f}{r[2]:>11.1%}")

print()
print()
print("The same, framed as the analyst experiences it: how many of the things")
print("you would report are real.")
print()
print(f"{'who':>28}{'comparisons':>13}{'reported':>11}{'real ones':>12}")
print("-" * 64)
WHO = [("human, an afternoon", 20), ("human, a week", 60),
       ("agent, a minute", 400), ("agent, an hour", 900)]
wh = {}
for label, n in WHO:
    r = explore(n)
    wh[label] = (n, r[0] + r[1], r[0])
    print(f"{label:>28}{n:>13}{r[0] + r[1]:>11.2f}{r[0]:>12.2f}")

print()
print()
print("Corrections. Bonferroni divides the threshold by the number of tests;")
print("FDR control is less severe; a held-out re-test is neither.")
print()
print(f"{'method':>22}" + "".join(f"{'n=' + str(n):>12}" for n in (25, 400, 900)))
print("-" * 58)
cm = {}
for label, kw in (("none", {}), ("Bonferroni", {"correction": "bonferroni"}),
                  ("FDR control", {"correction": "fdr"}),
                  ("held-out re-test", {"holdout": True})):
    row = tuple(explore(n, **kw)[2] for n in (25, 400, 900))
    cm[label] = row
    print(f"{label:>22}" + "".join(f"{v:>12.1%}" for v in row))

print()
print()
print("What each method costs in real findings missed, at 400 comparisons.")
print()
print(f"{'method':>22}{'true found':>12}{'false found':>13}{'precision':>11}")
print("-" * 58)
ct = {}
for label, kw in (("none", {}), ("Bonferroni", {"correction": "bonferroni"}),
                  ("FDR control", {"correction": "fdr"}),
                  ("held-out re-test", {"holdout": True})):
    r = explore(400, **kw)
    ct[label] = r
    print(f"{label:>22}{r[0]:>12.2f}{r[1]:>13.2f}{r[2]:>11.1%}")

print()
print()
print("And the interaction that matters: a correction assumes you know how many")
print("comparisons were run. An agent that explores and reports only what it")
print("found does not tell you.")
print()
print(f"{'actual comparisons':>20}{'corrected as if 25':>20}"
      f"{'corrected correctly':>21}")
print("-" * 61)
un = {}
for n in (25, 100, 400, 900):
    wrong = explore(n, correction="bonferroni", assume_tests=25)[2]
    right = explore(n, correction="bonferroni")[2]
    un[n] = (wrong, right)
    print(f"{n:>20}{wrong:>20.1%}{right:>21.1%}")

print(f"""
The first table's second column is the one to read first, because it does not
move. {N_REAL} real effects exist and about {tab[10][0]:.1f} of them are found by
{10} comparisons. Running {900} instead finds {tab[900][0]:.1f} -- the same ones.

The false column goes from {tab[10][1]:.2f} to {tab[900][1]:.2f}, and precision
from {tab[10][2]:.1%} to {tab[900][2]:.1%}.

**More exploration does not find more real effects. It finds more spurious ones**
(eq:exploration-is-a-search), because after the hypotheses you came with are
exhausted, every additional comparison is drawn from the null pool.

The second table is that result in the units that matter. A human exploring for an
afternoon reports {wh['human, an afternoon'][1]:.1f} findings of which
{wh['human, an afternoon'][2]:.1f} are real. An agent exploring for an hour reports
{wh['agent, an hour'][1]:.1f} of which {wh['agent, an hour'][2]:.1f} are real.

**Same discoveries, {wh['agent, an hour'][1] / wh['human, an afternoon'][1]:.0f}
times the noise.** That is what automating an activity with no verifier buys, and
it is worth stating in exactly those terms: the agent did not explore worse than
the human. It explored more, and more is the problem.

The correction table shows the standard remedies working. At {400} comparisons,
uncorrected precision is {cm['none'][1]:.1%}; Bonferroni gives {cm['Bonferroni'][1]:.1%}.

But the cost table is where the choice is made. Bonferroni at {400} comparisons
keeps {ct['Bonferroni'][0]:.2f} real findings out of the {ct['none'][0]:.2f}
available -- it discards
{1 - ct['Bonferroni'][0] / ct['none'][0]:.0%} of the genuine discoveries to buy its
precision.

A held-out re-test keeps {ct['held-out re-test'][0]:.2f} at
{ct['held-out re-test'][2]:.1%} precision.

**For exploration specifically, the holdout dominates the correction.** The purpose
of exploring is to generate candidate hypotheses, and a method that discards
{1 - ct['Bonferroni'][0] / ct['none'][0]:.0%} of them has defeated the purpose in
order to protect against reporting them. Re-testing on data the exploration never
saw keeps {ct['held-out re-test'][0] / ct['none'][0]:.0%} of the candidates and
still reaches {ct['held-out re-test'][2]:.1%}.

The last table is why this is not merely a statistics reminder, and it is the
chapter's actual argument.

A correction requires knowing how many comparisons were run. Correcting as though
{25} tests happened when {900} did gives {un[900][0]:.1%} precision against
{un[900][1]:.1%} for correcting correctly.

**An agent that explores and reports what it found does not tell you how much it
looked at.** It reports three interesting patterns; it does not report the eight
hundred it discarded, and often it has not counted. So the input that every
correction needs is precisely the quantity the automation destroys.

Which gives the rule for this chapter. **An exploratory agent must hold out data,
because it cannot be trusted to report its own denominator** -- and the holdout is
the only method here that does not require one.""")

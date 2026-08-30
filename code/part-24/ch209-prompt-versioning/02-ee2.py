# -*- coding: utf-8 -*-
# Extracted from: Chapter 209 — Prompt and Evaluation-Set Versioning
# Source: src/.../ch209-prompt-versioning.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""An evaluation set is a gate that decays, and nothing announces when it has.

A fixed evaluation set is a sample of the traffic distribution at the moment it was
built. Traffic moves -- new features, new users, new phrasings, new documents -- so the
set represents less of production every week.

The gate does not fail loudly when this happens. It keeps passing, at the same rate,
having stopped testing most of what the system now does
(eq:evaluation-sets-decay-silently).

This listing measures the decay, finds the refresh cadence that holds coverage, and shows
why the obvious alternative -- keep adding cases -- does not fix it.
"""
DRIFT_PER_WEEK = 0.035        # share of traffic distribution that is new each week
SET_SIZE = 900
WEEKS = [0, 4, 12, 26, 52, 104]
GATE_CATCH_ON_COVERED = 0.74  # P(the gate catches a defect in covered traffic)


def coverage(weeks_old):
    """Share of current traffic the set still represents."""
    return (1.0 - DRIFT_PER_WEEK) ** weeks_old


def gate_power(weeks_old):
    return coverage(weeks_old) * GATE_CATCH_ON_COVERED


print("Traffic distribution drifts %.1f%% a week. An evaluation set built once"
      % (DRIFT_PER_WEEK * 100))
print("covers less of it every week, and nothing in the gate reports this.")
print()
print(f"{'set age (weeks)':>17}{'coverage':>11}{'gate catches':>15}"
       f"{'escapes':>10}{'vs new set':>13}")
print("-" * 68)
tab = {}
for w in WEEKS:
    c = coverage(w)
    g = gate_power(w)
    tab[w] = (c, g, 1 - g)
    print(f"{w:>17}{c:>11.0%}{g:>15.0%}{1 - g:>10.0%}"
          f"{(1 - g) / (1 - gate_power(0)):>12.2f}x")

print()
print()
print("What the gate REPORTS while that happens: the pass rate on its own cases,")
print("which is unaffected by drift because those cases have not changed.")
print()
print(f"{'set age (weeks)':>17}{'pass rate reported':>21}{'true catch rate':>18}"
       f"{'gap':>9}")
print("-" * 68)
for w in WEEKS:
    reported = 0.93           # the suite keeps passing at its usual rate
    print(f"{w:>17}{reported:>21.0%}{tab[w][1]:>18.0%}"
          f"{reported - tab[w][1]:>9.0%}")

print()
print("The reported number does not move. That is the failure mode.")

print()
print()
print("Refresh cadence: how often the set must be regenerated to hold coverage.")
print()
print(f"{'refresh every':>16}{'coverage at worst':>20}{'mean coverage':>16}"
       f"{'regenerations/yr':>19}")
print("-" * 74)
cad = {}
for weeks in (2, 4, 8, 13, 26, 52):
    worst = coverage(weeks)
    mean = sum(coverage(w) for w in range(weeks)) / weeks
    cad[weeks] = (worst, mean, 52.0 / weeks)
    print(f"{weeks:>14}w{worst:>20.0%}{mean:>16.0%}{52.0 / weeks:>19.1f}")

print()
print()
print("The cost of each cadence, against what it prevents.")
print()
LABEL_COST_PER_CASE = 4.10
DEFECTS_PER_WEEK = 1.7
DEFECT_COST = 2400.0
print(f"{'refresh every':>16}{'labelling/yr':>15}{'escapes/yr':>13}"
       f"{'escape cost/yr':>17}{'total/yr':>12}")
print("-" * 76)
tot = {}
for weeks in (2, 4, 8, 13, 26, 52):
    label = SET_SIZE * LABEL_COST_PER_CASE * (52.0 / weeks)
    esc = DEFECTS_PER_WEEK * 52.0 * (1 - cad[weeks][1] * GATE_CATCH_ON_COVERED)
    tot[weeks] = (label, esc, esc * DEFECT_COST, label + esc * DEFECT_COST)
    print(f"{weeks:>14}w{label:>15,.0f}{esc:>13.1f}"
          f"{esc * DEFECT_COST:>17,.0f}{label + esc * DEFECT_COST:>12,.0f}")

best = min(tot, key=lambda k: tot[k][3])
print()
print(f"cheapest cadence: every {best} weeks at {tot[best][3]:,.0f} a year")

print()
print()
print("Why growing the set does not substitute for refreshing it.")
print()
print(f"{'strategy':>34}{'cases':>9}{'coverage':>11}{'catches':>10}"
       f"{'labelling/yr':>15}")
print("-" * 80)
STRATS = [
    ("900 cases, never refreshed",       900,  coverage(52), 0),
    ("1800 cases, never refreshed",     1800,  coverage(52), 900),
    ("3600 cases, never refreshed",     3600,  coverage(52), 2700),
    ("900 cases, refreshed quarterly",   900,  cad[13][1],   900 * 4),
    ("900 cases, refreshed monthly",     900,  cad[4][1],    900 * 13),
]
for label, n, cov, new_cases in STRATS:
    print(f"{label:>34}{n:>9}{cov:>11.0%}"
          f"{cov * GATE_CATCH_ON_COVERED:>10.0%}"
          f"{new_cases * LABEL_COST_PER_CASE:>15,.0f}")

print()
print()
print("And the sampling design that keeps a set current for less: replace the")
print("oldest slice each period rather than regenerating the whole set.")
print()
print(f"{'replace per month':>19}{'mean age (weeks)':>19}{'coverage':>11}"
       f"{'labelling/yr':>15}")
print("-" * 66)
roll = {}
for frac in (1.00, 0.50, 0.25, 0.10, 0.05):
    mean_age = 4.0 / (2.0 * frac) if frac > 0 else 999.0
    c = coverage(mean_age)
    cases = SET_SIZE * frac * 13.0
    roll[frac] = (mean_age, c, cases * LABEL_COST_PER_CASE)
    print(f"{frac:>19.0%}{mean_age:>19.1f}{c:>11.0%}"
          f"{cases * LABEL_COST_PER_CASE:>15,.0f}")

print(f"""
The decay table is the mechanism. A set built today covers {coverage(0):.0%} of traffic;
at {26} weeks it covers {tab[26][0]:.0%} and at {104} it covers {tab[104][0]:.0%}
(eq:evaluation-sets-decay-silently).

The gate's catch rate falls with it -- from {tab[0][1]:.0%} to {tab[104][1]:.0%} -- so
after two years a suite that was catching three quarters of defects catches under a
fifth.

The second table is why nobody notices. **The gate keeps reporting the same pass rate**,
because it is running the same cases against a system that still handles those cases.
Its own measurement is unaffected by the drift. The reported number stays at
{0.93:.0%} while the true catch rate falls to {tab[104][1]:.0%} -- a gap of
{0.93 - tab[104][1]:.0%}, with nothing to indicate it.

That is ch:sd-architecture's pattern once more, and this instance is particularly clean:
**the instrument is measuring exactly what it was built to measure, and what it was built
to measure stopped being the question.**

The cadence table gives the fix and its price. Refreshing every {4} weeks holds mean
coverage at {cad[4][1]:.0%}; every {26} weeks holds {cad[26][1]:.0%}; annually,
{cad[52][1]:.0%}.

The cost table finds the optimum at **every {best} weeks**, at {tot[best][3]:,.0f} a
year against {tot[52][3]:,.0f} for annual refresh -- driven by escape cost rather than
labelling cost, since labelling {SET_SIZE} cases is
{SET_SIZE * LABEL_COST_PER_CASE:,.0f} a time and one escaped defect is
{DEFECT_COST:,.0f}.

The growth table is the intervention teams reach for instead, and it does not work.
Quadrupling the set to {3600} cases without refreshing leaves coverage at
{coverage(52):.0%} and the catch rate at {coverage(52) * GATE_CATCH_ON_COVERED:.0%},
because **a bigger sample of an old distribution is still a sample of an old
distribution**. Refreshing {900} cases quarterly reaches
{cad[13][1] * GATE_CATCH_ON_COVERED:.0%} for less labelling than the quadrupling cost.

Size answers a variance question and age answers a bias question, and adding cases
addresses the wrong one.

The rolling table is the design that gets the coverage cheaply. Replacing
{0.25:.0%} of the set each month keeps mean case age at {roll[0.25][0]:.1f} weeks and
coverage at {roll[0.25][1]:.0%}, for {roll[0.25][2]:,.0f} a year of labelling --
against {roll[1.0][2]:,.0f} for full monthly regeneration.

**A rolling refresh gets most of a full regeneration's coverage for a quarter of the
labelling**, and it has a second advantage the table does not show: it produces a steady
small labelling workload rather than a periodic large one, which is the difference
between a process that survives and one that gets skipped when a quarter is busy.

One caution about all of this. The drift rate is the parameter everything depends on and
it is the one nobody measures. It is estimable -- compare the embedding distribution of
this month's traffic against the evaluation set's -- and until it is measured, the
cadence above is arithmetic on a guess.""")

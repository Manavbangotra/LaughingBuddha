# -*- coding: utf-8 -*-
# Extracted from: Chapter 220 — Online Evaluation, A/B Testing, and Regression Gates
# Source: src/.../ch220-online.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A regression gate is a hypothesis test run every release, and that is the problem.

Run one test at 5% and you get a false alarm one time in twenty. Run it on twelve metrics
across forty releases a quarter and you get several hundred opportunities, so false alarms
arrive weekly (eq:gate-alarms-multiply-with-metrics-and-releases).

Teams respond by ignoring the gate, which is the rational response to an instrument whose
alarms are mostly noise.

The deeper problem is sizing. A gate can only block what it can detect, and the minimum
detectable effect at your traffic is often larger than the regression you were trying to
refuse -- at which point the gate is decorative
(eq:a-gate-is-useless-if-mde-exceeds-tolerance).
"""
import math

RELEASES_PER_QUARTER = 40
ALPHA = 0.05
INVESTIGATION_COST = 3100.0


def family_alarms(metrics, alpha, releases):
    per_release = 1.0 - (1.0 - alpha) ** metrics
    return per_release, per_release * releases


print(f"{RELEASES_PER_QUARTER} releases a quarter, each gated at "
      f"alpha = {ALPHA:.2f}.")
print()
print(f"{'gating metrics':>16}{'P(alarm) per release':>23}"
      f"{'false alarms/quarter':>23}{'cost/quarter':>15}")
print("-" * 77)
fam = {}
for m in (1, 3, 6, 12, 25):
    pr, tot = family_alarms(m, ALPHA, RELEASES_PER_QUARTER)
    fam[m] = (pr, tot, tot * INVESTIGATION_COST)
    print(f"{m:>16}{pr:>23.1%}{tot:>23.1f}"
          f"{tot * INVESTIGATION_COST:>15,.0f}")

print()
print(f"at {12} metrics the gate cries wolf {fam[12][1]:.0f} times a quarter")
print("with nothing wrong")

print()
print()
print("Corrections, and what each costs in sensitivity.")
print()
print(f"{'correction':>28}{'effective alpha':>18}{'false alarms/qtr':>19}"
      f"{'power to see a real 3% drop':>31}")
print("-" * 96)
POWER_Z_A = 1.96
BASE_SE = 0.0138              # standard error of the metric at one release's traffic
REAL_DROP = 0.030


def power_at(alpha, effect, se):
    z = abs(effect) / se - (-math.log(alpha / 2.0)) ** 0.5 * 1.25
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


CORRECTIONS = [
    ("none, 12 metrics",            ALPHA),
    ("Bonferroni over 12",          ALPHA / 12),
    ("Bonferroni over 12 x 40",     ALPHA / (12 * 40)),
    ("pre-register 3 metrics",      ALPHA / 3),
    ("pre-register 3, one-sided",   2 * ALPHA / 3),
]
corr = {}
for name, a in CORRECTIONS:
    m = 12 if "12" in name else 3
    pr, tot = family_alarms(m, a, RELEASES_PER_QUARTER)
    pw = power_at(a, REAL_DROP, BASE_SE)
    corr[name] = (a, tot, pw)
    print(f"{name:>28}{a:>18.5f}{tot:>19.1f}{pw:>31.1%}")

print()
print("Every correction that removes false alarms removes power with it.")

print()
print()
print("The sizing question, which is the one that decides whether a gate works.")
print()
TRAFFIC = 42000.0
OBS_SHARE = 0.46
BASE_RATE = 0.612
print(f"{'canary share':>14}{'observations':>15}{'std error':>12}"
      f"{'MDE at 80% power':>19}{'blocks a 3% drop?':>20}")
print("-" * 80)
TOLERABLE = 0.03 * BASE_RATE
mde = {}
for share in (0.01, 0.05, 0.20, 0.50, 1.00):
    n = TRAFFIC * OBS_SHARE * share
    se = math.sqrt(2.0 * BASE_RATE * (1 - BASE_RATE) / max(n, 1.0))
    d = 2.80 * se
    mde[share] = (n, se, d)
    print(f"{share:>14.0%}{n:>15,.0f}{se:>12.5f}{d / BASE_RATE:>18.2%}"
          f"{('yes' if d <= TOLERABLE else 'no'):>20}")

print()
print(f"tolerable regression: {TOLERABLE / BASE_RATE:.1%} relative "
      f"({TOLERABLE:.4f} absolute)")

print()
print()
print("Same question across metrics, at a 20% canary for one day.")
print()
SHARE = 0.20
METRICS = [
    ("task completed",        0.612, 0.487, 0.46, 0.030),
    ("faithfulness (judge)",  0.734, 0.442, 1.00, 0.020),
    ("thumbs-up rate",        0.077, 0.267, 0.14, 0.050),
    ("issue resolved",        0.680, 0.466, 0.008, 0.030),
    ("p95 latency (s)",       3.100, 1.900, 1.00, 0.080),
]
print(f"{'metric':>22}{'obs in canary':>16}{'MDE':>10}"
      f"{'tolerance':>12}{'usable gate?':>15}{'days to usable':>17}")
print("-" * 92)
gate_ok = {}
for name, mu, sd, obs, tol in METRICS:
    n = TRAFFIC * obs * SHARE
    se = math.sqrt(2.0) * sd / math.sqrt(max(n, 1.0))
    d = 2.80 * se / mu
    days = (d / tol) ** 2
    gate_ok[name] = (n, d, tol, d <= tol, days)
    print(f"{name:>22}{n:>16,.0f}{d:>10.2%}{tol:>12.1%}"
          f"{('yes' if d <= tol else 'no'):>15}{days:>17.1f}")

print()
print("`days to usable` is how long the canary must run for the gate to be")
print("able to detect the regression it exists to refuse")

print()
print()
print("What a gate that cannot detect its tolerance actually does.")
print()
print(f"{'metric':>22}{'MDE / tolerance':>18}{'blocks':>26}"
      f"{'lets through':>26}")
print("-" * 92)
for name, mu, sd, obs, tol in METRICS:
    n, d, t, ok, days = gate_ok[name]
    ratio = d / t
    if ok:
        blocks = "the regression it targets"
        lets = "smaller ones"
    else:
        blocks = f"drops above {d:.1%}"
        lets = f"everything up to {d:.1%}"
    print(f"{name:>22}{ratio:>18.2f}{blocks:>26}{lets:>26}")

print()
print()
print("And the alternative that does not need statistical significance at all.")
print()
print(f"{'design':>34}{'what it tests':>28}{'traffic needed':>17}"
      f"{'catches':>16}")
print("-" * 95)
ALTS = [
    ("two-sided test on the mean", "did anything change", "high", "large drops"),
    ("one-sided non-inferiority",  "is it worse than -3%", "medium", "the tolerance"),
    ("paired offline replay",      "same inputs, both models", "none", "small drops"),
    ("frozen-case assertion",      "these 40 cases still pass", "none", "known regressions"),
    ("guardrail with a hard floor", "never below an absolute", "low", "catastrophes"),
]
for name, tests, traffic, catches in ALTS:
    print(f"{name:>34}{tests:>28}{traffic:>17}{catches:>16}")

print(f"""
The family table is the arithmetic every gated pipeline eventually runs into. Twelve gating
metrics at alpha {ALPHA:.2f} give a {fam[12][0]:.1%} chance of at least one false alarm per
release, which across {RELEASES_PER_QUARTER} releases is **{fam[12][1]:.0f} false alarms a
quarter** costing {fam[12][2]:,.0f} in investigation
(eq:gate-alarms-multiply-with-metrics-and-releases).

That is roughly two a week, on a system where nothing is wrong. Teams do not tolerate that
for long, and what they do about it is stop reading the alarms -- which is a rational response
to an instrument whose output is mostly noise, and which then applies to the real alarm too.

The correction table shows the trade every fix makes. Bonferroni over twelve metrics and
forty releases takes false alarms to {corr['Bonferroni over 12 x 40'][1]:.2f} a quarter and
power to see a genuine {REAL_DROP:.0%} drop to
{corr['Bonferroni over 12 x 40'][2]:.1%}. **A gate that never cries wolf also never barks.**

The row that works is `{CORRECTIONS[4][0]}`: pre-register a small set of metrics, test
one-sided because you only care about regressions, and accept
{corr['pre-register 3, one-sided'][1]:.1f} false alarms a quarter for
{corr['pre-register 3, one-sided'][2]:.1%} power. **Pre-registration is the cheap correction
and it is a discipline rather than a statistic** -- deciding in advance which three numbers
can block a release costs nothing and removes most of the family.

The sizing table is the more serious problem, and it is the one that makes gates decorative.
At a {0.05:.0%} canary the minimum detectable effect is {mde[0.05][2] / BASE_RATE:.1%}
relative, against a tolerable regression of {TOLERABLE / BASE_RATE:.1%}. **The gate cannot
see what it exists to refuse** (eq:a-gate-is-useless-if-mde-exceeds-tolerance), and it will
pass every release regardless of what is in it.

That is not a subtle failure. It is a gate reporting `pass` with no capacity to report
anything else, and it is indistinguishable from a working gate for as long as nothing goes
catastrophically wrong.

This is ch:ops-deployment's canary result meeting cite:card2020power's, and the two compose
badly: a small canary was chosen to limit exposure, and it limits detection by the same
factor.

The per-metric table says which gates in a realistic set are real, and the answer is one of
five. `{METRICS[4][0]}` is usable at a {SHARE:.0%} canary in
{gate_ok[METRICS[4][0]][4]:.1f} days, because it is observed on every request and its
tolerance is loose. `{METRICS[1][0]}` misses by a hair -- {gate_ok[METRICS[1][0]][1]:.2%}
against a {METRICS[1][4]:.1%} tolerance -- and reaches it in
{gate_ok[METRICS[1][0]][4]:.1f} days. `{METRICS[3][0]}` needs
{gate_ok[METRICS[3][0]][4]:.0f} days and will be gating on nothing until then.

**Four of the five gates in a plausible set cannot detect what they were written to
refuse**, at the canary size and duration a team would actually run.

The `days to usable` column is the number to compute before writing a gate. If it exceeds
your canary duration, you are not building a gate -- you are building a dashboard that
occasionally lights up, and ch:ops-deployment already priced what that costs.

The last table is the way out and its first two rows are the point. The default design tests
whether *anything changed*, which is a two-sided question requiring the most traffic and
answering the least useful question. A **one-sided non-inferiority test** -- is the new
system worse than the old by more than the tolerance? -- is the question a gate is actually
asking, and it is cheaper.

Cheaper still are the designs that need no live traffic at all. A paired offline replay runs
both models over the same inputs, which removes the between-user variance entirely -- the
same factor of three ch:ev-online's first listing measured. A frozen-case assertion asks
whether forty specific known-good cases still pass, which is not a statistical test at all
and catches exactly the regressions somebody has already seen.

**Most of what a regression gate needs to do does not require an experiment**, and the part
that does requires a one-sided test against a stated tolerance rather than a search for
significance. That is the whole recommendation, and it fits in a sentence: state the
regression you refuse to ship, check whether you can detect it, and if you cannot, fix the
design before writing the gate.""")

# -*- coding: utf-8 -*-
# Extracted from: Chapter 220 — Online Evaluation, A/B Testing, and Regression Gates
# Source: src/.../ch220-online.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The experiment you can afford measures a proxy; the decision you need is about the outcome.

cite:card2020power found that typical NLP test sets are underpowered for the differences
routinely claimed on them. Online experiments have the same problem with an additional
twist: the quantity a product cares about -- task success, resolution, retention -- is rarer
and noisier than the quantity that is easy to log
(eq:experiment-duration-is-set-by-outcome-variance).

So teams run the experiment on a proxy, get an answer in days instead of months, and inherit
a decision error whose size is set by the proxy's correlation with the outcome
(eq:a-fast-proxy-buys-speed-with-decision-error).

This listing computes both durations, prices the substitution, and measures what variance
reduction buys.
"""
import math

TRAFFIC_PER_DAY = 42000.0
POWER_Z = 2.80                # z(0.80) + z(0.975)

# (metric, mean, per-observation SD, share of sessions where it is observed,
#  correlation with the true outcome)
METRICS = [
    ("click on first result", 0.410, 0.492, 1.00, 0.31),
    ("session length",        4.900, 3.800, 1.00, 0.22),
    ("thumbs-up rate",        0.077, 0.267, 0.14, 0.58),
    ("task completed",        0.612, 0.487, 0.46, 0.86),
    ("issue resolved (survey)", 0.680, 0.466, 0.008, 1.00),
]
EFFECT_REL = 0.03             # the relative improvement we want to detect

print(f"{TRAFFIC_PER_DAY:,.0f} sessions a day. Detecting a "
      f"{EFFECT_REL:.0%} relative change.")
print()
dur = {}
for name, mu, sd, obs, rho in METRICS:
    d = EFFECT_REL * mu
    n = (POWER_Z ** 2) * 2.0 * sd ** 2 / (d ** 2)
    days = n / (TRAFFIC_PER_DAY * obs / 2.0)
    dur[name] = (n, days, rho)

survey_days = dur["issue resolved (survey)"][1]
print(f"{'metric':>24}{'observed on':>13}{'CV':>8}{'obs needed/arm':>17}"
      f"{'days':>9}{'vs outcome':>13}")
print("-" * 84)
for name, mu, sd, obs, rho in METRICS:
    print(f"{name:>24}{obs:>13.1%}{sd / mu:>8.2f}{dur[name][0]:>17,.0f}"
          f"{dur[name][1]:>9.1f}{dur[name][1] / survey_days:>12.3f}x")

print()
print("The metric the product is about takes "
      f"{dur['issue resolved (survey)'][1]:.0f} days;")
print(f"the one that is easy to log takes {dur['click on first result'][1]:.1f}.")

print()
print()
print("So the experiment runs on a proxy. What does that cost in decisions?")
print()
TRUE_EFFECT_SD = 0.020        # spread of true effects across the changes we test
print(f"{'proxy':>24}{'rho':>7}{'days':>8}{'P(right call)':>16}"
      f"{'P(ship a regression)':>23}{'P(kill a win)':>16}")
print("-" * 94)


def phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


dec = {}
for name, mu, sd, obs, rho in METRICS:
    # A change with true effect t shows a proxy effect rho*t plus independent noise.
    # Decide "ship" when the proxy is positive and significant.
    p_ship_given_good = phi(rho / math.sqrt(1 - rho ** 2 + 1e-9) * 0.9)
    p_ship_given_bad = 1.0 - p_ship_given_good
    right = 0.5 * p_ship_given_good + 0.5 * (1 - p_ship_given_bad)
    dec[name] = (rho, dur[name][1], right, 0.5 * p_ship_given_bad,
                 0.5 * (1 - p_ship_given_good))
    print(f"{name:>24}{rho:>7.2f}{dur[name][1]:>8.1f}{right:>16.1%}"
          f"{0.5 * p_ship_given_bad:>23.1%}{0.5 * (1 - p_ship_given_good):>16.1%}")

print()
print("Half the candidate changes are genuine improvements and half are not.")

print()
print()
print("Speed against correctness, priced per decision.")
print()
WIN_VALUE = 190000.0          # annual value of shipping a real improvement
REGRESSION_COST = 240000.0    # annual cost of shipping a regression
DELAY_COST_DAY = 2600.0       # cost of a day of experiment for one change
econ = {}
for name, mu, sd, obs, rho in METRICS:
    r, days, right, ship_bad, kill_good = dec[name]
    delay = days * DELAY_COST_DAY
    err = ship_bad * REGRESSION_COST + kill_good * WIN_VALUE
    econ[name] = (delay, err, delay + err)
base_tot = econ["issue resolved (survey)"][2]
print(f"{'proxy':>24}{'days':>8}{'delay cost':>13}{'error cost':>13}"
      f"{'total':>12}{'vs outcome':>13}")
print("-" * 83)
for name, mu, sd, obs, rho in METRICS:
    delay, err, tot = econ[name]
    print(f"{name:>24}{dec[name][1]:>8.1f}{delay:>13,.0f}{err:>13,.0f}"
          f"{tot:>12,.0f}{tot / base_tot:>12.2f}x")

best = min(econ, key=lambda n: econ[n][2])
print()
print(f"cheapest decision: {best} at {econ[best][2]:,.0f}")

print()
print()
print("Variance reduction: what a paired or covariate-adjusted design buys.")
print()
print(f"{'design':>34}{'variance factor':>18}{'days on task completed':>25}"
      f"{'saving':>10}")
print("-" * 87)
BASE_DAYS = dur["task completed"][1]
DESIGNS = [
    ("simple randomisation",            1.00),
    ("stratify by surface",             0.88),
    ("stratify by surface and tenure",  0.79),
    ("covariate adjustment on history", 0.54),
    ("paired: same query, both arms",   0.31),
]
vr = {}
for name, f in DESIGNS:
    d = BASE_DAYS * f
    vr[name] = d
    print(f"{name:>34}{f:>18.2f}{d:>25.1f}{BASE_DAYS - d:>10.1f}")

print()
print(f"the paired design takes `task completed` from {BASE_DAYS:.1f} days to "
      f"{vr['paired: same query, both arms']:.1f}")

print()
print()
print("Which changes the answer: with variance reduction, can we afford the")
print("metric we actually care about?")
print()
print(f"{'metric':>24}{'days, simple':>15}{'days, paired':>15}"
      f"{'affordable in a sprint?':>26}")
print("-" * 80)
SPRINT = 14.0
for name, mu, sd, obs, rho in METRICS:
    simple = dur[name][1]
    paired = simple * 0.31
    print(f"{name:>24}{simple:>15.1f}{paired:>15.1f}"
          f"{('yes' if paired <= SPRINT else 'no'):>26}")

print(f"""
The duration table is cite:card2020power's problem in production form. Detecting a
{EFFECT_REL:.0%} relative change on `{METRICS[0][0]}` takes
{dur[METRICS[0][0]][1]:.1f} days; on `{METRICS[4][0]}` it takes
{dur[METRICS[4][0]][1]:.0f} (eq:experiment-duration-is-set-by-outcome-variance).

Two things drive that gap and only one of them is variance. The survey metric is observed on
{METRICS[4][3]:.1%} of sessions against {METRICS[0][3]:.0%} for a click, so the effective
sample rate is {METRICS[0][3] / METRICS[4][3]:.0f} times lower before any noise is
considered. **The metric the product is about is the one you have least of.**

That is the pressure every experimentation programme is under, and the substitution it makes
is entirely rational: run on the proxy, decide in a week, move on.

The decision table prices what the substitution costs. At a proxy correlation of
{METRICS[0][4]:.2f}, the right call is made {dec[METRICS[0][0]][2]:.1%} of the time --
a regression is shipped {dec[METRICS[0][0]][3]:.1%} of the time and a genuine win is killed
{dec[METRICS[0][0]][4]:.1%} of the time (eq:a-fast-proxy-buys-speed-with-decision-error).

At {METRICS[3][4]:.2f} -- `{METRICS[3][0]}`, a proxy that is close to the outcome without
being it -- the right call is made {dec[METRICS[3][0]][2]:.1%} of the time.

**A proxy is not a cheaper measurement of the same thing. It is a different measurement with
a known error rate**, and the error rate is computable from the correlation, which almost
nobody estimates before adopting the proxy.

The economics table puts the two effects on one scale. Running on the survey metric costs
{econ[METRICS[4][0]][0]:,.0f} in delay and {econ[METRICS[4][0]][1]:,.0f} in decision error;
running on the click costs {econ[METRICS[0][0]][0]:,.0f} and
{econ[METRICS[0][0]][1]:,.0f}. The cheapest decision is on
`{best}` at {econ[best][2]:,.0f}.

Which is the useful result: **the right proxy is neither the fastest nor the most correlated,
it is the one minimising delay plus error**, and finding it requires estimating a correlation
and a delay cost -- two numbers most teams treat as unknowable and both of which are
estimable from a quarter of history.

The variance-reduction table is the intervention that changes the frontier rather than moving
along it. A paired design -- run the same query through both arms and compare -- takes
`{METRICS[3][0]}` from {BASE_DAYS:.1f} days to
{vr['paired: same query, both arms']:.1f}, because the query's own difficulty cancels.

That is the single most valuable design choice available in AI experimentation and it is
available in AI experimentation specifically. Most online experiments cannot pair, because a
user cannot be shown both arms. **A model comparison can**: the same input, both systems,
offline, at inference cost. ch:ev-llm-judge's both-orders protocol is the same idea applied to
a judge, and it works for the same reason.

The last table is the payoff. Under simple randomisation, `{METRICS[3][0]}` needs
{dur[METRICS[3][0]][1]:.1f} days and does not fit in a sprint. Paired, it needs
{dur[METRICS[3][0]][1] * 0.31:.1f} and does. The survey metric remains out of reach at
{dur[METRICS[4][0]][1] * 0.31:.0f} days even paired, which is honest: some questions cannot be
answered in a sprint and should be answered quarterly rather than proxied weekly.

Two rules to carry into ch:ev-online's second half. **Estimate the proxy's correlation before
adopting it**, because that number is the decision error you are buying. And **pair whenever
the design allows it**, because a factor of three on duration is worth more than any metric
choice on this list.""")

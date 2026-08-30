# -*- coding: utf-8 -*-
# Extracted from: Chapter 195 — Fault Tolerance: Retries, Timeouts, and Circuit Breakers
# Source: src/.../ch195-fault-tolerance.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A circuit breaker cannot trip on an error rate nobody measures.

ch:sd-architecture found circuit breakers surviving at 12%, because a breaker trips
on observable errors and semantic failure returns 200 OK. This listing asks the
follow-up question: what would it take to build a breaker that CAN see semantic
failure, and is it affordable?

The mechanism is sampling. Review a share of answers, watch the measured error rate,
and trip when it shifts. Detection time then falls with the sample rate and with the
square of the effect size (eq:detection-time-sets-the-blast-radius), and the damage a
regression does is traffic multiplied by that time.

The result is that the sample rate needed to catch a real regression quickly is far
lower than intuition suggests, and the reason teams do not have this instrument is
not that it is expensive.
"""
import math

TRAFFIC = 42000.0        # requests per day
BASE_ERR = 0.04          # semantic error rate before the regression
REVIEW_COST = 0.85       # cost of reviewing one sampled answer
ERROR_COST = 24.0        # cost of one wrong answer reaching a user
Z = 2.58                 # ~99% confidence, to avoid tripping on noise


def samples_needed(e0, e1):
    """Sampled answers required to distinguish e1 from e0 at Z confidence."""
    if e1 <= e0:
        return float("inf")
    ebar = (e0 + e1) / 2.0
    return (Z * Z * 2.0 * ebar * (1.0 - ebar)) / ((e1 - e0) ** 2)


def detect_hours(e1, rate):
    """Hours to detect a shift to e1, sampling `rate` of traffic."""
    n = samples_needed(BASE_ERR, e1)
    per_hour = TRAFFIC * rate / 24.0
    if per_hour <= 0:
        return float("inf")
    return n / per_hour


def damage(e1, hours):
    """Cost of the extra wrong answers served during the detection window."""
    return TRAFFIC * (hours / 24.0) * (e1 - BASE_ERR) * ERROR_COST


print("A service at %.0f requests/day with a %.0f%% semantic error rate."
      % (TRAFFIC, BASE_ERR * 100))
print("A regression raises that rate. How long until a sampled monitor notices?")
print()
SHIFTS = [0.06, 0.08, 0.12, 0.20, 0.35]
print(f"{'new error rate':>16}{'effect size':>13}{'samples needed':>17}"
      f"{'reviews/day at 1%':>20}")
print("-" * 66)
need = {}
for e1 in SHIFTS:
    n = samples_needed(BASE_ERR, e1)
    need[e1] = n
    print(f"{e1:>16.0%}{e1 - BASE_ERR:>13.0%}{n:>17.0f}"
          f"{TRAFFIC * 0.01:>20.0f}")

print()
print()
print("Detection time by sample rate. The column that matters is how long a")
print("regression runs before anything notices.")
print()
RATES = [0.001, 0.005, 0.02, 0.05, 0.15]
print(f"{'new error rate':>16}" + "".join(f"{r:>13.1%}" for r in RATES))
print("-" * 81)
grid = {}
for e1 in SHIFTS:
    row = [detect_hours(e1, r) for r in RATES]
    grid[e1] = row
    cells = "".join((f"{h:>12.1f}h" if h < 1000 else f"{'--':>13}") for h in row)
    print(f"{e1:>16.0%}{cells}")

print()
print()
print("What that detection window costs, in wrong answers reaching users.")
print()
print(f"{'new error rate':>16}" + "".join(f"{r:>13.1%}" for r in RATES))
print("-" * 81)
dmg = {}
for e1 in SHIFTS:
    row = [damage(e1, h) for h in grid[e1]]
    dmg[e1] = row
    cells = "".join((f"{d:>13.0f}" if d < 1e7 else f"{'--':>13}") for d in row)
    print(f"{e1:>16.0%}{cells}")

print()
print()
print("The trade. Sampling costs money every day; detection lag costs money only")
print("when a regression happens. Assume one regression a quarter.")
print()
REGRESSIONS_PER_YEAR = 4.0
TARGET = 0.12          # the regression size worth designing for
print(f"{'sample rate':>13}{'reviews/day':>14}{'review cost/yr':>17}"
      f"{'detect':>10}{'damage/yr':>13}{'total/yr':>12}")
print("-" * 79)
best = None
totals = {}
for r in RATES:
    reviews = TRAFFIC * r
    rc = reviews * REVIEW_COST * 365.0
    h = detect_hours(TARGET, r)
    d = damage(TARGET, h) * REGRESSIONS_PER_YEAR
    tot = rc + d
    totals[r] = (reviews, rc, h, d, tot)
    if best is None or tot < totals[best][4]:
        best = r
    print(f"{r:>13.1%}{reviews:>14.0f}{rc:>17.0f}{h:>9.1f}h{d:>13.0f}"
          f"{tot:>12.0f}")

print()
print(f"cheapest: {best:.1%} sampling, {totals[best][4]:.0f} per year total")

print()
print()
print("And the comparison that matters: what an availability-only breaker does.")
print()
print(f"{'breaker':>28}{'detects semantic':>19}{'detect time':>14}"
      f"{'damage/yr':>13}")
print("-" * 74)
print(f"{'availability / status code':>28}{'no':>19}{'never':>14}"
      f"{damage(TARGET, 24.0 * 90) * REGRESSIONS_PER_YEAR:>13.0f}")
for r in (0.005, 0.02):
    h = detect_hours(TARGET, r)
    print(f"{('sampled semantic at %.1f%%' % (r * 100)):>28}{'yes':>19}"
          f"{h:>13.1f}h{damage(TARGET, h) * REGRESSIONS_PER_YEAR:>13.0f}")

print(f"""
The samples-needed column is the first surprise. Detecting a shift from
{BASE_ERR:.0%} to {0.12:.0%} takes {need[0.12]:.0f} reviewed answers -- not
thousands, and not a share of traffic. It is an absolute count, and it is small.

That is because detection scales with the SQUARE of the effect size. A shift to
{0.06:.0%} -- an effect of two points -- needs {need[0.06]:.0f} samples. A shift to
{0.20:.0%} needs {need[0.2]:.0f}. **Big regressions, which are the ones that matter,
are cheap to detect** (eq:detection-time-sets-the-blast-radius), and the expensive
case is distinguishing small drifts that may not be worth tripping on anyway.

The detection grid turns that into wall-clock time. At {0.005:.1%} sampling -- one
answer in two hundred -- a shift to {0.12:.0%} is caught in
{grid[0.12][1]:.1f} hours. At {0.02:.0%} it is caught in {grid[0.12][2]:.1f} hours.

Those are hours, on a service doing {TRAFFIC:.0f} requests a day, for a review budget
of {TRAFFIC * 0.005:.0f} to {TRAFFIC * 0.02:.0f} answers a day.

The cost table prices the whole design. The cheapest configuration is
**{best:.1%} sampling** at {totals[best][4]:.0f} a year all-in -- {totals[best][1]:.0f}
in review cost and {totals[best][3]:.0f} in damage from the four regressions it
catches slightly late.

Sampling less is not cheaper. At {0.001:.1%} the review bill falls to
{totals[0.001][1]:.0f} but detection takes {totals[0.001][2]:.1f} hours and damage
rises to {totals[0.001][3]:.0f} -- a total of {totals[0.001][4]:.0f}, or
{totals[0.001][4] / totals[best][4]:.1f} times the optimum. **Under-sampling is a
false economy in the same shape as ch:sd-routing-caching's over-caching**: the saving
is visible on one line and the cost lands on another.

The last table is the one to take to a design review. An availability breaker never
detects this at all -- the regression returns 200 responses and the breaker has
nothing to trip on -- so the damage runs until a human notices, which the table prices
at a quarter's worth. A sampled semantic breaker at {best:.1%} catches it in
{detect_hours(TARGET, best):.1f} hours.

The ratio between those two damage figures is roughly
{damage(TARGET, 24.0 * 90) / damage(TARGET, detect_hours(TARGET, best)):.0f}
to one, and the instrument that closes it costs
{TRAFFIC * best * REVIEW_COST:.0f} a day.

So the conclusion is narrower and more useful than "you need better observability".
**The second instrument ch:sd-architecture said every model-backed system needs is
affordable at a sampling rate of {best:.1%} -- {TRAFFIC * best:.0f} reviewed answers
a day -- and it is the only thing in the stack that can drive a circuit breaker.**
The reason most systems lack it is not cost. It is that nobody has computed these two
columns and put them next to each other.

One caveat on the mechanism. A breaker driven by sampled review trips on a statistic,
so it inherits every property of the statistic -- including that it will occasionally
trip on noise, and that the Z of {Z:.2f} used here is what keeps that rate low at the
cost of the detection times in the grid. A breaker that trips too readily on a
{TRAFFIC:.0f}-request-a-day service is worse than no breaker, because the response to
a semantic trip is usually to fall back to a degraded mode, and degraded modes have
their own error rates.""")

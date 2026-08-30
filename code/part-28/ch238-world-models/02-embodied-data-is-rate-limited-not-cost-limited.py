# -*- coding: utf-8 -*-
# Extracted from: Chapter 238 — World Models and Embodied AI
# Source: src/.../ch238-world-models.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Embodied data is rate-limited, not cost-limited, and the transfer coefficient sets the fleet.

Everything else in this book buys data with money. Physical interaction is bought with
wall-clock: a robot performs a bounded number of attempts per hour, and no budget changes that
number (eq:embodied-data-is-rate-limited-not-cost-limited).

Simulation and video escape the rate limit and pay a transfer discount instead -- a simulated
trajectory is worth some fraction of a real one, and that fraction has a ceiling because the
residual gap is systematic rather than random.

Which makes the transfer coefficient the parameter that decides how large a robot fleet has to
be (eq:transfer-discount-sets-the-sim-real-mixture).
"""
import math

TARGET = 40_000_000        # real-equivalent trajectories for the capability
HOURS_PER_DAY = 10.0

# (source, trajectories per hour per unit, $ per trajectory, transfer coefficient,
#  share of the requirement it can cover before the residual gap binds)
SOURCES = [
    ("a real robot",             14.0,   9.40,  1.00, 1.00),
    ("teleoperation",            22.0,  31.00,  1.00, 1.00),
    ("simulation",            4_200.0,   0.011, 0.24, 0.55),
    ("internet video",       90_000.0,   0.0004, 0.06, 0.18),
    ("human demonstration video", 30.0,  4.20,  0.11, 0.09),
]

print("What each source of experience actually supplies.")
print()
print(f"{'source':>28}{'traj / unit-hour':>19}{'$ / trajectory':>17}"
      f"{'transfer':>11}{'real-equiv / $':>17}{'coverage cap':>15}")
print("-" * 107)
for name, rate, cost, tau, cap in SOURCES:
    print(f"{name:>28}{rate:>19,.0f}{cost:>17.4f}{tau:>11.2f}"
          f"{tau / cost:>17,.1f}{cap:>15.0%}")

print()
print(f"a simulated trajectory is worth {0.24:.2f} of a real one and costs"
      f" {9.40 / 0.011:,.0f}x less")
print(f"which is {(0.24 / 0.011) / (1.00 / 9.40):,.0f}x the real-equivalent per dollar")

print()
print()
print("So buy real-equivalent experience in order of what it costs.")
print()
print(f"target: {TARGET:,} real-equivalent trajectories")
print()
print(f"{'source':>28}{'real-equiv / $':>17}{'coverage taken':>17}"
      f"{'real-equiv supplied':>22}{'trajectories':>17}{'cost':>16}")
print("-" * 117)
remaining = 1.0
plan, spend = {}, 0.0
for name, rate, cost, tau, cap in sorted(SOURCES, key=lambda s: -s[3] / s[2]):
    take = min(cap, remaining)
    remaining -= take
    re = TARGET * take
    traj = re / tau
    c = traj * cost
    spend += c
    plan[name] = (take, re, traj, c)
    print(f"{name:>28}{tau / cost:>17,.2f}{take:>17.0%}{re:>22,.0f}"
          f"{traj:>17,.0f}{c:>16,.0f}")
print("-" * 117)
print(f"{'TOTAL':>28}{'':>17}{1.0 - remaining:>17.0%}{TARGET * (1.0 - remaining):>22,.0f}"
      f"{'':>17}{spend:>16,.0f}")

REAL_SHARE = plan["a real robot"][0]
REAL_TRAJ = plan["a real robot"][2]
real_cost = plan["a real robot"][3]
print()
print(f"{REAL_SHARE:.0%} of the requirement -- {REAL_TRAJ:,.0f} trajectories -- has to be real")
print(f"and that share is {real_cost / spend:.0%} of the money")
print(f"`human demonstration video` supplies {plan['human demonstration video'][0]:.0%}:"
      f" at {0.11 / 4.20:.3f} real-equivalent per dollar it is dominated")

print()
print()
print("But money is not what those trajectories cost. Wall-clock is.")
print()
PER_ROBOT_DAY = 14.0 * HOURS_PER_DAY
print(f"{'fleet size':>13}{'trajectories / day':>21}{'days to the residual':>23}"
      f"{'months':>10}{'hardware $':>15}")
print("-" * 82)
ROBOT_COST = 78_000.0
fleet_months = {}
for robots in (10, 50, 200, 1_000, 5_000, 20_000):
    per_day = robots * PER_ROBOT_DAY
    days = REAL_TRAJ / per_day
    fleet_months[robots] = days / 30.4
    print(f"{robots:>13,}{per_day:>21,.0f}{days:>23,.0f}"
          f"{days / 30.4:>10,.1f}{robots * ROBOT_COST:>15,.0f}")

print()
print(f"{50:,} robots need {fleet_months[50]:,.0f} months;"
      f" {5_000:,} need {fleet_months[5000]:,.1f}")
print("(eq:embodied-data-is-rate-limited-not-cost-limited)")

print()
print()
print("And the transfer coefficient decides how big that residual is.")
print()
print(f"{'simulation transfer':>21}{'sim coverage cap':>19}{'residual real share':>22}"
      f"{'real trajectories':>20}{'months at 1,000 robots':>25}")
print("-" * 107)
tau_rows = {}
for tau in (0.05, 0.12, 0.24, 0.40, 0.60, 0.80):
    cap = min(0.68, 0.55 * (tau / 0.24) ** 0.55)
    residual = max(0.05, 1.0 - cap - 0.18)
    traj = TARGET * residual
    months = traj / (1_000 * PER_ROBOT_DAY) / 30.4
    tau_rows[tau] = (cap, residual, traj, months)
    print(f"{tau:>21.2f}{cap:>19.0%}{residual:>22.0%}{traj:>20,.0f}"
          f"{months:>25,.1f}")

print()
print(f"transfer {0.12:.2f} needs {tau_rows[0.12][3]:,.0f} months at 1,000 robots;")
print(f"transfer {0.60:.2f} needs {tau_rows[0.60][3]:,.1f}")
print(f"a factor of {tau_rows[0.12][3] / max(tau_rows[0.60][3], 1e-9):,.1f}"
      f" from one coefficient")
print("(eq:transfer-discount-sets-the-sim-real-mixture)")

print()
print()
print("What actually moves the transfer coefficient.")
print()
GAP = [
    ("better renderer",              0.03, 1.9,  "graphics work"),
    ("domain randomisation",         0.09, 1.2,  "more sim compute"),
    ("system identification",        0.11, 1.4,  "real measurements"),
    ("fine-tune on real data",       0.14, 3.1,  "real trajectories"),
    ("contact and friction modelling", 0.16, 2.6, "physics engineering"),
    ("all of them",                  0.38, 10.2, "everything above"),
]
print(f"{'intervention':>34}{'transfer gain':>16}{'cost multiple':>16}"
      f"{'gain per unit cost':>22}{'what it needs':>22}")
print("-" * 110)
for name, gain, cost, needs in GAP:
    print(f"{name:>34}{gain:>16.2f}{cost:>15.1f}x{gain / cost:>22.3f}{needs:>22}")

best_gap = max(GAP[:-1], key=lambda g: g[1] / g[2])
print()
print(f"best gain per unit cost: {best_gap[0]} at {best_gap[1] / best_gap[2]:.3f}")
print(f"note that `fine-tune on real data` needs the thing that is rate-limited")

print()
print()
print("Three strategies, end to end.")
print()
STRATS = [
    ("all real, 1,000 robots",   0.00, 1_000),
    ("sim-heavy, 200 robots",    0.24,   200),
    ("sim-heavy, 1,000 robots",  0.24, 1_000),
    ("gap closed, 200 robots",   0.62,   200),
]
print(f"{'strategy':>28}{'sim transfer':>15}{'robots':>10}"
      f"{'real trajectories':>20}{'months':>10}{'total $':>16}")
print("-" * 99)
for name, tau, robots in STRATS:
    if tau == 0.0:
        residual, sim_cost = 1.0, 0.0
    else:
        cap = min(0.68, 0.55 * (tau / 0.24) ** 0.55)
        residual = max(0.05, 1.0 - cap - 0.18)
        sim_cost = TARGET * cap / tau * 0.011 + TARGET * 0.18 / 0.06 * 0.0004
    traj = TARGET * residual
    months = traj / (robots * PER_ROBOT_DAY) / 30.4
    total = traj * 9.40 + sim_cost + robots * ROBOT_COST
    print(f"{name:>28}{tau:>15.2f}{robots:>10,}{traj:>20,.0f}"
          f"{months:>10,.1f}{total:>16,.0f}")

print(f"""
The source table is the fact that makes embodied learning a different subject. A simulated
trajectory is worth {0.24:.2f} of a real one and costs {9.40 / 0.011:,.0f} times less, which is
{(0.24 / 0.011) / (1.00 / 9.40):,.0f} times the real-equivalent per dollar. On cost alone the
answer is obvious and it is also wrong, because of the last column.

Every non-real source has a **coverage cap**: a share of the requirement beyond which its
residual gap is systematic and more of it adds nothing. Simulation caps at {0.55:.0%}, internet
video at {0.18:.0%}. Those caps exist because the mismatch between a simulator and the world is
not noise -- it is a set of specific unmodelled effects, and drawing more samples from the same
simulator reproduces them exactly.

The plan table buys in order of real-equivalent per dollar and reports what is left.
**{REAL_SHARE:.0%} of the requirement -- {REAL_TRAJ:,.0f} trajectories -- has to be real**, and
that residual is {real_cost / spend:.0%} of the money.

`human demonstration video` is worth a note. It feels like cheap real-world data and it is the
most expensive source in the table per real-equivalent trajectory --
{0.11 / 4.20:.3f} against a real robot's {1.00 / 9.40:.3f} -- because a human demonstration
answers a different question than a robot attempt does, and the transfer coefficient charges for
the difference.

The fleet table is why the residual matters more than the money
(eq:embodied-data-is-rate-limited-not-cost-limited). A robot produces
{PER_ROBOT_DAY:,.0f} trajectories a day and no budget changes that. Fifty robots need
**{fleet_months[50]:,.0f} months**. A thousand need {fleet_months[1000]:,.1f}. Five thousand need
{fleet_months[5000]:,.1f} -- and cost {5_000 * ROBOT_COST:,.0f} in hardware before a single
trajectory is collected.

**This is the only chapter in this book where the binding constraint is wall-clock**, and it
changes what an optimisation looks like. Every other domain's answer to "we need more data" is a
budget line. Here it is a fleet, a building, and a year.

The transfer table is where the leverage is
(eq:transfer-discount-sets-the-sim-real-mixture). At transfer {0.12:.2f} the residual is
{tau_rows[0.12][1]:.0%} and a thousand robots need {tau_rows[0.12][3]:,.0f} months. At
{0.60:.2f} the residual is {tau_rows[0.60][1]:.0%} and the same fleet needs
{tau_rows[0.60][3]:,.1f} -- **a factor of
{tau_rows[0.12][3] / max(tau_rows[0.60][3], 1e-9):,.0f} from one coefficient.**

So the highest-leverage work in embodied learning is not the policy and not the world model. It
is whatever raises the fraction of simulated experience that transfers, because that fraction
divides the fleet size.

The gap table says what does, and the winner is instructive. `{best_gap[0]}` gives
{best_gap[1]:.2f} of transfer for {best_gap[2]:.1f}x the cost -- {best_gap[1] / best_gap[2]:.3f}
per unit, the best row in the table -- and what it needs is *measurements* of the real system
rather than trajectories from it. Measuring a robot's friction and inertia is cheap in exactly
the currency that is scarce here.

Now the row that is not. `fine-tune on real data` is the second-largest single gain at
{0.14:.2f} and one of the worst per unit cost, at {0.14 / 3.1:.3f} -- and it consumes exactly the
resource the whole exercise is short of. **The best-known way to close the reality gap requires
the thing the reality gap is preventing you from getting**, which is the circularity at the
centre of this field.

The strategy table puts it together. `all real, 1,000 robots` needs {TARGET:,} real trajectories
and {TARGET / (1_000 * PER_ROBOT_DAY) / 30.4:,.1f} months at {1_000 * ROBOT_COST + TARGET * 9.40:,.0f}.
`sim-heavy, 200 robots` cuts the real requirement to {REAL_SHARE:.0%} of it and finishes in
{TARGET * REAL_SHARE / (200 * PER_ROBOT_DAY) / 30.4:,.1f} months.

The row to read twice is `gap closed, 200 robots`: the same small fleet with transfer raised to
{0.62:.2f} finishes in {TARGET * 0.14 / (200 * PER_ROBOT_DAY) / 30.4:,.1f} months -- **half the time of the
same fleet without the gap work, and the cheapest strategy on the page.**

The one thing that beats it on time is `sim-heavy, 1,000 robots` at
{TARGET * REAL_SHARE / (1_000 * PER_ROBOT_DAY) / 30.4:,.1f} months, and it costs
{(TARGET * REAL_SHARE * 9.40 + 1_000 * ROBOT_COST) / (TARGET * 0.14 * 9.40 + 200 * ROBOT_COST):.1f}
times as much to get there.

**A better simulator substitutes for a bigger fleet, at a fraction of the cost**, and it does so
through a single coefficient that most teams never measure. That is an unusual conclusion for a
data-hungry field, and it is the practical content of this listing.""")

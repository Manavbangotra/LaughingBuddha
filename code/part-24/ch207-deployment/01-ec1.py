# -*- coding: utf-8 -*-
# Extracted from: Chapter 207 — Deployment Strategies and Rollback
# Source: src/.../ch207-deployment.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A canary sized for availability is far too small for a semantic signal.

A canary trades exposure against detection: send a small share of traffic to the new
version, watch for trouble, and roll back before it reaches everyone. The share is
usually chosen from habit -- one percent, five percent -- and that habit was formed on
availability signals, which move in seconds.

Semantic signals do not. ch:sd-fault-tolerance showed detection time scales with the
inverse square of the effect size, and a canary sees only its share of traffic. So the
detection time at a given canary share is the full-traffic detection time divided by that
share (eq:canary-share-divides-the-sample-rate).

This listing finds the canary size that minimises total damage, and finds it is much
larger than the habitual one.
"""
import math

TRAFFIC_PER_HOUR = 4200.0
BASE_ERR = 0.04
Z = 2.58
ERROR_COST = 24.0
SAMPLE_RATE = 0.005          # ch:sd-fault-tolerance's optimal review sampling
SHARES = [0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50]


def samples_needed(e0, e1):
    ebar = (e0 + e1) / 2.0
    return (Z * Z * 2.0 * ebar * (1.0 - ebar)) / ((e1 - e0) ** 2)


def detect_hours(e1, share):
    """Hours to detect, seeing only `share` of traffic at SAMPLE_RATE review."""
    n = samples_needed(BASE_ERR, e1)
    reviewed_per_hour = TRAFFIC_PER_HOUR * share * SAMPLE_RATE
    return n / reviewed_per_hour if reviewed_per_hour > 0 else float("inf")


def damage(e1, share):
    """Bad answers served during detection, plus the rollback tail."""
    h = detect_hours(e1, share)
    exposed = TRAFFIC_PER_HOUR * share * h
    return exposed * (e1 - BASE_ERR) * ERROR_COST


print("A service at %.0f requests/hour, %.0f%% baseline semantic error rate,"
      % (TRAFFIC_PER_HOUR, BASE_ERR * 100))
print("reviewing %.1f%% of answers. A bad deploy raises the error rate."
      % (SAMPLE_RATE * 100))
print()
print("Detection time by canary share, for a deploy that doubles the error rate.")
print()
E1 = 0.08
print(f"{'canary share':>14}{'reqs/hr in canary':>20}{'reviewed/hr':>14}"
      f"{'detect hrs':>13}{'exposed':>11}")
print("-" * 74)
tab = {}
for sh in SHARES:
    h = detect_hours(E1, sh)
    exposed = TRAFFIC_PER_HOUR * sh * h
    tab[sh] = (h, exposed, damage(E1, sh))
    print(f"{sh:>14.0%}{TRAFFIC_PER_HOUR * sh:>20.0f}"
          f"{TRAFFIC_PER_HOUR * sh * SAMPLE_RATE:>14.1f}{h:>13.1f}"
          f"{exposed:>11.0f}")

print()
print("Note the last column: exposure is share times time, and the two cancel.")

print()
print()
print("Which is the point. Damage during detection is INDEPENDENT of canary size,")
print("because a smaller canary detects proportionally more slowly.")
print()
print(f"{'canary share':>14}{'detect hrs':>13}{'bad answers':>14}"
      f"{'damage':>11}{'vs 1%':>9}")
print("-" * 62)
for sh in SHARES:
    h, exposed, d = tab[sh]
    print(f"{sh:>14.0%}{h:>13.1f}{exposed * (E1 - BASE_ERR):>14.0f}"
          f"{d:>11.0f}{d / tab[0.01][2]:>8.2f}x")

print()
print()
print("What DOES change with canary size: how long the rest of the fleet waits,")
print("and what happens if the deploy is fine.")
print()
ROLLOUT_VALUE_PER_HOUR = 180.0     # value of the improvement, if it is good
P_BAD = 0.14                        # share of deploys that are bad
# Having exposed a share of customers at all costs something independent of
# duration: notification, support load, and the trust that does not come back.
BLAST_COST = 400000.0
print(f"{'canary share':>14}{'detect hrs':>13}{'delay cost':>13}"
      f"{'damage':>11}{'blast':>11}{'expected':>11}")
print("-" * 72)
tot = {}
for sh in SHARES:
    h, exposed, d = tab[sh]
    delay_cost = ROLLOUT_VALUE_PER_HOUR * h * (1.0 - sh)
    blast = BLAST_COST * sh
    exp = (1 - P_BAD) * delay_cost + P_BAD * (d + blast)
    tot[sh] = (h, delay_cost, d, exp, blast)
    print(f"{sh:>14.0%}{h:>13.1f}{delay_cost:>13.0f}"
          f"{d:>11.0f}{blast:>11.0f}{exp:>11.0f}")

best = min(tot, key=lambda k: tot[k][3])
print()
print(f"cheapest canary share: {best:.0%} at expected cost {tot[best][3]:.0f}")

print()
print()
print("How the optimum moves with effect size. A subtle regression needs a")
print("bigger canary to be seen at all.")
print()
print(f"{'new error rate':>16}{'effect':>9}" +
      "".join(f"{('%.0f%%' % (s * 100)):>10}" for s in SHARES) + f"{'best':>8}")
print("-" * 92)
bysize = {}
for e1 in (0.05, 0.06, 0.08, 0.12, 0.20):
    row = {}
    cells = ""
    for sh in SHARES:
        h = detect_hours(e1, sh)
        d = damage(e1, sh)
        delay = ROLLOUT_VALUE_PER_HOUR * h * (1.0 - sh)
        row[sh] = (1 - P_BAD) * delay + P_BAD * (d + BLAST_COST * sh)
        cells += f"{row[sh]:>10.0f}"
    b = min(row, key=lambda k: row[k])
    bysize[e1] = b
    print(f"{e1:>16.0%}{e1 - BASE_ERR:>9.0%}{cells}{b:>7.0%}")

print()
print()
print("And the comparison with an availability signal, which is what the habit")
print("of a 1% canary was formed on.")
print()
AVAIL_SAMPLES = 40.0        # a 500 error is unambiguous; a few dozen suffice
print(f"{'signal':>22}{'samples needed':>17}{'reviewed share':>17}"
      f"{'detect at 1% canary':>22}")
print("-" * 80)
for label, n, rate in (("availability (500s)", AVAIL_SAMPLES, 1.0),
                       ("semantic, 2x error", samples_needed(BASE_ERR, 0.08),
                        SAMPLE_RATE),
                       ("semantic, 1.5x error", samples_needed(BASE_ERR, 0.06),
                        SAMPLE_RATE)):
    per_hour = TRAFFIC_PER_HOUR * 0.01 * rate
    print(f"{label:>22}{n:>17.0f}{rate:>17.1%}{n / per_hour:>21.1f}h")

print(f"""
The first table contains the result that reframes canary sizing, and it is easy to miss
because it looks like a coincidence.

At a {0.01:.0%} canary, detecting a doubled error rate takes {tab[0.01][0]:.1f} hours and
exposes {tab[0.01][1]:.0f} requests. At {0.50:.0%}, it takes {tab[0.5][0]:.1f} hours and
exposes {tab[0.5][1]:.0f} requests.

**The exposure is identical**, and it is identical at every share in between
(eq:canary-share-divides-the-sample-rate). A smaller canary exposes fewer requests per
hour and takes proportionally longer, and the two cancel exactly.

That is worth stating plainly because it demolishes the usual argument. A small canary is
chosen to limit blast radius. **It does not limit blast radius** -- it limits the rate at
which the blast radius accumulates, and stretches the accumulation over a proportionally
longer window. The integral is the same.

So if damage during detection does not depend on canary size, what does? Two things, and
they point in opposite directions.

The first is the delay to everyone else. While the canary runs, the other
{1 - 0.01:.0%} of traffic is on the old version, not getting whatever improvement the
deploy contained. At a {0.01:.0%} canary that delay is {tab[0.01][0]:.1f} hours; at
{0.50:.0%} it is {tab[0.5][0]:.1f}.

The second is that a larger canary exposes more distinct customers, and that carries a
cost independent of how long it lasts -- notification, support load, and trust that does
not come back. That term rises with share while the delay term falls, which is what
produces an interior optimum rather than a corner.

The expected-cost table gives the answer: **{best:.0%}**, at
{tot[best][3]:.0f} against {tot[0.01][3]:.0f} for a {0.01:.0%} canary --
{tot[0.01][3] / tot[best][3]:.1f} times cheaper.

The effect-size table shows the optimum is not a constant. For a subtle regression --
{0.05:.0%} against a {BASE_ERR:.0%} baseline -- the best share is
{bysize[0.05]:.0%}; for an obvious one it is {bysize[0.2]:.0%}.

**Subtle regressions need larger canaries**, which is the opposite of the instinct that
says a risky change deserves a small one. The instinct is right about *availability*
risk and wrong about semantic risk, and the difference is in the last table.

An availability regression needs about {AVAIL_SAMPLES:.0f} samples to establish -- a few
dozen 500s and nobody is in any doubt -- and every request is a sample. So at a
{0.01:.0%} canary it is detected in
{AVAIL_SAMPLES / (TRAFFIC_PER_HOUR * 0.01):.1f} hours.

A semantic regression of the same practical severity needs
{samples_needed(BASE_ERR, 0.08):.0f} *reviewed* answers, and only
{SAMPLE_RATE:.1%} of answers are reviewed. At a {0.01:.0%} canary that is
{samples_needed(BASE_ERR, 0.08) / (TRAFFIC_PER_HOUR * 0.01 * SAMPLE_RATE):.0f} hours.

**Two orders of magnitude between them**, and the one-percent canary is a convention
inherited from the fast case. Applied to the slow case it produces a canary that runs for
days, delays every good deploy for days, and does not reduce the damage from the bad ones
at all.""")

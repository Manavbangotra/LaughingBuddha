# -*- coding: utf-8 -*-
# Extracted from: Chapter 211 — Cost, Latency, and Governance
# Source: src/.../ch211-governance.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A budget enforced on billing data is a budget enforced yesterday.

Every cost control is a feedback loop, and every feedback loop has a delay. For AI spend
the delay is unusually long, because the cheapest place to read cost -- the provider's
dashboard or the cloud billing export -- is also the slowest.

So the overrun on a runaway is not set by the limit. It is set by how long the loop takes
to close (eq:budget-overrun-is-set-by-feedback-delay), and a limit read from a daily
export cannot bound a spend that moves hourly.

The second half is the part teams discover later: detecting the excess does not tell you
what to throttle, and the collateral damage of the throttle is set by how finely the
spend is attributed (eq:attribution-precedes-control).
"""
MONTHLY_BUDGET = 180_000.0
NORMAL_BURN_DAY = 5_800.0
INCIDENT_MULTIPLE = 6.4        # a retry storm, a loop, a prompt that tripled context
REQUESTS_DAY = 42_000.0
RUNAWAY_SHARE = 0.035          # share of traffic actually responsible

excess_day = NORMAL_BURN_DAY * (INCIDENT_MULTIPLE - 1.0)
excess_hour = excess_day / 24.0

print(f"Normal burn {NORMAL_BURN_DAY:,.0f} a day against a "
      f"{MONTHLY_BUDGET:,.0f} monthly budget.")
print(f"An incident takes burn to {INCIDENT_MULTIPLE:.1f}x normal: "
      f"{excess_day:,.0f} a day of excess,")
print(f"{excess_hour:,.0f} an hour.")
print()
print("What each detection channel costs, in overrun, before anyone can act.")
print()
CHANNELS = [
    ("cloud billing export",        26.0, 3.0),
    ("provider dashboard",           4.0, 2.0),
    ("nightly usage rollup",        14.0, 3.0),
    ("self-metered token counter",   0.1, 1.0),
    ("in-request accounting",        0.0, 0.5),
]
print(f"{'detection channel':>28}{'lag (h)':>10}{'reaction (h)':>15}"
      f"{'total (h)':>12}{'overrun':>12}{'% of budget':>14}")
print("-" * 91)
chan = {}
for name, lag, react in CHANNELS:
    total = lag + react
    over = excess_hour * total
    chan[name] = (total, over)
    print(f"{name:>28}{lag:>10.1f}{react:>15.1f}{total:>12.1f}"
          f"{over:>12,.0f}{over / MONTHLY_BUDGET:>14.1%}")

print()
print()
print("Detection is only half of it. To stop the excess you have to throttle")
print("something, and what you can throttle is what you can attribute.")
print()
ATTRIB = [
    ("nothing -- global throttle",        1.000, 0.0),
    ("by service",                        0.340, 0.5),
    ("by team",                           0.190, 1.0),
    ("by feature",                        0.080, 2.0),
    ("by feature and customer tier",      0.042, 3.5),
]
print(f"{'attribution granularity':>32}{'traffic throttled':>20}"
      f"{'collateral req/day':>21}{'effort':>9}")
print("-" * 82)
att = {}
for name, share, eff in ATTRIB:
    collateral = REQUESTS_DAY * (share - RUNAWAY_SHARE)
    att[name] = (share, collateral, eff)
    print(f"{name:>32}{share:>20.1%}{collateral:>21,.0f}{eff:>9.1f}")

print()
print("The runaway is %.1f%% of traffic. Everything above that is healthy traffic"
      % (RUNAWAY_SHARE * 100))
print("stopped because nothing could tell it apart.")

print()
print()
print("Pricing both halves together. Collateral is charged at the value of a")
print("blocked request; overrun at face value.")
print()
BLOCK_COST = 0.62              # revenue and goodwill lost per healthy request blocked
HOURS_TO_FIX = 6.0             # how long the throttle stays on
print(f"{'detection':>28}{'attribution':>32}{'overrun':>11}"
      f"{'collateral':>13}{'total':>11}")
print("-" * 95)
grid = {}
for cname, lag, react in CHANNELS:
    for aname, share, eff in ATTRIB:
        over = chan[cname][1]
        coll = att[aname][1] * (HOURS_TO_FIX / 24.0) * BLOCK_COST
        grid[(cname, aname)] = over + coll
for cname in ("cloud billing export", "provider dashboard",
              "self-metered token counter", "in-request accounting"):
    for aname in ("nothing -- global throttle", "by team",
                  "by feature and customer tier"):
        over = chan[cname][1]
        coll = att[aname][1] * (HOURS_TO_FIX / 24.0) * BLOCK_COST
        print(f"{cname:>28}{aname:>32}{over:>11,.0f}"
              f"{coll:>13,.0f}{over + coll:>11,.0f}")

best = min(grid, key=lambda k: grid[k])
worst = max(grid, key=lambda k: grid[k])
print()
print(f"worst pairing: {worst[0]} + {worst[1]} at {grid[worst]:,.0f}")
print(f"best pairing:  {best[0]} + {best[1]} at {grid[best]:,.0f}")
print(f"ratio: {grid[worst] / grid[best]:.0f}x")

print()
print()
print("Which of the two is worth fixing first, at a realistic incident rate.")
print()
INCIDENTS_YEAR = 7.0
BASE = ("cloud billing export", "nothing -- global throttle")
print(f"{'change':>44}{'incident cost':>16}{'annual':>12}{'saved/yr':>12}")
print("-" * 84)
base_cost = grid[BASE]
MOVES = [
    ("baseline: billing export, global throttle", BASE),
    ("fix detection only (token counter)",
     ("self-metered token counter", "nothing -- global throttle")),
    ("fix attribution only (feature + tier)",
     ("cloud billing export", "by feature and customer tier")),
    ("fix both", ("self-metered token counter", "by feature and customer tier")),
    ("in-request accounting, feature + tier",
     ("in-request accounting", "by feature and customer tier")),
]
mv = {}
for label, key in MOVES:
    c = grid[key]
    mv[label] = c
    print(f"{label:>44}{c:>16,.0f}{c * INCIDENTS_YEAR:>12,.0f}"
          f"{(base_cost - c) * INCIDENTS_YEAR:>12,.0f}")

print()
print()
print("And the governance question underneath: what a limit can actually promise.")
print()
print(f"{'control':>34}{'bounds':>26}{'guarantee':>26}")
print("-" * 86)
CONTROLS = [
    ("monthly budget alert",     "nothing",              "you find out"),
    ("daily spend cap",          "one day of excess",    f"{excess_day:,.0f}"),
    ("hourly spend cap",         "one hour of excess",   f"{excess_hour:,.0f}"),
    ("per-request cost cap",     "one request",          "the unit price"),
    ("pre-flight reservation",   "the request, before",  "ch:sd-apis-auth"),
]
for name, bounds, guar in CONTROLS:
    print(f"{name:>34}{bounds:>26}{guar:>26}")

print(f"""
The channel table is the arithmetic that decides everything else. Excess burn of
{excess_hour:,.0f} an hour costs {chan['cloud billing export'][1]:,.0f} before a billing
export even shows it -- **{chan['cloud billing export'][1] / MONTHLY_BUDGET:.0%} of the
monthly budget spent inside one incident**, on a control that was described as a budget
(eq:budget-overrun-is-set-by-feedback-delay).

A self-metered token counter closes the same loop in
{chan['self-metered token counter'][0]:.1f} hours for
{chan['self-metered token counter'][1]:,.0f}, which is
{chan['cloud billing export'][1] / chan['self-metered token counter'][1]:.0f} times less.

The thing to notice is what the fix costs. The token counter is arithmetic on data the
request handler already has -- it counted the tokens to build the prompt. It is not an
observability platform, it is a running total, and it closes the loop in
{chan['self-metered token counter'][0]:.1f} hours against
{chan['cloud billing export'][0]:.0f} for the channel most teams rely on.

The attribution table is the half nobody plans for. Detecting the excess in five minutes
does not tell you *what* to throttle. With no attribution the only available control is a
global throttle, which stops {REQUESTS_DAY:,.0f} requests a day to remove the
{RUNAWAY_SHARE:.1%} that are the problem --
{att['nothing -- global throttle'][1]:,.0f} healthy requests blocked
(eq:attribution-precedes-control).

Attribution by feature and customer tier cuts that to
{att['by feature and customer tier'][1]:,.0f}, for {3.5:.1f} units of effort spent
tagging requests.

**A cost signal you cannot attribute is not a control, it is a notification**, and the
distinction is invisible until the first incident, because until then the two look
identical on a dashboard.

The grid prices the pairing. The worst combination costs {grid[worst]:,.0f} an incident
and the best {grid[best]:,.0f} --
{grid[worst] / grid[best]:.0f} times less, from two changes that are both a
week of work.

The ranking table answers which to do first and the answer is detection, decisively.
Fixing detection alone takes an incident from {base_cost:,.0f} to
{mv['fix detection only (token counter)']:,.0f}. Fixing attribution alone reaches only
{mv['fix attribution only (feature + tier)']:,.0f}, because a precise throttle applied
{chan['cloud billing export'][0]:.0f} hours late is still
{chan['cloud billing export'][0]:.0f} hours late.

But look at what fixing detection does to the composition. Of the
{mv['fix detection only (token counter)']:,.0f} remaining, overrun is
{chan['self-metered token counter'][1]:,.0f} and collateral is
{mv['fix detection only (token counter)'] - chan['self-metered token counter'][1]:,.0f}
-- **{(mv['fix detection only (token counter)'] - chan['self-metered token counter'][1]) / mv['fix detection only (token counter)']:.0%}
of the cost is now the throttle rather than the spend.**

That is the ordering result and it is not obvious in advance. The two terms are not
independent priorities to be traded off; they are sequential, because
**closing the detection loop promotes attribution to the binding constraint**
(eq:attribution-precedes-control). A team that fixes only detection has bought a faster
way to block all of its traffic, and will conclude from the next incident that its cost
controls are too blunt -- which is correct, and is the second half of the same project.

The last table is the governance point stated plainly. A monthly budget alert bounds
nothing; it tells you afterwards. A daily cap bounds a day of excess, which is
{excess_day:,.0f}. An hourly cap bounds {excess_hour:,.0f}. Only a per-request cost cap
bounds a unit small enough to be uninteresting, and only ch:sd-apis-auth's pre-flight
reservation bounds it *before* the money is spent.

Which is why that chapter's result -- eq:cost-limits-need-a-reservation -- is a
governance result and not merely an API design one. Every control above it is a control
over a quantity that has already been consumed.""")

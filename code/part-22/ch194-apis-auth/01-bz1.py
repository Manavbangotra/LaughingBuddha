# -*- coding: utf-8 -*-
# Extracted from: Chapter 194 — APIs, Authentication, Authorization, and Rate Limiting
# Source: src/.../ch194-apis-auth.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A request-count rate limit cannot bound cost when requests are not equivalent.

Rate limiting assumes a request is a unit of work. ch:sd-architecture found that
assumption broken -- one request can cost forty times another -- and a limiter built
on the broken assumption fails in a specific way.

To bound worst-case spend with a request limit you must set it for the most expensive
request, which starves everyone sending cheap ones. To be fair to typical users you
must set it for the average, which admits a runaway
(eq:count-limits-cannot-bound-cost).

This listing measures both horns and prices the alternative.
"""
# A user population. (label, share of users, requests/min they want,
#                     mean cost per request, cost spread)
USERS = [
    ("light interactive",   0.62,  4.0,   1.0,  0.4),
    ("heavy interactive",   0.24, 11.0,   2.8,  1.6),
    ("document workload",   0.11,  6.0,  14.0, 11.0),
    ("agent workload",      0.03, 25.0,  38.0, 34.0),
]
COST_CEILING = 260.0    # cost units per minute the system can afford per tenant


def cost_wanted(u):
    return u[2] * u[3]


def peak_cost(u):
    """What this user costs when their requests land at the expensive end."""
    return u[2] * (u[3] + 2.0 * u[4])


print("Four user profiles. The cost of a request varies within a profile as well")
print("as between them.")
print()
print(f"{'profile':>20}{'share':>8}{'req/min':>10}{'cost/req':>11}"
      f"{'spread':>9}{'cost/min':>11}{'peak':>9}")
print("-" * 78)
for u in USERS:
    print(f"{u[0]:>20}{u[1]:>8.0%}{u[2]:>10.1f}{u[3]:>11.1f}{u[4]:>9.1f}"
          f"{cost_wanted(u):>11.1f}{peak_cost(u):>9.1f}")

worst_unit = max(u[3] + 2.0 * u[4] for u in USERS)
print()
print(f"most expensive single request: {worst_unit:.1f} cost units")
print(f"cheapest typical request:      {USERS[0][3]:.1f} cost units")
print(f"heterogeneity:                 {worst_unit / USERS[0][3]:.0f}x")

print()
print()
print("A request-count limit set to bound worst-case cost at %.0f per minute."
      % COST_CEILING)
print("The limit must assume every request is the most expensive kind.")
print()
SAFE_RPM = COST_CEILING / worst_unit
print(f"limit = {COST_CEILING:.0f} / {worst_unit:.1f} = {SAFE_RPM:.1f} requests/min")
print()
print(f"{'profile':>20}{'wants req/min':>16}{'allowed':>10}{'served':>10}"
      f"{'throttled':>12}")
print("-" * 68)
starve = {}
for u in USERS:
    served = min(u[2], SAFE_RPM)
    thr = 1.0 - served / u[2]
    starve[u[0]] = thr
    print(f"{u[0]:>20}{u[2]:>16.1f}{SAFE_RPM:>10.1f}{served:>10.1f}"
          f"{thr:>12.0%}")

weighted_starve = sum(u[1] * starve[u[0]] for u in USERS)
print()
print(f"population-weighted throttling: {weighted_starve:.0%} of requested traffic")

print()
print()
print("Now a request-count limit set fairly -- at what a typical user needs.")
print("Cost is no longer bounded.")
print()
FAIR_RPM = 12.0
print(f"limit = {FAIR_RPM:.0f} requests/min (covers the 86% of users who are")
print("interactive)")
print()
print(f"{'profile':>20}{'allowed req/min':>18}{'typical cost':>15}"
      f"{'peak cost':>12}{'vs ceiling':>13}")
print("-" * 78)
over = {}
for u in USERS:
    allowed = min(u[2], FAIR_RPM)
    typ = allowed * u[3]
    pk = allowed * (u[3] + 2.0 * u[4])
    over[u[0]] = pk / COST_CEILING
    print(f"{u[0]:>20}{allowed:>18.1f}{typ:>15.1f}{pk:>12.1f}"
          f"{pk / COST_CEILING:>12.2f}x")

print()
print()
print("The two horns, side by side. Neither request-count limit does both jobs.")
print()
print(f"{'limit':>26}{'throttling':>13}{'worst-case cost':>18}"
      f"{'bounds cost':>14}{'fair':>7}")
print("-" * 78)
for label, rpm in (("safe (%.1f req/min)" % SAFE_RPM, SAFE_RPM),
                   ("fair (%.0f req/min)" % FAIR_RPM, FAIR_RPM)):
    thr = sum(u[1] * (1.0 - min(u[2], rpm) / u[2]) for u in USERS)
    wc = max(min(u[2], rpm) * (u[3] + 2.0 * u[4]) for u in USERS)
    print(f"{label:>26}{thr:>13.0%}{wc:>18.1f}"
          f"{('yes' if wc <= COST_CEILING else 'no'):>14}"
          f"{('yes' if thr < 0.10 else 'no'):>7}")

print()
print()
print("A cost-based limiter instead: spend a budget of %.0f cost units per minute,"
      % COST_CEILING)
print("charging each request what it actually costs.")
print()
print(f"{'profile':>20}{'cost/min wanted':>18}{'budget':>10}{'served':>10}"
      f"{'throttled':>12}")
print("-" * 70)
cost_starve = {}
for u in USERS:
    want = cost_wanted(u)
    served_cost = min(want, COST_CEILING)
    thr = 1.0 - served_cost / want
    cost_starve[u[0]] = thr
    print(f"{u[0]:>20}{want:>18.1f}{COST_CEILING:>10.1f}{served_cost:>10.1f}"
          f"{thr:>12.0%}")

cost_weighted = sum(u[1] * cost_starve[u[0]] for u in USERS)
cost_worst = COST_CEILING
print()
print(f"population-weighted throttling: {cost_weighted:.0%}")
print(f"worst-case cost per tenant:     {cost_worst:.0f} (bounded by construction)")

print()
print()
print("All three, on the two questions a limiter has to answer.")
print()
print(f"{'limiter':>26}{'throttling':>13}{'worst-case cost':>18}"
      f"{'bounds cost':>14}{'fair':>7}")
print("-" * 78)
rows = [
    ("count, safe", weighted_starve,
     max(min(u[2], SAFE_RPM) * (u[3] + 2.0 * u[4]) for u in USERS)),
    ("count, fair", sum(u[1] * (1.0 - min(u[2], FAIR_RPM) / u[2]) for u in USERS),
     max(min(u[2], FAIR_RPM) * (u[3] + 2.0 * u[4]) for u in USERS)),
    ("cost-based", cost_weighted, cost_worst),
]
for label, thr, wc in rows:
    print(f"{label:>26}{thr:>13.0%}{wc:>18.1f}"
          f"{('yes' if wc <= COST_CEILING + 0.5 else 'no'):>14}"
          f"{('yes' if thr < 0.10 else 'no'):>7}")

print(f"""
The heterogeneity number is what breaks the usual design. The most expensive request
in this population costs {worst_unit:.1f} units and the cheapest typical one costs
{USERS[0][3]:.1f} -- a spread of **{worst_unit / USERS[0][3]:.0f} times** within a
single service. A request-count limiter has one number to set and that spread has to
fit inside it.

Set it safely and the arithmetic is brutal. Bounding spend at {COST_CEILING:.0f}
units a minute requires assuming every request is the most expensive kind, which
gives {SAFE_RPM:.1f} requests per minute. That throttles the light interactive
users -- who want {USERS[0][2]:.0f} requests a minute and cost
{cost_wanted(USERS[0]):.1f} units doing it -- by {starve['light interactive']:.0%},
and throttles {weighted_starve:.0%} of all requested traffic.

Look at what that means for the light users specifically. They are throttled
{starve['light interactive']:.0%} while consuming {cost_wanted(USERS[0]):.1f} units a
minute against a {COST_CEILING:.0f} unit budget -- **{cost_wanted(USERS[0]) / COST_CEILING:.1%}
of the ceiling the limit exists to protect.** The limit that stopped them was computed
from a request none of them will ever send.

Set it fairly and the cost bound evaporates. At {FAIR_RPM:.0f} requests a minute the
interactive users are served properly, and the agent workload -- {USERS[3][1]:.0%} of
users -- reaches {over['agent workload']:.2f} times the cost ceiling at peak
(eq:count-limits-cannot-bound-cost). One tenant on one plan can spend several times
what the plan was priced for, and the limiter records no violation because no
violation occurred.

That is the shape of the failure: **a request-count limiter is either unfair or
unbounded, and which one you get depends on a number set once by whoever configured
it.** There is no value that does both jobs, because the quantity being limited is
not the quantity being protected.

The cost-based limiter closes it by construction. Charging each request what it
actually costs and spending a budget of {COST_CEILING:.0f} units a minute throttles
{cost_weighted:.0%} of traffic -- against {weighted_starve:.0%} for the safe count
limit -- while bounding worst-case spend at exactly {cost_worst:.0f}, because the
bound IS the mechanism rather than an inference from one.

The light users are no longer throttled at all: at {cost_wanted(USERS[0]):.1f} units
a minute they are nowhere near a {COST_CEILING:.0f} budget, and their consumption is
priced correctly rather than assumed to be worst-case.

Two implementation notes follow from that, and both are awkward. The first is that
the cost of a request is not known when it arrives -- generation cost depends on
output length, which ch:sd-async established is unknown at admission. So a cost
limiter must either reserve a pessimistic amount and refund the difference, or admit
the request and charge afterwards, running slightly over budget by design.

The second is that a cost limiter needs a cost model, and a cost model is a thing that
drifts. When the model changes, the price per token changes, and every tenant's
effective rate limit changes with it -- silently, with no configuration edit and no
deployment anyone would connect to the resulting complaints.""")

# -*- coding: utf-8 -*-
# Extracted from: Chapter 204 — Cloud, Edge, and Local Deployment
# Source: src/.../ch204-cloud-edge-local.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Self-hosting beats an API at a specific utilisation, and it is higher than it looks.

The economic case for running your own inference is usually made on unit cost: a token
from your own GPU is cheaper than a token from an API. That is true at full utilisation
and only at full utilisation, because a rented GPU is billed by the hour whether or not
it is decoding (eq:self-hosting-is-a-utilisation-bet).

ch:inf-kubernetes established that utilisation is bounded by the trigger arithmetic --
a fleet that must absorb a ramp cannot run full. So the two results compose into a
crossover, and this listing computes where it sits.
"""
API_PER_MTOK = 0.62          # what a provider charges per million output tokens
GPU_PER_HOUR = 4.90
TOKENS_PER_SEC = 6500.0      # SUSTAINED, including prefill and real batch mix
SECONDS_PER_MONTH = 30.0 * 24.0 * 3600.0
OPS_PER_MONTH = 5800.0       # engineer time, monitoring, on-call, upgrades

MAX_TOK_MONTH = TOKENS_PER_SEC * SECONDS_PER_MONTH / 1e6   # millions


def self_cost(mtok_month, utilisation, replicas=None):
    """Monthly cost of self-hosting `mtok_month` million tokens."""
    capacity_per_replica = MAX_TOK_MONTH * utilisation
    if replicas is None:
        import math
        replicas = max(1, int(math.ceil(mtok_month / capacity_per_replica)))
    return replicas * GPU_PER_HOUR * 730.0 + OPS_PER_MONTH, replicas


print("One replica sustains %.0f output tokens/sec on a real request mix -- well"
      % TOKENS_PER_SEC)
print("below ch:inf-batching's peak, because production batches are not full and")
print("prefill shares the device. That is %.0f million tokens a month at"
      % MAX_TOK_MONTH)
print("100%% utilisation. A GPU costs %.2f an hour; ops overhead is %.0f a month."
      % (GPU_PER_HOUR, OPS_PER_MONTH))
print()
print("Unit cost of self-hosting, by utilisation. The API charges %.2f."
      % API_PER_MTOK)
print()
print(f"{'utilisation':>13}{'Mtok/month':>14}{'GPU cost':>11}"
      f"{'per Mtok':>11}{'vs API':>10}")
print("-" * 60)
unit = {}
for u in (1.00, 0.80, 0.60, 0.40, 0.20, 0.10):
    tok = MAX_TOK_MONTH * u
    gpu = GPU_PER_HOUR * 730.0
    per = gpu / tok
    unit[u] = (tok, gpu, per)
    print(f"{u:>13.0%}{tok:>14.0f}{gpu:>11.0f}{per:>11.4f}"
          f"{per / API_PER_MTOK:>9.2f}x")
print()
print("(GPU cost only -- ops overhead is added per month, not per token)")

print()
print()
print("Total monthly cost including ops, by volume. This is the comparison that")
print("decides it, and the ops term is what moves the crossover.")
print()
print(f"{'Mtok/month':>13}{'API cost':>11}{'replicas':>10}{'self cost':>12}"
      f"{'cheaper':>10}{'ratio':>9}")
print("-" * 66)
UTIL = 0.45                  # what ch:inf-kubernetes's trigger arithmetic permits
cross = None
tab = {}
for mtok in (5, 100, 1600, 6400, 25600, 51200, 102400):
    api = mtok * API_PER_MTOK
    sc, reps = self_cost(mtok, UTIL)
    tab[mtok] = (api, sc, reps)
    who = "API" if api < sc else "self"
    if cross is None and sc < api:
        cross = mtok
    print(f"{mtok:>13}{api:>11.0f}{reps:>10}{sc:>12.0f}{who:>10}"
          f"{max(api, sc) / min(api, sc):>8.2f}x")

# The true crossover, searched, accounting for replica rounding.
true_cross = None
m = 100
while m < 1000000:
    sc, _ = self_cost(m, UTIL)
    if sc < m * API_PER_MTOK:
        true_cross = m
        break
    m += 100
print()
print(f"crossover at {UTIL:.0%} utilisation: {true_cross} Mtok/month")
print(f"one-replica break-even (ignoring rounding): "
      f"{(GPU_PER_HOUR * 730.0 + OPS_PER_MONTH) / API_PER_MTOK:.0f} Mtok/month")

print()
print()
print("Where exactly, by utilisation. The break-even volume is the volume at")
print("which the API bill equals the fleet plus ops.")
print()
print(f"{'utilisation':>13}{'Mtok/replica':>15}{'break-even Mtok':>18}"
      f"{'as % of one replica':>22}{'replicas':>12}")
print("-" * 82)
be = {}
for u in (1.00, 0.80, 0.60, 0.45, 0.30, 0.15):
    cap = MAX_TOK_MONTH * u
    # One replica: cost = GPU*730 + OPS. Break even when mtok*API = that.
    fixed = GPU_PER_HOUR * 730.0 + OPS_PER_MONTH
    bev = fixed / API_PER_MTOK
    be[u] = (cap, bev, bev / cap)
    print(f"{u:>13.0%}{cap:>15.0f}{bev:>18.0f}{bev / cap:>21.0%}"
          f"{('1' if bev <= cap else '%.0f' % (bev / cap + 0.999)):>12}")

print()
print("(break-even is the same volume at every utilisation -- the fixed cost does")
print(" not change -- but whether ONE replica can serve it does)")

print()
print()
print("What the ops term does. It is the part teams underestimate, and it sets")
print("the floor on how small a self-hosted deployment can sensibly be.")
print()
print(f"{'ops per month':>15}{'break-even Mtok':>18}{'API bill there':>17}"
      f"{'ops share':>12}")
print("-" * 64)
opstab = {}
for ops in (0.0, 1500.0, 5800.0, 14000.0, 40000.0):
    fixed = GPU_PER_HOUR * 730.0 + ops
    bev = fixed / API_PER_MTOK
    opstab[ops] = (bev, ops / fixed)
    print(f"{ops:>15.0f}{bev:>18.0f}{bev * API_PER_MTOK:>17.0f}"
          f"{ops / fixed:>12.0%}")

print()
print()
print("And the reason the comparison is not only about money: three properties")
print("the price does not capture.")
print()
print(f"{'deployment':>22}{'per Mtok':>11}{'data leaves':>14}"
      f"{'p50 latency':>14}{'works offline':>15}")
print("-" * 78)
OPTIONS = [
    ("provider API",        API_PER_MTOK, "yes",   140.0, "no"),
    ("self-host, cloud",    unit[UTIL][2] if UTIL in unit else
     GPU_PER_HOUR * 730.0 / (MAX_TOK_MONTH * UTIL), "no", 95.0, "no"),
    ("self-host, on-prem",  0.31,         "no",    60.0,  "no"),
    ("edge appliance",      0.94,         "no",    45.0,  "yes"),
    ("on-device",           0.00,         "no",   310.0,  "yes"),
]
for label, per, leaves, lat, offline in OPTIONS:
    print(f"{label:>22}{per:>11.4f}{leaves:>14}{lat:>13.0f}m{offline:>15}")

print(f"""
The unit-cost table is the argument as usually made, and at the top of it the argument
is correct. At {1.0:.0%} utilisation a self-hosted replica delivers a million tokens for
{unit[1.0][2]:.4f} against the API's {API_PER_MTOK:.2f} --
{API_PER_MTOK / unit[1.0][2]:.0f} times cheaper.

Read down the column. At {0.40:.0%} utilisation it is {unit[0.4][2]:.4f}; at
{0.10:.0%} it is {unit[0.1][2]:.4f}, which is
{unit[0.1][2] / API_PER_MTOK:.2f} times the API price
(eq:self-hosting-is-a-utilisation-bet).

**A rented GPU bills by the hour whether or not it is decoding.** So the unit cost is
the hourly rate divided by whatever you actually put through it, and ch:inf-kubernetes
showed that "whatever you actually put through it" is bounded well below one by the
trigger arithmetic -- {1 - 0.45:.0%} of the fleet idle at a realistic ramp and cold
start.

The total-cost table adds the term the unit comparison omits. Self-hosting costs
{OPS_PER_MONTH:.0f} a month in engineer time, monitoring, on-call and upgrades before
a single token is served, and that is a fixed cost the API does not have.

At {5} million tokens a month the API costs {tab[5][0]:.0f} and self-hosting
{tab[5][1]:.0f} -- **{tab[5][1] / tab[5][0]:.0f} times more**. At {6400} million it is
{tab[6400][0]:.0f} against {tab[6400][1]:.0f}, still favouring the API. Self-hosting
first wins at **{true_cross} million tokens a month**.

To put that in perspective: {true_cross} million output tokens is roughly
{true_cross * 1e6 / 400 / 30 / 1e3:.0f} thousand generated answers a day at four hundred
tokens each. **Most products are nowhere near it**, and the ones that are usually know.

The break-even table separates two things that get conflated. The *fixed-cost*
break-even -- one replica plus ops against the API bill -- is
**{be[1.0][1]:.0f} million tokens a month**, and it does not move with utilisation
because fixed costs do not.

What utilisation changes is whether one replica can actually carry that volume. At
{1.0:.0%} utilisation the break-even volume is {be[1.0][2]:.0%} of a replica's capacity,
so one replica suffices. At {0.45:.0%} it is {be[0.45][2]:.0%} -- three replicas -- and
at {0.15:.0%}, {be[0.15][2]:.0%}, which is seven.

**Every replica added to cover the utilisation shortfall raises the break-even again**,
which is why the true crossover at {UTIL:.0%} utilisation is {true_cross} rather than
{be[1.0][1]:.0f}. The two compound: low utilisation means more replicas, more replicas
mean more fixed cost, more fixed cost means a higher volume is needed to justify it.

The ops table is where estimates go wrong, and it is worth dwelling on because the term
is invisible in every unit-cost comparison. At {0.0:.0f} a month of ops -- the implicit
assumption when someone compares token prices -- break-even is
{opstab[0.0][0]:.0f} million tokens. At a realistic {5800.0:.0f} it is
{opstab[5800.0][0]:.0f}. At {40000.0:.0f}, which is one experienced engineer, it is
{opstab[40000.0][0]:.0f}.

**The ops term is {opstab[5800.0][1]:.0%} of the fixed cost at a modest estimate and
{opstab[40000.0][1]:.0%} at a realistic one.** Self-hosting is mostly a payroll decision
wearing an infrastructure costume, and the GPU is the cheap part.

The last table is the reason none of this settles the question. An API is cheapest below
break-even and it sends your data somewhere else. On-device inference has no marginal
cost at all and a {310.0:.0f}ms latency that reflects a bandwidth-starved device
(ch:inf-cpu-gpu). An edge appliance works with no network.

**Three of the five options are chosen for reasons the cost column cannot express** --
data residency, offline operation, and latency floor -- and for those, the economics
determine only whether you can afford the choice you already had to make. The honest use
of this table is to price a constraint, not to select from it.""")

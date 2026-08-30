# -*- coding: utf-8 -*-
# Extracted from: Chapter 191 — Queues, Asynchronous Processing, and Streaming
# Source: src/.../ch191-async-streaming.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Heavy-tailed service times make queueing behave unlike anything in a web stack.

ch:sd-architecture found load balancing survives at 36% because requests stop being
equivalent. This listing measures what that actually does to a queue.

The mechanism is standard queueing theory, but the parameter regime is not. A web
request's service time has a coefficient of variation near 1; a generation request's
depends on OUTPUT LENGTH, which is not known when the request is admitted and varies
by an order of magnitude. Pollaczek-Khinchine says waiting time scales with the
SQUARE of that variability (eq:variance-not-mean-drives-wait).
"""
import math

# Six workloads with the same MEAN service time and different variability.
# (label, service times in seconds with equal probability each)
WORKLOADS = [
    ("uniform-ish",      [1.8, 1.9, 2.0, 2.1, 2.2]),
    ("mild spread",      [1.0, 1.5, 2.0, 2.5, 3.0]),
    ("web-like",         [0.5, 1.0, 1.8, 2.7, 4.0]),
    ("generation-like",  [0.4, 0.7, 1.2, 2.4, 5.3]),
    ("long tail",        [0.3, 0.4, 0.6, 1.2, 7.5]),
    ("very long tail",   [0.2, 0.3, 0.4, 0.6, 8.5]),
]


def moments(times):
    n = len(times)
    m1 = sum(times) / n
    m2 = sum(t * t for t in times) / n
    var = m2 - m1 * m1
    return m1, var, math.sqrt(var) / m1


def pk_wait(m1, m2_raw, lam):
    """Pollaczek-Khinchine mean waiting time in an M/G/1 queue."""
    rho = lam * m1
    if rho >= 1.0:
        return float("inf")
    return lam * m2_raw / (2 * (1 - rho))


print("Six workloads with the SAME mean service time and different variability.")
print("Coefficient of variation (CV) is the standard deviation over the mean.")
print()
print(f"{'workload':>18}{'mean':>8}{'variance':>11}{'CV':>8}{'CV squared':>13}")
print("-" * 58)
mom = {}
for label, times in WORKLOADS:
    m1, var, cv = moments(times)
    mom[label] = (m1, var, cv, sum(t * t for t in times) / len(times))
    print(f"{label:>18}{m1:>8.2f}{var:>11.3f}{cv:>8.2f}{cv * cv:>13.2f}")

print()
print()
print("Mean wait in the queue at 70% utilisation. The mean service time is")
print("identical across every row; only the spread differs.")
print()
LAM = 0.70 / mom["uniform-ish"][0]
print(f"{'workload':>18}{'CV':>8}{'mean wait':>12}{'vs uniform':>13}"
      f"{'total latency':>15}")
print("-" * 66)
waits = {}
for label, times in WORKLOADS:
    m1, var, cv, m2raw = mom[label]
    w = pk_wait(m1, m2raw, LAM)
    waits[label] = w
    ratio = w / pk_wait(*[mom["uniform-ish"][i] for i in (0, 3)], LAM)
    print(f"{label:>18}{cv:>8.2f}{w:>12.2f}s{ratio:>12.1f}x{w + m1:>14.2f}s")

print()
print()
print("The same workloads as utilisation rises. This is where a capacity plan")
print("built on mean service time goes wrong.")
print()
UTILS = [0.50, 0.70, 0.80, 0.90, 0.95]
print(f"{'workload':>18}" + "".join(f"{u:>11.0%}" for u in UTILS))
print("-" * 73)
grid = {}
for label, times in WORKLOADS:
    m1, var, cv, m2raw = mom[label]
    row = []
    for u in UTILS:
        lam = u / m1
        row.append(pk_wait(m1, m2raw, lam))
    grid[label] = row
    print(f"{label:>18}" + "".join(f"{w:>10.2f}s" for w in row))

print()
print()
print("What utilisation each workload can actually sustain under a 3-second")
print("wait budget -- the number a capacity plan needs and the mean cannot give.")
print()
BUDGET = 3.0
print(f"{'workload':>18}{'CV':>8}{'max utilisation':>18}{'headroom lost':>16}")
print("-" * 62)
cap = {}
for label, times in WORKLOADS:
    m1, var, cv, m2raw = mom[label]
    lo, hi = 0.0, 0.999
    for _ in range(60):
        mid = (lo + hi) / 2
        if pk_wait(m1, m2raw, mid / m1) <= BUDGET:
            lo = mid
        else:
            hi = mid
    cap[label] = lo
    print(f"{label:>18}{cv:>8.2f}{lo:>18.1%}"
          f"{cap['uniform-ish'] - lo:>16.1%}")

print()
print()
print("And the cost of that lost headroom, in machines. Serving the same traffic")
print("at the utilisation each workload can actually sustain:")
print()
print(f"{'workload':>18}{'max utilisation':>18}{'machines needed':>18}"
      f"{'vs uniform':>13}")
print("-" * 68)
base = 1.0 / cap["uniform-ish"]
fleet = {}
for label, times in WORKLOADS:
    n = 1.0 / cap[label]
    fleet[label] = n
    print(f"{label:>18}{cap[label]:>18.1%}{n:>18.2f}{n / base:>12.2f}x")

print(f"""
Every workload in the first table has the same mean service time. A capacity plan
built on means cannot tell them apart, and a dashboard reporting mean latency will
show them as identical systems.

They are not. At {0.70:.0%} utilisation the mean wait ranges from
{waits['uniform-ish']:.2f}s to {waits['very long tail']:.2f}s -- a factor of
{waits['very long tail'] / waits['uniform-ish']:.0f} between the tightest and the
loosest, driven entirely by variance (eq:variance-not-mean-drives-wait).

The reason is in the Pollaczek-Khinchine formula: waiting time depends on the
SECOND moment of service time, so it scales with the square of the coefficient of
variation. The `very long tail` row has a CV of
{mom['very long tail'][2]:.2f}, and {mom['very long tail'][2]:.2f} squared is
{mom['very long tail'][2] ** 2:.1f} -- which is most of the factor of
{waits['very long tail'] / waits['uniform-ish']:.0f}.

**Generation workloads live at the wrong end of this table.** Output length varies
by an order of magnitude and is unknown at admission, so a queue of generation
requests has a CV near the `generation-like` row
({mom['generation-like'][2]:.2f}) or worse, where the wait is already
{waits['generation-like'] / waits['uniform-ish']:.1f} times the uniform case.

The utilisation grid is where this becomes a capacity decision rather than a
curiosity. Under a {BUDGET:.0f}-second wait budget, the uniform workload sustains
{cap['uniform-ish']:.0%} utilisation. The `generation-like` workload sustains
{cap['generation-like']:.0%}, and `very long tail` sustains
{cap['very long tail']:.0%}.

That translates directly into hardware. Serving the same arrival rate at each
workload's sustainable utilisation needs
{fleet['generation-like'] / base:.2f} times the machines for `generation-like` and
{fleet['very long tail'] / base:.2f} times for `very long tail` -- **for identical
mean service time and identical traffic**.

So the practical rule is short and it contradicts the usual instinct. When a queue
of model calls is slow, the first move is not more capacity and not a faster model.
**It is reducing the variance of service time**, because variance is what the wait
is made of. Splitting long generations out of the main queue, capping output length,
or batching by expected length all attack the second moment directly, and each one
buys more than the equivalent spend on machines.

This is also why ch:sd-architecture found load balancing surviving at only
{0.36:.0%}. Balancing distributes requests evenly, which is the right policy when
requests are equivalent. Here it distributes a heavy tail evenly across every
worker, guaranteeing that every worker has one -- when the better policy is to
concentrate the tail somewhere it can be managed.""")

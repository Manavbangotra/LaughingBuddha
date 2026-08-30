# -*- coding: utf-8 -*-
# Extracted from: Chapter 196 — Latency Budgets and Performance Engineering
# Source: src/.../ch196-latency.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The sum of per-stage p99s is not the p99 of the sum, and budgeting as if it were
wastes most of the budget.

A latency budget gets divided among stages: the gateway gets 20ms, retrieval gets
150ms, generation gets 600ms. The natural way to do that division is to give each
stage a share of the target and require each to meet it at p99.

That is wrong, and expensively so. Stages do not hit their tails simultaneously, so
requiring every stage to meet a p99 budget provisions for a coincidence that almost
never happens (eq:sum-of-tails-overprovisions).

This listing measures the gap, and shows what to allocate instead.
"""
import math
import random

# Pipeline stages. (name, mean ms, standard deviation ms)
# Lognormal, because latency is non-negative and right-skewed.
STAGES = [
    ("gateway and auth",     8.0,    4.0),
    ("retrieval",          120.0,   95.0),
    ("rerank",              40.0,   22.0),
    ("prompt assembly",     14.0,    6.0),
    ("generation",         520.0,  310.0),
    ("validate and format", 18.0,   11.0),
]
TARGET_P99 = 2200.0
TRIALS = 160000
SEED = 20260829


def lognormal_params(mean, sd):
    var = math.log(1.0 + (sd * sd) / (mean * mean))
    return math.log(mean) - var / 2.0, math.sqrt(var)


PARAMS = [(n,) + lognormal_params(m, s) + (m, s) for n, m, s in STAGES]


def percentile(xs, q):
    xs = sorted(xs)
    i = min(len(xs) - 1, int(q * len(xs)))
    return xs[i]


def simulate():
    rng = random.Random(SEED)
    totals = []
    per_stage = [[] for _ in PARAMS]
    for _ in range(TRIALS):
        t = 0.0
        for i, (_, mu, sig, _, _) in enumerate(PARAMS):
            v = math.exp(rng.gauss(mu, sig))
            per_stage[i].append(v)
            t += v
        totals.append(t)
    return totals, per_stage


totals, per_stage = simulate()

print("A six-stage request path. Each stage's latency is lognormal.")
print()
print(f"{'stage':>22}{'mean':>10}{'p50':>10}{'p99':>10}{'p99/mean':>11}")
print("-" * 63)
stage_p99 = {}
stage_mean = {}
for i, (name, mu, sig, m, s) in enumerate(PARAMS):
    p99 = percentile(per_stage[i], 0.99)
    p50 = percentile(per_stage[i], 0.50)
    stage_p99[name] = p99
    stage_mean[name] = m
    print(f"{name:>22}{m:>9.1f}m{p50:>9.1f}m{p99:>9.1f}m{p99 / m:>11.2f}")

sum_mean = sum(m for _, _, _, m, _ in PARAMS)
sum_p99 = sum(stage_p99.values())
sys_p99 = percentile(totals, 0.99)
sys_p50 = percentile(totals, 0.50)
sys_mean = sum(totals) / len(totals)

print()
print()
print("Three ways to add those up. Only one of them is the system's p99.")
print()
print(f"{'quantity':>34}{'value':>12}{'vs true p99':>14}")
print("-" * 60)
print(f"{'sum of per-stage means':>34}{sum_mean:>11.1f}m{sum_mean / sys_p99:>13.2f}x")
print(f"{'measured system p50':>34}{sys_p50:>11.1f}m{sys_p50 / sys_p99:>13.2f}x")
print(f"{'measured system mean':>34}{sys_mean:>11.1f}m{sys_mean / sys_p99:>13.2f}x")
print(f"{'MEASURED SYSTEM p99':>34}{sys_p99:>11.1f}m{1.0:>13.2f}x")
print(f"{'sum of per-stage p99s':>34}{sum_p99:>11.1f}m{sum_p99 / sys_p99:>13.2f}x")

print()
print()
print("What that does to a budget. A %.0fms target divided by per-stage p99 share"
      % TARGET_P99)
print("asks every stage to fit a coincidence that almost never happens.")
print()
print(f"{'stage':>22}{'actual p99':>13}{'budget share':>15}"
      f"{'allocated':>12}{'slack':>10}")
print("-" * 72)
alloc = {}
for name, mu, sig, m, s in PARAMS:
    share = stage_p99[name] / sum_p99
    a = TARGET_P99 * share
    alloc[name] = a
    print(f"{name:>22}{stage_p99[name]:>12.1f}m{share:>15.1%}"
          f"{a:>11.1f}m{a - stage_p99[name]:>9.1f}m")

print()
print(f"total allocated: {sum(alloc.values()):.0f}ms")
print(f"true p99 of the pipeline as configured: {sys_p99:.0f}ms")
print(f"headroom the budget thinks it needs: {sum_p99 / sys_p99:.2f}x the real p99")

print()
print()
print("How much each stage can actually be allowed to grow before the SYSTEM")
print("misses its p99 target. One stage at a time, everything else held fixed.")
print()
print(f"{'stage':>22}{'current mean':>15}{'max mean':>12}"
      f"{'growth allowed':>17}")
print("-" * 68)


def system_p99_with(idx, scale):
    rng = random.Random(SEED)
    out = []
    for _ in range(TRIALS // 8):
        t = 0.0
        for i, (_, mu, sig, m, s) in enumerate(PARAMS):
            v = math.exp(rng.gauss(mu, sig))
            if i == idx:
                v *= scale
            t += v
        out.append(t)
    return percentile(out, 0.99)


room = {}
for i, (name, mu, sig, m, s) in enumerate(PARAMS):
    lo, hi = 1.0, 60.0
    for _ in range(18):
        mid = (lo + hi) / 2.0
        if system_p99_with(i, mid) <= TARGET_P99:
            lo = mid
        else:
            hi = mid
    room[name] = lo
    print(f"{name:>22}{m:>14.1f}m{m * lo:>11.1f}m{lo:>16.1f}x")

print()
print()
print("The two allocations side by side. One is derived from per-stage tails;")
print("the other from what the system can actually absorb.")
print()
print(f"{'stage':>22}{'p99-share budget':>19}{'absorbable':>13}"
      f"{'ratio':>10}")
print("-" * 66)
for name, mu, sig, m, s in PARAMS:
    ab = m * room[name]
    print(f"{name:>22}{alloc[name]:>18.1f}m{ab:>12.1f}m"
          f"{ab / alloc[name]:>10.2f}x")

print(f"""
The three-ways table is the arithmetic that gets this wrong. Summing per-stage means
gives {sum_mean:.1f}ms, which is {sum_mean / sys_p99:.2f} times the real p99 --
far too low, as everyone expects. Summing per-stage p99s gives {sum_p99:.1f}ms,
which is **{sum_p99 / sys_p99:.2f} times the real p99** and is the number a
per-stage budget is implicitly built on.

The true system p99 is {sys_p99:.1f}ms, and it sits between them
(eq:sum-of-tails-overprovisions). Neither summary statistic composes: means add but
tails do not, because a request that is unlucky in retrieval is usually ordinary in
generation.

The budget table shows the practical consequence, and it is worth reading the slack
column carefully. **Every single stage misses its allocation.** Generation is given
{alloc['generation']:.0f}ms against an actual p99 of
{stage_p99['generation']:.0f}ms; retrieval is given {alloc['retrieval']:.0f}ms
against {stage_p99['retrieval']:.0f}ms; even the gateway, at
{stage_p99['gateway and auth']:.1f}ms, is over its {alloc['gateway and auth']:.1f}ms
share.

Six stages, six failures, and a pipeline whose true p99 is {sys_p99:.0f}ms against a
{TARGET_P99:.0f}ms target -- **passing comfortably, with
{TARGET_P99 - sys_p99:.0f}ms to spare.**

That is the failure mode in one table. A per-stage p99 budget can report every
component out of compliance while the system it describes is meeting its target
easily, because the budget was computed by assuming all six stages hit their tails on
the same request. They do not, and the arithmetic that assumed they would is the
{sum_p99 / sys_p99:.2f}x gap in the row above.

The absorbable table is what the system can really take. Generation's mean can grow
to {room['generation']:.1f} times its current value before the system p99 misses
{TARGET_P99:.0f}ms; retrieval's can grow {room['retrieval']:.1f} times, and the
gateway's {room['gateway and auth']:.0f} times.

**The stages differ enormously in how much slack they have, and per-stage p99 budgets
do not reflect that at all.** The gateway is allocated
{alloc['gateway and auth']:.0f}ms and could absorb
{stage_mean['gateway and auth'] * room['gateway and auth']:.0f}ms; generation is
allocated {alloc['generation']:.0f}ms and could absorb
{stage_mean['generation'] * room['generation']:.0f}ms.

Two conclusions follow, and they point in opposite directions from the usual advice.

The first is that **a per-stage p99 budget over-constrains almost every stage**, and
teams spend engineering effort meeting numbers the system never needed. The
{sum_p99 / sys_p99:.2f}x factor is pure over-provisioning: work done to prevent a
coincidence.

The second is that the slack is not distributed the way the budget assumes. A stage
with a large mean and a tight distribution -- generation, here -- has less headroom
than its budget share suggests, while a small, noisy stage has vastly more. Allocating
by p99 share gets the ORDER right and the MAGNITUDES badly wrong.

The correct procedure is the one the last table performs: hold the system target
fixed, vary one stage, and measure what the system absorbs. It requires a simulation
or a load test rather than arithmetic on a dashboard, which is why it is rarely done
-- and it is the only method that answers the question a budget is supposed to
answer.""")

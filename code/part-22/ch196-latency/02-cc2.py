# -*- coding: utf-8 -*-
# Extracted from: Chapter 196 — Latency Budgets and Performance Engineering
# Source: src/.../ch196-latency.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Optimising the slowest stage is usually the wrong move for p99.

Given a latency problem, the instinct is to find the biggest number and shrink it.
That instinct optimises the MEAN, and the target is almost always the tail.

A stage contributes to the mean through its mean and to the tail through its
variance. Those rankings differ, so the stage worth working on depends on which
statistic you are trying to move (eq:tail-attribution-differs-from-mean).

This listing ranks six stages three ways -- by mean contribution, by tail
contribution, and by tail reduction per unit of engineering effort -- and finds the
three rankings disagree.
"""
import math
import random

# (name, mean ms, std dev ms, engineering weeks to halve its mean,
#  engineering weeks to halve its std dev)
STAGES = [
    ("gateway and auth",     8.0,    4.0,  2.0,  1.0),
    ("retrieval",          120.0,  240.0,  6.0,  3.0),
    ("rerank",              40.0,   22.0,  3.0,  2.0),
    ("prompt assembly",     14.0,    6.0,  1.0,  1.0),
    ("generation",         520.0,  200.0, 14.0, 10.0),
    ("validate and format", 18.0,   11.0,  2.0,  1.0),
]
TRIALS = 120000
SEED = 20260829


def ln(mean, sd):
    var = math.log(1.0 + (sd * sd) / (mean * mean))
    return math.log(mean) - var / 2.0, math.sqrt(var)


def percentile(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def run(stages):
    rng = random.Random(SEED)
    par = [ln(m, s) for _, m, s, _, _ in stages]
    out = []
    for _ in range(TRIALS):
        t = 0.0
        for mu, sig in par:
            t += math.exp(rng.gauss(mu, sig))
        out.append(t)
    return out


base = run(STAGES)
base_p99 = percentile(base, 0.99)
base_mean = sum(base) / len(base)
total_mean = sum(m for _, m, _, _, _ in STAGES)
total_var = sum(s * s for _, _, s, _, _ in STAGES)

print("Six stages, with what it would cost to improve each.")
print()
print(f"{'stage':>22}{'mean':>9}{'std dev':>10}{'weeks to':>11}{'weeks to':>11}")
print(f"{'':>22}{'':>9}{'':>10}{'halve mean':>11}{'halve sd':>11}")
print("-" * 63)
for n, m, s, wm, ws in STAGES:
    print(f"{n:>22}{m:>8.1f}m{s:>9.1f}m{wm:>11.1f}{ws:>11.1f}")

print()
print(f"system mean {base_mean:.1f}ms, p99 {base_p99:.1f}ms")

print()
print()
print("Three attributions of the same pipeline.")
print()
print(f"{'stage':>22}{'share of mean':>16}{'share of variance':>20}"
      f"{'rank by mean':>15}{'rank by var':>14}")
print("-" * 87)
by_mean = sorted(STAGES, key=lambda x: -x[1])
by_var = sorted(STAGES, key=lambda x: -(x[2] ** 2))
rm = {x[0]: i + 1 for i, x in enumerate(by_mean)}
rv = {x[0]: i + 1 for i, x in enumerate(by_var)}
for n, m, s, wm, ws in STAGES:
    print(f"{n:>22}{m / total_mean:>16.1%}{s * s / total_var:>20.1%}"
          f"{rm[n]:>15}{rv[n]:>14}")

print()
print()
print("Now measure it instead of inferring it: halve one stage's mean, and see")
print("what happens to the system p99.")
print()
print(f"{'stage halved (mean)':>22}{'new p99':>11}{'p99 saved':>12}"
      f"{'weeks':>8}{'ms per week':>14}")
print("-" * 67)
mean_eff = {}
for i, (n, m, s, wm, ws) in enumerate(STAGES):
    mod = list(STAGES)
    mod[i] = (n, m / 2.0, s, wm, ws)
    p = percentile(run(mod), 0.99)
    saved = base_p99 - p
    mean_eff[n] = (p, saved, saved / wm)
    print(f"{n:>22}{p:>10.1f}m{saved:>11.1f}m{wm:>8.1f}{saved / wm:>14.2f}")

print()
print()
print("And the other lever: halve one stage's standard deviation, leaving its")
print("mean alone.")
print()
print(f"{'stage halved (sd)':>22}{'new p99':>11}{'p99 saved':>12}"
      f"{'weeks':>8}{'ms per week':>14}")
print("-" * 67)
sd_eff = {}
for i, (n, m, s, wm, ws) in enumerate(STAGES):
    mod = list(STAGES)
    mod[i] = (n, m, s / 2.0, wm, ws)
    p = percentile(run(mod), 0.99)
    saved = base_p99 - p
    sd_eff[n] = (p, saved, saved / ws)
    print(f"{n:>22}{p:>10.1f}m{saved:>11.1f}m{ws:>8.1f}{saved / ws:>14.2f}")

print()
print()
print("Every option ranked by p99 milliseconds bought per engineering week.")
print()
opts = []
for n, m, s, wm, ws in STAGES:
    opts.append((n + " (mean)", mean_eff[n][1], wm, mean_eff[n][2]))
    opts.append((n + " (variance)", sd_eff[n][1], ws, sd_eff[n][2]))
opts.sort(key=lambda o: -o[3])
print(f"{'rank':>6}{'intervention':>34}{'p99 saved':>12}{'weeks':>8}"
      f"{'ms per week':>14}")
print("-" * 74)
for i, (label, saved, wk, eff) in enumerate(opts, 1):
    print(f"{i:>6}{label:>34}{saved:>11.1f}m{wk:>8.1f}{eff:>14.2f}")

print()
print()
print("The instinct against the measurement.")
print()
biggest = by_mean[0][0]
best = opts[0]
print(f"{'approach':>34}{'intervention':>26}{'ms per week':>14}")
print("-" * 74)
print(f"{'optimise the slowest stage':>34}{(biggest + ' (mean)'):>26}"
      f"{mean_eff[biggest][2]:>14.2f}")
print(f"{'optimise by measured efficiency':>34}{best[0]:>26}{best[3]:>14.2f}")

print(f"""
The attribution table is the first thing worth reading twice.
`{by_mean[0][0]}` is {by_mean[0][1] / total_mean:.1%} of the mean and only
{by_mean[0][2] ** 2 / total_var:.1%} of the variance. `{by_var[0][0]}` is
{by_var[0][1] / total_mean:.1%} of the mean and {by_var[0][2] ** 2 / total_var:.1%}
of the variance.

**The two rankings swap their top two entries**
(eq:tail-attribution-differs-from-mean). A stage that is large and steady dominates
the mean; a stage that is smaller and erratic dominates the tail. And a p99 target is
a statement about the tail.

The measured tables confirm it and add the part attribution cannot supply: cost.
Halving `{biggest}`'s mean -- the single largest number in the pipeline, and the
target anyone would nominate first -- saves {mean_eff[biggest][1]:.1f}ms of p99 for
{[w for n, m, sd, w, ws in STAGES if n == biggest][0]:.0f} weeks of work, or
{mean_eff[biggest][2]:.2f}ms per week.

Halving `{by_var[0][0]}`'s standard deviation saves {sd_eff[by_var[0][0]][1]:.1f}ms
for {[ws for n, m, sd, w, ws in STAGES if n == by_var[0][0]][0]:.0f} weeks --
**{sd_eff[by_var[0][0]][2]:.2f}ms per week**.

That is {sd_eff[by_var[0][0]][2] / mean_eff[biggest][2]:.1f} times the return, and it
is not a trade-off: the better option is also **more absolute improvement**
({sd_eff[by_var[0][0]][1]:.0f}ms against {mean_eff[biggest][1]:.0f}ms) for **less
work** ({[ws for n, m, sd, w, ws in STAGES if n == by_var[0][0]][0]:.0f} weeks against
{[w for n, m, sd, w, ws in STAGES if n == biggest][0]:.0f}). The instinct does not
merely pick a worse option; it picks one that is worse on every axis at once.

Note that this is not a general claim that variance always beats mean. Look at
`{biggest}` on its own: halving its mean returns
{mean_eff[biggest][2]:.2f}ms per week and halving its variance returns
{sd_eff[biggest][2]:.2f}. For that stage the conventional lever is the better one.
**The variance lever wins where the variance is**, which is a different stage from
where the time is, and that is the whole point.

The bottom of the ranked table is worth a look as well. Several interventions return
under {1.0:.1f}ms per week, and two are indistinguishable from zero -- narrowing
stages so small and so steady that the system p99 does not move at all. None of them
is an obviously foolish choice; they are stages a reasonable person could nominate in
a planning meeting after looking at a flame graph.

The spread from the best option to the worst positive one is
{opts[0][3] / max(min(o[3] for o in opts if o[3] > 0.05), 0.01):.0f} to one. **That
ratio is the cost of choosing by intuition instead of by measurement**, and it is
larger than any efficiency gain the chosen work is likely to deliver.

Three things follow for practice.

**Rank by variance contribution, not by mean.** The mean tells you where the time
goes; the variance tells you where the tail comes from. A percentile target is a
statement about the tail, and the two rankings are not the same list.

**Consider narrowing as well as shrinking.** Caching a slow path, capping a length,
bounding a retry, or removing an occasional expensive branch all attack variance --
and they are frequently cheaper than making the common path faster, because the common
path is the one that has already been optimised. Whether narrowing beats shrinking is
per-stage, and the table answers it per-stage.

**Measure the counterfactual rather than attributing.** Every number in the two
measured tables came from re-running the pipeline with one stage changed. That is a
few lines of simulation over a distribution obtainable from tracing, and it answers
the question directly instead of inferring it from a decomposition that assumes
independence and additivity -- neither of which survives contact with a real system,
as ch:sd-async's queueing coupling and ch:sd-retrieval-agents's correlated fan-out
both showed.""")

---
id: sd-latency
number: 196
part: XXII
tier: full
status: draft
requires: [variance-not-mean-drives-wait, fanout-amplifies-the-tail,
           streaming-capacity-is-set-by-ttft, three-properties-break-the-stack]
provides: [sum-of-tails-overprovisions, absorbable-slack-is-not-budget-share,
           tail-attribution-differs-from-mean, narrowing-competes-with-shrinking]
citations: [pope2022inference, kwon2023pagedattention, leviathan2023speculative]
---

## 1. Learning Objectives

By the end of this chapter you will be able to explain why the sum of per-stage p99s
overstates a system's p99, and compute the over-provisioning factor for a given
pipeline; show why a per-stage percentile budget can report every component failing
while the system passes comfortably; measure how much latency each stage can actually
absorb before a system target is missed, rather than inferring it from budget share;
distinguish a stage's contribution to the mean from its contribution to the tail, and
say which one a percentile target depends on; and rank latency work by milliseconds
of p99 bought per engineering week rather than by which number on the flame graph is
largest.

## 2. Why This Matters

This part has spent six chapters establishing that model-backed systems have a
variance problem: {{ch:sd-async}} in the queue, {{ch:sd-retrieval-agents}} in the
fan-out, {{ch:sd-storage}} in the derivation chain. This chapter is what that means
for the discipline of latency engineering, and it turns out to invalidate two
practices that are close to universal.

The first is the per-stage latency budget. Dividing a target among stages and
requiring each to meet its share at p99 assumes tails add, and they do not.
{{sec:9-practical-example}} measures a pipeline whose per-stage p99s sum to
**2314ms** against a true system p99 of **1827ms** — a **1.27×** over-provisioning
factor ({{eq:sum-of-tails-overprovisions}}). Under a 2200ms target allocated by p99
share, **every one of six stages misses its allocation** while the pipeline passes
with **373ms** to spare.

The second is optimising the largest number. In the same pipeline, generation is
**72.2%** of the mean and only **40.7%** of the variance, while retrieval is
**16.7%** of the mean and **58.6%** of the variance
({{eq:tail-attribution-differs-from-mean}}). The obvious target returns **11.82ms**
of p99 per engineering week; the measured best returns **93.39ms** — for less work
and more absolute improvement.

## 3. Prerequisites

You need {{eq:variance-not-mean-drives-wait}} from {{ch:sd-async}}, which established
that distributional shape rather than central tendency determines what users
experience. This chapter applies that to the composition of stages rather than to a
single queue.

{{eq:fanout-amplifies-the-tail}} from {{ch:sd-retrieval-agents}} is the same
composition question for parallel rather than sequential stages, and
{{sec:15-advanced-concepts}} connects them.

{{eq:streaming-capacity-is-set-by-ttft}} from {{ch:sd-async}} matters because it
determines *which* latency a budget should be written about.

{{eq:three-properties-break-the-stack}} supplies the reason any of this is unusual:
conventional pipelines have tighter distributions and the errors here are smaller.

Basic familiarity with percentiles and lognormal distributions is assumed.

## 4. Intuitive Explanation

Here is the mistake, and it is so natural that essentially everyone makes it.

You have a 2-second latency target. Your request passes through six stages. You divide
the budget: the gateway gets 20ms, retrieval gets 450ms, generation gets 1500ms, and
so on. Each team is told to hit their number at p99. Everyone agrees this is
reasonable.

Now ask what you have actually required. You have required that the gateway be fast
even on its worst 1-in-100 request, *and* that retrieval be fast on its worst
1-in-100, *and* that generation be fast on its worst 1-in-100 — all at once, on the
same request.

But that is not what happens. A request that has an unlucky retrieval usually has a
perfectly ordinary generation. The tails do not coincide, because the stages are
mostly independent, and the probability of all six being simultaneously unlucky is
astronomically small.

So the sum of the per-stage p99s describes a request that essentially never occurs.
Budget against it and you have committed your teams to preventing a coincidence.

The practical symptom is strange enough to be worth recognising: **every component
reports failing its budget while the system comfortably meets its target.** That is
not a measurement error and it is not a paradox. It is what happens when you add up
numbers that do not add.

The second half of the chapter is about where to spend effort once you have a real
target. The instinct is to look at a flame graph, find the widest bar, and make it
narrower. That instinct is correct if you are optimising the mean.

But a p99 target is not about the mean. It is about how far the *bad* requests are
from the typical ones, and that is a question about spread. A stage that always takes
exactly 520ms contributes enormously to your mean and nothing whatsoever to your tail
— it moves every request equally, including the fast ones. A stage that usually takes
30ms and occasionally takes 900ms contributes little to the mean and may be the
entire reason your p99 is what it is.

So there are two different questions with two different answers: *where does the time
go* and *where does the tail come from*. Flame graphs answer the first. Percentile
targets ask the second.

And there are correspondingly two kinds of intervention. You can make a stage
**faster** — shrink its mean — or you can make it **steadier** — shrink its spread.
Caching an occasional slow path, capping a length, or bounding a retry are all
steadying moves. They often cost less than making the common path faster, because the
common path is the one that has already been optimised.

Which of those wins is not a general rule. It is a per-stage question, and the only
honest way to answer it is to try each one and measure.

## 5. Formal Explanation

Let a request pass through $n$ independent stages with latencies $X_1,\ldots,X_n$ and
total $S = \sum_i X_i$. Means add exactly:

$$ \mathbb{E}[S] \;=\; \sum_i \mathbb{E}[X_i] $$

Variances also add, under independence:

$$ \operatorname{Var}(S) \;=\; \sum_i \operatorname{Var}(X_i) $$

Percentiles do neither. Writing $q_p(\cdot)$ for the $p$-th quantile, the general
relation is

$$ q_p(S) \;\le\; \sum_i q_p(X_i) $$ (eq:sum-of-tails-overprovisions)

with equality only in the degenerate case of perfectly comonotonic stages — stages
that are unlucky together, always. For independent stages the gap is large, and it
grows with $n$: summing $n$ upper quantiles requires all $n$ to be simultaneously
extreme, an event of probability $(1-p)^n$ rather than $1-p$.

A useful approximation for the independent case comes from the central limit
behaviour of the sum. If each stage contributes mean $\mu_i$ and variance
$\sigma_i^2$, then for moderate $n$,

$$ q_p(S) \;\approx\; \sum_i \mu_i \;+\; z_p\sqrt{\sum_i \sigma_i^2} $$

against the budgeting assumption $\sum_i (\mu_i + z_p\sigma_i)$. The ratio is

$$ \frac{\sum_i \sigma_i}{\sqrt{\sum_i \sigma_i^2}} $$

which is the $\ell_1/\ell_2$ norm ratio of the standard deviations — bounded above by
$\sqrt{n}$ and equal to 1 only when a single stage dominates. **The over-provisioning
factor is a property of how evenly variance is spread across stages**, and it
approaches $\sqrt{n}$ as they equalise.

The second structure concerns what a stage can absorb. Define the *absorbable growth*
of stage $i$ under target $T$ as the largest $\alpha$ with
$q_p(S \mid X_i \to \alpha X_i) \le T$. This is not proportional to the stage's budget
share:

$$ \alpha_i^\star \;\ne\; f\!\left(\frac{q_p(X_i)}{\sum_j q_p(X_j)}\right) $$ (eq:absorbable-slack-is-not-budget-share)

because scaling $X_i$ scales both $\mu_i$ and $\sigma_i$, and their contributions to
$q_p(S)$ enter linearly and in quadrature respectively.

## 6. Mathematical Foundation

The attribution result follows from differentiating the quantile approximation.
Stage $i$'s marginal effect on the system p99 through its mean is

$$ \frac{\partial q_p(S)}{\partial \mu_i} \;=\; 1 $$

— identical for every stage, regardless of size. Its marginal effect through its
standard deviation is

$$ \frac{\partial q_p(S)}{\partial \sigma_i} \;=\; \frac{z_p\,\sigma_i}{\sqrt{\sum_j \sigma_j^2}} $$ (eq:tail-attribution-differs-from-mean)

which is **proportional to $\sigma_i$**. So a unit reduction in any stage's mean buys
the same p99 as a unit reduction in any other's, while a unit reduction in standard
deviation buys p99 in proportion to how noisy that stage already is.

That asymmetry is the whole result. Mean reductions are size-blind; variance
reductions concentrate on the noisiest stage. And since improvements are usually
proportional rather than absolute — you halve a stage, you do not remove 40ms from it
— the comparison that matters is between halving $\mu_i$, worth $\mu_i/2$, and
halving $\sigma_i$, worth $z_p\sigma_i/(2\sqrt{\sum_j\sigma_j^2}) \cdot \sigma_i$.

Setting those equal gives the condition under which narrowing beats shrinking:

$$ \frac{\sigma_i^2}{\sum_j \sigma_j^2} \;>\; \frac{\mu_i}{z_p\sqrt{\sum_j \sigma_j^2}} $$ (eq:narrowing-competes-with-shrinking)

**Narrowing wins where a stage's share of total variance exceeds its mean scaled by
the system's overall spread.** This is a per-stage test, not a general preference, and
{{sec:9-practical-example}} finds it going both ways within one pipeline: narrowing
wins decisively for retrieval and loses for generation.

## 7. Internal Mechanics

**Which latency the budget is about.** {{eq:streaming-capacity-is-set-by-ttft}}
established that below the streaming crossover, perceived wait is time-to-first-token
and answer length is free. A budget written about total latency therefore constrains
something the user does not experience, while leaving the thing they do experience
unbudgeted. Systems that stream should budget TTFT and total separately, with
different targets and different owners.

**Where the variance comes from, stage by stage.** Retrieval variance is dominated by
cache state and index traversal depth. Generation variance is dominated by output
length, which {{ch:sd-async}} showed is unknown at admission. Prefill variance is
dominated by *input* length ({{cite:pope2022inference}}), which grows with retrieved
context — so a change in retrieval breadth raises variance in two stages at once, and
the second one is usually not attributed to it.

**The organisational reason budgets get summed.** The arithmetic error in
{{eq:sum-of-tails-overprovisions}} survives because it solves a real problem that the
correct method does not. A per-stage p99 allocation gives each team an independent,
checkable target they can work against without coordinating with anyone. Absorbable
growth does not: it is a property of the whole pipeline, it changes when any other
stage changes, and it cannot be verified by a team looking only at its own service.
So the wrong method is locally actionable and the right one is not, which is why the
wrong one wins. The practical resolution is to compute absorbable growth centrally
and publish it as a per-stage number that gets refreshed on a schedule -- teams get
their independent target, and the target is derived from something true.

**Why p99 and not p999.** The choice of percentile changes the answers here, not just
their magnitude. Higher percentiles weight the tail more, which raises $z_p$ in
{{eq:narrowing-competes-with-shrinking}} and shifts the balance further toward
variance work. It also makes every measurement noisier: estimating a p999 from a
trace requires roughly ten times the samples of a p99 for the same confidence, and
the counterfactual simulations in {{sec:9-practical-example}} would need
correspondingly more trials. Teams that target p999 without increasing their sample
budget are ranking interventions on noise.

**Batching couples the stages.** {{cite:kwon2023pagedattention}}'s continuous batching
means a request's generation latency depends on what else is in the batch, which
violates the independence assumption in {{eq:sum-of-tails-overprovisions}} in the
direction that *reduces* the over-provisioning factor — stages become partially
comonotonic. The correction is modest at typical batch sizes and grows with
concurrency, which is another reason to measure the counterfactual rather than
compute it.

**Speculative decoding is a variance intervention.**
{{cite:leviathan2023speculative}} raises throughput without changing outputs, and its
effect on the latency distribution is to compress it — acceptance rates vary, but the
worst case is bounded by the target model's own speed. Under
{{eq:narrowing-competes-with-shrinking}} that makes it disproportionately valuable
against percentile targets relative to what its mean-throughput headline suggests.

**Why the counterfactual method works.** Re-running a pipeline with one stage's
distribution modified requires only the per-stage distributions, which tracing already
provides. It makes no independence assumption beyond what the simulation encodes, it
handles non-additive effects correctly, and it costs a few seconds of compute. The
reason it is rarely done is not difficulty; it is that latency work is usually
prosecuted from a flame graph, and a flame graph shows means.

## 8. Implementation

The first listing measures the gap between summed per-stage tails and a real system
p99, and computes what each stage can actually absorb.

```python {tier=A name=cc1}
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
```

## 9. Practical Example

A six-stage path with lognormal per-stage latency:

```
                 stage      mean       p50       p99   p99/mean
---------------------------------------------------------------
      gateway and auth      8.0m      7.2m     21.6m       2.70
             retrieval    120.0m     94.1m    474.9m       3.96
                rerank     40.0m     35.1m    116.2m       2.90
       prompt assembly     14.0m     12.9m     33.3m       2.38
            generation    520.0m    447.3m   1611.0m       3.10
   validate and format     18.0m     15.4m     57.3m       3.18
```

Three ways of adding those up:

```
                          quantity       value   vs true p99
------------------------------------------------------------
            sum of per-stage means      720.0m         0.39x
               measured system p50      650.5m         0.36x
              measured system mean      720.2m         0.39x
               MEASURED SYSTEM p99     1827.4m         1.00x
             sum of per-stage p99s     2314.2m         1.27x
```

Summing means gives **0.39×** the real p99 — too low, as expected. Summing p99s gives
**1.27×** — and that is the number a per-stage budget is implicitly built on
({{eq:sum-of-tails-overprovisions}}).

Now allocate a 2200ms target by p99 share:

```
                 stage   actual p99   budget share   allocated     slack
------------------------------------------------------------------------
      gateway and auth        21.6m           0.9%       20.6m     -1.1m
             retrieval       474.9m          20.5%      451.4m    -23.4m
                rerank       116.2m           5.0%      110.4m     -5.7m
       prompt assembly        33.3m           1.4%       31.7m     -1.6m
            generation      1611.0m          69.6%     1531.4m    -79.5m
   validate and format        57.3m           2.5%       54.4m     -2.8m
```

**Every single stage misses its allocation.** Six stages, six failures — and the
pipeline's true p99 is **1827ms** against the 2200ms target, passing with **373ms**
to spare.

That is the failure mode in one table: a per-stage percentile budget can report every
component out of compliance while the system it describes meets its target easily.

What the system can really absorb:

```
                 stage   current mean    max mean   growth allowed
--------------------------------------------------------------------
      gateway and auth           8.0m      286.5m            35.8x
             retrieval         120.0m      287.3m             2.4x
                rerank          40.0m      300.4m             7.5x
       prompt assembly          14.0m      293.9m            21.0x
            generation         520.0m      627.8m             1.2x
   validate and format          18.0m      276.7m            15.4x
```

The gateway's mean can grow **35.8×**; generation's can grow **1.2×**. Against a
p99-share budget that allocated the gateway 20.6ms and generation 1531ms, this is a
completely different picture ({{eq:absorbable-slack-is-not-budget-share}}) — the
budget gets the *order* right and the *magnitudes* badly wrong.

```mermaid {#fig:budget caption="Per-stage p99 budgets provision for a coincidence: all stages hitting their tails on one request. Real slack is measured by varying one stage against a fixed system target."}
flowchart TD
  A["2200ms system target"] --> B["divide by p99 share"]
  A --> C["measure absorbable growth"]
  B --> D["every stage over budget<br/>system passes with 373ms spare"]
  C --> E["gateway 35.8x<br/>generation 1.2x"]
  D --> F["engineering spent<br/>preventing a coincidence"]
  E --> G["engineering spent<br/>where slack is scarce"]
```

The second listing asks which stage to actually work on.

```python {tier=A name=cc2}
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
```

Attributing the same pipeline two ways:

```
                 stage   share of mean   share of variance   rank by mean   rank by var
---------------------------------------------------------------------------------------
      gateway and auth            1.1%                0.0%              6             6
             retrieval           16.7%               58.6%              2             1
                rerank            5.6%                0.5%              3             3
       prompt assembly            1.9%                0.0%              5             5
            generation           72.2%               40.7%              1             2
   validate and format            2.5%                0.1%              4             4
```

**The two rankings swap their top two entries**
({{eq:tail-attribution-differs-from-mean}}). Generation is **72.2%** of the mean and
**40.7%** of the variance; retrieval is **16.7%** of the mean and **58.6%** of the
variance.

Measuring the counterfactual — halve one stage's mean:

```
   stage halved (mean)    new p99   p99 saved   weeks   ms per week
-------------------------------------------------------------------
      gateway and auth    1731.8m        3.4m     2.0          1.71
             retrieval    1523.3m      211.9m     6.0         35.31
                rerank    1715.5m       19.7m     3.0          6.57
       prompt assembly    1727.6m        7.6m     1.0          7.62
            generation    1569.7m      165.5m    14.0         11.82
   validate and format    1726.1m        9.1m     2.0          4.53
```

Halve one stage's standard deviation instead:

```
     stage halved (sd)    new p99   p99 saved   weeks   ms per week
-------------------------------------------------------------------
      gateway and auth    1735.6m       -0.4m     1.0         -0.37
             retrieval    1455.0m      280.2m     3.0         93.39
                rerank    1734.4m        0.8m     2.0          0.41
       prompt assembly    1734.5m        0.7m     1.0          0.66
            generation    1634.2m      101.0m    10.0         10.10
   validate and format    1735.8m       -0.6m     1.0         -0.56
```

The obvious target — halving generation's mean, the largest number in the pipeline —
returns **11.82ms** of p99 per week. Halving retrieval's standard deviation returns
**93.39ms** per week: **7.9×** the return, for **more** absolute improvement
(**280ms** against **166ms**) and **less** work (**3** weeks against **14**). The
instinct does not merely pick a worse option; it picks one worse on every axis at
once.

But note that this is not a blanket preference for variance work. On generation
itself, halving the mean returns **11.82** and halving the variance returns **10.10**
— there, the conventional lever is better ({{eq:narrowing-competes-with-shrinking}}).
**The variance lever wins where the variance is**, which is a different stage from
where the time is.

## 10. Production Considerations

Never budget by summing per-stage percentiles. Budget the system target, and derive
per-stage constraints by measuring absorbable growth against it.

Publish absorbable growth rather than budget share. It is the number a team needs to
know whether a proposed change is affordable, and it is the number nobody currently
has.

Budget time-to-first-token separately from total latency wherever streaming is
deployed. They have different owners, different targets, and by
{{eq:streaming-capacity-is-set-by-ttft}} different consequences for capacity.

Record per-stage latency *distributions*, not just percentiles. Everything in this
chapter needs the distribution, and a p50/p99 pair is not enough to reconstruct one.

Rank latency work by p99 milliseconds per engineering week, measured. The ranking
takes an afternoon to produce once distributions are available and the spread between
best and worst option is over two orders of magnitude.

Give every stage an owner who knows its absorbable growth, and require that number
in any design review proposing to add work to that stage. The commonest way a latency
target is lost is not a regression in an existing stage but the addition of a new one,
and the question "how much can this path absorb?" has an answer that nobody looks up
because it has never been published.

Keep the counterfactual simulation in the repository, next to the tests. It is a few
dozen lines, it runs in seconds, and its value is that it can be re-run by whoever is
proposing a change rather than commissioned from whoever owns performance. A model
that lives in one person's notebook gets consulted once; one that lives in the build
gets consulted every time.

Treat "cap the occasional slow path" as a first-class latency intervention. It is a
variance reduction, it is usually cheap, and flame-graph-driven planning never
surfaces it.

Re-measure after every change. The rankings depend on the current distribution, and
the change you just shipped altered it.

## 11. Common Mistakes

**Summing per-stage p99s to get a system p99.** Over-provisions by the $\ell_1/\ell_2$
norm ratio of the stage standard deviations.

**Reporting stages as failing their budget when the system passes.** The budget is
wrong, not the stages.

**Reading budget share as slack.** They differ by more than an order of magnitude
here.

**Optimising the widest bar on the flame graph.** Flame graphs show means; percentile
targets ask about spread.

**Assuming variance work always beats mean work.** It is per-stage, and
{{eq:narrowing-competes-with-shrinking}} decides.

**Budgeting total latency on a streaming surface.** Constrains a quantity users do not
experience.

## 12. Failure Modes

**Budget-driven over-engineering.** Teams meet allocations the system never required,
and the effort is invisible as waste because everyone hit their number.

**Slack exhaustion in the wrong stage.** A change consumes the little headroom
generation had while the budget said it had the most, and the system p99 regresses
with no stage exceeding its allocation.

**Attribution drift.** The variance ranking changes as the corpus, prompt, or traffic
mix moves, and last quarter's optimisation plan is now pointed at the wrong stage.

**Percentile-only telemetry.** Distributions are discarded at collection, making every
analysis in this chapter impossible after the fact. This is the most expensive of the
failure modes listed here because it is irreversible: the data needed to diagnose the
others was thrown away months before anyone needed it, and recovering it requires
waiting out a fresh collection window.

**New-stage creep.** Each added stage contributes its own variance to the sum under
the square root, so a sequence of individually negligible additions moves the system
p99 by an amount no single change would have been challenged over.

**Coupled-stage surprise.** Batching or shared infrastructure makes stages
comonotonic, the independence-based estimates drift, and nobody notices because the
estimates were never validated against measurement.

## 13. Alternatives

**Budget only at the system boundary.** No per-stage allocation at all; teams
coordinate through a shared measurement. Correct, and it fails organisationally when
teams need an independent target to work against.

**Comonotonic budgeting.** Assume stages are perfectly correlated and budget on the
sum of p99s deliberately, as a conservative bound. Defensible for safety-critical
paths where the coincidence must be survivable; wasteful everywhere else.

**Latency SLOs on the mean.** Cheap, composable, and measures something users do not
directly experience. Reasonable as a secondary indicator.

**Deadline propagation instead of budgets.** Pass a remaining-time budget with the
request and let each stage adapt — degrade retrieval breadth, cap generation length.
Replaces static allocation with dynamic, and handles the variance problem by
construction rather than by arithmetic.

**Speculative decoding and other throughput work.**
{{cite:leviathan2023speculative}} compresses the generation distribution; under
{{eq:narrowing-competes-with-shrinking}} its percentile value exceeds its
mean-throughput headline.

## 14. Evaluation

Report the over-provisioning factor — sum of per-stage p99s over measured system p99 —
as a standing metric. It tells you how wrong a per-stage budget would be, and it
changes as the pipeline does.

Measure absorbable growth per stage quarterly. It is the input to every capacity and
roadmap conversation about latency.

Validate independence by comparing the measured system p99 against the
quadrature estimate. A large discrepancy means stages are coupled, and the coupling is
worth understanding before any of this analysis is trusted.

Report the per-week efficiency ranking alongside any latency roadmap, and require a
justification when the plan does not follow it.

Track TTFT and total latency as separate SLOs where streaming is deployed, and report
the crossover concurrency from {{ch:sd-async}} beside them.

## 15. Advanced Concepts

The independence assumption is the load-bearing one, and it fails in both directions.
Batching makes generation latencies positively correlated across concurrent requests
but not across stages of one request; shared infrastructure — a saturated network, a
throttled dependency — makes stages of one request positively correlated, moving the
system toward the comonotonic case where summing p99s is actually right. **The
over-provisioning factor is therefore load-dependent**, largest when the system is
healthy and smallest during the incidents when the budget matters most. A budget
validated at low load is more wrong than it looks.

{{eq:fanout-amplifies-the-tail}} from {{ch:sd-retrieval-agents}} is the same question
for parallel composition, and the two compose in an unpleasant way. A sequential
pipeline sums latencies, so tails average out; a parallel fan-out takes the maximum,
so tails amplify. A pipeline containing a fan-out stage has a stage whose own
distribution is already a maximum-of-samples, which is heavy-tailed by construction —
and it will dominate the variance attribution even when its mean is unremarkable.
**Fan-out stages are where the tail lives**, and the two chapters' results identify
the same target from opposite directions.

A related subtlety concerns what "halving a stage" means when the stage is itself a
mixture. Retrieval in the worked pipeline has high variance because it is really two
populations -- cache hits at a few milliseconds and cache misses at hundreds -- and
its lognormal fit is a smooth approximation to something bimodal. Interventions act on
those populations differently: raising the hit rate moves mass between modes, while
speeding up the miss path moves one mode. The first is a variance reduction and the
second is closer to a mean reduction, and modelling the stage as a single lognormal
cannot distinguish them. Where a stage is known to be bimodal, fitting the mixture and
intervening on its components separately gives a materially better estimate, and the
components are usually already distinguishable in tracing by a flag the code sets.

The counterfactual method assumes an intervention scales a distribution
multiplicatively. Real interventions do not always: adding a cache creates a mixture
distribution with a fast mode and the original mode, which is a different shape
entirely and generally better for p99 than a proportional narrowing of the same mean
reduction. So the measured table understates cache-like interventions, and modelling
them as mixtures rather than scalings is worth the extra few lines.

## 16. Connection to Previous Chapters

{{eq:variance-not-mean-drives-wait}} from {{ch:sd-async}} said the second moment
drives waiting. {{eq:tail-attribution-differs-from-mean}} says it also drives where to
spend engineering effort, which is the same fact turned into a roadmap.

{{eq:fanout-amplifies-the-tail}} from {{ch:sd-retrieval-agents}} identifies the stages
this chapter's attribution will rank first, from the other direction.

{{eq:streaming-capacity-is-set-by-ttft}} determines which latency a budget should
constrain.

{{eq:three-properties-break-the-stack}} is why this part exists: conventional
pipelines have tighter distributions, so the errors this chapter documents are small
enough there to ignore. They are not small here.

## 17. Exercises

1. Compute the $\ell_1/\ell_2$ over-provisioning factor for a pipeline of eight stages
   with equal standard deviations. Compare to the listing's 1.27×.

2. Modify the first listing so two stages are positively correlated with coefficient
   0.7. How does the over-provisioning factor change?

3. Derive the condition in {{eq:narrowing-competes-with-shrinking}} for $z_p$
   corresponding to p999 rather than p99. Does narrowing become more or less
   attractive?

4. Model a cache as a mixture rather than a scaling in the second listing. How much
   does the measured efficiency of retrieval work change?

5. Take a real trace, fit per-stage distributions, and produce the per-week efficiency
   ranking for your own system. Does it match the current roadmap?

## 18. Interview Questions

1. Every service in our pipeline reports missing its latency budget and the end-to-end
   p99 is fine. Explain.

2. Why does summing per-stage p99s overstate the system p99, and by roughly how much?

3. Our slowest stage is 72% of the mean latency. Should we optimise it?

4. When does making a stage steadier beat making it faster?

5. You have distributions for every stage and one engineer for a quarter. How do you
   decide what they work on?

6. A team proposes adding a seventh stage with a 30ms mean. What do you need to know
   before approving it, and where would that number come from?

## 19. Research Questions

1. How correlated are pipeline stages in real systems, and how does the
   over-provisioning factor vary with load?

2. Can absorbable growth be estimated online from production traces without a
   simulation, and with what error?

3. What is the right budgeting primitive for streaming surfaces where TTFT and total
   latency have different owners and different crossover behaviour?

4. Does deadline propagation outperform static budgeting on realistic workloads, and
   what does it cost in complexity and in degraded-mode quality?

## 20. Chapter Summary

Percentiles do not add. The sum of per-stage p99s exceeds the system p99 by the
$\ell_1/\ell_2$ norm ratio of the stage standard deviations
({{eq:sum-of-tails-overprovisions}}) — **1.27×** in the worked pipeline, where
per-stage p99s sum to **2314ms** against a true **1827ms**.

Budgeting by p99 share therefore provisions for a coincidence. Under a 2200ms target,
**all six stages miss their allocation** while the pipeline passes with **373ms** to
spare. And budget share is not slack: the gateway can absorb **35.8×** its mean and
generation only **1.2×**, against allocations of 20.6ms and 1531ms respectively
({{eq:absorbable-slack-is-not-budget-share}}).

A stage contributes to the mean through its mean and to the tail through its variance,
and those rankings differ — generation is **72.2%** of the mean and **40.7%** of the
variance; retrieval is **16.7%** and **58.6%**
({{eq:tail-attribution-differs-from-mean}}).

Measured, the obvious target returns **11.82ms** of p99 per engineering week and the
best available returns **93.39ms** — **7.9×**, with more absolute improvement for less
work. But narrowing is not universally better than shrinking: on generation itself the
mean lever wins, **11.82** against **10.10**
({{eq:narrowing-competes-with-shrinking}}).

Both results come from the same source, and it is the one this whole part has been
circling. A model-backed pipeline has wide, right-skewed per-stage distributions, and
almost every convenient piece of latency practice -- adding percentiles, reading a
flame graph, budgeting by share -- is an approximation that is accurate when
distributions are tight and wrong when they are not. None of those practices is
foolish. They were correct for the systems they were developed on.

Carry forward: **budget the system, not the stages**, and **rank latency work by
measured milliseconds per week, not by the widest bar**.

## 21. Further Reading

- {{cite:pope2022inference}} — prefill versus decode; why input length drives TTFT
  variance.
- {{cite:kwon2023pagedattention}} — continuous batching, which couples stages and
  moves the over-provisioning factor.
- {{cite:leviathan2023speculative}} — speculative decoding as a variance intervention
  rather than a throughput one.

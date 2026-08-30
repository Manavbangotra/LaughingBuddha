---
id: ops-governance
number: 211
part: XXIV
tier: full
status: draft
requires: [count-limits-cannot-bound-cost, cost-limits-need-a-reservation,
           rework-cost-is-set-by-detection-lateness, tail-attribution-differs-from-mean]
provides: [budget-overrun-is-set-by-feedback-delay, attribution-precedes-control,
           agent-cost-is-heavy-tailed, per-request-cap-beats-aggregate-budget]
citations: [chen2023frugalgpt, sculley2015, breck2017, paleyes2020deployment]
---

## 1. Learning Objectives

By the end of this chapter you will be able to compute the overrun a cost control permits
from its feedback delay rather than from its limit; explain why detection speed and spend
attribution are sequential rather than competing investments; show that agent per-request
cost is heavy-tailed and that its mean is a tail statistic; explain why an aggregate budget
moves when nobody made a cost decision; and price a per-request cost cap in dollars per
successful request forgone.

## 2. Why This Matters

A cost control is a feedback loop, and the loop most teams have runs on billing data. At
$1,305 an hour of excess burn, a cloud billing export that lands 26 hours later — plus
three hours to react — permits **$37,845 of overrun, 21% of the monthly budget, inside one
incident** ({{eq:budget-overrun-is-set-by-feedback-delay}}). A self-metered token counter
closes the same loop in 1.1 hours for **$1,436**, and it is arithmetic on numbers the
request handler already computed.

Then the second half arrives. Detecting the excess does not tell you what to throttle.
Without attribution the only available control is a global throttle, which blocks **40,530
healthy requests a day** to remove the 3.5% causing the problem. Fixing detection first
takes an incident from $44,127 to $7,718 — of which **81% is now collateral rather than
spend** ({{eq:attribution-precedes-control}}). The two are not competing priorities; closing
the detection loop promotes attribution to the binding constraint.

The chapter's other half is why the budget itself is the wrong object. Agent per-request
cost is heavy-tailed: **the median request costs $0.0238 and the mean $0.0650**, with the
mean sitting at the **84th percentile** ({{eq:agent-cost-is-heavy-tailed}}). The top 1% of
requests are 29% of spend. So the mean is a tail statistic, and a slower dependency, a
prompt edit, and a harder customer segment together move monthly spend **51%** with no cost
decision taken by anyone.

A per-request cost cap at 8× the mean truncates **1.95%** of requests, removes **24.6%** of
spend, and loses **0.39%** of successes — because the expensive tail is disproportionately
the runs that were not going to work ({{eq:per-request-cap-beats-aggregate-budget}}).

## 3. Prerequisites

{{eq:count-limits-cannot-bound-cost}} from {{ch:sd-apis-auth}} is the result this chapter
extends from the API boundary to the organisation: a request count does not bound spend when
the per-request cost varies by two orders of magnitude, and {{sec:9-practical-example}}
shows exactly how much it varies.

{{eq:cost-limits-need-a-reservation}} from the same chapter is the terminal control here —
the only mechanism that acts before the money is spent rather than after.

{{eq:rework-cost-is-set-by-detection-lateness}} from {{ch:ops-lifecycle}} is the same
structure in a different currency. There, lateness priced defects; here it prices dollars,
and the arithmetic is identical.

{{eq:tail-attribution-differs-from-mean}} from {{ch:sd-latency}} matters because
the cost tail and the latency tail turn out to be the same requests, which makes one control
serve both budgets.

{{cite:chen2023frugalgpt}} is the cost-reduction backdrop; this chapter is about bounding
cost rather than reducing it, which is a different and more neglected problem.

## 4. Intuitive Explanation

Start with what a budget actually is, mechanically. It is a number, a measurement of spend
against that number, and an action taken when the measurement crosses it. Three parts, and
only the first one is usually written down.

The measurement is where the trouble is. Most organisations read AI spend from the place it
is easiest to read: the provider's billing dashboard, or the cloud billing export that lands
in a warehouse table overnight. Both are correct and both are slow. A billing export is
about a day behind.

Now put a runaway on top of that. A retry storm, an agent that starts looping, a prompt
change that tripled the context length — burn goes to six times normal. That is $1,305 an
hour of excess. By the time the billing export shows it, and someone reads it, and someone
decides to act, twenty-nine hours have passed and thirty-eight thousand dollars have gone.

Notice what did the damage. Not the size of the budget — the budget was never consulted.
The overrun is the burn rate times the loop delay, and the limit appears nowhere in that
product. **A limit read from a slow channel does not bound anything; it describes what
happened.**

The fix is embarrassingly cheap. Your request handler knows how many tokens it sent and
received, because it had to count them to build the prompt and to parse the response. It
knows the model's price. Multiply and accumulate. That is a running spend total with a
five-minute lag instead of a twenty-six-hour one, and it costs an afternoon.

So you fix detection, and then the next incident teaches you the second lesson.

You now know within five minutes that spend has gone to six times normal. What do you do?
The only lever available is to throttle, and throttle *what*? You know the total went up.
You do not know which feature, which team, which customer, or which code path is
responsible, because the request handler recorded the cost and not who incurred it.

So you throttle everything. Forty-two thousand requests a day stopped to remove the fifteen
hundred that are the problem.

That collateral is now the dominant term. After fixing detection, four-fifths of the
incident's cost is healthy traffic you blocked, not money you spent. Which is the ordering
result: **detection and attribution are sequential, not alternatives.** Fixing detection
alone buys you a faster way to block all of your traffic, and the team correctly concludes
after the next incident that the controls are too blunt — that conclusion is the second half
of the same project, not a new one.

Now the deeper problem, which is that the budget was aimed at the wrong statistic.

An agent decides its own length. It runs until it thinks it is done, which makes step count
a stopping-time distribution — mostly short, occasionally very long. And cost grows faster
than steps, because step twelve carries eleven steps of context into the prompt while step
one carried none.

The result is a cost distribution with a long right tail. The median request costs about two
and a half cents. The mean costs six and a half. The 99.9th percentile costs nearly three
dollars — forty-five times the mean.

The mean sits at the eighty-fourth percentile. Say that again slowly: **the average request
is more expensive than five-sixths of requests.** Every plan, forecast, and capacity model
phrased in average cost per request is a plan about a request that mostly does not happen.

And because the mean is a tail statistic, it moves for reasons that have nothing to do with
cost. A dependency gets slower, so retries go up, so more runs enter the long mode. Somebody
adds one example to the prompt, which lengthens context and makes continuing marginally more
attractive to the model. A harder customer segment signs up. Any of these moves the monthly
bill by fifteen to twenty percent; all three together move it by half.

None of them was a cost decision. None of them went through anything that could have
predicted a cost effect. The prompt edit in particular passed no gate at all, which
{{ch:ops-prompt-versioning}} established for entirely separate reasons.

Which brings the chapter to its actual recommendation. Stop trying to govern the aggregate
and govern the unit. Cap what a single request may cost.

The numbers on this are better than they sound. Capping at eight times the mean stops about
two percent of requests. It removes a quarter of spend — because that two percent is a
quarter of the money. And it costs you four-tenths of a percent of successful requests,
because the runs that get truncated are disproportionately the stuck ones, which were going
to fail anyway.

You are, in effect, paying about four dollars per marginal success by not capping. That is a
number a design review can argue about, which is more than can be said for "the budget feels
tight."

And the cap is free in a second currency. Steps cost time as well as money, so a cost cap is
a run-length cap is a latency cap. The tail that {{ch:sd-latency}} spent a chapter
attacking is the same tail. One control, two budgets, usually two teams and two dashboards.

## 5. Formal Explanation

**The delay result.** Let $b$ be the excess burn rate in dollars per hour during an incident,
$\ell$ the reporting lag of the detection channel, and $r$ the human reaction time. The
overrun before any control can act is $b(\ell + r)$. The budget limit $B$ does not appear.
This is a control-theoretic statement rather than a financial one: a loop whose feedback
delay exceeds the disturbance's timescale does not regulate.

**The attribution result.** Suppose a share $\rho$ of traffic is responsible for the excess,
and the finest available attribution partitions traffic into classes of which the smallest
containing the culprit has share $\alpha \geq \rho$. Throttling that class stops
$V(\alpha - \rho)$ healthy requests, where $V$ is volume. Total incident cost is
$b(\ell + r) + V(\alpha - \rho) \tau v$, with $\tau$ the throttle duration and $v$ the value
of a blocked request. The two terms are additive and independently controlled, but their
*relative* size is not fixed: reducing $\ell$ leaves the second term untouched, so it
becomes dominant. Attribution is therefore not an alternative investment but the successor
to detection.

**The distribution result.** Let $N$ be the number of steps a run takes, a stopping time.
Model continuation as a two-component mixture: with probability $1 - \pi$ the run is
"normal" and continues with probability $q_n$ at each step; with probability $\pi$ it is
"stuck" and continues with $q_s > q_n$. Cost per step grows with the step index because
context accumulates: step $i$ costs $c_0(1 + \gamma i)$, so a run of $n$ steps costs
$c_0 \sum_{i<n} (1 + \gamma i)$, quadratic in $n$. A geometric tail composed with a quadratic
cost function produces a distribution whose mean is dominated by its upper percentiles.

**The cap result.** Truncating at $K$ leaves expected cost $\mathbb{E}[\min(C, K)]$. Because
the tail is disproportionately the stuck component, and the stuck component has a much lower
success probability, the ratio of spend removed to successes lost is large. The decision
variable is the implied price per success forgone, which can be compared against the value of
a request directly.

## 6. Mathematical Foundation

Overrun as a function of the loop, not the limit:

$$O = b\,(\ell + r), \qquad \frac{\partial O}{\partial B} = 0$$ (eq:budget-overrun-is-set-by-feedback-delay)

With $b = 1{,}305$/hour: a 26-hour export plus 3 hours of reaction gives $O = 37{,}845$; a
5-minute counter plus 1 hour gives $O = 1{,}436$.

Total incident cost, and the composition shift that orders the two fixes:

$$L(\ell, \alpha) = \underbrace{b(\ell + r)}_{\text{overrun}} + \underbrace{V(\alpha - \rho)\,\tau\,v}_{\text{collateral}}, \qquad \lim_{\ell \to 0} \frac{\text{collateral}}{L} \to 1$$ (eq:attribution-precedes-control)

At $\ell = 26$ the collateral is 14% of loss; at $\ell = 0.1$ it is 81%. Same $\alpha$,
different binding constraint.

The cost distribution of a self-terminating run:

$$C(n) = c_0 \sum_{i=0}^{n-1} (1 + \gamma i), \qquad \Pr[N = n] = (1-\pi) q_n^{\,n-1}(1 - q_n) + \pi\, q_s^{\,n-1}(1 - q_s)$$ (eq:agent-cost-is-heavy-tailed)

giving $\text{median} = \$0.0238$, $\mathbb{E}[C] = \$0.0650$, $F(\mathbb{E}[C]) = 0.84$, and
$C_{99.9} / \mathbb{E}[C] = 45$.

And the cap, evaluated in the currency that matters:

$$\text{price per success} = \frac{\mathbb{E}[C] - \mathbb{E}[\min(C, K)]}{\Pr[C > K]\;\mathbb{E}[S \mid C > K]}$$ (eq:per-request-cap-beats-aggregate-budget)

At $K = 8\,\mathbb{E}[C]$: 24.6% of spend removed, 0.39% of successes lost, **$4.10 per
success forgone**.

## 7. Internal Mechanics

Why is the billing channel the default? Because it is the only channel that is
*authoritative*. The provider's invoice is what you actually pay, and a self-metered counter
is an estimate that can drift from it — cached prompt tokens are priced differently,
batch discounts apply, a model version changes its rate. Teams reach for the authoritative
number because reconciling two numbers is work.

The resolution is that these are different instruments for different jobs. The invoice is
for accounting and must be exact. The control loop is for stopping a runaway and needs to be
fast and approximately right. A counter that is 4% off and 26 hours early is a strictly
better control than one that is exact and late, and the mistake is using one instrument for
both jobs — which is {{cite:sculley2015}}'s configuration-debt pattern showing up in the
finance function.

Attribution is missing for a related reason. Cost is incurred at the model call, which
happens deep inside a request handler, several frames below whatever knew the feature name
and the customer tier. Propagating that context down means threading it through the call
stack or putting it in a context-local, and both look like plumbing at the moment the code
is written. Nobody omits attribution deliberately; it is omitted the way error context is
omitted, one frame at a time.

The heavy tail has a mechanical origin worth being precise about, because it determines
which interventions work. Two things compound. First, the run length is a stopping time, so
its distribution has a geometric-like tail rather than a bounded one. Second, cost per step
*increases* with the step index, because each step's prompt carries the accumulated
transcript. Either alone gives a manageable distribution. Together they give a quadratic cost
function composed with a geometric tail, which is where the 45× ratio comes from.

That composition also explains why a step limit is a weak control. A limit at 60 steps does
not remove the tail — it relocates it. Runs that would have gone further are stopped *at* 60,
which creates a point mass of maximum-cost runs at exactly the limit, each of which spent the
entire step budget before being cut off. The 99.9th and 99.99th percentiles in
{{sec:9-practical-example}} are both 60 steps for this reason. The step limit is a wall, and
the tail piles up against it.

Finally, the reason a cap's trade-off is so favourable: the stuck mode and the expensive mode
are the same mode. A run is long because it is not converging, and a run that is not
converging is unlikely to converge later. Truncation is therefore *selective* in the
direction you want, which is unusual for a blunt limit and is the reason this control is
worth more than it looks.

## 8. Implementation

The first listing measures the feedback loop and the attribution it enables.

```python {tier=A name=budget-overrun-is-set-by-feedback-delay}
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
```

## 9. Practical Example

At $1,305 an hour of excess burn, what each detection channel permits:

```
           detection channel   lag (h)   reaction (h)   total (h)     overrun   % of budget
-------------------------------------------------------------------------------------------
        cloud billing export      26.0            3.0        29.0      37,845         21.0%
          provider dashboard       4.0            2.0         6.0       7,830          4.4%
        nightly usage rollup      14.0            3.0        17.0      22,185         12.3%
  self-metered token counter       0.1            1.0         1.1       1,436          0.8%
       in-request accounting       0.0            0.5         0.5         653          0.4%
```

The billing export permits **$37,845 — 21% of the monthly budget — inside one incident**,
and the limit appears nowhere in that number
({{eq:budget-overrun-is-set-by-feedback-delay}}). A token counter closes the loop in **1.1
hours against 29**.

```
         attribution granularity   traffic throttled   collateral req/day   effort
----------------------------------------------------------------------------------
      nothing -- global throttle              100.0%               40,530      0.0
                      by service               34.0%               12,810      0.5
                         by team               19.0%                6,510      1.0
                      by feature                8.0%                1,890      2.0
    by feature and customer tier                4.2%                  294      3.5
```

The runaway is 3.5% of traffic. With no attribution, **40,530 healthy requests a day are
blocked** to remove it.

```
                   detection                     attribution    overrun   collateral      total
-----------------------------------------------------------------------------------------------
        cloud billing export      nothing -- global throttle     37,845        6,282     44,127
        cloud billing export    by feature and customer tier     37,845           46     37,891
  self-metered token counter      nothing -- global throttle      1,436        6,282      7,718
  self-metered token counter    by feature and customer tier      1,436           46      1,481
       in-request accounting    by feature and customer tier        653           46        698
```

Worst pairing **$44,127**, best **$698** — **63×**, from two changes of a week each.

```
                                      change   incident cost      annual    saved/yr
------------------------------------------------------------------------------------
   baseline: billing export, global throttle          44,127     308,890           0
          fix detection only (token counter)           7,718      54,024     254,867
       fix attribution only (feature + tier)          37,891     265,234      43,656
                                    fix both           1,481      10,367     298,523
```

Detection first, decisively — a precise throttle applied 29 hours late is still 29 hours
late. But after fixing detection, of the $7,718 remaining, **$6,282 (81%) is collateral
rather than spend** ({{eq:attribution-precedes-control}}). The fix promotes attribution to
the binding constraint.

The second listing measures the distribution the budget was aimed at.

```python {tier=A name=agent-cost-is-heavy-tailed}
"""The mean cost of an agent request is a tail statistic, so budgeting on it is unstable.

An agent runs until it decides to stop. That makes step count a stopping-time
distribution with a long right tail, and cost grows faster than steps because the context
carried into step i grows with i.

The result is a per-request cost distribution where the mean sits far above the median and
is dominated by a small share of runs (eq:agent-cost-is-heavy-tailed).

This listing computes that distribution exactly, shows why an aggregate budget built on
the mean moves when nobody changed anything, and finds the control that does bound it
(eq:per-request-cap-beats-aggregate-budget).
"""
MAX_STEPS = 60
P_STUCK = 0.08                # share of runs that enter a non-terminating pattern
CONT_NORMAL = 0.70            # P(take another step | normal)
CONT_STUCK = 0.94             # P(take another step | stuck)
COST_STEP_0 = 0.0065          # dollars for the first step
CTX_GROWTH = 0.22             # each step carries more context than the last
SUCC_NORMAL = 0.94
SUCC_STUCK = 0.19
REQUESTS_MONTH = 1_260_000.0
LAT_STEP_S = 2.35


def cum_cost(n):
    """Cost of a run that took n steps, with context growing each step."""
    return COST_STEP_0 * sum(1.0 + CTX_GROWTH * i for i in range(n))


def pmf(cont):
    """P(run takes exactly n steps), truncated at MAX_STEPS."""
    out = {}
    for n in range(1, MAX_STEPS + 1):
        p = (cont ** (n - 1)) * (1.0 - cont)
        out[n] = p
    tail = 1.0 - sum(out.values())
    out[MAX_STEPS] += tail       # anything longer is stopped by the framework
    return out


norm, stuck = pmf(CONT_NORMAL), pmf(CONT_STUCK)
JOINT = []
for n in range(1, MAX_STEPS + 1):
    JOINT.append((n, (1 - P_STUCK) * norm[n], SUCC_NORMAL))
    JOINT.append((n, P_STUCK * stuck[n], SUCC_STUCK))

rows = sorted(((cum_cost(n), p, n, s) for n, p, s in JOINT if p > 0))
mean_cost = sum(c * p for c, p, n, s in rows)
mean_steps = sum(n * p for c, p, n, s in rows)

print(f"Agent runs stop when they decide to. Mean {mean_steps:.1f} steps,")
print(f"mean cost ${mean_cost:.4f} a request, {REQUESTS_MONTH:,.0f} requests a month.")
print()
print("The per-request cost distribution.")
print()
print(f"{'percentile':>12}{'steps':>9}{'cost':>11}{'x mean':>10}{'x median':>11}")
print("-" * 53)


def at(q):
    acc = 0.0
    for c, p, n, s in rows:
        acc += p
        if acc >= q:
            return c, n
    return rows[-1][0], rows[-1][2]


median = at(0.50)[0]
pct = {}
for q in (0.50, 0.75, 0.90, 0.99, 0.999, 0.9999):
    c, n = at(q)
    pct[q] = (c, n)
    print(f"{q:>12.2%}{n:>9}{c:>11.4f}{c / mean_cost:>10.1f}{c / median:>11.1f}")

print()
print(f"mean ${mean_cost:.4f} sits at the {sum(p for c, p, n, s in rows if c <= mean_cost):.0%} "
      f"percentile -- above most requests.")

print()
print()
print("Where the money goes: share of spend by share of requests.")
print()
print(f"{'top share of requests':>23}{'share of spend':>17}{'concentration':>16}")
print("-" * 56)
desc = sorted(rows, key=lambda r: -r[0])
conc = {}
for frac in (0.001, 0.01, 0.05, 0.10, 0.25):
    acc_p, acc_c = 0.0, 0.0
    for c, p, n, s in desc:
        take = min(p, frac - acc_p)
        if take <= 0:
            break
        acc_c += c * take
        acc_p += take
    conc[frac] = acc_c / mean_cost
    print(f"{frac:>23.1%}{acc_c / mean_cost:>17.1%}{(acc_c / mean_cost) / frac:>15.1f}x")

print()
print()
print("Why an aggregate budget built on this mean is unstable: the mean is set")
print("by the tail, and the tail moves for reasons nobody is watching.")
print()
print(f"{'change':>38}{'stuck rate':>13}{'cont|stuck':>13}"
      f"{'mean cost':>12}{'monthly':>13}{'vs base':>10}")
print("-" * 99)


def mean_for(p_stuck, cont_stuck, cont_norm=CONT_NORMAL):
    st, nm = pmf(cont_stuck), pmf(cont_norm)
    m = 0.0
    for n in range(1, MAX_STEPS + 1):
        m += cum_cost(n) * ((1 - p_stuck) * nm[n] + p_stuck * st[n])
    return m


SHIFTS = [
    ("baseline",                          P_STUCK, CONT_STUCK, CONT_NORMAL),
    ("a tool gets slower, more retries",  0.11,    CONT_STUCK, CONT_NORMAL),
    ("a prompt change adds one example",  P_STUCK, 0.95,       CONT_NORMAL),
    ("a harder customer segment",         P_STUCK, CONT_STUCK, 0.74),
    ("all three",                         0.11,    0.95,       0.74),
]
shift = {}
for label, ps, cs, cn in SHIFTS:
    m = mean_for(ps, cs, cn)
    shift[label] = (m, m * REQUESTS_MONTH)
    print(f"{label:>38}{ps:>13.0%}{cs:>13.2f}{m:>12.4f}"
          f"{m * REQUESTS_MONTH:>13,.0f}{m / mean_cost:>9.2f}x")

print()
print()
print("The control that does bound it: a cost cap on the individual request.")
print()
print(f"{'cap':>16}{'cap $':>10}{'requests capped':>18}{'spend removed':>16}"
      f"{'successes lost':>17}{'success cost':>15}")
print("-" * 92)
base_succ = sum(p * s for c, p, n, s in rows)
caps = {}
for k in (30, 20, 12, 8, 5, 3):
    cap_c = k * mean_cost
    capped_p = sum(p for c, p, n, s in rows if c > cap_c)
    spend_after = sum(min(c, cap_c) * p for c, p, n, s in rows)
    succ_lost = sum(p * s for c, p, n, s in rows if c > cap_c)
    caps[k] = (cap_c, capped_p, 1 - spend_after / mean_cost, succ_lost)
    per_succ = ((mean_cost - spend_after) * REQUESTS_MONTH) / max(
        succ_lost * REQUESTS_MONTH, 1e-9)
    print(f"{k:>14}x{cap_c:>10.4f}{capped_p:>18.3%}"
          f"{1 - spend_after / mean_cost:>16.1%}{succ_lost:>17.3%}"
          f"{per_succ:>15,.0f}")

print()
print(f"baseline success rate: {base_succ:.1%}")

print()
print()
print("The same cap is a latency control, because steps cost time as well as")
print(f"money ({LAT_STEP_S:.2f}s a step).")
print()
print(f"{'cap':>16}{'max steps':>12}{'max latency':>14}{'p99.9 after':>14}")
print("-" * 56)
for k in (30, 20, 12, 8, 5, 3):
    cap_c = k * mean_cost
    max_n = max((n for c, p, n, s in rows if c <= cap_c), default=1)
    p999_n = min(pct[0.999][1], max_n)
    print(f"{k:>14}x{max_n:>12}{max_n * LAT_STEP_S:>13.0f}s"
          f"{p999_n * LAT_STEP_S:>13.0f}s")

print()
print()
print("Aggregate budget against per-request cap, as controls.")
print()
print(f"{'control':>26}{'bounds the mean?':>19}{'bounds a request?':>20}"
      f"{'acts before spend?':>21}")
print("-" * 86)
for name, a, b, c in (
        ("monthly budget",        "no -- reports it", "no",  "no"),
        ("daily spend cap",       "no",               "no",  "no"),
        ("step limit",            "partly",           "yes", "yes"),
        ("per-request cost cap",  "yes",              "yes", "yes"),
):
    print(f"{name:>26}{a:>19}{b:>20}{c:>21}")

print(f"""
The distribution table is the shape everything else follows from. The median request
costs ${median:.4f} and the mean is ${mean_cost:.4f} --
**{mean_cost / median:.1f} times the median** -- while the 99.9th percentile is
${pct[0.999][0]:.4f}, or {pct[0.999][0] / mean_cost:.0f} times the mean
(eq:agent-cost-is-heavy-tailed).

The mean sits at the {sum(p for c, p, n, s in rows if c <= mean_cost):.0%} percentile, which
is the compact statement of the problem: **the average request is more expensive than most
requests**, so any plan phrased in averages is a plan about a request that rarely happens.

The concentration table says where the money is. The most expensive
{0.01:.0%} of requests are {conc[0.01]:.0%} of spend --
{conc[0.01] / 0.01:.0f} times their share -- and the top {0.10:.0%} are
{conc[0.10]:.0%}.

That is not an anomaly to be cleaned up. It is what a stopping-time distribution looks
like when cost grows with the step index, and it will be true of any agent that decides
its own length.

The stability table is the part that makes aggregate budgeting fail in practice rather
than merely in principle. Nobody has to do anything wrong for the monthly bill to move.
A tool that gets slower and triggers more retries takes the stuck rate from
{P_STUCK:.0%} to {0.11:.0%} and the monthly spend from
{shift['baseline'][1]:,.0f} to {shift['a tool gets slower, more retries'][1]:,.0f}. A
prompt change that adds one example -- which lengthens context and makes continuation
marginally more attractive -- takes it to
{shift['a prompt change adds one example'][1]:,.0f}. All three together,
{shift['all three'][1]:,.0f}, or {shift['all three'][0] / mean_cost:.2f} times baseline.

**None of the three was a cost change.** One was a dependency's latency, one was a prompt
edit -- and ch:ops-prompt-versioning established that a prompt edit passes no gate at all
-- and one was who showed up. The budget moved
{shift['all three'][0] / mean_cost - 1:.0%} and nothing in the cost control system has an
opinion about any of these events, because none of them is filed as a cost event.

The cap table is the control that does work, and its numbers are better than they have
any right to be. A cap at {8}x the mean -- ${8 * mean_cost:.4f} -- truncates
{caps[8][1]:.2%} of requests, removes {caps[8][2]:.1%} of spend, and loses
{caps[8][3]:.2%} of successful requests
(eq:per-request-cap-beats-aggregate-budget).

The reason the trade is so favourable is in the model rather than in the arithmetic: the
expensive tail is disproportionately the stuck mode, which succeeds
{SUCC_STUCK:.0%} of the time against {SUCC_NORMAL:.0%} for normal runs. **The requests you
truncate are mostly the ones that were not going to work**, which is why capping is not
the blunt instrument it sounds like.

Note also what the {0.999:.1%} and {0.9999:.2%} percentiles have in common: both are
{pct[0.999][1]} steps, the framework's own step limit. The limit does not remove the tail,
it **piles it up at the boundary** -- a point mass of maximum-cost runs, every one of
which spent the full budget before being stopped.

The implied price is the number to put in a design review: at an {8}x cap you are paying
${((mean_cost - sum(min(c, 8 * mean_cost) * p for c, p, n, s in rows)) * REQUESTS_MONTH) / max(caps[8][3] * REQUESTS_MONTH, 1e-9):,.2f}
per additional success by *not* capping -- against the
$0.62 a blocked request was worth in the previous listing. Somebody should have to say
both numbers out loud before deciding the cap is too aggressive.

The latency table is the free consequence. The same cap bounds run length, so it bounds
the tail latency that ch:sd-latency spent a chapter attacking -- an
{8}x cost cap holds the run to {max((n for c, p, n, s in rows if c <= 8 * mean_cost), default=1)} steps and
{max((n for c, p, n, s in rows if c <= 8 * mean_cost), default=1) * LAT_STEP_S:.0f} seconds.

One control, two budgets, and they were being managed by separate teams with separate
dashboards.

The last table is the summary a governance document should contain. A monthly budget
bounds nothing and reports afterwards. A step limit bounds a request but only partly
bounds the mean, because ch:sd-apis-auth's result applies -- a count is not a cost. Only a
per-request cost cap bounds both, acts before the money is spent, and needs nothing that
the request handler does not already know.""")
```

```
  percentile    steps       cost    x mean   x median
-----------------------------------------------------
      50.00%        3     0.0238       0.4        1.0
      75.00%        5     0.0468       0.7        2.0
      90.00%        9     0.1100       1.7        4.6
      99.00%       34     1.0232      15.7       43.0
      99.90%       60     2.9211      45.0      122.8
      99.99%       60     2.9211      45.0      122.8
```

Median **$0.0238**, mean **$0.0650**, p99.9 **45× the mean**
({{eq:agent-cost-is-heavy-tailed}}). The mean sits at the **84th percentile** — the average
request is more expensive than five-sixths of requests. Both top percentiles are 60 steps,
the framework's own limit: **the limit relocates the tail rather than removing it**.

```
  top share of requests   share of spend   concentration
--------------------------------------------------------
                   0.1%             4.5%           45.0x
                   1.0%            29.4%           29.4x
                   5.0%            54.2%           10.8x
                  10.0%            64.6%            6.5x
                  25.0%            80.0%            3.2x
```

```
                                change   stuck rate   cont|stuck   mean cost      monthly   vs base
---------------------------------------------------------------------------------------------------
                              baseline           8%         0.94      0.0650       81,862     1.00x
      a tool gets slower, more retries          11%         0.94      0.0770       97,067     1.19x
      a prompt change adds one example           8%         0.95      0.0752       94,792     1.16x
             a harder customer segment           8%         0.94      0.0722       90,979     1.11x
                             all three          11%         0.95      0.0981      123,665     1.51x
```

**None of the three was a cost decision** — a dependency's latency, a prompt edit, and who
signed up — and together they move monthly spend **51%**.

```
             cap     cap $   requests capped   spend removed   successes lost   success cost
--------------------------------------------------------------------------------------------
            30x    1.9491            0.410%            4.4%           0.078%              4
            20x    1.2994            0.762%           10.0%           0.145%              4
            12x    0.7796            1.333%           18.1%           0.255%              5
             8x    0.5198            1.953%           24.6%           0.390%              4
             5x    0.3248            3.008%           31.9%           0.732%              3
             3x    0.1949            5.081%           39.2%           1.920%              1
```

An 8× cap truncates **1.95%** of requests, removes **24.6%** of spend, and loses **0.39%**
of successes — **$4.10 per success forgone**
({{eq:per-request-cap-beats-aggregate-budget}}), against the $0.62 a blocked request was
worth above.

```
             cap   max steps   max latency   p99.9 after
--------------------------------------------------------
            30x          48          113s          113s
            12x          29           68s           68s
             8x          23           54s           54s
             3x          12           28s           28s
```

The same cap bounds run length, so it bounds tail latency: **54 seconds at 8×**. One
control, two budgets, usually two teams.

## 10. Production Considerations

Meter tokens in the request handler and accumulate. It is a running total on numbers you
already computed, and it is 26 times faster than the channel most teams use.

Keep the meter and the invoice as separate instruments. One is for control and must be fast;
one is for accounting and must be exact. Reconciling them monthly is the right amount of
reconciliation.

Tag every model call with feature and customer tier before you need it. Attribution is
plumbing that is cheap at write time and expensive during an incident.

Set a per-request cost cap, and set it in dollars rather than steps. A step limit relocates
the tail to the limit; a cost cap bounds the quantity you care about, which is
{{eq:count-limits-cannot-bound-cost}} restated.

Publish the implied price per success forgone whenever the cap is debated. It converts an
argument about aggression into an arithmetic comparison against request value.

Report median and p99 cost per request, not mean. The mean is at the 84th percentile and
moves for reasons unrelated to any decision.

Alarm on the *stuck rate* rather than on spend. It is the upstream variable, it moves first,
and it is attributable in a way that a total is not.

## 11. Common Mistakes

**Treating a budget as a control.** It is a number with no actuator; the loop delay is the
control.

**Choosing the authoritative channel for the control loop.** Fast and approximate beats exact
and late for stopping a runaway.

**Fixing detection and stopping.** Eighty-one percent of what remains is collateral.

**Planning capacity on mean cost per request.** The mean is a tail statistic and it sits
above five-sixths of traffic.

**Using a step limit as a cost control.** It creates a point mass of maximum-cost runs at the
limit.

**Judging a cap by requests truncated.** Judge it by successes forgone; the two differ by
5× here.

## 12. Failure Modes

**Silent budget drift.** Monthly spend rises 20% across a quarter from three unrelated
changes, and every post-hoc investigation finds no cost decision to blame, because there was
none.

**Global throttle during an incident.** Spend is stopped in minutes and the day's traffic
with it, and the incident review records a successful cost response.

**Cap set from the mean of a truncated sample.** The cap is calibrated on data that already
excludes the tail, so it is set too high to bind.

**Attribution tags that stop at the service boundary.** Every call is tagged with the service
that made it, which is the one dimension that does not distinguish the runaway.

**Meter that misses cached tokens.** The running total drifts from the invoice, someone
notices, and the meter is retired as unreliable rather than corrected.

**Cost cap without a user-visible outcome.** The request is truncated and returns a partial
answer indistinguishable from a complete one, so the cap's true cost never appears in any
quality metric.

## 13. Alternatives

**Provider-side spend limits.** Hard caps configured with the provider. Genuinely bounding
and coarse — they stop everything, which is the global throttle with extra steps.

**Pre-flight cost reservation.** {{ch:sd-apis-auth}}'s design: estimate and reserve before
executing. The strongest control available and it requires an estimate the caller may not
have.

**Model cascade.** {{cite:chen2023frugalgpt}}'s approach — try cheap models first. Reduces
the mean substantially and does nothing about the tail, which is where governance lives.

**Chargeback to teams.** Push the budget to whoever incurs it. Aligns incentives, requires
the attribution this chapter argues for anyway, and takes a quarter to negotiate.

**Quota per tenant.** Bound spend per customer rather than per request. Right for multi-tenant
platforms and it does not stop one tenant's runaway from consuming that tenant's whole quota
in an hour.

## 14. Evaluation

Measure your own detection lag by injecting a known spend spike and timing the first channel
that shows it. Most teams have never done this and are surprised.

Compute your cost distribution — median, mean, p99, p99.9 — from a month of requests. The
ratio of mean to median is the single number that says whether aggregate budgeting can work
for you.

Count how many of your model calls carry a feature tag. The answer is usually zero.

Simulate a cap against historical traffic and report requests truncated, spend removed, and
successes lost. All three are computable from logs you have.

Track the stuck rate over time and correlate it against dependency latency and prompt
deploys. The correlation is the early-warning signal.

## 15. Advanced Concepts

The success model in {{sec:9-practical-example}} assumes the success probability is a
property of the mode rather than of the truncation point, which is generous to capping. A run
truncated at step 23 may have been one step from succeeding, and the model charges that as a
stuck-mode failure. The correction pushes the price per success forgone upward — how far
depends on the hazard rate of success late in a run, which is measurable from completed runs
and is, as far as I know, not published for any production agent. The direction of the bias
is against the chapter's recommendation, which is worth saying plainly: **capping looks
better in this model than it will in your data**, though the 5× gap between requests
truncated and successes lost leaves considerable room.

The two-mode mixture is also a simplification of something more interesting. Runs are not
born stuck; they become stuck, usually at an identifiable moment when a tool returns
something the agent cannot use and the agent begins to retry variations. That means the
stuck state is *detectable mid-run* — repeated similar tool calls, no progress in the state,
context growing without new information — and a detector on that signal is a better cap than
a cost threshold, because it fires earlier and with a reason attached. This connects directly
to {{ch:ops-agent-tracing}}'s field list: the same recorded state that makes a failure
localisable afterwards makes the stuck mode detectable during.

There is a governance point underneath the arithmetic that the tables do not carry. Every
control in this chapter moves the decision earlier — from the invoice to the dashboard to the
meter to the request itself — and the value of each move is the burn rate times the time
saved. That is the same shape as {{ch:ops-lifecycle}}'s rework result and
{{ch:sd-fault-tolerance}}'s detection-time result, and the three are worth reading as one
claim in three currencies: **the cost of a control is set by its position in time, not by its
strength.**

Finally, an interaction with {{cite:chen2023frugalgpt}} worth stating. Cascades reduce mean
cost by routing easy requests to cheap models, which improves the numerator of every ratio in
this chapter. But cascades act on the *easy* requests, and the governance problem lives in
the hard ones. A cascade that halves mean cost leaves the tail almost exactly where it was,
so it improves the bill and does not improve the bound — and a team that deploys one may
reasonably believe it has addressed cost governance when it has addressed cost.

## 16. Connection to Previous Chapters

{{eq:count-limits-cannot-bound-cost}} from {{ch:sd-apis-auth}} is confirmed at the
organisational scale: a step limit is a count limit, and it relocates the tail rather than
bounding spend.

{{eq:cost-limits-need-a-reservation}} from the same chapter is where this chapter ends —
the only control that acts before the money is spent.

{{eq:rework-cost-is-set-by-detection-lateness}} from {{ch:ops-lifecycle}} is the identical
structure in defects rather than dollars, and {{sec:15-advanced-concepts}} argues the two are
one result.

{{eq:tail-attribution-differs-from-mean}} from {{ch:sd-latency}} explains why a cost
cap is also a latency control: the expensive requests and the slow requests are the same
requests, so the two budgets share an actuator.

## 17. Exercises

1. Inject a known spend spike and measure your actual detection lag on each channel. What
   overrun does your current loop permit at your burn rate?

2. Compute the mean-to-median ratio of your per-request cost. At what ratio does aggregate
   budgeting stop being usable?

3. Simulate an 8× cap against a month of your traffic. What are requests truncated, spend
   removed, and successes lost?

4. Model the success hazard late in a run and recompute the price per success forgone. How
   much does {{sec:15-advanced-concepts}}'s correction move it?

5. Design a mid-run stuck detector from {{ch:ops-agent-tracing}}'s recorded fields. What
   would it fire on, and how much earlier than a cost cap?

## 18. Interview Questions

1. Our monthly AI budget was exceeded by 30%. What is the first question you ask?

2. Why is a fast approximate spend meter better than an exact one for cost control?

3. We can detect a spend spike in five minutes. Are we done?

4. Our mean cost per request is 2.7× the median. What does that tell you about our budget?

5. Why is a step limit a weak cost control?

6. How would you price a per-request cost cap for a design review?

## 19. Research Questions

1. What is the success hazard rate late in an agent run, and how much does it change the
   economics of truncation?

2. Can the stuck mode be detected mid-run reliably enough to replace a cost cap, and how much
   earlier does it fire?

3. How does per-request cost concentration vary across agent architectures, and is the tail a
   property of the task or of the framework?

4. What is the empirical distribution of AI spend detection lag across organisations, and
   what explains the variance?

## 20. Chapter Summary

A cost control is a feedback loop and its overrun is the burn rate times the loop delay, with
the limit appearing nowhere. At $1,305 an hour of excess, a billing export permits **$37,845
— 21% of the monthly budget — in one incident**; a self-metered token counter permits
**$1,436** ({{eq:budget-overrun-is-set-by-feedback-delay}}).

Detection is half the loop. Without attribution the only actuator is a global throttle that
blocks **40,530 healthy requests a day** to remove 3.5%. After fixing detection, **81% of the
remaining loss is collateral rather than spend** — closing the detection loop promotes
attribution to the binding constraint ({{eq:attribution-precedes-control}}), which makes them
sequential rather than alternative investments.

The budget is also aimed at the wrong statistic. Agent cost per request is heavy-tailed:
median **$0.0238**, mean **$0.0650**, p99.9 at **45× the mean**, and the mean sitting at the
**84th percentile** ({{eq:agent-cost-is-heavy-tailed}}). The top 1% of requests are **29%** of
spend. So the mean moves when nobody decided anything: a slower dependency, a prompt edit, and
a harder customer segment together move monthly spend **51%**.

The control that binds is per-request. An 8× cap truncates **1.95%** of requests, removes
**24.6%** of spend, loses **0.39%** of successes — **$4.10 per success forgone**
({{eq:per-request-cap-beats-aggregate-budget}}) — and bounds tail latency at **54 seconds** as
a free consequence.

What connects all of it is position in time. Every improvement in this chapter moves the
decision earlier: from the invoice to the dashboard, from the dashboard to the meter, from the
meter to the request itself. Nothing gets stronger; things get earlier, and earlier is what
the arithmetic rewards. That is also why the recommendations are cheap — an earlier control
usually needs less machinery than a later one, because it acts on a smaller quantity.

Carry forward: **overrun is set by the loop delay, not the limit**, and **govern the request,
not the budget**.

## 21. Further Reading

- {{cite:chen2023frugalgpt}} — cost reduction by cascade, which improves the bill and leaves
  the tail where governance lives.
- {{cite:sculley2015}} — configuration debt, of which an untagged model call is a clean
  instance.
- {{cite:breck2017}} — a readiness rubric with a monitoring section that predates per-request
  cost as a first-class concern.
- {{cite:paleyes2020deployment}} — deployment obstacles across the lifecycle, several of which
  are cost controls that arrived too late in the loop.

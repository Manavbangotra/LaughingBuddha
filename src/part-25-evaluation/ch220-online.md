---
id: ev-online
number: 220
part: XXV
tier: full
status: draft
requires: [gate-placement-is-set-by-cost-times-escape, canary-share-divides-the-sample-rate,
           evaluation-sets-decay-silently, coverage-is-a-union-not-a-sum]
provides: [experiment-duration-is-set-by-outcome-variance, a-fast-proxy-buys-speed-with-decision-error,
           gate-alarms-multiply-with-metrics-and-releases, a-gate-is-useless-if-mde-exceeds-tolerance]
citations: [card2020power, breck2017, sculley2015, singh2025leaderboard]
---

## 1. Learning Objectives

By the end of this chapter you will be able to compute experiment duration from a metric's
variance and observation rate, and explain why the outcome metric is the slowest one you
own; price the substitution of a proxy in decision errors rather than in days; choose the
proxy that minimises delay plus error rather than the fastest or most correlated one;
compute the family-wise false-alarm rate of a regression gate across metrics and releases;
compute a gate's minimum detectable effect and compare it against the regression you refuse
to ship; and identify the designs that make a gate work without a live experiment.

## 2. Why This Matters

Online evaluation is where {{ch:ev-framework}}'s gates meet real traffic, and the arithmetic
is unforgiving in a specific way: **the quantity a product cares about is the one you have
least of.**

At 42,000 sessions a day, detecting a 3% relative change takes **1.2 days** on a click and
**49 days** on a resolution survey ({{eq:experiment-duration-is-set-by-outcome-variance}}) —
because the survey is observed on **0.8%** of sessions against 100% for a click.

So the experiment runs on a proxy, and the proxy has a decision error. At a correlation of
**0.31**, the right call is made **61.5%** of the time — a regression is shipped **19.2%**
of the time ({{eq:a-fast-proxy-buys-speed-with-decision-error}}). At **0.86**, it is
**93.5%**. Priced together, the cheapest decision is neither the fastest metric nor the
outcome itself: it is `task completed` at **16,872** against the survey's **126,626**.

The second half is the gate. Twelve gating metrics at α = 0.05 across 40 releases produce
**18 false alarms a quarter** with nothing wrong
({{eq:gate-alarms-multiply-with-metrics-and-releases}}), and every correction that removes
them removes power: Bonferroni over 12 × 40 takes power to see a genuine 3% drop from **41%
to 4.0%**.

Worse, most gates cannot detect what they exist to refuse. At a 20% canary, `task completed`
has a minimum detectable effect of **5.07%** against a **3.0%** tolerance
({{eq:a-gate-is-useless-if-mde-exceeds-tolerance}}) — **four of five plausible gates fail
this test**, and a gate that cannot detect its tolerance reports `pass` with no capacity to
report anything else.

## 3. Prerequisites

{{eq:gate-placement-is-set-by-cost-times-escape}} from {{ch:ev-framework}} placed the gates;
this chapter asks whether the ones placed online can actually measure anything.

{{eq:canary-share-divides-the-sample-rate}} from {{ch:ops-deployment}} is half of this
chapter's sizing result: a small canary was chosen to limit exposure and it limits detection
by the same factor.

{{eq:evaluation-sets-decay-silently}} from {{ch:ops-prompt-versioning}} is why online
evaluation is needed at all — an offline set represents last quarter's traffic, and live
traffic is the only thing that represents live traffic.

{{eq:coverage-is-a-union-not-a-sum}} from {{ch:ev-framework}} governs how many online metrics
belong in the gating set, and {{sec:9-practical-example}} adds the reason to keep it small.

{{cite:card2020power}} is the source of every power calculation here, and its finding —
that standard practice is routinely underpowered for the differences it claims — transfers to
online experiments unchanged.

## 4. Intuitive Explanation

Offline evaluation asks whether the system is good. Online evaluation asks whether the change
made it better. Those are different questions and the second one is harder, for a reason that
is arithmetic rather than conceptual.

Every experiment's duration is set by three things: how much the metric varies, how big a
change you want to detect, and how often you observe the metric. The first two are familiar.
The third is where AI systems differ.

Consider four metrics on the same 42,000-session-a-day service. A click is observed on every
session. Session length, likewise. A thumbs-up is observed on 14% of sessions, because most
people do not rate anything. A resolution survey is observed on 0.8%, because most people do
not answer surveys.

Detecting a 3% relative change takes 1.2 days on the click and 49 days on the survey.

And the survey is the metric the product is actually about. **The quantity you care about is
the one you have least of**, and that is not an accident of instrumentation — it is a
property of what is expensive to elicit. Clicks are free because they are a side effect of
using the product. Resolution is expensive because somebody has to say so.

So the experiment runs on a proxy. Everybody does this and it is rational: an answer in a day
beats an answer in seven weeks, and the proxy is correlated with the outcome, so the answer
is mostly right.

The question nobody asks is: *how* mostly?

A proxy correlated at 0.31 with the outcome gets the right ship-or-kill call 61.5% of the
time. It ships a regression 19.2% of the time and kills a genuine win 19.2% of the time. That
is barely better than a coin flip dressed as a measurement.

A proxy correlated at 0.86 — task completion, say, which is close to the outcome without
being it — gets 93.5%.

**A proxy is not a cheaper measurement of the same thing. It is a different measurement with
a computable error rate**, and the error rate follows from the correlation, which almost
nobody estimates before adopting the proxy.

Put both effects on one scale and the answer is interesting. Running on the survey costs
126,626 per decision — all of it delay. Running on the click costs 85,792 — mostly decision
error. Running on task completion costs 16,872.

**The right proxy is neither the fastest nor the most correlated; it is the one minimising
delay plus error.** Finding it requires estimating a correlation and a delay cost, two numbers
that are usually treated as unknowable and are both estimable from a quarter of history.

There is one design change that moves the whole frontier rather than sliding along it, and it
is available in AI systems in a way it is not in most online experimentation.

Pair the comparison. Run the same input through both systems and compare the results, rather
than randomising users between arms. The query's own difficulty cancels, the between-user
variance disappears, and the duration falls by roughly a factor of three.

Most online experiments cannot do this — you cannot show a user both arms and ask which they
preferred without changing the thing you are measuring. A model comparison can: same input,
both systems, offline, at inference cost. {{ch:ev-llm-judge}}'s both-orders protocol is the
same idea applied to a judge, and it works for the same reason.

Under simple randomisation, task completion needs 1.1 days and fits in a sprint anyway. The
survey needs 49 days and does not fit even paired, at 15. Which is honest: some questions
cannot be answered in a sprint and should be answered quarterly rather than proxied weekly.

That is experimentation. Regression gates are the other half of online evaluation and they
fail differently.

A gate is a hypothesis test run every release. Run one test at 5% significance and you get a
false alarm one time in twenty — fine. Run it on twelve metrics across forty releases a
quarter and you have 480 opportunities. Eighteen false alarms a quarter, roughly two a week,
on a system where nothing is wrong.

Teams do not tolerate that for long, and what they do is stop reading the alarms. Which is
the rational response to an instrument whose output is mostly noise, and which then applies to
the real alarm when it comes.

The statistical corrections all make the same trade. Bonferroni over twelve metrics and forty
releases takes false alarms to essentially zero and takes the power to detect a genuine 3%
drop from 41% to 4.0%. **A gate that never cries wolf also never barks.**

The correction that works is not statistical. Pre-register three metrics — decide in advance
which numbers can block a release — and test one-sided, because you only care about
regressions. That gives 3.9 false alarms a quarter at 36.1% power, and costs nothing but a
decision.

But there is a deeper problem, and it is the one that makes most gates decorative.

A gate can only block what it can detect. At a 20% canary observed on 46% of sessions, the
minimum detectable effect on task completion is 5.07% relative. The regression you were
trying to refuse is 3.0%.

**The gate cannot see what it exists to refuse.** It will pass every release regardless of
what is in it, and it will look exactly like a working gate for as long as nothing goes
catastrophically wrong.

Run that check across a plausible set of five gating metrics and four of them fail it. The
p95 latency gate works, because latency is observed on every request and its tolerance is
loose. Faithfulness misses by a hair and needs 1.7 days of canary. Task completion needs 2.9.
Resolution needs 122.

The `days to usable` number is the one to compute before writing a gate. If it exceeds your
canary duration, you are not building a gate — you are building a dashboard that occasionally
lights up.

The way out has two parts. First, ask the right question. The default design tests whether
*anything changed*, which is two-sided, expensive, and not what a gate wants to know. The
question a gate asks is "is this worse than the old system by more than my tolerance?" — a
one-sided non-inferiority test, which is cheaper.

Second, and more usefully: most of what a regression gate needs to do does not require an
experiment at all. A paired offline replay runs both models over the same inputs and removes
between-user variance entirely. A frozen-case assertion asks whether forty specific
known-good cases still pass — not a statistical test, and it catches exactly the regressions
somebody has already seen. A hard floor catches catastrophes without needing to detect
anything subtle.

The recommendation fits in a sentence: **state the regression you refuse to ship, check
whether you can detect it, and if you cannot, fix the design before writing the gate.**

## 5. Formal Explanation

**Duration.** For a two-arm comparison detecting an absolute difference $\delta$ on a metric
with per-observation standard deviation $\sigma$, the per-arm sample requirement is $n
\approx z^2 \cdot 2\sigma^2/\delta^2$ with $z = z_{1-\alpha/2} + z_{1-\beta}$. With traffic
$T$ per day and observation rate $\pi$, duration is $2n / (T\pi)$. Writing $\delta = \epsilon
\mu$ for a relative effect gives

$$\text{days} = \frac{2 z^2 \cdot 2 \sigma^2}{\epsilon^2 \mu^2 T \pi} \;\propto\; \frac{\mathrm{CV}^2}{\pi},$$

so duration scales with the squared coefficient of variation divided by the observation rate.
Rare metrics are penalised linearly and noisy ones quadratically.

**Proxy decisions.** Suppose a change has true effect $t$ and the proxy responds as
$\rho t + \sqrt{1-\rho^2}\,\eta$ with $\eta$ independent noise. Deciding "ship" when the
proxy is positive gives, for a change drawn from a symmetric prior over improvements and
regressions,

$$\Pr[\text{correct}] = \Phi\!\left(\frac{\rho}{\sqrt{1-\rho^2}} \cdot c\right)$$

for a constant $c$ depending on the effect distribution. The error rate is a function of
$\rho$ alone, independent of how long the experiment runs — running the proxy experiment
longer reduces its noise and not its bias.

**Total cost.** With delay cost $\lambda$ per day, win value $V$, regression cost $R$, and
prior $\tfrac12$ on each:

$$L(\text{proxy}) = \lambda \cdot \text{days}(\text{proxy}) + \tfrac12 \Pr[\text{ship}\mid\text{bad}] R + \tfrac12 \Pr[\text{kill}\mid\text{good}] V,$$

minimised at an interior proxy, because days falls and error rises as correlation falls.

**Family-wise alarms.** With $m$ independent metrics at level $\alpha$, the per-release false
alarm probability is $1 - (1-\alpha)^m$; over $K$ releases the expected count is
$K[1 - (1-\alpha)^m]$. Correcting to $\alpha/m$ or $\alpha/(mK)$ reduces alarms and reduces
power at each metric, since power is increasing in $\alpha$.

**Minimum detectable effect.** At significance and power fixed, the MDE is
$\delta^\star = z\sqrt{2\sigma^2/n}$. A gate is *capable* iff $\delta^\star \le \tau$, the
tolerance. Since $n$ scales with canary share and duration, the days required for capability
is $(\delta^\star/\tau)^2$ times the duration used to compute $\delta^\star$ — a quantity
computable before the gate is written and almost never computed.

## 6. Mathematical Foundation

Duration as variance over observation rate:

$$\text{days} \;\propto\; \frac{\mathrm{CV}^2}{\pi}, \qquad \frac{\text{days}(\text{survey})}{\text{days}(\text{click})} = 41$$ (eq:experiment-duration-is-set-by-outcome-variance)

at $\pi = 0.008$ against $1.00$ — **1.2 days against 49**.

The decision error a proxy buys, independent of run length:

$$\Pr[\text{correct}] = \Phi\!\left(\frac{\rho}{\sqrt{1-\rho^2}}c\right), \qquad \rho = 0.31 \Rightarrow 61.5\%, \quad \rho = 0.86 \Rightarrow 93.5\%$$ (eq:a-fast-proxy-buys-speed-with-decision-error)

with the optimum at $\arg\min_p [\lambda\,\text{days}(p) + \text{error}(p)]$ — **16,872**
against the outcome metric's **126,626**.

Alarms multiplying across metrics and releases:

$$\mathbb{E}[\text{false alarms}] = K\left[1 - (1-\alpha)^m\right] = 18.4 \ \text{at}\ m=12,\ K=40,\ \alpha=0.05$$ (eq:gate-alarms-multiply-with-metrics-and-releases)

with power at $\alpha/(mK)$ falling from **41% to 4.0%**.

And the capability condition:

$$\delta^\star = z\sqrt{\frac{2\sigma^2}{n}} \le \tau, \qquad \text{days to capability} = \left(\frac{\delta^\star}{\tau}\right)^2$$ (eq:a-gate-is-useless-if-mde-exceeds-tolerance)

At a 20% canary: $\delta^\star = 5.07\%$ against $\tau = 3.0\%$, needing **2.9 days**.

## 7. Internal Mechanics

Why is the outcome metric always the rarest one? Because the metrics that are dense are the
ones the system emits as a side effect of operating — clicks, latencies, tool calls, token
counts — and the metrics that matter are the ones requiring a judgement somebody has to make.
That judgement is expensive whether it comes from a survey, a support ticket, or a human
rater, so it is sampled. **Density and relevance are inversely related by construction**, and
no amount of instrumentation work changes that, because the constraint is on the human side.

The consequence for experimentation is that the metric hierarchy every team ends up with —
dense proxies at the top of the dashboard, sparse outcomes at the bottom — is a hierarchy
ordered by *availability*, and it is routinely read as a hierarchy ordered by importance.

The proxy error has a property worth stating explicitly because it is counterintuitive:
**running the proxy experiment longer does not reduce the decision error.** Longer runs reduce
the noise in the proxy's estimate, so you learn the proxy's effect very precisely. The error
is in the mapping from the proxy to the outcome, and no amount of data on the proxy improves
that mapping. A team that runs a low-correlation proxy experiment for four weeks instead of
one has bought precision about the wrong quantity.

That also explains why the optimum is interior. Delay cost falls with correlation only because
high-correlation metrics happen to be sparse; error cost falls with correlation directly. Two
terms moving oppositely in the same variable give an interior minimum, and the minimum is at a
proxy nobody would have chosen by either criterion alone.

On the gate side, the family-wise problem has an organisational mechanism as well as a
statistical one. Gating metrics accumulate because adding one is easy and cheap to justify —
each new metric catches a failure somebody saw once — and removing one requires arguing that a
failure class is now acceptable. So $m$ grows monotonically, alarms grow with it, trust falls,
and eventually the gate is bypassed rather than pruned. **Pre-registration works because it
makes the set a decision rather than an accumulation**, which is the same reason it works in
clinical trials.

The MDE problem has the sharpest mechanism of the four and it is worth being precise about.
{{ch:ops-deployment}} found the canary share divides the effective sample rate for detection
while leaving total exposure invariant. Here the same division determines whether a gate can
function at all. So the canary size is being set by one team for exposure reasons and
consumed by another team as a detection budget, with no communication between them — and the
exposure argument always wins, because it is the one with a named risk attached.

Finally: the reason a decorative gate is so durable. A gate that cannot detect its tolerance
returns `pass` on every release. That is indistinguishable from a working gate on a healthy
system, and the first evidence of the difference is a regression shipping. At which point the
gate has one failure against many passes and looks unlucky rather than incapable. **The
capability check has to be done in advance because it cannot be done from the gate's output
history.**

## 8. Implementation

The first listing computes experiment duration and prices the proxy substitution.

```python {tier=A name=C:/Users/MANAVB~1/AppData/Local/Temp/claude/C--Github-LaughingBuddha/30a87753-43a8-48c3-8378-261faf976dbb/scratchpad/hi1}
"""The experiment you can afford measures a proxy; the decision you need is about the outcome.

cite:card2020power found that typical NLP test sets are underpowered for the differences
routinely claimed on them. Online experiments have the same problem with an additional
twist: the quantity a product cares about -- task success, resolution, retention -- is rarer
and noisier than the quantity that is easy to log
(eq:experiment-duration-is-set-by-outcome-variance).

So teams run the experiment on a proxy, get an answer in days instead of months, and inherit
a decision error whose size is set by the proxy's correlation with the outcome
(eq:a-fast-proxy-buys-speed-with-decision-error).

This listing computes both durations, prices the substitution, and measures what variance
reduction buys.
"""
import math

TRAFFIC_PER_DAY = 42000.0
POWER_Z = 2.80                # z(0.80) + z(0.975)

# (metric, mean, per-observation SD, share of sessions where it is observed,
#  correlation with the true outcome)
METRICS = [
    ("click on first result", 0.410, 0.492, 1.00, 0.31),
    ("session length",        4.900, 3.800, 1.00, 0.22),
    ("thumbs-up rate",        0.077, 0.267, 0.14, 0.58),
    ("task completed",        0.612, 0.487, 0.46, 0.86),
    ("issue resolved (survey)", 0.680, 0.466, 0.008, 1.00),
]
EFFECT_REL = 0.03             # the relative improvement we want to detect

print(f"{TRAFFIC_PER_DAY:,.0f} sessions a day. Detecting a "
      f"{EFFECT_REL:.0%} relative change.")
print()
dur = {}
for name, mu, sd, obs, rho in METRICS:
    d = EFFECT_REL * mu
    n = (POWER_Z ** 2) * 2.0 * sd ** 2 / (d ** 2)
    days = n / (TRAFFIC_PER_DAY * obs / 2.0)
    dur[name] = (n, days, rho)

survey_days = dur["issue resolved (survey)"][1]
print(f"{'metric':>24}{'observed on':>13}{'CV':>8}{'obs needed/arm':>17}"
      f"{'days':>9}{'vs outcome':>13}")
print("-" * 84)
for name, mu, sd, obs, rho in METRICS:
    print(f"{name:>24}{obs:>13.1%}{sd / mu:>8.2f}{dur[name][0]:>17,.0f}"
          f"{dur[name][1]:>9.1f}{dur[name][1] / survey_days:>12.3f}x")

print()
print("The metric the product is about takes "
      f"{dur['issue resolved (survey)'][1]:.0f} days;")
print(f"the one that is easy to log takes {dur['click on first result'][1]:.1f}.")

print()
print()
print("So the experiment runs on a proxy. What does that cost in decisions?")
print()
TRUE_EFFECT_SD = 0.020        # spread of true effects across the changes we test
print(f"{'proxy':>24}{'rho':>7}{'days':>8}{'P(right call)':>16}"
      f"{'P(ship a regression)':>23}{'P(kill a win)':>16}")
print("-" * 94)


def phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


dec = {}
for name, mu, sd, obs, rho in METRICS:
    # A change with true effect t shows a proxy effect rho*t plus independent noise.
    # Decide "ship" when the proxy is positive and significant.
    p_ship_given_good = phi(rho / math.sqrt(1 - rho ** 2 + 1e-9) * 0.9)
    p_ship_given_bad = 1.0 - p_ship_given_good
    right = 0.5 * p_ship_given_good + 0.5 * (1 - p_ship_given_bad)
    dec[name] = (rho, dur[name][1], right, 0.5 * p_ship_given_bad,
                 0.5 * (1 - p_ship_given_good))
    print(f"{name:>24}{rho:>7.2f}{dur[name][1]:>8.1f}{right:>16.1%}"
          f"{0.5 * p_ship_given_bad:>23.1%}{0.5 * (1 - p_ship_given_good):>16.1%}")

print()
print("Half the candidate changes are genuine improvements and half are not.")

print()
print()
print("Speed against correctness, priced per decision.")
print()
WIN_VALUE = 190000.0          # annual value of shipping a real improvement
REGRESSION_COST = 240000.0    # annual cost of shipping a regression
DELAY_COST_DAY = 2600.0       # cost of a day of experiment for one change
econ = {}
for name, mu, sd, obs, rho in METRICS:
    r, days, right, ship_bad, kill_good = dec[name]
    delay = days * DELAY_COST_DAY
    err = ship_bad * REGRESSION_COST + kill_good * WIN_VALUE
    econ[name] = (delay, err, delay + err)
base_tot = econ["issue resolved (survey)"][2]
print(f"{'proxy':>24}{'days':>8}{'delay cost':>13}{'error cost':>13}"
      f"{'total':>12}{'vs outcome':>13}")
print("-" * 83)
for name, mu, sd, obs, rho in METRICS:
    delay, err, tot = econ[name]
    print(f"{name:>24}{dec[name][1]:>8.1f}{delay:>13,.0f}{err:>13,.0f}"
          f"{tot:>12,.0f}{tot / base_tot:>12.2f}x")

best = min(econ, key=lambda n: econ[n][2])
print()
print(f"cheapest decision: {best} at {econ[best][2]:,.0f}")

print()
print()
print("Variance reduction: what a paired or covariate-adjusted design buys.")
print()
print(f"{'design':>34}{'variance factor':>18}{'days on task completed':>25}"
      f"{'saving':>10}")
print("-" * 87)
BASE_DAYS = dur["task completed"][1]
DESIGNS = [
    ("simple randomisation",            1.00),
    ("stratify by surface",             0.88),
    ("stratify by surface and tenure",  0.79),
    ("covariate adjustment on history", 0.54),
    ("paired: same query, both arms",   0.31),
]
vr = {}
for name, f in DESIGNS:
    d = BASE_DAYS * f
    vr[name] = d
    print(f"{name:>34}{f:>18.2f}{d:>25.1f}{BASE_DAYS - d:>10.1f}")

print()
print(f"the paired design takes `task completed` from {BASE_DAYS:.1f} days to "
      f"{vr['paired: same query, both arms']:.1f}")

print()
print()
print("Which changes the answer: with variance reduction, can we afford the")
print("metric we actually care about?")
print()
print(f"{'metric':>24}{'days, simple':>15}{'days, paired':>15}"
      f"{'affordable in a sprint?':>26}")
print("-" * 80)
SPRINT = 14.0
for name, mu, sd, obs, rho in METRICS:
    simple = dur[name][1]
    paired = simple * 0.31
    print(f"{name:>24}{simple:>15.1f}{paired:>15.1f}"
          f"{('yes' if paired <= SPRINT else 'no'):>26}")

print(f"""
The duration table is cite:card2020power's problem in production form. Detecting a
{EFFECT_REL:.0%} relative change on `{METRICS[0][0]}` takes
{dur[METRICS[0][0]][1]:.1f} days; on `{METRICS[4][0]}` it takes
{dur[METRICS[4][0]][1]:.0f} (eq:experiment-duration-is-set-by-outcome-variance).

Two things drive that gap and only one of them is variance. The survey metric is observed on
{METRICS[4][3]:.1%} of sessions against {METRICS[0][3]:.0%} for a click, so the effective
sample rate is {METRICS[0][3] / METRICS[4][3]:.0f} times lower before any noise is
considered. **The metric the product is about is the one you have least of.**

That is the pressure every experimentation programme is under, and the substitution it makes
is entirely rational: run on the proxy, decide in a week, move on.

The decision table prices what the substitution costs. At a proxy correlation of
{METRICS[0][4]:.2f}, the right call is made {dec[METRICS[0][0]][2]:.1%} of the time --
a regression is shipped {dec[METRICS[0][0]][3]:.1%} of the time and a genuine win is killed
{dec[METRICS[0][0]][4]:.1%} of the time (eq:a-fast-proxy-buys-speed-with-decision-error).

At {METRICS[3][4]:.2f} -- `{METRICS[3][0]}`, a proxy that is close to the outcome without
being it -- the right call is made {dec[METRICS[3][0]][2]:.1%} of the time.

**A proxy is not a cheaper measurement of the same thing. It is a different measurement with
a known error rate**, and the error rate is computable from the correlation, which almost
nobody estimates before adopting the proxy.

The economics table puts the two effects on one scale. Running on the survey metric costs
{econ[METRICS[4][0]][0]:,.0f} in delay and {econ[METRICS[4][0]][1]:,.0f} in decision error;
running on the click costs {econ[METRICS[0][0]][0]:,.0f} and
{econ[METRICS[0][0]][1]:,.0f}. The cheapest decision is on
`{best}` at {econ[best][2]:,.0f}.

Which is the useful result: **the right proxy is neither the fastest nor the most correlated,
it is the one minimising delay plus error**, and finding it requires estimating a correlation
and a delay cost -- two numbers most teams treat as unknowable and both of which are
estimable from a quarter of history.

The variance-reduction table is the intervention that changes the frontier rather than moving
along it. A paired design -- run the same query through both arms and compare -- takes
`{METRICS[3][0]}` from {BASE_DAYS:.1f} days to
{vr['paired: same query, both arms']:.1f}, because the query's own difficulty cancels.

That is the single most valuable design choice available in AI experimentation and it is
available in AI experimentation specifically. Most online experiments cannot pair, because a
user cannot be shown both arms. **A model comparison can**: the same input, both systems,
offline, at inference cost. ch:ev-llm-judge's both-orders protocol is the same idea applied to
a judge, and it works for the same reason.

The last table is the payoff. Under simple randomisation, `{METRICS[3][0]}` needs
{dur[METRICS[3][0]][1]:.1f} days and does not fit in a sprint. Paired, it needs
{dur[METRICS[3][0]][1] * 0.31:.1f} and does. The survey metric remains out of reach at
{dur[METRICS[4][0]][1] * 0.31:.0f} days even paired, which is honest: some questions cannot be
answered in a sprint and should be answered quarterly rather than proxied weekly.

Two rules to carry into ch:ev-online's second half. **Estimate the proxy's correlation before
adopting it**, because that number is the decision error you are buying. And **pair whenever
the design allows it**, because a factor of three on duration is worth more than any metric
choice on this list.""")
```

## 9. Practical Example

42,000 sessions a day, detecting a 3% relative change:

```
                  metric  observed on      CV   obs needed/arm     days   vs outcome
------------------------------------------------------------------------------------
   click on first result       100.0%    1.20           25,088      1.2       0.025x
          session length       100.0%    0.78           10,478      0.5       0.010x
          thumbs-up rate        14.0%    3.47          209,481     71.3       1.463x
          task completed        46.0%    0.80           11,032      1.1       0.023x
 issue resolved (survey)         0.8%    0.69            8,182     48.7       1.000x
```

**49 days on the metric the product is about, 1.2 on the one that is easy to log**
({{eq:experiment-duration-is-set-by-outcome-variance}}) — and the survey needs *fewer*
observations, it just cannot get them.

```
                   proxy    rho    days   P(right call)   P(ship a regression)   P(kill a win)
----------------------------------------------------------------------------------------------
   click on first result   0.31     1.2           61.5%                  19.2%           19.2%
          session length   0.22     0.5           58.0%                  21.0%           21.0%
          thumbs-up rate   0.58    71.3           73.9%                  13.0%           13.0%
          task completed   0.86     1.1           93.5%                   3.2%            3.2%
 issue resolved (survey)   1.00    48.7          100.0%                   0.0%            0.0%
```

**A proxy is a different measurement with a computable error rate**
({{eq:a-fast-proxy-buys-speed-with-decision-error}}), and the error is a function of $\rho$
alone — running longer does not reduce it.

```
                   proxy    days   delay cost   error cost       total   vs outcome
-----------------------------------------------------------------------------------
   click on first result     1.2        3,106       82,686      85,792        0.68x
          session length     0.5        1,297       90,209      91,507        0.72x
          thumbs-up rate    71.3      185,255       56,078     241,333        1.91x
          task completed     1.1        2,969       13,902      16,872        0.13x
 issue resolved (survey)    48.7      126,626            0     126,626        1.00x
```

**The right proxy is neither the fastest nor the most correlated** — `task completed` at
**16,872** against the survey's **126,626** and the click's **85,792**.

```
                            design   variance factor   days on task completed    saving
---------------------------------------------------------------------------------------
              simple randomisation              1.00                      1.1       0.0
   covariate adjustment on history              0.54                      0.6       0.5
     paired: same query, both arms              0.31                      0.3       0.8
```

**Pairing is available in AI experimentation and not in most others**: same input, both
systems, offline. It is {{ch:ev-llm-judge}}'s both-orders protocol applied to a comparison.

The second listing takes up the regression gate.

```python {tier=A name=C:/Users/MANAVB~1/AppData/Local/Temp/claude/C--Github-LaughingBuddha/30a87753-43a8-48c3-8378-261faf976dbb/scratchpad/hi2}
"""A regression gate is a hypothesis test run every release, and that is the problem.

Run one test at 5% and you get a false alarm one time in twenty. Run it on twelve metrics
across forty releases a quarter and you get several hundred opportunities, so false alarms
arrive weekly (eq:gate-alarms-multiply-with-metrics-and-releases).

Teams respond by ignoring the gate, which is the rational response to an instrument whose
alarms are mostly noise.

The deeper problem is sizing. A gate can only block what it can detect, and the minimum
detectable effect at your traffic is often larger than the regression you were trying to
refuse -- at which point the gate is decorative
(eq:a-gate-is-useless-if-mde-exceeds-tolerance).
"""
import math

RELEASES_PER_QUARTER = 40
ALPHA = 0.05
INVESTIGATION_COST = 3100.0


def family_alarms(metrics, alpha, releases):
    per_release = 1.0 - (1.0 - alpha) ** metrics
    return per_release, per_release * releases


print(f"{RELEASES_PER_QUARTER} releases a quarter, each gated at "
      f"alpha = {ALPHA:.2f}.")
print()
print(f"{'gating metrics':>16}{'P(alarm) per release':>23}"
      f"{'false alarms/quarter':>23}{'cost/quarter':>15}")
print("-" * 77)
fam = {}
for m in (1, 3, 6, 12, 25):
    pr, tot = family_alarms(m, ALPHA, RELEASES_PER_QUARTER)
    fam[m] = (pr, tot, tot * INVESTIGATION_COST)
    print(f"{m:>16}{pr:>23.1%}{tot:>23.1f}"
          f"{tot * INVESTIGATION_COST:>15,.0f}")

print()
print(f"at {12} metrics the gate cries wolf {fam[12][1]:.0f} times a quarter")
print("with nothing wrong")

print()
print()
print("Corrections, and what each costs in sensitivity.")
print()
print(f"{'correction':>28}{'effective alpha':>18}{'false alarms/qtr':>19}"
      f"{'power to see a real 3% drop':>31}")
print("-" * 96)
POWER_Z_A = 1.96
BASE_SE = 0.0138              # standard error of the metric at one release's traffic
REAL_DROP = 0.030


def power_at(alpha, effect, se):
    z = abs(effect) / se - (-math.log(alpha / 2.0)) ** 0.5 * 1.25
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


CORRECTIONS = [
    ("none, 12 metrics",            ALPHA),
    ("Bonferroni over 12",          ALPHA / 12),
    ("Bonferroni over 12 x 40",     ALPHA / (12 * 40)),
    ("pre-register 3 metrics",      ALPHA / 3),
    ("pre-register 3, one-sided",   2 * ALPHA / 3),
]
corr = {}
for name, a in CORRECTIONS:
    m = 12 if "12" in name else 3
    pr, tot = family_alarms(m, a, RELEASES_PER_QUARTER)
    pw = power_at(a, REAL_DROP, BASE_SE)
    corr[name] = (a, tot, pw)
    print(f"{name:>28}{a:>18.5f}{tot:>19.1f}{pw:>31.1%}")

print()
print("Every correction that removes false alarms removes power with it.")

print()
print()
print("The sizing question, which is the one that decides whether a gate works.")
print()
TRAFFIC = 42000.0
OBS_SHARE = 0.46
BASE_RATE = 0.612
print(f"{'canary share':>14}{'observations':>15}{'std error':>12}"
      f"{'MDE at 80% power':>19}{'blocks a 3% drop?':>20}")
print("-" * 80)
TOLERABLE = 0.03 * BASE_RATE
mde = {}
for share in (0.01, 0.05, 0.20, 0.50, 1.00):
    n = TRAFFIC * OBS_SHARE * share
    se = math.sqrt(2.0 * BASE_RATE * (1 - BASE_RATE) / max(n, 1.0))
    d = 2.80 * se
    mde[share] = (n, se, d)
    print(f"{share:>14.0%}{n:>15,.0f}{se:>12.5f}{d / BASE_RATE:>18.2%}"
          f"{('yes' if d <= TOLERABLE else 'no'):>20}")

print()
print(f"tolerable regression: {TOLERABLE / BASE_RATE:.1%} relative "
      f"({TOLERABLE:.4f} absolute)")

print()
print()
print("Same question across metrics, at a 20% canary for one day.")
print()
SHARE = 0.20
METRICS = [
    ("task completed",        0.612, 0.487, 0.46, 0.030),
    ("faithfulness (judge)",  0.734, 0.442, 1.00, 0.020),
    ("thumbs-up rate",        0.077, 0.267, 0.14, 0.050),
    ("issue resolved",        0.680, 0.466, 0.008, 0.030),
    ("p95 latency (s)",       3.100, 1.900, 1.00, 0.080),
]
print(f"{'metric':>22}{'obs in canary':>16}{'MDE':>10}"
      f"{'tolerance':>12}{'usable gate?':>15}{'days to usable':>17}")
print("-" * 92)
gate_ok = {}
for name, mu, sd, obs, tol in METRICS:
    n = TRAFFIC * obs * SHARE
    se = math.sqrt(2.0) * sd / math.sqrt(max(n, 1.0))
    d = 2.80 * se / mu
    days = (d / tol) ** 2
    gate_ok[name] = (n, d, tol, d <= tol, days)
    print(f"{name:>22}{n:>16,.0f}{d:>10.2%}{tol:>12.1%}"
          f"{('yes' if d <= tol else 'no'):>15}{days:>17.1f}")

print()
print("`days to usable` is how long the canary must run for the gate to be")
print("able to detect the regression it exists to refuse")

print()
print()
print("What a gate that cannot detect its tolerance actually does.")
print()
print(f"{'metric':>22}{'MDE / tolerance':>18}{'blocks':>26}"
      f"{'lets through':>26}")
print("-" * 92)
for name, mu, sd, obs, tol in METRICS:
    n, d, t, ok, days = gate_ok[name]
    ratio = d / t
    if ok:
        blocks = "the regression it targets"
        lets = "smaller ones"
    else:
        blocks = f"drops above {d:.1%}"
        lets = f"everything up to {d:.1%}"
    print(f"{name:>22}{ratio:>18.2f}{blocks:>26}{lets:>26}")

print()
print()
print("And the alternative that does not need statistical significance at all.")
print()
print(f"{'design':>34}{'what it tests':>28}{'traffic needed':>17}"
      f"{'catches':>16}")
print("-" * 95)
ALTS = [
    ("two-sided test on the mean", "did anything change", "high", "large drops"),
    ("one-sided non-inferiority",  "is it worse than -3%", "medium", "the tolerance"),
    ("paired offline replay",      "same inputs, both models", "none", "small drops"),
    ("frozen-case assertion",      "these 40 cases still pass", "none", "known regressions"),
    ("guardrail with a hard floor", "never below an absolute", "low", "catastrophes"),
]
for name, tests, traffic, catches in ALTS:
    print(f"{name:>34}{tests:>28}{traffic:>17}{catches:>16}")

print(f"""
The family table is the arithmetic every gated pipeline eventually runs into. Twelve gating
metrics at alpha {ALPHA:.2f} give a {fam[12][0]:.1%} chance of at least one false alarm per
release, which across {RELEASES_PER_QUARTER} releases is **{fam[12][1]:.0f} false alarms a
quarter** costing {fam[12][2]:,.0f} in investigation
(eq:gate-alarms-multiply-with-metrics-and-releases).

That is roughly two a week, on a system where nothing is wrong. Teams do not tolerate that
for long, and what they do about it is stop reading the alarms -- which is a rational response
to an instrument whose output is mostly noise, and which then applies to the real alarm too.

The correction table shows the trade every fix makes. Bonferroni over twelve metrics and
forty releases takes false alarms to {corr['Bonferroni over 12 x 40'][1]:.2f} a quarter and
power to see a genuine {REAL_DROP:.0%} drop to
{corr['Bonferroni over 12 x 40'][2]:.1%}. **A gate that never cries wolf also never barks.**

The row that works is `{CORRECTIONS[4][0]}`: pre-register a small set of metrics, test
one-sided because you only care about regressions, and accept
{corr['pre-register 3, one-sided'][1]:.1f} false alarms a quarter for
{corr['pre-register 3, one-sided'][2]:.1%} power. **Pre-registration is the cheap correction
and it is a discipline rather than a statistic** -- deciding in advance which three numbers
can block a release costs nothing and removes most of the family.

The sizing table is the more serious problem, and it is the one that makes gates decorative.
At a {0.05:.0%} canary the minimum detectable effect is {mde[0.05][2] / BASE_RATE:.1%}
relative, against a tolerable regression of {TOLERABLE / BASE_RATE:.1%}. **The gate cannot
see what it exists to refuse** (eq:a-gate-is-useless-if-mde-exceeds-tolerance), and it will
pass every release regardless of what is in it.

That is not a subtle failure. It is a gate reporting `pass` with no capacity to report
anything else, and it is indistinguishable from a working gate for as long as nothing goes
catastrophically wrong.

This is ch:ops-deployment's canary result meeting cite:card2020power's, and the two compose
badly: a small canary was chosen to limit exposure, and it limits detection by the same
factor.

The per-metric table says which gates in a realistic set are real, and the answer is one of
five. `{METRICS[4][0]}` is usable at a {SHARE:.0%} canary in
{gate_ok[METRICS[4][0]][4]:.1f} days, because it is observed on every request and its
tolerance is loose. `{METRICS[1][0]}` misses by a hair -- {gate_ok[METRICS[1][0]][1]:.2%}
against a {METRICS[1][4]:.1%} tolerance -- and reaches it in
{gate_ok[METRICS[1][0]][4]:.1f} days. `{METRICS[3][0]}` needs
{gate_ok[METRICS[3][0]][4]:.0f} days and will be gating on nothing until then.

**Four of the five gates in a plausible set cannot detect what they were written to
refuse**, at the canary size and duration a team would actually run.

The `days to usable` column is the number to compute before writing a gate. If it exceeds
your canary duration, you are not building a gate -- you are building a dashboard that
occasionally lights up, and ch:ops-deployment already priced what that costs.

The last table is the way out and its first two rows are the point. The default design tests
whether *anything changed*, which is a two-sided question requiring the most traffic and
answering the least useful question. A **one-sided non-inferiority test** -- is the new
system worse than the old by more than the tolerance? -- is the question a gate is actually
asking, and it is cheaper.

Cheaper still are the designs that need no live traffic at all. A paired offline replay runs
both models over the same inputs, which removes the between-user variance entirely -- the
same factor of three ch:ev-online's first listing measured. A frozen-case assertion asks
whether forty specific known-good cases still pass, which is not a statistical test at all
and catches exactly the regressions somebody has already seen.

**Most of what a regression gate needs to do does not require an experiment**, and the part
that does requires a one-sided test against a stated tolerance rather than a search for
significance. That is the whole recommendation, and it fits in a sentence: state the
regression you refuse to ship, check whether you can detect it, and if you cannot, fix the
design before writing the gate.""")
```

```
  gating metrics   P(alarm) per release   false alarms/quarter   cost/quarter
-----------------------------------------------------------------------------
               1                   5.0%                    2.0          6,200
               6                  26.5%                   10.6         32,849
              12                  46.0%                   18.4         56,995
              25                  72.3%                   28.9         89,604
```

**18 false alarms a quarter with nothing wrong**
({{eq:gate-alarms-multiply-with-metrics-and-releases}}) — roughly two a week, which is why
teams stop reading them.

```
                  correction   effective alpha   false alarms/qtr    power to see a real 3% drop
------------------------------------------------------------------------------------------------
            none, 12 metrics           0.05000               18.4                          41.0%
          Bonferroni over 12           0.00417                2.0                          17.6%
     Bonferroni over 12 x 40           0.00010                0.0                           4.0%
      pre-register 3 metrics           0.01667                2.0                          28.7%
   pre-register 3, one-sided           0.03333                3.9                          36.1%
```

**A gate that never cries wolf also never barks.** The row that works is not a statistical
correction — it is pre-registering three metrics and testing one-sided.

```
  canary share   observations   std error   MDE at 80% power   blocks a 3% drop?
--------------------------------------------------------------------------------
            1%            193     0.04958            22.68%                  no
            5%            966     0.02217            10.14%                  no
           20%          3,864     0.01109             5.07%                  no
           50%          9,660     0.00701             3.21%                  no
          100%         19,320     0.00496             2.27%                 yes
```

**The gate cannot see what it exists to refuse**
({{eq:a-gate-is-useless-if-mde-exceeds-tolerance}}) — this is
{{eq:canary-share-divides-the-sample-rate}} arriving as a capability failure.

```
                metric   obs in canary       MDE   tolerance   usable gate?   days to usable
--------------------------------------------------------------------------------------------
        task completed           3,864     5.07%        3.0%             no              2.9
  faithfulness (judge)           8,400     2.60%        2.0%             no              1.7
        thumbs-up rate           1,176    40.04%        5.0%             no             64.1
        issue resolved              67    33.10%        3.0%             no            121.8
       p95 latency (s)           8,400     2.65%        8.0%            yes              0.1
```

**Four of five plausible gates cannot detect what they were written to refuse.** `days to
usable` is the number to compute before writing one.

```
                            design               what it tests   traffic needed         catches
-----------------------------------------------------------------------------------------------
    two-sided test on the mean         did anything change             high     large drops
     one-sided non-inferiority        is it worse than -3%           medium   the tolerance
         paired offline replay    same inputs, both models             none     small drops
         frozen-case assertion   these 40 cases still pass             none  known regressions
   guardrail with a hard floor     never below an absolute              low    catastrophes
```

**Most of what a regression gate needs to do does not require an experiment.**

## 10. Production Considerations

Compute experiment duration per metric before choosing one. CV squared over observation rate
is one line and it settles most of the argument.

Estimate the proxy's correlation with the outcome from history before adopting it. That
number is the decision error you are buying, and it does not improve with run length.

Choose the proxy minimising delay plus error, not the fastest or the most correlated.

Pair the comparison wherever the design allows it. Same input, both systems, is worth a factor
of three and is available to model comparisons in a way it is not to most experiments.

Pre-register the gating metrics. Three, decided in advance, tested one-sided — not twelve
accumulated over two years.

Compute every gate's minimum detectable effect against its tolerance before writing it. If
`days to usable` exceeds your canary duration, the gate is decorative.

Prefer frozen-case assertions and hard floors to statistical gates wherever the failure is
one somebody has already seen. They need no traffic and no significance.

## 11. Common Mistakes

**Running the outcome experiment "when there is time."** At 49 days there will not be; decide
whether it is a quarterly measurement or a proxy.

**Adopting a proxy without measuring its correlation.** The error rate follows from a number
nobody estimated.

**Running a weak proxy experiment longer.** It buys precision about the wrong quantity.

**Accumulating gating metrics.** Twelve gives 18 false alarms a quarter and one gives 2.

**Correcting alpha instead of pruning metrics.** Bonferroni over 480 tests leaves 4% power.

**Writing a gate without computing its MDE.** Four of five plausible gates cannot detect their
own tolerance.

## 12. Failure Modes

**Gate that always passes.** The MDE exceeds the tolerance, every release passes, and it
looks exactly like a healthy system.

**Alarm fatigue then a real regression.** Two false alarms a week trained everyone to
dismiss them, and the eighteenth was real.

**Proxy win that reverses on the outcome.** The click improved, resolution did not, and the
change had already shipped.

**Sprint-length experiments on a quarterly question.** The survey metric was run for two
weeks, came back inconclusive, and was recorded as "no effect".

**Canary sized for exposure, consumed as a detection budget.** Two teams set and spend the
same parameter with no shared model of it.

**Gating metric added after every incident.** Each addition is justified, the set grows
monotonically, and nothing prunes it — {{cite:sculley2015}}'s configuration debt in the
release process.

## 13. Alternatives

**Interleaving.** Show results from both systems in one response and measure which is
engaged with. Much higher power per session, and it constrains the output format.

**Sequential testing with a spending function.** Peek continuously without inflating alpha.
Genuinely useful, and it does not change the MDE that decides whether a gate can function.

**Bandit allocation.** Shift traffic toward the better arm as evidence accumulates. Reduces
regret and complicates the fixed-horizon inference a gate depends on.

**Shadow deployment.** Run the new system on live traffic without serving it, and compare
offline. Removes exposure entirely, keeps the paired design, and cannot measure anything
downstream of the user seeing the output.

**Quarterly outcome measurement.** Accept that the survey metric is a quarterly instrument
and stop trying to run it weekly. Honest, unpopular, and usually correct.

## 14. Evaluation

Compute and publish duration per metric, with the observation rate beside it. Most teams have
never seen the observation rate as a column.

Estimate the correlation between each proxy and the outcome from historical experiments
where both were measured. Publish it next to the proxy on every dashboard.

Compute each gate's MDE and `days to usable` and publish them beside the gate. A gate whose
capability is not stated is a gate nobody has checked.

Count your false alarms per quarter and compare against the family-wise prediction. If they
match, the gate is working as designed and the design is wrong.

Track how many gating metrics you have and when each was added. The monotone growth is the
diagnostic.

## 15. Advanced Concepts

The independence assumed across gating metrics is generous and, unusually for this part, the
correction helps. Metrics on the same system are positively correlated — a bad release moves
several at once — so the family-wise false alarm rate is *lower* than $1 - (1-\alpha)^m$
suggests. That is real relief and it is bounded: with perfect correlation the family collapses
to one test, and typical correlations between quality metrics leave most of the inflation
intact. The practical consequence is that pruning metrics is still the right move and the
urgency is somewhat less than the table implies.

The proxy-error model assumes the correlation is stable, and there is a specific reason to
doubt it. A proxy adopted as a decision criterion becomes an optimisation target, and
{{ch:ev-llm-judge}}'s divergence result applies: selecting changes on the proxy advances the
proxy faster than the outcome, so **the correlation you measured before adopting the proxy is
an overestimate of the correlation you will have after**. The decay rate is the same variance
ratio as in that chapter, and it means a proxy's correlation should be re-estimated
periodically rather than measured once — which nobody does, because measuring it requires the
expensive outcome experiment the proxy was adopted to avoid.

The MDE result interacts with {{cite:singh2025leaderboard}}'s finding in a way worth naming.
That paper measured how unequal data access lets one participant fit an evaluation
distribution better than another. The same mechanism operates internally: a team that runs
many more experiments than another accumulates more information about the metric's noise
structure, and can time and size its experiments better. **Experiment capacity is itself a
resource with distributional consequences**, and a team with a tenth of the traffic is not
running the same programme more slowly — it is running a different, weaker programme.

Finally, the deepest issue in this chapter is one the model cannot express. Every calculation
here assumes the tolerance is known — that somebody has said "a 3% drop in task completion is
unacceptable and a 2% drop is fine." In practice that number is almost never stated, and the
gate is written against statistical significance instead, which is a threshold about noise
rather than about harm. **The MDE check is impossible without a stated tolerance**, which is
why so few teams perform it: the check would first require a decision the organisation has
been avoiding, and the gate exists partly to avoid making it.

## 16. Connection to Previous Chapters

{{eq:canary-share-divides-the-sample-rate}} from {{ch:ops-deployment}} is one half of this
chapter's capability failure: the canary was sized for exposure and spends the same parameter
as a detection budget.

{{eq:gate-placement-is-set-by-cost-times-escape}} from {{ch:ev-framework}} placed the gates;
this chapter asks whether the online ones can measure anything, and four of five cannot.

{{eq:evaluation-sets-decay-silently}} from {{ch:ops-prompt-versioning}} is why online
evaluation exists — the offline set represents last quarter — and this chapter is the price of
that necessity.

{{eq:optimising-against-a-judge-diverges}} from {{ch:ev-llm-judge}} governs the stability of
every proxy correlation here, and {{sec:15-advanced-concepts}} argues it makes measured
correlations optimistic.

## 17. Exercises

1. Compute duration for each of your metrics at a 3% relative effect. Which ones can you
   actually run?

2. Estimate the correlation between your headline proxy and your outcome metric from past
   experiments. What decision error does it imply?

3. Compute the total cost — delay plus error — for three candidate proxies. Which minimises
   it?

4. Compute each of your gates' MDE and `days to usable`. How many are decorative?

5. Model a proxy whose correlation decays as it is optimised against, and find how long it
   remains the cost-minimising choice.

## 18. Interview Questions

1. Our A/B test on click-through says the new model is better. Should we ship?

2. Why does running a proxy experiment longer not fix the proxy problem?

3. We have twelve regression gates and the team ignores them. What happened?

4. Our gate has passed every release for a year. Is that good news?

5. How would you make an experiment on a rare outcome metric affordable?

6. What is the first number you compute before writing a regression gate?

## 19. Research Questions

1. How fast does a proxy's correlation with the outcome decay once it becomes an optimisation
   target?

2. What are realistic correlations between common product proxies and outcome metrics, and how
   much do they vary by domain?

3. How much of the family-wise inflation survives realistic correlation between gating
   metrics?

4. What share of deployed regression gates have an MDE exceeding their intended tolerance?

## 20. Chapter Summary

Online evaluation is where the framework meets traffic, and traffic is scarcer than it looks.

**Duration scales as CV² over observation rate**, so the outcome metric is the slowest thing
you own: **49 days** for a resolution survey against **1.2** for a click, because the survey is
observed on **0.8%** of sessions ({{eq:experiment-duration-is-set-by-outcome-variance}}).
Density and relevance are inversely related by construction.

So the experiment runs on a proxy, and a proxy is a different measurement with a computable
error rate: **61.5%** correct calls at $\rho = 0.31$ against **93.5%** at $\rho = 0.86$
({{eq:a-fast-proxy-buys-speed-with-decision-error}}), and running longer does not improve it.
Priced with delay, the best choice is neither the fastest nor the most correlated —
**16,872** for task completion against **126,626** for the outcome and **85,792** for the
click. Pairing the comparison buys a factor of three and is available to model comparisons in
a way it is not to most experiments.

Regression gates fail twice. Twelve metrics across forty releases give **18 false alarms a
quarter** ({{eq:gate-alarms-multiply-with-metrics-and-releases}}), and every statistical
correction trades them for power — Bonferroni over 480 tests leaves **4.0%**. The fix is
pre-registration and one-sided testing, which is a decision rather than a statistic.

And most gates cannot detect their own tolerance: **5.07%** MDE against a **3.0%** tolerance
at a 20% canary, with **four of five** plausible gates failing the check
({{eq:a-gate-is-useless-if-mde-exceeds-tolerance}}). A gate that cannot detect its tolerance
returns `pass` forever and is indistinguishable from a working one until a regression ships.

What this chapter shares with the rest of the part is a gap between what an instrument
reports and what it is capable of reporting. A proxy reports an effect it can measure
precisely on a quantity nobody asked about. A gate reports `pass` from a test with no power to
say otherwise. In both cases the output is well-formed, the process is followed, and the
information content is near zero — and in both cases one arithmetic check, done before the
instrument is built, would have said so.

Carry forward: **price the proxy in decision errors, not days**, and **compute the MDE before
writing the gate**.

## 21. Further Reading

- {{cite:card2020power}} — the power analysis this chapter applies to online experiments,
  with the finding that standard practice is routinely underpowered.
- {{cite:breck2017}} — a production-readiness rubric whose monitoring section is the closest
  prior art for a gating discipline.
- {{cite:sculley2015}} — configuration debt, of which an accumulating gating-metric set is a
  clean instance.
- {{cite:singh2025leaderboard}} — unequal evaluation access and its distributional
  consequences, which {{sec:15-advanced-concepts}} applies to internal experiment capacity.

---
id: sd-fault-tolerance
number: 195
part: XXII
tier: full
status: draft
requires: [three-properties-break-the-stack, semantic-failure-has-no-instrument,
           retry-needs-a-verifier, hedging-beats-optimising-dependencies]
provides: [retry-value-depends-on-failure-kind, uniform-retry-inverts-its-budget,
           detection-time-sets-the-blast-radius, semantic-breaker-is-affordable]
citations: [cemri2025mast, kwon2023pagedattention, qin2023toolllm]
---

## 1. Learning Objectives

By the end of this chapter you will be able to separate failures into kinds with
different retry economics, and compute the expected value of a retry for each; explain
why a uniform retry policy allocates its budget in inverse proportion to where the
budget is useful, and why that is structural rather than a tuning error; quantify how
good a verifier must be before retrying a confidently-wrong answer is worth doing;
compute the sampling rate at which a semantic circuit breaker detects a regression
within a chosen number of hours; and price that instrument against the damage an
availability-only breaker allows.

## 2. Why This Matters

{{ch:sd-architecture}} found retry surviving at **9%** and circuit breakers at **12%**
under the three properties, and both for the same reason: a retry against a model is
a fresh sample rather than a second attempt, and a breaker cannot trip on an error
rate nobody measures.

Those numbers are correct and too coarse to act on. "Retries do not work" is not a
policy, and a team that hears it either keeps the retries it has or removes them all,
and both are wrong.

{{sec:9-practical-example}} separates failures into kinds and finds their retry values
differ by more than an order of magnitude — from **20.12** for transient
infrastructure to **-0.04** for systematic semantic failure
({{eq:retry-value-depends-on-failure-kind}}). More usefully, it finds that a uniform
three-retry policy spends **65%** of its budget on the two categories returning least,
and that the reason is structural: the kinds retries cannot fix are exactly the kinds
that keep failing, so they consume the full allowance every time
({{eq:uniform-retry-inverts-its-budget}}).

The second half asks what it would take to build a breaker that *can* see semantic
failure. The answer is a sampled review stream, and the surprise is the price:
detection of a realistic regression within **17.5 hours** costs **210** reviewed
answers a day on a 42,000-request service ({{eq:semantic-breaker-is-affordable}}).

## 3. Prerequisites

You need {{eq:three-properties-break-the-stack}} and
{{eq:semantic-failure-has-no-instrument}} from {{ch:sd-architecture}} — this chapter
is the constructive response to both.

{{eq:retry-needs-a-verifier}} from {{ch:ag-recovery}} is made quantitative here: the
verifier requirement turns out to have a threshold, and below it the retry is
negative-value regardless of recall.

{{eq:hedging-beats-optimising-dependencies}} from {{ch:sd-retrieval-agents}} is the
latency-side sibling of this chapter's retry discussion; a hedge is a retry issued
before the first attempt has failed.

Basic familiarity with two-proportion hypothesis testing helps in
{{sec:6-mathematical-foundation}} but is not assumed.

## 4. Intuitive Explanation

Start with why the standard advice fails. "Retry with exponential backoff" is one of
the most reliable pieces of engineering guidance ever written, and it is built on an
assumption: that a failure is a *thing that happened to the request*, not a *property
of the request*. Network blips, overloaded servers, transient contention — these go
away, and trying again works.

A model produces a failure that is a property of the request. Ask it something it
gets wrong, and asking again is a fresh draw from a distribution centred on the same
wrong place. Sometimes you get lucky. Mostly you pay for another call and receive the
same failure for the same reason.

But — and this is where "retries do not work" goes wrong — your system has both kinds
of failure in it, mixed together. The upstream really does return 503 sometimes. The
rate limiter really does say 429. Those are transient and retrying them works
beautifully. It is only the model-shaped failures where retrying is futile.

So the question is not *how many retries* but *which failures*. And here is the part
that makes a uniform policy actively perverse rather than merely suboptimal.

Imagine a hundred failures, split between "will succeed on retry" and "will never
succeed." You retry everything three times. The recoverable ones succeed on attempt
one or two and stop consuming your budget. The hopeless ones fail, fail, and fail
again — consuming the full three attempts, every single time.

**The failures that cannot be fixed consume the most budget precisely because they
cannot be fixed.** A uniform retry policy therefore spends most of its money on the
cases where money achieves nothing, and this is not a bug in the configuration. It is
what "retry until success or exhaustion" does when success probability varies.

The fix is a classifier, and it is much cheaper than it sounds. You do not need a
model to distinguish a 503 from a 429 from a schema-validation failure from a
verifier rejection. That is a switch statement over information the system already
has and currently throws away.

The second half of the chapter is about circuit breakers, and it starts from a
frustrating fact: a breaker watches an error rate, and semantic failure does not
appear in any error rate, because it returns a perfectly good 200.

The only way to see it is to look at some answers. Sample a fraction, have something
judge them, and watch the judged error rate. When it jumps, trip.

The obvious objection is cost — reviewing answers is expensive, and reviewing enough
of them to be statistically confident sounds prohibitive. That objection turns out to
be wrong, and for a specific reason: **detection difficulty scales with the square of
the effect size.** A regression that doubles your error rate is enormously easier to
catch than one that nudges it. And the regressions worth tripping on are the big ones.

## 5. Formal Explanation

Partition failures into kinds $k$ with share $\pi_k$ of all failures, retry success
probability $p_k$, retry harm probability $\eta_k$, and call cost $\kappa$. Let $\Lambda$
be the cost of an unrecovered failure.

For a kind the system can *observe* has failed, the expected value of one retry is

$$ V_k \;=\; p_k\Lambda \;-\; \kappa $$ (eq:retry-value-depends-on-failure-kind)

For a kind the system cannot observe — a confidently-wrong answer returning 200 — a
retry happens only when a verifier of recall $\tau$ flags it, and the retry may
replace a good answer with a bad one:

$$ V_{\text{wrong}} \;=\; \tau\bigl(p\Lambda - \eta\Lambda - \kappa\bigr) $$

which is positive only when $p > \eta + \kappa/\Lambda$. **The verifier does not merely
enable the retry; the retry must be net-positive before recall is applied**, and no
recall rescues a category where harm exceeds benefit.

Now the budget. Under a uniform policy allowing $n$ retries, the expected retries
consumed by kind $k$ is

$$ R_k \;=\; \pi_k \sum_{i=0}^{n-1}(1 - p_k)^i \;=\; \pi_k\,\frac{1 - (1-p_k)^n}{p_k} $$

and the value returned is $\pi_k\bigl(1 - (1-p_k)^n\bigr)\Lambda - R_k$. The share of
budget consumed by kind $k$ is $R_k / \sum_j R_j$, and since
$\sum_{i<n}(1-p_k)^i$ is **decreasing in $p_k$**, we have

$$ \frac{\partial}{\partial p_k}\left(\text{budget share of } k\right) \;<\; 0 \quad\text{while}\quad \frac{\partial V_k}{\partial p_k} \;>\; 0 $$ (eq:uniform-retry-inverts-its-budget)

**Budget share and value per retry move in opposite directions in $p_k$.** That is the
formal statement of the perversity: the policy allocates most where it returns least,
monotonically.

For the breaker, let the semantic error rate shift from $e_0$ to $e_1$. Detecting the
shift at confidence $z$ using a two-proportion test requires approximately

$$ n \;=\; \frac{z^2\,\cdot 2\bar{e}(1 - \bar{e})}{(e_1 - e_0)^2}, \qquad \bar{e} = \frac{e_0 + e_1}{2} $$

samples. Sampling a share $s$ of traffic arriving at rate $T$, detection time is
$n/(sT)$, and the damage during that window is

$$ D \;=\; T\cdot\frac{n}{sT}\cdot(e_1 - e_0)\cdot\Lambda \;=\; \frac{n(e_1-e_0)\Lambda}{s} $$ (eq:detection-time-sets-the-blast-radius)

Substituting $n$ gives $D \propto 1/\bigl(s(e_1 - e_0)\bigr)$ — **damage falls
linearly in the effect size**, so large regressions are both easier to detect and
less costly per unit of detection delay.

## 6. Mathematical Foundation

The sampling-rate decision follows from balancing a continuous cost against an
occasional one. Review costs $\rho$ per sample, so the annual review bill is
$365\,sT\rho$. With $\lambda_r$ regressions per year, expected annual damage is
$\lambda_r D$. Total annual cost is

$$ C(s) \;=\; 365\,sT\rho \;+\; \lambda_r\,\frac{n(e_1 - e_0)\Lambda}{s} $$

which is convex in $s$ with minimum at

$$ s^\star \;=\; \sqrt{\frac{\lambda_r\,n\,(e_1 - e_0)\,\Lambda}{365\,T\rho}} $$ (eq:semantic-breaker-is-affordable)

The square root is what makes this affordable. Because $s^\star$ scales with the
*square root* of the damage-to-review ratio, an order-of-magnitude error in either
parameter moves the optimal sampling rate by only about a factor of three — so the
decision is robust to exactly the estimates teams are least confident about.

Substituting $n$ from {{sec:5-formal-explanation}} gives
$s^\star \propto 1/\sqrt{e_1 - e_0}$: **larger regressions need less sampling**, not
more. {{sec:9-practical-example}} finds $s^\star = 0.5\%$ — 210 reviews a day — and
total annual cost **4.0×** higher at a tenth of that rate.

The asymmetry is worth stating plainly. Under-sampling is expensive in a way that
does not appear on the sampling line item; over-sampling is expensive in a way that
does. That is the same accounting asymmetry as
{{eq:cache-threshold-is-an-error-cost-decision}}, and it produces the same systematic
bias toward the cheap-looking wrong answer.

## 7. Internal Mechanics

**What a failure classifier actually needs.** The taxonomy that matters is
distinguishable from information already in the response path: HTTP status class,
whether the output parsed, whether it validated against a schema, whether a verifier
rejected it, and whether the same request signature failed recently. The last one is
the only nontrivial signal, and it is what separates *systematic* from *recoverable*
semantic failure — a request that failed the same way twice will fail the same way a
third time.

**Backoff is the wrong dimension for semantic failures.** Exponential backoff exists
because transient failures correlate with load and waiting lets load subside. A
semantic failure has no load correlation, so backing off before a resample achieves
nothing except latency. For semantic retries the useful variation is in the *request*
— a different temperature, a reformulated prompt, additional retrieved context —
which is a different operation from a retry and deserves a different name in the code.

**Idempotence is a precondition, not a detail.** A retry that reaches a tool with
side effects executes those side effects again. For read-only tools this is
irrelevant; for anything that writes, sends, or charges, the retry must carry an
idempotency key the tool honours, or the retry policy is a duplication policy. This
matters more in agent systems than in conventional ones because the retry decision is
often made several layers above the side effect -- an orchestrator retries a step, the
step re-invokes a tool, and the tool has no idea it is a repeat. The honest rule is
that a tool without idempotency support is not retryable, and the classifier should
treat it as a zero-retry kind regardless of what kind of failure occurred.

**Where the retry decision belongs.** Retrying inside the tool client, inside the
orchestrator, and inside the HTTP library at once produces a multiplicative retry
count that nobody configured: three layers each allowing three attempts is
twenty-seven calls. This is a common and expensive misconfiguration, and it is
invisible in each layer's own configuration. Retries should be allowed at exactly one
layer, and the others should be explicitly disabled -- a code review item rather than
an architectural one, but one worth writing down.

**Retries interact with the queue.** {{ch:sd-async}} showed wait scaling with the
second moment of service time. Retries add mass to the tail of that distribution
precisely during incidents, so a retry policy is also a capacity policy, and an
un-budgeted one amplifies exactly when capacity is scarce.
{{cite:kwon2023pagedattention}}'s continuous batching makes this sharper: retried
requests join the running batch and slow every request sharing it.

**Correlated failures break the retry model.** {{cite:cemri2025mast}}'s taxonomy shows
multi-agent failures clustering. When failures correlate, the per-kind $p_k$ measured
on average traffic overstates $p_k$ during an incident — retries are least likely to
succeed exactly when the most of them are being issued.

**Trip and recovery semantics for a semantic breaker.** A breaker that trips on a
semantic signal must have somewhere to fall back *to*, and the fallback needs its own
error rate. Falling back to a smaller model, a cached answer, or a refusal are all
defensible; falling back to "return the answer anyway" makes the breaker decorative.
Recovery should require a fresh sample above the threshold, not a timer, because the
condition that tripped it does not resolve on its own.

**Tool-level breakers.** {{cite:qin2023toolllm}}'s large tool collections make
per-tool breaking practical and necessary: one degraded tool should not take down an
agent that has alternatives, and the agent needs to be told the tool is unavailable
rather than discovering it through failures.

## 8. Implementation

The first listing separates failures into kinds and measures where a uniform retry
budget actually goes.

```python {tier=A name=cb1}
"""A blanket retry policy spends most of its budget where retries cannot help.

ch:sd-architecture found retry surviving at 9% under the three properties, and
ch:ag-recovery established why: a retry against a model is a fresh sample, not a
second attempt at the same computation.

But "retries do not work" is too coarse to act on. Failures come in kinds, and retries
have a different expected value for each. This listing separates them and finds where
a fixed retry budget actually goes (eq:retry-value-depends-on-failure-kind).

The result is that a uniform retry policy spends most of its budget on the two
categories where retrying is worthless or harmful, and the fix is a classifier, not a
smaller retry count.
"""
# Failure kinds. (label, share of all failures, P(a retry succeeds),
#                 P(a retry replaces a GOOD answer with a bad one), cost multiple)
KINDS = [
    ("transient infrastructure", 0.31, 0.88, 0.00, 1.0),
    ("rate limited upstream",    0.14, 0.72, 0.00, 1.0),
    ("semantic, recoverable",    0.22, 0.34, 0.00, 1.0),
    ("semantic, systematic",     0.19, 0.04, 0.00, 1.0),
    ("wrong but confident",      0.14, 0.31, 0.18, 1.0),
]
BASE_FAIL = 0.11        # share of requests that fail somehow
CALL_COST = 1.0
ERROR_COST = 24.0       # what an unrecovered failure costs
MAX_RETRIES = 3


def expected_value(kind, verifier_recall):
    """Net value of retrying one failure of this kind, once.

    Retrying costs a call. It may fix the failure, worth ERROR_COST. For the
    'wrong but confident' kind, the system does not KNOW it failed -- so a retry
    only happens if a verifier flags it, and an unflagged retry can also make a
    good answer worse.
    """
    label, share, p_fix, p_harm, cm = kind
    if label == "wrong but confident":
        # Only flagged cases get retried at all.
        p_retry = verifier_recall
        gain = p_retry * p_fix * ERROR_COST
        loss = p_retry * p_harm * ERROR_COST + p_retry * CALL_COST * cm
        return gain - loss
    return p_fix * ERROR_COST - CALL_COST * cm


print("Five kinds of failure, and what a retry does to each.")
print()
print(f"{'failure kind':>27}{'share':>9}{'retry fixes':>13}"
      f"{'retry harms':>13}")
print("-" * 62)
for k in KINDS:
    print(f"{k[0]:>27}{k[1]:>9.0%}{k[2]:>13.0%}{k[3]:>13.0%}")

print()
print()
print("Net value of one retry, per failure, with a verifier of the stated recall.")
print("An unrecovered failure costs %.0f; a call costs %.0f." % (ERROR_COST,
                                                                CALL_COST))
print()
for vr in (0.0, 0.5, 0.9):
    print(f"verifier recall {vr:.0%}:")
    print(f"{'failure kind':>27}{'net value':>12}{'verdict':>20}")
    print("  " + "-" * 57)
    for k in KINDS:
        ev = expected_value(k, vr)
        verdict = ("never retried" if ev == 0.0 else
                   "worth retrying" if ev > 1.0 else
                   "marginal" if ev > 0 else "actively harmful")
        print(f"{k[0]:>27}{ev:>12.2f}{verdict:>20}")
    print()

print()
print("Now a fixed retry budget under a uniform policy: retry everything that")
print("reports failure, up to %d times." % MAX_RETRIES)
print()
print(f"{'failure kind':>27}{'share of retries':>18}{'value returned':>17}"
      f"{'per retry':>12}")
print("-" * 76)
# Under a uniform policy, only kinds the system KNOWS failed get retried.
KNOWN = [k for k in KINDS if k[0] != "wrong but confident"]
known_mass = sum(k[1] for k in KNOWN)
uniform = {}
total_retries = 0.0
total_value = 0.0
for k in KINDS:
    if k[0] == "wrong but confident":
        share_of_retries = 0.0
        val = 0.0
        retries = 0.0
    else:
        # Expected retries spent on this kind before success or exhaustion.
        p = k[2]
        retries = sum((1 - p) ** i for i in range(MAX_RETRIES))
        share_of_retries = k[1] * retries
        val = k[1] * (1 - (1 - p) ** MAX_RETRIES) * ERROR_COST - share_of_retries
    uniform[k[0]] = (share_of_retries, val)
    total_retries += share_of_retries
    total_value += val

for k in KINDS:
    sr, val = uniform[k[0]]
    frac = sr / total_retries if total_retries else 0.0
    per = val / sr if sr else 0.0
    print(f"{k[0]:>27}{frac:>18.0%}{val:>17.2f}{per:>12.2f}")

print("-" * 76)
print(f"{'TOTAL':>27}{1.0:>18.0%}{total_value:>17.2f}"
      f"{total_value / total_retries:>12.2f}")

print()
print()
print("Where that budget goes, ranked. The categories are not equally worth")
print("spending on, and the policy does not know that.")
print()
rank = sorted([k for k in KINDS if uniform[k[0]][0] > 0],
              key=lambda k: -(uniform[k[0]][1] / uniform[k[0]][0]))
print(f"{'rank':>6}{'failure kind':>27}{'budget share':>15}{'value per retry':>18}")
print("-" * 66)
for i, k in enumerate(rank, 1):
    sr, val = uniform[k[0]]
    print(f"{i:>6}{k[0]:>27}{sr / total_retries:>15.0%}{val / sr:>18.2f}")

print()
print()
print("A classified policy: retry only the kinds where it pays, and stop after")
print("the retry count that kind actually warrants.")
print()
print(f"{'failure kind':>27}{'retries allowed':>17}{'budget share':>15}"
      f"{'value':>10}")
print("-" * 69)
POLICY = {
    "transient infrastructure": 3,
    "rate limited upstream":    3,
    "semantic, recoverable":    1,
    "semantic, systematic":     0,
    "wrong but confident":      0,
}
c_retries = 0.0
c_value = 0.0
cls = {}
for k in KINDS:
    n = POLICY[k[0]]
    if n == 0:
        cls[k[0]] = (0.0, 0.0)
        continue
    p = k[2]
    retries = sum((1 - p) ** i for i in range(n))
    sr = k[1] * retries
    val = k[1] * (1 - (1 - p) ** n) * ERROR_COST - sr
    cls[k[0]] = (sr, val)
    c_retries += sr
    c_value += val
for k in KINDS:
    sr, val = cls[k[0]]
    frac = sr / c_retries if c_retries else 0.0
    print(f"{k[0]:>27}{POLICY[k[0]]:>17}{frac:>15.0%}{val:>10.2f}")
print("-" * 69)
print(f"{'TOTAL':>27}{'':>17}{1.0:>15.0%}{c_value:>10.2f}")

print()
print()
print("The two policies compared.")
print()
print(f"{'policy':>22}{'retries spent':>16}{'value returned':>17}"
      f"{'value per retry':>18}")
print("-" * 73)
print(f"{'uniform, 3 retries':>22}{total_retries:>16.3f}{total_value:>17.2f}"
      f"{total_value / total_retries:>18.2f}")
print(f"{'classified':>22}{c_retries:>16.3f}{c_value:>17.2f}"
      f"{c_value / c_retries:>18.2f}")

print(f"""
The per-kind value table is the argument for classifying at all. A transient
infrastructure failure is worth {expected_value(KINDS[0], 0.0):.2f} to retry; a
systematic semantic failure is worth {expected_value(KINDS[3], 0.0):.2f}, which is
negative -- you pay for a call that reproduces the same failure for the same reason.

The `wrong but confident` row is the one ch:sd-architecture said had no instrument,
and its value depends entirely on the verifier. With no verifier it is worth
{expected_value(KINDS[4], 0.0):.2f} -- exactly zero, because nothing flags it and no
retry ever happens. With a {0.9:.0%}-recall verifier it is worth
{expected_value(KINDS[4], 0.9):.2f}.

Note how much of the theoretical value the harm term eats. A flagged retry on this
kind fixes the answer {KINDS[4][2]:.0%} of the time and makes a good answer bad
{KINDS[4][3]:.0%} of the time, so the gross gain of
{KINDS[4][2] * ERROR_COST:.2f} nets down to
{KINDS[4][2] * ERROR_COST - KINDS[4][3] * ERROR_COST - CALL_COST:.2f} per flagged
case before recall is applied. That is eq:retry-needs-a-verifier's requirement made
quantitative: **the verifier does not merely enable the retry, it has to be good
enough to outrun the harm the retry can do** -- and if the harm rate reached
{KINDS[4][2] - CALL_COST / ERROR_COST:.0%} the whole category would be negative at
any recall.

The budget table is the finding. Under a uniform three-retry policy,
{uniform['semantic, systematic'][0] / total_retries:.0%} of all retries are spent on
systematic semantic failures, returning
{uniform['semantic, systematic'][1] / uniform['semantic, systematic'][0]:.2f} per
retry (eq:retry-value-depends-on-failure-kind). Another
{uniform['semantic, recoverable'][0] / total_retries:.0%} goes to recoverable semantic
failures at {uniform['semantic, recoverable'][1] / uniform['semantic, recoverable'][0]:.2f}
per retry.

Together **{(uniform['semantic, systematic'][0] + uniform['semantic, recoverable'][0]) / total_retries:.0%}
of the retry budget goes to the two categories that return least**, and it goes there
for a structural reason: the kinds that retries cannot fix are precisely the kinds
that keep failing, so they consume the full retry allowance every time while the
recoverable ones succeed on the first attempt and stop consuming it.

**A uniform retry policy allocates its budget in inverse proportion to where the
budget is useful.** That is not a tuning error, it is what "retry until success or
exhaustion" does when success probability varies by kind.

The classified policy fixes it by asking what kind of failure this is before
retrying. It spends {c_retries:.3f} retries against the uniform policy's
{total_retries:.3f} -- **{1 - c_retries / total_retries:.0%} fewer** -- and returns
{c_value:.2f} against {total_value:.2f}, which is
{c_value / total_value:.0%} of the value for
{c_retries / total_retries:.0%} of the calls.

Per retry the improvement is {total_value / total_retries:.2f} to
{c_value / c_retries:.2f}, a factor of
{(c_value / c_retries) / (total_value / total_retries):.1f}.

The practical requirement this creates is a failure classifier, and it is a much
smaller ask than it sounds. Distinguishing "the upstream returned 503" from "the
upstream returned 429" from "the model produced output that failed schema validation"
from "the model produced valid output that the verifier rejected" needs no machine
learning at all -- it is a switch statement over things the system already knows.
**The information required to allocate retries well is almost always already present
and almost never used**, because the retry decision is made by a library that was
written before the failure taxonomy existed.""")
```

## 9. Practical Example

Five kinds of failure, and what a retry does to each:

```
               failure kind    share  retry fixes  retry harms
--------------------------------------------------------------
   transient infrastructure      31%          88%           0%
      rate limited upstream      14%          72%           0%
      semantic, recoverable      22%          34%           0%
       semantic, systematic      19%           4%           0%
        wrong but confident      14%          31%          18%
```

Net value of one retry, with a 90%-recall verifier:

```
               failure kind   net value             verdict
  ---------------------------------------------------------
   transient infrastructure       20.12      worth retrying
      rate limited upstream       16.28      worth retrying
      semantic, recoverable        7.16      worth retrying
       semantic, systematic       -0.04    actively harmful
        wrong but confident        1.91      worth retrying
```

A transient failure is worth **20.12** to retry; a systematic semantic failure is
worth **-0.04** ({{eq:retry-value-depends-on-failure-kind}}).

The `wrong but confident` row depends entirely on the verifier. With no verifier it
is **0.00** — nothing flags it, so no retry happens. With 90% recall it is **1.91**.
Note how much the harm term eats: a flagged retry fixes 31% of the time and harms 18%
of the time, so a gross gain of 7.44 nets to 2.12 per flagged case before recall
applies. If the harm rate reached **27%** the category would be negative at any
recall.

Now where a uniform three-retry budget goes:

```
               failure kind  share of retries   value returned   per retry
----------------------------------------------------------------------------
   transient infrastructure               23%             7.08       20.12
      rate limited upstream               12%             3.10       16.28
      semantic, recoverable               30%             3.30        7.16
       semantic, systematic               35%            -0.02       -0.04
        wrong but confident                0%             0.00        0.00
----------------------------------------------------------------------------
                      TOTAL              100%            13.45        8.68
```

**35%** of all retries go to systematic semantic failures returning **-0.04** each.
Together with recoverable semantic failures, **65%** of the budget goes to the two
categories returning least ({{eq:uniform-retry-inverts-its-budget}}).

The reason is structural: kinds that retries cannot fix keep failing, so they consume
the full allowance every time, while recoverable ones succeed early and stop
consuming it. **A uniform retry policy allocates its budget in inverse proportion to
where the budget is useful.**

```mermaid {#fig:retrybudget caption="Retry budget flows toward low-success failure kinds because low success means full allowance consumption. Classification redirects it."}
flowchart TD
  A["failure"] --> B{"classify"}
  B -->|"transient / 429"| C["retry up to 3<br/>value 20.12 per retry"]
  B -->|"semantic, recoverable"| D["retry once<br/>value 7.16"]
  B -->|"semantic, systematic"| E["do not retry<br/>value -0.04"]
  B -->|"wrong but confident"| F["verifier decides<br/>value 1.91 at 90% recall"]
```

Classifying first:

```
                policy   retries spent   value returned   value per retry
-------------------------------------------------------------------------
    uniform, 3 retries           1.550            13.45              8.68
            classified           0.762            11.75             15.42
```

The classified policy returns **87%** of the value for **49%** of the calls — a
per-retry improvement from **8.68** to **15.42**, a factor of **1.8**.

The second listing asks what a breaker that can see semantic failure would cost.

```python {tier=A name=cb2}
"""A circuit breaker cannot trip on an error rate nobody measures.

ch:sd-architecture found circuit breakers surviving at 12%, because a breaker trips
on observable errors and semantic failure returns 200 OK. This listing asks the
follow-up question: what would it take to build a breaker that CAN see semantic
failure, and is it affordable?

The mechanism is sampling. Review a share of answers, watch the measured error rate,
and trip when it shifts. Detection time then falls with the sample rate and with the
square of the effect size (eq:detection-time-sets-the-blast-radius), and the damage a
regression does is traffic multiplied by that time.

The result is that the sample rate needed to catch a real regression quickly is far
lower than intuition suggests, and the reason teams do not have this instrument is
not that it is expensive.
"""
import math

TRAFFIC = 42000.0        # requests per day
BASE_ERR = 0.04          # semantic error rate before the regression
REVIEW_COST = 0.85       # cost of reviewing one sampled answer
ERROR_COST = 24.0        # cost of one wrong answer reaching a user
Z = 2.58                 # ~99% confidence, to avoid tripping on noise


def samples_needed(e0, e1):
    """Sampled answers required to distinguish e1 from e0 at Z confidence."""
    if e1 <= e0:
        return float("inf")
    ebar = (e0 + e1) / 2.0
    return (Z * Z * 2.0 * ebar * (1.0 - ebar)) / ((e1 - e0) ** 2)


def detect_hours(e1, rate):
    """Hours to detect a shift to e1, sampling `rate` of traffic."""
    n = samples_needed(BASE_ERR, e1)
    per_hour = TRAFFIC * rate / 24.0
    if per_hour <= 0:
        return float("inf")
    return n / per_hour


def damage(e1, hours):
    """Cost of the extra wrong answers served during the detection window."""
    return TRAFFIC * (hours / 24.0) * (e1 - BASE_ERR) * ERROR_COST


print("A service at %.0f requests/day with a %.0f%% semantic error rate."
      % (TRAFFIC, BASE_ERR * 100))
print("A regression raises that rate. How long until a sampled monitor notices?")
print()
SHIFTS = [0.06, 0.08, 0.12, 0.20, 0.35]
print(f"{'new error rate':>16}{'effect size':>13}{'samples needed':>17}"
      f"{'reviews/day at 1%':>20}")
print("-" * 66)
need = {}
for e1 in SHIFTS:
    n = samples_needed(BASE_ERR, e1)
    need[e1] = n
    print(f"{e1:>16.0%}{e1 - BASE_ERR:>13.0%}{n:>17.0f}"
          f"{TRAFFIC * 0.01:>20.0f}")

print()
print()
print("Detection time by sample rate. The column that matters is how long a")
print("regression runs before anything notices.")
print()
RATES = [0.001, 0.005, 0.02, 0.05, 0.15]
print(f"{'new error rate':>16}" + "".join(f"{r:>13.1%}" for r in RATES))
print("-" * 81)
grid = {}
for e1 in SHIFTS:
    row = [detect_hours(e1, r) for r in RATES]
    grid[e1] = row
    cells = "".join((f"{h:>12.1f}h" if h < 1000 else f"{'--':>13}") for h in row)
    print(f"{e1:>16.0%}{cells}")

print()
print()
print("What that detection window costs, in wrong answers reaching users.")
print()
print(f"{'new error rate':>16}" + "".join(f"{r:>13.1%}" for r in RATES))
print("-" * 81)
dmg = {}
for e1 in SHIFTS:
    row = [damage(e1, h) for h in grid[e1]]
    dmg[e1] = row
    cells = "".join((f"{d:>13.0f}" if d < 1e7 else f"{'--':>13}") for d in row)
    print(f"{e1:>16.0%}{cells}")

print()
print()
print("The trade. Sampling costs money every day; detection lag costs money only")
print("when a regression happens. Assume one regression a quarter.")
print()
REGRESSIONS_PER_YEAR = 4.0
TARGET = 0.12          # the regression size worth designing for
print(f"{'sample rate':>13}{'reviews/day':>14}{'review cost/yr':>17}"
      f"{'detect':>10}{'damage/yr':>13}{'total/yr':>12}")
print("-" * 79)
best = None
totals = {}
for r in RATES:
    reviews = TRAFFIC * r
    rc = reviews * REVIEW_COST * 365.0
    h = detect_hours(TARGET, r)
    d = damage(TARGET, h) * REGRESSIONS_PER_YEAR
    tot = rc + d
    totals[r] = (reviews, rc, h, d, tot)
    if best is None or tot < totals[best][4]:
        best = r
    print(f"{r:>13.1%}{reviews:>14.0f}{rc:>17.0f}{h:>9.1f}h{d:>13.0f}"
          f"{tot:>12.0f}")

print()
print(f"cheapest: {best:.1%} sampling, {totals[best][4]:.0f} per year total")

print()
print()
print("And the comparison that matters: what an availability-only breaker does.")
print()
print(f"{'breaker':>28}{'detects semantic':>19}{'detect time':>14}"
      f"{'damage/yr':>13}")
print("-" * 74)
print(f"{'availability / status code':>28}{'no':>19}{'never':>14}"
      f"{damage(TARGET, 24.0 * 90) * REGRESSIONS_PER_YEAR:>13.0f}")
for r in (0.005, 0.02):
    h = detect_hours(TARGET, r)
    print(f"{('sampled semantic at %.1f%%' % (r * 100)):>28}{'yes':>19}"
          f"{h:>13.1f}h{damage(TARGET, h) * REGRESSIONS_PER_YEAR:>13.0f}")

print(f"""
The samples-needed column is the first surprise. Detecting a shift from
{BASE_ERR:.0%} to {0.12:.0%} takes {need[0.12]:.0f} reviewed answers -- not
thousands, and not a share of traffic. It is an absolute count, and it is small.

That is because detection scales with the SQUARE of the effect size. A shift to
{0.06:.0%} -- an effect of two points -- needs {need[0.06]:.0f} samples. A shift to
{0.20:.0%} needs {need[0.2]:.0f}. **Big regressions, which are the ones that matter,
are cheap to detect** (eq:detection-time-sets-the-blast-radius), and the expensive
case is distinguishing small drifts that may not be worth tripping on anyway.

The detection grid turns that into wall-clock time. At {0.005:.1%} sampling -- one
answer in two hundred -- a shift to {0.12:.0%} is caught in
{grid[0.12][1]:.1f} hours. At {0.02:.0%} it is caught in {grid[0.12][2]:.1f} hours.

Those are hours, on a service doing {TRAFFIC:.0f} requests a day, for a review budget
of {TRAFFIC * 0.005:.0f} to {TRAFFIC * 0.02:.0f} answers a day.

The cost table prices the whole design. The cheapest configuration is
**{best:.1%} sampling** at {totals[best][4]:.0f} a year all-in -- {totals[best][1]:.0f}
in review cost and {totals[best][3]:.0f} in damage from the four regressions it
catches slightly late.

Sampling less is not cheaper. At {0.001:.1%} the review bill falls to
{totals[0.001][1]:.0f} but detection takes {totals[0.001][2]:.1f} hours and damage
rises to {totals[0.001][3]:.0f} -- a total of {totals[0.001][4]:.0f}, or
{totals[0.001][4] / totals[best][4]:.1f} times the optimum. **Under-sampling is a
false economy in the same shape as ch:sd-routing-caching's over-caching**: the saving
is visible on one line and the cost lands on another.

The last table is the one to take to a design review. An availability breaker never
detects this at all -- the regression returns 200 responses and the breaker has
nothing to trip on -- so the damage runs until a human notices, which the table prices
at a quarter's worth. A sampled semantic breaker at {best:.1%} catches it in
{detect_hours(TARGET, best):.1f} hours.

The ratio between those two damage figures is roughly
{damage(TARGET, 24.0 * 90) / damage(TARGET, detect_hours(TARGET, best)):.0f}
to one, and the instrument that closes it costs
{TRAFFIC * best * REVIEW_COST:.0f} a day.

So the conclusion is narrower and more useful than "you need better observability".
**The second instrument ch:sd-architecture said every model-backed system needs is
affordable at a sampling rate of {best:.1%} -- {TRAFFIC * best:.0f} reviewed answers
a day -- and it is the only thing in the stack that can drive a circuit breaker.**
The reason most systems lack it is not cost. It is that nobody has computed these two
columns and put them next to each other.

One caveat on the mechanism. A breaker driven by sampled review trips on a statistic,
so it inherits every property of the statistic -- including that it will occasionally
trip on noise, and that the Z of {Z:.2f} used here is what keeps that rate low at the
cost of the detection times in the grid. A breaker that trips too readily on a
{TRAFFIC:.0f}-request-a-day service is worse than no breaker, because the response to
a semantic trip is usually to fall back to a degraded mode, and degraded modes have
their own error rates.""")
```

Detecting a shift from a 4% baseline:

```
  new error rate  effect size   samples needed   reviews/day at 1%
------------------------------------------------------------------
              6%           2%             1581                 420
              8%           4%              469                 420
             12%           8%              153                 420
             20%          16%               55                 420
             35%          31%               22                 420
```

Detecting a shift to 12% takes **153** reviewed answers — an absolute count, not a
share of traffic, and a small one. Because detection scales with the *square* of the
effect size, a shift to 20% needs **55** samples and a shift to 35% needs **22**.
**Big regressions, the ones that matter, are cheap to detect**
({{eq:detection-time-sets-the-blast-radius}}).

In wall-clock time on a 42,000-request-per-day service:

```
  new error rate         0.1%         0.5%         2.0%         5.0%        15.0%
---------------------------------------------------------------------------------
              6%       903.4h       180.7h        45.2h        18.1h         6.0h
              8%       268.2h        53.6h        13.4h         5.4h         1.8h
             12%        87.5h        17.5h         4.4h         1.7h         0.6h
             20%        31.4h         6.3h         1.6h         0.6h         0.2h
             35%        12.4h         2.5h         0.6h         0.2h         0.1h
```

And the full cost trade, assuming one regression a quarter:

```
  sample rate   reviews/day   review cost/yr    detect    damage/yr    total/yr
-------------------------------------------------------------------------------
         0.1%            42            13030     87.5h      1175786     1188817
         0.5%           210            65152     17.5h       235157      300310
         2.0%           840           260610      4.4h        58789      319399
         5.0%          2100           651525      1.7h        23516      675041
        15.0%          6300          1954575      0.6h        7839      1962414
```

The optimum is **0.5%** sampling — **210** reviewed answers a day — at **300310** a
year all-in. Sampling a tenth of that drops the review bill to **13030** and raises
damage to **1175786**, a total **4.0×** the optimum. **Under-sampling is a false
economy in the same shape as {{ch:sd-routing-caching}}'s over-caching**: the saving
is visible on one line and the cost lands on another.

Against an availability-only breaker:

```
                     breaker   detects semantic   detect time    damage/yr
--------------------------------------------------------------------------
  availability / status code                 no         never      2903040
   sampled semantic at 0.5%                 yes         17.5h       235157
   sampled semantic at 2.0%                 yes          4.4h        58789
```

An availability breaker never detects this — the regression returns 200 and there is
nothing to trip on — so damage runs until a human notices. The instrument that closes
a **123:1** damage ratio costs **178** a day
({{eq:semantic-breaker-is-affordable}}).

## 10. Production Considerations

Classify before retrying. The information is already in the response path and the
classifier is a switch statement; the win is **1.8×** value per retry and half the
calls.

Set per-kind retry counts, not a global one. Three for transient, one for recoverable
semantic, zero for systematic — and make "systematic" mean "this request signature
already failed this way."

Budget retries globally as a share of baseline load, and shed retries before shedding
requests. Without a cap, retries amplify exactly when capacity is scarce, which is
{{ch:sd-retrieval-agents}}'s hedge-storm failure in a different guise.

Build the sampled review stream. At **0.5%** of a 42,000-request service that is 210
answers a day, and it is the only instrument in the stack that can drive a semantic
breaker.

Give the semantic breaker somewhere to fall back to, with a known error rate. A
breaker whose fallback is "serve it anyway" is decorative.

Require a fresh above-threshold sample to close the breaker, not a timer. The
condition does not resolve on its own.

Measure $p_k$ per kind from your own incident data rather than assuming. Everything in
{{eq:uniform-retry-inverts-its-budget}} follows from it, and it is cheap to obtain
from logs you already keep.

## 11. Common Mistakes

**A single global retry count.** Allocates most budget where it returns least.

**Exponential backoff on semantic failures.** Adds latency and changes nothing; there
is no load to let subside.

**Retrying confidently-wrong answers without checking the harm rate.** Below
$p > \eta + \kappa/\Lambda$ no verifier recall makes it worthwhile.

**Treating a resample as a retry in the code.** They have different success models and
deserve different names, or the next engineer will tune them together.

**Assuming a semantic breaker is unaffordable.** It is 210 reviews a day in the
worked example.

**Closing a semantic breaker on a timer.** The failure does not heal itself.

## 12. Failure Modes

**Retry storm during a systematic regression.** Every request fails the same way,
every one consumes the full allowance, and load triples during the incident.

**Verifier-driven retry loop.** A verifier that rejects everything turns retries into
an unbounded resample; the retry budget cap is what stops it.

**Breaker flapping on noise.** A confidence level set too loose trips on sampling
variance, and the fallback's own error rate becomes the served error rate.

**Silent classifier drift.** A new upstream returns a status the classifier does not
recognise and falls through to the default retry policy, restoring the uniform
behaviour without any change to the retry configuration.

**Sampling bias.** Reviewing only the requests that are easy to review measures a
subpopulation, and the breaker becomes confident about the wrong distribution.

**Multiplied retries across layers.** Three layers each permitting three attempts
produce twenty-seven calls, and no single layer's configuration reveals it. The
symptom is load amplification during incidents that nobody can account for from the
settings they can see.

## 13. Alternatives

**No retries at all.** Defensible when the failure mix is dominated by semantic kinds,
and strictly worse than classification whenever transient failures exist — which is
always.

**Retry with modification rather than resampling.** Change temperature, add context,
reformulate. Higher success than a plain resample and no longer a retry in any useful
sense; belongs in {{ch:ag-recovery}}'s vocabulary.

**Hedging instead of retrying.** {{eq:hedging-beats-optimising-dependencies}} — issue
the duplicate before the first has failed. Attacks latency rather than correctness,
and only affordable where the call is cheap.

**Human review of everything.** Zero semantic error reaching users at prohibitive
cost; the $s \to 1$ limit of the sampling design.

**Proxy signals instead of review.** User retries, thumbs-down, escalation rates.
Cheaper than review and biased in unknown ways; a reasonable supplement and a poor
substitute for the breaker's primary signal.

## 14. Evaluation

Report retry value per retry, by kind. An aggregate retry success rate averages the
categories this chapter exists to separate.

Track retries consumed as a share of baseline load, with an alert on the ratio rather
than the count. Incidents show up here before they show up anywhere else.

Measure the semantic breaker's detection time empirically by injecting a known
regression into a shadow path. The analytic estimate is a starting point; the measured
value includes the review queue's own latency, which is usually the larger term.

Report the sampled semantic error rate beside availability, permanently. This is
{{ch:sd-architecture}}'s second instrument and it needs an owner and a target, not a
dashboard nobody reads.

Audit the classifier's fall-through rate. Requests landing in the default branch are
running the uniform policy this chapter argues against.

## 15. Advanced Concepts

The independence assumption across retries is optimistic in a way worth quantifying.
If a kind's retry successes are correlated — the same underlying condition governs
all attempts — then $1 - (1-p)^n$ overstates recovery badly. In the limit of perfect
correlation, $n$ retries recover exactly $p$ of failures rather than $1 - (1-p)^n$,
which for $p = 0.34, n = 3$ is **34%** against the model's **71%**. Real behaviour lies
between, and the correlation is measurable from retry logs by comparing observed
recovery against the independent prediction.

The breaker's sampling design assumes a step change in error rate. Real regressions
are sometimes gradual — a slowly drifting retrieval corpus, a prompt edited by
accretion — and a two-proportion test against a fixed baseline detects those poorly.
A CUSUM or exponentially-weighted control chart is the right instrument for drift and
has different sampling economics, since it accumulates evidence rather than
re-testing.

The value model also prices every failure identically at $\Lambda$, which flattens
a distinction {{ch:sd-architecture}} spent a chapter establishing. A wrong answer on
a low-stakes surface and a wrong answer in a price quotation differ by orders of
magnitude in cost, and since the retry decision is a comparison against $\kappa$,
the *same* failure kind can be worth retrying on one surface and not on another.
Properly, $\Lambda$ is per-surface and the retry policy is per-surface with it --
which means the classifier's output should be a (kind, surface) pair rather than a
kind alone. That roughly doubles the policy table and changes several of its entries,
and it is the same per-surface argument {{eq:cache-threshold-is-an-error-cost-decision}}
made about cache thresholds.

There is an unexplored composition between the two halves. A verifier good enough to
gate retries is also a verifier good enough to *be* the sampling instrument, so the
marginal cost of the semantic breaker in a system that already retries on verifier
rejections is near zero — the reviews are already happening. Systems that build the
verifier for retry purposes and then separately conclude they cannot afford semantic
monitoring have paid for the instrument twice and used it once.

## 16. Connection to Previous Chapters

{{eq:three-properties-break-the-stack}} put retry at 9% and circuit breakers at 12%.
This chapter is what the surviving fractions look like when engineered deliberately
rather than inherited.

{{eq:retry-needs-a-verifier}} from {{ch:ag-recovery}} gains a threshold here:
$p > \eta + \kappa/\Lambda$, below which no recall helps.

{{eq:semantic-failure-has-no-instrument}} from {{ch:sd-architecture}} is answered
constructively for the first time — {{eq:semantic-breaker-is-affordable}} prices the
missing instrument.

{{eq:variance-not-mean-drives-wait}} from {{ch:sd-async}} explains why an unbudgeted
retry policy is also a capacity risk.

## 17. Exercises

1. Derive the harm rate $\eta$ above which retrying a confidently-wrong answer is
   negative at any verifier recall, for $\Lambda = 24$, $\kappa = 1$.

2. Extend the first listing so retry successes are correlated with parameter $\rho$.
   At what $\rho$ does the classified policy's advantage disappear?

3. Compute $s^\star$ for a service at 4 million requests/day with review cost 2.50 and
   error cost 8. Is it larger or smaller as a share of traffic, and why?

4. Replace the two-proportion test with a CUSUM chart. How does the sampling economics
   change for a gradual drift?

5. Take your own incident log and estimate $\pi_k$ and $p_k$ for four failure kinds.
   What share of your retry budget is going to the worst two?

## 18. Interview Questions

1. Why does a uniform retry policy spend most of its budget on failures it cannot fix?

2. When is retrying a model call worse than not retrying it?

3. Our availability is 99.97% and our answers got worse last Tuesday. What instrument
   was missing, and what would it have cost?

4. Why are large regressions cheaper to detect than small ones?

5. A semantic breaker trips. What should the system do next, and what makes that
   answer hard?

6. Your orchestrator, tool client, and HTTP library each retry three times. How many
   calls does one failure produce, and where would you look to find that out?

## 19. Research Questions

1. How correlated are retry outcomes in practice, and what does that do to the
   independent-attempts model most retry libraries assume?

2. Can failure kind be predicted from the request rather than the response, allowing
   the retry decision to be made before the first attempt?

3. What is the right control chart for semantic drift, and how does its sampling
   economics compare to step-change detection?

4. How much does sharing the verifier between retry gating and breaker sampling
   actually save, and does the shared signal introduce a correlation that harms both?

## 20. Chapter Summary

Retries have a different expected value for each kind of failure — **20.12** for
transient infrastructure against **-0.04** for systematic semantic failure
({{eq:retry-value-depends-on-failure-kind}}). Retrying a confidently-wrong answer is
worth **1.91** at 90% verifier recall and **0.00** without a verifier, and is negative
at any recall once the harm rate exceeds **27%**.

A uniform three-retry policy sends **65%** of its budget to the two categories
returning least, because kinds that retries cannot fix consume the full allowance
every time while recoverable ones succeed early and stop
({{eq:uniform-retry-inverts-its-budget}}). Classifying first returns **87%** of the
value for **49%** of the calls, improving value per retry from **8.68** to **15.42**.

A circuit breaker cannot trip on semantic failure because it returns 200. Building one
that can requires sampled review, and detection scales with the *square* of the effect
size — **153** samples for a 4%-to-12% shift, **22** for a shift to 35%
({{eq:detection-time-sets-the-blast-radius}}).

The optimum sampling rate is **0.5%** — **210** answers a day on a 42,000-request
service — detecting in **17.5 hours** against an availability breaker's never, closing
a **123:1** damage ratio for **178** a day
({{eq:semantic-breaker-is-affordable}}).

Both halves rest on the same observation, which is worth carrying beyond this
chapter. A control that treats a heterogeneous population uniformly does not merely
perform averagely across it -- it performs worst exactly where the population is
most extreme, because the extremes are what consume the control's capacity. That was
true of the retry budget flowing to unfixable failures, and it was true of the
rate limiter in {{ch:sd-apis-auth}} sizing itself to the most expensive request.
Heterogeneity does not average out; it concentrates.

Carry forward: **classify the failure before retrying it**, and **the missing
instrument is affordable — build it**.

## 21. Further Reading

- {{cite:cemri2025mast}} — the failure taxonomy this chapter's kinds are drawn from,
  and evidence that failures correlate.
- {{cite:kwon2023pagedattention}} — continuous batching; why retries slow every
  request sharing a batch.
- {{cite:qin2023toolllm}} — large tool collections, where per-tool breaking becomes
  necessary.

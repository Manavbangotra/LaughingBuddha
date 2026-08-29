---
id: sd-async
number: 191
part: XXII
tier: full
status: draft
requires: [three-properties-break-the-stack, semantic-failure-has-no-instrument,
           cache-threshold-is-an-error-cost-decision, model-belongs-interleaved]
provides: [variance-not-mean-drives-wait, tail-concentration-beats-fair-balancing,
           streaming-helps-until-the-reader-catches-up,
           streaming-capacity-is-set-by-ttft]
citations: [kwon2023pagedattention, pope2022inference, leviathan2023speculative]
---

## 1. Learning Objectives

By the end of this chapter you will be able to explain why the mean service time is
almost useless for capacity planning a generation workload, and compute what the
variance costs instead; use the Pollaczek–Khinchine result to turn a service-time
distribution into a sustainable utilisation and a machine count; argue why fair load
balancing is the wrong policy for a heavy-tailed workload and what replaces it;
locate the concurrency at which streaming stops hiding latency, from two numbers you
already know; and explain why the percentage of latency that streaming saves
improves as the user experience degrades.

## 2. Why This Matters

{{ch:sd-architecture}} found load balancing to be the best-surviving classical
technique, at **36%** — and even that was degraded, because requests stop being
equivalent when one costs forty times another. This chapter is what the 36% actually
looks like, and the answer is more specific and more actionable than "requests
differ."

The specific fact is that generation service time depends on **output length**,
which is not known when the request is admitted and varies by an order of magnitude.
That gives a queue of model calls a coefficient of variation far above the near-1
that web stacks are built around, and queueing theory says waiting time scales with
the **square** of that quantity.

{{sec:9-practical-example}} measures six workloads with **identical mean service
time** and finds mean wait ranging from 2.34s to 8.50s at the same utilisation — a
factor of **3.6** driven entirely by variance ({{eq:variance-not-mean-drives-wait}}).
Under a 3-second wait budget, the tightest workload sustains **74.9%** utilisation
and the loosest **45.1%**, which is a **1.66×** difference in machines for the same
traffic and the same mean.

The second half of the chapter is streaming, the standard response to all of this,
and it contains the part's third instance of a metric that improves while the thing
it measures gets worse.

## 3. Prerequisites

You need {{ch:sd-architecture}}'s three properties
({{eq:three-properties-break-the-stack}}), particularly the *expensive* one, since
cost heterogeneity and time heterogeneity are the same phenomenon measured in
different units.

{{eq:semantic-failure-has-no-instrument}} is the pattern this chapter reproduces
twice more: a mean-latency dashboard and a streaming-effectiveness percentage are
both accurate about their own quantity and silent about the one that matters.

{{eq:cache-threshold-is-an-error-cost-decision}} from {{ch:sd-routing-caching}} is
assumed, because caching changes the length distribution reaching the queue and
therefore changes everything in this chapter.

Basic familiarity with queueing notation ($M/G/1$, utilisation $\rho$) helps but
{{sec:5-formal-explanation}} defines what it uses.

## 4. Intuitive Explanation

Here is the thing that makes generation queues behave unlike web queues, and it is
worth stating before any mathematics.

When a web request arrives, the server broadly knows what it is about to do. Fetch
some rows, render a template, return. Service times cluster. When a generation
request arrives, the server knows the prompt and does not know how long the answer
will be — and the answer might be forty tokens or four thousand. Service time is
proportional to something that has not been decided yet.

Now think about what a queue does with that. Queueing is fundamentally about
**collisions**: a request waits because something else is already being served. If
every job takes about the same time, a collision costs you about that time. If most
jobs are short and a few are enormous, then most collisions are cheap — but
occasionally you arrive behind a monster, and you wait for the monster.

The crucial part is that the monster's cost is not averaged away. It lands entirely
on whoever is behind it, and the longer the monster, the more people accumulate
behind it while it runs. This is why waiting time depends on the **square** of the
variability rather than on the average: a job twice as long is twice as slow *and*
blocks twice as many people.

The practical consequence contradicts the standard instinct. When a queue of model
calls gets slow, the reflex is to add capacity or find a faster model. Both work,
and both are expensive. The cheaper lever is usually to **make the service times
more alike** — cap output length, split long generations into their own queue, batch
by expected length — because you are attacking the term that is squared.

There is a second-order effect worth seeing before the formalism, because it
explains why this feels worse in practice than the arithmetic suggests. The people
who suffer most from a monster job are not sampled uniformly from your users. They
are the ones who happened to arrive during it — and arrivals cluster, because
traffic is bursty. So the wait that the average tells you is spread thinly across
everyone is in fact concentrated on whoever was unlucky, in bursts, repeatedly. The
mean is not merely uninformative here; it actively describes an experience nobody
has.

That also explains a common and confusing observation: latency complaints that do
not correlate with any dashboard. If your p50 is fine and your p99 is acceptable and
users still report the product being slow, the likely explanation is that the
distribution has a shape your percentiles are summarising away, and that the users
complaining are systematically the ones landing behind the tail.

Fair load balancing deserves its own warning here. Spreading requests evenly across
workers is exactly right when requests are equivalent. When one in twenty is a
monster, spreading evenly guarantees that **every worker gets a monster**, and
therefore that every worker has a queue of people stuck behind one. Concentrating
the long jobs on a subset of workers is worse for those jobs and much better for
everyone else, and since the long jobs were going to be slow anyway, that is usually
the right trade.

Streaming is the other half of the chapter, and its intuition is deceptively simple:
show the tokens as they arrive, so the user starts reading immediately instead of
staring at a spinner. This works beautifully — as long as tokens arrive faster than
the user reads them. A person reads about five tokens a second. As long as you emit
faster than that, the user never runs out of text and never experiences a wait after
the first token.

But the per-request emission rate falls as concurrency rises, because a shared
accelerator divides its throughput among everything in flight. So there is a
concurrency at which per-request emission drops to reading speed, and past it the
user is waiting on tokens again. **Streaming's protection has a cliff, and the cliff
is at a concurrency you can compute from two numbers.**

## 5. Formal Explanation

Consider an $M/G/1$ queue: Poisson arrivals at rate $\lambda$, one server, and
service times drawn from a general distribution $S$ with mean $\mathbb{E}[S]$ and
second moment $\mathbb{E}[S^2]$. Utilisation is $\rho = \lambda\,\mathbb{E}[S]$.

The Pollaczek–Khinchine formula gives the mean waiting time in queue:

$$ \mathbb{E}[W] \;=\; \frac{\lambda\,\mathbb{E}[S^2]}{2\,(1 - \rho)} $$ (eq:variance-not-mean-drives-wait)

Writing $\mathbb{E}[S^2] = \mathbb{E}[S]^2(1 + c_v^2)$ with $c_v$ the coefficient of
variation, this becomes

$$ \mathbb{E}[W] \;=\; \frac{\rho\,\mathbb{E}[S]\,(1 + c_v^2)}{2\,(1 - \rho)} $$

which separates the three things that drive waiting. Utilisation contributes the
familiar $\rho/(1-\rho)$ blow-up near saturation. Mean service time contributes
linearly. And variability contributes through $c_v^2$ — **quadratically in the
spread**. A workload with $c_v = 2$ waits five times as long as one with $c_v = 0$
at the same $\rho$ and the same mean.

For generation, $S \approx \alpha + \beta L$ where $L$ is output length, so
$c_v(S) \approx \beta\,\sigma_L / (\alpha + \beta\,\mathbb{E}[L])$ — the variability
of service time is inherited almost entirely from the variability of output length,
which is a property of the task and the prompt rather than of the infrastructure.

Now the balancing question. With $m$ workers and a job-size distribution containing a
heavy tail, consider two policies. **Fair balancing** sends each arrival to a
uniformly random worker, so each worker sees the full distribution: every worker has
$c_v$ equal to the global $c_v$, and every worker's queue suffers the full
{{eq:variance-not-mean-drives-wait}} penalty. **Size-segregated balancing** routes
jobs above a length threshold $\theta_L$ to a dedicated pool of $m_2$ workers and the
rest to $m_1 = m - m_2$.

Let $p$ be the share of jobs above $\theta_L$. The short pool then sees a truncated
distribution with substantially smaller $c_v$, and its waiting time falls by roughly
the ratio of $(1 + c_v^2)$ terms:

$$ \frac{\mathbb{E}[W_{\text{short}}]}{\mathbb{E}[W_{\text{fair}}]} \;\approx\; \frac{1 + c_{v,\text{short}}^2}{1 + c_v^2}\cdot\frac{1 - \rho}{1 - \rho_{\text{short}}} $$ (eq:tail-concentration-beats-fair-balancing)

Because $(1-p)$ of traffic gets the improved term and only $p$ gets the degraded one,
the traffic-weighted mean improves whenever the tail is thin in count and thick in
mass — which is exactly the generation case.

For streaming, let $R$ be the reader's consumption rate in tokens per second, $G$ the
per-request generation rate, and $T$ the time to first token. A user who begins
reading at $T$ consumes tokens at $R$ and receives them at $G$. If $G \ge R$ the
buffer never empties and perceived wait is $T$. If $G < R$ the reader starves, and
total perceived wait for an answer of length $L$ is

$$ P \;=\; T \;+\; \max\!\left(0,\; L\left(\frac{1}{G} - \frac{1}{R}\right)\right) $$ (eq:streaming-helps-until-the-reader-catches-up)

With aggregate server throughput $A$ shared over concurrency $c$, we have $G = A/c$,
so the starvation condition $G < R$ becomes $c > A/R$. **The crossover concurrency
is $A/R$ — server throughput divided by reading speed** — and it involves nothing
about the streaming implementation.

## 6. Mathematical Foundation

Two consequences of {{eq:streaming-helps-until-the-reader-catches-up}} deserve
separate statement.

Below the crossover, $P = T$ exactly, with no dependence on $L$. So the maximum
concurrency a perceived-wait budget $P^\star$ permits is obtained by inverting the
time-to-first-token function alone:

$$ c_{\max} \;=\; T^{-1}(P^\star), \qquad \text{independent of } L $$ (eq:streaming-capacity-is-set-by-ttft)

This is a strong and useful claim. **With streaming, answer length does not affect
capacity at all** — until the crossover, past which it affects it completely.
{{sec:9-practical-example}} shows the same maximum concurrency, 173, for answers from
80 to 1400 tokens, against 33 down to 2 without streaming.

The second consequence is the trap. The *fraction* of latency streaming removes is

$$ \frac{P_{\text{none}} - P_{\text{stream}}}{P_{\text{none}}} = \frac{L/G}{T + L/G} \quad (\text{below crossover}) $$

which **increases** with $L/G$ — that is, it improves as generation gets slower.
Meanwhile $P_{\text{stream}} = T$ is also growing, because time-to-first-token
degrades under the same concurrency that slowed $G$. So the percentage and the
experience move in opposite directions, and a team tracking the percentage sees
improvement while users wait longer. This is
{{eq:semantic-failure-has-no-instrument}}'s pattern in a latency dashboard.

Finally, note what {{eq:variance-not-mean-drives-wait}} implies for capacity
arithmetic. Setting $\mathbb{E}[W] = W^\star$ and solving for $\rho$:

$$ \rho^\star \;=\; \frac{2W^\star}{2W^\star + \mathbb{E}[S](1 + c_v^2)} $$

so the machine count needed for arrival rate $\lambda$ is
$m = \lambda\mathbb{E}[S]/\rho^\star$, which is **linear in $(1 + c_v^2)$**. Halving
the coefficient of variation is worth more than any plausible improvement in mean
service time.

## 7. Internal Mechanics

**Where the variance actually comes from.** Output length distribution is bimodal in
most products: a large mass of short answers and a thin tail of long ones, often
triggered by a recognisable subset of prompts (summarise this document, write the
code, explain in detail). That the tail is *identifiable from the prompt* is the
single most useful operational fact in this chapter, because it makes segregation
implementable — you do not need to know the exact length, only which side of a
threshold it is likely to fall on.

**Why admission control beats queue management.** Once a long job is in service, no
scheduling policy recovers the head-of-line blocking it causes. The lever has to be
applied at admission: route by predicted length, or cap `max_tokens` per surface.
Capping is blunt and effective; prediction is better and needs a small classifier
whose errors are cheap in one direction and expensive in the other.

**Why length caps work better than they should.** Capping `max_tokens` looks like
a crude instrument — it does not make anything faster, it just refuses to do some of
the work. But because wait scales with the second moment, truncating the tail of a
distribution removes far more of $\mathbb{E}[S^2]$ than it removes of
$\mathbb{E}[S]$. A cap that reduces mean service time by 5% can reduce the second
moment by 30%, and it is the second moment that the queue charges for. This is the
highest-leverage single line of configuration in most generation deployments, and it
is usually set by product instinct rather than by measurement.

**The predictor's error asymmetry.** A length predictor used for segregation makes
two kinds of mistake. Sending a short job to the long pool wastes a little capacity
and harms nobody. Sending a long job to the short pool reintroduces exactly the
head-of-line blocking the split was built to prevent, and one such job undoes the
benefit for everyone queued behind it. The predictor should therefore be tuned to
over-predict length — high recall on the long class, tolerating poor precision —
which is the opposite of how a classifier is usually tuned and worth stating
explicitly to whoever builds it.

**Continuous batching changes the model.** {{cite:kwon2023pagedattention}}'s memory
management enables requests to join and leave a running batch rather than waiting for
a batch boundary, which converts much of the classical queueing wait into per-token
contention. That does not remove the variance problem — it relocates it, from
waiting-before-service to slower-during-service — and the $A/c$ term in
{{eq:streaming-helps-until-the-reader-catches-up}} is exactly where it reappears.

**Prefill versus decode.** {{cite:pope2022inference}} separates the compute-bound
prefill phase from the memory-bandwidth-bound decode phase. Time-to-first-token is
dominated by prefill and therefore by *input* length; per-token rate is dominated by
decode and therefore by concurrency. Since {{eq:streaming-capacity-is-set-by-ttft}}
makes capacity a function of TTFT alone below the crossover, **long prompts cost
capacity in a way long answers do not**, which is the reverse of the intuition most
teams start with.

**Speculative decoding as a variance tool.** {{cite:leviathan2023speculative}} raises
$G$ without changing the model's outputs. Under
{{eq:streaming-helps-until-the-reader-catches-up}} that moves the crossover $A/R$
outward proportionally, which is a capacity gain and not merely a speed one.

## 8. Implementation

The first listing takes six workloads with identical mean service time and measures
what their variance costs in wait, in sustainable utilisation, and in machines.

```python {tier=A name=bw1}
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
```

## 9. Practical Example

Six workloads, same mean, different spread:

```
          workload    mean   variance      CV   CV squared
----------------------------------------------------------
       uniform-ish    2.00      0.020    0.07         0.01
       mild spread    2.00      0.500    0.35         0.13
          web-like    2.00      1.556    0.62         0.39
   generation-like    2.00      3.188    0.89         0.80
         long tail    2.00      7.660    1.38         1.92
    very long tail    2.00     10.580    1.63         2.65
```

A capacity plan built on means cannot tell these apart. At 70% utilisation:

```
          workload      CV   mean wait   vs uniform  total latency
------------------------------------------------------------------
       uniform-ish    0.07        2.34s         1.0x          4.34s
       mild spread    0.35        2.62s         1.1x          4.62s
          web-like    0.62        3.24s         1.4x          5.24s
   generation-like    0.89        4.19s         1.8x          6.19s
         long tail    1.38        6.80s         2.9x          8.80s
    very long tail    1.63        8.50s         3.6x         10.50s
```

A factor of **3.6** between tightest and loosest, from variance alone
({{eq:variance-not-mean-drives-wait}}). Generation workloads sit at the
`generation-like` row or worse, already **1.8×** the uniform case.

Under a 3-second wait budget the difference becomes a capacity number:

```
          workload      CV   max utilisation   headroom lost
--------------------------------------------------------------
       uniform-ish    0.07             74.9%            0.0%
       mild spread    0.35             72.7%            2.2%
          web-like    0.62             68.4%            6.6%
   generation-like    0.89             62.5%           12.4%
         long tail    1.38             50.7%           24.2%
    very long tail    1.63             45.1%           29.8%
```

The uniform workload sustains **74.9%**; `generation-like` sustains **62.5%** and
`very long tail` **45.1%**. Serving the same traffic therefore needs **1.20×** and
**1.66×** the machines respectively — for identical mean service time and identical
arrival rate.

This is also why {{ch:sd-architecture}} found load balancing surviving at only 36%.
Even distribution is correct for equivalent requests; here it guarantees every worker
inherits a copy of the tail ({{eq:tail-concentration-beats-fair-balancing}}).

The second listing turns to streaming.

```python {tier=A name=bw2}
"""Streaming hides latency until it does not, and it stops helping under load.

Streaming is the standard answer to slow generation: show tokens as they arrive and
the user starts reading immediately. The usual claim is that this converts a long
wait into a short one.

It converts a long wait into a short one ONLY while tokens arrive faster than the
user reads them. Below that rate the user catches up and waits at the reader's pace,
and the benefit collapses (eq:streaming-helps-until-the-reader-catches-up).

The sharp part is where that happens. Per-request token rate falls as concurrency
rises, because a shared accelerator divides its throughput among in-flight requests
(cite:kwon2023pagedattention). So streaming stops working precisely under load --
which is when the latency it was hiding actually appears.
"""
READ_RATE = 5.0        # tokens/sec a person reads, ~250 words per minute
TTFT_BASE = 0.35       # seconds to first token, unloaded
AGG_TOKENS = 900.0     # aggregate tokens/sec the server can emit across requests
LENGTHS = [80, 250, 600, 1400]


def rates(concurrency):
    """Per-request token rate and time-to-first-token at a concurrency level."""
    per = AGG_TOKENS / concurrency
    ttft = TTFT_BASE * (1 + 0.06 * concurrency)   # queueing ahead of first token
    return per, ttft


def perceived(length, concurrency, streaming):
    """Seconds the user spends waiting rather than reading."""
    per, ttft = rates(concurrency)
    if not streaming:
        # Nothing appears until the whole answer is generated.
        return ttft + length / per
    # Streaming: the user waits for the first token, then waits only for the
    # amount by which generation lags reading.
    starve = length * (1.0 / per - 1.0 / READ_RATE)
    return ttft + max(0.0, starve)


print("A shared accelerator emitting %.0f tokens/sec in aggregate. Per-request"
      % AGG_TOKENS)
print("rate falls as concurrency rises; a reader consumes %.0f tokens/sec."
      % READ_RATE)
print()
print(f"{'concurrency':>13}{'tokens/sec each':>18}{'time to first token':>21}"
      f"{'vs reader':>12}")
print("-" * 64)
CONC = [1, 8, 30, 90, 180, 360]
info = {}
for c in CONC:
    per, ttft = rates(c)
    info[c] = (per, ttft)
    print(f"{c:>13}{per:>18.1f}{ttft:>20.2f}s{per / READ_RATE:>11.1f}x")

print()
print()
print("Perceived wait for a 600-token answer -- the seconds the user spends")
print("looking at an incomplete screen rather than reading.")
print()
L = 600
print(f"{'concurrency':>13}{'no streaming':>15}{'streaming':>12}"
      f"{'saved':>10}{'saved %':>10}")
print("-" * 60)
save = {}
for c in CONC:
    a = perceived(L, c, False)
    b = perceived(L, c, True)
    save[c] = (a, b, a - b, (a - b) / a)
    print(f"{c:>13}{a:>14.2f}s{b:>11.2f}s{a - b:>9.2f}s{(a - b) / a:>10.0%}")

print()
print()
print("The same sweep across answer lengths. Streaming's benefit is a function of")
print("both length and load, and it disappears in the same corner from both.")
print()
print(f"{'answer length':>15}" + "".join(f"{c:>11}" for c in CONC))
print("-" * 81)
grid = {}
for length in LENGTHS:
    row = []
    for c in CONC:
        a = perceived(length, c, False)
        b = perceived(length, c, True)
        row.append((a - b) / a)
    grid[length] = row
    print(f"{length:>15}" + "".join(f"{v:>10.0%} " for v in row))

print()
print()
print("The threshold: the concurrency at which per-request generation drops to")
print("reading speed. Past it, streaming no longer hides anything.")
print()
CROSS = AGG_TOKENS / READ_RATE
print(f"aggregate throughput      {AGG_TOKENS:>8.0f} tokens/sec")
print(f"reader consumption rate   {READ_RATE:>8.0f} tokens/sec")
print(f"crossover concurrency     {CROSS:>8.0f} concurrent requests")
print()
print(f"{'concurrency':>13}{'tokens/sec each':>18}{'streaming still hides':>24}")
print("-" * 57)
for c in CONC:
    per, _ = info[c]
    verdict = "yes, fully" if per >= READ_RATE else "no, reader starves"
    print(f"{c:>13}{per:>18.1f}{verdict:>24}")

print()
print()
print("And what that does to a latency budget. A 4-second perceived-wait target,")
print("by answer length, with and without streaming:")
print()
TARGET = 4.0
print(f"{'answer length':>15}{'max concurrency (no stream)':>30}"
      f"{'max concurrency (stream)':>28}")
print("-" * 73)
cap = {}
for length in LENGTHS:
    best = {}
    for mode in (False, True):
        ok = 0
        for c in range(1, 2001):
            if perceived(length, c, mode) <= TARGET:
                ok = c
            else:
                break
        best[mode] = ok
    cap[length] = best
    print(f"{length:>15}{best[False]:>30}{best[True]:>28}")

print(f"""
The first table is the mechanism. At concurrency 1 each request gets
{info[1][0]:.0f} tokens/sec -- {info[1][0] / READ_RATE:.0f} times faster than a
person reads. At concurrency 360 each gets {info[360][0]:.1f} tokens/sec, which is
{info[360][0] / READ_RATE:.1f} times the reading rate. Somewhere between those the
reader stops being the bottleneck and the server starts being one.

The crossover is exact: {AGG_TOKENS:.0f} aggregate tokens/sec divided by a
{READ_RATE:.0f} tokens/sec reader gives **{CROSS:.0f} concurrent requests**
(eq:streaming-helps-until-the-reader-catches-up). Below it, streaming hides the
entire generation time. Above it, the reader has caught up and every additional
token is a token the user waits for.

What makes this worth a chapter is the SHAPE of the collapse. It is not a gradual
decay -- the saved-percentage column rises to {save[180][3]:.0%} at the crossover
and only then falls off, to {save[360][3]:.0%} at twice the crossover. A cliff, not
a slope, and nothing in the percentage warns you that you are approaching it.

Which sets up the trap. Read down the streaming column in absolute seconds:
{save[1][1]:.2f}s, {save[8][1]:.2f}s, {save[30][1]:.2f}s, {save[90][1]:.2f}s,
{save[180][1]:.2f}s. The user's wait has grown by a factor of
{save[180][1] / save[1][1]:.0f} while the headline saving improved from
{save[1][3]:.0%} to {save[180][3]:.0%}.

**The percentage saved gets better as the experience gets worse.** Streaming is
doing more work than ever -- it is hiding {save[180][2]:.0f} seconds at the
crossover -- and the user is still waiting {save[180][1]:.1f} seconds, because
time-to-first-token degrades with concurrency and streaming cannot hide the wait
before the first token.

That is this part's recurring failure, in a third form. ch:sd-architecture had an
availability graph that stayed green while answers went wrong; ch:sd-routing-caching
had a hit-rate dashboard that rose while total cost rose with it. Here a
streaming-effectiveness metric climbs to {save[180][3]:.0%} while the thing it
claims to measure gets {save[180][1] / save[1][1]:.0f} times worse. **Measure
perceived wait in seconds, never the percentage streaming saved.**

The length grid shows the benefit is real and large away from the cliff:
{grid[LENGTHS[-1]][2]:.0%} for a {LENGTHS[-1]}-token answer at concurrency 30
against {grid[LENGTHS[0]][2]:.0%} for an {LENGTHS[0]}-token one. Long answers are
where streaming earns its keep, and they are also where the queue cost is highest --
which is the tension the next paragraph turns into a warning.

The capacity table is why this belongs in system design rather than front-end work.
Under a {TARGET:.0f}-second perceived-wait target, a {L}-token answer supports
{cap[L][False]} concurrent requests without streaming and **{cap[L][True]}** with it
-- a factor of {cap[L][True] / max(cap[L][False], 1):.0f}. That is a genuine and
large capacity gain, and it is bounded by the crossover rather than by the
implementation: past {CROSS:.0f} concurrent requests no amount of front-end work
recovers it, and the remaining levers are the ones from this chapter's first
listing -- cut the variance, cap the length, or buy throughput.

One consequence to carry forward. Because streaming makes perceived wait nearly
independent of answer length below the crossover, it removes the user-facing reason
to prefer short answers -- while the queueing cost of length, which scales with the
second moment of service time (eq:variance-not-mean-drives-wait), is untouched.
**Streaming hides the cost of length from the user and not from the system.** A
product tuned on perceived latency alone drifts toward longer answers until the
queue notices, and the queue notices as a cliff.""")
```

Per-request emission rate falls with concurrency while time-to-first-token rises:

```
  concurrency   tokens/sec each  time to first token   vs reader
----------------------------------------------------------------
            1             900.0                0.37s      180.0x
            8             112.5                0.52s       22.5x
           30              30.0                0.98s        6.0x
           90              10.0                2.24s        2.0x
          180               5.0                4.13s        1.0x
          360               2.5                7.91s        0.5x
```

The crossover is exact: 900 aggregate tokens/sec over a 5 tokens/sec reader gives
**180 concurrent requests** ({{eq:streaming-helps-until-the-reader-catches-up}}).

For a 600-token answer:

```
  concurrency   no streaming   streaming     saved   saved %
------------------------------------------------------------
            1          1.04s       0.37s     0.67s       64%
            8          5.85s       0.52s     5.33s       91%
           30         20.98s       0.98s    20.00s       95%
           90         62.24s       2.24s    60.00s       96%
          180        124.13s       4.13s   120.00s       97%
          360        247.91s     127.91s   120.00s       48%
```

Two things here. First, the collapse is a **cliff, not a slope** — the saved
percentage climbs to **97%** right at the crossover and only then falls to **48%**.
Nothing in the percentage warns you it is coming.

Second, and worse: read the streaming column in absolute seconds — 0.37s, 0.52s,
0.98s, 2.24s, **4.13s**. The user's wait grew **11×** while the headline saving
improved from 64% to 97%. **The percentage saved gets better as the experience gets
worse**, because time-to-first-token degrades with concurrency and streaming cannot
hide the wait before the first token.

```mermaid {#fig:streaming caption="Below the crossover, streaming makes perceived wait equal to time-to-first-token regardless of answer length. Above it, the reader starves and length returns in full."}
flowchart TD
  A["request admitted"] --> B["prefill<br/>sets TTFT"]
  B --> C{"per-request rate<br/>vs reading rate"}
  C -->|"G >= R"| D["perceived wait = TTFT<br/>length does not matter"]
  C -->|"G < R"| E["reader starves<br/>wait grows with length"]
  D --> F["capacity set by TTFT alone"]
  E --> G["capacity set by throughput"]
```

That is the part's third instance of the same failure: an availability graph that
stayed green while answers went wrong, a hit-rate dashboard that rose while total
cost rose, and now a streaming metric that improves while users wait longer.
**Measure perceived wait in seconds, never the percentage streaming saved.**

The capacity table shows what streaming genuinely buys:

```
  answer length   max concurrency (no stream)    max concurrency (stream)
-------------------------------------------------------------------------
             80                            33                         173
            250                            12                         173
            600                             5                         173
           1400                             2                         173
```

Without streaming, capacity collapses with answer length — 33 down to 2. With
streaming it is **173 for every length**, because below the crossover perceived wait
is time-to-first-token and nothing else ({{eq:streaming-capacity-is-set-by-ttft}}).
For a 600-token answer that is a **35×** capacity gain, and it is bounded by the
crossover rather than by the front-end implementation.

## 10. Production Considerations

Instrument the service-time *distribution*, not its mean. Publish $c_v$ or the
second moment alongside p50 and p99; it is the parameter capacity planning needs and
almost nobody records it.

Segregate long generations at admission, using a prompt-side predictor or an explicit
surface distinction. The tail is usually identifiable before generation starts, which
is what makes {{eq:tail-concentration-beats-fair-balancing}} implementable rather
than merely true.

Cap `max_tokens` per surface and treat the cap as a capacity parameter, not a product
one. It is the bluntest available control on $c_v$ and it works immediately.

Alert on time-to-first-token, not on total latency, wherever streaming is deployed.
{{eq:streaming-capacity-is-set-by-ttft}} makes TTFT the binding constraint below the
crossover, and total latency will look alarming while the experience is fine.

Compute $A/R$ for your deployment and put it on the capacity dashboard as a hard
line. Concurrency approaching it is the single most useful leading indicator in this
chapter, and it is trivially derivable from numbers you already have.

Watch prompt length as a capacity variable. Since TTFT is prefill-dominated
({{cite:pope2022inference}}), growth in retrieved-context size consumes streaming
capacity directly — which links this chapter to {{ch:sd-retrieval-agents}}.

## 11. Common Mistakes

**Capacity planning on mean service time.** It cannot distinguish a 74.9%-sustainable
workload from a 45.1%-sustainable one.

**Balancing fairly across a heavy tail.** Guarantees every worker inherits the tail.

**Adding capacity when the problem is variance.** Machines scale $\rho$; they do not
touch $c_v^2$, and the second term is usually the larger one.

**Reporting streaming's percentage saving.** It improves as the experience degrades.

**Assuming streaming makes length free.** It makes length free *to the user* and
leaves it fully priced in the queue.

**Treating TTFT as a front-end concern.** It is the capacity constraint below the
crossover.

## 12. Failure Modes

**Silent crossover breach.** Concurrency rises past $A/R$ and perceived latency
changes character abruptly; dashboards showing percentage saved do not move in a way
anyone notices.

**Head-of-line collapse.** A batch of long generations arrives together and every
worker blocks simultaneously — the correlated version of the tail problem, and worse
than the independent case the formula assumes.

**Prompt-growth capacity leak.** Retrieved context grows, prefill grows, TTFT grows,
and streaming capacity falls with no change to the model, the traffic, or the answer
length.

**Cap-induced truncation.** A `max_tokens` cap set for capacity reasons silently
truncates answers, producing semantic failure that {{ch:sd-architecture}}'s missing
instrument would have caught and the latency dashboard will not.

**Batching-induced fairness inversion.** Continuous batching can starve long requests
indefinitely under sustained short-request load, converting a latency problem into a
liveness one.

## 13. Alternatives

**Synchronous request-response with a hard timeout.** Simple, and correct when
answers are short and uniform. Fails exactly where $c_v$ is large.

**Full asynchrony with polling or callbacks.** Removes the queueing-latency question
from the user path entirely, at the cost of a much more complex client and a
notification channel. Right for genuinely long work — the tail from
{{eq:tail-concentration-beats-fair-balancing}}, handled as a different product
surface rather than a different pool.

**Speculative decoding.** {{cite:leviathan2023speculative}} raises $G$ and pushes the
crossover outward without changing outputs. Composes with everything here.

**Shortest-remaining-processing-time scheduling.** Optimal for mean wait, but requires
knowing remaining length, which is the thing you do not know. Approximations using
predicted length capture some of the benefit.

**Vertical scaling of the accelerator.** Raises $A$, which moves the crossover
proportionally. Effective and expensive; usually dominated by variance reduction per
unit of spend.

## 14. Evaluation

Report the service-time coefficient of variation as a first-class metric. It is the
input to every capacity calculation in this chapter and its absence is why most
capacity plans are wrong in a predictable direction.

Measure sustainable utilisation against a stated wait budget rather than reporting
observed utilisation. The two differ by exactly the quantity this chapter is about.

For streaming, report perceived wait in seconds at several concurrency levels,
including one above $A/R$. A single-concurrency measurement, especially from staging,
tells you nothing about the cliff.

Track time-to-first-token separately from total latency, and break TTFT down by
prompt length so prefill growth is visible before it consumes capacity.

Validate the length-segregation threshold by measuring $c_v$ in each pool after the
split. If the short pool's $c_v$ has not fallen substantially, the threshold is in
the wrong place.

## 15. Advanced Concepts

The $M/G/1$ model assumes a single server and Poisson arrivals. Real deployments are
$M/G/m$ with continuous batching, where the batching itself couples service times —
a long request slows every request sharing its batch. That coupling makes the
effective $c_v$ seen by any individual request *higher* than the raw distribution
suggests, so the numbers in {{sec:9-practical-example}} understate the problem.

Arrivals are also not Poisson. Interactive traffic is bursty and, worse, correlated
with content: a document-summarisation feature produces long-generation bursts rather
than long generations sprinkled through ordinary traffic. Burst-correlated tails
break the independence that {{eq:tail-concentration-beats-fair-balancing}} assumes,
and segregation helps more rather than less under that violation, since the dedicated
pool absorbs the whole burst.

A further subtlety concerns what the reading rate really is. Treating $R$ as a
constant five tokens per second is a useful simplification, but readers do not
consume prose at a fixed rate: they skim structure, slow on dense passages, and stop
entirely once they have found what they came for. A reader who abandons an answer
halfway has an effective $R$ of infinity for the second half, which means the
crossover is softer than the model suggests and varies by surface. Conversational
answers are read; generated code is scanned for a specific line; a summary is often
consumed in a single glance at the first sentence. Estimating $R$ per surface, rather
than assuming a prose-reading constant everywhere, is the difference between a
crossover you can plan against and one that is merely indicative.

There is an unexplored interaction with {{ch:sd-routing-caching}}. A cache absorbs
repeated queries, which are disproportionately short and common, so caching *raises*
the $c_v$ of the traffic that reaches the model. A cache deployment that improves
mean latency can therefore degrade tail latency, and the capacity plan needs
recomputing after every significant change in hit rate.

## 16. Connection to Previous Chapters

{{eq:three-properties-break-the-stack}} listed load balancing as the best survivor at
36%. {{eq:tail-concentration-beats-fair-balancing}} is what the missing 64% consists
of and what to do about it.

{{eq:semantic-failure-has-no-instrument}} predicted a class of dashboard that is
accurate and irrelevant. This chapter supplies two more instances: mean latency and
streaming percentage saved.

{{eq:cache-threshold-is-an-error-cost-decision}} from {{ch:sd-routing-caching}}
interacts directly, since cache hits change the length distribution reaching the
queue.

{{eq:model-belongs-interleaved}} matters here because deterministic stages have
tight, predictable service times — so an interleaved pipeline has lower $c_v$ overall
than a model-everywhere one, adding a queueing argument to an architectural one.

## 17. Exercises

1. Compute the machine count for a workload with $\mathbb{E}[S] = 3$s, $c_v = 1.5$,
   $\lambda = 12$/s under a 2-second wait budget. Repeat with $c_v = 0.6$.

2. Extend the first listing to $M/G/m$ with $m = 8$ using the Allen–Cunneen
   approximation. Does the variance penalty grow or shrink with more servers?

3. Derive the length threshold $\theta_L$ that minimises traffic-weighted mean wait
   for a two-pool split, given a bimodal length distribution of your choice.

4. Compute $A/R$ for a deployment you have access to. How close is peak concurrency?

5. Show that under {{eq:streaming-capacity-is-set-by-ttft}}, doubling prompt length
   costs more capacity than doubling answer length. At what concurrency does that
   stop being true?

## 18. Interview Questions

1. Two services have identical mean latency and identical traffic. One needs 66% more
   machines. Why?

2. Your generation queue is slow. Argue for reducing variance over adding capacity.

3. Why is fair load balancing the wrong policy for this workload, and what is the
   right one?

4. A team reports that streaming cut perceived latency by 97%. What do you ask next?

5. With streaming enabled, does answer length affect how many concurrent users you
   can serve? Under what condition does your answer change?

## 19. Research Questions

1. How accurately can output length be predicted from the prompt alone, and what
   $c_v$ reduction does a predictor of given accuracy actually deliver?

2. What is the correct queueing model for continuous batching, and how far do
   $M/G/1$ estimates diverge from it in practice?

3. Does the crossover $A/R$ hold empirically when readers skim rather than read, and
   how should $R$ be estimated per surface?

4. Can length segregation be made adaptive — moving the threshold with observed
   $c_v$ — without oscillating?

## 20. Chapter Summary

Generation service time is proportional to output length, which is unknown at
admission and varies by an order of magnitude. That puts generation queues in a
variability regime web stacks are not built for.

Waiting time scales with the **square** of the coefficient of variation
({{eq:variance-not-mean-drives-wait}}). Six workloads with identical mean service
time wait from **2.34s** to **8.50s** at the same utilisation, sustain **74.9%** down
to **45.1%** utilisation under a 3-second budget, and need up to **1.66×** the
machines for the same traffic.

Fair balancing is the wrong policy because it gives every worker a copy of the tail;
segregating long jobs at admission attacks the squared term directly
({{eq:tail-concentration-beats-fair-balancing}}).

Streaming hides generation time only while tokens arrive faster than the user reads
them, and the crossover is server throughput over reading rate — **180 concurrent
requests** here ({{eq:streaming-helps-until-the-reader-catches-up}}). Below it,
capacity is set by time-to-first-token alone and answer length is free: **173**
concurrent requests for answers from 80 to 1400 tokens, against 33 down to 2 without
streaming ({{eq:streaming-capacity-is-set-by-ttft}}).

And the warning: streaming's saved-percentage rose to **97%** while the user's actual
wait grew **11×**. Carry forward: **plan capacity on variance, and measure perceived
wait in seconds.**

## 21. Further Reading

- {{cite:kwon2023pagedattention}} — paged KV cache and continuous batching; where
  classical queueing wait becomes per-token contention.
- {{cite:pope2022inference}} — prefill versus decode, and why time-to-first-token is
  an input-length problem.
- {{cite:leviathan2023speculative}} — speculative decoding; raises $G$ and moves the
  crossover outward without changing outputs.

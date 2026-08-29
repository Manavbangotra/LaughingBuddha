---
id: inf-batching
number: 199
part: XXIII
tier: full
status: draft
requires: [batch-is-the-mechanism-not-an-optimisation, decode-is-bandwidth-bound,
           batch-times-context-is-the-budget, variance-not-mean-drives-wait]
provides: [static-batching-pays-for-the-longest, continuous-batching-gain-grows,
           prefill-stalls-decode, chunk-size-has-a-cliff]
citations: [kwon2023pagedattention, agrawal2023sarathi, zhong2024distserve,
            patel2023splitwise]
---

## 1. Learning Objectives

By the end of this chapter you will be able to compute what static batching wastes on
a distribution of unequal generation lengths, and show why the waste grows with batch
size; explain continuous batching as a utilisation fix rather than a throughput trick,
and quantify its gain from a length distribution alone; describe how a prefill stalls
every decode sharing its step, and price the stall in tokens other users lost;
calculate the chunk size at which prefill rides a decode step for free, and identify
the cliff immediately past it; and compare chunked prefill against disaggregation on
throughput *per machine* rather than on throughput.

## 2. Why This Matters

{{ch:inf-cpu-gpu}} established that batching is the mechanism making a GPU worth using
for decode. This chapter is about the two things that stop a batch from delivering what
the arithmetic promises, and both are scheduling problems rather than hardware ones.

The first is unequal lengths. A **static** batch holds every slot until the longest
sequence finishes, so with generation lengths spanning 40 to 4200 tokens,
{{sec:9-practical-example}} measures batch-32 utilisation at **13.4%**
({{eq:static-batching-pays-for-the-longest}}). Nearly nine tenths of the capacity is
computing padding — not idle, but computing, at full power, producing nothing.

Worse, it degrades as you batch harder: the expected maximum of more draws reaches
further into the tail, so static utilisation falls from **37.8%** at batch 4 to
**10.9%** at batch 64 while continuous batching's advantage grows from **2.6×** to
**9.2×** ({{eq:continuous-batching-gain-grows}}).

The second is phase interference. A prefill and a decode want different things from the
same step, and a 3200-token prefill run as its own step costs **347 tokens** that other
users were waiting for ({{eq:prefill-stalls-decode}}). Two known fixes exist and they
are opposites — chunk the prefill into decode steps, or move prefill to different
machines — and {{sec:9-practical-example}} finds **neither dominates**.

## 3. Prerequisites

You need {{eq:batch-is-the-mechanism-not-an-optimisation}} and
{{eq:decode-is-bandwidth-bound}} from {{ch:inf-cpu-gpu}}. The balance point defined
there is what makes chunked prefill work, and the second listing computes directly
against it.

{{eq:batch-times-context-is-the-budget}} from {{ch:inf-gpu-memory}} bounds the batch
this chapter wants to fill.

{{eq:variance-not-mean-drives-wait}} from {{ch:sd-async}} is the same phenomenon at a
different layer: there, service-time variance drove queueing; here, generation-length
variance drives batch waste. The expected-maximum arithmetic is shared.

## 4. Intuitive Explanation

Picture thirty-two people boarding a bus that will not leave until it is full and will
not stop until everyone has reached their destination. Most are going three stops. One
is going to the end of the line.

Everyone rides to the end of the line.

The bus is not idle during this. It is driving, burning fuel, occupying a driver — doing
exactly as much work as if all thirty-two passengers genuinely needed the full route.
That distinction matters because it means the waste is invisible to every utilisation
metric the operator has: the vehicle is in service, moving, at capacity.

That is static batching. The batch is formed, it runs to completion, and every slot is
occupied until the longest sequence in it finishes. A slot whose sequence finished
forty tokens in does not become available — it sits there, being computed, producing
padding, until the four-thousand-token outlier is done.

The waste is not proportional to the length variation. It is proportional to the
*expected maximum*, and the expected maximum grows with the number of draws. Batch
four and you are probably not unlucky; batch sixty-four and you almost certainly have
someone going to the end of the line.

**So static batching gets worse exactly as you batch harder** — which is the same
unpleasant shape as {{ch:inf-cpu-gpu}}'s KV crossover and
{{ch:sd-retrieval-agents}}'s fan-out. The thing you want to increase makes the problem
you have worse.

Continuous batching fixes it with an idea that sounds trivial and is not: let a
finished sequence leave and let a waiting one take its slot on the very next step. No
batch boundary, no waiting for stragglers, every slot always doing real work. The
implementation is genuinely hard — the model's tensors must be reshaped between steps
and the KV cache must tolerate sequences appearing and disappearing, which is what
{{cite:kwon2023pagedattention}}'s paging enables — but the concept is just: stop making
people ride to the end of the line.

The second half of the chapter is a different interference, and it is subtler.

Prefill and decode are both "the model doing a forward pass," so it is natural to treat
them as the same kind of work. They are not. Prefill processes a whole prompt at once —
hundreds of tokens, compute-bound, taking real time. Decode processes one token per
sequence — memory-bound, fast, and mostly waiting on weights.

Put a prefill in a step and every decode in that step waits for it. One user submits a
long document; thirty-two other users' next tokens are delayed. The victim is never the
person who caused it.

Now the fix that follows from {{ch:inf-cpu-gpu}}, and it is elegant. A decode step is
memory-bound, which means the arithmetic units are mostly idle — hundreds of tokens'
worth of compute capacity sitting unused every single step. So: put a *piece* of the
prefill into the decode step, sized to use exactly that idle capacity. The step takes
the same time, because it was memory-bound and still is. **The prefill is free.**

It is free right up to the balance point and expensive immediately after, which makes
chunk size unusual: it is not a knob to tune by search, it is a number to compute.

The other fix is to give up on sharing entirely and put prefill on different machines.
That works perfectly for the decode machines — they never see a prefill — and it costs
a separate fleet plus shipping the KV cache between them. Which of the two wins is a
real question with a real crossover, and it depends mostly on how long your prompts
are.

## 5. Formal Explanation

Let generation lengths be drawn from a distribution with mean $\bar{L}$ and CDF $F$.
Under static batching with batch $m$, every slot is held for
$\mathbb{E}[\max_{i \le m} L_i]$ steps while the average sequence needs $\bar{L}$.
Utilisation is

$$ U_{\text{static}}(m) \;=\; \frac{m\bar{L}}{m\,\mathbb{E}[\max_{i\le m} L_i]} \;=\; \frac{\bar{L}}{\mathbb{E}[\max_{i \le m} L_i]} $$ (eq:static-batching-pays-for-the-longest)

with $\mathbb{E}[\max_{i\le m} L_i] = \int_0^\infty (1 - F(t)^m)\,dt$, increasing in
$m$. So **utilisation is monotonically decreasing in batch size**, and effective
throughput is $m U_{\text{static}}(m) / \tau$ for step time $\tau$.

Continuous batching holds $U = 1$ by construction, giving throughput $m/\tau$. The gain
is therefore

$$ G(m) \;=\; \frac{1}{U_{\text{static}}(m)} \;=\; \frac{\mathbb{E}[\max_{i\le m} L_i]}{\bar{L}} $$ (eq:continuous-batching-gain-grows)

**The gain is exactly the expected-maximum-to-mean ratio**, and it grows without bound
in $m$ for any distribution with unbounded support. This is why continuous batching's
reported benefit varies so much between deployments: it is a property of the length
distribution and the batch size, not of the implementation.

For phase interference, {{ch:inf-cpu-gpu}} gives step time as
$\tau(n) = \max(W/B,\; 2Pn/F)$ for $n$ tokens in the step. A prefill of $n_p$ tokens run
as its own step costs $\tau(n_p)$, during which $m$ decode slots produce nothing:

$$ \text{tokens lost} \;=\; m\,\frac{\tau(n_p)}{\tau(m)} $$ (eq:prefill-stalls-decode)

Chunked prefill adds $k$ prefill tokens to a decode step of $m$ sequences, costing
$\tau(m + k)$. Since $\tau$ is flat below the balance point $I^\star = F/B$,

$$ \tau(m + k) = \tau(m) \quad\text{for all}\quad k \le I^\star - m $$ (eq:chunk-size-has-a-cliff)

**and jumps to $2P(m+k)/F$ immediately past it.** The optimal chunk is
$k^\star = I^\star - m$ — computed from three hardware constants and the batch size,
not searched.

## 6. Mathematical Foundation

The crossover between chunking and disaggregation follows from comparing throughput per
machine.

Chunked prefill on $M$ machines delivers $M \cdot m/\tau(m+k^\star)$ decode tokens per
second, provided the chunk supply keeps up: with prefill arrival rate $\lambda$ and
$\lceil n_p/k^\star \rceil$ chunks per prefill, the constraint is

$$ \lambda \left\lceil \frac{n_p}{k^\star} \right\rceil \;\le\; \frac{M}{\tau(m + k^\star)} $$

Past that point chunks queue and steps must carry more than $k^\star$, pushing past the
cliff. **Chunking has a capacity limit proportional to $k^\star/n_p$** — long prompts
need many chunks, and many chunks exhaust the free supply.

Disaggregation delivers $M \cdot m/\tau(m)$ on the decode fleet, but requires
$M_p = \lambda\,\tau(n_p)$ additional prefill machines. Per-machine throughput is

$$ \frac{M \cdot m/\tau(m)}{M + \lambda\tau(n_p)} $$

which falls as $n_p$ grows. So both degrade with prompt length, by different
mechanisms — chunking through chunk-supply exhaustion, disaggregation through fleet
size — and the crossover is where the two curves meet.

{{sec:9-practical-example}} finds chunking ahead at 200, 900 and 3200-token prompts and
disaggregation ahead at 12,000. **The crossover sits inside the range real products
operate in**, which is why this is a live architectural choice rather than a solved
one.

One term the model omits favours disaggregation. {{cite:patel2023splitwise}}'s result
is that prefill and decode machines need not be the same generation: prefill wants
compute and decode wants bandwidth, so a heterogeneous fleet buys each phase what it
needs. A uniform-machine comparison understates disaggregation for exactly that reason.

## 7. Internal Mechanics

**What continuous batching requires of the memory system.** Sequences joining and
leaving at arbitrary steps means the KV cache cannot be a contiguous per-batch tensor.
{{cite:kwon2023pagedattention}}'s block-based allocation is what makes it implementable,
which is why paging and continuous batching arrived together and are often discussed as
one thing. They are separable — paging without continuous batching still fixes
fragmentation — but the reverse is impractical.

**Why the step-time floor makes chunking work.** The flatness of $\tau$ below the
balance point is not an approximation; it is what "memory-bound" means. The weights are
read regardless, and the arithmetic on 32 tokens versus 295 tokens both complete before
the read does. Chunked prefill is therefore not stealing capacity from decode — it is
using capacity that had no other use.

**The chunk cliff in practice.** {{sec:9-practical-example}} shows a 263-token chunk
costing exactly a decode step and a 512-token chunk costing **1.84×**. A stack that
defaults its chunk size without reference to batch size and hardware will land on the
wrong side of that cliff for some configurations, and the symptom — decode latency
rising when prefill load rises — looks like contention rather than misconfiguration.

**Why the stall victim is never the cause.** The request whose prompt triggered the
stall is *waiting for its own prefill* and experiences it as normal time-to-first-token.
The thirty-two sequences that lost a step experience it as an unexplained pause in
their token stream. So the metric that would reveal the problem — inter-token latency
variance — is on the requests that did nothing wrong, and the request that caused it
looks fine.

**Where the scheduler's admission decision sits.** Continuous batching makes
admission a per-step decision rather than a per-batch one, which changes what
backpressure means. Under static batching, refusing a request is a clean act: it never
entered a batch. Under continuous batching there is always a slot opening within a few
steps, so the natural behaviour is to admit everything and let the batch grow until
memory runs out -- at which point the failure is
{{eq:batch-times-context-is-the-budget}}'s, arriving as an allocator error rather than
as a policy decision. A serving system needs an explicit admission bound derived from
the token-slot budget, and it needs it precisely because continuous batching removed
the natural one.

**KV transfer as the disaggregation cost.** Shipping a 3200-token prompt's cache is 419
MB, which is 0.47 ms over a fast interconnect and roughly 6.5 seconds over a 1 Gb/s
network link. {{cite:zhong2024distserve}}'s gains assume the former.
**Disaggregation is an intra-rack technique**, and a design that spans racks needs the
transfer priced explicitly.

**Interaction with the length cap.** Capping generation length raises static-batching
utilisation substantially — {{sec:9-practical-example}} measures 13.4% to 57.0% for a
280-token cap — which is why deployments without continuous batching lean on caps. The
cap is buying batching efficiency with output quality, and that trade should be made
knowingly rather than inherited from a default.

## 8. Implementation

The first listing measures what static batching wastes on a realistic length
distribution.

```python {tier=A name=cf1}
"""Static batching wastes most of its capacity on a queue of unequal-length jobs.

ch:inf-cpu-gpu established that batching is what makes a GPU worth using. This listing
asks what it costs, and the answer depends entirely on how the batch is formed.

A STATIC batch runs to completion together: every sequence occupies a slot until the
LONGEST one finishes. With generation lengths varying by an order of magnitude, most
slots spend most of their time computing padding
(eq:static-batching-pays-for-the-longest).

CONTINUOUS batching lets a finished sequence leave and a waiting one take its slot
immediately. This listing measures the gap, and finds it is larger than the length
variation alone suggests.
"""
import math

# Generation length distribution: mostly short, occasionally very long.
# (length in tokens, share of requests)
LENGTHS = [
    (40, 0.31),
    (110, 0.27),
    (280, 0.21),
    (700, 0.13),
    (1800, 0.06),
    (4200, 0.02),
]
BATCHES = [1, 4, 8, 16, 32, 64]
STEP_MS = 18.0        # time for one decode step across the batch

mean_len = sum(l * p for l, p in LENGTHS)
print("Generation length distribution.")
print()
print(f"{'length':>10}{'share':>9}{'cumulative':>13}")
print("-" * 34)
c = 0.0
for l, p in LENGTHS:
    c += p
    print(f"{l:>10}{p:>9.0%}{c:>13.0%}")
print()
print(f"mean {mean_len:.0f} tokens, max {LENGTHS[-1][0]} tokens, "
      f"ratio {LENGTHS[-1][0] / LENGTHS[0][0]:.0f}x")


def expected_max(n):
    """E[max of n independent draws] from the length distribution."""
    total = 0.0
    prev = 0.0
    cum = 0.0
    for l, p in LENGTHS:
        cum += p
        total += l * (cum ** n - prev)
        prev = cum ** n
    return total


print()
print()
print("Static batching: every slot is held until the LONGEST sequence in the")
print("batch finishes. Useful work is the sum of lengths; paid work is batch")
print("size times the maximum.")
print()
print(f"{'batch':>8}{'E[max len]':>13}{'useful tokens':>16}"
      f"{'paid slots':>13}{'utilisation':>14}")
print("-" * 66)
static = {}
for b in BATCHES:
    emax = expected_max(b)
    useful = b * mean_len
    paid = b * emax
    static[b] = (emax, useful, paid, useful / paid)
    print(f"{b:>8}{emax:>13.0f}{useful:>16.0f}{paid:>13.0f}"
          f"{useful / paid:>14.1%}")

print()
print()
print("What that does to throughput. A step serves the whole batch, so throughput")
print("is batch size over step time -- but only for slots doing real work.")
print()
print(f"{'batch':>8}{'nominal tok/s':>16}{'effective tok/s':>18}"
      f"{'vs batch 1':>13}")
print("-" * 57)
eff_static = {}
for b in BATCHES:
    nominal = b / (STEP_MS / 1000.0)
    effective = nominal * static[b][3]
    eff_static[b] = (nominal, effective)
    print(f"{b:>8}{nominal:>16.0f}{effective:>18.0f}"
          f"{effective / eff_static[1][1]:>12.1f}x")

print()
print()
print("Continuous batching: a finished sequence leaves and a queued one takes")
print("its slot on the next step. Every slot is always doing real work.")
print()
print(f"{'batch':>8}{'static tok/s':>15}{'continuous tok/s':>19}"
      f"{'gain':>9}{'utilisation':>14}")
print("-" * 66)
cont = {}
for b in BATCHES:
    continuous = b / (STEP_MS / 1000.0)
    cont[b] = continuous
    print(f"{b:>8}{eff_static[b][1]:>15.0f}{continuous:>19.0f}"
          f"{continuous / eff_static[b][1]:>8.1f}x{1.0:>14.1%}")

print()
print()
print("Where the gap comes from: it grows with batch size, because E[max] grows")
print("with the number of draws while the mean does not.")
print()
print(f"{'batch':>8}{'E[max]/mean':>15}{'static util':>14}{'gap':>9}")
print("-" * 48)
for b in BATCHES:
    print(f"{b:>8}{static[b][0] / mean_len:>15.2f}{static[b][3]:>14.1%}"
          f"{cont[b] / eff_static[b][1]:>8.1f}x")

print()
print()
print("The same comparison against a tighter length distribution -- what happens")
print("if you cap generation length.")
print()
print(f"{'cap':>8}{'mean len':>11}{'E[max] at b=32':>17}"
      f"{'static util':>14}{'continuous gain':>18}")
print("-" * 70)
caps = {}
for cap in (4200, 1800, 700, 280):
    sub = [(min(l, cap), p) for l, p in LENGTHS]
    m = sum(l * p for l, p in sub)
    total = 0.0
    prev = 0.0
    cum = 0.0
    for l, p in sub:
        cum += p
        total += l * (cum ** 32 - prev)
        prev = cum ** 32
    util = m / total
    caps[cap] = (m, total, util)
    print(f"{cap:>8}{m:>11.0f}{total:>17.0f}{util:>14.1%}{1.0 / util:>17.1f}x")

print()
print()
print("And the latency side, which is what static batching is usually defending.")
print("A request arriving mid-batch must wait for the batch to finish forming.")
print()
print(f"{'batch':>8}{'static wait ms':>17}{'continuous wait ms':>21}"
      f"{'saved':>10}")
print("-" * 58)
ARRIVAL_RATE = 22.0     # requests per second
wait = {}
for b in BATCHES:
    # Static: wait to fill the batch, then wait for the previous batch to drain.
    fill = (b - 1) / (2.0 * ARRIVAL_RATE) * 1000.0
    drain = static[b][0] * STEP_MS
    sw = fill + drain
    # Continuous: wait only for a slot to free, on average one sequence's length
    # divided by the batch size.
    cw = mean_len * STEP_MS / b
    wait[b] = (sw, cw)
    print(f"{b:>8}{sw:>17.0f}{cw:>21.0f}{sw - cw:>10.0f}")

print(f"""
The length distribution is the whole problem. The mean generation is
{mean_len:.0f} tokens and the longest is {LENGTHS[-1][0]} --
{LENGTHS[-1][0] / mean_len:.0f} times the mean, arriving on {LENGTHS[-1][1]:.0%} of
requests.

Under static batching that {LENGTHS[-1][1]:.0%} sets the cost of the whole batch. At
batch {32} the expected maximum length is {static[32][0]:.0f} tokens against a mean of
{mean_len:.0f}, so every one of {32} slots is held for {static[32][0]:.0f} steps while
the average sequence needs {mean_len:.0f} -- a utilisation of
**{static[32][3]:.1%}** (eq:static-batching-pays-for-the-longest).

**Nearly nine tenths of the capacity is computing padding.** Not idle -- computing,
on real silicon, at full power, producing nothing.

The throughput table converts that into the number that matters. Nominal throughput at
batch {32} is {eff_static[32][0]:.0f} tokens a second; effective is
{eff_static[32][1]:.0f}. The gap is exactly the utilisation, and it is why a
benchmark run on equal-length sequences reports numbers a real deployment never sees.

Continuous batching closes it by construction. When a sequence finishes, its slot goes
to a waiting request on the next step, so utilisation is {1.0:.0%} and throughput is
the nominal figure. At batch {32} that is **{cont[32] / eff_static[32][1]:.1f} times**
static batching, for the same hardware and the same requests.

The gap table shows the shape, and it is the uncomfortable one. E[max]/mean rises with
batch size -- {static[4][0] / mean_len:.2f} at batch {4},
{static[64][0] / mean_len:.2f} at batch {64} -- because the maximum of more draws
reaches further into the tail. So **static batching gets worse exactly as you batch
harder**, which is the same shape ch:inf-cpu-gpu found for KV traffic and
ch:sd-retrieval-agents found for fan-out.

At batch {64} static batching achieves {static[64][3]:.1%} utilisation and continuous
batching is {cont[64] / eff_static[64][1]:.1f} times better. The gain is not a
constant factor; it grows with the thing you want to increase.

The cap table is the other lever, and it is worth pricing because teams reach for it.
Capping generation at {280} tokens raises static utilisation from
{caps[4200][2]:.1%} to {caps[280][2]:.1%} -- most of the way to continuous batching's
{1.0:.0%} -- at the cost of truncating every request that needed more.

**Capping length is a way of buying batching efficiency with output quality**, and the
table says how much of each. It is the right trade for some surfaces and a silent
semantic failure for others, which is ch:sd-architecture's missing instrument
appearing in a serving configuration.

The last table addresses the argument static batching usually gets defended with:
predictable latency. It does not survive. At batch {32}, static batching makes a
request wait {wait[32][0]:.0f}ms -- {31 / (2.0 * ARRIVAL_RATE) * 1000.0:.0f}ms to fill
the batch plus {static[32][0] * STEP_MS:.0f}ms to drain it -- against
{wait[32][1]:.0f}ms for continuous batching, a factor of
{wait[32][0] / wait[32][1]:.0f}.

Static batching is worse on throughput AND worse on latency. **The only thing it is
better at is being simple to implement**, which is a real advantage and the reason it
persists, but it should be chosen knowingly rather than inherited.""")
```

## 9. Practical Example

A generation length distribution spanning 40 to 4200 tokens with a mean of 384:

```
   batch   E[max len]   useful tokens   paid slots   utilisation
------------------------------------------------------------------
       1          384             384          384        100.0%
       4         1015            1536         4059         37.8%
       8         1528            3071        12222         25.1%
      16         2163            6142        34615         17.7%
      32         2866           12285        91717         13.4%
      64         3536           24570       226305         10.9%
```

At batch 32 the expected maximum is **2866** tokens against a mean of **384**, so
utilisation is **13.4%** ({{eq:static-batching-pays-for-the-longest}}). **Nearly nine
tenths of the capacity is computing padding.**

```
   batch   static tok/s   continuous tok/s     gain   utilisation
------------------------------------------------------------------
       1             56                 56     1.0x        100.0%
       4             84                222     2.6x        100.0%
       8            112                444     4.0x        100.0%
      16            158                889     5.6x        100.0%
      32            238               1778     7.5x        100.0%
      64            386               3556     9.2x        100.0%
```

Continuous batching is **7.5×** static at batch 32 and **9.2×** at batch 64. **The gain
grows with the thing you want to increase** ({{eq:continuous-batching-gain-grows}}),
because E[max]/mean rises from 2.64 at batch 4 to 9.21 at batch 64.

Capping length is the alternative lever:

```
     cap   mean len   E[max] at b=32   static util   continuous gain
----------------------------------------------------------------------
    4200        384             2866         13.4%              7.5x
    1800        336             1723         19.5%              5.1x
     700        248              700         35.4%              2.8x
     280        160              280         57.0%              1.8x
```

A 280-token cap takes static utilisation from **13.4%** to **57.0%** — most of the way
to continuous batching — by truncating every request that needed more. **Capping buys
batching efficiency with output quality.**

And the latency defence does not survive: at batch 32, static batching makes a request
wait **52,295 ms** (705 ms to fill plus 51,591 ms to drain) against **216 ms** for
continuous batching — a factor of **242**. Static batching is worse on throughput *and*
worse on latency; its only advantage is implementation simplicity.

The second listing turns to phase interference.

```python {tier=A name=cf2}
"""Prefill and decode interfere, and there are two opposite fixes.

Continuous batching solves the unequal-length problem. It does not solve a second one:
a prefill and a decode want completely different things from the same step.

Prefill is compute-bound and long; decode is memory-bound and short. Put a whole
prefill in a step and every decode sharing that step waits for it, so one large prompt
stalls every sequence in flight (eq:prefill-stalls-decode).

There are two known fixes and they are opposites. cite:agrawal2023sarathi CHUNKS the
prefill so each step carries a small piece alongside the decodes -- exploiting the fact
that a decode step has idle compute, which ch:inf-cpu-gpu measured. cite:zhong2024distserve
and cite:patel2023splitwise SEPARATE the phases onto different machines entirely.

This listing measures both against the same workload and finds neither dominates.
"""
import math

# From ch:inf-cpu-gpu: a step is max(weight traffic / bandwidth, FLOPs / peak).
WEIGHT_BYTES = 14.0e9
BANDWIDTH = 3.35e12
PEAK = 9.89e14
PARAMS = 7.0e9
BALANCE = PEAK / BANDWIDTH        # tokens per step to become compute-bound

PROMPTS = [200, 900, 3200, 12000]
BATCH = 32
PREFILL_RATE = 8.0                # prefills arriving per second
DECODE_MACHINES = 8.0


def step_ms(tokens):
    """Milliseconds for one step carrying `tokens` tokens of work."""
    t_mem = WEIGHT_BYTES / BANDWIDTH
    t_flop = 2.0 * PARAMS * tokens / PEAK
    return max(t_mem, t_flop) * 1000.0


DECODE_MS = step_ms(BATCH)
print("A step costs max(weight traffic / bandwidth, FLOPs / peak).")
print("Weights fix the floor at %.1f ms; compute overtakes it past %.0f tokens."
      % (WEIGHT_BYTES / BANDWIDTH * 1000.0, BALANCE))
print()
print(f"{'tokens in step':>16}{'step ms':>10}{'bound by':>12}"
      f"{'headroom to balance':>22}")
print("-" * 62)
for t in (32, 96, 200, 295, 400, 900):
    b = "memory" if t < BALANCE else "compute"
    print(f"{t:>16}{step_ms(t):>10.2f}{b:>12}"
          f"{max(0, BALANCE - t):>22.0f}")

print()
print("A decode step at batch %d carries %d tokens and is memory-bound, so"
      % (BATCH, BATCH))
print("%.0f tokens of compute headroom sit idle every step." % (BALANCE - BATCH))

print()
print()
print("Colocated and unchunked: a prefill runs as its own step.")
print()
print(f"{'prompt tokens':>15}{'prefill ms':>13}{'decode steps lost':>20}"
      f"{'tokens lost':>14}")
print("-" * 64)
stall = {}
for p in PROMPTS:
    ms = step_ms(p)
    steps = ms / DECODE_MS
    stall[p] = (ms, steps, steps * BATCH)
    print(f"{p:>15}{ms:>13.1f}{steps:>20.1f}{steps * BATCH:>14.0f}")

print()
print()
print("Sustained decode throughput at %.1f prefills per second." % PREFILL_RATE)
print()
ideal = BATCH / (DECODE_MS / 1000.0)
print(f"{'prompt tokens':>15}{'prefill duty':>15}{'decode tok/s':>15}"
      f"{'vs ideal':>11}")
print("-" * 58)
colocated = {}
for p in PROMPTS:
    duty = min(0.99, step_ms(p) / 1000.0 * PREFILL_RATE)
    tp = ideal * (1.0 - duty)
    colocated[p] = (duty, tp)
    print(f"{p:>15}{duty:>15.1%}{tp:>15.0f}{tp / ideal:>11.1%}")

print()
print()
print("Chunked prefill: put a chunk of prefill INTO a decode step, using the")
print("idle compute. A step of batch+chunk tokens costs the same as a step of")
print("batch tokens, as long as the total stays under the balance point.")
print()
print(f"{'chunk':>8}{'tokens/step':>14}{'step ms':>10}{'vs decode-only':>17}"
      f"{'prefill tok/step':>19}")
print("-" * 70)
CHUNKS = [64, 128, 256, 263, 512, 1024]
chunkcost = {}
for k in CHUNKS:
    ms = step_ms(BATCH + k)
    chunkcost[k] = ms
    print(f"{k:>8}{BATCH + k:>14}{ms:>10.2f}{ms / DECODE_MS:>16.2f}x"
          f"{k:>19}")

print()
print()
print("Choosing the chunk at the balance point, so prefill is free.")
print()
FREE_CHUNK = int(BALANCE - BATCH)
print(f"free chunk size: {FREE_CHUNK} tokens (batch {BATCH} + chunk = "
      f"{BATCH + FREE_CHUNK} tokens, balance {BALANCE:.0f})")
print()
print(f"{'prompt tokens':>15}{'chunks needed':>16}{'steps to prefill':>19}"
      f"{'decode tok/s':>15}{'vs ideal':>11}")
print("-" * 78)
chunked = {}
for p in PROMPTS:
    n_chunks = int(math.ceil(p / float(FREE_CHUNK)))
    # Each chunk rides a step that was happening anyway, at no extra step time.
    # The only cost is that prefill capacity is bounded by steps per second.
    steps_per_sec = 1000.0 / step_ms(BATCH + FREE_CHUNK)
    chunks_needed_per_sec = PREFILL_RATE * n_chunks
    if chunks_needed_per_sec <= steps_per_sec:
        eff = step_ms(BATCH + FREE_CHUNK)
    else:
        # Demand exceeds what free chunks can carry; the excess costs real time.
        excess = chunks_needed_per_sec - steps_per_sec
        eff = step_ms(BATCH + FREE_CHUNK) * (1.0 + excess / steps_per_sec)
    tp = BATCH / (eff / 1000.0)
    chunked[p] = (n_chunks, eff, tp)
    print(f"{p:>15}{n_chunks:>16}{n_chunks:>19}{tp:>15.0f}{tp / ideal:>11.1%}")

print()
print()
print("Disaggregated: prefill runs on separate machines. Decode machines never")
print("see a prefill, but the KV cache must be shipped between them.")
print()
KV_PER_TOKEN_MB = 0.131
LINK_GB_S = 900.0
print(f"KV per prompt token: {KV_PER_TOKEN_MB:.3f} MB, link {LINK_GB_S:.0f} GB/s")
print()
print(f"{'prompt tokens':>15}{'KV to ship MB':>16}{'ship ms':>10}"
      f"{'decode tok/s':>15}{'vs ideal':>11}")
print("-" * 68)
disagg = {}
for p in PROMPTS:
    kv_mb = p * KV_PER_TOKEN_MB
    ship_ms = kv_mb / (LINK_GB_S * 1000.0) * 1000.0
    disagg[p] = (kv_mb, ship_ms, ideal)
    print(f"{p:>15}{kv_mb:>16.1f}{ship_ms:>10.2f}{ideal:>15.0f}"
          f"{1.0:>11.1%}")

print()
print()
print("Machines required, since disaggregation buys its throughput with hardware.")
print()
print(f"{'prompt tokens':>15}{'prefill machines':>19}{'total machines':>17}"
      f"{'vs colocated':>15}")
print("-" * 68)
fleet = {}
for p in PROMPTS:
    load = step_ms(p) / 1000.0 * PREFILL_RATE
    fleet[p] = load
    print(f"{p:>15}{load:>19.2f}{DECODE_MACHINES + load:>17.2f}"
          f"{(DECODE_MACHINES + load) / DECODE_MACHINES:>14.2f}x")

print()
print()
print("Throughput per machine -- the comparison that decides it.")
print()
print(f"{'prompt tokens':>15}{'colocated':>12}{'chunked':>11}"
      f"{'disaggregated':>16}{'best':>16}")
print("-" * 72)
winner = {}
for p in PROMPTS:
    co = colocated[p][1] / DECODE_MACHINES
    ch = chunked[p][2] / DECODE_MACHINES
    di = disagg[p][2] / (DECODE_MACHINES + fleet[p])
    opts = {"colocated": co, "chunked": ch, "disaggregated": di}
    best = max(opts, key=lambda k: opts[k])
    winner[p] = (co, ch, di, best)
    print(f"{p:>15}{co:>12.0f}{ch:>11.0f}{di:>16.0f}{best:>16}")

print(f"""
The headroom table is the mechanism, and it comes straight from ch:inf-cpu-gpu. A step
is bound by weights until it carries {BALANCE:.0f} tokens. A decode step at batch
{BATCH} carries {BATCH} -- so **{BALANCE - BATCH:.0f} tokens of compute capacity sit
idle in every decode step the system runs.**

That idle capacity is what cite:agrawal2023sarathi spends. A step carrying
{BATCH} decodes plus a {FREE_CHUNK}-token prefill chunk takes
{chunkcost[263] if 263 in chunkcost else step_ms(BATCH + FREE_CHUNK):.2f}ms against a
decode-only step's {DECODE_MS:.2f}ms -- **the same time**, because both are still
memory-bound. The prefill is genuinely free until the balance point, and expensive
immediately after: a {512}-token chunk costs
{chunkcost[512] / DECODE_MS:.2f} times a decode step.

**Chunk size is not a tuning parameter with a smooth curve. It has a cliff at the
balance point**, and the correct value is {FREE_CHUNK} tokens for this batch --
computed, not searched.

The stall table shows what happens without that. A {PROMPTS[2]}-token prompt runs as
its own step costing {stall[PROMPTS[2]][0]:.1f}ms, which is
{stall[PROMPTS[2]][1]:.1f} decode steps during which every one of {BATCH} sequences
produces nothing. **One prompt costs {stall[PROMPTS[2]][2]:.0f} tokens other users were
waiting for** (eq:prefill-stalls-decode), and the victim is never the request that
caused it.

The duty table turns that into sustained throughput: at {PROMPTS[2]}-token prompts,
prefill occupies {colocated[PROMPTS[2]][0]:.1%} of step time and decode falls to
{colocated[PROMPTS[2]][1] / ideal:.1%} of ideal. At {PROMPTS[3]} tokens the device is
{colocated[PROMPTS[3]][0]:.0%} prefill. **A colocated server with long prompts is a
prefill server that occasionally decodes.**

Chunking recovers most of it: {chunked[PROMPTS[2]][2] / ideal:.1%} of ideal at
{PROMPTS[2]} tokens against colocated's {colocated[PROMPTS[2]][1] / ideal:.1%}, and
{chunked[PROMPTS[3]][2] / ideal:.1%} at {PROMPTS[3]} tokens against
{colocated[PROMPTS[3]][1] / ideal:.1%}.

Disaggregation recovers all of it -- decode machines run at {1.0:.0%} by construction --
at the cost of shipping {disagg[PROMPTS[2]][0]:.1f} MB per {PROMPTS[2]}-token prompt,
taking {disagg[PROMPTS[2]][1]:.2f}ms over a fast link. That is cheap, and it is cheap
**only over a fast link**; the same design across a datacentre network is a different
calculation entirely.

The per-machine table is the honest comparison, because disaggregation buys its
{1.0:.0%} by adding hardware. At {PROMPTS[0]} tokens the winner is
{winner[PROMPTS[0]][3]}; at {PROMPTS[3]} tokens it is {winner[PROMPTS[3]][3]}.

**Neither approach dominates**, and the crossover sits inside the range real products
operate in. The choice turns on three things these tables make explicit: prompt length,
interconnect speed, and whether the fleet can be heterogeneous at all --
cite:patel2023splitwise's contribution being precisely that prefill and decode machines
need not be the same generation, which this listing's uniform-machine model cannot
express and which moves the comparison in disaggregation's favour.

A design review that presents either as settled has skipped the measurement.""")
```

From {{ch:inf-cpu-gpu}}'s arithmetic, a step is flat until the balance point:

```
  tokens in step   step ms    bound by   headroom to balance
--------------------------------------------------------------
              32      4.18      memory                   263
              96      4.18      memory                   199
             200      4.18      memory                    95
             295      4.18      memory                     0
             400      5.66     compute                     0
             900     12.74     compute                     0
```

**A decode step at batch 32 leaves 263 tokens of compute capacity idle, every step.**

Unchunked, a prefill runs as its own step:

```
  prompt tokens   prefill ms   decode steps lost   tokens lost
----------------------------------------------------------------
            200          4.2                 1.0            32
            900         12.7                 3.0            98
           3200         45.3                10.8           347
          12000        169.9                40.6          1301
```

A 3200-token prompt costs **347 tokens** other users were waiting for
({{eq:prefill-stalls-decode}}); a 12,000-token prompt costs **1301**.

At 8 prefills per second, decode throughput falls to **63.8%** of ideal at 3200-token
prompts and **1.0%** at 12,000 — **a colocated server with long prompts is a prefill
server that occasionally decodes.**

The chunk cliff:

```
   chunk   tokens/step   step ms   vs decode-only   prefill tok/step
----------------------------------------------------------------------
      64            96      4.18            1.00x                 64
     128           160      4.18            1.00x                128
     256           288      4.18            1.00x                256
     263           295      4.18            1.00x                263
     512           544      7.70            1.84x                512
    1024          1056     14.95            3.58x               1024
```

A 263-token chunk is **free**; a 512-token chunk costs **1.84×** a decode step
({{eq:chunk-size-has-a-cliff}}). **Chunk size is computed, not searched:**
$k^\star = 295 - 32 = 263$.

```mermaid {#fig:chunkcliff caption="Step time is flat below the balance point, so a prefill chunk sized to the idle headroom rides a decode step at no cost. Past the balance point every additional token is paid in full."}
flowchart LR
  A["decode step<br/>32 tokens, 4.18ms"] --> B["add 263 prefill tokens"]
  B --> C["295 tokens, 4.18ms<br/>still memory-bound"]
  C --> D["prefill is FREE"]
  A --> E["add 512 prefill tokens"]
  E --> F["544 tokens, 7.70ms<br/>compute-bound"]
  F --> G["1.84x a decode step"]
```

And the comparison that decides it — throughput **per machine**:

```
  prompt tokens   colocated    chunked   disaggregated            best
------------------------------------------------------------------------
            200         925        957             953         chunked
            900         860        957             945         chunked
           3200         610        957             916         chunked
          12000          10        622             818   disaggregated
```

**Neither dominates.** Chunking wins to 3200 tokens; disaggregation wins at 12,000,
where chunk supply is exhausted (46 chunks per prefill at 8 prefills a second exceeds
the step rate). The crossover sits inside the range real products operate in, and
{{cite:patel2023splitwise}}'s heterogeneous-fleet result — which this uniform-machine
model cannot express — moves it further in disaggregation's favour.

## 10. Production Considerations

Use continuous batching. The gain is **7.5×** at batch 32 on this length distribution
and grows with batch; there is no configuration in which static batching is preferable
on merit.

Compute the chunk size rather than tuning it: $k^\star = F/B - m$. Recompute it when
either the hardware or the operating batch size changes, because it depends on both.

Measure your generation length distribution and publish E[max]/mean at your batch size.
It predicts continuous batching's benefit exactly, and it is the number that explains
why a vendor's reported gain does or does not reproduce.

Monitor inter-token latency variance on requests that are *not* prefilling. That is
where prefill stalls show up, and no per-request metric on the causing request will
reveal them.

Decide chunking versus disaggregation from a per-machine comparison at your prompt
length distribution, not from a throughput comparison. Throughput alone always favours
disaggregation, because disaggregation buys throughput with machines.

Price the KV transfer at your actual interconnect. Disaggregation is an intra-rack
technique at 419 MB per 3200-token prompt; across a slow link the arithmetic reverses.

Derive the admission bound from the token-slot budget and enforce it explicitly.
Continuous batching removes the natural backpressure that batch boundaries provided,
and without a replacement the system admits until it runs out of memory.

Treat generation length caps as a quality decision with a throughput justification, and
record which surfaces they apply to. Under continuous batching the throughput
justification largely evaporates, so caps inherited from a static-batching era should
be re-examined.

## 11. Common Mistakes

**Benchmarking on equal-length sequences.** Reports the nominal throughput a real
deployment never sees, and the gap is exactly the utilisation the benchmark assumed
away.

**Tuning chunk size by search.** It has a cliff, not a curve; compute it.

**Comparing chunking and disaggregation on throughput.** Disaggregation buys throughput
with machines; compare per machine.

**Attributing prefill stalls to contention.** They are a scheduling decision, and the
metric that reveals them is on the wrong requests.

**Keeping a length cap after adopting continuous batching.** Its throughput
justification was static batching's utilisation problem.

**Assuming a vendor's continuous-batching gain transfers.** It is E[max]/mean for
*their* distribution at *their* batch size.

## 12. Failure Modes

**Chunk cliff crossing after a batch increase.** Raising batch size lowers $k^\star$;
a fixed chunk configuration silently crosses the cliff and decode latency rises.

**Chunk supply exhaustion under long prompts.** Many chunks per prefill at high arrival
rate exceeds the free-step supply, and chunking degrades toward colocated behaviour
with no configuration change.

**Head-of-line blocking from an unchunked path.** One code path that bypasses chunking
— a priority request, an admin endpoint, a batch job using a different client — reintroduces
the full stall for every sequence in flight, and it does so intermittently enough to be
blamed on load.

**KV transfer over the wrong link.** Disaggregation deployed across racks pays seconds
rather than milliseconds, appearing as inexplicable time-to-first-token.

**Starvation under continuous batching.** Long sequences can be indefinitely
deprioritised by a steady stream of short ones, converting a throughput win into a
fairness failure — the liveness problem {{ch:sd-async}} flagged.

**Unbounded admission.** With no batch boundary to push back at, the scheduler
admits until the token-slot budget is exhausted, and the failure surfaces as an
allocator error rather than as the capacity decision it actually is.

## 13. Alternatives

**Static batching with tight length bucketing.** Group sequences by predicted length so
each batch is homogeneous. Recovers much of the utilisation without continuous
batching's implementation cost, and depends on length prediction being good.

**Priority-aware continuous batching.** Reserve slots for latency-sensitive traffic so
long generations cannot occupy the whole batch. Addresses the starvation failure at the
cost of some utilisation, and the reservation is the same kind of per-surface decision
{{ch:sd-routing-caching}} made about cache thresholds.

**Prefill-only and decode-only pools without full disaggregation.** Route by phase
within one fleet rather than across two. Captures some of the benefit without the KV
transfer.

**Speculative decoding.** Raises tokens per step, which by
{{eq:chunk-size-has-a-cliff}} consumes the same headroom chunking uses — so the two
compete for one resource and should be tuned together.

**Do nothing, at low load.** At batch 1 static and continuous batching are identical,
and a lightly-loaded deployment gains nothing from either. The complexity is worth it
only where the batch is real, and a team serving a handful of concurrent users is
better served by a smaller model or a cheaper device than by a scheduler.

## 14. Evaluation

Report utilisation, not just throughput. Nominal throughput times utilisation is what
users get, and only the product is meaningful.

Benchmark on your own length distribution and report E[max]/mean alongside. A
throughput number without the distribution it was measured on is not comparable to
anything.

Measure inter-token latency percentiles, not just time-to-first-token and total. Prefill
stalls live entirely in the gaps between tokens.

Test the chunk configuration at the operating batch size, and re-test after any batch
change. The cliff moves when the batch does.

For disaggregation, measure the KV transfer time on the real interconnect under real
concurrency, not from the link's rated bandwidth.

## 15. Advanced Concepts

The expected-maximum model treats generation lengths as independent, which they are not:
requests arrive in correlated bursts — a document-summarisation feature produces several
long generations together. Correlated lengths make static batching *worse* than the
independent model predicts, because a batch is more likely to be all-long or all-short
than the independent case allows, and the all-long batches are the expensive ones.

The chunk cliff analysis assumes the balance point is a sharp threshold. Real kernels
have a transition region: as token count approaches the balance point, achieved
bandwidth and achieved FLOP/s both fall short of peak in ways that depend on tiling and
occupancy. The practical consequence is that $k^\star$ computed from datasheet constants
is an upper bound, and a small safety margin — chunking at 80% of $k^\star$ — costs
little and avoids the cliff under measurement error.

The disaggregation model also assumes prefill machines and decode machines can be
sized independently, which holds only if the ratio between prefill and decode load is
stable. It is not: a product whose users shift from short questions to document uploads
changes the ratio within a day, and a fleet split 8:1 for one mix is wrong for the
other. Chunking has no equivalent exposure -- the same machines serve whatever mix
arrives -- which is a real operational advantage that does not appear in any
throughput comparison. **Disaggregation trades flexibility for peak efficiency**, and
that trade is worth naming because the efficiency is measurable in a benchmark and the
flexibility is only measurable in an incident.

There is an unexplored composition between chunking and continuous batching's slot
management. A chunk occupies compute headroom that varies with the *current* batch
occupancy, which under continuous batching changes every step. So $k^\star$ is not
constant at run time — it is $I^\star - m_t$ for the instantaneous batch $m_t$. A
scheduler that sized chunks dynamically against current occupancy would extract more
free prefill than a fixed chunk size, and as far as the author is aware no published
system does this.

## 16. Connection to Previous Chapters

{{eq:batch-is-the-mechanism-not-an-optimisation}} from {{ch:inf-cpu-gpu}} said the batch
is what makes the hardware worth using. This chapter says what stops the batch from
being full, and both answers are scheduling.

{{eq:decode-is-bandwidth-bound}} is what makes {{eq:chunk-size-has-a-cliff}} work: the
idle compute chunking spends exists precisely because decode is memory-bound.

{{eq:variance-not-mean-drives-wait}} from {{ch:sd-async}} and
{{eq:static-batching-pays-for-the-longest}} are the same expected-maximum arithmetic,
one in a queue and one in a batch.

{{eq:batch-times-context-is-the-budget}} from {{ch:inf-gpu-memory}} bounds the batch;
this chapter determines how much of that bound is usable.

## 17. Exercises

1. For a length distribution of your choice, compute E[max]/mean at batch 8, 32 and 128.
   What continuous-batching gain does that predict?

2. Derive $k^\star$ for a device with 400 TFLOP/s and 1.6 TB/s at batch 48. How much
   does it change if the batch doubles?

3. Extend the first listing so lengths are correlated within a batch. How much worse is
   static batching?

4. Find the prompt length at which chunk supply exhausts, for a given arrival rate and
   $k^\star$. Compare to the listing's crossover.

5. Model dynamic chunk sizing against instantaneous batch occupancy. How much additional
   free prefill does it capture?

## 18. Interview Questions

1. Our GPU shows 100% utilisation and our throughput is a seventh of the vendor
   benchmark. Explain.

2. Why does static batching get worse as you increase the batch size?

3. A user reports their token stream pausing for half a second. What would you look at?

4. What chunk size should we use for prefill, and how did you get it?

5. Should we chunk or disaggregate? What do you need to know to answer?

6. We adopted continuous batching and now get intermittent out-of-memory errors under
   load we used to handle. What changed, and what is missing?

## 19. Research Questions

1. How much additional free prefill does dynamic chunk sizing against instantaneous
   batch occupancy capture, and is the scheduling complexity worth it?

2. How correlated are generation lengths within an arrival burst, and what does that do
   to the expected-maximum model?

3. What is the right fairness policy for continuous batching that bounds starvation of
   long generations without materially costing utilisation?

4. Where exactly is the chunking-versus-disaggregation crossover on heterogeneous
   fleets, where {{cite:patel2023splitwise}}'s result applies?

## 20. Chapter Summary

Static batching holds every slot until the longest sequence finishes, so utilisation is
$\bar{L}/\mathbb{E}[\max]$ ({{eq:static-batching-pays-for-the-longest}}) — **13.4%** at
batch 32 on a distribution spanning 40 to 4200 tokens. Nearly nine tenths of the
capacity computes padding.

It degrades as you batch harder, from **37.8%** at batch 4 to **10.9%** at batch 64, so
continuous batching's gain *grows*: **2.6×** to **9.2×**
({{eq:continuous-batching-gain-grows}}). It is also worse on latency — **52,295 ms**
against **216 ms** at batch 32.

Prefill and decode interfere. A 3200-token prefill run as its own step costs **347
tokens** other users were waiting for, and at 8 prefills a second decode falls to
**63.8%** of ideal ({{eq:prefill-stalls-decode}}).

Because a decode step is memory-bound, **263 tokens of compute headroom sit idle every
step**. A chunk that size rides free; a 512-token chunk costs **1.84×**
({{eq:chunk-size-has-a-cliff}}). Chunk size is computed as $F/B - m$, not searched.

Chunking and disaggregation are opposite fixes and **neither dominates**: per machine,
chunking wins at 200, 900 and 3200-token prompts and disaggregation at 12,000.

Both halves of this chapter describe capacity that already exists and is not being
used: slots held by finished sequences, and arithmetic units idle during a
memory-bound step. Neither is recovered by buying anything. They are recovered by
scheduling — deciding what occupies a slot and what rides along in a step — which is
why the largest serving gains of the last few years came from systems work rather
than from silicon.

That is also why the gains are so variable between deployments. A system already
running short, uniform generations at low batch has little idle capacity to recover
and will measure almost nothing from either technique. The published numbers are
real and they are properties of a workload, and the first question to ask of any of
them is what distribution they were measured on.

Carry forward: **the batch's problem is scheduling, not hardware**, and **compute the
chunk size — it has a cliff, not a curve**.

## 21. Further Reading

- {{cite:kwon2023pagedattention}} — the paging that makes continuous batching
  implementable.
- {{cite:agrawal2023sarathi}} — chunked prefill; mixing the phases.
- {{cite:zhong2024distserve}} — disaggregation; separating them, with goodput under two
  latency constraints.
- {{cite:patel2023splitwise}} — heterogeneous fleets, the term this chapter's model
  cannot express.

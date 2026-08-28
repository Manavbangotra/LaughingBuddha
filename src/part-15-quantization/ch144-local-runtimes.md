---
id: q-runtimes
number: 144
part: XV
tier: full
status: draft
requires: [q-memory-math, q-gguf, q-activation-kv]
provides: [scheduling-policy, continuous-batching, chunked-prefill-scheduling,
           preemption-policy, wasted-work-fraction, regime-not-runtime]
citations: [kwon2023pagedattention, pope2022inference, dao2022flash,
            leviathan2023speculative, dettmers2023case4bit]
---

## 1. Learning Objectives

By the end of this chapter you will be able to explain the throughput difference
between scheduling policies without reference to kernels; identify the stall that
continuous batching introduces and price the fix; describe the three things a
runtime can do when the cache fills, and compute which is right from a hardware
ratio; measure **wasted work** rather than preemption count; and choose a runtime
by **which regime its defaults assume** rather than by a feature list.

## 2. Why This Matters

**This chapter is the one most at risk of being a catalogue**, and a catalogue of
current runtime features would be stale before it was useful. So it is about the
two things that do not change: **how work is scheduled**, and **what happens when
memory runs out.**

{{sec:9-practical-example}} simulates three scheduling policies on one workload,
one set of hardware constants, and identical kernels. **Static batching reaches
2.58 requests per second; continuous batching reaches 5.00** — **1.94×**, from
nothing but reusing a finished sequence's slot immediately.

**And the latency difference is larger than the throughput one.** Time to first
token: **34,701 ms** under static batching against **130 ms** under continuous —
a factor of **267**, because a request arriving one step after a batch starts
waits for the whole batch to drain.

**Then the problem continuous batching creates.** Admitting a request means
prefilling its prompt, and while that runs every decoding sequence is stopped.
Measured worst stall: **1,472 ms** — one and a half seconds during which every
streaming user's output freezes because somebody else sent a long prompt.

**Chunked prefill takes it to 57 ms**, a factor of **26** — and costs TTFT,
**130 → 187 ms**. **A real trade with no dominant side**, exposed as a dial.

**Then the behaviour that defines a runtime under load: what it does when the
cache fills.** Rejection completes **487 of 500** requests. Recompute completes
all 500 and **throws away 33.9% of all prefill work it performed**. Swap completes
all 500 and wastes **none**, paying transfers instead.

> **Preemption is not a feature to optimise. It is a symptom of being
> under-provisioned** — at four times the cache, the recompute policy preempts
> **zero** times and wastes **0.0%**.

{{maturity:ESTABLISHED}} Continuous batching, paged allocation.
{{maturity:MATURE}} Chunked prefill, preemption policies.
{{maturity:EMERGING}} Disaggregated prefill and decode.

## 3. Prerequisites

{{ch:q-memory-math}} for the capacity this chapter schedules against, and for
{{eq:admission-control-memory}}; {{ch:q-gguf}} for
{{eq:memory-bound-crossover}}, which supplies the step-time model;
{{ch:q-activation-kv}} for the allocator these policies depend on.

## 4. Intuitive Explanation

### Runtimes differ in scheduling, not in kernels

Fused attention, quantized matmuls and paged caches are largely shared across
stacks — often literally the same code. **What differs is when work runs.**

```text
   policy                 throughput   TTFT p50   worst stall
   ────────────────────   ──────────   ────────   ───────────
   static batching              2.58     34,701 ms      9,751 ms
   continuous batching          5.00        130 ms      1,472 ms
   continuous + chunked         4.92        187 ms         57 ms
```

**Three rows, one workload, identical hardware constants.**

### Why static batching wastes the machine

A batch advances as a unit and retires as a unit. So a request generating thirty
tokens **sits in the batch until the request generating two thousand finishes** —
its slot occupied and idle for the rest.

**The machine runs a smaller effective batch than it was configured for**, and
that is where the throughput went. Not into slower steps: into steps carrying
fewer live sequences than they could have.

**And the latency consequence is severe**: 34,701 ms to first token, because a
request arriving one step after a batch starts is not admitted until that batch
drains.

### Continuous batching, and the stall it introduces

Retire each sequence when it finishes; admit the next immediately. The batch stays
full, throughput rises **1.94×**, and TTFT falls by a factor of 267.

**But admitting means prefilling**, and prefill is compute-bound and proportional
to prompt length. **While it runs, every decoding sequence is stopped.**

A 32k-token prompt is over three seconds of prefill. **Every streaming user sees
their output freeze**, because of a request that is not theirs.

Measured worst stall: **1,472 ms.**

### Chunked prefill, and what it costs

Process 512 prompt tokens, run a decode step, process the next 512. **The long
prompt takes the same total time and no longer stops anyone.**

```text
   chunk    TTFT p50    TTFT p99    worst stall
   ─────    ────────    ────────    ───────────
     128      1,955 ms    7,299 ms         14 ms
     512        187 ms    1,704 ms         57 ms
   2,048        132 ms    1,048 ms        228 ms
   8,192        130 ms      934 ms        910 ms
```

**Chunking trades the prefilling request's latency for everyone else's
smoothness.** There is no setting best on both columns — which is why it is a
configuration parameter, and why **leaving it at a default is a decision about
your users that somebody else made.**

### And then the cache fills

{{ch:q-memory-math}} computes how many sequences fit. It does not say what happens
when that is exceeded — and it will be, because output lengths are unknown at
admission time.

**Three answers**, and every stack implements one:

```text
   policy      completed   throughput   preempts   wasted work
   ─────────   ─────────   ──────────   ────────   ───────────
   reject          487/500        2.40         13          0.0%
   recompute       500/500        2.02         30         33.9%
   swap            500/500        2.37         11          0.0%
```

**Read the completed column first**, because rejection is not answering the same
question: it *drops* the sequences it evicts, so its throughput is measuring a
service that fails requests.

> **A system under memory pressure reporting good latency is often reporting the
> latency of the requests it did not drop.**

**Between the two that keep everything**, the difference is what they pay.
Recompute discards the victim's cache and re-runs its prefill — simple, no host
transfers, and **33.9% of all prefill work performed was later thrown away**.
Swap moves the cache to host memory and back, wasting no work and paying
transfers.

**Which wins is a hardware ratio, not a design opinion**: re-prefilling $P$ tokens
against moving $P$ tokens of cache twice.

### Preemption is a symptom, not a feature

```text
   cache tokens   throughput   preempts   wasted work
   ────────────   ──────────   ────────   ───────────
         60,000         1.59         65         49.7%
        120,000         2.02         30         33.9%
        240,000         2.41          4          8.7%
        480,000         2.51          0          0.0%
```

**Every byte recovered by the previous chapters — paged allocation, GQA, KV
quantization — appears here as a preemption that does not happen.** The memory
chapters and the scheduling chapters are about the same resource.

## 5. Formal Explanation

### 5.1 Why static batching loses throughput

With batch size $B$ and per-sequence output lengths $O_i$, a static batch occupies
the machine for $\max_i O_i$ steps while carrying, at step $t$, only
$|\{i : O_i \ge t\}|$ live sequences. The mean occupancy is

$$ \bar{B}_{\text{static}} = \frac{\sum_i O_i}{\max_i O_i} $$ (eq:static-occupancy)

**For a heavy-tailed $O$, $\max_i O_i \gg \mathbb{E}[O_i]$**, so
$\bar{B}_{\text{static}} \ll B$. Continuous batching keeps occupancy at $B$ by
construction, giving

$$ \frac{\text{throughput}_{\text{cont}}}{\text{throughput}_{\text{static}}} \approx \frac{B}{\bar{B}_{\text{static}}} = \frac{B\max_i O_i}{\sum_i O_i} $$ (eq:scheduling-policy)

**{{eq:scheduling-policy}} predicts the measured 1.94× from the output-length
distribution alone** — no property of the hardware enters.

### 5.2 The prefill stall

Admitting a request with prompt length $P$ stalls every decoding sequence for

$$ t_{\text{stall}} = \frac{P}{R_{\text{prefill}}} $$ (eq:prefill-stall)

Chunking at size $C$ bounds it:

$$ t_{\text{stall}}^{\text{chunked}} = \frac{C}{R_{\text{prefill}}} $$ (eq:chunked-prefill-scheduling)

**and lengthens the prefilling request's own TTFT** to

$$ \text{TTFT} = \frac{P}{R_{\text{prefill}}} + \left\lceil\frac{P}{C}\right\rceil \cdot t_{\text{step}} $$ (eq:chunk-ttft-cost)

**{{eq:chunked-prefill-scheduling}} and {{eq:chunk-ttft-cost}} move in opposite
directions in $C$**, which is the Pareto trade the sweep measures: at $C = 128$
the stall is 14 ms and TTFT p50 is 1,955 ms; at $C = 8192$, 910 ms and 130 ms.

### 5.3 The preemption decision

Evicting a sequence with prompt $P$ and current cache length $\ell$ costs

$$ c_{\text{recompute}} = \frac{P}{R_{\text{prefill}}}, \qquad c_{\text{swap}} = \frac{2\ell}{R_{\text{swap}}} $$ (eq:preemption-policy)

so swapping wins when

$$ \frac{2\ell}{R_{\text{swap}}} < \frac{P}{R_{\text{prefill}}} \quad\Longleftrightarrow\quad \frac{\ell}{P} < \frac{R_{\text{swap}}}{2 R_{\text{prefill}}} $$ (eq:swap-versus-recompute)

**{{eq:swap-versus-recompute}} is a hardware ratio and a workload ratio**, not a
design preference. At the measured constants — 9,000 prefill tokens/s and 60,000
cache-tokens/s of transfer — the threshold is $\ell/P < 3.3$, satisfied for most
requests, which is why swap wins here.

**On a machine with a slow host link, or with GQA making $\ell$ small relative to
$P$, the inequality flips.** A stack that implements both and chooses by
measurement is doing the right thing.

### 5.4 Wasted work is the metric

$$ w = \frac{\text{prefill tokens processed} - \text{prefill tokens that survived}}{\text{prefill tokens processed}} $$ (eq:wasted-work-fraction)

**{{eq:wasted-work-fraction}} is what preemption count does not tell you.**
Measured: 30 preempts under recompute cost **33.9%** of all prefill work; 11
preempts under swap cost **0%**. **The counts are comparable and the damage is
not.**

> **IMPORTANT:** {{eq:wasted-work-fraction}} is the quantity that says the machine
> is running to stand still, and almost no serving stack exposes it. Preemption
> counters are common; wasted-work fractions are not.

### 5.5 Which regime a runtime assumes

$$ \text{value of scheduling} \;\propto\; (B - 1) $$ (eq:regime-not-runtime)

**{{eq:regime-not-runtime}} is trivial and decisive.** At $B = 1$, continuous
batching has no second sequence to admit, chunked prefill has nobody to protect,
and preemption never occurs — so every scheduler feature is pure overhead. At
$B = 64$ the scheduling policy *is* the performance.

**Runtimes are not fast or slow. They implement different points, and the point
follows from the $B$ they were built for.**

## 6. Mathematical Foundation

### 6.1 The occupancy loss, worked

For output lengths lognormal with the measured median 154 and a 4096 cap, the mean
is roughly 260 and the batch maximum over 64 draws is in the low thousands. From
{{eq:static-occupancy}}:

$$ \bar{B}_{\text{static}} \approx \frac{64 \times 260}{2000} \approx 8.3 $$

against a configured 64 — **an eightfold occupancy loss**, of which the measured
1.94× is the part that survives the fact that static batching also spends less
time prefilling per unit of decode.

**The prediction is directional and the mechanism is exact**: heavy-tailed output
lengths make batch-granular retirement expensive, and the heavier the tail the
worse it gets.

### 6.2 Why the stall is worse than the throughput number suggests

A 1,472 ms stall on a stream producing a token every 22 ms is **67 tokens'
worth of silence**. The throughput columns barely notice — it is one gap in a run
of many thousands of steps — but every user connected at that moment experiences
it.

$$ \text{stalls per user per minute} \approx \lambda \cdot P(\text{long prompt}) \cdot 60 $$

At 5 requests/s with 5% long prompts, **15 stalls per minute for every connected
user.** That is a product problem invisible in an aggregate throughput figure,
which is the general reason tail metrics belong beside means.

### 6.3 The provisioning relationship

From the cache sweep, wasted work falls **49.7% → 33.9% → 8.7% → 0.0%** as the
cache doubles twice. Throughput rises **1.59 → 2.51**, an improvement of **58%**
from memory alone with no change to any policy.

$$ \text{throughput} \approx \text{throughput}_{\infty}\,(1 - w) $$ (eq:wasted-work-throughput)

**{{eq:wasted-work-throughput}} closes the loop with {{part:15}}'s memory
chapters.** Cache quantization at 4× ({{ch:q-activation-kv}}) moves this workload
from the 120,000 row to the 480,000 row: **preemptions from 30 to zero, and
throughput up 24%** — a throughput gain delivered by a memory technique.

> **MATH NOTE:** {{eq:wasted-work-throughput}} makes the two halves of this part
> commensurable. Memory work and scheduling work are usually owned by different
> people and measured in different units, and this is the exchange rate: **memory
> recovered becomes throughput through the preemption term**, and only through it.

## 7. Internal Mechanics

```mermaid {#fig:scheduler caption="A serving scheduler as three decisions, each with a measured cost. When to retire a sequence (eq:scheduling-policy) is worth 1.94x. How to admit a new one (eq:chunked-prefill-scheduling) trades one request's first token against everyone else's smoothness. And what to do when the cache fills (eq:preemption-policy) is decided by a hardware ratio rather than by design taste — with the wasted-work fraction (eq:wasted-work-fraction), not the preemption count, as the quantity that matters."}
flowchart TB
    ARR["request arrives"] --> Q["queue"]
    Q --> ADM{{"room in cache?<br/>ch:q-memory-math"}}
    ADM -->|"yes"| PRE{{"prefill: whole prompt,<br/>or chunks of C?"}}
    PRE -->|"whole"| STALL["stalls all decoding<br/>eq:prefill-stall"]
    PRE -->|"chunked"| SMOOTH["bounded stall,<br/>slower own TTFT"]
    STALL --> DEC["decode step"]
    SMOOTH --> DEC
    DEC --> FIN{{"sequence finished?"}}
    FIN -->|"yes, retire now"| ADM
    FIN -->|"batch-granular"| WAIT["slot idle until<br/>the batch drains"]
    ADM -->|"no"| PMT{{"reject, recompute<br/>or swap?"}}
    PMT --> W["eq:wasted-work-fraction"]
```

### 7.1 What each design point optimises

| Built for | Batch | Wants | Gives up |
|---|---|---|---|
| one local user | 1 | startup time, footprint, single-stream latency | throughput it will never use |
| a few local users | 2–8 | simple continuous batching | tail-latency machinery |
| a shared server | 32–256 | occupancy, admission control, preemption | simplicity, startup time |
| a latency SLO | varies | chunked prefill, small batches | throughput |

**{{eq:regime-not-runtime}} is the whole table.** A runtime is a set of defaults
for one row, and running it in another row's regime produces the complaints people
have about it.

### 7.2 The questions to ask about a runtime

Not "is it fast", which is unanswerable without a regime. Instead:

1. **Does it retire sequences individually?** If not, {{eq:scheduling-policy}}
   says what it costs.
2. **Can it chunk prefill, and is the chunk size configurable?**
   {{eq:chunk-ttft-cost}} is a dial someone must set.
3. **What does it do when the cache fills**, and can you choose?
   {{eq:swap-versus-recompute}} is machine-dependent.
4. **Does it report the wasted-work fraction?** Almost none do, and it is the
   number that says whether the deployment is healthy.
5. **Does it separate residency from concurrent-prefill limits?**
   ({{ch:q-memory-math}}'s {{eq:admission-control-memory}}.)

### 7.3 Benchmark the regime, not the runtime

**Every stack looks similar at 30% cache utilisation.** The differences appear at
95%, which is where any economically-run deployment sits.

So a benchmark that never fills the cache has not tested the behaviour that will
define production — and one that sends requests sequentially cannot produce the
prefill stall, the preemption, or {{ch:q-memory-math}}'s overlapping-prefill
failure.

**A useful load test needs: concurrent arrivals, a realistic length distribution
including its tail, and enough offered load to fill the cache.** Anything less
measures the comfortable case.

### 7.4 Where the named runtimes actually sit

Naming products in a book is how a chapter goes stale, so this section names
**design points** and leaves the mapping to the reader, who can check it against
whatever exists when they read this.

**The single-user local point.** Batch 1, so {{eq:regime-not-runtime}} makes every
scheduling feature worthless. What matters instead is time from launch to first
token — model loading dominates — memory footprint on a machine that is also
running everything else, and the dequantization cost {{ch:q-gguf}} showed is
decisive on CPU. Such a runtime should ship weight-only quantized formats with
cheap unpacking, load lazily, and have almost no scheduler. **Complaints that it
"scales badly" are complaints that it is not the thing it is.**

**The shared-server point.** Batch 32–256, where {{eq:scheduling-policy}} is worth
1.94×, {{eq:chunked-prefill-scheduling}} is worth a factor of 26 in stall, and
{{eq:preemption-policy}} decides what happens on a bad afternoon. Such a runtime
needs a paged allocator, individual retirement, configurable chunking, a
preemption policy, and admission control on two dimensions. **Its startup time and
memory overhead are correctly deprioritised, and complaints about them are
complaints about the wrong axis.**

**The accelerator-specific point.** A runtime built for one memory architecture
can assume things the portable ones cannot — unified memory removes the
swap-versus-recompute question entirely, because there is nowhere to swap *to* and
nothing to transfer. {{eq:swap-versus-recompute}}'s ratio is not merely different
there; the question does not arise.

**The useful exercise is to read a runtime's defaults and infer its regime.**
Default batch size, whether chunking is on, whether preemption exists at all, and
what it does at startup will tell you which row of {{sec:7-internal-mechanics}}'s
table it was written for — and that is a more durable fact about it than any
benchmark.

## 8. Implementation

```python {tier=A name=scheduling-policy}
"""What actually separates one inference runtime from another: the scheduler.

Feature lists go stale. The scheduling policy does not, and it is where the
throughput differences between serving stacks come from -- not from kernels, which
are largely shared.

This listing simulates three policies on one workload with one set of hardware
constants (eq:scheduling-policy). Static batching collects a batch and runs it to
completion. Continuous batching admits a new request the moment a slot frees.
Chunked prefill additionally refuses to let a long prompt stall everyone else.

The metrics are the two that matter and disagree: aggregate throughput, and the
latency the individual user experiences.
"""
import numpy as np

rng = np.random.default_rng(277)

# Hardware constants, in the shape ch:q-gguf derived: decode is memory-bound and
# nearly batch-independent below the crossover, then compute-bound above it.
STEP_BASE_MS = 22.0          # one decode step, small batch
CROSSOVER_B = 48             # ch:q-gguf's eq:memory-bound-crossover
STEP_SLOPE_MS = 0.42         # per extra sequence above the crossover
PREFILL_TOK_PER_S = 9000.0   # prefill is compute-bound: tokens per second
MAX_SEQS = 64                # what the memory budget allows (ch:q-memory-math)


def step_ms(b):
    return STEP_BASE_MS + STEP_SLOPE_MS * max(0, b - CROSSOVER_B)


def workload(n, rate, seed=0):
    """`rate` is requests per second. Measuring throughput needs an overloaded
    system; measuring latency needs an underloaded one, so the two questions get
    two workloads."""
    r = np.random.default_rng(1000 + seed)
    prompt = np.clip(r.lognormal(6.4, 1.1, n).astype(int), 32, 32768)
    out = np.clip(r.lognormal(4.6, 0.8, n).astype(int), 8, 2048)
    arrive = np.cumsum(r.exponential(1000.0 / rate, n))
    return prompt, out, arrive


P, O, ARRIVE = None, None, None
N = 0


def use(rate, n=600):
    global P, O, ARRIVE, N
    P, O, ARRIVE = workload(n, rate)
    N = len(P)


def simulate(policy, chunk=None):
    """Returns per-request time-to-first-token and total latency, in ms."""
    now = 0.0
    nxt = 0                       # next request not yet admitted
    ttft = np.full(N, np.nan)
    done = np.full(N, np.nan)
    active, remaining = [], {}
    pending_prefill = []          # (index, tokens left to prefill)
    gaps, last_step = [], None    # interval between consecutive decode steps

    while nxt < N or active or pending_prefill:
        if policy == "static":
            # Fill a batch only when the previous one has fully drained.
            if not active and not pending_prefill:
                now = max(now, ARRIVE[nxt])
                take = []
                while nxt < N and len(take) < MAX_SEQS and ARRIVE[nxt] <= now:
                    take.append(nxt); nxt += 1
                if not take:
                    take = [nxt]; nxt += 1
                now += sum(P[i] for i in take) / PREFILL_TOK_PER_S * 1000.0
                for i in take:
                    ttft[i] = now - ARRIVE[i]
                    remaining[i] = O[i]
                active = take
        else:
            # Admit whenever there is room and a request has arrived.
            while (nxt < N and len(active) + len(pending_prefill) < MAX_SEQS
                   and ARRIVE[nxt] <= now):
                pending_prefill.append([nxt, int(P[nxt])]); nxt += 1
            if not active and not pending_prefill and nxt < N:
                now = max(now, ARRIVE[nxt])
                continue
            if pending_prefill:
                if policy == "continuous":
                    # A whole prompt is prefilled in one go, stalling decode.
                    i, tok = pending_prefill.pop(0)
                    now += tok / PREFILL_TOK_PER_S * 1000.0
                    ttft[i] = now - ARRIVE[i]
                    remaining[i] = O[i]
                    active.append(i)
                else:
                    # Chunked: one chunk per scheduler tick, then decode.
                    i, tok = pending_prefill[0]
                    piece = min(chunk, tok)
                    now += piece / PREFILL_TOK_PER_S * 1000.0
                    pending_prefill[0][1] -= piece
                    if pending_prefill[0][1] <= 0:
                        pending_prefill.pop(0)
                        ttft[i] = now - ARRIVE[i]
                        remaining[i] = O[i]
                        active.append(i)

        if active:
            if last_step is not None:
                gaps.append(now - last_step)
            now += step_ms(len(active))
            last_step = now
            finished = []
            for i in active:
                remaining[i] -= 1
                if remaining[i] <= 0:
                    done[i] = now - ARRIVE[i]
                    finished.append(i)
            for i in finished:
                active.remove(i)
        elif not pending_prefill and nxt >= N:
            break
    return ttft, done, now, np.array(gaps) if gaps else np.array([0.0])


use(40)
print(f"SATURATED: {N} requests offered far faster than any policy can serve,")
print(f"so the throughput column is each policy's ceiling. Prompt median "
      f"{int(np.median(P))} tokens, output median {int(np.median(O))}.")
print()
print(f"{'policy':>24}{'throughput':>14}{'vs static':>12}")
print("-" * 50)
sat = {}
for name, pol, ck in (("static batching", "static", None),
                      ("continuous batching", "continuous", None),
                      ("continuous + chunked", "chunked", 512)):
    _, _, span, _ = simulate(pol, ck)
    sat[name] = N / (span / 1000.0)
    print(f"{name:>24}{sat[name]:>14.2f}{sat[name]/sat['static batching']:>11.2f}x")

use(3.0)
print()
print()
print(f"UNDERLOADED at 3 requests per second, so these measure scheduling")
print("rather than queueing. 'Stall' is the gap between consecutive decode")
print("steps -- what a streaming user sees as the output pausing.")
print()
print(f"{'policy':>24}{'TTFT p50':>11}{'TTFT p99':>11}{'latency p50':>13}"
      f"{'stall p99':>13}{'worst stall':>13}")
print(f"{'':>24}{'ms':>11}{'ms':>11}{'ms':>13}{'ms':>13}{'ms':>13}")
print("-" * 85)
res = {}
for name, pol, ck in (("static batching", "static", None),
                      ("continuous batching", "continuous", None),
                      ("continuous + chunked", "chunked", 512)):
    t, d, span, g = simulate(pol, ck)
    res[name] = (np.nanpercentile(t, 50), np.nanpercentile(t, 99),
                 np.nanpercentile(d, 50), np.percentile(g, 99), g.max())
    print(f"{name:>24}{res[name][0]:>11.0f}{res[name][1]:>11.0f}"
          f"{res[name][2]:>13.0f}{res[name][3]:>13.0f}{res[name][4]:>13.0f}")

print()
print()
print("Chunk size is the dial between the prefilling request and everyone else.")
print("Underloaded, so these are scheduling effects.")
print()
print(f"{'chunk':>8}{'TTFT p50':>11}{'TTFT p99':>11}{'stall p99':>13}"
      f"{'worst stall':>13}")
print("-" * 56)
ck_rows = {}
for ck in (128, 512, 2048, 8192):
    t, d, span, g = simulate("chunked", ck)
    ck_rows[ck] = (np.nanpercentile(t, 50), np.nanpercentile(t, 99),
                   np.percentile(g, 99), g.max())
    print(f"{ck:>8}{ck_rows[ck][0]:>11.0f}{ck_rows[ck][1]:>11.0f}"
          f"{ck_rows[ck][2]:>13.0f}{ck_rows[ck][3]:>13.0f}")

st, co, chk = (res["static batching"], res["continuous batching"],
               res["continuous + chunked"])
print(f"""
Three policies, one workload, one set of hardware constants, and the same
kernels. Everything that differs between the rows is when work is scheduled.

Static batching reaches {sat['static batching']:.2f} requests per second.
Continuous batching reaches {sat['continuous batching']:.2f} --
{sat['continuous batching']/sat['static batching']:.2f}x more
(eq:scheduling-policy).

The mechanism is in the latency table. Under static batching a request generating
thirty tokens sits in the batch until the request generating two thousand
finishes, because the batch advances and retires as a unit. Its slot is occupied
and idle for most of that time, so the machine runs a smaller EFFECTIVE batch than
it was configured for. That is where the throughput went -- not into slower steps,
but into steps carrying fewer live sequences than they could have.

The latency consequence is brutal: TTFT p50 of {st[0]:.0f} ms against continuous
batching's {co[0]:.0f} ms, a factor of {st[0]/co[0]:.0f}. A request that arrives
one step after a batch starts waits for the entire batch to drain before it is
even admitted.

So continuous batching wins on both columns, which is why every serving stack
adopted it. The interesting question is what it does NOT fix, and that is the
stall column.

Admitting a request under continuous batching means prefilling its prompt, and
prefill is compute-bound and takes as long as the prompt is long. While it runs,
every sequence already decoding is stopped. The worst stall is {co[4]:.0f} ms --
one and a half seconds during which every streaming user's output freezes,
because somebody else sent a long prompt.

Chunked prefill interleaves: 512 prompt tokens, then a decode step, then the next
512. The long prompt takes the same total time and no longer stops anyone. The
worst stall falls to {chk[4]:.0f} ms, a factor of {co[4]/chk[4]:.0f}.

And it is not free, which is the part worth dwelling on. TTFT p50 rises from
{co[0]:.0f} to {chk[0]:.0f} ms and p99 from {co[1]:.0f} to {chk[1]:.0f}, because
the request being prefilled now has its prefill spread across many scheduler
ticks. Throughput drops slightly, from {sat['continuous batching']:.2f} to
{sat['continuous + chunked']:.2f}.

Chunking trades the prefilling request's latency for everyone else's smoothness.
That is a real trade with no dominant side, and the second table makes it a dial.

At a chunk of 128 the worst stall is {ck_rows[128][3]:.0f} ms and TTFT p50 is
{ck_rows[128][0]:.0f} ms. At 8192 the stall is {ck_rows[8192][3]:.0f} ms and TTFT
p50 is {ck_rows[8192][0]:.0f} ms. There is no setting that is best on both
columns, which is exactly why it is exposed as a configuration parameter rather
than chosen for you -- and why leaving it at a default is a decision about your
users that somebody else made.

Which is the durable way to think about inference runtimes, and it outlasts any
feature comparison. They are not fast or slow. They implement different points in
this space, and the point they implement follows from what they were built for.

A runtime designed for one user on a laptop has a batch size of one. Continuous
batching buys it nothing, because there is never a second sequence to admit.
Chunked prefill only delays its own first token, because there is nobody else to
protect. Every line of scheduler code is overhead against a fixed budget. Such a
runtime should optimise single-stream latency, startup time and memory footprint
-- and the ones that people run locally do exactly that.

A runtime designed to serve many users needs every row here, because at batch 1 it
is wasting the hardware and at batch 64 the scheduling policy IS the performance.
Its complexity is not gratuitous; it is the price of the
{sat['continuous batching']/sat['static batching']:.2f}x, and of the
{co[4]/chk[4]:.0f}x reduction in stall on top of it.

So the question to ask about a runtime is not which is faster. It is which regime
its defaults assume, and whether that is your regime. Running a server stack for a
single local user buys the overhead without the benefit. Running a single-stream
stack behind an API buys a fraction of the hardware you paid for. Both mistakes
are common, and neither shows up in a benchmark run the wrong way round.""")
```

The first listing is about scheduling when there is room. The second is about what
happens when there is not.

```python {tier=A name=preemption-policy}
"""What a runtime does when it runs out of cache, which is the thing it will do.

ch:q-memory-math computes how many sequences fit. It does not say what happens
when the number is exceeded, and that will happen: sequence lengths are unknown
when a request is admitted, so a scheduler that admits by current size will
eventually find the cache full with everything mid-generation.

There are three answers, and every serving stack implements one of them
(eq:preemption-policy). Refuse to admit until memory frees. Evict a running
sequence and RECOMPUTE its prefill later. Or evict it to host memory and SWAP it
back, paying the transfer instead of the recomputation.

This listing simulates all three and measures the quantity that distinguishes
them: how much of the machine's work was thrown away.
"""
import numpy as np

STEP_BASE_MS = 22.0
CROSSOVER_B = 48
STEP_SLOPE_MS = 0.42
PREFILL_TOK_PER_S = 9000.0
CACHE_TOKENS = 120_000          # total KV slots, from ch:q-memory-math
SWAP_TOK_PER_MS = 60.0          # host transfer rate, tokens of cache per ms


def step_ms(b):
    return STEP_BASE_MS + STEP_SLOPE_MS * max(0, b - CROSSOVER_B)


def workload(n, rate, seed=3):
    r = np.random.default_rng(seed)
    prompt = np.clip(r.lognormal(6.5, 1.2, n).astype(int), 32, 32768)
    out = np.clip(r.lognormal(5.0, 1.0, n).astype(int), 8, 4096)
    return prompt, out, np.cumsum(r.exponential(1000.0 / rate, n))


N = 500
P, O, ARRIVE = workload(N, 5.0)


def simulate(policy):
    now, nxt = 0.0, 0
    done = np.full(N, np.nan)
    active = {}                  # index -> [tokens generated, cache length]
    swapped = {}                 # index -> cache length, held in host memory
    queue = []
    prefill_tokens = 0.0         # total prefill work done, including redone
    useful_tokens = 0.0          # prefill work that was not later discarded
    preempts = 0

    def used():
        return sum(v[1] for v in active.values())

    while nxt < N or active or queue or swapped:
        while nxt < N and ARRIVE[nxt] <= now:
            queue.append(nxt); nxt += 1
        if not active and not queue and not swapped and nxt < N:
            now = max(now, ARRIVE[nxt]); continue

        # Admit if there is room. Swapped-out sequences come back first.
        if swapped and used() + max(swapped.values()) <= CACHE_TOKENS:
            i = min(swapped)
            now += swapped[i] / SWAP_TOK_PER_MS
            active[i] = [O[i] - (O[i] - active.get(i, [O[i], 0])[0]), swapped[i]]
            active[i] = [active[i][0], swapped.pop(i)]
        elif queue and used() + P[queue[0]] <= CACHE_TOKENS:
            i = queue.pop(0)
            now += P[i] / PREFILL_TOK_PER_S * 1000.0
            prefill_tokens += P[i]
            useful_tokens += P[i]
            active[i] = [O[i], int(P[i])]

        if not active:
            if queue or swapped:
                # Nothing fits and nothing is running: the cache is stuck.
                now += STEP_BASE_MS
            continue

        now += step_ms(len(active))
        for i in list(active):
            active[i][0] -= 1
            active[i][1] += 1
            if active[i][0] <= 0:
                done[i] = now - ARRIVE[i]
                del active[i]

        # Over budget? Apply the policy to the most recently admitted sequence.
        while used() > CACHE_TOKENS and active:
            victim = max(active, key=lambda k: active[k][1])
            preempts += 1
            if policy == "recompute":
                useful_tokens -= P[victim]      # that prefill is now wasted
                queue.insert(0, victim)
                del active[victim]
            elif policy == "swap":
                swapped[victim] = active[victim][1]
                now += active[victim][1] / SWAP_TOK_PER_MS
                del active[victim]
            else:                                # "reject": never over budget
                del active[victim]
                done[victim] = np.nan
    return done, now, prefill_tokens, useful_tokens, preempts


print(f"{N} requests at 5/s, cache holds {CACHE_TOKENS:,} tokens.")
print("Prompt median", int(np.median(P)), "tokens, output median",
      int(np.median(O)), "tokens.")
print()
print(f"{'policy':>14}{'completed':>11}{'throughput':>12}{'latency p50':>13}"
      f"{'latency p99':>13}{'preempts':>10}{'wasted work':>13}")
print("-" * 86)

res = {}
for pol in ("reject", "recompute", "swap"):
    d, span, pre, useful, pre_n = simulate(pol)
    ok = np.isfinite(d).sum()
    res[pol] = (ok, ok / (span / 1000.0), np.nanpercentile(d, 50),
                np.nanpercentile(d, 99), pre_n,
                1.0 - useful / max(pre, 1.0))
    print(f"{pol:>14}{ok:>11}{res[pol][1]:>12.2f}{res[pol][2]:>13.0f}"
          f"{res[pol][3]:>13.0f}{pre_n:>10}{res[pol][5]:>12.1%}")

print()
print()
print("How the answer moves with cache size. Recompute policy.")
print()
print(f"{'cache tokens':>14}{'throughput':>12}{'latency p99':>13}"
      f"{'preempts':>10}{'wasted work':>13}")
print("-" * 62)
grid = {}
for cap in (60_000, 120_000, 240_000, 480_000):
    CACHE_TOKENS = cap
    d, span, pre, useful, pre_n = simulate("recompute")
    ok = np.isfinite(d).sum()
    grid[cap] = (ok / (span / 1000.0), np.nanpercentile(d, 99), pre_n,
                 1.0 - useful / max(pre, 1.0))
    print(f"{cap:>14,}{grid[cap][0]:>12.2f}{grid[cap][1]:>13.0f}"
          f"{pre_n:>10}{grid[cap][3]:>12.1%}")

rj, rc, sw = res["reject"], res["recompute"], res["swap"]
print(f"""
The completed column is the first thing to read, because one policy is not
answering the same question as the others.

Rejection completes {rj[0]} of {N} requests. The other two complete
{rc[0]} and {sw[0]}. Rejection does not queue the sequence it evicts -- it drops
it -- so its throughput number is measuring a service that is failing requests,
and comparing it to the others on throughput alone would be a category error. It
is in the table to make that visible, because a system under memory pressure that
reports good latency is often reporting the latency of the requests it did not
drop (eq:preemption-policy).

Between the two policies that keep every request, the difference is what they pay
to free the memory.

Recompute discards the victim's cache and re-runs its prefill later. That is
simple, needs no host transfers, and throws work away: {rc[5]:.1%} of all prefill
tokens processed were later discarded and had to be done again. The machine did
that work and has nothing to show for it.

Swap moves the victim's cache to host memory and brings it back, paying a transfer
in each direction rather than a recomputation. It wastes {sw[5]:.1%} of prefill
work -- none, by construction -- and pays {sw[4]} transfers instead.

Which wins depends on a ratio you can compute rather than guess: the cost of
re-prefilling P tokens against the cost of moving P tokens of cache twice.
Prefill runs at {PREFILL_TOK_PER_S:,.0f} tokens per second and the transfer at
{SWAP_TOK_PER_MS * 1000:,.0f} cache-tokens per second, so swapping is cheaper
here -- and on a machine with a slow host link, or with grouped-query attention
making the cache small relative to the prefill, it would not be.

That is the useful form of the comparison. It is not a question about which
runtime is better designed; it is a hardware ratio, and a stack that supports both
and picks by measurement is doing the right thing.

The second table shows how the whole question dissolves with enough memory. At
{60_000:,} cache tokens the recompute policy wastes {grid[60_000][3]:.1%} of its
prefill work and preempts {grid[60_000][2]} times. At {480_000:,} it preempts
{grid[480_000][2]} times and wastes {grid[480_000][3]:.1%}.

So preemption is not a feature to optimise, it is a symptom of being under-
provisioned, and every ounce of memory recovered by the previous chapters --
paged allocation, grouped-query attention, KV quantization -- shows up here as
preemptions that do not happen. That is the connection worth carrying: the
memory chapters and the scheduling chapters are about the same resource, and
work done in one appears as a different quantity in the other.

Two consequences for choosing and configuring a runtime.

First, ask what it does under pressure, not what it does when comfortable. Every
stack looks similar at 30% cache utilisation. The differences appear at 95%, which
is where any economically-run deployment sits, and a benchmark that never fills
the cache has not tested the behaviour that will define production.

Second, watch the wasted-work fraction rather than the preemption count. Preempts
are cheap under swap and expensive under recompute, so the count alone does not
say whether anything is wrong. The fraction of prefill work discarded is the
quantity that tells you the machine is running to stand still.""")
```

## 9. Practical Example

**Scheduling, not kernels.** One workload, identical hardware constants: static
batching **2.58** requests per second, continuous batching **5.00** — **1.94×**
({{eq:scheduling-policy}}), from retiring sequences individually rather than by
batch.

**And the latency gap is larger than the throughput gap**: TTFT p50 **34,701 ms**
against **130 ms**, a factor of **267**, because
{{eq:static-occupancy}} makes a late-arriving request wait for a whole batch to
drain.

**Continuous batching introduces its own problem.** Worst stall **1,472 ms** —
{{eq:prefill-stall}}, one long prompt freezing every streaming user's output.
**Chunked prefill at 512 takes it to 57 ms, a factor of 26.**

**And it costs**: TTFT p50 **130 → 187 ms**, throughput **5.00 → 4.92**. The
sweep shows the Pareto shape — chunk 128 gives a **14 ms** stall for **1,955 ms**
TTFT; chunk 8192 gives **910 ms** for **130 ms**.

> **IMPORTANT:** {{eq:chunked-prefill-scheduling}} and {{eq:chunk-ttft-cost}} move
> in opposite directions in $C$, so **no setting is best on both columns.**
> Leaving the chunk size at a default is a decision about your users that somebody
> else made.

**Then what happens when the cache fills.** Rejection completes **487 of 500**;
recompute and swap complete all 500. **Read the completed column first** — a
system under pressure reporting good latency is often reporting the latency of the
requests it did not drop.

**Recompute threw away 33.9% of all prefill work it performed.** Swap wasted
**none**, paying 11 transfers instead. {{eq:swap-versus-recompute}}: at 9,000
prefill tokens/s against 60,000 cache-tokens/s of transfer, swapping wins whenever
$\ell/P < 3.3$ — **a hardware ratio, not a design opinion**, and it flips on a
machine with a slow host link.

**And preemption is a symptom.** Across cache sizes, wasted work falls **49.7% →
33.9% → 8.7% → 0.0%** and throughput rises **1.59 → 2.51**.

**{{eq:wasted-work-throughput}} is the exchange rate between the two halves of
this part**: 4× cache quantization ({{ch:q-activation-kv}}) moves this workload
from 30 preemptions to zero and lifts throughput 24% — **a throughput gain
delivered entirely by a memory technique.**

## 10. Production Considerations

**Load-test at the cache utilisation you will run at**, not at a comfortable one.

**Send concurrent arrivals with a realistic length tail.** Sequential benchmarks
cannot produce the failures that matter.

**Set the prefill chunk size deliberately** from
{{eq:chunked-prefill-scheduling}} and your latency target.

**Choose the preemption policy from {{eq:swap-versus-recompute}}**, measured on
your machine.

**Instrument the wasted-work fraction.** If your stack does not expose it, the
preemption counter plus prompt lengths reconstructs it.

**Treat rising preemptions as a provisioning signal**, not a tuning opportunity.

**Match the runtime to the regime.** A server stack for one local user is
overhead; a single-stream stack behind an API wastes the hardware.

## 11. Common Mistakes

**Comparing runtimes on throughput without stating the batch regime.**

**Benchmarking sequentially**, which cannot produce stalls or preemption.

**Never filling the cache in testing.**

**Reading preemption counts** instead of {{eq:wasted-work-fraction}}.

**Leaving the chunk size at a default** and then complaining about either TTFT or
stalls.

**Comparing a rejecting policy's throughput** to a queueing one's.

**Tuning the scheduler** when {{eq:wasted-work-throughput}} says the problem is
memory.

**Assuming the swap-versus-recompute answer transfers** between machines.

## 12. Failure Modes

**Output freezes intermittently for all users.** Cause:
{{eq:prefill-stall}} — an unchunked long prompt.

**Throughput far below the arithmetic.** Cause: batch-granular retirement
({{eq:static-occupancy}}), or preemption thrash
({{eq:wasted-work-throughput}}).

**Latency looks fine and users complain.** Cause: requests are being rejected and
excluded from the latency statistics.

**Throughput collapses as load rises.** Cause: recompute preemption — the machine
does more work and completes less.

**Benchmark does not reproduce production.** Cause: cache never filled, or
sequential arrivals.

**Chunking made TTFT worse and nobody expected it.** Cause:
{{eq:chunk-ttft-cost}}, which is the correct behaviour.

## 13. Alternatives

| Alternative | Trades | When |
|---|---|---|
| static batching | throughput, latency | offline batch jobs only |
| continuous batching | complexity | any multi-user serving |
| chunked prefill | own TTFT | when streaming smoothness matters |
| swap preemption | host bandwidth | when {{eq:swap-versus-recompute}} holds |
| recompute preemption | wasted work | simple stacks, small prompts |
| priority scheduling | fairness | mixed SLO classes |
| more memory | money | **usually the correct answer** |

**The last row is not a joke.** {{eq:wasted-work-throughput}} says that at 33.9%
wasted work, a third of the machine is producing nothing — and the cache sweep
shows that recovering memory removes the problem entirely rather than mitigating
it. **Scheduling sophistication is what you deploy when you cannot buy your way
out; it is not a substitute for adequate provisioning.**

## 14. Evaluation

**Report the batch regime with any throughput number.**

**Report TTFT, inter-token stall and total latency separately** — they respond to
different policies and can move in opposite directions.

**Report cache utilisation during the test.**

**Report completion rate**, so a rejecting policy cannot be mistaken for a fast
one.

**Report the wasted-work fraction**, or the preemption count and prompt lengths.

**Report the length distribution**, including its tail.

## 15. Advanced Concepts

**Disaggregated prefill and decode.** {{maturity:EMERGING}}
{{eq:prefill-stall}} exists because one machine does both phases, and they have
opposite bottlenecks ({{ch:q-memory-math}}). Running them on separate hardware
removes the stall structurally rather than bounding it — at the cost of shipping
the KV cache between machines, which is exactly {{eq:swap-versus-recompute}}'s
transfer term in a new place.

**Speculative decoding as a scheduling technique.** {{maturity:MATURE}}
{{cite:leviathan2023speculative}} spends the idle arithmetic {{ch:q-gguf}}
measured at 98.6%. It interacts with batching: at large batch there is no idle
arithmetic to spend, so **speculation and batching are substitutes rather than
complements** — which is not how they are usually presented.

**Priority and fairness.** {{maturity:MATURE}} Once sequences are retired
individually, admission order is a policy choice with real consequences.
Shortest-job-first minimises mean latency and starves long requests; the
scheduling literature's results apply directly and are rarely cited here.

**Wasted work as an SLO.** {{maturity:EMERGING}}
{{eq:wasted-work-fraction}} is a better health signal than utilisation, because a
machine can be 100% utilised and producing nothing. **It is not exposed by any
mainstream stack**, and reconstructing it from counters is a small amount of
work with a large diagnostic return.

**Regime detection.** {{maturity:RESEARCH FRONTIER}} A runtime could measure its
own $B$, cache utilisation and length distribution and select policies
accordingly, instead of exposing a dozen flags whose correct values are derivable.
**Nearly every constant in this chapter is computable at runtime.**

## 16. Connection to Previous Chapters

{{ch:q-memory-math}} computes the capacity this chapter schedules against, and
{{eq:wasted-work-throughput}} is the bridge: **its memory results become this
chapter's throughput through the preemption term.**
{{ch:q-gguf}}'s {{eq:memory-bound-crossover}} supplies the step-time model that
makes batching worth anything, and its idle-arithmetic measurement is what
speculative decoding spends.
{{ch:q-activation-kv}}'s paged allocator is what makes individual retirement
possible at all — a reservation-based cache cannot retire one sequence and admit
another of a different size.
Forward: {{ch:q-throughput-latency}} takes the latency/throughput tension this
chapter exhibits and makes it the subject; {{part:23}} owns deployment,
autoscaling and multi-machine serving.

## 17. Exercises

1. Derive {{eq:static-occupancy}} and compute it for output lengths drawn from
   your own workload.
2. From {{eq:scheduling-policy}}, predict the continuous-batching speedup for a
   workload with uniform output lengths. Why is it near 1?
3. Compute {{eq:chunk-ttft-cost}} for a 32k prompt at chunk sizes 256 and 4096.
4. Use {{eq:swap-versus-recompute}} to find the host bandwidth at which recompute
   becomes preferable on the listing's constants.
5. In `scheduling-policy`, make output lengths uniform. Which policy differences
   survive?
6. In `preemption-policy`, add a policy that preempts the sequence with the
   SHORTEST remaining output rather than the largest cache. Does it help?
7. Compute {{eq:wasted-work-throughput}}'s prediction for the 60,000-token cache
   row and compare with the measured throughput.
8. For a deployment you have: measure cache utilisation and preemption count, and
   reconstruct the wasted-work fraction.

## 18. Interview Questions

1. Why is continuous batching faster than static batching?
2. What does continuous batching make worse?
3. What does chunked prefill cost, and who pays?
4. A runtime reports excellent latency under load. What would you check?
5. Name the three things a runtime can do when the cache fills.
6. How would you choose between swap and recompute?
7. Why is preemption count a poor health metric?
8. Why does quantizing the KV cache improve throughput?
9. Why might speculative decoding help less at large batch?
10. How would you benchmark a serving stack so the result transfers to
    production?

## 19. Research Questions

1. {{eq:scheduling-policy}} predicts the speedup from the output-length
   distribution alone. How well does that hold on real workloads, and what does
   the residual attribute to?
2. {{eq:swap-versus-recompute}}'s threshold is computable at runtime. Would a
   policy that chooses per-victim beat either fixed policy, and by how much?
3. Speculation and batching both consume idle arithmetic. What is the optimal
   allocation between them as a function of offered load?
4. {{eq:wasted-work-fraction}} is not exposed by mainstream stacks. What
   proportion of production deployments are running with materially nonzero
   wasted work, and would knowing change provisioning?
5. Nearly every constant here is measurable at runtime. What would a
   self-configuring scheduler give up, and is the loss smaller than the tuning
   error it removes?

## 20. Chapter Summary

**Runtimes differ in scheduling, not in kernels.** One workload, identical
hardware constants: static batching **2.58** requests per second against
continuous batching's **5.00** — **1.94×** ({{eq:scheduling-policy}}) — and TTFT
**34,701 ms** against **130 ms**, a factor of **267**.
{{eq:static-occupancy}} is why: a batch that retires as a unit runs at a fraction
of its configured occupancy whenever output lengths are heavy-tailed.

**Continuous batching creates a stall it does not fix.** Admitting a request means
prefilling it, and prefill stops every decoding sequence: **1,472 ms** measured
({{eq:prefill-stall}}), sixty-seven tokens of silence for every connected user.
**Chunked prefill bounds it to 57 ms** and costs the prefilling request's own
first token — **130 → 187 ms**. {{eq:chunked-prefill-scheduling}} and
{{eq:chunk-ttft-cost}} move in opposite directions, so **no chunk size is best on
both columns.**

**And then the cache fills, which it will.** Rejection completed **487 of 500**
requests — **a system under pressure reporting good latency is often reporting the
latency of the requests it dropped.** Recompute completed all 500 and threw away
**33.9% of every prefill token it processed**. Swap completed all 500 and wasted
none, and {{eq:swap-versus-recompute}} says which is right from a hardware ratio
rather than from design taste.

**Preemption is a symptom of under-provisioning, not a feature to optimise.**
Across cache sizes, wasted work fell **49.7% → 33.9% → 8.7% → 0.0%** and
throughput rose **1.59 → 2.51**. {{eq:wasted-work-throughput}} makes this part's
two halves commensurable: **memory recovered becomes throughput through the
preemption term, and only through it.**

Which gives the way to think about runtimes that outlasts any feature list.
**{{eq:regime-not-runtime}}: the value of every scheduling technique is
proportional to $B - 1$.** A runtime is a set of defaults for one regime — and the
question is not which is faster, but **which regime its defaults assume, and
whether that is yours.** Running a server stack for one local user buys overhead
without benefit; running a single-stream stack behind an API buys a fraction of
the hardware. **Neither mistake shows up in a benchmark run the wrong way round.**

## 21. Further Reading

{{cite:kwon2023pagedattention}} for the allocator that makes individual retirement
possible, and note how much of its reported gain is scheduling rather than
computation — this chapter's simulation suggests most of it.
{{cite:pope2022inference}} for the analytical framing, developed properly in
{{ch:q-throughput-latency}}.
{{cite:dao2022flash}} for the kernel that these policies schedule, and which is
shared across every stack compared here.
{{cite:leviathan2023speculative}} for the technique that competes with batching
for the same idle arithmetic — a substitution that is rarely made explicit.
{{cite:dettmers2023case4bit}} as a reminder that the weight-precision decision,
which occupies most of this part, is upstream of everything measured here and
smaller than the scheduling decisions in its effect on serving throughput.

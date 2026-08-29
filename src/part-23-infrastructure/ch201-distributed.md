---
id: inf-distributed
number: 201
part: XXIII
tier: full
status: draft
requires: [parallelism-dimension-is-an-interconnect-decision, variance-not-mean-drives-wait,
           batch-times-context-is-the-budget, tensor-parallelism-is-in-node]
provides: [affinity-fights-balance, affinity-optimum-moves-with-load,
           parallel-group-is-one-failure-domain, smaller-domains-beat-more-replicas]
citations: [zhong2024distserve, patel2023splitwise, kwon2023pagedattention,
            shoeybi2019megatron]
---

## 1. Learning Objectives

By the end of this chapter you will be able to explain why prefix-aware routing and
load balancing are in direct conflict, and compute the affinity level that minimises
time-to-first-token; show why that optimum moves with fleet utilisation and with prefix
length, and why a fixed affinity weight is therefore wrong at one end of its own day;
compute the availability of a tensor-parallel group and show that it is the device
availability raised to the parallelism degree; quantify the redundancy that a given
parallelism degree forces; and argue for a topology on reliability grounds rather than
bandwidth grounds.

## 2. Why This Matters

{{ch:inf-parallelism}} treated a parallel group as a unit that works. A distributed
fleet is where that assumption stops holding, and where two questions appear that a
single-node analysis cannot pose.

The first is routing. A request whose prefix is already resident somewhere can skip
most of its prefill, so the router should send it there — but doing that concentrates
load, and {{ch:sd-async}} established what uneven load does to queueing.
{{sec:9-practical-example}} finds prefill saved rising from **203 ms** to **278 ms**
as affinity goes from zero to one, while queue wait rises from **1.6 ms** to
**204.8 ms** ({{eq:affinity-fights-balance}}). The optimum is **0.8** — neither
extreme — and it **moves with load**, sitting at 1.0 under light traffic and 0.0 under
heavy ({{eq:affinity-optimum-moves-with-load}}).

The second is failure. A tensor-parallel group stops if any member stops, so its
availability is the device availability raised to the degree. A 16-way group has
**16×** the downtime of one device ({{eq:parallel-group-is-one-failure-domain}}), and
with two replicas it is **256×** worse than a duplicated single device. Splitting the
same sixteen devices as 4-way tensor by 4-way pipeline is **16× better for the same
device count** ({{eq:smaller-domains-beat-more-replicas}}).

## 3. Prerequisites

You need {{eq:parallelism-dimension-is-an-interconnect-decision}} and
{{eq:tensor-parallelism-is-in-node}} from {{ch:inf-parallelism}} — this chapter adds
the reliability term those results omit.

{{eq:variance-not-mean-drives-wait}} from {{ch:sd-async}} supplies the queueing cost of
skew, which is what makes affinity routing a trade rather than a free win.

{{eq:batch-times-context-is-the-budget}} from {{ch:inf-gpu-memory}} bounds the prefix
cache each node can hold, and that bound is what makes replication expensive.

{{ch:as-state-machines}}'s replay requirement returns here as a hardware constraint.

## 4. Intuitive Explanation

Two things change when a model is served by many machines rather than one, and neither
is visible from a single-node benchmark. Both are consequences of the same shift: a
single node has one place to put a request and one thing that can break, and a fleet
has many of each. That turns two questions that did not previously exist into design
decisions with numbers attached.

**The first is that where a request goes matters.**

Most production traffic shares prefixes. A system prompt appears on every request. A
document being discussed appears on every request in that conversation. If the machine
you route to has already processed that prefix, its keys and values are sitting in
memory and the request skips straight to the interesting part.

So route requests to machines that have their prefix. Obviously.

Except that popular prefixes are popular. If forty percent of your traffic shares one
system prompt and you route all of it to the machine holding that prompt, that machine
now has forty percent of your traffic. {{ch:sd-async}} showed exactly what happens to
a queue when load stops being even, and it is not gradual.

The natural response is to replicate: put the popular prefix on several machines. That
works, and it costs memory — the same memory {{ch:inf-gpu-memory}} showed is the
binding constraint on batch size. Replicate a prefix onto every machine and every
machine spends cache capacity on it, evicting other prefixes to make room.

So there are three quantities in tension: how much prefill you save, how uneven your
load becomes, and how much cache you spend on duplicates. The router's affinity setting
trades all three, and there is an interior optimum that is not obvious and not stable —
it moves as the fleet fills up.

**The second thing that changes is what "a failure" means.**

On one machine, one machine fails. In a tensor-parallel group, the model's matrices are
split across every member, so losing any one of them stops the whole group. Sixteen
devices are not sixteen chances to keep working; they are sixteen chances to stop.

The arithmetic is the same product this book has met repeatedly — {{ch:ag-loop}}'s
chain, {{ch:sd-retrieval-agents}}'s fan-out — and it says that a group of sixteen
99.992%-available devices is 99.867% available. That sounds close. Over a year it is
eleven and a half hours of downtime against forty-four minutes.

Which means model parallelism costs redundancy, and the redundancy does not show up in
any throughput number. {{ch:inf-parallelism}}'s speedup tables all assume the group
works.

There is also a cost specific to this workload. When the group goes down it takes every
in-flight sequence's KV cache with it. Not just the requests it was about to serve —
the ones that were forty seconds into generating an answer. Without replay, those are
user-visible failures, and there are as many of them as your batch size.

## 5. Formal Explanation

**Routing.** Let a fleet have $N$ nodes and a prefix catalogue with prefix $i$ carrying
share $\sigma_i$ of traffic and $\ell_i$ tokens. An affinity parameter $\alpha \in [0,1]$
interpolates between uniform routing and pinning. Replication follows
$r(\alpha) = N - \alpha(N-1)$: uniform routing needs a prefix everywhere to be found,
pinning needs it in one place.

A node must hold share $r/N$ of the catalogue, so with cache capacity $C$ tokens the
resident set is the popularity-ordered prefixes fitting in $CN/r$. For a resident
prefix, the hit probability is

$$ h_i(\alpha) \;=\; \alpha \;+\; (1-\alpha)\frac{r(\alpha)}{N} $$

— affinity routing targets a holder; the residual routes uniformly and lands on one
with probability $r/N$. Expected prefill saved is $\sum_i \sigma_i h_i \ell_i \pi$ for
per-token prefill cost $\pi$.

Load skew grows with $\alpha$, and by {{eq:variance-not-mean-drives-wait}} the queueing
term is $\tau\rho/(1-\rho)$ with $\rho = \bar{\rho}\,s(\alpha)$. Total time to first
token is

$$ T(\alpha) \;=\; \underbrace{\textstyle\sum_i \sigma_i \ell_i \pi - \sum_i \sigma_i h_i(\alpha)\ell_i\pi}_{\text{prefill not saved}} \;+\; \underbrace{\frac{\tau\,\bar{\rho}s(\alpha)}{1 - \bar{\rho}s(\alpha)}}_{\text{queue}} $$ (eq:affinity-fights-balance)

The first term decreases in $\alpha$ and the second increases, so an interior minimum
exists whenever neither dominates.

The comparative static is what makes this operationally awkward. Differentiating the
queue term,

$$ \frac{\partial}{\partial\bar{\rho}}\left(\frac{\partial T}{\partial\alpha}\right) \;>\; 0 $$ (eq:affinity-optimum-moves-with-load)

because $\rho/(1-\rho)$ is convex: **the marginal cost of skew rises with base
utilisation.** So $\alpha^\star$ decreases as the fleet fills, and a router with a fixed
$\alpha$ is misconfigured at one end of its daily cycle.

**Failure.** Let a device have availability $a$. A parallel group of degree $n$ works
only if all members work:

$$ A(n) \;=\; a^n, \qquad \text{downtime} \;=\; (1 - a^n)\cdot 8760 \;\approx\; n(1-a)\cdot 8760 $$ (eq:parallel-group-is-one-failure-domain)

With $r$ independent replicas the service is down only if all are, giving
$1 - (1-a^n)^r$. Since $(1-a^n) \approx n(1-a)$, replicated downtime scales as $n^r$ —
**the degree enters the exponent's base and the replica count enters its power**, which
is why the r=2 column in {{sec:9-practical-example}} shows 256× at degree 16 rather
than 16×.

## 6. Mathematical Foundation

The topology consequence follows directly. Sixteen devices can be arranged as one
16-way tensor group, or as four 4-way tensor groups composed into a 4-stage pipeline.
Both use sixteen devices; their failure domains differ.

Duplicated, the annual downtime ratio is

$$ \frac{(1 - a^{16})^2}{(1 - a^{4})^2} \;\approx\; \left(\frac{16}{4}\right)^2 \;=\; 16 $$ (eq:smaller-domains-beat-more-replicas)

**Sixteen times better reliability for the same device count**, purchased entirely by
choosing a different arrangement of the same hardware. {{ch:inf-parallelism}}'s analysis
would have preferred the 16-way tensor group on a fast link, because it has no pipeline
bubble; this chapter says that preference has a reliability price the bandwidth
arithmetic does not contain.

Which dominates is a fleet-size question. For a handful of groups, the bandwidth term
wins — downtime of eleven hours a year on one group is tolerable. For a fleet of
hundreds, failures are continuous rather than occasional, and the topology that
minimises blast radius wins.

The repair-time lever is worth isolating because it is not a hardware purchase.
Availability is $a = M/(M + R)$ for mean time between failures $M$ and repair time $R$,
so $1 - a \approx R/M$ and the group's downtime scales **linearly in repair time**.
Halving detection-and-replacement time halves downtime at every degree simultaneously,
which no amount of redundancy does as cheaply. {{sec:9-practical-example}} finds that
cutting repair from 3.5 hours to 0.5 removes a replica at degree 64.

## 7. Internal Mechanics

**Why the affinity curve is not monotone.** {{sec:9-practical-example}} shows a dip at
$\alpha = 0.2$, and it is a real effect rather than a numerical artifact. At that
setting replication is 13 nodes — high enough that each node must hold most of the
catalogue and evicts, low enough that routing still misses 29% of the time.
**Partial replication is the worst of both**: most of the capacity cost of replicating,
little of the targeting benefit of pinning.

**Residency versus recency.** A router knows which node last *saw* a prefix; it does not
know whether that node still *holds* it. Eviction converts an affinity hit into a miss
silently, and the router's hit-rate estimate drifts from reality. Systems that report
residency back to the router do materially better, and the reporting is cheap — a
bloom filter per node, refreshed periodically.

**Affinity makes the fleet stateful.** A node's value depends on what it holds, so
draining one for maintenance costs more than its capacity share: the prefixes it held
must be re-prefilled elsewhere. {{ch:inf-kubernetes}} has to plan around this, and it is
the main reason model-serving fleets resist the cattle-not-pets treatment that
stateless services get.

**Why a failure destroys in-flight work.** The KV cache lives in device memory on the
group that failed. There is no checkpoint — {{cite:kwon2023pagedattention}}'s blocks are
allocated for speed, not durability — so recovery means re-prefilling the prompt and
regenerating every token produced so far. At batch 32 with a mean of 190 tokens already
generated, one failure costs 32 requests and 6,080 tokens of redone work.

**Why the queueing cost of skew is convex and the prefill benefit is not.** The two
terms in {{eq:affinity-fights-balance}} have different shapes, and that is what
guarantees an interior optimum rather than a corner one. Prefill saved is bounded above
by the total prefill cost -- there is only so much work to skip, and once the hit rate
reaches one there is nothing left to gain. Queueing cost is unbounded: as utilisation
on the hot node approaches one, wait time goes to infinity. So the benefit saturates
while the cost does not, and past some affinity the marginal trade is always
unfavourable. **Any system that trades a bounded benefit against an unbounded cost has
an interior optimum**, and recognising that shape is often faster than computing where
it lies.

**Disaggregation changes the failure story.** {{cite:zhong2024distserve}} and
{{cite:patel2023splitwise}} separate prefill and decode onto different machines, which
means a decode-machine failure does not lose the prefill work — the cache was shipped,
so it exists somewhere else. That is a reliability argument for disaggregation that
neither paper foregrounds and that {{ch:inf-batching}}'s throughput comparison does not
contain.

**Tensor-parallel groups cannot be partially replaced.** Because
{{cite:shoeybi2019megatron}}'s split makes every rank hold a slice of every layer, a
replacement device must load its slice and rejoin the collective before the group
serves again. Pipeline stages are independent enough to be swapped one at a time, which
is the mechanism behind {{eq:smaller-domains-beat-more-replicas}}.

## 8. Implementation

The first listing sweeps between load balancing and prefix pinning and finds the
optimum.

```python {tier=A name=da1}
"""Cache-aware routing and load balancing want opposite things.

Across a fleet, a request whose prefix is already resident on some node can skip most of
its prefill. So the router should send it there.

Two things stop that being free. Sending requests where their prefix lives concentrates
load, and ch:sd-async established what uneven load does to queueing. And a node's cache
is finite: routing uniformly means every node tries to hold every popular prefix, which
does not fit (eq:affinity-fights-balance).

This listing sweeps between pure load balancing and pure prefix pinning and finds the
optimum is at neither end -- and that it moves with fleet load and prefix length.
"""
NODES = 16
CACHE_TOKENS = 9000.0      # prefix tokens one node can keep resident

# Prefix popularity: a few shared system prompts and documents dominate.
# (label, share of requests, prefix tokens)
PREFIXES = [
    ("shared system prompt", 0.44, 1800),
    ("common document A",    0.14, 6200),
    ("common document B",    0.09, 5400),
    ("team template",        0.07, 2900),
    ("long tail",            0.26, 700),
]
PREFILL_MS_PER_TOK = 0.11
DECODE_MS = 4.18
CATALOGUE = sum(t for _, _, t in PREFIXES)


def replication(affinity):
    """How many nodes hold a given prefix. Uniform routing needs it everywhere;
    full affinity needs it in one place."""
    return NODES - affinity * (NODES - 1)


def resident(affinity):
    """Which prefixes fit on a node.

    At replication r each prefix sits on r of NODES nodes, so one node must hold
    a share r/NODES of the catalogue. Low affinity means high replication means
    every node tries to hold everything -- which does not fit. Popular prefixes
    are kept first.
    """
    budget = CACHE_TOKENS * NODES / replication(affinity)
    keep, used = set(), 0.0
    for label, share, toks in sorted(PREFIXES, key=lambda p: -p[1]):
        if used + toks <= budget:
            keep.add(label)
            used += toks
    return keep


def hit_rate_for(affinity, label, share):
    """P(this request lands on a node that has its prefix resident)."""
    if label not in resident(affinity):
        return 0.0
    r = replication(affinity)
    # Affinity routing targets a holder; the residual routes uniformly and hits
    # with probability r/NODES.
    return affinity + (1.0 - affinity) * (r / NODES)


def load_skew(affinity):
    """Busiest node's load over the mean. Pinning a prefix that carries 44% of
    traffic to a subset of nodes concentrates load on them."""
    return 1.0 + affinity * 2.5


def wait_ms(skew, rho_mean=0.28):
    rho = min(0.985, rho_mean * skew)
    return DECODE_MS * rho / (1.0 - rho)


print("A %d-node fleet, %.0f prefix tokens resident per node." % (NODES,
                                                                  CACHE_TOKENS))
print("The prefix catalogue totals %d tokens." % CATALOGUE)
print()
print(f"{'prefix':>24}{'share':>9}{'tokens':>9}{'prefill ms':>13}")
print("-" * 56)
for label, share, toks in PREFIXES:
    print(f"{label:>24}{share:>9.0%}{toks:>9}{toks * PREFILL_MS_PER_TOK:>13.0f}")
mean_prefill = sum(s * t for _, s, t in PREFIXES) * PREFILL_MS_PER_TOK
print()
print(f"mean prefill if nothing is cached: {mean_prefill:.0f} ms")

print()
print()
print("Sweeping affinity from pure load balancing to pure prefix pinning.")
print()
AFF = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
print(f"{'affinity':>10}{'replication':>13}{'hit rate':>11}"
      f"{'prefill saved':>15}{'load skew':>12}{'queue wait':>13}")
print("-" * 74)
sweep = {}
for a in AFF:
    hr = sum(s * hit_rate_for(a, l, s) for l, s, _ in PREFIXES)
    saved = sum(s * hit_rate_for(a, l, s) * t
                for l, s, t in PREFIXES) * PREFILL_MS_PER_TOK
    sk = load_skew(a)
    w = wait_ms(sk)
    sweep[a] = (hr, saved, sk, w)
    print(f"{a:>10.1f}{replication(a):>13.1f}{hr:>11.1%}{saved:>14.0f}m"
          f"{sk:>12.2f}{w:>12.1f}m")

print()
print()
print("Total time to first token: prefill not saved, plus queue wait.")
print()
print(f"{'affinity':>10}{'prefill ms':>13}{'queue ms':>11}{'TTFT ms':>11}"
      f"{'vs balanced':>14}")
print("-" * 62)
total = {}
for a in AFF:
    hr, saved, sk, w = sweep[a]
    pf = mean_prefill - saved
    total[a] = pf + w
    print(f"{a:>10.1f}{pf:>13.0f}{w:>11.1f}{pf + w:>11.1f}"
          f"{(pf + w) / total[0.0]:>13.2f}x")

best = min(total, key=lambda k: total[k])
print()
print(f"best affinity: {best:.1f} at {total[best]:.1f} ms TTFT")

print()
print()
print("How the optimum moves with fleet load. A busier fleet cannot afford skew.")
print()


def total_at(a, rho_mean):
    saved = sum(s * hit_rate_for(a, l, s) * t
                for l, s, t in PREFIXES) * PREFILL_MS_PER_TOK
    return (mean_prefill - saved) + wait_ms(load_skew(a), rho_mean)


print(f"{'mean utilisation':>18}" + "".join(f"{('a=%.1f' % a):>9}" for a in AFF)
      + f"{'best':>8}")
print("-" * 80)
bestrow = {}
for rho in (0.12, 0.20, 0.28, 0.36, 0.44):
    vals = {a: total_at(a, rho) for a in AFF}
    b = min(vals, key=lambda k: vals[k])
    bestrow[rho] = b
    print(f"{rho:>18.0%}" + "".join(f"{vals[a]:>9.0f}" for a in AFF)
          + f"{b:>8.1f}")

print()
print()
print("And with prefix length, which decides how much a hit is worth.")
print()


def total_scaled(a, scale):
    mp = sum(s * t * scale for _, s, t in PREFIXES) * PREFILL_MS_PER_TOK
    saved = sum(s * hit_rate_for(a, l, s) * t * scale
                for l, s, t in PREFIXES) * PREFILL_MS_PER_TOK
    return mp - saved + wait_ms(load_skew(a))


print(f"{'prefix scale':>14}{'mean prefill':>14}"
      + "".join(f"{('a=%.1f' % a):>9}" for a in AFF) + f"{'best':>8}")
print("-" * 90)
scalebest = {}
for scale in (0.2, 0.5, 1.0, 2.0, 5.0):
    mp = sum(s * t * scale for _, s, t in PREFIXES) * PREFILL_MS_PER_TOK
    vals = {a: total_scaled(a, scale) for a in AFF}
    b = min(vals, key=lambda k: vals[k])
    scalebest[scale] = b
    print(f"{scale:>14.1f}{mp:>13.0f}m"
          + "".join(f"{vals[a]:>9.0f}" for a in AFF) + f"{b:>8.1f}")

print()
print()
print("What the cache capacity does: it is why uniform routing cannot simply")
print("cache everything everywhere.")
print()
print(f"{'cache tokens/node':>19}{'catalogue fits at a=0':>24}"
      f"{'best affinity':>16}")
print("-" * 60)
for cap in (2000.0, 4500.0, 9000.0, 17000.0, 40000.0):
    CACHE_TOKENS = cap
    fits = CATALOGUE <= cap
    vals = {a: total_at(a, 0.28) for a in AFF}
    b = min(vals, key=lambda k: vals[k])
    print(f"{cap:>19.0f}{('yes' if fits else 'no'):>24}{b:>16.1f}")
CACHE_TOKENS = 9000.0

print(f"""
The sweep is the conflict in one table. Raising affinity from {0.0:.1f} to {1.0:.1f}
takes replication from {replication(0.0):.0f} nodes per prefix to
{replication(1.0):.0f}, the hit rate from {sweep[0.0][0]:.1%} to {sweep[1.0][0]:.1%},
and prefill saved from {sweep[0.0][1]:.0f}ms to {sweep[1.0][1]:.0f}ms.

It also takes load skew from {sweep[0.0][2]:.2f} to {sweep[1.0][2]:.2f}, and queue wait
from {sweep[0.0][3]:.1f}ms to {sweep[1.0][3]:.1f}ms
(eq:affinity-fights-balance). **Both columns are real and they point in opposite
directions.**

Prefix affinity is not load balancing with a cache benefit attached. It is a trade of
queueing behaviour for prefill, and ch:sd-async established precisely how expensive
queueing gets when load stops being even.

The total column resolves it at **{best:.1f}**, giving {total[best]:.1f}ms against
{total[0.0]:.1f}ms for pure balancing and {total[1.0]:.1f}ms for pure pinning.
**Neither extreme is right.**

The curve is not monotone, and the dip at {0.2:.1f} is worth explaining rather than
smoothing away. At that affinity replication is {replication(0.2):.0f} nodes, which is
low enough that each node must hold {CACHE_TOKENS * NODES / replication(0.2):.0f}
tokens of catalogue and high enough that it still misses on
{1 - sweep[0.2][0]:.0%} of requests. **Partial replication is the worst of both**: it
pays most of the capacity cost of replicating and gets little of the targeting benefit
of pinning. A router configured half-way between the two strategies without
understanding which regime it is in can land here, and the symptom is a hit rate lower
than either pure strategy would give.

The utilisation table shows the optimum is not a constant. At {0.12:.0%} mean
utilisation the best affinity is {bestrow[0.12]:.1f}; at {0.44:.0%} it is
{bestrow[0.44]:.1f}.

The reason is that the queueing term is convex: at low utilisation a hot node absorbs
the extra work and skew costs almost nothing, while near saturation the same skew costs
a great deal. **The correct router behaviour is therefore load-dependent** -- lean on
affinity when there is headroom, fall back to balancing when there is not -- and a
router with a fixed affinity weight is wrong at one end of its own day.

The prefix-length table is the other axis, and it gives a rule that needs no arithmetic:
at {0.2:.1f} times these prefix lengths the best affinity is {scalebest[0.2]:.1f}, and
at {5.0:.1f} times it is {scalebest[5.0]:.1f}. **The longer the shared prefix, the more
skew is worth tolerating.**

A deployment where every request carries a two-thousand-token system prompt should route
aggressively by affinity. One with short, unique prompts should not build the
infrastructure at all.

The capacity table explains why uniform routing cannot simply sidestep this by caching
everything on every node. The catalogue is {CATALOGUE} tokens; a node holding
{CACHE_TOKENS:.0f} cannot keep all of it, so at low affinity the least popular prefixes
are evicted everywhere and their hits are lost. **Cache capacity is what makes
replication expensive**, and it is the term that converts this from a load-balancing
question into a placement one.

Two implementation notes. The hit rate depends on the prefix being *resident*, not on
its having been seen -- so an eviction on the target node silently converts an affinity
hit into a miss, and the router cannot tell without asking. And affinity routing makes
the fleet stateful: a node's value depends on what it holds, so draining one for
maintenance costs more than its capacity share, which ch:inf-kubernetes has to plan
around.""")
```

## 9. Practical Example

A 16-node fleet with a concentrated prefix catalogue:

```
  affinity  replication   hit rate  prefill saved   load skew   queue wait
--------------------------------------------------------------------------
       0.0         16.0      84.0%           203m        1.00         1.6m
       0.2         13.0      71.4%           172m        1.50         3.0m
       0.4         10.0      72.1%           198m        2.00         5.3m
       0.6          7.0      77.5%           216m        2.50         9.8m
       0.8          4.0      85.0%           237m        3.00        21.9m
       1.0          1.0     100.0%           278m        3.50       204.8m
```

Prefill saved rises from **203 ms** to **278 ms**; queue wait rises from **1.6 ms** to
**204.8 ms** ({{eq:affinity-fights-balance}}). **Both columns are real and they point
in opposite directions.**

```
  affinity   prefill ms   queue ms    TTFT ms   vs balanced
--------------------------------------------------------------
       0.0           76        1.6       77.4         1.00x
       0.2          106        3.0      109.2         1.41x
       0.4           80        5.3       85.3         1.10x
       0.6           63        9.8       72.4         0.94x
       0.8           42       21.9       63.7         0.82x
       1.0            0      204.8      204.8         2.65x
```

The optimum is **0.8** at **63.7 ms**, against **77.4** for pure balancing and
**204.8** for pure pinning. The dip at 0.2 is partial replication being the worst of
both.

And the optimum moves:

```
  mean utilisation    a=0.0    a=0.2    a=0.4    a=0.6    a=0.8    a=1.0    best
--------------------------------------------------------------------------------
               12%       76      107       81       64       44        3     1.0
               20%       77      108       83       67       48       10     1.0
               28%       77      109       85       72       64      205     0.8
               36%       78      111       91      100      316      274     0.0
               44%       79      114      111      337      316      274     0.0
```

**From 1.0 at light load to 0.0 at heavy** ({{eq:affinity-optimum-moves-with-load}}),
because the queueing term is convex. A router with a fixed affinity weight is wrong at
one end of its own day.

Prefix length moves it the other way:

```
  prefix scale  mean prefill    a=0.0    a=0.2    a=0.4    a=0.6    a=0.8    a=1.0    best
------------------------------------------------------------------------------------------
           0.2           56m       17       24       21       22       30      205     0.0
           1.0          278m       79      111       85       72       64      205     0.8
           5.0         1392m      381      534      405      323      231      205     1.0
```

**The longer the shared prefix, the more skew is worth tolerating.** A deployment with
a two-thousand-token system prompt on every request should route by affinity
aggressively; one with short unique prompts should not build the infrastructure.

The second listing turns to failure.

```python {tier=A name=da2}
"""Model parallelism multiplies the failure rate, and replication has to pay for it.

A tensor-parallel group is one machine as far as failure is concerned: lose any device
and the whole group stops, because the layer's matrices are split across all of them.

So splitting a model across n devices multiplies its failure rate by n, and the
availability of the group is the device availability raised to the n
(eq:parallel-group-is-one-failure-domain).

That is the same product that has appeared throughout this book -- ch:ag-loop's chain,
ch:sd-retrieval-agents's fan-out -- arriving as a hardware reliability question. This
listing measures what it costs and what replication has to do about it.
"""
import math

MTBF_HOURS = 42000.0        # mean time between failures for one device
REPAIR_HOURS = 3.5          # time to detect, drain, and replace
DEVICE_AVAIL = MTBF_HOURS / (MTBF_HOURS + REPAIR_HOURS)
DEGREES = [1, 2, 4, 8, 16, 32, 64]
TARGET = 0.99999


def group_avail(n):
    return DEVICE_AVAIL ** n


def group_mtbf(n):
    return MTBF_HOURS / n


print("One device: MTBF %.0f hours, repair %.1f hours, availability %.5f."
      % (MTBF_HOURS, REPAIR_HOURS, DEVICE_AVAIL))
print()
print("A tensor-parallel group fails when ANY member fails.")
print()
print(f"{'degree':>8}{'group MTBF hrs':>17}{'group availability':>21}"
      f"{'downtime hrs/yr':>18}{'vs 1 device':>14}")
print("-" * 80)
tab = {}
for n in DEGREES:
    a = group_avail(n)
    down = (1.0 - a) * 8760.0
    tab[n] = (group_mtbf(n), a, down)
    print(f"{n:>8}{group_mtbf(n):>17.0f}{a:>21.5f}{down:>18.1f}"
          f"{down / tab[1][2]:>13.1f}x")

print()
print()
print("Downtime per year against replica count. This is the continuous form of")
print("the redundancy question -- integer replicas make the threshold coarse.")
print()
print(f"{'degree':>8}" + "".join(f"{('r=%d' % r):>16}" for r in (1, 2, 3))
      + f"{'r=2 vs degree 1':>18}")
print("-" * 74)
down = {}
for n in DEGREES:
    av = group_avail(n)
    row = [(1.0 - av) ** r * 8760.0 for r in (1, 2, 3)]
    down[n] = row
    print(f"{n:>8}" + "".join(f"{v:>15.4f}h" for v in row)
          + f"{row[1] / down[1][1]:>17.0f}x")

print()
print()
print("Replicas needed to hold a %.3f%% target." % (TARGET * 100))
print()
print(f"{'degree':>8}{'group avail':>14}{'replicas':>11}"
      f"{'devices total':>16}{'redundancy':>13}")
print("-" * 64)
need = {}
for n in DEGREES:
    av = group_avail(n)
    r = 1
    while 1.0 - (1.0 - av) ** r < TARGET and r < 40:
        r += 1
    need[n] = (r, r * n)
    print(f"{n:>8}{av:>14.5f}{r:>11}{r * n:>16}"
          f"{(r - 1) / float(r):>12.0%}")

print()
print()
print("How repair time moves it. Faster replacement is the cheapest lever,")
print("because it acts on every degree at once.")
print()
print(f"{'repair hours':>14}" + "".join(f"{('d=%d' % n):>10}" for n in
                                        (4, 8, 16, 32, 64)))
print("-" * 66)
rep = {}
for rh in (0.5, 1.0, 3.5, 8.0, 24.0):
    av = MTBF_HOURS / (MTBF_HOURS + rh)
    row = []
    for n in (4, 8, 16, 32, 64):
        a = av ** n
        r = 1
        while 1.0 - (1.0 - a) ** r < TARGET and r < 40:
            r += 1
        row.append(r)
    rep[rh] = row
    print(f"{rh:>14.1f}" + "".join(f"{v:>10}" for v in row))
print()
print("(replicas needed to hold the target)")

print()
print()
print("And the part that is easy to miss: a partial failure does not merely stop")
print("the group, it drops every in-flight sequence's KV cache.")
print()
BATCH = 32
MEAN_DONE = 190          # tokens already generated when the failure lands
STEP_MS = 4.18
print(f"{'degree':>8}{'failures/yr':>14}{'sequences lost/yr':>20}"
      f"{'tokens recomputed/yr':>23}")
print("-" * 68)
loss = {}
for n in DEGREES:
    f_per_yr = 8760.0 / group_mtbf(n)
    seqs = f_per_yr * BATCH
    toks = seqs * MEAN_DONE
    loss[n] = (f_per_yr, seqs, toks)
    print(f"{n:>8}{f_per_yr:>14.2f}{seqs:>20.0f}{toks:>23.0f}")

print()
print()
print("Pipeline parallelism has the same exposure per group but smaller groups")
print("are possible, because stages can be replicated independently.")
print()
print(f"{'topology':>34}{'failure domain':>17}{'availability':>15}"
      f"{'downtime at r=2':>18}")
print("-" * 84)
TOPO = [
    ("16-way tensor, one group",        16),
    ("8-way tensor x 2-way pipeline",    8),
    ("4-way tensor x 4-way pipeline",    4),
    ("2-way tensor x 8-way pipeline",    2),
]
for label, dom in TOPO:
    a = group_avail(dom)
    print(f"{label:>34}{dom:>17}{a:>15.5f}"
          f"{(1.0 - a) ** 2 * 8760.0:>17.4f}h")

print(f"""
The first table is the multiplication. A device with a {MTBF_HOURS:.0f}-hour MTBF and
{DEVICE_AVAIL:.5f} availability, put into a {16}-way tensor-parallel group, gives that
group an availability of {tab[16][1]:.5f} and
{tab[16][2]:.1f} hours of downtime a year against a single device's
{tab[1][2]:.1f} (eq:parallel-group-is-one-failure-domain).

**Sixteen times the downtime, because the group fails if any member does.** That is
ch:ag-loop's chain and ch:sd-retrieval-agents's fan-out in a rack: a product of
per-component reliabilities, and the exponent is the parallelism degree.

The downtime table prices it continuously. With {2} replicas, a degree-{1} service is
down {down[1][1]:.4f} hours a year and a degree-{16} service is down
{down[16][1]:.4f} -- **{down[16][1] / down[1][1]:.0f} times more**, because squaring a
worse number leaves a worse number.

The replica table converts that into hardware. Holding a {TARGET:.3%} target needs
{need[1][0]} replicas at degree 1 and {need[64][0]} at degree {64}, which is
{need[64][1]} devices for {64} devices of capacity --
{(need[64][0] - 1) / float(need[64][0]):.0%} redundancy.

**Model parallelism does not merely cost communication; it costs redundancy**, and the
redundancy cost is the one that does not appear in any throughput benchmark.
ch:inf-parallelism's speedup tables are all computed on a group that is assumed to
work.

The repair table is the cheapest available lever, and it is worth noticing that it is
not a hardware lever. At degree {64}, cutting repair time from {3.5:.1f} hours to
{0.5:.1f} takes the replicas needed from {rep[3.5][4]} to {rep[0.5][4]}; at
{24.0:.0f} hours it would take {rep[24.0][4]}. At degree {32} the same cut takes it
from {rep[8.0][3]} to {rep[0.5][3]}.

**Detection and replacement speed buys redundancy across every degree at once**, which
makes it the highest-leverage reliability investment in a model-parallel fleet --
automated draining, hot spares, and fast health detection rather than more hardware.

The in-flight table is the cost nobody budgets for. A failure does not merely stop the
group; it destroys the KV cache of every sequence in flight. At degree {16} that is
{loss[16][0]:.2f} failures a year, each dropping {BATCH} sequences that have already
generated {MEAN_DONE} tokens -- {loss[16][2]:.0f} tokens a year recomputed, and
{loss[16][1]:.0f} user-visible request failures unless something replays them.

**This is where ch:as-state-machines's replay requirement becomes a hardware
constraint.** A serving system without request-level replay converts every hardware
failure into a batch of user-visible errors, and the batch size is the batch size.

The topology table is the design response. A {16}-way tensor group has availability
{group_avail(16):.5f} and, duplicated, {(1 - group_avail(16)) ** 2 * 8760.0:.4f} hours
of annual downtime. Splitting the same sixteen devices as {4}-way tensor by {4}-way
pipeline gives each failure domain availability {group_avail(4):.5f} and
{(1 - group_avail(4)) ** 2 * 8760.0:.4f} hours --
**{((1 - group_avail(16)) ** 2) / ((1 - group_avail(4)) ** 2):.0f} times better for the
same device count**.

That is a real argument for pipeline parallelism that ch:inf-parallelism's
communication analysis does not contain: **smaller failure domains**. A pipeline stage
can be replaced independently while tensor-parallel ranks cannot, so the topology
decision has a reliability term alongside the bandwidth one -- and on a large fleet the
reliability term is frequently the larger.""")
```

A tensor-parallel group fails when any member fails:

```
  degree   group MTBF hrs   group availability   downtime hrs/yr   vs 1 device
--------------------------------------------------------------------------------
       1            42000              0.99992               0.7          1.0x
       4            10500              0.99967               2.9          4.0x
      16             2625              0.99867              11.7         16.0x
      64              656              0.99468              46.6         63.8x
```

**Sixteen times the downtime at degree 16** — 11.7 hours a year against 44 minutes
({{eq:parallel-group-is-one-failure-domain}}).

Replication compounds it rather than fixing it:

```
  degree             r=1             r=2             r=3   r=2 vs degree 1
--------------------------------------------------------------------------
       1         0.7299h         0.0001h         0.0000h                1x
       4         2.9194h         0.0010h         0.0000h               16x
      16        11.6717h         0.0156h         0.0000h              256x
      64        46.5937h         0.2478h         0.0013h             4075x
```

At two replicas, degree 16 is **256×** worse than degree 1 and degree 64 is
**4075×**, because the degree enters the base and the replica count the exponent.

```mermaid {#fig:domains caption="A tensor-parallel group is one failure domain: availability is the device availability raised to the degree. Rearranging the same devices into smaller domains improves reliability without buying anything."}
flowchart TD
  A["16 devices"] --> B["one 16-way tensor group<br/>0.99867, 0.0156h at r=2"]
  A --> C["4x4-way tensor,<br/>4-stage pipeline<br/>0.99967, 0.0010h at r=2"]
  B --> D["16x worse<br/>same hardware"]
  C --> D
```

Repair time is the cheapest lever:

```
  repair hours       d=4       d=8      d=16      d=32      d=64
------------------------------------------------------------------
           0.5         2         2         2         2         2
           3.5         2         2         2         2         3
          24.0         2         3         3         3         4
```

It acts on every degree at once, and it is an operations investment rather than a
hardware one.

The in-flight cost nobody budgets: at degree 16, **3.34 failures a year**, each dropping
**32 sequences** that have generated 190 tokens — **107 user-visible failures a year**
unless something replays them.

And the topology response: 16-way tensor gives **0.0156 h** annual downtime at two
replicas; 4-way tensor by 4-way pipeline gives **0.0010 h** — **16× better for the same
device count** ({{eq:smaller-domains-beat-more-replicas}}).

## 10. Production Considerations

Make affinity load-dependent. A single weight is wrong at one end of the daily cycle;
the router should relax toward balancing as utilisation rises, and the crossover is
computable from {{eq:affinity-optimum-moves-with-load}}.

Have nodes report cache residency rather than inferring it from recency. Eviction
silently converts hits to misses, and a periodic bloom filter is cheap.

Measure the prefix distribution before building affinity routing at all. If prefixes are
short or unique, the infrastructure is wasted; the scale table gives the threshold.

Choose the topology on reliability as well as bandwidth. On a large fleet the failure
term usually dominates, and {{eq:smaller-domains-beat-more-replicas}} is available for
free.

Invest in detection and replacement speed before buying replicas. It acts on every
degree simultaneously and it is the only lever here that is not hardware.

Implement request-level replay. Without it, every hardware failure produces a batch of
user-visible errors, and the batch is your batch size.

Place replicas across correlated-failure boundaries, and know where those boundaries
are. Two replicas in one chassis are not two replicas; the availability arithmetic
assumes independence that shared power and cooling do not provide.

Account for the drain cost of a stateful node. Removing a node from an affinity-routed
fleet costs more than its capacity share, and maintenance windows should be sized for
the re-prefill.

## 11. Common Mistakes

**A fixed affinity weight.** Wrong at one end of the daily load cycle, and the end it
is wrong at is the busy one, where being wrong costs most.

**Inferring residency from recency.** Eviction breaks the inference silently.

**Assuming replication fixes model-parallel fragility.** It squares a worse number.

**Choosing topology on bandwidth alone.** The reliability term is often larger on a
large fleet.

**Treating a serving node as stateless.** Affinity routing makes it stateful, and
draining costs more than capacity.

**Not implementing replay.** Converts each hardware failure into a batch of
user-visible errors, sized by the batch rather than by anything the user did.

## 12. Failure Modes

**Affinity-induced hotspot cascade.** A popular prefix pins traffic to a node, the node
saturates, its queue grows, and requests time out — while the rest of the fleet is idle.

**Silent hit-rate collapse after a cache-size change.** A configuration change reduces
resident prefixes, the affinity router keeps routing the same way, and prefill cost
rises with no routing change.

**Correlated device failure.** The availability model assumes independence; a shared
power rail, cooling zone, or firmware bug takes a whole group at once and replication
within that domain buys nothing.

**Rejoin storm.** A replaced device in a large tensor group must load its slice before
the group serves, and several simultaneous replacements can exceed the fabric's
capacity to deliver weights, extending an outage that hardware had already resolved.

**Replica cache divergence.** Under affinity routing, replicas hold different
prefixes, so losing one costs a prefill spike on top of the capacity loss - and the
spike is largest exactly when affinity is doing the most good.

**Drain-induced re-prefill spike.** Taking a node out for maintenance moves its prefixes
elsewhere, and the resulting prefill burst can exceed what
{{ch:inf-batching}}'s chunking budget absorbs.

## 13. Alternatives

**Pure load balancing.** Correct when prefixes are short or unique, and the scale table
says exactly when. It is also the right default while a fleet is small enough that
the routing infrastructure would cost more than the prefill it saves.

**Consistent hashing on prefix.** A common implementation of high affinity with cheap
routing state, requiring no central view of what each node holds. It inherits the
hotspot problem in full, and bounded-load variants that cap any single node's share
are what make it usable in practice - which is the same load-dependent relaxation
{{eq:affinity-optimum-moves-with-load}} argues for, implemented at the hash layer
instead of the router.

**Global prefix cache in shared storage.** Fetch the cache rather than routing to the
node holding it, which decouples routing from residency entirely and lets the router
balance load freely. It costs the transfer — {{ch:inf-gpu-memory}}'s 419 MB per
3200-token prefix — so it is viable intra-rack and not beyond, and it converts a
routing problem into a bandwidth one.

**Smaller parallel groups with more replicas.** The {{eq:smaller-domains-beat-more-replicas}}
answer, at the cost of pipeline bubbles.

**Fewer, larger nodes.** Reducing the parallelism degree by using devices with more
memory removes the reliability problem rather than mitigating it, and is the cleanest
answer when the hardware exists. It also removes the interconnect requirement from
{{ch:inf-parallelism}}, which is a second saving that rarely gets counted in the
purchase decision.

## 14. Evaluation

Report prefix hit rate and load skew together. Either alone is uninformative, and the
pair is the state of the routing trade.

Measure time-to-first-token by prefix popularity bucket. Affinity routing helps popular
prefixes and can hurt rare ones, and an aggregate figure hides both.

Report per-group availability, not per-device. The group is the failure unit and the
device number is misleadingly good by a factor equal to the parallelism degree.

Publish the replica placement map against the correlated-failure boundaries it is
meant to cross. Independence is an assumption in every calculation here, and it is
the assumption most easily violated by a routine rack consolidation that nobody
connected to serving availability.

Track in-flight request loss per failure, and compare it against the replay rate. The
gap is your user-visible failure count.

Measure mean time to replacement as a first-class operational metric. It is the input to
every redundancy calculation here and it is usually estimated rather than measured.

## 15. Advanced Concepts

The independence assumption in {{eq:parallel-group-is-one-failure-domain}} is the
weakest part of the failure model, and it fails in the expensive direction. Devices in
one group share a chassis, a power supply, a cooling zone, and a firmware version, so
their failures are positively correlated. Correlation makes $A(n)$ *better* than $a^n$
for the group — several devices failing together is one outage rather than several — but
makes replication *worse*, because replicas placed within the same correlated domain
fail together. The practical rule is that replicas must cross the correlation boundary,
and identifying that boundary is a datacentre question rather than a serving one.

The routing model treats the prefix catalogue as static. Real catalogues turn over:
documents enter and leave conversation, system prompts are versioned, and a prefix that
was popular this morning is cold this afternoon. That makes the resident set a function
of time, and the affinity optimum a function of catalogue churn as well as load and
length. A high-churn catalogue argues for lower affinity, because the cost of pinning to
a node that is about to hold a stale prefix is paid twice.

The redundancy arithmetic also assumes replicas are interchangeable, which affinity
routing violates. Two replicas of a parallel group hold different prefix caches, so
losing one does not merely halve capacity -- it loses whatever prefixes only that
replica held, and the survivor must re-prefill them. **Affinity routing and replication
interact badly**: the more distinct the replicas' caches, the more a failure costs
beyond its capacity share, and the effect is largest exactly where affinity is most
valuable. A fleet running high affinity should either replicate cache placement
deliberately across failure domains, accepting the capacity cost, or accept that a
failure produces a prefill spike as well as a capacity loss.

There is a composition this chapter does not settle. Affinity routing and disaggregation
interact: with prefill and decode separated, the prefix cache lives on prefill machines
and the decode machines are stateless, so affinity applies only to the prefill fleet
while load balancing applies freely to decode. That decomposition removes most of the
tension in {{eq:affinity-fights-balance}} — the two objectives now apply to different
machines — and it is a stronger argument for disaggregation than either the throughput
or the reliability one. Whether it survives the KV transfer cost at realistic prefix
sizes is unmeasured.

## 16. Connection to Previous Chapters

{{eq:variance-not-mean-drives-wait}} from {{ch:sd-async}} is what makes affinity a trade
rather than a free optimisation; the convexity of the queueing term is why
{{eq:affinity-optimum-moves-with-load}} holds.

{{eq:batch-times-context-is-the-budget}} from {{ch:inf-gpu-memory}} bounds the prefix
cache, and that bound is what makes replication expensive rather than free.

{{eq:parallelism-dimension-is-an-interconnect-decision}} from {{ch:inf-parallelism}}
chose topology on bandwidth; {{eq:smaller-domains-beat-more-replicas}} adds the term it
omitted.

{{eq:loop-is-not-a-chain}} from {{ch:ag-loop}} appears for the fourth time in this book,
now as hardware availability. The product form is the same; only the components differ.

## 17. Exercises

1. Derive the affinity at which prefill savings and queueing cost have equal derivatives,
   for a two-prefix catalogue.

2. Compute the group availability and two-replica downtime for a 32-way tensor group of
   devices with 30,000-hour MTBF and 2-hour repair.

3. Find the fleet size at which the reliability term in the topology decision exceeds
   the bandwidth term, for the parameters in {{ch:inf-parallelism}}.

4. Model prefix catalogue churn and find how much it lowers the optimal affinity.

5. For a deployment you have access to, measure the prefix distribution. What affinity
   would you choose, and how would you make it load-dependent?

## 18. Interview Questions

1. Why is routing every request to the node holding its prefix a bad idea?

2. Our affinity router works well overnight and badly at peak. Explain.

3. Sixteen 99.99%-available GPUs run one model. What is the model's availability?

4. Would you rather have one 16-way tensor group or four 4-way groups in a pipeline?
   On what grounds?

5. A GPU fails mid-batch. What did the users see, and what would have prevented it?

6. We run two replicas of a 16-way group and our availability is still worse than a
   duplicated single device. Is that expected? By how much?

## 19. Research Questions

1. How much of {{eq:affinity-fights-balance}}'s tension does disaggregation remove by
   putting the cache on a separate fleet, net of the KV transfer cost?

2. What is the right online policy for adapting affinity to load, and how does it behave
   under load oscillation?

3. How correlated are device failures within a chassis in practice, and where should
   replica placement boundaries be drawn?

4. Can prefix residency be predicted well enough to route on, without per-node
   reporting?

## 20. Chapter Summary

Prefix affinity and load balancing want opposite things. Across a 16-node fleet, raising
affinity from 0 to 1 lifts prefill saved from **203 ms** to **278 ms** and queue wait
from **1.6 ms** to **204.8 ms** ({{eq:affinity-fights-balance}}). The optimum is
**0.8** at **63.7 ms**, against **77.4** balanced and **204.8** pinned.

That optimum is not stable. Because the queueing term is convex, it moves from **1.0**
at 12% utilisation to **0.0** at 44% ({{eq:affinity-optimum-moves-with-load}}), and it
moves the other way with prefix length — 0.0 at short prefixes, 1.0 at long. **A fixed
affinity weight is wrong at one end of its own day.**

A tensor-parallel group is one failure domain: availability is $a^n$, so a 16-way group
has **16×** the downtime of one device ({{eq:parallel-group-is-one-failure-domain}}) and
**256×** at two replicas. Each failure also destroys every in-flight sequence's cache —
**107 user-visible failures a year** at degree 16 without replay.

The response is topology rather than redundancy: sixteen devices as 4-way tensor by
4-way pipeline give **0.0010 h** annual downtime against one 16-way group's **0.0156 h**
— **16× better for the same hardware**
({{eq:smaller-domains-beat-more-replicas}}). And repair speed is cheaper than either,
because it acts on every degree at once.

Both halves of this chapter describe costs that a single-node measurement cannot
produce and that a fleet pays continuously. Prefix affinity does not exist as a
question until there is somewhere else to route to; failure domains do not exist as
a question until a group is one of many. So a serving design validated on one node
and scaled out has two unexamined decisions in it, and both of them have optima that
move with conditions rather than sitting at a default.

Carry forward: **the router's affinity is a function, not a constant**, and **the
failure domain is the parallel group, not the device**.

## 21. Further Reading

- {{cite:zhong2024distserve}} — disaggregation, whose reliability implications neither
  it nor {{ch:inf-batching}} foreground.
- {{cite:patel2023splitwise}} — heterogeneous fleets, and phase separation as a fleet
  design.
- {{cite:kwon2023pagedattention}} — the cache blocks that a failure destroys.
- {{cite:shoeybi2019megatron}} — why tensor-parallel ranks cannot be replaced
  independently.

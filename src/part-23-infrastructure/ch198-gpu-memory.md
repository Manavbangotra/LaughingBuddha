---
id: inf-gpu-memory
number: 198
part: XXIII
tier: full
status: draft
requires: [decode-is-bandwidth-bound, kv-traffic-overtakes-weights,
           batch-is-the-mechanism-not-an-optimisation, access-shape-decides-the-store]
provides: [roofline-has-multiple-ridges, tier-crossing-has-a-ceiling,
           batch-times-context-is-the-budget, cache-quantisation-is-the-larger-lever]
citations: [dao2022flash, kwon2023pagedattention, pope2022inference,
            rajbhandari2020zero]
---

## 1. Learning Objectives

By the end of this chapter you will be able to construct a multi-tier roofline and say
which tier an operation is actually bound by; explain FlashAttention as a movement of
traffic between memory tiers rather than a reduction in arithmetic, and state the
ceiling that bounds any such optimisation; compute the batch-times-context frontier for
a device and show why maximum batch and maximum context are one setting rather than
two; separate weight quantisation from KV-cache quantisation and say which buys more
serving capacity; and quantify what memory fragmentation costs and therefore what
paging recovers.

## 2. Why This Matters

{{ch:inf-cpu-gpu}} used a single balance point and got the shape of the problem right.
This chapter corrects two simplifications in it, and both corrections change decisions.

The first is that there is not one roofline. A current datacentre GPU's HBM needs
**295** operations per byte to keep its arithmetic units fed; its shared memory needs
**8** — **38×** less, because it delivers 38× the bandwidth
({{eq:roofline-has-multiple-ridges}}). An operation hopelessly memory-bound against
HBM can be comfortably compute-bound against shared memory, and arranging that is what
{{cite:dao2022flash}} does: at 8192-token sequences it cuts HBM traffic **65×** while
computing the identical result.

The second is that memory *capacity*, not just bandwidth, bounds the batching
{{ch:inf-cpu-gpu}} showed to be essential. Weights are fixed; KV cache scales with
batch **times** context. So the two are a single hyperbola —
**476,837 token-slots** on an 80 GB device at bf16
({{eq:batch-times-context-is-the-budget}}) — and a system configured with both set
independently is oversubscribed by a factor nobody computed.

{{sec:9-practical-example}} also finds that quantising the *cache* buys **4.0×** the
serving capacity while quantising the *weights* buys **1.2×**
({{eq:cache-quantisation-is-the-larger-lever}}) — and the cache is the one usually left
alone.

## 3. Prerequisites

You need {{eq:decode-is-bandwidth-bound}} and
{{eq:batch-is-the-mechanism-not-an-optimisation}} from {{ch:inf-cpu-gpu}}; this chapter
refines both rather than replacing them.

{{eq:kv-traffic-overtakes-weights}} is the bandwidth-side result whose capacity-side
counterpart is {{eq:batch-times-context-is-the-budget}}. Holding both is the point of
the chapter.

{{eq:access-shape-decides-the-store}} from {{ch:sd-storage}} is the same
bandwidth-versus-capacity trade one level up, and the parallel is exact enough to be
worth noticing.

{{ch:q-memory-math}} supplies quantisation formats; {{ch:tf-masking-kv}} supplies the
cache.

## 4. Intuitive Explanation

The single-roofline picture says: compute the ratio of operations to bytes, compare it
to the device's ratio of FLOP/s to bandwidth, and you know whether you are compute-
or memory-bound. That is correct and it hides something.

It hides the question *which memory*. A GPU has a hierarchy — registers, then a small
fast scratchpad shared within a processor group, then a cache, then the big
high-bandwidth memory everyone means when they say "GPU memory," then the links to
other GPUs and to the host. Each tier is roughly an order of magnitude faster and an
order of magnitude smaller than the one below it.

So there is a balance point per tier, and they differ enormously. Against HBM you need
about three hundred operations per byte. Against the on-chip scratchpad you need
about eight. **Whether an operation is memory-bound is not a property of the
operation. It is a property of the operation and the tier you read from.**

That reframes what an optimisation can be. You cannot usually reduce the arithmetic —
the model does what it does. You cannot make HBM faster. What you *can* do is
restructure the computation so its intermediate results never reach HBM at all.

That is exactly FlashAttention. Naive attention computes an n-by-n score matrix,
writes it to HBM, reads it back to apply softmax, writes the result, reads it again to
multiply by V. For a long sequence that matrix is enormous and it crosses HBM four
times. Tiled attention processes the computation in blocks small enough that each
block's scores stay on-chip, and the matrix never exists in HBM at all.

The arithmetic is identical. The answer is identical, exactly, not approximately. The
only thing that changed is which tier the intermediate lived in — and the operation
moves from memory-bound to compute-bound.

There is a ceiling on this kind of win, and it is worth knowing before you go looking
for the next one: **once an operation is compute-bound, further traffic reductions buy
nothing.** You are at peak. The tiled attention in {{sec:9-practical-example}} runs at
100% of peak arithmetic, and no cleverness improves it.

The second half of the chapter is about capacity rather than bandwidth, and it
contains the single most common configuration error in serving.

Memory holds three things: weights, KV cache, and working buffers. Weights are fixed —
a 7B model at bf16 is 14 GB whether you serve one user or a thousand. Working buffers
are roughly fixed. Everything left is cache, and cache is consumed at a rate of
*bytes per token per sequence*.

Which means the amount you can serve is bounded by a product: batch size times context
length. Not by either one. A device that can hold 465 sequences at 1024 tokens holds 7
at 65,536 tokens, and those are the same device with the same setting.

Teams configure these as two independent numbers because the configuration file has two
fields. Set batch to 64 because 64 sounds reasonable, set max context to 16,384 because
that is what the model supports, and you have specified a configuration requiring
137 GB of cache on a device with 62 GB. It will pass every test, because tests do not
produce 64 simultaneous maximum-length requests. It will fail in production, with an
out-of-memory error, under a load the configuration explicitly permitted.

## 5. Formal Explanation

Let a device have peak arithmetic $F$ and a hierarchy of memory tiers indexed by $j$
with bandwidths $B_j$ and capacities $C_j$, ordered so $B_1 > B_2 > \cdots$ and
$C_1 < C_2 < \cdots$. Each tier has its own balance point

$$ I_j^\star \;=\; \frac{F}{B_j} $$ (eq:roofline-has-multiple-ridges)

An operation performing $\Phi$ operations while moving $M_j$ bytes through tier $j$
takes time

$$ T \;=\; \max\!\left(\frac{\Phi}{F},\; \max_j \frac{M_j}{B_j}\right) $$

so it is bound by **whichever tier it is worst against**, and an optimisation that
reduces $M_j$ for the binding tier is worthless once some other term dominates.

The achievable speedup from moving traffic out of tier $j$ into a faster tier is
therefore bounded:

$$ \frac{T_{\text{before}}}{T_{\text{after}}} \;\le\; \frac{M_j / B_j}{\Phi / F} \;=\; \frac{I_j^\star}{I_j} $$ (eq:tier-crossing-has-a-ceiling)

**The ceiling is the ratio by which the operation missed the balance point**, and once
reached, the operation is at peak and no further restructuring helps.

For capacity, let $W$ be weight bytes, $A$ fixed working-buffer bytes, and $\kappa$
the KV bytes per token per sequence. A device of capacity $C$ satisfies

$$ W + A + m\,c\,\kappa \;\le\; C \quad\Longrightarrow\quad m\,c \;\le\; \frac{C - W - A}{\kappa} $$ (eq:batch-times-context-is-the-budget)

The right-hand side is the **conserved quantity**: token-slots. Batch and context are
free to trade along the hyperbola $mc = \text{const}$, and neither is independently
meaningful.

Differentiating with respect to the two quantisation choices — $b_w$ bytes per weight
and $b_k$ bytes per cache element — gives

$$ mc \;=\; \frac{C - Pb_w - A}{2Lh_{kv}d_h b_k} $$

Comparing the two derivatives directly gives the result:

$$ \frac{\partial \log(mc)}{\partial \log b_k} = -1, \qquad \frac{\partial \log(mc)}{\partial \log b_w} = -\frac{Pb_w}{C - Pb_w - A} $$ (eq:cache-quantisation-is-the-larger-lever)

Token-slots scale as \(1/b_k\) exactly -- a unit elasticity --
while their sensitivity to weight precision is the weights' share of the *remaining*
capacity, which for a small model on a large device is well under one. **The cache
lever is multiplicative and the weight lever is additive**, and
{{sec:9-practical-example}} measures the gap at a factor of three.

## 6. Mathematical Foundation

The fragmentation result follows from the contiguity requirement. Without paging, a
sequence must reserve $c_{\max}$ tokens even if it uses $\bar{c}$, so achievable batch
is $m_{\text{naive}} = (C - W - A)/(c_{\max}\kappa)$ against a paged
$m_{\text{paged}} = (C - W - A)/(\bar{c}\kappa)$. The ratio is

$$ \frac{m_{\text{paged}}}{m_{\text{naive}}} \;=\; \frac{c_{\max}}{\bar{c}} \;=\; \frac{1}{u} $$

where $u$ is length utilisation. **Paging's gain is exactly the inverse of the
utilisation it replaces** — a deployment whose requests use a tenth of their allowed
context gets ten times the batch, and one whose requests use all of it gets nothing.
{{sec:9-practical-example}} measures **45.7×** at 2.2% utilisation and **1.0×** at
100%.

That is worth stating precisely because it bounds the claim. {{cite:kwon2023pagedattention}}
reports large throughput gains, and the gain is real and is a function of *your length
distribution*, not of the technique. A deployment where every request genuinely uses
the full context window gains nothing from paging, and one with a long tail of short
requests gains enormously.

Combining with {{ch:inf-cpu-gpu}}'s bandwidth result gives the complete picture.
Capacity permits $mc \le S$ token-slots; bandwidth efficiency is
$E = (1+\rho)/(1+m\rho)$ with $\rho = K(c)/W$. Maximising throughput subject to the
capacity constraint means choosing $m$ and $c$ on the frontier, and since $E$ falls in
$m$ while capacity is indifferent to how the product is split, **the throughput
optimum is at the largest $m$ the latency target permits and the smallest $c$ the task
permits** — which is a statement about product design as much as configuration.

## 7. Internal Mechanics

**Why the fast tier is small.** Bandwidth and capacity trade against each other
physically: SRAM close to the arithmetic units is fast because it is close and small,
and HBM is large because it is off-die. There is no tier that is both, and every
optimisation in this chapter is a way of arranging for the working set of a hot loop
to fit in a tier where it can be read quickly. {{sec:9-practical-example}}'s tile table
is that constraint in numbers: 228 KB of shared memory per processor group bounds the
tile, and a 512-row tile does not fit.

**What the allocator reserves.** The "working buffers" term is not small and not well
documented: CUDA context, the framework's caching allocator reserve, intermediate
activations for the largest layer, and communication buffers if parallelism is in play.
Several gigabytes is typical, and the number is usually discovered empirically after
the first out-of-memory error rather than budgeted for.

**Prefix sharing.** {{cite:kwon2023pagedattention}}'s paging makes it possible for two
sequences with a common prefix — a shared system prompt, a shared retrieved document —
to share the cache pages for that prefix. This is the one mechanism that genuinely
makes KV traffic *shared* rather than merely well-packed, and it is why a deployment
with a long fixed system prompt benefits far more than the fragmentation arithmetic
alone predicts.

**Why cache quantisation is harder than weight quantisation.** Weights are quantised
once, offline, with the full distribution available and time to calibrate. Cache
entries are produced at serving time, one token at a time, and outliers in the key and
value distributions arrive without warning. {{ch:q-activation-kv}} has the detail; the
consequence here is that the larger capacity lever is also the harder one, which
explains why deployments reach for the smaller one first.

**Why the frontier is a hyperbola and not a box.** Configuration interfaces present
batch and context as independent maxima because that is how a validator is easiest to
write: check each field against its own bound. Expressing the real constraint requires
a cross-field check, which most configuration schemas cannot state and most operators
would not expect. The result is that the invalid region of the configuration space is
not merely unguarded -- it is unrepresentable in the language operators use to
describe the system, which is why the failure recurs across otherwise well-run
deployments.

**Host offload is a different tier, not more memory.** Moving weights or cache to host
memory across PCIe means reading them at **64 GB/s** against HBM's **3350 GB/s** — a
52× penalty. {{cite:rajbhandari2020zero}}'s offload techniques are viable for
*training*, where each byte moved is amortised over a large batch of gradient
computation. For decode, where a byte of weight buys two operations, offload converts
a memory-bound workload into an almost unusable one.

## 8. Implementation

The first listing builds the multi-tier roofline and measures attention before and
after tiling.

```python {tier=A name=ce1}
"""One roofline is not enough: an operation can be starved at one memory tier while
having ample headroom at another.

ch:inf-cpu-gpu used a single balance point -- peak FLOP/s over HBM bandwidth. Real
devices have a hierarchy, and each level has its own bandwidth and its own balance
point. An operation is bound by whichever tier it is worst against
(eq:roofline-has-multiple-ridges).

That matters because the standard fix for one tier is useless for another, and
because the biggest single optimisation in transformer serving -- cite:dao2022flash --
is precisely a move of traffic from one tier to the next one down.

This listing builds the multi-tier roofline and measures where attention sits before
and after tiling.
"""
# Memory tiers on a current datacentre GPU.
# (tier, bandwidth bytes/s, capacity bytes)
TIERS = [
    ("registers",     2.20e14,  2.6e8),
    ("shared / L1",   1.28e14,  2.3e8),
    ("L2 cache",      1.10e13,  5.0e7),
    ("HBM",           3.35e12,  8.0e10),
    ("NVLink peer",   9.00e11,  0.0),
    ("PCIe host",     6.40e10,  0.0),
]
PEAK = 9.89e14           # dense bf16 FLOP/s

print("Memory tiers, each with its own balance point: the arithmetic intensity")
print("needed to keep the arithmetic units fed FROM THAT TIER.")
print()
print(f"{'tier':>16}{'bandwidth GB/s':>18}{'balance point':>16}"
      f"{'easier than HBM':>18}")
print("-" * 62)
bal = {}
hbm = None
for name, bw, cap in TIERS:
    b = PEAK / bw
    bal[name] = b
    if name == "HBM":
        hbm = b
for name, bw, cap in TIERS:
    print(f"{name:>16}{bw / 1e9:>18.0f}{bal[name]:>13.0f} F/B"
          f"{bal['HBM'] / bal[name]:>17.1f}x")

print()
print("A tier with more bandwidth needs LESS intensity to saturate. So an operation")
print("that is memory-bound against HBM may be compute-bound against shared memory,")
print("if you can arrange for it to read from there.")

print()
print()
print("Attention during prefill, the operation cite:dao2022flash addresses.")
print("Sequence length n, head dimension d.")
print()
D_HEAD = 128
BYTES = 2.0
SEQS = [512, 2048, 8192, 32768]


def naive_attention(n, d):
    """Traffic if the n-by-n score matrix is materialised in HBM.

    Write S = QK^T, read it back for softmax, write P, read it back for PV.
    """
    qkv = 3.0 * n * d * BYTES
    scores = 4.0 * n * n * BYTES          # two writes and two reads of n-by-n
    out = n * d * BYTES
    return qkv + scores + out


def tiled_attention(n, d):
    """Traffic if scores never leave on-chip memory (cite:dao2022flash).

    Q, K, V are read from HBM; the n-by-n intermediate lives in SRAM.
    """
    return 3.0 * n * d * BYTES + n * d * BYTES


def attention_flops(n, d):
    """QK^T then PV: two n-by-n-by-d matmuls."""
    return 2.0 * 2.0 * n * n * d


print(f"{'seq len':>10}{'naive HBM MB':>15}{'tiled HBM MB':>15}"
      f"{'reduction':>12}{'GFLOP':>10}")
print("-" * 62)
att = {}
for n in SEQS:
    nv = naive_attention(n, D_HEAD)
    tl = tiled_attention(n, D_HEAD)
    fl = attention_flops(n, D_HEAD)
    att[n] = (nv, tl, fl)
    print(f"{n:>10}{nv / 1e6:>15.1f}{tl / 1e6:>15.1f}{nv / tl:>11.1f}x"
          f"{fl / 1e9:>10.1f}")

print()
print()
print("Arithmetic intensity against HBM, before and after tiling, with the")
print("HBM balance point of %.0f for reference." % bal["HBM"])
print()
print(f"{'seq len':>10}{'naive I':>12}{'tiled I':>12}{'naive bound':>14}"
      f"{'tiled bound':>14}")
print("-" * 62)
inten = {}
for n in SEQS:
    nv, tl, fl = att[n]
    i_nv = fl / nv
    i_tl = fl / tl
    inten[n] = (i_nv, i_tl)
    print(f"{n:>10}{i_nv:>12.1f}{i_tl:>12.1f}"
          f"{('memory' if i_nv < bal['HBM'] else 'compute'):>14}"
          f"{('memory' if i_tl < bal['HBM'] else 'compute'):>14}")

print()
print()
print("What that does to time. A memory-bound op takes traffic/bandwidth; a")
print("compute-bound one takes FLOPs/peak.")
print()
print(f"{'seq len':>10}{'naive ms':>12}{'tiled ms':>12}{'speedup':>11}"
      f"{'tiled % of peak':>18}")
print("-" * 63)
times = {}
for n in SEQS:
    nv, tl, fl = att[n]
    t_nv = max(nv / TIERS[3][1], fl / PEAK)
    t_tl = max(tl / TIERS[3][1], fl / PEAK)
    times[n] = (t_nv, t_tl)
    print(f"{n:>10}{t_nv * 1000:>12.3f}{t_tl * 1000:>12.3f}"
          f"{t_nv / t_tl:>10.1f}x{fl / t_tl / PEAK:>18.1%}")

print()
print()
print("But tiling only works if a tile fits on-chip. Shared memory per streaming")
print("multiprocessor bounds the tile, which bounds how much can stay resident.")
print()
SMEM = 228.0 * 1024      # bytes of shared memory per SM
print(f"shared memory per SM: {SMEM / 1024:.0f} KB")
print()
print(f"{'tile rows':>11}{'tile bytes':>13}{'fits':>8}{'HBM passes over K,V':>22}")
print("-" * 56)
fits = {}
for br in (16, 32, 64, 128, 256, 512):
    # A tile holds Q rows, K rows, V rows and the score block.
    tile = (3.0 * br * D_HEAD + br * br) * BYTES
    ok = tile <= SMEM
    fits[br] = (tile, ok)
    passes = 1 if ok else 0
    print(f"{br:>11}{tile / 1024:>11.1f}K{('yes' if ok else 'no'):>8}"
          f"{(str(1) if ok else 'spills'):>22}")

print()
print()
print("And the tier that is easy to forget. Moving a model's weights across each")
print("link, for a 14 GB model:")
print()
WEIGHTS = 14.0e9
print(f"{'link':>16}{'bandwidth GB/s':>18}{'time to move 14 GB':>22}")
print("-" * 58)
move = {}
for name, bw, cap in TIERS:
    if name in ("registers", "shared / L1", "L2 cache"):
        continue
    t = WEIGHTS / bw
    move[name] = t
    print(f"{name:>16}{bw / 1e9:>18.0f}{t:>20.2f}s")

print(f"""
The tier table is the correction to ch:inf-cpu-gpu's single ridge. HBM needs
{bal['HBM']:.0f} operations per byte to keep the arithmetic units fed. Shared memory
needs {bal['shared / L1']:.0f} -- {bal['HBM'] / bal['shared / L1']:.0f} times less --
because it delivers {TIERS[1][1] / TIERS[3][1]:.0f} times the bandwidth
(eq:roofline-has-multiple-ridges).

**An operation that is hopelessly memory-bound against HBM can be comfortably
compute-bound against shared memory**, and the entire engineering question is whether
its working set can be arranged to live there.

The attention tables are that question answered for the operation where it matters
most. Naive attention at sequence length {SEQS[2]} moves
{att[SEQS[2]][0] / 1e6:.1f} MB through HBM, of which almost all is the
{SEQS[2]}-by-{SEQS[2]} score matrix written out and read back twice. Tiled attention
moves {att[SEQS[2]][1] / 1e6:.1f} MB -- a reduction of
{att[SEQS[2]][0] / att[SEQS[2]][1]:.0f} times -- because the score matrix never leaves
the chip.

Note what did NOT change: the FLOPs. Both do {att[SEQS[2]][2] / 1e9:.1f} GFLOP, and
cite:dao2022flash computes the same exact result. **The speedup is entirely a change
in which tier the traffic goes to**, which is why it required no accuracy trade-off
and why adoption was immediate.

The intensity table shows the regime change. Naive attention at {SEQS[2]} has HBM
intensity {inten[SEQS[2]][0]:.1f}, which is below the HBM balance point of
{bal['HBM']:.0f} -- memory-bound. Tiled has {inten[SEQS[2]][1]:.1f}, far above it --
compute-bound, running at {att[SEQS[2]][2] / times[SEQS[2]][1] / PEAK:.1%} of peak
arithmetic.

**That is what "moving an operation across the ridge" means in practice**, and it is
worth {times[SEQS[2]][0] / times[SEQS[2]][1]:.1f}x here.

Notice that the speedup does NOT keep growing: it is
{times[SEQS[1]][0] / times[SEQS[1]][1]:.1f}x at {SEQS[1]} and
{times[SEQS[-1]][0] / times[SEQS[-1]][1]:.1f}x at {SEQS[-1]}, essentially flat. That
is because tiling has already done everything it can -- the tiled column reaches
{att[SEQS[-1]][2] / times[SEQS[-1]][1] / PEAK:.0%} of peak arithmetic, and there is
nowhere left to go.

**A tier-crossing optimisation has a hard ceiling: peak.** Once an operation is
compute-bound, further reductions in memory traffic buy exactly nothing, which is
worth knowing before spending a quarter on the next one. The remaining lever at that
point is arithmetic -- fewer FLOPs, or cheaper ones -- and that is a different
project entirely.

The tile table is the constraint that makes this an engineering problem rather than a
free win. Shared memory is {SMEM / 1024:.0f} KB per streaming multiprocessor, so a
tile of {[b for b in fits if fits[b][1]][-1]} rows fits and
{[b for b in fits if not fits[b][1]][0]} rows does not. **The tier with the good
balance point is the tier with almost no capacity**, and that trade -- bandwidth
against capacity -- is what the whole hierarchy is.

The last table is the tier people forget until it ruins a design. Moving
{WEIGHTS / 1e9:.0f} GB of weights takes {move['HBM']:.2f}s from HBM,
{move['NVLink peer']:.2f}s across NVLink, and {move['PCIe host']:.2f}s across PCIe.

That last figure is why **model loading, host offload, and cross-node weight movement
are architectural decisions rather than implementation details**. A design that swaps
models per request pays {move['PCIe host']:.2f} seconds every time, which is
{move['PCIe host'] / (14.0e9 / TIERS[3][1]):.0f} times the cost of simply reading the
weights it already has resident -- and ch:inf-kubernetes has to plan capacity around
it.""")
```

## 9. Practical Example

Every tier has its own balance point:

```
            tier    bandwidth GB/s   balance point   easier than HBM
--------------------------------------------------------------------
       registers            220000            4 F/B             65.7x
     shared / L1            128000            8 F/B             38.2x
        L2 cache             11000           90 F/B              3.3x
             HBM              3350          295 F/B              1.0x
     NVLink peer               900         1099 F/B              0.3x
       PCIe host                64        15453 F/B              0.0x
```

Shared memory needs **8** operations per byte against HBM's **295**
({{eq:roofline-has-multiple-ridges}}). PCIe needs **15,453** — which is why host
offload is not a memory extension.

Attention, before and after tiling:

```
   seq len   naive HBM MB   tiled HBM MB   reduction     GFLOP
--------------------------------------------------------------
       512            2.6            0.5        5.0x       0.1
      2048           35.7            2.1       17.0x       2.1
      8192          545.3            8.4       65.0x      34.4
     32768         8623.5           33.6      257.0x     549.8
```

At 8192 tokens, tiling cuts HBM traffic **65×**. **The FLOPs did not change** — both
compute 34.4 GFLOP and {{cite:dao2022flash}} returns the exact same result. The
speedup is entirely a change of tier.

```
   seq len     naive I     tiled I   naive bound   tiled bound
--------------------------------------------------------------
       512        51.2       256.0        memory        memory
      2048        60.2      1024.0        memory       compute
      8192        63.0      4096.0        memory       compute
     32768        63.8     16384.0        memory       compute
```

Naive attention sits at intensity ~63 regardless of length — permanently memory-bound.
Tiled crosses the ridge at 2048 tokens.

```
   seq len    naive ms    tiled ms    speedup   tiled % of peak
---------------------------------------------------------------
       512       0.001       0.000       5.0x             86.7%
      2048       0.011       0.002       4.9x            100.0%
      8192       0.163       0.035       4.7x            100.0%
     32768       2.574       0.556       4.6x            100.0%
```

The speedup is **essentially flat** at ~4.7×, not growing. Tiled attention reaches
**100% of peak arithmetic**, and there is nowhere left to go
({{eq:tier-crossing-has-a-ceiling}}). **A tier-crossing optimisation has a hard
ceiling: peak.** Knowing that before funding the next one is worth something.

```mermaid {#fig:tiers caption="An operation is bound by whichever tier it is worst against. Restructuring to keep intermediates in a faster tier moves it across that tier's ridge — but the gain stops at peak arithmetic."}
flowchart LR
  A["naive attention<br/>I = 63 vs HBM ridge 295"] --> B["memory-bound<br/>4.7x slower than peak"]
  C["tiled attention<br/>scores stay in SRAM"] --> D["I = 4096<br/>compute-bound"]
  D --> E["100% of peak<br/>ceiling reached"]
```

The second listing turns to capacity.

```python {tier=A name=ce2}
"""What actually fits, and why the answer is a frontier rather than a number.

ch:inf-cpu-gpu found batch size to be the lever that makes a GPU worth using. This
listing asks what bounds it, and the answer is memory capacity: weights are fixed, KV
cache scales with batch TIMES context, and the two trade against each other along a
hyperbola (eq:batch-times-context-is-the-budget).

The practical consequence is that "maximum batch size" and "maximum context length"
are not two configuration values. They are one curve, and a system configured with
both set independently will either waste memory or fail under a load it was told it
could handle.

This listing also measures what fragmentation costs, which is the gap
cite:kwon2023pagedattention closes.
"""
HBM = 80.0e9
LAYERS = 32
D_MODEL = 4096
N_KV_HEADS = 8            # grouped-query
HEAD_DIM = 128
PARAMS = 7.0e9

CONTEXTS = [1024, 4096, 16384, 65536, 262144]
QUANTS = [("bf16", 2.0), ("fp8", 1.0), ("int4", 0.5)]


def kv_per_token(bytes_per=2.0):
    return 2.0 * LAYERS * N_KV_HEADS * HEAD_DIM * bytes_per


def weights(bytes_per):
    return PARAMS * bytes_per


def activation_overhead():
    """Working buffers, CUDA context, allocator reserve. Roughly fixed."""
    return 3.5e9


print("An 80 GB device. Fixed costs first.")
print()
print(f"{'weight format':>16}{'weights GB':>13}{'overhead GB':>14}"
      f"{'left for KV GB':>17}")
print("-" * 60)
free = {}
for name, b in QUANTS:
    w = weights(b)
    f = HBM - w - activation_overhead()
    free[name] = f
    print(f"{name:>16}{w / 1e9:>13.1f}{activation_overhead() / 1e9:>14.1f}"
          f"{f / 1e9:>17.1f}")

print()
print("KV cache per token, bf16 cache, %d KV heads: %.3f MB"
      % (N_KV_HEADS, kv_per_token() / 1e6))

print()
print()
print("The frontier: maximum batch size by context length, for each weight format.")
print("This is batch times context held to a constant.")
print()
print(f"{'context':>10}" + "".join(f"{n:>14}" for n, _ in QUANTS))
print("-" * 52)
front = {}
for c in CONTEXTS:
    row = []
    for name, b in QUANTS:
        m = int(free[name] / (c * kv_per_token()))
        row.append(m)
    front[c] = row
    print(f"{c:>10}" + "".join(f"{v:>14}" for v in row))
print()
print("(maximum concurrent sequences)")

print()
print()
print("The same as a product, which is the quantity actually conserved.")
print()
print(f"{'weight format':>16}{'batch x context':>18}{'vs bf16':>10}")
print("-" * 46)
prod = {}
for name, b in QUANTS:
    p = free[name] / kv_per_token()
    prod[name] = p
    print(f"{name:>16}{p:>18.0f}{p / prod['bf16']:>9.1f}x")

print()
print("Any (batch, context) pair whose product is under that number fits.")

print()
print()
print("Quantising the CACHE as well, which is a separate decision from quantising")
print("the weights.")
print()
print(f"{'weights':>10}{'KV cache':>11}{'KV MB/token':>14}"
      f"{'batch x context':>18}{'vs bf16/bf16':>15}")
print("-" * 70)
base = None
combo = {}
for wn, wb in QUANTS:
    for kn, kb in QUANTS:
        f = HBM - weights(wb) - activation_overhead()
        per = kv_per_token(kb)
        p = f / per
        if base is None:
            base = p
        combo[(wn, kn)] = p
        print(f"{wn:>10}{kn:>11}{per / 1e6:>14.3f}{p:>18.0f}{p / base:>14.1f}x")

print()
print()
print("What fragmentation costs. Without paging, a sequence reserves its MAXIMUM")
print("possible length up front, because the allocation must be contiguous.")
print()
MAXLEN = 8192
print(f"reserved length per sequence: {MAXLEN}")
print()
print(f"{'actual mean length':>20}{'utilisation':>14}{'effective batch':>18}"
      f"{'paged batch':>14}{'gain':>9}")
print("-" * 76)
frag = {}
for actual in (180, 640, 2100, 5400, 8192):
    naive_batch = int(free["bf16"] / (MAXLEN * kv_per_token()))
    paged_batch = int(free["bf16"] / (actual * kv_per_token()))
    util = actual / float(MAXLEN)
    frag[actual] = (util, naive_batch, paged_batch)
    print(f"{actual:>20}{util:>14.1%}{naive_batch:>18}{paged_batch:>14}"
          f"{paged_batch / float(naive_batch):>8.1f}x")

print()
print()
print("And the cost of getting the frontier wrong: configuring a batch and a")
print("context independently, then meeting a request mix that uses both.")
print()
CFG_BATCH = 64
CFG_CTX = 16384
need = CFG_BATCH * CFG_CTX * kv_per_token()
print(f"configured: batch {CFG_BATCH}, context {CFG_CTX}")
print(f"KV needed if both are used at once: {need / 1e9:.1f} GB")
print(f"available at bf16 weights:          {free['bf16'] / 1e9:.1f} GB")
print(f"shortfall:                          {(need - free['bf16']) / 1e9:.1f} GB")
print()
print(f"{'weight format':>16}{'KV available GB':>18}{'KV needed GB':>15}"
      f"{'fits':>8}")
print("-" * 58)
for name, b in QUANTS:
    print(f"{name:>16}{free[name] / 1e9:>18.1f}{need / 1e9:>15.1f}"
          f"{('yes' if free[name] >= need else 'no'):>8}")

print(f"""
The fixed-cost table sets up everything else. At bf16 weights, a
{HBM / 1e9:.0f} GB device has {free['bf16'] / 1e9:.1f} GB left for KV cache after
weights and working buffers. At int4 it has {free['int4'] / 1e9:.1f} GB --
{free['int4'] / free['bf16']:.1f} times more.

That ratio is the first thing worth noticing, because it is much smaller than the
weight saving suggests. Shrinking a {weights(2.0) / 1e9:.0f} GB model to
{weights(0.5) / 1e9:.1f} GB is a {weights(2.0) / weights(0.5):.0f}-fold reduction in
weights and only a {free['int4'] / free['bf16']:.1f}-fold gain in serving capacity,
because the weights were never the thing consuming most of the device.

**On an {HBM / 1e9:.0f} GB card, a {PARAMS / 1e9:.0f}B model's weights are
{weights(2.0) / HBM:.0%} of memory and the cache is most of the rest.** Quantising
weights is therefore a way to fit a model that did not fit, and only marginally a way
to serve more of one that did -- a distinction the memory-footprint framing loses, and
one the cache-quantisation table below makes sharp.

The frontier table is the chapter's main point. At {CONTEXTS[0]} tokens of context a
bf16 deployment serves {front[CONTEXTS[0]][0]} concurrent sequences; at
{CONTEXTS[3]} tokens it serves {front[CONTEXTS[3]][0]}
(eq:batch-times-context-is-the-budget).

**These are not two settings. They are one curve**, and the conserved quantity is the
product: {prod['bf16']:.0f} token-slots at bf16, {prod['int4']:.0f} at int4. Any
(batch, context) pair whose product is under that number fits, and any pair over it
does not, regardless of how the two numbers were arrived at.

The cache-quantisation table separates a decision that is usually made once for both.
Quantising weights to int4 while leaving the cache at bf16 gives
{combo[('int4', 'bf16')] / base:.1f}x the token-slots. Quantising the cache to int4
while leaving weights at bf16 gives {combo[('bf16', 'int4')] / base:.1f}x. Doing both
gives {combo[('int4', 'int4')] / base:.1f}x.

**The cache is the larger lever**, and it is the one usually left alone -- partly
because cache quantisation has a quality cost that is harder to measure than weight
quantisation's, and partly because the tooling makes weights the obvious knob.
ch:q-activation-kv has the quality side; the capacity side is this table.

The fragmentation table is what cite:kwon2023pagedattention addresses. Without
paging, a sequence must reserve its maximum possible length contiguously, so a
deployment allowing {MAXLEN}-token contexts reserves {MAXLEN} tokens for every
sequence -- even the ones that turn out to be {180} tokens long.

At a mean actual length of {640}, that is {frag[640][0]:.1%} utilisation and an
effective batch of {frag[640][1]} against a paged batch of {frag[640][2]} --
**{frag[640][2] / float(frag[640][1]):.1f} times the concurrency for the same
memory**.

The gain is exactly the inverse of the utilisation, which makes it predictable: a
deployment whose requests use a tenth of their allowed context gets roughly ten times
the batch from paging. **Paging does not make memory bigger; it stops a length
distribution from being charged at its maximum.**

The last table is the failure this all exists to prevent. Configuring batch
{CFG_BATCH} and context {CFG_CTX} independently looks reasonable -- each is a
defensible number -- and together they require {need / 1e9:.1f} GB of cache against
{free['bf16'] / 1e9:.1f} GB available. The configuration is
{need / free['bf16']:.1f} times oversubscribed.

It will also work perfectly in testing, because tests rarely produce {CFG_BATCH}
simultaneous {CFG_CTX}-token requests. **The failure arrives as an
out-of-memory error under a load the configuration explicitly permitted**, which is
the most confusing shape an incident can have: nothing exceeded a limit, and the
system still ran out.

The fix is to configure the product and derive the pair, which is one line of
arithmetic and almost never done.""")
```

Fixed costs on an 80 GB device:

```
   weight format   weights GB   overhead GB   left for KV GB
------------------------------------------------------------
            bf16         14.0           3.5             62.5
             fp8          7.0           3.5             69.5
            int4          3.5           3.5             73.0
```

A 7B model's weights are **18%** of the device. Quantising them 4× gains only **1.2×**
serving capacity, because the weights were never what was consuming it.

The frontier:

```
   context          bf16           fp8          int4
----------------------------------------------------
      1024           465           517           543
      4096           116           129           135
     16384            29            32            33
     65536             7             8             8
    262144             1             2             2
```

**These are not two settings, they are one curve** — the conserved quantity is
**476,837** token-slots at bf16 ({{eq:batch-times-context-is-the-budget}}).

Separating the two quantisation decisions:

```
   weights   KV cache   KV MB/token   batch x context   vs bf16/bf16
----------------------------------------------------------------------
      bf16       bf16         0.131            476837           1.0x
      bf16       int4         0.033           1907349           4.0x
      int4       bf16         0.131            556946           1.2x
      int4       int4         0.033           2227783           4.7x
```

Quantising the cache alone buys **4.0×**; quantising the weights alone buys **1.2×**
({{eq:cache-quantisation-is-the-larger-lever}}). **The cache is the larger lever and
it is the one usually left alone.**

What fragmentation costs:

```
  actual mean length   utilisation   effective batch   paged batch     gain
----------------------------------------------------------------------------
                 180          2.2%                58          2649    45.7x
                 640          7.8%                58           745    12.8x
                2100         25.6%                58           227     3.9x
                5400         65.9%                58            88     1.5x
                8192        100.0%                58            58     1.0x
```

Paging's gain is exactly the inverse of the utilisation it replaces — **45.7×** at
2.2%, **1.0×** at 100%. It does not make memory bigger; it stops a length distribution
being charged at its maximum.

And the configuration error this all exists to prevent: batch 64 with context 16,384
requires **137.4 GB** of cache against **62.5 GB** available — **2.2× oversubscribed**,
by a configuration in which neither number is unreasonable and no limit is exceeded.

## 10. Production Considerations

Configure the *product* and derive the pair. One line of arithmetic, and it prevents
the failure mode where nothing exceeded a limit and the system still ran out of memory.

Measure the working-buffer reserve on your actual stack rather than assuming. It is
several gigabytes, it varies with framework version and parallelism configuration, and
it is subtracted before anything else.

Quantise the cache before quantising the weights, if serving capacity is the goal.
Reverse that order if fitting the model at all is the goal — they are different
problems with different answers.

Measure your length utilisation before attributing gains to paging. The benefit is
$1/u$, so a deployment with high utilisation should not expect one.

Use prefix sharing where a system prompt or a common retrieved document is shared.
It is the only mechanism here that makes cache genuinely shared, and its benefit is
proportional to the shared prefix length.

Do not treat host memory as an extension of device memory for decode. The 52× penalty
converts a memory-bound workload into an unusable one; offload belongs to training.

Record the token-slot budget somewhere an operator will see it, expressed in the same
units as the configuration. The arithmetic is trivial and the failure it prevents is
one where nothing exceeded a limit; a number printed at startup saying "this
configuration permits 1,048,576 token-slots and the device holds 476,837" would end
the entire class of incident.

Test at the frontier rather than at typical load. A synthetic load that saturates the
configured batch *at* the configured context is the only test that exercises the
oversubscription, and it takes minutes to write. Most load tests use realistic request
mixes, which is exactly why they miss it.

Alert on token-slot utilisation, not on batch or context separately. It is the
conserved quantity and the only one whose exhaustion predicts failure.

## 11. Common Mistakes

**Reasoning from one roofline.** The binding tier depends on the operation, and an
optimisation aimed at the wrong tier changes nothing measurable.

**Expecting tier-crossing gains to compound.** They stop at peak.

**Configuring batch and context independently.** They are one hyperbola, and the
invalid region is the one no validator checks because it spans two fields.

**Quantising weights for serving capacity.** Buys 1.2× where the cache buys 4.0×.

**Attributing paging's gain to the technique rather than to your length
distribution.** The gain is $1/u$.

**Treating host offload as more memory.** It is a tier 52× slower.

## 12. Failure Modes

**Configuration oversubscription.** Batch and context each individually reasonable,
jointly impossible; fails in production and passes every test.

**Silent reserve growth.** A framework upgrade increases the allocator reserve and the
achievable batch drops with no configuration change.

**Fragmentation regression from a length-distribution shift.** A product change
lengthens requests, utilisation rises, and paging's benefit evaporates — appearing as
a throughput regression with no deploy to blame.

**Cache quantisation quality drift.** Outlier keys or values degrade accuracy on a
subset of requests, invisible to every capacity metric and to availability monitoring
({{eq:semantic-failure-has-no-instrument}}).

**Tile spill.** A kernel configured for a tile that does not fit shared memory falls
back to a slower path, losing the tier-crossing gain silently. The symptom is a
performance regression after a shape change with no error and no warning.

**Offload as a capacity fix.** Under memory pressure, enabling host offload appears
to solve the problem and converts a bandwidth-bound workload into one running at
1/52nd the bandwidth. The configuration change is small, the effect is catastrophic,
and the metric that would show it is achieved bandwidth rather than anything about
memory.

## 13. Alternatives

**Smaller models.** Frees both weight memory and, via fewer layers and heads, cache per
token. Usually the largest single lever and the one with the clearest quality cost,
and the only one on this list that improves both the bandwidth and the capacity side
at once rather than trading one for the other.

**Fewer KV heads.** {{ch:inf-cpu-gpu}} showed grouped-query moving both the bandwidth
crossover and, here, the capacity frontier. Fixed at pre-training.

**Cache eviction or compression.** Discard or compress old cache entries. Trades
quality on long-range dependencies for capacity, and the trade is task-dependent enough
that it needs measuring per deployment.

**More devices.** {{ch:inf-parallelism}} takes this up; note that sharding weights
across devices frees capacity superlinearly, since each device holds a fraction of $W$
while $C$ scales linearly.

**Accept a smaller context window.** Frequently correct and rarely considered, because
the maximum context the model supports is treated as a specification rather than a
choice.

## 14. Evaluation

Report token-slots, not maximum batch or maximum context. It is the conserved quantity
and the one that composes.

Measure achieved fraction of peak arithmetic per kernel, not per model. A model at 40%
of peak may contain one kernel at 95% and another at 5%, and only the second is worth
work.

Track length utilisation as a standing metric. It determines paging's value, it moves
with product changes, and nothing else reports it.

Validate the memory model against a real out-of-memory boundary once. If the device
fails at a token-slot count materially below the computed one, the working-buffer term
is larger than assumed and every capacity plan built on it is wrong.

Evaluate cache quantisation on long-context tasks specifically. Its failure mode is
degradation on long-range dependencies, which short-context evaluations cannot see.

## 15. Advanced Concepts

The tier model treats bandwidths as fixed, but achieved bandwidth depends on access
pattern: coalesced sequential reads reach near peak while scattered reads can achieve a
small fraction of it. A paged KV cache is, by construction, scattered — pages are
allocated wherever there is room — so paging trades a capacity gain for a bandwidth
loss. {{cite:kwon2023pagedattention}} keeps the loss small through careful block sizing,
but the trade exists and grows as fragmentation increases. **Paging's benefit is
therefore not purely additive with the bandwidth analysis of {{ch:inf-cpu-gpu}}**, and
a deployment at very high fragmentation can lose more bandwidth than it gains capacity.

{{eq:tier-crossing-has-a-ceiling}} assumes a single binding tier. With multiple tiers
active simultaneously — an operation reading weights from HBM while spilling
intermediates to L2 — the ceiling is set by the *sum* of tier times if the transfers
serialise and by the maximum if they overlap. Which regime applies depends on whether
the kernel is written to overlap, and that is why hand-tuned kernels can beat the
model's prediction: they are converting a sum into a maximum.

There is also a question about what "peak" means in {{eq:tier-crossing-has-a-ceiling}}.
The peak used here is dense bf16 arithmetic, but modern devices have several peaks --
sparse formats, lower precisions, and specialised units each with their own maximum.
An operation at 100% of dense bf16 peak is at roughly 50% of fp8 peak and a smaller
fraction still of sparse peak. So the ceiling is not absolute; it is the ceiling *for
the numeric format you chose*, and switching format raises it. That is a materially
different lever from tiling and it composes with it, which is why quantised kernels
and tiled kernels are developed by the same teams and shipped together.

The capacity frontier assumes cache is the only variable term, which stops being true
for very large batches where activation memory for the largest layer scales with batch.
At the batch sizes where {{ch:inf-cpu-gpu}}'s bandwidth argument wants to operate, that
term is usually second-order; for prefill with large chunk sizes it is not, and
{{ch:inf-batching}} takes it up.

## 16. Connection to Previous Chapters

{{eq:decode-is-bandwidth-bound}} from {{ch:inf-cpu-gpu}} is refined here: decode is
bandwidth-bound *against HBM*, and the question of whether any tier could serve it
better is answered negatively, because weights genuinely do not fit in SRAM.

{{eq:kv-traffic-overtakes-weights}} and {{eq:batch-times-context-is-the-budget}} are the
bandwidth and capacity faces of the same quantity. The first says batching stops
helping past a context; the second says it stops being *possible*.

{{eq:batch-is-the-mechanism-not-an-optimisation}} needs
{{eq:batch-times-context-is-the-budget}} to be actionable: knowing you need a large
batch is useless without knowing what bounds it.

{{eq:access-shape-decides-the-store}} from {{ch:sd-storage}} priced storage by
operations per byte. This chapter is the same trade inside a single device.

## 17. Exercises

1. Compute the balance point for each tier of a device with 400 TFLOP/s, 1.6 TB/s HBM,
   and 90 TB/s shared memory. Which operations change classification?

2. Derive {{eq:tier-crossing-has-a-ceiling}} and use it to bound the achievable
   speedup from a hypothetical optimisation that eliminates *all* HBM traffic for
   decode.

3. A 70B model at fp8, 80 layers, 8 KV heads, head dim 128, on a 141 GB device. Give
   the token-slot budget and the maximum batch at 32k context.

4. Extend the second listing so activation memory scales with batch. At what batch does
   it become the binding term?

5. Measure length utilisation for a workload you have access to. What would paging buy,
   and does that match what your stack reports?

## 18. Interview Questions

1. FlashAttention gives a 4× speedup and computes the same result. Where did the
   speedup come from?

2. Why does that speedup stop growing with sequence length?

3. We set max batch 64 and max context 16k. What did we actually configure?

4. Should we quantise weights or KV cache to serve more concurrent users? Justify with
   arithmetic.

5. Our vendor says paging gives 20× throughput. What do you need to know before
   believing it?

6. A kernel was running at 100% of peak and someone proposes another round of
   memory optimisation on it. What do you say, and what would you propose instead?

## 19. Research Questions

1. What is the achieved-bandwidth cost of paged cache scatter at realistic
   fragmentation, and where does it cancel the capacity gain?

2. Can KV cache be compressed with a quality cost measurable cheaply enough to tune per
   deployment?

3. Is there a tier assignment for decode weights — partial residency, streaming
   prefetch — that beats reading all of HBM every step, given real access patterns?

4. How much of the working-buffer reserve is genuinely necessary, and how much is
   allocator conservatism that could be recovered?

## 20. Chapter Summary

There is a roofline per memory tier, and an operation is bound by whichever it is worst
against ({{eq:roofline-has-multiple-ridges}}). HBM needs **295** operations per byte;
shared memory needs **8**; PCIe needs **15,453**.

{{cite:dao2022flash}} exploits that: tiling cuts HBM traffic **65×** at 8192 tokens
while computing the identical result, moving attention from intensity 63 to 4096 and
across the ridge. The speedup is **~4.7×** and **flat**, because tiled attention runs
at **100% of peak** and a tier-crossing optimisation cannot exceed peak
({{eq:tier-crossing-has-a-ceiling}}).

Capacity bounds the batching {{ch:inf-cpu-gpu}} showed to be essential. Weights are
fixed; cache scales with batch times context, so the conserved quantity is
**476,837 token-slots** on an 80 GB device
({{eq:batch-times-context-is-the-budget}}) — 465 sequences at 1024 tokens or 7 at
65,536, on one setting rather than two.

Quantising the cache buys **4.0×** those slots; quantising the weights buys **1.2×**
({{eq:cache-quantisation-is-the-larger-lever}}), because a 7B model's weights are only
**18%** of an 80 GB device.

Paging recovers exactly the inverse of length utilisation — **45.7×** at 2.2%, **1.0×**
at 100% — so its benefit is a property of your length distribution, not of the
technique.

Both halves of this chapter are corrections to a simplification that was useful
before it was wrong. One roofline is the right first model and it stops being right
the moment you try to explain why a specific optimisation worked. One memory number
is the right first model and it stops being right the moment two configuration
fields multiply. In each case the refinement does not overturn the original result;
it says which term the original was hiding, and the hidden term is where the
engineering lives.

Carry forward: **configure the product, not the pair**, and **the cache is where the
capacity is**.

## 21. Further Reading

- {{cite:dao2022flash}} — the canonical tier-crossing optimisation, and why exactness
  mattered for adoption.
- {{cite:kwon2023pagedattention}} — paging, prefix sharing, and the fragmentation this
  chapter prices.
- {{cite:pope2022inference}} — the memory arithmetic underlying both halves.
- {{cite:rajbhandari2020zero}} — partitioning rather than replicating; the training-side
  answer to the same capacity question.

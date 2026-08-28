---
id: q-memory-math
number: 143
part: XV
tier: full
status: draft
requires: [q-activation-kv, q-gguf, tf-complexity]
provides: [inference-budget, binding-term, prefill-is-the-peak,
           chunked-prefill, capacity-not-size, admission-control-memory]
citations: [kwon2023pagedattention, dao2022flash, pope2022inference,
            liu2024kivi, dettmers2023case4bit]
---

## 1. Learning Objectives

By the end of this chapter you will be able to compute every term of an inference
memory budget and name which one binds; explain why parameter count is close to
useless as a predictor of serving capacity; compute the **peak** rather than the
steady state, and say why a deployment sized on decode dies on its first long
prompt; and size chunked prefill and prefill admission control from the budget
rather than from a default.

## 2. Why This Matters

"A 70B model at 4 bits is 35 GB, so it fits on a 48 GB card" is the calculation
everyone does. **It is wrong more often than right**, because the weights are one
of five terms and usually not the one that binds.

{{sec:9-practical-example}} computes all of them and reports the **binding term**
across a grid. At 4k context and batch 1, weights bind — the case the folklore
describes, and one that occurs in demos rather than deployments. At 4k and batch
32, a 7B multi-head model has **68.7 GB of cache against 3.5 GB of weights**. **The
model everybody calls small has become a cache problem**, and quantizing its
weights further would not help at all.

**Then the result that should change how models are chosen.** On an 80 GB card at
8k context, a 70B GQA-8 model fits **16** concurrent sequences and a 7B
multi-head model fits **17**.

> **Ten times the parameters, the same number of concurrent users.** Capacity is
> governed by layers × KV heads × head dimension, not by parameter count — and an
> 8B GQA-8 model fits **69**, four times either of them.

**And a budget that fits can still fail.** A request has two phases.
{{sec:9-practical-example}} computes the prefill peak: at a 131k prompt, the
attention score matrix is **2,199 GB** without a fused kernel. One tensor, one
layer, one sequence.

**Fused attention removes it entirely** — and the remaining peak is **92.0 GB on
an 80 GB card**, so it *still* does not fit. Chunked prefill at 2048 tokens brings
it to **79.4 GB**.

**Worst of all, some configurations fail only under load.** A batch-48 deployment
survives one long prefill at **69.2 GB** and needs **107.1 GB** if two arrive
together — **a capacity bug that passes every sequential benchmark.**

{{maturity:ESTABLISHED}} Memory accounting, fused attention.
{{maturity:MATURE}} Chunked prefill. {{maturity:EMERGING}} Treating prefill
admission as a separate capacity.

## 3. Prerequisites

{{ch:q-activation-kv}} for {{eq:kv-scales-with-traffic}} and the allocator;
{{ch:q-gguf}} for {{eq:decode-roofline}}, of which this is the memory half;
{{ch:tf-complexity}} for the quadratic attention term.

## 4. Intuitive Explanation

### Five terms, and the useful output is a name

$$ \text{weights} \;+\; \text{KV cache} \;+\; \text{activations} \;+\; \text{framework} \;+\; \text{allocator waste} $$

**The number that matters is not the total but which term is largest**, because
that is the thing to fix and everything else is wasted effort.

```text
   model        context   batch   weights   KV     activ.   total   binding
   ──────────   ───────   ─────   ───────   ────   ──────   ─────   ───────
   7B  MHA        4,096       1       3.5    2.1     0.00     6.8   weights
   7B  MHA        4,096      32       3.5   68.7     0.00    73.4   kv
   7B  MHA       32,768       1       3.5   17.2     0.00    21.9   kv
   70B GQA-8      4,096       1      35.0    1.3     0.00    37.5   weights
   70B GQA-8     32,768      16      35.0  171.8     0.00   208.0   kv
```

**One row into a realistic configuration and the answer changes.** The 7B model at
batch 32 is a cache problem; further weight quantization does nothing for it.

### And optimising moves the constraint, or fails to

Applying 4-bit cache quantization to the "too big" rows moves **279.6 → 73.4** and
**208.0 → 79.2**, so they fit. **And the binding column does not change** — the
cache still binds in every row, because it was ten to eighty times the weights
before.

> **A 4× lever applied to a 20× problem leaves a 5× problem.** The next move is
> not more cache quantization but the architectural lever, or shorter contexts, or
> less concurrency — and the binding column says so before any of them are tried.

### Parameter count does not predict capacity

```text
   model         card    context   16-bit cache   4-bit cache
   ──────────   ─────    ───────   ────────────   ───────────
   7B  MHA      80 GB      8,192             17            70
   8B  GQA-8    80 GB      8,192             69           278
   70B GQA-8    80 GB      8,192             16            65
```

**A 70B model and a 7B model serve the same concurrency.** The 70B has more layers
and the 7B has eight times more KV heads, and those nearly cancel. The 8B GQA-8
model has the small model's layer count *and* the large model's head grouping, so
it serves four times either.

**The number in a model's name — which determines its price, its reputation, and
the hardware people budget for — tells you almost nothing about how many users it
can serve.**

### Prefill is a different machine

Decode processes one token per sequence, so its activations are negligible —
hundredths of a gigabyte in the table above. **Prefill processes the whole prompt
at once.**

```text
   prompt      KV   linear   scores    scores   peak, no   peak,
   tokens             activ.  naive     fused    fusion    fused
   ───────   ────   ──────   ──────   ──────   ────────   ─────
     16,384    5.4     1.61     34.4      0.0       77.5    43.2
     65,536   21.5     6.44    549.8      0.0      613.9    64.1
    131,072   42.9    12.88   2199.0      0.0     2291.1    92.0
```

**2,199 GB.** One tensor, one layer, one sequence — quadratic in the prompt
length, and quadratic terms do not stay small.

**A fused attention kernel never materialises it**, computing attention in tiles
and keeping only running softmax statistics.

> **Without fused attention, long-context inference is not slow — it is
> impossible, on any hardware, for prompts people now routinely send. The
> technique that made long context practical was not a bigger card.**

### And fusion is not sufficient

The fused peak at 131k is **92.0 GB against an 80 GB card**. Fusion removed a
2,199 GB term and left a configuration over budget by **12 GB**.

**Chunked prefill** processes the prompt in pieces of $C$ tokens, appending each
piece's keys and values to the cache. The linear activation term becomes
proportional to $C$:

```text
   chunk C     linear activ.    peak    fits 80 GB   passes
   ─────────   ─────────────   ──────   ──────────   ──────
   131,072          12.88 GB   92.0 GB          NO        1
     8,192           0.81 GB   80.0 GB         yes       16
     2,048           0.20 GB   79.4 GB         yes       64
```

**The KV cache still grows to the full prompt** — that is the point of prefilling.
Only the activation term is bounded, and the cost is sequential passes, not extra
arithmetic.

### The failure that passes every benchmark

```text
   batch   context   decode   prefill 1   prefill all   verdict
   ─────   ───────   ──────   ─────────   ───────────   ────────────────────
      16     8,192   46.9      47.7        59.8         fine
      16    32,768   79.2      82.4       130.7         OOM on one prompt
      48     8,192   68.4      69.2       107.1         OOM if prompts overlap
```

**Only the decode column appears in a steady-state calculation.** A deployment
sized on it picks a batch that fits comfortably, and then a long prompt arrives —
and prefill for that *one* sequence allocates activations proportional to its
whole prompt while every other sequence's cache is still resident.

**And the batch-48 row is the worst case**: it survives one long prefill and fails
when two overlap. **A load-dependent failure that will not reproduce under a
benchmark sending one request at a time.**

## 5. Formal Explanation

### 5.1 The budget

$$ M_{\text{total}} = \underbrace{P\,\frac{b_w}{8}}_{\text{weights}} + \underbrace{\frac{2 L h_{\text{kv}} d_h S B\, b_{\text{kv}}/8}{u}}_{\text{cache}} + \underbrace{c\,B\,T\,d_{\text{model}}\,b_a/8}_{\text{activations}} + \underbrace{F}_{\text{framework}} $$ (eq:inference-budget)

with $u$ the allocator utilisation and $T$ the tokens processed per step — **1
during decode, the prompt length during prefill.** That single substitution is the
difference between the two phases.

$$ \text{binding term} = \arg\max_{\text{terms}} M_i $$ (eq:binding-term)

**{{eq:binding-term}} is the output to compute**, because every optimisation acts
on one term and only the largest one matters.

### 5.2 Capacity, solved for batch

$$ B_{\max} = \frac{M_{\text{card}} - P b_w/8 - F}{2 L h_{\text{kv}} d_h S\, b_{\text{kv}}/8 + c\,d_{\text{model}} b_a/8} $$ (eq:capacity-not-size)

**{{eq:capacity-not-size}} has $P$ in the numerator and $L h_{\text{kv}} d_h$ in
the denominator**, and those are only loosely related. Hence the measured
coincidence: a 70B GQA-8 model and a 7B MHA model differ by 10× in $P$ and by
roughly 10× in $L h_{\text{kv}}$ too, so the ratio cancels.

$$ \frac{B_{\max}^{(1)}}{B_{\max}^{(2)}} \approx \frac{L^{(2)} h_{\text{kv}}^{(2)}}{L^{(1)} h_{\text{kv}}^{(1)}} \quad \text{once } P b_w/8 \ll M_{\text{card}} $$ (eq:capacity-ratio)

### 5.3 The prefill peak

Attention over $T$ tokens materialises, per layer and head, a $T \times T$ score
matrix:

$$ M_{\text{scores}} = B\,h\,T^{2}\,b_a/8 $$ (eq:prefill-is-the-peak)

**{{eq:prefill-is-the-peak}} is quadratic**, and at $T = 131{,}072$, $h = 64$,
$b_a = 16$ it is **2,199 GB** for a single sequence. Fused attention
({{cite:dao2022flash}}) computes the same result in tiles without ever forming the
matrix, so this term becomes zero.

**The remaining prefill peak is linear:**

$$ M_{\text{peak}} = P\frac{b_w}{8} + M_{\text{kv}}(S) + c\,T\,d_{\text{model}}\frac{b_a}{8} + F $$ (eq:fused-prefill-peak)

### 5.4 Chunked prefill bounds the last term

Processing the prompt in chunks of $C$:

$$ M_{\text{peak}}(C) = P\frac{b_w}{8} + M_{\text{kv}}(S) + c\,C\,d_{\text{model}}\frac{b_a}{8} + F $$ (eq:chunked-prefill)

**{{eq:chunked-prefill}} makes the activation term a design parameter** rather
than a property of the request. Solving for the largest safe chunk:

$$ C_{\max} = \frac{M_{\text{card}} - P b_w/8 - M_{\text{kv}}(S) - F}{c\,d_{\text{model}}\,b_a/8} $$ (eq:chunk-size)

**The cache term is unchanged**, so chunking bounds what can be bounded and
nothing else.

### 5.5 Concurrent prefills are a separate capacity

The cache term is per-sequence and always resident; the prefill activation term is
per-sequence **and only during prefill**. So with $n_p$ sequences prefilling
simultaneously out of $B$ resident:

$$ M_{\text{peak}} = P\frac{b_w}{8} + M_{\text{kv}}(S, B) + n_p\,c\,C\,d_{\text{model}}\frac{b_a}{8} + F $$ (eq:admission-control-memory)

> **IMPORTANT:** {{eq:admission-control-memory}} has **two** capacity limits, not
> one: how many sequences may be resident, and how many may be in prefill at once.
> A scheduler that tracks only the first will eventually put too many in the wrong
> phase simultaneously — **and the resulting failure depends on arrival timing, so
> it does not reproduce.**

## 6. Mathematical Foundation

### 6.1 Where the crossover between weights and cache sits

From {{eq:inference-budget}}, weights bind while

$$ S B \;<\; \frac{P b_w}{2 L h_{\text{kv}} d_h b_{\text{kv}}} $$

For the 7B MHA model at $b_w = 4$, $b_{\text{kv}} = 16$: the right side is about
**1.7 × 10³** token-slots. At 4k context that is under one sequence.

For the 70B GQA-8 model: about **1.3 × 10⁴** — still under two 8k sequences.

**In both cases the weights stop binding almost immediately**, which is why the
first row of the measured table is the only one where they do.

### 6.2 The quadratic term, in perspective

At $T = 16{,}384$, {{eq:prefill-is-the-peak}} gives **34.4 GB** — comparable to
the 4-bit weights. At $T = 65{,}536$ it is **549.8 GB**, and at 131k, **2,199 GB**.

$$ \frac{M_{\text{scores}}(2T)}{M_{\text{scores}}(T)} = 4 $$

**Every doubling of context multiplies it by four**, so there is no context length
at which a naive implementation becomes affordable by waiting for hardware. **A
4× memory growth per doubling outruns any plausible hardware curve.**

### 6.3 Why chunking costs nothing in arithmetic

Prefill over $S$ tokens in chunks of $C$ performs the same $2PS$ FLOPs of linear
work, plus attention that is *cheaper*: chunk $i$ attends only to tokens
$1 \dots iC$, so

$$ \sum_{i=1}^{S/C} C \cdot iC = \frac{C^2}{2}\frac{S}{C}\left(\frac{S}{C}+1\right) \approx \frac{S^2}{2} $$

which is the same total as one pass. **Chunking changes when the work happens and
not how much there is.**

> **MATH NOTE:** {{eq:chunked-prefill}} assumes the cache for already-processed
> chunks is retained, which it must be. So chunking trades **peak activation
> memory** for **sequential dependency**, and the cost is wall-clock through
> reduced parallelism rather than through extra work. On a machine already
> saturated by one chunk's arithmetic, that cost is close to zero — which is why
> chunk sizes in the low thousands are common.

## 7. Internal Mechanics

```mermaid {#fig:budget caption="The inference memory budget, with the phase substitution that produces most surprises. Every term but one is the same during prefill and decode; the activation term takes T = 1 during decode and T = prompt length during prefill (eq:inference-budget), which is why a deployment sized on the steady state can fail on its first long request. Fused attention removes the quadratic term entirely (eq:prefill-is-the-peak) and chunked prefill bounds what remains (eq:chunked-prefill)."}
flowchart TB
    W["weights: P b_w/8<br/>fixed"] --> TOT["total"]
    KVT["KV cache: grows with S x B<br/>eq:kv-scales-with-traffic"] --> TOT
    ACT{{"activations: c B T d b_a/8"}} --> TOT
    F["framework overhead"] --> TOT
    ACT -->|"T = 1"| DEC["decode: negligible"]
    ACT -->|"T = prompt"| PRE["prefill: the peak"]
    PRE --> SQ{{"plus B h T^2 b_a/8<br/>if attention is not fused"}}
    SQ -->|"fused kernel"| ZERO["term vanishes"]
    PRE -->|"chunk to C tokens"| BOUND["bounded: eq:chunk-size"]
    TOT --> BIND{{"eq:binding-term:<br/>which is largest?"}}
```

### 7.1 The checklist, in the order that avoids wasted work

1. **Compute {{eq:inference-budget}}** at your real context and batch, and read
   {{eq:binding-term}}.
2. **Compute the prefill peak** with the *longest prompt you will accept*, not the
   average one.
3. **Confirm fused attention is actually in use.** This is a factor of thousands,
   not percent.
4. **Set the chunk size from {{eq:chunk-size}}**, not from a default.
5. **Set a concurrent-prefill limit** from {{eq:admission-control-memory}}.
6. **Then optimise the binding term**, and recompute — it may have moved.

### 7.2 The terms people forget, ranked by how often

| Forgotten term | Typical size | Why it is missed |
|---|---|---|
| prefill activations | GB, transient | absent from steady-state maths |
| attention scores | hundreds of GB | assumed fused; sometimes is not |
| allocator waste | 90%+ of the cache | invisible unless measured |
| framework overhead | ~1 GB | small but decisive at the margin |
| a second concurrent prefill | doubles the transient | only appears under load |

**The first two are the ones that cause outages**, because they are large, they
are transient, and they do not appear in any number a capacity plan usually
contains.

### 7.3 Why "will it fit" needs three answers

**Will it load?** Weights plus framework. Almost always yes on the hardware people
consider.

**Will it serve $B$ sequences at $S$ context?** {{eq:capacity-not-size}}. This is
the number a capacity plan needs and it depends on architecture more than size.

**Will it survive the worst request?** {{eq:fused-prefill-peak}} at maximum prompt
length, times the concurrent-prefill limit. **This is the one that decides whether
the service stays up.**

### 7.4 Multi-GPU changes which term is divisible, not which term is large

Everything above assumes one device. Splitting across several changes the
arithmetic in a way worth being precise about, because two of the terms divide and
two do not.

**Tensor parallelism** splits each weight matrix across devices, so the weight
term divides by the device count. It also splits the attention heads, so the KV
cache divides too. Both of the large terms shrink, which is why tensor parallelism
is the standard answer to "it does not fit".

**What does not divide is the framework overhead**, which is paid once per device
and therefore grows with the device count, and **the activation term at the
boundaries**, since each device holds the full residual stream for its slice of
the batch and the devices exchange it every layer.

**Pipeline parallelism** divides differently: each device holds whole layers, so
the weight and cache terms divide, but the devices are active in sequence rather
than together, and the pipeline needs several micro-batches in flight to stay
busy. **Those micro-batches multiply the activation term** by the pipeline depth.

So the useful summary is that multi-device execution divides
{{eq:inference-budget}}'s first two terms and multiplies parts of the third, and
{{eq:binding-term}} should be recomputed rather than assumed to scale. **A
configuration that was cache-bound on one device is usually still cache-bound on
eight**, because the term that binds divided along with everything else.

### 7.5 A worked sizing, start to finish

A team wants to serve the 70B GQA-8 model on 80 GB cards, accepting prompts up to
32k, targeting 32 concurrent users.

**Will it load?** 35 GB of weights plus 1.2 GB of framework. Yes, comfortably.

**Will it serve 32 sequences at 32k?** {{eq:capacity-not-size}} with a 16-bit
cache gives 4. With a 4-bit cache, 16. **Not 32**, so the target is not reachable
on one card and the options are two cards, a shorter context limit, or fewer
users.

Take two cards with tensor parallelism: the weight and cache terms halve, so
{{eq:capacity-not-size}} gives 32 at 4-bit. **The target is met, on paper.**

**Will it survive the worst request?** At 32 resident sequences and a 32k prompt
arriving, {{eq:fused-prefill-peak}} adds the prefill activation for that sequence
on top of a cache that is already near the limit — and
{{eq:admission-control-memory}} says a second simultaneous prefill adds it again.
**So the configuration needs a concurrent-prefill limit**, and
{{eq:chunk-size}} says what chunk size makes that limit affordable.

**The plan that comes out is therefore not "two cards".** It is two cards, 4-bit
cache, paged allocation, fused attention, a chunk size computed from the residual
headroom, and a prefill admission limit — five decisions, each following from a
term in {{eq:inference-budget}}, and none of them discoverable from the parameter
count that started the conversation.

## 8. Implementation

```python {tier=A name=inference-budget}
"""Will this model fit? Every term, and which one is binding.

"A 70B model at 4 bits is 35 GB, so it fits on a 48 GB card" is the calculation
everybody does and it is wrong more often than it is right, because the weights
are only one of five terms and usually not the one that binds.

This listing computes all of them -- weights, KV cache, activations, framework
overhead and allocator waste -- across a grid of context lengths and batch sizes,
and reports which term is largest at each point (eq:inference-budget). The useful
output is not a number but a NAME: the thing to fix.
"""
import numpy as np

MODELS = {
    "7B  MHA":   dict(P=7e9,  L=32, h=32, hkv=32, d=128, dm=4096),
    "8B  GQA-8": dict(P=8e9,  L=32, h=32, hkv=8,  d=128, dm=4096),
    "70B GQA-8": dict(P=70e9, L=80, h=64, hkv=8,  d=128, dm=8192),
}

FRAMEWORK_GB = 1.2          # CUDA context, kernels, allocator metadata
ACT_TENSORS = 6             # live intermediates per layer during decode


def weights(m, wbits):
    return m["P"] * wbits / 8.0


def kv(m, ctx, batch, kvbits, util=1.0):
    return (2 * m["L"] * m["hkv"] * m["d"] * ctx * batch * kvbits / 8.0) / util


def activations(m, batch, tokens, abits=2):
    """Live intermediates. During decode `tokens` is 1 per sequence; during
    prefill it is the whole prompt, which is what makes prefill the peak."""
    return ACT_TENSORS * batch * tokens * m["dm"] * abits


def budget(m, ctx, batch, wbits=4, kvbits=16, util=1.0, tokens=1):
    w = weights(m, wbits)
    k = kv(m, ctx, batch, kvbits, util)
    a = activations(m, batch, tokens)
    f = FRAMEWORK_GB * 1e9
    return dict(weights=w, kv=k, activations=a, framework=f)


def gb(x):
    return x / 1e9


CARDS = {"24 GB": 24e9, "48 GB": 48e9, "80 GB": 80e9, "2x80 GB": 160e9}

print("Total inference memory and the BINDING term. Weights 4-bit, cache 16-bit,")
print("paged allocator (no waste), decode only.")
print()
print(f"{'model':>11}{'context':>9}{'batch':>7}" + "".join(f"{c:>10}"
      for c in ("weights", "KV cache", "activ.", "total"))
      + f"{'binding':>12}{'fits on':>10}")
print("-" * 89)

for name, m in MODELS.items():
    for ctx, batch in ((4096, 1), (4096, 32), (32768, 1), (32768, 16),
                       (131072, 4)):
        b = budget(m, ctx, batch)
        tot = sum(b.values())
        binding = max(b, key=b.get)
        card = next((c for c, v in CARDS.items() if tot < v), "too big")
        print(f"{name:>11}{ctx:>9,}{batch:>7}"
              f"{gb(b['weights']):>10.1f}{gb(b['kv']):>10.1f}"
              f"{gb(b['activations']):>10.2f}{gb(tot):>10.1f}"
              f"{binding:>12}{card:>10}")
    print()

print("The same grid, with the levers applied: 4-bit cache and paged allocation.")
print()
print(f"{'model':>11}{'context':>9}{'batch':>7}{'before':>10}{'after':>10}"
      f"{'binding':>12}{'fits on':>10}")
print("-" * 69)
for name, m in MODELS.items():
    for ctx, batch in ((32768, 16), (131072, 4)):
        b0 = budget(m, ctx, batch)
        b1 = budget(m, ctx, batch, kvbits=4)
        t0, t1 = sum(b0.values()), sum(b1.values())
        binding = max(b1, key=b1.get)
        card = next((c for c, v in CARDS.items() if t1 < v), "too big")
        print(f"{name:>11}{ctx:>9,}{batch:>7}{gb(t0):>10.1f}{gb(t1):>10.1f}"
              f"{binding:>12}{card:>10}")

print()
print()
print("How many concurrent sequences fit? Solving the budget for batch.")
print()
print(f"{'model':>11}{'card':>9}{'context':>9}" + "".join(f"{c:>14}" for c in
      ("16-bit cache", "4-bit cache")))
print("-" * 68)


def max_batch(m, cap, ctx, wbits=4, kvbits=16):
    free = cap - weights(m, wbits) - FRAMEWORK_GB * 1e9
    if free <= 0:
        return 0
    per = kv(m, ctx, 1, kvbits) + activations(m, 1, 1)
    return int(free / per)


caps = {}
for name, m in MODELS.items():
    for cardname, cap in (("48 GB", 48e9), ("80 GB", 80e9)):
        for ctx in (8192, 32768):
            a = max_batch(m, cap, ctx, kvbits=16)
            b = max_batch(m, cap, ctx, kvbits=4)
            caps[(name, cardname, ctx)] = a
            print(f"{name:>11}{cardname:>9}{ctx:>9,}{a:>14,}{b:>14,}")

m70 = MODELS["70B GQA-8"]
m7 = MODELS["7B  MHA"]
print(f"""
Read the first table's binding column before any of the numbers, because it is
the output that changes what you do.

At 4k context and batch 1 the weights bind for every model, and that is the case
the folklore describes -- the one where "the model is 35 GB" is the whole
calculation. It is also the case that almost never occurs in production, because
batch 1 at short context is a demo rather than a deployment.

Move one row down and the answer changes. At 4k context and batch 32 the KV cache
binds for the 7B multi-head model: {gb(kv(m7, 4096, 32, 16)):.1f} GB of cache
against {gb(weights(m7, 4)):.1f} GB of weights. The model everybody calls small
has become a cache problem, and quantizing its weights further would not help at
all (eq:inference-budget).

The GQA rows are the control. The 8B model has a nearly identical parameter count
to the 7B one and eight times fewer KV heads, and it stays weight-bound in
situations where the 7B model is cache-bound. That is the architectural lever from
ch:q-activation-kv, seen here as the difference between fitting and not.

The activations column is the one to notice for what it is NOT. During decode it
is negligible -- hundredths of a gigabyte -- because each sequence contributes one
token's worth of intermediates. It is in the table so that its absence is
explicit, and because the next listing shows what happens to it during prefill,
where it is not negligible at all.

The second table applies the cache lever and reports the binding term afterwards,
which is the part worth having. Quantizing the cache to 4 bits moves every "too
big" row onto a card -- 279.6 GB to 73.4, 208.0 to 79.2 -- so the intervention
works.

And the binding column has not changed. The cache still binds in every row after
a fourfold reduction, because it was ten to eighty times the weights before it.
That is the useful negative: a 4x lever applied to a 20x problem leaves a 5x
problem, and the configuration is still cache-limited.

So the next move is not more cache quantization -- 2-bit would buy another factor
of two against a term that needs another factor of five. It is the architectural
lever, or shorter contexts, or fewer concurrent sequences. The binding column is
what says so, and it says so before any of those are tried.

That is the discipline this listing is for. Every optimisation moves the budget
and may or may not move the constraint, and the next thing to do is a function of
where the constraint ended up rather than of what helped last time.

The last table converts the budget into the number a capacity planner actually
needs -- concurrent sequences -- and it contains the most striking row in the
chapter.

On an 80 GB card at 8k context with a 16-bit cache, the 70B GQA-8 model fits
{caps[('70B GQA-8', '80 GB', 8192)]:,} concurrent sequences. The 7B multi-head
model fits {caps[('7B  MHA', '80 GB', 8192)]:,}.

Ten times the parameters, and essentially the same number of concurrent users.

That is not a rounding artefact, it is the arithmetic. Serving capacity is
governed by the cache, and the cache scales with layers times KV heads times head
dimension -- not with parameter count. The 70B model has more layers and the 7B
model has eight times more KV heads, and those nearly cancel. The 8B GQA-8 model
on the same card fits {caps[('8B  GQA-8', '80 GB', 8192)]:,}, four times either
of them, because it has the small model's layer count AND the large model's head
grouping.

So parameter count is close to useless as a predictor of serving capacity, and
the number in a model's name -- the one that determines its price, its
reputation, and the hardware people budget for -- tells you almost nothing about
how many users it can serve at once.

Which gives the sentence this chapter is for. **Model size tells you whether it
loads. Architecture and context tell you whether it serves.** The first is the
question everybody asks; the second is the one that decides the deployment, and
it is answerable in advance with the arithmetic above rather than discovered
during a load test.""")
```

The first listing computes the steady state. The second computes the peak, which
is a different number and the one that fails.

```python {tier=A name=prefill-is-the-peak}
"""The budget that fits and the run that fails: prefill is a different machine.

The previous listing computed steady-state decode memory, and a deployment sized
by that arithmetic can still die on its first long request. The reason is that a
request has two phases with completely different memory profiles, and only one of
them is in the steady-state number.

Decode processes one token per sequence, so its activations are negligible.
PREFILL processes the entire prompt at once, so its activations scale with the
prompt length -- and, without the right attention kernel, with the prompt length
SQUARED (eq:prefill-is-the-peak).

This listing computes the peak rather than the average, and prices the two
standard remedies.
"""
import numpy as np

M = dict(P=70e9, L=80, h=64, hkv=8, d=128, dm=8192)
FRAMEWORK = 1.2e9
CARD = 80e9


def gb(x):
    return x / 1e9


def weights(wbits=4):
    return M["P"] * wbits / 8.0


def kv(ctx, batch, bits=16):
    return 2 * M["L"] * M["hkv"] * M["d"] * ctx * batch * bits / 8.0


def act_linear(tokens, batch, bytes_per=2, live=6):
    """Intermediates that scale with the number of tokens being processed."""
    return live * batch * tokens * M["dm"] * bytes_per


def act_scores(tokens, batch, bytes_per=2, concurrent_layers=1):
    """The attention score matrix, tokens x tokens per head. A fused kernel
    never materialises this; a naive implementation materialises one layer's
    worth at a time."""
    return concurrent_layers * batch * M["h"] * tokens * tokens * bytes_per


print(f"70B GQA-8, weights at 4 bits ({gb(weights()):.0f} GB), one 80 GB card.")
print("Peak memory during PREFILL of a prompt, batch 1.")
print()
print(f"{'prompt':>9}{'weights':>9}{'KV':>8}{'linear':>9}{'scores':>11}"
      f"{'scores':>11}{'peak, no':>11}{'peak,':>10}")
print(f"{'tokens':>9}{'':>9}{'':>8}{'activ.':>9}{'naive':>11}{'fused':>11}"
      f"{'fusion':>11}{'fused':>10}")
print("-" * 78)

rows = {}
for S in (1024, 4096, 16384, 65536, 131072):
    w, k = weights(), kv(S, 1)
    al = act_linear(S, 1)
    an, af = act_scores(S, 1), 0.0
    pn = w + k + al + an + FRAMEWORK
    pf = w + k + al + af + FRAMEWORK
    rows[S] = (pn, pf, an, al, k)
    print(f"{S:>9,}{gb(w):>9.1f}{gb(k):>8.1f}{gb(al):>9.2f}{gb(an):>11.1f}"
          f"{gb(af):>11.1f}{gb(pn):>11.1f}{gb(pf):>10.1f}")

print()
print()
print("Chunked prefill: process the prompt in pieces of C tokens.")
print("Fused attention, batch 1, 131072-token prompt.")
print()
print(f"{'chunk C':>10}{'linear activ.':>15}{'peak':>10}{'fits 80 GB':>13}"
      f"{'prefill passes':>16}")
print("-" * 64)
S = 131072
chunks = {}
for C in (131072, 32768, 8192, 2048, 512):
    al = act_linear(C, 1)
    peak = weights() + kv(S, 1) + al + FRAMEWORK
    chunks[C] = peak
    print(f"{C:>10,}{gb(al):>13.2f} GB{gb(peak):>8.1f} GB"
          f"{('yes' if peak < CARD else 'NO'):>13}{S // C:>16}")

print()
print()
print("The steady-state trap: a batch sized on decode, then given long prompts.")
print()
print(f"{'batch':>7}{'context':>9}{'decode':>10}{'prefill 1':>12}"
      f"{'prefill all':>13}{'verdict':>24}")
print("-" * 75)

for batch, ctx in ((16, 8192), (16, 32768), (48, 8192), (8, 65536)):
    dec = weights() + kv(ctx, batch, 4) + act_linear(1, batch) + FRAMEWORK
    pre1 = weights() + kv(ctx, batch, 4) + act_linear(ctx, 1) + FRAMEWORK
    preall = weights() + kv(ctx, batch, 4) + act_linear(ctx, batch) + FRAMEWORK
    v = ("fine" if preall < CARD else
         "OOM if prompts overlap" if pre1 < CARD else "OOM on one prompt")
    print(f"{batch:>7}{ctx:>9,}{gb(dec):>8.1f} GB{gb(pre1):>10.1f} GB"
          f"{gb(preall):>11.1f} GB  {v:>22}")

r16, r131 = rows[16384], rows[131072]
print(f"""
The first table is the failure that sizing on decode cannot predict.

Look at the naive-scores column. At a 16k prompt the attention score matrix is
{gb(r16[2]):.1f} GB; at 131k it is {gb(r131[2]):.1f} GB. That is one tensor, for
one layer, for one sequence -- and it is larger than the model, larger than the
card, larger by an amount no other term in the budget approaches. It is quadratic
in the prompt length, and quadratic terms do not stay small.

The fused column is zero, because a fused attention kernel never materialises the
score matrix at all: it computes attention in tiles and keeps only the running
softmax statistics. The entire difference between the last two columns is whether
your kernel does that (eq:prefill-is-the-peak).

That is worth stating in the strongest form the numbers support. Without fused
attention, long-context inference is not slow or memory-hungry -- it is
IMPOSSIBLE, on any hardware, for prompts of the length people now routinely send.
The technique that made long context practical was not a bigger card.

With fusion, the peak at 131k tokens is {gb(r131[1]):.1f} GB -- and the card is
{gb(CARD):.0f} GB, so it STILL does not fit. Fusion removed a 2199 GB term and
left a configuration that is over budget by
{gb(r131[1] - CARD):.0f} GB. The linear activation term is now the problem, and
the second table prices the standard answer to it.

Chunked prefill processes the prompt in pieces, running the model over C tokens at
a time and appending each piece's keys and values to the cache. The linear
activation term becomes proportional to C rather than to the whole prompt, and it
is the only term that changes -- the KV cache still grows to the full prompt,
because that is the point of prefilling.

At {131072:,} tokens in one pass the linear activations are
{gb(act_linear(131072, 1)):.2f} GB; in 2048-token chunks, {gb(act_linear(2048, 1)):.2f} GB.
The cost is {131072 // 2048} sequential passes instead of one, which is slower in
wall-clock but not in total arithmetic, since the same tokens are processed either
way.

The third table is the trap this listing exists for, and it is the one that
produces production incidents rather than benchmark surprises.

A deployment sized on decode memory picks a batch size that fits comfortably.
Then a request arrives with a long prompt, and prefill for that ONE sequence
allocates activations proportional to its whole prompt -- while every other
sequence's cache is still resident and cannot be freed. The decode column and the
prefill column are different numbers for the same configuration, and only the
first appears in a steady-state calculation.

Read the verdict column. Some configurations survive a single long prefill and
fail when two arrive at once, which means the failure is LOAD-DEPENDENT and will
not reproduce under a benchmark that sends one request at a time. That is the
worst kind of capacity bug: it passes every test, fails in production, and the
trigger is a coincidence of arrival times.

Three things follow, and they are the practical content of the chapter.

Size on the PEAK, not the steady state, and compute the peak with the longest
prompt you will accept rather than the average one. If you accept 128k prompts,
128k is the number in the calculation regardless of how rare they are.

Use chunked prefill, and set the chunk size from the memory budget rather than
from a default. It converts an unbounded term into a bounded one, and the
conversion is exact.

And enforce an admission limit on concurrent prefills. The cache term is shared
across sequences and the prefill activation term is not, so the number of
sequences that may be in prefill simultaneously is a separate capacity from the
number that may be resident -- and a scheduler that does not distinguish them will
eventually put too many in the wrong phase at the same moment.""")
```

## 9. Practical Example

**The binding term changes one row into a realistic configuration.** At 4k and
batch 1, weights bind for every model. At 4k and batch 32, the 7B multi-head model
has **68.7 GB of cache against 3.5 GB of weights** — a cache problem wearing a
small model's name.

**And optimising does not always move the constraint.** Quantizing the cache to 4
bits took **279.6 → 73.4 GB** and **208.0 → 79.2 GB**, fitting every previously
oversized row — **and the cache still binds in all of them**, because it was 10–80×
the weights to begin with. **A 4× lever on a 20× problem leaves a 5× problem**,
and the next move is architectural rather than numerical.

**Parameter count does not predict capacity.** On 80 GB at 8k context: 70B GQA-8
fits **16** sequences, 7B MHA fits **17**, 8B GQA-8 fits **69**.

> **IMPORTANT:** Ten times the parameters, the same concurrency.
> {{eq:capacity-ratio}} explains it — capacity depends on
> $L \, h_{\text{kv}} \, d_h$, which tracks parameter count only loosely. **The
> number in a model's name tells you almost nothing about how many users it
> serves.**

**Prefill is a different machine.** The attention score matrix at a 131k prompt is
**2,199 GB** — one tensor, one layer, one sequence, quadratic in length
({{eq:prefill-is-the-peak}}). **Every doubling of context multiplies it by four**,
so no hardware curve rescues it. Fused attention removes the term entirely.

**And fusion is not sufficient.** The fused peak at 131k is **92.0 GB on an 80 GB
card** — over budget by 12 GB after removing a 2,199 GB term. **Chunked prefill at
2048 tokens brings it to 79.4 GB** ({{eq:chunked-prefill}}), at the cost of 64
sequential passes and no extra arithmetic.

**The trap is the load-dependent one.** A batch-16 deployment at 32k context uses
**79.2 GB** during decode and needs **82.4 GB** for one prefill — it OOMs on a
single long prompt. A batch-48 deployment at 8k uses **68.4 GB** decoding, **69.2
GB** for one prefill, and **107.1 GB** if prefills overlap.

**That last row passes every sequential benchmark and fails in production**, and
the trigger is a coincidence of arrival times.
{{eq:admission-control-memory}}: **there are two capacity limits, and schedulers
usually track one.**

## 10. Production Considerations

**Compute the budget before provisioning**, and report the binding term rather
than the total.

**Size on the peak with your maximum accepted prompt length**, however rare those
prompts are.

**Verify fused attention is in use.** It is a factor of thousands.

**Set the chunk size from {{eq:chunk-size}}**, and recompute it when the context
limit changes.

**Set a concurrent-prefill limit** separately from the resident-sequence limit.

**Measure allocator utilisation**, which is invisible otherwise
({{ch:q-activation-kv}}).

**Recompute after every optimisation** — the binding term may have moved, or, as
measured here, may not have.

**Load-test with overlapping long prompts**, not sequential ones.

## 11. Common Mistakes

**Sizing on weights alone.**

**Sizing on decode and deploying with long prompts.**

**Choosing a model by parameter count** for a capacity-constrained deployment.

**Assuming attention is fused** without checking.

**Using a default prefill chunk size** unrelated to the memory budget.

**Tracking one capacity limit** when {{eq:admission-control-memory}} has two.

**Load-testing sequentially**, which cannot produce the overlapping-prefill
failure.

**Continuing to optimise a term that stopped binding** — or, equally,
**stopping because an optimisation helped** when the same term still binds.

## 12. Failure Modes

**OOM on the first long prompt.** Cause: {{eq:prefill-is-the-peak}} or
{{eq:fused-prefill-peak}}. Fix: fused attention, then chunking.

**Intermittent OOM under load, not reproducible.** Cause: overlapping prefills
({{eq:admission-control-memory}}).

**Capacity far below the arithmetic.** Cause: allocator waste, or attention not
fused.

**Cache quantization helped and the configuration is still constrained.** Cause:
expected — the lever was smaller than the problem.

**A larger model serves more users than a smaller one.** Cause: correct;
{{eq:capacity-ratio}}, and the smaller model has more KV heads.

**Throughput collapses at long context despite fitting.** Cause: chunked prefill's
sequential passes, which is a latency cost rather than a memory one.

## 13. Alternatives

| Alternative | Trades | When |
|---|---|---|
| fused attention | none | always; it is not optional |
| chunked prefill | prefill latency | whenever prompts can be long |
| paged allocation | kernel indirection | always |
| KV quantization | some quality | when the cache binds |
| GQA/MQA model | model choice | the largest capacity lever |
| shorter context limit | product capability | when nothing else fits |
| tensor parallelism | interconnect, complexity | when one card cannot hold it |
| offload to host memory | severe latency | last resort |

**The context-limit row is the one teams resist and should not.**
{{eq:inference-budget}} is linear in $S$ for the cache and, before fusion,
quadratic for the scores. **Halving the accepted context is often the largest
single change available and the only one that costs no accuracy at all** — it
costs a product capability, which is a decision someone should make explicitly
rather than discover through outages.

## 14. Evaluation

**Report the binding term**, not just the total.

**Report the maximum accepted prompt length** with any capacity figure.

**Report the allocator and whether attention is fused.**

**Report concurrency limits for residency and prefill separately.**

**Report architecture ($L$, $h_{\text{kv}}$, $d_h$) alongside parameter count** —
without it a capacity number does not transfer between models.

## 15. Advanced Concepts

**Prefill and decode as separable services.** {{maturity:EMERGING}}
The two phases have opposite bottlenecks — prefill is compute-bound and
memory-transient, decode is bandwidth-bound and memory-resident. **Running them on
different hardware, or at least in different scheduler classes, follows directly
from {{eq:inference-budget}}'s $T$ substitution**, and disaggregated serving is
that idea taken to its conclusion.

**Capacity as an architectural property.** {{maturity:MATURE}}
{{eq:capacity-ratio}} means model selection for serving should read $L$,
$h_{\text{kv}}$ and $d_h$ before parameter count. **Almost no model card presents
them prominently**, and the number that is presented is the least relevant one.

**Chunk size as a latency/memory dial.** {{maturity:MATURE}}
{{eq:chunk-size}} gives the maximum safe chunk; smaller chunks trade
time-to-first-token for headroom. **It is one of the few genuinely continuous
dials in a serving stack**, and it is usually left at a default.

**Two-dimensional admission control.** {{maturity:EMERGING}}
{{eq:admission-control-memory}}'s second limit is rarely implemented as a first-
class concept, which is why the failure it causes is usually diagnosed as a memory
leak or a fragmentation problem.

**Memory budgets as a design input.** {{maturity:RESEARCH FRONTIER}}
Every term in {{eq:inference-budget}} except the framework is an architectural
choice made during pretraining. **Designing a model backwards from a serving
budget** — choosing $L$, $h_{\text{kv}}$ and $d_h$ to hit a concurrency target on
known hardware — is straightforward arithmetic and rarely done explicitly.

## 16. Connection to Previous Chapters

{{ch:q-activation-kv}}'s {{eq:kv-scales-with-traffic}} is the second term of
{{eq:inference-budget}}, and its allocator result is the $u$ in the denominator.
{{ch:q-gguf}}'s {{eq:decode-roofline}} is the time half of what this chapter does
for memory, and its deferred $M_{\text{kv}}$ term is now explicit.
{{ch:q-theory}} and {{ch:q-int8-int4}} decide $b_w$ and $b_{\text{kv}}$, which are
two coefficients in one equation here.
{{ch:tf-complexity}} supplies the quadratic term that {{eq:prefill-is-the-peak}}
prices.
Forward: {{ch:q-runtimes}} is largely about which stacks implement fusion,
paging and chunking well; {{ch:q-throughput-latency}} spends the capacity this
chapter computes.

## 17. Exercises

1. Compute {{eq:inference-budget}} for a 13B GQA-8 model at 16k context and batch
   24, and name the binding term.
2. From {{eq:capacity-not-size}}, find the batch that fits on a 48 GB card for that
   model at 32k context.
3. Using {{eq:capacity-ratio}}, predict the concurrency ratio between two models
   from their $L$ and $h_{\text{kv}}$ alone, then check against
   `inference-budget`.
4. Compute {{eq:prefill-is-the-peak}} for a 32k prompt at batch 4 with 32 heads.
   How many cards would it take?
5. Derive {{eq:chunk-size}} and compute the largest safe chunk for the 70B model at
   64k context on an 80 GB card.
6. In `prefill-is-the-peak`, add a second concurrent prefill and find the batch at
   which the configuration fails.
7. Show that chunked prefill performs the same total attention arithmetic as one
   pass, and say where the wall-clock cost comes from.
8. For a deployment you have: compute all three answers in
   {{sec:7-internal-mechanics}} and say which is smallest.

## 18. Interview Questions

1. A 70B model at 4 bits is 35 GB. Does it fit on a 48 GB card?
2. What is the binding term in a typical serving configuration, and why?
3. Why does a 7B model sometimes serve no more users than a 70B one?
4. Why can a deployment that passes a load test fail in production?
5. What is the prefill peak, and why is it absent from steady-state maths?
6. Why is fused attention not optional at long context?
7. What does chunked prefill bound, and what does it not?
8. Why are there two concurrency limits rather than one?
9. Your cache quantization helped and you are still constrained. What next?
10. What would you read off a model card to predict serving capacity?

## 19. Research Questions

1. {{eq:capacity-ratio}} makes capacity an architectural property. What does a
   model designed backwards from a concurrency target look like, and what does it
   give up?
2. {{eq:chunk-size}} trades time-to-first-token for headroom. What is the optimal
   chunk size as a function of the arrival process, rather than of the worst case?
3. Prefill and decode have opposite bottlenecks. How much throughput does
   disaggregating them recover, and at what interconnect cost?
4. {{eq:admission-control-memory}}'s failure is arrival-timing dependent. Can a
   scheduler bound the probability of it analytically from the arrival
   distribution?
5. Allocator utilisation, fusion, chunking and quantization all act on
   {{eq:inference-budget}}. Is there a principled order of application, or is the
   binding-term heuristic the best available?

## 20. Chapter Summary

**An inference budget has five terms and the useful output is which one binds**
({{eq:inference-budget}}, {{eq:binding-term}}). Weights bind only at short context
and batch 1; one row into a realistic configuration the 7B model has **68.7 GB of
cache against 3.5 GB of weights.**

**And an optimisation moves the budget without necessarily moving the
constraint.** Quantizing the cache to 4 bits took **279.6 → 73.4 GB** and made
every oversized row fit — **and the cache still binds in all of them.** A 4× lever
on a 20× problem leaves a 5× problem, and the binding column says so before
anything else is tried.

**Parameter count is close to useless as a capacity predictor.** On 80 GB at 8k:
**70B GQA-8 fits 16 sequences, 7B MHA fits 17, 8B GQA-8 fits 69.**
{{eq:capacity-ratio}} — capacity depends on $L h_{\text{kv}} d_h$, and the number
in the model's name does not.

**Prefill is a different machine.** The score matrix at 131k tokens is **2,199 GB**
for one sequence ({{eq:prefill-is-the-peak}}), growing **4× per context
doubling**, so no hardware curve rescues it. **Fused attention removes the term
entirely — long context is not slow without it, it is impossible.**

**And fusion is not sufficient**: the fused peak is **92.0 GB on an 80 GB card**,
and chunked prefill at 2048 tokens brings it to **79.4 GB**
({{eq:chunked-prefill}}) for 64 sequential passes and no extra arithmetic.

**The failure worth designing against is load-dependent.** A batch-48 deployment
uses **68.4 GB** decoding, **69.2 GB** for one prefill, and **107.1 GB** when two
overlap. **It passes every sequential benchmark.**
{{eq:admission-control-memory}} has **two** capacity limits — residency and
concurrent prefill — and schedulers usually track one.

Which gives the chapter's three questions, in place of the one people ask: **will
it load, will it serve $B$ sequences at $S$ context, and will it survive the worst
request?** The first is almost always yes. The second decides the capacity plan.
**The third decides whether the service stays up**, and it is the only one that
requires computing a peak rather than an average.

## 21. Further Reading

{{cite:dao2022flash}} for fused attention, which this chapter's first table prices
at a factor of thousands rather than the percentages usually quoted — at long
context it is the difference between possible and not.
{{cite:kwon2023pagedattention}} for the allocator term, and for the scheduling
machinery that {{eq:admission-control-memory}}'s second limit belongs in.
{{cite:pope2022inference}} for the analytical approach this chapter applies to
memory and {{ch:q-throughput-latency}} applies to time.
{{cite:liu2024kivi}} for the cache term's coefficient.
{{cite:dettmers2023case4bit}} for $b_w$, and note how small a part of
{{eq:inference-budget}} that famous decision actually occupies.

---
id: q-throughput-latency
number: 145
part: XV
tier: full
status: draft
requires: [q-runtimes, q-gguf, q-memory-math]
provides: [latency-throughput-pareto, cache-caps-throughput,
           speculation-spends-idleness, acceptance-rate, batch-buys-throughput,
           regime-decides-the-technique]
citations: [pope2022inference, leviathan2023speculative, cai2024medusa,
            kwon2023pagedattention, dettmers2023case4bit]
---

## 1. Learning Objectives

By the end of this chapter you will be able to draw the latency/throughput
frontier for a model and machine and say where a configuration sits on it; explain
why maximum decode throughput **does not depend on the model's parameter count**;
state which end of the frontier each technique in this part raises; explain why
speculative decoding and batching are **substitutes**; and choose a speculation
depth from a measured acceptance rate rather than by sweeping.

## 2. Why This Matters

{{ch:q-runtimes}} showed scheduling moving throughput and latency in opposite
directions. **This chapter is why they must move in opposite directions**, and
what each technique in this part does about it.

{{sec:9-practical-example}} sweeps batch size for a 70B model on one machine. At
batch 1: **10.8 ms per token, 92 tokens per second total.** At batch 512:
**215.6 ms per token, 2,375 tokens per second.**

**Nothing was optimised between those rows.** Batching converts latency into
throughput at an exchange rate the hardware sets, and every serving configuration
is a choice of where to sit on that curve. **"Make it faster" is not well-formed
until someone names the axis.**

**Then a correction to this part's own earlier arithmetic.** {{ch:q-gguf}}
computed a crossover batch at which decode becomes compute-bound — with the KV
cache set to zero. Put the cache back and **the crossover never arrives**: every
configuration measured here is memory-bound at every batch size, because the cache
read grows with batch while the weight read does not.

**Which yields the chapter's most useful number.** Throughput approaches memory
bandwidth divided by cache bytes per token per unit of context: **2,496 tokens per
second** here.

> **The model's parameter count does not appear in that expression.** Maximum
> decode throughput is a property of memory bandwidth and attention architecture,
> and the weights — which this whole part has been quantizing — drop out entirely.

**So weight quantization is a low-batch technique.** At batch 1 it is worth
**3.9×** in latency; at batch 512, **1.15×**. **It stops working exactly where
throughput optimisation starts.**

**And speculative decoding is the same story from the other side.** It spends the
idle arithmetic {{ch:q-gguf}} measured — **1.3%** utilisation at batch 1 — giving
**2.65×** at batch 1 and **1.71×** at batch 256, because batching has already
spent the same resource.

{{maturity:ESTABLISHED}} Roofline reasoning, batching.
{{maturity:MATURE}} Speculative decoding. {{maturity:EMERGING}} Regime-aware
configuration.

## 3. Prerequisites

{{ch:q-gguf}} for {{eq:decode-is-bandwidth}} and the idle-arithmetic measurement
this chapter spends; {{ch:q-memory-math}} for the cache term that turns out to
dominate; {{ch:q-runtimes}} for the scheduling that moves along this frontier.

## 4. Intuitive Explanation

### One curve, and you choose a point on it

```text
   batch   lat 16-bit   tput 16-bit   lat 4-bit   tput 4-bit   weights are
   ─────   ──────────   ───────────   ─────────   ──────────   ───────────
       1      42.2 ms      24 tok/s     10.8 ms     92 tok/s           96%
      16      48.2 ms     332 tok/s     16.9 ms    949 tok/s           62%
     128      93.1 ms    1375 tok/s     61.7 ms   2074 tok/s           17%
     512     246.9 ms    2074 tok/s    215.6 ms   2375 tok/s            5%
```

**Batching converts latency into throughput.** At batch 1 the machine reads every
weight to produce one token; at batch 512 the same read serves 512 sequences.

### The crossover that does not happen

{{ch:q-gguf}} predicted decode becoming compute-bound above a crossover batch —
**with the KV cache omitted.**

**Put it back and every row above is memory-bound.** The weights-share column says
why: **96% of the bytes read at batch 1, 5% at batch 512.** The cache read grows
linearly with batch and the weight read does not, so raising the batch does not
raise arithmetic intensity the way a weights-only model predicts. **It changes
what you are reading.**

### Which gives an asymptote with the model missing from it

$$ \text{max throughput} \;\to\; \frac{\text{bandwidth}}{\text{cache bytes per token} \times \text{context}} $$

**2,496 tokens per second** for this machine and context.

> **No parameter count appears.** A 7B model and a 700B model with the same
> attention architecture have the same maximum decode throughput at the same
> context, because at high batch neither is reading its weights very much.

### So each technique raises a different end

- **Weight quantization** — batch 1: **3.9×**. Batch 512: **1.15×**. Lifts the
  low-batch end.
- **Cache quantization and GQA** — lift the long-context end, where the cache
  dominates the read.
- **Better kernels** — lift whatever compute-bound end remains, which at these
  numbers is none of it.

**None of them lifts all of it**, and knowing which end you are on decides which
is worth anything.

### Speculative decoding spends the idleness

{{ch:q-gguf}} measured decode at batch 1 using **1.3%** of the machine's
arithmetic. {{cite:leviathan2023speculative}} turns that into speed: a cheap draft
model proposes $k$ tokens, and the expensive model verifies all of them in **one
forward pass**, because verifying $k+1$ positions reads the same weights and only
costs more arithmetic — which was free.

**And the sampling rule makes the output distribution provably identical.** That
is rare enough to state twice: **a latency improvement with no quality cost.**

```text
   batch   k=4 latency   speedup   arithmetic used, plain → speculative
   ─────   ───────────   ───────   ────────────────────────────────────
       1       4.1 ms      2.65×                     1.3%  →   6.1%
       8       5.1 ms      2.65×                     8.3%  →  38.8%
      64      16.8 ms      2.15×                    25.1%  →  95.2%
     256      66.2 ms      1.71×                    32.0%  →  96.5%
```

**Everything speculation gained came out of that gap**, and at batch 256 there is
little gap left.

> **Speculative decoding and batching are substitutes, not complements.** They
> spend the same resource, and once one has spent it the other has nothing to work
> with.

**Which resolves a recurring confusion.** A local user at batch 1 reports a large
speculative speedup. A serving team at batch 128 enables it, measures almost
nothing, and concludes the implementation is broken. **Both are right, and the
disagreement is structural.**

### And the depth follows from the acceptance rate

```text
   acceptance   k=2     k=4     k=8    k=16   best
   ──────────   ────    ────    ────   ────   ────
         0.50   1.68×   1.78×   1.70×  1.49×   k=4
         0.65   1.99×   2.33×   2.39×  2.13×   k=8
         0.80   2.34×   3.10×   3.70×  3.64×   k=8
         0.90   2.60×   3.77×   5.23×  6.20×  k=16
```

Accepting $k$ tokens in a row has probability $\alpha^{k}$, so **yield saturates
and then the wasted draft work dominates.** The depth is not a parameter to sweep
blindly; it is computable from a measured $\alpha$.

**And it says what to select a draft model on.** Agreement enters exponentially
and quality does not enter at all — the target's distribution is preserved either
way — so **a slightly worse draft model that agrees more often beats a better one
that agrees less.**

## 5. Formal Explanation

### 5.1 The frontier

For batch $B$ at $b$ bits and context $S$:

$$ t_{\text{step}}(B) = \max\!\left(\frac{P b/8 + \kappa S B}{\text{BW}},\ \frac{2PB + dP}{C}\right) $$

with $\kappa$ the cache bytes per token. Then

$$ \text{latency} = t_{\text{step}}(B), \qquad \text{throughput} = \frac{B}{t_{\text{step}}(B)} $$ (eq:latency-throughput-pareto)

**{{eq:latency-throughput-pareto}} is a curve parameterised by $B$**, and no
configuration lies above it. Batching moves along it; the techniques in this part
move the curve.

### 5.2 Why the cache prevents the compute-bound crossover

{{ch:q-gguf}}'s {{eq:memory-bound-crossover}} set $\kappa = 0$, giving
$B^{*} = (b/16)(C/\text{BW})$. With $\kappa > 0$ the memory term also grows in
$B$:

$$ \frac{d\,t_{\text{mem}}}{dB} = \frac{\kappa S}{\text{BW}}, \qquad \frac{d\,t_{\text{cmp}}}{dB} = \frac{2P}{C} $$

so the machine stays memory-bound for all $B$ whenever

$$ \frac{\kappa S}{\text{BW}} > \frac{2P}{C} \quad\Longleftrightarrow\quad \kappa S > \frac{2P\,\text{BW}}{C} $$ (eq:cache-caps-throughput)

**{{eq:cache-caps-throughput}} holds at 4k context for the measured machine**, and
holds more strongly as context grows. **The crossover in {{ch:q-gguf}} was an
artefact of dropping a term**, and it is worth correcting explicitly rather than
leaving two chapters in disagreement.

### 5.3 The asymptote

As $B \to \infty$ the weight term becomes negligible:

$$ \text{throughput} \;\to\; \frac{\text{BW}}{\kappa S} $$ (eq:batch-buys-throughput)

**{{eq:batch-buys-throughput}} contains no $P$ and no $b$.** For
$\kappa = 327{,}680$ bytes/token, $S = 4096$, $\text{BW} = 3.35$ TB/s: **2,496
tokens per second**, whatever model is running.

**Which reframes the whole part.** Weight precision determines where on the curve
you can operate at a given latency; it does not determine the ceiling. **The
ceiling is set by $\kappa$**, which is {{ch:q-activation-kv}}'s subject.

### 5.4 Speculative decoding

With acceptance probability $\alpha$ per token and depth $k$, the expected tokens
per verification round is

$$ \mathbb{E}[\text{accepted}] = \sum_{j=0}^{k} \alpha^{j} = \frac{1 - \alpha^{k+1}}{1 - \alpha} $$ (eq:acceptance-rate)

and the round costs $k$ draft steps plus one target step over $k+1$ positions:

$$ \text{speedup} = \frac{t_{\text{target}}(B, 1)\cdot \mathbb{E}[\text{accepted}]}{k\,t_{\text{draft}}(B) + t_{\text{target}}(B, k+1)} $$ (eq:speculation-spends-idleness)

**The key term is $t_{\text{target}}(B, k+1)$.** In the memory-bound regime it
equals $t_{\text{target}}(B, 1)$ — verifying $k+1$ positions is free — and the
speedup approaches $\mathbb{E}[\text{accepted}]$. In the compute-bound regime it
is $(k+1)$ times larger and the speedup approaches 1.

$$ \text{speedup} \;\to\; \begin{cases} \mathbb{E}[\text{accepted}] & \text{memory-bound} \\ 1 & \text{compute-bound} \end{cases} $$

**Batching moves you from the first case to the second**, which is why the two
techniques are substitutes.

### 5.5 The optimal depth

Differentiating {{eq:speculation-spends-idleness}} with $t_{\text{draft}} = \rho\,
t_{\text{target}}$ gives an interior optimum, because
{{eq:acceptance-rate}} saturates at $1/(1-\alpha)$ while draft cost grows linearly
in $k$:

$$ k^{*} \approx \frac{\log\big(\rho^{-1}\log \alpha^{-1}\big)}{\log \alpha^{-1}} $$ (eq:optimal-depth)

> **IMPORTANT:** {{eq:optimal-depth}} depends on $\alpha$ and on the draft/target
> cost ratio $\rho$, and on nothing else. **Both are measurable in an afternoon**,
> and the measured optima — $k=4$ at $\alpha = 0.5$ rising to $k=16$ at
> $\alpha = 0.9$ — follow from it. **Sweeping $k$ is measuring $\alpha$ the
> expensive way.**

## 6. Mathematical Foundation

### 6.1 The exchange rate, differentiated

From {{eq:latency-throughput-pareto}} in the memory-bound regime:

$$ \frac{\partial\,\text{latency}}{\partial B} = \frac{\kappa S}{\text{BW}}, \qquad \frac{\partial\,\text{throughput}}{\partial B} = \frac{P b/8}{(P b/8 + \kappa S B)^2}\cdot\text{BW} $$

**The second derivative is positive and decreasing**, so throughput has
diminishing returns in $B$ while latency grows linearly. That asymmetry is the
whole shape of the frontier: **early batching is nearly free and late batching is
nearly pure latency.**

The knee is where $\kappa S B \approx P b/8$ — where the cache read equals the
weight read. Measured, that is **between batch 16 and 32** at 4 bits, matching the
62% and 45% weights-share entries.

### 6.2 What quantization is worth, by regime

$$ \frac{t_{16}}{t_{4}} = \frac{2P + \kappa S B}{0.5P + \kappa S B} $$ (eq:quantization-by-regime)

At $B = 1$: $\approx 4$. As $B \to \infty$: $\to 1$.
**{{eq:quantization-by-regime}} is the measured 3.9× and 1.15×**, and it makes
quantization's value a function of the operating point rather than a property of
the model.

### 6.3 The acceptance rate is exponential and quality is not

From {{eq:acceptance-rate}}, the marginal value of depth $k+1$ over $k$ is
$\alpha^{k+1}$ — **exponentially decaying** — while the draft cost of that depth
is constant. So the optimum is where $\alpha^{k+1}$ falls below the draft/target
cost ratio.

**Meanwhile the target model's output distribution is unchanged at every $k$.**

$$ \text{quality}(k) = \text{quality}(\text{target}), \quad \forall k, \forall \text{draft model} $$

> **MATH NOTE:** That invariance is the property worth remembering, and it is
> unusual. Every other technique in {{part:15}} trades quality for speed and asks
> you to price the trade. **Speculative decoding does not**, so a draft model can
> be chosen purely on agreement and cost — its own quality is irrelevant, which is
> counterintuitive enough that people select draft models on the wrong criterion.

## 7. Internal Mechanics

```mermaid {#fig:frontier caption="One frontier, several ends, and a different technique raising each. Batching moves along the curve, converting latency into throughput at a rate the hardware fixes (eq:latency-throughput-pareto). Weight quantization lifts the low-batch end and stops mattering at high batch (eq:quantization-by-regime). Cache quantization and GQA raise the ceiling itself, which eq:batch-buys-throughput shows does not depend on the model at all. Speculation spends the idle arithmetic — the same resource batching spends, which is why they are substitutes."}
flowchart LR
    B["batch size"] -->|"moves ALONG"| CURVE["latency / throughput frontier<br/>eq:latency-throughput-pareto"]
    WQ["weight quantization"] -->|"lifts the LOW-BATCH end<br/>eq:quantization-by-regime"| CURVE
    KVQ["cache quantization, GQA"] -->|"raises the CEILING<br/>eq:batch-buys-throughput"| CURVE
    SPEC["speculative decoding"] -->|"spends idle arithmetic"| IDLE[("arithmetic left idle<br/>by memory-bound decode")]
    B -->|"also spends it"| IDLE
    IDLE -->|"substitutes, not complements"| CURVE
    CURVE --> PICK{{"pick a point:<br/>a latency SLO"}}
```

### 7.1 Which technique for which regime

| Regime | Binding | What helps | What does not |
|---|---|---|---|
| batch 1, short context | weight read | weight quantization, speculation | batching, cache work |
| batch 1, long context | cache read | GQA, cache quantization | weight quantization |
| high batch, short context | cache read | GQA, cache quantization, more memory | weight quantization, speculation |
| high batch, long context | cache read, hard | GQA, shorter context, more machines | almost everything else |
| latency-SLO-bound | the SLO | quantization to buy back batch, speculation | raw batching |

**Every row of that table is a different answer to "how do I make it faster",**
and the row is determined by two numbers you can measure in minutes.

### 7.2 Reading a latency SLO as a throughput budget

A per-token latency target $L$ fixes the largest permissible batch through
{{eq:latency-throughput-pareto}}, and therefore the throughput and the cost per
token:

$$ B_{\max}(L) = \frac{L \cdot \text{BW} - P b/8}{\kappa S}, \qquad \text{throughput} = \frac{B_{\max}}{L} $$ (eq:slo-to-throughput)

**{{eq:slo-to-throughput}} is the arithmetic behind every pricing decision in
inference serving**, and it makes the value of quantization concrete: it does not
make anything faster at the SLO, it **buys back batch size**, which is
convertible directly into cost per token.

### 7.3 Time to first token is the other latency

This chapter has measured *inter-token* latency. **Time to first token is a
different quantity with a different bottleneck** — prefill is compute-bound
({{ch:q-gguf}}), so it responds to compute and to prompt length and barely at all
to weight precision.

**A product has both requirements and they are optimised by different things.**
{{ch:q-runtimes}}'s chunked prefill trades between them explicitly, and a serving
configuration that reports only one is reporting half of what a user experiences.

### 7.4 The economics, which is what the frontier is usually for

Every point on {{eq:latency-throughput-pareto}} has a cost per token, and it is the
quantity that decides most real configuration arguments.

With a machine costing $c$ per hour and throughput $T$ tokens per second:

$$ \text{cost per million tokens} = \frac{c}{3600\,T}\times 10^{6} $$

**Moving up the batch axis divides the cost linearly until the frontier flattens**,
after which additional batch buys latency degradation and nothing else. So the
economically-correct operating point is at the knee — where
{{sec:6-mathematical-foundation}} located it, at $\kappa S B pprox P b/8$ —
unless a latency SLO forces you below it.

**That is the sentence that connects this part to a budget.** The techniques here
do not make a service cheaper by making the model faster. They make it cheaper by
**moving the knee to a larger batch**, or by **raising the ceiling the knee sits
under**, and those are two different interventions with different owners:
quantization and kernel work for the first, attention architecture and context
policy for the second.

**And it explains a pattern worth recognising.** A team that cannot meet its
latency SLO buys more machines and runs each at low batch, paying several times
the cost per token that the same hardware would deliver at the knee. The
intervention that fixes it is not more hardware — it is whichever of
{{eq:quantization-by-regime}} or {{eq:batch-buys-throughput}} moves the frontier
far enough that the SLO and the knee are compatible.

### 7.5 Why this part's chapters kept producing the same shape of result

Six chapters in, a pattern has repeated often enough to be worth naming.

{{ch:q-theory}} found group size mattering more than bit-width.
{{ch:q-gguf}} found unpacking cost mattering more than bits saved, and only on
some machines. {{ch:q-activation-kv}} found the allocator mattering more than the
cache's precision. {{ch:q-memory-math}} found architecture mattering more than
parameter count. This chapter finds the operating regime mattering more than any
of the techniques.

**In every case the quantity people quote turned out to be the less important half
of a specification**, and in every case the more important half was cheap to
measure and rarely reported.

That is not a coincidence about quantization. It is what happens when a field
develops a headline number early — bits per weight, parameter count, tokens per
second — and then optimises everything around it. **The headline is usually a
correct measurement of something, and usually not of the thing that binds.**

The defence is the same in each chapter and worth stating once: **compute the
budget, find the binding term, and report both.** The binding term is what changes
what you do, and it is almost never the number in the model's name.

## 8. Implementation

```python {tier=A name=latency-throughput-pareto}
"""Latency and throughput are not two names for speed. They are in tension.

cite:pope2022inference established that low-latency generation and high-throughput
batch processing are distinct optimisation regimes with different answers. That is
easy to agree with and hard to act on until the trade is drawn.

This listing draws it. For one model on one machine, it sweeps batch size and
computes both quantities from the roofline, producing the Pareto frontier along
which every serving configuration sits (eq:latency-throughput-pareto). Then it
adds quantization, which does not move the frontier uniformly -- it moves one end
of it and leaves the other where it was.
"""
import numpy as np

P = 70e9                 # parameters
BW = 3.35e12             # bytes per second
C = 990e12               # FLOPs per second
DEQ_OPS = 4.0            # extra ops per weight to unpack a quantized value
KV_BYTES_PER_TOK = 2 * 80 * 8 * 128 * 2      # 80 layers, GQA-8, 16-bit


def step_s(batch, bits, ctx=4096):
    """One decode step: the slower of reading everything and computing."""
    read = P * bits / 8.0 + KV_BYTES_PER_TOK * ctx * batch
    t_mem = read / BW
    flops = 2.0 * P * batch + (DEQ_OPS * P if bits < 16 else 0.0)
    t_cmp = flops / C
    return max(t_mem, t_cmp), t_mem, t_cmp


BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)

print("70B GQA-8, 4k context. Per-token latency is what one user waits between")
print("tokens; throughput is what the machine produces in total.")
print()
print(f"{'batch':>7}" + "".join(f"{h:>13}" for h in
      ("lat 16-bit", "tput 16-bit", "lat 4-bit", "tput 4-bit"))
      + f"{'weights are':>13}{'bound by':>10}")
print(f"{'':>7}{'ms/token':>13}{'tok/s':>13}{'ms/token':>13}{'tok/s':>13}"
      f"{'of the read':>13}{'':>10}")
print("-" * 87)

rows = {}
for b in BATCHES:
    t16, m16, c16 = step_s(b, 16)
    t4, m4, c4 = step_s(b, 4)
    wshare = (P * 0.5) / (P * 0.5 + KV_BYTES_PER_TOK * 4096 * b)
    rows[b] = (t16 * 1000, b / t16, t4 * 1000, b / t4, wshare)
    print(f"{b:>7}{t16*1000:>13.1f}{b/t16:>13.0f}{t4*1000:>13.1f}"
          f"{b/t4:>13.0f}{wshare:>13.0%}"
          f"{('memory' if m4 > c4 else 'compute'):>10}")

ASYMPTOTE = BW / (KV_BYTES_PER_TOK * 4096)
print()
print(f"Asymptotic throughput as batch grows: {ASYMPTOTE:,.0f} tok/s.")
print("Note what is absent from that number: the model.")

print()
print()
print("What does a latency budget cost in throughput? Largest batch that meets")
print("a per-token latency target, and the throughput it yields.")
print()
print(f"{'target':>10}" + "".join(f"{h:>14}" for h in
      ("batch 16-bit", "tput 16-bit", "batch 4-bit", "tput 4-bit"))
      + f"{'4-bit gain':>13}")
print("-" * 79)

targets = {}
for tgt_ms in (25, 40, 60, 100, 200):
    best = {}
    for bits in (16, 4):
        ok = [b for b in BATCHES if step_s(b, bits)[0] * 1000 <= tgt_ms]
        b = max(ok) if ok else 0
        best[bits] = (b, b / step_s(b, bits)[0] if b else 0.0)
    targets[tgt_ms] = best
    g = best[4][1] / best[16][1] if best[16][1] else float("inf")
    print(f"{tgt_ms:>8} ms{best[16][0]:>14}{best[16][1]:>14.0f}"
          f"{best[4][0]:>14}{best[4][1]:>14.0f}"
          + (f"{g:>12.2f}x" if np.isfinite(g) else f"{'--':>13}"))

print()
print()
print("Context length moves the whole picture, because the cache joins the read.")
print()
print(f"{'context':>9}{'batch':>7}{'lat 4-bit':>12}{'tput 4-bit':>13}"
      f"{'KV share of':>14}")
print(f"{'':>9}{'':>7}{'ms/token':>12}{'tok/s':>13}{'bytes read':>14}")
print("-" * 55)
ctx_rows = {}
for ctx in (4096, 32768, 131072):
    for b in (1, 32):
        t, m, c = step_s(b, 4, ctx)
        kvshare = (KV_BYTES_PER_TOK * ctx * b) / (P * 0.5 + KV_BYTES_PER_TOK
                                                  * ctx * b)
        ctx_rows[(ctx, b)] = (t * 1000, b / t, kvshare)
        print(f"{ctx:>9,}{b:>7}{t*1000:>12.1f}{b/t:>13.0f}{kvshare:>13.0%}")

b1, b32, b512 = rows[1], rows[32], rows[512]
print(f"""
The first table is the tension, and it is worth being precise about why it exists
rather than treating it as a slogan.

At batch 1 the machine reads every weight to produce one token, so per-token
latency is as low as it will ever be -- {b1[2]:.1f} ms at 4 bits -- and throughput
is as low as it will ever be too: {b1[3]:.0f} tokens per second from hardware
capable of far more. At batch 512 the same weight read serves 512 sequences, so
throughput reaches {b512[3]:.0f} and each user now waits {b512[2]:.1f} ms between
tokens rather than {b1[2]:.1f}.

Nothing was optimised or mis-optimised between those rows. **Batching converts
latency into throughput at an exchange rate set by the hardware**
(eq:latency-throughput-pareto), and every serving configuration is a choice of
where on that curve to sit. "Make the system faster" is not a well-formed request
until someone names the axis.

Now the column that corrects something ch:q-gguf simplified.

That chapter computed a crossover batch at which decode stops being memory-bound
and becomes compute-bound, and it set the KV cache term to zero to do so. Put the
cache back and the crossover never arrives: every row here is memory-bound, at
every batch size tested.

The reason is in the weights-share column. At batch 1 the weights are
{b1[4]:.0%} of the bytes read per step. At batch 32, {b32[4]:.0%}. At batch 512,
{b512[4]:.0%}. **The cache read grows linearly with batch and the weight read does
not**, so raising the batch does not raise arithmetic intensity the way the
weights-only model predicted -- it just changes what you are reading
(eq:cache-caps-throughput).

Which gives the asymptote printed above the analysis, and it is the most useful
single number in the chapter. As batch grows, throughput approaches memory
bandwidth divided by the cache bytes per token per unit of context:
{ASYMPTOTE:,.0f} tokens per second here.

**The model's parameter count does not appear in that expression.** Maximum decode
throughput at a given context length is a property of the memory bandwidth and the
attention architecture, and the weights -- the thing this entire part has been
quantizing -- have dropped out of it entirely.

That reframes what weight quantization is for, and the 4-bit columns show it. At
batch 1, 4 bits gives {b1[2]:.1f} ms against 16 bits' {b1[0]:.1f} --
{b1[0]/b1[2]:.1f}x. At batch 512, {b512[2]:.1f} against {b512[0]:.1f}: a factor
of {b512[0]/b512[2]:.2f}. **Weight quantization is a low-batch technique, and it
stops working exactly where throughput optimisation starts.**

That is not a caveat. It is the shape of the practice, and it explains why
local-inference practitioners and serving engineers reach opposite conclusions
about the same technique while both measuring correctly.

The second table converts the trade into the form a product decision takes: a
latency budget, and what it costs.

Committing to a 40 ms per-token experience -- comfortable streaming speed --
allows a batch of {targets[40][16][0]} at 16 bits and {targets[40][4][0]} at 4
bits, for {targets[40][16][1]:.0f} and {targets[40][4][1]:.0f} tokens per second.
At 16 bits that target is unreachable at any batch, so the entry is zero: the
configuration cannot meet the SLO at all.

**Which is the most useful way to state quantization's value: it buys back batch
size at a fixed latency.** That is a far more actionable claim than "it makes the
model faster", and it is directly convertible into cost per token.

Relaxing the target to 200 ms allows batch {targets[200][4][0]} and
{targets[200][4][1]:.0f} tokens per second -- so the cost of a tight latency SLO
is a throughput multiple, computable in advance rather than discovered in
production.

The third table adds context length, which moves everything. At {131072:,} tokens
and batch 32 the cache is {ctx_rows[(131072, 32)][2]:.0%} of the bytes read, and
latency is {ctx_rows[(131072, 32)][0]:.1f} ms against
{ctx_rows[(4096, 32)][0]:.1f} ms at 4k.

**No weight format changes that.** The lever that does is ch:q-activation-kv's,
and this is the arithmetic showing that the two chapters are about the same
bottleneck at different context lengths.

So the shape to carry away is one curve with several ends, each raised by a
different technique. **Weight quantization lifts the low-batch end. Cache
quantization and grouped-query attention lift the long-context end. Better
kernels lift whatever compute-bound end remains.** None lifts all of it, and
knowing which end you are on decides which of them is worth anything to you.""")
```

The first listing draws the frontier. The second is about the one technique that
moves along it without a quality cost.

```python {tier=A name=speculation-spends-idleness}
"""Speculative decoding spends the idleness, so it competes with batching for it.

ch:q-gguf measured decode at batch 1 running at about one per cent of the
hardware's arithmetic balance point: the multiply-add units are idle almost all
the time, waiting for weights to arrive.

cite:leviathan2023speculative turns that idleness into speed. A cheap draft model
proposes k tokens; the expensive model verifies all of them in ONE forward pass,
because verifying k+1 positions reads the same weights as verifying one and only
costs more arithmetic -- which was free. The sampling rule makes the output
distribution provably identical to the target model's, so it is a latency
improvement with no quality cost, which is rare enough to be worth checking.

The part that is not usually stated is what happens as batch size rises, because
batching spends the same idleness (eq:speculation-spends-idleness).
"""
import numpy as np

P_TARGET = 70e9
P_DRAFT = 1.5e9
BW = 3.35e12
C = 990e12
KV_PER_TOK = 2 * 80 * 8 * 128 * 2
CTX = 4096
BITS = 4


def target_step(batch, positions=1):
    """One forward pass of the big model over `positions` token slots per
    sequence. The weight read is unchanged; only the arithmetic scales."""
    read = P_TARGET * BITS / 8.0 + KV_PER_TOK * CTX * batch
    return max(read / BW, 2.0 * P_TARGET * batch * positions / C)


def draft_step(batch):
    read = P_DRAFT * BITS / 8.0 + KV_PER_TOK * CTX * batch * (P_DRAFT / P_TARGET)
    return max(read / BW, 2.0 * P_DRAFT * batch / C)


def accepted(alpha, k):
    """Expected tokens accepted per verification round, including the bonus
    token the target model always contributes."""
    if alpha >= 1.0:
        return k + 1.0
    return (1.0 - alpha ** (k + 1)) / (1.0 - alpha)


def speculative(batch, k, alpha):
    t = k * draft_step(batch) + target_step(batch, k + 1)
    toks = accepted(alpha, k)
    return t / toks, batch * toks / t          # latency per token, throughput


def plain(batch):
    t = target_step(batch)
    return t, batch / t


ALPHA = 0.72
BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256)

print(f"70B target, 1.5B draft, acceptance rate {ALPHA:.0%}, {CTX:,} context.")
print("Speedup is speculative against plain decoding, at the same batch size.")
print()
print(f"{'batch':>7}" + "".join(f"{'k=' + str(k):>22}" for k in (2, 4, 8))
      + f"{'plain':>12}")
print(f"{'':>7}" + "".join(f"{'lat ms':>11}{'speedup':>11}" for _ in (2, 4, 8))
      + f"{'lat ms':>12}")
print("-" * 85)

rows = {}
for b in BATCHES:
    pl, ptp = plain(b)
    cells = []
    for k in (2, 4, 8):
        lat, tp = speculative(b, k, ALPHA)
        cells.append((lat * 1000, pl / lat))
        rows[(b, k)] = (lat * 1000, pl / lat, tp / ptp)
    print(f"{b:>7}"
          + "".join(f"{c[0]:>11.1f}{c[1]:>10.2f}x" for c in cells)
          + f"{pl*1000:>12.1f}")

print()
print()
print("Where does the idleness go? Arithmetic actually performed as a share of")
print("what the hardware could do in the same wall-clock time.")
print()
print(f"{'batch':>7}{'plain':>12}{'speculative k=4':>18}{'gap closed':>14}")
print("-" * 51)
util = {}
for b in (1, 8, 64, 256):
    tp, _ = plain(b)
    up = (2.0 * P_TARGET * b / C) / tp
    k4 = 4
    t = k4 * draft_step(b) + target_step(b, k4 + 1)
    us = (2.0 * P_TARGET * b * (k4 + 1) / C + k4 * 2.0 * P_DRAFT * b / C) / t
    util[b] = (up, us)
    print(f"{b:>7}{up:>11.1%}{us:>18.1%}{(us - up)/(1 - up):>13.0%}")

print()
print()
print("Acceptance rate is the other variable, and it is a property of the pair.")
print()
print(f"{'alpha':>8}" + "".join(f"{'k=' + str(k):>12}" for k in (2, 4, 8, 16))
      + f"{'best k':>10}")
print("-" * 60)
acc_rows = {}
for a in (0.5, 0.65, 0.8, 0.9):
    lats = []
    for k in (2, 4, 8, 16):
        lat, _ = speculative(1, k, a)
        lats.append(plain(1)[0] / lat)
    best = (2, 4, 8, 16)[int(np.argmax(lats))]
    acc_rows[a] = (lats, best)
    print(f"{a:>8.2f}" + "".join(f"{v:>11.2f}x" for v in lats)
          + f"{'k=' + str(best):>10}")

s1, s64, s256 = rows[(1, 4)], rows[(64, 4)], rows[(256, 4)]
print(f"""
Read the k=4 columns down the page and the effect shrinks as the batch grows.

At batch 1, speculating four tokens ahead gives {s1[1]:.2f}x lower per-token
latency. At batch 64, {s64[1]:.2f}x. At batch 256, {s256[1]:.2f}x.

The mechanism is in the utilisation table. At batch 1 plain decoding uses
{util[1][0]:.1%} of the machine's arithmetic capacity -- the number ch:q-gguf
measured, from the other direction. Speculation raises it to {util[1][1]:.1%},
and everything it gained came out of that gap. At batch 256 plain decoding is
already at {util[256][0]:.1%}, so there is far less gap left, and speculation's
extra arithmetic starts to cost real time rather than filling a hole
(eq:speculation-spends-idleness).

That is the finding, and it is not how the two techniques are usually presented.
**Speculative decoding and batching are substitutes, not complements.** They spend
the same resource -- the arithmetic that decode leaves idle -- and once one of
them has spent it, the other has nothing to work with.

Which resolves a common confusion. A local user at batch 1 measures a large
speculative speedup and reports it. A serving team at batch 128 enables the same
feature, measures almost nothing, and concludes the implementation is broken. Both
measurements are right, and the disagreement is structural rather than a
configuration error.

It also says where speculation belongs. It is a LATENCY technique for the
low-batch regime -- interactive single-user inference, or a serving tier with a
tight per-token SLO that forces small batches. It is not a throughput technique,
and at high batch it is not a technique at all.

The third table adds the variable that decides whether any of this works. The
acceptance rate is a property of the DRAFT AND TARGET PAIR, not of the algorithm,
and it enters as a power: accepting k tokens in a row has probability alpha^k, so
the expected yield saturates quickly.

At alpha={0.5:.2f}, speculating far ahead is pointless -- the best depth is
k={acc_rows[0.5][1]}, and going deeper wastes draft work on tokens that will be
rejected. At alpha={0.9:.2f} the best depth is k={acc_rows[0.9][1]} and the
speedup is {max(acc_rows[0.9][0]):.2f}x.

So the depth is not a tuning parameter to be swept blindly; it follows from the
measured acceptance rate, and the acceptance rate is what a draft model should be
selected on. A draft model that is slightly worse but agrees more often beats a
better one that agrees less, because agreement enters exponentially and quality
does not enter at all -- the target model's distribution is preserved exactly
either way.

Which is the property that makes this technique unusual and worth the chapter's
attention. Almost every other option in {{part:15}} trades quality for speed and
asks you to price the trade. Speculative decoding does not: the output
distribution is provably identical, so the only questions are whether you are in a
regime where it helps and whether you can find a draft model that agrees often
enough.""")
```

## 9. Practical Example

**One curve, and batching moves you along it.** Batch 1: **10.8 ms per token, 92
tok/s.** Batch 512: **215.6 ms per token, 2,375 tok/s.** Nothing was optimised
between them — {{eq:latency-throughput-pareto}} is an exchange rate the hardware
sets.

**And this part's earlier crossover was an artefact.** {{ch:q-gguf}} set the cache
term to zero; restored, **every configuration here is memory-bound at every
batch**, because the weights fall from **96% to 5%** of the bytes read as batch
grows ({{eq:cache-caps-throughput}}).

**Which gives an asymptote with no model in it**: **2,496 tok/s**
({{eq:batch-buys-throughput}}).

> **IMPORTANT:** Maximum decode throughput is bandwidth divided by cache bytes per
> token per unit of context. **The parameter count — the thing this part has spent
> six chapters quantizing — does not appear.** Weight precision determines where
> you can operate at a given latency, not the ceiling.

**So weight quantization is regime-dependent**: **3.9×** at batch 1, **1.15×** at
batch 512 ({{eq:quantization-by-regime}}). **It stops working exactly where
throughput optimisation starts** — which is why local practitioners and serving
engineers reach opposite conclusions while both measuring correctly.

**And long context moves the whole picture**: at 131k tokens and batch 32 the
cache is the overwhelming majority of the read, and no weight format changes it.
**That is {{ch:q-activation-kv}}'s lever, not this one's.**

**Speculative decoding spends the idle arithmetic.** Plain decoding uses **1.3%**
of the machine at batch 1; speculation at $k=4$ raises it to **6.1%** and gives
**2.65×**. At batch 256, plain is already at **32.0%** and speculation gives
**1.71×**.

**Batching and speculation are substitutes** — they spend the same resource
({{eq:speculation-spends-idleness}}). **Which is why one team measures a large
speedup and another measures almost none, and both are right.**

**And the depth follows from the acceptance rate**, not from a sweep: best $k$ is
**4** at $\alpha = 0.5$, **8** at 0.65 and 0.8, **16** at 0.9, with speedups from
**1.78×** to **6.20×** ({{eq:acceptance-rate}}, {{eq:optimal-depth}}).

**A draft model should be selected on agreement, not quality.** Agreement enters
exponentially; the draft model's own quality does not enter at all, because the
target's output distribution is preserved exactly.

## 10. Production Considerations

**Name the axis before optimising.** Latency and throughput are in tension and
"faster" is ambiguous.

**Compute {{eq:batch-buys-throughput}}** before planning capacity — it is the
ceiling, and it does not move with model choice.

**Determine your regime from two numbers**: batch size and context length. The
table in {{sec:7-internal-mechanics}} then says which techniques are worth
anything.

**State latency SLOs per token and convert with
{{eq:slo-to-throughput}}** to get cost per token.

**Measure the acceptance rate** before choosing a speculation depth, and select
draft models on it.

**Report TTFT and inter-token latency separately** — different bottlenecks,
different remedies.

**Do not enable speculation at high batch** and expect the low-batch result.

## 11. Common Mistakes

**Reporting "tokens per second" without the batch size**, which conflates the two
axes.

**Generalising a batch-1 measurement to a serving deployment**, or the reverse.

**Expecting weight quantization to help at high batch.**

**Assuming decode becomes compute-bound** — with the cache included, usually it
does not.

**Choosing a draft model by its quality** rather than by its agreement rate.

**Sweeping speculation depth** instead of measuring $\alpha$ and computing it.

**Optimising the model when {{eq:batch-buys-throughput}} says the ceiling is
architectural.**

## 12. Failure Modes

**Speculation gives no speedup in production.** Cause: the deployment is at high
batch ({{eq:speculation-spends-idleness}}). Expected.

**Quantization gives no speedup on the server.** Cause: high batch
({{eq:quantization-by-regime}}). Also expected.

**Throughput plateaus below the arithmetic.** Cause:
{{eq:batch-buys-throughput}} — you have reached the ceiling and the model is not
what is limiting.

**A latency SLO is unmeetable at any batch.** Cause:
{{eq:slo-to-throughput}} with a negative numerator; the weight read alone exceeds
the budget, and only quantization or a smaller model helps.

**Deeper speculation made things slower.** Cause: past
{{eq:optimal-depth}}; draft work on tokens that get rejected.

**Long-context throughput collapses.** Cause: $\kappa S$ dominating
{{eq:cache-caps-throughput}}; the lever is architectural.

## 13. Alternatives

| Alternative | Raises | Where it stops |
|---|---|---|
| larger batch | throughput | latency SLO, memory |
| weight quantization | low-batch latency | the crossover |
| cache quantization / GQA | the ceiling | quality, model choice |
| speculative decoding | low-batch latency | high batch |
| Medusa-style heads ({{cite:cai2024medusa}}) | same, no draft model | high batch |
| shorter context | the ceiling | product capability |
| more machines | everything | money |

**The Medusa row is worth its own note.** {{cite:cai2024medusa}} removes the
operational objection to speculation — finding, training and serving a second
model — by putting extra prediction heads on the target itself. **It does not
change {{eq:speculation-spends-idleness}}**, so it inherits the same regime
dependence: it is a low-batch technique with a better deployment story, not a
different technique.

## 14. Evaluation

**Report batch size and context with every latency and throughput number.**

**Report TTFT and inter-token latency separately.**

**Report the acceptance rate** with any speculative-decoding result — without it
the speedup is unreproducible.

**Report the ceiling from {{eq:batch-buys-throughput}}** alongside measured
throughput, so a reader can see how much headroom existed.

**Report which regime the measurement was taken in.** Most disagreements in this
area are two correct measurements in different regimes.

## 15. Advanced Concepts

**The ceiling is architectural, and it is the interesting number.**
{{maturity:EMERGING}} {{eq:batch-buys-throughput}} has no $P$ in it, so decode
throughput at scale is a statement about attention design and memory bandwidth.
**Model selection for high-throughput serving should read $\kappa$ before
parameter count**, and essentially no model card presents it.

**Speculation and batching as one budget.** {{maturity:EMERGING}}
Both spend idle arithmetic, so the right question is how to allocate it — and at
intermediate batch there is an optimum that uses some of each.
**Nothing currently schedules that jointly.**

**Draft-model selection is an agreement problem.** {{maturity:MATURE}}
{{eq:acceptance-rate}}'s exponential means a draft model should be chosen to
*agree*, which is not the same as being good. Distilling a draft from the target
({{ch:ft-merging}}'s technique) optimises exactly the right thing and is
under-used relative to picking a small model off the shelf.

**Prefill and decode want different machines.** {{maturity:EMERGING}}
Prefill is compute-bound and decode is memory-bound, so a fleet optimised for one
is wrong for the other. **Disaggregation follows from this chapter's arithmetic
rather than from any new idea.**

**Bandwidth is the axis that is not improving.** {{maturity:RESEARCH FRONTIER}}
$C/\text{BW}$ rises with every hardware generation, so
{{eq:cache-caps-throughput}} holds more strongly over time and decode becomes
*more* memory-bound. **Everything in this part becomes more valuable for
structural reasons, independently of algorithmic progress** — and the ceiling in
{{eq:batch-buys-throughput}} improves only as fast as memory bandwidth does.

## 16. Connection to Previous Chapters

{{ch:q-gguf}}'s {{eq:decode-is-bandwidth}} is this chapter's batch-1 corner, and
its {{eq:memory-bound-crossover}} is **corrected here** by restoring the cache
term — the crossover was an artefact of the simplification, and
{{eq:cache-caps-throughput}} says so explicitly.
Its measurement of decode's idle arithmetic is what
{{eq:speculation-spends-idleness}} spends.
{{ch:q-activation-kv}}'s $\kappa$ is the coefficient in
{{eq:batch-buys-throughput}}, which makes that chapter's architectural lever the
one that raises the ceiling.
{{ch:q-memory-math}} determines the largest batch available, and
{{ch:q-runtimes}} is how a scheduler moves along the frontier this chapter draws.
Forward: {{part:23}} owns multi-machine serving, autoscaling and the economics
{{eq:slo-to-throughput}} feeds.

## 17. Exercises

1. Derive {{eq:batch-buys-throughput}} and compute it for a GQA-8 model at 32k
   context on 2 TB/s of bandwidth.
2. From {{eq:cache-caps-throughput}}, find the context length at which a 7B MHA
   model stays memory-bound at every batch.
3. Compute {{eq:quantization-by-regime}} at $B = 1, 8, 64$ for a 13B model, and
   say at which batch 4-bit stops being worth the dequantization cost.
4. Use {{eq:slo-to-throughput}} to find the maximum batch meeting a 30 ms
   per-token SLO for the listing's model, at 16 and 4 bits.
5. Derive {{eq:acceptance-rate}} and compute the expected accepted tokens at
   $\alpha = 0.75$, $k = 6$.
6. In `speculation-spends-idleness`, set the draft model to 7B rather than 1.5B.
   At what acceptance rate does it still pay?
7. In `latency-throughput-pareto`, set the KV term to zero and confirm that
   {{ch:q-gguf}}'s crossover reappears. What does that tell you about model
   simplifications?
8. For your own deployment: measure batch and context, place yourself in
   {{sec:7-internal-mechanics}}'s table, and name the technique that is worth
   trying next.

## 18. Interview Questions

1. Why are latency and throughput in tension?
2. What sets the maximum decode throughput of a machine?
3. Why does the model's parameter count not appear in that answer?
4. Does decode become compute-bound at large batch? Answer carefully.
5. Why does weight quantization help at batch 1 and not at batch 512?
6. What resource does speculative decoding spend?
7. Why do speculation and batching compete?
8. How would you choose a speculation depth?
9. What should you select a draft model on, and why not quality?
10. Two teams report different results for the same optimisation. How do you
    reconcile them?

## 19. Research Questions

1. {{eq:batch-buys-throughput}} makes the ceiling architectural. What attention
   design maximises it at fixed quality, and how far is current practice from it?
2. Speculation and batching share one budget. What does jointly optimising the
   allocation buy at intermediate batch, and can it be scheduled online?
3. {{eq:optimal-depth}} needs $\alpha$ and $\rho$, both measurable at runtime.
   Would an adaptive depth beat a fixed one, and by how much?
4. Distilling a draft model to maximise agreement rather than quality is the
   objective {{eq:acceptance-rate}} implies. How much acceptance can that buy over
   an off-the-shelf small model?
5. $C/\text{BW}$ rises each hardware generation. Extrapolating
   {{eq:cache-caps-throughput}}, what does inference serving look like when it is
   ten times its current value?

## 20. Chapter Summary

**Latency and throughput are one frontier and batching moves you along it**:
**10.8 ms / 92 tok/s** at batch 1 against **215.6 ms / 2,375 tok/s** at batch 512,
with nothing optimised in between ({{eq:latency-throughput-pareto}}). **"Make it
faster" is not well-formed until someone names the axis.**

**And this part's earlier crossover was an artefact of a dropped term.**
{{ch:q-gguf}} set the cache to zero; restored, every configuration is
memory-bound at every batch, because the weights fall from **96% to 5%** of the
bytes read ({{eq:cache-caps-throughput}}).

**Which gives an asymptote with no model in it — 2,496 tokens per second**
({{eq:batch-buys-throughput}}): bandwidth over cache-bytes-per-token-per-context.
**The parameter count this part spent six chapters quantizing does not appear.**
Weight precision sets where you can operate at a given latency; **it does not set
the ceiling.**

**So each technique raises a different end.** Weight quantization is worth
**3.9×** at batch 1 and **1.15×** at batch 512
({{eq:quantization-by-regime}}) — it stops working exactly where throughput
optimisation starts. Cache quantization and GQA raise the ceiling. Nothing raises
all of it.

**Speculative decoding spends the idle arithmetic** — **1.3% → 6.1%** utilisation
at batch 1, for **2.65×** — and gives **1.71×** at batch 256 because batching has
already spent it. **They are substitutes, not complements**
({{eq:speculation-spends-idleness}}), which is why two teams measure the same
feature differently and both are right.

**And its depth follows from a measured acceptance rate** rather than a sweep:
**k=4** at $\alpha = 0.5$ through **k=16** at 0.9, for **1.78×** to **6.20×**
({{eq:optimal-depth}}). **Select draft models on agreement, not quality** —
agreement enters exponentially and quality does not enter at all, because the
target's output distribution is preserved exactly.

Which is the thread through the whole part, arriving at its end: **every technique
here has a regime, and the regime is two measurable numbers — batch size and
context length.** Most disagreements about inference optimisation are two correct
measurements taken in different regimes, and the way to settle them is to state
the regime rather than to argue about the technique.

## 21. Further Reading

{{cite:pope2022inference}} for the framing this chapter formalises, and read its
partitioning results with {{eq:cache-caps-throughput}} in mind — several of its
conclusions are statements about which term dominates.
{{cite:leviathan2023speculative}} for the sampling rule that makes the
distribution provably identical, which is the property the whole technique rests
on and the one most often left out of summaries.
{{cite:cai2024medusa}} for removing the draft model without changing the regime
dependence.
{{cite:kwon2023pagedattention}} for the memory management that determines the
largest batch this chapter's frontier can reach.
{{cite:dettmers2023case4bit}} as the bookend: it settled the weight-precision
question this part opened with, and {{eq:batch-buys-throughput}} shows how much of
serving performance that decision does not touch.

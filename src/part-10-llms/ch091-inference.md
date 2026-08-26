---
id: llm-inference
number: 91
part: X
tier: full
status: draft
requires: [llm-anatomy, llm-decoding, tf-masking-kv, tf-complexity, tf-multi-head,
           tf-positional, dl-forward]
provides: [serving-phases, decode-phase, prefill-decode-asymmetry, cache-memory-growth,
           time-to-first-token, inter-token-latency, batching-tradeoff,
           context-limit-behaviour, arithmetic-intensity-serving, throughput-latency-frontier]
citations: [vaswani2017, shazeer2019mqa, ainslie2023gqa, dao2022flash,
            touvron2023llama, brown2020, liu2023lost, hoffmann2022chinchilla]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Distinguish prefill from decode and state which is compute-bound and which is
   memory-bound.
2. Derive why output tokens are priced several times higher than input tokens.
3. Compute KV-cache memory for a given model, context and batch, and find the
   concurrency limit it implies.
4. Explain why batching helps decode enormously and prefill barely.
5. Explain time-to-first-token and inter-token latency in terms of the two
   phases, and say what changes each.
6. Describe what happens at the context limit and why every option is bad.
7. Read a serving configuration and predict its throughput–latency behaviour.

## 2. Why This Matters

**This chapter explains your bill.** Every provider charges more for output
tokens than input tokens, usually three to five times more. That ratio is not a
margin decision — it is arithmetic, derived here in half a page, and
understanding it lets you predict costs rather than discover them.

**It explains your latency, which is two numbers rather than one.**
Time-to-first-token and inter-token latency have different causes, respond to
different fixes, and trade against each other. A team optimising "latency" as a
single quantity is optimising a number that does not exist.

**And it explains why concurrency is limited by memory rather than compute.**
The KV cache grows with every generated token, per request. At long context and
realistic batch sizes it exceeds the model weights, and that — not FLOPs — is
what caps how many users a server holds.

**{{ch:tf-masking-kv}} and {{ch:tf-complexity}} derived all of this
architecturally.** This chapter spends it, which is the point: the transformer
part of the book earned these results and this is where they buy something.

## 3. Prerequisites

{{ch:tf-masking-kv}} for the KV cache and why it exists — this chapter assumes
the derivation. {{ch:tf-complexity}} for $2N$ per token, the memory terms, and
the roofline. {{ch:llm-anatomy}} for the forward pass being repeated.
{{ch:llm-decoding}} for the sampling loop that drives it.
{{ch:tf-multi-head}} for grouped-query attention. {{ch:tf-positional}} for what
constrains context length. {{ch:dl-forward}} for arithmetic intensity.

## 4. Intuitive Explanation

Generating a response is two different computations wearing the same name.

**Prefill.** You send a 2,000-token prompt. The model processes all 2,000
positions at once — they are all known, so there is no sequential dependency.
This is one big matrix multiplication per layer, the hardware's favourite
operation, and it saturates the accelerator.

**Decode.** Now the model generates token 2,001. It needs the attention keys and
values for all 2,000 previous positions — which it computed during prefill and
*cached*, because recomputing them every step would be quadratic waste. So
decode processes exactly **one** position, reading the entire model's weights
and the entire cache to do it.

> NOTE: That asymmetry is the whole chapter. Prefill does $T$ positions' work in
> one pass and is limited by arithmetic. Decode does one position's work per pass
> and is limited by how fast the hardware can *read* — the weights and the cache
> both have to come off memory, and almost no arithmetic is done with them.

**Why output tokens cost more.** Prefill reads the model's weights once and uses
them on 2,000 positions. Decode reads the same weights once per *token*. A
200-token response reads the model 200 times. That is the entire price
asymmetry, and it is arithmetic rather than policy.

**Why batching changes everything for decode and little for prefill.** Decode
reads 14 GB of weights to produce one token for one user, which is
extraordinarily wasteful. Serve 32 users at once and you read the same 14 GB
*once* to produce 32 tokens. The weights are amortised across the batch, so
throughput rises nearly linearly until something else binds. Prefill was already
saturating the hardware, so batching adds little.

**And the cache is what stops you.** Each request's cache grows with its
conversation. Thirty-two users at 8,000 tokens each can easily need more memory
for cache than for the model. **Concurrency is a memory question**, and the
memory in question is not the weights.

**The mental model:** prefill is a matrix multiply and decode is a memory scan
dressed as one. Where it breaks down: at very long contexts prefill's quadratic
attention term reasserts itself ({{ch:tf-complexity}}), and the clean story
becomes two regimes.

## 5. Formal Explanation

### 5.1 The two phases

For a prompt of $T$ tokens generating $n$ output tokens, with $N$ parameters:

$$
C_{\text{prefill}} = \underbrace{2NT}_{\text{parameters}}
 + \underbrace{4LT^2 d}_{\text{attention}}
$$ (eq:prefill-flops)

$$
C_{\text{decode}} = n\Big(\underbrace{2N}_{\text{parameters}}
 + \underbrace{4Lg\,d_k\,\bar{n}}_{\text{attention over the cache}}\Big)
$$ (eq:decode-flops)

with $g$ KV heads and $\bar{n}\approx T + n/2$ the mean cache length.

**Prefill is quadratic in the prompt; decode is linear in the output.** The two
halves dominate in different regimes, which is {{ch:tf-complexity}}'s
{{eq:request-flops}} restated for serving.

### 5.2 The bytes, which are what actually matter

$$
B_{\text{prefill}} \approx bN, \qquad B_{\text{decode}} \approx n\,bN
$$ (eq:phase-bytes)

**Prefill reads the weights once in total; decode reads them once per token.**
That single line is the price asymmetry.

Arithmetic intensity — FLOPs per byte read ({{ch:dl-forward}}) — follows:

$$
I_{\text{prefill}} \approx \frac{2NT}{bN} = \frac{2T}{b},
\qquad
I_{\text{decode}} \approx \frac{2NB}{bN} = \frac{2B}{b}
$$ (eq:arithmetic-intensity-phases)

for batch size $B$. At $b = 2$ bytes: prefill with $T = 2000$ gives intensity
2,000; decode at $B=1$ gives **1**. Modern accelerators have ridge points around
200–400 FLOPs/byte.

$\square$

**Decode at batch 1 is two to three orders of magnitude below the ridge**, so
the accelerator is idle waiting for memory. Prefill is far above it and is
compute-bound. They are different computations on the same hardware.

### 5.3 KV cache size

From {{ch:tf-masking-kv}}:

$$
M_{\text{cache}} = 2 \cdot L \cdot g \cdot d_k \cdot T_{\text{total}} \cdot b \cdot B
$$ (eq:kv-cache-serving)

— a factor of 2 for keys and values, $L$ layers, $g$ KV heads of dimension
$d_k$, the total sequence length, bytes per element, batch size.

**No term in $N$.** Two models of identical parameter count can differ by an
order of magnitude here, which is what grouped-query attention
({{cite:ainslie2023gqa}}, {{cite:shazeer2019mqa}}) exploits: reduce $g$ and the
cache shrinks proportionally while the parameter count barely moves.

### 5.4 The two latencies

$$
\text{TTFT} = \frac{C_{\text{prefill}}}{\text{throughput}} + \text{queue}
$$ (eq:ttft)

$$
\text{ITL} = \frac{\max\big(C_{\text{decode}}^{\text{step}} / \text{FLOPS},\
 B_{\text{decode}}^{\text{step}} / \text{bandwidth}\big)}{1}
$$ (eq:itl)

**TTFT scales with prompt length; ITL does not.** ITL is dominated by the
bandwidth term and is nearly constant per token — which is why streaming feels
smooth once it starts, and why a long prompt delays the start rather than
slowing the flow.

> IMPORTANT: These respond to different interventions. TTFT improves with
> prompt shortening, prefix caching, and faster prefill kernels. ITL improves
> with batching, quantisation, and speculative decoding. **A change that helps
> one frequently hurts the other** — larger batches raise throughput and ITL
> while lengthening queueing and therefore TTFT.

### 5.5 At the context limit

When $T_{\text{total}}$ reaches the model's maximum, four options, all bad:

{#tbl:context-limit-options caption="What can be done at the context limit. Every row loses information, moves cost, or degrades quality — there is no option that simply extends the window."}

| Option | What is lost | Cost |
|---|---|---|
| Truncate the start | earliest context, often the system prompt | free |
| Truncate the middle | whatever was there | free |
| Sliding window | anything older than the window | free, breaks long dependencies |
| Summarise and restart | fidelity, irreversibly | an extra generation |
| Extend positions | quality, via extrapolation | fine-tuning ({{ch:tf-positional}}) |

**The default in most stacks is truncating the start**, which silently removes
the system prompt — the single most damaging thing to drop, and the least
visible.

## 6. Mathematical Foundation

### 6.1 The price ratio, derived

Cost per token in each phase, taking the binding resource. For decode at batch
$B$, the weights are read once per step and amortised over $B$ tokens:

$$
\text{cost}^{\text{decode}}_{\text{per token}} \propto \frac{bN}{B}
$$

For prefill, the weights are read once and amortised over $T$ positions:

$$
\text{cost}^{\text{prefill}}_{\text{per token}} \propto \frac{bN}{T}
$$

The ratio:

$$
\frac{\text{cost}^{\text{decode}}}{\text{cost}^{\text{prefill}}}
 = \frac{T}{B}
$$ (eq:price-ratio)

$\square$

**With a typical prompt of $T \approx 1000$ and a decode batch of $B\approx 200$,
the ratio is about 5** — which is close to the input/output price ratio every
provider charges. The pricing is the arithmetic, and the two levers a provider
has are the batch size they can achieve and the prompt lengths they see.

### 6.2 Why batching is nearly free for decode

Decode's per-step cost is
$\max(\text{compute}, \text{memory})$. At batch $B$:

$$
\text{compute} = 2NB,\qquad \text{memory} = bN + (\text{cache reads})
$$

The memory term's dominant part, $bN$, **does not depend on $B$** — the same
weights serve every sequence in the batch.

$$
\text{time per step} \approx \max\left(\frac{2NB}{\text{FLOPS}},\
 \frac{bN}{\text{bandwidth}}\right)
$$ (eq:decode-step-time)

While the second term dominates, **step time is constant in $B$ and throughput
is linear in it**. The crossover is at

$$
B^* = \frac{b\cdot \text{FLOPS}}{2\cdot\text{bandwidth}}
$$ (eq:batch-crossover)

$\square$

For a device at $10^{15}$ FLOPS and 3 TB/s with $b=2$:
$B^* = 2\times10^{15}/(2\times 3\times10^{12}) \approx 333$.

**Below batch ~333 you get throughput for free.** Above it, decode becomes
compute-bound and further batching costs latency without buying throughput.
Real systems rarely reach it because the *cache* runs out first.

### 6.3 The concurrency limit

Available memory $M$ splits between weights and cache:

$$
B_{\max} = \frac{M - bN}{2Lg\,d_k\,T_{\text{total}}\,b}
$$ (eq:max-concurrency)

$\square$

This is the number that matters operationally, and it falls linearly with
context length. A server holding 60 concurrent 2k-token conversations holds 15
at 8k, and the transition is invisible until it happens — the failure mode is
not gradual slowdown but sudden rejection.

### 6.4 A worked serving calculation

A 7B model, bf16, on an 80 GB device. $L = 32$, $h = 32$, $g = 8$ (GQA),
$d_k = 128$.

**Weights:** $2\times 7\times10^9 = 14$ GB.

**Cache per token:**

$$
2 \times 32 \times 8 \times 128 \times 2 = 131{,}072\ \text{bytes} = 128\ \text{KB}
$$

**At 4,096 tokens per conversation:** $128\text{ KB}\times 4096 = 0.54$ GB per
request.

**Concurrency:** $(80 - 14)/0.54 \approx 122$ requests.

Now the same model without GQA ($g = 32$): cache per token is $512$ KB, per
request $2.15$ GB, and concurrency is **30**.

> NOTE: `kv-cache-and-concurrency` computes this table for four
> configurations, and one row is worth pausing on: a 70B model's weights are
> 140 GB, so it does not fit on an 80 GB device at all and its concurrency is
> zero before any cache is allocated. Serving a large model is a multi-device
> problem before it is a throughput problem, which is {{part:23}}'s subject.

**Grouped-query attention quadrupled the number of users a server holds**, at a
parameter cost of well under one per cent. That is the single highest-leverage
architectural decision in serving, and it appears nowhere in the parameter
count.

## 7. Internal Mechanics

```mermaid {#fig:prefill-decode caption="The two phases. Prefill processes every prompt position in one compute-bound pass and fills the cache; decode then produces one token per pass, reading all the weights and the whole cache each time. The loop on the right is where the time and the money go."}
graph TD
  A["prompt, T tokens"] --> B["PREFILL<br/>all T positions at once"]
  B --> C["KV cache filled<br/>2·L·g·d_k·T·b bytes"]
  B --> D["logits for position T"]
  D --> E["sample one token<br/>ch:llm-decoding"]
  E --> F["DECODE step<br/>ONE position"]
  C --> F
  F --> G["append K,V to cache"]
  G --> H["logits"]
  H --> I{"stop?"}
  I -- no --> E
  I -- yes --> J["done"]
  style B fill:#dfe,stroke:#5a5
  style F fill:#fde,stroke:#c69
```

**Continuous batching.** Naive batching waits for every sequence in a batch to
finish, so one long generation stalls thirty short ones. Continuous batching
evicts finished sequences and admits new ones at every step, keeping the batch
full. It is the single largest throughput win in modern serving and it is a
scheduling change rather than a model change.

**Paged attention.** The cache for a request grows unpredictably, so allocating
a contiguous maximum-length block per request wastes most of it. Paged
allocation — fixed-size blocks with an indirection table, exactly as virtual
memory works — cuts the waste dramatically and is what allows the concurrency
{{eq:max-concurrency}} predicts to be approached rather than merely computed.

**Prefix caching.** Many requests share a prefix: the same system prompt, the
same few-shot examples, the same retrieved document. The cache for a shared
prefix can be computed once and reused, turning that part of prefill into a
memory read. **For a system with a long fixed system prompt this is the largest
available TTFT win**, and it requires only that the prefix be byte-identical —
which is another reason the template must be stable
({{ch:fm-instruction-tuning}}).

**Where FlashAttention helps and does not.** {{cite:dao2022flash}} removes
attention's $O(T^2)$ *memory* traffic during prefill, which is a large win
there. During decode there is no score matrix to materialise — attention is over
one query against the cache — so it does nothing. **FlashAttention is a prefill
optimisation**, and expecting it to speed up generation is a common
misattribution.

**Speculative decoding**, by contrast, is a decode optimisation: a small model
proposes several tokens and the large model verifies them in one pass, which
converts several memory-bound steps into one. It provably preserves the
distribution ({{ch:llm-decoding}}), so it is free quality-wise.

**Why it works is exactly {{eq:arithmetic-intensity-phases}}.** Verifying $k$
draft tokens costs one forward pass — the same weight read as verifying one —
because the drafted positions are all known and can be processed in parallel,
like prefill. So speculative decoding converts $k$ memory-bound steps into one
step that is $k$ times more arithmetically intense. **It is buying back the
arithmetic intensity that decode threw away**, which is why its speedup is
bounded by the draft acceptance rate and why it helps most exactly where decode
is furthest below the ridge point: small batches.

That last point has a counter-intuitive consequence worth stating. Speculative
decoding and batching are *substitutes*, not complements — both work by raising
arithmetic intensity, and a system already batching at 256 has much less to gain
from speculation than a single-user deployment does. Teams frequently adopt both
and are surprised the gains do not compose.

## 8. Implementation

The cache arithmetic and the concurrency limit it implies.

```python {tier=A name=kv-cache-and-concurrency}
"""KV cache memory, concurrency, and what GQA buys. Equation (eq:kv-cache-serving)."""

BYTES = 2                       # bf16


def cache_bytes_per_token(layers, kv_heads, head_dim, bytes_per=BYTES):
    """Equation (eq:kv-cache-serving), per token per sequence."""
    return 2 * layers * kv_heads * head_dim * bytes_per


MODELS = {
    "7B, MHA (g=32)":  dict(params=7e9,  layers=32, kv_heads=32, head_dim=128),
    "7B, GQA (g=8)":   dict(params=7e9,  layers=32, kv_heads=8,  head_dim=128),
    "7B, MQA (g=1)":   dict(params=7e9,  layers=32, kv_heads=1,  head_dim=128),
    "70B, GQA (g=8)":  dict(params=70e9, layers=80, kv_heads=8,  head_dim=128),
}
DEVICE_GB = 80

print(f"device {DEVICE_GB} GB, bf16\n")
print(f"{'model':<18} {'weights':>9} {'KV/token':>10} {'KV @4k':>10} "
      f"{'concurrency @4k':>17} {'@32k':>7}")
for name, m in MODELS.items():
    w = m["params"] * BYTES / 1e9
    per_tok = cache_bytes_per_token(m["layers"], m["kv_heads"], m["head_dim"])
    at_4k = per_tok * 4096 / 1e9
    at_32k = per_tok * 32768 / 1e9
    free = DEVICE_GB - w
    c4 = int(free / at_4k) if free > 0 else 0
    c32 = int(free / at_32k) if free > 0 else 0
    print(f"{name:<18} {w:>8.1f}G {per_tok / 1024:>9.0f}K {at_4k:>9.2f}G "
          f"{c4:>17} {c32:>7}")

mha = cache_bytes_per_token(32, 32, 128)
gqa = cache_bytes_per_token(32, 8, 128)
print(f"\nGQA (g=8) against MHA (g=32): cache per token "
      f"{mha / 1024:.0f}K -> {gqa / 1024:.0f}K, a {mha / gqa:.0f}x reduction")
print("Parameter count is essentially unchanged — the K and V projections "
      "shrink, and they are a small share of the block (ch:tf-ffn-residual).")

# Equation (eq:max-concurrency): concurrency falls linearly with context.
m = MODELS["7B, GQA (g=8)"]
per_tok = cache_bytes_per_token(m["layers"], m["kv_heads"], m["head_dim"])
free_gb = DEVICE_GB - m["params"] * BYTES / 1e9
print(f"\n{'context':>9} {'cache/request':>15} {'max concurrency':>17}")
for ctx in (1024, 4096, 16384, 65536, 131072):
    per_req = per_tok * ctx / 1e9
    print(f"{ctx:>9,} {per_req:>14.2f}G {int(free_gb / per_req):>17}")

print("""
Concurrency falls linearly with context length, and the failure mode is not
gradual: a server sized for 4k conversations holds 128 of them and 8 at 64k. It
does not slow down as it approaches the limit — it rejects requests.

Note also that at 128k context a single request needs more than 16 GB of cache
for a 7B model whose weights are 14 GB. THE CACHE IS LARGER THAN THE MODEL, for
one user. Every long-context serving decision follows from that inversion.""")
```

Now the asymmetry that explains the pricing:

```python {tier=A name=prefill-decode-asymmetry}
"""Why output tokens cost more than input tokens. Equation (eq:price-ratio)."""

N = 7e9
BYTES = 2
DEVICE_FLOPS = 1e15
BANDWIDTH = 3e12                # bytes/second
MFU = 0.45

PROMPT, OUTPUT = 1000, 200


def prefill_time(T, batch=1):
    flops = 2 * N * T * batch
    bytes_read = BYTES * N                       # weights, once
    return max(flops / (DEVICE_FLOPS * MFU), bytes_read / BANDWIDTH)


def decode_step_time(batch):
    """Equation (eq:decode-step-time)."""
    flops = 2 * N * batch
    bytes_read = BYTES * N                       # weights, once per STEP
    return max(flops / (DEVICE_FLOPS * MFU), bytes_read / BANDWIDTH)


print(f"{N / 1e9:.0f}B model, prompt {PROMPT}, output {OUTPUT} tokens\n")

# Arithmetic intensity, equation (eq:arithmetic-intensity-phases).
ridge = DEVICE_FLOPS / BANDWIDTH
print(f"device ridge point: {ridge:.0f} FLOPs/byte")
print(f"{'phase':<22} {'FLOPs/byte':>12} {'bound by':>12}")
print(f"{'prefill (T=' + str(PROMPT) + ')':<22} {2 * PROMPT / BYTES:>12.0f} "
      f"{'compute':>12}")
for B in (1, 32, 256):
    ai = 2 * B / BYTES
    print(f"{'decode (batch ' + str(B) + ')':<22} {ai:>12.0f} "
          f"{('memory' if ai < ridge else 'compute'):>12}")

pf = prefill_time(PROMPT)
ds = decode_step_time(1)
print(f"\nprefill {PROMPT} tokens : {pf * 1000:>8.1f} ms  "
      f"({PROMPT / pf:>10,.0f} tokens/s)")
print(f"decode  1 token       : {ds * 1000:>8.1f} ms  "
      f"({1 / ds:>10,.0f} tokens/s)")
print(f"per-token ratio       : {(ds) / (pf / PROMPT):>8.0f}x more expensive "
      f"to generate than to read")

# Equation (eq:price-ratio) against real pricing.
print(f"\n{'decode batch':>13} {'predicted price ratio (T/B)':>30}")
for B in (50, 100, 200, 400):
    print(f"{B:>13} {PROMPT / B:>30.1f}")
print("Providers charge 3-5x for output tokens. The arithmetic gives the same "
      "range at realistic batch sizes — the pricing is eq:price-ratio.")

# Batching: free throughput until the crossover of eq:batch-crossover.
crossover = BYTES * DEVICE_FLOPS * MFU / (2 * BANDWIDTH)
print(f"\nbatch crossover (eq:batch-crossover): B* = {crossover:.0f}")
print(f"{'batch':>7} {'step ms':>9} {'tokens/s':>11} {'per-token ms':>14}")
for B in (1, 8, 32, 128, 256, 512, 1024):
    st = decode_step_time(B)
    print(f"{B:>7} {st * 1000:>9.2f} {B / st:>11,.0f} {st * 1000 / B:>14.4f}")

print("""
Step time is FLAT up to the crossover — the same weights serve every sequence in
the batch, so batch 128 costs the same wall-clock per step as batch 1 and
produces 128 times the tokens. That is the largest free win in LLM serving.

Past the crossover decode becomes compute-bound and step time rises linearly, so
further batching buys throughput only at proportional latency cost. Most systems
never get there, because equation (eq:max-concurrency) runs out of cache memory
first.""")
```

And the two latencies, which is what users actually experience:

```python {tier=A name=ttft-and-itl}
"""Time-to-first-token and inter-token latency respond to different things."""

N, BYTES, DEVICE_FLOPS, BANDWIDTH, MFU = 7e9, 2, 1e15, 3e12, 0.45


def ttft_ms(prompt_tokens, batch, cached_prefix=0):
    """Equation (eq:ttft). Cached prefix tokens skip prefill compute."""
    new = max(prompt_tokens - cached_prefix, 1)
    flops = 2 * N * new * batch
    t = max(flops / (DEVICE_FLOPS * MFU), BYTES * N / BANDWIDTH)
    return t * 1000


def itl_ms(batch):
    """Equation (eq:itl) — nearly constant in prompt length."""
    t = max(2 * N * batch / (DEVICE_FLOPS * MFU), BYTES * N / BANDWIDTH)
    return t * 1000


print(f"{'prompt':>9} {'TTFT (b=1)':>12} {'ITL (b=1)':>11} "
      f"{'TTFT (b=64)':>13} {'ITL (b=64)':>12}")
for p in (100, 1000, 4000, 16000, 64000):
    print(f"{p:>9,} {ttft_ms(p, 1):>11.1f}m {itl_ms(1):>10.1f}m "
          f"{ttft_ms(p, 64):>12.1f}m {itl_ms(64):>11.1f}m")

print("""
TTFT scales with prompt length and ITL does not. That is why a long prompt
delays the START of streaming rather than slowing it down, and why users
describe long-context requests as "slow to begin" rather than "slow".""")

# The tradeoff: batching improves throughput and ITL, and hurts TTFT via queueing.
print(f"\n{'batch':>7} {'ITL ms':>9} {'tokens/s':>11} {'queue wait ms':>15} "
      f"{'effective TTFT':>16}")
ARRIVAL_RATE = 40                # requests/second
for B in (1, 8, 32, 128, 256):
    itl = itl_ms(B)
    tput = B / (itl / 1000)
    # A larger batch means waiting for it to fill.
    queue = (B / ARRIVAL_RATE) * 1000 / 2
    print(f"{B:>7} {itl:>9.2f} {tput:>11,.0f} {queue:>15.1f} "
          f"{ttft_ms(1000, B) + queue:>15.1f}m")

print("""
The two metrics move in opposite directions. Larger batches raise throughput and
leave ITL unchanged until the crossover, and they lengthen the wait to assemble
a batch — so effective TTFT rises. A system tuned for throughput feels
unresponsive to start and fast once started.

This is why 'latency' is not one number. Decide which one your product is
sensitive to: a chat interface lives on TTFT, a batch summarisation job lives on
throughput, and they want opposite configurations.""")

# Prefix caching, the largest TTFT win for a fixed system prompt.
SYSTEM_PROMPT = 800
print(f"\nprefix caching with an {SYSTEM_PROMPT}-token system prompt:")
print(f"{'user text':>11} {'TTFT uncached':>15} {'TTFT cached':>13} "
      f"{'saving':>9}")
for user in (50, 200, 1000):
    total = SYSTEM_PROMPT + user
    un, ca = ttft_ms(total, 1), ttft_ms(total, 1, cached_prefix=SYSTEM_PROMPT)
    print(f"{user:>11,} {un:>14.1f}m {ca:>12.1f}m "
          f"{(1 - ca / un):>8.0%}")
print("The saving is largest exactly where the system prompt dominates the "
      "request — which is the common case for an assistant with detailed "
      "instructions.")
```

## 9. Practical Example

A team serves a document-analysis assistant: 12,000-token prompts (a retrieved
document plus instructions), 400-token answers, and a target of 200 concurrent
users. They have 8 × 80 GB devices and want to know whether that is enough, and
what will break first.

```python {tier=A name=capacity-planning}
"""Sizing a deployment: what binds, and at what point."""

MODEL = dict(params=7e9, layers=32, kv_heads=8, head_dim=128)
BYTES = 2
DEVICES, DEVICE_GB = 8, 80
DEVICE_FLOPS, BANDWIDTH, MFU = 1e15, 3e12, 0.45

PROMPT, OUTPUT, TARGET_USERS = 12_000, 400, 200

weights_gb = MODEL["params"] * BYTES / 1e9
per_token = 2 * MODEL["layers"] * MODEL["kv_heads"] * MODEL["head_dim"] * BYTES
per_request_gb = per_token * (PROMPT + OUTPUT) / 1e9

print(f"weights            : {weights_gb:.1f} GB")
print(f"cache per request  : {per_request_gb:.2f} GB "
      f"({PROMPT + OUTPUT:,} tokens x {per_token / 1024:.0f} KB)")
print(f"cache for {TARGET_USERS} users: "
      f"{per_request_gb * TARGET_USERS:.0f} GB\n")

total_gb = DEVICES * DEVICE_GB
usable = total_gb - weights_gb * DEVICES     # weights replicated per device
max_users_mem = int(usable / per_request_gb)
print(f"{'total device memory':<28} {total_gb:>8.0f} GB")
print(f"{'weights (replicated x' + str(DEVICES) + ')':<28} "
      f"{weights_gb * DEVICES:>8.0f} GB")
print(f"{'available for cache':<28} {usable:>8.0f} GB")
print(f"{'-> max concurrent users':<28} {max_users_mem:>8}")
print(f"{'target':<28} {TARGET_USERS:>8}")
print(f"{'verdict':<28} "
      f"{('FITS' if max_users_mem >= TARGET_USERS else 'DOES NOT FIT'):>8}\n")

# What binds: memory or compute?
prefill_flops = 2 * MODEL["params"] * PROMPT
decode_flops = 2 * MODEL["params"] * OUTPUT
per_request_flops = prefill_flops + decode_flops
cluster_flops = DEVICES * DEVICE_FLOPS * MFU

print(f"{'per-request FLOPs':<28} {per_request_flops:>10.2e}")
print(f"{'cluster FLOPs/s':<28} {cluster_flops:>10.2e}")
print(f"{'-> requests/second (compute)':<28} "
      f"{cluster_flops / per_request_flops:>10.1f}")

# And the split: how much of the work is prefill?
print(f"\n{'phase':<12} {'FLOPs':>12} {'share':>8}")
for name, f in [("prefill", prefill_flops), ("decode", decode_flops)]:
    print(f"{name:<12} {f:>12.2e} {f / per_request_flops:>7.0%}")

print(f"""
This workload is {prefill_flops / per_request_flops:.0%} PREFILL, which inverts
the usual advice. With a 12,000-token prompt and a 400-token answer, most of the
compute is reading the document, not writing the answer — so batching (which
helps decode) buys much less than it would for a chat workload, and prefill
throughput is what to optimise.

And the binding constraint is memory, not compute: the cluster could serve
{cluster_flops / per_request_flops:.0f} requests/second on arithmetic alone,
while cache memory caps concurrency at {max_users_mem}.""")

# The intervention that actually helps here.
print(f"\n{'intervention':<32} {'cache/request':>15} {'max users':>11}")
options = [
    ("as-is", per_request_gb, ""),
    ("KV cache in fp8", per_request_gb / 2, "halves cache, small quality cost"),
    ("halve the prompt (rerank first)", per_request_gb *
     (PROMPT / 2 + OUTPUT) / (PROMPT + OUTPUT), "ch:emb-reranking"),
    ("both", per_request_gb / 2 *
     (PROMPT / 2 + OUTPUT) / (PROMPT + OUTPUT), ""),
]
for label, cache, note in options:
    print(f"{label:<32} {cache:>14.2f}G {int(usable / cache):>11}")

print("""
Shortening the prompt helps twice over: less cache per request AND less prefill
compute, which is the phase this workload is dominated by. Retrieving fewer,
better passages is therefore not only a quality decision — it is the single
largest lever on both cost and capacity here, which is a good reason to read
Part XI before buying more hardware.""")
```

> PRODUCTION TIP: Compute {{eq:max-concurrency}} at your *maximum* context, not
> your median. Concurrency limits bind suddenly, and a service sized on median
> conversation length rejects requests the first time usage shifts long.

## 10. Production Considerations

**Report TTFT and ITL separately.** They have different causes and different
fixes, and a single "latency" number hides which one is failing.

**Size for maximum context.** {{eq:max-concurrency}} falls linearly with
context and the failure is rejection rather than slowdown.

**Enable continuous batching and paged attention.** Both are scheduling and
allocation changes rather than model changes, and together they are the
difference between the concurrency {{eq:max-concurrency}} predicts and what a
naive implementation achieves.

**Use prefix caching for any fixed system prompt.** It requires byte-identical
prefixes, which is another reason the chat template must be stable.

**Quantise the KV cache before quantising the weights** when memory-bound at
long context. The cache is where the memory is going ({{part:15}}).

**What to monitor:** TTFT and ITL at p50 and p99, cache utilisation as a
fraction of capacity, batch size achieved, request rejection rate, and the
distribution of context lengths. The last one predicts the others.

## 11. Common Mistakes

**Beginners:**

*Treating latency as one number.* TTFT and ITL respond to opposite
interventions.

*Assuming batching helps everything.* It transforms decode and barely touches
prefill.

*Ignoring cache memory when sizing.* At long context it exceeds the weights.

**Experienced practitioners:**

*Expecting FlashAttention to speed up generation.* It is a prefill optimisation;
decode has no score matrix to tile ({{sec:7-internal-mechanics}}).

*Sizing on median context.* {{eq:max-concurrency}} is a hard limit at the
maximum, not the median.

*Truncating from the start at the context limit.* It silently removes the
system prompt, which is the most damaging thing to lose and the least visible.

*Comparing models by parameter count for serving.* {{eq:kv-cache-serving}} has
no $N$ term — GQA changes concurrency fourfold at constant size.

*Assuming speculative decoding and batching compose.* Both work by raising
arithmetic intensity, so they are substitutes: a system already batching well
has little left for speculation to recover
({{sec:7-internal-mechanics}}).

## 12. Failure Modes

**Cache exhaustion.** *Symptom:* sudden request rejection at a particular
concurrency, with no gradual degradation. *Detection:* cache utilisation as a
monitored fraction. *Cause:* {{eq:max-concurrency}}.

**TTFT collapse under long prompts.** *Symptom:* users report the system
"hanging" before responding. *Cause:* prefill is linear in $T$ and quadratic
past $T > 6d$ ({{ch:tf-complexity}}).

**Head-of-line blocking.** One very long generation occupies a batch slot while
short requests queue. *Fix:* continuous batching.

**Silent context truncation.** The prompt exceeded the window and the start was
dropped. *Symptom:* the model ignoring its instructions, intermittently and
correlated with input length. *Detection:* log the truncation rate — this is
{{ch:nlp-preprocessing}}'s warning at a different layer.

**Quality degradation at long context.** Not a capacity failure —
{{cite:liu2023lost}}'s position effect, treated in {{ch:llm-long-context}}.
Worth naming here because it is frequently misdiagnosed as a serving problem.

**Throughput tuning that ruins interactivity.** Large batches raise throughput
and lengthen the queue. *Detection:* p99 TTFT, which is where it shows first.

## 13. Alternatives

{#tbl:serving-optimisations caption="Serving optimisations by which phase they help. The last column is the one to check before adopting anything: several widely-recommended techniques help only one phase, and workloads differ in which phase dominates."}

| Technique | Helps | Cost | Changes output? |
|---|---|---|---|
| Continuous batching | decode throughput | scheduler complexity | no |
| Paged attention | concurrency | allocator complexity | no |
| Prefix caching | TTFT | cache storage | no |
| FlashAttention | prefill | none | no |
| Speculative decoding | ITL | a draft model | **no** — provably |
| KV cache quantisation | concurrency | small quality cost | slightly |
| Weight quantisation | weights memory, ITL | quality cost | slightly |
| GQA / MQA | concurrency | architectural, at training time | yes |

**What genuinely differs.** The first five preserve the output distribution
exactly — they are systems work. The last three change the computed function,
and are therefore quality decisions rather than engineering ones. **Speculative
decoding is the one people assume is approximate and is not**: its acceptance
test is constructed so the resulting distribution is identical to the target
model's.

## 14. Evaluation

**Is the serving configuration right?**

1. **TTFT and ITL at p50 and p99**, measured under realistic prompt-length
   distribution rather than a fixed benchmark prompt.
2. **Achieved batch size** against {{eq:batch-crossover}} — if it is far below,
   something other than compute is binding, and it is usually cache.
3. **Cache utilisation** and the rejection rate.
4. **Cost per request**, split by phase using {{eq:price-ratio}}.

**Is the workload what you think?** Compute the prefill/decode split, as
`capacity-planning` does. A workload that is 97% prefill wants entirely
different optimisation from one that is 20% prefill, and teams routinely apply
chat-workload advice to document-analysis workloads.

## 15. Advanced Concepts

**Disaggregated serving.** {{maturity:EMERGING}} Running prefill and decode on
*separate* hardware pools, since one is compute-bound and the other
memory-bound. Lets each be provisioned for its actual constraint, at the cost of
transferring the cache between them.

**Chunked prefill.** {{maturity:ESTABLISHED}} Splitting a long prefill into
pieces interleaved with decode steps, so a long prompt does not stall everyone
else's generation. Trades one request's TTFT for everyone else's, which is a
fairness decision rather than a performance one — and note that it only makes
sense because the two phases have *different* bottlenecks: interleaving a
compute-bound chunk with memory-bound decode steps uses both resources at once
rather than alternating between saturating one and idling it.

**Cache compression and eviction.** {{maturity:EMERGING}} Dropping or
compressing cache entries for positions unlikely to be attended to. Directly
attacks {{eq:max-concurrency}} and interacts with
{{cite:liu2023lost}}'s position effects in ways that are not fully characterised.

**Speculative decoding variants.** {{maturity:ESTABLISHED}} Draft models,
$n$-gram lookup, and self-speculation via early layers. All preserve the target
distribution; they differ in how good the draft is and what it costs.

**Attention alternatives at serving time.** {{maturity:EMERGING}} State-space
models ({{ch:tf-efficient}}) have a constant-size state rather than a growing
cache, which removes {{eq:max-concurrency}}'s context dependence entirely. That
is their strongest practical argument and it is a serving argument rather than a
quality one.

## 16. Connection to Previous Chapters

**Backwards.** {{ch:tf-masking-kv}} derived the cache and this chapter sizes it;
{{ch:tf-complexity}}'s {{eq:request-flops}} and {{eq:request-bytes}} are
{{eq:prefill-flops}} and {{eq:phase-bytes}} restated for serving, and its
roofline gives {{eq:arithmetic-intensity-phases}}.
{{ch:tf-multi-head}}'s grouped-query attention is the highest-leverage lever in
{{eq:kv-cache-serving}}. {{ch:llm-decoding}}'s sampling loop is what drives the
decode phase. {{ch:llm-anatomy}}'s observation that unembedding all positions is
wasteful is a prefill optimisation.

**Forwards.** {{ch:llm-prompt-lifecycle}} places these phases in the full
request path. {{ch:llm-long-context}} takes up quality at long context, which is
a different problem from capacity. {{part:15}} quantises both weights and cache.
{{part:23}} builds the serving systems this chapter's arithmetic specifies, and
{{ch:emb-reranking}} is the lever `capacity-planning` identifies as largest.

## 17. Exercises

**Beginner**

1. Why does prefill process all prompt tokens at once and decode only one?
2. Compute the KV cache per token for $L=40$, $g=8$, $d_k=128$, bf16.
3. Why are output tokens more expensive than input tokens?

**Intermediate**

4. Using {{eq:max-concurrency}}, find the concurrency for a 13B model on 80 GB
   at 16k context with $L=40$, $g=8$, $d_k=128$.
5. Compute {{eq:batch-crossover}} for a device at $2\times10^{15}$ FLOPS and
   4 TB/s.
6. A workload has 500-token prompts and 2,000-token outputs. What fraction is
   prefill, and what should be optimised?

**Advanced**

7. Derive {{eq:price-ratio}} and explain what a provider can change to move it.
8. Show that decode step time is constant in batch size below the crossover, and
   say what that implies for pricing at low utilisation.
9. Explain why FlashAttention helps prefill and not decode, in terms of what is
   materialised in each phase.

**Implementation**

10. Extend `kv-cache-and-concurrency` with a paged allocator and measure the
    fragmentation waste against contiguous allocation at varying sequence
    lengths.
11. Implement continuous batching in simulation: a request stream with varying
    generation lengths, comparing naive and continuous batching on throughput
    and p99 TTFT.
12. Model chunked prefill and find the chunk size minimising p99 TTFT for a
    mixed workload of short and long prompts.
13. Add speculative decoding to `prefill-decode-asymmetry`, parameterised by
    draft acceptance rate, and find the rate at which it stops paying.

**Reasoning**

14. Your service is memory-bound at long context. Rank quantising the weights,
    quantising the cache, and switching to a GQA model, with reasons.
15. Explain why a chat product and a batch summarisation product want opposite
    serving configurations.

## 18. Interview Questions

**Beginner**

1. What is the difference between prefill and decode?
2. What is the KV cache and why does it exist?
3. Why do providers charge more for output tokens?

**Intermediate**

4. Why does batching help decode so much more than prefill?
5. What limits concurrency on an LLM server?
6. What is TTFT and what changes it?

**Senior**

7. Size a deployment for 200 concurrent users at 12k context. What binds?
8. Your p99 TTFT regressed after a throughput optimisation. Explain.
9. When would you choose disaggregated prefill and decode?

**Systems**

10. Design the scheduler for an LLM server. What does it optimise and what does
    it trade?
11. How would you detect cache exhaustion before it causes rejections?

## 19. Research Questions

**How much cache can be evicted without measurable quality loss?**
{{eq:max-concurrency}} makes cache the binding constraint, and eviction attacks
it directly. Measure quality against eviction policy and rate, and check the
interaction with {{cite:liu2023lost}}'s position effects — evicting the middle
may be nearly free precisely because the model was not using it.

**Is the price ratio stable?** {{eq:price-ratio}} gives $T/B$, so it depends on
the workload mix a provider sees. Whether observed pricing tracks it across
providers and over time is checkable from public price lists and is a good test
of whether the model in this chapter is right.

**Where is the disaggregation crossover?** Separating prefill and decode helps
when the phases' resource profiles differ enough to outweigh the cache transfer.
Characterise that boundary as a function of prompt length, output length and
interconnect bandwidth.

**Do constant-state architectures actually win at serving?** SSMs remove the
cache's context dependence entirely. Compare end-to-end throughput and quality
at matched parameters and long context — the serving argument is strong and the
quality comparison at matched budget is rarely made.

## 20. Chapter Summary

Generation is two computations. **Prefill** processes all $T$ prompt positions in
one pass, is compute-bound, and fills the KV cache. **Decode** produces one token
per pass, reading the entire model and the entire cache to do it, and is
memory-bound by two to three orders of magnitude
{{eq:arithmetic-intensity-phases}}.

**That asymmetry is the price of an output token.** Prefill reads the weights
once and amortises them over $T$ positions; decode reads them once per token.
{{eq:price-ratio}} gives the ratio as $T/B$ — about 5 at a 1,000-token prompt
and a decode batch of 200, which is the range every provider charges.

**Batching is free for decode and nearly useless for prefill.** Below the
crossover {{eq:batch-crossover}}, step time is *constant* in batch size because
the same weights serve every sequence — so throughput is linear and latency is
unchanged. Most systems never reach the crossover because
{{eq:max-concurrency}} runs out of cache memory first.

**Concurrency is a memory question, and the memory is not the weights.**
{{eq:kv-cache-serving}} has no term in $N$ at all: cache scales with layers, KV
heads, head dimension and sequence length. Grouped-query attention exploits
exactly this, cutting cache fourfold at negligible parameter cost — the single
highest-leverage serving decision, invisible in a model's advertised size. And
at 128k context a 7B model's cache for **one** user exceeds its weights.

**The two latencies move in opposite directions.** TTFT scales with prompt
length; ITL does not, which is why long prompts feel slow to *start* rather than
slow. Larger batches raise throughput and lengthen queueing, so a
throughput-tuned system feels unresponsive to begin and fast once going.

Finally, know your workload. `capacity-planning`'s document-analysis case is 97%
prefill, which inverts the standard advice — batching buys little, and the
largest lever on both cost and capacity is retrieving fewer, better passages.
Chat-workload guidance applied to a long-prompt workload optimises the wrong
phase.

## 21. Further Reading

{{cite:dao2022flash}} for FlashAttention, read here with the phase distinction in
mind: its argument is entirely about memory traffic during a pass that
materialises a $T\times T$ score matrix, which is prefill. That reading makes
obvious why it does nothing for decode.

{{cite:ainslie2023gqa}} and {{cite:shazeer2019mqa}} for grouped and multi-query
attention. They are short, and their motivation is exactly
{{eq:kv-cache-serving}} — read them as serving papers rather than architecture
papers, which is how they were intended.

{{cite:touvron2023llama}}'s configuration tables are worth having open while
computing {{eq:max-concurrency}} for a real model, since they give the $L$, $g$
and $d_k$ the equation needs.

**Where to go next:** {{ch:llm-prompt-lifecycle}} assembles everything from
{{ch:llm-anatomy}} onward into the complete path a request takes — template,
tokenizer, batch, prefill, sampling loop, stop conditions, detokenizer,
stream — and shows where each millisecond goes.

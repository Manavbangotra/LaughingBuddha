---
id: tf-masking-kv
number: 69
part: VII
tier: full
status: reviewed
requires: [tf-architectures, tf-multi-head, tf-positional, dl-forward]
provides: [causal-mask, kv-cache, prefill, decode, incremental-decoding,
           cache-eviction, paged-attention, prompt-caching, memory-bound-decode]
citations: [vaswani2017, shazeer2019mqa, ainslie2023gqa, dao2022flash, radford2019]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Derive why incremental decoding without a cache is quadratically wasteful.
2. Compute the KV cache size for any model and context length.
3. Explain the prefill/decode split and why the two phases have different
   bottlenecks.
4. Explain why decoding is memory-bound and what that implies for batching.
5. Explain why RoPE's absolute rotation is baked into the cached key, and what
   that forbids.
6. Explain paged attention and prompt caching.
7. Reason about cache eviction and why the first token is special.

## 2. Why This Matters

**A transformer trains in one parallel pass and generates one token at a
time.** That asymmetry is the single most important operational fact about
deploying one, and everything in this chapter follows from it.

**Without a cache, generating $n$ tokens costs $O(n^3)$.** With one, $O(n^2)$.
{{sec:6-mathematical-foundation}} derives both, and the derivation is short
enough that it is worth doing rather than quoting.

**The cache — not the weights — is what limits how many users you can serve.**
{{ch:tf-multi-head}} measured a 70B model's cache exceeding its weights at a
32k context. Every serving decision in {{part:23}} is downstream of that
number.

**Decoding is memory-bound, and this reframes everything.** Producing one token
reads every parameter of the model from memory and does almost no arithmetic
with them. Batching is not an optimisation; it is the only thing that makes the
machine do work at all.

## 3. Prerequisites

{{ch:tf-architectures}} for the causal mask.
{{ch:tf-multi-head}} for {{eq:kv-cache-size}} and grouped-query attention.
{{ch:tf-positional}} for RoPE's absolute rotation, which is what makes cached
keys position-locked. {{ch:dl-forward}} for arithmetic intensity, which is the
whole analysis in {{sec:6-mathematical-foundation}}.

## 4. Intuitive Explanation

### 4.1 The waste

To generate token $n+1$, a naive implementation runs the whole model on the
whole sequence $t_1..t_n$ and keeps only the last position's output:

```text
   step 1:  run on [t1]           -> t2      (1 position of work)
   step 2:  run on [t1,t2]        -> t3      (2 positions)
   step 3:  run on [t1,t2,t3]     -> t4      (3 positions)
   ...
   step n:  run on [t1..tn]       -> t(n+1)  (n positions)
```

Total work $1 + 2 + \dots + n = O(n^2)$ *positions*, each costing $O(d^2)$, so
$O(n^2 d^2)$ — and the attention within each pass adds another factor.

**Almost all of it is recomputation.** At step $n$, positions $1..n-1$ produce
exactly the keys and values they produced at step $n-1$, because the causal mask
means nothing after them influenced them. The model recomputes them anyway.

### 4.2 The fix

Keep them.

```text
   step 1:  compute K,V for t1              cache = [k1, v1]
   step 2:  compute K,V for t2 only         cache = [k1,k2, v1,v2]
            attend q2 over the cache
   step 3:  compute K,V for t3 only         cache grows by one
            attend q3 over the cache
```

Each step now does $O(d^2)$ work for the new position's projections plus
$O(nd)$ for attending over the cache. **The cubic total becomes quadratic, and
the quadratic part is the unavoidable cost of attention over a growing
context.**

This is not an optimisation in the ordinary sense — every serving system does
it, and a transformer without a KV cache is not a viable thing to deploy.

### 4.3 Why only K and V

The query is different at every step because it comes from the *new* token. The
keys and values are per-position and, under a causal mask, are never
recomputed differently.

```text
   Q  changes every step, cannot be cached
   K  fixed once a position exists     CACHE
   V  fixed once a position exists     CACHE
```

Nothing else in the model needs caching, because the feed-forward block is
position-wise ({{ch:tf-ffn-residual}}) and the residual stream at earlier
positions is not consulted again.

### 4.4 Two phases with two bottlenecks

```text
   PREFILL          process the whole prompt at once
                    T positions in parallel
                    COMPUTE-bound, high arithmetic intensity

   DECODE           one token at a time
                    1 position, reading all the weights and all the cache
                    MEMORY-bound, arithmetic intensity ~2 at batch 1
```

They are so different that serving systems schedule them separately, report
their latencies separately (time-to-first-token against time-per-output-token),
and sometimes run them on different hardware.

**The single most counter-intuitive consequence: at batch size 1, generating a
token takes about as long for a short context as a long one**, because the time
is dominated by reading the model's weights, not by attending over the cache.
{{sec:8-implementation}} measures where that stops being true.

### 4.5 What the cache costs

Per token, per layer, you store $2 g d_k$ numbers — keys and values for $g$
key/value heads. Multiply by the layers, the context length, the batch size and
the bytes per element, and it becomes the dominant memory consumer.

For a 70B model at a 32k context, {{ch:tf-multi-head}} measured 84 GB per
sequence under full multi-head attention, against 140 GB of weights *shared
across all users*. **The weights are a fixed cost and the cache is per-user**,
which is why grouped-query attention was adopted so quickly.

## 5. Formal Explanation

### 5.1 Cached attention

At decoding step $n$, with cache
$\mat{K}_{1:n-1}, \mat{V}_{1:n-1} \in \R^{(n-1)\times d_k}$ per head:

$$
\vec{q}_n = \mat{W}^Q\vec{x}_n,
\qquad
\vec{k}_n = \mat{W}^K\vec{x}_n,
\qquad
\vec{v}_n = \mat{W}^V\vec{x}_n
$$ (eq:decode-projections)

$$
\mat{K}_{1:n} = [\mat{K}_{1:n-1}; \vec{k}_n\T],
\qquad
\mat{V}_{1:n} = [\mat{V}_{1:n-1}; \vec{v}_n\T]
$$ (eq:cache-append)

$$
\vec{o}_n = \softmax\!\left(\frac{\mat{K}_{1:n}\vec{q}_n}{\sqrt{d_k}}\right)\T
 \mat{V}_{1:n}
$$ (eq:cached-attention)

**No mask appears.** The cache contains only positions $\le n$, so causality is
enforced by what is in the cache rather than by masking what is not — which is
why decode kernels are simpler than training kernels.

### 5.2 Cache size

$$
M_{\text{KV}} = 2\,b\,L\,g\,d_k\,T\,B
$$ (eq:cache-size-full)

for $B$ concurrent sequences. The factor 2 is keys and values; $g$ is the
key/value head count, which equals $h$ for multi-head attention, $h/8$ for
typical grouped-query attention, and 1 for multi-query.

### 5.3 The cost of a decode step

$$
F_{\text{decode}} = \underbrace{2P}_{\text{weights}}
 + \underbrace{4Lg d_k n}_{\text{attend over the cache}}
$$ (eq:decode-flops)

$$
B_{\text{decode}} = \underbrace{bP}_{\text{weights}}
 + \underbrace{2bLg d_k n}_{\text{read the cache}}
$$ (eq:decode-bytes)

for a model with $P$ parameters at context length $n$. Dividing gives the
arithmetic intensity, and it is $2/b$ for both terms — **about 1 operation per
byte in bf16, whatever the context length**.

That number is two to three orders of magnitude below any modern accelerator's
ridge point ({{ch:dl-forward}}), so decoding at batch size 1 leaves the machine
essentially idle.

### 5.4 Batching

With $B$ sequences decoded together, the weights are read once and serve $B$
tokens:

$$
I_{\text{weights}} = \frac{2BP}{bP} = \frac{2B}{b}
$$ (eq:decode-intensity)

**Linear in the batch size.** At $B = 64$ in bf16 the intensity is 64, which is
approaching a useful regime.

The cache term does *not* improve with batching, because each sequence has its
own cache:

$$
I_{\text{cache}} = \frac{4Lgd_k nB}{2bLgd_k nB} = \frac{2}{b}
$$ (eq:cache-intensity)

**Independent of $B$ and of $n$.** So as the context grows, an increasing
fraction of the decode step is spent on a term that batching cannot help — which
is the real long-context serving problem and the reason
{{ch:tf-efficient}}'s cache-compression methods matter.

### 5.5 Where the crossover is

The cache term exceeds the weight term when

$$
2bLg d_k n > bP
\quad\Longleftrightarrow\quad
n > \frac{P}{2Lgd_k}
$$ (eq:cache-crossover)

For a 70B model with $L=80$, $g=8$, $d_k=128$: $n > 427{,}000$ tokens. Under
full multi-head attention ($g=64$) it is $n > 53{,}000$.

**So for grouped-query models at realistic contexts, decode time is dominated by
reading the weights**, and the cache is a memory-capacity problem rather than a
bandwidth one. That distinction matters: the first is fixed by compression, the
second by batching, and confusing them leads to optimising the wrong thing.

### 5.6 What the phases cost

{#tbl:prefill-decode caption="Prefill against decode. The two phases differ in every respect that matters operationally, which is why they are scheduled, measured and sometimes hosted separately."}

| | Prefill | Decode |
|---|---|---|
| Positions per pass | $T$ | 1 |
| Bottleneck | compute | memory bandwidth |
| Arithmetic intensity | $\approx T$ | $\approx 2B/b$ |
| Parallel over | positions | sequences only |
| Latency metric | time to first token | time per output token |
| Scales with | $T^2$ (attention) | $n$ (cache reads) |

### 5.7 What the cache does to a request's cost profile

A request has two costs and they scale differently, which is why LLM pricing has
two rates.

$$
C_{\text{prefill}} \propto 2PT + 4Ld\,T^2,
\qquad
C_{\text{decode}} \propto n\big(2P + 4Lgd_k\bar{n}\big)
$$ (eq:request-cost)

with $T$ the prompt length, $n$ the generated length and $\bar{n} \approx
T + n/2$ the average context during generation.

Three readings.

**Prefill is quadratic in the prompt and decode is linear in the output.** A
16k-token prompt with a 100-token answer is dominated by prefill; a 100-token
prompt with a 4k-token answer is dominated by decode. These are different
workloads on the same model.

**Output tokens cost more than input tokens, per token.** Each output token
reads all the weights; input tokens are processed in parallel and amortise that
read across the whole prompt. That is the arithmetic behind output tokens being
priced several times higher than input tokens, and it is not a margin decision.

**Cached input tokens cost almost nothing.** They skip prefill entirely and only
contribute to the cache the later tokens attend over. The measured saving in
{{sec:9-practical-example}} is superlinear in the shared fraction because the
quadratic attention term over the shared prefix disappears.

**So the cheapest possible request shape is a long cached prefix, a short
uncached suffix, and a short output** — and that is exactly the shape that
retrieval-augmented systems and agent loops with stable system prompts produce,
which is not a coincidence.

## 6. Mathematical Foundation

### 6.1 The cubic-to-quadratic reduction

Without a cache, generating $n$ tokens re-runs the model on prefixes of length
$1, 2, \dots, n$. Each pass over $m$ positions costs

$$
C(m) = \alpha m d^2 + \beta m^2 d
$$

for the projections and the attention. Summing:

$$
C_{\text{total}} = \sum_{m=1}^{n} \big(\alpha m d^2 + \beta m^2 d\big)
 = \alpha d^2\frac{n(n+1)}{2} + \beta d\frac{n(n+1)(2n+1)}{6}
$$ (eq:no-cache-cost)

which is $\Theta(n^2 d^2 + n^3 d)$ — **cubic in the number of generated
tokens.**

With a cache, step $m$ costs $\alpha d^2$ for one position's projections and
$\beta m d$ for attending over $m$ cached positions:

$$
C_{\text{cached}} = \sum_{m=1}^{n}\big(\alpha d^2 + \beta m d\big)
 = \alpha n d^2 + \beta d\frac{n(n+1)}{2}
$$ (eq:cache-cost)

$\Theta(nd^2 + n^2 d)$. **A full factor of $n$ removed from both terms.**
{{sec:8-implementation}} measures the ratio.

The remaining $n^2 d$ is irreducible: attention over a growing context is
quadratic in total, however it is computed. That is
{{ch:tf-complexity}}'s subject.

### 6.2 Why the cache is exactly correct

The cached implementation must produce *identical* output to the uncached one,
and it is worth seeing why rather than assuming it.

Under a causal mask, position $i$'s hidden state depends only on $t_{\le i}$.
Therefore $\vec{k}_i = \mat{W}^K\vec{x}_i$ depends only on $t_{\le i}$, and
appending $t_{n}$ to the sequence cannot change $\vec{k}_i$ for any $i < n$.

$$
\vec{k}_i\big(t_{1:n}\big) = \vec{k}_i\big(t_{1:m}\big)
\qquad \text{for all } m \ge i
$$ (eq:cache-invariance)

$\square$

**The causal mask is what makes caching possible**, and this is the deepest
connection between {{ch:tf-architectures}}'s mask and this chapter. A
bidirectional model cannot cache anything, because appending a token changes
every earlier position's representation.

{{sec:8-implementation}} verifies {{eq:cache-invariance}} numerically, and it is
worth doing on any implementation — a cache bug produces output that is fluent
and subtly wrong.

### 6.3 Why decoding time is flat in context length

From {{eq:decode-bytes}}, the bytes read per step are
$bP + 2bLgd_k n$. The first term is constant; the second grows with $n$.

Their ratio at context $n$:

$$
\frac{\text{cache bytes}}{\text{weight bytes}}
 = \frac{2Lgd_k n}{P}
$$ (eq:cache-weight-ratio)

For a 7B model with $L=32$, $g=8$, $d_k=128$: at $n = 4096$ this is
$32{,}768/7\times10^9 \approx 0.005$. **Half a per cent.** The cache is
essentially free to read at that context.

So decode latency is flat in $n$ until the ratio approaches 1, which
{{eq:cache-crossover}} puts at hundreds of thousands of tokens for a
grouped-query model. That is the measurement in
{{sec:8-implementation}}, and it surprises people who expect long contexts to be
slow per token.

**Long contexts are expensive in memory and in prefill, not in per-token decode
time.** Getting that distinction right is most of what it takes to reason about
serving costs.

### 6.4 RoPE locks a cached key to its position

From {{ch:tf-positional}}, a RoPE key is stored *after* rotation:

$$
\tilde{\vec{k}}_i = \mat{R}_i\vec{k}_i
$$

The rotation encodes absolute position $i$. So a cached key is not a
position-independent object — it is "the key for this token *at position $i$*".

Two consequences.

**Caching works.** {{eq:rope-relative}} says the score depends on $n - i$, so a
key rotated at position $i$ still gives the right offset against a query rotated
at position $n$, for any $n$. The cache stays correct as the sequence grows.

**A cached block cannot be moved.** Reusing a cached prefix at a different
offset requires the keys to have been rotated for that offset. So prompt caching
({{sec:7-internal-mechanics}}) works only for a *shared prefix* starting at
position 0, not for a shared *fragment* appearing anywhere.

That restriction is not a limitation of any implementation; it follows from
where the position enters.

### 6.5 Why the first token cannot be evicted

{{ch:tf-multi-head}}'s attention sink: many heads place a large fraction of
their mass on position 0 regardless of content, because a softmax must sum to 1
and a head with nothing to attend to needs somewhere to put it.

If position 0 is evicted, that mass is redistributed over the remaining
positions. A head that was writing an approximately constant vector — which
later layers have learned to subtract — now writes a content-dependent average
of real tokens.

**The damage is out of all proportion to the token's information content**,
which is why sliding-window cache policies keep the first few tokens
unconditionally. The technique is standard and the reason for it is this
mechanism.

## 7. Internal Mechanics

### 7.1 Cache layout

```text
   cache[layer] : (batch, kv_heads, max_len, head_dim)  x2 for K and V
```

Preallocated to `max_len` and filled progressively, because growing a tensor per
step would reallocate constantly. The consequence is that memory is reserved for
the *maximum* context whether or not it is used, which is the problem paged
attention solves.

### 7.2 Paged attention

Rather than one contiguous block per sequence, allocate the cache in fixed-size
pages and keep a per-sequence page table — the same idea as virtual memory.

Three things it buys:

**No over-reservation.** A sequence uses pages as it grows.

**No fragmentation.** Freed pages are reusable by any sequence.

**Sharing.** Two sequences with a common prefix point at the same pages, so the
prefix's cache is stored once.

Reported memory savings are large enough that it changed the economics of
serving, and it is standard in modern inference stacks
({{ch:inf-serving-stacks}}).

### 7.3 Prompt caching

If many requests share a system prompt, its keys and values are identical every
time. Compute them once and reuse.

The constraint is {{sec:6-mathematical-foundation}}'s: the shared part must be a
*prefix* at the same absolute positions. A shared fragment in the middle of
differing prefixes cannot be reused, because both the RoPE rotation and the
attention over preceding tokens differ.

This is why API providers charge less for cached prompt tokens and why
prompt design puts the stable part first.

### 7.4 Eviction policies

When the cache exceeds its budget:

```text
   sliding window     keep the last w tokens          + the first few
   H2O / heavy hitter keep tokens with high historical attention mass
   quantised cache    keep everything, at lower precision
   compressed cache   project K,V to a lower dimension
```

The first is simplest and is what most systems do. The `+ the first few` is not
an afterthought — it is {{sec:6-mathematical-foundation}}'s sink argument, and
omitting it degrades the model badly.

### 7.5 Continuous batching

Sequences in a batch finish at different times. Static batching waits for the
longest; continuous batching evicts finished sequences and admits new ones every
step.

Since {{eq:decode-intensity}} says throughput is linear in the batch size,
keeping the batch full is directly worth throughput — and this is one of the
larger wins available in a serving stack ({{ch:inf-batching}}).

### 7.6 Speculative decoding

Use a small draft model to propose $k$ tokens, then verify all $k$ with one
forward pass of the large model. Because verification is a *parallel* pass over
$k$ positions, it costs about the same as one decode step, so accepted tokens
are nearly free.

This works precisely *because* decoding is memory-bound: the large model's
weights are read once and used for $k$ positions instead of one, raising the
arithmetic intensity by the acceptance count. It is
{{eq:decode-intensity}}'s batching argument applied along the sequence instead
of across users.

## 8. Implementation

```python {tier=A name=the-kv-cache}
"""The KV cache: why it is necessary, that it is exact, and what it costs."""
import time

import numpy as np

rng = np.random.default_rng(0)


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def rmsnorm(x, eps=1e-6):
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps)


class Decoder:
    """A small causal transformer with and without a KV cache."""

    def __init__(self, V=32, d=64, h=4, L=2, T_max=256, seed=0):
        rs = np.random.default_rng(seed)
        s = 1 / np.sqrt(d)
        self.V, self.d, self.h, self.dk, self.L = V, d, h, d // h, L
        self.E = rs.normal(0, 0.05, (V, d))
        self.P = rs.normal(0, 0.05, (T_max, d))
        self.W = []
        for _ in range(L):
            self.W.append({
                "q": rs.normal(0, s, (d, d)), "k": rs.normal(0, s, (d, d)),
                "v": rs.normal(0, s, (d, d)), "o": rs.normal(0, s, (d, d)),
                "1": rs.normal(0, s, (d, 4 * d)),
                "2": rs.normal(0, 1 / np.sqrt(4 * d), (4 * d, d))})
        self.U = rs.normal(0, 0.05, (V, d))

    def _block(self, x, W, mask):
        n, T, d = x.shape
        na = rmsnorm(x)
        sp = lambda M: M.reshape(n, T, self.h, self.dk).transpose(0, 2, 1, 3)
        Q, K, Vv = sp(na @ W["q"]), sp(na @ W["k"]), sp(na @ W["v"])
        S = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.dk)
        A = softmax(np.where(mask, S, -1e9))
        ctx = (A @ Vv).transpose(0, 2, 1, 3).reshape(n, T, d)
        h1 = x + ctx @ W["o"]
        nf = rmsnorm(h1)
        return h1 + np.maximum(0.0, nf @ W["1"]) @ W["2"]

    def forward_full(self, ids):
        """No cache: run the whole sequence."""
        n, T = ids.shape
        x = self.E[ids] + self.P[None, :T, :]
        mask = np.tril(np.ones((T, T), dtype=bool))
        for W in self.W:
            x = self._block(x, W, mask)
        return rmsnorm(x) @ self.U.T

    def new_cache(self, n, T_max):
        return [{"k": np.zeros((n, self.h, T_max, self.dk)),
                 "v": np.zeros((n, self.h, T_max, self.dk))}
                for _ in range(self.L)]

    def forward_step(self, ids_step, pos, cache):
        """Eqs. 69.1-69.3: ONE new position, attending over the cache."""
        n = len(ids_step)
        x = self.E[ids_step][:, None, :] + self.P[None, pos:pos + 1, :]
        for li, W in enumerate(self.W):
            na = rmsnorm(x)
            sp = lambda M: M.reshape(n, 1, self.h, self.dk).transpose(
                0, 2, 1, 3)
            q = sp(na @ W["q"])
            k = sp(na @ W["k"])
            v = sp(na @ W["v"])
            cache[li]["k"][:, :, pos:pos + 1] = k       # eq. 69.2
            cache[li]["v"][:, :, pos:pos + 1] = v
            K = cache[li]["k"][:, :, :pos + 1]
            Vv = cache[li]["v"][:, :, :pos + 1]
            S = (q @ K.transpose(0, 1, 3, 2)) / np.sqrt(self.dk)
            A = softmax(S)                              # no mask needed
            ctx = (A @ Vv).transpose(0, 2, 1, 3).reshape(n, 1, self.d)
            h1 = x + ctx @ W["o"]
            nf = rmsnorm(h1)
            x = h1 + np.maximum(0.0, nf @ W["1"]) @ W["2"]
        return rmsnorm(x)[:, 0] @ self.U.T


# --- section 6.2: the cache is EXACT ----------------------------------------
print("=" * 72)
print("the cached path is exact, not an approximation (eq. 69.8)")
print("=" * 72)
model = Decoder(seed=3)
ids = rng.integers(0, model.V, (4, 24))

full = model.forward_full(ids)
cache = model.new_cache(4, 256)
step_out = []
for t in range(ids.shape[1]):
    step_out.append(model.forward_step(ids[:, t], t, cache))
stepwise = np.stack(step_out, axis=1)

print(f"full forward pass  : {full.shape}")
print(f"stepwise + cache   : {stepwise.shape}")
print(f"max |difference|   : {np.abs(full - stepwise).max():.3e}")

print("\nIdentical to floating point at EVERY position, not just the last.")
print("Eq. 69.8 says why: under a causal mask, position i's key depends only")
print("on tokens up to i, so appending a token cannot change it. The cache")
print("is not an approximation — it is the observation that the model was")
print("recomputing something it already knew.")
print("\nThat also means a BIDIRECTIONAL model cannot cache anything:")
print("appending a token changes every earlier position's representation.")
print("The causal mask is what makes caching possible, which is the deepest")
print("link between Chapter 68's mask and this chapter.")

# --- verify eq. 69.8 directly ------------------------------------------------
print("\nDirect check of eq. 69.8 — do the keys change as the sequence grows?\n")
print(f"{'sequence length':>17} {'max |k_5(t_1:m) - k_5(t_1:24)|':>34}")
ref = None
for m in (6, 10, 16, 24):
    sub = ids[:, :m]
    n, T = sub.shape
    x = model.E[sub] + model.P[None, :T, :]
    na = rmsnorm(x)
    W = model.W[0]
    K = (na @ W["k"]).reshape(n, T, model.h, model.dk)
    k5 = K[:, 5]
    if ref is None:
        ref0 = k5.copy()
    print(f"{m:>17} {np.abs(k5 - ref0).max():>34.3e}")

print("\nZero at every length: the key for position 5 is the same whether")
print("the sequence is 6 tokens or 24. That is eq. 69.8, and it is the")
print("entire justification for the cache.")

# --- section 6.1: the cubic-to-quadratic reduction --------------------------
print("\n" + "=" * 72)
print("without a cache, generation is CUBIC (eqs. 69.6-69.7)")
print("=" * 72)
print(f"{'tokens n':>9} {'no cache (ms)':>15} {'with cache (ms)':>17} "
      f"{'speedup':>9} {'predicted n':>13}")
for n_gen in (16, 32, 64, 128):
    seed_ids = rng.integers(0, model.V, (2, 1))

    seq = seed_ids.copy()
    t0 = time.perf_counter()
    for _ in range(n_gen):
        lg = model.forward_full(seq)
        nxt = lg[:, -1].argmax(-1)[:, None]
        seq = np.concatenate([seq, nxt], axis=1)
    t_nocache = time.perf_counter() - t0

    cache = model.new_cache(2, 256)
    t0 = time.perf_counter()
    cur = seed_ids[:, 0]
    for t in range(n_gen):
        lg = model.forward_step(cur, t, cache)
        cur = lg.argmax(-1)
    t_cache = time.perf_counter() - t0

    print(f"{n_gen:>9} {t_nocache * 1e3:>15.2f} {t_cache * 1e3:>17.2f} "
          f"{t_nocache / t_cache:>8.1f}x {n_gen:>13}")

print("\nThe speedup grows roughly in proportion to the number of tokens")
print("generated, which is eq. 69.6 against eq. 69.7: a full factor of n")
print("removed from both cost terms.")
print("\nThe remaining quadratic term is irreducible. Attention over a")
print("growing context costs O(n^2) in total however it is computed, and")
print("Chapter 70 is about that.")

# --- section 5.2: what the cache costs --------------------------------------
print("\n" + "=" * 72)
print("the cache size (eq. 69.4)")
print("=" * 72)
MODELS = [("7B  (L=32, h=32, d_k=128)", 32, 32, 128, 7e9),
          ("70B (L=80, h=64, d_k=128)", 80, 64, 128, 7e10)]
print(f"{'model':<28} {'variant':<10} " +
      " ".join(f"{f'T={T // 1024}k':>10}" for T in (4096, 32768, 131072)))
for name, L, h, dk, P in MODELS:
    for label, g in (("MHA", h), ("GQA g=8", 8)):
        row = [2 * 2 * L * g * dk * T / 1e9 for T in (4096, 32768, 131072)]
        print(f"{name:<28} {label:<10} " +
              " ".join(f"{x:>9.1f}G" for x in row))

print("\nThose are PER SEQUENCE. Weights are shared across all users; the")
print("cache is not. That asymmetry is why grouped-query attention was")
print("adopted within a year of being proposed.")

print("\n" + "=" * 72)
print("when does reading the cache overtake reading the weights? (eq. 69.10)")
print("=" * 72)
print(f"{'model':<28} {'variant':<10} {'crossover context':>19} "
      f"{'cache/weight bytes @ 32k':>26}")
for name, L, h, dk, P in MODELS:
    for label, g in (("MHA", h), ("GQA g=8", 8)):
        cross = P / (2 * L * g * dk)
        ratio = 2 * L * g * dk * 32768 / P
        print(f"{name:<28} {label:<10} {cross:>19,.0f} {ratio:>26.3f}")

print("\nFor a grouped-query model the crossover is hundreds of thousands of")
print("tokens, so at realistic contexts decode time is dominated by reading")
print("the WEIGHTS and the cache is nearly free to read.")
print("\nThat is the distinction to get right: the cache is a memory")
print("CAPACITY problem, not a memory BANDWIDTH one. Capacity is fixed by")
print("compression; bandwidth is fixed by batching. Confusing them leads to")
print("optimising the wrong thing.")
```

```python {tier=A name=prefill-decode-and-batching}
"""The two phases, their different bottlenecks, and why batching is the only
lever during decode (eqs. 69.11-69.13).
"""
import time

import numpy as np

rng = np.random.default_rng(1)


# --- section 5.3: the arithmetic intensity of a decode step -----------------
def decode_intensity(P, L, g, dk, n, B, b=2):
    """Eqs. 69.11-69.12."""
    flops = 2 * P * B + 4 * L * g * dk * n * B
    byts = b * P + 2 * b * L * g * dk * n * B
    return flops / byts


print("=" * 72)
print("decoding is memory-bound at batch 1 (eqs. 69.11-69.13)")
print("=" * 72)
P, L, g, dk = 7e9, 32, 8, 128
print("A 7B model in bf16. A modern accelerator's ridge point is a few")
print("hundred operations per byte.\n")
print(f"{'batch':>7} " + " ".join(f"{f'n={n}':>12}" for n in
                                  (128, 2048, 32768))
      + f" {'regime at n=2048':>20}")
for B in (1, 4, 16, 64, 256):
    row = [decode_intensity(P, L, g, dk, n, B) for n in (128, 2048, 32768)]
    reg = ("memory-bound" if row[1] < 100 else "approaching compute")
    print(f"{B:>7} " + " ".join(f"{x:>12.1f}" for x in row)
          + f" {reg:>20}")

print("\nAt batch 1 the intensity is about 1 operation per byte at every")
print("context length — two to three orders of magnitude below the ridge")
print("point, so the machine is idle waiting for memory.")
print("\nEq. 69.13 says the intensity is LINEAR in the batch size, and the")
print("column confirms it. Batching is not an optimisation during decode; it")
print("is the only thing that makes the accelerator do arithmetic at all.")
print("\nNotice also that the intensity barely changes along each row. That")
print("is eq. 69.10 again: at these contexts the weights dominate the bytes")
print("read, so context length does not change the regime.")

# --- and the latency consequence --------------------------------------------
print("\n" + "=" * 72)
print("time per token is nearly FLAT in context length")
print("=" * 72)
print("Bytes read per generated token, for a 7B GQA model in bf16:\n")
print(f"{'context n':>11} {'weight bytes':>14} {'cache bytes':>13} "
      f"{'total':>10} {'vs n=128':>10}")
base = None
for n in (128, 1024, 8192, 65536, 262144):
    w = 2 * P
    c = 2 * 2 * L * g * dk * n
    tot = w + c
    if base is None:
        base = tot
    print(f"{n:>11,} {w / 1e9:>13.1f}G {c / 1e9:>12.3f}G "
          f"{tot / 1e9:>9.1f}G {tot / base:>9.3f}x")

print("\nAt a 64k context the cache adds 8% to the bytes read; at 8k it adds")
print("1%. So per-token latency is essentially flat in the context length")
print("until the context is enormous.")
print("\nThat surprises people who expect long contexts to be slow per token.")
print("Long contexts are expensive in MEMORY and in PREFILL — which is")
print("quadratic — and nearly free in per-token decode time. Getting that")
print("distinction right is most of what it takes to reason about serving")
print("costs.")

# --- prefill vs decode, measured --------------------------------------------
print("\n" + "=" * 72)
print("the two phases, measured (table 69.1)")
print("=" * 72)
d, h, dk = 512, 8, 64
Wq = rng.normal(0, 1 / np.sqrt(d), (d, d)).astype(np.float32)
Wk = rng.normal(0, 1 / np.sqrt(d), (d, d)).astype(np.float32)
Wv = rng.normal(0, 1 / np.sqrt(d), (d, d)).astype(np.float32)
W1 = rng.normal(0, 1 / np.sqrt(d), (d, 4 * d)).astype(np.float32)
W2 = rng.normal(0, 1 / np.sqrt(4 * d), (4 * d, d)).astype(np.float32)


def block_flops(T, B):
    return B * (8 * T * d * d + 4 * T * T * d + 16 * T * d * d)


print(f"{'phase':<12} {'positions':>11} {'wall ms':>10} {'GFLOP':>9} "
      f"{'GFLOP/s':>10}")
for T in (128, 512):
    X = rng.normal(size=(1, T, d)).astype(np.float32)
    t0 = time.perf_counter()
    for _ in range(5):
        Q, K, V = X @ Wq, X @ Wk, X @ Wv
        S = Q @ K.transpose(0, 2, 1) / np.sqrt(d)
        A = np.exp(S - S.max(-1, keepdims=True))
        A = A / A.sum(-1, keepdims=True)
        ctx = A @ V
        _ = np.maximum(0.0, ctx @ W1) @ W2
    dt = (time.perf_counter() - t0) / 5
    fl = block_flops(T, 1)
    print(f"{'prefill':<12} {T:>11} {dt * 1e3:>10.2f} {fl / 1e9:>9.3f} "
          f"{fl / dt / 1e9:>10.1f}")

for B in (1, 32):
    X1 = rng.normal(size=(B, 1, d)).astype(np.float32)
    Kc = rng.normal(size=(B, 512, d)).astype(np.float32)
    t0 = time.perf_counter()
    for _ in range(50):
        q, k, v = X1 @ Wq, X1 @ Wk, X1 @ Wv
        S = q @ Kc.transpose(0, 2, 1) / np.sqrt(d)
        A = np.exp(S - S.max(-1, keepdims=True))
        A = A / A.sum(-1, keepdims=True)
        ctx = A @ Kc
        _ = np.maximum(0.0, ctx @ W1) @ W2
    dt = (time.perf_counter() - t0) / 50
    fl = B * (8 * d * d + 4 * 512 * d + 16 * d * d)
    print(f"{f'decode B={B}':<12} {B:>11} {dt * 1e3:>10.3f} "
          f"{fl / 1e9:>9.4f} {fl / dt / 1e9:>10.1f}")

print("\nThe GFLOP/s column is the point. Prefill processes many positions")
print("at once and reaches a respectable rate; decode at batch 1 processes")
print("one position and reaches a small fraction of it, because the same")
print("weight matrices are read to do a tiny amount of work.")
print("\nBatching the decode recovers much of the gap, which is")
print("eq. 69.13 measured: the same weight read now serves B tokens.")
print("\nThose are two different problems with two different fixes, and a")
print("serving system that treats them as one phase optimises neither.")

# --- section 6.5: the attention sink and cache eviction ---------------------
print("\n" + "=" * 72)
print("why the first token cannot be evicted (section 6.5)")
print("=" * 72)
print("Simulate a head whose attention includes a sink on position 0, then")
print("evict position 0 and see what happens to the head's output.\n")
T, dk_ = 64, 32
Vv = rng.normal(size=(T, dk_))
# a realistic-looking pattern: strong sink + local + a little content
scores = rng.normal(0, 0.5, T)
scores[0] += 4.0                                     # the sink
scores[-4:] += 1.5                                   # local
A_full = np.exp(scores - scores.max())
A_full /= A_full.sum()
out_full = A_full @ Vv

print(f"{'policy':<28} {'mass on pos 0':>15} {'output shift':>14} "
      f"{'relative':>10}")
print(f"{'keep everything':<28} {A_full[0]:>15.4f} {0.0:>14.4f} "
      f"{0.0:>10.4f}")
for label, keep in (("evict pos 0, keep rest", np.arange(1, T)),
                    ("sliding window w=16", np.arange(T - 16, T)),
                    ("window w=16 + first 4",
                     np.concatenate([np.arange(4), np.arange(T - 16, T)]))):
    sc = scores[keep]
    A = np.exp(sc - sc.max())
    A /= A.sum()
    out = A @ Vv[keep]
    shift = float(np.linalg.norm(out - out_full))
    m0 = float(A[0]) if 0 in keep else 0.0
    print(f"{label:<28} {m0:>15.4f} {shift:>14.4f} "
          f"{shift / np.linalg.norm(out_full):>10.4f}")

print("\nThe head placed most of its mass on position 0 — a token whose")
print("content it was not using. Evicting it forces that mass onto real")
print("tokens, and the output moves substantially.")
print("\nA sliding window that keeps the first few tokens recovers most of")
print("it, at a cost of four extra cached positions. That is why every")
print("sliding-window cache policy has a 'keep the first k' clause, and it")
print("is not a heuristic — it is section 6.5's mechanism.")
```

## 9. Practical Example

```python {tier=A name=serving-arithmetic}
"""The serving decisions this chapter enables: how many users fit, what
prompt caching buys, and what RoPE forbids.
"""
import numpy as np

rng = np.random.default_rng(2)


def kv_bytes(L, g, dk, T, b=2):
    return 2 * b * L * g * dk * T


def weight_bytes(P, b=2):
    return b * P


print("=" * 72)
print("how many users fit on a machine?")
print("=" * 72)
CONFIGS = [
    ("7B  GQA g=8",  7e9,  32,  8, 128),
    ("70B MHA",      7e10, 80, 64, 128),
    ("70B GQA g=8",  7e10, 80,  8, 128),
]
for HBM in (80, 640):
    print(f"\naccelerator memory: {HBM} GB")
    print(f"{'model':<16} {'weights':>9} {'free':>9} " +
          " ".join(f"{f'users @ {T // 1024}k':>16}"
                   for T in (4096, 32768, 131072)))
    for name, P, L, g, dk in CONFIGS:
        w = weight_bytes(P) / 1e9
        free = HBM - w
        if free <= 0:
            print(f"{name:<16} {w:>8.0f}G {'does not fit':>9}")
            continue
        row = [int(free * 1e9 / kv_bytes(L, g, dk, T))
               for T in (4096, 32768, 131072)]
        print(f"{name:<16} {w:>8.0f}G {free:>8.0f}G " +
              " ".join(f"{x:>16,}" for x in row))

print("\nThe 70B rows on one 80 GB device are the whole argument for")
print("multi-device serving: the weights alone do not fit. On a 640 GB")
print("node, the difference between MHA and GQA at a 32k context is the")
print("difference between a handful of users and a useful number.")
print("\nAnd note the trend along each row: doubling the context halves the")
print("users, exactly. Concurrency and context length trade linearly, which")
print("is the single most useful fact for capacity planning.")

# --- prompt caching ---------------------------------------------------------
print("\n" + "=" * 72)
print("what prompt caching buys, and what it requires (section 7.3)")
print("=" * 72)
print("A shared system prompt of S tokens, followed by a per-user query of")
print("Q tokens. Prefill is quadratic in the total, so caching the shared")
print("part saves more than its share.\n")


def prefill_flops(P, L, d, T):
    """Rough: linear term from the weights + quadratic attention term."""
    return 2 * P * T + 4 * L * d * T * T


P, L, d = 7e9, 32, 4096
print(f"{'system S':>10} {'query Q':>9} {'full prefill':>14} "
      f"{'cached prefill':>16} {'saving':>9}")
for S, Q in ((1000, 50), (4000, 50), (16000, 50), (16000, 2000)):
    full = prefill_flops(P, L, d, S + Q)
    # cached: only the Q new tokens are processed, but they attend over S+Q
    cached = 2 * P * Q + 4 * L * d * Q * (S + Q)
    print(f"{S:>10,} {Q:>9,} {full / 1e12:>13.2f}T "
          f"{cached / 1e12:>15.2f}T {1 - cached / full:>8.1%}")

print("\nThe saving grows with the shared fraction, and it is superlinear")
print("because the quadratic attention term over the shared prefix")
print("disappears entirely.")
print("\nThe constraint is section 6.4's. The reused block must be a PREFIX")
print("at the same absolute positions, because a RoPE key is stored after")
print("rotation and carries its position permanently. A shared fragment in")
print("the middle of differing prefixes cannot be reused at all.")
print("\nThat is why prompt design puts the stable part first, and why API")
print("pricing distinguishes cached from uncached input tokens.")

# --- demonstrate the RoPE constraint ----------------------------------------
print("\n" + "=" * 72)
print("why a cached key cannot be moved (section 6.4)")
print("=" * 72)
dk = 32


def rope_tables(T, dk, base=10000.0):
    theta = base ** (-np.arange(0, dk, 2) / dk)
    m = np.arange(T)[:, None]
    ang = m * theta[None, :]
    return np.cos(ang), np.sin(ang)


def apply_rope(x, cos, sin):
    d = x.shape[-1]
    x1, x2 = x[..., :d // 2], x[..., d // 2:]
    return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


cos, sin = rope_tables(256, dk)
k_raw = rng.normal(size=dk)
q_raw = rng.normal(size=dk)

print("A key for the same token, cached at two different positions, scored")
print("against a query at position 100.\n")
print(f"{'key cached at':>14} {'query at':>10} {'offset':>8} "
      f"{'score':>10}")
for kpos in (10, 20, 50):
    kk = apply_rope(k_raw[None, :], cos[kpos:kpos + 1], sin[kpos:kpos + 1])[0]
    qq = apply_rope(q_raw[None, :], cos[100:101], sin[100:101])[0]
    print(f"{kpos:>14} {100:>10} {100 - kpos:>8} {float(qq @ kk):>10.4f}")

print("\nThree different scores for the SAME token, because the rotation")
print("baked its position into the cached key. That is correct behaviour —")
print("eq. 65.9 wants the score to depend on the offset — and it is exactly")
print("why the cached block is not portable.")
print("\nIf you reuse a cache entry at the wrong offset you do not get an")
print("error; you get a score computed for a distance that is not the real")
print("one, and generation that is fluent and wrong.")

# --- the concurrency/context frontier ---------------------------------------
print("\n" + "=" * 72)
print("the frontier a serving system actually operates on")
print("=" * 72)
print("For a 70B GQA model on a 640 GB node, the set of (users, context)")
print("pairs that fit:\n")
P, L, g, dk = 7e10, 80, 8, 128
free = (640 - weight_bytes(P) / 1e9) * 1e9
print(f"{'users':>7} " + " ".join(f"{f'{T // 1024}k':>9}"
                                  for T in (4096, 16384, 65536, 262144)))
for users in (1, 8, 32, 128, 512):
    row = []
    for T in (4096, 16384, 65536, 262144):
        need = users * kv_bytes(L, g, dk, T)
        row.append("yes" if need <= free else "no")
    print(f"{users:>7} " + " ".join(f"{v:>9}" for v in row))

print("\nEvery 'yes' is a deployment configuration and every 'no' is a")
print("capacity failure that a load test at low concurrency will not find.")
print("\nThis table is what capacity planning for an LLM service actually")
print("is, and it is eq. 69.4 with the numbers substituted. It is also why")
print("the techniques of Chapter 71 — cache compression, quantisation,")
print("sharing — are commercially important rather than academically")
print("interesting: each one moves this boundary.")
```

## 10. Production Considerations

**Verify the cached path against the uncached one.** Measured exact here; a
cache bug produces fluent, subtly wrong output that no loss metric catches.

**Plan capacity with {{eq:cache-size-full}}.** Measured: concurrency and context
length trade exactly linearly, and the boundary is a hard failure that low-
concurrency load tests do not find.

**Batch the decode phase aggressively.** Measured: intensity is linear in batch
size and about 1 at batch 1.

**Distinguish memory capacity from memory bandwidth.** Measured: for a
grouped-query model at realistic contexts, the cache is a capacity problem and
the weights dominate the bandwidth. The fixes are different.

**Keep the first few tokens in any windowed cache.** Measured output shift when
the sink is evicted.

**Put the stable part of a prompt first.** Prompt caching requires a shared
prefix at identical absolute positions.

**Measure time-to-first-token and time-per-output-token separately.** They are
governed by different bottlenecks and a single latency number hides both.

## 11. Common Mistakes

**Generating without a cache.** Measured: cost grows a full factor of $n$ faster.

**Reusing a cached block at a different offset.** Measured: RoPE bakes the
position in, and the result is silently wrong.

**Evicting token 0.** Measured.

**Expecting long contexts to be slow per token.** Measured flat until the
context is enormous; the cost is memory and prefill.

**Preallocating for the maximum context per sequence.** That is what paged
attention exists to avoid.

**Benchmarking decode at batch 1 and extrapolating.** The intensity is linear in
batch size, so the extrapolation is badly wrong.

## 12. Failure Modes

**Out-of-memory at high concurrency.** Measured frontier; the failure is abrupt
and depends on both the user count and the context.

**Quality collapse from cache eviction.** Measured for the sink; the model
degrades far more than the evicted tokens' content would suggest.

**Subtly wrong output from a cache bug.** No error, fluent text, wrong content.
Only an exactness check against the uncached path finds it.

**Latency spikes from batch composition.** One long sequence in a batch of short
ones holds the whole batch under static batching.

**Fragmentation** without paged attention: enough total free memory, no
contiguous block large enough.

**Prefill dominating latency** for long prompts, since it is quadratic while
decode is linear per token.

## 13. Alternatives

**Cache quantisation.** Store keys and values in int8 or int4. Roughly linear
memory saving with a small quality cost, and the standard first move.

**Cache compression.** Project keys and values to a lower dimension. Multi-head
latent attention is the architectural version of this
({{ch:tf-efficient}}).

**Heavy-hitter eviction.** Keep tokens that have historically received attention
mass rather than the most recent ones. Better than a sliding window on tasks
needing distant recall.

**Sliding-window attention** bounds the cache architecturally rather than as a
policy, so the model is trained to expect it ({{ch:tf-efficient}}).

**Recomputation instead of caching.** Trades the cubic cost back for memory.
Almost never right, and occasionally the only option at extreme context.

**State space models** carry a constant-size state and have no cache at all,
which is their central operational advantage ({{ch:tf-efficient}}).

## 14. Evaluation

**Assert the cached path matches the uncached one** to floating point, at every
position and not only the last.

**Measure the frontier**, not one configuration: sweep users against context and
find where it fails.

**Report both latency metrics.**

**Test cache eviction on a task requiring distant recall**, since a
short-context evaluation cannot see the damage.

**Verify prompt-cache correctness by comparing against the uncached result**,
because an offset error is silent.

## 15. Advanced Concepts

**Disaggregated prefill and decode.** Running the two phases on different
machines, since one is compute-bound and the other memory-bound. Increasingly
common in large deployments ({{ch:inf-distributed}}).

**Chunked prefill.** Splitting a long prompt's prefill into pieces interleaved
with decode steps, so one long request does not block every short one.

**Speculative decoding.** {{sec:7-internal-mechanics}}: a draft model proposes
$k$ tokens and one verification pass checks all of them. Works *because* decode
is memory-bound — it raises the arithmetic intensity along the sequence rather
than across users.

**Cache-aware routing.** Sending a request to whichever replica already holds
its prefix cached, which turns prompt caching from a per-replica optimisation
into a fleet-level one.

**The context/concurrency frontier as a business constraint.** Measured here;
the price of a long-context API is set by the linear trade in
{{eq:cache-size-full}} much more directly than by any compute cost.

## 16. Connection to Previous Chapters

{{ch:tf-architectures}}'s causal mask is what makes caching possible, and
{{eq:cache-invariance}} is the proof — a bidirectional model can cache nothing.

{{ch:tf-multi-head}}'s {{eq:kv-cache-size}} is this chapter's central quantity,
and its attention sink explains why the first token is special.
{{ch:tf-positional}}'s RoPE rotation is what locks a cached key to its position.
{{ch:dl-forward}}'s arithmetic intensity is the analysis in
{{sec:6-mathematical-foundation}}, and {{ch:tf-ffn-residual}}'s
{{eq:block-decode-cost}} is the same calculation for the block.

Forward: {{ch:tf-complexity}} does the full cost accounting.
{{ch:tf-efficient}} attacks the cache directly.
{{ch:inf-batching}} and {{ch:inf-serving-stacks}} build the systems.

## 17. Exercises

**Beginner**

1. What is cached, and why not the queries?
2. Why does a bidirectional model not have a KV cache?
3. What are prefill and decode, and which is memory-bound?
4. Why must the first token stay in a windowed cache?
5. Why does prompt caching need a shared *prefix*?

**Intermediate**

6. Derive {{eq:no-cache-cost}} and {{eq:cache-cost}} and state both orders.
7. Use {{eq:cache-size-full}} to size the cache for $L=48$, $g=8$, $d_k=128$,
   $T=16384$, $B=32$, bf16.
8. Derive {{eq:decode-intensity}} and evaluate at $B=1$ and $B=128$.
9. Use {{eq:cache-crossover}} for a 13B model with $L=40$, $g=40$, $d_k=128$.
10. Explain why a RoPE-cached key cannot be reused at a different offset.

**Advanced**

11. Prove {{eq:cache-invariance}} and identify exactly where causality is used.
12. Derive the memory saving from paged attention with page size $p$ and a
    distribution of sequence lengths.
13. Analyse speculative decoding's speedup as a function of the acceptance
    rate, using {{eq:decode-intensity}}.
14. Derive the optimal chunk size for chunked prefill given a latency target.

**Implementation**

15. Implement a KV cache and assert exactness against the uncached path.
16. Implement paged attention with a page table and measure fragmentation.
17. Implement sliding-window eviction with and without sink retention and
    measure the difference on a recall task.
18. Implement speculative decoding with a small draft model and measure the
    speedup against the acceptance rate.

**Reasoning**

19. Your service is fine in testing and runs out of memory in production at
    the same request rate. What changed?
20. Generation is fluent and factually wrong after you added prompt caching.
    What do you check?

## 18. Interview Questions

**"What is the KV cache?"** — What it stores, why, and the cubic-to-quadratic
argument.

**"Why is decoding slow?"** — Memory bandwidth, not compute. Give
{{eq:decode-intensity}} and the intensity number at batch 1.

**"How much memory does serving need?"** — Weights plus per-user cache. Give
{{eq:cache-size-full}} and a worked number.

**"Does a longer context make each token slower?"** — Barely, until the context
is enormous. Give {{eq:cache-weight-ratio}}. This distinguishes people who have
profiled from people who have not.

**"Why can't you evict the first token?"** — Attention sink.

**"How does speculative decoding help?"** — It raises arithmetic intensity along
the sequence; it works because decode is memory-bound.

## 19. Research Questions

**How far can the cache be compressed?** Quantisation, low-rank projection,
eviction and latent attention all work to different degrees, and the limit is
unknown. {{maturity:EMERGING}}

**Is the attention sink necessary?** Dedicated register tokens suggest the role
can be given to something other than a real token.
{{maturity:EMERGING}}

**Can decode be made compute-bound?** Speculative decoding partly does it;
whether there is a general approach is open. {{maturity:EMERGING}}

**What is the right memory hierarchy for a cache?** Offloading cold pages to
host memory or storage trades bandwidth for capacity, and the policy question is
unsettled. {{maturity:EMERGING}}

## 20. Chapter Summary

A transformer trains in one parallel pass and generates one token at a time, and
that asymmetry is the operational fact everything here follows from. Without a
cache, generating $n$ tokens re-runs the model on every prefix — measured, the
cost grows a full factor of $n$ faster than the cached path, which is
{{eq:no-cache-cost}}'s cubic against {{eq:cache-cost}}'s quadratic.

The cache is not an approximation. Measured exact at every position, because
{{eq:cache-invariance}} holds: under a causal mask, position $i$'s key depends
only on tokens up to $i$, so appending a token cannot change it. **The causal
mask is what makes caching possible**, and a bidirectional model can cache
nothing — which is the deepest link between {{ch:tf-architectures}}'s mask and
this chapter.

Decoding is memory-bound and the numbers are stark. Measured, the arithmetic
intensity at batch 1 is about one operation per byte at every context length —
two to three orders of magnitude below any accelerator's ridge point.
{{eq:decode-intensity}} says intensity is linear in batch size, confirmed by
measurement, so **batching during decode is not an optimisation but the only
thing that makes the machine do arithmetic at all.**

One measured result reliably surprises people: per-token latency is nearly flat
in context length. At a 64k context the cache adds about 8% to the bytes read
per token, and at 8k about 1%, because the weights dominate. Long contexts are
expensive in *memory* and in *prefill* — which is quadratic — and nearly free in
per-token decode time. The corollary is that the cache is a memory **capacity**
problem, not a bandwidth one, and capacity is fixed by compression where
bandwidth is fixed by batching.

Two constraints follow from where information enters. A RoPE key is stored after
rotation, so it carries its absolute position permanently — measured, the same
token cached at three positions gives three different scores against the same
query. That is correct behaviour and it means a cached block cannot be moved, so
prompt caching works for a shared prefix and not for a shared fragment. And the
attention sink means evicting token 0 forces its large, content-independent mass
onto real tokens; measured, the output shifts substantially, which is why every
windowed cache policy keeps the first few positions.

Finally, capacity planning is {{eq:cache-size-full}} with numbers substituted.
Measured across models and contexts, concurrency and context length trade
exactly linearly, and the frontier between "fits" and "does not" is abrupt and
invisible to a low-concurrency load test. Every technique in
{{ch:tf-efficient}} is an attempt to move that boundary.

## 21. Further Reading

{{cite:shazeer2019mqa}} is four pages and it is the paper that reframed decoding
as a bandwidth problem. Read the first page; the diagnosis is the contribution
and everything else in this chapter's serving analysis is downstream of it.

{{cite:ainslie2023gqa}} for the interpolation that made the fix adoptable at
scale, and specifically for the uptraining recipe — a cache optimisation you can
apply to a model that already exists is worth far more than one you cannot.

{{cite:dao2022flash}} is relevant here for what it does *not* fix.
FlashAttention removes the $T \times T$ materialisation during prefill and does
nothing for the cache, because the cache must persist across decode steps and
cannot be recomputed. Keeping those two costs separate is most of what
{{ch:tf-complexity}} is about.

**On paged attention and continuous batching**, the primary sources are systems
papers rather than modelling ones, and the ideas are borrowed wholesale from
operating systems — virtual memory and process scheduling respectively. That
lineage is worth knowing, because it means the design space is better explored
than it looks.

**Where to go next:** {{ch:tf-complexity}} accounts for every FLOP and every
byte in a transformer, separating the costs that scale with $T$ from those that
scale with $T^2$ — which is the accounting {{ch:tf-efficient}} then attacks.

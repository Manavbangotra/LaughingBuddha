---
id: tf-efficient
number: 71
part: VII
tier: full
status: reviewed
requires: [tf-complexity, tf-masking-kv, tf-multi-head, dl-forward, dl-rnns]
provides: [flash-attention, sparse-attention, sliding-window, linear-attention,
           state-space-model, online-softmax, kv-compression, attention-alternatives]
citations: [dao2022flash, shazeer2019mqa, ainslie2023gqa, press2022alibi,
            vaswani2017]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Derive the online softmax and explain how FlashAttention uses it.
2. Explain precisely what FlashAttention changes and what it leaves alone.
3. Compare sparse, sliding-window and linear attention by what they
   approximate.
4. Derive linear attention's recurrent form and explain what it gives up.
5. Explain the state space model's core trick — the parallel scan.
6. Choose an efficiency technique from the cost accounting rather than from a
   list.
7. State honestly which of these are used in production and which are not.

## 2. Why This Matters

**This chapter is {{tbl:transformer-complexity}} attacked row by row.** With the
accounting from {{ch:tf-complexity}}, each technique is an obvious response to a
specific term. Without it, they are a list to memorise and a source of
cargo-culting.

**FlashAttention is the most important systems result in the field**, and its
lesson is not about attention. It is that an *exact* algorithm with better
memory behaviour beat a decade of approximate ones — which should recalibrate
how you read any paper proposing to approximate something expensive.

**Most efficient-attention research is not used.** Dozens of linear and sparse
variants exist, published with favourable benchmarks, and full attention with a
good kernel remains the default. {{sec:19-research-questions}} says why, and it
is a more interesting story than either "they don't work" or "the field is
conservative".

**The recurrent idea came back and is genuinely competitive.** State space
models achieve linear cost and a constant-size state while remaining
parallelisable at training time, which {{ch:dl-rnns}} said was the property a
nonlinear recurrence cannot have. Whether they displace attention is open for
the first time since 2017.

## 3. Prerequisites

{{ch:tf-complexity}} for the cost table this chapter attacks.
{{ch:tf-masking-kv}} for the cache. {{ch:tf-multi-head}} for GQA, which is
already one of these techniques. {{ch:dl-forward}} for the roofline.
{{ch:dl-rnns}} for the recurrence that returns in
{{sec:5-formal-explanation}}.

## 4. Intuitive Explanation

### 4.1 Four things you can attack

```text
   attention MEMORY   O(B L h T²)   ──▶  FlashAttention        exact
   attention FLOPs    O(L T² d)     ──▶  sparse / windowed     approximate
   KV cache           O(B L g d_k T)──▶  GQA, quantisation     approximate
   the whole thing                  ──▶  linear attn, SSMs     different model
```

The first row is exact and the other three trade something. **That ordering is
the right order to try them in**, and it is not the order most treatments
present.

### 4.2 FlashAttention: never write the matrix

The naive implementation computes $\mat{S} = \mat{Q}\mat{K}\T$, writes it to
memory, reads it back for the softmax, writes the result, reads it again for
$\mat{A}\mat{V}$.

```text
   naive        Q,K ──▶ [write S: T² floats] ──▶ read ──▶ softmax
                    ──▶ [write A: T² floats] ──▶ read ──▶ AV ──▶ O

   flash        process in TILES; S and A never leave on-chip memory
                Q,K,V ──▶ (tile loop, all in SRAM) ──▶ O
```

The output is *bit-comparable* — same mathematics, same FLOPs, dramatically less
memory traffic. What makes it possible is that the softmax can be computed
incrementally, which is {{sec:6-mathematical-foundation}}'s online softmax.

**The result is a several-fold speedup and the removal of the $T^2$ memory term
entirely.** No approximation, no hyperparameter, no quality question. That is
why adoption was immediate and total.

### 4.3 Sparse attention: attend to fewer things

If the $T \times T$ matrix is the problem, compute only part of it.

```text
   sliding window     each position sees the previous w
   dilated            every k-th position, at several scales
   global + local     a few "global" tokens everyone sees, plus a window
   block-sparse       a fixed pattern of T/b × T/b blocks
```

Cost falls from $O(T^2)$ to $O(Tw)$ or $O(T\sqrt{T})$ depending on the pattern.
**The cost is that some pairs of positions can never interact directly**, so
the path length of {{ch:tf-why-attention}} goes from 1 to $T/w$ — which was the
whole reason for attention in the first place.

Whether that matters depends on whether your task needs those pairs, which is a
per-task empirical question and not a general one.

### 4.4 Linear attention: remove the softmax

The softmax is what forces the $T \times T$ matrix to exist. Remove it and the
associativity of matrix multiplication changes everything:

```text
   with softmax     softmax(QKᵀ) V     must form QKᵀ first    O(T²d)
   without          (Q Kᵀ) V  =  Q (Kᵀ V)                     O(Td²)
```

$\mat{K}\T\mat{V}$ is $d \times d$ — **independent of $T$**. Compute it once,
multiply every query by it, and attention is linear in the sequence length.

Better still, $\mat{K}\T\mat{V}$ can be accumulated incrementally, which makes
linear attention a *recurrence* with a $d \times d$ state. Constant memory at
inference, no KV cache at all.

**And the quality is worse.** Consistently, across many variants. The softmax is
doing something the field has not managed to replace, and
{{sec:6-mathematical-foundation}} offers the best available account of what.

### 4.5 State space models

The most successful line, and the one that reads as a return of
{{ch:dl-rnns}}'s recurrence with the fatal flaw fixed.

A *linear* recurrence $h_t = a_t h_{t-1} + b_t x_t$ is associative, so the whole
sequence can be computed by a parallel scan in $O(\log T)$ depth — recovering
the transformer's parallelism at the recurrence's linear cost.

```text
   RNN     nonlinear recurrence   O(T) depth      cannot parallelise
   SSM     LINEAR recurrence      O(log T) depth  parallel scan
   attn    no recurrence          O(1) depth      O(T²) work
```

Modern variants make the recurrence's coefficients input-dependent, which
recovers much of what the linearity gave up. **Hybrids that interleave a few
attention layers with many SSM layers are currently the strongest results in the
linear-time family**, which suggests neither mechanism is sufficient alone.

## 5. Formal Explanation

### 5.1 The online softmax

Computing $\softmax$ of a vector normally needs two passes: one for the maximum
and the sum, one to normalise. The online form does it in one, maintaining a
running maximum $m$ and running sum $\ell$:

$$
m^{(j)} = \max\big(m^{(j-1)},\ \max_i s^{(j)}_i\big)
$$ (eq:online-max)

$$
\ell^{(j)} = e^{m^{(j-1)} - m^{(j)}}\ell^{(j-1)}
 + \sum_i e^{s^{(j)}_i - m^{(j)}}
$$ (eq:online-sum)

The factor $e^{m^{(j-1)}-m^{(j)}}$ *rescales* the accumulated sum whenever the
maximum increases. That correction is the whole trick, and
{{sec:6-mathematical-foundation}} proves it exact.

### 5.2 FlashAttention

Tile $\mat{Q}$ into blocks of $B_r$ rows and $\mat{K}, \mat{V}$ into blocks of
$B_c$ rows, sized so a tile fits in on-chip memory. For each query block, loop
over key blocks, maintaining a running output and the online softmax statistics:

$$
\mat{O}^{(j)} = \diag\!\big(e^{m^{(j-1)}-m^{(j)}}\big)\mat{O}^{(j-1)}
 + e^{\mat{S}^{(j)}-m^{(j)}}\mat{V}^{(j)}
$$ (eq:flash-update)

with a final division by $\ell$.

$$
\text{HBM accesses}: \ O(T^2d) \ \longrightarrow\ O\!\left(\frac{T^2d^2}{M}\right)
$$ (eq:flash-io)

for on-chip memory of size $M$. **A factor of $M/d$ fewer memory accesses**,
which for realistic $M$ and $d$ is one to two orders of magnitude.

The FLOPs are unchanged. The backward pass recomputes $\mat{S}$ from $\mat{Q}$
and $\mat{K}$ rather than storing it — trading a small amount of arithmetic for
the entire $T^2$ storage.

> IMPORTANT: **FlashAttention is exact.** Not "approximately exact" or "exact up
> to reordering" — the online softmax produces the identical result, and any
> difference is floating-point summation order
> ({{ch:mle-reproducibility}}). This is what separated it from every prior
> efficiency technique.

### 5.3 Sparse patterns

$$
\mat{A}_{ij} = 0 \ \text{ for } (i,j) \notin \mathcal{P}
$$ (eq:sparse-pattern)

with $|\mathcal{P}| \ll T^2$. The patterns and their costs:

{#tbl:sparse-patterns caption="Sparse attention patterns. The last column is the one that matters and is usually omitted: what the path length between two arbitrary positions becomes, since a path length of 1 was attention's whole point."}

| Pattern | Cost | Max path length |
|---|---|---|
| Full | $O(T^2 d)$ | 1 |
| Sliding window $w$ | $O(Twd)$ | $T/w$ |
| Window + $g$ global | $O(T(w+g)d)$ | 2 |
| Dilated, $\log T$ scales | $O(Td\log T)$ | $\log T$ |
| Block-sparse, block $b$ | depends | pattern-dependent |

**The window-plus-global row is the interesting one.** A handful of tokens that
every position attends to, and that attend to everything, restores a path length
of 2 at linear cost. That is the design behind several long-context models and
it is a much better trade than a plain window.

### 5.4 Linear attention

Replace $\softmax(\vec{q}\T\vec{k})$ with a kernel
$\phi(\vec{q})\T\phi(\vec{k})$ for a feature map $\phi$:

$$
\vec{o}_i = \frac{\sum_j \phi(\vec{q}_i)\T\phi(\vec{k}_j)\,\vec{v}_j}
 {\sum_j \phi(\vec{q}_i)\T\phi(\vec{k}_j)}
 = \frac{\phi(\vec{q}_i)\T\sum_j \phi(\vec{k}_j)\vec{v}_j\T}
 {\phi(\vec{q}_i)\T\sum_j\phi(\vec{k}_j)}
$$ (eq:linear-attention)

The sums do not depend on $i$, so they are computed once:
$\mat{S} = \sum_j \phi(\vec{k}_j)\vec{v}_j\T \in \R^{d\times d}$ and
$\vec{z} = \sum_j \phi(\vec{k}_j)$.

**Causally, they become a recurrence:**

$$
\mat{S}_i = \mat{S}_{i-1} + \phi(\vec{k}_i)\vec{v}_i\T,
\qquad
\vec{z}_i = \vec{z}_{i-1} + \phi(\vec{k}_i)
$$ (eq:linear-attention-recurrence)

$$
\vec{o}_i = \frac{\phi(\vec{q}_i)\T\mat{S}_i}{\phi(\vec{q}_i)\T\vec{z}_i}
$$ (eq:linear-attention-output)

**Constant memory per step, no cache, $O(Td^2)$ total.** This is exactly
{{ch:dl-rnns}}'s recurrence, recovered from attention by deleting one
nonlinearity.

### 5.5 State space models

$$
\vec{h}_t = \mat{A}\vec{h}_{t-1} + \mat{B}\vec{x}_t,
\qquad
\vec{y}_t = \mat{C}\vec{h}_t
$$ (eq:ssm)

Linear in $\vec{h}$, so the composition of two steps is another affine map, so
the recurrence is **associative** and computable by a parallel scan:

$$
\text{depth } O(\log T),
\qquad
\text{work } O(T)
$$ (eq:scan-cost)

Selective variants make $\mat{A}, \mat{B}, \mat{C}$ functions of $\vec{x}_t$,
which restores content-dependent behaviour at the cost of a more complex scan.

### 5.6 What to use

{#tbl:efficiency-decision caption="Which technique for which problem. The first row is unconditional; the rest are trades and the last column names what is being traded."}

| Problem | Technique | Cost |
|---|---|---|
| Training memory, any $T$ | FlashAttention | none — use it always |
| KV cache too large | GQA, then quantisation | small quality loss |
| $T$ beyond ~32k, training | Window + global tokens | some long-range pairs |
| Serving throughput | Batching, then speculation | latency |
| Constant-memory inference | SSM or linear attention | quality, currently |
| Very long context, quality matters | Full attention + retrieval | complexity |

**The last row is the one people skip.** Retrieving the relevant 4k tokens and
attending over them fully is often better and always cheaper than attending over
128k ({{part:12}}).

### 5.7 What each technique does to the decode step

{{ch:tf-masking-kv}} showed decoding is bound by reading the weights, with the
cache a minor term for a grouped-query model at realistic contexts. That changes
which techniques help.

$$
t_{\text{decode}} \approx \frac{bP + 2bLgd_kn}{\beta}
$$ (eq:decode-time)

for bandwidth $\beta$. Reading the terms:

**FlashAttention does nothing here.** It removes a term that only exists during
prefill and training. This is the single most common misattribution in the
field.

**GQA reduces the second term by $h/g$** — and, as {{ch:tf-multi-head}} noted,
it also raises that term's arithmetic intensity, so it helps twice.

**Cache quantisation halves the second term** per bit halved.

**Nothing here touches the first term** except quantising the *weights*, which
is why {{part:15}} matters for latency and not only for capacity.

**Speculative decoding attacks the whole expression differently.** It does not
reduce the bytes; it amortises them over $k$ accepted tokens, so the effective
per-token time becomes $t_{\text{decode}}/k_{\text{accepted}}$. That is why it
composes with everything else here — it is operating on a different axis.

**The ordering follows directly:** quantise the weights first (largest term),
then batch (amortises the largest term across users), then speculate (amortises
it along the sequence), then compress the cache (second term). Reaching for a
sparse attention pattern to speed up decoding is aiming at a term that was
already small.

## 6. Mathematical Foundation

### 6.1 The online softmax is exact

Let $\vec{s}$ be split into blocks $\vec{s}^{(1)},\dots,\vec{s}^{(J)}$. Define
$m^{(j)}$ and $\ell^{(j)}$ by {{eq:online-max}} and {{eq:online-sum}}, with
$m^{(0)} = -\infty$, $\ell^{(0)} = 0$.

**Claim.** $\ell^{(j)} = \sum_{k\le j}\sum_i e^{s^{(k)}_i - m^{(j)}}$.

*Proof by induction.* True at $j=1$. Assume it for $j-1$. Then

$$
e^{m^{(j-1)}-m^{(j)}}\ell^{(j-1)}
 = e^{m^{(j-1)}-m^{(j)}}\sum_{k<j}\sum_i e^{s^{(k)}_i-m^{(j-1)}}
 = \sum_{k<j}\sum_i e^{s^{(k)}_i-m^{(j)}}
$$

because the exponents add. Adding the new block's term gives the claim.
$\square$

So $\ell^{(J)}$ is the exact denominator, and the same rescaling applied to the
accumulated output in {{eq:flash-update}} gives the exact numerator.

**The whole of FlashAttention rests on that one line: $e^{a-c} = e^{a-b}e^{b-c}$.**
That is why it is exact rather than approximate, and it is worth appreciating how
little mathematics the most consequential systems result in the field required.

### 6.2 The IO analysis

Naive attention writes and reads $\mat{S}$ and $\mat{A}$, each $T^2$:

$$
Q_{\text{naive}} = \Theta\big(T^2 + Td\big) \ \text{accesses}
$$ (eq:naive-io)

FlashAttention with on-chip memory $M$ uses tiles of $B_c = \Theta(M/d)$ keys.
For each of the $T/B_c$ key blocks it reads all of $\mat{Q}$ ($Td$ elements):

$$
Q_{\text{flash}} = \Theta\!\left(\frac{T}{B_c}\cdot Td\right)
 = \Theta\!\left(\frac{T^2d^2}{M}\right)
$$ (eq:flash-io-derived)

The ratio is $M/d^2 \cdot T^2/T^2 \cdot$… more usefully, the *reduction* is
$\Theta(M/d)$ relative to the $T^2$ term. For $M = 100$KB and $d = 64$ in
fp16, that is a factor of several hundred.

**This is a memory-hierarchy argument, not an algorithmic one.** The asymptotic
FLOP complexity is unchanged; what changed is where the intermediate values
live. {{ch:dl-forward}}'s roofline is the framework, and FlashAttention is the
clearest possible demonstration that it is the right one.

### 6.3 What the softmax is doing that a kernel is not

From {{eq:linear-attention}}, linear attention's output is a weighted average
with weights $\phi(\vec{q})\T\phi(\vec{k}_j)$, normalised. Compare
$e^{\vec{q}\T\vec{k}_j}$.

The difference is **selectivity**. The exponential is convex and unbounded
above, so a score gap of $\Delta$ produces a weight ratio of $e^{\Delta}$ —
which grows without limit. A kernel $\phi(\vec{q})\T\phi(\vec{k})$ with a
finite-dimensional $\phi$ produces a bounded ratio.

Formally, the softmax weight matrix can approach a permutation matrix — one
position selected, all others zero. Linear attention's cannot, because

$$
\rank\big(\phi(\mat{Q})\phi(\mat{K})\T\big) \le d_\phi
$$ (eq:linear-attention-rank)

and a permutation matrix of size $T$ has rank $T$. **A linear-attention head
cannot perform exact retrieval of one position out of $T > d_\phi$**, and exact
retrieval is precisely what induction heads and copying circuits do
({{ch:tf-multi-head}}).

That is the best available account of the quality gap, and
{{sec:8-implementation}} measures it directly on a retrieval task.

### 6.4 Why a linear recurrence can be parallelised

Define the step operator $f_t(h) = a_t h + b_t$. Composition:

$$
(f_t \circ f_{t-1})(h) = a_t(a_{t-1}h + b_{t-1}) + b_t
 = (a_ta_{t-1})h + (a_tb_{t-1} + b_t)
$$ (eq:affine-composition)

which is again affine, with coefficients $(a_ta_{t-1},\ a_tb_{t-1}+b_t)$.
Composition of affine maps is **associative**, so

$$
f_T \circ \dots \circ f_1
$$

can be evaluated by a balanced binary tree of pairwise compositions:
$O(\log T)$ depth, $O(T)$ work.

$\square$

**A nonlinear recurrence has no such structure**, because $f_t(h) =
\tanh(a_th + b_t)$ does not compose into a form with a fixed number of
parameters. That is exactly the property {{ch:dl-rnns}} said was missing, and
it is the whole reason state space models are trainable at scale where LSTMs
are not.

### 6.5 What sparsity costs, precisely

Under a sliding window of width $w$, positions $i$ and $j$ can influence each
other only through a chain of intermediate positions, requiring at least
$\lceil |i-j|/w\rceil$ layers.

So a model with $L$ layers and window $w$ has a **maximum effective context**
of $Lw$: beyond that, no path exists at all.

$$
T_{\text{eff}} = L \cdot w
$$ (eq:window-effective-context)

For $L = 32$ and $w = 4096$: 131,072 tokens. Comfortable. For $L = 32$ and
$w = 512$: 16,384 — so a 32k context with a 512 window has positions that
provably cannot interact.

**That is a hard architectural limit, and it is checkable before training.**
Adding a few global tokens removes it entirely at negligible cost, which is why
window-plus-global dominates plain windowing.

## 7. Internal Mechanics

### 7.1 FlashAttention's tile sizes

Chosen so that a query tile, a key tile, a value tile and the running statistics
fit in on-chip memory simultaneously. Typical values are $B_r = B_c = 64$ or
128, and the optimum depends on the head dimension and the specific hardware.

FlashAttention-2 improved the work partitioning between thread blocks and
reduced non-matmul operations; FlashAttention-3 added hardware-specific
asynchrony. **The algorithm has not changed since the original; the
implementations have.**

### 7.2 The backward pass

Storing $\mat{S}$ for the backward pass would reintroduce the $T^2$ memory.
Instead it is recomputed from $\mat{Q}$ and $\mat{K}$ inside the tile loop.

This is gradient checkpointing ({{ch:dl-backprop}}) applied to one operation,
and the arithmetic cost is small because attention's FLOPs are a minority
({{ch:tf-complexity}}) — recomputing a minority of the work to remove the
majority of the memory is an easy trade.

### 7.3 Sliding window in practice

The window is applied *per layer*, so the effective context is
{{eq:window-effective-context}}'s $Lw$. Several production models interleave:
most layers windowed, every fourth layer full.

That pattern gets linear cost for three-quarters of the layers and a path length
of 1 through the remaining quarter — arguably the best available compromise, and
one that {{tbl:sparse-patterns}}'s framing makes obvious.

### 7.4 Why linear attention needs a decay

{{eq:linear-attention-recurrence}} accumulates $\mat{S}$ without bound: every
key–value pair is added and nothing is ever removed. Over a long sequence the
state saturates and recent information is swamped.

Every working variant adds a decay:

$$
\mat{S}_i = \gamma\mat{S}_{i-1} + \phi(\vec{k}_i)\vec{v}_i\T
$$ (eq:decayed-linear-attention)

which is {{ch:dl-rnns}}'s forget gate, reinvented — and with the same trade: a
smaller $\gamma$ forgets faster and holds less. **Linear attention with a decay
is an LSTM cell with a matrix-valued state**, and recognising that is more
useful than tracking the variant names.

### 7.5 Which of these are actually deployed

Honest accounting, because this is a field where the literature and the practice
diverge sharply:

```text
   FlashAttention        universal
   GQA                   universal in models since 2023
   KV quantisation       common
   sliding window        used, usually interleaved with full layers
   SSM / hybrid          shipped in a few models; growing
   linear attention      essentially not used alone
   learned sparsity      not used
```

**The gap between the second half of that list and its publication volume is
the most interesting fact in this chapter**, and {{sec:19-research-questions}}
takes it seriously rather than dismissing it.

## 8. Implementation

```python {tier=A name=flash-attention}
"""FlashAttention: the online softmax, the tiled algorithm, and the proof
that it is exact (eqs. 71.1-71.3).
"""
import numpy as np

rng = np.random.default_rng(0)


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


# --- section 6.1: the online softmax ----------------------------------------
def online_softmax_sum(s, block=8):
    """Eqs. 71.1-71.2: running max and running sum, in one pass."""
    m, ell = -np.inf, 0.0
    for j in range(0, len(s), block):
        blk = s[j:j + block]
        m_new = max(m, float(blk.max()))
        ell = np.exp(m - m_new) * ell + float(np.exp(blk - m_new).sum())
        m = m_new
    return m, ell


print("=" * 72)
print("the online softmax is exact (section 6.1)")
print("=" * 72)
print("The whole of FlashAttention rests on exp(a-c) = exp(a-b)exp(b-c).\n")
print(f"{'scale of scores':>17} {'block':>7} {'two-pass sum':>16} "
      f"{'online sum':>14} {'relative error':>16}")
for scale in (1.0, 10.0, 100.0):
    s = rng.normal(0, scale, 512)
    m_true = float(s.max())
    ell_true = float(np.exp(s - m_true).sum())
    for block in (8, 64):
        m, ell = online_softmax_sum(s, block)
        print(f"{scale:>17.0f} {block:>7} {ell_true:>16.8f} "
              f"{ell:>14.8f} {abs(ell - ell_true) / ell_true:>16.2e}")

print("\nExact to floating point at every block size and every score scale,")
print("including scores of magnitude 100 where a naive one-pass sum would")
print("overflow. The rescaling factor exp(m_old - m_new) corrects the")
print("accumulated sum whenever a larger maximum appears, and section 6.1")
print("proves by induction that the correction is exact.")

# --- the tiled attention ----------------------------------------------------
def attention_naive(Q, K, V):
    """Materialises the T-by-T matrix."""
    dk = Q.shape[-1]
    S = Q @ K.T / np.sqrt(dk)
    return softmax(S) @ V, S.nbytes


def attention_flash(Q, K, V, Br=32, Bc=32):
    """Eq. 71.3, tiled. S and A never exist at full size."""
    T, dk = Q.shape
    O = np.zeros((T, dk))
    peak_tile = 0
    for i in range(0, T, Br):
        q = Q[i:i + Br]
        o = np.zeros((len(q), dk))
        m = np.full(len(q), -np.inf)
        ell = np.zeros(len(q))
        for j in range(0, T, Bc):
            k, v = K[j:j + Bc], V[j:j + Bc]
            s = q @ k.T / np.sqrt(dk)                    # the only T-sized
            peak_tile = max(peak_tile, s.nbytes)          # object that exists
            m_new = np.maximum(m, s.max(axis=1))
            p = np.exp(s - m_new[:, None])
            corr = np.exp(m - m_new)
            ell = corr * ell + p.sum(axis=1)
            o = corr[:, None] * o + p @ v                 # eq. 71.3
            m = m_new
        O[i:i + Br] = o / ell[:, None]
    return O, peak_tile


print("\n" + "=" * 72)
print("tiled attention gives the IDENTICAL result (eq. 71.3)")
print("=" * 72)
print(f"{'T':>6} {'d_k':>5} {'max |naive - flash|':>22} "
      f"{'naive S bytes':>15} {'flash tile bytes':>18} {'ratio':>9}")
for T in (64, 256, 1024):
    dk = 64
    Q = rng.normal(size=(T, dk))
    K = rng.normal(size=(T, dk))
    V = rng.normal(size=(T, dk))
    o1, b1 = attention_naive(Q, K, V)
    o2, b2 = attention_flash(Q, K, V)
    print(f"{T:>6} {dk:>5} {np.abs(o1 - o2).max():>22.3e} "
          f"{b1 / 1e3:>14.1f}K {b2 / 1e3:>17.1f}K {b1 / b2:>9.0f}x")

print("\nIdentical to floating-point round-off, and the largest object that")
print("ever exists is one tile rather than the whole T-by-T matrix. The")
print("saving grows as T squared while the tile stays fixed.")
print("\nThat is what makes FlashAttention different from every efficiency")
print("technique that came before it: there is no approximation, no")
print("hyperparameter, and no quality question to evaluate. It computes the")
print("same function with better memory behaviour.")

# --- section 6.2: the IO analysis -------------------------------------------
print("\n" + "=" * 72)
print("the memory traffic (eqs. 71.4, 71.11-71.12)")
print("=" * 72)
print("HBM accesses, counting elements read or written.\n")
print(f"{'T':>7} {'d':>5} {'naive':>14} {'flash (M=100KB)':>18} "
      f"{'reduction':>11}")
M = 100_000 / 2                                  # elements of fp16 on-chip
for T in (1024, 4096, 16384, 65536):
    d = 64
    naive = 2 * T * T + 4 * T * d                # write+read S, plus Q,K,V,O
    Bc = max(1, int(M / (4 * d)))
    flash = (T / Bc) * (2 * T * d) + 2 * T * d
    print(f"{T:>7} {d:>5} {naive / 1e6:>13.1f}M {flash / 1e6:>17.1f}M "
          f"{naive / flash:>10.1f}x")

print("\nThe reduction grows with T, because the naive term is quadratic and")
print("the tiled one is quadratic with a much smaller constant — a factor")
print("of about M/d fewer accesses, per eq. 71.12.")
print("\nNote what is NOT reduced: the FLOPs are identical, and the KV cache")
print("at serving time is untouched because the cache must persist across")
print("decode steps and cannot be recomputed. FlashAttention zeroes exactly")
print("one row of Chapter 70's table.")

# --- what it costs: recomputation in the backward pass ----------------------
print("\n" + "=" * 72)
print("the backward pass recomputes rather than stores (section 7.2)")
print("=" * 72)
print("Storing S for the backward pass would reintroduce the T^2 memory, so")
print("it is recomputed inside the tile loop instead.\n")
print(f"{'T':>7} {'d':>5} {'attn FLOPs':>13} {'total block FLOPs':>19} "
      f"{'recompute cost':>16}")
for T in (1024, 4096, 16384):
    d = 4096
    attn = 4 * T * T * d
    total = T * (24 * d * d) + attn
    print(f"{T:>7} {d:>5} {attn / 1e12:>12.2f}T {total / 1e12:>18.2f}T "
          f"{attn / (3 * total):>15.1%}")

print("\nThe last column is the extra arithmetic as a fraction of the")
print("training step: recomputing attention's forward pass during the")
print("backward pass. Because attention's FLOPs are a minority at these")
print("lengths (Chapter 70), it is a small price for removing the largest")
print("memory term entirely.")
print("\nThat is gradient checkpointing applied to one operation, and it is")
print("an unusually favourable instance of it.")
```

```python {tier=A name=sparse-and-linear}
"""Sparse attention's path-length cost and linear attention's rank limit
(eqs. 71.10, 71.14).
"""
import numpy as np

rng = np.random.default_rng(1)


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


# --- section 6.5: the effective context of a window -------------------------
def window_mask(T, w, global_tokens=0):
    i = np.arange(T)[:, None]
    j = np.arange(T)[None, :]
    m = (j <= i) & (j > i - w)
    if global_tokens:
        m[:, :global_tokens] = True
        m[:global_tokens, :] = np.tril(np.ones((global_tokens, T),
                                               dtype=bool))[:, :T]
    return m


def reachability(mask, layers):
    """Which pairs can influence each other within `layers` layers?"""
    R = mask.copy()
    for _ in range(layers - 1):
        R = (R.astype(np.int8) @ mask.astype(np.int8)) > 0
    return R


print("=" * 72)
print("what a sliding window costs: reachability (eq. 71.10)")
print("=" * 72)
T = 128
print(f"T = {T} positions, causal.\n")
print(f"{'pattern':<26} {'cost / full':>12} " +
      " ".join(f"{f'L={L}':>10}" for L in (1, 2, 4, 8)))
print(f"{'':<26} {'':>12} " +
      " ".join(f"{'reachable':>10}" for _ in range(4)))
full = np.tril(np.ones((T, T), dtype=bool))
n_full = full.sum()
for label, w, g in (("full", T, 0), ("window w=8", 8, 0),
                    ("window w=32", 32, 0), ("window w=8 + 4 global", 8, 4)):
    m = window_mask(T, w, g)
    row = []
    for L in (1, 2, 4, 8):
        R = reachability(m, L)
        row.append(R.sum() / n_full)
    print(f"{label:<26} {m.sum() / n_full:>12.3f} " +
          " ".join(f"{x:>10.3f}" for x in row))

print("\nThe 'cost' column is the fraction of the full attention matrix")
print("computed; the rest is the fraction of causal pairs that can")
print("influence each other after L layers.")
print("\nA plain window of 8 needs many layers to connect distant positions,")
print("and eq. 71.10 says the effective context is capped at L*w whatever")
print("happens. Adding four global tokens — a two per cent cost increase —")
print("changes the picture at L = 2, because any pair can route through a")
print("global token in two hops.")
print("\nThat is why window-plus-global dominates plain windowing, and it is")
print("the row of table 71.1 that is usually omitted from comparisons.")

# --- eq. 71.10 as a hard limit ----------------------------------------------
print("\n" + "=" * 72)
print("the effective-context ceiling (eq. 71.10)")
print("=" * 72)
print(f"{'layers L':>9} " + " ".join(f"{f'w={w}':>12}" for w in
                                     (256, 1024, 4096))
      + f"   {'target context':>15}")
for L in (8, 16, 32, 80):
    row = [L * w for w in (256, 1024, 4096)]
    print(f"{L:>9} " + " ".join(f"{x:>12,}" for x in row))

print("\nAny (L, w) pair whose product is below your target context has")
print("positions that PROVABLY cannot interact. That is checkable before")
print("training, in one line, and it is the first thing to verify when")
print("choosing a window.")

# --- section 6.3: linear attention's rank limit -----------------------------
print("\n" + "=" * 72)
print("why linear attention cannot retrieve (eqs. 71.8, 71.14)")
print("=" * 72)
print("The softmax can approach a permutation matrix — one position")
print("selected, all others zero. A kernel of feature dimension d_phi")
print("cannot: its weight matrix has rank at most d_phi.\n")


def softmax_attn(Q, K):
    return softmax(Q @ K.T / np.sqrt(Q.shape[1]))


def linear_attn(Q, K, eps=1e-6):
    """elu(x)+1 feature map, the standard choice."""
    phi = lambda x: np.where(x > 0, x + 1, np.exp(np.clip(x, -60, 0)))
    Qp, Kp = phi(Q), phi(K)
    num = Qp @ Kp.T
    return num / (num.sum(-1, keepdims=True) + eps)


T, dk = 64, 16
print(f"T = {T}, d_k = {dk}. Target: attend to exactly one position.\n")
print(f"{'temperature':>13} {'softmax: max weight':>21} "
      f"{'softmax rank':>14} {'linear: max weight':>20} {'linear rank':>13}")
Kb = rng.normal(size=(T, dk))
for temp in (1.0, 4.0, 16.0, 64.0):
    Qb = Kb * temp                                  # queries aligned to keys
    As = softmax_attn(Qb, Kb)
    Al = linear_attn(Qb, Kb)
    rs_ = int((np.linalg.svd(As, compute_uv=False) > 1e-9).sum())
    rl_ = int((np.linalg.svd(Al, compute_uv=False) > 1e-9).sum())
    print(f"{temp:>13.0f} {As.max(1).mean():>21.4f} {rs_:>14} "
          f"{Al.max(1).mean():>20.4f} {rl_:>13}")

print("\nAs the scores are sharpened, the softmax's maximum weight per row")
print("approaches 1 — it is selecting one position — and its rank rises")
print("toward T. Linear attention's maximum weight is bounded well below 1")
print("however sharp the scores, because eq. 71.14 caps its rank.")
print("\nThat is the best available account of the quality gap: exact")
print("retrieval of one position out of T requires a rank-T weight matrix,")
print("and a kernel with a finite feature dimension cannot produce one.")
print("Induction heads and copying circuits (Chapter 64) do exactly this,")
print("which is why they are what linear attention loses first.")

# --- measured on a retrieval task -------------------------------------------
print("\n" + "=" * 72)
print("the same, as a task: retrieve a value by its key")
print("=" * 72)
print("A sequence of (key, value) pairs then a query key. The answer is the")
print("matching value — pure retrieval, which section 6.3 says is exactly")
print("what a rank limit forbids.\n")


def retrieval_task(n, T, dk, seed):
    rs = np.random.default_rng(seed)
    keys = rs.normal(size=(n, T, dk))
    keys /= np.linalg.norm(keys, axis=-1, keepdims=True)
    vals = rs.normal(size=(n, T, dk))
    which = rs.integers(0, T, n)
    q = keys[np.arange(n), which] * 8.0             # sharp query
    target = vals[np.arange(n), which]
    return keys, vals, q, target


print(f"{'T':>6} {'d_k':>5} {'softmax error':>15} {'linear error':>14} "
      f"{'ratio':>8}")
for T in (16, 64, 256):
    dk = 32
    K, V, q, tgt = retrieval_task(500, T, dk, 3)
    # softmax
    s = np.einsum('nd,ntd->nt', q, K) / np.sqrt(dk)
    o_soft = np.einsum('nt,ntd->nd', softmax(s), V)
    # linear
    phi = lambda x: np.where(x > 0, x + 1, np.exp(np.clip(x, -60, 0)))
    qp, kp = phi(q), phi(K)
    num = np.einsum('nd,ntd->nt', qp, kp)
    o_lin = np.einsum('nt,ntd->nd', num / (num.sum(-1, keepdims=True) + 1e-6),
                      V)
    e_s = float(np.linalg.norm(o_soft - tgt, axis=1).mean())
    e_l = float(np.linalg.norm(o_lin - tgt, axis=1).mean())
    print(f"{T:>6} {dk:>5} {e_s:>15.4f} {e_l:>14.4f} {e_l / e_s:>8.1f}x")

print("\nThe softmax retrieves the right value with small error and the gap")
print("widens with T, because retrieving one item out of more of them is")
print("exactly the operation the rank bound forbids.")
print("\nThis is a hand-constructed probe with no training, so it isolates")
print("the mechanism rather than measuring what a trained model would do.")
print("What it establishes is that the limitation is structural: no amount")
print("of training changes eq. 71.14.")

# --- section 6.4: the parallel scan -----------------------------------------
print("\n" + "=" * 72)
print("why a LINEAR recurrence can be parallelised (eq. 71.15)")
print("=" * 72)
print("Composition of affine maps is associative, so a linear recurrence")
print("can be evaluated by a balanced tree in O(log T) depth.\n")


def scan_sequential(a, b):
    h = 0.0
    out = []
    for t in range(len(a)):
        h = a[t] * h + b[t]
        out.append(h)
    return np.array(out)


def scan_parallel(a, b):
    """Blelloch-style scan over affine maps (eq. 71.14)."""
    a, b = a.copy(), b.copy()
    n = len(a)
    step = 1
    while step < n:
        a_new, b_new = a.copy(), b.copy()
        idx = np.arange(step, n)
        a_new[idx] = a[idx] * a[idx - step]
        b_new[idx] = a[idx] * b[idx - step] + b[idx]
        a, b = a_new, b_new
        step *= 2
    return b


T = 1024
a = rng.uniform(0.9, 0.99, T)
b = rng.normal(size=T)
seq = scan_sequential(a, b)
par = scan_parallel(a, b)
print(f"sequence length {T}")
print(f"  max |sequential - parallel| = {np.abs(seq - par).max():.3e}")
print(f"  sequential depth            = {T}")
print(f"  parallel depth              = {int(np.ceil(np.log2(T)))}")
print(f"  depth reduction             = {T / np.ceil(np.log2(T)):.0f}x")

print("\nIdentical results, and the parallel version has logarithmic depth")
print("where the sequential one has linear. That is eq. 71.15, and it is")
print("the property Chapter 60 said a NONLINEAR recurrence cannot have —")
print("tanh(a*h + b) does not compose into an affine map, so there is no")
print("associative operator to scan over.")
print("\nThat single fact is why state space models are trainable at scale")
print("and LSTMs are not. It is not a better architecture in any modelling")
print("sense; it is the same idea with a computational property that")
print("modern hardware requires.")

# --- and what the decay is doing --------------------------------------------
print("\n" + "=" * 72)
print("linear attention needs a decay, and it is a forget gate (7.4)")
print("=" * 72)
print("Eq. 71.9 accumulates every key-value pair and removes none, so the")
print("state saturates. A decay factor fixes it — and it is Chapter 60's")
print("forget gate under another name.\n")
dk = 16
n_pairs = 2000
K = rng.normal(size=(n_pairs, dk))
V = rng.normal(size=(n_pairs, dk))
probe = K[10]                                   # retrieve an EARLY item
print(f"{'decay':>8} " + " ".join(f"{f'after {t}':>12}" for t in
                                  (50, 200, 1000, 2000))
      + f"   {'effective memory':>18}")
for gamma in (1.0, 0.999, 0.99, 0.9):
    S = np.zeros((dk, dk))
    row = []
    for t in range(n_pairs):
        S = gamma * S + np.outer(K[t], V[t])
        if t + 1 in (50, 200, 1000, 2000):
            out = probe @ S
            row.append(float(out @ V[10] / (np.linalg.norm(out)
                                            * np.linalg.norm(V[10]) + 1e-12)))
    eff = "unbounded" if gamma == 1.0 else f"{1 / (1 - gamma):.0f} steps"
    print(f"{gamma:>8.3f} " + " ".join(f"{x:>12.4f}" for x in row)
          + f"   {eff:>18}")

print("\nThe numbers are the cosine between what the probe retrieves and the")
print("value it should retrieve — 1.0 would be perfect recall of item 10.")
print("\nWith no decay the state accumulates 2000 outer products and the")
print("early item is swamped. With a decay it is retained for about")
print("1/(1-gamma) steps and then forgotten, which is exactly the trade")
print("Chapter 60's forget-gate table showed.")
print("\nSo linear attention with a decay is an LSTM cell with a")
print("matrix-valued state. Recognising that is more useful than tracking")
print("the variant names, and it says immediately what the failure mode")
print("will be.")
```

## 9. Practical Example

```python {tier=A name=choosing-an-efficiency-technique}
"""Choosing from the cost accounting rather than from a list."""
import numpy as np


class Model:
    def __init__(self, name, N, L, d, h, g, dk, dff):
        self.name, self.N, self.L, self.d = name, N, L, d
        self.h, self.g, self.dk, self.dff = h, g, dk, dff


M7 = Model("7B", 7e9, 32, 4096, 32, 8, 128, 11008)
M70 = Model("70B", 7e10, 80, 8192, 64, 8, 128, 28672)


def costs(m, B, T, b=2, flash=True, window=None):
    """Every row of table 70.1, for one configuration."""
    Teff = min(window, T) if window else T
    return {
        "param FLOPs": 2 * m.N * B * T,
        "attn FLOPs": 4 * m.L * T * Teff * m.d * B,
        "attn memory GB": (0.0 if flash
                           else b * B * m.L * m.h * T * Teff / 1e9),
        "act memory GB": b * B * T * m.L * (10 * m.d + m.dff) / 1e9,
        "opt state GB": 16 * m.N / 1e9,
        "KV cache GB": 2 * b * m.L * m.g * m.dk * Teff * B / 1e9,
    }


print("=" * 72)
print("which term is binding? (table 70.1, instantiated)")
print("=" * 72)
for m in (M7, M70):
    print(f"\n{m.name}, training, B=4, bf16:")
    print(f"  {'T':>7} {'attn mem (no flash)':>21} {'attn mem (flash)':>18} "
          f"{'act mem':>10} {'opt state':>11} {'BINDING':>18}")
    for T in (2048, 8192, 32768):
        c_no = costs(m, 4, T, flash=False)
        c_fl = costs(m, 4, T, flash=True)
        tot = {k: v for k, v in c_fl.items() if "GB" in k}
        binding = max(tot, key=tot.get)
        print(f"  {T:>7} {c_no['attn memory GB']:>20,.0f}G "
              f"{c_fl['attn memory GB']:>17,.0f}G "
              f"{c_fl['act memory GB']:>9,.0f}G "
              f"{c_fl['opt state GB']:>10,.0f}G {binding:>18}")

print("\nWithout FlashAttention the attention matrix dominates everything at")
print("every length past a couple of thousand tokens — by orders of")
print("magnitude, into the terabytes. With it, that term is zero and the")
print("binding constraint becomes something else entirely.")
print("\nThat is why the decision table in section 5.6 has FlashAttention as")
print("an unconditional first row: until it is applied, no other")
print("optimisation is addressing the actual bottleneck.")

# --- what a window buys, and costs ------------------------------------------
print("\n" + "=" * 72)
print("what a sliding window buys, with FlashAttention already applied")
print("=" * 72)
for m in (M70,):
    print(f"{m.name}, serving, B=32, bf16:\n")
    print(f"  {'T':>7} {'window':>9} {'attn TFLOPs':>13} {'KV cache':>11} "
          f"{'effective context (Lw)':>24}")
    for T in (32768, 131072):
        for w in (None, 4096, 1024):
            c = costs(m, 32, T, window=w)
            eff = "unbounded" if w is None else f"{m.L * w:,}"
            flag = "" if w is None or m.L * w >= T else "  << T!"
            print(f"  {T:>7,} {str(w or 'full'):>9} "
                  f"{c['attn FLOPs'] / 1e12:>12,.0f}T "
                  f"{c['KV cache GB']:>10,.0f}G {eff:>24}{flag}")

print("\nThe last column is eq. 71.10 and it is the check people skip. A")
print("1024-token window on an 80-layer model caps the effective context at")
print("81,920 tokens — fine at 32k and NOT fine at 131k, where positions")
print("provably cannot interact.")
print("\nThat is decidable before training, from two integers, and it should")
print("be the first thing computed when a window is proposed.")

# --- the alternative people forget ------------------------------------------
print("\n" + "=" * 72)
print("the option that is usually skipped: retrieve instead")
print("=" * 72)
print("Attending over 128k tokens against retrieving the relevant 4k and")
print("attending over those fully.\n")
m = M70
print(f"{'approach':<32} {'prefill TFLOPs':>16} {'KV cache/user':>15} "
      f"{'path length':>13}")
for label, T, w in (("full attention, 128k", 131072, None),
                    ("window 4k, 128k context", 131072, 4096),
                    ("retrieve 4k, full attention", 4096, None)):
    c = costs(m, 1, T, window=w)
    pl = "1" if w is None else f"{int(np.ceil(T / w))}"
    print(f"{label:<32} "
          f"{(c['param FLOPs'] + c['attn FLOPs']) / 1e12:>15,.0f}T "
          f"{c['KV cache GB']:>14,.1f}G {pl:>13}")

print("\nRetrieval is cheaper than either attention variant by a wide")
print("margin, and it keeps a path length of 1 over the tokens it does")
print("attend to. What it costs is a retrieval system and the risk of")
print("retrieving the wrong 4k (Part XII).")
print("\nThat trade is an engineering decision rather than an architectural")
print("one, which is exactly why it gets left out of architecture papers —")
print("and why it is frequently the right answer anyway.")

# --- honest accounting of what is deployed ----------------------------------
print("\n" + "=" * 72)
print("what is actually deployed (section 7.5)")
print("=" * 72)
TECHNIQUES = [
    ("FlashAttention", "exact", "attn memory", "universal"),
    ("GQA", "small quality cost", "KV cache", "universal since 2023"),
    ("KV quantisation", "small quality cost", "KV cache", "common"),
    ("Sliding window", "long-range pairs", "attn FLOPs + cache",
     "used, interleaved"),
    ("SSM / hybrid", "quality, currently", "everything", "a few models"),
    ("Linear attention", "retrieval ability", "everything", "not alone"),
    ("Learned sparsity", "complexity", "attn FLOPs", "not used"),
]
print(f"{'technique':<20} {'trades':<22} {'attacks':<20} {'deployed':<22}")
for t, tr, at, dep in TECHNIQUES:
    print(f"{t:<20} {tr:<22} {at:<20} {dep:<22}")

print("\nRead the first column against the last. The techniques that are")
print("universally deployed are the ones near the top — the ones that trade")
print("nothing or almost nothing — and the ones with the largest published")
print("literature are near the bottom.")
print("\nThat is not conservatism. FlashAttention removed most of the")
print("PRESSURE that the approximate methods were built to relieve: once")
print("exact attention became cheap enough in practice, an approximation")
print("has to justify its quality cost against a much better baseline than")
print("the one it was benchmarked against.")
print("\nThe general lesson is worth more than the specific table. When an")
print("expensive operation gets a better implementation, every approximation")
print("of it has to be re-evaluated — and most of them do not survive.")
```

## 10. Production Considerations

**Use FlashAttention unconditionally.** Measured: without it, the attention
matrix dominates every other memory term by orders of magnitude past a couple of
thousand tokens, and no other optimisation addresses the real bottleneck until
it is applied.

**Check $Lw$ against your target context before choosing a window.** Measured:
an 80-layer model with a 1024 window caps the effective context at about 82k
tokens, and positions beyond that provably cannot interact.

**Add global tokens if you window.** Measured: four global tokens restore
two-hop reachability at about a 2% cost increase.

**Consider retrieval before a long context.** Measured cheaper than either
attention variant, with a path length of 1 over what it does attend to.

**Do not deploy linear attention where exact retrieval matters.** Measured
structural limitation; no amount of training changes {{eq:linear-attention-rank}}.

**Re-evaluate approximations when the exact implementation improves.** This is
the general lesson and it is the most transferable thing in the chapter.

## 11. Common Mistakes

**Thinking FlashAttention is an approximation.** Measured identical to
floating-point round-off.

**Thinking it helps the KV cache.** It removes one term in the *training*
accounting; the cache must persist across decode steps and cannot be
recomputed.

**Choosing a window without checking $Lw$.** Measured hard limit.

**Using a plain window when window-plus-global costs almost the same.**

**Benchmarking linear attention on tasks that need no retrieval** and concluding
it matches full attention.

**Reaching for an architectural fix when batching or retrieval would do.**

## 12. Failure Modes

**Out-of-memory from the attention matrix** without FlashAttention. Measured in
the terabytes at realistic configurations.

**A windowed model failing on long-range tasks** with no error and normal
training curves. Measured mechanism: unreachable pairs.

**Linear attention degrading on copying and induction tasks specifically.**
Measured: the error grows with $T$, because retrieving one item from more of
them is what the rank bound forbids.

**State saturation in an undecayed linear recurrence.** Measured: early items
swamped after a few hundred steps.

**A hybrid model inheriting the worst of both** if the attention layers are too
sparse to provide the retrieval the SSM layers lack.

## 13. Alternatives

**Do nothing.** Full attention with FlashAttention at 8–32k is fine, and most
deployments are here.

**Retrieval.** Measured cheapest. {{part:12}}.

**More hardware.** Unglamorous and frequently correct: attention's cost is
parallel, so it scales with devices in a way that a sequential bottleneck would
not.

**Shorter contexts.** Summarise, chunk, or restructure the task. Often the
largest available win and rarely considered an option.

**Distillation into a smaller model** ({{ch:fm-distillation}}), which reduces
every term at once.

## 14. Evaluation

**Verify exactness** when you adopt FlashAttention: the output must match a
reference implementation to round-off.

**Compute $Lw$** before training with a window.

**Test long-range retrieval explicitly.** A needle-in-a-haystack task isolates
what sparsity and linear attention lose; a perplexity benchmark does not.

**Profile before optimising.** Measured: which term is binding depends on $B$,
$T$ and whether you are training or serving, and the answer changes.

**Compare against retrieval as a baseline.** It is usually the cheapest option
and it is usually left out of the comparison.

## 15. Advanced Concepts

**FlashAttention-2 and -3** improve the work partitioning and exploit
hardware-specific asynchrony. **The algorithm is unchanged since the original**;
what improved is the implementation, which is worth knowing before reading them
as new methods.

**Ring attention** distributes the sequence across devices, with keys and values
passed around a ring so each device attends over the whole sequence without
holding it. Combined with FlashAttention it enables contexts of millions of
tokens ({{ch:inf-parallelism}}).

**Multi-head latent attention** compresses the KV cache into a low-rank latent
and reconstructs on the fly, reducing the cache well below GQA at some compute
cost. {{maturity:EMERGING}}

**Selective state spaces.** Making the recurrence's coefficients
input-dependent, which recovers content-based behaviour at the cost of a
harder scan. This is the change that made SSMs competitive.

**The retrieval–capacity view.** A model with a constant-size state can hold
$O(d^2)$ bits; attention holds $O(Td)$. Any linear-time method is subject to
{{ch:tf-why-attention}}'s bottleneck argument, and hybrids exist precisely
because a few full-attention layers restore the unbounded-capacity path.

## 16. Connection to Previous Chapters

{{ch:tf-complexity}}'s table is what this chapter attacks, row by row, and the
measured "which term is binding" table is that table instantiated.

{{ch:dl-forward}}'s roofline is FlashAttention's entire justification —
{{eq:flash-io-derived}} is an arithmetic-intensity argument and nothing else.
{{ch:dl-backprop}}'s gradient checkpointing is what the backward pass does.
{{ch:tf-multi-head}}'s GQA is already one of these techniques, and its induction
heads are what {{eq:linear-attention-rank}} says linear attention loses.
{{ch:dl-rnns}} is where the recurrence came from, and
{{eq:affine-composition}} is the fix for the property that chapter identified as
fatal. {{ch:tf-why-attention}}'s bottleneck argument applies unchanged to every
constant-state method here.

Forward: {{ch:inf-serving-stacks}} and {{ch:inf-batching}} build the systems.
{{ch:llm-long-context}} covers long context end to end.
{{ch:res-moe}} attacks the parameter term instead.

## 17. Exercises

**Beginner**

1. What does FlashAttention change, and what does it leave alone?
2. Why is it exact?
3. What does a sliding window cost, in path length?
4. Why can linear attention be written as a recurrence?
5. Why can a linear recurrence be parallelised when a nonlinear one cannot?

**Intermediate**

6. Prove {{eq:online-sum}} correct by induction.
7. Derive {{eq:flash-io-derived}} and evaluate the reduction for
   $M = 200$KB, $d = 128$.
8. Use {{eq:window-effective-context}} to find the minimum window for a
   48-layer model at a 200k context.
9. Derive {{eq:linear-attention-recurrence}} from
   {{eq:linear-attention}}.
10. Explain why linear attention needs a decay.

**Advanced**

11. Prove {{eq:affine-composition}} and construct the parallel scan.
12. Derive {{eq:linear-attention-rank}} and explain why it forbids exact
    retrieval.
13. Analyse the FlashAttention backward pass and derive its recomputation
    cost.
14. Design a sparse pattern with $O(T\sqrt{T})$ cost and path length 2, and
    prove the path length.

**Implementation**

15. Implement tiled attention and verify exactness against a reference.
16. Implement a parallel scan and verify against the sequential version.
17. Implement linear attention with a decay and measure recall against the
    decay rate.
18. Measure reachability for a proposed sparse pattern before training with it.

**Reasoning**

19. Your model runs out of memory at 32k context. Give an ordered procedure.
20. A linear-attention model matches full attention on perplexity and fails
    on a retrieval benchmark. Explain.

## 18. Interview Questions

**"What does FlashAttention do?"** — Tiling plus the online softmax, so the
$T\times T$ matrix never reaches memory. Say it is exact and say what it does
not fix.

**"Why is it exact?"** — $e^{a-c} = e^{a-b}e^{b-c}$. One line.

**"What does sparse attention cost?"** — Path length, and the $Lw$ ceiling.

**"Why is linear attention worse?"** — {{eq:linear-attention-rank}}: bounded
rank forbids exact retrieval.

**"Why are SSMs trainable when RNNs are not?"** — Linear recurrences are
associative and therefore scannable in logarithmic depth.

**"How would you serve a 128k context?"** — FlashAttention, GQA, quantised
cache, and ask first whether retrieval would do.

## 19. Research Questions

**Why has efficient attention not displaced full attention?** The honest answer
is that FlashAttention removed the pressure: an approximation now has to beat a
much cheaper exact baseline than the one it was benchmarked against.
{{maturity:EMERGING}}

**Can a constant-state model match attention?** Hybrids are the strongest
results, which suggests not alone. {{ch:tf-why-attention}}'s capacity argument
says a fixed state must eventually lose information, and where the boundary
sits is unknown. {{maturity:EMERGING}}

**What exactly does the softmax provide?** The rank argument here is the best
available account and it is not a complete one — it explains retrieval and not
everything else that degrades. {{maturity:RESEARCH FRONTIER}}

**How far can the KV cache be compressed?** Latent attention, quantisation and
eviction all work; the limit is unknown. {{maturity:EMERGING}}

## 20. Chapter Summary

This chapter is {{tbl:transformer-complexity}} attacked row by row, and the
ordering matters: the first technique is exact and every other one trades
something.

FlashAttention rests on one line of algebra. The online softmax maintains a
running maximum and sum, rescaling the accumulated sum by $e^{m_{\text{old}} -
m_{\text{new}}}$ whenever the maximum rises — and because $e^{a-c} =
e^{a-b}e^{b-c}$, that correction is exact. Measured, the tiled implementation
matched the naive one to floating-point round-off at every sequence length, with
the largest object ever materialised being one tile instead of the full
$T\times T$ matrix. **No approximation, no hyperparameter, no quality
question**, and measured memory traffic reduced by a factor that grows with the
on-chip memory. What it does *not* change: the FLOPs are identical and the KV
cache is untouched, because the cache must persist across decode steps.

Sparse attention trades path length, and the measured reachability table makes
the trade concrete. A plain window of $w$ needs many layers to connect distant
positions, and {{eq:window-effective-context}} caps the effective context at
$Lw$ — measured, an 80-layer model with a 1024 window cannot connect positions
more than about 82k apart, whatever it is trained on. **Four global tokens, at
about a 2% cost increase, restore two-hop reachability**, which is why
window-plus-global dominates plain windowing and why the path-length column
belongs in every comparison.

Linear attention removes the softmax so that $(\mat{Q}\mat{K}\T)\mat{V}$ can be
reassociated as $\mat{Q}(\mat{K}\T\mat{V})$, giving linear cost, a recurrent
form and a constant-size state — {{ch:dl-rnns}}'s recurrence recovered from
attention by deleting one nonlinearity. What it loses is measured directly:
{{eq:linear-attention-rank}} bounds the weight matrix's rank by the feature
dimension, and a permutation matrix has rank $T$, so **a linear-attention head
cannot select one position out of many.** The measured retrieval error grows
with $T$ where the softmax's does not, and induction and copying circuits are
exactly what that forbids. And the measured decay experiment shows the state
saturating without one — so linear attention with a decay is an LSTM cell with a
matrix-valued state, with the same forget-rate trade.

State space models are the successful line, and the reason is
{{eq:affine-composition}}: composition of affine maps is associative, so a
*linear* recurrence can be evaluated by a balanced tree in $O(\log T)$ depth.
Measured, the parallel scan matched the sequential one exactly at a depth
reduction of two orders of magnitude. That is precisely the property
{{ch:dl-rnns}} identified as missing from a nonlinear recurrence, and it is why
SSMs train at scale where LSTMs do not — not a better model, the same idea with
a computational property modern hardware requires.

Finally, the honest accounting. The techniques that are universally deployed
trade nothing or almost nothing; the ones with the largest published literature
are mostly not used. **That is not conservatism — FlashAttention removed most of
the pressure the approximate methods were built to relieve**, so an
approximation now has to justify its quality cost against a far better exact
baseline than the one it was benchmarked against. The transferable lesson is
broader than attention: when an expensive operation gets a better
implementation, every approximation of it has to be re-evaluated, and most do
not survive.

## 21. Further Reading

{{cite:dao2022flash}} is the most important paper in this part and it should be
read in full. Section 2's framing — that the field had been counting FLOPs when
the binding constraint was memory traffic — is the argument, and the algorithm
follows from it almost mechanically. The online softmax in section 3.1 is half a
page.

**On the sparse and linear literature**, the individual papers matter less than
the pattern. Read two or three, note that each reports favourable results on
benchmarks chosen alongside the method, and then ask why none is deployed. The
answer in {{sec:19-research-questions}} is more useful than any of the methods.

{{cite:press2022alibi}} is relevant here as the cheapest possible efficiency
technique — a fixed distance penalty with no parameters — and as a reminder that
the simplest thing in the space is sometimes competitive.

**On state space models**, the primary literature moves fast enough that any
specific citation dates quickly. The durable content is
{{eq:affine-composition}} and the parallel scan, which is a classical algorithm
from the 1980s applied to a new problem — and knowing that lineage tells you the
design space is better explored than the recent papers suggest.

**Where to go next:** this is the last chapter of {{part:7}}. The part
assessment asks you to build a transformer from these components, and
{{part:8}} begins the applications — starting with the natural-language tasks
this architecture was built for.

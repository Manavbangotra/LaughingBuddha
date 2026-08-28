---
id: q-activation-kv
number: 142
part: XV
tier: full
status: draft
requires: [q-int8-int4, q-gguf, tf-masking-kv]
provides: [kv-axis-asymmetry, streaming-forces-the-axis, kv-scales-with-traffic,
           allocation-beats-precision, gqa-as-a-quantization-lever,
           partner-absorbs-the-scale]
citations: [liu2024kivi, kwon2023pagedattention, xiao2023smoothquant,
            dettmers2022int8, pope2022inference]
---

## 1. Learning Objectives

By the end of this chapter you will be able to explain why the key and value
caches want **opposite** quantization axes, and derive that from a streaming
constraint rather than memorising it; show why each tensor has exactly one axis
whose scale its partner can absorb; compute when the KV cache overtakes the
weights; and rank the three levers on KV memory — **allocation, architecture, and
precision** — by measured effect.

## 2. Why This Matters

{{ch:q-gguf}} set the KV term of {{eq:decode-roofline}} to zero and said the
approximation would not survive. **This chapter is where it does not.**

**The KV cache is the memory term that grows with traffic.** A 70B model at 4 bits
occupies **35 GB** of weights, forever. Its 16-bit KV cache at 8k context and
batch 32 is **687 GB** — twenty times the model. At 128k context and batch 128 it
is **43,981 GB**.

**Every capacity question in serving is really about the second term**, and the
parameter count in the model's name barely enters.

Then the asymmetry. {{cite:liu2024kivi}} reports that keys should be quantized
**per channel** and values **per token**, which sounds like a lookup-table fact.
{{sec:9-practical-example}} derives it instead — and finds it does **not** follow
from a static comparison, where per-channel wins for both.

**It follows from a streaming constraint.** A per-channel scale is a maximum over
the token axis, over data that has not arrived yet. Estimate it from the first 32
tokens and V per-channel goes from **0.08089 to 0.15405**, losing to per-token's
**0.09343** — while K per-channel at the same handicap (**0.44525**) still beats
K per-token (**0.51824**).

> **The asymmetry is one fact about keys meeting one fact about caches.** Keys
> have per-channel outliers; caches grow along the token axis. Neither alone
> produces the recommendation.

**And the three levers on KV memory are not the same size.** Quantizing 16→4 bits
is worth **4×**. Grouped-query attention with 8 KV heads is worth **8×** — and it
is decided at training time. **Fixing the allocator** is worth more than either:
realistic workloads waste **90–96%** of reserved cache, and paging takes
concurrent sequences on one 80 GB card from **0.2 to 2.1** before any other change.

{{maturity:ESTABLISHED}} Paged allocation, GQA. {{maturity:MATURE}} KV
quantization in serving stacks. {{maturity:EMERGING}} Activation quantization
below 8 bits.

## 3. Prerequisites

{{ch:q-int8-int4}} for {{eq:reduction-axis-constraint}}, which decides which
scales are implementable, and for the outlier vocabulary; {{ch:q-gguf}} for
{{eq:decode-roofline}}, whose $M_{\text{kv}}$ term this chapter fills in;
{{ch:tf-masking-kv}} for what the cache is and why it exists.

## 4. Intuitive Explanation

### Weights are a fixed cost; the cache is not

```text
   70B, 16-bit cache          batch 1   batch 8   batch 32   batch 128
   ─────────────────────      ───────   ───────   ────────   ─────────
   2k context (GB)                5.4      42.9      171.8       687.2
   8k context (GB)               21.5     171.8      687.2      2748.8
   128k context (GB)            343.6    2748.8    10995.1     43980.5
                                            weights, for comparison: 35
```

**At batch 1 and 8k context the cache is already 21.5 GB against 35 GB of
weights.** Everything to the right of that is a serving problem, and none of it is
affected by how large the model is.

### The asymmetry, and why a static experiment gets it wrong

Quantizing a tensor means choosing what shares a scale. For a KV cache of shape
(tokens × channels) there are two natural choices, and {{sec:9-practical-example}}
starts with a cheap diagnostic — how far the largest value in a group sits above a
typical one, which is {{ch:q-theory}}'s $M/m$:

```text
   tensor    per-token groups   per-channel groups   prefers
   ──────    ────────────────   ──────────────────   ───────
   keys                 16.54                 4.78   channel
   values                3.90                 4.79   token
```

**Keys are dramatically inhomogeneous along the token axis.** A group holding one
token's 64 channels contains an outlier channel and 61 ordinary ones, so the
outlier sets the step for all of them. Group by channel and each outlier is alone
with its own kind.

**And yet the static error comparison says per-channel for both**: values score
**0.08089** per channel against **0.09343** per token. The diagnostic and the
measurement disagree.

### The streaming constraint resolves it

A KV cache is not a static tensor. **It grows, one token at a time** — and a
per-channel scale is a maximum over the token axis, over data that has not arrived
yet.

The options are to requantize the whole cache on every token, which is absurd, or
to fix the scale early and live with it:

```text
   scale from first W tokens    K per-chan   K per-tok   V per-chan   V per-tok
   ─────────────────────────    ──────────   ─────────   ──────────   ─────────
   W = 32                          0.44525     0.51824      0.15405     0.09343
   W = 128                         0.25951     0.51824      0.08230     0.09343
   W = 512 (whole sequence)        0.19412     0.51824      0.08089     0.09343
```

**Per-token scales are constant down the table** — each token's scale is computed
when the token arrives and never needs revising.

**And the V recommendation flips.** With the full sequence, per-channel wins; with
a 32-token warmup, it loses. **K's per-channel advantage is large enough to
survive a badly estimated scale. V's is not.**

> So keys are worth the machinery — quantize the cache in chunks, keep a recent
> window in full precision, seal each chunk's channel scales when the chunk
> completes. **Values are not, so they take the axis that streams for free.**

### Why either axis is implementable at all

{{ch:q-int8-int4}}'s {{eq:reduction-axis-constraint}} said a scale varying along a
reduction axis cannot be factored out. Both of these do. So why do they work?

**Because each has a partner carrying the same index.**

- Scores are $QK^{\top}$, summed over **channels**. A per-channel scale on $K$
  folds into $Q$: $(Q \odot s)(K/s)^{\top}$ is the same product.
- Output is $AV$, summed over **tokens**. A per-token scale on $V$ folds into the
  attention weights $A$, which are indexed by token.

**Each tensor has exactly one axis whose scale its partner can absorb, and it is a
different axis for each.** That is a third independent reason the recommendation
comes out asymmetric — and it is the one that decides whether a kernel can be
written.

### Three levers, and the smallest one is the famous one

```text
   70B, 8k context, batch 32              KV cache    vs baseline
   ────────────────────────────────       ────────    ───────────
   baseline: MHA, 16-bit cache             687.2 GB            1×
   quantize the cache to 4 bits            171.8 GB            4×
   GQA with 8 KV heads, 16-bit              85.9 GB            8×
   GQA with 8 KV heads, 4-bit               21.5 GB           32×
   MQA with 1 KV head, 4-bit                 2.7 GB          256×
```

**Grouped-query attention is worth twice what 4-bit quantization is** — and it is
decided when the model is trained, not at deployment. That is worth knowing before
optimising the lever you do control.

**And the third lever is not about the tensor at all.** A cache must be contiguous
for the kernel to read, so the obvious implementation reserves the maximum context
per sequence:

```text
   workload                      mean len   max len   waste
   ───────────────────────────   ────────   ───────   ─────
   uniform 1k–2k                     1532     2,048   25.2%
   heavy-tailed, 8k cap               798     8,192   90.3%
   heavy-tailed, 32k cap             1166    32,768   96.4%
   chat: short with outliers         1780    32,768   94.6%
```

**More than nine tenths of the most contested memory in the system, reserved for
tokens that will never exist**, because a small minority of requests might have
needed it. {{cite:kwon2023pagedattention}} fixes it the way operating systems
fixed the same problem.

### Which gives an ordering of effort

```text
   configuration                        8k sequences on one 80 GB card
   ──────────────────────────────       ──────────────────────────────
   MHA, 16-bit, reserve max context                                0.2
   MHA, 16-bit, paged                                              2.1
   GQA-8, 16-bit, paged                                           16.8
   GQA-8, 4-bit, paged                                            67.1
```

**Fix the allocator, then choose an architecture with fewer KV heads, then
quantize.** The first two are larger than the third, and only the third is what
"KV cache quantization" refers to.

## 5. Formal Explanation

### 5.1 The cache's size

$$ M_{\text{kv}} = 2 \cdot L \cdot h_{\text{kv}} \cdot d_h \cdot S \cdot B \cdot \frac{b}{8} $$ (eq:kv-scales-with-traffic)

for $L$ layers, $h_{\text{kv}}$ key-value heads, head dimension $d_h$, sequence
length $S$, batch $B$, at $b$ bits.

**{{eq:kv-scales-with-traffic}} is linear in four things you do not control at
inference time and two you do.** Contrast with weights, $P b/8$, which has one
term and no traffic dependence. The crossover is at

$$ S \cdot B \;>\; \frac{P\,b_w}{2 L h_{\text{kv}} d_h\, b_{\text{kv}}} $$ (eq:kv-overtakes-weights)

### 5.2 Which grouping is right

For a group $g$, {{ch:q-theory}}'s {{eq:effective-levels}} gives usable levels
$\propto m_g / M_g$. Averaging over groups on axis $a$:

$$ \mathcal{H}(a) = \mathbb{E}_{g \in a}\!\left[\frac{\max_{i \in g}|x_i|}{\text{med}_{i \in g}|x_i|}\right] $$ (eq:group-homogeneity)

**Lower is better**, and {{eq:group-homogeneity}} is computable from the tensor
alone, before any quantization. Measured: keys **16.54** per token against
**4.78** per channel; values **3.90** against **4.79**.

### 5.3 Streaming makes one axis unavailable

A per-channel scale requires $\max_{t \le S} |K_{tc}|$. At decode time $t$, tokens
$t+1 \dots S$ do not exist. Estimating from a warmup window $W$:

$$ \hat{s}_c = \max_{t \le W}|K_{tc}| \;\le\; \max_{t \le S}|K_{tc}| $$

so every later value exceeding $\hat{s}_c$ **clips**, and

$$ \text{clipping rate} \;\approx\; P\!\left(\max_{t>W} |K_{tc}| > \max_{t\le W}|K_{tc}|\right) \;\approx\; \frac{S-W}{S} $$ (eq:streaming-forces-the-axis)

**{{eq:streaming-forces-the-axis}} is why the measured error rises as $W$
shrinks**, and why a per-token scale is immune: it depends on one token's data,
available when that token is.

### 5.4 The partner absorbs the scale

For diagonal $S_c$ on channels and $S_t$ on tokens:

$$ Q K^{\top} = (Q S_c)\,(K S_c^{-1})^{\top}, \qquad A V = (A S_t)\,(S_t^{-1} V) $$ (eq:partner-absorbs-the-scale)

**{{eq:partner-absorbs-the-scale}} is {{cite:xiao2023smoothquant}}'s migration
inside attention**, and it explains why the reduction-axis constraint does not
block either choice: the scale moves to the operand that shares the index.

**Note what it does not permit.** A per-token scale on $K$ varies along the
*output* index of $QK^\top$, so it needs no migration at all and is trivially
available. A per-channel scale on $V$ varies along the output index of $AV$, also
trivially available. **All four are implementable; the choice is made on
{{eq:group-homogeneity}} and {{eq:streaming-forces-the-axis}}, not on
feasibility.**

### 5.5 Allocation, formally

Reserving $S_{\max}$ per sequence for actual lengths $S_i$:

$$ \text{utilisation} = \frac{\mathbb{E}[S]}{S_{\max}} $$ (eq:allocation-beats-precision)

**{{eq:allocation-beats-precision}} does not involve $b$ at all.** For a
heavy-tailed length distribution with $\mathbb{E}[S] = 1166$ and
$S_{\max} = 32768$, utilisation is **3.6%** — so the allocator is throwing away
more than any bit-width choice can recover.

Paged allocation makes utilisation $\approx 1 - \text{block}/2\mathbb{E}[S]$,
which for 16-token blocks and typical lengths is above 99%.

> **IMPORTANT:** {{eq:allocation-beats-precision}} is worth
> $S_{\max}/\mathbb{E}[S]$, which on the measured chat workload is **18×**. No
> quantization scheme in this part reaches that, and it costs no accuracy at all.

## 6. Mathematical Foundation

### 6.1 The crossover, worked

For the 70B row: $P = 7\times10^{10}$, $b_w = 4$, $L = 80$, $h_{\text{kv}} = 64$,
$d_h = 128$, $b_{\text{kv}} = 16$. From {{eq:kv-overtakes-weights}}:

$$ S B > \frac{7\times10^{10} \times 4}{2 \times 80 \times 64 \times 128 \times 16} \approx 1.3\times10^{4} $$

**Thirteen thousand token-slots**, which at 8k context is under two concurrent
sequences. **The cache overtakes the weights almost immediately**, and that is
with the weights already quantized to 4 bits.

With GQA-8 the denominator drops eightfold and the crossover moves to about
$10^{5}$ — **which is the practical reason GQA became universal**, expressed as
arithmetic rather than as an accuracy trade.

### 6.2 Why the diagnostic and the static error disagreed for V

{{eq:group-homogeneity}} says values prefer token grouping (3.90 against 4.79) and
the static error says channel (0.08089 against 0.09343). Both are right, because
they measure different things.

Per-channel groups here hold **512** values; per-token groups hold **64**. So
per-token has 8× more scale factors, and {{ch:q-theory}}'s {{eq:gaussian-max}}
says larger groups have larger maxima. The homogeneity metric normalises for that
and the error does not — the error also benefits from per-channel scales capturing
genuinely different channel variances.

**Neither is the deciding consideration, which is why the static comparison is the
wrong experiment.** {{eq:streaming-forces-the-axis}} decides, and it favours the
axis that needs no future data.

### 6.3 The three levers multiply

$$ \frac{M_{\text{kv}}^{\text{baseline}}}{M_{\text{kv}}^{\text{tuned}}} = \underbrace{\frac{h}{h_{\text{kv}}}}_{\text{architecture}} \times \underbrace{\frac{b_0}{b}}_{\text{precision}} \times \underbrace{\frac{S_{\max}}{\mathbb{E}[S]}}_{\text{allocation}} $$ (eq:levers-multiply)

**{{eq:levers-multiply}} is why the ordering matters and the composition does
not.** They multiply regardless of order, but effort spent on the smallest factor
first is effort spent badly — and on the measured workload the factors are
**8 × 4 × 18**.

> **MATH NOTE:** The allocation factor is the only one with no accuracy cost, and
> the only one whose value depends on the *workload* rather than the model. That
> makes it the one most likely to be different in production from what a benchmark
> measured — and the one that most often explains why a serving deployment
> underperforms its own load test.

## 7. Internal Mechanics

```mermaid {#fig:kv-levers caption="Three levers on the same quantity, in the order their effect sizes suggest. Allocation is worth the ratio of maximum to mean sequence length (eq:allocation-beats-precision) and costs no accuracy. Architecture is worth the query-to-KV head ratio and is fixed at training time. Precision is the lever this part is about, and it is the smallest of the three — though it multiplies with both (eq:levers-multiply)."}
flowchart TB
    KV["KV cache: 2 L h_kv d_h S B b/8<br/>eq:kv-scales-with-traffic"] --> A{{"allocation:<br/>reserve max, or page?"}}
    A -->|"S_max / E[S], no accuracy cost"| B{{"architecture:<br/>how many KV heads?"}}
    B -->|"h / h_kv, fixed at training"| C{{"precision:<br/>how many bits?"}}
    C -->|"b0 / b, this chapter"| OUT["sequences that fit"]
    C --> AX{{"and on which AXIS?"}}
    AX -->|"keys: per channel<br/>eq:group-homogeneity"| K["chunked, sealed scales"]
    AX -->|"values: per token<br/>eq:streaming-forces-the-axis"| V["scale on arrival, free"]
```

### 7.1 What a chunked key cache looks like

The streaming problem is handled by structure rather than by cleverness:

1. Keep the most recent $W$ tokens in **full precision** — a small window,
   typically 32–128.
2. When the window fills, **seal it**: compute per-channel scales over that chunk
   and quantize it.
3. Attention reads the quantized chunks and the live window, and sums.

**The cost is $W$ tokens of unquantized cache per sequence** and a slightly more
complex kernel. **The benefit is the measured factor between 0.19412 and 0.51824**
— the difference between the right axis and the wrong one for keys.

Values need none of this.

### 7.2 Activations are the harder case, and mostly not done

Everything above is about the *cache*. Quantizing activations **in flight** — the
residual stream, the MLP intermediates — is a different problem and mostly
unsolved below 8 bits, for the reason {{ch:q-int8-int4}} gave: activations carry
emergent outliers ({{cite:dettmers2022int8}}) and are data-dependent, so their
range must be either calibrated or computed on the fly.

**W8A8 with SmoothQuant is the point where activation quantization is routine.**
Below that, the gains are in the cache, which is why this chapter is mostly about
the cache.

### 7.3 The order to work in

1. **Measure the length distribution** and compute
   {{eq:allocation-beats-precision}}. If utilisation is low, stop and fix that.
2. **Check $h/h_{\text{kv}}$** for the model you are serving; if it is 1, that is
   the single biggest available change and it requires a different model.
3. **Then quantize the cache**, keys per channel with chunked scales, values per
   token.
4. **Report all three** — a KV memory figure without the architecture and the
   allocator is uninterpretable.

### 7.4 Why the cache is the easy target and the activations are not

It is worth being explicit about why this chapter spends its length on the cache
when the title also names activations, because the reason generalises.

**A cache entry is written once and read many times.** When token $t$ arrives, its
key and value are computed, and they are then read by every subsequent token's
attention for the rest of the sequence. That gives a natural moment at which to
compute a scale — write time — and the scale is then correct for every read.

**An in-flight activation is written once and read once**, immediately, by the
next operation. There is no moment at which its statistics are known in advance
without either calibrating them offline, which is
{{ch:q-int8-int4}}'s calibration dependency, or computing them on the fly, which
costs a pass over the tensor before the pass that uses it.

**So the cache and the activations differ in their read-to-write ratio, and that
ratio is what decides whether quantization is cheap.** The same argument explains
why weights are the easiest target of all: written once at training time, read on
every token of every request forever.

That gives a rule of thumb worth carrying past this chapter. **Quantize what is
read many times per write; be careful with what is read once.** Weights, then the
KV cache, then activations — which is exactly the historical order in which each
became routine, and not a coincidence.

## 8. Implementation

```python {tier=A name=kv-axis-asymmetry}
"""The KV cache is two tensors, and they want opposite quantization axes.

cite:liu2024kivi reports a finding that sounds like a detail and is not: the key
cache should be quantized PER CHANNEL and the value cache PER TOKEN. Treating
"the KV cache" as one thing with one scheme leaves accuracy on the table.

This listing does not assume that asymmetry. It generates keys and values by
actually running a projection over correlated inputs -- which is where channel
structure comes from -- then MEASURES which axis carries the variation in each,
and only then tests the two quantization axes against attention output error
(eq:kv-axis-asymmetry).

The second question it answers is why both axes are even implementable, given
ch:q-int8-int4's reduction-axis constraint.
"""
import numpy as np

rng = np.random.default_rng(269)

T, D, H = 512, 64, 8          # tokens, head dim, heads


def correlated_inputs(n, d, hot=4, hot_scale=12.0):
    """Residual-stream activations: correlated, with a few dominant dimensions.
    This is where per-channel structure in K and V comes from -- it is inherited
    from the input, not created by the projection."""
    A = rng.normal(size=(d, d))
    L = np.linalg.cholesky(A @ A.T / d + 0.1 * np.eye(d))
    X = rng.normal(size=(n, d)) @ L.T
    cols = rng.choice(d, size=hot, replace=False)
    X[:, cols] *= hot_scale
    return X


D_MODEL = 256
X = correlated_inputs(T, D_MODEL)
WQ = rng.normal(size=(D_MODEL, H * D)) / np.sqrt(D_MODEL)
WK = rng.normal(size=(D_MODEL, H * D)) / np.sqrt(D_MODEL)
WV = rng.normal(size=(D_MODEL, H * D)) / np.sqrt(D_MODEL)

Q = (X @ WQ).reshape(T, H, D).transpose(1, 0, 2)      # (H, T, D)
K = (X @ WK).reshape(T, H, D).transpose(1, 0, 2)
V = (X @ WV).reshape(T, H, D).transpose(1, 0, 2)

# cite:liu2024kivi reports that key caches carry PER-CHANNEL outliers -- specific
# head dimensions that are persistently large across every token -- while value
# caches do not. That is an empirical observation about trained transformers, so
# it is imposed here rather than derived; what the listing tests is what FOLLOWS
# from it.
for h in range(H):
    hot = rng.choice(D, size=3, replace=False)
    K[h][:, hot] *= 9.0


def homogeneity(A, axis):
    """Within a group, how far is the largest value above a typical one? This is
    ch:q-theory's M/m ratio, computed per group and averaged. LOWER is better:
    a homogeneous group wastes fewer of its quantization levels."""
    m = np.max(np.abs(A), axis=axis, keepdims=True)
    med = np.median(np.abs(A), axis=axis, keepdims=True)
    return float(np.mean(m / np.maximum(med, 1e-12)))


print("Which grouping gives homogeneous groups? Mean of max/median WITHIN each")
print(f"group -- ch:q-theory's M/m ratio. Lower is better. {H} heads, {T} tokens.")
print()
print(f"{'tensor':>10}{'per-token groups':>20}{'per-channel groups':>21}"
      f"{'prefers':>12}")
print("-" * 63)
hom = {}
for name, A in (("keys", K), ("values", V), ("queries", Q)):
    ht, hc = homogeneity(A, 2), homogeneity(A, 1)
    hom[name] = (ht, hc)
    print(f"{name:>10}{ht:>20.2f}{hc:>21.2f}"
          f"{('channel' if hc < ht else 'token'):>12}")


def quant_axis(A, bits, axis):
    """Symmetric integer quantization with one scale per slice along `axis`.
    axis=2 gives a scale per TOKEN (a scale for each row of the T x D block);
    axis=1 gives a scale per CHANNEL."""
    qmax = 2 ** (bits - 1) - 1
    s = np.maximum(np.max(np.abs(A), axis=axis, keepdims=True) / qmax, 1e-12)
    return np.clip(np.round(A / s), -qmax, qmax) * s


def attend(q, k, v):
    z = q @ k.transpose(0, 2, 1) / np.sqrt(D)
    z = z - z.max(axis=-1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=-1, keepdims=True)
    return p @ v


REF = attend(Q, K, V)


def err(out):
    return float(np.linalg.norm(out - REF) / np.linalg.norm(REF))


print()
print()
print("Attention output error for each combination of axes.")
print("'token' = one scale per token; 'channel' = one scale per channel.")
print()
print(f"{'bits':>6}{'K axis':>10}{'V axis':>10}{'output error':>15}")
print("-" * 41)

rows = {}
for bits in (4, 3):
    for ka in ("token", "channel"):
        for va in ("token", "channel"):
            kq = quant_axis(K, bits, 2 if ka == "token" else 1)
            vq = quant_axis(V, bits, 2 if va == "token" else 1)
            e = err(attend(Q, kq, vq))
            rows[(bits, ka, va)] = e
            print(f"{bits:>6}{ka:>10}{va:>10}{e:>15.5f}")
    print()

print("Is the good combination good because of K, or because of V?")
print("Each tensor quantized alone, the other left exact.")
print()
print(f"{'bits':>6}{'tensor':>9}{'per-token':>13}{'per-channel':>14}"
      f"{'better':>12}")
print("-" * 54)
solo = {}
for bits in (4, 3):
    kt = err(attend(Q, quant_axis(K, bits, 2), V))
    kc = err(attend(Q, quant_axis(K, bits, 1), V))
    vt = err(attend(Q, K, quant_axis(V, bits, 2)))
    vc = err(attend(Q, K, quant_axis(V, bits, 1)))
    solo[bits] = (kt, kc, vt, vc)
    print(f"{bits:>6}{'keys':>9}{kt:>13.5f}{kc:>14.5f}"
          f"{('channel' if kc < kt else 'token'):>12}")
    print(f"{bits:>6}{'values':>9}{vt:>13.5f}{vc:>14.5f}"
          f"{('channel' if vc < vt else 'token'):>12}")

print()
print()
print("The constraint a static comparison cannot see: a KV cache GROWS.")
print("A per-channel scale needs the max over all tokens, which is not known")
print("until the sequence ends. Below, it is estimated from the first W tokens.")
print()
print(f"{'warmup W':>10}{'K per-channel':>16}{'K per-token':>14}"
      f"{'V per-channel':>16}{'V per-token':>14}")
print("-" * 70)


def quant_prefix_scale(A, bits, warm):
    """Per-channel scale fixed from the first `warm` tokens, then applied to the
    whole sequence -- which is what a streaming cache is forced to do."""
    qmax = 2 ** (bits - 1) - 1
    s = np.maximum(np.max(np.abs(A[:, :warm, :]), axis=1, keepdims=True) / qmax,
                   1e-12)
    return np.clip(np.round(A / s), -qmax, qmax) * s


stream = {}
for warm in (32, 128, 512):
    kc = err(attend(Q, quant_prefix_scale(K, 4, warm), V))
    kt = err(attend(Q, quant_axis(K, 4, 2), V))
    vc = err(attend(Q, K, quant_prefix_scale(V, 4, warm)))
    vt = err(attend(Q, K, quant_axis(V, 4, 2)))
    stream[warm] = (kc, kt, vc, vt)
    print(f"{warm:>10}{kc:>16.5f}{kt:>14.5f}{vc:>16.5f}{vt:>14.5f}")

k4, v4 = solo[4][:2], solo[4][2:]
k3 = solo[3][:2]
best4 = min(rows[(4, a, b)] for a in ("token", "channel")
            for b in ("token", "channel"))
worst4 = max(rows[(4, a, b)] for a in ("token", "channel")
             for b in ("token", "channel"))
print(f"""
The first table is a cheap diagnostic that turns out to predict everything else.
It asks, for each candidate grouping, how far the largest value in a group sits
above a typical one -- ch:q-theory's M/m ratio, which sets how many quantization
levels the ordinary values actually get.

Keys are dramatically inhomogeneous along the token axis: {hom['keys'][0]:.1f}
against {hom['keys'][1]:.1f} per channel. That is the per-channel outlier
structure cite:liu2024kivi reports, seen from the quantizer's side -- a group
containing one token's 64 channels contains both an outlier channel and 61
ordinary ones, so the outlier sets the step for all of them. Group along
CHANNELS instead and each outlier channel is alone with its own kind.

Values are the reverse and much milder: {hom['values'][0]:.1f} per token against
{hom['values'][1]:.1f} per channel.

The second and third tables confirm the keys half emphatically. At 4 bits, keys
quantized per-channel give {k4[1]:.5f} against per-token's {k4[0]:.5f} -- a
factor of {k4[0]/k4[1]:.1f}. At 3 bits, {k3[0]/k3[1]:.1f}x. Choosing the axis is
free, and choosing it wrongly costs more than a bit of width.

And they do NOT confirm the values half. Values come out slightly better
per-channel ({v4[1]:.5f}) than per-token ({v4[0]:.5f}), despite the homogeneity
diagnostic preferring tokens. The gap is small, and the reason it exists is that
these value channels have modestly different variances, which a per-channel scale
captures.

So a static comparison says: quantize both per channel. That is the wrong answer,
and the last table says why.

A KV cache is not a static tensor. It GROWS, one token at a time, and a
per-channel scale is a maximum over the token axis -- over data that has not
arrived yet. The only options are to recompute the scale and requantize the whole
cache on every token, which is absurd, or to fix it early and live with it.

Fix it from the first 32 tokens and the key error rises from {stream[512][0]:.5f}
to {stream[32][0]:.5f}. Later tokens exceed the early maximum and clip. Per-token
scales are unaffected at every warmup -- {stream[32][1]:.5f} at every row --
because each token's scale is computed when that token arrives and never needs
revising (eq:streaming-forces-the-axis).

And now look at the values columns, because this is where the recommendation
actually comes from. With the full sequence available, V per-channel wins
({stream[512][2]:.5f} against {stream[512][3]:.5f}). With a 32-token warmup it
LOSES: {stream[32][2]:.5f} against per-token's {stream[32][3]:.5f}. The static
comparison and the streaming comparison give opposite answers for V.

For K they do not. Even with the worst warmup, per-channel keys
({stream[32][0]:.5f}) still beat per-token keys ({stream[32][1]:.5f}). The keys'
per-channel advantage is large enough to survive a badly estimated scale; the
values' is not.

That is the constraint the papers' recommendation is really made under, and it
explains the shape of the actual design rather than the one a static experiment
suggests. Keys need per-channel scales badly enough -- a factor of
{k4[0]/k4[1]:.1f} -- to be worth handling the streaming problem: quantize the
cache in chunks, keeping a small recent window in full precision and sealing each
chunk's channel scales once the chunk is complete. Values do not need per-channel
scales badly enough to be worth any of that, so they take the axis that streams
for free.

The asymmetry is therefore not two facts about two tensors. It is one fact about
the keys -- they have per-channel outliers -- meeting one fact about caches, that
they grow along the token axis. Neither alone produces the recommendation.

One more piece, because it explains why per-channel scales on K are implementable
at all. Attention scores are Q K^T, a sum over channels, so by
ch:q-int8-int4's eq:reduction-axis-constraint a per-channel scale on K cannot be
factored out of the dot product. It can be folded into Q, which carries the same
channel index: (Q * s)(K/s)^T is the same product. The attention output is A V, a
sum over tokens, so a per-token scale on V folds into A the same way.

Each tensor has exactly one axis whose scale its partner can absorb, and they are
different axes. That is a third independent reason the recommendation comes out
asymmetric, and it is the one that decides whether a kernel can be written at
all.""")
```

The first listing is about how to quantize the cache. The second is about whether
that is the thing to do.

```python {tier=A name=kv-scales-with-traffic}
"""The KV cache is the memory term that grows with traffic, and three things
compete to shrink it.

Weights are a fixed cost: a 70B model occupies the same bytes whether it serves
one request or a thousand. The KV cache is not. It grows with context length, and
it grows again with every concurrent sequence, so past a certain load it is the
term that decides how many requests fit (eq:kv-scales-with-traffic).

Three levers act on it, and they are not the same size. This listing prices all
three against each other: the architectural one (how many KV heads the model has),
the numerical one (how many bits each element takes), and the one that is not
about the tensor at all -- how the memory is ALLOCATED, which
cite:kwon2023pagedattention found was worth more than either.
"""
import numpy as np

# layers, model dim, query heads, head dim -- shapes in the usual proportions.
MODELS = {
    "7B":  dict(P=7e9,  L=32, heads=32, hdim=128),
    "70B": dict(P=70e9, L=80, heads=64, hdim=128),
}


def kv_bytes(m, ctx, batch, bits, kv_heads=None):
    """Two tensors, per layer, per KV head, per token, per sequence."""
    kvh = kv_heads if kv_heads else m["heads"]
    return 2 * m["L"] * kvh * m["hdim"] * ctx * batch * bits / 8.0


def w_bytes(m, bits):
    return m["P"] * bits / 8.0


def gb(x):
    return x / 1e9


print("When does the KV cache overtake the weights? 70B, weights at 4 bits,")
print("cache at 16 bits, full multi-head attention.")
print()
m = MODELS["70B"]
print(f"{'context':>10}" + "".join(f"{'batch ' + str(b):>12}"
                                   for b in (1, 8, 32, 128)))
print(f"{'':>10}{'KV cache GB (weights are ' + f'{gb(w_bytes(m, 4)):.0f} GB)':>48}")
print("-" * 60)
for ctx in (2048, 8192, 32768, 131072):
    row = [gb(kv_bytes(m, ctx, b, 16)) for b in (1, 8, 32, 128)]
    print(f"{ctx:>10,}" + "".join(f"{v:>12.1f}" for v in row))

print()
print()
print("Three levers on the same quantity. 70B, 8k context, batch 32.")
print()
print(f"{'configuration':>34}{'KV cache':>12}{'vs baseline':>14}")
print("-" * 60)
base = kv_bytes(m, 8192, 32, 16)
levers = [
    ("baseline: MHA, 16-bit cache", kv_bytes(m, 8192, 32, 16)),
    ("quantize the cache to 8 bits", kv_bytes(m, 8192, 32, 8)),
    ("quantize the cache to 4 bits", kv_bytes(m, 8192, 32, 4)),
    ("quantize the cache to 2 bits", kv_bytes(m, 8192, 32, 2)),
    ("GQA with 8 KV heads, 16-bit", kv_bytes(m, 8192, 32, 16, 8)),
    ("GQA with 8 KV heads, 4-bit", kv_bytes(m, 8192, 32, 4, 8)),
    ("MQA with 1 KV head, 4-bit", kv_bytes(m, 8192, 32, 4, 1)),
]
for name, v in levers:
    print(f"{name:>34}{gb(v):>10.1f} GB{base/v:>13.0f}x")

print()
print()
print("The third lever: allocation. Reserving the maximum context per sequence")
print("against allocating pages as the sequence actually grows.")
print()
print(f"{'workload':>28}{'mean len':>10}{'max len':>10}{'reserved':>11}"
      f"{'used':>9}{'waste':>9}")
print("-" * 77)

rng = np.random.default_rng(271)
WORKLOADS = [
    ("uniform 1k-2k", lambda n: rng.integers(1024, 2048, n), 2048),
    ("heavy-tailed, 8k cap", lambda n: np.clip(
        rng.lognormal(6.2, 1.0, n).astype(int), 32, 8192), 8192),
    ("heavy-tailed, 32k cap", lambda n: np.clip(
        rng.lognormal(6.2, 1.3, n).astype(int), 32, 32768), 32768),
    ("chat: short with outliers", lambda n: np.where(
        rng.random(n) < 0.05, rng.integers(8000, 32768, n),
        rng.integers(200, 1500, n)), 32768),
]
alloc = {}
for name, gen, cap in WORKLOADS:
    lens = gen(4000)
    reserved = cap * len(lens)
    used = lens.sum()
    alloc[name] = (float(lens.mean()), cap, used / reserved)
    print(f"{name:>28}{lens.mean():>10.0f}{cap:>10,}"
          f"{reserved/1e6:>9.1f}M{used/1e6:>8.1f}M"
          f"{1 - used/reserved:>9.1%}")

print()
print()
print("What each lever buys in concurrent sequences, on one 80 GB card.")
print()
print(f"{'configuration':>40}{'KV budget':>12}{'sequences':>12}")
print("-" * 64)
BUDGET = 80e9 - w_bytes(m, 4)
per_seq = {}
for label, bits, kvh, eff in [
    ("MHA, 16-bit, reserve max context", 16, None, alloc["heavy-tailed, 8k cap"][2]),
    ("MHA, 16-bit, paged", 16, None, 1.0),
    ("GQA-8, 16-bit, paged", 16, 8, 1.0),
    ("GQA-8, 4-bit, paged", 4, 8, 1.0),
]:
    b = kv_bytes(m, 8192, 1, bits, kvh) / eff
    per_seq[label] = BUDGET / b
    print(f"{label:>40}{gb(BUDGET):>10.0f} GB{BUDGET/b:>12.1f}")

q4 = base / kv_bytes(m, 8192, 32, 4)
g8 = base / kv_bytes(m, 8192, 32, 16, 8)
both = base / kv_bytes(m, 8192, 32, 4, 8)
ht = alloc["heavy-tailed, 32k cap"]
chat = alloc["chat: short with outliers"]
print(f"""
The first table is why this chapter exists. A 70B model at 4 bits occupies
{gb(w_bytes(m, 4)):.0f} GB of weights, and that number never changes. At 8k
context and batch 32 the KV cache is {gb(kv_bytes(m, 8192, 32, 16)):.0f} GB --
larger than the model. At 128k context and batch 128 it is
{gb(kv_bytes(m, 131072, 128, 16)):.0f} GB, which is not a number any single
machine has.

That is the structural point. Weights are a fixed cost paid once; the cache is a
variable cost paid per concurrent token. Every capacity question in serving is
really a question about the second (eq:kv-scales-with-traffic), and the model's
parameter count -- the number in its name -- barely enters.

The second table puts the three levers side by side, and the ranking is not the
one the quantization literature would suggest.

Quantizing the cache from 16 bits to 4 is worth {q4:.0f}x. Switching from
multi-head attention to grouped-query attention with 8 KV heads is worth
{g8:.0f}x, and it is an architectural decision made when the model was trained,
not something you apply at deployment. Doing both is worth {both:.0f}x.

So the largest single lever on KV memory is one you do not control at serving
time. That is worth knowing before optimising the one you do -- and it explains
why grouped-query attention became universal so quickly. It bought more than any
amount of numerical cleverness could, at a small and measurable quality cost, and
it composes with quantization rather than competing.

The third table is the lever that is not about the tensor at all, and it is
cite:kwon2023pagedattention's contribution.

A KV cache has to live somewhere contiguous for the attention kernel to read it,
so the obvious implementation reserves the maximum supported context for every
sequence when it starts. Look at what that costs on realistic length
distributions. A heavy-tailed workload with a 32k cap has a mean length of
{ht[0]:.0f} tokens and wastes {1-ht[2]:.0%} of what it reserved. A chat workload
where 5% of conversations are long wastes {1-chat[2]:.0%}.

Those are not inefficiencies at the margin. On the chat row, more than nine tenths
of the most contested memory in the system is reserved for tokens that will never
exist, because a small minority of requests might have needed it.

Paging fixes it the way operating systems fixed the same problem: allocate the
cache in fixed-size blocks that need not be contiguous, and hand out blocks as the
sequence actually grows. The attention kernel takes an indirection through a block
table. Nothing about the tensor's contents changes.

The last table puts all of it together in the unit that matters -- how many
concurrent 8k-context sequences fit alongside the weights on one 80 GB card.

Reserving the maximum with a 16-bit MHA cache fits
{per_seq['MHA, 16-bit, reserve max context']:.1f}. Not one sequence: the
reservation for a single request exceeds the entire remaining budget, which is
the arithmetic form of "this configuration does not work". Paging the same cache
fits {per_seq['MHA, 16-bit, paged']:.1f}. Adding grouped-query attention,
{per_seq['GQA-8, 16-bit, paged']:.1f}. Adding 4-bit quantization on top,
{per_seq['GQA-8, 4-bit, paged']:.1f}.

Read that sequence and the ordering of effort follows. Fix the allocator, then
choose an architecture with fewer KV heads, then quantize. The first two are
larger than the third, and only the third is what "KV cache quantization" refers
to.

None of which makes the quantization worthless -- it is the last multiplier on the
stack and it composes with everything before it. It makes the point that a
throughput problem attributed to the cache's PRECISION is usually a problem with
its ALLOCATION, and that the cheapest fix is not the one this part is about.""")
```

## 9. Practical Example

**The cache overtakes the weights almost immediately.** A 70B model at 4 bits is
**35 GB**; its 16-bit cache at 8k context and batch 32 is **687 GB**, at 128k and
batch 128 is **43,981 GB**. {{eq:kv-overtakes-weights}} puts the crossover at
about **13,000 token-slots** — under two concurrent 8k sequences.

**The axis diagnostic is cheap and points the right way for keys.**
{{eq:group-homogeneity}}: keys **16.54** per-token against **4.78** per-channel;
values **3.90** against **4.79**.

**But the static error comparison says per-channel for both** (values **0.08089**
against **0.09343**), because per-channel groups here are 8× larger and capture
genuine channel-variance differences. **The diagnostic and the static measurement
disagree, and the static measurement is the wrong experiment.**

**The streaming constraint decides.** Estimating a per-channel scale from the first
32 tokens: **K per-channel 0.44525** (still beating per-token's **0.51824**) but
**V per-channel 0.15405**, now *losing* to per-token's **0.09343**.
{{eq:streaming-forces-the-axis}}.

> **IMPORTANT:** So {{cite:liu2024kivi}}'s recommendation is one fact about keys —
> they carry per-channel outliers — meeting one fact about caches — they grow along
> the token axis. **Neither alone produces it**, and a static experiment produces
> the wrong answer for values.

**And all four axes are implementable**, because
{{eq:partner-absorbs-the-scale}} lets each scale migrate to the operand sharing
its index. **The choice is made on homogeneity and streaming, not on feasibility.**

**The three levers are not the same size.** Quantizing 16→4 bits: **4×**. GQA-8:
**8×**, decided at training time. Both: **32×**. MQA-1 at 4 bits: **256×**.

**And allocation is larger than either.** Realistic workloads waste **90.3%,
96.4%, 94.6%** of reserved cache ({{eq:allocation-beats-precision}}), so
{{cite:kwon2023pagedattention}}'s paging is worth **18×** on the chat workload —
**at no accuracy cost at all.**

**In the unit that matters, sequences on one 80 GB card**: reserve-max **0.2**
(the reservation for a single request exceeds the whole budget), paged **2.1**,
plus GQA-8 **16.8**, plus 4-bit **67.1**.

**Fix the allocator, then the architecture, then the precision.** Only the last is
what "KV cache quantization" means, and it is the smallest of the three —
though {{eq:levers-multiply}} means it still multiplies everything before it.

## 10. Production Considerations

**Measure your length distribution before anything else.**
{{eq:allocation-beats-precision}} is workload-dependent and usually the largest
factor.

**Use a paged allocator.** There is no accuracy trade to weigh.

**Check $h/h_{\text{kv}}$** when selecting a model — it is a serving-capacity
number as much as a quality one.

**Quantize keys per channel with chunked scales**, values per token.

**Keep a small recent window unquantized** — it is what makes per-channel key
scales possible at all.

**Report architecture, allocator and precision together.** A KV memory figure
without all three does not transfer.

**Do not extrapolate cache behaviour from a load test with different length
statistics** — the allocation factor changes with the workload and nothing else
does.

## 11. Common Mistakes

**Treating "the KV cache" as one tensor.** It is two, with different statistics
and different consumers.

**Choosing the quantization axis from a static comparison.** It gives the wrong
answer for values.

**Reserving maximum context per sequence**, then attributing the resulting
capacity limit to precision.

**Comparing KV memory across models without noting $h_{\text{kv}}$.**

**Assuming the reduction-axis constraint rules out per-channel key scales.**
{{eq:partner-absorbs-the-scale}}.

**Quantizing activations in flight below 8 bits** without addressing outliers.

**Benchmarking cache capacity on a workload with unrepresentative lengths.**

## 12. Failure Modes

**Quality degrades as context grows.** Cause:
{{eq:streaming-forces-the-axis}} — a per-channel scale fixed too early, clipping
later tokens.

**Capacity far below the arithmetic.** Cause: reservation-based allocation.

**KV quantization helps far less than expected.** Cause: the model already uses
GQA, so the baseline was already 8× smaller.

**Long-context requests evict short ones unpredictably.** Cause: reservation
granularity; paging fixes it.

**Quantized cache fine offline, poor in production.** Cause: the length
distribution differs, so both the allocation factor and the warmup window differ.

**Accuracy loss concentrated in early tokens.** Cause: the unquantized recent
window is protecting late tokens and not early ones — which is the correct
behaviour and worth recognising rather than debugging.

## 13. Alternatives

| Alternative | Trades | When |
|---|---|---|
| paged allocation | kernel indirection | always; no accuracy cost |
| GQA / MQA | some quality | chosen at training time |
| KV quantization to 8 bits | little | safe default |
| KV quantization to 4 bits | some quality | when capacity binds |
| KIVI-style 2-bit | more quality, complexity | extreme capacity pressure |
| sliding-window attention | long-range recall | when the workload allows |
| prefix sharing | none, when prefixes repeat | system prompts, few-shot |

**The last row is the one most often left on the table.**
{{cite:kwon2023pagedattention}}'s paging makes identical prefixes share physical
blocks, so a shared system prompt across a thousand requests is stored once.
**For many production workloads that is larger than every other row combined**,
and like paging it costs nothing.

## 14. Evaluation

**Report the length distribution**, not just the maximum context.

**Report $h_{\text{kv}}$ and the allocator** with any KV memory number.

**Report accuracy as a function of position in the sequence**, since
{{eq:streaming-forces-the-axis}} predicts position-dependent damage.

**Report the recent-window size** for chunked key quantization.

**Measure at the context lengths you serve**, since the crossover in
{{eq:kv-overtakes-weights}} moves everything.

## 15. Advanced Concepts

**Prefix sharing is an allocation win, not a caching one.**
{{maturity:MATURE}} Once the cache is paged, identical prefixes can point at the
same blocks. **This is the same insight as paging applied one level up**, and for
workloads with shared system prompts it dominates.

**Position-dependent quality.** {{maturity:EMERGING}}
{{eq:streaming-forces-the-axis}} predicts that chunked quantization damages
different sequence positions differently. Almost no evaluation measures accuracy
by position, so this failure mode is largely invisible in reported numbers.

**Attention sinks as KV outliers.** {{maturity:EMERGING}} The first few tokens of
a sequence often carry disproportionate attention mass, and their keys are
correspondingly extreme. **Keeping them unquantized is nearly free and is a
different exception from the recent window**, since one is at the start and one at
the end.

**Activation quantization in flight.** {{maturity:EXPERIMENTAL}} Below 8 bits this
remains hard for the reasons {{ch:q-int8-int4}} gave. The cache is the easier
target because it is written once and read many times, so its statistics can be
computed at write time.

**Cache compression beyond quantization.** {{maturity:RESEARCH FRONTIER}}
Eviction, merging and low-rank projection of the cache all attack
{{eq:kv-scales-with-traffic}}'s $S$ term rather than its $b$ term. **Attacking $S$
changes what the model can attend to and attacking $b$ does not**, which is the
distinction to keep clear when the two are compared.

## 16. Connection to Previous Chapters

{{ch:q-int8-int4}}'s {{eq:reduction-axis-constraint}} appeared to forbid
per-channel key scales, and {{eq:partner-absorbs-the-scale}} shows the migration
that permits them — the same idea as {{cite:xiao2023smoothquant}}'s, applied
inside attention.
{{ch:q-theory}}'s {{eq:effective-levels}} is what {{eq:group-homogeneity}} makes
into a diagnostic, and its group-size result is the axis choice in another form.
{{ch:q-gguf}}'s {{eq:decode-roofline}} had $M_{\text{kv}}$ set to zero; this
chapter fills it in and shows where the approximation fails.
{{ch:tf-masking-kv}} established what the cache is for.
Forward: {{ch:q-memory-math}} assembles every term into one budget;
{{ch:q-runtimes}} is largely about which stacks implement the allocator well; and
{{ch:q-throughput-latency}} uses the capacity this chapter recovers.

## 17. Exercises

1. From {{eq:kv-scales-with-traffic}}, compute the cache for a 13B GQA-8 model at
   32k context and batch 16.
2. Derive {{eq:kv-overtakes-weights}} and find the crossover for that model.
3. Compute {{eq:allocation-beats-precision}} for a workload whose lengths are
   lognormal with median 800 and a 16k cap.
4. Show {{eq:partner-absorbs-the-scale}} algebraically for both attention
   products, and state which of the four axis choices need no migration.
5. In `kv-axis-asymmetry`, remove the imposed key outliers. Does the asymmetry
   survive, and what does that say about its cause?
6. In the same listing, sweep the warmup window from 8 to 512. Where does K
   per-channel stop beating K per-token?
7. In `kv-scales-with-traffic`, add prefix sharing: assume 60% of every sequence
   is a shared system prompt. Recompute the last table.
8. For a workload you have: measure the length distribution and compute all three
   factors of {{eq:levers-multiply}}.

## 18. Interview Questions

1. Why does the KV cache matter more than the model size for serving capacity?
2. Why are keys quantized per channel and values per token?
3. A static experiment says per-channel for both. What is it missing?
4. Why is a per-channel scale on the keys implementable at all?
5. What is the biggest lever on KV memory, and can you apply it at deployment?
6. What fraction of a reserved cache is typically wasted, and why?
7. Why does prefix sharing require paged allocation?
8. Why might quality degrade with position in the sequence under KV
   quantization?
9. Why is quantizing the cache easier than quantizing activations in flight?
10. Your KV quantization helped less than a published result. Name two likely
    reasons.

## 19. Research Questions

1. {{eq:streaming-forces-the-axis}} predicts position-dependent damage. How large
   is it on real long-context tasks, and does the recent window fully mask it?
2. Attention sinks and the recent window are two exceptions at opposite ends of
   the sequence. Is there a principled rule for which tokens to keep unquantized?
3. {{eq:levers-multiply}}'s allocation factor is workload-dependent. Can it be
   predicted from a workload's length distribution well enough to size a cluster
   in advance?
4. Cache eviction attacks $S$ and quantization attacks $b$. At equal memory
   saving, which damages long-context reasoning less?
5. Are key per-channel outliers necessary, or an artefact of positional encoding
   and normalisation? An architecture without them would make 2-bit keys
   straightforward.

## 20. Chapter Summary

**The KV cache is the memory term that grows with traffic.** A 70B model's weights
are **35 GB** forever; its 16-bit cache at 8k context and batch 32 is **687 GB**,
and {{eq:kv-overtakes-weights}} puts the crossover at about **13,000 token-slots**
— under two concurrent sequences.

**The key/value axis asymmetry is derivable, not arbitrary.**
{{eq:group-homogeneity}} shows keys at **16.54** per-token against **4.78** per
channel. But a static error comparison says per-channel for *both* — and
{{eq:streaming-forces-the-axis}} shows why that is the wrong experiment: with a
32-token warmup, **V per-channel goes to 0.15405 and loses to per-token's
0.09343**, while **K per-channel stays ahead at 0.44525 against 0.51824**.

**So the recommendation is one fact about keys meeting one fact about caches** —
per-channel outliers, and growth along the token axis. Neither alone produces it.
And **all four axes are implementable**
({{eq:partner-absorbs-the-scale}}), so the choice is made on homogeneity and
streaming rather than on feasibility.

**Three levers act on the same quantity and they are not the same size.**
Precision 16→4 bits: **4×**. Grouped-query attention: **8×**, decided at training
time. **Allocation: 18×** on a realistic chat workload, because reservation-based
caches waste **90–96%** of what they reserve — **and it costs no accuracy at
all.**

**In sequences on one 80 GB card: 0.2 → 2.1 → 16.8 → 67.1**, fixing the
allocator, then the architecture, then the precision.

Which is the chapter's ordering and its uncomfortable part: **only the last of
those is what "KV cache quantization" refers to, and it is the smallest.**
{{eq:levers-multiply}} means it still multiplies everything before it — but a
throughput problem blamed on the cache's *precision* is usually a problem with its
*allocation*, and the cheapest fix is not the one this part is about.

## 21. Further Reading

{{cite:liu2024kivi}} for the axis asymmetry, read with
{{eq:streaming-forces-the-axis}} in mind: the paper's chunked scheme is what the
streaming constraint forces, and the static intuition for it is incomplete.
{{cite:kwon2023pagedattention}} for paging, and note how much of the reported
2–4× throughput gain is allocation rather than computation — this chapter's
measurement suggests most of it.
{{cite:xiao2023smoothquant}} for the migration idea that
{{eq:partner-absorbs-the-scale}} reuses inside attention.
{{cite:dettmers2022int8}} for why in-flight activation quantization remains the
harder problem.
{{cite:pope2022inference}} for the serving-capacity framing this chapter's
arithmetic feeds into, developed in {{ch:q-throughput-latency}}.

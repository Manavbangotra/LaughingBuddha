---
id: tf-complexity
number: 70
part: VII
tier: full
status: reviewed
requires: [tf-masking-kv, tf-ffn-residual, tf-multi-head, dl-forward, dl-backprop]
provides: [transformer-flops, activation-memory-transformer, attention-memory,
           quadratic-cost, roofline-transformer, training-compute,
           chinchilla-accounting, memory-vs-compute]
citations: [vaswani2017, dao2022flash, kaplan2020scaling, hoffmann2022chinchilla]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Count every FLOP in a transformer forward and backward pass.
2. Derive the $6ND$ rule for training compute and say what it approximates.
3. Separate the costs that scale with $T$ from those that scale with $T^2$.
4. Account for training memory: parameters, gradients, optimiser state,
   activations.
5. Explain why attention's *memory* is quadratic even when its FLOPs are a
   minority.
6. Apply the roofline model to decide whether an operation is compute- or
   memory-bound.
7. Estimate the cost of a training run or a serving deployment from first
   principles.

## 2. Why This Matters

**"Attention is quadratic" is true and routinely misapplied.** Quadratic in
*memory*, always. Quadratic in *FLOPs* only past $T \approx 6d$, and below that
threshold the linear projections and the feed-forward block dominate. Conflating
the two leads to optimising the wrong thing, and
{{cite:dao2022flash}} is the clearest evidence — it attacks the memory and
leaves the arithmetic untouched.

**The $6ND$ rule lets you cost a training run on the back of an envelope.**
Six FLOPs per parameter per token, and everything else is engineering. It is the
single most useful number in large-scale machine learning and it takes four
lines to derive.

**Memory, not compute, is what usually stops you.** A model that fits in
arithmetic will fail on activations, on optimiser state, or on the KV cache
long before it runs out of FLOPs. {{sec:6-mathematical-foundation}} accounts for
all four.

**Every optimisation in {{ch:tf-efficient}} targets one line of this
accounting.** Without the accounting, those techniques are a list to memorise;
with it, each one is an obvious response to a specific term.

## 3. Prerequisites

{{ch:tf-masking-kv}} for the prefill/decode split and the cache.
{{ch:tf-ffn-residual}} for the block's parameter and FLOP split.
{{ch:tf-multi-head}} for the attention cost.
{{ch:dl-forward}} for the roofline model and arithmetic intensity — this chapter
is that chapter applied to one architecture.
{{ch:dl-backprop}} for the three-times rule.

## 4. Intuitive Explanation

### 4.1 Four numbers

Everything in this chapter is one of four quantities:

```text
   FLOPs      how much arithmetic
   bytes      how much memory traffic
   capacity   how much memory is occupied
   depth      how long the critical path is
```

They fail independently. A model can be FLOP-cheap and capacity-bound (long
context), or capacity-cheap and bandwidth-bound (single-stream decoding), or
fine on all three and latency-bound by depth (a very deep model).

**Naming which one is binding is most of the skill.** The rest is arithmetic.

### 4.2 Where the FLOPs are

Per token, per layer:

```text
   attention projections   8d²        (Q, K, V, O)
   attention scores        4Td        (QKᵀ then AV)
   feed-forward           16d²        (two matrices of 4d width)
                         ─────
   total                  24d² + 4Td
```

**The quadratic term is $4Td$ and the linear terms are $24d^2$.** They are equal
when $T = 6d$ — so for $d = 4096$, attention's quadratic part only overtakes the
rest past 24,000 tokens.

That is the number to remember, because it contradicts the usual framing. At
ordinary context lengths a transformer is dominated by matrix multiplications
against weights, not by attention.

### 4.3 Where the memory is

Four consumers, and they dominate at different times:

```text
   parameters       P · b                     fixed
   gradients        P · b                     training only
   optimiser state  2P · 4                    training only, fp32
   activations      B·T·(stuff) · b           training, scales with batch
   KV cache         2·b·L·g·d_k·T·B           serving only, scales with users
```

**During training, activations usually dominate.** During serving, the KV cache
does. They are different problems with different fixes, and a technique aimed at
one does nothing for the other.

### 4.4 The one genuinely quadratic thing

The $T \times T$ attention matrix.

```text
   per layer, per head:   T² floats
   times h heads, L layers, B sequences:  B·L·h·T²
```

At $B=8$, $L=32$, $h=32$, $T=8192$ in bf16 that is **1.1 terabytes**. Not
gigabytes. This is the term that makes long contexts impossible, and it is why
FlashAttention — which never writes it to memory — mattered so much.

Note what FlashAttention does *not* change: the $4Td$ FLOPs are still performed,
the KV cache is still stored. **It removes one specific term from the memory
accounting and nothing else.**

### 4.5 The 6ND rule

For a model with $N$ parameters trained on $D$ tokens:

$$
C \approx 6ND \ \text{FLOPs}
$$

Two FLOPs per parameter for the forward pass (a multiply and an add), and four
for the backward ({{ch:dl-backprop}}'s three-times rule: one forward's worth
plus two more).

**That is the whole derivation.** For a 70B model on 2 trillion tokens:
$6 \times 7\times10^{10} \times 2\times10^{12} = 8.4\times10^{23}$ FLOPs. At
$10^{15}$ effective FLOP/s per accelerator that is 27 accelerator-years.

## 5. Formal Explanation

### 5.1 Parameters

$$
N = \underbrace{2Vd}_{\text{embeddings}}
 + \underbrace{L\big(4d^2 + 2d\,d_{\text{ff}}\big)}_{\text{blocks}}
 \approx 12Ld^2 \ \ \text{for } d_{\text{ff}}=4d,\ V \ll 6Ld
$$ (eq:transformer-params)

The approximation drops the embeddings, which is safe for large models and wrong
for small ones ({{ch:tf-embeddings}} measured the crossover).

### 5.2 Forward FLOPs

Per token:

$$
F_{\text{fwd}} = \underbrace{2N}_{\text{all matmuls}}
 + \underbrace{4LTd}_{\text{attention scores and values}}
 + \underbrace{2dV}_{\text{unembedding}}
$$ (eq:forward-flops-transformer)

The first term is the key identity: **every parameter participates in exactly
one multiply-accumulate per token**, which is 2 FLOPs. That is why $2N$ appears
and why the rule is so simple.

The second term is the only part not attributable to a parameter — the score
matrix and the weighted sum involve no weights at all.

### 5.3 Training compute

Backward is about twice forward ({{ch:dl-backprop}}), so

$$
F_{\text{step}} \approx 3 F_{\text{fwd}} = 6N + 12LTd \ \text{per token}
$$ (eq:training-flops)

$$
C_{\text{total}} \approx 6ND\Big(1 + \frac{2LTd}{N}\Big)
 \approx 6ND\Big(1 + \frac{T}{6d}\Big)
$$ (eq:6nd-with-attention)

using $N \approx 12Ld^2$.

**The correction term is $T/6d$.** At $T = 2048$, $d = 4096$ it is 8% — safely
ignorable. At $T = 128000$, $d = 4096$ it is 5.2, so attention is *five times*
the parameter cost and the $6ND$ rule is badly wrong.

> IMPORTANT: $6ND$ is a long-context-blind approximation. Quote it with the
> correction, or say which regime you are in. Most published compute figures
> use plain $6ND$, which is fine for the 2k–8k contexts they were computed at
> and misleading for long-context training.

### 5.4 Activation memory

Per layer, per token, the values that must be stored for the backward pass:

$$
M_{\text{act}}^{\text{layer}} = b\Big(
 \underbrace{\vphantom{d_{\text{ff}}}\alpha d}_{\text{stream, norms, projections}}
 + \underbrace{d_{\text{ff}}}_{\text{FFN intermediate}}
 + \underbrace{hT}_{\text{attention matrix}}\Big)
$$ (eq:activation-memory-transformer)

with $\alpha \approx 10$ counting the various intermediates a naive
implementation keeps.

Total: $M_{\text{act}} = BTL\,M^{\text{layer}}$, so

$$
M_{\text{act}} = b\,BTL\big(\alpha d + 4d + hT\big)
$$ (eq:total-activation-memory)

**The $hT$ term is the attention matrix and it is the only quadratic one.** It
overtakes the rest when $hT > (\alpha+4)d$, that is $T > 14d/h$ — for
$d=4096$, $h=32$: $T > 1792$ tokens.

**So beyond a couple of thousand tokens, the attention matrix is the largest
single consumer of training memory.** That is the term FlashAttention removes.

### 5.5 Total training memory

$$
M_{\text{train}} = \underbrace{bN}_{\text{weights}}
 + \underbrace{bN}_{\text{gradients}}
 + \underbrace{8N}_{\text{Adam, fp32}}
 + \underbrace{4N}_{\text{fp32 master}}
 + M_{\text{act}}
$$ (eq:training-memory)

In mixed precision with $b = 2$: $2N + 2N + 8N + 4N = 16N$ bytes before a single
activation. **A 7B model needs 112 GB of state to train and 14 GB to serve.**

### 5.6 The complexity table

{#tbl:transformer-complexity caption="Every cost in a transformer, by what it scales with. The last column is the technique that attacks it, and the table is the map for the next chapter."}

| Quantity | Scaling | Dominant when | Attacked by |
|---|---|---|---|
| Parameter FLOPs | $O(NT)$ | always | sparsity, MoE |
| Attention FLOPs | $O(LT^2d)$ | $T > 6d$ | sparse/linear attention |
| Attention memory | $O(BLhT^2)$ | $T > 14d/h$ | FlashAttention |
| Activation memory | $O(BTLd)$ | large batch | checkpointing |
| Optimiser state | $O(N)$ | always, training | sharding, 8-bit Adam |
| KV cache | $O(BLgd_kT)$ | serving | GQA, quantisation |
| Decode bandwidth | $O(N + Lgd_kT)$ | serving | batching, speculation |

### 5.7 Serving cost, per request

Training cost is one number; serving cost is two, and they scale differently.

$$
C_{\text{req}} \approx \underbrace{2NT + 4LdT^2}_{\text{prefill}}
 + \underbrace{n\big(2N + 4Lgd_k\bar{n}\big)}_{\text{decode}}
$$ (eq:request-flops)

with $T$ the prompt, $n$ the output and $\bar{n} \approx T + n/2$.

**Prefill is quadratic in the prompt and decode is linear in the output**, so
the two halves dominate in different regimes. A retrieval-augmented question
with a 32k prompt and a 200-token answer is almost entirely prefill; an agent
loop with a 500-token prompt and 4k of reasoning is almost entirely decode.

The bytes tell a different story again, and it is the one that sets the bill:

$$
B_{\text{req}} \approx \underbrace{bN}_{\text{prefill, once}}
 + \underbrace{n\,bN}_{\text{decode, every token}}
$$ (eq:request-bytes)

**Decode reads the entire model once per output token and prefill reads it
once in total.** That single asymmetry is why output tokens are priced several
times higher than input tokens across every provider, and it is arithmetic
rather than a margin decision — {{ch:tf-masking-kv}} derived the same result
from the other direction.

## 6. Mathematical Foundation

### 6.1 Why every parameter costs exactly 2 FLOPs per token

A weight matrix $\mat{W} \in \R^{m\times n}$ applied to a vector performs $mn$
multiply-accumulates, which is $2mn$ FLOPs by the standard convention — and
$\mat{W}$ has exactly $mn$ parameters.

So for any network built from matrix multiplications:

$$
F_{\text{fwd}} = 2 \times (\text{number of parameters used per token})
$$ (eq:two-flops-per-param)

$\square$

This is why {{eq:forward-flops-transformer}}'s first term is $2N$ and not
something architecture-specific. **It holds for any dense architecture** and it
is the reason the $6ND$ rule generalises beyond transformers.

The exceptions are precisely the operations with no parameters: attention's
scores and weighted sum, the normalisations, the activations. Only the first is
large.

### 6.2 The backward factor of three, precisely

From {{ch:dl-backprop}}, a matmul $\vec{y} = \mat{W}\vec{x}$ needs two backward
matmuls: $\bar{\vec{x}} = \mat{W}\T\bar{\vec{y}}$ and
$\bar{\mat{W}} = \bar{\vec{y}}\vec{x}\T$, each the same size as the forward one.

$$
F_{\text{step}} = F_{\text{fwd}} + 2F_{\text{fwd}} = 3F_{\text{fwd}}
$$ (eq:three-times-rule)

Hence $6N$ per token for the parameters.

Two caveats worth stating, because they are where the rule leaks.

**The first layer does not need $\bar{\vec{x}}$**, saving one matmul out of
$3L$ — negligible.

**Gradient checkpointing adds a forward pass**, making it $4F_{\text{fwd}}$ and
turning $6ND$ into $8ND$. Published compute figures rarely say whether
checkpointing was used, which is a real source of discrepancy when reproducing
someone's numbers.

### 6.3 Where the quadratic term takes over, exactly

Setting the attention FLOPs equal to the parameter FLOPs per token:

$$
4LTd = 2N = 24Ld^2
\quad\Longleftrightarrow\quad
T = 6d
$$ (eq:flop-crossover)

And for memory, from {{eq:total-activation-memory}}:

$$
hT = (\alpha + 4)d
\quad\Longleftrightarrow\quad
T = \frac{(\alpha+4)d}{h} \approx \frac{14d}{h}
$$ (eq:memory-crossover)

**The two crossovers differ by a factor of $6h/14 \approx 14$ for $h = 32$.**

That is the entire reason "attention is quadratic" is confusing. Memory goes
quadratic at $T \approx 1800$ for a typical model; FLOPs only at
$T \approx 24000$. Between those two lengths — which is where most models
operate — **attention is a memory problem and not a compute problem.**

{{cite:dao2022flash}} is exactly the response to that observation, which is why
its title says *IO-awareness* rather than anything about arithmetic.

### 6.4 The roofline for each phase

From {{ch:dl-forward}}, an operation is memory-bound when its arithmetic
intensity $I = W/Q$ falls below the machine's ridge point $\pi/\beta$.

**Training / prefill.** Per layer, $24Td^2 + 4T^2d$ FLOPs against roughly
$b(12d^2 + 3Td + hT^2)$ bytes. For $T \gg 1$ and $d \gg 1$ this is
$O(T d^2)$ over $O(bTd)$, giving

$$
I_{\text{prefill}} \approx \frac{24d}{b} \cdot \frac{1}{1 + \dots}
$$

which for $d = 4096$, $b = 2$ is in the thousands. **Firmly compute-bound.**

**Decode.** {{ch:tf-masking-kv}}'s {{eq:decode-intensity}}: $2B/b$, which is 1
at batch 1 in bf16. **Firmly memory-bound**, by three orders of magnitude.

**Those are the two extremes of the roofline, in the same model, minutes
apart.** No single optimisation serves both, and this is why prefill and decode
are increasingly run on different hardware.

### 6.5 Compute-optimal allocation

Given a compute budget $C \approx 6ND$, how should it be split between model
size $N$ and tokens $D$?

{{cite:kaplan2020scaling}} found power-law scaling of loss in $N$, $D$ and $C$,
and recommended spending most of a marginal budget on $N$.
{{cite:hoffmann2022chinchilla}} redid the analysis with a corrected
learning-rate schedule and found the opposite balance:

$$
N_{\text{opt}} \propto C^{0.5},
\qquad
D_{\text{opt}} \propto C^{0.5}
\quad\Longrightarrow\quad
D_{\text{opt}} \approx 20 N_{\text{opt}}
$$ (eq:chinchilla)

**Roughly 20 tokens per parameter.** Models trained before this were
substantially undertrained — a 175B model on 300B tokens is at 1.7 tokens per
parameter, about a twelfth of optimal.

> NOTE: {{eq:chinchilla}} minimises training loss for a fixed *training* budget.
> It says nothing about inference cost, and a model that will serve billions of
> tokens should be *smaller and trained longer* than Chinchilla-optimal, because
> inference cost scales with $N$ and not with $D$. That is why modern
> open-weight models are trained far past 20 tokens per parameter.

### 6.6 The inference-aware correction

Total lifetime cost for a model serving $D_{\text{inf}}$ tokens:

$$
C_{\text{total}} \approx \underbrace{6ND_{\text{train}}}_{\text{training}}
 + \underbrace{2ND_{\text{inf}}}_{\text{inference}}
$$ (eq:lifetime-cost)

Inference is $2N$ per token, not $6N$, because there is no backward pass.

For a model serving $10^{14}$ tokens over its life — plausible for a popular
API — inference at $2N$ per token exceeds training at $6ND_{\text{train}}$ once
$D_{\text{inf}} > 3D_{\text{train}}$, which is easily reached.

**So for a widely-served model, most of the lifetime compute is inference**, and
the optimal $N$ is smaller than {{eq:chinchilla}}'s. That single observation
explains most of the difference between 2022 and 2026 training practice.

## 7. Internal Mechanics

### 7.1 What the FLOP count omits

$6ND$ counts matmuls and nothing else. Missing:

```text
   normalisations       O(BTLd)     memory-bound, cheap in FLOPs
   activations          O(BTLd_ff)  same
   softmax              O(BLhT²)    same order as the scores
   the optimiser step   O(N)        memory-bound, per step not per token
   dropout, masking     O(...)      elementwise
```

None contributes meaningfully to the FLOP total and together they can be a
substantial fraction of the *time*, because they are all bandwidth-bound
({{ch:dl-forward}} measured a quarter of a small model's step going to
elementwise work).

**That is why measured MFU — model FLOPs utilisation — tops out well below
100% even on a perfectly implemented model.** Reported figures of 40–55% are
good, and the gap is mostly these terms plus communication.

### 7.2 Sequence length in the batch

Memory scales with $B \times T$, so a fixed token budget can be spent as many
short sequences or few long ones. Most training pipelines pack sequences to a
fixed length precisely so this stays constant.

The attention term does *not* stay constant under packing: $hT^2$ per sequence
means $B$ sequences of length $T$ cost $BhT^2$, while one sequence of length
$BT$ costs $h(BT)^2 = B^2hT^2$. **Packing short sequences together is
quadratically cheaper than one long one** — provided the attention mask prevents
cross-sequence attention, which is a real implementation detail with a real
correctness consequence.

### 7.3 Estimating a run

The procedure, which is worth internalising:

```text
   1.  N from the architecture           eq. 70.1
   2.  C = 6ND (+ correction if T large) eq. 70.5
   3.  wall-clock = C / (accelerators × peak × MFU)
   4.  memory = 16N + activations        eq. 70.9
   5.  check memory fits BEFORE step 3
```

Step 5 is the one people skip, and it is the one that fails.

### 7.4 Measuring rather than estimating

MFU is the ratio of achieved to peak FLOPs:

$$
\text{MFU} = \frac{6ND_{\text{observed}}/t}{\text{peak FLOP/s}}
$$ (eq:mfu)

It is the right headline metric for a training run because it is comparable
across models and hardware. A low MFU points at communication, at data loading,
or at the elementwise terms of {{sec:7-internal-mechanics}} — and which one is
found by profiling, not by guessing.

### 7.5 Precision changes the accounting

Halving $b$ halves every memory term and every bandwidth term, and leaves the
FLOP count unchanged — but accelerators have higher peak throughput at lower
precision, so the effective FLOPs available roughly double.

**So quantisation buys memory linearly and speed roughly linearly for
memory-bound work, and it does not change the arithmetic.** That is why it helps
decode enormously and prefill much less ({{part:15}}).

## 8. Implementation

```python {tier=A name=counting-flops-and-memory}
"""Every cost in a transformer, counted (eqs. 70.1-70.9)."""
import numpy as np


class Config:
    def __init__(self, name, V, L, d, h, d_ff=None, g=None, b=2):
        self.name, self.V, self.L, self.d, self.h = name, V, L, d, h
        self.d_ff = d_ff or 4 * d
        self.g = g if g is not None else h
        self.dk = d // h
        self.b = b

    def params(self):
        emb = 2 * self.V * self.d
        blocks = self.L * (4 * self.d ** 2 + 2 * self.d * self.d_ff)
        return emb + blocks, emb, blocks

    def fwd_flops_per_token(self, T):
        """Eq. 70.2."""
        N, emb, blocks = self.params()
        param_flops = 2 * (blocks + self.V * self.d)   # blocks + unembedding
        attn_flops = 4 * self.L * T * self.d
        return param_flops + attn_flops, param_flops, attn_flops

    def act_bytes_per_token(self, T):
        """Eq. 70.6-70.7, alpha = 10."""
        per_layer = self.b * (10 * self.d + self.d_ff + self.h * T)
        return self.L * per_layer

    def kv_bytes_per_token(self):
        return 2 * self.b * self.L * self.g * self.dk


MODELS = [
    Config("GPT-2 small", 50257, 12, 768, 12),
    Config("1.3B", 50257, 24, 2048, 16),
    Config("7B (GQA 8)", 32000, 32, 4096, 32, d_ff=11008, g=8),
    Config("70B (GQA 8)", 128000, 80, 8192, 64, d_ff=28672, g=8),
]

print("=" * 72)
print("parameters (eq. 70.1)")
print("=" * 72)
print(f"{'model':<14} {'total':>12} {'embeddings':>12} {'blocks':>12} "
      f"{'embed %':>9} {'12Ld^2 approx':>15} {'error':>8}")
for c in MODELS:
    N, emb, blk = c.params()
    approx = 12 * c.L * c.d ** 2
    print(f"{c.name:<14} {N / 1e9:>11.2f}B {emb / 1e9:>11.3f}B "
          f"{blk / 1e9:>11.2f}B {emb / N:>9.1%} {approx / 1e9:>14.2f}B "
          f"{abs(approx - N) / N:>7.1%}")

print("\nThe 12Ld^2 approximation is good for large models and poor for")
print("small ones, and the embed-% column says why — it drops the")
print("embeddings, which are a third of GPT-2 small and one per cent of a")
print("70B model. That is Chapter 66's crossover appearing in the FLOP")
print("accounting.")
print("\nNote also that the 7B and 70B rows use a gated feed-forward block,")
print("so d_ff is about 8d/3 across three matrices rather than 4d across")
print("two — the 12Ld^2 shorthand still lands close because eq. 67.6 was")
print("chosen to keep the parameter count matched.")

# --- section 6.3: where the quadratic term takes over -----------------------
print("\n" + "=" * 72)
print("where attention's FLOPs overtake everything else (eq. 70.10)")
print("=" * 72)
print(f"{'model':<14} {'6d (predicted)':>16} " +
      " ".join(f"{f'T={T}':>14}" for T in (2048, 8192, 32768, 131072)))
print(f"{'':<14} {'':>16} " +
      " ".join(f"{'attn % of FLOPs':>14}" for _ in range(4)))
for c in MODELS:
    row = []
    for T in (2048, 8192, 32768, 131072):
        tot, par, att = c.fwd_flops_per_token(T)
        row.append(att / tot)
    print(f"{c.name:<14} {6 * c.d:>16,} " +
          " ".join(f"{x:>14.1%}" for x in row))

print("\nEq. 70.10 says the crossover — where attention is half the FLOPs —")
print("is at T = 6d. The columns confirm it: attention is a minority of the")
print("arithmetic at every context below that and a majority above.")
print("\nFor a 70B model that threshold is about 49,000 tokens. So at any")
print("ordinary context length a transformer is dominated by matrix")
print("multiplications against WEIGHTS, not by attention — which is the")
print("opposite of the usual framing.")

# --- but the memory crossover is much earlier -------------------------------
print("\n" + "=" * 72)
print("...but attention's MEMORY overtakes much earlier (eq. 70.11)")
print("=" * 72)
print(f"{'model':<14} {'14d/h (predicted)':>19} " +
      " ".join(f"{f'T={T}':>14} " for T in (2048, 8192, 32768)))
print(f"{'':<14} {'':>19} " +
      " ".join(f"{'attn % of act':>15}" for _ in range(3)))
for c in MODELS:
    row = []
    for T in (2048, 8192, 32768):
        per_layer_other = c.b * (10 * c.d + c.d_ff)
        per_layer_attn = c.b * c.h * T
        row.append(per_layer_attn / (per_layer_other + per_layer_attn))
    print(f"{c.name:<14} {14 * c.d // c.h:>19,} " +
          " ".join(f"{x:>15.1%}" for x in row))

print("\nEq. 70.11 puts this crossover at 14d/h, which is roughly FOURTEEN")
print("TIMES EARLIER than the FLOP crossover for a typical head count.")
print("\nThat gap is the entire reason 'attention is quadratic' is")
print("confusing. Memory goes quadratic around two thousand tokens; FLOPs")
print("only around fifty thousand. In between — which is where most models")
print("operate — attention is a MEMORY problem and not a compute problem.")
print("\nAnd that is precisely why FlashAttention's title says IO-awareness")
print("and says nothing about arithmetic: it removes the memory term and")
print("performs the same FLOPs.")

# --- the absolute numbers ---------------------------------------------------
print("\n" + "=" * 72)
print("the attention matrix, in absolute terms (section 4.4)")
print("=" * 72)
print(f"{'model':<14} {'batch':>6} " +
      " ".join(f"{f'T={T}':>13}" for T in (2048, 8192, 32768)))
for c in MODELS[2:]:
    for B in (1, 8):
        row = [c.b * B * c.L * c.h * T * T / 1e9
               for T in (2048, 8192, 32768)]
        print(f"{c.name:<14} {B:>6} " +
              " ".join(f"{x:>12,.0f}G" for x in row))

print("\nThose are the attention matrices alone, in gigabytes, if they are")
print("materialised. At a 32k context the numbers are in the hundreds of")
print("terabytes — not a memory-pressure problem, an impossibility.")
print("\nFlashAttention makes them zero by never writing the matrix out.")
print("That is the single largest term in the training-memory accounting")
print("above about two thousand tokens, and removing it is what made long")
print("contexts feasible at all.")
```

```python {tier=A name=the-6nd-rule}
"""The 6ND rule (eqs. 70.4-70.5), its correction term, and estimating a
training run.
"""
import numpy as np


def params(V, L, d, d_ff=None):
    d_ff = d_ff or 4 * d
    return 2 * V * d + L * (4 * d * d + 2 * d * d_ff)


print("=" * 72)
print("the 6ND rule and its correction (eqs. 70.4-70.5)")
print("=" * 72)
print("6ND counts only the parameter matmuls. The attention term adds a")
print("relative correction of T/(6d).\n")
print(f"{'model':<14} {'d':>7} " +
      " ".join(f"{f'T={T // 1024}k':>12}" for T in (2048, 8192, 32768, 131072)))
print(f"{'':<14} {'':>7} " +
      " ".join(f"{'6ND error':>12}" for _ in range(4)))
for name, V, L, d, dff in (("GPT-2 small", 50257, 12, 768, None),
                           ("7B", 32000, 32, 4096, 11008),
                           ("70B", 128000, 80, 8192, 28672)):
    N = params(V, L, d, dff)
    row = []
    for T in (2048, 8192, 32768, 131072):
        exact = 6 * N + 12 * L * T * d
        row.append(exact / (6 * N) - 1)
    print(f"{name:<14} {d:>7,} " + " ".join(f"{x:>11.1%}" for x in row))

print("\nAt the 2k-8k contexts most published compute figures were computed")
print("at, the correction is a few per cent and 6ND is fine. At 128k it is")
print("a factor of several and 6ND is badly wrong.")
print("\nSo quote 6ND with the regime it applies to. A long-context training")
print("run costed with plain 6ND will come in far over budget, and the")
print("error is entirely predictable from eq. 70.5.")

# --- costing a run ----------------------------------------------------------
print("\n" + "=" * 72)
print("costing a training run (section 7.3)")
print("=" * 72)
PEAK = 1e15          # effective FLOP/s per accelerator, bf16
print(f"assuming {PEAK / 1e12:.0f} TFLOP/s peak per accelerator\n")
print(f"{'model':<10} {'N':>8} {'D tokens':>10} {'C (FLOP)':>11} "
      f"{'accel-days @ MFU':>18}")
for name, V, L, d, dff, D in (("1B", 32000, 24, 2048, None, 20e9),
                              ("7B", 32000, 32, 4096, 11008, 2e12),
                              ("70B", 128000, 80, 8192, 28672, 15e12)):
    N = params(V, L, d, dff)
    C = 6 * N * D
    for mfu in (0.45,):
        days = C / (PEAK * mfu) / 86400
        print(f"{name:<10} {N / 1e9:>7.1f}B {D / 1e12:>9.1f}T "
              f"{C:>11.2e} {days:>15.0f} d")

print("\nThose are single-accelerator days; divide by the fleet size. A 70B")
print("run at 15T tokens is about 160 accelerator-years, which on 2048")
print("devices is under a month — and that arithmetic is the whole reason")
print("large-model training is a capital-expenditure question.")

# --- section 6.5: compute-optimal allocation --------------------------------
print("\n" + "=" * 72)
print("compute-optimal allocation (eq. 70.12)")
print("=" * 72)
print("Chinchilla: at a fixed TRAINING budget, N and D should both scale as")
print("sqrt(C), giving about 20 tokens per parameter.\n")
print(f"{'model':<16} {'N':>8} {'D actual':>10} {'tokens/param':>14} "
      f"{'Chinchilla D':>14} {'ratio':>8}")
HIST = [("GPT-3 (2020)", 175e9, 300e9),
        ("Chinchilla (2022)", 70e9, 1.4e12),
        ("Llama-2 7B (2023)", 7e9, 2e12),
        ("modern 8B (2024+)", 8e9, 15e12)]
for name, N, D in HIST:
    print(f"{name:<16} {N / 1e9:>7.0f}B {D / 1e12:>9.2f}T "
          f"{D / N:>14.1f} {20 * N / 1e12:>13.2f}T {D / (20 * N):>8.2f}x")

print("\nGPT-3 is at 1.7 tokens per parameter — about a twelfth of optimal,")
print("which is what Hoffmann et al. established. And modern small models")
print("are at hundreds of tokens per parameter, an order of magnitude PAST")
print("Chinchilla-optimal.")
print("\nBoth of those look like mistakes and only one is. Eq. 70.12")
print("minimises loss for a fixed TRAINING budget and says nothing about")
print("inference, which is the next table.")

# --- section 6.6: the inference-aware correction ----------------------------
print("\n" + "=" * 72)
print("why modern models are 'overtrained' (eq. 70.13)")
print("=" * 72)
print("Lifetime compute is 6*N*D_train + 2*N*D_inference. Inference is 2N")
print("per token because there is no backward pass.\n")
print(f"{'served tokens':>15} " +
      " ".join(f"{f'{n / 1e9:.0f}B model':>16}" for n in (8e9, 70e9)))
print(f"{'':>15} " +
      " ".join(f"{'train / total':>16}" for _ in range(2)))
for Dinf in (1e11, 1e13, 1e15, 1e17):
    row = []
    for N, Dtr in ((8e9, 15e12), (70e9, 15e12)):
        tr = 6 * N * Dtr
        inf = 2 * N * Dinf
        row.append(tr / (tr + inf))
    print(f"{Dinf:>15.0e} " + " ".join(f"{x:>16.1%}" for x in row))

print("\nOnce a model serves more than about three times its training")
print("tokens, inference dominates the lifetime compute — and inference")
print("cost scales with N and not with D.")
print("\nSo for a widely-served model the right move is a SMALLER model")
print("trained on MORE tokens than eq. 70.12 recommends: you pay more")
print("training compute once to pay less inference compute forever. That is")
print("the entire explanation for the shift from GPT-3's ratio to a modern")
print("8B model's, and neither is a mistake — they are optimising different")
print("objectives.")

# --- training memory --------------------------------------------------------
print("\n" + "=" * 72)
print("training memory (eq. 70.9)")
print("=" * 72)
print("Mixed precision: bf16 weights and gradients, fp32 master copy and")
print("two Adam moments. 16 bytes per parameter before any activation.\n")
print(f"{'model':<10} {'N':>8} {'weights':>9} {'grads':>8} {'Adam':>8} "
      f"{'master':>8} {'state total':>12} {'serve bf16':>11}")
for name, V, L, d, dff in (("1B", 32000, 24, 2048, None),
                           ("7B", 32000, 32, 4096, 11008),
                           ("70B", 128000, 80, 8192, 28672)):
    N = params(V, L, d, dff)
    w, g_, a, m = 2 * N, 2 * N, 8 * N, 4 * N
    print(f"{name:<10} {N / 1e9:>7.1f}B {w / 1e9:>8.1f}G {g_ / 1e9:>7.1f}G "
          f"{a / 1e9:>7.1f}G {m / 1e9:>7.1f}G {(w + g_ + a + m) / 1e9:>11.1f}G "
          f"{2 * N / 1e9:>10.1f}G")

print("\nThe last two columns are the ratio that decides infrastructure: a")
print("model needs eight times as much memory to train as to serve, before")
print("a single activation is stored.")
print("\nAnd the Adam column is half the total. That is why optimiser-state")
print("sharding is the first thing any large-scale training framework does,")
print("and why 8-bit Adam is worth the complexity — it removes a quarter of")
print("the state outright.")
```

## 9. Practical Example

```python {tier=A name=roofline-in-practice}
"""The roofline applied to each phase, and what each optimisation moves."""
import time

import numpy as np

rng = np.random.default_rng(0)

# --- section 6.4: the two phases sit at opposite ends of the roofline -------
print("=" * 72)
print("prefill and decode are opposite ends of the same roofline (6.4)")
print("=" * 72)


def intensity(flops, byts):
    return flops / byts


def phase_intensity(N, L, d, h, g, dk, T, B, phase, b=2):
    if phase == "prefill":
        fl = B * T * (2 * N + 4 * L * T * d)
        by = b * (N + B * T * L * (10 * d + 4 * d) + B * L * h * T * T)
    else:
        fl = B * (2 * N + 4 * L * g * dk * T)
        by = b * (N + 2 * B * L * g * dk * T)
    return intensity(fl, by)


N, L, d, h, g, dk = 7e9, 32, 4096, 32, 8, 128
print("7B model, GQA g=8, bf16. Ridge point is a few hundred ops/byte.\n")
print(f"{'phase':<10} {'batch':>6} " +
      " ".join(f"{f'T={T}':>13}" for T in (512, 4096, 32768)))
for phase in ("prefill", "decode"):
    for B in (1, 32):
        row = [phase_intensity(N, L, d, h, g, dk, T, B, phase)
               for T in (512, 4096, 32768)]
        print(f"{phase:<10} {B:>6} " + " ".join(f"{x:>13.1f}" for x in row))

print("\nPrefill is in the hundreds or thousands: compute-bound, the machine")
print("is doing arithmetic. Decode at batch 1 is around one: memory-bound by")
print("three orders of magnitude, the machine is waiting.")
print("\nThose are the SAME MODEL, minutes apart in the same request. No")
print("single optimisation serves both, which is why prefill and decode are")
print("increasingly scheduled — and sometimes hosted — separately.")

# --- what each optimisation moves -------------------------------------------
print("\n" + "=" * 72)
print("what each optimisation actually changes (table 70.1)")
print("=" * 72)
B, T = 8, 8192
base = {
    "param FLOPs": 2 * N * B * T,
    "attn FLOPs": 4 * L * T * T * d * B,
    "attn memory": 2 * B * L * h * T * T,
    "activation memory": 2 * B * T * L * 14 * d,
    "optimiser state": 12 * N,
    "KV cache (serving)": 2 * 2 * L * g * dk * T * B,
}
print(f"baseline, B={B}, T={T}:\n")
for k, v in base.items():
    unit = "GFLOP" if "FLOP" in k else "GB"
    print(f"  {k:<22} {v / 1e9:>12,.1f} {unit}")

OPTS = {
    "FlashAttention": {"attn memory": 0.0},
    "GQA g=8 (already on)": {"KV cache (serving)": 1.0},
    "gradient checkpointing": {"activation memory": 0.15},
    "8-bit Adam": {"optimiser state": 0.5},
    "sliding window w=1024": {"attn FLOPs": 1024 / T,
                              "attn memory": 1024 / T,
                              "KV cache (serving)": 1024 / T},
    "int8 KV cache": {"KV cache (serving)": 0.5},
}
print(f"\n{'optimisation':<24} " +
      " ".join(f"{k.split()[0][:9]:>11}" for k in base))
for name, effect in OPTS.items():
    row = []
    for k in base:
        f = effect.get(k, 1.0)
        row.append("—" if f == 1.0 else
                   "0" if f == 0.0 else f"{f:.2f}x")
    print(f"{name:<24} " + " ".join(f"{v:>11}" for v in row))

print("\nEvery column is one line of the accounting and every row touches")
print("one or two of them. That is the point of building the table before")
print("the techniques: FlashAttention is not 'making attention faster', it")
print("is zeroing exactly one term, and it leaves the FLOPs and the KV cache")
print("untouched.")
print("\nAnd it explains why the techniques compose: they act on different")
print("terms. Applying all of them is not redundant — each removes a")
print("different bottleneck, and which one is binding depends on B, T and")
print("whether you are training or serving.")

# --- measure the elementwise gap --------------------------------------------
print("\n" + "=" * 72)
print("why measured MFU tops out well below 100% (section 7.1)")
print("=" * 72)
print("6ND counts matmuls. Everything else is cheap in FLOPs and not cheap")
print("in TIME, because it is bandwidth-bound.\n")
dd, TT, BB = 512, 256, 4
X = rng.normal(size=(BB, TT, dd)).astype(np.float32)
W1 = rng.normal(0, 0.02, (dd, 4 * dd)).astype(np.float32)
W2 = rng.normal(0, 0.02, (4 * dd, dd)).astype(np.float32)
Wq = rng.normal(0, 0.02, (dd, dd)).astype(np.float32)


def timeit(fn, reps=20):
    fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


t_mm = timeit(lambda: np.maximum(0.0, X @ W1) @ W2)
t_only_mm = timeit(lambda: (X @ W1) @ W2)
t_norm = timeit(lambda: X / np.sqrt((X ** 2).mean(-1, keepdims=True) + 1e-6))
t_proj = timeit(lambda: X @ Wq)

mm_flops = 2 * BB * TT * dd * 4 * dd * 2
print(f"{'operation':<28} {'ms':>9} {'GFLOP':>9} {'GFLOP/s':>10}")
print(f"{'FFN matmuls only':<28} {t_only_mm * 1e3:>9.3f} "
      f"{mm_flops / 1e9:>9.3f} {mm_flops / t_only_mm / 1e9:>10.1f}")
print(f"{'FFN with ReLU':<28} {t_mm * 1e3:>9.3f} "
      f"{mm_flops / 1e9:>9.3f} {mm_flops / t_mm / 1e9:>10.1f}")
print(f"{'RMSNorm alone':<28} {t_norm * 1e3:>9.3f} "
      f"{3 * X.size / 1e9:>9.5f} {3 * X.size / t_norm / 1e9:>10.2f}")
print(f"{'one projection':<28} {t_proj * 1e3:>9.3f} "
      f"{2 * BB * TT * dd * dd / 1e9:>9.3f} "
      f"{2 * BB * TT * dd * dd / t_proj / 1e9:>10.1f}")

print(f"\nReLU alone costs {(t_mm - t_only_mm) * 1e3:.3f} ms and "
      f"{X.size * 4 / 1e9:.4f} GFLOP.")
print(f"RMSNorm reaches {3 * X.size / t_norm / 1e9:.2f} GFLOP/s against the")
print(f"matmuls' {mm_flops / t_only_mm / 1e9:.0f} — two orders of magnitude")
print("apart, on the same hardware, in the same model.")
print("\nThat gap is where MFU goes. The elementwise operations contribute")
print("almost nothing to the 6ND count and a real fraction of the wall")
print("clock, because they are bandwidth-bound and the matmuls are not.")
print("\nSo a reported MFU of 45% is not 55% of the machine sitting idle. It")
print("is mostly this, plus communication, and closing it is a kernel-fusion")
print("problem rather than a scheduling one (Chapter 51).")

# --- putting it together: can this run fit? ---------------------------------
print("\n" + "=" * 72)
print("the check people skip: does it FIT? (section 7.3, step 5)")
print("=" * 72)


def training_memory_gb(N, L, d, h, d_ff, B, T, b=2, flash=True,
                       checkpoint=False):
    state = 16 * N
    act_per_layer = b * B * T * (10 * d + d_ff)
    attn = 0.0 if flash else b * B * L * h * T * T
    act = L * act_per_layer + attn
    if checkpoint:
        act = act * 0.15 + act_per_layer * np.sqrt(L)
    return (state + act) / 1e9


N7, L7, d7, h7, dff7 = 7e9, 32, 4096, 32, 11008
print("7B model on one 80 GB accelerator, mixed precision.\n")
print(f"{'batch':>6} {'T':>7} {'flash':>7} {'ckpt':>6} {'memory':>10} "
      f"{'fits 80G?':>11}")
for B_ in (1, 4, 16):
    for T_ in (2048, 8192):
        for flash in (False, True):
            for ck in (False, True):
                m = training_memory_gb(N7, L7, d7, h7, dff7, B_, T_,
                                       flash=flash, checkpoint=ck)
                print(f"{B_:>6} {T_:>7,} {str(flash):>7} {str(ck):>6} "
                      f"{m:>9.1f}G {('yes' if m < 80 else 'NO'):>11}")

print("\nThe state alone is 112 GB, so a 7B model does not train on one")
print("80 GB device under ANY of these settings — which is the answer step")
print("5 of section 7.3 is supposed to produce before anyone estimates a")
print("wall-clock time.")
print("\nRead the rows against each other anyway: FlashAttention's effect")
print("grows with T squared and checkpointing's is roughly constant, so")
print("which one you need depends entirely on the context length. That is")
print("the accounting doing its job — telling you which term is binding")
print("before you pick a technique.")
```

## 10. Production Considerations

**Check memory before estimating time.** Measured: a 7B model's optimiser state
alone exceeds an 80 GB device, so no throughput calculation matters until
sharding is in the plan.

**Quote $6ND$ with its regime.** Measured: the correction is a few per cent at
2–8k context and a factor of several at 128k.

**Separate the FLOP crossover from the memory crossover.** Measured about
fourteen times apart. Attention is a memory problem long before it is a compute
problem.

**Report MFU, and expect 40–55%.** Measured: the elementwise operations run two
orders of magnitude below the matmuls' rate and contribute almost nothing to the
FLOP count.

**Pack short sequences rather than padding to a long one.** Attention memory is
$hT^2$ per sequence, so $B$ short ones are quadratically cheaper — provided the
mask prevents cross-sequence attention.

**Cost the lifetime, not the training run.** Measured: past about three times
the training tokens, inference dominates, and inference scales with $N$ alone.

## 11. Common Mistakes

**Saying "attention is quadratic" without saying in what.** Measured: memory
always, FLOPs only past $T = 6d$.

**Using $6ND$ at long context.** Measured error.

**Sizing a training job by parameter count.** Measured: state is $16N$, eight
times the serving footprint.

**Assuming FlashAttention fixes the KV cache.** It zeroes one term in the
training accounting and touches nothing at serving time.

**Benchmarking prefill and reporting it as throughput.** Measured: the two
phases differ by three orders of magnitude in arithmetic intensity.

**Treating a 45% MFU as 55% waste.** Measured: most of it is bandwidth-bound
work the FLOP count does not see.

## 12. Failure Modes

**Out-of-memory at a longer context**, with everything else unchanged. The
$hT^2$ term grows fastest.

**A training run costing several times its estimate.** Usually plain $6ND$ at
long context, or unaccounted gradient checkpointing at $8ND$.

**Throughput collapsing at a longer sequence** even when memory fits, because
attention's FLOPs crossed $T = 6d$.

**Serving capacity failing at concurrency** while single-request latency looks
fine ({{ch:tf-masking-kv}}).

**MFU dropping after a change** that added elementwise work — a new
normalisation, an unfused activation — with no change in the FLOP count.

## 13. Alternatives

**Mixture of experts** breaks the $2N$-per-token identity: parameters grow and
active parameters per token do not, so $6ND$ uses the *active* count
({{ch:res-moe}}).

**Sparse and linear attention** attack the $4LT^2d$ term directly
({{ch:tf-efficient}}).

**Gradient checkpointing** trades the activation term for a fourth forward pass,
$6ND \to 8ND$ ({{ch:dl-backprop}}).

**Lower precision** halves every memory and bandwidth term and roughly doubles
effective peak FLOPs ({{part:15}}).

**Distributed training** partitions each term differently — data parallelism
replicates the state, tensor parallelism splits it, pipeline parallelism splits
the layers ({{ch:inf-parallelism}}).

## 14. Evaluation

**Compute all four numbers before running anything**: FLOPs, bandwidth,
capacity, depth.

**Measure MFU and compare against 40–55%.**

**Profile the elementwise fraction.** Measured to be substantial and invisible
in the FLOP count.

**Sweep the context length and find your own crossovers.** They depend on $d$
and $h$ and the derived formulae give the right order.

**Check the memory estimate against actual peak usage.** A large discrepancy
means something is being retained that should not be
({{ch:dl-forward}}).

## 15. Advanced Concepts

**Scaling laws with inference.** {{eq:lifetime-cost}} is the simplest version;
richer treatments include quantisation, distillation and the possibility of
serving a distilled model.

**MFU against HFU.** Hardware FLOPs utilisation counts recomputation from
gradient checkpointing; model FLOPs utilisation does not. A system reporting HFU
looks better and is measuring something less useful.

**Communication in the accounting.** At scale, all-reduce bandwidth becomes a
fifth term and often the binding one, which is why
{{ch:inf-parallelism}}'s partitioning choices matter more than the arithmetic.

**Arithmetic intensity as a design target.** Architectures can be chosen to
raise it — grouped-query attention does exactly this
({{ch:tf-multi-head}}) — which is designing for the roofline rather than around
it.

**The critical batch size.** Beyond some batch size, more parallelism stops
buying faster convergence ({{ch:dl-lr-schedules}}), which puts a ceiling on how
much of the compute term can be traded for wall-clock.

## 16. Connection to Previous Chapters

{{ch:dl-forward}}'s roofline is the entire analytical framework here, and
{{ch:dl-backprop}}'s three-times rule is where the 6 in $6ND$ comes from.
{{ch:tf-ffn-residual}}'s parameter split gives $24d^2$ per block, and
{{ch:tf-multi-head}}'s attention cost gives $4Td$.
{{ch:tf-masking-kv}}'s decode analysis is the second half of
{{sec:6-mathematical-foundation}}'s roofline comparison, and its
{{eq:cache-size-full}} is one row of {{tbl:transformer-complexity}}.

Forward: {{ch:tf-efficient}} attacks the rows of that table one at a time, and
{{ch:fm-scaling-laws}} develops {{eq:chinchilla}} properly.
{{part:23}} builds the systems that make the numbers achievable.

## 17. Exercises

**Beginner**

1. Why does each parameter cost 2 FLOPs per token?
2. Where does the 6 in $6ND$ come from?
3. Which term in a transformer is genuinely quadratic in $T$?
4. How much memory does Adam need per parameter?
5. Why is decoding memory-bound?

**Intermediate**

6. Derive {{eq:flop-crossover}} and evaluate it for $d = 2048$.
7. Derive {{eq:memory-crossover}} and evaluate for $d=4096$, $h=32$.
8. Use {{eq:6nd-with-attention}} to find the correction at $T=65536$,
   $d=8192$.
9. Compute the training memory for a 13B model at $B=4$, $T=8192$, with and
   without FlashAttention.
10. Explain why packing short sequences beats one long sequence.

**Advanced**

11. Derive {{eq:lifetime-cost}} and find the $N$ minimising it for
    $D_{\text{inf}} = 10^{15}$.
12. Derive the arithmetic intensity of prefill and show it is compute-bound.
13. Account for gradient checkpointing exactly and derive the $8ND$ figure.
14. Extend {{tbl:transformer-complexity}} with a communication row for
    data-parallel training.

**Implementation**

15. Write a FLOP and memory calculator for an arbitrary configuration and
    validate it against a framework's profiler.
16. Measure MFU on a real training step and account for the gap.
17. Reproduce the crossover tables for your own architecture.
18. Implement the memory estimator and compare against measured peak usage.

**Reasoning**

19. Your run fits at $T=4096$ and fails at $T=8192$ with the same batch size.
    Which term, and what do you do?
20. A vendor quotes a training cost using $6ND$ for a 200k-context model. By
    how much are they wrong?

## 18. Interview Questions

**"How many FLOPs to train a model?"** — $6ND$, with the derivation and the
long-context caveat.

**"Is attention quadratic?"** — In memory always, in FLOPs past $T = 6d$. Give
both crossovers and note they differ by an order of magnitude.

**"How much memory to train a 7B model?"** — $16N$ of state plus activations.
Give the number and note it is eight times the serving footprint.

**"What does FlashAttention change?"** — Zeroes one term in the memory
accounting. Say what it leaves alone.

**"What is Chinchilla-optimal and why do modern models ignore it?"** — 20 tokens
per parameter minimises loss at fixed training compute; inference cost scales
with $N$, so a served model should be smaller and trained longer.

**"Why is MFU only 45%?"** — Bandwidth-bound elementwise work and communication,
neither of which the FLOP count sees.

## 19. Research Questions

**What is the right inference-aware scaling law?**
{{eq:lifetime-cost}} is a first approximation and the interaction with
distillation and quantisation is not well characterised.
{{maturity:EMERGING}}

**Can the quadratic memory term be removed architecturally?** FlashAttention
removes it from the implementation; whether an architecture that never needs it
can match full attention is {{ch:tf-efficient}}'s open question.
{{maturity:EMERGING}}

**How much of the MFU gap is irreducible?** Fusion closes part of it and the
floor is not established. {{maturity:EMERGING}}

**Do scaling laws hold at long context?** They were fitted at 1–2k contexts and
{{eq:6nd-with-attention}} says the compute accounting itself changes.
{{maturity:EMERGING}}

## 20. Chapter Summary

Every cost in a transformer is one of four quantities — FLOPs, memory bandwidth,
memory capacity, critical-path depth — and they fail independently. Naming which
is binding is most of the skill; the rest is arithmetic.

The arithmetic starts from one identity: **every parameter participates in
exactly one multiply-accumulate per token**, so the forward pass costs $2N$ and
the backward twice that. That gives $6ND$, derived in four lines and applicable
to any dense architecture. Measured, its correction term $T/6d$ is a few per
cent at the 2–8k contexts most published figures were computed at, and a factor
of several at 128k — so the rule is long-context-blind and should be quoted with
its regime.

The central confusion this chapter resolves is what "attention is quadratic"
means. Measured, attention's FLOPs cross the parameter FLOPs at $T = 6d$ —
around 49,000 tokens for a 70B model — so at any ordinary context a transformer
is dominated by matrix multiplications against weights. But attention's *memory*
crosses at $T \approx 14d/h$, about **fourteen times earlier**, roughly two
thousand tokens. In between, which is where most models operate, **attention is
a memory problem and not a compute problem.** That gap is exactly why
{{cite:dao2022flash}}'s title says IO-awareness: it zeroes one memory term and
performs the identical arithmetic.

The absolute numbers make the point unarguable. Measured, the materialised
attention matrices for a 70B model at a 32k context run to hundreds of terabytes
— not memory pressure, impossibility — and FlashAttention makes them zero.

Training memory is $16N$ bytes of state before any activation: weights,
gradients, an fp32 master copy and two Adam moments. Measured, that is eight
times the serving footprint, and Adam alone is half of it — which is why
optimiser-state sharding is the first thing every large-scale framework does.
Measured on one 80 GB device, a 7B model does not train under any combination of
FlashAttention and checkpointing, which is the check that belongs *before* any
throughput estimate.

The roofline places the two phases at opposite extremes of the same model.
Measured, prefill's arithmetic intensity is in the hundreds or thousands and
decode's is about one at batch 1 — three orders of magnitude apart, minutes
apart in the same request. No single optimisation serves both.

Finally, {{eq:chinchilla}}'s 20 tokens per parameter minimises loss at fixed
*training* compute, and measured against {{eq:lifetime-cost}}, inference
dominates once a model serves more than about three times its training tokens.
Since inference scales with $N$ and not $D$, a widely-served model should be
smaller and trained longer than Chinchilla-optimal. GPT-3 at 1.7 tokens per
parameter and a modern 8B at nearly 2,000 are not one mistake and one
correction — **they are optimising different objectives**, and knowing which one
you are optimising is the point of the whole accounting.

## 21. Further Reading

{{cite:kaplan2020scaling}} and {{cite:hoffmann2022chinchilla}} should be read as
a pair and in order. The first established that loss follows clean power laws in
$N$, $D$ and $C$; the second found that the first's recommended allocation was
wrong, and traced the error to a learning-rate schedule that was not adapted to
each run's length. That is an unusually clean example of a methodological detail
overturning a headline conclusion, and it is worth knowing about for reasons
beyond scaling laws.

{{cite:dao2022flash}} again, read here for its cost analysis rather than its
algorithm. The paper's framing — that the field had been counting FLOPs when the
binding constraint was memory traffic — is the argument this chapter is built
around, and section 2 of the paper makes it better than any secondary source.

{{cite:vaswani2017}}'s table 1 is where the complexity comparison starts, and it
is worth returning to now that the numbers mean something concrete.

**On the roofline model** generally, the original is from high-performance
computing rather than machine learning, and it is worth reading in that context:
the deep learning application is unusually clean because the two regimes are so
sharply separated, but the tool is general.

**Where to go next:** {{ch:tf-efficient}} takes {{tbl:transformer-complexity}}
row by row. With the accounting in hand, each technique is an obvious response
to a specific term rather than an item on a list.

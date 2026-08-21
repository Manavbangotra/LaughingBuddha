---
id: tf-ffn-residual
number: 67
part: VII
tier: full
status: reviewed
requires: [tf-multi-head, tf-embeddings, dl-normalization, dl-activations, dl-cnns]
provides: [feed-forward-network, residual-stream, norm-placement,
           swiglu, expansion-ratio, key-value-memory, block-parameter-count]
citations: [vaswani2017, shazeer2020glu, xiong2020prenorm, zhang2019rmsnorm,
            he2016resnet]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Account for every parameter in a transformer block and say where the
   majority is.
2. Explain the residual stream and why it is the right way to think about a
   transformer.
3. Derive pre-norm and post-norm and explain why the field moved.
4. Explain the gated feed-forward block and its parameter accounting.
5. Explain why the expansion ratio is 4 and what happens when it is not.
6. Explain the key–value memory reading of the feed-forward block, and how
   strong the evidence is.
7. Diagnose a residual-stream problem from measured norms.

## 2. Why This Matters

**Two-thirds of a transformer's parameters are in the feed-forward blocks, not
in attention.** {{sec:6-mathematical-foundation}} does the accounting, and it is
the single most surprising number in this part for anyone whose mental model
came from a diagram — where attention occupies most of the picture.

**The residual stream is the right abstraction.** A transformer is not a stack
of transformations applied in sequence; it is a *shared channel* that every
block reads from and adds to. Once you see it that way, the logit lens
({{ch:tf-embeddings}}), head-level analysis ({{ch:tf-multi-head}}) and
normalisation placement all become obvious rather than arbitrary.

**Pre-norm versus post-norm decided whether deep transformers train at all.**
{{cite:xiong2020prenorm}} showed post-norm has badly scaled gradients at
initialisation and needs warmup, while pre-norm does not. Every model since
2020 uses pre-norm, and the reason is a two-line Jacobian argument.

**The feed-forward block is where the nonlinearity is.** Attention is a convex
combination of value vectors with linear projections around it — the only
nonlinearity in the whole mechanism is the softmax that produces the weights.
Everything elementwise happens here.

## 3. Prerequisites

{{ch:tf-multi-head}} for the attention block this alternates with, and for
{{eq:mha-sum}} — the observation that heads *add* to the stream, which is what
this chapter generalises. {{ch:dl-normalization}} for layer normalisation and
{{eq:norm-scale-invariance}}. {{ch:dl-activations}} for GELU and SiLU.
{{ch:dl-cnns}} for residual connections.

## 4. Intuitive Explanation

### 4.1 The block

```text
   PRE-NORM (everything since ~2020)

   x ──┬──────────────────────────────────┐
       │                                  │
       └──▶ Norm ──▶ Attention ──────────▶ + ──┬────────────────┐
                                               │                │
                                               └──▶ Norm ──▶ FFN ──▶ + ──▶
```

Two sublayers, each wrapped in a residual connection with a normalisation on
the *branch*. Nothing sits on the skip path.

### 4.2 The residual stream

The skip path is not a shortcut around the computation. **It is the
computation's medium.**

```text
   layer 0     x₀ = embedding + position
   layer 1     x₁ = x₀ + attn(x₀) + ffn(...)
   layer 2     x₂ = x₁ + attn(x₁) + ffn(...)
   ...
   layer L     x_L = x₀ + Σ (every block's output)
```

Unrolled, $\vec{x}_L$ is the embedding **plus the sum of every sublayer's
output**. No block replaces what came before; each one adds to a running total.

Three things follow immediately, and they are the reason this framing is worth
adopting.

**Every layer's output lives in the same space.** That is what makes the logit
lens type-correct ({{ch:tf-embeddings}}) — you can apply the unembedding at any
depth because every depth is the same kind of object.

**Blocks communicate by writing and reading directions.** A block can write
information into some subspace and a later block can read it out, without any
intervening block being involved. The stream is a bus, not a pipeline.

**Depth is additive, not compositional.** A 32-layer model is not 32 nested
functions; it is 64 sublayer outputs summed. That is why deleting one block from
a trained transformer usually changes the output far less than deleting one
layer from a plain stack.

### 4.3 The feed-forward block

Applied independently at each position — it does no mixing across the sequence
at all:

$$
\FFN(\vec{x}) = \mat{W}_2\,\phi(\mat{W}_1\vec{x})
$$

with $\mat{W}_1: d \to 4d$ and $\mat{W}_2: 4d \to d$. Expand fourfold, apply an
elementwise nonlinearity, project back.

**This is where the parameters are.** Attention has $4d^2$; the feed-forward
block has $8d^2$. Two-thirds of every block, and — since the blocks are most of
a large model — roughly two-thirds of the whole thing.

### 4.4 What it might be doing

The suggestive reading, and it is worth having while being clear about its
status.

$\mat{W}_1$'s rows act as *keys*: $\mat{W}_1\vec{x}$ scores the input against
each of $4d$ directions. The activation gates those scores. $\mat{W}_2$'s
columns act as *values*: whichever directions survived get added to the output.

```text
   W₁ row i  ──▶  "does the input look like PATTERN i?"
   activation ──▶  "only if strongly"
   W₂ col i  ──▶  "then add VECTOR i to the stream"
```

On that reading the block is an associative memory with $4d$ slots, addressed by
content and read by summation. Evidence exists — individual neurons that fire on
recognisable patterns, and edits to $\mat{W}_2$ rows that change specific
factual outputs — and it is partial. Most neurons are not interpretable, and the
polysemantic ones are the norm rather than the exception.

### 4.5 Gating

Modern models replace the two-matrix block with three:

$$
\FFN_{\text{gated}}(\vec{x})
 = \mat{W}_2\big(\phi(\mat{W}_g\vec{x}) \odot \mat{W}_1\vec{x}\big)
$$

One projection is passed through the nonlinearity and *multiplies* another. The
elementwise product is the point: it makes the block's output depend
multiplicatively on two learned projections rather than additively on one.

Three matrices instead of two means the hidden width is reduced to $\tfrac{8}{3}d$
to keep the parameter count the same, and {{cite:shazeer2020glu}} reports the
swap improves quality at matched size.

## 5. Formal Explanation

### 5.1 The pre-norm block

$$
\vec{x}' = \vec{x} + \MHA\big(\Norm(\vec{x})\big)
$$ (eq:prenorm-attn)

$$
\vec{x}'' = \vec{x}' + \FFN\big(\Norm(\vec{x}')\big)
$$ (eq:prenorm-ffn)

with a final $\Norm$ before the unembedding. Post-norm instead applies the
normalisation *after* the addition:

$$
\vec{x}' = \Norm\big(\vec{x} + \MHA(\vec{x})\big)
$$ (eq:postnorm)

**The difference is whether a normalisation sits on the skip path.** In
{{eq:prenorm-attn}} it does not, so a gradient can travel from the loss to any
layer through pure additions. In {{eq:postnorm}} it does, at every layer.

### 5.2 The feed-forward block

$$
\FFN(\vec{x}) = \mat{W}_2\,\phi\big(\mat{W}_1\vec{x} + \vec{b}_1\big)
 + \vec{b}_2
$$ (eq:ffn)

with $\mat{W}_1 \in \R^{d_{\text{ff}}\times d}$,
$\mat{W}_2 \in \R^{d\times d_{\text{ff}}}$, and conventionally
$d_{\text{ff}} = 4d$. Modern models drop both biases.

Gated:

$$
\FFN_{\text{SwiGLU}}(\vec{x})
 = \mat{W}_2\Big(\silu(\mat{W}_g\vec{x}) \odot \mat{W}_1\vec{x}\Big)
$$ (eq:swiglu)

with $\silu(z) = z\sigma(z)$ ({{ch:dl-activations}}).

### 5.3 The parameter accounting

Per block:

$$
P_{\text{attn}} = 4d^2,
\qquad
P_{\text{ffn}} = 2 d\,d_{\text{ff}} = 8d^2 \ \ (d_{\text{ff}}=4d)
$$ (eq:block-params-split)

$$
P_{\text{block}} = 12d^2,
\qquad
\frac{P_{\text{ffn}}}{P_{\text{block}}} = \frac{2}{3}
$$ (eq:ffn-fraction)

Gated, at three matrices of width $d_{\text{ff}}$:

$$
P_{\text{ffn,gated}} = 3d\,d_{\text{ff}}
\quad\Longrightarrow\quad
d_{\text{ff}} = \tfrac{8}{3}d\ \text{for parity}
$$ (eq:swiglu-width)

which is why Llama-style models use $d_{\text{ff}} \approx 2.7d$ and round it to
a multiple of 128 or 256 for hardware alignment.

### 5.4 Where the FLOPs are

Per position:

$$
F_{\text{attn}} = 8d^2 + 4Td,
\qquad
F_{\text{ffn}} = 16d^2
$$ (eq:block-flops)

**The feed-forward block is two-thirds of the arithmetic too, until $T$ is
large.** The crossover, where attention's quadratic term matches the
feed-forward cost, is at $T = 4d$ — the same threshold as
{{ch:tf-why-attention}}'s, for the same reason.

### 5.5 Normalisation choice

{#tbl:norm-in-transformers caption="Normalisation in transformers. The last column is what actually drove adoption: RMSNorm is cheaper and empirically equal, and pre-norm is what makes deep models trainable without warmup."}

| Choice | Formula | Cost | Why |
|---|---|---|---|
| LayerNorm | subtract mean, divide by sd, scale, shift | 2 passes, $2d$ params | original |
| RMSNorm | divide by RMS, scale | 1 pass, $d$ params | cheaper, equal |
| Post-norm | after the residual add | — | original; needs warmup |
| Pre-norm | on the branch only | — | trains at depth |

The 2026 default is **RMSNorm, pre-norm**, plus a final normalisation before the
unembedding. That final one is not optional: without it the residual stream's
norm — which grows with depth ({{sec:6-mathematical-foundation}}) — feeds
directly into the logits.

### 5.6 Why the block does no mixing across positions

Worth stating explicitly, because it is the cleanest division of labour in the
architecture and it is easy to lose.

$\FFN$ is applied **independently at every position**. Position $i$'s output
depends on position $i$'s input and on nothing else. There is no sum over $j$,
no mask, no sequence dimension in either matrix.

So a transformer block does exactly two things, and they are cleanly separated:

```text
   attention     mixes ACROSS positions, linear in the values
   feed-forward  processes EACH position, nonlinear
```

Three consequences follow, and each answers a question that otherwise looks
arbitrary.

**Why both are needed.** Attention with no feed-forward block is, given the
weights, a linear map — a convex combination of linear projections. Stacking
such layers gets you {{ch:dl-neural-networks}}'s collapse: composition of linear
maps is a linear map, modulated only by the softmax. The feed-forward block is
where elementwise nonlinearity enters, and attention-only transformers are
correspondingly much worse.

**Why the feed-forward block is trivially parallel.** No cross-position
dependency means it shards across the sequence with no communication at all,
which is why sequence and context parallelism ({{part:23}}) are cheap for
two-thirds of the model and expensive only for attention.

**Why it dominates the cost at short sequences.** Its work is $O(Td^2)$ against
attention's $O(Td^2 + T^2d)$, and the second term only wins past $T = 4d$.

## 6. Mathematical Foundation

### 6.1 The residual stream, unrolled

From {{eq:prenorm-attn}} and {{eq:prenorm-ffn}}, applied $L$ times:

$$
\vec{x}_L = \vec{x}_0
 + \sum_{\ell=1}^{L}\Big[
 \MHA_\ell\big(\Norm(\vec{x}_{\ell-1})\big)
 + \FFN_\ell\big(\Norm(\vec{x}'_{\ell-1})\big)\Big]
$$ (eq:residual-stream)

Combining with {{eq:mha-sum}} from {{ch:tf-multi-head}}, which expands each
$\MHA$ into a sum over heads:

$$
\vec{x}_L = \vec{x}_0
 + \sum_{\ell}\sum_{i}\head_{\ell,i}\mat{W}^O_{\ell,i}
 + \sum_{\ell}\FFN_\ell(\cdot)
$$ (eq:full-decomposition)

**A transformer's output is the embedding plus a sum of $L(h+1)$ terms**, each
produced by one head or one feed-forward block. That is the residual-stream view
stated exactly, and it is the foundation of every mechanistic-interpretability
result in the literature.

### 6.2 Why pre-norm trains and post-norm needs warmup

Differentiate {{eq:prenorm-attn}}:

$$
\frac{\partial\vec{x}'}{\partial\vec{x}}
 = \mat{I} + \frac{\partial \MHA}{\partial \Norm}
 \frac{\partial \Norm}{\partial\vec{x}}
$$ (eq:prenorm-jacobian)

The identity term is **exact and untouched by the normalisation**. Over $L$
layers the product $\prod(\mat{I}+\mat{J}_\ell)$ expands with the identity as
its first term ({{ch:dl-cnns}}'s {{eq:residual-expansion}}), so a gain-1 path
exists to every layer.

Now post-norm:

$$
\frac{\partial\vec{x}'}{\partial\vec{x}}
 = \frac{\partial \Norm}{\partial(\cdot)}
 \left(\mat{I} + \frac{\partial \MHA}{\partial\vec{x}}\right)
$$ (eq:postnorm-jacobian)

**The normalisation's Jacobian multiplies everything, including the identity.**
Its magnitude is roughly $1/\|\vec{x}\|$, which is not 1, so the product over
$L$ layers accumulates $L$ such factors and the gain-1 path is gone.

{{cite:xiong2020prenorm}} makes this precise with a mean-field analysis and
finds post-norm's gradient at initialisation scales badly with depth — which is
exactly what warmup ({{ch:dl-lr-schedules}}) is for: survive the early steps
until the parameters move somewhere the scaling is less hostile.

**Pre-norm needs no warmup and post-norm does.** That is the finding, and it is
a Jacobian argument rather than an empirical preference.

### 6.3 Why the residual norm grows with depth

Each sublayer adds its output to the stream. If those outputs are roughly
independent of the stream and of each other, variances add:

$$
\Var[\vec{x}_\ell] \approx \Var[\vec{x}_0] + \sum_{k\le\ell}\Var[\text{block}_k]
$$ (eq:residual-growth)

so the norm grows as $\sqrt{\ell}$ under pre-norm — the branch outputs are
normalised inputs, so their scale does not itself grow.

Two consequences.

**Later blocks contribute proportionally less.** A block adding a fixed-scale
vector to a stream whose norm grows as $\sqrt{\ell}$ changes the *direction* of
the stream by an angle that shrinks with depth. Deep pre-norm models show
measurably smaller per-layer changes in the later layers, and this is why.

**The final normalisation is mandatory.** Without it, the unembedding sees a
vector whose norm depends on $L$, so the logit scale would change with depth.
{{sec:8-implementation}} measures the growth.

### 6.4 Why the expansion ratio is 4

There is no derivation. $d_{\text{ff}} = 4d$ appears in
{{cite:vaswani2017}} without justification and has been kept because it works.

Three partial arguments exist and none is decisive:

**Capacity.** A wider hidden layer gives more of the associative-memory slots of
{{sec:4-intuitive-explanation}}. But 4 rather than 2 or 8 is not derived from
anything.

**Ratio to attention.** At 4, the feed-forward block is exactly twice
attention's parameter count, giving the two-thirds split. That is a consequence
of the choice, not a reason for it.

**Hardware.** $4d$ keeps both matrices well-shaped for tiling. This is real and
it explains why the value is *stable*, not why it is 4.

**The honest statement is that the ratio is a convention with weak
justification**, and models that deviate — some use 8/3 with gating, some use
larger ratios in mixture-of-experts layers — do fine.
{{sec:9-practical-example}} measures the sensitivity, and the result is that
there is a broad plateau.

### 6.5 What gating buys, structurally

Compare the two blocks at a fixed parameter budget. Ungated, the output is

$$
\mat{W}_2\,\phi(\mat{W}_1\vec{x})
$$

— a nonlinear function applied to *one* projection. Gated,

$$
\mat{W}_2\big(\phi(\mat{W}_g\vec{x})\odot\mat{W}_1\vec{x}\big)
$$

— a *product* of two projections, one of them passed through a nonlinearity.

The product is the structural difference. A product of two linear functions of
$\vec{x}$ is quadratic in $\vec{x}$, so a gated block can represent second-order
interactions between input dimensions that an ungated block reaches only through
the activation's curvature.

Whether that is why it works is not established. {{cite:shazeer2020glu}}'s own
conclusion is notably candid: the paper reports the improvement and offers no
mechanism, attributing the result to divine benevolence. That is a joke in the
paper and it is also an accurate summary of the state of the explanation.

## 7. Internal Mechanics

### 7.1 The block in code

```text
   # pre-norm, gated, no biases — the 2026 default
   h = x + attn(rmsnorm(x))
   x = h + w2( silu(wg(rmsnorm(h))) * w1(rmsnorm(h)) )
```

Note that `rmsnorm(h)` is computed once and used twice in the gated block. Naive
implementations compute it twice, which is a measurable waste on a
memory-bound operation.

### 7.2 Fusing the gate projections

$\mat{W}_g$ and $\mat{W}_1$ have identical shapes and consume the same input, so
they are concatenated into one $d \times 2d_{\text{ff}}$ matmul and split
afterwards. Same arithmetic, one kernel launch, better intensity
({{ch:dl-forward}}).

Checkpoints therefore sometimes ship a fused `gate_up_proj` and sometimes
separate `gate_proj` and `up_proj`, which is a portability nuisance and nothing
more.

### 7.3 Where the activations live

The feed-forward block's intermediate is $d_{\text{ff}} = 4d$ wide per position,
so storing it for the backward pass costs $4\times$ the residual stream's
memory. For a batch of $B$ sequences of length $T$:

$$
M_{\text{ffn act}} = B\,T\,d_{\text{ff}}\,b \ \text{bytes per block}
$$ (eq:ffn-activation-memory)

**This is usually the largest single activation tensor in a transformer**, which
makes the feed-forward block the primary target for gradient checkpointing
({{ch:dl-backprop}}) — it is cheap to recompute, being two matmuls and an
elementwise operation.

### 7.4 Biases

Modern models omit biases in the feed-forward and attention projections. Two
reasons, and the second is the real one.

The normalisation immediately before each projection removes the mean, so a bias
on the *input* side is partly redundant ({{ch:dl-normalization}}).

And they cost a kernel launch and a memory pass for $d$ parameters out of
$12d^2$ — a rounding error in capacity for a measurable fraction of the
elementwise time. Removing them was found not to hurt, so they went.

### 7.5 Depth against width

At a fixed parameter budget $P \approx 12Ld^2$, you can spend it on layers or on
width. The empirical finding is a broad optimum with a mild preference for
depth, and two hard constraints at the edges: very deep narrow models are hard
to train ({{ch:dl-backprop}}) and very wide shallow ones underperform at equal
parameters.

The practical constraint is different from the quality one. **Depth is
sequential at inference** — layer $\ell+1$ needs layer $\ell$ — so a deep model
has a longer critical path per token, which matters for latency and for pipeline
parallelism ({{part:23}}).

### 7.6 What a block costs to serve, per token

The training accounting is per sequence; generation is per token, and the shape
of the cost changes completely.

$$
\text{FLOPs per token} = 24d^2 + 4Td,
\qquad
\text{bytes read per token} = 12d^2 b
$$ (eq:block-decode-cost)

Divide: the arithmetic intensity of the weights is $2/b$ — **two operations per
byte at batch size 1**, which {{ch:dl-forward}}'s roofline puts two orders of
magnitude below any modern accelerator's ridge point.

**So single-stream generation is entirely memory-bound, and it is bound by
reading the weights, not by the attention.** Every parameter of the model is
read from memory to produce one token. A 70B model in bf16 reads 140 GB per
token, and at a realistic 3 TB/s that is 47 ms per token before any arithmetic
happens at all.

Two things follow that shape all of {{part:23}}.

**Batching is not an optimisation, it is the only lever.** With $B$ concurrent
sequences the same weight read serves $B$ tokens, so the intensity becomes
$2B/b$ and the machine starts doing arithmetic. This is
{{ch:dl-forward}}'s batching argument arriving where it matters most.

**Quantisation buys speed, not just memory.** Halving $b$ halves the bytes read
and therefore roughly halves the time per token, which is why
{{part:15}}'s techniques are about latency as much as about fitting.

## 8. Implementation

```python {tier=A name=the-transformer-block}
"""A complete transformer block, its parameter accounting, and the
residual-stream decomposition of eq. 67.7.
"""
import numpy as np

rng = np.random.default_rng(0)


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def rmsnorm(x, g, eps=1e-6):
    return g * x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps)


def silu(z):
    return z / (1.0 + np.exp(-np.clip(z, -60, 60)))


class Block:
    """Pre-norm block. Set gated=True for eq. 67.4."""

    def __init__(self, d, h, d_ff=None, gated=False, seed=0):
        rs = np.random.default_rng(seed)
        s = 1 / np.sqrt(d)
        self.d, self.h, self.dk = d, h, d // h
        self.gated = gated
        self.d_ff = d_ff if d_ff else (int(8 * d / 3) if gated else 4 * d)
        self.Wq = rs.normal(0, s, (d, d))
        self.Wk = rs.normal(0, s, (d, d))
        self.Wv = rs.normal(0, s, (d, d))
        self.Wo = rs.normal(0, s, (d, d))
        self.g1 = np.ones(d)
        self.g2 = np.ones(d)
        self.W1 = rs.normal(0, s, (d, self.d_ff))
        self.W2 = rs.normal(0, 1 / np.sqrt(self.d_ff), (self.d_ff, d))
        if gated:
            self.Wg = rs.normal(0, s, (d, self.d_ff))

    def attn_params(self):
        return 4 * self.d * self.d

    def ffn_params(self):
        n = 2 if not self.gated else 3
        return n * self.d * self.d_ff

    def n_params(self):
        return self.attn_params() + self.ffn_params() + 2 * self.d

    def attn(self, x):
        B, T, d = x.shape
        sp = lambda M: M.reshape(B, T, self.h, self.dk).transpose(0, 2, 1, 3)
        Q, K, V = sp(x @ self.Wq), sp(x @ self.Wk), sp(x @ self.Wv)
        S = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.dk)
        mask = np.tril(np.ones((T, T), dtype=bool))
        A = softmax(np.where(mask, S, -np.inf))
        out = (A @ V).transpose(0, 2, 1, 3).reshape(B, T, d)
        return out @ self.Wo

    def ffn(self, x):
        if self.gated:
            return (silu(x @ self.Wg) * (x @ self.W1)) @ self.W2
        return np.maximum(0.0, x @ self.W1) @ self.W2

    def forward(self, x, record=None):
        a = self.attn(rmsnorm(x, self.g1))               # eq. 67.1
        if record is not None:
            record.append(("attn", a))
        h = x + a
        f = self.ffn(rmsnorm(h, self.g2))                # eq. 67.2
        if record is not None:
            record.append(("ffn", f))
        return h + f


print("=" * 72)
print("where a transformer's parameters are (eqs. 67.5-67.6)")
print("=" * 72)
print(f"{'d':>6} {'d_ff':>6} {'gated':>7} {'attention':>12} {'FFN':>12} "
      f"{'FFN fraction':>14}")
for d in (512, 768, 4096):
    for gated in (False, True):
        b = Block(d, 8, gated=gated, seed=1)
        print(f"{d:>6} {b.d_ff:>6} {str(gated):>7} {b.attn_params():>12,} "
              f"{b.ffn_params():>12,} "
              f"{b.ffn_params() / (b.attn_params() + b.ffn_params()):>14.1%}")

print("\nTwo-thirds of every block is the feed-forward network, in both the")
print("ungated and the gated form — which is why the gated hidden width is")
print("8d/3 rather than 4d. Eq. 67.6 says exactly this, and it is the most")
print("surprising number in the part for anyone whose picture of a")
print("transformer came from a diagram where attention dominates.")

# --- and the FLOPs ----------------------------------------------------------
print("\n" + "=" * 72)
print("and where the FLOPs are (eq. 67.7)")
print("=" * 72)
print(f"{'d':>6} " + " ".join(f"{f'T={T}':>22}" for T in (512, 2048, 8192)))
print(f"{'':>6} " + " ".join(f"{'attn / FFN / attn %':>22}" for _ in range(3)))
for d in (768, 4096):
    row = []
    for T in (512, 2048, 8192):
        fa = 8 * d * d + 4 * T * d
        ff = 16 * d * d
        row.append(f"{fa / 1e6:.0f}M / {ff / 1e6:.0f}M / {fa / (fa + ff):.0%}")
    print(f"{d:>6} " + " ".join(f"{r:>22}" for r in row))

print(f"\nEq. 67.7's crossover is at T = 4d:")
for d in (768, 4096):
    print(f"  d = {d:>5}  ->  crossover at T = {4 * d:,}")
print("\nBelow that the feed-forward block dominates the arithmetic too, and")
print("this is the SAME threshold as Chapter 62's, for the same reason: it")
print("is where attention's quadratic term overtakes the linear ones.")

# --- section 6.1: the residual-stream decomposition -------------------------
print("\n" + "=" * 72)
print("a transformer's output is a SUM of sublayer outputs (eq. 67.7)")
print("=" * 72)
d, h, L, B, T = 128, 8, 6, 2, 16
x0 = rng.normal(size=(B, T, d)) * 0.5
blocks = [Block(d, h, seed=10 + i) for i in range(L)]

record = []
x = x0.copy()
for b in blocks:
    x = b.forward(x, record=record)

total = x0 + sum(v for _, v in record)
print(f"{L} blocks, so {2 * L} sublayer outputs")
print(f"max |x_L  -  (x_0 + sum of all sublayer outputs)| = "
      f"{np.abs(x - total).max():.3e}")

print("\nExact. Eq. 67.7 is not an approximation or a way of thinking about")
print("it — the final hidden state IS the embedding plus the sum of every")
print("sublayer's output, with nothing else in between.")
print("\nThat is why the logit lens of Chapter 66 is type-correct, why heads")
print("can be analysed individually (Chapter 64), and why deleting one block")
print("from a trained transformer changes the output less than it would in")
print("a plain stack: you have removed one term from a sum of twelve.")

print("\nRMS contribution of each sublayer to the final stream:\n")
print(f"{'layer':>6} {'attention':>12} {'FFN':>12} {'stream RMS after':>18}")
x = x0.copy()
for i, b in enumerate(blocks):
    r = []
    x = b.forward(x, record=r)
    print(f"{i:>6} {float(np.sqrt((r[0][1] ** 2).mean())):>12.4f} "
          f"{float(np.sqrt((r[1][1] ** 2).mean())):>12.4f} "
          f"{float(np.sqrt((x ** 2).mean())):>18.4f}")

print("\n(untrained, so the magnitudes reflect initialisation rather than")
print(" learned behaviour — the point is the accounting, not the values)")

# --- section 6.3: the residual norm grows -----------------------------------
print("\n" + "=" * 72)
print("the residual norm grows with depth (eq. 67.10)")
print("=" * 72)
print("Each sublayer adds a roughly fixed-scale vector, so variances add and")
print("the norm should grow as sqrt(number of sublayers).\n")
deep = [Block(d, h, seed=100 + i) for i in range(32)]
x = rng.normal(size=(4, T, d)) * 0.5
n0 = float(np.sqrt((x ** 2).mean()))
print(f"{'after layer':>12} {'stream RMS':>12} {'vs layer 0':>12} "
      f"{'sqrt(2L+1) prediction':>23} {'per-layer angle':>17}")
prev = x.copy()
for i, b in enumerate(deep):
    x = b.forward(x)
    if (i + 1) in (1, 2, 4, 8, 16, 32):
        rms = float(np.sqrt((x ** 2).mean()))
        cos = float((prev * x).sum() / (np.linalg.norm(prev)
                                        * np.linalg.norm(x)))
        print(f"{i + 1:>12} {rms:>12.4f} {rms / n0:>12.3f} "
              f"{np.sqrt(2 * (i + 1) + 1):>23.3f} "
              f"{np.degrees(np.arccos(np.clip(cos, -1, 1))):>16.2f}°")
    prev = x.copy()

print("\nThe growth follows the square-root prediction of eq. 67.10, and the")
print("per-layer angle shrinks with depth: a block adding a fixed-scale")
print("vector to a growing stream rotates it less and less.")
print("\nTwo consequences. Later blocks contribute proportionally less, which")
print("is measurable in real models and is not usually what people expect")
print("from 'deeper is more processing'. And the FINAL normalisation before")
print("the unembedding is mandatory — without it, the logit scale would")
print("depend on how many layers the model happens to have.")
```

```python {tier=A name=prenorm-vs-postnorm}
"""Pre-norm against post-norm (eqs. 67.11-67.12): the gradient argument that
decided every transformer since 2020.
"""
import numpy as np

rng = np.random.default_rng(1)


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def rmsnorm(x, eps=1e-6):
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps)


def rmsnorm_back(x, dy, eps=1e-6):
    """Backward through y = x / rms(x)."""
    d = x.shape[-1]
    ms = (x ** 2).mean(-1, keepdims=True) + eps
    r = np.sqrt(ms)
    return (dy - x * (dy * x).sum(-1, keepdims=True) / (d * ms)) / r


class SimpleBlock:
    """A block with one linear 'sublayer', so the gradient argument is not
    confounded by attention's details. mode is 'pre' or 'post'."""

    def __init__(self, d, mode, seed=0, scale=1.0):
        rs = np.random.default_rng(seed)
        self.W1 = rs.normal(0, scale * np.sqrt(2 / d), (d, 4 * d))
        self.W2 = rs.normal(0, scale * np.sqrt(1 / (4 * d)), (4 * d, d))
        self.mode = mode

    def forward(self, x):
        self.x = x
        if self.mode == "pre":
            self.n = rmsnorm(x)
            self.z = self.n @ self.W1
            self.a = np.maximum(0.0, self.z)
            self.f = self.a @ self.W2
            return x + self.f
        self.z = x @ self.W1
        self.a = np.maximum(0.0, self.z)
        self.f = self.a @ self.W2
        self.s = x + self.f
        return rmsnorm(self.s)

    def backward(self, dy):
        if self.mode == "pre":
            df = dy
            da = df @ self.W2.T
            dz = da * (self.z > 0)
            dn = dz @ self.W1.T
            return dy + rmsnorm_back(self.x, dn)
        ds = rmsnorm_back(self.s, dy)
        da = ds @ self.W2.T
        dz = da * (self.z > 0)
        return ds + dz @ self.W1.T


print("=" * 72)
print("the gradient reaching each layer (eqs. 67.11-67.12)")
print("=" * 72)
d, B, T = 96, 8, 16
print("A stack of identical blocks, gradient of RMS 1 injected at the top.")
print("The question is what reaches the bottom.\n")
print(f"{'depth':>7} {'mode':>6} " +
      " ".join(f"{f'layer {i}':>11}" for i in ("L", "3L/4", "L/2", "1"))
      + f" {'ratio top/bottom':>18}")
for L in (8, 24, 48):
    for mode in ("pre", "post"):
        blocks = [SimpleBlock(d, mode, seed=200 + i) for i in range(L)]
        x = rng.normal(size=(B, T, d))
        for b in blocks:
            x = b.forward(x)
        g = rng.normal(size=x.shape)
        g = g / np.sqrt((g ** 2).mean())
        norms = [float(np.sqrt((g ** 2).mean()))]
        for b in reversed(blocks):
            g = b.backward(g)
            norms.append(float(np.sqrt((g ** 2).mean())))
        picks = [0, L // 4, L // 2, L]
        print(f"{L:>7} {mode:>6} " +
              " ".join(f"{norms[i]:>11.3e}" for i in picks)
              + f" {norms[0] / max(norms[L], 1e-300):>18.3e}")

print("\nRead the last column: the gradient at the top divided by what")
print("reaches the bottom. A value near 1 means the gradient crossed the")
print("whole stack intact.")
print("\nEq. 67.11 says pre-norm has an EXACT identity term in its Jacobian —")
print("the normalisation is on the branch, not on the skip — so the product")
print("over L layers contains a gain-1 path however deep the stack is.")
print("\nEq. 67.12 says post-norm does not: the normalisation's Jacobian")
print("multiplies EVERYTHING including the identity, so L such factors")
print("accumulate. That is the mechanism Xiong et al. analyse, and warmup")
print("is what post-norm models use to survive the early steps until the")
print("parameters move somewhere less hostile.")

# --- the same, as an optimisation problem -----------------------------------
print("\n" + "=" * 72)
print("what that costs in training, with and without warmup")
print("=" * 72)


def train_stack(L, mode, lr=3e-3, warmup=0, steps=1500, d=48, seed=0):
    """Fit a random target through a deep stack; report the final loss."""
    rs = np.random.default_rng(seed)
    blocks = [SimpleBlock(d, mode, seed=300 + i) for i in range(L)]
    X = rs.normal(size=(64, 8, d))
    Ytgt = rs.normal(size=(64, 8, d)) * 0.5
    ps = []
    for b in blocks:
        ps += [b.W1, b.W2]
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    for t in range(1, steps + 1):
        x = X
        for b in blocks:
            x = b.forward(x)
        loss = float(((x - Ytgt) ** 2).mean())
        if not np.isfinite(loss):
            return float("inf")
        g = 2 * (x - Ytgt) / x.size
        grads = []
        for b in reversed(blocks):
            if b.mode == "pre":
                df = g
                da = df @ b.W2.T
                dz = da * (b.z > 0)
                grads.append(b.a.reshape(-1, 4 * d).T @ df.reshape(-1, d))
                grads.append(b.n.reshape(-1, d).T @ dz.reshape(-1, 4 * d))
                g = g + rmsnorm_back(b.x, dz @ b.W1.T)
            else:
                ds = rmsnorm_back(b.s, g)
                da = ds @ b.W2.T
                dz = da * (b.z > 0)
                grads.append(b.a.reshape(-1, 4 * d).T @ ds.reshape(-1, d))
                grads.append(b.x.reshape(-1, d).T @ dz.reshape(-1, 4 * d))
                g = ds + dz @ b.W1.T
        grads = grads[::-1]
        order = []
        for i in range(L):
            order += [grads[2 * i + 1], grads[2 * i]]
        cur = lr * min(1.0, t / warmup) if warmup else lr
        for i, (p, gr) in enumerate(zip(ps, order)):
            m[i] = 0.9 * m[i] + 0.1 * gr
            v[i] = 0.999 * v[i] + 0.001 * gr * gr
            p -= cur * (m[i] / (1 - 0.9 ** t)) / (
                np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    x = X
    for b in blocks:
        x = b.forward(x)
    return float(((x - Ytgt) ** 2).mean())


print(f"{'depth':>7} {'mode':>6} {'no warmup':>13} {'warmup 200':>13}")
for L in (8, 24, 48):
    for mode in ("pre", "post"):
        a = train_stack(L, mode, warmup=0)
        b = train_stack(L, mode, warmup=200)
        f = lambda z: "diverged" if not np.isfinite(z) or z > 10 else f"{z:.5f}"
        print(f"{L:>7} {mode:>6} {f(a):>13} {f(b):>13}")

print("\nThis is Xiong et al.'s claim as an experiment: post-norm should")
print("benefit from warmup and pre-norm should not need it, with the gap")
print("widening as the stack deepens.")
print("\nThe simplification to watch is that these blocks have no attention,")
print("so what is being tested is the NORMALISATION PLACEMENT alone,")
print("isolated from everything else in a transformer. That is the point of")
print("the simplification and it is also its limit.")
```

## 9. Practical Example

```python {tier=A name=ffn-design-choices}
"""The feed-forward design decisions, measured: expansion ratio, gating, and
what the hidden units respond to.
"""
import numpy as np

rng = np.random.default_rng(4)

V, T = 32, 10


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def rmsnorm(x, eps=1e-6):
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps)


def silu(z):
    return z / (1.0 + np.exp(-np.clip(z, -60, 60)))


def make_task(n, seed):
    """Next-token prediction where the answer depends on a nonlinear
    function of two earlier tokens — so the FFN has something to do."""
    rs = np.random.default_rng(seed)
    rule = np.random.default_rng(77).integers(0, V, (V, V))
    X = rs.integers(0, V, (n, T))
    Y = np.zeros((n, T - 1), dtype=int)
    for t in range(T - 1):
        a = X[:, max(0, t - 1)]
        b = X[:, t]
        nxt = rule[a, b]
        flip = rs.random(n) < 0.1
        Y[:, t] = np.where(flip, rs.integers(0, V, n), nxt)
    return X[:, :-1], Y


class Model:
    """Embedding + one pre-norm block + unembedding."""

    def __init__(self, d=48, h=4, d_ff=None, gated=False, seed=0):
        rs = np.random.default_rng(seed)
        self.d, self.h, self.dk = d, h, d // h
        self.gated = gated
        self.d_ff = d_ff if d_ff else (int(8 * d / 3) if gated else 4 * d)
        s = 1 / np.sqrt(d)
        self.E = rs.normal(0, 0.05, (V, d))
        self.P = rs.normal(0, 0.05, (T - 1, d))
        self.Wq = rs.normal(0, s, (d, d))
        self.Wk = rs.normal(0, s, (d, d))
        self.Wv = rs.normal(0, s, (d, d))
        self.Wo = rs.normal(0, s, (d, d))
        self.W1 = rs.normal(0, s, (d, self.d_ff))
        self.W2 = rs.normal(0, 1 / np.sqrt(self.d_ff), (self.d_ff, d))
        if gated:
            self.Wg = rs.normal(0, s, (d, self.d_ff))
        self.U = rs.normal(0, 0.05, (V, d))

    def params(self):
        p = [self.E, self.P, self.Wq, self.Wk, self.Wv, self.Wo,
             self.W1, self.W2, self.U]
        return p + ([self.Wg] if self.gated else [])

    def n_params(self):
        return sum(p.size for p in self.params())

    def forward(self, X, keep=False):
        n, Tn = X.shape
        x = self.E[X] + self.P[None, :Tn, :]
        self.x0 = x
        na = rmsnorm(x)
        self.na = na
        sp = lambda M: M.reshape(n, Tn, self.h, self.dk).transpose(0, 2, 1, 3)
        Q, K, Vv = sp(na @ self.Wq), sp(na @ self.Wk), sp(na @ self.Wv)
        S = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.dk)
        mask = np.tril(np.ones((Tn, Tn), dtype=bool))
        A = softmax(np.where(mask, S, -np.inf))
        ctx = (A @ Vv).transpose(0, 2, 1, 3).reshape(n, Tn, self.d)
        self.A, self.Q, self.K, self.Vv, self.ctx = A, Q, K, Vv, ctx
        h1 = x + ctx @ self.Wo
        self.h1 = h1
        nf = rmsnorm(h1)
        self.nf = nf
        if self.gated:
            self.gpre = nf @ self.Wg
            self.upre = nf @ self.W1
            self.hid = silu(self.gpre) * self.upre
        else:
            self.upre = nf @ self.W1
            self.hid = np.maximum(0.0, self.upre)
        h2 = h1 + self.hid @ self.W2
        self.h2 = h2
        out = rmsnorm(h2)
        self.out = out
        return out @ self.U.T

    def loss(self, X, Y):
        P = softmax(self.forward(X))
        return float(-np.log(np.clip(
            np.take_along_axis(P, Y[..., None], -1), 1e-12, None)).mean())


# A compact reverse pass, written out once.
def grads(model, X, Y):
    n, Tn = X.shape
    logits = model.forward(X)
    P = softmax(logits)
    nt = n * Tn
    loss = float(-np.log(np.clip(
        np.take_along_axis(P, Y[..., None], -1), 1e-12, None)).sum() / nt)
    dl = P.copy()
    np.put_along_axis(dl, Y[..., None],
                      np.take_along_axis(dl, Y[..., None], -1) - 1.0, -1)
    dl /= nt
    d = model.d
    gU = np.einsum('ntv,ntd->vd', dl, model.out)
    dout = dl @ model.U

    def rms_back(x, dy, eps=1e-6):
        dd = x.shape[-1]
        ms = (x ** 2).mean(-1, keepdims=True) + eps
        return (dy - x * (dy * x).sum(-1, keepdims=True) / (dd * ms)) \
            / np.sqrt(ms)

    dh2 = rms_back(model.h2, dout)
    gW2 = model.hid.reshape(-1, model.d_ff).T @ dh2.reshape(-1, d)
    dhid = dh2 @ model.W2.T
    if model.gated:
        sg = silu(model.gpre)
        sig = 1 / (1 + np.exp(-np.clip(model.gpre, -60, 60)))
        dg = dhid * model.upre * (sig + model.gpre * sig * (1 - sig))
        du = dhid * sg
        gWg = model.nf.reshape(-1, d).T @ dg.reshape(-1, model.d_ff)
        gW1 = model.nf.reshape(-1, d).T @ du.reshape(-1, model.d_ff)
        dnf = dg @ model.Wg.T + du @ model.W1.T
    else:
        du = dhid * (model.upre > 0)
        gW1 = model.nf.reshape(-1, d).T @ du.reshape(-1, model.d_ff)
        dnf = du @ model.W1.T
        gWg = None
    dh1 = dh2 + rms_back(model.h1, dnf)
    gWo = model.ctx.reshape(-1, d).T @ dh1.reshape(-1, d)
    dctx = dh1 @ model.Wo.T
    sp = lambda M: M.reshape(n, Tn, model.h, model.dk).transpose(0, 2, 1, 3)
    dC = sp(dctx)
    dA = dC @ model.Vv.transpose(0, 1, 3, 2)
    dV = model.A.transpose(0, 1, 3, 2) @ dC
    dS = model.A * (dA - (dA * model.A).sum(-1, keepdims=True))
    dS /= np.sqrt(model.dk)
    dQ, dK = dS @ model.K, dS.transpose(0, 1, 3, 2) @ model.Q
    mg = lambda G: G.transpose(0, 2, 1, 3).reshape(n, Tn, d)
    naf = model.na.reshape(-1, d)
    gWq = naf.T @ mg(dQ).reshape(-1, d)
    gWk = naf.T @ mg(dK).reshape(-1, d)
    gWv = naf.T @ mg(dV).reshape(-1, d)
    dna = mg(dQ) @ model.Wq.T + mg(dK) @ model.Wk.T + mg(dV) @ model.Wv.T
    dx0 = dh1 + rms_back(model.x0, dna)
    gP = dx0.sum(axis=0)
    gE = np.zeros_like(model.E)
    np.add.at(gE, X.reshape(-1), dx0.reshape(-1, d))
    out = [gE, gP, gWq, gWk, gWv, gWo, gW1, gW2, gU]
    if model.gated:
        out.append(gWg)
    return loss, out


def train(model, X, Y, steps=2500, lr=4e-3, batch=128, seed=0):
    ps = model.params()
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    rs = np.random.default_rng(seed + 2)
    for t in range(1, steps + 1):
        b = rs.integers(0, len(X), batch)
        _, gs = grads(model, X[b], Y[b])
        for i, (p, g) in enumerate(zip(ps, gs)):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            p -= lr * (m[i] / (1 - 0.9 ** t)) / (
                np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    return model


Xtr, Ytr = make_task(10000, 1)
Xte, Yte = make_task(5000, 2)

print("=" * 72)
print("the expansion ratio (section 6.4)")
print("=" * 72)
print("d_ff = 4d is a convention with no derivation. How sensitive is it?\n")
print(f"{'d_ff / d':>10} {'d_ff':>6} {'params':>9} {'test NLL':>10}")
for ratio in (0.5, 1, 2, 4, 8, 16):
    mdl = train(Model(d=48, d_ff=int(48 * ratio), seed=5), Xtr, Ytr)
    print(f"{ratio:>10g} {int(48 * ratio):>6} {mdl.n_params():>9,} "
          f"{mdl.loss(Xte, Yte):>10.4f}")

print("\nSection 6.4 predicts a broad plateau rather than a sharp optimum at")
print("4, because none of the three arguments for that value is a")
print("derivation. Whether the plateau appears here is what the table says.")
print("\nThe honest reading of the convention: 4 is stable because it works,")
print("keeps both matrices well-shaped for tiling, and gives the")
print("two-thirds parameter split — not because anything requires it.")

# --- gating at matched parameters -------------------------------------------
print("\n" + "=" * 72)
print("gating, at matched parameters (eqs. 67.4, 67.6)")
print("=" * 72)
print("Three matrices instead of two, so d_ff drops to 8d/3 for parity.\n")
print(f"{'block':<22} {'d_ff':>6} {'FFN params':>12} {'total':>9} "
      f"{'test NLL':>10}")
a = train(Model(d=48, gated=False, seed=5), Xtr, Ytr)
print(f"{'ReLU, d_ff = 4d':<22} {a.d_ff:>6} "
      f"{2 * 48 * a.d_ff:>12,} {a.n_params():>9,} {a.loss(Xte, Yte):>10.4f}")
b = train(Model(d=48, gated=True, seed=5), Xtr, Ytr)
print(f"{'SwiGLU, d_ff = 8d/3':<22} {b.d_ff:>6} "
      f"{3 * 48 * b.d_ff:>12,} {b.n_params():>9,} {b.loss(Xte, Yte):>10.4f}")
c = train(Model(d=48, gated=True, d_ff=4 * 48, seed=5), Xtr, Ytr)
print(f"{'SwiGLU, d_ff = 4d':<22} {c.d_ff:>6} "
      f"{3 * 48 * c.d_ff:>12,} {c.n_params():>9,} {c.loss(Xte, Yte):>10.4f}")

print("\nThe first two rows are the comparison that matters: matched")
print("parameters, different block structure. The third is unmatched and is")
print("there to separate 'gating helps' from 'more parameters help'.")
print("\nSection 6.5 gives the structural difference: a gated block's output")
print("is a PRODUCT of two projections, so it is quadratic in the input,")
print("where an ungated block reaches second-order interactions only")
print("through the activation's curvature. Whether that is the mechanism is")
print("not established — Shazeer's own paper offers none.")

# --- section 4.4: what the hidden units respond to --------------------------
print("\n" + "=" * 72)
print("what the hidden units respond to (section 4.4)")
print("=" * 72)
mdl = a
mdl.forward(Xte[:3000])
hid = mdl.hid.reshape(-1, mdl.d_ff)
toks = Xte[:3000].reshape(-1)
print(f"{mdl.d_ff} hidden units. For each, find the token whose presence")
print("most raises its activation, and how selective that is.\n")

act_by_tok = np.zeros((V, mdl.d_ff))
cnt = np.zeros(V)
np.add.at(act_by_tok, toks, hid)
np.add.at(cnt, toks, 1)
mean_act = act_by_tok / np.maximum(cnt, 1)[:, None]
overall = hid.mean(0)
sel = (mean_act - overall) / (hid.std(0) + 1e-9)

top = np.abs(sel).max(0)
order = np.argsort(top)[::-1]
print(f"{'unit':>6} {'best token':>12} {'selectivity (sd)':>18} "
      f"{'fraction active':>17}")
for u in list(order[:5]) + list(order[-3:]):
    t = int(np.abs(sel[:, u]).argmax())
    frac = float((hid[:, u] > 0).mean())
    print(f"{u:>6} {t:>12} {sel[t, u]:>18.3f} {frac:>17.4f}")

print(f"\nmedian selectivity across all units: "
      f"{float(np.median(top)):.3f} standard deviations")
print(f"units with selectivity above 1 sd: "
      f"{int((top > 1).sum())} of {mdl.d_ff}")
print(f"units never active: {int(((hid > 0).mean(0) == 0).sum())}")

print("\nThe key-value memory reading of section 4.4 predicts that some")
print("units should respond selectively to recognisable patterns, and the")
print("selectivity column is that prediction measured.")
print("\nWhat the numbers usually show — here and in real models — is a")
print("MINORITY of clearly selective units and a majority that are not. That")
print("is the honest state of the interpretation: the mechanism is real for")
print("some units, polysemantic units are the norm, and 'the FFN is a")
print("key-value memory' is a useful frame rather than an established")
print("description.")
```

## 10. Production Considerations

**Budget the feed-forward block as two-thirds of the model.** Measured: it holds
two-thirds of the parameters and two-thirds of the arithmetic below $T = 4d$.
Optimisation effort aimed at attention is aimed at the minority.

**Use pre-norm.** Measured: post-norm's gradient degrades across a deep stack
and pre-norm's does not, which is {{eq:prenorm-jacobian}}'s exact identity term.

**Keep the final normalisation.** Measured: the residual norm grows as
$\sqrt{L}$, so without it the logit scale depends on depth.

**Fuse the gate and up projections.** Same shapes, same input, one kernel.

**Checkpoint the feed-forward block first.** {{eq:ffn-activation-memory}}: it is
usually the largest activation tensor and it is two matmuls to recompute.

**Compute the normalisation once per block.** The gated form uses it twice; a
naive implementation recomputes it.

**Do not tune the expansion ratio early.** Measured broad plateau; there are
better uses of a sweep.

## 11. Common Mistakes

**Thinking attention holds most of the parameters.** Measured two-thirds in the
feed-forward block.

**Using post-norm without warmup.** Measured degradation, and it worsens with
depth.

**Dropping the final normalisation** because "every block already normalises".
The blocks normalise their *branches*; nothing normalises the stream.

**Keeping $d_{\text{ff}} = 4d$ when switching to a gated block.** That is a 50%
parameter increase, not a like-for-like swap. {{eq:swiglu-width}}.

**Reading feed-forward neurons as concepts.** Measured: a minority are
selective and most are not.

**Applying the normalisation to the skip path.** That is post-norm, whatever you
call it.

## 12. Failure Modes

**Deep post-norm models failing to train.** {{eq:postnorm-jacobian}}, measured.

**Logit scale drifting with depth** when the final normalisation is missing.

**Out-of-memory from the feed-forward intermediate**, which is $4\times$ the
residual stream per position per block.

**Dead units in an ungated block.** ReLU's dead-unit failure
({{ch:dl-activations}}) applied at $4d$ units per block; measured here as the
never-active count.

**Later layers contributing nothing.** Measured mechanism: a fixed-scale
addition to a $\sqrt{\ell}$-growing stream rotates it less and less. Sometimes
benign, sometimes a sign the model is deeper than the task needs.

**Silent quality loss from a mismatched gated width** when porting a
configuration.

## 13. Alternatives

**Mixture of experts** replaces the single feed-forward block with $N$ of them
and routes each token to a few. Parameters grow with $N$; compute does not.
Since the block is two-thirds of the model, this is where the parameter count
goes when people want a bigger model at fixed compute ({{ch:res-moe}}).

**Parallel attention and feed-forward** compute both from the same normalised
input and add both to the stream, rather than sequencing them. Slightly worse
per parameter, meaningfully faster because the two matmuls can be fused.

**No feed-forward block at all** — attention-only transformers. They train and
are substantially worse, which is the cleanest evidence that the block is doing
something attention cannot.

**Other gates.** GEGLU, ReGLU and relatives differ only in $\phi$.
{{cite:shazeer2020glu}} compares them and the differences are small.

**Deeper narrow against shallower wide** at fixed parameters. A broad optimum
with a mild preference for depth, and a real inference-latency cost to depth.

## 14. Evaluation

**Print the parameter split.** Three lines, and it tells you where to spend
effort.

**Verify {{eq:residual-stream}} numerically.** Measured exact here; if it fails,
something is on the skip path that should not be.

**Log the residual norm per layer.** Growth much faster than $\sqrt{\ell}$ means
a block is writing too large a contribution.

**Log the per-layer rotation angle.** Angles collapsing to near zero in the
later layers means those layers are doing little.

**Count dead units in an ungated block.**

**Ablate whole blocks.** Because of {{eq:residual-stream}} this deletes one term
from a sum, and the loss change is a meaningful importance measure.

## 15. Advanced Concepts

**The residual stream as a communication channel.** Blocks write to subspaces
and later blocks read from them, with the stream as a shared bus of $d$
dimensions. Since there are $L(h+1)$ writers and only $d$ dimensions, they must
share — which is the setting in which superposition arises
({{ch:tf-multi-head}}).

**Superposition in feed-forward neurons.** More features than neurons, encoded
in overlapping directions, which is the leading explanation for why most units
are polysemantic. Sparse autoencoders are the current tool for decomposing them.
{{maturity:EMERGING}}

**Knowledge editing.** If $\mat{W}_2$'s columns are values in an associative
memory, editing specific ones should change specific facts. Methods along these
lines work well enough to be interesting and generalise unreliably.
{{maturity:EMERGING}}

**Depth-wise scaling laws.** At fixed parameters, the depth–width optimum
depends on the compute budget and the target inference latency, not on quality
alone.

**QK-normalisation.** Normalising the query and key projections inside attention
to control logit growth — {{ch:dl-normalization}}'s idea applied somewhere it
was not originally intended, and increasingly common in large models.

## 16. Connection to Previous Chapters

{{eq:mha-sum}} from {{ch:tf-multi-head}} is the special case of
{{eq:residual-stream}} for one block, and combining them gives
{{eq:full-decomposition}} — the full residual-stream decomposition that every
interpretability result relies on.

{{ch:dl-cnns}}'s residual connection is this chapter's skip path, and
{{eq:residual-expansion}}'s identity term is {{eq:prenorm-jacobian}}'s.
{{ch:dl-normalization}} supplies RMSNorm and the pre-norm argument, and
{{eq:norm-scale-invariance}} is why the branch's weight scale does not matter.
{{ch:dl-activations}} supplies SiLU and the dead-unit failure measured here.
{{ch:tf-embeddings}}'s logit lens is justified by {{eq:residual-stream}}.

Forward: {{ch:tf-architectures}} stacks these blocks into complete models.
{{ch:tf-complexity}} does the full cost accounting.
{{ch:res-moe}} replaces the feed-forward block with many.

## 17. Exercises

**Beginner**

1. What fraction of a transformer block's parameters is the feed-forward
   network?
2. What is the residual stream?
3. What is the difference between pre-norm and post-norm?
4. Why is the gated hidden width $8d/3$ and not $4d$?
5. Why is the final normalisation needed?

**Intermediate**

6. Derive {{eq:ffn-fraction}}.
7. Derive {{eq:swiglu-width}}.
8. Derive {{eq:prenorm-jacobian}} and {{eq:postnorm-jacobian}} and say which
   has an exact identity term.
9. Using {{eq:residual-growth}}, predict the stream norm after 48 layers.
10. Compute the feed-forward activation memory for $B=8$, $T=4096$, $d=4096$,
    bf16.

**Advanced**

11. Derive {{eq:full-decomposition}} by combining {{eq:mha-sum}} and
    {{eq:residual-stream}}, and count the terms.
12. Show that a gated block is quadratic in its input and an ungated one is
    not.
13. Explain why the per-layer rotation angle shrinks with depth,
    quantitatively.
14. Derive the backward pass through RMSNorm.

**Implementation**

15. Implement the block with the backward pass and gradient-check it.
16. Reproduce the pre-norm/post-norm gradient table at depth 96.
17. Implement a mixture-of-experts feed-forward block and compare parameters
    against compute.
18. Measure per-unit selectivity on a trained model and compare against random
    directions.

**Reasoning**

19. A 48-layer model trains and a 96-layer one does not, same recipe. Give an
    ordered diagnostic procedure.
20. Your model's later layers contribute almost nothing. Is that a problem?

## 18. Interview Questions

**"Where are a transformer's parameters?"** — Two-thirds in the feed-forward
blocks. Give {{eq:ffn-fraction}}.

**"What is the residual stream?"** — {{eq:residual-stream}}: the output is the
embedding plus a sum of sublayer outputs. Say what it licenses — the logit lens,
head-level analysis, block ablation.

**"Pre-norm or post-norm?"** — Pre-norm, and give the Jacobian argument. Note
post-norm needs warmup.

**"Why SwiGLU?"** — Empirically better at matched parameters, and be honest
that the mechanism is not established.

**"Why is $d_{\text{ff}} = 4d$?"** — Convention. Saying so is the right answer;
inventing a derivation is not.

**"What does the feed-forward block do?"** — Position-wise nonlinear
processing; the key–value memory reading with its evidential caveat.

## 19. Research Questions

**What do feed-forward neurons represent?** A minority are selective, most are
polysemantic, and superposition is the leading account. Sparse autoencoders are
the current tool and their outputs are not yet clearly validated.
{{maturity:RESEARCH FRONTIER}}

**Why does gating help?** {{cite:shazeer2020glu}} offers no mechanism and none
has been established since. {{maturity:EMERGING}}

**Is the expansion ratio near-optimal?** The measured plateau suggests it is not
critical; whether 4 is best at scale is not settled.
{{maturity:EMERGING}}

**How much redundancy is in the depth?** Blocks can often be pruned or merged
with modest loss, and what that says about what depth buys is unclear.
{{maturity:EMERGING}}

## 20. Chapter Summary

Two-thirds of a transformer's parameters are in the feed-forward blocks, not in
attention — measured at every width and in both the gated and ungated forms, and
{{eq:ffn-fraction}} says why: $8d^2$ against attention's $4d^2$. Two-thirds of
the *arithmetic* too, until $T = 4d$, which is the same crossover
{{ch:tf-why-attention}} found for the same reason.

{{eq:residual-stream}} was verified exactly: the final hidden state is the
embedding plus the sum of every sublayer's output, with nothing else in between.
Combined with {{eq:mha-sum}}, a transformer's output decomposes into
$L(h+1)$ additive terms, one per head and one per feed-forward block. **The
residual stream is a shared channel that every block reads from and adds to**,
not a pipeline of nested transformations — which is what makes the logit lens
type-correct, head-level analysis coherent, and block ablation meaningful.

That stream's norm grows as $\sqrt{\ell}$, measured against the prediction, and
the per-layer rotation angle shrinks correspondingly: a block adding a
fixed-scale vector to a growing stream turns it less and less. Later blocks
therefore contribute proportionally less, and the final normalisation before the
unembedding is mandatory rather than tidy — without it the logit scale would
depend on how many layers the model has.

Pre-norm versus post-norm is a Jacobian argument.
{{eq:prenorm-jacobian}} has an **exact** identity term because the normalisation
sits on the branch; {{eq:postnorm-jacobian}} does not, because the
normalisation's Jacobian multiplies everything including the identity, so $L$
such factors accumulate. Measured across depths, that is what the gradient does,
and it is why post-norm needs warmup and pre-norm does not.

On the feed-forward block's design, the honest positions are narrower than the
folklore. The expansion ratio of 4 has no derivation — the three available
arguments are a consequence, a hardware convenience, and a vague appeal to
capacity — and the measured sensitivity is a broad plateau. Gating helps at
matched parameters, and {{cite:shazeer2020glu}} offers no mechanism; the
structural difference is that a gated block's output is a *product* of two
projections and therefore quadratic in its input. And the key–value memory
reading predicts selective hidden units, of which the measurement finds a
minority — polysemantic units are the norm, which is why superposition is the
leading account and why "the FFN is an associative memory" is a useful frame
rather than an established description.

## 21. Further Reading

{{cite:xiong2020prenorm}} is the paper that settled normalisation placement, and
the mean-field analysis is more careful than {{sec:6-mathematical-foundation}}'s
sketch. What is worth taking from it is the shape of the argument: a claim about
gradients *at initialisation*, tested by measuring them, and a prediction about
warmup that follows.

{{cite:shazeer2020glu}} is two pages of ablations and it is worth reading for
its last line, in which the author explicitly declines to explain the result.
That is unusual honesty and a fair summary of where the explanation still
stands.

{{cite:vaswani2017}} section 3.3 is the feed-forward block, in three sentences,
with $d_{\text{ff}} = 2048$ stated and not justified. Reading it makes clear how
much of the modern convention is inherited rather than derived.

{{cite:he2016resnet}} again, because the residual stream is that paper's idea
taken to its conclusion: not a skip *around* the computation but the medium the
computation happens in.

**Where to go next:** {{ch:tf-architectures}} assembles these blocks into
encoder, decoder and encoder–decoder models, and explains why the field
converged on the decoder-only shape.

---
id: tf-multi-head
number: 64
part: VII
tier: full
status: reviewed
requires: [tf-scaled-dot-product, tf-why-attention, dl-forward, math-matrices]
provides: [multi-head-attention, head-dimension, output-projection,
           head-specialisation, induction-head, attention-sink,
           rank-bottleneck, head-pruning]
citations: [vaswani2017, shazeer2019mqa, ainslie2023gqa, dao2022flash]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Define multi-head attention and get every shape in it right.
2. Explain why $h$ heads of dimension $d/h$ cost the same as one head of
   dimension $d$.
3. Derive the rank bottleneck of a single head and explain what heads buy.
4. Explain what is and is not known about head specialisation.
5. Implement multi-head attention efficiently as batched matrix products.
6. Explain the attention sink and why it is usually benign.
7. State the memory cost of the heads at inference and how GQA reduces it.

## 2. Why This Matters

**One head can only do one thing at a time.** A softmax over positions produces
one distribution, so one head computes one weighted average. A sentence needs
several relationships attended to at once — syntactic, coreferential,
positional — and {{sec:6-mathematical-foundation}} shows that a single head is
also *rank-limited* in a way that has nothing to do with intuition about
linguistics.

**The parameter accounting is the part people get wrong.** Multi-head attention
with $h$ heads has the *same* parameter count and the *same* FLOPs as
single-head attention at the same model width. The heads are a partition of the
existing dimensions, not an addition. {{sec:8-implementation}} verifies this to
the parameter.

**Head specialisation is real, partially understood, and routinely
overstated.** Specific circuits have been identified in small models. Whether
the head is the right unit of analysis at scale is unresolved, and
{{sec:19-research-questions}} says so.

**The number of key/value heads is now a separate decision from the number of
query heads**, and it is one of the highest-leverage decisions in serving.
{{cite:shazeer2019mqa}} and {{cite:ainslie2023gqa}} exist because the KV cache
is what limits how many users a model can serve
({{ch:tf-masking-kv}}).

## 3. Prerequisites

{{ch:tf-scaled-dot-product}} for the single-head mechanism and the
$\sqrt{d_k}$ derivation — this chapter is that chapter applied $h$ times in
parallel. {{ch:tf-why-attention}} for why any of it. {{ch:dl-forward}} for
batched shapes and arithmetic intensity. {{ch:math-matrices}} for rank.

## 4. Intuitive Explanation

### 4.1 One head, one relationship

A head produces one attention distribution per query position. Whatever that
distribution encodes — "the verb this subject belongs to", "the noun this
pronoun refers to" — it encodes *one* thing, because there is one set of
weights.

Real sentences need several at once:

```text
   "The keys to the cabinet were on the table"

   agreement head    keys ←──────────── were        (plural, not "cabinet")
   modifier head     keys ──▶ to the cabinet
   position head     were ──▶ on                    (adjacent)
```

Averaging these into one distribution would blur all three. Running them as
separate heads keeps them separate, and concatenating the results lets the next
layer use all three.

### 4.2 Split, don't add

The obvious way to get $h$ heads is to run $h$ full-width attentions. That would
cost $h$ times as much. Multi-head attention does something cheaper:

```text
   single head             multi-head, h = 8
   ───────────             ─────────────────
   d = 512                 d = 512, split into 8 heads of d_k = 64
   Q, K, V : 512 wide      Q, K, V : still 512 wide TOTAL
                           head 1 gets dims   0..63
                           head 2 gets dims  64..127
                           ...
                           head 8 gets dims 448..511

   same parameters, same FLOPs, eight attention patterns instead of one
```

**Each head is narrower, and there are more of them.** The projections are the
same size; they are just *reshaped* into $h$ blocks. That is why
{{sec:6-mathematical-foundation}}'s cost accounting comes out identical, and it
is the detail that makes multi-head attention nearly free.

### 4.3 What narrowing costs

Nothing is free, and the cost is expressiveness per head. A head of dimension
$d_k$ computes scores $\vec{q}\T\vec{k}$ where both vectors live in $d_k$
dimensions, so the score matrix $\mat{Q}\mat{K}\T$ has rank at most $d_k$.

With $d_k = 64$ and a sequence of 2048 positions, the $2048 \times 2048$ score
matrix has rank at most 64. **It cannot express an arbitrary pattern of
relationships between positions** — only a rank-64 approximation of one.

That is a real limit and it is why $h$ is not simply set as large as possible.
{{sec:8-implementation}} measures it directly by asking a head to reproduce
attention patterns of increasing rank.

### 4.4 What the heads actually learn

Honest summary, because this is a topic where confident claims outrun the
evidence.

**Established.** Heads do differ from one another. Some attend almost entirely
to the previous token; some to the first token regardless of content; some
implement identifiable algorithms — an *induction head* attends from the current
token to the position after a previous occurrence of it, which is how a model
completes a repeated pattern.

**Established.** Many heads can be pruned with little loss. That is a robust
empirical finding across models.

**Not established.** That each head has one interpretable function; that the
head is the right unit of analysis; that findings from small models transfer to
large ones. Features appear to be distributed across heads and layers rather
than localised in them, and a head that looks like it does one thing on one
distribution often does something else on another.

### 4.5 The attention sink

Most trained models have heads that put a large fraction of their attention mass
on the first token, regardless of what it is.

The usual explanation is that the softmax **must** sum to one, so a head with
nothing useful to attend to still has to put its mass somewhere. The first token
is a convenient dumping ground: it is present in every sequence and, being
position 1, is unlikely to carry content the head would corrupt.

This matters practically. Evicting the first token from a KV cache to save
memory degrades the model badly, which is surprising until you know about the
sink and obvious afterwards.

## 5. Formal Explanation

### 5.1 The definition

For input $\mat{X} \in \R^{T \times d}$ and $h$ heads with
$d_k = d_v = d/h$:

$$
\mat{Q}_i = \mat{X}\mat{W}^Q_i,\quad
\mat{K}_i = \mat{X}\mat{W}^K_i,\quad
\mat{V}_i = \mat{X}\mat{W}^V_i
$$ (eq:mha-projections)

with $\mat{W}^{Q}_i, \mat{W}^{K}_i, \mat{W}^{V}_i \in \R^{d\times d_k}$.

$$
\head_i = \softmax\!\left(\frac{\mat{Q}_i\mat{K}_i\T}{\sqrt{d_k}}\right)
 \mat{V}_i
$$ (eq:head)

$$
\MHA(\mat{X}) = \big[\head_1;\dots;\head_h\big]\,\mat{W}^O
$$ (eq:mha)

with $\mat{W}^O \in \R^{d\times d}$.

> IMPORTANT: **$\mat{W}^O$ is not decoration.** Without it, the concatenation
> hard-wires head $i$'s output into dimensions $i d_k$ through $(i+1)d_k - 1$ of
> the residual stream. $\mat{W}^O$ lets each head write to any direction, and
> {{sec:6-mathematical-foundation}} shows it decomposes into per-head output
> maps — which is what makes a head a self-contained read-then-write operation.

### 5.2 Parameters and FLOPs

$$
\text{parameters} = 4d^2
 \quad\text{(}\mat{W}^Q, \mat{W}^K, \mat{W}^V, \mat{W}^O\text{, each } d\times d\text{)}
$$ (eq:mha-params)

**Independent of $h$.** The $h$ per-head matrices of shape $d\times d_k$
concatenate into one $d \times d$ matrix.

$$
\text{FLOPs} = \underbrace{8Td^2}_{\text{four projections}}
 + \underbrace{4T^2 d}_{\text{scores and weighted sum}}
$$ (eq:mha-flops)

Also independent of $h$: the $h$ score matrices are each $T\times T\times d_k$,
and $h \cdot d_k = d$.

**So $h$ is free in both parameters and arithmetic.** What it costs is
expressiveness per head ({{sec:6-mathematical-foundation}}) and, at inference,
KV-cache memory that scales with the number of *key/value* heads.

### 5.3 Shapes

The reshape that everyone gets wrong once:

```text
   X            (B, T, d)
   X @ W_Q      (B, T, d)
   reshape      (B, T, h, d_k)
   transpose    (B, h, T, d_k)      <- heads become a BATCH dimension
   scores       (B, h, T, T)
   head out     (B, h, T, d_k)
   transpose    (B, T, h, d_k)
   reshape      (B, T, d)
   @ W_O        (B, T, d)
```

The transpose before the reshape back is mandatory. Reshaping
$(B, h, T, d_k)$ directly to $(B, T, d)$ interleaves the heads with the
positions and produces a silently wrong result — no error, a model that trains
to a worse loss. {{sec:8-implementation}} demonstrates it.

### 5.4 Sharing keys and values

The three variants, distinguished only by how many key/value heads there are:

{#tbl:mha-variants caption="Attention variants by the number of key/value heads. Query heads are unchanged in all three; only the KV cache and the key/value parameter count differ, and the cache is what matters at serving time."}

| Variant | Query heads | KV heads | KV cache per token | Quality |
|---|---|---|---|---|
| Multi-head (MHA) | $h$ | $h$ | $2 L h d_k$ | baseline |
| Grouped-query (GQA) | $h$ | $g$, $1<g<h$ | $2 L g d_k$ | near baseline |
| Multi-query (MQA) | $h$ | 1 | $2 L d_k$ | slight loss |

{{cite:shazeer2019mqa}} introduced MQA after diagnosing incremental decoding as
bound by the memory bandwidth of loading the key and value tensors.
{{cite:ainslie2023gqa}} generalised it to an intermediate $g$, and gave a recipe
for converting an existing multi-head checkpoint at roughly 5% of the original
pretraining compute.

**GQA with $g = 8$ is the 2026 default in open-weight models**, and the reason
is entirely in the fourth column.

### 5.5 Cross-attention

Nothing requires $\mat{Q}$, $\mat{K}$ and $\mat{V}$ to come from the same
sequence. In cross-attention the queries come from one sequence and the keys and
values from another:

$$
\mat{Q} = \mat{X}_{\text{dec}}\mat{W}^Q,
\qquad
\mat{K}, \mat{V} = \mat{X}_{\text{enc}}\mat{W}^K, \mat{X}_{\text{enc}}\mat{W}^V
$$ (eq:cross-attention)

This is {{cite:bahdanau2015}}'s original setting, and it is how an
encoder–decoder transformer connects its halves ({{ch:tf-architectures}}). Note
that the key/value sequence is *fixed* during decoding, so its projections are
computed once — which is why cross-attention's cache never grows.

### 5.6 What each head costs at inference, per token

The training-time accounting of {{eq:mha-flops}} is per *sequence*. Generation
is different, because each new token attends to a cache of length $T$ rather
than to a fresh $T \times T$ block:

$$
\text{FLOPs per generated token}
 = \underbrace{8d^2}_{\text{projections}}
 + \underbrace{4Td}_{\text{attend over the cache}}
$$ (eq:mha-decode-flops)

$$
\text{bytes read per generated token}
 = \underbrace{8d^2 b}_{\text{weights}}
 + \underbrace{2Tgd_k b}_{\text{the cache}}
$$ (eq:mha-decode-bytes)

Divide them and you get {{ch:dl-forward}}'s arithmetic intensity. For the
weights it is $1/b$ — one operation per byte, hopelessly memory-bound at batch
size 1, which is why serving batches requests. For the cache term the intensity
is $2h/(gb)$: **independent of $T$, and proportional to the ratio of query heads
to key/value heads.**

That is a second and less obvious argument for grouped-query attention. Sharing
key/value heads does not only shrink the cache; it *raises the arithmetic
intensity* of reading it, because $h$ query heads now do their work against
$g$ loaded key/value heads instead of $h$. Multi-query attention at $g=1$ gives
$h$ operations per loaded element where multi-head gives one.

**So MQA and GQA help twice, and the second effect is the one people miss.**
{{cite:shazeer2019mqa}}'s title — *one write-head is all you need* — is about
the bandwidth, not the memory.

## 6. Mathematical Foundation

### 6.1 The rank bottleneck

The score matrix for one head is

$$
\mat{S} = \frac{\mat{Q}\mat{K}\T}{\sqrt{d_k}}
 = \frac{\mat{X}\mat{W}^Q(\mat{X}\mat{W}^K)\T}{\sqrt{d_k}}
 = \frac{\mat{X}\,\mat{W}^Q\mat{W}^{K\top}\,\mat{X}\T}{\sqrt{d_k}}
$$ (eq:score-factorisation)

$\mat{S}$ is $T \times T$ and factors through $\mat{W}^Q\mat{W}^{K\top}$, which
is $d \times d$ of rank at most $d_k$. Hence

$$
\rank(\mat{S}) \le \min(T,\ d_k)
$$ (eq:rank-bound)

**A head cannot express an arbitrary $T\times T$ pattern once $T > d_k$.** With
$d_k = 64$ and $T = 2048$, it is confined to a rank-64 subspace of a
$2048\times2048$ space.

Two consequences worth separating.

**Heads are a way to raise the total rank.** $h$ heads of rank $d_k$ give an
attention *operator* whose combined action has rank up to $h d_k = d$. Splitting
one $d$-dimensional head into $h$ narrower ones does not lose total rank — it
redistributes it into $h$ independently-normalised pieces.

**But each piece is separately softmaxed.** That is what makes them different
from one rank-$d$ head: the softmax is applied per head, so the $h$ patterns are
each a valid probability distribution rather than components of one. Multi-head
attention is not a low-rank factorisation of single-head attention; it is a
genuinely different operator.

### 6.2 $\mat{W}^O$ decomposes per head

Partition $\mat{W}^O$ by rows into $h$ blocks $\mat{W}^O_i \in
\R^{d_k \times d}$. Then

$$
\MHA(\mat{X}) = \big[\head_1;\dots;\head_h\big]\mat{W}^O
 = \sum_{i=1}^{h}\head_i\,\mat{W}^O_i
$$ (eq:mha-sum)

**Multi-head attention is a SUM of per-head contributions, not a
concatenation.** Each head independently reads from the residual stream (through
$\mat{W}^Q_i, \mat{W}^K_i, \mat{W}^V_i$) and independently writes back to it
(through $\mat{W}^O_i$).

This is the observation that makes head-level analysis coherent at all: a head
is a self-contained circuit with its own input and output maps, and the block's
output is the sum of what the heads wrote. It is also why pruning a head is
well-defined — you delete one term from {{eq:mha-sum}}.

### 6.3 Only two matrices matter per head

From {{eq:score-factorisation}} the scores depend on $\mat{W}^Q$ and
$\mat{W}^K$ only through their product $\mat{W}^{QK} = \mat{W}^Q\mat{W}^{K\top}$.
Similarly the output depends on $\mat{W}^V$ and $\mat{W}^O_i$ only through
$\mat{W}^{VO}_i = \mat{W}^V_i\mat{W}^O_i$.

So each head has two functionally meaningful $d\times d$ matrices, both of rank
at most $d_k$:

$$
\mat{W}^{QK}_i \ \text{(where to read from)},
\qquad
\mat{W}^{VO}_i \ \text{(what to write)}
$$ (eq:qk-ov-circuits)

**The four-matrix presentation is a parameterisation, not four independent
things.** Any factorisation of $\mat{W}^{QK}$ into $\mat{W}^Q\mat{W}^{K\top}$
gives an identical function, which is a gauge freedom of the same kind as
{{ch:dl-autoencoders}}'s linear autoencoder. It matters when you try to
interpret $\mat{W}^Q$ on its own: you cannot, because it is not identifiable.

### 6.4 Why the softmax forces a sink

For any query, $\sum_j \alpha_{ij} = 1$ exactly. There is no "attend to nothing"
option.

Suppose a head's useful behaviour is conditional — it should fire on some inputs
and not others. On inputs where it should not fire, its scores are all
uninformative and the softmax spreads mass over all $T$ positions, so the head
writes the *mean* value vector into the residual stream. That is a nonzero and
input-dependent perturbation the model must tolerate.

The alternative the model can learn is to put nearly all the mass on one fixed,
low-information position, so the head writes an approximately constant vector
that later layers can subtract. **The first token is the natural choice: it
exists in every sequence and its content is usually a start-of-sequence marker.**

This predicts something testable: heads should sink more on inputs where they
have nothing to do, and the sink token's *value* vector should matter less than
its presence. Both are observed, and the practical consequence —
never evict token 0 from the cache — follows directly.

### 6.5 The KV cache arithmetic

During generation ({{ch:tf-masking-kv}}), keys and values for all previous
positions are cached. For $L$ layers, $g$ key/value heads and head dimension
$d_k$, in $b$ bytes per element:

$$
M_{\text{KV}} = 2\,b\,L\,g\,d_k\,T\ \text{per sequence}
$$ (eq:kv-cache-size)

For a 70B-class model — $L = 80$, $h = 64$, $d_k = 128$, bf16 — with full
multi-head attention at $T = 8192$:

$$
2 \times 2 \times 80 \times 64 \times 128 \times 8192
 \approx 21\ \text{GB per sequence}
$$

**Per sequence.** Serving ten concurrent users needs 210 GB of cache on top of
140 GB of weights. With GQA at $g = 8$ the cache falls to 2.7 GB per sequence,
which is the difference between serving one user and serving eighty.

That single calculation is why {{tbl:mha-variants}}'s fourth column is the
column that decided the architecture.

## 7. Internal Mechanics

### 7.1 One matmul, not $h$

The $h$ projections $\mat{X}\mat{W}^Q_i$ are computed as a single
$\mat{X}\mat{W}^Q$ with $\mat{W}^Q \in \R^{d\times d}$, then reshaped. Doing $h$
separate small matmuls would be far slower for identical arithmetic —
{{ch:dl-forward}}'s intensity argument.

Many implementations fuse further, concatenating $\mat{W}^Q$, $\mat{W}^K$ and
$\mat{W}^V$ into one $d \times 3d$ matrix so the whole projection is one matmul.
This is why checkpoint files often contain a single `qkv_proj` tensor.

### 7.2 The heads are a batch dimension

After the transpose, the score computation is a batched matrix multiply over
$B \times h$ independent problems. No loop over heads appears in a competent
implementation, and the head count affects only the shape.

### 7.3 What FlashAttention changes here

{{cite:dao2022flash}} never materialises the $(B, h, T, T)$ score tensor. It
tiles the computation so the scores exist only in on-chip memory, block by
block, and it produces the *same* output.

The consequence for this chapter is that the $h$ in the memory cost largely
disappears at training time, while the KV cache's $g$ at inference time does
not — because the cache must persist across decoding steps and cannot be
recomputed. **Training memory and serving memory are different problems and
FlashAttention only addresses one of them.**

### 7.4 Pruning

Because of {{eq:mha-sum}}, dropping head $i$ means dropping one summand. In
practice, per-head importance is estimated by the change in loss when the head
is zeroed, and a substantial fraction can go with little degradation.

The finding is robust and its interpretation is not. A prunable head might be
redundant, or it might matter only on inputs the evaluation set does not
contain. Pruning studies measure the first and are usually read as establishing
the second.

### 7.5 Head dimension in practice

$d_k = 64$ is near-universal — GPT-2, Llama, Mistral all use it — and $h$ is
varied to reach the target width. That convention predates the hardware reason
for it, which is that 64 is a comfortable tile size, and it has survived because
{{eq:rank-bound}} at $d_k = 64$ evidently suffices.

Some recent models use $d_k = 128$. Whether the extra rank helps or the tiling
does is not cleanly separated in the published comparisons.

## 8. Implementation

```python {tier=A name=multi-head-from-scratch}
"""Multi-head attention: the shapes, the parameter accounting, and the
transpose that silently breaks it.
"""
import numpy as np

rng = np.random.default_rng(0)


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


class MultiHeadAttention:
    """Eqs. 64.1-64.3, written with the heads as a batch dimension."""

    def __init__(self, d, h, seed=0):
        assert d % h == 0, "d must be divisible by h"
        rs = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(d)
        self.Wq = rs.normal(0, s, (d, d))
        self.Wk = rs.normal(0, s, (d, d))
        self.Wv = rs.normal(0, s, (d, d))
        self.Wo = rs.normal(0, s, (d, d))
        self.d, self.h, self.dk = d, h, d // h

    def n_params(self):
        return 4 * self.d * self.d

    def _split(self, X):
        B, T, _ = X.shape
        return X.reshape(B, T, self.h, self.dk).transpose(0, 2, 1, 3)

    def _merge(self, X):
        B, h, T, dk = X.shape
        return X.transpose(0, 2, 1, 3).reshape(B, T, h * dk)

    def forward(self, X, mask=None, keep=False):
        Q = self._split(X @ self.Wq)                  # (B, h, T, dk)
        K = self._split(X @ self.Wk)
        V = self._split(X @ self.Wv)
        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.dk)
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        A = softmax(scores)                           # (B, h, T, T)
        out = self._merge(A @ V) @ self.Wo
        if keep:
            self.A = A
        return out


print("=" * 72)
print("multi-head attention costs the SAME as single-head (eqs. 64.5-64.6)")
print("=" * 72)
d, T, B = 512, 64, 2
X = rng.normal(size=(B, T, d))
print(f"model width d = {d}, sequence length T = {T}\n")
print(f"{'heads h':>9} {'d_k':>6} {'parameters':>12} {'proj MFLOPs':>13} "
      f"{'score MFLOPs':>14} {'output shape':>16}")
outs = {}
for h in (1, 2, 4, 8, 16, 64):
    m = MultiHeadAttention(d, h, seed=1)
    out = m.forward(X)
    outs[h] = out
    proj = 8 * T * d * d
    score = 4 * T * T * d
    print(f"{h:>9} {d // h:>6} {m.n_params():>12,} {proj / 1e6:>13.2f} "
          f"{score / 1e6:>14.2f} {str(out.shape):>16}")

print("\nEvery row is identical in parameters and in FLOPs. The h per-head")
print("matrices of shape (d, d_k) concatenate into one (d, d) matrix, and")
print("the h score matrices are each T-by-T-by-d_k with h*d_k = d.")
print("\nSo the number of heads is FREE in both. That is the fact people")
print("most often get wrong about multi-head attention: it is not h copies")
print("of anything, it is a partition of dimensions that already existed.")

# --- section 5.3: the transpose that silently breaks it ---------------------
print("\n" + "=" * 72)
print("the reshape that fails silently (section 5.3)")
print("=" * 72)
m = MultiHeadAttention(d, 8, seed=1)
Q = m._split(X @ m.Wq)
K = m._split(X @ m.Wk)
V = m._split(X @ m.Wv)
A = softmax(Q @ K.transpose(0, 1, 3, 2) / np.sqrt(m.dk))
heads = A @ V                                       # (B, h, T, dk)

correct = heads.transpose(0, 2, 1, 3).reshape(B, T, d)
wrong = heads.reshape(B, T, d)                      # NO transpose

print(f"head output tensor            : {heads.shape}")
print(f"correct merge (transpose then reshape) : {correct.shape}")
print(f"wrong merge   (reshape only)           : {wrong.shape}")
print(f"\nSAME SHAPE, so nothing raises. Are they the same values?")
print(f"  max |correct - wrong| = {np.abs(correct - wrong).max():.4f}")
print(f"  fraction of entries that differ = "
      f"{float((np.abs(correct - wrong) > 1e-12).mean()):.4f}")

print("\nThe wrong version interleaves the head axis with the position axis,")
print("so token t's output vector is assembled from other tokens' head")
print("outputs. It has the right shape, the right dtype, and it trains — to")
print("a worse loss, for no visible reason.")
print("\nThis is Chapter 51's silent-broadcast lesson in a new costume, and")
print("the remedy is the same: assert the shape AND check a known-good")
print("value. A single-head model is the check — at h = 1 the transpose is")
print("a no-op, so correct and wrong must agree exactly:")
m1 = MultiHeadAttention(d, 1, seed=1)
Q1, K1, V1 = m1._split(X @ m1.Wq), m1._split(X @ m1.Wk), m1._split(X @ m1.Wv)
h1 = softmax(Q1 @ K1.transpose(0, 1, 3, 2) / np.sqrt(m1.dk)) @ V1
print(f"  h=1: max |transposed - not| = "
      f"{np.abs(h1.transpose(0, 2, 1, 3).reshape(B, T, d) - h1.reshape(B, T, d)).max():.3e}")

# --- section 6.2: MHA is a SUM over heads -----------------------------------
print("\n" + "=" * 72)
print("multi-head attention is a SUM of per-head terms (eq. 64.8)")
print("=" * 72)
h = 8
m = MultiHeadAttention(d, h, seed=1)
full = m.forward(X)

Q = m._split(X @ m.Wq)
K = m._split(X @ m.Wk)
V = m._split(X @ m.Wv)
A = softmax(Q @ K.transpose(0, 1, 3, 2) / np.sqrt(m.dk))
per_head = A @ V                                     # (B, h, T, dk)

total = np.zeros_like(full)
contrib = []
for i in range(h):
    Wo_i = m.Wo[i * m.dk:(i + 1) * m.dk, :]          # rows for head i
    term = per_head[:, i] @ Wo_i                     # (B, T, d)
    total += term
    contrib.append(float(np.sqrt((term ** 2).mean())))

print(f"max |sum of per-head terms  -  full MHA output| = "
      f"{np.abs(total - full).max():.3e}")
print("\nExact to floating point. Eq. 64.8 says W_O partitions by rows into")
print("h blocks, one per head, so each head reads from the residual stream")
print("and writes back to it independently and the block's output is the")
print("SUM of what they wrote.")
print("\nThat is what makes head-level analysis coherent: a head is a")
print("self-contained circuit, not a slice of an entangled computation. It")
print("is also why pruning a head is well defined — you delete one term.\n")
print("RMS contribution of each head to the output:")
for i, c in enumerate(contrib):
    bar = "#" * int(40 * c / max(contrib))
    print(f"  head {i}  {c:>8.4f}  {bar}")
print("\n(untrained, so these differ only by the random draw — the point is")
print(" that the decomposition EXISTS, not what it says about this model)")
```

```python {tier=A name=rank-and-heads}
"""The rank bottleneck of eq. 64.7, measured: what a single narrow head can
and cannot express, and what several of them buy.
"""
import numpy as np

rng = np.random.default_rng(2)


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


# --- section 6.1: the rank bound, verified ----------------------------------
print("=" * 72)
print("a head's score matrix has rank at most d_k (eq. 64.7)")
print("=" * 72)
T, d = 128, 256
X = rng.normal(size=(T, d))
print(f"T = {T} positions, model width d = {d}\n")
print(f"{'d_k':>6} {'score matrix':>15} {'numerical rank':>16} "
      f"{'bound min(T, d_k)':>19}")
for dk in (4, 16, 64, 128, 256):
    Wq = rng.normal(0, 1 / np.sqrt(d), (d, dk))
    Wk = rng.normal(0, 1 / np.sqrt(d), (d, dk))
    S = (X @ Wq) @ (X @ Wk).T / np.sqrt(dk)
    sv = np.linalg.svd(S, compute_uv=False)
    r = int((sv > sv.max() * 1e-10).sum())
    print(f"{dk:>6} {str(S.shape):>15} {r:>16} {min(T, dk):>19}")

print("\nThe bound is tight: the score matrix's rank is exactly min(T, d_k)")
print("at every width. A head of dimension 64 attending over 128 positions")
print("is confined to a rank-64 subspace of a 128-by-128 space, and at a")
print("realistic T = 2048 it is a rank-64 subspace of a 2048-by-2048 one.")

# --- what that costs: fit a target attention pattern ------------------------
print("\n" + "=" * 72)
print("what the rank bound COSTS: fitting a target attention pattern")
print("=" * 72)
print("Construct a target attention matrix of known rank and ask heads of")
print("various widths to reproduce it. This isolates eq. 64.7 from every")
print("other property of a trained model.\n")


def make_target(T, rank, seed):
    """A row-stochastic T-by-T matrix built from a rank-r score matrix."""
    rs = np.random.default_rng(seed)
    U = rs.normal(size=(T, rank))
    Vv = rs.normal(size=(T, rank))
    return softmax(U @ Vv.T * 2.0, axis=-1)


def fit_head(X, target, dk, steps=3000, lr=0.02, seed=0):
    """Learn W_q, W_k so that softmax(XWq (XWk)^T / sqrt(dk)) ~ target."""
    rs = np.random.default_rng(seed)
    d = X.shape[1]
    Wq = rs.normal(0, 1 / np.sqrt(d), (d, dk))
    Wk = rs.normal(0, 1 / np.sqrt(d), (d, dk))
    ps = [Wq, Wk]
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    for t in range(1, steps + 1):
        Q, K = X @ Wq, X @ Wk
        S = Q @ K.T / np.sqrt(dk)
        A = softmax(S, axis=-1)
        # cross-entropy between target rows and predicted rows
        dS = (A - target) / len(X)
        gWq = X.T @ (dS @ K) / np.sqrt(dk)
        gWk = X.T @ (dS.T @ Q) / np.sqrt(dk)
        for i, (p, g) in enumerate(zip(ps, [gWq, gWk])):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            p -= lr * (m[i] / (1 - 0.9 ** t)) / (
                np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    A = softmax((X @ Wq) @ (X @ Wk).T / np.sqrt(dk), axis=-1)
    return float(np.abs(A - target).mean())


T, d = 64, 128
X = rng.normal(size=(T, d))
print(f"{'target rank':>12} " + " ".join(f"{f'd_k={dk}':>10}"
                                         for dk in (2, 8, 32, 64)))
for rank in (2, 8, 32):
    tgt = make_target(T, rank, seed=rank)
    row = [fit_head(X, tgt, dk, seed=5) for dk in (2, 8, 32, 64)]
    print(f"{rank:>12} " + " ".join(f"{e:>10.4f}" for e in row))

print("\nRead the leftmost column down: a head of width 2 handles a rank-2")
print("target and does markedly worse on higher-rank ones. That is eq. 64.7")
print("binding.")
print("\nBut notice how quickly the constraint relaxes. A head of width 8")
print("fits a rank-32 target almost exactly, which the rank bound alone")
print("would not predict.")
print("\nThe reason is that eq. 64.7 bounds the SCORE matrix, and the target")
print("here is an ATTENTION matrix — the softmax of a score matrix. The")
print("softmax is nonlinear, so a low-rank score matrix can produce a very")
print("good approximation of a higher-rank row-stochastic matrix. Sharpening")
print("a few directions is enough to reproduce most of the mass.")
print("\nThat is worth getting right, because the rank bound is frequently")
print("quoted as though it limited attention PATTERNS. It limits the scores.")
print("The patterns are the softmax of those scores, and the softmax buys")
print("back a great deal — which is part of why d_k = 64 has been adequate")
print("for sequence lengths in the thousands.")

# --- and what SEVERAL heads buy ---------------------------------------------
print("\n" + "=" * 72)
print("what several narrow heads buy over one narrow head")
print("=" * 72)
print("A target that is a MIXTURE of distinct low-rank patterns — the")
print("situation section 4.1 describes, where a sentence needs several")
print("relationships attended to at once.\n")


def make_mixture_target(T, n_patterns, rank, seed):
    """n distinct attention patterns; each query row uses one of them."""
    rs = np.random.default_rng(seed)
    pats = [make_target(T, rank, seed=seed * 10 + i)
            for i in range(n_patterns)]
    which = rs.integers(0, n_patterns, T)
    return np.stack([pats[which[i]][i] for i in range(T)]), pats, which


def fit_multihead(X, target, h, dk, steps=3000, lr=0.02, seed=0):
    """h heads, each of width dk, averaged. Fits the MEAN of the heads to
    the target — a crude stand-in for what W_O's mixing allows."""
    rs = np.random.default_rng(seed)
    d = X.shape[1]
    Wq = [rs.normal(0, 1 / np.sqrt(d), (d, dk)) for _ in range(h)]
    Wk = [rs.normal(0, 1 / np.sqrt(d), (d, dk)) for _ in range(h)]
    g = rs.normal(0, 0.5, (len(X), h))            # per-position head mixing
    ps = Wq + Wk + [g]
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    for t in range(1, steps + 1):
        As, Qs, Ks = [], [], []
        for i in range(h):
            Q, K = X @ Wq[i], X @ Wk[i]
            As.append(softmax(Q @ K.T / np.sqrt(dk), axis=-1))
            Qs.append(Q)
            Ks.append(K)
        w = softmax(g, axis=-1)                   # (T, h)
        A = sum(w[:, i:i + 1] * As[i] for i in range(h))
        dA = (A - target) / len(X)
        gg = np.stack([(dA * As[i]).sum(axis=1) for i in range(h)], axis=1)
        gg = w * (gg - (gg * w).sum(axis=1, keepdims=True))
        grads = []
        for i in range(h):
            dAi = dA * w[:, i:i + 1]
            dS = As[i] * (dAi - (dAi * As[i]).sum(axis=1, keepdims=True))
            grads.append(X.T @ (dS @ Ks[i]) / np.sqrt(dk))
        for i in range(h):
            dAi = dA * w[:, i:i + 1]
            dS = As[i] * (dAi - (dAi * As[i]).sum(axis=1, keepdims=True))
            grads.append(X.T @ (dS.T @ Qs[i]) / np.sqrt(dk))
        grads.append(gg)
        for i, (p, gr) in enumerate(zip(ps, grads)):
            m[i] = 0.9 * m[i] + 0.1 * gr
            v[i] = 0.999 * v[i] + 0.001 * gr * gr
            p -= lr * (m[i] / (1 - 0.9 ** t)) / (
                np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    As = [softmax((X @ Wq[i]) @ (X @ Wk[i]).T / np.sqrt(dk), axis=-1)
          for i in range(h)]
    w = softmax(g, axis=-1)
    A = sum(w[:, i:i + 1] * As[i] for i in range(h))
    return float(np.abs(A - target).mean())


T, d = 48, 96
X = rng.normal(size=(T, d))
tgt, pats, which = make_mixture_target(T, 4, rank=6, seed=3)
print(f"target: 4 distinct rank-6 patterns, each query row using one\n")
print(f"{'configuration':<26} {'total rank budget':>19} {'fit error':>12}")
for label, h, dk in (("1 head of width 24", 1, 24),
                     ("1 head of width 6", 1, 6),
                     ("4 heads of width 6", 4, 6),
                     ("8 heads of width 3", 8, 3)):
    e = fit_multihead(X, tgt, h, dk, seed=5)
    print(f"{label:<26} {h * dk:>19} {e:>12.4f}")

print("\nThe first two rows are the same total rank budget spent two ways —")
print("one wide head against one narrow one — and the wide one wins, which")
print("is just eq. 64.7 again.")
print("\nThe interesting comparison is rows 1 and 3: the SAME total rank")
print("budget of 24, as one head of width 24 or four heads of width 6. The")
print("target is a mixture of four distinct patterns, and four heads can")
print("hold one each while one head must find a single rank-24 score matrix")
print("that produces all four after a softmax.")
print("\nThat is the argument of section 4.1 made concrete, and it is a")
print("statement about the SOFTMAX rather than about rank. Each head is")
print("normalised separately, so h heads give h independent probability")
print("distributions — which is something one head of any width cannot")
print("produce, because it has only one softmax.")
```

## 9. Practical Example

```python {tier=A name=heads-in-a-trained-model}
"""Head specialisation, the attention sink, and the KV-cache arithmetic that
decided the architecture.
"""
import numpy as np

rng = np.random.default_rng(6)


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


# --- a task with several distinct relationships -----------------------------
V, T = 16, 12


def make_task(n, seed):
    """Three relationships in one sequence, so a good model needs at least
    three distinct attention patterns:
      - the answer depends on the PREVIOUS token (a local relationship)
      - and on the FIRST token (a global one)
      - and on the token that MATCHES the last one (a content-based one)
    """
    rs = np.random.default_rng(seed)
    X = rs.integers(1, V, (n, T))
    prev = X[:, -2]
    first = X[:, 0]
    last = X[:, -1]
    match = np.zeros(n, dtype=int)
    for i in range(n):
        hits = np.where(X[i, :-1] == last[i])[0]
        match[i] = X[i, hits[0] + 1] if len(hits) else 0
    y = (prev + first + match) % V
    return X, y


class TinyTransformerBlock:
    """One multi-head attention block plus a readout. Enough to see heads
    specialise, small enough to train in NumPy."""

    def __init__(self, d=48, h=3, seed=0):
        rs = np.random.default_rng(seed)
        self.E = rs.normal(0, 0.3, (V, d))
        self.P = rs.normal(0, 0.3, (T, d))
        s = 1 / np.sqrt(d)
        self.Wq = rs.normal(0, s, (d, d))
        self.Wk = rs.normal(0, s, (d, d))
        self.Wv = rs.normal(0, s, (d, d))
        self.Wo = rs.normal(0, s, (d, d))
        self.Wr = rs.normal(0, s, (d, V))
        self.br = np.zeros(V)
        self.d, self.h, self.dk = d, h, d // h

    def params(self):
        return [self.E, self.P, self.Wq, self.Wk, self.Wv, self.Wo,
                self.Wr, self.br]

    def forward(self, X, keep=False):
        n = len(X)
        self.X = X
        H = self.E[X] + self.P[None, :, :]           # (n, T, d)
        self.H = H
        Q = (H @ self.Wq).reshape(n, T, self.h, self.dk).transpose(0, 2, 1, 3)
        K = (H @ self.Wk).reshape(n, T, self.h, self.dk).transpose(0, 2, 1, 3)
        Vv = (H @ self.Wv).reshape(n, T, self.h, self.dk).transpose(0, 2, 1, 3)
        S = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.dk)
        A = softmax(S)
        self.A, self.Q, self.K, self.Vv = A, Q, K, Vv
        Hd = A @ Vv                                  # (n, h, T, dk)
        self.Hd = Hd
        merged = Hd.transpose(0, 2, 1, 3).reshape(n, T, self.d)
        self.merged = merged
        self.O = merged @ self.Wo
        self.read = self.O[:, -1, :]                 # read from the last pos
        return self.read @ self.Wr + self.br

    def grads(self, X, y):
        n = len(X)
        logits = self.forward(X)
        m_ = logits.max(axis=1, keepdims=True)
        e = np.exp(logits - m_)
        p = e / e.sum(axis=1, keepdims=True)
        loss = float(-np.log(np.clip(p[np.arange(n), y], 1e-12, None)).mean())
        dl = p.copy()
        dl[np.arange(n), y] -= 1.0
        dl /= n
        gWr, gbr = self.read.T @ dl, dl.sum(axis=0)
        dread = dl @ self.Wr.T                       # (n, d)
        dO = np.zeros_like(self.O)
        dO[:, -1, :] = dread
        gWo = self.merged.reshape(-1, self.d).T @ dO.reshape(-1, self.d)
        dmerged = dO @ self.Wo.T
        dHd = dmerged.reshape(n, T, self.h, self.dk).transpose(0, 2, 1, 3)
        dA = dHd @ self.Vv.transpose(0, 1, 3, 2)
        dV = self.A.transpose(0, 1, 3, 2) @ dHd
        dS = self.A * (dA - (dA * self.A).sum(axis=-1, keepdims=True))
        dS /= np.sqrt(self.dk)
        dQ = dS @ self.K
        dK = dS.transpose(0, 1, 3, 2) @ self.Q
        back = lambda G: G.transpose(0, 2, 1, 3).reshape(n, T, self.d)
        Hf = self.H.reshape(-1, self.d)
        gWq = Hf.T @ back(dQ).reshape(-1, self.d)
        gWk = Hf.T @ back(dK).reshape(-1, self.d)
        gWv = Hf.T @ back(dV).reshape(-1, self.d)
        dH = (back(dQ) @ self.Wq.T + back(dK) @ self.Wk.T
              + back(dV) @ self.Wv.T)
        gP = dH.sum(axis=0)
        gE = np.zeros_like(self.E)
        np.add.at(gE, X.reshape(-1), dH.reshape(-1, self.d))
        return loss, [gE, gP, gWq, gWk, gWv, gWo, gWr, gbr]


def train(net, X, y, steps=6000, lr=3e-3, batch=128, seed=0):
    ps = net.params()
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    rs = np.random.default_rng(seed + 11)
    for t in range(1, steps + 1):
        b = rs.integers(0, len(X), batch)
        _, gs = net.grads(X[b], y[b])
        for i, (p, g) in enumerate(zip(ps, gs)):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            p -= lr * (m[i] / (1 - 0.9 ** t)) / (
                np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    return net


Xtr, ytr = make_task(12000, 1)
Xte, yte = make_task(4000, 2)

print("=" * 72)
print("does head count matter on a task with several relationships?")
print("=" * 72)
print(f"The label depends on the previous token, the FIRST token, and a")
print(f"content match — three different relationships. Chance is "
      f"{1 / V:.4f}.\n")
print(f"{'heads':>7} {'d_k':>5} {'params':>9} {'test accuracy':>15}")
nets = {}
for h in (1, 2, 3, 6):
    net = train(TinyTransformerBlock(d=48, h=h, seed=3), Xtr, ytr)
    nets[h] = net
    acc = float((net.forward(Xte).argmax(1) == yte).mean())
    print(f"{h:>7} {48 // h:>5} {sum(p.size for p in net.params()):>9,} "
          f"{acc:>15.4f}")

print("\nEvery row has an identical parameter count — eq. 64.5 — so any")
print("difference is the head structure and nothing else.")

# --- what the heads attend to -----------------------------------------------
print("\n" + "=" * 72)
print("what the heads attend to")
print("=" * 72)
net = nets[3]
net.forward(Xte[:2000])
A = net.A                                            # (n, h, T, T)
last = A[:, :, -1, :]                                # queries from position T-1
print("Attention from the LAST position (where the readout happens),")
print(f"averaged over 2000 test sequences, for each of {net.h} heads:\n")
print(f"{'head':>5}  " + " ".join(f"{j:>5}" for j in range(T)))
for i in range(net.h):
    row = last[:, i, :].mean(axis=0)
    print(f"{i:>5}  " + " ".join(f"{a:>5.2f}" for a in row))
print(f"{'':>5}  " + " ".join(f"{'':>5}" for _ in range(T - 2))
      + f"{'prev':>5} {'self':>5}")
print(f"\nposition 0 is the FIRST token; position {T - 2} is the previous one")

ent = -(last * np.log(last + 1e-12)).sum(-1).mean(0)
print(f"\nper-head entropy (max is ln {T} = {np.log(T):.3f}):")
for i in range(net.h):
    print(f"  head {i}: {ent[i]:.3f}")

print("\nThe heads are not identical, which is the claim section 4.4 calls")
print("ESTABLISHED. Whether each row corresponds to one of the three")
print("relationships in the task is a much stronger claim, and this table")
print("cannot support it — a head can contribute to several, and the")
print("residual path carries information the attention map does not show.")

# --- the attention sink -----------------------------------------------------
print("\n" + "=" * 72)
print("the attention sink (section 6.4)")
print("=" * 72)
allq = A.mean(axis=(0, 2))                           # (h, T): avg over queries
print(f"attention mass on each key position, averaged over all queries:\n")
print(f"{'head':>5}  " + " ".join(f"{j:>5}" for j in range(T)))
for i in range(net.h):
    print(f"{i:>5}  " + " ".join(f"{a:>5.2f}" for a in allq[i]))
print(f"\nuniform would be {1 / T:.3f} everywhere")
print(f"mass on position 0, averaged over heads: {allq[:, 0].mean():.4f} "
      f"({allq[:, 0].mean() * T:.1f}x uniform)")

print("\nWhether a sink appears in a model this small and this briefly")
print("trained is not guaranteed, and the table above is the answer rather")
print("than a claim. In large trained models it is pronounced and reliable.")
print("\nThe mechanism section 6.4 gives is that a softmax MUST sum to one,")
print("so a head with nothing useful to attend to still has to put its mass")
print("somewhere, and a fixed low-information position is the cheapest")
print("place. The practical consequence does not depend on the")
print("explanation being right: do not evict token 0 from a KV cache.")

# --- section 6.5: the KV cache arithmetic -----------------------------------
print("\n" + "=" * 72)
print("the arithmetic that decided the architecture (eq. 64.10)")
print("=" * 72)


def kv_gb(L, g, dk, T, b=2):
    return 2 * b * L * g * dk * T / 1e9


print("A 70B-class model: L = 80 layers, h = 64 query heads, d_k = 128,")
print("bf16. Weights are about 140 GB.\n")
print(f"{'variant':<22} {'KV heads':>9} " +
      " ".join(f"{f'T={T}':>10}" for T in (2048, 8192, 32768))
      + f" {'vs MHA':>8}")
for label, g in (("MHA", 64), ("GQA g=8", 8), ("MQA", 1)):
    row = [kv_gb(80, g, 128, T) for T in (2048, 8192, 32768)]
    print(f"{label:<22} {g:>9} " + " ".join(f"{x:>9.2f}G" for x in row)
          + f" {64 / g:>7.0f}x")

print("\nThose numbers are PER SEQUENCE. Under full multi-head attention at")
print("a 32k context, one user's cache exceeds the model's own weights.")
print("\nNow the serving question. On a machine with 640 GB of memory, after")
print("140 GB of weights:\n")
print(f"{'variant':<22} " + " ".join(f"{f'T={T}':>18}"
                                     for T in (2048, 8192, 32768)))
for label, g in (("MHA", 64), ("GQA g=8", 8), ("MQA", 1)):
    row = [int((640 - 140) / kv_gb(80, g, 128, T)) for T in
           (2048, 8192, 32768)]
    print(f"{label:<22} " + " ".join(f"{f'{x} users':>18}" for x in row))

print("\nThat is the whole argument for grouped-query attention, and it is")
print("arithmetic rather than a modelling claim. GQA gives up a little")
print("quality (ainslie2023gqa measures it) and multiplies the number of")
print("users a machine can serve by the head-sharing ratio.")
print("\nNote what it does NOT change: the parameter count and the training")
print("FLOPs are barely affected, because the key and value projections are")
print("a quarter of the attention parameters and attention is a third of")
print("the model. The decision is made almost entirely on serving memory.")
```

## 10. Production Considerations

**Choose the KV-head count from serving arithmetic, not from quality.**
Measured: at a 32k context, full multi-head attention's cache exceeds the
weights per sequence, and GQA at $g=8$ multiplies concurrent users by eight.

**Never evict token 0 from the cache.** The attention sink means it carries
disproportionate mass; dropping it degrades the model out of proportion to its
content.

**Assert the merge transpose.** Measured: the wrong reshape has the correct
shape and produces entirely different values. Check against a single-head model,
where the transpose is a no-op.

**Fuse the QKV projections into one matmul.** {{ch:dl-forward}}'s
arithmetic-intensity argument; it is why checkpoints ship a single `qkv` tensor.

**Do not tune $h$ for speed.** Measured: parameters and FLOPs are identical
across head counts. What $h$ trades is per-head rank against the number of
independent attention patterns.

**Treat head-pruning results carefully.** A head that can be pruned on your
evaluation set may matter on inputs it does not contain.

## 11. Common Mistakes

**Reshaping without the transpose.** Measured silent corruption.

**Believing more heads costs more.** Measured identical.

**Setting $d_k$ very small to get many heads.** {{eq:rank-bound}} measured: a
head cannot express a pattern of higher rank than its width, and the fit error
in {{sec:8-implementation}} shows exactly where it fails.

**Interpreting $\mat{W}^Q$ on its own.** {{eq:qk-ov-circuits}}: only the product
$\mat{W}^Q\mat{W}^{K\top}$ is identifiable.

**Assuming FlashAttention fixes the KV cache.** It removes the $T\times T$
score matrix at training time; the cache must persist across decoding steps and
cannot be recomputed.

**Reading a head's average attention map as its function.** Averages hide
conditional behaviour, which is precisely what most heads have.

## 12. Failure Modes

**Silently wrong merge.** No error, worse loss, no obvious symptom.

**Out-of-memory from the KV cache at long context.** Measured: it dominates the
weights past a few tens of thousands of tokens.

**Head collapse.** Several heads learning the same pattern, wasting the rank
budget. Detectable by correlating head outputs.

**Rank starvation.** $d_k$ too small for the patterns the task needs. Measured
in the fit-error table; the symptom is a loss floor that more training does not
move.

**Sink disruption from cache eviction or from a changed prompt prefix.**

## 13. Alternatives

**Multi-query attention** {{cite:shazeer2019mqa}} — one KV head. Maximum cache
saving, slight quality cost.

**Grouped-query attention** {{cite:ainslie2023gqa}} — the interpolation, and the
2026 default. Note the uptraining recipe: an existing multi-head checkpoint can
be converted for about 5% of pretraining compute.

**Multi-head latent attention** compresses the KV cache into a low-rank latent
and reconstructs it, reducing cache further than GQA at some compute cost.
{{maturity:EMERGING}}

**Cross-layer KV sharing** reuses one layer's cache in others, trading quality
for a proportional cache reduction. {{maturity:EMERGING}}

**A single wide head.** Cheaper to reason about and measurably worse when the
task needs several distinct patterns, as the mixture experiment shows.

## 14. Evaluation

**Check the merge against a single-head model.** Two lines.

**Verify {{eq:mha-sum}} numerically.** Measured exact here; if it fails, the
$\mat{W}^O$ partition is wrong.

**Compute the cache size before choosing a context length.**
{{eq:kv-cache-size}}.

**Measure per-head entropy and per-head output correlation.** The first
separates selecting from averaging; the second finds collapsed heads.

**Ablate heads individually** and record the loss change — the standard
importance measure, with the caveat above.

## 15. Advanced Concepts

**The QK and OV circuits.** {{eq:qk-ov-circuits}} names the two matrices that
actually determine a head's behaviour: where it reads from and what it writes.
This decomposition is the foundation of transformer-circuit analysis, and it
follows directly from {{eq:mha-sum}}.

**Induction heads.** A two-head circuit that completes repeated patterns: one
head copies the previous token's identity forward, a second attends from the
current token to the position after a previous occurrence of it. Identified in
small models and correlated with in-context learning ability.
{{maturity:EMERGING}}

**Superposition.** Evidence that models represent more features than they have
dimensions, by encoding them in overlapping directions. If true, the head is not
a natural unit and interpretability must work at the feature level instead.
{{maturity:RESEARCH FRONTIER}}

**Attention-sink tokens by design.** Some architectures add a learnable
"registers" token specifically to absorb the sink, freeing real tokens from the
role. {{maturity:EMERGING}}

**Head dimension against head count at fixed width.** {{eq:rank-bound}} says
this is a rank-versus-multiplicity trade, and the measured mixture experiment
says the multiplicity matters because of the per-head softmax. Where the optimum
sits is not settled and is likely task-dependent.

## 16. Connection to Previous Chapters

{{ch:tf-scaled-dot-product}} is the single head; this chapter runs $h$ of them
in parallel and shows the cost is unchanged. The $\sqrt{d_k}$ in
{{eq:head}} is that chapter's variance argument, applied at the *head*
dimension rather than the model dimension — which is why narrowing the heads
does not require re-deriving it.

{{ch:tf-why-attention}} gave the reason for attention at all, and its measured
non-identifiability of attention weights is sharpened here by
{{eq:qk-ov-circuits}}: not only are the weights ambiguous, so is the
factorisation that produced them.

{{ch:dl-forward}} explains why the projections are one fused matmul.
{{ch:math-matrices}} supplies the rank bound. {{ch:dl-autoencoders}}'s gauge
freedom in a linear autoencoder is the same phenomenon as
{{eq:qk-ov-circuits}}'s.

Forward: {{ch:tf-masking-kv}} makes the cache concrete.
{{ch:tf-complexity}} accounts for the $h$ that this chapter shows is free in
FLOPs and not free in memory. {{ch:tf-efficient}} is what to do about it.

## 17. Exercises

**Beginner**

1. Write the shape of every intermediate tensor in multi-head attention.
2. Why do $h$ heads cost the same as one head?
3. What does $\mat{W}^O$ do?
4. What is the attention sink?
5. What does GQA share, and why?

**Intermediate**

6. Derive {{eq:rank-bound}} from {{eq:score-factorisation}}.
7. Derive {{eq:mha-sum}} by partitioning $\mat{W}^O$.
8. Use {{eq:kv-cache-size}} to size the cache for $L=32$, $h=32$, $d_k=128$,
   $T=4096$, bf16, and again with $g=4$.
9. Explain why only $\mat{W}^Q\mat{W}^{K\top}$ is identifiable.
10. Why does the wrong merge produce a correctly shaped tensor?

**Advanced**

11. Show that multi-head attention is not a low-rank factorisation of
    single-head attention, and identify the operation responsible.
12. Construct a target attention pattern that $h$ heads of width $d_k$ can
    represent and one head of width $h d_k$ cannot.
13. Derive the backward pass through the head split and merge.
14. Explain how an induction head works and what two-layer structure it needs.

**Implementation**

15. Implement multi-head attention with the backward pass and gradient-check
    it.
16. Reproduce the rank-fitting table and extend it to $T = 256$.
17. Implement GQA and verify the cache saving matches {{eq:kv-cache-size}}.
18. Measure per-head output correlation on a trained model to find collapse.

**Reasoning**

19. Your model trains to a worse loss after you refactored the attention
    block, with no error. What do you check first?
20. You have memory for 8 concurrent users and need 64. What are your options,
    in order of how much they cost you?

## 18. Interview Questions

**"Why multiple heads?"** — Several relationships at once, and the per-head
softmax that makes them independent distributions. The strong answer adds
{{eq:rank-bound}}.

**"Do more heads cost more?"** — No, in parameters or FLOPs. Yes, in per-head
rank. Say which.

**"What does $\mat{W}^O$ do?"** — {{eq:mha-sum}}: it lets each head write to any
direction, and it makes the block a sum of per-head contributions.

**"What is GQA and why does it exist?"** — The cache arithmetic. Give a number.

**"What is the attention sink?"** — The softmax must sum to one. Say the
practical consequence about cache eviction.

**"Can you interpret an attention head?"** — Partially. Say what is established
(heads differ, some circuits identified, many are prunable) and what is not (one
function per head, transfer to scale).

## 19. Research Questions

**Is the head the right unit of analysis?** Superposition results suggest
features are distributed across heads and layers rather than localised in them.
{{maturity:RESEARCH FRONTIER}}

**Why can so many heads be pruned?** Redundancy, or importance on inputs the
evaluation misses, are not cleanly separated by existing studies.
{{maturity:EMERGING}}

**What is the optimal head dimension?** $d_k = 64$ is conventional and the
convention predates a clear justification; recent models using 128 do not
isolate rank from tiling. {{maturity:EMERGING}}

**How far can the KV cache be compressed?** Latent attention and cross-layer
sharing both work; the limit is unknown. {{maturity:EMERGING}}

## 20. Chapter Summary

Multi-head attention runs $h$ attention operations in parallel over a partition
of the model's dimensions, and the measurement confirms the accounting people
most often get wrong: **parameters and FLOPs are identical across every head
count**, because the $h$ per-head matrices concatenate into the same
$d \times d$ projections and the $h$ score matrices have $h d_k = d$. The number
of heads is free in both.

What it costs is rank. {{eq:rank-bound}} says a head's $T \times T$ score matrix
factors through a rank-$d_k$ product, verified here to be exactly tight, and the
fitting experiment turns the inequality into a capability: a head wider than a
target pattern's rank reproduces it and a narrower one cannot. At $d_k = 64$ and
$T = 2048$, a head is confined to a rank-64 subspace of a $2048^2$-dimensional
space.

Heads are not simply a low-rank factorisation of one wide head, and the
measurement isolates why. Given the same total rank budget, four heads of width
6 beat one head of width 24 on a target that mixes four distinct patterns —
because **each head is softmaxed separately**, so $h$ heads produce $h$
independent probability distributions and one head of any width produces one.
The operation responsible is the normalisation, not the projection.

{{eq:mha-sum}} was verified exactly: partitioning $\mat{W}^O$ by rows shows
multi-head attention is a *sum* of per-head contributions rather than a
concatenation. Each head reads from the residual stream and writes back to it
independently, which is what makes head-level analysis coherent and pruning
well-defined. The same decomposition gives {{eq:qk-ov-circuits}}: only
$\mat{W}^Q\mat{W}^{K\top}$ and $\mat{W}^V\mat{W}^O_i$ are identifiable, so
interpreting $\mat{W}^Q$ alone is not meaningful.

On specialisation, the honest position is narrow. The measured heads differ from
one another — that much is established, along with identified circuits in small
models and the robust finding that many heads prune cheaply. That each head has
one interpretable function, that the head is the right unit, and that small-model
findings scale, are not established.

Finally, the arithmetic that decided the architecture. {{eq:kv-cache-size}}
measured on a 70B-class model gives a per-sequence cache that *exceeds the
weights* at a 32k context under full multi-head attention. Grouped-query
attention at $g = 8$ cuts it eightfold and multiplies concurrent users by the
same factor, at a small measured quality cost and — via
{{cite:ainslie2023gqa}}'s uptraining recipe — about 5% of pretraining compute to
convert an existing model. The decision is made almost entirely on serving
memory, and it barely touches the parameter count or the training FLOPs.

## 21. Further Reading

{{cite:vaswani2017}} section 3.2.2 is where multi-head attention is defined, and
it is worth noticing how brief the justification is: essentially one sentence
about attending to information from different representation subspaces. The
rank argument of {{sec:6-mathematical-foundation}} is a later reconstruction.

{{cite:shazeer2019mqa}} is four pages and the diagnosis is the contribution.
Read the first page for how cleanly it identifies memory bandwidth — not
compute — as the incremental-decoding bottleneck; the fix follows immediately.

{{cite:ainslie2023gqa}} for grouped-query attention and, more usefully, for the
uptraining recipe. The observation that an existing checkpoint can be converted
for a few per cent of pretraining compute is what made adoption immediate.

{{cite:dao2022flash}} again, read here for what it does and does not fix. It
removes the $T \times T$ materialisation at training time and leaves the KV
cache untouched, and keeping those two costs separate is most of what
{{ch:tf-complexity}} is about.

**Where to go next:** {{ch:tf-positional}}. Attention as defined here is
permutation-equivariant — shuffle the input positions and the outputs shuffle
identically — so the architecture has no notion of order at all until something
supplies it.

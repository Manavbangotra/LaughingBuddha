---
id: tf-embeddings
number: 66
part: VII
tier: full
status: reviewed
requires: [tf-positional, tf-multi-head, dl-losses, dl-initialization]
provides: [token-embedding, unembedding, weight-tying, logit-lens,
           embedding-initialisation, vocabulary-size, softmax-bottleneck,
           embedding-memory]
citations: [vaswani2017, press2017tying, dosovitskiy2021vit, zhang2019rmsnorm]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Explain what an embedding table is and why it is a lookup rather than a
   matrix multiply.
2. Explain weight tying, derive its parameter saving, and state the argument
   against it.
3. Derive the softmax bottleneck and say what it limits.
4. Account for the embedding's share of a model's parameters at several scales.
5. Explain why embedding initialisation does not follow the fan-in rule.
6. Explain the logit lens and what it does and does not show.
7. Explain how the vocabulary size trades against sequence length and compute.

## 2. Why This Matters

**The embedding is where discrete meets continuous**, and it is the only place
in a transformer where that conversion happens. Everything between the
embedding and the unembedding is smooth vector arithmetic; the two ends are
lookups into a table indexed by token identity.

**At small scale the embedding is most of the model.** For GPT-2 small — a 50k
vocabulary at $d = 768$ — the two embedding matrices are 77M parameters against
85M for all twelve blocks, so they are nearly half the model. That ratio inverts as models
grow, and knowing where you are on that curve changes what you should optimise.

**Weight tying is a real decision with a real trade.** {{cite:press2017tying}}
showed it helps small models substantially. Most large models do not tie, and
{{sec:6-mathematical-foundation}} explains why the argument reverses.

**The softmax bottleneck is a genuine limitation with a clean derivation.** The
log-probability matrix a model can produce has rank at most $d$, so a
$d$-dimensional model cannot express an arbitrary distribution over a $V$-token
vocabulary for every context. It is one of the few hard expressiveness limits in
the architecture.

## 3. Prerequisites

{{ch:tf-positional}} for what gets added to the embedding.
{{ch:tf-multi-head}} for the residual stream the embedding writes into.
{{ch:dl-losses}} for the softmax and cross-entropy at the output.
{{ch:dl-initialization}} for the variance argument this chapter is an
exception to.

## 4. Intuitive Explanation

### 4.1 Two tables, one at each end

```text
   token id 4271 ──▶ [EMBEDDING TABLE] ──▶ vector in R^d
                            (V, d)
                              │
                              ▼
                     ... L transformer blocks ...
                              │
                              ▼
   vector in R^d ──▶ [UNEMBEDDING] ──▶ logits over V tokens
                          (d, V)
```

The embedding turns a token identity into a vector; the unembedding turns a
vector back into a score for every token. Both are $V \times d$ matrices and the
question of whether they should be the *same* matrix is
{{sec:5-formal-explanation}}'s subject.

### 4.2 A lookup, not a multiply

Mathematically, embedding a token is a one-hot vector times a matrix:

$$
\vec{e} = \vec{o}\T\mat{E},
\qquad \vec{o} = \text{one-hot}(t)
$$

That formulation is correct and nobody implements it. A one-hot times a matrix
is just row $t$ of the matrix, so the operation is an *indexed read*. For
$V = 50000$, the multiply would do 50000 multiply-adds per token to produce what
a single memory read produces.

**The consequence for the backward pass is the interesting part.** Only the rows
that appeared in the batch receive gradient, so the embedding's gradient is
naturally sparse — a batch of 4096 tokens touches at most 4096 of 50000 rows.
{{sec:7-internal-mechanics}} covers what optimisers do about that, and it is a
place where Adam behaves in a way people do not expect.

### 4.3 The unembedding is not a lookup

The output side has no such shortcut. Producing logits requires the *full*
$d \times V$ matrix multiply, because you need a score for every token:

```text
   embedding    one read of d numbers          O(d)
   unembedding  a d-by-V matmul per position   O(dV)
```

At $V = 128000$ and $d = 4096$, that is half a billion multiply-adds per
position — comparable to an entire transformer layer. **The unembedding is one
of the most expensive single operations in a language model**, and it is why
vocabulary size is a compute decision and not only a tokenisation one.

### 4.4 Weight tying

Use the same matrix for both ends: $\mat{U} = \mat{E}\T$.

The argument for it is that both matrices are doing the same job. The embedding
maps a token to a direction; the unembedding scores a direction against every
token. If two tokens are similar, they should be near each other in both.

The argument against is that they are *not* quite the same job. The embedding is
read at the input, where a token's identity matters; the unembedding is applied
to the residual stream after $L$ layers of processing, where the geometry has
changed. Forcing them to share means one matrix must serve both.

**Empirically the trade flips with scale**, and
{{sec:6-mathematical-foundation}} says why: the parameter saving is what matters
when embeddings are most of the model, and the constraint is what matters when
they are not.

### 4.5 The softmax bottleneck

Here is a limit that is easy to state and easy to miss.

The model produces logits $\vec{z} = \mat{U}\vec{h}$ where $\vec{h} \in \R^d$.
Over $N$ different contexts, the matrix of all logits is $\mat{H}\mat{U}\T$,
which is $N \times V$ and factors through $d$ dimensions — so its rank is at
most $d$.

**The matrix of true log-probabilities has no such constraint.** If language
requires a rank-2000 log-probability matrix and your model has $d = 768$, it
cannot express it, no matter how well trained.

This is not a training problem or a data problem. It is an expressiveness limit
that follows from the architecture in two lines.

### 4.6 Why the two ends are not symmetric

It is tempting to see the embedding and the unembedding as inverses — one maps
tokens to vectors, the other vectors to tokens. They are not, and the asymmetry
is worth naming because it is the source of several of this chapter's results.

**The embedding is exact and the unembedding is not.** Given a token, the
embedding produces *the* vector for it. Given a vector, the unembedding produces
a score for every token and a distribution over all of them. One direction is a
function; the other is an inference.

**They see different geometry.** The embedding writes into the residual stream
at layer 0, where the stream contains nothing but embeddings and positions. The
unembedding reads it after $L$ blocks have each added their output, so the norm
is much larger and the directions that matter are ones the blocks constructed.

**They are optimised by different gradients.** The unembedding gets a gradient
from every position for every token — the softmax touches all $V$ rows at every
step. The embedding gets a gradient only for the tokens present
({{eq:embed-gradient}}). **So in an untied model the unembedding is trained
$V/BT$ times more densely than the embedding**, which for a large vocabulary is
a factor of ten or more.

That last asymmetry is the one to remember, and it reframes weight tying: tying
does not merely share parameters, it **exposes the embedding to the
unembedding's dense gradient**. {{cite:press2017tying}}'s gradient analysis says
exactly this — the tied matrix evolves like the untied *output* embedding, not
like the untied input one. On that reading, tying's benefit at small scale may
have been about optimisation as much as about parameter count, which is a
different claim from the one usually made and one the measurement in
{{sec:9-practical-example}} can speak to.

## 5. Formal Explanation

### 5.1 The two matrices

$$
\mat{E} \in \R^{V\times d}
\qquad\text{(embedding)},
\qquad
\mat{U} \in \R^{V\times d}
\qquad\text{(unembedding)}
$$ (eq:embedding-matrices)

Forward:

$$
\vec{h}^{(0)}_i = \mat{E}[t_i] + \vec{p}_i,
\qquad
\vec{z}_i = \mat{U}\,\vec{h}^{(L)}_i,
\qquad
p(t \mid \cdot) = \softmax(\vec{z}_i)
$$ (eq:embed-unembed)

with $\vec{p}_i$ the positional term, absent under RoPE
({{ch:tf-positional}}).

### 5.2 Parameter accounting

$$
P_{\text{embed}} = 2Vd \ \text{(untied)}
\qquad\text{or}\qquad
Vd\ \text{(tied)}
$$ (eq:embed-params)

$$
P_{\text{blocks}} \approx 12 L d^2
\qquad\text{(4}d^2\text{ attention + 8}d^2\text{ FFN per block)}
$$ (eq:block-params)

The ratio is what matters:

$$
\frac{P_{\text{embed}}}{P_{\text{blocks}}}
 = \frac{2Vd}{12Ld^2} = \frac{V}{6Ld}
$$ (eq:embed-fraction)

**Embeddings dominate when $V > 6Ld$.** For GPT-2 small ($V=50257$, $L=12$,
$d=768$) that threshold is 55296 — so the model sits almost exactly at the
crossover, and its embeddings are indeed about a third of its parameters. For a
70B model ($V=128000$, $L=80$, $d=8192$) the threshold is $3.9\times10^6$ and
embeddings are under 1%.

### 5.3 Weight tying

$$
\mat{U} = \mat{E}
$$ (eq:weight-tying)

{{cite:press2017tying}} reports consistent improvements on language modelling at
the scales tested, and the saving is exactly $Vd$ parameters.

Three consequences worth separating:

**Parameters halve** in the embedding term of {{eq:embed-fraction}}.

**Regularisation.** One matrix trained by two objectives — appearing at the
input and the output — is constrained more than two independent ones.

**Scale coupling.** The embedding is added to the residual stream at layer 0 and
the unembedding is applied after layer $L$, where the residual stream's norm is
typically much larger ({{ch:tf-ffn-residual}}). One matrix must work at both
scales, which is why tied models often need an output scaling factor.

### 5.4 Initialisation

{{ch:dl-initialization}}'s fan-in rule does not apply. An embedding lookup is
not a sum over $n$ inputs — it selects one row — so there is no fan-in and
$\sqrt{2/n}$ has nothing to compute from.

The convention is a small fixed standard deviation:

$$
\mat{E}_{ij} \sim \mathcal{N}(0, 0.02^2)
$$ (eq:embed-init)

The reasoning is about the residual stream rather than about the embedding.
Layer 0's input is $\mat{E}[t] + \vec{p}$, and the whole stack is calibrated to
a residual stream of roughly unit scale after normalisation. A small
initialisation keeps the first block's input modest and lets the model grow the
embedding norms as it needs them.

> NOTE: A common variant scales the embedding output by $\sqrt{d}$ before the
> first block, which {{cite:vaswani2017}} does. Under weight tying this is not
> cosmetic: it lets the shared matrix be small at the input, where the residual
> stream is small, and effectively large at the output, where the logits need
> range. It is a partial fix for the scale-coupling problem above.

### 5.5 Vocabulary size

$V$ trades against three things at once:

{#tbl:vocab-tradeoffs caption="What the vocabulary size trades. The third column is the one usually forgotten: a larger vocabulary means fewer tokens per document, which reduces sequence length quadratically in attention cost."}

| Larger $V$ | Effect |
|---|---|
| Embedding parameters | grows linearly, $2Vd$ |
| Unembedding compute | grows linearly, $2dV$ per position |
| Tokens per document | *falls* — better compression |
| Attention cost | falls as the square of the token count |
| Rare-token quality | worse — fewer examples per row |

**The third and fourth rows are why large vocabularies won.** Going from 32k to
128k tokens costs 4× the embedding parameters and saves perhaps 15% of the
tokens, and attention's cost is quadratic in that count. Whether the trade pays
depends on $T$, which is why long-context models pushed vocabularies up.

## 6. Mathematical Foundation

### 6.1 The softmax bottleneck, derived

Let $\mat{H} \in \R^{N\times d}$ hold the final hidden states for $N$ contexts,
and $\mat{U} \in \R^{V\times d}$ the unembedding. The logit matrix is
$\mat{Z} = \mat{H}\mat{U}\T \in \R^{N\times V}$, so

$$
\rank(\mat{Z}) \le d
$$ (eq:logit-rank)

The log-probability matrix is $\mat{Z}$ with each row shifted by its own
constant $-\logsumexp$, so

$$
\log\mat{P} = \mat{Z} - \vec{c}\vec{1}\T,
\qquad
\rank(\log\mat{P}) \le d + 1
$$ (eq:logprob-rank)

**If the true $\log\mat{P}$ has rank greater than $d+1$, the model cannot
represent it.** The bound is on the *architecture*, not on the training.

Whether natural language actually needs a high-rank log-probability matrix is an
empirical question, and the evidence is that it needs more than the $d$ of small
models. {{sec:8-implementation}} measures the effect directly by constructing
targets of known rank.

The standard fixes both break the single-softmax structure: a **mixture of
softmaxes** takes a weighted sum of several softmax outputs, whose log is not
constrained to rank $d$; and simply making $d$ larger, which is what the field
did instead.

### 6.2 The tying argument, and where it reverses

From {{eq:embed-fraction}}, tying saves $Vd$ parameters out of
$2Vd + 12Ld^2$, a fractional saving of

$$
\frac{Vd}{2Vd + 12Ld^2} = \frac{1}{2 + 12Ld/V}
$$ (eq:tying-saving)

At GPT-2-small scale ($Ld/V = 0.18$) the saving is 31% of all parameters. At
70B scale ($Ld/V = 5.1$) it is 1.6%.

**So the benefit falls by twenty-fold across that range while the constraint
does not.** The constraint is that one matrix must serve two roles at two
different residual-stream scales, and it does not get milder as the model grows
— if anything it gets worse, because deeper stacks grow the residual norm more.

That is the whole explanation for why small models tie and large models mostly
do not, and it follows from arithmetic rather than from taste.

### 6.3 Why the embedding gradient is sparse

From {{eq:embed-unembed}}, $\vec{h}^{(0)}_i$ depends on $\mat{E}$ only through
row $t_i$. So

$$
\frac{\partial \Like}{\partial \mat{E}[v]}
 = \sum_{i\,:\,t_i = v} \frac{\partial \Like}{\partial \vec{h}^{(0)}_i}
$$ (eq:embed-gradient)

**Zero for every token not in the batch.** A batch of $B \times T$ tokens
touches at most $BT$ distinct rows, which for a large vocabulary is a small
fraction.

This interacts badly with Adam. From {{ch:dl-optimizers}}, $\vec{v}$ decays by
$\beta_2$ every step whether or not the parameter received gradient. For a rare
token's row, $\vec{v}$ decays toward zero between appearances, so when the token
finally appears, the update $\hat m/\sqrt{\hat v}$ is divided by something very
small.

**Rare tokens therefore get systematically larger steps than common ones**,
which is the opposite of what you would choose. {{sec:8-implementation}}
measures the effect.

### 6.4 What the unembedding costs

Per position, the unembedding is $2dV$ FLOPs. Per block, attention and the
feed-forward network are together about $24d^2 + 4Td$ FLOPs per position
({{ch:tf-complexity}}). The unembedding equals one block's cost when

$$
2dV \approx 24d^2
\quad\Longleftrightarrow\quad
V \approx 12d
$$ (eq:unembed-crossover)

For $d = 768$: $V \approx 9216$. **Every realistic vocabulary is far above
that**, so the unembedding costs *more* than a transformer block — for GPT-2
small at $V = 50257$ it is about 5.5 blocks' worth, out of 12.

That is a startling fraction and it is why the output layer is a target for
optimisation, and why some architectures compute the loss over a sampled subset
of the vocabulary during training.

### 6.5 The logit lens

Apply the unembedding to an *intermediate* hidden state rather than the final
one:

$$
\vec{z}^{(\ell)}_i = \mat{U}\,\vec{h}^{(\ell)}_i
$$ (eq:logit-lens)

Because the residual stream is a sum of block outputs
({{ch:tf-ffn-residual}}), $\vec{h}^{(\ell)}$ lives in the same space as
$\vec{h}^{(L)}$, so this is type-correct. It produces a distribution over tokens
at every depth, and reading it shows predictions sharpening layer by layer.

**Two cautions.** The intermediate states have not been through the final
normalisation, so their scale is wrong and the resulting distributions are
badly calibrated unless the normalisation is applied first. And the lens assumes
the unembedding is the right readout at every layer, which is an assumption —
later work fits a per-layer linear map instead, and finds it changes the
picture.

## 7. Internal Mechanics

### 7.1 The lookup

```python
h = E[token_ids]        # (B, T, d) from (V, d) and (B, T)
```

An indexed gather. Cost is $O(BTd)$ memory reads and no arithmetic. The backward
pass is a scatter-add, which needs atomics on a GPU when the same token appears
twice in a batch — a common source of nondeterminism
({{ch:mle-reproducibility}}).

### 7.2 Sparse optimiser updates

Two implementations, and they are different algorithms:

**Dense.** Materialise the full $(V, d)$ gradient, mostly zeros, and run the
optimiser over all of it. Simple, and it costs $O(Vd)$ per step regardless of
how many tokens appeared.

**Sparse.** Update only the touched rows. Far cheaper, and it changes Adam's
behaviour: $\vec{v}$ no longer decays for untouched rows, which *fixes* the
rare-token problem of {{eq:embed-gradient}} — accidentally, as a side effect of
an efficiency optimisation.

Frameworks disagree about the default, and models trained under the two can
differ measurably. This is worth checking rather than assuming.

### 7.3 Vocabulary sharding

At $V = 128000$ and $d = 8192$, the embedding is 1 GB in bf16 and the logit
tensor for a batch of 4096 tokens is 1 GB. Both are sharded across devices in
large-scale training, with the loss computed in a way that avoids gathering the
full logit tensor anywhere.

The standard trick computes the cross-entropy in two passes — a max and a sum of
exponentials, then the indexed logit — so that only per-shard reductions cross
the network. This is {{ch:dl-losses}}'s logsumexp, distributed.

### 7.4 Untrained rows

A tokeniser typically defines tokens that never appear in the training corpus.
Their embedding rows stay at initialisation, and their unembedding rows too —
so they receive whatever logit a random direction produces. Occasionally that is
large, and the model emits a token nobody has ever seen it produce.

The standard defence is to detect unused rows and set their unembedding bias (or
row) to a large negative value.

### 7.5 Embeddings as a diagnostic

The embedding matrix is a $V \times d$ table of learned vectors and it is
directly inspectable: nearest neighbours in it show what the model considers
similar *before any context*. It is the cheapest interpretability artefact in
the whole model and it is often skipped.

## 8. Implementation

```python {tier=A name=embeddings-and-tying}
"""The embedding table, weight tying, and the parameter arithmetic that
decides it.
"""
import numpy as np

rng = np.random.default_rng(0)


# --- section 4.2: a lookup, not a multiply ----------------------------------
print("=" * 72)
print("an embedding is an indexed read, not a matrix product")
print("=" * 72)
import time

V, d, n = 50000, 768, 4096
E = rng.normal(0, 0.02, (V, d))
ids = rng.integers(0, V, n)

t0 = time.perf_counter()
out_lookup = E[ids]
t_lookup = time.perf_counter() - t0

oh = np.zeros((n, V), dtype=np.float32)
oh[np.arange(n), ids] = 1.0
t0 = time.perf_counter()
out_matmul = oh @ E.astype(np.float32)
t_matmul = time.perf_counter() - t0

print(f"vocabulary {V:,}, width {d}, {n:,} tokens\n")
print(f"  indexed read      {t_lookup * 1e3:>9.3f} ms")
print(f"  one-hot @ E       {t_matmul * 1e3:>9.3f} ms   "
      f"({t_matmul / t_lookup:.0f}x slower)")
print(f"  results agree     max |diff| = "
      f"{np.abs(out_lookup - out_matmul).max():.3e}")
print(f"  wasted multiplies {2 * n * V * d / 1e9:.1f} GFLOP to read "
      f"{n * d * 8 / 1e6:.1f} MB")

print("\nSame answer, and the matmul does billions of operations to produce")
print("what a gather produces with none. Every framework implements the")
print("gather; the one-hot formulation is a notational device.")

# --- section 6.3: the gradient is sparse ------------------------------------
print("\n" + "=" * 72)
print("the embedding gradient is sparse (eq. 66.7)")
print("=" * 72)
print(f"{'batch tokens':>14} {'distinct rows':>15} {'fraction of V':>15} "
      f"{'zero rows':>12}")
for n_ in (256, 1024, 4096, 16384, 65536):
    b = rng.integers(0, V, n_)
    distinct = len(np.unique(b))
    print(f"{n_:>14,} {distinct:>15,} {distinct / V:>15.4f} "
          f"{V - distinct:>12,}")

print("\nEven a batch of 65,536 tokens leaves most of a 50,000-row table")
print("untouched, and a realistic batch touches a few per cent. So the")
print("embedding gradient is mostly zeros, by construction.")

# --- what that does to Adam -------------------------------------------------
print("\n" + "=" * 72)
print("what sparsity does to Adam (section 6.3)")
print("=" * 72)
print("Adam's v decays by beta_2 EVERY step, whether or not the parameter")
print("got a gradient. Track one row that appears every k steps.\n")


def adam_step_sizes(period, steps=4000, b1=0.9, b2=0.999, lr=1e-3, g=1.0):
    """Return the update magnitude on the steps where the row appears."""
    m = v = 0.0
    sizes = []
    for t in range(1, steps + 1):
        grad = g if (t % period == 0) else 0.0
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad * grad
        mh, vh = m / (1 - b1 ** t), v / (1 - b2 ** t)
        if grad != 0.0 and t > steps // 2:
            sizes.append(lr * abs(mh) / (np.sqrt(vh) + 1e-8))
    return float(np.mean(sizes)) if sizes else float("nan")


print(f"{'appears every':>15} {'appearances/1000 steps':>24} "
      f"{'mean |update| when seen':>26} {'vs every-step':>14}")
base = adam_step_sizes(1)
for period in (1, 10, 100, 500, 1000):
    s = adam_step_sizes(period)
    print(f"{period:>15} {1000 / period:>24.1f} {s:>26.3e} "
          f"{s / base:>13.1f}x")

print("\nThe effect is real and it is smaller than the mechanism might")
print("suggest: a row seen every thousandth step takes an update about two")
print("and a half times larger than one seen every step, from the identical")
print("gradient. Between appearances v decays toward zero and Adam divides")
print("by its square root, but the bias correction 1/(1-b2^t) partly")
print("compensates, which is why the factor is not the hundreds the naive")
print("reading would predict.")
print("\nThe direction is what matters: rare tokens take systematically")
print("BIGGER steps than common ones, which is the opposite of what anyone")
print("would choose, and it is eq. 66.7's sparsity meeting Chapter 54's")
print("dense optimiser.")
print("\nA SPARSE optimiser implementation, which updates only the touched")
print("rows, does not decay v for untouched ones and removes the effect")
print("entirely. It is a correctness fix disguised as an efficiency")
print("optimisation, and frameworks disagree about the default.")

# --- section 5.2: the parameter accounting ----------------------------------
print("\n" + "=" * 72)
print("where the parameters are (eqs. 66.3-66.5)")
print("=" * 72)
MODELS = [
    ("GPT-2 small",  50257, 12, 768),
    ("GPT-2 XL",     50257, 48, 1600),
    ("7B-class",    32000, 32, 4096),
    ("70B-class",  128000, 80, 8192),
]
print(f"{'model':<14} {'V':>8} {'L':>4} {'d':>6} {'embed M':>9} "
      f"{'blocks M':>10} {'embed %':>9} {'tying saves':>12}")
for name, V_, L_, d_ in MODELS:
    emb = 2 * V_ * d_
    blk = 12 * L_ * d_ * d_
    print(f"{name:<14} {V_:>8,} {L_:>4} {d_:>6} {emb / 1e6:>9.1f} "
          f"{blk / 1e6:>10.1f} {emb / (emb + blk):>8.1%} "
          f"{(V_ * d_) / (emb + blk):>11.1%}")

print("\nRead the last two columns together. At GPT-2-small scale the")
print("embeddings are nearly HALF the model and tying saves a quarter of")
print("all parameters; at 70B scale they are three per cent and tying saves")
print("under two.")
print("\nThat is eq. 66.9, and it is the whole explanation for why small")
print("models tie and large ones mostly do not. The BENEFIT falls by more")
print("than an order of magnitude across that range; the CONSTRAINT — one")
print("matrix serving two roles at two different residual-stream scales —")
print("does not get any milder.")

# --- section 6.4: the unembedding is expensive ------------------------------
print("\n" + "=" * 72)
print("the unembedding costs more than a transformer block (eq. 66.10)")
print("=" * 72)
print(f"{'model':<14} {'unembed GFLOP/tok':>19} {'per block':>12} "
      f"{'unembed = N blocks':>20} {'of L':>6}")
for name, V_, L_, d_ in MODELS:
    un = 2 * d_ * V_
    blk = 24 * d_ * d_
    print(f"{name:<14} {un / 1e9:>19.4f} {blk / 1e9:>12.4f} "
          f"{un / blk:>19.1f} {L_:>6}")

print(f"\nEq. 66.10 says the crossover is at V = 12d:")
for name, V_, L_, d_ in MODELS:
    print(f"  {name:<14} 12d = {12 * d_:>7,}   actual V = {V_:>7,}   "
          f"{'ABOVE' if V_ > 12 * d_ else 'below'}")

print("\nThree of the four are above the crossover, so their output")
print("projection costs more arithmetic than a transformer block — for")
print("GPT-2 small it is worth about five and a half of its twelve.")
print("\nThe 7B row is below it, and the reason is instructive: that")
print("configuration pairs a comparatively small 32,000-token vocabulary")
print("with a wide d = 4096, so 12d exceeds V. Modern models with 128k")
print("vocabularies are back above the line even at large d, which is the")
print("70B row.")
print("\nSo the rule is not 'always above'; it is that the crossover sits")
print("at V = 12d and vocabularies have been growing faster than widths.")
print("\nThat is why the vocabulary size is a compute decision and not only")
print("a tokenisation one, and why large-scale training shards the logits")
print("and computes the loss without ever gathering them.")
```

```python {tier=A name=softmax-bottleneck}
"""The softmax bottleneck (eq. 66.6): a hard expressiveness limit that
follows from the architecture in two lines.
"""
import numpy as np

rng = np.random.default_rng(2)


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


# --- section 6.1: the rank bound --------------------------------------------
print("=" * 72)
print("the logit matrix has rank at most d (eq. 66.6)")
print("=" * 72)
N, V = 200, 64
print(f"{N} contexts, vocabulary {V}\n")
print(f"{'d':>5} {'logit matrix':>15} {'numerical rank':>16} "
      f"{'log-prob rank':>15} {'bound d+1':>11}")
for d in (4, 8, 16, 64):
    H = rng.normal(size=(N, d))
    U = rng.normal(size=(V, d))
    Z = H @ U.T
    P = softmax(Z)
    logP = np.log(P)
    r1 = int((np.linalg.svd(Z, compute_uv=False) > 1e-9).sum())
    r2 = int((np.linalg.svd(logP - logP.mean(), compute_uv=False)
              > 1e-9 * np.abs(logP).max()).sum())
    print(f"{d:>5} {str(Z.shape):>15} {r1:>16} {r2:>15} {d + 1:>11}")

print("\nThe logit matrix's rank is exactly d and the log-probability")
print("matrix's is at most d+1, because the softmax subtracts a per-row")
print("constant — one extra rank-one term and nothing more.")
print("\nThat bound is on the ARCHITECTURE. No amount of training moves it,")
print("and no choice of unembedding does either.")

# --- what it costs: fit a target distribution of known rank -----------------
print("\n" + "=" * 72)
print("what the bound costs: fitting distributions of known rank")
print("=" * 72)
print("Construct a target log-probability matrix of controlled rank and ask")
print("models of various widths to reproduce it. This isolates eq. 66.6")
print("from every other property of a language model.\n")


def make_target(N, V, rank, seed):
    rs = np.random.default_rng(seed)
    A = rs.normal(size=(N, rank))
    B = rs.normal(size=(rank, V))
    return softmax(A @ B * 1.2, axis=-1)


def fit(target, d, steps=4000, lr=0.05, seed=0):
    """Learn H (N,d) and U (V,d) to match the target distribution."""
    rs = np.random.default_rng(seed)
    N_, V_ = target.shape
    H = rs.normal(0, 0.3, (N_, d))
    U = rs.normal(0, 0.3, (V_, d))
    ps = [H, U]
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    for t in range(1, steps + 1):
        P = softmax(H @ U.T, axis=-1)
        dZ = (P - target) / N_
        gH, gU = dZ @ U, dZ.T @ H
        for i, (p, g) in enumerate(zip(ps, [gH, gU])):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            p -= lr * (m[i] / (1 - 0.9 ** t)) / (
                np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    P = softmax(H @ U.T, axis=-1)
    kl = float((target * np.log(np.clip(target / np.clip(P, 1e-12, None),
                                        1e-12, None))).sum(1).mean())
    return kl


N, V = 128, 48
print(f"{'target rank':>12} " + " ".join(f"{f'd={d}':>11}"
                                         for d in (2, 4, 8, 16, 32)))
for rank in (2, 4, 8, 16):
    tgt = make_target(N, V, rank, seed=rank)
    row = [fit(tgt, d, seed=5) for d in (2, 4, 8, 16, 32)]
    print(f"{rank:>12} " + " ".join(f"{k:>11.5f}" for k in row))
print("\n(entries are KL(target || model) in nats; 0 is exact)")

print("\nRead along each row: a model whose width matches or exceeds the")
print("target's rank fits it essentially exactly, and a narrower one cannot")
print("— it plateaus at a nonzero KL that more training does not move.")
print("\nThat is eq. 66.6 as a capability rather than an inequality. The")
print("rank bound is exactly the set of conditional distributions a")
print("d-dimensional model is able to produce.")

# --- and what a mixture of softmaxes does -----------------------------------
print("\n" + "=" * 72)
print("breaking the bound: a mixture of softmaxes")
print("=" * 72)
print("A weighted SUM of several softmax outputs. Its logarithm is not")
print("constrained to rank d, because log of a sum is not a sum of logs.\n")


def fit_mos(target, d, K, steps=4000, lr=0.05, seed=0):
    rs = np.random.default_rng(seed)
    N_, V_ = target.shape
    H = [rs.normal(0, 0.3, (N_, d)) for _ in range(K)]
    U = rs.normal(0, 0.3, (V_, d))
    W = rs.normal(0, 0.3, (N_, K))
    ps = H + [U, W]
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    for t in range(1, steps + 1):
        comps = [softmax(H[k] @ U.T, axis=-1) for k in range(K)]
        pi = softmax(W, axis=-1)
        P = sum(pi[:, k:k + 1] * comps[k] for k in range(K))
        dP = (P - target) / N_ / np.clip(P, 1e-9, None) * np.clip(P, 1e-9, None)
        grads = []
        for k in range(K):
            dZ = pi[:, k:k + 1] * comps[k] * (
                dP - (dP * comps[k]).sum(1, keepdims=True))
            grads.append(dZ @ U)
        gU = sum(
            (pi[:, k:k + 1] * comps[k] * (
                dP - (dP * comps[k]).sum(1, keepdims=True))).T @ H[k]
            for k in range(K))
        gpi = np.stack([(dP * comps[k]).sum(1) for k in range(K)], 1)
        gW = pi * (gpi - (gpi * pi).sum(1, keepdims=True))
        grads += [gU, gW]
        for i, (p, g) in enumerate(zip(ps, grads)):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            p -= lr * (m[i] / (1 - 0.9 ** t)) / (
                np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    comps = [softmax(H[k] @ U.T, axis=-1) for k in range(K)]
    pi = softmax(W, axis=-1)
    P = sum(pi[:, k:k + 1] * comps[k] for k in range(K))
    return float((target * np.log(np.clip(target / np.clip(P, 1e-12, None),
                                          1e-12, None))).sum(1).mean())


tgt = make_target(N, V, 16, seed=16)
print(f"target rank 16\n")
print(f"{'model':<28} {'effective params':>18} {'KL to target':>14}")
for d in (4, 8):
    print(f"{f'single softmax, d={d}':<28} {N * d + V * d:>18,} "
          f"{fit(tgt, d, seed=5):>14.5f}")
for d, K in ((4, 4), (4, 8)):
    print(f"{f'mixture of {K}, d={d}':<28} {K * N * d + V * d + N * K:>18,} "
          f"{fit_mos(tgt, d, K, seed=5):>14.5f}")
print(f"{'single softmax, d=32':<28} {N * 32 + V * 32:>18,} "
      f"{fit(tgt, 32, seed=5):>14.5f}")

print("\nThe mixture rows do beat the single softmax at the same width, and")
print("they improve as K grows — so eq. 66.6's bound is genuinely escaped.")
print("The reason it CAN be is that the log of a sum is not a sum of logs,")
print("so the mixture's log-probability matrix is not constrained to the")
print("rank of any component.")
print("\nBut read the last row. A single softmax at d = 32 — the same")
print("parameter count as the 8-component mixture at d = 4 — fits the target")
print("two orders of magnitude better. The mixture escapes the bound and")
print("does not escape it EFFICIENTLY.")
print("\nThat comparison is the whole reason the field's response to the")
print("softmax bottleneck was neither of these. It made d larger. At")
print("d = 4096 the bound is far above anything language plausibly needs,")
print("and the problem stopped being interesting for the same reason many")
print("small-model problems did.")

```

## 9. Practical Example

```python {tier=A name=tying-and-the-logit-lens}
"""Weight tying measured at two scales, and the logit lens."""
import numpy as np

rng = np.random.default_rng(5)


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


V, T = 40, 12


def make_lm_task(n, seed):
    """A small language-modelling-like task: predict the next token, where
    the next token depends on a learnable function of the last two."""
    rs = np.random.default_rng(seed)
    rule = np.random.default_rng(99).integers(0, V, (V, V))
    X = np.zeros((n, T), dtype=int)
    X[:, 0] = rs.integers(0, V, n)
    X[:, 1] = rs.integers(0, V, n)
    for t in range(2, T):
        nxt = rule[X[:, t - 2], X[:, t - 1]]
        flip = rs.random(n) < 0.15                    # 15% noise
        X[:, t] = np.where(flip, rs.integers(0, V, n), nxt)
    return X[:, :-1], X[:, 1:]


class TinyLM:
    """Embedding -> attention -> FFN -> unembedding, optionally tied."""

    def __init__(self, d=32, tied=False, seed=0, out_scale=1.0):
        rs = np.random.default_rng(seed)
        self.E = rs.normal(0, 0.02, (V, d))
        self.P = rs.normal(0, 0.02, (T - 1, d))
        s = 1 / np.sqrt(d)
        self.Wq = rs.normal(0, s, (d, d))
        self.Wk = rs.normal(0, s, (d, d))
        self.Wv = rs.normal(0, s, (d, d))
        self.W1 = rs.normal(0, s, (d, 4 * d))
        self.W2 = rs.normal(0, np.sqrt(1 / (4 * d)), (4 * d, d))
        self.tied, self.d, self.out_scale = tied, d, out_scale
        if not tied:
            self.U = rs.normal(0, 0.02, (V, d))
        self.bo = np.zeros(V)

    def params(self):
        base = [self.E, self.P, self.Wq, self.Wk, self.Wv, self.W1, self.W2,
                self.bo]
        return base if self.tied else base + [self.U]

    def n_params(self):
        return sum(p.size for p in self.params())

    def unemb(self):
        return self.E if self.tied else self.U

    def forward(self, X, keep_layers=False):
        n, Tn = X.shape
        H0 = self.E[X] + self.P[None, :Tn, :]
        Q, K, Vv = H0 @ self.Wq, H0 @ self.Wk, H0 @ self.Wv
        S = Q @ K.transpose(0, 2, 1) / np.sqrt(self.d)
        mask = np.tril(np.ones((Tn, Tn), dtype=bool))
        S = np.where(mask, S, -np.inf)
        self.A = softmax(S)
        H1 = H0 + self.A @ Vv                          # residual
        Z = H1 @ self.W1
        Hh = np.maximum(0.0, Z)
        H2 = H1 + Hh @ self.W2                         # residual
        self.H0, self.H1, self.H2, self.Z, self.Hh = H0, H1, H2, Z, Hh
        self.X, self.Vv, self.Q, self.K = X, Vv, Q, K
        if keep_layers:
            self.layers = {"embed": H0, "after attn": H1, "after ffn": H2}
        return H2 @ self.unemb().T * self.out_scale + self.bo

    def grads(self, X, Y):
        n, Tn = X.shape
        logits = self.forward(X)
        P = softmax(logits)
        nt = n * Tn
        loss = float(-np.log(np.clip(
            np.take_along_axis(P, Y[..., None], -1), 1e-12, None)).sum() / nt)
        dl = P.copy()
        np.put_along_axis(dl, Y[..., None],
                          np.take_along_axis(dl, Y[..., None], -1) - 1.0, -1)
        dl /= nt
        U = self.unemb()
        gU = np.einsum('ntv,ntd->vd', dl, self.H2) * self.out_scale
        gbo = dl.sum(axis=(0, 1))
        dH2 = (dl @ U) * self.out_scale
        gW2 = np.einsum('nth,ntd->hd', self.Hh, dH2)
        dHh = dH2 @ self.W2.T
        dZ = dHh * (self.Z > 0)
        gW1 = np.einsum('ntd,nth->dh', self.H1, dZ)
        dH1 = dH2 + dZ @ self.W1.T
        dctx = dH1
        dA = dctx @ self.Vv.transpose(0, 2, 1)
        dV = self.A.transpose(0, 2, 1) @ dctx
        dS = self.A * (dA - (dA * self.A).sum(-1, keepdims=True))
        dS /= np.sqrt(self.d)
        dQ, dK = dS @ self.K, dS.transpose(0, 2, 1) @ self.Q
        H0f = self.H0.reshape(-1, self.d)
        gWq = H0f.T @ dQ.reshape(-1, self.d)
        gWk = H0f.T @ dK.reshape(-1, self.d)
        gWv = H0f.T @ dV.reshape(-1, self.d)
        dH0 = dH1 + dQ @ self.Wq.T + dK @ self.Wk.T + dV @ self.Wv.T
        gP = dH0.sum(axis=0)
        gE = np.zeros_like(self.E)
        np.add.at(gE, X.reshape(-1), dH0.reshape(-1, self.d))
        if self.tied:
            gE = gE + gU                               # BOTH paths
            return loss, [gE, gP, gWq, gWk, gWv, gW1, gW2, gbo]
        return loss, [gE, gP, gWq, gWk, gWv, gW1, gW2, gbo, gU]


def train(net, X, Y, steps=4000, lr=3e-3, batch=128, seed=0):
    ps = net.params()
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    rs = np.random.default_rng(seed + 3)
    for t in range(1, steps + 1):
        b = rs.integers(0, len(X), batch)
        _, gs = net.grads(X[b], Y[b])
        for i, (p, g) in enumerate(zip(ps, gs)):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            p -= lr * (m[i] / (1 - 0.9 ** t)) / (
                np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    return net


def evaluate(net, X, Y):
    P = softmax(net.forward(X))
    nll = float(-np.log(np.clip(
        np.take_along_axis(P, Y[..., None], -1), 1e-12, None)).mean())
    return nll


Xtr, Ytr = make_lm_task(8000, 1)
Xte, Yte = make_lm_task(4000, 2)

print("=" * 72)
print("weight tying at two scales (eq. 66.9)")
print("=" * 72)
print(f"vocabulary {V}. The embedding fraction changes with d, so the")
print("benefit of tying should change with it too.\n")
print(f"{'d':>5} {'embed fraction':>16} {'untied: params':>16} {'NLL':>8}  "
      f"{'tied: params':>14} {'NLL':>8}  {'tying helps?':>13}")
for d in (8, 16, 32, 64):
    a = train(TinyLM(d=d, tied=False, seed=7), Xtr, Ytr)
    b = train(TinyLM(d=d, tied=True, seed=7), Xtr, Ytr)
    na, nb = evaluate(a, Xte, Yte), evaluate(b, Xte, Yte)
    frac = 2 * V * d / a.n_params()
    print(f"{d:>5} {frac:>16.1%} {a.n_params():>16,} {na:>8.4f}  "
          f"{b.n_params():>14,} {nb:>8.4f}  "
          f"{('yes' if nb < na else 'no'):>13}")

print("\nEq. 66.9 predicts the direction: tying's benefit is the parameter")
print("saving, which is large when embeddings are most of the model and")
print("small when they are not. The 'embed fraction' column is that")
print("quantity and the last column is the outcome.")
print("\nThe cost of tying does not shrink with scale. One matrix must serve")
print("as the input map, where the residual stream is small, and as the")
print("output map, where the logits need range — section 5.3's scale")
print("coupling.")

# --- and the output scale fix -----------------------------------------------
print("\n" + "=" * 72)
print("the output scaling factor that tied models need (section 5.4 note)")
print("=" * 72)
print("Under tying, one matrix works at two scales. Scaling the logits")
print("decouples them partially.\n")
print(f"{'output scale':>14} {'tied NLL, d=32':>17}")
for sc in (0.5, 1.0, 2.0, 4.0, 8.0):
    net = train(TinyLM(d=32, tied=True, seed=7, out_scale=sc), Xtr, Ytr)
    print(f"{sc:>14.1f} {evaluate(net, Xte, Yte):>17.4f}")

print("\nIf the scale matters, the row spread is the scale-coupling problem")
print("measured. Vaswani et al. multiply the embedding by sqrt(d) for")
print("exactly this reason, and it is one of the details that gets dropped")
print("when people reimplement from a diagram.")

# --- section 6.5: the logit lens --------------------------------------------
print("\n" + "=" * 72)
print("the logit lens (eq. 66.11)")
print("=" * 72)
net = train(TinyLM(d=32, tied=False, seed=7), Xtr, Ytr)
net.forward(Xte[:2000], keep_layers=True)
U = net.unemb()

print("Apply the unembedding to the hidden state at each depth. The")
print("residual stream is a sum of block outputs, so every intermediate")
print("state lives in the same space and this is type-correct.\n")
print(f"{'read from':<16} {'NLL':>9} {'top-1 acc':>11} {'mean max prob':>15} "
      f"{'entropy':>9}")
for name, H in net.layers.items():
    Z = H @ U.T * net.out_scale + net.bo
    P = softmax(Z)
    tgt = Yte[:2000]
    nll = float(-np.log(np.clip(
        np.take_along_axis(P, tgt[..., None], -1), 1e-12, None)).mean())
    acc = float((P.argmax(-1) == tgt).mean())
    ent = float(-(P * np.log(P + 1e-12)).sum(-1).mean())
    print(f"{name:<16} {nll:>9.4f} {acc:>11.4f} {P.max(-1).mean():>15.4f} "
          f"{ent:>9.4f}")
print(f"\n(uniform entropy is ln {V} = {np.log(V):.3f})")

print("\nThe prediction sharpens with depth — that is the observation the")
print("logit lens is built on, and it is genuinely informative: it says the")
print("residual stream is being progressively shaped into something the")
print("unembedding can read.")
print("\nTwo cautions from section 6.5. The intermediate states have not")
print("been through the final normalisation, so their SCALE is wrong and")
print("the distributions are badly calibrated — read the ordering, not the")
print("probabilities. And the lens assumes the final unembedding is the")
print("right readout at every depth, which is an assumption; fitting a")
print("per-layer readout instead gives a different and usually sharper")
print("picture.")

# --- the embedding as a diagnostic ------------------------------------------
print("\n" + "=" * 72)
print("the embedding table is directly inspectable (section 7.5)")
print("=" * 72)
E = net.E
En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
sim = En @ En.T
np.fill_diagonal(sim, -np.inf)
print("nearest neighbour of each of the first 8 tokens, by cosine:\n")
print(f"{'token':>7} {'nearest':>9} {'cosine':>9}   "
      f"{'unembedding nearest':>21} {'cosine':>9}")
Un = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)
simU = Un @ Un.T
np.fill_diagonal(simU, -np.inf)
for t in range(8):
    j, ju = int(sim[t].argmax()), int(simU[t].argmax())
    print(f"{t:>7} {j:>9} {sim[t, j]:>9.4f}   {ju:>21} {simU[t, ju]:>9.4f}")

agree = float((sim.argmax(1) == simU.argmax(1)).mean())
print(f"\nfraction of tokens whose nearest neighbour is the SAME in both "
      f"tables: {agree:.4f}")
print(f"mean |cos(E_t, U_t)| for the same token: "
      f"{float(np.abs((En * Un).sum(1)).mean()):.4f}")

print("\nThat last number is the untied model's answer to the tying")
print("question, measured rather than argued: if the two matrices were")
print("doing the same job, each token's input and output vectors would")
print("align. How much they do is what the number says, and it is the")
print("cheapest evidence available about whether tying is throwing")
print("something away.")
```

## 10. Production Considerations

**Check whether your framework uses a sparse or dense embedding update.**
Measured: under a dense Adam, a row seen every thousandth step gets a far larger
update than one seen every step, from the same gradient. The sparse path removes
it.

**Budget the unembedding as a transformer block, or several.** Measured: above
$V = 12d$ it costs more than a block, and every realistic vocabulary is above
that.

**Tie weights below the crossover, not above it.** {{eq:tying-saving}} gives the
benefit; measured at 31% of parameters at GPT-2-small scale and 1.6% at 70B.

**If you tie, add an output scale.** One matrix serving two residual-stream
scales is the constraint, and a scalar is the cheapest partial fix.

**Detect and suppress untrained vocabulary rows.** They receive whatever logit a
random direction gives, occasionally a large one.

**Shard the vocabulary before it forces you to.** At $V = 128000$, $d = 8192$
the logit tensor for a modest batch is gigabytes.

## 11. Common Mistakes

**Implementing the embedding as a one-hot matmul.** Measured: billions of
wasted operations to perform a memory read.

**Applying the fan-in initialisation rule to embeddings.** There is no fan-in;
{{ch:dl-initialization}}'s derivation does not apply.

**Tying weights in a large model to save parameters.** Measured: it saves under
2% and imposes the same constraint it did at 30%.

**Reading logit-lens probabilities as calibrated.** They skip the final
normalisation.

**Forgetting that a tied model's embedding gets gradient from both ends.** The
implementation must add both contributions; missing one halves the gradient
silently.

**Growing the vocabulary without checking the compute.** Measured crossover.

## 12. Failure Modes

**Rare tokens over-updated.** Measured under dense Adam.

**Untrained rows producing spurious tokens.**

**Out-of-memory on the logit tensor** rather than on the model.

**A tied model that will not fit the output distribution** because the input
scale constrains the output range.

**The softmax bottleneck as a loss floor.** Measured: a model narrower than the
target's rank plateaus at a nonzero KL that more training does not move. The
symptom is a loss that stops improving with no other explanation.

**Nondeterminism from the scatter-add** when a token repeats in a batch
({{ch:mle-reproducibility}}).

## 13. Alternatives

**Adaptive embeddings** give frequent tokens more dimensions than rare ones,
cutting parameters at some quality cost for the tail.

**Factorised embeddings** write $\mat{E} = \mat{A}\mat{B}$ with an inner
dimension below $d$, which decouples the vocabulary size from the model width.
ALBERT's contribution.

**Mixture of softmaxes** breaks {{eq:logprob-rank}} at the cost of $K$ times the
output computation. Measured here; not adopted, because increasing $d$ was
simpler.

**Byte-level models** eliminate the vocabulary entirely, at the cost of far
longer sequences — which attention charges quadratically for
({{ch:tf-complexity}}).

**Sampled softmax** computes the loss over a subset of the vocabulary during
training. Standard in very large vocabularies; introduces a bias that has to be
corrected.

## 14. Evaluation

**Compute the embedding fraction before deciding to tie.** One line, and it is
the whole argument.

**Measure the alignment between $\mat{E}$ and $\mat{U}$ in an untied model.**
Measured here; it says directly how much tying would cost.

**Inspect the embedding's nearest neighbours.** The cheapest interpretability
artefact in the model.

**Check for untrained rows** by looking for embeddings still at their
initialisation norm.

**Test the softmax bottleneck by widening.** If the loss floor moves with $d$
and not with training, the bound is what you are hitting.

## 15. Advanced Concepts

**The residual stream as a communication channel.** The embedding writes into
it, every block reads and writes, and the unembedding reads from it
({{ch:tf-ffn-residual}}). On that view $\mat{E}$ and $\mat{U}$ are the channel's
endpoints and the logit lens is a tap on the wire.

**The tuned lens.** Fitting a per-layer affine readout instead of reusing
$\mat{U}$, which corrects the calibration problem and gives a different picture
of where predictions form. {{maturity:EMERGING}}

**Embedding geometry.** Trained embeddings occupy a narrow cone rather than
filling the space, and their norms correlate with token frequency. Both are
robust observations and neither is fully explained.
{{maturity:EMERGING}}

**Vocabulary transplant.** Replacing a model's tokeniser after training, by
initialising new rows from the old ones. Works better than it should, and it is
how models get adapted to new languages cheaply.

**Scaling laws for vocabulary.** Recent work argues the optimal $V$ grows with
model size and that most models are under-vocabularised. {{maturity:EMERGING}}

## 16. Connection to Previous Chapters

{{ch:dl-losses}}'s softmax and its logsumexp are what the unembedding feeds, and
the distributed loss computation of {{sec:7-internal-mechanics}} is that
chapter's numerical-stability argument applied across devices.

{{ch:dl-initialization}} is the chapter this one is an *exception* to, and
naming the exception precisely — no fan-in, because a lookup is not a sum —
is more useful than the rule.
{{ch:dl-optimizers}}'s Adam meets {{eq:embed-gradient}}'s sparsity here, with
the measured consequence for rare tokens.
{{ch:tf-positional}} supplies what gets added to the embedding, and
{{ch:tf-multi-head}}'s residual stream is what it writes into.

Forward: {{ch:tf-ffn-residual}} makes the residual stream explicit, which is
what justifies the logit lens.
{{ch:llm-anatomy}} assembles all of this into a working language model.
{{ch:emb-what-they-are}} takes the embedding table itself as the object of
study.

## 17. Exercises

**Beginner**

1. Why is an embedding a lookup rather than a matrix multiply?
2. What is weight tying and what does it save?
3. Why does the fan-in initialisation rule not apply to embeddings?
4. What is the softmax bottleneck?
5. Why is the unembedding expensive when the embedding is not?

**Intermediate**

6. Use {{eq:embed-fraction}} to find the $V$ at which embeddings are half of
   a model with $L=24$, $d=1024$.
7. Derive {{eq:logprob-rank}}.
8. Use {{eq:unembed-crossover}} to find where the unembedding equals one block
   for $d = 4096$.
9. Explain why a rare token gets a larger Adam update under a dense
   implementation.
10. Explain the scale-coupling problem in a tied model.

**Advanced**

11. Derive {{eq:tying-saving}} and evaluate it across three model scales.
12. Show that a mixture of $K$ softmaxes is not rank-limited, and bound the
    rank it can achieve.
13. Derive the distributed cross-entropy that avoids gathering the logits.
14. Explain why embedding norms correlate with token frequency.

**Implementation**

15. Implement tied and untied models and reproduce the crossover.
16. Reproduce the rank-fitting table and extend it to $V = 256$.
17. Implement a sparse Adam for embeddings and measure the rare-token effect.
18. Implement the logit lens with and without the final normalisation and
    compare.

**Reasoning**

19. Your model's loss plateaus and widening it helps while training longer does
    not. What are you hitting?
20. A model occasionally emits a token that never appears in the training data.
    Explain and fix.

## 18. Interview Questions

**"What is weight tying and when would you use it?"** — The parameter saving and
where it matters. Give {{eq:tying-saving}} and a number at two scales.

**"Why is the embedding gradient sparse?"** — Only rows in the batch appear.
Add the Adam consequence for the strong answer.

**"What is the softmax bottleneck?"** — Two lines of rank argument. Say that it
is a limit on the architecture, not on the training.

**"How expensive is the output layer?"** — More than a transformer block above
$V = 12d$. Derive it.

**"How would you initialise an embedding?"** — Small fixed standard deviation,
and say why the fan-in rule does not apply.

**"What is the logit lens and what is wrong with it?"** — Apply $\mat{U}$ at
intermediate depths; the states skip the final normalisation and the readout is
assumed rather than fitted.

## 19. Research Questions

**What is the optimal vocabulary size?** Recent scaling-law work suggests most
models are under-vocabularised, and the trade against sequence length is not
fully characterised. {{maturity:EMERGING}}

**Why do embeddings occupy a narrow cone?** Robustly observed, several proposed
explanations, none conclusive. {{maturity:EMERGING}}

**Does the softmax bottleneck bind in practice at modern widths?** At
$d = 4096$ the bound is far above what language plausibly needs, and nobody has
shown where it starts to matter. {{maturity:EMERGING}}

**Where in a network do predictions actually form?** The logit lens and the
tuned lens disagree, which means the question is partly about the readout rather
than about the model. {{maturity:RESEARCH FRONTIER}}

## 20. Chapter Summary

An embedding is an indexed read into a $V \times d$ table, and the one-hot
formulation is a notational device — measured, the matmul does billions of
operations to produce what a gather produces with none. Its gradient is
correspondingly sparse: even a 65,536-token batch leaves most of a 50,000-row
table untouched.

That sparsity meets Adam badly. Measured, a row appearing every thousandth step
receives a far larger update than one appearing every step *from the identical
gradient*, because $\vec{v}$ decays toward zero between appearances and Adam
divides by its square root. **Rare tokens take systematically bigger steps than
common ones**, which is the opposite of what anyone would choose — and a sparse
optimiser implementation removes it as a side effect of an efficiency
optimisation.

The output side has no shortcut. The unembedding is a full $d \times V$ matmul
per position, and {{eq:unembed-crossover}} puts the break-even against a
transformer block at $V = 12d$. Every realistic vocabulary is above it; measured
at GPT-2-small scale the output projection is worth about five and a half of the
model's twelve blocks.

Weight tying's benefit is arithmetic. Measured across four model scales, the
embedding fraction falls from about a third to about a per cent, and
{{eq:tying-saving}}'s saving falls with it — from 31% of all parameters to 1.6%.
The *constraint* does not shrink: one matrix must serve as the input map, where
the residual stream is small, and as the output map, where the logits need
range. That asymmetry, not taste, is why small models tie and large ones mostly
do not, and it is why tied models want an output scale factor.

The softmax bottleneck is a hard limit with a two-line derivation. The logit
matrix factors through $d$ dimensions so its rank is at most $d$, and the
log-probability matrix's is at most $d+1$ — verified exactly. Measured as a
capability, a model narrower than a target distribution's rank plateaus at a
nonzero KL that more training does not move. It is a bound on the architecture,
and the field's response was not a clever fix but simply making $d$ large enough
that the bound stopped binding.

Finally, both tables are directly inspectable and rarely inspected. Nearest
neighbours in the embedding show what a model considers similar before any
context, and the measured alignment between the embedding and unembedding rows
for the same token says, empirically, how much of the same job they are doing —
which is the tying question answered by measurement rather than by argument.

## 21. Further Reading

{{cite:press2017tying}} is the paper for weight tying and it is worth reading
for its era: it was written when models were small enough that a 30% parameter
saving was decisive, and reading it with {{eq:tying-saving}} in hand shows
exactly why its conclusion did not survive the scaling that followed.

{{cite:vaswani2017}} section 3.4 is four sentences on embeddings, and one of
them is the $\sqrt{d}$ scaling. It is presented without justification, which is
worth noticing — {{sec:5-formal-explanation}}'s explanation is a reconstruction,
and a fairly convincing one, but the paper does not make it.

**On the softmax bottleneck**, the original argument is short and the derivation
in {{sec:6-mathematical-foundation}} is essentially all of it. What is worth
reading around it is the response: the field's answer was to increase $d$ rather
than to adopt any of the proposed fixes, which is a recurring pattern worth
recognising.

{{cite:zhang2019rmsnorm}} is relevant here for an unobvious reason — the final
normalisation before the unembedding is what makes the logit lens's calibration
problem fixable, and knowing exactly what that normalisation does is what lets
you correct for it.

**Where to go next:** {{ch:tf-ffn-residual}} makes the residual stream explicit
— the channel this chapter's two matrices write into and read from — and shows
that the feed-forward block, not attention, holds most of a transformer's
parameters.

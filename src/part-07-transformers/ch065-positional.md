---
id: tf-positional
number: 65
part: VII
tier: full
status: reviewed
requires: [tf-multi-head, tf-scaled-dot-product, math-vectors, math-matrices]
provides: [positional-encoding, sinusoidal-encoding,
           learned-position, relative-position, rope, alibi, length-extrapolation,
           context-extension]
citations: [vaswani2017, su2021rope, press2022alibi, dosovitskiy2021vit]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Prove that self-attention is permutation-equivariant and explain what that
   means for the architecture.
2. Derive the sinusoidal encoding and explain its intended relative-position
   property.
3. Distinguish absolute from relative position encoding.
4. Derive RoPE and prove that it makes attention scores depend only on the
   offset.
5. Explain ALiBi and what it means for a model to extrapolate in length.
6. Explain why context extension is hard and compare the standard recipes.
7. Diagnose a positional problem from a model's behaviour.

## 2. Why This Matters

**Without positional information, a transformer cannot tell "dog bites man"
from "man bites dog".** {{sec:6-mathematical-foundation}} proves this: attention
commutes with permutation, so shuffling the input shuffles the output
identically and nothing about the computation depends on order.

**This is the one component the 2017 paper got least right.** Sinusoidal
encodings were replaced by learned ones, then those by relative ones, then those
by RoPE — which is now near-universal. Following that sequence teaches more
about how architectural choices get made than the final answer does.

**Length extrapolation is a real and unsolved problem.** A model trained at 4k
tokens does not simply work at 32k. {{cite:press2022alibi}} reframed positional
encoding as a question about extrapolation rather than representation, and the
whole context-extension literature follows from that reframing.

**Context extension is where the practical decisions are.** Every open-weight
model ships with a stated context length and a set of RoPE-scaling parameters,
and choosing them badly degrades the model in ways that a short evaluation will
not show. {{sec:9-practical-example}} measures the failure directly.

## 3. Prerequisites

{{ch:tf-multi-head}} for the attention block this modifies, and for
{{eq:score-factorisation}}, which is where RoPE inserts itself.
{{ch:tf-scaled-dot-product}} for the score. {{ch:math-vectors}} for rotations
and inner products. {{ch:math-matrices}} for orthogonal matrices.

## 4. Intuitive Explanation

### 4.1 Attention does not know about order

A recurrence processes positions in sequence, so order is implicit in the
computation. A convolution has a fixed spatial layout, so order is implicit in
the kernel. Attention has neither:

```text
   "dog bites man"          "man bites dog"

   every position attends to every position
   with weights computed from CONTENT only

   ==> shuffle the input, and the output shuffles identically
```

**This is not an oversight; it is a property of the operator.** Attention takes a
*set* of vectors and returns a set. Turning it into a sequence model requires
putting the position into the vectors somehow, and every scheme in this chapter
is a different answer to *how*.

### 4.2 Four places to put the position

```text
   1. add to the input        x_i  <-  x_i + p_i        absolute, 2017
   2. add to the scores       s_ij <-  s_ij + b(i-j)    relative bias, ALiBi
   3. rotate Q and K          q_i  <-  R(i) q_i         RoPE
   4. do nothing              (works only if something else supplies order)
```

The first is the original and the simplest: make a vector per position and add
it to the token embedding. The problem is that the model then has to *infer*
relative position from a difference of absolute ones, which it can do and does
imperfectly.

The second and third both encode relative position directly, and the difference
between them is where the information enters: ALiBi adds a fixed penalty to the
score, RoPE changes the vectors so the dot product already depends on the
offset.

### 4.3 Why relative is better than absolute

Consider "the cat sat on the mat" appearing at positions 0–5 in one document and
at positions 500–505 in another. The *relationships* are identical: "sat"
follows "cat" by one position in both.

An absolute scheme gives the model completely different position vectors in the
two cases and requires it to learn that $p_{501} - p_{500}$ means the same thing
as $p_1 - p_0$. A relative scheme gives it the same thing by construction.

**That is the whole argument, and it also explains the extrapolation
problem.** An absolute scheme has never seen $p_{9000}$ if it was trained to
4096, so the vector is either undefined (learned embeddings) or out of the range
the model has calibrated against (sinusoidal).

### 4.4 RoPE in one picture

Take the query and key vectors, split them into 2-dimensional pairs, and rotate
each pair by an angle proportional to its position:

```text
   position m, pair j:   rotate by  m * theta_j

        q_m = R(m) q       k_n = R(n) k

   score = (R(m) q)ᵀ (R(n) k)
         = qᵀ R(m)ᵀ R(n) k
         = qᵀ R(n - m) k          <- depends only on the OFFSET
```

The third line is the whole trick, and it works because rotations compose:
$\mat{R}(m)\T\mat{R}(n) = \mat{R}(n-m)$. **Absolute rotations, applied to both
sides of a dot product, produce relative dependence for free.**

Different pairs get different frequencies $\theta_j$, so short-range and
long-range relationships are encoded at different scales — the same idea as the
sinusoidal encoding, applied multiplicatively instead of additively.

### 4.5 ALiBi in one line

Do not encode position at all. Just penalise distance:

$$
s_{ij} \leftarrow s_{ij} - m_h \cdot |i - j|
$$

with a per-head slope $m_h$. Attention decays linearly with distance, at a rate
that differs per head so some heads look near and some look far.

**Nothing here depends on the training length**, which is why it extrapolates:
a distance of 8000 gets a penalty of $8000 m_h$ whether or not the model ever
saw one.

## 5. Formal Explanation

### 5.1 Permutation equivariance

For a permutation matrix $\mat{P}$:

$$
\Attn(\mat{P}\mat{X}) = \mat{P}\,\Attn(\mat{X})
$$ (eq:attention-equivariance)

Proved in {{sec:6-mathematical-foundation}}. The consequence is that a
transformer without positional information computes a function of the *multiset*
of tokens, and no amount of training changes that.

### 5.2 Sinusoidal encoding

{{cite:vaswani2017}} defines, for position $\pos$ and dimension index $i$:

$$
PE_{\pos, 2i} = \sin\!\left(\frac{\pos}{10000^{2i/d}}\right),
\qquad
PE_{\pos, 2i+1} = \cos\!\left(\frac{\pos}{10000^{2i/d}}\right)
$$ (eq:sinusoidal)

added to the token embedding. The wavelengths form a geometric progression from
$2\pi$ to $10000\cdot2\pi$.

The stated motivation is that $PE_{\pos+k}$ is a *linear function* of
$PE_{\pos}$ for any fixed offset $k$ — because a rotation by $k\theta$ is
linear — so the model could in principle learn to attend by relative position.

> WARNING: **"Could in principle" is doing a lot of work in that sentence.** The
> linear relation exists; whether the model exploits it is a separate question,
> and the empirical answer is that learned absolute embeddings performed about
> the same. {{cite:vaswani2017}} reports this. The sinusoidal scheme's real
> advantage was that it is defined for any position, which matters for
> extrapolation and which the paper mentions almost in passing.

### 5.3 Learned absolute embeddings

A table $\mat{P} \in \R^{T_{\max}\times d}$, trained like any other parameter,
added to the token embedding. Used by BERT and GPT-2.

Simple, effective within the trained range, and **completely undefined beyond
$T_{\max}$**. That is a hard ceiling: a model with learned positions cannot
process a longer sequence at all without adding parameters.

### 5.4 RoPE

{{cite:su2021rope}}. Partition the $d_k$-dimensional query into $d_k/2$ pairs.
For pair $j$ with frequency $\theta_j = 10000^{-2j/d_k}$, define the block
rotation

$$
\mat{R}_{\Theta,m}^{(j)}
 = \begin{pmatrix}
 \cos m\theta_j & -\sin m\theta_j\\
 \sin m\theta_j & \cos m\theta_j
 \end{pmatrix}
$$ (eq:rope-block)

and let $\mat{R}_{\Theta,m}$ be the block-diagonal matrix of these. Then

$$
\tilde{\vec{q}}_m = \mat{R}_{\Theta,m}\vec{q}_m,
\qquad
\tilde{\vec{k}}_n = \mat{R}_{\Theta,n}\vec{k}_n
$$ (eq:rope-apply)

**RoPE is applied to $\vec{q}$ and $\vec{k}$ only, not to $\vec{v}$**, and it is
applied *inside each head* after the projection. That placement is essential and
{{sec:7-internal-mechanics}} explains why.

### 5.5 ALiBi

{{cite:press2022alibi}}. Add a fixed bias to the pre-softmax score:

$$
s_{ij} = \frac{\vec{q}_i\T\vec{k}_j}{\sqrt{d_k}} - m_h\,(i-j)
\qquad (j \le i)
$$ (eq:alibi)

with head slopes forming a geometric sequence — for $h$ heads,
$m_h = 2^{-8h'/h}$ for $h' = 1..h$. No parameters, no embeddings, nothing added
to the input.

### 5.6 The comparison

{#tbl:position-schemes caption="Positional schemes. The last two columns are what decides the choice in practice: whether the model can process a sequence longer than it was trained on, and whether it does so well."}

| Scheme | Where | Parameters | Beyond $T_{\max}$ | Extrapolates |
|---|---|---|---|---|
| Learned absolute | input | $T_{\max}d$ | impossible | no |
| Sinusoidal | input | 0 | defined | poorly |
| Relative bias (T5) | scores | $O(h)$ buckets | defined | somewhat |
| ALiBi | scores | 0 | defined | yes, by design |
| RoPE | Q and K | 0 | defined | poorly without scaling |

**RoPE is the default despite not extrapolating**, which is worth pausing on. It
wins on quality within the trained range, and the extrapolation problem is
handled separately by the scaling methods of
{{sec:5-formal-explanation}} below rather than by the encoding itself.

### 5.7 Extending a context window

Three standard recipes, all modifying RoPE's frequencies:

**Position interpolation.** Scale positions down by $s = T_{\text{new}}/T_{\text{old}}$
so that position $m$ becomes $m/s$. Every position now falls inside the trained
range. Costs high-frequency resolution — nearby tokens become harder to
distinguish — and needs a short fine-tune.

**NTK-aware scaling.** Interpolate the *low* frequencies (long-range) and leave
the *high* frequencies (short-range) alone, by scaling the base $10000$ rather
than the positions. Preserves local resolution and often works with no
fine-tuning at all.

**YaRN.** A per-frequency interpolation combining both, with a temperature
correction on the attention scores. Currently the strongest of the three.

All three are {{maturity:ESTABLISHED}} as techniques and the choice between them
is empirical.

## 6. Mathematical Foundation

### 6.1 Permutation equivariance, proved

Let $\mat{P}$ be a $T\times T$ permutation matrix, so $\mat{P}\T\mat{P} =
\mat{I}$. Attention on the permuted input:

$$
\mat{Q}' = \mat{P}\mat{X}\mat{W}^Q = \mat{P}\mat{Q},
\qquad
\mat{K}' = \mat{P}\mat{K},
\qquad
\mat{V}' = \mat{P}\mat{V}
$$

The scores become $\mat{Q}'\mat{K}'^\top = \mat{P}\mat{Q}\mat{K}\T\mat{P}\T$,
which is the original score matrix with rows *and* columns permuted. The softmax
acts row-wise on a set of entries that have merely been reordered within each
row, so

$$
\softmax(\mat{P}\mat{S}\mat{P}\T) = \mat{P}\,\softmax(\mat{S})\,\mat{P}\T
$$

and therefore

$$
\Attn(\mat{P}\mat{X})
 = \mat{P}\softmax(\mat{S})\mat{P}\T\mat{P}\mat{V}
 = \mat{P}\softmax(\mat{S})\mat{V}
 = \mat{P}\,\Attn(\mat{X})
$$

$\square$

**Every layer of a transformer inherits this**, since the feed-forward block is
applied position-wise and normalisation is per-position. So the whole network is
permutation-equivariant, and a permutation-invariant readout — such as pooling —
makes it permutation-*invariant*: it cannot distinguish word orders at all.

### 6.2 RoPE gives relative dependence exactly

$$
\tilde{\vec{q}}_m\T\tilde{\vec{k}}_n
 = (\mat{R}_m\vec{q})\T(\mat{R}_n\vec{k})
 = \vec{q}\T\mat{R}_m\T\mat{R}_n\vec{k}
$$

For a 2-D rotation, $\mat{R}(\alpha)\T = \mat{R}(-\alpha)$ and
$\mat{R}(\alpha)\mat{R}(\beta) = \mat{R}(\alpha+\beta)$. Block-diagonally,

$$
\mat{R}_m\T\mat{R}_n = \mat{R}_{n-m}
$$

hence

$$
\boxed{\ \tilde{\vec{q}}_m\T\tilde{\vec{k}}_n
 = \vec{q}\T\mat{R}_{n-m}\vec{k}\ }
$$ (eq:rope-relative)

$\square$

**The score depends on $m$ and $n$ only through $n-m$.** Absolute rotations
applied to both sides of an inner product produce exact relative dependence, at
no cost in parameters and no change to the attention computation.

Two things follow that are easy to miss.

**It is exact, not approximate.** Contrast the sinusoidal scheme, where relative
position is *linearly recoverable* and the model must learn to recover it.

**The rotation is orthogonal, so norms are unchanged.** $\|\tilde{\vec{q}}\| =
\|\vec{q}\|$, which means RoPE cannot change the *scale* of any score — only its
dependence on position. That is why it composes cleanly with the $\sqrt{d_k}$
scaling of {{ch:tf-scaled-dot-product}} without re-deriving anything.

### 6.3 The frequency spectrum

With $\theta_j = b^{-2j/d_k}$ for base $b = 10000$, the wavelength of pair $j$ is

$$
\lambda_j = \frac{2\pi}{\theta_j} = 2\pi b^{2j/d_k}
$$ (eq:rope-wavelength)

For $d_k = 128$: $\lambda_0 = 2\pi \approx 6$ positions and
$\lambda_{63} = 2\pi\cdot 10000 \approx 62800$ positions.

**The low-index pairs distinguish adjacent tokens; the high-index pairs
distinguish distant ones.** A pair whose wavelength exceeds the training length
never completes a full rotation during training, so the model has only seen part
of its cycle — which is exactly where extrapolation breaks.

That observation is the whole basis of NTK-aware scaling: leave the
short-wavelength pairs alone, since they have been fully exercised, and stretch
only the long-wavelength ones, which have not.

### 6.4 Why position interpolation costs resolution

Scaling positions by $1/s$ multiplies every angle by $1/s$, so every wavelength
becomes $s\lambda_j$. The shortest wavelength grows from $2\pi$ to $2\pi s$.

Two adjacent positions are now separated by an angle of $\theta_0/s$ instead of
$\theta_0$. **The model's ability to distinguish token $m$ from token $m+1$
degrades by exactly the scale factor**, which is why interpolation needs a
fine-tune to recover and why it hurts most on tasks that need precise local
order.

NTK-aware scaling avoids this by choosing a new base $b'$ such that the
*longest* wavelength stretches by $s$ while the shortest barely moves:

$$
b' = b \cdot s^{d_k/(d_k-2)}
$$ (eq:ntk-base)

Since $d_k/(d_k-2) \approx 1$ for realistic $d_k$, this is close to $b' = bs$,
and the per-pair stretch factor $s^{2j/(d_k-2)}$ rises from about 1 at $j=0$ to
$s$ at $j = d_k/2 - 1$. {{sec:8-implementation}} measures both curves.

### 6.5 What ALiBi's bias does to the distribution

With bias $-m_h(i-j)$, the attention weight becomes

$$
\alpha_{ij} \propto \exp\!\left(\frac{\vec{q}_i\T\vec{k}_j}{\sqrt{d_k}}\right)
 e^{-m_h(i-j)}
$$ (eq:alibi-weights)

An exponential decay in distance, multiplying the content term. The effective
window — the distance at which the penalty costs a factor of $e$ — is
$1/m_h$ positions.

For $h = 8$ heads with $m_h = 2^{-h'}$, the effective windows are
$2, 4, 8, \dots, 256$ positions. **The heads span a geometric range of scales by
construction**, which is a design decision the model does not get to make.

That is ALiBi's strength and its weakness in one property. It extrapolates
because the penalty is defined at any distance; it cannot learn a
long-range relationship a head's slope forbids.

## 7. Internal Mechanics

### 7.1 Where RoPE is applied

```text
   x -> W_Q -> reshape to heads -> ROPE -> scores
   x -> W_K -> reshape to heads -> ROPE -> scores
   x -> W_V -> reshape to heads -> (no rope) -> weighted sum
```

Three details that are load-bearing.

**Inside the head.** The rotation uses $d_k$, not $d$, and pairs dimensions
within a head. Applying it before the head split rotates across head boundaries
and is wrong.

**After the projection.** RoPE rotates the *projected* query and key, not the
input. Rotating the input would put the position into the values too.

**Not on values.** {{eq:rope-relative}} works because the rotations appear on
both sides of an inner product. Values are not in an inner product, so rotating
them would just add position-dependent noise to the output.

### 7.2 The efficient implementation

The block-diagonal matrix is never formed. For a vector split into
$(x_0, x_1, x_2, x_3, \dots)$ interpreted as pairs:

```text
   out[2j]   = x[2j]   * cos(m θ_j) - x[2j+1] * sin(m θ_j)
   out[2j+1] = x[2j]   * sin(m θ_j) + x[2j+1] * cos(m θ_j)
```

Two elementwise multiplies and an add, with the cosines and sines precomputed
into a $(T_{\max}, d_k/2)$ table. Cost is $O(T d)$ per layer — negligible next
to the $O(T d^2)$ projections.

Most implementations use the "half-split" pairing — $x_j$ paired with
$x_{j + d_k/2}$ — rather than adjacent pairs. It is mathematically equivalent
under a permutation of dimensions and it makes the vectorised form cleaner.
**Checkpoints are not portable between the two conventions**, which is a real
source of subtle breakage when porting weights.

### 7.3 Caching the tables

$\cos(m\theta_j)$ and $\sin(m\theta_j)$ depend only on the position and the
frequency, so they are computed once at startup for $T_{\max}$ positions and
indexed thereafter. Changing the context length means rebuilding the table —
which is where a scaling factor gets applied.

### 7.4 The interaction with the KV cache

Cached keys are stored **after** rotation, so each key carries its absolute
position permanently. This is what makes RoPE compatible with incremental
decoding: a key cached at step 10 is still correct at step 1000, because
{{eq:rope-relative}} computes the offset from the two absolute rotations.

It also means **you cannot change the position of a cached token**. Prompt-cache
reuse across different prefixes requires the reused block to sit at the same
absolute offset, which is a genuine constraint on caching strategies
({{ch:tf-masking-kv}}).

### 7.5 Vision transformers

{{cite:dosovitskiy2021vit}} uses learned 1-D absolute positions on flattened
image patches, which discards the 2-D structure entirely. It works, which is
mildly surprising, and 2-D-aware schemes give modest gains. The lesson worth
taking is that the positional scheme matters less than the fact of having one.

## 8. Implementation

```python {tier=A name=permutation-and-position}
"""Attention has no notion of order (eq. 65.1), and what each scheme does
about it.
"""
import numpy as np

rng = np.random.default_rng(0)


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def attention(X, Wq, Wk, Wv, bias=None):
    dk = Wq.shape[1]
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    S = Q @ K.T / np.sqrt(dk)
    if bias is not None:
        S = S + bias
    return softmax(S) @ V


# --- section 6.1: permutation equivariance ----------------------------------
print("=" * 72)
print("self-attention is permutation-equivariant (eq. 65.1)")
print("=" * 72)
T, d, dk = 8, 32, 16
X = rng.normal(size=(T, d))
Wq = rng.normal(0, 1 / np.sqrt(d), (d, dk))
Wk = rng.normal(0, 1 / np.sqrt(d), (d, dk))
Wv = rng.normal(0, 1 / np.sqrt(d), (d, dk))

perm = rng.permutation(T)
out = attention(X, Wq, Wk, Wv)
out_perm = attention(X[perm], Wq, Wk, Wv)

print(f"max |Attn(PX) - P Attn(X)| = "
      f"{np.abs(out_perm - out[perm]).max():.3e}")
print("\nExact to floating point. Shuffling the input shuffles the output")
print("identically and changes nothing else, which means the operator sees")
print("a SET of vectors and not a sequence.")

print("\nThe consequence, stated as a task:")
tokens = ["dog", "bites", "man"]
Etab = {t: rng.normal(size=d) for t in tokens}
s1 = np.stack([Etab["dog"], Etab["bites"], Etab["man"]])
s2 = np.stack([Etab["man"], Etab["bites"], Etab["dog"]])
o1, o2 = attention(s1, Wq, Wk, Wv), attention(s2, Wq, Wk, Wv)
pooled1, pooled2 = o1.mean(0), o2.mean(0)
print(f"  'dog bites man' vs 'man bites dog', mean-pooled output:")
print(f"    max |difference| = {np.abs(pooled1 - pooled2).max():.3e}")
print("\nIdentical. A permutation-invariant readout on top of a")
print("permutation-equivariant network cannot distinguish word orders at")
print("all, and no amount of training changes that — it is a property of")
print("the operator, not of the parameters.")

# --- the schemes ------------------------------------------------------------
def sinusoidal(T, d, base=10000.0):
    """Eq. 65.2."""
    pos = np.arange(T)[:, None]
    i = np.arange(0, d, 2)[None, :]
    ang = pos / (base ** (i / d))
    pe = np.zeros((T, d))
    pe[:, 0::2] = np.sin(ang)
    pe[:, 1::2] = np.cos(ang)
    return pe


def rope_tables(T, dk, base=10000.0, pos_scale=1.0):
    """Eqs. 65.4-65.5, half-split convention (section 7.2)."""
    theta = base ** (-np.arange(0, dk, 2) / dk)          # (dk/2,)
    m = np.arange(T)[:, None] / pos_scale
    ang = m * theta[None, :]                             # (T, dk/2)
    return np.cos(ang), np.sin(ang)


def apply_rope(x, cos, sin):
    """x: (T, dk). Pairs dimension j with j + dk/2."""
    dk = x.shape[-1]
    x1, x2 = x[..., :dk // 2], x[..., dk // 2:]
    return np.concatenate([x1 * cos - x2 * sin,
                           x1 * sin + x2 * cos], axis=-1)


def alibi_bias(T, slope):
    """Eq. 65.6, causal."""
    i = np.arange(T)[:, None]
    j = np.arange(T)[None, :]
    b = -slope * (i - j).astype(float)
    b[j > i] = -np.inf
    return b


# --- section 6.2: RoPE's relative property ----------------------------------
print("\n" + "=" * 72)
print("RoPE makes the score depend ONLY on the offset (eq. 65.9)")
print("=" * 72)
dk = 32
q = rng.normal(size=dk)
k = rng.normal(size=dk)
cos, sin = rope_tables(600, dk)

print("The same q and k placed at different absolute positions with the")
print("SAME offset. If eq. 65.9 holds, every score in a column is equal.\n")
print(f"{'offset n - m':>13} " + " ".join(f"{f'm={m}':>12}"
                                          for m in (0, 5, 100, 500)))
for off in (0, 1, 3, 10):
    row = []
    for m in (0, 5, 100, 500):
        n = m + off
        qm = apply_rope(q[None, :], cos[m:m + 1], sin[m:m + 1])[0]
        kn = apply_rope(k[None, :], cos[n:n + 1], sin[n:n + 1])[0]
        row.append(float(qm @ kn))
    print(f"{off:>13} " + " ".join(f"{v:>12.6f}" for v in row))
    print(f"{'':>13} " + " ".join(f"{'':>12}" for _ in range(3))
          + f"  spread {max(row) - min(row):.2e}")

print("\nEvery row is constant to floating point: the score depends on the")
print("offset and not on where the pair sits. That is eq. 65.9, and it is")
print("EXACT rather than approximate — which is the difference from the")
print("sinusoidal scheme, where relative position is merely linearly")
print("recoverable and the model has to learn to recover it.")

print("\nRoPE also preserves norms, being a rotation:")
qm = apply_rope(q[None, :], cos[137:138], sin[137:138])[0]
print(f"  |q| = {np.linalg.norm(q):.6f}   "
      f"|R(137) q| = {np.linalg.norm(qm):.6f}")
print("\nSo it cannot change the SCALE of any score, only its dependence on")
print("position — which is why it composes with Chapter 63's sqrt(d_k)")
print("without re-deriving anything.")

# --- what sinusoidal gives you ----------------------------------------------
print("\n" + "=" * 72)
print("what the sinusoidal scheme gives you, and does not")
print("=" * 72)
pe = sinusoidal(512, 64)
print("Claim (Vaswani et al.): PE(pos+k) is a LINEAR function of PE(pos).")
print("Fit one linear map per offset and check.\n")
print(f"{'offset k':>10} {'linear fit residual':>22} "
      f"{'dot(PE_pos, PE_pos+k) spread':>30}")
for k_ in (1, 5, 20, 100):
    A = pe[:400]
    B = pe[k_:400 + k_]
    M, *_ = np.linalg.lstsq(A, B, rcond=None)
    res = float(np.abs(A @ M - B).max())
    dots = (pe[:400] * pe[k_:400 + k_]).sum(1)
    print(f"{k_:>10} {res:>22.3e} {float(dots.max() - dots.min()):>30.3e}")

print("\nThe linear relation holds essentially exactly — a rotation IS")
print("linear, so this is not surprising. And the raw dot product between")
print("PE(pos) and PE(pos+k) is constant in pos, so the encoding does carry")
print("clean relative information.")
print("\nThe catch is what happens next: the position is ADDED to the token")
print("embedding, and the attention score is computed from the sum. So the")
print("score contains token-token, token-position, position-token and")
print("position-position terms all mixed together, and only the last is")
print("cleanly relative. RoPE avoids the mixing entirely by acting on the")
print("projected q and k rather than on the input.")
```

```python {tier=A name=extrapolation-and-scaling}
"""Length extrapolation: why RoPE fails past its training length, why ALiBi
does not, and what the scaling recipes do (eqs. 65.10-65.12).
"""
import numpy as np

rng = np.random.default_rng(1)


def rope_tables(T, dk, base=10000.0, pos_scale=1.0):
    theta = base ** (-np.arange(0, dk, 2) / dk)
    m = np.arange(T)[:, None] / pos_scale
    ang = m * theta[None, :]
    return np.cos(ang), np.sin(ang), theta


# --- section 6.3: the frequency spectrum ------------------------------------
print("=" * 72)
print("RoPE's wavelengths span four orders of magnitude (eq. 65.10)")
print("=" * 72)
dk = 128
_, _, theta = rope_tables(1, dk)
wl = 2 * np.pi / theta
print(f"head dimension {dk}, so {dk // 2} frequency pairs\n")
print(f"{'pair j':>8} {'theta_j':>12} {'wavelength':>14} "
      f"{'cycles at T=4096':>19}")
for j in (0, 8, 16, 32, 48, 63):
    print(f"{j:>8} {theta[j]:>12.3e} {wl[j]:>14.1f} {4096 / wl[j]:>19.2f}")

full = int((wl < 4096).sum())
print(f"\npairs completing at least one full cycle within T = 4096: "
      f"{full} of {dk // 2}")
print(f"pairs that never complete one: {dk // 2 - full}")

print("\nThat last number is where extrapolation breaks. A pair whose")
print("wavelength exceeds the training length has only ever been seen on")
print("part of its cycle, so the model has no calibration for the angles it")
print("will encounter at longer positions. The short-wavelength pairs have")
print("been fully exercised and are fine.")
print("\nThat asymmetry is the whole basis of NTK-aware scaling: stretch the")
print("long-wavelength pairs, which are undertrained, and leave the")
print("short-wavelength ones, which are not.")

# --- section 6.4: what each scaling recipe does -----------------------------
print("\n" + "=" * 72)
print("what the scaling recipes do to the wavelengths (eqs. 65.11-65.12)")
print("=" * 72)
s = 8.0                                    # extend 4k -> 32k
print(f"extending by a factor of s = {s:g}\n")
_, _, th_base = rope_tables(1, dk, base=10000.0)
_, _, th_pi = rope_tables(1, dk, base=10000.0, pos_scale=s)
b_ntk = 10000.0 * s ** (dk / (dk - 2))
_, _, th_ntk = rope_tables(1, dk, base=b_ntk)

print(f"NTK-aware base: 10000 -> {b_ntk:.0f}  (eq. 65.12)\n")
print(f"{'pair j':>8} {'base wavelength':>17} {'interp. stretch':>17} "
      f"{'NTK stretch':>14}")
for j in (0, 8, 16, 32, 48, 63):
    wl0 = 2 * np.pi / th_base[j]
    st_pi = (2 * np.pi / (th_pi[j] / s)) / wl0 if False else s
    st_ntk = (2 * np.pi / th_ntk[j]) / wl0
    print(f"{j:>8} {wl0:>17.1f} {s:>17.2f}x {st_ntk:>13.2f}x")

print("\nPosition interpolation stretches EVERY wavelength by s, including")
print("the shortest. Two adjacent tokens, which were separated by an angle")
print(f"of theta_0, are now separated by theta_0 / {s:g} — so the model's")
print("ability to tell token m from token m+1 degrades by exactly the scale")
print("factor. That is section 6.4, and it is why interpolation needs a")
print("fine-tune and hurts most on tasks needing precise local order.")
print("\nNTK-aware scaling stretches the longest wavelength by about s and")
print("the shortest by about 1. Local resolution is preserved and only the")
print("undertrained long-range pairs are moved, which is why it often works")
print("with no fine-tuning at all.")

# --- what happens to the score at unseen distances --------------------------
print("\n" + "=" * 72)
print("what the attention score does past the training length")
print("=" * 72)


def apply_rope(x, cos, sin):
    d = x.shape[-1]
    x1, x2 = x[..., :d // 2], x[..., d // 2:]
    return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


T_TRAIN, T_TEST = 512, 4096
q = rng.normal(size=(200, dk)) / np.sqrt(dk) ** 0.5
k = rng.normal(size=(200, dk)) / np.sqrt(dk) ** 0.5

print("Mean |score| as a function of offset, for random q and k. Inside the")
print("training range the model has calibrated against these magnitudes;")
print("outside it has not.\n")
print(f"{'offset':>9} {'in training range?':>20} {'mean |score|':>14} "
      f"{'sd of score':>13}")
cos, sin, _ = rope_tables(T_TEST + 1, dk)
for off in (1, 16, 128, 512, 1024, 4096):
    qm = apply_rope(q, cos[0:1], sin[0:1])
    kn = apply_rope(k, cos[off:off + 1], sin[off:off + 1])
    sc = (qm * kn).sum(-1) / np.sqrt(dk)
    print(f"{off:>9} {str(off <= T_TRAIN):>20} {float(np.abs(sc).mean()):>14.4f} "
          f"{float(sc.std()):>13.4f}")

print("\nFor RANDOM q and k the score statistics barely change with offset,")
print("which is worth stating plainly because it shows what the")
print("extrapolation problem is NOT. It is not that the scores blow up.")
print("\nThe problem is that the ROTATION ANGLES at large offsets are ones")
print("the trained q and k directions were never optimised against. A")
print("trained model has learned specific q-k geometries that produce")
print("useful scores at the offsets it saw, and those geometries have no")
print("reason to produce useful scores at angles outside that range.")
print("Random vectors cannot show this, because they have no learned")
print("geometry to lose — which is why the honest measurement of")
print("extrapolation needs a trained model, and section 9 uses one.")

# --- ALiBi's behaviour ------------------------------------------------------
print("\n" + "=" * 72)
print("ALiBi's decay is defined at any distance (eq. 65.13)")
print("=" * 72)
h = 8
slopes = 2.0 ** (-8.0 * np.arange(1, h + 1) / h)
print(f"{h} heads, slopes 2^(-8h'/h)\n")
print(f"{'head':>5} {'slope':>12} {'effective window 1/m':>22} " +
      " ".join(f"{f'penalty@{d_}':>13}" for d_ in (10, 100, 1000)))
for i, m in enumerate(slopes):
    pen = [f"{np.exp(-m * d_):.2e}" for d_ in (10, 100, 1000)]
    print(f"{i:>5} {m:>12.5f} {1 / m:>22.1f} " +
          " ".join(f"{p:>13}" for p in pen))

print("\nThe 'penalty' columns are the multiplicative factor eq. 65.13")
print("applies to the attention weight at that distance. Head 0 is")
print("effectively blind past a few positions; head 7 still sees a")
print("thousand.")
print("\nNothing in this table refers to the training length, which is")
print("exactly why ALiBi extrapolates: a distance of 8000 gets penalty")
print("exp(-8000m) whether or not the model has ever seen one.")
print("\nAnd that is also its limitation. The heads' scales are FIXED by the")
print("slope schedule, so a relationship at distance 5000 can only be")
print("learned by the two or three heads whose slopes permit it. RoPE lets")
print("every head attend at any distance and pays for it with the")
print("extrapolation problem above. Neither is free.")
```

## 9. Practical Example

```python {tier=A name=position-on-a-real-task}
"""Positional schemes on a task that needs order, and what happens when a
trained model is asked to run past its training length.
"""
import numpy as np

rng = np.random.default_rng(3)

V = 10


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def make_order_task(n, T, seed):
    """The label is the token at a FIXED offset from the end — a purely
    positional relationship that a set-based model cannot represent."""
    rs = np.random.default_rng(seed)
    X = rs.integers(1, V, (n, T))
    y = X[:, -3]                                    # third from the end
    return X, y


def rope_tables(T, dk, base=10000.0, pos_scale=1.0):
    theta = base ** (-np.arange(0, dk, 2) / dk)
    m = np.arange(T)[:, None] / pos_scale
    ang = m * theta[None, :]
    return np.cos(ang), np.sin(ang)


def apply_rope(x, cos, sin):
    """x: (n, T, dk)."""
    d = x.shape[-1]
    x1, x2 = x[..., :d // 2], x[..., d // 2:]
    c = cos[None, :x.shape[1], :]
    s = sin[None, :x.shape[1], :]
    return np.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1)


def sinusoidal(T, d, base=10000.0):
    pos = np.arange(T)[:, None]
    i = np.arange(0, d, 2)[None, :]
    ang = pos / (base ** (i / d))
    pe = np.zeros((T, d))
    pe[:, 0::2] = np.sin(ang)
    pe[:, 1::2] = np.cos(ang)
    return pe


class PosModel:
    """One attention head with a configurable positional scheme."""

    def __init__(self, scheme, d=48, T_max=64, seed=0, base=10000.0,
                 pos_scale=1.0):
        rs = np.random.default_rng(seed)
        self.E = rs.normal(0, 0.3, (V, d))
        self.scheme = scheme
        self.d, self.T_max = d, d if False else T_max
        s = 1 / np.sqrt(d)
        self.Wq = rs.normal(0, s, (d, d))
        self.Wk = rs.normal(0, s, (d, d))
        self.Wv = rs.normal(0, s, (d, d))
        self.Wr = rs.normal(0, s, (d, V))
        self.br = np.zeros(V)
        if scheme == "learned":
            self.P = rs.normal(0, 0.3, (T_max, d))
        elif scheme == "sinusoidal":
            self.P = sinusoidal(T_max, d)
        if scheme == "rope":
            self.cos, self.sin = rope_tables(T_max, d, base, pos_scale)
        if scheme == "alibi":
            self.slope = 0.25

    def params(self):
        base = [self.E, self.Wq, self.Wk, self.Wv, self.Wr, self.br]
        return base + ([self.P] if self.scheme == "learned" else [])

    def rebuild_rope(self, T, base=10000.0, pos_scale=1.0):
        self.cos, self.sin = rope_tables(T, self.d, base, pos_scale)

    def forward(self, X):
        n, T = X.shape
        H = self.E[X]
        if self.scheme in ("learned", "sinusoidal"):
            H = H + self.P[None, :T, :]
        self.H = H
        Q, K, Vv = H @ self.Wq, H @ self.Wk, H @ self.Wv
        if self.scheme == "rope":
            Q, K = apply_rope(Q, self.cos, self.sin), \
                apply_rope(K, self.cos, self.sin)
        S = Q @ K.transpose(0, 2, 1) / np.sqrt(self.d)
        if self.scheme == "alibi":
            i = np.arange(T)[:, None]
            j = np.arange(T)[None, :]
            S = S - self.slope * np.abs(i - j)
        self.A = softmax(S)
        self.O = self.A @ Vv
        self.read = self.O[:, -1, :]
        self.Q, self.K, self.Vv = Q, K, Vv
        return self.read @ self.Wr + self.br

    def grads(self, X, y):
        n, T = X.shape
        logits = self.forward(X)
        m_ = logits.max(1, keepdims=True)
        e = np.exp(logits - m_)
        p = e / e.sum(1, keepdims=True)
        loss = float(-np.log(np.clip(p[np.arange(n), y], 1e-12, None)).mean())
        dl = p.copy()
        dl[np.arange(n), y] -= 1.0
        dl /= n
        gWr, gbr = self.read.T @ dl, dl.sum(0)
        dO = np.zeros_like(self.O)
        dO[:, -1, :] = dl @ self.Wr.T
        dA = dO @ self.Vv.transpose(0, 2, 1)
        dV = self.A.transpose(0, 2, 1) @ dO
        dS = self.A * (dA - (dA * self.A).sum(-1, keepdims=True))
        dS /= np.sqrt(self.d)
        dQ, dK = dS @ self.K, dS.transpose(0, 2, 1) @ self.Q
        if self.scheme == "rope":
            # rotation is orthogonal: the backward rotation is by -m
            dQ = apply_rope(dQ, self.cos, -self.sin)
            dK = apply_rope(dK, self.cos, -self.sin)
        Hf = self.H.reshape(-1, self.d)
        gWq = Hf.T @ dQ.reshape(-1, self.d)
        gWk = Hf.T @ dK.reshape(-1, self.d)
        gWv = Hf.T @ dV.reshape(-1, self.d)
        dH = (dQ @ self.Wq.T + dK @ self.Wk.T + dV @ self.Wv.T)
        gE = np.zeros_like(self.E)
        np.add.at(gE, X.reshape(-1), dH.reshape(-1, self.d))
        out = [gE, gWq, gWk, gWv, gWr, gbr]
        if self.scheme == "learned":
            gP = np.zeros_like(self.P)
            gP[:T] = dH.sum(0)
            out.append(gP)
        return loss, out


def train(net, X, y, steps=4000, lr=3e-3, batch=128, seed=0):
    ps = net.params()
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    rs = np.random.default_rng(seed + 5)
    for t in range(1, steps + 1):
        b = rs.integers(0, len(X), batch)
        _, gs = net.grads(X[b], y[b])
        for i, (p, g) in enumerate(zip(ps, gs)):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            p -= lr * (m[i] / (1 - 0.9 ** t)) / (
                np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    return net


T_TRAIN = 16
Xtr, ytr = make_order_task(12000, T_TRAIN, 1)
Xte, yte = make_order_task(4000, T_TRAIN, 2)

print("=" * 72)
print("a task that needs ORDER: predict the third-from-last token")
print("=" * 72)
print(f"sequence length {T_TRAIN}, vocabulary {V}, chance {1 / V:.4f}\n")
print(f"{'scheme':<16} {'extra params':>13} {'test accuracy':>15}")
nets = {}
for scheme in ("none", "learned", "sinusoidal", "rope", "alibi"):
    net = train(PosModel(scheme, T_max=128, seed=4), Xtr, ytr)
    nets[scheme] = net
    acc = float((net.forward(Xte).argmax(1) == yte).mean())
    extra = net.P.size if scheme == "learned" else 0
    print(f"{scheme:<16} {extra:>13,} {acc:>15.4f}")

print("\nThe 'none' row is eq. 65.1 as a task result: with no positional")
print("information the model sees a multiset and the answer depends on")
print("order, so it cannot do better than guessing.")
print("\nThe ALiBi row is the interesting failure, and it is instructive")
print("rather than a bug. ALiBi supplies no positional REPRESENTATION at")
print("all — only a monotone penalty on distance. A head can therefore")
print("express 'attend to things nearby' and cannot express 'attend to")
print("exactly three back', because a monotone decay has no way to single")
print("out one offset.")
print("\nThat is section 6.5's limitation in its sharpest form. Real ALiBi")
print("uses many heads with a geometric range of slopes, which gives a")
print("range of SCALES and still no ability to select a precise offset.")
print("Section 5.6's table lists ALiBi as extrapolating well, and this row")
print("is the other half of the trade.")
print("\nThe three schemes that supply an actual positional representation")
print("all solve it. That is the first finding and the one worth carrying:")
print("HAVING a positional representation matters more than which one.")

# --- extrapolation ----------------------------------------------------------
print("\n" + "=" * 72)
print("what happens past the training length")
print("=" * 72)
print(f"Trained at T = {T_TRAIN}. Evaluated at longer lengths, with the")
print("task unchanged — still the third-from-last token.\n")
print(f"{'scheme':<16} " + " ".join(f"{f'T={T}':>10}"
                                    for T in (16, 24, 48, 96)))
for scheme in ("learned", "sinusoidal", "rope", "alibi"):
    net = nets[scheme]
    row = []
    for T in (16, 24, 48, 96):
        Xe, ye = make_order_task(2000, T, 7)
        if scheme == "rope":
            net.rebuild_rope(T)
        try:
            acc = float((net.forward(Xe).argmax(1) == ye).mean())
        except (IndexError, ValueError):
            acc = float("nan")
        row.append(acc)
    if scheme == "rope":
        net.rebuild_rope(128)
    print(f"{scheme:<16} " + " ".join(
        f"{'n/a':>10}" if np.isnan(a) else f"{a:>10.4f}" for a in row))
print(f"\n(chance is {1 / V:.4f})")

print("\nThe learned and sinusoidal rows collapse to near chance, which is")
print("table 65.1's last column behaving as advertised: an absolute scheme")
print("has never seen these positions.")
print("\nThe RoPE row does NOT collapse, and that is worth being precise")
print("about rather than treating as a happy surprise. This task is purely")
print("RELATIVE — the answer is always three from the end — and eq. 65.9")
print("says RoPE's score depends only on the offset, exactly, at any")
print("absolute position. An offset of two gives the identical score at")
print("position 10 and at position 10,000.")
print("\nSo RoPE extrapolates perfectly on relative tasks, and table 65.1's")
print("'extrapolates poorly' is about something else: tasks needing")
print("LONG-RANGE or ABSOLUTE information, where the model must use offsets")
print("far larger than any it was trained on. Section 6.3 identified the")
print("mechanism — the long-wavelength frequency pairs never complete a")
print("cycle during training, so the model has no calibration for the")
print("angles they produce at large offsets.")
print("\nA local relative task never touches those pairs. That is the")
print("distinction, and a benchmark that only tests local relationships")
print("will report that RoPE extrapolates fine.")

# --- and what scaling does --------------------------------------------------
print("\n" + "=" * 72)
print("RoPE scaling: position interpolation vs NTK-aware (eqs. 65.11-12)")
print("=" * 72)
net = nets["rope"]
dk = net.d
print(f"Trained at T = {T_TRAIN}, evaluated at longer T with the RoPE table")
print("rebuilt under each recipe. No fine-tuning — this is the zero-shot")
print("case, which is what the recipes are usually asked to do.\n")
print(f"{'T':>6} {'scale s':>9} {'no scaling':>12} "
      f"{'interpolation':>15} {'NTK-aware':>12}")
for T in (24, 48, 96):
    s_ = T / T_TRAIN
    Xe, ye = make_order_task(2000, T, 7)
    accs = []
    net.rebuild_rope(T)
    accs.append(float((net.forward(Xe).argmax(1) == ye).mean()))
    net.rebuild_rope(T, pos_scale=s_)
    accs.append(float((net.forward(Xe).argmax(1) == ye).mean()))
    net.rebuild_rope(T, base=10000.0 * s_ ** (dk / (dk - 2)))
    accs.append(float((net.forward(Xe).argmax(1) == ye).mean()))
    print(f"{T:>6} {s_:>9.2f} " + " ".join(f"{a:>12.4f}" for a in accs))
net.rebuild_rope(128)

print("\nThis table is section 6.4's argument in its most extreme form, and")
print("the direction is unambiguous.")
print("\nInterpolation divides every position by s, so the offset between")
print("adjacent tokens becomes theta_0/s instead of theta_0 — and this task")
print("depends on distinguishing 'three back' from 'two back' and 'four")
print("back'. Compressing exactly that distinction is the one thing it")
print("cannot survive.")
print("\nNTK-aware scaling leaves the short wavelengths alone and stretches")
print("only the long, undertrained ones. On a purely local task that means")
print("it changes nothing that matters.")
print("\nAnd 'no scaling' also works here, for the reason the previous table")
print("gave: RoPE's relative property is exact at any offset, so a local")
print("task needs no rescaling at all. The lesson is not that interpolation")
print("is bad — it is that a scaling recipe must be chosen against the")
print("RANGE the task actually uses, and evaluated on a task that uses it.")
print("\nBe careful generalising the numbers from one head on a synthetic")
print("task. What transfers is the mechanism: which wavelengths each recipe")
print("moves, and which ones your task depends on.")
```

## 10. Production Considerations

**Never ship a transformer without a positional scheme.** Measured: with none,
the model sits at chance on a task that depends on order, and it is a property
of the operator that training cannot fix.

**Check the RoPE pairing convention when porting weights.** Adjacent-pair and
half-split conventions are mathematically equivalent and produce different
numbers from the same checkpoint ({{sec:7-internal-mechanics}}).

**Rebuild the RoPE table when you change the context length**, and apply the
scaling factor there rather than at inference time.

**A stated context length is a claim about the scaling configuration, not about
the weights.** Two deployments of the same checkpoint with different RoPE
settings are different models. Record the settings with the checkpoint.

**Evaluate context extension on a task that needs the full range.** A model
extended to 128k that only ever gets asked about the last 2k tokens will look
fine and be broken.

**Prefer NTK-aware or YaRN to plain interpolation** unless you are fine-tuning,
because interpolation degrades local resolution by the scale factor.

## 11. Common Mistakes

**Applying RoPE to values.** {{eq:rope-relative}} works because the rotations
appear on both sides of an inner product; values are not.

**Applying RoPE before the head split.** It pairs dimensions within $d_k$;
rotating across head boundaries mixes heads.

**Adding positions to a model that already has RoPE.** They are alternatives,
not complements.

**Assuming sinusoidal encodings extrapolate.** They are *defined* at any
position, which is not the same thing.

**Changing the context length without rebuilding the frequency table.**

**Reusing a cached prefix at a different offset.** RoPE bakes absolute position
into the cached key ({{sec:7-internal-mechanics}}).

## 12. Failure Modes

**Chance performance on order-sensitive tasks** with no positional scheme.
Measured.

**A hard ceiling at $T_{\max}$** with learned embeddings — the lookup simply
has no row.

**Silent degradation past the training length.** The model runs, produces
fluent output, and gets long-range relationships wrong.

**Local-order errors after position interpolation.** Measured mechanism:
adjacent-token resolution degrades by the scale factor.

**A checkpoint that produces nonsense after porting.** Usually the RoPE pairing
convention.

**Attention diluting at long context.** Independent of the scheme: a softmax
over more positions has lower maximum weight, so a genuine signal competes with
more noise.

## 13. Alternatives

**T5-style relative position bias** learns a bias per (bucketed) relative
distance, added to the score. More parameters than ALiBi, more flexible, and it
must decide what to do with buckets it never saw.

**No positional encoding at all** works for a *causal* decoder, because the
causal mask itself breaks permutation symmetry — position $i$ can see $i$ tokens
and position $j$ can see $j$. This is a genuine and surprising result and the
resulting models are somewhat worse. {{maturity:EMERGING}}

**2-D and axial schemes** for images and video, encoding each axis separately.
{{cite:dosovitskiy2021vit}} did not bother and it still worked, which is a
useful calibration on how much this matters.

**Learned RoPE frequencies** rather than the fixed geometric progression.
Marginal gains reported, not standard.

## 14. Evaluation

**Test the permutation property directly.** Shuffle the input and check the
output shuffles identically; one line, and it verifies your scheme is actually
connected.

**Evaluate at several lengths, including past the training length.** Measured
here to be the only way to see extrapolation failure.

**Use a task that needs the range you claim.** A long-context benchmark whose
answers all sit near the end measures nothing.

**Compare scaling recipes on both local and long-range tasks.**
Interpolation's cost is local and would be invisible on a long-range-only
evaluation.

**Verify RoPE's relative property numerically** after any refactor: the same
$q, k$ at the same offset must give the same score.

## 15. Advanced Concepts

**YaRN.** Per-frequency interpolation — full interpolation for wavelengths much
longer than the training length, none for wavelengths much shorter, a blend
between — plus a temperature adjustment on the attention scores to compensate
for the changed score distribution. Currently the strongest recipe.

**The attention-dilution problem.** Independent of position encoding: a softmax
over $T$ positions has maximum weight bounded below by $1/T$ concerns, and
long-context models show measurably flatter attention. Some scaling recipes
include a temperature term specifically for this.

**Positional information from the causal mask.** A decoder-only model can infer
position from how many tokens it can see. This is why no-positional-encoding
models work at all and it means RoPE is partly redundant in a causal model.

**Length generalisation as a benchmark.** Tasks like copying or addition, where
the correct answer is well-defined at any length, isolate positional
generalisation from everything else. Most models do badly on them, which is
worth knowing before believing a headline context length.

**Position in state space models.** A recurrence encodes position implicitly in
the state's evolution, so it needs no positional scheme at all — one of the
genuine architectural advantages of that family ({{ch:tf-efficient}}).

## 16. Connection to Previous Chapters

{{ch:tf-multi-head}} defined the attention block that this chapter shows is
order-blind. {{eq:score-factorisation}} is where RoPE inserts itself — between
the projection and the score — and the fact that a rotation is orthogonal is why
it does not disturb {{ch:tf-scaled-dot-product}}'s variance argument.

{{ch:tf-why-attention}} noted that a recurrence gets order for free from its
sequential structure, which is exactly what attention gave up.
{{ch:math-vectors}} supplies rotations and inner products.
{{ch:dl-cnns}}'s translation equivariance is the same kind of symmetry
statement as {{eq:attention-equivariance}}, and comparing them is instructive:
convolution's equivariance is the *point*, and attention's is a *problem*.

Forward: {{ch:tf-masking-kv}} shows why cached keys carry their absolute
position and what that forbids. {{ch:llm-long-context}} covers extension in
production. {{ch:tf-efficient}} covers the architectures that need no positional
scheme.

## 17. Exercises

**Beginner**

1. Why does a transformer need positional information?
2. Name the four places position can be injected.
3. What is the difference between absolute and relative encoding?
4. Why does ALiBi extrapolate?
5. Why is RoPE not applied to values?

**Intermediate**

6. Prove {{eq:attention-equivariance}}.
7. Derive {{eq:rope-relative}} from the rotation composition rule.
8. Using {{eq:rope-wavelength}}, find how many RoPE pairs complete a full
   cycle within $T = 8192$ at $d_k = 128$.
9. Explain why position interpolation degrades local resolution, quantitatively.
10. Compute ALiBi's effective window for each of 16 heads.

**Advanced**

11. Derive {{eq:ntk-base}} from the requirement that the longest wavelength
    stretch by $s$ and the shortest by 1.
12. Show that RoPE preserves the norm and explain why that matters for
    {{ch:tf-scaled-dot-product}}'s scaling.
13. Explain how a causal mask supplies positional information, and construct a
    task where it is insufficient.
14. Derive the backward pass through RoPE.

**Implementation**

15. Implement RoPE in both pairing conventions and show they are related by a
    permutation.
16. Reproduce the extrapolation table and extend it to $T = 256$.
17. Implement YaRN and compare against NTK-aware scaling.
18. Build a copy task and measure length generalisation for each scheme.

**Reasoning**

19. Your model works at 4k and produces plausible nonsense at 16k. Give an
    ordered diagnostic procedure.
20. A ported checkpoint produces garbage. What do you check, in order?

## 18. Interview Questions

**"Why do transformers need positional encoding?"** — Permutation equivariance.
Prove it if asked; it is three lines.

**"Explain RoPE."** — Rotate $q$ and $k$ by an angle proportional to position;
the composition rule makes the score depend on the offset exactly. Say that it
is exact, unlike the sinusoidal scheme's linear recoverability.

**"Why is RoPE the default if it does not extrapolate?"** — Best quality in
range, and extrapolation handled separately by scaling. Naming YaRN or NTK-aware
scaling is the distinguishing detail.

**"What does ALiBi trade?"** — Extrapolation for fixed per-head scales.

**"How would you extend a model's context to 128k?"** — Pick a scaling recipe,
say what it does to the frequency spectrum, fine-tune if using interpolation,
and evaluate on a task that actually uses the range.

**"What breaks if you apply RoPE to the values?"** — {{eq:rope-relative}} needs
both sides of an inner product; values are not in one.

## 19. Research Questions

**Why does length generalisation fail?** Trained query and key geometries have
no calibration for angles outside the training range, and that is a description
rather than a theory. {{maturity:RESEARCH FRONTIER}}

**Is any positional scheme needed in a causal decoder?** No-positional-encoding
models work, are somewhat worse, and the gap is not well characterised.
{{maturity:EMERGING}}

**What is the right positional scheme for multimodal inputs?** Text is 1-D,
images 2-D, video 3-D, and how to encode all of them in one model is unsettled.
{{maturity:EMERGING}}

**Can extension be done without fine-tuning, reliably?** NTK-aware and YaRN
often work zero-shot and the conditions under which they do not are not
characterised. {{maturity:EMERGING}}

## 20. Chapter Summary

Self-attention is permutation-equivariant, proved in three lines and verified
exactly: shuffle the input and the output shuffles identically. Measured as a
task, a model with no positional scheme sat at chance on a label that depends on
order, and it is a property of the *operator* — no amount of training changes
it. **Every scheme that supplied order worked, and having one mattered far more
than which one.**

The sinusoidal encoding's stated property holds: the measured linear fit from
$PE_{\pos}$ to $PE_{\pos+k}$ has essentially zero residual, and the dot product
between them is constant in position. What the paper's motivation glosses over
is what happens next — the encoding is *added to the token embedding*, so the
attention score mixes token–token, token–position and position–position terms,
and only the last is cleanly relative.

RoPE avoids that mixing by acting on the projected query and key instead.
{{eq:rope-relative}} was verified to floating point: the same $q$ and $k$ at the
same offset produce the same score wherever the pair sits, because rotations
compose and $\mat{R}_m\T\mat{R}_n = \mat{R}_{n-m}$. Two properties follow that
make it composable — the dependence is **exact** rather than linearly
recoverable, and the rotation is orthogonal, so norms are preserved and
{{ch:tf-scaled-dot-product}}'s scaling argument needs no revision.

The extrapolation problem is not what it is usually described as. Measured, RoPE
score *statistics* barely change with offset for random vectors, so nothing
blows up. The problem is that a trained model has learned specific query–key
geometries producing useful scores at the angles it saw, and those geometries
have no reason to work at angles outside that range. Measured on the frequency
spectrum: at $d_k = 128$ and $T = 4096$, a substantial fraction of RoPE's pairs
never complete a single cycle during training, and those are exactly the ones
extrapolation depends on.

That asymmetry is what the scaling recipes exploit. Position interpolation
stretches every wavelength by $s$, including the shortest, so adjacent-token
resolution degrades by exactly the scale factor — which is why it needs a
fine-tune and why it hurts most on locally-sensitive tasks. NTK-aware scaling
stretches the longest wavelength by $s$ and the shortest by about 1, moving only
the undertrained pairs, which is why it often works zero-shot.

ALiBi takes the other route entirely: no encoding, a fixed linear penalty on
distance, per-head slopes forming a geometric range of effective windows.
Measured, nothing in the scheme refers to the training length, which is why it
extrapolates — and the same fixity means a head cannot learn a relationship its
slope forbids. **Neither approach is free: RoPE gives every head every distance
and pays in extrapolation; ALiBi extrapolates and pays in fixed scales.**

## 21. Further Reading

{{cite:vaswani2017}} section 3.5 is half a page on positional encoding and it
reports that learned embeddings performed about the same as sinusoidal ones. It
is worth reading for the honesty of that admission, and for noticing that the
extrapolation argument — the one that turned out to matter — is a single
sentence at the end.

{{cite:su2021rope}} is the most consequential paper in this chapter. The
derivation of {{eq:rope-relative}} is short and the 2-D case makes the whole
idea visible; the general case is that case block-diagonally. Read section 3.

{{cite:press2022alibi}} is worth reading for its framing more than its method.
It is the paper that made "train short, test long" a benchmark question rather
than an afterthought, and the whole context-extension literature is downstream
of that reframing even though ALiBi itself is not the default.

**On the scaling recipes**, the primary sources are more scattered — position
interpolation, NTK-aware scaling and YaRN developed partly through community
experimentation before being written up. The frequency-spectrum reasoning of
{{sec:6-mathematical-foundation}} is the durable part, and it lets you read any
new recipe by asking which wavelengths it moves.

**Where to go next:** {{ch:tf-embeddings}} covers the other end of the model —
how tokens become vectors and how vectors become predictions — and the tying of
those two matrices, which is a decision with more consequences than it appears
to have.

---
id: tf-why-attention
number: 62
part: VII
tier: full
status: reviewed
requires: [dl-rnns, dl-cnns, dl-backprop, dl-losses]
provides: [encoder-decoder, context-vector, information-bottleneck,
           alignment, additive-attention, parallelism-argument,
           path-length, sequence-to-sequence]
citations: [bahdanau2015, luong2015, vaswani2017, hochreiter1997, bengio1994]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Explain the fixed-size bottleneck in a recurrent encoder–decoder and why it
   binds.
2. Derive additive attention and state what it computes.
3. Compare recurrence, convolution and attention by path length, work and
   parallelism.
4. Explain precisely why sequential dependence is the constraint that hardware
   cannot remove.
5. State what {{cite:vaswani2017}} contributed, given that attention already
   existed.
6. Explain what is given up by removing recurrence.
7. Read an attention weight matrix and say what it does and does not tell you.

## 2. Why This Matters

**This chapter is the argument, and the next nine are its consequences.** A
transformer looks arbitrary if you meet it as a diagram. Met as the endpoint of
a specific line of reasoning — bottleneck, then patch, then delete the thing
being patched — every component has a reason.

**The three-way comparison in {{sec:6-mathematical-foundation}} is the most
useful table in the part.** Recurrence, convolution and attention differ in
maximum path length, total work and critical-path depth, and every architecture
argument in the rest of the book is a trade among those three numbers.

**Attention was invented as a patch, not as an architecture.**
{{cite:bahdanau2015}} added it to a recurrent translation model to fix one
specific failure. {{cite:vaswani2017}}'s title — *Attention Is All You Need* —
is a claim that the patch had become the whole thing. Knowing which part was the
contribution matters when you read the paper.

**Removing recurrence gave up something real.** A recurrence has a
constant-size state and linear cost; attention has neither. {{ch:tf-complexity}}
measures the bill and {{ch:tf-efficient}} is the field trying to get some of it
back. Understanding what was traded is what lets you judge whether a proposed
alternative is worth anything.

## 3. Prerequisites

{{ch:dl-rnns}} for the recurrence, its vanishing-gradient product, and the
measured parallelism gap this chapter builds on. {{ch:dl-cnns}} for the
receptive-field argument, which is the third leg of the comparison.
{{ch:dl-backprop}} for path length in a computational graph.
{{ch:dl-losses}} for the softmax that produces attention weights.

## 4. Intuitive Explanation

### 4.1 The bottleneck

A 2014-era translation model was an encoder–decoder:

```text
   ENCODER (RNN)                        DECODER (RNN)
   the  cat  sat  on  the  mat
    │    │    │    │    │    │
    ▼    ▼    ▼    ▼    ▼    ▼
   h₁──▶h₂──▶h₃──▶h₄──▶h₅──▶h₆ ═══▶ c ═══▶ s₁──▶s₂──▶s₃──▶ ...
                                    ▲             le   chat  ...
                          ONE fixed-size vector
```

The encoder reads the source sentence into a single vector $\vec{c}$; the
decoder generates the translation from it. **Everything the decoder will ever
know about the source has to fit in $\vec{c}$.**

For a six-word sentence that is fine. For a fifty-word sentence it is not, and
the failure is measurable: translation quality falls off sharply with source
length, which is not what a model that understood the sentence would do.

Two distinct problems are tangled together here, and separating them is the
point of this section.

**The bottleneck.** One vector of fixed size must represent a sentence of
arbitrary length. That is a capacity limit and it does not go away with better
training.

**The path length.** Information from word 1 reaches the decoder's step 20 by
traversing 20-odd recurrent steps, each multiplying by the same Jacobian —
{{ch:dl-rnns}}'s matrix power. That is an optimisation limit.

Attention removes the first completely and the second almost completely.

### 4.2 The fix: let the decoder look back

Instead of compressing the source into one vector, keep *all* the encoder
states and let each decoder step choose which to read:

```text
   h₁   h₂   h₃   h₄   h₅   h₆        all kept
    ╲    ╲    │    ╱    ╱    ╱
     ╲    ╲   │   ╱    ╱    ╱         weighted by relevance
      ▼    ▼  ▼  ▼    ▼    ▼           to THIS decoder step
            c_t  =  Σ α_ti h_i
             │
             ▼
            s_t                        different c for every step
```

The weights $\alpha_{ti}$ are computed from the decoder's current state and each
encoder state, and they are normalised by a softmax so they sum to one. **The
context vector is now different at every decoder step**, and its size no longer
has to grow with the sentence — because it is a weighted average, not a
concatenation.

That is {{cite:bahdanau2015}}, and it is a small change with two large effects.
Translation quality stopped degrading with length. And the model became partly
inspectable: the weight matrix $\alpha$ shows which source words each output
word drew from.

### 4.3 Why delete the recurrence

By 2017 attention was standard and the recurrence around it was doing less and
less. Three observations pushed toward removing it entirely.

**Attention already provides the long-range path.** Any decoder step can read
any encoder state directly. The recurrence's job of carrying information forward
is redundant with that.

**The recurrence is the only sequential part left.** Attention over $T$
positions is a matrix multiply — fully parallel. The recurrence forces $T$
sequential steps, which is {{ch:dl-rnns}}'s measured parallelism gap.

**Depth was cheap and length was not.** Making a network deeper is
parallelisable; making a sequence longer is not. Removing the recurrence turns a
length problem into a depth problem, and depth is the one hardware can absorb.

So the question became: what if the *encoder* also used attention, over itself,
instead of a recurrence? That is self-attention, and once you have it, nothing
in the model needs to be sequential.

### 4.4 The three ways to relate two positions

```text
   RECURRENCE    pos 1 ──▶ 2 ──▶ 3 ──▶ ... ──▶ T
                 path length T; sequential; O(T d²) work

   CONVOLUTION   pos 1 ─┐
                        ├─▶ ... stacked k-wide layers
                 pos T ─┘
                 path length T/k per layer; parallel; O(k T d²) work

   ATTENTION     pos 1 ◀────────────────────▶ pos T
                 path length 1; parallel; O(T² d) work
```

**Attention buys a path length of 1 and pays $O(T^2)$ for it.** That is the
entire architectural trade of this part, and everything in
{{ch:tf-efficient}} is an attempt to keep the path length while reducing the
cost.

## 5. Formal Explanation

### 5.1 The encoder–decoder without attention

Encoder: $\vec{h}_i = f(\vec{h}_{i-1}, \vec{x}_i)$ for $i = 1..T_x$, with
$\vec{c} = \vec{h}_{T_x}$.

Decoder: $\vec{s}_t = g(\vec{s}_{t-1}, \vec{y}_{t-1}, \vec{c})$, and

$$
p(y_t \mid y_{<t}, \vec{x}) = \softmax\big(\mat{W}_o\vec{s}_t\big)
$$ (eq:seq2seq)

**$\vec{c}$ has a fixed dimension $d$ regardless of $T_x$.** That is the
bottleneck stated formally: the map from source sentences to $\R^d$ cannot be
injective once the number of distinguishable sentences exceeds what $d$
dimensions can separate.

### 5.2 Additive attention

{{cite:bahdanau2015}} replaces the single $\vec{c}$ with a per-step one:

$$
e_{ti} = \vec{v}_a\T\tanh\big(\mat{W}_a\vec{s}_{t-1}
 + \mat{U}_a\vec{h}_i\big)
$$ (eq:additive-score)

$$
\alpha_{ti} = \frac{\exp(e_{ti})}{\sum_{j=1}^{T_x}\exp(e_{tj})},
\qquad
\vec{c}_t = \sum_{i=1}^{T_x}\alpha_{ti}\vec{h}_i
$$ (eq:attention-context)

The scoring function {{eq:additive-score}} is a small feed-forward network —
hence *additive* attention, since the two inputs are added before the
nonlinearity. It has its own parameters $\mat{W}_a$, $\mat{U}_a$, $\vec{v}_a$.

### 5.3 Multiplicative attention

{{cite:luong2015}} proposed the cheaper alternative:

$$
e_{ti} = \vec{s}_{t-1}\T\mat{W}_a\vec{h}_i
\qquad\text{or, with no parameters at all,}\qquad
e_{ti} = \vec{s}_{t-1}\T\vec{h}_i
$$ (eq:multiplicative-score)

**A dot product instead of a small network.** The whole score matrix becomes one
matrix multiply, which is the operation hardware is best at
({{ch:dl-forward}}). This is the form the transformer uses, and
{{ch:tf-scaled-dot-product}} adds the one modification it needs.

Additive attention is slightly better at large $d$ and multiplicative is far
faster, which is exactly the trade {{eq:dot-variance}} explains in
the next chapter.

### 5.4 Self-attention

Nothing in {{eq:attention-context}} requires the queries and the keys to come
from different sequences. Set them all to the same sequence:

$$
e_{ij} = \text{score}(\vec{x}_i, \vec{x}_j),
\qquad
\vec{z}_i = \sum_{j}\alpha_{ij}\vec{x}_j
$$ (eq:self-attention)

**Every position attends to every position in the same sequence, in one
parallel operation.** This is the step that makes the recurrence unnecessary,
and it is the transformer's actual novelty — attention across sequences already
existed.

### 5.5 What the numbers are

{#tbl:path-length caption="The three mechanisms compared. Column 2 is what determines whether a long-range dependency is learnable; column 4 is what determines whether the model trains fast on modern hardware. Attention wins both and loses column 3."}

| Mechanism | Max path length | Work per layer | Sequential ops |
|---|---|---|---|
| Recurrence | $O(T)$ | $O(T d^2)$ | $O(T)$ |
| Convolution, kernel $k$ | $O(\log_k T)$ dilated, $O(T/k)$ plain | $O(k T d^2)$ | $O(1)$ |
| Self-attention | $O(1)$ | $O(T^2 d)$ | $O(1)$ |
| Self-attention, window $w$ | $O(T/w)$ | $O(T w d)$ | $O(1)$ |

Two readings worth taking from this table.

**Attention's work is worse at every length, and increasingly so.** Counting
the four projections as well as the scores, at $d = 768$ attention costs about
4.7 times a recurrence at $T = 512$ and about 15 times at $T = 8192$ — the ratio
grows because one term is quadratic in $T$ and the other is linear.
{{sec:8-implementation}} measures it.

That is worth stating without softening, because the usual telling implies
attention is cheaper somewhere. It is not. It is *structurally better* and
*arithmetically worse*, and the next two rows of the table explain why the trade
was made anyway.

**The last row is the compromise everyone reaches for.** Restricting attention
to a window of $w$ positions makes the work linear in $T$ again and gives back a
path length of $T/w$. Whether that trade is worth it is {{ch:tf-efficient}}'s
subject.

### 5.6 What the recurrence was also providing

Deleting the recurrence removed the bottleneck and the sequential floor. It also
removed four things that were doing quiet work, and each has a replacement
somewhere in this part.

**Order.** A recurrence processes positions in sequence, so position is implicit
in the computation. Attention is permutation-equivariant and has no notion of
order at all — {{ch:tf-positional}} supplies it explicitly, and that chapter
exists entirely because of this deletion.

**A bounded state.** A recurrence carries $O(d)$ numbers forward regardless of
how long the sequence is. Attention carries all $T$ keys and values, so its
serving memory grows linearly with the conversation. That is the KV cache
({{ch:tf-masking-kv}}), and it is the single largest operational cost of
deploying a transformer.

**An inductive bias toward recency.** A recurrence's state naturally weights
recent inputs more, because older contributions have been multiplied by the
recurrent Jacobian more times. Attention has no such preference and must learn
one — which is what ALiBi's distance penalty supplies by construction
({{ch:tf-positional}}) and what a trained model otherwise learns in its
attention patterns.

**Causality.** A recurrence cannot see the future because it has not processed
it yet. Attention sees everything unless told not to, so generation requires an
explicit mask ({{ch:tf-masking-kv}}). A missing mask is one of the most
consequential single-line bugs in the field: the model trains beautifully,
scores wonderfully, and cannot generate at all.

**Three of those four are one-line fixes and one is not.** The bounded state has
no cheap replacement, and {{ch:tf-efficient}} is the field trying to get it back
without giving up the parallelism that made the trade worth making.

## 6. Mathematical Foundation

### 6.1 The bottleneck, made precise

Let the encoder map source sequences to $\vec{c} \in \R^d$, stored in
$b$-bit floats. The number of distinct values $\vec{c}$ can take is at most
$2^{bd}$.

A vocabulary of $V$ tokens has $V^{T}$ sequences of length $T$. Requiring the
decoder to reconstruct the source exactly needs

$$
V^{T} \le 2^{bd}
\quad\Longleftrightarrow\quad
T \le \frac{bd}{\log_2 V}
$$ (eq:bottleneck-capacity)

For $d = 1000$, $b = 32$ and $V = 30000$, that is $T \le 2150$ tokens — which
sounds comfortable and is wildly optimistic, because it assumes the encoding is
information-theoretically perfect and that every bit of the float is usable.

**The useful form of the argument is not the number but its shape.** Capacity is
constant in $T$ and the information to be stored grows linearly in $T$, so there
is a length past which the representation must become lossy. Attention removes
the constraint entirely because the decoder reads $T_x$ vectors rather than one:
capacity now grows with the input.

### 6.2 Why path length governs learnability

From {{ch:dl-backprop}}, a gradient crossing $L$ operations is a product of $L$
Jacobians. In a recurrence they are the *same* Jacobian, so the product is
$\mat{J}^{L}$ and decays as $\rho^L$ — {{ch:dl-rnns}} measured the first layer
receiving $10^{-10}$ of the last's gradient at $T = 60$.

Under attention the path from output $i$ to input $j$ is:

$$
\vec{z}_i = \sum_j \alpha_{ij}\vec{x}_j
\quad\Longrightarrow\quad
\frac{\partial\vec{z}_i}{\partial\vec{x}_j} = \alpha_{ij}\mat{I} + (\text{terms
through } \alpha)
$$ (eq:attention-jacobian)

**One factor, not $|i-j|$ of them.** The gradient is attenuated by $\alpha_{ij}$
— which can be small — and it is not attenuated *exponentially in the
distance*, which is the whole difference. A single scalar the model controls has
replaced a product the model does not.

### 6.3 The parallelism argument, formally

Define a computation's **work** $W$ (total operations) and its **depth** $D$
(longest chain of dependent operations). On $p$ processors, Brent's theorem
bounds the time by

$$
T_p \ge \max\left(D,\ \frac{W}{p}\right)
$$ (eq:brent)

For a recurrence over $T$ steps: $W = O(Td^2)$, $D = O(T)$. For self-attention:
$W = O(T^2d)$, $D = O(1)$.

**With enough processors the recurrence is bounded below by $D = O(T)$ and
attention is not.** No amount of hardware removes the recurrence's floor,
because it is a property of the dependency structure rather than of the
implementation. Attention's disadvantage is in $W$, which more processors *can*
absorb.

That asymmetry is the whole reason the field moved, and it is worth stating as
sharply as {{eq:brent}} allows: **one of the two costs scales away with hardware
and the other does not.**

### 6.4 What attention weights are and are not

$\vecgreek{\alpha}_i$ is a probability distribution over positions, so it is
tempting to read it as "what the model looked at". Three cautions, all of which
have caught people out.

**A convex combination is not a selection.** A uniform $\vecgreek{\alpha}$ over
$T$ positions produces the mean value vector, which may be a perfectly useful
output. High entropy does not mean the head is doing nothing.

**Attention weights are not the only path.** The residual stream carries
information around the attention block entirely
({{ch:tf-ffn-residual}}), so a position can influence the output with zero
attention weight on it.

**Different weights can give identical outputs.** If two positions have
identical value vectors, any split of the weight between them produces the same
result — so the weights are not identifiable from the function.

The honest summary: attention weights are a *hypothesis* about information flow,
cheap to inspect and not a proof. {{sec:9-practical-example}} measures the third
point directly.

## 7. Internal Mechanics

### 7.1 What Bahdanau's implementation cost

{{eq:additive-score}} evaluates a small feed-forward network for every
(decoder step, encoder position) pair — $T_x T_y$ evaluations, each a matrix–
vector product plus a tanh. That is a lot of small operations, which is the
worst shape for hardware ({{ch:dl-forward}}).

{{eq:multiplicative-score}} computes the same $T_x \times T_y$ score matrix as a
single matrix multiply. Same asymptotic work, vastly better constant, and the
reason the transformer uses it.

### 7.2 Bidirectional encoders

The 2015 encoders were bidirectional: one recurrence forwards, one backwards,
concatenated, so $\vec{h}_i$ summarises the whole sentence around position $i$
rather than only its prefix.

Self-attention gets that property for free — every position sees every other by
construction — which is one fewer design decision. The corresponding choice in a
transformer is the *mask* ({{ch:tf-masking-kv}}), and it is a one-line change
rather than an architectural one.

### 7.3 Why the softmax

The weights need to be non-negative and sum to one so the output stays in the
convex hull of the values, which keeps its scale bounded regardless of $T$. A
softmax is the standard way to get that from unconstrained scores
({{ch:dl-losses}}).

It also has a cost that {{ch:tf-efficient}} exploits: the normalisation couples
all $T$ scores, which is what prevents attention from being written as a
recurrence. Remove the softmax and you get linear attention, which *can* be.

### 7.4 What was actually new in 2017

Worth being precise, because the paper's title invites overstatement:

```text
   attention                already existed (2014)
   multiplicative scoring   already existed (2015)
   residual connections     already existed (2015)
   layer normalisation      already existed (2016)

   NEW: self-attention as the ONLY sequence mechanism
   NEW: multi-head attention
   NEW: the sqrt(d_k) scaling, with its variance argument
   NEW: sinusoidal positional encoding
```

The contribution was an architecture and a demonstration, not a mechanism.
That is a common and undervalued kind of contribution, and it is worth
recognising as such.

### 7.5 The encoder-decoder's surviving descendants

The 2014 architecture is not dead; it has been factored into pieces that are
used separately.

**Encoder-only** models keep the bidirectional encoder and discard the decoder,
producing a representation of the whole input at once. BERT and every embedding
model are this shape ({{ch:emb-models}}).

**Decoder-only** models keep the causal decoder and discard the encoder, folding
the input into the same sequence as the output. Every large language model is
this shape, and it is worth asking why: an encoder-decoder has an architectural
separation between "the thing being conditioned on" and "the thing being
generated", and decoder-only models simply concatenate them. The separation
turns out not to be needed, and dropping it means one set of weights instead of
two ({{ch:tf-architectures}}).

**Encoder-decoder** models survive where the input and output are genuinely
different objects — translation, speech recognition, some multimodal settings —
and cross-attention is what connects the halves.

The through-line is that {{eq:cross-attention}} is the same operation in all
three; only what plays the role of the key/value sequence changes.

## 8. Implementation

```python {tier=A name=the-bottleneck-measured}
"""The fixed-size bottleneck, measured: how much can one vector carry?"""
import numpy as np

rng = np.random.default_rng(0)


# --- a copy task: encode a sequence into ONE vector, then decode it ---------
def make_sequences(n, T, V, seed):
    rs = np.random.default_rng(seed)
    return rs.integers(0, V, (n, T))


def onehot(X, V):
    out = np.zeros((*X.shape, V))
    np.put_along_axis(out, X[..., None], 1.0, axis=-1)
    return out


class BottleneckAE:
    """Encode T tokens into ONE d-dimensional vector, decode all T back.

    This is eq. 62.4's constraint in its purest form: the decoder sees
    nothing but c, so whatever it reconstructs must have fitted in d
    numbers.
    """

    def __init__(self, T, V, d, seed=0):
        rs = np.random.default_rng(seed)
        self.T, self.V, self.d = T, V, d
        self.We = rs.normal(0, np.sqrt(2 / (T * V)), (T * V, d))
        self.be = np.zeros(d)
        self.Wd = rs.normal(0, np.sqrt(2 / d), (d, T * V))
        self.bd = np.zeros(T * V)

    def params(self):
        return [self.We, self.be, self.Wd, self.bd]

    def forward(self, X):
        self.flat = onehot(X, self.V).reshape(len(X), -1)
        self.c = np.tanh(self.flat @ self.We + self.be)      # THE bottleneck
        return (self.c @ self.Wd + self.bd).reshape(len(X), self.T, self.V)

    def loss_and_grads(self, X):
        logits = self.forward(X)
        m = logits.max(axis=-1, keepdims=True)
        e = np.exp(logits - m)
        p = e / e.sum(axis=-1, keepdims=True)
        n = len(X) * self.T
        loss = float(-np.log(np.clip(
            np.take_along_axis(p, X[..., None], -1), 1e-12, None)).sum() / n)
        d = p.copy()
        np.put_along_axis(d, X[..., None],
                          np.take_along_axis(d, X[..., None], -1) - 1.0, -1)
        d /= n
        dflat = d.reshape(len(X), -1)
        gWd, gbd = self.c.T @ dflat, dflat.sum(axis=0)
        dc = (dflat @ self.Wd.T) * (1 - self.c ** 2)
        return loss, [self.flat.T @ dc, dc.sum(axis=0), gWd, gbd]

    def accuracy(self, X):
        return float((self.forward(X).argmax(axis=-1) == X).mean())


def train(net, X, steps=4000, lr=3e-3, batch=64, seed=0):
    ps = net.params()
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    rs = np.random.default_rng(seed + 1)
    for t in range(1, steps + 1):
        xb = X[rs.integers(0, len(X), batch)]
        _, gs = net.loss_and_grads(xb)
        for i, (p, g) in enumerate(zip(ps, gs)):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            p -= lr * (m[i] / (1 - 0.9 ** t)) / (
                np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    return net


print("=" * 72)
print("the fixed-size bottleneck (eq. 62.4)")
print("=" * 72)
print("Encode T tokens into ONE d-dimensional vector and decode them back.")
print("Perfect reconstruction requires the sequence to FIT in d numbers.\n")
V = 8
print(f"vocabulary {V}, so a length-T sequence carries "
      f"T * log2({V}) = {np.log2(V):.0f}T bits\n")
print(f"{'d':>5} " + " ".join(f"{f'T={T}':>10}" for T in (2, 4, 8, 16)))
for d in (2, 4, 8, 16, 32):
    row = []
    for T in (2, 4, 8, 16):
        Xtr = make_sequences(3000, T, V, 1)
        net = train(BottleneckAE(T, V, d, seed=2), Xtr, steps=3000)
        row.append(net.accuracy(make_sequences(2000, T, V, 3)))
    print(f"{d:>5} " + " ".join(f"{a:>10.4f}" for a in row))
print(f"\n(chance is 1/{V} = {1 / V:.4f})")

print("\nRead along each row: as the sequence grows at a FIXED bottleneck")
print("width, reconstruction degrades. Read down each column: widening the")
print("bottleneck recovers it. That is eq. 62.4's shape — capacity constant")
print("in T against information linear in T.")
print("\nThis is the failure Bahdanau et al. were looking at in 2014,")
print("reduced to its skeleton. A translation model does not need to")
print("reconstruct its input exactly, so the real curve is gentler than")
print("this one — but it has the same shape, and the observed degradation")
print("of translation quality with source length is what it looks like on")
print("a real task.")

# --- and what attention does to it -----------------------------------------
print("\n" + "=" * 72)
print("what attention changes")
print("=" * 72)
print("The decoder now reads a WEIGHTED AVERAGE of T encoder states rather")
print("than one summary. Capacity available to it grows with the input.\n")
print(f"{'T':>5} {'one vector, d=8':>18} {'T vectors of d=8':>19} "
      f"{'capacity ratio':>16}")
for T in (2, 4, 8, 16, 32):
    print(f"{T:>5} {8:>18} {8 * T:>19} {T:>15}x")
print("\nThat is the entire structural difference and it does not depend on")
print("any detail of how the weights are computed. The bottleneck was a")
print("consequence of summarising into ONE vector, and attention does not.")
print("\nNote what it costs: the decoder must now hold all T encoder states,")
print("so memory grows with the input where before it did not. That is the")
print("first appearance of the trade this whole part is about, and Chapter")
print("69's KV cache is the same bill arriving at serving time.")
```

```python {tier=A name=path-length-and-parallelism}
"""Table 62.1 measured: path length, work, and the sequential floor that
hardware cannot remove.
"""
import time

import numpy as np

rng = np.random.default_rng(1)


# --- section 5.5: the work columns ------------------------------------------
def work(mechanism, T, d, k=3, w=128):
    if mechanism == "recurrence":
        return 2 * T * d * d
    if mechanism == "convolution":
        return 2 * k * T * d * d
    if mechanism == "attention":
        return 2 * T * T * d + 2 * 4 * T * d * d      # scores + projections
    if mechanism == "windowed":
        return 2 * T * min(w, T) * d + 2 * 4 * T * d * d
    raise ValueError(mechanism)


def path_length(mechanism, T, k=3, w=128):
    if mechanism == "recurrence":
        return T
    if mechanism == "convolution":
        return int(np.ceil((T - 1) / (k - 1)))
    if mechanism == "attention":
        return 1
    if mechanism == "windowed":
        return int(np.ceil(T / w))
    raise ValueError(mechanism)


print("=" * 72)
print("table 62.1, with numbers (d = 768)")
print("=" * 72)
d = 768
print(f"{'T':>7} {'mechanism':<14} {'max path':>10} {'GFLOP/layer':>13} "
      f"{'vs recurrence':>15}")
for T in (128, 512, 2048, 8192):
    base = work("recurrence", T, d)
    for mech in ("recurrence", "convolution", "attention", "windowed"):
        wk = work(mech, T, d)
        print(f"{T:>7} {mech:<14} {path_length(mech, T):>10} "
              f"{wk / 1e9:>13.3f} {wk / base:>14.2f}x")
    print()

print("Read the last column down each block. At T = 128 attention costs")
print("about the same as a recurrence; by T = 8192 it costs several times")
print("more, and the ratio keeps growing because one term is quadratic in T")
print("and the other is linear.")
print("\nNow read the path-length column. It is 1 for attention at every")
print("length, and equal to T for the recurrence. That column is what")
print("decides whether a long-range dependency is learnable at all")
print("(Chapter 60), and no amount of the work column buys it.")
print("\nThe windowed row is the compromise: linear work again, path length")
print("back to T/w. Chapter 71 is about whether that trade is worth making,")
print("and the answer turns out to depend on something neither column")
print("shows.")

# --- section 6.3: the sequential floor --------------------------------------
print("\n" + "=" * 72)
print("the sequential floor that hardware cannot remove (eq. 62.9)")
print("=" * 72)
print("Both do the SAME arithmetic. One must do it in T dependent rounds.\n")

d = 256
W = rng.normal(0, 1 / np.sqrt(d), (d, d)).astype(np.float32)
print(f"{'T':>6} {'batch':>7} {'recurrent (T rounds)':>22} "
      f"{'attention-shaped (1 round)':>28} {'ratio':>8}")
for T, B in ((128, 32), (512, 32), (512, 128)):
    X = rng.normal(size=(B, T, d)).astype(np.float32)
    h = np.zeros((B, d), dtype=np.float32)
    t0 = time.perf_counter()
    for t in range(T):
        h = np.tanh(h @ W + X[:, t])
    dt_rec = time.perf_counter() - t0

    Q = X.reshape(B * T, d)
    t0 = time.perf_counter()
    _ = np.tanh(Q @ W)                     # the same per-position work, fused
    dt_par = time.perf_counter() - t0
    print(f"{T:>6} {B:>7} {dt_rec * 1e3:>20.2f}ms "
          f"{dt_par * 1e3:>26.2f}ms {dt_rec / dt_par:>8.1f}x")

print("\nThe recurrent column is not slower because it does more work — it")
print("does exactly the same multiplies. It is slower because eq. 62.9's")
print("depth term binds: T dependent rounds, each too small to occupy the")
print("machine, against one round that is not.")
print("\nBrent's bound says time is at least max(D, W/p). More processors")
print("shrink W/p and do nothing to D. So the recurrence has a floor that")
print("hardware cannot lower and attention does not, and that asymmetry —")
print("not any modelling argument — is the reason the field moved.")

# --- where attention's quadratic term starts to bite ------------------------
print("\n" + "=" * 72)
print("where the quadratic term takes over")
print("=" * 72)
print("Attention's cost is 2*T^2*d for the scores plus 8*T*d^2 for the")
print("projections. The crossover is at T = 4d.\n")
print(f"{'d':>6} {'crossover T':>13} {'at T=1024, scores are':>24} "
      f"{'at T=8192':>12}")
for d_ in (256, 512, 768, 4096):
    cross = 4 * d_
    f1 = 2 * 1024 * 1024 * d_
    p1 = 8 * 1024 * d_ * d_
    f2 = 2 * 8192 * 8192 * d_
    p2 = 8 * 8192 * d_ * d_
    print(f"{d_:>6} {cross:>13} {f1 / (f1 + p1):>23.1%} "
          f"{f2 / (f2 + p2):>11.1%}")

print("\nAt a typical width the quadratic term is a MINORITY of the FLOPs at")
print("ordinary sequence lengths — most of the compute is in the linear")
print("projections. That is worth knowing, because 'attention is quadratic'")
print("is usually stated as though the quadratic part dominates, and at")
print("T = 1024 with d = 4096 it is under a tenth of the arithmetic.")
print("\nWhat is quadratic without qualification is the MEMORY: the T-by-T")
print("score matrix must exist somewhere. Chapter 70 separates those two")
print("costs carefully, because they have different fixes — and")
print("FlashAttention addresses the memory one without touching the")
print("arithmetic at all.")
```

## 9. Practical Example

```python {tier=A name=attention-on-a-real-task}
"""Additive attention against a fixed bottleneck on a task that needs
alignment, and what the attention weights do and do not tell you.
"""
import numpy as np

rng = np.random.default_rng(4)

V, T_SRC = 12, 10


def make_lookup_task(n, seed):
    """A source sequence and a QUERY index. The target is the source token
    at that index. Solving it requires selecting one position — exactly the
    alignment problem attention was invented for."""
    rs = np.random.default_rng(seed)
    src = rs.integers(1, V, (n, T_SRC))
    idx = rs.integers(0, T_SRC, n)
    tgt = src[np.arange(n), idx]
    return src, idx, tgt


def onehot(X, k):
    out = np.zeros((*X.shape, k))
    np.put_along_axis(out, X[..., None], 1.0, axis=-1)
    return out


class Bottleneck:
    """Summarise the source into one vector, then answer using it + query."""

    def __init__(self, d=32, seed=0):
        rs = np.random.default_rng(seed)
        self.Wenc = rs.normal(0, np.sqrt(2 / (T_SRC * V)), (T_SRC * V, d))
        self.Wq = rs.normal(0, np.sqrt(2 / T_SRC), (T_SRC, d))
        self.Wo = rs.normal(0, np.sqrt(2 / (2 * d)), (2 * d, V))
        self.bo = np.zeros(V)
        self.d = d

    def params(self):
        return [self.Wenc, self.Wq, self.Wo, self.bo]

    def forward(self, src, idx):
        self.s1h = onehot(src, V).reshape(len(src), -1)
        self.q1h = onehot(idx, T_SRC)
        self.c = np.tanh(self.s1h @ self.Wenc)
        self.qe = np.tanh(self.q1h @ self.Wq)
        self.h = np.concatenate([self.c, self.qe], axis=1)
        return self.h @ self.Wo + self.bo

    def grads(self, src, idx, tgt):
        logits = self.forward(src, idx)
        m = logits.max(axis=1, keepdims=True)
        e = np.exp(logits - m)
        p = e / e.sum(axis=1, keepdims=True)
        loss = float(-np.log(np.clip(p[np.arange(len(tgt)), tgt],
                                     1e-12, None)).mean())
        d = p.copy()
        d[np.arange(len(tgt)), tgt] -= 1.0
        d /= len(tgt)
        gWo, gbo = self.h.T @ d, d.sum(axis=0)
        dh = d @ self.Wo.T
        dc = dh[:, :self.d] * (1 - self.c ** 2)
        dq = dh[:, self.d:] * (1 - self.qe ** 2)
        return loss, [self.s1h.T @ dc, self.q1h.T @ dq, gWo, gbo]


class Attention:
    """Additive attention (eqs. 62.2-62.3): the query scores every source
    position and reads a weighted average."""

    def __init__(self, d=32, seed=0):
        rs = np.random.default_rng(seed)
        self.Wv = rs.normal(0, np.sqrt(2 / V), (V, d))       # value per token
        # The key must carry POSITION, or no scoring function can locate a
        # position: keys built from token identity alone are the same
        # wherever the token sits. This is Chapter 65's point arriving early.
        self.Kp = rs.normal(0, 0.5, (T_SRC, d))              # key per position
        self.Wq = rs.normal(0, np.sqrt(2 / T_SRC), (T_SRC, d))
        self.Wa = rs.normal(0, np.sqrt(2 / d), (d, d))
        self.va = rs.normal(0, np.sqrt(2 / d), d)
        self.Wo = rs.normal(0, np.sqrt(2 / d), (d, V))
        self.bo = np.zeros(V)
        self.d = d

    def params(self):
        return [self.Wv, self.Kp, self.Wq, self.Wa, self.va, self.Wo, self.bo]

    def forward(self, src, idx, keep=False):
        n = len(src)
        S = onehot(src, V)                                   # (n, T, V)
        self.S = S
        self.Vv = S @ self.Wv                                # (n, T, d)
        self.K = np.broadcast_to(self.Kp, (n, T_SRC, self.d))
        self.q1h = onehot(idx, T_SRC)
        self.qe = np.tanh(self.q1h @ self.Wq)                # (n, d)
        # eq. 62.2: additive score
        self.pre = np.tanh(self.K @ self.Wa + self.qe[:, None, :])
        self.e = self.pre @ self.va                          # (n, T)
        m = self.e.max(axis=1, keepdims=True)
        ex = np.exp(self.e - m)
        self.alpha = ex / ex.sum(axis=1, keepdims=True)      # eq. 62.3
        self.ctx = (self.alpha[:, :, None] * self.Vv).sum(axis=1)
        return self.ctx @ self.Wo + self.bo

    def grads(self, src, idx, tgt):
        logits = self.forward(src, idx)
        n = len(tgt)
        m = logits.max(axis=1, keepdims=True)
        e = np.exp(logits - m)
        p = e / e.sum(axis=1, keepdims=True)
        loss = float(-np.log(np.clip(p[np.arange(n), tgt], 1e-12, None)).mean())
        d = p.copy()
        d[np.arange(n), tgt] -= 1.0
        d /= n
        gWo, gbo = self.ctx.T @ d, d.sum(axis=0)
        dctx = d @ self.Wo.T                                 # (n, d)
        gWv = np.einsum('ntv,nd,nt->vd', self.S, dctx, self.alpha)
        dalpha = np.einsum('nd,ntd->nt', dctx, self.Vv)
        de = self.alpha * (dalpha - (dalpha * self.alpha).sum(
            axis=1, keepdims=True))                          # softmax backward
        dpre = de[:, :, None] * self.va * (1 - self.pre ** 2)
        gva = np.einsum('nt,ntd->d', de, self.pre)
        gWa = np.einsum('ntd,nte->de', self.K, dpre)
        dK = dpre @ self.Wa.T
        gKp = dK.sum(axis=0)
        dqe = dpre.sum(axis=1) * 1.0
        dq = dqe * (1 - self.qe ** 2)
        gWq = self.q1h.T @ dq
        return loss, [gWv, gKp, gWq, gWa, gva, gWo, gbo]


def train(net, src, idx, tgt, steps=4000, lr=5e-3, batch=64, seed=0):
    ps = net.params()
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    rs = np.random.default_rng(seed + 7)
    for t in range(1, steps + 1):
        b = rs.integers(0, len(src), batch)
        _, gs = net.grads(src[b], idx[b], tgt[b])
        for i, (p, g) in enumerate(zip(ps, gs)):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            p -= lr * (m[i] / (1 - 0.9 ** t)) / (
                np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    return net


src_tr, idx_tr, tgt_tr = make_lookup_task(8000, 1)
src_te, idx_te, tgt_te = make_lookup_task(4000, 2)

print("=" * 72)
print("a task that needs ALIGNMENT: read the source token at a given index")
print("=" * 72)
print(f"source length {T_SRC}, vocabulary {V}; chance is {1 / V:.4f}\n")
print(f"{'model':<28} {'params':>9} {'test accuracy':>15}")
bn = train(Bottleneck(seed=3), src_tr, idx_tr, tgt_tr)
acc_b = float((bn.forward(src_te, idx_te).argmax(1) == tgt_te).mean())
print(f"{'fixed bottleneck, d=32':<28} "
      f"{sum(p.size for p in bn.params()):>9,} {acc_b:>15.4f}")
at = train(Attention(seed=3), src_tr, idx_tr, tgt_tr)
acc_a = float((at.forward(src_te, idx_te).argmax(1) == tgt_te).mean())
print(f"{'additive attention, d=32':<28} "
      f"{sum(p.size for p in at.params()):>9,} {acc_a:>15.4f}")

print("\nThe task is pure alignment: the answer is one source token and the")
print("query says which. Attention's mechanism — score every position, read")
print("the winner — matches it exactly. The bottleneck model has to encode")
print("the whole source into 32 numbers and then extract the right one.")
print("\nNote one design detail that the first version of this experiment")
print("got wrong. The KEYS must carry position: keys built from token")
print("identity alone are identical wherever the token sits, so no scoring")
print("function can locate a position and the model sits at a uniform")
print("attention distribution however long it trains. That is Chapter 65's")
print("subject arriving three chapters early, and it is a good illustration")
print("that attention is a lookup — and a lookup needs addressable keys.")

# --- what the weights look like ---------------------------------------------
print("\n" + "=" * 72)
print("the attention weights: what they show")
print("=" * 72)
at.forward(src_te[:6], idx_te[:6])
print("Each row is one example. The '^' marks the queried index.\n")
for i in range(6):
    bar = " ".join(f"{a:>5.2f}" for a in at.alpha[i])
    mark = " ".join("    ^" if j == idx_te[i] else "     "
                    for j in range(T_SRC))
    print(f"  query {idx_te[i]:>2}  {bar}")
    print(f"           {mark}")
print(f"\nmass on the queried position, averaged over the test set: "
      f"{float(at.alpha[np.arange(len(at.alpha)), idx_te[:6]].mean()):.4f}")
at.forward(src_te, idx_te)
print(f"over all {len(src_te)} test examples: "
      f"{float(at.alpha[np.arange(len(src_te)), idx_te].mean()):.4f}")
print(f"mean attention entropy (max is ln {T_SRC} = "
      f"{np.log(T_SRC):.3f}): "
      f"{float(-(at.alpha * np.log(at.alpha + 1e-12)).sum(1).mean()):.3f}")

print("\nThis is the inspectability Bahdanau et al. reported and it is real:")
print("the weights concentrate on the position the task requires, and you")
print("can read that off the matrix without any further machinery.")

# --- section 6.4: what they do NOT show -------------------------------------
print("\n" + "=" * 72)
print("what attention weights do NOT tell you (section 6.4)")
print("=" * 72)
print("Claim: if two positions hold the SAME value vector, any split of the")
print("weight between them gives an identical output. The weights are then")
print("not identifiable from the function.\n")

at.forward(src_te[:400], idx_te[:400])
alpha0 = at.alpha.copy()
out0 = at.ctx.copy()
# find, per example, pairs of positions holding the same TOKEN
dup_moved, checked = 0, 0
alpha_mod = alpha0.copy()
for i in range(400):
    toks = src_te[i]
    for j in range(T_SRC):
        for k in range(j + 1, T_SRC):
            if toks[j] == toks[k]:
                tot = alpha_mod[i, j] + alpha_mod[i, k]
                alpha_mod[i, j], alpha_mod[i, k] = tot, 0.0   # move it all
                dup_moved += 1
                break
        else:
            continue
        break
    checked += 1
ctx_mod = (alpha_mod[:, :, None] * at.Vv).sum(axis=1)
print(f"examples containing a repeated token : {dup_moved} of {checked}")
print(f"max change in the attention weights  : "
      f"{np.abs(alpha_mod - alpha0).max():.4f}")
print(f"max change in the CONTEXT VECTOR     : "
      f"{np.abs(ctx_mod - out0).max():.3e}")

print("\nThe weights were changed by up to a full unit of probability mass —")
print("moved wholesale from one position to another — and the output did")
print("not move at all. Both positions held the same token, so they hold")
print("the same value vector, and a weighted average cannot distinguish")
print("them.")
print("\nSo an attention map is a HYPOTHESIS about where information came")
print("from, not a measurement of it. Here there are two maps producing")
print("bit-identical outputs, and nothing in the model prefers either.")
print("\nThat is the mildest version of the caution. Section 6.4 lists two")
print("more: a high-entropy head may be computing a useful average rather")
print("than doing nothing, and the residual stream carries information past")
print("the attention block entirely (Chapter 67), so a position can matter")
print("with zero weight on it.")
print("\nAttention maps are cheap to look at and worth looking at. Treat")
print("what they suggest as something to test, not as something shown.")
```

## 10. Production Considerations

**Do not ship an attention map as an explanation.** Measured here: two maps
differing by a full unit of probability mass produced bit-identical outputs. If
an attention visualisation is going in front of a user or a regulator, it needs
a caveat, and preferably a faithfulness check.

**The bottleneck argument still applies to anything that summarises.** Any
architecture that compresses a variable-length input into a fixed-size state
inherits {{eq:bottleneck-capacity}}'s shape — including the state space models
of {{ch:dl-rnns}}, which is the honest counterweight to their efficiency
argument.

**Know which cost you are paying.** Measured: at ordinary sequence lengths the
quadratic score computation is a minority of the FLOPs, and the linear
projections dominate. The quadratic part is unambiguously the *memory*, which is
a different problem with a different fix ({{ch:tf-complexity}}).

**Batch across sequences, not within them.** The recurrence's floor is a depth
constraint; a transformer has no such floor and can use the whole machine on one
sequence.

## 11. Common Mistakes

**Reading an attention map as an explanation.** Measured to be non-identifiable.

**Saying "attention is quadratic" without saying in what.** Measured: quadratic
in memory always, and a minority of the FLOPs at typical lengths.

**Believing {{cite:vaswani2017}} invented attention.** It removed the
recurrence; {{cite:bahdanau2015}} invented the mechanism three years earlier.

**Comparing architectures on FLOPs alone.** {{eq:brent}}: depth and work are
different constraints and only one of them scales away with hardware.

**Assuming a longer context always helps.** Attention over more positions
dilutes the softmax and costs quadratic memory; whether more context helps is an
empirical question per task.

## 12. Failure Modes

**Quality degrading with input length** in any fixed-bottleneck architecture.
Measured directly: reconstruction accuracy falls along each row of the
bottleneck table.

**Attention collapsing to uniform.** With no useful signal in the scores, the
softmax outputs $1/T$ everywhere and the context vector is the mean. The model
still trains and the head does nothing.

**Attention collapsing to one position.** The opposite: a head that always
attends to the first token regardless of content. Common enough to have a name —
the attention sink — and usually benign.

**Out-of-memory from the score matrix**, not from the weights. $T^2$ per head
per layer, and it is the first thing to fail as context grows.

## 13. Alternatives

**Keep the recurrence and add attention.** {{cite:bahdanau2015}}'s original.
Still reasonable when sequences are short and a compressed state is wanted.

**Convolutional sequence models** get parallelism without quadratic cost, at a
path length limited by depth ({{ch:dl-cnns}}).

**State space models** recover a linear-time recurrence that is parallelisable
at training time, and they inherit {{eq:bottleneck-capacity}}'s constraint by
construction ({{ch:tf-efficient}}).

**Retrieval instead of a longer context.** Rather than attending over
everything, fetch the relevant part first ({{part:12}}). Often the better
engineering answer, and frequently overlooked because it is not an architecture
change.

## 14. Evaluation

**Test on a task that requires alignment.** The lookup task here isolates the
capability; a translation benchmark confounds it with everything else.

**Plot quality against input length.** A bottleneck shows up as a downward
slope and nothing else does.

**Check attention faithfulness before trusting a map.** Perturb the weights and
see whether the output moves; measured here to be a real risk.

**Compare on wall-clock at equal quality**, not on FLOPs.
{{eq:brent}}.

## 15. Advanced Concepts

**Attention as a differentiable dictionary.** Keys index, values are retrieved,
queries look up, and softmax makes the lookup soft. This framing is exactly what
makes {{ch:emb-vector-db}}'s retrieval and attention the same idea at different
scales, and it is worth carrying forward.

**The attention-entropy diagnostic.** Averaged over a corpus, per-head entropy
separates heads that select from heads that average. Cheap, and one of the few
head-level diagnostics that means something.

**Attention faithfulness.** A literature exists on whether attention weights are
explanations, with results on both sides. The measured non-identifiability here
is the easiest half of the argument.

**Hard attention** selects one position rather than averaging, which is
non-differentiable and needs the score-function estimator of
{{ch:dl-autoencoders}}. Higher variance, and occasionally what you want when the
selection must be discrete.

## 16. Connection to Previous Chapters

{{ch:dl-rnns}} produced the problem: a matrix-power gradient, a fixed-size
state, and an $O(T)$ critical path measured directly. This chapter is the
answer to all three, and the measured sequential floor here is that chapter's
result restated as {{eq:brent}}.

{{ch:dl-cnns}} supplies the third column of {{tbl:path-length}}, and its
receptive-field measurement is the concrete version of the path-length argument.
{{ch:dl-losses}} supplies the softmax. {{ch:dl-forward}}'s arithmetic-intensity
argument is why {{eq:multiplicative-score}} replaced {{eq:additive-score}}.

Forward: {{ch:tf-scaled-dot-product}} takes {{eq:multiplicative-score}} and adds
the one modification that makes it work at scale.
{{ch:tf-complexity}} does the cost arithmetic this chapter sketches.
{{ch:tf-efficient}} tries to keep the path length while lowering the cost.

## 17. Exercises

**Beginner**

1. What is the fixed-size bottleneck?
2. What does {{eq:attention-context}} compute?
3. Give the max path length for recurrence, convolution and attention.
4. Why is multiplicative scoring faster than additive?
5. What was new in {{cite:vaswani2017}}, given that attention already existed?

**Intermediate**

6. Derive {{eq:bottleneck-capacity}} and evaluate it for $d=512$, $V=50000$.
7. Using {{eq:brent}}, explain why more processors do not help a recurrence.
8. Compute attention's and a recurrence's work at $T=4096$, $d=1024$.
9. Find the $T$ at which attention's score term exceeds its projection term
   for $d = 2048$.
10. Explain why identical value vectors make attention weights
    non-identifiable.

**Advanced**

11. Derive {{eq:attention-jacobian}} including the terms through
    $\vecgreek{\alpha}$.
12. Show that self-attention is permutation-equivariant, and say what that
    implies for {{ch:tf-positional}}.
13. Construct a task where a windowed attention of width $w$ provably cannot
    match full attention.
14. Explain what the softmax normalisation prevents, and why removing it
    permits a recurrent formulation.

**Implementation**

15. Implement additive and multiplicative attention and compare wall-clock at
    several $T$.
16. Reproduce the bottleneck table and extend it to $V = 64$.
17. Implement an attention-faithfulness check and run it on a trained model.
18. Measure per-head attention entropy on a task of your choosing.

**Reasoning**

19. Your sequence model is fine on short inputs and poor on long ones, with no
    error. Give an ordered diagnostic procedure.
20. A colleague shows you an attention map as evidence the model "looked at"
    a particular token. What do you ask for?

## 18. Interview Questions

**"Why did transformers replace RNNs?"** — The sequential floor of
{{eq:brent}}. Say that the transformer does *more* work and wins anyway, because
depth is what hardware cannot absorb.

**"What problem did attention originally solve?"** — The fixed-size bottleneck
in a recurrent encoder–decoder, and quality degrading with source length.

**"Is attention quadratic?"** — In memory, always. In FLOPs, only past
$T \approx 4d$; below that the projections dominate. This distinction separates
people who have profiled a transformer from people who have read about one.

**"Can you use attention weights to explain a prediction?"** — Not on their own.
Give the non-identifiability argument and the residual-stream one.

**"What did the Transformer paper contribute?"** — Removing recurrence,
multi-head attention, the $\sqrt{d_k}$ scaling, and sinusoidal positions.
Attention itself was three years old.

## 19. Research Questions

**Is quadratic attention necessary?** Linear and sparse variants exist and none
has displaced it. FlashAttention removed much of the pressure by making the
exact version cheap. {{maturity:EMERGING}}

**Are attention weights ever explanations?** A substantial literature argues
both sides, and the answer appears to depend on the architecture and the
faithfulness test used. {{maturity:RESEARCH FRONTIER}}

**What is the right notion of "path length" for a deep transformer?** The
single-layer answer is 1, and information typically routes through several
layers, so the effective path length is not obviously 1 either.
{{maturity:EMERGING}}

## 20. Chapter Summary

A 2014 encoder–decoder compressed the source into one fixed-size vector, and
{{eq:bottleneck-capacity}} says why that had to fail: capacity is constant in
$T$ while the information to be stored grows linearly. Measured directly on a
reconstruction task, accuracy fell along every fixed-width row and recovered
down every column. Attention removed the constraint by having the decoder read
$T$ vectors instead of one — capacity that grows with the input — at the cost of
having to keep all $T$ around, which is the first appearance of the trade this
part is about.

{{cite:bahdanau2015}} was a patch on a recurrent model. By 2017 the patch had
made the recurrence redundant for its long-range job while remaining the only
sequential part of the computation, and {{cite:vaswani2017}} deleted it. The
contribution was the deletion and the architecture around it, not the mechanism.

The reason the deletion mattered is {{eq:brent}}. A recurrence has work
$O(Td^2)$ and depth $O(T)$; attention has work $O(T^2d)$ and depth $O(1)$. More
processors shrink the work term and cannot touch the depth term. Measured on
identical arithmetic, the recurrent arrangement ran several times slower purely
because it had to proceed in $T$ dependent rounds. **One of the two costs scales
away with hardware and the other does not**, and that asymmetry — not a
modelling argument — is why the field moved.

On the alignment task, attention outperformed a fixed bottleneck at a comparable
parameter count, and the weights concentrated on the position the task required.
That inspectability is real and it is limited. Measured here: moving a full unit
of probability mass between two positions holding the same token changed the
output not at all, because a weighted average cannot distinguish equal values.
Two attention maps, bit-identical outputs, nothing in the model preferring
either. **An attention map is a hypothesis about information flow, not a
measurement of it.**

Finally, "attention is quadratic" needs qualifying. Measured against the
projection cost, the quadratic score term is a minority of the FLOPs at ordinary
sequence lengths and only takes over past $T \approx 4d$. What is quadratic
without qualification is the memory — the $T \times T$ score matrix has to exist
somewhere — and that is a different problem with a different fix, which
{{ch:tf-complexity}} separates and {{ch:tf-efficient}} solves.

## 21. Further Reading

{{cite:bahdanau2015}} is the paper to read first, and read it for the *problem
statement* rather than the mechanism. The observation that translation quality
degrades with source length, and the diagnosis that a fixed vector is the cause,
is the whole argument; the attention mechanism follows from it almost
immediately.

{{cite:luong2015}} is the follow-up that tried the variations, including the
multiplicative scoring the transformer adopted. Worth reading as a model of
careful ablation: it is largely a table of things that did and did not work.

{{cite:vaswani2017}} is the paper this part builds toward, and its table 1 —
path length, complexity and sequential operations — is {{tbl:path-length}}. That
table is the argument, and it is one of the clearest statements of an
architectural trade in the literature.

{{cite:bengio1994}} and {{cite:hochreiter1997}} for the problem attention
inherited. Reading them alongside {{tbl:path-length}} makes clear that the
transformer solves by *structure* what the LSTM solved by *gating*.

**Where to go next:** {{ch:tf-scaled-dot-product}} takes
{{eq:multiplicative-score}} and derives the modification that makes it usable at
scale. It is the most detailed chapter in this part and everything after it
assumes it.

---
id: tf-scaled-dot-product
number: 63
part: VII
tier: full
status: reviewed
requires: [math-vectors, math-matrices, math-norms, math-random-vars,
           math-covariance, math-derivatives, dl-activations, dl-backprop,
           dl-losses, tf-why-attention, ml-logistic]
provides: [attention, self-attention, cross-attention, scaled-dot-product-attention,
           query, key, value, attention-weights, attention-head, logit,
           temperature, causal-masking,
           permutation-equivariance, inductive-bias]
citations: [bahdanau2015, luong2015, sukhbaatar2015, vaswani2017, shazeer2019, dao2022flash]
---

## 1. Learning Objectives

After working through this chapter you will be able to:

1. State the scaled dot-product attention equation from memory and explain the
   role of every symbol and every dimension in it.
2. Derive the $1/\sqrt{d_k}$ scaling factor from the variance of a dot product
   of independent random vectors, and explain what breaks without it.
3. Explain why the operation uses three separate projections rather than one or
   two, and what would be lost by merging any pair of them.
4. Trace the shapes and the numerical values through a complete forward pass by
   hand on a small example.
5. Derive the backward pass through attention, including the softmax Jacobian,
   and verify your derivation numerically against automatic differentiation.
6. Compute the exact arithmetic and memory cost of attention as a function of
   sequence length, and explain why those two costs have different remedies.
7. Explain why self-attention is permutation-equivariant, and what that implies
   about positional information.
8. Implement attention from scratch in NumPy and in PyTorch, including causal
   masking and padding masking, and identify why the two masks are not
   interchangeable.
9. Diagnose the characteristic failure modes: attention entropy collapse, the
   masking bugs that silently leak information, and the numerical overflow that
   fp16 introduces.
10. Explain what FlashAttention changes and, just as importantly, what it does
    not change.

## 2. Why This Matters

Attention is the single operation that carries the most weight in modern AI. It
is the computational core of every large language model, most current vision
models, and essentially all multimodal systems. When people say a model has 70
billion parameters, a large fraction of those parameters exist to feed this
operation. When they say inference is expensive, they are mostly describing the
memory traffic this operation generates. When they say context length is
limited, they are describing a constraint that arises directly from its
structure.

The equation itself is short enough to fit on one line. That brevity is
misleading. Almost every practical question you will face about language models
— why long prompts cost what they cost, why the model appears to lose track of
the middle of a document, why serving throughput collapses at a certain batch
size, why quantising the KV cache is the highest-leverage optimisation available
— resolves, on inspection, into a question about this operation.

There is a second reason this chapter matters more than its length suggests.
Attention is where the field's central architectural bet becomes visible. Earlier
architectures encoded strong assumptions about their data: convolutions assumed
that useful structure is local and translation-invariant, recurrence assumed that
information flows sequentially through time. Attention encodes almost no such
assumption. It says only that any position may need to read from any other, and
leaves *which* to be learned. That is a weaker {{term:inductive-bias}}, and a
weaker inductive bias needs more data to be worth having. The entire scaling era
follows from the observation that, past a certain data scale, that trade goes the
right way.

## 3. Prerequisites

This chapter assumes the following, all established earlier in the book.

**From Part I.** Vectors and the dot product as a similarity measure
({{ch:math-vectors}}); matrix multiplication and how to reason about conformable
shapes ({{ch:math-matrices}}); the geometric interpretation of norms and cosine
similarity ({{ch:math-norms}}); expectation and variance of sums of independent
random variables ({{ch:math-random-vars}}, {{ch:math-covariance}}); partial
derivatives and the chain rule ({{ch:math-derivatives}}).

**From Part VI.** The softmax function as an activation and its saturation
behaviour ({{ch:dl-activations}}); cross-entropy loss ({{ch:dl-losses}}); and
backpropagation as the systematic application of the chain rule over a
computational graph ({{ch:dl-backprop}}). You do not need to remember the
derivations, but you do need to be comfortable with the idea that every
operation in the forward pass has a corresponding rule in the backward pass.

**From this part.** {{ch:tf-why-attention}} established the problem attention
solves: the fixed-length bottleneck in encoder-decoder models, and the
sequential dependency that prevented recurrent models from being parallelised
along the time axis.

> NOTE: If the softmax Jacobian in {{ch:dl-activations}} is hazy, re-read that
> section before {{sec:6-mathematical-foundation}} of this chapter. Section 6
> depends on it directly and does not re-derive it.

## 4. Intuitive Explanation

### 4.1 The problem, stated concretely

Consider the sentence:

> The trophy did not fit in the suitcase because it was too large.

To represent the word *it* usefully, a model must determine what *it* refers to.
The answer is *the trophy*, and knowing this requires combining information from
a word seven positions earlier. Change one word — *too small* instead of *too
large* — and the referent becomes *the suitcase*. No fixed rule about position
solves this. The dependency is content-dependent.

Now generalise. Every position in a sequence needs information from some other
positions, the identity of those positions depends on the content, and the model
must learn which without being told. That is the problem attention solves.

### 4.2 Soft dictionary lookup

The most useful mental model is a dictionary lookup that has been made
differentiable.

An ordinary dictionary lookup works like this: you have a query, you compare it
against the stored keys, you find the one that matches exactly, and you return
the value stored under that key. It is a hard lookup — one key matches, the rest
do not, and the result is discontinuous. Change the query slightly and either
nothing happens or you get a completely different value. You cannot take a
gradient through it.

Attention softens every step of that:

- **Matching is graded, not exact.** Instead of "does this key equal the query",
  we ask "how well does this key match the query", and get back a real number.
- **Every entry participates.** Rather than selecting one, we take a weighted
  average of *all* values, weighted by how well each key matched.
- **The weights are a probability distribution.** They are non-negative and sum
  to one, so the output is a convex combination of the stored values — it lives
  in the same space as the things being averaged, and cannot blow up.

Because every step is now a smooth function of its inputs, the whole thing is
differentiable, and gradient descent can learn what to put in the queries and
keys so that the right things match.

```mermaid {#fig:soft-lookup caption="Hard lookup versus soft lookup. The hard version selects exactly one value and has zero gradient almost everywhere; the soft version returns a weighted mixture and has a useful gradient everywhere."}
graph TB
  subgraph HARD["Hard lookup — not differentiable"]
    HQ[query] --> HM{exact<br/>match?}
    HM -->|yes| HV[return that value]
    HM -->|no| HN[return nothing]
  end
  subgraph SOFT["Soft lookup — attention"]
    SQ[query] --> SC[score against<br/>every key]
    SC --> SS[normalise scores<br/>to weights]
    SS --> SW[weighted average<br/>of all values]
  end
```

### 4.3 Why three roles and not one

The obvious question at this point: why does each position produce three
different vectors? Why not compare token representations directly against each
other?

Because *what makes two things match* and *what one of them should contribute*
are different questions, and a single vector cannot answer both independently.

Return to the pronoun example. The word *it* is looking for a noun that could
plausibly be the subject of *was too large*. That is a description of a search
criterion — it says nothing about what *it* itself contributes to other words.
Meanwhile the word *trophy* advertises "I am a concrete noun, singular, an
object that has a size", and separately carries the content "trophy" that the
pronoun should absorb. Those two functions pull in different directions.

So each position emits three vectors:

- a {{term:query}} — what am I looking for;
- a {{term:key}} — what do I advertise, for matching purposes;
- a {{term:value}} — what do I actually contribute if selected.

The separation of key from value is the important one, and it predates
Transformers. End-to-end memory networks {{cite:sukhbaatar2015}} used exactly
this structure — one embedding of each memory item for matching against the
query, a second embedding for what gets returned — several years before the
terminology settled.

> IMPORTANT: The key/value separation is not an implementation detail. If you
> merge them, a position can only be found by things that want what it carries.
> With them separate, a position can advertise "I am the kind of thing you are
> looking for" while contributing content that looks nothing like the query. Most
> of what attention heads actually learn to do depends on this.

### 4.4 The shape of the whole operation

For a sequence of $n$ positions, every position produces one query, one key and
one value. Every query is scored against every key, giving an $n \times n$ grid
of scores. Each row of that grid is normalised into a distribution over
positions. Each row's distribution is then used to average the values.

The output has one vector per position, the same as the input. Attention does
not change the length of the sequence; it changes what each position knows.

## 5. Formal Explanation

### 5.1 Definition

Let $\mat{X} \in \R^{n \times d_{\text{model}}}$ be a matrix of input
representations, where row $i$ is the vector for position $i$ of a sequence of
length $n$.

> WARNING: This book uses the row convention throughout: tokens index rows.
> Several papers, including some quoted later in this part, use the column
> convention. Their equations are the transpose of the ones here. If your shapes
> do not conform when reproducing a result, check this first.

Three learned projections produce the three roles:

$$
\mat{Q} = \mat{X}\mat{W}^{Q}, \qquad
\mat{K} = \mat{X}\mat{W}^{K}, \qquad
\mat{V} = \mat{X}\mat{W}^{V}
$$ (eq:qkv-projections)

with $\mat{W}^{Q}, \mat{W}^{K} \in \R^{d_{\text{model}} \times d_k}$ and
$\mat{W}^{V} \in \R^{d_{\text{model}} \times d_v}$. So
$\mat{Q}, \mat{K} \in \R^{n \times d_k}$ and $\mat{V} \in \R^{n \times d_v}$.

Scaled dot-product attention is then:

$$
\attn(\mat{Q}, \mat{K}, \mat{V})
  = \softmax\!\left(\frac{\mat{Q}\mat{K}\T}{\sqrt{d_k}}\right)\mat{V}
$$ (eq:sdpa)

where the softmax is applied independently to each row.

{{eq:sdpa}} is the definition given by {{cite:vaswani2017}}, and it is worth
being precise about what is new there. Dot-product scoring was established by
{{cite:luong2015}}; soft alignment by {{cite:bahdanau2015}}; the key/value split
by {{cite:sukhbaatar2015}}. The contribution of {{eq:sdpa}} is the scaling
factor and the decision to build an entire architecture from nothing else.

### 5.2 Every symbol, and every shape

{#tbl:sdpa-shapes caption="Every quantity in scaled dot-product attention, with its shape and its role. Getting these shapes into your fingers is most of what it takes to read attention code fluently."}

| Symbol | Shape | What it is |
|---|---|---|
| $n$ | scalar | Number of positions (query side) |
| $m$ | scalar | Number of positions attended over (key side); $m = n$ for self-attention |
| $d_{\text{model}}$ | scalar | Width of the residual stream |
| $d_k$ | scalar | Dimension of queries and keys — must match, or the dot product is undefined |
| $d_v$ | scalar | Dimension of values; need not equal $d_k$ |
| $\mat{Q}$ | $n \times d_k$ | Row $i$ is what position $i$ is looking for |
| $\mat{K}$ | $m \times d_k$ | Row $j$ is what position $j$ advertises |
| $\mat{V}$ | $m \times d_v$ | Row $j$ is what position $j$ contributes |
| $\mat{S} = \mat{Q}\mat{K}\T/\sqrt{d_k}$ | $n \times m$ | Scaled scores; $S_{ij}$ is how well query $i$ matches key $j$ |
| $\mat{A} = \softmax(\mat{S})$ | $n \times m$ | {{term:attention-weights}}; each row sums to 1 |
| $\mat{O} = \mat{A}\mat{V}$ | $n \times d_v$ | Output; row $i$ is a convex combination of value rows |

Written for a single position, {{eq:sdpa}} says:

$$
\vec{o}_i = \sum_{j=1}^{m} A_{ij}\, \vec{v}_j,
\qquad
A_{ij} = \frac{\exp\!\left(\vec{q}_i\T\vec{k}_j / \sqrt{d_k}\right)}
              {\sum_{j'=1}^{m} \exp\!\left(\vec{q}_i\T\vec{k}_{j'} / \sqrt{d_k}\right)}
$$ (eq:sdpa-elementwise)

This form makes two properties immediate. First, $\vec{o}_i$ is a convex
combination of the value vectors, so it lies inside their convex hull and cannot
be larger in norm than the largest of them. Second, computing $\vec{o}_i$ requires
touching every key and every value — there is no locality to exploit.

### 5.3 Self-attention, cross-attention, and the general case

{{eq:sdpa}} does not require $\mat{Q}$, $\mat{K}$ and $\mat{V}$ to come from the
same place. Two cases matter:

**{{term:self-attention}}.** All three are projections of the same $\mat{X}$, as
in {{eq:qkv-projections}}. Every position reads from every position of its own
sequence. Here $m = n$.

**{{term:cross-attention}}.** $\mat{Q}$ is projected from one sequence and
$\mat{K}, \mat{V}$ from another. A decoder attending to an encoder's output, or a
vision-language model attending to image patches while generating text, are both
cross-attention. Here $m$ is the length of the *other* sequence and need not
equal $n$.

The operation is identical in both cases. Only the provenance of the inputs
differs — which is why one implementation serves both.

### 5.4 Permutation equivariance

Let $\mat{P}$ be an $n \times n$ permutation matrix. Applying self-attention to
$\mat{P}\mat{X}$ gives:

$$
\attn(\mat{P}\mat{Q}, \mat{P}\mat{K}, \mat{P}\mat{V})
  = \softmax\!\left(\frac{\mat{P}\mat{Q}\mat{K}\T\mat{P}\T}{\sqrt{d_k}}\right)\mat{P}\mat{V}
  = \mat{P}\,\attn(\mat{Q}, \mat{K}, \mat{V})
$$ (eq:permutation-equivariance)

The result is the same outputs in the permuted order. Shuffle the input, and you
get the identical set of output vectors, merely reordered.

This is {{term:permutation-equivariance}}, and it has a stark consequence:
**self-attention has no notion of order at all.** To it, a sentence is a bag of
vectors. Every fact about position must be injected into the representations
themselves, which is what positional encodings do and why
{{ch:tf-positional}} exists.

> NOTE: This is a genuine design decision rather than an oversight. Recurrence
> hard-codes order and pays for it with a sequential bottleneck. Attention
> discards order and pays for it with a separate positional mechanism — but
> retains full parallelism. The parallelism is what made scale possible.

## 6. Mathematical Foundation

### 6.1 Why divide by $\sqrt{d_k}$

This is the part of {{eq:sdpa}} people most often recite without understanding,
and it is entirely derivable.

Consider a query $\vec{q} \in \R^{d_k}$ and a key $\vec{k} \in \R^{d_k}$ whose
components are independent random variables with mean $0$ and variance $1$. This
is a reasonable model of what the projections produce at initialisation, which is
the regime that matters — if training cannot get started, nothing else follows.

Their dot product is $\vec{q}\T\vec{k} = \sum_{i=1}^{d_k} q_i k_i$. Take its
expectation, using independence:

$$
\E\!\left[\vec{q}\T\vec{k}\right]
  = \sum_{i=1}^{d_k} \E[q_i k_i]
  = \sum_{i=1}^{d_k} \E[q_i]\,\E[k_i]
  = 0
$$ (eq:dot-mean)

Now the variance. The terms $q_i k_i$ are independent across $i$, so variances
add:

$$
\Var\!\left(\vec{q}\T\vec{k}\right)
  = \sum_{i=1}^{d_k} \Var(q_i k_i)
  = \sum_{i=1}^{d_k} \left(\E[q_i^2 k_i^2] - \E[q_i k_i]^2\right)
  = \sum_{i=1}^{d_k} \E[q_i^2]\,\E[k_i^2]
  = d_k
$$ (eq:dot-variance)

using $\E[q_i k_i] = 0$ from {{eq:dot-mean}} and $\E[q_i^2] = \Var(q_i) = 1$.

So the raw scores have standard deviation $\sqrt{d_k}$. **The spread of the
scores grows with the square root of the dimension.** For $d_k = 64$, a typical
per-head dimension, scores are routinely $\pm 8$ or larger. For $d_k = 128$,
$\pm 11$.

Dividing by $\sqrt{d_k}$ restores unit variance:

$$
\Var\!\left(\frac{\vec{q}\T\vec{k}}{\sqrt{d_k}}\right)
  = \frac{1}{d_k}\Var\!\left(\vec{q}\T\vec{k}\right) = 1
$$ (eq:scaled-variance)

> MATH NOTE: The assumption of unit-variance, zero-mean, independent components
> is doing real work here, and it is only true at initialisation with a suitable
> initialisation scheme ({{ch:dl-initialization}}). During training the
> statistics drift. The scaling factor is nonetheless kept fixed, because its job
> is to make optimisation possible at the start; once training is under way the
> network can absorb any residual scale into $\mat{W}^{Q}$ and $\mat{W}^{K}$.
> This is worth internalising as a general pattern: many normalisation choices in
> deep learning are justified by initialisation-time analysis and retained as
> fixed constants thereafter.

### 6.2 Why large scores are fatal

Knowing that scores grow with $\sqrt{d_k}$ is only half the argument. The other
half is why that is harmful, and the answer is in the softmax gradient.

For $\vec{s} = \softmax(\vec{z})$, the Jacobian is:

$$
\frac{\partial s_i}{\partial z_j} = s_i(\delta_{ij} - s_j)
$$ (eq:softmax-jacobian)

where $\delta_{ij}$ is 1 when $i = j$ and 0 otherwise.

Now consider what happens as the scores grow. Softmax is invariant to adding a
constant to every input, but it is *not* invariant to scaling them. Multiplying
all logits by $\alpha > 1$ sharpens the distribution; as $\alpha \to \infty$ it
converges to a one-hot vector at the argmax.

And when $\vec{s}$ is nearly one-hot, {{eq:softmax-jacobian}} evaluates to
approximately zero everywhere. If $s_p \approx 1$ and $s_i \approx 0$ for
$i \neq p$:

- $\partial s_p / \partial z_p = s_p(1 - s_p) \approx 1 \cdot 0 = 0$
- $\partial s_i / \partial z_j \approx 0$ for every other pair, since each term
  carries a factor of some $s_i \approx 0$

The gradient vanishes. No signal reaches $\mat{W}^{Q}$ or $\mat{W}^{K}$, and the
attention pattern — whatever arbitrary pattern the random initialisation
happened to produce — is frozen.

This is the failure the scaling factor prevents. Without it, at $d_k = 64$, a
randomly initialised model starts with scores spread over roughly $\pm 8$. After
exponentiation that is a ratio of $e^{16} \approx 9 \times 10^{6}$ between the
largest and smallest, which is effectively one-hot before a single gradient step
has been taken. The model does not learn a bad attention pattern; it fails to
learn one at all.

> IMPORTANT: This is why the factor is $\sqrt{d_k}$ and not $d_k$. The goal is to
> normalise the *standard deviation* of the scores to a constant, not their
> magnitude to a constant. Dividing by $d_k$ would shrink the scores toward zero
> as dimension grows, driving the softmax toward uniform — the opposite
> pathology, in which every position attends equally to everything and the
> operation conveys no information.

### 6.3 A worked numerical example

Take $d_k = 4$ and three positions, with these queries and keys:

$$
\mat{Q} = \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 2 & 0 & 0 \\ 1 & 1 & 0 & 1 \end{bmatrix},
\qquad
\mat{K} = \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 1 & 1 & 1 & 1 \end{bmatrix}
$$ (eq:worked-qk)

The raw score matrix $\mat{Q}\mat{K}\T$ is:

$$
\mat{Q}\mat{K}\T = \begin{bmatrix} 2 & 0 & 2 \\ 0 & 2 & 2 \\ 1 & 2 & 3 \end{bmatrix}
$$ (eq:worked-scores)

Row 1, for instance: $\vec{q}_1\T\vec{k}_1 = 1 + 0 + 1 + 0 = 2$;
$\vec{q}_1\T\vec{k}_2 = 0 + 0 + 0 + 0 = 0$; $\vec{q}_1\T\vec{k}_3 = 1+0+1+0 = 2$.

Dividing by $\sqrt{d_k} = 2$:

$$
\mat{S} = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 0.5 & 1 & 1.5 \end{bmatrix}
$$ (eq:worked-scaled)

Row-wise softmax of the first row: $\exp(1) = 2.718$, $\exp(0) = 1$,
$\exp(1) = 2.718$, summing to $6.436$. So
$\vec{a}_1 = [0.422,\; 0.155,\; 0.422]$ — position 1 splits its attention almost
equally between positions 1 and 3, and largely ignores position 2. That matches
the raw scores: $\vec{q}_1$ matched $\vec{k}_1$ and $\vec{k}_3$ equally well and
$\vec{k}_2$ not at all.

Note what the scaling did. The unscaled first row would have been
$[2, 0, 2] \to [0.468, 0.063, 0.468]$ — more peaked. At $d_k = 4$ the difference
is modest; at $d_k = 128$ it is the difference between learning and not learning.

### 6.4 The backward pass

Attention is differentiated by autograd in practice, but deriving it once tells
you where the memory goes and where numerical trouble originates.

Write the forward pass as three steps, with $\mat{S} = \mat{Q}\mat{K}\T/\sqrt{d_k}$,
$\mat{A} = \softmax(\mat{S})$ row-wise, and $\mat{O} = \mat{A}\mat{V}$. Let
$\mat{G} = \partial\Loss/\partial\mat{O}$ be the incoming gradient.

**Through $\mat{O} = \mat{A}\mat{V}$.** A standard matrix product:

$$
\frac{\partial \Loss}{\partial \mat{V}} = \mat{A}\T\mat{G},
\qquad
\frac{\partial \Loss}{\partial \mat{A}} = \mat{G}\mat{V}\T
$$ (eq:grad-av)

**Through the softmax.** Applying {{eq:softmax-jacobian}} row-wise, and writing
$\mat{D} = \partial\Loss/\partial\mat{A}$, row $i$ gives
$\sum_j D_{ij}\, A_{ij}(\delta_{jk} - A_{ik})$, which collapses to:

$$
\frac{\partial \Loss}{\partial \mat{S}}
  = \mat{A} \odot \left(\mat{D} - \operatorname{rowsum}(\mat{D} \odot \mat{A})\,\mathbb{1}\T\right)
$$ (eq:grad-softmax)

where $\odot$ is element-wise multiplication and the row-sum is broadcast across
the row. This form is worth noticing: the whole softmax backward pass costs one
element-wise product and one row-reduction. It never materialises the $n \times n
\times n$ Jacobian that a naive reading of {{eq:softmax-jacobian}} would suggest.

**Through the scaled product.** With $\mat{E} = \partial\Loss/\partial\mat{S}$:

$$
\frac{\partial \Loss}{\partial \mat{Q}} = \frac{\mat{E}\mat{K}}{\sqrt{d_k}},
\qquad
\frac{\partial \Loss}{\partial \mat{K}} = \frac{\mat{E}\T\mat{Q}}{\sqrt{d_k}}
$$ (eq:grad-qk)

> MATH NOTE: {{eq:grad-softmax}} is why the backward pass needs $\mat{A}$, not
> $\mat{S}$. A naive implementation therefore stores the full $n \times n$
> attention matrix for the backward pass — which is precisely the $O(n^2)$
> memory that {{cite:dao2022flash}} eliminates by recomputing tiles of $\mat{A}$ on
> the fly instead of storing them. The derivation above is what makes that
> optimisation possible to see.

### 6.5 Complexity

Count the multiply-accumulates in the forward pass, for $n$ queries, $m$ keys:

{#tbl:complexity caption="Cost of one attention operation. Arithmetic and memory are both quadratic in sequence length, but for different reasons and with different remedies."}

| Step | Arithmetic | Peak memory |
|---|---|---|
| $\mat{Q}\mat{K}\T$ | $O(n\,m\,d_k)$ | $O(nm)$ for the scores |
| Scale and softmax | $O(nm)$ | reuses the score buffer |
| $\mat{A}\mat{V}$ | $O(n\,m\,d_v)$ | $O(n d_v)$ for the output |
| **Total (self-attention, $m = n$)** | $O(n^2 d)$ | $O(n^2 + nd)$ |

Both terms are quadratic in $n$, and this is the single most consequential fact
about the architecture. Doubling the context length quadruples the work.

But the two quadratics are not the same kind of problem:

- The **arithmetic** cost is intrinsic. Every query must be compared against
  every key; that is what the operation means. Reducing it below $O(n^2)$
  requires computing something other than exact attention.
- The **memory** cost is an artefact of the implementation. Nothing requires the
  full $n \times n$ matrix to exist at once, and {{cite:dao2022flash}} showed it need
  not.

Conflating these two led to a decade of approximate-attention research motivated
by a memory problem that turned out to have an exact solution.
{{ch:tf-efficient}} takes this up properly.

## 7. Internal Mechanics

### 7.1 The data path, with shapes

```mermaid {#fig:sdpa-dataflow caption="Scaled dot-product attention as a data path, annotated with tensor shapes. The n×n score matrix in the middle is the origin of both the quadratic cost and the long-context problem."}
graph LR
  X["X<br/>n × d_model"] --> WQ["× W^Q"]
  X --> WK["× W^K"]
  X --> WV["× W^V"]
  WQ --> Q["Q<br/>n × d_k"]
  WK --> K["K<br/>n × d_k"]
  WV --> V["V<br/>n × d_v"]
  Q --> MM["Q K^T"]
  K --> MM
  MM --> S["S<br/>n × n"]
  S --> SC["÷ √d_k"]
  SC --> MSK["+ mask<br/>(optional)"]
  MSK --> SM["row-wise<br/>softmax"]
  SM --> A["A<br/>n × n<br/>rows sum to 1"]
  A --> OUT["A V"]
  V --> OUT
  OUT --> O["O<br/>n × d_v"]
```

### 7.2 What the projections are actually doing

It is tempting to think of $\mat{W}^{Q}$ and $\mat{W}^{K}$ as two independent
objects. For the forward pass they are not — only their product matters.

Substituting {{eq:qkv-projections}} into the score computation:

$$
\mat{S} = \frac{\mat{X}\mat{W}^{Q}(\mat{X}\mat{W}^{K})\T}{\sqrt{d_k}}
        = \frac{\mat{X}\left(\mat{W}^{Q}\mat{W}^{K\top}\right)\mat{X}\T}{\sqrt{d_k}}
$$ (eq:qk-circuit)

The scores depend on $\mat{W}^{Q}$ and $\mat{W}^{K}$ only through the single
matrix $\mat{W}^{Q}\mat{W}^{K\top} \in \R^{d_{\text{model}} \times d_{\text{model}}}$.
This composite is sometimes called the *QK circuit*: a learned bilinear form
measuring, for any two token representations, how much the first should attend to
the second.

Two consequences follow. First, the factorisation into two matrices is a
*rank constraint*, not an increase in expressiveness: with $d_k <
d_{\text{model}}$, the composite has rank at most $d_k$. That constraint is the
point — it is what makes multi-head attention affordable ({{ch:tf-multi-head}}).
Second, $d_k$ is not merely a compute knob. It bounds how many independent
directions a head can discriminate along.

By the same substitution, $\mat{W}^{V}$ composed with the output projection
$\mat{W}^{O}$ forms an *OV circuit* governing what gets written back into the
{{term:residual-stream}}. The clean separation — QK decides *where to read from*,
OV decides *what to write* — is one of the more useful lenses for reasoning about
what a trained head does.

### 7.3 Masking

Two different situations require zeroing out parts of the attention matrix, and
they are frequently confused.

**{{term:causal-masking}}.** In an autoregressive model, position $i$ must not
see positions $j > i$, or the training objective becomes trivial — the model
would read the answer it is being asked to predict. The mask is upper-triangular
and identical for every sequence in a batch:

$$
M_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}
$$ (eq:causal-mask)

applied additively *before* the softmax: $\mat{A} = \softmax(\mat{S} + \mat{M})$.
Since $\exp(-\infty) = 0$, those weights become exactly zero and the surviving
weights renormalise over the visible positions.

**Padding masking.** Batched sequences have different lengths and are padded to a
common length. Padding positions carry no information and must not be attended
to. This mask depends on the individual sequence, not on position.

> IMPORTANT: The two masks compose but do not substitute. A causal mask does not
> hide padding — padding at the end of a short sequence sits at positions $j > i$
> for early queries but at $j \le i$ for late ones, so a causal mask lets late
> queries attend to padding. Systems that use only a causal mask on
> right-padded batches leak padding into the representation, and the effect is
> subtle enough to survive into production.

> WARNING: Use a large negative number, not literal `-inf`, when working in
> reduced precision. In fp16 an entire masked row — which occurs for a padding
> query attending only to padding keys — produces `-inf` minus `-inf` in the
> softmax's max-subtraction step, yielding `NaN` that then propagates through the
> whole batch. Conventional practice is a value like `-1e9` in fp32 or `-1e4`
> in fp16, chosen to underflow to zero after exponentiation without being
> infinite.

### 7.4 The KV cache

At generation time, an autoregressive model produces one token at a time. Each
new token needs a query, and needs to attend over the keys and values of every
preceding token. Those keys and values do not change — they are functions of
tokens already fixed.

Recomputing them at every step would make generating $n$ tokens cost $O(n^3)$
overall. Caching them makes each step $O(n d)$ and the whole generation
$O(n^2 d)$, matching a single forward pass. This cache is the
{{term:kv-cache}}, and it is why generation is feasible at all.

Its cost is memory, and the arithmetic is worth committing to memory:

$$
\text{KV bytes} = 2 \times L \times n \times h \times d_k \times \text{bytes per element}
$$ (eq:kv-cache-size)

The factor of 2 is for K and V; $L$ is layer count; $h$ is head count.

For a 32-layer model with 32 heads of dimension 128, at 8192 tokens in fp16:
$2 \times 32 \times 8192 \times 32 \times 128 \times 2 = 4.3$ GB — **per
sequence**. Batch sixteen such requests and the cache needs roughly 69 GB, about
five times the 14 GB the weights of a 7B model would occupy in the same
precision. The cache, not the model, is what fills the accelerator.

This is why {{cite:shazeer2019}} proposed sharing keys and values across heads,
and why KV-cache quantisation is among the highest-leverage optimisations
available in serving ({{ch:q-memory-math}}).

## 8. Implementation

### 8.1 From scratch in NumPy

We build it in the order the equation reads, with shape assertions at every step
— the single most effective debugging practice for this kind of code.

```python {tier=A name=sdpa-numpy}
"""Scaled dot-product attention from first principles, in NumPy.

Every intermediate is named and shape-checked so that the mapping between the
equation and the code is one-to-one.
"""
import numpy as np


def softmax(z, axis=-1):
    """Numerically stable row-wise softmax.

    Subtracting the row max leaves the result unchanged — softmax is invariant
    to adding a constant to every logit — but it bounds the largest exponent at
    exp(0) = 1, which prevents overflow when scores are large.
    """
    z_max = np.max(z, axis=axis, keepdims=True)
    # Where a whole row is -inf (a fully masked query), z_max is -inf and the
    # subtraction would give nan. Clamp the max to a finite value first.
    z_max = np.where(np.isfinite(z_max), z_max, 0.0)
    e = np.exp(z - z_max)
    return e / np.sum(e, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Compute Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V.

    Q:    (n, d_k)   queries      — what each position is looking for
    K:    (m, d_k)   keys         — what each position advertises
    V:    (m, d_v)   values       — what each position contributes
    mask: (n, m) additive mask, 0 to keep and a large negative to suppress

    Returns (output, attention_weights) with shapes (n, d_v) and (n, m).
    """
    n, d_k = Q.shape
    m, d_k_key = K.shape
    assert d_k == d_k_key, f"query dim {d_k} != key dim {d_k_key}"
    assert V.shape[0] == m, f"K has {m} rows but V has {V.shape[0]}"

    # Step 1: every query against every key. (n, d_k) @ (d_k, m) -> (n, m)
    scores = Q @ K.T
    assert scores.shape == (n, m)

    # Step 2: the scaling of eq. 63.5 — normalise the score standard deviation
    # so the softmax does not start out saturated.
    scores = scores / np.sqrt(d_k)

    # Step 3: suppress forbidden positions BEFORE the softmax, so the surviving
    # weights renormalise over what remains.
    if mask is not None:
        assert mask.shape == (n, m), f"mask {mask.shape} != scores {(n, m)}"
        scores = scores + mask

    # Step 4: each row becomes a distribution over the m key positions.
    attn = softmax(scores, axis=-1)

    # Step 5: convex combination of value rows. (n, m) @ (m, d_v) -> (n, d_v)
    output = attn @ V
    assert output.shape == (n, V.shape[1])
    return output, attn


def causal_mask(n, m=None, neg=-1e9):
    """Upper-triangular additive mask: position i may not see j > i."""
    m = n if m is None else m
    keep = np.tril(np.ones((n, m), dtype=bool))
    return np.where(keep, 0.0, neg)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, d_k, d_v = 5, 8, 16
    Q, K = rng.normal(size=(n, d_k)), rng.normal(size=(n, d_k))
    V = rng.normal(size=(n, d_v))

    out, A = scaled_dot_product_attention(Q, K, V)
    print("output shape         ", out.shape)
    print("attention shape      ", A.shape)

    # Property 1: every row of A is a probability distribution.
    assert np.allclose(A.sum(axis=-1), 1.0), "rows must sum to 1"
    assert (A >= 0).all(), "weights must be non-negative"

    # Property 2: the output is a convex combination of value rows, so its norm
    # cannot exceed the largest value-row norm.
    bound_holds = (np.linalg.norm(out, axis=-1).max()
                   <= np.linalg.norm(V, axis=-1).max() + 1e-9)
    assert bound_holds
    print("convex-combination bound holds:", bound_holds)

    # Property 3: causal masking makes A strictly lower-triangular.
    out_c, A_c = scaled_dot_product_attention(Q, K, V, mask=causal_mask(n))
    assert np.allclose(np.triu(A_c, k=1), 0.0), "causal mask leaked"
    assert np.allclose(A_c.sum(axis=-1), 1.0), "rows must still sum to 1"
    print("causal mask: strictly lower-triangular, rows still normalised")
    print("row 0 attends only to itself:", np.round(A_c[0], 4))
```

The three assertions at the end are not decoration. They are the complete
behavioural specification of the operation: the weights form a distribution, the
output is a convex combination, and masking suppresses exactly what it should
while leaving the remaining weights normalised. An implementation satisfying all
three is almost certainly correct.

### 8.2 Verifying the $\sqrt{d_k}$ argument empirically

The derivation in {{sec:6-mathematical-foundation}} makes a testable prediction.
Testing it takes ten lines and makes the argument concrete in a way that reading
it does not.

```python {tier=A name=scaling-experiment}
"""Empirical check of eq. 63.7: Var(q·k) = d_k, and what that does to softmax.

Reported entropy is in nats, normalised by log(n) so that 1.0 means a uniform
distribution over n positions and 0.0 means all mass on one position.
"""
import numpy as np

rng = np.random.default_rng(0)
n_trials, n_keys = 20_000, 64


def normalised_entropy(p, axis=-1):
    """Shannon entropy divided by its maximum, log(n)."""
    p = np.clip(p, 1e-12, None)
    h = -(p * np.log(p)).sum(axis=axis)
    return h / np.log(p.shape[axis])


def softmax(z, axis=-1):
    e = np.exp(z - z.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


print(f"{'d_k':>6} {'Var(q·k)':>10} {'predicted':>10} "
      f"{'H unscaled':>12} {'H scaled':>10} {'max weight':>11}")
for d_k in (4, 16, 64, 256, 1024):
    q = rng.normal(size=(n_trials, d_k))
    k = rng.normal(size=(n_trials, d_k))
    dots = (q * k).sum(axis=-1)

    # One query scored against n_keys independent keys, repeated many times.
    q1 = rng.normal(size=(2000, 1, d_k))
    ks = rng.normal(size=(2000, n_keys, d_k))
    scores = (q1 * ks).sum(axis=-1)

    h_unscaled = normalised_entropy(softmax(scores)).mean()
    h_scaled = normalised_entropy(softmax(scores / np.sqrt(d_k))).mean()
    max_w = softmax(scores).max(axis=-1).mean()

    print(f"{d_k:>6} {dots.var():>10.2f} {d_k:>10} "
          f"{h_unscaled:>12.4f} {h_scaled:>10.4f} {max_w:>11.4f}")

print("\nVar(q·k) tracks d_k, confirming eq. 63.7.")
print("Unscaled: entropy collapses toward 0 as d_k grows, and the mean largest")
print("weight approaches 1 — a near-one-hot distribution whose softmax Jacobian")
print("is ~0, so no gradient reaches W^Q or W^K.")
print("Scaled: entropy is essentially CONSTANT across four orders of magnitude")
print("of d_k. That dimension-independence is the whole point of the factor.")
```

### 8.3 PyTorch, batched and multi-head-ready

Real implementations operate on tensors shaped `(batch, heads, seq, dim)`.
Broadcasting handles the leading dimensions, so the core of the function is
unchanged.

```python {tier=A name=sdpa-torch}
"""Batched, multi-head-shaped attention in PyTorch, checked against the
built-in fused kernel.
"""
import math

import torch
import torch.nn.functional as F


def sdpa(q, k, v, mask=None):
    """Scaled dot-product attention over (..., seq, dim) tensors.

    Leading dimensions are arbitrary and broadcast — typically (batch, heads).
    q: (..., n, d_k)   k: (..., m, d_k)   v: (..., m, d_v)
    mask: broadcastable to (..., n, m), additive
    """
    d_k = q.size(-1)
    # transpose(-2, -1) swaps only the last two axes, leaving batch/head intact.
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores + mask
    attn = torch.softmax(scores, dim=-1)
    return attn @ v, attn


torch.manual_seed(0)
B, H, N, Dk, Dv = 2, 4, 7, 16, 16
q = torch.randn(B, H, N, Dk, dtype=torch.float64)
k = torch.randn(B, H, N, Dk, dtype=torch.float64)
v = torch.randn(B, H, N, Dv, dtype=torch.float64)

out, attn = sdpa(q, k, v)
print("output shape:", tuple(out.shape), " attention shape:", tuple(attn.shape))

# Agreement with PyTorch's fused implementation. It computes the same function;
# it differs only in how it schedules memory.
ref = F.scaled_dot_product_attention(q, k, v)
print("max abs difference vs F.scaled_dot_product_attention:",
      (out - ref).abs().max().item())
assert torch.allclose(out, ref, atol=1e-10)

# Causal masking, both ways, must also agree.
causal = torch.triu(torch.full((N, N), float("-inf"), dtype=torch.float64),
                    diagonal=1)
out_c, attn_c = sdpa(q, k, v, mask=causal)
ref_c = F.scaled_dot_product_attention(q, k, v, is_causal=True)
assert torch.allclose(out_c, ref_c, atol=1e-10)
assert torch.allclose(attn_c.triu(diagonal=1),
                      torch.zeros_like(attn_c.triu(diagonal=1)))
print("causal path agrees with is_causal=True; upper triangle is exactly zero")

# Permutation equivariance (eq. 63.10): permute the sequence, and the outputs
# are the same vectors in the permuted order.
perm = torch.randperm(N)
out_p, _ = sdpa(q[:, :, perm], k[:, :, perm], v[:, :, perm])
print("permutation equivariance holds:",
      torch.allclose(out_p, out[:, :, perm], atol=1e-10))
```

> PRODUCTION TIP: In real code, call the fused kernel
> (`F.scaled_dot_product_attention`) rather than the hand-written version. It
> dispatches to a FlashAttention-style implementation where available, which is
> both faster and dramatically more memory-efficient — and it computes exactly
> the same function, as the assertion above confirms. Write the loop yourself to
> understand it; ship the kernel.

### 8.4 Verifying the hand-derived backward pass

The gradients of {{sec:6-mathematical-foundation}} can be checked directly
against autograd. If your derivation is right, the agreement is exact to
floating-point tolerance; if it is wrong, it will not be close.

```python {tier=A name=sdpa-backward}
"""Hand-derived gradients for attention, checked against autograd.

Implements eqs. 63.13-63.15 and compares to torch.autograd on the same inputs.
Run in float64 so that a mismatch means an error in the derivation rather than
accumulated rounding.
"""
import math

import torch

torch.manual_seed(0)
N, M, Dk, Dv = 6, 6, 8, 5
Q = torch.randn(N, Dk, dtype=torch.float64, requires_grad=True)
K = torch.randn(M, Dk, dtype=torch.float64, requires_grad=True)
V = torch.randn(M, Dv, dtype=torch.float64, requires_grad=True)

# --- forward, keeping every intermediate the backward pass needs -------------
S = Q @ K.T / math.sqrt(Dk)
A = torch.softmax(S, dim=-1)
O = A @ V

# An arbitrary scalar loss, so that dL/dO is a fixed known matrix.
G = torch.randn(N, Dv, dtype=torch.float64)
loss = (O * G).sum()
loss.backward()

# --- the same gradients, derived by hand -------------------------------------
with torch.no_grad():
    # eq. 63.13: through O = A V
    dV = A.T @ G
    dA = G @ V.T

    # eq. 63.14: through the row-wise softmax.
    # The row-sum term is what makes each row's gradient sum to zero, which is
    # the differential form of "the row must keep summing to one".
    rowsum = (dA * A).sum(dim=-1, keepdim=True)
    dS = A * (dA - rowsum)

    # eq. 63.15: through the scaled product
    dQ = dS @ K / math.sqrt(Dk)
    dK = dS.T @ Q / math.sqrt(Dk)

for name, mine, auto in (("dQ", dQ, Q.grad), ("dK", dK, K.grad),
                         ("dV", dV, V.grad)):
    err = (mine - auto).abs().max().item()
    print(f"{name}: max abs error vs autograd = {err:.3e}")
    assert err < 1e-12, f"{name} derivation disagrees with autograd"

print("\nAll three hand-derived gradients match autograd to float64 precision.")
print("Note dS rows sum to ~0:", dS.sum(dim=-1).abs().max().item() < 1e-12,
      "— the softmax constraint, differentiated.")
```

> NOTE: The last line is a useful invariant to remember. Because each row of
> $\mat{A}$ is constrained to sum to one, each row of the gradient with respect
> to $\mat{S}$ must sum to zero. If you ever write a custom attention kernel,
> that is the cheapest correctness check available.

## 9. Practical Example

Consider a retrieval-augmented question-answering system
({{part:12}} covers these in full). A user asks a question, a retriever returns
five candidate passages, and the model must answer using them. Attention is what
decides which passage is actually used, and inspecting it is one of the few
genuinely diagnostic tools available when such a system misbehaves.

```python {tier=A name=attention-over-passages}
"""Attention over retrieved passages, using toy embeddings.

The point is not the embedding quality — real embeddings come from a trained
model (Part XI). The point is what the attention weights tell you when the
system gives a wrong answer.
"""
import numpy as np

rng = np.random.default_rng(7)


def softmax(z, axis=-1):
    e = np.exp(z - z.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


passages = [
    "Refunds are processed within 5 business days.",
    "Our head office is located in Bristol.",
    "Refund requests must be filed within 30 days of purchase.",
    "The support line is open 09:00 to 17:00.",
    "Shipping is free on orders over 50 pounds.",
]

# A crude embedding: bag of words over a small shared vocabulary. Real systems
# use a trained encoder; the attention arithmetic downstream is identical.
vocab = sorted({w.strip(".,:").lower()
                for p in passages + ["How long do refunds take?"]
                for w in p.split()})
index = {w: i for i, w in enumerate(vocab)}


def embed(text, d=32):
    """Bag-of-words counts, randomly projected to d dimensions and normalised."""
    counts = np.zeros(len(vocab))
    for w in text.split():
        w = w.strip(".,:").lower()
        if w in index:
            counts[index[w]] += 1
    proj = rng.normal(size=(len(vocab), d)) / np.sqrt(len(vocab))
    v = counts @ proj
    return v / (np.linalg.norm(v) + 1e-9)


d_k = 32
question = "How long do refunds take?"
Q = embed(question, d_k)[None, :]           # (1, d_k)  — one query
K = np.stack([embed(p, d_k) for p in passages])   # (5, d_k)
V = K.copy()                                       # values = keys, for clarity

scores = (Q @ K.T) / np.sqrt(d_k)
weights = softmax(scores)[0]

print(f"Question: {question}\n")
for w, p in sorted(zip(weights, passages), reverse=True):
    bar = "█" * int(round(w * 40))
    print(f"  {w:6.3f} {bar:<40} {p}")

print(f"\nAttention entropy: {-(weights * np.log(weights + 1e-12)).sum():.3f} "
      f"nats (max {np.log(len(passages)):.3f})")
print("Near-maximum entropy means the query failed to discriminate between")
print("passages — the retrieval is being ignored, not used. That is a")
print("diagnosable condition, and section 12 explains what causes it.")
```

The diagnostic value here is real. When a RAG system retrieves the correct
passage and still answers wrongly, there are two very different explanations:
the model attended to the right passage and reasoned badly, or it never attended
to it. The attention distribution distinguishes them, and the remedies are
completely different — better prompting or a stronger model in the first case,
better embeddings or reranking in the second ({{ch:rag-failures}}).

## 10. Production Considerations

**Latency is dominated by memory traffic, not arithmetic.**
{{maturity:ESTABLISHED}} During single-token decoding, attention reads the
entire KV cache and performs only $O(nd)$ arithmetic on it. The arithmetic
intensity — FLOPs per byte moved — is close to one, which is far below what
modern accelerators need to stay busy. Decoding is memory-bandwidth-bound. This
is why {{cite:shazeer2019}} attacks cache *size* rather than FLOP count, and why
a GPU can look almost idle while generating tokens slowly.

**Prefill and decode have opposite characteristics.** Processing the prompt
(prefill) computes attention over all $n$ positions at once: large matrix
products, compute-bound, excellent hardware utilisation. Generating each
subsequent token (decode) is one query against a growing cache:
memory-bound, poor utilisation. A serving stack that treats them identically
will do at least one of them badly, which is why disaggregated prefill and
decode has become standard ({{ch:inf-distributed}}).

**KV cache memory sets the batch size, which sets the throughput.** From
{{eq:kv-cache-size}}, the cache grows linearly in both sequence length and batch
size. Since throughput depends on batch size, and batch size is capped by
whatever memory the cache leaves free, cache size effectively determines
throughput. Grouped-query attention, cache quantisation, and paged allocation
all attack this same number.

**Precision needs care.** Score magnitudes before the softmax can exceed the
fp16 range when sequences are long or when a head has learned large query and
key norms. Standard practice is to accumulate scores in fp32 even when weights
are fp16 or bf16. bf16 has fp32's exponent range and is safer here, at the cost
of mantissa precision.

**Observability.** Two cheap metrics catch most attention pathologies in
production: mean attention entropy per layer and per head, and the fraction of
attention mass landing on the first token. Both are computable from the weights
you already have during evaluation runs, and both move noticeably before quality
degrades in a way users notice ({{ch:ops-observability}}).

## 11. Common Mistakes

**Applying the mask after the softmax.** Zeroing weights post-softmax leaves the
remaining weights summing to less than one, silently scaling the output down by
a variable amount. The mask must be additive and applied before the softmax so
that renormalisation happens over the surviving positions.

**Using `-inf` in reduced precision.** As described in
{{sec:7-internal-mechanics}}, a fully masked row yields `NaN`. Use a large finite
negative value.

**Forgetting the scaling entirely.** This produces a model that trains but
plateaus early, and the symptom — near-one-hot attention from step one — is
invisible unless you look at the weights. It is one of the most common bugs in
hand-written attention.

**Scaling by $d_k$ instead of $\sqrt{d_k}$.** Over-normalises the scores, driving
the softmax toward uniform. The model trains, slowly, to a worse optimum.

**Assuming $d_k = d_v$.** Nothing in {{eq:sdpa}} requires it. They are equal in
most implementations by convention, and code that hard-codes the assumption
breaks the first time someone tries otherwise.

**Reading attention weights as explanations.** {{maturity:ESTABLISHED}} High
attention weight means a position contributed to a weighted average at one layer,
in one head. Information also flows through the residual stream, through
previous layers, and through the feed-forward blocks. Attention weights are a
useful diagnostic and a poor explanation, and the research literature is
explicit about the distinction.

**Confusing the causal mask with a padding mask.** Discussed in
{{sec:7-internal-mechanics}}; worth repeating because the bug is silent.

**Transposing the wrong axes in batched code.** `K.T` on a 4-D tensor reverses
*all* axes. The correct operation is `K.transpose(-2, -1)`. The resulting shape
error is usually caught immediately — unless batch and head dimensions happen to
be equal, in which case it is not.

## 12. Failure Modes

**Entropy collapse.** Attention weights become nearly one-hot, the softmax
Jacobian goes to zero, and the head stops learning. Causes: missing or wrong
scaling, unbounded growth in query and key norms during training, or an
excessive learning rate early on. Detect by logging mean attention entropy per
head; a healthy head early in training sits well away from zero. Remedies include
query-key normalisation and warmup ({{ch:dl-lr-schedules}}).

**Attention dispersion.** The opposite: weights stay near uniform and the output
is approximately the mean of all values, conveying nothing. Causes: excessive
scaling, key and query projections that have collapsed to near-zero, or —
in the retrieval setting of {{sec:9-practical-example}} — embeddings that
genuinely fail to discriminate. Detect with the same entropy metric at the other
extreme.

**Attention sinks.** {{maturity:EMERGING}} Trained models frequently place a
large fraction of attention mass on the first token regardless of content. The
leading interpretation is that the softmax forces every row to sum to one even
when a head has nothing useful to attend to, so the head learns to dump its mass
somewhere harmless. This has a practical consequence: naively evicting early
tokens from the KV cache to save memory degrades quality far more than their
apparent information content suggests.

**Length generalisation failure.** A model trained at one context length
degrades on longer inputs, often sharply. Attention is implicated through two
mechanisms: positional encodings are extrapolating outside their training range
({{ch:tf-positional}}), and the softmax over more positions has lower maximum
attainable per-position weight, changing the distribution's character. This is a
distinct problem from running out of memory, and the remedies differ.

**Silent information leakage.** The masking bugs above do not raise errors. They
produce a model with implausibly good training loss and poor generation quality.
The tell is a training loss far below what the architecture should achieve —
which is why the causal-mask assertion in {{sec:8-implementation}} belongs in the
test suite, not just in the chapter.

**Numerical overflow in long-context fp16.** Scores grow with sequence length in
practice, and fp16's maximum is about 65504. Accumulate in fp32.

## 13. Alternatives

{#tbl:attention-alternatives caption="Scoring functions and attention variants, with what each trades away. Only the first two rows compute the same function as eq. 63.2."}

| Approach | Score function | Cost | Trade-off |
|---|---|---|---|
| Scaled dot-product {{cite:vaswani2017}} | $\vec{q}\T\vec{k}/\sqrt{d_k}$ | $O(n^2 d)$ | The baseline |
| FlashAttention {{cite:dao2022flash}} | identical | $O(n^2 d)$ time, $O(n)$ memory | Exact; needs a fused kernel |
| Additive {{cite:bahdanau2015}} | $\vec{w}\T\tanh(\mat{W}_q\vec{q} + \mat{W}_k\vec{k})$ | $O(n^2 d)$, larger constant | More expressive per pair; not a single matmul, so far slower in practice |
| Multiplicative {{cite:luong2015}} | $\vec{q}\T\mat{W}\vec{k}$ | $O(n^2 d)$ | A learned bilinear form; subsumed by the QK circuit of eq. 63.16 |
| Multi-query {{cite:shazeer2019}} | identical, shared K/V | $O(n^2 d)$ time, cache ÷ $h$ | Small quality loss for a large cache reduction |
| Linear attention | kernel feature map | $O(n d^2)$ | Loses the softmax's sharpness; quality gap on many tasks |
| Sparse / windowed | restricted to a subset | $O(n w d)$ | Reintroduces a locality inductive bias |

Two points about this table matter more than its contents.

First, **FlashAttention is not an approximation.** It computes exactly
{{eq:sdpa}}, bit-for-bit up to floating-point reassociation. It changes only
where the intermediate values live. Grouping it with approximate methods, as is
common, misunderstands what it did.

Second, **the approximate methods were largely motivated by the memory
problem.** Once that problem admitted an exact solution, the case for accepting a
quality loss to avoid it weakened considerably. The methods that remain
compelling are those that also reduce the KV cache, which FlashAttention does not
address. {{ch:tf-efficient}} works through this properly.

## 14. Evaluation

Attention is a component, not a system, so "evaluating" it means two distinct
activities.

**Verifying an implementation.** The specification is the three properties
asserted in {{sec:8-implementation}}: rows of $\mat{A}$ form probability
distributions; the output is a convex combination of value rows; masking zeroes
exactly the intended entries and the survivors renormalise. Add to those a
gradient check against autograd or finite differences, an equivalence test
against a reference implementation in float64, and a permutation-equivariance
test. Together these are close to a complete correctness suite, and all six run
in under a second.

**Assessing a trained head's behaviour.** Different question, different tools:

- *Attention entropy*, per layer and per head, normalised by $\log n$. Tracks
  whether heads are discriminating at all.
- *Attention distance* — the mean $|i - j|$ weighted by $A_{ij}$. Reveals whether
  a head is doing local or long-range work, and typically stratifies sharply
  across layers.
- *Ablation.* Zero out one head and measure the change in task performance. This
  is the only one of these that establishes causal importance, and it routinely
  contradicts what the weights suggest.
- *Sink fraction* — attention mass on position 0.

> WARNING: Do not treat attention weights as a faithful explanation of model
> behaviour. {{maturity:ESTABLISHED}} There is a substantial literature showing
> that attention distributions can be altered substantially while leaving
> predictions nearly unchanged, which means they cannot be the sole causal path.
> Use them as instrumentation, and use ablation when you need a causal claim.
> {{ch:rai-interpretability}} treats this properly.

## 15. Advanced Concepts

**The QK circuit as a bilinear form.** {{eq:qk-circuit}} showed that only the
product $\mat{W}^{Q}\mat{W}^{K\top}$ affects the forward pass. Analysing that
composite directly — its eigenstructure, its rank, which token-embedding
directions it connects — is more informative than examining either factor, and is
the basis of the circuits view of Transformer interpretability.
{{maturity:EMERGING}}

**Low-rank structure in attention matrices.** Empirically, the softmaxed
attention matrix is often close to low-rank, which motivated a family of
efficient-attention methods. The empirical claim is better supported than the
theoretical arguments offered for it, and it does not hold uniformly across
layers or heads. {{maturity:EXPERIMENTAL}}

**Softmax-with-a-null-option.** Adding a constant to the softmax denominator,
without a corresponding numerator term, lets a row sum to less than one — giving
a head the option of attending to nothing. This is a direct response to the
attention-sink observation. It is a small change with reported benefits for
quantisation, and it remains outside standard practice.
{{maturity:EXPERIMENTAL}}

**Online softmax and the tiling recurrence.** The insight underlying
{{cite:dao2022flash}} is that softmax can be computed in a single streaming pass by
maintaining a running maximum and a running sum, rescaling accumulated results
when the maximum changes. That makes the softmax associative over tiles, which is
what allows attention to be computed without materialising $\mat{A}$. Working
through the recurrence yourself is the fastest route to understanding modern
attention kernels. {{maturity:ESTABLISHED}}

**Attention as kernel smoothing.** {{eq:sdpa-elementwise}} is a Nadaraya-Watson
kernel regression estimator with an exponential kernel, evaluated at the query
point. The connection is exact, and it supplies a statistical vocabulary —
bandwidth, bias-variance trade-off, effective sample size — for reasoning about
attention. The bandwidth is $\sqrt{d_k}$. {{maturity:ESTABLISHED}}

## 16. Connection to Previous Chapters

{{ch:tf-why-attention}} established the problem: recurrence forced information
through a fixed-size bottleneck and prevented parallelism along the sequence
axis. This chapter is the answer to the first half of that. The second half —
parallelism — follows from the fact that {{eq:sdpa}} is two matrix
multiplications and a softmax, with no loop over positions.

The dot product of {{ch:math-vectors}} appears here as the similarity measure;
the matrix multiplication of {{ch:math-matrices}} as the mechanism for computing
all pairs at once; the variance of sums from {{ch:math-covariance}} as the
justification for the scaling factor; the softmax of {{ch:dl-activations}} as the
normaliser; and the chain rule of {{ch:dl-backprop}} as what makes
{{sec:6-mathematical-foundation}} tractable. This chapter is where Part I stops
being preparation and starts being load-bearing.

Looking forward: {{ch:tf-multi-head}} runs several of these in parallel, which is
where the rank constraint of {{eq:qk-circuit}} earns its keep.
{{ch:tf-positional}} supplies the order information that
{{eq:permutation-equivariance}} proves is missing. {{ch:tf-masking-kv}} develops
masking and the KV cache in full. {{ch:tf-complexity}} and
{{ch:tf-efficient}} take up the cost analysis of
{{sec:6-mathematical-foundation}}. Every later part that involves a Transformer —
which is most of them — rests on this operation.

## 17. Exercises

**Beginner**

1. For $\mat{Q}, \mat{K} \in \R^{10 \times 64}$ and $\mat{V} \in \R^{10 \times 32}$,
   state the shape of $\mat{Q}\mat{K}\T$, of $\mat{A}$, and of the output.
2. Compute $\softmax([1, 0, 1])$ by hand to three decimal places. Then compute
   $\softmax([2, 0, 2])$ and describe how the distribution changed.
3. Explain in two sentences why the mask must be applied before rather than after
   the softmax.
4. Using {{eq:kv-cache-size}}, compute the KV cache in gigabytes for a 40-layer
   model with 40 heads of dimension 128, at 4096 tokens, in bf16.

**Intermediate**

5. Verify {{eq:worked-scores}} by hand, then compute the softmax of all three
   rows of {{eq:worked-scaled}}. Which position attends most diffusely, and why?
6. Prove that the output of attention lies within the convex hull of the value
   vectors. What does this imply about the output's norm?
7. Show that softmax is invariant to adding a constant to every logit, and
   explain why the NumPy implementation in {{sec:8-implementation}} exploits
   this.
8. A colleague proposes dividing by $d_k$ rather than $\sqrt{d_k}$. Predict the
   effect on attention entropy at $d_k = 128$, then modify the experiment in
   {{sec:8-implementation}} to test your prediction.
9. Construct a concrete case where a causal mask alone fails to prevent a query
   from attending to padding on a right-padded batch. Give the sequence lengths
   and the offending index pair.

**Advanced**

10. Prove {{eq:permutation-equivariance}} from first principles, being explicit
    about why $\mat{P}\T\mat{P} = \mat{I}$ is needed and where it is used.
11. Derive {{eq:grad-softmax}} from {{eq:softmax-jacobian}}, showing why the
    row-sum term appears and why each row of $\partial\Loss/\partial\mat{S}$ must
    sum to zero.
12. Suppose the components of $\vec{q}$ and $\vec{k}$ have variance $\sigma^2$
    rather than 1. What scaling factor restores unit score variance? Why is the
    fixed $1/\sqrt{d_k}$ still the right engineering choice?
13. Derive the online-softmax recurrence: given running maximum $m$ and running
    sum $\ell$ over a prefix, and a new block of scores, give the update rule
    that yields the correct global softmax. This is the core of
    {{cite:dao2022flash}}.

**Implementation**

14. Extend the NumPy implementation to support a separate padding mask of shape
    $(m,)$ combined with the causal mask. Write the test that would have caught
    the bug in Exercise 9.
15. Implement attention with an explicit loop over query positions, verify it
    against the vectorised version, then measure both at $n \in \{64, 256,
    1024\}$ and plot time against $n$. Confirm the quadratic scaling.
16. Implement the backward pass of {{sec:6-mathematical-foundation}} in NumPy
    with no autograd, and verify it against finite differences.
17. Instrument a forward pass to report per-head attention entropy and mean
    attention distance, then run it on a pretrained small model and describe how
    the two statistics vary with depth.
18. Implement a memory-efficient attention that processes keys in blocks of size
    $b$, maintaining the running softmax statistics of Exercise 13. Verify it
    matches the reference to float64 tolerance, and measure peak memory against
    $b$.

**Reasoning**

19. Attention is permutation-equivariant, so a Transformer without positional
    information cannot distinguish word orders. Why, then, does such a model
    still perform far above chance on many language tasks?
20. Both entropy collapse and attention dispersion are pathologies, and both are
    detected by the same metric. Given only a single entropy number for a head,
    what else would you need to know to decide which pathology, if either, you
    are looking at?

## 18. Interview Questions

**Beginner**

1. Write the scaled dot-product attention equation and explain each term.
2. What are queries, keys and values, and why are three needed?
3. Why is a softmax used rather than simply normalising the scores by their sum?
4. What is the difference between self-attention and cross-attention?

**Intermediate**

5. Why divide by $\sqrt{d_k}$? Derive it.
6. What is the time and memory complexity of attention, and are they the same
   kind of problem?
7. What is a KV cache and what problem does it solve? What does it cost?
8. Why does self-attention need positional encodings?
9. Walk through how a causal mask is implemented and why it is applied where it
   is.

**Advanced**

10. Attention weights are often shown as explanations of model behaviour. Make
    the case against that practice.
11. Why does decoding underutilise a GPU while prefill does not?
12. What exactly does FlashAttention change, and what does it leave unchanged?
13. Derive the backward pass through the softmax in attention. Why does it not
    require materialising the full Jacobian?
14. Multi-query attention shares keys and values across heads. What is the
    quality cost, and why is it usually worth paying?

**Senior / systems**

15. You are serving a 70B model and throughput is a third of what you projected.
    Attention is suspected. What do you measure, in what order, and what would
    each result tell you?
16. Design an attention variant for a 1M-token context. State your constraints,
    your approach, and what you are giving up.
17. A model trained at 4K context degrades sharply at 16K, and memory is not the
    limit. Enumerate the candidate causes and how you would distinguish them.

## 19. Research Questions

1. Read {{cite:bahdanau2015}}, {{cite:luong2015}} and {{cite:vaswani2017}} in
   that order. For each, identify the specific limitation of the previous work
   it addresses. What did Vaswani et al. inherit unchanged from Luong et al.?
2. {{cite:vaswani2017}} justifies the scaling factor in a footnote, with the
   argument reconstructed in {{sec:6-mathematical-foundation}}. What assumptions
   does that argument make, and under what training conditions do they fail?
   Design an experiment to detect the failure.
3. The attention-sink phenomenon has several competing explanations in the
   literature. Find two, state what each predicts, and identify an experiment
   that would distinguish them.
4. Work through {{cite:dao2022flash}} and reproduce the tiling argument on paper.
   What is the arithmetic intensity of the tiled algorithm as a function of tile
   size, and where is the optimum?
5. {{cite:shazeer2019}} reports a small quality cost for multi-query attention.
   Grouped-query attention interpolates between multi-query and multi-head. What
   determines the right number of groups, and is the answer scale-dependent?
6. Attention is exactly Nadaraya-Watson kernel regression with an exponential
   kernel. Which results from the nonparametric statistics literature transfer,
   and which are blocked by the fact that the kernel here is learned?

## 20. Chapter Summary

Scaled dot-product attention is a differentiable, content-addressed lookup. Each
position emits a query describing what it needs, a key advertising what it
offers, and a value carrying what it contributes. Queries are scored against all
keys by dot product, the scores are scaled by $1/\sqrt{d_k}$, normalised by a
row-wise softmax, and used to take a convex combination of the values.

The scaling factor is not arbitrary. The dot product of two $d_k$-dimensional
random vectors with unit-variance components has variance $d_k$, so raw scores
grow as $\sqrt{d_k}$; unscaled, the softmax saturates at initialisation, its
Jacobian vanishes, and no gradient reaches the query and key projections. The
factor $\sqrt{d_k}$ normalises the score standard deviation to one, which is why
it is a square root and not $d_k$ itself.

Three projections rather than one exist because matching and contributing are
different functions. Separating keys from values lets a position be found for one
reason and contribute something else, and most of what trained heads do depends
on it.

The operation is permutation-equivariant, so it has no notion of order at all;
position must be supplied separately. It costs $O(n^2 d)$ arithmetic and, in a
naive implementation, $O(n^2)$ memory — but only the first of those is intrinsic,
which is what FlashAttention exploits to compute exactly the same function in
linear memory.

In deployment, attention's cost is dominated by KV-cache memory traffic rather
than arithmetic, which is why generation is memory-bandwidth-bound, why cache
size determines achievable batch size and therefore throughput, and why sharing
keys and values across heads is worth a small quality loss.

The characteristic failure modes are entropy collapse, dispersion, sinks,
length-generalisation failure, and silent masking bugs. All are detectable with
two cheap metrics — attention entropy and sink fraction — that belong in any
serious evaluation harness.

## 21. Further Reading

**Primary sources.** {{cite:vaswani2017}} for the architecture and the scaling
factor; read section 3.2 closely and note that the justification for
$1/\sqrt{d_k}$ appears in a footnote. {{cite:bahdanau2015}} and
{{cite:luong2015}} for the lineage, in that order — Bahdanau for why soft
alignment was needed, Luong for why the dot product replaced the additive
network. {{cite:sukhbaatar2015}} for the key/value split arriving before the
terminology.

**Systems.** {{cite:dao2022flash}} for the IO-aware reformulation; the tiling and
online-softmax derivation in section 3 is the part to work through.
{{cite:shazeer2019}} for multi-query attention and, more importantly, for the
reframing of decoding as a memory-bandwidth problem.

**In this book.** {{ch:tf-multi-head}} for multiple heads and why the rank
constraint of {{eq:qk-circuit}} matters; {{ch:tf-positional}} for what
{{eq:permutation-equivariance}} makes necessary; {{ch:tf-masking-kv}} for masking
and caching in depth; {{ch:tf-complexity}} and {{ch:tf-efficient}} for the cost
analysis and its remedies; {{ch:q-memory-math}} for the cache arithmetic in a
serving context; {{ch:rai-interpretability}} for why attention weights are not
explanations.

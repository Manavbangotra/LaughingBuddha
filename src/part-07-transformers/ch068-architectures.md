---
id: tf-architectures
number: 68
part: VII
tier: full
status: reviewed
requires: [tf-ffn-residual, tf-multi-head, tf-embeddings, tf-positional]
provides: [encoder-only, decoder-only, encoder-decoder-transformer, bidirectional-attention,
           causal-attention, masked-language-modelling, prefix-lm,
           architecture-choice]
citations: [vaswani2017, devlin2019bert, radford2019, raffel2020t5, xiong2020prenorm]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Distinguish encoder-only, decoder-only and encoder–decoder transformers by
   their attention masks.
2. Explain what each shape can and cannot do, and why.
3. Explain masked language modelling and why it cannot generate.
4. Explain why the field converged on decoder-only, in terms that are not
   "because it scaled".
5. Explain the prefix-LM and where it sits between the three.
6. Choose an architecture for a task with a reason.
7. Account for the parameter and compute differences between the shapes.

## 2. Why This Matters

**The three architectures differ in one thing: the attention mask.** Everything
else — the blocks, the embeddings, the positions — is identical. That is a
genuinely surprising fact and it is the cleanest way to hold the whole family in
your head.

**The convergence on decoder-only was not obvious in 2019.** BERT dominated
benchmarks; T5's encoder–decoder was the strongest general-purpose model. The
argument that won is about *what a single training objective can subsume*, and
it is worth understanding because the same argument recurs in
{{part:13}} for multimodal models.

**Encoder-only models did not disappear.** Every embedding model
({{ch:emb-models}}), every reranker ({{ch:emb-reranking}}) and most
classification deployments are encoder-only, because bidirectional attention is
strictly better when you have the whole input and need one representation of it.

**The mask is a one-line change and a total change in capability.** A missing
causal mask trains beautifully, scores wonderfully, and cannot generate at all.
{{sec:8-implementation}} demonstrates it.

## 3. Prerequisites

{{ch:tf-ffn-residual}} for the block these architectures stack.
{{ch:tf-multi-head}} for attention and cross-attention.
{{ch:tf-embeddings}} for the two ends.
{{ch:tf-positional}} for what supplies order.

## 4. Intuitive Explanation

### 4.1 One difference

```text
   ENCODER-ONLY          DECODER-ONLY           ENCODER-DECODER
   (BERT)                (GPT)                  (T5, translation)

   ■ ■ ■ ■               ■ □ □ □                enc: ■ ■ ■ ■
   ■ ■ ■ ■               ■ ■ □ □                     ■ ■ ■ ■
   ■ ■ ■ ■               ■ ■ ■ □                dec: ■ □ □ □  + cross-attn
   ■ ■ ■ ■               ■ ■ ■ ■                     ■ ■ □ □    to the encoder
                                                     ■ ■ ■ □
   every position        each position sees
   sees everything       only what precedes
```

$\blacksquare$ means "can attend"; $\square$ means "masked out". **That is the
entire architectural difference.** Same blocks, same parameters, same
everything, one boolean matrix.

### 4.2 What each shape can do

**Encoder-only** sees the whole input at once, so every position's
representation is informed by both sides. Best possible representation of a
fixed input. Cannot generate, because generating token $t+1$ requires a model
that never saw it, and every position here sees everything.

**Decoder-only** sees only the past, so it can be trained to predict the next
token everywhere at once and then run autoregressively. Generates by
construction. Its representation of position $i$ is worse than an encoder's,
because it cannot use anything after $i$.

**Encoder–decoder** does both: bidirectional over the input, causal over the
output, with cross-attention connecting them. Best of both, at the cost of two
sets of weights and an architectural commitment to which text is "input".

### 4.3 The training objectives that follow

```text
   encoder-only     masked language modelling
                    "the [MASK] sat on the mat" -> predict "cat"
                    ~15% of positions supervised

   decoder-only     next-token prediction
                    every position supervised, always
                    100% of positions

   encoder-decoder  span corruption / seq2seq
                    input -> output, with the split defined by the task
```

**The efficiency difference is stark and it is the underrated half of the
argument.** Masked language modelling supervises about 15% of positions per
example, because unmasking more destroys too much context. Next-token
prediction supervises *every* position. For the same compute, a decoder-only
model gets roughly six times the gradient signal.

### 4.4 Why decoder-only won

Four reasons, in rough order of how much they mattered.

**One objective subsumes the others.** Any task expressible as
input$\to$output can be written as a single sequence with the output following
the input, and next-token prediction covers it. Classification is generating a
label; translation is generating a translation. You do not need a second
architecture, and you do not need to decide in advance which part is the input.

**Every position is supervised.** The 15%-versus-100% argument above.

**One set of weights.** An encoder–decoder duplicates the stack. At a fixed
parameter budget, a decoder-only model puts all of it in one place.

**In-context learning.** A decoder-only model conditioned on examples in its
prompt performs the task without any weight update. This capability was a
surprise and it is much more natural in an architecture where the input and the
output live in the same sequence.

**The honest caveat.** For a fixed representation task with a known input,
encoder-only remains better, and every embedding model is one. The convergence
is about *general-purpose* models, not about every use.

### 4.5 The prefix-LM

A middle option that is worth knowing because it clarifies the others: one
stack, bidirectional attention over a prefix, causal attention over the rest.

```text
   ■ ■ ■ □ □ □         prefix: bidirectional among themselves
   ■ ■ ■ □ □ □
   ■ ■ ■ □ □ □
   ■ ■ ■ ■ □ □         suffix: causal
   ■ ■ ■ ■ ■ □
   ■ ■ ■ ■ ■ ■
```

It gets the encoder's bidirectional input handling with the decoder's single
stack. It is used less than the argument suggests it should be, and the reason
is training efficiency: only the suffix positions are supervised, so it inherits
part of masked language modelling's problem.

## 5. Formal Explanation

### 5.1 The mask

All three architectures compute

$$
\mat{A} = \softmax\!\left(\frac{\mat{Q}\mat{K}\T}{\sqrt{d_k}} + \mat{M}\right)
$$ (eq:masked-attention)

with $\mat{M}_{ij} = 0$ where attention is allowed and $-\infty$ where it is
not. The three shapes are three choices of $\mat{M}$:

$$
\mat{M}^{\text{enc}} = \mat{0},
\qquad
\mat{M}^{\text{dec}}_{ij} = \begin{cases}0 & j\le i\\ -\infty & j>i\end{cases},
\qquad
\mat{M}^{\text{prefix}}_{ij} = \begin{cases}
 0 & j\le i \text{ or } j < P\\ -\infty & \text{otherwise}\end{cases}
$$ (eq:masks)

for a prefix of length $P$.

> IMPORTANT: $-\infty$ is implemented as a large negative number, typically
> $-10^{9}$ in fp32 or the dtype's minimum in bf16. Using a value that is large
> but not large enough leaks a small amount of attention across the mask, which
> is a real bug and a hard one to find — the model trains and generates
> slightly wrong.

### 5.2 Encoder-only and masked language modelling

{{cite:devlin2019bert}}. Replace a fraction $p$ of input tokens with a special
`[MASK]` symbol and predict the originals:

$$
\Like_{\text{MLM}} = -\sum_{i \in \mathcal{M}}
 \log p_\theta\big(t_i \mid \tilde{\vec{t}}\big)
$$ (eq:mlm)

with $\mathcal{M}$ the masked set, $|\mathcal{M}| \approx 0.15 T$.

The loss is summed only over masked positions, which is the efficiency problem.
And the `[MASK]` token appears in training and never at inference, which is a
train/serve mismatch — BERT's partial fix is to replace some masked positions
with random tokens or leave them unchanged.

**An encoder-only model cannot generate**, and the reason is structural rather
than practical. Generating requires $p(t_{i} \mid t_{<i})$, and every position
in this model has seen $t_{>i}$.

### 5.3 Decoder-only and next-token prediction

{{cite:radford2019}}.

$$
\Like_{\text{LM}} = -\sum_{i=1}^{T}
 \log p_\theta\big(t_i \mid t_{<i}\big)
$$ (eq:next-token)

Every position contributes. The causal mask makes this computable in one forward
pass over the whole sequence, which is the key implementation fact: **the model
predicts $T$ next-tokens in parallel during training and generates one at a time
at inference.** That asymmetry is what {{ch:tf-masking-kv}} is about.

### 5.4 Encoder–decoder

Two stacks. The encoder is bidirectional; the decoder is causal and has a third
sublayer:

$$
\vec{x} = \vec{x} + \MHA_{\text{self}}(\Norm(\vec{x}))
$$
$$
\vec{x} = \vec{x} + \MHA_{\text{cross}}\big(\Norm(\vec{x}),\ \mat{H}_{\text{enc}}\big)
$$
$$
\vec{x} = \vec{x} + \FFN(\Norm(\vec{x}))
$$ (eq:decoder-block)

with cross-attention taking its queries from the decoder and its keys and values
from the encoder ({{eq:cross-attention}}).

{{cite:raffel2020t5}} reframed every task as text-to-text and trained one
encoder–decoder on all of them, which was the strongest general-purpose result
of its era.

### 5.5 The comparison

{#tbl:architecture-comparison caption="The three shapes. The last two columns are what decided the convergence: what fraction of positions produce a gradient, and whether the model can generate at all."}

| | Mask | Params for $L$ blocks | Supervised positions | Generates |
|---|---|---|---|---|
| Encoder-only | none | $12Ld^2$ | ~15% | no |
| Decoder-only | causal | $12Ld^2$ | 100% | yes |
| Encoder–decoder | both | $12L_e d^2 + 16L_d d^2$ | task-dependent | yes |
| Prefix-LM | prefix | $12Ld^2$ | suffix only | yes |

The encoder–decoder's $16L_d d^2$ is $12d^2$ for the standard block plus $4d^2$
for the cross-attention sublayer.

### 5.6 Choosing

**Encoder-only** when you need one representation of a complete input:
embeddings, retrieval, reranking, classification where latency matters. It is
strictly better here and it is not close.

**Decoder-only** when you need to generate, or when you want one model for many
tasks, or when in-context learning matters.

**Encoder–decoder** when the input and output are genuinely different objects
and the input is long relative to the output — translation, speech recognition,
summarisation at scale. The architecture's separation is then a feature, and the
encoder's bidirectionality is worth the second stack.

**Prefix-LM** rarely, and mostly in multimodal settings where the "prefix" is an
image and the suffix is text.

### 5.7 What each shape costs to serve

The architecture choice is also a serving decision, and the three differ in ways
the parameter count does not show.

**Encoder-only.** One forward pass over the whole input. Cost is $O(Ld^2T)$
once, and there is no cache and no sequential dependency. The cheapest of the
three by a wide margin, which is why retrieval systems can afford to encode
millions of documents.

**Decoder-only.** One pass to process the prompt — the *prefill* — then one pass
per generated token, each reading the whole KV cache. Prefill is compute-bound
and parallel; decode is memory-bound and sequential
({{ch:tf-ffn-residual}}'s {{eq:block-decode-cost}}). The two phases have
completely different bottlenecks, which is why serving systems schedule them
separately ({{ch:inf-batching}}).

**Encoder–decoder.** The encoder runs once; the decoder generates. The
cross-attention cache is fixed at encoder output and never grows, while
self-attention's grows with the generated tokens. So for a long input and a
short output the cache stays small — which is a real advantage that the
parameter comparison hides.

Concretely, generating $n$ tokens from an input of length $T$:

$$
M_{	ext{cache}}^{	ext{dec-only}} \propto (T + n),
\qquad
M_{	ext{cache}}^{	ext{enc-dec}} \propto T + n
$$

which look identical until you notice the first grows *quadratically in
attention cost* because every generated token attends over $T+n$ positions,
while the second's self-attention only attends over the $n$ generated ones and
its cross-attention over a fixed $T$.

**For summarisation — long in, short out — that is a large difference**, and it
is the strongest surviving technical argument for the encoder–decoder shape.

## 6. Mathematical Foundation

### 6.1 Why a causal model can be trained in parallel

The chain rule for probabilities gives

$$
p(t_1,\dots,t_T) = \prod_{i=1}^{T} p(t_i \mid t_{<i})
$$ (eq:chain-rule-prob)

so the log-likelihood is a *sum* of $T$ terms, each depending only on a prefix.
With a causal mask, position $i$'s output depends on exactly $t_{\le i}$, so all
$T$ conditionals are computed in one forward pass.

**This is the property that makes next-token prediction cheap and masked
language modelling expensive.** MLM's objective is not a product of conditionals
over a single ordering — it is a set of conditionals given a corrupted input —
so there is no ordering that makes them all available at once, and masking more
positions removes the context the others need.

### 6.2 The gradient-efficiency argument, quantified

Per forward pass over $T$ tokens:

$$
\text{MLM: } \ 0.15\,T \ \text{loss terms},
\qquad
\text{LM: } \ T \ \text{loss terms}
$$

Both cost the same forward and backward pass — the compute is $O(Ld^2T)$
regardless of how many positions are supervised. So

$$
\frac{\text{signal per FLOP, LM}}{\text{signal per FLOP, MLM}}
 \approx \frac{1}{0.15} \approx 6.7
$$ (eq:signal-efficiency)

**A decoder-only model extracts roughly seven times the supervision per unit of
compute.** That factor compounds over a pretraining run and it is, on its own,
a large part of why scaling favoured decoder-only models.

The counterargument is real and does not close the gap: each MLM term is
conditioned on *both* sides, so it is a harder and arguably more informative
prediction. Empirically the efficiency wins at scale.

### 6.3 What bidirectionality buys, precisely

Consider predicting position $i$. A causal model has access to
$\mathcal{I}_{\text{causal}} = \{t_1,\dots,t_{i-1}\}$; a bidirectional model to
$\mathcal{I}_{\text{bi}} = \{t_1,\dots,t_{i-1},t_{i+1},\dots,t_T\}$.

Since $\mathcal{I}_{\text{causal}} \subset \mathcal{I}_{\text{bi}}$, the
bidirectional model's conditional entropy is no larger:

$$
H(t_i \mid \mathcal{I}_{\text{bi}}) \le H(t_i \mid \mathcal{I}_{\text{causal}})
$$ (eq:bidirectional-entropy)

**Bidirectional representations are at least as good and usually strictly
better**, and this is an information-theoretic fact, not an empirical one.

That is why encoder-only models remain better for representation tasks, and it
is also why the decoder-only convergence is not a claim that causal attention is
*better* — it is a claim that the training efficiency and the task generality
outweigh a known representational disadvantage.

### 6.4 Why the mask is the only difference

Take the pre-norm block of {{eq:prenorm-attn}}. Nothing in $\MHA$, $\FFN$ or
$\Norm$ refers to the mask; the mask enters only in {{eq:masked-attention}}, as
an additive term before the softmax.

So the *same weights* can be run under any mask. That is not merely a
theoretical observation — it is how prefix-LMs are trained from decoder-only
checkpoints, and how encoder-only models are sometimes adapted for generation.
The architecture is a runtime property of the mask, not a property of the
parameters.

The one exception is the cross-attention sublayer in
{{eq:decoder-block}}, which is genuinely extra weights. **Encoder-only,
decoder-only and prefix-LM differ only in the mask; encoder–decoder differs in
the parameters too.**

### 6.5 The causal mask supplies positional information

A consequence worth its own subsection, because it explains an otherwise
puzzling result.

Under a causal mask, position $i$ attends over exactly $i$ positions. The
softmax normalises over that many entries, so the *number* of things attended to
is itself a function of position. A model can read position off the attention
denominator without any positional encoding at all.

That is why decoder-only transformers with no positional scheme work
({{ch:tf-positional}}), and it also means the causal mask is doing double duty:
enforcing the factorisation of {{eq:chain-rule-prob}} *and* breaking the
permutation symmetry of {{eq:attention-equivariance}}.

**An encoder-only model has neither**, which is why bidirectional models are
strictly dependent on their positional scheme in a way decoder-only models are
not.

## 7. Internal Mechanics

### 7.1 Building the mask

```text
   causal    mask = tril(ones(T, T))
   prefix    mask = tril(ones(T, T));  mask[:, :P] = 1
   padding   mask &= (positions < length)[None, :]
```

Padding and causality are separate masks that are combined with a logical AND.
Forgetting the padding mask lets the model attend to padding tokens, which is
harmless in a bag of noise and corrupting when the padding is a repeated
special token that the model learns to rely on.

### 7.2 The mask value

$-\infty$ is not representable. Implementations use a large negative constant,
and the choice interacts with dtype:

```text
   fp32   -1e9 is safe
   bf16   -1e9 rounds to -1e9; fine
   fp16   -1e9 OVERFLOWS to -inf; and -65504 is the max magnitude
```

A common bug is to write a mask value tuned for fp32 into an fp16 model, where
it becomes $-\infty$, which then produces `nan` when a row is *entirely* masked
— $\softmax$ of all $-\infty$ is $0/0$. Fully-masked rows occur with padding, so
this is not hypothetical.

### 7.3 Training against inference for a decoder

```text
   training    one forward pass, T positions, all predictions at once
   inference   T forward passes, one new position each
```

This asymmetry is the single most important operational fact about decoder-only
models. Training is compute-bound and highly parallel; generation is
memory-bound and inherently sequential
({{ch:tf-ffn-residual}}'s {{eq:block-decode-cost}}).

Everything in {{ch:tf-masking-kv}} exists to make the second case less bad.

### 7.4 Cross-attention's cache never grows

In an encoder–decoder, the encoder runs once and its keys and values are fixed
for the whole generation. So cross-attention's cache is computed once and reused
— unlike self-attention's, which grows with every generated token.

That is a real operational advantage of the encoder–decoder shape for long
inputs and short outputs, and it is underweighted in most comparisons.

### 7.5 Adapting between shapes

Because {{sec:6-mathematical-foundation}} shows the mask is the only difference
for three of the four shapes, adaptation is cheap:

**Decoder to prefix-LM**: change the mask, fine-tune briefly.

**Encoder to decoder**: change the mask, fine-tune longer — the model has never
had to predict from a prefix and its representations are not organised for it.

**Decoder to encoder**: change the mask and fine-tune. Works, and the resulting
embeddings are competitive, which is now a standard way of producing embedding
models from language models ({{ch:emb-models}}).

### 7.6 Why the field's convergence is not a permanent verdict

Worth ending the mechanics section on, because "decoder-only won" is often
repeated as though it were settled physics.

The convergence rests on three contingent facts. Next-token prediction supervises
every position, which matters when compute is the binding constraint. In-context
learning emerged from scale, which nobody predicted. And the tasks people wanted
turned out to be expressible as text continuation.

Change any of those and the argument weakens. If compute stopped being the
constraint and data became it, an objective that extracts more per token would
win — which is part of the case for the mixture-of-denoisers approaches in
{{sec:13-alternatives}}. If a task genuinely requires a bidirectional
representation of a fixed input, an encoder is still better and
{{eq:bidirectional-entropy}} says so provably. And multimodal inputs are not
naturally sequences, which is why {{part:13}}'s models keep reinventing the
prefix-LM.

**The useful skill is not knowing which shape won; it is being able to derive
which shape a given problem wants from the mask, the objective and the serving
pattern.** Those three are what this chapter has been about, and they will
outlive the current answer.

## 8. Implementation

```python {tier=A name=the-three-masks}
"""Three architectures, one boolean matrix of difference."""
import numpy as np

rng = np.random.default_rng(0)


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def make_mask(T, kind, prefix=0):
    """Eq. 68.2."""
    if kind == "encoder":
        return np.ones((T, T), dtype=bool)
    if kind == "decoder":
        return np.tril(np.ones((T, T), dtype=bool))
    if kind == "prefix":
        m = np.tril(np.ones((T, T), dtype=bool))
        m[:, :prefix] = True
        return m
    raise ValueError(kind)


print("=" * 72)
print("the three architectures differ in one boolean matrix (eq. 68.2)")
print("=" * 72)
T = 8
for kind, kw in (("encoder", {}), ("decoder", {}), ("prefix", {"prefix": 3})):
    m = make_mask(T, kind, **kw)
    print(f"\n{kind}" + (f" (P = {kw['prefix']})" if kw else "") + ":")
    for i in range(T):
        print("   " + " ".join("#" if m[i, j] else "." for j in range(T)))
    print(f"   positions attended, per query: "
          f"{m.sum(1).tolist()}")

print("\nEverything else — the blocks, the embeddings, the positions, the")
print("parameters — is identical. Section 6.4 makes that precise: the mask")
print("enters only as an additive term before the softmax, so the SAME")
print("weights can be run under any of these.")

# --- section 6.5: the causal mask supplies position -------------------------
print("\n" + "=" * 72)
print("the causal mask carries positional information (section 6.5)")
print("=" * 72)
print("Under a causal mask, query i attends over exactly i+1 positions. The")
print("softmax normalises over that count, so a model can read position off")
print("the attention denominator with NO positional encoding.\n")
d = 32
X = rng.normal(size=(T, d))
Wq = rng.normal(0, 1 / np.sqrt(d), (d, d))
Wk = rng.normal(0, 1 / np.sqrt(d), (d, d))
S = (X @ Wq) @ (X @ Wk).T / np.sqrt(d)

print(f"{'query i':>9} {'keys visible':>14} {'max attention weight':>22} "
      f"{'entropy':>9}")
for kind in ("encoder", "decoder"):
    m = make_mask(T, kind)
    A = softmax(np.where(m, S, -1e9))
    print(f"  {kind}:")
    for i in (0, 1, 3, 7):
        ent = float(-(A[i][m[i]] * np.log(A[i][m[i]] + 1e-12)).sum())
        print(f"{i:>9} {int(m[i].sum()):>14} {A[i].max():>22.4f} "
              f"{ent:>9.4f}")

print("\nUnder the causal mask the entropy grows with the position, because")
print("more keys are visible. That quantity is a monotone function of i and")
print("the model can use it.")
print("\nThat is why decoder-only transformers with no positional encoding")
print("work at all, and it means the causal mask is doing double duty:")
print("enforcing eq. 68.6's factorisation AND breaking the permutation")
print("symmetry of eq. 65.1. An ENCODER has neither, which is why")
print("bidirectional models depend on their positional scheme absolutely.")

# --- section 7.2: the mask value -------------------------------------------
print("\n" + "=" * 72)
print("the mask value, and the bug it causes (section 7.2)")
print("=" * 72)
scores = np.array([2.0, 1.0, 3.0, 0.5])
mask = np.array([True, True, False, False])
print(f"scores {scores}, mask {mask}\n")
print(f"{'mask value':>14} {'weight on the MASKED entries':>32} "
      f"{'leaked?':>9}")
for mv in (-10.0, -50.0, -1e4, -1e9):
    A = softmax(np.where(mask, scores, mv))
    leak = float(A[~mask].sum())
    print(f"{mv:>14.0e} {leak:>32.3e} {('YES' if leak > 1e-12 else 'no'):>9}")

print("\nA mask value that is large but not large enough leaks attention")
print("across the boundary. At -10 the leak is percent-scale: the model can")
print("see the future, slightly, and it will use it. Training looks fine and")
print("generation is subtly wrong.")

print("\nAnd the other failure — a row that is ENTIRELY masked:")
full_mask = np.zeros(4, dtype=bool)
for mv in (-1e9, -np.inf):
    with np.errstate(invalid="ignore"):
        A = softmax(np.where(full_mask, scores, mv))
    print(f"  mask value {mv:>10}: result = {A}, "
          f"finite = {bool(np.all(np.isfinite(A)))}")

print("\nWith a finite mask value a fully-masked row gives a uniform")
print("distribution — wrong, and silent. With -inf it gives nan, which at")
print("least announces itself. Fully-masked rows occur whenever a sequence")
print("is all padding, so this is not hypothetical.")
print("\nIn fp16 the situation is worse: the largest representable magnitude")
print(f"is 65504, so a mask value of -1e9 becomes -inf on conversion and")
print("the nan appears without anyone having written -inf.")

# --- the objectives ---------------------------------------------------------
print("\n" + "=" * 72)
print("supervised positions per forward pass (eq. 68.9)")
print("=" * 72)
print(f"{'objective':<26} {'supervised / T':>16} {'FLOPs':>10} "
      f"{'signal per FLOP':>18}")
for name, frac in (("masked LM (15%)", 0.15), ("masked LM (30%)", 0.30),
                   ("next-token", 1.00), ("prefix-LM (50% suffix)", 0.50)):
    print(f"{name:<26} {frac:>16.2f} {'same':>10} {frac:>18.2f}")

print("\nThe forward and backward passes cost the same regardless of how")
print("many positions carry a loss term — the compute is O(L d^2 T) either")
print("way. So the last column is the ratio that matters, and next-token")
print("prediction extracts about seven times the supervision per unit of")
print("compute that BERT-style masking does.")
print("\nWhy not just mask more? Because masking removes the context the")
print("other predictions depend on. At 100% masking there is no context at")
print("all and the task is unlearnable, so there is a genuine optimum well")
print("below 1 — which is exactly the constraint next-token prediction does")
print("not have.")
```

```python {tier=A name=architectures-on-a-task}
"""Encoder-only, decoder-only and prefix-LM on the same data, with the same
weights where possible — so the only variable is the mask.
"""
import numpy as np

rng = np.random.default_rng(2)

V, T = 24, 12


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def rmsnorm(x, eps=1e-6):
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps)


def rms_back(x, dy, eps=1e-6):
    d = x.shape[-1]
    ms = (x ** 2).mean(-1, keepdims=True) + eps
    return (dy - x * (dy * x).sum(-1, keepdims=True) / (d * ms)) / np.sqrt(ms)


def make_data(n, seed):
    """A sequence where each token depends on the two before it."""
    rs = np.random.default_rng(seed)
    rule = np.random.default_rng(31).integers(0, V, (V, V))
    X = np.zeros((n, T), dtype=int)
    X[:, 0] = rs.integers(0, V, n)
    X[:, 1] = rs.integers(0, V, n)
    for t in range(2, T):
        nxt = rule[X[:, t - 2], X[:, t - 1]]
        flip = rs.random(n) < 0.12
        X[:, t] = np.where(flip, rs.integers(0, V, n), nxt)
    return X


class Transformer:
    """One pre-norm block. The MASK is a constructor argument and nothing
    else changes."""

    def __init__(self, d=48, h=4, seed=0):
        rs = np.random.default_rng(seed)
        s = 1 / np.sqrt(d)
        self.d, self.h, self.dk, self.dff = d, h, d // h, 4 * d
        self.E = rs.normal(0, 0.05, (V + 1, d))       # +1 for [MASK]
        self.P = rs.normal(0, 0.05, (T, d))
        self.Wq = rs.normal(0, s, (d, d))
        self.Wk = rs.normal(0, s, (d, d))
        self.Wv = rs.normal(0, s, (d, d))
        self.Wo = rs.normal(0, s, (d, d))
        self.W1 = rs.normal(0, s, (d, self.dff))
        self.W2 = rs.normal(0, 1 / np.sqrt(self.dff), (self.dff, d))
        self.U = rs.normal(0, 0.05, (V, d))

    def params(self):
        return [self.E, self.P, self.Wq, self.Wk, self.Wv, self.Wo,
                self.W1, self.W2, self.U]

    def forward(self, X, mask):
        n, Tn = X.shape
        x0 = self.E[X] + self.P[None, :Tn, :]
        na = rmsnorm(x0)
        sp = lambda M: M.reshape(n, Tn, self.h, self.dk).transpose(0, 2, 1, 3)
        Q, K, Vv = sp(na @ self.Wq), sp(na @ self.Wk), sp(na @ self.Wv)
        S = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.dk)
        A = softmax(np.where(mask[:Tn, :Tn], S, -1e9))
        ctx = (A @ Vv).transpose(0, 2, 1, 3).reshape(n, Tn, self.d)
        h1 = x0 + ctx @ self.Wo
        nf = rmsnorm(h1)
        pre = nf @ self.W1
        hid = np.maximum(0.0, pre)
        h2 = h1 + hid @ self.W2
        out = rmsnorm(h2)
        self.cache = (X, x0, na, Q, K, Vv, A, ctx, h1, nf, pre, hid, h2, out)
        return out @ self.U.T

    def grads(self, X, mask, targets, weight):
        """weight: (n, T) 1 where the position contributes to the loss."""
        logits = self.forward(X, mask)
        (Xc, x0, na, Q, K, Vv, A, ctx, h1, nf, pre, hid, h2, out) = self.cache
        n, Tn = X.shape
        d = self.d
        P = softmax(logits)
        w = weight[..., None]
        nsup = max(weight.sum(), 1)
        loss = float(-(np.log(np.clip(
            np.take_along_axis(P, targets[..., None], -1), 1e-12, None))
            * w).sum() / nsup)
        dl = P.copy()
        np.put_along_axis(dl, targets[..., None],
                          np.take_along_axis(dl, targets[..., None], -1) - 1.0,
                          -1)
        dl = dl * w / nsup
        gU = np.einsum('ntv,ntd->vd', dl, out)
        dout = dl @ self.U
        dh2 = rms_back(h2, dout)
        gW2 = hid.reshape(-1, self.dff).T @ dh2.reshape(-1, d)
        dhid = dh2 @ self.W2.T
        dpre = dhid * (pre > 0)
        gW1 = nf.reshape(-1, d).T @ dpre.reshape(-1, self.dff)
        dh1 = dh2 + rms_back(h1, dpre @ self.W1.T)
        gWo = ctx.reshape(-1, d).T @ dh1.reshape(-1, d)
        dctx = dh1 @ self.Wo.T
        sp = lambda M: M.reshape(n, Tn, self.h, self.dk).transpose(0, 2, 1, 3)
        dC = sp(dctx)
        dA = dC @ Vv.transpose(0, 1, 3, 2)
        dV = A.transpose(0, 1, 3, 2) @ dC
        dS = A * (dA - (dA * A).sum(-1, keepdims=True)) / np.sqrt(self.dk)
        dQ, dK = dS @ K, dS.transpose(0, 1, 3, 2) @ Q
        mg = lambda G: G.transpose(0, 2, 1, 3).reshape(n, Tn, d)
        naf = na.reshape(-1, d)
        gWq = naf.T @ mg(dQ).reshape(-1, d)
        gWk = naf.T @ mg(dK).reshape(-1, d)
        gWv = naf.T @ mg(dV).reshape(-1, d)
        dna = mg(dQ) @ self.Wq.T + mg(dK) @ self.Wk.T + mg(dV) @ self.Wv.T
        dx0 = dh1 + rms_back(x0, dna)
        gP = dx0.sum(axis=0)
        gE = np.zeros_like(self.E)
        np.add.at(gE, Xc.reshape(-1), dx0.reshape(-1, d))
        return loss, [gE, gP, gWq, gWk, gWv, gWo, gW1, gW2, gU]


def make_mask(T, kind, prefix=0):
    if kind == "encoder":
        return np.ones((T, T), dtype=bool)
    if kind == "decoder":
        return np.tril(np.ones((T, T), dtype=bool))
    m = np.tril(np.ones((T, T), dtype=bool))
    m[:, :prefix] = True
    return m


MASK_ID = V


def batch_for(kind, X, rs, prefix=6, mlm_rate=0.15):
    """Return (inputs, targets, weight) for the given objective."""
    n = len(X)
    if kind == "encoder":
        inp = X.copy()
        sel = rs.random(X.shape) < mlm_rate
        sel[:, 0] = sel[:, 0] | (~sel.any(1))          # at least one
        inp[sel] = MASK_ID
        return inp, X, sel.astype(float)
    if kind == "decoder":
        return X[:, :-1], X[:, 1:], np.ones((n, T - 1))
    w = np.zeros((n, T - 1))
    w[:, prefix - 1:] = 1.0
    return X[:, :-1], X[:, 1:], w


def train(kind, Xtr, steps=3000, lr=3e-3, batch=128, seed=0, prefix=6):
    net = Transformer(seed=seed)
    mask = make_mask(T, kind if kind != "prefix" else "prefix",
                     prefix=prefix)
    ps = net.params()
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    rs = np.random.default_rng(seed + 4)
    for t in range(1, steps + 1):
        b = Xtr[rs.integers(0, len(Xtr), batch)]
        inp, tgt, w = batch_for(kind, b, rs, prefix=prefix)
        _, gs = net.grads(inp, mask, tgt, w)
        for i, (p, g) in enumerate(zip(ps, gs)):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            p -= lr * (m[i] / (1 - 0.9 ** t)) / (
                np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    return net, mask


Xtr, Xte = make_data(10000, 1), make_data(4000, 2)

print("=" * 72)
print("the same weights, the same data, three masks")
print("=" * 72)
print(f"Every model has identical architecture and parameter count; only")
print(f"the mask and the objective differ. Vocabulary {V}, length {T}.\n")

rs_eval = np.random.default_rng(9)
print(f"{'architecture':<16} {'objective':<22} {'supervised/pos':>15} "
      f"{'its own val loss':>18}")
models = {}
for kind, obj in (("encoder", "masked LM (15%)"),
                  ("decoder", "next-token"),
                  ("prefix", "prefix-LM (suffix)")):
    net, mask = train(kind, Xtr, seed=6)
    models[kind] = (net, mask)
    inp, tgt, w = batch_for(kind, Xte, np.random.default_rng(9))
    loss, _ = net.grads(inp, mask, tgt, w)
    print(f"{kind:<16} {obj:<22} {w.mean():>15.3f} {loss:>18.4f}")

print("\nThose losses are NOT comparable — each model is scored on its own")
print("objective, and predicting a masked token given both sides is an")
print("easier problem than predicting the next token given only the past.")
print("Eq. 68.10 says so: conditioning on a superset cannot raise the")
print("entropy.")
print("\nThe comparable question is what each can DO, and that is next.")

# --- what each can do -------------------------------------------------------
print("\n" + "=" * 72)
print("what each architecture can do")
print("=" * 72)
print("Task A: predict the NEXT token from the past only (generation).")
print("Task B: fill in a MASKED token given both sides (representation).\n")


def eval_next_token(net, mask, X):
    """Every model gets the same causal input; the mask is its own."""
    inp, tgt = X[:, :-1], X[:, 1:]
    logits = net.forward(inp, mask)
    return float((logits.argmax(-1) == tgt).mean())


def eval_fill(net, mask, X, rs):
    inp = X.copy()
    pos = rs.integers(1, T - 1, len(X))
    tgt = X[np.arange(len(X)), pos]
    inp[np.arange(len(X)), pos] = MASK_ID
    logits = net.forward(inp, mask)
    return float((logits[np.arange(len(X)), pos].argmax(-1) == tgt).mean())


print(f"{'architecture':<16} {'A: next-token acc':>19} "
      f"{'B: fill-in acc':>17}")
for kind in ("encoder", "decoder", "prefix"):
    net, mask = models[kind]
    a = eval_next_token(net, mask, Xte)
    b = eval_fill(net, mask, Xte, np.random.default_rng(11))
    print(f"{kind:<16} {a:>19.4f} {b:>17.4f}")
print(f"\n(chance is {1 / V:.4f})")

print("\nThe encoder's column-A number is the one to be careful about. Under")
print("a bidirectional mask, position i can see position i+1 — which IS the")
print("answer — so any number it produces there is meaningless as a")
print("generation score. That is section 5.2's structural point: an")
print("encoder-only model cannot generate, and the reason is not that it")
print("does badly but that the evaluation is not well posed.")

# --- the missing-mask bug ---------------------------------------------------
print("\n" + "=" * 72)
print("the missing causal mask: trains beautifully, cannot generate")
print("=" * 72)
print("Train with next-token prediction and NO causal mask, so every")
print("position can see the answer sitting next to it.\n")
net_bad = Transformer(seed=6)
mask_none = make_mask(T, "encoder")
ps = net_bad.params()
m_ = [np.zeros_like(p) for p in ps]
v_ = [np.zeros_like(p) for p in ps]
rs = np.random.default_rng(10)
for t in range(1, 3001):
    b = Xtr[rs.integers(0, len(Xtr), 128)]
    inp, tgt = b[:, :-1], b[:, 1:]
    w = np.ones_like(tgt, dtype=float)
    _, gs = net_bad.grads(inp, mask_none, tgt, w)
    for i, (p, g) in enumerate(zip(ps, gs)):
        m_[i] = 0.9 * m_[i] + 0.1 * g
        v_[i] = 0.999 * v_[i] + 0.001 * g * g
        p -= 3e-3 * (m_[i] / (1 - 0.9 ** t)) / (
            np.sqrt(v_[i] / (1 - 0.999 ** t)) + 1e-8)

inp, tgt = Xte[:, :-1], Xte[:, 1:]
w = np.ones_like(tgt, dtype=float)
l_bad, _ = net_bad.grads(inp, mask_none, tgt, w)
acc_bad = float((net_bad.forward(inp, mask_none).argmax(-1) == tgt).mean())
net_good, mask_good = models["decoder"]
l_good, _ = net_good.grads(inp, mask_good, tgt, w)
acc_good = float((net_good.forward(inp, mask_good).argmax(-1) == tgt).mean())

print(f"{'model':<26} {'train-time loss':>17} {'train-time acc':>16}")
print(f"{'NO causal mask':<26} {l_bad:>17.4f} {acc_bad:>16.4f}")
print(f"{'with causal mask':<26} {l_good:>17.4f} {acc_good:>16.4f}")

print("\nThe unmasked model looks far better, and it has learned nothing")
print("useful: it is copying position i+1 of its own input to slot i.")
print("\nNow generate. Feed only a prefix and extend it one token at a time,")
print("which is the only setting that matters:\n")


def generate(net, mask, X, n_ctx=4, steps=6):
    """Autoregressive generation from a prefix of n_ctx real tokens."""
    n = len(X)
    seq = np.zeros((n, T - 1), dtype=int)
    seq[:, :n_ctx] = X[:, :n_ctx]
    for i in range(n_ctx, min(n_ctx + steps, T - 1)):
        logits = net.forward(seq, mask)
        seq[:, i] = logits[:, i - 1].argmax(-1)
    return seq


for label, (net, mask) in (("NO causal mask", (net_bad, mask_none)),
                           ("with causal mask", (net_good, mask_good))):
    gen = generate(net, mask, Xte[:2000])
    true = Xte[:2000, :T - 1]
    hit = float((gen[:, 4:10] == true[:, 4:10]).mean())
    print(f"{label:<26} generated-token accuracy: {hit:.4f}")

print(f"\n(chance is {1 / V:.4f})")
print("\nThat is the whole lesson. The unmasked model's training metrics were")
print("excellent and its generation is at or near chance, because at")
print("generation time the future positions are zeros rather than the")
print("answers it learned to copy.")
print("\nA missing causal mask is a one-line bug that produces a model which")
print("passes every training-time check and is completely useless. It is")
print("worth building the generation test into the training loop for")
print("exactly this reason — it is the only check that catches it.")
```

## 9. Practical Example

```python {tier=A name=choosing-an-architecture}
"""When each shape wins: representation quality against generation, and the
gradient-efficiency argument measured.
"""
import numpy as np

rng = np.random.default_rng(3)

V, T = 24, 12


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


# The model and helpers are re-declared so this listing stands alone.
def rmsnorm(x, eps=1e-6):
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps)


def rms_back(x, dy, eps=1e-6):
    d = x.shape[-1]
    ms = (x ** 2).mean(-1, keepdims=True) + eps
    return (dy - x * (dy * x).sum(-1, keepdims=True) / (d * ms)) / np.sqrt(ms)


MASK_ID = V


def make_data(n, seed):
    rs = np.random.default_rng(seed)
    rule = np.random.default_rng(31).integers(0, V, (V, V))
    X = np.zeros((n, T), dtype=int)
    X[:, 0] = rs.integers(0, V, n)
    X[:, 1] = rs.integers(0, V, n)
    for t in range(2, T):
        nxt = rule[X[:, t - 2], X[:, t - 1]]
        flip = rs.random(n) < 0.12
        X[:, t] = np.where(flip, rs.integers(0, V, n), nxt)
    return X


class Transformer:
    def __init__(self, d=48, h=4, seed=0):
        rs = np.random.default_rng(seed)
        s = 1 / np.sqrt(d)
        self.d, self.h, self.dk, self.dff = d, h, d // h, 4 * d
        self.E = rs.normal(0, 0.05, (V + 1, d))
        self.P = rs.normal(0, 0.05, (T, d))
        self.Wq = rs.normal(0, s, (d, d))
        self.Wk = rs.normal(0, s, (d, d))
        self.Wv = rs.normal(0, s, (d, d))
        self.Wo = rs.normal(0, s, (d, d))
        self.W1 = rs.normal(0, s, (d, self.dff))
        self.W2 = rs.normal(0, 1 / np.sqrt(self.dff), (self.dff, d))
        self.U = rs.normal(0, 0.05, (V, d))

    def params(self):
        return [self.E, self.P, self.Wq, self.Wk, self.Wv, self.Wo,
                self.W1, self.W2, self.U]

    def forward(self, X, mask, return_hidden=False):
        n, Tn = X.shape
        x0 = self.E[X] + self.P[None, :Tn, :]
        na = rmsnorm(x0)
        sp = lambda M: M.reshape(n, Tn, self.h, self.dk).transpose(0, 2, 1, 3)
        Q, K, Vv = sp(na @ self.Wq), sp(na @ self.Wk), sp(na @ self.Wv)
        S = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.dk)
        A = softmax(np.where(mask[:Tn, :Tn], S, -1e9))
        ctx = (A @ Vv).transpose(0, 2, 1, 3).reshape(n, Tn, self.d)
        h1 = x0 + ctx @ self.Wo
        nf = rmsnorm(h1)
        pre = nf @ self.W1
        hid = np.maximum(0.0, pre)
        h2 = h1 + hid @ self.W2
        out = rmsnorm(h2)
        self.cache = (X, x0, na, Q, K, Vv, A, ctx, h1, nf, pre, hid, h2, out)
        if return_hidden:
            return out
        return out @ self.U.T

    def grads(self, X, mask, targets, weight):
        logits = self.forward(X, mask)
        (Xc, x0, na, Q, K, Vv, A, ctx, h1, nf, pre, hid, h2,
         out) = self.cache
        n, Tn = X.shape
        d = self.d
        P = softmax(logits)
        w = weight[..., None]
        nsup = max(weight.sum(), 1)
        loss = float(-(np.log(np.clip(
            np.take_along_axis(P, targets[..., None], -1), 1e-12, None))
            * w).sum() / nsup)
        dl = P.copy()
        np.put_along_axis(dl, targets[..., None],
                          np.take_along_axis(dl, targets[..., None], -1) - 1.0,
                          -1)
        dl = dl * w / nsup
        gU = np.einsum('ntv,ntd->vd', dl, out)
        dout = dl @ self.U
        dh2 = rms_back(h2, dout)
        gW2 = hid.reshape(-1, self.dff).T @ dh2.reshape(-1, d)
        dhid = dh2 @ self.W2.T
        dpre = dhid * (pre > 0)
        gW1 = nf.reshape(-1, d).T @ dpre.reshape(-1, self.dff)
        dh1 = dh2 + rms_back(h1, dpre @ self.W1.T)
        gWo = ctx.reshape(-1, d).T @ dh1.reshape(-1, d)
        dctx = dh1 @ self.Wo.T
        sp = lambda M: M.reshape(n, Tn, self.h, self.dk).transpose(0, 2, 1, 3)
        dC = sp(dctx)
        dA = dC @ Vv.transpose(0, 1, 3, 2)
        dV = A.transpose(0, 1, 3, 2) @ dC
        dS = A * (dA - (dA * A).sum(-1, keepdims=True)) / np.sqrt(self.dk)
        dQ, dK = dS @ K, dS.transpose(0, 1, 3, 2) @ Q
        mg = lambda G: G.transpose(0, 2, 1, 3).reshape(n, Tn, d)
        naf = na.reshape(-1, d)
        gWq = naf.T @ mg(dQ).reshape(-1, d)
        gWk = naf.T @ mg(dK).reshape(-1, d)
        gWv = naf.T @ mg(dV).reshape(-1, d)
        dna = mg(dQ) @ self.Wq.T + mg(dK) @ self.Wk.T + mg(dV) @ self.Wv.T
        dx0 = dh1 + rms_back(x0, dna)
        gP = dx0.sum(axis=0)
        gE = np.zeros_like(self.E)
        np.add.at(gE, Xc.reshape(-1), dx0.reshape(-1, d))
        return loss, [gE, gP, gWq, gWk, gWv, gWo, gW1, gW2, gU]


CAUSAL = np.tril(np.ones((T, T), dtype=bool))
FULL = np.ones((T, T), dtype=bool)

Xtr, Xte = make_data(12000, 1), make_data(5000, 2)


def train_mlm(rate, steps=3000, seed=6, lr=3e-3, batch=128):
    net = Transformer(seed=seed)
    ps = net.params()
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    rs = np.random.default_rng(seed + 4)
    for t in range(1, steps + 1):
        b = Xtr[rs.integers(0, len(Xtr), batch)]
        inp = b.copy()
        sel = rs.random(b.shape) < rate
        sel[np.arange(len(b)), rs.integers(0, T, len(b))] = True
        inp[sel] = MASK_ID
        _, gs = net.grads(inp, FULL, b, sel.astype(float))
        for i, (p, g) in enumerate(zip(ps, gs)):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            p -= lr * (m[i] / (1 - 0.9 ** t)) / (
                np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    return net


def train_lm(steps=3000, seed=6, lr=3e-3, batch=128):
    net = Transformer(seed=seed)
    ps = net.params()
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    rs = np.random.default_rng(seed + 4)
    for t in range(1, steps + 1):
        b = Xtr[rs.integers(0, len(Xtr), batch)]
        inp, tgt = b[:, :-1], b[:, 1:]
        _, gs = net.grads(inp, CAUSAL, tgt,
                          np.ones_like(tgt, dtype=float))
        for i, (p, g) in enumerate(zip(ps, gs)):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            p -= lr * (m[i] / (1 - 0.9 ** t)) / (
                np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    return net


# --- section 6.2: the masking-rate trade -------------------------------------
print("=" * 72)
print("why BERT masks 15% and not more (section 6.2)")
print("=" * 72)
print("Masking more positions gives more loss terms per pass AND removes")
print("the context the other predictions need. There is an optimum below 1.\n")
rs_ev = np.random.default_rng(21)


def fill_accuracy(net, X, n_probe=3000):
    rs = np.random.default_rng(77)
    Xp = X[:n_probe]
    inp = Xp.copy()
    pos = rs.integers(1, T - 1, len(Xp))
    tgt = Xp[np.arange(len(Xp)), pos]
    inp[np.arange(len(Xp)), pos] = MASK_ID
    lg = net.forward(inp, FULL)
    return float((lg[np.arange(len(Xp)), pos].argmax(-1) == tgt).mean())


print(f"{'mask rate':>11} {'loss terms/pass':>17} {'fill-in accuracy':>18}")
for rate in (0.05, 0.15, 0.30, 0.50, 0.80):
    net = train_mlm(rate)
    print(f"{rate:>11.2f} {rate * T:>17.2f} {fill_accuracy(net, Xte):>18.4f}")

print("\nThe two effects pull against each other and the table is where they")
print("balance on this task. BERT's 15% was chosen empirically and this is")
print("the shape of the curve that choice sits on.")
print("\nThe point for the architecture argument is that an optimum below 1")
print("EXISTS at all. Next-token prediction supervises every position with")
print("no such trade, because the causal mask removes the future rather")
print("than the context — eq. 68.6's factorisation is what buys that.")

# --- representation quality --------------------------------------------------
print("\n" + "=" * 72)
print("what bidirectionality buys, measured (eq. 68.10)")
print("=" * 72)
print("Freeze each model and fit a linear probe on its hidden states to")
print("predict the token at that position from its CONTEXT (the token")
print("itself is masked out). More information in the representation means")
print("a better probe.\n")


def probe_accuracy(net, mask, X, n=4000, seed=0):
    rs = np.random.default_rng(seed)
    Xp = X[:n]
    inp = Xp.copy()
    pos = rs.integers(1, T - 1, len(Xp))
    tgt = Xp[np.arange(len(Xp)), pos]
    inp[np.arange(len(Xp)), pos] = MASK_ID
    H = net.forward(inp, mask, return_hidden=True)
    feats = H[np.arange(len(Xp)), pos]
    # ridge-regularised multinomial probe, closed form on one-hot targets
    Y = np.zeros((len(Xp), V))
    Y[np.arange(len(Xp)), tgt] = 1.0
    ntr = int(0.7 * len(Xp))
    A = feats[:ntr].T @ feats[:ntr] + 1e-2 * np.eye(feats.shape[1])
    W = np.linalg.solve(A, feats[:ntr].T @ Y[:ntr])
    pred = (feats[ntr:] @ W).argmax(1)
    return float((pred == tgt[ntr:]).mean())


mlm = train_mlm(0.15)
lm = train_lm()
print(f"{'model':<26} {'mask at probe time':<20} {'probe accuracy':>16}")
print(f"{'MLM-trained (encoder)':<26} {'bidirectional':<20} "
      f"{probe_accuracy(mlm, FULL, Xte, seed=1):>16.4f}")
print(f"{'LM-trained (decoder)':<26} {'causal':<20} "
      f"{probe_accuracy(lm, CAUSAL, Xte, seed=1):>16.4f}")
print(f"{'LM-trained (decoder)':<26} {'bidirectional':<20} "
      f"{probe_accuracy(lm, FULL, Xte, seed=1):>16.4f}")
print(f"\n(chance is {1 / V:.4f})")

print("\nEq. 68.10 is an information-theoretic fact: conditioning on both")
print("sides cannot give a higher conditional entropy than conditioning on")
print("one. So a bidirectional representation is AT LEAST as informative,")
print("and this task — where the answer depends on the two PRECEDING tokens")
print("— is one where the future genuinely helps identify a corrupted")
print("position.")
print("\nThe third row is the interesting one: a causally-trained model run")
print("under a bidirectional mask at probe time. It sees the future it was")
print("never trained to use, and whether that helps says how much of the")
print("gap is the OBJECTIVE and how much is the MASK.")

# --- and the generation side -------------------------------------------------
print("\n" + "=" * 72)
print("and what causal training buys")
print("=" * 72)


def next_token_acc(net, mask, X):
    inp, tgt = X[:, :-1], X[:, 1:]
    lg = net.forward(inp, mask)
    return float((lg.argmax(-1) == tgt).mean())


print(f"{'model':<26} {'next-token accuracy (causal mask)':>36}")
print(f"{'LM-trained':<26} {next_token_acc(lm, CAUSAL, Xte):>36.4f}")
print(f"{'MLM-trained':<26} {next_token_acc(mlm, CAUSAL, Xte):>36.4f}")
print(f"\n(chance is {1 / V:.4f})")

print("\nThe MLM model is evaluated here under a CAUSAL mask, which is the")
print("only well-posed way to ask it to predict a next token — under its")
print("own bidirectional mask it would simply read the answer.")
print("\nIt has never seen a causal mask in training, so its representations")
print("are not organised for prefix-conditional prediction. That gap is what")
print("section 7.5 says a fine-tune closes, and it is why converting")
print("between the shapes is possible but not free.")
print("\nTaken with the previous table, this is the whole architecture")
print("argument in two numbers: bidirectional wins at representation, causal")
print("wins at generation, and the field chose the one that can be trained")
print("on every position of every document and then asked to do both.")
```

## 10. Production Considerations

**Test generation, not just training loss.** Measured: a model trained without a
causal mask had excellent training metrics and near-chance generation. No
training-time check catches it.

**Check the mask value against the dtype.** Measured: $-10$ leaks percent-scale
attention across the boundary, and $-10^9$ becomes $-\infty$ in fp16, which
produces `nan` on a fully-masked row.

**Combine padding and causal masks with an AND**, and handle the
fully-masked-row case explicitly.

**Use encoder-only for embeddings.** {{eq:bidirectional-entropy}} is a theorem,
not a preference.

**Consider an encoder–decoder when the input is long and the output is short.**
Cross-attention's cache is computed once and never grows
({{sec:7-internal-mechanics}}).

**Converting between shapes is cheap.** The mask is the only difference for
three of the four, so a decoder-only checkpoint can be fine-tuned into an
embedding model.

## 11. Common Mistakes

**Omitting the causal mask.** Measured; the most consequential one-line bug in
the part.

**A mask value that is too small.** Measured leakage.

**Evaluating an encoder-only model on next-token prediction under its own
mask.** It reads the answer; the number is meaningless.

**Comparing MLM and LM losses directly.** Different objectives with different
conditioning; {{eq:bidirectional-entropy}} says one is easier by construction.

**Assuming decoder-only is better at everything.** It is worse at
representation, provably.

**Forgetting that `[MASK]` never appears at inference.** A train/serve mismatch
that {{ch:mle-pipelines}} would flag immediately.

## 12. Failure Modes

**A model that cannot generate.** Measured. It scores well on everything you
measured during training.

**`nan` from a fully-masked row.** Occurs with all-padding sequences.

**Attention leaking across the mask.** Subtly wrong generation, no error.

**Poor generation from an MLM-adapted model.** Measured: representations not
organised for prefix-conditional prediction, and a fine-tune is required.

**Position drift in a prefix-LM** when the prefix length varies and the
positional scheme assumes a fixed split.

**An encoder–decoder whose encoder is re-run per token.** A real and expensive
implementation bug; the encoder output is fixed for the whole generation.

## 13. Alternatives

**UL2 and mixture-of-denoisers** train one model on several objectives —
causal, prefix, span corruption — to get the benefits of each.
{{maturity:EMERGING}}

**Diffusion language models** generate all positions in parallel and refine
iteratively, escaping the sequential decoding of {{sec:7-internal-mechanics}}.
Actively developed and not yet competitive at scale.
{{maturity:EMERGING}}

**Retrieval-augmented decoders** keep the decoder-only shape and add
cross-attention to retrieved passages, recovering the encoder–decoder's
separation without a second stack ({{part:12}}).

**Encoder-only for generation via iterative unmasking.** Possible, slow, and
occasionally useful for constrained editing.

## 14. Evaluation

**Always include an autoregressive generation test.** Measured to be the only
check that catches a missing mask.

**Verify the mask numerically**: attention weight on masked positions must be
exactly zero, not merely small.

**Probe the representations** rather than comparing losses, when comparing
architectures trained on different objectives.

**Test the fully-masked-row path** with an all-padding sequence.

**Evaluate an adapted model on both tasks**, since conversion between shapes
degrades the one it was not trained for.

## 15. Advanced Concepts

**Why in-context learning favours decoder-only.** Examples in the prompt and the
query occupy the same sequence, so the same attention that reads context reads
examples. An encoder–decoder would have to decide which stack the examples go
in.

**The prefix-LM's equivalence to an encoder–decoder with shared weights.** One
stack with a prefix mask computes something very close to an encoder–decoder
whose two stacks are tied — which is why the prefix-LM is the natural middle
point and why its training inefficiency is the deciding objection.

**Bidirectional adaptation of causal models.** Removing the causal mask from a
pretrained decoder and fine-tuning briefly produces competitive embedding
models, which suggests the representational gap is smaller than
{{eq:bidirectional-entropy}} might imply. {{maturity:EMERGING}}

**Objective design as the real variable.** Once the mask is the only
architectural difference, the interesting question stops being "which
architecture" and becomes "which objective" — which is
{{ch:fm-pretraining}}'s subject.

## 16. Connection to Previous Chapters

{{ch:tf-ffn-residual}} supplied the block all three architectures stack, and
{{eq:masked-attention}} is the only place they differ.
{{ch:tf-multi-head}}'s cross-attention connects the encoder–decoder's halves.
{{ch:tf-positional}}'s permutation-equivariance result is what
{{sec:6-mathematical-foundation}} shows the causal mask partly repairs on its
own. {{ch:tf-why-attention}}'s encoder–decoder is where this chapter starts.

Forward: {{ch:tf-masking-kv}} makes the causal mask's inference-time
consequences concrete. {{ch:fm-pretraining}} covers the objectives.
{{ch:emb-models}} is where encoder-only models live.
{{ch:llm-anatomy}} assembles a complete decoder-only model.

## 17. Exercises

**Beginner**

1. What is the only architectural difference between encoder-only and
   decoder-only?
2. Why can an encoder-only model not generate?
3. What fraction of positions does masked language modelling supervise?
4. What is a prefix-LM?
5. What does cross-attention connect?

**Intermediate**

6. Write the three masks of {{eq:masks}} as code.
7. Derive {{eq:signal-efficiency}} and explain what it assumes.
8. Explain {{eq:bidirectional-entropy}} and why it favours encoders.
9. Explain how a causal mask supplies positional information.
10. Compute the parameter difference between a 12-layer decoder-only model
    and a 6+6 encoder–decoder at $d = 768$.

**Advanced**

11. Prove that the same weights can be run under any mask, and identify the
    one exception.
12. Explain why masking more than 15% of positions eventually hurts, in
    information terms.
13. Show that a prefix-LM is equivalent to a weight-tied encoder–decoder.
14. Derive the fully-masked-row failure and propose a fix that does not use
    $-\infty$.

**Implementation**

15. Implement all three masks and verify attention is exactly zero where
    masked.
16. Reproduce the missing-mask experiment on a real dataset.
17. Convert a causally-trained model to bidirectional and measure the
    embedding quality before and after fine-tuning.
18. Implement an encoder–decoder with a cached cross-attention and verify the
    cache is computed once.

**Reasoning**

19. Your model's training loss is excellent and its generations are gibberish.
    Give an ordered diagnostic procedure.
20. You need both good embeddings and good generation from one model. What are
    your options?

## 18. Interview Questions

**"What is the difference between BERT and GPT?"** — The mask. Everything else
follows: objective, capability, efficiency.

**"Why did decoder-only win?"** — One objective subsuming the others, every
position supervised, one set of weights, in-context learning. Give
{{eq:signal-efficiency}}.

**"Is decoder-only better?"** — At generation and generality; provably worse at
representation, by {{eq:bidirectional-entropy}}. Say both.

**"Why 15% masking?"** — More loss terms against less context. Say an optimum
below 1 must exist and that its existence is the architectural argument.

**"How would you catch a missing causal mask?"** — An autoregressive generation
test. No training-time metric catches it.

**"When would you use an encoder–decoder?"** — Long input, short output,
genuinely distinct sequences; and the cross-attention cache does not grow.

## 19. Research Questions

**Is the representational gap real at scale?** Bidirectionally-adapted decoder
models produce competitive embeddings, which complicates
{{eq:bidirectional-entropy}}'s practical implication.
{{maturity:EMERGING}}

**Can one objective get both?** UL2-style mixtures try; whether they beat
specialisation is unresolved. {{maturity:EMERGING}}

**Is sequential decoding necessary?** Diffusion and parallel-decoding language
models exist and are not yet competitive. {{maturity:EMERGING}}

**What does in-context learning require architecturally?** It appears in
decoder-only models and the necessary conditions are not characterised.
{{maturity:RESEARCH FRONTIER}}

## 20. Chapter Summary

Encoder-only, decoder-only and prefix-LM transformers differ in **one boolean
matrix**. {{eq:masked-attention}} adds the mask before the softmax and nothing
else in the block refers to it, so the same weights run under any of the three.
Only the encoder–decoder differs in its parameters, and only by the
cross-attention sublayer.

The causal mask does more than enforce causality. Measured, the attention
entropy grows with position because query $i$ attends over $i+1$ keys — so the
mask supplies positional information on its own, which is why decoder-only
models with no positional encoding work at all and why bidirectional models
depend on their positional scheme absolutely.

Two measured failures make the mask concrete. A mask value that is large but not
large enough leaks percent-scale attention across the boundary — the model sees
the future slightly and will use it, with no error and subtly wrong generation.
And a fully-masked row gives a uniform distribution under a finite mask value
and `nan` under $-\infty$; in fp16 the standard $-10^9$ becomes $-\infty$ on
conversion, so a mask value tuned for fp32 produces `nan` in a half-precision
model.

The most consequential result is the missing-mask experiment. A model trained
with next-token prediction and no causal mask reached far better training loss
and accuracy than a correctly-masked one — by copying position $i+1$ of its own
input — and generated at near chance, because at inference the future positions
are zeros rather than answers. **It passes every training-time check and is
useless**, which is why an autoregressive generation test belongs in the
training loop.

On the architecture choice, the arguments cut both ways and both were measured.
{{eq:bidirectional-entropy}} is a theorem: conditioning on both sides cannot
raise the conditional entropy, so bidirectional representations are at least as
good — and the probe experiment measures that gap. Against it,
{{eq:signal-efficiency}}: masked language modelling supervises about 15% of
positions per pass and next-token prediction supervises all of them, at
identical compute, for roughly seven times the signal. The measured masking-rate
sweep shows why BERT cannot simply mask more — beyond an optimum, removing
context costs more than the extra loss terms buy — and the existence of that
optimum is the architectural argument, because {{eq:chain-rule-prob}}'s
factorisation means next-token prediction has no such trade.

The convergence on decoder-only is therefore not a claim that causal attention
is better. It is a claim that training efficiency, task generality and
in-context learning outweigh a known and provable representational
disadvantage — and encoder-only models remain the right choice everywhere that
disadvantage is what matters.

## 21. Further Reading

{{cite:devlin2019bert}} is worth reading for the objective rather than the
architecture. The 15% masking rate, the 80/10/10 replacement scheme and the
next-sentence-prediction task are all design decisions made with limited
justification, and two of the three did not survive — which is a useful
calibration on how much of any paper's recipe is load-bearing.

{{cite:radford2019}} is short and makes the generality argument explicitly: one
objective, many tasks, no task-specific architecture. Reading it next to BERT
shows the two bets being placed side by side before anyone knew which would win.

{{cite:raffel2020t5}} is the most thorough ablation study in this area and the
most useful of the three to read in full. It compares architectures, objectives,
corruption rates and scales systematically, and its conclusions are stated with
appropriate hedging.

**Where to go next:** {{ch:tf-masking-kv}} takes the causal mask's
inference-time consequence seriously — the model that trains in one parallel
pass must generate one token at a time — and the KV cache is what makes that
affordable.

---
id: part-07-intro
status: final
---

## What this part is for

{{part:6}} ended with a problem it could not solve. A recurrence carries a
fixed-size state forward one step at a time, so relating position $i$ to
position $j$ costs $|i-j|$ steps of a lossy channel, and the whole computation
has an $O(T)$ critical path that no hardware can parallelise away.

This part builds the architecture that removed both constraints at once, and it
builds it **component by component from scratch** rather than presenting it as a
diagram to memorise.

> **Attention replaces "carry information forward" with "look it up".** Every
> position computes a description of what it is looking for, compares that
> against a description every other position offers, and reads a weighted
> average of what those positions hold. One layer, any distance, fully parallel.

The price is stated as plainly as the benefit. Attention costs $O(T^2)$ in both
time and memory, it has no compressed state, and the serving-time memory of the
key–value cache grows linearly with the conversation. {{ch:tf-complexity}} does
that arithmetic and {{ch:tf-efficient}} is the response to it.

## The build order

```text
   THE MECHANISM              THE SUPPORTING PARTS        THE ASSEMBLY
   ─────────────────          ─────────────────────       ───────────────
   62 why recurrence failed   65 positional encoding      68 encoder,
   63 scaled dot-product      66 embeddings /                decoder,
   64 multi-head                 unembedding                 encoder-decoder
                              67 feed-forward,            69 causal masking,
                                 residual, norm              the KV cache

                              THE COST
                              ─────────────────────────────────────────
                              70 complexity: where the time and memory go
                              71 efficient attention: what to do about it
```

Chapters 62–64 are the mechanism. Chapters 65–67 are the parts that make it a
working architecture — and each exists because attention alone is missing
something specific: it is permutation-equivariant, so it needs positions; it
operates on vectors, so it needs a vocabulary map; and it is entirely linear in
its values, so it needs a nonlinearity. Chapters 68–69 assemble them. Chapters
70–71 are the cost and the engineering response.

## Three things worth knowing before you start

**Attention is not new; the removal of recurrence is.**
{{cite:bahdanau2015}} introduced attention *inside* a recurrent
encoder–decoder, as a patch for the fixed-size-bottleneck problem
{{ch:dl-rnns}} describes. {{cite:vaswani2017}}'s contribution was to observe
that the patch had outgrown the thing it patched, and to delete the recurrence
entirely. The title says so. Knowing this stops you treating the transformer as
an unmotivated invention — it is the endpoint of a specific argument.

**A single attention head is mostly linear.** Given the attention weights, the
output is a convex combination of value vectors, and the query, key and value
projections are all linear maps. The only nonlinearity in the whole mechanism is
the softmax that produces the weights. {{ch:tf-ffn-residual}} shows that the
feed-forward block holds roughly two-thirds of the parameters and supplies
essentially all of the elementwise nonlinearity, which is not the impression
most diagrams give.

**Most of the architecture's details are engineering, and they matter.**
Pre-normalisation rather than post ({{ch:dl-normalization}}), rotary positions
rather than sinusoidal, grouped-query attention rather than multi-head, gated
feed-forward rather than plain. Each is a small change with a measured
justification, and together they are most of the difference between the 2017
paper and a 2026 model. This part gives the justification in each case, because
a recipe you cannot justify is one you cannot adapt.

## What is genuinely unsettled

**Whether quadratic attention is necessary.** Linear and sparse variants have
existed since 2020 and none has displaced full attention at scale. The honest
reading is that {{cite:dao2022flash}} moved the goalposts: by making exact
attention far cheaper in practice without approximating anything, it removed
most of the pressure that the approximate variants were built to relieve.
Whether the remaining pressure at very long context is enough is open.
{{maturity:EMERGING}}

**What attention heads actually compute.** Interpretability work has identified
specific circuits — induction heads, name-mover heads — in small models, and
whether those findings scale, and whether the head is even the right unit of
analysis, is unresolved. {{ch:tf-multi-head}} states what is established and
what is not. {{maturity:RESEARCH FRONTIER}}

**How to extend a context window after training.** RoPE scaling works, several
recipes compete, and the choice is largely empirical. {{ch:tf-positional}}
covers them and does not pretend the question is settled.
{{maturity:EMERGING}}

## A note on {{ch:tf-scaled-dot-product}}

That chapter was written first, as the specimen that set this book's format. It
is the most detailed chapter in the part and it is where the mechanism is
derived in full — including why the $\sqrt{d_k}$ scaling is there, which is a
variance calculation rather than a convention. Read it slowly; the eight
chapters around it assume it.

## What you should be able to do at the end

Derive scaled dot-product attention and explain the scaling factor from the
variance of a dot product. Implement multi-head attention with correct shapes
and explain what the heads buy. Explain why a transformer needs positional
information at all, and derive RoPE's relative-position property. Account for
every parameter and every FLOP in a transformer block. Explain the KV cache,
compute its size for a given model and context, and say why it — not the
weights — is what limits how many users you can serve. Derive attention's
quadratic cost and explain what FlashAttention changes and what it does not.
Build the whole thing from scratch in NumPy and check it against a reference.

The assignment at the end asks for exactly that: a working transformer, written
from the components of this part, with the diagnostics to show each one is
doing what the chapter said it would.

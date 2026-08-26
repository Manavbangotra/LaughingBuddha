---
id: llm-anatomy
number: 88
part: X
tier: full
status: draft
requires: [tf-architectures, tf-embeddings, tf-masking-kv, tf-complexity,
           nlp-subword, fm-instruction-tuning, dl-normalization]
provides: [llm-forward-pass, logit-vector, residual-stream-trace, unembedding-step,
           shape-discipline, model-as-function, final-norm, next-token-distribution]
citations: [vaswani2017, radford2019, brown2020, touvron2023llama,
            shazeer2020glu, zhang2019rmsnorm, su2021rope, press2017tying]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Trace a prompt from string to logit vector, naming every transformation and
   its output shape.
2. State what an LLM *is*, mathematically, in one sentence that survives
   scrutiny.
3. Explain what the residual stream carries and why every block reads from and
   writes to it.
4. Explain the role of the final normalisation and why removing it breaks the
   logit scale.
5. Compute the parameter count of a model from its configuration, and check it
   against the published number.
6. Identify which properties users attribute to "the model" that are not in the
   weights at all.
7. Implement a complete forward pass and verify it against shape assertions.

## 2. Why This Matters

**This part is a debugging manual, and this chapter is its map.** Everything in
{{part:10}} — decoding, prompting, structured output, hallucination, long
context — is a question about what happens between a string arriving and a token
leaving. You cannot localise a fault in a pipeline you cannot trace.

**The single most useful idea in this part is stated here.** An LLM is a
function from a token sequence to a probability distribution over the next
token. That is all it is. Temperature is not in it. Top-p is not in it. The
system prompt is not privileged. Whether a tool gets called is not a property of
the weights. **Nearly every complaint about LLM behaviour resolves to the
decoding strategy or the serving configuration rather than to the model**, and
being able to make that attribution is what separates useful debugging from
guessing.

**It is also where the shapes finally get pinned down.** {{part:7}} built each
component; here they are assembled with every intermediate dimension stated and
asserted, because shape confusion is the most common source of bugs when people
first implement or modify these models.

**And the parameter arithmetic pays for itself immediately.** Being able to
compute a model's size from its configuration lets you check a published claim,
estimate memory before loading anything, and notice when a configuration is not
what it says it is.

## 3. Prerequisites

{{ch:tf-architectures}} for the decoder-only stack — this chapter assembles it
rather than re-deriving it. {{ch:tf-embeddings}} for the embedding and
unembedding matrices and weight tying. {{ch:tf-masking-kv}} for causal masking.
{{ch:tf-complexity}} for the parameter and FLOP accounting.
{{ch:nlp-subword}} for what a token is. {{ch:fm-instruction-tuning}} for the
chat template that wraps the user's text before any of this begins.
{{ch:dl-normalization}} for RMSNorm.

## 4. Intuitive Explanation

A user types a sentence. Some milliseconds later a word appears. Between those
two events is a sequence of transformations, and every one of them is
inspectable.

**The string becomes numbers.** The chat template wraps it
({{ch:fm-instruction-tuning}}), the tokenizer splits it
({{ch:nlp-subword}}), and you have a list of integers — say 47 of them.

**The integers become vectors.** Each token ID indexes a row of the embedding
matrix, giving a $47 \times d$ array. This is the only place the token
*identity* enters; from here on the model sees vectors.

**The vectors are refined, repeatedly.** Each transformer block reads the
current representation, computes something, and *adds* it back. Attention lets
each position gather information from earlier positions; the feed-forward
network transforms each position independently. After $L$ blocks you still have
a $47\times d$ array — the same shape as the input.

> NOTE: That the shape never changes is the most important structural fact about
> a transformer. The representation is a **residual stream** that every block
> reads from and writes to, rather than a pipeline that transforms one thing
> into another. A block that has nothing to contribute can add approximately
> zero and the information passes through untouched.

**The last position becomes a distribution.** Take the final row — the
representation at the last token — normalise it, and multiply by the unembedding
matrix to get one score per vocabulary item. That is the **logit vector**, and
it has length $|V|$: 128,000 numbers, one per possible next token.

**And then the model is done.** It has produced a distribution over next tokens.
It has not chosen one. Choosing is {{ch:llm-decoding}}'s subject, it happens
outside the model, and it is where temperature and top-p live.

**Why only the last position?** Every position produced a logit vector — the
model predicts a next token *at every position*, which is what made training
efficient ({{ch:fm-pretraining}}). At inference you already know the actual next
tokens for every position but the last, so only the last one's prediction is
new.

**The mental model:** the model is a function $\R^{T} \to \R^{|V|}$ — tokens in,
one score per vocabulary item out — applied to a growing sequence. Where it
breaks down: the function is applied repeatedly with its own output appended,
and that loop is where almost everything interesting and almost everything
wrong happens. The loop is not the model.

## 5. Formal Explanation

### 5.1 The model as a function

An LLM with parameters $\theta$ is a map

$$
f_\theta : \{1,\dots,|V|\}^{T} \to \R^{|V|}
$$ (eq:llm-as-function)

taking a token sequence to a vector of logits for the next token. Composing with
a softmax gives a distribution:

$$
P(x_{T+1} = v \given x_{1:T}) = \softmax\big(f_\theta(x_{1:T})\big)_v
$$ (eq:next-token-distribution)

**Everything in {{part:10}} is either about computing $f_\theta$ efficiently, or
about what to do with {{eq:next-token-distribution}} once you have it.** Keeping
that boundary clear is the chapter's main contribution.

### 5.2 The trace, with shapes

Let $T$ be sequence length, $d$ the model width, $L$ the number of layers, $h$
the number of heads, $d_{\text{ff}}$ the feed-forward width, and $|V|$ the
vocabulary.

**1. Embedding.** Token IDs $\vec{x}\in\{1..|V|\}^{T}$ index the embedding
matrix $\mat{E}\in\R^{|V|\times d}$:

$$
\mat{H}^{(0)} = \mat{E}[\vec{x}] \in \R^{T\times d}
$$ (eq:embed-step)

**2. Blocks.** For $\ell = 1,\dots,L$, with pre-normalisation
({{ch:tf-ffn-residual}}):

$$
\mat{H}' = \mat{H}^{(\ell-1)} + \Attn\big(\Norm(\mat{H}^{(\ell-1)})\big)
$$ (eq:attn-sublayer)

$$
\mat{H}^{(\ell)} = \mat{H}' + \FFN\big(\Norm(\mat{H}')\big)
$$ (eq:ffn-sublayer)

Both sublayers output $\R^{T\times d}$ and both are *added*. The shape is
invariant across all $L$ blocks.

**3. Final normalisation.**

$$
\mat{H}^{\text{final}} = \Norm\big(\mat{H}^{(L)}\big) \in \R^{T\times d}
$$ (eq:final-norm)

**4. Unembedding**, at the last position only:

$$
\vec{z} = \mat{H}^{\text{final}}_{T,:}\,\mat{W}_U \in \R^{|V|},
\qquad \mat{W}_U \in \R^{d\times|V|}
$$ (eq:unembed-step)

With weight tying ({{ch:tf-embeddings}}), $\mat{W}_U = \mat{E}\T$.

### 5.3 Why the final norm is load-bearing

{{eq:final-norm}} looks like housekeeping and is not. The residual stream
accumulates $2L$ additive contributions, so its magnitude grows through the
stack — and {{eq:unembed-step}} multiplies it directly by the unembedding
matrix.

Without the final norm, the *scale* of the logits is whatever the residual
stream's norm happens to be, which varies with depth, with input, and with
training progress. Since the softmax is scale-sensitive — multiplying logits by
$c$ is exactly applying temperature $1/c$ ({{ch:llm-decoding}}) — an
unnormalised stream means an uncontrolled, input-dependent temperature.

> IMPORTANT: This is not hypothetical. {{ch:fm-pretraining}}'s listing failed
> its own initial-loss assertion for precisely this reason: a tied unembedding
> with no final norm produced logits whose scale was set by the residual
> stream's magnitude, and the loss at initialisation was 64 instead of
> $\log|V| = 2.8$. The model still trained. It simply started somewhere
> arbitrary.

### 5.4 Parameter accounting

From {{ch:tf-complexity}}, per block:

$$
N_{\text{block}} = \underbrace{4d^2}_{\text{attention}}
 + \underbrace{2\,d\,d_{\text{ff}}}_{\text{FFN}}
$$ (eq:block-params)

and for a gated FFN (SwiGLU, {{cite:shazeer2020glu}}) the second term is
$3\,d\,d_{\text{ff}}$ because there are three matrices rather than two.

Total:

$$
N = \underbrace{|V|d}_{\text{embedding}}
 + L\,N_{\text{block}}
 + \underbrace{|V|d}_{\text{unembedding, if untied}}
$$ (eq:total-params)

**The embedding term is not negligible at small scale.** At $|V| = 128{,}000$
and $d = 2048$ it is 262M parameters — larger than the entire block stack for a
small model, which is why weight tying matters most exactly where models are
smallest.

### 5.5 What is not in the model

Worth stating explicitly because so much confusion attaches to it:

{#tbl:not-in-the-model caption="Properties users routinely attribute to a model that are not in its weights. Every row is a decision made outside f_theta, and every row is a place a fault can be localised that is not the model's fault."}

| Property | Where it actually lives | Chapter |
|---|---|---|
| Temperature, top-p, top-k | the sampling loop | {{ch:llm-decoding}} |
| Context window length | positional scheme + serving config | {{ch:llm-inference}} |
| "System prompt" privilege | a training convention, then a template | {{ch:llm-prompting}} |
| Valid JSON output | a token mask at decode time | {{ch:llm-structured-output}} |
| Whether a tool is called | template + dispatch loop | {{ch:llm-function-calling}} |
| Stopping | a stop-token check outside the model | {{ch:llm-prompt-lifecycle}} |
| Streaming | how the loop's output is delivered | {{ch:llm-prompt-lifecycle}} |

## 6. Mathematical Foundation

### 6.1 The residual stream as a sum

Unrolling {{eq:attn-sublayer}} and {{eq:ffn-sublayer}} across all layers:

$$
\mat{H}^{(L)} = \mat{H}^{(0)}
 + \sum_{\ell=1}^{L}\Big[\Attn_\ell\big(\Norm(\cdot)\big)
 + \FFN_\ell\big(\Norm(\cdot)\big)\Big]
$$ (eq:residual-stream-sum)

$\square$

**The final representation is the embedding plus the sum of every sublayer's
contribution.** Three consequences follow directly:

1. **Every sublayer has a direct path to the output.** The gradient reaches
   layer 1 without passing through a product of $L$ Jacobians, which is why deep
   transformers train at all ({{ch:dl-backprop}}).
2. **Sublayers can be near-inert.** A block contributing approximately zero
   leaves the stream unchanged, so depth is not forced to do work.
3. **The stream is a shared workspace, not a pipeline.** Blocks communicate by
   reading and writing a common representation, which is the basis for the
   interpretability view of transformers.

### 6.2 Why the logit scale is the temperature

Let $\vec{z}$ be logits and consider scaling: $\vec{z}' = c\vec{z}$.

$$
\softmax(c\vec{z})_i = \frac{e^{cz_i}}{\sum_j e^{cz_j}}
 = \frac{e^{z_i/(1/c)}}{\sum_j e^{z_j/(1/c)}}
 = \softmax(\vec{z}; T = 1/c)_i
$$ (eq:scale-is-temperature)

$\square$

**Scaling logits by $c$ is identical to sampling at temperature $1/c$.** So
anything that changes the logit magnitude changes the effective temperature:
the final norm's gain parameter, the unembedding's scale, and numerical
precision at the output all do. This identity reappears throughout
{{ch:llm-decoding}} and is the reason the final norm is not optional.

### 6.3 A worked parameter count

A model with $L = 32$, $d = 4096$, $d_{\text{ff}} = 11008$ (a gated FFN),
$|V| = 32000$, untied embeddings.

Per block:

$$
4d^2 = 4(4096)^2 = 67.1\text{M},
\qquad
3\,d\,d_{\text{ff}} = 3(4096)(11008) = 135.3\text{M}
$$

$$
N_{\text{block}} = 202.4\text{M}
$$

Total:

$$
N = 32\times 202.4\text{M} + 2\times 32000\times 4096
 = 6.48\text{B} + 0.26\text{B} = 6.74\text{B}
$$

**About 6.7B**, which matches the published size of models with this
configuration to better than a per cent — `parameter-accounting` checks four
configurations this way, and the agreement is close enough that a discrepancy is
diagnostic rather than expected. Note the split: **the FFN is two-thirds of the block** — 135M
against 67M — which is {{ch:tf-ffn-residual}}'s result and is where most of a
model's parameters and most of its FLOPs live, despite attention getting the
attention.

### 6.4 What one forward pass costs

From {{ch:tf-complexity}}, prefill over $T$ tokens:

$$
C_{\text{prefill}} \approx 2NT + 4LT^2 d
$$

At $N = 6.7\times10^9$, $T = 512$, $L = 32$, $d = 4096$:

$$
2NT = 6.9\times10^{12},
\qquad
4LT^2d = 4(32)(512^2)(4096) = 1.4\times10^{11}
$$

so the parameter term is about fifty times the attention term at this length —
which is {{ch:tf-complexity}}'s $T < 6d$ regime, and a useful reminder that
"attention is the expensive part" is false for ordinary prompt lengths.

## 7. Internal Mechanics

```mermaid {#fig:llm-forward caption="One forward pass. The shape is T x d from the embedding through to the final norm — every block reads the residual stream and adds to it. Only the last row is unembedded at inference, because the predictions at earlier positions are for tokens already known."}
graph TD
  A["token IDs<br/>(T,)"] --> B["embedding lookup<br/>(T, d)"]
  B --> C["block 1"]
  C --> D["block 2 … block L<br/>shape invariant: (T, d)"]
  D --> E["final norm<br/>(T, d)"]
  E --> F["take last row<br/>(d,)"]
  F --> G["unembed: (d,) x (d, |V|)<br/>-> logits (|V|,)"]
  G --> H["softmax -> distribution<br/>MODEL ENDS HERE"]
  H -.->|"outside the model"| I["sampling: temperature,<br/>top-k, top-p"]
  style H fill:#dfe,stroke:#5a5
  style I fill:#fde,stroke:#c69
```

**Inside one block.** Attention projects the normalised stream to queries, keys
and values ($T\times d$ each, reshaped to $h$ heads of width $d/h$), computes
the masked score matrix ($h\times T\times T$ — the only quadratic object),
weights the values, concatenates, and projects back to $T\times d$. The FFN
projects up to $d_{\text{ff}}$, applies a nonlinearity, and projects back.

**Where the memory goes at inference.** Weights are $bN$ bytes and constant. The
KV cache is $2Lg d_k T b$ and grows with every generated token
({{ch:tf-masking-kv}}). The activations are transient. **For long conversations
the cache overtakes the weights**, which is {{ch:llm-inference}}'s subject.

**Why only the last row is unembedded.** Computing logits for all $T$ positions
costs $2dT|V|$ FLOPs — at $T = 512$, $d = 4096$ and $|V| = 32000$ that is
$1.3\times10^{11}$, comparable to the entire rest of the forward pass. During
*training* it is necessary because every position supplies a loss term. During
*inference* it is pure waste, and skipping it is one of the first optimisations
any serving stack makes.

**The template runs before any of this.** By the time token IDs exist, the chat
template of {{ch:fm-instruction-tuning}} has already wrapped the user's text in
role markers. A model appears to "know" it is in a conversation because those
markers are in its input, not because of any architectural provision for
dialogue.

**Batching changes the shapes and nothing else.** Serving processes many
sequences at once, so every array in {{sec:5-formal-explanation}} gains a
leading batch dimension: $(B, T, d)$ through the stack, $(B, h, T, T)$ for the
scores, $(B, |V|)$ for the logits. Sequences in a batch have different lengths,
so they are padded to a common $T$ and an attention mask marks the padding —
which must be combined with the causal mask rather than replacing it. **Getting
that combination wrong is a common bug and it is asymmetric**: attending to
padding degrades quality quietly, while masking real tokens produces obvious
nonsense, so the dangerous direction is the one that does not announce itself.

**Numerical precision at the output deserves its own note.** The stack runs in
bf16 on modern hardware, but {{eq:unembed-step}} produces a $|V|$-length vector
that is then softmaxed, and bf16 has about three significant decimal digits.
Differences between the tail probabilities — precisely the region top-p
truncation operates on ({{ch:llm-decoding}}) — are near the representable
resolution. Serving stacks therefore compute the final projection and the
softmax in float32 even when everything upstream is bf16. It is a one-line
decision with a measurable effect on sampling behaviour, and it is invisible in
any architecture diagram.

**Where the trace can be cut short.** Nothing requires the full $L$ blocks to
run before unembedding — {{eq:residual-stream-sum}} means the stream is in the
same space at every depth, so the unembedding matrix can be applied to layer
$\ell$'s output and yields a valid, if worse, distribution. That observation is
the basis of the logit lens in {{sec:15-advanced-concepts}} and of early-exit
inference, and it is a direct consequence of the residual structure rather than
an architectural addition.

## 8. Implementation

A complete forward pass, with every shape asserted.

```python {tier=A name=llm-forward-pass}
"""A full LLM forward pass in numpy, with every intermediate shape checked."""
import numpy as np

rng = np.random.default_rng(0)

# A model small enough to inspect, with a real model's structure.
V, D, L, H, D_FF = 512, 64, 4, 4, 176      # vocab, width, layers, heads, ffn
D_HEAD = D // H
assert D % H == 0

params = {
    "embed": rng.normal(0, 0.02, (V, D)),
    "final_norm_gain": np.ones(D),
    "blocks": [],
}
for _ in range(L):
    params["blocks"].append({
        "n1_gain": np.ones(D), "n2_gain": np.ones(D),
        "wq": rng.normal(0, 0.02, (D, D)), "wk": rng.normal(0, 0.02, (D, D)),
        "wv": rng.normal(0, 0.02, (D, D)), "wo": rng.normal(0, 0.02, (D, D)),
        "w_gate": rng.normal(0, 0.02, (D, D_FF)),
        "w_up": rng.normal(0, 0.02, (D, D_FF)),
        "w_down": rng.normal(0, 0.02, (D_FF, D)),
    })


def rmsnorm(x, gain, eps=1e-6):
    """ch:dl-normalization — no mean subtraction, just scale."""
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps) * gain


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def silu(x):
    return x / (1 + np.exp(-x))


def attention(x, p, trace):
    T = x.shape[0]
    q = (x @ p["wq"]).reshape(T, H, D_HEAD).transpose(1, 0, 2)   # (H, T, dh)
    k = (x @ p["wk"]).reshape(T, H, D_HEAD).transpose(1, 0, 2)
    v = (x @ p["wv"]).reshape(T, H, D_HEAD).transpose(1, 0, 2)
    trace["q"] = q.shape

    scores = q @ k.transpose(0, 2, 1) / np.sqrt(D_HEAD)          # (H, T, T)
    trace["scores"] = scores.shape
    # Causal mask, ch:tf-masking-kv: position t may not see t' > t.
    mask = np.triu(np.full((T, T), -np.inf), 1)
    weights = softmax(scores + mask)
    assert np.allclose(weights.sum(-1), 1.0), "attention rows must be distributions"
    assert np.allclose(np.triu(weights, 1), 0.0), "no attention to future positions"

    out = (weights @ v).transpose(1, 0, 2).reshape(T, D)          # (T, D)
    return out @ p["wo"]


def ffn(x, p):
    """Gated FFN — three matrices, not two (shazeer2020glu)."""
    return (silu(x @ p["w_gate"]) * (x @ p["w_up"])) @ p["w_down"]


def forward(token_ids, verbose=True):
    T = len(token_ids)
    trace = {}
    h = params["embed"][token_ids]                                # (T, D)
    if verbose:
        print(f"{'stage':<28} {'shape':>14}")
        print(f"{'token ids':<28} {str((T,)):>14}")
        print(f"{'after embedding':<28} {str(h.shape):>14}")
    assert h.shape == (T, D)

    for i, p in enumerate(params["blocks"]):
        h = h + attention(rmsnorm(h, p["n1_gain"]), p, trace)     # eq:attn-sublayer
        assert h.shape == (T, D), "attention sublayer must preserve shape"
        h = h + ffn(rmsnorm(h, p["n2_gain"]), p)                  # eq:ffn-sublayer
        assert h.shape == (T, D), "ffn sublayer must preserve shape"
        if verbose and i == 0:
            print(f"{'  qkv per head':<28} {str(trace['q']):>14}")
            print(f"{'  attention scores':<28} {str(trace['scores']):>14}")
            print(f"{'after block 1':<28} {str(h.shape):>14}")

    h = rmsnorm(h, params["final_norm_gain"])                     # eq:final-norm
    if verbose:
        print(f"{'after block ' + str(L):<28} {str(h.shape):>14}")
        print(f"{'after final norm':<28} {str(h.shape):>14}")

    last = h[-1]                                                  # (D,)
    logits = last @ params["embed"].T                             # weight tying
    if verbose:
        print(f"{'last position only':<28} {str(last.shape):>14}")
        print(f"{'logits':<28} {str(logits.shape):>14}")
    assert logits.shape == (V,)
    return logits


tokens = rng.integers(0, V, size=12)
logits = forward(tokens)
probs = softmax(logits)

print(f"\nlogit range        : [{logits.min():+.3f}, {logits.max():+.3f}]")
print(f"probabilities sum  : {probs.sum():.6f}")
print(f"top-5 token ids    : {np.argsort(-probs)[:5].tolist()}")
print(f"entropy            : {-(probs * np.log(probs + 1e-12)).sum():.4f} nats")
print(f"uniform entropy    : {np.log(V):.4f} nats  (untrained model, so close)")

# The shape invariance of eq:residual-stream-sum, verified across lengths.
print(f"\n{'sequence length':>16} {'logits shape':>14}")
for T in (1, 5, 40):
    lg = forward(rng.integers(0, V, size=T), verbose=False)
    print(f"{T:>16} {str(lg.shape):>14}")
print("\nThe logit vector is (|V|,) whatever the input length — the model maps "
      "any sequence to one distribution over the next token, which is "
      "equation (eq:llm-as-function).")
```

Now the parameter accounting, checked against a real configuration:

```python {tier=A name=parameter-accounting}
"""Compute a model's parameter count from its configuration and check it."""

CONFIGS = {
    # (layers, width, ffn width, vocab, heads, gated FFN, tied embeddings)
    "GPT-2 small":   dict(L=12, d=768,  d_ff=3072,  V=50257, gated=False, tied=True),
    "GPT-2 XL":      dict(L=48, d=1600, d_ff=6400,  V=50257, gated=False, tied=True),
    "7B-class":      dict(L=32, d=4096, d_ff=11008, V=32000, gated=True,  tied=False),
    "13B-class":     dict(L=40, d=5120, d_ff=13824, V=32000, gated=True,  tied=False),
}

PUBLISHED = {"GPT-2 small": 124e6, "GPT-2 XL": 1558e6,
             "7B-class": 6.74e9, "13B-class": 13.0e9}


def count_params(L, d, d_ff, V, gated, tied):
    attn = 4 * d * d                      # eq:block-params
    ff = (3 if gated else 2) * d * d_ff
    blocks = L * (attn + ff)
    embed = V * d
    unembed = 0 if tied else V * d
    return dict(blocks=blocks, attn=L * attn, ff=L * ff,
                embed=embed, unembed=unembed,
                total=blocks + embed + unembed)


print(f"{'model':<14} {'computed':>11} {'published':>11} {'error':>8} "
      f"{'embed share':>12} {'FFN share of block':>20}")
for name, cfg in CONFIGS.items():
    c = count_params(**cfg)
    pub = PUBLISHED[name]
    embed_share = (c["embed"] + c["unembed"]) / c["total"]
    ff_share = c["ff"] / c["blocks"]
    print(f"{name:<14} {c['total'] / 1e9:>10.3f}B {pub / 1e9:>10.3f}B "
          f"{abs(c['total'] - pub) / pub:>7.1%} {embed_share:>12.1%} "
          f"{ff_share:>20.1%}")

print("""
Three things to read off.

The accounting is accurate to well under 1% against published figures, using
nothing but the config. It omits biases, normalisation gains and position
embeddings, all of which are O(Ld) rather than O(Ld^2) and so vanish at scale.
If YOUR computed count is off by more than a per cent or two, the configuration
is not what you think it is — that is the check this listing is for.

The embedding share collapses with scale: 31% of GPT-2 small, 2.5% of a 13B
model. That is why weight tying matters most exactly where models are smallest,
and why vocabulary size is a real design constraint for a small model and a
rounding error for a large one.

And the FFN is almost exactly two-thirds of every block, in every configuration
here. Attention gets the attention; the feed-forward network holds the
parameters.""")
```

## 9. Practical Example

A team is choosing between two models for a latency-sensitive feature. One is
"3B" and one is "7B". The names suggest a 2.3x difference in everything. They
are not the same shape, and the differences that matter for serving are not
proportional to the name.

```python {tier=A name=model-shape-comparison}
"""What a model's configuration tells you that its parameter count does not."""

MODELS = {
    "A (3B, wide)":   dict(L=26, d=3072, d_ff=8192,  V=32000, h=24, kv_h=24),
    "B (3B, deep)":   dict(L=40, d=2560, d_ff=6912,  V=32000, h=20, kv_h=20),
    "C (7B, GQA)":    dict(L=32, d=4096, d_ff=11008, V=32000, h=32, kv_h=8),
}

BYTES = 2                      # bf16
CONTEXT = 8192
BATCH = 16


def analyse(L, d, d_ff, V, h, kv_h):
    d_head = d // h
    params = L * (4 * d * d + 3 * d * d_ff) + 2 * V * d
    weights_gb = params * BYTES / 1e9
    # KV cache: 2 (K and V) x layers x kv_heads x head_dim x tokens x bytes
    kv_per_token = 2 * L * kv_h * d_head * BYTES
    kv_gb = kv_per_token * CONTEXT * BATCH / 1e9
    return dict(params=params, weights_gb=weights_gb,
                kv_per_token=kv_per_token, kv_gb=kv_gb,
                depth=L, flops_per_token=2 * params)


print(f"context {CONTEXT:,}, batch {BATCH}, bf16\n")
print(f"{'model':<15} {'params':>9} {'weights':>9} {'KV/token':>10} "
      f"{'KV total':>10} {'total GB':>10} {'depth':>7}")
rows = {}
for name, cfg in MODELS.items():
    a = analyse(**cfg)
    rows[name] = a
    total = a["weights_gb"] + a["kv_gb"]
    print(f"{name:<15} {a['params'] / 1e9:>8.2f}B {a['weights_gb']:>8.1f}G "
          f"{a['kv_per_token']:>9,}B {a['kv_gb']:>9.1f}G {total:>9.1f}G "
          f"{a['depth']:>7}")

a, b, c = rows["A (3B, wide)"], rows["B (3B, deep)"], rows["C (7B, GQA)"]
print(f"\nA and B are both '3B' and differ by "
      f"{abs(a['params'] - b['params']) / a['params']:.1%} in parameters.")
print(f"  KV cache per token : {a['kv_per_token']:,} vs {b['kv_per_token']:,} "
      f"({b['kv_per_token'] / a['kv_per_token']:.2f}x)")
print(f"  depth              : {a['depth']} vs {b['depth']} layers "
      f"-> B has {b['depth'] / a['depth']:.2f}x the sequential steps per token")

print(f"\nC is {c['params'] / a['params']:.1f}x A's parameters, but its KV "
      f"cache per token is {c['kv_per_token'] / a['kv_per_token']:.2f}x —")
print(f"grouped-query attention ({MODELS['C (7B, GQA)']['h']} query heads, "
      f"{MODELS['C (7B, GQA)']['kv_h']} KV heads) decouples the two.")

print(f"\ntotal memory at this batch and context:")
for name in rows:
    r = rows[name]
    print(f"  {name:<15} {r['weights_gb'] + r['kv_gb']:>6.1f} GB "
          f"({r['kv_gb'] / (r['weights_gb'] + r['kv_gb']):.0%} of it cache)")

print("""
The parameter count answers one question — how much arithmetic per token — and
is silent on the two that decide serving.

Depth sets the sequential critical path: decoding is one token at a time, so a
40-layer model has 40 sequential dependencies per token against a 26-layer
model's 26, at identical parameter count. That shows up directly in inter-token
latency and cannot be batched away.

And the KV cache is a function of layers, KV heads and head dimension, with no
term in the parameter count at all. C has more than twice A's parameters and a
SMALLER cache per token, because grouped-query attention decoupled them. At
batch 16 and 8k context the cache is a large share of total memory, which is
what actually limits concurrency.

'3B' and '7B' are marketing. The configuration is the specification.""")
```

> PRODUCTION TIP: Read the config file, not the model name. Layers, KV heads and
> head dimension determine your serving characteristics, and two models with the
> same parameter count can differ by a factor of two in cache footprint and in
> sequential depth.

## 10. Production Considerations

**Log the shapes once, at startup.** Sequence length, batch size, and the
resulting KV-cache footprint should be computed and logged when a model loads.
Most memory surprises are visible in that arithmetic before any request arrives.

**Check the parameter count against the config.** `parameter-accounting` takes
seconds and catches a mis-specified configuration — a wrong `d_ff`, an
unexpectedly untied embedding — before it becomes a memory failure.

**Do not compute logits for all positions at inference.** It costs roughly as
much as the rest of the forward pass. Every serving stack does this correctly;
hand-rolled inference code frequently does not.

**Version the tokenizer, the template, and the weights together.** All three are
part of $f_\theta$ in practice even though only one is in the checkpoint
({{ch:mle-registry}}).

**Attribute faults to the right layer.** {{tbl:not-in-the-model}} is the
checklist. Before investigating a model's behaviour, establish whether the
behaviour is even a model property — most of the time it is a decoding or
template question, and those are far cheaper to fix.

## 11. Common Mistakes

**Beginners:**

*Believing temperature is a model property.* It is applied to the logits after
the model has finished. {{tbl:not-in-the-model}}.

*Thinking the model chooses a token.* It produces a distribution. Choosing is
{{ch:llm-decoding}} and happens outside $f_\theta$.

*Confusing the logit vector's length with the sequence length.* Logits are
$|V|$-dimensional — one per vocabulary item — not $T$-dimensional.

**Experienced practitioners:**

*Comparing models by parameter count alone.* `model-shape-comparison` shows
depth and KV configuration decide serving behaviour, and neither is in the
count.

*Omitting the final norm when writing a model from scratch.* The logit scale
becomes input-dependent, which is an uncontrolled temperature
{{eq:scale-is-temperature}} — the exact bug that broke
{{ch:fm-pretraining}}'s listing.

*Assuming the FFN is a minor component.* It is two-thirds of every block in
parameters and FLOPs.

*Forgetting that the template is part of the input.* A model does not know it
is in a conversation; it sees role markers that were inserted before
tokenization.

## 12. Failure Modes

**Shape mismatch on a modified model.** *Symptom:* a crash, if you are lucky; a
silently wrong result if a broadcast happens to succeed. *Detection:* the shape
assertions in `llm-forward-pass`. Broadcasting is what makes this dangerous —
numpy and torch will happily combine $(T, d)$ with $(d,)$.

**Missing or misplaced causal mask.** *Symptom:* excellent training loss and
incoherent generation, because the model learned to read the answer.
*Detection:* the `np.triu(weights, 1) == 0` assertion, and the generation check
from {{ch:tf-masking-kv}}.

**Uncontrolled logit scale.** *Symptom:* a model that behaves as though its
temperature is wrong and does not respond as expected to temperature changes.
*Cause:* missing final norm, or an unembedding whose scale drifted.

**Cache growth exceeding memory mid-conversation.** *Symptom:* failures that
correlate with conversation length rather than request rate. *Detection:* the
cache arithmetic in `model-shape-comparison`, computed at the maximum context
rather than the typical one.

**Configuration/weights mismatch.** A config claiming a different `d_ff` or head
count than the checkpoint. *Symptom:* loads successfully, produces nonsense.
*Detection:* the parameter-count check.

## 13. Alternatives

{#tbl:llm-architecture-variants caption="Variants on the standard decoder-only stack that change the trace in this chapter. The first three are near-universal now; the last two change the shape story materially."}

| Variant | Changes | Effect on this chapter's trace |
|---|---|---|
| Pre-norm ({{cite:vaswani2017}} post-norm originally) | norm position | stabilises deep stacks; the trace shown |
| RMSNorm ({{cite:zhang2019rmsnorm}}) | drops mean subtraction | cheaper, same shapes |
| SwiGLU ({{cite:shazeer2020glu}}) | 3 FFN matrices | FFN parameters rise 1.5x |
| RoPE ({{cite:su2021rope}}) | position applied in attention | no separate position embedding |
| Grouped-query attention | fewer KV heads | KV cache shrinks; parameters barely move |
| Mixture of experts | routed FFN | total ≫ active parameters; $2N$ identity breaks |

**Which change the function and which change the cost.** RMSNorm, SwiGLU and
GQA all alter the computed function. RoPE changes where position enters but not
the shape. **Mixture of experts is the one that breaks this chapter's
accounting**: with routed experts, {{eq:total-params}} counts parameters that
are not used per token, so the parameter count stops predicting FLOPs and
"model size" becomes ambiguous ({{ch:res-moe}}).

## 14. Evaluation

**Is the implementation correct?** Five checks, all cheap:

1. **Shapes** at every stage — the assertions in `llm-forward-pass`.
2. **Attention rows sum to 1** and are zero above the diagonal.
3. **Logits have length $|V|$**, not $T$ and not $d$.
4. **Parameter count matches the configuration** to within a few per cent.
5. **Loss at initialisation $\approx\log|V|$** ({{ch:fm-pretraining}}) — which,
   as {{sec:5-formal-explanation}} notes, is really a check on the logit scale
   and therefore on the final norm.

**Is the model behaving?** That question belongs to the rest of this part, and
the first step is always attribution: use {{tbl:not-in-the-model}} to establish
whether the behaviour under investigation is a model property at all. A
surprising number of model investigations end at that table.

## 15. Advanced Concepts

**The residual stream as an interpretability object.**
{{maturity:EMERGING}} {{eq:residual-stream-sum}} makes the final representation
a sum of contributions, so individual heads' and layers' contributions to a
logit can be attributed. This decomposition is the foundation of most mechanistic
interpretability work ({{part:27}}).

**Logit lens.** {{maturity:EMERGING}} Applying the unembedding to intermediate
layers' residual streams, to see what the model would predict if it stopped
early. Works because the stream lives in a consistent space throughout —
{{eq:residual-stream-sum}} again.

**Weight tying's tradeoff.** {{maturity:ESTABLISHED}}
{{cite:press2017tying}} saves $|V|d$ parameters and constrains the embedding and
unembedding to be transposes, which is a real restriction. Large models
increasingly untie, because the parameter saving stops mattering.

**Multi-token prediction.** {{maturity:EMERGING}} Predicting several future
tokens per position, changing {{eq:llm-as-function}}'s codomain. Motivated by
speculative decoding and by the observation that one-token-at-a-time is an
arbitrary granularity ({{ch:nlp-subword}}).

**Depth versus width at fixed parameters.** {{maturity:EMERGING}} The tradeoff
`model-shape-comparison` exposes. Deeper is more sequential and often more
capable per parameter; wider parallelises better. The optimum depends on serving
constraints as much as on quality.

## 16. Connection to Previous Chapters

**Backwards.** {{ch:tf-architectures}} built the stack this chapter traces, and
{{ch:tf-ffn-residual}}'s parameter split is {{eq:block-params}}.
{{ch:tf-masking-kv}}'s causal mask is asserted in `llm-forward-pass`, and its
KV-cache accounting is what `model-shape-comparison` computes.
{{ch:tf-complexity}}'s $2N$-per-token identity underlies every cost figure here.
{{ch:tf-embeddings}} supplied weight tying. {{ch:fm-pretraining}}'s
initial-loss failure is {{eq:scale-is-temperature}} in practice.
{{ch:fm-instruction-tuning}}'s template runs before the first line of this
chapter's trace.

**Forwards.** {{ch:llm-next-token}} takes {{eq:next-token-distribution}} and
asks what its probabilities mean. {{ch:llm-decoding}} is everything that happens
to the logit vector afterwards. {{ch:llm-inference}} follows the KV cache.
{{ch:llm-prompt-lifecycle}} wraps the whole trace in the serving path.
{{part:23}} builds the systems that run it, and {{part:27}} uses
{{eq:residual-stream-sum}} for interpretability.

## 17. Exercises

**Beginner**

1. State what an LLM is as a function, with domain and codomain.
2. For $T = 20$, $d = 512$, $|V| = 32000$: what are the shapes after embedding,
   after block 3, and of the logits?
3. Why is only the last position unembedded at inference?

**Intermediate**

4. Compute the parameter count for $L=24$, $d=2048$, $d_{\text{ff}}=5632$,
   $|V|=32000$, gated FFN, tied embeddings.
5. Using {{eq:scale-is-temperature}}, show that doubling the final norm's gain
   is equivalent to halving the temperature.
6. Two models have equal parameter counts; one is 24 layers of width 2048 and
   the other 48 of width 1448. Compare KV cache per token and sequential depth.

**Advanced**

7. Derive {{eq:residual-stream-sum}} and state the three consequences.
8. Prove that omitting the final norm makes the effective temperature
   input-dependent, and describe the symptom a user would report.
9. Explain why mixture of experts breaks {{eq:total-params}}'s usefulness, and
   propose what should be reported instead of "parameter count".

**Implementation**

10. Extend `llm-forward-pass` with a KV cache and assert that cached
    incremental decoding produces bit-identical logits to a full recompute.
11. Implement the logit lens: unembed every layer's residual stream and show how
    the top prediction evolves with depth.
12. Add RoPE to the attention function and verify that relative positions behave
    as {{ch:tf-positional}} describes.
13. Deliberately remove the final norm and measure how the logit scale varies
    with input length. Relate the spread to an equivalent temperature range.

**Reasoning**

14. A user reports the model is "too random". Enumerate every place in the
    pipeline that could cause it, in the order you would check them.
15. Explain why a model does not "know" it is in a conversation, and what makes
    it behave as though it does.

## 18. Interview Questions

**Beginner**

1. What does an LLM output?
2. What shape is the logit vector and why?
3. Where does temperature get applied?

**Intermediate**

4. Walk through a forward pass with shapes.
5. What is the residual stream and why does the shape never change?
6. Compute a model's parameter count from its config.

**Senior**

7. Two models are both called 7B. What do you need to know before choosing?
8. What is the final norm for, and what breaks without it?
9. A user says the model is behaving badly. How do you decide whether it is the
   model?

**Systems**

10. What would you log at model load time, and why?
11. Why is computing logits for all positions wasteful at inference but
    necessary at training?

## 19. Research Questions

**Is one-token-at-a-time the right granularity?**
{{eq:llm-as-function}}'s codomain is a distribution over single tokens, which is
inherited from {{ch:nlp-subword}}'s arbitrary segmentation. Multi-token
prediction changes it. Measure quality and speed against prediction width to
find where the tradeoff sits.

**What is the right depth/width ratio for a serving constraint?**
`model-shape-comparison` shows depth costs sequential latency and width costs
parallel FLOPs. Optimise for a latency target rather than for loss and see how
far the answer moves from the loss-optimal shape.

**How much does weight tying cost at scale?** It saves $|V|d$ and constrains the
two matrices to be transposes. Measure the quality cost as a function of model
size and locate the crossover where untying starts paying.

**Can the residual stream's contributions be attributed reliably?**
{{eq:residual-stream-sum}} makes the decomposition exact, but attributing a
*behaviour* to a term requires the terms to be independently meaningful, which
the nonlinearity between them undercuts. Characterising when the decomposition
is interpretable is largely open.

## 20. Chapter Summary

An LLM is a function from a token sequence to a vector of logits over the
vocabulary {{eq:llm-as-function}}, which a softmax turns into a distribution
over the next token {{eq:next-token-distribution}}. **That is the whole of what
the model does.** It does not choose a token, it has no temperature, and it has
no notion of a conversation beyond the role markers a template inserted into its
input.

The trace is short and every shape is fixed. Token IDs index the embedding
matrix to give $T\times d$; each of $L$ blocks reads the residual stream,
computes an attention or feed-forward contribution, and *adds* it back, leaving
the shape unchanged; a final normalisation is applied; and the last row alone is
multiplied by the unembedding matrix to give $|V|$ logits.

**The shape invariance is the important structural fact.**
{{eq:residual-stream-sum}} shows the final representation is the embedding plus
the sum of every sublayer's contribution — which is why gradients reach layer 1
directly, why blocks may be near-inert, and why the stream is a shared workspace
rather than a pipeline.

**The final norm is load-bearing rather than cosmetic.**
{{eq:scale-is-temperature}} shows that scaling logits by $c$ is exactly sampling
at temperature $1/c$, so without normalisation the effective temperature is set
by whatever the residual stream's magnitude happens to be. That is the bug that
broke {{ch:fm-pretraining}}'s initial-loss check.

**Parameter count is computable and is not the specification.**
{{eq:total-params}} matches published sizes to a few per cent, and the breakdown
matters more than the total: the FFN is two-thirds of every block, and the
embedding share falls from a fifth at GPT-2 small to under 4% at 13B. Meanwhile
`model-shape-comparison` shows two models of equal parameter count differing
sharply in KV-cache footprint and sequential depth — neither of which appears in
the count, and both of which decide serving behaviour.

Finally, {{tbl:not-in-the-model}}: temperature, context length, system-prompt
privilege, valid JSON, tool invocation, stopping, and streaming are all outside
$f_\theta$. Most complaints about model behaviour are complaints about one of
those, and the first debugging step in this entire part is deciding which.

## 21. Further Reading

{{cite:vaswani2017}} remains the reference for the block structure, though the
architecture in this chapter differs from it in three respects that all became
standard afterwards: pre-normalisation, RMSNorm, and a gated feed-forward
network. Reading it now is most useful as a way of seeing which of its choices
survived.

{{cite:touvron2023llama}}'s §2 is the clearest published description of a modern
configuration, and it is specific enough to check `parameter-accounting`
against — which is the exercise worth doing.

{{cite:radford2019}} for the decoder-only formulation and the observation that a
single next-token predictor serves as a general interface. Its §2 is two pages.

{{cite:press2017tying}} for weight tying, which is a short paper about a
one-line change with a measurable effect, and a good example of how much of this
architecture is accumulated small decisions rather than design.

**Where to go next:** {{ch:llm-next-token}} asks what the numbers in the logit
vector actually mean — whether a probability of 0.7 corresponds to being right
70% of the time, and what follows when it does not.

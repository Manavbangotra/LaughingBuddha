---
id: nlp-bert
number: 76
part: VIII
tier: full
status: draft
requires: [nlp-contextual, nlp-subword, tf-architectures, tf-multi-head,
           dl-losses, dl-lr-schedules, ml-metrics]
provides: [mlm-objective, masking-rate, mask-token-mismatch, next-sentence-prediction,
           sentence-order-prediction, cls-token, segment-embedding, roberta-recipe,
           dynamic-masking, replaced-token-detection, encoder-distillation,
           curse-of-multilinguality, benchmark-saturation]
citations: [devlin2019bert, liu2019roberta, lan2020albert, clark2020electra,
            sanh2019, conneau2020xlmr, wang2019glue, rajpurkar2016, peters2018,
            howard2018, radford2019, levy2015]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Derive the masked language modelling objective and compute its sample
   efficiency relative to causal language modelling.
2. Explain the `[MASK]` train/test mismatch precisely, state the 80/10/10 patch,
   and say what it does and does not fix.
3. Explain why 15% is the masking rate, and why the number is less principled
   than it appears.
4. State what {{cite:liu2019roberta}} changed and what its result implies about
   the encoder literature of 2019-2020.
5. Explain why next-sentence prediction failed, using the evidence from both
   RoBERTa and ALBERT.
6. Describe the BERT input format — `[CLS]`, `[SEP]`, segment embeddings — and
   what each element is for.
7. Explain why bidirectional encoders lost pretraining and won retrieval.

## 2. Why This Matters

**This chapter is where the encoder era's answers were fixed**, and every one of
them is still in production somewhere. Cross-encoder rerankers
({{ch:emb-reranking}}) are BERT. Embedding models ({{ch:emb-models}}) are BERT
descendants. The `[CLS]`-token convention, the segment embedding, and the
fine-tune-a-head recipe all originate here.

**It also contains the most useful methodological result in the part.**
{{cite:liu2019roberta}} changed no architecture — only training budget, data
volume, and two objective details — and reached state of the art. Its stated
conclusion is that BERT was significantly undertrained, and the implication is
uncomfortable: **a large share of the architecture papers published between BERT
and RoBERTa were measuring training budget while reporting architecture.** This
is the same failure {{cite:levy2015}} found in the embedding literature, and
{{part:9}} finds again at pretraining scale.

**Understanding why MLM lost pretraining explains why decoder-only models won.**
The reasons are specific and countable — sample efficiency, task alignment,
generation — and each is a number rather than a preference. Without them,
"decoder-only won" is a fact to memorise.

**And this is where a benchmark saturated in public.** GLUE
({{cite:wang2019glue}}) was built as a hard multi-task target and was
substantially solved within about a year. {{part:25}} is largely about not
repeating that, and it helps to have watched it happen once.

## 3. Prerequisites

{{ch:nlp-contextual}} for contextual representations, feature-based versus
fine-tuning transfer, and the concatenation-versus-joint-conditioning gap this
chapter closes. {{ch:nlp-subword}} for the WordPiece vocabulary BERT uses.
{{ch:tf-architectures}} for the encoder stack and bidirectional attention.
{{ch:tf-multi-head}} for what the layers compute. {{ch:dl-losses}} for
cross-entropy. {{ch:dl-lr-schedules}} for warmup, which BERT needs.
{{ch:ml-metrics}} for the benchmark aggregation discussion in
{{sec:14-evaluation}}.

## 4. Intuitive Explanation

{{ch:nlp-contextual}} left a specific gap. ELMo ran a left-to-right model and a
right-to-left model and glued their outputs together. No single computation in
that model ever saw both sides of a word at once — the concatenation happens
after the fact, so each half formed its opinion in ignorance of the other.

Why not simply train a bidirectional language model directly? Because the task
becomes trivial. If the objective is "predict word $t$" and the model can see
word $t$, it reads the answer. In a multi-layer bidirectional network the
information leaks upward through the stack: layer two's representation of
position $t-1$ already contains position $t$.

**BERT's answer is to delete the word being predicted.** Replace 15% of the
tokens with a `[MASK]` placeholder and train the model to reconstruct them from
everything else. Now the model may look in both directions freely, because what
it is looking for is not there.

That is masked language modelling, and it is a cloze test — the fill-in-the-blank
exercise used in language teaching for a century.

**The immediate objection is also correct.** The model now spends its training
life looking at `[MASK]` tokens, and `[MASK]` never appears at inference time.
This is a train/test distribution mismatch introduced deliberately by the
objective, and BERT patches it — 80% of the time the token is masked, 10% it is
replaced with a random word, 10% it is left alone — so the model cannot know
whether a given position needs correcting, and must therefore build a good
representation of every position.

> NOTE: The patch is a mitigation, not a fix. `[MASK]` still appears in 12% of
> tokens during training and 0% at inference. {{cite:clark2020electra}} removes
> the token entirely, which is a better answer, and it did not win — which is
> worth remembering when a cleaner method loses.

**And there is a cost nobody advertised.** Only the masked positions produce a
learning signal, so BERT learns from 15% of its tokens while a causal language
model learns from 100% of them. That factor of six or seven, compounded over a
pretraining run, is a large part of the reason the generative models scaled
further.

**The mental model:** MLM buys bidirectionality by destroying part of the input,
and the price is paid in sample efficiency and in a token that exists only during
training. That trade is excellent for encoding a document you already have and
poor for generating one you do not — which is the whole story of what happened
next.

## 5. Formal Explanation

### 5.1 The masked language modelling objective

Given a token sequence $\vec{x} = (x_1,\dots,x_T)$, sample a mask set
$M \subset \{1..T\}$ with $|M| \approx pT$, and let $\tilde{\vec{x}}$ be the
sequence with positions in $M$ corrupted. The objective is

$$
\Loss_{\text{MLM}} = -\sum_{i \in M}
 \log P\big(x_i \given \tilde{\vec{x}};\ \theta\big)
$$ (eq:mlm-objective)

Compare with the causal objective of {{ch:tf-architectures}}:

$$
\Loss_{\text{CLM}} = -\sum_{i=1}^{T} \log P\big(x_i \given x_{<i};\ \theta\big)
$$ (eq:clm-objective)

**Two differences, both consequential.** The conditioning set for MLM is all of
$\tilde{\vec{x}}$ — both sides — while CLM sees only the left. And the sum for
MLM runs over $|M| \approx 0.15T$ terms while CLM's runs over all $T$.

### 5.2 The corruption rule

For each $i \in M$, {{cite:devlin2019bert}} applies:

$$
\tilde{x}_i = \begin{cases}
 \texttt{[MASK]} & \text{with probability } 0.8\\
 u \sim \text{Uniform}(V) & \text{with probability } 0.1\\
 x_i & \text{with probability } 0.1
\end{cases}
$$ (eq:bert-corruption)

The loss is computed at all of $M$ regardless of which branch fired — including
the 10% where the token was left unchanged, where the model is asked to predict a
token it can see.

**Why the third branch exists.** If every masked position were replaced by
`[MASK]`, the model would learn that positions holding `[MASK]` need a
prediction and all others do not — so it would only build predictive
representations at mask positions. Leaving 10% unchanged means the model cannot
identify which positions are under test, and must represent every position well.
That is the property fine-tuning depends on, since fine-tuning uses all positions
and masks none.

**Why the second branch exists.** Random replacement forces the model to detect
that a token is wrong from its context, rather than only to fill an obvious hole.
It is a small dose of the denoising signal that {{cite:clark2020electra}} later
makes the entire objective.

### 5.3 Sample efficiency

Per sequence of $T$ tokens, MLM produces $pT$ prediction targets and CLM produces
$T$:

$$
\frac{\text{signal}_{\text{MLM}}}{\text{signal}_{\text{CLM}}} = p = 0.15
$$ (eq:mlm-sample-efficiency)

The forward and backward passes cost the same in both cases, so **MLM extracts
roughly one-seventh of the learning signal per unit of compute.**

This is the accounting {{cite:clark2020electra}} attacks directly by scoring
every position, and it is one concrete, quantified reason the generative side
scaled further. It is not the only reason — see {{sec:12-failure-modes}} — but it
is the one that is a number.

### 5.4 Choosing the masking rate

The rate $p$ trades two effects against each other:

- **Higher $p$**: more prediction targets per sequence, so more signal per unit
  of compute — but less context remains to predict from, so each prediction is
  harder and the representation of context degrades.
- **Lower $p$**: richer context, better per-prediction accuracy, less signal.

At $p = 1$ the task is unconditional generation with no context; at $p \to 0$ the
task is trivially easy and almost no learning occurs per sequence.

> RESEARCH NOTE: 15% was chosen empirically by {{cite:devlin2019bert}} and copied
> almost universally for years. Later work has trained successfully at
> substantially higher rates, particularly for larger models, which suggests the
> optimum depends on model capacity rather than being a constant of the task.
> Treat 15% as a well-tested default, not as a derived quantity — it belongs on
> the same list as {{eq:noise-distribution}}'s $3/4$ exponent.

### 5.5 The input format

BERT's input is a fixed construction:

$$
\texttt{[CLS]}\ x_1 \dots x_n\ \texttt{[SEP]}\ y_1 \dots y_m\ \texttt{[SEP]}
$$ (eq:bert-input)

and the embedding of position $i$ is a sum of three vectors:

$$
\vec{e}_i = \mat{E}^{\text{token}}_{x_i}
 + \mat{E}^{\text{segment}}_{s_i}
 + \mat{E}^{\text{position}}_{i}
$$ (eq:bert-embedding-sum)

**`[CLS]`** is a position with no token content whose final representation is
used as the sequence representation for classification. It is a convention rather
than a mechanism: nothing in {{eq:mlm-objective}} trains it to summarise
anything, which is why {{ch:nlp-similarity}} finds that mean pooling usually
beats it on an unfine-tuned model.

**`[SEP]`** delimits the two segments, and the **segment embedding** $s_i \in
\{0,1\}$ tells the model which side of the separator a token is on — necessary
for pair tasks such as entailment and question answering, where the same token
means different things in the premise and the hypothesis.

### 5.6 Next-sentence prediction, and its failure

BERT trains a second objective: given two segments, classify whether B follows A
in the corpus or was sampled at random.

$$
\Loss_{\text{BERT}} = \Loss_{\text{MLM}} + \Loss_{\text{NSP}}
$$ (eq:bert-total-loss)

The intent was to teach inter-sentence relationships for entailment and question
answering.

**Two papers took it apart.** {{cite:liu2019roberta}} removed NSP entirely and
matched or improved every downstream result. {{cite:lan2020albert}} explained
why: a randomly sampled segment B usually comes from a different document, so it
differs in *topic*, and topic mismatch is detectable from word overlap alone.
The task is solvable without any inter-sentence reasoning.

ALBERT's replacement, **sentence-order prediction**, uses two consecutive
segments and asks whether they have been swapped. Topic is now identical in both
classes, so the only available signal is discourse order — and this version
helps.

**The general lesson is worth more than the specific objective.** An auxiliary
task teaches what its *negatives* require you to distinguish. Get the negatives
wrong and you have built an easy task that trains nothing, while the loss curve
looks perfectly healthy.

## 6. Mathematical Foundation

### 6.1 Why bidirectional conditioning leaks without masking

Consider an $L$-layer bidirectional encoder with representations
$\vec{h}^{(\ell)}_i$, where every layer attends to all positions:

$$
\vec{h}^{(\ell)}_i = f\big(\vec{h}^{(\ell-1)}_1,\dots,\vec{h}^{(\ell-1)}_T\big)
$$ (eq:bidirectional-layer)

Define the dependency set $S^{(\ell)}_i$ as the set of input positions that
$\vec{h}^{(\ell)}_i$ depends on. Then $S^{(0)}_i = \{i\}$ and

$$
S^{(\ell)}_i = \bigcup_{j=1}^{T} S^{(\ell-1)}_j = \{1,\dots,T\}
 \quad\text{for all } \ell \ge 1
$$ (eq:dependency-closure)

**After a single bidirectional layer, every position depends on every input** —
including its own target. So predicting $x_i$ from $\vec{h}^{(L)}_j$ for any $j$
is reading the answer, and no amount of architectural care fixes it while $x_i$
is present in the input.

$\square$

This is why the token must be *removed from the input* rather than merely
excluded from some attention pattern. It is also why "just mask the diagonal"
does not work: the leak is transitive, and one layer closes it.

### 6.2 The effective batch size of MLM

Let $B$ be sequences per batch and $T$ tokens per sequence. The number of
gradient-contributing predictions is

$$
n_{\text{MLM}} = pBT, \qquad n_{\text{CLM}} = BT
$$ (eq:effective-predictions)

For BERT's configuration, $B = 256$, $T = 512$, $p = 0.15$:

$$
n_{\text{MLM}} = 0.15 \times 256\times 512 = 19{,}660
$$

against $131{,}072$ for a causal model at identical compute. Over BERT's
1,000,000 steps that is $1.97\times 10^{10}$ prediction targets — versus
$1.31\times 10^{11}$ for the causal model at the same cost.

**Same FLOPs, one-seventh the supervision.** This single line is the strongest
quantitative statement available about why the two objectives scaled differently.

### 6.3 What the corruption rule costs

The model sees `[MASK]` in $0.8p = 12\%$ of positions during pretraining and
$0\%$ during fine-tuning and inference. Treating the pretraining input
distribution as $\Data_{\text{pre}}$ and the fine-tuning one as
$\Data_{\text{fine}}$, the shift is

$$
\Prob_{\text{pre}}(\texttt{[MASK]}) = 0.12,
\qquad
\Prob_{\text{fine}}(\texttt{[MASK]}) = 0
$$ (eq:mask-distribution-shift)

The 80/10/10 rule reduces the *conditional* mismatch — the model cannot infer
from a position's contents whether it is a prediction target — but it does not
remove the marginal shift above. **The mismatch is structural to the objective**,
and only an objective that never introduces the token removes it, which is what
{{cite:clark2020electra}}'s replaced-token detection does.

### 6.4 A worked masking example

Sequence of 20 tokens, $p = 0.15$ → 3 masked positions. Under
{{eq:bert-corruption}} the expected split across those 3 is
$(2.4,\ 0.3,\ 0.3)$: about two `[MASK]`, and roughly one position in three
sequences gets a random replacement or is left unchanged.

The loss {{eq:mlm-objective}} is a sum over exactly those 3 positions. The other
17 tokens do the work of providing context and receive no direct supervision.

At initialisation, with $|V| = 30{,}522$, the model predicts uniformly and

$$
\Loss_{\text{MLM}} = -\log\frac{1}{30522} = 10.33
$$ (eq:mlm-baseline-loss)

**This is the number to check first in any MLM training run.** A loss starting
near 10.3 and falling means the setup is correct. A loss starting elsewhere means
the vocabulary size, the label alignment, or the loss masking is wrong — and
those bugs otherwise present as a model that simply trains poorly.

## 7. Internal Mechanics

```mermaid {#fig:mlm-mechanics caption="One masked-language-modelling step. Only the masked positions contribute to the loss — the other 85% of positions do the work of providing context and receive no direct supervision, which is the sample-inefficiency term of equation (eq:mlm-sample-efficiency)."}
graph TD
  A["the cat sat on the mat"] --> B["sample 15% of positions"]
  B --> C["80%: → [MASK]<br/>10%: → random token<br/>10%: → unchanged"]
  C --> D["the [MASK] sat on the mat"]
  D --> E["embedding sum<br/>token + segment + position"]
  E --> F["L bidirectional encoder layers<br/>every position sees every other"]
  F --> G["MLM head at masked positions only<br/>softmax over |V|"]
  G --> H["cross-entropy against 'cat'"]
  F --> I["all other positions:<br/>no loss term"]
  style I fill:#fdd,stroke:#c66
  style H fill:#dfe,stroke:#5a5
```

**The MLM head and weight tying.** The output projection maps $\R^d \to \R^{|V|}$
and is normally tied to the input embedding matrix ({{ch:tf-embeddings}}), which
saves $|V|d$ parameters — 23M of BERT-base's 110M at $|V|=30{,}522$, $d=768$.
The head is discarded after pretraining; only the encoder is fine-tuned.

**Static versus dynamic masking.** BERT masked its corpus once during
preprocessing, duplicating the data ten times with different masks so a sequence
was seen with ten patterns across 40 epochs — meaning each mask pattern was seen
four times. {{cite:liu2019roberta}} masks freshly every time a sequence is drawn,
so no pattern repeats. **Dynamic masking is free** — it costs a random number
generator call — and it is one of the changes that made RoBERTa better.

**Where the compute goes.** For BERT-base, 110M parameters at $T=512$: the
encoder dominates, and the MLM head is applied at only 15% of positions, so the
$|V|$-wide softmax that would dominate a causal model's output layer is cheap
here. This is the one place where MLM's sparsity helps rather than hurts.

**Why warmup is required.** BERT uses 10,000 warmup steps
({{ch:dl-lr-schedules}}). Without it, early large gradients through a deep
pre-norm-less stack destabilise training — the same reason
{{cite:howard2018}}'s slanted triangular schedule exists.

## 8. Implementation

The masking function, implemented exactly and verified against the probabilities
in {{eq:bert-corruption}}:

```python {tier=A name=mlm-masking}
"""BERT's 80/10/10 corruption rule, and the property each branch buys."""
import numpy as np

rng = np.random.default_rng(0)

VOCAB = ["[PAD]", "[CLS]", "[SEP]", "[MASK]"] + [f"w{i}" for i in range(96)]
MASK_ID, SPECIAL = 3, {0, 1, 2, 3}
V, MASK_RATE = len(VOCAB), 0.15
IGNORE = -100          # positions excluded from the loss


def mask_tokens(ids, rng):
    """Returns the corrupted input and the label array. Equation (eq:bert-corruption)."""
    ids = np.asarray(ids)
    labels = np.full_like(ids, IGNORE)

    maskable = np.array([i for i, t in enumerate(ids) if t not in SPECIAL])
    n = max(1, int(round(MASK_RATE * len(maskable))))
    chosen = rng.choice(maskable, size=n, replace=False)

    labels[chosen] = ids[chosen]           # the loss is computed only here
    out = ids.copy()

    draw = rng.random(n)
    out[chosen[draw < 0.8]] = MASK_ID                                   # 80%
    mid = chosen[(draw >= 0.8) & (draw < 0.9)]
    out[mid] = rng.integers(len(SPECIAL), V, size=len(mid))             # 10%
    # the remaining 10% keep their original token, and are still labelled
    return out, labels


seq = [1] + list(rng.integers(4, V, size=18)) + [2]
corrupted, labels = mask_tokens(seq, rng)

print("original :", " ".join(VOCAB[t] for t in seq))
print("corrupted:", " ".join(VOCAB[t] for t in corrupted))
print("supervised positions:", int((labels != IGNORE).sum()), "of", len(seq))

# Verify the branch probabilities over many draws.
counts = {"mask": 0, "random": 0, "unchanged": 0, "total": 0}
for _ in range(4000):
    s = [1] + list(rng.integers(4, V, size=48)) + [2]
    c, l = mask_tokens(s, rng)
    for i in np.flatnonzero(l != IGNORE):
        counts["total"] += 1
        if c[i] == MASK_ID:
            counts["mask"] += 1
        elif c[i] == l[i]:
            counts["unchanged"] += 1
        else:
            counts["random"] += 1

t = counts["total"]
print(f"\nover {t:,} masked positions:")
for k, target in [("mask", 0.80), ("random", 0.10), ("unchanged", 0.10)]:
    print(f"  {k:<10} {counts[k] / t:6.3f}   (target {target:.2f})")

assert abs(counts["mask"] / t - 0.80) < 0.02
assert abs(counts["unchanged"] / t - 0.10) < 0.02

# The 12% figure from equation (eq:mask-distribution-shift).
print(f"\n[MASK] occupies {0.8 * MASK_RATE:.0%} of tokens in pretraining "
      f"and 0% at inference — the mismatch the rule mitigates but cannot remove.")
```

Now a complete MLM training loop, small enough to run in seconds and large enough
to show the loss falling below the uniform baseline of
{{eq:mlm-baseline-loss}}:

```python {tier=A name=mlm-training-loop}
"""A tiny masked language model. The loss must start near log|V| and fall."""
import math
import torch
import torch.nn as nn

torch.manual_seed(0)

# A structured toy language: each sentence is SUBJ VERB OBJ, drawn from three
# domains whose words never mix. Reconstructing a masked token therefore
# requires reading the rest of the sentence — which is the point of the task.
DOMAINS = {
    "med": (["doctor", "nurse", "surgeon"], ["examined", "treated", "admitted"],
            ["patient", "child", "athlete"]),
    "tech": (["engineer", "developer", "architect"], ["deployed", "debugged", "refactored"],
             ["service", "pipeline", "database"]),
    "food": (["chef", "baker", "cook"], ["seasoned", "baked", "plated"],
             ["dish", "bread", "dessert"]),
}
SENTENCES = [[s, v, o] for subj, vb, ob in DOMAINS.values()
             for s in subj for v in vb for o in ob]

words = sorted({w for s in SENTENCES for w in s})
VOCAB = ["[MASK]"] + words
idx = {w: i for i, w in enumerate(VOCAB)}
V, D, T = len(VOCAB), 32, 3
MASK_ID, IGNORE = 0, -100

X = torch.tensor([[idx[w] for w in s] for s in SENTENCES])
print(f"{len(SENTENCES)} sentences, {V} vocabulary items")
print(f"uniform-prediction loss = log|V| = {math.log(V):.4f}   "
      f"<- equation (eq:mlm-baseline-loss)")


class TinyEncoder(nn.Module):
    """A one-layer bidirectional transformer encoder with a tied MLM head."""

    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(V, D)
        self.pos = nn.Embedding(T, D)
        self.attn = nn.MultiheadAttention(D, num_heads=4, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.ff = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))

    def forward(self, x):
        h = self.tok(x) + self.pos(torch.arange(x.shape[1]))
        a, _ = self.attn(h, h, h, need_weights=False)     # no mask: bidirectional
        h = self.norm1(h + a)
        h = self.norm2(h + self.ff(h))
        return h @ self.tok.weight.T                      # weight tying


def corrupt(x, gen):
    """Mask exactly one position per sentence — 1/3, since T = 3."""
    labels = torch.full_like(x, IGNORE)
    pos = torch.randint(0, T, (x.shape[0],), generator=gen)
    rows = torch.arange(x.shape[0])
    labels[rows, pos] = x[rows, pos]
    out = x.clone()
    draw = torch.rand(x.shape[0], generator=gen)
    out[rows[draw < 0.8], pos[draw < 0.8]] = MASK_ID
    rand = (draw >= 0.8) & (draw < 0.9)
    out[rows[rand], pos[rand]] = torch.randint(1, V, (int(rand.sum()),), generator=gen)
    return out, labels


gen = torch.Generator().manual_seed(0)
model = TinyEncoder()
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

for step in range(1, 601):
    inp, lab = corrupt(X, gen)
    logits = model(inp)
    loss = nn.functional.cross_entropy(
        logits.reshape(-1, V), lab.reshape(-1), ignore_index=IGNORE)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 150 == 0 or step == 1:
        print(f"step {step:>4}: MLM loss {loss.item():.4f}")

# Does the model use both sides of the mask?
model.eval()
with torch.no_grad():
    probes = [["[MASK]", "examined", "patient"],
              ["engineer", "[MASK]", "pipeline"],
              ["chef", "seasoned", "[MASK]"]]
    print()
    for p in probes:
        x = torch.tensor([[idx[w] for w in p]])
        out = model(x)[0, p.index("[MASK]")]
        top = torch.topk(out.softmax(-1), 3)
        preds = [(VOCAB[i], round(float(v), 3)) for v, i in zip(top.values, top.indices)]
        print(f"{' '.join(p):<34} -> {preds}")

print("\nThe first probe masks position 0 and is only solvable by reading to "
      "the RIGHT — which a causal model could not do.")
```

The first probe is the demonstration: the mask is at position 0, so every clue is
to its right. A left-to-right model has nothing to condition on there, and the
bidirectional encoder recovers the domain from the words that follow.

Finally, the sample-efficiency accounting of {{eq:effective-predictions}}, which
is the argument rather than an intuition:

```python {tier=A name=mlm-vs-clm-efficiency}
"""Same compute, different amounts of supervision."""

BATCH, SEQ_LEN, MASK_RATE, STEPS = 256, 512, 0.15, 1_000_000
VOCAB = 30_522
PARAMS = 110e6

tokens_per_step = BATCH * SEQ_LEN
mlm_targets = MASK_RATE * tokens_per_step
clm_targets = tokens_per_step

# Compute is 6ND per token for training (see ch:tf-complexity) and is the same
# for both objectives — the masking changes the loss, not the forward pass.
flops_per_step = 6 * PARAMS * tokens_per_step

print(f"{'objective':<12} {'targets/step':>13} {'targets total':>16} "
      f"{'PFLOPs total':>14} {'targets/PFLOP':>15}")
for name, targets in [("MLM (15%)", mlm_targets), ("causal LM", clm_targets)]:
    total = targets * STEPS
    pflops = flops_per_step * STEPS / 1e15
    print(f"{name:<12} {targets:>13,.0f} {total:>16.3e} "
          f"{pflops:>14,.0f} {total / pflops:>15,.0f}")

print(f"\nsupervision ratio: {clm_targets / mlm_targets:.2f}x in favour of causal LM")
print(f"equation (eq:mlm-sample-efficiency) predicts 1/{MASK_RATE:.2f} = "
      f"{1 / MASK_RATE:.2f}x — the same number")

# What ELECTRA changes: score every position, not 15% of them.
print(f"\nELECTRA scores all {tokens_per_step:,} positions per step rather than "
      f"{mlm_targets:,.0f} — which is exactly this ratio, recovered.")
```

## 9. Practical Example

A team is choosing a pretrained encoder for a support-ticket classifier serving
30 requests per second with a 50 ms budget. Four candidates: BERT-base,
RoBERTa-base, DistilBERT, and a multilingual model, since 30% of tickets are not
in English.

The interesting part is that the ranking on GLUE does not decide it. Latency and
the multilingual tradeoff do, and both are quantifiable in advance.

```python {tier=A name=encoder-selection}
"""Encoder selection under a latency budget, with the multilingual tradeoff priced."""

BUDGET_MS, RPS, NON_ENGLISH = 50.0, 30, 0.30

# Representative figures, not verified benchmark results: parameter counts are
# exact, GLUE scores are approximate, and latencies stand in for a 128-token
# sequence on one CPU core. Substitute your own measurements before deciding
# anything — the listing is the shape of the argument, not a source of numbers.
CANDIDATES = {
    "BERT-base":      dict(params=110e6, layers=12, latency_ms=42, glue=79.6, multi=False),
    "RoBERTa-base":   dict(params=125e6, layers=12, latency_ms=44, glue=83.2, multi=False),
    "DistilBERT":     dict(params=66e6,  layers=6,  latency_ms=21, glue=77.0, multi=False),
    "XLM-R-base":     dict(params=270e6, layers=12, latency_ms=48, glue=80.4, multi=True),
}

print(f"{'model':<15} {'params':>9} {'ms':>6} {'GLUE':>6} {'fits 50ms':>10} "
      f"{'cores@30rps':>12} {'non-EN':>8}")
for name, c in CANDIDATES.items():
    fits = c["latency_ms"] < BUDGET_MS
    cores = RPS * c["latency_ms"] / 1000
    print(f"{name:<15} {c['params'] / 1e6:>8.0f}M {c['latency_ms']:>6} "
          f"{c['glue']:>6.1f} {str(fits):>10} {cores:>12.1f} "
          f"{('yes' if c['multi'] else 'no'):>8}")

print()
best_glue = max(CANDIDATES, key=lambda k: CANDIDATES[k]["glue"])
cheapest = min(CANDIDATES, key=lambda k: CANDIDATES[k]["latency_ms"])
print(f"best GLUE:     {best_glue} ({CANDIDATES[best_glue]['glue']})")
print(f"lowest latency:{cheapest} ({CANDIDATES[cheapest]['latency_ms']} ms, "
      f"{CANDIDATES[best_glue]['glue'] - CANDIDATES[cheapest]['glue']:.1f} GLUE points behind)")

# The 30% of non-English traffic cannot be served by a monolingual model at all.
english_only_coverage = 1 - NON_ENGLISH
print(f"\nA monolingual model serves {english_only_coverage:.0%} of traffic. "
      f"The remaining {NON_ENGLISH:.0%} needs either XLM-R or a second model.")

# Two-model cascade: DistilBERT for English, XLM-R for the rest.
cascade_ms = english_only_coverage * CANDIDATES["DistilBERT"]["latency_ms"] \
    + NON_ENGLISH * CANDIDATES["XLM-R-base"]["latency_ms"]
print(f"cascade (DistilBERT for EN, XLM-R otherwise): "
      f"{cascade_ms:.1f} ms weighted mean, "
      f"{RPS * cascade_ms / 1000:.1f} cores — versus "
      f"{RPS * CANDIDATES['XLM-R-base']['latency_ms'] / 1000:.1f} for XLM-R alone.")
print("\nTwo models cost more to operate than one. That is the real decision, "
      "and no benchmark column contains it.")
```

**The benchmark column is the least decisive number on the table.** The 6-point
GLUE spread across these models is smaller than the effect of a week of labelling
work, and the latency spread is a factor of two that shows up in the
infrastructure bill every month.

> PRODUCTION TIP: {{cite:sanh2019}} exists precisely for this situation, and
> distillation should be evaluated before a larger model is accepted on
> benchmark grounds. A 40% smaller and 60% faster model that gives up two GLUE
> points is usually the right trade for a classifier, and usually the wrong one
> for a reranker where the quality is the product.

## 10. Production Considerations

**Encoders are cheap to serve and that is why they are still here.** A
distilled encoder classifies at single-digit milliseconds on a CPU. An LLM call
for the same classification is two orders of magnitude more expensive and adds
network latency. At volume this decides architecture, and it is why
{{ch:nlp-extraction}}'s comparison is not a historical footnote.

**Fine-tuned models multiply, encoders do not.** Each fine-tuned copy is a full
set of weights to store, version, and monitor. If you have many tasks, either
feature-based transfer ({{ch:nlp-contextual}}) or the adapter methods of
{{part:14}} are what keeps the footprint manageable.

**Sequence length is the cost variable.** Attention is $O(T^2)$
({{ch:tf-complexity}}), and BERT's 512-token limit is a hard architectural bound
from its learned positional embeddings — not a configuration value. Longer
documents must be chunked, which is the problem {{ch:rag-chunking}} treats
properly.

**Multilinguality is bought with capacity.** {{cite:conneau2020xlmr}} documents
the tradeoff directly: at fixed model size, adding languages helps low-resource
ones and hurts high-resource ones. Serving one multilingual model is
operationally simpler and per-language weaker than serving several monolingual
ones.

**What to monitor:** p99 latency by input length, the truncation rate at 512
tokens, the class distribution of predictions over time (drift shows up here
before it shows up in labelled metrics), and — for fine-tuned models —
performance on a general held-out set that fine-tuning never touched.

## 11. Common Mistakes

**Beginners:**

*Using `[CLS]` as a sentence embedding without fine-tuning.* Nothing in
{{eq:mlm-objective}} trains `[CLS]` to summarise anything; it becomes a summary
only when a fine-tuning objective makes it one. Mean pooling is the better
default ({{ch:nlp-similarity}}).

*Computing the loss at unmasked positions.* The label array must be `IGNORE`
everywhere outside the mask set. Getting this wrong produces a model that trains
to copy its input, and the loss curve looks excellent while doing so.

*Expecting BERT to generate text.* It has no autoregressive factorisation. It can
fill masks, and that is not generation.

**Experienced practitioners:**

*Static masking.* Masking once during preprocessing wastes the free variance that
dynamic masking provides. This is one of RoBERTa's changes and it costs nothing
to adopt.

*Comparing pretrained models without equalising training budget.* This is
{{cite:liu2019roberta}}'s whole point, and it is
{{cite:levy2015}}'s point restated at a larger scale. An architectural comparison
across different budgets measures the budgets.

*Copying 15% without thinking.* It is a well-tested default chosen empirically,
and the optimum appears to depend on model size. Treat it as a hyperparameter you
have inherited rather than one that has been settled.

*Keeping next-sentence prediction.* Removing it is free and slightly better.
Anything derived from the original BERT recipe should drop it.

*Fine-tuning with no general held-out set.* Catastrophic forgetting is invisible
in the fine-tuning metrics by construction.

## 12. Failure Modes

**The `[MASK]` mismatch.** The token appears in 12% of pretraining positions and
never at inference {{eq:mask-distribution-shift}}. *Symptom:* a gap between
pretraining loss quality and fine-tuned performance that is smaller than expected
from the loss. *Structural:* it cannot be removed within MLM, only mitigated.

**NSP-style easy auxiliary tasks.** A task whose negatives are separable by a
shortcut trains the shortcut. *Symptom:* the auxiliary loss falls quickly to near
zero and downstream performance does not improve. *Detection:* try to solve the
auxiliary task with a bag-of-words baseline — if that works, the task is teaching
nothing the model needed a transformer for.

**Length truncation at 512.** BERT's positional embeddings are learned for 512
positions and do not extrapolate ({{ch:tf-positional}}). *Symptom:* systematically
worse performance on long documents, silently.

**Capacity dilution in multilingual models.** {{cite:conneau2020xlmr}}'s curse of
multilinguality: at fixed size, per-language quality falls as languages are
added. *Symptom:* a multilingual model underperforming a monolingual one on the
high-resource language it was supposed to also serve.

**Benchmark saturation.** {{cite:wang2019glue}} was built to be hard and was
substantially solved within about a year, at which point differences between
models on it stopped being informative. *Symptom:* a leaderboard whose top
entries are separated by less than the run-to-run variance. *This is not a model
failure*; it is an evaluation failure, and {{part:25}} is about avoiding it.

**Fine-tuning instability on small datasets.** BERT fine-tuning on a few thousand
examples has high variance across random seeds — different seeds give materially
different results. *Detection:* fine-tune with several seeds and report the
spread. A single-seed result on a small dataset is not a measurement.

## 13. Alternatives

{#tbl:pretraining-objectives caption="Pretraining objectives for text, by what fraction of positions produce learning signal and what the resulting model can do. The last column is the honest reason each survives or does not."}

| Objective | Signal per token | Bidirectional | Can generate | Where it is used now |
|---|---|---|---|---|
| MLM {{cite:devlin2019bert}} | 15% | yes | no | embeddings, rerankers |
| MLM, no NSP {{cite:liu2019roberta}} | 15% | yes | no | the encoder default |
| Sentence-order + MLM {{cite:lan2020albert}} | 15% | yes | no | rarely, as a diagnostic |
| Replaced-token detection {{cite:clark2020electra}} | **100%** | yes | no | a minority choice |
| Causal LM {{cite:radford2019}} | 100% | no | **yes** | everything generative |
| Distillation {{cite:sanh2019}} | teacher-defined | inherits | inherits | deployment |

**Which compute the same function.** The first three differ only in auxiliary
objective and training budget; RoBERTa is BERT with the budget corrected.
Replaced-token detection is a genuinely different objective that fixes MLM's
biggest measured inefficiency — and lost anyway, which is the entry worth
sitting with.

**Why causal LM won pretraining despite being unidirectional.** Three reasons,
in order of weight: it supervises every position; its objective is the task
itself, so no fine-tuning is needed for generation; and generation turned out to
be a general interface to every other task. Bidirectionality is a real advantage
for encoding and it lost to those three.

**Why encoders kept retrieval.** For turning a fixed document into one vector,
both sides of every token are available and there is nothing to generate. The
argument that beat MLM at pretraining does not apply, and
{{ch:nlp-similarity}} shows the architecture that follows.

## 14. Evaluation

**Is the pretraining implementation correct?** Four checks, all cheap:

1. **Initial loss $\approx \log|V|$** {{eq:mlm-baseline-loss}}. Any other
   starting value indicates a label-alignment or vocabulary bug.
2. **Masking rate and branch proportions** match {{eq:bert-corruption}} —
   measured, as in `mlm-masking`, not assumed.
3. **The loss is computed only at masked positions.** Test by masking nothing and
   confirming the loss is undefined or zero, not small.
4. **Bidirectionality is real** — mask position 0 and confirm the model does
   better than chance, which requires reading rightward.

**Is the pretrained model good?**

1. **Downstream fine-tuning on your task**, several seeds, with the spread
   reported.
2. **Linear probing per layer** ({{ch:nlp-contextual}}) if using it
   feature-based.
3. **Benchmark scores as a weak prior only.** GLUE-style aggregates average over
   tasks with different sizes and metrics, which makes the aggregate hard to
   interpret and easy to game — and a saturated benchmark carries almost no
   information about the differences you care about.

**The methodological rule this part keeps returning to:** equalise the budget
before comparing. {{cite:liu2019roberta}} is what happens when someone finally
does.

## 15. Advanced Concepts

**Replaced-token detection.** {{maturity:ESTABLISHED}} {{cite:clark2020electra}}
uses a small generator to corrupt tokens and trains the encoder to classify every
position as original or replaced. Signal at 100% of positions, no `[MASK]` at
inference, and substantially better compute efficiency — a better idea that
remains a minority choice.

**Parameter sharing.** {{maturity:EMERGING}} {{cite:lan2020albert}} shares
parameters across layers and factorises the embedding matrix, decoupling
vocabulary width from hidden width. Reduces parameters without reducing compute,
which is the wrong axis for most deployments — and is why it stayed niche.

**Distillation.** {{maturity:MATURE}} {{cite:sanh2019}} trains a 6-layer student
against BERT's outputs, retaining most performance at 40% fewer parameters and
60% faster inference. The standard deployment step for encoders, treated
generally in {{part:14}}.

**Cross-lingual transfer.** {{maturity:ESTABLISHED}} {{cite:conneau2020xlmr}}
trains one MLM over a hundred languages, achieving transfer to languages with
almost no labelled data — at the cost of the capacity dilution that
{{sec:12-failure-modes}} describes.

**Encoders inside generative systems.** {{maturity:ESTABLISHED}} The most common
deployment of BERT today is not classification. It is retrieval — bi-encoders for
recall and cross-encoders for precision ({{part:11}}, {{part:12}}) — inside
systems whose visible component is a decoder-only model.

## 16. Connection to Previous Chapters

**Backwards.** {{ch:nlp-contextual}} identified the gap this chapter closes:
ELMo's two directional models are concatenated, and {{eq:dependency-closure}}
shows why joint bidirectional training requires the target to be removed from the
input rather than merely masked in attention. {{ch:tf-architectures}} supplied the
encoder stack and the bidirectional attention pattern. {{ch:nlp-subword}} supplied
the WordPiece vocabulary. {{ch:dl-losses}} supplied the cross-entropy that
{{eq:mlm-objective}} sums. {{ch:dl-lr-schedules}} supplied the warmup that BERT
requires. {{ch:tf-embeddings}} supplied the weight tying that makes the MLM head
nearly free.

**Forwards.** {{ch:nlp-extraction}} fine-tunes this encoder for token-level
tasks. {{ch:nlp-similarity}} fixes what BERT does badly out of the box — sentence
vectors — and produces the bi-encoder that {{part:11}} builds on. {{part:9}}
takes the pretraining question to scale and finds {{cite:liu2019roberta}}'s
lesson again in {{ch:fm-scaling-laws}}. {{part:14}} generalises fine-tuning to
methods that do not copy the whole model. {{part:25}} treats the benchmark
saturation observed here as the central problem of evaluation.

## 17. Exercises

**Beginner**

1. Why can a bidirectional model not be trained with the objective of
   {{eq:clm-objective}}?
2. State the three branches of {{eq:bert-corruption}} and what each is for.
3. What is `[CLS]` and what trains it to be useful?

**Intermediate**

4. Compute the number of supervised predictions in a BERT pretraining run with
   $B=512$, $T=256$, $p=0.15$, 500k steps. Compare against a causal model at the
   same compute.
5. Explain, with reference to {{eq:dependency-closure}}, why masking the
   attention diagonal does not substitute for removing the token.
6. NSP was removed with no loss. Give ALBERT's explanation and the experiment
   that supports it.

**Advanced**

7. Derive the optimal masking rate under a stated model of the
   signal-versus-context tradeoff, and say which of your assumptions is doing the
   work.
8. {{eq:mask-distribution-shift}} shows the mismatch is structural. Prove that no
   corruption rule that ever inserts a special token can eliminate it, and state
   what that implies about ELECTRA.
9. Argue whether RoBERTa's result means the 2019-2020 encoder literature was
   uninformative, or something weaker. Be precise about what it does and does
   not establish.

**Implementation**

10. Extend `mlm-training-loop` with next-sentence prediction, then with
    sentence-order prediction, and measure how quickly each auxiliary loss falls.
    Relate the difference to ALBERT's diagnosis.
11. Sweep the masking rate over $\{0.05, 0.15, 0.30, 0.50\}$ and plot final MLM
    loss and probe accuracy on a downstream task against it.
12. Implement dynamic masking and compare against static masking at an equal
    number of steps, holding everything else fixed.
13. Implement replaced-token detection: a small generator fills masks, and the
    main model classifies every position as original or replaced. Compare
    convergence per step against MLM.

**Reasoning**

14. A colleague proposes MLM pretraining for a code-completion model. Explain
    what is wrong, in terms of the objective rather than by appeal to convention.
15. Encoders lost pretraining and kept retrieval. Explain both halves from the
    properties of {{eq:mlm-objective}}.

## 18. Interview Questions

**Beginner**

1. What is masked language modelling?
2. Why does BERT use `[MASK]`, and what problem does it create?
3. What is the difference between BERT and GPT in one sentence?

**Intermediate**

4. Explain the 80/10/10 rule and the purpose of each branch.
5. What did RoBERTa change and what did it establish?
6. Why did next-sentence prediction not work?

**Senior**

7. Would you pretrain an encoder today? Under what circumstances?
8. How would you choose between BERT-base, DistilBERT, and a multilingual model
   for a production classifier?
9. Explain why MLM is sample-inefficient and what ELECTRA does about it.

**Systems**

10. Design the serving architecture for twenty text classifiers built from one
    pretrained encoder.
11. Your fine-tuned classifier degrades over six months with no code change.
    Enumerate causes and the monitoring that would have caught each.

## 19. Research Questions

**What is the optimal masking rate as a function of model size?** 15% is
inherited. Later work trains successfully at much higher rates, particularly for
large models. Run the sweep at three model sizes and see whether the optimum
moves — the experiment is small and the answer is genuinely useful.

**Why did ELECTRA lose?** It fixes MLM's most-cited inefficiency, is more
compute-efficient, and removes the `[MASK]` mismatch. It is not the default.
Determining whether this is a real limitation, a timing accident, or an ecosystem
effect is a question about how the field adopts methods, and it is answerable.

**How much of the 2019-2020 encoder literature survives budget equalisation?**
{{cite:liu2019roberta}} did this for one model. Do it for five architectural
claims from that period at matched budget and report what remains. A negative
result would be more valuable than most positive ones.

**Is bidirectional pretraining still worth revisiting at scale?** The argument
against it was sample efficiency and the inability to generate. Both have
changed: compute is far cheaper, and hybrid objectives exist. Nobody has run the
large-scale bidirectional experiment with a modern budget, and the reasons are
partly sociological.

## 20. Chapter Summary

Masked language modelling makes joint bidirectional pretraining possible by
deleting what the model must predict. {{eq:dependency-closure}} shows why nothing
weaker works: after a single bidirectional layer every position depends on every
input, so the target must be removed from the input rather than hidden inside the
computation.

The objective {{eq:mlm-objective}} has two costs, both quantified.
{{eq:mlm-sample-efficiency}} shows that only 15% of positions produce learning
signal, so at equal compute MLM extracts roughly one-seventh of the supervision a
causal objective does. {{eq:mask-distribution-shift}} shows that `[MASK]` occupies
12% of pretraining tokens and none at inference; the 80/10/10 rule
{{eq:bert-corruption}} removes the model's ability to identify prediction targets
but cannot remove the marginal shift.

**{{cite:liu2019roberta}} changed no architecture and reached state of the art**
by training longer on more data with dynamic masking and no NSP. Its finding —
BERT was significantly undertrained — implies that a large share of the
contemporaneous architecture literature was measuring training budget. That is
the same failure {{cite:levy2015}} found in the embedding literature, and the same
one {{ch:fm-scaling-laws}} finds at pretraining scale.

**Next-sentence prediction failed because its negatives were too easy.** A random
segment differs in topic, and topic mismatch is detectable from word overlap.
{{cite:lan2020albert}}'s sentence-order prediction holds topic constant and
works. The transferable lesson is that an auxiliary task teaches exactly what
distinguishing its negatives requires.

Encoders lost pretraining to causal models on sample efficiency, task alignment,
and the ability to generate — and kept retrieval, where none of those arguments
applies and bidirectional encoding of a fixed document is exactly the right job.
That division is why this chapter's models are still running in production
underneath systems whose visible component is a decoder.

## 21. Further Reading

{{cite:devlin2019bert}} is worth reading for §3.1 and the ablations in §5.1. The
ablations are the more interesting half: they are the paper arguing with itself
about which of its components matter, and read next to
{{cite:liu2019roberta}} they show how hard that question is to answer without
budget control.

{{cite:liu2019roberta}} is the most useful paper in this chapter and the least
glamorous. It is a replication study. Read §3 and §4 as a template for how to
compare two systems honestly, then notice that this is a rarer thing to publish
than a new architecture.

{{cite:clark2020electra}} for the objective that fixes MLM's inefficiency. Read
it asking why it is not the default, and treat that as a live question rather
than a settled one.

{{cite:lan2020albert}} for §3's sentence-order argument specifically. The
parameter-sharing contributions matter less than the diagnosis of why NSP was
useless.

{{cite:wang2019glue}} is best read backwards from today, knowing it saturated. Its
introduction argues the benchmark is hard and will remain informative, and it is
a short and useful lesson in how confident that kind of claim usually is.

**Where to go next:** {{ch:nlp-extraction}} puts this encoder to work on the
token-level tasks the encoder era was built for, and asks honestly whether an LLM
should be doing them instead.

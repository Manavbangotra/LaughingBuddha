---
id: fm-pretraining
number: 80
part: IX
tier: full
status: draft
requires: [fm-what-they-are, nlp-bert, tf-architectures, tf-complexity,
           dl-optimizers, dl-lr-schedules, dl-normalization, mle-reproducibility]
provides: [causal-language-modelling, pretraining-run, loss-spike, checkpoint-restart,
           data-order, token-budget, unigram-baseline,
           training-instability, curriculum, packing]
citations: [brown2020, radford2019, touvron2023llama, hoffmann2022chinchilla,
            gao2020pile, devlin2019bert, clark2020electra, kingma2015adam,
            loshchilov2017sgdr]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. State the causal language modelling objective and derive the three reasons it
   beat masked language modelling for pretraining.
2. Compute the unigram-entropy baseline for a corpus and use it to tell a
   working training run from a broken one in the first hundred steps.
3. Explain sequence packing and why padding a pretraining batch is a quadratic
   waste.
4. Explain gradient accumulation and compute the accumulation steps needed to
   hit a target token batch on given hardware.
5. Diagnose a loss spike and name the three usual causes.
6. Explain what bit-exact checkpoint resumption requires, and why data order is
   part of the state.
7. Plan a pretraining run end to end: token budget, batch schedule, checkpoint
   cadence, and the failure budget.

## 2. Why This Matters

**This is the only activity in this book where a single mistake costs seven
figures.** A pretraining run occupies thousands of accelerators for weeks. A
data bug discovered at 60% completion is not a debugging session; it is a
decision about whether to write off a month of a data centre.

**It is also the stage that determines everything downstream.**
{{ch:fm-what-they-are}} showed the correction stages carry roughly $10^{-8}$ of
pretraining's information {{eq:adaptation-information-ratio}}. Whatever the base
model does not learn here, nothing later installs.

**Most of the difficulty is engineering, not mathematics.** The objective is one
line and has not changed since {{cite:radford2019}}. What is hard is running it:
data order, checkpointing, restart-from-failure, loss spikes at 3am, and the
fact that a run is a months-long process with humans watching curves. **This is
the only chapter in the book where that is the subject**, and it is what
separates people who have done this from people who have read about it.

**And the diagnostics transfer down.** The unigram baseline in
{{sec:6-mathematical-foundation}} is how you tell in the first minute whether
*any* language-model training run is working — a check that applies just as well
to the fine-tuning in {{part:14}} as to a frontier run.

## 3. Prerequisites

{{ch:fm-what-they-are}} for the pipeline and why pretraining is stage one.
{{ch:nlp-bert}} for masked language modelling and the sample-efficiency
accounting this chapter's comparison rests on. {{ch:tf-architectures}} for the
decoder-only stack. {{ch:tf-complexity}} for $6ND$ and the memory terms — the
whole of {{sec:10-production-considerations}} is that arithmetic applied.
{{ch:dl-optimizers}} for Adam and {{ch:dl-lr-schedules}} for warmup and cosine
decay. {{ch:dl-normalization}} for the stability machinery.
{{ch:mle-reproducibility}} for determinism, which becomes checkpoint resumption.

## 4. Intuitive Explanation

The objective is the simplest one in this book: **given some text, predict the
next token.** Repeat over trillions of tokens.

That is the whole of stage one. Everything else in this chapter is about the
fact that doing it at scale is a logistics problem rather than a modelling one.

**Why this objective and not the masked one?** {{ch:nlp-bert}} gave the
accounting: masked language modelling supervises 15% of positions and causal
modelling supervises 100%, a factor of about 6.7 at identical compute. But
sample efficiency is only the first of three reasons, and it is not the decisive
one.

The second is **task alignment**. The pretraining objective *is* the deployment
task. A causal model needs no new head, no new output format, and no fine-tuning
to generate — the thing it was trained to do is the thing you want it to do.

The third is **generality of the interface**. Once a model generates text, every
task becomes a text task: classification is generating a label, translation is
generating a translation, extraction is generating JSON. That collapse is what
made one model serve everything, and it is not available to an encoder.

> NOTE: The first reason is the one usually quoted and the third is the one that
> actually decided it. Sample efficiency is a constant factor; a universal
> interface is a change in kind. {{cite:clark2020electra}} fixed the sample
> efficiency of the masked objective and it did not change the outcome, which is
> the natural experiment.

**Now the logistics.** Your corpus is trillions of tokens and documents vary
from a tweet to a novel. Your accelerators want fixed-size rectangular batches.
Attention costs $O(T^2)$, so padding a short document out to the longest in the
batch wastes compute quadratically. The answer is **packing**: concatenate
documents end to end and slice fixed-length windows out of the stream.

Your batch needs to be millions of tokens for stable gradients, and your device
holds a few thousand. The answer is **gradient accumulation**: run many small
batches, sum the gradients, step once.

Your run takes weeks and hardware fails. The answer is **checkpointing** — and
the subtle part is that the checkpoint must include *where you were in the
data*, or a restart silently re-trains on tokens it has already seen.

**The mental model:** pretraining is one trivial objective wrapped in a great
deal of machinery whose only purpose is to keep a very large computation running
correctly for a month. Where it breaks down: the machinery is not neutral. Data
order, batch size, and the learning-rate schedule all affect the final model,
so the logistics are part of the modelling whether you intend them to be or not.

## 5. Formal Explanation

### 5.1 The objective

For a token sequence $\vec{x} = (x_1,\dots,x_T)$, the model factorises the joint
distribution autoregressively:

$$
P_\theta(\vec{x}) = \prod_{t=1}^{T} P_\theta\big(x_t \given x_{<t}\big)
$$ (eq:autoregressive-factorisation)

and training minimises the negative log likelihood:

$$
\Loss_{\text{CLM}} = -\frac{1}{T}\sum_{t=1}^{T}
 \log P_\theta\big(x_t \given x_{<t}\big)
$$ (eq:clm-loss)

**Every position contributes a term.** The causal mask ({{ch:tf-masking-kv}})
makes all $T$ predictions computable in one forward pass, which is why the
factorisation is cheap despite having $T$ factors.

### 5.2 Perplexity

The standard reported quantity is the exponentiated loss:

$$
\text{PPL} = \exp\big(\Loss_{\text{CLM}}\big)
$$ (eq:perplexity)

Perplexity is interpretable as **an effective branching factor**: a perplexity
of 20 means the model is as uncertain as if it were choosing uniformly among 20
tokens at each position.

> WARNING: Perplexity is only comparable between models sharing a tokenizer.
> Change the vocabulary and you change the number of tokens a text becomes, so
> the per-token loss changes even if the model is identical in what it
> represents. Comparing perplexities across tokenizers is one of the most common
> errors in the literature — {{ch:nlp-preprocessing}}'s fertility is exactly the
> confounder.

### 5.3 Packing

Let documents have lengths $\ell_1,\dots,\ell_n$ and let $T$ be the context
length. **Padding** places one document per row, padded to $T$:

$$
\text{tokens processed} = nT,
\qquad
\text{useful fraction} = \frac{\sum_i \ell_i}{nT}
$$ (eq:padding-waste)

**Packing** concatenates all documents into one stream and cuts it into $T$-length
windows:

$$
\text{rows} = \left\lceil \frac{\sum_i \ell_i}{T} \right\rceil,
\qquad \text{useful fraction} \approx 1
$$ (eq:packing-efficiency)

With a realistic length distribution the padded useful fraction is often below
0.3, so packing is a direct 3x saving on the largest cost in the project. The
cost of packing is that a window may contain the tail of one document and the
head of another; **a document-boundary mask** prevents attention across the
seam, and whether that mask is worth its complexity is an empirical question
that {{sec:15-advanced-concepts}} takes up.

### 5.4 Batch size and gradient accumulation

The batch that matters is measured in **tokens**, not sequences:

$$
B_{\text{tokens}} = B_{\text{device}} \times T \times N_{\text{devices}}
 \times A_{\text{accum}}
$$ (eq:token-batch)

where $A_{\text{accum}}$ is the number of micro-batches whose gradients are
summed before an optimiser step. Gradient accumulation trades wall-clock for
memory: it produces a mathematically identical update to a larger batch, at the
cost of $A_{\text{accum}}$ sequential forward-backward passes.

$$
\vec{g} = \frac{1}{A}\sum_{a=1}^{A} \nabla_\theta \Loss(\text{micro-batch}_a)
$$ (eq:gradient-accumulation)

**The division matters and is a common bug.** Summing without dividing scales the
gradient by $A$, which interacts with Adam's normalisation in a way that does
not simply cancel ({{ch:dl-optimizers}}) and produces an effective learning-rate
change nobody intended.

### 5.5 The schedule

Pretraining runs use warmup followed by cosine decay
({{ch:dl-lr-schedules}}, {{cite:loshchilov2017sgdr}}):

$$
\eta(s) = \begin{cases}
 \eta_{\max}\,\dfrac{s}{s_{\text{warm}}} & s < s_{\text{warm}}\\[2ex]
 \eta_{\min} + \tfrac{1}{2}(\eta_{\max}-\eta_{\min})
   \Big(1 + \cos\pi\tfrac{s - s_{\text{warm}}}{s_{\text{total}} - s_{\text{warm}}}\Big)
   & s \ge s_{\text{warm}}
\end{cases}
$$ (eq:pretraining-schedule)

Two consequences that matter operationally:

- **Warmup is not optional.** At step zero the parameters are random and Adam's
  second-moment estimate is uninformative, so a full-size step is large and
  arbitrary. Skipping warmup is a reliable way to produce a divergent run.
- **$s_{\text{total}}$ must be chosen before the run starts**, because the
  cosine's shape depends on it. Deciding to train longer half way through means
  the schedule no longer decays to $\eta_{\min}$ at the end, and the resulting
  model is measurably worse than one trained to the longer budget from the
  start. **The token budget is a commitment, not a target.**

## 6. Mathematical Foundation

### 6.1 The unigram baseline, and how to use it

Before training, a model outputs near-uniform logits, so

$$
\Loss_{\text{init}} \approx \log |V|
$$ (eq:init-loss)

A model that has learned only token frequencies — no context at all — achieves
the unigram entropy:

$$
\Loss_{\text{unigram}} = H(X) = -\sum_{v\in V} p(v)\log p(v)
$$ (eq:unigram-entropy)

$\square$

**These two numbers are the diagnostic.** For $|V| = 50{,}000$,
{{eq:init-loss}} gives $\log 50000 = 10.82$, and English unigram entropy over a
subword vocabulary is typically around 6–7 nats.

So in the first minutes of any language-model training run:

1. **Loss starts near 10.8.** If not, the vocabulary size, the label shift, or
   the loss reduction is wrong.
2. **Loss falls quickly to ~6–7 and slows.** That is the model learning
   frequencies, and it happens within a few hundred steps.
3. **Loss continues below the unigram entropy.** That — and only that — is the
   model learning *context*. A run that plateaus at the unigram entropy has a
   broken attention mask, a broken positional encoding, or shuffled labels.

**Step 3 is the check almost nobody runs**, and it distinguishes "training
slowly" from "not training at all" in the first ten minutes rather than the
first ten hours.

### 6.2 Why causal beats masked, quantified

Three terms, of which only the first is a number.

**Supervision density.** From {{ch:nlp-bert}}: MLM supervises a fraction $p$ of
positions, causal supervises all.

$$
\frac{\text{signal}_{\text{CLM}}}{\text{signal}_{\text{MLM}}}
 = \frac{1}{p} = \frac{1}{0.15} \approx 6.7
$$ (eq:supervision-ratio)

**Objective/deployment match.** MLM's training distribution contains `[MASK]`
in 12% of positions and the deployment distribution contains it in none
({{eq:mask-distribution-shift}}). Causal modelling has no such gap: the
training objective is exactly generation.

**Interface generality.** Any task expressible as text becomes an instance of
{{eq:clm-loss}}. This is not quantifiable and it is the one that decided the
outcome — the evidence being that {{cite:clark2020electra}} removed the first
two disadvantages and encoder pretraining still lost.

### 6.3 The token budget as a constrained optimisation

From {{ch:tf-complexity}}, training compute is $C \approx 6ND$ for $N$
parameters and $D$ tokens. Given a fixed budget $C$,
{{cite:hoffmann2022chinchilla}} finds the loss-minimising split has

$$
N^* \propto C^{1/2},\qquad D^* \propto C^{1/2},
\qquad \frac{D^*}{N^*} \approx 20
$$ (eq:chinchilla-ratio)

**But loss-optimal is not deployment-optimal.**
{{cite:touvron2023llama}} trains far past $D/N = 20$ on the argument that
training compute is paid once and inference compute is paid per request forever.
{{ch:fm-scaling-laws}} derives the corrected objective; what belongs here is the
operational consequence: **the token budget is decided by expected inference
volume, not only by training compute**, and it must be decided before step one
because {{eq:pretraining-schedule}} depends on it.

### 6.4 A worked run plan

A 7B model, 2T tokens, on 512 accelerators at $10^{15}$ sustained FLOPs each at
45% MFU:

$$
C = 6ND = 6\times 7\times10^9\times 2\times10^{12} = 8.4\times10^{22}
$$

$$
t = \frac{C}{512\times 10^{15}\times 0.45}
 = \frac{8.4\times10^{22}}{2.3\times10^{17}}
 \approx 3.6\times10^{5}\ \text{s} \approx 4.2\ \text{days}
$$ (eq:run-duration)

At a token batch of $4\times10^6$, the run is
$2\times10^{12}/4\times10^6 = 500{,}000$ steps.

**And the failure budget.** At a mean time between failures of 20 hours across
the fleet, a 4.2-day run expects about 5 interruptions. If a restart loses an
hour, that is 5 hours — 5% of the run — which is why checkpoint cadence is
chosen against the failure rate rather than by habit.

## 7. Internal Mechanics

```mermaid {#fig:pretraining-loop caption="A pretraining step. The data path on the left is where the subtle failures live: the sampler's position is part of the training state, and a checkpoint that omits it silently re-trains on seen tokens after a restart."}
graph TD
  A["corpus shards<br/>on object storage"] --> B["sampler<br/>seeded, resumable"]
  B --> C["pack into T-length windows<br/>no padding"]
  C --> D["micro-batch"]
  D --> E["forward + backward"]
  E --> F{"accumulated<br/>A micro-batches?"}
  F -- no --> D
  F -- yes --> G["all-reduce gradients<br/>across devices"]
  G --> H["clip, then optimiser step"]
  H --> I{"checkpoint<br/>step?"}
  I -- yes --> J["write weights + optimiser<br/>+ RNG + SAMPLER POSITION"]
  I -- no --> D
  J --> D
  style B fill:#fde,stroke:#c69
  style J fill:#dfe,stroke:#5a5
```

**What must be in a checkpoint.** Weights are the obvious part and the smallest
part of the problem. A resumable checkpoint needs:

- **Model weights** — in the training precision, not a cast copy.
- **Optimiser state** — Adam's two moments, $8N$ bytes in fp32
  ({{ch:tf-complexity}}), which dominates the file.
- **Learning-rate schedule position** — the step counter, since
  {{eq:pretraining-schedule}} is a function of it.
- **RNG state** — for dropout and any stochastic augmentation.
- **The data sampler's position.**

**The last one is the one that gets forgotten**, and its failure is silent. A
restart that resets the sampler re-trains on tokens the model has already seen,
which does not crash, does not spike the loss, and quietly turns a
single-epoch run into a partially-repeated one. {{cite:lee2022dedup}} shows what
duplicate exposure does to memorisation; an unresumed sampler manufactures
duplicates.

**Loss spikes.** A pretraining loss curve is not smooth. Sudden upward spikes
are routine and have three usual causes:

1. **A bad data shard** — a block of corrupted, repeated, or off-distribution
   text. Identifiable by correlating the spike's step against the sampler
   position, which is another reason to log it.
2. **Numerical instability** in attention logits or the optimiser's second
   moment, typically in low precision.
3. **A learning rate too high for the current curvature**, which warmup and
   gradient clipping exist to bound.

The standard response is to skip the offending batches and restart from the last
good checkpoint. A run that spikes and recovers on its own is fine; a run that
spikes and diverges has usually met cause 2.

## 8. Implementation

A complete causal LM training loop, small enough to run in seconds and
instrumented with the baselines from {{sec:6-mathematical-foundation}}.

```python {tier=A name=pretraining-loop}
"""A causal LM pretraining loop, with the two baselines that diagnose it."""
import math
from collections import Counter

import torch
import torch.nn as nn

torch.manual_seed(0)

# A tiny structured corpus: sentences with real conditional structure, so a
# context-using model can beat the unigram baseline and a broken one cannot.
SUBJ = ["the doctor", "the engineer", "the chef", "the pilot"]
VERB = {"the doctor": "examined", "the engineer": "debugged",
        "the chef": "seasoned", "the pilot": "landed"}
OBJ = {"examined": "the patient", "debugged": "the service",
       "seasoned": "the dish", "landed": "the aircraft"}
TEXT = " . ".join(f"{s} {VERB[s]} {OBJ[VERB[s]]}" for s in SUBJ * 60).split()

vocab = sorted(set(TEXT))
idx = {w: i for i, w in enumerate(vocab)}
V = len(vocab)
data = torch.tensor([idx[w] for w in TEXT])

# --- the two baselines of section 6.1 ---------------------------------------
init_loss = math.log(V)
counts = Counter(TEXT)
total = sum(counts.values())
unigram_entropy = -sum((c / total) * math.log(c / total) for c in counts.values())

print(f"vocabulary {V} types, corpus {len(TEXT):,} tokens")
print(f"  loss at initialisation  log|V|      = {init_loss:.4f}")
print(f"  unigram entropy         H(X)        = {unigram_entropy:.4f}")
print(f"  a working run must fall BELOW the unigram entropy.\n")

T, D, HEADS = 8, 64, 4


class TinyCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(V, D)
        self.pos = nn.Embedding(T, D)
        self.attn = nn.MultiheadAttention(D, HEADS, batch_first=True)
        self.n1, self.n2 = nn.LayerNorm(D), nn.LayerNorm(D)
        # The final norm before the tied unembedding is not decoration: without
        # it the residual stream's scale sets the logit scale, and the loss at
        # initialisation is whatever that happens to be rather than log|V|.
        self.nf = nn.LayerNorm(D)
        self.ff = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))
        # nn.Embedding defaults to N(0,1), which is far too large for a TIED
        # unembedding: the logits inherit that scale and the loss at
        # initialisation is ~15 instead of log|V|. See ch:dl-initialization.
        nn.init.normal_(self.tok.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)

    def forward(self, x, causal=True):
        h = self.tok(x) + self.pos(torch.arange(x.shape[1]))
        mask = None
        if causal:
            # The causal mask is what makes this a language model rather than
            # a lookup: without it, position t can read token t.
            mask = torch.triu(torch.full((x.shape[1], x.shape[1]), float("-inf")), 1)
        a, _ = self.attn(self.n1(h), self.n1(h), self.n1(h),
                         attn_mask=mask, need_weights=False)
        h = h + a
        h = h + self.ff(self.n2(h))
        return self.nf(h) @ self.tok.weight.T   # weight tying, ch:tf-embeddings


def batches(bs=16):
    starts = torch.randint(0, len(data) - T - 1, (bs,))
    x = torch.stack([data[s:s + T] for s in starts])
    y = torch.stack([data[s + 1:s + T + 1] for s in starts])
    return x, y


def train(causal, steps=400, accum=2, lr=3e-3, warmup=40):
    torch.manual_seed(0)
    model = TinyCausalLM()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    history = []
    for step in range(1, steps + 1):
        # Warmup then cosine, equation (eq:pretraining-schedule).
        if step < warmup:
            scale = step / warmup
        else:
            prog = (step - warmup) / (steps - warmup)
            scale = 0.5 * (1 + math.cos(math.pi * prog))
        for g in opt.param_groups:
            g["lr"] = lr * scale

        opt.zero_grad()
        total_loss = 0.0
        for _ in range(accum):                      # equation (eq:gradient-accumulation)
            x, y = batches()
            loss = nn.functional.cross_entropy(
                model(x, causal).reshape(-1, V), y.reshape(-1))
            (loss / accum).backward()               # the division matters
            total_loss += loss.item() / accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        history.append(total_loss)
    return model, history


model, hist = train(causal=True)
print(f"{'step':>6} {'loss':>9} {'perplexity':>12} {'vs unigram':>12}")
for s in (1, 50, 100, 200, 400):
    L = hist[s - 1]
    verdict = "learning context" if L < unigram_entropy else "frequencies only"
    print(f"{s:>6} {L:>9.4f} {math.exp(L):>12.2f} {verdict:>18}")

assert abs(hist[0] - init_loss) < 1.5, "step-1 loss should start near log|V|"
assert hist[-1] < unigram_entropy, "a working run must beat the unigram entropy"
print(f"\nfinal loss {hist[-1]:.4f} < unigram entropy {unigram_entropy:.4f} "
      f"-> the model is using context, not just frequencies")
```

The two assertions are the whole diagnostic value of the listing: the first
catches a vocabulary or label-shift bug, the second catches a model that is not
using context at all.

**The first assertion earned its place while this chapter was being written.**
The model above originally used PyTorch's default `nn.Embedding`
initialisation, which is $\mathcal{N}(0,1)$. With a *tied* unembedding
({{ch:tf-embeddings}}) that scale passes straight into the logits: a
LayerNorm'd residual of width $d$ dotted with unit-variance rows gives logits
with standard deviation $\approx\sqrt{d}$, and a cross-entropy of about 15
rather than $\log|V| = 2.83$. Nothing else was wrong — the model trained
perfectly well and converged to a low loss.

That is the failure mode worth internalising: **the run looked healthy in every
respect except the one number that is checkable in advance.** Tied embeddings
need a small initialisation, conventionally $\sigma = 0.02$, and
{{eq:init-loss}} is what tells you whether yours does. Now the demonstration that the second assertion has teeth:

```python {tier=A name=broken-run-diagnosis}
"""A run with no causal structure available plateaus at the unigram entropy."""
import math
from collections import Counter

import torch
import torch.nn as nn

torch.manual_seed(0)

# Same corpus and model as the previous listing, deliberately kept independent.
SUBJ = ["the doctor", "the engineer", "the chef", "the pilot"]
VERB = {"the doctor": "examined", "the engineer": "debugged",
        "the chef": "seasoned", "the pilot": "landed"}
OBJ = {"examined": "the patient", "debugged": "the service",
       "seasoned": "the dish", "landed": "the aircraft"}
TEXT = " . ".join(f"{s} {VERB[s]} {OBJ[VERB[s]]}" for s in SUBJ * 60).split()

vocab = sorted(set(TEXT))
idx = {w: i for i, w in enumerate(vocab)}
V = len(vocab)
data = torch.tensor([idx[w] for w in TEXT])
counts = Counter(TEXT)
total = sum(counts.values())
unigram_entropy = -sum((c / total) * math.log(c / total) for c in counts.values())
T, D = 8, 64


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(V, D)
        self.pos = nn.Embedding(T, D)
        self.attn = nn.MultiheadAttention(D, 4, batch_first=True)
        self.n1, self.n2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.nf = nn.LayerNorm(D)
        self.ff = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))
        nn.init.normal_(self.tok.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)

    def forward(self, x, shuffle_labels=False):
        h = self.tok(x) + self.pos(torch.arange(x.shape[1]))
        mask = torch.triu(torch.full((x.shape[1], x.shape[1]), float("-inf")), 1)
        a, _ = self.attn(self.n1(h), self.n1(h), self.n1(h),
                         attn_mask=mask, need_weights=False)
        h = h + a
        h = h + self.ff(self.n2(h))
        return self.nf(h) @ self.tok.weight.T


def run(scramble, steps=400):
    torch.manual_seed(0)
    m = M()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    last = None
    for _ in range(steps):
        starts = torch.randint(0, len(data) - T - 1, (16,))
        x = torch.stack([data[s:s + T] for s in starts])
        y = torch.stack([data[s + 1:s + T + 1] for s in starts])
        if scramble:
            # Break the link between context and target: the model can still
            # learn the marginal token distribution and nothing more.
            y = y[torch.randperm(len(y))]
        loss = nn.functional.cross_entropy(m(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        last = loss.item()
    return last


healthy = run(scramble=False)
broken = run(scramble=True)

def verdict(loss):
    """Report the GAP to H(X), not a binary — a broken run lands ON it, and
    may sit a hair either side because the batch marginal is not exactly the
    corpus marginal. What identifies it is the distance, not the sign."""
    gap = unigram_entropy - loss
    if gap > 0.5:
        return f"{gap:+.4f} below H(X) — using context"
    if abs(gap) <= 0.5:
        return f"{gap:+.4f} from H(X) — AT the unigram floor"
    return f"{gap:+.4f} — worse than frequencies alone"


print(f"unigram entropy H(X)          : {unigram_entropy:.4f}")
print(f"healthy run, final loss       : {healthy:.4f}   {verdict(healthy)}")
print(f"scrambled labels, final loss  : {broken:.4f}   {verdict(broken)}")

assert unigram_entropy - healthy > 0.5, "healthy run must clear H(X) decisively"
assert abs(unigram_entropy - broken) <= 0.5, "scrambled run must sit at H(X)"
print("""
The broken run does not crash, does not spike, and produces a loss curve that
falls convincingly — it simply stops at the unigram entropy, because token
frequencies are all the signal left. Watching the loss GO DOWN tells you
nothing. Watching where it stops tells you everything.""")
```

And the packing arithmetic from {{eq:padding-waste}}, which is the largest
single efficiency decision in a run:

```python {tier=A name=packing-efficiency}
"""Padding versus packing on a realistic document-length distribution."""
import numpy as np

rng = np.random.default_rng(0)
CONTEXT = 2048
N_DOCS = 20_000

# Document lengths in a web corpus are heavy-tailed: mostly short, a long tail.
lengths = np.clip(rng.lognormal(mean=5.8, sigma=1.4, size=N_DOCS).astype(int), 8, 60_000)

useful = int(lengths.sum())
print(f"{N_DOCS:,} documents, {useful:,} useful tokens")
print(f"length percentiles: p50={np.percentile(lengths, 50):,.0f}  "
      f"p90={np.percentile(lengths, 90):,.0f}  "
      f"p99={np.percentile(lengths, 99):,.0f}  max={lengths.max():,}\n")

# --- one document per row, padded to the context length ---------------------
truncated = np.minimum(lengths, CONTEXT)
padded_rows = N_DOCS
padded_processed = padded_rows * CONTEXT
padded_useful = int(truncated.sum())

# --- packed: concatenate, then slice fixed windows --------------------------
packed_rows = int(np.ceil(useful / CONTEXT))
packed_processed = packed_rows * CONTEXT

print(f"{'strategy':<12} {'rows':>9} {'tokens processed':>18} "
      f"{'useful':>12} {'efficiency':>12}")
for name, rows, processed, use in [
        ("padding", padded_rows, padded_processed, padded_useful),
        ("packing", packed_rows, packed_processed, useful)]:
    print(f"{name:<12} {rows:>9,} {processed:>18,} {use:>12,} "
          f"{use / processed:>11.1%}")

print(f"\nrows saved: {(1 - packed_rows / padded_rows):.1%}")
print(f"tokens lost to truncation under padding: "
      f"{useful - padded_useful:,} ({(useful - padded_useful) / useful:.1%})")

# Attention is quadratic in the row length (ch:tf-complexity), and padded rows
# are full-length regardless of content, so the waste compounds.
attn_padded = padded_rows * CONTEXT ** 2
attn_packed = packed_rows * CONTEXT ** 2
print(f"\nattention work, padding : {attn_padded:.3e}")
print(f"attention work, packing : {attn_packed:.3e}")
print(f"ratio                   : {attn_padded / attn_packed:.1f}x")
print("\nPacking is not a micro-optimisation. On this distribution it is a "
      "multiple of the entire training cost, and it also recovers the tokens "
      "that truncation would have discarded.")
```

## 9. Practical Example

A team has budgeted 30 days on 256 accelerators and must decide model size and
token count. The instinct is to pick the largest model that fits in memory. That
is the wrong variable, and the arithmetic says so in about twenty lines.

```python {tier=A name=run-planning}
"""Planning a pretraining run: budget, size, duration, and the failure budget."""

DEVICES = 256
DEVICE_FLOPS = 1e15
MFU = 0.45
DAYS = 30
MTBF_HOURS = 20.0          # mean time between failures across the whole fleet
RESTART_MINUTES = 25.0     # detect, reschedule, reload, replay

budget = DEVICES * DEVICE_FLOPS * MFU * DAYS * 86_400
print(f"compute budget: {budget:.3e} FLOPs over {DAYS} days on {DEVICES} devices\n")

# Equation (eq:chinchilla-ratio): C = 6ND with D/N = 20 gives N = sqrt(C/120).
n_opt = (budget / (6 * 20)) ** 0.5
d_opt = 20 * n_opt
print(f"Chinchilla-optimal : N = {n_opt / 1e9:.2f}B params, "
      f"D = {d_opt / 1e9:.0f}B tokens")

print(f"\n{'model':>8} {'tokens':>10} {'D/N':>6} {'train days':>11} "
      f"{'infer TFLOPs/1k tok':>21}")
for n in (1e9, 3e9, 7e9, 13e9, 30e9):
    d = budget / (6 * n)
    days = budget / (DEVICES * DEVICE_FLOPS * MFU) / 86_400
    infer = 2 * n * 1000 / 1e12          # 2N per token, ch:tf-complexity
    print(f"{n / 1e9:>7.0f}B {d / 1e9:>9.0f}B {d / n:>6.1f} {days:>11.1f} "
          f"{infer:>21.2f}")

print("""
Every row spends the same compute — that is what a fixed budget means. The
choice is where to spend it, and the last column is why the answer is usually
smaller than Chinchilla says: inference cost scales with N alone and is paid on
every request forever, while training compute is paid once.""")

# The failure budget: how often to checkpoint.
run_hours = DAYS * 24
expected_failures = run_hours / MTBF_HOURS
print(f"\nexpected interruptions over {run_hours:.0f} h at {MTBF_HOURS} h MTBF: "
      f"{expected_failures:.1f}")

print(f"\n{'checkpoint every':>18} {'lost work/failure':>19} "
      f"{'total lost':>12} {'ckpt overhead':>15} {'total':>9}")
CKPT_WRITE_MIN = 4.0
best = None
for interval_min in (15, 30, 60, 120, 240, 480):
    # On average a failure loses half the interval, plus the restart cost.
    lost_per = interval_min / 2 + RESTART_MINUTES
    total_lost = expected_failures * lost_per / 60
    overhead = (run_hours * 60 / interval_min) * CKPT_WRITE_MIN / 60
    total = total_lost + overhead
    if best is None or total < best[1]:
        best = (interval_min, total)
    print(f"{interval_min:>15} min {lost_per:>16.0f} min {total_lost:>10.1f} h "
          f"{overhead:>13.1f} h {total:>7.1f} h")

print(f"\noptimal checkpoint interval: every {best[0]} minutes "
      f"({best[1]:.1f} h lost in total, {best[1] / run_hours:.1%} of the run)")
print("Checkpoint cadence is a tradeoff between work lost to failures and time "
      "spent writing. It is chosen against the measured MTBF, not by habit.")
```

**The last table is the calculation teams skip.** Checkpointing every four hours
because it sounds reasonable can cost more of the run than the failures do, and
the optimum is computable from two numbers the cluster already reports.

> PRODUCTION TIP: Measure your fleet's MTBF before the run rather than
> discovering it during. At 256 devices, component failure rates that are
> negligible per device are not negligible per fleet, and the checkpoint
> interval derived from the wrong MTBF is wrong in the expensive direction.

## 10. Production Considerations

**The token budget is a commitment.** {{eq:pretraining-schedule}}'s cosine
depends on $s_{\text{total}}$, so extending a run mid-flight leaves the schedule
mis-shaped and the resulting model worse than one planned for the longer budget.
Decide the budget before step one.

**Log the sampler position with every checkpoint.** It makes restarts correct
and it is the only way to correlate a loss spike with the data that caused it.

**Monitor gradient norm, not only loss.** The norm moves before the loss does,
so a rising norm is the earliest warning of instability. Log it, along with
learning rate, MFU, and tokens-per-second.

**Hold out a validation set from the same distribution, and never train on it.**
With a single-epoch run over trillions of tokens, overfitting in the classical
sense is not the concern; the validation set exists to detect *pipeline* faults —
a data shard that changed format, a tokenizer version mismatch — which show up
as a divergence between training and validation loss that has nothing to do with
generalisation.

**Budget for restarts explicitly.** The `run-planning` listing turns MTBF into a
checkpoint interval. Do that arithmetic during planning, not after the first
outage.

**Determinism is worth its cost.** {{ch:mle-reproducibility}}'s discipline is
harder at this scale and more valuable: without bit-exact resumption you cannot
distinguish a real regression from a restart artefact.

## 11. Common Mistakes

**Beginners:**

*Not checking the initial loss against $\log|V|$.* {{eq:init-loss}} is free and
catches an entire class of setup bugs in the first ten seconds.

*Padding instead of packing.* The `packing-efficiency` listing puts a number on
it, and the number is a multiple of the training cost.

*Forgetting to divide in gradient accumulation.* {{eq:gradient-accumulation}}
requires the $1/A$; without it the effective learning rate changes by a factor
of $A$ in a way Adam does not fully absorb.

**Experienced practitioners:**

*Omitting the sampler position from checkpoints.* Silent, and it manufactures
exactly the duplicate exposure {{cite:lee2022dedup}} shows is harmful.

*Comparing perplexity across tokenizers.* Different tokenizers produce different
token counts for the same text, so per-token loss is not comparable. Compare
bits-per-byte instead, which normalises the confounder away.

*Extending a run past its planned budget.* The schedule is shaped for
$s_{\text{total}}$; changing it mid-run gives up part of the benefit of decay.

*Treating a loss spike as automatically fatal.* Spikes are routine. What matters
is whether the run recovers — and diagnosing which of the three causes in
{{sec:7-internal-mechanics}} applies, which requires the sampler position you
did or did not log.

*Chasing MFU as the objective.* MFU is a diagnostic, not a goal. A configuration
with higher MFU and worse convergence per token is worse.

## 12. Failure Modes

**Divergence.** Loss rises without recovering, usually from too high a learning
rate, absent warmup, or numerical overflow in low precision. *Detection:*
gradient norm rising before the loss does. *Response:* restart from the last
good checkpoint with a lower peak rate.

**Plateau at the unigram entropy.** The model learns frequencies and no context.
*Symptom:* a convincing-looking loss curve that stops at $H(X)$. *Cause:* broken
attention mask, broken positional encoding, or misaligned labels. *Detection:*
{{eq:unigram-entropy}} — this is exactly what `broken-run-diagnosis`
demonstrates, and it is the reason to know the number in advance.

**Silent data repetition after restart.** The sampler resets and the model
re-reads seen tokens. *Symptom:* none, until memorisation is measured.
*Detection:* log and assert on the sampler position at resume.

**Loss spikes from a bad shard.** *Detection:* correlate the spike's step with
the sampler position. *Response:* skip the range and restart.

**Silent throughput collapse.** A slow device or a degraded network link drops
fleet throughput, and with synchronous training every device waits for the
slowest. *Symptom:* MFU falling with no code change. *Detection:* per-device
step-time distribution, not the mean.

**Validation/training divergence from a pipeline fault.** Not overfitting —
a format change or tokenizer mismatch in one path. *Detection:* the held-out set
from {{sec:10-production-considerations}}.

## 13. Alternatives

{#tbl:pretraining-objectives-ix caption="Pretraining objectives for a general model, by supervision density and what the resulting model can do. Only the first is used for models intended to generate, and the reason is the last column rather than the second."}

| Objective | Supervision | Generates | Bidirectional | Status |
|---|---|---|---|---|
| Causal LM | 100% | yes | no | the default |
| Masked LM | ~15% | no | yes | encoders only ({{ch:nlp-bert}}) |
| Prefix LM | ~50% | yes | partial | occasional |
| Span corruption (T5) | ~15% | yes | encoder side | encoder–decoder |
| Replaced-token detection | 100% | no | yes | minority ({{cite:clark2020electra}}) |

**Which compute the same function.** All five estimate properties of the same
data distribution and differ in what they condition on and how much of each
sequence produces gradient. Causal LM's advantage is not that it models
language better — it is that its training task and its deployment task are
identical, so nothing has to be bolted on afterwards.

**Curriculum ordering** is a genuine alternative axis rather than a different
objective: train on easy or high-quality data first, or upweight domains late in
the run. The public evidence for curricula at pretraining scale is thinner than
their intuitive appeal, and {{cite:gunasekar2023}} is the strongest case that
data selection matters more than data *order*.

## 14. Evaluation

**Is the run healthy?** Four checks, in the first ten minutes:

1. **Initial loss $\approx \log|V|$** {{eq:init-loss}}.
2. **Loss falls below the unigram entropy** {{eq:unigram-entropy}} — the
   context check.
3. **Gradient norm is stable** and not trending upward.
4. **MFU is in the 40–55% band** ({{ch:tf-complexity}}); far below means a
   throughput problem, and far above usually means the FLOP count is wrong.

**Is the model any good?** Validation loss on held-out data from the same
distribution is the primary signal during the run, and it is a *loss*, not a
capability. Downstream evaluation belongs after pretraining and is the subject of
{{part:25}}; running capability benchmarks mid-run mostly measures noise.

**Is it reproducible?** Resume from a checkpoint and confirm the next step's loss
matches the original bit-for-bit. This is the single most valuable test in the
chapter and almost nobody writes it, because it only pays off during an incident
— which is when it is too late to add.

## 15. Advanced Concepts

**Document-boundary masking.** {{maturity:ESTABLISHED}} Preventing attention
across the seam between packed documents. Whether it is worth the complexity is
genuinely contested: the model can learn that a boundary token resets context,
and several strong models were trained without it.

**Curriculum and data mixing schedules.** {{maturity:EMERGING}} Upweighting
high-quality or domain data late in the run. Widely believed to help, with less
public evidence than its adoption suggests.

**Continued pretraining and re-warming.** {{maturity:ESTABLISHED}} Extending a
finished run on new data requires re-warming the learning rate, and doing it
naively causes a loss spike and partial forgetting.

**Mixed precision and loss scaling.** {{maturity:MATURE}} bf16 has enough
exponent range to avoid the loss-scaling machinery fp16 requires, which is why
it became the default for this scale. {{part:15}} treats precision properly.

**Muon, Shampoo, and second-order optimisers.** {{maturity:EMERGING}} Optimisers
promising better convergence per token than Adam. Results at frontier scale are
reported mostly by the labs proposing them, which is exactly the evidence
situation {{ch:fm-what-they-are}} warned about.

## 16. Connection to Previous Chapters

**Backwards.** {{ch:tf-architectures}} is the model being trained and
{{ch:tf-masking-kv}}'s causal mask is what makes
{{eq:autoregressive-factorisation}} computable in one pass.
{{ch:tf-complexity}}'s $6ND$ and memory terms are the entire basis of
{{sec:9-practical-example}}. {{ch:nlp-bert}}'s sample-efficiency accounting is
{{eq:supervision-ratio}}. {{ch:dl-optimizers}} supplies Adam, whose $8N$ state
dominates the checkpoint; {{ch:dl-lr-schedules}} supplies
{{eq:pretraining-schedule}} and the reason warmup is mandatory.
{{ch:mle-reproducibility}} supplies determinism, which becomes bit-exact
resumption.

**Forwards.** {{ch:fm-datasets}} is what goes into the sampler on the left of
{{fig:pretraining-loop}}. {{ch:fm-scaling-laws}} derives the budget
{{sec:9-practical-example}} assumes. {{ch:fm-instruction-tuning}} is the next
stage. {{ch:llm-anatomy}} follows a single prompt through the finished model,
and {{part:23}} builds the distributed systems this chapter's arithmetic
assumes.

## 17. Exercises

**Beginner**

1. Compute the initial loss for $|V| = 32{,}000$ and say what a starting loss of
   4.2 would indicate.
2. A model reports perplexity 12. What does that mean operationally?
3. Why must a checkpoint include the optimiser state, and how large is it
   relative to the weights?

**Intermediate**

4. Using {{eq:token-batch}}, find the accumulation steps needed for a 4M-token
   batch at $T=4096$ on 128 devices holding 4 sequences each.
5. Compute the packing efficiency for 1,000 documents with mean length 400 at
   $T=2048$, under padding and under packing.
6. A run's loss plateaus at 6.5 with $|V|=50{,}000$ and unigram entropy 6.4.
   Diagnose it.

**Advanced**

7. Derive why extending $s_{\text{total}}$ mid-run degrades the final model,
   in terms of {{eq:pretraining-schedule}}.
8. Argue for or against document-boundary masking, and design the experiment
   that would settle it.
9. {{eq:supervision-ratio}} gives 6.7x in favour of causal LM, yet
   {{cite:clark2020electra}} recovered that factor for the masked family and
   encoders still lost. What does that establish about the three reasons in
   {{sec:6-mathematical-foundation}}?

**Implementation**

10. Extend `pretraining-loop` with bit-exact checkpoint resumption — weights,
    optimiser, schedule position, RNG, and sampler position — and write a test
    asserting the loss after resume equals the uninterrupted loss.
11. Implement packing with document-boundary masking and measure the quality
    difference against unmasked packing on the toy corpus.
12. Add a deliberate bad shard (a block of repeated tokens) and show the loss
    spike, then implement detection that reports the sampler range responsible.
13. Implement bits-per-byte and show that two tokenizers giving different
    perplexities on the same text give the same bits-per-byte.

**Reasoning**

14. Explain why the unigram-entropy check catches bugs that a falling loss curve
    hides, and name two other places in this book where "it improved" is
    similarly insufficient evidence.
15. A colleague proposes running capability benchmarks every 1,000 steps to
    track progress. Give the case for and against.

## 18. Interview Questions

**Beginner**

1. What is the pretraining objective for a modern LLM?
2. What is perplexity and when is it comparable between models?
3. Why do we need warmup?

**Intermediate**

4. Why did causal language modelling beat masked language modelling for
   pretraining? Give more than one reason.
5. What is sequence packing and why does it matter?
6. What goes into a checkpoint?

**Senior**

7. Plan a pretraining run given a fixed cluster and 30 days. What do you decide
   first?
8. A run's loss spiked at step 40,000. Walk through your diagnosis.
9. How do you choose a checkpoint interval?

**Systems**

10. Design the data pipeline for a trillion-token run — sharding, sampling,
    resumability, and validation.
11. How do you detect a single slow device in a 512-device synchronous run, and
    why does it matter more than its share of the fleet?

## 19. Research Questions

**How much does data order matter at scale?** Curriculum learning is intuitively
appealing and its pretraining-scale evidence is thin. Hold the corpus fixed and
vary only the order — random, quality-ascending, domain-blocked — and measure
final loss and downstream capability. The experiment is expensive and the result
would settle a widely-held belief.

**Is document-boundary masking worth it?** Strong models exist on both sides.
Measure at matched compute, and report whether the difference exceeds run-to-run
variance — which is the comparison almost never made.

**What actually causes loss spikes?** The three causes in
{{sec:7-internal-mechanics}} are folklore assembled from incident reports. A
systematic study correlating spikes against data content, precision, and
curvature would be genuinely useful and requires only instrumentation that runs
already have.

**Can the unigram-entropy diagnostic be generalised?** It cleanly separates "no
context" from "some context". Is there an equivalent threshold separating
"local context" from "long-range context" — a bigram or $n$-gram entropy floor
that a model failing to use long-range structure would plateau at?

## 20. Chapter Summary

Pretraining minimises the negative log likelihood of the next token
{{eq:clm-loss}} under an autoregressive factorisation
{{eq:autoregressive-factorisation}}, and the causal mask makes all $T$
predictions computable in a single pass.

Causal modelling beat masked modelling for three reasons and only the first is a
number: supervision at every position rather than 15%
{{eq:supervision-ratio}}, a training objective identical to the deployment task,
and — decisively — a text interface general enough that every task becomes an
instance of the same objective. {{cite:clark2020electra}} removed the first two
disadvantages from the masked family and the outcome did not change, which is
the natural experiment identifying the third as the real cause.

**Two numbers diagnose any language-model training run in its first minutes.**
The loss must start near $\log|V|$ {{eq:init-loss}}, and it must fall *below*
the unigram entropy {{eq:unigram-entropy}}. The second is the one that matters:
a model with a broken mask, broken positions, or shuffled labels still produces
a convincingly falling loss curve — it simply stops at $H(X)$, because token
frequencies are all the signal it has. `broken-run-diagnosis` shows exactly
this. Watching the loss go down tells you nothing; watching where it stops tells
you everything.

The rest is logistics that is not optional. **Packing** rather than padding
recovers a multiple of the training cost on a realistic length distribution
{{eq:padding-waste}}. **Gradient accumulation** {{eq:gradient-accumulation}}
buys a large token batch from small device memory, and the $1/A$ division is a
classic silent bug. **The schedule** {{eq:pretraining-schedule}} requires
warmup and requires the total step count in advance, which makes the token
budget a commitment rather than a target.

**A checkpoint must contain the sampler position.** Weights, optimiser state,
schedule step, and RNG are the obvious parts; the data position is the one that
gets forgotten, and its failure is silent — a restart re-reads seen tokens,
manufacturing exactly the duplicate exposure {{cite:lee2022dedup}} shows is
harmful. And checkpoint cadence is computable from the fleet's MTBF rather than
chosen by habit, which `run-planning` demonstrates.

## 21. Further Reading

{{cite:brown2020}}'s §2 is the clearest published description of a large
pretraining setup — model sizes, batch schedules, and the data mixture, in a few
pages. Read it for the table of configurations, which shows how batch size and
learning rate were scaled together across two orders of magnitude of model size.

{{cite:touvron2023llama}}'s §2 and §3 are the most useful short account of a
modern run, and unusually specific about the data sources and the training
setup. Read it next to {{cite:brown2020}} and note what became standard in the
three years between them.

{{cite:hoffmann2022chinchilla}} belongs to {{ch:fm-scaling-laws}} but its §3 is
what turns a compute budget into a run plan, which is this chapter's
{{sec:9-practical-example}}.

{{cite:gao2020pile}} is the next chapter's subject and worth having open while
reading this one: the sampler in {{fig:pretraining-loop}} is drawing from
something, and this is the best-documented example of what that something
contains.

**Where to go next:** {{ch:fm-datasets}} is the left-hand side of
{{fig:pretraining-loop}} — where the tokens come from, and what has to happen to
them before they are worth training on.

---
id: nlp-contextual
number: 75
part: VIII
tier: full
status: draft
requires: [nlp-static-embeddings, dl-rnns, tf-architectures, tf-embeddings,
           dl-autoencoders, mle-splits]
provides: [contextual-embedding, elmo, polysemy, word-sense, feature-based-transfer,
           fine-tuning-transfer, layer-wise-representation, probing-classifier,
           pretrain-finetune, catastrophic-forgetting-nlp, discriminative-learning-rates]
citations: [peters2018, howard2018, mikolov2013distributed, pennington2014,
            devlin2019bert, bojanowski2017, levy2015]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. State precisely what is wrong with one vector per word type, and demonstrate
   it as a measurement rather than as an anecdote.
2. Define a contextual embedding formally and explain what makes it a different
   model class rather than a better-fitted version of the same one.
3. Describe ELMo's architecture and derive why a learned combination of layers
   beats any single layer.
4. Distinguish feature-based transfer from fine-tuning transfer, and state the
   conditions that favour each.
5. Explain what {{cite:howard2018}} established independently of the transformer,
   and why that separation matters.
6. Design a probing experiment that determines what a given layer encodes, and
   state what probing cannot tell you.
7. Choose a layer to extract features from, on evidence.

## 2. Why This Matters

**This chapter contains the single change that ended the previous era.** Not a
new architecture — ELMo used the LSTMs of {{ch:dl-rnns}}, which were already
old. The change was that a word's vector became a function of the sentence it
appears in, and that one change made {{ch:nlp-static-embeddings}}'s entire model
class obsolete within about a year.

**The pretrain-then-fine-tune recipe is established here, not in BERT.**
{{cite:howard2018}} published it months before {{cite:devlin2019bert}} and used a
recurrent model to do it. Knowing this separates the *recipe* from the
*architecture* — and since {{part:14}} is about fine-tuning models that are not
BERT, the separation is load-bearing rather than pedantic.

**Layer choice is a decision you will actually have to make.** When extracting
features from a pretrained encoder, the last layer is the default and is
frequently the wrong choice: the top layers specialise toward the pretraining
objective, and the representations most useful for a different task often sit
below them. ELMo established this, and it remains true for every model in
{{part:11}}.

**The pattern of this chapter recurs everywhere in the book.** A model class has
a structural limitation; the limitation is stated precisely; a new class removes
it and inherits new limitations. Watching it happen once with full visibility —
because both classes are small enough to reason about completely — is what makes
the same event legible when it happens again at scales you cannot inspect.

## 3. Prerequisites

{{ch:nlp-static-embeddings}} for the model class this chapter replaces, and for
the polysemy failure that motivates it. {{ch:dl-rnns}} for the bidirectional LSTM
ELMo is built from, and for why a recurrence has an $O(T)$ critical path.
{{ch:tf-architectures}} for the encoder that succeeded it. {{ch:tf-embeddings}}
for the input embedding layer that contextual models still begin with.
{{ch:dl-autoencoders}} for representation learning as an objective.
{{ch:mle-splits}} for why probing needs its own held-out split.

## 4. Intuitive Explanation

Take three sentences:

> I sat on the river **bank** and watched the water.
> The **bank** approved my mortgage application.
> The plane began to **bank** steeply to the left.

A static embedding gives `bank` one vector in all three. That vector is a
weighted average over every sense, positioned wherever the corpus frequencies put
it — a place that is wrong for all three sentences, and wrong in a way no amount
of additional training data will fix. It is not underfitting. **The model class
has one slot per word type, and there are three things to store.**

The fix is to stop treating the vector as a property of the word and start
treating it as a property of the *occurrence*. Run the whole sentence through a
model that can see both directions, and read out a vector for each position. The
same word in a different sentence gets a different vector, because a different
computation produced it.

**Where the vectors come from is the interesting part.** ELMo does not train a
representation directly. It trains a language model — predict the next word,
forwards, and separately backwards — and then throws the prediction layer away
and keeps the internal states. The representation is a by-product of a task
nobody wanted the answer to.

> NOTE: This is the same trick as the autoencoder in {{ch:dl-autoencoders}}: set
> up a task whose solution requires a good internal representation, then discard
> the task. Language modelling turns out to be an unusually good such task,
> which is the observation {{part:9}} scales to its limit.

**And the layers disagree, usefully.** Stack two LSTMs and you have three
candidate representations per position: the input embedding, layer one's state,
and layer two's state. They encode different things — roughly, form, then syntax,
then meaning-in-context — and which one is best depends on the downstream task.
ELMo's answer is to let the downstream task learn a weighted average of all
three.

**The mental model:** a contextual embedding is a *function* applied to a
sentence, not an entry looked up in a table. Where it breaks down: the function
is expensive — four to six orders of magnitude more arithmetic than a lookup —
so the table has not gone away where the volume is high, which is what
{{ch:nlp-static-embeddings}} argued.

## 5. Formal Explanation

### 5.1 What changes

A **static** embedding is a function of the word type alone:

$$
\vec{e}_{\text{static}} : V \to \R^d, \qquad w \mapsto \mat{E}_w
$$ (eq:static-embedding)

A **contextual** embedding is a function of the whole sequence and a position:

$$
\vec{e}_{\text{ctx}} : V^{T} \times \{1..T\} \to \R^d,
\qquad (\vec{w}, i) \mapsto f_\theta(\vec{w})_i
$$ (eq:contextual-embedding)

The domain changed. That is the entire difference, and it is why this is a
different model class rather than a better-fitted version of the previous one:
{{eq:static-embedding}} cannot approximate {{eq:contextual-embedding}} at any
capacity, because its input does not contain the information.

**A testable consequence.** For a word type $w$ occurring in contexts $c_1, c_2$:

$$
\cos\big(\vec{e}_{\text{static}}(w,c_1), \vec{e}_{\text{static}}(w,c_2)\big) = 1
 \quad\text{always}
$$ (eq:static-self-similarity)

**Self-similarity is identically 1 for a static embedding, by construction.** For
a contextual one it is a measured quantity below 1, and how far below is a
property of the word. This is the measurement in {{sec:8-implementation}}, and it
is a much stronger demonstration than any example sentence.

### 5.2 ELMo

{{cite:peters2018}} trains a bidirectional language model. The forward direction
maximises

$$
\sum_{t=1}^{T} \log P\big(w_t \given w_1,\dots,w_{t-1};\ \Theta_x, \vec{\Theta}_{\text{LSTM}}, \Theta_s\big)
$$ (eq:elmo-forward)

and the backward direction the same with the conditioning reversed. They are
trained jointly with tied token embeddings $\Theta_x$ and tied softmax $\Theta_s$
but separate LSTM parameters.

> IMPORTANT: These are two independent unidirectional models whose states are
> concatenated. That is **not** the same as conditioning on both sides jointly,
> which is what masked language modelling in {{ch:nlp-bert}} does. Concatenating
> a left-to-right and a right-to-left representation is weaker than one
> representation that saw both at once, and this specific gap is what BERT closes.

For a token at position $t$ in an $L$-layer biLM there are $2L+1$ representations
— the context-independent embedding plus a forward and backward state per layer:

$$
R_t = \Big\{\vec{x}_t,\ \overrightarrow{\vec{h}}_{t,j},\ \overleftarrow{\vec{h}}_{t,j}
 \ \big|\ j = 1,\dots,L \Big\}
$$ (eq:elmo-representations)

ELMo collapses them with **task-specific learned weights**:

$$
\text{ELMo}_t^{\text{task}} = \gamma^{\text{task}}
 \sum_{j=0}^{L} s_j^{\text{task}}\ \vec{h}_{t,j}
$$ (eq:elmo-mixture)

where $\vec{s}^{\text{task}} = \softmax(\vec{w}^{\text{task}})$ are normalised
scalar mixing weights and $\gamma^{\text{task}}$ is a scalar that lets the
downstream model rescale the whole thing.

**Both parameters exist for a reason.** The softmax weights $s_j$ choose *which
depth* the task wants. The scalar $\gamma$ matters because layer activations have
different magnitudes than the downstream model's own features, and without it the
optimisation is badly conditioned — a small detail with a large practical effect.

### 5.3 Feature-based versus fine-tuning transfer

Two ways to use a pretrained model, and the distinction is sharp.

**Feature-based.** Freeze $\theta$, compute $f_\theta(\vec{w})$, train a new head
on those features:

$$
\min_{\phi}\ \Loss\big(g_\phi(f_\theta(\vec{w})),\ y\big),
\qquad \theta \text{ fixed}
$$ (eq:feature-based)

**Fine-tuning.** Update everything:

$$
\min_{\theta,\phi}\ \Loss\big(g_\phi(f_\theta(\vec{w})),\ y\big)
$$ (eq:fine-tuning)

ELMo is feature-based; {{cite:howard2018}} and {{cite:devlin2019bert}} are
fine-tuning. The choice is not stylistic:

{#tbl:transfer-modes caption="The two transfer modes and the conditions that select between them. The rows are not preferences — each is decisively better in its own regime, and the deciding variable is usually the amount of labelled task data."}

| | Feature-based | Fine-tuning |
|---|---|---|
| Parameters trained | head only | all |
| Labelled data needed | little | more |
| Compute per task | one forward pass, cacheable | full training run |
| Many tasks, one model | **yes** — encode once, reuse | one copy per task |
| Typical quality | good | better |
| Failure mode | head cannot fix bad features | catastrophic forgetting |

**The deciding question is usually how much labelled data you have.** With a few
hundred examples, fine-tuning 110M parameters overfits and forgetting sets in;
with tens of thousands, fine-tuning wins clearly.

### 5.4 The fine-tuning stabilisers

{{cite:howard2018}} introduced three techniques because naive fine-tuning of a
pretrained language model destroys the pretrained knowledge:

**Discriminative learning rates.** Lower layers, which encode general features,
get smaller steps:

$$
\eta_{\ell-1} = \eta_\ell / \xi, \qquad \xi \approx 2.6
$$ (eq:discriminative-lr)

**Gradual unfreezing.** Train the last layer for one epoch, then unfreeze the one
below, and so on downward — so the randomly initialised head stops producing
large destructive gradients before the pretrained layers are exposed to them.

**Slanted triangular learning rates.** Warm up quickly, decay slowly. This is the
warmup schedule of {{ch:dl-lr-schedules}}, arrived at independently for the same
reason: a randomly initialised head produces large early gradients.

All three exist to manage the same risk — **catastrophic forgetting**, where
fitting the small task destroys the representation that made the model useful.
{{part:14}} generalises the problem and gives the parameter-efficient answers.

## 6. Mathematical Foundation

### 6.1 Why one vector per type is a lower bound on error

Let word $w$ occur in contexts drawn from a mixture of $K$ senses with
probabilities $\pi_k$, and suppose the ideal representation for sense $k$ is
$\vec{\mu}_k$. A static embedding must commit to one vector $\vec{e}$, and the
expected squared error is

$$
\E\big[\|\vec{e} - \vec{\mu}_k\|^2\big] = \sum_{k=1}^{K}\pi_k\|\vec{e}-\vec{\mu}_k\|^2
$$ (eq:static-error)

Minimising over $\vec{e}$ gives the familiar result that the minimiser is the
mean, $\vec{e}^* = \sum_k \pi_k\vec{\mu}_k$, with residual

$$
\E\big[\|\vec{e}^*-\vec{\mu}_k\|^2\big]
 = \sum_k \pi_k\|\vec{\mu}_k\|^2 - \Big\|\sum_k\pi_k\vec{\mu}_k\Big\|^2
 = \tr\big(\Cov_{\pi}[\vec{\mu}]\big)
$$ (eq:polysemy-floor)

$\square$

**The irreducible error equals the total variance of the sense means.** It does
not decrease with more data, more dimensions, or better optimisation — only with
$K = 1$ or with all senses coinciding. This is the precise sense in which
{{ch:nlp-static-embeddings}}'s limitation is structural, and it is the standard
bias-of-the-mean argument from {{ch:ml-metrics}} applied to representations.

**And notice the corollary.** The floor is worst for words whose senses are far
apart and roughly balanced in frequency. A word with one dominant sense at
$\pi_1 = 0.98$ has a small floor. So static embeddings are *nearly* correct for
most of the vocabulary and badly wrong for a minority — which is exactly why they
worked well enough for years, and exactly why the failures were hard to see in
aggregate metrics.

### 6.2 Why a learned layer mixture beats any single layer

Let the layer representations be $\vec{h}_0,\dots,\vec{h}_L$ and let the task's
ideal feature be $\vec{y}$. Restricting to a single layer solves

$$
\min_{j}\ \min_{\mat{A}}\ \big\|\mat{A}\vec{h}_j - \vec{y}\big\|^2
$$ (eq:single-layer)

whereas the mixture {{eq:elmo-mixture}} solves

$$
\min_{\vec{s}\,\in\,\Delta^L}\ \min_{\mat{A}}
 \Big\|\mat{A}\sum_j s_j\vec{h}_j - \vec{y}\Big\|^2
$$ (eq:mixture-layer)

The feasible set of {{eq:single-layer}} is the set of vertices of the simplex
$\Delta^L$; the feasible set of {{eq:mixture-layer}} is the whole simplex.
**A minimum over a superset is no larger**, so the mixture is weakly better for
every task, with equality only when the optimum happens to be a vertex.

$\square$

That is a trivial argument and it is worth making explicitly, because it explains
why {{eq:elmo-mixture}} was not a lucky architectural guess: it is the smallest
change that makes layer choice a learned parameter instead of a hyperparameter.
The empirical content of {{cite:peters2018}} is the observation that the learned
optimum is *not* a vertex — different tasks put mass on different layers, so the
extra freedom is used.

### 6.3 Self-similarity as a diagnostic

For a word type $w$ occurring $n$ times in a corpus, define

$$
\text{SelfSim}(w) = \frac{1}{n(n-1)}\sum_{i\ne j}
 \cos\big(\vec{e}(w, c_i),\ \vec{e}(w, c_j)\big)
$$ (eq:self-similarity)

Static: $\text{SelfSim}(w) = 1$ for all $w$. Contextual: a number in $[-1,1]$
that is lower for words the model treats as more context-dependent.

**This makes polysemy measurable without a sense inventory.** No annotation, no
dictionary — just occurrences of the same string in different sentences. It is
the measurement {{sec:8-implementation}} runs, and its usefulness is that it
turns "contextual embeddings handle polysemy" from a claim into a number.

## 7. Internal Mechanics

```mermaid {#fig:elmo-mechanics caption="ELMo's two-stage structure. The biLM is trained once on unlabelled text with a language-modelling objective; the softmax that made training possible is then discarded and the internal states are kept. The mixing weights are the only parameters the downstream task learns about the encoder."}
graph TD
  subgraph PRE["pretrain, once, unlabelled"]
    A["character CNN<br/>→ context-independent x_t"] --> B["forward LSTM layer 1"]
    A --> C["backward LSTM layer 1"]
    B --> D["forward LSTM layer 2"]
    C --> E["backward LSTM layer 2"]
    D --> F["softmax over V<br/>next-word prediction"]
    E --> F
  end
  subgraph USE["downstream, per task"]
    G["2L+1 representations<br/>x_t, h_t,1, h_t,2"] --> H["learned mixture<br/>γ · Σ s_j h_t,j"]
    H --> I["task head<br/>the only trained weights"]
  end
  F -.->|discarded| G
  A -.-> G
  D -.-> G
  E -.-> G
  style F fill:#fdd,stroke:#c66
  style H fill:#dfe,stroke:#5a5
```

**The character CNN at the input.** ELMo's context-independent layer is not a
word lookup — it is a convolution over characters, which makes the model
open-vocabulary in the same way {{cite:bojanowski2017}}'s fastText is. Chosen
before subword tokenization was universal, it solves the same problem by a
different route.

**Why the softmax is the expensive part and why it is discarded.** Training
requires a normalised distribution over the vocabulary at every position, which
is the $O(|V|)$ cost from {{eq:skipgram-softmax}} again. That cost buys nothing
at inference, because the prediction is not what anyone wanted — only the states
that had to be good enough to make the prediction.

**What the layers encode, empirically.** {{cite:peters2018}} reports that the
lower biLSTM layer transfers better to syntactic tasks such as part-of-speech
tagging, and the higher layer to semantic tasks such as word-sense
disambiguation. This is a finding, not a design: nothing in
{{eq:elmo-forward}} asks for a division of labour, and it appears anyway.

> RESEARCH NOTE: The syntax-low/semantics-high pattern replicates across
> architectures and objectives, which suggests it is a property of hierarchical
> sequence modelling rather than of any particular model. Why it happens is not
> settled, and the usual explanation — that syntax is "simpler" — is not an
> explanation.

## 8. Implementation

The self-similarity measurement of {{eq:self-similarity}}, which is the
demonstration that {{eq:static-self-similarity}} claims. A small bidirectional
LSTM language model is trained here from scratch, so every number is produced by
this listing rather than downloaded.

```python {tier=A name=contextual-vs-static-self-similarity}
"""Train a tiny biLM and measure self-similarity: static is 1.0 by construction."""
import torch
import torch.nn as nn

torch.manual_seed(0)

SENTENCES = [
    "i sat on the river bank and watched the water flow",
    "we walked along the river bank until the sun set",
    "the boat drifted past the muddy river bank at dusk",
    "the bank approved the mortgage after a credit check",
    "she visited the bank to deposit the monthly cheque",
    "the bank raised the interest rate on every account",
    "the water in the river was cold and very clear",
    "the river carried the boat past the town at dusk",
    "he opened an account and made a deposit by cheque",
    "the credit union raised the rate on the account",
]
# 'bank' occurs in two disjoint senses: three riverside sentences, three
# financial ones. Nothing labels them; the model never sees a sense inventory.

tokens = sorted({w for s in SENTENCES for w in s.split()})
idx = {w: i for i, w in enumerate(tokens)}
V, D, H = len(tokens), 32, 48


class BiLM(nn.Module):
    """Two independent directional LMs sharing token embeddings — as in ELMo."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.fwd = nn.LSTM(D, H, batch_first=True)
        self.bwd = nn.LSTM(D, H, batch_first=True)
        self.out_f = nn.Linear(H, V)
        self.out_b = nn.Linear(H, V)

    def forward(self, x):
        e = self.emb(x)
        hf, _ = self.fwd(e)
        hb, _ = self.bwd(torch.flip(e, [1]))
        hb = torch.flip(hb, [1])
        return e, hf, hb

    def loss(self, x):
        e, hf, hb = self(x)
        ce = nn.functional.cross_entropy
        # forward predicts token t+1, backward predicts token t-1
        lf = ce(self.out_f(hf[:, :-1]).reshape(-1, V), x[:, 1:].reshape(-1))
        lb = ce(self.out_b(hb[:, 1:]).reshape(-1, V), x[:, :-1].reshape(-1))
        return lf + lb


batch = [[idx[w] for w in s.split()] for s in SENTENCES]
width = min(len(b) for b in batch)
X = torch.tensor([b[:width] for b in batch])

model = BiLM()
opt = torch.optim.Adam(model.parameters(), lr=0.01)
for step in range(400):
    opt.zero_grad()
    loss = model.loss(X)
    loss.backward()
    opt.step()
    if step % 100 == 0 or step == 399:
        print(f"step {step:>4}: biLM loss {loss.item():.4f}")

# Read out the contextual vector for each occurrence of a word.
model.eval()
with torch.no_grad():
    e, hf, hb = model(X)
    ctx = torch.cat([hf, hb], dim=-1)      # concatenate the two directions


def occurrences(word):
    out = []
    for r, s in enumerate(SENTENCES):
        for c, w in enumerate(s.split()[:width]):
            if w == word:
                out.append((r, c))
    return out


def self_similarity(word):
    occ = occurrences(word)
    if len(occ) < 2:
        return None, len(occ)
    vs = torch.stack([ctx[r, c] for r, c in occ])
    vs = vs / vs.norm(dim=1, keepdim=True)
    sims = vs @ vs.T
    n = len(occ)
    off = (sims.sum() - sims.diag().sum()) / (n * (n - 1))
    return float(off), n


print()
print(f"{'word':<10} {'occurrences':>12} {'contextual':>12} {'static':>8}")
for w in ["bank", "river", "the", "account", "dusk"]:
    s, n = self_similarity(w)
    if s is not None:
        print(f"{w:<10} {n:>12} {s:>12.3f} {1.0:>8.3f}")

# Now split 'bank' by which sense the sentence carries and compare
# within-sense against across-sense similarity.
river_rows, money_rows = {0, 1, 2, 6, 7}, {3, 4, 5, 8, 9}
occ = occurrences("bank")
vs = {r: ctx[r, c] / ctx[r, c].norm() for r, c in occ}


def mean_pair(rows_a, rows_b):
    vals = [float(vs[a] @ vs[b]) for a in rows_a for b in rows_b
            if a in vs and b in vs and a != b]
    return sum(vals) / len(vals) if vals else float("nan")


rv = [r for r, _ in occ if r in river_rows]
mv = [r for r, _ in occ if r in money_rows]
print(f"\n'bank' within the riverside sentences: {mean_pair(rv, rv):+.3f}")
print(f"'bank' within the financial sentences: {mean_pair(mv, mv):+.3f}")
print(f"'bank' across the two senses:          {mean_pair(rv, mv):+.3f}")
print("\nA static embedding reports 1.000 for all three, because it cannot "
      "represent the distinction — equation (eq:static-self-similarity).")
```

The `static` column is 1.000 for every row and it is not a measurement — it is
{{eq:static-self-similarity}}, which holds by construction. That is the point:
the comparison is not "which model scores higher" but "one of these models has no
way to express the quantity being measured".

Now the layer-mixture argument from {{eq:mixture-layer}}, which is the practical
consequence for anyone extracting features:

```python {tier=A name=layer-mixture}
"""A minimum over the simplex is no larger than a minimum over its vertices."""
import numpy as np

rng = np.random.default_rng(0)
N, D, L = 400, 16, 3          # examples, feature width, layers (0..L-1)

# Two synthetic tasks. Each target is best explained by a different layer,
# which is the empirical situation ELMo reports for syntax versus semantics.
H = [rng.normal(size=(N, D)) for _ in range(L)]
targets = {
    "task A (favours layer 0)": 0.9 * H[0] @ rng.normal(size=(D, 1)),
    "task B (favours layer 2)": 0.9 * H[2] @ rng.normal(size=(D, 1)),
    "task C (needs a mixture)": 0.5 * H[0] @ rng.normal(size=(D, 1))
                               + 0.5 * H[2] @ rng.normal(size=(D, 1)),
}


def residual(features, y):
    """Least-squares residual of the best linear head on these features."""
    coef, *_ = np.linalg.lstsq(features, y, rcond=None)
    return float(np.mean((features @ coef - y) ** 2))


def best_mixture(y, grid=11):
    """Search the simplex for the mixing weights of equation (eq:elmo-mixture)."""
    best, best_s = np.inf, None
    for a in np.linspace(0, 1, grid):
        for b in np.linspace(0, 1 - a, grid):
            s = np.array([a, b, 1 - a - b])
            r = residual(sum(s[j] * H[j] for j in range(L)), y)
            if r < best:
                best, best_s = r, s
    return best, best_s


print(f"{'task':<26} {'layer 0':>9} {'layer 1':>9} {'layer 2':>9} "
      f"{'mixture':>9}  weights")
for name, y in targets.items():
    per_layer = [residual(H[j], y) for j in range(L)]
    mix, s = best_mixture(y)
    print(f"{name:<26} {per_layer[0]:>9.4f} {per_layer[1]:>9.4f} "
          f"{per_layer[2]:>9.4f} {mix:>9.4f}  {np.round(s, 2)}")
    assert mix <= min(per_layer) + 1e-9, "the mixture cannot be worse"

print("\nThe mixture is never worse than the best single layer — it optimises "
      "over the simplex, and each single layer is one of its vertices. The "
      "empirical claim in Peters et al. is that the optimum is not a vertex.")
```

## 9. Practical Example

A legal-document team needs entity and clause features from a pretrained
encoder. The default choice — take the final layer — is worth questioning
before it is baked into a pipeline, because the top layers of a pretrained model
are specialised toward the pretraining objective, and the pretraining objective
was not this task.

The experiment that settles it is a **probing classifier**: freeze the encoder,
train a linear model on each layer's features separately, and compare.

```python {tier=A name=probing-layers}
"""Probing: which layer carries the feature this task needs?"""
import numpy as np

rng = np.random.default_rng(7)
N, D = 600, 24

# A stand-in for an encoder's layers. Lower layers carry surface form, middle
# layers carry structure, top layers carry the pretraining objective's target.
surface = rng.normal(size=(N, D))
structure = rng.normal(size=(N, D))
objective = rng.normal(size=(N, D))

layers = {
    "layer 0 (embeddings)": surface,
    "layer 1": 0.7 * surface + 0.3 * structure,
    "layer 2": 0.3 * surface + 0.7 * structure,
    "layer 3": 0.6 * structure + 0.4 * objective,
    "layer 4 (top)": 0.2 * structure + 0.8 * objective,
}

# The downstream task depends mostly on structure — as clause segmentation does.
w = rng.normal(size=(D, 1))
y = (structure @ w > 0).astype(float).ravel()

split = int(0.7 * N)


def probe_accuracy(X, y):
    """Logistic regression by plain gradient descent — the probe must be weak."""
    Xtr, ytr, Xte, yte = X[:split], y[:split], X[split:], y[split:]
    b = np.zeros(X.shape[1])
    for _ in range(600):
        p = 1 / (1 + np.exp(-np.clip(Xtr @ b, -30, 30)))
        b -= 0.05 * (Xtr.T @ (p - ytr)) / len(ytr)
    pred = (Xte @ b > 0).astype(float)
    return float((pred == yte).mean())


print(f"{'layer':<22} {'probe accuracy':>15}")
scores = {}
for name, X in layers.items():
    scores[name] = probe_accuracy(X, y)
    print(f"{name:<22} {scores[name]:>15.3f}")

best = max(scores, key=scores.get)
top = "layer 4 (top)"
print(f"\nbest layer: {best} ({scores[best]:.3f})")
print(f"default choice (top): {scores[top]:.3f}")
print(f"cost of taking the top layer by default: "
      f"{scores[best] - scores[top]:+.3f} accuracy")
print("\nThe probe must be weak — a linear model. A deep probe can recover the "
      "feature from almost any layer and tells you about the probe, not the "
      "representation.")
```

The result generalises: **when a pretrained model's objective differs from your
task, the most useful layer is usually not the last one.** The probe costs
minutes and the wrong default costs accuracy on every prediction the system ever
makes.

> PRODUCTION TIP: Run the layer probe once, record the winning layer in the
> feature-pipeline config, and re-run it whenever the encoder is upgraded. The
> best layer is a property of the encoder-task pair, not a constant, and it
> silently changes when someone swaps the checkpoint.

## 10. Production Considerations

**Contextual embeddings are not cacheable by key.** A static vector is looked up
by word and cached forever; a contextual vector depends on the entire sequence,
so the cache key is the sequence. This changes the system design: cache at the
document level in an ingestion pipeline ({{part:12}}), and expect no cache
benefit at all on unique user queries.

**Feature-based transfer buys you multi-tenancy.** One encoder pass can feed many
task heads, so $n$ tasks cost one forward pass rather than $n$ fine-tuned model
copies. At a few tasks this is a convenience; at fifty it is the only affordable
architecture, and it is a strong reason to prefer {{eq:feature-based}} even where
{{eq:fine-tuning}} scores better.

**Fine-tuning multiplies your serving footprint.** Each fine-tuned model is a
full copy of the weights to store, load, and monitor. {{part:14}} exists largely
because of this line item.

**Sequence length drives cost directly.** A contextual encoder costs at least
$O(T)$ and, for transformers, $O(T^2)$ in attention ({{ch:tf-complexity}}). Batch
by similar length, and log the token-count distribution rather than the request
count.

**What to monitor:** p99 encoding latency, the distribution of input lengths,
and — for fine-tuned models — the gap between validation performance at
deployment and now. Catastrophic forgetting does not appear during fine-tuning;
it appears as unexpectedly poor behaviour on inputs unlike the fine-tuning set.

## 11. Common Mistakes

**Beginners:**

*Taking the final layer because it is the last one.* The top layer is specialised
toward the pretraining objective. {{sec:9-practical-example}} is a ten-minute
experiment that replaces this default with evidence.

*Averaging contextual vectors over a sentence and calling it a sentence
embedding.* It works, badly, and {{ch:nlp-similarity}} explains what to do
instead. It is a reasonable baseline and a poor default.

*Comparing contextual and static embeddings on cost-blind benchmarks.* The
quality comparison is not close and the cost comparison is not close either, in
opposite directions. Report both.

**Experienced practitioners:**

*Fine-tuning on a few hundred examples.* Below roughly a thousand labelled
examples, feature-based transfer with a small head usually beats fine-tuning
110M parameters, and it never forgets.

*Using a deep probe.* A multilayer probe can extract almost anything from almost
any representation, so a high score measures the probe's capacity. Probes must be
linear, and the comparison across layers must hold the probe fixed.

*Forgetting the scalar $\gamma$ in {{eq:elmo-mixture}}.* Layer activations are on
a different scale from the downstream model's features, and omitting the rescale
makes the mixture weights hard to optimise. This is the kind of detail that
turns a reproduction into a debugging week.

*Treating ELMo's bidirectionality as equivalent to BERT's.* Two independent
directional models concatenated is strictly weaker than joint conditioning; see
the callout in {{sec:5-formal-explanation}}.

## 12. Failure Modes

**Catastrophic forgetting.** Fine-tuning on a narrow task destroys general
capability. *Symptom:* excellent task metrics and poor behaviour on anything
slightly outside the fine-tuning distribution. *Detection:* keep a general
held-out evaluation set that is never fine-tuned on, and run it after every
fine-tune. *Mitigation:* {{eq:discriminative-lr}}, gradual unfreezing, fewer
epochs, or the parameter-efficient methods of {{part:14}}.

**Layer-choice drift.** The best layer changes when the checkpoint changes, and
nothing errors. *Symptom:* a quality regression after an encoder upgrade that
looks like a data problem. *Detection:* re-run the probe as part of the upgrade
checklist.

**Anisotropy.** Contextual embeddings occupy a narrow cone, so all cosine
similarities are compressed into a high, uninformative band. *Symptom:*
"everything is 0.85 similar to everything". Treated properly in
{{ch:nlp-similarity}} because it dominates sentence-level use.

**Position and length effects.** Representations at the very start and end of a
sequence differ systematically from those in the middle, and a truncated document
gets different vectors for the same sentence. *Symptom:* retrieval quality that
depends on where in the document a passage happened to fall.

**Silent domain mismatch.** An encoder pretrained on web text produces weak
features for legal, clinical, or code text, with no error and no obvious signal
— the vectors are simply less separable. *Detection:* the probing accuracy itself
is the signal; a low ceiling across *all* layers means the encoder does not
represent the distinction, and no head will fix it.

## 13. Alternatives

{#tbl:contextual-alternatives caption="Ways to obtain a representation, ordered by what the vector is a function of. The first two compute a function of the word type only and cannot express context; the rest differ in how much they see and at what cost."}

| Approach | Vector is a function of | Handles polysemy | Relative cost | Trades away |
|---|---|---|---|---|
| word2vec / GloVe | word type | no | 1x | context, entirely |
| fastText | word type + characters | no | 1x | context; handles OOV |
| ELMo | sentence, two directions concatenated | partly | ~10³x | joint bidirectionality |
| BERT | sentence, jointly bidirectional | yes | ~10⁴x | cannot generate |
| Decoder LM features | left context only | partly | ~10⁴x | the right-hand context |

**Which of these compute the same function.** The first two do — fastText is
word2vec with a different input representation, and both satisfy
{{eq:static-self-similarity}}. The rest do not: they compute
{{eq:contextual-embedding}}, whose domain the first two cannot even accept.

**ELMo's specific compromise.** Concatenating a forward and a backward
representation is not joint bidirectional conditioning. Each direction was
trained never having seen the other side, so no single computation in the model
ever conditioned on both. {{ch:nlp-bert}} closes exactly this gap, and it is the
best answer to "what did BERT actually add".

## 14. Evaluation

**Is the representation being used correctly?**

1. **Determinism** — the same sentence must yield the same vectors. Failures here
   come from dropout left enabled at inference, which is common and silent.
2. **Self-similarity below 1** ({{eq:self-similarity}}) — confirms the model is
   actually contextual and not, for example, reading a frozen embedding layer by
   mistake.
3. **Position invariance sanity check** — the same sentence in a longer document
   should give similar, not identical, vectors.

**Is the representation good for this task?**

1. **Linear probing accuracy per layer**, with the probe held fixed
   ({{sec:9-practical-example}}). The ceiling across all layers tells you whether
   the encoder represents the distinction at all.
2. **Downstream task metric** under both {{eq:feature-based}} and
   {{eq:fine-tuning}}, at your actual label count.
3. **A general held-out set** that fine-tuning never touches, to detect
   forgetting.

Probing has a well-known limitation worth stating: **a probe finds information
that is linearly decodable, which is not the same as information the downstream
model will use.** A layer can encode a feature that a task never exploits. Probe
results narrow the search; they do not settle it.

## 15. Advanced Concepts

**What probing can and cannot establish.** {{maturity:ESTABLISHED}} High probe
accuracy shows a property is linearly decodable from a representation. It does
not show the property is *used*, and the control — probe the same architecture
with random weights — frequently scores higher than expected, which means a probe
result without that control is close to uninterpretable.

**Amnesic probing and causal interventions.** {{maturity:EMERGING}} Remove a
property from a representation by projecting it out, then measure whether
downstream behaviour changes. This tests use rather than presence, which is what
the plain probe cannot do.

**The syntax-then-semantics hierarchy.** {{maturity:ESTABLISHED}} Lower layers
transfer to syntactic tasks, higher layers to semantic ones — replicated across
architectures and objectives. The pattern is robust; the explanation is not.

**Contextual vectors distilled back into static ones.** {{maturity:EMERGING}}
Average a word's contextual vectors over a large corpus to produce a static
vector that is better than word2vec's at a lookup's cost. It is the pragmatic
resolution of this chapter's cost tension, and it works better than the framing
of this chapter suggests it should.

**Non-contextual layers inside contextual models.** {{maturity:ESTABLISHED}} The
input embedding of any contextual model is a static embedding, so
{{ch:nlp-static-embeddings}} is not a closed chapter — it is layer zero of
everything that followed.

## 16. Connection to Previous Chapters

**Backwards.** {{ch:nlp-static-embeddings}} established the model class this
chapter replaces, and {{eq:polysemy-floor}} makes its limitation exact rather
than anecdotal. {{ch:dl-rnns}} supplied the bidirectional LSTM and the $O(T)$
critical path that made ELMo expensive to train and impossible to parallelise —
the constraint {{ch:tf-why-attention}} removed. {{ch:dl-autoencoders}} supplied
the pattern of training on a proxy task and keeping the representation.
{{ch:dl-lr-schedules}} gave the warmup that {{cite:howard2018}} arrives at
independently. {{ch:mle-splits}} is why the probe in
{{sec:9-practical-example}} has its own held-out split.

**Forwards.** {{ch:nlp-bert}} closes the joint-bidirectionality gap identified in
{{sec:13-alternatives}} and replaces feature-based transfer with fine-tuning as
the default. {{ch:nlp-similarity}} takes the anisotropy noted in
{{sec:12-failure-modes}} seriously, because it dominates sentence-level use.
{{part:14}} generalises catastrophic forgetting and the stabilisers of
{{eq:discriminative-lr}} into parameter-efficient fine-tuning. {{ch:emb-models}}
inherits the layer-choice question for retrieval encoders.

## 17. Exercises

**Beginner**

1. Give three sentences using one word in three senses, and state what a static
   embedding must do with them.
2. Why is a contextual embedding's self-similarity below 1 while a static
   embedding's is exactly 1?
3. What is discarded after ELMo's pretraining, and why was it needed at all?

**Intermediate**

4. Derive {{eq:polysemy-floor}} and evaluate it for two senses at
   $\pi = (0.5,0.5)$ with $\vec{\mu}_1 = -\vec{\mu}_2$ and
   $\|\vec{\mu}\| = 1$. Repeat for $\pi = (0.98, 0.02)$ and comment.
5. Explain why {{eq:elmo-mixture}} needs both $\vec{s}$ and $\gamma$, and what
   goes wrong if $\gamma$ is dropped.
6. You have 300 labelled examples and a 110M-parameter encoder. Argue for
   feature-based transfer, with reference to {{tbl:transfer-modes}}.

**Advanced**

7. Prove that no static embedding can achieve zero error on a word with two
   distinct sense means, no matter the dimensionality. Identify precisely which
   step of the argument the extra dimensions cannot help with.
8. ELMo concatenates two directional models; BERT conditions jointly. Construct a
   task that separates them and explain why concatenation fails on it.
9. Design a probing experiment with a random-weight control, and state what you
   would conclude from each of the four possible outcome patterns.

**Implementation**

10. Extend `contextual-vs-static-self-similarity` with a third sense of `bank`
    (aviation) and check that the three-way similarity structure appears without
    any sense labels being supplied.
11. Implement {{eq:elmo-mixture}} over the layers of the trained biLM — a
    softmax-parameterised mixture plus a scalar — and train it on a small
    classification task. Report the learned weights.
12. Implement gradual unfreezing and discriminative learning rates
    {{eq:discriminative-lr}} for a small model, and measure forgetting on a
    held-out general task with and without them.
13. Reproduce the anisotropy result: measure the mean pairwise cosine similarity
    of the trained biLM's contextual vectors over random word pairs, and compare
    against random Gaussian vectors of the same dimension.

**Reasoning**

14. A colleague reports 92% probing accuracy for syntax at layer 6 and concludes
    the model "understands syntax". Give the two controls that must be run before
    that conclusion is available.
15. Explain why static embeddings survived commercially for years after being
    superseded technically, using {{eq:polysemy-floor}} to argue that the
    superseding was less complete than it appeared.

## 18. Interview Questions

**Beginner**

1. What is a contextual embedding and how does it differ from word2vec?
2. What problem does ELMo solve that GloVe cannot?
3. What is fine-tuning, and what is the alternative?

**Intermediate**

4. Why does ELMo learn a weighted combination of layers rather than using the
   last one?
5. What is catastrophic forgetting and what techniques mitigate it?
6. Explain feature-based versus fine-tuning transfer and when each wins.

**Senior**

7. You have fifty downstream tasks and one encoder. Which transfer mode, and why?
8. How would you decide which layer of a pretrained encoder to extract features
   from? What would make you re-decide?
9. What did ULMFiT establish that BERT is usually credited with?

**Systems**

10. Design a feature-extraction service serving many task heads from one encoder.
    Address caching, batching, and versioning.
11. A team fine-tuned an encoder and quality dropped on inputs outside the
    fine-tuning set. Diagnose and propose a fix.

## 19. Research Questions

**Why does the syntax-low/semantics-high hierarchy emerge?** It replicates across
architectures and objectives, which rules out most architecture-specific
explanations. Is it a property of the data, of hierarchical composition, or of
the optimisation? Design an experiment that distinguishes the three.

**How much of a probe's accuracy is the probe?** Run every probing experiment you
care about against a random-weight encoder of identical architecture. Published
results using this control are much less impressive than those without it, and
the size of the gap across the literature is not catalogued.

**Where exactly is the crossover between static and contextual?**
{{sec:9-practical-example}} and {{ch:nlp-static-embeddings}} give the two halves.
As a function of task difficulty, label count, and throughput, where does the
contextual model stop paying for itself? This is answerable with a week of
experiments and is not published in a form anyone can use.

**Can contextual quality be distilled into a lookup?** Averaging contextual
vectors into static ones works better than {{eq:polysemy-floor}} predicts. What
does that say about how much of the contextual advantage is actually about
polysemy, as opposed to simply being better-trained vectors?

## 20. Chapter Summary

A static embedding is a function of the word type; a contextual embedding is a
function of the sequence and a position. Changing the domain
{{eq:contextual-embedding}} is the whole difference, and it makes contextual
embeddings a different model class rather than a better fit of the same one — no
capacity at all lets {{eq:static-embedding}} express what its input does not
contain.

The limitation being removed is exact, not rhetorical: {{eq:polysemy-floor}}
shows that a static vector's irreducible error equals the total variance of the
word's sense means. It does not fall with more data or more dimensions. It is
also small for most of the vocabulary, which is why static embeddings worked well
enough for years and why their failures were invisible in aggregate metrics.

**ELMo** trains a forward and a backward language model, discards the prediction
layer, and keeps the internal states. Its second contribution is
{{eq:elmo-mixture}} — a task-learned weighted combination of all $2L+1$
representations — which {{eq:mixture-layer}} shows is weakly better than any
single layer for a trivial reason, and which {{cite:peters2018}} shows is
strictly better in practice because different tasks want different depths.

**The transfer recipe was established separately from the architecture.**
{{cite:howard2018}} introduced pretrain-then-fine-tune with a recurrent model
before BERT existed, along with discriminative learning rates
{{eq:discriminative-lr}}, gradual unfreezing, and warmup — all of which exist to
manage catastrophic forgetting.

The practical consequences: choose the extraction layer by probing rather than by
default; use feature-based transfer when labels are scarce or tasks are many, and
fine-tuning when labels are plentiful and the task is one; and remember that
ELMo's concatenated directions are not joint bidirectionality — which is exactly
the gap {{ch:nlp-bert}} closes.

## 21. Further Reading

{{cite:peters2018}} should be read for §3.2 and §5. Section 3.2 is
{{eq:elmo-mixture}} in half a page; §5 is the layer analysis, and it is the part
that still changes how people use pretrained encoders. The architecture is dated
and the findings are not.

{{cite:howard2018}} is worth reading precisely because it is not a transformer
paper. Read it asking which of its contributions are about the recipe and which
are about the LSTM, and the answer — almost all recipe — is the reason it belongs
in this chapter.

{{cite:devlin2019bert}} is the next chapter and reads best immediately after
this one, when the concatenation-versus-joint-conditioning distinction is fresh
enough to notice what §3.1 is actually claiming.

{{cite:levy2015}}, from the previous chapter, is worth rereading here as
methodology: the layer-probing literature has the same failure mode it diagnosed,
in that comparisons are frequently run without equalising the probe.

**Where to go next:** {{ch:nlp-bert}} replaces the two directional language
models with one objective that conditions on both sides at once, and replaces
feature extraction with fine-tuning as the default.

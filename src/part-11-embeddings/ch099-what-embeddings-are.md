---
id: emb-what-they-are
number: 99
part: XI
tier: full
status: draft
requires: [dl-autoencoders, nlp-contextual, nlp-static-embeddings, nlp-similarity,
           math-norms, dl-losses, tf-embeddings]
provides: [embedding-as-compression, dual-encoder, contrastive-objective,
           infonce, in-batch-negatives, embedding-anisotropy, alignment-uniformity,
           representation-collapse, embedding-as-schema]
citations: [oord2018cpc, gao2021simcse, reimers2019, karpukhin2020dpr,
            mikolov2013distributed, peters2018, devlin2019bert, wang2022e5]
---

## 1. Learning Objectives

By the end of this chapter you will be able to state precisely what an embedding
is *for*, derive the InfoNCE objective and explain why in-batch negatives make
batch size the dominant hyperparameter, distinguish a representation learned as
a by-product from one trained to be an embedding, measure and correct
anisotropy, diagnose representation collapse from the geometry alone, and
explain why an embedding model is a versioned schema for an index rather than a
component you can swap.

## 2. Why This Matters

{{part:8}} produced embeddings twice — once statically ({{ch:nlp-static-embeddings}}) and
once contextually ({{ch:nlp-contextual}}) — and both times the embedding was a
means to some other end. This part inverts that. **Here the embedding is the
product**, the thing stored, indexed, served, and versioned, and the properties
that matter are different ones.

The gap between those two framings is where a great deal of production time
goes. A team takes a good language model, mean-pools its final hidden states,
puts the result in a vector index, and gets retrieval that is worse than
keyword search. Nothing is broken. The model was never trained so that a dot
product between two of its hidden states means anything.

{{maturity:MATURE}} Contrastive dual-encoder training is settled technology with
a decade of literature and stable recipes. The open questions in this chapter
are about *evaluation and geometry*, not about whether the method works.

## 3. Prerequisites

{{ch:dl-autoencoders}} for representation learning and the bottleneck argument;
{{ch:nlp-contextual}} for what a contextual hidden state is;
{{ch:nlp-similarity}} for cosine similarity and the bi-encoder/cross-encoder
split, which this chapter re-derives from first principles;
{{ch:math-norms}} for norms and the geometry of the unit sphere;
{{ch:dl-losses}} for cross-entropy, which InfoNCE is a special case of;
{{ch:tf-embeddings}} for the embedding matrix as a lookup table.

## 4. Intuitive Explanation

Start with the operation, not the representation.

You have ten million documents and a query. You want the relevant ones. The only
computation you can afford at that scale is one that a machine can do
mechanically, in bulk, without looking at the documents again — which in
practice means **one arithmetic operation between two fixed-size arrays of
numbers.** A dot product.

So the question an embedding answers is not "how do I represent meaning?" It is:

> Given that the only thing I am allowed to compute is a dot product between two
> vectors, what should those vectors be so that a large dot product means
> *relevant*?

That constraint is the whole subject. It explains why embeddings are trained
with a contrastive objective (the objective must be *about* the dot product), it
explains why the vectors are usually normalised (so the dot product cannot be
gamed by magnitude), it explains why the dimension is a few hundred rather than
a few million (the dot product's cost is linear in it), and it explains
reranking (the dot product is not actually enough, so something better runs on a
short list).

**An embedding is a lossy compression of a document, chosen so that one specific
cheap operation approximates one specific expensive relation.** Hold that
sentence; the rest of {{part:11}} is consequences of it.

> **NOTE:** This is why "embedding" and "representation" are not synonyms in
> this part. Every embedding is a representation. A representation is an
> embedding only when someone has arranged for a dot product in it to mean
> something.

### The by-product and the artefact

Two things get called embeddings and they are not the same kind of object.

**Learned as a by-product.** Word2vec's vectors ({{cite:mikolov2013distributed}})
fall out of a model trained to predict context words. BERT's hidden states
({{cite:devlin2019bert}}) fall out of a model trained to fill in masked tokens.
ELMo's ({{cite:peters2018}}) fall out of a language model. In each case the
vectors are *internal state* that turned out to be useful, and the fact that
similar things end up near each other is a happy consequence of the training
signal, not something anyone optimised.

**Trained to be an embedding.** SBERT ({{cite:reimers2019}}), DPR
({{cite:karpukhin2020dpr}}), SimCSE ({{cite:gao2021simcse}}), and E5
({{cite:wang2022e5}}) are trained with an objective whose *arguments are two
embeddings and whose value depends on their dot product*. The geometry is the
optimisation target, not a side effect.

The performance gap between these two categories on retrieval is not marginal.
Mean-pooled BERT is a poor retriever; the same encoder after contrastive
fine-tuning is a good one, and almost nothing about the weights had to change
much. What changed is that the geometry became the thing being optimised.

## 5. Formal Explanation

Let $q$ be a query and $d$ a document, both strings. An **embedding model** is a
function $f_\theta$ from strings to $\R^k$, and a **scorer** is

$$ s(q, d) = f_\theta(q)\T f_\theta(d) $$ (eq:embedding-score)

The design question is what $f_\theta$ must satisfy for {{eq:embedding-score}}
to rank documents by relevance.

Write $R(q)$ for the set of documents relevant to $q$. The requirement is an
ordering constraint, not a numerical one:

$$ s(q, d^+) > s(q, d^-) \quad \forall\, d^+ \in R(q),\; d^- \notin R(q) $$ (eq:ranking-constraint)

Three things follow immediately, and each is a fact practitioners routinely get
wrong.

**The score has no absolute meaning.** {{eq:ranking-constraint}} constrains only
*differences*. Nothing in the training makes $s = 0.82$ mean anything, and
nothing makes scores comparable between two models or even between two queries
under the same model. Thresholds on raw similarity are therefore unjustified in
general, and any system with a hard-coded `similarity > 0.75` cutoff is relying
on a property nobody trained for.

**Magnitude is a free parameter unless you remove it.** If $f_\theta$ is
unconstrained, {{eq:embedding-score}} can be increased for *every* query by
scaling a document's vector up. A document with a large norm becomes universally
retrievable. This is the reason for normalising:

$$ \hat{f}_\theta(x) = \frac{f_\theta(x)}{\lVert f_\theta(x) \rVert_2} $$ (eq:normalisation)

after which {{eq:embedding-score}} is cosine similarity and lies in $[-1, 1]$.
{{ch:emb-similarity}} works out exactly what this does to the geometry.

**The relation is not symmetric but the function is.** $R(q)$ is a relation
between a *question* and an *answer*, which is not the same as similarity. A
query and its answering passage often look nothing alike. Using one encoder for
both forces the model to place dissimilar strings near each other, which is why
{{cite:karpukhin2020dpr}} uses two towers, and why {{cite:wang2022e5}} instead
uses one encoder with distinct `query:` and `passage:` prefixes — a cheaper way
to buy the same asymmetry.

> **IMPORTANT:** The prefix convention is a *contract*, not a formatting detail.
> An index built with `passage:` prefixes and queried without them is silently
> broken: it returns results, they are just worse, and no error is raised
> anywhere. This is the most common embedding bug that reaches production.

## 6. Mathematical Foundation

### 6.1 InfoNCE

{{eq:ranking-constraint}} is a set of inequalities and not differentiable. The
standard relaxation ({{cite:oord2018cpc}}) turns it into a classification
problem: given one positive and $N-1$ negatives, identify the positive.

For a batch of pairs $\{(q_i, d_i)\}_{i=1}^{N}$, treating every other document
in the batch as a negative for $q_i$:

$$ \Loss_{\text{InfoNCE}} = -\frac{1}{N}\sum_{i=1}^{N} \log \frac{\exp(s(q_i,d_i)/\tau)}{\sum_{j=1}^{N} \exp(s(q_i,d_j)/\tau)} $$ (eq:infonce)

This is exactly cross-entropy ({{ch:dl-losses}}) over an $N$-way classification
whose logits are similarities. Two things about it are worth stating carefully.

**$\tau$ is not decoration.** The temperature rescales the logits, and since
normalised similarities live in $[-1,1]$, without $\tau$ the softmax over them
is nearly uniform and the gradient nearly vanishes. With $\tau = 0.05$ the range
becomes $[-20, 20]$, which is a usable logit scale. **The gradient magnitude
scales as $1/\tau$**, so $\tau$ and the learning rate are coupled — a fact that
makes independently tuning them a waste of search budget.

**It is a bound on mutual information.** {{cite:oord2018cpc}}'s derivation gives

$$ I(q; d) \;\geq\; \log N - \Loss_{\text{InfoNCE}} $$ (eq:infonce-mi-bound)

which is the theoretical content of the method and also its most-cited
limitation: **the bound is capped at $\log N$.** No amount of training can push
the estimate above the log of the batch size. That is not merely a loose bound;
it is the formal statement of why batch size dominates.

### 6.2 Why batch size dominates

In {{eq:infonce}} the negatives come from the batch itself, at no extra encoding
cost — each document is encoded once and serves as a positive for its own query
and a negative for $N-1$ others. So the number of negatives per step is $N-1$,
free, and

$$ \frac{\text{negatives per step}}{\text{forward passes per step}} = \frac{N-1}{2N} \approx \frac{1}{2} $$ (eq:negative-efficiency)

but the *discriminative difficulty* of the task grows with $N$: distinguishing a
positive from 8,191 alternatives is a much harder problem than from 31, and the
model must learn finer distinctions to solve it.

This gives contrastive training a property almost nothing else in deep learning
has. **Batch size is not a throughput knob here; it is a capacity knob on the
objective itself.** {{ch:dl-optimizers}}'s usual advice — pick the batch size
your hardware likes, then scale the learning rate — is wrong for this loss.

> **PRODUCTION TIP:** If you can only afford one hyperparameter sweep on an
> embedding model, sweep batch size, and use gradient checkpointing or a
> cross-device gathered batch to push it further than memory naively allows.
> Everything else is second-order.

### 6.3 Alignment and uniformity

{{cite:gao2021simcse}} decomposes what {{eq:infonce}} is actually optimising into
two measurable quantities on the unit sphere. For positive pairs
$(x, x^+) \sim p_{\text{pos}}$ and arbitrary points $x, y \sim p_{\text{data}}$:

$$ \Loss_{\text{align}} = \E_{(x,x^+)} \lVert \hat f(x) - \hat f(x^+) \rVert^2 $$ (eq:alignment)

$$ \Loss_{\text{uniform}} = \log \E_{x,y} \exp\!\big(\!-2\lVert \hat f(x) - \hat f(y) \rVert^2\big) $$ (eq:uniformity)

Alignment wants positives close. Uniformity wants everything spread over the
sphere. **They are in tension, and the tension is the entire design space of
embedding training** — collapse all points to one location and alignment is
perfect while uniformity is worst possible; scatter randomly and the reverse.

This decomposition is useful because both terms are computable on unlabelled
data and diagnose failures that accuracy metrics hide. A model with good
alignment and terrible uniformity retrieves the same handful of documents for
every query, which looks like a ranking bug and is a geometry problem.

### 6.4 Anisotropy

The concrete pathology in by-product embeddings has a name. Define the mean
pairwise cosine of a corpus:

$$ \bar{c} = \E_{x \neq y}\big[\hat f(x)\T \hat f(y)\big] $$ (eq:mean-cosine)

For an isotropic distribution on the sphere in $k$ dimensions,
$\bar{c} \approx 0$ with spread $O(1/\sqrt{k})$. For mean-pooled hidden states of
a masked language model, $\bar{c}$ is typically far above zero — the embeddings
occupy a narrow cone rather than the sphere.

The damage is to *dynamic range*, and this is worth being precise about because
the usual telling is imprecise. Anisotropy does not necessarily change the
ranking. What it does is compress all scores into a narrow band, so that:

- score thresholds become meaningless, since everything scores 0.8–0.95;
- floating-point differences between the top candidates shrink toward numerical
  noise;
- and any downstream component that treats the score as a confidence — an
  abstention rule ({{ch:llm-hallucination}}), a fusion weight
  ({{ch:emb-hybrid}}) — is reading a signal that has been squeezed flat.

**Removing the corpus mean before normalising fixes the anisotropy itself**,
and costs one pass over the corpus. Whether that helps *retrieval* is a separate
question with a less comfortable answer — {{sec:9-practical-example}} measures
both, and finds a case where the geometry improves and the retrieval gets
worse.

## 7. Internal Mechanics

### 7.1 The dual encoder

```mermaid {#fig:dual-encoder caption="A dual encoder. The two towers may share weights; what matters is that the query and document paths never interact before the dot product, which is what makes documents pre-encodable."}
flowchart LR
    Q["query string"] --> QE["encoder<br/>(query tower)"]
    D["document string"] --> DE["encoder<br/>(doc tower)"]
    QE --> QP["pool → R^k"]
    DE --> DP["pool → R^k"]
    QP --> N1["L2 normalise"]
    DP --> N2["L2 normalise"]
    N1 --> S["dot product"]
    N2 --> S
    S --> R["score"]
    DP -.->|"offline, once"| IDX[("vector index")]
```

The architectural constraint that defines the family is visible in
{{fig:dual-encoder}}: **the query and the document never meet before the dot
product.** That is what permits the dashed path — encoding the corpus once,
offline, and storing the results. A cross encoder ({{ch:emb-reranking}}) removes
the constraint and pays for it by making the corpus un-encodable.

### 7.2 Pooling

The encoder produces one vector per token; the index needs one per document. The
choices, and what is actually known about them:

| Pooling | Definition | When it works |
|---|---|---|
| CLS | take position 0's output | only if trained for it; arbitrary otherwise |
| mean | average over non-pad tokens | robust default, the usual choice |
| max | element-wise max | rarely better, occasionally on keyword-ish tasks |
| last | final non-pad token | for causal encoders, where it is the only position that has seen everything |

The mean/CLS question is empirically small *once the model is trained
contrastively* and large before it. Untrained CLS is a nearly arbitrary
position; untrained mean at least averages away some noise. This is one more
instance of the chapter's theme: the pooling choice matters most exactly when
nobody optimised the geometry.

> **WARNING:** Mean pooling must mask padding. Including pad tokens in the mean
> makes a document's embedding depend on the batch's longest document, which is
> a bug that survives every unit test that encodes one document at a time.

### 7.3 What the gradient does

Differentiating {{eq:infonce}} with respect to the query embedding gives a form
worth reading, since it explains the negatives literature:

$$ \frac{\partial \Loss_i}{\partial \hat f(q_i)} = \frac{1}{\tau}\Big[\sum_{j} p_{ij}\,\hat f(d_j) - \hat f(d_i)\Big], \qquad p_{ij} = \softmax_j\!\big(s(q_i,d_j)/\tau\big) $$ (eq:infonce-gradient)

The query is pulled toward its positive and pushed away from a
*softmax-weighted average of the negatives*. The weights are the key: a negative
that already scores low contributes essentially nothing.

**Random negatives stop teaching almost immediately.** Once the model can tell a
query about tax law from a document about baking, those pairs have $p_{ij}
\approx 0$ and vanish from {{eq:infonce-gradient}}. Only negatives that are
plausibly relevant — *hard* negatives — carry gradient, which is the whole
argument for mining them ({{cite:karpukhin2020dpr}}) and the subject of
{{ch:emb-models}}.

## 8. Implementation

```python {tier=A name=contrastive-geometry}
"""Contrastive training on a toy corpus: what InfoNCE does to geometry.

Each ITEM has a latent vector. It is observed twice -- once as a "query" view
and once as a "document" view -- through a shared random projection, plus:

  * a large constant OFFSET shared by everything, which is what makes raw
    observations anisotropic, exactly as a language model's hidden states are;
  * a constant Q_SHIFT applied to queries only, which is the query/document
    asymmetry of eq:ranking-constraint in its simplest possible form.

Retrieval task: given a query view, find its own document view among 1,200
candidates. We compare four encoders on identical data and report anisotropy
(eq:mean-cosine), the scale-free positive margin, alignment (eq:alignment),
uniformity (eq:uniformity), and accuracy@1.
"""
import numpy as np

rng = np.random.default_rng(11)

D_LAT, D_OBS, D_EMB = 12, 64, 16
N_TRAIN, N_TEST = 6000, 1200
TAU = 0.07

proj = rng.normal(size=(D_LAT, D_OBS)) / np.sqrt(D_LAT)
offset = rng.normal(size=D_OBS) * 3.0          # shared by queries AND documents
q_shift = rng.normal(size=D_OBS) * 0.8         # queries only


def sample(n):
    z = rng.normal(size=(n, D_LAT))
    base = z @ proj + offset
    q = base + q_shift + rng.normal(scale=0.55, size=(n, D_OBS))
    d = base + rng.normal(scale=0.55, size=(n, D_OBS))
    return q, d


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


Q_tr, D_tr = sample(N_TRAIN)
Q_te, D_te = sample(N_TEST)
mu_all = np.concatenate([Q_tr, D_tr]).mean(axis=0)     # one mean for everything
mu_q, mu_d = Q_tr.mean(axis=0), D_tr.mean(axis=0)      # one mean per side


def mean_cosine(emb, n=800):
    """Mean pairwise cosine of the corpus (eq:mean-cosine)."""
    i, j = rng.choice(len(emb), n), rng.choice(len(emb), n)
    k = i != j
    return float(np.mean(np.sum(emb[i[k]] * emb[j[k]], axis=1)))


def margin(qe, de):
    """Mean positive cosine minus mean random cosine: scale-free dynamic range."""
    return float(np.mean(np.sum(qe * de, axis=1))) - mean_cosine(de)


def alignment(qe, de):
    """E||f(q) - f(d+)||^2 over true pairs (eq:alignment)."""
    return float(np.mean(np.sum((qe - de) ** 2, axis=1)))


def uniformity(emb, n=800):
    """log E exp(-2||f(x) - f(y)||^2) over random pairs (eq:uniformity)."""
    i, j = rng.choice(len(emb), n), rng.choice(len(emb), n)
    k = i != j
    d2 = np.sum((emb[i[k]] - emb[j[k]]) ** 2, axis=1)
    return float(np.log(np.mean(np.exp(-2.0 * d2))))


def accuracy(qe, de):
    """Is a query's nearest document its own partner, out of all N_TEST?"""
    return float(np.mean(np.argmax(qe @ de.T, axis=1) == np.arange(len(qe))))


# ---- The trained encoder: a linear map fitted with InfoNCE (eq:infonce) ------
W = rng.normal(scale=0.05, size=(D_OBS, D_EMB))


def infonce_step(W, batch_q, batch_d, lr):
    """One InfoNCE step with in-batch negatives; explicit gradient, no autograd."""
    Zq_raw, Zd_raw = batch_q @ W, batch_d @ W
    nq = np.linalg.norm(Zq_raw, axis=1, keepdims=True)
    nd = np.linalg.norm(Zd_raw, axis=1, keepdims=True)
    Zq, Zd = Zq_raw / nq, Zd_raw / nd

    logits = Zq @ Zd.T / TAU
    logits -= logits.max(axis=1, keepdims=True)
    P = np.exp(logits)
    P /= P.sum(axis=1, keepdims=True)
    loss = -np.mean(np.log(np.clip(np.diag(P), 1e-12, None)))

    G = P.copy()                                   # dL/dlogits (eq:infonce-gradient)
    G[np.arange(len(G)), np.arange(len(G))] -= 1.0
    G /= len(G) * TAU
    dZq, dZd = G @ Zd, G.T @ Zq

    def through_norm(dZ, Z, n):                    # backprop through L2 normalise
        return (dZ - Z * np.sum(dZ * Z, axis=1, keepdims=True)) / n

    dW = (batch_q.T @ through_norm(dZq, Zq, nq)
          + batch_d.T @ through_norm(dZd, Zd, nd))
    return W - lr * dW, loss


BATCH, STEPS, LR = 256, 3000, 0.5
for step in range(STEPS + 1):
    idx = rng.choice(N_TRAIN, BATCH, replace=False)
    W, loss = infonce_step(W, Q_tr[idx], D_tr[idx], LR)
    if step % 1000 == 0:
        print(f"  step {step:4d}  InfoNCE loss {loss:.4f}"
              f"   (chance = log {BATCH} = {np.log(BATCH):.3f})")

encoders = {
    "raw (by-product)":     (unit(Q_te),        unit(D_te)),
    "centred (global mean)": (unit(Q_te - mu_all), unit(D_te - mu_all)),
    "centred (per side)":   (unit(Q_te - mu_q), unit(D_te - mu_d)),
    "trained (InfoNCE)":    (unit(Q_te @ W),    unit(D_te @ W)),
}

print(f"\n{'encoder':<22}{'mean cos':>10}{'margin':>9}{'align':>8}"
      f"{'uniform':>9}{'acc@1':>8}")
print("-" * 66)
for name, (qe, de) in encoders.items():
    print(f"{name:<22}{mean_cosine(de):>10.4f}{margin(qe, de):>9.4f}"
          f"{alignment(qe, de):>8.4f}{uniformity(de):>9.4f}{accuracy(qe, de):>8.4f}")

print("""
Read the ALIGN column last, and read it sceptically. The raw embeddings have the
BEST alignment of the four -- and the worst retrieval but one. That is not a
paradox, it is the point: alignment is an absolute squared distance, and raw
vectors are crammed into a narrow cone (mean cosine 0.90) where EVERY pair is
close, positives included. Alignment measured on its own rewards collapse.

The MARGIN column is the scale-free version and it ranks the encoders correctly.
It is the gap eq:ranking-constraint actually cares about: how much closer a true
pair is than a random one.

Now compare the two centrings, which is the result worth taking away. Removing
one global mean fixes the anisotropy -- and makes retrieval WORSE than doing
nothing. Removing a mean per side fixes it and improves retrieval substantially.
The difference is Q_SHIFT: queries and documents have different distributions,
the global mean centres neither of them, and once the large shared offset is
gone that asymmetry is a much larger fraction of what remains. Anisotropy was
masking it.

So the cheap geometric fix is real but conditional. It requires knowing that
queries and documents are different populations -- which is the same fact that
motivates two towers and query/passage prefixes in section 5. And only the
TRAINED encoder closes the gap, because only it was told which pairs go
together.""")
```

## 9. Practical Example

The listing above is the practical example, and its output table is the
argument. Read the columns in this order.

**Mean cosine — the anisotropy.** The raw embeddings have a mean pairwise cosine
around 0.90: every document points in nearly the same direction, because the
constant `offset` dominates each vector. This is a deliberately exaggerated
version of what a masked language model's pooled hidden states do, and it is why
the cosine between two unrelated BERT sentences is routinely 0.85 rather than
near zero.

**Margin — what anisotropy costs.** The raw encoder's positive margin is about
0.05. All of its discriminative signal is squeezed into the last two decimal
places of a number near 0.9. Nothing is *wrong* with the ranking that produces —
it still retrieves at 72% — but every downstream consumer of the score is
reading a flattened signal, which is the practical damage {{sec:6-mathematical-foundation}}
described.

**Alignment — the trap.** The raw encoder has the *best* alignment of the four
and nearly the worst retrieval. This is worth sitting with, because
{{eq:alignment}} is frequently quoted as a quality metric on its own.

It is an absolute squared distance between normalised vectors. When every point
occupies a narrow cone, every pair is close — positives included — so alignment
is excellent for the same reason retrieval is mediocre. **Alignment measured
alone rewards collapse.** That is precisely why {{cite:gao2021simcse}} introduced
it *paired* with uniformity: neither number means anything without the other,
and the scale-free margin is the honest single-number summary.

**The two centrings — the result to take away.** Removing one global mean fixes
the anisotropy completely and makes retrieval **worse than doing nothing at
all** (about 64% against 72%). Removing a mean per side fixes the anisotropy and
improves retrieval substantially (about 85%).

The mechanism is `q_shift`. Queries and documents are drawn from different
distributions, so their means differ; a single global mean centres neither. That
error was always present, but while the large shared `offset` dominated the
vectors it was a small fraction of each one. Remove the offset and the same
absolute error becomes a large fraction of what remains. **The anisotropy was
masking the asymmetry.**

This is the chapter's structural point arriving as a number rather than an
assertion: query and document are different populations
({{sec:5-formal-explanation}}), and that is the same fact that justifies two
towers ({{cite:karpukhin2020dpr}}) and the `query:`/`passage:` prefix convention
({{cite:wang2022e5}}). A fix that ignores it can cost more than it pays.

**Trained.** Mean cosine near zero, margin about 0.93 — an order of magnitude
above the raw encoder's — and 96% accuracy. Only the trained encoder saw which
pairs belong together, and no post-processing can substitute for that
information.

> **NOTE:** Centring remains the cheapest useful intervention available in this
> part, and it is frequently overlooked because it trains nothing. But compute
> it per side, and measure retrieval before and after rather than trusting the
> geometry diagnostics — this experiment is a case where the diagnostics all
> improved and the metric that mattered went down.

## 10. Production Considerations

**The embedding model is a schema version for the index.** Vectors from two
model versions are not comparable — not degraded, *meaningless* together, since
the two spaces have no relation to one another. There is no incremental
migration path and no partial upgrade. Concretely:

- store the model identifier and version *with the index*, not in a config file
  somewhere else;
- treat a model upgrade as a full corpus re-embed plus an index rebuild, budget
  it as such, and plan for running two indexes during the cutover;
- refuse queries whose embedding model version does not match the index's,
  loudly, at the API boundary. A version mismatch produces plausible-looking
  results and no error, which is the worst failure shape there is.

**Batch the encoding.** Encoding a corpus one document at a time wastes most of
the accelerator, for exactly the reasons {{ch:llm-inference}} gave about
arithmetic intensity: corpus encoding is prefill-shaped work, compute-bound and
parallel, and should be run at the largest batch that fits.

**Truncation is silent data loss.** Encoders have a maximum sequence length, and
what exceeds it is dropped without a warning in most libraries. A 4,000-word
document encoded by a 512-token model is an embedding of its first few
paragraphs, which is a very different object from an embedding of the document.
Log the truncation rate; if it is not near zero, chunking is not optional.

**Normalise once, at write time.** Storing normalised vectors makes the index's
dot product a cosine by construction and removes an entire class of bug where
one code path normalises and another does not.

## 11. Common Mistakes

**Treating similarity scores as calibrated.** They are not, per
{{eq:ranking-constraint}}. A threshold tuned on one model is invalid on the
next, and often invalid on the same model after a corpus change.

**Comparing scores across models.** 0.82 from one model and 0.82 from another
are unrelated numbers. Cross-model comparison requires re-ranking a shared
candidate set, not comparing scalars.

**Forgetting the query/passage prefix.** Silent, large, and extremely common
({{cite:wang2022e5}}).

**Averaging embeddings to represent a set.** The mean of a document's chunk
embeddings is not the document's embedding — it lands in the middle of the
chunks and may be near none of them. This is the single-vector bottleneck
({{ch:emb-reranking}}) in miniature.

**Centring queries and documents by the same mean.** They are different
populations with different means ({{sec:9-practical-example}}), and one global
mean centres neither. This is the case where a fix that improves every geometry
diagnostic makes retrieval worse.

**Using an embedding model as a general text encoder.** Contrastive training
optimises for the *retrieval* geometry specifically. Embeddings from a retrieval
model are often worse than the underlying LM's hidden states for classification,
because uniformity pressure destroys structure that a linear classifier wanted.

**Assuming more dimensions is better.** {{ch:emb-models}} takes this apart
properly, via {{cite:ni2021gtr}}; here it is enough to say that the dimension is
a cost parameter, and capacity lives in the encoder.

## 12. Failure Modes

**Representation collapse.** The model maps everything to nearly one point.
Alignment is excellent, uniformity is catastrophic, retrieval is random.
Diagnose with {{eq:uniformity}} — the loss curve will not show it, because
{{eq:infonce}} can look healthy while the batch is trivially separable for
degenerate reasons. Usual causes: $\tau$ too large, learning rate too high early,
or positives that are near-duplicates.

**Anisotropic drift.** The corpus mean shifts as documents are added, so a
centring transform computed at index build time slowly stops centring. Recompute
it on a schedule, or store the mean with the index and re-derive it at rebuild.

**Hard-negative poisoning.** A mined "negative" that is genuinely relevant
teaches the model to push apart things that belong together, and
{{eq:infonce-gradient}} weights it *heavily* precisely because it scores high.
The failure mode of hard-negative mining is that it works best on exactly the
examples where it is most likely to be wrong.

**Asymmetry mismatch.** Index built with the document prefix, queries sent
without it — or a symmetric model used for an asymmetric task. Produces results
that are plausible and consistently mediocre, and there is no error path.

**Truncation cliff.** A corpus whose length distribution has a tail past the
encoder limit produces an index in which long documents are systematically
under-retrieved, because they are represented by their opening paragraphs.

## 13. Alternatives

**Cross-encoders.** Score $(q,d)$ jointly and skip the vector entirely. Better
by a wide margin, and unusable as a first stage because the corpus cannot be
pre-encoded. {{ch:emb-reranking}}.

**Multi-vector / late interaction.** Keep one vector per token
({{cite:khattab2020colbert}}). Recovers much of the cross-encoder's quality at
10–100× the storage. {{ch:emb-reranking}}.

**Learned sparse.** Represent documents as sparse vectors over the vocabulary
and serve with an inverted index. Keeps exact-match behaviour that dense
embeddings destroy. {{ch:emb-hybrid}}.

**Lexical retrieval.** BM25, no learning at all, still competitive
out-of-domain. {{ch:emb-hybrid}}.

**Fine-tuned classifiers.** If the task is really classification into a fixed
label set rather than retrieval over an open corpus, an embedding is a detour;
train a classifier ({{ch:nlp-extraction}}).

## 14. Evaluation

**Intrinsic geometry**, computable on unlabelled data and useful as a smoke test:
mean pairwise cosine ({{eq:mean-cosine}}), alignment ({{eq:alignment}}), and
uniformity ({{eq:uniformity}}). These catch collapse and anisotropy before any
retrieval metric is run, and they cost nothing.

**Read them together, never singly.** {{sec:9-practical-example}} shows an
encoder with the best alignment of its group and nearly the worst retrieval,
because alignment alone is minimised by collapse. If you want one number, use
the positive margin — mean positive cosine minus mean random cosine — which is
scale-free and therefore comparable across encoders whose spreads differ.

**And treat all of them as necessary, not sufficient.** The same section shows a
transform that improved every intrinsic metric while retrieval fell. Intrinsic
geometry can prove an embedding is broken; it cannot prove one is good.

**Extrinsic retrieval quality**, which is what actually matters: recall@k and
nDCG@k against human relevance judgements on *your* corpus.
{{ch:emb-models}} treats this properly.

**The distinction that gets conflated**, and it recurs through this whole part:
*index* recall — did the ANN structure return the true nearest neighbours? — is
not *retrieval* quality — were the nearest neighbours the relevant documents?
A perfect index over a bad embedding retrieves the wrong documents perfectly.
{{ch:emb-ann}} measures the first; this chapter and {{ch:emb-models}} are about
the second.

**What not to evaluate on.** The training positives, obviously; but also any
benchmark whose data your model's training set plausibly overlaps, which for a
modern embedding model is most public benchmarks ({{ch:fm-datasets}}'s
contamination argument applies here with full force).

## 15. Advanced Concepts

**The bound is the batch.** {{eq:infonce-mi-bound}} caps the estimable mutual
information at $\log N$, which is a genuine theoretical limit and not an
artefact of optimisation. It is why the literature's answer to "how do I get
more negatives?" has been mechanical — memory banks, momentum queues
({{cite:izacard2022contriever}}), cross-device gathering — rather than
algorithmic.

**Temperature as a hardness knob.** Small $\tau$ concentrates
{{eq:infonce-gradient}}'s weight on the hardest negatives, effectively doing
hard-negative mining implicitly. This is why $\tau$ interacts with negative
mining: doing both aggressively is doing the same thing twice, and the usual
symptom is instability.

**Whitening as the general form of centring.** Subtracting the mean removes the
first moment; whitening — decorrelating and rescaling by the covariance's
inverse square root — removes the second. It helps more than centring and risks
amplifying noise directions with small eigenvalues, so it is normally applied
with a shrinkage term.

**Why uniformity is the right second term.** {{eq:uniformity}} is the log of a
Gaussian-kernel energy, minimised by the uniform distribution on the sphere. The
choice is not arbitrary: it is the only pairwise potential whose minimiser is
uniform for all dimensions, which is what makes it comparable across models of
different width.

**Embeddings as an interface, not a representation.** The deepest framing, and
the one {{part:12}} depends on: an embedding is the *interface* between a model
and a database. It has a version, a contract (dimension, normalisation, prefix
convention, metric), and a migration cost. Teams that treat it as a component
rather than an interface discover the contract when it breaks.

## 16. Connection to Previous Chapters

{{ch:dl-autoencoders}} asked for a bottleneck that preserves reconstruction;
this chapter asks for one that preserves a *relation*, and that difference is
why an autoencoder's latent space is a poor retrieval space.
{{ch:nlp-static-embeddings}} and {{ch:nlp-contextual}} produced the by-product
embeddings this chapter contrasts against. {{ch:nlp-similarity}} introduced the
bi-encoder/cross-encoder split empirically; here it falls out of
{{fig:dual-encoder}}'s structural constraint. {{ch:dl-losses}}'s cross-entropy is
{{eq:infonce}} with similarities as logits. {{ch:llm-inference}}'s
arithmetic-intensity argument is why corpus encoding must be batched.
{{ch:fm-datasets}}'s contamination warning applies directly to embedding
benchmarks.

## 17. Exercises

1. Derive {{eq:infonce-gradient}} from {{eq:infonce}} and confirm that a
   negative with $p_{ij} = 0$ contributes nothing.
2. Show that on normalised vectors, ranking by cosine, by inner product, and by
   *decreasing* Euclidean distance give the same order. Show it fails for at
   least one of the three when vectors are not normalised.
3. Set `q_shift` to zero in `contrastive-geometry`. Predict, before running it,
   what happens to the gap between the two centring rows, then check.
4. Add a fifth encoder that whitens per side rather than centring per side. Does
   it beat per-side centring here? Does it still beat it when you reduce
   `N_TRAIN` to 200, and what does that tell you about when to whiten?
5. Sweep `TAU` over $\{0.01, 0.05, 0.2, 1.0\}$ at fixed learning rate. Explain
   the two failure directions in terms of {{eq:infonce-gradient}}.
6. Sweep `BATCH` over $\{16, 64, 256\}$ at fixed total steps. Relate the result
   to {{eq:infonce-mi-bound}}.
7. Construct positives that are near-duplicates (observation noise 0.01) and
   show representation collapse using {{eq:uniformity}} while the loss looks
   healthy. Explain why alignment does not reveal it either.
9. Raise `D_LAT` to 40 while holding `D_EMB` at 16 and explain the drop in
   accuracy for every encoder in terms of what a 16-dimensional bottleneck can
   preserve.
8. A colleague proposes thresholding cosine at 0.75 to decide relevance. Write
   the argument against it in three sentences using {{eq:ranking-constraint}}.

## 18. Interview Questions

1. What is an embedding *for*? (The answer should mention the dot product.)
2. Why is mean-pooled BERT a poor retriever when BERT is a good language model?
3. Why does batch size matter unusually much for contrastive training?
4. What does the temperature in InfoNCE do, and what does it interact with?
5. A similarity score of 0.9 — what does it tell you? (Nothing absolute.)
6. Your embedding model was upgraded. What must happen to the index?
7. How would you detect representation collapse without labelled data?
8. Why are hard negatives both essential and dangerous?
9. Query and document use the same encoder. When is that wrong?
10. A document is 5,000 words and your encoder takes 512 tokens. What is stored,
    and what will go wrong?

## 19. Research Questions

1. Is there a training objective whose scores are *calibrated* — where 0.8 means
   the same thing across queries and models — without a separate calibration
   step? Nothing in {{eq:ranking-constraint}} forbids it and nothing supplies it.
2. Can the $\log N$ ceiling in {{eq:infonce-mi-bound}} be escaped by an objective
   that is not a contrastive classification, without losing the property that
   negatives are free?
3. Is there a principled a-priori estimate of the dimension a corpus requires,
   as a function of measurable corpus properties?
4. Can two embedding spaces from different models be aligned post hoc well
   enough to avoid a full re-embed on upgrade? Procrustes alignment works
   partially; nobody has made it good enough to trust an index to.
5. Uniformity is defined against the uniform distribution on the sphere. Is that
   the right target when the relevance structure is hierarchical rather than flat?

## 20. Chapter Summary

An embedding is a lossy compression of a document chosen so that a dot product
between two of them approximates relevance. That definition — not "a vector
representation of meaning" — is what generates every design decision in
{{part:11}}.

Three consequences dominate. **The score is ordinal**
({{eq:ranking-constraint}}): it has no absolute meaning, no cross-model
comparability, and thresholds on it are unjustified. **The geometry must be the
optimisation target**: representations learned as a by-product of some other
objective are anisotropic ({{eq:mean-cosine}}) and poorly aligned, which is why
contrastive training with InfoNCE ({{eq:infonce}}) exists and why centring alone
recovers part of the gap. **Batch size is a capacity knob, not a throughput
knob**, because in-batch negatives make the discrimination task harder as $N$
grows and {{eq:infonce-mi-bound}} caps the objective at $\log N$.

Alignment and uniformity ({{eq:alignment}}, {{eq:uniformity}}) make the
optimisation legible: they are computable without labels, they are in tension,
and they diagnose collapse and anisotropy that a loss curve hides. They must be
read together — {{sec:9-practical-example}} exhibits an encoder with the best
alignment of its group and nearly the worst retrieval, because alignment on its
own is minimised by collapse — and they are necessary rather than sufficient:
the same experiment improves every intrinsic metric with a transform that makes
retrieval worse.

Operationally, the fact to carry forward is that **an embedding model is a
versioned schema for an index**, with no incremental migration path. The rest of
this part builds on the geometry: {{ch:emb-similarity}} works out what the dot
product does in high dimension, {{ch:emb-models}} how to train and choose one,
and {{ch:emb-vector-db}} through {{ch:emb-reranking}} how to search the result.

## 21. Further Reading

{{cite:oord2018cpc}} for InfoNCE and the mutual-information bound — read
Section 2.3 for the derivation of {{eq:infonce-mi-bound}}.
{{cite:gao2021simcse}} for alignment/uniformity and the dropout-as-augmentation
trick; its anisotropy analysis is the clearest short treatment available.
{{cite:reimers2019}} for the original argument that BERT needs contrastive
fine-tuning to be a sentence encoder.
{{cite:karpukhin2020dpr}} for the dual encoder as a retrieval component and for
hard negatives.
{{cite:wang2022e5}} for the asymmetric-prefix convention and for where training
pairs come from at scale.

---
id: nlp-similarity
number: 78
part: VIII
tier: full
status: draft
requires: [nlp-bert, nlp-contextual, nlp-static-embeddings, math-norms,
           math-eigen, ml-pca, ml-metrics, tf-complexity]
provides: [sentence-embedding, bi-encoder, cross-encoder, mean-pooling,
           cls-pooling, anisotropy, siamese-network, contrastive-sentence-training,
           retrieve-then-rerank, semantic-textual-similarity, hard-negatives,
           embedding-benchmark]
citations: [reimers2019, devlin2019bert, muennighoff2023mteb, pennington2014,
            mikolov2013distributed, sanh2019, liu2019roberta, levy2015]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Explain why a good token-level encoder does not automatically give good
   sentence vectors, and demonstrate it.
2. Compare CLS and mean pooling and say why mean pooling usually wins on an
   unfine-tuned model.
3. Define anisotropy, measure it, and explain what it does to cosine similarity.
4. Derive the cost asymmetry between bi-encoders and cross-encoders, and show
   that retrieve-then-rerank follows from it rather than being a design choice.
5. Describe siamese training with a contrastive objective and explain the role of
   hard negatives.
6. Choose an embedding model from measurement on your own task, and explain why
   the leaderboard cannot make the choice for you.
7. Explain how everything in {{part:11}} and {{part:12}} rests on this chapter.

## 2. Why This Matters

**This is the chapter the rest of the book uses most.** Every retrieval system in
{{part:11}}, every RAG pipeline in {{part:12}}, every semantic-search feature and
every clustering of documents runs on the architecture defined here. If you read
one chapter of this part for its downstream value, it is this one.

**It also contains the part's most useful negative result.**
{{cite:reimers2019}} reports that out-of-the-box BERT sentence vectors — the CLS
token, or averaged token vectors — perform **worse** than averaged GloVe vectors
on semantic textual similarity. A 110M-parameter contextual model losing to a
lookup table calibrates expectations better than any benchmark table: a
representation that is excellent for one purpose can be actively bad for another,
and the training objective is what decides which.

**The bi-encoder/cross-encoder split is one of the most transferable
architectural patterns in the book.** It is the same shape as the cascade in
{{ch:nlp-extraction}} and the same shape as speculative decoding in
{{part:23}}: a cheap high-recall stage in front of an expensive precise one, with
the split point set by arithmetic rather than by taste. Here the arithmetic is
clean enough to derive completely.

**And it is where the contrastive objective from {{ch:nlp-static-embeddings}}
comes back.** Negative sampling trained word vectors in 2013; the same structure
— attract the positive, repel sampled negatives — trains sentence encoders now.
Recognising it as the same objective is what makes {{ch:emb-models}} feel like a
continuation rather than a new subject.

## 3. Prerequisites

{{ch:nlp-bert}} for the encoder, the CLS convention, and why MLM does not train a
sentence representation. {{ch:nlp-contextual}} for per-token contextual vectors
and the anisotropy noted there. {{ch:nlp-static-embeddings}} for the contrastive
objective and the averaged-GloVe baseline. {{ch:math-norms}} for cosine
similarity. {{ch:math-eigen}} and {{ch:ml-pca}} for the principal-component
analysis used to measure and remove anisotropy. {{ch:ml-metrics}} for ranking
metrics. {{ch:tf-complexity}} for the encoder cost that drives
{{sec:6-mathematical-foundation}}'s argument.

## 4. Intuitive Explanation

You have 1,000,000 documents and a query. Which document answers it?

**The accurate way** is to show the model both texts at once and let attention
relate every query token to every document token. That is a **cross-encoder**: it
takes a *pair* and returns a score. It is as accurate as the encoder allows,
because nothing is hidden from it.

It is also unusable. One million documents means one million forward passes per
query — at 10 ms each, about three hours.

**The cheap way** is to encode each document once into a single vector, in
advance, and encode the query into a vector at request time, then compare with a
dot product. That is a **bi-encoder**. The million documents are encoded once,
offline. The query costs one forward pass, and a million dot products take
milliseconds.

The bi-encoder is worse, and for a precise reason: **the document was encoded
without knowing the query.** Its vector must be a summary useful for every
possible question, and a fixed-size summary cannot preserve everything.

> NOTE: This is the same tradeoff as the fixed-size bottleneck in
> {{ch:tf-why-attention}} — compress-then-compare versus attend-across. There,
> attention won because sequences were short enough to afford it. Here the corpus
> is a million documents, and it is not. The answer is to use both, in order.

**Retrieve then rerank.** The bi-encoder scores everything and keeps the top 100.
The cross-encoder scores those 100 properly and reorders them. One hundred
forward passes instead of a million, with most of the accuracy. The architecture
falls out of the arithmetic, and {{sec:6-mathematical-foundation}} does the sums.

**Now the awkward part.** You have a pretrained BERT. It produces excellent token
vectors. Surely averaging them gives a sentence vector? It does, and it is bad —
worse than averaging GloVe vectors. Two reasons, both fixable: nothing in
{{eq:mlm-objective}} ever asked for a sentence representation, and the vectors
are **anisotropic**, all crowded into a narrow cone so that every pair of
sentences looks similar. Fine-tuning with a contrastive objective on sentence
pairs fixes both, and that fine-tuning is what an "embedding model" is.

**The mental model:** a sentence embedding is a lossy summary optimised so that
*distance in the summary space* matches the relation you care about — and which
relation that is depends entirely on what you trained on. Where it breaks down:
"similar" is not one thing. A model trained for paraphrase similarity will
happily rank a contradiction of your query above its answer, because a
contradiction is lexically and topically very close.

## 5. Formal Explanation

### 5.1 Two architectures

**Cross-encoder.** Concatenate and score jointly:

$$
s_{\text{cross}}(a,b) = \vec{w}\T\,f_\theta\big(\texttt{[CLS]}\ a\ \texttt{[SEP]}\ b\big)_{\texttt{[CLS]}}
$$ (eq:cross-encoder)

**Bi-encoder.** Encode separately, compare geometrically:

$$
s_{\text{bi}}(a,b) = \cos\big(g_\theta(a),\ g_\theta(b)\big),
\qquad g_\theta(x) = \text{pool}\big(f_\theta(x)\big)
$$ (eq:bi-encoder)

The functional difference: {{eq:cross-encoder}} lets every token of $a$ attend to
every token of $b$; {{eq:bi-encoder}} forbids any interaction until the two have
each been compressed to $d$ numbers.

**The bi-encoder is a strictly smaller hypothesis class.** Any function it can
express, a cross-encoder can — set the cross-encoder to ignore the interaction —
and the converse fails. So the bi-encoder can never be more accurate; it is
faster, which is the entire argument for it.

### 5.2 Pooling

Given token vectors $\vec{h}_1,\dots,\vec{h}_T$ and an attention mask $m_i$:

$$
\text{pool}_{\text{cls}} = \vec{h}_{\texttt{[CLS]}},
\qquad
\text{pool}_{\text{mean}} = \frac{\sum_{i} m_i\vec{h}_i}{\sum_i m_i}
$$ (eq:pooling)

**Mask-aware averaging is not optional.** Averaging over padding positions mixes
padding embeddings into the sentence vector, and the amount mixed in depends on
the batch's longest sequence — so the same sentence gets different vectors in
different batches. It is a silent, reproducibility-destroying bug.

**Why mean pooling usually beats CLS without fine-tuning.** Nothing in
{{eq:mlm-objective}} trains `[CLS]` to summarise. It is trained by NSP in
original BERT — a task {{cite:liu2019roberta}} showed was useless
({{ch:nlp-bert}}) — and by nothing at all in models that dropped NSP. So `[CLS]`
is an arbitrary position whose representation was never optimised for the job.
The mean at least aggregates positions that *were* optimised.

After contrastive fine-tuning the gap largely closes, because now something is
training the pooled vector directly.

### 5.3 Anisotropy

Define the **mean cosine similarity** of a representation over a corpus:

$$
\bar{s} = \E_{x\ne y}\big[\cos(\vec{v}_x, \vec{v}_y)\big]
$$ (eq:mean-cosine)

For $d$-dimensional vectors drawn isotropically, $\bar{s} \approx 0$. Pretrained
contextual encoders instead give $\bar{s}$ that is large and positive — the
vectors occupy a narrow cone rather than filling the space.

The consequence is a loss of *resolution*, not of information. All similarities
are compressed into a short interval near $\bar{s}$, so the difference between a
relevant and an irrelevant document may be 0.86 against 0.84 — a real signal
crushed into two decimal places, where floating-point noise and threshold
selection become fragile.

Two standard fixes:

$$
\vec{v}' = \vec{v} - \bar{\vec{v}}
 \quad\text{(centering)},
\qquad
\vec{v}'' = \vec{v}' - \sum_{k=1}^{K}(\vec{v}'\T\vec{u}_k)\vec{u}_k
 \quad\text{(remove top-}K\text{ PCs)}
$$ (eq:anisotropy-fix)

with $\vec{u}_k$ the top principal components ({{ch:ml-pca}}). Both are
post-processing, both cost almost nothing, and both are subsumed by contrastive
fine-tuning, which produces a well-spread space directly.

### 5.4 Siamese training

{{cite:reimers2019}} fine-tunes one encoder applied twice — shared weights, hence
*siamese* — on sentence pairs. Three objectives are used in practice.

**Classification** over an entailment dataset, with the feature vector

$$
\vec{u} = g_\theta(a),\quad \vec{v} = g_\theta(b),
\qquad
P = \softmax\big(\mat{W}[\vec{u};\vec{v};|\vec{u}-\vec{v}|]\big)
$$ (eq:sbert-classification)

The $|\vec{u}-\vec{v}|$ term is the one that matters: without an explicit
elementwise difference, the head must learn comparison from concatenation, and it
does so badly.

**Regression** on graded similarity scores, minimising
$\big(\cos(\vec{u},\vec{v}) - y\big)^2$.

**Contrastive**, which is what modern embedding models use. For a batch of $N$
positive pairs $(a_i, b_i)$, treat the other $N-1$ documents as negatives:

$$
\Loss_i = -\log
 \frac{\exp\big(\cos(\vec{u}_i,\vec{v}_i)/\tau\big)}
      {\sum_{j=1}^{N}\exp\big(\cos(\vec{u}_i,\vec{v}_j)/\tau\big)}
$$ (eq:infonce)

**This is {{eq:negative-sampling}} with a softmax normalisation instead of
independent logistic terms**, and with in-batch negatives instead of sampled
ones. The temperature $\tau$ controls how sharply the loss distinguishes near
misses; small $\tau$ makes the model sensitive to hard negatives and unstable,
large $\tau$ makes it lazy.

**Hard negatives are the main quality lever.** A random negative is trivially
separable — different topic, different vocabulary — so it produces almost no
gradient. A hard negative is topically identical and semantically wrong, and it
is where the useful signal is. In-batch negatives are free and mostly easy;
mined hard negatives are expensive and mostly what matters.

### 5.5 Retrieve then rerank

$$
\text{top-}k \xleftarrow{\ \text{bi-encoder}\ } \text{corpus},
\qquad
\text{reorder} \xleftarrow{\ \text{cross-encoder}\ } \text{top-}k
$$ (eq:retrieve-rerank)

Recall is bounded by the first stage — a document the bi-encoder does not
retrieve can never be reranked into position — so the first stage is tuned for
**recall@k** and the second for precision at the top. Choosing $k$ is choosing
where to spend: larger $k$ raises the recall ceiling and costs $k$ cross-encoder
passes.

## 6. Mathematical Foundation

### 6.1 The cost asymmetry, derived

Let $n$ be corpus size, $q$ queries, $C_{\text{enc}}$ the cost of one encoder
forward pass, and $C_{\text{dot}}$ the cost of one $d$-dimensional dot product.

**Cross-encoder over everything:**

$$
C_{\text{cross}} = q\,n\,C_{\text{enc}}
$$ (eq:cross-cost)

**Bi-encoder:**

$$
C_{\text{bi}} = \underbrace{n\,C_{\text{enc}}}_{\text{offline, once}}
 + \underbrace{q\big(C_{\text{enc}} + n\,C_{\text{dot}}\big)}_{\text{per query}}
$$ (eq:bi-cost)

The ratio of the per-query terms:

$$
\frac{C_{\text{cross}}^{\text{query}}}{C_{\text{bi}}^{\text{query}}}
 = \frac{n\,C_{\text{enc}}}{C_{\text{enc}} + n\,C_{\text{dot}}}
 \;\xrightarrow[\ n \to \infty\ ]{}\;
 \frac{C_{\text{enc}}}{C_{\text{dot}}}
$$ (eq:cost-ratio)

**The asymptotic advantage is exactly the ratio of a forward pass to a dot
product.** For BERT-base at $T=128$, a forward pass is roughly
$2\times 110\text{M}\times 128 \approx 2.8\times 10^{10}$ FLOPs; a
768-dimensional dot product is about $1.5\times 10^3$ FLOPs.

$$
\frac{C_{\text{enc}}}{C_{\text{dot}}} \approx \frac{2.8\times10^{10}}{1.5\times10^3}
 \approx 1.9\times 10^{7}
$$ (eq:seven-orders)

$\square$

**Seven orders of magnitude.** This is not an optimisation; it is the difference
between a system existing and not existing, and it is why every retrieval system
has this shape.

### 6.2 The cascade cost

With reranking of the top $k$:

$$
C_{\text{cascade}}^{\text{query}} = C_{\text{enc}} + n\,C_{\text{dot}}
 + k\,C_{\text{enc}}
 = (k+1)C_{\text{enc}} + n\,C_{\text{dot}}
$$ (eq:cascade-cost)

Against the full cross-encoder's $n\,C_{\text{enc}}$, the saving is a factor of
about $n/(k+1)$ — for $n = 10^6$ and $k=100$, roughly **10,000x**, while the
reranker sees every candidate the retriever surfaced.

The quality cost is bounded by one quantity only:

$$
\text{accuracy}_{\text{cascade}} \le \text{recall@}k_{\text{bi}}
$$ (eq:recall-ceiling)

**Whatever the first stage misses is lost permanently.** So the retriever should
be evaluated on recall@k and never on precision@1 — which is the most common
evaluation error in RAG systems, and {{part:12}} returns to it.

### 6.3 Why random vectors are nearly orthogonal, and pretrained ones are not

For two independent vectors uniform on the unit sphere in $\R^d$, the cosine
similarity has mean 0 and variance $1/d$:

$$
\E[\cos] = 0,\qquad \Var[\cos] = \frac{1}{d}
$$ (eq:random-cosine)

so typical similarities are $O(1/\sqrt{d})$ — about $0.036$ at $d = 768$. **In
high dimensions, random directions are nearly orthogonal.**

Measured pretrained encoder representations give $\bar{s}$ far above this, and
often above $0.5$. That gap is anisotropy, and {{eq:random-cosine}} is the
reference point that makes it a measurement rather than an impression:
$\bar{s}$ should be compared against $0$ with a spread of $1/\sqrt{d}$, not
against nothing.

### 6.4 A worked cost comparison

Corpus $n = 10^6$, queries $q = 10^4$ per day, $C_{\text{enc}} = 10$ ms,
$C_{\text{dot}} = 1$ ns, $k = 100$.

**Full cross-encoder:**
$10^4 \times 10^6 \times 10\ \text{ms} = 10^{11}\ \text{ms} \approx 3.2$
**years of compute per day.**

**Bi-encoder:** offline $10^6\times 10\ \text{ms} = 2.8$ hours, once. Per query
$10\ \text{ms} + 10^6\times 1\ \text{ns} = 11$ ms, so
$10^4 \times 11\ \text{ms} = 110$ seconds per day.

**Cascade:** per query $101\times 10\ \text{ms} + 1\ \text{ms} = 1.01$ s, so
$10^4\times 1.01\ \text{s} = 2.8$ hours per day.

Three architectures, spanning **years, hours, and minutes** for the same task.
The cascade buys back most of the cross-encoder's accuracy for about 0.1% of a
full cross-encoder pass — and the fact that this arithmetic is so lopsided is why
there is essentially one retrieval architecture in production everywhere.

## 7. Internal Mechanics

```mermaid {#fig:bi-vs-cross caption="Bi-encoder and cross-encoder. The bi-encoder's documents are encoded once, offline, and never see the query; the cross-encoder sees both texts in one forward pass and must therefore run per pair. The cascade uses the first to reduce the candidate set the second must consider."}
graph TD
  subgraph BI["bi-encoder — offline once, then dot products"]
    A1["document"] --> B1["encoder"] --> C1["pool → vector"] --> D1[("vector index")]
    A2["query"] --> B2["encoder<br/>same weights"] --> C2["pool → vector"]
    C2 --> E1["cosine against index<br/>~1 ns each"]
    D1 --> E1
    E1 --> F1["top-k candidates"]
  end
  subgraph CROSS["cross-encoder — per pair, at query time"]
    F1 --> G["[CLS] query [SEP] document"]
    G --> H["encoder<br/>full attention across both"]
    H --> I["score"] --> J["reordered top-k"]
  end
  style D1 fill:#dfe,stroke:#5a5
  style H fill:#fde,stroke:#c69
```

**Asymmetry in what gets encoded when.** Documents are encoded at index time and
their vectors stored; queries are encoded at request time. This means a model
change requires **re-encoding the entire corpus**, which for a large index is a
migration project rather than a deployment. Version the index with the model that
produced it and never mix.

**Normalisation and the metric.** If vectors are L2-normalised, cosine similarity
and the dot product coincide, and maximum inner product search becomes nearest
neighbour search — which is what vector indexes are built for
({{ch:emb-ann}}). Normalise at write time, once, rather than at query time,
repeatedly.

**Where the reranker's advantage comes from.** In the cross-encoder, attention
computes query-token to document-token interactions directly — the term the
bi-encoder deleted when it compressed each side independently. That is the whole
of the quality difference, and it is why the gap is largest for queries whose
relevance depends on a specific overlapping detail rather than on topic.

**Batch encoding.** Documents vary in length and attention is $O(T^2)$
({{ch:tf-complexity}}), so a batch padded to its longest member wastes compute
quadratically. Sort by length before batching; this routinely halves indexing
time and changes nothing else.

## 8. Implementation

First the pooling and anisotropy measurement, against the {{eq:random-cosine}}
reference:

```python {tier=A name=pooling-and-anisotropy}
"""Pooling strategies, and anisotropy measured against the isotropic baseline."""
import numpy as np

rng = np.random.default_rng(0)
N, T, D = 500, 24, 128


def mean_cosine(V):
    """Equation (eq:mean-cosine), over all distinct pairs."""
    U = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    S = U @ U.T
    n = len(V)
    return float((S.sum() - np.trace(S)) / (n * (n - 1)))


def similarity_stats(V):
    U = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    s = (U @ U.T)[np.triu_indices(len(V), 1)]
    return float(s.mean()), float(s.std()), float(s.min()), float(s.max())


def top_pc_share(V, k=1):
    """Fraction of total variance in the top k principal components."""
    Xc = V - V.mean(0)
    sv = np.linalg.svd(Xc, compute_uv=False)
    return float((sv[:k] ** 2).sum() / (sv ** 2).sum())


# 1. The isotropic reference: mean 0, standard deviation 1/sqrt(d).
iso = rng.normal(size=(N, D))
print(f"isotropic reference, d={D}")
print(f"  mean cosine      {mean_cosine(iso):+.4f}   (equation eq:random-cosine "
      f"predicts 0.0000)")
print(f"  predicted spread {1 / np.sqrt(D):.4f}")

# 2. An anisotropic representation: two shared directions with positive,
#    varying coefficients plus noise — the shape a pretrained encoder's
#    vectors empirically take.
c1 = rng.normal(size=D)
c1 /= np.linalg.norm(c1)
c2 = rng.normal(size=D)
c2 -= (c2 @ c1) * c1
c2 /= np.linalg.norm(c2)
aniso = (np.abs(rng.normal(1.5, 0.8, (N, 1))) * c1
         + np.abs(rng.normal(1.0, 0.6, (N, 1))) * c2
         + 0.12 * rng.normal(size=(N, D)))

# 3. The two fixes of equation (eq:anisotropy-fix), measured on three axes.
centred = aniso - aniso.mean(0)
_, _, Vt = np.linalg.svd(centred, full_matrices=False)
stripped = centred - (centred @ Vt[:2].T) @ Vt[:2]

print()
print(f"{'representation':<18} {'mean cos':>10} {'sd':>7} {'range':>16} "
      f"{'PC1 var share':>15}")
for name, V in [("raw", aniso), ("centred", centred),
                ("centred, -PC1,2", stripped), ("isotropic ref", iso)]:
    m, sd, lo, hi = similarity_stats(V)
    print(f"{name:<18} {m:>+10.3f} {sd:>7.3f} "
          f"{f'[{lo:+.2f}, {hi:+.2f}]':>16} {top_pc_share(V):>15.3f}")

print("""
Read the columns separately, because they do not agree.

  * Centering removes the high mean entirely and INCREASES the spread — the
    similarity score becomes usable again. This is the reliable fix.
  * Removing principal components leaves the mean where centering put it and
    SHRINKS the spread. It removed variance, and variance is not automatically
    noise. On a ranking task it can cost accuracy.

Anisotropy compresses the usable range of the similarity score. It does not
necessarily destroy the ranking, which is why a system can have a badly
anisotropic space and still retrieve acceptably — and why fixing the geometry
is not the same as fixing the retrieval.""")

# 4. Mean pooling must respect the attention mask.
H = rng.normal(size=(3, T, D))
lengths = np.array([20, 8, 3])
mask = (np.arange(T)[None, :] < lengths[:, None]).astype(float)

naive = H.mean(axis=1)
masked = (H * mask[..., None]).sum(1) / mask.sum(1, keepdims=True)

print(f"\n{'sentence':>9} {'true length':>12} {'padding in mean':>17} "
      f"{'cos(naive, masked)':>20}")
for i, L in enumerate(lengths):
    c = float(naive[i] @ masked[i]
              / (np.linalg.norm(naive[i]) * np.linalg.norm(masked[i])))
    print(f"{i:>9} {L:>12} {1 - L / T:>16.0%} {c:>20.3f}")
print("\nThe shorter the sentence, the more padding the naive mean absorbs — "
      "and the batch's longest member decides how much, so the same sentence "
      "gets a different vector in a different batch.")
```

Now the cost model that {{sec:6-mathematical-foundation}} derived, computed
rather than asserted:

```python {tier=A name=retrieval-cost-model}
"""Bi-encoder, cross-encoder, and the cascade. Equations (eq:cross-cost) onward."""

CORPUS = 1_000_000
QUERIES_PER_DAY = 10_000
ENCODER_MS = 10.0            # one forward pass over a 128-token pair
DOT_NS = 1.0                 # one 768-dimensional dot product
TOP_K = 100

enc_s = ENCODER_MS / 1000
dot_s = DOT_NS / 1e9

cross_query = CORPUS * enc_s
bi_query = enc_s + CORPUS * dot_s
cascade_query = (TOP_K + 1) * enc_s + CORPUS * dot_s

SECONDS_PER_DAY = 86_400


def human(seconds):
    for unit, size in [("years", 365 * 86400), ("days", 86400),
                       ("hours", 3600), ("minutes", 60)]:
        if seconds >= size:
            return f"{seconds / size:,.1f} {unit}"
    return f"{seconds:,.2f} seconds"


print(f"corpus {CORPUS:,}  queries/day {QUERIES_PER_DAY:,}  "
      f"encoder {ENCODER_MS} ms  top-k {TOP_K}\n")
print(f"{'architecture':<18} {'per query':>14} {'compute per day':>20} "
      f"{'index build':>14}")
for name, per_query, build in [
        ("full cross-encoder", cross_query, 0.0),
        ("bi-encoder only", bi_query, CORPUS * enc_s),
        ("cascade", cascade_query, CORPUS * enc_s)]:
    print(f"{name:<18} {human(per_query):>14} "
          f"{human(per_query * QUERIES_PER_DAY):>20} {human(build):>14}")

print(f"\ncost ratio, equation (eq:cost-ratio): "
      f"{enc_s / dot_s:,.0f}x  (a forward pass against a dot product)")
print(f"cascade saving over full cross-encoder: "
      f"{cross_query / cascade_query:,.0f}x")
print(f"real-time feasibility at 1 query: "
      f"cross {cross_query:,.0f} s, cascade {cascade_query:.2f} s, "
      f"bi {bi_query * 1000:.1f} ms")

# The recall ceiling of equation (eq:recall-ceiling).
print(f"\n{'recall@k of the retriever':<28} {'cascade accuracy ceiling':>26}")
for recall in [0.80, 0.90, 0.95, 0.99]:
    print(f"{recall:<28.2f} {recall:>26.2f}")
print("\nNo reranker can exceed the retriever's recall@k — which is why the "
      "first stage is tuned for recall and never for precision@1.")
```

Finally, contrastive training with in-batch negatives, showing that hard
negatives are where the gradient lives:

```python {tier=A name=contrastive-sentence-training}
"""In-batch contrastive training — equation (eq:infonce) — and hard negatives."""
import math
import torch
import torch.nn as nn

torch.manual_seed(0)
D, N_TOPICS, DIM, TAU = 64, 8, 32, 0.07

# Synthetic sentence features: each topic has a centre, a "sentence" is that
# centre plus noise, and a positive pair is two sentences from one topic.
centres = torch.randn(N_TOPICS, D)
centres = centres / centres.norm(dim=1, keepdim=True)


def sample(topics, noise=0.12):
    return centres[topics] + noise * torch.randn(len(topics), D)


encoder = nn.Sequential(nn.Linear(D, 64), nn.Tanh(), nn.Linear(64, DIM))
opt = torch.optim.Adam(encoder.parameters(), lr=1e-2)


def embed(X):
    Z = encoder(X)
    return Z / (Z.norm(dim=1, keepdim=True) + 1e-9)


def infonce(a, b, tau=TAU):
    """Positives on the diagonal, every other column an in-batch negative."""
    S = embed(a) @ embed(b).T / tau
    return nn.functional.cross_entropy(S, torch.arange(len(a))), S


topics = torch.arange(N_TOPICS)
print(f"random-guess loss for {N_TOPICS} in-batch candidates = "
      f"log {N_TOPICS} = {math.log(N_TOPICS):.4f}")
for step in range(1, 401):
    loss, _ = infonce(sample(topics), sample(topics))
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step in (1, 100, 200, 300, 400):
        print(f"step {step:>4}: InfoNCE loss {loss.item():.4f}")

# Did the space separate by topic?
with torch.no_grad():
    labels = topics.repeat_interleave(6)
    Z = embed(sample(labels))
    S = Z @ Z.T
    same = labels[:, None] == labels[None, :]
    off = ~torch.eye(len(labels), dtype=bool)
    within = S[same & off].mean().item()
    between = S[~same].mean().item()

print(f"\nmean similarity, same topic:      {within:+.3f}")
print(f"mean similarity, different topic: {between:+.3f}")
print(f"separation:                       {within - between:+.3f}")

# Where does the gradient come from? An easy batch versus a hard one.
with torch.no_grad():
    easy = infonce(sample(topics), sample(topics))                     # 8 topics
    zeros = torch.zeros(N_TOPICS, dtype=torch.long)
    hard = infonce(sample(zeros), sample(zeros))                       # all one topic

print(f"\n{'batch':<28} {'loss':>8} {'P(correct)':>12} {'gradient signal':>17}")
for name, (l, S) in [("random negatives (easy)", easy),
                     ("same-topic negatives (hard)", hard)]:
    pc = S.softmax(-1).diag().mean().item()
    print(f"{name:<28} {l.item():>8.4f} {pc:>12.3f} {1 - pc:>17.3f}")

print("\nAfter training, random negatives are perfectly separated: the loss is "
      "~0 and so is the gradient, so those batches teach nothing further. The "
      "hard batch is back at the random-guess baseline — all the remaining "
      "learning is there, which is why hard-negative mining is the main quality "
      "lever in an embedding model.")
```

## 9. Practical Example

A documentation search over 200,000 pages. The team ships a bi-encoder using a
pretrained BERT with mean pooling, no fine-tuning, and results are poor in a
characteristic way: everything is *topically* right and specifically wrong. A
query about "rotating API keys" returns twelve pages about API keys, none about
rotation.

That symptom is diagnostic. Topical-but-not-specific is what a bi-encoder does
when its space was never trained to separate near neighbours: the fixed-size
summary preserved the topic and discarded the detail. The space is probably also
anisotropic, and the interesting question — which the measurement below answers —
is whether that is a second cause or merely a second symptom.

Three interventions, in increasing order of cost:

```python {tier=A name=search-quality-interventions}
"""Three interventions on a weak bi-encoder, evaluated on recall@k and MRR."""
import numpy as np

rng = np.random.default_rng(3)
D, N_TOPICS, N_ASPECTS, PER_CELL = 64, 25, 8, 5
N_QUERIES, K = 200, 50

# Every document has a TOPIC ("API keys") and an ASPECT ("rotating", "creating").
# A query names both, and the distractors share its topic while differing in
# aspect — which is the 'twelve pages about API keys, none about rotation'
# situation, and the one bi-encoders actually fail at.
topic_vec = rng.normal(size=(N_TOPICS, D))
topic_vec /= np.linalg.norm(topic_vec, axis=1, keepdims=True)
aspect_vec = rng.normal(size=(N_ASPECTS, D))
aspect_vec /= np.linalg.norm(aspect_vec, axis=1, keepdims=True)

meta = np.array([(t_, a) for t_ in range(N_TOPICS) for a in range(N_ASPECTS)
                 for _ in range(PER_CELL)])
N_DOCS = len(meta)
docs = (topic_vec[meta[:, 0]] + 0.9 * aspect_vec[meta[:, 1]]
        + 0.10 * rng.normal(size=(N_DOCS, D)))

gold = rng.choice(N_DOCS, N_QUERIES, replace=False)
queries = (topic_vec[meta[gold, 0]] + 0.9 * aspect_vec[meta[gold, 1]]
           + 0.20 * rng.normal(size=(N_QUERIES, D)))

# The weak encoder: a bottleneck that keeps the topic and attenuates the aspect,
# plus a shared offset direction. That is a summary which preserved the topic
# and discarded the detail — plus anisotropy on top.
shared = rng.normal(size=D)
shared /= np.linalg.norm(shared)
aspect_basis = np.linalg.qr(aspect_vec.T)[0][:, :N_ASPECTS]


def weak_encoder(X):
    aspect_part = (X @ aspect_basis) @ aspect_basis.T
    return (X - aspect_part) + 0.10 * aspect_part + 2.5 * shared


def normalise(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def evaluate(dv, qv, k=10):
    S = normalise(qv) @ normalise(dv).T
    order = np.argsort(-S, axis=1)
    ranks = np.array([np.where(order[i] == gold[i])[0][0] for i in range(N_QUERIES)])
    mean_cos = float(
        (normalise(dv) @ normalise(dv).T)[np.triu_indices(N_DOCS, 1)].mean())
    return dict(r1=float((ranks == 0).mean()), r10=float((ranks < k).mean()),
                mrr=float((1 / (ranks + 1)).mean()), cos=mean_cos), S


results = {}
dv, qv = weak_encoder(docs), weak_encoder(queries)
results["1. as shipped"], _ = evaluate(dv, qv)

# Intervention A: centre the space — equation (eq:anisotropy-fix).
mu = dv.mean(0)
results["2. + centering"], _ = evaluate(dv - mu, qv - mu)

# Intervention B: also remove the top principal component.
Xc = dv - mu
_, _, Vt = np.linalg.svd(Xc, full_matrices=False)
strip = lambda X: X - (X @ Vt[:1].T) @ Vt[:1]
results["3. + remove PC1"], S3 = evaluate(strip(Xc), strip(qv - mu))

# Intervention C: rerank the top K with a scorer that sees both texts jointly
# and can therefore use the aspect dimensions the summary compressed away.
ranks = []
for i in range(N_QUERIES):
    cand = np.argsort(-S3[i])[:K]
    joint = -np.linalg.norm(docs[cand] - queries[i], axis=1)
    reordered = cand[np.argsort(-joint)]
    pos = np.where(reordered == gold[i])[0]
    ranks.append(int(pos[0]) if len(pos) else K + 1)
ranks = np.array(ranks)
results[f"4. + rerank top-{K}"] = dict(
    r1=float((ranks == 0).mean()), r10=float((ranks < 10).mean()),
    mrr=float((1 / (ranks + 1)).mean()), cos=results["3. + remove PC1"]["cos"])

print(f"{N_DOCS} documents, {N_QUERIES} queries, {N_TOPICS} topics x "
      f"{N_ASPECTS} aspects\n")
print(f"{'stage':<22} {'recall@1':>10} {'recall@10':>11} {'MRR':>8} {'mean cos':>10}")
for name, m in results.items():
    print(f"{name:<22} {m['r1']:>10.3f} {m['r10']:>11.3f} {m['mrr']:>8.3f} "
          f"{m['cos']:>10.3f}")

retriever_recall = float(np.mean(
    [gold[i] in np.argsort(-S3[i])[:K] for i in range(N_QUERIES)]))
print(f"\nretriever recall@{K} = {retriever_recall:.3f}   "
      f"<- the ceiling from equation (eq:recall-ceiling)")
print(f"reranked recall@10 = {results[f'4. + rerank top-{K}']['r10']:.3f}, "
      f"which cannot exceed it")

print("""
Three things in this table are worth more than the headline improvement.

  * Centering fixed the geometry completely (mean cosine 0.8 -> 0.0) and did
    NOT fix the retrieval. Removing a principal component made recall@1 very
    slightly worse. The anisotropy was real, and it was not the problem.
  * Reranking moved recall@1 by an order of magnitude, because the failure was
    information the bi-encoder's summary had discarded — which is exactly what
    a joint scorer can recover and a geometric fix cannot.
  * Reranked recall@10 is capped by the retriever's recall@50. The reranker
    reorders the candidate set; it can never add to it.""")
```

**The order matters, and so does the negative result.** The free
post-processing goes first because it costs nothing and *rules out* a geometry
problem — which here it does, by fixing the geometry and changing nothing else.
That is a useful outcome rather than a wasted step: it tells the team the
anisotropy was cosmetic and the real loss was in the bottleneck, which is what
the reranker then confirms.

The general point is worth stating plainly, because the folk version of this
advice is wrong: **anisotropy makes similarity scores uninformative, not
rankings incorrect.** If you are thresholding on an absolute score, or reporting
similarity to a user, or correlating against human judgements, fix it. If you
are taking a top-$k$, it may cost you nothing at all — and removing principal
components discards variance, which is sometimes signal.

Fine-tuning the encoder on in-domain pairs — not shown here because it needs
labelled data — usually beats all three interventions, and is the right answer
once the team can produce a few thousand query-document pairs.

> PRODUCTION TIP: "Everything is topically right and specifically wrong" is the
> signature of a bi-encoder failing at the level a cross-encoder fixes. If the
> failures were topically *wrong*, the problem is the retriever or the index,
> not the reranker — and adding a reranker will not help.

## 10. Production Considerations

**Changing the embedding model means re-encoding the corpus.** Document vectors
are only comparable to query vectors from the same model. An upgrade is a
migration: build the new index alongside, verify on a held-out query set, then
switch. Store the model identity and version in the index metadata and refuse
queries from a mismatched encoder.

**Normalise at write time.** L2-normalising once at indexing makes the dot
product a cosine and lets the index use inner-product search directly
({{ch:emb-ann}}).

**Dimensionality is a storage decision.** At $n = 10^7$ documents,
768-dimensional float32 vectors are 30 GB; at 384 dimensions in int8, they are
3.8 GB. {{cite:sanh2019}}-style distilled encoders often produce smaller vectors
with a small quality cost, and that trade is usually worth taking for the
retrieval stage — but not for the reranker, where quality is the product.

**Set $k$ from measured recall.** {{eq:recall-ceiling}} makes $k$ the accuracy
ceiling of the whole system. Measure recall@k on real queries and pick the knee;
guessing 10 because it looks tidy is the most common way to cap a RAG system's
quality below what its generator could do.

**Batch by length when indexing.** Attention is quadratic
({{ch:tf-complexity}}), so length-sorted batching typically halves index build
time.

**What to monitor:** recall@k on a fixed labelled query set run daily, the
distribution of top-1 similarity scores (a distribution shift here indicates
corpus drift or an encoder mismatch), the abstention rate if using a score
threshold, and index build time.

## 11. Common Mistakes

**Beginners:**

*Using `[CLS]` from an unfine-tuned model.* Nothing trained it to summarise.
Mean pooling is the better default, and both are beaten by a model actually
trained for the job.

*Averaging over padding.* {{eq:pooling}} requires the mask. Without it a
sentence's vector depends on what else was in its batch.

*Comparing vectors from two different models.* The spaces are unrelated. A
cosine similarity between them is a number with no meaning.

**Experienced practitioners:**

*Evaluating the retriever with precision@1.* {{eq:recall-ceiling}} says the
retriever's job is recall@k; precision is the reranker's job. Optimising the
first stage for precision makes the whole cascade worse.

*Training only on in-batch negatives.* They are free and mostly easy, so the
gradient decays quickly. Mined hard negatives are where the remaining quality is
— the `contrastive-sentence-training` listing measures exactly this.

*Assuming "similar" is one relation.* A model trained on paraphrase pairs ranks
contradictions highly, because a contradiction is lexically and topically almost
identical to what it contradicts. If your task needs entailment, train on
entailment.

*Trusting a leaderboard for model choice.* {{cite:muennighoff2023mteb}} shows no
model dominates and that rankings depend on the task — and a widely used
leaderboard attracts optimisation against it. Use it to shortlist, then measure
on your own queries.

*Reusing thresholds after an encoder change.* Score distributions shift with the
model, so a similarity threshold of 0.8 means something different afterwards.
Recalibrate as part of the migration.

## 12. Failure Modes

**Anisotropy.** Vectors in a narrow cone; all similarities crowded near a high
mean. *Symptom:* "everything is 0.85 similar to everything", and thresholds that
work one week and not the next. *Detection:* {{eq:mean-cosine}} against the
$1/\sqrt{d}$ reference of {{eq:random-cosine}}. *Fix:* centering
{{eq:anisotropy-fix}}, or contrastive fine-tuning. **Note what this does not
fix:** anisotropy degrades the *score*, and a top-$k$ ranking may be entirely
unaffected — {{sec:9-practical-example}} measures a case where the geometry is
badly anisotropic and correcting it changes retrieval by nothing. Fix it when
you threshold on the score, report it, or correlate it against human judgement;
do not expect it to rescue a ranking.

**Topically right, specifically wrong.** The fixed-size summary preserved topic
and lost the distinguishing detail. *Symptom:* the failure mode in
{{sec:9-practical-example}}. *Fix:* a reranker, or fine-tuning on in-domain
pairs. *Not fixed by:* a larger $k$, which retrieves more of the same.

**Recall ceiling.** The correct document is not in the top $k$, so no reranking
recovers it. *Symptom:* a RAG system whose answers are confidently wrong on
questions whose answers are demonstrably in the corpus. *Detection:* measure
recall@k directly with known-answer queries. This is the single most
underdiagnosed failure in RAG systems.

**Asymmetric query and document lengths.** Queries are short and documents are
long, so their vectors come from different distributions even from the same
encoder. *Symptom:* similarity scores correlated with document length rather than
relevance. *Fix:* train with the asymmetry present, or use models designed for
it.

**Stale index.** Documents changed and their vectors did not. *Symptom:*
retrieval that gets slowly worse with no deployment to blame. *Detection:*
track the age distribution of index entries against source modification times.

**Score-threshold drift.** An absolute similarity threshold set on one corpus
does not transfer to another, or to the same corpus after it grows. *Fix:*
prefer relative criteria — top-$k$, or a margin against the $k+1$th score — over
absolute thresholds.

## 13. Alternatives

{#tbl:similarity-architectures caption="Ways to score the similarity of two texts, by how much interaction between them the architecture allows. Interaction and cost move together, and the whole design space is a choice of where on that line to sit."}

| Architecture | Interaction | Query cost over n docs | Corpus precompute | Where it is used |
|---|---|---|---|---|
| Averaged static vectors | none | $n$ dot products | trivial | high-volume first pass |
| Bi-encoder | none until pooling | $1$ pass + $n$ dots | $n$ passes | retrieval, everywhere |
| Late interaction | token-level, after encoding | $n$ small matmuls | $n$ passes, $T$ vectors each | precision-critical retrieval |
| Cross-encoder | full, every token pair | $n$ passes | none possible | reranking a shortlist |
| Cascade | staged | $1 + k$ passes + $n$ dots | $n$ passes | production default |

**What is approximating what.** The bi-encoder approximates the cross-encoder's
scoring function under a rank constraint — it is forced to factorise the score
into a product of independently computed vectors. Late interaction relaxes that
by keeping one vector per token and deferring the comparison, buying accuracy for
storage that grows by a factor of $T$. Averaged static vectors approximate the
bi-encoder without the encoder.

**Why the cascade is the answer nearly everywhere.**
{{eq:cascade-cost}} gives $n/(k+1)$ savings against a full cross-encoder while
the reranker still sees every candidate the retriever surfaced. The only
architectural question left is $k$, and {{eq:recall-ceiling}} says how to choose
it.

## 14. Evaluation

**Is the implementation correct?**

1. **Pooling respects the mask** — the same sentence must produce the identical
   vector in a batch of 1 and a batch of 64. This is a one-line test that catches
   the most common bug in this chapter.
2. **Normalisation** — every stored vector has unit norm, if the index assumes
   it.
3. **Determinism** — dropout disabled at inference; the same input gives
   bit-identical output.
4. **Encoder and index agree** — the index metadata names the model that built
   it, and the query encoder matches.

**Is the retrieval good?**

1. **Recall@k of the retriever**, on real queries with known answers. This is the
   primary number, and {{eq:recall-ceiling}} is why.
2. **MRR or nDCG after reranking** for end-to-end quality
   ({{ch:ml-metrics}}).
3. **Mean cosine** {{eq:mean-cosine}} as a geometry health check, compared
   against $1/\sqrt{d}$.
4. **Per-query-type breakdown.** Aggregate retrieval metrics hide the specific
   failure this chapter is about — a system can score well overall and fail
   entirely on queries whose answer depends on a distinguishing detail.

**On benchmarks.** {{cite:muennighoff2023mteb}} evaluates across eight task types
and finds no dominant model; the ranking depends on which task is measured. That
is the empirical basis for refusing to answer "which embedding model is best"
without naming the task. Shortlist from the leaderboard, decide on your own
queries — and note that a leaderboard in wide use is one that models are tuned
against, which is {{cite:wang2019glue}}'s story ({{ch:nlp-bert}}) repeating.

## 15. Advanced Concepts

**Late interaction.** {{maturity:ESTABLISHED}} Keep one vector per token instead
of one per document and compute a maximum-similarity aggregation at query time.
Recovers much of the cross-encoder's accuracy at a storage cost of roughly $T$
times a bi-encoder's, which for a large corpus is the binding constraint.

**Matryoshka representations.** {{maturity:EMERGING}} Train so that the first $m$
dimensions of a vector are themselves a usable embedding for any $m$. One model
serves a cheap 64-dimensional first pass and an accurate 768-dimensional rerank,
which collapses two indexes into one.

**Hard-negative mining.** {{maturity:ESTABLISHED}} Retrieve candidates with the
current model, take high-scoring non-relevant ones as negatives, and retrain.
The main quality lever in embedding training, and the reason strong embedding
models are trained in rounds rather than in one pass.

**Instruction-aware embeddings.** {{maturity:EMERGING}} Prefix the input with a
description of the intended relation ("represent this passage for retrieval"), so
one model serves several notions of similarity. A direct response to the
"'similar' is not one thing" problem in {{sec:11-common-mistakes}}.

**Embeddings from decoder-only models.** {{maturity:EMERGING}} Pooling the hidden
states of a generative model, usually with a contrastive fine-tune. Competitive,
and it undercuts the neat story that encoders own retrieval — the architecture
matters less than the objective the vectors were trained under.

## 16. Connection to Previous Chapters

**Backwards.** {{ch:nlp-bert}} supplied the encoder and the `[CLS]` token whose
inadequacy {{sec:5-formal-explanation}} explains — and the explanation is
{{cite:liu2019roberta}}'s NSP result. {{ch:nlp-static-embeddings}} supplied both
the averaged-GloVe baseline that beats unfine-tuned BERT and the contrastive
objective that {{eq:infonce}} restates. {{ch:nlp-contextual}} noted the anisotropy
that {{sec:5-formal-explanation}} finally treats. {{ch:math-norms}} supplied
cosine similarity, {{ch:ml-pca}} the component removal in
{{eq:anisotropy-fix}}, and {{ch:tf-complexity}} the encoder cost that makes
{{eq:seven-orders}} a seven-order-of-magnitude number rather than a vague
"expensive". {{ch:tf-why-attention}}'s fixed-size bottleneck is the bi-encoder's
limitation, restated at document scale.

**Forwards.** {{ch:emb-what-they-are}} and {{ch:emb-similarity}} generalise this
geometry; {{ch:emb-models}} is how these encoders are trained at scale;
{{ch:emb-ann}} is how the dot products in {{eq:bi-cost}} are made sublinear;
{{ch:emb-reranking}} is the second stage of {{eq:retrieve-rerank}} in full.
{{part:12}} builds on all of it, and {{ch:rag-failures}} is largely a catalogue
of {{eq:recall-ceiling}} being violated in production. {{part:25}} takes up
benchmark selection as its own discipline.

## 17. Exercises

**Beginner**

1. Why is `[CLS]` a poor sentence representation without fine-tuning?
2. State the cost of scoring one query against 1,000,000 documents with a
   cross-encoder at 10 ms per pair.
3. What does mean pooling need the attention mask for?

**Intermediate**

4. Using {{eq:cost-ratio}}, compute the bi-encoder's advantage for
   $C_{\text{enc}} = 5$ ms, $C_{\text{dot}} = 2$ ns.
5. A retriever has recall@10 of 0.85 and a perfect reranker. What is the system's
   maximum accuracy, and what does that imply about where to spend effort?
6. Explain why {{eq:random-cosine}} makes anisotropy measurable rather than
   impressionistic.

**Advanced**

7. Prove that the bi-encoder's hypothesis class is a strict subset of the
   cross-encoder's, and construct a relevance function no bi-encoder can express
   at any dimension.
8. Derive the optimal $k$ for {{eq:cascade-cost}} given a latency budget and a
   measured recall@k curve.
9. Explain why removing the top principal component helps similarity tasks even
   though it discards information, and state when it would hurt.

**Implementation**

10. Extend `pooling-and-anisotropy` with max pooling and with a learned attention
    pooling, and compare all four on a retrieval task.
11. Implement the recall@k curve: for the synthetic corpus in
    `search-quality-interventions`, plot recall against $k$ from 1 to 500 and
    locate the knee. Relate it to {{eq:cascade-cost}}.
12. Implement hard-negative mining: train with in-batch negatives, retrieve
    candidates, take the top-scoring non-relevant ones as negatives, retrain, and
    measure the improvement over a second round.
13. Build the mask-correctness test from {{sec:14-evaluation}} — the same
    sentence in a batch of 1 and a batch of 64 must give identical vectors — and
    verify it fails against a deliberately naive mean.

**Reasoning**

14. A RAG system answers confidently and wrongly on questions whose answers are
    in the corpus. Give the three most likely causes in order and the measurement
    that distinguishes them.
15. Explain why an embedding model trained on paraphrase pairs might rank a
    contradiction of the query above its answer, and what to train on instead.

## 18. Interview Questions

**Beginner**

1. What is a sentence embedding and what is it for?
2. What is the difference between a bi-encoder and a cross-encoder?
3. Why can you not compare vectors from two different embedding models?

**Intermediate**

4. Why is out-of-the-box BERT bad at sentence similarity?
5. What is anisotropy and how would you detect it?
6. Explain retrieve-then-rerank and why it exists.

**Senior**

7. Design search over 10 million documents with a 200 ms budget. Justify each
   component with arithmetic.
8. How do you choose an embedding model? What role does a leaderboard play?
9. Your retrieval quality degraded after an encoder upgrade. Walk through the
   diagnosis.

**Systems**

10. How do you deploy a new embedding model over an existing 50-million-document
    index with no downtime?
11. What do you monitor for a production retrieval system, and what does each
    metric catch?

## 19. Research Questions

**How much of the bi-encoder gap is the objective rather than the
architecture?** The usual explanation is the fixed-size bottleneck. Hold the
bottleneck fixed and vary only the training objective and negative-mining
strategy, and measure how much of the gap to a cross-encoder closes. If most of
it does, the standard explanation is wrong.

**What is the actual dimensionality of the useful subspace?** Matryoshka results
suggest a 64-dimensional prefix retains much of the quality of 768. Measure the
recall-versus-dimension curve on real corpora and find the knee — the answer
directly sets storage cost for every vector index in production.

**Can one embedding serve several notions of similarity?** Instruction-aware
embeddings claim so. Test it adversarially: construct pairs that are paraphrases
under one instruction and contradictions under another, and see whether one model
can place them correctly under both.

**Is the MTEB leaderboard still informative?** {{cite:muennighoff2023mteb}} is
now widely optimised against, which is the mechanism {{cite:wang2019glue}}
demonstrated for GLUE. Measure the correlation between leaderboard rank and
performance on fresh, unpublished retrieval tasks. A weak correlation would be
worth knowing and is not currently established.

## 20. Chapter Summary

A **cross-encoder** scores a pair of texts in one forward pass with full
attention across both {{eq:cross-encoder}}. A **bi-encoder** encodes each side
independently and compares the resulting vectors {{eq:bi-encoder}}. The
bi-encoder's hypothesis class is a strict subset of the cross-encoder's, so it
cannot be more accurate — it is faster, and {{eq:seven-orders}} shows by roughly
seven orders of magnitude, which is the difference between a system existing and
not existing.

**Retrieve-then-rerank follows from that arithmetic rather than from taste.** The
bi-encoder scores everything cheaply and the cross-encoder reorders a shortlist,
for a saving of about $n/(k+1)$ {{eq:cascade-cost}}. The one binding constraint
is {{eq:recall-ceiling}}: the cascade's accuracy cannot exceed the retriever's
recall@k, so the first stage is evaluated on recall and never on precision@1 —
and violations of this are the most underdiagnosed failure in RAG systems.

**A pretrained encoder does not give good sentence vectors.**
{{cite:reimers2019}} found unfine-tuned BERT worse than averaged GloVe at
semantic similarity, for two reasons. Nothing in {{eq:mlm-objective}} trains a
sentence representation — `[CLS]` was trained by NSP, which
{{cite:liu2019roberta}} showed taught nothing. And the vectors are anisotropic,
occupying a narrow cone so that all similarities crowd near a high mean;
{{eq:random-cosine}} gives the $1/\sqrt{d}$ reference that makes this a
measurement. Centering and top-component removal {{eq:anisotropy-fix}} are free
partial fixes; contrastive fine-tuning is the real one.

**Siamese contrastive training** {{eq:infonce}} is {{eq:negative-sampling}} from
{{ch:nlp-static-embeddings}} with a softmax normalisation and in-batch negatives
— the same objective that trained word vectors in 2013, now training sentence
encoders. Hard negatives are the main quality lever, because random negatives are
already separated and contribute almost no gradient.

Model choice cannot be delegated to a leaderboard:
{{cite:muennighoff2023mteb}} shows no model dominates and that ranking depends on
the task. Shortlist from the benchmark, decide on your own queries — and expect a
widely used leaderboard to be optimised against.

## 21. Further Reading

{{cite:reimers2019}} is the paper this chapter is built on and it is short. Read
§4 for the pooling comparison and §6 for the finding that matters most: BERT's
out-of-the-box sentence vectors underperforming averaged GloVe. That single table
is the best calibration available against assuming a bigger model is better at
every job.

{{cite:muennighoff2023mteb}} is best read as a set of tables. The argument is
entirely in the observation that the rankings reorder across task types, and the
useful takeaway is a procedure — shortlist, then measure locally — rather than a
result.

{{cite:mikolov2013distributed}} is worth rereading here, specifically §2.2 on
negative sampling, alongside {{eq:infonce}}. Ten years and several model
generations apart, the objective is the same shape, and seeing that directly is
worth more than being told it.

{{cite:sanh2019}} for the distillation that makes the retrieval stage affordable
at scale — the encoder in a bi-encoder is usually a distilled one, and the
reasoning is the cost argument of {{sec:10-production-considerations}}.

**Where to go next:** this is the last chapter of {{part:8}}. The part assessment
asks you to build a tokenizer from scratch and evaluate it honestly. Then
{{part:9}} leaves the encoder era for the models that superseded it — taking with
it the tokenizer of {{ch:nlp-subword}}, the transfer recipe of
{{ch:nlp-contextual}}, and the retrieval architecture of this chapter, all of
which the generative era inherited unchanged.

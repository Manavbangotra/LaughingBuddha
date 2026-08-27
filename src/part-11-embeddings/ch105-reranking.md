---
id: emb-reranking
number: 105
part: XI
tier: full
status: draft
requires: [emb-what-they-are, emb-similarity, emb-models, emb-ann, emb-hybrid,
           nlp-similarity, llm-routing]
provides: [cross-encoder-reranking, reranking, late-interaction, maxsim,
           first-stage-recall-ceiling, rerank-depth, multi-vector-retrieval,
           generative-reranking, retrieval-cascade]
citations: [nogueira2019monobert, nogueira2020monot5, khattab2020colbert,
            santhanam2022colbertv2, reimers2019, thakur2021beir,
            karpukhin2020dpr, formal2021splade]
---

## 1. Learning Objectives

By the end of this chapter you will be able to explain the bi-encoder /
cross-encoder distinction as a question of *when the query meets the document*,
and derive from it why a cross-encoder cannot be a first stage; compute rerank
depth from a latency budget and show why first-stage recall@$k$ is a hard
ceiling; place late interaction on the quality/storage curve between the two;
explain why generative reranking transfers zero-shot better than a
classification head; and recognise this as the same cascade that has appeared
four times already.

## 2. Why This Matters

Everything in {{part:11}} so far has been about making one number — a dot
product — carry as much relevance information as possible. This chapter is about
what happens when you stop insisting on that.

The gain is large. {{sec:9-practical-example}} measures a bi-encoder at nDCG@10
of 0.346 and the same system with a cross-encoder over its top 100 at 0.712 —
more than double, for a hundred model calls. **Reranking is usually the largest
single quality improvement available to a retrieval system**, larger than
switching embedding models and far cheaper than training one.

The constraint is equally sharp, and it is the operational fact this chapter
exists to install: **a reranker cannot recover a document the retriever did not
return.** First-stage recall@$k$ is a hard ceiling on everything downstream, and
teams routinely spend months improving a reranker that is already saturated
against a retriever nobody measured.

{{maturity:ESTABLISHED}} Cross-encoder reranking has been standard since 2019.
{{maturity:EMERGING}} LLM-based rerankers are displacing fine-tuned cross
encoders and the cost model is still moving.

## 3. Prerequisites

{{ch:nlp-similarity}} introduced the bi-encoder/cross-encoder split empirically;
{{ch:emb-what-they-are}} for the structural constraint that produces it;
{{ch:emb-models}} for what the first stage is trained to do;
{{ch:emb-ann}} and {{ch:emb-hybrid}} for where candidates come from;
{{ch:llm-routing}} for the cascade arithmetic this chapter reuses.

## 4. Intuitive Explanation

### One question: when does the query meet the document?

Everything follows from this.

A **bi-encoder** encodes them separately and compares the results. The document
never sees the query. That is what allows the corpus to be encoded once, offline,
and stored in an index — and it is also what forces the document's entire meaning
into a single vector chosen without knowing what will be asked of it.

A **cross-encoder** feeds query and document *together* through one model. Every
layer can compare them: attention can align a query term to its answer, notice a
negation, resolve a pronoun against the query's subject. Nothing is compressed
away, because nothing was compressed.

The cost is total. There is no document representation to precompute — the
computation depends on both arguments — so scoring a corpus of ten million
documents means ten million forward passes. **Per query.**

```text
   BI-ENCODER                         CROSS-ENCODER
   ─────────────────────────          ─────────────────────────
   encode(doc) once, offline          encode(query, doc) per pair
   compare with a dot product         full attention between them
                                      
   10M docs = 10M stored vectors      10M docs = 10M forward passes
   query cost: one ANN search         query cost: 10M x model
   quality: bounded by the vector     quality: bounded by the model
```

So the cross-encoder is strictly better and strictly unusable as a first stage,
which forces the architecture: **something cheap produces a short list, and the
expensive model orders it.** This is the fifth appearance of that pattern in the
book — after retrieve-then-rerank in {{ch:nlp-similarity}}, encoder-then-LLM in
{{ch:nlp-extraction}}, model routing in {{ch:llm-routing}}, and IVF-PQ-then-exact
in {{ch:emb-ann}} — and by now the right response to seeing it is recognition,
not derivation.

### The ceiling

The consequence people miss. If the retriever's top 100 does not contain the best
answer, no reranker will find it. The reranker only reorders what it was given.

So a retrieval system has two independent quality knobs and they are *not*
interchangeable: the first stage sets a ceiling, and the second stage determines
how close to it you get. **Improving a reranker that is already near its ceiling
does nothing at all**, and this is invisible unless first-stage recall@$k$ is
measured separately — which is why {{sec:14-evaluation}} insists on it.

## 5. Formal Explanation

### 5.1 The two scorers

$$ s_{\text{bi}}(q,d) = f(q)\T g(d), \qquad s_{\text{cross}}(q,d) = h\big([\,q\ ;\ d\,]\big) $$ (eq:bi-vs-cross)

The distinction is not architectural taste, it is *factorisability*.
$s_{\text{bi}}$ factors into a function of $q$ times a function of $d$;
$s_{\text{cross}}$ does not. Factorisability is exactly the property that permits
precomputation, and therefore indexing.

$$ \text{precomputable} \iff s(q,d) \text{ factors} \iff \text{the corpus can be an index} $$ (eq:factorisation-constraint)

Every retrieval architecture in this part is a position on how much
factorisability to give up.

### 5.2 The cascade, and its ceiling

Retrieve $k$ candidates, rerank, return the top $n$:

$$ \text{quality}(k) \;\leq\; \text{quality}^{*}\big(\mathcal{C}_k(q)\big), \qquad \mathcal{C}_k(q) = \text{top-}k \text{ by the first stage} $$ (eq:rerank-ceiling)

where $\text{quality}^{*}$ is what a *perfect* reranker achieves on that
candidate set. In recall terms the statement is exact:

$$ \text{recall@}n \text{ of the pipeline} \;\leq\; \text{recall@}k \text{ of the first stage} $$ (eq:recall-ceiling)

**{{eq:recall-ceiling}} is the most important inequality in the chapter.** It says
the two stages have separate failure modes, that they must be measured
separately, and that the first stage's metric is recall@$k$ — not recall@10, not
nDCG — because its job is a candidate set and nothing else.

### 5.3 Cost

$$ C_{\text{total}} = \underbrace{C_{\text{retrieve}}(k)}_{\text{ANN, } \sim\log N} + \underbrace{k \cdot C_{\text{cross}}}_{\text{dominant}} $$ (eq:rerank-cost)

The second term is linear in $k$ and $C_{\text{cross}}$ is thousands of times
$C_{\text{bi}}$, so **$k$ is a latency budget, not a quality parameter**:

$$ k^{*} = \left\lfloor \frac{L_{\text{budget}} - L_{\text{retrieve}}}{L_{\text{cross}}} \right\rfloor $$ (eq:rerank-depth)

Reranking is embarrassingly parallel across candidates, so $L_{\text{cross}}$ is
the *batched* per-document latency, which is usually far below the single-pair
figure. This is {{ch:llm-inference}}'s prefill arithmetic: reranking is
prefill-shaped work, compute-bound and parallel, and it should be run at the
largest batch that fits.

Note the shape difference from {{eq:cascade-cost}} in {{ch:llm-routing}}. There
the expensive stage ran on a *fraction* of requests; here it runs on *every*
request, over $k$ documents. Reranking is not a way to save money — it is a way
to spend a bounded amount of money for a large quality gain.

### 5.4 Late interaction

{{cite:khattab2020colbert}} occupies the middle. Keep one vector per *token*,
and score by summing, over query tokens, the best match among document tokens:

$$ s_{\text{late}}(q,d) = \sum_{i \in q} \max_{j \in d} \; \hat{f}(q)_i\T \hat{g}(d)_j $$ (eq:maxsim)

The document encoding still does not depend on the query, so
{{eq:factorisation-constraint}} is satisfied and the corpus remains
precomputable. What changes is that a document is no longer summarised by one
point: the $\max$ lets each query term find its own evidence, so a document
relevant to two unrelated queries for two unrelated reasons can serve both.

The price is storage:

$$ \frac{\text{storage}_{\text{late}}}{\text{storage}_{\text{single}}} = \bar{T} \cdot \frac{d_{\text{late}}}{d_{\text{single}}} $$ (eq:late-interaction-storage)

with $\bar{T}$ the mean tokens per document. ColBERT reduces $d_{\text{late}}$ to
128, but $\bar{T} \approx 100$ still gives 10–100×.
{{cite:santhanam2022colbertv2}} cut it by an order of magnitude with residual
compression — the same idea as {{ch:emb-ann}}'s product quantization, applied to
a different problem.

### 5.5 The single-vector bottleneck, stated precisely

Why late interaction exists. A document $d$ relevant to queries $q_1, q_2$ that
are themselves dissiminar requires

$$ \hat{g}(d)\T \hat{f}(q_1) \text{ large} \;\wedge\; \hat{g}(d)\T \hat{f}(q_2) \text{ large}, \qquad \hat{f}(q_1)\T \hat{f}(q_2) \approx 0 $$ (eq:bottleneck)

On the unit sphere, a single point cannot be close to two near-orthogonal
directions: the best it can do is the normalised bisector, scoring
$1/\sqrt{2} \approx 0.707$ with each while some specialist document scores 1.0
with one of them. **The document is squeezed between its own meanings**, and
{{sec:9-practical-example}} measures the cost.

Chunking is the practical response — split the document so each piece has one
meaning — and it is worth naming as what it is: **the poor practitioner's
multi-vector retrieval.** It works, and it moves the problem to chunk-boundary
selection rather than removing it ({{part:12}}).

## 6. Mathematical Foundation

### 6.1 Why generative scoring transfers better

{{cite:nogueira2019monobert}} adds a randomly-initialised classification head to
BERT and fine-tunes. {{cite:nogueira2020monot5}} instead has the model *generate*
the token `true` or `false`, and scores by the logit difference:

$$ s_{\text{monoT5}}(q,d) = \log P(\texttt{true} \given q, d) - \log P(\texttt{false} \given q, d) $$ (eq:generative-scoring)

The zero-shot transfer is much better, and the reason is worth stating because it
generalises well beyond reranking. A classification head is a new parameter
matrix with no pretrained meaning; everything it knows comes from the fine-tuning
data, so it inherits that data's distribution completely. **The tokens `true` and
`false` already have meaning in the pretrained model**, so
{{eq:generative-scoring}} routes the relevance decision through machinery that
pretraining already built.

This is the same argument that makes prompting work at all ({{ch:llm-prompting}}),
and it is why LLM-based rerankers and LLM-as-judge scoring
({{part:19}}) are the natural continuation rather than a separate technique.

### 6.2 Choosing the rerank depth

{{eq:recall-ceiling}} gives the ceiling and {{eq:rerank-cost}} the price. Model
first-stage recall as saturating:

$$ R(k) = 1 - e^{-\lambda k} \quad\Longrightarrow\quad \frac{dR}{dk} = \lambda e^{-\lambda k} $$ (eq:recall-saturation)

Marginal recall decays exponentially in $k$ while marginal cost is constant, so
**there is always a $k$ past which reranking deeper is waste**, and it is usually
much smaller than people set. The correct procedure is to measure $R(k)$ on your
own data and stop where it flattens, rather than to adopt a number from a paper —
$\lambda$ depends entirely on how good the first stage is.

### 6.3 Two knobs, and which one to turn

Decompose pipeline quality:

$$ Q_{\text{pipeline}} = \underbrace{R(k)}_{\text{first stage}} \times \underbrace{P(\text{correct order} \given \text{candidates})}_{\text{reranker}} $$ (eq:two-stage-decomposition)

approximately, and the diagnostic follows immediately:

- $R(k)$ low, reranker good → **improve retrieval or raise $k$.** Reranker work is
  wasted.
- $R(k)$ high, reranker weak → **improve the reranker.** Retrieval work is wasted.
- Both high → the remaining errors are in the labels or the embedding.

**Measure $R(k)$ before touching anything.** It is one number and it decides
which half of the system to work on, and {{sec:9-practical-example}} shows the
two halves differing by more than a factor of two in achievable quality.

## 7. Internal Mechanics

```mermaid {#fig:rerank-cascade caption="The retrieval cascade in full. Each stage sees fewer documents and scores them more precisely; the query and document meet later and later, until at the final stage they are processed together. The dashed line is the ceiling: nothing after the first stage can recover a document it did not return."}
flowchart LR
    Q["query"] --> A["first stage:<br/>ANN + BM25, fused"]
    A -->|"top k ~ 100"| B["late interaction<br/>(optional, eq:maxsim)"]
    B -->|"top ~ 30"| C["cross-encoder<br/>(eq:bi-vs-cross)"]
    C -->|"top n ~ 10"| D["results"]
    A -.->|"eq:recall-ceiling:<br/>a hard bound on everything right of here"| D
```

### 7.1 The three positions on the curve

| | query meets doc | corpus precomputable | storage/doc | quality |
|---|---|---|---|---|
| bi-encoder | never | yes | 1 vector | baseline |
| late interaction | at scoring | yes | $\bar{T}$ vectors | between |
| cross-encoder | at layer 1 | **no** | n/a | best |

Reading the middle column as the organising axis: **you may precompute exactly as
much as your scorer factorises.** Late interaction factorises the encoding but
not the comparison, which is why it keeps the index and loses the storage.

### 7.2 What a reranker actually fixes

Worth being concrete, because "better relevance" is not actionable. Cross
encoders systematically fix:

- **Negation.** "documents *not* about tax" — a bi-encoder embeds this almost
  identically to "documents about tax", since the topic dominates the vector.
- **Term binding.** "papers by Smith citing Jones" versus "papers by Jones citing
  Smith" — identical bags of concepts, different relations.
- **Numeric and unit conditions.** "under 50mg" is a comparison, not a topic.
- **Multi-hop conditions within one document**, where relevance depends on two
  facts appearing together.

Each is a *relational* property between query and document, which is precisely
what {{eq:bi-vs-cross}}'s factorisation destroys. If your query log contains few
of these, reranking will help less than the literature suggests — and if it is
full of them, a reranker is the highest-return change available.

### 7.3 Serving a reranker

The reranker is a second model in the request path, and the operational
consequences are different from the retriever's in three ways worth planning for.

**It is the tail-latency risk.** {{eq:rerank-cost}}'s dominant term is
$k \cdot C_{\text{cross}}$, so anything that makes the model slower — a longer
document, a colder cache, a busier GPU — is multiplied by $k$. The retriever's
latency is roughly constant; the reranker's is not.

**Its batch is fixed and known.** Unlike a generative model
({{ch:llm-inference}}), a reranker's work is exactly $k$ pairs, available all at
once, with no autoregressive dependency. That makes it the easiest thing in a
retrieval stack to saturate a GPU with — and the most wasteful to run one pair at
a time.

**It is optional at runtime.** Because {{eq:recall-ceiling}} guarantees the
first stage already returned a usable ordering, a reranker can be dropped under
load and the system degrades rather than fails. Building that fallback is a few
lines and it converts the reranker from an availability risk into a quality knob.

### 7.4 Distillation closes the loop

The cross-encoder's scores are a graded relevance signal over any pair you care
to evaluate, so they make excellent training targets for the bi-encoder
({{ch:emb-models}}). This *raises the ceiling* rather than working beneath it,
and it structurally solves the false-negative problem
{{ch:emb-models}} identified: a mined false negative receives a high target score
from the cross-encoder instead of a wrong binary label.

**The strongest open retrievers are distilled rather than contrastively trained
alone**, and this is why.

## 8. Implementation

```python {tier=A name=rerank-cascade}
"""The rerank cascade, its cost, and its ceiling.

Ground truth is a graded relevance label per document. Two scorers see it
through different amounts of noise:

  bi-encoder     -- cheap, noisy; scores the whole corpus
  cross-encoder  -- expensive, precise; scores only what it is given

We sweep the rerank depth k and report three things: the first stage's recall@k
of the true top-10 (which is eq:recall-ceiling), the nDCG@10 a PERFECT reranker
would reach on that candidate set (the hard ceiling), and the nDCG@10 the actual
cross-encoder reaches.
"""
import numpy as np

rng = np.random.default_rng(13)

N_DOC, N_QUERY, N_REL = 20_000, 300, 20
BI_NOISE, CROSS_NOISE, LABEL_NOISE = 1.0, 0.45, 0.25
DEPTHS = [10, 20, 50, 100, 200, 500, 1000]


def ndcg(order, rel, k=10):
    gains = rel[order[:k]]
    discount = np.log2(np.arange(2, k + 2))
    dcg = np.sum((2 ** gains - 1) / discount)
    ideal = np.sort(rel)[::-1][:k]
    idcg = np.sum((2 ** ideal - 1) / discount)
    return dcg / idcg if idcg > 0 else 0.0


def make_query():
    """Graded relevance labels, plus the two noisy views of them."""
    rel = np.zeros(N_DOC)
    rel[rng.choice(N_DOC, N_REL, replace=False)] = rng.integers(1, 4, N_REL)
    latent = rel + rng.normal(scale=LABEL_NOISE, size=N_DOC)
    return (rel,
            latent + rng.normal(scale=BI_NOISE, size=N_DOC),
            latent + rng.normal(scale=CROSS_NOISE, size=N_DOC))


print(f"{'k':>6}{'recall@k of top-10':>21}{'nDCG@10 oracle':>17}"
      f"{'nDCG@10 actual':>17}{'CE calls':>10}")
print("-" * 71)

table = {}
for k in DEPTHS:
    recalls, oracle, actual = [], [], []
    for _ in range(N_QUERY):
        rel, bi, cross = make_query()
        cand = np.argpartition(-bi, k)[:k]
        best10 = set(np.argsort(-rel)[:10].tolist())
        recalls.append(len(best10 & set(cand.tolist())) / 10)
        oracle.append(ndcg(cand[np.argsort(-rel[cand])], rel))    # perfect rerank
        actual.append(ndcg(cand[np.argsort(-cross[cand])], rel))  # real rerank
    table[k] = (np.mean(recalls), np.mean(oracle), np.mean(actual))
    print(f"{k:>6}{table[k][0]:>21.4f}{table[k][1]:>17.4f}"
          f"{table[k][2]:>17.4f}{k:>10}")

bi_only, cross_all = [], []
for _ in range(N_QUERY):
    rel, bi, cross = make_query()
    bi_only.append(ndcg(np.argsort(-bi), rel))
    cross_all.append(ndcg(np.argsort(-cross), rel))
bi_only, cross_all = float(np.mean(bi_only)), float(np.mean(cross_all))

print(f"\n{'bi-encoder alone':<24}nDCG@10 {bi_only:.4f}   (0 CE calls)")
print(f"{'cross-encoder on ALL':<24}nDCG@10 {cross_all:.4f}   ({N_DOC} CE calls)")

r100, o100, a100 = table[100]
print(f"""
Start with the two baselines. The bi-encoder alone reaches {bi_only:.4f}. Running
the cross-encoder over the entire corpus reaches {cross_all:.4f} -- and costs
{N_DOC} forward passes PER QUERY, which is why nobody does it. The whole cascade
exists to capture as much of that gap as {N_DOC // 100}x less compute can buy.

At k=100 it captures a lot: nDCG@10 goes from {bi_only:.4f} to {a100:.4f} for one
hundred cross-encoder calls. That is the single largest quality improvement
available to most retrieval systems, and it requires training nothing.

Now the two ceiling columns, which are the point of the experiment. At k=100 the
first stage returns only {r100:.1%} of the true top-10, so a PERFECT reranker
could reach no more than {o100:.4f} on that candidate set -- and the real
cross-encoder reaches {a100:.4f}, within {100 * (o100 - a100) / o100:.0f}% of it.

Read that as a diagnosis, because it is the one eq:two-stage-decomposition
prescribes. The reranker is nearly saturated against its candidate set. Improving
it can buy at most the gap to the oracle column; improving the FIRST STAGE, or
simply raising k, moves the oracle column itself. A team optimising the reranker
here would be working on the smaller half of the problem, and would have no way
of knowing without this table.""")
```

```python {tier=A name=single-vector-bottleneck}
"""What one vector per document costs, and what MaxSim recovers.

Each document covers several unrelated ASPECTS -- a product page with a
description, a spec table, and a review section; a paper with a method and an
application. Queries target ONE aspect.

  single vector   -- the mean of the aspect vectors (eq:bottleneck)
  late interaction -- one vector per aspect, scored by MaxSim (eq:maxsim)

Both keep the corpus precomputable. Only the storage differs.
"""
import numpy as np

rng = np.random.default_rng(29)

N_DOC, DIM, N_QUERY = 4000, 64, 500


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


print(f"{'aspects/doc':>12}{'single-vector':>15}{'MaxSim':>10}"
      f"{'storage ratio':>15}{'mean cos(aspects)':>19}")
print("-" * 71)

for n_aspect in [1, 2, 4, 8]:
    # Aspects within a document are unrelated to each other -- that is what
    # makes the document hard to summarise with one point.
    aspects = unit(rng.normal(size=(N_DOC, n_aspect, DIM)))
    pooled = unit(aspects.mean(axis=1))

    # Mean cosine between two aspects of the same document, for reference.
    if n_aspect > 1:
        cos_within = float(np.mean(np.einsum('nd,nd->n',
                                             aspects[:, 0], aspects[:, 1])))
    else:
        cos_within = 1.0

    hits_single, hits_late = [], []
    for _ in range(N_QUERY):
        i = int(rng.integers(0, N_DOC))
        a = int(rng.integers(0, n_aspect))
        # A query about ONE aspect of document i, with a little noise.
        q = unit(aspects[i, a] + rng.normal(scale=0.35, size=DIM))

        single = pooled @ q
        late = np.max(aspects @ q, axis=1)          # MaxSim over aspects

        hits_single.append(float(i in np.argpartition(-single, 10)[:10]))
        hits_late.append(float(i in np.argpartition(-late, 10)[:10]))

    print(f"{n_aspect:>12}{np.mean(hits_single):>15.3f}{np.mean(hits_late):>10.3f}"
          f"{n_aspect:>14}x{cos_within:>19.3f}")

print("""
At one aspect per document the two are identical -- the mean of one vector is
that vector, so there is nothing to lose and MaxSim buys nothing. Every row below
that is the cost of compression.

As documents cover more unrelated aspects, the single-vector recall falls while
MaxSim holds up. The mechanism is eq:bottleneck: averaging near-orthogonal
directions lands the document vector on their bisector, which is far from all of
them. With two orthogonal aspects the best a single point can score against
either is 1/sqrt(2) = 0.707, while a document dedicated to one of them scores
1.0 -- so the generalist loses to specialists on every query, including the ones
it is genuinely relevant to.

MaxSim keeps the aspects separate and lets each query find its own evidence. Note
what it does NOT give up: the document encoding still does not depend on the
query, so eq:factorisation-constraint holds and the corpus is still
precomputable. That is the entire reason late interaction is a retrieval method
and a cross-encoder is not.

What it costs is the storage-ratio column, linear in aspects per document. In
ColBERT the unit is a token rather than an aspect, so the ratio is 10-100x, and
that number -- not quality -- is the argument against multi-vector retrieval.

Chunking is the cheap approximation of this table: split the document so each
piece has one aspect, and you are back on the top row with a single vector per
piece. It works, and it moves the problem to choosing the boundaries.""")
```

## 9. Practical Example

**The cascade.** The bi-encoder alone reaches nDCG@10 of 0.346. A cross-encoder
over the entire corpus reaches 0.944 and costs 20,000 forward passes per query.
Reranking the top 100 reaches 0.712 — **more than double the bi-encoder, for a
two-hundredth of the exhaustive cost, and it requires training nothing.** That
ratio is why reranking is the first thing to add to a retrieval system.

**The ceiling is the real lesson.** At $k=100$ the first stage returns only
49.7% of the true top-10, so a *perfect* reranker on that candidate set is
capped at nDCG 0.723 — and the actual cross-encoder reaches 0.712, within 1.6%
of its own ceiling.

That is a diagnosis, and it is the one {{eq:two-stage-decomposition}} prescribes.
**The reranker is all but saturated; the first stage is the bottleneck.**
Improving the reranker can buy at most the remaining 1.6%, while the gap between
that ceiling and exhaustive scoring is 23% — fifteen times larger. Raising $k$
moves the ceiling itself: at $k=1000$ recall reaches 0.804 and actual nDCG 0.906.

Notice also how the reranker's own gap *grows* with $k$ — 1.6% at $k=100$, 4.3%
at $k=1000$. **Which stage is the bottleneck is not a fixed property of the
system; it depends on $k$.** A team without this table would have no way of
knowing which half to work on, and would be wrong at one $k$ and right at
another.

> **IMPORTANT:** Report first-stage recall@$k$ and the oracle-rerank ceiling
> alongside end-to-end quality. Two numbers, computed from data you already have,
> and together they say which component to work on. This is the single most
> under-measured quantity in production retrieval.

**The bottleneck.** The second listing isolates why late interaction exists. At
one aspect per document, single-vector and MaxSim are identical at 0.490 — there
is nothing to compress. At two aspects the single vector falls to 0.194 while
MaxSim holds 0.436; at eight, 0.046 against 0.236 — **a fivefold difference from
the representation alone**, with the same encoder and the same queries.

That is {{eq:bottleneck}} exactly: the aspects are near-orthogonal (measured mean
cosine ≈ 0.000), so averaging them lands the document on a bisector far from all
of them, and the generalist loses to specialists even on queries it genuinely
answers.

MaxSim keeps the corpus precomputable — {{eq:factorisation-constraint}} still
holds — which is what makes it a retrieval method rather than a reranker. It pays
in storage, linearly in vectors per document, and that number is the whole
argument against it.

## 10. Production Considerations

**Set $k$ from the latency budget** ({{eq:rerank-depth}}), then check where
$R(k)$ flattens ({{eq:recall-saturation}}). If it flattened well below your $k$,
you are paying for nothing.

**Batch the reranker.** It is prefill-shaped, parallel work
({{ch:llm-inference}}); scoring 100 candidates one at a time wastes the
accelerator entirely.

**Truncate documents for the reranker deliberately.** Cross-encoder cost is
quadratic in sequence length, so the passage you feed it is a cost decision, and
feeding the whole document is usually the wrong one.

**Cache aggressively.** Query-document pairs repeat far more than query strings
do, and the cache key is the pair.

**Consider an LLM reranker before training a cross-encoder.**
{{eq:generative-scoring}}'s zero-shot transfer is good enough that a prompted
model often beats a cross-encoder fine-tuned on adjacent data, with no training
at all. Measure both; the cost model differs by an order of magnitude and the
quality frequently does not.

**Distil the reranker into the retriever** ({{sec:7-internal-mechanics}}). It
raises the ceiling instead of working under it.

**Degrade gracefully.** If the reranker times out, return the first stage's
order. It is worse, not broken, and a cascade that fails closed turns a quality
regression into an outage.

**Version the reranker and the retriever together.** The rerank depth $k$ was
chosen against a particular first stage's $R(k)$ curve
({{eq:recall-saturation}}), so improving the retriever silently makes the old $k$
wrong — usually too large, since better recall saturates sooner. Re-derive $k$
whenever either stage changes, and treat the pair as one deployed unit rather
than two independently upgradeable components.

## 11. Common Mistakes

**Optimising the reranker without measuring first-stage recall.** The chapter's
central error. {{eq:recall-ceiling}}.

**Measuring the first stage with nDCG@10.** Its job is a candidate set; the
metric is recall@$k$.

**Setting $k$ from a paper.** {{eq:recall-saturation}}'s $\lambda$ depends on
your first stage.

**Reranking with an unbatched loop.**

**Assuming a reranker fixes retrieval.** It reorders; it cannot retrieve.

**Using a reranker trained on one domain without checking transfer.**
{{cite:thakur2021beir}} applies to rerankers too, and
{{eq:generative-scoring}} is the reason to prefer generative scoring when you
must.

**Feeding whole documents to the cross-encoder** when a passage would do, at
quadratic cost.

## 12. Failure Modes

**Ceiling saturation.** Reranker improvements stop moving end-to-end quality.
Diagnosable only with the oracle column.

**Latency blowup under load.** {{eq:rerank-cost}} is linear in $k$ and the
reranker is the dominant term, so a traffic spike hits it hardest — exactly when
you can least afford it.

**Domain shift on the reranker.** Trained on web search, deployed on legal
documents. Fails quietly and looks like a retrieval problem.

**Position bias in listwise LLM rerankers.** Ordering the candidates by
first-stage score biases the LLM toward keeping that order — so the reranker
appears to agree with the retriever and is partly just echoing it. Shuffle the
input order.

**Cache poisoning across model versions.** A cached score from the previous
reranker, served after an upgrade.

**Silent truncation.** The document exceeds the cross-encoder's length limit and
the relevant passage was past the cut.

## 13. Alternatives

**No reranker.** Correct when the first stage is strong and latency is tight.
Measure the oracle gap before deciding — if it is small, there is nothing to win.

**Late interaction as the only stage** ({{cite:khattab2020colbert}}). Removes the
separate rerank step at 10–100× storage.

**LLM listwise reranking.** Show the model all candidates and ask for an
ordering. Better than pointwise, more expensive, and position-biased.

**Learned sparse first stage** ({{cite:formal2021splade}}). Raising the ceiling
beats improving the reranker, and this is one way to do it.

**Distillation** ({{sec:7-internal-mechanics}}). The other, and usually better,
way to raise it.

**More candidates.** Sometimes raising $k$ is simply cheaper than any of the
above, and it is always the first thing to try because it is a config change.

## 14. Evaluation

**First-stage recall@$k$**, separately and always ({{eq:recall-ceiling}}).

**The oracle-rerank ceiling** — sort the candidate set by ground-truth relevance.
The gap between oracle and actual is the reranker's remaining headroom; the gap
between oracle and perfect is the retriever's.

**End-to-end nDCG@$n$** at the $n$ users see.

**Latency at p99 under realistic $k$**, which is where {{eq:rerank-cost}} bites.

**Per-slice**, especially on the relational query types
({{sec:7-internal-mechanics}}) a reranker exists to fix. Aggregate numbers hide
exactly the queries that justify the component.

**Cost per query**, since reranking is the dominant term in a retrieval
pipeline's bill.

**Agreement with the first stage.** If the reranker's output order is highly
correlated with the input order, it is either adding little or — for listwise LLM
rerankers — echoing the position bias it was given. Measure Kendall tau between
the input and output orderings; a value near 1 means the component is not
earning its cost, and a value near 1 *only* for listwise scoring is the signature
of position bias rather than agreement.

## 15. Advanced Concepts

**The factorisation spectrum.** {{eq:factorisation-constraint}} orders every
method in this part: full factorisation (bi-encoder, one vector), partial (late
interaction, factorised encoding and joint comparison), none (cross-encoder).
**Precomputability and quality are opposite ends of one axis**, and every
architecture here is a choice of where to stand.

**Listwise beats pointwise, in principle.** Relevance is comparative — whether a
document is good depends on what else is available — and a pointwise scorer
cannot see that. Listwise LLM rerankers exploit it, at the cost of position bias
and a permutation-sized output space.

**Rerankers as judges.** {{eq:generative-scoring}} is structurally identical to
LLM-as-judge scoring ({{part:19}}): a pretrained model asked for a graded verdict
via tokens it already understands. The reranking literature's findings about
position bias, calibration, and prompt sensitivity transfer directly, and are
mostly rediscovered rather than cited.

**Why the cascade appears everywhere.** Five times now, and the reason is general:
whenever a precise scorer costs $\alpha$ times a cheap one and the cheap one has
recall $R(k)$ at $k \ll N$, a cascade beats both endpoints. **The pattern is a
consequence of that inequality, not a retrieval technique**, which is why it
recurs in model routing, ANN search, and information extraction alike.

**Reranking cannot fix a bad embedding space.** If the first stage's geometry is
degenerate ({{ch:emb-similarity}}), recall@$k$ is poor at every $k$ and
{{eq:recall-ceiling}} binds hard. Reranking is an amplifier, and an amplifier
needs signal.

## 16. Connection to Previous Chapters

{{ch:nlp-similarity}} introduced the bi-encoder/cross-encoder split; here it
follows from {{eq:factorisation-constraint}}. {{ch:emb-what-they-are}}'s
lossy-compression framing is what {{eq:bottleneck}} makes quantitative.
{{ch:emb-similarity}}'s geometry is why averaging near-orthogonal aspects fails.
{{ch:emb-models}}'s false-negative problem is solved structurally by distilling
from a cross-encoder. {{ch:emb-ann}}'s IVF-PQ-then-exact is
{{fig:rerank-cascade}} with different stages, and {{ch:llm-routing}}'s cascade
equation is {{eq:rerank-cost}} with the expensive stage moved from a fraction of
requests to a fraction of documents. {{ch:emb-hybrid}}'s conclusion — concatenate
and rerank rather than fuse — is this chapter's stage doing that job.

## 17. Exercises

1. Prove {{eq:recall-ceiling}} and state the assumption it needs about the
   reranker.
2. Derive the $1/\sqrt{2}$ in {{sec:5-formal-explanation}} for two orthogonal
   aspects, and generalise to $m$ mutually orthogonal ones.
3. Use {{eq:rerank-depth}} to compute $k$ for a 400 ms budget, 40 ms retrieval,
   and a reranker at 1.5 ms per document batched. Now unbatched at 15 ms.
4. In `rerank-cascade`, raise `BI_NOISE` to 2.0. Which column moves most, and
   what does {{eq:two-stage-decomposition}} say to do about it?
5. Lower `CROSS_NOISE` to 0.02 in the same listing. How much does end-to-end
   quality improve at $k=100$, and why is it so little?
9. Raise `CROSS_NOISE` to 0.7. The cascade at $k=1000$ now *beats* the
   cross-encoder run over the whole corpus. Explain why this is correct rather
   than a bug, and state what it implies about the first stage's role beyond
   cost reduction.
6. In `single-vector-bottleneck`, make the aspects correlated rather than
   independent. At what within-document cosine does the single vector catch up?
7. Add a chunking baseline to that listing: index each aspect as its own
   document with a single vector. Compare against MaxSim on both recall and
   storage.
8. Your reranker improved by 4 points offline and end-to-end quality did not
   move. Write the diagnosis and the next measurement.

## 18. Interview Questions

1. Bi-encoder against cross-encoder — what is the actual difference?
2. Why can a cross-encoder not be the first stage?
3. What limits how much a reranker can help?
4. How do you choose the rerank depth?
5. What is late interaction and what does it cost?
6. Why does monoT5's generative scoring transfer better than a classifier head?
7. Your reranker got better; end-to-end quality did not. Diagnose.
8. When would you skip reranking?
9. How would you use a cross-encoder to improve the retriever itself?
10. What kinds of query does reranking fix, specifically?

## 19. Research Questions

1. Is there a scorer with a cross-encoder's relational modelling and a
   bi-encoder's factorisability, or is {{eq:factorisation-constraint}} a genuine
   information-theoretic barrier?
2. Is {{cite:santhanam2022colbertv2}}'s storage reduction extensible to
   single-vector parity, or is multi-vector storage intrinsic?
3. Can rerank depth be chosen *per query* from a cheap difficulty signal, rather
   than fixed globally? This is {{ch:llm-routing}}'s question in a new place.
4. Listwise rerankers are position-biased. Is there a permutation-invariant
   listwise architecture that is also efficient?
5. Distillation from a cross-encoder raises the ceiling. Is there a principled
   stopping point — a way to know when the bi-encoder has absorbed everything the
   cross-encoder can teach?

## 20. Chapter Summary

The bi-encoder/cross-encoder distinction is one question — **when does the query
meet the document** — and everything else follows.
{{eq:factorisation-constraint}} says a corpus can be precomputed exactly when the
scorer factorises, which is why a cross-encoder is strictly better and strictly
unusable as a first stage, and why the architecture must be a cascade. That is
the fifth appearance of this pattern in the book, and
{{sec:15-advanced-concepts}} shows it is a consequence of a cost inequality
rather than a retrieval technique.

**Reranking is usually the largest available quality gain.** Measured: nDCG@10
from 0.346 to 0.712 for one hundred cross-encoder calls, against 0.944 for the
infeasible exhaustive version.

**And first-stage recall@$k$ is a hard ceiling** ({{eq:recall-ceiling}}). At
$k=100$ the retriever returned 49.7% of the true top-10, capping a perfect
reranker at 0.723 — and the real one reached 0.712, within 1.6% of its own
ceiling, against a 23% gap to exhaustive scoring. The diagnosis
{{eq:two-stage-decomposition}} prescribes is then unambiguous: the reranker is
saturated and the retriever is the bottleneck — though the balance shifts with
$k$, since the reranker's gap grows to 4.3% by $k=1000$. **Report the oracle
ceiling next to end-to-end quality**; it is computed from data you already have
and it says which half of the system to work on.

**Late interaction** ({{eq:maxsim}}) sits between the two, keeping the corpus
precomputable while letting each query term find its own evidence. It exists
because of {{eq:bottleneck}}: one point cannot be close to two near-orthogonal
directions, so a document covering unrelated aspects is squeezed between its own
meanings — measured, single-vector recall falls with aspect count while MaxSim
holds. It pays in storage, linearly, and chunking is the cheap approximation.

Finally, **distil the reranker into the retriever**. It raises the ceiling rather
than working beneath it, and it structurally solves {{ch:emb-models}}'s
false-negative problem by replacing a wrong binary label with a graded score.

## 21. Further Reading

{{cite:nogueira2019monobert}} is four pages and established the pattern.
{{cite:nogueira2020monot5}} for generative scoring and the zero-shot transfer
result behind {{eq:generative-scoring}}.
{{cite:khattab2020colbert}} for late interaction — Section 3.2 has
{{eq:maxsim}} and the pruning that makes it servable.
{{cite:santhanam2022colbertv2}} for the compression that made it deployable.
{{cite:thakur2021beir}} for reranker transfer, which is measured there too and
usually ignored.

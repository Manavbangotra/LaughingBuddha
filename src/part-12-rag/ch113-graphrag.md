---
id: rag-graph
number: 113
part: XII
tier: full
status: draft
requires: [rag-why, rag-chunking, rag-indexing, rag-query-understanding,
           rag-advanced-retrieval, emb-reranking, llm-long-context]
provides: [local-versus-global-questions, knowledge-graph-index,
           entity-relation-extraction, community-detection-summarisation,
           graph-traversal-retrieval, path-reliability-compounding,
           hub-entity-dilution, graph-build-economics]
citations: [edge2024graphrag, guo2024lightrag, gutierrez2025hipporag2,
            xiang2025whengraphs, sarthi2024raptor, trivedi2023ircot,
            yang2018hotpotqa, trivedi2022musique, gao2023ragsurvey]
---

## 1. Learning Objectives

By the end of this chapter you will be able to state precisely which questions
top-$k$ retrieval cannot answer *in principle* rather than merely badly — and
prove it rather than assert it; show that for a corpus-level question similarity
ranking is worse than **random sampling**, and explain why; derive the
reliability of an $h$-hop traversal from per-edge extraction accuracy and use it
to bound usable graph depth; identify the corpus property — entity degree, not
"multi-hop" — that decides whether a graph beats plain retrieval; and price a
knowledge-graph index in build cost, update cost, and the recurring cost that
community detection makes structurally worse than {{ch:rag-advanced-retrieval}}'s
hierarchy.

## 2. Why This Matters

Every technique so far has improved *which chunks are found*. This chapter is
about the class of questions where no set of chunks is the answer.

Ask a corpus of three thousand incident reports: **"what are the recurring causes
of outages this year?"** The answer is not in any report. It is a property of the
distribution of reports, and {{ch:rag-indexing}}'s retrieval — return the $k$
nearest chunks — cannot compute a property of a distribution from $k$ samples it
selected by similarity to a query that has no natural target.

{{cite:edge2024graphrag}} named this the **global** question and it is the real
contribution: an identification, not an architecture. The architecture that came
with it — extract an entity graph over the whole corpus, detect communities,
summarise each — answers the question and costs a whole-corpus LLM pass to build
and another on every material update.

So this chapter is a cost argument with a measurement in the middle, and it
reaches two conclusions the enthusiastic literature does not.
{{sec:9-practical-example}} finds that a plain **random sample** of the corpus
beats similarity retrieval on global questions and gets within reach of community
summaries at a larger budget, for zero build cost. And the multi-hop case turns
out not to be about hops at all: the same graph is decisively right on one corpus
and decisively wrong on another, with **entity degree** as the variable.

{{maturity:EMERGING}} Graph retrieval is well past proof-of-concept and its
economics are still moving fast; {{cite:xiang2025whengraphs}} exists because
GraphRAG frequently underperforms plain RAG on real tasks, which is not what the
2024 coverage suggested.

## 3. Prerequisites

{{ch:rag-indexing}} for top-$k$ retrieval as a *selection rule*, which is what
this chapter attacks; {{ch:rag-query-understanding}} for decomposition and
{{eq:multi-hop-containment}}, the alternative to traversal;
{{ch:rag-advanced-retrieval}} for {{eq:hierarchical-build-cost}}, the cost model
this chapter extends; {{ch:rag-chunking}}'s {{eq:chunk-dilution}};
{{ch:emb-reranking}} for what has to clean up after a traversal; and
{{ch:llm-long-context}} for why "just read the whole corpus" is not the escape it
appears to be.

## 4. Intuitive Explanation

### Two kinds of question, and only one of them is a search

A retrieval system answers a question by **selecting**. That is the right move
when the answer is *somewhere in particular*.

> **Local question.** "What was the root cause of the March 14 outage?" The
> answer is a span. Find the span, and you are done. Everything in
> {{part:11}} and in this part so far has been about finding spans.

> **Global question.** "What are the recurring causes of outages this year?" The
> answer is a *summary of three thousand spans*. No span contains it. Selecting
> ten spans does not approximate it — and the reason is worth sitting with,
> because it is not the obvious one.

The obvious objection is that ten is too few. That is true and it is not the
problem. The problem is **which** ten.

### Similarity ranking is a biased sample

Top-$k$ retrieval returns the $k$ chunks closest to the query vector. For a
global question there *is* no natural query vector — "what are the main themes"
does not point anywhere in particular — so it lands somewhere generic, and
whatever it lands near gets over-represented.

The consequence is precise: **the retrieved set is not a sample of the corpus, it
is a sample of one region of the corpus.** An estimate built from it is
systematically wrong, and the error is *bias*, not variance. Raising $k$ adds
more chunks from the same neighbourhood and does not converge on the truth.

The sharpest way to state it, and {{sec:9-practical-example}} measures exactly
this: **for a global question, a uniformly random sample of $k$ chunks is a
better answer than the top-$k$ by similarity.** Similarity ranking is not merely
insufficient here — it is actively counterproductive, because it is optimising
for the wrong property.

### What a graph adds, in one sentence each

Two distinct mechanisms travel under one name and they should be priced
separately.

**Community summarisation** answers global questions by pre-computing a
*stratified* view. Cluster the corpus, summarise each cluster, and answer from
the summaries. Each summary stands for hundreds of chunks and carries their
count, so a handful of them covers the corpus in a way a handful of chunks never
can. Notice that **the graph is incidental to this** — clustering is what does
the work, which is why {{cite:sarthi2024raptor}}'s much cheaper summarisation
tree is a real competitor rather than a lesser cousin.

**Traversal** answers multi-hop questions by walking edges instead of searching.
"Which supplier serves the depot handling the Lyon account" needs two facts that
may never appear in the same chunk; a graph that stores both edges reaches the
answer by following them. This is genuinely something retrieval cannot do —
{{ch:rag-query-understanding}}'s decomposition is the alternative, and it needs
$h$ round trips instead of one traversal.

### The bill

Both mechanisms require the same expensive thing: **an LLM pass over every chunk
to extract entities and relations**, then a second pass to summarise communities.
That is the honest headline. A graph index is not a data structure you build over
your corpus; it is a *derived corpus* you generate from it, at a price per chunk
comparable to answering a query about that chunk, and it goes stale exactly like
{{ch:rag-indexing}}'s index does — except worse, because community membership is
a global property and one new document can re-partition the graph.

## 5. Formal Explanation

### 5.1 Local and global questions, defined

Let the corpus be chunks $\mathcal{C} = \{c_1, \dots, c_N\}$ and let a question
$q$ have an answer $a(q)$. Define the question to be **local of order $m$** if
there is a set $S \subseteq \mathcal{C}$ with $|S| \le m$ and

$$ a(q) = g\big(S\big) \quad\text{for some extraction } g $$ (eq:local-question)

and **global** if no such bounded $S$ exists — the answer depends on an aggregate
over $\mathcal{C}$:

$$ a(q) = g\big(\mathcal{C}\big) = g\big(\{c_1, \dots, c_N\}\big), \qquad N \gg k $$ (eq:global-aggregate)

Retrieval with budget $k$ can answer any local question of order $m \le k$,
provided the ranker finds the right $S$. **It cannot answer a global question at
all**, and this is a statement about the class of computable outputs, not about
ranker quality. No improvement to embeddings, reranking, or chunking changes it.

### 5.2 Why the estimate is biased, not merely noisy

A global question is usually answerable *approximately* from a sample, which is
why the situation is not hopeless. Write the aggregate as an expectation over the
corpus, $a(q) = \mathbb{E}_{c \sim \mathcal{C}}[\phi(c)]$ for some per-chunk
statistic $\phi$. A sample $S$ gives the estimator $\hat{a}(S) = \frac{1}{|S|}
\sum_{c \in S} \phi(c)$, and the sampling rule decides whether it is any good:

$$ \mathbb{E}\big[\hat{a}(S_{\text{rand}})\big] = a(q), \qquad \mathbb{E}\big[\hat{a}(S_{\text{top-}k})\big] = \mathbb{E}\big[\phi(c) \mid c \in \text{top-}k(q)\big] \ne a(q) $$ (eq:selection-bias)

The random sample is **unbiased** with error $O(1/\sqrt{k})$. The similarity
sample is **conditioned on proximity to $q$**, so its error contains a bias term
that does not vanish as $k$ grows:

$$ \text{err}(S_{\text{top-}k}) \;\xrightarrow[k \to \infty]{}\; \underbrace{\big|\mathbb{E}[\phi \mid \text{near } q] - \mathbb{E}[\phi]\big|}_{\text{bias floor} \,>\, 0} $$ (eq:bias-floor)

{{eq:bias-floor}} is the chapter's first result and it predicts a specific,
checkable shape: **similarity error plateaus and random error keeps falling.**
{{sec:9-practical-example}} measures both curves.

### 5.3 Community summaries as stratification

Partition $\mathcal{C}$ into communities $\{\mathcal{K}_1, \dots,
\mathcal{K}_M\}$ and give each a summary $s_j$ reporting its content and its size
$n_j$. Answering from $t$ summaries gives

$$ \hat{a}_{\text{comm}} = \frac{\sum_{j \in T} n_j \, \phi(s_j)}{\sum_{j \in T} n_j} $$ (eq:community-stratification)

which is a **size-weighted stratified estimator**, and stratification is why $t$
summaries beat $t$ chunks by so much: one summary carries information about $n_j$
chunks rather than one.

It is not free of error, and the error has a different source:

$$ \phi(s_j) \ne \frac{1}{n_j}\sum_{c \in \mathcal{K}_j} \phi(c) \quad\text{whenever the summary drops minority content} $$ (eq:summary-lossiness)

A summary of two hundred chunks names what the community is mostly about. Content
below a few per cent of a community disappears, so the estimator has a **floor**
set by summarisation fidelity rather than by budget. {{sec:9-practical-example}}
models this explicitly, because a lossless summary would make the comparison a
fiction.

### 5.4 Traversal, and what it costs in reliability

Build the graph by extracting relations, with per-edge recall $p_e$ — the
probability that a relation stated in the corpus becomes an edge. An $h$-hop
answer needs every edge on its path:

$$ \Prob[\text{path recovered}] = \prod_{i=1}^{h} p_e = p_e^{\,h} $$ (eq:path-reliability)

This is {{ch:llm-function-calling}}'s compounding-reliability argument arriving
early, and it bounds usable depth hard. At an excellent $p_e = 0.9$: two hops
0.81, three 0.73, four 0.66. **Graph depth is bought with reliability at a
geometric rate**, and papers reporting five-hop traversal are reporting an
architecture, not a success rate.

The precision side is worse. Traversal to depth $h$ from a node of degree $b$
touches

$$ |N_h| \approx \sum_{i=1}^{h} b^i \approx b^h \quad (b > 1) $$ (eq:traversal-explosion)

entities, so the traversal returns a *neighbourhood*, not an answer. Recall is
bought with precision, geometrically, and something downstream —
{{ch:emb-reranking}}'s cross-encoder, usually — has to pay it back.

### 5.5 When traversal beats retrieval

The comparison the literature states badly. One-shot retrieval fails on a
two-hop question because the chunks mentioning the query entities are
indistinguishable from one another: if $X$ and $Z$ each appear in about $d$
chunks, the ranker sees $2d$ candidates and nothing separates the two that
matter. With budget $k$,

$$ \Prob[\text{both path chunks retrieved}] \approx \left(\frac{\min(k, 2d)}{2d}\right)^{2} \quad\text{— unity when } k \ge 2d $$ (eq:hub-dilution)

So the decision rule is not about hops:

$$ \text{graph wins} \iff p_e^{\,h} \;>\; \Prob[\text{retrieval covers the path}] \approx \min\!\left(1, \tfrac{k}{2d}\right)^{2} $$ (eq:graph-decision-rule)

**The variable is $d$, the entity degree, not $h$.** On a corpus where entities
appear in a handful of chunks, retrieval with a generous $k$ answers multi-hop
questions outright and the graph is a large bill for a regression. On a corpus of
hub entities — a company wiki where "the platform team" appears in nine hundred
documents — $k/2d$ is tiny and the graph is the only thing that works.

### 5.6 The build and update cost

$$ C_{\text{build}} = \underbrace{N \cdot c_{\text{extract}}}_{\text{every chunk}} + \underbrace{M \cdot c_{\text{summarise}}}_{\text{every community}} + \underbrace{c_{\text{cluster}}}_{\text{usually negligible}} $$ (eq:graph-build-cost)

The first term dominates and it is **linear in corpus size with an LLM call as
the constant**. Compare {{eq:hierarchical-build-cost}}: RAPTOR's summarisation
tree pays $N/(b-1)$ LLM calls, a graph pays $N$ — roughly $b-1$ times more, for
the extraction pass alone.

Update is where the structural difference lives:

$$ \frac{\partial(\text{invalidated summaries})}{\partial(\text{one edited document})} = \underbrace{O(\log N)}_{\text{RAPTOR: one path to the root}} \quad\text{versus}\quad \underbrace{O(M)}_{\text{communities: the partition can move}} $$ (eq:community-instability)

{{eq:community-instability}} is the honest reason graph indexes are hard to keep
fresh. A hierarchy has a fixed shape and an edit invalidates one path.
**Community detection has no fixed shape** — adding documents can merge or split
communities, so in principle every summary is suspect. {{cite:guo2024lightrag}}'s
contribution is precisely to make insertion incremental rather than re-deriving
the partition, and that is a bigger deal than its retrieval results.

## 6. Mathematical Foundation

### 6.1 A hand-checkable global question

Take a corpus of $N = 1000$ chunks over three themes with true proportions
$(0.6, 0.3, 0.1)$, and let $\phi$ be the theme indicator, so $a(q) = (0.6, 0.3,
0.1)$.

Suppose the generic global query lands near theme 1, so the top-$k$ set is 90%
theme 1, 10% theme 2, 0% theme 3. Then for any $k$:

$$ \hat{a}(S_{\text{top-}k}) = (0.9,\, 0.1,\, 0.0), \qquad \text{TV}\big(\hat{a}, a\big) = \tfrac{1}{2}\big(0.3 + 0.2 + 0.1\big) = 0.30 $$ (eq:tv-worked)

using total variation, $\text{TV}(p,q) = \frac12\sum_i |p_i - q_i|$. The error is
**0.30 at $k=10$ and 0.30 at $k=1000$** — {{eq:bias-floor}} made concrete.

A random sample of $k=10$ has expected proportions $(0.6, 0.3, 0.1)$ with
standard error $\sqrt{0.6 \cdot 0.4/10} \approx 0.155$ on the first component, so
its typical TV error is roughly $0.15$ — **half the bias floor, from a rule that
does no work at all.** At $k = 100$ the standard error falls to $0.049$ and the
gap widens further.

Now three community summaries, one per theme, each reporting its size: the
estimate is exact, $\text{TV} = 0$, from a budget of three. That is
{{eq:community-stratification}}'s advantage in one line — and it is *not* a
retrieval advantage, it is the advantage of having pre-computed the aggregate.

> **MATH NOTE:** The exactness is an artefact of assuming lossless summaries. With
> {{eq:summary-lossiness}} — a theme below $\tau$ of its community goes unreported
> — theme 3 vanishes from any community where it is a minority, and the estimate
> floors out above zero. Which floor you get is a property of your summarisation
> prompt, not of your corpus, and it is measurable: summarise a community whose
> composition you know and compare.

### 6.2 Depth against reliability, worked

From {{eq:path-reliability}}, the maximum depth at which a traversal is still
right with probability $\ge \rho$:

$$ h_{\max} = \left\lfloor \frac{\log \rho}{\log p_e} \right\rfloor $$ (eq:max-usable-depth)

At $p_e = 0.85$ and $\rho = 0.7$: $h_{\max} = \lfloor \log 0.7 / \log 0.85
\rfloor = \lfloor 2.19 \rfloor = 2$. **Two hops.** At $p_e = 0.95$ it is
$\lfloor 6.95 \rfloor = 6$.

The sensitivity is the point: a ten-point improvement in extraction accuracy
tripled usable depth. **Extraction accuracy is the parameter that decides what
your graph can do**, and it is almost never reported — papers report end-to-end
answer quality, which confounds it with everything else.

### 6.3 Where the crossover actually sits

Setting {{eq:graph-decision-rule}} to equality with $k \le 2d$ and $h = 2$:

$$ p_e^{2} = \left(\frac{k}{2d}\right)^{2} \iff d^{*} = \frac{k}{2 p_e} $$ (eq:degree-crossover)

At $k = 25$ and $p_e = 0.85$, $d^{*} \approx 15$. **Below degree 15, retrieval
wins; above it, the graph does.** One number, computable from two quantities you
can measure this afternoon — the median number of chunks mentioning your top
entities, and your extractor's edge recall on fifty hand-labelled chunks.

{{sec:9-practical-example}} tests this and finds the crossover roughly where
{{eq:degree-crossover}} puts it, which is the useful outcome: the model is crude
and it is not wrong.

## 7. Internal Mechanics

```mermaid {#fig:graphrag-pipeline caption="The two mechanisms that travel under one name. The build path (top) is one LLM call per chunk plus one per community — the cost that decides adoption. The query path splits: local questions traverse, global questions read summaries, and routing between them is a decision the system must make per query."}
flowchart TB
    D["corpus chunks"] -->|"LLM call PER CHUNK<br/>(the dominant cost)"| E["entities + relations"]
    E --> G[("knowledge graph")]
    G -->|"community detection<br/>(partition is unstable<br/>under insertion)"| K["communities"]
    K -->|"LLM call PER COMMUNITY"| S["community summaries<br/>with member counts"]
    Q["query"] --> R{"local or<br/>global?"}
    R -->|"local: names entities"| T["seed, then traverse h hops"]
    G --> T
    T -->|"returns a NEIGHBOURHOOD<br/>of size b to the h"| RR["rerank / filter"]
    R -->|"global: aggregate"| M["map over summaries,<br/>reduce to an answer"]
    S --> M
    RR --> A["answer"]
    M --> A
```

### 7.1 The build path, step by step

1. **Extract.** For each chunk, an LLM emits entities and typed relations. This
   is $N$ calls and it is the whole cost story.
2. **Resolve.** "Acme Corp", "Acme", and "ACME" must become one node. Entity
   resolution is the quiet failure point: under-merge and the graph fragments
   into synonym islands; over-merge and two real entities become one, which
   produces confidently wrong traversals.
3. **Detect communities.** Leiden or similar, at several resolutions, giving a
   hierarchy of coarse and fine groupings.
4. **Summarise.** One LLM call per community, at every level.

### 7.2 The two query paths

**Local search** seeds at the entities named in the query, traverses $h$ hops,
collects the text attached to the visited nodes and edges, and hands the lot to
the generator. Note what it *is*: a recall-maximising expansion whose precision
problem is deferred, exactly as {{eq:traversal-explosion}} predicts.

**Global search** is map-reduce over community summaries: score every summary at
a chosen level against the query, generate a partial answer from each, then
reduce. This touches every community — that is not a flaw, it is what makes it a
census rather than a sample, and it is why global search is expensive per query
as well as per build.

### 7.3 Which mechanism you are buying

| You have | The mechanism that helps | The cheap alternative |
|---|---|---|
| corpus-level "what are the themes" questions | community summaries | {{cite:sarthi2024raptor}}'s tree, or a large random sample |
| multi-hop questions over hub entities | traversal | decomposition ({{ch:rag-query-understanding}}) |
| multi-hop over low-degree entities | *neither* — raise $k$ | already solved |
| an actual database of entities | a real graph database | not this chapter |

**The last row deserves emphasis.** If your entities and relations already exist
as structured records, do not have an LLM re-extract them from prose. Query the
records ({{ch:rag-structured}}). Extraction is for corpora where the relations
exist only in text.

## 8. Implementation

```python {tier=A name=global-question-reachability}
"""Global questions, and the reason top-k retrieval cannot answer them.

ch:rag-indexing's retrieval returns the k chunks most similar to the query. For a
LOCAL question that is exactly the right object: the answer lives in a few spans
and similarity is how you find them. For a GLOBAL question -- "what are the main
themes across these documents" -- the answer is a property of the corpus
(eq:global-aggregate) and no k chunks contain it.

The claim worth testing is sharper than "k is too small". Similarity ranking is a
BIASED sample of the corpus (eq:selection-bias), so the estimate it supports is
wrong in a way that raising k does not fix. This listing measures that against
two alternatives at an EQUAL budget: a uniform random sample, and one summary per
community.
"""
import numpy as np

rng = np.random.default_rng(11)

N_THEME, N_CHUNK, DIM = 24, 4000, 32
N_TRIAL = 40


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


# A corpus with a skewed theme distribution -- a few large topics and a long
# tail of small ones, which is what real corpora look like.
prevalence = 1.0 / (1 + np.arange(N_THEME)) ** 0.9
prevalence /= prevalence.sum()
theme_vec = unit(rng.normal(size=(N_THEME, DIM)))

chunk_theme = rng.choice(N_THEME, size=N_CHUNK, p=prevalence)
chunk_vec = unit(0.80 * theme_vec[chunk_theme] + 0.42 * rng.normal(size=(N_CHUNK, DIM)))

true_dist = np.bincount(chunk_theme, minlength=N_THEME) / N_CHUNK


def kmeans(X, k, iters=25):
    idx = rng.choice(len(X), size=k, replace=False)
    cent = X[idx].copy()
    for _ in range(iters):
        assign = np.argmax(X @ cent.T, axis=1)
        for j in range(k):
            m = assign == j
            if m.any():
                cent[j] = X[m].mean(axis=0)
        cent = unit(cent)
    return np.argmax(X @ cent.T, axis=1), cent


# Communities: clustering stands in for the entity-graph community detection of
# cite:edge2024graphrag. What matters for this measurement is not how the
# partition is found but that a summary REPRESENTS its members and carries their
# count -- which is what makes it a stratified rather than a biased sample.
N_COMM = 28
comm, comm_vec = kmeans(chunk_vec, N_COMM)
comm_size = np.bincount(comm, minlength=N_COMM)

# What a community summary REPORTS is not what the community contains. A summary
# of two hundred chunks names the community's dominant subjects and drops the
# rest, so anything below TAU of its community disappears. Modelling the summary
# as lossless would make this comparison a fiction, and the loss is exactly where
# global search fails in practice (eq:summary-lossiness).
TAU = 0.05
comm_mix = np.zeros((N_COMM, N_THEME))
for j in range(N_COMM):
    m = comm == j
    if m.any():
        counts = np.bincount(chunk_theme[m], minlength=N_THEME).astype(float)
        counts[counts / max(comm_size[j], 1) < TAU] = 0.0
        comm_mix[j] = counts


def tv(p, q):
    """Total variation distance -- half the L1 gap between two distributions."""
    return 0.5 * np.abs(p - q).sum()


def score(est_counts):
    est = est_counts / est_counts.sum() if est_counts.sum() else est_counts
    covered = (est_counts > 0).sum() / N_THEME
    return covered, tv(est, true_dist)


def global_query():
    """A generic corpus-level question. It has to embed SOMEWHERE, and the
    friendliest realistic model is near the corpus centroid -- which is already
    dominated by the head of the theme distribution."""
    return unit(chunk_vec.mean(axis=0) + rng.normal(scale=0.35, size=DIM))


print(f"corpus: {N_CHUNK} chunks, {N_THEME} themes, {N_COMM} communities")
print(f"largest theme {true_dist.max():.1%} of the corpus, "
      f"smallest {true_dist.min():.1%}\n")
print(f"{'budget k':>9}  {'similarity top-k':>26}  {'random k':>22}  "
      f"{'communities, top-k':>26}  {'communities, largest':>26}")
print(f"{'':>9}  {'cover':>12}{'TV err':>14}  {'cover':>10}{'TV err':>12}  "
      f"{'cover':>12}{'TV err':>14}  {'cover':>12}{'TV err':>14}")
print("-" * 118)

for k in (5, 10, 20, 40, 80):
    acc = np.zeros((4, 2))
    for _ in range(N_TRIAL):
        q = global_query()

        top = np.argpartition(-(chunk_vec @ q), k)[:k]
        acc[0] += score(np.bincount(chunk_theme[top], minlength=N_THEME).astype(float))

        rnd = rng.choice(N_CHUNK, size=k, replace=False)
        acc[1] += score(np.bincount(chunk_theme[rnd], minlength=N_THEME).astype(float))

        ck = min(k, N_COMM)
        ctop = np.argpartition(-(comm_vec @ q), ck - 1)[:ck]
        acc[2] += score(comm_mix[ctop].sum(axis=0))

        cbig = np.argsort(-comm_size)[:ck]
        acc[3] += score(comm_mix[cbig].sum(axis=0))
    acc /= N_TRIAL
    print(f"{k:>9}  {acc[0,0]:>12.3f}{acc[0,1]:>14.3f}  "
          f"{acc[1,0]:>10.3f}{acc[1,1]:>12.3f}  "
          f"{acc[2,0]:>12.3f}{acc[2,1]:>14.3f}  "
          f"{acc[3,0]:>12.3f}{acc[3,1]:>14.3f}")

print("""
Read the first two column groups against each other, because that comparison is
the chapter's first result. Similarity top-k is WORSE THAN RANDOM SAMPLING at
every budget, on both metrics. It is not close and it does not close: at k=80 --
2% of the corpus -- similarity has covered 74% of themes with a distribution
error of 0.329, while a random draw of the same size covers 85% with an error of
0.182.

That is eq:selection-bias measured. The top-k set is conditioned on proximity to
the query, so it is a sample of one region rather than of the corpus, and the
error it carries is bias. Random sampling is unbiased and its error falls with k;
similarity sampling is biased and its error falls toward a floor
(eq:bias-floor) it will not cross. For a question whose answer is a property of
the whole corpus, ranking by similarity is optimising for the wrong thing.

The community columns are what pre-computed structure buys. At a budget of FIVE,
community summaries cover 51% of themes against similarity's 16%, because each
summary stands for hundreds of chunks and reports their count -- a stratified
estimator rather than a sample (eq:community-stratification).

But look where the community columns stop: 0.958 coverage and 0.155 error, and
they stay there no matter how much budget you add. That floor is
eq:summary-lossiness, not the budget. Themes that never reach 5% of any community
are not in any summary, so no amount of reading summaries recovers them. A
lossless-summary model would have printed 0.000 here and told you something
false.

Now read the whole table as a buying decision, because the honest conclusion is
uncomfortable for the technique. Community summaries win decisively at small
budgets. At k=80 a plain random sample -- zero build cost, no LLM calls, no
partition to maintain -- reaches 0.853 coverage against the summaries' 0.958 and
an error of 0.182 against 0.155. If your global questions tolerate that gap, you
have just discovered that the correct architecture is `SELECT ... ORDER BY
random() LIMIT 80`.""")
```

The second listing turns to traversal, and to the corpus property that decides
whether it is worth anything.

```python {tier=A name=graph-hop-economics}
"""Multi-hop questions: what a graph actually buys, and what it charges.

A two-hop question -- "which supplier serves the depot that handles the Lyon
account" -- names X and Z, and the answer needs the chunk linking X to Y and the
chunk linking Y to Z. ch:rag-query-understanding showed why one-shot retrieval
struggles: the second hop's search terms do not exist until the first hop has
been answered.

A graph replaces search with traversal, and traversal succeeds only if EVERY edge
on the path was extracted correctly (eq:path-reliability). This listing measures
both sides on the same corpus: vector retrieval's success as a function of entity
degree, and graph traversal's as a function of per-edge extraction accuracy --
plus the neighbourhood explosion that is traversal's precision cost.
"""
import numpy as np

rng = np.random.default_rng(23)

N_ENT = 800
N_QUERY = 400
FP_RATE = 0.10          # spurious edges, as a fraction of true edges


def build_corpus(degree):
    """Each true relation is stated in exactly one chunk, which mentions its two
    endpoints. A chunk is therefore an edge, which is what makes the comparison
    between retrieval and traversal a fair one."""
    edges = set()
    while len(edges) < N_ENT * degree // 2:
        a, b = rng.integers(0, N_ENT, size=2)
        if a != b:
            edges.add((min(a, b), max(a, b)))
    edges = np.array(sorted(edges))
    inc = [[] for _ in range(N_ENT)]          # entity -> chunk ids mentioning it
    adj = [set() for _ in range(N_ENT)]
    for i, (a, b) in enumerate(edges):
        inc[a].append(i)
        inc[b].append(i)
        adj[a].add(b)
        adj[b].add(a)
    return edges, inc, adj


def sample_path(adj, hops):
    """A path X -> ... -> Z with no shortcut edge, so the question genuinely
    needs every hop."""
    for _ in range(200):
        path = [int(rng.integers(0, N_ENT))]
        ok = True
        for _ in range(hops):
            nxt = list(adj[path[-1]] - set(path))
            if not nxt:
                ok = False
                break
            path.append(nxt[int(rng.integers(0, len(nxt)))])
        if ok and len(path) == hops + 1 and path[-1] not in adj[path[0]]:
            return path
    return None


def edge_id(edges_lookup, a, b):
    return edges_lookup.get((min(a, b), max(a, b)))


def vector_recall(inc, adj, lookup, hops, k):
    """One-shot retrieval. The query names X and Z, so every chunk mentioning
    either is a candidate and nothing distinguishes the ones on the path.
    Success means every path chunk lands in the top k (eq:hub-dilution)."""
    hit = 0
    for _ in range(N_QUERY):
        path = sample_path(adj, hops)
        if path is None:
            continue
        x, z = path[0], path[-1]
        cand = sorted(set(inc[x]) | set(inc[z]))
        # Chunks mentioning a query entity are indistinguishable to the ranker;
        # noise decides the order among them.
        order = [cand[i] for i in np.argsort(rng.random(len(cand)))][:k]
        need = {edge_id(lookup, path[i], path[i + 1]) for i in range(hops)}
        hit += int(need <= set(order))
    return hit / N_QUERY


def graph_recall(edges, lookup, adj, hops, p_e):
    """Traversal over an EXTRACTED graph: each true edge survives extraction with
    probability p_e, and spurious edges are added at FP_RATE. The path is
    recovered only if every edge on it survived."""
    keep = rng.random(len(edges)) < p_e
    ext = [set() for _ in range(N_ENT)]
    for i, (a, b) in enumerate(edges):
        if keep[i]:
            ext[a].add(b)
            ext[b].add(a)
    for _ in range(int(FP_RATE * len(edges))):
        a, b = rng.integers(0, N_ENT, size=2)
        if a != b:
            ext[int(a)].add(int(b))
            ext[int(b)].add(int(a))

    hit, frontier_total, n = 0, 0, 0
    for _ in range(N_QUERY):
        path = sample_path(adj, hops)
        if path is None:
            continue
        hit += int(all(path[i + 1] in ext[path[i]] for i in range(hops)))
        seen, frontier = {path[0]}, {path[0]}
        for _ in range(hops):
            frontier = set().union(*(ext[v] for v in frontier)) - seen if frontier else set()
            seen |= frontier
        frontier_total += len(seen)
        n += 1
    return hit / max(n, 1), frontier_total / max(n, 1)


print(f"{N_ENT} entities; every relation stated in exactly one chunk; "
      f"{FP_RATE:.0%} spurious extracted edges\n")

print("ONE-SHOT VECTOR RETRIEVAL -- success by entity degree and budget k")
print(f"{'degree':>8}{'chunks':>9}{'k=10':>9}{'k=25':>9}{'k=50':>9}{'k=100':>9}")
print("-" * 53)
vec = {}
for degree in (4, 8, 20, 50):
    edges_g, inc_g, adj_g = build_corpus(degree)
    lookup_g = {(int(a), int(b)): i for i, (a, b) in enumerate(edges_g)}
    row = [vector_recall(inc_g, adj_g, lookup_g, 2, k) for k in (10, 25, 50, 100)]
    vec[degree] = row
    print(f"{degree:>8}{len(edges_g):>9}" + "".join(f"{v:>9.3f}" for v in row))

print("\nGRAPH TRAVERSAL -- success by per-edge extraction accuracy, and the")
print("size of the neighbourhood the traversal returns (degree 8)")
edges_g, inc_g, adj_g = build_corpus(8)
lookup_g = {(int(a), int(b)): i for i, (a, b) in enumerate(edges_g)}
print(f"{'p_edge':>8}{'2-hop':>9}{'|N_2|':>9}{'3-hop':>9}{'|N_3|':>9}{'4-hop':>9}")
print("-" * 53)
gr = {}
for p_e in (0.60, 0.75, 0.85, 0.95, 1.00):
    r2, n2 = graph_recall(edges_g, lookup_g, adj_g, 2, p_e)
    r3, n3 = graph_recall(edges_g, lookup_g, adj_g, 3, p_e)
    r4, _ = graph_recall(edges_g, lookup_g, adj_g, 4, p_e)
    gr[p_e] = (r2, n2, r3, n3, r4)
    print(f"{p_e:>8.2f}{r2:>9.3f}{n2:>9.0f}{r3:>9.3f}{n3:>9.0f}{r4:>9.3f}")

print(f"""
Read the first table before the second. At degree 4 one-shot vector retrieval
answers two-hop questions well ({vec[4][1]:.3f} at k=25) -- the entities are
mentioned in few chunks, so a generous k simply retrieves all of them and the
path is inside the retrieved set. The widely repeated claim that vector retrieval
cannot do multi-hop is false at low degree, and small benchmark corpora are
exactly where degree is low.

Degree is what breaks it. At degree 50 the same k=25 scores {vec[50][1]:.3f},
because the two query entities are mentioned in about a hundred chunks and
nothing in the ranking distinguishes the two that matter. This is the honest
statement of the problem a graph solves: not "multi-hop", but HUB ENTITIES, where
the number of chunks mentioning the query terms exceeds the budget
(eq:hub-dilution).

The second table prices the alternative. Traversal is exact when extraction is
exact -- the p_edge = 1.00 row is a perfect {gr[1.00][0]:.3f} at every depth --
and extraction is never exact. At p_edge = 0.85, which is a good entity-and-
relation extractor, two-hop success is {gr[0.85][0]:.3f} and four-hop is
{gr[0.85][4]:.3f}: each additional hop multiplies by p_edge again
(eq:path-reliability), so depth is bought with reliability at a compounding rate.

Compare the two tables at the setting where the graph is supposed to win. At
degree 50 and k=25 retrieval scores {vec[50][1]:.3f} and traversal at p_edge=0.85
scores {gr[0.85][0]:.3f}, so the graph is decisively right. At degree 4 retrieval
scores {vec[4][1]:.3f} and the same graph still scores {gr[0.85][0]:.3f}, so the
graph is decisively WRONG -- it has spent a whole-corpus extraction pass to make
a solved problem worse. Nothing about the technique changed between those two
rows. The corpus did.

And the neighbourhood column is the cost nobody quotes. Traversal does not return
an answer, it returns everything within h hops: {gr[0.85][1]:.0f} entities at two
hops and {gr[0.85][3]:.0f} at three, from a graph of {N_ENT}. By the third hop
the traversal has touched almost half the corpus, so the graph has converted a
precision problem into a recall problem and handed the precision problem back to
whatever reranks the result (ch:emb-reranking).

One detail worth noticing before trusting these numbers: two-hop success at
p_edge = 0.85 is {gr[0.85][0]:.3f}, slightly ABOVE the {0.85 ** 2:.3f} that
eq:path-reliability predicts. The excess is spurious edges accidentally
reconnecting a pair that extraction dropped -- the traversal is right for the
wrong reason. It is a small effect here and a warning in general: a graph with a
false-positive rate can answer correctly by coincidence, and a system evaluated
only on answers will not notice.""")
```

## 9. Practical Example

**The global question.** The first result is the comparison nobody runs.
**Similarity top-$k$ is worse than random sampling at every budget, on both
metrics.** At $k = 80$ — two per cent of the corpus — similarity covers 73.6% of
themes with a distribution error of 0.329, while a uniform random draw of the
same size covers 85.3% with an error of 0.182.

That is {{eq:selection-bias}} measured rather than argued. The top-$k$ set is
conditioned on proximity to the query, so it samples one region; its error
contains bias, and bias does not average out. Random sampling is unbiased and
improves with $k$; similarity sampling improves toward a floor
({{eq:bias-floor}}) it will not cross. **For a question whose answer is a
property of the corpus, ranking by similarity optimises for the wrong thing** —
and the system will nonetheless return a fluent, confident answer built from the
wrong two per cent.

Community summaries show what pre-computed structure buys. At a budget of
**five**, they cover 50.9% of themes against similarity's 16.3%, because each
summary stands for hundreds of chunks and reports their count —
{{eq:community-stratification}}'s stratified estimator against a biased sample.

**And then they stop.** Coverage plateaus at 0.958 and error at 0.155,
regardless of budget. That floor is {{eq:summary-lossiness}}: themes that never
reach 5% of any community appear in no summary, so reading more summaries cannot
recover them. This is the failure mode global search actually has in production —
**the long tail is invisible, and the answer reads as complete**. A
lossless-summary model would have printed a triumphant 0.000 and told you
something false.

> **IMPORTANT:** The uncomfortable conclusion is in the last row. At $k = 80$ a
> plain random sample — zero build cost, no LLM calls, no partition to maintain —
> reaches 0.853 coverage against community summaries' 0.958, and 0.182 error
> against 0.155. **Community summarisation's advantage over random sampling is
> real, modest, and expensive**, and it is largest exactly where budgets are
> tightest. Before commissioning an extraction pass over your corpus, run the
> random-sample baseline. It takes an afternoon and it is the number your
> proposal has to beat.

**The multi-hop question.** The second listing overturns the standard framing.
At degree 4, one-shot vector retrieval answers two-hop questions **perfectly**
(1.000 at $k = 25$): the entities appear in few chunks, so a generous $k$
retrieves all of them and the path is inside the retrieved set. The widely
repeated claim that vector retrieval cannot do multi-hop is **false on
low-degree corpora — which is what small benchmarks are.**

Degree is the variable. At degree 50 the same $k = 25$ scores **0.072**, because
each query entity is mentioned in about a hundred chunks and nothing in the
ranking distinguishes the two that matter. {{eq:hub-dilution}}, measured. The
problem a graph solves is not "multi-hop", it is **hub entities**.

Traversal is exact when extraction is exact — the $p_e = 1.00$ row is 1.000 at
every depth — and extraction is never exact. At $p_e = 0.85$: two hops 0.735,
four hops 0.522. {{eq:path-reliability}} compounding, and
{{eq:max-usable-depth}}'s two-hop ceiling confirmed.

**Put the two tables side by side and the decision rule is stark.** At degree 50
retrieval scores 0.072 and traversal 0.735 — the graph is decisively right. At
degree 4 retrieval scores 1.000 and the same graph still scores 0.735 — the graph
is decisively *wrong*, having spent a whole-corpus extraction pass to make a
solved problem worse. **Nothing about the technique changed between those rows.
The corpus did.** {{eq:degree-crossover}} put the crossover near degree 15 and
the measurement brackets it between 8 and 20, which is close enough to use.

The neighbourhood column is the cost that goes unquoted: 63 entities reached at
two hops, 358 at three, from a graph of 800. **By the third hop the traversal has
touched almost half the corpus** — recall bought with precision, geometrically,
and the precision handed back to {{ch:emb-reranking}}'s reranker to pay.

One last detail, and it is a warning. Two-hop success at $p_e = 0.85$ is 0.735,
slightly **above** the 0.722 that {{eq:path-reliability}} predicts. The excess is
spurious edges accidentally reconnecting a pair that extraction dropped: the
traversal is right *for the wrong reason*. Small here, and general — a graph with
a false-positive rate answers correctly by coincidence sometimes, and a system
evaluated only on final answers will never find out.

## 10. Production Considerations

**Measure entity degree before anything else.** The median number of chunks
mentioning your most common entities is the single number that decides whether a
graph can help ({{eq:degree-crossover}}). It is a `GROUP BY` away.

**Measure extraction edge recall on fifty hand-labelled chunks.** $p_e$ sets your
usable depth through {{eq:max-usable-depth}}, and no end-to-end metric will tell
you what it is.

**Run the random-sample baseline for global questions.** Cheap, unbiased, and the
number a build proposal must beat.

**Price the rebuild, not just the build.** {{eq:graph-build-cost}} is one LLM call
per chunk; {{eq:community-instability}} says an edit can invalidate an
unbounded number of summaries. Estimate the monthly bill at your actual churn
rate before adopting.

**Budget for entity resolution and monitor it.** Track merge and split rates.
Over-merging produces confidently wrong traversals, and it is invisible in answer
quality until someone notices two customers sharing a record.

**Cap traversal depth at two** unless you have measured $p_e > 0.95$. Log the
neighbourhood size per query; when it exceeds a few hundred nodes, the graph has
stopped answering and started guessing.

**Route rather than choose.** Local and global questions want different machinery
({{sec:7-internal-mechanics}}), and misrouting a local question into global
search is expensive and vague. This is {{ch:llm-routing}}'s decision again, and
{{ch:rag-corrective}} makes it adaptive.

## 11. Common Mistakes

**Building a graph because the corpus "is relational".** The question
distribution decides, not the data.

**Believing vector retrieval cannot do multi-hop.** At low degree it does it
perfectly; the listing measures 1.000.

**Never measuring extraction accuracy** — the parameter that determines what the
graph can do.

**Traversing to depth 3+** and being surprised by irrelevant results
({{eq:traversal-explosion}}).

**Treating community summaries as lossless.** The tail is missing and the answer
reads as complete.

**Ignoring the rebuild cost** until the first large document update.

**Skipping entity resolution**, then debugging the retrieval layer for a week.

**Comparing against a weak baseline.** Against RAPTOR's tree
({{cite:sarthi2024raptor}}) or an 80-chunk random sample, the margin is much
smaller than against naive top-5.

## 12. Failure Modes

**Silent tail loss in global answers.** Symptom: summaries name the same four
themes every time and never a minor one. Detect by seeding a known rare theme and
asking whether global search surfaces it.

**Entity over-merge.** Symptom: traversals connect entities that have no real
relation, confidently. Detect by sampling merged nodes and checking them by hand.

**Entity fragmentation.** Symptom: traversal fails on entities you know are
connected. Detect by counting nodes with near-identical surface forms.

**Stale communities after ingest.** Symptom: a new product line's documents are
retrievable individually but absent from every global answer.
{{eq:community-instability}}.

**Neighbourhood flooding.** Symptom: the generator receives hundreds of loosely
related facts and produces a vague, hedged answer.
{{eq:traversal-explosion}}.

**Right answer, wrong path.** A spurious edge completes a traversal that should
have failed. Invisible to answer-level evaluation; detect by scoring the returned
*path* against known relations, not just the answer.

**Extraction schema drift.** The relation types the extractor emits change with a
model version, so new edges do not connect to old ones. Symptom: retrieval
quality falls off a cliff at a date boundary.

## 13. Alternatives

| Alternative | What it gives up | When it wins |
|---|---|---|
| Hierarchical summarisation ({{cite:sarthi2024raptor}}) | traversal; no entity structure | almost always for global questions — $N/(b-1)$ calls against $N$ |
| Random sample + long context | precision; token cost per query | global questions, small corpora, zero build budget |
| Query decomposition ({{ch:rag-query-understanding}}) | one round trip becomes $h$ | multi-hop at low degree, or when the corpus changes hourly |
| Incremental graph ({{cite:guo2024lightrag}}) | partition optimality | corpora with continuous ingest |
| Query-time graph ({{cite:gutierrez2025hipporag2}}) | offline summaries | when build cost is the binding constraint |
| A real graph database | LLM extraction entirely | the relations already exist as records ({{ch:rag-structured}}) |

**The first row is the one to take seriously.** For global questions specifically,
a summarisation tree does most of what community summaries do at a fraction of
the cost, and {{eq:community-instability}} says it also updates more predictably.
The graph earns its price when you need *traversal*, and traversal is a local-
question mechanism.

## 14. Evaluation

**Separate the two mechanisms.** Global-question quality and multi-hop traversal
quality are different systems sharing a build pipeline. One aggregate number
hides which half is working.

**Evaluate global answers on coverage and calibration**, not relevance. Does the
answer name the themes that are actually there, in roughly the right proportions?
{{sec:9-practical-example}}'s coverage-and-TV pair is the shape to use, and
seeded rare themes are how you get labels cheaply.

**Evaluate traversal on the path, not the answer.** Path precision and recall
against known relations. Otherwise {{sec:9-practical-example}}'s
right-for-the-wrong-reason case scores as a success.

**Report extraction quality as a first-class metric**: entity precision/recall,
relation precision/recall, and the resolution merge/split rate.

**Always include the cheap baselines** — random sample, RAPTOR tree, decomposition
— alongside the graph. {{cite:xiang2025whengraphs}} exists because they were
usually omitted.

**Price per query as well as per build.** Global search touches every community
summary, so its query cost scales with $M$, not with $k$.

## 15. Advanced Concepts

**Query-time graphs.** {{maturity:EMERGING}} {{cite:gutierrez2025hipporag2}} runs
Personalized PageRank over a passage-and-entity graph and makes the retrieval
decision online, which moves cost from build to query. That is the right trade
when the corpus changes faster than it is queried — the reverse of the assumption
{{cite:edge2024graphrag}} makes.

**Incremental construction.** {{maturity:EMERGING}}
{{cite:guo2024lightrag}}'s dual-level index inserts documents without re-deriving
the partition. Given {{eq:community-instability}}, this is the most important
practical advance in the area, and it is under-discussed relative to retrieval
scores.

**The unified frame.** Flat chunks, RAPTOR's tree, and a community graph are
three points on one axis: **how much structure you pre-compute, and therefore how
much you pay to keep it true.** {{ch:rag-advanced-retrieval}}'s
{{eq:hierarchical-build-cost}} and this chapter's {{eq:graph-build-cost}} are the
same equation with different constants, and the difference between them is
$b - 1$.

**Extraction accuracy as the binding constraint.** {{maturity:EMERGING}}
{{eq:max-usable-depth}} says a ten-point gain in $p_e$ triples usable depth. The
research attention goes to traversal algorithms; the leverage is in extraction,
and better extractors would do more for graph RAG than better traversal.

**Graph structure as an evaluation instrument.** {{maturity:EXPERIMENTAL}} A
knowledge graph over your corpus is a *map of what the corpus says*, independent
of retrieval. Contradictory edges reveal conflicting documents;
low-degree components reveal orphaned knowledge. That diagnostic value may
outlast the retrieval application, and it is a reason to build one even where
{{eq:degree-crossover}} says traversal will not pay.

## 16. Connection to Previous Chapters

{{ch:rag-indexing}}'s top-$k$ retrieval is what {{eq:global-aggregate}} shows to
be the wrong shape for a whole class of questions, and {{eq:selection-bias}}
explains why improving the ranker cannot fix it.
{{ch:rag-query-understanding}}'s decomposition is traversal's direct competitor,
trading round trips for a build pass. {{ch:rag-advanced-retrieval}}'s
{{eq:hierarchical-build-cost}} is the cheaper structure this chapter must justify
itself against, and {{eq:community-instability}} is why it also updates better.
{{ch:emb-reranking}} is what pays for {{eq:traversal-explosion}}.
{{ch:llm-function-calling}}'s compounding reliability reappears as
{{eq:path-reliability}}, and it will return again in {{ch:rag-agentic}}, where
the loop is the thing that compounds. {{ch:llm-routing}}'s escalation decision is
the local/global router.

## 17. Exercises

1. Derive {{eq:bias-floor}} and state the condition under which the bias term is
   zero. What would have to be true of the query embedding?
2. In `global-question-reachability`, set `TAU = 0` to make summaries lossless.
   Which columns change, and what does the change tell you about where the
   remaining error came from?
3. Raise `N_COMM` to 60 in the same listing. Does finer partitioning raise or
   lower the plateau, and why?
4. Using {{eq:degree-crossover}}, compute the crossover degree at $k=50$,
   $p_e=0.9$. Verify it against `graph-hop-economics` by adding that degree.
5. Modify `graph-hop-economics` so extraction errors are *correlated* — a chunk
   the extractor mishandles loses all its edges. Does {{eq:path-reliability}}
   still hold?
6. Add a "decomposition" strategy to the same listing: retrieve for $X$, read off
   $Y$, retrieve for $Y$ and $Z$. At which degree does it beat traversal?
7. Price {{eq:graph-build-cost}} for 2 million chunks at $500$ tokens each,
   $\$0.50$ per million input tokens, plus community summaries at $b=8$. Now add
   10% monthly churn under {{eq:community-instability}}'s worst case.
8. Design the entity-degree measurement of {{sec:10-production-considerations}}.
   What exactly do you count, and which entities do you count it over?

## 18. Interview Questions

1. What is a global question, and why can top-$k$ retrieval not answer one?
2. Why is similarity retrieval *worse than random sampling* for a global
   question?
3. What does a community summary give you that $k$ chunks do not?
4. What does GraphRAG cost to build, and what does it cost to update?
5. Derive the reliability of a three-hop traversal at 90% per-edge extraction
   accuracy.
6. Under what corpus property does a graph beat vector retrieval on multi-hop
   questions?
7. Why does traversal depth hurt precision, and what pays for it?
8. When is RAPTOR the better choice than a knowledge graph?
9. How would you measure whether your graph's edges are correct?
10. Your global answers always name the same four themes. Diagnose.

## 19. Research Questions

1. {{eq:summary-lossiness}}'s floor is set by the summarisation prompt. Can a
   summary be made *tail-preserving* — explicitly reporting minority content —
   without growing linearly in community size?
2. Community detection is unstable under insertion. Is there a partition
   objective that is provably stable to $\epsilon$ new documents?
3. {{eq:degree-crossover}} is derived under a uniform-degree model. What is the
   right crossover for a power-law degree distribution, where most entities are
   low-degree and the queries concentrate on hubs?
4. Extraction accuracy dominates via {{eq:max-usable-depth}}. How much of the
   published gap between graph and vector RAG is explained by extractor quality
   alone?
5. {{cite:xiang2025whengraphs}} finds graphs frequently underperform. Is there a
   cheap query-side classifier that predicts, per query, which mechanism will
   win?

## 20. Chapter Summary

{{cite:edge2024graphrag}}'s contribution is an **identification**: some questions
are properties of a corpus rather than of any span, and {{eq:global-aggregate}}
says retrieval cannot answer them at any $k$. The architecture that came with the
identification is one option among several for acting on it.

**Similarity ranking is not neutral on these questions — it is harmful.**
{{eq:selection-bias}} makes the top-$k$ set a sample of one region, so its error
is bias with a floor ({{eq:bias-floor}}), and the measurement is unambiguous: at
every budget tested, similarity top-$k$ covered fewer themes and estimated the
distribution worse than a **uniform random sample of the same size** — 0.736
against 0.853 coverage at $k = 80$.

**Community summaries win by stratification, not by graph structure.**
{{eq:community-stratification}} is why five summaries beat five chunks three to
one on coverage. And they plateau — 0.958 coverage, 0.155 error, forever —
because {{eq:summary-lossiness}} deletes the tail. The plateau is above a large
random sample but not by much, and the gap is the entire return on a whole-corpus
LLM pass.

**Traversal solves a problem that is about degree, not hops.** At degree 4,
retrieval answers two-hop questions perfectly and the graph scores 0.735 — the
graph makes it *worse*. At degree 50, retrieval scores 0.072 and the graph 0.735.
{{eq:degree-crossover}} puts the crossover near $d^{*} = k/2p_e$, and the
measurement agrees. **The technique did not change between those rows; the corpus
did**, which is the general lesson: a retrieval architecture is not good or bad,
it is matched or mismatched to a measurable corpus property.

**And traversal is bounded by extraction.** {{eq:path-reliability}} compounds
$p_e$ per hop, so {{eq:max-usable-depth}} gives two usable hops at $p_e = 0.85$
and six at $0.95$ — a ten-point difference that triples what the system can do,
in a parameter almost nobody measures. Meanwhile {{eq:traversal-explosion}} turns
depth into a precision bill that {{ch:emb-reranking}} has to pay.

The decision procedure fits in three measurements: **the entity degree, the
extraction accuracy, and the random-sample baseline.** Take them before writing
the extraction prompt.

## 21. Further Reading

{{cite:edge2024graphrag}} for the local/global distinction — read Section 2 for
the identification, and read the cost discussion knowing it is the part later
work disputes.
{{cite:xiang2025whengraphs}} next, and in that order: it is the corrective, and
its framing that graph RAG frequently underperforms plain RAG is the sentence to
carry into any adoption meeting.
{{cite:guo2024lightrag}} for incremental construction, which
{{eq:community-instability}} makes the central practical problem.
{{cite:gutierrez2025hipporag2}} for moving the work to query time.
{{cite:sarthi2024raptor}} again as the cheaper competitor for global questions.
{{cite:trivedi2023ircot}} for the decomposition alternative to traversal, and
{{cite:yang2018hotpotqa}} with {{cite:trivedi2022musique}} for multi-hop
evaluation — the latter specifically because it measures how much apparent
multi-hop performance is shortcut exploitation.
{{cite:gao2023ragsurvey}} for where graph retrieval sits in the standard
taxonomy.

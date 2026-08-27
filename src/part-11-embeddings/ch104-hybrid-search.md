---
id: emb-hybrid
number: 104
part: XI
tier: full
status: draft
requires: [emb-what-they-are, emb-similarity, emb-models, emb-vector-db,
           nlp-preprocessing, ml-metrics]
provides: [sparse-retrieval, inverted-index, bm25, term-saturation,
           length-normalisation, hybrid-search, reciprocal-rank-fusion,
           score-fusion, learned-sparse-retrieval, retriever-complementarity]
citations: [robertson2009bm25, cormack2009rrf, formal2021splade, thakur2021beir,
            karpukhin2020dpr, izacard2022contriever, wang2022e5, sennrich2016]
---

## 1. Learning Objectives

By the end of this chapter you will be able to derive BM25's saturation and
length-normalisation terms rather than quote them, and say what $k_1$ and $b$
control; explain why BM25 remains the baseline dense retrieval must beat and
where it still wins; implement reciprocal rank fusion and — more importantly —
state the condition under which it *hurts*; decide whether your workload
justifies hybrid search at all; and place learned sparse retrieval as the third
option that may make the question obsolete.

## 2. Why This Matters

{{part:8}} through {{ch:emb-ann}} built a dense retrieval stack. This chapter is
where the book admits that the thing it replaced is still winning a substantial
fraction of the time.

{{cite:thakur2021beir}} is the evidence: dense retrievers trained on MS MARCO
frequently *lose* to BM25 on out-of-domain collections. {{cite:izacard2022contriever}}
beat BM25 on 11 of 15 BEIR datasets without supervision, and
{{cite:wang2022e5}} claims the first zero-shot model to beat it without labels —
both framed as achievements *against BM25*, which tells you what the baseline is.

So the honest state of play, and the sentence this chapter exists to justify:
**dense retrieval wins on paraphrase and in-domain; BM25 wins on rare terms,
exact identifiers, and unfamiliar domains; neither dominates.** Hybrid search
follows from that, and so does its main failure mode — because fusing a strong
retriever with a weak one on a query the strong one already answered makes the
answer worse, and {{sec:9-practical-example}} measures exactly how much.

{{maturity:ESTABLISHED}} BM25 is forty years old and unmoved. {{maturity:EMERGING}}
Learned sparse retrieval is genuinely live: it may eliminate the need for fusion
by getting both behaviours from one index, and the evidence is suggestive rather
than settled.

## 3. Prerequisites

{{ch:nlp-preprocessing}} for tokenisation, which is where sparse retrieval's
behaviour is actually decided; {{ch:emb-what-they-are}} and
{{ch:emb-similarity}} for the dense side; {{ch:emb-models}} for what dense
retrievers are trained against; {{ch:emb-vector-db}} for indexes;
{{ch:ml-metrics}} for recall and rank metrics.

## 4. Intuitive Explanation

### What a dense embedding structurally cannot do

{{ch:emb-what-they-are}} defined an embedding as a lossy compression chosen so a
dot product approximates relevance. The word doing the work is **lossy**.

A 768-dimensional vector holds roughly 768 floats of information about a
document. A product code, a customer ID, a git SHA, a statute number, a rare
surname — each is a token that appears in a handful of documents and identifies
them exactly. There are millions of such tokens and 768 numbers. **The
compression cannot preserve them, and no amount of training changes the
arithmetic.**

This is not a training deficiency, it is a capacity bound, and it is why the
argument for keeping a lexical index is structural rather than nostalgic. An
inverted index stores an explicit posting list per term: a term appearing in
three documents costs three entries and retrieves those three exactly, forever,
at any corpus size.

Turn it around and the complement is just as clean. A lexical index cannot match
*"how do I stop my car making that noise"* to a document about brake pad wear,
because they share no terms. The dense retriever handles that trivially.

**The two methods fail on disjoint sets of queries.** That is the entire argument
for hybrid search — and, as {{sec:9-practical-example}} shows, also the reason
fusion is more delicate than it looks.

### Why BM25 has the shape it has

Three intuitions, each corresponding to a term in {{eq:bm25}}:

**Rare terms matter more.** A document containing "sennrich" tells you far more
than one containing "the". This is inverse document frequency.

**The tenth occurrence tells you less than the second.** A document mentioning
"kubernetes" ten times is about kubernetes; one mentioning it a hundred times is
not ten times more about it. Term frequency must *saturate*.

**Long documents win by accident.** A document twice as long contains twice as
many of everything, so it scores higher without being more relevant. Length must
be normalised — but only partly, because a longer document genuinely can be more
thorough.

BM25 is those three intuitions with two parameters controlling how hard the
second and third are applied. It is not a heuristic pile; it is the closed form
of a probabilistic model ({{cite:robertson2009bm25}}), and the parameters are
where the model's assumptions are exposed.

## 5. Formal Explanation

### 5.1 BM25

For query $q$ with terms $t$, document $d$, corpus of $N$ documents:

$$ \text{BM25}(q,d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t,d)\,(k_1+1)}{f(t,d) + k_1\big(1 - b + b\,\frac{|d|}{\text{avgdl}}\big)} $$ (eq:bm25)

with $f(t,d)$ the term's frequency in $d$, $|d|$ the document length,
$\text{avgdl}$ the mean, and

$$ \text{IDF}(t) = \log\left(1 + \frac{N - n_t + 0.5}{n_t + 0.5}\right) $$ (eq:bm25-idf)

for $n_t$ the number of documents containing $t$. The $+0.5$ terms are the
smoothing that keeps a term appearing in every document from producing a
negative or infinite weight.

### 5.2 Saturation, derived

Set $b = 0$ to isolate the frequency term:

$$ \text{sat}(f) = \frac{f(k_1+1)}{f + k_1} $$ (eq:bm25-saturation)

Three properties make this the right function, and each is worth checking:

$$ \text{sat}(0) = 0, \qquad \text{sat}(1) = 1, \qquad \lim_{f \to \infty} \text{sat}(f) = k_1 + 1 $$ (eq:saturation-limits)

So it is zero when absent, exactly 1 for a single occurrence — the calibration
point — and **bounded above by $k_1 + 1$ however many times the term appears**.
That bound is the whole point: no term can dominate a multi-term query through
repetition alone, which is what makes BM25 robust to keyword stuffing without
any special-case handling.

$k_1$ controls how fast it saturates. As $k_1 \to 0$, {{eq:bm25-saturation}}
approaches a step function — presence/absence, with frequency ignored. As
$k_1 \to \infty$ it approaches raw $f$, unbounded. The conventional
$k_1 \in [1.2, 2.0]$ sits closer to the step function than to raw counts, which
is the empirical finding that occurrence matters far more than count.

### 5.3 Length normalisation, derived

The denominator's $\big(1 - b + b\,|d|/\text{avgdl}\big)$ interpolates between two
positions:

$$ b = 0: \text{ no normalisation}, \qquad b = 1: \text{ full normalisation} $$ (eq:length-norm-endpoints)

The reason neither endpoint is right is the useful part. With $b=0$, long
documents win by containing more of everything. With $b=1$, length is divided out
completely — which assumes a document is long *only* because it is verbose, and
penalises a genuinely comprehensive document exactly as hard as a padded one.

The truth is in between, hence $b \approx 0.75$: **most of the length advantage
is spurious, but not all of it.** That single sentence is what the parameter
encodes, and it is why $b$ is the parameter worth tuning per corpus — a corpus of
uniform-length product descriptions and one of mixed abstracts and full papers
want different values.

### 5.4 Reciprocal rank fusion

Two retrievers, two ranked lists, incomparable scores — a BM25 score has no
bounded range and a cosine lives in $[-1,1]$, and {{ch:emb-similarity}} showed
neither is calibrated. {{cite:cormack2009rrf}}'s answer is to discard the scores
entirely and fuse the **ranks**:

$$ \text{RRF}(d) = \sum_{r \in \mathcal{R}} \frac{1}{k + \text{rank}_r(d)} $$ (eq:rrf)

with $k = 60$ from the original paper. Being rank-based, it needs no calibration,
no score normalisation, and no per-corpus tuning — which is why it became the
default in every vector database that offers hybrid search.

> **RESEARCH NOTE:** $k = 60$ is one 2009 experiment's value on TREC data, and it
> is almost never re-tuned. It is not arbitrary in effect: $k$ sets how sharply
> rank position is discounted, so small $k$ makes the fusion behave like
> "whichever retriever ranked it first wins" and large $k$ flattens toward equal
> weight over the whole list. If your two retrievers differ greatly in quality,
> $k$ is the knob that should be moved and is not.

### 5.5 When fusion helps, and when it hurts

The property {{eq:rrf}} has that is rarely stated: **a document ranked
moderately by both retrievers can outrank a document ranked first by one.**
Concretely, with $k=60$, a document at rank 1 in one list and absent from the
other scores $1/61 = 0.0164$; a document at rank 40 in both scores
$2/101 = 0.0198$ and wins.

That is a *feature* when the two retrievers are partially redundant, because
agreement is evidence. It is a **defect** when they are complementary, because
then "ranked first by one and absent from the other" is exactly what a correct
answer looks like.

$$ \text{fusion helps} \iff \text{retrievers agree often enough that agreement carries information} $$ (eq:fusion-condition)

{{sec:9-practical-example}} measures both regimes and finds the effect large in
each direction. **This is the chapter's most useful result and it contradicts how
hybrid search is usually sold.**

## 6. Mathematical Foundation

### 6.1 Where BM25 comes from

{{eq:bm25}} is not assembled from intuitions; it is the tractable form of the
probabilistic relevance framework. Rank by the odds of relevance:

$$ \text{score}(d) = \log \frac{\Prob(R = 1 \given d, q)}{\Prob(R = 0 \given d, q)} $$ (eq:prf-odds)

Assume term independence given relevance and this decomposes into a sum of
per-term log-odds — which is {{eq:bm25-idf}} for binary presence. The remaining
problem is that presence is a poor summary of a term's evidence, and modelling
$f(t,d)$ properly requires a distribution over counts. The **eliteness** model
posits a hidden binary variable — is the document *about* this term — with $f$
drawn from a mixture, and integrating it out yields precisely
{{eq:bm25-saturation}}'s functional form.

The value of knowing this is that it tells you where BM25 will fail: **the term
independence assumption.** BM25 cannot know that "new" and "york" together mean
something neither means apart. Every extension — BM25F, phrase queries, n-gram
indexing — attacks that one assumption.

### 6.2 What an inverted index costs

$$ \text{posting entries} = \sum_{t} n_t = \sum_{d} |\{\text{unique terms in } d\}| \approx N \cdot \bar{u} $$ (eq:inverted-index-size)

Linear in corpus size times unique terms per document, with each entry a
document id and a frequency — typically 4–8 bytes compressed. For $N = 10^7$ with
$\bar{u} = 200$: about 2 billion entries, 8–16 GB.

Compare {{ch:emb-vector-db}}'s dense index at the same $N$: 30.7 GB for
768-dimensional float32 vectors. **The lexical index is roughly half the size and
needs no GPU to build.** Query cost is proportional to the posting lists of the
query's terms, so a rare term is nearly free — the opposite of dense retrieval,
where every query costs the same regardless of how selective it is.

### 6.3 Complementarity, measured

Whether hybrid search is justified is an empirical property of your workload, and
it has a number. For query set $Q$ and top-$k$ results $A_k(q)$, $B_k(q)$:

$$ \text{overlap} = \frac{1}{|Q|}\sum_{q} \frac{|A_k(q) \cap B_k(q)|}{k} $$ (eq:retriever-overlap)

Near 1, the retrievers are redundant and fusion adds cost without information.
Near 0, they are complementary — and by {{eq:fusion-condition}}, RRF is the
*wrong* combiner, since it will bury each retriever's confident answers.

$$ \text{overlap} \approx 0.3\!-\!0.6 \;\Rightarrow\; \text{fuse}; \qquad \text{overlap} \approx 0 \;\Rightarrow\; \text{route or concatenate} $$ (eq:fusion-decision)

**Measure this before building hybrid search.** It is one pass over a few hundred
queries and it decides the architecture.

## 7. Internal Mechanics

```mermaid {#fig:hybrid-architecture caption="Hybrid retrieval. The two paths share nothing but the query text and the fusion step, which is what makes them independently tunable — and what makes the fusion step the only place their incomparable scores can be reconciled."}
flowchart LR
    Q["query text"] --> T["tokenise<br/>(ch:nlp-preprocessing)"]
    Q --> E["encode<br/>(ch:emb-models)"]
    T --> I[("inverted index")]
    E --> V[("vector index")]
    I -->|"top-100 by BM25"| F["fuse: RRF (eq:rrf)<br/>or interleave, or route"]
    V -->|"top-100 by cosine"| F
    F -->|"top-k"| R["rerank<br/>(ch:emb-reranking)"]
```

### 7.1 Tokenisation is where sparse retrieval is decided

An inverted index's behaviour is a function of its analyser, not its scorer, and
this is where most sparse-retrieval failures live.

- **Stemming** merges "running" and "run" — helpful for recall, harmful when
  "SAP" is stemmed into oblivion.
- **Stopword removal** was essential when indexes were small and is mostly
  harmful now: it destroys "to be or not to be" and BM25's IDF already suppresses
  common terms automatically.
- **Case folding** helps except for acronyms.
- **Sub-tokenisation of identifiers.** Splitting `ERR_TIMEOUT_5031` into three
  tokens changes it from an exact identifier into three common ones, and destroys
  exactly the capability the lexical index was kept for.

> **PRODUCTION TIP:** If you keep a lexical index for identifiers, verify the
> analyser preserves them. Index a document containing a known product code and
> search for it. This ten-second test catches the most common way a hybrid
> system's lexical half becomes useless.

### 7.2 Combiners other than RRF

{{eq:rrf}} is the default, not the only option, and
{{eq:fusion-condition}} implies the others are sometimes better:

| Method | Needs | Best when |
|---|---|---|
| RRF | nothing | overlap moderate; the safe default |
| weighted score fusion | per-corpus score normalisation | one retriever is reliably better |
| interleaving | nothing | retrievers complementary; preserves each one's confident answers |
| routing by query class | a classifier | query types are distinguishable and disjoint |
| concatenate then rerank | a reranker | you have one anyway — often the best answer |

**Interleaving deserves more use than it gets.** Taking alternate results from
each list preserves each retriever's top result exactly, which is precisely what
RRF destroys under complementarity. It is one line of code and has no parameters.

**And "concatenate then rerank" is frequently the right answer**, because a
cross-encoder ({{ch:emb-reranking}}) scores the union directly and needs no
fusion heuristic at all. If the pipeline already has a reranker, fusion is
solving a problem that stage will solve better.

### 7.3 Learned sparse retrieval

{{cite:formal2021splade}} produces a sparse vector over the vocabulary from a
transformer, with explicit sparsity regularisation and term expansion — so a
document about brake pads acquires nonzero weight on "noise" without containing
the word.

It is served by an inverted index, so it keeps exact-match behaviour and the
mature infrastructure, while learning the semantic expansion that dense
retrieval provides. **If it works, hybrid search becomes unnecessary**, because
one index does both jobs.

The catch is cost. SPLADE's expansion makes documents much denser than natural
language — hundreds of active terms rather than the natural distribution's tail —
so posting lists grow and query latency with them. The trade-off is real and the
question is open.

## 8. Implementation

```python {tier=A name=lexical-vs-dense}
"""Where each retriever wins, and what fusion does about it.

A synthetic corpus with three retrievable signals:
  * topic vocabulary  -- semantic content a dense encoder can represent
  * an identifier     -- one token unique to each document, which a fixed-width
                         embedding structurally cannot preserve
  * a category token  -- shared by about twenty documents, so it NARROWS but does
                         not identify

Three query types exercise them separately, and the point of the experiment is
that the right combiner depends on which type you actually get.
"""
from collections import Counter
import numpy as np

rng = np.random.default_rng(5)

N_DOC, N_TOPIC, VOCAB, DOC_LEN = 4000, 40, 3000, 60
K1, B, RRF_K, DEPTH, K = 1.2, 0.75, 60, 100, 10
N_QUERY = 150

# Topic-conditional word distributions: sparse and overlapping.
topic_w = np.zeros((N_TOPIC, VOCAB))
for t in range(N_TOPIC):
    idx = rng.choice(VOCAB, 120, replace=False)
    topic_w[t, idx] = rng.random(120) ** 2
topic_w /= topic_w.sum(axis=1, keepdims=True)

mixture = rng.dirichlet(np.ones(N_TOPIC) * 0.3, size=N_DOC)
identifier = VOCAB + np.arange(N_DOC)              # unique per document
N_CAT = N_DOC // 20
category = VOCAB + N_DOC + rng.integers(0, N_CAT, size=N_DOC)

docs = []
for i in range(N_DOC):
    p = mixture[i] @ topic_w
    toks = rng.choice(VOCAB, DOC_LEN, p=p).tolist()
    toks += [int(identifier[i]), int(category[i])]
    docs.append(toks)

# ---- BM25 (eq:bm25) --------------------------------------------------------
tf = [Counter(d) for d in docs]
doc_len = np.array([len(d) for d in docs], dtype=float)
avgdl = doc_len.mean()
df = Counter()
for c in tf:
    for w in c:
        df[w] += 1
idf = {w: np.log(1 + (N_DOC - n + 0.5) / (n + 0.5)) for w, n in df.items()}


def bm25(q_tokens):
    scores = np.zeros(N_DOC)
    for w in q_tokens:
        if w not in idf:
            continue
        weight = idf[w]
        for i, counts in enumerate(tf):
            f = counts.get(w, 0)
            if f:
                scores[i] += weight * f * (K1 + 1) / (
                    f + K1 * (1 - B + B * doc_len[i] / avgdl))
    return scores


# ---- Dense: document = its topic mixture; query = topic posterior of its ----
# tokens. Tokens outside the topic vocabulary contribute NOTHING, which is the
# capacity bound of section 4 made literal.
E = mixture + rng.normal(scale=0.005, size=mixture.shape)
E /= np.linalg.norm(E, axis=1, keepdims=True)


def dense(q_tokens):
    q = np.full(N_TOPIC, 1e-6)
    for w in q_tokens:
        if w < VOCAB:
            q += topic_w[:, w]
    return E @ (q / np.linalg.norm(q))


def rrf(*ranked):
    """Fuse the top-DEPTH of each list (eq:rrf). A document outside a
    retriever's top-DEPTH receives nothing from it -- as deployed."""
    s = np.zeros(N_DOC)
    for scores in ranked:
        for rank, d in enumerate(np.argsort(-scores)[:DEPTH]):
            s[d] += 1.0 / (RRF_K + rank + 1)
    return s


def interleave(*ranked):
    """Take alternate results from each list, preserving each one's top hit."""
    lists = [np.argsort(-s)[:DEPTH] for s in ranked]
    out, seen = [], set()
    for pos in range(DEPTH):
        for lst in lists:
            d = int(lst[pos])
            if d not in seen:
                seen.add(d)
                out.append(d)
    return out


def make_query(i, kind):
    p = mixture[i] @ topic_w
    if kind == "semantic":                       # different words, same topic
        return rng.choice(VOCAB, 20, p=p).tolist()
    if kind == "identifier":                     # contains the exact token
        return rng.choice(VOCAB, 3, p=p).tolist() + [int(identifier[i])]
    return rng.choice(VOCAB, 10, p=p).tolist() + [int(category[i])]  # partial


def evaluate(kind):
    hits = {"bm25": [], "dense": [], "rrf": [], "interleave": []}
    overlaps = []
    for _ in range(N_QUERY):
        i = int(rng.integers(0, N_DOC))
        q = make_query(i, kind)
        sb, sd = bm25(q), dense(q)

        top_b = set(np.argsort(-sb)[:K].tolist())
        top_d = set(np.argsort(-sd)[:K].tolist())
        overlaps.append(len(top_b & top_d) / K)

        hits["bm25"].append(float(i in top_b))
        hits["dense"].append(float(i in top_d))
        hits["rrf"].append(float(i in set(np.argsort(-rrf(sb, sd))[:K].tolist())))
        hits["interleave"].append(float(i in set(interleave(sb, sd)[:K])))
    out = {k: float(np.mean(v)) for k, v in hits.items()}
    # An oracle router: send each query type to whichever retriever is better on
    # it. This is the ceiling a perfect query classifier would reach.
    out["oracle"] = max(out["bm25"], out["dense"])
    return out, float(np.mean(overlaps))


results, overlaps = {}, {}
print(f"{'query type':<13}{'BM25':>8}{'dense':>8}{'RRF':>8}{'interleave':>12}"
      f"{'oracle route':>14}{'overlap':>10}")
print("-" * 73)
for kind in ["semantic", "identifier", "partial"]:
    results[kind], overlaps[kind] = evaluate(kind)
    r = results[kind]
    print(f"{kind:<13}{r['bm25']:>8.3f}{r['dense']:>8.3f}{r['rrf']:>8.3f}"
          f"{r['interleave']:>12.3f}{r['oracle']:>14.3f}{overlaps[kind]:>10.3f}")

mixed = {k: float(np.mean([results[c][k] for c in results])) for k in results["semantic"]}
print(f"{'MIXED':<13}{mixed['bm25']:>8.3f}{mixed['dense']:>8.3f}{mixed['rrf']:>8.3f}"
      f"{mixed['interleave']:>12.3f}{mixed['oracle']:>14.3f}"
      f"{np.mean(list(overlaps.values())):>10.3f}")

print(f"""
Read the three query-type rows first. They are the whole argument for keeping a
lexical index: on IDENTIFIER queries BM25 scores {results['identifier']['bm25']:.3f}
and the dense retriever {results['identifier']['dense']:.3f}, and no amount of
training fixes that -- a unique token cannot survive compression into a
fixed-width vector. On SEMANTIC queries the positions reverse just as sharply.

Now read the RRF column against the best single retriever in each row, because
this is where the usual story about hybrid search breaks. On the two rows where
one retriever DOMINATES, fusion is much worse than simply using it. That is
eq:fusion-condition: with k=60, a document at rank 40 in both lists scores
2/101 = 0.0198 and outranks a document at rank 1 in one list and absent from the
other at 1/61 = 0.0164. When the retrievers are complementary, "first in one and
absent from the other" is exactly what a CORRECT answer looks like -- and RRF
buries it.

On the PARTIAL row, where both retrievers have real but incomplete signal, RRF
beats both. That is the case fusion was designed for and it works.

The MIXED row is the honest justification for hybrid search, and note what it
actually says: RRF beats either single retriever ACROSS THE WORKLOAD while losing
to the better one on most individual query types. Fusion is insurance against
query heterogeneity, not a way to make any given query better.

Now the OVERLAP column, which predicted all of this in advance. It sits near
{np.mean(list(overlaps.values())):.2f} -- the two retrievers almost never return
the same documents -- and eq:fusion-decision says that at that value RRF is the
wrong combiner. Compare INTERLEAVE, which is one line of code with no parameters
and simply takes alternate results, preserving each retriever's top hit by
construction. It beats RRF on three of the four rows, including the mixed
workload, and it is the better default whenever overlap is low.

ORACLE ROUTE is the ceiling: send each query type to whichever retriever is
better on it. It beats every combiner, and the gap between it and interleaving is
what a query classifier would actually be worth. That is the same argument as
ch:llm-routing -- when a cheap signal predicts which system should answer,
routing beats blending -- arriving here in a new setting.""")
```

```python {tier=A name=bm25-parameters}
"""What k1 and b actually do, and why neither endpoint is right.

Saturation (eq:bm25-saturation) and length normalisation are usually quoted.
Here they are plotted as functions and then tested on a corpus with a planted
pathology: one document that repeats a query term many times without being about
it, and one that is genuinely relevant but long.
"""
import numpy as np

K1_VALUES = [0.0001, 0.5, 1.2, 2.0, 8.0]
B_VALUES = [0.0, 0.25, 0.75, 1.0]


def saturation(f, k1):
    return f * (k1 + 1) / (f + k1)


print("Term-frequency saturation: sat(f) = f(k1+1)/(f+k1)   (eq:bm25-saturation)")
print(f"{'k1':>10}" + "".join(f"{'f=' + str(f):>9}" for f in [1, 2, 5, 20, 100])
      + f"{'limit':>9}")
print("-" * 64)
for k1 in K1_VALUES:
    row = "".join(f"{saturation(f, k1):>9.3f}" for f in [1, 2, 5, 20, 100])
    print(f"{k1:>10.4f}{row}{k1 + 1:>9.3f}")

print("""
Every row equals exactly 1.000 at f=1 -- that is the calibration point of
eq:saturation-limits -- and every row is bounded by k1+1 no matter how large f
gets. The bound is what makes BM25 immune to keyword stuffing with no special
handling: repeating a term a hundred times cannot buy more than k1+1.

Read the extremes. At k1 -> 0 the function is a step: present or absent, count
ignored. At k1 = 8 it is still climbing at f=100. The conventional 1.2 sits far
closer to the step than to raw counts, which IS the empirical finding -- whether
a term occurs matters enormously and how often matters remarkably little.
""")

# ---- The pathology, on a small corpus ------------------------------------
CORPUS = {
    "rel-short":  {"kubernetes": 3, "scheduler": 2, "pod": 2},
    "rel-long":   {"kubernetes": 6, "scheduler": 4, "pod": 5, "filler": 85},
    "stuffed":    {"kubernetes": 40, "buy": 30, "cheap": 30},
    "off-topic":  {"database": 4, "index": 3},
}
QUERY = ["kubernetes", "scheduler"]
N_D = len(CORPUS)
lengths = {d: sum(c.values()) for d, c in CORPUS.items()}
avgdl = float(np.mean(list(lengths.values())))
idf = {}
for term in QUERY:
    n_t = sum(1 for c in CORPUS.values() if term in c)
    idf[term] = np.log(1 + (N_D - n_t + 0.5) / (n_t + 0.5))


def score(doc, k1, b):
    c = CORPUS[doc]
    total = 0.0
    for term in QUERY:
        f = c.get(term, 0)
        if f:
            total += idf[term] * f * (k1 + 1) / (
                f + k1 * (1 - b + b * lengths[doc] / avgdl))
    return total


print(f"query: {QUERY}   lengths: "
      + ", ".join(f"{d}={lengths[d]}" for d in CORPUS))
print(f"\n{'setting':<32}" + "".join(f"{d:>11}" for d in CORPUS) + "   winner")
print("-" * 80)
table = {}
for label, k1, b in [("k1->0    (presence only)", 0.0001, 0.75),
                     ("k1=1.2,  b=0    (no len norm)", 1.2, 0.0),
                     ("k1=1.2,  b=0.75 (standard)", 1.2, 0.75),
                     ("k1=1.2,  b=1    (full len norm)", 1.2, 1.0),
                     ("k1=8,    b=0.75 (weak satur.)", 8.0, 0.75)]:
    s = {d: score(d, k1, b) for d in CORPUS}
    table[label] = s
    print(f"{label:<32}" + "".join(f"{s[d]:>11.3f}" for d in CORPUS)
          + f"   {max(s, key=s.get)}")

std = table["k1=1.2,  b=0.75 (standard)"]
weak = table["k1=8,    b=0.75 (weak satur.)"]
no_norm = table["k1=1.2,  b=0    (no len norm)"]
full_norm = table["k1=1.2,  b=1    (full len norm)"]

print(f"""
Follow the STUFFED column. That document contains "kubernetes" forty times and is
about nothing. Under standard saturation it scores {std['stuffed']:.3f}, well
below both genuinely relevant documents. Weaken the saturation to k1=8 and it
jumps to {weak['stuffed']:.3f} -- a factor of
{weak['stuffed'] / std['stuffed']:.1f} -- close enough to the relevant long
document's {weak['rel-long']:.3f} to start displacing real results. The cap at
k1+1 in eq:saturation-limits is doing the anti-spam work, and it needs no rule
about repetition to do it.

Now the two length settings, on the two documents that are BOTH genuinely
relevant. With b=0 the long document wins ({no_norm['rel-long']:.3f} against
{no_norm['rel-short']:.3f}) purely by containing more of everything -- the
spurious advantage of section 5.3. Turn normalisation on and the ordering flips:
at the standard b=0.75 the short document leads {std['rel-short']:.3f} to
{std['rel-long']:.3f}, and at b=1 the gap widens further to
{full_norm['rel-short']:.3f} against {full_norm['rel-long']:.3f}.

That widening is the argument against b=1. Full normalisation divides length out
entirely, which assumes a document is long ONLY because it is padded -- so it
penalises a thorough document exactly as hard as a stuffed one. The conventional
0.75 is the claim that most of a long document's advantage is spurious but not
all of it, and b is the parameter genuinely worth tuning per corpus: uniform
product descriptions and mixed abstracts want different answers.""")
```

## 9. Practical Example

The first listing's table is the chapter, and it says three things.

**The capacity bound is real.** On identifier queries BM25 retrieves the target
essentially always and the dense retriever essentially never. This is not a
training problem — a token unique to one document cannot survive compression into
a fixed-width vector, and {{sec:4-intuitive-explanation}}'s arithmetic says so.
On semantic queries the positions reverse just as sharply. **The two methods fail
on disjoint query sets**, which is the argument for keeping both.

**RRF hurts badly when one retriever dominates.** On semantic queries it scores
0.307 against the dense retriever's 0.820; on identifier queries 0.693 against
BM25's 1.000. The mechanism is {{eq:rrf}} itself: with $k=60$ a document at rank
40 in both lists scores $2/101 = 0.0198$ and beats one at rank 1 in a single list
at $1/61 = 0.0164$, so fusion systematically buries exactly the confident answers
that complementarity produces. This is the opposite of how hybrid search is
usually described, and it follows directly from the formula.

RRF does win where it was designed to: on partial-signal queries, where both
retrievers have real but incomplete information, it scores 0.527 against 0.253
and 0.400.

**Interleaving beats it almost everywhere, and this was the surprise.** With no
parameters and one line of code, interleaving scores 0.707 and 0.967 on the two
single-signal types against RRF's 0.307 and 0.693, and 0.691 against 0.509 on
the mixed workload. It preserves each retriever's top hit by construction, which
is precisely what {{eq:rrf}} sacrifices.

**And the overlap statistic predicted this in advance.** {{eq:retriever-overlap}}
measures about 0.04 across every query type — the two retrievers almost never
return the same documents — and {{eq:fusion-decision}} says that at that value
RRF is the wrong combiner. It is, by a large margin. The diagnostic costs one
pass over a few hundred queries and it would have chosen correctly here without
running the comparison at all.

**Oracle routing is the ceiling.** Sending each query type to whichever retriever
is better on it reaches 0.740 on the mixed workload against RRF's 0.509 — and
interleaving, at 0.691, recovers most of that gap with no classifier at all. The
oracle gap quantifies what query classification is worth, and it is
{{ch:llm-routing}}'s argument arriving in a new setting: when a cheap signal
predicts which system should answer, routing beats blending.

> **IMPORTANT:** Measure {{eq:retriever-overlap}} on your own queries before
> building hybrid search. One pass over a few hundred queries decides between
> fusion, interleaving, routing, and not bothering — and the number that
> justifies each is in {{eq:fusion-decision}}.

The second listing confirms the parameters do what {{sec:5-formal-explanation}}
derives. Every $k_1$ gives exactly 1.000 at $f=1$ and is bounded by $k_1+1$, and
the consequence is visible in the keyword-stuffed document: at the standard
$k_1 = 1.2$ it scores 0.748, and weakening saturation to $k_1 = 8$ multiplies
that by 3.2 to 2.413 — close enough to the genuinely relevant long document's
2.452 to start displacing real results. **The cap is doing the anti-spam work,
with no rule about repetition anywhere in the system.**

Length normalisation flips the ordering of the two relevant documents exactly as
predicted: at $b = 0$ the long one leads 1.827 to 1.514 on bulk alone; at
$b = 0.75$ the short one leads 1.950 to 1.609; at $b = 1$ the gap widens to
2.160 against 1.548. That widening is the argument against the $b=1$ endpoint —
it penalises a thorough document exactly as hard as a padded one.

## 10. Production Considerations

**Test that the analyser preserves identifiers.** {{sec:7-internal-mechanics}}'s
ten-second check. If it does not, the lexical index is not doing the job you kept
it for.

**Measure retriever overlap before choosing a combiner**
({{eq:retriever-overlap}}).

**Tune $b$ per corpus, leave $k_1$ alone.** $b$ encodes an assumption about your
length distribution that varies enormously; $k_1$'s conventional range is
insensitive.

**Prefer interleaving to RRF when overlap is low.** {{sec:9-practical-example}}
finds it beating RRF by a wide margin at an overlap near zero, with no parameters
to tune. RRF's advantage appears only where both retrievers have partial signal.

**Consider routing rather than fusing** if query types are distinguishable. The
oracle gap in {{sec:9-practical-example}} is what is on the table.

**If you already have a reranker, concatenate rather than fuse.** A cross-encoder
scores the union directly and needs no combiner ({{ch:emb-reranking}}).

**Budget for two indexes.** {{eq:inverted-index-size}} against
{{ch:emb-vector-db}}'s dense equation: the lexical index is roughly half the size
of the dense one, so hybrid is about 1.5× the memory and two write paths — which
also means two consistency problems and two failure modes.

**Keep the lexical index even if dense wins today.** It is the fallback when the
embedding model is being migrated ({{ch:emb-models}}), and during a re-embed it
may be the only working retriever you have.

## 11. Common Mistakes

**Assuming hybrid always beats both.** {{sec:9-practical-example}} shows RRF
losing badly on single-signal queries.

**Reaching for RRF because it is the default.** It is the right combiner in one
regime — moderate overlap — and measurably the wrong one outside it. Interleaving
is simpler, has no parameters, and won on three of four rows in the listing.

**Fusing raw scores without normalisation.** BM25 scores are unbounded and
cosines live in $[-1,1]$; adding them weights BM25 arbitrarily. Fuse ranks, or
normalise first and know what the normalisation assumes.

**Using $k = 60$ without thought.** It is one 2009 experiment's value.

**Removing stopwords.** IDF already handles them and removal destroys phrase-like
queries.

**Sub-tokenising identifiers in the lexical index.** Removes its whole advantage.

**Reporting hybrid gains on a benchmark whose query mix differs from yours.** The
mixed row is a weighted average, and the weights are your traffic.

**Treating BM25 as a legacy baseline.** {{cite:thakur2021beir}} says it is the
one to beat out of domain.

## 12. Failure Modes

**Analyser drift.** A library upgrade changes stemming or tokenisation; the
lexical half silently degrades. Nothing errors.

**Fusion burying confident answers.** {{eq:fusion-condition}}, and the users who
notice are the ones who searched for something exact.

**Score-scale drift.** With weighted score fusion, a change to either retriever
shifts its score distribution and silently re-weights the blend. This is the
argument for rank-based fusion.

**Vocabulary mismatch after re-embedding.** The dense half is migrated, the
lexical half is not, and their relative quality changes without the fusion
weights being revisited.

**Index divergence.** Two indexes, two write paths, and a document present in one
but not the other. Detectable only by an explicit reconciliation job.

**IDF drift on a growing corpus.** {{eq:bm25-idf}} depends on $N$ and $n_t$, so a
term's weight changes as the corpus grows. Usually benign, occasionally not — a
term that was rare and became common silently loses its discriminating power.

## 13. Alternatives

**Learned sparse** ({{cite:formal2021splade}}). One index, both behaviours, at
higher query cost. The most likely thing to make this chapter obsolete.

**Dense only.** Correct when the corpus has no identifiers and the domain matches
the model's training. Rarer than teams assume.

**Lexical only.** Correct more often than the literature suggests, especially for
code, logs, and catalogues — and it needs no GPU, no re-embed, and no
migration.

**Query expansion.** Rewrite the query with an LLM to add synonyms, then run
BM25. Captures much of dense retrieval's benefit with no vector index, at the
cost of a generation call per query.

**Multi-field BM25 (BM25F).** Weight title, body, and tags differently. Often a
larger win than adding a dense retriever and much cheaper.

## 14. Evaluation

**Per query type, never only in aggregate.** The mixed row hides everything that
matters, and the aggregate is a weighted average whose weights are your traffic.

**Retriever overlap** ({{eq:retriever-overlap}}), which decides the architecture.

**Against both single-retriever baselines.** A hybrid system that does not beat
both on your workload is 1.5× the cost for nothing.

**On out-of-domain queries specifically**, which is where BM25's advantage lives
({{cite:thakur2021beir}}).

**Recall@k at the reranker's $k$** ({{ch:emb-reranking}}).

**Identifier queries as a named slice.** They are a small fraction of traffic and
a large fraction of user frustration, and they are the reason the lexical index
exists.

## 15. Advanced Concepts

**BM25's independence assumption is its ceiling.** {{eq:prf-odds}} decomposes
into a sum only under term independence given relevance, so BM25 structurally
cannot represent that "new york" means something neither word does. Every
extension attacks this.

**Why RRF's rank-basis is both its strength and its weakness.** Discarding scores
makes it calibration-free — the property that made it universal — and also
discards *confidence*. A retriever that is certain and a retriever that is
guessing contribute identically at the same rank. Score fusion keeps the
confidence and pays with a calibration problem, which is
{{ch:emb-similarity}}'s unsolved one.

**Learned sparse as the synthesis.** {{cite:formal2021splade}} is what you get by
asking a transformer to output the sparse representation directly: semantic
expansion with exact-match preservation, in one index. Whether the posting-list
cost is fundamental or an artefact of current sparsity regularisers is the open
question.

**Tokenisation connects the two halves.** {{cite:sennrich2016}}'s subword
vocabulary is what the dense encoder sees, and the lexical analyser is a
different tokenisation of the same text. **Hybrid systems have two tokenisers and
usually nobody owns their consistency** — which is why an identifier can be
preserved in one path and destroyed in the other.

**Fusion is a special case of ensembling** with the usual condition: ensembles
help when members are accurate and diverse, and {{eq:fusion-condition}} is that
condition specialised to rankings. The literature on ensemble diversity transfers
directly and is largely unread by retrieval practitioners.

## 16. Connection to Previous Chapters

{{ch:emb-what-they-are}}'s "lossy compression" is the exact reason a dense vector
cannot hold an identifier — the argument that justifies this whole chapter.
{{ch:emb-similarity}}'s incomparable-scores result is why {{eq:rrf}} fuses ranks.
{{ch:emb-models}}'s BEIR discussion is the evidence that BM25 remains the
baseline. {{ch:emb-vector-db}}'s memory arithmetic is what
{{eq:inverted-index-size}} is compared against. {{ch:nlp-preprocessing}}'s
tokenisation decisions turn out to determine the lexical half's behaviour
entirely. And {{ch:llm-routing}}'s central claim — that routing beats blending
when a cheap signal predicts the right system — is quantified here by the oracle
column.

## 17. Exercises

1. Verify {{eq:saturation-limits}} algebraically for all three limits.
2. Show that as $k_1 \to 0$, {{eq:bm25-saturation}} approaches the indicator
   $\Ind[f > 0]$, and interpret.
3. Derive the condition under which a document at rank $r$ in both lists
   outranks a document at rank 1 in one list under {{eq:rrf}}. Solve for $r$ at
   $k = 60$ and at $k = 5$.
4. In `lexical-vs-dense`, sweep `RRF_K` over $\{5, 20, 60, 200\}$. Which query
   type is most sensitive, and does any value make RRF beat the oracle?
5. Add a weighted score-fusion combiner with min-max normalised scores. Compare
   it against RRF on all three query types, and explain the difference using
   {{sec:15-advanced-concepts}}'s confidence argument.
6. Compute {{eq:retriever-overlap}} for each query type in the listing. Does
   {{eq:fusion-decision}} predict the right combiner in each case?
7. In `bm25-parameters`, add a document that is relevant, short, and mentions
   each query term once. What $b$ is needed for it to win?
8. Design the identifier-query evaluation slice for a system you know. How would
   you collect it, and how would you know it was representative?

## 18. Interview Questions

1. Why does BM25 still matter?
2. What do $k_1$ and $b$ control? Which would you tune?
3. Why does term frequency saturate?
4. Why does RRF fuse ranks instead of scores?
5. When does hybrid search make results *worse*?
6. Your users complain that searching for an exact product code fails. Diagnose.
7. What is learned sparse retrieval and what problem does it solve?
8. How would you decide whether to build hybrid search?
9. Fusion or routing?
10. What does an inverted index cost compared to a vector index?

## 19. Research Questions

1. Is there a fusion method that keeps RRF's calibration-freedom while
   preserving each retriever's confident answers, without a tuned parameter?
2. Can retriever complementarity ({{eq:retriever-overlap}}) be predicted from
   corpus and query statistics, rather than measured?
3. Does {{cite:formal2021splade}}'s posting-list cost have a lower bound, or is
   it an artefact of current sparsity regularisers?
4. BM25's parameters are fitted per corpus by grid search. Is there a way to set
   them from measurable corpus statistics — length distribution, vocabulary
   growth — directly?
5. Hybrid systems maintain two tokenisers. Is there a shared tokenisation that
   serves both a subword encoder and a lexical index without compromising
   either?

## 20. Chapter Summary

**A dense embedding structurally cannot store an identifier.** A fixed-width
vector holds a fixed amount of information and there are millions of rare tokens;
this is a capacity bound, not a training deficiency, and it is the whole argument
for keeping a lexical index. Measured: on identifier queries BM25 retrieves the
target essentially always and a dense retriever essentially never, with the
positions sharply reversed on paraphrase queries.

**BM25 is derived, not assembled.** {{eq:bm25}} is the tractable form of
{{eq:prf-odds}} under term independence. Saturation is bounded by $k_1+1$ and
calibrated to 1.000 at $f=1$ ({{eq:saturation-limits}}), which is why keyword
stuffing does not work; $b \approx 0.75$ encodes that most of a long document's
advantage is spurious but not all of it. Its ceiling is the independence
assumption.

**Reciprocal rank fusion hurts when retrievers are complementary.** This is the
chapter's most useful result and it contradicts the usual pitch. Under
{{eq:rrf}} with $k=60$, a document ranked 40th by both beats one ranked first by
one and missed by the other — and under complementarity, that second pattern is
what a correct answer looks like. Measured, fusion loses badly on both
single-signal query types and wins on the type where both retrievers have partial
signal.

**Measure overlap before choosing a combiner.** {{eq:retriever-overlap}} came out
near 0.04 in the listing — the retrievers almost never agree — and
{{eq:fusion-decision}} predicts from that alone that RRF is wrong here. It is:
parameter-free interleaving, which preserves each retriever's top hit, scored
0.691 on the mixed workload against RRF's 0.509, and oracle routing 0.740. The
diagnostic is one pass over a few hundred queries and it chooses the architecture.

The honest case for hybrid search is therefore narrower than usually stated: it
is a hedge against query heterogeneity, it should usually be interleaving or
routing rather than rank fusion, and it costs something on every query where you
would have known which retriever to ask.

Learned sparse retrieval ({{cite:formal2021splade}}) may make the question
obsolete by getting both behaviours from one index, at a posting-list cost that
is currently real and possibly not fundamental.

## 21. Further Reading

{{cite:robertson2009bm25}} is the definitive treatment — Sections 3 and 4 derive
{{eq:bm25}} from {{eq:prf-odds}}, including the eliteness model behind
saturation. It is the one reference in this chapter to read in full.
{{cite:cormack2009rrf}} is two pages and contains the whole method, including the
$k=60$ that nobody has re-tuned since.
{{cite:formal2021splade}} for learned sparse.
{{cite:thakur2021beir}} for the evidence that BM25 remains the out-of-domain
baseline.

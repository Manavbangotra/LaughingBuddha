# Part XI — Embeddings and Vector Search: research notes

Research pass run 2026-08-27, before writing. Full tier: 21 sections per
chapter, 4,200-word floor, seven chapters. Twenty-two new bibliography entries,
each verified against an arXiv abstract page, a proceedings listing, or a
journal record on the date above. 166 entries total, none unverified.

## What this part is, and what it is not

{{part:8}} built static and contextual word embeddings and stopped at the
sentence level. {{part:10}} finished the single-model story. **This part is the
first part of the book about infrastructure rather than models** — and the
hazard changes accordingly.

Parts IX and X were haunted by unrefereed sources and folklore. This part has
the opposite problem: **the literature is excellent and forty years deep, and
the practice ignores most of it.** BM25 was derived from a probabilistic model
in the 1970s–90s and is still the baseline that dense retrieval loses to on
out-of-domain queries; the curse of dimensionality was proved precisely in 1999
and is routinely invoked in ways the theorem does not support; HNSW's parameters
are tuned by folklore in systems whose authors have not read the paper.

The rule for this part: **when a widely repeated claim has a theorem behind it,
state the theorem's actual hypotheses.** The two places this matters most are
{{cite:beyer1999nn}} (distance concentration does *not* say vector search is
impossible — it says something narrower, and learned embeddings violate its
assumptions) and {{cite:cormack2009rrf}} (RRF's $k=60$ is one experiment's
number and almost nobody re-tunes it).

## The organising idea

**Representation and search are one design, and treating them as two is the most
expensive mistake in this part.**

Every choice on the representation side has a cost on the search side, and the
book's chapter split hides this unless it is said explicitly:

| Representation choice | What it does to search |
|---|---|
| dimension $d$ | index memory and distance cost are both linear in $d$ |
| normalisation | decides whether cosine and inner product are the same problem |
| single- vs multi-vector | multiplies the index size by tokens-per-document |
| dense vs learned sparse | changes the *index data structure*, not a parameter |
| asymmetric query/doc prefixes | makes the index model-version-locked |

And symmetrically, every search choice constrains representation: PQ assumes
subspaces are roughly independent, HNSW assumes a metric space, an inverted
index assumes sparsity.

The through-line to state in {{ch:emb-what-they-are}} and return to in
{{ch:emb-reranking}}: **an embedding is a lossy compression of meaning chosen so
that one specific operation — a dot product — approximates one specific relation
— relevance. Everything in this part is a consequence of that sentence.**
Reranking exists because the compression is lossy. ANN exists because even the
compressed comparison is too slow at scale. Hybrid search exists because the
compression discards exact lexical identity.

## What changes at this tier for this material

The material is *more* mathematical than Part X and much more measurable. Both
change what the 21 sections should contain:

- **§6 Mathematical Foundation** is genuinely load-bearing here for the first
  time since {{part:7}}: the cosine/inner-product/Euclidean equivalences under
  normalisation, the concentration-of-distances result, the PQ error
  decomposition, the BM25 saturation derivation, and the recall/QPS frontier as
  a Pareto object.
- **§9 Practical Example** should be a *measurement*, not a demonstration.
  Almost every claim in this part is checkable on a laptop with numpy, and the
  listings should check them rather than assert them. The book has an unusual
  opportunity here: HNSW, IVF, PQ, BM25, and RRF can all be implemented from
  scratch in under 80 lines each, and doing so is far more instructive than
  calling a library.
- **§12 Failure Modes** is unusually rich and unusually under-documented in the
  literature, because the failures are operational: index/model version skew,
  recall silently degrading as the corpus grows, normalisation mismatches,
  metric mismatches, and deletion tombstones degrading graph connectivity.
- **§14 Evaluation** must distinguish *retrieval* quality from *index* quality.
  These are measured differently, fail differently, and are constantly
  conflated: recall@k against the true nearest neighbours is an index metric;
  nDCG against human judgements is a retrieval metric. An index at 100% recall
  of a bad embedding is a bad retriever.

## The genuinely live questions

State these as open, with the evidence, rather than resolving them.

### 1. Does distance concentration actually matter?

{{cite:beyer1999nn}} proves that under i.i.d.-ish assumptions the ratio of
farthest to nearest distance tends to 1 as $d \to \infty$, which would make
nearest-neighbour queries meaningless. Vector search with $d = 768$ demonstrably
works.

The resolution is that the theorem's hypotheses fail: learned embeddings are not
i.i.d. across dimensions, they occupy a low-dimensional manifold, and the
relevant quantity is *intrinsic* dimension, not ambient. **But this is worth
demonstrating rather than asserting** — a listing that measures the
nearest/farthest ratio on i.i.d. Gaussian vectors versus on vectors drawn from a
low-rank manifold makes the point in a way no paragraph can.

What remains genuinely unresolved: nobody has a usable a-priori estimate of when
a given embedding space is "concentrated enough" to hurt, and intrinsic
dimension estimators disagree substantially.

### 2. Has dense retrieval actually beaten BM25?

{{cite:thakur2021beir}} is the honest answer and it is "not uniformly". Dense
retrievers trained on MS MARCO frequently *lose* to BM25 on BEIR's
out-of-domain sets. {{cite:izacard2022contriever}} beat BM25 on 11 of 15 BEIR
datasets at Recall@100 without supervision, and {{cite:wang2022e5}} claims the
first zero-shot model to beat BM25 without labelled data.

So the state of play, stated carefully: **dense retrieval wins in-domain and on
paraphrase; BM25 wins on rare terms, exact identifiers, and unfamiliar domains;
neither dominates.** That is the entire justification for
{{ch:emb-hybrid}} existing, and it should be stated as a measured result
with the citation, not as a truism.

Live: whether learned sparse ({{cite:formal2021splade}}) makes hybrid
unnecessary by getting both behaviours from one index. The evidence is
suggestive and not settled, and SPLADE's index is much more expensive than BM25's.

### 3. Is the single-vector bottleneck fundamental?

A document compressed to one 768-dimensional vector cannot represent a document
that is relevant to two unrelated queries for unrelated reasons; the vector must
sit somewhere between them and be good for neither.
{{cite:khattab2020colbert}} is the direct attack, and it works — at 10–100× the
storage.

Live question: whether the multi-vector storage cost is intrinsic or an
artefact. {{cite:santhanam2022colbertv2}} cut it by an order of magnitude with
residual compression, which is evidence for "artefact"; nobody has cut it to
single-vector cost, which is evidence for "intrinsic".

**Do not claim chunking solves this.** Chunking a document into passages is the
poor-man's multi-vector approach and it is what most systems actually do; say so
plainly, and note that it moves the problem to chunk-boundary selection rather
than removing it.

### 4. What does an embedding model's benchmark score predict?

Very little about a specific application. MTEB ({{cite:muennighoff2023mteb}})
aggregates across task types that need different geometries — clustering wants
uniformity, retrieval wants alignment, classification wants linear separability
— and an average over them is not a quantity anyone's system cares about.

Worse, MTEB is now a target. Models are trained on data adjacent to its test
sets, so the top of the leaderboard is compressed to within noise and the
ordering carries little signal.

**The honest recommendation to give: benchmark scores narrow the candidate set
to about five models; the choice among those five requires a domain evaluation
set of a few hundred labelled pairs, and building that set is the highest-return
work in the whole retrieval pipeline.** This should be stated with the reasoning,
not as advice-shaped filler.

### 5. Is the vector database a product category or a feature?

Genuinely contested, and worth stating as such rather than picking a side.

The case for a feature: pgvector, and the vector indexes now in Elasticsearch,
Mongo, Redis, and SQLite, all put ANN next to data that already exists, which
removes an entire class of consistency problem. The case for a product:
dedicated systems are substantially faster at scale, and quantization, filtered
search, and index rebuild are hard enough to be worth specialising in.

The engineering content that survives whichever way this resolves: **the hard
part is not the index, it is the metadata filter.** Pre-filtering breaks the
graph's connectivity assumptions; post-filtering blows up latency when the
filter is selective. Every vector database solves this differently and none
solves it well, and this is a much more useful thing for a reader to know than a
product comparison.

### 6. Do embeddings need to be re-computed when the model changes?

Yes, entirely, and this is the operational fact that most surprises teams.
Vectors from two model versions are not comparable — not "slightly degraded",
but meaningless together, because the spaces have no relation.

This makes the embedding model a *versioned schema* for the index, and the
migration a full re-embed of the corpus. There is no incremental path. Say this
early ({{ch:emb-models}}) and design the failure modes around it.

## Per-chapter findings

### 99 — What Embeddings Are: Representation Learning Revisited

Anchor to {{ch:dl-autoencoders}} and {{ch:nlp-contextual}} rather than
re-teaching. New content: the distinction between an embedding *learned as a
by-product* (word2vec's, BERT's hidden states) and one *trained to be an
embedding* (a contrastive dual encoder) — this is the difference that explains
why mean-pooled BERT is a poor retriever and SimCSE is not.

Core equations: InfoNCE ({{cite:oord2018cpc}}), and the alignment/uniformity
decomposition from {{cite:gao2021simcse}}. The anisotropy result is the one to
demonstrate numerically: raw BERT embeddings have mean pairwise cosine far above
zero, which destroys the dynamic range of the similarity score.

Listing: measure anisotropy on random-but-correlated vectors and show what
whitening does to the retrieval ranking. Do NOT need transformers for this.

### 100 — Similarity Measures and the Geometry of Embedding Space

The chapter that must be got exactly right, because everything downstream
inherits it. Content:

- cosine, inner product, and Euclidean are the *same ranking* on normalised
  vectors and different rankings otherwise — with the algebra shown
- therefore: "cosine vs dot product" is a question about whether your model
  normalises, not a preference
- the concentration result ({{cite:beyer1999nn}}) with its hypotheses stated
- why inner product breaks HNSW's assumptions (it is not a metric; no triangle
  inequality) and what systems do about it
- the score is not a probability and is not comparable across models; a cosine
  of 0.82 means nothing absolute

Listing: the nearest/farthest ratio experiment (i.i.d. Gaussian vs low intrinsic
dimension), and a demonstration that un-normalised inner product ranks by
magnitude.

### 101 — Embedding Models: Training, Choosing, and Evaluating

Training: dual encoder, InfoNCE, and the fact that **in-batch negatives make
batch size the dominant hyperparameter** — an unusual property worth dwelling
on. Hard negative mining ({{cite:karpukhin2020dpr}}) is where the quality
actually comes from, and mined-then-filtered negatives are subtle: a negative
that is actually relevant is a labelling error that teaches the wrong thing.

Choosing: dimension is not the quality knob ({{cite:ni2021gtr}} — scale the
model, hold the dimension); Matryoshka ({{cite:kusupati2022matryoshka}}) makes
dimension a serving-time decision; asymmetric prefixes ({{cite:wang2022e5}}) are
a compatibility contract, and getting them wrong degrades silently.

Evaluating: MTEB's limits as above; the domain evaluation set as the real answer.

Listing: train a small dual encoder with InfoNCE on synthetic paired data and
measure the effect of batch size and of hard vs random negatives. Both effects
are large and reproducible at toy scale.

### 102 — Vector Databases and Index Structures

Resist the product survey. The chapter is about **what a vector database is for
beyond the index**: filtered search, CRUD against an immutable-ish index,
persistence, replication, and multi-tenancy.

The genuinely hard problem to develop: pre- vs post-filtering. Post-filter with
a 1% selective filter and you must retrieve 100× to fill $k$; pre-filter and the
graph is disconnected. Show the arithmetic.

Deletion is the second: HNSW cannot truly delete, so tombstones accumulate and
recall degrades until a rebuild. Systems differ in whether they tell you.

Listing: simulate filtered search over a toy index and measure the
over-retrieval factor required as a function of filter selectivity. This is a
pure-numpy experiment and the curve is dramatic.

### 103 — Approximate Nearest Neighbors: HNSW, IVF, and Product Quantization

The most technical chapter in the part, and the one where from-scratch
implementations pay off most.

- NSW ({{cite:malkov2014nsw}}) → HNSW ({{cite:malkov2020hnsw}}): the insight
  that insertion order creates long-range links, and the layer hierarchy
- IVF: partition, probe $n$ cells, and the recall/cost knob is $n_{probe}$
- PQ ({{cite:jegou2011pq}}): the product codebook, and asymmetric distance
  computation as the reason it costs less accuracy than expected
- ScaNN ({{cite:guo2020scann}}): reconstruction error is the wrong objective for
  MIPS; penalise the parallel component. This is the deepest idea in the chapter
  and generalises far beyond ANN
- LSH ({{cite:indyk1998lsh}}) as the historical framing, honestly labelled as
  displaced empirically
- the recall/QPS frontier ({{cite:aumuller2020annbench}}) as the only correct
  way to report an ANN result

Listings: implement HNSW's greedy graph search and measure recall vs ef;
implement PQ and measure the compression/recall trade-off. Both are ~60 lines.

### 104 — Sparse Retrieval, BM25, and Hybrid Search

BM25 derived, not quoted ({{cite:robertson2009bm25}}): where saturation comes
from, what $k_1$ and $b$ actually control, and why length normalisation is not
obvious.

Then the case for hybrid, stated as the BEIR result rather than as folklore, and
RRF ({{cite:cormack2009rrf}}) as the fusion method — with the key point that it
fuses *ranks*, which is why it needs no score calibration between incomparable
scales. Note that $k=60$ is one paper's number.

SPLADE ({{cite:formal2021splade}}) as the third option that may make the
question obsolete.

Listing: implement BM25 from scratch on a small corpus, show a query where it
beats a dense retriever and one where it loses, and fuse with RRF to show the
fusion beating both. If RRF does *not* beat both, report that — it does not
always, and the condition under which it fails (one retriever much worse than
the other) is instructive.

### 105 — Reranking and Cross-Encoders

The third appearance of the cheap-then-expensive cascade, after
{{ch:nlp-similarity}} and {{ch:llm-routing}}. Name it as the same pattern
explicitly — the book's cross-part payoff depends on the reader seeing this.

Content: bi-encoder vs cross-encoder as a question of *when the query meets the
document* ({{cite:nogueira2019monobert}}); why the cross encoder cannot be
pre-computed and therefore cannot be the first stage; monoT5's generative
scoring ({{cite:nogueira2020monot5}}) and why it transfers zero-shot better;
late interaction ({{cite:khattab2020colbert}}) as the middle point.

The cost arithmetic is the practical core and mirrors {{eq:cascade-cost}}:
reranking $k$ candidates costs $k$ cross-encoder forward passes, so $k$ is a
latency budget, and the recall of the first stage at $k$ bounds everything.
**A reranker cannot recover a document the retriever did not return** — the
single most important operational fact in the chapter.

Listing: build a two-stage pipeline with a cheap scorer and an expensive one,
sweep $k$, and plot nDCG against total cost. Show the recall ceiling explicitly.

## Cross-part bookkeeping

- **Do not** teach RAG here. Chunking, context assembly, and generation are
  {{part:12}}. This part stops at "a ranked list of documents".
- {{ch:emb-reranking}} should forward-reference {{part:12}} once and stop.
- Multimodal embeddings (CLIP) belong to {{part:13}}; mention that the geometry
  chapter's results are modality-independent and leave it.
- Serving cost language should match {{part:10}}'s: TTFT/ITL do not apply, but
  the arithmetic-intensity framing does, and the reranker cascade should
  explicitly reuse {{eq:cascade-cost}}'s shape.
- Terminology collision check before writing: `embedding`, `similarity`,
  `retrieval`, `index`, `recall`, `sparse`, `dense`, `quantization` are all
  likely to already exist in the glossary from Parts IV, VI, VIII, and X.
  `quantization` in particular is defined in the model-compression sense in
  {{part:14}}'s neighbourhood — the vector-quantization sense needs a distinct
  term id.

---
id: part-11-intro
status: final
---

## What this part is for

{{part:10}} finished the single-model story. This part is **the first part of the
book about infrastructure rather than models**, and the change in subject brings
a change in hazard.

Parts IX and X were haunted by unrefereed sources and by folklore with a paper
somewhere behind it. Here the opposite is true: **the literature is excellent and
forty years deep, and practice ignores most of it.** BM25 was derived from a
probabilistic model before most of this book's readers were born and is still the
baseline dense retrieval loses to out of domain. The curse of dimensionality was
proved precisely in 1999 and is routinely cited for conclusions the theorem does
not support. HNSW's parameters are tuned by copied blog posts in systems whose
authors have not read the paper.

> **The rule adopted for this part: when a widely repeated claim has a theorem
> behind it, state the theorem's actual hypotheses.** That is where the
> engineering lives. {{cite:beyer1999nn}}'s concentration result does not say
> high-dimensional search is hopeless — it says something narrower, and learned
> embeddings violate its assumptions. {{cite:cormack2009rrf}}'s $k = 60$ is one
> 2009 experiment's number that almost nobody has re-tuned.

## The organising idea

**Representation and search are one design, and treating them as two is the most
expensive mistake in this part.**

The chapter split hides this unless it is said outright. Every choice on the
representation side has a cost on the search side:

| Representation choice | What it does to search |
|---|---|
| dimension $k$ | index memory and distance cost are both linear in it |
| normalisation | decides whether cosine and inner product are the same problem |
| single- versus multi-vector | multiplies index size by vectors per document |
| dense versus learned sparse | changes the index *data structure*, not a parameter |
| asymmetric query/document prefixes | makes the index model-version-locked |

And symmetrically: product quantization assumes subspaces are near-independent,
graph indexes assume a metric, an inverted index assumes sparsity. A retrieval
system is one design with two halves, and the seam between them is where the
failures live.

```text
   REPRESENTATION                SEARCH                    GIVING UP
   ─────────────────────────     ────────────────────      ──────────────────
   99  what an embedding IS      102 the database          105 the cascade —
   100 the geometry it lives         around the index          when the query
       in                        103 approximate search        finally meets
   101 how to train and              — graphs, IVF, PQ         the document
       choose one                104 what compression
                                     cannot preserve
```

The through-line, stated in {{ch:emb-what-they-are}} and returned to in
{{ch:emb-reranking}}: **an embedding is a lossy compression of meaning chosen so
that one cheap operation — a dot product — approximates one expensive relation —
relevance.** Everything in this part is a consequence of that sentence. Reranking
exists because the compression is lossy. Approximate search exists because even
the compressed comparison is too slow. Hybrid search exists because the
compression discards exact lexical identity, and no amount of training recovers
it.

## Four things worth knowing before you start

**The score is a rank, not a measurement.** {{eq:ranking-constraint}} constrains
only *differences*, so a cosine of 0.82 means nothing absolute, means something
different under another model, and drifts as a corpus grows. Only the margin over
the corpus's background ({{eq:score-margin}}) carries information. Every
hard-coded similarity threshold in production is relying on a property nobody
trained for.

**Concentration tracks intrinsic dimension, not stored dimension.**
{{eq:beyer-condition}}'s hypothesis is about the distance distribution's relative
variance, which is governed by the data's *effective* dimension. Learned
embeddings are low-rank, so they sit outside the theorem's regime — demonstrated
in {{ch:emb-similarity}} by holding intrinsic dimension at 8 while sweeping the
stored width to 512 and watching relative contrast stay flat while the i.i.d.
control collapses. **This is why 768-dimensional retrieval works at all.**

**An embedding model is a versioned schema for its index.** Vectors from two
model versions are not degraded together, they are meaningless together, because
the two spaces have no relation. There is no incremental migration and no partial
upgrade — only a full re-embed. That single fact makes model choice
({{ch:emb-models}}) more consequential than any benchmark difference between
candidates.

**The cascade is everywhere and it is not a retrieval technique.** Cheap
imprecise stage, then expensive precise stage: {{ch:emb-ann}}'s
IVF-then-PQ-then-exact, {{ch:emb-reranking}}'s retrieve-then-rerank, and before
them {{ch:llm-routing}}'s model cascade and {{ch:nlp-extraction}}'s
encoder-then-LLM. {{ch:emb-reranking}} shows why: it follows from a cost
inequality, so it appears wherever that inequality holds.

## What this part does not cover

Chunking, context assembly, and generation are {{part:12}} — this part stops at
*a ranked list of documents*. Multimodal embeddings are {{part:13}}, though
{{ch:emb-similarity}}'s geometry is modality-independent and transfers unchanged.

## How the chapters build

Chapters 99 and 100 are load-bearing and should be read in order; 100 in
particular is the metric decision that every later chapter inherits and that an
index compiles in. Chapter 101 is the practitioner core — if you read one
chapter, read that one. Chapters 102 through 104 are what breaks in production,
and 102's filtering result is the one most likely to surprise you. Chapter 105 is
the largest quality win available and the chapter that explains where to spend
effort.

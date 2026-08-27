---
id: part-11-assessment
status: final
---

## How to use this

Four sections. The knowledge check takes about two hours and tells you what to
re-read. The assignment builds a retrieval system end to end and — as in
{{part:10}} — the deliverable is the measurement table rather than the code,
because every important decision in this part is settled by a number you either
measured or did not. The challenge is open-ended. The interview section is what
to rehearse.

No answers are provided. Every question is answerable from the chapters.

## Knowledge check

**What an embedding is**

1. State the definition {{ch:emb-what-they-are}} builds everything from, and
   derive from it why embeddings are normalised and why their dimension is a few
   hundred rather than a few million.
2. Explain the difference between a representation learned as a *by-product* and
   one trained *to be* an embedding. Why is mean-pooled BERT a poor retriever
   when BERT is a good language model?
3. Derive {{eq:infonce-gradient}} and use it to explain why random negatives stop
   teaching. Give the suppression factor at $\tau = 0.07$ and margin 0.3.
4. State {{eq:infonce-mi-bound}}. What does it establish about batch size that is
   different from batch size's role anywhere else in deep learning?
5. {{ch:emb-what-they-are}} measures an encoder with the *best* alignment of its
   group and nearly the worst retrieval. Explain, and say what follows about
   using {{eq:alignment}} on its own.
6. In the same experiment, centring by one global mean improved every geometry
   diagnostic and made retrieval worse. What was the mechanism, and what is the
   correct fix?

**Geometry**

7. Prove {{eq:rank-equivalence}} from {{eq:l2-ip-identity}}, stating exactly
   where the constant-norm assumption enters.
8. Inner product and negative-L2 selected the same top document 0.00% of the time
   on un-normalised vectors. Explain using {{eq:magnitude-bias}} why this is not
   noise.
9. Rank correlation between two scorers was 0.815 while top-1 agreement was 4.3%.
   What does that tell you about evaluating a retrieval change with Spearman or
   Kendall tau?
10. State {{eq:beyer-condition}} with its hypotheses. Then explain why
    768-dimensional vector search works, and why "reduce dimensions to beat the
    curse of dimensionality" does not follow.
11. Define $k$-occurrence ({{eq:k-occurrence}}) and explain hubness. In the
    measured i.i.d. control at 512 dimensions, what fraction of the corpus was
    unreachable, and why is that number invisible to every recall metric?
12. Why is inner product not a metric, and what does that break? Give the exact
    reduction in {{eq:mips-augmentation}} and the condition under which it
    degrades.

**Training and choosing**

13. {{eq:capacity-allocation}}: why do hard negatives help fine-grained
    retrieval far more than coarse retrieval? What does that imply about an
    evaluation set drawn uniformly from the corpus?
14. Mined negatives showed run-to-run variance 4.7× lower than random ones, and
    the random variance exceeded the effect being measured. State the
    consequence for how an embedding fine-tune must be evaluated.
15. Explain why {{ch:emb-models}} could not reproduce false-negative poisoning
    at realistic mining rates, and why that is consistent with the hazard being
    real. In which round does it bite?
16. Separate dimension from capacity using {{eq:embedding-costs}}. Which is a
    serving cost and which an ingest cost, and which carries quality?
17. Explain {{eq:matryoshka-loss}} and why the *widest* prefix is slightly worse
    than a model trained at that width alone.
18. Use {{eq:paired-eval-size}} to say how many labelled pairs a domain
    evaluation set needs, and why the paired figure is so much smaller than the
    unpaired one.

**Indexes and search**

19. Derive {{eq:postfilter-budget}} and compute the budget needed at 0.2%
    selectivity for $k=10$. At what point has the index bought nothing?
20. State {{eq:percolation-threshold}} and explain why pre-filtering shatters a
    graph index. Predict $s_c$ for $M = 16$ and say what the largest connected
    component means operationally.
21. Both filtering strategies fail in the same regime. Name the symptom of each
    and say which is more dangerous.
22. Derive {{eq:strategy-crossover}} and compute $s^{*}$ at $N = 10^7$, $k=10$.
    Why is the answer surprising?
23. Explain {{eq:ef-recall-model}}'s shape and why every ANN index has a knee.
24. Why does {{eq:pq-code}}'s product codebook give $2^{bm}$ reconstructions from
    $m2^b$ stored centroids?
25. Derive why ADC beats SDC and state what it costs relative to SDC at query
    time.
26. Raw PQ recall at 32× compression was 39.1%; after reranking the top 100 it
    was 92.0%. Restate PQ's job in one sentence given that.
27. State {{cite:guo2020scann}}'s anisotropic argument, and give one application
    of the same principle outside ANN.

**Lexical, hybrid, and reranking**

28. Explain why a dense embedding structurally cannot store an identifier. Is
    this a training problem?
29. Derive {{eq:bm25-saturation}}'s three limits. What does the $k_1+1$ bound buy
    you with no rule about repetition anywhere in the system?
30. Why is neither $b = 0$ nor $b = 1$ correct? What claim does $b = 0.75$
    encode?
31. Show that under {{eq:rrf}} with $k = 60$, a document at rank 40 in both lists
    beats one at rank 1 in a single list. Then state
    {{eq:fusion-condition}} and explain when RRF *hurts*.
32. Parameter-free interleaving beat RRF on three of four measured rows. Explain
    the mechanism, and say what {{eq:retriever-overlap}} would have told you in
    advance.
33. State {{eq:factorisation-constraint}} and use it to explain why a
    cross-encoder cannot be a first stage and late interaction can.
34. Derive {{eq:recall-ceiling}}. A reranker improved 4 points offline and
    end-to-end quality did not move — give the diagnosis and the next
    measurement.
35. In the measured cascade, the reranker was within 1.6% of its ceiling at
    $k=100$ and 4.3% at $k=1000$. What does that say about "which stage is the
    bottleneck"?
36. Explain {{eq:bottleneck}} and derive the $1/\sqrt{2}$. What is chunking, in
    terms of this equation?

## Practical assignment: a retrieval system and its measurement table

Build a complete retrieval pipeline over a corpus of at least 100,000 documents
that you did not generate — a Wikipedia dump, a package registry, your own
codebase, an email archive. The corpus must be real, because half the failures in
this part come from properties synthetic data does not have.

**Required components.**

1. **An embedding pipeline** with the schema pinned: model version, dimension,
   normalisation, prefix convention, metric, and max sequence length, stored
   *with* the index and validated at query time
   ({{ch:emb-what-they-are}}). Log the truncation rate at ingest.
2. **A geometry report** on the resulting vectors: mean pairwise cosine
   ({{eq:mean-cosine}}), alignment and uniformity ({{eq:alignment}},
   {{eq:uniformity}}), relative contrast ({{eq:relative-contrast}}), and the
   $k$-occurrence distribution ({{eq:k-occurrence}}) — including **the fraction
   of your corpus with $N_k = 0$**, which is the number nobody measures.
3. **A graph or IVF index implemented yourself**, not a library, reported as a
   recall/QPS frontier ({{eq:recall-qps-frontier}}) rather than a point.
4. **Product quantization**, with the ADC/SDC comparison and the exact-rerank
   column ({{ch:emb-ann}}).
5. **A metadata filter** at three selectivities spanning
   {{eq:strategy-crossover}}, with both strategies measured: post-filter recall
   at a capped budget, and the largest connected component of the pre-filtered
   graph.
6. **BM25 from scratch** and an RRF and an interleaving combiner, with
   {{eq:retriever-overlap}} reported first.
7. **A cross-encoder reranker** with the depth swept, reporting first-stage
   recall@$k$ and the **oracle-rerank ceiling** beside end-to-end quality.

**Required evaluation set.** Two to three hundred query–document judgements you
collect yourself, drawn from plausible real queries and deliberately including a
named **identifier slice** and a named **paraphrase slice**. Build this first.
{{eq:paired-eval-size}} says it is enough; {{ch:emb-models}} says it is the
highest-return artefact in the pipeline.

**The deliverable is one table** with a row per configuration and columns for
quality, latency, memory, and build time — plus a paragraph for each row saying
which of {{eq:two-stage-decomposition}}'s two knobs the number implicates.
Anyone can wire up a retrieval pipeline. The part is about knowing which half of
it to fix.

## Advanced challenge

Pick one.

**Find your unreachable documents.** Compute {{eq:k-occurrence}} over your corpus
and identify every document with $N_k = 0$ for a realistic query distribution.
Characterise them: are they short, long, duplicated, off-manifold? Then test
whether per-side centring ({{ch:emb-what-they-are}}), a different pooling, or
chunking rescues them. Report what fraction of the corpus was retrievable in
principle but never in practice, and what it would have cost to find out any
other way.

**Break your own filter.** Measure the selectivity distribution of real
predicates in your system, then find the selectivity at which your index silently
loses recall. Predict it first from {{eq:percolation-threshold}} and your graph's
measured degree, then verify. Report the gap between the prediction and the
measurement, and explain it in terms of filter/position correlation.

**Beat the ceiling instead of the reranker.** Establish that your reranker is
saturated ({{eq:recall-ceiling}}), then raise the ceiling three ways — larger
$k$, hybrid first stage, and distilling the cross-encoder into the bi-encoder —
and compare the three on quality per unit of cost. The distillation arm should
also be evaluated on whether it fixed {{ch:emb-models}}'s false-negative problem;
design that measurement.

## Interview preparation

**"What is an embedding?"** The answer mentions the dot product. A representation
is an embedding only when someone arranged for a dot product in it to mean
something.

**"Cosine or dot product?"** A question about whether your model normalises, not
a preference. On the unit sphere they are the same ranking exactly.

**"A similarity of 0.9 — is that good?"** Nothing absolute. Ask for the corpus's
background similarity. Candidates who answer "yes" have never debugged a
threshold.

**"Doesn't the curse of dimensionality break vector search?"** The theorem is
about intrinsic dimension. Learned embeddings are low-rank and sit outside its
hypotheses. A strong answer states the hypothesis rather than the conclusion.

**"We're switching embedding models next sprint."** Full corpus re-embed, index
rebuild, two indexes during cutover, and the schema validated at query time. It
is a migration, not an upgrade.

**"How do you pick an embedding model?"** Leaderboards narrow it to about five;
two to three hundred paired in-domain judgements decide it. A strong answer says
the evaluation set is built *first*.

**"Why is filtering hard?"** Because an ANN index works by not looking at most of
the data, and a predicate removes part of the data the structure was built from.
Then: post-filter costs $k/s$, pre-filter shatters the graph, and both fail in
the same regime.

**"Our vector search got worse over three months and nothing was deployed."**
Tombstones, codebook staleness, corpus growth, or a shifted selectivity
distribution. A strong answer asks whether recall against exact search is being
measured on a fixed probe set — and knows that nothing else would have caught it.

**"Users can't find things by product code."** The capacity bound. A fixed-width
vector cannot hold millions of identifiers; you need a lexical index, and you
need to check the analyser is not sub-tokenising them.

**"Should we add hybrid search?"** Measure {{eq:retriever-overlap}} first. Then:
fusion is insurance against query heterogeneity, it is often interleaving rather
than RRF, and if you already have a reranker, concatenate instead of fusing.

**"Our reranker improved and the product metric didn't move."** The ceiling.
Report first-stage recall@$k$ and the oracle-rerank number. This question
separates people who have run a retrieval system from people who have read about
one.

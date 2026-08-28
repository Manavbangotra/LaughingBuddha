---
id: part-12-assessment
status: final
---

## How to use this

Four sections. The knowledge check takes about two hours and tells you what to
re-read. The assignment builds a RAG system end to end and — as in {{part:11}} —
the deliverable is the **measurement table and the diagnosis**, not the code,
because every architectural decision in this part is settled by a number you
either measured or did not. The challenge is open-ended. The interview section is
what to rehearse.

No answers are provided. Every question is answerable from the chapters.

## Knowledge check

**Why RAG, and what it costs**

1. State the four properties that decide between parametric and non-parametric
   knowledge ({{ch:rag-why}}), and say which of them fine-tuning can supply.
2. Derive {{eq:rag-ceiling}}. What does it say about the best possible RAG system
   over a corpus that does not contain the answer, and which probe in
   {{ch:rag-failures}} tests it?
3. Using {{eq:cost-ratio}}, state the conditions under which long-context
   stuffing beats retrieval and the conditions under which the comparison flips.
   Why is the honest answer a router rather than a winner?
4. {{ch:llm-long-context}} showed usable context is well below advertised
   context. Explain why that changes the RAG-versus-long-context argument from a
   capability question to a *selection* question, and why explicit selection is
   auditable where implicit selection is not.

**Ingestion and chunking**

5. Define ingestion loss rate ({{eq:ingestion-loss}}) and explain why
   {{eq:true-recall}} means a recall metric computed over the index is an
   overstatement. What is the fix, and why does nobody apply it?
6. Explain {{eq:chunk-dilution}} and use it to predict what happens to a relevant
   sentence buried in a 2,000-token chunk.
7. {{ch:rag-chunking}} measured the optimal chunk size for fact-lookup and
   synthesis queries and found them far apart. State the consequence for every
   published "use 512 tokens" recommendation.
8. Derive {{eq:overlap-return}}. What bounds the benefit of overlap, and what is
   its cost in index size?
9. State {{eq:span-containment}} and describe the measurement that sets your
   window size. What exactly do you count, over what sample?

**Indexing, generation, and citation**

10. Explain {{eq:no-order-in-embedding}} and give two distinct consequences —
    one in {{ch:rag-indexing}} and one in {{ch:rag-structured}}.
11. Why is post-filtering by permission a security problem and not only a recall
    problem? Contrast {{eq:prefilter-permissions}} and
    {{eq:postfilter-permissions}}.
12. State {{eq:deletion-asymmetry}} and explain why a deleted document is harder
    to remove from a RAG system than to add.
13. Derive {{eq:marginal-chunk-value}} and explain why the marginal value of the
    $k$-th chunk eventually goes negative.
14. Using {{eq:u-shape-ordering}}, state where the best chunk should go and why
    "sorted by score" is the wrong assembly order.
15. Explain the claim that **an unverified citation is decoration**. What is
    {{eq:post-verification-rate}} measuring, and why is the verification cheap
    relative to what it catches?

**Query understanding and advanced retrieval**

16. State the asymmetry argument for why a user's question is a poor retrieval
    key ({{eq:asymmetric-score}}), and say what dual encoders fix and what they
    do not ({{eq:asymmetry-coverage}}).
17. Explain why HyDE works even when the hypothetical document is factually
    wrong — and, from {{ch:rag-query-understanding}}'s measurement, the regime in
    which that stops being true.
18. State {{eq:flat-coupling}} and explain, in one sentence, why
    {{ch:rag-chunking}}'s dilemma was never a fact about retrieval.
19. Prove {{eq:pc-dominance}} and give the condition under which parent–child and
    flat chunking are equivalent.
20. {{ch:rag-advanced-retrieval}} found flat $L=12$ beating every parent–child
    configuration on the widest queries. Explain the coupling that
    {{eq:pc-dominance}} does not model, and what it argues for.
21. Derive {{eq:augmentation-condition}}. Which chunks does contextual
    augmentation help, which does it hurt, and what does
    {{eq:context-length-constraint}} recommend as a result?

**Graphs, correction, and loops**

22. Define a global question ({{eq:global-aggregate}}) and prove that top-$k$
    retrieval cannot answer one.
23. {{ch:rag-graph}} measured similarity top-$k$ scoring *worse than a uniform
    random sample* on global questions. Explain via {{eq:selection-bias}} and
    {{eq:bias-floor}}, and say what shape each error curve has.
24. Community summaries plateaued at 0.958 coverage regardless of budget. Name
    the equation responsible and say what a lossless-summary model would have
    reported instead.
25. Derive {{eq:degree-crossover}}. Compute $d^{*}$ at $k = 50$, $p_e = 0.9$, and
    say which two quantities you would measure in your own corpus to apply it.
26. State {{eq:path-reliability}} and {{eq:max-usable-depth}}. How much does a
    ten-point improvement in extraction accuracy change usable traversal depth,
    and why is that parameter almost never reported?
27. Explain why a standard RAG pipeline has **no error condition**, and what
    fraction of its output was confidently wrong in
    {{ch:rag-corrective}}'s measurement.
28. Define a terminal and a recoverable handler ({{eq:terminal-handler}},
    {{eq:recoverable-handler}}). State the measured degradation of each under
    grader noise and the design rule that follows.
29. Why does abstention never raise accuracy? Answer using
    {{eq:retrieval-risk-coverage}}, and say what reporting error this most often
    produces.
30. Derive {{eq:breakeven-solved}}. Explain why the difficulty and the value of
    routing move in opposite directions, and what the first artefact of an
    adaptive-RAG project should therefore be.
31. State {{eq:loop-degenerates}}. Under what condition is an agentic loop
    exactly a chain, and which component of {{ch:rag-corrective}} removes that
    condition?
32. Derive {{eq:quadratic-context}} and compute the cost ratio of a 12-step to a
    1-step query at $B_0 = 900$, $c = 1100$.
33. Explain {{eq:adverse-selection}} and why dead-end queries consumed 53% of
    the token budget while being 15% of traffic.
34. Why is "the model says it is done" a bad termination rule? Answer using
    {{eq:felt-versus-real}}, and give the signal that should replace it.

**Structured data and diagnosis**

35. Prove {{eq:aggregate-unreachable}} and explain why {{ch:rag-graph}}'s
    community-summary escape is unavailable here
    ({{eq:aggregate-combinatorics}}).
36. State {{eq:text-to-sql-factored}} and {{eq:schema-recall-hard-ceiling}}. Why
    does a bigger schema budget eventually lower end-to-end accuracy, and what
    does {{eq:schema-scaling}} predict about benchmark results at scale?
37. Explain {{eq:execution-observability}}. What does running a generated query
    tell you, what does it not, and why is execution success not an accuracy
    metric?
38. Derive {{eq:marginal-stage-value}} and use it to explain why "we tried a
    better embedding model and nothing changed" is usually not evidence about the
    embedding model.
39. Show that individual oracle headrooms do not sum to the gap
    ({{eq:headroom-shortfall}}), and state the direction of the bias
    ({{eq:underrating-factor}}). Why is there no reason to use the biased
    procedure?
40. State {{eq:symptom-collapse}} and explain why a symptom-based triage will
    never name two of the four stages, whatever the system's actual fault.

## Practical assignment: a RAG system and its diagnosis

Build a complete RAG pipeline over a corpus of at least 5,000 **real** documents
that you did not generate — a documentation site, a policy archive, a codebase, a
mailbox, a set of PDFs. It must include PDFs or scans, because
{{ch:rag-ingestion}}'s failures do not occur in clean text and they are where
production systems break.

**Required components.**

1. **An ingestion pipeline that reports its loss rate.** Round-trip a sample of
   documents, measure what fraction of source content reaches the index
   ({{eq:ingestion-loss}}), and report table structure separately from prose.
   **Keep the parsed text store** — {{ch:rag-failures}} shows the whole diagnosis
   depends on it.
2. **A chunking sweep** over at least four sizes, evaluated on *two named query
   types* separately ({{ch:rag-chunking}}). Report the two optima and the
   compromise, and measure your corpus's **orphan-chunk rate** by reading fifty
   chunks by hand.
3. **A parent–child or sentence-window index** compared against your best flat
   configuration **at an equal context budget** ({{eq:parent-child-budget}}), on
   both query types.
4. **Prompt assembly with position control**, measuring answer accuracy against
   the gold chunk's position ({{eq:u-shape-ordering}}), and **citation
   verification** reporting {{eq:post-verification-rate}}.
5. **A retrieval grader** with its own error rate measured against a hundred
   hand-labelled (query, retrieved-set) pairs — not just its presence — plus a
   retry path that is genuinely independent ({{eq:retry-independence}}) and an
   abstention threshold *after* it.
6. **The distraction penalty $\delta$** for your system: the same evaluation with
   retrieval forced on and forced off, compared on the subset the model answers
   correctly without retrieval. Then the **query mix**: two hundred real queries
   labelled for whether the corpus was needed.
7. **The diagnostic ladder** of {{ch:rag-failures}}: prefix-substitution headroom
   per stage ({{eq:prefix-decomposition}}), and the four probes wired as a
   "diagnose this query" command that runs in under a minute.

**Required evaluation set.** Two to three hundred question–answer pairs you write
yourself from the corpus, including a named **fact-lookup slice**, a named
**synthesis slice**, a named **multi-hop slice**, and — the one everyone omits —
a named **unanswerable slice** at its true production rate. Build this first.

**The deliverable is two artefacts.** A table with a row per configuration and
columns for retrieval quality, answer accuracy, harm rate, coverage, latency, and
cost — with retrieval quality and generation quality never collapsed into one
number. And a **one-page diagnosis**: which stage is losing your queries, what
the prefix decomposition says it is worth, and what you would do next. Anyone can
assemble a RAG pipeline. This part is about knowing which stage to fix.

## Advanced challenge

Pick one.

**Measure your ingestion loss and act on it.** Establish the loss rate by
document type ({{eq:ingestion-loss}}), then take the worst type and fix it three
ways: a better parser, a table-aware extractor, and page-image retrieval
({{cite:faysse2025colpali}}). Compare on end-to-end accuracy per unit of cost,
and report how much of your system's headroom was in a stage that no retrieval
metric was measuring.

**Build the corrective loop and prove the ordering.** Implement retrieval
grading, an independent retry, and abstention. Then deliberately degrade the
grader — add noise to its output — and reproduce {{ch:rag-corrective}}'s
asymmetry on your own system: measure how much accuracy each handler loses per
unit of grader noise. Report whether the terminal/recoverable gap survives real,
correlated grader errors, and by how much it shrinks.

**Decide the graph question with numbers.** Measure your corpus's entity degree
distribution and your extractor's edge recall on fifty hand-labelled chunks. Use
{{eq:degree-crossover}} to predict whether a graph index would beat your current
retrieval, and {{eq:graph-build-cost}} plus {{eq:community-instability}} to price
building and maintaining it. Then run the cheap baselines it must beat — a
RAPTOR-style summary tree and a large random sample — and write the one-page
recommendation. **A well-argued "no" is a complete answer to this challenge.**

**Localise, fix, and re-measure.** Run the full prefix-substitution ladder, fix
the top stage, then run it again. Report how much the *other* stages' headroom
moved as a result, and whether the ranking changed — which is
{{ch:rag-failures}}'s central claim, tested on a real system.

## Interview preparation

**"Walk me through a RAG pipeline."** Weak answers list five boxes. Strong
answers name each stage's failure mode and say which is most likely in
production — and mention ingestion, which the box-list never does.

**"What chunk size should we use?"** The wrong question. It depends on the query
distribution, the two common query types have optima that are far apart, and the
better move is to decouple the retrieval unit from the generation unit so the
question stops mattering.

**"Doesn't long context make RAG obsolete?"** A cost argument, not a capability
one, plus the point that usable context is well below advertised context. The
answer is a router, and it moves with token prices.

**"Our answers are wrong. What do you check?"** Answer with the ladder, in order,
starting with *is the correct answer in the corpus at all*. Candidates who start
with the prompt have debugged a demo, not a system.

**"Our retrieval recall@10 is 0.94 and users are unhappy."** Recall against
what — the index, or the corpus? Ingestion loss makes index-recall an
overstatement. Then generation: was the retrieved text used?

**"How do you know the model used what you retrieved?"** Citation verification
against the retrieved spans, not the model's assertion. An unverified citation is
decoration.

**"When would you use GraphRAG?"** When entity degree is high enough that
retrieval slots are diluted, and extraction accuracy is high enough that
traversal survives the hops. Give the two numbers you would measure. A strong
answer also names the cheaper baselines and says they usually win.

**"Should we make it agentic?"** Only if the required depth is unpredictable from
the question. Then: the grader comes first, because a loop without observability
is a chain with a bigger bill, and the cost is quadratic in steps.

**"What does your system do when retrieval fails?"** If the answer is "it
generates anyway", that is the finding. Then: grade, retry independently, abstain
last, and terminate on progress rather than on the model's opinion.

**"Can we point it at our data warehouse?"** Not by embedding the rows — no
top-$k$ contains a `SUM`. Text-to-SQL, and then the real problem is schema
retrieval, which is where the benchmark numbers actually collapse.

**"How would you evaluate this?"** Retrieval quality and generation quality
separately, always. Then: by query type, with an unanswerable slice, at equal
context budgets, and with the evaluation set re-derived from production traffic
because offline validation is not validation.

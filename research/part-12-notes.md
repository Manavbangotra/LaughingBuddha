# Part XII — Retrieval-Augmented Generation: research notes

Research pass run 2026-08-27, before writing. Full tier: 21 sections per
chapter, 4,200-word floor, twelve chapters — the longest part in the book.
Fourteen new bibliography entries, each verified against an arXiv abstract page
or proceedings listing on the date above. 181 entries total, none unverified.

## What this part is, and what it is not

{{part:11}} stopped at *a ranked list of documents*. This part is what happens
next, and it is where two previously separate halves of the book meet:
{{part:10}}'s generation mechanics and {{part:11}}'s retrieval infrastructure.

**The hazard here is different from every previous part's.** Parts IX and X were
haunted by unrefereed sources; {{part:11}} by an excellent literature that
practice ignores. RAG's problem is that **the literature is almost entirely
about the interesting five percent and the failures are almost entirely in the
boring ninety-five.** There are hundreds of papers on retrieval strategies and
approximately none on PDF table extraction, which is where more production RAG
systems fail than on anything the papers discuss.

The rule for this part: **when the literature's attention and the failure mass
are in different places, say so and follow the failure mass.** Concretely, that
means {{ch:rag-ingestion}} and {{ch:rag-chunking}} get full weight despite
having almost no citable literature, and {{ch:rag-graph}} gets an honest
cost accounting rather than an enthusiastic one.

## The organising idea

**RAG is not a technique. It is a decision about where knowledge lives, and every
chapter is a consequence of having moved it out of the weights.**

Once knowledge is external, four problems appear that did not exist before, and
they map onto the part:

```text
   THE MOVE                    WHAT IT CREATES              WHAT IT COSTS
   ───────────────────         ───────────────────────      ────────────────
   106 why knowledge left      107 getting documents in     117 an entirely
       the weights             108 cutting them up              new failure
                               109 finding them again           surface, and
                               110 putting them in a            no single
                                   prompt, with citations       place to
                                                                look
   WHEN THE BASIC LOOP FAILS
   ─────────────────────────────────────────────
   111 the query was wrong          113 the answer isn't in any chunk
   112 the chunk was wrong          114 retrieval itself was wrong
                                    115 one retrieval was never enough
                                    116 the knowledge isn't text
```

The through-line to state in {{ch:rag-why}} and return to in
{{ch:rag-failures}}: **every RAG failure is a failure of one of four stages —
the document never made it in, the chunk was wrong, the retrieval missed, or the
generator ignored what it got — and they need completely different fixes.**
Chapter 117 is where that becomes a diagnostic procedure.

## What changes at this tier for this material

The material is *less* mathematical than {{part:11}} and much more
architectural. That is a hazard, because 21 sections of architecture without
measurement is exactly how a RAG chapter turns into vendor documentation.

- **§6 Mathematical Foundation** must be earned. Where there is genuine
  mathematics — the retrieval/context-budget trade, chunk-boundary probability,
  the cost comparison against long context, citation-verification precision — do
  it properly. Where there is not, use §6 for a **cost model** instead, which is
  the honest quantitative content of most of these chapters.
- **§9 Practical Example** should measure something that changes a decision.
  Chunk size against answer accuracy, context position against use, retrieval
  quality against generation quality. The listings are the defence against
  architecture-as-opinion.
- **§12 Failure Modes** is the richest section in the part and the one readers
  will actually return to. Every chapter's failure modes should be *diagnosable*
  — a symptom, a measurement that distinguishes it from its neighbours, and a
  fix.
- **§14 Evaluation** must separate retrieval quality from generation quality
  every single time. Conflating them is the field's characteristic error and
  {{cite:es2023ragas}} exists because of it.

## The genuinely live questions

### 1. Does long context kill RAG?

The most-asked question about this part, and the answer is a cost argument, not a
capability one. {{cite:li2024ragvslongcontext}} is the systematic study: long
context wins on quality when everything fits, RAG wins on cost by orders of
magnitude, and the paper's own conclusion is a *router*.

Add the second argument the papers underweight: {{ch:llm-long-context}}'s
result that usable context is well below advertised context, and
{{cite:liu2023lost}}'s U-shape. **Stuffing a million tokens does not mean the
model uses a million tokens.** So the comparison is not "retrieval versus no
retrieval" but "explicit selection versus implicit selection", and explicit
selection is auditable while implicit selection is not.

**Do not claim RAG is obsolete and do not claim it is safe.** State the cost
ratio, state the attention-dilution evidence, and note that the answer moves with
token prices.

### 2. Is chunking a solved problem or an admission of defeat?

Both, and the honest framing is {{ch:emb-reranking}}'s: chunking is the poor
practitioner's multi-vector retrieval. It exists because a single embedding
cannot represent a document with several meanings, and it moves the problem to
boundary selection.

Live: whether hierarchical indexing ({{cite:sarthi2024raptor}}) makes the choice
unnecessary by indexing several granularities at once. The evidence is good and
the cost — an LLM summarisation pass over the whole corpus at build time — is
rarely stated alongside it.

**The thing to demonstrate rather than assert:** that the optimal chunk size
depends on the query distribution, not on the corpus, which is why every
published recommendation disagrees.

### 3. Does GraphRAG earn its cost?

{{cite:edge2024graphrag}} identifies a real gap — global questions whose answer
is a property of the corpus rather than of any chunk — and the identification is
the contribution. Whether the answer is worth it is a different question, and the
cost is large: entity extraction and community summarisation over the whole
corpus, re-run on every material update.

**State the cost per document explicitly and let the reader decide.** The
literature almost never does, and the honest comparison is against the much
cheaper alternative of pre-computing summaries at a few granularities
({{cite:sarthi2024raptor}}).

### 4. Should retrieval always happen?

{{cite:asai2023selfrag}}'s reflection tokens make this a learned decision, and
{{cite:jiang2023flare}} makes it a repeated one. The framing worth carrying:
**always-retrieve is a policy, not a default**, and it has costs — latency, token
budget, and the real risk of injecting irrelevant context into a query the model
could have answered.

This is {{ch:llm-routing}}'s escalation decision in a new setting, and the
chapters should say so rather than re-derive it.

### 5. Can groundedness be measured without labels?

{{cite:es2023ragas}} says yes and the tooling assumes it. The caveats are the
LLM-as-judge caveats ({{part:19}}): position bias, self-preference, and
correlation with human judgement that is decent in aggregate and poor per
example.

**The honest recommendation:** reference-free metrics are a monitoring signal,
not an evaluation. They catch regressions; they do not establish quality. A small
human-labelled set is still required, and this is {{ch:emb-models}}'s domain
evaluation set argument again.

## Per-chapter findings

### 106 — Why RAG Exists

Not a motivation chapter. The content is the *decision*: parametric versus
non-parametric knowledge, and the four properties that decide it — freshness,
attribution, access control, and cost of update.
{{cite:izacard2022atlas}} is the quantitative claim (retrieval substitutes for
parameters) and {{cite:lewis2020rag}} the origin.

Must include the long-context comparison ({{cite:li2024ragvslongcontext}}) and
{{ch:fm-what-they-are}}'s adaptation-information-ratio argument: fine-tuning
teaches format reliably and facts poorly, which is *why* facts must be retrieved.

Listing: cost model comparing RAG against long-context stuffing across corpus
sizes and query volumes, with the crossover computed.

### 107 — Document Ingestion and Parsing

The chapter with the least literature and the most production failures. Content:
PDF extraction and why it is genuinely hard, tables, multi-column layout, headers
and footers polluting chunks, OCR, encoding, and deduplication.

**The measurement to make:** ingestion loss rate. Almost nobody measures what
fraction of source content reaches the index, and it is frequently 10–30% for
PDF-heavy corpora.

Listing: a parse-quality harness — round-trip a structured document, measure what
survives, and show table structure destruction concretely.

### 108 — Chunking Strategies

Fixed, recursive, semantic, and document-structure-aware. The honest finding to
demonstrate: **there is no universally optimal chunk size, and the optimum
depends on the query distribution.**

Listing: sweep chunk size against retrieval accuracy for two different query
types (fact-lookup and synthesis) and show the optima disagree. That single plot
retires the "what chunk size should I use" question properly.

Overlap deserves an honest treatment: it is a hedge against boundary loss, it
costs index size linearly, and its benefit is bounded by how often answers
actually straddle boundaries — which is measurable.

### 109 — Indexing, Metadata, and Retrieval

Mostly a bridge back to {{part:11}}, so the new content must be RAG-specific:
metadata schema design, filtering by access control (which is
{{ch:emb-vector-db}}'s pre/post-filter problem with a security consequence),
freshness and incremental update, and multi-tenancy.

Do not re-teach ANN. Reference it and move on.

### 110 — Prompt Construction, Generation, and Citation

The most under-written stage in the field relative to its impact. Content:
context ordering ({{cite:liu2023lost}} — put the best chunk first or last, never
in the middle), the token budget as a constrained optimisation, instructions for
abstention, and **citation as a verification mechanism rather than a UI feature**.

The strong claim worth making and demonstrating: **a citation that is not
verified against the retrieved text is decoration.** Post-hoc verification —
check each claim's cited span actually supports it — is cheap and almost never
done.

Listing: measure answer accuracy against the position of the gold chunk in the
context, reproducing the U-shape in a RAG setting rather than a synthetic one.

### 111 — Query Understanding

{{cite:ma2023rewrite}} and {{cite:gao2023hyde}}. The point to develop: the user's
query is not a good retrieval key, for a specific and derivable reason — it is a
question and the target is an answer, and {{ch:emb-what-they-are}}'s asymmetry
argument applies directly.

HyDE's counterintuitive part deserves emphasis: **the hypothetical document being
factually wrong does not matter**, because it is used only as a retrieval key.

Multi-query and decomposition, with the cost stated: $n$ queries is $n$
retrievals and a fusion step, and {{ch:emb-hybrid}}'s fusion caveats apply.

### 112 — Advanced Retrieval

Parent–child (retrieve small, generate with large), contextual chunk
augmentation, and hierarchical indexing ({{cite:sarthi2024raptor}}). The
organising idea: **these all decouple the retrieval unit from the generation
unit**, which is the single most useful architectural idea in the part.

Listing: parent–child against flat chunking on the same corpus, with the index
cost of each.

### 113 — GraphRAG

Per live question 3. Give the local/global distinction properly, give the cost
honestly, and give the cheaper alternatives their due.

### 114 — Corrective and Adaptive RAG

{{cite:yan2024crag}} and {{cite:asai2023selfrag}}. The framing: **bad retrieval is
an expected condition with a defined handler, not an error.** Retrieval grading,
fallback to web search, and the abstention decision.

Connect explicitly to {{ch:llm-hallucination}}'s abstention material and
{{ch:llm-routing}}'s escalation threshold — this is the same decision.

### 115 — Agentic RAG

{{cite:yao2023react}} and {{cite:jiang2023flare}}. Iterative retrieval, retrieval
as a tool the model calls, and multi-hop. The cost discipline from
{{ch:llm-function-calling}}'s tool-loop equation applies unchanged, and the
compounding-reliability argument is the thing to carry: $n$ retrieval steps at
92% each is not a system.

Forward-reference {{part:17}} and stop. This chapter is about retrieval that
loops, not about agents.

### 116 — Structured and Multimodal RAG

Text-to-SQL as retrieval, table representation for embedding (a genuinely hard
and under-discussed problem), and image retrieval. Forward-reference
{{part:13}} for the multimodal encoders and keep this chapter about the
*retrieval* question.

The point worth making: **structured data does not need to be embedded to be
retrieved**, and a query planner over a schema beats a vector index over
serialised rows almost always.

### 117 — RAG Failure Modes and How to Debug Them

The chapter the part exists for, and it should be a *procedure*, not a list. Four
stages, a measurement that localises the failure to one of them, and a fix per
stage. This is where the part's through-line lands.

Listing: an end-to-end diagnostic harness that takes a failing query and reports
which stage lost it — with the stage-attribution logic made explicit.

## Cross-part bookkeeping

- **Do not** re-teach ANN, embeddings, or reranking. {{part:11}} owns them.
- Agents are {{part:17}}; this part stops at retrieval that loops.
- Multimodal encoders are {{part:13}}.
- Evaluation infrastructure is {{part:25}}; this part uses
  {{cite:es2023ragas}}-style metrics and forward-references the judge caveats.
- Terminology collision check before writing: `chunk`, `context`, `grounding`,
  `citation`, `retrieval`, `ingestion`, `metadata` — `context` and `retrieval`
  almost certainly exist already from Parts X and XI.
- Reuse, do not restate: {{eq:cascade-cost}}, {{eq:risk-coverage}},
  {{eq:groundedness}}, {{eq:u-shape}}, {{eq:tool-chain-success}},
  {{eq:recall-ceiling}}.

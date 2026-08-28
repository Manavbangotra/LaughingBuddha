---
id: part-12-intro
status: final
---

## What this part is for

{{part:11}} stopped at *a ranked list of documents*. This part is what happens
next, and it is where two previously separate halves of the book meet:
{{part:10}}'s generation mechanics and {{part:11}}'s retrieval infrastructure.

**The hazard here is different from every previous part's.** Parts IX and X were
haunted by unrefereed sources; {{part:11}} by an excellent literature that
practice ignores. RAG's problem is that **the literature is almost entirely about
the interesting five percent and the failures are almost entirely in the boring
ninety-five.** There are hundreds of papers on retrieval strategies and
approximately none on PDF table extraction, which is where more production RAG
systems fail than on anything the papers discuss.

> **The rule adopted for this part: when the literature's attention and the
> failure mass are in different places, say so and follow the failure mass.**

Concretely that means {{ch:rag-ingestion}} and {{ch:rag-chunking}} get full weight
despite having almost no citable literature, {{ch:rag-graph}} gets a cost
accounting rather than an enthusiastic one, and {{ch:rag-failures}} — the chapter
with the least novelty and the most operational value — is where the part lands.

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
                                   prompt, with citations       place to look

   WHEN THE BASIC LOOP FAILS
   ─────────────────────────────────────────────
   111 the query was wrong          113 the answer isn't in any chunk
   112 the chunk was wrong          114 retrieval itself was wrong
                                    115 one retrieval was never enough
                                    116 the knowledge isn't text
```

The through-line, stated in {{ch:rag-why}} and returned to in
{{ch:rag-failures}}: **every RAG failure is a failure of one of four stages — the
document never made it in, the chunk was wrong, the retrieval missed, or the
generator ignored what it got — and they need completely different fixes.**
Chapter 117 turns that into a diagnostic procedure, and measures the alternative:
the best possible triage based on what a user can *tell* you gets the stage right
51% of the time, and never once names two of the four stages.

## Six things worth knowing before you start

**The chunk is doing two jobs, and it should not be.** {{ch:rag-chunking}} finds
that fact-lookup and synthesis queries have incompatible optimal chunk sizes, and
{{ch:rag-advanced-retrieval}} shows the dilemma exists *only* because one object
is both the retrieval unit and the generation unit. Separate them and the
constraint disappears rather than being traded off. **This is the highest-value
architectural idea in the part relative to its complexity**, and it recurs — in
{{ch:rag-graph}}'s community summaries, in {{ch:rag-structured}}'s
row-with-header, and anywhere the thing you search by and the thing you send
differ.

**A citation that is not verified against the retrieved text is decoration.**
{{ch:rag-generation}} makes the case that citations are a *verification
mechanism* rather than a UI feature, and post-hoc checking — does the cited span
actually support the claim? — is cheap and almost never done. Without it,
citations make a wrong answer more convincing rather than less.

**Retrieval cannot fail loudly.** Similarity search always returns its $k$
nearest neighbours, however far away they are, so a standard RAG pipeline has
**no error condition** ({{ch:rag-corrective}}). Every other component in a
production system has a failure path; this one silently returns the least-bad
garbage available, and the generator writes a confident paragraph from it.

**Some questions are not retrieval problems at all.** {{ch:rag-graph}} shows that
a question whose answer is a property of the whole corpus cannot be answered by
any top-$k$ — and, more sharply, that similarity ranking is *worse than random
sampling* for such a question, because its error is bias rather than variance.
{{ch:rag-structured}} makes the exact version of the same point: no selection of
$k$ rows contains a `SUM`.

**Unreliable steps are affordable exactly when they are observable and
undoable.** {{ch:rag-corrective}} finds that the same grader does less than half
the damage when its mistakes are recoverable rather than terminal, and
{{ch:rag-agentic}} finds that step *observability* buys more end-to-end success
than step *accuracy* does. One sentence governs both chapters, and it governs
{{part:17}} as well.

**Almost every recommendation in this part is a measurement you have not taken.**
Ingestion loss rate, orphan-chunk rate, answer-span width, entity degree,
extraction accuracy, grader error rate, the distraction penalty, the query mix,
schema recall. Each decides an architecture, each takes an afternoon, and the
literature reports approximately none of them.

## What this part does not cover

Agents as an architecture are {{part:17}} — this part stops at *retrieval that
loops*, and {{ch:rag-agentic}} says so explicitly. Multimodal encoders are
{{part:13}}; {{ch:rag-structured}} covers only the retrieval question about them.
Evaluation infrastructure at scale is {{part:25}}, and the LLM-as-judge caveats
that reference-free RAG metrics inherit belong there. ANN indexes, embedding
models, and reranking are {{part:11}} and are not re-taught here.

## How the chapters build

{{ch:rag-why}} is the decision — parametric against non-parametric knowledge —
and everything else is a consequence of it, so read it first even if the material
seems familiar. {{ch:rag-ingestion}} and {{ch:rag-chunking}} are the two chapters
with the least literature and the most production failures; if you are debugging
a system rather than building one, they are the likeliest answer.
{{ch:rag-indexing}} is mostly a bridge back to {{part:11}} and can be skimmed by
anyone who read it.

{{ch:rag-generation}} is the most under-written stage in the field relative to
its impact. {{ch:rag-query-understanding}} and {{ch:rag-advanced-retrieval}} are
the two highest-return improvements to a working system, in that order.

{{ch:rag-graph}}, {{ch:rag-corrective}}, {{ch:rag-agentic}} and
{{ch:rag-structured}} are the four escalations, and each is written so its cost
is visible: read them to decide whether you need them, which is usually the
correct answer to arrive at.

{{ch:rag-failures}} is the chapter to return to. It is a procedure, it depends on
all eleven before it, and it is the only one that will still be useful when the
techniques have moved on.

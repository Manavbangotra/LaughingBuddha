---
id: fm-how-to-read
status: final
---

## Who this book is for

The book is written for one reader in particular: someone with basic programming
experience, little or no formal machine-learning background, who intends to
reach professional AI engineering competence and then research literacy. That
reader is assumed to be willing to learn Python and to work through mathematics
rather than around it.

Several other readers will find it useful, and should not read it the same way.

{#tbl:reader-paths caption="Reading paths by starting point. Every path ends in the same place; they differ in where they begin and what they skim."}

| If you are | Start at | Skim | Do not skip |
|---|---|---|---|
| New to the field entirely | Part I | Nothing yet | Parts I–VI in order |
| Comfortable with Python and statistics, new to ML | Part IV | Parts I–III as reference | Part VI, then Part VII |
| An ML engineer new to language models | Part VII | Parts I–V | Parts VII–X in order |
| Already building with LLM APIs | Part VII | Parts I–VI | Parts VII, X, XII, XVII, XXV |
| Building RAG or agent systems in production | Part XI | Parts I–VI | Parts XI–XII, XVII–XIX, XXII–XXVI |
| Preparing for interviews | Part IV | — | The Interview Questions section of every chapter |
| Moving toward research | Part VII | Parts II–III | Parts IX, XVI, XXVIII and the bibliography |

The one instruction that applies to everyone: do not skip Part VII. Almost
everything after it is a variation on, extension of, or reaction to the material
in those ten chapters.

## Chapter structure

Chapters use a fixed template so that the book works both as a course and as a
reference. Once you know the template, you know where to look.

Chapters in Parts I–V use a twelve-section **focused** template. Chapters in
Parts VI–XXVIII use a twenty-one-section **full** template, which adds internal
mechanics, production considerations, failure modes, alternatives, evaluation,
research-level extensions, and interview and research questions.

The sections are ordered deliberately. *Intuitive Explanation* comes before
*Formal Explanation* because an intuition you can later make precise is more
useful than a definition you cannot picture. *Failure Modes* comes after
*Production Considerations* because you cannot recognise a failure mode you have
no model of. *Connection to Previous Chapters* exists because the single most
common way to misunderstand this field is to learn its parts in isolation.

Read the whole template on your first few chapters. Afterwards, read what you
need — but read *Common Mistakes* and *Failure Modes* even when you are in a
hurry, because those are the sections that contain what the tutorials leave out.

## Callouts

> NOTE: Elaboration, context, or a connection worth making explicit. Safe to
> skim on a first pass.

> IMPORTANT: A point that is easy to get wrong and expensive to get wrong. Do
> not skim these.

> WARNING: A genuine hazard — a security issue, a silent-corruption bug, a cost
> trap, or a result that does not mean what it appears to mean.

> PRODUCTION TIP: Something that only becomes apparent once a system is serving
> real traffic.

> RESEARCH NOTE: Where the current literature stands, including where it
> disagrees with itself.

> MATH NOTE: A derivation step, a notational subtlety, or a piece of intuition
> about why an equation has the form it has.

> HISTORY: How an idea arrived, which is often the fastest route to why it looks
> the way it does.

## Maturity labels

Any claim tied to a specific technology carries a label:

- {{maturity:ESTABLISHED}} — reproduced independently over years; safe to build on.
- {{maturity:MATURE}} — widely deployed, well understood, with known limits.
- {{maturity:EMERGING}} — real and working, but conventions are still moving.
- {{maturity:EXPERIMENTAL}} — promising results, thin independent replication.
- {{maturity:RESEARCH FRONTIER}} — open problem; treat any claim as provisional.

These are judgements, and they will age. When a section's label and your own
experience disagree, trust your experience and check the date.

## Code

Every code listing carries a verification tier.

**Tier A** listings were executed. They run on a CPU with the dependencies
recorded in the repository, and the build re-runs them; a listing that stops
working fails the build.

**Tier B** listings require a GPU, an API key, network access, or model weights
too large to run on a laptop. They are checked for syntax and import
resolvability and reviewed by hand, and they are labelled in the text as *not
executed locally*. This is an honest limitation, not an oversight: the book
would rather tell you which code was run than imply that all of it was.

**Tier C** listings are illustrative fragments — a few lines making a point in
context — and are marked as not standalone.

Important algorithms are implemented from scratch before any library is used.
The from-scratch version is not a toy for its own sake; it is the version you
can put a print statement inside.

## Mathematics

The mathematics is not decorative and is not skippable, but it is also not
assumed. Part I teaches exactly the mathematics the rest of the book uses,
starting from notation.

Every equation defines its symbols. The notation appendix lists every symbol the
book uses with its fixed meaning, and the meanings do not change between parts.
Where a research paper uses a conflicting convention — and they frequently do,
particularly over whether tokens index rows or columns — the chapter says so
explicitly rather than silently switching.

If a derivation defeats you, read on and return to it. Almost every derivation
in the book is used later in a concrete setting, and the concrete setting often
makes the derivation obvious in retrospect.

## Exercises, and the honest way to use them

Each chapter ends with exercises graded from beginner to implementation-level,
plus interview questions and research questions. Each part ends with a knowledge
check, a practical assignment, and an advanced challenge.

The implementation exercises are the ones that matter. Reading about
backpropagation produces a comfortable feeling of understanding that survives
until you try to write it, at which point you discover which parts you actually
understood. That discovery is the point.

## Pace

At two chapters a week — which is a realistic pace for someone doing this
alongside work — the book is roughly two years of study. At one part a month it
is a little over two years including the projects.

Neither number should be treated as a target. The projects are worth more than
the chapters, and a reader who spends three weeks building the RAG system
properly has learned more than one who read six chapters in the same time.

---
id: <part-prefix>-<concept-slug>          # must match book.yaml
number: <n>                               # must match computed chapter number
part: <ROMAN>
tier: full
status: draft                             # draft | reviewed | final
requires: [<concept-or-chapter-id>, ...]  # must resolve to EARLIER chapters
provides: [<concept>, ...]                # concepts this chapter defines
citations: [<bibkey>, ...]                # every key must be verified
---

## 1. Learning Objectives

Numbered, specific, and testable. "Understand attention" is not an objective;
"derive the 1/√d_k scaling factor from the variance of a dot product" is.

## 2. Why This Matters

The real-world stakes. What breaks, costs money, or becomes impossible without
this. No marketing register.

## 3. Prerequisites

Which earlier chapters, and which specific ideas from them. Use {{ch:...}}
cross-references so the build can verify them.

## 4. Intuitive Explanation

Teach an intelligent beginner. Build one load-bearing mental model rather than
three loose analogies, and say where the model breaks down.

## 5. Formal Explanation

Rigorous definitions. State every shape. Distinguish what is definitional from
what is conventional.

## 6. Mathematical Foundation

Equations with every variable defined, derivations worked rather than asserted,
and at least one numerical example small enough to check by hand. Label
equations that are referenced later: `$$ ... $$ (eq:name)`.

## 7. Internal Mechanics

What actually happens, step by step. A Mermaid data-flow diagram annotated with
shapes belongs here.

## 8. Implementation

From scratch first, then the library. Tag every block: ```python {tier=A
name=slug}. Tier A must execute. Explain the code in prose around it, not only
in comments.

## 9. Practical Example

A realistic scenario, not a toy. Ideally one that shows the concept being used
as a diagnostic.

## 10. Production Considerations

Scalability, reliability, latency, cost, security, observability — whichever
genuinely apply. Say which metric to log.

## 11. Common Mistakes

What beginners get wrong, and separately what experienced people get wrong.
Include the silent failures.

## 12. Failure Modes

Where the approach breaks, what the symptom looks like, and how to detect it.

## 13. Alternatives

A comparison table, with what each alternative trades away. Be explicit about
which alternatives compute the same function and which approximate it.

## 14. Evaluation

How success is measured. Separate "is my implementation correct" from "is this
component behaving well".

## 15. Advanced Concepts

Research-level extensions, each with a {{maturity:...}} label.

## 16. Connection to Previous Chapters

Backwards: which earlier ideas this depends on and where they were established.
Forwards: which later chapters build on this.

## 17. Exercises

Beginner, Intermediate, Advanced, Implementation, and Reasoning subheadings.
The implementation exercises are the ones that matter.

## 18. Interview Questions

Beginner through senior/systems level.

## 19. Research Questions

Questions that send the reader to papers or to an experiment.

## 20. Chapter Summary

The key ideas, restated compactly. A reader who forgot everything else should
retain this.

## 21. Further Reading

Primary sources with {{cite:...}}, then pointers within this book.

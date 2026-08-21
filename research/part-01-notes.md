# Part I — Mathematical Foundations: research notes

Research pass run 2026-08-13, before writing. Records what was checked, what was
decided, and what deliberately was not included.

## State of the subject

Part I is the one part of this book whose content is genuinely stable. Linear
algebra, probability and calculus have not changed; what changes is only which
parts of them matter for machine learning, and even that has moved slowly. There
is therefore no "current state of the art" to track here, and the research pass
was correspondingly narrow: verify the reference works, confirm the standard
pedagogical sequence, and check that nothing in the modern-AI stack has made a
previously minor topic important.

One thing *has* shifted since the pre-2017 generation of "maths for ML" texts,
and Part I reflects it: **the singular value decomposition and the geometry of
high-dimensional inner products now carry far more weight than determinants,
Cramer's rule, or hand-computed matrix inversion.** Chapter 6 is built around the
SVD and low-rank approximation because those are what LoRA (Part XIV), PCA
(Part IV) and the rank constraint inside multi-head attention (Part VII) all
depend on. Determinants are mentioned once and not developed, because in four
years of the rest of this book they are never needed.

A second shift: the variance of inner products in high dimension deserves a
first-class treatment, not a footnote. It is the entire justification for the
attention scaling factor ({{ch:tf-scaled-dot-product}}), and it recurs in
initialisation ({{ch:dl-initialization}}). Chapter 9 therefore covers it
explicitly rather than leaving it to be rediscovered in Part VII.

## Sequence decisions

The conventional ordering is linear algebra → calculus → probability. Part I uses
linear algebra → probability → calculus instead, for one reason: the calculus
chapters (11 and 12) are the hardest in the part, and putting optimisation last
means it can draw on both the vector geometry of Chapters 3–6 and the
expectations of Chapters 7–10. Gradient descent is easier to motivate once the
reader has seen an expectation over a data distribution.

Chapter 5 (norms) is separated from Chapter 3 (vectors) rather than folded into
it, because norms, distances and similarity measures are used as a *toolkit*
throughout Parts XI and XII, and a reader will want to return to one place for
them.

## References checked

All verified against primary sources on 2026-08-13. Recorded in
`data/bibliography.yaml` with `verified_via`.

| Key | What it is | Checked against |
|---|---|---|
| `deisenroth2020` | *Mathematics for Machine Learning*, CUP 2020 | mml-book.github.io — authors, publisher, year, free PDF confirmed |
| `boyd2004` | *Convex Optimization*, CUP 2004 | Cambridge Core record — year, ISBN, DOI |
| `goodfellow2016` | *Deep Learning*, MIT Press 2016 | deeplearningbook.org — authors, publisher, year |
| `robbins1951` | *A Stochastic Approximation Method* | Project Euclid — journal, vol 22(3), 400–407, DOI |
| `kingma2015` | *Adam: A Method for Stochastic Optimization* | arXiv 1412.6980 — v1 2014-12-22, ICLR 2015 |
| `strang2010` | MIT 18.06 Linear Algebra | MIT OCW — instructor, term, CC BY-NC-SA 4.0 |

`deisenroth2020` is the recommended companion for readers who want a second
treatment of Part I; it is freely available, which matters for a book whose
premise is that no other resource should be *required*.

Bishop's *Pattern Recognition and Machine Learning* and Murphy's *Probabilistic
Machine Learning* were considered and deliberately deferred to Part IV, where
their content actually begins. Recommending them in Part I would mislead a
beginner about the prerequisites.

## Not included, and why

- **Determinants beyond a one-line definition.** Not used again in this book.
- **Cramer's rule, adjugates, cofactor expansion.** Superseded by decompositions
  for every purpose this book has.
- **Measure-theoretic probability.** The right level for a reader who wants to
  read modern ML papers is the calculus-based treatment in Chapters 7–10;
  measure theory is needed for a small and clearly marked set of results, all of
  which are flagged where they arise rather than front-loaded.
- **Manual matrix inversion by Gauss-Jordan.** Replaced by solving linear systems
  numerically, which is what any implementation does and is numerically better
  behaved.
- **Multivariable integration.** Used only in stating continuous expectations;
  developed no further than that requires.

## Citations actually used in Part I

Part I is deliberately light on citations. Mathematics of this age has no
"original paper" a reader should go and read, and manufacturing citations for
standard results would be dishonest. Two exceptions are cited because their
origin genuinely matters: `robbins1951` for stochastic approximation, which is
where SGD comes from and which explains its convergence conditions; and
`kingma2015` in Chapter 12, as a forward pointer to how the optimiser the reader
will actually use relates to plain gradient descent.

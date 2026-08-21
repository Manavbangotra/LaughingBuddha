---
id: part-01-intro
status: final
---

## What this part is for

Machine learning is applied mathematics. Not in the sense that you must be a
mathematician to practise it, but in the sense that every design decision you
will make later in this book — why attention divides by a square root, why
learning rates need schedules, why PCA and LoRA are secretly the same idea, why
your A/B test needs more traffic than you expected — is a mathematical fact
wearing engineering clothes.

You can build systems without knowing this. Many people do. What you cannot do
without it is *predict* how those systems will behave, diagnose them when they
fail, or read the literature that explains why anyone chose these designs. This
part exists so that the rest of the book can be honest with you rather than
asking you to accept things.

## What is here, and what is deliberately not

Twelve chapters, covering the mathematics this book actually uses:

- **Chapters 1–2** — notation and functions. How to read mathematical writing
  without stalling on symbols, and the small family of functions that recur
  everywhere: exponentials, logarithms, and the logistic curve.
- **Chapters 3–6** — linear algebra. Vectors and dot products, matrices as
  transformations, norms and distances, and decompositions. This is the
  language every model in this book is written in.
- **Chapters 7–10** — probability and statistics. Uncertainty, distributions,
  the relationships between variables, and how to draw a defensible conclusion
  from a finite sample.
- **Chapters 11–12** — calculus and optimisation. Derivatives and gradients,
  then gradient descent, which is how every model here is trained.

Just as important is what is absent. There are no determinants beyond a
one-line mention, no Cramer's rule, no cofactor expansions, no
measure-theoretic foundations, and no hand-computed matrix inversion. These are
staples of undergraduate courses and are not used again in this book. A
mathematics course must be complete; a book like this one must be *selective*,
and the selection has been made by asking a single question of each topic: does
Part VII through Part XXVIII need it?

Two topics get more space than a conventional treatment gives them, for the
same reason. The **singular value decomposition** (Chapter 6) is developed
properly, because low-rank structure turns out to be the connecting idea behind
PCA, embeddings, LoRA, and the rank constraint inside multi-head attention. And
the **variance of a dot product in high dimension** (Chapter 9) is treated as a
result in its own right rather than an exercise, because it is what justifies
the attention scaling factor and the standard weight initialisations.

## The order, and why it is not the usual one

Conventional sequences run linear algebra → calculus → probability. This part
runs linear algebra → probability → calculus, because the two calculus chapters
are the hardest here and putting optimisation last lets it draw on everything
before it. Gradient descent is much easier to motivate once you can already
write down an expectation over a data distribution.

```mermaid {#fig:part1-deps caption="Dependencies within Part I. The three tracks — linear algebra, probability, calculus — are largely independent until Chapter 12 draws all of them together."}
graph LR
  C1[1 · Notation] --> C2[2 · Functions]
  C2 --> C3[3 · Vectors]
  C3 --> C4[4 · Matrices]
  C3 --> C5[5 · Norms]
  C4 --> C6[6 · Eigen & SVD]
  C5 --> C6
  C2 --> C7[7 · Probability]
  C7 --> C8[8 · Random variables]
  C8 --> C9[9 · Covariance]
  C3 --> C9
  C9 --> C10[10 · Inference]
  C2 --> C11[11 · Derivatives]
  C3 --> C11
  C11 --> C12[12 · Optimisation]
  C5 --> C12
  C8 --> C12
```

If you are impatient, the shortest honest path to Part VI is Chapters 2, 3, 4,
5, 11 and 12 — functions, the linear algebra, and the calculus. You will need
Chapters 7 through 10 before Part III, and Chapter 6 before Part IV.

## How these chapters are structured

Part I uses the book's *focused* twelve-section chapter template rather than
the twenty-one-section template of Parts VI onward. That is a difference of
scope, not of rigour: derivations are still worked rather than asserted, every
concept still gets a numerical example small enough to check by hand, and
almost every chapter has runnable code.

The pattern each chapter follows is deliberate:

1. **Intuition first, then formalism.** An intuition you can later make precise
   beats a definition you cannot picture.
2. **A numerical example you can verify.** If you cannot check a formula on
   three numbers, you do not yet understand it.
3. **Code that runs.** Every listing marked Tier A in this part was executed;
   the arithmetic in the text and the arithmetic in the code agree because both
   are checked by the build.
4. **A forward pointer.** Each chapter says where in the book the idea is
   actually used, so nothing here is motivated purely by "you will need this
   later".

## If the mathematics is hard

It is normal for this to be slow. Reading mathematics is not like reading prose
and cannot be done at the same speed; a page an hour is a reasonable rate for
material you have not seen before, and rereading is the method rather than a
sign of failure.

Three techniques are worth adopting deliberately:

**Write the shapes down.** Above every symbol in an equation, note its
dimension. Most confusion about a formula dissolves once you can see that a
product is conformable and which index is being summed over.

**Try the scalar case.** Set every dimension to one. If the one-dimensional
version is obvious, the general version is usually the same idea applied along
an axis.

**Compute a small example by hand, then check it in code.** Three-element
vectors and 2×2 matrices are enough to expose almost any misunderstanding, and
they are small enough that you can find your own arithmetic mistakes.

> IMPORTANT: Do not attempt to memorise this part. It is a reference as much as
> a course, and the notation appendix and formula sheet exist precisely so that
> you do not have to hold it all in your head. What you should aim to retain is
> the *shape* of each idea — what a dot product measures, what an eigenvector
> is for, what a p-value does and does not say. The details can be looked up;
> the shapes cannot.

## What you should be able to do at the end

Read an equation in a machine learning paper and know what type each symbol is.
Multiply matrices and predict the output shape before computing it. Explain
what a dot product measures and why cosine similarity normalises it. Apply
Bayes' theorem to a concrete problem and get the counter-intuitive answer
right. Say what a p-value means without saying anything false. Compute a
gradient by hand for a small composed function. Explain why gradient descent
works, and the two distinct ways it can fail.

The knowledge check and assignments at the end of this part test exactly these.

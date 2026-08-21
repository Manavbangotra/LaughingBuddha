---
id: math-vectors
number: 3
part: I
tier: focused
status: reviewed
requires: [math-notation, math-functions]
provides: [vector, vector-space, linear-combination, dot-product, orthogonality]
citations: [deisenroth2020, strang2010]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Read a vector as both a list of numbers and a point or arrow in space, and
   switch between the two readings deliberately.
2. Add, scale, and take linear combinations of vectors, and say what each
   operation does geometrically.
3. Compute a dot product, and state the three things it measures: agreement,
   projection, and angle.
4. Derive the geometric form $\vec{x}\T\vec{y} = \norm{\vec{x}}\norm{\vec{y}}\cos\theta$
   from the algebraic definition.
5. Determine whether two vectors are orthogonal and explain what orthogonality
   means for the information they carry.
6. Compute the projection of one vector onto another and interpret it.
7. Explain what a basis and a span are, and why dimension is the number of
   independent directions rather than the number of stored numbers.
8. Describe what changes about geometric intuition in high dimensions, and why
   that matters for embeddings.

## 2. Why This Matters

Everything in this book is a vector.

That is not a figure of speech. A word is a vector ({{ch:nlp-static-embeddings}}).
A sentence is a vector ({{ch:nlp-similarity}}). An image patch is a vector
({{ch:mm-vit}}). A user's preferences, a document, a molecule, the internal
state of a language model at layer 30 — all vectors. The reason modern AI can
treat text, images, and audio with the same machinery is that all three are
first converted into the same kind of object: a list of numbers in a space where
distance means something.

And the operation performed on those vectors more than any other, by an enormous
margin, is the dot product. A single forward pass through a large language model
is on the order of $10^{11}$ dot products. Attention is dot products
({{ch:tf-scaled-dot-product}}). Retrieval is dot products
({{ch:emb-similarity}}). A neural network layer is dot products
({{ch:dl-forward}}). Learning what a dot product measures is therefore not
preliminary work — it is most of the intuition you need for the rest of the
book.

## 3. Prerequisites

{{ch:math-notation}} for summation notation and typography;
{{ch:math-functions}} for function signatures. Basic coordinate geometry — the
idea that $(3, 4)$ names a point on a plane — is helpful but is re-derived here.

## 4. Intuitive Explanation

### 4.1 Two readings of the same object

A {{term:vector}} is an ordered list of numbers:

$$
\vec{x} = \begin{bmatrix} 3 \\ 4 \end{bmatrix}
$$

There are two ways to read it, and fluency means holding both at once.

**As a point.** $\vec{x}$ names the location three units along the first axis
and four along the second. This reading is natural for data: a house with three
bedrooms and four hundred square metres is a point in "house space".

**As an arrow.** $\vec{x}$ is a displacement — a direction together with a
distance — starting at the origin. This reading is natural for change: a
gradient is an arrow saying which way to move.

Neither reading is more correct. The point reading dominates when you think
about data; the arrow reading dominates when you think about operations. The
book switches between them without warning, as all mathematical writing does,
and the switch is usually signalled by the verb: points *are somewhere*, arrows
*point somewhere*.

> NOTE: In this book, vectors are column vectors — written vertically, or
> horizontally with a transpose, as $\vec{x} = [3, 4]\T$. This is a convention,
> not a fact, but it is a consistent one, and it determines the shape of every
> matrix product in {{ch:math-matrices}}.

### 4.2 Addition is following one arrow then another

Adding two vectors adds them componentwise:

$$
\begin{bmatrix} 3 \\ 4 \end{bmatrix} + \begin{bmatrix} 1 \\ -2 \end{bmatrix}
= \begin{bmatrix} 4 \\ 2 \end{bmatrix}
$$

Geometrically: walk along the first arrow, then from where you land, walk along
the second. You arrive at the sum. It follows immediately that order does not
matter — walking east then north lands you where walking north then east does.

Scaling multiplies every component by the same number, which stretches the arrow
without turning it. A negative scalar reverses it. Note what scaling does *not*
do: it never changes direction except to reverse it, which is why "direction"
and "magnitude" can be separated cleanly ({{ch:math-norms}}).

### 4.3 The dot product measures agreement

Take two vectors, multiply them component by component, and add the results:

$$
\vec{x}\T\vec{y} = \sum_{i=1}^{n} x_i y_i
$$ (eq:dot-product)

This is the {{term:dot-product}}, and it is the most important operation in this
book. What it measures is *agreement*, in a precise sense: it is large and
positive when the two vectors point the same way, near zero when they are
perpendicular, and negative when they oppose.

The intuition is worth building carefully. Consider matching a query against
three documents, where each is described by how much it concerns three topics —
sport, finance, cooking:

$$
\vec{q} = \begin{bmatrix} 5 \\ 0 \\ 0 \end{bmatrix},\quad
\vec{d}_1 = \begin{bmatrix} 4 \\ 1 \\ 0 \end{bmatrix},\quad
\vec{d}_2 = \begin{bmatrix} 0 \\ 5 \\ 0 \end{bmatrix},\quad
\vec{d}_3 = \begin{bmatrix} 0 \\ 0 \\ 6 \end{bmatrix}
$$

The dot products are $\vec{q}\T\vec{d}_1 = 20$, $\vec{q}\T\vec{d}_2 = 0$,
$\vec{q}\T\vec{d}_3 = 0$. The query is about sport; document 1 is mostly about
sport and scores highly; documents 2 and 3 are about something else entirely and
score zero. This is, in outline, exactly how dense retrieval works
({{ch:emb-similarity}}) and exactly how an attention head decides what to read
({{ch:tf-scaled-dot-product}}).

Two things about {{eq:dot-product}} are worth noticing immediately. It is
*cheap*: $n$ multiplications and $n-1$ additions, and modern hardware does
thousands in parallel. And it collapses two lists into a single number — it is a
compression from $\R^n \times \R^n$ to $\R$, which is what makes it usable as a
score.

### 4.4 Orthogonality is independence of direction

When the dot product is zero, the vectors are {{term:orthogonality}} — the
generalisation of perpendicular to any number of dimensions.

The useful way to think about orthogonal directions is that they carry
independent information. Moving along one changes nothing about your position
along the other. In an embedding space, if the "plural" direction and the
"past tense" direction are orthogonal, a model can vary one without disturbing
the other, and both can be read out independently.

This is not merely elegant. A residual stream of width $d$ can hold at most $d$
mutually orthogonal directions, which puts a hard ceiling on how many features
can be represented without interference — a constraint that
{{ch:tf-embeddings}} returns to, and one reason model width matters.

## 5. Formal Explanation

### 5.1 Vectors and vector spaces

A vector in $\R^{n}$ is an ordered $n$-tuple of real numbers. The set $\R^{n}$
with componentwise addition and scalar multiplication is a
{{term:vector-space}}: it is closed under both operations, and satisfies the
usual axioms (associativity and commutativity of addition, an additive identity
$\vec{0}$ and inverses, and compatibility of scalar multiplication).

The two operations, componentwise:

$$
(\vec{x} + \vec{y})_i = x_i + y_i, \qquad (c\vec{x})_i = c\,x_i
$$ (eq:vector-ops)

A {{term:linear-combination}} of vectors $\vec{v}_1, \ldots, \vec{v}_k$ is any

$$
c_1\vec{v}_1 + c_2\vec{v}_2 + \cdots + c_k\vec{v}_k = \sum_{j=1}^{k} c_j\vec{v}_j
$$ (eq:linear-combination)

for scalars $c_j$. This is the single most important construction in linear
algebra, and it is worth registering how much of this book is an instance of it:
a neuron's pre-activation is a linear combination of its inputs; an attention
output is a linear combination of value vectors, with the coefficients
constrained to be non-negative and sum to one; a PCA reconstruction is a linear
combination of principal directions.

The **span** of a set of vectors is the set of all their linear combinations —
everything reachable. Vectors are **linearly independent** if none is a linear
combination of the others; equivalently, if $\sum_j c_j\vec{v}_j = \vec{0}$ only
when every $c_j = 0$. A **basis** is a linearly independent set that spans the
whole space, and the **dimension** is the size of any basis.

> IMPORTANT: Dimension counts *independent directions*, not stored numbers.
> Three vectors in $\R^{3}$ that all lie in one plane span only a
> two-dimensional subspace, however many components each has. This distinction
> is the whole content of {{term:rank}} ({{ch:math-matrices}}) and the reason
> low-rank approximation works ({{ch:math-eigen}}).

### 5.2 The dot product, formally

For $\vec{x}, \vec{y} \in \R^{n}$, {{eq:dot-product}} defines the dot product.
Its algebraic properties:

$$
\vec{x}\T\vec{y} = \vec{y}\T\vec{x}
\qquad\text{(symmetric)}
$$ (eq:dot-symmetric)

$$
\vec{x}\T(\vec{y} + \vec{z}) = \vec{x}\T\vec{y} + \vec{x}\T\vec{z},
\qquad
(c\vec{x})\T\vec{y} = c(\vec{x}\T\vec{y})
\qquad\text{(bilinear)}
$$ (eq:dot-bilinear)

$$
\vec{x}\T\vec{x} = \sum_i x_i^{2} \ge 0,
\quad\text{with equality iff } \vec{x} = \vec{0}
\qquad\text{(positive definite)}
$$ (eq:dot-positive)

{{eq:dot-positive}} is what connects the dot product to length: the Euclidean
norm is defined as $\norm{\vec{x}}_2 = \sqrt{\vec{x}\T\vec{x}}$
({{ch:math-norms}}).

Two vectors are **orthogonal** when $\vec{x}\T\vec{y} = 0$. Note that
$\vec{0}$ is orthogonal to everything, which is a degenerate but consistent case.

### 5.3 The geometric form

The algebraic definition {{eq:dot-product}} and the geometric statement

$$
\vec{x}\T\vec{y} = \norm{\vec{x}}\,\norm{\vec{y}}\cos\theta
$$ (eq:dot-geometric)

are the same thing. {{sec:6-mathematical-foundation}} proves it. Taking it for
now, it explains everything claimed in {{sec:4-intuitive-explanation}}:

- $\theta = 0$: vectors aligned, $\cos\theta = 1$, dot product maximal.
- $\theta = 90°$: perpendicular, $\cos\theta = 0$, dot product zero.
- $\theta = 180°$: opposed, $\cos\theta = -1$, dot product maximally negative.

It also exposes something important: the dot product mixes *direction* and
*magnitude*. A long vector pointing somewhat the wrong way can outscore a short
vector pointing exactly the right way. Whether that is desirable depends
entirely on the application, and it is the reason cosine similarity exists
({{ch:math-norms}}) — it divides the magnitudes out, keeping only the angle.

### 5.4 Projection

The **scalar projection** of $\vec{x}$ onto $\vec{y}$ is how far $\vec{x}$
extends in the direction of $\vec{y}$:

$$
\text{comp}_{\vec{y}}\vec{x} = \frac{\vec{x}\T\vec{y}}{\norm{\vec{y}}}
$$ (eq:scalar-projection)

The **vector projection** is that length, pointed along $\vec{y}$:

$$
\text{proj}_{\vec{y}}\vec{x}
  = \frac{\vec{x}\T\vec{y}}{\vec{y}\T\vec{y}}\,\vec{y}
$$ (eq:vector-projection)

Projection is the operation behind least-squares regression — the fitted values
are the projection of the targets onto the space the features can span
({{ch:ml-linear-regression}}) — and behind PCA, where data is projected onto the
directions of greatest variance ({{ch:ml-pca}}).

The residual $\vec{x} - \text{proj}_{\vec{y}}\vec{x}$ is always orthogonal to
$\vec{y}$. That is worth verifying yourself, and Exercise 11 asks you to.

## 6. Mathematical Foundation

### 6.1 Proving the geometric form

{{eq:dot-geometric}} looks like a separate definition. It is a theorem, and it
follows from the law of cosines.

Consider the triangle with sides $\vec{x}$, $\vec{y}$, and $\vec{x} - \vec{y}$,
with $\theta$ the angle between $\vec{x}$ and $\vec{y}$. The law of cosines
gives

$$
\norm{\vec{x} - \vec{y}}^{2}
  = \norm{\vec{x}}^{2} + \norm{\vec{y}}^{2} - 2\norm{\vec{x}}\norm{\vec{y}}\cos\theta
$$ (eq:law-of-cosines)

Now expand the left side algebraically, using $\norm{\vec{v}}^2 = \vec{v}\T\vec{v}$
and bilinearity {{eq:dot-bilinear}}:

$$
\norm{\vec{x} - \vec{y}}^{2}
  = (\vec{x} - \vec{y})\T(\vec{x} - \vec{y})
  = \vec{x}\T\vec{x} - 2\vec{x}\T\vec{y} + \vec{y}\T\vec{y}
  = \norm{\vec{x}}^{2} - 2\vec{x}\T\vec{y} + \norm{\vec{y}}^{2}
$$ (eq:expand-difference)

Setting {{eq:law-of-cosines}} equal to {{eq:expand-difference}}, the
$\norm{\vec{x}}^{2}$ and $\norm{\vec{y}}^{2}$ terms cancel from both sides,
leaving

$$
-2\vec{x}\T\vec{y} = -2\norm{\vec{x}}\norm{\vec{y}}\cos\theta
$$

and dividing by $-2$ gives {{eq:dot-geometric}}. The algebraic and geometric
definitions are one definition seen from two directions.

> MATH NOTE: This proof is the reason {{eq:dot-geometric}} can be *used* rather
> than merely asserted. It also explains why the angle between two vectors in
> $\R^{1000}$ is a perfectly meaningful quantity even though you cannot picture
> it: the angle is *defined* by rearranging {{eq:dot-geometric}}, and every
> property it has follows from the algebra, not from a picture.

### 6.2 The Cauchy-Schwarz inequality

An immediate consequence, since $\lvert\cos\theta\rvert \le 1$:

$$
\lvert \vec{x}\T\vec{y} \rvert \le \norm{\vec{x}}\,\norm{\vec{y}}
$$ (eq:cauchy-schwarz)

with equality exactly when the vectors are parallel. This bounds how large a dot
product can be, and it is what guarantees that cosine similarity lies in
$[-1, 1]$ ({{ch:math-norms}}). It is also the reason a normalised embedding's
inner product can be read directly as a similarity score without further
rescaling.

### 6.3 A worked numerical example

Take $\vec{x} = [3, 4]\T$ and $\vec{y} = [4, 3]\T$.

**Dot product.** $3 \cdot 4 + 4 \cdot 3 = 12 + 12 = 24$.

**Norms.** $\norm{\vec{x}} = \sqrt{9 + 16} = 5$ and
$\norm{\vec{y}} = \sqrt{16 + 9} = 5$.

**Angle.** From {{eq:dot-geometric}}, $\cos\theta = 24 / (5 \cdot 5) = 0.96$, so
$\theta = \arccos(0.96) \approx 16.26°$. The two vectors are nearly aligned,
which matches the picture — both point up and to the right at similar slopes.

**Projection.** From {{eq:vector-projection}},
$\text{proj}_{\vec{y}}\vec{x} = \frac{24}{25}[4, 3]\T = [3.84, 2.88]\T$.

**Residual.** $\vec{x} - \text{proj}_{\vec{y}}\vec{x} = [-0.84, 1.12]\T$. Check
orthogonality: $(-0.84)(4) + (1.12)(3) = -3.36 + 3.36 = 0$. It is orthogonal, as
promised.

### 6.4 High dimensions behave differently

Geometric intuition is trained in two and three dimensions, and it does not
transfer cleanly. Since embeddings live in hundreds or thousands of dimensions,
the differences matter.

**Random vectors are nearly orthogonal.** Draw two vectors uniformly at random
from the unit sphere in $\R^{n}$. The expected cosine of the angle between them
is 0, and the standard deviation is approximately $1/\sqrt{n}$. In $\R^{2}$
random vectors are at all sorts of angles; in $\R^{1000}$ they are almost
always within a few degrees of perpendicular.

This is genuinely useful rather than merely curious. It means a
high-dimensional space can hold an enormous number of *approximately*
orthogonal directions — far more than $n$ — so a model can assign nearly
independent directions to far more features than it has dimensions. This
observation is the starting point for the superposition hypothesis in
interpretability ({{ch:rai-interpretability}}).

**Distances concentrate.** As dimension grows, the ratio between the farthest
and nearest points in a random set approaches 1. Everything is roughly the same
distance from everything else, which is why nearest-neighbour search in high
dimensions is harder than it sounds and why approximate methods
({{ch:emb-ann}}) are not merely a speed optimisation.

**Volume moves to the shell.** Almost all the volume of a high-dimensional ball
lies near its surface. A "typical" random point is not near the centre.

{{sec:7-implementation}} demonstrates all three numerically, which is the only
way they become believable.

## 7. Implementation

```python {tier=A name=vectors-and-dot-products}
"""Vectors, dot products, projection, and what changes in high dimensions.

Everything the chapter asserts geometrically is checked numerically here.
"""
import numpy as np

# --- basics ------------------------------------------------------------------
x = np.array([3.0, 4.0])
y = np.array([4.0, 3.0])

print("x + y      :", x + y)
print("2x         :", 2 * x)
print("x . y      :", x @ y, "| same via sum:", np.sum(x * y))

# The three equivalent spellings of a dot product in NumPy. Prefer @ or np.dot;
# `*` is ELEMENTWISE and silently gives a vector, not a scalar.
assert x @ y == np.dot(x, y) == np.sum(x * y) == 24.0
print("elementwise x * y :", x * y, " <- NOT the dot product")

# --- norms and the angle (eq. 3.9) ------------------------------------------
nx, ny = np.linalg.norm(x), np.linalg.norm(y)
cos_theta = (x @ y) / (nx * ny)
print(f"\n|x| = {nx}, |y| = {ny}")
print(f"cos(theta) = {cos_theta:.4f}, theta = {np.degrees(np.arccos(cos_theta)):.2f} deg")
assert np.isclose(cos_theta, 0.96)

# --- projection (eq. 3.11) and the orthogonal residual ----------------------
proj = ((x @ y) / (y @ y)) * y
residual = x - proj
print(f"\nprojection of x onto y : {proj}")
print(f"residual               : {residual}")
print(f"residual . y           : {residual @ y:.2e}  <- orthogonal, as claimed")
assert abs(residual @ y) < 1e-12

# --- eq. 3.13: Cauchy-Schwarz, checked on random pairs ----------------------
rng = np.random.default_rng(0)
for _ in range(1000):
    a, b = rng.normal(size=5), rng.normal(size=5)
    assert abs(a @ b) <= np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
print("\nCauchy-Schwarz holds on 1000 random pairs")

# --- orthogonality carries independent information --------------------------
e1, e2 = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
v = 3 * e1 + 7 * e2
print(f"\nv = 3*e1 + 7*e2 = {v}")
print(f"read off the e1 component with a dot product: {v @ e1}")
print(f"read off the e2 component: {v @ e2}   <- unaffected by the e1 part")

# --- high dimensions: random vectors are nearly orthogonal ------------------
print(f"\n{'dim':>6} {'mean |cos|':>12} {'std cos':>10} {'1/sqrt(n)':>11} "
      f"{'% within 5 deg of 90':>22}")
for n in (2, 3, 10, 100, 1000, 10000):
    a = rng.normal(size=(4000, n))
    b = rng.normal(size=(4000, n))
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    cos = np.sum(a * b, axis=1)
    angles = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    near_perp = np.mean(np.abs(angles - 90) < 5) * 100
    print(f"{n:>6} {np.abs(cos).mean():>12.4f} {cos.std():>10.4f} "
          f"{1/np.sqrt(n):>11.4f} {near_perp:>21.1f}%")

print("\nstd of the cosine tracks 1/sqrt(n): in high dimensions two random")
print("directions are almost always close to perpendicular.")

# --- high dimensions: distances concentrate ---------------------------------
print(f"\n{'dim':>6} {'nearest':>10} {'farthest':>10} {'ratio':>8}")
for n in (2, 10, 100, 1000):
    pts = rng.normal(size=(500, n))
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    d = d[~np.eye(len(pts), dtype=bool)]
    print(f"{n:>6} {d.min():>10.3f} {d.max():>10.3f} {d.max()/d.min():>8.2f}")

print("\nThe far/near ratio collapses toward 1 as dimension grows — which is")
print("why exact nearest-neighbour search stops being informative (Part XI).")
```

## 8. Practical Example

Semantic search is a dot product, and seeing that concretely is worth more than
a paragraph of description.

A retrieval system stores each document as a vector produced by an embedding
model ({{ch:emb-models}}), arranged so that documents about similar things point
in similar directions. A query is embedded the same way. Ranking is then a
matter of computing one dot product per document and sorting.

```python {tier=A name=semantic-search-toy}
"""Ranking by dot product — the arithmetic underneath semantic search.

The embeddings here are hand-built so the geometry is inspectable. Real ones
come from a trained model (Part XI); the ranking arithmetic is identical.
"""
import numpy as np

# Three interpretable axes: [sport, finance, cooking]
docs = {
    "Match report: late winner":      np.array([9.0, 0.0, 0.0]),
    "Club posts record revenues":     np.array([6.0, 7.0, 0.0]),
    "Interest rates held steady":     np.array([0.0, 9.0, 0.0]),
    "Braising for beginners":         np.array([0.0, 0.0, 8.0]),
    "Stadium caterer wins award":     np.array([3.0, 1.0, 6.0]),
}
names = list(docs)
D = np.stack([docs[k] for k in names])          # (5, 3)

query = np.array([5.0, 4.0, 0.0])               # "football club finances"

# One matrix-vector product scores every document at once (Chapter 4).
scores = D @ query
order = np.argsort(-scores)

print("query: football club finances -> [sport=5, finance=4, cooking=0]\n")
print(f"{'score':>7}  {'cos':>6}  document")
for i in order:
    cos = scores[i] / (np.linalg.norm(D[i]) * np.linalg.norm(query))
    print(f"{scores[i]:>7.1f}  {cos:>6.3f}  {names[i]}")

print("\nNote the disagreement between the two columns:")
best_dot = names[int(np.argmax(scores))]
cosines = scores / (np.linalg.norm(D, axis=1) * np.linalg.norm(query))
best_cos = names[int(np.argmax(cosines))]
print(f"  highest dot product : {best_dot}")
print(f"  highest cosine      : {best_cos}")
print("The dot product rewards long vectors — documents that are simply")
print("'about more' — while the cosine judges direction alone. Which you want")
print("is a design decision, not a detail (Chapter 5).")
```

The disagreement that code prints is the practical lesson. The dot product
conflates *relevance* with *magnitude*, and embedding magnitude often encodes
something incidental — document length, or how confidently the encoder placed
it. Most retrieval systems normalise their embeddings for exactly this reason,
which turns the dot product into cosine similarity and is why
{{ch:math-norms}} exists.

## 9. Common Mistakes

**Using `*` for the dot product in NumPy.** `x * y` is elementwise and returns a
vector; `x @ y` returns a scalar. The bug is silent when downstream code
broadcasts, which it often does.

**Confusing dimension with number of components.** Three vectors in $\R^{3}$
lying in a plane span two dimensions, not three. Dimension counts independent
directions.

**Assuming the dot product measures similarity directly.** It measures
similarity *scaled by both magnitudes*. Unless your vectors are normalised, a
high dot product may mean "very relevant" or merely "very long".

**Trusting two- and three-dimensional intuition in high dimensions.** Nearly
everything about distance and angle behaves differently at $n = 1000$. The
demonstration in {{sec:7-implementation}} is worth running rather than reading.

**Forgetting that orthogonal does not mean independent in the statistical
sense.** Orthogonality is a geometric property of two fixed vectors; statistical
independence is a property of random variables ({{ch:math-probability}}). Zero
correlation and orthogonality of centred data vectors do coincide
({{ch:math-covariance}}), which makes the confusion easy — but they are
different claims.

**Mixing row and column conventions.** $\vec{x}\T\vec{y}$ is a scalar;
$\vec{x}\vec{y}\T$ is an $n \times n$ matrix. The notation differs by one
transpose and the results differ by everything.

## 10. Connection to Previous Chapters

{{ch:math-notation}} gave the summation notation that {{eq:dot-product}} is
written in, and the typography that distinguishes $\vec{x}$ from $x$.
{{ch:math-functions}} supplied the function signatures — the dot product is a
function $\R^{n} \times \R^{n} \to \R$.

Forward: {{ch:math-matrices}} shows that a matrix product is a grid of dot
products, and that a matrix is a function acting on vectors.
{{ch:math-norms}} builds length and distance on {{eq:dot-positive}} and derives
cosine similarity from {{eq:dot-geometric}}. {{ch:math-eigen}} finds the
special directions a transformation does not rotate. {{ch:math-covariance}}
shows that covariance is a dot product of centred data, and
{{ch:math-derivatives}} shows that the directional derivative is a dot product
with the gradient.

Beyond Part I, {{ch:tf-scaled-dot-product}} is this chapter's dot product used
as a learned matching score, and {{ch:emb-similarity}} is it used as a retrieval
score. {{cite:strang2010}} is the recommended free course for readers who want a
fuller treatment of the linear algebra.

## 11. Exercises

**Beginner**

1. For $\vec{a} = [1, 2, 3]\T$ and $\vec{b} = [4, 5, 6]\T$, compute
   $\vec{a} + \vec{b}$, $3\vec{a}$, and $\vec{a}\T\vec{b}$.
2. Are $[1, 2]\T$ and $[-2, 1]\T$ orthogonal? Show the computation.
3. Compute $\norm{[6, 8]\T}$ and find the unit vector in the same direction.
4. Write $[7, 5]\T$ as a linear combination of $[1, 0]\T$ and $[0, 1]\T$.
5. Give a vector orthogonal to $[3, 0, 4]\T$. How many such vectors are there?

**Intermediate**

6. Compute the angle between $[1, 0]\T$ and $[1, 1]\T$ using
   {{eq:dot-geometric}}.
7. Do $[1, 2]\T$, $[2, 4]\T$ and $[3, 1]\T$ span $\R^{2}$? Which of them is
   redundant, and why?
8. Compute the projection of $[5, 2]\T$ onto $[1, 1]\T$, and verify that the
   residual is orthogonal to $[1, 1]\T$.
9. Two documents have embeddings $[3, 4]\T$ and $[30, 40]\T$. Compare them by
   dot product and by angle. Which comparison would you want in a search system,
   and why?
10. Explain why the dot product of a vector with itself can never be negative,
    and what that fact is used for.

**Advanced**

11. Prove that $\vec{x} - \text{proj}_{\vec{y}}\vec{x}$ is orthogonal to
    $\vec{y}$ for any nonzero $\vec{y}$.
12. Prove the Cauchy-Schwarz inequality {{eq:cauchy-schwarz}} *without* using
    {{eq:dot-geometric}}. (Hint: consider
    $\norm{\vec{x} - t\vec{y}}^{2} \ge 0$ as a quadratic in $t$ and require its
    discriminant to be non-positive.)
13. Prove the triangle inequality
    $\norm{\vec{x} + \vec{y}} \le \norm{\vec{x}} + \norm{\vec{y}}$ from
    Cauchy-Schwarz.
14. Show that if $\{\vec{v}_1, \ldots, \vec{v}_k\}$ are mutually orthogonal and
    nonzero, they are linearly independent. What upper bound does this place on
    $k$ in $\R^{n}$?

**Implementation**

15. Implement the dot product with an explicit loop and verify it against `@` on
    random vectors of length 1000. Then time both, and explain the difference.
16. Implement `project(x, y)` returning the vector projection, and write a test
    asserting the residual is orthogonal.
17. Extend the high-dimensional experiment in {{sec:7-implementation}} to
    measure what fraction of a unit ball's volume lies within the inner 90% of
    its radius, as a function of $n$. Do this by sampling, and explain the
    result.
18. Take a set of 10 random vectors in $\R^{2}$ and in $\R^{500}$. In each case,
    compute all pairwise cosines and report the maximum. Explain the difference
    in terms of the near-orthogonality result.

**Reasoning**

19. A model's residual stream has width 768. Argue that it can represent far
    more than 768 distinguishable features, and state precisely what is given up.
20. Retrieval systems usually normalise embeddings before indexing. Give one
    situation where *not* normalising is the better choice.

## 12. Chapter Summary

A vector is an ordered list of numbers, readable as a point or as an arrow; both
readings are used constantly and fluency means switching between them freely.
Addition and scaling operate componentwise and correspond to following arrows in
sequence and to stretching them.

A linear combination is a weighted sum of vectors, and it is the construction
underlying almost every operation in this book — neuron pre-activations,
attention outputs, PCA reconstructions. The span of a set of vectors is
everything their linear combinations reach; the dimension of a space is the
number of independent directions in it, which is not the same as the number of
components stored.

The dot product multiplies componentwise and sums. It measures agreement:
positive for aligned vectors, zero for orthogonal ones, negative for opposed
ones. Algebraically it is symmetric, bilinear and positive definite;
geometrically it equals $\norm{\vec{x}}\norm{\vec{y}}\cos\theta$, and the two
descriptions are provably the same via the law of cosines. Cauchy-Schwarz bounds
it by the product of the norms.

Because the dot product mixes direction with magnitude, it is not by itself a
similarity measure — a long irrelevant vector can outscore a short relevant one.
Removing the magnitudes gives cosine similarity, which is why retrieval systems
normalise.

Orthogonal directions carry independent information, and a $d$-dimensional space
holds at most $d$ mutually orthogonal directions but vastly more nearly
orthogonal ones. That, plus the concentration of distances, is why
high-dimensional geometry does not match two-dimensional intuition and why
embedding spaces behave as they do.

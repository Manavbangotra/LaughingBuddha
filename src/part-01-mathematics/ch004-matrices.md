---
id: math-matrices
number: 4
part: I
tier: focused
status: reviewed
requires: [math-vectors]
provides: [matrix, matrix-multiplication, transpose, identity-matrix,
           matrix-inverse, linear-map, rank, broadcasting]
citations: [strang2010, deisenroth2020]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Read a matrix as a function rather than a table, and state the mapping
   $\R^{n} \to \R^{m}$ it represents.
2. Multiply matrices, predict the output shape before computing, and explain
   why the inner dimensions must agree.
3. Explain why matrix multiplication is associative but not commutative, and
   what each fact implies for neural networks.
4. Interpret a matrix-vector product in three equivalent ways: as dot products,
   as a linear combination of columns, and as a transformation of space.
5. Use the transpose and its reversal rule $(\mat{A}\mat{B})\T = \mat{B}\T\mat{A}\T$.
6. Determine when a matrix is invertible, and explain what non-invertibility
   means about lost information.
7. Define rank, compute it for small matrices, and explain why low rank is
   useful rather than merely deficient.
8. Apply NumPy broadcasting rules correctly and recognise the bugs they hide.

## 2. Why This Matters

A neural network is a stack of matrix multiplications with nonlinearities
between them. That is not a simplification for beginners — it is what the code
does. Strip away the framework and a transformer layer is six or seven matrix
products, an elementwise function, and a normalisation.

This has a direct consequence for how you should think about model size and
speed. When people say a model has 70 billion parameters, nearly all of those
parameters are entries of matrices. When they say training needs a GPU, it is
because GPUs multiply matrices thousands of times faster than CPUs. When a model
runs out of memory, it is usually because an intermediate matrix product did not
fit. Matrix multiplication is the unit of computation in this field, in the same
way the instruction is the unit of computation in ordinary programming.

There is also a conceptual payoff, and it is the one this chapter is really
about. Seeing a matrix as a *table of numbers* gets you through the arithmetic.
Seeing it as a *function that transforms space* is what makes eigenvectors,
PCA, LoRA, and the QK circuit inside attention comprehensible rather than
memorised. The shift from the first view to the second is the single most
valuable thing in this chapter.

## 3. Prerequisites

{{ch:math-vectors}} for vectors, dot products, linear combinations, span and
linear independence. {{ch:math-notation}} for summation notation.

## 4. Intuitive Explanation

### 4.1 A matrix is a function

A {{term:matrix}} is a rectangular grid of numbers:

$$
\mat{A} = \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix}
$$

The grid view is how you store it. The useful view is that $\mat{A}$ is a
*machine that eats vectors and produces vectors*. Feed it $[1, 1]\T$ and it
returns $[2, 3]\T$. Feed it $[5, 0]\T$ and it returns $[10, 0]\T$. This
particular matrix stretches everything by 2 horizontally and 3 vertically.

Once you adopt this view, questions that seemed arbitrary become natural. *Why
must the inner dimensions match in a product?* Because you cannot feed a
three-dimensional vector into a machine that expects two. *Why is multiplication
not commutative?* Because rotating then stretching is not the same as stretching
then rotating. *What is an eigenvector?* A direction the machine does not turn.

{#tbl:matrix-actions caption="Small matrices and what they do to the plane. Reading matrices as actions rather than as tables is the shift this chapter is built around."}

| Matrix | Action on $\R^{2}$ |
|---|---|
| $\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$ | nothing — the identity |
| $\begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix}$ | uniform scaling by 2 |
| $\begin{bmatrix} 2 & 0 \\ 0 & 0.5 \end{bmatrix}$ | stretch horizontally, squash vertically |
| $\begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$ | rotate 90° anticlockwise |
| $\begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$ | shear — slide the top sideways |
| $\begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}$ | project onto the horizontal axis |
| $\begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}$ | collapse the plane onto a single line |

The last two are worth pausing on. They destroy information: after projecting
onto the horizontal axis, the vertical component is gone and no operation can
recover it. Such matrices have no inverse, and that is what non-invertibility
*means* — not a technical obstruction, but a statement that the transformation
threw something away.

### 4.2 Matrix-vector products, three ways

For $\mat{A} \in \R^{m \times n}$ and $\vec{x} \in \R^{n}$, the product
$\mat{A}\vec{x}$ can be read three ways. All give the same answer; each is
useful in different places.

**As dot products (the row view).** Entry $i$ of the output is the dot product
of row $i$ of $\mat{A}$ with $\vec{x}$. This is how you compute by hand, and how
a neural network layer is usually described: each output neuron takes a weighted
sum of all inputs.

**As a linear combination of columns (the column view).** The output is
$x_1$ times column 1, plus $x_2$ times column 2, and so on. This view is the
more illuminating one: it says the reachable outputs are exactly the span of the
columns, which is what {{term:rank}} measures.

**As a transformation (the geometric view).** $\mat{A}$ moves the point
$\vec{x}$ somewhere else. This is the view that makes {{ch:math-eigen}} possible.

Worked, with $\mat{A} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$ and
$\vec{x} = [5, 6]\T$:

- Row view: $[1 \cdot 5 + 2 \cdot 6,\; 3 \cdot 5 + 4 \cdot 6]\T = [17, 39]\T$.
- Column view: $5\begin{bmatrix}1\\3\end{bmatrix} + 6\begin{bmatrix}2\\4\end{bmatrix}
  = \begin{bmatrix}5\\15\end{bmatrix} + \begin{bmatrix}12\\24\end{bmatrix}
  = \begin{bmatrix}17\\39\end{bmatrix}$.

Same answer, different story.

### 4.3 Matrix-matrix products are composition

Multiplying two matrices builds the single matrix that does what applying both
in turn would do:

$$
(\mat{A}\mat{B})\vec{x} = \mat{A}(\mat{B}\vec{x})
$$ (eq:matmul-composition)

Apply $\mat{B}$ first, then $\mat{A}$. This is exactly the composition of
{{ch:math-functions}}, and it reads right to left for the same reason.

It also explains, at a stroke, why a neural network needs nonlinearities. Two
stacked linear layers compute $\mat{A}(\mat{B}\vec{x}) = (\mat{A}\mat{B})\vec{x}$
— a single matrix. Ten stacked linear layers are also a single matrix. Without
something nonlinear between them, depth buys nothing at all
({{ch:dl-activations}}).

## 5. Formal Explanation

### 5.1 Definitions

A matrix $\mat{A} \in \R^{m \times n}$ has $m$ rows and $n$ columns, with entry
$A_{ij}$ in row $i$, column $j$. It represents a {{term:linear-map}}
$f: \R^{n} \to \R^{m}$, meaning

$$
f(a\vec{x} + b\vec{y}) = a\,f(\vec{x}) + b\,f(\vec{y})
$$ (eq:linearity)

Every linear map between finite-dimensional spaces is a matrix, and every matrix
is a linear map. The correspondence is exact, which is why the two words are
used interchangeably.

> IMPORTANT: The shape is written rows-by-columns, and the map goes the *other*
> way: an $m \times n$ matrix maps $\R^{n} \to \R^{m}$. This ordering catches
> everyone at least once. The way to remember it is that $\mat{A}\vec{x}$
> requires $\vec{x}$ to have as many entries as $\mat{A}$ has columns.

### 5.2 Matrix multiplication

For $\mat{A} \in \R^{m \times n}$ and $\mat{B} \in \R^{n \times p}$, the product
$\mat{C} = \mat{A}\mat{B} \in \R^{m \times p}$ has entries

$$
C_{ij} = \sum_{k=1}^{n} A_{ik}B_{kj}
$$ (eq:matmul)

Every entry of the result is a dot product: row $i$ of $\mat{A}$ against column
$j$ of $\mat{B}$. The inner dimensions must agree — $\mat{A}$'s columns must
equal $\mat{B}$'s rows — because that is the length of the vectors being dotted.

The shape rule is worth internalising as a picture:

$$
(m \times \underline{n}) \cdot (\underline{n} \times p) \;\to\; (m \times p)
$$

The underlined inner dimensions must match and then vanish; the outer ones
survive.

Properties:

$$
(\mat{A}\mat{B})\mat{C} = \mat{A}(\mat{B}\mat{C})
\qquad\text{(associative)}
$$ (eq:matmul-associative)

$$
\mat{A}(\mat{B} + \mat{C}) = \mat{A}\mat{B} + \mat{A}\mat{C}
\qquad\text{(distributive)}
$$ (eq:matmul-distributive)

$$
\mat{A}\mat{B} \neq \mat{B}\mat{A}
\qquad\text{in general (not commutative)}
$$ (eq:matmul-noncommutative)

{{eq:matmul-noncommutative}} is not a technicality. Often $\mat{B}\mat{A}$ is
not even defined — the shapes forbid it. When both products exist they are
usually different matrices, because the order in which you transform space
matters.

> PRODUCTION TIP: {{eq:matmul-associative}} has real consequences for cost.
> Computing $\mat{A}\mat{B}\mat{C}$ with $\mat{A} \in \R^{1000 \times 5}$,
> $\mat{B} \in \R^{5 \times 1000}$, $\mat{C} \in \R^{1000 \times 5}$ costs about
> $10^{7}$ operations as $(\mat{A}\mat{B})\mat{C}$ but only about $5 \times
> 10^{4}$ as $\mat{A}(\mat{B}\mat{C})$ — a factor of 200 for the same answer.
> This is exactly the trick that makes LoRA cheap ({{ch:ft-lora}}).

### 5.3 Transpose

The {{term:transpose}} $\mat{A}\T$ swaps rows and columns:
$(\mat{A}\T)_{ij} = A_{ji}$. An $m \times n$ matrix becomes $n \times m$.

$$
(\mat{A}\T)\T = \mat{A}, \qquad
(\mat{A} + \mat{B})\T = \mat{A}\T + \mat{B}\T, \qquad
(\mat{A}\mat{B})\T = \mat{B}\T\mat{A}\T
$$ (eq:transpose-rules)

The last rule — the order reverses — is the one to memorise. It follows directly
from {{eq:matmul}} and it appears constantly in gradient derivations, including
the attention backward pass in {{ch:tf-scaled-dot-product}}.

A matrix is **symmetric** if $\mat{A}\T = \mat{A}$. Covariance matrices
({{ch:math-covariance}}) and Hessians ({{ch:math-derivatives}}) are symmetric,
and symmetry buys real structure: {{ch:math-eigen}} shows such matrices always
have a full set of orthogonal eigenvectors.

### 5.4 Identity, inverse, and rank

The {{term:identity-matrix}} $\mat{I}_{n}$ has ones on the diagonal, zeros
elsewhere, and satisfies $\mat{A}\mat{I} = \mat{I}\mat{A} = \mat{A}$. It is the
transformation that does nothing.

The {{term:matrix-inverse}} $\mat{A}\inv$, when it exists, satisfies

$$
\mat{A}\mat{A}\inv = \mat{A}\inv\mat{A} = \mat{I}
$$ (eq:inverse-def)

It exists only for square matrices, and only for those that lose no information.
Formally, $\mat{A} \in \R^{n \times n}$ is invertible iff its columns are
linearly independent, iff its rank is $n$, iff its determinant is nonzero, iff
it has no zero singular value ({{ch:math-eigen}}). These are all the same
condition wearing different clothes.

The {{term:rank}} of a matrix is the dimension of its column span — how many
genuinely independent directions its outputs can reach. Equivalently, and
non-obviously, it equals the dimension of its *row* span; the two are always the
same number. For $\mat{A} \in \R^{m \times n}$,
$\rank(\mat{A}) \le \min(m, n)$, and the matrix is **full rank** when equality
holds and **rank-deficient** otherwise.

Rank is the concept that carries the most weight later in this book. Low rank
means the transformation squeezes its input into a thin slice of the output
space, and *that is often exactly what you want*:

- PCA keeps the top-$k$ directions of variance and discards the rest
  ({{ch:ml-pca}}).
- LoRA constrains a weight update to rank $r \ll d$, cutting trainable
  parameters by orders of magnitude ({{ch:ft-lora}}).
- An attention head's QK circuit is rank-limited by $d_k$, which is what makes
  running many heads affordable ({{ch:tf-multi-head}}).

> NOTE: Determinants get one sentence in this book. The determinant is the
> factor by which a transformation scales volume, and it is zero exactly when
> the transformation collapses space. That is the whole of what you need; every
> computational use of determinants in machine learning has been superseded by
> decompositions, which are both more informative and better conditioned.

### 5.5 Broadcasting

{{term:broadcasting}} is not mathematics — it is a convention numerical
libraries use so that arrays of different shapes can be combined without
explicit copying. It is included here because more beginner bugs come from
broadcasting than from any actual matrix algebra.

The rule: align shapes from the right. Each pair of dimensions must either be
equal, or one of them must be 1, in which case it is logically repeated.

$$
(3, 4) + (4,) \to (3, 4) \qquad \text{row vector added to every row}
$$
$$
(3, 4) + (3, 1) \to (3, 4) \qquad \text{column vector added to every column}
$$
$$
(3, 4) + (3,) \to \text{error} \qquad \text{4 and 3 are incompatible}
$$

The danger is not the error case — errors are helpful. The danger is the
*silent* case, where a shape mistake broadcasts into something valid but wrong.
{{sec:7-implementation}} shows one.

## 6. Mathematical Foundation

### 6.1 Why the transpose reverses order

$(\mat{A}\mat{B})\T = \mat{B}\T\mat{A}\T$ is used so often, and so often
misremembered, that it is worth deriving.

Take entry $(i, j)$ of the left side. By the definition of transpose, that is
entry $(j, i)$ of $\mat{A}\mat{B}$:

$$
\big((\mat{A}\mat{B})\T\big)_{ij} = (\mat{A}\mat{B})_{ji} = \sum_{k} A_{jk}B_{ki}
$$

Now entry $(i, j)$ of the right side, using {{eq:matmul}} and then the
definition of transpose twice:

$$
(\mat{B}\T\mat{A}\T)_{ij} = \sum_{k} (\mat{B}\T)_{ik}(\mat{A}\T)_{kj}
                           = \sum_{k} B_{ki}A_{jk}
$$

The two sums have identical terms — $A_{jk}B_{ki}$ and $B_{ki}A_{jk}$ differ
only in the order of two scalars — so the matrices are equal. The order reverses
because the transpose swaps which index is being summed against.

### 6.2 A worked product

Compute $\mat{A}\mat{B}$ for

$$
\mat{A} = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix},
\qquad
\mat{B} = \begin{bmatrix} 7 & 8 \\ 9 & 10 \\ 11 & 12 \end{bmatrix}
$$ (eq:worked-matrices)

Shapes first: $(2 \times \underline{3})(\underline{3} \times 2) \to (2 \times 2)$.
The inner 3s match, so the product exists and has four entries.

$$
C_{11} = 1(7) + 2(9) + 3(11) = 7 + 18 + 33 = 58
$$
$$
C_{12} = 1(8) + 2(10) + 3(12) = 8 + 20 + 36 = 64
$$
$$
C_{21} = 4(7) + 5(9) + 6(11) = 28 + 45 + 66 = 139
$$
$$
C_{22} = 4(8) + 5(10) + 6(12) = 32 + 50 + 72 = 154
$$

$$
\mat{A}\mat{B} = \begin{bmatrix} 58 & 64 \\ 139 & 154 \end{bmatrix}
$$ (eq:worked-product)

Now note that $\mat{B}\mat{A}$ is $(3 \times 2)(2 \times 3) \to (3 \times 3)$ —
a different shape entirely. The two products are not merely unequal; they are
not the same kind of object.

### 6.3 Rank, computed

Consider

$$
\mat{M} = \begin{bmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \\ 1 & 1 & 1 \end{bmatrix}
$$ (eq:rank-example)

Row 2 is exactly twice row 1, so it contributes no new direction. Rows 1 and 3
are not multiples of each other, so they contribute two. The rank is 2.

Geometrically: $\mat{M}$ maps $\R^{3}$ into a two-dimensional plane inside
$\R^{3}$. One whole dimension of input is annihilated. There is a nonzero vector
$\vec{v}$ with $\mat{M}\vec{v} = \vec{0}$ — the null space is nontrivial — and
consequently $\mat{M}$ has no inverse: two different inputs produce the same
output, so the map cannot be undone.

> MATH NOTE: The rank-nullity theorem states this precisely: for
> $\mat{A} \in \R^{m \times n}$, $\rank(\mat{A}) + \dim(\text{null}(\mat{A})) = n$.
> Every dimension of the input is either preserved as an independent output
> direction or destroyed. Here $2 + 1 = 3$. This is the cleanest way to see that
> information loss and non-invertibility are the same phenomenon.

### 6.4 A neural network layer, in matrix form

Putting it together. A fully connected layer applied to a batch of $B$ examples,
each with $d_{\text{in}}$ features:

$$
\mat{H} = \phi\big(\mat{X}\mat{W} + \vec{b}\big)
$$ (eq:dense-layer)

with $\mat{X} \in \R^{B \times d_{\text{in}}}$,
$\mat{W} \in \R^{d_{\text{in}} \times d_{\text{out}}}$,
$\vec{b} \in \R^{d_{\text{out}}}$ broadcast across rows, $\phi$ an elementwise
nonlinearity, and $\mat{H} \in \R^{B \times d_{\text{out}}}$.

Every idea in this chapter appears in that one line: the shape rule determines
what fits; the row view says each output feature is a weighted sum of inputs;
the column view says the output is a linear combination of $\mat{W}$'s columns;
broadcasting adds the bias to every row; and $\phi$ is there precisely because
{{eq:matmul-composition}} would otherwise collapse the whole network into a
single matrix.

The parameter count is $d_{\text{in}} \times d_{\text{out}} + d_{\text{out}}$,
and the cost of the forward pass is about $2 B\, d_{\text{in}} d_{\text{out}}$
floating-point operations. Those two formulas, applied repeatedly, are how model
size and training cost are estimated throughout {{part:23}}.

## 7. Implementation

```python {tier=A name=matrices-numpy}
"""Matrices as transformations, the three readings of a product, rank,
associativity as a cost decision, and the broadcasting bug that hides.
"""
import numpy as np

A = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0]])          # (2, 3) : maps R^3 -> R^2
B = np.array([[7.0, 8.0],
              [9.0, 10.0],
              [11.0, 12.0]])             # (3, 2) : maps R^2 -> R^3

print("A", A.shape, " B", B.shape)
print("A @ B  ->", (A @ B).shape, "\n", A @ B)
print("B @ A  ->", (B @ A).shape, " (a different object entirely)")
assert np.array_equal(A @ B, [[58, 64], [139, 154]])   # eq. 4.10

# --- the three readings of a matrix-vector product --------------------------
M = np.array([[1.0, 2.0], [3.0, 4.0]])
x = np.array([5.0, 6.0])

row_view = np.array([M[0] @ x, M[1] @ x])              # dot products
col_view = x[0] * M[:, 0] + x[1] * M[:, 1]             # linear combination
assert np.allclose(row_view, col_view) and np.allclose(row_view, M @ x)
print(f"\nrow view {row_view} == column view {col_view} == M @ x {M @ x}")

# --- eq. 4.11: the transpose reverses order ---------------------------------
assert np.allclose((A @ B).T, B.T @ A.T)
print("(A B)^T == B^T A^T verified")

# --- matrices as actions on the plane (table 4.1) ---------------------------
actions = {
    "identity":    np.array([[1.0, 0.0], [0.0, 1.0]]),
    "scale x2":    np.array([[2.0, 0.0], [0.0, 2.0]]),
    "rotate 90":   np.array([[0.0, -1.0], [1.0, 0.0]]),
    "shear":       np.array([[1.0, 1.0], [0.0, 1.0]]),
    "project x":   np.array([[1.0, 0.0], [0.0, 0.0]]),
    "collapse":    np.array([[1.0, 2.0], [2.0, 4.0]]),
}
v = np.array([1.0, 1.0])
print(f"\n{'action':<12} {'M @ [1,1]':<18} {'rank':>5} {'invertible':>11}")
for name, Mx in actions.items():
    r = np.linalg.matrix_rank(Mx)
    print(f"{name:<12} {str(Mx @ v):<18} {r:>5} {str(r == 2):>11}")

# --- rank and information loss (eq. 4.13) -----------------------------------
Mr = np.array([[1.0, 2.0, 3.0],
               [2.0, 4.0, 6.0],       # exactly 2x row 1 — no new direction
               [1.0, 1.0, 1.0]])
print(f"\nrank of the 3x3 example: {np.linalg.matrix_rank(Mr)} (not 3)")

# Two different inputs map to the same output, so the map cannot be inverted.
null_vec = np.linalg.svd(Mr)[2][-1]          # last right-singular vector
print(f"a null-space direction: {np.round(null_vec, 4)}")
print(f"M @ that direction    : {np.round(Mr @ null_vec, 12)}  <- zero")
u = np.array([1.0, 1.0, 1.0])
print(f"M @ u == M @ (u + null): "
      f"{np.allclose(Mr @ u, Mr @ (u + null_vec))}  <- information destroyed")

# --- eq. 4.7: associativity is a cost decision ------------------------------
rng = np.random.default_rng(0)
P = rng.normal(size=(1000, 5))
Q = rng.normal(size=(5, 1000))
R = rng.normal(size=(1000, 5))
left = (P @ Q) @ R            # builds a 1000x1000 intermediate
right = P @ (Q @ R)           # builds a 5x5 intermediate
assert np.allclose(left, right)
cost_left = 1000 * 5 * 1000 + 1000 * 1000 * 5
cost_right = 5 * 1000 * 5 + 1000 * 5 * 5
print(f"\nsame result, different cost: (PQ)R ~ {cost_left:,} ops, "
      f"P(QR) ~ {cost_right:,} ops  ({cost_left // cost_right}x)")
print("This is exactly the trick that makes LoRA cheap (Part XIV).")

# --- stacked linear layers collapse to one matrix ---------------------------
W1, W2, W3 = rng.normal(size=(4, 6)), rng.normal(size=(6, 5)), rng.normal(size=(5, 3))
xin = rng.normal(size=(2, 4))
stacked = ((xin @ W1) @ W2) @ W3
single = xin @ (W1 @ W2 @ W3)
assert np.allclose(stacked, single)
print("\nthree linear layers == one matrix — which is why nonlinearities exist")

# --- broadcasting: the helpful error and the silent bug ---------------------
X = np.arange(12, dtype=float).reshape(3, 4)
print("\n(3,4) + (4,)  ->", (X + np.ones(4)).shape, " row vector added to each row")
print("(3,4) + (3,1) ->", (X + np.ones((3, 1))).shape, " column added to each column")
try:
    X + np.ones(3)
except ValueError as exc:
    print("(3,4) + (3,)  -> ValueError:", str(exc)[:60])

# The dangerous case: no error, wrong answer. Intending a per-row offset but
# passing a (3,1) instead of a (1,3) produces a valid array of the wrong shape.
per_column = np.array([10.0, 20.0, 30.0, 40.0])      # intended: one per column
wrong = np.array([10.0, 20.0, 30.0]).reshape(3, 1)   # typo: one per row
print(f"\nintended shape {(X + per_column).shape}, "
      f"typo also valid with shape {(X + wrong).shape} — no error raised")
print("Broadcasting turns a shape mistake into a plausible wrong answer.")
print("Assert your shapes; do not rely on an exception.")
```

## 8. Practical Example

The clearest place to see matrices doing real work is a batched neural network
layer, where the shape reasoning has to be exactly right.

```python {tier=A name=dense-layer}
"""A fully connected layer, forward pass only, with shapes asserted at every
step and the parameter and FLOP counts computed from the formulas of section 6.4.
"""
import numpy as np

rng = np.random.default_rng(0)

B, d_in, d_out = 8, 128, 64          # batch, input width, output width

X = rng.normal(size=(B, d_in))                       # a batch of examples
W = rng.normal(size=(d_in, d_out)) * np.sqrt(2.0 / d_in)   # He initialisation
b = np.zeros(d_out)


def relu(z):
    return np.maximum(z, 0.0)


def dense_forward(X, W, b, activation=relu):
    """eq. 4.15 : H = phi(XW + b)."""
    assert X.ndim == 2 and W.ndim == 2
    assert X.shape[1] == W.shape[0], (
        f"inner dims must match: X is {X.shape}, W is {W.shape}")
    Z = X @ W + b                     # (B, d_in) @ (d_in, d_out) -> (B, d_out)
    assert Z.shape == (X.shape[0], W.shape[1])
    return activation(Z)


H = dense_forward(X, W, b)
print(f"X {X.shape}  @  W {W.shape}  +  b {b.shape}   ->   H {H.shape}")

params = d_in * d_out + d_out
flops = 2 * B * d_in * d_out
print(f"\nparameters : {params:,}   (d_in*d_out + d_out)")
print(f"FLOPs      : {flops:,}   (2 * B * d_in * d_out)")

# Scale that up to a realistic transformer feed-forward block and the numbers
# stop being abstract.
d_model, d_ff, seq, batch = 4096, 11008, 2048, 1
ffn_params = 2 * d_model * d_ff
ffn_flops = 2 * 2 * batch * seq * d_model * d_ff
print(f"\none transformer FFN block at d_model={d_model}, d_ff={d_ff}:")
print(f"  parameters : {ffn_params/1e6:,.1f} M")
print(f"  FLOPs for {seq} tokens : {ffn_flops/1e9:,.1f} G")
print(f"  across 32 layers : {32*ffn_params/1e9:,.2f} B parameters")

# --- the column view, made visible ------------------------------------------
# Output feature j is a weighted sum of inputs; equivalently the output vector
# is a linear combination of W's columns with the input as coefficients.
x0 = X[0]
combo = sum(x0[i] * W[i] for i in range(d_in))
assert np.allclose(combo, x0 @ W)
print("\ncolumn view confirmed: x @ W == sum_i x_i * W[i, :]")
```

The parameter and FLOP formulas printed there are the ones used throughout
{{part:23}} to size models and estimate serving cost. They are not
approximations for teaching — they are the actual arithmetic, and knowing them
is what lets you predict whether a model will fit on a given accelerator before
you try.

## 9. Common Mistakes

**Getting the shape rule backwards.** An $m \times n$ matrix maps
$\R^{n} \to \R^{m}$. Rows-by-columns, but the map goes right to left.

**Assuming $\mat{A}\mat{B} = \mat{B}\mat{A}$.** Usually false, and often the
second product is not even defined. Order is meaning.

**Forgetting to reverse in the transpose rule.** $(\mat{A}\mat{B})\T$ is
$\mat{B}\T\mat{A}\T$, not $\mat{A}\T\mat{B}\T$. This error propagates silently
through gradient derivations.

**Using `A * B` in NumPy for matrix multiplication.** `*` is elementwise. Use
`@`. When the shapes happen to be compatible for both, the bug is silent.

**Relying on broadcasting to catch shape errors.** It frequently will not.
Assert your shapes explicitly, as the code above does.

**Treating rank deficiency as a numerical nuisance.** It is a statement about
information: a rank-deficient map has thrown something away, and no algorithm
recovers it. Relatedly, `np.linalg.matrix_rank` uses a tolerance, so a matrix
that is near-singular in exact arithmetic may report full rank — check the
singular values ({{ch:math-eigen}}) rather than trusting the integer.

**Inverting a matrix to solve a linear system.** Never compute $\mat{A}\inv
\vec{b}$; use `np.linalg.solve(A, b)`. It is faster and substantially more
accurate, for reasons {{ch:math-eigen}} explains via the condition number.

**Ignoring the order of chained products.** {{eq:matmul-associative}} says the
answer is the same; it does not say the cost is. A two-hundred-fold difference
is available for free.

## 10. Connection to Previous Chapters

{{ch:math-vectors}} supplied the dot product, which {{eq:matmul}} performs in
bulk, and the notions of span and linear independence, which are what
{{term:rank}} counts. {{ch:math-functions}} supplied composition, which
{{eq:matmul-composition}} shows is matrix multiplication.

Forward: {{ch:math-norms}} measures the size of the transformations defined
here. {{ch:math-eigen}} finds the directions a matrix does not rotate and
factorises every matrix into interpretable pieces, which is where rank becomes
genuinely powerful. {{ch:math-derivatives}} arranges partial derivatives into
the Jacobian, a matrix, and the chain rule becomes matrix multiplication.
{{ch:math-covariance}} builds the covariance matrix.

Beyond Part I: {{ch:dl-forward}} is {{eq:dense-layer}} repeated;
{{ch:ml-pca}} and {{ch:ft-lora}} are both applications of low rank; and
{{ch:tf-scaled-dot-product}} is built entirely from the products in this
chapter, with its QK circuit being an instance of {{eq:matmul-associative}}
regrouped.

## 11. Exercises

**Beginner**

1. For $\mat{A} \in \R^{3 \times 5}$ and $\mat{B} \in \R^{5 \times 2}$, give the
   shapes of $\mat{A}\mat{B}$, $\mat{B}\T\mat{A}\T$, and $\mat{A}\T$. Which of
   $\mat{B}\mat{A}$ and $\mat{A}\mat{B}$ is defined?
2. Compute $\begin{bmatrix}1&2\\3&4\end{bmatrix}\begin{bmatrix}5\\6\end{bmatrix}$
   using the row view, then again using the column view.
3. Write down the $2 \times 2$ matrix that reflects the plane across the
   horizontal axis, and verify it on $[3, 5]\T$.
4. What is the rank of $\begin{bmatrix}1&2\\2&4\end{bmatrix}$? Is it invertible?
5. Give the shape resulting from broadcasting $(5, 3)$ with $(3,)$, with
   $(5, 1)$, and with $(5,)$.

**Intermediate**

6. Verify {{eq:worked-product}} by hand, then compute $\mat{B}\mat{A}$ and state
   its shape.
7. Show that $(\mat{A}\mat{B}\mat{C})\T = \mat{C}\T\mat{B}\T\mat{A}\T$.
8. A dense layer maps 512 features to 2048. How many parameters, including
   biases? How many FLOPs for a batch of 32?
9. Explain, using {{eq:matmul-composition}}, why a network of purely linear
   layers has the same expressive power as a single linear layer.
10. Give a $3 \times 3$ matrix of rank 1 and describe geometrically what it does
    to $\R^{3}$.
11. For $\mat{A} \in \R^{1000 \times 10}$, $\mat{B} \in \R^{10 \times 1000}$ and
    $\vec{x} \in \R^{1000}$, compare the cost of $(\mat{A}\mat{B})\vec{x}$ with
    $\mat{A}(\mat{B}\vec{x})$.

**Advanced**

12. Prove that $\rank(\mat{A}\mat{B}) \le \min(\rank(\mat{A}), \rank(\mat{B}))$.
13. Prove that a square matrix is invertible if and only if its columns are
    linearly independent.
14. Show that the product of two symmetric matrices is symmetric only when they
    commute.
15. Show that any rank-$r$ matrix $\mat{A} \in \R^{m \times n}$ can be written
    as $\mat{U}\mat{V}$ with $\mat{U} \in \R^{m \times r}$ and
    $\mat{V} \in \R^{r \times n}$. Count the parameters of each form and state
    when the factored form is cheaper. This is the LoRA argument.

**Implementation**

16. Implement matrix multiplication with three nested loops, verify it against
    `@` on random matrices, then time both at $256 \times 256$ and explain the
    gap.
17. Write `safe_matmul(A, B)` that raises an informative error naming both
    shapes when the inner dimensions disagree.
18. Construct a broadcasting bug that produces a plausible wrong answer without
    raising, then write the assertion that would have caught it.
19. Empirically compare `np.linalg.inv(A) @ b` with `np.linalg.solve(A, b)` on a
    deliberately ill-conditioned system, measuring both accuracy and time.

**Reasoning**

20. Why do GPUs help so much more with deep learning than with, say, sorting?
    Answer in terms of what {{eq:matmul}} allows to happen simultaneously.
21. LoRA freezes a weight matrix $\mat{W}$ and learns a low-rank update
    $\Delta\mat{W} = \mat{B}\mat{A}$. Using rank, explain what is being assumed
    about how fine-tuning changes a model, and when that assumption might fail.

## 12. Chapter Summary

A matrix is a linear map, not a table. An $m \times n$ matrix transforms
$\R^{n} \to \R^{m}$, and reading matrices as actions on space — stretching,
rotating, shearing, projecting — is what makes the rest of linear algebra
comprehensible rather than procedural.

A matrix-vector product has three equivalent readings: dot products of rows with
the input, a linear combination of columns weighted by the input, and a
transformation of a point. The column view is the one that explains rank.

Matrix multiplication is composition: $\mat{A}\mat{B}$ applies $\mat{B}$ first.
Inner dimensions must match and then vanish. The operation is associative and
distributive but not commutative, and associativity is a genuine optimisation
opportunity — regrouping a chain can change its cost by orders of magnitude,
which is the arithmetic behind LoRA.

The transpose swaps indices and reverses the order of a product:
$(\mat{A}\mat{B})\T = \mat{B}\T\mat{A}\T$. Symmetric matrices, which include
covariance matrices and Hessians, have special structure exploited in
{{ch:math-eigen}}.

Rank is the number of independent directions a matrix's output can span.
Full-rank square matrices are invertible; rank-deficient ones have destroyed
information and cannot be undone. Low rank is frequently a feature rather than a
defect, and it is the shared idea behind PCA, LoRA, and multi-head attention's
per-head dimension.

Because stacked linear layers collapse into a single matrix, depth is worthless
without a nonlinearity between layers — which is the entire reason activation
functions exist.

Broadcasting lets arrays of different shapes combine, aligning from the right.
Its errors are helpful; its silent successes are not. Assert shapes explicitly.

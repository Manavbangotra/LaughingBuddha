---
id: math-eigen
number: 6
part: I
tier: focused
status: reviewed
requires: [math-matrices, math-norms]
provides: [eigenvector, eigenvalue, diagonalisation,
           singular-value-decomposition, low-rank-approximation, spectral-norm,
           condition-number]
citations: [strang2010, deisenroth2020]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Define an eigenvector and eigenvalue, and explain what each means
   geometrically.
2. Compute eigenvalues and eigenvectors of a $2 \times 2$ matrix by hand.
3. State the spectral theorem for symmetric matrices and say why symmetry
   matters.
4. State the singular value decomposition, explain the geometric meaning of
   each factor, and say why it exists for every matrix while diagonalisation
   does not.
5. Truncate an SVD to obtain the best rank-$k$ approximation, and state the
   Eckart-Young theorem that guarantees it is best.
6. Compute the spectral norm and condition number from the singular values, and
   explain what a large condition number costs you.
7. Explain how PCA, LoRA, and embedding compression are all the same operation.

## 2. Why This Matters

This is the chapter where linear algebra stops being arithmetic and starts
paying for itself.

The central observation is that most matrices arising from real data are
*approximately low rank*. A user-item ratings matrix with a million users and a
hundred thousand items is not a hundred billion independent numbers; it is
better described by a few hundred latent factors. A batch of image embeddings
does not fill its 768-dimensional space uniformly; it concentrates near a much
lower-dimensional surface. The weight update learned during fine-tuning does not
touch all directions equally; it concentrates in a few.

The singular value decomposition is the tool that finds this structure, and it
does so optimally — a claim that can be stated and proved precisely rather than
merely asserted. Once you have it, a surprising number of apparently separate
techniques turn out to be the same thing:

- **PCA** is the SVD of centred data ({{ch:ml-pca}}).
- **Low-rank matrix factorisation** for recommenders is a truncated SVD
  ({{ch:ds-recsys}}).
- **LoRA** constrains a fine-tuning update to low rank, on the empirical
  hypothesis that the update is intrinsically low-rank anyway
  ({{ch:ft-lora}}).
- **Embedding compression** truncates the SVD of an embedding matrix.
- **Whitening** and preconditioning rescale by the singular values.

Eigenvalues also explain a class of training failures directly. Whether
gradients explode or vanish through a deep network is a question about repeated
multiplication by a matrix, and repeated multiplication is governed entirely by
eigenvalues ({{ch:dl-initialization}}). Whether gradient descent zigzags or
descends cleanly is a question about the condition number
({{ch:math-optimization}}).

## 3. Prerequisites

{{ch:math-matrices}} for matrices as linear maps, matrix multiplication,
transpose, rank and invertibility. {{ch:math-norms}} for the $L_2$, Frobenius
and spectral norms. {{ch:math-vectors}} for orthogonality and linear
independence.

## 4. Intuitive Explanation

### 4.1 Directions a transformation leaves alone

{{ch:math-matrices}} established that a matrix is a transformation of space. Most
vectors get both rotated and stretched by it. But for most matrices there exist
a few special directions that are *not* rotated at all — only stretched or
shrunk.

Those directions are the {{term:eigenvector}}s, and the stretch factor along
each is its {{term:eigenvalue}}.

$$
\mat{A}\vec{v} = \lambda\vec{v}
$$ (eq:eigen-def)

Read this as: applying $\mat{A}$ to $\vec{v}$ has the same effect as multiplying
$\vec{v}$ by a number. The transformation acts, along this one direction, like
simple scaling.

Take $\mat{A} = \begin{bmatrix} 2 & 0 \\ 0 & 3\end{bmatrix}$. The horizontal
direction $[1, 0]\T$ is stretched by 2 and not turned: it is an eigenvector with
eigenvalue 2. The vertical direction is an eigenvector with eigenvalue 3. Any
other direction, say $[1, 1]\T$, becomes $[2, 3]\T$ — which points somewhere
different, so it is not an eigenvector.

### 4.2 Why anyone cares: repeated application

The reason eigenvectors matter is that they make repeated application trivial to
reason about.

Applying $\mat{A}$ a hundred times to a general vector is a hundred matrix
products, and the result is hard to characterise. But if $\vec{v}$ is an
eigenvector, then

$$
\mat{A}^{100}\vec{v} = \lambda^{100}\vec{v}
$$ (eq:eigen-power)

and the whole question reduces to one scalar raised to a power. Since any vector
can be decomposed into eigenvector components (when a full set exists), the
behaviour of $\mat{A}^{k}$ on *anything* is governed by the eigenvalues.

The consequence is stark. If the largest eigenvalue exceeds 1, repeated
application blows up. If all eigenvalues are below 1, everything decays to zero.
This is not an abstract observation: it is exactly why gradients explode or
vanish in deep networks ({{ch:dl-initialization}}), why recurrent networks
struggle with long sequences ({{ch:dl-rnns}}), and why weight initialisation
schemes are designed around keeping the relevant spectrum near 1.

### 4.3 The SVD: every matrix is rotate, stretch, rotate

Eigenvectors have two serious limitations: they are only defined for square
matrices, and even some square matrices lack a full set.

The {{term:singular-value-decomposition}} has neither problem. It says that
**every** matrix — square or not, invertible or not — can be written as

$$
\mat{A} = \mat{U}\mat{\Sigma}\mat{V}\T
$$ (eq:svd)

where $\mat{U}$ and $\mat{V}$ are rotations (more precisely, orthogonal
matrices) and $\mat{\Sigma}$ is diagonal with non-negative entries.

Geometrically: **every linear transformation, without exception, is a rotation,
then an axis-aligned stretch, then another rotation.** That is a remarkable
statement. Shears, projections, reflections, and every squashing and skewing
transformation you can construct all decompose this way.

```mermaid {#fig:svd-geometry caption="The SVD as three geometric steps. Any matrix, however complicated it looks, does exactly this: rotate the input onto a set of orthogonal axes, scale along those axes, then rotate into the output space."}
graph LR
  X["input<br/>unit circle"] -->|"V<sup>T</sup><br/>rotate"| R1["aligned<br/>circle"]
  R1 -->|"Σ<br/>scale each axis<br/>by σ<sub>i</sub>"| E["ellipse<br/>axes = σ<sub>1</sub>, σ<sub>2</sub>"]
  E -->|"U<br/>rotate"| Y["output<br/>ellipse in place"]
```

The **singular values** $\sigma_1 \ge \sigma_2 \ge \cdots \ge 0$ on the diagonal
of $\mat{\Sigma}$ are the stretch factors, ordered largest first. That ordering
is what makes the SVD useful for compression: the first few singular values
capture the directions in which the matrix does most of its work, and the tail
often contributes almost nothing.

### 4.4 Low rank is compression

If the singular values decay quickly, you can throw the small ones away.

Keeping only the top $k$ gives a rank-$k$ matrix $\mat{A}_k$ that approximates
$\mat{A}$. Instead of storing $mn$ numbers you store $k(m + n + 1)$ — and for
$k \ll \min(m, n)$ that is a large saving.

The Eckart-Young theorem, stated in {{sec:5-formal-explanation}}, says something
stronger and genuinely surprising: this truncation is the *best possible* rank-$k$
approximation, in both the Frobenius and spectral norms. Not a good heuristic —
provably optimal. No cleverer method exists.

## 5. Formal Explanation

### 5.1 Eigenvalues and eigenvectors

For a square $\mat{A} \in \R^{n \times n}$, a nonzero $\vec{v}$ satisfying
{{eq:eigen-def}} is an eigenvector with eigenvalue $\lambda$. Rearranging:

$$
(\mat{A} - \lambda\mat{I})\vec{v} = \vec{0}
$$ (eq:eigen-rearranged)

For a nonzero $\vec{v}$ to exist, $\mat{A} - \lambda\mat{I}$ must be singular,
which happens exactly when

$$
\det(\mat{A} - \lambda\mat{I}) = 0
$$ (eq:characteristic)

This is the **characteristic equation**, a degree-$n$ polynomial in $\lambda$
whose roots are the eigenvalues.

> NOTE: {{eq:characteristic}} is the *definition* route, and it is how
> eigenvalues are computed by hand for $2 \times 2$ and $3 \times 3$ matrices.
> It is emphatically **not** how they are computed numerically — polynomial
> root-finding is unstable, and real implementations use iterative methods such
> as QR. This is the one place the determinant appears in this book, and it
> appears only to define, not to compute.

Eigenvalues may be complex even for real matrices (a rotation matrix has no real
eigenvector, which makes sense — a rotation turns everything). Eigenvectors are
determined only up to scale, since scaling $\vec{v}$ in {{eq:eigen-def}} changes
nothing; by convention they are normalised to unit length.

### 5.2 The spectral theorem

Symmetric matrices are special, and since covariance matrices
({{ch:math-covariance}}) and Hessians ({{ch:math-derivatives}}) are symmetric,
the special case covers much of what this book needs.

**Spectral theorem.** If $\mat{A} \in \R^{n \times n}$ is symmetric, then all its
eigenvalues are real, and it has $n$ mutually orthogonal eigenvectors. It can
therefore be written

$$
\mat{A} = \mat{Q}\mat{\Lambda}\mat{Q}\T
$$ (eq:spectral-decomposition)

with $\mat{Q}$ orthogonal ($\mat{Q}\T\mat{Q} = \mat{I}$) and $\mat{\Lambda}$
diagonal holding the eigenvalues.

This is {{term:diagonalisation}}, and the orthogonality of $\mat{Q}$ is what
makes it well behaved: $\mat{Q}\inv = \mat{Q}\T$, so the inverse is free and
numerically stable.

A symmetric matrix is **positive semi-definite** if all its eigenvalues are
$\ge 0$, equivalently if $\vec{x}\T\mat{A}\vec{x} \ge 0$ for every $\vec{x}$.
Covariance matrices always are, which is why variance along any direction is
never negative.

### 5.3 The singular value decomposition

For any $\mat{A} \in \R^{m \times n}$:

$$
\mat{A} = \mat{U}\mat{\Sigma}\mat{V}\T
$$

with $\mat{U} \in \R^{m \times m}$ orthogonal, $\mat{V} \in \R^{n \times n}$
orthogonal, and $\mat{\Sigma} \in \R^{m \times n}$ diagonal with
$\sigma_1 \ge \sigma_2 \ge \cdots \ge 0$.

The columns of $\mat{U}$ are the **left singular vectors**, those of $\mat{V}$
the **right singular vectors**. The relationship to eigenvalues is direct:

$$
\mat{A}\T\mat{A} = \mat{V}\mat{\Sigma}\T\mat{\Sigma}\mat{V}\T,
\qquad
\mat{A}\mat{A}\T = \mat{U}\mat{\Sigma}\mat{\Sigma}\T\mat{U}\T
$$ (eq:svd-eigen-relation)

So $\mat{V}$ holds the eigenvectors of $\mat{A}\T\mat{A}$, $\mat{U}$ holds those
of $\mat{A}\mat{A}\T$, and the singular values are the square roots of the
shared nonzero eigenvalues. Both $\mat{A}\T\mat{A}$ and $\mat{A}\mat{A}\T$ are
symmetric and positive semi-definite, so the spectral theorem applies to each —
which is, in outline, why the SVD always exists.

Key facts that follow immediately:

$$
\rank(\mat{A}) = \#\{i : \sigma_i > 0\}
$$ (eq:rank-from-svd)

$$
\norm{\mat{A}}_2 = \sigma_1, \qquad
\norm{\mat{A}}_F = \sqrt{\textstyle\sum_i \sigma_i^{2}}
$$ (eq:norms-from-svd)

{{eq:rank-from-svd}} is the numerically meaningful definition of rank: count the
singular values that are not negligible. Exact zero is not a robust test in
floating point.

### 5.4 Low-rank approximation and Eckart-Young

Write the SVD as a sum of rank-1 pieces:

$$
\mat{A} = \sum_{i=1}^{r} \sigma_i\,\vec{u}_i\vec{v}_i\T
$$ (eq:svd-outer-sum)

Each term is a rank-1 matrix scaled by its singular value, and the terms are
ordered by importance. Truncating after $k$ terms gives

$$
\mat{A}_k = \sum_{i=1}^{k} \sigma_i\,\vec{u}_i\vec{v}_i\T
$$ (eq:truncated-svd)

**Eckart-Young theorem.** $\mat{A}_k$ minimises $\norm{\mat{A} - \mat{B}}$ over
all matrices $\mat{B}$ of rank at most $k$, simultaneously for the Frobenius and
spectral norms, with errors

$$
\norm{\mat{A} - \mat{A}_k}_2 = \sigma_{k+1},
\qquad
\norm{\mat{A} - \mat{A}_k}_F = \sqrt{\textstyle\sum_{i>k}\sigma_i^{2}}
$$ (eq:eckart-young-error)

This is a strong statement and it is worth being clear about how strong. Among
*all* rank-$k$ matrices — including any produced by an arbitrarily clever
algorithm — none approximates $\mat{A}$ better than the truncated SVD. And
{{eq:eckart-young-error}} tells you the error in advance, from the singular
values you already computed, before deciding where to truncate.

### 5.5 Condition number

The {{term:condition-number}} of a matrix is

$$
\kappa(\mat{A}) = \frac{\sigma_{\max}}{\sigma_{\min}}
$$ (eq:condition-number)

It measures how much the matrix distorts space unevenly: $\kappa = 1$ means a
uniform scaling in all directions, while a large $\kappa$ means the matrix
stretches enormously in one direction and barely at all in another.

Its practical meaning is error amplification. When solving $\mat{A}\vec{x} =
\vec{b}$, a relative error $\epsilon$ in $\vec{b}$ can become a relative error of
up to $\kappa(\mat{A})\epsilon$ in $\vec{x}$. With $\kappa = 10^{8}$ and
double-precision inputs accurate to about $10^{-16}$, you can lose half your
significant digits.

An ill-conditioned problem is *intrinsically* hard. No algorithm fixes it,
because the difficulty is a property of the problem rather than the method.
{{ch:math-optimization}} shows the optimisation consequence: gradient descent on
a badly conditioned objective zigzags, and the number of iterations it needs
scales with $\kappa$.

## 6. Mathematical Foundation

### 6.1 Eigenvalues of a $2 \times 2$ matrix, by hand

Take

$$
\mat{A} = \begin{bmatrix} 4 & 1 \\ 2 & 3 \end{bmatrix}
$$ (eq:eigen-example)

Form $\mat{A} - \lambda\mat{I}$ and set its determinant to zero:

$$
\det\begin{bmatrix} 4-\lambda & 1 \\ 2 & 3-\lambda \end{bmatrix}
  = (4-\lambda)(3-\lambda) - (1)(2) = 0
$$

Expanding: $12 - 7\lambda + \lambda^{2} - 2 = \lambda^{2} - 7\lambda + 10 = 0$,
which factors as $(\lambda - 5)(\lambda - 2) = 0$. So $\lambda_1 = 5$ and
$\lambda_2 = 2$.

For $\lambda_1 = 5$, solve $(\mat{A} - 5\mat{I})\vec{v} = \vec{0}$:

$$
\begin{bmatrix} -1 & 1 \\ 2 & -2 \end{bmatrix}\begin{bmatrix} v_1 \\ v_2\end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}
$$

Both rows give $v_1 = v_2$, so $\vec{v}_1 = [1, 1]\T$ (up to scale). Check:
$\mat{A}[1,1]\T = [5, 5]\T = 5[1,1]\T$. Correct.

For $\lambda_2 = 2$, solve $(\mat{A} - 2\mat{I})\vec{v} = \vec{0}$:

$$
\begin{bmatrix} 2 & 1 \\ 2 & 1 \end{bmatrix}\vec{v} = \vec{0}
$$

giving $2v_1 + v_2 = 0$, so $\vec{v}_2 = [1, -2]\T$. Check:
$\mat{A}[1,-2]\T = [2, -4]\T = 2[1,-2]\T$. Correct.

Two useful checks, valid for any square matrix: the eigenvalues sum to the trace
($5 + 2 = 7 = 4 + 3$ ✓) and multiply to the determinant
($5 \times 2 = 10 = 4(3) - 1(2)$ ✓).

Note that these eigenvectors are *not* orthogonal — $[1,1]\T \cdot [1,-2]\T =
-1 \neq 0$ — because $\mat{A}$ is not symmetric. That is exactly what the
spectral theorem guarantees you get back when the matrix is symmetric.

### 6.2 A worked SVD

Take the rank-1 matrix

$$
\mat{B} = \begin{bmatrix} 3 & 0 \\ 4 & 0 \end{bmatrix}
$$ (eq:svd-example)

Compute $\mat{B}\T\mat{B} = \begin{bmatrix} 25 & 0 \\ 0 & 0\end{bmatrix}$, whose
eigenvalues are 25 and 0. The singular values are their square roots:
$\sigma_1 = 5$, $\sigma_2 = 0$.

One nonzero singular value means rank 1, which matches: the second column is
zero, so the map sends all of $\R^{2}$ onto a single line.

The right singular vectors are the eigenvectors of $\mat{B}\T\mat{B}$, namely
$[1,0]\T$ and $[0,1]\T$. The first left singular vector is
$\vec{u}_1 = \mat{B}\vec{v}_1/\sigma_1 = [3,4]\T/5 = [0.6, 0.8]\T$.

So $\mat{B}$ takes everything, projects onto the horizontal axis, scales by 5,
and rotates onto the direction $[0.6, 0.8]\T$. The condition number is
$5/0 = \infty$ — the matrix is singular, as expected.

### 6.3 Why the truncated SVD is optimal, in outline

A complete proof of Eckart-Young is longer than this chapter warrants, but the
structure of the argument is worth seeing.

Orthogonal matrices preserve both the Frobenius and spectral norms, so
multiplying by $\mat{U}\T$ on the left and $\mat{V}$ on the right changes
nothing about the size of an error. That reduces the problem to approximating
the *diagonal* matrix $\mat{\Sigma}$ by a rank-$k$ matrix.

For a diagonal matrix, the best rank-$k$ approximation is evidently the one
keeping the $k$ largest diagonal entries and zeroing the rest — any other choice
discards a larger entry and keeps a smaller one, increasing the error. Rotating
back gives {{eq:truncated-svd}}.

The genuine content is the first step: that the problem can be rotated into a
frame where the answer is obvious. That is what the SVD provides, and it is why
it turns up whenever an optimality claim about low rank is needed.

### 6.4 Singular value spectra tell you about your data

The list of singular values — the **spectrum** — is diagnostic in its own right.

**Fast decay** means the data is genuinely low-dimensional and compresses well.
A ratings matrix whose spectrum drops sharply after 50 values says there are
about 50 meaningful latent factors.

**Slow decay** means the data really does occupy all its dimensions, and
truncation will cost you.

**A sharp cliff** — a large gap between $\sigma_k$ and $\sigma_{k+1}$ — is the
clearest signal available that the intrinsic dimension is $k$. Where no cliff
exists, the choice of $k$ is a judgement call, usually made by fixing a target
fraction of retained energy:

$$
\text{energy retained} = \frac{\sum_{i \le k}\sigma_i^{2}}{\sum_i \sigma_i^{2}}
$$ (eq:energy-retained)

The square appears because the Frobenius norm is a root-sum-square, so squared
singular values are the additively meaningful quantity. Choosing $k$ to retain
95% or 99% of the energy is the standard convention in PCA ({{ch:ml-pca}}).

## 7. Implementation

```python {tier=A name=eigen-and-svd}
"""Eigenvalues, the spectral theorem, the SVD, and Eckart-Young optimality.

The optimality claim is tested against random rank-k competitors rather than
taken on trust.
"""
import numpy as np

# --- eigenvalues by hand vs numerically (section 6.1) -----------------------
A = np.array([[4.0, 1.0],
              [2.0, 3.0]])
vals, vecs = np.linalg.eig(A)
order = np.argsort(-vals)
vals, vecs = vals[order], vecs[:, order]
print("eigenvalues :", np.round(vals, 6), " (hand-computed: 5 and 2)")
print("trace check :", np.trace(A), "==", vals.sum().real)
print("det check   :", round(np.linalg.det(A), 6), "==", round(np.prod(vals).real, 6))

for i in range(2):
    v = vecs[:, i]
    print(f"  A v{i} == lambda{i} v{i}: "
          f"{np.allclose(A @ v, vals[i] * v)}   v{i} = {np.round(v, 4)}")

print(f"eigenvectors orthogonal? {abs(vecs[:,0] @ vecs[:,1]) < 1e-9} "
      f"(A is not symmetric, so no reason they should be)")

# --- the spectral theorem: symmetry buys orthogonality ----------------------
rng = np.random.default_rng(0)
M = rng.normal(size=(5, 5))
S = M + M.T                                  # any M + M^T is symmetric
w, Q = np.linalg.eigh(S)                     # eigh: for symmetric matrices
print(f"\nsymmetric matrix: eigenvalues all real? {np.isrealobj(w)}")
print(f"eigenvectors orthonormal? {np.allclose(Q.T @ Q, np.eye(5))}")
assert np.allclose(Q @ np.diag(w) @ Q.T, S)  # eq. 6.5
print("Q Lambda Q^T reconstructs S exactly (eq. 6.5)")

# Positive semi-definiteness: A^T A always has non-negative eigenvalues.
G = M.T @ M
print(f"eigenvalues of M^T M all >= 0? {np.all(np.linalg.eigvalsh(G) > -1e-10)}")

# --- eq. 6.6: the SVD of a non-square, rank-deficient matrix ----------------
B = np.array([[3.0, 0.0],
              [4.0, 0.0]])
U, s, Vt = np.linalg.svd(B)
print(f"\nsingular values of B: {np.round(s, 6)}  (hand-computed: 5 and 0)")
print(f"rank from SVD: {np.sum(s > 1e-10)}  | numpy rank: {np.linalg.matrix_rank(B)}")
print(f"u_1 = {np.round(U[:, 0], 4)}  (hand-computed [0.6, 0.8], up to sign)")
assert np.allclose(U @ np.diag(s) @ Vt, B)

# --- eq. 6.9: norms come straight from the singular values ------------------
C = rng.normal(size=(7, 4))
sc = np.linalg.svd(C, compute_uv=False)
print(f"\nspectral norm : {np.linalg.norm(C, 2):.6f} == sigma_1 = {sc[0]:.6f}")
print(f"frobenius norm: {np.linalg.norm(C, 'fro'):.6f} == "
      f"sqrt(sum sigma^2) = {np.sqrt((sc**2).sum()):.6f}")
print(f"condition number: {np.linalg.cond(C):.4f} == "
      f"sigma_max/sigma_min = {sc[0]/sc[-1]:.4f}")

# --- Eckart-Young, tested against random competitors ------------------------
# Build a matrix that is genuinely low-rank plus noise.
m, n, true_rank = 60, 40, 6
L = rng.normal(size=(m, true_rank)) @ rng.normal(size=(true_rank, n))
A2 = L + 0.15 * rng.normal(size=(m, n))

U2, s2, Vt2 = np.linalg.svd(A2, full_matrices=False)

print(f"\n{'k':>3} {'SVD error':>12} {'predicted':>12} {'best random':>13} "
      f"{'energy':>8}")
for k in (1, 3, 6, 10, 20):
    Ak = (U2[:, :k] * s2[:k]) @ Vt2[:k]
    err = np.linalg.norm(A2 - Ak, "fro")
    predicted = np.sqrt((s2[k:] ** 2).sum())        # eq. 6.12

    # 200 random rank-k competitors, each least-squares fitted to give them the
    # best possible chance.
    best_random = np.inf
    for _ in range(200):
        R = rng.normal(size=(m, k))
        coef, *_ = np.linalg.lstsq(R, A2, rcond=None)
        best_random = min(best_random, np.linalg.norm(A2 - R @ coef, "fro"))

    energy = (s2[:k] ** 2).sum() / (s2 ** 2).sum()
    print(f"{k:>3} {err:>12.5f} {predicted:>12.5f} {best_random:>13.5f} "
          f"{energy:>7.1%}")
    assert np.isclose(err, predicted)
    assert err <= best_random + 1e-9      # no competitor ever beats the SVD

print("\nThe SVD error matches eq. 6.12 exactly, and no random rank-k")
print("competitor ever beats it — Eckart-Young, demonstrated.")

# --- the spectrum reveals the intrinsic dimension ---------------------------
print(f"\nsingular values of a rank-{true_rank} matrix plus noise:")
print(" ", np.round(s2[:12], 3))
gaps = s2[:-1] / s2[1:]
print(f"largest consecutive ratio at index {int(np.argmax(gaps[:15])) + 1} "
      f"(ratio {gaps[:15].max():.2f}) — the cliff marks the true rank")

# --- compression arithmetic --------------------------------------------------
print(f"\nstoring a {m}x{n} matrix:")
print(f"  full            : {m*n:,} numbers")
for k in (3, 6, 10):
    print(f"  rank-{k:<2} truncation: {k*(m+n+1):,} numbers "
          f"({k*(m+n+1)/(m*n):.0%} of full), "
          f"error {np.sqrt((s2[k:]**2).sum())/np.linalg.norm(A2,'fro'):.1%}")
```

## 8. Practical Example

Low-rank approximation is how a recommender system works, and the arithmetic is
small enough to see in full.

```python {tier=A name=low-rank-recommender}
"""A ratings matrix, factorised. This is the SVD doing the job that
recommender systems, PCA, and LoRA all ask of it.
"""
import numpy as np

rng = np.random.default_rng(11)

# 300 users, 120 items, but only 4 latent taste factors generate the ratings.
n_users, n_items, n_factors = 300, 120, 4
user_taste = rng.normal(size=(n_users, n_factors))
item_profile = rng.normal(size=(n_factors, n_items))
ratings = user_taste @ item_profile + 0.4 * rng.normal(size=(n_users, n_items))

U, s, Vt = np.linalg.svd(ratings, full_matrices=False)

print("top 10 singular values:")
print(" ", np.round(s[:10], 2))
print(f"\nratio sigma_4 / sigma_5 = {s[3]/s[4]:.2f}  <- the cliff at the true "
      f"number of factors")

energy = np.cumsum(s ** 2) / np.sum(s ** 2)
for k in (1, 2, 3, 4, 5, 10):
    print(f"  rank {k:>2}: {energy[k-1]:6.1%} of energy retained")

# Reconstruct with the true number of factors and compare to the CLEAN signal,
# not to the noisy observations — the point is that truncation removes noise.
k = 4
approx = (U[:, :k] * s[:k]) @ Vt[:k]
clean = user_taste @ item_profile

err_vs_noisy = np.linalg.norm(ratings - approx, "fro") / np.linalg.norm(ratings, "fro")
err_vs_clean = np.linalg.norm(clean - approx, "fro") / np.linalg.norm(clean, "fro")
err_noisy_vs_clean = np.linalg.norm(ratings - clean, "fro") / np.linalg.norm(clean, "fro")

print(f"\nrank-{k} reconstruction:")
print(f"  distance from the noisy observations : {err_vs_noisy:.1%}")
print(f"  distance from the clean signal       : {err_vs_clean:.1%}")
print(f"  distance of raw data from the signal : {err_noisy_vs_clean:.1%}")
print("\nThe approximation is CLOSER to the truth than the raw data is:")
print("discarding the small singular values discarded mostly noise.")

print(f"\nstorage: {n_users*n_items:,} numbers -> "
      f"{k*(n_users+n_items+1):,} ({k*(n_users+n_items+1)/(n_users*n_items):.1%})")
```

That last result is the one worth remembering: the rank-4 reconstruction is
*closer to the underlying signal than the observed data is*. Truncation removed
more noise than signal. This is the mechanism behind denoising by PCA, and it
is also the honest justification for LoRA — if the thing you are trying to
estimate is genuinely low-rank, constraining your estimate to be low-rank is not
a compromise but an improvement.

> RESEARCH NOTE: The LoRA hypothesis ({{ch:ft-lora}}) is that the *weight update*
> during fine-tuning has low intrinsic rank, even though the weights themselves
> do not. That is an empirical claim about how fine-tuning changes models rather
> than a mathematical necessity, and it holds well for adaptation to a narrow
> task and less well when a model must acquire genuinely new capabilities.
> {{maturity:ESTABLISHED}} for the technique; the underlying claim about
> intrinsic dimension is {{maturity:EMERGING}}.

## 9. Common Mistakes

**Computing eigenvalues via the characteristic polynomial in code.** Numerically
unstable. Use `np.linalg.eig`, or `np.linalg.eigh` for symmetric matrices — the
latter is both faster and more accurate, and guarantees real eigenvalues.

**Expecting real eigenvalues from a non-symmetric matrix.** Rotations have
complex ones. If your code assumes real, assert symmetry first.

**Using `eig` where `eigh` applies.** For a symmetric matrix, `eig` may return
eigenvalues in arbitrary order with tiny imaginary parts from rounding. `eigh`
returns them sorted and real.

**Treating rank as an exact integer test.** In floating point, "zero" singular
values are around $10^{-16}$ times the largest. Use a tolerance, and look at the
actual spectrum rather than trusting a single integer.

**Forgetting that singular vectors have a sign ambiguity.** $\vec{u}_i$ and
$\vec{v}_i$ can both be negated without changing $\mat{A}$. Two correct SVD
implementations can disagree on signs; comparisons must account for it.

**Assuming the SVD requires a square matrix.** It does not. That is its main
advantage over diagonalisation.

**Choosing $k$ by intuition rather than by the spectrum.** Plot the singular
values. If there is a cliff, use it; if there is not, pick an energy target and
say so.

**Ignoring the condition number until something breaks.** A large $\kappa$ means
your problem is intrinsically sensitive. Check it before blaming the solver.

## 10. Connection to Previous Chapters

{{ch:math-matrices}} established matrices as transformations and defined rank
and invertibility; this chapter explains what rank means spectrally and gives
the numerically robust way to measure it. {{ch:math-norms}} defined the spectral
and Frobenius norms, and {{eq:norms-from-svd}} now identifies both in terms of
singular values. {{ch:math-vectors}} supplied orthogonality, which is what makes
$\mat{U}$ and $\mat{V}$ well behaved.

Forward: {{ch:math-covariance}} builds the covariance matrix, whose eigenvectors
are the principal directions of variation. {{ch:math-optimization}} uses the
condition number to explain why gradient descent zigzags, and the Hessian's
eigenvalues to classify stationary points.

Beyond Part I: {{ch:ml-pca}} is the SVD of centred data;
{{ch:ds-recsys}} is {{sec:8-practical-example}} at scale;
{{ch:ft-lora}} constrains a weight update to low rank;
{{ch:dl-initialization}} uses the spectrum to keep signals from exploding or
vanishing; and {{ch:tf-multi-head}} exploits the rank constraint that a
per-head dimension $d_k < d_{\text{model}}$ imposes.

## 11. Exercises

**Beginner**

1. Verify that $[1, 0]\T$ is an eigenvector of
   $\begin{bmatrix}2&0\\0&3\end{bmatrix}$ and give its eigenvalue.
2. Find the eigenvalues of $\begin{bmatrix}5&0\\0&-2\end{bmatrix}$ by
   inspection.
3. For a matrix with singular values $[10, 3, 0.1]$, give the rank, the spectral
   norm, the Frobenius norm and the condition number.
4. What is the condition number of the identity matrix? Of a singular matrix?
5. A $100 \times 50$ matrix is truncated to rank 5. How many numbers are stored,
   as a fraction of the original?

**Intermediate**

6. Compute the eigenvalues and eigenvectors of
   $\begin{bmatrix}3&1\\1&3\end{bmatrix}$ by hand. Verify they are orthogonal,
   and say why you should have expected that.
7. Verify the trace and determinant checks of {{sec:6-mathematical-foundation}}
   on {{eq:eigen-example}}.
8. Explain why a rotation matrix has no real eigenvectors, geometrically.
9. Given singular values $[8, 6, 2, 1, 0.5]$, compute the energy retained at
   $k = 2$ and $k = 3$, and the Frobenius error at each.
10. Why is the SVD defined for every matrix while diagonalisation is not?

**Advanced**

11. Prove that the eigenvalues of a symmetric real matrix are real.
12. Prove that eigenvectors of a symmetric matrix corresponding to distinct
    eigenvalues are orthogonal.
13. Derive {{eq:svd-eigen-relation}} from {{eq:svd}}, and use it to show
    $\sigma_i = \sqrt{\lambda_i(\mat{A}\T\mat{A})}$.
14. Show that $\norm{\mat{A}}_F^{2} = \sum_i \sigma_i^{2}$ using the fact that
    orthogonal matrices preserve the Frobenius norm.
15. Prove that for symmetric positive definite $\mat{A}$, the singular values
    equal the eigenvalues. What happens when $\mat{A}$ is symmetric but
    indefinite?

**Implementation**

16. Implement the power iteration method for the dominant eigenvector — repeatedly
    apply $\mat{A}$ and renormalise — and compare against `np.linalg.eig`.
    Investigate how convergence speed depends on $\sigma_1/\sigma_2$.
17. Compress a grayscale image with a truncated SVD at several values of $k$,
    plotting reconstruction error and storage against $k$. Identify the knee.
18. Reproduce the Eckart-Young test in {{sec:7-implementation}} but replace the
    random competitors with a rank-$k$ matrix found by gradient descent. Confirm
    it still cannot beat the SVD.
19. Generate matrices with condition numbers from $10^{1}$ to $10^{12}$, solve
    $\mat{A}\vec{x} = \vec{b}$ for each with a known solution, and plot the
    achieved accuracy against $\kappa$. Confirm the loss of digits scales as
    predicted.

**Reasoning**

20. A colleague reports that their embedding matrix has a slowly decaying
    spectrum with no cliff. What does that say about compressing it, and what
    would you advise?
21. LoRA assumes the fine-tuning update is low-rank. Describe a fine-tuning task
    for which you would expect this to hold well, and one for which you would
    expect it to fail, and justify both.

## 12. Chapter Summary

An eigenvector is a direction a matrix does not rotate, only scales; its
eigenvalue is the scale factor. Eigenvectors matter because they make repeated
application trivial — $\mat{A}^{k}\vec{v} = \lambda^{k}\vec{v}$ — which is why
eigenvalues govern whether gradients explode or vanish through depth.

The spectral theorem says symmetric matrices have real eigenvalues and a full set
of orthogonal eigenvectors, so they diagonalise as $\mat{Q}\mat{\Lambda}\mat{Q}\T$
with $\mat{Q}$ orthogonal. Covariance matrices and Hessians are symmetric, so
this case covers most of what the book needs.

The singular value decomposition writes *every* matrix as
$\mat{U}\mat{\Sigma}\mat{V}\T$ — rotate, scale along axes, rotate. It exists
unconditionally, unlike diagonalisation, and its singular values give the rank,
the spectral norm, the Frobenius norm and the condition number directly.

Truncating the SVD to the top $k$ singular values gives the best possible
rank-$k$ approximation in both the Frobenius and spectral norms — the
Eckart-Young theorem — with an error known in advance from the discarded
singular values. When data is genuinely low-rank plus noise, the truncation can
be closer to the underlying signal than the observed data is.

The condition number $\sigma_{\max}/\sigma_{\min}$ measures how unevenly a matrix
distorts space and bounds how much it amplifies error. A badly conditioned
problem is intrinsically hard; no algorithm repairs it.

PCA, low-rank recommender factorisation, embedding compression and LoRA are all
the same operation applied to different matrices. That unification is the reason
this chapter earns its place in a book about modern AI rather than a course in
linear algebra.

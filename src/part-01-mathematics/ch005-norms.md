---
id: math-norms
number: 5
part: I
tier: focused
status: reviewed
requires: [math-vectors]
provides: [norm, cosine-similarity, unit-vector, euclidean-distance,
           manhattan-distance]
citations: [deisenroth2020]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. State the four axioms a norm must satisfy and check whether a candidate
   function qualifies.
2. Compute the $L_1$, $L_2$ and $L_\infty$ norms of a vector and describe the
   unit ball of each.
3. Explain why $L_1$ regularisation produces sparse solutions and $L_2$ does
   not, in terms of the geometry of those unit balls.
4. Convert between a norm and a distance, and state when a distance is *not*
   induced by a norm.
5. Compute cosine similarity and explain precisely what it discards relative to
   the dot product.
6. Choose between Euclidean distance, cosine similarity and the dot product for
   a retrieval task, and justify the choice.
7. Explain the equivalence between cosine similarity and Euclidean distance on
   normalised vectors, and why vector databases exploit it.

## 2. Why This Matters

Once things are vectors, the immediate question is how to compare them — and
there is no single right answer. "How similar are these two documents" and "how
far is this prediction from the truth" are both questions about distance, but
they call for different measures, and choosing the wrong one degrades a system
in ways that are hard to attribute.

The choice appears everywhere in this book. A loss function is a distance
between prediction and target, and whether you pick $L_1$ or $L_2$ changes how
the model treats outliers ({{ch:ml-metrics}}). Regularisation is a penalty on
the size of the weights, and whether you pick $L_1$ or $L_2$ determines whether
you get a sparse model or a merely small one ({{ch:dl-regularization}}).
Retrieval ranks documents by similarity, and whether you pick cosine or dot
product determines whether long documents are systematically favoured
({{ch:emb-similarity}}). Gradient clipping bounds a norm ({{ch:dl-optimizers}}).
Vector databases build their indexes around a specific metric and cannot switch
after the fact ({{ch:emb-vector-db}}).

None of these are arbitrary defaults to be copied from a tutorial. Each is a
decision about what "close" should mean, and this chapter is where you learn to
make it deliberately.

## 3. Prerequisites

{{ch:math-vectors}} for vectors, dot products, and {{eq:dot-geometric}} relating
the dot product to the angle. {{ch:math-functions}} for absolute value and
powers.

## 4. Intuitive Explanation

### 4.1 Size is not one thing

How big is the vector $[3, 4]\T$?

If it is a displacement on a map, the answer is 5 — the straight-line distance,
by Pythagoras. If it is a journey through a city laid out on a grid, the answer
is 7 — you must travel three blocks then four. If it is a set of measurements
and you care about the worst one, the answer is 4 — the largest component.

All three answers are correct. They are the $L_2$, $L_1$ and $L_\infty$
{{term:norm}}s, and each is the right answer to a different question. There is no
fact of the matter about which is "the" size; there is only a choice about what
you are measuring.

### 4.2 The three that matter

{#tbl:norms caption="The three norms used throughout this book, on the example vector [3, −4]. Each answers a different question about size."}

| Norm | Formula | On $[3, -4]$ | Answers |
|---|---|---|---|
| $L_1$ (Manhattan) | $\sum_i \lvert x_i \rvert$ | $3 + 4 = 7$ | total magnitude across all components |
| $L_2$ (Euclidean) | $\sqrt{\sum_i x_i^{2}}$ | $\sqrt{9+16} = 5$ | straight-line length |
| $L_\infty$ (max) | $\max_i \lvert x_i \rvert$ | $4$ | the single largest component |

Always $\norm{\vec{x}}_\infty \le \norm{\vec{x}}_2 \le \norm{\vec{x}}_1$, which
the numbers above illustrate: $4 \le 5 \le 7$.

The differences come from how each treats a *concentration* of magnitude. The
vectors $[10, 0]\T$ and $[7.07, 7.07]\T$ have the same $L_2$ norm of 10, but
their $L_1$ norms are 10 and 14.14. The $L_1$ norm therefore prefers
concentration — it charges less for putting everything in one component — while
$L_2$ is indifferent to how magnitude is spread. That single observation is the
whole explanation of why $L_1$ regularisation produces sparse models, developed
properly in {{sec:6-mathematical-foundation}}.

### 4.3 Unit balls make the difference visible

The clearest way to see how norms differ is to draw the set of vectors whose
norm equals 1 — the **unit ball**.

```mermaid {#fig:unit-balls caption="Unit balls of the three norms in two dimensions. The corners of the L1 diamond lie exactly on the axes, which is where one coordinate is zero — the geometric origin of sparsity."}
graph TB
  subgraph L1["L1 : a diamond"]
    A1["corners at (±1,0) and (0,±1)<br/>— on the axes<br/>— pointed"]
  end
  subgraph L2["L2 : a circle"]
    A2["all directions equal<br/>— rotationally symmetric<br/>— smooth everywhere"]
  end
  subgraph LINF["L-infinity : a square"]
    A3["corners at (±1,±1)<br/>— off the axes<br/>— flat sides"]
  end
```

The $L_2$ ball is a circle: perfectly symmetric, no direction preferred. The
$L_1$ ball is a diamond whose corners sit *on the coordinate axes*, and a corner
on an axis is a point where one coordinate is exactly zero. The $L_\infty$ ball
is a square whose corners sit *off* the axes, where all coordinates are equally
large.

Those corners are not decoration. When you constrain a solution to lie inside a
ball and push it as far as you can in some direction, you tend to end up at a
corner — and for $L_1$, corners mean zeros. That is sparsity, and it is a
geometric fact rather than a statistical one.

### 4.4 Distance and similarity are different questions

A norm gives a distance: $d(\vec{x}, \vec{y}) = \norm{\vec{x} - \vec{y}}$. But
distance is not always what you want.

Take two documents about football, one a 100-word match report and the other a
5000-word tactical analysis. Their embeddings point in nearly the same
direction, but the second is far longer, so its vector is longer too. The
Euclidean distance between them is large. Yet in any sensible sense they are
similar.

{{term:cosine-similarity}} answers the other question. It measures the angle and
ignores the lengths entirely:

$$
\cos\theta = \frac{\vec{x}\T\vec{y}}{\norm{\vec{x}}\,\norm{\vec{y}}}
$$ (eq:cosine-similarity)

Same direction gives 1, perpendicular gives 0, opposite gives $-1$. Length is
divided out. Whether that is an improvement depends entirely on whether length
carries information you care about — and in embeddings it usually does not, so
cosine is usually the right default.

## 5. Formal Explanation

### 5.1 The axioms

A function $\norm{\cdot} : \R^{n} \to \R$ is a **norm** if for all
$\vec{x}, \vec{y}$ and all scalars $c$:

$$
\norm{\vec{x}} \ge 0 \qquad\text{(non-negativity)}
$$ (eq:norm-nonneg)

$$
\norm{\vec{x}} = 0 \iff \vec{x} = \vec{0} \qquad\text{(definiteness)}
$$ (eq:norm-definite)

$$
\norm{c\vec{x}} = \lvert c \rvert\,\norm{\vec{x}} \qquad\text{(homogeneity)}
$$ (eq:norm-homogeneous)

$$
\norm{\vec{x} + \vec{y}} \le \norm{\vec{x}} + \norm{\vec{y}} \qquad\text{(triangle inequality)}
$$ (eq:norm-triangle)

These are exactly the properties "size" must have to behave sensibly. Homogeneity
says doubling a vector doubles its size; the triangle inequality says a detour is
never shorter than going direct.

> NOTE: A common near-miss is the *squared* $L_2$ norm, $\norm{\vec{x}}_2^{2}$.
> It is not a norm — it violates homogeneity, since scaling by $c$ multiplies it
> by $c^{2}$. It is nonetheless used constantly as a loss and a penalty, because
> it is differentiable everywhere (unlike $\norm{\vec{x}}_2$ at the origin) and
> avoids a square root. When this book writes "the L2 penalty" it almost always
> means the squared norm, and says so where the distinction matters.

### 5.2 The $L_p$ family

For $p \ge 1$:

$$
\norm{\vec{x}}_p = \left(\sum_{i=1}^{n} \lvert x_i \rvert^{p}\right)^{1/p}
$$ (eq:lp-norm)

with $p = 1$ and $p = 2$ as above, and $L_\infty$ as the limit:

$$
\norm{\vec{x}}_\infty = \lim_{p \to \infty}\norm{\vec{x}}_p = \max_i \lvert x_i \rvert
$$ (eq:linf-norm)

The limit is worth understanding rather than accepting: as $p$ grows, the
largest term dominates the sum so completely that the others become negligible,
and taking the $p$-th root recovers exactly that largest term.

The $L_2$ norm connects to {{ch:math-vectors}} directly:

$$
\norm{\vec{x}}_2 = \sqrt{\vec{x}\T\vec{x}}
$$ (eq:l2-from-dot)

which is why {{eq:dot-positive}} — the positive-definiteness of the dot product
— was worth stating there.

> WARNING: "$L_0$" is widely used to mean the count of nonzero entries. It is
> **not** a norm: it violates homogeneity, since scaling a vector does not
> change how many entries are nonzero. It is also not convex, which is why
> minimising it is computationally hard and why $L_1$ is used as its tractable
> surrogate.

### 5.3 Matrix norms

Matrices have norms too, and two matter here.

The **Frobenius norm** treats the matrix as one long vector:

$$
\norm{\mat{A}}_F = \sqrt{\sum_{i,j} A_{ij}^{2}} = \sqrt{\tr(\mat{A}\T\mat{A})}
$$ (eq:frobenius)

The **spectral norm** is the largest factor by which the matrix can stretch any
vector:

$$
\norm{\mat{A}}_2 = \max_{\vec{x} \neq \vec{0}} \frac{\norm{\mat{A}\vec{x}}_2}{\norm{\vec{x}}_2}
$$ (eq:spectral-norm)

The Frobenius norm is what weight decay penalises. The spectral norm controls
how much a layer can amplify its input, which is why it appears in the analysis
of exploding gradients ({{ch:dl-initialization}}). {{ch:math-eigen}} shows that
the spectral norm is exactly the largest singular value.

### 5.4 Distances and metrics

A **metric** is a distance function $d(\vec{x}, \vec{y})$ satisfying
non-negativity, identity of indiscernibles ($d = 0$ iff the points coincide),
symmetry, and the triangle inequality.

Every norm induces a metric via $d(\vec{x}, \vec{y}) = \norm{\vec{x} - \vec{y}}$.
The converse fails: there are perfectly good metrics not arising from any norm.

This matters practically. **Cosine distance**, defined as
$1 - \cos\theta$, is *not* a metric — it violates the triangle inequality.
Vector database index structures that rely on metric properties for their
correctness guarantees therefore cannot use it directly, and instead exploit the
identity in {{sec:6-mathematical-foundation}} to convert the problem into a
Euclidean one ({{ch:emb-ann}}).

### 5.5 Normalisation

A {{term:unit-vector}} has norm 1. Any nonzero vector can be normalised:

$$
\hat{\vec{x}} = \frac{\vec{x}}{\norm{\vec{x}}_2}
$$ (eq:normalise)

This discards magnitude and keeps direction. On normalised vectors,
{{eq:cosine-similarity}} simplifies to a plain dot product, because both
denominators are 1 — which is the single most important practical fact in this
chapter, and the reason retrieval systems normalise at index time.

## 6. Mathematical Foundation

### 6.1 Cosine and Euclidean distance are the same thing on the unit sphere

For normalised $\vec{x}$ and $\vec{y}$, expand the squared Euclidean distance:

$$
\norm{\vec{x} - \vec{y}}_2^{2}
  = (\vec{x} - \vec{y})\T(\vec{x} - \vec{y})
  = \vec{x}\T\vec{x} - 2\vec{x}\T\vec{y} + \vec{y}\T\vec{y}
$$

Since both are unit vectors, $\vec{x}\T\vec{x} = \vec{y}\T\vec{y} = 1$, and
since they are normalised, $\vec{x}\T\vec{y} = \cos\theta$. So:

$$
\norm{\vec{x} - \vec{y}}_2^{2} = 2 - 2\cos\theta
$$ (eq:cosine-euclidean)

Euclidean distance is a strictly decreasing function of cosine similarity.
**Ranking by one is identical to ranking by the other**, provided the vectors are
normalised.

> IMPORTANT: {{eq:cosine-euclidean}} is why vector databases can offer "cosine
> similarity" while internally building a Euclidean index: normalise on
> insertion, and the two orderings coincide exactly. It also means that if your
> vectors are *not* normalised, choosing "cosine" versus "L2" in a vector
> database changes your results — and the change can be large. This is a
> configuration decision people make casually and then debug for a week
> ({{ch:emb-vector-db}}).

### 6.2 Why $L_1$ gives sparsity

This is the most important argument in the chapter, and it is entirely
geometric.

Consider minimising a loss $\Loss(\vec{w})$ subject to a budget on the size of
$\vec{w}$ — either $\norm{\vec{w}}_1 \le t$ or $\norm{\vec{w}}_2 \le t$. The
constrained optimum lies where the lowest reachable contour of $\Loss$ touches
the boundary of the feasible region.

For $L_2$, that boundary is a circle: smooth, with no distinguished points. A
contour arriving from a generic direction touches it at a generic point, where
both coordinates are typically nonzero. The solution is *small*, not sparse.

For $L_1$, the boundary is a diamond with corners lying exactly on the axes. The
corners stick out, so a contour arriving from a generic direction is
disproportionately likely to touch one — and at a corner, one coordinate is
exactly zero.

Concretely in two dimensions: the $L_1$ ball has four corners at $(\pm t, 0)$
and $(0, \pm t)$, each of which sets one coordinate to zero. In $n$ dimensions
the effect strengthens: the $L_1$ ball has $2n$ corners but $2^{n}$ faces, and
the low-dimensional faces — the ones where many coordinates are zero — occupy a
disproportionate share of the directions from which a contour can arrive.

This argument also explains why the effect requires the *non-smoothness*.
Sparsity comes from the corner, and the corner exists because $\lvert x \rvert$
is not differentiable at zero. Any smooth penalty, including $L_2$, gives
coefficients that shrink toward zero without reaching it.

### 6.3 A worked comparison

Take $\vec{x} = [3, 4]\T$ and $\vec{y} = [6, 8]\T$ — note that
$\vec{y} = 2\vec{x}$, so they point in exactly the same direction.

**Norms.** $\norm{\vec{x}}_2 = 5$, $\norm{\vec{y}}_2 = 10$.

**Euclidean distance.** $\norm{\vec{x} - \vec{y}}_2 = \norm{[-3,-4]\T}_2 = 5$.
By this measure they are as far apart as $\vec{x}$ is from the origin.

**Dot product.** $3(6) + 4(8) = 18 + 32 = 50$.

**Cosine similarity.** $50 / (5 \times 10) = 1.0$ — perfect similarity, since
they are parallel.

Three measures, three completely different verdicts about the same pair. If
these were document embeddings, Euclidean distance would call them dissimilar,
the raw dot product would give a high but uninterpretable score, and cosine
would correctly identify them as being about the same thing at different
lengths.

### 6.4 What each measure is blind to

{#tbl:similarity-choice caption="Choosing a similarity measure. The question is always what you want the measure to ignore."}

| Measure | Range | Sensitive to | Blind to | Use when |
|---|---|---|---|---|
| Dot product | $(-\infty, \infty)$ | direction and both magnitudes | nothing | magnitude carries real signal, e.g. learned attention scores |
| Cosine | $[-1, 1]$ | direction only | both magnitudes | comparing text embeddings of differing length |
| Euclidean | $[0, \infty)$ | absolute positions | nothing | coordinates are physically meaningful, e.g. clustering |
| Manhattan | $[0, \infty)$ | per-axis differences | interaction between axes | features are independent and equally scaled |

The recurring question is *what should this measure ignore*. Cosine ignores
magnitude; that is a feature when magnitude reflects document length and a bug
when it reflects confidence.

## 7. Implementation

```python {tier=A name=norms-and-distances}
"""Norms, distances, and similarity — with the chapter's claims checked.

Includes the L1-versus-L2 sparsity effect, demonstrated rather than asserted.
"""
import numpy as np

x = np.array([3.0, -4.0])

print(f"{'norm':<14} {'value':>8}")
print(f"{'L1':<14} {np.linalg.norm(x, 1):>8.4f}")
print(f"{'L2':<14} {np.linalg.norm(x, 2):>8.4f}")
print(f"{'L-infinity':<14} {np.linalg.norm(x, np.inf):>8.4f}")
assert np.linalg.norm(x, np.inf) <= np.linalg.norm(x, 2) <= np.linalg.norm(x, 1)

# --- the norm axioms, checked on random vectors (eqs. 5.2-5.5) --------------
rng = np.random.default_rng(0)
for p in (1, 2, np.inf):
    for _ in range(500):
        a, b = rng.normal(size=6), rng.normal(size=6)
        c = rng.normal()
        assert np.linalg.norm(a, p) >= 0
        assert np.isclose(np.linalg.norm(c * a, p),
                          abs(c) * np.linalg.norm(a, p))            # homogeneity
        assert (np.linalg.norm(a + b, p)
                <= np.linalg.norm(a, p) + np.linalg.norm(b, p) + 1e-9)  # triangle
print("\nnorm axioms hold for p = 1, 2, inf on 1500 random cases")

# The squared L2 "norm" is not one: it fails homogeneity.
a = rng.normal(size=6)
print(f"||2a||^2 = {np.sum((2*a)**2):.4f} but 2*||a||^2 = {2*np.sum(a**2):.4f}"
      "  <- squared L2 is not a norm")

# --- concentration: why L1 and L2 disagree ----------------------------------
spread = np.array([7.071, 7.071])
concentrated = np.array([10.0, 0.0])
print(f"\n{'vector':<22} {'L1':>8} {'L2':>8}")
for name, v in (("spread [7.07, 7.07]", spread), ("concentrated [10, 0]", concentrated)):
    print(f"{name:<22} {np.linalg.norm(v,1):>8.3f} {np.linalg.norm(v,2):>8.3f}")
print("Equal L2, different L1: L1 charges less for concentrating magnitude.")

# --- eq. 5.11: cosine and Euclidean coincide on normalised vectors ----------
def cosine(a, b):
    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))


u, v = rng.normal(size=64), rng.normal(size=64)
un, vn = u / np.linalg.norm(u), v / np.linalg.norm(v)
lhs = np.linalg.norm(un - vn) ** 2
rhs = 2 - 2 * cosine(u, v)
print(f"\n||u_hat - v_hat||^2 = {lhs:.10f}")
print(f"2 - 2 cos(theta)    = {rhs:.10f}   <- eq. 5.11")
assert np.isclose(lhs, rhs)

# Consequently, ranking by one equals ranking by the other — but only once
# normalised.
docs = rng.normal(size=(200, 64))
q = rng.normal(size=64)
docs_n = docs / np.linalg.norm(docs, axis=1, keepdims=True)
q_n = q / np.linalg.norm(q)
rank_cos = np.argsort(-(docs_n @ q_n))
rank_l2 = np.argsort(np.linalg.norm(docs_n - q_n, axis=1))
assert np.array_equal(rank_cos, rank_l2)
print("normalised: cosine ranking == Euclidean ranking (identical order)")

rank_cos_raw = np.argsort(-(docs @ q / (np.linalg.norm(docs, axis=1) * np.linalg.norm(q))))
rank_dot_raw = np.argsort(-(docs @ q))
agree = np.mean(rank_cos_raw[:10] == rank_dot_raw[:10])
print(f"UNnormalised: cosine and dot-product top-10 agree on only "
      f"{agree:.0%} of positions")

# --- the three measures disagree about the same pair (section 6.3) ----------
a2, b2 = np.array([3.0, 4.0]), np.array([6.0, 8.0])
print(f"\nx = {a2}, y = {b2}  (y = 2x, so identical direction)")
print(f"  euclidean distance : {np.linalg.norm(a2 - b2):.3f}   <- 'far apart'")
print(f"  dot product        : {a2 @ b2:.3f}  <- large but uninterpretable")
print(f"  cosine similarity  : {cosine(a2, b2):.3f}   <- 'identical'")

# --- section 6.2: L1 produces sparsity, L2 does not -------------------------
# Fit y = Xw with a penalty, by plain gradient descent, and count exact zeros.
n, d = 200, 60
X = rng.normal(size=(n, d))
w_true = np.zeros(d)
w_true[:5] = [3.0, -2.0, 1.5, 4.0, -1.0]     # only 5 of 60 features matter
y = X @ w_true + 0.1 * rng.normal(size=n)


def fit(penalty, lam, steps=4000, lr=2e-3):
    w = np.zeros(d)
    for _ in range(steps):
        grad = X.T @ (X @ w - y) / n
        if penalty == "l2":
            w -= lr * (grad + 2 * lam * w)
        else:
            # Proximal step: gradient on the loss, then soft-threshold. The
            # threshold is what can set a coefficient to EXACTLY zero; a plain
            # subgradient step only ever approaches zero.
            w -= lr * grad
            w = np.sign(w) * np.maximum(np.abs(w) - lr * lam, 0.0)
    return w


for penalty, lam in (("l2", 0.05), ("l1", 0.30)):
    w = fit(penalty, lam)
    exact_zeros = int(np.sum(np.abs(w) == 0.0))
    tiny = int(np.sum(np.abs(w) < 1e-3))
    print(f"\n{penalty.upper()} penalty: {exact_zeros}/{d} coefficients are "
          f"EXACTLY zero, {tiny}/{d} are below 1e-3")
    print(f"  recovered first 5 (true {w_true[:5]}):")
    print(f"  {np.round(w[:5], 3)}")

print("\nL2 shrinks every coefficient toward zero without reaching it;")
print("L1 sets most of them to exactly zero. The difference is the corner")
print("on the L1 ball, and it is why L1 is used for feature selection.")
```

## 8. Practical Example

Choosing a similarity measure for a retrieval system is a real decision with a
measurable consequence, and the failure mode is subtle: the system works, but
systematically prefers the wrong documents.

```python {tier=A name=retrieval-measure-choice}
"""How the choice of similarity measure changes retrieval results.

Documents differ in length, which shows up as embedding magnitude. Cosine
ignores that; the raw dot product does not.
"""
import numpy as np

rng = np.random.default_rng(3)

# Two topics as directions, plus a length factor that scales the magnitude.
topic_a = np.array([1.0, 0.2, 0.0, 0.1]); topic_a /= np.linalg.norm(topic_a)
topic_b = np.array([0.0, 0.1, 1.0, 0.2]); topic_b /= np.linalg.norm(topic_b)

corpus = [
    ("short, exactly on topic A",   topic_a, 1.0),
    ("long, exactly on topic A",    topic_a, 4.0),
    ("short, mostly topic A",       0.85 * topic_a + 0.15 * topic_b, 1.0),
    ("very long, topic B",          topic_b, 6.0),
    ("medium, topic B",             topic_b, 2.5),
]
names = [c[0] for c in corpus]
D = np.stack([length * (v / np.linalg.norm(v)) for _, v, length in corpus])

query = topic_a.copy()

dot = D @ query
cos = dot / (np.linalg.norm(D, axis=1) * np.linalg.norm(query))
euc = np.linalg.norm(D - query, axis=1)

print(f"{'document':<28} {'|d|':>6} {'dot':>7} {'cos':>7} {'euclid':>8}")
for i, name in enumerate(names):
    print(f"{name:<28} {np.linalg.norm(D[i]):>6.2f} {dot[i]:>7.3f} "
          f"{cos[i]:>7.3f} {euc[i]:>8.3f}")

print("\nTop result by each measure:")
print(f"  dot product : {names[int(np.argmax(dot))]}")
print(f"  cosine      : {names[int(np.argmax(cos))]}")
print(f"  euclidean   : {names[int(np.argmin(euc))]}")

print("\nThe dot product ranks the LONG topic-B document above the short")
print("topic-A one purely because it is longer. Cosine gets it right.")
print("Euclidean prefers the short document because the query is short —")
print("it is measuring length agreement as well as topical agreement.")
```

> PRODUCTION TIP: Normalise embeddings once, at index time, and store the
> normalised vectors. You then get cosine semantics from a plain dot product —
> which is faster than cosine, since the division disappears — and by
> {{eq:cosine-euclidean}} the Euclidean index gives the same ordering. Deciding
> this after a corpus has been indexed means re-embedding everything, so it is
> worth getting right at the start ({{ch:emb-vector-db}}).

## 9. Common Mistakes

**Calling the squared $L_2$ norm a norm.** It fails homogeneity. It is still the
right thing to minimise in most cases, because it is differentiable at the
origin and avoids a square root — but the distinction matters when you are
checking whether a theorem applies.

**Assuming cosine distance is a metric.** $1 - \cos\theta$ violates the triangle
inequality. Index structures with metric guarantees cannot use it directly.

**Comparing cosine similarities across differently-scaled feature spaces.** A
cosine of 0.8 in one embedding space is not comparable to 0.8 in another. The
number has no absolute meaning; only the ordering within one space does.

**Mixing normalised and unnormalised vectors in one index.** The comparison
becomes meaningless, and no error is raised. Normalise everything or nothing.

**Using $L_1$ regularisation with a plain subgradient step and expecting
sparsity.** A subgradient step approaches zero without reaching it; only a
proximal or soft-thresholding step produces exact zeros. This is why the code in
{{sec:7-implementation}} uses soft-thresholding, and why a naive $L_1$
implementation appears not to work.

**Using Euclidean distance on high-dimensional embeddings without
normalising.** Distances concentrate ({{ch:math-vectors}}), and unnormalised
magnitude dominates. Normalise first.

**Forgetting that norms are not scale-invariant across features.** A feature
measured in metres and one measured in millimetres contribute wildly differently
to any $L_p$ norm. Standardise before measuring distance
({{ch:math-covariance}}).

## 10. Connection to Previous Chapters

{{ch:math-vectors}} defined the dot product and proved {{eq:dot-geometric}};
this chapter uses both, defining $\norm{\vec{x}}_2$ from the dot product in
{{eq:l2-from-dot}} and deriving cosine similarity from the angle formula. The
Cauchy-Schwarz inequality proved there is what guarantees cosine lies in
$[-1, 1]$. {{ch:math-matrices}} supplied the matrices whose norms
{{sec:5-formal-explanation}} measures.

Forward: {{ch:math-eigen}} shows the spectral norm is the largest singular
value, and that the Frobenius norm is the root-sum-square of all of them.
{{ch:math-optimization}} uses norms as regularisation penalties and derives the
sparsity result properly. {{ch:math-covariance}} standardises features so that
distances become comparable.

Beyond Part I: {{ch:ml-metrics}} chooses between $L_1$ and $L_2$ losses;
{{ch:dl-regularization}} applies weight decay; {{ch:dl-optimizers}} clips
gradient norms; {{ch:emb-similarity}} and {{ch:emb-vector-db}} are this chapter
applied at scale, and {{ch:emb-ann}} depends on the metric properties discussed
in {{sec:5-formal-explanation}}.

## 11. Exercises

**Beginner**

1. Compute the $L_1$, $L_2$ and $L_\infty$ norms of $[1, -2, 3, -4]\T$.
2. Normalise $[3, 4]\T$ and confirm the result has norm 1.
3. Compute the cosine similarity between $[1, 0]\T$ and $[1, 1]\T$.
4. Compute the Euclidean and Manhattan distances between $[1, 2, 3]\T$ and
   $[4, 6, 3]\T$.
5. Which of $[5, 0, 0]\T$ and $[3, 3, 3]\T$ has the larger $L_1$ norm? The
   larger $L_2$ norm?

**Intermediate**

6. Verify that $\norm{\vec{x}}_\infty \le \norm{\vec{x}}_2 \le \norm{\vec{x}}_1$
   for $[2, -3, 6]\T$, and explain why the ordering always holds.
7. Show that the squared $L_2$ norm violates {{eq:norm-homogeneous}}.
8. Two embeddings have cosine similarity 0.95 but Euclidean distance 12. What
   does that tell you about their magnitudes?
9. Verify {{eq:cosine-euclidean}} numerically on a pair of normalised vectors of
   your choosing.
10. Explain why "$L_0$" is not a norm, and why $L_1$ is used in its place.

**Advanced**

11. Prove the triangle inequality for $L_1$ directly from the triangle
    inequality for absolute values.
12. Prove {{eq:linf-norm}}: that $\lim_{p\to\infty}\norm{\vec{x}}_p =
    \max_i \lvert x_i \rvert$.
13. Give three vectors demonstrating that cosine distance violates the triangle
    inequality.
14. Show that $\norm{\mat{A}}_F^{2} = \tr(\mat{A}\T\mat{A})$.
15. Explain why the $L_1$ ball's corners lie on the coordinate axes while the
    $L_\infty$ ball's do not, and connect this to why $L_\infty$ regularisation
    encourages coefficients of *equal* magnitude rather than sparse ones.

**Implementation**

16. Implement all three norms without `np.linalg.norm` and verify against it.
17. Write `similarity_report(a, b)` returning dot product, cosine, Euclidean and
    Manhattan, and use it on pairs where the measures disagree.
18. Reproduce the sparsity experiment in {{sec:7-implementation}}, sweeping
    $\lambda$ for both penalties and plotting the number of exact zeros against
    $\lambda$.
19. Build a 5,000-document toy index. Measure how often the top-1 result differs
    between normalised and unnormalised dot-product retrieval, as a function of
    the spread of document lengths.

**Reasoning**

20. A recommender uses cosine similarity between user preference vectors. A
    colleague proposes switching to Euclidean distance. What changes, and who is
    helped or harmed?
21. Weight decay penalises $\norm{\vec{w}}_2^{2}$ rather than
    $\norm{\vec{w}}_1$. Given that sparse models are smaller and faster, why is
    $L_2$ the standard choice in deep learning?

## 12. Chapter Summary

A norm measures the size of a vector and must satisfy four axioms:
non-negativity, definiteness, homogeneity, and the triangle inequality. The
squared $L_2$ norm fails homogeneity and is therefore not a norm, though it is
usually the right thing to optimise.

Three norms carry the load. $L_1$ sums absolute values and charges less for
concentrating magnitude in few components. $L_2$ is straight-line length,
computed from the dot product, and is rotationally symmetric. $L_\infty$ takes
the largest component. They are always ordered
$\norm{\vec{x}}_\infty \le \norm{\vec{x}}_2 \le \norm{\vec{x}}_1$.

The unit balls make the differences visible. The $L_1$ ball is a diamond whose
corners sit on the coordinate axes, and a corner on an axis is a point with a
zero coordinate. That geometry — and specifically the non-smoothness of
$\lvert x \rvert$ at zero — is why $L_1$ regularisation produces exactly-zero
coefficients while $L_2$ merely shrinks them.

Every norm induces a distance, but not every distance comes from a norm. Cosine
distance is not a metric, which constrains how vector indexes can use it.

Cosine similarity divides out both magnitudes and measures only the angle,
making it the right default for embeddings where magnitude reflects document
length rather than relevance. On normalised vectors, cosine similarity is just a
dot product, and ranking by cosine is identical to ranking by Euclidean
distance — which is why retrieval systems normalise once at index time and use
whichever is faster.

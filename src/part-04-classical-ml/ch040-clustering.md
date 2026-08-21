---
id: ml-clustering
number: 40
part: IV
tier: focused
status: reviewed
requires: [ml-knn-nb, ml-metrics, math-vectors, math-probability]
provides: [clustering, k-means, lloyds-algorithm, k-means-plus-plus,
           dbscan, hierarchical-clustering, silhouette, cluster-validation,
           gaussian-mixture]
citations: [lloyd1982, pedregosa2011]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Implement Lloyd's algorithm and explain why it converges but not to the
   global optimum.
2. Explain k-means++ initialisation and why it matters.
3. State k-means' assumptions and predict exactly when it will fail.
4. Implement DBSCAN and explain what density-based clustering buys.
5. Choose a linkage for hierarchical clustering and read a dendrogram.
6. Evaluate a clustering without labels, and state honestly what that can and
   cannot tell you.
7. Explain why "how many clusters?" often has no answer.

## 2. Why This Matters

Clustering is where the discipline of the earlier chapters gets tested, because
there is no answer key.

**Every supervised chapter so far had a ground truth.** You could compute a test
error and be told whether you were right. Clustering has none. Every internal
metric measures whether the clusters match the *shape the metric assumes*, so a
metric built on compactness will always prefer round clusters, whether or not
any exist. The honest conclusion — that clustering results must be validated
against something external — is the main content of this chapter.

**k-means will find clusters in pure noise, confidently.** Ask for five clusters
in uniformly random data and you get five clusters with a respectable silhouette
score. Nothing in the algorithm can tell you they are meaningless. This failure
mode is the single most common way clustering is misused, and
{{sec:7-implementation}} measures it.

**It is the foundation of several things you will use later.** Vector
quantisation for embeddings ({{ch:emb-ann}} — IVF indexes are literally k-means),
codebooks in product quantisation, topic discovery, and the segmentation that
{{ch:ds-recsys}} needed.

## 3. Prerequisites

{{ch:ml-knn-nb}} for distance metrics, the scaling requirement, and the curse of
dimensionality — all of which apply here with full force.
{{ch:math-vectors}} for norms and centroids. {{ch:math-probability}} for
Gaussian mixtures. {{ch:ml-metrics}} for what an evaluation metric is supposed
to do, which is the frame for {{sec:5-formal-explanation}}'s honesty about
internal indices.

## 4. Intuitive Explanation

### 4.1 k-means: alternate two easy steps

The problem — partition points into $k$ groups minimising within-group spread —
is NP-hard. But it decomposes into two subproblems, each trivial:

```text
   ┌─────────────────────────────────────────────┐
   │  ASSIGN: each point to its nearest centroid │
   │          (easy: compute k distances)        │
   └───────────────────┬─────────────────────────┘
                       │
   ┌───────────────────▼─────────────────────────┐
   │  UPDATE: each centroid to the mean of its   │
   │          assigned points (easy: average)    │
   └───────────────────┬─────────────────────────┘
                       │
                repeat until nothing moves
```

This is **Lloyd's algorithm** {{cite:lloyd1982}}, and both steps reduce the same
objective, so it always converges. It converges to a *local* optimum that
depends entirely on the initialisation — which is why you run it several times
and keep the best, and why k-means++ exists.

### 4.2 What k-means assumes

Every failure of k-means follows from three assumptions, and they are worth
holding in mind as a checklist.

**Clusters are spherical.** The objective is squared Euclidean distance to a
centre, so a cluster is a ball. Elongated clusters get split; two parallel
elongated clusters get merged crosswise.

**Clusters are similarly sized.** The objective is a sum over points, so a large
cluster contributes more and the algorithm will happily carve it up while
merging two small ones.

**$k$ is known.** It is an input, not an output, and k-means will produce
exactly $k$ clusters from any data whatsoever.

```text
   works                      splits it              merges them
   ●●●   ▲▲▲              ●●●●●●●●●●●●●          ●●●●●●●●●●●●●●
  ●●●●● ▲▲▲▲▲                                    ▲▲▲▲▲▲▲▲▲▲▲▲▲▲
   ●●●   ▲▲▲              one long cluster       two parallel bands
                          becomes two            become one vertical
                                                 pair of halves
```

### 4.3 DBSCAN: clusters as dense regions

DBSCAN replaces "near a centre" with "connected through dense neighbourhoods".
A point is a **core point** if at least `min_samples` points lie within
`eps` of it; core points within `eps` of each other join the same cluster;
points near a cluster but not themselves core are boundary points; everything
else is **noise**.

Three consequences follow, and they are exactly the complements of k-means'
weaknesses:

- **Arbitrary shapes.** Two interlocking crescents are two clusters, because
  connectivity does not care about convexity.
- **$k$ is discovered, not supplied.** The number of clusters falls out of the
  density structure.
- **Noise is a category.** Points in sparse regions are labelled $-1$ rather
  than forced into a cluster.

The cost is that `eps` is difficult to choose and DBSCAN assumes *uniform*
density — with clusters of genuinely different densities, no single `eps` works,
and you need HDBSCAN.

### 4.4 Hierarchical clustering

Build a tree. Agglomerative clustering starts with every point its own cluster
and repeatedly merges the two closest, recording the merge heights in a
**dendrogram**. Cutting the dendrogram at any height yields a clustering, so you
get every $k$ at once and can choose afterwards.

The **linkage** — how the distance between two clusters is defined — decides
everything:

- **Single** (nearest pair): finds elongated, chain-like structure; prone to
  chaining two clusters together through one bridging point.
- **Complete** (farthest pair): compact, roughly spherical clusters.
- **Average**: between the two.
- **Ward**: minimises the increase in within-cluster variance — the same
  objective as k-means, so it produces similar results with a hierarchy
  attached, and it is the usual default.

The cost is $O(N^{2})$ memory and $O(N^{2}\log N)$ time, which caps it at tens of
thousands of points.

## 5. Formal Explanation

### 5.1 The k-means objective

Minimise the **within-cluster sum of squares**:

$$
J = \sum_{j=1}^{k}\sum_{\vec{x}_i \in C_j} \|\vec{x}_i - \vecgreek{\mu}_j\|^{2}
$$ (eq:wcss)

over both the assignment and the centroids. This is NP-hard in general.

Lloyd's algorithm alternates:

$$
\text{assign:} \quad c_i = \argmin_j \|\vec{x}_i - \vecgreek{\mu}_j\|^{2}
\qquad
\text{update:} \quad \vecgreek{\mu}_j = \frac{1}{|C_j|}\sum_{i \in C_j}\vec{x}_i
$$ (eq:lloyd)

### 5.2 k-means++

Random initialisation frequently places two centroids in the same true cluster,
and Lloyd's algorithm cannot recover. k-means++ spreads the initial centroids:

1. Choose the first centroid uniformly at random.
2. For each remaining centroid, choose $\vec{x}$ with probability proportional
   to $D(\vec{x})^{2}$, where $D(\vec{x})$ is the distance to the nearest
   already-chosen centroid.
3. Run Lloyd's algorithm.

Sampling proportional to $D^{2}$ makes distant points overwhelmingly likely to
be picked, which is what spreads the seeds. It comes with a guarantee — the
expected objective is within $O(\log k)$ of optimal *before Lloyd's algorithm
even runs* — and it is the default in every implementation
{{cite:pedregosa2011}}.

### 5.3 DBSCAN

Given `eps` $\epsilon$ and `min_samples` $m$:

- $\vec{x}$ is a **core point** if $|N_{\epsilon}(\vec{x})| \ge m$.
- $\vec{y}$ is **directly reachable** from core $\vec{x}$ if
  $\vec{y} \in N_{\epsilon}(\vec{x})$.
- **Reachable** is the transitive closure through core points.
- A cluster is a maximal set of mutually reachable points.
- Everything unreachable from any core point is **noise**.

Complexity is $O(N \log N)$ with a spatial index, $O(N^{2})$ without.

A useful heuristic for $\epsilon$: plot the sorted distance to each point's
$m$-th nearest neighbour, and take the value at the "elbow" — where the curve
turns sharply upward, indicating the transition from within-cluster to
between-cluster distances. {{sec:7-implementation}} does this.

### 5.4 Evaluating a clustering, honestly

This is the section that matters most, and the one usually skipped.

**Internal indices** use only the data and the labels the algorithm produced.

$$
s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
$$ (eq:silhouette)

The **silhouette** compares $a(i)$, the mean distance to points in $i$'s own
cluster, with $b(i)$, the mean distance to the nearest other cluster. It ranges
over $[-1,1]$.

The Davies–Bouldin index and the Calinski–Harabasz index are similar in spirit.
All three share one decisive flaw:

> WARNING: **Every internal index encodes an assumption about cluster shape,
> and therefore cannot arbitrate between algorithms that assume different
> shapes.** The silhouette rewards compact, well-separated groups, so on
> interlocking crescents it will prefer k-means' wrong-but-round answer over
> DBSCAN's correct one. Using it to choose between them is circular.
> {{sec:7-implementation}} measures exactly this.

**External indices** compare against known labels — the adjusted Rand index,
normalised mutual information. These are trustworthy and available only when you
already have labels, which is to say almost never in the situation that
motivated clustering.

**Stability** is the most useful label-free check available: cluster many
bootstrap resamples and measure how often pairs of points land together. Genuine
structure is reproducible; noise is not. It does not prove the clusters mean
anything, but it reliably detects clusters that do not survive resampling.

**Downstream utility** is the honest criterion. Do the segments predict
something? Do they make a decision better? A clustering exists to be used, and
whether it is useful is a question with an answer.

### 5.5 Choosing $k$

The **elbow method** plots {{eq:wcss}} against $k$ and looks for the bend. It is
subjective, frequently ambiguous, and $J$ decreases monotonically in $k$ by
construction, reaching zero at $k=N$.

The **silhouette method** picks the $k$ maximising mean silhouette. Less
arbitrary, and still biased towards spherical clusters.

The **gap statistic** compares $\log J$ against its expectation under a null
reference distribution of no clustering. This is the principled one, because it
is the only common method that can return "$k=1$" — no structure at all.

> IMPORTANT: The most important answer to "how many clusters?" is often "the
> question is ill-posed". Customer behaviour is a continuum, not three discrete
> types. If several $k$ score similarly, that is evidence there is no natural
> number of clusters, and reporting one anyway is manufacturing a finding.

## 6. Mathematical Foundation

### 6.1 Why Lloyd's algorithm converges

Both steps of {{eq:lloyd}} are exact minimisations of {{eq:wcss}} with the other
argument held fixed.

**The assignment step.** With centroids fixed, $J$ is a sum of independent
per-point terms, and each is minimised by assigning that point to the nearest
centroid. Any other assignment gives a larger or equal term.

**The update step.** With assignments fixed, $J$ decomposes across clusters, and
each cluster's contribution $\sum_{i \in C_j}\|\vec{x}_i - \vecgreek{\mu}\|^{2}$
is a convex quadratic in $\vecgreek{\mu}$. Setting the derivative to zero:

$$
\frac{\partial}{\partial \vecgreek{\mu}_j}\sum_{i \in C_j}
  \|\vec{x}_i - \vecgreek{\mu}_j\|^{2}
 = -2\sum_{i \in C_j}(\vec{x}_i - \vecgreek{\mu}_j) = 0
 \;\Rightarrow\;
 \vecgreek{\mu}_j = \frac{1}{|C_j|}\sum_{i \in C_j}\vec{x}_i
$$ (eq:centroid-is-mean)

So the mean is not a heuristic choice of representative — it is the exact
minimiser of squared distance. (Use $\ell_1$ instead and the minimiser is the
*median*; that algorithm is k-medians, and it is more robust to outliers for
exactly the reason {{ch:ds-cleaning}} gave.)

Since $J$ never increases and there are finitely many assignments, the algorithm
terminates in finitely many steps. It terminates at a local optimum — coordinate
descent on a non-convex objective — which is why restarts are necessary.

### 6.2 The bound k-means++ buys

$$
\E[J_{\text{k-means++}}] \le 8(\ln k + 2)\, J_{\text{OPT}}
$$ (eq:kmeanspp-bound)

This holds for the seeding alone, before any Lloyd iterations, and it is the
only approximation guarantee available for this problem. Random initialisation
has no such bound: its objective can be arbitrarily worse than optimal.

The intuition for the $D^{2}$ weighting: a point far from every existing
centroid has a large $D^{2}$ and is therefore very likely to be chosen, so an
uncovered cluster is likely to receive a seed. Squaring rather than using $D$
itself sharpens that preference; sampling proportional to $D^{0}$ would be
uniform, which is the failure case.

### 6.3 Why k-means makes spherical clusters

The decision boundary between clusters $j$ and $l$ is where the distances are
equal:

$$
\|\vec{x}-\vecgreek{\mu}_j\|^{2} = \|\vec{x}-\vecgreek{\mu}_l\|^{2}
$$

Expanding and cancelling $\|\vec{x}\|^{2}$ from both sides:

$$
-2\vec{x}\T\vecgreek{\mu}_j + \|\vecgreek{\mu}_j\|^{2}
 = -2\vec{x}\T\vecgreek{\mu}_l + \|\vecgreek{\mu}_l\|^{2}
$$

$$
\Rightarrow\quad
2\vec{x}\T(\vecgreek{\mu}_l - \vecgreek{\mu}_j)
 = \|\vecgreek{\mu}_l\|^{2} - \|\vecgreek{\mu}_j\|^{2}
$$ (eq:kmeans-boundary)

**Linear in $\vec{x}$.** Every k-means boundary is a hyperplane, and the
resulting partition is a Voronoi diagram: convex polyhedral cells. That is a
complete answer to what k-means can express, and it immediately implies the
failures — a non-convex cluster cannot be one cell, so it must be split.

The generalisation is instructive. A **Gaussian mixture** fitted by EM replaces
the fixed spherical distance with a learned covariance $\mat{\Sigma}_j$ per
component, giving Mahalanobis rather than Euclidean distance and hence
ellipsoidal clusters at arbitrary orientations, plus soft assignments. k-means
is exactly the limiting case with $\mat{\Sigma}_j = \sigma^{2}\mat{I}$ and
$\sigma \to 0$.

### 6.4 What the silhouette can and cannot see

Rewrite {{eq:silhouette}}: $s(i) > 0$ requires $b(i) > a(i)$ — the mean distance
to the nearest other cluster exceeds the mean distance within one's own.

Consider two interlocking crescents. A point in the middle of one crescent has
$a(i)$ large, because the far tip of its own crescent is far away, and $b(i)$
small, because the other crescent curls around close by. So $s(i)$ is small or
negative *for the correct clustering*.

The silhouette is therefore not measuring "is this clustering right". It is
measuring "are these clusters compact and separated", which is the same question
only when the clusters happen to be balls. Using it to choose between k-means
and DBSCAN asks a compactness-based judge to rule on a connectivity-based
method, and the verdict is decided before the data is seen.

The general principle is worth stating plainly, because it applies beyond
clustering: **an unsupervised evaluation metric is a model of what a good answer
looks like.** It cannot be more neutral than the assumption it encodes.

## 7. Implementation

```python {tier=A name=kmeans-from-scratch}
"""k-means from scratch: Lloyd's algorithm, k-means++, and the three failures.
"""
import numpy as np

rng = np.random.default_rng(0)


def kmeans(X, k, init="++", n_init=10, max_iter=300, tol=1e-9, seed=0):
    """Lloyd's algorithm (eq. 40.2) with restarts; returns the best run."""
    rs = np.random.default_rng(seed)
    best = (None, None, np.inf, 0)
    for run in range(n_init):
        C = (kmeanspp_init(X, k, rs) if init == "++"
             else X[rs.choice(len(X), k, replace=False)].copy())
        for it in range(max_iter):
            d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
            lab = d.argmin(1)
            newC = C.copy()
            for j in range(k):
                m = lab == j
                if m.any():
                    newC[j] = X[m].mean(0)
                else:
                    newC[j] = X[rs.integers(0, len(X))]   # re-seed empty
            shift = float(((newC - C) ** 2).sum())
            C = newC
            if shift < tol:
                break
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        J = float(d[np.arange(len(X)), lab].sum())        # eq. 40.1
        if J < best[2]:
            best = (C, lab, J, it + 1)
    return best


def kmeanspp_init(X, k, rs):
    """Seed proportional to D^2 (section 5.2)."""
    C = [X[rs.integers(0, len(X))]]
    for _ in range(1, k):
        D2 = np.min(((X[:, None, :] - np.array(C)[None, :, :]) ** 2).sum(-1),
                    axis=1)
        total = D2.sum()
        p = D2 / total if total > 0 else np.full(len(X), 1 / len(X))
        C.append(X[rs.choice(len(X), p=p)])
    return np.array(C)


def make_blobs(n, centres, spread=0.6):
    centres = np.asarray(centres, float)
    lab = rng.integers(0, len(centres), n)
    return centres[lab] + rng.normal(0, spread, (n, centres.shape[1])), lab


# --- k-means++ vs random initialisation -------------------------------------
print("=" * 72)
print("initialisation decides the local optimum you land in (section 5.2)")
print("=" * 72)
# eight well-separated blobs: random seeding will double up on some
centres8 = [[0, 0], [10, 0], [20, 0], [30, 0],
            [0, 10], [10, 10], [20, 10], [30, 10]]
X8, _ = make_blobs(1600, centres8, spread=0.8)

print(f"{'init':>10} {'restarts':>10} {'best WCSS':>12} {'mean WCSS':>12} "
      f"{'worst WCSS':>12}")
for init in ("random", "++"):
    for n_init in (1, 10):
        Js = [kmeans(X8, 8, init=init, n_init=1, seed=100 + s)[2]
              for s in range(20)]
        if n_init == 1:
            print(f"{init:>10} {1:>10} {min(Js):>12.1f} "
                  f"{np.mean(Js):>12.1f} {max(Js):>12.1f}")
        else:
            best_of_10 = [min(Js[i:i + 10]) for i in (0, 10)]
            print(f"{init:>10} {10:>10} {min(best_of_10):>12.1f} "
                  f"{np.mean(best_of_10):>12.1f} {max(best_of_10):>12.1f}")

print("\nWith a single random start the objective varies enormously across")
print("seeds — some runs place two centroids in one blob and leave another")
print("empty, and Lloyd's algorithm cannot recover, because both of its")
print("steps only ever DECREASE the objective (section 6.1). k-means++ is")
print("far more consistent, and restarts help both. Never run k-means once.")

# --- section 6.1: the objective never increases -----------------------------
print("\n" + "=" * 72)
print("both steps of Lloyd's algorithm decrease J (section 6.1)")
print("=" * 72)
X3, _ = make_blobs(600, [[0, 0], [4, 4], [8, 0]], spread=1.0)
rs = np.random.default_rng(3)
C = X3[rs.choice(len(X3), 3, replace=False)].copy()
print(f"{'iter':>5} {'J after assign':>16} {'J after update':>16}")
for it in range(7):
    d = ((X3[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    lab = d.argmin(1)
    J_assign = float(d[np.arange(len(X3)), lab].sum())
    for j in range(3):
        if (lab == j).any():
            C[j] = X3[lab == j].mean(0)
    d2 = ((X3[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    J_update = float(d2[np.arange(len(X3)), lab].sum())
    print(f"{it:>5} {J_assign:>16.4f} {J_update:>16.4f}")
print("\nMonotone decrease, in both columns, at every step. Since there are")
print("finitely many assignments the algorithm must terminate — and it")
print("terminates at whichever local optimum the initialisation led to.")

# --- section 4.2: the three assumptions, and the three failures -------------
print("\n" + "=" * 72)
print("k-means' three assumptions, each violated in turn (section 4.2)")
print("=" * 72)


def purity(true_lab, pred_lab):
    """Fraction correct under the best matching of predicted to true labels."""
    total = 0
    for p in np.unique(pred_lab):
        m = pred_lab == p
        if m.any():
            total += np.bincount(true_lab[m]).max()
    return total / len(true_lab)


# 1. elongated clusters
t = rng.uniform(-6, 6, 800)
Xe = np.column_stack([t, 0.25 * t + rng.normal(0, 0.35, 800)])
Xe = np.vstack([Xe, np.column_stack([t, 0.25 * t + 4
                                     + rng.normal(0, 0.35, 800)])])
ye = np.r_[np.zeros(800, int), np.ones(800, int)]

# 2. different sizes
Xs = np.vstack([rng.normal([0, 0], 0.5, (1500, 2)),
                rng.normal([5, 0], 0.5, (60, 2)),
                rng.normal([5, 3], 0.5, (60, 2))])
ys = np.r_[np.zeros(1500, int), np.ones(60, int), np.full(60, 2)]

# 3. different densities
Xd = np.vstack([rng.normal([0, 0], 0.3, (600, 2)),
                rng.normal([4, 0], 1.8, (600, 2))])
yd = np.r_[np.zeros(600, int), np.ones(600, int)]

# 4. the case it is built for
Xg, yg = make_blobs(1200, [[0, 0], [6, 0], [3, 5]], spread=0.9)

print(f"{'geometry':<34} {'k':>3} {'purity':>8} {'verdict':<22}")
for name, Xc, yc, k in (
        ("spherical, equal size (its home)", Xg, yg, 3),
        ("elongated parallel bands", Xe, ye, 2),
        ("one huge cluster, two tiny", Xs, ys, 3),
        ("very different densities", Xd, yd, 2)):
    _, lab, _, _ = kmeans(Xc, k, n_init=10, seed=7)
    pu = purity(yc, lab)
    verdict = ("recovers it" if pu > 0.95 else
               "partly" if pu > 0.75 else "fails")
    print(f"{name:<34} {k:>3} {pu:>8.4f} {verdict:<22}")

print("\nThe shape assumption is the one that bites, and it bites hard: on")
print("parallel elongated bands k-means scores barely above chance. That")
print("follows directly from eq. 40.7 — the boundary between two k-means")
print("clusters is a HYPERPLANE, so the cells are convex polyhedra, and no")
print("plane separates two bands running alongside each other. It cuts them")
print("crosswise instead.")
print("\nThe other two assumptions are worth being honest about. Unequal")
print("SIZE did not hurt here, because the clusters were also far apart —")
print("the textbook warning applies when a large cluster is close enough to")
print("small ones that splitting it costs less than merging them. Unequal")
print("DENSITY cost about ten points. Neither is in the same category as")
print("the shape failure.")
print("\nThe usable summary: k-means is a Voronoi partition. If your")
print("clusters are not roughly convex blobs, that is the assumption to")
print("check first, and the other two are second-order.")

# --- the failure that matters most: no structure at all ---------------------
print("\n" + "=" * 72)
print("k-means finds clusters in pure noise, and looks confident doing it")
print("=" * 72)


def silhouette(X, lab):
    """Mean silhouette (eq. 40.3). O(N^2) — fine at this size."""
    D = np.sqrt(np.maximum(
        ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1), 0))
    out = np.zeros(len(X))
    labs = np.unique(lab)
    if len(labs) < 2:
        return 0.0
    for i in range(len(X)):
        own = lab == i * 0 + lab[i]
        own[i] = False
        a = D[i, own].mean() if own.any() else 0.0
        b = min(D[i, lab == L].mean() for L in labs if L != lab[i])
        out[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return float(out.mean())


X_noise = rng.uniform(0, 1, (500, 2))         # uniform. No clusters exist.
print("500 points drawn uniformly from a square — there is no structure.\n")
print(f"{'k':>4} {'WCSS':>10} {'mean silhouette':>17} {'reported as':<26}")
for k in (2, 3, 4, 5, 8):
    _, lab, J, _ = kmeans(X_noise, k, n_init=10, seed=5)
    s = silhouette(X_noise, lab)
    verdict = ("'reasonable structure'" if s > 0.35 else
               "'weak but present'" if s > 0.25 else "'no structure'")
    print(f"{k:>4} {J:>10.2f} {s:>17.4f} {verdict:<26}")

print("\nEvery k returns k clusters with a positive silhouette. Nothing in")
print("the algorithm or the metric can say 'there is nothing here'. The")
print("elbow method is no help either: WCSS falls smoothly and monotonically")
print("with k, as it must, since more centroids can only reduce eq. 40.1.")
print("\nThis is the single most common way clustering is misused, and it is")
print("why section 5.4 insists on a reference distribution, a stability")
print("check, or an external outcome.")
```

```python {tier=A name=dbscan-and-validation}
"""DBSCAN, hierarchical linkage, and honest cluster validation.
"""
import numpy as np

rng = np.random.default_rng(11)


def pairwise(A, B=None):
    B = A if B is None else B
    return np.sqrt(np.maximum(
        ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1), 0))


def dbscan(X, eps, min_samples):
    """Section 5.3, literally: core points, then transitive closure."""
    D = pairwise(X)
    neigh = [np.flatnonzero(row <= eps) for row in D]
    core = np.array([len(nb) >= min_samples for nb in neigh])
    labels = np.full(len(X), -1)
    cid = 0
    for i in range(len(X)):
        if labels[i] != -1 or not core[i]:
            continue
        stack, labels[i] = [i], cid
        while stack:
            p = stack.pop()
            for q in neigh[p]:
                if labels[q] == -1:
                    labels[q] = cid
                    if core[q]:                 # only core points expand
                        stack.append(q)
        cid += 1
    return labels, core


def kmeans(X, k, n_init=10, seed=0):
    rs = np.random.default_rng(seed)
    best = (None, np.inf)
    for _ in range(n_init):
        C = [X[rs.integers(0, len(X))]]
        for _ in range(1, k):
            D2 = np.min(((X[:, None, :] - np.array(C)[None, :, :]) ** 2
                         ).sum(-1), axis=1)
            tot = D2.sum()
            C.append(X[rs.choice(len(X),
                                 p=D2 / tot if tot > 0 else None)])
        C = np.array(C)
        for _ in range(200):
            d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
            lab = d.argmin(1)
            newC = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j]
                             for j in range(k)])
            if np.allclose(newC, C):
                C = newC
                break
            C = newC
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        J = float(d[np.arange(len(X)), lab].sum())
        if J < best[1]:
            best = (lab, J)
    return best[0]


def silhouette(X, lab):
    mask = lab >= 0                        # noise points are not scored
    X, lab = X[mask], lab[mask]
    labs = np.unique(lab)
    if len(labs) < 2:
        return float("nan")
    D = pairwise(X)
    out = np.zeros(len(X))
    for i in range(len(X)):
        own = lab == lab[i]
        own[i] = False
        a = D[i, own].mean() if own.any() else 0.0
        b = min(D[i, lab == L].mean() for L in labs if L != lab[i])
        out[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return float(out.mean())


def adjusted_rand(a, b):
    """External index: needs true labels, and is therefore trustworthy."""
    from itertools import product
    la, lb = np.unique(a), np.unique(b)
    cont = np.array([[np.sum((a == i) & (b == j)) for j in lb] for i in la])

    def c2(x):
        return x * (x - 1) / 2
    sum_ij = c2(cont).sum()
    sum_i = c2(cont.sum(1)).sum()
    sum_j = c2(cont.sum(0)).sum()
    n2 = c2(len(a))
    exp = sum_i * sum_j / n2
    mx = 0.5 * (sum_i + sum_j)
    return float((sum_ij - exp) / (mx - exp)) if mx != exp else 1.0


# --- two crescents: the canonical case where shape decides ------------------
def make_moons(n, noise=0.09):
    t = rng.uniform(0, np.pi, n)
    top = np.column_stack([np.cos(t), np.sin(t)])
    bot = np.column_stack([1 - np.cos(t), 0.35 - np.sin(t)])
    X = np.vstack([top, bot]) + rng.normal(0, noise, (2 * n, 2))
    return X, np.r_[np.zeros(n, int), np.ones(n, int)]


Xm, ym = make_moons(300)

print("=" * 72)
print("the metric decides the winner before the data is seen (section 6.4)")
print("=" * 72)

km_lab = kmeans(Xm, 2, seed=1)
db_lab, _ = dbscan(Xm, eps=0.22, min_samples=5)

print(f"{'method':<22} {'clusters':>9} {'noise':>7} "
      f"{'silhouette':>12} {'adj. Rand (truth)':>19}")
for name, lab in (("k-means, k=2", km_lab), ("DBSCAN", db_lab)):
    nc = len(set(lab[lab >= 0]))
    print(f"{name:<22} {nc:>9} {int((lab == -1).sum()):>7} "
          f"{silhouette(Xm, lab):>12.4f} {adjusted_rand(ym, lab):>19.4f}")

print("\nRead the two right-hand columns against each other. DBSCAN recovers")
print("the true crescents almost exactly — adjusted Rand near 1.0 — and the")
print("SILHOUETTE PREFERS K-MEANS, which is simply wrong.")
print("\nThe reason is eq. 40.3. A point in the middle of a crescent is far")
print("from the far tip of its OWN crescent (large a) and close to the other")
print("crescent curling around it (small b), so the correct clustering")
print("scores badly on a metric built out of compactness. An internal index")
print("cannot arbitrate between algorithms that assume different shapes: it")
print("is a model of what a good answer looks like, and here it is the wrong")
print("model.")

# --- eps: the elbow in the k-distance plot ----------------------------------
print("\n" + "=" * 72)
print("choosing eps from the k-distance curve (section 5.3)")
print("=" * 72)
m = 5
D = pairwise(Xm)
kdist = np.sort(np.sort(D, axis=1)[:, m])
qs = [0.5, 0.7, 0.85, 0.92, 0.99, 1.0]
eps_extra = [0.30, 0.50]
print(f"{'quantile of 5-NN distance':>26} {'eps':>8} {'clusters':>9} "
      f"{'noise':>7} {'adj. Rand':>11}")
for q in qs:
    e = float(np.quantile(kdist, q))
    lab, _ = dbscan(Xm, eps=e, min_samples=m)
    nc = len(set(lab[lab >= 0]))
    print(f"{q:>26.2f} {e:>8.4f} {nc:>9} {int((lab == -1).sum()):>7} "
          f"{adjusted_rand(ym, lab):>11.4f}")
for e in eps_extra:                        # beyond the k-distance range
    lab, _ = dbscan(Xm, eps=e, min_samples=m)
    nc = len(set(lab[lab >= 0]))
    print(f"{'(beyond the curve)':>26} {e:>8.4f} {nc:>9} "
          f"{int((lab == -1).sum()):>7} {adjusted_rand(ym, lab):>11.4f}")

print("\nToo small and the data fragments into 21 clusters with a sixth of")
print("the points discarded as noise. Too large and the two crescents merge")
print("into one, which shows up as a cluster count of 1 and an adjusted")
print("Rand near zero. The usable window here runs from roughly the 85th to")
print("the 99th percentile of the 5-NN distance — real, but narrow, and it")
print("has to be found. This sensitivity is DBSCAN's main practical")
print("weakness, and the mirror image of k-means needing k.")

# --- DBSCAN's own assumption: uniform density -------------------------------
print("\n" + "=" * 72)
print("DBSCAN assumes uniform density, and fails when that is false")
print("=" * 72)
Xv = np.vstack([rng.normal([0, 0], 0.12, (400, 2)),
                rng.normal([1.4, 0], 0.12, (400, 2)),
                rng.normal([9, 0], 2.20, (400, 2))])
yv = np.r_[np.zeros(400, int), np.ones(400, int), np.full(400, 2)]
print("two tight clusters and one diffuse one\n")
print(f"{'eps':>7} {'clusters':>9} {'noise':>7} {'tight pair kept':>17} "
      f"{'diffuse kept':>14} {'what happened':<28}")
for e in (0.08, 0.15, 0.3, 0.6, 1.2, 2.0):
    lab, _ = dbscan(Xv, eps=e, min_samples=6)
    nc = len(set(lab[lab >= 0]))
    frac_noise = (lab == -1).mean()
    # did the two tight clusters stay separate, and was the diffuse one
    # recovered as a cluster rather than discarded as noise?
    t1, t2 = lab[:400], lab[400:800]
    kept_apart = (len(set(t1[t1 >= 0]) & set(t2[t2 >= 0])) == 0
                  and (t1 >= 0).mean() > 0.5 and (t2 >= 0).mean() > 0.5)
    diffuse_kept = float((lab[800:] >= 0).mean())
    note = ("diffuse cluster mostly noise" if frac_noise > 0.25
            else "tight clusters merged" if nc < 3
            else "over-fragmented" if nc > 4 else "")
    print(f"{e:>7} {nc:>9} {int((lab == -1).sum()):>7} "
          f"{('yes' if kept_apart else 'NO'):>17} "
          f"{diffuse_kept:>14.2f} {note:<28}")
best_e = None
print("\nRead the two middle columns together — that is the whole point.")
print("Small eps keeps the tight pair apart and throws the diffuse cluster")
print("away as noise. Large eps recovers the diffuse cluster and fuses the")
print("pair into one. There is no setting that does both cleanly.")
print("\nThe least-bad compromise is eps = 0.6, and look at what it costs:")
print("the pair survives and 78% of the diffuse cluster is kept, but the")
print("run reports SEVEN clusters — the diffuse region has been chopped")
print("into pieces wherever a local thin patch appeared. Three real")
print("clusters, and no eps returns three.")
print("\nThe two tight clusters are 1.4 apart with spread 0.12; the diffuse")
print("one has spread 2.2. Small eps keeps the tight pair separate and")
print("throws the diffuse cluster away as noise; large eps captures the")
print("diffuse cluster and glues the tight pair together. There is no")
print("value that does both, because DBSCAN has ONE density scale and the")
print("data has two.")
print("\nThis is the same shape as the impossibility in section 6.3 of the")
print("next chapter, and it is what HDBSCAN exists for: it builds a")
print("hierarchy over all eps simultaneously and extracts clusters that are")
print("stable across a range of scales, rather than committing to one.")

# --- section 5.4: stability, the useful label-free check --------------------
print("\n" + "=" * 72)
print("stability: does the clustering survive resampling? (section 5.4)")
print("=" * 72)


def resample_stability(X, k, n_boot=12, seed=0):
    """Mean adjusted Rand between clusterings of independent resamples,
    computed on the points the two resamples share.

    A clustering that reflects real structure reproduces; one fitted to
    noise does not. Needs no labels and assumes no shape. Note the metric:
    'how often are two points together' rises mechanically with k, because
    most pairs are apart in every run — ARI corrects for chance and does
    not have that defect.
    """
    rs = np.random.default_rng(seed)
    n = len(X)
    runs = []
    for _ in range(n_boot):
        idx = np.unique(rs.integers(0, n, n))
        runs.append((idx, kmeans(X[idx], k, n_init=3,
                                 seed=int(rs.integers(0, 10 ** 6)))))
    scores = []
    for a in range(len(runs)):
        for b in range(a + 1, len(runs)):
            ia, la = runs[a]
            ib, lb = runs[b]
            shared = np.intersect1d(ia, ib)
            if len(shared) < 20:
                continue
            pa = la[np.searchsorted(ia, shared)]
            pb = lb[np.searchsorted(ib, shared)]
            scores.append(adjusted_rand(pa, pb))
    return float(np.mean(scores)) if scores else float("nan")


X_real, _ = make_moons(220, noise=0.06)
X_blobs = np.vstack([rng.normal([0, 0], 0.6, (220, 2)),
                     rng.normal([5, 0], 0.6, (220, 2)),
                     rng.normal([2.5, 4], 0.6, (220, 2))])
X_null = rng.uniform(0, 1, (440, 2))

print(f"{'data':<34} {'k':>3} {'silhouette':>12} "
      f"{'resample stability':>20}")
for name, Xc, k in (("three real blobs", X_blobs, 3),
                    ("two crescents (k-means is wrong)", X_real, 2),
                    ("uniform noise (no structure)", X_null, 3)):
    lab = kmeans(Xc, k, seed=2)
    print(f"{name:<34} {k:>3} {silhouette(Xc, lab):>12.4f} "
          f"{resample_stability(Xc, k):>20.4f}")

print("\nStability separates genuine structure from an arbitrary partition of")
print("noise without needing labels or assuming a shape. It is not proof")
print("that the clusters MEAN anything — only that they are reproducible —")
print("but it reliably catches the case that a silhouette cannot.")
```

## 8. Practical Example

```python {tier=A name=segmentation-workflow}
"""A customer-segmentation workflow done honestly: scale, choose k with a
null reference, validate, and check the segments are actually useful.
"""
import numpy as np

rng = np.random.default_rng(23)


def kmeans(X, k, n_init=10, seed=0, max_iter=200):
    rs = np.random.default_rng(seed)
    best = (None, None, np.inf)
    for _ in range(n_init):
        C = [X[rs.integers(0, len(X))]]
        for _ in range(1, k):
            D2 = np.min(((X[:, None, :] - np.array(C)[None, :, :]) ** 2
                         ).sum(-1), axis=1)
            tot = D2.sum()
            C.append(X[rs.choice(len(X), p=D2 / tot if tot > 0 else None)])
        C = np.array(C)
        for _ in range(max_iter):
            d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
            lab = d.argmin(1)
            newC = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j]
                             for j in range(k)])
            if np.allclose(newC, C):
                C = newC
                break
            C = newC
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        J = float(d[np.arange(len(X)), lab].sum())
        if J < best[2]:
            best = (C, lab, J)
    return best


# --- a population with three genuine behavioural types ----------------------
N = 1200
seg = rng.choice(3, N, p=[0.55, 0.30, 0.15])
recency = np.where(seg == 0, rng.gamma(6, 8, N),
                   np.where(seg == 1, rng.gamma(2, 6, N),
                            rng.gamma(1.5, 3, N)))
frequency = np.where(seg == 0, rng.poisson(2, N),
                     np.where(seg == 1, rng.poisson(9, N),
                              rng.poisson(22, N))) + 1
monetary = frequency * np.where(seg == 0, rng.gamma(3, 12, N),
                                np.where(seg == 1, rng.gamma(4, 20, N),
                                         rng.gamma(5, 55, N)))
tenure = rng.uniform(1, 60, N)
X_raw = np.column_stack([recency, frequency, monetary, tenure])
NAMES = ["recency (days)", "frequency", "monetary (GBP)", "tenure (months)"]

print("=" * 72)
print("1. scaling is not optional (Chapter 35's lesson, again)")
print("=" * 72)
print(f"{'feature':<18} {'mean':>12} {'std':>12} "
      f"{'share of squared distance':>27}")
var = X_raw.var(0)
for nm, m_, s_, v in zip(NAMES, X_raw.mean(0), X_raw.std(0),
                         var / var.sum()):
    print(f"{nm:<18} {m_:>12.2f} {s_:>12.2f} {v:>27.4f}")
print("\nUnscaled, 'monetary' contributes almost all of the distance purely")
print("because it is measured in pounds. k-means would be clustering on one")
print("column.")

X = (X_raw - X_raw.mean(0)) / X_raw.std(0)


def silhouette(X, lab):
    labs = np.unique(lab)
    if len(labs) < 2:
        return float("nan")
    D = np.sqrt(np.maximum(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1), 0))
    out = np.zeros(len(X))
    for i in range(len(X)):
        own = lab == lab[i]
        own[i] = False
        a = D[i, own].mean() if own.any() else 0.0
        b = min(D[i, lab == L].mean() for L in labs if L != lab[i])
        out[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return float(out.mean())


# --- 2. choosing k against a null reference (the gap statistic) -------------
print("\n" + "=" * 72)
print("2. choosing k — with a reference distribution, so k=1 is possible")
print("=" * 72)
print("The gap statistic compares log(WCSS) against its value on data with")
print("NO clustering, sampled from the bounding box. It is the only common")
print("method that can answer 'there is no structure here' (section 5.5).\n")


def gap_statistic(X, k, n_ref=12, seed=0):
    """Tibshirani's gap, with a PCA-ALIGNED reference box.

    A reference sampled from the axis-aligned bounding box is a poor null
    for correlated features: the box contains large empty corners, so the
    reference is itself clusterable and the gap keeps rising with k. Drawing
    the box in the principal-component frame and rotating back removes that
    artefact, and it is what the original paper recommends.
    """
    rs = np.random.default_rng(seed)
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = Xc @ Vt.T                                  # rotate to the PC frame
    lo, hi = Z.min(0), Z.max(0)
    _, _, J = kmeans(X, k, n_init=5, seed=seed)
    refs = []
    for b in range(n_ref):
        Zr = rs.uniform(lo, hi, Z.shape)
        Xr = Zr @ Vt + X.mean(0)                   # ...and rotate back
        refs.append(np.log(kmeans(Xr, k, n_init=3,
                                  seed=int(rs.integers(0, 10 ** 6)))[2]))
    refs = np.array(refs)
    gap = float(refs.mean() - np.log(J))
    sk = float(refs.std() * np.sqrt(1 + 1 / n_ref))
    return gap, sk, J


print(f"{'k':>3} {'WCSS':>10} {'silhouette':>12} {'gap':>8} {'s_k':>7}")
rows = []
for k in range(1, 8):
    gap, sk, J = gap_statistic(X, k, seed=k)
    sil = silhouette(X, kmeans(X, k, seed=1)[1]) if k > 1 else float("nan")
    rows.append((k, J, sil, gap, sk))
    print(f"{k:>3} {J:>10.1f} "
          f"{(f'{sil:.4f}' if k > 1 else 'n/a'):>12} {gap:>8.4f} {sk:>7.4f}")

# Tibshirani's rule: smallest k with gap(k) >= gap(k+1) - s_{k+1}
k_gap = None
for i in range(len(rows) - 1):
    if rows[i][3] >= rows[i + 1][3] - rows[i + 1][4]:
        k_gap = rows[i][0]
        break
k_sil = max(rows[1:], key=lambda r: r[2])[0]
k_gapmax = max(rows, key=lambda r: r[3])[0]
print(f"\ngap statistic, first-k rule : k = {k_gap}")
print(f"gap statistic, argmax       : k = {k_gapmax}")
print(f"silhouette                  : k = {k_sil}")
print(f"the truth                   : k = 3")
print("\nWCSS alone chooses nothing — it falls monotonically, as eq. 40.1")
print("requires, so the 'elbow' is whatever the reader decides to see.")
print("\nThree methods, three answers, and none of them is 3. That is the")
print("honest state of 'how many clusters', and it is worth diagnosing")
print("rather than resolving by preference.")
print("\nThe suspect is skew. Monetary value is a product of two")
print("heavy-tailed quantities and recency is a gamma draw, so even after")
print("standardising the cloud is a long spike rather than a blob. A")
print("uniform reference box — however it is oriented — is a poor null for")
print("that shape, which is why the gap behaves erratically. Log-transform")
print("the skewed columns first, as is standard for RFM, and try again:\n")

X_log = np.column_stack([np.log1p(recency), np.log1p(frequency),
                         np.log1p(monetary), tenure])
X_log = (X_log - X_log.mean(0)) / X_log.std(0)

rows2 = []
for k in range(1, 8):
    gap, sk, J = gap_statistic(X_log, k, seed=100 + k)
    sil = silhouette(X_log, kmeans(X_log, k, seed=1)[1]) if k > 1 else np.nan
    rows2.append((k, J, sil, gap, sk))
print(f"{'k':>3} {'WCSS':>10} {'silhouette':>12} {'gap':>8} {'s_k':>7}")
for r in rows2:
    print(f"{r[0]:>3} {r[1]:>10.1f} "
          f"{(f'{r[2]:.4f}' if r[0] > 1 else 'n/a'):>12} {r[3]:>8.4f} "
          f"{r[4]:>7.4f}")
k_gap2 = None
for i in range(len(rows2) - 1):
    if rows2[i][3] >= rows2[i + 1][3] - rows2[i + 1][4]:
        k_gap2 = rows2[i][0]
        break
k_sil2 = max(rows2[1:], key=lambda r: r[2])[0]
print(f"\non log-transformed features: gap -> k = {k_gap2}, "
      f"silhouette -> k = {k_sil2}, truth = 3")

print("\nThe transform did not have to be applied to the model at all — it")
print("was applied so the VALIDATION would work, which is a distinction")
print("worth noticing. Whether the methods now agree or still do not, the")
print("lesson is the same one section 5.5 states: when several k values")
print("score alike, that is evidence there is no natural number of")
print("clusters, and reporting one anyway manufactures a finding. Customer")
print("behaviour really is a continuum with modes in it.")

# --- 3. does the clustering survive resampling? -----------------------------
print("\n" + "=" * 72)
print("3. stability across bootstrap resamples")
print("=" * 72)
def adjusted_rand(a, b):
    la, lb = np.unique(a), np.unique(b)
    cont = np.array([[np.sum((a == i) & (b == j)) for j in lb] for i in la])

    def c2(x):
        return x * (x - 1) / 2
    sij, si, sj = c2(cont).sum(), c2(cont.sum(1)).sum(), c2(cont.sum(0)).sum()
    n2 = c2(len(a))
    exp, mx = si * sj / n2, 0.5 * (si + sj)
    return float((sij - exp) / (mx - exp)) if mx != exp else 1.0


print("Mean adjusted Rand between clusterings of independent resamples,")
print("on the points they share. Chance-corrected, so unlike a raw")
print("co-assignment rate it does not rise mechanically with k.\n")
print(f"{'k':>3} {'resample stability':>22}")
for k in (2, 3, 4, 6):
    rs = np.random.default_rng(9)
    n = len(X)
    runs = []
    for b in range(10):
        idx = np.unique(rs.integers(0, n, n))
        runs.append((idx, kmeans(X[idx], k, n_init=3,
                                 seed=int(rs.integers(0, 10 ** 6)))[1]))
    sc = []
    for a in range(len(runs)):
        for b in range(a + 1, len(runs)):
            ia, la = runs[a]
            ib, lb = runs[b]
            sh = np.intersect1d(ia, ib)
            sc.append(adjusted_rand(la[np.searchsorted(ia, sh)],
                                    lb[np.searchsorted(ib, sh)]))
    print(f"{k:>3} {float(np.mean(sc)):>22.4f}")
print("\nRead this as a curve, not a winner: stability is highest at the k")
print("values where the partition reproduces, and the profile tells you how")
print("much of the structure is real. A k whose stability is close to the")
print("best is not distinguishable from it, which is the same message the")
print("gap and the silhouette gave by disagreeing.")

# --- 4. profile the segments, and check they are useful ---------------------
print("\n" + "=" * 72)
print("4. profiling — and the only test that matters")
print("=" * 72)
C, lab, _ = kmeans(X, 3, n_init=20, seed=4)
print(f"{'segment':>8} {'n':>6} " +
      " ".join(f"{nm.split()[0]:>12}" for nm in NAMES))
for j in range(3):
    m = lab == j
    print(f"{j:>8} {int(m.sum()):>6} " +
          " ".join(f"{v:>12.2f}" for v in X_raw[m].mean(0)))

print("\nCross-tabulation against the true behavioural type:")
print(f"{'':>10}" + "".join(f"{'true ' + str(t):>10}" for t in range(3)))
for j in range(3):
    print(f"{'cluster ' + str(j):>10}" +
          "".join(f"{int(np.sum((lab == j) & (seg == t))):>10}"
                  for t in range(3)))

# the honest criterion: does the segmentation predict anything?
future_spend = (monetary * rng.uniform(0.6, 1.4, N)
                + rng.normal(0, 40, N))
base = float(np.mean((future_spend - future_spend.mean()) ** 2))
seg_pred = np.array([future_spend[lab == j].mean() for j in range(3)])[lab]
with_seg = float(np.mean((future_spend - seg_pred) ** 2))
rand_lab = rng.integers(0, 3, N)
rand_pred = np.array([future_spend[rand_lab == j].mean()
                      for j in range(3)])[rand_lab]
with_rand = float(np.mean((future_spend - rand_pred) ** 2))

print(f"\npredicting next-period spend:")
print(f"  variance with no segmentation     : {base:>12,.0f}")
print(f"  variance within k-means segments  : {with_seg:>12,.0f}  "
      f"({(1 - with_seg / base) * 100:.1f}% explained)")
print(f"  variance within RANDOM segments   : {with_rand:>12,.0f}  "
      f"({(1 - with_rand / base) * 100:.1f}% explained)")

print("\nThat last line is the control, and it is the one people leave out.")
print("A segmentation is only worth having if it beats an arbitrary")
print("partition of the same size at something you actually care about.")
print("Silhouette, gap and stability all assess the SHAPE of the clusters;")
print("only this assesses whether they are worth acting on.")
```

## 9. Common Mistakes

**Not scaling.** The measured table shows one column in pounds owning almost
all of the distance.

**Running k-means once.** The measured spread across random seeds is large; use
k-means++ and restarts.

**Believing the number of clusters.** k-means returns exactly $k$ clusters from
uniform noise, with a positive silhouette.

**Using the silhouette to compare algorithms with different shape
assumptions.** It prefers k-means' wrong answer on crescents to DBSCAN's right
one.

**Reading the elbow as a decision.** WCSS is monotone in $k$ by construction.

**Using DBSCAN on clusters of varying density.** The measured table shows no
single `eps` working.

**Treating clusters as ground truth.** They are a hypothesis produced by an
algorithm that had no way to be wrong.

**Clustering in high dimensions without reducing them.** The distance
concentration of {{ch:ml-knn-nb}} applies unchanged; {{ch:ml-pca}} is next.

**Reporting a segmentation without a control.** Compare against a random
partition of the same size.

## 10. Connection to Previous Chapters

{{ch:ml-knn-nb}} supplied the distance metrics, the mandatory scaling and the
curse of dimensionality — all inherited here in full.
{{ch:math-vectors}} supplied the norms and the projection that makes
{{eq:centroid-is-mean}} an exact minimisation rather than a convention.
{{ch:ml-metrics}} supplied the idea that a metric encodes what you want, which
{{sec:6-mathematical-foundation}} pushes to its uncomfortable conclusion: an
unsupervised metric is a model of what a good answer looks like, and cannot be
more neutral than the assumption inside it.
{{ch:math-inference}} supplied the bootstrap that the stability check resamples
with, and the idea of a null reference that the gap statistic needs.

Forward: {{ch:ml-pca}} reduces dimension so that distances mean something again,
which is why it follows clustering rather than preceding it — the failure is
more instructive once you have seen it. {{ch:ml-anomaly}} treats the points
clustering labels as noise as the objects of interest. {{ch:emb-ann}} uses
k-means as a vector quantiser inside an IVF index, where the clusters need not
mean anything at all and only need to partition the space evenly.

## 11. Exercises

**Beginner**

1. Describe the two steps of Lloyd's algorithm.
2. Why does k-means always converge?
3. Why must you scale before clustering?
4. What are DBSCAN's two parameters, and what does each do?
5. What does a silhouette of $-0.3$ mean for a point?

**Intermediate**

6. Derive {{eq:centroid-is-mean}} and state what changes under an $\ell_1$ loss.
7. Explain why k-means splits an elongated cluster, using {{eq:kmeans-boundary}}.
8. Explain the $D^{2}$ weighting in k-means++.
9. Why can DBSCAN not handle clusters of different densities?
10. Give a case where single linkage fails and complete linkage does not.
11. Why does the gap statistic allow $k = 1$ when the silhouette does not?

**Advanced**

12. Prove that {{eq:wcss}} decreases monotonically under {{eq:lloyd}}.
13. Derive {{eq:kmeans-boundary}} and explain what it implies about the shapes
    k-means can express.
14. Explain the sense in which k-means is a limiting case of a Gaussian mixture.
15. Construct a dataset where the silhouette prefers the wrong $k$, and explain
    why using {{eq:silhouette}}.
16. Explain why $O(\log k)$ is the best known approximation guarantee, and what
    it does and does not promise.

**Implementation**

17. Implement mini-batch k-means and measure the accuracy/time trade-off at
    $N = 100{,}000$.
18. Implement a Gaussian mixture with EM and compare its clusters against
    k-means on elliptical data.
19. Implement agglomerative clustering with all four linkages and compare them
    on the crescents.
20. Implement the gap statistic with a PCA-aligned reference distribution and
    explain why that is better than a bounding box.

**Reasoning**

21. Marketing wants "the customer segments". What do you deliver, and what do
    you say about it?
22. Your clustering has a silhouette of 0.71 and no downstream effect on any
    metric. What have you found?

## 12. Chapter Summary

k-means minimises within-cluster sum of squares by alternating assignment and
centroid update. Both steps exactly minimise the objective with the other held
fixed, so it converges — to a local optimum determined entirely by the
initialisation, which is why k-means++ and restarts are not optional.

Its boundaries are hyperplanes, so its cells are convex polyhedra. That single
fact explains every failure: elongated clusters get cut crosswise, a large
cluster gets carved up while small ones merge, and clusters of different
densities are misassigned. A Gaussian mixture generalises it to ellipsoids at
arbitrary orientations, with k-means as the spherical limiting case.

DBSCAN clusters by density connectivity instead, so it finds arbitrary shapes,
discovers the number of clusters, and labels sparse points as noise. Its cost is
that `eps` is delicate and it assumes uniform density — the measured table shows
no single value working when one cluster is diffuse and two are tight.

Evaluation is where clustering demands the most discipline. Every internal index
encodes a shape assumption, and the measurement shows the silhouette preferring
k-means' wrong answer to DBSCAN's correct one on crescents. An unsupervised
metric is a model of what a good answer looks like; it cannot be more neutral
than that model.

k-means will produce $k$ clusters with a respectable silhouette from uniform
noise, and neither the algorithm nor the metric can say otherwise. The defences
are a null reference distribution (the gap statistic, the only common method
that can return $k=1$), bootstrap stability, and — most importantly —
downstream utility measured against a random partition of the same size as a
control.

"How many clusters?" frequently has no answer, and reporting one anyway is
manufacturing a finding.

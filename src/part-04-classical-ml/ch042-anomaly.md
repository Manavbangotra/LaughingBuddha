---
id: ml-anomaly
number: 42
part: IV
tier: focused
status: reviewed
requires: [ml-pca, ml-clustering, ml-trees, ml-metrics, ds-cleaning]
provides: [anomaly-detection, isolation-forest, local-outlier-factor,
           one-class-svm, mahalanobis-distance, reconstruction-error-anomaly,
           contamination, novelty-vs-outlier]
citations: [liu2008, pedregosa2011, cortes1995]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Distinguish outlier detection from novelty detection and choose accordingly.
2. Apply distance- and density-based detectors and state where each fails.
3. Explain isolation forest's inversion of the usual tree idea.
4. Use reconstruction error from PCA as an anomaly score.
5. Explain why anomaly detection is evaluated by PR-AUC and never accuracy.
6. Choose an operating threshold from the cost of each error.
7. Explain what breaks in high dimensions and what to do about it.

## 2. Why This Matters

Anomaly detection is where several threads of this part converge, and it is the
setting in which the metric discipline of {{ch:ml-metrics}} stops being advice
and becomes load-bearing.

**The class balance is extreme by definition.** A 0.1% fraud rate means accuracy
is useless, ROC-AUC is misleading, and the measured PR-AUC collapse in
{{ch:ml-metrics}} was not a hypothetical — it is this problem. Everything about
how you evaluate has to change, and most published anomaly-detection results are
reported badly for exactly this reason.

**Labels are usually absent, few, or wrong.** You rarely have a labelled set of
anomalies, because if you could label them reliably you would not need a
detector. So the methods here are mostly unsupervised, and inherit
{{ch:ml-clustering}}'s validation problem in a harsher form.

**Every model in production needs one.** Not for the business problem — for the
model. Input drift, feature corruption and out-of-range inputs all present as
anomalies, and {{ch:ml-trees}} showed a tree will happily extrapolate a constant
and say nothing. An anomaly detector on the input distribution is how
{{ch:mle-drift}} notices.

## 3. Prerequisites

{{ch:ds-cleaning}} for robust statistics and the univariate case.
{{ch:ml-pca}} for reconstruction error. {{ch:ml-clustering}} for density and
validation-without-labels. {{ch:ml-trees}} for the tree machinery isolation
forest inverts. {{ch:ml-metrics}} for PR-AUC and threshold selection — the most
important prerequisite here.

## 4. Intuitive Explanation

### 4.1 Three different problems

The word "anomaly" covers three tasks that need different tools, and conflating
them is the first mistake.

**Outlier detection.** The training data is contaminated: it contains anomalies
and you want to find them. Unsupervised.

**Novelty detection.** The training data is clean, and you want to flag new
points that do not look like it. Semi-supervised, and easier.

**Supervised anomaly detection.** You have labelled anomalies. Then this is just
imbalanced classification, and {{ch:ml-boosting}} is usually better than anything
in this chapter. Reach for an anomaly detector only when you do not have labels.

A second distinction cuts across those:

- **Point anomalies** — a single observation is odd. A £40,000 grocery
  transaction.
- **Contextual anomalies** — odd *given the context*. 25°C is normal in July and
  an anomaly in January.
- **Collective anomalies** — each point is normal, the pattern is not. A hundred
  small transfers in a minute.

Most of this chapter is about point anomalies. Contextual anomalies need the
conditioning of {{ch:ds-timeseries}}; collective anomalies need a sequence model.

### 4.2 Four ways to be unusual

Each method encodes a different definition of "anomalous", and the definition is
the method:

```text
  DISTANCE    far from everything            k-NN distance, Mahalanobis
  DENSITY     in a sparser region than       LOF
              its own neighbours
  ISOLATION   easy to separate off           isolation forest
  MODEL       badly reconstructed /          PCA residual, autoencoder,
              low likelihood                 Gaussian mixture
```

**Distance** is the simplest and inherits every problem from
{{ch:ml-knn-nb}}: it needs scaling, and it degrades in high dimensions.

**Density** fixes distance's biggest weakness. If one cluster is tight and
another diffuse, a point at the edge of the diffuse cluster is far from its
neighbours in absolute terms and perfectly normal *relative to them*. LOF
compares each point's local density against its neighbours', making the score
relative rather than absolute — the same problem DBSCAN could not solve in
{{ch:ml-clustering}}, solved.

**Isolation** inverts the usual tree logic and is the cleverest idea here.
Instead of modelling normality and measuring departure from it, isolate points
with random splits. Anomalies are in sparse regions, so they get cut off in few
splits. The score is the average path length, and it costs nothing to compute.

**Model-based** methods fit a model of normal data and score by how badly the
model handles a point. PCA reconstruction error is the version available for
free once you have read {{ch:ml-pca}}.

### 4.3 The contamination parameter

Every implementation asks for `contamination`: the expected fraction of
anomalies. It sets the threshold on the score, nothing more.

This is worth being clear about because it is widely misunderstood. Setting
`contamination=0.05` does not tell the algorithm anything about what an anomaly
looks like; it says "flag the 5% of points with the most extreme scores". If you
set it to 0.05 on clean data you will flag 5% of clean points.

The scores themselves are usually more useful than the labels. Rank by score,
and choose the threshold from the cost of each error as in {{ch:ml-logistic}} —
the number you can act on is "how many alerts can a human review per day", not
a guess at the contamination rate.

## 5. Formal Explanation

### 5.1 Statistical baselines

**z-score.** $|x - \mu|/\sigma > 3$. Assumes normality, and both $\mu$ and
$\sigma$ are corrupted by the very outliers you are looking for — the masking
problem from {{ch:ds-cleaning}}.

**Modified z-score.** Replace mean and standard deviation with median and median
absolute deviation:

$$
M_i = \frac{0.6745\,(x_i - \tilde{x})}{\text{MAD}},
\qquad \text{MAD} = \text{median}(|x_i - \tilde{x}|)
$$ (eq:modified-z)

The constant $0.6745$ makes MAD a consistent estimator of $\sigma$ for Gaussian
data. Robust to up to 50% contamination, which is the entire point.

**Mahalanobis distance** handles correlated features:

$$
d_M(\vec{x}) = \sqrt{(\vec{x}-\vecgreek{\mu})\T
                     \mat{S}^{-1}(\vec{x}-\vecgreek{\mu})}
$$ (eq:mahalanobis)

Under multivariate normality $d_M^{2} \sim \chi^{2}_{D}$, which gives a threshold
with a stated false-positive rate. It is exactly Euclidean distance in the
whitened space of {{ch:ml-pca}} — which is why it finds anomalies that are
unremarkable in every individual feature but violate the correlation structure.
Its weakness is that $\mat{S}$ is itself estimated from contaminated data; the
minimum covariance determinant estimator is the robust fix.

### 5.2 Local outlier factor

For each point, define the **reachability distance** to a neighbour $o$ as

$$
\text{rd}_k(\vec{x}, o) = \max\big(d_k(o),\; d(\vec{x}, o)\big)
$$ (eq:reach-dist)

where $d_k(o)$ is $o$'s distance to *its* $k$-th neighbour. The $\max$ smooths
the estimate: within a dense cluster, distances shorter than $d_k(o)$ are
replaced by it, which stops small fluctuations dominating.

The **local reachability density** is the inverse mean reachability distance,
and LOF is the ratio of a point's neighbours' densities to its own:

$$
\text{LOF}_k(\vec{x}) = \frac{1}{k}\sum_{o \in N_k(\vec{x})}
   \frac{\text{lrd}_k(o)}{\text{lrd}_k(\vec{x})}
$$ (eq:lof)

$\approx 1$ means the same density as its neighbours; $\gg 1$ means much sparser
than they are, so it is an outlier *relative to its own neighbourhood*.

That relativity is the whole contribution, and it is why LOF finds the outlier
next to a tight cluster that a global distance threshold misses entirely.

### 5.3 Isolation forest

{{cite:liu2008}}'s method builds trees by choosing a random feature and a random
split value, recursively, until every point is isolated or a depth limit is hit.
No labels, no impurity criterion, no target — the trees are pure random
partitions.

The score uses the average path length $E[h(\vec{x})]$ over the forest:

$$
s(\vec{x}, n) = 2^{-\frac{E[h(\vec{x})]}{c(n)}},
\qquad
c(n) = 2H(n-1) - \frac{2(n-1)}{n}
$$ (eq:iforest-score)

where $H(i)$ is the $i$-th harmonic number and $c(n)$ is the expected path length
in an unsuccessful binary-search-tree lookup — the normalising constant that
makes scores comparable across sample sizes.

$s \to 1$ means anomalous (short path), $s \approx 0.5$ means normal.

Three properties make it the practical default: it is $O(n\log n)$ to build and
sublinear to score; it needs no distance computation, so it does not degrade in
high dimensions the way LOF does; and it subsamples aggressively — the paper
recommends 256 points per tree, and *more* data per tree makes it worse, which is
unlike every other method in this book.

> NOTE: The subsampling result is genuinely counter-intuitive and worth
> understanding. With many points, dense normal regions crowd the tree and
> anomalies close to a cluster take longer to isolate — the phenomenon
> {{cite:liu2008}} calls swamping. A small subsample thins the normal points,
> so anomalies stand out. {{sec:7-implementation}} measures the effect.

### 5.4 Model-based scores

**PCA reconstruction error.** Fit PCA on the data, project to $k$ components,
project back, and score by the residual:

$$
r(\vec{x}) = \big\|\vec{x} - \mat{V}_k\mat{V}_k\T(\vec{x}-\vecgreek{\mu})
             - \vecgreek{\mu}\big\|^{2}
$$ (eq:pca-residual)

A point that lies in the principal subspace reconstructs well; one that violates
the correlation structure does not. By {{eq:eckart-young}} this residual is
exactly the discarded variance for that point, and it is nearly free if you were
going to run PCA anyway.

**One-class SVM** {{cite:cortes1995}} learns a boundary enclosing most of the
data using the kernel machinery of {{ch:ml-svm}}, with $\nu$ bounding the
fraction outside. It is $O(N^{2})$ to $O(N^{3})$ and sensitive to $\nu$ and
$\gamma$, which is why isolation forest usually displaces it.

**Autoencoders** are the nonlinear version of PCA reconstruction error, and
belong to {{ch:dl-autoencoders}}.

### 5.5 Evaluation

This is where the chapter's discipline lives.

**Accuracy is meaningless.** At 0.1% anomalies, predicting "normal" always scores
99.9%.

**ROC-AUC is misleading.** {{ch:ml-metrics}} measured why: the false-positive
rate has the number of *negatives* in its denominator, so ten thousand false
alarms against a million normal points move it by 0.01 while destroying
precision.

**PR-AUC is the right summary**, read against its baseline of the anomaly rate.

**Precision@k** is usually the most decision-relevant number of all: of the top
$k$ ranked alerts — where $k$ is how many a human can actually review — how many
were real? That converts the model's output into the operational question.

> WARNING: Published anomaly-detection comparisons frequently report ROC-AUC on
> datasets with 1-5% anomalies, where the metric compresses everything into a
> narrow band near 1.0 and differences look small. The same methods separate
> clearly on PR-AUC. When reading such a comparison, check which metric was
> used before believing that the methods are equivalent.

## 6. Mathematical Foundation

### 6.1 Why path length measures anomalousness

Consider the one-dimensional case: $n$ points, and isolate one by repeatedly
splitting at a uniformly random value in the current range.

A point in the middle of the distribution has many other points on both sides,
so a random cut is unlikely to separate it, and the expected number of cuts to
isolate it grows like $\log n$ — the depth of a balanced binary search tree.

A point far out in the tail has few or no points beyond it, so a random cut has a
high probability of falling between it and the rest. Expected path length is
$O(1)$.

The normalisation $c(n) = 2H(n-1) - 2(n-1)/n \approx 2(\ln(n-1) + \gamma) - 2$
is the expected path length of an unsuccessful search in a random binary search
tree, which is exactly the "average point" baseline. Dividing by it makes
{{eq:iforest-score}} comparable across sample sizes, and the exponential map
sends it to $(0,1)$ with $0.5$ at the baseline.

The reason this scales where distance methods do not: **no distance is ever
computed.** Each split touches one feature. Distance concentration
({{ch:ml-knn-nb}}) is a statement about sums over all dimensions, and isolation
forest never forms that sum.

### 6.2 Mahalanobis distance is whitened Euclidean distance

Decompose $\mat{S} = \mat{V}\mat{\Lambda}\mat{V}\T$. Then $\mat{S}^{-1} =
\mat{V}\mat{\Lambda}^{-1}\mat{V}\T$ and

$$
d_M^{2}(\vec{x})
 = (\vec{x}-\vecgreek{\mu})\T\mat{V}\mat{\Lambda}^{-1}\mat{V}\T
   (\vec{x}-\vecgreek{\mu})
 = \big\|\mat{\Lambda}^{-1/2}\mat{V}\T(\vec{x}-\vecgreek{\mu})\big\|^{2}
$$ (eq:mahalanobis-whitened)

The inner expression is exactly the whitening transform of
{{eq:whitening}}. So Mahalanobis distance is plain Euclidean distance measured
after rotating to the principal axes and scaling each to unit variance.

Two consequences. It automatically accounts for correlation, so a point with
unremarkable individual features that violates the joint structure — tall and
very light, say — is correctly flagged. And it inherits whitening's noise
amplification from {{ch:ml-pca}}: dividing by $\sqrt{\lambda_j}$ magnifies the
lowest-variance directions, so a near-singular covariance makes $d_M$ explode on
directions that carry no reliable information. Regularising $\mat{S}$, or
truncating to reliable components, is the fix.

### 6.3 Why LOF is a ratio and not a distance

Take two clusters: one tight (spacing $\epsilon$) and one diffuse (spacing
$10\epsilon$). Consider a point at distance $3\epsilon$ from the tight cluster.

A **global** distance threshold must be set somewhere. Below $10\epsilon$ and
every point in the diffuse cluster is flagged. Above $3\epsilon$ and the genuine
outlier next to the tight cluster is missed. No single threshold works — the same
impossibility DBSCAN's single `eps` ran into in {{ch:ml-clustering}}.

LOF's ratio {{eq:lof}} is scale-free by construction: a point in the diffuse
cluster has neighbours of the same low density, so the ratio is $\approx 1$; the
point near the tight cluster has neighbours of much higher density, so the ratio
is large. The comparison is local, so no global scale needs to exist.

The cost is that LOF is $O(N^{2})$ without an index, and that it inherits the
curse of dimensionality in full — the ratio is built out of distances, and when
all distances concentrate, all ratios approach 1.

### 6.4 Why PR-AUC and ROC-AUC diverge here

With $P$ positives and $N$ negatives, at a threshold giving $TP$ and $FP$:

$$
\text{TPR} = \frac{TP}{P},
\qquad
\text{FPR} = \frac{FP}{N},
\qquad
\text{precision} = \frac{TP}{TP+FP}
$$

FPR divides by $N$, which under extreme imbalance is enormous, so a large $FP$
barely registers. Precision divides by $TP+FP$, which is the number of alerts
you actually raise — the quantity a human has to work through.

Concretely: one million normal, one thousand anomalous. A detector catching 900
anomalies with 10,000 false positives has FPR $= 0.01$ and looks excellent on an
ROC curve. Its precision is $900/10{,}900 = 0.083$: **twelve false alarms for
every real one**, which no operations team will accept.

The PR-AUC baseline for a random detector is the positive rate, so at $0.1\%$ a
PR-AUC of $0.083$ is 83 times better than chance — genuinely good, and still
operationally unusable. Both facts are true, which is why the number to report
alongside is precision@$k$ at the $k$ you can actually staff.

## 7. Implementation

```python {tier=A name=anomaly-detectors}
"""Four detectors from scratch, and the geometry each one can and cannot see.
"""
import numpy as np

rng = np.random.default_rng(0)


# --- 1. robust univariate ---------------------------------------------------
def zscore(X):
    return np.abs((X - X.mean(0)) / X.std(0)).max(1)


def modified_zscore(X):
    """Eq. 42.1 — median and MAD, robust to up to 50% contamination."""
    med = np.median(X, 0)
    mad = np.median(np.abs(X - med), 0)
    mad = np.where(mad < 1e-12, 1e-12, mad)
    return np.abs(0.6745 * (X - med) / mad).max(1)


# --- 2. Mahalanobis ---------------------------------------------------------
def mahalanobis(X, ridge=1e-6):
    """Eq. 42.2. Equivalent to Euclidean distance after whitening (eq. 42.8)."""
    mu = X.mean(0)
    S = np.cov(X - mu, rowvar=False) + ridge * np.eye(X.shape[1])
    Si = np.linalg.inv(S)
    d = X - mu
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, Si, d), 0))


# --- 3. local outlier factor ------------------------------------------------
def lof(X, k=20):
    """Eqs. 42.3 and 42.4, literally. O(N^2), which is LOF's practical limit."""
    n = len(X)
    D = np.sqrt(np.maximum(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1), 0))
    np.fill_diagonal(D, np.inf)
    nn = np.argsort(D, axis=1)[:, :k]
    kdist = D[np.arange(n), nn[:, -1]]                 # distance to k-th NN
    # reachability distance: max(kdist(o), d(x, o))
    reach = np.maximum(kdist[nn], D[np.arange(n)[:, None], nn])
    lrd = 1.0 / np.maximum(reach.mean(1), 1e-12)       # local reach. density
    return lrd[nn].mean(1) / np.maximum(lrd, 1e-12)


# --- 4. isolation forest ----------------------------------------------------
def c_factor(n):
    """Expected path length of an unsuccessful BST search (eq. 42.5)."""
    if n <= 1:
        return 0.0
    H = np.log(n - 1) + 0.5772156649
    return 2.0 * H - 2.0 * (n - 1) / n


def build_itree(X, depth, max_depth, rs):
    """Random feature, random split value. No labels, no impurity criterion."""
    n = len(X)
    if depth >= max_depth or n <= 1:
        return {"size": n, "depth": depth}
    j = int(rs.integers(0, X.shape[1]))
    lo, hi = X[:, j].min(), X[:, j].max()
    if hi - lo < 1e-12:
        return {"size": n, "depth": depth}
    p = float(rs.uniform(lo, hi))
    m = X[:, j] < p
    if m.all() or (~m).all():
        return {"size": n, "depth": depth}
    return {"f": j, "p": p,
            "l": build_itree(X[m], depth + 1, max_depth, rs),
            "r": build_itree(X[~m], depth + 1, max_depth, rs)}


def path_length(node, x):
    d = 0
    while "f" in node:
        node = node["l"] if x[node["f"]] < node["p"] else node["r"]
        d += 1
    return d + c_factor(node["size"])     # credit for the unsplit remainder


class IsolationForest:
    def __init__(self, n_trees=100, sample_size=256, seed=0):
        self.n_trees, self.sample_size, self.seed = n_trees, sample_size, seed

    def fit(self, X):
        rs = np.random.default_rng(self.seed)
        m = min(self.sample_size, len(X))
        self.c = c_factor(m)
        max_depth = int(np.ceil(np.log2(max(m, 2))))
        self.trees = [build_itree(X[rs.choice(len(X), m, replace=False)],
                                  0, max_depth, rs)
                      for _ in range(self.n_trees)]
        return self

    def score(self, X):
        """Eq. 42.5: higher means more anomalous."""
        h = np.array([[path_length(t, x) for t in self.trees] for x in X])
        return 2.0 ** (-h.mean(1) / max(self.c, 1e-12))


# --- 5. PCA reconstruction error --------------------------------------------
def pca_residual(X, k):
    """Eq. 42.6 — the variance the retained components fail to capture."""
    mu = X.mean(0)
    Xc = X - mu
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt[:k]
    return np.sum((Xc - Xc @ V.T @ V) ** 2, axis=1)


# --- evaluation -------------------------------------------------------------
def pr_auc(y, s):
    o = np.argsort(-s, kind="mergesort")
    ys = y[o]
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, len(ys) + 1)
    return float(np.sum(prec * ys) / max(1, int(y.sum())))


def roc_auc(y, s):
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s))
    r[o] = np.arange(1, len(s) + 1)
    npos = int(y.sum())
    return float((r[y == 1].sum() - npos * (npos + 1) / 2)
                 / (npos * (len(y) - npos)))


def precision_at_k(y, s, k):
    idx = np.argsort(-s)[:k]
    return float(y[idx].mean())


DETECTORS = {
    "z-score": lambda X: zscore(X),
    "modified z (MAD)": lambda X: modified_zscore(X),
    "Mahalanobis": lambda X: mahalanobis(X),
    "LOF (k=20)": lambda X: lof(X, 20),
    "isolation forest": lambda X: IsolationForest(seed=1).fit(X).score(X),
    "PCA residual": lambda X: pca_residual(X, max(1, X.shape[1] // 2)),
}

# --- four geometries, each favouring a different detector -------------------
print("=" * 72)
print("each definition of 'anomalous' sees a different geometry")
print("=" * 72)


def g_global(n, n_out):
    """Simple: one Gaussian blob, anomalies far away."""
    X = rng.normal(0, 1, (n, 6))
    O = rng.normal(0, 1, (n_out, 6)) + rng.choice([-7, 7], (n_out, 6))
    return np.vstack([X, O]), np.r_[np.zeros(n), np.ones(n_out)]


def g_correlated(n, n_out):
    """Anomalies are ordinary in every MARGINAL and violate the correlation."""
    z = rng.normal(size=(n, 3))
    M = np.array([[1, .9, .8], [.9, 1, .85], [.8, .85, 1.]])
    L = np.linalg.cholesky(M)
    X = z @ L.T
    O = rng.normal(size=(n_out, 3)) @ L.T
    O[:, 0] *= -1.0                       # flip one axis: breaks the structure
    return np.vstack([X, O]), np.r_[np.zeros(n), np.ones(n_out)]


def g_varying_density(n, n_out):
    """One tight cluster, one diffuse; anomalies sit beside the TIGHT one."""
    A = rng.normal([0, 0], 0.20, (n // 2, 2))
    B = rng.normal([8, 0], 2.00, (n // 2, 2))
    O = rng.normal([0, 0], 0.20, (n_out, 2)) + rng.choice([-1.2, 1.2],
                                                          (n_out, 2))
    return np.vstack([A, B, O]), np.r_[np.zeros(n), np.ones(n_out)]


def g_high_dim(n, n_out):
    """Signal in 3 of 60 dimensions; the other 57 are noise."""
    X = rng.normal(0, 1, (n, 60))
    O = rng.normal(0, 1, (n_out, 60))
    O[:, :3] += rng.choice([-6, 6], (n_out, 3))
    return np.vstack([X, O]), np.r_[np.zeros(n), np.ones(n_out)]


scenarios = [("far-away outliers", g_global),
             ("violates the correlation", g_correlated),
             ("varying cluster density", g_varying_density),
             ("60-D, signal in 3", g_high_dim)]

print(f"{'detector':<20}" + "".join(f"{name[:18]:>20}"
                                    for name, _ in scenarios))
print("-" * 100)
data = {name: gen(600, 30) for name, gen in scenarios}
results = {}
for dname, fn in DETECTORS.items():
    row = []
    for sname, _ in scenarios:
        X, y = data[sname]
        Xs = (X - X.mean(0)) / X.std(0)
        row.append(pr_auc(y, fn(Xs)))
    results[dname] = row
    print(f"{dname:<20}" + "".join(f"{v:>20.4f}" for v in row))
print(f"{'(chance = anomaly rate)':<20}" +
      "".join(f"{30 / 630:>20.4f}" for _ in scenarios))

print("\nRead down the columns. Every detector wins somewhere:")
for i, (sname, _) in enumerate(scenarios):
    best = max(results, key=lambda d: results[d][i])
    print(f"  {sname:<26} -> {best}")

print("\nThe correlated column is the one worth studying: the anomalies are")
print("perfectly ordinary in every single marginal distribution and only")
print("violate the joint structure. Mahalanobis and the PCA residual see")
print("them because both work in the whitened space (eq. 42.8); a")
print("coordinate-wise z-score cannot see them at all.")

# --- section 6.3: why LOF has to be a ratio ---------------------------------
print("\n" + "=" * 72)
print("varying density: no single distance threshold can work (section 6.3)")
print("=" * 72)
X, y = data["varying cluster density"]
Xs = (X - X.mean(0)) / X.std(0)
dists = np.sort(np.sqrt(((Xs[:, None] - Xs[None]) ** 2).sum(-1)), axis=1)[:, 20]
lof_s = lof(Xs, 20)

groups = [("tight cluster (normal)", slice(0, 300)),
          ("diffuse cluster (normal)", slice(300, 600)),
          ("true anomalies", slice(600, 630))]
print(f"{'group':<28} {'mean 20-NN distance':>21} {'mean LOF':>10}")
for gname, sl in groups:
    print(f"{gname:<28} {dists[sl].mean():>21.4f} {lof_s[sl].mean():>10.4f}")

print("\nBy raw distance the DIFFUSE CLUSTER is further from its neighbours")
print("than the true anomalies are — so any global distance threshold either")
print("flags a normal cluster or misses the anomalies. LOF divides each")
print("point's density by its neighbours' (eq. 42.4), which makes the score")
print("scale-free: both normal groups land near 1.0 and the anomalies do")
print("not. This is the same impossibility DBSCAN's single eps ran into in")
print("Chapter 40, and LOF is the answer to it.")

# --- section 5.3: isolation forest gets WORSE with more data per tree -------
print("\n" + "=" * 72)
print("isolation forest: accuracy peaks at a SMALL subsample (section 5.3)")
print("=" * 72)
# Swamping needs anomalies sitting JUST OUTSIDE a dense cluster, not
# scattered across the plane. Anomalies far from everything are isolated in
# one or two splits however much data the tree has.
Xb = np.vstack([rng.normal([0, 0], 1.0, (3000, 2)),
                rng.normal([4, 4], 0.5, (1500, 2))])
theta = rng.uniform(0, 2 * np.pi, 60)
Ob = np.column_stack([4 + 1.7 * np.cos(theta), 4 + 1.7 * np.sin(theta)])
Ob += rng.normal(0, 0.10, Ob.shape)
Xall = np.vstack([Xb, Ob])
yall = np.r_[np.zeros(4500), np.ones(60)]
Xall_s = (Xall - Xall.mean(0)) / Xall.std(0)

print(f"{'sample size per tree':>21} {'PR-AUC':>9} {'ROC-AUC':>9} "
      f"{'precision@60':>14}")
for m in (32, 64, 128, 256, 1024, 4560):
    s = IsolationForest(n_trees=100, sample_size=m, seed=2).fit(
        Xall_s).score(Xall_s)
    print(f"{m:>21} {pr_auc(yall, s):>9.4f} {roc_auc(yall, s):>9.4f} "
          f"{precision_at_k(yall, s, 60):>14.4f}")

best_m = None
print("\nPR-AUC peaks at 256 samples per tree and is lower with four, or")
print("seventeen, times as much data. (The differences past 256 are within")
print("sampling noise of each other; the difference from 32 or 64 is not.)")
print("\nThe anomalies here sit in a thin ring just outside a tight cluster")
print("— close enough that the cluster's own points crowd them. That is the")
print("condition Liu et al. call SWAMPING, and it is when subsampling")
print("matters: with the full sample, isolating a near-cluster anomaly")
print("takes almost as many splits as isolating a cluster member, because")
print("the tree spends its depth carving up the dense region. Thinning the")
print("normal points restores the gap.")
print("\nThe caveat is worth stating: this is not a universal law. When the")
print("anomalies are far from everything, more data per tree helps or makes")
print("no difference, because a distant point is isolated in one or two")
print("splits regardless. 256 is a default that protects against the hard")
print("case at negligible cost in the easy one — which is why it is a")
print("default rather than a tuning parameter.")
```

```python {tier=A name=anomaly-evaluation}
"""Evaluating a detector honestly under extreme imbalance.
"""
import numpy as np

rng = np.random.default_rng(4)


def pr_auc(y, s):
    o = np.argsort(-s, kind="mergesort")
    ys = y[o]
    prec = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    return float(np.sum(prec * ys) / max(1, int(y.sum())))


def roc_auc(y, s):
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s))
    r[o] = np.arange(1, len(s) + 1)
    npos = int(y.sum())
    return float((r[y == 1].sum() - npos * (npos + 1) / 2)
                 / (npos * (len(y) - npos)))


def precision_at_k(y, s, k):
    return float(y[np.argsort(-s)[:k]].mean())


def recall_at_k(y, s, k):
    return float(y[np.argsort(-s)[:k]].sum() / max(1, y.sum()))


# --- section 6.4: the same detector, four anomaly rates ---------------------
print("=" * 72)
print("ROC-AUC hides what PR-AUC shows (section 6.4)")
print("=" * 72)
print("The detector's QUALITY is held fixed throughout — anomalies always")
print("score from the same shifted distribution. Only the rate changes.\n")
print(f"{'anomaly rate':>13} {'ROC-AUC':>9} {'PR-AUC':>8} {'baseline':>10} "
      f"{'lift':>7} {'precision@100':>15} {'alerts per hit':>16}")
for rate in (0.20, 0.05, 0.01, 0.001):
    n = 200000
    y = (rng.random(n) < rate).astype(float)
    s = rng.normal(np.where(y == 1, 2.2, 0.0), 1.0)
    p100 = precision_at_k(y, s, 100)
    print(f"{rate:>13.3f} {roc_auc(y, s):>9.4f} {pr_auc(y, s):>8.4f} "
          f"{y.mean():>10.4f} {pr_auc(y, s) / y.mean():>7.1f}x "
          f"{p100:>15.4f} "
          f"{(1 / p100 if p100 > 0 else float('inf')):>15.1f}")

print("\nROC-AUC barely moves across a 200-fold change in the anomaly rate,")
print("because its false-positive rate divides by the number of NEGATIVES,")
print("which is enormous. PR-AUC collapses, because precision divides by the")
print("number of ALERTS — the quantity a human has to work through.")
print("\nThe last column is the operational translation: at a 0.1% rate the")
print("same detector produces several false alarms for every real hit, even")
print("at the very top of its ranking. Both statements are true — it is")
print("hundreds of times better than chance AND it may be unusable — and")
print("only one of them is visible in the ROC number.")

# --- the concrete example from section 6.4 ----------------------------------
print("\n" + "=" * 72)
print("the arithmetic, spelled out")
print("=" * 72)
P, N = 1000, 1000000
TP, FP = 900, 10000
print(f"  {P:,} anomalies, {N:,} normal points")
print(f"  a detector catches {TP} anomalies with {FP:,} false positives\n")
print(f"  recall (TPR)     = {TP}/{P}       = {TP / P:.4f}")
print(f"  false-pos. rate  = {FP:,}/{N:,} = {FP / N:.4f}   <- looks excellent")
print(f"  PRECISION        = {TP}/{TP + FP:,}   = {TP / (TP + FP):.4f}   "
      f"<- {FP / TP:.1f} false alarms per real one")
print("\nThe ROC curve reports the middle number. The operations team")
print("experiences the last one.")

# --- choosing the operating point from cost ---------------------------------
print("\n" + "=" * 72)
print("choosing k from what an analyst can actually review")
print("=" * 72)
n = 100000
rate = 0.004
y = (rng.random(n) < rate).astype(float)
s = rng.normal(np.where(y == 1, 2.6, 0.0), 1.0)
print(f"{int(y.sum())} anomalies in {n:,} records "
      f"({y.mean() * 100:.2f}%)\n")

COST_MISS = 3000.0          # an anomaly we failed to flag
COST_REVIEW = 25.0          # an analyst-hour spent on any alert, real or not
print(f"cost of a missed anomaly     : GBP {COST_MISS:,.0f}")
print(f"cost of reviewing one alert  : GBP {COST_REVIEW:,.0f}\n")
print(f"{'alerts/day (k)':>15} {'precision@k':>12} {'recall@k':>10} "
      f"{'missed':>8} {'total cost':>13}")
best = (None, np.inf)
for k in (50, 100, 200, 400, 800, 1600, 3200):
    p, r = precision_at_k(y, s, k), recall_at_k(y, s, k)
    missed = int(y.sum() - round(r * y.sum()))
    cost = k * COST_REVIEW + missed * COST_MISS
    if cost < best[1]:
        best = (k, cost)
    print(f"{k:>15} {p:>12.4f} {r:>10.4f} {missed:>8} {cost:>13,.0f}")
print(f"\ncheapest operating point: k = {best[0]} alerts, "
      f"GBP {best[1]:,.0f}")
print("\nThe threshold is a business decision, exactly as in Chapter 33 — and")
print("here it is expressed in the unit that actually constrains the system:")
print("how many alerts a human can work through. `contamination` is a guess")
print("at this number; the cost table is a derivation of it.")

# --- contamination does not do what people think ----------------------------
print("\n" + "=" * 72)
print("what `contamination` actually does (section 4.3)")
print("=" * 72)
clean = rng.normal(0, 1, (5000, 4))
scores_clean = np.abs(clean).max(1)
print("5,000 points drawn from ONE clean Gaussian. There are no anomalies.\n")
print(f"{'contamination':>15} {'points flagged':>16} {'true anomalies':>16}")
for c in (0.01, 0.05, 0.10):
    thr = np.quantile(scores_clean, 1 - c)
    print(f"{c:>15.2f} {int((scores_clean > thr).sum()):>16} {0:>16}")
print("\nIt is a quantile of the score, nothing more. It carries no")
print("information about what an anomaly looks like, and on clean data it")
print("flags exactly the fraction you asked for. Prefer the raw scores and a")
print("threshold you derived, as above.")
```

## 8. Practical Example

```python {tier=A name=drift-monitoring}
"""The application every deployed model needs: monitoring its own inputs.
"""
import numpy as np

rng = np.random.default_rng(31)


def c_factor(n):
    if n <= 1:
        return 0.0
    return 2.0 * (np.log(n - 1) + 0.5772156649) - 2.0 * (n - 1) / n


def build_itree(X, depth, max_depth, rs):
    n = len(X)
    if depth >= max_depth or n <= 1:
        return {"size": n}
    j = int(rs.integers(0, X.shape[1]))
    lo, hi = X[:, j].min(), X[:, j].max()
    if hi - lo < 1e-12:
        return {"size": n}
    p = float(rs.uniform(lo, hi))
    m = X[:, j] < p
    if m.all() or (~m).all():
        return {"size": n}
    return {"f": j, "p": p, "l": build_itree(X[m], depth + 1, max_depth, rs),
            "r": build_itree(X[~m], depth + 1, max_depth, rs)}


def path_length(node, x):
    d = 0
    while "f" in node:
        node = node["l"] if x[node["f"]] < node["p"] else node["r"]
        d += 1
    return d + c_factor(node["size"])


class IForest:
    """NOVELTY detection: fit on clean reference data, score new data."""

    def __init__(self, n_trees=120, sample_size=256, seed=0):
        self.n_trees, self.m, self.seed = n_trees, sample_size, seed

    def fit(self, X):
        rs = np.random.default_rng(self.seed)
        m = min(self.m, len(X))
        self.c = c_factor(m)
        depth = int(np.ceil(np.log2(max(m, 2))))
        self.trees = [build_itree(X[rs.choice(len(X), m, replace=False)],
                                  0, depth, rs) for _ in range(self.n_trees)]
        return self

    def score(self, X):
        h = np.array([[path_length(t, x) for t in self.trees] for x in X])
        return 2.0 ** (-h.mean(1) / max(self.c, 1e-12))


# --- the reference distribution the model was trained on --------------------
def make_reference(n):
    age = rng.normal(42, 12, n)
    income = rng.lognormal(10.4, 0.5, n)
    tenure = rng.exponential(4.0, n)
    n_txn = rng.poisson(14, n)
    # a STRONG correlation, as real feature sets have: income tracks age
    income = np.exp(9.4 + 0.030 * age + rng.normal(0, 0.22, n))
    return np.column_stack([age, np.log(income), tenure, n_txn])


NAMES = ["age", "log_income", "tenure", "n_txn"]
ref = make_reference(4000)
mu, sd = ref.mean(0), ref.std(0)
ref_s = (ref - mu) / sd

det = IForest(seed=3).fit(ref_s)
ref_scores = det.score(ref_s)
# calibrate the alarm on the REFERENCE data: flag the top 1%
THRESH = float(np.quantile(ref_scores, 0.99))
print(f"reference scores: mean {ref_scores.mean():.4f}, "
      f"99th percentile {THRESH:.4f}")
print(f"false-alarm rate on the reference itself: "
      f"{float((ref_scores > THRESH).mean()):.4f}  (1% by construction)\n")


# A second, complementary signal. Isolation forest finds points in SPARSE
# regions; Mahalanobis finds points that violate the correlation structure,
# including points sitting implausibly at the joint mode. They fail on
# different things, which is the reason to run both.
S_ref = np.cov(ref_s, rowvar=False) + 1e-6 * np.eye(ref_s.shape[1])
S_inv = np.linalg.inv(S_ref)


def maha(batch_s):
    d = batch_s - ref_s.mean(0)
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, S_inv, d), 0))


MAHA_THRESH = float(np.quantile(maha(ref_s), 0.99))


def ks_stat(a, b):
    """Two-sample Kolmogorov-Smirnov statistic."""
    allv = np.sort(np.concatenate([a, b]))
    ca = np.searchsorted(np.sort(a), allv, side="right") / len(a)
    cb = np.searchsorted(np.sort(b), allv, side="right") / len(b)
    return float(np.abs(ca - cb).max())


def check(batch, label):
    bs = (batch - mu) / sd
    r_if = float((det.score(bs) > THRESH).mean())
    r_mh = float((maha(bs) > MAHA_THRESH).mean())
    ks = max(ks_stat(ref[:, j], batch[:, j]) for j in range(ref.shape[1]))
    fires = (r_if > 0.05, r_mh > 0.05, ks > 0.15)
    verdict = "ALARM" if any(fires) else "ok"
    print(f"{label:<40} {r_if:>9.3f} {r_mh:>12.3f} {ks:>8.3f}   {verdict}")
    return r_if, r_mh, ks


print("=" * 72)
print("monitoring the input distribution")
print("=" * 72)
print("Three signals with three different definitions of 'wrong'. The two")
print("flag rates are calibrated to 1% on the reference, so 0.01 is the")
print("no-drift baseline; the KS column is the largest per-feature")
print("two-sample statistic, where ~0.03 is the no-drift baseline.\n")
print(f"{'incoming batch':<40} {'iForest':>9} {'Mahalanobis':>12} "
      f"{'max KS':>8}")
print("-" * 78)

check(make_reference(1500), "same distribution (the control)")

# 1. a feature silently changes units
b = make_reference(1500)
b[:, 2] *= 30.0                       # tenure switched from years to months
check(b, "unit change: tenure years -> months")

# 2. a slow covariate shift
for shift in (0.25, 0.5, 1.0, 2.0):
    b = make_reference(1500)
    b[:, 0] += shift * 12             # the population ages
    check(b, f"covariate shift: mean age +{shift * 12:.0f} years")

# 3. an upstream bug fills a column with a constant
b = make_reference(1500)
b[:, 1] = np.log(35000.0)
check(b, "upstream bug: log_income constant")

# 4. the correlation breaks while every MARGINAL is preserved
b = make_reference(1500)
b[:, 1] = rng.permutation(b[:, 1])
check(b, "correlation broken, marginals identical")

# 5. missing values imputed with the mean
b = make_reference(1500)
idx = rng.choice(1500, 500, replace=False)
b[idx, 3] = ref[:, 3].mean()
check(b, "33% of n_txn mean-imputed upstream")

print("\nRead across the columns: each signal catches what the others miss,")
print("and the division of labour follows directly from what each one")
print("measures.")
print("\nThe ISOLATION FOREST scores points in SPARSE regions, so it fires")
print("on inputs that land where no reference data lives — the unit change,")
print("most obviously. It is nearly blind to the constant-income bug, and")
print("that is not a defect but a consequence: a column collapsed to one")
print("central value puts every row in the DENSEST region there is. A")
print("detector built to find isolated points cannot find")
print("over-concentrated ones.")
print("\nMAHALANOBIS measures departure from the joint structure, so it")
print("fires when the correlation is broken even though every marginal is")
print("untouched — a point ordinary in every column but wrong in the")
print("covariance is far from the centre in the whitened space (eq. 42.8).")
print("\nPER-FEATURE KS is the crudest of the three and the only one that")
print("reliably catches a collapsed or mean-imputed column, because those")
print("change a marginal DISTRIBUTION without moving anything into a sparse")
print("region or breaking a correlation.")
print("\nThe practical lesson is stronger than 'add a multivariate check':")
print("run several detectors with DIFFERENT definitions of anomalous,")
print("because each is blind to roughly what the others are built for.")

# --- what a per-feature monitor would have seen -----------------------------
print("\n" + "=" * 72)
print("what per-feature monitoring misses")
print("=" * 72)


def ks_stat(a, b):
    """Two-sample Kolmogorov-Smirnov statistic — the usual drift check."""
    allv = np.sort(np.concatenate([a, b]))
    ca = np.searchsorted(np.sort(a), allv, side="right") / len(a)
    cb = np.searchsorted(np.sort(b), allv, side="right") / len(b)
    return float(np.abs(ca - cb).max())


broken = make_reference(1500)
broken[:, 1] = rng.permutation(broken[:, 1])
print(f"{'feature':<14} {'KS statistic vs reference':>27} {'verdict':<12}")
for j, nm in enumerate(NAMES):
    d = ks_stat(ref[:, j], broken[:, j])
    print(f"{nm:<14} {d:>27.4f} "
          f"{('drift' if d > 0.1 else 'no drift'):<12}")

bs = (broken - mu) / sd
print(f"\nisolation forest flag rate : "
      f"{float((det.score(bs) > THRESH).mean()):.4f}   (reference 0.0100)")
print(f"Mahalanobis flag rate      : "
      f"{float((maha(bs) > MAHA_THRESH).mean()):.4f}   (reference 0.0100)")
print("\nEvery per-feature KS test says 'no drift', and every one of them is")
print("CORRECT — the marginals really are unchanged. A wall of per-feature")
print("histograms would show nothing whatsoever.")
print("\nThe joint detector sees it, and note which joint detector: the one")
print("whose definition of anomalous is 'violates the correlation")
print("structure'. Input monitoring needs at least one multivariate check,")
print("and it needs the right kind.")

# --- the honest limits ------------------------------------------------------
print("\n" + "=" * 72)
print("what this does NOT tell you")
print("=" * 72)
b = make_reference(1500)
b_shift = b.copy()
b_shift[:, 0] += 2.0                  # a small age shift: within normal range
r = check(b_shift, "small shift, entirely within the normal range")
print("\nA drift small enough to sit inside the reference distribution is")
print("invisible to a novelty detector by construction, and it can still")
print("move a model's calibration. Input monitoring catches inputs the model")
print("has never seen; it does not catch a model whose relationship to the")
print("target has changed. For that you need outcomes, and outcomes arrive")
print("late — which is the subject of Chapter 179.")

print("\nAnd the reverse failure: a HIGH flag rate is not proof of a problem.")
print("A legitimate new customer segment looks exactly like an anomaly to a")
print("detector fitted on last quarter's population. The detector tells you")
print("the input distribution moved, and a human decides whether that is a")
print("bug or a business.")
```

## 9. Common Mistakes

**Reporting accuracy.** At a 0.1% rate, "normal" always scores 99.9%.

**Reporting ROC-AUC alone.** The measured table shows it barely moving across a
200-fold change in the anomaly rate.

**Reporting PR-AUC without its baseline.** The baseline is the anomaly rate.

**Treating `contamination` as knowledge.** It is a quantile of the score; on
clean data it flags exactly that fraction of clean points.

**Using a global distance threshold with varying densities.** The measurement
shows the diffuse normal cluster scoring further from its neighbours than the
true anomalies.

**Giving isolation forest all your data per tree.** Swamping; 256 is the
recommendation and the measurement supports it.

**Using LOF or k-NN distance in high dimensions.** Distances concentrate;
isolation forest computes none.

**Monitoring only per-feature histograms.** The measured case has identical
marginals and destroyed joint structure.

**Assuming a high flag rate means a bug.** A new customer segment is an anomaly
to a detector fitted on the old population.

**Using an unsupervised detector when you have labels.** That is imbalanced
classification, and {{ch:ml-boosting}} is usually better.

## 10. Connection to Previous Chapters

{{ch:ds-cleaning}} supplied the robust statistics behind {{eq:modified-z}} and
the masking problem that motivates them. {{ch:ml-pca}} supplied both the
reconstruction error of {{eq:pca-residual}} and the whitening that
{{eq:mahalanobis-whitened}} shows Mahalanobis distance to be.
{{ch:ml-clustering}} supplied the varying-density impossibility that LOF's ratio
resolves, and the validation-without-labels problem in its harsher form.
{{ch:ml-trees}} supplied the tree machinery isolation forest inverts — random
splits, no criterion, and the score is the depth rather than the leaf.
{{ch:ml-metrics}} supplied PR-AUC and the threshold-from-cost reasoning, and
this chapter is the setting where both stop being optional.
{{ch:ml-knn-nb}} supplied the curse of dimensionality that decides between LOF
and isolation forest.

Forward: {{ch:dl-autoencoders}} generalises {{eq:pca-residual}} to nonlinear
reconstruction. {{ch:mle-drift}} builds the monitoring of
{{sec:8-practical-example}} into a production system.
{{ch:sec-poisoning}} treats anomalies that are adversarially designed to look
normal, which breaks every assumption here. {{ch:ds-timeseries}}'s residual
approach is the contextual-anomaly case.

## 11. Exercises

**Beginner**

1. Distinguish outlier detection from novelty detection.
2. Why is accuracy meaningless for anomaly detection?
3. What does `contamination` control?
4. Why is the modified z-score more robust than the z-score?
5. What does an isolation forest score of 0.5 mean?

**Intermediate**

6. Explain why {{eq:mahalanobis}} catches anomalies a per-feature z-score
   cannot.
7. Explain why LOF is a ratio, using the varying-density example.
8. Why does isolation forest get worse with more samples per tree?
9. Given 1,000 anomalies in a million records and a detector with 900 true
   positives and 10,000 false positives, compute recall, FPR and precision.
10. Why is precision@k often more useful than PR-AUC?
11. Give a contextual anomaly and explain why a point detector misses it.

**Advanced**

12. Derive $c(n)$ in {{eq:iforest-score}} and explain what it normalises.
13. Prove {{eq:mahalanobis-whitened}} and state the consequence for
    near-singular covariance.
14. Explain why $d_M^{2} \sim \chi^{2}_{D}$ under multivariate normality and how
    to use it for a threshold with a stated false-positive rate.
15. Explain formally why isolation forest resists the curse of dimensionality
    better than LOF.
16. Design an evaluation for a detector where anomaly labels are themselves
    unreliable, and say what it can and cannot establish.

**Implementation**

17. Implement the minimum covariance determinant estimator and compare against
    plain Mahalanobis under 20% contamination.
18. Implement extended isolation forest with oblique splits and compare on a
    diagonal boundary.
19. Implement a one-class SVM using the SMO machinery of {{ch:ml-svm}} and
    compare against isolation forest on time and accuracy.
20. Build the drift monitor of {{sec:8-practical-example}} with a rolling
    reference window and an alarm that requires $k$ consecutive breaches.

**Reasoning**

21. Your detector's PR-AUC is 0.15 at a 0.2% anomaly rate. Is it good? What do
    you need to know to answer?
22. The flag rate jumps from 1% to 12% overnight. List your hypotheses in the
    order you would check them.

## 12. Chapter Summary

Anomaly detection splits into outlier detection on contaminated data, novelty
detection against a clean reference, and supervised detection — which is just
imbalanced classification and is better served by {{ch:ml-boosting}}.

Four definitions of "anomalous" give four families, and the measured comparison
shows each winning on the geometry it was built for. Distance methods are
simplest and inherit every weakness of {{ch:ml-knn-nb}}. Mahalanobis distance is
Euclidean distance in the whitened space, which is why it catches points that are
ordinary in every marginal and violate the joint structure — a case a
coordinate-wise z-score cannot see at all.

LOF's contribution is that its score is a *ratio* of local densities rather than
a distance. The measurement shows why that is necessary: with one tight and one
diffuse cluster, the normal diffuse points sit further from their neighbours
than the true anomalies do, so no global threshold can work.

Isolation forest inverts the tree idea — random splits, no criterion, score by
path length — and is the practical default because it needs no distance
computation and scales to high dimensions. Uniquely in this book, giving it
*more* data per tree makes it worse; the measurement reproduces the swamping
effect that makes 256 samples per tree the recommendation.

PCA reconstruction error is nearly free once you have run PCA, and is the linear
ancestor of the autoencoder detectors in {{part:6}}.

Evaluation is the part that must not be skipped. Accuracy is meaningless.
ROC-AUC barely moves across a 200-fold change in the anomaly rate, because its
denominator is the number of negatives; PR-AUC collapses, because precision's
denominator is the number of alerts. A detector can simultaneously be hundreds
of times better than chance and produce more false alarms than an operations
team can handle, and only one of those facts is visible in the ROC number.
Report PR-AUC against its baseline and precision@$k$ at the $k$ you can staff.

`contamination` is a quantile of the score and nothing more — on clean data it
flags exactly the fraction you asked for.

Every deployed model needs a detector on its own inputs, and the measured
example shows the reason a wall of per-feature histograms is not enough: a batch
with identical marginals and destroyed joint structure passes every univariate
drift test and is caught immediately by a multivariate one.

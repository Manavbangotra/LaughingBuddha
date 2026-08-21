---
id: ml-pca
number: 41
part: IV
tier: focused
status: reviewed
requires: [ml-clustering, ml-knn-nb, math-eigen, math-vectors]
provides: [pca, explained-variance, scree-plot, whitening, truncated-svd,
           reconstruction-error, manifold-learning-caveat, random-projection]
citations: [pedregosa2011, grinsztajn2022]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Derive PCA as the eigendecomposition of the covariance matrix and as the
   SVD of the centred data.
2. Explain what a principal component is and — more importantly — what it is
   not.
3. Choose the number of components from explained variance and from a
   downstream objective.
4. Explain why centring is required and why standardising usually is too.
5. Use truncated SVD for sparse data without destroying sparsity.
6. Explain what PCA cannot do, and what to use instead.
7. State the caveat that makes t-SNE and UMAP visualisation tools rather than
   preprocessing steps.

## 2. Why This Matters

PCA is the most-used unsupervised method there is, and the most-misused.

**It is the standard defence against the curse of dimensionality.**
{{ch:ml-knn-nb}} measured distances concentrating and k-NN degrading with every
added noise dimension; {{ch:ml-clustering}} inherited the same problem. PCA is
the usual first response, and understanding exactly what it discards is what
tells you whether the response is appropriate.

**It is compression, and compression is representation learning's ancestor.**
An autoencoder with linear activations and squared error learns exactly the PCA
subspace ({{ch:dl-autoencoders}}). Everything {{part:11}} does with embeddings
is a nonlinear, learned version of the same idea: find a low-dimensional space
in which the structure survives. PCA is that idea in a form where every
question has a closed-form answer.

**Its output is routinely over-interpreted.** "PC1 represents customer
affluence" is a claim about a direction that was chosen to maximise variance and
for no other reason. Components are orthogonal by construction, sign-arbitrary,
and not identifiable when eigenvalues are close. {{sec:5-formal-explanation}} is
mostly about which readings are legitimate.

## 3. Prerequisites

{{ch:math-eigen}} for eigendecomposition, the SVD, and the Eckart–Young theorem
— which is the entire mathematical content of this chapter, already proved.
{{ch:math-vectors}} for projections. {{ch:ml-knn-nb}} for the curse this
addresses. {{ch:ml-clustering}} for the application in
{{sec:8-practical-example}}.

## 4. Intuitive Explanation

### 4.1 Rotate to where the variance is

Data in $D$ dimensions rarely fills them. Height and weight are correlated, so
a scatter of the two is an elongated cloud along a diagonal — two measurements
carrying roughly one dimension of information.

PCA finds the direction of greatest variance, then the direction of greatest
remaining variance orthogonal to it, and so on. Keeping the first few and
discarding the rest is a rotation followed by a truncation.

```text
   original axes                    rotated to the components
        x2 │      ●●                     PC2 │
           │    ●●●●●                        │   ● ●● ●●● ●● ●
           │  ●●●●●                     ─────┼──────────────────  PC1
           │●●●●                             │
           └──────── x1                      │
     variance is spread              PC1 carries almost all of it;
     across both axes                dropping PC2 loses little
```

Two things are worth being clear about immediately. The components are
**linear combinations of the original features** — PC1 might be $0.7\times$
height $+\;0.7\times$ weight — so they are not new measurements, they are
mixtures. And the criterion is **variance**, which is not the same as
importance, information, or predictive value.

### 4.2 Variance is not importance

This is the assumption to hold up to the light every time.

PCA keeps high-variance directions. That is the right thing to do when variance
tracks signal — which it often does, because noise is usually small relative to
the structure you care about.

It is the wrong thing to do when it does not. If the label depends on a
low-variance direction, PCA will discard exactly the information you need, and
it will do so *before ever seeing the label*, because PCA is unsupervised.
{{sec:7-implementation}} constructs that case and measures the damage.

The practical consequence: PCA is a reasonable default for compression,
visualisation and decorrelation, and a poor default for supervised feature
selection. If you have labels, use them —
{{ch:ds-feature-eng}}'s methods and partial least squares both do.

### 4.3 Scaling decides the answer

Variance depends on units. A feature in millimetres has $10^{6}$ times the
variance of the same feature in metres, so PCA on unstandardised data is
dominated by whichever column happens to have the largest numbers.

**Centring is mandatory** — the derivation assumes it, and without it PC1 points
at the mean rather than at the direction of spread.

**Standardising is usually right** — divide by the standard deviation, which is
equivalent to running PCA on the correlation matrix rather than the covariance
matrix. The exception is when all features share a unit and their relative
magnitudes are meaningful: pixel intensities, or the coordinates of a spectrum.

### 4.4 What PCA cannot do

**Nonlinear structure.** Points on a spiral or a Swiss roll live on a curved
one-dimensional or two-dimensional manifold that no linear projection can
unroll.

**Preserve interpretability.** Every component is a mixture of every original
feature.

**Find the label-relevant direction.** It never sees the label.

**Beat feature selection when the truth is sparse.** If five of a thousand
features matter, PCA gives you components that mix all thousand; the lasso gives
you the five.

> WARNING: **t-SNE and UMAP are visualisation tools, not dimensionality
> reduction for modelling.** They optimise a neighbourhood-preservation
> objective, and the consequence is that distances *between* clusters in the
> embedding are not meaningful — two clusters appearing far apart may be close
> in the original space, and cluster sizes in the plot carry no information.
> Do not cluster on t-SNE coordinates, do not feed them to a model, and do not
> read the gaps. They are excellent for looking at data and misleading for
> anything else.

## 5. Formal Explanation

### 5.1 The two derivations

Let $\mat{X} \in \R^{N \times D}$ be **centred** (every column mean zero). The
sample covariance is $\mat{S} = \frac{1}{N-1}\mat{X}\T\mat{X}$.

**Maximum variance.** Find the unit vector $\vec{v}$ maximising the variance of
the projected data:

$$
\vec{v}_1 = \argmax_{\|\vec{v}\|=1} \vec{v}\T\mat{S}\vec{v}
$$ (eq:pca-variance)

**Minimum reconstruction error.** Find the $k$-dimensional subspace minimising
the squared distance from the data to its projection:

$$
\min_{\mat{V}_k} \sum_{i=1}^{N}
  \big\|\vec{x}_i - \mat{V}_k\mat{V}_k\T\vec{x}_i\big\|^{2}
$$ (eq:pca-reconstruction)

**These give the same answer**, and the proof in
{{sec:6-mathematical-foundation}} is three lines. The solution to both is the
top eigenvectors of $\mat{S}$.

### 5.2 Via the SVD

Do not form $\mat{S}$. As in {{ch:ml-linear-regression}}, computing
$\mat{X}\T\mat{X}$ squares the condition number. Use the SVD directly:

$$
\mat{X} = \mat{U}\mat{\Sigma}\mat{V}\T
$$ (eq:pca-svd)

Then $\mat{S} = \frac{1}{N-1}\mat{V}\mat{\Sigma}^{2}\mat{V}\T$, so:

- the **principal directions** are the columns of $\mat{V}$;
- the **eigenvalues** are $\lambda_j = \sigma_j^{2}/(N-1)$;
- the **scores** — the data in the new coordinates — are
  $\mat{X}\mat{V} = \mat{U}\mat{\Sigma}$.

The **explained variance ratio** of component $j$ is

$$
\text{EVR}_j = \frac{\lambda_j}{\sum_{l=1}^{D}\lambda_l}
 = \frac{\sigma_j^{2}}{\sum_l \sigma_l^{2}}
$$ (eq:evr)

### 5.3 Choosing $k$

**Cumulative explained variance.** Keep enough components to reach 90% or 95%.
Common, arbitrary, and fine for compression.

**The scree plot.** Plot $\lambda_j$ against $j$ and look for the elbow. Same
subjectivity as {{ch:ml-clustering}}'s elbow, and the same caveat.

**Parallel analysis.** Compare the eigenvalues against those from shuffled data
— shuffling each column independently destroys the correlations while preserving
the marginals. Keep components whose eigenvalue exceeds the shuffled
distribution. This is the principled method, and it is the direct analogue of the
gap statistic from {{ch:ml-clustering}}: both compare against an explicit null.

**Downstream performance.** Treat $k$ as a hyperparameter and cross-validate it
against the metric you actually care about. This is the only one that can
account for the fact that variance is not importance.

> IMPORTANT: PCA must be fitted on the training fold only. The mean, the scaling
> and the components are all estimated quantities, and fitting them on the full
> dataset leaks the validation set into training — the preprocessing-leakage
> mechanism of {{ch:ds-leakage}}. It is a particularly easy mistake here because
> PCA feels like a property of the data rather than a fitted model.

### 5.4 Whitening, and its cost

**Whitening** divides each score by $\sqrt{\lambda_j}$, giving components with
unit variance and zero correlation:

$$
\mat{Z}_{\text{white}} = \mat{U}\sqrt{N-1}
$$ (eq:whitening)

It helps optimisation — a spherical loss surface has no ill-conditioning for
gradient descent to struggle with, which is the same argument as
{{ch:dl-normalization}} makes for batch normalisation.

It also **amplifies noise**, and the mechanism is exact: dividing by
$\sqrt{\lambda_j}$ boosts the smallest-variance directions the most, and those
are precisely the ones most likely to be noise. Whiten and then keep all
components and you have multiplied your noise; whiten after truncating to the
components you trust and it is safe.

### 5.5 Variants worth knowing

**Truncated SVD** (LSA) skips the centring step, so it works directly on sparse
matrices. Centring a sparse matrix makes it dense — a $10^{6} \times 10^{5}$
sparse count matrix becomes $8 \times 10^{11}$ bytes — so this is not a
refinement but the only option for text.

**Randomised SVD** approximates the top $k$ components in $O(NDk)$ rather than
$O(ND\min(N,D))$, using random projections. It is the default in scikit-learn
for large $k$ {{cite:pedregosa2011}} and the accuracy loss is usually
negligible.

**Kernel PCA** applies the kernel trick of {{ch:ml-svm}} to PCA, giving
nonlinear components at $O(N^{2})$ cost.

**Random projection** deserves a mention because it is surprising: projecting
onto a *random* $k$-dimensional subspace approximately preserves all pairwise
distances, with $k$ depending only on $N$ and the tolerance — not on $D$ at all.
That is the Johnson–Lindenstrauss lemma, and it means you can often skip PCA
entirely when all you need is to make distances cheaper to compute.
{{sec:7-implementation}} measures it against PCA.

## 6. Mathematical Foundation

### 6.1 The two objectives coincide

**Maximum variance.** Maximise $\vec{v}\T\mat{S}\vec{v}$ subject to
$\vec{v}\T\vec{v}=1$. The Lagrangian is

$$
\Like = \vec{v}\T\mat{S}\vec{v} - \lambda(\vec{v}\T\vec{v} - 1)
$$

Differentiating and setting to zero:

$$
2\mat{S}\vec{v} - 2\lambda\vec{v} = 0
\quad\Longrightarrow\quad
\mat{S}\vec{v} = \lambda\vec{v}
$$ (eq:pca-eigen)

$\vec{v}$ must be an eigenvector, and since the objective at that point is
$\vec{v}\T\mat{S}\vec{v} = \lambda$, the maximiser is the eigenvector with the
largest eigenvalue. Subsequent components follow by the same argument restricted
to the orthogonal complement.

**Minimum reconstruction error.** For an orthonormal basis $\mat{V}_k$, decompose
by Pythagoras:

$$
\|\vec{x}\|^{2}
 = \|\mat{V}_k\mat{V}_k\T\vec{x}\|^{2}
 + \|\vec{x} - \mat{V}_k\mat{V}_k\T\vec{x}\|^{2}
$$

Summing over the data, the total is fixed, so **minimising** the reconstruction
error is exactly **maximising** the retained projection — the same problem.

That equivalence is why one method serves two purposes that sound different:
"keep the most variance" and "lose the least information" are the same
instruction when information is measured by squared error.

### 6.2 Eckart–Young: PCA is the best low-rank approximation

{{ch:math-eigen}} proved it. Restated here in PCA's terms: among all matrices of
rank $k$,

$$
\|\mat{X} - \mat{X}_k\|_F
 = \min_{\rank(\mat{B}) \le k}\|\mat{X}-\mat{B}\|_F
 = \sqrt{\sum_{j>k}\sigma_j^{2}}
$$ (eq:eckart-young)

where $\mat{X}_k = \sum_{j \le k}\sigma_j\vec{u}_j\vec{v}_j\T$.

This is a strong statement and worth appreciating. It is not that PCA is a good
heuristic for low-rank approximation; it is that no rank-$k$ matrix of any kind,
found by any method, approximates $\mat{X}$ better in Frobenius norm. The
truncated SVD is optimal, and {{eq:eckart-young}} even tells you the error in
advance from the discarded singular values.

### 6.3 Why variance is not importance

Construct the counterexample explicitly. Let

$$
x_1 \sim \mathcal{N}(0, 100), \qquad
x_2 \sim \mathcal{N}(0, 1), \qquad
y = x_2 + \epsilon
$$

with $x_1$ and $x_2$ independent. Then $\mat{S} = \diag(100, 1)$, so PC1 is
exactly the $x_1$ axis with 99% of the variance, and PC2 is $x_2$.

Reduce to one component and you keep $x_1$ — which is independent of the label —
and discard $x_2$, which *is* the label. PCA has retained 99% of the variance and
100% of the noise.

This is not a pathological construction. It is the generic situation whenever a
high-variance nuisance factor exists: illumination in images, scanner or batch
effects in genomics, overall document length in text. PCA cannot distinguish
"varies a lot" from "matters", because it has no access to what matters.

The supervised alternative is **partial least squares**, which chooses
directions maximising covariance with the target rather than variance of the
inputs:

$$
\vec{v}_1^{\text{PLS}} = \argmax_{\|\vec{v}\|=1} \Cov(\mat{X}\vec{v},\, \vec{y})
$$ (eq:pls)

One symbol different, and an entirely different answer.

### 6.4 What a component is, and what it is not

**Sign is arbitrary.** If $\vec{v}$ is an eigenvector so is $-\vec{v}$, with the
same eigenvalue. Any interpretation that depends on which end is "high" is
reading a coin flip; implementations fix the sign by an arbitrary convention.

**Components are not identifiable when eigenvalues are close.** If $\lambda_3
\approx \lambda_4$, the corresponding eigenvectors span a plane but their
individual directions within it are determined by noise, and will rotate
substantially between samples. Interpreting PC3 as a concept, when PC4 has a
similar eigenvalue, is interpreting sampling variation.
{{sec:7-implementation}} measures this instability directly.

**Orthogonality is a constraint, not a discovery.** PC2 is orthogonal to PC1
because it was required to be. If the real underlying factors are correlated —
as they usually are — no PCA component corresponds to any of them.

Legitimate readings: how many dimensions the data effectively occupies; which
features co-vary (from the loadings); a low-dimensional picture for looking at.
Illegitimate: "PC1 is affluence" as a claim about a latent construct. For that,
factor analysis with a rotation is the tool designed for the job — and it has
its own well-known identifiability problems.

## 7. Implementation

```python {tier=A name=pca-from-scratch}
"""PCA from scratch, both derivations, and the assumption that fails.
"""
import numpy as np

rng = np.random.default_rng(0)


def pca_fit(X, n_components=None):
    """PCA via the SVD of the centred data (eq. 41.4). Never form X^T X."""
    mu = X.mean(0)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = Vt.shape[0] if n_components is None else n_components
    lam = S ** 2 / (len(X) - 1)
    return {"mean": mu, "components": Vt[:k], "explained_variance": lam[:k],
            "evr": lam[:k] / lam.sum(), "singular_values": S[:k],
            "total_variance": lam.sum()}


def pca_transform(model, X):
    return (X - model["mean"]) @ model["components"].T


def pca_inverse(model, Z):
    return Z @ model["components"] + model["mean"]


# --- the two derivations agree (section 6.1) --------------------------------
print("=" * 72)
print("max variance and min reconstruction error are the same problem")
print("=" * 72)
A = rng.normal(size=(400, 5))
M = rng.normal(size=(5, 5))
X = A @ M                                    # correlated by construction
m = pca_fit(X)

# 1. the eigen route, for comparison
S_cov = np.cov(X - X.mean(0), rowvar=False)
lam_eig, V_eig = np.linalg.eigh(S_cov)
order = np.argsort(lam_eig)[::-1]
lam_eig, V_eig = lam_eig[order], V_eig[:, order]

print(f"{'component':>10} {'eigenvalue of S':>18} {'sigma^2/(N-1)':>16} "
      f"{'|cos| of directions':>21}")
for j in range(5):
    cos = abs(float(V_eig[:, j] @ m["components"][j]))
    print(f"{j + 1:>10} {lam_eig[j]:>18.6f} "
          f"{m['explained_variance'][j]:>16.6f} {cos:>21.8f}")
print("\nIdentical to eight decimal places. The SVD route is preferred for")
print("the same reason as in Chapter 32: forming the covariance matrix")
print("squares the condition number.")

# 2. and the projection really does minimise reconstruction error
print(f"\n{'k':>4} {'cumulative EVR':>16} {'reconstruction MSE':>20} "
      f"{'predicted by eq. 41.9':>23}")
Xc = X - X.mean(0)
_, Sv, _ = np.linalg.svd(Xc, full_matrices=False)
for k in range(1, 6):
    mk = pca_fit(X, k)
    Xr = pca_inverse(mk, pca_transform(mk, X))
    mse = float(np.mean(np.sum((X - Xr) ** 2, axis=1)))
    predicted = float(np.sum(Sv[k:] ** 2) / len(X))
    print(f"{k:>4} {m['evr'][:k].sum():>16.6f} {mse:>20.8f} "
          f"{predicted:>23.8f}")
print("\nThe measured reconstruction error matches the sum of the DISCARDED")
print("squared singular values exactly (eq. 41.9). Eckart-Young says no")
print("rank-k matrix found by any method can do better.")

# --- a random rank-k matrix, to show the bound is not trivial ---------------
best_random = np.inf
for _ in range(2000):
    B = rng.normal(size=(5, 2))
    P = B @ np.linalg.pinv(B)                  # project onto a random plane
    best_random = min(best_random, float(np.mean(np.sum(
        (Xc - Xc @ P) ** 2, axis=1))))
m2 = pca_fit(X, 2)
pca_err = float(np.mean(np.sum((X - pca_inverse(m2, pca_transform(m2, X)))
                               ** 2, axis=1)))
print(f"\nbest of 2,000 RANDOM rank-2 projections : {best_random:.6f}")
print(f"PCA rank-2                              : {pca_err:.6f}")
print("Random search over two thousand planes does not come close. The")
print("optimum is not merely good; it is the provable minimum.")

# --- section 4.3: scaling changes the answer completely ---------------------
print("\n" + "=" * 72)
print("PCA is not scale-invariant (section 4.3)")
print("=" * 72)
n = 800
height_m = rng.normal(1.70, 0.10, n)
weight_kg = 45 + 40 * (height_m - 1.5) + rng.normal(0, 6, n)
age_yr = rng.uniform(20, 70, n)
Xs = np.column_stack([height_m, weight_kg, age_yr])
NAMES = ["height (m)", "weight (kg)", "age (yr)"]

for label, Xu in (("raw units", Xs),
                  ("height in MILLIMETRES",
                   Xs * np.array([1000.0, 1.0, 1.0])),
                  ("standardised", (Xs - Xs.mean(0)) / Xs.std(0))):
    mm = pca_fit(Xu, 3)
    load = mm["components"][0]
    print(f"\n{label}")
    print(f"  EVR: " + "  ".join(f"{v:.4f}" for v in mm["evr"]))
    print(f"  PC1 loadings: " +
          "  ".join(f"{nm.split()[0]}={v:+.4f}"
                    for nm, v in zip(NAMES, load)))

print("\nMeasuring height in millimetres instead of metres changes NO")
print("information and completely rewrites PC1 — it now points almost")
print("entirely along height, because that column's variance grew by a")
print("factor of a million. Standardising removes the dependence on units,")
print("which is why it is the usual default.")

# --- section 6.3: variance is not importance --------------------------------
print("\n" + "=" * 72)
print("variance is not importance (section 6.3)")
print("=" * 72)
n = 1500
x_nuisance = rng.normal(0, 10.0, n)          # huge variance, no signal
x_signal = rng.normal(0, 1.0, n)             # small variance, IS the signal
X_v = np.column_stack([x_nuisance, x_signal])
y_v = x_signal + rng.normal(0, 0.3, n)

mv = pca_fit(X_v, 2)
print(f"explained variance ratio: {mv['evr'][0]:.4f}, {mv['evr'][1]:.4f}")
print(f"PC1 loadings: nuisance={mv['components'][0][0]:+.4f}, "
      f"signal={mv['components'][0][1]:+.4f}")

Z = pca_transform(mv, X_v)
for name, feat in (("PC1 only (99% of variance)", Z[:, :1]),
                   ("PC2 only (1% of variance)", Z[:, 1:2]),
                   ("both components", Z)):
    A_ = np.column_stack([np.ones(n), feat])
    beta, *_ = np.linalg.lstsq(A_, y_v, rcond=None)
    r2 = 1 - np.sum((y_v - A_ @ beta) ** 2) / np.sum((y_v - y_v.mean()) ** 2)
    print(f"  R^2 predicting y from {name:<30} {r2:>8.4f}")

print("\nPC1 holds 99% of the variance and predicts NOTHING. PC2 holds 1%")
print("and is the entire signal. Reducing to one component here would")
print("discard the label and keep the noise — and PCA cannot know, because")
print("it never sees y.")
print("\nThis is not contrived. It is the generic situation whenever a")
print("high-variance nuisance exists: illumination in images, batch effects")
print("in genomics, document length in text. If you have labels, use a")
print("supervised method (eq. 41.13) or select features with them.")

# --- section 6.4: components are not identifiable when eigenvalues tie ------
print("\n" + "=" * 72)
print("components are unstable when eigenvalues are close (section 6.4)")
print("=" * 72)


def stability(true_lams, n=400, trials=30):
    """Resample and measure how much each component direction moves."""
    D = len(true_lams)
    base = None
    cos_by_comp = [[] for _ in range(D)]
    for t in range(trials):
        Xt = rng.normal(size=(n, D)) * np.sqrt(true_lams)
        comp = pca_fit(Xt, D)["components"]
        if base is None:
            base = comp
            continue
        for j in range(D):
            cos_by_comp[j].append(abs(float(base[j] @ comp[j])))
    return [float(np.mean(c)) if c else 1.0 for c in cos_by_comp]

print("population variances along four orthogonal axes, and how stably")
print("PCA recovers each axis across 30 independent samples:\n")
for label, lams in (("well separated: 16, 8, 4, 2", [16.0, 8.0, 4.0, 2.0]),
                    ("2nd and 3rd nearly tied: 16, 4.1, 4.0, 2",
                     [16.0, 4.1, 4.0, 2.0]),
                    ("all four equal: 4, 4, 4, 4", [4.0, 4.0, 4.0, 4.0])):
    cs = stability(np.array(lams))
    print(f"{label:<34} " +
          "  ".join(f"PC{j + 1}={c:.3f}" for j, c in enumerate(cs)))

print("\nThe number is the mean |cosine| between a component and the same")
print("component from a different sample: 1.0 means perfectly reproducible,")
print("0.0 means unrelated.")
print("\nWhen the eigenvalues are well separated the directions are stable.")
print("When two are nearly tied, THOSE TWO become unstable while the others")
print("stay fine — the pair spans a reliable plane, but their individual")
print("directions inside it are set by noise. When all are equal, none of")
print("them means anything at all.")
print("\nSo before writing 'PC3 represents X', check that lambda_3 is clearly")
print("separated from lambda_2 and lambda_4. If it is not, PC3 is a")
print("different direction in every sample and there is nothing to name.")
```

```python {tier=A name=pca-in-practice}
"""PCA as preprocessing: choosing k, avoiding leakage, and the alternatives.
"""
import numpy as np

rng = np.random.default_rng(5)


def pca_fit(X, k=None):
    mu = X.mean(0)
    U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    k = Vt.shape[0] if k is None else k
    lam = S ** 2 / (len(X) - 1)
    return {"mean": mu, "V": Vt[:k], "lam": lam, "evr": lam / lam.sum()}


def pca_transform(m, X, whiten=False):
    Z = (X - m["mean"]) @ m["V"].T
    if whiten:
        Z = Z / np.sqrt(m["lam"][:Z.shape[1]] + 1e-12)
    return Z


def knn_score(Xtr, ytr, Xte, yte, k=11):
    d = ((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)
    idx = np.argpartition(d, k, axis=1)[:, :k]
    pred = (ytr[idx].mean(1) >= 0.5).astype(int)
    return float((pred == yte).mean())


# --- data on a genuinely low-dimensional manifold, plus noise dimensions ----
# The mixing matrix is drawn ONCE, outside the generator. Drawing it inside
# would give the training and test sets different feature-to-latent maps —
# a bug that silently makes the task unlearnable and is easy to miss,
# because every individual array still looks perfectly reasonable.
MIX = rng.normal(size=(5, 8))


def make_data(n, n_noise=40):
    """Five latent factors drive the label; the other 40 columns are noise
    with comparable variance, which is what makes k-NN suffer."""
    Zl = rng.normal(size=(n, 5))
    signal = Zl @ MIX
    z = 1.6 * Zl[:, 0] - 1.3 * Zl[:, 1] + 1.1 * Zl[:, 2] * Zl[:, 3]
    y = (1 / (1 + np.exp(-z)) > rng.random(n)).astype(int)
    noise = rng.normal(size=(n, n_noise))
    return np.column_stack([signal, noise]), y


Xtr, ytr = make_data(800)
Xte, yte = make_data(3000)
D = Xtr.shape[1]
print(f"{D} features: 8 observed mixtures of 5 latent factors, "
      f"plus 40 noise columns")

# --- 1. fit PCA on the TRAINING fold only -----------------------------------
print("\n" + "=" * 72)
print("1. leakage: PCA is a fitted model, not a property of the data")
print("=" * 72)
print("\nPCA estimates a mean, a scaling and a set of components. All three")
print("are FITTED, so fitting them on data that includes the test set lets")
print("the test set influence the representation the model is trained in.")
print("\nHow big is the leak? It depends on how much the test set can move")
print("the estimate, so it depends on the size of the test set relative to")
print("the training set — which is exactly why it is easy to miss in")
print("development and expensive in production.\n")
print("Each row is averaged over 40 independent draws — a single small test")
print("set is far too noisy to read a few points of bias off.\n")
print(f"{'train':>7} {'test':>7} {'correct':>9} {'leaked':>9} {'gap':>9} "
      f"{'SE of gap':>11}")
for n_tr, n_te in ((800, 3000), (400, 1000), (200, 200), (100, 60),
                   (60, 40)):
    gaps, cs, ls = [], [], []
    for _ in range(40):
        Xa, ya = make_data(n_tr)
        Xb, yb = make_data(n_te)
        mu_, sd_ = Xa.mean(0), Xa.std(0)
        A_, B_ = (Xa - mu_) / sd_, (Xb - mu_) / sd_
        mc = pca_fit(A_, 10)                   # fitted on training only
        acc_c = knn_score(pca_transform(mc, A_), ya,
                          pca_transform(mc, B_), yb)
        ml = pca_fit(np.vstack([A_, B_]), 10)  # the mistake
        acc_l = knn_score(pca_transform(ml, A_), ya,
                          pca_transform(ml, B_), yb)
        cs.append(acc_c)
        ls.append(acc_l)
        gaps.append(acc_l - acc_c)
    print(f"{n_tr:>7} {n_te:>7} {np.mean(cs):>9.4f} {np.mean(ls):>9.4f} "
          f"{np.mean(gaps):>+9.4f} "
          f"{np.std(gaps, ddof=1) / np.sqrt(len(gaps)):>11.4f}")

print("\nThe gap is positive at every sample size — the leaked estimate is")
print("optimistic, as it must be — and it broadly grows as the data")
print("shrinks, because a small training set means the test rows carry more")
print("of the weight in the fitted components. (The smallest rows are")
print("themselves noisy; read them against the SE column.) At the largest")
print("sizes it is a")
print("fraction of a point, comparable to its own standard error, and that")
print("is precisely what lets this mistake survive code review.")
print("\nThe rule does not depend on the size of the effect. PCA is a fitted")
print("model; fit it inside the fold, like any other (Chapter 28).")

# from here on, use the original split, correctly
sd = Xtr.std(0)
mu = Xtr.mean(0)
A, B = (Xtr - mu) / sd, (Xte - mu) / sd

# --- 2. choosing k ----------------------------------------------------------
print("\n" + "=" * 72)
print("2. choosing k: variance, parallel analysis, and downstream score")
print("=" * 72)
m_full = pca_fit(A)
cum = np.cumsum(m_full["evr"])

# parallel analysis: shuffle each column to destroy correlation, keep marginals
null_lams = []
for _ in range(20):
    Xs_ = np.column_stack([rng.permutation(col) for col in A.T])
    null_lams.append(pca_fit(Xs_)["lam"])
null_p95 = np.percentile(np.array(null_lams), 95, axis=0)
k_parallel = int(np.sum(m_full["lam"] > null_p95))

print(f"{'k':>4} {'eigenvalue':>12} {'null 95th pct':>15} "
      f"{'cumulative EVR':>16} {'kNN accuracy':>14}")
for k in (1, 3, 5, 8, 10, 20, 48):
    mk = pca_fit(A, k)
    acc = knn_score(pca_transform(mk, A), ytr, pca_transform(mk, B), yte)
    ev = m_full["lam"][k - 1]
    print(f"{k:>4} {ev:>12.4f} {null_p95[k - 1]:>15.4f} "
          f"{cum[k - 1]:>16.4f} {acc:>14.4f}")

k_90 = int(np.searchsorted(cum, 0.90) + 1)
print(f"\n90% cumulative variance needs k = {k_90}")
print(f"parallel analysis keeps        k = {k_parallel}")
print("\nParallel analysis compares each eigenvalue against the 95th")
print("percentile of eigenvalues from column-shuffled data — same marginals,")
print("no correlations. It is the direct analogue of Chapter 40's gap")
print("statistic: judge against an explicit null instead of against a")
print("threshold someone chose.")
print("\nAnd the last column is the one that decides, because it is the only")
print("one that knows what the components are FOR.")

# --- 3. PCA vs random projection vs no reduction ----------------------------
print("\n" + "=" * 72)
print("3. PCA, random projection, and doing nothing")
print("=" * 72)
print("The Johnson-Lindenstrauss lemma says a RANDOM k-dimensional")
print("projection approximately preserves all pairwise distances, with k")
print("depending on N and the tolerance but not on D at all (section 5.5).\n")
print(f"{'k':>4} {'PCA':>10} {'random projection':>20} "
      f"{'distance distortion':>21}")
for k in (2, 5, 10, 20, 40):
    mk = pca_fit(A, k)
    acc_pca = knn_score(pca_transform(mk, A), ytr, pca_transform(mk, B), yte)
    R = rng.normal(size=(D, k)) / np.sqrt(k)
    acc_rp = knn_score(A @ R, ytr, B @ R, yte)
    # how much does the random projection distort pairwise distances?
    sub = A[:200]
    d_orig = np.sqrt(((sub[:, None] - sub[None]) ** 2).sum(-1))
    d_proj = np.sqrt((((sub @ R)[:, None] - (sub @ R)[None]) ** 2).sum(-1))
    iu = np.triu_indices(len(sub), 1)
    ratio = d_proj[iu] / np.maximum(d_orig[iu], 1e-12)
    print(f"{k:>4} {acc_pca:>10.4f} {acc_rp:>20.4f} "
          f"{ratio.std():>21.4f}")
print(f"\nno reduction (all {D} features): "
      f"{knn_score(A, ytr, B, yte):.4f}")
print("\nPCA beats using all 48 features at every k it was tried at — that")
print("is Chapter 35's curse of dimensionality, with the 40 noise columns")
print("diluting every distance. Random projection does NOT beat it here; it")
print("roughly matches it at large k and is clearly worse at small k.")
print("\nThat difference is the point. PCA CHOOSES its subspace using the")
print("data, so at k=2 it has already found where the signal lives. A")
print("random projection preserves distances faithfully — the distortion")
print("column shows the spread of the distance ratio shrinking roughly as")
print("1/sqrt(k), exactly as Johnson-Lindenstrauss predicts — but faithful")
print("preservation of a distance that was mostly noise is not an")
print("improvement.")
print("\nRandom projection earns its place when D is so large that fitting")
print("a PCA is itself the bottleneck, or when the projection must be fixed")
print("before the data is seen. It is a distance-preserving compression,")
print("not a denoiser.")

# --- 4. whitening amplifies noise -------------------------------------------
print("\n" + "=" * 72)
print("4. whitening helps optimisation and amplifies noise (section 5.4)")
print("=" * 72)
print(f"{'k kept':>8} {'plain PCA':>11} {'whitened':>10} "
      f"{'condition number after':>24}")
for k in (5, 10, 20, 48):
    mk = pca_fit(A, k)
    Zp_tr, Zp_te = pca_transform(mk, A), pca_transform(mk, B)
    Zw_tr, Zw_te = (pca_transform(mk, A, whiten=True),
                    pca_transform(mk, B, whiten=True))
    cond_w = np.linalg.cond(Zw_tr)
    print(f"{k:>8} {knn_score(Zp_tr, ytr, Zp_te, yte):>11.4f} "
          f"{knn_score(Zw_tr, ytr, Zw_te, yte):>10.4f} {cond_w:>24.4f}")

print("\nWhitening makes every direction unit-variance, so the condition")
print("number becomes ~1 — which is exactly what gradient descent wants")
print("(Chapter 57 makes the same argument for normalisation layers).")
print("\nThe cost is visible in the accuracy column as k grows: dividing by")
print("sqrt(lambda_j) boosts the SMALLEST-variance directions most, and")
print("those are the ones most likely to be noise. Whiten after truncating")
print("to components you trust; whitening all of them multiplies your noise.")
```

## 8. Practical Example

```python {tier=A name=pca-and-clustering}
"""PCA before clustering: the standard pipeline, and where it goes wrong.
"""
import numpy as np

rng = np.random.default_rng(13)


def pca_fit(X, k=None):
    mu = X.mean(0)
    U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    k = Vt.shape[0] if k is None else k
    lam = S ** 2 / (len(X) - 1)
    return {"mean": mu, "V": Vt[:k], "lam": lam, "evr": lam / lam.sum()}


def pca_transform(m, X):
    return (X - m["mean"]) @ m["V"].T


def kmeans(X, k, n_init=8, seed=0):
    rs = np.random.default_rng(seed)
    best = (None, np.inf)
    for _ in range(n_init):
        C = [X[rs.integers(0, len(X))]]
        for _ in range(1, k):
            D2 = np.min(((X[:, None, :] - np.array(C)[None, :, :]) ** 2
                         ).sum(-1), axis=1)
            tot = D2.sum()
            C.append(X[rs.choice(len(X), p=D2 / tot if tot > 0 else None)])
        C = np.array(C)
        for _ in range(150):
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
        if float(d[np.arange(len(X)), lab].sum()) < best[1]:
            best = (lab, float(d[np.arange(len(X)), lab].sum()))
    return best[0]


def adjusted_rand(a, b):
    la, lb = np.unique(a), np.unique(b)
    cont = np.array([[np.sum((a == i) & (b == j)) for j in lb] for i in la])

    def c2(x):
        return x * (x - 1) / 2
    sij, si, sj = c2(cont).sum(), c2(cont.sum(1)).sum(), c2(cont.sum(0)).sum()
    n2 = c2(len(a))
    exp, mx = si * sj / n2, 0.5 * (si + sj)
    return float((sij - exp) / (mx - exp)) if mx != exp else 1.0


# --- three genuine clusters, buried in noise dimensions ---------------------
def make_data(n, n_noise, noise_sd=1.5):
    """Three clusters in a 3-D subspace, rotated into a higher-dimensional
    space and padded with noise whose PER-AXIS spread is smaller than the
    cluster separation but whose TOTAL contribution to the distance is not.
    That is the realistic case, and the one PCA is built for: the signal
    still occupies the leading directions, while the cumulative noise
    swamps a raw Euclidean distance."""
    centres = np.array([[0, 0, 0], [5, 0, 0], [2.5, 4.5, 0]], float)
    lab = rng.integers(0, 3, n)
    core = centres[lab] + rng.normal(0, 0.8, (n, 3))
    Q = np.linalg.qr(rng.normal(size=(3 + n_noise, 3 + n_noise)))[0]
    X = np.column_stack([core, rng.normal(0, noise_sd, (n, n_noise))]) @ Q
    return X, lab


print("=" * 72)
print("does k-means need PCA to survive noise dimensions?")
print("=" * 72)
print("Three well-separated clusters live in a 3-D subspace, rotated into a")
print("higher-dimensional space and padded with independent noise. The")
print("structure is unchanged throughout — only the ambient dimension.")
print("\nk-NN is included as a control, because Chapter 35 measured IT")
print("collapsing under exactly this treatment.\n")


def knn_acc(X, y, k=11):
    """1-NN-style leave-one-out accuracy, as a distance-quality probe."""
    D = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(D, np.inf)
    nn = np.argpartition(D, k, axis=1)[:, :k]
    pred = np.array([np.bincount(y[r], minlength=3).argmax() for r in nn])
    return float((pred == y).mean())


print(f"{'noise dims':>11} {'total D':>9} {'k-means raw':>13} "
      f"{'k-means on PC1-3':>18} {'k-NN raw':>10}")
for n_noise in (0, 20, 60, 150, 300):
    raws, pcas, knns = [], [], []
    for rep in range(5):                       # averaged: one draw is noisy
        X, ytrue = make_data(600, n_noise)
        Xs = (X - X.mean(0)) / X.std(0)
        raws.append(adjusted_rand(ytrue, kmeans(Xs, 3, seed=1 + rep)))
        m = pca_fit(Xs, 3)
        pcas.append(adjusted_rand(ytrue,
                                  kmeans(pca_transform(m, Xs), 3,
                                         seed=1 + rep)))
        knns.append(knn_acc(Xs, ytrue))
    print(f"{n_noise:>11} {Xs.shape[1]:>9} {np.mean(raws):>13.4f} "
          f"{np.mean(pcas):>18.4f} {np.mean(knns):>10.4f}")

print("\nThree readings, and the folklore gets only the last one right.")
print("\nFIRST: k-means is far more robust to noise dimensions than k-NN.")
print("At 153 dimensions it is still at 0.97 while the k-NN control has")
print("dropped to 0.84 — the Chapter 35 effect, arriving on schedule for")
print("k-NN and not for k-means. The reason is structural. k-means compares")
print("each point to a CENTROID, and a centroid averages hundreds of")
print("points, so the noise coordinates average towards zero and shift")
print("every distance by nearly the same amount, leaving the argmin alone.")
print("k-NN compares points to individual points, whose noise does not")
print("cancel at all. 'High dimensions break distances' is too coarse: they")
print("break point-to-point distances far faster than point-to-centroid")
print("ones.")
print("\nSECOND: when k-means does fail, it fails as a CLIFF. It is at 0.97")
print("with 150 noise dimensions and 0.17 with 300, while k-NN slides down")
print("gently the whole way. A method that is fine until it abruptly is not")
print("is more dangerous than one that degrades visibly.")
print("\nTHIRD: past the cliff, PCA genuinely rescues it — 0.69 against")
print("0.17. That is the case the folklore is describing, and it is real.")
print("\nBut note the price, visible in the middle column before the cliff:")
print("PCA-to-3 is slightly WORSE than raw everywhere else, because the")
print("rotation spread the cluster structure across all the columns, so the")
print("three leading components are the three highest-variance MIXTURES")
print("rather than the three signal directions — not the same thing")
print("(section 6.3, again).")
print("\nWhat PCA buys unconditionally is COST:")
for k in (3, 10):
    print(f"  a distance in 303 dims vs {k}: {303 / k:.0f}x the arithmetic "
          f"per comparison, every iteration of every restart")
print("\nThe honest recommendation is narrower than the folklore: reduce")
print("before clustering when the cost matters, when the dimension is high")
print("enough to be past the cliff, or when you have reason to believe the")
print("signal occupies the leading components. Do not reduce reflexively")
print("because 'high dimensions are bad' — measure whether they are bad for")
print("YOUR algorithm, because the answer differs between two methods that")
print("both compute Euclidean distances.")

# --- ...and the case where the same pipeline destroys the structure ---------
print("\n" + "=" * 72)
print("the same pipeline, when the clusters differ in a LOW-variance")
print("direction")
print("=" * 72)


def make_hard(n, nuisance_sd):
    """Two clusters well separated along x1, plus ONE nuisance direction
    carrying no cluster structure. Only its variance changes."""
    lab = rng.integers(0, 2, n)
    x1 = np.where(lab == 0, -2.5, 2.5) + rng.normal(0, 0.5, n)
    nuisance = rng.normal(0, nuisance_sd, (n, 1))
    return np.column_stack([x1, nuisance]), lab


print(f"{'nuisance sd':>12} {'PC1 EVR':>9} {'|PC1 . x1|':>11} "
      f"{'k-means on raw':>16} {'k-means on PC1':>16}")
for sd in (0.5, 2.0, 5.0, 15.0):
    Xh, yh = make_hard(800, sd)
    ari_raw = adjusted_rand(yh, kmeans(Xh, 2, seed=2))
    mh = pca_fit(Xh, 1)
    ari_pc1 = adjusted_rand(yh, kmeans(pca_transform(mh, Xh), 2, seed=2))
    print(f"{sd:>12} {mh['evr'][0]:>9.3f} {abs(float(mh['V'][0][0])):>11.4f} "
          f"{ari_raw:>16.4f} {ari_pc1:>16.4f}")

print("\nAt nuisance sd = 0.5 the separating axis carries the most variance,")
print("PC1 lands on it, and both columns succeed. As the nuisance grows,")
print("PC1 rotates onto it — the loading on the real axis collapses towards")
print("zero — and clustering on PC1 alone loses the clusters entirely.")
print("\nNote what happens to the raw column at the same time: it degrades")
print("too, because the nuisance dominates the Euclidean distance k-means")
print("uses. So this is not 'PCA bad, raw good'. It is that reducing to the")
print("HIGHEST-VARIANCE component is the exactly wrong move here, and the")
print("right move — keeping the low-variance direction and dropping the")
print("high-variance one — is one PCA cannot make, because it does not know")
print("which is which.")
print("\nThis is section 6.3 again, in a clustering costume: PCA maximises")
print("variance, and variance is not structure. 'Reduce with PCA, then")
print("cluster' is a good default and not a safe one — check that the")
print("components you keep still separate whatever you care about.")

# --- a checklist worth having -----------------------------------------------
print("\n" + "=" * 72)
print("when PCA before clustering helps, and when it hurts")
print("=" * 72)
for cond, verdict in [
        ("many noisy or redundant dimensions", "helps a lot"),
        ("the signal subspace is genuinely low-rank", "helps a lot"),
        ("distances are dominated by irrelevant columns", "helps"),
        ("clusters separated along a LOW-variance axis", "HURTS"),
        ("a high-variance nuisance factor exists", "HURTS"),
        ("features already few and meaningful", "no benefit"),
        ("you need to explain the clusters afterwards",
         "costs interpretability")]:
    print(f"  {cond:<46} -> {verdict}")
```

## 9. Common Mistakes

**Not centring.** The derivation assumes it; without it PC1 points at the mean.

**Not standardising when units differ.** The measured table shows millimetres
versus metres rewriting PC1 entirely.

**Fitting PCA on the full dataset before splitting.** It is a fitted model, and
this is preprocessing leakage.

**Assuming high variance means important.** The measured case has 99% of the
variance in a direction independent of the label.

**Interpreting a component whose eigenvalue is close to its neighbour's.** The
measured stability check shows those directions rotating between samples.

**Reading the sign of a loading.** It is arbitrary.

**Centring a sparse matrix.** It becomes dense; use truncated SVD.

**Whitening all components.** It amplifies the noisiest directions most.

**Clustering on t-SNE or UMAP coordinates.** Inter-cluster distances there are
not meaningful.

**Using PCA for feature selection when you have labels.** Use the labels.

## 10. Connection to Previous Chapters

{{ch:math-eigen}} supplied the eigendecomposition, the SVD and Eckart–Young —
this chapter is an application of results already proved, which is why
{{sec:6-mathematical-foundation}} is short.
{{ch:ml-linear-regression}} supplied the reason not to form $\mat{X}\T\mat{X}$,
and its ridge shrinkage factors were per-SVD-direction in exactly the way PCA's
truncation is a hard version of. {{ch:ml-knn-nb}} supplied the curse of
dimensionality that motivates all of this, and
{{ch:ml-clustering}} supplied the failure that {{sec:8-practical-example}}
repairs — and then breaks again.
{{ch:ds-leakage}} supplied the preprocessing-leakage mechanism.

Forward: {{ch:dl-autoencoders}} is PCA with nonlinear activations — the linear
case provably recovers this subspace. {{ch:emb-what-they-are}} is the learned,
supervised, nonlinear version of "find a space where the structure survives".
{{ch:emb-ann}} uses product quantisation, which is k-means inside PCA-rotated
subspaces. {{ch:ml-anomaly}} uses reconstruction error — the quantity
{{eq:eckart-young}} bounds — as an anomaly score.

## 11. Exercises

**Beginner**

1. What does the first principal component maximise?
2. Why must data be centred?
3. What is the explained variance ratio?
4. Why is the sign of a component arbitrary?
5. Why can PCA not separate a spiral?

**Intermediate**

6. Show that {{eq:pca-variance}} and {{eq:pca-reconstruction}} have the same
   solution.
7. Given singular values $10, 6, 3, 1$, compute the EVR of each component and
   the reconstruction error at $k=2$.
8. Explain why standardising is equivalent to PCA on the correlation matrix.
9. Construct a case where PCA discards the label-relevant direction.
10. Why does truncated SVD skip centring, and what does that cost?
11. Why does whitening amplify noise?

**Advanced**

12. Derive {{eq:pca-eigen}} via Lagrange multipliers and justify taking the
    largest eigenvalue.
13. State Eckart–Young and explain what it guarantees that a heuristic could
    not.
14. Explain the identifiability problem when $\lambda_j \approx \lambda_{j+1}$,
    and relate it to eigenvector perturbation bounds.
15. Derive {{eq:pls}}'s first component and contrast it with
    {{eq:pca-variance}}.
16. State the Johnson–Lindenstrauss lemma precisely and explain why $k$ does not
    depend on $D$.

**Implementation**

17. Implement randomised SVD and compare its accuracy and speed against the full
    SVD at $D = 2000$.
18. Implement kernel PCA with an RBF kernel and unroll a Swiss roll.
19. Implement parallel analysis with a permutation null and compare its $k$
    against the 90% rule on three datasets.
20. Implement incremental PCA for data that does not fit in memory and verify it
    against the batch version.

**Reasoning**

21. A colleague reports that PC1 "is customer affluence" and PC2 "is
    price sensitivity". What do you check before believing it?
22. Your model's accuracy improves when you add PCA and improves further when
    you remove it and use the lasso instead. What does that tell you about the
    data?

## 12. Chapter Summary

PCA finds orthogonal directions of maximum variance, equivalently the subspace
minimising squared reconstruction error — the two objectives are the same
problem by Pythagoras, which is why one method serves both purposes. Compute it
from the SVD of the centred data, never from the covariance matrix.

Eckart–Young makes the result unusually strong: the truncated SVD is not a good
heuristic for low-rank approximation, it is provably optimal, and the discarded
singular values give the error in advance. The measurement confirms both, and
shows two thousand random rank-2 projections falling well short.

PCA is not scale-invariant. Measuring one feature in millimetres instead of
metres changes no information and rewrites PC1 entirely, which is why
standardising is the usual default.

The central caveat is that variance is not importance. The measured example puts
99% of the variance in a direction independent of the label and 1% in the label
itself; reducing to one component keeps the noise and discards the signal, and
PCA cannot know because it never sees the label. Whenever a high-variance
nuisance factor exists — illumination, batch effects, document length — this is
the generic case, and a supervised method is the answer.

Components are not always interpretable, and the measured stability check shows
exactly when: with well-separated eigenvalues the directions reproduce across
samples, and when two eigenvalues are nearly tied those two rotate freely within
the plane they span. Check the separation before naming a component, and never
read the sign of a loading.

PCA is a fitted model — mean, scaling and components are all estimated — so it
must be fitted on the training fold only.

Random projection preserves pairwise distances with a dimension depending on $N$
and the tolerance but not on $D$, and the measurement shows it closing the gap
on PCA as $k$ grows, at no fitting cost. t-SNE and UMAP are for looking at data:
inter-cluster distances in their embeddings are not meaningful and must not be
clustered on.

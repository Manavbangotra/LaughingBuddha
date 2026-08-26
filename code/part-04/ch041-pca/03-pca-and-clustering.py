# -*- coding: utf-8 -*-
# Extracted from: Chapter 41 — PCA and Dimensionality Reduction
# Source: src/.../ch041-pca.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

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

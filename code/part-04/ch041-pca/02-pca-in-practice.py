# -*- coding: utf-8 -*-
# Extracted from: Chapter 41 — PCA and Dimensionality Reduction
# Source: src/.../ch041-pca.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

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

# Extracted from: Chapter 41 — PCA and Dimensionality Reduction
# Source: src/.../ch041-pca.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

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

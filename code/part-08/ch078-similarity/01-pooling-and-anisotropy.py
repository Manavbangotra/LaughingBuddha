# Extracted from: Chapter 78 — Semantic Similarity and Sentence Embeddings
# Source: src/.../ch078-similarity.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Pooling strategies, and anisotropy measured against the isotropic baseline."""
import numpy as np

rng = np.random.default_rng(0)
N, T, D = 500, 24, 128


def mean_cosine(V):
    """Equation (eq:mean-cosine), over all distinct pairs."""
    U = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    S = U @ U.T
    n = len(V)
    return float((S.sum() - np.trace(S)) / (n * (n - 1)))


def similarity_stats(V):
    U = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    s = (U @ U.T)[np.triu_indices(len(V), 1)]
    return float(s.mean()), float(s.std()), float(s.min()), float(s.max())


def top_pc_share(V, k=1):
    """Fraction of total variance in the top k principal components."""
    Xc = V - V.mean(0)
    sv = np.linalg.svd(Xc, compute_uv=False)
    return float((sv[:k] ** 2).sum() / (sv ** 2).sum())


# 1. The isotropic reference: mean 0, standard deviation 1/sqrt(d).
iso = rng.normal(size=(N, D))
print(f"isotropic reference, d={D}")
print(f"  mean cosine      {mean_cosine(iso):+.4f}   (equation eq:random-cosine "
      f"predicts 0.0000)")
print(f"  predicted spread {1 / np.sqrt(D):.4f}")

# 2. An anisotropic representation: two shared directions with positive,
#    varying coefficients plus noise — the shape a pretrained encoder's
#    vectors empirically take.
c1 = rng.normal(size=D)
c1 /= np.linalg.norm(c1)
c2 = rng.normal(size=D)
c2 -= (c2 @ c1) * c1
c2 /= np.linalg.norm(c2)
aniso = (np.abs(rng.normal(1.5, 0.8, (N, 1))) * c1
         + np.abs(rng.normal(1.0, 0.6, (N, 1))) * c2
         + 0.12 * rng.normal(size=(N, D)))

# 3. The two fixes of equation (eq:anisotropy-fix), measured on three axes.
centred = aniso - aniso.mean(0)
_, _, Vt = np.linalg.svd(centred, full_matrices=False)
stripped = centred - (centred @ Vt[:2].T) @ Vt[:2]

print()
print(f"{'representation':<18} {'mean cos':>10} {'sd':>7} {'range':>16} "
      f"{'PC1 var share':>15}")
for name, V in [("raw", aniso), ("centred", centred),
                ("centred, -PC1,2", stripped), ("isotropic ref", iso)]:
    m, sd, lo, hi = similarity_stats(V)
    print(f"{name:<18} {m:>+10.3f} {sd:>7.3f} "
          f"{f'[{lo:+.2f}, {hi:+.2f}]':>16} {top_pc_share(V):>15.3f}")

print("""
Read the columns separately, because they do not agree.

  * Centering removes the high mean entirely and INCREASES the spread — the
    similarity score becomes usable again. This is the reliable fix.
  * Removing principal components leaves the mean where centering put it and
    SHRINKS the spread. It removed variance, and variance is not automatically
    noise. On a ranking task it can cost accuracy.

Anisotropy compresses the usable range of the similarity score. It does not
necessarily destroy the ranking, which is why a system can have a badly
anisotropic space and still retrieve acceptably — and why fixing the geometry
is not the same as fixing the retrieval.""")

# 4. Mean pooling must respect the attention mask.
H = rng.normal(size=(3, T, D))
lengths = np.array([20, 8, 3])
mask = (np.arange(T)[None, :] < lengths[:, None]).astype(float)

naive = H.mean(axis=1)
masked = (H * mask[..., None]).sum(1) / mask.sum(1, keepdims=True)

print(f"\n{'sentence':>9} {'true length':>12} {'padding in mean':>17} "
      f"{'cos(naive, masked)':>20}")
for i, L in enumerate(lengths):
    c = float(naive[i] @ masked[i]
              / (np.linalg.norm(naive[i]) * np.linalg.norm(masked[i])))
    print(f"{i:>9} {L:>12} {1 - L / T:>16.0%} {c:>20.3f}")
print("\nThe shorter the sentence, the more padding the naive mean absorbs — "
      "and the batch's longest member decides how much, so the same sentence "
      "gets a different vector in a different batch.")

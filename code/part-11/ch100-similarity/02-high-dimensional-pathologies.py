# -*- coding: utf-8 -*-
# Extracted from: Chapter 100 — Similarity Measures and the Geometry of Embedding Space
# Source: src/.../ch100-similarity.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Concentration and hubness track INTRINSIC dimension, not stored dimension.

Two regimes at each ambient dimension, same corpus size:

  iid gaussian  -- every coordinate contributes independent variance, so
                   intrinsic dimension = ambient dimension. This is exactly the
                   setting of Beyer's hypothesis (eq:beyer-condition).
  intrinsic 8   -- an 8-dimensional Gaussian pushed through a random linear map
                   into the ambient space. The coordinates are heavily
                   dependent; intrinsic dimension stays 8 however wide we store it.

Measured: relative contrast (eq:relative-contrast) and the k-occurrence
distribution (eq:k-occurrence, eq:hubness-skew).
"""
import numpy as np
from scipy.stats import skew

rng = np.random.default_rng(3)
N, K, LATENT = 5000, 10, 8
N_QUERY = 200
AMBIENT = [2, 8, 32, 128, 512]


def pairwise_sq(A, B):
    """Squared Euclidean distances via eq:l2-ip-identity, clipped for stability."""
    return np.maximum(np.sum(A ** 2, axis=1)[:, None]
                      + np.sum(B ** 2, axis=1)[None, :] - 2.0 * A @ B.T, 0.0)


def contrast(X):
    """Mean of (Dmax - Dmin)/Dmin over held-out queries -- eq:relative-contrast."""
    q = X[rng.choice(len(X), N_QUERY, replace=False)]
    D = np.sqrt(pairwise_sq(q, X))
    # Each query is a member of X, so its own zero distance must be removed.
    np.put_along_axis(D, np.argmin(D, axis=1)[:, None], np.inf, axis=1)
    d_min = D.min(axis=1)
    d_max = np.where(np.isinf(D), -np.inf, D).max(axis=1)
    return float(np.mean((d_max - d_min) / d_min))


def hubness(X):
    """Skew of the k-occurrence distribution, its max, and the unreachable share."""
    D = pairwise_sq(X, X)
    np.fill_diagonal(D, np.inf)
    nn = np.argpartition(D, K, axis=1)[:, :K]
    counts = np.bincount(nn.ravel(), minlength=len(X))
    return float(skew(counts)), int(counts.max()), float(np.mean(counts == 0))


latent = rng.normal(size=(N, LATENT))
summary = {}

print(f"{'ambient':>8} {'regime':<15}{'contrast':>10}{'k-occ skew':>12}"
      f"{'max k-occ':>11}{'never (%)':>11}")
print("-" * 68)
for dim in AMBIENT:
    corpora = {
        "iid gaussian": rng.normal(size=(N, dim)),
        "intrinsic 8": latent @ (rng.normal(size=(LATENT, dim)) / np.sqrt(LATENT)),
    }
    for name, X in corpora.items():
        c = contrast(X)
        s, mx, never = hubness(X)
        summary[(dim, name)] = (c, s, mx, never)
        print(f"{dim:>8} {name:<15}{c:>10.3f}{s:>12.3f}{mx:>11d}{100 * never:>11.1f}")

c_lo = summary[(AMBIENT[0], "iid gaussian")][0]
c_hi, s_hi, mx_hi, nv_hi = summary[(AMBIENT[-1], "iid gaussian")]
c_int, s_int, _, nv_int = summary[(AMBIENT[-1], "intrinsic 8")]

print(f"""
Read the CONTRAST column down the iid rows. It collapses from {c_lo:.0f} at
ambient {AMBIENT[0]} to {c_hi:.2f} at ambient {AMBIENT[-1]} -- that is
eq:concentration-limit happening, and at the bottom the nearest and farthest
points differ by a fraction of the nearest distance. Beyer's
theorem is not an abstraction; this is it.

Now read the intrinsic-8 rows. Same corpus size, same ambient dimensions, same
distance function -- and the contrast is flat and healthy at every width, still
{c_int:.2f} at ambient {AMBIENT[-1]}. The data is 8-dimensional and behaves
8-dimensionally whether we store it in 32
coordinates or 512. That is eq:intrinsic-concentration, and it is why a
768-dimensional embedding index works.

The hubness columns tell the same story in a form you can see in production. In
the iid regime at ambient {AMBIENT[-1]}, the k-occurrence skew is {s_hi:.1f}, one
point is returned for {mx_hi} queries when the expected count is {K}, and
{100 * nv_hi:.0f}% of the corpus is returned for NOTHING -- unreachable by any
query, no matter how the index is built. In the intrinsic-8 regime at the same
width the skew is {s_int:.2f} and {100 * nv_int:.1f}% is unreachable.

Both pathologies are real, both are consequences of eq:beyer-condition holding,
and both are governed by intrinsic dimension. The practical reading: if your
retrieval degrades as you scale the corpus, measure contrast and k-occurrence
skew before you touch the index -- they distinguish "the geometry is bad" from
"the index is losing recall", and those need opposite fixes.""")

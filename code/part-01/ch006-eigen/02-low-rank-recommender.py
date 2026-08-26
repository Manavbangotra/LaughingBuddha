# -*- coding: utf-8 -*-
# Extracted from: Chapter 6 — Eigenvalues, Eigenvectors, and the Singular Value Decomposition
# Source: src/.../ch006-eigen.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

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

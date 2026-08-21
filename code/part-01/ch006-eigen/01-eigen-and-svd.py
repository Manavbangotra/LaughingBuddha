# Extracted from: Chapter 6 — Eigenvalues, Eigenvectors, and the Singular Value Decomposition
# Source: src/.../ch006-eigen.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Eigenvalues, the spectral theorem, the SVD, and Eckart-Young optimality.

The optimality claim is tested against random rank-k competitors rather than
taken on trust.
"""
import numpy as np

# --- eigenvalues by hand vs numerically (section 6.1) -----------------------
A = np.array([[4.0, 1.0],
              [2.0, 3.0]])
vals, vecs = np.linalg.eig(A)
order = np.argsort(-vals)
vals, vecs = vals[order], vecs[:, order]
print("eigenvalues :", np.round(vals, 6), " (hand-computed: 5 and 2)")
print("trace check :", np.trace(A), "==", vals.sum().real)
print("det check   :", round(np.linalg.det(A), 6), "==", round(np.prod(vals).real, 6))

for i in range(2):
    v = vecs[:, i]
    print(f"  A v{i} == lambda{i} v{i}: "
          f"{np.allclose(A @ v, vals[i] * v)}   v{i} = {np.round(v, 4)}")

print(f"eigenvectors orthogonal? {abs(vecs[:,0] @ vecs[:,1]) < 1e-9} "
      f"(A is not symmetric, so no reason they should be)")

# --- the spectral theorem: symmetry buys orthogonality ----------------------
rng = np.random.default_rng(0)
M = rng.normal(size=(5, 5))
S = M + M.T                                  # any M + M^T is symmetric
w, Q = np.linalg.eigh(S)                     # eigh: for symmetric matrices
print(f"\nsymmetric matrix: eigenvalues all real? {np.isrealobj(w)}")
print(f"eigenvectors orthonormal? {np.allclose(Q.T @ Q, np.eye(5))}")
assert np.allclose(Q @ np.diag(w) @ Q.T, S)  # eq. 6.5
print("Q Lambda Q^T reconstructs S exactly (eq. 6.5)")

# Positive semi-definiteness: A^T A always has non-negative eigenvalues.
G = M.T @ M
print(f"eigenvalues of M^T M all >= 0? {np.all(np.linalg.eigvalsh(G) > -1e-10)}")

# --- eq. 6.6: the SVD of a non-square, rank-deficient matrix ----------------
B = np.array([[3.0, 0.0],
              [4.0, 0.0]])
U, s, Vt = np.linalg.svd(B)
print(f"\nsingular values of B: {np.round(s, 6)}  (hand-computed: 5 and 0)")
print(f"rank from SVD: {np.sum(s > 1e-10)}  | numpy rank: {np.linalg.matrix_rank(B)}")
print(f"u_1 = {np.round(U[:, 0], 4)}  (hand-computed [0.6, 0.8], up to sign)")
assert np.allclose(U @ np.diag(s) @ Vt, B)

# --- eq. 6.9: norms come straight from the singular values ------------------
C = rng.normal(size=(7, 4))
sc = np.linalg.svd(C, compute_uv=False)
print(f"\nspectral norm : {np.linalg.norm(C, 2):.6f} == sigma_1 = {sc[0]:.6f}")
print(f"frobenius norm: {np.linalg.norm(C, 'fro'):.6f} == "
      f"sqrt(sum sigma^2) = {np.sqrt((sc**2).sum()):.6f}")
print(f"condition number: {np.linalg.cond(C):.4f} == "
      f"sigma_max/sigma_min = {sc[0]/sc[-1]:.4f}")

# --- Eckart-Young, tested against random competitors ------------------------
# Build a matrix that is genuinely low-rank plus noise.
m, n, true_rank = 60, 40, 6
L = rng.normal(size=(m, true_rank)) @ rng.normal(size=(true_rank, n))
A2 = L + 0.15 * rng.normal(size=(m, n))

U2, s2, Vt2 = np.linalg.svd(A2, full_matrices=False)

print(f"\n{'k':>3} {'SVD error':>12} {'predicted':>12} {'best random':>13} "
      f"{'energy':>8}")
for k in (1, 3, 6, 10, 20):
    Ak = (U2[:, :k] * s2[:k]) @ Vt2[:k]
    err = np.linalg.norm(A2 - Ak, "fro")
    predicted = np.sqrt((s2[k:] ** 2).sum())        # eq. 6.12

    # 200 random rank-k competitors, each least-squares fitted to give them the
    # best possible chance.
    best_random = np.inf
    for _ in range(200):
        R = rng.normal(size=(m, k))
        coef, *_ = np.linalg.lstsq(R, A2, rcond=None)
        best_random = min(best_random, np.linalg.norm(A2 - R @ coef, "fro"))

    energy = (s2[:k] ** 2).sum() / (s2 ** 2).sum()
    print(f"{k:>3} {err:>12.5f} {predicted:>12.5f} {best_random:>13.5f} "
          f"{energy:>7.1%}")
    assert np.isclose(err, predicted)
    assert err <= best_random + 1e-9      # no competitor ever beats the SVD

print("\nThe SVD error matches eq. 6.12 exactly, and no random rank-k")
print("competitor ever beats it — Eckart-Young, demonstrated.")

# --- the spectrum reveals the intrinsic dimension ---------------------------
print(f"\nsingular values of a rank-{true_rank} matrix plus noise:")
print(" ", np.round(s2[:12], 3))
gaps = s2[:-1] / s2[1:]
print(f"largest consecutive ratio at index {int(np.argmax(gaps[:15])) + 1} "
      f"(ratio {gaps[:15].max():.2f}) — the cliff marks the true rank")

# --- compression arithmetic --------------------------------------------------
print(f"\nstoring a {m}x{n} matrix:")
print(f"  full            : {m*n:,} numbers")
for k in (3, 6, 10):
    print(f"  rank-{k:<2} truncation: {k*(m+n+1):,} numbers "
          f"({k*(m+n+1)/(m*n):.0%} of full), "
          f"error {np.sqrt((s2[k:]**2).sum())/np.linalg.norm(A2,'fro'):.1%}")

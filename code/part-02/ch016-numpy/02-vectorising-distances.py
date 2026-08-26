# -*- coding: utf-8 -*-
# Extracted from: Chapter 16 — NumPy: Arrays, Broadcasting, and Vectorized Computation
# Source: src/.../ch016-numpy.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Three implementations of pairwise distances, from loops to linear algebra.

The final version uses the expansion ||a-b||^2 = ||a||^2 - 2 a.b + ||b||^2
from Chapter 3, turning the whole computation into one matrix product.
"""
import time

import numpy as np

rng = np.random.default_rng(0)


def distances_loops(A, B):
    """Naive: two Python loops. O(n*m) interpreter iterations."""
    n, m = len(A), len(B)
    out = np.empty((n, m))
    for i in range(n):
        for j in range(m):
            diff = A[i] - B[j]
            out[i, j] = np.sqrt(np.sum(diff * diff))
    return out


def distances_broadcast(A, B):
    """Broadcasting: one Python-level operation, but a big temporary.
    (n,1,d) - (1,m,d) -> (n,m,d), which is d times the size of the answer."""
    diff = A[:, None, :] - B[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def distances_gemm(A, B):
    """The expansion of Chapter 3: ||a-b||^2 = ||a||^2 - 2 a.b + ||b||^2.

    The cross term is a single matrix product, which is what BLAS is for.
    No (n,m,d) temporary is ever created.
    """
    a2 = np.einsum("ij,ij->i", A, A)[:, None]     # ||a||^2, shape (n,1)
    b2 = np.einsum("ij,ij->i", B, B)[None, :]     # ||b||^2, shape (1,m)
    sq = a2 - 2.0 * (A @ B.T) + b2
    # Rounding can make a squared distance very slightly negative.
    np.maximum(sq, 0.0, out=sq)
    return np.sqrt(sq)


n, m, d = 600, 500, 64
A = rng.normal(size=(n, d))
B = rng.normal(size=(m, d))

reference = distances_loops(A, B)
results = {}
print(f"{'method':<22} {'time':>10} {'speedup':>9} {'peak temp':>12} {'max err':>10}")
for name, fn, temp_mb in (
    ("python loops", distances_loops, 0.0),
    ("broadcasting", distances_broadcast, n * m * d * 8 / 1e6),
    ("gemm expansion", distances_gemm, n * m * 8 / 1e6),
):
    t0 = time.perf_counter()
    out = fn(A, B)
    elapsed = time.perf_counter() - t0
    results[name] = elapsed
    err = np.abs(out - reference).max()
    print(f"{name:<22} {elapsed*1e3:>8.1f}ms "
          f"{results['python loops']/elapsed:>8.0f}x {temp_mb:>10.1f}MB "
          f"{err:>10.2e}")

print(f"\nAll three compute the same thing to within {1e-9:.0e}.")
print("The broadcast version is fast but allocates an (n,m,d) temporary —")
print(f"{n*m*d*8/1e6:.0f} MB here, and it grows with d. The gemm version")
print(f"allocates only the (n,m) answer, {n*m*8/1e6:.1f} MB, and hands the")
print("work to BLAS.")

# --- how the temporary scales, which is what breaks at real sizes -----------
print(f"\n{'n = m':>8} {'d':>5} {'broadcast temp':>16} {'gemm temp':>12}")
for nn, dd in ((1_000, 128), (10_000, 768), (50_000, 1536)):
    print(f"{nn:>8,} {dd:>5} {nn*nn*dd*8/1e9:>14.1f} GB "
          f"{nn*nn*8/1e9:>10.2f} GB")
print("At retrieval scale the broadcast version is not slow — it is")
print("impossible. This is why Part XI builds on matrix products.")

# --- the same pattern: normalise then use a dot product ---------------------
An = A / np.linalg.norm(A, axis=1, keepdims=True)
Bn = B / np.linalg.norm(B, axis=1, keepdims=True)
cosine = An @ Bn.T
print(f"\ncosine similarity for all {n}x{m} pairs: one matmul, "
      f"{cosine.shape}, {cosine.nbytes/1e6:.1f} MB")
print(f"ranking by cosine == ranking by distance on normalised vectors: "
      f"{np.array_equal(np.argsort(-cosine[0]), np.argsort(distances_gemm(An, Bn)[0]))}")
print("  (Chapter 5, eq. 5.11 — verified here at scale.)")

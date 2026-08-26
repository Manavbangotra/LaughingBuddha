# -*- coding: utf-8 -*-
# Extracted from: Chapter 4 — Matrices, Matrix Multiplication, and Linear Maps
# Source: src/.../ch004-matrices.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Matrices as transformations, the three readings of a product, rank,
associativity as a cost decision, and the broadcasting bug that hides.
"""
import numpy as np

A = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0]])          # (2, 3) : maps R^3 -> R^2
B = np.array([[7.0, 8.0],
              [9.0, 10.0],
              [11.0, 12.0]])             # (3, 2) : maps R^2 -> R^3

print("A", A.shape, " B", B.shape)
print("A @ B  ->", (A @ B).shape, "\n", A @ B)
print("B @ A  ->", (B @ A).shape, " (a different object entirely)")
assert np.array_equal(A @ B, [[58, 64], [139, 154]])   # eq. 4.10

# --- the three readings of a matrix-vector product --------------------------
M = np.array([[1.0, 2.0], [3.0, 4.0]])
x = np.array([5.0, 6.0])

row_view = np.array([M[0] @ x, M[1] @ x])              # dot products
col_view = x[0] * M[:, 0] + x[1] * M[:, 1]             # linear combination
assert np.allclose(row_view, col_view) and np.allclose(row_view, M @ x)
print(f"\nrow view {row_view} == column view {col_view} == M @ x {M @ x}")

# --- eq. 4.11: the transpose reverses order ---------------------------------
assert np.allclose((A @ B).T, B.T @ A.T)
print("(A B)^T == B^T A^T verified")

# --- matrices as actions on the plane (table 4.1) ---------------------------
actions = {
    "identity":    np.array([[1.0, 0.0], [0.0, 1.0]]),
    "scale x2":    np.array([[2.0, 0.0], [0.0, 2.0]]),
    "rotate 90":   np.array([[0.0, -1.0], [1.0, 0.0]]),
    "shear":       np.array([[1.0, 1.0], [0.0, 1.0]]),
    "project x":   np.array([[1.0, 0.0], [0.0, 0.0]]),
    "collapse":    np.array([[1.0, 2.0], [2.0, 4.0]]),
}
v = np.array([1.0, 1.0])
print(f"\n{'action':<12} {'M @ [1,1]':<18} {'rank':>5} {'invertible':>11}")
for name, Mx in actions.items():
    r = np.linalg.matrix_rank(Mx)
    print(f"{name:<12} {str(Mx @ v):<18} {r:>5} {str(r == 2):>11}")

# --- rank and information loss (eq. 4.13) -----------------------------------
Mr = np.array([[1.0, 2.0, 3.0],
               [2.0, 4.0, 6.0],       # exactly 2x row 1 — no new direction
               [1.0, 1.0, 1.0]])
print(f"\nrank of the 3x3 example: {np.linalg.matrix_rank(Mr)} (not 3)")

# Two different inputs map to the same output, so the map cannot be inverted.
null_vec = np.linalg.svd(Mr)[2][-1]          # last right-singular vector
print(f"a null-space direction: {np.round(null_vec, 4)}")
print(f"M @ that direction    : {np.round(Mr @ null_vec, 12)}  <- zero")
u = np.array([1.0, 1.0, 1.0])
print(f"M @ u == M @ (u + null): "
      f"{np.allclose(Mr @ u, Mr @ (u + null_vec))}  <- information destroyed")

# --- eq. 4.7: associativity is a cost decision ------------------------------
rng = np.random.default_rng(0)
P = rng.normal(size=(1000, 5))
Q = rng.normal(size=(5, 1000))
R = rng.normal(size=(1000, 5))
left = (P @ Q) @ R            # builds a 1000x1000 intermediate
right = P @ (Q @ R)           # builds a 5x5 intermediate
assert np.allclose(left, right)
cost_left = 1000 * 5 * 1000 + 1000 * 1000 * 5
cost_right = 5 * 1000 * 5 + 1000 * 5 * 5
print(f"\nsame result, different cost: (PQ)R ~ {cost_left:,} ops, "
      f"P(QR) ~ {cost_right:,} ops  ({cost_left // cost_right}x)")
print("This is exactly the trick that makes LoRA cheap (Part XIV).")

# --- stacked linear layers collapse to one matrix ---------------------------
W1, W2, W3 = rng.normal(size=(4, 6)), rng.normal(size=(6, 5)), rng.normal(size=(5, 3))
xin = rng.normal(size=(2, 4))
stacked = ((xin @ W1) @ W2) @ W3
single = xin @ (W1 @ W2 @ W3)
assert np.allclose(stacked, single)
print("\nthree linear layers == one matrix — which is why nonlinearities exist")

# --- broadcasting: the helpful error and the silent bug ---------------------
X = np.arange(12, dtype=float).reshape(3, 4)
print("\n(3,4) + (4,)  ->", (X + np.ones(4)).shape, " row vector added to each row")
print("(3,4) + (3,1) ->", (X + np.ones((3, 1))).shape, " column added to each column")
try:
    X + np.ones(3)
except ValueError as exc:
    print("(3,4) + (3,)  -> ValueError:", str(exc)[:60])

# The dangerous case: no error, wrong answer. Intending a per-row offset but
# passing a (3,1) instead of a (1,3) produces a valid array of the wrong shape.
per_column = np.array([10.0, 20.0, 30.0, 40.0])      # intended: one per column
wrong = np.array([10.0, 20.0, 30.0]).reshape(3, 1)   # typo: one per row
print(f"\nintended shape {(X + per_column).shape}, "
      f"typo also valid with shape {(X + wrong).shape} — no error raised")
print("Broadcasting turns a shape mistake into a plausible wrong answer.")
print("Assert your shapes; do not rely on an exception.")

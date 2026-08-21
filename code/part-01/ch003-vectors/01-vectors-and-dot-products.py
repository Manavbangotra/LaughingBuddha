# Extracted from: Chapter 3 — Vectors, Dot Products, and Geometric Intuition
# Source: src/.../ch003-vectors.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Vectors, dot products, projection, and what changes in high dimensions.

Everything the chapter asserts geometrically is checked numerically here.
"""
import numpy as np

# --- basics ------------------------------------------------------------------
x = np.array([3.0, 4.0])
y = np.array([4.0, 3.0])

print("x + y      :", x + y)
print("2x         :", 2 * x)
print("x . y      :", x @ y, "| same via sum:", np.sum(x * y))

# The three equivalent spellings of a dot product in NumPy. Prefer @ or np.dot;
# `*` is ELEMENTWISE and silently gives a vector, not a scalar.
assert x @ y == np.dot(x, y) == np.sum(x * y) == 24.0
print("elementwise x * y :", x * y, " <- NOT the dot product")

# --- norms and the angle (eq. 3.9) ------------------------------------------
nx, ny = np.linalg.norm(x), np.linalg.norm(y)
cos_theta = (x @ y) / (nx * ny)
print(f"\n|x| = {nx}, |y| = {ny}")
print(f"cos(theta) = {cos_theta:.4f}, theta = {np.degrees(np.arccos(cos_theta)):.2f} deg")
assert np.isclose(cos_theta, 0.96)

# --- projection (eq. 3.11) and the orthogonal residual ----------------------
proj = ((x @ y) / (y @ y)) * y
residual = x - proj
print(f"\nprojection of x onto y : {proj}")
print(f"residual               : {residual}")
print(f"residual . y           : {residual @ y:.2e}  <- orthogonal, as claimed")
assert abs(residual @ y) < 1e-12

# --- eq. 3.13: Cauchy-Schwarz, checked on random pairs ----------------------
rng = np.random.default_rng(0)
for _ in range(1000):
    a, b = rng.normal(size=5), rng.normal(size=5)
    assert abs(a @ b) <= np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
print("\nCauchy-Schwarz holds on 1000 random pairs")

# --- orthogonality carries independent information --------------------------
e1, e2 = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
v = 3 * e1 + 7 * e2
print(f"\nv = 3*e1 + 7*e2 = {v}")
print(f"read off the e1 component with a dot product: {v @ e1}")
print(f"read off the e2 component: {v @ e2}   <- unaffected by the e1 part")

# --- high dimensions: random vectors are nearly orthogonal ------------------
print(f"\n{'dim':>6} {'mean |cos|':>12} {'std cos':>10} {'1/sqrt(n)':>11} "
      f"{'% within 5 deg of 90':>22}")
for n in (2, 3, 10, 100, 1000, 10000):
    a = rng.normal(size=(4000, n))
    b = rng.normal(size=(4000, n))
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    cos = np.sum(a * b, axis=1)
    angles = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    near_perp = np.mean(np.abs(angles - 90) < 5) * 100
    print(f"{n:>6} {np.abs(cos).mean():>12.4f} {cos.std():>10.4f} "
          f"{1/np.sqrt(n):>11.4f} {near_perp:>21.1f}%")

print("\nstd of the cosine tracks 1/sqrt(n): in high dimensions two random")
print("directions are almost always close to perpendicular.")

# --- high dimensions: distances concentrate ---------------------------------
print(f"\n{'dim':>6} {'nearest':>10} {'farthest':>10} {'ratio':>8}")
for n in (2, 10, 100, 1000):
    pts = rng.normal(size=(500, n))
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    d = d[~np.eye(len(pts), dtype=bool)]
    print(f"{n:>6} {d.min():>10.3f} {d.max():>10.3f} {d.max()/d.min():>8.2f}")

print("\nThe far/near ratio collapses toward 1 as dimension grows — which is")
print("why exact nearest-neighbour search stops being informative (Part XI).")

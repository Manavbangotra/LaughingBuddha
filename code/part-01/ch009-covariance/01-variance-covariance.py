# Extracted from: Chapter 9 — Variance, Covariance, and Correlation
# Source: src/.../ch009-covariance.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Variance, covariance, correlation, and the dot-product variance result.

Includes the catastrophic-cancellation failure of the naive variance formula
and the zero-correlation-with-perfect-dependence counterexample.
"""
import numpy as np

rng = np.random.default_rng(0)

# --- variance and standard deviation ----------------------------------------
a = np.array([50.0, 50.0, 50.0])
b = np.array([0.0, 50.0, 100.0])
print(f"{'data':<20} {'mean':>7} {'var':>10} {'sd':>8}")
for name, d in (("[50, 50, 50]", a), ("[0, 50, 100]", b)):
    print(f"{name:<20} {d.mean():>7.1f} {d.var():>10.1f} {d.std():>8.2f}")

# --- eq. 9.11: the computational form, and where it breaks ------------------
x = rng.normal(100.0, 5.0, 10_000)
print(f"\ntwo-pass  : {np.mean((x - x.mean())**2):.10f}")
print(f"E[X^2]-E[X]^2: {np.mean(x**2) - x.mean()**2:.10f}   (agree here)")

# Now shift the data far from zero, in single precision.
big = (x + 1e8).astype(np.float32)
naive = np.mean(big**2) - np.mean(big)**2
twopass = np.mean((big - big.mean())**2)
print(f"\nsame data shifted by 1e8, in float32:")
print(f"  naive E[X^2]-E[X]^2 : {naive:>14.4f}   <- catastrophic cancellation")
print(f"  two-pass            : {twopass:>14.4f}")
print(f"  true variance       : {25.0:>14.4f}")
print("The naive form is one pass and numerically unusable at scale.")

# --- eq. 9.14: variance adds only without covariance ------------------------
n = 200_000
u = rng.normal(0, 1, n)
v = rng.normal(0, 1, n)                 # independent of u
w = 0.8 * u + 0.6 * rng.normal(0, 1, n)  # correlated with u

print(f"\n{'pair':<22} {'Var(X)+Var(Y)':>14} {'Var(X+Y)':>10} {'2Cov':>8}")
for name, (p, q) in (("independent", (u, v)), ("correlated", (u, w))):
    print(f"{name:<22} {p.var()+q.var():>14.4f} {(p+q).var():>10.4f} "
          f"{2*np.cov(p, q)[0,1]:>8.4f}")
print("The gap is exactly 2Cov(X, Y) — eq. 9.14.")

# --- section 6.3: zero correlation, perfect dependence ----------------------
X = np.array([-2, -1, 0, 1, 2], dtype=float)
Y = X ** 2
print(f"\nX = {X}\nY = X^2 = {Y}")
print(f"correlation: {np.corrcoef(X, Y)[0,1]:.10f}")
print("Y is a deterministic function of X, yet the correlation is zero.")
print("Correlation measures LINEAR association only.")

# A mutual-information-style check confirms they are far from independent:
# knowing |X| determines Y exactly.
print(f"knowing X determines Y: {np.array_equal(Y, X**2)}")

# --- eq. 9.20: correlation is cosine similarity of centred data -------------
p = rng.normal(size=500)
q = 0.5 * p + rng.normal(size=500)
pc, qc = p - p.mean(), q - q.mean()
cosine_centred = (pc @ qc) / (np.linalg.norm(pc) * np.linalg.norm(qc))
print(f"\ncorrelation        : {np.corrcoef(p, q)[0,1]:.10f}")
print(f"cosine of centred  : {cosine_centred:.10f}   <- identical (eq. 9.20)")
assert np.isclose(np.corrcoef(p, q)[0, 1], cosine_centred)

# --- eq. 9.19: the variance of a dot product is d ---------------------------
print(f"\n{'d':>7} {'Var(q.k) sim':>14} {'predicted = d':>15} {'sd':>9} "
      f"{'sqrt(d)':>9}")
for d in (2, 8, 64, 256, 1024):
    qs = rng.normal(size=(60_000, d))
    ks = rng.normal(size=(60_000, d))
    dots = (qs * ks).sum(axis=1)
    print(f"{d:>7} {dots.var():>14.2f} {d:>15} {dots.std():>9.3f} "
          f"{np.sqrt(d):>9.3f}")
print("\nThe spread grows as sqrt(d). This single fact justifies BOTH the")
print("1/sqrt(d_k) in attention and the 1/sqrt(fan_in) in initialisation.")

# --- the covariance matrix and its eigenvectors -----------------------------
true_cov = np.array([[4.0, 3.0], [3.0, 9.0]])
L = np.linalg.cholesky(true_cov)
data = rng.normal(size=(50_000, 2)) @ L.T

S = np.cov(data, rowvar=False)
print(f"\nsample covariance matrix:\n{np.round(S, 3)}")
print(f"true:\n{true_cov}")

vals, vecs = np.linalg.eigh(S)
order = np.argsort(-vals)
vals, vecs = vals[order], vecs[:, order]
print(f"\neigenvalues (variance along each principal direction): "
      f"{np.round(vals, 3)}")
print(f"top eigenvector (direction of greatest spread): {np.round(vecs[:,0], 3)}")

# eq. 9.17: variance along a direction is w^T Sigma w
w = vecs[:, 0]
print(f"variance of the projection onto it : "
      f"{np.var(data @ w):.4f}")
print(f"w^T Sigma w                        : {w @ S @ w:.4f}   <- eq. 9.17")

# No direction has more variance than the top eigenvector — that is PCA.
best = max((np.var(data @ np.array([np.cos(t), np.sin(t)])), t)
           for t in np.linspace(0, np.pi, 2000))
print(f"best variance over 2000 random directions: {best[0]:.4f} "
      f"(top eigenvalue {vals[0]:.4f})")

# --- eq. 9.18: standardisation ----------------------------------------------
raw = np.column_stack([rng.normal(1000, 200, 5000),    # e.g. price
                       rng.normal(3, 0.5, 5000)])       # e.g. rating
z = (raw - raw.mean(axis=0)) / raw.std(axis=0)
print(f"\nraw column sds : {np.round(raw.std(axis=0), 3)}")
print(f"standardised   : {np.round(z.std(axis=0), 6)}")
print(f"raw covariance matrix condition number : "
      f"{np.linalg.cond(np.cov(raw, rowvar=False)):.1f}")
print(f"standardised                           : "
      f"{np.linalg.cond(np.cov(z, rowvar=False)):.1f}")
print("Standardising fixes the conditioning — which is why gradient descent")
print("on unscaled features zigzags (Chapter 12).")

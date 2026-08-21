# Extracted from: Chapter 9 — Variance, Covariance, and Correlation
# Source: src/.../ch009-covariance.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Screening features by correlation — and the features it silently misses.

A realistic mixture: linearly related features, redundant features, a
nonlinearly related one, and pure noise.
"""
import numpy as np

rng = np.random.default_rng(4)
n = 4000

age = rng.uniform(18, 80, n)
income = 20_000 + 900 * age + rng.normal(0, 12_000, n)     # linear in age
income_k = income / 1000                                    # redundant, rescaled
distance = rng.uniform(-1, 1, n)                            # symmetric
noise = rng.normal(0, 1, n)

# The target depends on income linearly AND on distance quadratically.
target = 0.4 * (income / 1000) + 30 * distance**2 + rng.normal(0, 4, n)

features = {"age": age, "income": income, "income_k": income_k,
            "distance": distance, "noise": noise}

print(f"{'feature':<12} {'corr with target':>18} {'|corr|':>8}")
for name, f in features.items():
    r = np.corrcoef(f, target)[0, 1]
    print(f"{name:<12} {r:>18.4f} {abs(r):>8.4f}")

print("\n'distance' has near-zero correlation but drives 30*distance^2.")
print("A correlation screen would discard the second-strongest predictor.")

# Squaring it first reveals the relationship immediately.
r2 = np.corrcoef(distance**2, target)[0, 1]
print(f"corr(distance^2, target) = {r2:.4f}  <- there it is")

# --- multicollinearity: income and income_k are the same feature ------------
names = list(features)
M = np.corrcoef(np.stack([features[k] for k in names]))
print(f"\nfeature-feature correlation matrix:")
print(f"{'':<11}" + "".join(f"{k:>10}" for k in names))
for i, k in enumerate(names):
    print(f"{k:<11}" + "".join(f"{M[i,j]:>10.3f}" for j in range(len(names))))

print("\nincome and income_k correlate at 1.000 — perfectly redundant.")
print("A linear model cannot separate their coefficients; the design matrix")
print("is rank-deficient (Chapter 4) and the fit is unstable (Chapter 32).")

# A quick diagnostic: the condition number of the feature covariance matrix.
X = np.stack([features[k] for k in names], axis=1)
Xz = (X - X.mean(0)) / X.std(0)
print(f"\ncondition number of the standardised covariance matrix: "
      f"{np.linalg.cond(np.cov(Xz, rowvar=False)):.3e}")
print("Very large — the signature of multicollinearity (Chapter 6).")

X2 = np.stack([features[k] for k in names if k != "income_k"], axis=1)
X2z = (X2 - X2.mean(0)) / X2.std(0)
print(f"after dropping the redundant column: "
      f"{np.linalg.cond(np.cov(X2z, rowvar=False)):.2f}")

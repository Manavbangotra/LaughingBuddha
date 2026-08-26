# -*- coding: utf-8 -*-
# Extracted from: Chapter 32 — Linear Regression from First Principles
# Source: src/.../ch032-linear-regression.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Least squares from the projection argument, and why not to invert.
"""
import numpy as np

rng = np.random.default_rng(0)

# --- section 6.1: fit by projection -----------------------------------------
n, d = 200, 4
X_raw = rng.normal(size=(n, d))
beta_true = np.array([3.0, -1.5, 0.0, 2.0])
y = 5.0 + X_raw @ beta_true + rng.normal(0, 1.0, n)
X = np.column_stack([np.ones(n), X_raw])           # intercept column


def fit_ols_lstsq(X, y):
    """SVD-based least squares — the way you should actually do it."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def fit_ols_qr(X, y):
    """QR factorisation — the projection made explicit."""
    Q, R = np.linalg.qr(X)
    return np.linalg.solve(R, Q.T @ y)


def fit_ols_normal(X, y):
    """The normal equations. Correct in exact arithmetic, fragile in floats."""
    return np.linalg.inv(X.T @ X) @ X.T @ y


beta_hat = fit_ols_lstsq(X, y)
print("true       :", np.round(np.r_[5.0, beta_true], 4))
print("lstsq      :", np.round(beta_hat, 4))
print("QR         :", np.round(fit_ols_qr(X, y), 4))
print("normal eqns:", np.round(fit_ols_normal(X, y), 4))

# --- the orthogonality property (eq. 32.7) ----------------------------------
resid = y - X @ beta_hat
print(f"\nX^T e (should be all zero): {np.abs(X.T @ resid).max():.2e}")
print(f"sum of residuals          : {resid.sum():.2e}")
print(f"corr(residual, fitted)    : "
      f"{np.corrcoef(resid, X @ beta_hat)[0, 1]:.2e}")
print("These are algebraic identities, not evidence the model is right.")

# --- the hat matrix (eq. 32.8) ----------------------------------------------
H = X @ np.linalg.inv(X.T @ X) @ X.T
print(f"\nH idempotent? max|H@H - H| = {np.abs(H @ H - H).max():.2e}")
print(f"trace(H) = {np.trace(H):.4f}   (should equal d+1 = {d + 1})")
lev = np.diag(H)
print(f"leverages: mean {lev.mean():.4f} (= (d+1)/n = {(d + 1) / n:.4f}), "
      f"max {lev.max():.4f}")

# --- section 6.2: what conditioning does to the normal equations ------------
print("\n" + "=" * 72)
print("why NOT to form X^T X  (eq. 32.10)")
print("=" * 72)
print(f"{'kappa(X)':>12} {'kappa(X^T X)':>16} {'lstsq err':>13} "
      f"{'QR err':>13} {'normal-eq err':>15}")

for exponent in (2, 5, 8, 10):
    # build a design matrix with a controlled condition number via its SVD
    m, k = 300, 6
    U, _ = np.linalg.qr(rng.normal(size=(m, k)))
    V, _ = np.linalg.qr(rng.normal(size=(k, k)))
    s = np.logspace(0, -exponent, k)
    Xc = U @ np.diag(s) @ V.T
    b = rng.normal(size=k)
    yc = Xc @ b                                     # noiseless: exact answer known

    kx = np.linalg.cond(Xc)
    kxx = np.linalg.cond(Xc.T @ Xc)

    def err(fn):
        try:
            return np.linalg.norm(fn(Xc, yc) - b) / np.linalg.norm(b)
        except np.linalg.LinAlgError:
            return np.inf

    print(f"{kx:>12.2e} {kxx:>16.2e} {err(fit_ols_lstsq):>13.2e} "
          f"{err(fit_ols_qr):>13.2e} {err(fit_ols_normal):>15.2e}")

print("\nSquaring the condition number costs you half your digits. At")
print("kappa(X) = 1e8 the normal equations have no correct digits left,")
print("while lstsq and QR are still usable.")

# --- section 5.3: multicollinearity inflates variance, not error ------------
print("\n" + "=" * 72)
print("multicollinearity: coefficients explode, predictions do not")
print("=" * 72)


print(f"{'corr(x1,x2)':>12} {'VIF':>9} {'sd(beta1) over 300 fits':>26} "
      f"{'test RMSE':>11}")
for rho in (0.0, 0.9, 0.99, 0.999):
    coefs, rmses = [], []
    for trial in range(300):
        z1 = rng.normal(size=400)
        z2 = rho * z1 + np.sqrt(max(1e-12, 1 - rho ** 2)) * rng.normal(size=400)
        Xm = np.column_stack([z1, z2])
        ym = 2.0 * z1 + 3.0 * z2 + rng.normal(0, 1.0, 400)
        A = np.column_stack([np.ones(300), Xm[:300]])
        bhat = np.linalg.lstsq(A, ym[:300], rcond=None)[0]
        coefs.append(bhat[1])
        B = np.column_stack([np.ones(100), Xm[300:]])
        rmses.append(np.sqrt(np.mean((B @ bhat - ym[300:]) ** 2)))
    v = 1.0 / (1 - rho ** 2) if rho < 1 else np.inf
    print(f"{rho:>12.3f} {v:>9.1f} {np.std(coefs):>26.4f} "
          f"{np.mean(rmses):>11.4f}")

print("\nThe standard deviation of the estimated coefficient grows without")
print("bound as the features align, while out-of-sample RMSE barely moves.")
print("Collinearity is an interpretation problem, not a prediction problem.")

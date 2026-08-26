# -*- coding: utf-8 -*-
# Extracted from: Chapter 32 — Linear Regression from First Principles
# Source: src/.../ch032-linear-regression.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Ridge and lasso: shrinkage in the SVD basis, and why one produces zeros.
"""
import numpy as np

rng = np.random.default_rng(1)


def standardise(Xtr, Xte):
    """Regularisation is scale-dependent, so this is mandatory, not optional."""
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (Xtr - mu) / sd, (Xte - mu) / sd


def ridge_fit(X, y, lam):
    """Closed form (eq. 32.13), intercept handled by centring rather than
    by penalising it."""
    ybar = y.mean()
    A = X.T @ X + lam * np.eye(X.shape[1])
    beta = np.linalg.solve(A, X.T @ (y - ybar))
    return ybar, beta


def lasso_fit(X, y, lam, n_iter=2000, tol=1e-9):
    """Coordinate descent with soft thresholding (eq. 32.17).

    Cycle over coordinates; for each, the univariate solution is the OLS
    residual correlation passed through the soft-threshold operator.
    """
    n, d = X.shape
    ybar = y.mean()
    yc = y - ybar
    beta = np.zeros(d)
    col_ss = (X ** 2).sum(0)
    r = yc - X @ beta
    for _ in range(n_iter):
        delta_max = 0.0
        for j in range(d):
            if col_ss[j] < 1e-12:
                continue
            rho = X[:, j] @ (r + X[:, j] * beta[j])
            new = np.sign(rho) * max(abs(rho) - lam / 2, 0.0) / col_ss[j]
            delta = new - beta[j]
            if delta != 0.0:
                r -= X[:, j] * delta
                beta[j] = new
                delta_max = max(delta_max, abs(delta))
        if delta_max < tol:
            break
    return ybar, beta


# --- section 6.3: ridge as per-direction shrinkage --------------------------
print("=" * 72)
print("ridge shrinks each SVD direction by sigma^2/(sigma^2 + lambda)")
print("=" * 72)

n, d = 300, 8
U, _ = np.linalg.qr(rng.normal(size=(n, d)))
V, _ = np.linalg.qr(rng.normal(size=(d, d)))
svals = np.logspace(0.5, -2.0, d)          # a deliberately ill-conditioned X
X = U @ np.diag(svals) @ V.T
beta_true = rng.normal(size=d)
y = X @ beta_true + rng.normal(0, 0.02, n)

print(f"{'sigma_k':>10} " + " ".join(f"{'l=' + str(l):>10}"
                                     for l in (0.0, 0.001, 0.01, 0.1)))
for k, s in enumerate(svals):
    row = [f"{s ** 2 / (s ** 2 + lam):>10.4f}" if lam > 0 else f"{1.0:>10.4f}"
           for lam in (0.0, 0.001, 0.01, 0.1)]
    print(f"{s:>10.4f} " + " ".join(row))

print("\nHigh-variance directions pass through almost untouched; low-variance")
print("directions — exactly the ones with the worst coefficient variance —")
print("are suppressed. Effective degrees of freedom (eq. 32.15):")
for lam in (0.0, 0.001, 0.01, 0.1, 1.0):
    df = np.sum(svals ** 2 / (svals ** 2 + lam)) if lam > 0 else float(d)
    print(f"  lambda = {lam:<7} df = {df:.3f}  (of {d} parameters)")

# --- section 6.4: soft thresholding vs proportional shrinkage ---------------
print("\n" + "=" * 72)
print("what the two penalties do to a coefficient (eq. 32.17, orthonormal X)")
print("=" * 72)
lam = 1.0
print(f"{'OLS beta':>10} {'ridge':>10} {'lasso':>10}")
for b in (0.05, 0.2, 0.49, 0.51, 1.0, 4.0):
    print(f"{b:>10.2f} {b / (1 + lam):>10.4f} "
          f"{np.sign(b) * max(abs(b) - lam / 2, 0.0):>10.4f}")
print("\nRidge scales; lasso subtracts a constant and clips. Everything below")
print("lambda/2 = 0.50 is set to exactly zero by lasso and merely halved by")
print("ridge. That is feature selection as a side effect of the penalty.")

# --- the penalty is a PRIOR: each wins when its prior is right --------------
def rmse(pred, truth):
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def bake_off(beta_true, label, n=80, n_test=800):
    d = len(beta_true)
    Xall = rng.normal(size=(n + n_test, d))
    yall = Xall @ beta_true + rng.normal(0, 1.0, n + n_test)
    Xtr, ytr = Xall[:n], yall[:n]
    Xte, yte = Xall[n:], yall[n:]
    Xtr, Xte = standardise(Xtr, Xte)

    print("\n" + "=" * 72)
    print(label)
    print("=" * 72)
    bo = np.linalg.lstsq(np.column_stack([np.ones(n), Xtr]), ytr, rcond=None)[0]
    print(f"{'model':<22} {'test RMSE':>10} {'nonzero':>9} {'true kept':>11}")
    print(f"{'OLS':<22} "
          f"{rmse(np.column_stack([np.ones(len(Xte)), Xte]) @ bo, yte):>10.4f} "
          f"{np.sum(np.abs(bo[1:]) > 1e-6):>9d} {'-':>11}")

    for lam in (5.0, 10.0, 20.0, 40.0):
        ic, br = ridge_fit(Xtr, ytr, lam)
        print(f"{'ridge lambda=' + str(lam):<22} "
              f"{rmse(ic + Xte @ br, yte):>10.4f} "
              f"{np.sum(np.abs(br) > 1e-6):>9d} {'-':>11}")

    n_true = int(np.sum(beta_true != 0))
    for lam in (3.0, 10.0, 40.0):
        ic, bl = lasso_fit(Xtr, ytr, lam)
        hits = int(np.sum((np.abs(bl) > 1e-6) & (beta_true != 0)))
        print(f"{'lasso lambda=' + str(lam):<22} "
              f"{rmse(ic + Xte @ bl, yte):>10.4f} "
              f"{np.sum(np.abs(bl) > 1e-6):>9d} {f'{hits}/{n_true}':>11}")


d = 60
sparse = np.zeros(d)
sparse[rng.choice(d, 5, replace=False)] = rng.normal(0, 3.0, 5)
bake_off(sparse, "SPARSE truth: 5 strong features hidden among 60")

dense = rng.normal(0, 0.35, d)      # every feature contributes a little
bake_off(dense, "DENSE truth: all 60 features contribute weakly")

print("\nA penalty is a prior, and it pays off only when the prior is right.")
print("On the sparse problem the lasso more than halves the error and keeps")
print("all five real features while discarding 52 of the 55 fakes, whereas")
print("NO ridge setting meaningfully beats plain OLS — shrinking five")
print("genuinely large coefficients is nearly pure bias with little variance")
print("to buy back. On the dense problem the ordering reverses: ridge is")
print("best, and the lasso stays competitive only by keeping most of the")
print("features, because every zero it sets is a real effect deleted.")
print("'Which regulariser?' is the question 'what do you believe about the")
print("coefficients?' in disguise.")

# --- correlated features: the case where they differ most -------------------
print("\n" + "=" * 72)
print("correlated features: ridge splits credit, lasso picks one")
print("=" * 72)
z = rng.normal(size=(400, 1))
Xc = np.hstack([z + rng.normal(0, 0.01, (400, 1)) for _ in range(3)])
Xc = np.hstack([Xc, rng.normal(size=(400, 2))])
yc = 6.0 * z[:, 0] + rng.normal(0, 0.5, 400)
Xc_s, _ = standardise(Xc, Xc)

_, br = ridge_fit(Xc_s, yc, 1.0)
_, bl = lasso_fit(Xc_s, yc, 20.0)
print("three near-identical copies of one signal, plus two noise features\n")
print(f"{'feature':<12} {'ridge':>10} {'lasso':>10}")
for j in range(5):
    tag = f"copy {j+1}" if j < 3 else f"noise {j-2}"
    print(f"{tag:<12} {br[j]:>10.4f} {bl[j]:>10.4f}")
print(f"{'sum of 3':<12} {br[:3].sum():>10.4f} {bl[:3].sum():>10.4f}")
print("\nRidge divides the signal roughly equally between the copies; lasso")
print("concentrates it. The totals are similar, so predictions agree — but")
print("the lasso's choice among identical features is arbitrary, and would")
print("change with a slightly different sample. Never read a lasso's")
print("selection among correlated features as a finding.")

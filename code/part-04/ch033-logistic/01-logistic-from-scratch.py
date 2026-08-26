# -*- coding: utf-8 -*-
# Extracted from: Chapter 33 — Logistic Regression and Classification
# Source: src/.../ch033-logistic.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Logistic regression from scratch: gradient descent, Newton/IRLS, and the
properties the derivation predicts.
"""
import numpy as np

rng = np.random.default_rng(0)


def sigmoid(z):
    """Numerically stable sigmoid — never exp() a large positive number."""
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def nll(X, y, w, lam=0.0):
    """Mean negative log-likelihood (eq. 33.5), computed stably.

    log(1+exp(z)) via logaddexp avoids overflow for large |z|.
    """
    z = X @ w
    loss = np.mean(np.logaddexp(0, z) - y * z)
    return loss + lam * np.sum(w[1:] ** 2)


def fit_gd(X, y, lr=0.5, n_iter=4000, lam=0.0):
    """Plain gradient descent using eq. 33.11."""
    w = np.zeros(X.shape[1])
    for _ in range(n_iter):
        p = sigmoid(X @ w)
        g = X.T @ (p - y) / len(y)
        g[1:] += 2 * lam * w[1:]            # intercept is never penalised
        w -= lr * g
    return w


def fit_newton(X, y, n_iter=25, lam=1e-8, tol=1e-10):
    """Newton / IRLS using the Hessian of eq. 33.12."""
    w = np.zeros(X.shape[1])
    for it in range(n_iter):
        p = sigmoid(X @ w)
        g = X.T @ (p - y) / len(y)
        g[1:] += 2 * lam * w[1:]
        S = p * (1 - p)
        H = (X * S[:, None]).T @ X / len(y) + 2 * lam * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < tol:
            break
    return w, it + 1


# --- data with known coefficients -------------------------------------------
n, d = 4000, 4
Xr = rng.normal(size=(n, d))
w_true = np.array([-0.4, 1.2, -0.8, 0.5, 1.5])       # intercept first
X = np.column_stack([np.ones(n), Xr])
p_true = sigmoid(X @ w_true)
y = (rng.random(n) < p_true).astype(float)

w_gd = fit_gd(X, y)
w_nt, iters = fit_newton(X, y)
print("true      :", np.round(w_true, 4))
print("grad desc :", np.round(w_gd, 4), f" (4000 iterations)")
print("Newton    :", np.round(w_nt, 4), f" ({iters} iterations)")
print(f"\nfinal loss: GD {nll(X, y, w_gd):.8f}   Newton {nll(X, y, w_nt):.8f}")
print("The same optimum, reached in 7 steps instead of 4000. Newton reads the")
print("step size off the curvature (eq. 33.12) instead of being told one, and")
print("because the problem is convex there is only one optimum to reach.")

# --- eq. 33.11 predicts aggregate calibration -------------------------------
p_hat = sigmoid(X @ w_nt)
print(f"\nsum of fitted probabilities : {p_hat.sum():.4f}")
print(f"number of positives         : {y.sum():.4f}")
print("Equal by construction: setting the gradient to zero forces it")
print("(section 6.2). Logistic regression is calibrated in aggregate whether")
print("or not the model is any good.")

# --- ...but aggregate calibration is not calibration ------------------------
print("\ncalibration by predicted-probability decile:")
print(f"{'bin':>14} {'n':>6} {'mean predicted':>16} {'observed rate':>15}")
edges = np.quantile(p_hat, np.linspace(0, 1, 11))
for i in range(10):
    m = (p_hat >= edges[i]) & (p_hat <= edges[i + 1])
    print(f"  [{edges[i]:.3f},{edges[i+1]:.3f}] {m.sum():>6} "
          f"{p_hat[m].mean():>16.4f} {y[m].mean():>15.4f}")

# --- section 6.1/6.2: why cross-entropy, not squared error ------------------
print("\n" + "=" * 72)
print("the gradient of a confidently WRONG prediction")
print("=" * 72)
print(f"{'z':>7} {'p=sigma(z)':>12} {'sigma prime':>13} "
      f"{'d(CE)/dz':>11} {'d(MSE)/dz':>12}")
for z in (-6.0, -3.0, -1.0, 0.0, 1.0, 3.0, 6.0):
    p = 1 / (1 + np.exp(-z))
    sp = p * (1 - p)
    y_true = 1.0                                  # truth is 1 throughout
    d_ce = p - y_true                             # eq. 33.10
    d_mse = 2 * (p - y_true) * sp                 # chain rule keeps sigma'
    print(f"{z:>7.1f} {p:>12.6f} {sp:>13.6f} {d_ce:>11.6f} {d_mse:>12.8f}")

print("\nAt z = -6 the model says 0.0025 when the truth is 1 — as wrong as it")
print("gets. Cross-entropy delivers a gradient of -0.9975; squared error")
print("delivers -0.00496, two hundred times smaller. Squared error learns")
print("least exactly where it is most wrong, because sigma' has vanished.")
print("Cross-entropy cancels that factor exactly (eq. 33.10).")

# --- section 5.4 / 6.4: complete separation ---------------------------------
print("\n" + "=" * 72)
print("complete separation: the estimate does not exist")
print("=" * 72)
Xs = np.column_stack([np.ones(40), np.linspace(-2, 2, 40)])
ys = (Xs[:, 1] > 0).astype(float)            # perfectly separable by design

print(f"{'iterations':>12} {'|w|':>14} {'loss':>14}")
for it in (100, 1000, 10000, 50000):
    w = fit_gd(Xs, ys, lr=1.0, n_iter=it)
    print(f"{it:>12} {np.linalg.norm(w):>14.4f} {nll(Xs, ys, w):>14.8f}")
print("\nThe norm grows without bound and the loss creeps towards zero but")
print("never arrives. There is no minimiser (section 6.4) — more iterations")
print("only produce larger numbers.")

print(f"\n{'lambda':>12} {'|w| (Newton)':>14} {'|w| (50k GD steps)':>20} "
      f"{'penalised loss':>16}")
for lam in (1e-4, 1e-2, 1e-1, 1.0):
    w_n, _ = fit_newton(Xs, ys, lam=lam)
    w_g = fit_gd(Xs, ys, lr=0.2, n_iter=50000, lam=lam)
    print(f"{lam:>12} {np.linalg.norm(w_n):>14.4f} "
          f"{np.linalg.norm(w_g):>20.4f} {nll(Xs, ys, w_n, lam):>16.8f}")
print("\nAny penalty at all makes the optimum finite, and the two optimisers")
print("now agree on where it is — which they could not do before, because")
print("there was nowhere to agree on. This is why every library regularises")
print("by default.")

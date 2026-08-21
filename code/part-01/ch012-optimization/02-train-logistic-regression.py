# Extracted from: Chapter 12 — Convexity, Gradient Descent, and Numerical Optimization
# Source: src/.../ch012-optimization.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Logistic regression trained by gradient descent, from first principles.

No scikit-learn. Every component comes from Part I.
"""
import numpy as np

rng = np.random.default_rng(3)

# --- data, deliberately on very different scales ----------------------------
n = 4000
age = rng.uniform(18, 80, n)
income = rng.uniform(15_000, 200_000, n)
logit_true = -6.0 + 0.06 * age + 0.00004 * income
y = (rng.random(n) < 1 / (1 + np.exp(-logit_true))).astype(float)

X_raw = np.column_stack([age, income])
mu, sd = X_raw.mean(0), X_raw.std(0)
X_std = (X_raw - mu) / sd                       # eq. 9.18

print(f"positive class rate: {y.mean():.3f}")
print(f"raw feature scales : {np.round(sd, 1)}")


def sigmoid(z):
    out = np.empty_like(z)
    pos, neg = z >= 0, z < 0
    out[pos] = 1 / (1 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1 + ez)
    return out


def add_bias(X):
    return np.column_stack([np.ones(len(X)), X])


def loss_and_grad(w, X, y, lam=0.0):
    """Cross-entropy (eq. 12.16) with an optional L2 penalty (eq. 12.11)."""
    z = X @ w
    p = np.clip(sigmoid(z), 1e-12, 1 - 1e-12)
    ce = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    # eq. 11.19 generalised to a batch: X^T (p - y) / n
    grad = X.T @ (p - y) / len(y)
    if lam:
        penalty_w = w.copy()
        penalty_w[0] = 0.0                      # never regularise the bias
        ce += lam * np.sum(penalty_w ** 2)
        grad += 2 * lam * penalty_w
    return ce, grad


def train(X, y, eta, steps=3000, lam=0.0, beta=0.0):
    w = np.zeros(X.shape[1])
    v = np.zeros_like(w)
    history = []
    for _ in range(steps):
        l, g = loss_and_grad(w, X, y, lam)
        v = beta * v + g                        # eq. 12.10
        w = w - eta * v
        history.append(l)
    return w, np.array(history)


# --- standardised vs raw: the same model, a different optimisation problem --
Xs, Xr = add_bias(X_std), add_bias(X_raw)
print(f"\n{'features':<16} {'eta':>8} {'loss @100':>11} {'loss @3000':>12} "
      f"{'status':<12}")
for name, Xd, eta in (("standardised", Xs, 0.5),
                      ("raw", Xr, 0.5),
                      ("raw", Xr, 1e-9)):
    with np.errstate(over="ignore", invalid="ignore"):
        w, hist = train(Xd, y, eta)
    status = "diverged" if not np.isfinite(hist[-1]) else "ok"
    h100 = hist[100] if np.isfinite(hist[100]) else float("nan")
    print(f"{name:<16} {eta:>8.0e} {h100:>11.4f} {hist[-1]:>12.4f} "
          f"{status:<12}")
print("Raw features diverge at a sensible learning rate and crawl at a safe")
print("one. Standardisation fixes the conditioning (section 6.2).")

# --- momentum ----------------------------------------------------------------
print(f"\n{'optimiser':<22} {'loss after 300 steps':>22}")
for name, beta in (("plain gradient descent", 0.0), ("momentum beta=0.9", 0.9)):
    _, hist = train(Xs, y, eta=0.3, steps=300, beta=beta)
    print(f"{name:<22} {hist[-1]:>22.6f}")

# --- regularisation, and the bias-variance trade ----------------------------
split = 3000
Xtr, ytr, Xte, yte = Xs[:split], y[:split], Xs[split:], y[split:]
print(f"\n{'lambda':>10} {'train loss':>12} {'test loss':>11} "
      f"{'||w||':>8} {'test acc':>10}")
for lam in (0.0, 0.001, 0.01, 0.1, 1.0):
    w, _ = train(Xtr, ytr, eta=0.5, steps=3000, lam=lam)
    tr, _ = loss_and_grad(w, Xtr, ytr)
    te, _ = loss_and_grad(w, Xte, yte)
    acc = np.mean((sigmoid(Xte @ w) > 0.5) == yte)
    print(f"{lam:>10.3f} {tr:>12.4f} {te:>11.4f} "
          f"{np.linalg.norm(w[1:]):>8.3f} {acc:>10.4f}")
print("Stronger regularisation shrinks ||w|| and raises training loss — it is")
print("buying variance reduction with bias (eq. 10.5).")

# --- recover the true coefficients -------------------------------------------
w_final, _ = train(Xs, y, eta=0.5, steps=8000)
# Undo the standardisation to compare against the generating coefficients.
coef = w_final[1:] / sd
intercept = w_final[0] - np.sum(w_final[1:] * mu / sd)
print(f"\nrecovered  : intercept {intercept:+.4f}, coefficients "
      f"{np.array2string(coef, precision=6)}")
print(f"true       : intercept {-6.0:+.4f}, coefficients [0.06     0.00004 ]")

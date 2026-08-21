# Extracted from: Chapter 12 — Convexity, Gradient Descent, and Numerical Optimization
# Source: src/.../ch012-optimization.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Gradient descent: the learning-rate threshold, conditioning, momentum, and
SGD — each claim in the chapter verified numerically.
"""
import numpy as np

rng = np.random.default_rng(0)

# --- eq. 12.10: the stability threshold is exactly 2/a ----------------------
a = 4.0            # f(x) = a x^2 / 2, so f'(x) = a x, curvature a


def run_1d(eta, steps=60, x0=1.0):
    x, traj = x0, [x0]
    for _ in range(steps):
        x = x - eta * (a * x)
        traj.append(x)
        if abs(x) > 1e12:
            break
    return np.array(traj)


print(f"f(x) = {a}x^2/2, so the predicted stability threshold is "
      f"2/a = {2/a}\n")
print(f"{'eta':>8} {'1 - eta*a':>11} {'final |x|':>14} {'behaviour':<26}")
for eta in (0.1, 0.25, 0.4, 0.49, 0.5, 0.55):
    traj = run_1d(eta)
    factor = 1 - eta * a
    final = abs(traj[-1])
    if final > 1e10:
        behaviour = "DIVERGED"
    elif abs(factor) < 1e-12:
        behaviour = "converged in one step"
    elif factor > 0:
        behaviour = "monotone convergence"
    elif abs(factor) < 1:
        behaviour = "oscillating convergence"
    else:
        behaviour = "oscillates forever"
    print(f"{eta:>8.2f} {factor:>11.3f} {final:>14.3e} {behaviour:<26}")
print(f"\nEverything below eta = {2/a} converges; everything above diverges.")

# --- section 6.2: conditioning determines how many steps you need -----------
print("\n" + "=" * 64)
print("condition number vs iterations to converge")
print("=" * 64)


def run_2d(lam1, lam2, steps=200_000, tol=1e-6, beta=0.0):
    """Gradient descent on (lam1 x^2 + lam2 y^2)/2, at the stable learning rate."""
    eta = 1.0 / lam1                      # near-optimal for the steep direction
    v = np.zeros(2)
    p = np.array([1.0, 1.0])
    lams = np.array([lam1, lam2])
    for i in range(steps):
        g = lams * p
        v = beta * v + g
        p = p - eta * v
        if np.linalg.norm(p) < tol:
            return i + 1
    return steps


print(f"{'kappa':>8} {'plain GD steps':>16} {'with momentum':>15} "
      f"{'speed-up':>10}")
for kappa in (1, 10, 100, 1000):
    plain = run_2d(1.0, 1.0 / kappa)
    withmom = run_2d(1.0, 1.0 / kappa, beta=0.9)
    print(f"{kappa:>8} {plain:>16,} {withmom:>15,} "
          f"{plain/max(withmom,1):>9.1f}x")
print("\nIterations scale roughly with kappa (eq. 12.6). Momentum cancels the")
print("oscillation across the valley and accelerates along it.")

# --- standardisation improves conditioning ----------------------------------
n = 2000
raw = np.column_stack([rng.normal(0, 1, n), rng.normal(0, 100, n)])
w_true = np.array([2.0, 0.05])
y = raw @ w_true + rng.normal(0, 0.1, n)

H_raw = raw.T @ raw / n                            # Hessian of the squared loss
std = (raw - raw.mean(0)) / raw.std(0)
H_std = std.T @ std / n
print(f"\ncondition number of the Hessian, raw features        : "
      f"{np.linalg.cond(H_raw):>12,.0f}")
print(f"condition number after standardisation               : "
      f"{np.linalg.cond(H_std):>12,.2f}")
print("Standardising the features is a change to the OPTIMISATION problem,")
print("not to the model. It is why unscaled features train so slowly.")

# --- eq. 12.8: the minibatch gradient is unbiased ---------------------------
print("\n" + "=" * 64)
print("stochastic gradient descent")
print("=" * 64)

N, d = 20_000, 10
X = rng.normal(size=(N, d))
w_star = rng.normal(size=d)
Y = X @ w_star + rng.normal(0, 0.5, N)


def full_gradient(w):
    return X.T @ (X @ w - Y) / N


def batch_gradient(w, size):
    idx = rng.choice(N, size=size, replace=False)
    Xb, Yb = X[idx], Y[idx]
    return Xb.T @ (Xb @ w - Yb) / size


w0 = rng.normal(size=d)
exact = full_gradient(w0)
print(f"{'batch size':>11} {'bias (norm)':>13} {'noise (norm sd)':>17} "
      f"{'cos to exact':>13}")
for bs in (1, 8, 32, 256, 2048):
    grads = np.stack([batch_gradient(w0, bs) for _ in range(600)])
    bias = np.linalg.norm(grads.mean(0) - exact)
    noise = np.linalg.norm(grads - exact, axis=1).std()
    cos = np.mean([g @ exact / (np.linalg.norm(g) * np.linalg.norm(exact))
                   for g in grads])
    print(f"{bs:>11} {bias:>13.5f} {noise:>17.4f} {cos:>13.4f}")
print("\nBias stays near zero at every batch size (eq. 12.8) while the noise")
print("falls as 1/sqrt(batch). Noisy, but never systematically wrong.")

# --- wall-clock: many cheap steps beat few exact ones -----------------------
def train(batch_size, epochs=6, eta=0.05):
    w = np.zeros(d)
    grads_computed = 0
    for _ in range(epochs):
        order = rng.permutation(N)
        for start in range(0, N, batch_size):
            idx = order[start:start + batch_size]
            Xb, Yb = X[idx], Y[idx]
            w = w - eta * (Xb.T @ (Xb @ w - Yb) / len(idx))
            grads_computed += len(idx)
    return np.linalg.norm(w - w_star), grads_computed


print(f"\n{'batch size':>11} {'steps taken':>13} {'gradient evals':>16} "
      f"{'final error':>13}")
for bs in (32, 512, N):
    err, evals = train(bs)
    print(f"{bs:>11} {6*N//bs:>13,} {evals:>16,} {err:>13.5f}")
print("Identical gradient budget; small batches take far more STEPS and get")
print("much closer. This is the whole argument for SGD.")

# --- eq. 12.9: the Robbins-Monro conditions ---------------------------------
print("\n" + "=" * 64)
print("Robbins-Monro step-size conditions (eq. 12.9)")
print("=" * 64)
T = 100_000
t = np.arange(1, T + 1)
schedules = {
    "constant  eta=0.1": np.full(T, 0.1),
    "1/t":               1.0 / t,
    "1/sqrt(t)":         1.0 / np.sqrt(t),
    "1/t^2":             1.0 / t**2,
}
print(f"{'schedule':<20} {'sum eta':>12} {'sum eta^2':>12} "
      f"{'converges?':>12}")
for name, sched in schedules.items():
    s1, s2 = sched.sum(), (sched**2).sum()
    # Condition 1 needs sum eta -> infinity; condition 2 needs sum eta^2 finite.
    ok = s1 > 50 and s2 < 100
    print(f"{name:<20} {s1:>12.2f} {s2:>12.4f} {str(ok):>12}")
print("\n1/t satisfies both. A constant rate fails the second, so it converges")
print("only to a NEIGHBOURHOOD — which is why schedules decay at the end.")

# --- section 6.4: cross-entropy IS maximum likelihood -----------------------
print("\n" + "=" * 64)
print("minimising cross-entropy == maximising likelihood (eq. 12.16)")
print("=" * 64)
probs = np.array([0.9, 0.6, 0.2, 0.75])       # model's p(true label)
likelihood = np.prod(probs)
ce = -np.mean(np.log(probs))
print(f"likelihood  (product)      : {likelihood:.6f}")
print(f"log-likelihood (sum)       : {np.sum(np.log(probs)):.6f}")
print(f"cross-entropy (-mean log)  : {ce:.6f}")
print(f"check: -N * CE == log-lik  : {-len(probs)*ce:.6f}")
assert np.isclose(-len(probs) * ce, np.sum(np.log(probs)))
print("\nThe two are the same objective up to sign and a positive constant,")
print("so they have the same minimiser. This is eq. 1.1 from Chapter 1,")
print("now derived rather than merely read.")

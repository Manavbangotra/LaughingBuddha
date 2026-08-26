# -*- coding: utf-8 -*-
# Extracted from: Chapter 8 — Random Variables, Distributions, and Expectation
# Source: src/.../ch008-random-variables.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Random variables, distributions, and expectation — verified by simulation.

Every analytic formula in the chapter is checked against a large sample.
"""
import numpy as np

rng = np.random.default_rng(0)
N = 500_000

# --- a random variable is a function on the sample space --------------------
d1, d2 = rng.integers(1, 7, N), rng.integers(1, 7, N)
X = d1 + d2                                  # X: Omega -> R, the sum

print(f"E[X] simulated : {X.mean():.4f}")
print(f"E[X] analytic  : {7.0}   (3.5 + 3.5 by linearity)")
values, counts = np.unique(X, return_counts=True)
pmf = counts / N
print(f"pmf sums to    : {pmf.sum():.6f}")
print(f"E[X] from pmf  : {(values * pmf).sum():.4f}")

# --- eq. 8.7: linearity of expectation does NOT need independence -----------
# Deliberately dependent: Y is a function of X.
Y = X ** 2
lhs = (3 * X + 2 * Y).mean()
rhs = 3 * X.mean() + 2 * Y.mean()
print(f"\nE[3X + 2Y] = {lhs:.3f}, 3E[X] + 2E[Y] = {rhs:.3f}  -> equal")
assert np.isclose(lhs, rhs, rtol=1e-9)

# But E[XY] = E[X]E[Y] fails badly for dependent variables (eq. 8.8).
print(f"E[XY]      = {(X * Y).mean():.2f}")
print(f"E[X]E[Y]   = {X.mean() * Y.mean():.2f}   <- not equal; X and Y depend")

ind_a, ind_b = rng.normal(size=N), rng.normal(size=N)
print(f"independent: E[AB] = {(ind_a*ind_b).mean():+.4f}, "
      f"E[A]E[B] = {ind_a.mean()*ind_b.mean():+.4f}   -> equal")

# --- eq. 8.9: Bernoulli mean and variance -----------------------------------
print(f"\n{'p':>5} {'E[X] sim':>10} {'E[X] = p':>10} {'Var sim':>10} "
      f"{'Var = p(1-p)':>13}")
for p in (0.1, 0.3, 0.5, 0.9):
    s = (rng.random(N) < p).astype(float)
    print(f"{p:>5} {s.mean():>10.4f} {p:>10.4f} {s.var():>10.4f} "
          f"{p*(1-p):>13.4f}")
print("variance peaks at p = 0.5 — maximum uncertainty")

# --- densities can exceed 1 -------------------------------------------------
# Uniform on [0, 0.1] has density 10 everywhere on that interval.
u = rng.uniform(0, 0.1, N)
hist, edges = np.histogram(u, bins=10, range=(0, 0.1), density=True)
print(f"\nuniform[0, 0.1] density estimate: {hist.mean():.2f}  <- above 1")
print(f"but it integrates to {(hist * np.diff(edges)).sum():.4f}")
print("A density is probability PER UNIT, not probability.")

# --- Gaussian: the 68-95-99.7 rule ------------------------------------------
g = rng.normal(0.0, 1.0, N)
for k in (1, 2, 3):
    print(f"within {k} sd: {np.mean(np.abs(g) < k):.4f}")

# --- eq. 8.14: the law of the unconscious statistician ----------------------
# E[g(X)] computed from p(x) directly, without deriving the law of g(X).
values, counts = np.unique(d1, return_counts=True)
p_x = counts / N
for name, g in (("X^2", lambda v: v**2), ("log X", np.log),
                ("1/X", lambda v: 1.0 / v)):
    lotus = (g(values) * p_x).sum()
    direct = g(d1).mean()
    print(f"\nE[{name:<5}] via LOTUS: {lotus:.5f} | direct average: {direct:.5f}")
    assert np.isclose(lotus, direct, rtol=1e-3)

# --- eq. 8.19: cross-entropy is a proper scoring rule -----------------------
q = 0.7                                      # true probability of class 1


def expected_ce(p_hat, q=q):
    return -(q * np.log(p_hat) + (1 - q) * np.log(1 - p_hat))


print(f"\ntrue class-1 probability q = {q}")
print(f"{'prediction':>11} {'expected loss':>15}")
for p_hat in (0.5, 0.6, 0.7, 0.8, 0.9, 0.99):
    marker = "  <- minimum, at p_hat = q" if np.isclose(p_hat, q) else ""
    print(f"{p_hat:>11.2f} {expected_ce(p_hat):>15.4f}{marker}")

grid = np.linspace(0.01, 0.99, 9999)
best = grid[np.argmin(expected_ce(grid))]
print(f"\nnumerical minimiser: {best:.4f}  (true q = {q})")
assert abs(best - q) < 0.001

entropy = expected_ce(q)
print(f"minimum achievable loss = entropy of the labels = {entropy:.4f}")
print("No model can do better. Irreducible label noise floors the loss.")

# --- the minibatch gradient is unbiased (section 6.2) -----------------------
per_example_grads = rng.normal(loc=2.0, scale=5.0, size=10_000)
full_gradient = per_example_grads.mean()
batch_means = [rng.choice(per_example_grads, size=32, replace=False).mean()
               for _ in range(3000)]
print(f"\nfull-data gradient        : {full_gradient:.4f}")
print(f"mean of 3000 minibatches  : {np.mean(batch_means):.4f}  <- unbiased")
print(f"sd of a single minibatch  : {np.std(batch_means):.4f}   <- noisy")
print("Noisy but not systematically wrong. That is the entire case for SGD.")

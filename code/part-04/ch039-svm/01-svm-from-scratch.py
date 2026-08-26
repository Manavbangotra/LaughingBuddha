# -*- coding: utf-8 -*-
# Extracted from: Chapter 39 — Support Vector Machines and Kernels
# Source: src/.../ch039-svm.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""SVM from scratch: margin geometry, hinge loss, the dual, and the kernel
trick verified numerically.
"""
import math

import numpy as np

rng = np.random.default_rng(0)

# --- section 6.3: the kernel trick is an identity, not an analogy -----------
print("=" * 72)
print("the RBF kernel IS an inner product in an infinite space (eq. 39.11)")
print("=" * 72)


def rbf_scalar(x, z, gamma=0.5):
    return float(np.exp(-gamma * (x - z) ** 2))


def rbf_feature_map(x, k_max=30):
    """The explicit map of eq. 39.12, truncated at k_max. For gamma = 1/2.

    math.factorial, not a numpy product: 30! is about 2.7e32 and overflows
    int64 silently, which turns the sqrt into a nan.
    """
    k = np.arange(k_max + 1)
    fact = np.array([float(math.factorial(int(i))) for i in k])
    return np.exp(-x ** 2 / 2) * x.astype(float) ** k / np.sqrt(fact) \
        if isinstance(x, np.ndarray) else \
        np.exp(-x ** 2 / 2) * np.power(float(x), k) / np.sqrt(fact)


print(f"{'x':>7} {'z':>7} {'K(x,z) directly':>18} "
      f"{'phi(x).phi(z), 30 terms':>26} {'difference':>13}")
for x, z in ((0.5, 0.7), (1.0, -1.0), (0.0, 2.0), (1.5, 1.6), (-2.0, 2.5)):
    direct = rbf_scalar(x, z)
    viafeat = float(rbf_feature_map(x) @ rbf_feature_map(z))
    print(f"{x:>7.1f} {z:>7.1f} {direct:>18.12f} {viafeat:>26.12f} "
          f"{abs(direct - viafeat):>13.2e}")

print("\nThe two columns agree to twelve decimal places. The left one costs")
print("one exponential; the right one required truncating an INFINITE vector")
print("at 30 terms and taking a dot product. That is the entire trick: the")
print("algorithm never needs phi, only phi(x).phi(z), and K computes that")
print("directly.")

# --- section 5.1: the margin, and which points determine it -----------------
print("\n" + "=" * 72)
print("the maximum-margin hyperplane and its support vectors")
print("=" * 72)


def fit_linear_svm(X, y, C=1.0, n_iter=8000, lr0=0.5):
    """Subgradient descent on eq. 39.3, the unconstrained hinge form.

    y must be in {-1, +1}. lambda = 1/(C*N) as in eq. 39.3.
    """
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    lam = 1.0 / (C * n)
    for t in range(1, n_iter + 1):
        lr = lr0 / (1 + 0.001 * t)
        margins = y * (X @ w + b)
        viol = margins < 1                       # eq. 39.14: zero gradient
        gw = lam * w - (X[viol].T @ y[viol]) / n
        gb = -y[viol].sum() / n
        w -= lr * gw
        b -= lr * gb
    return w, b


# a well-separated problem, so the geometry is unambiguous
n = 200
Xa = rng.normal([-2.5, -1.5], 0.7, (n // 2, 2))
Xb = rng.normal([2.5, 1.8], 0.7, (n // 2, 2))
X = np.vstack([Xa, Xb])
y = np.r_[-np.ones(n // 2), np.ones(n // 2)]

w, b = fit_linear_svm(X, y, C=100.0)
margins = y * (X @ w + b)
print(f"||w||          = {np.linalg.norm(w):.4f}")
print(f"margin 2/||w|| = {2 / np.linalg.norm(w):.4f}   (eq. 39.7)")
print(f"training accuracy = {np.mean(np.sign(X @ w + b) == y):.4f}")
print("\nthe constraint of eq. 39.1 is y_i * f(x_i) >= 1. The margins,")
print("sorted, closest points first:")
srt = np.sort(margins)
print("  five smallest :", " ".join(f"{v:6.3f}" for v in srt[:5]))
print("  five largest  :", " ".join(f"{v:6.3f}" for v in srt[-5:]))
print(f"  median        : {np.median(margins):6.3f}")
print(f"  points within 20% of the closest: "
      f"{int((margins < 1.2 * srt[0]).sum())} of {n}")
print("\nA few points sit at the edge of the street and the rest are several")
print("times further out. Section 6.4 says every point with y*f > 1")
print("contributes EXACTLY zero gradient, so the solution cannot depend on")
print("the far ones at all. Listing 2 makes that exact rather than visual,")
print("by solving the dual and reading off which alpha_i are nonzero.")

# --- section 5.2: what C does -----------------------------------------------
print("\n" + "=" * 72)
print("C prices margin violations (eq. 39.2)")
print("=" * 72)
# a small, well-separated sample with three mislabelled points planted deep
# inside the opposite class — the situation where C genuinely matters
# a small, OVERLAPPING sample with five mislabelled points planted deep
# inside the opposite class. C matters most when the classes overlap and the
# labels are noisy — which is when the boundary's exact position is contested.
n = 60
ctr = np.array([0.8, 0.5])
Xo = np.vstack([rng.normal(-ctr, 1.0, (n // 2, 2)),
                rng.normal(ctr, 1.0, (n // 2, 2))])
yo = np.r_[-np.ones(n // 2), np.ones(n // 2)]
Xo[:5] = rng.normal(ctr * 1.3, 0.25, (5, 2))
yo[:5] = -1.0                      # planted mislabels

Xte = np.vstack([rng.normal(-ctr, 1.0, (4000, 2)),
                 rng.normal(ctr, 1.0, (4000, 2))])
yte = np.r_[-np.ones(4000), np.ones(4000)]

print(f"{'C':>10} {'||w||':>9} {'margin':>9} {'train acc':>11} "
      f"{'test acc':>10} {'# violating':>13}")
for C in (0.003, 0.03, 0.3, 3.0, 100.0):
    wc, bc = fit_linear_svm(Xo, yo, C=C)
    m = yo * (Xo @ wc + bc)
    print(f"{C:>10} {np.linalg.norm(wc):>9.4f} "
          f"{2 / max(np.linalg.norm(wc), 1e-9):>9.4f} "
          f"{np.mean(np.sign(Xo @ wc + bc) == yo):>11.4f} "
          f"{np.mean(np.sign(Xte @ wc + bc) == yte):>10.4f} "
          f"{int((m < 1).sum()):>13}")
print("\nFive of the sixty training points are mislabelled and sit deep")
print("inside the wrong class. Small C treats them as violations to be")
print("tolerated and keeps a wide margin; large C tries harder to classify")
print("them and pulls the boundary towards them.")
print("\nRead the margin and violation columns: they move by an order of")
print("magnitude and monotonically, which is exactly what eq. 39.2 says C")
print("controls. At C = 0.003 violations are so cheap that the model")
print("essentially gives up, keeping a margin of 21 and a training accuracy")
print("of 0.65.")
print("\nNow read the test-accuracy column, which is the honest part: past")
print("C = 0.03 it barely moves. C is a real lever on the SHAPE of the")
print("solution and, on this data, a small one on its accuracy. That is")
print("worth knowing before you spend a grid search on it — the failure to")
print("watch for is C far too SMALL, which is visible immediately in the")
print("training accuracy, rather than C somewhat too large.")
print("\nNote the direction: larger C means LESS regularisation, because C")
print("multiplies the loss rather than the penalty (eq. 39.2). This is the")
print("same inverted convention as scikit-learn's logistic regression, and")
print("it catches people out constantly.")

# --- section 6.4: hinge vs log loss -----------------------------------------
print("\n" + "=" * 72)
print("hinge vs log loss (table 39.1)")
print("=" * 72)
print(f"{'y*f':>7} {'hinge loss':>12} {'hinge grad':>12} "
      f"{'log loss':>10} {'log grad':>10}")
for m in (-2.0, -0.5, 0.0, 0.5, 0.999, 1.0, 1.5, 3.0, 10.0):
    hl = max(0.0, 1 - m)
    hg = -1.0 if m < 1 else 0.0
    ll = float(np.logaddexp(0, -m))
    lg = -1.0 / (1 + np.exp(m))
    print(f"{m:>7.3f} {hl:>12.4f} {hg:>12.4f} {ll:>10.4f} {lg:>10.6f}")

print("\nAt y*f = 3 the hinge gradient is EXACTLY zero and the log-loss")
print("gradient is -0.0474. That exact zero is where support vectors come")
print("from: a confidently correct point leaves the optimisation problem")
print("entirely. It is also where the lack of calibration comes from — the")
print("model has no reason to prefer y*f = 3 to y*f = 10, so its scores")
print("carry no probabilistic information.")

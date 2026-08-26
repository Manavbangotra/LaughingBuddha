# -*- coding: utf-8 -*-
# Extracted from: Chapter 52 — Loss Functions
# Source: src/.../ch052-losses.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Every loss in this chapter, derived and verified: minimisers, gradients,
and the numerical failures of the naive implementations.
"""
import numpy as np

rng = np.random.default_rng(0)


# --- section 6.2: which statistic does each loss recover? -------------------
print("=" * 72)
print("what each regression loss estimates (section 6.2)")
print("=" * 72)


def best_constant(y, loss, grid):
    """Minimise the average loss over a fine grid of constants."""
    return grid[np.argmin([loss(np.full_like(y, a), y).mean() for a in grid])]


mse = lambda p, t: (p - t) ** 2
mae = lambda p, t: np.abs(p - t)


def huber(delta):
    def f(p, t):
        r = np.abs(p - t)
        return np.where(r <= delta, 0.5 * r ** 2, delta * (r - 0.5 * delta))
    return f


clean = rng.normal(10.0, 1.0, 4000)
contaminated = clean.copy()
contaminated[:40] = 200.0                       # 1% gross outliers

grid = np.linspace(0, 40, 8001)
print(f"{'data':<16} {'loss':<12} {'minimiser':>10} {'true mean':>11} "
      f"{'true median':>13}")
for label, y in (("clean", clean), ("1% outliers", contaminated)):
    for name, f in (("squared", mse), ("absolute", mae),
                    ("huber(1.0)", huber(1.0))):
        print(f"{label:<16} {name:<12} {best_constant(y, f, grid):>10.3f} "
              f"{y.mean():>11.3f} {np.median(y):>13.3f}")

print("\nSquared error tracks the MEAN and absolute error the MEDIAN, exactly")
print("as section 6.2 proves. With 1% of the data at 200, the mean moves by")
print("about two units and the median does not move at all — so the squared-")
print("error fit is dragged toward points it will never predict well, and the")
print("absolute-error fit ignores them. Huber sits with the median here")
print("because delta=1.0 puts the outliers deep in its linear region.")

# --- and the gradient argument, which is the REASON -------------------------
print("\nthe reason, in one line: gradient contributed by ONE example")
print(f"{'residual':>10} {'squared':>12} {'absolute':>12} {'huber(1.0)':>12}")
for r in (0.1, 1.0, 10.0, 100.0):
    g_sq, g_ab = 2 * r, 1.0
    g_hu = r if abs(r) <= 1.0 else 1.0
    print(f"{r:>10.1f} {g_sq:>12.1f} {g_ab:>12.1f} {g_hu:>12.1f}")
print("Squared error's gradient is UNBOUNDED in the residual; the other two")
print("are bounded by construction. One mislabelled point at residual 100")
print("contributes 200x the gradient of a typical point under squared error")
print("and 1x under absolute error.")

# --- section 6.4: the softmax cross-entropy gradient, verified numerically --
print("\n" + "=" * 72)
print("the softmax cross-entropy gradient is p - y (eq. 52.15)")
print("=" * 72)


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def ce_from_logits(z, y_idx):
    """Eq. 52.10: stable, and never forms the probabilities."""
    m = z.max(axis=-1, keepdims=True)
    lse = m[..., 0] + np.log(np.exp(z - m).sum(axis=-1))
    return lse - z[np.arange(len(z)), y_idx]


C = 5
z = rng.normal(size=(3, C)) * 2.0
y_idx = rng.integers(0, C, size=3)
onehot = np.eye(C)[y_idx]

analytic = softmax(z) - onehot
numeric = np.zeros_like(z)
eps = 1e-6
for i in range(z.shape[0]):
    for k in range(C):
        zp, zm = z.copy(), z.copy()
        zp[i, k] += eps
        zm[i, k] -= eps
        numeric[i, k] = (ce_from_logits(zp, y_idx)[i]
                         - ce_from_logits(zm, y_idx)[i]) / (2 * eps)

print(f"max |analytic - numerical| = {np.abs(analytic - numeric).max():.3e}")
print("\nThe boxed result of eq. 52.15 is confirmed to central-difference")
print("accuracy. The whole gradient of a classifier's output layer is one")
print("subtraction, because the 1/p_c that would explode for a confident")
print("wrong answer is cancelled exactly by the p_c in the softmax Jacobian.")

# --- section 6.5: why squared error on a sigmoid fails ----------------------
print("\n" + "=" * 72)
print("squared error on a sigmoid saturates; cross-entropy does not")
print("=" * 72)
print("target y = 1, varying how wrong the model is\n")
print(f"{'p_hat':>8} {'|dCE/dz|':>12} {'|dMSE/dz|':>12} {'ratio':>10}")
for p in (0.001, 0.01, 0.1, 0.5, 0.9, 0.99):
    g_ce = abs(p - 1.0)
    g_mse = abs(2 * (p - 1.0) * p * (1 - p))
    print(f"{p:>8.3f} {g_ce:>12.5f} {g_mse:>12.7f} {g_ce / g_mse:>10.1f}x")

print("\nRead the top row. The model assigns probability 0.001 to the correct")
print("class — it could not be more wrong — and squared error responds with a")
print("gradient of 0.002. Cross-entropy responds with 0.999.")
print("\nThe MSE gradient is largest in the MIDDLE and vanishes at both ends,")
print("so a network that starts confidently wrong under this pairing cannot")
print("dig itself out. That is eq. 52.17, and it is the sharpest argument in")
print("the chapter for choosing the loss and the output activation together.")

# --- section 5.4: where the naive implementations break ---------------------
print("\n" + "=" * 72)
print("numerical stability: where each implementation fails (eq. 52.11)")
print("=" * 72)


def ce_naive(z, y_idx):
    """softmax then log — both steps can fail."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        e = np.exp(z)
        p = e / e.sum(axis=-1, keepdims=True)
        return -np.log(p[np.arange(len(z)), y_idx])


def ce_halfstable(z, y_idx):
    """max-subtracted softmax, then log — fixes overflow, not underflow."""
    with np.errstate(divide="ignore"):
        p = softmax(z)
        return -np.log(p[np.arange(len(z)), y_idx])


fmt = lambda v: ("nan" if np.isnan(v) else
                 ("inf" if v > 0 else "-inf") if np.isinf(v) else f"{v:.4f}")

for case, true_is_max in (("A: the model is confidently WRONG "
                           "(true class has the small logit)", False),
                          ("B: the model is confidently RIGHT "
                           "(true class has the large logit)", True)):
    print(f"\ncase {case}")
    print(f"{'max logit':>10} {'naive':>14} {'softmax+log':>14} {'fused':>14}")
    for scale in (1.0, 10.0, 100.0, 400.0, 700.0, 800.0):
        zz = np.zeros((1, 4))
        zz[0, 0] = scale
        yy = np.array([0 if true_is_max else 1])
        a, b, c = (ce_naive(zz, yy)[0], ce_halfstable(zz, yy)[0],
                   ce_from_logits(zz, yy)[0])
        print(f"{scale:>10.0f} {fmt(a):>14} {fmt(b):>14} {fmt(c):>14}")

print("\nThe two cases fail differently and that is the whole point.")
print("\nIn case B the max-subtraction is exactly what is needed: the naive")
print("version overflows once exp(z) exceeds float64's range near 710, while")
print("subtracting the max keeps every exponent at or below zero and the")
print("answer stays correct.")
print("\nIn case A the max-subtraction does not help at all. Both unfused")
print("versions fail at the same magnitude, for different reasons: the")
print("naive one because exp(800) overflows, and the max-subtracted one")
print("because exp(-800) underflows to exactly zero, making the true")
print("class's probability zero and its logarithm -inf. Rearranging the")
print("softmax cannot fix the second — the information was destroyed when")
print("the probability was rounded away.")
print("\nOnly the fused form of eq. 52.10 survives both, because it never")
print("forms a probability at all: it subtracts a logit from a logsumexp and")
print("both are ordinary-sized numbers. This is exactly the case that")
print("matters, since a confidently wrong prediction early in training is")
print("routine. In float16 these failures arrive at logit magnitudes around")
print("11 rather than 700, which any real training run reaches.")

# --- section 6.6: label smoothing bounds the logits -------------------------
print("\n" + "=" * 72)
print("label smoothing replaces an unreachable optimum (eq. 52.20)")
print("=" * 72)
print(f"{'epsilon':>9} {'target p_y':>12} {'optimal logit gap':>19}")
for epsA in (0.0, 0.01, 0.05, 0.1, 0.2):
    C10 = 10
    if epsA == 0.0:
        print(f"{epsA:>9.2f} {1.0:>12.4f} {'unbounded':>19}")
        continue
    t = (1 - epsA) + epsA / C10
    gap = np.log(t / (epsA / C10))
    print(f"{epsA:>9.2f} {t:>12.4f} {gap:>19.2f}")

print("\nWith no smoothing the target probability is exactly 1, which no")
print("finite logit achieves, so the logits grow throughout training with no")
print("stopping point. Any epsilon at all makes the optimum finite — 4.51")
print("nats at the standard epsilon=0.1 with ten classes.")
print("\nThat is the whole mechanism. The regularisation and the calibration")
print("benefit both follow from the optimum being reachable, and so does the")
print("cost: the model can no longer express 'certain', which degrades any")
print("downstream use that needs a confident ranking.")

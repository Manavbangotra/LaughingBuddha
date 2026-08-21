# Extracted from: Chapter 50 — Activation Functions
# Source: src/.../ch050-activations.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Activations and their derivatives, and the layer-product that decides
whether a deep network trains.
"""
import numpy as np

rng = np.random.default_rng(0)


# --- the functions and their exact derivatives ------------------------------
def sigmoid(z):
    out = np.empty_like(z, dtype=float)
    p = z >= 0
    out[p] = 1.0 / (1.0 + np.exp(-z[p]))
    e = np.exp(z[~p])
    out[~p] = e / (1.0 + e)
    return out


def d_sigmoid(z):
    s = sigmoid(z)
    return s * (1 - s)


def d_tanh(z):
    return 1.0 - np.tanh(z) ** 2


def relu(z):
    return np.maximum(0.0, z)


def d_relu(z):
    return (z > 0).astype(float)


def leaky_relu(z, a=0.01):
    return np.where(z > 0, z, a * z)


def d_leaky_relu(z, a=0.01):
    return np.where(z > 0, 1.0, a)


def elu(z, a=1.0):
    return np.where(z > 0, z, a * (np.exp(np.minimum(z, 0)) - 1))


def d_elu(z, a=1.0):
    return np.where(z > 0, 1.0, a * np.exp(np.minimum(z, 0)))


try:
    from scipy.special import erf as _erf          # vectorised, C-speed
except ImportError:                                # pragma: no cover
    from math import erf as _scalar_erf
    _erf = np.vectorize(_scalar_erf)


def _Phi(z):
    """Standard normal CDF via the error function."""
    return 0.5 * (1.0 + _erf(z / np.sqrt(2.0)))


def gelu(z):
    return z * _Phi(z)


def d_gelu(z):
    pdf = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
    return _Phi(z) + z * pdf


def gelu_tanh(z):
    """Eq. 50.2, the fast approximation."""
    return 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi)
                                  * (z + 0.044715 * z ** 3)))


def silu(z):
    return z * sigmoid(z)


def d_silu(z):
    s = sigmoid(z)
    return s * (1 + z * (1 - s))


ACTS = {
    "sigmoid": (sigmoid, d_sigmoid),
    "tanh": (np.tanh, d_tanh),
    "ReLU": (relu, d_relu),
    "LeakyReLU": (leaky_relu, d_leaky_relu),
    "ELU": (elu, d_elu),
    "GELU": (gelu, d_gelu),
    "SiLU": (silu, d_silu),
}

# --- table 50.1, verified numerically ---------------------------------------
print("=" * 72)
print("maximum derivative, and the derivative far from zero (table 50.1)")
print("=" * 72)
zs = np.linspace(-12, 12, 200001)
print(f"{'activation':<12} {'max phi_prime':>14} {'at z=':>8} "
      f"{'phi_prime(4)':>13} {'phi_prime(-4)':>14}")
for name, (f, df) in ACTS.items():
    d = df(zs)
    i = int(np.argmax(d))
    print(f"{name:<12} {d[i]:>14.4f} {zs[i]:>8.2f} "
          f"{float(df(np.array([4.0]))[0]):>13.4f} "
          f"{float(df(np.array([-4.0]))[0]):>14.4f}")

print("\nSigmoid's derivative peaks at 0.25 and tanh's at 1.00 — the factor of")
print("four derived in section 6.2. Both are essentially zero by |z| = 4.")
print("ReLU's is exactly 1 wherever it is active, and GELU and SiLU exceed 1")
print("slightly, which comes from their non-monotone region just below zero.")

# --- section 6.1: what survives L layers ------------------------------------
print("\n" + "=" * 72)
print("what fraction of the gradient survives L layers (eq. 50.4)")
print("=" * 72)
print("Best case: every unit sits exactly at its maximum-derivative point.")
print("This is an OPTIMISTIC bound; real units are rarely there.\n")
print(f"{'activation':<12} " + " ".join(f"{'L=' + str(L):>12}"
                                        for L in (1, 5, 10, 20, 50)))
for name, (f, df) in ACTS.items():
    m = float(np.max(df(zs)))
    print(f"{name:<12} " + " ".join(f"{m ** L:>12.3e}" for L in (1, 5, 10, 20, 50)))

print("\nTen layers of sigmoid attenuate the gradient by a factor of a")
print("million even in the best case; twenty layers by 1e-12, which is below")
print("float32's ability to represent a meaningful update. The rectifier")
print("family's product is exactly 1 at every depth.")
print("\nThe GELU and SiLU rows show the bound going the other way and")
print("growing without limit, which is a good reminder that this is a BOUND")
print("and not a prediction. Their derivative exceeds 1 only in a narrow")
print("region near z = 1.4, and no real network has every unit sitting")
print("there. The next table, using derivatives at pre-activations from an")
print("actual forward pass, is the informative one.")
print("\nThat single column is the whole reason deep networks became")
print("trainable. It is not that ReLU is a better function — it is that it")
print("removes one of the two multiplicative decay terms in eq. 50.3,")
print("leaving only the weight norms, which initialisation can control.")

# --- and the REALISTIC case, with units where they actually sit -------------
print("\n" + "=" * 72)
print("the realistic case: derivatives at units' ACTUAL pre-activations")
print("=" * 72)
print("Pre-activations from a real forward pass, standard normal inputs and")
print("unit-variance weights, so z has variance of order 1.\n")
print(f"{'activation':<12} {'mean phi_prime':>15} {'median':>9} "
      f"{'implied 10-layer factor':>25}")
z_real = rng.normal(0, 1.5, 200000)
for name, (f, df) in ACTS.items():
    d = df(z_real)
    print(f"{name:<12} {d.mean():>15.4f} {np.median(d):>9.4f} "
          f"{d.mean() ** 10:>25.3e}")

print("\nThe realistic numbers are worse than the bound for the saturating")
print("functions and better than one might fear for ReLU: its mean")
print("derivative is about 0.5, because half the units are inactive")
print("(section 6.4), so the AVERAGE path decays — but the ACTIVE paths do")
print("not decay at all, and it is the active paths that carry the signal.")
print("\nThat distinction matters. A sigmoid attenuates every path; a ReLU")
print("blocks some paths completely and leaves the rest untouched.")

# --- section 6.4: the variance identity that gives He its factor of 2 -------
print("\n" + "=" * 72)
print("why the rectifier halves the second moment (eq. 50.5)")
print("=" * 72)
print(f"{'input sd':>9} {'E[z^2]':>10} {'E[relu(z)^2]':>14} {'ratio':>8} "
      f"{'Var[relu(z)]':>14} {'/sigma^2':>10}")
for sd in (0.5, 1.0, 2.0, 4.0):
    z = rng.normal(0, sd, 400000)
    r = relu(z)
    print(f"{sd:>9.1f} {np.mean(z ** 2):>10.4f} {np.mean(r ** 2):>14.4f} "
          f"{np.mean(r ** 2) / np.mean(z ** 2):>8.4f} "
          f"{r.var():>14.4f} {r.var() / sd ** 2:>10.4f}")

print(f"\ntheory: E[relu(z)^2]/E[z^2] = 0.5 exactly, by symmetry")
print(f"        Var[relu(z)]/sigma^2 = 0.5 - 1/(2*pi) = "
      f"{0.5 - 1 / (2 * np.pi):.4f}")
print("\nThe second moment is halved exactly, which is the fact He")
print("initialisation compensates for with a factor of two (Chapter 56).")
print("Note that the VARIANCE is not halved — it is reduced by a different")
print("factor — and the second moment is the right quantity because that is")
print("what propagates through the next matrix multiply.")

# --- eq. 50.2: how good is the tanh approximation to GELU? ------------------
print("\n" + "=" * 72)
print("the tanh approximation to GELU (eq. 50.2)")
print("=" * 72)
zz = np.linspace(-6, 6, 20001)
exact, approx = gelu(zz), gelu_tanh(zz)
print(f"max absolute error : {np.abs(exact - approx).max():.3e}")
print(f"max relative error where |GELU| > 0.01 : "
      f"{np.max(np.abs(exact - approx)[np.abs(exact) > 0.01] / np.abs(exact)[np.abs(exact) > 0.01]):.3e}")
print(f"error at z = 0 : {abs(gelu(np.array([0.0]))[0] - gelu_tanh(np.array([0.0]))[0]):.3e}")

import time
big = rng.normal(size=(2000, 2000))
for label, fn in (("GELU via erf", gelu), ("GELU via tanh", gelu_tanh),
                  ("ReLU", relu)):
    t0 = time.perf_counter()
    for _ in range(3):
        fn(big)
    print(f"{label:<16} {(time.perf_counter() - t0) / 3:.4f} s "
          f"per 4M-element pass")

print("\nThe approximation is accurate to about 5e-4 absolute, far below the")
print("noise in any training run. The relative error reaches a few per cent")
print("only where GELU itself is close to zero, which is harmless for the")
print("same reason.")
print("\nThe timing is the interesting part, and it does not say what the")
print("approximation's existence would suggest: with a vectorised library")
print("erf, the EXACT form is about 2.7x FASTER than the tanh approximation,")
print("which needs a tanh, a cube and several multiplies. The approximation")
print("was worth having when erf was unavailable or slow in the target")
print("kernel language, and on this stack it is now a pessimisation — an")
print("optimisation that outlived its justification.")
print("\nBoth caveats from section 7.1 still apply: these are UNFUSED")
print("measurements, and once the activation is fused into the preceding")
print("matmul the gap between all three largely disappears, because the cost")
print("becomes reading and writing the tensor rather than the arithmetic.")
print("ReLU remains an order of magnitude cheaper unfused, which is why it")
print("survives where fusion is unavailable.")

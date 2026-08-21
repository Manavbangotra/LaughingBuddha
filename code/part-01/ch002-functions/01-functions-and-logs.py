# Extracted from: Chapter 2 — Functions, Exponents, and Logarithms
# Source: src/.../ch002-functions.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Exponentials, logarithms, the logistic curve, and the log-sum-exp trick.

Every identity stated in the chapter is checked numerically here rather than
taken on trust.
"""
import numpy as np

# --- eq. 2.4 / 2.6: exponent and logarithm laws -----------------------------
x, y, n = 7.0, 3.0, 4
assert np.isclose(np.exp(x) * np.exp(y), np.exp(x + y))
assert np.isclose(np.log(x * y), np.log(x) + np.log(y))
assert np.isclose(np.log(x / y), np.log(x) - np.log(y))
assert np.isclose(np.log(x ** n), n * np.log(x))
print("exponent and logarithm laws verified")

# --- logarithms compress scale ----------------------------------------------
values = np.array([1e-9, 1e-3, 1.0, 1e3, 1e9])
print("\nvalues :", values)
print("log10  :", np.log10(values), " <- 18 orders of magnitude become 18 units")


# --- the logistic function and its derivative -------------------------------
def sigmoid(x):
    """Numerically stable logistic function.

    The naive form 1/(1 + exp(-x)) overflows for large negative x, because
    exp(-x) becomes inf. For x < 0 the algebraically identical form
    exp(x)/(1 + exp(x)) keeps every exponent negative and cannot overflow.
    """
    out = np.empty_like(x, dtype=float)
    pos, neg = x >= 0, x < 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out


def sigmoid_derivative(x):
    """eq. 2.16 — expressible in terms of the output alone."""
    s = sigmoid(x)
    return s * (1.0 - s)


xs = np.array([0.0, 2.0, 5.0, 10.0, 20.0])
print(f"\n{'x':>6} {'sigma(x)':>14} {'sigma-prime(x)':>16}")
for xi, s, d in zip(xs, sigmoid(xs), sigmoid_derivative(xs)):
    print(f"{xi:>6.1f} {s:>14.10f} {d:>16.3e}")

# eq. 2.14: the symmetry identity, and the peak derivative of 1/4
assert np.allclose(sigmoid(-xs), 1.0 - sigmoid(xs))
assert np.isclose(sigmoid_derivative(np.array([0.0]))[0], 0.25)

# The naive form really does overflow where the stable one does not.
with np.errstate(over="ignore"):
    naive = 1.0 / (1.0 + np.exp(-np.array([-800.0])))
print(f"\nnaive sigmoid(-800) = {naive[0]}  (overflow in exp(800))")
print(f"stable sigmoid(-800) = {sigmoid(np.array([-800.0]))[0]}")

# --- vanishing gradients are just multiplication -----------------------------
print("\nGradient reaching layer 1 through a stack of sigmoids, best case:")
for depth in (1, 5, 10, 20):
    print(f"  depth {depth:>2}: {0.25 ** depth:.3e}")


# --- eq. 2.18: the log-sum-exp trick ----------------------------------------
def logsumexp(z):
    c = np.max(z)
    return c + np.log(np.sum(np.exp(z - c)))


z = np.array([1000.0, 1001.0, 1002.0])
with np.errstate(over="ignore", invalid="ignore"):
    naive_lse = np.log(np.sum(np.exp(z)))
print(f"\nnaive  log-sum-exp: {naive_lse}")
print(f"stable log-sum-exp: {logsumexp(z):.4f}")
assert np.isclose(logsumexp(z), 1002.4076, atol=1e-4)

# It agrees with the naive form wherever the naive form works at all.
small = np.array([1.0, 2.0, 3.0])
assert np.isclose(logsumexp(small), np.log(np.sum(np.exp(small))))
print("stable and naive agree on inputs the naive form can handle")

# --- eq. 2.12: monotonicity preserves the argmax ----------------------------
rng = np.random.default_rng(0)
likelihoods = rng.random(8) + 0.01
assert likelihoods.argmax() == np.log(likelihoods).argmax()
print("\nargmax of a likelihood == argmax of its log (eq. 2.13)")
print(f"  max likelihood {likelihoods.max():.4f} at index {likelihoods.argmax()}")
print(f"  max log-lik   {np.log(likelihoods).max():.4f} at index "
      f"{np.log(likelihoods).argmax()}  <- same index, different value")

# --- why products of probabilities need logs --------------------------------
probs = rng.uniform(0.3, 0.9, size=2000)
print(f"\nproduct of 2000 probabilities : {np.prod(probs)}  <- underflowed to 0")
print(f"sum of their logs             : {np.sum(np.log(probs)):.2f}  <- fine")

# Extracted from: Chapter 56 — Initialization and Signal Propagation
# Source: src/.../ch056-initialization.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Symmetry breaking, and what orthogonal initialisation preserves that a
Gaussian one does not.
"""
import numpy as np

rng = np.random.default_rng(0)

# --- section 4.2: symmetry ---------------------------------------------------
print("=" * 72)
print("symmetry breaking: what a constant initialisation costs")
print("=" * 72)


def train_tiny(init, steps=400, lr=0.1, seed=0, hidden=8):
    """A tiny network on XOR-like data; returns the hidden units it learns."""
    rs = np.random.default_rng(seed)
    X = rs.normal(size=(400, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(float)[:, None]
    W2 = rs.normal(0, np.sqrt(2 / hidden), (hidden, 1))
    if init == "W1 zero":
        W1 = np.zeros((2, hidden))
    elif init == "W1 constant":
        W1 = np.full((2, hidden), 0.5)
    elif init == "both constant":
        W1 = np.full((2, hidden), 0.5)
        W2 = np.full((hidden, 1), 0.5)          # NO asymmetry anywhere
    else:
        W1 = rs.normal(0, np.sqrt(2 / 2), (2, hidden))
    b1 = np.zeros(hidden)
    b2 = np.zeros(1)
    for _ in range(steps):
        h = np.maximum(0.0, X @ W1 + b1)
        p = 1 / (1 + np.exp(-np.clip(h @ W2 + b2, -60, 60)))
        d2 = (p - y) / len(X)
        d1 = (d2 @ W2.T) * ((X @ W1 + b1) > 0)
        W2 -= lr * (h.T @ d2)
        b2 -= lr * d2.sum(axis=0)
        W1 -= lr * (X.T @ d1)
        b1 -= lr * d1.sum(axis=0)
    h = np.maximum(0.0, X @ W1 + b1)
    p = 1 / (1 + np.exp(-np.clip(h @ W2 + b2, -60, 60)))
    acc = float(((p > 0.5) == y).mean())
    # count distinct hidden functions, up to rounding
    sig = {tuple(np.round(h[:, j], 4)) for j in range(hidden)}
    return acc, len(sig), float(np.abs(W1).std())


print(f"{'initialisation':<16} {'accuracy':>10} {'distinct hidden units':>23} "
      f"{'sd of W1':>10}")
for init in ("W1 zero", "W1 constant", "both constant", "random"):
    acc, n, sd = train_tiny(init)
    print(f"{init:<16} {acc:>10.4f} {n:>23} {sd:>10.4f}")

print("\nThe three failures are NOT the same failure, and the usual")
print("one-line account ('constant weights are symmetric') runs them")
print("together.")
print("\n'both constant' is the pure symmetry case: nothing anywhere")
print("distinguishes the units, so they receive identical gradients")
print("forever, and eight units collapse to one function. One rectified")
print("unit cannot represent XOR, so the network is stuck well below what")
print("the architecture is capable of.")
print("\n'W1 zero' fails for a DIFFERENT reason. Every pre-activation is")
print("exactly zero, so ReLU's mask (z > 0) is false everywhere and the")
print("first layer receives exactly zero gradient at every step. The layer")
print("is not symmetric, it is dead — and it stays dead however long you")
print("train it.")
print("\n'W1 constant' with a RANDOM output layer breaks the symmetry and")
print("trains. The units start identical, and the backward pass multiplies")
print("by W2, which differs per unit — so they receive different gradients")
print("from the very first step and separate. It ends behind the random")
print("initialisation and far ahead of chance.")
print("\nThe correct statement is therefore narrower than the slogan:")
print("symmetry is broken if ANY layer on the path is asymmetric. What you")
print("must not do is make every layer constant, and what you must not do")
print("separately is put a rectifier in a state where its mask is")
print("identically false.")

# --- section 6.4: what orthogonal preserves ---------------------------------
print("\n" + "=" * 72)
print("Gaussian preserves the norm on AVERAGE; orthogonal preserves it")
print("exactly (section 6.4)")
print("=" * 72)


def singular_spread(n, kind, seed=0):
    rs = np.random.default_rng(seed)
    if kind == "gaussian":
        W = rs.normal(0, np.sqrt(1.0 / n), (n, n))
    else:
        Q, R = np.linalg.qr(rs.normal(size=(n, n)))
        W = Q * np.sign(np.diag(R))          # make the QR sign convention fixed
    s = np.linalg.svd(W, compute_uv=False)
    return s


print(f"{'n':>6} {'kind':<12} {'min sv':>9} {'max sv':>9} {'mean sv':>9} "
      f"{'sv spread':>11} {'mean sv^2':>11}")
for n in (64, 256):
    for kind in ("gaussian", "orthogonal"):
        s = singular_spread(n, kind)
        print(f"{n:>6} {kind:<12} {s.min():>9.4f} {s.max():>9.4f} "
              f"{s.mean():>9.4f} {s.max() - s.min():>11.4f} "
              f"{float(np.mean(s ** 2)):>11.4f}")

print("\nBoth have mean squared singular value near 1, which is what the")
print("variance argument of section 6.1 controls — so both 'preserve the")
print("norm' in the sense that calculation means.")
print("\nBut the Gaussian matrix's singular values run from near zero to")
print("about two. Directions aligned with the small ones are crushed and")
print("directions aligned with the large ones are amplified, and across")
print("many layers those distortions compound. The orthogonal matrix's are")
print("all exactly one, so EVERY direction is preserved exactly.")

# --- and what that does across depth ----------------------------------------
print("\n" + "=" * 72)
print("the consequence across depth: a deep LINEAR network")
print("=" * 72)
print("No nonlinearity, so this isolates the matrix product of eq. 53.9.\n")


def deep_linear_spread(depth, n=64, kind="gaussian", seed=1):
    rs = np.random.default_rng(seed)
    M = np.eye(n)
    for _ in range(depth):
        if kind == "gaussian":
            W = rs.normal(0, np.sqrt(1.0 / n), (n, n))
        else:
            Q, R = np.linalg.qr(rs.normal(size=(n, n)))
            W = Q * np.sign(np.diag(R))
        M = W @ M
    s = np.linalg.svd(M, compute_uv=False)
    return s


print(f"{'depth':>7} {'kind':<12} {'min sv':>12} {'max sv':>12} "
      f"{'condition number':>18}")
for depth in (1, 5, 20, 50):
    for kind in ("gaussian", "orthogonal"):
        s = deep_linear_spread(depth, kind=kind)
        print(f"{depth:>7} {kind:<12} {s.min():>12.3e} {s.max():>12.3e} "
              f"{s.max() / max(s.min(), 1e-300):>18.3e}")

print("\nThe orthogonal product stays exactly orthogonal at every depth —")
print("a product of orthogonal matrices is orthogonal, so the condition")
print("number is 1 forever. The Gaussian product's condition number grows")
print("without bound.")
print("\nThat is eq. 53.9's product seen through its singular values, and it")
print("is why eq. 53.16's bound is pessimistic in a specific way: the")
print("NORM can be preserved while the CONDITIONING degrades, and the")
print("second is what makes optimisation hard.")
print("\nThe honest caveat: this is a linear network. Once a nonlinearity is")
print("inserted the product is no longer a product of the weight matrices")
print("alone, and orthogonal initialisation's guarantee weakens to")
print("something closer to the Gaussian one. That is why it is standard in")
print("recurrent networks, where the same matrix recurs, and not elsewhere.")

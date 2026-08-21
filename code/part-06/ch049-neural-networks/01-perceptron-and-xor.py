# Extracted from: Chapter 49 — Neural Networks and the Perceptron
# Source: src/.../ch049-neural-networks.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The perceptron, its convergence bound, and the limitation that stopped
the field for fifteen years.
"""
import numpy as np

rng = np.random.default_rng(0)


# --- the rule (eq. 49.1) ----------------------------------------------------
def perceptron_fit(X, y, max_epochs=1000):
    """y in {-1, +1}. Updates only on mistakes. Returns (w, b, n_updates)."""
    n, d = X.shape
    w, b, updates = np.zeros(d), 0.0, 0
    for epoch in range(max_epochs):
        errors = 0
        for i in range(n):
            if y[i] * (X[i] @ w + b) <= 0:          # a mistake
                w += y[i] * X[i]
                b += y[i]
                updates += 1
                errors += 1
        if errors == 0:
            return w, b, updates, epoch + 1
    return w, b, updates, max_epochs


def margin_under(X, y, w):
    """Margin achieved by a KNOWN unit-norm separator w.

    A random search for the best separator fails badly in high dimensions —
    a random unit vector in 20-D is essentially never a good separator — so
    the honest thing is to use the w the data was generated from. That gives
    a valid lower bound on the true margin, hence a valid UPPER bound on
    (R/gamma)^2, which is what eq. 49.7 needs.
    """
    return float(np.min(y * (X @ w)))


# --- section 6.1: the convergence bound, checked ----------------------------
print("=" * 72)
print("the perceptron convergence bound (eq. 49.7)")
print("=" * 72)
print("The theorem says mistakes <= (R/gamma)^2, independent of n and d.\n")
print(f"{'n':>6} {'d':>4} {'R':>7} {'gamma':>8} {'(R/gamma)^2':>13} "
      f"{'actual updates':>16}")
for n, d, sep in ((50, 2, 1.5), (500, 2, 1.5), (50, 20, 1.5),
                  (500, 20, 1.5), (500, 2, 0.4)):
    # a linearly separable problem with a controlled gap
    w_true = rng.normal(size=d)
    w_true /= np.linalg.norm(w_true)
    X = rng.normal(size=(n, d))
    proj = X @ w_true
    keep = np.abs(proj) > sep / 2                     # carve out a margin
    X, proj = X[keep], proj[keep]
    y = np.sign(proj)
    R = float(np.max(np.linalg.norm(X, axis=1)))
    gamma = margin_under(X, y, w_true)
    _, _, upd, _ = perceptron_fit(X, y)
    bound = (R / gamma) ** 2 if gamma > 0 else np.inf
    print(f"{len(y):>6} {d:>4} {R:>7.3f} {gamma:>8.4f} {bound:>13.1f} "
          f"{upd:>16}")

print("\nThe actual number of updates stays far below the bound in every row,")
print("and — the point of the theorem — it does not grow with n. Going from")
print("14 examples to several hundred, or from 2 dimensions to 20, barely")
print("moves it.")
print("\nThe last row is the one that does move the bound: halving the")
print("enforced gap shrinks gamma and the bound grows as 1/gamma^2, exactly")
print("as eq. 49.7 says. Difficulty for a perceptron is measured by how close")
print("the classes come, not by how much data there is.")

# --- section 6.2: XOR is not representable ----------------------------------
print("\n" + "=" * 72)
print("XOR: the failure is REPRESENTATIONAL, not a training failure")
print("=" * 72)
X_xor = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
y_xor = np.array([-1., 1., 1., -1.])

w, b, upd, epochs = perceptron_fit(X_xor, y_xor, max_epochs=2000)
pred = np.where(X_xor @ w + b > 0, 1.0, -1.0)     # break the tie at zero
print(f"after {epochs:,} epochs and {upd:,} updates the rule has still not")
print("converged — it cycles forever, because there is nothing to converge")
print("to:")
print(f"  weights {np.round(w, 3)}, bias {b:.3f}")
print(f"  accuracy {np.mean(pred == y_xor):.2f}  (chance is 0.50)")
print("\nThe weights returned to exactly zero: the updates cancel over a")
print("cycle. Section 6.1's guarantee assumed separability, and without it")
print("the theorem says nothing at all — not that convergence is slow, but")
print("that there is no fixed point.")

# exhaustive search over a fine grid of ALL possible perceptrons
print("\nsearching every perceptron on a grid, to rule out bad luck:")
best_acc, n_tried = 0.0, 0
grid = np.linspace(-4, 4, 81)
for w1 in grid:
    for w2 in grid:
        for bb in np.linspace(-4, 4, 41):
            n_tried += 1
            acc = np.mean(np.sign(X_xor @ np.array([w1, w2]) + bb) == y_xor)
            best_acc = max(best_acc, acc)
print(f"  {n_tried:,} weight settings tried")
print(f"  best accuracy achievable by ANY perceptron: {best_acc:.2f}")
print("\nNo perceptron reaches 1.00, and the four-line proof in section 6.2")
print("says none ever will. This is the underfitting of Chapter 34: a")
print("property of the hypothesis space, not of the optimiser.")

# --- section 7.3: two hidden units are enough -------------------------------
print("\n" + "=" * 72)
print("a hidden layer changes the hypothesis space (table 49.1)")
print("=" * 72)
W1 = np.array([[1.0, 1.0],        # h1 = OR
               [1.0, 1.0]])       # h2 = AND
b1 = np.array([-0.5, -1.5])
W2 = np.array([[1.0, -1.0]])      # output = h1 - h2
b2 = np.array([-0.5])


def step(z):
    return (z > 0).astype(float)


H = step(X_xor @ W1.T + b1)
out = step(H @ W2.T + b2).ravel()
print(f"{'x1':>4} {'x2':>4} {'h1(OR)':>8} {'h2(AND)':>9} {'output':>8} "
      f"{'XOR':>5}")
for i in range(4):
    print(f"{X_xor[i,0]:>4.0f} {X_xor[i,1]:>4.0f} {H[i,0]:>8.0f} "
          f"{H[i,1]:>9.0f} {out[i]:>8.0f} {(y_xor[i] > 0) * 1:>5}")
print(f"\naccuracy: {np.mean((out > 0.5) == (y_xor > 0)):.2f}")

print("\nLook at the (h1, h2) columns. The four inputs map to three distinct")
print("points and the two POSITIVE cases now coincide at (1, 0). In those")
print("coordinates the classes are linearly separable, and the output unit")
print("is an ordinary perceptron. The hidden layer did not add power to the")
print("classifier; it changed the coordinates the classifier works in.")

# --- section 6.3: without a nonlinearity, depth is free of charge -----------
print("\n" + "=" * 72)
print("no nonlinearity, no depth (eq. 49.9)")
print("=" * 72)
rs = np.random.default_rng(4)
x = rs.normal(size=(6, 5))
Ws = [rs.normal(size=(7, 5)) * 0.5, rs.normal(size=(9, 7)) * 0.5,
      rs.normal(size=(3, 9)) * 0.5]
bs = [rs.normal(size=7) * 0.1, rs.normal(size=9) * 0.1, rs.normal(size=3) * 0.1]

h = x
for W, b in zip(Ws, bs):
    h = h @ W.T + b                              # NO activation
deep_linear = h

W_eq = Ws[2] @ Ws[1] @ Ws[0]
b_eq = Ws[2] @ Ws[1] @ bs[0] + Ws[2] @ bs[1] + bs[2]
single = x @ W_eq.T + b_eq

print(f"3 linear layers (5 -> 7 -> 9 -> 3), {sum(W.size + b.size for W, b in zip(Ws, bs))} parameters")
print(f"equivalent single layer (5 -> 3), {W_eq.size + b_eq.size} parameters")
print(f"max |difference| in outputs: {np.abs(deep_linear - single).max():.2e}")
print("\nIdentical to machine precision. The three-layer network is an")
print("elaborate parameterisation of a 5->3 affine map and represents")
print("nothing a single layer cannot.")

# ...and with a nonlinearity it is not
h = x
for W, b in zip(Ws, bs):
    h = np.tanh(h @ W.T + b)
print(f"\nwith tanh between the layers, max |difference| from the single")
print(f"equivalent layer: {np.abs(h - single).max():.4f}  (no longer affine)")

# --- section 7.4: symmetry breaking -----------------------------------------
print("\n" + "=" * 72)
print("why weights cannot be initialised to a constant (section 7.4)")
print("=" * 72)


def train_mlp(X, y, hidden=8, epochs=600, lr=0.5, init="random", seed=0):
    """A minimal two-layer MLP with tanh and squared error, trained by
    explicit gradients. Backpropagation is derived properly in Chapter 53;
    this is the two-layer case written out by hand."""
    rs = np.random.default_rng(seed)
    d = X.shape[1]
    if init == "zeros":
        W1, W2 = np.zeros((hidden, d)), np.zeros((1, hidden))
    elif init == "constant":
        W1, W2 = np.full((hidden, d), 0.5), np.full((1, hidden), 0.5)
    else:
        W1 = rs.normal(0, 0.8, (hidden, d))
        W2 = rs.normal(0, 0.8, (1, hidden))
    b1, b2 = np.zeros(hidden), np.zeros(1)
    for _ in range(epochs):
        z1 = X @ W1.T + b1
        h1 = np.tanh(z1)
        out = (h1 @ W2.T + b2).ravel()
        err = out - y
        gW2 = (err[:, None] * h1).mean(0, keepdims=True)
        gb2 = err.mean(keepdims=True)
        dh = err[:, None] * W2
        dz = dh * (1 - h1 ** 2)
        gW1 = dz.T @ X / len(y)
        gb1 = dz.mean(0)
        W1 -= lr * gW1
        b1 -= lr * gb1
        W2 -= lr * gW2
        b2 -= lr * gb2
    z1 = X @ W1.T + b1
    h1 = np.tanh(z1)
    out = (h1 @ W2.T + b2).ravel()
    return float(np.mean((out - y) ** 2)), W1, h1


y01 = (y_xor > 0).astype(float)
print(f"{'initialisation':<16} {'final MSE':>11} {'distinct hidden units':>24}")
for init in ("zeros", "constant", "random"):
    mse, W1, h1 = train_mlp(X_xor, y01, init=init)
    n_distinct = len(np.unique(np.round(W1, 6), axis=0))
    print(f"{init:<16} {mse:>11.6f} {n_distinct:>24}")

print("\nWith identical initial weights every hidden unit computes the same")
print("function, receives the same gradient, and stays identical forever —")
print("an 8-unit layer with the capacity of one unit. Random initialisation")
print("breaks the symmetry, and only then does the network solve XOR.")
print("\nBiases can safely start at zero: the weights already break the tie.")

# Extracted from: Chapter 49 — Neural Networks and the Perceptron
# Source: src/.../ch049-neural-networks.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Depth versus width at a fixed parameter budget, and what hidden layers
learn.
"""
import numpy as np

rng = np.random.default_rng(7)


# --- a minimal MLP trained with explicit gradients --------------------------
class MLP:
    """Fully connected, tanh hidden, linear output, squared error.

    Written out by hand rather than with a framework so the parameter count
    and the gradient are both visible. Chapter 53 replaces this with
    reverse-mode autodiff.
    """

    def __init__(self, sizes, seed=0, scale=None):
        rs = np.random.default_rng(seed)
        self.sizes = sizes
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            fan_in = sizes[i]
            s = scale if scale is not None else np.sqrt(1.0 / fan_in)
            self.W.append(rs.normal(0, s, (sizes[i + 1], sizes[i])))
            self.b.append(np.zeros(sizes[i + 1]))

    @property
    def n_params(self):
        return sum(W.size + b.size for W, b in zip(self.W, self.b))

    def forward(self, X):
        acts = [X]
        h = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W.T + b
            h = z if i == len(self.W) - 1 else np.tanh(z)
            acts.append(h)
        return h.ravel() if h.shape[1] == 1 else h, acts

    def fit(self, X, y, epochs=4000, lr=0.05, batch=128, seed=0):
        rs = np.random.default_rng(seed)
        n = len(y)
        for ep in range(epochs):
            idx = rs.integers(0, n, min(batch, n))
            xb, yb = X[idx], y[idx]
            out, acts = self.forward(xb)
            grad = (2.0 * (out - yb) / len(yb))[:, None]
            for i in range(len(self.W) - 1, -1, -1):
                h_in = acts[i]
                gW = grad.T @ h_in
                gb = grad.sum(0)
                if i > 0:
                    grad = (grad @ self.W[i]) * (1 - acts[i] ** 2)
                self.W[i] -= lr * gW
                self.b[i] -= lr * gb
        return self

    def mse(self, X, y):
        out, _ = self.forward(X)
        return float(np.mean((out - y) ** 2))


# --- a target with genuine compositional structure --------------------------
def compositional_target(X, depth):
    """f_depth o ... o f_1, each stage a fixed smooth map of the previous.

    A target built by composition should favour a model that composes. A
    target that is a simple sum should not, and both are measured below.
    """
    h = X.copy()
    for k in range(depth):
        h = np.tanh(1.7 * h @ ROT[k].T + SHIFT[k])
    return h[:, 0] * 2.0


D_IN = 4
ROT = [np.linalg.qr(np.random.default_rng(100 + k).normal(size=(D_IN, D_IN)))[0]
       for k in range(6)]
SHIFT = [np.random.default_rng(200 + k).normal(0, 0.4, D_IN) for k in range(6)]


def additive_target(X, depth=None):
    """A target with no compositional structure: a sum of one-dimensional
    functions. Depth should buy nothing here."""
    return (np.sin(1.5 * X[:, 0]) + 0.8 * X[:, 1]
            - np.abs(X[:, 2]) + 0.5 * X[:, 3] ** 2)


print("=" * 72)
print("depth vs width at a MATCHED parameter budget (section 5.4)")
print("=" * 72)
Xtr = rng.uniform(-2, 2, (3000, D_IN))
Xva = rng.uniform(-2, 2, (1500, D_IN))
Xte = rng.uniform(-2, 2, (4000, D_IN))

LRS = (0.2, 0.05, 0.01, 0.003, 0.001)


def best_fit(sizes, Xtr, ytr, Xva, yva, seed=1):
    """Each architecture gets its own learning rate, selected on VALIDATION
    data. Without this the comparison is unfair: a wide layer produces
    larger gradients and diverges at a rate a narrow one is happy with —
    which is itself a preview of Chapter 56."""
    best = (None, np.inf, None)
    for lr in LRS:
        # rates that are too large diverge to inf/nan; that is information,
        # not an error, so the overflow warnings are suppressed and the
        # non-finite result simply loses the selection
        with np.errstate(over="ignore", invalid="ignore"):
            m = MLP(sizes, seed=seed).fit(Xtr, ytr, epochs=6000, lr=lr, seed=2)
            v = m.mse(Xva, yva)
        if np.isfinite(v) and v < best[1]:
            best = (m, v, lr)
    return best

# architectures chosen to have comparable parameter counts
# widths chosen so the parameter counts genuinely match — a comparison at
# unmatched budgets would be measuring size, not shape
archs = [
    ("1 hidden x 433", [D_IN, 433, 1]),
    ("2 hidden x 48", [D_IN, 48, 48, 1]),
    ("3 hidden x 34", [D_IN, 34, 34, 34, 1]),
    ("5 hidden x 24", [D_IN, 24, 24, 24, 24, 24, 1]),
]

for target_name, target_fn, depth in (
        ("additive (no composition)", additive_target, None),
        ("composition of 4 stages", compositional_target, 4)):
    ytr = (target_fn(Xtr, depth) if depth else target_fn(Xtr)) \
        + rng.normal(0, 0.02, len(Xtr))
    yva = target_fn(Xva, depth) if depth else target_fn(Xva)
    yte = target_fn(Xte, depth) if depth else target_fn(Xte)
    var = float(np.var(yte))
    print(f"\n{target_name}   (target variance {var:.4f})")
    print(f"{'architecture':<18} {'params':>8} {'best lr':>9} "
          f"{'test MSE':>11} {'fraction unexplained':>22}")
    for name, sizes in archs:
        m, _, lr = best_fit(sizes, Xtr, ytr, Xva, yva)
        mse = m.mse(Xte, yte)
        print(f"{name:<18} {m.n_params:>8} {lr:>9} {mse:>11.5f} "
              f"{mse / var:>22.4f}")

print("\nNote the learning-rate column before the accuracy column: the wide")
print("shallow network needs a substantially smaller step than the narrow")
print("deep ones, and diverges outright at rates they are happy with. That")
print("is a preview of Chapter 56 — the scale of a layer's gradient depends")
print("on its width — and it is why this comparison tunes the rate per")
print("architecture rather than fixing one.")
print("\nTwo effects are visible and they need separating.")
print("\nFIRST: going from one hidden layer to two helps on BOTH targets, and")
print("substantially. A single wide layer is simply harder to optimise, and")
print("that is an optimisation fact rather than a representational one — the")
print("theorem in section 5.3 says the wide layer COULD express either")
print("target.")
print("\nSECOND, and this is the depth-separation signature: past two layers")
print("the two targets diverge. On the ADDITIVE target the error flattens —")
print("three and five layers are no better than two, because there is")
print("nothing left to compose. On the COMPOSED target it keeps falling all")
print("the way to five layers, ending roughly thirty times better than the")
print("single wide layer.")
print("\nThat is the mechanism made concrete: a deep network can build a")
print("feature once and reuse it downstream, and a shallow one must")
print("re-derive it for every combination. Depth pays in proportion to how")
print("much compositional structure the target actually has.")
print("\nNote the honest limit of this experiment. It shows depth helping on")
print("a target that was DEFINED by composition. It does not show that real")
print("problems are compositional — that is an empirical bet the field has")
print("made and largely won, not a theorem.")

# --- what the hidden units actually learn -----------------------------------
print("\n" + "=" * 72)
print("what a hidden layer learns (section 4.5)")
print("=" * 72)
print("A two-dimensional problem, so the learned features can be read.\n")


def two_ring(n, rs):
    r = rs.uniform(0, 3, n)
    th = rs.uniform(0, 2 * np.pi, n)
    X = np.column_stack([r * np.cos(th), r * np.sin(th)])
    return X, (r > 1.6).astype(float)


rs = np.random.default_rng(11)
Xr, yr = two_ring(2000, rs)
Xr_te, yr_te = two_ring(3000, rs)

net = MLP([2, 6, 1], seed=3).fit(Xr, yr, epochs=8000, lr=0.1, seed=4)
out, acts = net.forward(Xr_te)
acc = float(np.mean((out > 0.5) == (yr_te > 0.5)))
print(f"test accuracy on concentric rings: {acc:.4f}")

# each hidden unit is a half-plane in the INPUT space; the output layer is
# a linear readout of those half-planes
h = acts[1]
print(f"\n{'hidden unit':>12} {'weight vector':>22} {'output weight':>14} "
      f"{'corr with radius':>18}")
radius = np.linalg.norm(Xr_te, axis=1)
for j in range(6):
    w = net.W[0][j]
    print(f"{j:>12} {str(np.round(w, 2)):>22} {net.W[1][0, j]:>14.3f} "
          f"{np.corrcoef(h[:, j], radius)[0, 1]:>18.3f}")

print("\nEach hidden unit is still a linear boundary in the INPUT space —")
print("that has not changed. What changed is that the output layer sees six")
print("of them at once, and a weighted combination of half-planes can carve")
print("out a ring that no single half-plane can.")
print("\nNotice the last column: no individual unit correlates strongly with")
print("the radius, which is the quantity that actually determines the label.")
print("The representation is DISTRIBUTED — the information is in the pattern")
print("across units, not in any one of them. That is the general case, and it")
print("is why interpreting individual neurons is so difficult (Chapter 229).")

# --- parameter and FLOP arithmetic (eqs. 49.3, 49.4) ------------------------
print("\n" + "=" * 72)
print("why fully connected layers do not scale to images (eq. 49.3)")
print("=" * 72)
print(f"{'input':<22} {'first layer':>14} {'parameters':>16} "
      f"{'GFLOPs @ B=32':>15}")
for label, n_in, n_out in (("4 features", 4, 96),
                           ("784 (28x28 grey)", 784, 1000),
                           ("150,528 (224x224x3)", 224 * 224 * 3, 1000)):
    params = n_out * (n_in + 1)
    flops = 2 * 32 * n_out * n_in / 1e9
    print(f"{label:<22} {n_out:>14,} {params:>16,} {flops:>15.3f}")

print("\nA single fully connected layer on a modest image is 150 million")
print("parameters — more than most complete modern vision models — and it")
print("has learned nothing about images in the process: it treats a pixel and")
print("its neighbour as unrelated coordinates. Chapter 59's convolution is")
print("the response, and it is an argument about ARITHMETIC as much as about")
print("inductive bias.")

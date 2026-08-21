# Extracted from: Chapter 53 — Backpropagation Derived from Scratch
# Source: src/.../ch053-backpropagation.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Reverse-mode automatic differentiation in about eighty lines: the tape of
Chapter 51, walked backwards with eq. 53.13.
"""
import numpy as np

rng = np.random.default_rng(0)


class Tensor:
    """A value, its gradient, and how it was produced."""

    def __init__(self, data, parents=(), backward=None, name=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self.parents = parents
        self._backward = backward or (lambda: None)
        self.name = name

    # --- operations: each records its own VJP (eq. 53.13) -------------------
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), name="add")

        def back():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)
        out._backward = back
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), name="mul")

        def back():
            self.grad += _unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += _unbroadcast(self.data * out.grad, other.data.shape)
        out._backward = back
        return out

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, (self, other), name="matmul")

        def back():                                    # table 53.2, row 1
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = back
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), name="tanh")

        def back():
            self.grad += (1 - t ** 2) * out.grad       # table 53.2, row 2
        out._backward = back
        return out

    def relu(self):
        out = Tensor(np.maximum(0.0, self.data), (self,), name="relu")

        def back():
            self.grad += (self.data > 0) * out.grad
        out._backward = back
        return out

    def sum(self):
        out = Tensor(self.data.sum(), (self,), name="sum")

        def back():
            self.grad += np.ones_like(self.data) * out.grad
        out._backward = back
        return out

    def mean(self):
        out = Tensor(self.data.mean(), (self,), name="mean")

        def back():
            self.grad += np.ones_like(self.data) * out.grad / self.data.size
        out._backward = back
        return out

    def log_softmax_nll(self, y_idx):
        """Fused loss (Chapter 52): stable forward, one-subtraction backward."""
        z = self.data
        m = z.max(axis=1, keepdims=True)
        lse = m[:, 0] + np.log(np.exp(z - m).sum(axis=1))
        loss = np.mean(lse - z[np.arange(len(z)), y_idx])
        out = Tensor(loss, (self,), name="ce")

        def back():
            e = np.exp(z - m)
            p = e / e.sum(axis=1, keepdims=True)
            p[np.arange(len(z)), y_idx] -= 1.0
            self.grad += p / len(z) * out.grad         # eq. 53.3
        out._backward = back
        return out

    # --- the reverse traversal ---------------------------------------------
    def backward(self):
        """Section 7.1: reverse topological order, ACCUMULATING gradients."""
        order, seen = [], set()

        def build(t):
            if id(t) in seen:
                return
            seen.add(id(t))
            for p in t.parents:
                build(p)
            order.append(t)

        build(self)
        self.grad = np.ones_like(self.data)
        for t in reversed(order):
            t._backward()


def _unbroadcast(g, shape):
    """Undo NumPy broadcasting by summing the axes that were expanded."""
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g


# --- it differentiates arbitrary code, not just recognised layers -----------
print("=" * 72)
print("reverse-mode AD, verified on a graph with reuse and broadcasting")
print("=" * 72)

Xd = rng.normal(size=(6, 4))
yd = rng.integers(0, 3, size=6)
params = {
    "W1": Tensor(rng.normal(0, 0.7, (4, 5)), name="W1"),
    "b1": Tensor(np.zeros((1, 5)), name="b1"),
    "W2": Tensor(rng.normal(0, 0.6, (5, 3)), name="W2"),
    "b2": Tensor(np.zeros((1, 3)), name="b2"),
}


def model(p):
    x = Tensor(Xd)
    h = (x @ p["W1"] + p["b1"])
    # deliberately reuse h in two branches AND broadcast b1: both are the
    # cases a naive implementation gets wrong
    h = h.tanh() * h.relu()
    return (h @ p["W2"] + p["b2"]).log_softmax_nll(yd)


for t in params.values():
    t.grad = np.zeros_like(t.data)
loss = model(params)
loss.backward()
print(f"loss = {float(loss.data):.6f}\n")

# Snapshot every gradient NOW. The perturbation loop below re-runs the model,
# which rebinds every .grad — reading them lazily would compare against zeros.
snapshot = {name: t.grad.copy() for name, t in params.items()}

print(f"{'parameter':<10} {'shape':>10} {'max relative error':>21}")
worst = 0.0
for name, t in params.items():
    flat = t.data.reshape(-1)
    gflat = snapshot[name].reshape(-1)
    rels = []
    for i in range(len(flat)):
        orig = flat[i]
        flat[i] = orig + 1e-6
        for u in params.values():
            u.grad = np.zeros_like(u.data)
        lp = float(model(params).data)
        flat[i] = orig - 1e-6
        lm = float(model(params).data)
        flat[i] = orig
        num = (lp - lm) / 2e-6
        rels.append(abs(gflat[i] - num) / max(abs(gflat[i]), abs(num), 1e-8))
    r = float(np.max(rels))
    worst = max(worst, r)
    print(f"{name:<10} {str(t.data.shape):>10} {r:>21.3e}")
print(f"\nworst: {worst:.3e}  ->",
      "CORRECT" if worst < 1e-6 else "CHECK THIS")
print("\nNote what was verified. `b1` has shape (1, 5) and is broadcast over")
print("six rows, so its gradient must sum over the batch — that is what")
print("_unbroadcast does, and getting it wrong gives a gradient six times")
print("too small. And `h` feeds two branches, so its gradient accumulates.")
print("\nNeither case required a special rule. Both fall out of the two")
print("lines that every operation shares: `+=` rather than `=`, and a VJP")
print("that respects the shape it was given.")

# --- eq. 53.11: reverse vs forward mode, measured ---------------------------
print("\n" + "=" * 72)
print("why reverse mode, and not forward mode (table 53.1)")
print("=" * 72)
import time


def timed_reverse(P):
    W = Tensor(rng.normal(0, 0.3, (P, 1)))
    x = Tensor(rng.normal(size=(8, P)))
    t0 = time.perf_counter()
    out = (x @ W).tanh().sum()
    W.grad = np.zeros_like(W.data)
    out.backward()
    return time.perf_counter() - t0


def timed_forward_one(P):
    """Cost of ONE forward-mode pass (one directional derivative)."""
    W = rng.normal(0, 0.3, (P, 1))
    x = rng.normal(size=(8, P))
    v = np.zeros_like(W)
    v[0] = 1.0
    t0 = time.perf_counter()
    z = x @ W
    dz = x @ v
    _ = (1 - np.tanh(z) ** 2) * dz
    return time.perf_counter() - t0


print(f"{'parameters':>11} {'reverse: all grads':>20} "
      f"{'forward: ONE grad':>19} {'forward: all grads':>20}")
for P in (100, 1000, 10000):
    tr = min(timed_reverse(P) for _ in range(5))
    tf = min(timed_forward_one(P) for _ in range(5))
    print(f"{P:>11} {tr * 1e6:>17.1f} us {tf * 1e6:>16.1f} us "
          f"{tf * P:>17.3f} s")
print("\nThe middle column is not slow — a single forward-mode pass is")
print("cheap. The last column is the problem: forward mode needs one pass")
print("PER PARAMETER, so the total scales with the parameter count while")
print("reverse mode does not.")
print("\nThat is table 53.1 as arithmetic. For a model with 10^9 parameters")
print("the last column would be measured in years. Reverse mode is not a")
print("preference; it is the only mode in which training is possible.")

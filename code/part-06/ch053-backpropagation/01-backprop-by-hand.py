# Extracted from: Chapter 53 — Backpropagation Derived from Scratch
# Source: src/.../ch053-backpropagation.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Backpropagation for a dense network, written from eqs. 53.3-53.6 and
verified against central differences.
"""
import numpy as np

rng = np.random.default_rng(0)


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class Net:
    """A dense network with an explicit, equation-by-equation backward pass."""

    def __init__(self, sizes, act="tanh", seed=0):
        rs = np.random.default_rng(seed)
        self.W = [rs.normal(0, np.sqrt(2.0 / sizes[i]),
                            (sizes[i], sizes[i + 1]))
                  for i in range(len(sizes) - 1)]
        self.b = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]
        self.act = act

    def _phi(self, z):
        return np.tanh(z) if self.act == "tanh" else np.maximum(0.0, z)

    def _dphi(self, z, h):
        return 1 - h ** 2 if self.act == "tanh" else (z > 0).astype(z.dtype)

    def forward(self, X):
        """Eq. 53.1. Stores Z and H because eqs. 53.4-53.5 need them."""
        self.Z, self.H = [], [X]
        h = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            self.Z.append(z)
            h = self._phi(z) if i < len(self.W) - 1 else z
            self.H.append(h)
        return h                                   # logits

    def loss(self, X, y_idx):
        logits = self.forward(X)
        m = logits.max(axis=1, keepdims=True)
        lse = m[:, 0] + np.log(np.exp(logits - m).sum(axis=1))
        return float(np.mean(lse - logits[np.arange(len(X)), y_idx]))

    def backward(self, y_idx):
        """Eqs. 53.3, 53.7, 53.8 — one line each, in order."""
        B = len(y_idx)
        onehot = np.eye(self.W[-1].shape[1])[y_idx]
        delta = (softmax(self.H[-1]) - onehot)            # eq. 53.3
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for l in reversed(range(len(self.W))):
            gW[l] = self.H[l].T @ delta / B               # eq. 53.8
            gb[l] = delta.mean(axis=0)                    # eq. 53.8
            if l > 0:
                delta = (delta @ self.W[l].T) * self._dphi(   # eq. 53.7
                    self.Z[l - 1], self.H[l])
        return gW, gb


# --- section 6.7: verify against central differences ------------------------
def gradient_check(net, X, y_idx, eps=1e-5, n_samples=60, seed=0):
    """Eq. 53.18 on a random sample of coordinates, compared by eq. 53.20."""
    net.forward(X)                     # backward() reads what forward stored
    gW, gb = net.backward(y_idx)
    rs = np.random.default_rng(seed)
    worst, results = 0.0, []
    for name, params, grads in (("W", net.W, gW), ("b", net.b, gb)):
        for l, (P, G) in enumerate(zip(params, grads)):
            flat, gflat = P.reshape(-1), G.reshape(-1)
            idx = rs.choice(len(flat), size=min(n_samples, len(flat)),
                            replace=False)
            rels = []
            for i in idx:
                orig = flat[i]
                flat[i] = orig + eps
                lp = net.loss(X, y_idx)
                flat[i] = orig - eps
                lm = net.loss(X, y_idx)
                flat[i] = orig
                num = (lp - lm) / (2 * eps)
                ana = gflat[i]
                rels.append(abs(ana - num)
                            / max(abs(ana), abs(num), 1e-8))
            r = float(np.max(rels))
            results.append((f"{name}[{l}]", P.shape, r))
            worst = max(worst, r)
    net.forward(X)
    return worst, results


print("=" * 72)
print("gradient check: hand-derived backward vs central differences")
print("=" * 72)
X = rng.normal(size=(12, 7))
y = rng.integers(0, 4, size=12)
net = Net([7, 9, 6, 4], act="tanh", seed=1)
worst, results = gradient_check(net, X, y)
print(f"{'parameter':<12} {'shape':>12} {'max relative error':>21}")
for name, shape, r in results:
    print(f"{name:<12} {str(shape):>12} {r:>21.3e}")
print(f"\nworst relative error: {worst:.3e}")
print("verdict:", "CORRECT (< 1e-7)" if worst < 1e-7 else
      "SUSPICIOUS" if worst < 1e-4 else "BUG (> 1e-4)")
print("\nEvery one of eqs. 53.3, 53.7 and 53.8 is confirmed. This check is")
print("ten lines and it is the reason you never have to wonder whether a")
print("hand-written backward pass is right.")

# --- the epsilon trade-off of eq. 53.19 -------------------------------------
print("\n" + "=" * 72)
print("choosing epsilon: truncation against roundoff (eq. 53.19)")
print("=" * 72)
print(f"{'epsilon':>10} {'max rel error':>16} {'dominated by':<24}")
for e in (1e-1, 1e-2, 1e-3, 1e-5, 1e-7, 1e-9, 1e-11, 1e-13):
    w, _ = gradient_check(net, X, y, eps=e, n_samples=25)
    which = ("truncation (O(eps^2))" if e > 1e-5
             else "balanced" if e > 1e-7 else "roundoff (O(eps^-1))")
    print(f"{e:>10.0e} {w:>16.3e} {which:<24}")
print("\nThe error is V-shaped in epsilon, exactly as eq. 53.19 predicts: too")
print("large and the quadratic truncation term dominates, too small and")
print("catastrophic cancellation in the numerator does. The minimum sits")
print("near 1e-5 in float64, which is where the standard advice comes from.")

# --- the ReLU caveat --------------------------------------------------------
print("\n" + "=" * 72)
print("why gradient checking fails on ReLU (section 6.7 warning)")
print("=" * 72)
for act in ("tanh", "relu"):
    net_a = Net([7, 9, 6, 4], act=act, seed=1)
    w, _ = gradient_check(net_a, X, y, n_samples=200, seed=3)
    print(f"{act:<6} worst relative error over 200 coordinates: {w:.3e}")

print("\nBoth pass, which is the honest result and not the one the warning")
print("in section 6.7 might lead you to expect. On a random network no")
print("pre-activation happened to land within epsilon of zero, so the kink")
print("was never crossed and the check never saw it.")
print("\nThe failure is real; it just has to be provoked. Here is the kink")
print("in isolation, checking d/dz of relu(z) at a few distances from it:\n")
eps = 1e-5
print(f"{'z':>12} {'analytic':>10} {'central diff':>14} {'relative error':>16}")
for z in (1.0, 1e-3, 2e-5, 1e-5, 1e-6, 0.0, -1e-6, -2e-5):
    ana = 1.0 if z > 0 else 0.0
    num = (max(0.0, z + eps) - max(0.0, z - eps)) / (2 * eps)
    rel = abs(ana - num) / max(abs(ana), abs(num), 1e-8)
    print(f"{z:>12.1e} {ana:>10.1f} {num:>14.4f} {rel:>16.4f}")

print("\nOutside a band of width 2*epsilon the check is exact. Inside it,")
print("the perturbation straddles the kink and the central difference")
print("returns the AVERAGE of the two one-sided derivatives, so a correct")
print("implementation reports a relative error up to 1.0.")
print("\nThis matters because the natural reaction — 'my ReLU gradient is")
print("broken' — sends people looking for a bug that is not there. The band")
print("is narrow, so on a small check you will usually miss it and on a")
print("large one you will eventually hit it. Check with a smooth activation,")
print("or skip coordinates whose pre-activation is within epsilon of a kink.")

# --- the accumulate-vs-assign bug of section 7.1 ----------------------------
print("\n" + "=" * 72)
print("the accumulate-vs-assign bug, on a branching graph (section 7.1)")
print("=" * 72)
print("f(x) = sum(tanh(Wx) * relu(Wx)) — W is used ONCE but its output")
print("feeds TWO consumers, so its gradient has two contributions.\n")

W0 = rng.normal(size=(5, 5))
x0 = rng.normal(size=(4, 5))


def f(W):
    z = x0 @ W
    return float(np.sum(np.tanh(z) * np.maximum(0.0, z)))


z0 = x0 @ W0
t, r = np.tanh(z0), np.maximum(0.0, z0)
dz_tanh = r * (1 - t ** 2)                    # through the tanh branch
dz_relu = t * (z0 > 0)                        # through the relu branch

g_both = x0.T @ (dz_tanh + dz_relu)           # correct: ACCUMULATE
g_one = x0.T @ dz_relu                        # bug: last write wins

num = np.zeros_like(W0)
for i in range(W0.shape[0]):
    for j in range(W0.shape[1]):
        Wp, Wm = W0.copy(), W0.copy()
        Wp[i, j] += 1e-6
        Wm[i, j] -= 1e-6
        num[i, j] = (f(Wp) - f(Wm)) / 2e-6

rel = lambda g: float(np.max(np.abs(g - num)
                             / np.maximum(np.abs(num), 1e-8)))
print(f"accumulating both branches : relative error {rel(g_both):.3e}")
print(f"assigning (one branch lost): relative error {rel(g_one):.3e}")
print(f"cosine similarity of the two gradients: "
      f"{float(g_both.ravel() @ g_one.ravel() / (np.linalg.norm(g_both) * np.linalg.norm(g_one))):.4f}")
print("\nThat cosine is the reason this bug survives. The wrong gradient")
print("still points broadly downhill, so the model still trains — worse,")
print("and for reasons that look like a bad learning rate. The `+=` in")
print("section 7.1 is not a stylistic choice.")

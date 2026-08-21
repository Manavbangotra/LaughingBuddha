# Extracted from: Chapter 67 — Feed-Forward Networks, Residuals, and Normalization Placement
# Source: src/.../ch067-ffn-residual.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Pre-norm against post-norm (eqs. 67.11-67.12): the gradient argument that
decided every transformer since 2020.
"""
import numpy as np

rng = np.random.default_rng(1)


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def rmsnorm(x, eps=1e-6):
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps)


def rmsnorm_back(x, dy, eps=1e-6):
    """Backward through y = x / rms(x)."""
    d = x.shape[-1]
    ms = (x ** 2).mean(-1, keepdims=True) + eps
    r = np.sqrt(ms)
    return (dy - x * (dy * x).sum(-1, keepdims=True) / (d * ms)) / r


class SimpleBlock:
    """A block with one linear 'sublayer', so the gradient argument is not
    confounded by attention's details. mode is 'pre' or 'post'."""

    def __init__(self, d, mode, seed=0, scale=1.0):
        rs = np.random.default_rng(seed)
        self.W1 = rs.normal(0, scale * np.sqrt(2 / d), (d, 4 * d))
        self.W2 = rs.normal(0, scale * np.sqrt(1 / (4 * d)), (4 * d, d))
        self.mode = mode

    def forward(self, x):
        self.x = x
        if self.mode == "pre":
            self.n = rmsnorm(x)
            self.z = self.n @ self.W1
            self.a = np.maximum(0.0, self.z)
            self.f = self.a @ self.W2
            return x + self.f
        self.z = x @ self.W1
        self.a = np.maximum(0.0, self.z)
        self.f = self.a @ self.W2
        self.s = x + self.f
        return rmsnorm(self.s)

    def backward(self, dy):
        if self.mode == "pre":
            df = dy
            da = df @ self.W2.T
            dz = da * (self.z > 0)
            dn = dz @ self.W1.T
            return dy + rmsnorm_back(self.x, dn)
        ds = rmsnorm_back(self.s, dy)
        da = ds @ self.W2.T
        dz = da * (self.z > 0)
        return ds + dz @ self.W1.T


print("=" * 72)
print("the gradient reaching each layer (eqs. 67.11-67.12)")
print("=" * 72)
d, B, T = 96, 8, 16
print("A stack of identical blocks, gradient of RMS 1 injected at the top.")
print("The question is what reaches the bottom.\n")
print(f"{'depth':>7} {'mode':>6} " +
      " ".join(f"{f'layer {i}':>11}" for i in ("L", "3L/4", "L/2", "1"))
      + f" {'ratio top/bottom':>18}")
for L in (8, 24, 48):
    for mode in ("pre", "post"):
        blocks = [SimpleBlock(d, mode, seed=200 + i) for i in range(L)]
        x = rng.normal(size=(B, T, d))
        for b in blocks:
            x = b.forward(x)
        g = rng.normal(size=x.shape)
        g = g / np.sqrt((g ** 2).mean())
        norms = [float(np.sqrt((g ** 2).mean()))]
        for b in reversed(blocks):
            g = b.backward(g)
            norms.append(float(np.sqrt((g ** 2).mean())))
        picks = [0, L // 4, L // 2, L]
        print(f"{L:>7} {mode:>6} " +
              " ".join(f"{norms[i]:>11.3e}" for i in picks)
              + f" {norms[0] / max(norms[L], 1e-300):>18.3e}")

print("\nRead the last column: the gradient at the top divided by what")
print("reaches the bottom. A value near 1 means the gradient crossed the")
print("whole stack intact.")
print("\nEq. 67.11 says pre-norm has an EXACT identity term in its Jacobian —")
print("the normalisation is on the branch, not on the skip — so the product")
print("over L layers contains a gain-1 path however deep the stack is.")
print("\nEq. 67.12 says post-norm does not: the normalisation's Jacobian")
print("multiplies EVERYTHING including the identity, so L such factors")
print("accumulate. That is the mechanism Xiong et al. analyse, and warmup")
print("is what post-norm models use to survive the early steps until the")
print("parameters move somewhere less hostile.")

# --- the same, as an optimisation problem -----------------------------------
print("\n" + "=" * 72)
print("what that costs in training, with and without warmup")
print("=" * 72)


def train_stack(L, mode, lr=3e-3, warmup=0, steps=1500, d=48, seed=0):
    """Fit a random target through a deep stack; report the final loss."""
    rs = np.random.default_rng(seed)
    blocks = [SimpleBlock(d, mode, seed=300 + i) for i in range(L)]
    X = rs.normal(size=(64, 8, d))
    Ytgt = rs.normal(size=(64, 8, d)) * 0.5
    ps = []
    for b in blocks:
        ps += [b.W1, b.W2]
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    for t in range(1, steps + 1):
        x = X
        for b in blocks:
            x = b.forward(x)
        loss = float(((x - Ytgt) ** 2).mean())
        if not np.isfinite(loss):
            return float("inf")
        g = 2 * (x - Ytgt) / x.size
        grads = []
        for b in reversed(blocks):
            if b.mode == "pre":
                df = g
                da = df @ b.W2.T
                dz = da * (b.z > 0)
                grads.append(b.a.reshape(-1, 4 * d).T @ df.reshape(-1, d))
                grads.append(b.n.reshape(-1, d).T @ dz.reshape(-1, 4 * d))
                g = g + rmsnorm_back(b.x, dz @ b.W1.T)
            else:
                ds = rmsnorm_back(b.s, g)
                da = ds @ b.W2.T
                dz = da * (b.z > 0)
                grads.append(b.a.reshape(-1, 4 * d).T @ ds.reshape(-1, d))
                grads.append(b.x.reshape(-1, d).T @ dz.reshape(-1, 4 * d))
                g = ds + dz @ b.W1.T
        grads = grads[::-1]
        order = []
        for i in range(L):
            order += [grads[2 * i + 1], grads[2 * i]]
        cur = lr * min(1.0, t / warmup) if warmup else lr
        for i, (p, gr) in enumerate(zip(ps, order)):
            m[i] = 0.9 * m[i] + 0.1 * gr
            v[i] = 0.999 * v[i] + 0.001 * gr * gr
            p -= cur * (m[i] / (1 - 0.9 ** t)) / (
                np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    x = X
    for b in blocks:
        x = b.forward(x)
    return float(((x - Ytgt) ** 2).mean())


print(f"{'depth':>7} {'mode':>6} {'no warmup':>13} {'warmup 200':>13}")
for L in (8, 24, 48):
    for mode in ("pre", "post"):
        a = train_stack(L, mode, warmup=0)
        b = train_stack(L, mode, warmup=200)
        f = lambda z: "diverged" if not np.isfinite(z) or z > 10 else f"{z:.5f}"
        print(f"{L:>7} {mode:>6} {f(a):>13} {f(b):>13}")

print("\nThis is Xiong et al.'s claim as an experiment: post-norm should")
print("benefit from warmup and pre-norm should not need it, with the gap")
print("widening as the stack deepens.")
print("\nThe simplification to watch is that these blocks have no attention,")
print("so what is being tested is the NORMALISATION PLACEMENT alone,")
print("isolated from everything else in a transformer. That is the point of")
print("the simplification and it is also its limit.")

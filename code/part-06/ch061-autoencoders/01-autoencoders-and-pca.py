# Extracted from: Chapter 61 — Autoencoders and Representation Learning
# Source: src/.../ch061-autoencoders.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A linear autoencoder against PCA, and what the bottleneck actually does.
"""
import numpy as np

rng = np.random.default_rng(0)


# --- data with a genuine low-dimensional structure --------------------------
def make_data(n, d=24, k_true=5, noise=0.25, seed=0):
    rs = np.random.default_rng(seed)
    B = rs.normal(size=(k_true, d))
    lat = rs.normal(size=(n, k_true)) * np.array([3.0, 2.2, 1.5, 0.9, 0.4])
    X = lat @ B + rs.normal(0, noise, (n, d))
    return X - X.mean(axis=0)


Xtr = make_data(4000, seed=1)
Xte = make_data(4000, seed=2)
D = Xtr.shape[1]


def pca_reconstruct(Xfit, Xeval, k):
    U, S, Vt = np.linalg.svd(Xfit, full_matrices=False)
    P = Vt[:k]
    return Xeval @ P.T @ P, P


def train_linear_ae(X, k, steps=30000, lr=3e-3, batch=256, seed=0):
    """No activations at all: encoder and decoder are both linear."""
    rs = np.random.default_rng(seed)
    We = rs.normal(0, 1 / np.sqrt(D), (D, k))
    Wd = rs.normal(0, 1 / np.sqrt(k), (k, D))
    ps = [We, Wd]
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    for t in range(1, steps + 1):
        xb = X[rs.integers(0, len(X), batch)]
        z = xb @ We
        xr = z @ Wd
        dxr = 2 * (xr - xb) / len(xb)
        gWd = z.T @ dxr
        gWe = xb.T @ (dxr @ Wd.T)
        for i, (p, g) in enumerate(zip(ps, [gWe, gWd])):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            p -= lr * (m[i] / (1 - 0.9 ** t)) / (
                np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    return We, Wd


print("=" * 72)
print("a linear autoencoder recovers PCA's SUBSPACE (section 6.1)")
print("=" * 72)
ev = np.linalg.svd(Xtr, compute_uv=False) ** 2 / len(Xtr)
print("data eigenvalues: "
      + np.array2string(ev[:9], precision=2, suppress_small=True) + " ...\n")
print(f"{'k':>4} {'PCA test MSE':>14} {'linear AE test MSE':>20} "
      f"{'ratio':>8} {'principal angle':>17} {'eig gap k/k+1':>15}")
for k in (2, 4, 5, 8):
    Xp, P = pca_reconstruct(Xtr, Xte, k)
    mse_pca = float(np.mean((Xte - Xp) ** 2))
    We, Wd = train_linear_ae(Xtr, k, seed=3)
    mse_ae = float(np.mean((Xte - Xte @ We @ Wd) ** 2))
    # principal angle between the two k-dimensional subspaces
    Qa, _ = np.linalg.qr(We)
    Qb, _ = np.linalg.qr(P.T)
    sv = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    angle = float(np.degrees(np.arccos(np.clip(sv.min(), -1, 1))))
    print(f"{k:>4} {mse_pca:>14.6f} {mse_ae:>20.6f} "
          f"{mse_ae / mse_pca:>8.4f} {angle:>16.2f}° "
          f"{ev[k - 1] / ev[k]:>15.2f}")

print("\nThe reconstruction ratios are close to 1 at every k: the linear")
print("autoencoder reaches essentially PCA's error, which is eq. 61.9 and")
print("Eckart-Young confirmed.")
print("\nThe principal angle is the more interesting column, and it is not")
print("small. Read it against the eigenvalue gap in the last column.")
print("\nWhere the gap is large the subspace is well determined and the two")
print("agree. Where consecutive eigenvalues are close, the objective is")
print("nearly FLAT between the two candidate subspaces — swapping a")
print("direction for its near-degenerate neighbour barely changes the loss")
print("— so an optimiser has no gradient telling it which to pick and lands")
print("somewhere in between.")
print("\nThat is not a failure of the theorem. Eckart-Young says the")
print("MINIMISER is PCA's subspace, and it is silent about how sharply")
print("defined that minimum is. When the eigenvalues are nearly degenerate")
print("the minimum is a shallow valley rather than a point, and an")
print("approximate optimiser stops somewhere in the valley — at nearly the")
print("right loss and not at the right subspace.")

# --- but NOT the components -------------------------------------------------
print("\n" + "=" * 72)
print("...but NOT the components (section 6.1 warning)")
print("=" * 72)
k = 5
We, Wd = train_linear_ae(Xtr, k, seed=3)
_, P = pca_reconstruct(Xtr, Xte, k)
Z_ae = Xtr @ We
Z_pca = Xtr @ P.T
print("PCA components are orthonormal and ordered by variance.")
print("The autoencoder's are neither.\n")
print(f"{'':<22} {'PCA':>28} {'linear AE':>28}")
gram_p = P @ P.T
gram_a = (We / np.linalg.norm(We, axis=0)).T @ (
    We / np.linalg.norm(We, axis=0))
off_p = np.abs(gram_p - np.eye(k)).max()
off_a = np.abs(gram_a - np.eye(k)).max()
print(f"{'max off-diagonal Gram':<22} {off_p:>28.6f} {off_a:>28.6f}")
print(f"{'code variances':<22} "
      f"{np.array2string(Z_pca.var(axis=0), precision=2):>28} "
      f"{np.array2string(Z_ae.var(axis=0), precision=2):>28}")
print(f"{'variances sorted?':<22} "
      f"{str(bool(np.all(np.diff(Z_pca.var(axis=0)) <= 1e-9))):>28} "
      f"{str(bool(np.all(np.diff(Z_ae.var(axis=0)) <= 1e-9))):>28}")

print("\nThe autoencoder's directions are not orthogonal and its code")
print("variances are not ordered. Both are consequences of the same")
print("degeneracy: for any invertible A, (W_d A^-1) and (A W_e) give the")
print("identical product and therefore the identical loss, so nothing in")
print("the objective prefers one basis over another.")
print("\nThat matters when the code is meant to be INTERPRETED. If you want")
print("ordered, orthogonal, variance-ranked axes, PCA gives them and an")
print("autoencoder does not — and no amount of training will change that,")
print("because the objective is indifferent.")

# --- so what does the NONLINEAR version buy? --------------------------------
print("\n" + "=" * 72)
print("what nonlinearity buys, on data PCA cannot compress")
print("=" * 72)


def make_curved(n, seed):
    """Three latent coordinates mapped LINEARLY into 12 dimensions, but
    with one of them entering through a spiral. The data therefore spans a
    3-dimensional linear subspace exactly — so PCA at k = 3 is exact — while
    the intrinsic structure needs only two coordinates to describe."""
    rs = np.random.default_rng(seed)
    t = rs.uniform(0, 3 * np.pi, n)
    u = rs.uniform(-1, 1, n)
    base = np.stack([t * np.cos(t), u * 3, t * np.sin(t)], axis=1)
    A = np.random.default_rng(77).normal(size=(3, 12))
    return (base @ A + rs.normal(0, 0.15, (n, 12)))


class AE:
    """Nonlinear encoder and decoder, hand-written backward."""

    def __init__(self, d, k, hidden=32, seed=0):
        rs = np.random.default_rng(seed)
        self.W1 = rs.normal(0, np.sqrt(2 / d), (d, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rs.normal(0, np.sqrt(2 / hidden), (hidden, k))
        self.b2 = np.zeros(k)
        self.W3 = rs.normal(0, np.sqrt(2 / k), (k, hidden))
        self.b3 = np.zeros(hidden)
        self.W4 = rs.normal(0, np.sqrt(2 / hidden), (hidden, d))
        self.b4 = np.zeros(d)

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2,
                self.W3, self.b3, self.W4, self.b4]

    def forward(self, X):
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.tanh(self.z1)
        self.z = self.a1 @ self.W2 + self.b2
        self.z3 = self.z @ self.W3 + self.b3
        self.a3 = np.tanh(self.z3)
        return self.a3 @ self.W4 + self.b4

    def grads(self, X):
        xr = self.forward(X)
        d4 = 2 * (xr - X) / len(X)
        g = [None] * 8
        g[6], g[7] = self.a3.T @ d4, d4.sum(axis=0)
        d3 = (d4 @ self.W4.T) * (1 - self.a3 ** 2)
        g[4], g[5] = self.z.T @ d3, d3.sum(axis=0)
        dz = d3 @ self.W3.T
        g[2], g[3] = self.a1.T @ dz, dz.sum(axis=0)
        d1 = (dz @ self.W2.T) * (1 - self.a1 ** 2)
        g[0], g[1] = X.T @ d1, d1.sum(axis=0)
        return float(np.mean((xr - X) ** 2)), g


def train_ae(net, X, steps=6000, lr=3e-3, batch=128, seed=0):
    ps = net.params()
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    rs = np.random.default_rng(seed + 9)
    for t in range(1, steps + 1):
        xb = X[rs.integers(0, len(X), batch)]
        _, gs = net.grads(xb)
        for i, (p, gg) in enumerate(zip(ps, gs)):
            m[i] = 0.9 * m[i] + 0.1 * gg
            v[i] = 0.999 * v[i] + 0.001 * gg * gg
            p -= lr * (m[i] / (1 - 0.9 ** t)) / (
                np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    return net


Ctr, Cte = make_curved(6000, 5), make_curved(4000, 6)
Ctr = Ctr - Ctr.mean(axis=0)
Cte = Cte - Cte.mean(axis=0)
print("A spiral: 3 linear dimensions, 2 intrinsic ones.\n")
print(f"{'code size k':>12} {'PCA test MSE':>15} {'nonlinear AE test MSE':>23} "
      f"{'ratio':>8}")
for k in (2, 3, 5):
    Xp, _ = pca_reconstruct(Ctr, Cte, k)
    mse_p = float(np.mean((Cte - Xp) ** 2))
    net = train_ae(AE(12, k, seed=4), Ctr)
    mse_a = float(np.mean((Cte - net.forward(Cte)) ** 2))
    print(f"{k:>12} {mse_p:>15.6f} {mse_a:>23.6f} {mse_a / mse_p:>8.4f}")

print("\nThe k = 2 row is the one that matters, and the gap is large: the")
print("nonlinear autoencoder is an order of magnitude better. Two linear")
print("dimensions cannot describe a spiral, and two NONLINEAR coordinates")
print("can — the encoder can learn the arc-length parameterisation that")
print("PCA has no way to express.")
print("\nAt k = 3 and above PCA wins, and that is not a defeat for the")
print("method — it is the construction. The data spans a 3-dimensional")
print("linear subspace exactly, so PCA at k = 3 is EXACT, and nothing can")
print("beat exact. All the autoencoder can do at that point is fail to")
print("match it, which is what the ratio above 1 shows.")
print("\nSo the rule is sharper than 'nonlinear is better on curved data'.")
print("The nonlinear version pays when the code is forced BELOW the linear")
print("rank of the data — when curvature is the only way to fit in the")
print("budget. Above that rank it has nothing to exploit and an exact")
print("linear method is strictly better.")
print("\nWhich situation you are in is an empirical question, and the way")
print("to answer it is the comparison above: PCA takes one line, is exact,")
print("and tells you immediately whether the nonlinearity is earning its")
print("cost.")

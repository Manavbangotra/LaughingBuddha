# Extracted from: Chapter 61 — Autoencoders and Representation Learning
# Source: src/.../ch061-autoencoders.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The application that survives: train on normal data, flag high
reconstruction error — and the failure mode that catches people.
"""
import numpy as np

rng = np.random.default_rng(0)

D = 20


def make_normal(n, seed):
    """Normal data lies near a curved 3-D manifold in 20 dimensions."""
    rs = np.random.default_rng(seed)
    t = rs.uniform(0, 2 * np.pi, n)
    u = rs.uniform(-1, 1, n)
    w = rs.normal(0, 1, n)
    base = np.stack([np.cos(t) * (2 + u), np.sin(t) * (2 + u), w], axis=1)
    A = np.random.default_rng(31).normal(size=(3, D))
    return base @ A + rs.normal(0, 0.2, (n, D))


def make_anomalies(n, kind, seed):
    rs = np.random.default_rng(seed)
    if kind == "off-manifold":
        return rs.normal(0, 3.0, (n, D))            # ignores the structure
    if kind == "scaled":
        return make_normal(n, seed) * 2.5           # right shape, wrong scale
    if kind == "on-manifold":
        # ON the manifold but in a region the training data does not cover
        t = rs.uniform(0, 0.3, n)
        u = rs.uniform(-1, 1, n)
        w = rs.normal(0, 1, n)
        base = np.stack([np.cos(t) * (2 + u), np.sin(t) * (2 + u), w],
                        axis=1)
        A = np.random.default_rng(31).normal(size=(3, D))
        return base @ A + rs.normal(0, 0.2, (n, D))
    raise ValueError(kind)


class AE:
    def __init__(self, d, k, hidden=48, seed=0):
        rs = np.random.default_rng(seed)
        self.p = [rs.normal(0, np.sqrt(2 / d), (d, hidden)),
                  np.zeros(hidden),
                  rs.normal(0, np.sqrt(2 / hidden), (hidden, k)),
                  np.zeros(k),
                  rs.normal(0, np.sqrt(2 / k), (k, hidden)),
                  np.zeros(hidden),
                  rs.normal(0, np.sqrt(2 / hidden), (hidden, d)),
                  np.zeros(d)]

    def forward(self, X):
        W1, b1, W2, b2, W3, b3, W4, b4 = self.p
        self.a1 = np.tanh(X @ W1 + b1)
        self.z = self.a1 @ W2 + b2
        self.a3 = np.tanh(self.z @ W3 + b3)
        return self.a3 @ W4 + b4

    def grads(self, X):
        W1, b1, W2, b2, W3, b3, W4, b4 = self.p
        xr = self.forward(X)
        d4 = 2 * (xr - X) / len(X)
        g = [None] * 8
        g[6], g[7] = self.a3.T @ d4, d4.sum(axis=0)
        d3 = (d4 @ W4.T) * (1 - self.a3 ** 2)
        g[4], g[5] = self.z.T @ d3, d3.sum(axis=0)
        dz = d3 @ W3.T
        g[2], g[3] = self.a1.T @ dz, dz.sum(axis=0)
        d1 = (dz @ W2.T) * (1 - self.a1 ** 2)
        g[0], g[1] = X.T @ d1, d1.sum(axis=0)
        return g

    def errors(self, X):
        return ((self.forward(X) - X) ** 2).mean(axis=1)


def train(net, X, steps=6000, lr=3e-3, batch=128, noise=0.0, seed=0):
    m = [np.zeros_like(p) for p in net.p]
    v = [np.zeros_like(p) for p in net.p]
    rs = np.random.default_rng(seed + 4)
    for t in range(1, steps + 1):
        xb = X[rs.integers(0, len(X), batch)]
        # eq. 61.4: a denoising autoencoder reconstructs the CLEAN input
        inp = xb + rs.normal(0, noise, xb.shape) if noise else xb
        xr = net.forward(inp)
        d4 = 2 * (xr - xb) / len(xb)
        W1, b1, W2, b2, W3, b3, W4, b4 = net.p
        g = [None] * 8
        g[6], g[7] = net.a3.T @ d4, d4.sum(axis=0)
        d3 = (d4 @ W4.T) * (1 - net.a3 ** 2)
        g[4], g[5] = net.z.T @ d3, d3.sum(axis=0)
        dz = d3 @ W3.T
        g[2], g[3] = net.a1.T @ dz, dz.sum(axis=0)
        d1 = (dz @ W2.T) * (1 - net.a1 ** 2)
        g[0], g[1] = inp.T @ d1, d1.sum(axis=0)
        for i, (p, gg) in enumerate(zip(net.p, g)):
            m[i] = 0.9 * m[i] + 0.1 * gg
            v[i] = 0.999 * v[i] + 0.001 * gg * gg
            p -= lr * (m[i] / (1 - 0.9 ** t)) / (
                np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    return net


def auc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    npos, nneg = labels.sum(), (1 - labels).sum()
    return (ranks[labels == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


Xn_tr = make_normal(6000, 1)
Xn_cal = make_normal(3000, 2)          # held-out NORMAL, for the threshold
Xn_te = make_normal(3000, 3)

print("=" * 72)
print("anomaly detection by reconstruction error (section 5.5)")
print("=" * 72)
print("Trained ONLY on normal data. Three kinds of anomaly.\n")
net = train(AE(D, 3, seed=5), Xn_tr)
err_cal = net.errors(Xn_cal)
thresh = float(np.quantile(err_cal, 0.99))          # section 7.5
print(f"threshold at the 99th percentile of held-out NORMAL error: "
      f"{thresh:.5f}")
print(f"false-positive rate on fresh normal data: "
      f"{float((net.errors(Xn_te) > thresh).mean()):.4f}\n")
print(f"{'anomaly type':<18} {'mean error':>12} {'x normal':>10} "
      f"{'detection rate':>16} {'AUC':>8}")
base = float(err_cal.mean())
for kind in ("off-manifold", "scaled", "on-manifold"):
    Xa = make_anomalies(1500, kind, 9)
    ea = net.errors(Xa)
    labels = np.concatenate([np.zeros(len(Xn_te)), np.ones(len(ea))])
    scores = np.concatenate([net.errors(Xn_te), ea])
    print(f"{kind:<18} {ea.mean():>12.5f} {ea.mean() / base:>10.2f} "
          f"{float((ea > thresh).mean()):>16.4f} "
          f"{auc(scores, labels):>8.4f}")

print("\nThe first two are detected easily: they are far from the manifold")
print("the autoencoder learned, so it cannot reconstruct them and the error")
print("is large.")
print("\nThe third row is the failure mode, and it is the one that matters.")
print("Those points lie ON the learned manifold — they satisfy every")
print("structural property of the normal data — and they are in a region")
print("the training data does not cover. The autoencoder reconstructs them")
print("comfortably and the detector says nothing.")
print("\nThat is the honest limitation of reconstruction-based anomaly")
print("detection: it measures DISTANCE FROM THE MANIFOLD, not distance from")
print("the training distribution. Those are different quantities, and an")
print("anomaly that respects the structure while violating the density is")
print("invisible to it. A density-based method (Chapter 42) sees this case")
print("and misses others.")

# --- the bottleneck size ----------------------------------------------------
print("\n" + "=" * 72)
print("the bottleneck size is the whole hyperparameter")
print("=" * 72)
print("The true manifold is 3-dimensional. k = 20 means the code is as WIDE")
print("as the input, so the identity map is available.\n")
print(f"{'code size k':>12} {'normal MSE':>12} {'anomaly MSE':>13} "
      f"{'off-manifold AUC':>18} {'on-manifold AUC':>17}")
Xoff = make_anomalies(1500, "off-manifold", 9)
Xon = make_anomalies(1500, "on-manifold", 9)
for k in (1, 2, 3, 6, 12, 20):
    nk = train(AE(D, k, seed=5), Xn_tr)
    en = nk.errors(Xn_te)
    eo = nk.errors(Xoff)
    row = []
    for Xa in (Xoff, Xon):
        sc = np.concatenate([en, nk.errors(Xa)])
        lb = np.concatenate([np.zeros(len(en)), np.ones(1500)])
        row.append(auc(sc, lb))
    print(f"{k:>12} {float(en.mean()):>12.5f} {float(eo.mean()):>13.4f} "
          f"{row[0]:>18.4f} {row[1]:>17.4f}")

print("\nBelow the true dimension the autoencoder cannot reconstruct even")
print("NORMAL data, and the normal MSE column shows it. Above it, normal")
print("MSE keeps improving as expected.")
print("\nThe result worth pausing on is the off-manifold column, because it")
print("is not what the standard warning predicts. 'Too wide a bottleneck")
print("learns the identity and reconstructs everything' — at k = 20 the")
print("code IS as wide as the input, the identity is available, and the")
print("network did not learn it. The anomaly MSE column is still hundreds")
print("of times the normal one and detection is perfect.")
print("\nThe reason is Chapter 58's implicit regularisation. Gradient")
print("descent from a small initialisation, on data that only ever lies")
print("near a 3-dimensional manifold, has no reason to learn the identity —")
print("nothing in the training signal ever asks what to do with an")
print("off-manifold point. The decoder's learned range stays close to the")
print("manifold whatever the code width allows.")
print("\nSo the bottleneck is a much weaker lever here than its reputation")
print("suggests, and the practical reading is: check whether it matters on")
print("YOUR data rather than assuming a narrow code is doing the work. What")
print("it clearly does control is how well normal data is fitted, and that")
print("sets the noise floor the anomaly signal has to clear.")
print("\nThe on-manifold column is unmoved at every width, which is the")
print("point of the previous table restated: no bottleneck size fixes an")
print("anomaly that respects the manifold, because the quantity being")
print("measured is the wrong one.")

# --- denoising --------------------------------------------------------------
print("\n" + "=" * 72)
print("the denoising variant removes the need for a narrow code (eq. 61.4)")
print("=" * 72)
print("A code as WIDE as the input, so the identity is available. Only the")
print("corruption stops the network from learning it.\n")
print("A code as WIDE as the input, so the identity is available. Does")
print("corruption change what the network learns?\n")
print(f"{'code size k':>12} {'train noise':>12} {'clean-input MSE':>17} "
      f"{'dist. to identity':>19} {'off-manifold AUC':>18}")
Xprobe = np.random.default_rng(55).normal(0, 3.0, (500, D))
for k, noise in ((D, 0.0), (D, 0.3), (D, 0.8), (3, 0.0)):
    nk = train(AE(D, k, seed=5), Xn_tr, noise=noise)
    en = nk.errors(Xn_te)
    Xa = make_anomalies(1500, "off-manifold", 9)
    sc = np.concatenate([en, nk.errors(Xa)])
    lb = np.concatenate([np.zeros(len(en)), np.ones(1500)])
    # how close is the map to the identity, probed OFF the manifold?
    idty = float(np.mean((nk.forward(Xprobe) - Xprobe) ** 2)
                 / np.mean(Xprobe ** 2))
    print(f"{k:>12} {noise:>12.1f} {float(en.mean()):>17.5f} "
          f"{idty:>19.4f} {auc(sc, lb):>18.4f}")

print("\nThe 'dist. to identity' column probes the learned map with points")
print("far off the manifold and asks how close it is to the identity there.")
print("A value near 0 would mean the network learned to copy its input; a")
print("value near 1 means it discards off-manifold input entirely.")
print("\nAt k = 20 with no corruption the identity is available and the")
print("network did not take it — Chapter 58's implicit regularisation")
print("again. So on this data the corruption is not NEEDED, and the table")
print("shows it changing the numbers only modestly.")
print("\nThat is the honest result and it is narrower than the usual")
print("claim. The denoising variant's importance is not that it is required")
print("here; it is that it demonstrates the constraint does not have to be")
print("DIMENSIONAL. It can be a corruption process, applied to a code as")
print("wide as you like.")
print("\nThat generalisation is what mattered. Masked language modelling is")
print("eq. 61.4 with the corruption being 'delete some tokens' and no")
print("bottleneck anywhere, and it is what made self-supervised pretraining")
print("work at scale. The idea outlived the architecture that introduced")
print("it, which is the note this chapter and this part end on.")

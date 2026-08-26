# -*- coding: utf-8 -*-
# Extracted from: Chapter 57 — Normalization: Batch, Layer, and RMSNorm
# Source: src/.../ch057-normalization.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Does normalisation actually help, and which one? Measured on a deep
network, with the internal-covariate-shift test of Santurkar et al.
"""
import numpy as np

rng = np.random.default_rng(5)

D, C = 24, 5
_rs = np.random.default_rng(88)
A1, A2 = _rs.normal(size=(D, 16)), _rs.normal(size=(16, C))


def make_data(n, seed):
    rs = np.random.default_rng(seed)
    X = rs.normal(size=(n, D))
    logits = np.tanh(X @ A1) @ A2 * 1.6
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y = (p.cumsum(axis=1) < rs.random((n, 1))).sum(axis=1).clip(0, C - 1)
    return X, y


Xtr, ytr = make_data(40000, 1)
Xte, yte = make_data(10000, 2)
_p = np.exp(np.tanh(Xte @ A1) @ A2 * 1.6)
_p /= _p.sum(axis=1, keepdims=True)
BAYES = float(-np.log(_p[np.arange(len(yte)), yte]).mean())


class Net:
    """Deep MLP with a configurable normalisation, hand-written backward."""

    def __init__(self, depth, width, norm="none", scale=1.0, seed=0,
                 inject_shift=0.0):
        rs = np.random.default_rng(seed)
        self.depth, self.norm, self.inject = depth, norm, inject_shift
        self.W = [rs.normal(0, scale * np.sqrt(2.0 / D), (D, width))]
        for _ in range(depth - 1):
            self.W.append(rs.normal(0, scale * np.sqrt(2.0 / width),
                                    (width, width)))
        self.g = [np.ones(width) for _ in range(depth)]
        self.b = [np.zeros(width) for _ in range(depth)]
        self.Wout = rs.normal(0, np.sqrt(2.0 / width), (width, C))
        self.bout = np.zeros(C)
        self.shift_rs = np.random.default_rng(seed + 999)

    def _fwd_norm(self, z, l):
        if self.norm == "none":
            return z, None
        if self.norm == "batch":
            mu, var = z.mean(axis=0), z.var(axis=0)
            xhat = (z - mu) / np.sqrt(var + 1e-5)
            axis = 0
        else:                                   # layer
            mu = z.mean(axis=1, keepdims=True)
            var = z.var(axis=1, keepdims=True)
            xhat = (z - mu) / np.sqrt(var + 1e-5)
            axis = 1
        out = self.g[l] * xhat + self.b[l]
        if self.inject:
            # Santurkar et al.'s test: deliberately RESTORE covariate shift
            # by adding noise whose MEAN and VARIANCE change every step,
            # AFTER the normalisation has done its work. The magnitude is a
            # parameter because it decides whether this restores covariate
            # shift or simply destroys the signal.
            a = self.inject
            mu_t = self.shift_rs.normal(0, a, out.shape[1])
            sd_t = np.abs(self.shift_rs.normal(a, a / 2, out.shape[1]))
            out = out + mu_t + sd_t * self.shift_rs.normal(0, 1, out.shape)
        return out, (xhat, np.sqrt(var + 1e-5), axis)

    def forward(self, X):
        self.cache = []
        h = X
        for l in range(self.depth):
            z = h @ self.W[l]
            n, ncache = self._fwd_norm(z, l)
            a = np.maximum(0.0, n)
            self.cache.append((h, z, n, ncache))
            h = a
        self.hL = h
        return h @ self.Wout + self.bout

    def loss_and_grads(self, X, y):
        logits = self.forward(X)
        m = logits.max(axis=1, keepdims=True)
        e = np.exp(logits - m)
        loss = float(np.mean(m[:, 0] + np.log(e.sum(axis=1))
                             - logits[np.arange(len(X)), y]))
        d = e / e.sum(axis=1, keepdims=True)
        d[np.arange(len(X)), y] -= 1.0
        d /= len(X)
        gWout, gbout = self.hL.T @ d, d.sum(axis=0)
        dh = d @ self.Wout.T
        gW = [None] * self.depth
        gg = [None] * self.depth
        gb = [None] * self.depth
        for l in reversed(range(self.depth)):
            h_in, z, n, ncache = self.cache[l]
            dn = dh * (n > 0)
            if ncache is None:
                dz = dn
                gg[l] = np.zeros_like(self.g[l])
                gb[l] = np.zeros_like(self.b[l])
            else:
                xhat, s, axis = ncache
                gg[l] = (dn * xhat).sum(axis=0)
                gb[l] = dn.sum(axis=0)
                dxhat = dn * self.g[l]
                N = xhat.shape[axis]
                if axis == 0:
                    dz = (N * dxhat - dxhat.sum(axis=0)
                          - xhat * (dxhat * xhat).sum(axis=0)) / (N * s)
                else:
                    dz = (N * dxhat - dxhat.sum(axis=1, keepdims=True)
                          - xhat * (dxhat * xhat).sum(axis=1, keepdims=True)
                          ) / (N * s)
            gW[l] = h_in.T @ dz
            dh = dz @ self.W[l].T
        return loss, gW, gg, gb, gWout, gbout


def train(depth, width, norm, scale=1.0, steps=3000, lr=2e-3, batch=128,
          seed=0, inject_shift=0.0):
    net = Net(depth, width, norm, scale, seed, inject_shift)
    params = net.W + net.g + net.b + [net.Wout, net.bout]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    rs = np.random.default_rng(seed + 30)
    for t in range(1, steps + 1):
        idx = rs.integers(0, len(Xtr), batch)
        _, gW, gg, gb, gWout, gbout = net.loss_and_grads(Xtr[idx], ytr[idx])
        grads = gW + gg + gb + [gWout, gbout]
        for i, (pp, g) in enumerate(zip(params, grads)):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            pp -= lr * (m[i] / (1 - 0.9 ** t)) / (
                np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    net.inject = 0.0                        # evaluate without the injection
    te, _, _, _, _, _ = net.loss_and_grads(Xte, yte)
    acc = float((net.forward(Xte).argmax(axis=1) == yte).mean())
    return te - BAYES, acc


print("=" * 72)
print("does normalisation help, and at what depth?")
print("=" * 72)
print(f"Bayes-optimal test cross-entropy: {BAYES:.4f}")
print("Excess test loss above that floor; lower is better.\n")
print(f"{'normalisation':<16} " + " ".join(f"{f'depth {d}':>12}"
                                           for d in (2, 6, 16)))
for norm in ("none", "batch", "layer"):
    row = []
    for depth in (2, 6, 16):
        ex, acc = train(depth, 64, norm)
        row.append("diverged" if not np.isfinite(ex) else f"{ex:.4f}")
    print(f"{norm:<16} " + " ".join(f"{v:>12}" for v in row))

print("\nBoth normalisations give a consistent improvement at every depth,")
print("and — read the gap against the unnormalised row — it does NOT grow")
print("with depth over this range. That is not what the usual account")
print("predicts, and the reason is in the previous chapter: these networks")
print("are He-initialised, and Chapter 56 measured He initialisation")
print("keeping the variance profile flat to fifty layers on its own. There")
print("is no signal-propagation problem here for normalisation to solve.")
print("\nSo this table is NOT where normalisation earns its place. The next")
print("one is.")

# --- robustness to a bad initialisation scale -------------------------------
print("\n" + "=" * 72)
print("normalisation buys robustness to the initialisation scale (6.2)")
print("=" * 72)
print("Depth 16, weights multiplied by a factor the scheme did not intend.\n")
print(f"{'init scale':>11} " + " ".join(f"{n:>13}" for n in
                                        ("none", "batch", "layer")))
cols = {n: [] for n in ("none", "batch", "layer")}
for scale in (0.25, 0.5, 1.0, 2.0, 4.0):
    row = []
    for norm in ("none", "batch", "layer"):
        ex, acc = train(16, 64, norm, scale=scale)
        cols[norm].append(ex)
        row.append("diverged" if not np.isfinite(ex) or ex > 5
                   else f"{ex:.4f}")
    print(f"{scale:>11.2f} " + " ".join(f"{v:>13}" for v in row))

print(f"\n{'spread':>11} " + " ".join(
    f"{('inf' if not all(np.isfinite(v) and v < 5 for v in cols[n]) else f'{max(cols[n]) - min(cols[n]):.4f}'):>13}"
    for n in ("none", "batch", "layer")))

print("\nRead the SPREAD row, not the individual values. The question is")
print("how much the result depends on a scale factor the network should not")
print("care about at all.")
print("\nThe unnormalised network's spread is unbounded — it diverges at")
print("the top of the range. Batch normalisation's is much smaller, which")
print("is eq. 57.10 doing exactly what it says: the output does not depend")
print("on |W|, so the scale cannot matter.")
print("\nLayer normalisation sits between them, and the reason is worth")
print("noticing. It normalises across FEATURES within one example, so it")
print("controls the scale of each layer's output but not the relative")
print("scale of the weight matrix against the input — and at the smallest")
print("init scale it does noticeably worse. eq. 57.10's invariance is a")
print("property of what the layer is normalising over, and the two")
print("normalisations are normalising over different things.")
print("\nThe practical form of what normalisation buys is that spread row:")
print("it removes a hyperparameter you would otherwise have to get right,")
print("and it is more useful than a small improvement in the best")
print("achievable loss.")

# --- Santurkar et al.'s test ------------------------------------------------
print("\n" + "=" * 72)
print("the internal covariate shift test (section 6.3)")
print("=" * 72)
print("Santurkar et al.'s experiment: inject noise with a RANDOM, time-")
print("varying mean AFTER each normalisation layer. This deliberately")
print("restores — and worsens — internal covariate shift while keeping the")
print("normalisation. If Ioffe and Szegedy's explanation were right, this")
print("should destroy the benefit.\n")
none_a, _ = train(16, 64, "none")
fmt = lambda v: "diverged" if not np.isfinite(v) or v > 5 else f"{v:.4f}"
print(f"unnormalised baseline: {fmt(none_a)}\n")
print(f"{'injection scale':>16} " + " ".join(f"{n:>14}" for n in
                                             ("batch", "layer")))
for a_ in (0.0, 0.05, 0.15, 0.5, 1.0):
    row = []
    for norm in ("batch", "layer"):
        ex, _ = train(16, 64, norm, inject_shift=a_)
        row.append(fmt(ex))
    print(f"{a_:>16.2f} " + " ".join(f"{v:>14}" for v in row))

print("\nRead the batch-norm column against the unnormalised baseline.")
print("\nAt injection 0.05 and 0.15 the distribution downstream of every")
print("normalisation layer is being shifted and rescaled by a DIFFERENT")
print("random amount at every single step — internal covariate shift,")
print("deliberately restored and worse than anything the network would")
print("produce on its own. Batch normalisation's result is unchanged, and")
print("still comfortably better than the unnormalised baseline.")
print("\nThat is Santurkar et al.'s finding, reproduced. If normalisation")
print("worked by removing covariate shift, restoring covariate shift should")
print("have given back the unnormalised result. It did not.")
print("\nThe two largest injections do degrade both, and that is a")
print("different experiment: at that magnitude the added noise is")
print("comparable to the signal itself, so the damage is the noise")
print("destroying the representation rather than the covariate shift")
print("mattering. Distinguishing those two regimes is why the sweep is")
print("here rather than a single injection level — a one-row version of")
print("this experiment can accidentally test the wrong thing.")
print("\nNote that layer normalisation degrades earlier than batch")
print("normalisation does. It normalises within one example, so an")
print("injected per-feature offset is not averaged over anything.")
print("\nThis remains a small-scale reproduction. What matters is that the")
print("explanation was testable, someone tested it, and the field's")
print("standard account of its most-cited normalisation technique did not")
print("survive. The technique is ESTABLISHED. The explanation is not.")
print("Those are different claims and it is worth keeping them apart.")

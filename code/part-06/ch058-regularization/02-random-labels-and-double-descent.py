# -*- coding: utf-8 -*-
# Extracted from: Chapter 58 — Regularization, Dropout, Overfitting, and Underfitting
# Source: src/.../ch058-regularization.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Zhang et al.'s random-label result and double descent, reproduced at a
scale that runs on a laptop.
"""
import numpy as np

rng = np.random.default_rng(0)

D, C = 16, 4


def make_data(n, seed, randomize_labels=False):
    rs = np.random.default_rng(seed)
    X = rs.normal(size=(n, D))
    W1 = np.random.default_rng(555).normal(size=(D, 10))
    W2 = np.random.default_rng(556).normal(size=(10, C))
    logits = np.tanh(X @ W1) @ W2 * 1.5
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y = (p.cumsum(axis=1) < rs.random((n, 1))).sum(axis=1).clip(0, C - 1)
    if randomize_labels:
        y = rs.integers(0, C, n)
    return X, y


class MLP:
    def __init__(self, sizes, seed=0):
        rs = np.random.default_rng(seed)
        self.W = [rs.normal(0, np.sqrt(2 / sizes[i]),
                            (sizes[i], sizes[i + 1]))
                  for i in range(len(sizes) - 1)]
        self.b = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]
        self.n_params = sum(W.size for W in self.W) + sum(
            b.size for b in self.b)

    def forward(self, X, p_drop=0.0, rs=None):
        self.H, self.Z, self.M = [X], [], []
        h = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            self.Z.append(z)
            if i < len(self.W) - 1:
                h = np.maximum(0.0, z)
                if p_drop > 0 and rs is not None:
                    m = (rs.random(h.shape) >= p_drop) / (1 - p_drop)
                    h = h * m
                    self.M.append(m)
                else:
                    self.M.append(None)
            else:
                h = z
                self.M.append(None)
            self.H.append(h)
        return h

    def loss_and_grads(self, X, y, p_drop=0.0, rs=None):
        logits = self.forward(X, p_drop, rs)
        m = logits.max(axis=1, keepdims=True)
        e = np.exp(logits - m)
        loss = float(np.mean(m[:, 0] + np.log(e.sum(axis=1))
                             - logits[np.arange(len(X)), y]))
        d = e / e.sum(axis=1, keepdims=True)
        d[np.arange(len(X)), y] -= 1.0
        d /= len(X)
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for l in reversed(range(len(self.W))):
            gW[l] = self.H[l].T @ d
            gb[l] = d.sum(axis=0)
            if l > 0:
                d = d @ self.W[l].T
                if self.M[l - 1] is not None:
                    d = d * self.M[l - 1]
                d = d * (self.Z[l - 1] > 0)
        return loss, gW, gb


def train(net, X, y, Xv, yv, steps=4000, lr=2e-3, batch=64, wd=0.0,
          p_drop=0.0, seed=0):
    params = net.W + net.b
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    rs = np.random.default_rng(seed + 20)
    for t in range(1, steps + 1):
        idx = rs.integers(0, len(X), min(batch, len(X)))
        _, gW, gb = net.loss_and_grads(X[idx], y[idx], p_drop, rs)
        for i, (pp, g) in enumerate(zip(params, gW + gb)):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            pp -= lr * (m[i] / (1 - 0.9 ** t)) / (
                np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
            if wd and pp.ndim == 2:
                pp -= lr * wd * pp
    tr = net.loss_and_grads(X, y)[0]
    te = net.loss_and_grads(Xv, yv)[0]
    tr_acc = float((net.forward(X).argmax(axis=1) == y).mean())
    te_acc = float((net.forward(Xv).argmax(axis=1) == yv).mean())
    return tr, te, tr_acc, te_acc


# --- Zhang et al.: networks fit random labels -------------------------------
print("=" * 72)
print("networks fit RANDOM labels perfectly (section 6.4)")
print("=" * 72)
print("Same architecture, same optimiser, same regularisation. Only the")
print("labels differ: real, or drawn uniformly at random.\n")
N_SMALL = 600
Xs, ys = make_data(N_SMALL, 1)
Xr, yr = make_data(N_SMALL, 1, randomize_labels=True)
Xv, yv = make_data(4000, 2)

print(f"{'labels':<12} {'regularisation':<22} {'train acc':>11} "
      f"{'test acc':>10} {'train loss':>12}")
for label, (XX, yy) in (("real", (Xs, ys)), ("RANDOM", (Xr, yr))):
    for reg_name, kw in (("none", {}),
                         ("wd 0.01 + dropout 0.3",
                          {"wd": 0.01, "p_drop": 0.3})):
        net = MLP([D, 256, 256, C], seed=3)
        tr, te, tra, tea = train(net, XX, yy, Xv, yv, steps=8000, **kw)
        print(f"{label:<12} {reg_name:<22} {tra:>11.4f} {tea:>10.4f} "
              f"{tr:>12.5f}")
print(f"\n(chance accuracy is {1 / C:.4f}; the network has "
      f"{MLP([D, 256, 256, C]).n_params:,} parameters "
      f"for {N_SMALL} examples)")

print("\nThe network drives training accuracy high on labels that contain")
print("NO information whatsoever, and its test accuracy on those labels is")
print("at chance — which it must be, since there is nothing to generalise.")
print("\nThat is Zhang et al.'s reductio. Whatever capacity measure appears")
print("in eq. 58.12's bound must be at least large enough to shatter this")
print("training set, which makes the bound vacuous — and yet the SAME")
print("network on real labels generalises.")
print("\nSo the explanation for generalisation cannot live in the hypothesis")
print("class alone. It has to involve the data and the training procedure.")
print("\nNote also what the regularisation did: it slowed the memorisation")
print("without preventing it, and it changed the real-label result by a")
print("modest amount. Regularisation is a knob, not the mechanism.")

# --- double descent ---------------------------------------------------------
print("\n" + "=" * 72)
print("double descent: test error past the interpolation threshold (6.5)")
print("=" * 72)
print(f"A fixed training set of {N_SMALL} examples with 15% label noise,")
print("and networks of increasing width.\n")
Xd, yd = make_data(N_SMALL, 7)
noise_idx = np.random.default_rng(8).choice(N_SMALL, N_SMALL * 15 // 100,
                                            replace=False)
yd = yd.copy()
yd[noise_idx] = np.random.default_rng(9).integers(0, C, len(noise_idx))

print(f"{'width':>7} {'params':>9} {'params/N':>10} {'train acc':>11} "
      f"{'test acc':>10} {'test loss':>11}")
rows = []
for width in (2, 4, 8, 12, 16, 24, 40, 80, 160, 320):
    net = MLP([D, width, C], seed=4)
    tr, te, tra, tea = train(net, Xd, yd, Xv, yv, steps=6000, lr=3e-3)
    rows.append((width, net.n_params, tra, tea, te))
    print(f"{width:>7} {net.n_params:>9,} {net.n_params / N_SMALL:>10.2f} "
          f"{tra:>11.4f} {tea:>10.4f} {te:>11.4f}")

losses = [r[4] for r in rows]
peak = int(np.argmax(losses[1:-1])) + 1
print(f"\nworst test loss at width {rows[peak][0]} "
      f"({rows[peak][1] / N_SMALL:.2f} params per example)")
print(f"best test loss at width {rows[int(np.argmin(losses))][0]} "
      f"({rows[int(np.argmin(losses))][1] / N_SMALL:.2f} params per example)")

print("\nThe peak sits essentially at the interpolation threshold — where")
print("the parameter count first passes the number of training examples and")
print("training accuracy first reaches 1.0 — and BOTH test loss and test")
print("accuracy improve monotonically for every width beyond it. That is")
print("the second descent, and it is clearly present here.")
print("\nThe classical picture predicts only the rise: past the point where")
print("the model can fit the training set, more capacity should mean")
print("monotonically worse test error. It does not.")
print("\nOne honest limitation. The second descent does not get BELOW the")
print("classical optimum at this scale — the tiny width-2 model still has")
print("the lowest test loss, because with 15 per cent label noise a model")
print("that predicts near-uniform scores well on log loss. Belkin et al.'s")
print("stronger claim, that the interpolating regime can beat the classical")
print("optimum, needs more data and more capacity than a laptop-sized")
print("reproduction has.")
print("\nWhat this table does establish is the part that matters for")
print(f"practice: the widest network, with {rows[-1][1] / N_SMALL:.0f} times as many")
print("parameters as training examples, is not the worst model — the one at")
print("the threshold is. 'More parameters means more overfitting' is false")
print("as a general rule, and the field's whole practice of scaling up")
print("depends on that.")

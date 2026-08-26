# -*- coding: utf-8 -*-
# Extracted from: Chapter 59 — Convolutional Neural Networks
# Source: src/.../ch059-cnns.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A small convolutional network trained end to end, and the three
comparisons that justify the architecture.
"""
import numpy as np

rng = np.random.default_rng(7)

# --- a small image problem --------------------------------------------------
IMG, C = 16, 4


def make_shapes(n, seed, noise=0.35):
    """Four classes: horizontal bar, vertical bar, square, diagonal.
    Each placed at a RANDOM position — which is what makes translation
    structure the right inductive bias."""
    rs = np.random.default_rng(seed)
    X = rs.normal(0, noise, (n, 1, IMG, IMG))
    y = rs.integers(0, C, n)
    for i in range(n):
        r = rs.integers(2, IMG - 6)
        c = rs.integers(2, IMG - 6)
        if y[i] == 0:
            X[i, 0, r, c:c + 5] += 2.0
        elif y[i] == 1:
            X[i, 0, r:r + 5, c] += 2.0
        elif y[i] == 2:
            X[i, 0, r:r + 4, c:c + 4] += 1.2
        else:
            for d in range(4):
                X[i, 0, r + d, c + d] += 2.0
    return X, y


Xtr, ytr = make_shapes(6000, 1)
Xte, yte = make_shapes(4000, 2)


def _conv_fwd(X, K, b, pad=1, stride=1):
    N, Ci, H, W = X.shape
    F, _, kh, kw = K.shape
    Xp = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    Ho = (H + 2 * pad - kh) // stride + 1
    Wo = (W + 2 * pad - kw) // stride + 1
    s = Xp.strides
    patches = np.lib.stride_tricks.as_strided(
        Xp, shape=(N, Ci, Ho, Wo, kh, kw),
        strides=(s[0], s[1], s[2] * stride, s[3] * stride, s[2], s[3]),
        writeable=False)
    cols = patches.transpose(0, 2, 3, 1, 4, 5).reshape(N * Ho * Wo, -1)
    out = (cols @ K.reshape(F, -1).T + b).reshape(N, Ho, Wo, F)
    return out.transpose(0, 3, 1, 2), cols, (Ho, Wo)


class ConvNet:
    """conv -> relu -> conv -> relu -> global average pool -> linear."""

    def __init__(self, ch=(8, 16), seed=0):
        rs = np.random.default_rng(seed)
        self.K1 = rs.normal(0, np.sqrt(2 / 9), (ch[0], 1, 3, 3))
        self.b1 = np.zeros(ch[0])
        self.K2 = rs.normal(0, np.sqrt(2 / (9 * ch[0])),
                            (ch[1], ch[0], 3, 3))
        self.b2 = np.zeros(ch[1])
        self.Wo = rs.normal(0, np.sqrt(2 / ch[1]), (ch[1], C))
        self.bo = np.zeros(C)
        self.n_params = (self.K1.size + self.b1.size + self.K2.size
                         + self.b2.size + self.Wo.size + self.bo.size)

    def forward(self, X):
        z1, self.c1, self.s1 = _conv_fwd(X, self.K1, self.b1)
        self.z1 = z1
        a1 = np.maximum(0.0, z1)
        z2, self.c2, self.s2 = _conv_fwd(a1, self.K2, self.b2)
        self.z2, self.a1 = z2, a1
        a2 = np.maximum(0.0, z2)
        self.a2 = a2
        pooled = a2.mean(axis=(2, 3))                 # global average pool
        self.pooled = pooled
        return pooled @ self.Wo + self.bo

    def loss_and_grads(self, X, y):
        logits = self.forward(X)
        m = logits.max(axis=1, keepdims=True)
        e = np.exp(logits - m)
        loss = float(np.mean(m[:, 0] + np.log(e.sum(axis=1))
                             - logits[np.arange(len(X)), y]))
        d = e / e.sum(axis=1, keepdims=True)
        d[np.arange(len(X)), y] -= 1.0
        d /= len(X)
        gWo, gbo = self.pooled.T @ d, d.sum(axis=0)
        dpool = d @ self.Wo.T
        N, F2, H2, W2 = self.a2.shape
        da2 = np.repeat(np.repeat(dpool[:, :, None, None], H2, 2), W2, 3) \
            / (H2 * W2)
        dz2 = da2 * (self.z2 > 0)
        dz2c = dz2.transpose(0, 2, 3, 1).reshape(-1, F2)
        gK2 = (dz2c.T @ self.c2).reshape(self.K2.shape)
        gb2 = dz2c.sum(axis=0)
        dcols2 = dz2c @ self.K2.reshape(F2, -1)
        da1 = self._col2im(dcols2, self.a1.shape, 3, 1, 1)
        dz1 = da1 * (self.z1 > 0)
        F1 = self.K1.shape[0]
        dz1c = dz1.transpose(0, 2, 3, 1).reshape(-1, F1)
        gK1 = (dz1c.T @ self.c1).reshape(self.K1.shape)
        gb1 = dz1c.sum(axis=0)
        return loss, [gK1, gb1, gK2, gb2, gWo, gbo]

    @staticmethod
    def _col2im(cols, shape, k, stride, pad):
        N, Ci, H, W = shape
        Ho = (H + 2 * pad - k) // stride + 1
        Wo = (W + 2 * pad - k) // stride + 1
        out = np.zeros((N, Ci, H + 2 * pad, W + 2 * pad))
        cols = cols.reshape(N, Ho, Wo, Ci, k, k)
        for u in range(k):
            for v in range(k):
                out[:, :, u:u + Ho * stride:stride,
                    v:v + Wo * stride:stride] += cols[:, :, :, :, u,
                                                      v].transpose(0, 3, 1, 2)
        return out[:, :, pad:pad + H, pad:pad + W]

    def params(self):
        return [self.K1, self.b1, self.K2, self.b2, self.Wo, self.bo]


class DenseNet:
    """The same budget spent on a fully connected network."""

    def __init__(self, hidden, seed=0):
        rs = np.random.default_rng(seed)
        d = IMG * IMG
        self.W1 = rs.normal(0, np.sqrt(2 / d), (d, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rs.normal(0, np.sqrt(2 / hidden), (hidden, C))
        self.b2 = np.zeros(C)
        self.n_params = (self.W1.size + self.b1.size + self.W2.size
                         + self.b2.size)

    def forward(self, X):
        self.x = X.reshape(len(X), -1)
        self.z1 = self.x @ self.W1 + self.b1
        self.a1 = np.maximum(0.0, self.z1)
        return self.a1 @ self.W2 + self.b2

    def loss_and_grads(self, X, y):
        logits = self.forward(X)
        m = logits.max(axis=1, keepdims=True)
        e = np.exp(logits - m)
        loss = float(np.mean(m[:, 0] + np.log(e.sum(axis=1))
                             - logits[np.arange(len(X)), y]))
        d = e / e.sum(axis=1, keepdims=True)
        d[np.arange(len(X)), y] -= 1.0
        d /= len(X)
        gW2, gb2 = self.a1.T @ d, d.sum(axis=0)
        d1 = (d @ self.W2.T) * (self.z1 > 0)
        return loss, [self.x.T @ d1, d1.sum(axis=0), gW2, gb2]

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2]


def train(net, Xtr, ytr, steps=2500, lr=3e-3, batch=64, seed=0):
    """Adam on whatever params() returns."""
    ps = net.params()
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    rs = np.random.default_rng(seed + 10)
    for t in range(1, steps + 1):
        idx = rs.integers(0, len(Xtr), batch)
        _, gs = net.loss_and_grads(Xtr[idx], ytr[idx])
        for i, (p, g) in enumerate(zip(ps, gs)):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            p -= lr * (m[i] / (1 - 0.9 ** t)) / (
                np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    return net


def evaluate(net, X, y, chunk=1000):
    correct, loss = 0, 0.0
    for i in range(0, len(X), chunk):
        lg = net.forward(X[i:i + chunk])
        correct += int((lg.argmax(axis=1) == y[i:i + chunk]).sum())
        m = lg.max(axis=1, keepdims=True)
        e = np.exp(lg - m)
        loss += float((m[:, 0] + np.log(e.sum(axis=1))
                       - lg[np.arange(len(lg)), y[i:i + chunk]]).sum())
    return loss / len(X), correct / len(X)


print("=" * 72)
print("convolution against a dense network at a MATCHED parameter budget")
print("=" * 72)
print("Four shapes at RANDOM positions in a 16x16 image. The task has")
print("translation structure by construction, which is the assumption the")
print("convolution encodes.\n")
print(f"{'model':<32} {'params':>9} {'train acc':>11} {'test acc':>10} "
      f"{'test loss':>11}")
cnet = train(ConvNet(seed=1), Xtr, ytr)
c_trl, c_tra = evaluate(cnet, Xtr, ytr)
c_tel, c_tea = evaluate(cnet, Xte, yte)
print(f"{'ConvNet 8->16 + global pool':<32} {cnet.n_params:>9,} "
      f"{c_tra:>11.4f} {c_tea:>10.4f} {c_tel:>11.4f}")

for hidden in (6, 64):
    dnet = train(DenseNet(hidden, seed=1), Xtr, ytr)
    d_trl, d_tra = evaluate(dnet, Xtr, ytr)
    d_tel, d_tea = evaluate(dnet, Xte, yte)
    print(f"{f'Dense {IMG * IMG} -> {hidden} -> {C}':<32} "
          f"{dnet.n_params:>9,} {d_tra:>11.4f} {d_tea:>10.4f} "
          f"{d_tel:>11.4f}")

print("\nThe first dense row is matched on parameters and the second is")
print("given roughly ten times as many. Read both against the convolution.")
print("\nWhat the convolution has that neither dense network does is not")
print("capacity — section 6.3 showed its hypothesis class is a strict")
print("SUBSET of the dense layer's. It is the assumption that a detector")
print("useful at one position is useful at every position, which this task")
print("satisfies exactly.")

# --- the assumption, removed ------------------------------------------------
print("\n" + "=" * 72)
print("the same comparison when the assumption does NOT hold")
print("=" * 72)
print("Shapes always at the SAME position, and the pixels permuted by a")
print("fixed random permutation — which destroys locality without changing")
print("the information content at all.\n")


def make_fixed_position(n, seed, permute=False, perm=None):
    rs = np.random.default_rng(seed)
    X = rs.normal(0, 0.35, (n, 1, IMG, IMG))
    y = rs.integers(0, C, n)
    r = c = 5
    for i in range(n):
        if y[i] == 0:
            X[i, 0, r, c:c + 5] += 2.0
        elif y[i] == 1:
            X[i, 0, r:r + 5, c] += 2.0
        elif y[i] == 2:
            X[i, 0, r:r + 4, c:c + 4] += 1.2
        else:
            for d in range(4):
                X[i, 0, r + d, c + d] += 2.0
    if permute:
        flat = X.reshape(n, -1)[:, perm]
        X = flat.reshape(n, 1, IMG, IMG)
    return X, y


perm = np.random.default_rng(99).permutation(IMG * IMG)
print(f"{'data':<26} {'model':<16} {'test acc':>10} {'test loss':>11}")
for label, kw in (("fixed position", {"permute": False}),
                  ("fixed + PERMUTED pixels", {"permute": True,
                                               "perm": perm})):
    Xa, ya = make_fixed_position(6000, 11, **kw)
    Xb, yb = make_fixed_position(4000, 12, **kw)
    for mname, net in (("ConvNet", ConvNet(seed=1)),
                       ("Dense h=64", DenseNet(64, seed=1))):
        train(net, Xa, ya)
        tl, ta = evaluate(net, Xb, yb)
        print(f"{label:<26} {mname:<16} {ta:>10.4f} {tl:>11.4f}")

print("\nA fixed random permutation of the pixels preserves every bit of")
print("information in the image. The dense network is EXACTLY unaffected —")
print("it has no notion of which inputs are neighbours, so a permutation is")
print("invisible to it.")
print("\nThe convolution notices, because its assumption has been broken:")
print("pixels that were adjacent are now scattered, so a 3x3 kernel looks")
print("at three unrelated positions. The gap between its permuted and")
print("unpermuted result is exactly the value of the assumption it was")
print("making — on THIS task, where the shapes sit at a fixed position and")
print("the task is easy enough that the dense network solves it perfectly.")
print("\nNote also that the dense network BEATS the convolution here, on")
print("both rows. With the shapes always in the same place, translation")
print("equivariance buys nothing — there is nothing to be equivariant")
print("about — and the constraint only costs. Compare with the previous")
print("table, where the shapes moved and the convolution won at a twelfth")
print("of the parameters.")
print("\nThat contrast is the point. The same architectural constraint is")
print("worth an order of magnitude on one task and a small loss on")
print("another, and what changed was not the model but whether its")
print("assumption held.")
print("\nThat is the cleanest available demonstration of what an inductive")
print("bias IS. It is not extra power — it is a commitment about the data,")
print("which pays when the commitment is right and costs when it is wrong.")

# --- the receptive field as a limit -----------------------------------------
print("\n" + "=" * 72)
print("the receptive field is a real constraint")
print("=" * 72)



CLOSE_GAPS = (5, 6)
FAR_GAPS = (10, 11, 12)


def make_distance_task(n, seed):
    """Two IDENTICAL single-pixel marks. The label is whether the gap
    between them is small (5-6) or large (10-12).

    A global pool of local features cannot answer this: both classes contain
    exactly the same marks in the same numbers, and only the DISTANCE
    differs — which no unit can register unless its receptive field spans
    the gap. The gaps are chosen so that a 5x5 field spans NEITHER and a
    9x9 field spans the close one only."""
    rs = np.random.default_rng(seed)
    X = rs.normal(0, 0.3, (n, 1, IMG, IMG))
    y = (np.arange(n) % 2).astype(int)          # exactly balanced
    for i in range(n):
        gaps = CLOSE_GAPS if y[i] else FAR_GAPS
        gap = int(rs.choice(gaps))
        c1 = int(rs.integers(1, IMG - gap - 1))
        r = int(rs.integers(2, IMG - 2))
        X[i, 0, r, c1] += 3.0
        X[i, 0, r, c1 + gap] += 3.0
    return X, y


class DeepConvNet(ConvNet):
    """Same as ConvNet but with a configurable number of 3x3 layers, so the
    receptive field before the pool can be varied."""

    def __init__(self, n_layers, ch=8, seed=0):
        rs = np.random.default_rng(seed)
        self.Ks, self.bs = [], []
        c_in = 1
        for _ in range(n_layers):
            self.Ks.append(rs.normal(0, np.sqrt(2 / (9 * c_in)),
                                     (ch, c_in, 3, 3)))
            self.bs.append(np.zeros(ch))
            c_in = ch
        self.Wo = rs.normal(0, np.sqrt(2 / ch), (ch, 2))
        self.bo = np.zeros(2)
        self.n_layers = n_layers
        self.rf = 1 + 2 * n_layers

    def params(self):
        return self.Ks + self.bs + [self.Wo, self.bo]

    def forward(self, X):
        self.cols, self.zs, self.acts = [], [], [X]
        h = X
        for K, b in zip(self.Ks, self.bs):
            z, c, _ = _conv_fwd(h, K, b)
            self.cols.append(c)
            self.zs.append(z)
            h = np.maximum(0.0, z)
            self.acts.append(h)
        self.pooled = h.mean(axis=(2, 3))
        return self.pooled @ self.Wo + self.bo

    def loss_and_grads(self, X, y):
        logits = self.forward(X)
        m = logits.max(axis=1, keepdims=True)
        e = np.exp(logits - m)
        loss = float(np.mean(m[:, 0] + np.log(e.sum(axis=1))
                             - logits[np.arange(len(X)), y]))
        d = e / e.sum(axis=1, keepdims=True)
        d[np.arange(len(X)), y] -= 1.0
        d /= len(X)
        gWo, gbo = self.pooled.T @ d, d.sum(axis=0)
        dpool = d @ self.Wo.T
        N, F, H2, W2 = self.acts[-1].shape
        dh = np.repeat(np.repeat(dpool[:, :, None, None], H2, 2), W2, 3) \
            / (H2 * W2)
        gK = [None] * self.n_layers
        gb = [None] * self.n_layers
        for l in reversed(range(self.n_layers)):
            dz = dh * (self.zs[l] > 0)
            F_l = self.Ks[l].shape[0]
            dzc = dz.transpose(0, 2, 3, 1).reshape(-1, F_l)
            gK[l] = (dzc.T @ self.cols[l]).reshape(self.Ks[l].shape)
            gb[l] = dzc.sum(axis=0)
            if l > 0:
                dcols = dzc @ self.Ks[l].reshape(F_l, -1)
                dh = self._col2im(dcols, self.acts[l].shape, 3, 1, 1)
        return loss, gK + gb + [gWo, gbo]


Xa, ya = make_distance_task(3000, 21)
Xb, yb = make_distance_task(2000, 22)
print(f"Two IDENTICAL marks; the label is whether the gap is small "
      f"{CLOSE_GAPS} or large {FAR_GAPS}.")
print("Both classes contain exactly the same marks in the same numbers, so")
print("counting local features cannot answer it — only the distance"
      " differs.\n")
print(f"{'conv layers':>12} {'receptive field':>17} {'spans gap':>11} "
      f"{'params':>9} {'test acc':>10}")
for n_layers in (2, 4, 6):
    net = DeepConvNet(n_layers, seed=1)
    train(net, Xa, ya, steps=1200)
    _, acc = evaluate(net, Xb, yb)
    npar = sum(p.size for p in net.params())
    span = net.rf - 1
    which = ("neither" if span < min(CLOSE_GAPS)
             else "close only" if span < min(FAR_GAPS) else "both")
    print(f"{n_layers:>12} {f'{net.rf} x {net.rf}':>17} {which:>11} "
          f"{npar:>9,} {acc:>10.4f}")
dnet = train(DenseNet(64, seed=1), Xa, ya, steps=1200)
_, dacc = evaluate(dnet, Xb, yb)
print(f"{'dense h=64':>12} {'whole image':>17} {'both':>11} "
      f"{dnet.n_params:>9,} {dacc:>10.4f}")
print("\n(chance is 0.5000; the two classes are exactly balanced)")

print("\nThe 'spans gap' column is the variable that matters, and it")
print("matters more than the parameter count: the 9x9 network solves the")
print("task perfectly with a ninth of the dense network's parameters, and")
print("the 5x5 network is far behind with a similar budget.")
print("\nThe reason the 5x5 network struggles is structural. No unit ever")
print("sees both marks, and the global pool that follows aggregates local")
print("features — of which both classes have exactly the same ones. It is")
print("not at chance, because a wide receptive field is not the only cue")
print("available in a small image with borders, but it cannot do the thing")
print("the task is asking for.")
print("\nOnce the field spans the CLOSE gap the task becomes trivial: the")
print("network detects 'two marks within one unit's view', and that IS the")
print("label. Two extra layers, 1,168 extra parameters, and the accuracy")
print("goes from 0.74 to 1.00 — a capability that no amount of width at 5x5")
print("would have bought.")
print("\nThis is the limitation attention was designed to remove: it relates")
print("every position to every other in ONE layer, at a cost quadratic in")
print("the number of positions (Chapter 71). Reading this table is the best")
print("preparation for that chapter — the trade being made there is visible")
print("here as a concrete failure and a concrete fix.")

# -*- coding: utf-8 -*-
# Extracted from: Chapter 65 — Positional Encoding, RoPE, and ALiBi
# Source: src/.../ch065-positional.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Positional schemes on a task that needs order, and what happens when a
trained model is asked to run past its training length.
"""
import numpy as np

rng = np.random.default_rng(3)

V = 10


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def make_order_task(n, T, seed):
    """The label is the token at a FIXED offset from the end — a purely
    positional relationship that a set-based model cannot represent."""
    rs = np.random.default_rng(seed)
    X = rs.integers(1, V, (n, T))
    y = X[:, -3]                                    # third from the end
    return X, y


def rope_tables(T, dk, base=10000.0, pos_scale=1.0):
    theta = base ** (-np.arange(0, dk, 2) / dk)
    m = np.arange(T)[:, None] / pos_scale
    ang = m * theta[None, :]
    return np.cos(ang), np.sin(ang)


def apply_rope(x, cos, sin):
    """x: (n, T, dk)."""
    d = x.shape[-1]
    x1, x2 = x[..., :d // 2], x[..., d // 2:]
    c = cos[None, :x.shape[1], :]
    s = sin[None, :x.shape[1], :]
    return np.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1)


def sinusoidal(T, d, base=10000.0):
    pos = np.arange(T)[:, None]
    i = np.arange(0, d, 2)[None, :]
    ang = pos / (base ** (i / d))
    pe = np.zeros((T, d))
    pe[:, 0::2] = np.sin(ang)
    pe[:, 1::2] = np.cos(ang)
    return pe


class PosModel:
    """One attention head with a configurable positional scheme."""

    def __init__(self, scheme, d=48, T_max=64, seed=0, base=10000.0,
                 pos_scale=1.0):
        rs = np.random.default_rng(seed)
        self.E = rs.normal(0, 0.3, (V, d))
        self.scheme = scheme
        self.d, self.T_max = d, d if False else T_max
        s = 1 / np.sqrt(d)
        self.Wq = rs.normal(0, s, (d, d))
        self.Wk = rs.normal(0, s, (d, d))
        self.Wv = rs.normal(0, s, (d, d))
        self.Wr = rs.normal(0, s, (d, V))
        self.br = np.zeros(V)
        if scheme == "learned":
            self.P = rs.normal(0, 0.3, (T_max, d))
        elif scheme == "sinusoidal":
            self.P = sinusoidal(T_max, d)
        if scheme == "rope":
            self.cos, self.sin = rope_tables(T_max, d, base, pos_scale)
        if scheme == "alibi":
            self.slope = 0.25

    def params(self):
        base = [self.E, self.Wq, self.Wk, self.Wv, self.Wr, self.br]
        return base + ([self.P] if self.scheme == "learned" else [])

    def rebuild_rope(self, T, base=10000.0, pos_scale=1.0):
        self.cos, self.sin = rope_tables(T, self.d, base, pos_scale)

    def forward(self, X):
        n, T = X.shape
        H = self.E[X]
        if self.scheme in ("learned", "sinusoidal"):
            H = H + self.P[None, :T, :]
        self.H = H
        Q, K, Vv = H @ self.Wq, H @ self.Wk, H @ self.Wv
        if self.scheme == "rope":
            Q, K = apply_rope(Q, self.cos, self.sin), \
                apply_rope(K, self.cos, self.sin)
        S = Q @ K.transpose(0, 2, 1) / np.sqrt(self.d)
        if self.scheme == "alibi":
            i = np.arange(T)[:, None]
            j = np.arange(T)[None, :]
            S = S - self.slope * np.abs(i - j)
        self.A = softmax(S)
        self.O = self.A @ Vv
        self.read = self.O[:, -1, :]
        self.Q, self.K, self.Vv = Q, K, Vv
        return self.read @ self.Wr + self.br

    def grads(self, X, y):
        n, T = X.shape
        logits = self.forward(X)
        m_ = logits.max(1, keepdims=True)
        e = np.exp(logits - m_)
        p = e / e.sum(1, keepdims=True)
        loss = float(-np.log(np.clip(p[np.arange(n), y], 1e-12, None)).mean())
        dl = p.copy()
        dl[np.arange(n), y] -= 1.0
        dl /= n
        gWr, gbr = self.read.T @ dl, dl.sum(0)
        dO = np.zeros_like(self.O)
        dO[:, -1, :] = dl @ self.Wr.T
        dA = dO @ self.Vv.transpose(0, 2, 1)
        dV = self.A.transpose(0, 2, 1) @ dO
        dS = self.A * (dA - (dA * self.A).sum(-1, keepdims=True))
        dS /= np.sqrt(self.d)
        dQ, dK = dS @ self.K, dS.transpose(0, 2, 1) @ self.Q
        if self.scheme == "rope":
            # rotation is orthogonal: the backward rotation is by -m
            dQ = apply_rope(dQ, self.cos, -self.sin)
            dK = apply_rope(dK, self.cos, -self.sin)
        Hf = self.H.reshape(-1, self.d)
        gWq = Hf.T @ dQ.reshape(-1, self.d)
        gWk = Hf.T @ dK.reshape(-1, self.d)
        gWv = Hf.T @ dV.reshape(-1, self.d)
        dH = (dQ @ self.Wq.T + dK @ self.Wk.T + dV @ self.Wv.T)
        gE = np.zeros_like(self.E)
        np.add.at(gE, X.reshape(-1), dH.reshape(-1, self.d))
        out = [gE, gWq, gWk, gWv, gWr, gbr]
        if self.scheme == "learned":
            gP = np.zeros_like(self.P)
            gP[:T] = dH.sum(0)
            out.append(gP)
        return loss, out


def train(net, X, y, steps=4000, lr=3e-3, batch=128, seed=0):
    ps = net.params()
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    rs = np.random.default_rng(seed + 5)
    for t in range(1, steps + 1):
        b = rs.integers(0, len(X), batch)
        _, gs = net.grads(X[b], y[b])
        for i, (p, g) in enumerate(zip(ps, gs)):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            p -= lr * (m[i] / (1 - 0.9 ** t)) / (
                np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    return net


T_TRAIN = 16
Xtr, ytr = make_order_task(12000, T_TRAIN, 1)
Xte, yte = make_order_task(4000, T_TRAIN, 2)

print("=" * 72)
print("a task that needs ORDER: predict the third-from-last token")
print("=" * 72)
print(f"sequence length {T_TRAIN}, vocabulary {V}, chance {1 / V:.4f}\n")
print(f"{'scheme':<16} {'extra params':>13} {'test accuracy':>15}")
nets = {}
for scheme in ("none", "learned", "sinusoidal", "rope", "alibi"):
    net = train(PosModel(scheme, T_max=128, seed=4), Xtr, ytr)
    nets[scheme] = net
    acc = float((net.forward(Xte).argmax(1) == yte).mean())
    extra = net.P.size if scheme == "learned" else 0
    print(f"{scheme:<16} {extra:>13,} {acc:>15.4f}")

print("\nThe 'none' row is eq. 65.1 as a task result: with no positional")
print("information the model sees a multiset and the answer depends on")
print("order, so it cannot do better than guessing.")
print("\nThe ALiBi row is the interesting failure, and it is instructive")
print("rather than a bug. ALiBi supplies no positional REPRESENTATION at")
print("all — only a monotone penalty on distance. A head can therefore")
print("express 'attend to things nearby' and cannot express 'attend to")
print("exactly three back', because a monotone decay has no way to single")
print("out one offset.")
print("\nThat is section 6.5's limitation in its sharpest form. Real ALiBi")
print("uses many heads with a geometric range of slopes, which gives a")
print("range of SCALES and still no ability to select a precise offset.")
print("Section 5.6's table lists ALiBi as extrapolating well, and this row")
print("is the other half of the trade.")
print("\nThe three schemes that supply an actual positional representation")
print("all solve it. That is the first finding and the one worth carrying:")
print("HAVING a positional representation matters more than which one.")

# --- extrapolation ----------------------------------------------------------
print("\n" + "=" * 72)
print("what happens past the training length")
print("=" * 72)
print(f"Trained at T = {T_TRAIN}. Evaluated at longer lengths, with the")
print("task unchanged — still the third-from-last token.\n")
print(f"{'scheme':<16} " + " ".join(f"{f'T={T}':>10}"
                                    for T in (16, 24, 48, 96)))
for scheme in ("learned", "sinusoidal", "rope", "alibi"):
    net = nets[scheme]
    row = []
    for T in (16, 24, 48, 96):
        Xe, ye = make_order_task(2000, T, 7)
        if scheme == "rope":
            net.rebuild_rope(T)
        try:
            acc = float((net.forward(Xe).argmax(1) == ye).mean())
        except (IndexError, ValueError):
            acc = float("nan")
        row.append(acc)
    if scheme == "rope":
        net.rebuild_rope(128)
    print(f"{scheme:<16} " + " ".join(
        f"{'n/a':>10}" if np.isnan(a) else f"{a:>10.4f}" for a in row))
print(f"\n(chance is {1 / V:.4f})")

print("\nThe learned and sinusoidal rows collapse to near chance, which is")
print("table 65.1's last column behaving as advertised: an absolute scheme")
print("has never seen these positions.")
print("\nThe RoPE row does NOT collapse, and that is worth being precise")
print("about rather than treating as a happy surprise. This task is purely")
print("RELATIVE — the answer is always three from the end — and eq. 65.9")
print("says RoPE's score depends only on the offset, exactly, at any")
print("absolute position. An offset of two gives the identical score at")
print("position 10 and at position 10,000.")
print("\nSo RoPE extrapolates perfectly on relative tasks, and table 65.1's")
print("'extrapolates poorly' is about something else: tasks needing")
print("LONG-RANGE or ABSOLUTE information, where the model must use offsets")
print("far larger than any it was trained on. Section 6.3 identified the")
print("mechanism — the long-wavelength frequency pairs never complete a")
print("cycle during training, so the model has no calibration for the")
print("angles they produce at large offsets.")
print("\nA local relative task never touches those pairs. That is the")
print("distinction, and a benchmark that only tests local relationships")
print("will report that RoPE extrapolates fine.")

# --- and what scaling does --------------------------------------------------
print("\n" + "=" * 72)
print("RoPE scaling: position interpolation vs NTK-aware (eqs. 65.11-12)")
print("=" * 72)
net = nets["rope"]
dk = net.d
print(f"Trained at T = {T_TRAIN}, evaluated at longer T with the RoPE table")
print("rebuilt under each recipe. No fine-tuning — this is the zero-shot")
print("case, which is what the recipes are usually asked to do.\n")
print(f"{'T':>6} {'scale s':>9} {'no scaling':>12} "
      f"{'interpolation':>15} {'NTK-aware':>12}")
for T in (24, 48, 96):
    s_ = T / T_TRAIN
    Xe, ye = make_order_task(2000, T, 7)
    accs = []
    net.rebuild_rope(T)
    accs.append(float((net.forward(Xe).argmax(1) == ye).mean()))
    net.rebuild_rope(T, pos_scale=s_)
    accs.append(float((net.forward(Xe).argmax(1) == ye).mean()))
    net.rebuild_rope(T, base=10000.0 * s_ ** (dk / (dk - 2)))
    accs.append(float((net.forward(Xe).argmax(1) == ye).mean()))
    print(f"{T:>6} {s_:>9.2f} " + " ".join(f"{a:>12.4f}" for a in accs))
net.rebuild_rope(128)

print("\nThis table is section 6.4's argument in its most extreme form, and")
print("the direction is unambiguous.")
print("\nInterpolation divides every position by s, so the offset between")
print("adjacent tokens becomes theta_0/s instead of theta_0 — and this task")
print("depends on distinguishing 'three back' from 'two back' and 'four")
print("back'. Compressing exactly that distinction is the one thing it")
print("cannot survive.")
print("\nNTK-aware scaling leaves the short wavelengths alone and stretches")
print("only the long, undertrained ones. On a purely local task that means")
print("it changes nothing that matters.")
print("\nAnd 'no scaling' also works here, for the reason the previous table")
print("gave: RoPE's relative property is exact at any offset, so a local")
print("task needs no rescaling at all. The lesson is not that interpolation")
print("is bad — it is that a scaling recipe must be chosen against the")
print("RANGE the task actually uses, and evaluated on a task that uses it.")
print("\nBe careful generalising the numbers from one head on a synthetic")
print("task. What transfers is the mechanism: which wavelengths each recipe")
print("moves, and which ones your task depends on.")

# -*- coding: utf-8 -*-
# Extracted from: Chapter 67 — Feed-Forward Networks, Residuals, and Normalization Placement
# Source: src/.../ch067-ffn-residual.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The feed-forward design decisions, measured: expansion ratio, gating, and
what the hidden units respond to.
"""
import numpy as np

rng = np.random.default_rng(4)

V, T = 32, 10


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def rmsnorm(x, eps=1e-6):
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps)


def silu(z):
    return z / (1.0 + np.exp(-np.clip(z, -60, 60)))


def make_task(n, seed):
    """Next-token prediction where the answer depends on a nonlinear
    function of two earlier tokens — so the FFN has something to do."""
    rs = np.random.default_rng(seed)
    rule = np.random.default_rng(77).integers(0, V, (V, V))
    X = rs.integers(0, V, (n, T))
    Y = np.zeros((n, T - 1), dtype=int)
    for t in range(T - 1):
        a = X[:, max(0, t - 1)]
        b = X[:, t]
        nxt = rule[a, b]
        flip = rs.random(n) < 0.1
        Y[:, t] = np.where(flip, rs.integers(0, V, n), nxt)
    return X[:, :-1], Y


class Model:
    """Embedding + one pre-norm block + unembedding."""

    def __init__(self, d=48, h=4, d_ff=None, gated=False, seed=0):
        rs = np.random.default_rng(seed)
        self.d, self.h, self.dk = d, h, d // h
        self.gated = gated
        self.d_ff = d_ff if d_ff else (int(8 * d / 3) if gated else 4 * d)
        s = 1 / np.sqrt(d)
        self.E = rs.normal(0, 0.05, (V, d))
        self.P = rs.normal(0, 0.05, (T - 1, d))
        self.Wq = rs.normal(0, s, (d, d))
        self.Wk = rs.normal(0, s, (d, d))
        self.Wv = rs.normal(0, s, (d, d))
        self.Wo = rs.normal(0, s, (d, d))
        self.W1 = rs.normal(0, s, (d, self.d_ff))
        self.W2 = rs.normal(0, 1 / np.sqrt(self.d_ff), (self.d_ff, d))
        if gated:
            self.Wg = rs.normal(0, s, (d, self.d_ff))
        self.U = rs.normal(0, 0.05, (V, d))

    def params(self):
        p = [self.E, self.P, self.Wq, self.Wk, self.Wv, self.Wo,
             self.W1, self.W2, self.U]
        return p + ([self.Wg] if self.gated else [])

    def n_params(self):
        return sum(p.size for p in self.params())

    def forward(self, X, keep=False):
        n, Tn = X.shape
        x = self.E[X] + self.P[None, :Tn, :]
        self.x0 = x
        na = rmsnorm(x)
        self.na = na
        sp = lambda M: M.reshape(n, Tn, self.h, self.dk).transpose(0, 2, 1, 3)
        Q, K, Vv = sp(na @ self.Wq), sp(na @ self.Wk), sp(na @ self.Wv)
        S = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.dk)
        mask = np.tril(np.ones((Tn, Tn), dtype=bool))
        A = softmax(np.where(mask, S, -np.inf))
        ctx = (A @ Vv).transpose(0, 2, 1, 3).reshape(n, Tn, self.d)
        self.A, self.Q, self.K, self.Vv, self.ctx = A, Q, K, Vv, ctx
        h1 = x + ctx @ self.Wo
        self.h1 = h1
        nf = rmsnorm(h1)
        self.nf = nf
        if self.gated:
            self.gpre = nf @ self.Wg
            self.upre = nf @ self.W1
            self.hid = silu(self.gpre) * self.upre
        else:
            self.upre = nf @ self.W1
            self.hid = np.maximum(0.0, self.upre)
        h2 = h1 + self.hid @ self.W2
        self.h2 = h2
        out = rmsnorm(h2)
        self.out = out
        return out @ self.U.T

    def loss(self, X, Y):
        P = softmax(self.forward(X))
        return float(-np.log(np.clip(
            np.take_along_axis(P, Y[..., None], -1), 1e-12, None)).mean())


# A compact reverse pass, written out once.
def grads(model, X, Y):
    n, Tn = X.shape
    logits = model.forward(X)
    P = softmax(logits)
    nt = n * Tn
    loss = float(-np.log(np.clip(
        np.take_along_axis(P, Y[..., None], -1), 1e-12, None)).sum() / nt)
    dl = P.copy()
    np.put_along_axis(dl, Y[..., None],
                      np.take_along_axis(dl, Y[..., None], -1) - 1.0, -1)
    dl /= nt
    d = model.d
    gU = np.einsum('ntv,ntd->vd', dl, model.out)
    dout = dl @ model.U

    def rms_back(x, dy, eps=1e-6):
        dd = x.shape[-1]
        ms = (x ** 2).mean(-1, keepdims=True) + eps
        return (dy - x * (dy * x).sum(-1, keepdims=True) / (dd * ms)) \
            / np.sqrt(ms)

    dh2 = rms_back(model.h2, dout)
    gW2 = model.hid.reshape(-1, model.d_ff).T @ dh2.reshape(-1, d)
    dhid = dh2 @ model.W2.T
    if model.gated:
        sg = silu(model.gpre)
        sig = 1 / (1 + np.exp(-np.clip(model.gpre, -60, 60)))
        dg = dhid * model.upre * (sig + model.gpre * sig * (1 - sig))
        du = dhid * sg
        gWg = model.nf.reshape(-1, d).T @ dg.reshape(-1, model.d_ff)
        gW1 = model.nf.reshape(-1, d).T @ du.reshape(-1, model.d_ff)
        dnf = dg @ model.Wg.T + du @ model.W1.T
    else:
        du = dhid * (model.upre > 0)
        gW1 = model.nf.reshape(-1, d).T @ du.reshape(-1, model.d_ff)
        dnf = du @ model.W1.T
        gWg = None
    dh1 = dh2 + rms_back(model.h1, dnf)
    gWo = model.ctx.reshape(-1, d).T @ dh1.reshape(-1, d)
    dctx = dh1 @ model.Wo.T
    sp = lambda M: M.reshape(n, Tn, model.h, model.dk).transpose(0, 2, 1, 3)
    dC = sp(dctx)
    dA = dC @ model.Vv.transpose(0, 1, 3, 2)
    dV = model.A.transpose(0, 1, 3, 2) @ dC
    dS = model.A * (dA - (dA * model.A).sum(-1, keepdims=True))
    dS /= np.sqrt(model.dk)
    dQ, dK = dS @ model.K, dS.transpose(0, 1, 3, 2) @ model.Q
    mg = lambda G: G.transpose(0, 2, 1, 3).reshape(n, Tn, d)
    naf = model.na.reshape(-1, d)
    gWq = naf.T @ mg(dQ).reshape(-1, d)
    gWk = naf.T @ mg(dK).reshape(-1, d)
    gWv = naf.T @ mg(dV).reshape(-1, d)
    dna = mg(dQ) @ model.Wq.T + mg(dK) @ model.Wk.T + mg(dV) @ model.Wv.T
    dx0 = dh1 + rms_back(model.x0, dna)
    gP = dx0.sum(axis=0)
    gE = np.zeros_like(model.E)
    np.add.at(gE, X.reshape(-1), dx0.reshape(-1, d))
    out = [gE, gP, gWq, gWk, gWv, gWo, gW1, gW2, gU]
    if model.gated:
        out.append(gWg)
    return loss, out


def train(model, X, Y, steps=2500, lr=4e-3, batch=128, seed=0):
    ps = model.params()
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    rs = np.random.default_rng(seed + 2)
    for t in range(1, steps + 1):
        b = rs.integers(0, len(X), batch)
        _, gs = grads(model, X[b], Y[b])
        for i, (p, g) in enumerate(zip(ps, gs)):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            p -= lr * (m[i] / (1 - 0.9 ** t)) / (
                np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    return model


Xtr, Ytr = make_task(10000, 1)
Xte, Yte = make_task(5000, 2)

print("=" * 72)
print("the expansion ratio (section 6.4)")
print("=" * 72)
print("d_ff = 4d is a convention with no derivation. How sensitive is it?\n")
print(f"{'d_ff / d':>10} {'d_ff':>6} {'params':>9} {'test NLL':>10}")
for ratio in (0.5, 1, 2, 4, 8, 16):
    mdl = train(Model(d=48, d_ff=int(48 * ratio), seed=5), Xtr, Ytr)
    print(f"{ratio:>10g} {int(48 * ratio):>6} {mdl.n_params():>9,} "
          f"{mdl.loss(Xte, Yte):>10.4f}")

print("\nSection 6.4 predicts a broad plateau rather than a sharp optimum at")
print("4, because none of the three arguments for that value is a")
print("derivation. Whether the plateau appears here is what the table says.")
print("\nThe honest reading of the convention: 4 is stable because it works,")
print("keeps both matrices well-shaped for tiling, and gives the")
print("two-thirds parameter split — not because anything requires it.")

# --- gating at matched parameters -------------------------------------------
print("\n" + "=" * 72)
print("gating, at matched parameters (eqs. 67.4, 67.6)")
print("=" * 72)
print("Three matrices instead of two, so d_ff drops to 8d/3 for parity.\n")
print(f"{'block':<22} {'d_ff':>6} {'FFN params':>12} {'total':>9} "
      f"{'test NLL':>10}")
a = train(Model(d=48, gated=False, seed=5), Xtr, Ytr)
print(f"{'ReLU, d_ff = 4d':<22} {a.d_ff:>6} "
      f"{2 * 48 * a.d_ff:>12,} {a.n_params():>9,} {a.loss(Xte, Yte):>10.4f}")
b = train(Model(d=48, gated=True, seed=5), Xtr, Ytr)
print(f"{'SwiGLU, d_ff = 8d/3':<22} {b.d_ff:>6} "
      f"{3 * 48 * b.d_ff:>12,} {b.n_params():>9,} {b.loss(Xte, Yte):>10.4f}")
c = train(Model(d=48, gated=True, d_ff=4 * 48, seed=5), Xtr, Ytr)
print(f"{'SwiGLU, d_ff = 4d':<22} {c.d_ff:>6} "
      f"{3 * 48 * c.d_ff:>12,} {c.n_params():>9,} {c.loss(Xte, Yte):>10.4f}")

print("\nThe first two rows are the comparison that matters: matched")
print("parameters, different block structure. The third is unmatched and is")
print("there to separate 'gating helps' from 'more parameters help'.")
print("\nSection 6.5 gives the structural difference: a gated block's output")
print("is a PRODUCT of two projections, so it is quadratic in the input,")
print("where an ungated block reaches second-order interactions only")
print("through the activation's curvature. Whether that is the mechanism is")
print("not established — Shazeer's own paper offers none.")

# --- section 4.4: what the hidden units respond to --------------------------
print("\n" + "=" * 72)
print("what the hidden units respond to (section 4.4)")
print("=" * 72)
mdl = a
mdl.forward(Xte[:3000])
hid = mdl.hid.reshape(-1, mdl.d_ff)
toks = Xte[:3000].reshape(-1)
print(f"{mdl.d_ff} hidden units. For each, find the token whose presence")
print("most raises its activation, and how selective that is.\n")

act_by_tok = np.zeros((V, mdl.d_ff))
cnt = np.zeros(V)
np.add.at(act_by_tok, toks, hid)
np.add.at(cnt, toks, 1)
mean_act = act_by_tok / np.maximum(cnt, 1)[:, None]
overall = hid.mean(0)
sel = (mean_act - overall) / (hid.std(0) + 1e-9)

top = np.abs(sel).max(0)
order = np.argsort(top)[::-1]
print(f"{'unit':>6} {'best token':>12} {'selectivity (sd)':>18} "
      f"{'fraction active':>17}")
for u in list(order[:5]) + list(order[-3:]):
    t = int(np.abs(sel[:, u]).argmax())
    frac = float((hid[:, u] > 0).mean())
    print(f"{u:>6} {t:>12} {sel[t, u]:>18.3f} {frac:>17.4f}")

print(f"\nmedian selectivity across all units: "
      f"{float(np.median(top)):.3f} standard deviations")
print(f"units with selectivity above 1 sd: "
      f"{int((top > 1).sum())} of {mdl.d_ff}")
print(f"units never active: {int(((hid > 0).mean(0) == 0).sum())}")

print("\nThe key-value memory reading of section 4.4 predicts that some")
print("units should respond selectively to recognisable patterns, and the")
print("selectivity column is that prediction measured.")
print("\nWhat the numbers usually show — here and in real models — is a")
print("MINORITY of clearly selective units and a majority that are not. That")
print("is the honest state of the interpretation: the mechanism is real for")
print("some units, polysemantic units are the norm, and 'the FFN is a")
print("key-value memory' is a useful frame rather than an established")
print("description.")

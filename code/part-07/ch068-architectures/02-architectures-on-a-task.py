# Extracted from: Chapter 68 — Encoder, Decoder, and Encoder–Decoder Transformers
# Source: src/.../ch068-architectures.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Encoder-only, decoder-only and prefix-LM on the same data, with the same
weights where possible — so the only variable is the mask.
"""
import numpy as np

rng = np.random.default_rng(2)

V, T = 24, 12


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def rmsnorm(x, eps=1e-6):
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps)


def rms_back(x, dy, eps=1e-6):
    d = x.shape[-1]
    ms = (x ** 2).mean(-1, keepdims=True) + eps
    return (dy - x * (dy * x).sum(-1, keepdims=True) / (d * ms)) / np.sqrt(ms)


def make_data(n, seed):
    """A sequence where each token depends on the two before it."""
    rs = np.random.default_rng(seed)
    rule = np.random.default_rng(31).integers(0, V, (V, V))
    X = np.zeros((n, T), dtype=int)
    X[:, 0] = rs.integers(0, V, n)
    X[:, 1] = rs.integers(0, V, n)
    for t in range(2, T):
        nxt = rule[X[:, t - 2], X[:, t - 1]]
        flip = rs.random(n) < 0.12
        X[:, t] = np.where(flip, rs.integers(0, V, n), nxt)
    return X


class Transformer:
    """One pre-norm block. The MASK is a constructor argument and nothing
    else changes."""

    def __init__(self, d=48, h=4, seed=0):
        rs = np.random.default_rng(seed)
        s = 1 / np.sqrt(d)
        self.d, self.h, self.dk, self.dff = d, h, d // h, 4 * d
        self.E = rs.normal(0, 0.05, (V + 1, d))       # +1 for [MASK]
        self.P = rs.normal(0, 0.05, (T, d))
        self.Wq = rs.normal(0, s, (d, d))
        self.Wk = rs.normal(0, s, (d, d))
        self.Wv = rs.normal(0, s, (d, d))
        self.Wo = rs.normal(0, s, (d, d))
        self.W1 = rs.normal(0, s, (d, self.dff))
        self.W2 = rs.normal(0, 1 / np.sqrt(self.dff), (self.dff, d))
        self.U = rs.normal(0, 0.05, (V, d))

    def params(self):
        return [self.E, self.P, self.Wq, self.Wk, self.Wv, self.Wo,
                self.W1, self.W2, self.U]

    def forward(self, X, mask):
        n, Tn = X.shape
        x0 = self.E[X] + self.P[None, :Tn, :]
        na = rmsnorm(x0)
        sp = lambda M: M.reshape(n, Tn, self.h, self.dk).transpose(0, 2, 1, 3)
        Q, K, Vv = sp(na @ self.Wq), sp(na @ self.Wk), sp(na @ self.Wv)
        S = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.dk)
        A = softmax(np.where(mask[:Tn, :Tn], S, -1e9))
        ctx = (A @ Vv).transpose(0, 2, 1, 3).reshape(n, Tn, self.d)
        h1 = x0 + ctx @ self.Wo
        nf = rmsnorm(h1)
        pre = nf @ self.W1
        hid = np.maximum(0.0, pre)
        h2 = h1 + hid @ self.W2
        out = rmsnorm(h2)
        self.cache = (X, x0, na, Q, K, Vv, A, ctx, h1, nf, pre, hid, h2, out)
        return out @ self.U.T

    def grads(self, X, mask, targets, weight):
        """weight: (n, T) 1 where the position contributes to the loss."""
        logits = self.forward(X, mask)
        (Xc, x0, na, Q, K, Vv, A, ctx, h1, nf, pre, hid, h2, out) = self.cache
        n, Tn = X.shape
        d = self.d
        P = softmax(logits)
        w = weight[..., None]
        nsup = max(weight.sum(), 1)
        loss = float(-(np.log(np.clip(
            np.take_along_axis(P, targets[..., None], -1), 1e-12, None))
            * w).sum() / nsup)
        dl = P.copy()
        np.put_along_axis(dl, targets[..., None],
                          np.take_along_axis(dl, targets[..., None], -1) - 1.0,
                          -1)
        dl = dl * w / nsup
        gU = np.einsum('ntv,ntd->vd', dl, out)
        dout = dl @ self.U
        dh2 = rms_back(h2, dout)
        gW2 = hid.reshape(-1, self.dff).T @ dh2.reshape(-1, d)
        dhid = dh2 @ self.W2.T
        dpre = dhid * (pre > 0)
        gW1 = nf.reshape(-1, d).T @ dpre.reshape(-1, self.dff)
        dh1 = dh2 + rms_back(h1, dpre @ self.W1.T)
        gWo = ctx.reshape(-1, d).T @ dh1.reshape(-1, d)
        dctx = dh1 @ self.Wo.T
        sp = lambda M: M.reshape(n, Tn, self.h, self.dk).transpose(0, 2, 1, 3)
        dC = sp(dctx)
        dA = dC @ Vv.transpose(0, 1, 3, 2)
        dV = A.transpose(0, 1, 3, 2) @ dC
        dS = A * (dA - (dA * A).sum(-1, keepdims=True)) / np.sqrt(self.dk)
        dQ, dK = dS @ K, dS.transpose(0, 1, 3, 2) @ Q
        mg = lambda G: G.transpose(0, 2, 1, 3).reshape(n, Tn, d)
        naf = na.reshape(-1, d)
        gWq = naf.T @ mg(dQ).reshape(-1, d)
        gWk = naf.T @ mg(dK).reshape(-1, d)
        gWv = naf.T @ mg(dV).reshape(-1, d)
        dna = mg(dQ) @ self.Wq.T + mg(dK) @ self.Wk.T + mg(dV) @ self.Wv.T
        dx0 = dh1 + rms_back(x0, dna)
        gP = dx0.sum(axis=0)
        gE = np.zeros_like(self.E)
        np.add.at(gE, Xc.reshape(-1), dx0.reshape(-1, d))
        return loss, [gE, gP, gWq, gWk, gWv, gWo, gW1, gW2, gU]


def make_mask(T, kind, prefix=0):
    if kind == "encoder":
        return np.ones((T, T), dtype=bool)
    if kind == "decoder":
        return np.tril(np.ones((T, T), dtype=bool))
    m = np.tril(np.ones((T, T), dtype=bool))
    m[:, :prefix] = True
    return m


MASK_ID = V


def batch_for(kind, X, rs, prefix=6, mlm_rate=0.15):
    """Return (inputs, targets, weight) for the given objective."""
    n = len(X)
    if kind == "encoder":
        inp = X.copy()
        sel = rs.random(X.shape) < mlm_rate
        sel[:, 0] = sel[:, 0] | (~sel.any(1))          # at least one
        inp[sel] = MASK_ID
        return inp, X, sel.astype(float)
    if kind == "decoder":
        return X[:, :-1], X[:, 1:], np.ones((n, T - 1))
    w = np.zeros((n, T - 1))
    w[:, prefix - 1:] = 1.0
    return X[:, :-1], X[:, 1:], w


def train(kind, Xtr, steps=3000, lr=3e-3, batch=128, seed=0, prefix=6):
    net = Transformer(seed=seed)
    mask = make_mask(T, kind if kind != "prefix" else "prefix",
                     prefix=prefix)
    ps = net.params()
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    rs = np.random.default_rng(seed + 4)
    for t in range(1, steps + 1):
        b = Xtr[rs.integers(0, len(Xtr), batch)]
        inp, tgt, w = batch_for(kind, b, rs, prefix=prefix)
        _, gs = net.grads(inp, mask, tgt, w)
        for i, (p, g) in enumerate(zip(ps, gs)):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            p -= lr * (m[i] / (1 - 0.9 ** t)) / (
                np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    return net, mask


Xtr, Xte = make_data(10000, 1), make_data(4000, 2)

print("=" * 72)
print("the same weights, the same data, three masks")
print("=" * 72)
print(f"Every model has identical architecture and parameter count; only")
print(f"the mask and the objective differ. Vocabulary {V}, length {T}.\n")

rs_eval = np.random.default_rng(9)
print(f"{'architecture':<16} {'objective':<22} {'supervised/pos':>15} "
      f"{'its own val loss':>18}")
models = {}
for kind, obj in (("encoder", "masked LM (15%)"),
                  ("decoder", "next-token"),
                  ("prefix", "prefix-LM (suffix)")):
    net, mask = train(kind, Xtr, seed=6)
    models[kind] = (net, mask)
    inp, tgt, w = batch_for(kind, Xte, np.random.default_rng(9))
    loss, _ = net.grads(inp, mask, tgt, w)
    print(f"{kind:<16} {obj:<22} {w.mean():>15.3f} {loss:>18.4f}")

print("\nThose losses are NOT comparable — each model is scored on its own")
print("objective, and predicting a masked token given both sides is an")
print("easier problem than predicting the next token given only the past.")
print("Eq. 68.10 says so: conditioning on a superset cannot raise the")
print("entropy.")
print("\nThe comparable question is what each can DO, and that is next.")

# --- what each can do -------------------------------------------------------
print("\n" + "=" * 72)
print("what each architecture can do")
print("=" * 72)
print("Task A: predict the NEXT token from the past only (generation).")
print("Task B: fill in a MASKED token given both sides (representation).\n")


def eval_next_token(net, mask, X):
    """Every model gets the same causal input; the mask is its own."""
    inp, tgt = X[:, :-1], X[:, 1:]
    logits = net.forward(inp, mask)
    return float((logits.argmax(-1) == tgt).mean())


def eval_fill(net, mask, X, rs):
    inp = X.copy()
    pos = rs.integers(1, T - 1, len(X))
    tgt = X[np.arange(len(X)), pos]
    inp[np.arange(len(X)), pos] = MASK_ID
    logits = net.forward(inp, mask)
    return float((logits[np.arange(len(X)), pos].argmax(-1) == tgt).mean())


print(f"{'architecture':<16} {'A: next-token acc':>19} "
      f"{'B: fill-in acc':>17}")
for kind in ("encoder", "decoder", "prefix"):
    net, mask = models[kind]
    a = eval_next_token(net, mask, Xte)
    b = eval_fill(net, mask, Xte, np.random.default_rng(11))
    print(f"{kind:<16} {a:>19.4f} {b:>17.4f}")
print(f"\n(chance is {1 / V:.4f})")

print("\nThe encoder's column-A number is the one to be careful about. Under")
print("a bidirectional mask, position i can see position i+1 — which IS the")
print("answer — so any number it produces there is meaningless as a")
print("generation score. That is section 5.2's structural point: an")
print("encoder-only model cannot generate, and the reason is not that it")
print("does badly but that the evaluation is not well posed.")

# --- the missing-mask bug ---------------------------------------------------
print("\n" + "=" * 72)
print("the missing causal mask: trains beautifully, cannot generate")
print("=" * 72)
print("Train with next-token prediction and NO causal mask, so every")
print("position can see the answer sitting next to it.\n")
net_bad = Transformer(seed=6)
mask_none = make_mask(T, "encoder")
ps = net_bad.params()
m_ = [np.zeros_like(p) for p in ps]
v_ = [np.zeros_like(p) for p in ps]
rs = np.random.default_rng(10)
for t in range(1, 3001):
    b = Xtr[rs.integers(0, len(Xtr), 128)]
    inp, tgt = b[:, :-1], b[:, 1:]
    w = np.ones_like(tgt, dtype=float)
    _, gs = net_bad.grads(inp, mask_none, tgt, w)
    for i, (p, g) in enumerate(zip(ps, gs)):
        m_[i] = 0.9 * m_[i] + 0.1 * g
        v_[i] = 0.999 * v_[i] + 0.001 * g * g
        p -= 3e-3 * (m_[i] / (1 - 0.9 ** t)) / (
            np.sqrt(v_[i] / (1 - 0.999 ** t)) + 1e-8)

inp, tgt = Xte[:, :-1], Xte[:, 1:]
w = np.ones_like(tgt, dtype=float)
l_bad, _ = net_bad.grads(inp, mask_none, tgt, w)
acc_bad = float((net_bad.forward(inp, mask_none).argmax(-1) == tgt).mean())
net_good, mask_good = models["decoder"]
l_good, _ = net_good.grads(inp, mask_good, tgt, w)
acc_good = float((net_good.forward(inp, mask_good).argmax(-1) == tgt).mean())

print(f"{'model':<26} {'train-time loss':>17} {'train-time acc':>16}")
print(f"{'NO causal mask':<26} {l_bad:>17.4f} {acc_bad:>16.4f}")
print(f"{'with causal mask':<26} {l_good:>17.4f} {acc_good:>16.4f}")

print("\nThe unmasked model looks far better, and it has learned nothing")
print("useful: it is copying position i+1 of its own input to slot i.")
print("\nNow generate. Feed only a prefix and extend it one token at a time,")
print("which is the only setting that matters:\n")


def generate(net, mask, X, n_ctx=4, steps=6):
    """Autoregressive generation from a prefix of n_ctx real tokens."""
    n = len(X)
    seq = np.zeros((n, T - 1), dtype=int)
    seq[:, :n_ctx] = X[:, :n_ctx]
    for i in range(n_ctx, min(n_ctx + steps, T - 1)):
        logits = net.forward(seq, mask)
        seq[:, i] = logits[:, i - 1].argmax(-1)
    return seq


for label, (net, mask) in (("NO causal mask", (net_bad, mask_none)),
                           ("with causal mask", (net_good, mask_good))):
    gen = generate(net, mask, Xte[:2000])
    true = Xte[:2000, :T - 1]
    hit = float((gen[:, 4:10] == true[:, 4:10]).mean())
    print(f"{label:<26} generated-token accuracy: {hit:.4f}")

print(f"\n(chance is {1 / V:.4f})")
print("\nThat is the whole lesson. The unmasked model's training metrics were")
print("excellent and its generation is at or near chance, because at")
print("generation time the future positions are zeros rather than the")
print("answers it learned to copy.")
print("\nA missing causal mask is a one-line bug that produces a model which")
print("passes every training-time check and is completely useless. It is")
print("worth building the generation test into the training loop for")
print("exactly this reason — it is the only check that catches it.")

# Extracted from: Chapter 68 — Encoder, Decoder, and Encoder–Decoder Transformers
# Source: src/.../ch068-architectures.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""When each shape wins: representation quality against generation, and the
gradient-efficiency argument measured.
"""
import numpy as np

rng = np.random.default_rng(3)

V, T = 24, 12


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


# The model and helpers are re-declared so this listing stands alone.
def rmsnorm(x, eps=1e-6):
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps)


def rms_back(x, dy, eps=1e-6):
    d = x.shape[-1]
    ms = (x ** 2).mean(-1, keepdims=True) + eps
    return (dy - x * (dy * x).sum(-1, keepdims=True) / (d * ms)) / np.sqrt(ms)


MASK_ID = V


def make_data(n, seed):
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
    def __init__(self, d=48, h=4, seed=0):
        rs = np.random.default_rng(seed)
        s = 1 / np.sqrt(d)
        self.d, self.h, self.dk, self.dff = d, h, d // h, 4 * d
        self.E = rs.normal(0, 0.05, (V + 1, d))
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

    def forward(self, X, mask, return_hidden=False):
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
        if return_hidden:
            return out
        return out @ self.U.T

    def grads(self, X, mask, targets, weight):
        logits = self.forward(X, mask)
        (Xc, x0, na, Q, K, Vv, A, ctx, h1, nf, pre, hid, h2,
         out) = self.cache
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


CAUSAL = np.tril(np.ones((T, T), dtype=bool))
FULL = np.ones((T, T), dtype=bool)

Xtr, Xte = make_data(12000, 1), make_data(5000, 2)


def train_mlm(rate, steps=3000, seed=6, lr=3e-3, batch=128):
    net = Transformer(seed=seed)
    ps = net.params()
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    rs = np.random.default_rng(seed + 4)
    for t in range(1, steps + 1):
        b = Xtr[rs.integers(0, len(Xtr), batch)]
        inp = b.copy()
        sel = rs.random(b.shape) < rate
        sel[np.arange(len(b)), rs.integers(0, T, len(b))] = True
        inp[sel] = MASK_ID
        _, gs = net.grads(inp, FULL, b, sel.astype(float))
        for i, (p, g) in enumerate(zip(ps, gs)):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            p -= lr * (m[i] / (1 - 0.9 ** t)) / (
                np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    return net


def train_lm(steps=3000, seed=6, lr=3e-3, batch=128):
    net = Transformer(seed=seed)
    ps = net.params()
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    rs = np.random.default_rng(seed + 4)
    for t in range(1, steps + 1):
        b = Xtr[rs.integers(0, len(Xtr), batch)]
        inp, tgt = b[:, :-1], b[:, 1:]
        _, gs = net.grads(inp, CAUSAL, tgt,
                          np.ones_like(tgt, dtype=float))
        for i, (p, g) in enumerate(zip(ps, gs)):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            p -= lr * (m[i] / (1 - 0.9 ** t)) / (
                np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    return net


# --- section 6.2: the masking-rate trade -------------------------------------
print("=" * 72)
print("why BERT masks 15% and not more (section 6.2)")
print("=" * 72)
print("Masking more positions gives more loss terms per pass AND removes")
print("the context the other predictions need. There is an optimum below 1.\n")
rs_ev = np.random.default_rng(21)


def fill_accuracy(net, X, n_probe=3000):
    rs = np.random.default_rng(77)
    Xp = X[:n_probe]
    inp = Xp.copy()
    pos = rs.integers(1, T - 1, len(Xp))
    tgt = Xp[np.arange(len(Xp)), pos]
    inp[np.arange(len(Xp)), pos] = MASK_ID
    lg = net.forward(inp, FULL)
    return float((lg[np.arange(len(Xp)), pos].argmax(-1) == tgt).mean())


print(f"{'mask rate':>11} {'loss terms/pass':>17} {'fill-in accuracy':>18}")
for rate in (0.05, 0.15, 0.30, 0.50, 0.80):
    net = train_mlm(rate)
    print(f"{rate:>11.2f} {rate * T:>17.2f} {fill_accuracy(net, Xte):>18.4f}")

print("\nThe two effects pull against each other and the table is where they")
print("balance on this task. BERT's 15% was chosen empirically and this is")
print("the shape of the curve that choice sits on.")
print("\nThe point for the architecture argument is that an optimum below 1")
print("EXISTS at all. Next-token prediction supervises every position with")
print("no such trade, because the causal mask removes the future rather")
print("than the context — eq. 68.6's factorisation is what buys that.")

# --- representation quality --------------------------------------------------
print("\n" + "=" * 72)
print("what bidirectionality buys, measured (eq. 68.10)")
print("=" * 72)
print("Freeze each model and fit a linear probe on its hidden states to")
print("predict the token at that position from its CONTEXT (the token")
print("itself is masked out). More information in the representation means")
print("a better probe.\n")


def probe_accuracy(net, mask, X, n=4000, seed=0):
    rs = np.random.default_rng(seed)
    Xp = X[:n]
    inp = Xp.copy()
    pos = rs.integers(1, T - 1, len(Xp))
    tgt = Xp[np.arange(len(Xp)), pos]
    inp[np.arange(len(Xp)), pos] = MASK_ID
    H = net.forward(inp, mask, return_hidden=True)
    feats = H[np.arange(len(Xp)), pos]
    # ridge-regularised multinomial probe, closed form on one-hot targets
    Y = np.zeros((len(Xp), V))
    Y[np.arange(len(Xp)), tgt] = 1.0
    ntr = int(0.7 * len(Xp))
    A = feats[:ntr].T @ feats[:ntr] + 1e-2 * np.eye(feats.shape[1])
    W = np.linalg.solve(A, feats[:ntr].T @ Y[:ntr])
    pred = (feats[ntr:] @ W).argmax(1)
    return float((pred == tgt[ntr:]).mean())


mlm = train_mlm(0.15)
lm = train_lm()
print(f"{'model':<26} {'mask at probe time':<20} {'probe accuracy':>16}")
print(f"{'MLM-trained (encoder)':<26} {'bidirectional':<20} "
      f"{probe_accuracy(mlm, FULL, Xte, seed=1):>16.4f}")
print(f"{'LM-trained (decoder)':<26} {'causal':<20} "
      f"{probe_accuracy(lm, CAUSAL, Xte, seed=1):>16.4f}")
print(f"{'LM-trained (decoder)':<26} {'bidirectional':<20} "
      f"{probe_accuracy(lm, FULL, Xte, seed=1):>16.4f}")
print(f"\n(chance is {1 / V:.4f})")

print("\nEq. 68.10 is an information-theoretic fact: conditioning on both")
print("sides cannot give a higher conditional entropy than conditioning on")
print("one. So a bidirectional representation is AT LEAST as informative,")
print("and this task — where the answer depends on the two PRECEDING tokens")
print("— is one where the future genuinely helps identify a corrupted")
print("position.")
print("\nThe third row is the interesting one: a causally-trained model run")
print("under a bidirectional mask at probe time. It sees the future it was")
print("never trained to use, and whether that helps says how much of the")
print("gap is the OBJECTIVE and how much is the MASK.")

# --- and the generation side -------------------------------------------------
print("\n" + "=" * 72)
print("and what causal training buys")
print("=" * 72)


def next_token_acc(net, mask, X):
    inp, tgt = X[:, :-1], X[:, 1:]
    lg = net.forward(inp, mask)
    return float((lg.argmax(-1) == tgt).mean())


print(f"{'model':<26} {'next-token accuracy (causal mask)':>36}")
print(f"{'LM-trained':<26} {next_token_acc(lm, CAUSAL, Xte):>36.4f}")
print(f"{'MLM-trained':<26} {next_token_acc(mlm, CAUSAL, Xte):>36.4f}")
print(f"\n(chance is {1 / V:.4f})")

print("\nThe MLM model is evaluated here under a CAUSAL mask, which is the")
print("only well-posed way to ask it to predict a next token — under its")
print("own bidirectional mask it would simply read the answer.")
print("\nIt has never seen a causal mask in training, so its representations")
print("are not organised for prefix-conditional prediction. That gap is what")
print("section 7.5 says a fine-tune closes, and it is why converting")
print("between the shapes is possible but not free.")
print("\nTaken with the previous table, this is the whole architecture")
print("argument in two numbers: bidirectional wins at representation, causal")
print("wins at generation, and the field chose the one that can be trained")
print("on every position of every document and then asked to do both.")

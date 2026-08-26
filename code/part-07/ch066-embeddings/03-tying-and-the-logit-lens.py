# -*- coding: utf-8 -*-
# Extracted from: Chapter 66 — Token Embeddings and the Unembedding Matrix
# Source: src/.../ch066-embeddings.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Weight tying measured at two scales, and the logit lens."""
import numpy as np

rng = np.random.default_rng(5)


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


V, T = 40, 12


def make_lm_task(n, seed):
    """A small language-modelling-like task: predict the next token, where
    the next token depends on a learnable function of the last two."""
    rs = np.random.default_rng(seed)
    rule = np.random.default_rng(99).integers(0, V, (V, V))
    X = np.zeros((n, T), dtype=int)
    X[:, 0] = rs.integers(0, V, n)
    X[:, 1] = rs.integers(0, V, n)
    for t in range(2, T):
        nxt = rule[X[:, t - 2], X[:, t - 1]]
        flip = rs.random(n) < 0.15                    # 15% noise
        X[:, t] = np.where(flip, rs.integers(0, V, n), nxt)
    return X[:, :-1], X[:, 1:]


class TinyLM:
    """Embedding -> attention -> FFN -> unembedding, optionally tied."""

    def __init__(self, d=32, tied=False, seed=0, out_scale=1.0):
        rs = np.random.default_rng(seed)
        self.E = rs.normal(0, 0.02, (V, d))
        self.P = rs.normal(0, 0.02, (T - 1, d))
        s = 1 / np.sqrt(d)
        self.Wq = rs.normal(0, s, (d, d))
        self.Wk = rs.normal(0, s, (d, d))
        self.Wv = rs.normal(0, s, (d, d))
        self.W1 = rs.normal(0, s, (d, 4 * d))
        self.W2 = rs.normal(0, np.sqrt(1 / (4 * d)), (4 * d, d))
        self.tied, self.d, self.out_scale = tied, d, out_scale
        if not tied:
            self.U = rs.normal(0, 0.02, (V, d))
        self.bo = np.zeros(V)

    def params(self):
        base = [self.E, self.P, self.Wq, self.Wk, self.Wv, self.W1, self.W2,
                self.bo]
        return base if self.tied else base + [self.U]

    def n_params(self):
        return sum(p.size for p in self.params())

    def unemb(self):
        return self.E if self.tied else self.U

    def forward(self, X, keep_layers=False):
        n, Tn = X.shape
        H0 = self.E[X] + self.P[None, :Tn, :]
        Q, K, Vv = H0 @ self.Wq, H0 @ self.Wk, H0 @ self.Wv
        S = Q @ K.transpose(0, 2, 1) / np.sqrt(self.d)
        mask = np.tril(np.ones((Tn, Tn), dtype=bool))
        S = np.where(mask, S, -np.inf)
        self.A = softmax(S)
        H1 = H0 + self.A @ Vv                          # residual
        Z = H1 @ self.W1
        Hh = np.maximum(0.0, Z)
        H2 = H1 + Hh @ self.W2                         # residual
        self.H0, self.H1, self.H2, self.Z, self.Hh = H0, H1, H2, Z, Hh
        self.X, self.Vv, self.Q, self.K = X, Vv, Q, K
        if keep_layers:
            self.layers = {"embed": H0, "after attn": H1, "after ffn": H2}
        return H2 @ self.unemb().T * self.out_scale + self.bo

    def grads(self, X, Y):
        n, Tn = X.shape
        logits = self.forward(X)
        P = softmax(logits)
        nt = n * Tn
        loss = float(-np.log(np.clip(
            np.take_along_axis(P, Y[..., None], -1), 1e-12, None)).sum() / nt)
        dl = P.copy()
        np.put_along_axis(dl, Y[..., None],
                          np.take_along_axis(dl, Y[..., None], -1) - 1.0, -1)
        dl /= nt
        U = self.unemb()
        gU = np.einsum('ntv,ntd->vd', dl, self.H2) * self.out_scale
        gbo = dl.sum(axis=(0, 1))
        dH2 = (dl @ U) * self.out_scale
        gW2 = np.einsum('nth,ntd->hd', self.Hh, dH2)
        dHh = dH2 @ self.W2.T
        dZ = dHh * (self.Z > 0)
        gW1 = np.einsum('ntd,nth->dh', self.H1, dZ)
        dH1 = dH2 + dZ @ self.W1.T
        dctx = dH1
        dA = dctx @ self.Vv.transpose(0, 2, 1)
        dV = self.A.transpose(0, 2, 1) @ dctx
        dS = self.A * (dA - (dA * self.A).sum(-1, keepdims=True))
        dS /= np.sqrt(self.d)
        dQ, dK = dS @ self.K, dS.transpose(0, 2, 1) @ self.Q
        H0f = self.H0.reshape(-1, self.d)
        gWq = H0f.T @ dQ.reshape(-1, self.d)
        gWk = H0f.T @ dK.reshape(-1, self.d)
        gWv = H0f.T @ dV.reshape(-1, self.d)
        dH0 = dH1 + dQ @ self.Wq.T + dK @ self.Wk.T + dV @ self.Wv.T
        gP = dH0.sum(axis=0)
        gE = np.zeros_like(self.E)
        np.add.at(gE, X.reshape(-1), dH0.reshape(-1, self.d))
        if self.tied:
            gE = gE + gU                               # BOTH paths
            return loss, [gE, gP, gWq, gWk, gWv, gW1, gW2, gbo]
        return loss, [gE, gP, gWq, gWk, gWv, gW1, gW2, gbo, gU]


def train(net, X, Y, steps=4000, lr=3e-3, batch=128, seed=0):
    ps = net.params()
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    rs = np.random.default_rng(seed + 3)
    for t in range(1, steps + 1):
        b = rs.integers(0, len(X), batch)
        _, gs = net.grads(X[b], Y[b])
        for i, (p, g) in enumerate(zip(ps, gs)):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            p -= lr * (m[i] / (1 - 0.9 ** t)) / (
                np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    return net


def evaluate(net, X, Y):
    P = softmax(net.forward(X))
    nll = float(-np.log(np.clip(
        np.take_along_axis(P, Y[..., None], -1), 1e-12, None)).mean())
    return nll


Xtr, Ytr = make_lm_task(8000, 1)
Xte, Yte = make_lm_task(4000, 2)

print("=" * 72)
print("weight tying at two scales (eq. 66.9)")
print("=" * 72)
print(f"vocabulary {V}. The embedding fraction changes with d, so the")
print("benefit of tying should change with it too.\n")
print(f"{'d':>5} {'embed fraction':>16} {'untied: params':>16} {'NLL':>8}  "
      f"{'tied: params':>14} {'NLL':>8}  {'tying helps?':>13}")
for d in (8, 16, 32, 64):
    a = train(TinyLM(d=d, tied=False, seed=7), Xtr, Ytr)
    b = train(TinyLM(d=d, tied=True, seed=7), Xtr, Ytr)
    na, nb = evaluate(a, Xte, Yte), evaluate(b, Xte, Yte)
    frac = 2 * V * d / a.n_params()
    print(f"{d:>5} {frac:>16.1%} {a.n_params():>16,} {na:>8.4f}  "
          f"{b.n_params():>14,} {nb:>8.4f}  "
          f"{('yes' if nb < na else 'no'):>13}")

print("\nEq. 66.9 predicts the direction: tying's benefit is the parameter")
print("saving, which is large when embeddings are most of the model and")
print("small when they are not. The 'embed fraction' column is that")
print("quantity and the last column is the outcome.")
print("\nThe cost of tying does not shrink with scale. One matrix must serve")
print("as the input map, where the residual stream is small, and as the")
print("output map, where the logits need range — section 5.3's scale")
print("coupling.")

# --- and the output scale fix -----------------------------------------------
print("\n" + "=" * 72)
print("the output scaling factor that tied models need (section 5.4 note)")
print("=" * 72)
print("Under tying, one matrix works at two scales. Scaling the logits")
print("decouples them partially.\n")
print(f"{'output scale':>14} {'tied NLL, d=32':>17}")
for sc in (0.5, 1.0, 2.0, 4.0, 8.0):
    net = train(TinyLM(d=32, tied=True, seed=7, out_scale=sc), Xtr, Ytr)
    print(f"{sc:>14.1f} {evaluate(net, Xte, Yte):>17.4f}")

print("\nIf the scale matters, the row spread is the scale-coupling problem")
print("measured. Vaswani et al. multiply the embedding by sqrt(d) for")
print("exactly this reason, and it is one of the details that gets dropped")
print("when people reimplement from a diagram.")

# --- section 6.5: the logit lens --------------------------------------------
print("\n" + "=" * 72)
print("the logit lens (eq. 66.11)")
print("=" * 72)
net = train(TinyLM(d=32, tied=False, seed=7), Xtr, Ytr)
net.forward(Xte[:2000], keep_layers=True)
U = net.unemb()

print("Apply the unembedding to the hidden state at each depth. The")
print("residual stream is a sum of block outputs, so every intermediate")
print("state lives in the same space and this is type-correct.\n")
print(f"{'read from':<16} {'NLL':>9} {'top-1 acc':>11} {'mean max prob':>15} "
      f"{'entropy':>9}")
for name, H in net.layers.items():
    Z = H @ U.T * net.out_scale + net.bo
    P = softmax(Z)
    tgt = Yte[:2000]
    nll = float(-np.log(np.clip(
        np.take_along_axis(P, tgt[..., None], -1), 1e-12, None)).mean())
    acc = float((P.argmax(-1) == tgt).mean())
    ent = float(-(P * np.log(P + 1e-12)).sum(-1).mean())
    print(f"{name:<16} {nll:>9.4f} {acc:>11.4f} {P.max(-1).mean():>15.4f} "
          f"{ent:>9.4f}")
print(f"\n(uniform entropy is ln {V} = {np.log(V):.3f})")

print("\nThe prediction sharpens with depth — that is the observation the")
print("logit lens is built on, and it is genuinely informative: it says the")
print("residual stream is being progressively shaped into something the")
print("unembedding can read.")
print("\nTwo cautions from section 6.5. The intermediate states have not")
print("been through the final normalisation, so their SCALE is wrong and")
print("the distributions are badly calibrated — read the ordering, not the")
print("probabilities. And the lens assumes the final unembedding is the")
print("right readout at every depth, which is an assumption; fitting a")
print("per-layer readout instead gives a different and usually sharper")
print("picture.")

# --- the embedding as a diagnostic ------------------------------------------
print("\n" + "=" * 72)
print("the embedding table is directly inspectable (section 7.5)")
print("=" * 72)
E = net.E
En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
sim = En @ En.T
np.fill_diagonal(sim, -np.inf)
print("nearest neighbour of each of the first 8 tokens, by cosine:\n")
print(f"{'token':>7} {'nearest':>9} {'cosine':>9}   "
      f"{'unembedding nearest':>21} {'cosine':>9}")
Un = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)
simU = Un @ Un.T
np.fill_diagonal(simU, -np.inf)
for t in range(8):
    j, ju = int(sim[t].argmax()), int(simU[t].argmax())
    print(f"{t:>7} {j:>9} {sim[t, j]:>9.4f}   {ju:>21} {simU[t, ju]:>9.4f}")

agree = float((sim.argmax(1) == simU.argmax(1)).mean())
print(f"\nfraction of tokens whose nearest neighbour is the SAME in both "
      f"tables: {agree:.4f}")
print(f"mean |cos(E_t, U_t)| for the same token: "
      f"{float(np.abs((En * Un).sum(1)).mean()):.4f}")

print("\nThat last number is the untied model's answer to the tying")
print("question, measured rather than argued: if the two matrices were")
print("doing the same job, each token's input and output vectors would")
print("align. How much they do is what the number says, and it is the")
print("cheapest evidence available about whether tying is throwing")
print("something away.")

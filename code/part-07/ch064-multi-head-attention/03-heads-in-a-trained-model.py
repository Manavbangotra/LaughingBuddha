# -*- coding: utf-8 -*-
# Extracted from: Chapter 64 — Multi-Head Attention
# Source: src/.../ch064-multi-head-attention.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Head specialisation, the attention sink, and the KV-cache arithmetic that
decided the architecture.
"""
import numpy as np

rng = np.random.default_rng(6)


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


# --- a task with several distinct relationships -----------------------------
V, T = 16, 12


def make_task(n, seed):
    """Three relationships in one sequence, so a good model needs at least
    three distinct attention patterns:
      - the answer depends on the PREVIOUS token (a local relationship)
      - and on the FIRST token (a global one)
      - and on the token that MATCHES the last one (a content-based one)
    """
    rs = np.random.default_rng(seed)
    X = rs.integers(1, V, (n, T))
    prev = X[:, -2]
    first = X[:, 0]
    last = X[:, -1]
    match = np.zeros(n, dtype=int)
    for i in range(n):
        hits = np.where(X[i, :-1] == last[i])[0]
        match[i] = X[i, hits[0] + 1] if len(hits) else 0
    y = (prev + first + match) % V
    return X, y


class TinyTransformerBlock:
    """One multi-head attention block plus a readout. Enough to see heads
    specialise, small enough to train in NumPy."""

    def __init__(self, d=48, h=3, seed=0):
        rs = np.random.default_rng(seed)
        self.E = rs.normal(0, 0.3, (V, d))
        self.P = rs.normal(0, 0.3, (T, d))
        s = 1 / np.sqrt(d)
        self.Wq = rs.normal(0, s, (d, d))
        self.Wk = rs.normal(0, s, (d, d))
        self.Wv = rs.normal(0, s, (d, d))
        self.Wo = rs.normal(0, s, (d, d))
        self.Wr = rs.normal(0, s, (d, V))
        self.br = np.zeros(V)
        self.d, self.h, self.dk = d, h, d // h

    def params(self):
        return [self.E, self.P, self.Wq, self.Wk, self.Wv, self.Wo,
                self.Wr, self.br]

    def forward(self, X, keep=False):
        n = len(X)
        self.X = X
        H = self.E[X] + self.P[None, :, :]           # (n, T, d)
        self.H = H
        Q = (H @ self.Wq).reshape(n, T, self.h, self.dk).transpose(0, 2, 1, 3)
        K = (H @ self.Wk).reshape(n, T, self.h, self.dk).transpose(0, 2, 1, 3)
        Vv = (H @ self.Wv).reshape(n, T, self.h, self.dk).transpose(0, 2, 1, 3)
        S = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.dk)
        A = softmax(S)
        self.A, self.Q, self.K, self.Vv = A, Q, K, Vv
        Hd = A @ Vv                                  # (n, h, T, dk)
        self.Hd = Hd
        merged = Hd.transpose(0, 2, 1, 3).reshape(n, T, self.d)
        self.merged = merged
        self.O = merged @ self.Wo
        self.read = self.O[:, -1, :]                 # read from the last pos
        return self.read @ self.Wr + self.br

    def grads(self, X, y):
        n = len(X)
        logits = self.forward(X)
        m_ = logits.max(axis=1, keepdims=True)
        e = np.exp(logits - m_)
        p = e / e.sum(axis=1, keepdims=True)
        loss = float(-np.log(np.clip(p[np.arange(n), y], 1e-12, None)).mean())
        dl = p.copy()
        dl[np.arange(n), y] -= 1.0
        dl /= n
        gWr, gbr = self.read.T @ dl, dl.sum(axis=0)
        dread = dl @ self.Wr.T                       # (n, d)
        dO = np.zeros_like(self.O)
        dO[:, -1, :] = dread
        gWo = self.merged.reshape(-1, self.d).T @ dO.reshape(-1, self.d)
        dmerged = dO @ self.Wo.T
        dHd = dmerged.reshape(n, T, self.h, self.dk).transpose(0, 2, 1, 3)
        dA = dHd @ self.Vv.transpose(0, 1, 3, 2)
        dV = self.A.transpose(0, 1, 3, 2) @ dHd
        dS = self.A * (dA - (dA * self.A).sum(axis=-1, keepdims=True))
        dS /= np.sqrt(self.dk)
        dQ = dS @ self.K
        dK = dS.transpose(0, 1, 3, 2) @ self.Q
        back = lambda G: G.transpose(0, 2, 1, 3).reshape(n, T, self.d)
        Hf = self.H.reshape(-1, self.d)
        gWq = Hf.T @ back(dQ).reshape(-1, self.d)
        gWk = Hf.T @ back(dK).reshape(-1, self.d)
        gWv = Hf.T @ back(dV).reshape(-1, self.d)
        dH = (back(dQ) @ self.Wq.T + back(dK) @ self.Wk.T
              + back(dV) @ self.Wv.T)
        gP = dH.sum(axis=0)
        gE = np.zeros_like(self.E)
        np.add.at(gE, X.reshape(-1), dH.reshape(-1, self.d))
        return loss, [gE, gP, gWq, gWk, gWv, gWo, gWr, gbr]


def train(net, X, y, steps=6000, lr=3e-3, batch=128, seed=0):
    ps = net.params()
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    rs = np.random.default_rng(seed + 11)
    for t in range(1, steps + 1):
        b = rs.integers(0, len(X), batch)
        _, gs = net.grads(X[b], y[b])
        for i, (p, g) in enumerate(zip(ps, gs)):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            p -= lr * (m[i] / (1 - 0.9 ** t)) / (
                np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    return net


Xtr, ytr = make_task(12000, 1)
Xte, yte = make_task(4000, 2)

print("=" * 72)
print("does head count matter on a task with several relationships?")
print("=" * 72)
print(f"The label depends on the previous token, the FIRST token, and a")
print(f"content match — three different relationships. Chance is "
      f"{1 / V:.4f}.\n")
print(f"{'heads':>7} {'d_k':>5} {'params':>9} {'test accuracy':>15}")
nets = {}
for h in (1, 2, 3, 6):
    net = train(TinyTransformerBlock(d=48, h=h, seed=3), Xtr, ytr)
    nets[h] = net
    acc = float((net.forward(Xte).argmax(1) == yte).mean())
    print(f"{h:>7} {48 // h:>5} {sum(p.size for p in net.params()):>9,} "
          f"{acc:>15.4f}")

print("\nEvery row has an identical parameter count — eq. 64.5 — so any")
print("difference is the head structure and nothing else.")

# --- what the heads attend to -----------------------------------------------
print("\n" + "=" * 72)
print("what the heads attend to")
print("=" * 72)
net = nets[3]
net.forward(Xte[:2000])
A = net.A                                            # (n, h, T, T)
last = A[:, :, -1, :]                                # queries from position T-1
print("Attention from the LAST position (where the readout happens),")
print(f"averaged over 2000 test sequences, for each of {net.h} heads:\n")
print(f"{'head':>5}  " + " ".join(f"{j:>5}" for j in range(T)))
for i in range(net.h):
    row = last[:, i, :].mean(axis=0)
    print(f"{i:>5}  " + " ".join(f"{a:>5.2f}" for a in row))
print(f"{'':>5}  " + " ".join(f"{'':>5}" for _ in range(T - 2))
      + f"{'prev':>5} {'self':>5}")
print(f"\nposition 0 is the FIRST token; position {T - 2} is the previous one")

ent = -(last * np.log(last + 1e-12)).sum(-1).mean(0)
print(f"\nper-head entropy (max is ln {T} = {np.log(T):.3f}):")
for i in range(net.h):
    print(f"  head {i}: {ent[i]:.3f}")

print("\nThe heads are not identical, which is the claim section 4.4 calls")
print("ESTABLISHED. Whether each row corresponds to one of the three")
print("relationships in the task is a much stronger claim, and this table")
print("cannot support it — a head can contribute to several, and the")
print("residual path carries information the attention map does not show.")

# --- the attention sink -----------------------------------------------------
print("\n" + "=" * 72)
print("the attention sink (section 6.4)")
print("=" * 72)
allq = A.mean(axis=(0, 2))                           # (h, T): avg over queries
print(f"attention mass on each key position, averaged over all queries:\n")
print(f"{'head':>5}  " + " ".join(f"{j:>5}" for j in range(T)))
for i in range(net.h):
    print(f"{i:>5}  " + " ".join(f"{a:>5.2f}" for a in allq[i]))
print(f"\nuniform would be {1 / T:.3f} everywhere")
print(f"mass on position 0, averaged over heads: {allq[:, 0].mean():.4f} "
      f"({allq[:, 0].mean() * T:.1f}x uniform)")

print("\nWhether a sink appears in a model this small and this briefly")
print("trained is not guaranteed, and the table above is the answer rather")
print("than a claim. In large trained models it is pronounced and reliable.")
print("\nThe mechanism section 6.4 gives is that a softmax MUST sum to one,")
print("so a head with nothing useful to attend to still has to put its mass")
print("somewhere, and a fixed low-information position is the cheapest")
print("place. The practical consequence does not depend on the")
print("explanation being right: do not evict token 0 from a KV cache.")

# --- section 6.5: the KV cache arithmetic -----------------------------------
print("\n" + "=" * 72)
print("the arithmetic that decided the architecture (eq. 64.10)")
print("=" * 72)


def kv_gb(L, g, dk, T, b=2):
    return 2 * b * L * g * dk * T / 1e9


print("A 70B-class model: L = 80 layers, h = 64 query heads, d_k = 128,")
print("bf16. Weights are about 140 GB.\n")
print(f"{'variant':<22} {'KV heads':>9} " +
      " ".join(f"{f'T={T}':>10}" for T in (2048, 8192, 32768))
      + f" {'vs MHA':>8}")
for label, g in (("MHA", 64), ("GQA g=8", 8), ("MQA", 1)):
    row = [kv_gb(80, g, 128, T) for T in (2048, 8192, 32768)]
    print(f"{label:<22} {g:>9} " + " ".join(f"{x:>9.2f}G" for x in row)
          + f" {64 / g:>7.0f}x")

print("\nThose numbers are PER SEQUENCE. Under full multi-head attention at")
print("a 32k context, one user's cache exceeds the model's own weights.")
print("\nNow the serving question. On a machine with 640 GB of memory, after")
print("140 GB of weights:\n")
print(f"{'variant':<22} " + " ".join(f"{f'T={T}':>18}"
                                     for T in (2048, 8192, 32768)))
for label, g in (("MHA", 64), ("GQA g=8", 8), ("MQA", 1)):
    row = [int((640 - 140) / kv_gb(80, g, 128, T)) for T in
           (2048, 8192, 32768)]
    print(f"{label:<22} " + " ".join(f"{f'{x} users':>18}" for x in row))

print("\nThat is the whole argument for grouped-query attention, and it is")
print("arithmetic rather than a modelling claim. GQA gives up a little")
print("quality (ainslie2023gqa measures it) and multiplies the number of")
print("users a machine can serve by the head-sharing ratio.")
print("\nNote what it does NOT change: the parameter count and the training")
print("FLOPs are barely affected, because the key and value projections are")
print("a quarter of the attention parameters and attention is a third of")
print("the model. The decision is made almost entirely on serving memory.")

# Extracted from: Chapter 62 — Why Recurrence Failed: The Road to Attention
# Source: src/.../ch062-why-attention.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Additive attention against a fixed bottleneck on a task that needs
alignment, and what the attention weights do and do not tell you.
"""
import numpy as np

rng = np.random.default_rng(4)

V, T_SRC = 12, 10


def make_lookup_task(n, seed):
    """A source sequence and a QUERY index. The target is the source token
    at that index. Solving it requires selecting one position — exactly the
    alignment problem attention was invented for."""
    rs = np.random.default_rng(seed)
    src = rs.integers(1, V, (n, T_SRC))
    idx = rs.integers(0, T_SRC, n)
    tgt = src[np.arange(n), idx]
    return src, idx, tgt


def onehot(X, k):
    out = np.zeros((*X.shape, k))
    np.put_along_axis(out, X[..., None], 1.0, axis=-1)
    return out


class Bottleneck:
    """Summarise the source into one vector, then answer using it + query."""

    def __init__(self, d=32, seed=0):
        rs = np.random.default_rng(seed)
        self.Wenc = rs.normal(0, np.sqrt(2 / (T_SRC * V)), (T_SRC * V, d))
        self.Wq = rs.normal(0, np.sqrt(2 / T_SRC), (T_SRC, d))
        self.Wo = rs.normal(0, np.sqrt(2 / (2 * d)), (2 * d, V))
        self.bo = np.zeros(V)
        self.d = d

    def params(self):
        return [self.Wenc, self.Wq, self.Wo, self.bo]

    def forward(self, src, idx):
        self.s1h = onehot(src, V).reshape(len(src), -1)
        self.q1h = onehot(idx, T_SRC)
        self.c = np.tanh(self.s1h @ self.Wenc)
        self.qe = np.tanh(self.q1h @ self.Wq)
        self.h = np.concatenate([self.c, self.qe], axis=1)
        return self.h @ self.Wo + self.bo

    def grads(self, src, idx, tgt):
        logits = self.forward(src, idx)
        m = logits.max(axis=1, keepdims=True)
        e = np.exp(logits - m)
        p = e / e.sum(axis=1, keepdims=True)
        loss = float(-np.log(np.clip(p[np.arange(len(tgt)), tgt],
                                     1e-12, None)).mean())
        d = p.copy()
        d[np.arange(len(tgt)), tgt] -= 1.0
        d /= len(tgt)
        gWo, gbo = self.h.T @ d, d.sum(axis=0)
        dh = d @ self.Wo.T
        dc = dh[:, :self.d] * (1 - self.c ** 2)
        dq = dh[:, self.d:] * (1 - self.qe ** 2)
        return loss, [self.s1h.T @ dc, self.q1h.T @ dq, gWo, gbo]


class Attention:
    """Additive attention (eqs. 62.2-62.3): the query scores every source
    position and reads a weighted average."""

    def __init__(self, d=32, seed=0):
        rs = np.random.default_rng(seed)
        self.Wv = rs.normal(0, np.sqrt(2 / V), (V, d))       # value per token
        # The key must carry POSITION, or no scoring function can locate a
        # position: keys built from token identity alone are the same
        # wherever the token sits. This is Chapter 65's point arriving early.
        self.Kp = rs.normal(0, 0.5, (T_SRC, d))              # key per position
        self.Wq = rs.normal(0, np.sqrt(2 / T_SRC), (T_SRC, d))
        self.Wa = rs.normal(0, np.sqrt(2 / d), (d, d))
        self.va = rs.normal(0, np.sqrt(2 / d), d)
        self.Wo = rs.normal(0, np.sqrt(2 / d), (d, V))
        self.bo = np.zeros(V)
        self.d = d

    def params(self):
        return [self.Wv, self.Kp, self.Wq, self.Wa, self.va, self.Wo, self.bo]

    def forward(self, src, idx, keep=False):
        n = len(src)
        S = onehot(src, V)                                   # (n, T, V)
        self.S = S
        self.Vv = S @ self.Wv                                # (n, T, d)
        self.K = np.broadcast_to(self.Kp, (n, T_SRC, self.d))
        self.q1h = onehot(idx, T_SRC)
        self.qe = np.tanh(self.q1h @ self.Wq)                # (n, d)
        # eq. 62.2: additive score
        self.pre = np.tanh(self.K @ self.Wa + self.qe[:, None, :])
        self.e = self.pre @ self.va                          # (n, T)
        m = self.e.max(axis=1, keepdims=True)
        ex = np.exp(self.e - m)
        self.alpha = ex / ex.sum(axis=1, keepdims=True)      # eq. 62.3
        self.ctx = (self.alpha[:, :, None] * self.Vv).sum(axis=1)
        return self.ctx @ self.Wo + self.bo

    def grads(self, src, idx, tgt):
        logits = self.forward(src, idx)
        n = len(tgt)
        m = logits.max(axis=1, keepdims=True)
        e = np.exp(logits - m)
        p = e / e.sum(axis=1, keepdims=True)
        loss = float(-np.log(np.clip(p[np.arange(n), tgt], 1e-12, None)).mean())
        d = p.copy()
        d[np.arange(n), tgt] -= 1.0
        d /= n
        gWo, gbo = self.ctx.T @ d, d.sum(axis=0)
        dctx = d @ self.Wo.T                                 # (n, d)
        gWv = np.einsum('ntv,nd,nt->vd', self.S, dctx, self.alpha)
        dalpha = np.einsum('nd,ntd->nt', dctx, self.Vv)
        de = self.alpha * (dalpha - (dalpha * self.alpha).sum(
            axis=1, keepdims=True))                          # softmax backward
        dpre = de[:, :, None] * self.va * (1 - self.pre ** 2)
        gva = np.einsum('nt,ntd->d', de, self.pre)
        gWa = np.einsum('ntd,nte->de', self.K, dpre)
        dK = dpre @ self.Wa.T
        gKp = dK.sum(axis=0)
        dqe = dpre.sum(axis=1) * 1.0
        dq = dqe * (1 - self.qe ** 2)
        gWq = self.q1h.T @ dq
        return loss, [gWv, gKp, gWq, gWa, gva, gWo, gbo]


def train(net, src, idx, tgt, steps=4000, lr=5e-3, batch=64, seed=0):
    ps = net.params()
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    rs = np.random.default_rng(seed + 7)
    for t in range(1, steps + 1):
        b = rs.integers(0, len(src), batch)
        _, gs = net.grads(src[b], idx[b], tgt[b])
        for i, (p, g) in enumerate(zip(ps, gs)):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            p -= lr * (m[i] / (1 - 0.9 ** t)) / (
                np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    return net


src_tr, idx_tr, tgt_tr = make_lookup_task(8000, 1)
src_te, idx_te, tgt_te = make_lookup_task(4000, 2)

print("=" * 72)
print("a task that needs ALIGNMENT: read the source token at a given index")
print("=" * 72)
print(f"source length {T_SRC}, vocabulary {V}; chance is {1 / V:.4f}\n")
print(f"{'model':<28} {'params':>9} {'test accuracy':>15}")
bn = train(Bottleneck(seed=3), src_tr, idx_tr, tgt_tr)
acc_b = float((bn.forward(src_te, idx_te).argmax(1) == tgt_te).mean())
print(f"{'fixed bottleneck, d=32':<28} "
      f"{sum(p.size for p in bn.params()):>9,} {acc_b:>15.4f}")
at = train(Attention(seed=3), src_tr, idx_tr, tgt_tr)
acc_a = float((at.forward(src_te, idx_te).argmax(1) == tgt_te).mean())
print(f"{'additive attention, d=32':<28} "
      f"{sum(p.size for p in at.params()):>9,} {acc_a:>15.4f}")

print("\nThe task is pure alignment: the answer is one source token and the")
print("query says which. Attention's mechanism — score every position, read")
print("the winner — matches it exactly. The bottleneck model has to encode")
print("the whole source into 32 numbers and then extract the right one.")
print("\nNote one design detail that the first version of this experiment")
print("got wrong. The KEYS must carry position: keys built from token")
print("identity alone are identical wherever the token sits, so no scoring")
print("function can locate a position and the model sits at a uniform")
print("attention distribution however long it trains. That is Chapter 65's")
print("subject arriving three chapters early, and it is a good illustration")
print("that attention is a lookup — and a lookup needs addressable keys.")

# --- what the weights look like ---------------------------------------------
print("\n" + "=" * 72)
print("the attention weights: what they show")
print("=" * 72)
at.forward(src_te[:6], idx_te[:6])
print("Each row is one example. The '^' marks the queried index.\n")
for i in range(6):
    bar = " ".join(f"{a:>5.2f}" for a in at.alpha[i])
    mark = " ".join("    ^" if j == idx_te[i] else "     "
                    for j in range(T_SRC))
    print(f"  query {idx_te[i]:>2}  {bar}")
    print(f"           {mark}")
print(f"\nmass on the queried position, averaged over the test set: "
      f"{float(at.alpha[np.arange(len(at.alpha)), idx_te[:6]].mean()):.4f}")
at.forward(src_te, idx_te)
print(f"over all {len(src_te)} test examples: "
      f"{float(at.alpha[np.arange(len(src_te)), idx_te].mean()):.4f}")
print(f"mean attention entropy (max is ln {T_SRC} = "
      f"{np.log(T_SRC):.3f}): "
      f"{float(-(at.alpha * np.log(at.alpha + 1e-12)).sum(1).mean()):.3f}")

print("\nThis is the inspectability Bahdanau et al. reported and it is real:")
print("the weights concentrate on the position the task requires, and you")
print("can read that off the matrix without any further machinery.")

# --- section 6.4: what they do NOT show -------------------------------------
print("\n" + "=" * 72)
print("what attention weights do NOT tell you (section 6.4)")
print("=" * 72)
print("Claim: if two positions hold the SAME value vector, any split of the")
print("weight between them gives an identical output. The weights are then")
print("not identifiable from the function.\n")

at.forward(src_te[:400], idx_te[:400])
alpha0 = at.alpha.copy()
out0 = at.ctx.copy()
# find, per example, pairs of positions holding the same TOKEN
dup_moved, checked = 0, 0
alpha_mod = alpha0.copy()
for i in range(400):
    toks = src_te[i]
    for j in range(T_SRC):
        for k in range(j + 1, T_SRC):
            if toks[j] == toks[k]:
                tot = alpha_mod[i, j] + alpha_mod[i, k]
                alpha_mod[i, j], alpha_mod[i, k] = tot, 0.0   # move it all
                dup_moved += 1
                break
        else:
            continue
        break
    checked += 1
ctx_mod = (alpha_mod[:, :, None] * at.Vv).sum(axis=1)
print(f"examples containing a repeated token : {dup_moved} of {checked}")
print(f"max change in the attention weights  : "
      f"{np.abs(alpha_mod - alpha0).max():.4f}")
print(f"max change in the CONTEXT VECTOR     : "
      f"{np.abs(ctx_mod - out0).max():.3e}")

print("\nThe weights were changed by up to a full unit of probability mass —")
print("moved wholesale from one position to another — and the output did")
print("not move at all. Both positions held the same token, so they hold")
print("the same value vector, and a weighted average cannot distinguish")
print("them.")
print("\nSo an attention map is a HYPOTHESIS about where information came")
print("from, not a measurement of it. Here there are two maps producing")
print("bit-identical outputs, and nothing in the model prefers either.")
print("\nThat is the mildest version of the caution. Section 6.4 lists two")
print("more: a high-entropy head may be computing a useful average rather")
print("than doing nothing, and the residual stream carries information past")
print("the attention block entirely (Chapter 67), so a position can matter")
print("with zero weight on it.")
print("\nAttention maps are cheap to look at and worth looking at. Treat")
print("what they suggest as something to test, not as something shown.")

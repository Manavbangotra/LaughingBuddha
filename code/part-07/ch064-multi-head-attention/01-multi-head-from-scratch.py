# Extracted from: Chapter 64 — Multi-Head Attention
# Source: src/.../ch064-multi-head-attention.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Multi-head attention: the shapes, the parameter accounting, and the
transpose that silently breaks it.
"""
import numpy as np

rng = np.random.default_rng(0)


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


class MultiHeadAttention:
    """Eqs. 64.1-64.3, written with the heads as a batch dimension."""

    def __init__(self, d, h, seed=0):
        assert d % h == 0, "d must be divisible by h"
        rs = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(d)
        self.Wq = rs.normal(0, s, (d, d))
        self.Wk = rs.normal(0, s, (d, d))
        self.Wv = rs.normal(0, s, (d, d))
        self.Wo = rs.normal(0, s, (d, d))
        self.d, self.h, self.dk = d, h, d // h

    def n_params(self):
        return 4 * self.d * self.d

    def _split(self, X):
        B, T, _ = X.shape
        return X.reshape(B, T, self.h, self.dk).transpose(0, 2, 1, 3)

    def _merge(self, X):
        B, h, T, dk = X.shape
        return X.transpose(0, 2, 1, 3).reshape(B, T, h * dk)

    def forward(self, X, mask=None, keep=False):
        Q = self._split(X @ self.Wq)                  # (B, h, T, dk)
        K = self._split(X @ self.Wk)
        V = self._split(X @ self.Wv)
        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.dk)
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        A = softmax(scores)                           # (B, h, T, T)
        out = self._merge(A @ V) @ self.Wo
        if keep:
            self.A = A
        return out


print("=" * 72)
print("multi-head attention costs the SAME as single-head (eqs. 64.5-64.6)")
print("=" * 72)
d, T, B = 512, 64, 2
X = rng.normal(size=(B, T, d))
print(f"model width d = {d}, sequence length T = {T}\n")
print(f"{'heads h':>9} {'d_k':>6} {'parameters':>12} {'proj MFLOPs':>13} "
      f"{'score MFLOPs':>14} {'output shape':>16}")
outs = {}
for h in (1, 2, 4, 8, 16, 64):
    m = MultiHeadAttention(d, h, seed=1)
    out = m.forward(X)
    outs[h] = out
    proj = 8 * T * d * d
    score = 4 * T * T * d
    print(f"{h:>9} {d // h:>6} {m.n_params():>12,} {proj / 1e6:>13.2f} "
          f"{score / 1e6:>14.2f} {str(out.shape):>16}")

print("\nEvery row is identical in parameters and in FLOPs. The h per-head")
print("matrices of shape (d, d_k) concatenate into one (d, d) matrix, and")
print("the h score matrices are each T-by-T-by-d_k with h*d_k = d.")
print("\nSo the number of heads is FREE in both. That is the fact people")
print("most often get wrong about multi-head attention: it is not h copies")
print("of anything, it is a partition of dimensions that already existed.")

# --- section 5.3: the transpose that silently breaks it ---------------------
print("\n" + "=" * 72)
print("the reshape that fails silently (section 5.3)")
print("=" * 72)
m = MultiHeadAttention(d, 8, seed=1)
Q = m._split(X @ m.Wq)
K = m._split(X @ m.Wk)
V = m._split(X @ m.Wv)
A = softmax(Q @ K.transpose(0, 1, 3, 2) / np.sqrt(m.dk))
heads = A @ V                                       # (B, h, T, dk)

correct = heads.transpose(0, 2, 1, 3).reshape(B, T, d)
wrong = heads.reshape(B, T, d)                      # NO transpose

print(f"head output tensor            : {heads.shape}")
print(f"correct merge (transpose then reshape) : {correct.shape}")
print(f"wrong merge   (reshape only)           : {wrong.shape}")
print(f"\nSAME SHAPE, so nothing raises. Are they the same values?")
print(f"  max |correct - wrong| = {np.abs(correct - wrong).max():.4f}")
print(f"  fraction of entries that differ = "
      f"{float((np.abs(correct - wrong) > 1e-12).mean()):.4f}")

print("\nThe wrong version interleaves the head axis with the position axis,")
print("so token t's output vector is assembled from other tokens' head")
print("outputs. It has the right shape, the right dtype, and it trains — to")
print("a worse loss, for no visible reason.")
print("\nThis is Chapter 51's silent-broadcast lesson in a new costume, and")
print("the remedy is the same: assert the shape AND check a known-good")
print("value. A single-head model is the check — at h = 1 the transpose is")
print("a no-op, so correct and wrong must agree exactly:")
m1 = MultiHeadAttention(d, 1, seed=1)
Q1, K1, V1 = m1._split(X @ m1.Wq), m1._split(X @ m1.Wk), m1._split(X @ m1.Wv)
h1 = softmax(Q1 @ K1.transpose(0, 1, 3, 2) / np.sqrt(m1.dk)) @ V1
print(f"  h=1: max |transposed - not| = "
      f"{np.abs(h1.transpose(0, 2, 1, 3).reshape(B, T, d) - h1.reshape(B, T, d)).max():.3e}")

# --- section 6.2: MHA is a SUM over heads -----------------------------------
print("\n" + "=" * 72)
print("multi-head attention is a SUM of per-head terms (eq. 64.8)")
print("=" * 72)
h = 8
m = MultiHeadAttention(d, h, seed=1)
full = m.forward(X)

Q = m._split(X @ m.Wq)
K = m._split(X @ m.Wk)
V = m._split(X @ m.Wv)
A = softmax(Q @ K.transpose(0, 1, 3, 2) / np.sqrt(m.dk))
per_head = A @ V                                     # (B, h, T, dk)

total = np.zeros_like(full)
contrib = []
for i in range(h):
    Wo_i = m.Wo[i * m.dk:(i + 1) * m.dk, :]          # rows for head i
    term = per_head[:, i] @ Wo_i                     # (B, T, d)
    total += term
    contrib.append(float(np.sqrt((term ** 2).mean())))

print(f"max |sum of per-head terms  -  full MHA output| = "
      f"{np.abs(total - full).max():.3e}")
print("\nExact to floating point. Eq. 64.8 says W_O partitions by rows into")
print("h blocks, one per head, so each head reads from the residual stream")
print("and writes back to it independently and the block's output is the")
print("SUM of what they wrote.")
print("\nThat is what makes head-level analysis coherent: a head is a")
print("self-contained circuit, not a slice of an entangled computation. It")
print("is also why pruning a head is well defined — you delete one term.\n")
print("RMS contribution of each head to the output:")
for i, c in enumerate(contrib):
    bar = "#" * int(40 * c / max(contrib))
    print(f"  head {i}  {c:>8.4f}  {bar}")
print("\n(untrained, so these differ only by the random draw — the point is")
print(" that the decomposition EXISTS, not what it says about this model)")

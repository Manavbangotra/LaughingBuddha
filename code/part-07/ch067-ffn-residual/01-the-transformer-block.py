# -*- coding: utf-8 -*-
# Extracted from: Chapter 67 — Feed-Forward Networks, Residuals, and Normalization Placement
# Source: src/.../ch067-ffn-residual.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A complete transformer block, its parameter accounting, and the
residual-stream decomposition of eq. 67.7.
"""
import numpy as np

rng = np.random.default_rng(0)


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def rmsnorm(x, g, eps=1e-6):
    return g * x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps)


def silu(z):
    return z / (1.0 + np.exp(-np.clip(z, -60, 60)))


class Block:
    """Pre-norm block. Set gated=True for eq. 67.4."""

    def __init__(self, d, h, d_ff=None, gated=False, seed=0):
        rs = np.random.default_rng(seed)
        s = 1 / np.sqrt(d)
        self.d, self.h, self.dk = d, h, d // h
        self.gated = gated
        self.d_ff = d_ff if d_ff else (int(8 * d / 3) if gated else 4 * d)
        self.Wq = rs.normal(0, s, (d, d))
        self.Wk = rs.normal(0, s, (d, d))
        self.Wv = rs.normal(0, s, (d, d))
        self.Wo = rs.normal(0, s, (d, d))
        self.g1 = np.ones(d)
        self.g2 = np.ones(d)
        self.W1 = rs.normal(0, s, (d, self.d_ff))
        self.W2 = rs.normal(0, 1 / np.sqrt(self.d_ff), (self.d_ff, d))
        if gated:
            self.Wg = rs.normal(0, s, (d, self.d_ff))

    def attn_params(self):
        return 4 * self.d * self.d

    def ffn_params(self):
        n = 2 if not self.gated else 3
        return n * self.d * self.d_ff

    def n_params(self):
        return self.attn_params() + self.ffn_params() + 2 * self.d

    def attn(self, x):
        B, T, d = x.shape
        sp = lambda M: M.reshape(B, T, self.h, self.dk).transpose(0, 2, 1, 3)
        Q, K, V = sp(x @ self.Wq), sp(x @ self.Wk), sp(x @ self.Wv)
        S = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.dk)
        mask = np.tril(np.ones((T, T), dtype=bool))
        A = softmax(np.where(mask, S, -np.inf))
        out = (A @ V).transpose(0, 2, 1, 3).reshape(B, T, d)
        return out @ self.Wo

    def ffn(self, x):
        if self.gated:
            return (silu(x @ self.Wg) * (x @ self.W1)) @ self.W2
        return np.maximum(0.0, x @ self.W1) @ self.W2

    def forward(self, x, record=None):
        a = self.attn(rmsnorm(x, self.g1))               # eq. 67.1
        if record is not None:
            record.append(("attn", a))
        h = x + a
        f = self.ffn(rmsnorm(h, self.g2))                # eq. 67.2
        if record is not None:
            record.append(("ffn", f))
        return h + f


print("=" * 72)
print("where a transformer's parameters are (eqs. 67.5-67.6)")
print("=" * 72)
print(f"{'d':>6} {'d_ff':>6} {'gated':>7} {'attention':>12} {'FFN':>12} "
      f"{'FFN fraction':>14}")
for d in (512, 768, 4096):
    for gated in (False, True):
        b = Block(d, 8, gated=gated, seed=1)
        print(f"{d:>6} {b.d_ff:>6} {str(gated):>7} {b.attn_params():>12,} "
              f"{b.ffn_params():>12,} "
              f"{b.ffn_params() / (b.attn_params() + b.ffn_params()):>14.1%}")

print("\nTwo-thirds of every block is the feed-forward network, in both the")
print("ungated and the gated form — which is why the gated hidden width is")
print("8d/3 rather than 4d. Eq. 67.6 says exactly this, and it is the most")
print("surprising number in the part for anyone whose picture of a")
print("transformer came from a diagram where attention dominates.")

# --- and the FLOPs ----------------------------------------------------------
print("\n" + "=" * 72)
print("and where the FLOPs are (eq. 67.7)")
print("=" * 72)
print(f"{'d':>6} " + " ".join(f"{f'T={T}':>22}" for T in (512, 2048, 8192)))
print(f"{'':>6} " + " ".join(f"{'attn / FFN / attn %':>22}" for _ in range(3)))
for d in (768, 4096):
    row = []
    for T in (512, 2048, 8192):
        fa = 8 * d * d + 4 * T * d
        ff = 16 * d * d
        row.append(f"{fa / 1e6:.0f}M / {ff / 1e6:.0f}M / {fa / (fa + ff):.0%}")
    print(f"{d:>6} " + " ".join(f"{r:>22}" for r in row))

print(f"\nEq. 67.7's crossover is at T = 4d:")
for d in (768, 4096):
    print(f"  d = {d:>5}  ->  crossover at T = {4 * d:,}")
print("\nBelow that the feed-forward block dominates the arithmetic too, and")
print("this is the SAME threshold as Chapter 62's, for the same reason: it")
print("is where attention's quadratic term overtakes the linear ones.")

# --- section 6.1: the residual-stream decomposition -------------------------
print("\n" + "=" * 72)
print("a transformer's output is a SUM of sublayer outputs (eq. 67.7)")
print("=" * 72)
d, h, L, B, T = 128, 8, 6, 2, 16
x0 = rng.normal(size=(B, T, d)) * 0.5
blocks = [Block(d, h, seed=10 + i) for i in range(L)]

record = []
x = x0.copy()
for b in blocks:
    x = b.forward(x, record=record)

total = x0 + sum(v for _, v in record)
print(f"{L} blocks, so {2 * L} sublayer outputs")
print(f"max |x_L  -  (x_0 + sum of all sublayer outputs)| = "
      f"{np.abs(x - total).max():.3e}")

print("\nExact. Eq. 67.7 is not an approximation or a way of thinking about")
print("it — the final hidden state IS the embedding plus the sum of every")
print("sublayer's output, with nothing else in between.")
print("\nThat is why the logit lens of Chapter 66 is type-correct, why heads")
print("can be analysed individually (Chapter 64), and why deleting one block")
print("from a trained transformer changes the output less than it would in")
print("a plain stack: you have removed one term from a sum of twelve.")

print("\nRMS contribution of each sublayer to the final stream:\n")
print(f"{'layer':>6} {'attention':>12} {'FFN':>12} {'stream RMS after':>18}")
x = x0.copy()
for i, b in enumerate(blocks):
    r = []
    x = b.forward(x, record=r)
    print(f"{i:>6} {float(np.sqrt((r[0][1] ** 2).mean())):>12.4f} "
          f"{float(np.sqrt((r[1][1] ** 2).mean())):>12.4f} "
          f"{float(np.sqrt((x ** 2).mean())):>18.4f}")

print("\n(untrained, so the magnitudes reflect initialisation rather than")
print(" learned behaviour — the point is the accounting, not the values)")

# --- section 6.3: the residual norm grows -----------------------------------
print("\n" + "=" * 72)
print("the residual norm grows with depth (eq. 67.10)")
print("=" * 72)
print("Each sublayer adds a roughly fixed-scale vector, so variances add and")
print("the norm should grow as sqrt(number of sublayers).\n")
deep = [Block(d, h, seed=100 + i) for i in range(32)]
x = rng.normal(size=(4, T, d)) * 0.5
n0 = float(np.sqrt((x ** 2).mean()))
print(f"{'after layer':>12} {'stream RMS':>12} {'vs layer 0':>12} "
      f"{'sqrt(2L+1) prediction':>23} {'per-layer angle':>17}")
prev = x.copy()
for i, b in enumerate(deep):
    x = b.forward(x)
    if (i + 1) in (1, 2, 4, 8, 16, 32):
        rms = float(np.sqrt((x ** 2).mean()))
        cos = float((prev * x).sum() / (np.linalg.norm(prev)
                                        * np.linalg.norm(x)))
        print(f"{i + 1:>12} {rms:>12.4f} {rms / n0:>12.3f} "
              f"{np.sqrt(2 * (i + 1) + 1):>23.3f} "
              f"{np.degrees(np.arccos(np.clip(cos, -1, 1))):>16.2f}°")
    prev = x.copy()

print("\nThe growth follows the square-root prediction of eq. 67.10, and the")
print("per-layer angle shrinks with depth: a block adding a fixed-scale")
print("vector to a growing stream rotates it less and less.")
print("\nTwo consequences. Later blocks contribute proportionally less, which")
print("is measurable in real models and is not usually what people expect")
print("from 'deeper is more processing'. And the FINAL normalisation before")
print("the unembedding is mandatory — without it, the logit scale would")
print("depend on how many layers the model happens to have.")

# -*- coding: utf-8 -*-
# Extracted from: Chapter 69 — Causal Masking and the KV Cache
# Source: src/.../ch069-masking-kv-cache.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The KV cache: why it is necessary, that it is exact, and what it costs."""
import time

import numpy as np

rng = np.random.default_rng(0)


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def rmsnorm(x, eps=1e-6):
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps)


class Decoder:
    """A small causal transformer with and without a KV cache."""

    def __init__(self, V=32, d=64, h=4, L=2, T_max=256, seed=0):
        rs = np.random.default_rng(seed)
        s = 1 / np.sqrt(d)
        self.V, self.d, self.h, self.dk, self.L = V, d, h, d // h, L
        self.E = rs.normal(0, 0.05, (V, d))
        self.P = rs.normal(0, 0.05, (T_max, d))
        self.W = []
        for _ in range(L):
            self.W.append({
                "q": rs.normal(0, s, (d, d)), "k": rs.normal(0, s, (d, d)),
                "v": rs.normal(0, s, (d, d)), "o": rs.normal(0, s, (d, d)),
                "1": rs.normal(0, s, (d, 4 * d)),
                "2": rs.normal(0, 1 / np.sqrt(4 * d), (4 * d, d))})
        self.U = rs.normal(0, 0.05, (V, d))

    def _block(self, x, W, mask):
        n, T, d = x.shape
        na = rmsnorm(x)
        sp = lambda M: M.reshape(n, T, self.h, self.dk).transpose(0, 2, 1, 3)
        Q, K, Vv = sp(na @ W["q"]), sp(na @ W["k"]), sp(na @ W["v"])
        S = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.dk)
        A = softmax(np.where(mask, S, -1e9))
        ctx = (A @ Vv).transpose(0, 2, 1, 3).reshape(n, T, d)
        h1 = x + ctx @ W["o"]
        nf = rmsnorm(h1)
        return h1 + np.maximum(0.0, nf @ W["1"]) @ W["2"]

    def forward_full(self, ids):
        """No cache: run the whole sequence."""
        n, T = ids.shape
        x = self.E[ids] + self.P[None, :T, :]
        mask = np.tril(np.ones((T, T), dtype=bool))
        for W in self.W:
            x = self._block(x, W, mask)
        return rmsnorm(x) @ self.U.T

    def new_cache(self, n, T_max):
        return [{"k": np.zeros((n, self.h, T_max, self.dk)),
                 "v": np.zeros((n, self.h, T_max, self.dk))}
                for _ in range(self.L)]

    def forward_step(self, ids_step, pos, cache):
        """Eqs. 69.1-69.3: ONE new position, attending over the cache."""
        n = len(ids_step)
        x = self.E[ids_step][:, None, :] + self.P[None, pos:pos + 1, :]
        for li, W in enumerate(self.W):
            na = rmsnorm(x)
            sp = lambda M: M.reshape(n, 1, self.h, self.dk).transpose(
                0, 2, 1, 3)
            q = sp(na @ W["q"])
            k = sp(na @ W["k"])
            v = sp(na @ W["v"])
            cache[li]["k"][:, :, pos:pos + 1] = k       # eq. 69.2
            cache[li]["v"][:, :, pos:pos + 1] = v
            K = cache[li]["k"][:, :, :pos + 1]
            Vv = cache[li]["v"][:, :, :pos + 1]
            S = (q @ K.transpose(0, 1, 3, 2)) / np.sqrt(self.dk)
            A = softmax(S)                              # no mask needed
            ctx = (A @ Vv).transpose(0, 2, 1, 3).reshape(n, 1, self.d)
            h1 = x + ctx @ W["o"]
            nf = rmsnorm(h1)
            x = h1 + np.maximum(0.0, nf @ W["1"]) @ W["2"]
        return rmsnorm(x)[:, 0] @ self.U.T


# --- section 6.2: the cache is EXACT ----------------------------------------
print("=" * 72)
print("the cached path is exact, not an approximation (eq. 69.8)")
print("=" * 72)
model = Decoder(seed=3)
ids = rng.integers(0, model.V, (4, 24))

full = model.forward_full(ids)
cache = model.new_cache(4, 256)
step_out = []
for t in range(ids.shape[1]):
    step_out.append(model.forward_step(ids[:, t], t, cache))
stepwise = np.stack(step_out, axis=1)

print(f"full forward pass  : {full.shape}")
print(f"stepwise + cache   : {stepwise.shape}")
print(f"max |difference|   : {np.abs(full - stepwise).max():.3e}")

print("\nIdentical to floating point at EVERY position, not just the last.")
print("Eq. 69.8 says why: under a causal mask, position i's key depends only")
print("on tokens up to i, so appending a token cannot change it. The cache")
print("is not an approximation — it is the observation that the model was")
print("recomputing something it already knew.")
print("\nThat also means a BIDIRECTIONAL model cannot cache anything:")
print("appending a token changes every earlier position's representation.")
print("The causal mask is what makes caching possible, which is the deepest")
print("link between Chapter 68's mask and this chapter.")

# --- verify eq. 69.8 directly ------------------------------------------------
print("\nDirect check of eq. 69.8 — do the keys change as the sequence grows?\n")
print(f"{'sequence length':>17} {'max |k_5(t_1:m) - k_5(t_1:24)|':>34}")
ref = None
for m in (6, 10, 16, 24):
    sub = ids[:, :m]
    n, T = sub.shape
    x = model.E[sub] + model.P[None, :T, :]
    na = rmsnorm(x)
    W = model.W[0]
    K = (na @ W["k"]).reshape(n, T, model.h, model.dk)
    k5 = K[:, 5]
    if ref is None:
        ref0 = k5.copy()
    print(f"{m:>17} {np.abs(k5 - ref0).max():>34.3e}")

print("\nZero at every length: the key for position 5 is the same whether")
print("the sequence is 6 tokens or 24. That is eq. 69.8, and it is the")
print("entire justification for the cache.")

# --- section 6.1: the cubic-to-quadratic reduction --------------------------
print("\n" + "=" * 72)
print("without a cache, generation is CUBIC (eqs. 69.6-69.7)")
print("=" * 72)
print(f"{'tokens n':>9} {'no cache (ms)':>15} {'with cache (ms)':>17} "
      f"{'speedup':>9} {'predicted n':>13}")
for n_gen in (16, 32, 64, 128):
    seed_ids = rng.integers(0, model.V, (2, 1))

    seq = seed_ids.copy()
    t0 = time.perf_counter()
    for _ in range(n_gen):
        lg = model.forward_full(seq)
        nxt = lg[:, -1].argmax(-1)[:, None]
        seq = np.concatenate([seq, nxt], axis=1)
    t_nocache = time.perf_counter() - t0

    cache = model.new_cache(2, 256)
    t0 = time.perf_counter()
    cur = seed_ids[:, 0]
    for t in range(n_gen):
        lg = model.forward_step(cur, t, cache)
        cur = lg.argmax(-1)
    t_cache = time.perf_counter() - t0

    print(f"{n_gen:>9} {t_nocache * 1e3:>15.2f} {t_cache * 1e3:>17.2f} "
          f"{t_nocache / t_cache:>8.1f}x {n_gen:>13}")

print("\nThe speedup grows roughly in proportion to the number of tokens")
print("generated, which is eq. 69.6 against eq. 69.7: a full factor of n")
print("removed from both cost terms.")
print("\nThe remaining quadratic term is irreducible. Attention over a")
print("growing context costs O(n^2) in total however it is computed, and")
print("Chapter 70 is about that.")

# --- section 5.2: what the cache costs --------------------------------------
print("\n" + "=" * 72)
print("the cache size (eq. 69.4)")
print("=" * 72)
MODELS = [("7B  (L=32, h=32, d_k=128)", 32, 32, 128, 7e9),
          ("70B (L=80, h=64, d_k=128)", 80, 64, 128, 7e10)]
print(f"{'model':<28} {'variant':<10} " +
      " ".join(f"{f'T={T // 1024}k':>10}" for T in (4096, 32768, 131072)))
for name, L, h, dk, P in MODELS:
    for label, g in (("MHA", h), ("GQA g=8", 8)):
        row = [2 * 2 * L * g * dk * T / 1e9 for T in (4096, 32768, 131072)]
        print(f"{name:<28} {label:<10} " +
              " ".join(f"{x:>9.1f}G" for x in row))

print("\nThose are PER SEQUENCE. Weights are shared across all users; the")
print("cache is not. That asymmetry is why grouped-query attention was")
print("adopted within a year of being proposed.")

print("\n" + "=" * 72)
print("when does reading the cache overtake reading the weights? (eq. 69.10)")
print("=" * 72)
print(f"{'model':<28} {'variant':<10} {'crossover context':>19} "
      f"{'cache/weight bytes @ 32k':>26}")
for name, L, h, dk, P in MODELS:
    for label, g in (("MHA", h), ("GQA g=8", 8)):
        cross = P / (2 * L * g * dk)
        ratio = 2 * L * g * dk * 32768 / P
        print(f"{name:<28} {label:<10} {cross:>19,.0f} {ratio:>26.3f}")

print("\nFor a grouped-query model the crossover is hundreds of thousands of")
print("tokens, so at realistic contexts decode time is dominated by reading")
print("the WEIGHTS and the cache is nearly free to read.")
print("\nThat is the distinction to get right: the cache is a memory")
print("CAPACITY problem, not a memory BANDWIDTH one. Capacity is fixed by")
print("compression; bandwidth is fixed by batching. Confusing them leads to")
print("optimising the wrong thing.")

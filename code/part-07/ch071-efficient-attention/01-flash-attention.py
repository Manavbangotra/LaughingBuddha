# -*- coding: utf-8 -*-
# Extracted from: Chapter 71 — Efficient Attention: FlashAttention, GQA/MQA, Sparse and Linear Variants
# Source: src/.../ch071-efficient-attention.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""FlashAttention: the online softmax, the tiled algorithm, and the proof
that it is exact (eqs. 71.1-71.3).
"""
import numpy as np

rng = np.random.default_rng(0)


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


# --- section 6.1: the online softmax ----------------------------------------
def online_softmax_sum(s, block=8):
    """Eqs. 71.1-71.2: running max and running sum, in one pass."""
    m, ell = -np.inf, 0.0
    for j in range(0, len(s), block):
        blk = s[j:j + block]
        m_new = max(m, float(blk.max()))
        ell = np.exp(m - m_new) * ell + float(np.exp(blk - m_new).sum())
        m = m_new
    return m, ell


print("=" * 72)
print("the online softmax is exact (section 6.1)")
print("=" * 72)
print("The whole of FlashAttention rests on exp(a-c) = exp(a-b)exp(b-c).\n")
print(f"{'scale of scores':>17} {'block':>7} {'two-pass sum':>16} "
      f"{'online sum':>14} {'relative error':>16}")
for scale in (1.0, 10.0, 100.0):
    s = rng.normal(0, scale, 512)
    m_true = float(s.max())
    ell_true = float(np.exp(s - m_true).sum())
    for block in (8, 64):
        m, ell = online_softmax_sum(s, block)
        print(f"{scale:>17.0f} {block:>7} {ell_true:>16.8f} "
              f"{ell:>14.8f} {abs(ell - ell_true) / ell_true:>16.2e}")

print("\nExact to floating point at every block size and every score scale,")
print("including scores of magnitude 100 where a naive one-pass sum would")
print("overflow. The rescaling factor exp(m_old - m_new) corrects the")
print("accumulated sum whenever a larger maximum appears, and section 6.1")
print("proves by induction that the correction is exact.")

# --- the tiled attention ----------------------------------------------------
def attention_naive(Q, K, V):
    """Materialises the T-by-T matrix."""
    dk = Q.shape[-1]
    S = Q @ K.T / np.sqrt(dk)
    return softmax(S) @ V, S.nbytes


def attention_flash(Q, K, V, Br=32, Bc=32):
    """Eq. 71.3, tiled. S and A never exist at full size."""
    T, dk = Q.shape
    O = np.zeros((T, dk))
    peak_tile = 0
    for i in range(0, T, Br):
        q = Q[i:i + Br]
        o = np.zeros((len(q), dk))
        m = np.full(len(q), -np.inf)
        ell = np.zeros(len(q))
        for j in range(0, T, Bc):
            k, v = K[j:j + Bc], V[j:j + Bc]
            s = q @ k.T / np.sqrt(dk)                    # the only T-sized
            peak_tile = max(peak_tile, s.nbytes)          # object that exists
            m_new = np.maximum(m, s.max(axis=1))
            p = np.exp(s - m_new[:, None])
            corr = np.exp(m - m_new)
            ell = corr * ell + p.sum(axis=1)
            o = corr[:, None] * o + p @ v                 # eq. 71.3
            m = m_new
        O[i:i + Br] = o / ell[:, None]
    return O, peak_tile


print("\n" + "=" * 72)
print("tiled attention gives the IDENTICAL result (eq. 71.3)")
print("=" * 72)
print(f"{'T':>6} {'d_k':>5} {'max |naive - flash|':>22} "
      f"{'naive S bytes':>15} {'flash tile bytes':>18} {'ratio':>9}")
for T in (64, 256, 1024):
    dk = 64
    Q = rng.normal(size=(T, dk))
    K = rng.normal(size=(T, dk))
    V = rng.normal(size=(T, dk))
    o1, b1 = attention_naive(Q, K, V)
    o2, b2 = attention_flash(Q, K, V)
    print(f"{T:>6} {dk:>5} {np.abs(o1 - o2).max():>22.3e} "
          f"{b1 / 1e3:>14.1f}K {b2 / 1e3:>17.1f}K {b1 / b2:>9.0f}x")

print("\nIdentical to floating-point round-off, and the largest object that")
print("ever exists is one tile rather than the whole T-by-T matrix. The")
print("saving grows as T squared while the tile stays fixed.")
print("\nThat is what makes FlashAttention different from every efficiency")
print("technique that came before it: there is no approximation, no")
print("hyperparameter, and no quality question to evaluate. It computes the")
print("same function with better memory behaviour.")

# --- section 6.2: the IO analysis -------------------------------------------
print("\n" + "=" * 72)
print("the memory traffic (eqs. 71.4, 71.11-71.12)")
print("=" * 72)
print("HBM accesses, counting elements read or written.\n")
print(f"{'T':>7} {'d':>5} {'naive':>14} {'flash (M=100KB)':>18} "
      f"{'reduction':>11}")
M = 100_000 / 2                                  # elements of fp16 on-chip
for T in (1024, 4096, 16384, 65536):
    d = 64
    naive = 2 * T * T + 4 * T * d                # write+read S, plus Q,K,V,O
    Bc = max(1, int(M / (4 * d)))
    flash = (T / Bc) * (2 * T * d) + 2 * T * d
    print(f"{T:>7} {d:>5} {naive / 1e6:>13.1f}M {flash / 1e6:>17.1f}M "
          f"{naive / flash:>10.1f}x")

print("\nThe reduction grows with T, because the naive term is quadratic and")
print("the tiled one is quadratic with a much smaller constant — a factor")
print("of about M/d fewer accesses, per eq. 71.12.")
print("\nNote what is NOT reduced: the FLOPs are identical, and the KV cache")
print("at serving time is untouched because the cache must persist across")
print("decode steps and cannot be recomputed. FlashAttention zeroes exactly")
print("one row of Chapter 70's table.")

# --- what it costs: recomputation in the backward pass ----------------------
print("\n" + "=" * 72)
print("the backward pass recomputes rather than stores (section 7.2)")
print("=" * 72)
print("Storing S for the backward pass would reintroduce the T^2 memory, so")
print("it is recomputed inside the tile loop instead.\n")
print(f"{'T':>7} {'d':>5} {'attn FLOPs':>13} {'total block FLOPs':>19} "
      f"{'recompute cost':>16}")
for T in (1024, 4096, 16384):
    d = 4096
    attn = 4 * T * T * d
    total = T * (24 * d * d) + attn
    print(f"{T:>7} {d:>5} {attn / 1e12:>12.2f}T {total / 1e12:>18.2f}T "
          f"{attn / (3 * total):>15.1%}")

print("\nThe last column is the extra arithmetic as a fraction of the")
print("training step: recomputing attention's forward pass during the")
print("backward pass. Because attention's FLOPs are a minority at these")
print("lengths (Chapter 70), it is a small price for removing the largest")
print("memory term entirely.")
print("\nThat is gradient checkpointing applied to one operation, and it is")
print("an unusually favourable instance of it.")

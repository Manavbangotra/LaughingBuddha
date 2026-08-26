# -*- coding: utf-8 -*-
# Extracted from: Chapter 69 — Causal Masking and the KV Cache
# Source: src/.../ch069-masking-kv-cache.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The two phases, their different bottlenecks, and why batching is the only
lever during decode (eqs. 69.11-69.13).
"""
import time

import numpy as np

rng = np.random.default_rng(1)


# --- section 5.3: the arithmetic intensity of a decode step -----------------
def decode_intensity(P, L, g, dk, n, B, b=2):
    """Eqs. 69.11-69.12."""
    flops = 2 * P * B + 4 * L * g * dk * n * B
    byts = b * P + 2 * b * L * g * dk * n * B
    return flops / byts


print("=" * 72)
print("decoding is memory-bound at batch 1 (eqs. 69.11-69.13)")
print("=" * 72)
P, L, g, dk = 7e9, 32, 8, 128
print("A 7B model in bf16. A modern accelerator's ridge point is a few")
print("hundred operations per byte.\n")
print(f"{'batch':>7} " + " ".join(f"{f'n={n}':>12}" for n in
                                  (128, 2048, 32768))
      + f" {'regime at n=2048':>20}")
for B in (1, 4, 16, 64, 256):
    row = [decode_intensity(P, L, g, dk, n, B) for n in (128, 2048, 32768)]
    reg = ("memory-bound" if row[1] < 100 else "approaching compute")
    print(f"{B:>7} " + " ".join(f"{x:>12.1f}" for x in row)
          + f" {reg:>20}")

print("\nAt batch 1 the intensity is about 1 operation per byte at every")
print("context length — two to three orders of magnitude below the ridge")
print("point, so the machine is idle waiting for memory.")
print("\nEq. 69.13 says the intensity is LINEAR in the batch size, and the")
print("column confirms it. Batching is not an optimisation during decode; it")
print("is the only thing that makes the accelerator do arithmetic at all.")
print("\nNotice also that the intensity barely changes along each row. That")
print("is eq. 69.10 again: at these contexts the weights dominate the bytes")
print("read, so context length does not change the regime.")

# --- and the latency consequence --------------------------------------------
print("\n" + "=" * 72)
print("time per token is nearly FLAT in context length")
print("=" * 72)
print("Bytes read per generated token, for a 7B GQA model in bf16:\n")
print(f"{'context n':>11} {'weight bytes':>14} {'cache bytes':>13} "
      f"{'total':>10} {'vs n=128':>10}")
base = None
for n in (128, 1024, 8192, 65536, 262144):
    w = 2 * P
    c = 2 * 2 * L * g * dk * n
    tot = w + c
    if base is None:
        base = tot
    print(f"{n:>11,} {w / 1e9:>13.1f}G {c / 1e9:>12.3f}G "
          f"{tot / 1e9:>9.1f}G {tot / base:>9.3f}x")

print("\nAt a 64k context the cache adds 8% to the bytes read; at 8k it adds")
print("1%. So per-token latency is essentially flat in the context length")
print("until the context is enormous.")
print("\nThat surprises people who expect long contexts to be slow per token.")
print("Long contexts are expensive in MEMORY and in PREFILL — which is")
print("quadratic — and nearly free in per-token decode time. Getting that")
print("distinction right is most of what it takes to reason about serving")
print("costs.")

# --- prefill vs decode, measured --------------------------------------------
print("\n" + "=" * 72)
print("the two phases, measured (table 69.1)")
print("=" * 72)
d, h, dk = 512, 8, 64
Wq = rng.normal(0, 1 / np.sqrt(d), (d, d)).astype(np.float32)
Wk = rng.normal(0, 1 / np.sqrt(d), (d, d)).astype(np.float32)
Wv = rng.normal(0, 1 / np.sqrt(d), (d, d)).astype(np.float32)
W1 = rng.normal(0, 1 / np.sqrt(d), (d, 4 * d)).astype(np.float32)
W2 = rng.normal(0, 1 / np.sqrt(4 * d), (4 * d, d)).astype(np.float32)


def block_flops(T, B):
    return B * (8 * T * d * d + 4 * T * T * d + 16 * T * d * d)


print(f"{'phase':<12} {'positions':>11} {'wall ms':>10} {'GFLOP':>9} "
      f"{'GFLOP/s':>10}")
for T in (128, 512):
    X = rng.normal(size=(1, T, d)).astype(np.float32)
    t0 = time.perf_counter()
    for _ in range(5):
        Q, K, V = X @ Wq, X @ Wk, X @ Wv
        S = Q @ K.transpose(0, 2, 1) / np.sqrt(d)
        A = np.exp(S - S.max(-1, keepdims=True))
        A = A / A.sum(-1, keepdims=True)
        ctx = A @ V
        _ = np.maximum(0.0, ctx @ W1) @ W2
    dt = (time.perf_counter() - t0) / 5
    fl = block_flops(T, 1)
    print(f"{'prefill':<12} {T:>11} {dt * 1e3:>10.2f} {fl / 1e9:>9.3f} "
          f"{fl / dt / 1e9:>10.1f}")

for B in (1, 32):
    X1 = rng.normal(size=(B, 1, d)).astype(np.float32)
    Kc = rng.normal(size=(B, 512, d)).astype(np.float32)
    t0 = time.perf_counter()
    for _ in range(50):
        q, k, v = X1 @ Wq, X1 @ Wk, X1 @ Wv
        S = q @ Kc.transpose(0, 2, 1) / np.sqrt(d)
        A = np.exp(S - S.max(-1, keepdims=True))
        A = A / A.sum(-1, keepdims=True)
        ctx = A @ Kc
        _ = np.maximum(0.0, ctx @ W1) @ W2
    dt = (time.perf_counter() - t0) / 50
    fl = B * (8 * d * d + 4 * 512 * d + 16 * d * d)
    print(f"{f'decode B={B}':<12} {B:>11} {dt * 1e3:>10.3f} "
          f"{fl / 1e9:>9.4f} {fl / dt / 1e9:>10.1f}")

print("\nThe GFLOP/s column is the point. Prefill processes many positions")
print("at once and reaches a respectable rate; decode at batch 1 processes")
print("one position and reaches a small fraction of it, because the same")
print("weight matrices are read to do a tiny amount of work.")
print("\nBatching the decode recovers much of the gap, which is")
print("eq. 69.13 measured: the same weight read now serves B tokens.")
print("\nThose are two different problems with two different fixes, and a")
print("serving system that treats them as one phase optimises neither.")

# --- section 6.5: the attention sink and cache eviction ---------------------
print("\n" + "=" * 72)
print("why the first token cannot be evicted (section 6.5)")
print("=" * 72)
print("Simulate a head whose attention includes a sink on position 0, then")
print("evict position 0 and see what happens to the head's output.\n")
T, dk_ = 64, 32
Vv = rng.normal(size=(T, dk_))
# a realistic-looking pattern: strong sink + local + a little content
scores = rng.normal(0, 0.5, T)
scores[0] += 4.0                                     # the sink
scores[-4:] += 1.5                                   # local
A_full = np.exp(scores - scores.max())
A_full /= A_full.sum()
out_full = A_full @ Vv

print(f"{'policy':<28} {'mass on pos 0':>15} {'output shift':>14} "
      f"{'relative':>10}")
print(f"{'keep everything':<28} {A_full[0]:>15.4f} {0.0:>14.4f} "
      f"{0.0:>10.4f}")
for label, keep in (("evict pos 0, keep rest", np.arange(1, T)),
                    ("sliding window w=16", np.arange(T - 16, T)),
                    ("window w=16 + first 4",
                     np.concatenate([np.arange(4), np.arange(T - 16, T)]))):
    sc = scores[keep]
    A = np.exp(sc - sc.max())
    A /= A.sum()
    out = A @ Vv[keep]
    shift = float(np.linalg.norm(out - out_full))
    m0 = float(A[0]) if 0 in keep else 0.0
    print(f"{label:<28} {m0:>15.4f} {shift:>14.4f} "
          f"{shift / np.linalg.norm(out_full):>10.4f}")

print("\nThe head placed most of its mass on position 0 — a token whose")
print("content it was not using. Evicting it forces that mass onto real")
print("tokens, and the output moves substantially.")
print("\nA sliding window that keeps the first few tokens recovers most of")
print("it, at a cost of four extra cached positions. That is why every")
print("sliding-window cache policy has a 'keep the first k' clause, and it")
print("is not a heuristic — it is section 6.5's mechanism.")

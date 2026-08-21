# Extracted from: Chapter 70 — Computational and Memory Complexity of Attention
# Source: src/.../ch070-complexity.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The 6ND rule (eqs. 70.4-70.5), its correction term, and estimating a
training run.
"""
import numpy as np


def params(V, L, d, d_ff=None):
    d_ff = d_ff or 4 * d
    return 2 * V * d + L * (4 * d * d + 2 * d * d_ff)


print("=" * 72)
print("the 6ND rule and its correction (eqs. 70.4-70.5)")
print("=" * 72)
print("6ND counts only the parameter matmuls. The attention term adds a")
print("relative correction of T/(6d).\n")
print(f"{'model':<14} {'d':>7} " +
      " ".join(f"{f'T={T // 1024}k':>12}" for T in (2048, 8192, 32768, 131072)))
print(f"{'':<14} {'':>7} " +
      " ".join(f"{'6ND error':>12}" for _ in range(4)))
for name, V, L, d, dff in (("GPT-2 small", 50257, 12, 768, None),
                           ("7B", 32000, 32, 4096, 11008),
                           ("70B", 128000, 80, 8192, 28672)):
    N = params(V, L, d, dff)
    row = []
    for T in (2048, 8192, 32768, 131072):
        exact = 6 * N + 12 * L * T * d
        row.append(exact / (6 * N) - 1)
    print(f"{name:<14} {d:>7,} " + " ".join(f"{x:>11.1%}" for x in row))

print("\nAt the 2k-8k contexts most published compute figures were computed")
print("at, the correction is a few per cent and 6ND is fine. At 128k it is")
print("a factor of several and 6ND is badly wrong.")
print("\nSo quote 6ND with the regime it applies to. A long-context training")
print("run costed with plain 6ND will come in far over budget, and the")
print("error is entirely predictable from eq. 70.5.")

# --- costing a run ----------------------------------------------------------
print("\n" + "=" * 72)
print("costing a training run (section 7.3)")
print("=" * 72)
PEAK = 1e15          # effective FLOP/s per accelerator, bf16
print(f"assuming {PEAK / 1e12:.0f} TFLOP/s peak per accelerator\n")
print(f"{'model':<10} {'N':>8} {'D tokens':>10} {'C (FLOP)':>11} "
      f"{'accel-days @ MFU':>18}")
for name, V, L, d, dff, D in (("1B", 32000, 24, 2048, None, 20e9),
                              ("7B", 32000, 32, 4096, 11008, 2e12),
                              ("70B", 128000, 80, 8192, 28672, 15e12)):
    N = params(V, L, d, dff)
    C = 6 * N * D
    for mfu in (0.45,):
        days = C / (PEAK * mfu) / 86400
        print(f"{name:<10} {N / 1e9:>7.1f}B {D / 1e12:>9.1f}T "
              f"{C:>11.2e} {days:>15.0f} d")

print("\nThose are single-accelerator days; divide by the fleet size. A 70B")
print("run at 15T tokens is about 160 accelerator-years, which on 2048")
print("devices is under a month — and that arithmetic is the whole reason")
print("large-model training is a capital-expenditure question.")

# --- section 6.5: compute-optimal allocation --------------------------------
print("\n" + "=" * 72)
print("compute-optimal allocation (eq. 70.12)")
print("=" * 72)
print("Chinchilla: at a fixed TRAINING budget, N and D should both scale as")
print("sqrt(C), giving about 20 tokens per parameter.\n")
print(f"{'model':<16} {'N':>8} {'D actual':>10} {'tokens/param':>14} "
      f"{'Chinchilla D':>14} {'ratio':>8}")
HIST = [("GPT-3 (2020)", 175e9, 300e9),
        ("Chinchilla (2022)", 70e9, 1.4e12),
        ("Llama-2 7B (2023)", 7e9, 2e12),
        ("modern 8B (2024+)", 8e9, 15e12)]
for name, N, D in HIST:
    print(f"{name:<16} {N / 1e9:>7.0f}B {D / 1e12:>9.2f}T "
          f"{D / N:>14.1f} {20 * N / 1e12:>13.2f}T {D / (20 * N):>8.2f}x")

print("\nGPT-3 is at 1.7 tokens per parameter — about a twelfth of optimal,")
print("which is what Hoffmann et al. established. And modern small models")
print("are at hundreds of tokens per parameter, an order of magnitude PAST")
print("Chinchilla-optimal.")
print("\nBoth of those look like mistakes and only one is. Eq. 70.12")
print("minimises loss for a fixed TRAINING budget and says nothing about")
print("inference, which is the next table.")

# --- section 6.6: the inference-aware correction ----------------------------
print("\n" + "=" * 72)
print("why modern models are 'overtrained' (eq. 70.13)")
print("=" * 72)
print("Lifetime compute is 6*N*D_train + 2*N*D_inference. Inference is 2N")
print("per token because there is no backward pass.\n")
print(f"{'served tokens':>15} " +
      " ".join(f"{f'{n / 1e9:.0f}B model':>16}" for n in (8e9, 70e9)))
print(f"{'':>15} " +
      " ".join(f"{'train / total':>16}" for _ in range(2)))
for Dinf in (1e11, 1e13, 1e15, 1e17):
    row = []
    for N, Dtr in ((8e9, 15e12), (70e9, 15e12)):
        tr = 6 * N * Dtr
        inf = 2 * N * Dinf
        row.append(tr / (tr + inf))
    print(f"{Dinf:>15.0e} " + " ".join(f"{x:>16.1%}" for x in row))

print("\nOnce a model serves more than about three times its training")
print("tokens, inference dominates the lifetime compute — and inference")
print("cost scales with N and not with D.")
print("\nSo for a widely-served model the right move is a SMALLER model")
print("trained on MORE tokens than eq. 70.12 recommends: you pay more")
print("training compute once to pay less inference compute forever. That is")
print("the entire explanation for the shift from GPT-3's ratio to a modern")
print("8B model's, and neither is a mistake — they are optimising different")
print("objectives.")

# --- training memory --------------------------------------------------------
print("\n" + "=" * 72)
print("training memory (eq. 70.9)")
print("=" * 72)
print("Mixed precision: bf16 weights and gradients, fp32 master copy and")
print("two Adam moments. 16 bytes per parameter before any activation.\n")
print(f"{'model':<10} {'N':>8} {'weights':>9} {'grads':>8} {'Adam':>8} "
      f"{'master':>8} {'state total':>12} {'serve bf16':>11}")
for name, V, L, d, dff in (("1B", 32000, 24, 2048, None),
                           ("7B", 32000, 32, 4096, 11008),
                           ("70B", 128000, 80, 8192, 28672)):
    N = params(V, L, d, dff)
    w, g_, a, m = 2 * N, 2 * N, 8 * N, 4 * N
    print(f"{name:<10} {N / 1e9:>7.1f}B {w / 1e9:>8.1f}G {g_ / 1e9:>7.1f}G "
          f"{a / 1e9:>7.1f}G {m / 1e9:>7.1f}G {(w + g_ + a + m) / 1e9:>11.1f}G "
          f"{2 * N / 1e9:>10.1f}G")

print("\nThe last two columns are the ratio that decides infrastructure: a")
print("model needs eight times as much memory to train as to serve, before")
print("a single activation is stored.")
print("\nAnd the Adam column is half the total. That is why optimiser-state")
print("sharding is the first thing any large-scale training framework does,")
print("and why 8-bit Adam is worth the complexity — it removes a quarter of")
print("the state outright.")

# Extracted from: Chapter 70 — Computational and Memory Complexity of Attention
# Source: src/.../ch070-complexity.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The roofline applied to each phase, and what each optimisation moves."""
import time

import numpy as np

rng = np.random.default_rng(0)

# --- section 6.4: the two phases sit at opposite ends of the roofline -------
print("=" * 72)
print("prefill and decode are opposite ends of the same roofline (6.4)")
print("=" * 72)


def intensity(flops, byts):
    return flops / byts


def phase_intensity(N, L, d, h, g, dk, T, B, phase, b=2):
    if phase == "prefill":
        fl = B * T * (2 * N + 4 * L * T * d)
        by = b * (N + B * T * L * (10 * d + 4 * d) + B * L * h * T * T)
    else:
        fl = B * (2 * N + 4 * L * g * dk * T)
        by = b * (N + 2 * B * L * g * dk * T)
    return intensity(fl, by)


N, L, d, h, g, dk = 7e9, 32, 4096, 32, 8, 128
print("7B model, GQA g=8, bf16. Ridge point is a few hundred ops/byte.\n")
print(f"{'phase':<10} {'batch':>6} " +
      " ".join(f"{f'T={T}':>13}" for T in (512, 4096, 32768)))
for phase in ("prefill", "decode"):
    for B in (1, 32):
        row = [phase_intensity(N, L, d, h, g, dk, T, B, phase)
               for T in (512, 4096, 32768)]
        print(f"{phase:<10} {B:>6} " + " ".join(f"{x:>13.1f}" for x in row))

print("\nPrefill is in the hundreds or thousands: compute-bound, the machine")
print("is doing arithmetic. Decode at batch 1 is around one: memory-bound by")
print("three orders of magnitude, the machine is waiting.")
print("\nThose are the SAME MODEL, minutes apart in the same request. No")
print("single optimisation serves both, which is why prefill and decode are")
print("increasingly scheduled — and sometimes hosted — separately.")

# --- what each optimisation moves -------------------------------------------
print("\n" + "=" * 72)
print("what each optimisation actually changes (table 70.1)")
print("=" * 72)
B, T = 8, 8192
base = {
    "param FLOPs": 2 * N * B * T,
    "attn FLOPs": 4 * L * T * T * d * B,
    "attn memory": 2 * B * L * h * T * T,
    "activation memory": 2 * B * T * L * 14 * d,
    "optimiser state": 12 * N,
    "KV cache (serving)": 2 * 2 * L * g * dk * T * B,
}
print(f"baseline, B={B}, T={T}:\n")
for k, v in base.items():
    unit = "GFLOP" if "FLOP" in k else "GB"
    print(f"  {k:<22} {v / 1e9:>12,.1f} {unit}")

OPTS = {
    "FlashAttention": {"attn memory": 0.0},
    "GQA g=8 (already on)": {"KV cache (serving)": 1.0},
    "gradient checkpointing": {"activation memory": 0.15},
    "8-bit Adam": {"optimiser state": 0.5},
    "sliding window w=1024": {"attn FLOPs": 1024 / T,
                              "attn memory": 1024 / T,
                              "KV cache (serving)": 1024 / T},
    "int8 KV cache": {"KV cache (serving)": 0.5},
}
print(f"\n{'optimisation':<24} " +
      " ".join(f"{k.split()[0][:9]:>11}" for k in base))
for name, effect in OPTS.items():
    row = []
    for k in base:
        f = effect.get(k, 1.0)
        row.append("—" if f == 1.0 else
                   "0" if f == 0.0 else f"{f:.2f}x")
    print(f"{name:<24} " + " ".join(f"{v:>11}" for v in row))

print("\nEvery column is one line of the accounting and every row touches")
print("one or two of them. That is the point of building the table before")
print("the techniques: FlashAttention is not 'making attention faster', it")
print("is zeroing exactly one term, and it leaves the FLOPs and the KV cache")
print("untouched.")
print("\nAnd it explains why the techniques compose: they act on different")
print("terms. Applying all of them is not redundant — each removes a")
print("different bottleneck, and which one is binding depends on B, T and")
print("whether you are training or serving.")

# --- measure the elementwise gap --------------------------------------------
print("\n" + "=" * 72)
print("why measured MFU tops out well below 100% (section 7.1)")
print("=" * 72)
print("6ND counts matmuls. Everything else is cheap in FLOPs and not cheap")
print("in TIME, because it is bandwidth-bound.\n")
dd, TT, BB = 512, 256, 4
X = rng.normal(size=(BB, TT, dd)).astype(np.float32)
W1 = rng.normal(0, 0.02, (dd, 4 * dd)).astype(np.float32)
W2 = rng.normal(0, 0.02, (4 * dd, dd)).astype(np.float32)
Wq = rng.normal(0, 0.02, (dd, dd)).astype(np.float32)


def timeit(fn, reps=20):
    fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


t_mm = timeit(lambda: np.maximum(0.0, X @ W1) @ W2)
t_only_mm = timeit(lambda: (X @ W1) @ W2)
t_norm = timeit(lambda: X / np.sqrt((X ** 2).mean(-1, keepdims=True) + 1e-6))
t_proj = timeit(lambda: X @ Wq)

mm_flops = 2 * BB * TT * dd * 4 * dd * 2
print(f"{'operation':<28} {'ms':>9} {'GFLOP':>9} {'GFLOP/s':>10}")
print(f"{'FFN matmuls only':<28} {t_only_mm * 1e3:>9.3f} "
      f"{mm_flops / 1e9:>9.3f} {mm_flops / t_only_mm / 1e9:>10.1f}")
print(f"{'FFN with ReLU':<28} {t_mm * 1e3:>9.3f} "
      f"{mm_flops / 1e9:>9.3f} {mm_flops / t_mm / 1e9:>10.1f}")
print(f"{'RMSNorm alone':<28} {t_norm * 1e3:>9.3f} "
      f"{3 * X.size / 1e9:>9.5f} {3 * X.size / t_norm / 1e9:>10.2f}")
print(f"{'one projection':<28} {t_proj * 1e3:>9.3f} "
      f"{2 * BB * TT * dd * dd / 1e9:>9.3f} "
      f"{2 * BB * TT * dd * dd / t_proj / 1e9:>10.1f}")

print(f"\nReLU alone costs {(t_mm - t_only_mm) * 1e3:.3f} ms and "
      f"{X.size * 4 / 1e9:.4f} GFLOP.")
print(f"RMSNorm reaches {3 * X.size / t_norm / 1e9:.2f} GFLOP/s against the")
print(f"matmuls' {mm_flops / t_only_mm / 1e9:.0f} — two orders of magnitude")
print("apart, on the same hardware, in the same model.")
print("\nThat gap is where MFU goes. The elementwise operations contribute")
print("almost nothing to the 6ND count and a real fraction of the wall")
print("clock, because they are bandwidth-bound and the matmuls are not.")
print("\nSo a reported MFU of 45% is not 55% of the machine sitting idle. It")
print("is mostly this, plus communication, and closing it is a kernel-fusion")
print("problem rather than a scheduling one (Chapter 51).")

# --- putting it together: can this run fit? ---------------------------------
print("\n" + "=" * 72)
print("the check people skip: does it FIT? (section 7.3, step 5)")
print("=" * 72)


def training_memory_gb(N, L, d, h, d_ff, B, T, b=2, flash=True,
                       checkpoint=False):
    state = 16 * N
    act_per_layer = b * B * T * (10 * d + d_ff)
    attn = 0.0 if flash else b * B * L * h * T * T
    act = L * act_per_layer + attn
    if checkpoint:
        act = act * 0.15 + act_per_layer * np.sqrt(L)
    return (state + act) / 1e9


N7, L7, d7, h7, dff7 = 7e9, 32, 4096, 32, 11008
print("7B model on one 80 GB accelerator, mixed precision.\n")
print(f"{'batch':>6} {'T':>7} {'flash':>7} {'ckpt':>6} {'memory':>10} "
      f"{'fits 80G?':>11}")
for B_ in (1, 4, 16):
    for T_ in (2048, 8192):
        for flash in (False, True):
            for ck in (False, True):
                m = training_memory_gb(N7, L7, d7, h7, dff7, B_, T_,
                                       flash=flash, checkpoint=ck)
                print(f"{B_:>6} {T_:>7,} {str(flash):>7} {str(ck):>6} "
                      f"{m:>9.1f}G {('yes' if m < 80 else 'NO'):>11}")

print("\nThe state alone is 112 GB, so a 7B model does not train on one")
print("80 GB device under ANY of these settings — which is the answer step")
print("5 of section 7.3 is supposed to produce before anyone estimates a")
print("wall-clock time.")
print("\nRead the rows against each other anyway: FlashAttention's effect")
print("grows with T squared and checkpointing's is roughly constant, so")
print("which one you need depends entirely on the context length. That is")
print("the accounting doing its job — telling you which term is binding")
print("before you pick a technique.")

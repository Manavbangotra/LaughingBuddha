# Extracted from: Chapter 62 — Why Recurrence Failed: The Road to Attention
# Source: src/.../ch062-why-attention.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Table 62.1 measured: path length, work, and the sequential floor that
hardware cannot remove.
"""
import time

import numpy as np

rng = np.random.default_rng(1)


# --- section 5.5: the work columns ------------------------------------------
def work(mechanism, T, d, k=3, w=128):
    if mechanism == "recurrence":
        return 2 * T * d * d
    if mechanism == "convolution":
        return 2 * k * T * d * d
    if mechanism == "attention":
        return 2 * T * T * d + 2 * 4 * T * d * d      # scores + projections
    if mechanism == "windowed":
        return 2 * T * min(w, T) * d + 2 * 4 * T * d * d
    raise ValueError(mechanism)


def path_length(mechanism, T, k=3, w=128):
    if mechanism == "recurrence":
        return T
    if mechanism == "convolution":
        return int(np.ceil((T - 1) / (k - 1)))
    if mechanism == "attention":
        return 1
    if mechanism == "windowed":
        return int(np.ceil(T / w))
    raise ValueError(mechanism)


print("=" * 72)
print("table 62.1, with numbers (d = 768)")
print("=" * 72)
d = 768
print(f"{'T':>7} {'mechanism':<14} {'max path':>10} {'GFLOP/layer':>13} "
      f"{'vs recurrence':>15}")
for T in (128, 512, 2048, 8192):
    base = work("recurrence", T, d)
    for mech in ("recurrence", "convolution", "attention", "windowed"):
        wk = work(mech, T, d)
        print(f"{T:>7} {mech:<14} {path_length(mech, T):>10} "
              f"{wk / 1e9:>13.3f} {wk / base:>14.2f}x")
    print()

print("Read the last column down each block. At T = 128 attention costs")
print("about the same as a recurrence; by T = 8192 it costs several times")
print("more, and the ratio keeps growing because one term is quadratic in T")
print("and the other is linear.")
print("\nNow read the path-length column. It is 1 for attention at every")
print("length, and equal to T for the recurrence. That column is what")
print("decides whether a long-range dependency is learnable at all")
print("(Chapter 60), and no amount of the work column buys it.")
print("\nThe windowed row is the compromise: linear work again, path length")
print("back to T/w. Chapter 71 is about whether that trade is worth making,")
print("and the answer turns out to depend on something neither column")
print("shows.")

# --- section 6.3: the sequential floor --------------------------------------
print("\n" + "=" * 72)
print("the sequential floor that hardware cannot remove (eq. 62.9)")
print("=" * 72)
print("Both do the SAME arithmetic. One must do it in T dependent rounds.\n")

d = 256
W = rng.normal(0, 1 / np.sqrt(d), (d, d)).astype(np.float32)
print(f"{'T':>6} {'batch':>7} {'recurrent (T rounds)':>22} "
      f"{'attention-shaped (1 round)':>28} {'ratio':>8}")
for T, B in ((128, 32), (512, 32), (512, 128)):
    X = rng.normal(size=(B, T, d)).astype(np.float32)
    h = np.zeros((B, d), dtype=np.float32)
    t0 = time.perf_counter()
    for t in range(T):
        h = np.tanh(h @ W + X[:, t])
    dt_rec = time.perf_counter() - t0

    Q = X.reshape(B * T, d)
    t0 = time.perf_counter()
    _ = np.tanh(Q @ W)                     # the same per-position work, fused
    dt_par = time.perf_counter() - t0
    print(f"{T:>6} {B:>7} {dt_rec * 1e3:>20.2f}ms "
          f"{dt_par * 1e3:>26.2f}ms {dt_rec / dt_par:>8.1f}x")

print("\nThe recurrent column is not slower because it does more work — it")
print("does exactly the same multiplies. It is slower because eq. 62.9's")
print("depth term binds: T dependent rounds, each too small to occupy the")
print("machine, against one round that is not.")
print("\nBrent's bound says time is at least max(D, W/p). More processors")
print("shrink W/p and do nothing to D. So the recurrence has a floor that")
print("hardware cannot lower and attention does not, and that asymmetry —")
print("not any modelling argument — is the reason the field moved.")

# --- where attention's quadratic term starts to bite ------------------------
print("\n" + "=" * 72)
print("where the quadratic term takes over")
print("=" * 72)
print("Attention's cost is 2*T^2*d for the scores plus 8*T*d^2 for the")
print("projections. The crossover is at T = 4d.\n")
print(f"{'d':>6} {'crossover T':>13} {'at T=1024, scores are':>24} "
      f"{'at T=8192':>12}")
for d_ in (256, 512, 768, 4096):
    cross = 4 * d_
    f1 = 2 * 1024 * 1024 * d_
    p1 = 8 * 1024 * d_ * d_
    f2 = 2 * 8192 * 8192 * d_
    p2 = 8 * 8192 * d_ * d_
    print(f"{d_:>6} {cross:>13} {f1 / (f1 + p1):>23.1%} "
          f"{f2 / (f2 + p2):>11.1%}")

print("\nAt a typical width the quadratic term is a MINORITY of the FLOPs at")
print("ordinary sequence lengths — most of the compute is in the linear")
print("projections. That is worth knowing, because 'attention is quadratic'")
print("is usually stated as though the quadratic part dominates, and at")
print("T = 1024 with d = 4096 it is under a tenth of the arithmetic.")
print("\nWhat is quadratic without qualification is the MEMORY: the T-by-T")
print("score matrix must exist somewhere. Chapter 70 separates those two")
print("costs carefully, because they have different fixes — and")
print("FlashAttention addresses the memory one without touching the")
print("arithmetic at all.")

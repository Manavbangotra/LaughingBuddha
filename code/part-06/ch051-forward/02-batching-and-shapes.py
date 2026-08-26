# -*- coding: utf-8 -*-
# Extracted from: Chapter 51 — Forward Propagation and Computational Graphs
# Source: src/.../ch051-forward.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Arithmetic intensity, the batching curve, and the shape rules that
prevent most deep learning bugs.
"""
import time

import numpy as np

rng = np.random.default_rng(1)


# --- section 6.3: the roofline, measured ------------------------------------
def intensity(B, n, m, bytes_per=4):
    """Eq. 51.6: arithmetic operations per byte moved."""
    work = 2 * B * n * m
    data = bytes_per * (B * n + n * m + B * m)
    return work / data


print("=" * 72)
print("arithmetic intensity and the batching curve (eqs. 51.5, 51.6)")
print("=" * 72)
n = m = 1024
A_w = rng.normal(size=(n, m)).astype(np.float32)
print(f"a {n}x{m} dense layer, float32\n")
print(f"{'batch':>7} {'intensity':>11} {'GFLOP/s':>10} "
      f"{'us per example':>16} {'regime':<18}")

ridge_guess = None
prev_per_ex = None
for B in (1, 2, 8, 32, 128, 512, 2048):
    X = rng.normal(size=(B, n)).astype(np.float32)
    reps = max(20, int(4e9 / (2 * B * n * m)))
    for _ in range(5):
        X @ A_w                                    # warm up
    # median of five timed blocks: a single block is badly contaminated by
    # thread-pool wake-up and frequency scaling at small batch sizes
    trials = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(reps):
            X @ A_w
        trials.append((time.perf_counter() - t0) / reps)
    dt = float(np.median(trials))
    gflops = 2 * B * n * m / dt / 1e9
    per_ex = dt / B * 1e6
    I = intensity(B, n, m)
    regime = "memory-bound" if I < 30 else "approaching compute"
    print(f"{B:>7} {I:>11.1f} {gflops:>10.1f} {per_ex:>16.2f} {regime:<18}")

print("\nThe per-example cost falls steeply and then flattens — a knee, not")
print("a slope. Eq. 51.6 explains it: at batch 1 the weight matrix is loaded")
print("from memory to be used once, giving an intensity below 1, and the")
print("machine spends nearly all its time waiting. Each additional example")
print("reuses that same loaded matrix.")
print("\nPast the knee the operation is arithmetic-limited and further")
print("batching buys nothing per example — it only costs memory. That is the")
print("real reason to batch, and it is a bandwidth argument rather than a")
print("parallelism one.")
print("\nTwo honest caveats about this measurement. The very small batches")
print("are noisy even with the median of five timed blocks, because the")
print("per-call overhead of the library's threading is comparable to the")
print("work itself. And the largest batch does not improve on the previous")
print("one, because once the operation is compute-bound the only thing left")
print("to gain would be better cache behaviour, and a larger working set")
print("makes that worse rather than better.")

# --- the same argument in reverse: why serving batches requests -------------
print("\n" + "=" * 72)
print("the inference consequence")
print("=" * 72)
X1 = rng.normal(size=(1, n)).astype(np.float32)
X64 = rng.normal(size=(64, n)).astype(np.float32)
for label, X in (("one request at a time", X1), ("64 requests batched", X64)):
    reps = 200
    X @ A_w
    t0 = time.perf_counter()
    for _ in range(reps):
        X @ A_w
    dt = (time.perf_counter() - t0) / reps
    print(f"{label:<24} {dt * 1e6:>9.1f} us total   "
          f"{dt / len(X) * 1e6:>8.2f} us per request")
print("\nThat is the trade an inference server makes, and it is worth being")
print("precise about which direction each number moves. Batching 64 requests")
print("makes each individual request take LONGER end to end — the whole")
print("batch must finish before any of it returns — while cutting the cost")
print("per request severalfold. Throughput improves; per-request latency")
print("degrades. Section 5.5's note that serving lives at batch 1 is why the")
print("trade is usually worth making anyway, and the batching window is")
print("chosen so that the added latency stays inside the service objective.")

# --- shape discipline -------------------------------------------------------
print("\n" + "=" * 72)
print("the shape rule that prevents most bugs (section 5.2)")
print("=" * 72)
print("Read a matmul as: consume the LAST axis of the left operand against")
print("the FIRST axis of the right; leave every other axis alone.\n")
cases = [
    ("dense, batched", (32, 784), (784, 256)),
    ("sequence, batched", (8, 128, 512), (512, 2048)),
    ("image features", (16, 49, 768), (768, 768)),
]
for label, sa, sb in cases:
    a, b = np.zeros(sa), np.zeros(sb)
    out = a @ b
    print(f"{label:<20} {str(sa):>18} @ {str(sb):<12} -> {str(out.shape)}")
print("\nNo new rule was needed for the three-dimensional cases. A batch or")
print("sequence axis is simply an axis that nothing consumed.")

# --- the broadcasts that fail SILENTLY --------------------------------------
print("\n" + "=" * 72)
print("broadcasts that do the wrong thing without erroring")
print("=" * 72)

B_, C = 6, 4
logits = rng.normal(size=(B_, C))
bias_ok = rng.normal(size=(C,))                # per-CLASS bias, correct
bias_bad = rng.normal(size=(B_,))              # per-EXAMPLE, a mistake

print(f"logits {logits.shape}, correct bias {bias_ok.shape}")
print(f"  logits + bias  -> {(logits + bias_ok).shape}   correct\n")
print(f"logits {logits.shape}, wrong bias {bias_bad.shape}")
try:
    r = logits + bias_bad
    print(f"  logits + bias  -> {r.shape}")
except ValueError as e:
    print(f"  raises: {str(e)[:60]}")
print(f"  ...but reshaped to {(B_, 1)} it broadcasts happily:")
print(f"  logits + bias[:, None] -> {(logits + bias_bad[:, None]).shape}   "
      f"WRONG, and silent")

print("\nThe second form adds a per-example constant to every class, which")
print("leaves the softmax output completely unchanged (it is shift-invariant")
print("per row) — so the bug produces no error, no shape mismatch, and no")
print("visible symptom beyond a bias vector that never learns anything.")

# a subtler one: the accidental outer product
print("\nthe accidental outer product:")
pred = rng.normal(size=(B_,))
targ = rng.normal(size=(B_,))
correct = float(np.mean((pred - targ) ** 2))
wrong = float(np.mean((pred[:, None] - targ[None, :]) ** 2))
print(f"  mean((pred - targ)**2)                     = {correct:.4f}  correct")
print(f"  mean((pred[:,None] - targ[None,:])**2)     = {wrong:.4f}  WRONG")
print(f"  the second compares every prediction against every target:")
print(f"  it computed a {B_}x{B_} matrix where a length-{B_} vector was meant")
print("\nBoth of these run, produce a plausible float, and train to a")
print("plausible-looking loss curve. Shape assertions at layer boundaries")
print("are cheap and catch all of them:\n")


def assert_shape(t, expected, name):
    if t.shape != expected:
        raise AssertionError(f"{name}: expected {expected}, got {t.shape}")
    return t


try:
    assert_shape(pred[:, None] - targ[None, :], (B_,), "residual")
except AssertionError as e:
    print(f"  caught: {e}")

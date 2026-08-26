# Extracted from: Chapter 89 — Next-Token Prediction and Cross-Entropy Loss
# Source: src/.../ch089-next-token.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Temperature scaling: one parameter, fitted on held-out data."""
import numpy as np

rng = np.random.default_rng(1)
N, K = 8000, 20

# A model that is systematically overconfident by a constant factor — the
# situation temperature scaling is designed for.
true_logits = rng.normal(size=(N, K))
p_true = np.exp(true_logits - true_logits.max(1, keepdims=True))
p_true /= p_true.sum(1, keepdims=True)
y = np.array([rng.choice(K, p=row) for row in p_true])
model_logits = true_logits * 1.8            # the miscalibration


def softmax_T(z, T):
    s = z / T
    s = s - s.max(1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(1, keepdims=True)


def nll(z, y, T):
    p = softmax_T(z, T)
    return float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean())


def ece(z, y, T, n_bins=15):
    p = softmax_T(z, T)
    conf, pred = p.max(1), p.argmax(1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    tot = 0.0
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum():
            tot += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return tot


split = N // 2
val_z, val_y = model_logits[:split], y[:split]
test_z, test_y = model_logits[split:], y[split:]

# Fit T on the validation half by minimising NLL — one scalar parameter.
grid = np.linspace(0.5, 4.0, 200)
losses = [nll(val_z, val_y, T) for T in grid]
T_hat = float(grid[int(np.argmin(losses))])

print(f"fitted temperature: {T_hat:.3f}  (the miscalibration was a 1.8x "
      f"logit scale, so the correct T is 1.8)")
print(f"\n{'':<22} {'NLL':>9} {'ECE':>9} {'accuracy':>10}")
for label, T in [("uncorrected (T=1)", 1.0), (f"scaled (T={T_hat:.2f})", T_hat)]:
    print(f"{label:<22} {nll(test_z, test_y, T):>9.4f} "
          f"{ece(test_z, test_y, T):>9.4f} "
          f"{(softmax_T(test_z, T).argmax(1) == test_y).mean():>10.4f}")

print("\nAccuracy is unchanged — dividing every logit by a constant cannot "
      "reorder them. Only the probabilities moved.")

# Where a single scalar cannot help: input-dependent miscalibration.
easy = rng.normal(size=(N // 2, K))
hard = rng.normal(size=(N // 2, K)) * 0.4          # genuinely more uncertain
mixed_true = np.vstack([easy, hard])
p_m = np.exp(mixed_true - mixed_true.max(1, keepdims=True))
p_m /= p_m.sum(1, keepdims=True)
y_m = np.array([rng.choice(K, p=row) for row in p_m])
# Overconfident on the hard half only.
mixed_logits = np.vstack([easy, hard * 3.0])

grid_losses = [nll(mixed_logits, y_m, T) for T in grid]
T_mixed = float(grid[int(np.argmin(grid_losses))])
print(f"\ninput-dependent miscalibration: best single T = {T_mixed:.3f}")
print(f"{'subset':<14} {'ECE at T=1':>12} {'ECE at fitted T':>17}")
for name, sl in [("easy half", slice(0, N // 2)), ("hard half", slice(N // 2, N))]:
    print(f"{name:<14} {ece(mixed_logits[sl], y_m[sl], 1.0):>12.4f} "
          f"{ece(mixed_logits[sl], y_m[sl], T_mixed):>17.4f}")

print("""
One scalar applies the same correction everywhere. When the miscalibration
differs across inputs, fitting T trades one subset's calibration against the
other's — the fitted value is a compromise that is wrong for both. That is the
limit of the method, and it is why calibration should be reported per slice
rather than in aggregate.""")

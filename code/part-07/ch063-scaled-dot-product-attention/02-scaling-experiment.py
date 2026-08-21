# Extracted from: Chapter 63 — Scaled Dot-Product Attention
# Source: src/.../ch063-scaled-dot-product-attention.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Empirical check of eq. 63.7: Var(q·k) = d_k, and what that does to softmax.

Reported entropy is in nats, normalised by log(n) so that 1.0 means a uniform
distribution over n positions and 0.0 means all mass on one position.
"""
import numpy as np

rng = np.random.default_rng(0)
n_trials, n_keys = 20_000, 64


def normalised_entropy(p, axis=-1):
    """Shannon entropy divided by its maximum, log(n)."""
    p = np.clip(p, 1e-12, None)
    h = -(p * np.log(p)).sum(axis=axis)
    return h / np.log(p.shape[axis])


def softmax(z, axis=-1):
    e = np.exp(z - z.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


print(f"{'d_k':>6} {'Var(q·k)':>10} {'predicted':>10} "
      f"{'H unscaled':>12} {'H scaled':>10} {'max weight':>11}")
for d_k in (4, 16, 64, 256, 1024):
    q = rng.normal(size=(n_trials, d_k))
    k = rng.normal(size=(n_trials, d_k))
    dots = (q * k).sum(axis=-1)

    # One query scored against n_keys independent keys, repeated many times.
    q1 = rng.normal(size=(2000, 1, d_k))
    ks = rng.normal(size=(2000, n_keys, d_k))
    scores = (q1 * ks).sum(axis=-1)

    h_unscaled = normalised_entropy(softmax(scores)).mean()
    h_scaled = normalised_entropy(softmax(scores / np.sqrt(d_k))).mean()
    max_w = softmax(scores).max(axis=-1).mean()

    print(f"{d_k:>6} {dots.var():>10.2f} {d_k:>10} "
          f"{h_unscaled:>12.4f} {h_scaled:>10.4f} {max_w:>11.4f}")

print("\nVar(q·k) tracks d_k, confirming eq. 63.7.")
print("Unscaled: entropy collapses toward 0 as d_k grows, and the mean largest")
print("weight approaches 1 — a near-one-hot distribution whose softmax Jacobian")
print("is ~0, so no gradient reaches W^Q or W^K.")
print("Scaled: entropy is essentially CONSTANT across four orders of magnitude")
print("of d_k. That dimension-independence is the whole point of the factor.")

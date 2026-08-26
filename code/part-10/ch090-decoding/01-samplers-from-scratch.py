# -*- coding: utf-8 -*-
# Extracted from: Chapter 90 — Decoding: Softmax, Temperature, Top-k, Top-p, and Beam Search
# Source: src/.../ch090-decoding.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Greedy, temperature, top-k, top-p — the complete set, from logits."""
import numpy as np

rng = np.random.default_rng(0)


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def apply_temperature(logits, T):
    """Equation (eq:decoding-temperature). T -> 0 is greedy, T -> inf uniform."""
    if T <= 0:
        out = np.full_like(logits, -1e9)
        out[logits.argmax()] = 0.0
        return out
    return logits / T


def top_k_filter(logits, k):
    """Equation (eq:top-k). Mask all but the k highest logits."""
    if k <= 0 or k >= len(logits):
        return logits
    threshold = np.partition(logits, -k)[-k]
    return np.where(logits < threshold, -1e9, logits)


def top_p_filter(logits, p):
    """Equation (eq:top-p). Keep the smallest prefix reaching cumulative p."""
    if not (0 < p < 1):
        return logits
    probs = softmax(logits)
    order = np.argsort(-probs)
    cumulative = np.cumsum(probs[order])
    # Keep everything up to and INCLUDING the token that crosses p.
    n_keep = int(np.searchsorted(cumulative, p) + 1)
    keep = order[:n_keep]
    out = np.full_like(logits, -1e9)
    out[keep] = logits[keep]
    return out


# A realistic-looking distribution: a few plausible tokens and a long tail.
V = 2000
logits = np.concatenate([
    np.array([6.0, 5.6, 3.1, 3.0, 2.9]),
    rng.normal(-2.0, 1.0, V - 5),
])

base = softmax(logits)
print(f"vocabulary {V}, top-5 probabilities: "
      f"{np.round(np.sort(base)[::-1][:5], 4).tolist()}")
print(f"mass in the tail beyond the top 5: {1 - np.sort(base)[::-1][:5].sum():.4f}")
print("That tail is small per token and there are 1,995 of them — which is why "
      "unfiltered sampling eventually picks something absurd.\n")


def nucleus_size(logits, p):
    probs = softmax(logits)
    return int(np.searchsorted(np.cumsum(np.sort(probs)[::-1]), p) + 1)


print(f"{'T':>5} {'entropy':>9} {'top-1 prob':>12} {'nucleus @0.9':>14} "
      f"{'nucleus @0.95':>15}")
for T in (0.2, 0.5, 0.7, 1.0, 1.3, 2.0):
    z = apply_temperature(logits, T)
    pr = softmax(z)
    ent = float(-(pr * np.log(pr + 1e-12)).sum())
    print(f"{T:>5.1f} {ent:>9.4f} {pr.max():>12.4f} "
          f"{nucleus_size(z, 0.90):>14} {nucleus_size(z, 0.95):>15}")

print("""
Entropy rises monotonically with T, which is equation
(eq:entropy-temperature-derivative). And the nucleus GROWS with temperature —
so temperature and top-p are not independent knobs. Raising temperature while
holding p fixed widens the sampled set twice over: once by flattening the
distribution and once by admitting more tokens into the nucleus.""")

# Top-k against top-p on distributions of different sharpness.
sharp = np.concatenate([np.array([9.0, 2.0, 1.5]), rng.normal(-3.0, 1.0, V - 3)])
flat = np.concatenate([np.linspace(3.0, 2.0, 40), rng.normal(-1.0, 1.0, V - 40)])

print(f"\n{'distribution':<12} {'top-1':>8} {'entropy':>9} "
      f"{'top-k=50 keeps':>16} {'top-p=0.9 keeps':>17}")
for name, lg in [("sharp", sharp), ("flat", flat)]:
    pr = softmax(lg)
    ent = float(-(pr * np.log(pr + 1e-12)).sum())
    print(f"{name:<12} {pr.max():>8.4f} {ent:>9.4f} {50:>16} "
          f"{nucleus_size(lg, 0.9):>17}")

print("""
This is the argument for top-p in one table. A fixed k=50 keeps fifty tokens
whatever the distribution looks like: on the sharp one that admits 49 tokens of
noise, and on the flat one it truncates a genuine continuum. Top-p keeps a
handful on the sharp distribution and many more on the flat one, because it
fixes the MASS and lets the set size follow.""")

# The full chain, in the order of section 5.6.
def sample(logits, T=1.0, k=0, p=1.0, generator=None):
    z = apply_temperature(logits, T)
    z = top_k_filter(z, k)
    z = top_p_filter(z, p)
    probs = softmax(z)
    g = generator or rng
    return int(g.choice(len(probs), p=probs))


counts = {}
for _ in range(4000):
    tok = sample(logits, T=1.0, k=0, p=0.9)
    counts[tok] = counts.get(tok, 0) + 1
print(f"\n4,000 draws at T=1.0, p=0.9 selected {len(counts)} distinct tokens "
      f"(nucleus size {nucleus_size(logits, 0.9)})")
assert len(counts) <= nucleus_size(logits, 0.9), "sampling must stay in the nucleus"
print("Sampling never left the nucleus — truncation is a hard constraint, not "
      "a bias.")

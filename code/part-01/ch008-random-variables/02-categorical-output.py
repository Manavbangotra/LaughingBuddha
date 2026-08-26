# -*- coding: utf-8 -*-
# Extracted from: Chapter 8 — Random Variables, Distributions, and Expectation
# Source: src/.../ch008-random-variables.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The output of a language model is a categorical distribution.

Sampling strategies are transformations of that distribution before drawing
from it. Here the effects are measured rather than described.
"""
import numpy as np

rng = np.random.default_rng(7)

vocab = ["the", "a", "cat", "dog", "runs", "sleeps", "quantum", "purple"]
logits = np.array([3.2, 2.8, 1.5, 1.2, 0.4, 0.1, -2.0, -3.5])


def softmax(z, temperature=1.0):
    z = z / temperature
    e = np.exp(z - z.max())
    return e / e.sum()


probs = softmax(logits)
print(f"{'token':<10} {'logit':>7} {'p':>8}")
for t, lg, p in zip(vocab, logits, probs):
    print(f"{t:<10} {lg:>7.1f} {p:>8.4f}")
print(f"{'sum':<10} {'':>7} {probs.sum():>8.4f}  <- a categorical distribution")


def entropy(p):
    p = p[p > 0]
    return -(p * np.log(p)).sum()


# Temperature reshapes the distribution before sampling (Chapter 90).
print(f"\n{'temperature':>12} {'entropy':>9} {'max p':>8} "
      f"{'effective choices':>18}")
for tau in (0.2, 0.5, 1.0, 1.5, 3.0):
    p = softmax(logits, tau)
    h = entropy(p)
    print(f"{tau:>12.1f} {h:>9.4f} {p.max():>8.4f} {np.exp(h):>18.2f}")
print("exp(entropy) is the 'perplexity' — roughly how many tokens are")
print("genuinely in play. Low temperature narrows it toward 1.")

# Top-k truncation: keep k tokens, renormalise. This is conditioning (Ch. 7)
# on the event 'the token is one of these k'.
def top_k(p, k):
    out = np.zeros_like(p)
    idx = np.argsort(-p)[:k]
    out[idx] = p[idx]
    return out / out.sum()


print(f"\n{'k':>4} {'kept':>34} {'P(nonsense) removed':>21}")
nonsense = {"quantum", "purple"}
nonsense_idx = [vocab.index(w) for w in nonsense]
for k in (1, 2, 4, 8):
    p = top_k(probs, k)
    kept = ", ".join(vocab[i] for i in np.argsort(-probs)[:k])
    removed = probs[nonsense_idx].sum() - p[nonsense_idx].sum()
    print(f"{k:>4} {kept[:34]:>34} {removed:>21.5f}")

# Empirical draws converge to the distribution — the law of large numbers
# (Chapter 10), and the reason a sampled model is consistent in aggregate.
draws = rng.choice(len(vocab), size=200_000, p=probs)
emp = np.bincount(draws, minlength=len(vocab)) / len(draws)
print(f"\n{'token':<10} {'analytic p':>11} {'empirical':>11} {'abs diff':>10}")
for i, t in enumerate(vocab):
    print(f"{t:<10} {probs[i]:>11.4f} {emp[i]:>11.4f} "
          f"{abs(probs[i]-emp[i]):>10.5f}")
assert np.allclose(probs, emp, atol=0.01)

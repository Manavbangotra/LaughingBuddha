# -*- coding: utf-8 -*-
# Extracted from: Chapter 76 — BERT, RoBERTa, and Masked Language Modeling
# Source: src/.../ch076-bert.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""BERT's 80/10/10 corruption rule, and the property each branch buys."""
import numpy as np

rng = np.random.default_rng(0)

VOCAB = ["[PAD]", "[CLS]", "[SEP]", "[MASK]"] + [f"w{i}" for i in range(96)]
MASK_ID, SPECIAL = 3, {0, 1, 2, 3}
V, MASK_RATE = len(VOCAB), 0.15
IGNORE = -100          # positions excluded from the loss


def mask_tokens(ids, rng):
    """Returns the corrupted input and the label array. Equation (eq:bert-corruption)."""
    ids = np.asarray(ids)
    labels = np.full_like(ids, IGNORE)

    maskable = np.array([i for i, t in enumerate(ids) if t not in SPECIAL])
    n = max(1, int(round(MASK_RATE * len(maskable))))
    chosen = rng.choice(maskable, size=n, replace=False)

    labels[chosen] = ids[chosen]           # the loss is computed only here
    out = ids.copy()

    draw = rng.random(n)
    out[chosen[draw < 0.8]] = MASK_ID                                   # 80%
    mid = chosen[(draw >= 0.8) & (draw < 0.9)]
    out[mid] = rng.integers(len(SPECIAL), V, size=len(mid))             # 10%
    # the remaining 10% keep their original token, and are still labelled
    return out, labels


seq = [1] + list(rng.integers(4, V, size=18)) + [2]
corrupted, labels = mask_tokens(seq, rng)

print("original :", " ".join(VOCAB[t] for t in seq))
print("corrupted:", " ".join(VOCAB[t] for t in corrupted))
print("supervised positions:", int((labels != IGNORE).sum()), "of", len(seq))

# Verify the branch probabilities over many draws.
counts = {"mask": 0, "random": 0, "unchanged": 0, "total": 0}
for _ in range(4000):
    s = [1] + list(rng.integers(4, V, size=48)) + [2]
    c, l = mask_tokens(s, rng)
    for i in np.flatnonzero(l != IGNORE):
        counts["total"] += 1
        if c[i] == MASK_ID:
            counts["mask"] += 1
        elif c[i] == l[i]:
            counts["unchanged"] += 1
        else:
            counts["random"] += 1

t = counts["total"]
print(f"\nover {t:,} masked positions:")
for k, target in [("mask", 0.80), ("random", 0.10), ("unchanged", 0.10)]:
    print(f"  {k:<10} {counts[k] / t:6.3f}   (target {target:.2f})")

assert abs(counts["mask"] / t - 0.80) < 0.02
assert abs(counts["unchanged"] / t - 0.10) < 0.02

# The 12% figure from equation (eq:mask-distribution-shift).
print(f"\n[MASK] occupies {0.8 * MASK_RATE:.0%} of tokens in pretraining "
      f"and 0% at inference — the mismatch the rule mitigates but cannot remove.")

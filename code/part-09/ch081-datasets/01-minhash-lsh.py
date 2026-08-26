# -*- coding: utf-8 -*-
# Extracted from: Chapter 81 — Pretraining Dataset Construction and Curation
# Source: src/.../ch081-datasets.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Near-duplicate detection: MinHash signatures, LSH banding, measured."""
import hashlib
from collections import defaultdict

import numpy as np

rng = np.random.default_rng(0)
SHINGLE_K, NUM_HASHES = 5, 128
BANDS, ROWS = 32, 4                    # BANDS * ROWS must equal NUM_HASHES
assert BANDS * ROWS == NUM_HASHES

MERSENNE = (1 << 61) - 1
coeffs = rng.integers(1, MERSENNE, size=(NUM_HASHES, 2))


def shingles(text, k=SHINGLE_K):
    """Equation (eq:shingles), over whitespace tokens."""
    toks = text.split()
    if len(toks) < k:
        return {" ".join(toks)}
    return {" ".join(toks[i:i + k]) for i in range(len(toks) - k + 1)}


def base_hash(s):
    return int(hashlib.blake2b(s.encode(), digest_size=8).hexdigest(), 16)


def signature(text):
    """m independent min-hashes — equation (eq:minhash-property)."""
    sh = np.array([base_hash(s) for s in shingles(text)], dtype=np.uint64)
    if len(sh) == 0:
        return np.zeros(NUM_HASHES, dtype=np.uint64)
    a, b = coeffs[:, 0][:, None], coeffs[:, 1][:, None]
    hashed = (a * sh[None, :].astype(object) + b) % MERSENNE
    return np.array([min(row) for row in hashed], dtype=object)


def exact_jaccard(t1, t2):
    s1, s2 = shingles(t1), shingles(t2)
    return len(s1 & s2) / len(s1 | s2)


def estimated_jaccard(sig1, sig2):
    """Equation (eq:minhash-estimator)."""
    return float(np.mean([x == y for x, y in zip(sig1, sig2)]))


BASE = ("the quarterly report shows revenue growth across all regions with "
        "particular strength in the enterprise segment and continued expansion "
        "of the subscription business during the period under review")


def perturb(text, fraction):
    """Replace a fraction of tokens — a stand-in for editorial variation."""
    toks = text.split()
    n = int(len(toks) * fraction)
    idx = rng.choice(len(toks), size=n, replace=False)
    for i in idx:
        toks[i] = f"tok{rng.integers(0, 999)}"
    return " ".join(toks)


print("MinHash estimate against exact Jaccard\n")
print(f"{'perturbation':>13} {'exact J':>9} {'estimated':>11} {'error':>8}")
sig_base = signature(BASE)
for frac in (0.0, 0.05, 0.15, 0.30, 0.50, 0.80):
    variant = perturb(BASE, frac)
    ex = exact_jaccard(BASE, variant)
    es = estimated_jaccard(sig_base, signature(variant))
    print(f"{frac:>12.0%} {ex:>9.3f} {es:>11.3f} {abs(ex - es):>8.3f}")

print(f"\nestimator standard error at J=0.5, m={NUM_HASHES}: "
      f"{(0.5 * 0.5 / NUM_HASHES) ** 0.5:.4f}")


def lsh_buckets(signatures):
    """Band the signatures; documents sharing any band are candidates."""
    buckets = defaultdict(list)
    for doc_id, sig in signatures.items():
        for band in range(BANDS):
            key = (band, tuple(sig[band * ROWS:(band + 1) * ROWS]))
            buckets[key].append(doc_id)
    pairs = set()
    for members in buckets.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pairs.add(tuple(sorted((members[i], members[j]))))
    return pairs


# Build a corpus with a known duplicate structure.
docs, truth = {}, {}
docs["orig"] = BASE
for name, frac in [("edit-0.5%", 0.005), ("edit-1%", 0.01), ("edit-2%", 0.02),
                   ("edit-5%", 0.05), ("edit-10%", 0.10), ("edit-20%", 0.20),
                   ("edit-40%", 0.40)]:
    docs[name] = perturb(BASE, frac)
    truth[name] = exact_jaccard(BASE, docs[name])
docs["unrelated"] = " ".join(f"q{i % 200}" for i in range(400))
truth["unrelated"] = exact_jaccard(BASE, docs["unrelated"])

sigs = {k: signature(v) for k, v in docs.items()}
candidates = lsh_buckets(sigs)

print(f"\nLSH with b={BANDS}, r={ROWS} "
      f"(threshold ~ (1/b)^(1/r) = {(1 / BANDS) ** (1 / ROWS):.2f})\n")
print(f"{'document':>11} {'exact J':>9} {'flagged':>9} {'predicted P':>13}")
for name in ("edit-0.5%", "edit-1%", "edit-2%", "edit-5%", "edit-10%",
             "edit-20%", "edit-40%", "unrelated"):
    s = truth[name]
    flagged = ("orig", name) in candidates or (name, "orig") in candidates
    predicted = 1 - (1 - s ** ROWS) ** BANDS      # equation (eq:lsh-probability)
    print(f"{name:>11} {s:>9.3f} {str(flagged):>9} {predicted:>13.3f}")

print("\nThe S-curve does the work: high-similarity documents are caught almost "
      "always, low-similarity ones almost never, and the transition is sharp "
      "enough to separate a republished article from an independent one on the "
      "same topic.")

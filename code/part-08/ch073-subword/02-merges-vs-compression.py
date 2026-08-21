# Extracted from: Chapter 73 — Subword Tokenization: BPE, WordPiece, and SentencePiece
# Source: src/.../ch073-subword.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""How much compression does each additional merge buy?"""
from collections import Counter

CORPUS = """
the quick brown fox jumps over the lazy dog the quick brown fox
lowest lower low newest newer new widest wider wide
the lowest of the low and the newest of the new
tokenization tokenizer tokenized tokenizing token tokens
statistical machine learning learns statistics from machines
"""


# A minimal trainer, re-declared so this listing stands alone.
def _wf(t):
    return Counter(t.split())


def _pairs(sp, fr):
    p = Counter()
    for w, s in sp.items():
        for a, b in zip(s, s[1:]):
            p[(a, b)] += fr[w]
    return p


def _merge(sp, pair):
    a, b = pair
    out = {}
    for w, s in sp.items():
        m, i = [], 0
        while i < len(s):
            if i < len(s) - 1 and s[i] == a and s[i + 1] == b:
                m.append(a + b); i += 2
            else:
                m.append(s[i]); i += 1
        out[w] = m
    return out


def total_tokens(splits, freqs):
    return sum(len(s) * freqs[w] for w, s in splits.items())


freqs = _wf(CORPUS)
splits = {w: list(w) + ["_"] for w in freqs}
base = total_tokens(splits, freqs)
n_words = sum(freqs.values())
n_chars = sum(len(w) * f for w, f in freqs.items())

print(f"{'merges':>7} {'vocab':>7} {'tokens':>8} {'fertility':>10} "
      f"{'chars/token':>12} {'marginal':>9}")
prev = base
for step in range(0, 61):
    if step:
        p = _pairs(splits, freqs)
        if not p:
            break
        best = max(p, key=lambda k: (p[k], k))
        splits = _merge(splits, best)
    if step % 10 == 0:
        t = total_tokens(splits, freqs)
        marginal = (prev - t) / 10 if step else 0.0
        print(f"{step:>7} {len(set(s for v in splits.values() for s in v)):>7} "
              f"{t:>8} {t / n_words:>10.2f} {n_chars / t:>12.2f} "
              f"{marginal:>9.1f}")
        prev = t

print("\nThe marginal column is the point: the first merges buy a great deal "
      "and later ones buy progressively less, which is the curve that should "
      "decide the vocabulary size.")

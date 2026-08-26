# -*- coding: utf-8 -*-
# Extracted from: Chapter 73 — Subword Tokenization: BPE, WordPiece, and SentencePiece
# Source: src/.../ch073-subword.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Byte pair encoding: train a merge table, then apply it. Complete."""
from collections import Counter

CORPUS = """
the quick brown fox jumps over the lazy dog the quick brown fox
lowest lower low newest newer new widest wider wide
the lowest of the low and the newest of the new
tokenization tokenizer tokenized tokenizing token tokens
"""


def word_freqs(text):
    """Training operates on unique words with counts, not the raw stream."""
    return Counter(text.split())


def pair_counts(splits, freqs):
    """Count adjacent symbol pairs, weighted by the frequency of their word."""
    pairs = Counter()
    for word, symbols in splits.items():
        f = freqs[word]
        for a, b in zip(symbols, symbols[1:]):
            pairs[(a, b)] += f
    return pairs


def merge_pair(splits, pair):
    """Replace every adjacent occurrence of `pair` with the joined symbol."""
    a, b = pair
    joined = a + b
    out = {}
    for word, symbols in splits.items():
        merged, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                merged.append(joined)
                i += 2
            else:
                merged.append(symbols[i])
                i += 1
        out[word] = merged
    return out


def train_bpe(text, n_merges):
    """Returns the ordered merge list and the resulting vocabulary."""
    freqs = word_freqs(text)
    # '_' marks the end of a word, so merges cannot cross word boundaries and
    # a suffix at word-end is distinguishable from the same letters inside.
    splits = {w: list(w) + ["_"] for w in freqs}
    vocab = {s for symbols in splits.values() for s in symbols}
    merges = []
    for _ in range(n_merges):
        pairs = pair_counts(splits, freqs)
        if not pairs:
            break
        best = max(pairs, key=lambda p: (pairs[p], p))   # deterministic tie-break
        if pairs[best] < 2:
            break
        merges.append(best)
        vocab.add(best[0] + best[1])
        splits = merge_pair(splits, best)
    return merges, vocab


def encode(word, merges):
    """Replay the merges in the order they were learned. Order is load-bearing."""
    symbols = list(word) + ["_"]
    for a, b in merges:
        merged, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                merged.append(a + b)
                i += 2
            else:
                merged.append(symbols[i])
                i += 1
        symbols = merged
    return symbols


merges, vocab = train_bpe(CORPUS, n_merges=40)

print(f"vocabulary: {len(vocab)} symbols after {len(merges)} merges")
print(f"first ten merges: {[a + b for a, b in merges[:10]]}\n")

for w in ["token", "tokenizer", "tokenizers", "unbelievable", "lowest"]:
    pieces = encode(w, merges)
    print(f"{w:<14} -> {' '.join(pieces):<40} ({len(pieces)} tokens)")

# The critical property: a word never seen in training is still representable,
# because the character symbols are always in the vocabulary.
assert "unbelievable" not in CORPUS
assert "".join(encode("unbelievable", merges)).rstrip("_") == "unbelievable"
print("\nunseen word encodes losslessly — no UNK is possible")

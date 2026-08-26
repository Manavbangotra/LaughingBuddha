# -*- coding: utf-8 -*-
# Extracted from: Chapter 63 — Scaled Dot-Product Attention
# Source: src/.../ch063-scaled-dot-product-attention.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Attention over retrieved passages, using toy embeddings.

The point is not the embedding quality — real embeddings come from a trained
model (Part XI). The point is what the attention weights tell you when the
system gives a wrong answer.
"""
import numpy as np

rng = np.random.default_rng(7)


def softmax(z, axis=-1):
    e = np.exp(z - z.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


passages = [
    "Refunds are processed within 5 business days.",
    "Our head office is located in Bristol.",
    "Refund requests must be filed within 30 days of purchase.",
    "The support line is open 09:00 to 17:00.",
    "Shipping is free on orders over 50 pounds.",
]

# A crude embedding: bag of words over a small shared vocabulary. Real systems
# use a trained encoder; the attention arithmetic downstream is identical.
vocab = sorted({w.strip(".,:").lower()
                for p in passages + ["How long do refunds take?"]
                for w in p.split()})
index = {w: i for i, w in enumerate(vocab)}


def embed(text, d=32):
    """Bag-of-words counts, randomly projected to d dimensions and normalised."""
    counts = np.zeros(len(vocab))
    for w in text.split():
        w = w.strip(".,:").lower()
        if w in index:
            counts[index[w]] += 1
    proj = rng.normal(size=(len(vocab), d)) / np.sqrt(len(vocab))
    v = counts @ proj
    return v / (np.linalg.norm(v) + 1e-9)


d_k = 32
question = "How long do refunds take?"
Q = embed(question, d_k)[None, :]           # (1, d_k)  — one query
K = np.stack([embed(p, d_k) for p in passages])   # (5, d_k)
V = K.copy()                                       # values = keys, for clarity

scores = (Q @ K.T) / np.sqrt(d_k)
weights = softmax(scores)[0]

print(f"Question: {question}\n")
for w, p in sorted(zip(weights, passages), reverse=True):
    bar = "█" * int(round(w * 40))
    print(f"  {w:6.3f} {bar:<40} {p}")

print(f"\nAttention entropy: {-(weights * np.log(weights + 1e-12)).sum():.3f} "
      f"nats (max {np.log(len(passages)):.3f})")
print("Near-maximum entropy means the query failed to discriminate between")
print("passages — the retrieval is being ignored, not used. That is a")
print("diagnosable condition, and section 12 explains what causes it.")

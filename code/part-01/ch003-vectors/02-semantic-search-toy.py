# -*- coding: utf-8 -*-
# Extracted from: Chapter 3 — Vectors, Dot Products, and Geometric Intuition
# Source: src/.../ch003-vectors.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Ranking by dot product — the arithmetic underneath semantic search.

The embeddings here are hand-built so the geometry is inspectable. Real ones
come from a trained model (Part XI); the ranking arithmetic is identical.
"""
import numpy as np

# Three interpretable axes: [sport, finance, cooking]
docs = {
    "Match report: late winner":      np.array([9.0, 0.0, 0.0]),
    "Club posts record revenues":     np.array([6.0, 7.0, 0.0]),
    "Interest rates held steady":     np.array([0.0, 9.0, 0.0]),
    "Braising for beginners":         np.array([0.0, 0.0, 8.0]),
    "Stadium caterer wins award":     np.array([3.0, 1.0, 6.0]),
}
names = list(docs)
D = np.stack([docs[k] for k in names])          # (5, 3)

query = np.array([5.0, 4.0, 0.0])               # "football club finances"

# One matrix-vector product scores every document at once (Chapter 4).
scores = D @ query
order = np.argsort(-scores)

print("query: football club finances -> [sport=5, finance=4, cooking=0]\n")
print(f"{'score':>7}  {'cos':>6}  document")
for i in order:
    cos = scores[i] / (np.linalg.norm(D[i]) * np.linalg.norm(query))
    print(f"{scores[i]:>7.1f}  {cos:>6.3f}  {names[i]}")

print("\nNote the disagreement between the two columns:")
best_dot = names[int(np.argmax(scores))]
cosines = scores / (np.linalg.norm(D, axis=1) * np.linalg.norm(query))
best_cos = names[int(np.argmax(cosines))]
print(f"  highest dot product : {best_dot}")
print(f"  highest cosine      : {best_cos}")
print("The dot product rewards long vectors — documents that are simply")
print("'about more' — while the cosine judges direction alone. Which you want")
print("is a design decision, not a detail (Chapter 5).")

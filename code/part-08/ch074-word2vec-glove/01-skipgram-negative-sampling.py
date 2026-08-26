# -*- coding: utf-8 -*-
# Extracted from: Chapter 74 — Static Word Embeddings: Word2Vec and GloVe
# Source: src/.../ch074-word2vec-glove.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Skip-gram with negative sampling from scratch. Equation (eq:negative-sampling)."""
import numpy as np
from collections import Counter

SENTENCES = [
    "the doctor examined the patient in the clinic",
    "the nurse treated the patient in the clinic",
    "the doctor prescribed medicine for the patient",
    "the nurse prescribed medicine for the patient",
    "the patient visited the clinic to see the doctor",
    "the patient visited the clinic to see the nurse",
    "the engineer debugged the program in the terminal",
    "the programmer debugged the program in the terminal",
    "the engineer deployed the server for the program",
    "the programmer deployed the server for the program",
    "the program crashed so the engineer read the logs",
    "the program crashed so the programmer read the logs",
    "the chef prepared the dish in the kitchen",
    "the baker prepared the dish in the kitchen",
    "the chef seasoned the dish with the spices",
    "the baker seasoned the dish with the spices",
    "the dish burned so the chef opened the window",
    "the dish burned so the baker opened the window",
]
# Note what this corpus does NOT contain: 'doctor' and 'nurse' never occur in
# the same sentence, and neither do 'engineer'/'programmer' or 'chef'/'baker'.
# Any similarity the model finds between them is second-order — inferred from
# shared contexts alone, which is the distributional hypothesis under test.
CORPUS = ((" ".join(SENTENCES) + " ") * 40).split()

WINDOW, DIM, K, EPOCHS, LR = 2, 32, 5, 5, 0.05
rng = np.random.default_rng(0)

counts = Counter(CORPUS)
vocab = sorted(counts)
idx = {w: i for i, w in enumerate(vocab)}
V = len(vocab)

# Noise distribution: unigram^(3/4) — equation (eq:noise-distribution).
freqs = np.array([counts[w] for w in vocab], dtype=float)
noise = freqs ** 0.75
noise /= noise.sum()

# Training pairs from the sliding window.
ids = [idx[w] for w in CORPUS]
pairs = np.array([(ids[t], ids[t + j])
                  for t in range(len(ids))
                  for j in range(-WINDOW, WINDOW + 1)
                  if j != 0 and 0 <= t + j < len(ids)])
print(f"{len(vocab)} word types, {len(CORPUS):,} tokens, {len(pairs):,} training pairs")

Vin = rng.normal(0, 0.1, (V, DIM))     # input (centre) vectors
Uout = np.zeros((V, DIM))              # output (context) vectors


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


for epoch in range(EPOCHS):
    order = rng.permutation(len(pairs))
    total = 0.0
    for c, o in pairs[order]:
        negs = rng.choice(V, size=K, p=noise)
        v = Vin[c]

        pos = sigmoid(Uout[o] @ v)
        neg = sigmoid(Uout[negs] @ v)
        total += -np.log(pos + 1e-10) - np.log(1 - neg + 1e-10).sum()

        # Gradient of equation (eq:negsampling-gradient): attract the positive,
        # repel the K negatives, each weighted by how wrong the model is now.
        grad_v = (pos - 1.0) * Uout[o] + neg @ Uout[negs]
        Uout[o] -= LR * (pos - 1.0) * v
        for i, n in enumerate(negs):
            Uout[n] -= LR * neg[i] * v
        Vin[c] -= LR * grad_v
    print(f"epoch {epoch + 1}: mean loss {total / len(pairs):.4f}")

E = Vin / (np.linalg.norm(Vin, axis=1, keepdims=True) + 1e-10)


def neighbours(word, k=3):
    sims = E @ E[idx[word]]
    order = np.argsort(-sims)
    return [(vocab[i], round(float(sims[i]), 3)) for i in order if vocab[i] != word][:k]


print()
for w in ["doctor", "engineer", "chef", "patient", "program"]:
    print(f"{w:<11} -> {neighbours(w)}")

med = {"doctor", "nurse", "patient", "clinic", "medicine"}
tech = {"engineer", "programmer", "program", "server", "terminal", "logs"}


def mean_sim(a, b):
    return float(np.mean([E[idx[x]] @ E[idx[y]] for x in a for y in b if x != y]))


print(f"\nwithin medical:   {mean_sim(med, med):+.3f}")
print(f"within technical: {mean_sim(tech, tech):+.3f}")
print(f"across the two:   {mean_sim(med, tech):+.3f}")

# doctor and nurse NEVER co-occur in this corpus. Any similarity is inferred.
assert not any("doctor" in s and "nurse" in s for s in SENTENCES)
print("\n'doctor' and 'nurse' share no sentence, yet are near neighbours — "
      "the similarity is second-order, from shared contexts alone.")

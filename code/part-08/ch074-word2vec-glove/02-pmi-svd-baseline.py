# -*- coding: utf-8 -*-
# Extracted from: Chapter 74 — Static Word Embeddings: Word2Vec and GloVe
# Source: src/.../ch074-word2vec-glove.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Build the co-occurrence matrix, shift its PMI, factorise with SVD."""
import numpy as np

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

WINDOW, DIM, K = 2, 32, 5
vocab = sorted(set(CORPUS))
idx = {w: i for i, w in enumerate(vocab)}
V = len(vocab)

# Co-occurrence counts X_ij.
X = np.zeros((V, V))
ids = [idx[w] for w in CORPUS]
for t, c in enumerate(ids):
    for j in range(-WINDOW, WINDOW + 1):
        if j != 0 and 0 <= t + j < len(ids):
            X[c, ids[t + j]] += 1

D = X.sum()
row, col = X.sum(1, keepdims=True), X.sum(0, keepdims=True)

with np.errstate(divide="ignore", invalid="ignore"):
    pmi = np.log((X * D) / (row * col))
pmi[~np.isfinite(pmi)] = 0.0

# Shifted positive PMI — equation (eq:sgns-is-pmi) says SGNS targets PMI - log K.
sppmi = np.maximum(pmi - np.log(K), 0.0)

U, S, _ = np.linalg.svd(sppmi)
E = U[:, :DIM] * np.sqrt(S[:DIM])
E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-10


def neighbours(word, k=3):
    sims = E @ E[idx[word]]
    order = np.argsort(-sims)
    return [(vocab[i], round(float(sims[i]), 3)) for i in order if vocab[i] != word][:k]


print("Nearest neighbours from a truncated SVD of the shifted PMI matrix:")
for w in ["doctor", "engineer", "chef", "patient"]:
    print(f"  {w:<11} -> {neighbours(w)}")

print()
print(f"{'pair':<22} {'raw count':>10} {'PMI':>8}")
for a, b in [("doctor", "patient"), ("doctor", "the")]:
    print(f"{a + ' / ' + b:<22} {X[idx[a], idx[b]]:>10.0f} "
          f"{pmi[idx[a], idx[b]]:>+8.3f}")

print("\nRaw counts rank 'the' far above 'patient'; PMI divides out the marginal "
      "frequency and reverses the ranking. No gradient descent produced these "
      "vectors — only counting and an SVD.")

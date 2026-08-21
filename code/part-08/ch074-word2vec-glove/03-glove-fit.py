# Extracted from: Chapter 74 — Static Word Embeddings: Word2Vec and GloVe
# Source: src/.../ch074-word2vec-glove.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""GloVe: weighted least squares on log co-occurrence. Equation (eq:glove-objective)."""
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

WINDOW, DIM, XMAX, ALPHA, EPOCHS, LR = 2, 32, 100.0, 0.75, 300, 0.05
rng = np.random.default_rng(0)

vocab = sorted(set(CORPUS))
idx = {w: i for i, w in enumerate(vocab)}
V = len(vocab)
ids = [idx[w] for w in CORPUS]

X = np.zeros((V, V))
for t, c in enumerate(ids):
    for j in range(-WINDOW, WINDOW + 1):
        if j != 0 and 0 <= t + j < len(ids):
            X[c, ids[t + j]] += 1.0 / abs(j)      # GloVe weights by distance

nz_i, nz_j = np.nonzero(X)
nz_x = X[nz_i, nz_j]
print(f"co-occurrence matrix: {V}x{V} = {V * V} cells, "
      f"{len(nz_x)} nonzero ({100 * len(nz_x) / V ** 2:.1f}% dense)")

# f(x) from equation (eq:glove-weighting): caps frequent pairs, and f(0)=0
# excludes the zeros entirely — which is why only nonzeros are iterated.
w = np.minimum((nz_x / XMAX) ** ALPHA, 1.0)
logx = np.log(nz_x)

W = rng.normal(0, 0.1, (V, DIM))
Wt = rng.normal(0, 0.1, (V, DIM))
b, bt = np.zeros(V), np.zeros(V)

# AdaGrad, as in the GloVe paper: the per-parameter step size matters here
# because word frequencies span orders of magnitude, and plain SGD at a rate
# large enough for rare words diverges on frequent ones.
aW, aWt = np.ones_like(W), np.ones_like(Wt)
ab, abt = np.ones(V), np.ones(V)

for epoch in range(EPOCHS):
    pred = np.einsum("ij,ij->i", W[nz_i], Wt[nz_j]) + b[nz_i] + bt[nz_j]
    diff = pred - logx
    loss = float((w * diff ** 2).sum())

    g = (2 * w * diff)[:, None]
    dW, dWt = np.zeros_like(W), np.zeros_like(Wt)
    db, dbt = np.zeros(V), np.zeros(V)
    np.add.at(dW, nz_i, g * Wt[nz_j])
    np.add.at(dWt, nz_j, g * W[nz_i])
    np.add.at(db, nz_i, 2 * w * diff)
    np.add.at(dbt, nz_j, 2 * w * diff)

    aW += dW ** 2; aWt += dWt ** 2; ab += db ** 2; abt += dbt ** 2
    W -= LR * dW / np.sqrt(aW)
    Wt -= LR * dWt / np.sqrt(aWt)
    b -= LR * db / np.sqrt(ab)
    bt -= LR * dbt / np.sqrt(abt)

    if epoch % 100 == 0 or epoch == EPOCHS - 1:
        print(f"epoch {epoch:>4}: weighted squared error {loss:9.2f}")

E = W + Wt                       # the standard choice: sum the two matrices
E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-10

print()
for word in ["doctor", "engineer", "chef"]:
    sims = E @ E[idx[word]]
    order = [i for i in np.argsort(-sims) if vocab[i] != word][:3]
    print(f"{word:<11} -> {[(vocab[i], round(float(sims[i]), 3)) for i in order]}")

print("\nSame corpus, same neighbourhoods, a least-squares fit to log counts "
      "instead of sampled gradient steps.")

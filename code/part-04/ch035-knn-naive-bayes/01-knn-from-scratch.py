# Extracted from: Chapter 35 — k-Nearest Neighbors and Naive Bayes
# Source: src/.../ch035-knn-naive-bayes.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""k-NN from scratch: the k knob, scaling, metrics, and the curse.
"""
import numpy as np

rng = np.random.default_rng(0)


def knn_predict(Xtr, ytr, Xte, k=5, metric="euclidean", weighted=False):
    """Brute-force k-NN (eq. 35.1). O(N_te * N_tr * D) — deliberately."""
    if metric == "euclidean":
        d = np.sqrt(((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1))
    elif metric == "manhattan":
        d = np.abs(Xte[:, None, :] - Xtr[None, :, :]).sum(-1)
    elif metric == "cosine":
        A = Xte / np.maximum(np.linalg.norm(Xte, axis=1, keepdims=True), 1e-12)
        B = Xtr / np.maximum(np.linalg.norm(Xtr, axis=1, keepdims=True), 1e-12)
        d = 1.0 - A @ B.T
    else:
        raise ValueError(metric)

    idx = np.argpartition(d, min(k, d.shape[1] - 1), axis=1)[:, :k]
    lab = ytr[idx]
    if not weighted:
        return (lab.mean(axis=1) >= 0.5).astype(int)
    w = 1.0 / np.maximum(np.take_along_axis(d, idx, axis=1), 1e-12)
    return ((lab * w).sum(1) / w.sum(1) >= 0.5).astype(int)


def make_moons(n, noise=0.25):
    """Two interleaving crescents: not linearly separable, locally smooth."""
    t = rng.uniform(0, np.pi, n)
    top = np.column_stack([np.cos(t), np.sin(t)])
    bot = np.column_stack([1 - np.cos(t), 0.5 - np.sin(t)])
    X = np.vstack([top, bot]) + rng.normal(0, noise, (2 * n, 2))
    y = np.r_[np.zeros(n, int), np.ones(n, int)]
    p = rng.permutation(len(y))
    return X[p], y[p]


X, y = make_moons(600)
cut = 800
Xtr, ytr, Xte, yte = X[:cut], y[:cut], X[cut:], y[cut:]

# --- k is a bias-variance knob (Chapter 34) ---------------------------------
print("=" * 72)
print("k moves directly along the bias-variance curve")
print("=" * 72)
print(f"{'k':>5} {'train acc':>11} {'test acc':>10} {'gap':>8}")
for k in (1, 3, 5, 11, 25, 75, 201, len(ytr)):
    tr = (knn_predict(Xtr, ytr, Xtr, k) == ytr).mean()
    te = (knn_predict(Xtr, ytr, Xte, k) == yte).mean()
    print(f"{k:>5} {tr:>11.4f} {te:>10.4f} {tr - te:>8.4f}")
print("\nk=1 gets every training point right by construction — it is its own")
print("nearest neighbour — and the gap is pure variance. At k=N every")
print("prediction is the majority class: pure bias. Test accuracy peaks in")
print("between, which is the whole of Chapter 34 in one column.")

# --- distance weighting -----------------------------------------------------
print(f"\n{'k':>5} {'uniform':>9} {'distance-weighted':>19} {'difference':>12}")
for k in (1, 5, 25, 101):
    u = (knn_predict(Xtr, ytr, Xte, k) == yte).mean()
    w = (knn_predict(Xtr, ytr, Xte, k, weighted=True) == yte).mean()
    print(f"{k:>5} {u:>9.4f} {w:>19.4f} {w - u:>+12.4f}")
print("Distance weighting is often described as a clear improvement. On")
print("this data it is worth a quarter of a point at k=5 and slightly")
print("NEGATIVE at k=25 and above — all of these differences are within the")
print("noise of a 400-row test set. Its real value is making the prediction")
print("continuous in x, which matters for regression and for ranking; treat")
print("accuracy gains as a hypothesis to test, not a given.")

# --- scaling is not optional ------------------------------------------------
print("\n" + "=" * 72)
print("what happens when one feature is measured in different units")
print("=" * 72)
print(f"{'scale of feature 2':>20} {'raw k-NN':>10} {'standardised':>14}")
for scale in (1, 10, 100, 10000):
    Xs = X.copy()
    Xs[:, 1] *= scale
    a, b = Xs[:cut], Xs[cut:]
    raw = (knn_predict(a, ytr, b, 11) == yte).mean()
    mu, sd = a.mean(0), a.std(0)
    std = (knn_predict((a - mu) / sd, ytr, (b - mu) / sd, 11) == yte).mean()
    print(f"{scale:>20,} {raw:>10.4f} {std:>14.4f}")
print("\nMultiplying one column by 10,000 does not change the information in")
print("the data at all, and destroys raw k-NN: the distance is now entirely")
print("that one feature. Standardisation is immune. This is the single most")
print("common way to get k-NN silently wrong.")

# --- section 6.2: the curse, measured ---------------------------------------
print("\n" + "=" * 72)
print("distance concentration (eq. 35.6, 35.7)")
print("=" * 72)
print(f"{'D':>5} {'mean dist':>11} {'sd/mean':>9} "
      f"{'(dmax-dmin)/dmin':>18} {'side for 1% (eq 35.5)':>23}")
for D in (2, 5, 20, 100, 1000):
    P = rng.uniform(0, 1, (600, D))
    q = rng.uniform(0, 1, (40, D))
    d = np.sqrt(((q[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    ratio = np.mean((d.max(1) - d.min(1)) / d.min(1))
    print(f"{D:>5} {d.mean():>11.3f} {d.std() / d.mean():>9.4f} "
          f"{ratio:>18.4f} {0.01 ** (1 / D):>23.4f}")
print("\nAt D=1000 the farthest point is 12% further than the nearest, and a")
print("'neighbourhood' holding 1% of the data spans 99.5% of every axis.")
print("'Nearest neighbour' has stopped meaning anything.")

# --- and what that costs in accuracy ----------------------------------------
print("\n" + "=" * 72)
print("the accuracy cost of adding PURE NOISE features")
print("=" * 72)
print(f"{'noise dims added':>18} {'total D':>9} {'k-NN acc':>10} "
      f"{'logistic acc':>14}")


def fit_logistic_simple(A, b, B, n_iter=400, lr=0.4):
    A1 = np.column_stack([np.ones(len(A)), A])
    B1 = np.column_stack([np.ones(len(B)), B])
    w = np.zeros(A1.shape[1])
    for _ in range(n_iter):
        p = 1 / (1 + np.exp(-np.clip(A1 @ w, -30, 30)))
        w -= lr * (A1.T @ (p - b) / len(b))
    return (1 / (1 + np.exp(-np.clip(B1 @ w, -30, 30))) >= 0.5).astype(int)


for n_noise in (0, 2, 5, 10, 30, 100):
    Xn = np.column_stack([X, rng.normal(size=(len(X), n_noise))]) \
        if n_noise else X
    a, b_ = Xn[:cut], Xn[cut:]
    mu, sd = a.mean(0), a.std(0)
    a, b_ = (a - mu) / sd, (b_ - mu) / sd
    acc = (knn_predict(a, ytr, b_, 11) == yte).mean()
    lacc = (fit_logistic_simple(a, ytr.astype(float), b_) == yte).mean()
    print(f"{n_noise:>18} {Xn.shape[1]:>9} {acc:>10.4f} {lacc:>14.4f}")

print("\nThe two informative features are untouched throughout; everything")
print("added is independent noise. k-NN degrades steadily because the noise")
print("dimensions dominate the distance. This is why feature selection")
print("(Chapter 27) matters far more for distance-based methods than for")
print("models that can learn to ignore a feature by giving it a small")
print("coefficient.")

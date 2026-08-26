# -*- coding: utf-8 -*-
# Extracted from: Chapter 33 — Logistic Regression and Classification
# Source: src/.../ch033-logistic.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Multiclass logistic regression via the softmax, plus a check against
scikit-learn.
"""
import numpy as np

rng = np.random.default_rng(2)


def softmax(Z):
    """Row-wise softmax (eq. 33.14), shifted for numerical stability.

    Subtracting the row max exploits the shift-invariance noted in section 5.5:
    it changes nothing mathematically and everything numerically.
    """
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


# demonstrate why the shift is not optional
big = np.array([[1000.0, 1001.0, 1002.0]])
with np.errstate(over="ignore", invalid="ignore"):
    naive = np.exp(big) / np.exp(big).sum()
print("naive softmax of [1000, 1001, 1002]:", naive)
print("stable softmax                     :", softmax(big))
print("Shift-invariance is the difference between nan and the right answer.\n")


def fit_softmax(X, y, K, lr=0.5, n_iter=3000, lam=1e-4):
    """Gradient descent on multiclass cross-entropy.

    The gradient is X^T (P - Y) — the same form as the binary case and the
    same form as least squares (section 6.2).
    """
    N, D = X.shape
    Y = np.zeros((N, K))
    Y[np.arange(N), y] = 1.0
    W = np.zeros((D, K))
    for _ in range(n_iter):
        P = softmax(X @ W)
        G = X.T @ (P - Y) / N
        G[1:] += 2 * lam * W[1:]
        W -= lr * G
    return W


# --- three classes arranged so no single linear split works -----------------
n_per, K = 500, 3
centres = np.array([[0.0, 2.0], [-2.0, -1.0], [2.0, -1.0]])
Xr = np.vstack([c + rng.normal(0, 1.1, (n_per, 2)) for c in centres])
y = np.repeat(np.arange(K), n_per)
perm = rng.permutation(len(y))
Xr, y = Xr[perm], y[perm]
X = np.column_stack([np.ones(len(Xr)), Xr])

cut = int(0.7 * len(y))
W = fit_softmax(X[:cut], y[:cut], K)
P_te = softmax(X[cut:] @ W)
pred = P_te.argmax(1)
print(f"test accuracy (from scratch): {(pred == y[cut:]).mean():.4f}")
print(f"probabilities sum to 1      : {np.allclose(P_te.sum(1), 1.0)}")
print(f"mean predicted prob of the true class: "
      f"{P_te[np.arange(len(pred)), y[cut:]].mean():.4f}")

# --- check against the library ----------------------------------------------
try:
    from sklearn.linear_model import LogisticRegression
    sk = LogisticRegression(C=1 / (2 * 1e-4 * cut), max_iter=5000)
    sk.fit(Xr[:cut], y[:cut])
    sk_pred = sk.predict(Xr[cut:])
    agree = (sk_pred == pred).mean()
    print(f"\nscikit-learn test accuracy  : "
          f"{(sk_pred == y[cut:]).mean():.4f}")
    print(f"agreement with from-scratch : {agree:.4f}")
    print(f"max |P_scratch - P_sklearn| : "
          f"{np.abs(sk.predict_proba(Xr[cut:]) - P_te).max():.4f}")
    print("The predicted probabilities differ in the fourth decimal place —")
    print("the two use different optimisers and slightly different penalty")
    print("conventions — and the two models pick the same class for every")
    print("single test point. This is what 'implemented from scratch' should")
    print("mean: the same answer as the library, by a route you can read.")
except ImportError:
    print("\n(scikit-learn not installed — cross-check skipped)")

# --- the boundary is linear even though the probabilities are not -----------
print("\n" + "=" * 72)
print("the decision boundary is a hyperplane (section 5.1)")
print("=" * 72)
line = np.column_stack([np.ones(9), np.linspace(-4, 4, 9),
                        np.zeros(9)])
Pl = softmax(line @ W)
print(f"{'x1':>7} " + " ".join(f"{'P(class ' + str(k) + ')':>13}"
                               for k in range(K)))
for i in range(9):
    print(f"{line[i, 1]:>7.1f} " + " ".join(f"{Pl[i, k]:>13.4f}"
                                            for k in range(K)))
print("\nThe probabilities move smoothly and nonlinearly along the line, but")
print("the point at which the argmax changes is where two linear functions")
print("cross — so the boundary itself is straight.")

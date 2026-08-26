# -*- coding: utf-8 -*-
# Extracted from: Chapter 5 — Norms, Distances, and Similarity Measures
# Source: src/.../ch005-norms.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Norms, distances, and similarity — with the chapter's claims checked.

Includes the L1-versus-L2 sparsity effect, demonstrated rather than asserted.
"""
import numpy as np

x = np.array([3.0, -4.0])

print(f"{'norm':<14} {'value':>8}")
print(f"{'L1':<14} {np.linalg.norm(x, 1):>8.4f}")
print(f"{'L2':<14} {np.linalg.norm(x, 2):>8.4f}")
print(f"{'L-infinity':<14} {np.linalg.norm(x, np.inf):>8.4f}")
assert np.linalg.norm(x, np.inf) <= np.linalg.norm(x, 2) <= np.linalg.norm(x, 1)

# --- the norm axioms, checked on random vectors (eqs. 5.2-5.5) --------------
rng = np.random.default_rng(0)
for p in (1, 2, np.inf):
    for _ in range(500):
        a, b = rng.normal(size=6), rng.normal(size=6)
        c = rng.normal()
        assert np.linalg.norm(a, p) >= 0
        assert np.isclose(np.linalg.norm(c * a, p),
                          abs(c) * np.linalg.norm(a, p))            # homogeneity
        assert (np.linalg.norm(a + b, p)
                <= np.linalg.norm(a, p) + np.linalg.norm(b, p) + 1e-9)  # triangle
print("\nnorm axioms hold for p = 1, 2, inf on 1500 random cases")

# The squared L2 "norm" is not one: it fails homogeneity.
a = rng.normal(size=6)
print(f"||2a||^2 = {np.sum((2*a)**2):.4f} but 2*||a||^2 = {2*np.sum(a**2):.4f}"
      "  <- squared L2 is not a norm")

# --- concentration: why L1 and L2 disagree ----------------------------------
spread = np.array([7.071, 7.071])
concentrated = np.array([10.0, 0.0])
print(f"\n{'vector':<22} {'L1':>8} {'L2':>8}")
for name, v in (("spread [7.07, 7.07]", spread), ("concentrated [10, 0]", concentrated)):
    print(f"{name:<22} {np.linalg.norm(v,1):>8.3f} {np.linalg.norm(v,2):>8.3f}")
print("Equal L2, different L1: L1 charges less for concentrating magnitude.")

# --- eq. 5.11: cosine and Euclidean coincide on normalised vectors ----------
def cosine(a, b):
    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))


u, v = rng.normal(size=64), rng.normal(size=64)
un, vn = u / np.linalg.norm(u), v / np.linalg.norm(v)
lhs = np.linalg.norm(un - vn) ** 2
rhs = 2 - 2 * cosine(u, v)
print(f"\n||u_hat - v_hat||^2 = {lhs:.10f}")
print(f"2 - 2 cos(theta)    = {rhs:.10f}   <- eq. 5.11")
assert np.isclose(lhs, rhs)

# Consequently, ranking by one equals ranking by the other — but only once
# normalised.
docs = rng.normal(size=(200, 64))
q = rng.normal(size=64)
docs_n = docs / np.linalg.norm(docs, axis=1, keepdims=True)
q_n = q / np.linalg.norm(q)
rank_cos = np.argsort(-(docs_n @ q_n))
rank_l2 = np.argsort(np.linalg.norm(docs_n - q_n, axis=1))
assert np.array_equal(rank_cos, rank_l2)
print("normalised: cosine ranking == Euclidean ranking (identical order)")

rank_cos_raw = np.argsort(-(docs @ q / (np.linalg.norm(docs, axis=1) * np.linalg.norm(q))))
rank_dot_raw = np.argsort(-(docs @ q))
agree = np.mean(rank_cos_raw[:10] == rank_dot_raw[:10])
print(f"UNnormalised: cosine and dot-product top-10 agree on only "
      f"{agree:.0%} of positions")

# --- the three measures disagree about the same pair (section 6.3) ----------
a2, b2 = np.array([3.0, 4.0]), np.array([6.0, 8.0])
print(f"\nx = {a2}, y = {b2}  (y = 2x, so identical direction)")
print(f"  euclidean distance : {np.linalg.norm(a2 - b2):.3f}   <- 'far apart'")
print(f"  dot product        : {a2 @ b2:.3f}  <- large but uninterpretable")
print(f"  cosine similarity  : {cosine(a2, b2):.3f}   <- 'identical'")

# --- section 6.2: L1 produces sparsity, L2 does not -------------------------
# Fit y = Xw with a penalty, by plain gradient descent, and count exact zeros.
n, d = 200, 60
X = rng.normal(size=(n, d))
w_true = np.zeros(d)
w_true[:5] = [3.0, -2.0, 1.5, 4.0, -1.0]     # only 5 of 60 features matter
y = X @ w_true + 0.1 * rng.normal(size=n)


def fit(penalty, lam, steps=4000, lr=2e-3):
    w = np.zeros(d)
    for _ in range(steps):
        grad = X.T @ (X @ w - y) / n
        if penalty == "l2":
            w -= lr * (grad + 2 * lam * w)
        else:
            # Proximal step: gradient on the loss, then soft-threshold. The
            # threshold is what can set a coefficient to EXACTLY zero; a plain
            # subgradient step only ever approaches zero.
            w -= lr * grad
            w = np.sign(w) * np.maximum(np.abs(w) - lr * lam, 0.0)
    return w


for penalty, lam in (("l2", 0.05), ("l1", 0.30)):
    w = fit(penalty, lam)
    exact_zeros = int(np.sum(np.abs(w) == 0.0))
    tiny = int(np.sum(np.abs(w) < 1e-3))
    print(f"\n{penalty.upper()} penalty: {exact_zeros}/{d} coefficients are "
          f"EXACTLY zero, {tiny}/{d} are below 1e-3")
    print(f"  recovered first 5 (true {w_true[:5]}):")
    print(f"  {np.round(w[:5], 3)}")

print("\nL2 shrinks every coefficient toward zero without reaching it;")
print("L1 sets most of them to exactly zero. The difference is the corner")
print("on the L1 ball, and it is why L1 is used for feature selection.")

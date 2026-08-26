# -*- coding: utf-8 -*-
# Extracted from: Chapter 28 — Data Leakage, Imbalanced Data, and Dataset Pathologies
# Source: src/.../ch028-leakage.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The four leakage mechanisms, each demonstrated with its cost measured.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)


def auc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos, neg = labels.sum(), (1 - labels).sum()
    if pos == 0 or neg == 0:
        return 0.5
    return (ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)


def fit_logistic(X, y, steps=300, lr=0.4):
    X = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(X.shape[1])
    for _ in range(steps):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        w -= lr * (X.T @ (p - y) / len(y))
    return w


def predict(X, w):
    X = np.column_stack([np.ones(len(X)), X])
    return 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))


# --- 1. target leakage --------------------------------------------------------
print("=" * 72)
print("1. TARGET LEAKAGE: a feature that is a consequence of the outcome")
print("=" * 72)
n = 8000
tenure = rng.normal(0, 1, n)
churn = (rng.random(n) < 1 / (1 + np.exp(-(-0.5 - 0.8 * tenure)))).astype(int)
# Only populated for churned users — it exists BECAUSE they churned.
cancel_survey = np.where(churn, rng.normal(3, 1, n), np.nan)
has_survey = (~np.isnan(cancel_survey)).astype(float)

split = n // 2
for label, cols in (("legitimate features only", [tenure]),
                    ("+ cancellation-survey flag", [tenure, has_survey])):
    X = np.column_stack(cols)
    w = fit_logistic(X[:split], churn[:split])
    print(f"  {label:<30} test AUC {auc(predict(X[split:], w), churn[split:]):.3f}")
print("  The second is near-perfect and worthless: the flag only exists after")
print("  the user has already churned (eq. 28.1).")

# --- 2. temporal leakage ------------------------------------------------------
print("\n" + "=" * 72)
print("2. TEMPORAL LEAKAGE: shuffling time-ordered data")
print("=" * 72)
T = 4000
t = np.arange(T)
level = np.cumsum(rng.normal(0, 1, T))              # a random walk
feature = level + rng.normal(0, 0.4, T)
y_t = (np.diff(level, prepend=level[0]) > 0).astype(int)

# WRONG: shuffle, then split.
idx = rng.permutation(T)
tr, te = idx[:T//2], idx[T//2:]
w = fit_logistic(feature[tr, None], y_t[tr])
shuffled_auc = auc(predict(feature[te, None], w), y_t[te])

# RIGHT: train on the past, test on the future.
w = fit_logistic(feature[:T//2, None], y_t[:T//2])
temporal_auc = auc(predict(feature[T//2:, None], w), y_t[T//2:])

print(f"  random shuffle split : AUC {shuffled_auc:.3f}")
print(f"  time-ordered split   : AUC {temporal_auc:.3f}")
print("  Shuffling lets the model interpolate between surrounding time points,")
print("  which it can never do in production.")

# --- 3. group leakage ---------------------------------------------------------
print("\n" + "=" * 72)
print("3. GROUP LEAKAGE: the same entity in train and test")
print("=" * 72)


def knn_predict(Xtr, ytr, Xte, k=3):
    """1-3 nearest neighbours. A model with the CAPACITY to memorise, which is
    what makes group leakage visible — a linear model cannot memorise and so
    cannot exhibit this leak at all."""
    d = np.abs(Xte[:, None, 0] - Xtr[None, :, 0])
    nn = np.argsort(d, axis=1)[:, :k]
    return ytr[nn].mean(axis=1)


n_users, per_user = 500, 10
user = np.repeat(np.arange(n_users), per_user)

# The feature identifies WHICH user a row belongs to (rows from one user
# cluster tightly), but carries no information about the label. The label
# depends on a per-user tendency that is NOT in the feature.
user_position = rng.normal(0, 5.0, n_users)          # visible via x
user_tendency = rng.normal(0, 2.0, n_users)          # hidden, drives y
x_g = user_position[user] + rng.normal(0, 0.05, len(user))
y_g = (rng.random(len(user)) <
       1 / (1 + np.exp(-user_tendency[user]))).astype(int)

# WRONG: split rows at random — the same user lands on both sides.
perm = rng.permutation(len(user))
tr, te = perm[:len(user) // 2], perm[len(user) // 2:]
row_auc = auc(knn_predict(x_g[tr, None], y_g[tr], x_g[te, None]), y_g[te])
overlap = len(set(user[tr]) & set(user[te])) / n_users

# RIGHT: split by user, so no user appears on both sides.
u_perm = rng.permutation(n_users)
train_users = set(u_perm[:n_users // 2].tolist())
mask = np.isin(user, list(train_users))
group_auc = auc(knn_predict(x_g[mask, None], y_g[mask], x_g[~mask, None]),
                y_g[~mask])

print(f"  random row split : AUC {row_auc:.3f}  "
      f"({overlap:.0%} of users appear in BOTH sides)")
print(f"  grouped split    : AUC {group_auc:.3f}  (0% overlap)")
print(f"  optimism from group leakage: {row_auc - group_auc:+.3f}")
print("  The feature carries NO information about the label — it only says")
print("  which user a row came from. The random split scores well anyway, by")
print("  looking up other rows from the same user. The grouped split is at")
print("  chance, which is the truth (Chapter 26's unit argument).")

# --- 4. preprocessing leakage, and the shuffled-target test -----------------
print("\n" + "=" * 72)
print("4. PREPROCESSING LEAKAGE — caught by the shuffled-target test")
print("=" * 72)
m, p = 400, 3000                                # wide data, no real signal
X_noise = rng.normal(size=(m, p))
y_noise = (rng.random(m) < 0.5).astype(int)

# WRONG: select the most correlated features using ALL the data, then split.
corrs = np.array([abs(np.corrcoef(X_noise[:, j], y_noise)[0, 1])
                  for j in range(p)])
top = np.argsort(-corrs)[:15]
Xs = X_noise[:, top]
w = fit_logistic(Xs[:m//2], y_noise[:m//2])
leaky_auc = auc(predict(Xs[m//2:], w), y_noise[m//2:])

# RIGHT: select inside the training half only.
tr_c = np.array([abs(np.corrcoef(X_noise[:m//2, j], y_noise[:m//2])[0, 1])
                 for j in range(p)])
top_tr = np.argsort(-tr_c)[:15]
w = fit_logistic(X_noise[:m//2][:, top_tr], y_noise[:m//2])
honest_auc = auc(predict(X_noise[m//2:][:, top_tr], w), y_noise[m//2:])

print(f"  the target is a coin flip and the features are pure noise.")
print(f"  select on ALL data, then split : AUC {leaky_auc:.3f}   <- fabricated")
print(f"  select inside train only       : AUC {honest_auc:.3f}   <- chance")
print(f"\n  eq. 28.5: a shuffled target must give AUC 0.5. The leaky procedure")
print(f"  scores {leaky_auc:.3f} on data with no relationship at all, which")
print(f"  proves the PIPELINE is leaking rather than any single feature.")

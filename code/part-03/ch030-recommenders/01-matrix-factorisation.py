# Extracted from: Chapter 30 — Recommendation Systems
# Source: src/.../ch030-recommenders.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Matrix factorisation from scratch, and why the SVD cannot be used directly.
"""
import numpy as np

rng = np.random.default_rng(0)

# --- a synthetic ratings matrix with known latent structure -----------------
n_users, n_items, k_true = 800, 400, 6
P_true = rng.normal(0, 0.7, (n_users, k_true))
Q_true = rng.normal(0, 0.7, (n_items, k_true))
bu_true = rng.normal(0, 0.4, n_users)
bi_true = rng.normal(0, 0.5, n_items)
MU = 3.6
full = MU + bu_true[:, None] + bi_true[None, :] + P_true @ Q_true.T
full = np.clip(full + rng.normal(0, 0.25, full.shape), 1, 5)

# Observe only ~4% of cells, as a real system would.
density = 0.04
mask = rng.random(full.shape) < density
rows, cols = np.where(mask)
ratings = full[rows, cols]
print(f"matrix {n_users}x{n_items} = {full.size:,} cells")
print(f"observed: {mask.sum():,} ({mask.mean():.1%})")

# hold out 20% of the OBSERVED entries
perm = rng.permutation(len(ratings))
cut = int(0.8 * len(ratings))
tr, te = perm[:cut], perm[cut:]


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


# --- baselines ---------------------------------------------------------------
print("\n" + "=" * 72)
print("baselines first")
print("=" * 72)
mu = ratings[tr].mean()
print(f"global mean only          RMSE {rmse(mu, ratings[te]):.4f}")

# bias-only model, fitted by alternating averages
bu = np.zeros(n_users)
bi = np.zeros(n_items)
for _ in range(15):
    resid = ratings[tr] - mu - bi[cols[tr]]
    for u in range(n_users):
        m = rows[tr] == u
        bu[u] = resid[m].sum() / (m.sum() + 8) if m.any() else 0.0
    resid = ratings[tr] - mu - bu[rows[tr]]
    for i in range(n_items):
        m = cols[tr] == i
        bi[i] = resid[m].sum() / (m.sum() + 8) if m.any() else 0.0
bias_pred = np.clip(mu + bu[rows[te]] + bi[cols[te]], 1, 5)
print(f"biases only (eq. 30.2)    RMSE {rmse(bias_pred, ratings[te]):.4f}")

# --- eq. 30.5/30.6: matrix factorisation by SGD -----------------------------
print("\n" + "=" * 72)
print("matrix factorisation (eqs. 30.5-30.6)")
print("=" * 72)


def fit_mf(k=10, epochs=40, lr=0.012, lam=0.06):
    P = rng.normal(0, 0.05, (n_users, k))
    Q = rng.normal(0, 0.05, (n_items, k))
    bu_ = np.zeros(n_users)
    bi_ = np.zeros(n_items)
    order = tr.copy()
    for ep in range(epochs):
        rng.shuffle(order)
        for idx in order:
            u, i, r = rows[idx], cols[idx], ratings[idx]
            pred = mu + bu_[u] + bi_[i] + P[u] @ Q[i]
            e = r - pred
            bu_[u] += lr * (e - lam * bu_[u])
            bi_[i] += lr * (e - lam * bi_[i])
            pu = P[u].copy()
            P[u] += lr * (e * Q[i] - lam * P[u])
            Q[i] += lr * (e * pu - lam * Q[i])
    return P, Q, bu_, bi_


print(f"{'k':>4} {'train RMSE':>12} {'test RMSE':>11}")
for k in (2, 6, 12, 30):
    P, Q, bu_, bi_ = fit_mf(k=k)
    def predict(idx):
        return np.clip(mu + bu_[rows[idx]] + bi_[cols[idx]]
                       + np.sum(P[rows[idx]] * Q[cols[idx]], axis=1), 1, 5)
    print(f"{k:>4} {rmse(predict(tr), ratings[tr]):>12.4f} "
          f"{rmse(predict(te), ratings[te]):>11.4f}")
print(f"\ntrue latent dimension is {k_true}. Beyond it, train error keeps")
print("falling and test error does not — the regularisation of eq. 30.3 is")
print("what stops it diverging entirely.")

# --- section 4.4: why the SVD cannot be used directly -----------------------
print("\n" + "=" * 72)
print("why not just take the SVD? (section 4.4)")
print("=" * 72)

k = 6
filled_zero = np.zeros_like(full)
filled_zero[rows[tr], cols[tr]] = ratings[tr]
filled_mean = np.full_like(full, mu)
filled_mean[rows[tr], cols[tr]] = ratings[tr]

for label, M in (("fill missing with 0", filled_zero),
                 ("fill missing with the mean", filled_mean)):
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    approx = (U[:, :k] * s[:k]) @ Vt[:k]
    pred = np.clip(approx[rows[te], cols[te]], 1, 5)
    print(f"  {label:<28} test RMSE {rmse(pred, ratings[te]):.4f}")

P, Q, bu_, bi_ = fit_mf(k=6)
mf_pred = np.clip(mu + bu_[rows[te]] + bi_[cols[te]]
                  + np.sum(P[rows[te]] * Q[cols[te]], axis=1), 1, 5)
print(f"  {'MF on observed entries only':<28} test RMSE "
      f"{rmse(mf_pred, ratings[te]):.4f}")
print("\nFilling makes the SVD applicable and wrong: it spends its capacity")
print("reproducing invented values. Fitting the observed cells only is a")
print("different, non-convex problem (eq. 30.7) with no closed form.")

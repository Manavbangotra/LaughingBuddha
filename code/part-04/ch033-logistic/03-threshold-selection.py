# -*- coding: utf-8 -*-
# Extracted from: Chapter 33 — Logistic Regression and Classification
# Source: src/.../ch033-logistic.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Credit-default scoring: the threshold is a business decision, not 0.5.
"""
import numpy as np

rng = np.random.default_rng(11)
n = 12000


def sigmoid(z):
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1 / (1 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1 + e)
    return out


# --- a deliberately imbalanced problem --------------------------------------
income = rng.lognormal(10.5, 0.5, n)
utilisation = np.clip(rng.beta(2, 5, n), 0, 1)
n_late = rng.poisson(0.4, n)
years = rng.uniform(0, 25, n)

z = (-3.2
     - 0.9 * (np.log(income) - 10.5)
     + 3.0 * utilisation
     + 0.55 * n_late
     - 0.04 * years)
y = (rng.random(n) < sigmoid(z)).astype(float)
print(f"default rate: {y.mean():.4f}  ({int(y.sum())} of {n})")

Xr = np.column_stack([np.log(income), utilisation, n_late, years])
mu, sd = Xr[:8000].mean(0), Xr[:8000].std(0)
Xs = (Xr - mu) / sd
X = np.column_stack([np.ones(n), Xs])
Xtr, ytr, Xte, yte = X[:8000], y[:8000], X[8000:], y[8000:]


def fit_newton(X, y, lam=1e-4, n_iter=50):
    w = np.zeros(X.shape[1])
    for _ in range(n_iter):
        p = sigmoid(X @ w)
        g = X.T @ (p - y) / len(y)
        g[1:] += 2 * lam * w[1:]
        S = np.maximum(p * (1 - p), 1e-9)
        H = (X * S[:, None]).T @ X / len(y) + 2 * lam * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < 1e-11:
            break
    return w


w = fit_newton(Xtr, ytr)
names = ["log income", "utilisation", "late payments", "years"]
print(f"\n{'feature':<16} {'coef (log-odds)':>17} {'odds ratio':>12}")
for j, nm in enumerate(names):
    print(f"{nm:<16} {w[j + 1]:>17.4f} {np.exp(w[j + 1]):>12.4f}")
print("Coefficients are per standard deviation because the features were")
print("standardised: 'one SD more utilisation multiplies the odds by "
      f"{np.exp(w[2]):.2f}'.")

p_te = sigmoid(Xte @ w)

# --- why 0.5 is the wrong threshold here ------------------------------------
print("\n" + "=" * 72)
print("the default threshold of 0.5")
print("=" * 72)
pred50 = (p_te >= 0.5).astype(float)
print(f"predictions above 0.5: {int(pred50.sum())} of {len(pred50)}")
print(f"accuracy             : {(pred50 == yte).mean():.4f}")
print(f"always-predict-zero  : {(yte == 0).mean():.4f}")
print("The model beats the trivial baseline by almost nothing on accuracy")
print("while flagging almost no one. Accuracy is the wrong metric and 0.5")
print("is the wrong threshold (Chapter 34).")

# --- choose the threshold from costs ----------------------------------------
print("\n" + "=" * 72)
print("choosing the threshold from the cost of each error")
print("=" * 72)
COST_FN = 4000.0        # a default we approved: the money we lose
COST_FP = 250.0         # a good customer we declined: the margin forgone
print(f"cost of a missed default (FN): GBP {COST_FN:,.0f}")
print(f"cost of a declined good customer (FP): GBP {COST_FP:,.0f}")

print(f"\n{'threshold':>10} {'flagged':>8} {'TP':>6} {'FP':>6} {'FN':>6} "
      f"{'recall':>8} {'precision':>10} {'total cost':>13}")
best = (None, np.inf)
for t in (0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50):
    pred = (p_te >= t)
    tp = int(np.sum(pred & (yte == 1)))
    fp = int(np.sum(pred & (yte == 0)))
    fn = int(np.sum(~pred & (yte == 1)))
    cost = fn * COST_FN + fp * COST_FP
    rec = tp / max(1, tp + fn)
    prec = tp / max(1, tp + fp)
    if cost < best[1]:
        best = (t, cost)
    print(f"{t:>10.2f} {int(pred.sum()):>8} {tp:>6} {fp:>6} {fn:>6} "
          f"{rec:>8.3f} {prec:>10.3f} {cost:>13,.0f}")

t_star, c_star = best
c_half = (np.sum((p_te < 0.5) & (yte == 1)) * COST_FN
          + np.sum((p_te >= 0.5) & (yte == 0)) * COST_FP)
print(f"\ncheapest threshold: {t_star:.2f} at GBP {c_star:,.0f}")
print(f"threshold of 0.50 : GBP {c_half:,.0f}")
print(f"difference        : GBP {c_half - c_star:,.0f} "
      f"({(c_half - c_star) / max(c_half, 1) * 100:.1f}% of the cost)")

theory = COST_FP / (COST_FP + COST_FN)
print(f"\ntheoretical optimum = COST_FP / (COST_FP + COST_FN) = {theory:.4f}")
print("Expected cost is minimised by flagging whenever")
print("p * COST_FN > (1 - p) * COST_FP, i.e. p > COST_FP/(COST_FP+COST_FN).")
print("This requires CALIBRATED probabilities — the rule is meaningless if")
print("p is merely a score. It is the main practical reason to care about")
print("calibration rather than only about ranking.")

# --- and the calibration that makes the rule valid --------------------------
print("\ncalibration check on the test set, with the noise floor:")
print(f"{'predicted band':>18} {'n':>6} {'mean p':>9} {'observed':>10} "
      f"{'expected':>9} {'+-2 SE':>16}")
edges = np.quantile(p_te, np.linspace(0, 1, 9))
for i in range(8):
    m = (p_te >= edges[i]) & (p_te <= edges[i + 1])
    k, pbar = int(m.sum()), p_te[m].mean()
    se = np.sqrt(pbar * (1 - pbar) / k)
    print(f"  [{edges[i]:.3f}, {edges[i+1]:.3f}] {k:>6} "
          f"{pbar:>9.4f} {yte[m].mean():>10.4f} {pbar * k:>9.1f} "
          f"[{pbar - 2 * se:>6.4f},{pbar + 2 * se:>6.4f}]")
print("\nEvery observed rate but one falls inside two standard errors of the")
print("prediction, and the one that does not is a band containing about")
print("eighteen expected events — where a handful either way moves the rate")
print("by a third. A calibration table without its noise floor invites you")
print("to diagnose a model problem that is really a sample-size problem")
print("(Chapter 8). Chapter 34 turns this into a single metric.")

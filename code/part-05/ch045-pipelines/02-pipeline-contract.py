# -*- coding: utf-8 -*-
# Extracted from: Chapter 45 — Data and Feature Pipelines
# Source: src/.../ch045-pipelines.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A pipeline with state, a contract that guards it, and the fit/transform
discipline that keeps state out of the validation set.
"""
import numpy as np

rng = np.random.default_rng(4)


class ContractViolation(AssertionError):
    pass


class Contract:
    """Section 5.3's checks, learned from a reference sample and asserted on
    every batch thereafter."""

    def __init__(self, names, categorical=()):
        self.names, self.categorical = list(names), set(categorical)
        self.ref = None

    def fit(self, X):
        self.ref = {
            "n_cols": X.shape[1],
            "min": X.min(0), "max": X.max(0),
            "null_rate": np.isnan(X).mean(0),
            "categories": {j: set(np.unique(X[:, j][~np.isnan(X[:, j])]))
                           for j in self.categorical},
            "mean": np.nanmean(X, axis=0), "sd": np.nanstd(X, axis=0),
        }
        return self

    def check(self, X, *, tol_range=0.25, tol_null=0.05, label="batch"):
        r, problems = self.ref, []
        if X.shape[1] != r["n_cols"]:
            raise ContractViolation(
                f"[{label}] column count {X.shape[1]} != {r['n_cols']}")
        for j, nm in enumerate(self.names):
            span = max(r["max"][j] - r["min"][j], 1e-9)
            lo, hi = np.nanmin(X[:, j]), np.nanmax(X[:, j])
            if lo < r["min"][j] - tol_range * span:
                problems.append(f"{nm}: min {lo:.4g} below reference "
                                f"{r['min'][j]:.4g}")
            if hi > r["max"][j] + tol_range * span:
                problems.append(f"{nm}: max {hi:.4g} above reference "
                                f"{r['max'][j]:.4g}")
            nr = float(np.isnan(X[:, j]).mean())
            if nr > r["null_rate"][j] + tol_null:
                problems.append(f"{nm}: null rate {nr:.3f} vs reference "
                                f"{r['null_rate'][j]:.3f}")
            if j in self.categorical:
                new = set(np.unique(X[:, j][~np.isnan(X[:, j])])) \
                    - r["categories"][j]
                if new:
                    problems.append(f"{nm}: unseen categories "
                                    f"{sorted(new)[:4]}")
        if problems:
            raise ContractViolation(f"[{label}] " + "; ".join(problems))
        return True


# --- the reference batch ----------------------------------------------------
NAMES = ["amount_gbp", "tenure_months", "n_txn_30d", "region_code"]


def make_batch(n, seed, *, amount_scale=1.0, region_max=5, null_extra=0.0):
    rs = np.random.default_rng(seed)
    amount = rs.lognormal(4.0, 0.7, n) * amount_scale
    tenure = rs.uniform(0, 120, n)
    n_txn = rs.poisson(6, n).astype(float)
    region = rs.integers(0, region_max, n).astype(float)
    X = np.column_stack([amount, tenure, n_txn, region])
    if null_extra:
        m = rs.random(n) < null_extra
        X[m, 0] = np.nan
    return X


ref = make_batch(4000, 0)
contract = Contract(NAMES, categorical=[3]).fit(ref)

print("=" * 72)
print("the contract, against batches that each break one thing")
print("=" * 72)
cases = [
    ("same distribution (control)", make_batch(2000, 1)),
    ("amounts switched to pence", make_batch(2000, 2, amount_scale=100.0)),
    ("a new region code appears", make_batch(2000, 3, region_max=7)),
    ("upstream join started missing", make_batch(2000, 4, null_extra=0.22)),
    ("a column was dropped", make_batch(2000, 5)[:, :3]),
]
for label, batch in cases:
    try:
        contract.check(batch, label=label)
        print(f"  PASS  {label}")
    except ContractViolation as e:
        msg = str(e).split("] ", 1)[1]
        print(f"  FAIL  {label}\n          {msg[:96]}")

print("\nEach of these is a real incident shape, each produces numbers a")
print("model will happily consume, and none of them raises an exception")
print("anywhere else in the pipeline. The contract is the only thing between")
print("them and a silently degraded prediction.")

# --- fit/transform: state must not cross the boundary -----------------------
print("\n" + "=" * 72)
print("fit/transform: how much does leaking a scaler actually cost?")
print("=" * 72)
print("Section 6.3 says a mean-based transform leaks O(1/n), so this should")
print("be negligible at large n and material at small n. Measured:\n")


def target_encode(cat_tr, y_tr, cat_all, smoothing=0.0):
    """A stateful transform with a much worse leak than a scaler."""
    prior = float(y_tr.mean())
    out = np.full(len(cat_all), prior)
    for c in np.unique(cat_tr):
        m = cat_tr == c
        k = m.sum()
        enc = (y_tr[m].sum() + smoothing * prior) / (k + smoothing)
        out[cat_all == c] = enc
    return out


def knn_auc(Xtr, ytr, Xte, yte, k=9):
    d = ((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)
    idx = np.argpartition(d, k, axis=1)[:, :k]
    s = ytr[idx].mean(1)
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s))
    r[o] = np.arange(1, len(s) + 1)
    npos = int(yte.sum())
    if npos == 0 or npos == len(yte):
        return float("nan")
    return float((r[yte == 1].sum() - npos * (npos + 1) / 2)
                 / (npos * (len(yte) - npos)))


print(f"{'n':>7} {'scaler fitted in-fold':>23} {'scaler fitted on all':>22} "
      f"{'gap':>9}")
for n in (40, 100, 400, 2000, 10000):
    gaps = []
    for rep in range(30):
        rs = np.random.default_rng(1000 + rep)
        X = rs.normal(size=(2 * n, 5))
        yv = (rs.random(2 * n)
              < 1 / (1 + np.exp(-(X[:, 0] - 0.7 * X[:, 1])))).astype(int)
        Xtr, ytr, Xte, yte = X[:n], yv[:n], X[n:], yv[n:]
        mu_i, sd_i = Xtr.mean(0), Xtr.std(0) + 1e-9
        a = knn_auc((Xtr - mu_i) / sd_i, ytr, (Xte - mu_i) / sd_i, yte)
        mu_a, sd_a = X.mean(0), X.std(0) + 1e-9    # the mistake
        b = knn_auc((Xtr - mu_a) / sd_a, ytr, (Xte - mu_a) / sd_a, yte)
        gaps.append((a, b))
    a = float(np.mean([g[0] for g in gaps]))
    b = float(np.mean([g[1] for g in gaps]))
    print(f"{n:>7} {a:>23.4f} {b:>22.4f} {b - a:>+9.4f}")

print("\nThe scaler leak is small at every size and vanishes as n grows,")
print("exactly as the O(1/n) argument predicts. That is worth knowing")
print("because it tells you where NOT to spend your attention.")

print("\nNow the same experiment with a target encoder, which section 6.3")
print("says leaks O(1) for rare categories rather than O(1/n):\n")
print(f"{'categories':>11} {'rows/category':>14} {'out-of-fold enc':>17} "
      f"{'fitted on all':>15} {'gap':>9}")
for n_cat in (10, 50, 200, 800):
    n = 1600
    outs, alls = [], []
    for rep in range(20):
        rs = np.random.default_rng(2000 + rep)
        cat = rs.integers(0, n_cat, 2 * n)
        eff = rs.normal(0, 1.0, n_cat)[cat]
        yv = (rs.random(2 * n) < 1 / (1 + np.exp(-eff))).astype(int)
        ctr, cte = cat[:n], cat[n:]
        ytr, yte = yv[:n], yv[n:]
        # honest: encoding learned from training rows only
        enc_tr = target_encode(ctr, ytr, ctr)
        enc_te = target_encode(ctr, ytr, cte)
        outs.append(knn_auc(enc_tr[:, None], ytr, enc_te[:, None], yte))
        # the mistake: encoding learned from everything
        enc_all = target_encode(cat, yv, cat)
        alls.append(knn_auc(enc_all[:n, None], ytr, enc_all[n:, None], yte))
    print(f"{n_cat:>11} {2 * n / n_cat:>14.0f} {np.mean(outs):>17.4f} "
          f"{np.mean(alls):>15.4f} {np.mean(alls) - np.mean(outs):>+9.4f}")

print("\nWith 800 categories over 3,200 rows — four rows each — the leaked")
print("encoder reports a substantially better model than the honest one,")
print("because each category's encoding is largely determined by the labels")
print("of the very rows being encoded.")
print("\nThat is the practical rule behind fit/transform: the danger is not")
print("proportional to how complicated the transform is, it is proportional")
print("to how much of any single row's own label ends up in that row's")
print("features. A scaler averages over everything and is safe; a target")
print("encoder on rare categories barely averages at all.")

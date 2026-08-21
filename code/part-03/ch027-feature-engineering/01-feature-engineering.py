# Extracted from: Chapter 27 — Feature Engineering and Feature Selection
# Source: src/.../ch027-feature-engineering.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Encodings, aggregates, transforms — and the leak that target encoding
creates when computed in-fold.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)

# --- the four families, on a realistic table --------------------------------
n = 30_000
today = pd.Timestamp("2026-08-13")
df = pd.DataFrame({
    "user_id": rng.integers(1, 6000, n),
    "signup": today - pd.to_timedelta(rng.integers(30, 900, n), "D"),
    "last_order": today - pd.to_timedelta(rng.integers(0, 200, n), "D"),
    "n_orders": rng.poisson(8, n) + 1,
    "total_spend": rng.gamma(3, 90, n).round(2),
    "city": rng.choice([f"city_{i}" for i in range(120)], n),
})

feat = pd.DataFrame(index=df.index)
# temporal
feat["tenure_days"] = (today - df["signup"]).dt.days
feat["recency_days"] = (today - df["last_order"]).dt.days
feat["signup_dow"] = df["signup"].dt.dayofweek
# ratios
feat["avg_order_value"] = df["total_spend"] / df["n_orders"]
feat["orders_per_month"] = df["n_orders"] / (feat["tenure_days"] / 30.44)
# aggregates and self-relative features
feat["city_mean_spend"] = df.groupby("city")["total_spend"].transform("mean")
feat["spend_vs_city"] = df["total_spend"] / feat["city_mean_spend"]
feat["expected_gap"] = feat["tenure_days"] / df["n_orders"]
feat["recency_vs_own_gap"] = feat["recency_days"] / feat["expected_gap"]

print("engineered features:")
print(feat.describe().T[["mean", "std", "min", "max"]].round(2).to_string())
print("\nrecency_vs_own_gap compares each user against their OWN rhythm —")
print("the kind of feature a model cannot construct and a human can.")

# --- eq. 27.1: smoothed target encoding as a posterior mean -----------------
print("\n" + "=" * 72)
print("target encoding: smoothing is a Bayesian posterior mean (eq. 27.5)")
print("=" * 72)

target = (rng.random(n) < 0.25).astype(int)
global_mean = target.mean()

tmp = pd.DataFrame({"city": df["city"], "y": target})
stats_ = tmp.groupby("city")["y"].agg(["count", "mean"])

print(f"global mean = {global_mean:.4f}\n")
print(f"{'m':>5} " + " ".join(f"{f'n={k}':>9}" for k in (2, 5, 20, 100, 500)))
for m in (0, 5, 20, 100):
    row = []
    for nc in (2, 5, 20, 100, 500):
        raw = 0.60                                  # a category with a high raw mean
        enc = (nc * raw + m * global_mean) / (nc + m)
        row.append(f"{enc:>9.3f}")
    print(f"{m:>5} " + " ".join(row))
print("\nRows: smoothing strength m. Columns: category size. A category seen")
print("twice with a raw mean of 0.60 is pulled almost to the global mean at")
print("m=20; one seen 500 times keeps its own estimate. m is a pseudo-count.")

# --- eq. 27.4: in-fold target encoding leaks --------------------------------
print("\n" + "=" * 72)
print("target encoding computed IN-FOLD leaks — demonstrated on pure noise")
print("=" * 72)

m_rows = 8000
noise_id = rng.integers(0, m_rows // 4, m_rows)     # ~4 rows per category
y = (rng.random(m_rows) < 0.5).astype(int)          # NO relationship at all

frame = pd.DataFrame({"cat": noise_id, "y": y})
split = m_rows // 2
train, test = frame.iloc[:split].copy(), frame.iloc[split:].copy()

# WRONG: encode using the row's own target.
means = train.groupby("cat")["y"].mean()
train["enc_leaky"] = train["cat"].map(means)
test["enc_leaky"] = test["cat"].map(means).fillna(train["y"].mean())

# RIGHT: out-of-fold encoding (eq. 27.2).
K = 5
fold = rng.integers(0, K, len(train))
train["enc_oof"] = np.nan
for k in range(K):
    other = train[fold != k]
    fold_means = other.groupby("cat")["y"].mean()
    train.loc[fold == k, "enc_oof"] = (
        train.loc[fold == k, "cat"].map(fold_means).fillna(other["y"].mean()).values)
test["enc_oof"] = test["cat"].map(means).fillna(train["y"].mean())


def auc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos, neg = labels.sum(), (1 - labels).sum()
    return (ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)


print(f"the target is a coin flip; the category is a random ID.\n")
print(f"{'encoding':<22} {'train AUC':>11} {'test AUC':>10} {'verdict'}")
for name, col in (("in-fold (leaky)", "enc_leaky"), ("out-of-fold", "enc_oof")):
    tr = auc(train[col].to_numpy(), train["y"].to_numpy())
    te = auc(test[col].to_numpy(), test["y"].to_numpy())
    verdict = "LEAK" if tr - te > 0.1 else "honest"
    print(f"{name:<22} {tr:>11.3f} {te:>10.3f} {verdict}")

print(f"\ncorr(leaky encoding, target) on train : "
      f"{np.corrcoef(train['enc_leaky'], train['y'])[0,1]:.3f}")
print(f"eq. 27.6 predicts about 1/sqrt(n_c) = "
      f"{1/np.sqrt(4):.3f} for ~4 rows per category")
print("\nThe in-fold encoding shows strong training signal on data with NO")
print("relationship whatsoever. Out-of-fold encoding shows chance on both.")

# --- eq. 27.7: log transforms linearise multiplicative relationships --------
print("\n" + "=" * 72)
print("log transforms")
print("=" * 72)
x = rng.uniform(1, 100, 20_000)
y_mult = 3.0 * x ** 1.7 * np.exp(rng.normal(0, 0.2, 20_000))

print(f"corr(x, y)             : {np.corrcoef(x, y_mult)[0,1]:.4f}")
print(f"corr(log x, log y)     : "
      f"{np.corrcoef(np.log(x), np.log(y_mult))[0,1]:.4f}   <- eq. 27.7")
slope, intercept = np.polyfit(np.log(x), np.log(y_mult), 1)
print(f"fitted exponent b      : {slope:.3f}  (true 1.700)")
print(f"fitted coefficient a   : {np.exp(intercept):.3f}  (true 3.000)")
print("\nThe relationship is exactly linear in log-log space, and the fitted")
print("parameters recover the generating process.")

# Separately: on a genuinely lognormal variable, log restores symmetry.
print("\non a lognormal variable (revenue, session length, file size):")
revenue = rng.lognormal(4.0, 1.1, 50_000)
for label, v in (("raw", revenue), ("log", np.log(revenue))):
    z = (v - v.mean()) / v.std()
    print(f"  {label:>4}: skew {float((z**3).mean()):+6.2f}, "
          f"mean/median {v.mean()/np.median(v):.2f}")
print("Log removes the skew because the variable IS multiplicative.")
print("\nNote it does not always help: log(y) above inherits the skew of")
print("log(x), because x was uniform rather than lognormal. A transform")
print("fixes skew when the generating process is multiplicative, not by fiat.")

# Extracted from: Chapter 29 — Time Series, Forecasting, and Anomaly Detection
# Source: src/.../ch029-timeseries.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A forecast evaluated against a naive baseline, and anomalies found in the
residuals rather than in the raw series.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(5)

# --- a realistic series: trend + weekly + annual + noise, plus anomalies ----
T = 730
t = np.arange(T)
dates = pd.date_range("2024-01-01", periods=T, freq="D")
level = 200 + 0.08 * t
weekly = 25 * np.sin(2 * np.pi * t / 7 - 1.0)
annual = 40 * np.sin(2 * np.pi * t / 365.25)
y = level + weekly + annual + rng.normal(0, 8, T)

# plant genuine anomalies
anomaly_idx = np.array([120, 300, 301, 302, 560])
y[anomaly_idx] += np.array([90, -70, -80, -75, 110])
s = pd.Series(y, index=dates)

# --- baselines are the bar to beat ------------------------------------------
print("=" * 72)
print("forecast evaluation: always compare against naive")
print("=" * 72)

df = pd.DataFrame({"y": s})
df["naive"] = df["y"].shift(1)                    # tomorrow = today
df["seasonal_naive"] = df["y"].shift(7)           # tomorrow = same day last week
df["lag1"] = df["y"].shift(1)
df["lag7"] = df["y"].shift(7)
df["roll28"] = df["y"].shift(1).rolling(28).mean()
for period, name in ((7, "w"), (365.25, "a")):
    df[f"sin_{name}"] = np.sin(2 * np.pi * np.arange(T) / period)
    df[f"cos_{name}"] = np.cos(2 * np.pi * np.arange(T) / period)
df["trend"] = np.arange(T)
df = df.dropna()

feats = ["lag1", "lag7", "roll28", "sin_w", "cos_w", "sin_a", "cos_a", "trend"]


def fit_ols(X, y_):
    A = np.column_stack([np.ones(len(X)), X])
    b, *_ = np.linalg.lstsq(A, y_, rcond=None)
    return b


def pred(X, b):
    return np.column_stack([np.ones(len(X)), X]) @ b


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


# walk-forward: expanding window, five folds
X = df[feats].to_numpy()
yv = df["y"].to_numpy()
n = len(df)
start = n // 2
fold_size = (n - start) // 5

model_err, naive_err, seasonal_err = [], [], []
for i in range(5):
    a, b = start + i * fold_size, start + (i + 1) * fold_size
    beta = fit_ols(X[:a], yv[:a])
    model_err.append(mae(pred(X[a:b], beta), yv[a:b]))
    naive_err.append(mae(df["naive"].to_numpy()[a:b], yv[a:b]))
    seasonal_err.append(mae(df["seasonal_naive"].to_numpy()[a:b], yv[a:b]))

print(f"{'method':<22} {'MAE (walk-forward)':>20} {'MASE':>8}")
naive_mean = np.mean(naive_err)
for label, errs in (("naive (t-1)", naive_err),
                    ("seasonal naive (t-7)", seasonal_err),
                    ("regression model", model_err)):
    print(f"{label:<22} {np.mean(errs):>20.3f} {np.mean(errs)/naive_mean:>8.3f}")

best = min(("naive", np.mean(naive_err)), ("seasonal", np.mean(seasonal_err)),
           ("model", np.mean(model_err)), key=lambda kv: kv[1])
print(f"\nbest: {best[0]}")
print("MASE below 1 means better than the naive forecast. A model that cannot")
print("beat 'tomorrow equals today' has demonstrated nothing (section 6.2).")

# --- eq. 29.8: anomalies live in the residuals ------------------------------
print("\n" + "=" * 72)
print("anomaly detection: model the expectation, then examine residuals")
print("=" * 72)

beta_full = fit_ols(X, yv)
fitted = pred(X, beta_full)
resid = yv - fitted

# Robust threshold — the MAD is unaffected by the anomalies themselves
# (Chapter 23's masking argument).
med = np.median(resid)
mad = np.median(np.abs(resid - med))
robust_z = 0.6745 * (resid - med) / mad
flagged = np.abs(robust_z) > 4.0

# Naive alternative: flag extreme RAW values.
raw = df["y"].to_numpy()
raw_z = (raw - raw.mean()) / raw.std()
raw_flagged = np.abs(raw_z) > 2.5

planted = set(dates[anomaly_idx])
found_resid = set(df.index[flagged])
found_raw = set(df.index[raw_flagged])
detectable = planted & set(df.index)

print(f"planted anomalies within the modelled window : {len(detectable)}")
print(f"\n{'method':<26} {'flagged':>9} {'true positives':>16} "
      f"{'false positives':>17}")
for label, found in (("raw value threshold", found_raw),
                     ("residual + robust MAD", found_resid)):
    tp = len(found & detectable)
    fp = len(found - detectable)
    print(f"{label:<26} {len(found):>9} {tp:>16} {fp:>17}")

print(f"\ndates flagged by the residual method:")
for d in sorted(found_resid)[:8]:
    i = df.index.get_loc(d)
    marker = " <- planted" if d in detectable else ""
    print(f"  {d.date()}  residual {resid[i]:+7.1f}  "
          f"robust z {robust_z[i]:+6.1f}{marker}")

print("\nThresholding the raw series flags ordinary seasonal peaks and misses")
print("anomalies that happen to fall near the mean. Modelling the expectation")
print("first is what makes 'unusual for this day' computable.")

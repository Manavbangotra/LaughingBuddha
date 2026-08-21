# Extracted from: Chapter 29 — Time Series, Forecasting, and Anomaly Detection
# Source: src/.../ch029-timeseries.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Ordered data: what breaks, and the corrections.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)

# --- a series with trend, weekly seasonality and noise ----------------------
T = 1000
t = np.arange(T)
trend = 50 + 0.05 * t
weekly = 8 * np.sin(2 * np.pi * t / 7)
noise = np.cumsum(rng.normal(0, 0.6, T))          # a random-walk component
y = trend + weekly + noise
dates = pd.date_range("2024-01-01", periods=T, freq="D")
s = pd.Series(y, index=dates)

# --- stationarity, checked by halves ----------------------------------------
print("=" * 72)
print("stationarity (eq. 29.1)")
print("=" * 72)
h1, h2 = s.iloc[:T // 2], s.iloc[T // 2:]
print(f"{'':<16} {'first half':>12} {'second half':>13} {'ratio':>8}")
print(f"{'mean':<16} {h1.mean():>12.2f} {h2.mean():>13.2f} "
      f"{h2.mean()/h1.mean():>8.2f}")
print(f"{'std':<16} {h1.std():>12.2f} {h2.std():>13.2f} "
      f"{h2.std()/h1.std():>8.2f}")
diff = s.diff().dropna()
d1, d2 = diff.iloc[:len(diff)//2], diff.iloc[len(diff)//2:]
print(f"\nafter differencing:")
print(f"{'mean':<16} {d1.mean():>12.3f} {d2.mean():>13.3f}")
print(f"{'std':<16} {d1.std():>12.3f} {d2.std():>13.3f}")
print("Levels are non-stationary (the mean moves); differences are stationary.")

# --- eq. 29.3: autocorrelation destroys the standard error ------------------
print("\n" + "=" * 72)
print("autocorrelation inflates the variance of the mean (eq. 29.3)")
print("=" * 72)


def ar1(n, rho, sd=1.0, rng=rng):
    x = np.zeros(n)
    innov = rng.normal(0, sd * np.sqrt(1 - rho ** 2), n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + innov[i]
    return x


print(f"{'rho':>6} {'naive SE':>10} {'true SE':>10} {'ratio':>8} "
      f"{'predicted sqrt(DEFF)':>22}")
for rho in (0.0, 0.5, 0.8, 0.9):
    means = np.array([ar1(500, rho).mean() for _ in range(2000)])
    one = ar1(500, rho)
    naive_se = one.std(ddof=1) / np.sqrt(500)
    true_se = means.std(ddof=1)
    deff = (1 + rho) / (1 - rho)
    print(f"{rho:>6.1f} {naive_se:>10.5f} {true_se:>10.5f} "
          f"{true_se/naive_se:>8.2f} {np.sqrt(deff):>22.2f}")
print("\nAt rho=0.9 the honest standard error is over four times the naive")
print("one. 'We have 500 observations' is not 500 independent observations.")

# --- k-fold vs walk-forward --------------------------------------------------
print("\n" + "=" * 72)
print("random k-fold trains on the future")
print("=" * 72)

frame = pd.DataFrame({"y": y}, index=dates)
frame["lag1"] = frame["y"].shift(1)
frame["lag7"] = frame["y"].shift(7)
frame["roll7"] = frame["y"].shift(1).rolling(7).mean()      # SHIFTED: no leak
frame["dow_sin"] = np.sin(2 * np.pi * frame.index.dayofweek / 7)
frame["dow_cos"] = np.cos(2 * np.pi * frame.index.dayofweek / 7)
frame = frame.dropna()

feat_cols = ["lag1", "lag7", "roll7", "dow_sin", "dow_cos"]
X, target = frame[feat_cols].to_numpy(), frame["y"].to_numpy()


def fit_ols(Xtr, ytr):
    A = np.column_stack([np.ones(len(Xtr)), Xtr])
    beta, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    return beta


def pred(Xte, beta):
    return np.column_stack([np.ones(len(Xte)), Xte]) @ beta


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


# WRONG: random k-fold
folds = rng.integers(0, 5, len(X))
kfold_errors = []
for k in range(5):
    tr, te = folds != k, folds == k
    kfold_errors.append(rmse(pred(X[te], fit_ols(X[tr], target[tr])), target[te]))

# RIGHT: expanding-window walk-forward
wf_errors = []
n_splits, min_train = 5, len(X) // 3
for i in range(n_splits):
    end_tr = min_train + i * (len(X) - min_train) // (n_splits + 1)
    end_te = end_tr + (len(X) - min_train) // (n_splits + 1)
    if end_te > len(X):
        break
    beta = fit_ols(X[:end_tr], target[:end_tr])
    wf_errors.append(rmse(pred(X[end_tr:end_te], beta), target[end_tr:end_te]))

print(f"random k-fold RMSE   : {np.mean(kfold_errors):.4f}")
print(f"walk-forward RMSE    : {np.mean(wf_errors):.4f}")
print(f"optimism             : {np.mean(wf_errors) - np.mean(kfold_errors):+.4f}")
print("\nThe k-fold estimate is better than anything achievable in production,")
print("because each fold trains on points surrounding its own test points.")

# --- the rolling-window leak -------------------------------------------------
print("\n" + "=" * 72)
print("the .rolling() leak: forgetting .shift(1)")
print("=" * 72)
leaky = pd.DataFrame({"y": y}, index=dates)
leaky["roll_leaky"] = leaky["y"].rolling(7).mean()          # INCLUDES today
leaky["roll_safe"] = leaky["y"].shift(1).rolling(7).mean()  # excludes today
leaky = leaky.dropna()

for col in ("roll_leaky", "roll_safe"):
    c = np.corrcoef(leaky[col], leaky["y"])[0, 1]
    beta = fit_ols(leaky[[col]].to_numpy(), leaky["y"].to_numpy())
    e = rmse(pred(leaky[[col]].to_numpy(), beta), leaky["y"].to_numpy())
    print(f"  {col:<12} corr with target {c:.4f}   in-sample RMSE {e:.3f}")
print("\nThe unshifted rolling mean contains 1/7th of the target itself.")
print("It looks like a strong feature and cannot be computed in production.")

# --- eq. 29.7: cyclical encoding --------------------------------------------
print("\n" + "=" * 72)
print("cyclical encoding (eq. 29.7)")
print("=" * 72)
months = np.arange(1, 13)
sin_m, cos_m = np.sin(2*np.pi*months/12), np.cos(2*np.pi*months/12)
print(f"{'pair':<18} {'integer distance':>18} {'cyclical distance':>19}")
for a, b, label in ((12, 1, "Dec - Jan"), (6, 7, "Jun - Jul"),
                    (1, 7, "Jan - Jul")):
    int_d = abs(a - b)
    cyc_d = np.hypot(sin_m[a-1] - sin_m[b-1], cos_m[a-1] - cos_m[b-1])
    print(f"{label:<18} {int_d:>18} {cyc_d:>19.4f}")
print("\nInteger encoding puts December and January 11 apart — further than")
print("January and July. Cyclical encoding makes adjacent months adjacent.")

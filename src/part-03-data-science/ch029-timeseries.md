---
id: ds-timeseries
number: 29
part: III
tier: focused
status: reviewed
requires: [ds-leakage, ds-feature-eng]
provides: [stationarity, autocorrelation, seasonality, walk-forward-validation,
           timeseries-anomaly-detection]
citations: [kaufman2012]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Explain why ordered data breaks the independence assumption underlying
   everything in Parts I and IV.
2. Define stationarity and test for it informally.
3. Decompose a series into trend, seasonality and residual.
4. Explain autocorrelation and its effect on standard errors.
5. Build lag, rolling and calendar features without leaking.
6. Validate with walk-forward splits and explain why k-fold is invalid.
7. Choose and interpret forecasting metrics.
8. Detect anomalies in a series with trend and seasonality.

## 2. Why This Matters

Almost everything in {{part:1}} and {{part:4}} assumes observations are
independent and identically distributed. Time-ordered data violates both halves
of that assumption, and the violations are not minor.

**Not independent.** Today's value is correlated with yesterday's. That inflates
the effective standard error in exactly the way {{ch:ds-collection}}'s design
effect does, so every confidence interval computed with $n$ observations is too
narrow.

**Not identically distributed.** The mean drifts, the variance changes, the
relationship between features and target evolves. A model fitted on last year is
describing a world that may no longer exist ({{ch:ds-what-it-is}}).

The practical consequence is that the standard toolkit misbehaves silently.
Random k-fold cross-validation trains on the future and reports a score that
cannot be achieved. A model fitted on a trending series extrapolates the trend
forever. An anomaly detector tuned on a seasonal series flags every Monday.

This chapter is deliberately about what breaks and what to do instead, rather
than about classical time-series modelling — because the constraint that
transfers to the sequence models of {{part:7}} is the ordering, not
Box-Jenkins.

## 3. Prerequisites

{{ch:ds-leakage}} for temporal leakage, which this chapter develops properly;
{{ch:ds-feature-eng}} for the feature families; {{ch:math-inference}} for
standard errors and the independence assumption they rest on.

## 4. Intuitive Explanation

### 4.1 Order is information, and a constraint

In a table of customers the row order means nothing. In a table of daily sales
it means everything: row 100 comes after row 99, and using row 101 to predict
row 100 is not a modelling choice but a mistake.

That single fact changes three things:

- **Splitting.** Train must precede test in time.
- **Features.** Any feature must use only past information.
- **Uncertainty.** Adjacent observations are correlated, so the effective sample
  size is smaller than the row count.

### 4.2 The three components

Most series decompose into:

```text
  observed  =  trend  +  seasonality  +  residual
              ▲         ▲               ▲
        long-run     repeating       what is
        direction    at fixed        left over
                     period
```

**Trend** is the long-run movement. A model that ignores it will be
systematically wrong in one direction and increasingly so.

**{{term:seasonality}}** is a pattern repeating at a fixed period — hourly,
daily, weekly, annually. A series can have several at once: retail sales have
day-of-week, month-of-year and holiday effects simultaneously.

**Residual** is what remains. If it still shows structure, the decomposition is
incomplete.

The decomposition may be additive ($y = T + S + R$) or multiplicative
($y = T \times S \times R$). Multiplicative is right when the seasonal swing
grows with the level — which is common, and is handled by taking logs and
fitting an additive model ({{ch:math-functions}}).

### 4.3 Stationarity

A series is {{term:stationarity}} if its statistical properties do not change
over time: constant mean, constant variance, and an autocovariance depending
only on the lag.

Most real series are not stationary. They trend, their variance grows with the
level, and their seasonal amplitude changes.

Stationarity matters because it is the assumption under which "fit on the past,
predict the future" is coherent. Without it, the past describes a different
process. The standard responses are differencing (model changes rather than
levels), log-transforming (stabilise variance), and explicitly modelling the
trend and seasonality so the residual is stationary.

### 4.4 Why k-fold cross-validation is invalid

Random k-fold puts some future observations in the training set for every fold.

The model then learns from data it could never have had, and the effect is not
small: with autocorrelated data, a point's immediate neighbours are highly
informative about it, so training on both sides of a test point makes prediction
nearly trivial. The reported score can be dramatically better than anything
achievable, and {{ch:ds-leakage}} demonstrated exactly this.

The correct scheme is {{term:walk-forward-validation}}: train on a prefix, test
on the segment that follows, then slide forward.

```text
  fold 1  ████████░░░░░░░░░░░░░░░░░░░░
  fold 2  ████████████░░░░░░░░░░░░░░░░
  fold 3  ████████████████░░░░░░░░░░░░
  fold 4  ████████████████████░░░░░░░░
          ▲ train        ▲ test
```

## 5. Formal Explanation

### 5.1 Stationarity, formally

A series $\{X_t\}$ is **weakly stationary** if

$$
\E[X_t] = \mu, \qquad
\Var(X_t) = \sigma^{2}, \qquad
\Cov(X_t, X_{t+k}) = \gamma(k)
$$ (eq:weak-stationarity)

for all $t$ — mean and variance constant, and covariance depending only on the
lag $k$, not on when you look.

Informal tests: split the series into halves and compare means and variances; a
large difference indicates non-stationarity. Plot rolling statistics; a rolling
mean that drifts is a trend, a rolling standard deviation that grows is
heteroscedasticity. Formal tests such as the augmented Dickey-Fuller exist and
are worth knowing about, but the plot usually settles it.

### 5.2 Autocorrelation

The autocorrelation at lag $k$ is

$$
\rho(k) = \frac{\Cov(X_t, X_{t+k})}{\Var(X_t)} = \frac{\gamma(k)}{\gamma(0)}
$$ (eq:acf)

Its consequence for inference is the important part. For a series with
autocorrelation $\rho$ at lag 1 (an AR(1) process), the variance of the sample
mean is inflated relative to the independent case by approximately

$$
\text{DEFF} \approx \frac{1 + \rho}{1 - \rho}
$$ (eq:ar1-design-effect)

At $\rho = 0.8$ that factor is 9, so the effective sample size is $n/9$ and
standard errors computed assuming independence are three times too small. This
is {{ch:ds-collection}}'s design effect in a temporal guise.

> IMPORTANT: This is why "we have five years of daily data, so $n = 1825$" is
> usually wrong. With strong autocorrelation the effective sample size may be in
> the low hundreds, and every interval and every significance test computed on
> the raw count is overconfident.

### 5.3 Features without leakage

Every feature must be computable from information available at the prediction
time. Three families, and the trap in each:

**Lags.** $x_{t-1}, x_{t-7}, x_{t-365}$. Safe by construction, provided the lag
exceeds the forecast horizon. Predicting three days ahead means lag-1 is not
available at prediction time — you would not yet know yesterday's value for the
day you are predicting.

**Rolling statistics.** Mean, standard deviation, min and max over a trailing
window. The trap is a centred window, which includes future points. Always use
a trailing window, and shift by one so the current value is excluded.

**Calendar.** Day of week, month, holiday flags, days since an event. Safe, and
cyclical variables should be encoded as sine and cosine pairs so that December
is adjacent to January:

$$
\sin\!\left(\frac{2\pi m}{12}\right), \qquad \cos\!\left(\frac{2\pi m}{12}\right)
$$ (eq:cyclical-encoding)

Encoding month as the integer 1-12 tells a linear model that December and
January are eleven units apart, which is exactly wrong.

> WARNING: The most common temporal leak is a rolling feature computed with
> pandas `.rolling()` without a `.shift(1)`. The default window *includes* the
> current row, so a rolling mean of the target includes the target. The result
> looks excellent and is unusable.

### 5.4 Validation

{#tbl:temporal-validation caption="Validation schemes for ordered data. Only the last three are valid, and the choice among them depends on whether the process is stationary."}

| Scheme | Valid? | Note |
|---|---|---|
| Random k-fold | **no** | trains on the future |
| Single holdout at the end | yes | one estimate, high variance |
| Expanding window | yes | uses all history; assumes stationarity |
| Sliding window | yes | fixed training size; adapts to drift |
| Blocked CV with a gap | yes | gap prevents leakage through autocorrelation |

The **gap** matters when features use long windows. If a feature is a 30-day
rolling mean, the last 30 days of the training set overlap information in the
test set, so a gap of at least the window length should separate them.

### 5.5 Forecasting metrics

{#tbl:forecast-metrics caption="Forecasting metrics and what each is for."}

| Metric | Formula | Note |
|---|---|---|
| MAE | $\frac{1}{n}\sum\lvert y - \hat{y}\rvert$ | robust; same units |
| RMSE | $\sqrt{\frac{1}{n}\sum(y-\hat{y})^{2}}$ | penalises large errors |
| MAPE | $\frac{100}{n}\sum\lvert\frac{y-\hat{y}}{y}\rvert$ | **undefined at $y = 0$**; asymmetric |
| sMAPE | symmetric variant | bounded, still awkward |
| MASE | MAE relative to a naive forecast | scale-free, comparable across series |

MAPE is the most requested and the most flawed: it is undefined when the actual
is zero, and it penalises over-forecasting more heavily than under-forecasting,
because the denominator is the actual value.

**Always compare against a naive baseline.** For most series, "tomorrow equals
today" or "this week equals last week" is a strong forecast. A model that does
not beat it has demonstrated nothing, and this comparison is what MASE
formalises.

### 5.6 Series that are not continuous

Much of the standard toolkit assumes a smooth, continuously-valued series.
Two common cases break that assumption and need different treatment.

**Intermittent demand.** Many items sell zero units on most days. The series is
mostly zeros with occasional positive values, and a mean-based forecast returns
a fractional number of units that is never correct. MAPE is undefined on the
zeros ({{tbl:forecast-metrics}}). The standard response decomposes the problem:
model the *interval between* non-zero events and the *size* of an event
separately, then combine — which is what Croston's method does.

**Count data.** Website visits, support tickets and defect counts are
non-negative integers, frequently with variance growing alongside the mean. A
model assuming constant-variance Gaussian errors will produce negative
predictions and badly calibrated intervals. A Poisson or negative-binomial
formulation respects both constraints, and the negative binomial additionally
handles the overdispersion that real count data almost always shows.

The general lesson is that the error distribution is a modelling choice like any
other. Least squares assumes additive Gaussian noise
({{ch:math-optimization}}), and when the data is bounded below by zero,
integer-valued, or heteroscedastic, that assumption is doing visible damage —
usually in the form of prediction intervals that extend below zero.

### 5.7 Anomaly detection

In a series, an anomaly is a point unlikely given the *expected* behaviour at
that time — not simply a large value. A Monday spike in a weekly-seasonal series
is expected.

The workable approach is therefore: model the expected value, then look at the
residual.

$$
r_t = y_t - \hat{y}_t, \qquad
\text{anomaly if } \lvert r_t \rvert > k \cdot \text{MAD}(r)
$$ (eq:residual-anomaly)

Using the median absolute deviation rather than the standard deviation matters
for the masking reason of {{ch:ds-cleaning}}: anomalies inflate the standard
deviation used to detect them.

## 6. Mathematical Foundation

### 6.1 Deriving the autocorrelation inflation

For an AR(1) process $X_t = \rho X_{t-1} + \varepsilon_t$ with stationary
variance $\sigma^{2}$, the autocovariance at lag $k$ is
$\gamma(k) = \sigma^{2}\rho^{k}$.

The variance of the sample mean of $n$ consecutive observations is

$$
\Var(\bar{X}) = \frac{\sigma^{2}}{n}\left[1 + 2\sum_{k=1}^{n-1}
  \left(1 - \frac{k}{n}\right)\rho^{k}\right]
$$ (eq:ar1-mean-variance)

For large $n$ the bracketed term converges to

$$
1 + 2\sum_{k=1}^{\infty}\rho^{k} = 1 + \frac{2\rho}{1-\rho} = \frac{1+\rho}{1-\rho}
$$ (eq:ar1-limit)

giving {{eq:ar1-design-effect}}. At $\rho = 0.5$ the inflation is 3; at
$\rho = 0.9$ it is 19.

The intuition is that consecutive observations largely repeat information, so
$n$ correlated points carry roughly the information of $n(1-\rho)/(1+\rho)$
independent ones.

### 6.2 Why a naive forecast is hard to beat

For a random walk $y_t = y_{t-1} + \varepsilon_t$ with
$\varepsilon \sim \mathcal{N}(0, \sigma^{2})$, the optimal one-step forecast is
exactly $\hat{y}_t = y_{t-1}$, with expected squared error $\sigma^{2}$.

No model can do better, because the increment is unpredictable by construction.
Any model reporting a lower error on such a series is either fitting noise or
leaking.

Many business series are close to random walks, which is why the naive baseline
is so strong and why beating it by a few percent is a real result rather than a
disappointing one.

### 6.3 Why cyclical encoding needs two components

Encoding an angle $\theta$ with $\sin\theta$ alone is ambiguous: $\sin$ takes the
same value at $\theta$ and $\pi - \theta$. The pair $(\sin\theta, \cos\theta)$ is
a point on the unit circle and is unique.

The property that makes it correct for calendar features is that the Euclidean
distance between two encoded times depends only on their angular separation:

$$
\|(\sin\alpha, \cos\alpha) - (\sin\beta, \cos\beta)\|^{2}
  = 2 - 2\cos(\alpha - \beta)
$$ (eq:cyclical-distance)

which is {{eq:cosine-euclidean}} from {{ch:math-norms}} applied to unit vectors.
December and January are adjacent, as they should be, while an integer encoding
places them at maximum distance.

## 7. Implementation

```python {tier=A name=timeseries-validation}
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
```

## 8. Practical Example

Forecasting with an honest baseline, walk-forward validation, and residual-based
anomaly detection.

```python {tier=A name=forecast-and-anomalies}
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
```

## 9. Common Mistakes

**Random k-fold on ordered data.** Trains on the future.

**Rolling features without `.shift(1)`.** The window includes the current value.

**Using a lag shorter than the forecast horizon.** Not available at prediction
time.

**Treating $n$ observations as $n$ independent samples.** Autocorrelation
inflates the variance of the mean by $(1+\rho)/(1-\rho)$.

**No naive baseline.** Most series are close to random walks; beating them is
the bar.

**MAPE on data containing zeros.** Undefined, and asymmetric even when defined.

**Ignoring non-stationarity.** A model fitted on a trending series extrapolates
forever.

**Integer-encoding cyclical variables.** December and January end up maximally
distant.

**Flagging anomalies on raw values.** Flags every seasonal peak and misses
genuine anomalies near the mean.

**Using standard deviation for anomaly thresholds.** Masking
({{ch:ds-cleaning}}).

**Retraining on all history indefinitely.** If the process drifts, old data
describes a different world; a sliding window may beat an expanding one.

## 10. Connection to Previous Chapters

{{ch:ds-leakage}} introduced temporal leakage; this chapter develops the
validation schemes that prevent it. {{ch:ds-collection}} supplied the design
effect, which {{eq:ar1-design-effect}} is the temporal case of.
{{ch:ds-cleaning}} supplied the MAD and the masking argument that
{{eq:residual-anomaly}} relies on. {{ch:ds-feature-eng}} supplied the feature
families, constrained here by availability. {{ch:math-norms}} supplied
{{eq:cosine-euclidean}}, which {{eq:cyclical-distance}} reuses.
{{cite:kaufman2012}} treats temporal leakage formally.

Forward: {{ch:ds-recsys}} covers the other common i.i.d. violation, where the
system's own output shapes the next dataset.

Beyond Part III: {{ch:dl-rnns}} and {{part:7}} handle sequences with learned
models, and the ordering constraint established here is exactly what causal
masking enforces in {{ch:tf-masking-kv}}. {{ch:mle-drift}} monitors the
non-stationarity this chapter describes.

## 11. Exercises

**Beginner**

1. Why is random k-fold invalid for time series?
2. Name the three components of a decomposition.
3. What does stationarity mean, informally?
4. Why must a rolling feature be shifted?
5. Give two naive baselines for a daily series.

**Intermediate**

6. Using {{eq:ar1-design-effect}}, compute the effective sample size for 1,000
   daily observations with $\rho = 0.85$.
7. You forecast three days ahead. Which lags are available at prediction time?
8. Explain why MAPE is asymmetric, with a numerical example.
9. Encode the hour of day cyclically and verify that 23:00 and 00:00 are
   adjacent.
10. Give a case where a sliding window beats an expanding one.
11. Why should an anomaly detector work on residuals rather than raw values?

**Advanced**

12. Derive {{eq:ar1-limit}} from {{eq:ar1-mean-variance}}.
13. Show that for a random walk the optimal one-step forecast is the last value,
    and derive its expected squared error.
14. Explain why a gap is needed between train and test when features use long
    windows, and how large it should be.
15. Derive {{eq:cyclical-distance}} and relate it to
    {{eq:cosine-euclidean}}.

**Implementation**

16. Implement expanding-window and sliding-window validators with a configurable
    gap, and compare their estimates on a drifting series.
17. Write a feature builder that raises if any feature uses information from at
    or after the prediction time.
18. Implement MASE and compare several models against a seasonal naive baseline.
19. Build an anomaly detector using a seasonal-trend decomposition and robust
    residual thresholds, and measure precision and recall against planted
    anomalies.

**Reasoning**

20. Most business series are near random walks. What does that imply about the
    value of forecasting effort?
21. Should a production model retrain on all history or a recent window? What
    would you measure to decide?

## 12. Chapter Summary

Ordered data violates both halves of the i.i.d. assumption. Observations are
autocorrelated, so the effective sample size is smaller than the row count; and
the process is usually non-stationary, so the past describes a different world.

Series decompose into trend, seasonality and residual, additively or —
when the seasonal swing grows with the level — multiplicatively, which a log
transform converts to the additive case.

Autocorrelation inflates the variance of the sample mean by
$(1+\rho)/(1-\rho)$, which at $\rho = 0.9$ is a factor of nineteen. Confidence
intervals computed on the raw observation count are correspondingly
overconfident.

Random k-fold cross-validation trains on the future and reports a score that
cannot be achieved. Walk-forward validation — expanding or sliding, with a gap
when features use long windows — is the correct scheme.

Every feature must be computable from information available at the prediction
time. Lags must exceed the forecast horizon, rolling windows must be trailing
and shifted, and cyclical calendar variables need sine-cosine encoding so that
December and January are adjacent rather than maximally distant.

Always compare against a naive baseline. For a random walk the last value is the
provably optimal forecast, and many business series are close enough that
beating "tomorrow equals today" by a few percent is a genuine result. MASE
formalises this comparison; MAPE is undefined at zero and asymmetric.

Anomalies are points unlikely given the expected behaviour at that time, not
simply large values. Model the expectation, then apply a robust threshold to the
residuals — using the MAD rather than the standard deviation, because anomalies
inflate the statistic used to detect them.

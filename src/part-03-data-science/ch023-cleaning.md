---
id: ds-cleaning
number: 23
part: III
tier: focused
status: reviewed
requires: [ds-collection, py-pandas]
provides: [outlier, robust-statistic, winsorisation, cardinality]
citations: [wickham2014]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Classify missingness as MCAR, MAR or MNAR, and explain why the class
   determines what is safe to do.
2. Choose an imputation strategy with a stated justification and know what it
   distorts.
3. Distinguish the four reasons a value is an outlier and respond appropriately
   to each.
4. Use robust statistics and explain the breakdown point.
5. Classify feature types correctly, including the cases that look numeric and
   are not.
6. Reshape data into tidy form and say which tools require it.
7. Build a cleaning pipeline that is reproducible and fitted on training data
   only.

## 2. Why This Matters

Cleaning is the largest single block of effort on a real project
({{ch:ds-what-it-is}}), and it is where the most consequential silent decisions
get made.

The reason it is dangerous is that cleaning decisions are *invisible in the
output*. Drop rows with missing income and your model is now about people who
disclose income. Impute the mean and you have shrunk the variance and attenuated
every correlation ({{ch:math-covariance}}). Remove outliers by a fixed
z-score threshold and you may have deleted exactly the fraud cases you were
trying to detect. None of these raise an error, and all of them change the
answer.

The second reason is that cleaning is where leakage most often enters. Fitting
an imputer, a scaler, or an encoder on the full dataset before splitting leaks
test-set information into training, and the resulting validation score is
optimistic by an amount you cannot estimate ({{ch:ds-leakage}}).

So the discipline this chapter teaches is not a set of recipes. It is: **for
every cleaning decision, state why, and make it a fitted transformation rather
than an edit.**

## 3. Prerequisites

{{ch:ds-collection}} for provenance and contracts; {{ch:py-pandas}} for
missing-data mechanics and dtypes; {{ch:math-covariance}} for what imputation
does to variance and correlation.

## 4. Intuitive Explanation

### 4.1 Missingness has a mechanism

The critical question about a missing value is never *how many* but *why*. Three
standard classes, and they demand different responses.

**MCAR — missing completely at random.** The probability of being missing is
unrelated to anything. A sensor dropped packets at random. Dropping those rows
loses precision but introduces no bias.

**MAR — missing at random.** Missingness depends on *observed* variables. Older
users skip the income question more often, and you have age. Conditioning on the
observed variables makes it ignorable, so model-based imputation works.

**MNAR — missing not at random.** Missingness depends on the *unobserved value
itself*. High earners decline to state income. This is the difficult case: no
imputation from the observed data can recover it, because the information is
systematically absent.

```text
MCAR   missingness ⊥ everything          drop is unbiased
MAR    missingness ← observed vars       impute conditionally
MNAR   missingness ← the missing value   cannot be fixed from the data
```

> IMPORTANT: You cannot distinguish MAR from MNAR using the data alone — the
> distinguishing information is by definition unobserved. It is a claim about
> the data-generating process ({{ch:ds-what-it-is}}), and the honest response
> when you cannot rule out MNAR is an indicator column plus a caveat, not a
> cleverer imputation.

### 4.2 Missingness is often the signal

A frequently better response than imputing is to keep the fact of absence:

```python {tier=C name=missingness-indicator}
df["income_missing"] = df["income"].isna().astype(int)
df["income"] = df["income"].fillna(df["income"].median())
```

Two columns instead of one. The model can now learn both from the value and from
its absence, and if missingness carries signal — which under MAR and MNAR it
does — that signal is preserved rather than destroyed.

This costs one column and is almost always worth it.

### 4.3 Four kinds of outlier

An {{term:outlier}} is a value far from the rest. What to do depends entirely on
which of four things it is, and the distinction cannot be made statistically:

{#tbl:outlier-kinds caption="The four reasons a value is extreme. Only the first two justify removal, and telling them apart requires knowing the domain."}

| Kind | Example | Response |
|---|---|---|
| Data error | age = 999, a sentinel | fix or remove |
| Unit error | height 1.8 vs 180 | convert |
| Rare genuine event | a £2m order | **keep** — often the point |
| Different population | a bot in human traffic | segment, or exclude explicitly |

The third row is where automated outlier removal does the most damage. In fraud
detection, anomaly detection, and risk modelling, the extreme values *are the
signal*. A pipeline that clips at three standard deviations deletes the cases
the project exists to find.

### 4.4 Feature types

Getting the type wrong causes silent misbehaviour:

- **Numeric continuous** — height, revenue. Arithmetic is meaningful.
- **Numeric discrete** — counts. Arithmetic meaningful, values integral.
- **Ordinal** — small/medium/large. Order matters, spacing does not.
- **Nominal** — country, colour. No order at all.
- **Numeric-looking but nominal** — postcodes, user IDs, phone numbers.

The last category is the trap. A postcode stored as an integer will be averaged,
scaled and split on by any model that receives it, and every one of those
operations is meaningless. So will a category encoded as 1, 2, 3, which imposes
an ordering and a spacing that do not exist.

Ordinals need care in the other direction: encoding small/medium/large as 0/1/2
asserts that the gap from small to medium equals the gap from medium to large,
which may be false but is often a reasonable approximation — and is better than
discarding the order entirely with one-hot.

## 5. Formal Explanation

### 5.1 Imputation strategies

{#tbl:imputation caption="Imputation strategies, what each assumes, and what each distorts."}

| Strategy | Assumes | Distorts |
|---|---|---|
| Drop rows | MCAR | sample size; bias if not MCAR |
| Drop column | mostly missing | discards any signal it held |
| Mean / median | MCAR or MAR | **shrinks variance, attenuates correlation** |
| Mode (categorical) | MCAR | inflates the majority category |
| Constant sentinel | absence is meaningful | invents a value; needs a model that can use it |
| Forward fill | value persists over time | **leaks the future if applied before sorting** |
| Model-based (kNN, iterative) | MAR | understates uncertainty; can leak |
| Indicator + simple fill | anything | one extra column |

The last row is the pragmatic default. It is robust to the mechanism because it
does not try to guess the value; it records that the value was absent and lets
the model decide what that means.

### 5.2 What mean imputation does

Imputing the mean of the observed values leaves the mean unchanged and shrinks
the variance. If a fraction $f$ of values are missing and are replaced by the
observed mean, the variance of the completed variable is

$$
\Var_{\text{imputed}} = (1 - f)\,\Var_{\text{observed}}
$$ (eq:imputation-variance)

because the imputed values contribute zero squared deviation.

The consequence for correlation follows. Since
$\rho = \Cov(X,Y)/(\sigma_X\sigma_Y)$ ({{ch:math-covariance}}) and the imputed
rows contribute nothing to the covariance either, the correlation is attenuated
by approximately

$$
\rho_{\text{imputed}} \approx \rho_{\text{true}}\sqrt{1 - f}
$$ (eq:correlation-attenuation)

At 20% missing, correlations shrink by about 11%; at 50%, by 29%. The model then
sees a weaker relationship than exists, and every downstream coefficient is
biased toward zero. {{sec:7-implementation}} measures both effects.

### 5.3 Robust statistics

A {{term:robust-statistic}} resists contamination. The measure is the
**breakdown point**: the fraction of arbitrarily corrupted observations the
estimator tolerates before it can be driven anywhere.

{#tbl:breakdown caption="Breakdown points. One bad value is enough to destroy the mean; half the data must be corrupted to destroy the median."}

| Statistic | Breakdown point |
|---|---|
| Mean | 0% — one point moves it arbitrarily |
| Standard deviation | 0% |
| Median | 50% |
| Interquartile range | 25% |
| Median absolute deviation | 50% |
| Trimmed mean (10%) | 10% |

For outlier detection this matters directly. The z-score rule uses the mean and
standard deviation, both of which the outliers themselves have already
corrupted — a phenomenon called **masking**. The robust alternative uses the
median and the median absolute deviation:

$$
\text{MAD} = \operatorname{median}\big(\lvert x_i - \operatorname{median}(x) \rvert\big)
$$ (eq:mad)

$$
z_{\text{robust}} = \frac{0.6745\,(x_i - \operatorname{median}(x))}{\text{MAD}}
$$ (eq:robust-z)

The constant $0.6745$ makes the MAD a consistent estimator of the standard
deviation for normally distributed data, so the robust score is interpretable on
the same scale as an ordinary z-score.

### 5.4 Tidy data

{{cite:wickham2014}} defines a target shape for tabular data:

1. Each variable is a column.
2. Each observation is a row.
3. Each type of observational unit is a table.

{{term:tidy-data}} matters because most tools assume it. Plotting libraries,
statistical models, and `groupby` all expect one row per observation. Common
violations and their fixes:

- **Column headers are values** (`2023`, `2024` as columns) → `melt`.
- **Multiple variables in one column** (`"height_cm"`, `"weight_kg"` stacked) →
  split then `pivot`.
- **Variables in both rows and columns** → `pivot` after separating.
- **Multiple observational units in one table** → split into several tables.

### 5.5 Cleaning as a fitted transformation

The single most important structural point in this chapter.

Every cleaning step that computes something from the data — a mean to impute, a
median for outlier bounds, a set of known categories, a scaler's statistics — is
a **parameter learned from the data**. It must therefore be fitted on the
training set only and applied to validation and test, exactly like a model.

```python {tier=C name=fit-transform-discipline}
# WRONG: statistics computed over everything, then split
df["income"] = df["income"].fillna(df["income"].median())
train, test = split(df)

# RIGHT: split first, fit on train, apply to both
train, test = split(df)
median = train["income"].median()          # a fitted parameter
train["income"] = train["income"].fillna(median)
test["income"] = test["income"].fillna(median)
```

The first version leaks the test set's distribution into training. The effect is
usually small for a median and can be large for target-derived quantities
({{ch:ds-feature-eng}}), but the discipline should not depend on estimating how
much you got away with.

## 6. Mathematical Foundation

### 6.1 Deriving the variance shrinkage

Let $n$ values have observed mean $\mu$ and variance $\sigma^{2}$ over the
$m = (1-f)n$ observed entries. Replace the $fn$ missing entries with $\mu$.

The completed mean is unchanged, since the added values equal the mean. The
completed variance is

$$
\Var_{\text{imp}} = \frac{1}{n}\left[\sum_{\text{observed}}(x_i - \mu)^{2}
  + \sum_{\text{imputed}}(\mu - \mu)^{2}\right]
= \frac{m\sigma^{2}}{n} = (1-f)\sigma^{2}
$$ (eq:variance-shrinkage-derivation)

The imputed terms contribute exactly zero. Every imputed point sits precisely at
the mean, which is the one location contributing nothing to the spread.

The same argument applies to the covariance: imputed rows have zero deviation in
the imputed variable, so they contribute nothing to $\Cov(X, Y)$, while still
counting in $n$. With missingness in $X$ only:

$$
\Cov_{\text{imp}} = (1 - f)\Cov_{\text{true}},
\qquad
\sigma_{X,\text{imp}} = \sqrt{1-f}\,\sigma_{X}
$$

so

$$
\rho_{\text{imp}} = \frac{(1-f)\Cov}{\sqrt{1-f}\,\sigma_X \sigma_Y}
  = \sqrt{1-f}\;\rho_{\text{true}}
$$ (eq:attenuation-derivation)

which is {{eq:correlation-attenuation}}.

### 6.2 Why the z-score rule fails on the outliers it is looking for

Take nine values near 10 and one value of 1000.

The mean is $(9 \times 10 + 1000)/10 = 109$. The standard deviation is about
297. The z-score of the 1000 is $(1000 - 109)/297 \approx 3.0$ — right at the
usual threshold, and it may not be flagged at all.

Worse, the nine normal values now have z-scores around $(10-109)/297 = -0.33$,
so nothing looks unusual anywhere.

The robust version: the median is 10, and the MAD is the median of
$\{0,0,0,0,0,1,1,1,1,990\}$-style deviations — for nine values at exactly 10 and
one at 1000, the deviations are nine zeros and one 990, so the MAD is 0. When
the MAD is zero the scaled score is undefined, and implementations fall back to
a small epsilon, giving an enormous robust score for the outlier and zero for
everything else.

With slightly noisier normal values the MAD is small but nonzero, and the robust
z-score of the outlier is in the hundreds. {{sec:7-implementation}} shows both
regimes.

> MATH NOTE: This is **masking**: outliers inflate the very statistics used to
> detect them, hiding themselves and each other. It is worse with several
> outliers than with one, which is exactly the situation where detection matters
> most.

### 6.3 The bounds used in practice

Two standard rules:

$$
\text{Tukey}: \quad [Q_1 - 1.5\,\text{IQR},\; Q_3 + 1.5\,\text{IQR}]
$$ (eq:tukey-fences)

$$
\text{Robust z}: \quad \lvert z_{\text{robust}} \rvert > 3.5
$$ (eq:robust-z-threshold)

For normally distributed data, Tukey's fences flag about 0.7% of points and the
robust-z rule about 0.05%. Neither constant is derived from anything deep; both
are conventions chosen to flag few points under normality.

That has an implication worth stating: **on skewed data, Tukey's fences flag a
large fraction of the upper tail even when nothing is wrong.** Income, revenue,
session length and file size are all right-skewed, and applying the rule
unmodified marks legitimate large values as outliers. Log-transform first, or
use quantile-based {{term:winsorisation}} instead.

## 7. Implementation

```python {tier=A name=missingness-and-outliers}
"""Missingness mechanisms, imputation distortion, and robust outlier detection.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)

# --- eq. 23.1 / 23.2: what mean imputation costs ----------------------------
print("=" * 70)
print("mean imputation shrinks variance and attenuates correlation")
print("=" * 70)

n = 200_000
x = rng.normal(50, 10, n)
y = 0.7 * x + rng.normal(0, 7.14, n)          # correlation ~0.7
true_rho = np.corrcoef(x, y)[0, 1]
true_var = x.var()

print(f"complete data: var(x) = {true_var:.2f}, corr = {true_rho:.4f}\n")
print(f"{'missing':>8} {'var after':>11} {'predicted':>11} "
      f"{'corr after':>12} {'predicted':>11}")
for f in (0.1, 0.2, 0.4, 0.6):
    xm = x.copy()
    mask = rng.random(n) < f                   # MCAR
    xm[mask] = np.nan
    filled = np.where(np.isnan(xm), np.nanmean(xm), xm)
    obs_f = mask.mean()
    print(f"{obs_f:>8.1%} {filled.var():>11.2f} {(1-obs_f)*true_var:>11.2f} "
          f"{np.corrcoef(filled, y)[0,1]:>12.4f} "
          f"{true_rho*np.sqrt(1-obs_f):>11.4f}")

print("\nBoth match eqs. 23.1 and 23.2. Every coefficient fitted on imputed")
print("data is biased toward zero by a factor you can compute in advance.")

# --- the three mechanisms behave differently --------------------------------
print("\n" + "=" * 70)
print("MCAR / MAR / MNAR: dropping rows is only safe for one of them")
print("=" * 70)

income = rng.lognormal(10.5, 0.6, n)
age = rng.uniform(18, 80, n)
true_mean = income.mean()

mcar = rng.random(n) < 0.3                                     # unrelated
mar = rng.random(n) < (age - 18) / 62 * 0.6                    # depends on age
mnar = rng.random(n) < 1 / (1 + np.exp(-(income - 60_000) / 20_000))

print(f"true mean income: £{true_mean:,.0f}\n")
print(f"{'mechanism':<8} {'% missing':>10} {'mean after dropping':>21} "
      f"{'bias':>10}")
for name, mask in (("MCAR", mcar), ("MAR", mar), ("MNAR", mnar)):
    kept = income[~mask].mean()
    print(f"{name:<8} {mask.mean():>10.1%} £{kept:>19,.0f} "
          f"{kept - true_mean:>+10,.0f}")

print("\nMCAR: unbiased. MAR: biased here because age correlates with income,")
print("and conditioning on age would fix it. MNAR: badly biased and NOT")
print("fixable from the observed data — the high earners are simply absent.")

# --- masking: outliers hide from the z-score rule ---------------------------
print("\n" + "=" * 70)
print("masking: the z-score rule fails on the outliers it looks for")
print("=" * 70)


def zscore_flags(v, thresh=3.0):
    z = (v - v.mean()) / v.std(ddof=0)
    return np.abs(z) > thresh


def robust_flags(v, thresh=3.5):
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    if mad == 0:
        mad = 1e-9
    z = 0.6745 * (v - med) / mad               # eq. 23.5
    return np.abs(z) > thresh


base = rng.normal(10, 1, 40)
for n_out, label in ((1, "one outlier"), (5, "five outliers")):
    data = np.concatenate([base, np.full(n_out, 1000.0)])
    zf, rf = zscore_flags(data), robust_flags(data)
    z_of_outlier = ((data - data.mean()) / data.std(ddof=0))[-1]
    med, mad = np.median(data), np.median(np.abs(data - np.median(data)))
    r_of_outlier = 0.6745 * (1000 - med) / max(mad, 1e-9)
    print(f"\n{label} of 1000 among 40 values near 10:")
    print(f"  mean {data.mean():>8.1f}, sd {data.std(ddof=0):>8.1f}  "
          f"-> z-score of the outlier {z_of_outlier:>6.2f}")
    print(f"  median {med:>6.1f}, MAD {mad:>7.2f}  "
          f"-> robust z {r_of_outlier:>10.1f}")
    print(f"  z-score rule flags {zf.sum()}/{n_out}, "
          f"robust rule flags {rf.sum()}/{n_out}")

print("\nWith five outliers the z-score rule finds NONE of them: they have")
print("inflated the standard deviation enough to hide themselves. The robust")
print("rule is unaffected because the median and MAD ignore them.")

# --- eq. 23.6: Tukey fences on skewed data ----------------------------------
print("\n" + "=" * 70)
print("Tukey's fences assume symmetry")
print("=" * 70)


def tukey_flags(v):
    q1, q3 = np.percentile(v, [25, 75])
    iqr = q3 - q1
    return (v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)


normal_data = rng.normal(0, 1, 100_000)
skewed = rng.lognormal(0, 1, 100_000)          # legitimately right-skewed
logged = np.log(skewed)

print(f"{'data':<28} {'% flagged by Tukey':>20}")
print(f"{'normal':<28} {tukey_flags(normal_data).mean():>19.2%}")
print(f"{'lognormal (raw)':<28} {tukey_flags(skewed).mean():>19.2%}")
print(f"{'lognormal (log-transformed)':<28} {tukey_flags(logged).mean():>19.2%}")
print("\nOn skewed data the rule flags 4% of perfectly legitimate values.")
print("Log-transform first, or use quantile clipping instead.")

# --- winsorisation keeps the row and limits its influence -------------------
print("\n" + "=" * 70)
print("winsorising vs dropping")
print("=" * 70)
revenue = np.concatenate([rng.gamma(2, 100, 5000), [250_000, 310_000]])
lo, hi = np.percentile(revenue, [1, 99])
wins = np.clip(revenue, lo, hi)
dropped = revenue[(revenue >= lo) & (revenue <= hi)]

print(f"{'treatment':<16} {'n':>7} {'mean':>10} {'sd':>10} {'total':>12}")
for label, v in (("raw", revenue), ("winsorised", wins), ("dropped", dropped)):
    print(f"{label:<16} {len(v):>7,} {v.mean():>10.1f} {v.std():>10.1f} "
          f"{v.sum():>12,.0f}")
lost_rows = len(revenue) - len(dropped)
lost_value = revenue.sum() - dropped.sum()
print(f"\nDropping removes {lost_rows} rows and £{lost_value:,.0f} of recorded")
print("revenue. Winsorising keeps every customer and caps their leverage,")
print(f"costing £{revenue.sum() - wins.sum():,.0f} of recorded value instead.")
print("Which is right depends entirely on whether those two orders were real.")
```

## 8. Practical Example

A cleaning pipeline that is fitted rather than applied is the deliverable this
chapter is aiming at. It must produce identical results on new data and must
never see the test set.

```python {tier=A name=fitted-cleaning-pipeline}
"""A cleaning pipeline as a fitted transformation.

Every learned quantity — medians, categories, bounds — is a parameter fitted on
training data and applied unchanged to anything else. That is what makes the
validation score honest and the production behaviour predictable.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

rng = np.random.default_rng(5)


@dataclass
class Cleaner:
    """Fit on train only; transform anything."""

    numeric: list[str]
    categorical: list[str]
    clip_quantiles: tuple[float, float] = (0.01, 0.99)

    medians_: dict = field(default_factory=dict)
    bounds_: dict = field(default_factory=dict)
    categories_: dict = field(default_factory=dict)
    fitted_: bool = False

    def fit(self, df: pd.DataFrame) -> "Cleaner":
        for c in self.numeric:
            s = pd.to_numeric(df[c], errors="coerce")
            self.medians_[c] = float(s.median())
            self.bounds_[c] = tuple(
                float(v) for v in s.quantile(self.clip_quantiles))
        for c in self.categorical:
            norm = df[c].astype("string").str.strip().str.lower()
            self.categories_[c] = sorted(norm.dropna().unique())
        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("transform called before fit")
        out = df.copy()
        for c in self.numeric:
            s = pd.to_numeric(out[c], errors="coerce")
            out[f"{c}_missing"] = s.isna().astype("int8")   # keep the signal
            lo, hi = self.bounds_[c]
            out[c] = s.fillna(self.medians_[c]).clip(lo, hi)
        for c in self.categorical:
            norm = out[c].astype("string").str.strip().str.lower()
            known = set(self.categories_[c])
            out[f"{c}_unseen"] = (~norm.isin(known) & norm.notna()).astype("int8")
            out[c] = norm.where(norm.isin(known), "__other__").fillna("__missing__")
        return out


# --- a messy dataset with realistic defects ---------------------------------
def make_data(n, seed, shift=False):
    r = np.random.default_rng(seed)
    income = r.lognormal(10.4 + (0.2 if shift else 0), 0.6, n)
    income[r.random(n) < 0.18] = np.nan                     # missing
    return pd.DataFrame({
        "income": income,
        "age": np.where(r.random(n) < 0.05, 999, r.integers(18, 85, n)),
        "city": r.choice(["London", " london", "LEEDS", "Bristol", None], n,
                         p=[.3, .1, .3, .25, .05]),
        "target": (r.random(n) < 0.3).astype(int),
    })


train = make_data(6000, seed=1)
test = make_data(2000, seed=2, shift=True)      # deliberately shifted

print("BEFORE cleaning:")
print(f"  income nulls   : {train['income'].isna().mean():.1%}")
print(f"  age == 999     : {(train['age'] == 999).sum()} sentinel values")
print(f"  city variants  : {sorted(train['city'].dropna().unique())}")

# The sentinel must be handled before anything statistical touches it.
for d in (train, test):
    d.loc[d["age"] == 999, "age"] = np.nan

cleaner = Cleaner(numeric=["income", "age"], categorical=["city"]).fit(train)

print(f"\nfitted parameters (learned from TRAIN only):")
print(f"  income median  : £{cleaner.medians_['income']:,.0f}")
print(f"  income bounds  : £{cleaner.bounds_['income'][0]:,.0f} – "
      f"£{cleaner.bounds_['income'][1]:,.0f}")
print(f"  known cities   : {cleaner.categories_['city']}")

train_c = cleaner.transform(train)
test_c = cleaner.transform(test)

print(f"\nAFTER cleaning:")
print(f"  nulls remaining     : {int(train_c.isna().sum().sum())}")
print(f"  city variants       : {sorted(train_c['city'].unique())}")
print(f"  indicator columns   : "
      f"{[c for c in train_c.columns if c.endswith(('_missing', '_unseen'))]}")

# --- the test set is transformed with TRAIN parameters ----------------------
print(f"\nthe test set is shifted upward, and is clipped using the TRAIN")
print(f"bounds — which is exactly what will happen in production:")
raw_test_income = pd.to_numeric(test["income"], errors="coerce")
print(f"  test raw max      : £{raw_test_income.max():,.0f}")
print(f"  test after clip   : £{test_c['income'].max():,.0f}  "
      f"(train's 99th percentile)")
print(f"  fraction clipped  : "
      f"{(raw_test_income > cleaner.bounds_['income'][1]).mean():.1%}")

# --- leakage: what fitting on everything would have cost --------------------
print("\n" + "=" * 70)
print("why the fit/transform split matters")
print("=" * 70)
combined = pd.concat([train, test], ignore_index=True)
leaky = Cleaner(numeric=["income", "age"], categorical=["city"]).fit(combined)
print(f"median fitted on train only : £{cleaner.medians_['income']:,.0f}")
print(f"median fitted on everything : £{leaky.medians_['income']:,.0f}")
print(f"difference                  : "
      f"£{leaky.medians_['income'] - cleaner.medians_['income']:,.0f}")
print("\nSmall here, and it is the wrong thing to measure. The leaky version")
print("used information from the test set, so its validation score no longer")
print("estimates production performance — by an amount you cannot know.")
print("The discipline should not depend on how much you got away with.")

# --- transform is deterministic and idempotent ------------------------------
again = cleaner.transform(test)
assert again.equals(test_c)
print(f"\ntransform is deterministic: repeated application is identical.")
```

## 9. Common Mistakes

**Imputing without asking why the value is missing.** The mechanism determines
what is valid.

**Mean-imputing and then reporting correlations.** They are attenuated by
$\sqrt{1-f}$ ({{eq:correlation-attenuation}}).

**Dropping rows with any missing value.** On wide data this can discard most of
the dataset, and it is only unbiased under MCAR.

**Removing outliers automatically.** In fraud, anomaly and risk work the
outliers are the target.

**Using the z-score rule to find outliers.** Masking: the outliers corrupt the
statistics used to detect them.

**Applying Tukey's fences to skewed data.** Flags a large fraction of a
legitimate tail. Transform first.

**Treating numeric-looking identifiers as numeric.** Postcodes and user IDs get
averaged and scaled, meaninglessly.

**Encoding nominal categories as integers.** Imposes a false order and spacing.

**Cleaning before splitting.** Leakage ({{ch:ds-leakage}}).

**Cleaning as edits rather than a fitted transformation.** Cannot be applied
consistently to new data, and cannot be reproduced.

**Silently dropping unseen categories at prediction time.** Decide explicitly
what happens, and record that it happened.

## 10. Connection to Previous Chapters

{{ch:ds-collection}} supplied the contracts that catch many of these defects at
ingestion, and the provenance needed to answer why a value is missing.
{{ch:math-covariance}} supplied the variance and correlation that
{{sec:6-mathematical-foundation}} shows imputation distorts, and the robust
statistics discussion extends its treatment of the mean.
{{ch:py-pandas}} supplied the `NaN` mechanics and the fit/transform discipline
that {{sec:5-formal-explanation}} formalises. {{cite:wickham2014}} defines the
tidy target shape.

Forward: {{ch:ds-eda}} looks at the cleaned data and frequently sends you back
here; {{ch:ds-feature-eng}} builds on the cleaned columns; {{ch:ds-leakage}}
formalises why fitting before splitting is a mistake.

Beyond Part III: {{ch:mle-pipelines}} makes the `Cleaner` of
{{sec:8-practical-example}} a first-class pipeline stage, and
{{ch:ml-knn-nb}} shows how model choice interacts with imputation.

## 11. Exercises

**Beginner**

1. Classify each as MCAR, MAR or MNAR: a sensor dropping random readings; older
   users skipping a question; high earners declining to state income.
2. Give the two-line pattern that preserves missingness as a signal.
3. Name the four kinds of outlier and the right response to each.
4. Which of postcode, age, satisfaction rating (1-5) and user ID are genuinely
   numeric?
5. State the three rules of tidy data.

**Intermediate**

6. Using {{eq:correlation-attenuation}}, predict the observed correlation when
   the true value is 0.6 and 35% of one variable is imputed with its mean.
7. Construct data where the z-score rule fails to flag an obvious outlier, and
   show the robust rule succeeds.
8. Explain why Tukey's fences flag ~4% of lognormal data and ~0.7% of normal
   data.
9. Give a case where dropping rows with missing values changes the population
   the model is about.
10. Why must a category encoder be fitted on training data only? What should
    happen to an unseen category at prediction time?
11. Convert a wide table with year columns into tidy form and back.

**Advanced**

12. Derive {{eq:variance-shrinkage-derivation}} and extend it to the case where
    imputation uses the median rather than the mean.
13. Explain masking formally: show how $k$ outliers inflate the sample standard
    deviation and derive the threshold at which they become undetectable.
14. Multiple imputation generates several completed datasets. Explain what it
    fixes that single imputation does not.
15. You cannot distinguish MAR from MNAR from the data. What external evidence
    would let you argue for one, and what would you report if you cannot?

**Implementation**

16. Extend the `Cleaner` to support ordinal columns with a specified order, and
    to serialise its fitted parameters to JSON.
17. Implement Tukey and robust-z detectors and compare their flag rates on
    normal, lognormal and contaminated data.
18. Write a test proving the cleaner is deterministic, never mutates its input,
    and raises if `transform` is called before `fit`.
19. Empirically verify {{eq:correlation-attenuation}} across missingness
    fractions from 0 to 0.8 and plot predicted against observed.

**Reasoning**

20. Automated cleaning tools apply default strategies. Given this chapter, what
    is the risk, and when is it acceptable?
21. An agent proposes dropping all rows with any missing value, leaving 40% of
    the data. What questions do you ask before accepting?

## 12. Chapter Summary

Cleaning is the largest block of effort and the place where the most
consequential invisible decisions are made. Every decision should have a stated
reason and should be a fitted transformation rather than an edit.

Missingness has a mechanism. MCAR permits dropping without bias; MAR is
ignorable after conditioning on observed variables; MNAR cannot be repaired from
the observed data. The three cannot be distinguished from the data alone,
because the distinguishing information is by definition unobserved — which makes
an indicator column plus a simple fill the robust default.

Mean imputation shrinks variance by exactly $(1-f)$ and attenuates correlations
by $\sqrt{1-f}$, so every coefficient fitted on imputed data is biased toward
zero by a computable amount.

Outliers come in four kinds and only two justify removal. Automated removal is
actively harmful in fraud, anomaly and risk work, where the extreme values are
the signal.

The z-score rule fails at exactly the job it is used for: outliers inflate the
mean and standard deviation used to detect them, masking themselves and each
other. The median and MAD have 50% breakdown points and are unaffected. Tukey's
fences assume symmetry and flag a large fraction of any legitimate right-skewed
tail, so transform before applying them.

Feature types must be classified correctly. Numeric-looking identifiers and
integer-encoded nominal categories both invite arithmetic that is meaningless.

Every cleaning step that computes a statistic is a fitted parameter. Fit on
training data, apply everywhere else — because the alternative leaks, and the
discipline should not depend on estimating how much.

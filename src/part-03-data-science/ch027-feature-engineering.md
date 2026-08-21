---
id: ds-feature-eng
number: 27
part: III
tier: focused
status: reviewed
requires: [ds-eda, py-pandas]
provides: [one-hot-encoding, target-encoding, feature-engineering,
           feature-selection, feature-store]
citations: [automind2025]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Explain what feature engineering buys and when it stops mattering.
2. Encode categorical variables appropriately for their cardinality.
3. Apply target encoding without leaking, using out-of-fold computation.
4. Build aggregate, ratio, temporal and interaction features.
5. Transform skewed variables and explain what each transform assumes.
6. Select features by filter, wrapper and embedded methods, and know each one's
   failure mode.
7. Explain training-serving skew and what a feature store solves.

## 2. Why This Matters

For tabular data, feature engineering is usually worth more than model choice.
The difference between a linear model on well-constructed features and gradient
boosting on raw columns is frequently in the former's favour, and the difference
between two model families on the same good features is usually small
({{ch:ml-boosting}}).

The reason is that a feature encodes knowledge the model cannot derive. A model
sees `signup_date` and `last_login` as two numbers; it does not know that their
difference is tenure and that tenure predicts everything. It can learn a
threshold on each, but it cannot construct the subtraction. Supplying it is
supplying domain knowledge in a form the model can use.

Two cautions frame the chapter. **Deep learning changed where this applies.**
For text, images and audio, learned representations comprehensively beat
hand-engineered features, and Parts VI onward are largely about that. For
tabular data with tens of columns and thousands of rows — which is most business
data — hand-engineered features remain competitive and often superior.

**Feature engineering is where leakage enters.** Target encoding, aggregates
computed over the full dataset, and any feature using information from after the
prediction point all leak, and all look reasonable. {{ch:ds-leakage}} is the
next chapter for a reason.

## 3. Prerequisites

{{ch:ds-eda}} for what exploration reveals about which features to build;
{{ch:py-pandas}} for `groupby`/`transform`; {{ch:ds-cleaning}} for the
fit-on-train-only discipline this chapter depends on entirely.

## 4. Intuitive Explanation

### 4.1 A feature encodes knowledge

Consider predicting churn. The raw table has `signup_date`, `last_login`,
`total_spend`, `n_orders`.

A model can use those directly. What it cannot do is construct:

- **tenure** = today − signup_date
- **recency** = today − last_login
- **average order value** = total_spend / n_orders
- **order frequency** = n_orders / tenure
- **recency relative to their own norm** = recency / median gap between orders

Every one of these is a simple arithmetic combination, and every one encodes
something a human knows about how customers behave. The last is the most
valuable and the least likely to be discovered by a model: it compares a user
against their own baseline rather than against the population.

Tree models can approximate ratios by splitting repeatedly, but they need far
more data to do it and the result is a step function rather than a smooth
relationship. Linear models cannot represent a ratio at all.

### 4.2 The four families

Most useful features fall into four groups:

**Aggregates.** Group statistics: mean order value per customer, transaction
count per merchant per day, standard deviation of a sensor per hour. Computed
with `groupby`/`transform` ({{ch:py-pandas}}).

**Ratios and differences.** Rates, proportions, per-unit values, and the
difference between a value and a reference — a group mean, a previous value, or
an expected value.

**Temporal.** Time since an event, time until a deadline, day of week, hour of
day, rolling statistics over a window. Almost always high-value and almost
always the source of leakage.

**Interactions.** Products or combinations of features. Trees find these
automatically given enough depth; linear models need them supplied.

### 4.3 Encoding categories

The right encoding depends on {{term:cardinality}}:

{#tbl:encoding-choice caption="Encoding categorical variables by cardinality. The high-cardinality row is where leakage enters."}

| Cardinality | Encoding | Note |
|---|---|---|
| 2 | binary 0/1 | trivial |
| 3-15 | {{term:one-hot-encoding}} | safe, interpretable |
| 15-50 | one-hot, or grouping rare levels | width becomes a concern |
| 50-1000 | {{term:target-encoding}}, frequency encoding | **leakage risk** |
| >1000 | target encoding, hashing, or learned embeddings | ({{part:11}}) |

Ordinal variables are the exception: encode them as ordered integers, preserving
the order that one-hot would discard.

> WARNING: Never encode a nominal category as an arbitrary integer for a linear
> model or a distance-based one. Assigning London=1, Leeds=2, Bristol=3 asserts
> that Leeds is between the other two and equidistant from both. Tree models
> tolerate it because they only split on thresholds, but even there it forces
> arbitrary groupings.

### 4.4 Target encoding, and why it is dangerous

Target encoding replaces a category with the mean target for that category. For
high-cardinality columns it is compact and powerful.

It is also the single easiest way to leak. If a user ID appears once and you
encode it with that row's own target, the feature *is* the target. Validation
accuracy will be near perfect and production performance will be at baseline.

The fix has two parts, and both are required:

**Out-of-fold computation.** A row's encoding is computed from the *other*
folds, never from its own row.

**Smoothing toward the global mean.** A category with three observations should
not get its raw mean, because that mean is mostly noise.

## 5. Formal Explanation

### 5.1 Smoothed target encoding

For category $c$ with $n_c$ observations and mean target $\bar{y}_c$, against a
global mean $\bar{y}$:

$$
\text{enc}(c) = \frac{n_c\,\bar{y}_c + m\,\bar{y}}{n_c + m}
$$ (eq:target-encoding)

The smoothing parameter $m$ is a pseudo-count: it acts as $m$ extra observations
at the global mean. Categories with $n_c \gg m$ keep their own mean; categories
with $n_c \ll m$ are pulled toward the global one.

This is exactly a Bayesian posterior mean with a prior centred on $\bar{y}$ and
strength $m$ — the shrinkage argument of {{ch:math-probability}} applied to a
per-category rate.

Choosing $m$: it is the category size at which you trust the category's own mean
half as much as the global mean. Values of 10-50 are typical, and it should be
tuned by cross-validation like any hyperparameter.

### 5.2 Out-of-fold encoding

$$
\text{enc}_i(c) = f\big(\{y_j : \text{cat}_j = c,\; j \notin \text{fold}(i)\}\big)
$$ (eq:out-of-fold-encoding)

Row $i$'s encoding uses only rows outside its own fold. At prediction time the
encoding is computed from the *entire* training set, since test rows were never
part of it.

> IMPORTANT: The asymmetry is deliberate and correct. During training each row
> must be encoded without seeing its own target, or the feature leaks. At
> prediction time there is no such constraint, because the new row contributed
> nothing to the statistics. Getting this backwards — using out-of-fold encoding
> at prediction time, or full-data encoding during training — is the standard
> way this goes wrong.

### 5.3 Transformations for skew

{#tbl:transforms caption="Transformations for skewed variables, and what each requires."}

| Transform | Formula | Requires | Use for |
|---|---|---|---|
| log | $\log(x)$ | $x > 0$ | multiplicative, right-skewed |
| log1p | $\log(1+x)$ | $x \ge 0$ | counts with zeros |
| square root | $\sqrt{x}$ | $x \ge 0$ | mild skew, Poisson-like |
| Box-Cox | $(x^{\lambda}-1)/\lambda$ | $x > 0$ | fitted $\lambda$ |
| Yeo-Johnson | piecewise | any real | handles negatives |
| rank / quantile | to uniform or normal | any | robust; discards magnitude |

The log transform is the workhorse because so many quantities are
multiplicative. Revenue, income, session length, file size and city population
are all approximately lognormal, and $\log$ makes them approximately normal
({{ch:math-functions}}).

A transform also changes what a linear model means. Regressing $\log y$ on $x$
models *proportional* change: a coefficient of 0.05 means a 5% increase in $y$
per unit of $x$, not an increase of 0.05.

### 5.4 Feature selection

Three families, with different costs and failure modes:

**Filter** — score each feature independently, keep the top $k$. Fast, and blind
to interactions and redundancy. A feature useless alone but valuable in
combination is discarded, and two identical features both score highly.

**Wrapper** — search subsets, evaluating a model on each. Forward selection,
backward elimination, recursive feature elimination. Finds interactions;
expensive, and prone to overfitting the selection criterion when the search
space is large.

**Embedded** — selection happens during fitting. $L_1$ regularisation
({{ch:math-norms}}) drives coefficients to exactly zero; tree ensembles produce
importances. Usually the best default: it costs nothing extra and accounts for
redundancy.

> WARNING: Feature selection is a fitted step and must happen **inside**
> cross-validation. Selecting features on the full dataset and then
> cross-validating produces optimistic scores, because the selection already saw
> the validation folds. {{ch:ds-leakage}} quantifies how large this effect can
> be — with enough noise features it can manufacture apparent signal from pure
> noise.

### 5.5 Training-serving skew and feature stores

A feature computed one way in a training notebook and another way in production
means the model was validated on inputs it will never receive. This is
{{term:training-serving-skew}}, and it is one of the most common production
failures.

Typical causes: a training aggregate computed over all history while production
computes it over 30 days; a timezone difference; a null handled differently; a
category normalised in one path and not the other.

A {{term:feature-store}} solves it by defining each feature once and serving it
consistently to both paths. The essential property is a **single definition**,
not the infrastructure — for a small project a shared library function achieves
the same thing.

### 5.6 What automation has changed

Automated feature engineering — generating aggregates and combinations
systematically — is mature, and agent frameworks now propose features from a
description of the problem {{cite:automind2025}}.
{{maturity:EMERGING}}

What these do well is enumerate the mechanical families of
{{sec:4-intuitive-explanation}} exhaustively. What they do not do is know that
`recency relative to that user's own median gap` is meaningful because of how
subscriptions work, or that a particular ratio is the one the domain cares
about, or that a proposed feature will not exist at prediction time. The first
requires domain knowledge and the last requires knowing the serving architecture
— and it is the failure mode of {{ch:ds-leakage}}, which automated generation
makes *more* likely by producing many plausible candidates quickly.

## 6. Mathematical Foundation

### 6.1 Why smoothing is a posterior mean

Model the target for category $c$ as Bernoulli with rate $\theta_c$, and place a
Beta prior centred on the global rate $\bar{y}$ with strength $m$:

$$
\theta_c \sim \text{Beta}\big(m\bar{y},\; m(1-\bar{y})\big)
$$

Observing $k$ successes in $n_c$ trials, the Beta-Binomial conjugacy gives a
posterior mean of

$$
\E[\theta_c \mid \text{data}] = \frac{k + m\bar{y}}{n_c + m}
  = \frac{n_c\bar{y}_c + m\bar{y}}{n_c + m}
$$ (eq:posterior-mean-encoding)

which is exactly {{eq:target-encoding}}. The smoothing parameter is the prior's
strength in pseudo-observations, and the formula is not a heuristic but the
Bayesian answer.

The behaviour follows: with $n_c \to \infty$ the estimate approaches
$\bar{y}_c$; with $n_c = 0$ it is exactly $\bar{y}$. A category seen three times
is shrunk hard toward the global mean, which is correct, because three
observations carry little information.

### 6.2 How much a leaking encoding inflates the score

Suppose a categorical column has one distinct value per row — a user ID, say.
Encode it with the row's own target and the feature equals the target exactly.

More realistically, with $n_c$ rows per category and no smoothing, the row's own
target contributes $1/n_c$ of the encoding. The correlation between the
encoding and the target is then approximately

$$
\rho \approx \sqrt{\frac{1}{n_c}}
$$ (eq:leak-correlation)

for a category with no genuine signal. At $n_c = 4$ that is a correlation of
0.5 with the target, manufactured entirely from the leak. The model finds it,
validation accuracy rises, and production accuracy does not.

{{sec:7-implementation}} demonstrates this on pure noise: a target-encoded
random ID column achieves high in-sample accuracy and chance accuracy on held-out
data.

### 6.3 What a log transform does to a relationship

If $y = a x^{b}$, then

$$
\log y = \log a + b \log x
$$ (eq:log-log)

A multiplicative power relationship becomes linear in log-log space. This is why
log transforms so often "fix" a curved scatter plot: the underlying relationship
was multiplicative rather than additive.

For a single-sided transform, regressing $\log y$ on $x$ gives

$$
\log y = \alpha + \beta x
\quad\Longrightarrow\quad
\frac{\dd y / y}{\dd x} = \beta
$$ (eq:semi-log-interpretation)

so $\beta$ is a proportional change per unit of $x$ — approximately $100\beta$
percent for small $\beta$. Reporting such a coefficient as an absolute effect is
a common misreading.

## 7. Implementation

```python {tier=A name=feature-engineering}
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
```

## 8. Practical Example

A feature pipeline that is fitted, leak-free, and consistent between training
and serving.

```python {tier=A name=feature-pipeline}
"""A feature pipeline with a single definition used by both paths.

The same function computes features for training and for a single serving
request, which is what eliminates training-serving skew (section 5.5).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

rng = np.random.default_rng(11)


@dataclass
class FeaturePipeline:
    """One definition, fitted on train, applied identically everywhere."""

    smoothing: float = 20.0
    n_folds: int = 5

    global_mean_: float = 0.0
    city_stats_: pd.DataFrame = field(default_factory=pd.DataFrame)
    fitted_: bool = False

    # ---- the single shared definition of the deterministic features -------
    @staticmethod
    def _base_features(df: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)
        f["tenure_days"] = (as_of - df["signup"]).dt.days
        f["recency_days"] = (as_of - df["last_order"]).dt.days
        f["avg_order_value"] = df["total_spend"] / df["n_orders"]
        f["orders_per_month"] = df["n_orders"] / np.maximum(
            f["tenure_days"] / 30.44, 0.1)
        f["expected_gap"] = f["tenure_days"] / df["n_orders"]
        f["recency_vs_own_gap"] = f["recency_days"] / np.maximum(
            f["expected_gap"], 1.0)
        f["log_spend"] = np.log1p(df["total_spend"])
        return f

    def fit(self, df: pd.DataFrame, y: np.ndarray,
            as_of: pd.Timestamp) -> "FeaturePipeline":
        self.global_mean_ = float(y.mean())
        tmp = pd.DataFrame({"city": df["city"].to_numpy(), "y": y})
        agg = tmp.groupby("city")["y"].agg(["count", "mean"])
        # eq. 27.1, computed on the full training set for use at serving time
        agg["enc"] = ((agg["count"] * agg["mean"]
                       + self.smoothing * self.global_mean_)
                      / (agg["count"] + self.smoothing))
        self.city_stats_ = agg
        self.fitted_ = True
        return self

    def transform_train(self, df, y, as_of) -> pd.DataFrame:
        """Training features: the city encoding is computed OUT OF FOLD."""
        if not self.fitted_:
            raise RuntimeError("fit first")
        f = self._base_features(df, as_of)
        fold = rng.integers(0, self.n_folds, len(df))
        enc = np.empty(len(df))
        for k in range(self.n_folds):
            mask = fold == k
            other = pd.DataFrame({"city": df["city"].to_numpy()[~mask],
                                  "y": y[~mask]})
            a = other.groupby("city")["y"].agg(["count", "mean"])
            gm = other["y"].mean()
            a["enc"] = (a["count"] * a["mean"] + self.smoothing * gm) / \
                       (a["count"] + self.smoothing)
            enc[mask] = pd.Series(df["city"].to_numpy()[mask]).map(
                a["enc"]).fillna(gm).to_numpy()
        f["city_target_enc"] = enc
        return f

    def transform_serve(self, df, as_of) -> pd.DataFrame:
        """Serving features: the encoding uses the FULL training statistics.

        This asymmetry is correct — a serving row contributed nothing to those
        statistics, so there is nothing to leak (section 5.2)."""
        if not self.fitted_:
            raise RuntimeError("fit first")
        f = self._base_features(df, as_of)
        f["city_target_enc"] = (df["city"].map(self.city_stats_["enc"])
                                .fillna(self.global_mean_).to_numpy())
        return f


# --- data ---------------------------------------------------------------------
def make(n, seed):
    r = np.random.default_rng(seed)
    as_of = pd.Timestamp("2026-08-13")
    d = pd.DataFrame({
        "signup": as_of - pd.to_timedelta(r.integers(40, 900, n), "D"),
        "last_order": as_of - pd.to_timedelta(r.integers(0, 250, n), "D"),
        "n_orders": r.poisson(7, n) + 1,
        "total_spend": r.gamma(3, 90, n).round(2),
        "city": r.choice([f"city_{i}" for i in range(60)], n),
    })
    tenure = (as_of - d["signup"]).dt.days
    recency = (as_of - d["last_order"]).dt.days
    p = 1 / (1 + np.exp(-(recency / np.maximum(tenure / d["n_orders"], 1) - 1.2)))
    return d, (r.random(n) < p).astype(int), as_of


train_df, y_train, as_of = make(12_000, 1)
serve_df, y_serve, _ = make(4_000, 2)

pipe = FeaturePipeline().fit(train_df, y_train, as_of)
F_train = pipe.transform_train(train_df, y_train, as_of)
F_serve = pipe.transform_serve(serve_df, as_of)

print("feature columns:", list(F_train.columns))
print(f"\ntrain shape {F_train.shape}, serve shape {F_serve.shape}")
print(f"same columns, same order: "
      f"{list(F_train.columns) == list(F_serve.columns)}")

# --- the point: one definition means no skew --------------------------------
print("\n" + "=" * 72)
print("no training-serving skew: the deterministic features are computed by")
print("the SAME function in both paths")
print("=" * 72)
single_row = serve_df.iloc[[0]]
batch_version = pipe.transform_serve(serve_df, as_of).iloc[[0]]
single_version = pipe.transform_serve(single_row, as_of)
matching = np.allclose(batch_version.to_numpy(), single_version.to_numpy())
print(f"batch and single-row serving produce identical features: {matching}")
assert matching

print(f"\n{'feature':<22} {'train mean':>12} {'serve mean':>12} {'ratio':>8}")
for c in F_train.columns:
    tr, se = F_train[c].mean(), F_serve[c].mean()
    print(f"{c:<22} {tr:>12.3f} {se:>12.3f} {se/tr:>8.3f}")
print("\nRatios near 1 indicate the two paths agree. A ratio far from 1 in")
print("production is the signature of training-serving skew, and monitoring")
print("it is cheap (Chapter 48).")

# --- unseen categories are handled explicitly -------------------------------
novel = serve_df.iloc[[0]].copy()
novel["city"] = "city_never_seen"
out = pipe.transform_serve(novel, as_of)
print(f"\nunseen city -> encoding falls back to the global mean "
      f"{out['city_target_enc'].iloc[0]:.4f} "
      f"(global {pipe.global_mean_:.4f})")
print("An explicit, tested fallback — not a silent NaN that fails downstream.")
```

## 9. Common Mistakes

**Target encoding in-fold.** Manufactures signal from noise, as
{{sec:7-implementation}} shows.

**Target encoding without smoothing.** Small categories get pure noise as their
value.

**Selecting features on the full dataset before cross-validating.** The
selection saw the validation folds; scores are optimistic.

**One-hot encoding a high-cardinality column.** Thousands of near-empty columns.

**Integer-encoding nominal categories for a linear or distance-based model.**
Imposes a false order.

**Computing aggregates over the whole dataset including the future.** Temporal
leakage ({{ch:ds-leakage}}).

**Using a feature that will not exist at prediction time.** The most common
production failure.

**Computing features differently in training and serving.** One definition, used
by both.

**Ignoring unseen categories at serving.** Decide, implement and test the
fallback.

**Log-transforming without checking for zeros or negatives.** Use `log1p`, or
Yeo-Johnson.

**Reporting a log-model coefficient as an absolute effect.** It is proportional
({{eq:semi-log-interpretation}}).

## 10. Connection to Previous Chapters

{{ch:ds-eda}} reveals which features are worth building and which relationships
need a transform. {{ch:py-pandas}} supplies `groupby`/`transform`, which is how
every aggregate and group-relative feature is computed. {{ch:ds-cleaning}}
supplies the fit-on-train discipline that {{sec:5-formal-explanation}} extends
to encodings. {{ch:math-probability}} supplies the Bayesian shrinkage that
{{sec:6-mathematical-foundation}} shows target smoothing to be.
{{ch:math-norms}} supplies the $L_1$ penalty behind embedded selection, and
{{ch:math-functions}} the log transform.

Forward: {{ch:ds-leakage}} is the immediate sequel, because the two most
powerful techniques here are also the two most common sources of leakage.
{{ch:ds-timeseries}} covers temporal features properly.

Beyond Part III: {{ch:ml-boosting}} discusses when feature engineering stops
paying off; {{part:6}} onward replaces it with learned representations for
unstructured data; {{ch:mle-pipelines}} formalises the pipeline of
{{sec:8-practical-example}}.

## 11. Exercises

**Beginner**

1. Given `signup_date` and `order_date`, list five features you could build.
2. Which encoding for a column with 4 categories? With 4,000?
3. Why is integer-encoding a nominal category harmful for a linear model?
4. Compute the smoothed encoding for a category with 5 observations and mean
   0.8, global mean 0.3, $m = 20$.
5. When would you use `log1p` rather than `log`?

**Intermediate**

6. Explain out-of-fold target encoding and why prediction time is different.
7. Using {{eq:leak-correlation}}, predict the leaked correlation for categories
   averaging 10 rows, and verify it.
8. Give a feature that is valuable in combination and useless alone. Which
   selection method finds it?
9. Explain training-serving skew with a concrete example and its symptom.
10. Regressing $\log(\text{revenue})$ on `tenure` gives a coefficient of 0.02.
    Interpret it.
11. Why must feature selection happen inside cross-validation?

**Advanced**

12. Derive {{eq:posterior-mean-encoding}} from the Beta-Binomial posterior.
13. Derive {{eq:leak-correlation}} for a category of size $n_c$ with no genuine
    signal.
14. Compare filter, wrapper and embedded selection on cost, interaction
    handling and overfitting risk, and say when each is right.
15. Design a feature-monitoring scheme that would detect training-serving skew
    in production, stating what you compare and how often.

**Implementation**

16. Implement smoothed out-of-fold target encoding as a fit/transform class and
    verify no leakage on a pure-noise target.
17. Build automated aggregate generation over specified group keys, and measure
    how many of the generated features survive an embedded selection.
18. Implement forward selection with cross-validation and compare against $L_1$
    on the same data for cost and result.
19. Demonstrate the selection-leakage effect: select features on the full
    dataset versus inside CV, on data with many noise features, and report both
    scores.

**Reasoning**

20. Agents can generate hundreds of candidate features quickly. Does that make
    feature engineering easier or more dangerous? Be specific.
21. Deep learning replaced feature engineering for text and images and not for
    tabular data. Why?

## 12. Chapter Summary

A feature encodes knowledge the model cannot derive. For tabular data this is
frequently worth more than model choice, and the most valuable features compare
an entity against its own baseline rather than against the population.

Four families cover most of it: aggregates, ratios and differences, temporal
features, and interactions. Temporal features are the highest-value and the most
likely to leak.

Encoding follows cardinality. One-hot for a handful of levels; target encoding
for high cardinality. Target encoding must be computed out-of-fold during
training and smoothed toward the global mean, and the smoothing formula is
exactly a Bayesian posterior mean with the smoothing parameter as a
pseudo-count. In-fold target encoding manufactures signal from pure noise, with
a correlation of roughly $1/\sqrt{n_c}$ for a category of size $n_c$.

The asymmetry between training and serving is deliberate: training rows must be
encoded without seeing their own target, while serving rows may use the full
training statistics because they contributed nothing to them.

Log transforms linearise multiplicative relationships, which is why they so
often straighten a curved scatter plot, and they change a coefficient's meaning
from absolute to proportional.

Feature selection comes in filter, wrapper and embedded forms. Embedded is the
usual default, and all of them must run inside cross-validation, because
selection is a fitted step and selecting on the full dataset produces optimistic
scores.

Training-serving skew — the same feature computed differently in the two paths —
is a common production failure, and the fix is a single shared definition rather
than any particular infrastructure.

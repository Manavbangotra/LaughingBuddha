---
id: ds-leakage
number: 28
part: III
tier: focused
status: reviewed
requires: [ds-feature-eng, ds-experiments]
provides: [data-leakage-term, target-leakage, temporal-leakage, group-leakage,
           training-serving-skew, class-imbalance, resampling, class-weighting]
citations: [kaufman2012, chawla2002]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Define leakage precisely and explain why it produces excellent validation
   scores and poor production performance.
2. Identify the four mechanisms by which leakage enters.
3. Detect leakage in an unfamiliar pipeline.
4. Choose a validation scheme that matches how the model will be used.
5. Explain why accuracy is meaningless under class imbalance.
6. Choose between resampling, class weighting and threshold tuning.
7. Explain why resampling must happen inside the training fold only.

## 2. Why This Matters

Leakage is the standard reason a model that looked excellent in development
fails in production, and {{cite:kaufman2012}} is the reference treatment.

The pattern is always the same. Validation accuracy is 0.97. Everyone is
pleased. The model ships and performs at baseline. Nothing errors, no code was
wrong, and the discrepancy is often attributed to "distribution shift" when the
real cause is that the validation score was never an estimate of production
performance in the first place.

The reason it is so common is that every mechanism looks reasonable at the time.
Scaling before splitting is tidy. Including every available column is thorough.
Shuffling before splitting is standard practice. Resampling to fix imbalance is
recommended everywhere. Each of these leaks, and each is what a competent person
would do without a specific reason not to.

The second half of the chapter covers class imbalance, which belongs here for
two reasons: the standard remedies leak if applied in the wrong place, and
imbalance is the situation where a misleading metric most resembles a leak — a
model with 99.9% accuracy that has learned nothing.

## 3. Prerequisites

{{ch:ds-feature-eng}} for target encoding and aggregates, the two most common
leak vectors; {{ch:ds-experiments}} for the randomisation-unit argument, which
reappears here as group leakage; {{ch:math-probability}} for the base-rate
reasoning behind imbalance.

## 4. Intuitive Explanation

### 4.1 What leakage is

{{term:data-leakage-term}} is information available to the model during training
that will **not** be available when it makes a real prediction.

The test is a question about time and availability, not about statistics:

> At the moment this prediction must be made, in production, would this value
> actually be known?

If not, the feature leaks. The model learns to use it, validation rewards it,
and production cannot supply it — or supplies a different value.

```text
      training                         production
  ┌──────────────────┐            ┌──────────────────┐
  │ everything in    │            │ only what exists │
  │ the dataset      │   ──────▶  │ at prediction    │
  │ (including the   │            │ time             │
  │  future)         │            │                  │
  └──────────────────┘            └──────────────────┘
        ▲ leakage lives in the gap between these two
```

### 4.2 The four mechanisms

**{{term:target-leakage}}** — a feature that is a consequence of the outcome.
`cancellation_reason` when predicting cancellation. `discharge_date` when
predicting admission length. `days_to_payment` when predicting default. Each is
populated only after the thing you are predicting has happened.

**{{term:temporal-leakage}}** — training on data from after the prediction
point. Caused automatically by shuffling time-ordered data before splitting, and
by any aggregate computed over the whole history.

**{{term:group-leakage}}** — rows from the same entity in both train and test.
Multiple visits by one patient, multiple sessions by one user, multiple sentences
from one document. The model memorises the entity rather than learning the
pattern.

**Preprocessing leakage** — a fitted transformation computed over all the data.
Scalers, imputers, encoders and feature selection all learn parameters, and
learning them from the full dataset means the test set influenced training.

### 4.3 Why it is invisible

Leakage does not raise errors, and it produces exactly the outcome everyone
wants — a high validation score. Three properties make it hard to catch:

**It is rewarded.** Every incentive points toward accepting a good number.

**It hides in reasonable steps.** No individual line looks wrong.

**Its symptom resembles other problems.** A model that performs worse in
production is attributed to drift, to a distribution difference, or to
implementation error — all of which also occur, which is what makes the
misattribution plausible.

> IMPORTANT: The strongest signal is a validation score that is *too good*. A
> churn model at 0.99 AUC is not a triumph; it is a leak until proven otherwise.
> Treat implausibly good results as bugs and investigate them with the same
> urgency as failures.

### 4.4 Class imbalance

When one class is rare, accuracy becomes useless. A fraud detector predicting
"not fraud" always scores 99.9% on a 0.1% fraud rate.

This is the base-rate problem of {{ch:math-probability}}, and the same
arithmetic applies: precision — the probability of genuine fraud given a fraud
prediction — depends on the base rate, not only on the model's quality.

## 5. Formal Explanation

### 5.1 A formal definition

For instance $i$ with prediction time $t_i$, a feature $x_{ij}$ leaks if its
value depends on information generated after $t_i$, or on the target $y_i$:

$$
x_{ij} = f(\text{data available at } t_i) \quad\text{— required}
$$ (eq:leakage-condition)

The formulation makes clear why leakage is not detectable from the data alone in
general: whether a value was available at $t_i$ is a fact about the systems that
produced it, not about the column.

### 5.2 Validation schemes

The validation scheme must mirror deployment. The question is always: *what does
the model see in production that it has not seen before?*

{#tbl:validation-schemes caption="Validation schemes and the leak each prevents. The right choice follows from how the model will be used."}

| Scheme | Prevents | Use when |
|---|---|---|
| Random k-fold | nothing structural | rows are genuinely independent |
| Stratified k-fold | class-imbalanced folds | classification with rare classes |
| Grouped k-fold | group leakage | repeated entities |
| Time-series split | temporal leakage | ordered data |
| Grouped + time split | both | the common real case |
| Nested CV | selection leakage | tuning and evaluating together |

**Nested cross-validation** deserves emphasis. If hyperparameters are tuned on
the same folds used to report performance, the reported score is optimistic by
exactly the selection mechanism of {{ch:math-inference}}. Nested CV uses an
inner loop for tuning and an outer loop for estimation.

### 5.3 Detecting leakage

Five checks, in the order of cost:

1. **Is the score implausible?** Compare against a domain-plausible ceiling and
   a trivial baseline.
2. **Which features dominate?** One feature carrying almost all the importance
   is a leak signature.
3. **Does a single feature nearly predict the target alone?** Fit a
   one-feature model per column and look for anomalies.
4. **Check availability timestamps.** For each feature, when is it populated
   relative to the prediction point?
5. **Shuffle the target.** With a randomised target, performance must fall to
   chance. If it does not, the pipeline itself leaks — a definitive test.

The last is the most powerful and the least used. It catches preprocessing and
selection leakage that no feature-level inspection would find.

### 5.4 Class imbalance: metrics

Under imbalance, accuracy is dominated by the majority class. Use instead:

$$
\text{precision} = \frac{TP}{TP + FP},
\qquad
\text{recall} = \frac{TP}{TP + FN}
$$ (eq:precision-recall)

Precision is $\Prob(\text{positive} \mid \text{predicted positive})$ — exactly
the base-rate-dependent quantity of {{ch:math-probability}}. Recall is
$\Prob(\text{predicted positive} \mid \text{positive})$, which is not.

**Precision depends on prevalence and recall does not.** A model with fixed
sensitivity and specificity has a precision that changes entirely with the base
rate, which is why a detector that worked in one market fails in another with a
lower incidence, without the model having changed.

For imbalanced problems, the precision-recall curve is more informative than the
ROC curve, because ROC's false-positive rate has the large majority class in its
denominator and therefore moves very little.

### 5.5 Class imbalance: remedies

{#tbl:imbalance-remedies caption="Remedies for class imbalance. Threshold tuning is the simplest and is frequently sufficient."}

| Remedy | Mechanism | Caution |
|---|---|---|
| Threshold tuning | move the decision boundary | **try this first** |
| {{term:class-weighting}} | weight the loss by inverse frequency | simple, no data change |
| Random oversampling | duplicate minority rows | overfits duplicates |
| Random undersampling | discard majority rows | discards information |
| SMOTE {{cite:chawla2002}} | synthesise between minority neighbours | interpolates into empty regions |
| Collect more minority data | fixes the actual problem | usually impossible |

> WARNING: Any resampling must occur **inside the training fold only**.
> Oversampling before splitting places copies of the same minority row in both
> train and test, so the model is evaluated on rows it memorised. This is
> group leakage with the group being a single duplicated observation, and it
> inflates validation scores dramatically. {{sec:7-implementation}} measures it.

Threshold tuning is under-used. A well-calibrated model trained on imbalanced
data with no resampling at all, evaluated at a threshold chosen for the
operating point you care about, frequently beats every resampling scheme —
and it changes no data, so it cannot leak.

## 6. Mathematical Foundation

### 6.1 Why precision depends on the base rate

From Bayes' theorem ({{ch:math-probability}}), with sensitivity $s$,
false-positive rate $f$, and prevalence $\pi$:

$$
\text{precision} = \frac{s\pi}{s\pi + f(1-\pi)}
$$ (eq:precision-from-bayes)

Neither $s$ nor $f$ involves $\pi$; they are properties of the model. Precision
does, and strongly.

For a model with $s = 0.90$ and $f = 0.05$:

{#tbl:precision-by-prevalence caption="Precision at fixed model quality across prevalences. The model is identical in every row."}

| Prevalence | Precision |
|---|---|
| 50% | 94.7% |
| 10% | 66.7% |
| 1% | 15.4% |
| 0.1% | 1.8% |

The same model is excellent at 50% prevalence and nearly useless at 0.1%. This
is the base-rate fallacy of {{ch:math-probability}} in its operational form, and
it explains why a fraud model validated on a balanced sample disappoints on live
traffic.

> IMPORTANT: This has a direct consequence for resampling. If you balance the
> training data, the model's implied prior is 50% rather than the true
> prevalence, so its output probabilities are systematically too high. They must
> be recalibrated before being used as probabilities — and if you only need a
> ranking, resampling bought nothing that threshold tuning would not.

### 6.2 How much resampling before splitting inflates a score

Suppose the minority class has $n_{\min}$ rows and is oversampled by a factor
$k$ before an 80/20 split. Each original minority row now has $k$ copies
distributed at random across train and test.

The probability that a given original row has at least one copy in train and at
least one in test is

$$
\Prob(\text{split across}) = 1 - 0.8^{k} - 0.2^{k}
$$ (eq:duplicate-split)

For $k = 5$: $1 - 0.328 - 0.0003 = 0.672$. Two thirds of minority rows appear on
both sides, and the model can memorise them.

The resulting inflation is largest exactly where imbalance is most severe,
because that is where $k$ must be largest — which means the technique fails
worst in the situation it was adopted for.

### 6.3 The shuffled-target test

Under a randomly permuted target, any honest pipeline must achieve chance
performance. Formally, if $\tilde{y} = \sigma(y)$ for a random permutation
$\sigma$, then $\tilde{y}$ is independent of $\mat{X}$, so

$$
\E[\text{AUC}] = 0.5
$$ (eq:shuffled-auc)

for any procedure that only uses $\mat{X}$ to predict $\tilde{y}$.

Performance above chance therefore proves that the *procedure* is using target
information illegitimately — through preprocessing fitted on all data, feature
selection that saw the labels, or resampling before splitting. It cannot be
explained by a leaking feature, since no feature relates to a shuffled target.

This makes it a strict test of the pipeline rather than of the data, and it is
the only check in {{sec:5-formal-explanation}} with that property.

## 7. Implementation

```python {tier=A name=leakage-detection}
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
```

## 8. Practical Example

Class imbalance, with the remedies compared honestly — including the one that
leaks.

```python {tier=A name=imbalance-remedies}
"""Class imbalance: why accuracy lies, and which remedy actually helps.
"""
import numpy as np

rng = np.random.default_rng(3)


def auc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos, neg = labels.sum(), (1 - labels).sum()
    return (ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)


def fit_logistic(X, y, weights=None, steps=600, lr=0.5):
    X = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(X.shape[1])
    sw = np.ones(len(y)) if weights is None else weights
    sw = sw / sw.mean()
    for _ in range(steps):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        w -= lr * (X.T @ (sw * (p - y)) / len(y))
    return w


def predict(X, w):
    X = np.column_stack([np.ones(len(X)), X])
    return 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))


def metrics(scores, y, thresh):
    pred = (scores >= thresh).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"accuracy": (tp + tn) / len(y), "precision": prec,
            "recall": rec, "f1": f1}


# --- eq. 28.4: precision collapses with prevalence, model unchanged ---------
print("=" * 72)
print("the same model at different prevalences (eq. 28.4)")
print("=" * 72)
s, f = 0.90, 0.05
print(f"sensitivity {s}, false-positive rate {f} — FIXED\n")
print(f"{'prevalence':>12} {'precision':>11} {'accuracy':>10}")
for pi in (0.5, 0.1, 0.01, 0.001):
    prec = s * pi / (s * pi + f * (1 - pi))
    acc = s * pi + (1 - f) * (1 - pi)
    print(f"{pi:>12.1%} {prec:>11.1%} {acc:>10.1%}")
print("\nAccuracy RISES as the problem gets harder, because the majority")
print("baseline improves. Precision collapses. Only one of these is useful.")

# --- a realistically imbalanced problem --------------------------------------
n = 30_000
x1 = rng.normal(0, 1, n)
x2 = rng.normal(0, 1, n)
logit = -5.2 + 1.4 * x1 + 0.9 * x2
y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
X = np.column_stack([x1, x2])

split = int(0.7 * n)
Xtr, ytr, Xte, yte = X[:split], y[:split], X[split:], y[split:]
print(f"\npositive rate: {y.mean():.2%}  "
      f"({y.sum():,} positives in {n:,} rows)")

print("\n" + "=" * 72)
print("remedies compared")
print("=" * 72)
results = {}

# baseline
w = fit_logistic(Xtr, ytr)
results["none (0.5 threshold)"] = (predict(Xte, w), 0.5)

# threshold tuned on TRAIN to maximise F1 — no data change, cannot leak
tr_scores = predict(Xtr, w)
grid = np.quantile(tr_scores, np.linspace(0.5, 0.9999, 300))
best_t = max(grid, key=lambda t: metrics(tr_scores, ytr, t)["f1"])
results["threshold tuned"] = (predict(Xte, w), best_t)

# class weighting
weights = np.where(ytr == 1, (ytr == 0).sum() / max((ytr == 1).sum(), 1), 1.0)
w_cw = fit_logistic(Xtr, ytr, weights=weights)
results["class weighting"] = (predict(Xte, w_cw), 0.5)

# oversampling INSIDE the training fold — correct
pos_idx = np.where(ytr == 1)[0]
extra = rng.choice(pos_idx, size=(ytr == 0).sum() - len(pos_idx), replace=True)
idx_bal = np.concatenate([np.arange(len(ytr)), extra])
w_os = fit_logistic(Xtr[idx_bal], ytr[idx_bal])
results["oversample (in-fold)"] = (predict(Xte, w_os), 0.5)

print(f"{'remedy':<24} {'accuracy':>10} {'precision':>11} {'recall':>9} "
      f"{'F1':>8} {'AUC':>8}")
for name, (scores, t) in results.items():
    m = metrics(scores, yte, t)
    print(f"{name:<24} {m['accuracy']:>10.4f} {m['precision']:>11.3f} "
          f"{m['recall']:>9.3f} {m['f1']:>8.3f} {auc(scores, yte):>8.4f}")

print(f"\n{'always predict 0':<24} {1-yte.mean():>10.4f} "
      f"{0.0:>11.3f} {0.0:>9.3f} {0.0:>8.3f} {0.5:>8.4f}")
print("\nThe do-nothing baseline has the highest accuracy and is useless.")
print("Note the AUC barely moves across remedies — they change the operating")
print("point, not the ranking. Threshold tuning gets most of the benefit")
print("without touching the data.")

# --- eq. 28.3: oversampling BEFORE the split leaks --------------------------
print("\n" + "=" * 72)
print("oversampling before the split (eq. 28.3)")
print("=" * 72)


def knn_scores(Xtr, ytr, Xte, k=5):
    """A memorising model. With a linear model this leak is invisible, because
    duplicated rows only reweight the loss — they cannot be looked up."""
    out = np.empty(len(Xte))
    for start in range(0, len(Xte), 2000):
        block = Xte[start:start + 2000]
        d = ((block[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)
        nn = np.argpartition(d, k, axis=1)[:, :k]
        out[start:start + 2000] = ytr[nn].mean(axis=1)
    return out


# Keep k modest so the demonstration is quick and the arithmetic is clear.
sub = rng.choice(n, 6000, replace=False)
Xs, ys = X[sub], y[sub]
pos_all = np.where(ys == 1)[0]
k_factor = 6
extra_all = rng.choice(pos_all, size=len(pos_all) * (k_factor - 1), replace=True)
idx_all = np.concatenate([np.arange(len(ys)), extra_all])
rng.shuffle(idx_all)

cut = int(0.7 * len(idx_all))
tr_idx, te_idx = idx_all[:cut], idx_all[cut:]
overlap = len(set(tr_idx.tolist()) & set(te_idx.tolist())) / len(set(te_idx.tolist()))

leaky = knn_scores(Xs[tr_idx], ys[tr_idx], Xs[te_idx])
honest = knn_scores(Xs[tr_idx], ys[tr_idx], Xte)

print(f"oversampling factor for the minority class : {k_factor}")
print(f"eq. 28.3 predicts {1 - 0.7**k_factor - 0.3**k_factor:.1%} of duplicated "
      f"rows appear on both sides")
print(f"measured overlap of distinct rows          : {overlap:.1%}")
print(f"\n{'evaluation':<34} {'AUC':>8}")
print(f"{'on the OVERSAMPLED test split':<34} {auc(leaky, ys[te_idx]):>8.4f}"
      f"   <- inflated")
print(f"{'on the untouched real test set':<34} {auc(honest, yte):>8.4f}"
      f"   <- the truth")
print(f"\noptimism: {auc(leaky, ys[te_idx]) - auc(honest, yte):+.4f}")
print("\nThe same model, evaluated two ways. Resampling before the split")
print("evaluates the model on rows it has literally already seen.")
print("\nNote this leak is invisible to a linear model, which cannot memorise")
print("a row — it appears with k-NN, trees and boosting, which can.")
```

## 9. Common Mistakes

**Including a feature populated after the outcome.** Ask when each field is
written, not whether it correlates.

**Shuffling time-ordered data before splitting.** Use a temporal split.

**Splitting rows when entities repeat.** Split by entity.

**Fitting a scaler, imputer or encoder before splitting.** Fit inside the fold.

**Selecting features on the full dataset.** Selection is a fitted step, and
{{sec:7-implementation}} shows it fabricating an AUC of 0.68 from pure noise.

**Tuning and reporting on the same folds.** Use nested cross-validation.

**Resampling before splitting.** Evaluates on memorised rows — a 0.27 AUC
inflation in {{sec:8-practical-example}}.

**Reporting accuracy on imbalanced data.** The trivial baseline usually wins.

**Trusting a suspiciously high score.** Investigate good results as hard as bad
ones.

**Balancing the training set and reading the outputs as probabilities.** The
implied prior is now 50%; recalibrate, or use ranks only.

**Reaching for SMOTE before trying a threshold.** Threshold tuning is simpler,
cheaper and cannot leak.

**Assuming a linear model exposes these leaks.** Group leakage and
resampling leakage require a model with the capacity to memorise. They are
invisible with logistic regression and severe with k-NN, trees and boosting —
which is what you will actually deploy.

## 10. Connection to Previous Chapters

{{ch:ds-feature-eng}} supplies the two most common leak vectors — target
encoding and full-dataset aggregates — and the fit/transform discipline that
prevents preprocessing leakage. {{ch:ds-experiments}} supplies the
randomisation-unit argument, which is group leakage in an experimental setting:
the same requirement that the unit of assignment be the unit of analysis.
{{ch:math-probability}} supplies the base-rate reasoning behind
{{eq:precision-from-bayes}}, and {{ch:math-inference}} the selection effect that
makes tuning on the reported folds optimistic. {{ch:ds-cleaning}} established
fitting on training data only, which is the general form of the preprocessing
rule here. {{ch:ds-what-it-is}}'s audit found two of these mechanisms in a real
table before any model existed.

Forward: {{ch:ds-timeseries}} develops temporal validation in full, including
the gap that autocorrelated features require.

Beyond Part III: {{ch:mle-splits}} formalises validation design;
{{ch:ml-metrics}} treats precision, recall and calibration properly;
{{ch:ev-framework}} builds a harness that runs these checks by default rather
than on request. {{cite:kaufman2012}} is the reference treatment of leakage;
{{cite:chawla2002}} introduced SMOTE.

## 11. Exercises

**Beginner**

1. Define leakage in terms of availability at prediction time.
2. Name the four mechanisms and give an example of each.
3. Why is accuracy useless at a 0.1% positive rate?
4. Which validation scheme for data with repeated patients?
5. What is the first thing to suspect about a 0.99 AUC churn model?

**Intermediate**

6. Using {{eq:precision-from-bayes}}, compute precision at 2% prevalence with
   sensitivity 0.85 and false-positive rate 0.03.
7. Using {{eq:duplicate-split}}, compute the fraction of minority rows appearing
   on both sides when oversampling by 8 with a 75/25 split.
8. Explain why the shuffled-target test catches pipeline leakage that
   feature-level inspection cannot.
9. A model uses `account_closed_date` to predict churn. Which mechanism, and how
   would you find it?
10. Why must feature selection sit inside cross-validation?
11. Explain why balancing the training data miscalibrates the output
    probabilities.
12. Why did the group-leakage demonstration in {{sec:7-implementation}} require
    a nearest-neighbour model rather than logistic regression?

**Advanced**

13. Derive {{eq:precision-from-bayes}} from Bayes' theorem.
14. Derive {{eq:duplicate-split}} and generalise it to an arbitrary train
    fraction.
15. Explain nested cross-validation and quantify the optimism of the non-nested
    version.
16. Design a leakage-detection suite that runs automatically in CI, stating what
    each check can and cannot catch.
17. SMOTE interpolates between minority neighbours. Describe a feature geometry
    in which this creates synthetic points in a region no real example occupies,
    and say what that does to the decision boundary.

**Implementation**

18. Implement grouped time-series cross-validation and demonstrate it prevents
    both leaks simultaneously.
19. Implement the shuffled-target test as a reusable function and run it against
    a pipeline of your own.
20. Reproduce {{sec:8-practical-example}} with SMOTE instead of random
    oversampling and compare the inflation.
21. Build a feature-availability audit: given a table with a populated-at
    timestamp per field, report which fields are written after the prediction
    point.

**Reasoning**

22. Leakage produces exactly the result everyone wants. What organisational
    practice would make it more likely to be caught, and what would it cost?
23. Threshold tuning frequently matches resampling. Why do tutorials recommend
    resampling first?

## 12. Chapter Summary

Leakage is information available during training that will not be available at
prediction time. It produces excellent validation scores and production
performance at baseline, and the gap is routinely misattributed to drift.

Four mechanisms: target leakage from features that are consequences of the
outcome; temporal leakage from training on the future; group leakage from
entities appearing on both sides of a split; and preprocessing leakage from any
transformation fitted before splitting.

It is hard to catch because every step looks reasonable, nothing errors, and the
result is rewarded. The strongest signal is a score that is too good — treat
implausible results as bugs and investigate them with the urgency given to
failures.

The validation scheme must mirror deployment: grouped splits for repeated
entities, temporal splits for ordered data, both together in the common case,
and nested cross-validation when tuning and reporting come from the same data.

The shuffled-target test is the strictest check available. Under a permuted
target any honest pipeline must score at chance, so above-chance performance
proves the *procedure* leaks regardless of any individual feature — which is why
it catches preprocessing and selection leakage that feature inspection cannot.

Two of these leaks require a model with the capacity to memorise. Group leakage
and resampling-before-splitting are invisible to a linear model and severe with
nearest neighbours, trees and boosting, so a pipeline validated with logistic
regression can hide a leak that appears the moment the real model is fitted.

Under class imbalance, accuracy is dominated by the majority class. Precision
depends on prevalence while sensitivity and specificity do not, so the same
model can be excellent at 50% prevalence and useless at 0.1% — which is why a
detector validated on a balanced sample disappoints on live traffic.

Resampling must occur inside the training fold. Oversampling before the split
places copies of the same row on both sides and evaluates the model on data it
memorised. Threshold tuning is simpler, changes no data, cannot leak, and
frequently matches or beats every resampling scheme.

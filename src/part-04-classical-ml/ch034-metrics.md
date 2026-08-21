---
id: ml-metrics
number: 34
part: IV
tier: focused
status: reviewed
requires: [ml-logistic, ml-linear-regression, ds-leakage, math-inference]
provides: [bias-variance-decomposition, confusion-matrix, precision-recall,
           roc-auc, pr-auc, calibration-metrics, learning-curve,
           cross-validation-model-selection, regression-metrics]
citations: [pedregosa2011, breiman2001cultures]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Derive the bias-variance-noise decomposition and state what each term is.
2. Diagnose underfitting and overfitting from a learning curve.
3. Choose a classification metric from the cost structure of the problem.
4. Explain when ROC-AUC misleads and why PR-AUC is preferred under imbalance.
5. Measure calibration and explain why it is independent of discrimination.
6. Choose a regression metric and say what each one is robust to.
7. Design a model-selection procedure that does not leak.
8. Explain why the test set may be used exactly once.

## 2. Why This Matters

This is the hinge of Part IV, and the chapter that makes the rest of it legible.

**The bias-variance decomposition is the organising frame.** Every algorithm
after this chapter is an answer to one question: how do you get a model complex
enough to capture the pattern without capturing the noise? Bagging
({{ch:ml-forests}}) attacks variance. Boosting ({{ch:ml-boosting}}) attacks bias.
Regularisation trades one for the other. Pruning ({{ch:ml-trees}}) picks a point
on the curve. Without the decomposition these are nine unrelated tricks; with it
they are four answers to one question.

**Choosing a metric is choosing what the model optimises for.** A metric is not
a report. It selects the hyperparameters, decides which model ships, and
determines what the system does in the cases that matter. Optimising accuracy on
a 1% positive rate produces a model that predicts "no" forever and scores 99%.
The metric is a specification of what you want, and a wrong one is a wrong
specification.

**Almost every model-selection procedure leaks.** {{ch:ds-leakage}} covered
leakage in features; this chapter covers leakage through *repeated evaluation*,
which is subtler and at least as common. A test set consulted a hundred times
is a training set.

## 3. Prerequisites

{{ch:ml-logistic}} for probabilistic predictions and thresholds.
{{ch:ml-linear-regression}} for the regularisation path.
{{ch:ds-leakage}} for validation design. {{ch:math-inference}} for sampling
variability, standard errors, and the multiple-comparisons problem that
model selection is a disguised instance of.

## 4. Intuitive Explanation

### 4.1 Three sources of error

Imagine refitting your model on many different samples from the same population
and watching one particular prediction.

**Bias** — the average prediction is wrong. The model is too rigid to represent
the truth; fitting a line to a curve gives the same wrong answer on every
sample.

**Variance** — the predictions scatter. The model is flexible enough to chase
the noise, so each sample yields a different answer, and their average may be
fine while any individual one is not.

**Noise** — the target is not a deterministic function of the features. No model
can remove this, and any model that appears to has memorised the sample.

```text
       low variance          high variance
      ┌─────────────┐      ┌─────────────┐
 low  │    ◎ ● ●    │      │  ●   ◎   ●  │
 bias │     ●●●     │      │   ●     ●   │      ◎ = truth
      │             │      │      ●      │      ● = prediction
      └─────────────┘      └─────────────┘        from one sample
      ┌─────────────┐      ┌─────────────┐
 high │  ◎          │      │ ◎    ●      │
 bias │      ●●●    │      │     ●   ●   │
      │       ●●    │      │   ●    ●    │
      └─────────────┘      └─────────────┘
       underfitting          overfitting
```

The trade is that model complexity moves bias and variance in opposite
directions, so the total has a minimum somewhere in between. Finding it is what
model selection is.

> NOTE: "Trade-off" is not quite right for modern deep networks, which sit far
> to the right of the classical minimum and get *better* rather than worse — the
> double-descent phenomenon touched on in {{sec:5-formal-explanation}}. For every
> model in Part IV the classical picture holds and the U-shape is real.

### 4.2 The learning curve tells you which one you have

Plot training and validation error against training-set size.

```text
 error │╲                          error │╲
       │ ╲___________ validation         │ ╲_________ validation
       │ ╱‾‾‾‾‾‾‾‾‾‾‾ training           │            ← large gap
       │╱                                │  __________ training
       └─────────────── N                └─────────────── N
        HIGH BIAS: curves meet            HIGH VARIANCE: gap persists
        at a high error. More data        More data narrows the gap.
        will not help. Add capacity.      Add data or regularise.
```

This single diagnostic answers the question people most often get wrong: *would
more data help?* If the curves have converged, no — the model is at its
representational ceiling, and collecting another million rows is wasted money.
If a gap persists, yes.

### 4.3 Metrics answer different questions

For a classifier there is no single number. A confusion matrix has four cells
and every metric is a different summary of them, discarding different
information.

{#tbl:metric-choice caption="Which classification metric answers which question. Choose from the cost structure, not from convention."}

| Question | Metric |
|---|---|
| Of those I flagged, how many were right? | precision |
| Of those that exist, how many did I find? | recall |
| One number balancing both? | $F_1$ |
| How well do I rank, at any threshold? | ROC-AUC |
| …when positives are rare? | PR-AUC |
| Do my probabilities mean anything? | calibration, Brier score |
| What will this cost? | expected cost |

The last is usually the honest one, and the one nobody computes.

## 5. Formal Explanation

### 5.1 The decomposition

For squared error at a fixed input $\vec{x}_0$, with $y = f(\vec{x}_0) +
\epsilon$, $\E[\epsilon] = 0$, $\Var[\epsilon] = \sigma^{2}$, and $\hat{f}$
fitted on a random training set:

$$
\E\big[(y - \hat{f}(\vec{x}_0))^{2}\big]
 = \underbrace{\big(\E[\hat{f}(\vec{x}_0)] - f(\vec{x}_0)\big)^{2}}_{\text{bias}^{2}}
 + \underbrace{\Var\big[\hat{f}(\vec{x}_0)\big]}_{\text{variance}}
 + \underbrace{\sigma^{2}}_{\text{noise}}
$$ (eq:bias-variance)

The expectation is over training sets, not over test points — which is why the
decomposition cannot be computed from a single fit and must be estimated by
resampling ({{sec:7-implementation}}).

The noise term is irreducible. It is the ceiling on any model's performance, and
a validation score below it means leakage rather than skill.

> IMPORTANT: {{eq:bias-variance}} is exact for squared error and does *not*
> decompose so cleanly for 0-1 loss or cross-entropy. Analogous decompositions
> exist but bias and variance can interact rather than add — with 0-1 loss,
> variance can even *reduce* error when the average prediction is on the wrong
> side of the boundary. The intuition transfers; the algebra does not.

### 5.2 Classification metrics

From the confusion matrix:

$$
\text{precision} = \frac{TP}{TP+FP}, \qquad
\text{recall} = \frac{TP}{TP+FN}, \qquad
F_1 = \frac{2 \cdot \text{prec} \cdot \text{rec}}{\text{prec}+\text{rec}}
$$ (eq:precision-recall)

Precision and recall trade against each other as the threshold moves; reporting
one without the other is meaningless, since either can be driven to 1 by an
absurd threshold.

**ROC-AUC** is the area under true-positive rate against false-positive rate. It
has an exact probabilistic meaning: the probability that a randomly chosen
positive is scored above a randomly chosen negative. It is threshold-free and
**invariant to class balance** — which is its strength for comparing rankers and
its weakness for imbalanced problems.

$$
\text{FPR} = \frac{FP}{FP+TN}
$$ (eq:fpr)

The denominator is the number of negatives, which under imbalance is enormous.
Ten thousand false positives against a million negatives moves FPR by 0.01 —
invisible on an ROC curve — while destroying precision if there are only a
thousand positives. **PR-AUC** uses precision instead of FPR, so those same ten
thousand false positives are fully visible. Under heavy imbalance, report
PR-AUC.

The PR-AUC baseline for a random classifier is the positive rate, not 0.5, so a
PR-AUC of 0.4 is excellent at a 1% base rate and terrible at a 45% one. Always
report it against its baseline.

### 5.3 Calibration is not discrimination

A model can rank perfectly and have meaningless probabilities: multiply every
predicted probability by 0.5 and ROC-AUC is unchanged while every probability is
wrong. The two properties are independent and both matter.

**Brier score** is mean squared error on probabilities:

$$
\text{BS} = \frac{1}{N}\sum_i (p_i - y_i)^{2}
$$ (eq:brier)

It is a **proper scoring rule** — uniquely minimised by reporting your true
belief — and so is log loss {{eq:cross-entropy}}. Accuracy is not: it can be
improved by lying about your confidence. This is exactly why models are trained
on log loss and reported on accuracy, and why the two sometimes disagree about
which model is better.

**Expected calibration error** bins predictions and averages the gap:

$$
\text{ECE} = \sum_{b=1}^{B}\frac{n_b}{N}
   \big|\,\overline{y}_b - \overline{p}_b\,\big|
$$ (eq:ece)

ECE is sensitive to the binning and has no natural noise floor, so it should be
read alongside a reliability diagram and the standard errors of
{{ch:math-inference}} — the calibration table in {{ch:ml-logistic}} showed a bin
that looked broken and was pure sampling noise.

### 5.4 Regression metrics

$$
\text{RMSE} = \sqrt{\tfrac{1}{N}\textstyle\sum(y_i-\hat{y}_i)^{2}}, \quad
\text{MAE} = \tfrac{1}{N}\textstyle\sum|y_i-\hat{y}_i|, \quad
R^{2} = 1 - \frac{\sum(y_i-\hat{y}_i)^{2}}{\sum(y_i-\bar{y})^{2}}
$$ (eq:regression-metrics)

RMSE penalises large errors quadratically and is minimised by the conditional
*mean*; MAE penalises linearly and is minimised by the conditional *median*.
That is the substantive difference: they ask for different predictions, so
switching metric changes the optimal model rather than merely rescoring it.
Choose RMSE when large errors are disproportionately costly and MAE when they
are not — and note that MAE's robustness to outliers is the same robustness that
makes it ignore the tail you may care about.

$R^{2}$ is scale-free and therefore comparable across problems, but only against
the mean baseline and only on the data it was computed on. It is negative for a
model worse than predicting the mean, which is possible on a test set and always
worth reporting rather than hiding.

**MAPE** is popular and dangerous: undefined at zero, unbounded for small
actuals, and asymmetric — it penalises over-prediction more than
under-prediction, so optimising it biases forecasts low. Where a percentage error
is genuinely wanted, prefer a log-scale error or a symmetric variant, and state
which.

### 5.5 Model selection without leaking

Selecting a model on the validation set makes the validation score optimistic
about the *selected* model — the selection effect of {{ch:math-inference}}, and
the reason the split is three-way:

```text
   train ──▶ fit parameters
   validation ──▶ choose hyperparameters, compare models   (used many times)
   test ──▶ estimate performance of the final choice       (used ONCE)
```

**Nested cross-validation** is the standard remedy when data is too scarce for
a held-out test set: an inner loop selects hyperparameters and an outer loop
scores the whole selection procedure, so no score is ever computed on data that
participated in choosing the model. It costs $k_{\text{outer}} \times
k_{\text{inner}}$ times as many fits as a single search.

It is often described as unbiased. It is not, quite, and the direction is worth
knowing: each outer fold selects and fits on $(k-1)/k$ of the data, so nested CV
estimates the procedure applied to a *smaller* dataset than the one you will
train on, and is therefore mildly **pessimistic** — as measured in
{{sec:8-practical-example}}, where it errs low while the naive score errs high.
Conservative is usually the right direction to be wrong in, but reporting it as
exact is not.

The deeper point is that every honest estimate costs something: a held-out test
set costs data, nested CV costs compute and a conservative bias, and a naive CV
score costs you the truth.

> WARNING: If you evaluate 100 models on the same validation set, the best
> score is a maximum of 100 noisy draws and is biased upward by roughly the
> standard error times $\sqrt{2\log 100} \approx 3$. With a validation standard
> error of 1% that is a 3% illusion — larger than most reported improvements.
> The test set exists to absorb this, and it can only do so if it is untouched.

## 6. Mathematical Foundation

### 6.1 Deriving the decomposition

Write $\hat{f} = \hat{f}(\vec{x}_0)$, $f = f(\vec{x}_0)$, $\bar{f} = \E[\hat{f}]$.
Since $\epsilon$ is independent of the training data,

$$
\E\big[(y-\hat{f})^{2}\big] = \E\big[(f + \epsilon - \hat{f})^{2}\big]
 = \E\big[(f-\hat{f})^{2}\big] + \sigma^{2}
$$

because the cross-term $2\E[\epsilon(f-\hat{f})] = 2\E[\epsilon]\E[f-\hat{f}] =
0$. Now add and subtract $\bar{f}$:

$$
\E\big[(f-\hat{f})^{2}\big]
 = \E\big[(f - \bar{f} + \bar{f} - \hat{f})^{2}\big]
 = (f-\bar{f})^{2} + \E\big[(\bar{f}-\hat{f})^{2}\big]
$$

again because the cross-term $2(f-\bar{f})\E[\bar{f}-\hat{f}] = 0$ by the
definition of $\bar{f}$. The first term is squared bias and the second is
variance, giving {{eq:bias-variance}}.

Both cancellations rely on an expectation being zero — the noise having zero
mean, and the deviation from the average prediction having zero mean by
construction. That is the whole proof.

### 6.2 Why AUC equals a ranking probability

Let $S^{+}$ and $S^{-}$ be scores of a random positive and a random negative.
The ROC curve at threshold $t$ is $(\Prob(S^{-} > t), \Prob(S^{+} > t))$. The
area under it is

$$
\text{AUC} = \int_0^1 \text{TPR}\,\dd(\text{FPR})
 = \Prob(S^{+} > S^{-}) + \tfrac{1}{2}\Prob(S^{+} = S^{-})
$$ (eq:auc-probability)

Substituting $u = \Prob(S^{-} > t)$ and integrating over the score distribution
of the negatives turns the area into exactly that probability. Consequences fall
out immediately: AUC is invariant under any strictly increasing transformation
of the scores, so it says nothing about calibration; it is 0.5 for random
ranking; and it depends only on the *order* of scores, so it can be estimated by
counting concordant pairs — the Mann–Whitney $U$ statistic in another guise.

### 6.3 Decomposing the Brier score

The Brier score splits into interpretable parts. Bin predictions into $B$ groups
with mean prediction $\overline{p}_b$ and observed rate $\overline{y}_b$; with
$\bar{y}$ the overall rate,

$$
\text{BS} = \underbrace{\sum_b \tfrac{n_b}{N}(\overline{p}_b-\overline{y}_b)^{2}}_{\text{calibration}}
 - \underbrace{\sum_b \tfrac{n_b}{N}(\overline{y}_b-\bar{y})^{2}}_{\text{resolution}}
 + \underbrace{\bar{y}(1-\bar{y})}_{\text{uncertainty}}
$$ (eq:brier-decomposition)

**Calibration** (lower is better) is how far predicted probabilities are from
observed frequencies. **Resolution** (higher is better) is how much the
predictions separate groups with different outcome rates. **Uncertainty** is the
base-rate variance, a property of the problem that no model can change.

This is {{eq:bias-variance}}'s counterpart for probabilistic classification, and
it makes the independence of the two failure modes formal: a model can be
perfectly calibrated with zero resolution (always predict the base rate) or have
high resolution and terrible calibration (a good ranker with meaningless
numbers). The first is useless and honest; the second is useful and dangerous.

### 6.4 The optimism of selecting a maximum

If $k$ models have true score $\mu$ and independent noise $\sigma_v$, the
expected maximum observed score is approximately

$$
\E\big[\max_i \hat{s}_i\big] \approx \mu + \sigma_v\sqrt{2\log k}
$$ (eq:max-optimism)

The bias grows with the number of models tried, only as $\sqrt{\log k}$ — which
is slow, but not slow enough to ignore. At $k = 1000$ the factor is 3.7. The
selected model is not better than the others; it got the luckiest validation
fold, and the deficit reappears in production.

The counter-measures are all forms of the same idea: use fewer, better-motivated
candidates; use repeated or nested cross-validation to shrink $\sigma_v$; prefer
the simplest model within one standard error of the best rather than the
argmax; and keep a test set that has never been looked at.

## 7. Implementation

```python {tier=A name=bias-variance-measured}
"""The bias-variance decomposition, measured by resampling (eq. 34.1).
"""
import numpy as np

rng = np.random.default_rng(0)

TRUE_SIGMA = 0.30


def true_f(x):
    return np.sin(1.6 * x) + 0.35 * x


def sample(n):
    x = rng.uniform(-3, 3, n)
    return x, true_f(x) + rng.normal(0, TRUE_SIGMA, n)


def fit_poly(x, y, degree):
    A = np.vander(x, degree + 1)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return beta


def predict_poly(beta, x):
    return np.vander(x, len(beta)) @ beta


# a fixed grid of test points at which we measure bias and variance
x_test = np.linspace(-2.6, 2.6, 60)
f_test = true_f(x_test)

N_TRAIN, N_REPS = 40, 400

print(f"true noise variance sigma^2 = {TRUE_SIGMA ** 2:.4f}")
print(f"{N_REPS} independent training sets of {N_TRAIN} points each\n")
print(f"{'degree':>7} {'bias^2':>10} {'variance':>10} {'noise':>9} "
      f"{'total':>10} {'measured MSE':>14}")

results = {}
for degree in (0, 1, 2, 3, 5, 9, 15):
    preds = np.empty((N_REPS, len(x_test)))
    for r in range(N_REPS):
        xt, yt = sample(N_TRAIN)
        preds[r] = predict_poly(fit_poly(xt, yt, degree), x_test)

    mean_pred = preds.mean(axis=0)
    bias2 = np.mean((mean_pred - f_test) ** 2)
    var = np.mean(preds.var(axis=0))
    total = bias2 + var + TRUE_SIGMA ** 2

    # measure the same thing directly: fresh noisy targets at the test points
    y_fresh = f_test[None, :] + rng.normal(0, TRUE_SIGMA, preds.shape)
    mse = np.mean((preds - y_fresh) ** 2)

    results[degree] = (bias2, var, total, mse)
    print(f"{degree:>7} {bias2:>10.4f} {var:>10.4f} "
          f"{TRUE_SIGMA ** 2:>9.4f} {total:>10.4f} {mse:>14.4f}")

best = min(results, key=lambda d: results[d][2])
print(f"\nbias^2 + variance + noise reproduces the measured MSE to within")
print(f"sampling error at every degree — eq. 34.1 is an identity, not an")
print(f"analogy. The total is minimised at degree {best}.")
print("\nTwo details worth not glossing over. Bias barely improves from")
print("degree 1 to 2 (0.475 -> 0.467): the target sin(1.6x) + 0.35x is an")
print("ODD function, so an added x^2 term buys almost nothing and the added")
print("x^3 term at degree 3 buys a great deal. Complexity helps only when it")
print("is the right kind. And at degree 15 — 16 parameters for 40 points —")
print("variance is 900x the noise floor and bias^2 has RISEN, because wild")
print("fits distort the average prediction too. Past the point of collapse")
print("the neat monotone story stops holding.")

print("\nthe irreducible floor:")
print(f"  no model can beat MSE = {TRUE_SIGMA ** 2:.4f} on this problem.")
print("  A validation score below the noise floor is leakage, not skill.")

# --- more data moves the variance term, not the bias term -------------------
print("\n" + "=" * 72)
print("what more data does to each term")
print("=" * 72)
print(f"{'N':>6} " + " ".join(f"{'deg ' + str(d) + ' bias2':>14}"
                              for d in (1, 9)) +
      " " + " ".join(f"{'deg ' + str(d) + ' var':>13}" for d in (1, 9)))
for n_train in (20, 40, 100, 400, 2000):
    row = {}
    for degree in (1, 9):
        preds = np.empty((150, len(x_test)))
        for r in range(150):
            xt, yt = sample(n_train)
            preds[r] = predict_poly(fit_poly(xt, yt, degree), x_test)
        row[degree] = (np.mean((preds.mean(0) - f_test) ** 2),
                       np.mean(preds.var(0)))
    print(f"{n_train:>6} " +
          " ".join(f"{row[d][0]:>14.4f}" for d in (1, 9)) + " " +
          " ".join(f"{row[d][1]:>13.4f}" for d in (1, 9)))

print("\nVariance falls roughly as 1/N for both degrees: a hundredfold more")
print("data cuts it about a hundredfold. Degree 1's bias does not move at")
print("all — 0.470 at N=20 and 0.480 at N=2000 — because bias is a property")
print("of the hypothesis space (Chapter 31), not of the sample.")
print("\nDegree 9's bias column needs a caveat: it reads 3.43 at N=20 and")
print("~0 thereafter. That is not bias falling with data. With 10 parameters")
print("and 20 points the fits are so unstable that the AVERAGE prediction is")
print("itself garbage, and measured bias absorbs it. Once there is enough")
print("data to fit the model at all, degree 9's bias is ~0 and stays there.")
print("\nThe usable conclusion is unchanged: more data buys down variance and")
print("never buys down bias, which is why it fixes overfitting and never")
print("fixes underfitting — and why the learning curve below can tell you")
print("which one you have before you spend the money.")

# --- the learning curve, which is the practical form of the above -----------
print("\n" + "=" * 72)
print("learning curves: would more data help?")
print("=" * 72)
x_big, y_big = sample(4000)
x_val, y_val = sample(4000)

for degree, label in ((1, "degree 1 (too rigid)"), (9, "degree 9 (flexible)")):
    print(f"\n{label}")
    print(f"{'N':>6} {'train RMSE':>12} {'val RMSE':>10} {'gap':>8}")
    for n_train in (10, 20, 50, 200, 1000, 4000):
        beta = fit_poly(x_big[:n_train], y_big[:n_train], degree)
        tr = np.sqrt(np.mean((predict_poly(beta, x_big[:n_train])
                              - y_big[:n_train]) ** 2))
        va = np.sqrt(np.mean((predict_poly(beta, x_val) - y_val) ** 2))
        print(f"{n_train:>6} {tr:>12.4f} {va:>10.4f} {va - tr:>8.4f}")

print(f"\nnoise floor RMSE = {TRUE_SIGMA:.4f}")
print("Degree 1 converges quickly to a validation RMSE well ABOVE the noise")
print("floor and the gap closes: high bias, and more data is wasted money.")
print("Degree 9 starts with a large gap that keeps closing towards the floor:")
print("high variance, and more data is exactly what it needs.")
```

```python {tier=A name=classification-metrics}
"""Classification metrics: where each one misleads, measured.
"""
import numpy as np

rng = np.random.default_rng(4)


def confusion(y, pred):
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    return tp, fp, fn, tn


def prf(y, pred):
    tp, fp, fn, _ = confusion(y, pred)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return prec, rec, f1


def roc_auc(y, score):
    """AUC as the probability a positive outranks a negative (eq. 34.9).

    Computed via ranks, which handles ties correctly and is O(n log n).
    """
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0     # average rank for ties
        i = j + 1
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def pr_auc(y, score):
    """Average precision: the step-wise area under the PR curve."""
    order = np.argsort(-score, kind="mergesort")
    y_s = y[order]
    tp = np.cumsum(y_s)
    prec = tp / np.arange(1, len(y_s) + 1)
    n_pos = max(1, int(y.sum()))
    return float(np.sum(prec * y_s) / n_pos)


# --- the AUC identity, checked by brute force -------------------------------
y_small = np.array([1, 0, 1, 1, 0, 0, 1, 0.])
s_small = np.array([0.9, 0.8, 0.7, 0.4, 0.4, 0.3, 0.2, 0.1])
pos, neg = s_small[y_small == 1], s_small[y_small == 0]
pairs = [(1.0 if p > n else 0.5 if p == n else 0.0) for p in pos for n in neg]
print(f"AUC by rank formula   : {roc_auc(y_small, s_small):.6f}")
print(f"AUC by pair counting  : {np.mean(pairs):.6f}")
print("Identical — AUC IS the probability that a positive outranks a")
print("negative, ties counting a half (eq. 34.9).\n")

# --- ROC-AUC's blind spot under imbalance -----------------------------------
print("=" * 72)
print("the same ranker, four different base rates")
print("=" * 72)
print(f"{'positive rate':>14} {'ROC-AUC':>9} {'PR-AUC':>8} "
      f"{'PR baseline':>12} {'lift over baseline':>19}")

for rate in (0.50, 0.10, 0.01, 0.001):
    n = 200000
    y = (rng.random(n) < rate).astype(float)
    # score quality held FIXED: same two Gaussians regardless of base rate
    score = rng.normal(np.where(y == 1, 1.4, 0.0), 1.0)
    print(f"{rate:>14.3f} {roc_auc(y, score):>9.4f} {pr_auc(y, score):>8.4f} "
          f"{y.mean():>12.4f} {pr_auc(y, score) / y.mean():>19.1f}x")

print("\nROC-AUC is essentially constant: it is a property of the ranker and")
print("is invariant to class balance. PR-AUC collapses, because precision")
print("depends on how many negatives are competing for the top of the list.")
print("At a 0.1% base rate the ranker is unchanged and the product built on")
print("it is unusable. Report PR-AUC against its baseline under imbalance.")

# --- accuracy under imbalance -----------------------------------------------
print("\n" + "=" * 72)
print("accuracy is uninformative under imbalance")
print("=" * 72)
n = 20000
y = (rng.random(n) < 0.01).astype(float)
score = rng.normal(np.where(y == 1, 1.6, 0.0), 1.0)
print(f"{'model':<26} {'accuracy':>9} {'precision':>10} {'recall':>8} "
      f"{'F1':>7} {'ROC-AUC':>9}")
always0 = np.zeros(n)
p, r, f = prf(y, always0)
print(f"{'always predict negative':<26} {(always0 == y).mean():>9.4f} "
      f"{p:>10.4f} {r:>8.4f} {f:>7.4f} {0.5:>9.4f}")
for t in (0.5, 1.0, 2.0):
    pred = (score >= t).astype(float)
    p, r, f = prf(y, pred)
    print(f"{'threshold ' + str(t):<26} {(pred == y).mean():>9.4f} "
          f"{p:>10.4f} {r:>8.4f} {f:>7.4f} {roc_auc(y, score):>9.4f}")
print("\nThe useless model wins on accuracy and scores zero on everything")
print("that measures whether it found anything.")

# --- calibration and discrimination are independent -------------------------
print("\n" + "=" * 72)
print("calibration and discrimination are independent (section 5.3)")
print("=" * 72)


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def ece(y, p, n_bins=10):
    """Expected calibration error (eq. 34.7), equal-count bins."""
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    total = 0.0
    for i in range(n_bins):
        m = (p >= edges[i]) & (p <= edges[i + 1])
        if m.sum():
            total += m.sum() / len(p) * abs(y[m].mean() - p[m].mean())
    return total


n = 40000
z = rng.normal(0, 1.5, n)
p_true = 1 / (1 + np.exp(-z))
y = (rng.random(n) < p_true).astype(float)

variants = {
    "perfectly calibrated": p_true,
    "halved (same ranking)": p_true * 0.5,
    "over-confident": np.clip(1 / (1 + np.exp(-2.5 * z)), 1e-6, 1 - 1e-6),
    "always the base rate": np.full(n, y.mean()),
}
print(f"{'model':<24} {'ROC-AUC':>9} {'Brier':>9} {'ECE':>8} {'log loss':>10}")
for name, p in variants.items():
    ll = -np.mean(y * np.log(np.clip(p, 1e-12, 1))
                  + (1 - y) * np.log(np.clip(1 - p, 1e-12, 1)))
    print(f"{name:<24} {roc_auc(y, p):>9.4f} {brier(y, p):>9.4f} "
          f"{ece(y, p):>8.4f} {ll:>10.4f}")

print("\nThe first three have IDENTICAL ROC-AUC — halving or sharpening every")
print("probability preserves the order — while Brier, ECE and log loss")
print("separate them completely. The last has perfect calibration and zero")
print("resolution (eq. 34.10): honest and useless. No single number covers")
print("both failure modes, which is why you report at least two.")
```

## 8. Practical Example

```python {tier=A name=model-selection}
"""Model selection that does not lie: nested CV, one-SE rule, and a
measurement of the optimism from evaluating too many candidates.
"""
import numpy as np

rng = np.random.default_rng(21)


def make_data(n):
    X = rng.normal(size=(n, 12))
    z = 1.4 * X[:, 0] - 1.1 * X[:, 1] + 0.8 * X[:, 2] * X[:, 3] - 0.4
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(float)
    return X, y


def sigmoid(z):
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1 / (1 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1 + e)
    return out


def fit_logistic(X, y, lam, n_iter=60):
    """Newton with an L2 penalty; ridge-stabilised so it never fails."""
    A = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(A.shape[1])
    for _ in range(n_iter):
        p = sigmoid(A @ w)
        g = A.T @ (p - y) / len(y)
        g[1:] += 2 * lam * w[1:]
        S = np.maximum(p * (1 - p), 1e-9)
        H = (A * S[:, None]).T @ A / len(y) + (2 * lam + 1e-8) * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < 1e-10:
            break
    return w


def score_logistic(w, X, y):
    p = sigmoid(np.column_stack([np.ones(len(X)), X]) @ w)
    return float(np.mean((p >= 0.5) == (y == 1)))


def kfold(n, k, seed):
    idx = np.random.default_rng(seed).permutation(n)
    return [(np.concatenate([idx[:i * n // k], idx[(i + 1) * n // k:]]),
             idx[i * n // k:(i + 1) * n // k]) for i in range(k)]


GRID = np.logspace(-4, 1, 12)      # log grid, per section 5.6 of Chapter 32

# --- 1. plain CV, and the optimism it hides ---------------------------------
X, y = make_data(600)
print("=" * 72)
print("1. cross-validation on a grid of 12 lambdas")
print("=" * 72)
folds = kfold(len(y), 5, 1)
means, ses = [], []
for lam in GRID:
    accs = [score_logistic(fit_logistic(X[tr], y[tr], lam), X[va], y[va])
            for tr, va in folds]
    means.append(np.mean(accs))
    ses.append(np.std(accs, ddof=1) / np.sqrt(len(accs)))
means, ses = np.array(means), np.array(ses)

best_i = int(means.argmax())
print(f"{'lambda':>10} {'CV accuracy':>13} {'SE':>8}")
for i, lam in enumerate(GRID):
    mark = "  <-- best" if i == best_i else ""
    print(f"{lam:>10.5f} {means[i]:>13.4f} {ses[i]:>8.4f}{mark}")

# the one-standard-error rule: simplest model within 1 SE of the best
threshold = means[best_i] - ses[best_i]
one_se_i = int(np.max(np.where(means >= threshold)[0]))   # largest lambda
print(f"\nbest lambda      : {GRID[best_i]:.5f}  (CV {means[best_i]:.4f})")
print(f"one-SE rule picks: {GRID[one_se_i]:.5f}  (CV {means[one_se_i]:.4f})")
print("The one-SE rule takes the strongest regularisation whose score is")
print("statistically indistinguishable from the best, on the grounds that")
print("the argmax of a noisy grid is itself a noisy quantity.")

# --- 2. the CV score of the SELECTED model is biased ------------------------
print("\n" + "=" * 72)
print("2. how optimistic is that CV score? (eq. 34.11)")
print("=" * 72)
Xh, yh = make_data(20000)          # a large fresh sample as ground truth
true_best = score_logistic(fit_logistic(X, y, GRID[best_i]), Xh, yh)
true_1se = score_logistic(fit_logistic(X, y, GRID[one_se_i]), Xh, yh)
print(f"CV score of the chosen model     : {means[best_i]:.4f}")
print(f"its accuracy on 20,000 fresh rows: {true_best:.4f}")
print(f"optimism                         : {means[best_i] - true_best:+.4f}")
print(f"\none-SE choice, fresh accuracy    : {true_1se:.4f}")

# --- 3. optimism grows with the number of candidates ------------------------
print("\n" + "=" * 72)
print("3. optimism grows with how many models you try")
print("=" * 72)
print("Candidates are RANDOM SEEDS for the same model on the same data, so")
print("every genuine difference between them is exactly zero. Any apparent")
print("winner is noise by construction.")
print("\nOne run of this experiment says nothing — the whole point is that a")
print("single maximum is a lucky draw. So it is repeated on 25 independent")
print("datasets and the optimism averaged.\n")

N_REPS_SEL, K_MAX = 25, 64
Xh2, yh2 = make_data(20000)
ks = (1, 2, 4, 8, 16, 32, 64)
opt_by_k = {k: [] for k in ks}
se_samples = []

for rep in range(N_REPS_SEL):
    X2, y2 = make_data(300)
    folds2 = kfold(len(y2), 5, 5000 + rep)
    scored = []
    for c in range(K_MAX):
        # identical model, different bootstrap of the training folds: the
        # candidates are interchangeable, so all differences are sampling noise
        g = np.random.default_rng(90000 + 1000 * rep + c)
        accs = []
        for tr, va in folds2:
            boot = g.choice(len(tr), len(tr), replace=True)
            w = fit_logistic(X2[tr][boot], y2[tr][boot], 0.01)
            accs.append(score_logistic(w, X2[va], y2[va]))
        scored.append((float(np.mean(accs)), boot, w))
        se_samples.append(np.std(accs, ddof=1) / np.sqrt(5))
    for k in ks:
        cv_best, _, w_best = max(scored[:k], key=lambda t: t[0])
        opt_by_k[k].append(cv_best - score_logistic(w_best, Xh2, yh2))

se_typical = float(np.mean(se_samples))
print(f"typical fold-to-fold SE of one candidate: {se_typical:.4f}\n")
base = float(np.mean(opt_by_k[1]))
print(f"{'candidates':>11} {'mean optimism':>15} {'rise over k=1':>15} "
      f"{'SE x sqrt(2 log k)':>20}")
for k in ks:
    predicted = se_typical * np.sqrt(2 * np.log(k)) if k > 1 else 0.0
    print(f"{k:>11} {np.mean(opt_by_k[k]):>15.4f} "
          f"{np.mean(opt_by_k[k]) - base:>15.4f} {predicted:>20.4f}")

print("\nRead the third column, not the second. The absolute level starts")
print("negative because each candidate is fitted on a bootstrap of its")
print("training folds and is therefore slightly worse than a full fit — a")
print("constant offset that says nothing about selection. What matters is")
print("that the winner's optimism RISES monotonically with the number of")
print("candidates it beat, by 0.038 accuracy points from k=1 to k=64, while")
print("not one of these models is genuinely better than another.")
print("\nThe last column over-predicts by about a factor of two, and the")
print("reason is instructive: eq. 34.11 assumes k INDEPENDENT candidates.")
print("These share a dataset and a model, so their scores are strongly")
print("correlated and the effective number of independent draws is far")
print("below 64. The formula is an upper bound in practice, and the")
print("qualitative claim — sqrt(log k) growth, never zero — survives.")
print("\nThis is why the test set is touched once, and why 'we tried 300")
print("configurations' should make you trust a reported improvement LESS.")

# --- 4. nested CV: the honest estimate of the whole procedure ---------------
print("\n" + "=" * 72)
print("4. nested cross-validation")
print("=" * 72)
print("The selection space is now realistic: 6 lambdas x 8 feature subsets")
print("= 48 candidates, on 300 rows. Every subset keeps the four informative")
print("features and adds three of the eight noise features, so the")
print("candidates are genuinely interchangeable — as in section 3, any")
print("winner is a winner by luck. Section 3 says a search this wide will")
print("produce a visibly optimistic score.\n")

Xs, ys = make_data(300)
LAM_GRID = np.logspace(-3, 0, 6)
SUBSETS = [np.concatenate([[0, 1, 2, 3],
                           np.random.default_rng(7000 + s).choice(
                               np.arange(4, 12), 3, replace=False)])
           for s in range(8)]
CANDIDATES = [(lam, cols) for lam in LAM_GRID for cols in SUBSETS]


def select(Xsel, ysel, k_inner, seed):
    """Run the search. Returns the winner, its CV score, and the per-fold
    models that produced that score — keeping them lets us re-score exactly
    those fits on fresh data, with no difference in training-set size to
    confound the comparison."""
    inner = kfold(len(ysel), k_inner, seed)
    best, best_score, best_models = None, -np.inf, None
    for lam, cols in CANDIDATES:
        models = [fit_logistic(Xsel[itr][:, cols], ysel[itr], lam)
                  for itr, _ in inner]
        a = [score_logistic(m, Xsel[iva][:, cols], ysel[iva])
             for m, (_, iva) in zip(models, inner)]
        if np.mean(a) > best_score:
            best, best_score, best_models = (lam, cols), float(np.mean(a)), models
    return best, best_score, best_models


def nested_cv(Xn, yn, k_outer, seed):
    """Inner loop selects, outer loop scores what the inner loop chose."""
    scores = []
    for oi, (tr, te) in enumerate(kfold(len(yn), k_outer, seed)):
        (lam_i, cols_i), _, _ = select(Xn[tr], yn[tr], 4, seed + 1 + oi)
        scores.append(score_logistic(
            fit_logistic(Xn[tr][:, cols_i], yn[tr], lam_i),
            Xn[te][:, cols_i], yn[te]))
    return float(np.mean(scores))


# All three estimators, on the same 12 independent datasets, so they are
# directly comparable. The honest figure re-scores the winner's OWN per-fold
# fits on 20,000 fresh rows — same models, same training-set size, so the only
# difference is whether the scoring data was also used to choose them.
naive_scores, honest_scores, nested_scores = [], [], []
for rep in range(12):
    Xr_, yr_ = make_data(300)
    (lam_r, cols_r), inner_r, models_r = select(Xr_, yr_, 5, 6000 + rep)
    naive_scores.append(inner_r)
    honest_scores.append(np.mean([score_logistic(m, Xh[:, cols_r], yh)
                                  for m in models_r]))
    nested_scores.append(nested_cv(Xr_, yr_, 5, 8000 + 20 * rep))
    print(f"  run {rep + 1:>2}: naive {inner_r:.4f}   "
          f"nested {nested_scores[-1]:.4f}   honest {honest_scores[-1]:.4f}")


def se(v):
    return float(np.std(v, ddof=1) / np.sqrt(len(v)))


naive_bias = np.array(naive_scores) - np.array(honest_scores)
nested_bias = np.array(nested_scores) - np.array(honest_scores)

print(f"\naveraged over 12 independent runs of the 48-candidate search:")
print(f"{'estimator':<42} {'value':>8} {'bias':>9} {'SE of bias':>11}")
print(f"{'naive: winner CV score (what gets quoted)':<42} "
      f"{np.mean(naive_scores):>8.4f} {np.mean(naive_bias):>+9.4f} "
      f"{se(naive_bias):>11.4f}")
print(f"{'nested CV':<42} "
      f"{np.mean(nested_scores):>8.4f} {np.mean(nested_bias):>+9.4f} "
      f"{se(nested_bias):>11.4f}")
print(f"{'honest: same fits on 20,000 fresh rows':<42} "
      f"{np.mean(honest_scores):>8.4f} {0.0:>+9.4f} {'-':>11}")

print("\nRead this carefully, because it does not say what the slogan says.")
print("\nThe naive score is biased UPWARD, as predicted: same models, same")
print("training-set size, same everything — the only difference is that the")
print("data used to score them also chose them.")
print("\nNested CV is biased DOWNWARD, and by more. That is not a bug and it")
print("is well known: each outer fold selects and fits on 4/5 of the data,")
print("so nested CV estimates the quality of the procedure applied to a")
print("SMALLER dataset than the one you will actually train on. It is a")
print("conservative estimator, not an unbiased one — and 'conservative' is")
print("usually the direction you want to be wrong in.")
print("\nBoth biases here are a fraction of a point, and comparable to their")
print("own standard errors, because the 48 candidates were deliberately")
print("interchangeable and 300 rows is not tiny. With a genuinely wide")
print("search over scarce data the naive bias grows (section 3) while the")
print("nested penalty does not, which is when nested CV earns its 20x cost.")
print("\nThe durable lesson is not 'always use nested CV'. It is that a score")
print("computed on data that participated in choosing the model is not a")
print("measurement, and that every honest alternative costs something —")
print("compute, data, or a conservative bias. Pick which one you can afford.")
```

## 9. Common Mistakes

**Reporting accuracy on imbalanced data.** The trivial model wins.

**Using ROC-AUC under heavy imbalance.** It is invariant to base rate; PR-AUC is
not, and the difference is measured in {{sec:7-implementation}}.

**Reporting PR-AUC without its baseline.** 0.4 is excellent at 1% and terrible
at 45%.

**Assuming a good ranker has good probabilities.** Halving every probability
leaves AUC untouched and destroys calibration.

**Optimising a threshold on the test set.** That is fitting; use validation.

**Reusing the test set.** Once it has selected anything, it is a validation set.

**Reading a single CV number without its standard error.** Differences of 0.3%
across folds with a 1% standard error are noise.

**Reporting the inner CV score of a tuned model.** It is the maximum of a noisy
grid; nested CV gives the honest number.

**Using MAPE.** Undefined at zero, asymmetric, and biases forecasts low.

**Ignoring the noise floor.** A score better than irreducible error means
leakage, not skill.

## 10. Connection to Previous Chapters

{{ch:math-inference}} supplied the sampling variability behind every standard
error here, and the selection effect that {{eq:max-optimism}} quantifies —
model selection is multiple comparisons wearing a different hat.
{{ch:ds-leakage}} supplied the validation designs; this chapter adds leakage
through repeated evaluation, which no feature audit will catch.
{{ch:ml-linear-regression}} supplied the ridge path whose $\lambda$ traces the
bias-variance curve of {{eq:bias-variance}} exactly. {{ch:ml-logistic}} supplied
the calibrated probabilities that {{eq:brier-decomposition}} decomposes, and the
cost-based threshold that {{tbl:metric-choice}} calls expected cost.

Forward: every remaining chapter in this part is placed on the bias-variance
curve. {{ch:ml-knn-nb}} is a variance-dominated method with a complexity knob
($k$) that moves along it directly. {{ch:ml-trees}} is high-variance and low-bias
until pruned. {{ch:ml-forests}} reduces variance by averaging.
{{ch:ml-boosting}} reduces bias by accumulating. {{ch:dl-regularization}} returns
to the same trade-off where {{eq:bias-variance}}'s classical U-shape no longer
describes what happens.

## 11. Exercises

**Beginner**

1. Define bias, variance and noise in one sentence each.
2. A model scores 0.99 accuracy on data with a 1% positive rate. What do you
   know?
3. Compute precision, recall and $F_1$ for $TP=20$, $FP=30$, $FN=10$.
4. What does ROC-AUC = 0.5 mean? What does 0.3 mean?
5. Why is $R^{2}$ sometimes negative on a test set?

**Intermediate**

6. Given a learning curve with a large persistent gap, what do you do?
7. Explain why ROC-AUC is invariant to class balance and PR-AUC is not.
8. Why is accuracy not a proper scoring rule? Give a case where lying about
   confidence improves it.
9. Explain the difference between calibration and resolution in
   {{eq:brier-decomposition}}.
10. When would you prefer MAE to RMSE, and what does that imply about the
    prediction being requested?
11. Why does the one-standard-error rule prefer a simpler model?

**Advanced**

12. Derive {{eq:bias-variance}} and state exactly where independence of the noise
    is used.
13. Prove {{eq:auc-probability}} and derive the tie correction.
14. Derive {{eq:brier-decomposition}} and verify it numerically.
15. Explain why {{eq:bias-variance}} does not decompose cleanly for 0-1 loss, and
    give a case where variance *reduces* 0-1 error.
16. Design an evaluation for a model whose errors have asymmetric, non-linear
    costs, and say why no standard metric suffices.

**Implementation**

17. Extend {{sec:7-implementation}} to measure the decomposition for a ridge
    path and confirm bias rises and variance falls with $\lambda$.
18. Implement bootstrap confidence intervals for AUC and report the width at
    several sample sizes.
19. Build a reliability diagram with binomial error bars and calibrate a model
    by Platt scaling and by isotonic regression, comparing the two.
20. Reproduce the optimism experiment for a realistic hyperparameter search and
    report the gap.

**Reasoning**

21. A team reports a 2% improvement after evaluating 400 configurations on one
    validation split, with a fold standard error of 1.2%. What is your
    assessment?
22. When is a worse-scoring model the right one to ship?

## 12. Chapter Summary

Prediction error decomposes exactly, for squared loss, into squared bias,
variance and irreducible noise. The measurement in {{sec:7-implementation}}
reproduces the identity across polynomial degrees, shows bias falling and
variance rising with complexity, and shows more data reducing variance while
leaving bias untouched. That last fact is why more data fixes overfitting and
never fixes underfitting.

The learning curve is the practical form of the same diagnostic: converged
curves at a high error mean bias and no amount of data will help; a persistent
gap means variance and it will.

No single classification metric is sufficient. Accuracy is uninformative under
imbalance. ROC-AUC measures ranking and is invariant to base rate, which makes
it blind to the false positives that destroy a rare-event product; PR-AUC is
not, and must be read against its baseline. Calibration and discrimination are
independent properties — halving every probability leaves AUC unchanged and
ruins the probabilities — so report at least one of each.

Brier score and log loss are proper scoring rules, uniquely minimised by honest
reporting; accuracy is not. The Brier decomposition separates calibration,
resolution and irreducible uncertainty, making the two failure modes formal.

RMSE and MAE request different predictions — the conditional mean and the
conditional median — so switching between them changes the optimal model rather
than merely rescoring it. MAPE should be avoided.

Model selection is multiple comparisons. The best of $k$ noisy validation scores
is optimistic by roughly $\sigma_v\sqrt{2\log k}$; the measured experiment
confirms the growth with candidate count, and also shows the formula
over-predicting about twofold because real candidates share data and are
therefore correlated rather than independent. Nested cross-validation scores the
procedure rather than the winner and is mildly *pessimistic* rather than
unbiased, since each outer fold trains on less data than the final model will.
The one-standard-error rule resists the noise in an argmax, and the test set may
be used exactly once.

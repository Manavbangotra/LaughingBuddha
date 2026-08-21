---
id: math-inference
number: 10
part: I
tier: focused
status: reviewed
requires: [math-covariance]
provides: [estimator, sampling-distribution, law-of-large-numbers,
           central-limit-theorem, confidence-interval, hypothesis-test, p-value,
           statistical-power]
citations: [deisenroth2020]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Distinguish a population parameter from a sample statistic, and explain why
   the second is a random variable.
2. Explain what a sampling distribution is and why it is the central idea of
   inference.
3. State the law of large numbers and the central limit theorem, and say what
   each does and does not promise.
4. Compute a standard error and explain the $1/\sqrt{n}$ law and its practical
   consequences.
5. Construct and correctly interpret a confidence interval.
6. Conduct a hypothesis test and state precisely what a p-value means — and
   what it does not.
7. Compute the sample size needed to detect an effect of a given size, and
   explain why underpowered experiments are worse than no experiment.
8. Explain multiple-comparisons inflation and why it matters for model
   selection.

## 2. Why This Matters

Every number you will ever report about a model is an estimate from a finite
sample, and is therefore uncertain. "This model achieves 94.2% accuracy" is not
a fact about the model; it is a fact about the model *and* the particular test
set it was measured on. Run it on a different sample and you get a different
number.

That distinction is the whole subject of this chapter, and getting it wrong has
concrete costs.

**Model comparison.** Model A scores 94.2% and model B scores 93.8%. Is A
better? Without knowing the uncertainty in those numbers, the question has no
answer. On a 1,000-example test set the standard error is about 0.7 percentage
points, so a 0.4-point difference is noise ({{ch:ml-metrics}}).

**A/B testing.** Deploying a change because it improved a metric by 2% in a
week-long test, when the noise level is 3%, is a decision made by coin flip
({{ch:ds-experiments}}).

**Benchmark results.** Published differences between models on small benchmarks
are frequently within noise, and the field's reproducibility problems are partly
a consequence of not saying so ({{ch:ev-llm-benchmarks}}).

**Hyperparameter search.** Try 200 configurations and pick the best validation
score, and you have selected partly for genuine quality and partly for lucky
noise. The winner's score is systematically optimistic — this is the multiple
comparisons problem, and it is why a held-out test set exists at all
({{ch:mle-splits}}).

## 3. Prerequisites

{{ch:math-covariance}} for variance, standard deviation, and the fact that
variances of independent quantities add. {{ch:math-random-vars}} for random
variables, expectation and the Gaussian distribution.
{{ch:math-probability}} for conditioning — the definition of a p-value is a
conditional probability, and reading it in the wrong direction is the most
common error in the subject.

## 4. Intuitive Explanation

### 4.1 Parameters and statistics

The **population** is everything you could have measured — every possible input
your model will ever see. Its true accuracy is a fixed number you can never
observe, called a **parameter**.

A **sample** is what you actually have: your test set. The accuracy you compute
on it is a **statistic**, and it is a random variable, because a different
sample would have given a different value.

Statistical inference is the business of saying something about the parameter
using only the statistic, while being honest about the gap between them.

Notation follows a convention worth adopting: Greek letters for parameters
($\mu$, $\sigma$, $\theta$), Latin or hatted symbols for statistics
($\bar{x}$, $s$, $\hat{\theta}$).

### 4.2 The sampling distribution

Here is the idea everything else rests on.

Imagine drawing your test set again and again — a hundred different test sets of
1,000 examples each — and computing the accuracy on every one. You would get a
hundred slightly different numbers. Their distribution is the
{{term:sampling-distribution}}, and it describes how much your single measured
accuracy could have varied by luck.

You cannot actually draw a hundred test sets. That is the point: the
{{term:central-limit-theorem}} tells you what that distribution looks like
*without* having to draw it. From one sample you can infer the spread of the
statistic across samples you never took.

This is the conceptual move that makes statistics possible, and it is the one
beginners most often skip. Every confidence interval and every p-value is a
statement about the sampling distribution.

### 4.3 Two theorems

The **{{term:law-of-large-numbers}}** says the sample mean converges to the true
mean as the sample grows. Averages settle down. This is why more test data gives
a better estimate — but it says nothing about how fast.

The **central limit theorem** says how fast, and more: it says the sample mean
is approximately Gaussian around the true mean, with standard deviation
$\sigma/\sqrt{n}$, **regardless of the distribution of the individual values**.

That last clause is what makes the theorem remarkable. The underlying data can
be binary, skewed, bimodal, or arbitrarily strange; the distribution of its mean
is still approximately normal. It is why the Gaussian appears everywhere in
statistics, and it is why one formula for a confidence interval works across
wildly different problems.

### 4.4 The tyranny of $\sqrt{n}$

The **standard error** — the standard deviation of the sampling distribution — is

$$
\text{SE} = \frac{\sigma}{\sqrt{n}}
$$ (eq:standard-error)

The square root is the single most consequential fact in experimental design.
**To halve your uncertainty you must quadruple your data.** To improve it
tenfold you need a hundred times as much.

{#tbl:sqrt-n caption="Precision against sample size for a binary metric near 50%, where σ ≈ 0.5. Diminishing returns set in quickly, and they set in at the same rate for every problem."}

| $n$ | Standard error | 95% interval half-width |
|---|---|---|
| 100 | 5.0% | ±9.8% |
| 1,000 | 1.6% | ±3.1% |
| 10,000 | 0.5% | ±1.0% |
| 100,000 | 0.16% | ±0.31% |
| 1,000,000 | 0.05% | ±0.10% |

This table explains a great deal about how AI evaluation actually behaves. A
1,000-example benchmark cannot reliably distinguish models differing by less
than about three percentage points, no matter how carefully the evaluation is
run. Reported differences smaller than that are not measuring model quality
({{ch:ev-llm-benchmarks}}).

## 5. Formal Explanation

### 5.1 Estimators

An {{term:estimator}} $\hat{\theta}$ is a function of the sample used to
estimate a parameter $\theta$. Being a function of random data, it is itself a
random variable with a distribution, a mean, and a variance.

$$
\text{Bias}(\hat{\theta}) = \E[\hat{\theta}] - \theta
$$ (eq:bias)

An estimator is **unbiased** when the bias is zero — it is right *on average*
across samples, though not on any particular one. The sample mean is unbiased
for the population mean. The sample variance with Bessel's correction
({{ch:math-covariance}}) is unbiased; dividing by $n$ instead of $n-1$ gives a
biased estimator that systematically underestimates.

Unbiasedness is not the only virtue, and it is sometimes worth giving up. The
**mean squared error** decomposes as

$$
\text{MSE}(\hat{\theta}) = \text{Bias}(\hat{\theta})^{2} + \Var(\hat{\theta})
$$ (eq:mse-decomposition)

A biased estimator with much lower variance can beat an unbiased one. That is
precisely the argument for regularisation ({{ch:math-optimization}}) and the
subject of the bias-variance trade-off ({{ch:ml-metrics}}).

### 5.2 The two limit theorems

**Law of large numbers.** For iid $X_1, \ldots, X_n$ with finite mean $\mu$, the
sample mean $\bar{X}_n \to \mu$ in probability as $n \to \infty$.

**Central limit theorem.** For iid $X_i$ with mean $\mu$ and *finite variance*
$\sigma^{2}$:

$$
\frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \;\xrightarrow{d}\; \mathcal{N}(0, 1)
$$ (eq:clt)

Equivalently, $\bar{X}_n$ is approximately $\mathcal{N}(\mu, \sigma^{2}/n)$ for
large $n$.

> WARNING: The CLT requires finite variance and independent observations, and
> both conditions fail in practice more often than people expect. Heavy-tailed
> quantities — request latencies, file sizes, income — may have variance so
> large that convergence needs an impractical $n$. And observations are not
> independent when the same user appears in many rows, when examples come from
> the same document, or when time-series points are autocorrelated. In each
> case the effective sample size is smaller than $n$, sometimes by a large
> factor, and every interval computed from {{eq:standard-error}} is too narrow
> ({{ch:ds-experiments}}).

The standard error {{eq:standard-error}} follows from
{{eq:variance-sum-independent}}: the variance of a sum of $n$ independent
variables is $n\sigma^{2}$, so the variance of their average is
$n\sigma^{2}/n^{2} = \sigma^{2}/n$, and the standard deviation is
$\sigma/\sqrt{n}$. The square root comes from taking a square root of a
variance, nothing more mysterious.

### 5.3 Confidence intervals

A 95% {{term:confidence-interval}} for a mean is

$$
\bar{x} \pm 1.96\,\frac{s}{\sqrt{n}}
$$ (eq:confidence-interval)

where $s$ is the sample standard deviation and 1.96 is the standard normal
quantile leaving 2.5% in each tail.

> IMPORTANT: The correct interpretation is about the *procedure*, not the
> interval. "If I repeated this experiment many times and built an interval each
> time, 95% of those intervals would contain the true value." It is **not**
> "there is a 95% probability the true value is in this particular interval" —
> under a frequentist reading the true value is fixed and either is or is not
> inside, so that probability is 0 or 1. The distinction sounds pedantic and is
> not: it is exactly the $\Prob(A \given B)$ versus $\Prob(B \given A)$
> confusion of {{ch:math-probability}}, and it is why Bayesian credible
> intervals — which *do* support the natural reading — are a different object.

For proportions, such as accuracy, $\sigma^{2} = p(1-p)$ by {{eq:bernoulli}}, so

$$
\hat{p} \pm 1.96\sqrt{\frac{\hat{p}(1-\hat{p})}{n}}
$$ (eq:proportion-ci)

This is the formula that should accompany every reported accuracy, and almost
never does.

### 5.4 Hypothesis testing

A {{term:hypothesis-test}} asks whether the data are surprising enough under a
"nothing is happening" assumption to justify abandoning it.

1. State a **null hypothesis** $H_0$ (no effect) and an alternative $H_1$.
2. Choose a significance level $\alpha$, conventionally 0.05.
3. Compute a test statistic.
4. Compute the {{term:p-value}}: the probability of a statistic at least this
   extreme *if $H_0$ were true*.
5. Reject $H_0$ if $p < \alpha$.

$$
p = \Prob(\text{statistic at least as extreme} \given H_0)
$$ (eq:p-value)

> IMPORTANT: Read {{eq:p-value}} carefully. The conditioning is on $H_0$. A
> p-value is **not** the probability that $H_0$ is true, and it is not the
> probability that your result is a fluke. Those would be
> $\Prob(H_0 \given \text{data})$, which requires a prior and Bayes' theorem
> ({{ch:math-probability}}) and is a different number entirely.

Two error types, and they trade off:

{#tbl:error-types caption="The two ways a hypothesis test can be wrong. α is chosen; β follows from the sample size and the true effect size."}

| | $H_0$ true | $H_0$ false |
|---|---|---|
| **Reject $H_0$** | Type I error (rate $\alpha$) | correct |
| **Fail to reject** | correct | Type II error (rate $\beta$) |

{{term:statistical-power}} is $1 - \beta$: the probability of detecting a real
effect. It depends on the effect size, the noise, the sample size, and $\alpha$.

### 5.5 Power and sample size

For comparing two proportions, the sample size per group needed to detect a
difference $\delta$ with power $1-\beta$ at level $\alpha$ is approximately

$$
n \approx \frac{2\,(z_{\alpha/2} + z_{\beta})^{2}\,\bar{p}(1-\bar{p})}{\delta^{2}}
$$ (eq:sample-size)

with $z_{0.025} = 1.96$ and $z_{0.20} = 0.84$ for the conventional 5% level and
80% power.

The $\delta^{2}$ in the denominator is the important part: **halving the effect
you want to detect quadruples the sample you need.** Detecting small
improvements is expensive, and this formula tells you how expensive before you
run the experiment rather than after.

> WARNING: An underpowered experiment is worse than no experiment, and the
> reason is not obvious. With low power, most effects go undetected — but among
> the effects that *are* detected, a large fraction are noise, and those that
> are real are systematically overestimated in magnitude. So an underpowered
> study that reaches significance reports an inflated effect size. Compute the
> power before running the experiment ({{ch:ds-experiments}}).

## 6. Mathematical Foundation

### 6.1 Deriving the standard error

Let $X_1, \ldots, X_n$ be iid with variance $\sigma^{2}$. Using linearity of
expectation and {{eq:variance-scaling}} with
{{eq:variance-sum-independent}}:

$$
\Var(\bar{X}) = \Var\!\left(\frac{1}{n}\sum_{i=1}^{n} X_i\right)
  = \frac{1}{n^{2}}\Var\!\left(\sum_{i=1}^{n} X_i\right)
  = \frac{1}{n^{2}} \cdot n\sigma^{2}
  = \frac{\sigma^{2}}{n}
$$ (eq:variance-of-mean)

so $\text{SE} = \sigma/\sqrt{n}$.

Note exactly where independence entered: the step
$\Var(\sum X_i) = n\sigma^{2}$ used {{eq:variance-sum-independent}}, which
requires zero covariance. With correlated observations the sum picks up
covariance terms and the true standard error is larger — often much larger.
This is the single most common way that reported confidence intervals are wrong.

### 6.2 A worked comparison

Two models are evaluated on the same 1,000-example test set. Model A gets 94.2%,
model B gets 93.8%. Is A better?

The standard error for A, from {{eq:proportion-ci}}:

$$
\text{SE}_A = \sqrt{\frac{0.942 \times 0.058}{1000}} = \sqrt{0.0000546} = 0.0074
$$

so the 95% interval is $0.942 \pm 0.0145$, or $[92.8\%, 95.6\%]$.

For B: $\text{SE}_B = \sqrt{0.938 \times 0.062 / 1000} = 0.0076$, interval
$[92.3\%, 95.3\%]$.

The intervals overlap heavily. The difference is $0.4$ percentage points; the
standard error of the difference, for independent samples, is
$\sqrt{0.0074^{2} + 0.0076^{2}} = 0.0106$. The difference is $0.4/1.06 = 0.38$
standard errors — nowhere near significant.

**How much data would be needed?** From {{eq:sample-size}} with
$\delta = 0.004$ and $\bar{p} = 0.94$:

$$
n \approx \frac{2(1.96 + 0.84)^{2}(0.94)(0.06)}{0.004^{2}}
  = \frac{2(7.84)(0.0564)}{0.000016} \approx 55{,}300
$$

About 55,000 examples per model. That is the honest answer to "is A better than
B", and it explains why marginal improvements on small benchmarks should not be
believed.

> PRODUCTION TIP: Because both models are evaluated on the *same* test set,
> their errors are correlated — both find the same easy examples easy. A paired
> test, such as McNemar's test, uses that correlation and needs far fewer
> examples than the independent-samples calculation above. Whenever you compare
> models on a shared test set, use a paired test ({{ch:ev-framework}}).

### 6.3 Multiple comparisons

Test one hypothesis at $\alpha = 0.05$ and you have a 5% chance of a false
positive. Test 20 independent hypotheses and the probability of *at least one*
false positive is

$$
1 - (1 - 0.05)^{20} = 1 - 0.358 = 0.642
$$ (eq:familywise-error)

Nearly two thirds. With 100 tests it is 99.4%.

This is not a hypothetical concern in machine learning; it is the normal
condition. Every hyperparameter search is hundreds of comparisons. Every "we
tried these twelve architectures" is twelve. Every dashboard with fifty metrics
is fifty tests being run continuously.

The consequence is that **the best validation score in a large search is
systematically optimistic**. You selected for a combination of quality and luck,
and the luck does not transfer to new data. This is precisely why a separate
test set, untouched during search, is not bureaucratic caution but a
mathematical necessity ({{ch:mle-splits}}).

The standard corrections are Bonferroni — test at $\alpha/m$ for $m$ tests,
conservative but simple — and Benjamini-Hochberg, which controls the false
discovery rate and is less conservative.

## 7. Implementation

```python {tier=A name=inference-simulation}
"""Statistical inference by simulation — every claim in the chapter checked by
drawing many samples rather than trusting the formula.
"""
import numpy as np

rng = np.random.default_rng(0)

# --- the sampling distribution, made visible --------------------------------
TRUE_ACC = 0.94
print("Drawing 20,000 test sets of 1,000 examples from a model whose TRUE")
print(f"accuracy is exactly {TRUE_ACC}:\n")

for n in (100, 1_000, 10_000):
    samples = rng.binomial(n, TRUE_ACC, size=20_000) / n
    se_predicted = np.sqrt(TRUE_ACC * (1 - TRUE_ACC) / n)
    print(f"  n = {n:>6}: observed accuracies range "
          f"{samples.min():.3f} to {samples.max():.3f}, "
          f"sd {samples.std():.4f} (predicted {se_predicted:.4f})")

print("\nA single measured accuracy is one draw from that spread.")

# --- eq. 10.6: the CLT works regardless of the underlying distribution ------
print(f"\n{'source distribution':<26} {'skew of X':>10} "
      f"{'skew of mean(n=200)':>21}")
sources = {
    "uniform":     lambda k: rng.uniform(0, 1, k),
    "exponential": lambda k: rng.exponential(1.0, k),
    "bernoulli":   lambda k: (rng.random(k) < 0.1).astype(float),
    "bimodal":     lambda k: np.where(rng.random(k) < 0.5,
                                      rng.normal(-3, 0.4, k),
                                      rng.normal(3, 0.4, k)),
}


def skew(v):
    z = (v - v.mean()) / v.std()
    return float((z ** 3).mean())


for name, draw in sources.items():
    raw = draw(200_000)
    means = np.array([draw(200).mean() for _ in range(4000)])
    print(f"{name:<26} {skew(raw):>10.3f} {skew(means):>21.3f}")
print("Whatever the source, the distribution of the MEAN is nearly symmetric.")

# --- eq. 10.9: do confidence intervals actually cover 95% of the time? ------
print("\nCoverage check: build 20,000 intervals and count how many contain")
print("the true value.\n")
for n in (30, 100, 1000):
    covered = 0
    trials = 20_000
    for _ in range(trials):
        s = rng.binomial(n, TRUE_ACC) / n
        se = np.sqrt(max(s * (1 - s), 1e-12) / n)
        if s - 1.96 * se <= TRUE_ACC <= s + 1.96 * se:
            covered += 1
    print(f"  n = {n:>4}: {covered/trials:.1%} of 95% intervals covered the truth")
print("At small n the normal approximation under-covers — the interval is")
print("too narrow, and the nominal 95% is a lie.")

# --- section 6.2: comparing two models --------------------------------------
n_test = 1000
acc_a, acc_b = 0.942, 0.938
se_a = np.sqrt(acc_a * (1 - acc_a) / n_test)
se_b = np.sqrt(acc_b * (1 - acc_b) / n_test)
se_diff = np.sqrt(se_a**2 + se_b**2)
z = (acc_a - acc_b) / se_diff

print(f"\nmodel A: {acc_a:.3f} +/- {1.96*se_a:.4f}")
print(f"model B: {acc_b:.3f} +/- {1.96*se_b:.4f}")
print(f"difference {acc_a-acc_b:.4f}, SE of difference {se_diff:.4f}, "
      f"z = {z:.2f}")
print(f"|z| < 1.96, so the difference is indistinguishable from noise.")


# --- eq. 10.13: how much data would settle it? ------------------------------
def required_n(delta, p_bar, alpha_z=1.96, beta_z=0.84):
    return 2 * (alpha_z + beta_z) ** 2 * p_bar * (1 - p_bar) / delta ** 2


print(f"\n{'effect to detect':>18} {'n per group':>14}")
for delta in (0.004, 0.01, 0.02, 0.05):
    print(f"{delta:>17.1%} {required_n(delta, 0.94):>14,.0f}")
print("Halving the effect quadruples the sample — the delta^2 in eq. 10.13.")

# --- eq. 10.14: multiple comparisons ----------------------------------------
print(f"\n{'tests':>7} {'P(>=1 false positive)':>24} {'simulated':>11}")
for m in (1, 5, 20, 100):
    analytic = 1 - 0.95 ** m
    sims = rng.random((20_000, m)) < 0.05        # each test, under a true null
    simulated = sims.any(axis=1).mean()
    print(f"{m:>7} {analytic:>23.1%} {simulated:>11.1%}")

# The selection effect: the winner of a large search is optimistic.
print("\nSelection bias in hyperparameter search.")
print("50 configurations, ALL with identical true accuracy of 0.90:\n")
for n_val in (200, 2000):
    winners_val, winners_test = [], []
    for _ in range(2000):
        val = rng.binomial(n_val, 0.90, size=50) / n_val
        best = int(np.argmax(val))
        winners_val.append(val[best])
        # The winner re-measured on fresh data of the same size.
        winners_test.append(rng.binomial(n_val, 0.90) / n_val)
    print(f"  validation set n = {n_val}:")
    print(f"    winner's validation score : {np.mean(winners_val):.4f}")
    print(f"    same model on fresh data  : {np.mean(winners_test):.4f}")
    print(f"    optimism                  : "
          f"{np.mean(winners_val) - np.mean(winners_test):+.4f}")
print("\nEvery configuration was equally good. The gap is pure selection")
print("bias — which is exactly why a held-out test set is not optional.")
```

## 8. Practical Example

Deciding whether a model improvement is real is the most common inferential
question in this field, and it is worth doing properly once.

```python {tier=A name=paired-model-comparison}
"""Comparing two models on a shared test set — the right way.

Because both models see the same examples, their errors are correlated. A
paired test exploits that and is far more sensitive than treating the two
accuracy figures as independent.
"""
import numpy as np

rng = np.random.default_rng(1)

n = 1000
# Per-example difficulty, shared by both models — this is what correlates them.
difficulty = rng.random(n)
# Model B is genuinely better, by 1.5 percentage points.
correct_a = rng.random(n) < (0.94 - 0.10 * difficulty)
correct_b = rng.random(n) < (0.955 - 0.10 * difficulty)

acc_a, acc_b = correct_a.mean(), correct_b.mean()
print(f"model A accuracy: {acc_a:.4f}")
print(f"model B accuracy: {acc_b:.4f}")
print(f"observed difference: {acc_b - acc_a:+.4f}")

# --- the naive, unpaired analysis -------------------------------------------
se_a = np.sqrt(acc_a * (1 - acc_a) / n)
se_b = np.sqrt(acc_b * (1 - acc_b) / n)
z_unpaired = (acc_b - acc_a) / np.sqrt(se_a**2 + se_b**2)
print(f"\nunpaired z = {z_unpaired:.3f}  -> "
      f"{'significant' if abs(z_unpaired) > 1.96 else 'NOT significant'}")

# --- the paired analysis: McNemar's test -------------------------------------
# Only the disagreements carry information. Examples both got right, or both
# got wrong, tell you nothing about which model is better.
b_only = int(np.sum(~correct_a & correct_b))     # B right, A wrong
a_only = int(np.sum(correct_a & ~correct_b))     # A right, B wrong
both_right = int(np.sum(correct_a & correct_b))
both_wrong = int(np.sum(~correct_a & ~correct_b))

print(f"\ncontingency table:")
print(f"  both right : {both_right:>4}      (uninformative)")
print(f"  both wrong : {both_wrong:>4}      (uninformative)")
print(f"  B only     : {b_only:>4}      <- evidence for B")
print(f"  A only     : {a_only:>4}      <- evidence for A")

# Under H0 the disagreements split 50/50, so the count is Binomial(m, 0.5).
m = a_only + b_only
z_paired = (b_only - a_only) / np.sqrt(m) if m else 0.0
print(f"\npaired z = {z_paired:.3f}  -> "
      f"{'significant' if abs(z_paired) > 1.96 else 'NOT significant'}")
print(f"the paired test uses only the {m} disagreements, not all {n} examples,")
print("and is more sensitive precisely because it removes the shared difficulty.")

# --- how often does each test find a real effect? (power) -------------------
def trial(true_a=0.940, true_b=0.955, n=1000):
    diff = rng.random(n)
    ca = rng.random(n) < (true_a + 0.05 - 0.10 * diff)
    cb = rng.random(n) < (true_b + 0.05 - 0.10 * diff)
    pa, pb = ca.mean(), cb.mean()
    sa = np.sqrt(max(pa*(1-pa), 1e-12) / n)
    sb = np.sqrt(max(pb*(1-pb), 1e-12) / n)
    zu = (pb - pa) / np.sqrt(sa**2 + sb**2)
    ao = int(np.sum(ca & ~cb)); bo = int(np.sum(~ca & cb)); mm = ao + bo
    zp = (bo - ao) / np.sqrt(mm) if mm else 0.0
    return abs(zu) > 1.96, abs(zp) > 1.96


results = np.array([trial() for _ in range(3000)])
print(f"\npower over 3000 simulated experiments (the effect IS real):")
print(f"  unpaired test detects it: {results[:,0].mean():.1%}")
print(f"  paired test detects it  : {results[:,1].mean():.1%}")
print("\nSame data, same effect. The paired test finds it far more often.")
```

## 9. Common Mistakes

**Reporting a metric without an interval.** "94.2% accuracy" alone is not a
result. On 1,000 examples it means $94.2\% \pm 1.5\%$.

**Interpreting a p-value as $\Prob(H_0 \given \text{data})$.** It is the
conditional in the other direction. This is the single most common error in
applied statistics.

**Interpreting a confidence interval as a probability statement about the
parameter.** It is a statement about the procedure across repetitions.

**Concluding "no effect" from a non-significant result.** Absence of evidence is
not evidence of absence, particularly at low power. Report the interval; if it
is wide, say so.

**Ignoring multiple comparisons.** Twenty tests at $\alpha = 0.05$ give a 64%
chance of at least one false positive. Hyperparameter search is hundreds of
tests.

**Reporting the best validation score as an unbiased estimate.** It is
systematically optimistic, by an amount the simulation in
{{sec:7-implementation}} measures.

**Treating correlated observations as independent.** Multiple rows per user,
multiple examples per document, autocorrelated time series — all inflate the
effective standard error above {{eq:standard-error}}, sometimes several-fold.

**Using an unpaired test on a shared test set.** Wasteful. The paired test is
strictly more powerful, as {{sec:8-practical-example}} demonstrates.

**Peeking at results and stopping when significant.** Repeatedly testing as data
accumulates inflates the false positive rate dramatically. Fix the sample size
in advance, or use a sequential test designed for it
({{ch:ds-experiments}}).

## 10. Connection to Previous Chapters

{{ch:math-covariance}} supplied {{eq:variance-sum-independent}}, which is the
one ingredient in the derivation of {{eq:standard-error}} — and the place where
the independence assumption enters, which is why correlated data breaks
inference. {{ch:math-random-vars}} supplied the Gaussian that the CLT converges
to and the Bernoulli variance $p(1-p)$ used for proportions.
{{ch:math-probability}} supplied conditioning, and the p-value fallacy is
exactly its $\Prob(A \given B)$ versus $\Prob(B \given A)$ confusion.

Forward: {{ch:math-optimization}} uses {{eq:mse-decomposition}} to justify
regularisation as a deliberate bias-for-variance trade.

Beyond Part I: {{ch:ds-experiments}} builds A/B testing on this chapter;
{{ch:mle-splits}} explains why the selection bias measured in
{{sec:7-implementation}} makes a held-out test set mandatory;
{{ch:ml-metrics}} attaches intervals to every metric; and
{{ch:ev-framework}} builds an evaluation harness that reports uncertainty by
default.

## 11. Exercises

**Beginner**

1. A sample of 100 has mean 50 and standard deviation 10. Compute the standard
   error of the mean.
2. A model scores 90% on 400 test examples. Give a 95% confidence interval.
3. How many examples are needed to estimate an accuracy to within ±1
   percentage point at 95% confidence, assuming accuracy near 90%?
4. State in one sentence what a p-value of 0.03 means.
5. Distinguish a Type I from a Type II error.

**Intermediate**

6. Derive {{eq:variance-of-mean}}, stating exactly where independence is used.
7. A test set of 500 gives 88% accuracy. Compute the interval. How much larger
   must the test set be to halve the interval width?
8. Twenty-five hyperparameter configurations are compared at $\alpha = 0.05$.
   What is the probability of at least one false positive? What Bonferroni-
   corrected level restores 5% overall?
9. Explain why the paired test in {{sec:8-practical-example}} is more powerful.
10. An experiment has 30% power. It reaches significance. What should you
    conclude about the effect size?
11. Two models are compared on 2,000 examples: 91.0% versus 92.5%. Is the
    difference significant at 5%? Show the calculation.

**Advanced**

12. Prove {{eq:mse-decomposition}}.
13. Show that dividing by $n$ rather than $n-1$ gives a biased variance
    estimator, and compute the bias.
14. Derive {{eq:sample-size}} from the requirement that the test statistic
    exceed $z_{\alpha/2}$ with probability $1-\beta$.
15. Observations are equicorrelated with correlation $\rho$ within clusters of
    size $k$. Derive the design effect — the factor by which the variance of the
    mean is inflated — and evaluate it for $\rho = 0.1$, $k = 10$.
16. Explain why the CLT fails to help for a Cauchy-distributed quantity, and
    what the sample mean does instead.

**Implementation**

17. Verify the CLT by sampling from three distributions of your choice and
    plotting the distribution of the mean at $n \in \{1, 5, 30, 200\}$.
18. Empirically measure the coverage of {{eq:proportion-ci}} for $p = 0.5$ and
    $p = 0.02$ at $n = 50$. Explain the difference, and look up the Wilson
    interval as a fix.
19. Simulate a peeking experiment: test for significance after every 50 new
    observations and stop when $p < 0.05$. Measure the true false-positive rate
    under a null effect.
20. Implement a bootstrap confidence interval for a median — a statistic with no
    simple closed-form standard error — and compare it with the normal interval
    for a mean on the same data.

**Reasoning**

21. A leaderboard shows models within 0.3 percentage points of each other on a
    1,000-example benchmark. What can you legitimately conclude?
22. Your monitoring dashboard tracks 40 metrics and alerts at $p < 0.05$. How
    many false alerts per day should you expect, and what would you change?

## 12. Chapter Summary

A parameter is a fixed property of the population; a statistic is computed from
a sample and is therefore a random variable. Inference is the business of saying
something about the first using only the second.

The sampling distribution — how a statistic varies across the samples you might
have drawn — is the central object. The law of large numbers says the sample
mean converges to the truth; the central limit theorem says how fast, and that
the sampling distribution of a mean is approximately Gaussian regardless of the
underlying distribution, provided the variance is finite and the observations
are independent.

The standard error is $\sigma/\sqrt{n}$. Halving uncertainty requires
quadrupling the data, which means a 1,000-example benchmark cannot reliably
resolve differences below about three percentage points.

A confidence interval is a statement about the procedure across repetitions, not
a probability statement about the parameter. A p-value is
$\Prob(\text{data} \given H_0)$, not $\Prob(H_0 \given \text{data})$; reading it
the other way is the most consequential error in applied statistics.

Power is the probability of detecting a real effect. Required sample size scales
as $1/\delta^{2}$, so detecting small effects is expensive. Underpowered
experiments are worse than none: they miss most real effects and inflate the
magnitude of the ones they do detect.

Multiple comparisons inflate false positives fast — twenty tests give a 64%
chance of at least one. Hyperparameter search is hundreds of comparisons, so the
best validation score is systematically optimistic, which is precisely why an
untouched test set is a mathematical necessity rather than a procedural nicety.

When comparing two models on a shared test set, use a paired test. It exploits
the correlation between their errors and is substantially more sensitive than
treating the two scores as independent.

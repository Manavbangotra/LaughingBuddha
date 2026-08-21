---
id: ds-eda
number: 24
part: III
tier: focused
status: reviewed
requires: [ds-cleaning, py-visualization]
provides: [exploratory-data-analysis]
citations: [automind2025]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. State what exploratory analysis is for, and how it differs from reporting.
2. Work through a systematic EDA rather than an ad-hoc one.
3. Read a univariate distribution and identify skew, multimodality and
   truncation.
4. Investigate relationships between variables of every type combination.
5. Explain why aggregates hide structure, and demonstrate it.
6. Detect data problems that survived cleaning.
7. Explain what automated EDA tools do well and what they systematically miss.

## 2. Why This Matters

Exploratory data analysis is the stage where you find out whether the dataset
supports the question you intend to ask. It is cheap, it is fast, and skipping
it is how a project spends three weeks modelling something that was never going
to work.

The specific things EDA catches, that nothing downstream will:

**Assumption violations.** The target is bimodal, so a single regression is
fitting the gap between two populations. The relationship is nonlinear, so a
linear model will be systematically wrong ({{ch:py-visualization}}).

**Residual data problems.** A spike at exactly zero that turns out to be a
sentinel. A date range that stops abruptly. A category that appears only after
March.

**The structure that aggregates hide.** {{ch:py-visualization}} made the point
with Anscombe's quartet; this chapter makes it with real analytical questions,
where a pooled correlation can have the opposite sign to every subgroup's
({{ch:ds-causation}}).

There is also a discipline point. EDA is where confirmation bias does the most
damage, because you are looking at data with a hypothesis in mind and the space
of possible views is enormous. A systematic pass — the same questions every
time, in the same order — is protection against seeing only what you came for.

## 3. Prerequisites

{{ch:ds-cleaning}} for the cleaning that precedes exploration;
{{ch:py-visualization}} for the plotting; {{ch:math-covariance}} for correlation
and its limits; {{ch:math-inference}} for the fact that a difference between two
groups may be noise.

## 4. Intuitive Explanation

### 4.1 Exploration is not reporting

The two activities have opposite goals and produce opposite artefacts.

{#tbl:eda-vs-reporting caption="Exploratory analysis and reporting are different activities with different standards. Confusing them produces either unreadable reports or unexplored data."}

| | Exploration | Reporting |
|---|---|---|
| Audience | you | someone else |
| Goal | find out what is there | communicate one thing |
| Number of views | dozens | one or two |
| Polish | none | high |
| Bias risk | confirmation | oversimplification |
| Output | a list of findings and concerns | a decision-supporting figure |

A common failure is doing reporting-quality work during exploration — polishing
the third plot when you should have made thirty. Another is publishing an
exploratory plot, with its unlabelled axes and default colours, to an audience
who will misread it.

### 4.2 A systematic pass

Ad-hoc exploration finds what you expected. A fixed sequence finds what is
there. The order that works:

```text
1. shape        rows, columns, types, memory
2. quality      nulls, duplicates, constants, cardinality
3. univariate   one variable at a time: distribution, range, outliers
4. target       the outcome's distribution; class balance
5. bivariate    each feature against the target
6. interactions features against each other; redundancy
7. temporal     does anything change over time?
8. segments     does the story hold within subgroups?
```

Step 8 is the one most often skipped and the one that most often changes the
conclusion ({{ch:ds-causation}}).

### 4.3 What a distribution tells you

Reading a single histogram well answers several questions at once:

- **Skew.** A long right tail means the mean exceeds the median, and any method
  assuming symmetry will misbehave. Revenue, session length and file size are
  all right-skewed.
- **Multimodality.** Two humps almost always mean two populations mixed
  together. Fitting one model to both fits neither.
- **Spikes.** A spike at a round number is usually a default, a sentinel, or a
  cap. A spike at zero may be a genuine zero or a missing value that was filled.
- **Truncation.** A hard edge means a filter was applied upstream, and you are
  looking at a selected sample ({{ch:ds-collection}}).
- **Granularity.** Values only at multiples of 5 mean rounding, and any
  precision below that is illusory.

### 4.4 Aggregates hide structure

The single most useful habit in EDA is to **disaggregate before believing an
aggregate**.

An overall conversion rate of 4% might be 8% on desktop and 1% on mobile. An
overall flat trend might be two segments moving in opposite directions. A
correlation of zero overall might be strongly positive in every subgroup with
the groups offset from each other.

The last case is not a curiosity. It is Simpson's paradox
({{ch:ds-causation}}), and it means an aggregate can point in the opposite
direction to every part of the data it summarises.

## 5. Formal Explanation

### 5.1 Univariate summaries and what they miss

The five-number summary — minimum, $Q_1$, median, $Q_3$, maximum — plus the
mean and standard deviation describes a distribution compactly and incompletely.

Useful derived quantities:

$$
\text{skewness} = \E\!\left[\left(\frac{X-\mu}{\sigma}\right)^{3}\right],
\qquad
\text{excess kurtosis} = \E\!\left[\left(\frac{X-\mu}{\sigma}\right)^{4}\right] - 3
$$ (eq:skew-kurtosis)

Positive skew means a longer right tail; positive excess kurtosis means heavier
tails than a normal distribution. A quick diagnostic that costs nothing:

$$
\frac{\text{mean} - \text{median}}{\sigma}
$$ (eq:skew-heuristic)

Substantially positive indicates right skew. If mean and median differ
appreciably, reporting the mean alone is misleading.

> IMPORTANT: No set of moments determines the distribution. Anscombe's quartet
> ({{ch:py-visualization}}) shares two moments and a correlation across four
> unlike datasets, and it is possible to construct datasets agreeing on many
> more. Summaries are a supplement to looking, not a substitute.

### 5.2 Choosing a bivariate view

{#tbl:bivariate-views caption="How to look at a relationship, by the types involved."}

| Feature | Target | View |
|---|---|---|
| numeric | numeric | scatter (+ smoother); correlation |
| numeric | categorical | box or violin per class; histogram overlay |
| categorical | numeric | box per category; mean with intervals |
| categorical | categorical | contingency table; mosaic; rate per cell |
| numeric | binary | rate by decile of the feature |

The last row is the most useful for classification work and is under-used.
Binning the feature into deciles and plotting the positive rate per bin reveals
monotonicity, nonlinearity and thresholds in one view, and works no matter how
skewed the feature is.

### 5.3 Redundancy and multicollinearity

Two features carrying the same information cause instability in linear models
({{ch:math-eigen}}) and split importance arbitrarily in tree models.

Detect with a correlation matrix for linear redundancy, and with the condition
number of the standardised feature covariance matrix for the general case:

$$
\kappa(\hat{\mat{\Sigma}}) = \frac{\lambda_{\max}}{\lambda_{\min}}
$$ (eq:feature-condition-number)

A condition number above roughly $10^{3}$ indicates near-collinearity, and
coefficient estimates become unstable in a way that {{ch:math-eigen}}
quantifies exactly.

Correlation only catches *pairwise* linear redundancy. A feature that is a
linear combination of three others may correlate weakly with each while being
perfectly predictable from the set — which is why the condition number is worth
computing alongside.

### 5.4 Is this difference real?

EDA generates differences constantly, and most of them are noise. Two habits
prevent the resulting wild-goose chases.

**Attach an interval to every group comparison.** {{ch:math-inference}} gives
the arithmetic; a difference smaller than the intervals is not a finding.

**Count how many comparisons you have made.** Twenty group comparisons at the
5% level produce at least one spurious "significant" result 64% of the time
({{eq:familywise-error}}). Exploration is inherently a multiple-comparisons
exercise, and the correct response is not to correct the p-values — it is to
treat every EDA finding as a *hypothesis to be tested on fresh data*, not as a
result.

> IMPORTANT: This is the honest reason to hold out a test set before exploring.
> Anything discovered during exploration has been selected for by looking, and
> its effect size is biased upward by the same selection mechanism measured in
> {{ch:math-inference}}.

### 5.5 Recording what you find

Exploration generates dozens of small observations and forgets most of them. The
ones that are lost are rediscovered later, expensively, usually by someone
reading a wrong number in a report.

A findings log costs almost nothing and should record, per item: what was
observed, which column, the severity, whether it is resolved, and what was
decided. Three categories are worth separating.

**Blockers** — the analysis cannot proceed until resolved. A target that is
mostly missing; a join key that is not unique; a date range that does not cover
the question.

**Caveats** — the analysis proceeds and the finding constrains the conclusion. A
segment under-represented; a column reliable only after a certain date; a metric
that changed definition mid-period.

**Follow-ups** — worth investigating and not now.

The caveats matter most, because they are the ones that must travel with the
result. A conclusion reported without them will be applied to cases it does not
cover, and the person doing so will not know they are doing it.

> PRODUCTION TIP: Write the caveats into the same artefact as the numbers, not
> into a separate document. A caveat in a different file has, in practice,
> already been lost.

### 5.6 Automated EDA

Tools that generate a full profile — every distribution, every correlation,
every missingness pattern — are genuinely useful and now largely subsume step 1
through step 3 of {{sec:4-intuitive-explanation}}. Agent frameworks extend this
to proposing and running the whole analysis {{cite:automind2025}}.
{{maturity:EMERGING}}

What they do well: exhaustive coverage without fatigue, consistent formatting,
and never forgetting to check something mechanical.

What they systematically miss:

- **That a column should not exist** — because it is populated after the
  outcome ({{ch:ds-leakage}}).
- **That a spike at 1970-01-01 is an epoch default**, not a real date.
- **That two categories are the same thing spelled differently** in a way that
  requires domain knowledge to see.
- **Which of forty findings matters.** A profile reports everything with equal
  emphasis, and prioritisation is the actual work.
- **What is absent.** No tool reports the segment that should be in the data and
  is not.

The division is the one from {{ch:ds-what-it-is}}: mechanical coverage is
automated, and judgement about what the coverage means is not.

## 6. Mathematical Foundation

### 6.1 Why the mean and median diverge

For a right-skewed distribution the mean exceeds the median, and the gap
measures the skew. For the lognormal — a good model of revenue, income and
session length — with underlying normal parameters $\mu$ and $\sigma$:

$$
\text{median} = e^{\mu},
\qquad
\text{mean} = e^{\mu + \sigma^{2}/2}
$$ (eq:lognormal-moments)

so the ratio is

$$
\frac{\text{mean}}{\text{median}} = e^{\sigma^{2}/2}
$$ (eq:mean-median-ratio)

At $\sigma = 1$ the mean is $1.65\times$ the median; at $\sigma = 1.5$ it is
$3.08\times$. Reporting "average revenue per user" on such a distribution
describes a customer who does not exist, and a small number of large accounts
determine the figure.

This is why revenue is reported as a median or a distribution, and why
"average" is the wrong summary for almost any quantity that is bounded below by
zero and unbounded above.

### 6.2 How much a histogram's bin width matters

A histogram is a density estimate, and the bin width is a bias-variance
trade-off ({{ch:math-inference}}). Too narrow and you see noise; too wide and
you smooth away real structure such as bimodality.

Two standard rules:

$$
\text{Sturges}: \quad k = \lceil \log_2 n \rceil + 1
$$ (eq:sturges)

$$
\text{Freedman-Diaconis}: \quad h = 2\,\frac{\text{IQR}}{n^{1/3}}
$$ (eq:freedman-diaconis)

Freedman-Diaconis is preferable in practice because it uses the IQR and is
therefore robust to outliers ({{ch:ds-cleaning}}), while Sturges assumes
approximate normality and produces too few bins for large $n$.

The practical advice is simply to try several. A feature that appears unimodal
at 10 bins and bimodal at 50 is telling you something, and which view is
"correct" is not a question the data answers by itself.

### 6.3 Correlation is not the only dependence

{{ch:math-covariance}} showed that $Y = X^{2}$ has zero correlation. Two
measures catch what correlation misses.

**Spearman's rank correlation** applies Pearson's formula to the ranks, and
therefore detects any *monotonic* relationship, linear or not. It is also robust
to outliers, because a single extreme value affects its rank by at most $n-1$
positions rather than affecting the mean without bound.

**Mutual information** detects any dependence at all:

$$
I(X; Y) = \sum_{x,y} p(x,y)\log\frac{p(x,y)}{p(x)p(y)}
$$ (eq:mutual-information)

$I(X;Y) = 0$ if and only if $X$ and $Y$ are independent — the exact condition
that correlation fails to characterise. The price is that it must be estimated,
usually by binning, which introduces its own bias and requires more data.

A practical screening strategy is to compute all three: large Pearson means
linear, large Spearman with small Pearson means monotonic-but-curved, and large
mutual information with both small means non-monotonic — the case a correlation
screen discards entirely ({{ch:ds-feature-eng}}).

## 7. Implementation

```python {tier=A name=systematic-eda}
"""A systematic EDA pass, and the structure aggregates hide.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
pd.set_option("display.width", 110)

# --- a dataset with several things worth finding ----------------------------
n = 20_000
device = rng.choice(["desktop", "mobile"], n, p=[0.35, 0.65])
# Two populations with genuinely different behaviour.
session = np.where(device == "desktop",
                   rng.lognormal(5.6, 0.5, n),
                   rng.lognormal(4.2, 0.6, n))
spend = np.where(device == "desktop",
                 rng.gamma(3, 40, n), rng.gamma(2, 12, n))
# A sentinel that survived cleaning, and a capped column.
age = rng.integers(18, 80, n).astype(float)
age[rng.random(n) < 0.03] = 0                       # 0 used as "unknown"
rating = np.clip(rng.normal(3.9, 1.1, n).round(), 1, 5)

df = pd.DataFrame({"device": device, "session_s": session.round(1),
                   "spend": spend.round(2), "age": age, "rating": rating})
df["converted"] = (rng.random(n) <
                   np.where(device == "desktop", 0.08, 0.01)).astype(int)

# --- step 1-2: shape and quality --------------------------------------------
print("=" * 72)
print("1-2. shape and quality")
print("=" * 72)
print(f"rows {len(df):,}  columns {df.shape[1]}  "
      f"memory {df.memory_usage(deep=True).sum()/1e6:.1f} MB")
print(f"\n{'column':<12} {'dtype':<10} {'null%':>7} {'nunique':>9} "
      f"{'constant?':>10}")
for c in df.columns:
    s = df[c]
    print(f"{c:<12} {str(s.dtype):<10} {s.isna().mean():>6.1%} "
          f"{s.nunique():>9,} {str(s.nunique() <= 1):>10}")

# --- step 3: univariate, with the diagnostics of section 6 ------------------
print("\n" + "=" * 72)
print("3. univariate: skew, spikes, granularity")
print("=" * 72)
print(f"{'column':<12} {'mean':>10} {'median':>10} {'mean/med':>9} "
      f"{'skew':>8} {'top value share':>17}")
for c in ["session_s", "spend", "age", "rating"]:
    s = df[c]
    z = (s - s.mean()) / s.std()
    skew = (z ** 3).mean()
    top = s.value_counts(normalize=True).iloc[0]
    top_val = s.value_counts().index[0]
    print(f"{c:<12} {s.mean():>10.2f} {s.median():>10.2f} "
          f"{s.mean()/s.median():>9.2f} {skew:>8.2f} "
          f"{top:>10.1%} at {top_val:g}")

print("\nfindings:")
print("  session_s and spend are right-skewed (mean/median > 1): report")
print("    medians, and expect the mean to be driven by a few large values.")
print(f"  age has {(df['age'] == 0).mean():.1%} of values at exactly 0 —")
print("    impossible for an age, and a sentinel that survived cleaning.")
print("  rating takes only integer values 1-5: it is ordinal, not continuous.")

# eq. 24.4: how far the mean/median ratio goes on lognormal data
print(f"\n{'sigma':>7} {'mean/median (eq. 24.4)':>24} {'simulated':>11}")
for sigma in (0.5, 1.0, 1.5, 2.0):
    sample = rng.lognormal(0, sigma, 400_000)
    print(f"{sigma:>7.1f} {np.exp(sigma**2/2):>24.2f} "
          f"{sample.mean()/np.median(sample):>11.2f}")

# --- step 5-6: bivariate, and the decile view --------------------------------
print("\n" + "=" * 72)
print("5. numeric feature vs binary target: the decile view")
print("=" * 72)
df["spend_decile"] = pd.qcut(df["spend"], 10, labels=False, duplicates="drop")
by_decile = df.groupby("spend_decile").agg(
    n=("converted", "size"), rate=("converted", "mean"),
    spend_lo=("spend", "min"), spend_hi=("spend", "max"))
by_decile["se"] = np.sqrt(by_decile["rate"] * (1 - by_decile["rate"])
                          / by_decile["n"])
print(f"{'decile':>7} {'spend range':>20} {'n':>7} {'conv rate':>11} "
      f"{'95% CI':>18}")
for i, r in by_decile.iterrows():
    lo, hi = r["rate"] - 1.96*r["se"], r["rate"] + 1.96*r["se"]
    print(f"{i:>7} {f'{r.spend_lo:>7.0f}-{r.spend_hi:<7.0f}':>20} "
          f"{int(r.n):>7,} {r['rate']:>10.2%} "
          f"{f'[{lo:.2%}, {hi:.2%}]':>18}")
print("\nThe rate rises with spend — but the intervals overlap heavily between")
print("adjacent deciles, so only the overall trend is supported, not any")
print("individual step (Chapter 10).")

# --- step 8: disaggregate before believing the aggregate --------------------
print("\n" + "=" * 72)
print("8. segments: the aggregate hides the mechanism")
print("=" * 72)
overall = np.corrcoef(df["session_s"], df["converted"])[0, 1]
print(f"overall corr(session_s, converted) = {overall:+.4f}")
print(f"\n{'device':<10} {'n':>8} {'mean session':>14} {'conv rate':>11} "
      f"{'within-group corr':>19}")
for dev, g in df.groupby("device"):
    r = np.corrcoef(g["session_s"], g["converted"])[0, 1]
    print(f"{dev:<10} {len(g):>8,} {g['session_s'].mean():>14.1f} "
          f"{g['converted'].mean():>10.2%} {r:>19.4f}")

print("\nThe overall correlation is driven almost entirely by device: desktop")
print("users have both longer sessions and higher conversion. Within each")
print("device the association is far weaker. Concluding 'longer sessions")
print("cause conversion' from the pooled number would be wrong (Chapter 25).")

# --- eq. 24.7: correlation vs rank correlation vs mutual information --------
print("\n" + "=" * 72)
print("what correlation misses")
print("=" * 72)

m = 20_000
x = rng.uniform(-3, 3, m)
relationships = {
    "linear":        3 * x + rng.normal(0, 1, m),
    "monotonic":     x ** 11 + rng.normal(0, 1, m),   # strictly increasing
    "quadratic (U)": x ** 2 + rng.normal(0, 1, m),
    "independent":   rng.normal(0, 1, m),
}


def spearman(a, b):
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    return np.corrcoef(ra, rb)[0, 1]


def mutual_info(a, b, bins=24):
    joint, _, _ = np.histogram2d(a, b, bins=bins)
    p = joint / joint.sum()
    px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log(p[nz] / (px @ py)[nz])).sum())


# The diagnostic is the RELATIONSHIP between the three numbers, not any
# absolute threshold: Spearman exceeding Pearson means monotonic-but-curved,
# and mutual information without either means non-monotonic dependence.
print(f"{'relationship':<16} {'Pearson':>9} {'Spearman':>10} "
      f"{'MI':>7} {'S-P gap':>9}  {'verdict'}")
for name, y in relationships.items():
    pe, sp, mi = np.corrcoef(x, y)[0, 1], spearman(x, y), mutual_info(x, y)
    gap = abs(sp) - abs(pe)
    if mi < 0.05:
        verdict = "genuinely independent"
    elif abs(pe) < 0.15 and abs(sp) < 0.15:
        verdict = "dependent but NON-MONOTONIC — both correlations blind"
    elif gap > 0.08:
        verdict = "monotonic but curved — Spearman beats Pearson"
    else:
        verdict = "linear — Pearson is adequate"
    print(f"{name:<16} {pe:>+9.3f} {sp:>+10.3f} {mi:>7.3f} {gap:>+9.3f}  "
          f"{verdict}")

print("\nThe quadratic row is the one that matters: Pearson and Spearman both")
print("report near zero, and mutual information detects it. A correlation")
print("screen would discard this feature entirely (Chapter 27).")
```

## 8. Practical Example

The reusable artefact from this chapter is a profiling function that runs the
systematic pass and — crucially — *ranks* what it finds, which is the step
automated profilers omit.

```python {tier=A name=eda-profile}
"""An EDA profiler that prioritises its findings.

Generating every statistic is easy and is what automated tools do. Deciding
which of forty observations deserves attention is the part that is still
manual, so this profiler assigns a severity and sorts.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)


def profile(df: pd.DataFrame, target: str | None = None,
            time_col: str | None = None) -> pd.DataFrame:
    """Return findings ordered by severity rather than by column."""
    findings = []

    def add(severity, column, finding, action):
        findings.append({"severity": severity, "column": column,
                         "finding": finding, "suggested action": action})

    n = len(df)
    if df.duplicated().sum():
        add(3, "—", f"{df.duplicated().sum():,} exact duplicate rows",
            "deduplicate; check ingestion idempotency")

    for c in df.columns:
        s = df[c]
        nunique, null_frac = s.nunique(dropna=True), s.isna().mean()

        if nunique <= 1:
            add(2, c, "constant column", "drop — carries no information")
            continue
        if null_frac > 0.5:
            add(2, c, f"{null_frac:.0%} missing",
                "consider dropping; check why it is absent")
        elif null_frac > 0:
            add(1, c, f"{null_frac:.1%} missing", "impute + indicator column")

        if pd.api.types.is_numeric_dtype(s):
            v = s.dropna()
            if len(v) < 10:
                continue
            # Impossible or sentinel-looking values.
            top_share = v.value_counts(normalize=True).iloc[0]
            top_val = v.value_counts().index[0]
            if top_share > 0.02 and top_val in (0, -1, -999, 999, 9999):
                add(3, c, f"{top_share:.1%} of values equal {top_val:g}",
                    "likely a sentinel — verify before treating as numeric")
            if v.median() != 0 and v.mean() / max(v.median(), 1e-9) > 1.5:
                add(1, c, f"right-skewed (mean/median "
                          f"{v.mean()/v.median():.1f})",
                    "report median; consider a log transform")
            if nunique < 15 and pd.api.types.is_integer_dtype(s):
                add(1, c, f"only {nunique} distinct integer values",
                    "probably ordinal or categorical, not continuous")
        else:
            if nunique > 0.5 * n:
                add(2, c, f"cardinality {nunique:,} of {n:,} rows",
                    "identifier-like — exclude, or use target encoding")
            # Categories that differ only by case or whitespace.
            norm = s.astype("string").str.strip().str.lower()
            if norm.nunique() < nunique:
                add(3, c, f"{nunique - norm.nunique()} categories collapse "
                          f"after trim/lowercase",
                    "normalise before encoding — these are the same value")

        if target and c != target and pd.api.types.is_numeric_dtype(s):
            paired = df[[c, target]].dropna()
            if len(paired) > 30 and paired[c].nunique() > 1:
                r = np.corrcoef(paired[c], paired[target])[0, 1]
                if abs(r) > 0.95:
                    add(3, c, f"correlation {r:+.3f} with the target",
                        "suspiciously high — check for leakage (Chapter 28)")

        if time_col and c != time_col and df[c].isna().any():
            monthly = df.set_index(time_col)[c].resample("MS").apply(
                lambda g: g.isna().mean()).dropna()
            if len(monthly) > 1 and monthly.max() - monthly.min() > 0.4:
                add(3, c, "null rate changes sharply over time",
                    "schema change or backfill — check provenance")

    out = pd.DataFrame(findings).sort_values(
        ["severity", "column"], ascending=[False, True])
    return out.reset_index(drop=True)


# --- a dataset with a planted problem of each severity ----------------------
n = 12_000
frame = pd.DataFrame({
    "ts": pd.to_datetime("2026-01-01") + pd.to_timedelta(
        rng.integers(0, 210, n), "D"),
    "user_id": rng.integers(1, 40_000, n),
    "country": rng.choice(["GB", "gb ", "US", "FR"], n, p=[.4, .1, .3, .2]),
    "revenue": rng.lognormal(3.2, 1.1, n).round(2),
    "age": np.where(rng.random(n) < 0.04, -999,
                    rng.integers(18, 80, n)).astype(float),
    "rating": rng.integers(1, 6, n),
    "region": "EMEA",                                     # constant
    "target": rng.integers(0, 2, n),
})
frame["score"] = frame["target"] * 100 + rng.normal(0, 1, n)   # leak
frame["nps"] = np.where(frame["ts"] < "2026-04-01", np.nan,
                        rng.integers(0, 11, n))                # backfilled
frame = pd.concat([frame, frame.iloc[:150]], ignore_index=True)  # duplicates

report = profile(frame, target="target", time_col="ts")

labels = {3: "HIGH", 2: "MEDIUM", 1: "LOW"}
print(f"{'sev':<7} {'column':<10} {'finding':<46} {'action'}")
print("-" * 118)
for _, r in report.iterrows():
    print(f"{labels[r.severity]:<7} {r['column']:<10} {r['finding']:<46} "
          f"{r['suggested action']}")

print(f"\n{len(report)} findings, {(report.severity == 3).sum()} high severity.")
print("\nThe ranking is the contribution. An automated profiler would report")
print("all of these with equal weight, plus fifty more distributions. Knowing")
print("that 'score correlates 0.999 with the target' outranks 'revenue is")
print("skewed' is judgement, and it is what decides whether the next hour is")
print("well spent.")
```

## 9. Common Mistakes

**Skipping EDA to get to modelling.** The cheapest stage, and the one that
prevents the most wasted work.

**Only looking at aggregates.** Disaggregate before believing anything.

**Reporting a mean for skewed data.** {{eq:mean-median-ratio}}: at $\sigma = 1.5$
the mean is triple the median and describes nobody.

**Using one bin width.** Structure appears and disappears with binning.

**Screening features by Pearson correlation.** Blind to non-monotonic
relationships, as {{sec:7-implementation}} demonstrates.

**Treating every difference as real.** Attach intervals; count comparisons.

**Exploring the test set.** Anything found there is no longer an honest
estimate.

**Confusing exploration with reporting.** Different standards, different
artefacts.

**Trusting an automated profile as a finished analysis.** It covers the
mechanical checks and cannot prioritise or notice what is absent.

**Not writing findings down as you go.** Exploration generates dozens of small
observations, and the ones you fail to record are the ones you rediscover
expensively later.

## 10. Connection to Previous Chapters

{{ch:ds-cleaning}} precedes this and is frequently revisited from it — EDA is
where surviving sentinels and mis-typed columns are found.
{{ch:py-visualization}} supplies the plots and the Anscombe argument that
motivates looking at all. {{ch:math-covariance}} supplies correlation and the
$Y = X^2$ counterexample that {{sec:6-mathematical-foundation}} extends with
Spearman and mutual information. {{ch:math-inference}} supplies the intervals
and the multiple-comparisons warning that makes EDA findings hypotheses rather
than results.

Forward: {{ch:ds-causation}} takes the disaggregation lesson of
{{sec:4-intuitive-explanation}} to its conclusion; {{ch:ds-feature-eng}} builds
features from what EDA reveals; {{ch:ds-leakage}} formalises the
suspiciously-high-correlation finding.

Beyond Part III: {{part:20}} returns to automated analysis with agent
capabilities available.

## 11. Exercises

**Beginner**

1. List the eight steps of a systematic EDA pass.
2. What does it mean if the mean is much larger than the median?
3. What might a spike at exactly zero indicate? Give three possibilities.
4. What view would you use for a categorical feature against a numeric target?
5. Why is a constant column worth flagging?

**Intermediate**

6. Using {{eq:mean-median-ratio}}, compute the mean/median ratio for a lognormal
   with $\sigma = 1.2$ and say what that implies for reporting.
7. A feature has a Pearson correlation of 0.02 with the target and a mutual
   information of 0.4. What does that mean, and what should you do?
8. Explain why exploring the test set invalidates it, referring to selection.
9. Plot one variable with 10, 30 and 100 bins. Describe what changes and how
   you would choose.
10. An aggregate conversion rate is 4%. What is the first thing you check?
11. Compute the condition number of a feature covariance matrix and interpret
    it.

**Advanced**

12. Derive {{eq:mean-median-ratio}} from the lognormal moments.
13. Explain why Freedman-Diaconis is more robust than Sturges, referring to the
    breakdown point ({{ch:ds-cleaning}}).
14. Show that mutual information is zero if and only if the variables are
    independent, and explain why correlation lacks this property.
15. Design an EDA protocol that is protected against confirmation bias. What
    would you fix in advance?

**Implementation**

16. Extend the profiler to detect a numeric column whose values are all
    multiples of some constant, and explain why that matters.
17. Add a check for pairs of columns with correlation above 0.98 and report them
    as redundant.
18. Implement the decile view as a reusable function returning rates with
    confidence intervals, and use it on a dataset of your own.
19. Compare your profiler's output against an off-the-shelf profiling library on
    the same data. What does each find that the other does not?

**Reasoning**

20. Automated tools now generate complete profiles in seconds. What is the
    remaining human contribution, specifically?
21. EDA is inherently a multiple-comparisons exercise. Does that mean its
    findings are worthless? What is the right epistemic status for them?

## 12. Chapter Summary

Exploratory analysis determines whether the data supports the question, and it
is the cheapest stage at which a project can be stopped or redirected. It is a
different activity from reporting: many rough views for yourself, rather than
one polished view for someone else.

A systematic pass — shape, quality, univariate, target, bivariate,
interactions, temporal, segments — finds what is there. An ad-hoc pass finds
what you expected. The segment step is the most often skipped and the most
likely to change the conclusion.

A single distribution reveals skew, multimodality, sentinel spikes, truncation
and granularity. For right-skewed data the mean exceeds the median by
$e^{\sigma^{2}/2}$, which at $\sigma = 1.5$ is over triple — so reporting an
average describes a customer who does not exist.

Aggregates hide structure. Disaggregating before believing an aggregate is the
single most valuable habit here, because a pooled statistic can point the
opposite way to every subgroup within it.

Correlation detects only linear association. Spearman extends this to monotonic
relationships and is outlier-robust; mutual information detects any dependence
at all and is zero exactly when the variables are independent. A correlation
screen silently discards every non-monotonic feature.

Every difference EDA turns up is a hypothesis, not a result: exploration is a
multiple-comparisons exercise by construction, and findings must be confirmed on
data that was not explored.

Automated profiling now covers the mechanical passes well and cannot prioritise,
cannot recognise a column that should not exist, and cannot notice what is
absent from the data entirely.

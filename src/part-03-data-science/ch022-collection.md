---
id: ds-collection
number: 22
part: III
tier: focused
status: reviewed
requires: [ds-what-it-is, py-io-apis-sql]
provides: [selection-bias, survivorship-bias, provenance, schema-drift,
           data-contract]
citations: [sculley2015]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Identify how a sample was selected and what that rules out.
2. Recognise survivorship bias and construct the population that is missing.
3. Choose a sampling scheme and state what each one buys.
4. Explain what provenance is, and record enough of it to trace a number back
   to its source.
5. Detect schema drift before it corrupts downstream numbers.
6. Specify a data contract with enforceable constraints.
7. Choose between batch and streaming ingestion, and design idempotent loads.

## 2. Why This Matters

Every conclusion you will draw is a conclusion about whoever ended up in the
dataset. If that group differs systematically from the group you care about, no
amount of modelling repairs it — the model will be an accurate description of
the wrong population.

This failure is not rare and it is not obvious. A dataset of customers excludes
everyone who did not become one. A dataset of loan repayments excludes everyone
who was refused. A survey of app users excludes everyone who uninstalled it. In
each case the missing group is precisely the one the question was about, and the
data contains no trace of them.

The second theme is that data sources change without telling you.
{{cite:sculley2015}} names undeclared data dependencies as a distinct kind of
technical debt for exactly this reason: an upstream team renames a column,
changes a unit, or starts populating a field differently, and your numbers move
while your code stays identical. Nothing errors. The only defence is to state
your expectations explicitly and check them at the boundary.

## 3. Prerequisites

{{ch:ds-what-it-is}} for the data-generating process; {{ch:py-io-apis-sql}} for
reading files, APIs and databases, and for the idempotency material that
{{sec:5-formal-explanation}} builds on.

## 4. Intuitive Explanation

### 4.1 The sample is not the population

{{term:selection-bias}} arises whenever the probability of being in your data
depends on something related to what you are studying.

The mechanism is worth being precise about, because it is easy to nod at and
then commit anyway:

```text
population you care about   ████████████████████████████████
                                    │ selection process
sample you actually have    ░░░░████████░░░░░░░░████░░░░░░░░
                                 ▲                ▲
                        who got in, and why?  who did not?
```

If selection is independent of the outcome, the sample is representative and
everything works. If not, every estimate is biased, and the bias does not shrink
with more data — a million biased observations are just as wrong as a thousand.

> IMPORTANT: This is the sharpest distinction between a small-sample problem and
> a biased-sample problem. Small samples give wide confidence intervals, which
> more data fixes. Biased samples give narrow confidence intervals around the
> wrong number, which more data makes *worse* by making you more confident.

### 4.2 Survivorship bias

The most famous case is Abraham Wald's wartime analysis of returning aircraft.
Engineers proposed armouring the areas with the most bullet holes. Wald's
observation was that the data consisted only of planes that *returned*, so the
holes marked the survivable hits. The armour belonged where the returning planes
showed no damage, because planes hit there did not come back.

{{term:survivorship-bias}} is a selection bias where the filter is survival, and
it is endemic:

- Analysing successful companies to find what causes success, ignoring the
  identically-behaving failures.
- Backtesting a strategy on currently-listed stocks, excluding those that
  delisted.
- Studying users who completed onboarding to learn what makes onboarding work.
- Evaluating a model on requests it did not reject.

The diagnostic question is always the same: **what had to happen for a row to be
in this table?**

### 4.3 Provenance

{{term:provenance}} is the recorded history of a dataset — where each field came
from, what transformed it, and when. Without it you cannot answer the questions
that decide whether a number is trustworthy.

The practical minimum for any derived dataset:

- the source system and extraction timestamp
- the query or code that produced it, by version
- the transformation steps applied
- the row count at each stage
- who to ask when it looks wrong

That last item is not a joke. Most provenance questions are resolved by finding
the person who wrote the ingestion, and most datasets do not record who that
was.

### 4.4 Schema drift

{{term:schema-drift}} is an unannounced change to an input's structure or
meaning. The dangerous cases are not the ones that break:

{#tbl:schema-drift-kinds caption="Kinds of schema drift, by how they fail. Only the top row is safe, because it is the only one that stops the pipeline."}

| Change | Symptom |
|---|---|
| column removed | pipeline errors — **loud, therefore safe** |
| column renamed | pipeline errors, or silently becomes all-null |
| type changed (int → string) | silent coercion, or comparisons stop matching |
| **units changed** (dollars → cents) | **silent, every downstream number ×100** |
| **encoding changed** (`"Y"` → `true`) | **silent, filters stop matching** |
| new category appears | one-hot encoding shape changes, or the row is dropped |
| **backfill applied** | **history rewritten; yesterday's report no longer reproduces** |

The bolded rows are the expensive ones. They produce plausible numbers that are
wrong, which is strictly worse than a crash.

## 5. Formal Explanation

### 5.1 Sampling schemes

{#tbl:sampling-schemes caption="Sampling schemes and what each buys. The scheme determines which estimates are unbiased and what the standard error actually is."}

| Scheme | Method | Use when |
|---|---|---|
| Simple random | each unit equally likely | the population is homogeneous and enumerable |
| Stratified | sample within each stratum | subgroups differ and you need all of them represented |
| Cluster | sample whole groups | units are expensive to reach individually |
| Systematic | every $k$-th unit | a convenient ordered frame exists |
| Convenience | whoever is available | **never, for inference** |

**Stratified sampling** is the one worth knowing well. Sampling within strata
guarantees each subgroup appears, and it *reduces* variance relative to simple
random sampling when the strata differ:

$$
\Var(\bar{x}_{\text{strat}}) = \sum_h w_h^{2}\frac{\sigma_h^{2}}{n_h}
\;\le\;
\Var(\bar{x}_{\text{srs}})
$$ (eq:stratified-variance)

with $w_h$ the stratum's population share. The gain comes from removing
between-stratum variability from the estimate entirely — you are no longer at
the mercy of how many of each group happened to be drawn.

**Cluster sampling** has the opposite effect and it is the one people
accidentally use. Sampling whole groups means observations within a cluster are
correlated, which *inflates* the variance by the **design effect**:

$$
\text{DEFF} = 1 + (\bar{m} - 1)\,\rho
$$ (eq:design-effect)

with $\bar{m}$ the cluster size and $\rho$ the intra-cluster correlation. The
effective sample size is $n / \text{DEFF}$.

> WARNING: {{eq:design-effect}} is why "we have ten million rows" is not the
> same as ten million independent observations. Ten million events from
> 50,000 users, with modest within-user correlation, can have an effective
> sample size in the tens of thousands. Every confidence interval computed with
> $n = 10^{7}$ is then far too narrow, and {{sec:7-implementation}} measures how
> far.

### 5.2 Data contracts

A {{term:data-contract}} makes expectations explicit and enforceable at the
boundary. Minimally it specifies, per field: type, nullability, allowed range or
value set, and uniqueness. Plus, per table: expected row-count range, freshness,
and what counts as a breaking change.

```python {tier=C name=contract-sketch}
CONTRACT = {
    "user_id":   {"type": "int64",   "nullable": False, "unique": True},
    "signup_at": {"type": "datetime","nullable": False,
                  "min": "2020-01-01", "max": "now"},
    "plan":      {"type": "category","allowed": {"free", "pro", "enterprise"}},
    "spend_gbp": {"type": "float64", "min": 0, "max": 100_000},
}
```

The point is to **fail at ingestion** rather than to discover the problem in a
report three weeks later. A contract violation should stop the pipeline, and the
error should name the field and the specific expectation that failed.

> PRODUCTION TIP: Include an expected row-count *range*, not just a minimum. A
> load that produces ten times the usual rows is as suspicious as one producing
> none — it usually means a join fanned out ({{ch:py-pandas}}) or a batch was
> loaded twice.

### 5.3 What you are permitted to collect

A chapter on collection that ignores permission is incomplete, and the
constraints are practical rather than only ethical.

**Purpose limitation.** Data collected for one stated purpose frequently cannot
lawfully be used for another. A dataset gathered to deliver a service may not be
usable to train a model, depending on what was disclosed. This is a question to
resolve before building, not after.

**Retention.** Many jurisdictions require data to be deleted after a period, or
on request. A training set assembled from records that must later be erased
creates an obligation you may not be able to discharge — a trained model cannot
easily forget one row.

**Minimisation.** Collecting fields "in case they are useful later" increases
exposure without a corresponding benefit, and the useful-later fields are
frequently the sensitive ones.

**Special categories.** Health, biometric, political and similar data attract
stricter rules almost everywhere, and *inferred* membership of such a category
can attract them too — a model predicting pregnancy from purchase history is
processing health data whether or not the input columns look medical.

> PRODUCTION TIP: Record the lawful basis and the stated purpose alongside the
> provenance metadata of {{sec:4-intuitive-explanation}}. It costs one field at
> ingestion and is very expensive to reconstruct later, when someone asks
> whether a particular model was permitted to be trained on a particular table.

{{ch:rai-privacy}} treats this properly. The point here is that it belongs at
collection time, because that is the only moment when the answer is cheap.

### 5.4 Batch and streaming

{#tbl:batch-vs-streaming caption="Batch and streaming ingestion. The right choice follows from how fresh the data must be, not from which is more modern."}

| | Batch | Streaming |
|---|---|---|
| Latency | minutes to hours | seconds |
| Reprocessing | easy — rerun the job | hard — replay a log |
| Failure handling | rerun the window | checkpoints and offsets |
| Complexity | low | substantially higher |
| Right when | daily reporting, training data | fraud, personalisation, monitoring |

The default should be batch. Streaming is a real increase in operational
complexity, and freshness requirements are frequently asserted rather than
justified.

### 5.5 Idempotent loading

An ingestion job will be run twice. The network will drop, a scheduler will
retry, an engineer will rerun a failed window. If running twice produces
duplicates, every downstream number is silently wrong — which is exactly the
defect the audit in {{ch:ds-what-it-is}} found.

Three standard approaches:

**Upsert on a natural key.** `INSERT ... ON CONFLICT UPDATE`. Requires a
genuinely unique key.

**Partition overwrite.** Write each time window to its own partition and replace
it wholesale. Simple, robust, and the usual choice for batch.

**Deduplicate on read.** Keep everything, deduplicate by key and timestamp at
query time. Costs storage and query complexity, and preserves history.

This is {{term:idempotency}} from {{ch:py-io-apis-sql}}, applied to loads rather
than requests.

## 6. Mathematical Foundation

### 6.1 Selection bias, quantified

Let $S = 1$ denote inclusion in the sample. What you can estimate is
$\E[Y \mid S = 1]$; what you want is $\E[Y]$.

By the law of total expectation:

$$
\E[Y] = \E[Y \mid S=1]\,\Prob(S=1) + \E[Y \mid S=0]\,\Prob(S=0)
$$ (eq:selection-decomposition)

The bias is therefore

$$
\E[Y \mid S=1] - \E[Y] = \Prob(S=0)\,\big(\E[Y \mid S=1] - \E[Y \mid S=0]\big)
$$ (eq:selection-bias-magnitude)

Two things follow immediately.

The bias is zero exactly when $\E[Y \mid S=1] = \E[Y \mid S=0]$ — when the
included and excluded groups have the same mean outcome. That is precisely the
assumption that selection is unrelated to the outcome.

**The bias does not depend on $n$.** Nothing in
{{eq:selection-bias-magnitude}} involves sample size. Collecting more biased
data leaves the bias unchanged while shrinking the standard error, so your
interval tightens around the wrong value.

### 6.2 Survivorship bias worked

Suppose a strategy is evaluated on funds that still exist. Let 60% of funds
survive a decade, survivors return 8% on average, and closed funds returned
$-3\%$ before closing.

The observed mean is 8%. The true mean, from
{{eq:selection-decomposition}}:

$$
\E[Y] = (0.08)(0.6) + (-0.03)(0.4) = 0.048 - 0.012 = 0.036
$$

3.6%, not 8%. The bias is 4.4 percentage points, which is larger than most
effects anyone is trying to detect. From
{{eq:selection-bias-magnitude}}: $0.4 \times (0.08 - (-0.03)) = 0.044$ ✓.

Note that the bias grows with both the attrition rate and the difference between
the groups — and in survivorship situations those are usually both large,
because whatever caused the attrition is usually related to the outcome.

### 6.3 The design effect, worked

A dataset has 2,000,000 events from 40,000 users, so $\bar{m} = 50$ events per
user. Within-user correlation on the metric is $\rho = 0.3$.

$$
\text{DEFF} = 1 + (50 - 1)(0.3) = 1 + 14.7 = 15.7
$$

$$
n_{\text{eff}} = \frac{2{,}000{,}000}{15.7} \approx 127{,}000
$$

The effective sample is 127,000, not 2,000,000. Since the standard error scales
as $1/\sqrt{n}$ ({{ch:math-inference}}), treating the events as independent
understates it by $\sqrt{15.7} \approx 4.0\times$ — so a stated 95% confidence
interval is roughly a quarter of its correct width, and the real coverage is far
below 95%.

> IMPORTANT: This is the most common way that confidence intervals in industry
> analyses are wrong, and it is invisible: the arithmetic is correct, the code
> runs, and the interval is simply four times too narrow. The fix is to
> aggregate to the independent unit — one row per user — or to use a method that
> accounts for clustering. {{ch:ds-experiments}} returns to this, because the
> randomisation unit is the same question.

## 7. Implementation

```python {tier=A name=sampling-and-bias}
"""Selection bias, the design effect, and why more data does not help.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)

# --- eq. 22.3: bias does not shrink with n ----------------------------------
print("=" * 70)
print("selection bias is not a small-sample problem")
print("=" * 70)

TRUE_MEAN = 50.0


def biased_sample(n):
    """Users are sampled with probability increasing in their value —
    a convenience sample of the engaged."""
    pop = rng.normal(TRUE_MEAN, 15, n * 6)
    p_include = 1 / (1 + np.exp(-(pop - 50) / 8))      # value-dependent
    keep = rng.random(len(pop)) < p_include
    return pop[keep][:n], pop


print(f"{'n':>10} {'sample mean':>13} {'95% CI half-width':>19} "
      f"{'covers 50?':>12}")
for n in (100, 1_000, 10_000, 100_000):
    sample, _ = biased_sample(n)
    m = sample.mean()
    hw = 1.96 * sample.std(ddof=1) / np.sqrt(len(sample))
    covers = abs(m - TRUE_MEAN) < hw
    print(f"{n:>10,} {m:>13.3f} {hw:>19.3f} {str(covers):>12}")

print("\nThe interval narrows and the estimate does not approach 50. More")
print("data makes the wrong answer more confident (eq. 22.3).")

# --- eq. 22.4: survivorship bias, worked ------------------------------------
print("\n" + "=" * 70)
print("survivorship bias")
print("=" * 70)
n_funds = 20_000
true_return = rng.normal(0.036, 0.12, n_funds)
# Funds with poor returns are more likely to close.
p_survive = 1 / (1 + np.exp(-(true_return - 0.0) / 0.06))
survived = rng.random(n_funds) < p_survive

print(f"all funds        : mean return {true_return.mean():>7.3%}  "
      f"n={n_funds:,}")
print(f"survivors only   : mean return {true_return[survived].mean():>7.3%}  "
      f"n={survived.sum():,}")
print(f"closed funds     : mean return {true_return[~survived].mean():>7.3%}  "
      f"n={(~survived).sum():,}")

# Verify eq. 22.3 numerically.
p_out = (~survived).mean()
predicted_bias = p_out * (true_return[survived].mean()
                          - true_return[~survived].mean())
actual_bias = true_return[survived].mean() - true_return.mean()
print(f"\nbias predicted by eq. 22.3 : {predicted_bias:>7.4f}")
print(f"bias measured              : {actual_bias:>7.4f}")
assert abs(predicted_bias - actual_bias) < 1e-9
print("Analysing only survivors overstates returns by "
      f"{actual_bias*100:.1f} percentage points.")

# --- eq. 22.1: stratification reduces variance ------------------------------
print("\n" + "=" * 70)
print("stratified vs simple random sampling")
print("=" * 70)

# Three strata with very different means — the case where stratifying helps.
strata = {"small": (0.60, 20.0, 5.0), "medium": (0.30, 60.0, 8.0),
          "large": (0.10, 150.0, 20.0)}
population = np.concatenate([
    rng.normal(mu, sd, int(400_000 * w)) for w, mu, sd in strata.values()])
labels = np.concatenate([
    np.full(int(400_000 * w), name) for name, (w, _, _) in strata.items()])
true_mean = population.mean()

n_sample = 600
srs_means, strat_means = [], []
for _ in range(2000):
    idx = rng.choice(len(population), n_sample, replace=False)
    srs_means.append(population[idx].mean())

    total = 0.0
    for name, (w, _, _) in strata.items():
        pool = population[labels == name]
        take = max(2, int(n_sample * w))
        total += w * rng.choice(pool, take, replace=False).mean()
    strat_means.append(total)

print(f"true population mean        : {true_mean:.3f}")
print(f"{'scheme':<22} {'mean of estimates':>18} {'sd of estimates':>17}")
print(f"{'simple random':<22} {np.mean(srs_means):>18.3f} "
      f"{np.std(srs_means):>17.3f}")
print(f"{'stratified':<22} {np.mean(strat_means):>18.3f} "
      f"{np.std(strat_means):>17.3f}")
print(f"\nvariance reduction: {np.var(srs_means)/np.var(strat_means):.1f}x "
      f"for the same sample size (eq. 22.1)")
print("Both are unbiased; stratifying removes the luck of the draw across")
print("strata, which is where most of the variance was.")

# --- eq. 22.2: the design effect --------------------------------------------
print("\n" + "=" * 70)
print("clustered data: 'we have millions of rows' is not millions of samples")
print("=" * 70)

n_users, per_user = 40_000, 50
user_effect = rng.normal(0, 6.0, n_users)          # persistent per-user level
noise = rng.normal(0, 9.2, (n_users, per_user))
events = user_effect[:, None] + noise
flat = events.ravel()

# Intra-cluster correlation: share of variance that is between-user.
var_between = user_effect.var()
var_within = noise.var()
rho = var_between / (var_between + var_within)
deff = 1 + (per_user - 1) * rho

naive_se = flat.std(ddof=1) / np.sqrt(len(flat))
user_means = events.mean(axis=1)
correct_se = user_means.std(ddof=1) / np.sqrt(n_users)

print(f"{len(flat):,} events from {n_users:,} users ({per_user} each)")
print(f"intra-cluster correlation rho : {rho:.3f}")
print(f"design effect (eq. 22.2)      : {deff:.1f}")
print(f"effective sample size         : {len(flat)/deff:,.0f}")
print(f"\nnaive SE (events independent) : {naive_se:.5f}")
print(f"correct SE (user-level)       : {correct_se:.5f}")
print(f"understatement factor         : {correct_se/naive_se:.1f}x "
      f"(sqrt(DEFF) = {np.sqrt(deff):.1f})")
print("\nA 95% interval computed the naive way is about a quarter of its")
print("correct width. The arithmetic is right; the independence assumption")
print("is not.")
```

## 8. Practical Example

Enforcing a data contract at the boundary is the single highest-value habit in
ingestion. Below is a small validator and the drift it catches.

```python {tier=A name=data-contract-validation}
"""A data contract enforced at ingestion, catching drift before it spreads.

Each violation is a change that produces plausible wrong numbers rather than
an error, which is why it must be checked explicitly.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(3)


class ContractViolation(Exception):
    pass


CONTRACT = {
    "fields": {
        "user_id":   {"dtype": "int64",    "nullable": False, "unique": True},
        "signup_at": {"dtype": "datetime", "nullable": False},
        "plan":      {"dtype": "object",   "allowed": {"free", "pro",
                                                       "enterprise"}},
        "spend_gbp": {"dtype": "float",    "min": 0.0, "max": 50_000.0,
                      "max_null_frac": 0.20},
    },
    "rows": {"min": 800, "max": 5_000},
}


def validate(df: pd.DataFrame, contract: dict, name: str = "batch") -> None:
    problems = []

    n = len(df)
    lo, hi = contract["rows"]["min"], contract["rows"]["max"]
    if not lo <= n <= hi:
        problems.append(f"row count {n:,} outside expected [{lo:,}, {hi:,}]")

    for field, rules in contract["fields"].items():
        if field not in df.columns:
            problems.append(f"{field}: missing entirely")
            continue
        s = df[field]

        if rules["dtype"] == "datetime" and not pd.api.types.is_datetime64_any_dtype(s):
            problems.append(f"{field}: expected datetime, got {s.dtype}")
        elif rules["dtype"] == "int64" and not pd.api.types.is_integer_dtype(s):
            problems.append(f"{field}: expected integer, got {s.dtype}")
        elif rules["dtype"] == "float" and not pd.api.types.is_float_dtype(s):
            problems.append(f"{field}: expected float, got {s.dtype}")

        if not rules.get("nullable", True) and s.isna().any():
            problems.append(f"{field}: {s.isna().sum():,} nulls, none allowed")
        if "max_null_frac" in rules and s.isna().mean() > rules["max_null_frac"]:
            problems.append(f"{field}: null fraction {s.isna().mean():.1%} "
                            f"exceeds {rules['max_null_frac']:.0%}")
        if rules.get("unique") and s.duplicated().any():
            problems.append(f"{field}: {s.duplicated().sum():,} duplicates, "
                            f"must be unique")
        if "allowed" in rules:
            unexpected = set(s.dropna().unique()) - rules["allowed"]
            if unexpected:
                problems.append(f"{field}: unexpected values {sorted(unexpected)}")
        if "min" in rules and s.dropna().lt(rules["min"]).any():
            problems.append(f"{field}: values below {rules['min']}")
        if "max" in rules and s.dropna().gt(rules["max"]).any():
            worst = s.max()
            problems.append(f"{field}: max {worst:,.0f} exceeds {rules['max']:,.0f}")

    if problems:
        raise ContractViolation(
            f"{name}: {len(problems)} violation(s)\n  - "
            + "\n  - ".join(problems))
    print(f"{name}: passed ({n:,} rows)")


def make_batch(n=2000, **defects):
    df = pd.DataFrame({
        "user_id": np.arange(1, n + 1),
        "signup_at": pd.to_datetime("2026-01-01")
                     + pd.to_timedelta(rng.integers(0, 200, n), "D"),
        "plan": rng.choice(["free", "pro", "enterprise"], n, p=[.7, .25, .05]),
        "spend_gbp": np.where(rng.random(n) < 0.08, np.nan,
                              rng.gamma(2, 60, n).round(2)),
    })
    if defects.get("units_changed"):        # pounds became pence
        df["spend_gbp"] *= 100
    if defects.get("new_category"):
        df.loc[df.index[:40], "plan"] = "team"
    if defects.get("duplicated_batch"):
        df = pd.concat([df, df.iloc[:300]], ignore_index=True)
    if defects.get("more_nulls"):
        mask = rng.random(len(df)) < 0.35
        df.loc[mask, "spend_gbp"] = np.nan
    return df


print("a clean batch:")
validate(make_batch(), CONTRACT, "2026-08-13")

print("\nnow the four drifts from table 22.1 that do NOT raise an error on")
print("their own — each would silently corrupt downstream numbers:\n")

for label, defect in [
    ("units changed (GBP -> pence)", {"units_changed": True}),
    ("new category appeared", {"new_category": True}),
    ("batch loaded twice", {"duplicated_batch": True}),
    ("upstream started dropping values", {"more_nulls": True}),
]:
    try:
        validate(make_batch(**defect), CONTRACT, label)
    except ContractViolation as exc:
        first = str(exc).splitlines()[1].strip()
        print(f"  {label:<34} CAUGHT: {first[2:]}")

print("\nWithout the contract:")
bad = make_batch(units_changed=True)
clean = make_batch()
print(f"  mean spend, clean batch  : £{clean['spend_gbp'].mean():>12,.2f}")
print(f"  mean spend, units changed: £{bad['spend_gbp'].mean():>12,.2f}")
print("  Both are plausible-looking numbers. Only one is in pounds.")

# --- idempotent loading ------------------------------------------------------
print("\n" + "=" * 70)
print("idempotent loading: the job WILL run twice")
print("=" * 70)

warehouse = pd.DataFrame(columns=["user_id", "day", "spend"])
batch = pd.DataFrame({"user_id": [1, 2, 3], "day": "2026-08-13",
                      "spend": [10.0, 20.0, 30.0]})


def naive_append(wh, b):
    return pd.concat([wh, b], ignore_index=True)


def partition_overwrite(wh, b, day):
    kept = wh[wh["day"] != day]
    return pd.concat([kept, b], ignore_index=True)


naive = warehouse.copy()
idem = warehouse.copy()
for attempt in range(1, 4):
    naive = naive_append(naive, batch)
    idem = partition_overwrite(idem, batch, "2026-08-13")
    print(f"  after run {attempt}: naive total £{naive['spend'].sum():>6.0f}, "
          f"idempotent total £{idem['spend'].sum():>6.0f}")

print("\nThe naive load triples revenue after two retries. The partition")
print("overwrite is unchanged — which is what makes a rerun safe.")
```

## 9. Common Mistakes

**Assuming the sample represents the population.** Ask what had to happen for a
row to exist.

**Believing more data fixes bias.** It does not; it narrows the interval around
the wrong value ({{eq:selection-bias-magnitude}}).

**Analysing only survivors.** The excluded group is usually the informative one.

**Treating clustered observations as independent.** The design effect can be
15× or more; intervals are then far too narrow.

**Using convenience samples for inference.** Fine for exploration, invalid for
estimating a population quantity.

**Not recording provenance.** A number that cannot be traced cannot be defended.

**No data contract.** Silent schema drift produces plausible wrong numbers.

**Checking only a minimum row count.** Ten times the expected rows is as
suspicious as none.

**Non-idempotent loads.** The job will be rerun, and duplicates inflate
everything.

**Choosing streaming by default.** It is a substantial operational cost;
freshness requirements are often asserted rather than justified.

## 10. Connection to Previous Chapters

{{ch:ds-what-it-is}} introduced the data-generating process; this chapter is the
part of it you can actually control and record.
{{ch:math-inference}} supplied the standard error that
{{eq:design-effect}} corrects, and the independence assumption whose failure
that correction quantifies. {{ch:py-io-apis-sql}} supplied idempotency, applied
here to loads, and the format choices that determine whether types survive
ingestion.

Forward: {{ch:ds-cleaning}} handles the defects a contract catches;
{{ch:ds-experiments}} meets the clustering question again as the choice of
randomisation unit, which is the same mathematics; {{ch:ds-leakage}} shows what
happens when the selection process is related to the target.

Beyond Part III: {{ch:mle-pipelines}} formalises validated ingestion, and
{{ch:ops-versioning}} extends provenance to full data lineage.
{{cite:sculley2015}} is the reference for undeclared data dependencies as
technical debt.

## 11. Exercises

**Beginner**

1. For a dataset of completed purchases, name three populations it excludes.
2. Give an example of survivorship bias from a field you know.
3. When is stratified sampling preferable to simple random sampling?
4. List five things a minimal provenance record should contain.
5. Which kinds of schema drift are safe, and why?

**Intermediate**

6. Using {{eq:selection-bias-magnitude}}, compute the bias when 30% are
   excluded, included mean 70, excluded mean 40.
7. Compute the design effect for 20 observations per cluster with
   $\rho = 0.15$, and the effective sample size for 500,000 rows.
8. Write a data contract for a table you have worked with, including a
   row-count range.
9. Explain why a load must be idempotent, with a concrete failure.
10. A dataset has one row per event and you analyse it per event. When is that
    valid?
11. Give a schema change that raises no error and changes every downstream
    number.

**Advanced**

12. Derive {{eq:selection-bias-magnitude}} from
    {{eq:selection-decomposition}}.
13. Derive the design effect {{eq:design-effect}} for equal-sized clusters with
    exchangeable correlation $\rho$.
14. Under what conditions can selection bias be corrected by reweighting, and
    what must you know to do it?
15. Design an ingestion system detecting all seven drift types in
    {{tbl:schema-drift-kinds}}, stating which need historical comparison.

**Implementation**

16. Extend the validator to check freshness and to compare a batch's
    distribution against a stored reference.
17. Implement stratified sampling for a dataframe and empirically verify the
    variance reduction of {{eq:stratified-variance}}.
18. Write a function estimating intra-cluster correlation and effective sample
    size from a clustered dataset.
19. Implement all three idempotent load strategies and demonstrate each is
    unaffected by a double run.

**Reasoning**

20. Your team has 50 million rows and wants tighter confidence intervals. What
    do you ask before agreeing that the sample is large?
21. Data contracts add friction between teams. Argue both sides, and say where
    you would place the boundary.

## 12. Chapter Summary

Every conclusion is about whoever ended up in the dataset. Selection bias arises
when inclusion depends on something related to the outcome, and its magnitude is
$\Prob(S=0)$ times the difference in means between the included and excluded
groups — a quantity that does not involve sample size. More biased data does not
help; it narrows the interval around the wrong value.

Survivorship bias is selection on survival and is endemic: successful companies,
listed funds, users who completed onboarding. The diagnostic question is always
what had to happen for a row to exist.

Sampling scheme determines which estimates are unbiased and what the standard
error is. Stratifying reduces variance when strata differ, by removing
between-stratum luck from the estimate. Clustering increases it by the design
effect $1 + (\bar{m}-1)\rho$, which is frequently an order of magnitude and is
the most common reason industry confidence intervals are too narrow.

Provenance — source, extraction time, code version, transformations, row counts,
and who to ask — is what makes a number defensible. It is not recoverable after
the fact.

Schema drift changes an input without announcing it. The dangerous cases are the
silent ones: changed units, changed encodings, and backfills, all of which
produce plausible numbers that are wrong. A data contract stating types,
nullability, ranges, allowed values and an expected row-count *range* turns
those into ingestion-time failures.

Ingestion jobs get rerun. Loads must be idempotent — by upsert, partition
overwrite, or deduplication on read — or a retry silently multiplies every
aggregate.

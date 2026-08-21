---
id: ds-what-it-is
number: 21
part: III
tier: focused
status: reviewed
requires: [py-pandas, math-inference]
provides: [data-generating-process, data-drift-term, concept-drift-term]
citations: [sculley2015, automind2025]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. State what distinguishes data science from statistics, analytics and machine
   learning engineering, and why the boundaries are blurry.
2. Describe the actual lifecycle of a data-science project and where the effort
   really goes.
3. Explain why understanding the data-generating process matters more than
   choosing a model.
4. Recognise the ways projects fail that have nothing to do with modelling.
5. Distinguish data drift from concept drift and explain why both make a
   deployed model decay.
6. State honestly what AI agents have and have not automated about this work.
7. Frame a business question as an answerable data question, and identify when
   it cannot be.

## 2. Why This Matters

Most introductions to data science are a list of techniques. That is a poor
guide to the work, because the techniques are the easy part and are increasingly
performed for you.

Two facts shape everything in this part.

**The modelling is a small fraction of the effort.** {{cite:sculley2015}} made
the point memorably about deployed ML systems: the model code is a small box in
the middle of a large diagram of everything else. The same is true of analysis.
Understanding where the data came from, getting it into a usable state, and
determining whether the answer is trustworthy consumes most of the time on any
real project.

**Most failures are not modelling failures.** A project fails because the
question was not answerable from the available data, because the training data
did not resemble production, because a correlation was interpreted causally, or
because the metric optimised was not the one anyone cared about. A better model
does not fix any of these.

This chapter sets up the rest of the part by naming those failure modes
explicitly, so the chapters that follow can address them one at a time.

## 3. Prerequisites

{{ch:py-pandas}} for dataframes, {{ch:math-inference}} for the fact that every
measured number carries uncertainty. Nothing else.

## 4. Intuitive Explanation

### 4.1 What the job actually is

Data science is the practice of answering questions with data, under
uncertainty, in a way that survives scrutiny.

That definition is deliberately about *answering questions* rather than
*building models*. A model is one way of answering some questions. Many
valuable analyses involve no model at all, and many models answer no question
anyone asked.

The neighbouring disciplines overlap heavily:

{#tbl:adjacent-fields caption="How the adjacent disciplines differ in emphasis. The boundaries are institutional rather than principled, and any real role spans several columns."}

| Discipline | Primary output | Emphasis |
|---|---|---|
| Statistics | an inference with quantified uncertainty | correctness, assumptions |
| Analytics | a description of what happened | clarity, timeliness |
| Data science | a decision-supporting answer | judgement across the whole chain |
| ML engineering | a system that serves predictions | reliability, latency, scale |
| Research | a new method | novelty, generality |

The distinguishing feature of data science as practised is **span**: the same
person is expected to interrogate the source, clean it, analyse it, model it,
evaluate it, and explain the result to someone who will act on it. Errors leak
across those stages, which is why owning the whole chain matters.

### 4.2 Where the effort goes

The commonly-repeated claim is that data scientists spend 80% of their time on
data preparation. The number is folklore and the shape is right. A realistic
breakdown of a project that reaches production:

```text
question framing      ▓▓▓                      what are we actually asking?
data understanding    ▓▓▓▓▓▓▓                  where did this come from?
cleaning & joining    ▓▓▓▓▓▓▓▓▓▓▓▓             the bulk of it
exploration           ▓▓▓▓▓▓
feature engineering   ▓▓▓▓▓▓▓
modelling             ▓▓▓                      the part tutorials cover
evaluation            ▓▓▓▓▓
communication         ▓▓▓▓
deployment & monitor  ▓▓▓▓▓▓▓
```

The two largest blocks are the two that no tutorial teaches. That is not an
accident: they are the parts that depend on the specific dataset and cannot be
demonstrated generically.

### 4.3 The data-generating process

The single most valuable thing to know about a dataset is not in it.

The {{term:data-generating-process}} is the real-world mechanism that produced
your rows: who was measured, when, by what instrument, under what conditions,
and — critically — who was left out. Questions worth asking before any analysis:

- **Who is in this data, and who is missing?** A dataset of customers excludes
  everyone who did not become a customer, which is usually the population you
  care about.
- **When was each field populated?** A field written after the event you are
  predicting cannot be used to predict it ({{ch:ds-leakage}}).
- **What does a missing value mean here?** Not measured, does not apply, or
  genuinely zero ({{ch:ds-cleaning}}).
- **Has the definition changed?** Columns get redefined and backfilled, and
  nothing in the table records it.
- **What decisions were made using this data before?** If a model already acts
  on these users, the data reflects the model's behaviour, not their
  preferences ({{ch:ds-recsys}}).

> IMPORTANT: You cannot recover this by looking at the data harder. It requires
> asking the people who produced it, reading the ingestion code, or examining
> the source system. An analyst who will not do that is guessing, however
> sophisticated the model.

### 4.4 Why deployed models decay

A model is a statement about a relationship that held in the past. The world
does not hold still, and there are two distinct ways it moves.

**{{term:data-drift-term}}** — the inputs change. Your users get younger, a new
market opens, a sensor is recalibrated. $p(x)$ moves away from the training
distribution.

**{{term:concept-drift-term}}** — the *relationship* changes. The same input now
implies a different outcome: a price point that signalled a premium customer now
signals a discount hunter. $p(y \mid x)$ moves.

The second is harder, because the inputs may look entirely normal. Input
monitoring catches drift; only outcome monitoring catches concept drift, and
outcomes often arrive late ({{ch:mle-drift}}).

## 5. Formal Explanation

### 5.1 The lifecycle

```mermaid {#fig:ds-lifecycle caption="The data-science lifecycle. The loops are the point: almost every stage sends you back to an earlier one, and a project that proceeds linearly has usually skipped the checking."}
graph TD
  Q[frame the question] --> D[understand the source]
  D --> C[clean and join]
  C --> E[explore]
  E --> D
  E --> F[engineer features]
  F --> M[model]
  M --> V[evaluate]
  V --> F
  V --> Q
  V --> S[ship]
  S --> MON[monitor]
  MON --> C
  MON --> Q
```

The backward edges carry most of the value. Exploration routinely reveals that
the cleaning was wrong. Evaluation routinely reveals that the question was
mis-framed. Monitoring routinely reveals that the world moved.

### 5.2 Framing a question

Business questions are rarely answerable as stated. The work of translation has
a standard shape.

{#tbl:question-framing caption="Translating business questions into answerable ones. The right-hand column is what you can actually compute; the gap between the two columns is where most disagreement lives."}

| As asked | As answerable |
|---|---|
| "Why are sales down?" | "Which segments declined, by how much, relative to what baseline?" |
| "Will this customer churn?" | "What is P(no activity in 30 days \| features as of today)?" |
| "Is the new design better?" | "Does variant B raise conversion by ≥1pp at 80% power?" |
| "Can we predict demand?" | "What is next week's units sold, and what error is acceptable?" |

Three things must be pinned down before any work starts.

**The decision.** What will be done differently depending on the answer? If
nothing, the analysis is not worth doing.

**The metric.** Precisely defined, including the population and window. "Churn"
means nothing until you say inactive for how long, measured from when.

**The threshold.** What magnitude of effect matters? This determines the sample
size you need ({{ch:math-inference}}) and should be agreed before you look.

> WARNING: Agreeing the threshold *after* seeing the result is how analyses
> become negotiations. It is the same selection problem as
> {{ch:math-inference}}'s multiple comparisons, applied to the acceptance
> criterion instead of the hypothesis.

### 5.3 How projects fail

{#tbl:failure-modes caption="How data-science projects fail, and where each is addressed. Only one row is a modelling problem."}

| Failure | Where addressed |
|---|---|
| The question was not answerable from this data | this chapter |
| The sample does not represent the population | {{ch:ds-collection}} |
| Missing values were handled without knowing why they were missing | {{ch:ds-cleaning}} |
| A correlation was interpreted causally | {{ch:ds-causation}} |
| The experiment was underpowered or peeked at | {{ch:ds-experiments}} |
| A feature will not exist at prediction time | {{ch:ds-leakage}} |
| The validation split leaked | {{ch:ds-leakage}} |
| The metric optimised was not the one that mattered | {{ch:ml-metrics}} |
| The model was fine and the world changed | {{ch:mle-drift}} |
| Nobody acted on the result | this chapter |

The last row is worth taking as seriously as the others. An analysis nobody uses
has the same value as an incorrect one, and the usual causes are that the
decision was never identified, or the result arrived after the decision was
made.

### 5.4 What agents have automated

The mechanical stages are now substantially automated, and it is worth being
specific about which.

**Largely automated.** Writing SQL from a description. Generating a first-pass
exploratory analysis. Proposing feature transformations. Sweeping model families
and hyperparameters. Producing plots. Writing the boilerplate around all of it.
Frameworks now attempt whole workflows end to end
{{cite:automind2025}}. {{maturity:EMERGING}}

**Not automated.** Knowing which question is worth asking. Knowing how the data
was generated and what that rules out. Recognising confounding. Noticing that a
join changed the row count. Judging whether an effect is large enough to act on.
Deciding what to do when the answer is inconvenient.

The pattern is that **generation is cheap and verification is not**. The tasks
that became free were the ones with a checkable output and no requirement to
know anything outside the dataset. What remains requires context the data does
not contain.

> RESEARCH NOTE: The practical consequence is counter-intuitive. Because a
> plausible-looking analysis can now be produced in seconds, the marginal value
> of the failure modes in {{tbl:failure-modes}} has gone *up*. An agent will
> generate a confounded comparison as readily as a human and considerably
> faster. The skill that has appreciated is recognising that an answer is wrong.
> {{part:20}} develops this once agents have been covered properly.

## 6. Mathematical Foundation

### 6.1 What a model is claiming

A supervised model estimates a conditional distribution:

$$
\hat{p}(y \mid \vec{x}) \approx p(y \mid \vec{x})
$$ (eq:model-claim)

Two assumptions are buried in that approximation, and both fail routinely.

**The training data is drawn from the same distribution as deployment.**
Formally, training pairs and future pairs share a joint distribution
$p(\vec{x}, y)$. Drift is exactly the failure of this.

**Observations are independent.** This underlies every standard error in
{{ch:math-inference}}. It fails whenever the same user appears in many rows,
when observations are ordered in time ({{ch:ds-timeseries}}), or when the
system's own output influenced the data ({{ch:ds-recsys}}).

Neither assumption is checkable from the data alone. Both are claims about the
data-generating process.

### 6.2 Decomposing drift

Write the joint distribution two ways:

$$
p(\vec{x}, y) = p(y \mid \vec{x})\,p(\vec{x}) = p(\vec{x} \mid y)\,p(y)
$$ (eq:joint-factorisations)

Drift is a change in this joint distribution between training time and
deployment, and the factorisation names the kinds:

- **Covariate shift**: $p(\vec{x})$ changes, $p(y \mid \vec{x})$ does not. The
  relationship still holds; you are seeing different inputs. Often recoverable
  by reweighting.
- **Concept drift**: $p(y \mid \vec{x})$ changes. The relationship itself is
  different. Requires retraining on new labels.
- **Label shift**: $p(y)$ changes, $p(\vec{x} \mid y)$ does not. Common in
  fraud and disease prevalence, and correctable if you can estimate the new
  base rate — which is the Bayes' theorem argument of
  {{ch:math-probability}}.

The practical distinction is what you can detect and when. Input drift is
detectable immediately from features alone; concept drift is only detectable
once labels arrive, which may be weeks later
({{ch:mle-drift}}).

### 6.3 Why sample size and effect size determine feasibility

Before a project starts, {{eq:sample-size}} from {{ch:math-inference}} already
tells you whether the question is answerable with the data available.

For detecting a difference $\delta$ in a proportion near $\bar{p}$ at 5%
significance and 80% power:

$$
n \approx \frac{2(1.96 + 0.84)^{2}\,\bar{p}(1-\bar{p})}{\delta^{2}}
$$

With a conversion rate near 5% and a target of detecting a 0.5 percentage-point
absolute improvement:

$$
n \approx \frac{2(7.84)(0.05)(0.95)}{0.005^{2}} = \frac{0.745}{0.000025} \approx 29{,}800
$$

per variant. If the site sees 2,000 visitors a week, that is fifteen weeks. That
calculation takes two minutes and can end an infeasible project before anyone
builds anything — which is one of the highest-value things a data scientist
does. {{sec:7-implementation}} computes it across a range of cases.

## 7. Implementation

```python {tier=A name=project-feasibility}
"""Framing and feasibility: the arithmetic that should precede a project.

Nothing here is modelling. All of it decides whether modelling is worth doing.
"""
import numpy as np
import pandas as pd

# --- eq. 21.4: is the question answerable with the traffic available? -------
def required_n(delta, p_bar, alpha_z=1.96, beta_z=0.84):
    """Per-variant sample size to detect an absolute difference delta."""
    return 2 * (alpha_z + beta_z) ** 2 * p_bar * (1 - p_bar) / delta ** 2


print("Feasibility before any modelling: how long must an experiment run?\n")
weekly_traffic = 2000
print(f"{'baseline':>9} {'target lift':>12} {'n per arm':>11} "
      f"{'weeks at 2k/wk':>16}")
for p_bar in (0.05, 0.20):
    for delta in (0.005, 0.01, 0.02, 0.05):
        n = required_n(delta, p_bar)
        weeks = 2 * n / weekly_traffic          # two arms
        flag = "  <- infeasible" if weeks > 12 else ""
        print(f"{p_bar:>9.1%} {delta:>12.1%} {n:>11,.0f} "
              f"{weeks:>16.1f}{flag}")

print("\nDetecting a 0.5pp lift on a 5% baseline needs ~30k per arm — thirty")
print("weeks of traffic here. That answer costs two minutes and can stop an")
print("infeasible project before anyone writes a model.")

# --- the effort distribution, measured on a real-ish pipeline ---------------
print("\n" + "=" * 68)
print("where the effort actually goes")
print("=" * 68)

rng = np.random.default_rng(0)
n = 40_000

# A deliberately messy source, of the kind that actually arrives.
raw = pd.DataFrame({
    "user_id": rng.integers(1, 9000, n),
    "signup": pd.to_datetime("2025-01-01") + pd.to_timedelta(
        rng.integers(0, 500, n), "D"),
    "channel": rng.choice(["web", "Web", "app", " app", None], n,
                          p=[0.35, 0.1, 0.3, 0.1, 0.15]),
    "revenue": np.where(rng.random(n) < 0.12, np.nan,
                        rng.gamma(2, 30, n)),
    "country": rng.choice(["GB", "gb", "US", "FR", ""], n,
                          p=[0.3, 0.1, 0.3, 0.2, 0.1]),
})

issues = {
    "rows": len(raw),
    "duplicate user_ids": int(raw["user_id"].duplicated().sum()),
    "missing revenue": int(raw["revenue"].isna().sum()),
    "missing channel": int(raw["channel"].isna().sum()),
    "channel variants": raw["channel"].dropna().nunique(),
    "country variants": raw["country"].nunique(),
    "empty-string countries": int((raw["country"] == "").sum()),
}
print("what an unexamined dataset contains:")
for k, v in issues.items():
    print(f"  {k:<26} {v:>8,}")

cleaned = raw.assign(
    channel=raw["channel"].str.strip().str.lower(),
    country=raw["country"].str.upper().replace("", np.nan),
)
print(f"\nafter two lines of normalisation:")
print(f"  channel variants           {cleaned['channel'].dropna().nunique():>8}"
      f"   (was {issues['channel variants']})")
print(f"  country variants           {cleaned['country'].dropna().nunique():>8}"
      f"   (was {issues['country variants']})")
print("Those variants would have become separate categories in a one-hot")
print("encoding, splitting the same signal across duplicate columns.")

# --- eq. 21.2/21.3: simulating the two kinds of drift -----------------------
print("\n" + "=" * 68)
print("data drift vs concept drift")
print("=" * 68)

def fit_threshold(x, y):
    """A trivial model: the threshold on x that best separates the classes."""
    candidates = np.quantile(x, np.linspace(0.05, 0.95, 60))
    accs = [((x > t) == y).mean() for t in candidates]
    return candidates[int(np.argmax(accs))]


# Training world: y = 1 when x is large.
x_train = rng.normal(50, 10, 20_000)
y_train = (x_train + rng.normal(0, 3, 20_000)) > 55
thr = fit_threshold(x_train, y_train)
base_acc = ((x_train > thr) == y_train).mean()
print(f"trained threshold {thr:.2f}, training accuracy {base_acc:.3f}")

# Covariate shift: inputs move, the RELATIONSHIP is unchanged.
x_cov = rng.normal(62, 10, 20_000)                       # p(x) moved
y_cov = (x_cov + rng.normal(0, 3, 20_000)) > 55          # p(y|x) same
acc_cov = ((x_cov > thr) == y_cov).mean()

# Concept drift: inputs look identical, the relationship inverted.
x_con = rng.normal(50, 10, 20_000)                       # p(x) unchanged
y_con = (x_con + rng.normal(0, 3, 20_000)) < 45          # p(y|x) CHANGED
acc_con = ((x_con > thr) == y_con).mean()

print(f"\n{'scenario':<22} {'mean(x)':>9} {'accuracy':>10} "
      f"{'detectable from x alone?':>26}")
print(f"{'training':<22} {x_train.mean():>9.2f} {base_acc:>10.3f} "
      f"{'—':>26}")
print(f"{'covariate shift':<22} {x_cov.mean():>9.2f} {acc_cov:>10.3f} "
      f"{'yes: the mean moved':>26}")
print(f"{'concept drift':<22} {x_con.mean():>9.2f} {acc_con:>10.3f} "
      f"{'NO: inputs look normal':>26}")

print("\nMonitoring the input distribution catches the first and is blind to")
print("the second — which is the more damaging of the two, and only shows up")
print("once labels arrive (Chapter 48).")

# --- the cost of not identifying the decision -------------------------------
print("\n" + "=" * 68)
print("framing: does the answer change a decision?")
print("=" * 68)
questions = [
    ("Which segment has the highest churn?",
     "yes — determines where retention spend goes"),
    ("What is our average session length?",
     "no decision attached — a number for a slide"),
    ("Will this user churn in 30 days?",
     "yes — triggers an intervention, if one exists"),
    ("How accurate could a churn model be?",
     "only if the accuracy threshold for acting is agreed first"),
]
for q, verdict in questions:
    print(f"  {q:<42} {verdict}")
print("\nAn analysis with no decision attached has the same value as a wrong")
print("one, and costs the same to produce.")
```

## 8. Practical Example

Interrogating a dataset before analysing it is the habit this chapter is really
teaching. The following is a reusable audit that asks the
{{sec:4-intuitive-explanation}} questions mechanically.

```python {tier=A name=dataset-audit}
"""A provenance and sanity audit, run before any analysis.

Most of what this finds is invisible in a .head() and fatal downstream.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(11)

# A dataset with several realistic, individually-plausible defects.
n = 30_000
start = pd.to_datetime("2025-06-01")
df = pd.DataFrame({
    "user_id": rng.integers(1, 12_000, n),
    "event_time": start + pd.to_timedelta(rng.integers(0, 300, n), "D"),
    "plan": rng.choice(["free", "pro", "enterprise"], n, p=[.7, .25, .05]),
    "spend": rng.gamma(2, 40, n).round(2),
    "region": rng.choice(["EU", "NA", "APAC"], n, p=[.4, .45, .15]),
})
# Defect 1: a column that was only backfilled from a certain date.
df["nps_score"] = np.where(df["event_time"] < "2025-09-01", np.nan,
                           rng.integers(0, 11, n))
# Defect 2: a field populated only AFTER the outcome — target leakage.
df["churned"] = rng.random(n) < 0.18
df["cancellation_reason"] = np.where(df["churned"],
                                     rng.choice(["price", "bug", "moved"], n),
                                     None)
# Defect 3: a duplicated batch, from an ingestion retry.
df = pd.concat([df, df.iloc[:900]], ignore_index=True)


def audit(df, time_col=None, id_col=None, target=None):
    """Ask the questions of section 4.3 mechanically."""
    print(f"rows {len(df):,}  columns {df.shape[1]}\n")

    print(f"{'column':<22} {'dtype':<12} {'null%':>7} {'nunique':>9} "
          f"{'sample':<20}")
    for c in df.columns:
        s = df[c]
        sample = str(s.dropna().iloc[0])[:18] if s.notna().any() else "—"
        print(f"{c:<22} {str(s.dtype):<12} {s.isna().mean():>6.1%} "
              f"{s.nunique():>9,} {sample:<20}")

    print(f"\nexact duplicate rows: {df.duplicated().sum():,}")
    if id_col:
        dup_ids = df[id_col].duplicated().sum()
        print(f"repeated {id_col}: {dup_ids:,} "
              f"({dup_ids/len(df):.1%}) — is one row per {id_col} expected?")

    # Missingness that varies over time almost always means a schema change.
    if time_col:
        print(f"\nmissingness over time (a jump means the column changed):")
        monthly = df.set_index(time_col).resample("MS").apply(
            lambda g: g.isna().mean())
        for c in df.columns:
            if df[c].isna().any():
                series = monthly[c].dropna()
                if len(series) > 1 and series.max() - series.min() > 0.4:
                    first_full = series[series < 0.5].index.min()
                    print(f"  {c}: {series.min():.0%}–{series.max():.0%} "
                          f"across months; only populated from "
                          f"{first_full.date()}")

    # A column almost perfectly determined by the target is a leak.
    if target:
        print(f"\ncolumns suspiciously related to '{target}':")
        for c in df.columns:
            if c == target:
                continue
            present = df[c].notna()
            if present.nunique() < 2:
                continue
            agreement = max((present == df[target]).mean(),
                            (present != df[target]).mean())
            if agreement > 0.95:
                print(f"  {c}: presence agrees with {target} "
                      f"{agreement:.1%} of the time  <- likely target leakage")


audit(df, time_col="event_time", id_col="user_id", target="churned")

print("\n" + "=" * 68)
print("what the audit found, and what each means")
print("=" * 68)
print("1. 900 duplicate rows — an ingestion retry. Every sum and count is")
print("   inflated by 3% until they are removed.")
print("2. nps_score is entirely missing before 2025-09-01 — the column was")
print("   added mid-stream. Training on the full history teaches the model")
print("   that 'missing NPS' means 'earlier period', which is a date proxy.")
print("3. cancellation_reason is populated exactly when churned is true. It")
print("   is a consequence of the target, not a predictor of it. Including it")
print("   gives a model that appears near-perfect and is useless.")
print("\nNone of these are visible in df.head(), and all three would have")
print("produced a confident, wrong result. Chapters 23 and 28 handle them.")
```

## 9. Common Mistakes

**Starting with the model.** The model is the least uncertain part of the
project.

**Not identifying the decision.** If no action depends on the answer, the
analysis has no value.

**Accepting a metric definition without pinning down the window and
population.** "Churn" and "active user" mean nothing until specified.

**Not asking where the data came from.** Provenance is not in the data and
determines what conclusions are available.

**Assuming the sample represents the population.** It represents whoever got
into the dataset ({{ch:ds-collection}}).

**Treating missing values as a nuisance rather than a signal.** Why a value is
missing is frequently informative ({{ch:ds-cleaning}}).

**Agreeing the success threshold after seeing results.** The multiple-comparisons
problem applied to the acceptance criterion.

**Ignoring drift after deployment.** A model is a claim about the past.

**Assuming an agent's output is correct because it is fluent.** Generation is
cheap; verification is the work.

**Delivering after the decision was made.** A correct answer that arrives late
is not a correct answer to anything.

## 10. Connection to Previous Chapters

{{ch:math-inference}} supplies the sample-size arithmetic of
{{sec:6-mathematical-foundation}}, which is what makes feasibility assessable
before a project starts, and the multiple-comparisons argument that explains why
thresholds must be set in advance. {{ch:math-probability}} supplies the base-rate
reasoning behind label shift. {{ch:py-pandas}} supplies the tooling the audit in
{{sec:8-practical-example}} is built from.

Forward: the failure modes listed in {{tbl:failure-modes}} are the syllabus for
the rest of this part. {{ch:ds-collection}} takes up provenance and sampling;
{{ch:ds-cleaning}} takes up missingness; {{ch:ds-causation}} takes up the causal
error; {{ch:ds-leakage}} takes up the two defects found by the audit.

Beyond Part III: {{ch:mle-drift}} monitors for the drift decomposed in
{{eq:joint-factorisations}}; {{ch:ml-metrics}} covers choosing the right metric;
{{part:20}} returns to what agents can and cannot do, with the agent material of
{{part:17}} available. {{cite:sculley2015}} is the reference for why the model
is the small box.

## 11. Exercises

**Beginner**

1. Give three questions that data can answer and two it cannot, and say what
   distinguishes them.
2. For "are our users happy?", write an answerable version with a metric,
   population and window.
3. Distinguish data drift from concept drift with an example of each.
4. Name four things about a dataset's provenance that are not in the dataset.
5. Why is "we should build a model" a poor project starting point?

**Intermediate**

6. Using {{eq:sample-size}}, compute the traffic needed to detect a 1pp lift on
   a 10% baseline, and how long it takes at 5,000 visitors a week.
7. A dataset of loan defaults contains only approved applicants. What can and
   cannot be concluded from it?
8. Give a case where covariate shift is harmless and one where it is fatal.
9. Explain why concept drift is harder to detect than data drift, in terms of
   {{eq:joint-factorisations}}.
10. A stakeholder asks for "a churn model". List the five questions you would
    ask before writing any code.
11. A column is 100% missing before a certain date. Give two distinct
    explanations and say how you would distinguish them.

**Advanced**

12. Formalise the independence assumption behind {{eq:model-claim}} and give
    three realistic situations in which it fails.
13. Label shift is correctable if the new base rate can be estimated. Derive the
    correction using Bayes' theorem ({{ch:math-probability}}).
14. Argue for and against the claim that data science is becoming a verification
    discipline rather than a generation one.
15. Design a monitoring scheme detecting both kinds of drift, stating what each
    signal can and cannot catch and the delay before it fires.

**Implementation**

16. Extend the audit in {{sec:8-practical-example}} to flag columns whose
    cardinality is close to the row count, and explain why that matters.
17. Write a function that, given a dataframe and a time column, reports every
    column whose null rate changes by more than 30 percentage points between
    months.
18. Build a feasibility calculator taking baseline rate, minimum detectable
    effect and weekly traffic, and returning the required duration with a
    warning above twelve weeks.
19. Simulate covariate shift and concept drift and demonstrate that an
    input-distribution monitor detects only the first.

**Reasoning**

20. Agents can generate an analysis in seconds. Does that make the failure modes
    in this chapter more or less important? Argue carefully.
21. The effort distribution in {{sec:4-intuitive-explanation}} shows modelling as
    a small slice. Why do courses and tutorials concentrate on it?

## 12. Chapter Summary

Data science is answering questions with data under uncertainty, in a way that
survives scrutiny. Its distinguishing feature as practised is span: the same
person owns the chain from source to decision, because errors leak across the
stages.

Modelling is a small fraction of the effort and of the risk. Most failures are
not modelling failures — the question was unanswerable, the sample was
unrepresentative, a correlation was read causally, a feature will not exist at
prediction time, or nobody acted on the result.

The most valuable thing to know about a dataset is not in it. The
data-generating process — who was measured, when, by what, and who was excluded
— determines which conclusions are available, and recovering it requires asking
people or reading ingestion code rather than looking at the table harder.

Framing requires three things fixed in advance: the decision the answer will
change, the metric with its population and window, and the effect size that
matters. The last of these plus the sample-size formula tells you whether the
project is feasible at all, in about two minutes.

Deployed models decay in two distinct ways. Data drift moves $p(x)$ and is
detectable from inputs alone. Concept drift moves $p(y \mid x)$, leaves the
inputs looking normal, and is only detectable once labels arrive — which makes
it both more damaging and slower to catch.

AI has automated the execution of data science far faster than the judgement.
Writing queries, generating exploratory analyses, proposing features and
sweeping models are largely solved; knowing which question matters, how the data
was made, and whether an answer is confounded is not. Because generating a
plausible analysis is now nearly free, recognising a wrong one has become the
scarcer and more valuable skill.

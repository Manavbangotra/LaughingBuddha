---
id: part-03-assessment
status: final
---

## How to use this

Part III is about not being fooled, so this assessment is weighted toward
finding errors rather than producing results. The assignment asks you to break
a pipeline someone else wrote, and the challenge asks you to establish a causal
claim from observational data and then say honestly how much you believe it.

---

## Knowledge Check

Twenty questions.

1. What is the single most valuable thing to know about a dataset, and why is it
   not in the dataset? [{{ch:ds-what-it-is}}]

2. Distinguish data drift from concept drift. Which is harder to detect, and
   why? [{{ch:ds-what-it-is}}]

3. Selection bias does not shrink with sample size. Explain why, referring to
   {{eq:selection-bias-magnitude}}. [{{ch:ds-collection}}]

4. You have 2 million events from 40,000 users. Why is that not 2 million
   independent observations, and by roughly how much are your intervals wrong?
   [{{ch:ds-collection}}]

5. Which kinds of schema drift are dangerous, and what makes them so?
   [{{ch:ds-collection}}]

6. Classify MCAR, MAR and MNAR. Which can you distinguish from the data alone?
   [{{ch:ds-cleaning}}]

7. Mean-imputing 30% of a variable attenuates its correlations by what factor?
   [{{ch:ds-cleaning}}]

8. Why does the z-score rule fail to detect outliers when there are several of
   them? [{{ch:ds-cleaning}}]

9. Give three things a single histogram can tell you. [{{ch:ds-eda}}]

10. A feature has Pearson 0.01, Spearman 0.02, and high mutual information with
    the target. What is going on, and what would a correlation screen do?
    [{{ch:ds-eda}}]

11. State the fundamental problem of causal inference. [{{ch:ds-causation}}]

12. Why does randomisation work even for confounders you have not measured?
    [{{ch:ds-causation}}]

13. In Simpson's paradox, which answer is correct — pooled or subgroup — and
    what determines it? [{{ch:ds-causation}}]

14. What is a collider, and what happens if you adjust for one?
    [{{ch:ds-causation}}]

15. Why must the randomisation unit equal the analysis unit?
    [{{ch:ds-experiments}}]

16. Explain why peeking inflates the false-positive rate. Is each individual
    test valid? [{{ch:ds-experiments}}]

17. Why must target encoding be computed out of fold during training but not at
    prediction time? [{{ch:ds-feature-eng}}]

18. Name the four leakage mechanisms and give an example of each.
    [{{ch:ds-leakage}}]

19. Why is precision base-rate dependent when sensitivity is not? What does that
    imply for deploying a model in a new market? [{{ch:ds-leakage}}]

20. Why is random k-fold invalid for time series, and what replaces it?
    [{{ch:ds-timeseries}}]

**Bonus.** Why can the SVD not be applied directly to a ratings matrix, and why
does the feedback loop mean offline evaluation favours the incumbent model?
[{{ch:ds-recsys}}]

---

## Practical Assignment

### Audit a pipeline, then rebuild it

You are given — or you write, then hand to a colleague, then get back — an
end-to-end analysis with defects planted in it. Your job is to find them all,
document each, and produce a corrected version whose numbers you can defend.

**Part 1 — Build the flawed version (or use one you are given).**

Write a churn analysis containing at least eight of the following, without
marking them:

- a feature populated only after the outcome
- a rolling feature computed without shifting
- a scaler or imputer fitted before the split
- feature selection performed on the full dataset
- a random split on data with repeated users
- a random split on time-ordered data
- target encoding computed in-fold
- oversampling applied before splitting
- a join that inflates the row count
- a correlation reported as a causal effect
- a comparison that reverses on disaggregation
- accuracy reported on a 2% positive rate

**Part 2 — Audit it.** Produce a findings report, one row per defect: the
mechanism, how you detected it, the severity, and your estimate of how much it
moves the headline number. Use the checks of {{ch:ds-leakage}} in cost order,
and include the shuffled-target test.

**Part 3 — Rebuild it correctly.**

- Validated ingestion with a data contract ({{ch:ds-collection}}).
- Cleaning as a fitted transformation, with a documented reason for every
  decision ({{ch:ds-cleaning}}).
- A systematic EDA pass with findings ranked by severity ({{ch:ds-eda}}).
- Features that will exist at prediction time, with target encoding out of fold
  ({{ch:ds-feature-eng}}).
- Grouped and time-aware validation ({{ch:ds-leakage}}, {{ch:ds-timeseries}}).
- Metrics appropriate to the class balance, with confidence intervals
  ({{ch:math-inference}}).
- Every causal statement either supported by randomisation or explicitly labelled
  as associational ({{ch:ds-causation}}).

**Part 4 — Report the difference.** State the headline number before and after,
and attribute the gap to specific defects. This is the deliverable: not the
model, but a defensible account of why the first number was wrong.

**Acceptance criteria**

- Every planted defect found, with its mechanism named.
- The shuffled-target test returns chance performance on the corrected pipeline.
- Row counts asserted around every join.
- No causal language attached to an observational estimate.
- A written statement of what the corrected analysis *cannot* conclude.

---

## Advanced Challenge

### Make a causal claim from observational data, and say how much you believe it

Take a dataset where a randomised experiment is impossible — a public dataset,
or logs from a system you know — and attempt to estimate a causal effect.

**Part A — Draw the causal graph first.** Before touching the data, write down
what you believe causes what. Identify, for every available covariate, whether
it is a confounder, a mediator, a collider, or an outcome-only predictor
({{tbl:adjustment-rules}}). Commit to an adjustment set and record why.

**Part B — Estimate.** Produce a naive estimate, a stratified estimate, and a
regression-adjusted estimate. Where they differ, explain which assumption
accounts for the difference.

**Part C — Attack your own estimate.** For each of the following, either rule it
out with an argument or state that you cannot: an unmeasured confounder; reverse
causation; a collider among your adjustment set; selection into the sample; a
Simpson reversal on a variable you did not consider.

**Part D — Bound the damage.** Using {{eq:confounding-bias}}, compute how strong
an unmeasured confounder would have to be to explain away your entire effect.
State that number. It is the single most informative sentence in the analysis,
and it is what separates a defensible observational claim from an indefensible
one.

**Part E — Say what would settle it.** Design the experiment that would answer
the question properly, including the randomisation unit, sample size and
duration ({{ch:ds-experiments}}). State what it would cost.

**Deliverable.** A report whose conclusion is calibrated: not "X causes Y", but
"the association is this large, it survives adjustment for these variables, and
an unmeasured confounder of this strength would be needed to explain it away."

---

## Interview Preparation

### Junior

1. What is the difference between correlation and causation?
2. How do you handle missing values?
3. What is an outlier and what should you do about it?
4. What is A/B testing and why randomise?
5. Why is accuracy a bad metric for rare events?
6. What is data leakage?
7. Why can you not use random cross-validation on time series?

### Mid-level

8. Explain Simpson's paradox and how you would decide which answer to report.
9. A colleague's model has 0.99 AUC. What do you check?
10. When should you adjust for a variable and when should you not?
11. Explain why peeking at an experiment inflates false positives.
12. How would you target-encode a high-cardinality column safely?
13. Your test shows a 2% conversion lift and 300 ms extra latency. What now?
14. Explain the difference between MCAR, MAR and MNAR and why it matters.
15. What is training-serving skew and how do you prevent it?

### Senior

16. Design an experiment for a two-sided marketplace. What breaks?
17. You have 10 million rows from 50,000 users. How many independent
    observations do you have, and what does that do to your intervals?
18. How would you detect leakage in a pipeline you did not write?
19. How would you decide whether an observational estimate is good enough to act
    on?
20. Explain the recommender feedback loop and its consequences for evaluation.
21. Your model's offline metrics improved and the A/B test was flat. Give four
    explanations and how you would distinguish them.
22. When is more data not the answer?

### Systems and judgement

23. An agent produces a complete analysis in thirty seconds. What do you check
    before believing it, and in what order?
24. A stakeholder wants to act on a correlation you consider confounded. How do
    you handle it?
25. You inherit a churn model in production with no documentation. Describe your
    first week.

---

## Before moving on

You are ready for {{part:4}} when you can, without reference:

- Ask the right questions about a dataset's provenance before analysing it.
- Justify every cleaning decision and implement it as a fitted transformation.
- Run a systematic EDA and rank what it finds.
- Construct Simpson's paradox and explain why the data cannot resolve it.
- Decide what to adjust for from a causal graph rather than from correlations.
- Design and analyse an experiment without peeking, with the right unit.
- Find leakage in an unfamiliar pipeline, including with the shuffled-target
  test.
- Validate a time-ordered model without training on the future.

{{part:4}} builds models on the data this part prepared. It assumes you can tell
whether the resulting number means anything.

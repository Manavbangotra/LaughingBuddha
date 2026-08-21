---
id: part-04-assessment
status: final
---

## Knowledge Check

Answer without looking anything up. The answers are in the chapters named at the
end of each question.

**Foundations**

1. State the no-free-lunch result precisely, and then state what it does *not*
   imply about choosing an algorithm. ({{ch:ml-what-it-is}})

2. A model achieves zero training error. What have you learned about its
   generalisation? ({{ch:ml-what-it-is}})

3. Why is `inv(X.T @ X) @ X.T @ y` wrong, and what should you write instead?
   Quantify the failure. ({{ch:ml-linear-regression}})

4. Multicollinearity is present with VIF above 200. Under what circumstances is
   this fatal, and under what circumstances is it irrelevant?
   ({{ch:ml-linear-regression}})

5. Ridge and lasso are both applied to a dataset where five of sixty features
   are genuinely predictive with large coefficients. Predict which wins and
   explain the mechanism. ({{ch:ml-linear-regression}})

6. Cross-entropy is not a design choice. What is it?
   ({{ch:ml-logistic}})

7. In the logistic gradient $\mat{X}\T(\vec{p}-\vec{y})/N$, a factor cancelled.
   Which one, where did each half come from, and what would break if it did
   not cancel? ({{ch:ml-logistic}})

8. Your logistic regression reports a convergence warning and coefficients of
   magnitude $10^{4}$. Give two possible causes, ranked, and the one-line fix
   for the benign one. ({{ch:ml-logistic}})

**Evaluation**

9. Write the bias-variance-noise decomposition and say which term more data
   reduces. ({{ch:ml-metrics}})

10. A learning curve shows training and validation error meeting at 0.31 with
    the irreducible noise at 0.09. What do you do next, and what would be a
    waste of money? ({{ch:ml-metrics}})

11. Two models have identical ROC-AUC. One has ECE 0.004 and the other 0.25.
    Name two downstream uses that still work with the second, and two that
    break. ({{ch:ml-metrics}})

12. Explain why a team that evaluated 400 configurations on one validation split
    should be trusted *less*, and give the approximate size of the effect.
    ({{ch:ml-metrics}})

13. Nested cross-validation is usually described as unbiased. In which direction
    is it actually biased, and why? ({{ch:ml-metrics}})

**Algorithms**

14. Give the quantitative statement of the curse of dimensionality, and explain
    why 1536-dimensional embeddings work as a k-NN space anyway.
    ({{ch:ml-knn-nb}})

15. Naive Bayes returns 0.99999 on a problem where it is right 81% of the time.
    Explain the mechanism, and say what it costs you and what it does not.
    ({{ch:ml-knn-nb}})

16. Why does setting `min_impurity_decrease=0.01` risk destroying a tree, and
    what should you do instead? ({{ch:ml-trees}})

17. `feature_importances_` ranks `customer_id` first. What has happened, and
    what should you use? ({{ch:ml-trees}})

18. Write the variance of an average of $B$ correlated estimators and identify
    the term that more trees cannot reduce. ({{ch:ml-forests}})

19. Bagging needs deep trees and boosting needs shallow ones. Give the
    one-sentence reason for each. ({{ch:ml-forests}}, {{ch:ml-boosting}})

20. Out-of-bag error and 5-fold CV both come out above the held-out score. Give
    the separate reason for each. ({{ch:ml-forests}})

21. In what sense is gradient boosting gradient descent? Be precise about what
    the parameter is and what the step is. ({{ch:ml-boosting}})

22. `max_depth=1` in a boosted model. What class of function can the ensemble
    represent, and what can it never represent? ({{ch:ml-boosting}})

23. Why does early stopping on log loss stop earlier than early stopping on AUC,
    and what does that tell you about what the extra rounds were doing?
    ({{ch:ml-boosting}})

24. An SVM's $\ell_2$ penalty is not a regularisation term added to an
    objective. What is it? ({{ch:ml-svm}})

25. Explain the kernel trick in three sentences, and state what makes it
    complete rather than merely convenient. ({{ch:ml-svm}})

26. Why can k-means never separate two parallel elongated bands? Answer with the
    equation. ({{ch:ml-clustering}})

27. The silhouette prefers k-means' answer to DBSCAN's on interlocking
    crescents, and DBSCAN's is correct. Explain, and give the general principle.
    ({{ch:ml-clustering}})

28. PCA retains 99% of the variance and your model gets worse. Explain how, and
    name the supervised alternative. ({{ch:ml-pca}})

29. When is it safe to write "PC3 represents customer affluence", and how would
    you check? ({{ch:ml-pca}})

30. A detector has ROC-AUC 0.97 at a 0.1% anomaly rate. Why is that nearly
    uninformative, and what two numbers would you ask for instead?
    ({{ch:ml-anomaly}})

## Practical Assignment

**Build a complete tabular pipeline, and justify every choice by measurement.**

Use a dataset of your own with at least 5,000 rows, a binary target, and a mix
of numeric and categorical features. Public options if you have none: the UCI
Adult income dataset, the Give Me Some Credit dataset, or the Telco churn
dataset.

**Deliverable: one notebook or script and a one-page written summary.**

*Part 1 — the honest baseline.*

1. Split three ways before touching anything: train, validation, test. Write
   down the date, or the entity, or whatever grouping makes the split honest
   ({{ch:ds-leakage}}), and say why a random split would or would not have been
   acceptable.
2. Report the majority-class rate. This is the number every model must beat.
3. Fit logistic regression with default settings and report PR-AUC, ROC-AUC,
   ECE and precision at the top decile. Four numbers, not one.

*Part 2 — the comparison.*

4. Fit, tune on validation only, and report the same four metrics for: a
   regularised linear model, k-NN, a single tree, a random forest, and gradient
   boosting.
5. Produce the learning curve for the two best models and state, with the
   evidence, whether more data would help.
6. For every model, report the wall-clock fit time and the per-prediction
   latency. State which model you would actually deploy and why — and if it is
   not the most accurate one, say so explicitly.

*Part 3 — the parts people skip.*

7. Compute permutation importance on the validation set for your best model, and
   compare it against the built-in `feature_importances_`. Explain any
   disagreement ({{ch:ml-trees}}).
8. Produce a reliability diagram with binomial error bars. If your model is
   miscalibrated, fix it with temperature scaling or isotonic regression and
   show that accuracy is unchanged.
9. State the cost of a false positive and a false negative in your problem's
   units, derive the optimal threshold, and report what it costs against the
   0.5 default ({{ch:ml-logistic}}).
10. Touch the test set exactly once, at the end, and report the gap between your
    validation and test numbers. If the gap is large, say what you think caused
    it.

*Part 4 — the write-up.*

One page. What you built, what you measured, which choice surprised you, and one
thing you would do differently with another week. Include at least one plot and
at most three.

**Marking yourself honestly:** the assignment is passed if a colleague reading
the summary could reproduce your decision without rerunning anything, and if
every claim in it is traceable to a number in the notebook.

## Advanced Challenge

**Reproduce the tabular question of {{ch:ml-boosting}} and take a position on
it.**

{{cite:grinsztajn2022}} identifies three mechanisms behind tree-based dominance
on tabular data: robustness to uninformative features, invariance to feature
rotation, and the ability to fit irregular target functions.
{{cite:hollmann2025}} reports a tabular foundation model beating tuned ensembles
below roughly ten thousand rows.

Design and run an experiment that tests the three mechanisms directly rather
than benchmarking end to end.

1. **Uninformative features.** Take a dataset where boosting and a neural
   network are close. Add $k$ pure-noise columns for $k \in \{0, 10, 50, 200\}$
   and plot both models' accuracy against $k$. Does the gap widen as the
   mechanism predicts?

2. **Rotation.** Apply a random orthogonal rotation to the feature matrix — this
   destroys no information and makes the axes meaningless. Measure both models
   before and after. {{ch:ml-trees}} measured a seventeen-fold error increase
   for a single tree on a rotated diagonal boundary; how much does an ensemble
   recover?

3. **Irregular targets.** Construct two synthetic targets with matched
   signal-to-noise, one smooth and one with sharp thresholds and
   discontinuities. Predict which model wins each before running it, then run
   it.

4. **The sample-size axis.** Repeat your best experiment at $N \in \{200, 1000,
   5000, 20000\}$ and locate the crossover, if there is one.

5. **Take a position.** In 500 words, state what you now believe about when to
   use gradient boosting on tabular data in 2026, and what evidence would change
   your mind. Distinguish what you measured from what you are extrapolating.

**Stretch:** install a tabular foundation model and add it to your comparison
under 10,000 rows. If you do, report its constraints — maximum rows, columns and
classes — because they are part of the answer.

## Interview Preparation

The questions people are actually asked, with what a strong answer contains.

**"Walk me through how you would approach a new tabular classification
problem."**

A strong answer is a sequence, not a list of algorithms: establish the honest
split first; state the majority-class baseline; fit logistic regression before
anything else; move to gradient boosting; tune on validation only; pick a metric
from the cost structure; check calibration; and touch the test set once. Naming
the baseline first is what separates a practitioner from someone who has read
about models.

**"Explain the bias-variance trade-off."**

Give the decomposition, then say what each term responds to — capacity for both,
data for variance only. The follow-up is almost always "so would more data help
here?", and the answer is a learning curve, not an opinion. Mention that the
classical U-shape does not describe modern deep networks if you want to signal
depth; do not lead with it.

**"When would you use a random forest over gradient boosting?"**

When you need a result today with no tuning; when you want free validation from
out-of-bag error; when the data is small enough that boosting's extra capacity
is a liability; when you need to parallelise across machines. And the honest
addendum: at medium and large scale on tabular data, tuned boosting usually
wins, so the question is what you are trading for.

**"Your model has 99% accuracy. Are you happy?"**

The expected answer is "what is the base rate", and the strong answer continues:
which errors cost what, what is precision at the operating point, and is the
model calibrated. If the interviewer says "1% positive", the correct response is
that the model may be doing nothing at all.

**"How do you know your model is not overfitting?"**

Weak: "I used cross-validation." Strong: an untouched test set, the number of
configurations evaluated, the gap between validation and test, and — if the
search was wide and the data small — nested CV, with the caveat that it is
conservative rather than unbiased.

**"Explain regularisation to a non-technical stakeholder."**

The best short version is about memorising versus learning: a model with enough
freedom will memorise the training data including its noise, and regularisation
is how you keep it from doing that. Then be ready to give the technical version
— the penalty term, and $\lambda$ chosen by cross-validation — because they will
ask.

**"What is the kernel trick?"**

The dual only needs inner products; a kernel computes an inner product in a
space you never construct; the RBF kernel's space is infinite-dimensional and
costs one exponential. If you can add *why nothing is lost* — the representer
theorem, because the solution always lies in the span of the training data —
that is a distinguishing answer.

**"How would you detect fraud with no labels?"**

Name the three problem types and pick the right one. Then: isolation forest as
the default, because it scales and needs no distance computation; scores rather
than a contamination guess; PR-AUC and precision@$k$ rather than accuracy or
ROC-AUC; and the operating point set by how many alerts an analyst can review.
Mentioning that you would rather have labels — because then it is imbalanced
classification and boosting beats everything here — shows you know when not to
use the fancy method.

**"Your model works in validation and fails in production. What happened?"**

Have a ranked list ready: leakage in the training features; a split that was not
honest for this data; distribution drift; a training/serving skew in the
preprocessing; and the selection optimism of having tried many configurations.
The ordering matters more than the length — leakage first, because it is the
most common and the most embarrassing.

**"Which is better, PCA or feature selection?"**

The trap is answering. The right response distinguishes: PCA for compression,
decorrelation and visualisation; feature selection when the truth is sparse or
you need to explain the model; and neither, if you have labels and are trying to
find what predicts — use the labels. Mention that PCA maximises variance and
variance is not importance, and give the counterexample if pressed.

**Two questions worth preparing that are less often asked but distinguish
strongly:**

*"Why is your metric the right one?"* — most candidates never justify a metric.

*"What did you decide not to do?"* — scope discipline is the scarcer skill, and
a good answer names a technique, says why it did not earn its complexity, and
gives the condition under which you would revisit it.

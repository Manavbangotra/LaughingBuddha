---
id: part-05-assessment
status: final
---

## Knowledge Check

Answer without looking anything up. The chapter is named at the end of each
question.

**Honest evaluation**

1. A random split assumes something about the rows. State it, and give two
   ways real data violates it. ({{ch:mle-splits}})

2. Your data has an intraclass correlation of 0.4 across customers. What does
   a random split do to your error estimate, and what does it do to your
   confidence intervals? Give both, quantitatively. ({{ch:mle-splits}})

3. A feature is a 14-day trailing mean and the label resolves 60 days later.
   How long must the embargo be, and which of the two constraints binds?
   ({{ch:mle-splits}})

4. Four people each evaluated a handful of models on the test set over six
   months. Why is that a problem, and roughly how large is it?
   ({{ch:mle-splits}})

5. A split was correct when written and is now leaking. Name the mechanism
   and the defence. ({{ch:mle-splits}})

**Search**

6. Why does random search beat grid search? Give the geometric argument, not
   a benchmark result. ({{ch:mle-hpo}})

7. How many random trials give a 95% chance of landing in the top 5% of a
   search space, and on what does that number depend? ({{ch:mle-hpo}})

8. Explain why every rung of successive halving costs the same, and what that
   buys. ({{ch:mle-hpo}})

9. A colleague attributes their speed-up to Bayesian optimisation. What
   question do you ask, and what did the measurement in
   {{ch:mle-hpo}} actually find?

10. Why is the best score from a 2,000-trial search not an estimate of
    anything, and what one extra step fixes it? ({{ch:mle-hpo}})

**Pipelines**

11. Name the four causes of training/serving skew. Which one does a shared
    feature definition eliminate, and what remains?
    ({{ch:mle-pipelines}})

12. What is the difference between event time and availability time, and why
    does it matter to an as-of join? ({{ch:mle-pipelines}})

13. Your five cross-validation folds all agree closely. Why is that not
    evidence that your join is correct? ({{ch:mle-pipelines}})

14. A scaler fitted on all the data leaks $O(1/n)$; a target encoder leaks
    $O(1)$. Explain the difference and say what determines which case you are
    in. ({{ch:mle-pipelines}})

**Reproducibility**

15. Distinguish experiment tracking from reproducibility, and give a situation
    where a team plainly has one and not the other.
    ({{ch:mle-reproducibility}})

16. Why is one global seed insufficient, and what replaces it?
    ({{ch:mle-reproducibility}})

17. Floating-point nondeterminism was described as a tie-breaking mechanism
    rather than a perturbation mechanism. Explain what that means and what
    the measurement showed. ({{ch:mle-reproducibility}})

18. Which level of reproducibility would you target for a credit model, and
    what does it cost? ({{ch:mle-reproducibility}})

19. What number makes a reported improvement meaningful, and how do you get
    it? ({{ch:mle-reproducibility}})

**Deployment and monitoring**

20. Name six things a registry entry must carry besides the weights, and say
    which one people most often omit. ({{ch:mle-registry}})

21. Why non-inferiority rather than superiority, and where should the margin
    come from? ({{ch:mle-registry}})

22. The ML Test Score aggregates as a minimum. Give the failure model that
    justifies it. ({{ch:mle-registry}})

23. A canary has served 1% of traffic for an hour and the conversion rate
    looks unchanged. What have you learned? ({{ch:mle-registry}})

24. Give three effects of a prediction that a rollback does not undo.
    ({{ch:mle-registry}})

25. Which of covariate shift, label shift and concept drift is invisible to
    input monitoring, and why? ({{ch:mle-drift}})

26. Why must the conjunction alert be asymmetric? ({{ch:mle-drift}})

27. Thirty features tested daily at $\alpha = 0.05$. How often does a
    completely stable system alert, and what is the one-line fix?
    ({{ch:mle-drift}})

28. The measurement found the conventional PSI threshold of 0.25 never firing
    — on stable data or on a genuine shift. What does that tell you, and what
    should you do instead? ({{ch:mle-drift}})

29. Your monitor fires. Name five possible responses and the condition under
    which each is right. ({{ch:mle-drift}})

30. Why is retraining not automatically the right response to drift?
    ({{ch:mle-drift}})

## Practical Assignment

**Take a model you have already built and make it deployable.**

Use anything from Part IV's assignment, or any model of your own. The point is
not to improve it — it is to build everything around it that Part IV ignored.

**Deliverable: a repository and a one-page operations note.**

*Part 1 — an honest number.*

1. Identify the grouping and time structure in your data, and write the split
   as a function with an assertion layer that checks its own invariants.
   Deliberately break one invariant and show the assertion firing.
2. Estimate the intraclass correlation across your grouping variable. Report
   the effective sample size and how much narrower your naive confidence
   intervals were.
3. Establish a holdout with a ledger. Report $K_{\text{total}}$ at the end of
   the assignment and the optimism correction it implies.

*Part 2 — a defensible search.*

4. Define your search space with explicit scales, and justify each log scale in
   one sentence.
5. Run the search twice, once with a pruner and once without, at matched
   budget. Report both the best score and the total compute.
6. Apply a stopping rule based on your measured fold-to-fold standard error,
   and say at which trial it would have fired.
7. Re-evaluate the winning configuration on a fresh split. Report both numbers.

*Part 3 — a pipeline that cannot lie.*

8. Write your feature computation once and call it from both a batch path and
   a single-row path. Add a parity test that fails if they disagree.
9. If your data has any time structure, implement an as-of join using
   availability timestamps and report the staleness distribution.
10. Add a contract that checks schema, ranges, null rates and row counts
    against a reference. Feed it four deliberately broken batches and show it
    catching each.

*Part 4 — a run you can rebuild.*

11. Capture the full run record: code commit with a clean-tree check, content
    hashes of every input, environment, resolved configuration including
    defaults, and per-component seeds.
12. Measure your pipeline's run-to-run variance over at least ten seeds.
    Record it, and use it everywhere you later report a difference.
13. Write a verifier that re-checks the record and blocks if anything no
    longer resolves. Demonstrate it blocking after you append a row to your
    input data.

*Part 5 — the handoff and the aftermath.*

14. Write a registry entry containing everything from
    {{eq:registry-entry}}, including per-slice evaluation for at least three
    slices you chose in advance.
15. Implement a promotion gate with at least six checks, with the
    non-inferiority margin derived from your measured noise. Run it against
    three candidates, at least one of which must fail for a reason other than
    the aggregate metric.
16. Implement a monitor with thresholds calibrated from a stable period at a
    stated false-alarm rate, an asymmetric alerting rule, and hysteresis.
    Simulate at least three incident types and report what each rule catches.

*Part 6 — the operations note.*

One page, written for whoever is on call. What the model does, what it must
not be used for, what each alert means, what to do about it, and how to roll
back. If it takes longer than a page, the system is too complicated.

**Marking yourself honestly:** the assignment is passed if someone else could
take your repository, retrain the model, verify they got the same thing, deploy
it behind the gate, and know what to do when it pages — without asking you
anything.

## Advanced Challenge

**Build a skew detector that finds a bug nobody planted.**

Every measurement in this part used a bug that was constructed on purpose,
which is the easy case: you knew what to look for. The hard version is finding
one you did not plant.

1. **Instrument for parity.** Take any real system you have access to — your
   own project, an open-source ML repository, or a public pipeline — and log,
   for a sample of decisions, the feature values computed by the training path
   and by the serving path, keyed by decision id.

2. **Build the comparison.** Report per-feature disagreement rates, the
   distribution of relative differences, and the staleness of each value. Set
   a tolerance and justify it.

3. **Hunt for time travel specifically.** For every feature with a timestamp,
   check that the value used was available strictly before the decision time.
   Report any feature where the check cannot be performed, because "we do not
   record availability time" is itself a finding.

4. **Estimate what a leak would be worth.** For each feature, measure how much
   validation performance would change if it were computed from one day later.
   That is the sensitivity of your pipeline to a one-day timestamp error, and
   it tells you which joins deserve scrutiny.

5. **Write it up as an incident report** for a bug that has not happened yet:
   what would break, how you would notice, how long it would take to detect,
   and what it would cost per day undetected. Use {{ch:mle-registry}}'s
   incident-cost arithmetic.

**Stretch:** repeat step 4 for staleness rather than time travel, and produce
a ranked list of features by how much a one-hour cache delay would cost. That
ranking is the input to deciding what needs a real-time path and what does
not — which is the feature-store adoption question of {{ch:mle-pipelines}},
answered with evidence rather than aspiration.

## Interview Preparation

The questions that get asked about this material, and what a strong answer
contains.

**"How would you split this data?"**

Almost always asked with a dataset that has hidden group structure. The strong
answer asks what a row is before answering, then asks what production will look
like, and only then proposes a split. Naming the intraclass correlation as
something you would estimate — rather than guessing — is what distinguishes a
practitioner. Mentioning the embargo for time-structured data is a bonus that
few candidates offer.

**"Your validation AUC is 0.94 and production is 0.71. What happened?"**

The single most common ML engineering interview question. A strong answer is a
ranked list, not a guess: leakage in features or joins first, because it is the
most common; then a split that was not honest for this data; then training/
serving skew; then distribution shift; then selection optimism from a wide
search. Saying which you would check first *and why it is first* matters more
than the length of the list.

**"How do you tune hyperparameters?"**

Weak: "grid search" or "Optuna". Strong: random or model-based sampling with a
pruner, log scales for anything multiplicative, a budget set in advance, a
stopping rule based on the fold-to-fold standard error, and a final
re-evaluation of the winner on a fresh split. If you can add that the pruner
and the sampler are separate components with comparable contributions, you are
signalling that you have measured this rather than read about it.

**"What is training/serving skew and how do you prevent it?"**

The trap is answering "use a feature store". The strong answer names the four
causes, says that a shared definition fixes exactly one of them, and describes
the parity test that would catch the others. Point-in-time correctness with
availability timestamps is the detail that shows depth.

**"How do you know a model is reproducible?"**

Weak: "we set a seed." Strong: distinguish tracking from reproducibility, name
what the run record must contain, and mention that you measure run-to-run
variance rather than assuming determinism. If asked how far you would go, the
right answer is a question back — what happens if we cannot reproduce it?

**"What goes in a model registry?"**

The strong answer starts with "the pipeline", because that is what people
forget, and includes per-slice evaluation and the output contract. If you can
say why the gate should require non-inferiority rather than superiority, and
where the margin comes from, that is a distinguishing answer.

**"How would you monitor this model in production?"**

The question that separates people who have operated a model from people who
have trained one. Ask when the labels arrive — it is the first question, and
most candidates never ask it. Then: input distribution, prediction
distribution, a validated proxy, and the true metric when it lands. Then the
alerting rule, and the reason it must be asymmetric. Then what each alert
means and what you would do.

**"When would you retrain?"**

The trap is "when drift is detected". The strong answer distinguishes the
causes: a broken pipeline must be fixed upstream, because retraining would
learn the bug; a genuine change in the world justifies a retrain; and either
way the retrained model is a candidate that must pass the same gate. Mentioning
scheduled retraining as a way of keeping the path exercised shows operational
experience.

**"Your monitoring alerts three times a week and the model has always been
fine. What do you do?"**

Not a threshold question, and answering it as one is the trap. The strong
answer diagnoses: are these multiple-testing artefacts, is the drift real but
harmless, is there no downstream signal in the rule at all? Then proposes the
conjunction, and notes that raising the threshold trades false alarms for
missed detections one-for-one while the conjunction does not.

**Two questions worth preparing that are rarely asked and distinguish
strongly:**

*"What is the run-to-run variance of your pipeline?"* — almost nobody has
measured it, and every reported improvement depends on it.

*"How many times has your test set been evaluated?"* — the answer is almost
always "I don't know", and knowing that you should know is the point.

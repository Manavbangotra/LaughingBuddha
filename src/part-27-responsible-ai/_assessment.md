---
id: part-27-assessment
status: draft
---

## How to use this

Four sections. The knowledge check takes about ninety minutes. The assignment is an
**instrument audit**, because this part's rule was to name the question each instrument answers
and put it next to the number — and the commonest finding is that a report's headline quantity
is defensible, precise, and about something other than what the reader believes.

No answers are provided. Every question is answerable from the chapters.

## Knowledge check

**Bias and fairness**

1. State {{eq:three-fairness-criteria-cannot-hold-together}}. Name the two special cases in
   which the conflict vanishes, and say why a deployed system is in neither.
2. At base rates of 34% and 13% with identical score quality, a shared threshold gives PPV
   0.740 against 0.453. What happens to the true-positive rates if you equalise PPV instead?
3. Why is the size of the compromise a property of the population rather than of the model?
4. What d-prime would be needed to close the conflict by improving the model, and what AUC is
   that? What is a typical deployed value?
5. State {{eq:disparity-decomposes-and-only-some-parts-are-fixable}}. Which component has no
   model-side remedy, and how large is it here?
6. Threshold adjustment returns 0.375 per unit of effort and data collection 0.020. Why is the
   cheap remedy the one that is usually skipped?
7. State {{eq:tokenisation-imposes-a-cost-disparity-before-any-model-runs}}. Why would this
   disparity appear in no fairness report?

**Explainability and interpretability**

8. State {{eq:attribution-is-not-an-intervention-effect}}. Under what condition do the two
   coincide?
9. At correlation 0.97, attribution is 0.211 and the intervention effect is 0.012. Explain the
   factor of 17.2 in terms of what each quantity is computed over.
10. What does {{cite:lundberg2017shap}}'s uniqueness theorem actually assert, and over what?
11. State {{eq:local-fidelity-does-not-extend}}. What is the surrogate's fit at the distance
    where policy is written?
12. Attributions rank by contribution; users need actionability. Give an example where the
    top-attributed feature is useless to the user.
13. A generated explanation accounts for 36% of what moved the decision. What is in the other
    64%, and how often is the demographic cue named?
14. Reading detects 4% of unfaithful explanations and a swapped re-run detects 62%. Why is
    reading so weak?
15. State {{eq:an-explanation-serves-one-audience}}. Why does every single-artefact design have
    a worst column of 0.24 or below?

**Privacy, data governance and copyright**

16. State {{eq:epsilon-bounds-the-posterior-shift}}. What does epsilon = 8 permit, starting from
    a 1% prior?
17. State {{eq:privacy-budget-composes-across-queries}}. Twenty queries at epsilon 0.5: what is
    the dataset's epsilon?
18. Reported spend is 0.5 and actual spend is 8,833.6. Name the five consumers that are not
    counted.
19. Epsilon 8 to 3 costs 8 utility points for a 148× tighter bound; 3 to 1 costs 12 more for
    20×. Where would you stop, and what does the answer depend on?
20. State {{eq:deletion-is-a-product-over-derived-artefacts}}. Why is the reported completeness
    0.3231 and the honest one exactly zero?
21. Retraining per deletion request costs $1,400,000; batched annually, $22.58 and a year of
    latency. Why is neither a deletion guarantee?
22. State {{eq:copyright-exposure-is-the-memorisation-rate}}. Why does DP score 0.93 on privacy
    and 0.44 on copyright?

**Regulation and risk management**

23. State {{eq:compliance-cost-is-a-step-function}}. What is the step from limited to high risk,
    in money and in weeks?
24. Four readers reach three conclusions about one system. Which classification factors are
    judgements rather than measurements?
25. State {{eq:tier-boundaries-create-design-incentives}}. What is the boundary worth here, and
    how many of the five tier-reducing moves change a label rather than a risk?
26. Why does the tier account for roughly 98% of the cost variance and the system's complexity
    for the rest?
27. State {{eq:most-compliance-evidence-is-engineering-you-already-do}}. Which two obligations
    have the largest gaps, and what do they have in common?
28. Why is the correct planning unit an artefact rather than an obligation?
29. State {{eq:evidence-must-be-contemporaneous}}. Which of the six critical facts is
    recoverable, and why is it the exception?

**Human oversight**

30. State {{eq:review-helps-only-when-catch-exceeds-override-odds}}. Derive it from the team
    accuracy expression.
31. The team beats the model in 4 of 5 tasks and the better of its two members in 2. Explain why
    those are different claims.
32. The fraud queue's bar is 15.7 against a reviewer ratio of 5.5. What should be done, and what
    is usually done?
33. Break-even is at 0.921 model accuracy. What happens to a review process after a model
    upgrade, and what in the pipeline signals it?
34. State {{eq:an-explanation-raises-confidence-faster-than-accuracy}}. Why does a plausible
    wrong explanation lower *both* the catch rate and the override rate?
35. Why is an uncalibrated confidence score worse than no confidence score?
36. Routing the bottom 20% beats reviewing everything; routing the bottom 5% does not. Why?
37. Perfect routing is worth 4.2 points and no more. What is the binding constraint, and what
    moves it?
38. State {{eq:oversight-is-a-conjunction-of-preconditions}}. Which factor binds in five of six
    arrangements, and which one do organisations actually design for?
39. An appeal process scores the best per-item quality in the table. Why is it not the system's
    oversight?
40. Complete verification takes 16.0 minutes against a 90-second budget. Why does a 30% budget
    increase buy nothing?
41. State {{eq:reviewers-bear-the-cost-of-rejecting-not-approving}}. Derive the 94% threshold.
42. Which two interventions move the threshold, and what do they have in common that the other
    three lack?

## Assignment: an instrument audit

Take an AI system you are responsible for or can inspect, and a report about it that someone
actually reads — a fairness review, a model card, a privacy assessment, a compliance package, an
oversight design. Produce a written audit with six sections.

**1. The question gap.** For every headline number in the report, write two lines: the question
the instrument answers, and the question the reader believes it answers. Mark every row where
they differ. This is the part's rule applied to your own documents and it usually takes an hour.

**2. The fairness decision.** Compute your groups' base rates and score quality. Show the
criterion conflict at your numbers, state which criterion you are choosing, and publish the
violations of the other two. Then compute token fertility across the languages you serve, and
say whether that disparity is larger than anything in your fairness report.

**3. The second measurement.** Pick one of the part's three: re-run explanations with the input
order swapped and count how many change; meter the total epsilon spent by every consumer, not
the accounted queries; run a blind re-adjudication and estimate your reviewers' catch and
override rates. Each is an afternoon. Report the gap between what you measured and what your
current documents say.

**4. The evidence inventory.** Map the eleven conformity obligations onto the artefacts you
already produce and compute your coverage. Then, for each gap, determine whether the underlying
facts still persist. Partition the gap into a cost and a loss, and price both.

**5. The oversight arrangement.** Score every oversight arrangement in your system on authority,
information, time and incentive, multiply, and multiply again by coverage. Identify the minimum
factor. Then time the verification steps for one item type against your actual SLA, and compute
what fraction of the catchable errors your reviewers can reach.

**6. What you could not measure.** Every quantity this audit needed that you could not obtain.
As in the audits for {{part:22}} through {{part:26}}, this is the most valuable section, and
here it will usually be dominated by the harm-to-refusal cost ratio, the classification, and
your reviewers' actual catch and override rates.

Length: eight to twelve pages and a spreadsheet.

## Challenge problems

**A. The fairness criterion elicitation.** {{eq:three-fairness-criteria-cannot-hold-together}}
converts an argument into a decision and does not make it. Design a process that elicits the
choice from an organisation — from incident history, from revealed preference in past appeals,
from the relative cost of the two error types — run it, and report which criterion the
organisation is currently behaving as though it holds.

**B. The faithfulness test battery.** {{ch:rai-interpretability}} shows a swapped re-run detects
62% of unfaithful explanations against 4% for reading. Design a battery of cheap perturbation
tests, measure their union coverage per {{eq:coverage-is-a-union-not-a-sum}}, and determine the
smallest set that reaches 90%.

**C. The full epsilon meter.** Build the accounting that turns 0.5 into 8,833.6: instrument every
consumer of the sensitive dataset, including sweeps, ablations, dashboards and analysts. Report
what your real epsilon is and what it would cost to bring it under a publishable value.

**D. Manifest-only retention.** {{ch:rai-regulation}}'s {{sec:15-advanced-concepts}} proposes
retaining manifests and decisions rather than data, to satisfy the evidentiary and deletion
obligations simultaneously. Specify such a scheme concretely — what is retained, what is
destroyed, what reproducibility is given up — and test it against both obligations.

**E. The oversight value of out-of-distribution failure.** Everything in {{ch:rai-oversight}}
prices a reviewer against in-distribution accuracy, which on a good model recommends removing
them. Design an instrument that measures a reviewer's contribution on failures the evaluation
set does not contain, and say honestly whether it can be run before an incident.

**F. The whole-part residual.** Combine {{ch:rai-bias}}'s disparity floor,
{{ch:rai-privacy}}'s deletion completeness, {{ch:rai-regulation}}'s evidence coverage and
{{ch:rai-oversight}}'s effective oversight into a single responsible-AI residual for one system.
Which term dominates, and does the ranking of remedies change when they are optimised jointly
rather than separately?

## Interview preparation

Rehearse these until the answer is a structure rather than a recollection.

1. "Our model is fair — we equalised the false-positive rates." — Which criterion, then what the
   theorem says the other two are doing.
2. "SHAP says income is the biggest driver." — Attribution versus intervention, then the 17.2.
3. "We give every user an explanation of their decision." — Which audience, then 36%, then what
   the user can act on.
4. "We use differential privacy." — What epsilon, then who else queried the dataset.
5. "We honour deletion requests within thirty days." — Nine destinations, then the weights, then
   0.3231 against zero.
6. "We only train on data we have rights to." — What share has unresolved provenance, and when
   was that recorded.
7. "We're classified as limited risk." — Who read it that way, then the step, then get it in
   writing.
8. "Compliance is a separate workstream starting next quarter." — 62% already done, and 5 of 6
   facts expiring.
9. "A human reviews every decision." — Catch rate, override rate, and the model's odds of being
   right.
10. "We show reviewers the model's reasoning to help them." — Confidence 0.78, catch rate 0.19.
11. "Our reviewers reject 0.3% of items, so quality is high." — 94% threshold, then who bears
    the cost of saying no.
12. "We have a named accountable executive." — Authority 0.99, product 0.0067.

The pattern across all twelve: **name the question the instrument answers, ask what the second
measurement would show, and trace the constraint upstream to the decision that was cheap when it
was made.** That is the part in one sentence and the question worth having ready.

---
id: part-25-assessment
status: draft
---

## How to use this

Four sections. The knowledge check takes about ninety minutes. The assignment is an
**instrument audit**, because this part's rule was that every metric is a decision rule
with assumed parameters, and the commonest finding is that a number everyone trusts is
precise about a different question. The challenge problems are open-ended. The interview
section is what to rehearse.

No answers are provided. Every question is answerable from the chapters.

## Knowledge check

**Why evaluating AI is hard**

1. Show that raising a smooth capability curve to the power $k$ produces an apparent
   threshold, and say what determines where it sits.
2. State {{eq:metric-choice-manufactures-the-finding}}. The same capability "emerges" at 3B
   with a two-token answer and 20B with a twelve-token one. What changed?
3. Why can a metric that reads zero not support a funding decision? Give the extrapolation
   numbers.
4. A single-reference metric marks 99.4% of correct summaries wrong. Derive that from the
   size of the acceptable-answer space.
5. State {{eq:reference-scoring-penalises-valid-answers}}. Why does adding references not fix
   it, and for which tasks does it?
6. Why does execution-based grading escape the problem entirely? What does a test do that a
   reference does not?
7. At 81% annotator agreement, what is the highest correlation any metric can have with true
   quality? Derive it and state {{eq:agreement-caps-measurable-quality}}.

**Classical metrics revisited**

8. What cost ratio does F1 assume, and how would you recover it for your own score
   distribution?
9. At a 40:1 business ratio, following F1's threshold costs 141% more than the optimum. Where
   exactly is the loss?
10. Why is F1's advice worst on *balanced* problems? Explain the mechanism.
11. Construct two models with identical AUC that differ 2.3× in precision at a fixed recall.
    What structural difference produces it?
12. State {{eq:calibration-is-required-for-decisions}}. Why can AUC not detect
    miscalibration?
13. Recalibration is a monotone map. What does that guarantee, and what does it change?

**Benchmarks and their limits**

14. Contamination inflates scores and compresses gaps. Which of those is correctable, and
    why?
15. State {{eq:headroom-sets-benchmark-lifespan}}. Why does growing a benchmark not extend
    its life?
16. The same benchmark survives 18 years for coarse comparisons and 2 for generational
    progress. Explain.
17. A scenario scoring 0.470 has closed −23% of its headroom. What does that mean, and what
    two numbers were needed to say it?
18. Two models tie on an aggregate gain. Give three summaries of the same pair that
    disagree.
19. Why does optimising a suite aggregate reward polishing? State the Goodhart mechanism.
20. What was {{cite:liang2022helm}}'s 17.9% measuring, and why does it invalidate most
    published comparisons of its era?

**Human evaluation**

21. Why does label error behave as noise in a survey and as bias in an evaluation set?
22. Derive the gap compression $(1-2e)$ and compute it at $e = 0.14$.
23. State {{eq:budget-splits-between-items-and-annotators}}. Show that redundancy never
    reduces the labelling needed to detect a difference.
24. Under what constraint is redundant annotation required? What does it convert into what?
25. Decompose annotator disagreement into its four sources and rank the remedies by payback.
26. Why does randomising presentation order *lower* measured agreement, and why is that an
    improvement?
27. What two things does a double-labelled pilot produce that nothing else in the process
    does?

**LLM-as-a-judge**

28. A judge agrees with humans 82% of the time. What is the strongest claim that supports?
29. Express position bias in quality-equivalent units. What share of candidate pairs does a
    0.06 advantage decide?
30. The both-orders protocol raises accuracy from 74.6% to 88.7% and refuses 36% of pairs.
    Where does the gain come from? What happens if you break the ties?
31. State {{eq:self-preference-distorts-the-ranking}}. Why does the magnitude that matters
    keep shrinking?
32. Why does an ensemble of five samples from one model not fix self-preference?
33. State {{eq:optimising-against-a-judge-diverges}}. Why does the judge's agreement rate not
    bound the loop's divergence?
34. Why does increasing the number of variants per round not improve the ratio?

**RAG evaluation**

35. Write end-to-end RAG accuracy as a product. Which term has no standard metric?
36. Why is a point of recall@k worth 0.377 points of end-to-end accuracy rather than one?
37. Give an intervention that lowers recall and improves the product. Why would a
    recall-gated team reject it?
38. Faithfulness is 0.734 and usefulness is 0.627. Which quadrant is the gap, and what is the
    user experiencing?
39. State {{eq:faithfulness-and-usefulness-are-different-axes}}. Why can faithfulness not
    raise the ceiling?
40. Measured faithfulness rose after a chunking change. What happened, and what will the team
    conclude?
41. {{cite:barnett2024sevenfailures}} lists seven failure points. How many instruments
    localise them, and how many does an end-to-end score distinguish?

**Agent evaluation**

42. Reconcile a 58% single-run success rate with a 27% pass^8. What does the gap measure?
43. Two changes both move pass^1 by +0.035 and pass^8 by +0.005 and +0.055. What is the
    difference between them?
44. Why does an unconditional retry budget spend most of its attempts on tasks that will not
    respond?
45. Why is a *fast* failure a better retry signal than a slow one?
46. State {{eq:outcome-evaluation-credits-lucky-trajectories}}. In which two directions is
    outcome scoring wrong, and why do the errors not cancel?
47. Why does trajectory matching fail harder than single-answer reference matching?
48. What distinguishes a sound invariant from a disguised trajectory matcher?

**Building a framework, and online evaluation**

49. Nine instruments sum to 1.827 coverage and cover 0.956 together. Why, and what does that
    imply for planning?
50. Why is the optimal portfolio at a realistic budget 100% reference-free?
51. State {{eq:gate-placement-is-set-by-cost-times-escape}}. Why is "as early as possible"
    wrong for an expensive check?
52. Derive a gate's blocking threshold. Why does adding an early gate lower every later
    gate's threshold?
53. Why is the outcome metric always the slowest experiment you can run?
54. Why does running a low-correlation proxy experiment longer not reduce the decision error?
55. Twelve metrics across forty releases give 18 false alarms a quarter. What is the fix that
    is not a statistical correction?
56. State {{eq:a-gate-is-useless-if-mde-exceeds-tolerance}}. Why can this not be diagnosed
    from a gate's output history?

## Assignment: an instrument audit

Take an AI system whose evaluation you are responsible for or can inspect. Produce a written
audit with six sections.

**1. The instrument inventory.** List every number your team looks at to decide whether the
system is good. For each, write one sentence naming what it is *exactly* about, following
{{part:25}}'s organising table. Mark every row where that differs from what it is used to
decide.

**2. The four missing measurements.** Estimate, from your own system: the size of the
acceptable-answer space $|A|$ for your top task; your annotator agreement and the metric
ceiling it implies; your judge's position advantage in quality-equivalent units; and your
utilisation term if you run retrieval. Each is an afternoon. State which of the four you
already knew.

**3. The compression audit.** For each comparison your team makes — model versus model,
release versus release — identify which of {{part:25}}'s four compression mechanisms applies
and estimate its factor. Multiply them. That number is how much of a real difference your
measurements can see.

**4. The portfolio.** Draw your instrument-by-failure-class matrix, compute the union
coverage, and run the greedy build on your own costs. Compare the resulting order against the
order you actually built in, and name the instruments you own that the greedy build would not
have selected.

**5. The capability check.** For every gate that can block a release, compute its minimum
detectable effect and its stated tolerance. If the tolerance is not stated, say so — that is
the finding. Report how many of your gates are capable.

**6. What you could not measure.** Every quantity this audit needed that you could not
obtain. As in the audits for {{part:22}} through {{part:24}}, this is the most valuable
section, and here it will usually be dominated by $|A|$ and by the tolerance.

Length: six to ten pages and a spreadsheet.

## Challenge problems

**A. One compression, four mechanisms.** {{eq:contamination-inflates-and-flattens}},
{{eq:budget-splits-between-items-and-annotators}}'s $(1-2e)$,
{{eq:position-advantage-decides-close-pairs}} and
{{eq:a-fast-proxy-buys-speed-with-decision-error}} all multiply a true difference by a factor
below one. Write the general form, derive each as a special case, and determine whether the
factors compose multiplicatively when the mechanisms co-occur — as they do in every real
evaluation.

**B. Measuring $|A|$.** Design and cost a protocol that estimates the acceptable-answer space
size for a task from independent double-writing, including how to handle a non-uniform answer
distribution. Run it on one of your tasks and report the number, which as far as this book
found is unpublished for any production task.

**C. The correlated portfolio.** {{ch:ev-framework}} assumes instruments detect
independently. Model realistic correlation between text-inspecting instruments and recompute
the greedy build. Which additions survive, and how much flatter is the coverage curve?

**D. Judge correlation decay.** {{ch:ev-online}} argues a proxy's correlation with the
outcome decays once it becomes an optimisation target, at
{{eq:optimising-against-a-judge-diverges}}'s rate. Model the decay, find how long a proxy
remains the cost-minimising choice, and design a re-estimation schedule that does not require
the expensive experiment the proxy was adopted to avoid.

**E. The stated tolerance.** {{ch:ev-online}} finds the MDE check impossible without a stated
tolerance, and that the tolerance is usually unstated because stating it requires a decision
nobody wants to make. Design a process that elicits it — from incident history, from
willingness-to-pay, or from revealed preference in past rollbacks — and test it on three
metrics.

**F. The whole-part portfolio.** Combine {{ch:ev-framework}}'s coverage model with
{{ch:ev-online}}'s capability check and {{ch:ops-lifecycle}}'s period. Find the evaluation
programme that maximises detected defects subject to a budget *and* a loop-period ceiling.
Which instruments does the period constraint remove that the budget constraint does not?

## Interview preparation

Rehearse these until the answer is a structure rather than a recollection.

1. "This capability emerged at 20B parameters." — Ask for the answer length, then re-measure
   continuously.
2. "Our model scores 72% on the benchmark." — Expert baseline, non-expert baseline, then
   headroom closed.
3. "Our F1 improved by 3 points." — What cost ratio, and does it match yours.
4. "Two models have the same AUC." — Crossing curves, then precision at the committed recall.
5. "Our annotators only agree 76% of the time." — Guideline first, presentation second,
   people last, and report the ambiguity floor.
6. "Our judge agrees with humans 82% of the time." — Against what human–human rate.
7. "Our judge score improved 15% over six months." — Length, self-preference, and the
   spot-check rate.
8. "Recall@k went from 0.78 to 0.92 and nothing changed." — Utilisation, then the 0.377.
9. "Our faithfulness is 0.95." — Faithful to what, and was the context sufficient.
10. "The agent succeeds on 60% of tasks." — pass^k, then the task-class distribution.
11. "The A/B test on click-through says ship." — Correlation with the outcome, then the
    decision error.
12. "Our regression gate has passed every release for a year." — MDE against tolerance.

The pattern across all twelve: **name what the instrument is exactly about, name what the
decision needs, and measure the gap.** That is the part in one sentence and the question
worth having ready.

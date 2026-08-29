---
id: part-20-assessment
status: final
---

## How to use this

Four sections. The knowledge check takes about ninety minutes. The assignment is an
**audit of a real analysis**, because this part's findings are mostly things you can
measure on work you have already done. The challenge problems are open-ended. The
interview section is what to rehearse.

No answers are provided. Every question is answerable from the chapters.

## Knowledge check

**The stack**

1. Explain why benchmark attention in data science tracks gradeability rather than
   time, and say why that is structural rather than an oversight.
2. State {{eq:amdahl-bounds-the-stack}} and compute the speedup from perfectly
   automating an activity worth $11\%$ of the time.
3. Only about a third of a practitioner's day is gradeable. What follows for any
   claim of the form "agents can do $X\%$ of data science"?
4. State {{eq:pipeline-fails-at-the-weakest-verifier}} and explain why the
   check-placement rule here differs from {{ch:as-failures}}'.
5. You may either place a check or build a verifier. Why do those point at opposite
   ends of the pipeline?
6. Why does a held-out score not validate the analysis?

**Text-to-SQL**

7. Name the four sub-problems and say which carries the realistic-database gap.
8. Derive {{eq:grounding-not-syntax}} and explain why a sub-problem's *level*
   matters less than its *gap*.
9. A human analyst scores $92.96\%$. What would you have to supply a model to match
   that, and is any of it modelling work?
10. State {{eq:silent-failure-dominates}} and explain why it is entailed by the
    decomposition rather than an independent fact.
11. Which check on the ladder is free, and why is it free?
12. Automated reconciliation catches roughly twice what a human SQL review catches
    at a twenty-fifth the cost. Which chapter's ordering is that?
13. Why show the user the query if most users will not read it?

**Exploration and cleaning**

14. State {{eq:more-exploration-finds-only-noise}} and explain why the true-findings
    column is flat.
15. Why does Bonferroni correction suit confirmation and not exploration?
16. What input does every multiple-comparisons correction need, and why does an
    exploratory agent not supply it?
17. Why must a holdout be enforced by access control rather than instruction?
18. Explain why selecting for "interesting" makes precision worse.
19. State {{eq:cleaning-choices-move-the-answer}} and say why more data makes the
    multiverse problem relatively worse.
20. Why is the risk of automating cleaning not that the agent chooses badly?

**Feature engineering and model selection**

21. Explain why automated feature engineering is structurally a search for leakage.
22. Derive the sign of $\partial V/\partial g$ and $\partial D/\partial g$ for a
    leakage guard, and state the organisational consequence.
23. Your churn model validates at $99\%$. Give the detector that costs one sentence.
24. Why does the validation-deployment gap *narrow* as more features are selected?
25. Show that the real share of an AutoML search's apparent gain is independent of
    the search size, and say what it does depend on.
26. Explain why a noisier validation estimate selects a worse configuration and not
    merely a worse-measured one.
27. State {{eq:search-must-be-scored-off-search}} and name three instances of it in
    this part.
28. Why does a final holdout fix selection optimism completely and leakage not at
    all?

**Autonomous work and oversight**

29. State {{eq:self-judging-measures-correlation}} and explain what a self-judged
    acceptance rate is a measurement of.
30. A reviewer achieves near-human agreement on paper scores. Why does that not
    license the acceptance claim?
31. Why do neither a better same-family judge nor more of them help?
32. Generation costs \$15. Derive the cost per usable result and say which term
    dominates asymptotically.
33. State {{eq:volume-overflows-the-filter}} and give its limit.
34. Explain why reviewer attention behaves as a commons.
35. State {{eq:humans-go-where-nothing-else-is}} and explain why the highest-error
    stage can rank last.
36. Removing automated checks costs $28$ points at every human placement. What does
    that establish?
37. Which half of a task should be delegated, and on what criterion?
38. Why does "agent proposes, human judges" tie a solo human on judgement-heavy
    work?
39. Distinguish review from sampling formally.
40. State {{eq:the-middle-is-the-frontier}} and say what it implies about the
    literature's coverage.

## Assignment: audit a real analysis

Take an analysis you or your team completed — ideally one that informed a decision.

**Part 1: the time map.** Reconstruct where the hours actually went, by the seven
stages of {{ch:aids-stack}}. Compute the Amdahl bound for perfectly automating each.
Identify the tool you were about to buy and what its ceiling is.

**Part 2: the verifier map.** For each stage, state what checked it — automated,
human, or nothing. Seed a known error at three stages and measure what your existing
checks catch. Report detection rates rather than estimating them.

**Part 3: the multiverse.** Enumerate the defensible cleaning and analytic choices.
Run the grid. Report the spread against the confidence interval you originally
reported, and identify which two decisions carry most of it.

**Part 4: the denominators.** How many comparisons were run before the reported
findings? How many model configurations? Was any result reported from data used to
select it? If you cannot answer these from the record, that is the finding.

**Part 5: the conclusion check.** Ask {{ch:aids-oversight}}'s four questions of the
conclusion: does it follow, what would change it, is the effect worth acting on,
what was the denominator.

**Part 6: reallocate.** Given your measured detection rates and a fixed review
budget, compute where human attention should have gone and compare with where it
did.

**Part 7: the specification.** Write down one standard that was previously implicit
— what a cleaned table must satisfy, what would count as adequate exploration, what
threshold would change the decision. Make it executable if you can.

Deliverable: eight pages, with the multiverse spread as figure one.

## Challenge problems

**A. A verifier for exploration.** {{ch:aids-stack}} and {{ch:aids-oversight}} both
name this as the part's largest open problem. Propose a checkable definition of
exploration adequacy — coverage of the data, sensitivity to analytic choices,
whether the pre-stated questions were answered — implement it, and measure whether
it correlates with analyses later found sound.

**B. Automated multiverse reporting.** Enumerate analytic decisions from a notebook
automatically, run the grid, and summarise the spread and its attribution.
{{eq:speed-makes-the-multiverse-free}} says this is practical and almost nothing
implements it.

**C. Own-output reviewer agreement.** Run {{ch:aids-autonomous}}'s missing
experiment: measure an automated reviewer's agreement with independent human experts
*on its own generator's output* rather than on a general corpus. Report both numbers
and the gap.

**D. Point-in-time verification.** Build a checker that verifies a feature
pipeline's temporal validity mechanically, and run it against a production pipeline.
Report what it rejects and what that does to the validation score.

**E. Outcome-linked grading.** Connect a set of completed analyses to the decisions
they informed and those decisions' outcomes. Ask whether that supplies a usable
delayed verifier for the ungradeable stages, and at what latency and noise level.

**F. The redesign effect.** Design a study that measures the value of analyses
performed *because* they became cheap — questions that would not have been asked at
the old cost. This is the effect this part concedes it cannot see, and it may be
larger than everything it measured.

## Interview preparation

Rehearse until the mechanism comes out before the technique's name.

1. What does a $30\%$ score on a data science agent benchmark license you to say?
2. You perfectly automate model selection. How much faster is the project?
3. Text-to-SQL scores $40\%$ on real databases. Where is the gap, and what fixes it?
4. Your system reports that $95\%$ of generated queries execute. What have you
   learned?
5. What is the cheapest correctness check you are probably not doing?
6. Your agent explored for an hour and found forty patterns. What do you conclude?
7. Would you use Bonferroni correction on exploratory output?
8. Your analysis reports $0.33 \pm 0.05$. What is missing?
9. Why is automated feature engineering a search for leakage?
10. You install a leakage guard and validation drops four points. What do you say?
11. AutoML tried two thousand configurations and reports $0.90$. What is it worth?
12. Larger search or more folds, at a fixed budget?
13. A pipeline reports that $80\%$ of its output passes its own review. And?
14. It costs \$15 a paper. What does a usable paper cost?
15. You have one analyst-day a week for review. Where does it go?
16. Your cleaning stage has the highest error rate. Should you review it?
17. Is human review a substitute for automated checks?
18. Which half of an analysis would you give an agent, and why that half?

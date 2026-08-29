---
id: part-21-assessment
status: final
---

## How to use this

Four sections. The knowledge check takes about ninety minutes. The assignment is a
**pipeline readiness audit**, because this part's central finding is that the binding
constraint is the environment rather than the model. The challenge problems are
open-ended. The interview section is what to rehearse.

No answers are provided. Every question is answerable from the chapters.

## Knowledge check

**Generation**

1. Why is acceptance rate not a quality measure, and at what moment is it recorded?
2. State {{eq:ratio-decides-acceptance}} and give the rule it implies for which
   suggestions to accept.
3. Why is generated code reviewed less carefully than written code? Give the
   mechanism, not the motive.
4. Apparent saving $+66\%$, true saving $-171\%$. Explain how both are correct.
5. Writing is $15\%$ of a task. What is the speedup from making it free, and why is
   the share small?
6. State {{eq:visible-half-is-what-is-reported}} and say why the bias is signed
   rather than noisy.

**Repository understanding**

7. On an unseen repository, where does issue resolution fail? Give the shares.
8. Why is localisation a ceiling rather than a term?
9. A model names the buggy file from the issue text alone $76\%$ of the time on
   benchmark repositories and $53\%$ elsewhere. What is the $23$-point gap measuring,
   and what does it become end-to-end?
10. State {{eq:structure-finds-what-text-cannot}} and explain why text-only retrieval
    collapses with change span.
11. Why does improving text search improve structural retrieval but not the reverse?
12. Name the four things a failing test supplies.
13. A multi-file task fails. Should you retrieve more files? Justify it.

**SWE agents**

14. State {{eq:tests-are-a-partial-specification}} and give the inflation expression.
15. What did differential testing find about patches that pass the suite?
16. Why does adding differential testing look like a regression?
17. Show that precision among passing patches is independent of the plausible-patch
    rate, and say what follows as agents improve.
18. Two scaffold components measure approximately zero added alone and eighteen
    points removed from the full system. Which, and why?
19. Why does iteration work for coding agents and not for exploratory analysis?
20. A model $20\%$ better with no scaffold, or the baseline model with a full
    scaffold?

**Testing and refactoring**

21. Why does a suite generated from code achieve the highest coverage and the lowest
    detection?
22. Decompose detection into two factors and say which one coverage reports.
23. What single change to the prompt recovers most of the independence, and what does
    it cost?
24. Why must independence be arranged structurally rather than requested?
25. State {{eq:refactoring-safety-is-coverage}} and explain why debugging's success
    barely moves with coverage.
26. State {{eq:worst-code-is-least-covered}} and say why it is not a coincidence.
27. When is it correct to generate tests from the implementation? Give the
    distinguishing question.
28. A characterisation test fails during a refactor. What should happen?

**Pipelines and autonomy**

29. Derive {{eq:gate-by-blast-radius-not-author}} and show that volume cancels.
30. Why does gating everything perform worse in a running pipeline than on paper?
31. Documentation is your largest change category and ranks last for review. Explain.
32. State {{eq:automatability-is-verify-times-reverse}} and say why the activity
    ranking has a cliff rather than a slope.
33. Why is architecture the least automatable activity, and why is that not a claim
    about difficulty?
34. Where is a verifier worth most, and why is that the opposite of where they exist?
35. State {{eq:no-mode-dominates}} and give the task property that selects the mode.
36. Why can full autonomy applied uniformly be worse than manual work?
37. Show that the human-hours objective has no task dependence, and say what an
    organisation optimising it will conclude.
38. Reconcile the measured $19\%$ slowdown with the break-even analysis.
39. List the seven environment prerequisites and say which depends on which.
40. Why can model skill not substitute for containment?

## Assignment: a pipeline readiness audit

**Part 1: the corrections.** Take a currently reported coding-agent resolution rate.
Apply the contamination correction, the verification correction, and your own
coverage adjustment. State what you would plan against and show the arithmetic.

**Part 2: measure your stages.** For a sample of recent tasks, record time by stage:
understand, locate, write, review, get it working, integrate. Compute your Amdahl
bound for automating the writing stage.

**Part 3: measure localisation.** Take twenty resolved issues. For each, check
whether a retrieval system finds the full change set — not just the file the issue
names. Report recall against change span.

**Part 4: measure your suites.** Run mutation testing on a module with hand-written
tests and one with generated tests. Report coverage and mutation score for both, and
compute the implied independence.

**Part 5: build the gating table.** For each change type: defect rate, CI catch rate,
and escape cost from incident history. Rank by cost avoided per review-minute and
compare with your current policy.

**Part 6: locate yourself.** Score your environment on the seven prerequisites —
coverage, test independence, reproduction, CI latency, gating, rollback time,
executable architectural constraints. Identify the next one to build.

**Part 7: the recommendation.** A routing rule by task type, a revised gating policy,
and one prerequisite to build next, each justified with a number from parts 1–6.

Deliverable: eight pages, with the gating table as figure one.

## Challenge problems

**A. Reference-free differential testing.** Implement the check that a patch changes
behaviour on the reported case and nowhere else, using generated inputs and the
pre-patch version. Measure what it catches on real merged patches, and how much of
{{cite:wang2025solvedcorrectly}}'s divergence rate it recovers without a reference.

**B. Contamination-free localisation measurement.** Measure a model's buggy-file
identification accuracy on repositories created after its training cutoff, and
compare with {{cite:liang2025swebenchillusion}}'s $76\%/53\%$ figures.

**C. Acceptance-time verification.** Build test selection fast enough to run the
relevant tests at suggestion latency, and measure the change in escaped defects
against the latency cost.

**D. Co-change retrieval.** Mine version control for files that historically change
together, use it as a retrieval edge, and measure the marginal recall over
call-graph expansion alone.

**E. Executable architecture decision records.** Design a format in which an ADR
carries machine-checkable constraints, implement checkers for three real constraints
in a codebase you have, and measure what they reject.

**F. Your own randomised trial.** Replicate {{cite:becker2025devproductivity}}'s
design at small scale on your own team: randomise tasks to allow or disallow agent
use, measure completion time, and collect self-estimates. Report the gap.

## Interview preparation

Rehearse until the mechanism comes out before the technique's name.

1. Your completion tool reports a $35\%$ acceptance rate. What have you learned?
2. Which suggestions should you accept without close review?
3. You make writing code free. How much faster is the project?
4. Developers report $20\%$ faster and measurement shows $19\%$ slower. Are they
   lying?
5. Where does issue resolution actually fail?
6. What does a $76\%$ file-identification rate from issue text alone tell you?
7. Your retrieval found the file the issue mentions and the patch is still
   incomplete. Why?
8. What is the best localiser available, and why is it usually used as something
   else?
9. An agent resolves $65\%$ of a benchmark. What has been established?
10. Why does adding differential testing look like a regression?
11. Your agent improved and your defect rate did not fall. Why?
12. You add a test runner and resolution does not improve. Do you keep it?
13. Your generated tests reach $92\%$ coverage. What have you learned?
14. When is it correct to generate tests from the implementation?
15. Which pull requests should require human review?
16. Documentation is your largest category. Should you review it?
17. Why is architecture hard to automate?
18. You can invest in a better model or in your pipeline. Which, and by how much?

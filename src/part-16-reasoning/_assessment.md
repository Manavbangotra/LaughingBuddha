---
id: part-16-assessment
status: final
---

## How to use this

Four sections. The knowledge check takes about two hours. The assignment is a
**verification plan with arithmetic**, because this part's finding is that
verification binds everything and it is the component nobody budgets for. The
challenge problems are open-ended. The interview section is what to rehearse.

No answers are provided. Every question is answerable from the chapters.

## Knowledge check

**Reasoning versus generation**

1. State why held-out accuracy cannot separate computing an answer from fitting
   one, and name the three perturbations {{ch:rsn-vs-generation}} offers instead.
2. Two of those three need no ground truth. Which, and why does that make them the
   ones you can run on production traffic?
3. Ensemble spread gave an AUROC of 1.00 for out-of-distribution detection.
   Explain why that result should be read with suspicion, using the phrase
   "diversity of extrapolation".

**Chain-of-thought**

4. Derive {{eq:tokens-buy-steps}} from {{eq:depth-bounds-serial-steps}} and state
   {{cite:merrill2024cotexpressive}}'s stratification result.
5. Direct models were 100% inside the trained range of $k$ and scattered
   10.7–56.3% outside it. Explain why the listing says this is *not* "the network
   ran out of layers".
6. A step model at 72.6% per step gave 76.2/55.8/32.1/19.6% over 1/2/4/8 steps.
   State the equation and compute the 20-step figure.
7. Derive {{eq:length-tradeoff}} and explain why improving per-step reliability
   moves the optimal chain length *out*.
8. The shortcut carried 53.3% of the answer head's weight and zeroing it left
   accuracy at 100%. Explain why that is worse than the model never having learned
   the computation.
9. Stated-reason quality was flat at ~28% while accuracy went 100% → 6.3%. State
   what that implies about reading a chain of thought, and give the operational
   test that replaces it.

**Test-time compute**

10. State {{eq:coverage-selection-gap}} and explain why coverage is a hard ceiling
    on any selector.
11. Two generators with identical 9.4% single-sample accuracy voted to 99.6% and
    8.8% at $n=256$. State {{eq:vote-condition}} and explain the difference.
12. Systematic error capped coverage at 89.2% against 100%. Why does that defeat
    both halves of the decomposition at once?
13. Moving a verifier from $q=0.5$ to $q=1.0$ was worth +11.7 points at $n=8$ and
    +39.7 at $n=256$. Explain the direction, and say what it implies about
    sequencing investment.
14. Derive {{eq:marginal-value-of-a-sample}} and show it is maximised near
    $p = 1/(n+1)$. Why does "spend more on the hard problems" lose to uniform?
15. Early stopping beat an oracle allocation that knew every $p_i$ by 2.4 points.
    Explain why that is not a contradiction, and name its one requirement.

**Self-consistency and reflection**

16. State {{eq:marginalise-over-paths}} and explain self-consistency without using
    the words "checks" or "errors cancel".
17. Single-sample accuracy fell monotonically with temperature while the vote
    rose. What does that imply about a system tuned for single-answer quality?
18. Verifier-argmax accuracy *fell* from 52.6% to 38.1% as diversity rose while
    coverage stayed flat. Derive {{eq:argmax-extreme-value}} and explain.
19. Weighted voting lost to plain voting at a chance-level verifier. Explain why,
    and state the rule for choosing between the three selection rules.
20. Two critics with identical confusion matrices gave different outcomes. State
    {{eq:recoverable-mass}} and identify the term responsible.
21. The self-critic accepted a correct proposal 10.6% of the time on problems whose
    mode was wrong. Explain the mechanism in one sentence.
22. Revising toward one's own critique reached 36.5% against a vote of 36.8%.
    State {{eq:reflection-is-voting}} and give the condition under which it does
    *not* bind.
23. Reconcile {{cite:madaan2023selfrefine}} with {{cite:huang2024selfcorrect}}
    using that condition.

**Supervision**

24. Define the lucky-chain rate {{eq:lucky-chain}} and say how it varies with chain
    length and with the size of the answer space.
25. State {{eq:outcome-signal-bias}} and explain why it makes outcome supervision
    biased rather than noisy.
26. A fourfold increase in outcome labels bought +0.6 points and the same increase
    in process labels bought +3.1. What is the diagnosis, and what data do you
    already have to make it?
27. The process model's advantage reversed from +7.9 to −2.1 points as the
    lucky-chain rate fell from 8.6% to 2.0%. Use this to reconcile
    {{cite:uesato2022process}} with {{cite:lightman2023verify}}.
28. Imputed step labels beat both alternatives at low budget and lost 6.3 points to
    plain outcome supervision when lucky chains were rare. Identify both terms of
    {{eq:imputation-error}} and say which dominates in each regime.
29. Why must chain scores aggregate step scores with a minimum or product rather
    than a mean?

**Tools**

30. State {{eq:check-turns-coverage-into-accuracy}} and explain why the curve's
    *shape* changes rather than its level.
31. An incomplete specification capped accuracy at 66%, 37% and 22%. Derive
    {{eq:incomplete-check-limit}} and note that $n$ does not appear.
32. The pass rate went to 100% while shipped correctness stayed at 22%. Explain why
    this failure is quieter than the one tools were introduced to fix.
33. State the three-term model {{eq:tool-error-reallocation}} and derive the
    crossover $k^{*}$.
34. A per-step tool lost at every chain length and a single call won from $k=5$,
    with the same tool. Explain in terms of where the parse cost is paid.
35. Improving translation accuracy has an exponential return and improving parse
    accuracy a constant one. Show this from the same equation.

**Benchmarks**

36. Derive {{eq:variance-decomposition}} and explain why more trials per item does
    not reduce the rendering term.
37. The worse model won on 21.5% of renderings at a 1.2-point true gap. State
    {{eq:ranking-stability}} and compute how many renderings a 0.4-point gap needs.
38. Form share was 38.8% and 96.7% for two models of nearly equal ability. Say what
    that number measures and why it is free.
39. A familiarity detector's precision fell from 100% to 38%. Explain why the
    failure regime is the realistic one.
40. A clean form-fitted model produced a +9.6 point gap and a 25%-contaminated
    model +9.5. State {{eq:gap-is-not-identifiable}} and explain why this is
    acceptable in practice.

## Assignment: a verification plan, with arithmetic

Pick a task your team actually runs through a model. **The deliverable is a
two-page verification plan and the calculations behind it.** The point is not to
be right about the model — it is to produce a plan where every number came from a
measurement rather than from a technique's reputation.

**Establish what you are measuring**

1. Run {{ch:rsn-vs-generation}}'s paraphrase-consistency test on 200 production
   inputs. Report agreement and mean disagreement. This costs one extra call per
   input and no labels.
2. Generate at least eight renderings of your evaluation set. Report the mean, the
   standard deviation, and the form share {{eq:form-variance-share}}. State the
   smallest true gap your $k$ can resolve, via {{eq:ranking-stability}}.
3. Score a reworded copy of any public benchmark you quote. Report the gap and
   treat it as the discount on the claim.

**Establish the shape of your errors**

4. Sample $n=32$ per input on 200 problems. Report coverage, majority-vote
   accuracy, and $q_{\max}$ — the largest share held by any single wrong answer.
   State whether {{eq:vote-condition}} holds.
5. From that same pool, say whether your errors are systematic or unsystematic,
   and therefore whether voting is worth anything on your task.
6. Fit {{eq:coverage-grows}} to your coverage curve and solve for the effective
   independent sample count. Compare it with $n$.

**Establish the verifier**

7. Say what your verifier is: executable, learned, or the model itself. If
   executable, estimate the undetected-defect mass by introducing three defect
   classes your tests were not written against.
8. Measure its pairwise ranking accuracy on a pool containing both correct and
   incorrect samples. That number picks between voting, weighted voting and
   argmax.
9. If you have a reflection loop, measure the critic's acceptance and rejection
   rates **conditioned on whether the system was right**. Report the two rows
   separately.
10. Estimate your lucky-chain rate {{eq:lucky-chain}} by grading 200 solutions
    both ways. That number is the expected value of process supervision.

**Establish the budget**

11. Compute the marginal value of one more sample {{eq:marginal-value-of-a-sample}}
    at your current $n$ for three difficulty bands. Say where the money is.
12. If your task has a checkable answer, price early stopping against your current
    uniform allocation.
13. If you use a tool, measure $p_e$, $p_t$ and $p_r$ and compute $k^{*}$. Count
    your boundary crossings per request.
14. Report reasoning tokens as a share of output tokens and the resulting cost,
    using {{part:15}}'s decode economics.

**State the plan**

15. One page: what you will change, what you expect it to buy in points, and which
    of the measurements above justifies each claim. **Every number traceable to an
    equation or a measurement you took.**

## Challenge problems

**A. The identifiability problem.** {{eq:gap-is-not-identifiable}} shows the
aggregate original-versus-reworded gap cannot separate contamination from
form-fitting. Design a statistic that can, using item-level structure and no
training-data access, and test it against
{{ch:rsn-benchmarks}}'s second listing.

**B. A selector that does not degrade.** Both majority voting and verifier-argmax
converge to a proxy's maximiser as $n$ grows
({{eq:vote-converges-to-mode}}, {{eq:argmax-extreme-value}}). Construct a
selection rule with voting's floor and argmax's ceiling, or argue that the
trade-off is fundamental.

**C. Decorrelating a critic.** {{eq:recoverable-mass}} says a loop's value is
driven by error covariance rather than by critic accuracy. Build two critics from
one base model whose errors are measurably less correlated with the solver's than
the self-check critic's, and quantify the gain.

**D. Specification completeness as a training signal.** {{ch:rsn-tool-assisted}}
shows the specification is the ceiling. Design a training signal for
*completeness* rather than for passing a given specification, and say how you
would evaluate it without a ground-truth defect list.

**E. Predicting where reflection helps.** {{eq:reflection-is-voting}} binds when
recognition and generation are the same computation. Find a measurable property of
a task that predicts which side of that boundary it falls on, and validate it on
three tasks.

**F. The reliability gap, measured.** Take one task family, measure benchmark
score and production performance under matched conditions, and decompose the
difference into the four contributions {{sec:7-internal-mechanics}} of
{{ch:rsn-benchmarks}} names. This is the most useful thing in this part that
nobody has published.

## Interview preparation

Rehearse these until the mechanism comes out before the name of the technique.

1. What do intermediate tokens give a transformer that a bigger transformer
   cannot?
2. A model has 95% per-step accuracy. What is its 30-step accuracy, and what does
   that imply about system design?
3. A paper reports pass@100 of 80%. What have you learned about the deployed
   system?
4. Two models have identical benchmark accuracy; one benefits from
   self-consistency and one does not. What differs?
5. Why does a better verifier become *more* valuable as the sample budget grows?
6. Why can an allocation policy that knows nothing about difficulty beat one that
   knows every problem's difficulty exactly?
7. Why does picking the highest-scoring sample get *worse* as you sample more?
8. Two critics have the same accuracy; one improves your loop and one does not.
   What differs, and how do you measure it?
9. Reconcile a paper reporting large gains from self-refinement with one reporting
   none.
10. When is outcome supervision better than process supervision — not just cheaper?
11. You double your outcome-label budget and accuracy does not move. Diagnose it.
12. When does adding a calculator make a model worse at arithmetic?
13. Your agent passes its tests on every task and ships bugs. What do you measure
    first?
14. A model scores 72 and another 70 on the same benchmark. What can you conclude?
15. Why does running more trials per item not reduce the uncertainty in a benchmark
    comparison?
16. How would you measure contamination without access to anyone's training data —
    and what would that measurement fail to tell you?

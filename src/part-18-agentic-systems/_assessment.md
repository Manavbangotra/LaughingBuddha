---
id: part-18-assessment
status: final
---

## How to use this

Four sections. The knowledge check takes about two hours. The assignment is an
**architecture audit at equal cost**, because this part's finding is that almost
every multi-agent claim in circulation was measured against the wrong baseline. The
challenge problems are open-ended. The interview section is what to rehearse.

No answers are provided. Every question is answerable from the chapters.

## Knowledge check

**The single-agent baseline**

1. A bare loop completes $6.8\%$ and the same model with {{part:17}}'s components
   completes $89.6\%$ at identical per-action accuracy. Explain where the difference
   comes from, given that no step got more reliable.
2. State {{eq:components-interact-superadditively}} and explain why it makes
   one-at-a-time addition the wrong evaluation methodology.
3. Informative errors were worth $+23.4$ added to nothing and $+43.6$ removed from
   everything. Which number would a standard ablation report, and what would a team
   conclude?
4. Decompose a single agent's residual failure into its three parts and say which
   one no architecture in this part can touch.
5. Sweeping per-step accuracy $85\% \to 99.5\%$ moved completion $0.6$ points.
   State what that implies about waiting for a better model.
6. A fourfold budget increase moved the bare loop $1.3$ points. Why did it not
   spend what it had?

**Multi-agent and roles**

7. State {{eq:handoff-is-a-bottleneck}} and explain why an equal-cost comparison is
   the only meaningful one.
8. Three role-structured agents scored $18.2\%$ against one agent's $35.1\%$. Name
   the two distinct costs roles impose.
9. A same-model different-prompt critic was *worse than no critic*. State the
   correlation crossover and explain what puts a prompt-role critic above it.
10. State {{eq:critic-must-beat-more-attempts}} and say why "more attempts" is the
    correct comparison rather than "no critic".
11. Explain the difference between a label role and a capability role, and why only
    one earns its handoffs.
12. Why does a critic configured to advise have a floor that a gating critic does
    not?

**Graphs and state machines**

13. A graph lost to a free loop at *zero* tail mass. Give the mechanism and compute
    the cost of three branches at $96\%$.
14. State {{eq:graph-surrenders-the-tail}} and explain why a hybrid's escape hatch
    does not recover the testability the graph was bought for.
15. What is the honest argument for a graph, and why is it not a reliability
    argument?
16. Explain why a state machine has $p_e = 1$ and what it surrenders in exchange.
17. State {{eq:replay-needs-idempotence}}. A workflow engine guarantees at-least-once
    execution. What has it actually guaranteed?
18. A deduplication key took every idempotence level to $100\%$. Compare its cost
    with per-step checkpointing and say which removes the term rather than shrinking
    it.
19. Where must the key be written relative to the effect, and what window does the
    wrong order leave?
20. Omitting the tried set cost $38.6$ points and omitting the position cost $3.3$.
    Explain the ranking, and say why no workflow engine stores the expensive one.
21. Persisting nothing used *fewer* steps than persisting everything. Why is that not
    an efficiency result?

**Long-running work**

22. With recovery in place, exhaustion was $0.0\%$ at every horizon. State
    {{eq:recovery-converts-failure-to-cost}} and say what recovery turns step failure
    into.
23. State {{eq:horizon-changes-the-failure}} and explain why drift enters the
    exponent regardless while step failure does not.
24. Re-validation took horizon 300 from $0.7\%$ to $76.8\%$ for $50\%$ more steps.
    Derive why the optimum is a corner, and state the condition under which it moves
    inward.
25. Re-validation's value peaked at an *intermediate* staleness rate. Explain both
    ends.
26. Twelve targeted pauses matched a hundred uniform ones. State
    {{eq:placement-beats-frequency}} and give the efficiency ratio in terms of
    consequence density.
27. An idealised reviewer reaches $97.9\%$ where the real one reaches $36.6\%$. What
    does that say about buying oversight with volume?
28. State {{eq:oversight-has-a-horizon-limit}} and give the three responses in order
    of preference.

**Specialization**

29. Detection correlates $0.96$ with task success and difficulty $0.71$. Explain how
    research can have the highest per-step success and finish third.
30. State {{eq:retry-needs-a-verifier}} and say which of {{part:17}}'s mechanisms
    inherit the same factor.
31. Reversibility explained $83.3\%$ of the spread and fidelity $23.4\%$. Which was
    the listing written to promote, and why does that matter for how you read it?
32. Interventions worth $+1.0$ and $+0.3$ alone were worth $+54.5$ together. State
    {{eq:domain-properties-are-complementary}} and give its research-management
    consequence.
33. Explain why the action space enters logarithmically under a faithful observation
    and linearly under an unfaithful one.
34. Give four ways to manufacture an undo, and say which one concentrates
    irreversibility into a gateable step.

**Failure modes**

35. State {{eq:correlation-cuts-both-ways}} and explain why $r^k$ is pessimistic for
    a chain and optimistic for a vote.
36. A vote of nine was worth $+14.4$ points independent and $+1.2$ correlated.
    Derive $k_{\text{eff}}$ and evaluate it at $\rho = 0.9$, $k = 9$.
37. All-agents-fail rose from $0.01\%$ to $9.53\%$. Name the four sources of shared
    cause in decreasing order of contribution.
38. State {{eq:detection-decays-with-lag}} and explain why downstream agreement is
    not corroboration.
39. Four early critics lost to four spread ones despite the lowest lag. Explain via
    {{eq:coverage-before-freshness}}, and say why a coverage gap is structurally
    different from a freshness deficit.
40. The default architecture is a long chain with a reviewer at the end. Why does
    that arise, and how bad is it at twenty-four agents?

## Assignment: an architecture audit at equal cost

Take a multi-agent system — yours, an open-source one, or one from a paper — and
produce a report a skeptical reviewer would accept.

**Part 1: the baseline.** Build {{ch:as-single-agent}}'s properly-equipped single
agent for the same task. Not a bare loop. Report both at *equal token cost*, not
equal agent count. Most of the audit's value is in this step.

**Part 2: correlation.** Estimate $\rho$ for any panel, vote or critic in the system,
from joint failures on cases with known ground truth. Report $k_{\text{eff}}$
alongside every panel size. If the system has no panel, estimate the correlation
between its critic and its generator instead.

**Part 3: the affordance audit.** For every tool: is it idempotent, is it
reversible, and what is the detection rate of whatever checks its output — measured
on your error distribution, not claimed. This is one table and it drives durability,
oversight placement and the verdict on whether the domain is ready.

**Part 4: durability.** List what the system persists. Check it against
{{eq:tried-set-is-the-missing-field}}'s five fields and say which of the three
agent-state fields are missing.

**Part 5: placement.** Locate every critic and every human gate. Report coverage
gaps first, then mean lag. Recommend a placement at the same budget.

**Part 6: the recommendation.** State whether the multi-agent structure is earning
its cost, and if it is, which mechanism it is earning it through. "Decorrelation" and
"capability partitioning" are the two acceptable answers; if neither applies, say so.

Deliverable: eight pages, with the equal-cost comparison as figure one.

## Challenge problems

**A. The complementarity trap.** Design an evaluation protocol that correctly values
an affordance improvement while other constraints still bind, and validate it against
{{ch:as-specialized}}'s counterfactual table. This is the methodological problem
behind a great deal of apparently-stalled agent research.

**B. Optimal critic placement.** Given an error-position distribution estimated from
traces and a critic budget $m$, compute the placement minimising expected loss under
{{eq:coverage-before-freshness}}. Compare against even spacing. When does even
spacing lose?

**C. Diversity as portfolio construction.** Formalise panel selection as minimising
measured error correlation subject to a mean-accuracy floor, and test whether the
resulting panel beats a same-size panel selected for individual accuracy.

**D. Drift-aware evaluation.** {{cite:zhou2024webarena}} and
{{cite:liu2024agentbench}} use static environments, so the dominant long-horizon
failure cannot occur. Design an environment with controlled staleness and report what
current systems score.

**E. Agent state as an engine primitive.** Specify what a workflow engine would need
to persist the three agent-state fields as a first-class concept, and implement it
over an existing engine.

**F. Estimating $H^\*$.** Given a system's consequence density, drift rate and
reviewer habituation curve, compute the horizon past which gate-based oversight
cannot hold harm below a target. Validate against a real workload.

## Interview preparation

Rehearse until the mechanism comes out before the technique's name.

1. What does a second agent actually buy?
2. Your multi-agent system beats a bare loop. What have you shown?
3. Why is one-at-a-time ablation the wrong methodology here?
4. When does a role separation earn its handoffs?
5. Why did a graph lose to a free loop at zero tail mass?
6. What is the honest argument for a control-flow graph?
7. Your engine guarantees at-least-once execution. What does that guarantee?
8. Why is a deduplication key better than checkpointing more often?
9. Your durable state has position and outputs. What is missing, and what does it
   cost?
10. Your long-running agent completed and returned a wrong answer with nothing in the
    logs. What happened?
11. You doubled the budget on a long-running workflow and nothing changed. Why?
12. Would you pause more often for approval on a week-long run?
13. You have budget for twelve approvals. Where do they go?
14. Why do coding agents work better than research agents when research tasks have
    higher per-step success?
15. What does retry require to be a correction rather than a resample?
16. Your fidelity improvement did not move the eval. What do you conclude?
17. Your eight-agent pipeline computes to $27\%$ and measures at $60\%$. Explain.
18. You added six voters and accuracy did not move. Why?
19. Where do four critics go in a twelve-agent chain, and why not at the end?
20. What is the probability every agent in your system is wrong at once?

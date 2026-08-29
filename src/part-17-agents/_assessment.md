---
id: part-17-assessment
status: final
---

## How to use this

Four sections. The knowledge check takes about two hours. The assignment is an
**agent design review with arithmetic**, because this part's finding is that nine
separate questions each have a numeric answer and almost nobody computes them. The
challenge problems are open-ended. The interview section is what to rehearse.

No answers are provided. Every question is answerable from the chapters.

## Knowledge check

**What an agent is**

1. State the one property that distinguishes an agent from a workflow, and explain
   why it is binary rather than a spectrum.
2. Derive {{eq:tail-mass-decides}} and compute the crossover tail mass for
   $p_r = 0.94$, $p = 0.93$, $k_{\text{head}} = 3$, $k_{\text{tail}} = 6$.
3. The agent won on success at $20\%$ tail mass and never on cost per call. State
   {{eq:failure-to-call-ratio}} and say what decides which crossover governs.
4. An agent's cost had mean $6.29$, p99 $17$, max $40$; a router's was $2$ at every
   percentile. What does that do to a capacity plan?
5. Explain why the expected cost of an unbudgeted loop is finite while its maximum
   is not, and what that implies about stating a p100.
6. $531{,}441$ paths at horizon 12 and $0.09\%$ covered by 500 tests. State the
   correctness argument that replaces coverage, and its explicit remainder.

**Tool calling**

7. Name {{cite:schick2023toolformer}}'s four decisions and say which one depends on
   the size of the inventory.
8. Selection was $100.0\%$ at 4 tools and at 128. Explain via
   {{eq:distinctness-not-count}}, and say what *does* destroy it.
9. Quadrupling a distinct inventory cost nothing; two families at 64 tools cost 23
   points. State the metric to track as an inventory grows.
10. Enumerating three arguments beat removing two. Derive
    {{eq:enumerate-before-remove}} and say why the ordering is convenient.
11. Six optional arguments cost $9.3$ points. Explain
    {{eq:optional-is-not-free}} in one sentence.
12. Three retries bought $0.9$ points against an opaque error and $16.1$ against an
    informative one. Explain via {{eq:error-message-as-selector}}, and connect it to
    {{ch:rsn-test-time-compute}}'s coverage/selection split.

**The loop**

13. With a perfect stopping judgement, $75\%$ and $99\%$ per-action agents both
    completed $100\%$. State {{eq:loop-is-not-a-chain}} and explain where
    {{eq:chain-accuracy-compounds}} stops applying.
14. False stops $5\% \to 1\%$ bought $+19.6$; recognition $85\% \to 95\%$ bought
    $+0.1$. Explain the asymmetry from {{eq:asymmetric-stopping-errors}}.
15. Why is "ran out of budget" a better failure than "stopped early"?
16. A naive agent spent $35\%$ of its budget on repeats. Explain
    {{eq:no-progress-signal}} without using the word "confused".
17. Rank deduplication, temperature-on-failure and a bigger horizon by measured
    effect, and say why the third is the one teams reach for.

**ReAct and planning**

18. Derive {{eq:crossover-independent-of-length}} and state what it rules out about
    "use ReAct for complex tasks".
19. Replanning on surprise beat both pure strategies at every level. Explain via the
    exponents in {{eq:replan-on-surprise}}, and name the component it requires that
    they do not.
20. A thought bought $+0.3$ points at composition depth 1 and $+42.3$ at depth 4.
    State {{eq:thought-buys-composition}} and give two reasons to emit a thought
    that have nothing to do with accuracy.
21. Plan quality and replan frequency fix disjoint losses. State
    {{eq:disjoint-losses}} and compute which dominates at $q = 0.9$, $k = 12$,
    $\delta = 0.08$.
22. Six segments took a 12-step task from $48.7\%$ to $97.5\%$. State
    {{eq:checkpoints-cap-the-exponent}} and explain why 12 segments did worse than 6.
23. Four segments at a budget of 14 completed **zero**. Explain
    {{eq:budget-split-coupling}} and say what that implies about changing one
    parameter at a time.

**Memory**

24. Name the three mechanisms in {{eq:three-memories}} and the dependency each one
    serves.
25. Extending the window from 6 to 14 steps took recall of *recent* facts from
    $21.0\%$ to $10.5\%$. Explain {{eq:longer-window-hurts-recent}}.
26. A scratchpad bought $+21.1$ points at one reuse. Derive the break-even from
    {{eq:scratchpad-removes-an-exponent}}.
27. Store accuracy rose monotonically with size while $34.1\%$ of answers came from
    stale entries. Explain why the harm is invisible and what metric would reveal it.
28. Curation's value went from $2.3$ points at 50 entries to $24.2$ at 3,000. Is the
    trigger size or staleness? Justify from the fixed-size sweep.

**Recovery, termination, security**

29. Blind retry beat self-assessment by $14.9$ points. State
    {{eq:gating-costs-a-retry}} and explain the mechanism.
30. Derive {{eq:feedback-quality-threshold}} and say what bar a gating signal must
    clear.
31. Localisation bought $+15.5$ points and diagnosis $+1.2$. Explain the ordering
    from {{eq:localise-before-diagnose}}.
32. Localisation at $20\%$ accuracy still beat restarting. Prove
    {{eq:localisation-is-a-free-option}} and say what objection it answers.
33. Three reviewers gating 3,000 actions reached a $2.2\%$ catch rate. Show from
    {{eq:habituation}} that total catches saturate, and give the capacity.
34. Harm avoided per hour ranged $0.07$ to $34.52$ at a fixed review budget. State
    {{eq:gate-on-consequence}} and say why confidence is the wrong key.
35. A per-task cap held at $41\%$ across a sixfold budget increase. Explain
    {{eq:per-task-cap-wastes-budget}}, and say why pooling still needs a cap.
36. Why does escalation sit on a different part of the habituation curve from
    confirmation? Use {{eq:escalate-not-confirm}}.
37. Explain {{eq:no-channel-separation}} and why a prompt-level defence cannot work.
38. At $99\%$ detector recall, $69$ tasks were broken per prevented incident, and the
    ratio worsens with tuning. Derive it from {{eq:detection-ratio}}.
39. As injection prevalence rose a hundredfold, detection degraded and containment
    did not. State the property of {{eq:contain-do-not-detect}} responsible.
40. A per-tool review missed $69\%$ of the risk surface. State
    {{eq:blast-radius-is-a-union}} and say why per-tool review scales badly.
41. Splitting sixteen tools between two agents bought **nothing**; partitioning
    capabilities cut composed risk $43\%$. Explain
    {{eq:partition-capabilities-not-tools}}.

## Assignment: an agent design review, with arithmetic

Take an agent you run or intend to run. **The deliverable is a four-page review and
the calculations behind it.** Every recommendation must trace to a number you
measured.

**Establish the architecture**

1. Sample 200 real requests and classify them against the shapes you have flows for.
   Report the **tail mass**, and evaluate {{eq:tail-mass-decides}}.
2. Estimate $c_f/c_m$ — the cost of a failed task over the cost of a model call —
   and say which crossover in {{eq:failure-to-call-ratio}} governs you.
3. Report your agent's cost distribution: p50, p90, p99, observed max, and the
   budget-termination rate.

**Establish per-step reliability**

4. Decompose tool-call failure into {{eq:four-decisions}}'s four stages from your
   traces. Which stage is your loss in?
5. Embed your tool descriptions and report the nearest-neighbour distance
   distribution. Name your three most confusable pairs.
6. Report your per-tool $\phi$ — the fraction of failed calls whose retry succeeds.
   That is your error-message quality, and it is in your logs already.

**Establish the loop**

7. Measure $\alpha$ and $\beta$ for your stopping decision by running with the stop
   suppressed. Report the three-way outcome split.
8. Report repeats per run. What share of your budget is
   {{eq:no-progress-signal}} consuming?
9. Measure observation informativeness: for each step, could the correct action have
   been determined before the previous result? Evaluate
   {{eq:crossover-independent-of-length}}.

**Establish structure**

10. Plot your failure-position distribution. How much prefix is a restart wasting?
11. Identify where verified checkpoints could go, expressed as *state* assertions.
    Compute the optimal segment length from {{eq:budget-split-coupling}}.
12. Classify your step dependencies into {{eq:three-memories}}'s four categories.
    What does the histogram say you should build?

**Establish the controls**

13. Measure your reviewers' catch rate under load, by seeding known-bad actions at
    varying volumes. Report harm avoided per human hour for your current gate.
14. Enumerate your **capability union** and the dangerous pairs it composes. Compare
    with what a per-tool review found.
15. For every irreversible capability, state whether it is required, and design the
    undo path that would move it down a tier.

**State the plan**

16. One page: the three changes you will make, what each is expected to buy in
    points, and which measurement justifies it. **No recommendation without a
    number.**

## Challenge problems

**A. The gate-versus-advise router.** {{eq:feedback-quality-threshold}} says a signal
should gate only above a threshold. Build a layer that measures each signal's
position online and routes it to gating or advising automatically. Evaluate against
static assignment.

**B. Learned failure localisation.** {{eq:localisation-is-a-free-option}} makes any
accuracy positive, so this can be deployed before it is good. Train it from free
labels — every successful resume confirms a localisation — and find where the
corrupted-resume mode starts to bind.

**C. Non-uniform checkpoint placement.** Steps are not equally reliable. Formulate
optimal placement as a dynamic program over measured per-step reliabilities and
quantify the gain over equal splits.

**D. Capability reachability beyond pairs.** {{eq:blast-radius-is-a-union}} counts
pairs; real graphs have chains. Build the reachability computation over a capability
graph and measure how much larger the true surface is on a real inventory.

**E. Progress without an answer key.** Several chapters needed a signal that the run
is going well and none exists. Propose one — state-diff size, distance-to-goal,
information gain — and evaluate it against the productive-looking non-progress case
that deduplication misses.

**F. The habituation curve, measured.** Every number in {{ch:ag-termination}} turns
on a curve nobody has published. Run the experiment: seed known-bad actions into a
real review queue at varying volumes and fit {{eq:habituation}}.

## Interview preparation

Rehearse until the mechanism comes out before the technique's name.

1. What distinguishes an agent from a workflow, in one sentence?
2. Your router handles 94% of requests. When is an agent better?
3. Why is adding the 50th tool cheaper than you would expect?
4. Should you remove an argument or constrain it?
5. An agent takes 12 steps at 90% each. What is its success rate?
6. Why is a missed completion cheap and a false stop expensive?
7. Why does an agent repeat a failing action, and what is the cheapest fix?
8. Does the plan-versus-interleave crossover move with task length?
9. A 12-step task at 90% per step succeeds 28% of the time. How do you reach 90%
   without changing the model?
10. Why can a longer context window make an agent worse?
11. When is it worth writing a derived value down?
12. Why can an agent that checks its own work do worse than one that does not?
13. Rank detection, localisation and diagnosis, and justify it.
14. Why is requiring approval for every action a bad policy?
15. You doubled the step budget and nothing changed. What do you check?
16. Why can prompt injection not be fixed in the prompt?
17. Two tools are individually safe. When are they not?
18. You split your agent's tools between two agents. What did that buy?

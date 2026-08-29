---
id: part-22-assessment
status: draft
---

## How to use this

Four sections. The knowledge check takes about ninety minutes. The assignment is an
**instrument audit**, because this part's central finding is that the metrics a
platform emits by default are silent about the thing that decides whether the system
works. The challenge problems are open-ended. The interview section is what to
rehearse.

No answers are provided. Every question is answerable from the chapters.

## Knowledge check

**Architecture**

1. Name the three properties, and say which classical technique each one destroys
   most completely.
2. State {{eq:three-properties-break-the-stack}}. Why is the product form the
   important part — what would an additive model get wrong?
3. Expense and being-wrong halve the same number of techniques. Why is being-wrong
   still the worse property?
4. A service reports $99.900\%$ availability with a $6\%$ semantic error rate. Give
   the overspend against a $99.9\%$ target and explain what the dashboard is measuring.
5. State {{eq:boundary-decides-testability}} and explain why coverage is a product.
6. `classify intent` ranks third by coverage-per-cost and sits second in the pipeline.
   Why does that one observation invalidate the boundary framing?
7. What does interleaving restore that model-everywhere does not, and where does it
   restore it?

**Routing and caching**

8. Why can a cascade with a perfect judge beat the expensive model on accuracy? State
   the mechanism, not the number.
9. State {{eq:cascade-is-a-verifier-bet}}. What does $\phi$ cost, and why twice?
10. Give the break-even judge recall for the worked cascade and say what design
    question it converts into.
11. Describe the random-split test and say what it catches.
12. State {{eq:cache-threshold-is-an-error-cost-decision}}. Why is the optimum not a
    property of the cache?
13. Three surfaces, error costs $2$, $20$, $200$. Give the optimal hit rates and
    explain the direction.
14. A tight cache raises accuracy above fresh generation. Explain the mechanism.

**Queues and streaming**

15. State {{eq:variance-not-mean-drives-wait}}. Why does waiting scale with the square
    of the coefficient of variation?
16. Six workloads, identical mean service time, sustainable utilisation from $74.9\%$
    to $45.1\%$. What is the machine-count consequence?
17. Why is fair load balancing the wrong policy for a heavy-tailed workload, and what
    replaces it?
18. Derive the streaming crossover concurrency from two numbers. Which two?
19. Below the crossover, does answer length affect capacity? Above it?
20. Streaming's saved-percentage rose to $97\%$ while the user's wait grew $11\times$.
    Explain how both are true and say which to report.

**Retrieval and agents at scale**

21. State {{eq:fanout-amplifies-the-tail}}. Why does parallelism help latency and not
    reliability?
22. Fan-out $20$ under a $95\%$ budget requires $99.74\%$ per-call reliability against
    $98.70\%$ achieved. Why is that gap not closeable by optimising the dependency?
23. Describe hedging, give its cost in the worked example, and state the condition
    under which it buys nothing.
24. Corpus growth drives coverage down while precision rises. Give the mechanism.
25. Why does increasing $k$ fail to recover coverage on a large corpus?
26. Diversity reranking gains $+8.6$ points at $1{,}000$ documents and $+25.3$ at
    $10$M. Why does the gain grow?

**Storage**

27. State {{eq:access-shape-decides-the-store}}. Which ratios govern the crossovers,
    and why does that make the rule scale-free?
28. "A cache is cheap access sold with expensive storage attached." Justify it from
    the price table and give the condition under which a cache wins.
29. Chat transcripts are $940$ GB and belong in the database. Why does a size
    threshold get this wrong?
30. Five derived copies contradict each other on $27.36\%$ of queries. Where does the
    window come from, and why does synchronisation barely help?
31. Removing one copy beat halving every lag. Explain why, from
    {{eq:depth-beats-speed-for-staleness}}.
32. Both {{ch:sd-async}} and {{ch:sd-storage}} say "not the mean". Do they mean the
    same thing? Justify.

**APIs, auth, rate limiting**

33. State {{eq:count-limits-cannot-bound-cost}}. Why are the two requirements
    satisfied by disjoint ranges?
34. A safe count limit throttles light users $39\%$ while they consume $1.5\%$ of the
    ceiling. Explain how the limit was derived.
35. Why can a cost limiter not simply charge after the request completes?
36. State {{eq:delegation-moves-the-check}}. Under a service account, what is standing
    in for the missing authorization check?
37. Row-level filtering moved over-permission from $50.8\%$ to $43.6\%$ only. Why is
    that the wrong shape for a mitigation?
38. `update_record` is $51\%$ of exposure in $6\%$ of requests. Why does exposure
    concentrate, and what follows?

**Fault tolerance**

39. Give the expected value of a retry for transient infrastructure and for systematic
    semantic failure, and explain the sign.
40. State the condition under which retrying a confidently-wrong answer is negative at
    *any* verifier recall.
41. State {{eq:uniform-retry-inverts-its-budget}}. Why is the misallocation structural
    rather than a tuning error?
42. Detection scales with the square of the effect size. What does that imply about
    which regressions are expensive to catch?
43. Give the optimal sampling rate for the worked service and the damage ratio it
    closes.
44. Why is under-sampling a false economy "in the same shape as over-caching"?

**Latency**

45. State {{eq:sum-of-tails-overprovisions}}. Give the over-provisioning factor in
    terms of the stage standard deviations.
46. All six stages miss their allocation and the system passes with $373$ms to spare.
    Which number is wrong, and why?
47. Why is budget share not slack? Give the gateway and generation figures.
48. State {{eq:tail-attribution-differs-from-mean}}. Why are mean reductions size-blind
    and variance reductions not?
49. The obvious target returns $11.82$ms per week and the best returns $93.39$ms. Name
    both interventions and explain the gap.
50. On generation itself the mean lever wins. Reconcile that with the previous answer.

## Assignment: an instrument audit

Take a system you have access to — yours, your employer's, or an open-source
deployment you can instrument. Produce a written audit with five sections.

**1. The instrument inventory.** List every metric the system currently alerts on or
reviews weekly. For each, state the quantity it actually measures and the quantity a
reader would assume it measures. Mark every row where those differ. The table in
{{part:22}}'s introduction is the template; your version should be specific to your
system.

**2. The missing second instrument.** Determine whether anything measures whether
answers are *right*, as distinct from whether responses arrived. If nothing does,
price one: use {{eq:semantic-breaker-is-affordable}} with your own traffic rate,
review cost, and estimated error cost, and state the sampling rate and annual figure.
If something does, state its sampling rate and what it would detect within 24 hours.

**3. Two ratios.** Compute reads-per-gigabyte for your three largest pieces of state
({{eq:access-shape-decides-the-store}}), and the failure-kind breakdown of your retry
budget ({{eq:uniform-retry-inverts-its-budget}}). Both are derivable from data you
already retain. Report what you found, including if the answer is "correctly placed"
and "well allocated" — a negative result is a result.

**4. One counterfactual.** Pick the latency, cost, or reliability problem your team is
currently working on. Produce the shape-based alternative from the table in
{{part:22}}'s introduction and price both. State which you would now choose and what
would change your mind.

**5. What you could not measure.** List every quantity this audit needed that your
telemetry does not retain. This section is the most valuable one, and it is the reason
to do the audit before you need it.

Length: enough to be actionable, no more. A good audit is six pages and a spreadsheet.

## Challenge problems

**A. Correlated everything.** Every model in this part assumes independence somewhere.
Take three of them — {{eq:fanout-amplifies-the-tail}},
{{eq:sum-of-tails-overprovisions}}, and {{eq:derived-copies-multiply-contradiction}} —
and work out the direction each error runs under positive correlation. Which of the
part's recommendations survive, and which reverse?

**B. The composed exposure problem.** {{ch:sd-apis-auth}} computes over-permission per
tool. An agent that can both read and transmit can exfiltrate, which is a property of
the *pair*. Formulate exposure over reachable pairs, compute it for the listing's tool
set, and determine whether the two-tool removal is still the right intervention.

**C. Interleaving under correlation.** {{ch:sd-architecture}} argues that a
deterministic stage between two model stages breaks the correlation between their
failures by re-grounding the input. Design an experiment that would measure this, state
what result would falsify it, and estimate what it would cost to run.

**D. The unified budget.** {{ch:sd-apis-auth}}'s cost limiter, {{ch:sd-async}}'s
capacity model, and {{ch:sd-latency}}'s latency budget are three allocations of the
same underlying resource. Write down a single formulation that produces all three, and
say what it reveals that the separate treatments do not.

**E. When does the boundary move?** {{ch:sd-architecture}} ranks stages by coverage per
unit cost. Model the trajectory of that ranking as model cost falls by an order of
magnitude and deterministic tooling improves by half. Which stages cross over, in what
order, and what would you build differently today knowing that?

## Interview preparation

Rehearse these until the answer is a structure rather than a recollection.

1. "Our availability is $99.97\%$ and users say the answers got worse." — Name the
   missing instrument, price it, and say what it would have caught.
2. "We cut inference cost $70\%$ with a cascade." — Ask the accuracy question, then
   the random-split question, then the judge-recall question, in that order.
3. "Our cache has a $60\%$ hit rate." — Ask what a wrong answer costs on that surface
   before saying anything else.
4. "Every tool has a p95 under 200ms and the agent's p95 is 1.6 seconds." — Fan-out,
   with the arithmetic.
5. "Our rate limit is 100 requests/minute and the bill is $3\times$ forecast." — The
   two horns, and why no third number exists.
6. "Should we optimise the slowest stage?" — Mean versus variance attribution, and the
   per-stage test.
7. "Why is retrying a model call different from retrying a database call?" — Resample,
   verifier, harm rate.
8. "How many copies of a fact does your system store?" — Ask it of them; most teams
   have never counted, and the count predicts a failure they have seen.

The pattern across all eight: **name the quantity being measured, name the quantity
that matters, and say whether they are the same.** That question is the part in one
sentence, and it is the one worth having ready.

---
id: part-28-assessment
status: draft
---

## How to use this

Four sections. The knowledge check takes about ninety minutes. The assignment is a **currency
audit**, because this part's rule was to name the currency and find the exchange rate, and the
commonest finding is that a team is optimising a budget that stopped binding some time ago. The
challenge problems are open-ended and several are genuinely unanswered. The interview section is
what to rehearse.

No answers are provided. Every question is answerable from the chapters.

## Knowledge check

**Scaling laws revisited**

1. Derive the compute-optimal split condition and show it depends on the exponents alone.
2. Two exponent pairs give 325.0 and 11.9 tokens per parameter. What changed, and what did not?
3. State {{eq:scaling-exponents-set-allocation-not-the-ceiling}}. What does the floor do as the
   budget grows?
4. Why does each halving of the reducible loss cost 86×? Derive it from $a$ and $b$.
5. State {{eq:the-training-optimum-is-not-the-deployment-optimum}}. Why does the optimum shrink
   as serving volume rises?
6. A smooth run scores 4.2 on loss and 1152.9 on 20-token exact match. What produces the
   difference?
7. Why is a two-decade fit without a floor term 48.9% wrong six decades out, and in which
   direction?

**Mixture of experts and sparse models**

8. Why is the FLOP-matched comparison the wrong one, and what is the right one?
9. State {{eq:sparsity-moves-the-bottleneck-from-flops-to-memory}}. Why is 3.0× the FLOPs and
   85.7× the weights a bad trade for serving?
10. Compute the break-even serving volume for a sparse design. What decides it?
11. Why does a batch of 512 read all 128 experts?
12. State {{eq:capacity-factor-trades-dropped-tokens-for-wasted-compute}}. Why does doubling
    capacity remove less than two thirds of the drops?
13. Why do balance and specialisation trade against each other rather than being independent?
14. Which of the two balance-loss failure modes is dangerous, and why?

**Long context and memory**

15. State {{eq:effective-context-is-shorter-than-nominal}}. What is mean recall at 512,000
    tokens?
16. Why does five-fact success fall to 0.279 when one-fact success is 0.775?
17. Why does the largest window cost 69× the smallest per solved task?
18. State {{eq:memory-is-compression-times-retrieval}}. Contrast a sliding window with a vector
    index on both terms.
19. At which horizons does no single architecture reach 0.90 recall, and what fixes it?
20. Why does a window plus an index beat either alone? Which earlier result is that?

**Test-time training and test-time compute**

21. State {{eq:test-time-compute-has-a-ceiling-training-does-not}}. What sets the ceiling?
22. The first 16 samples deliver 77.9% of the available gain. What does a 4,096-sample budget
    buy?
23. Training to 0.90 costs $1,262M; sampling costs $0.000277 per request. Why does the crossover
    at $4.6\times10^{12}$ requests not matter?
24. State {{eq:adaptation-must-amortise-over-reuse}}. Why does per-request adaptation fail it?
25. Why does adaptation gain peak on the session rather than the request or the corpus?
26. Why does full-weight adaptation cost 591× throughput when the compute overhead is 6.9%?

**Continual, online and self-improving systems**

27. State {{eq:plasticity-and-retention-are-one-dial}}. Why can no mechanism be plastic about new
    information and rigid about old?
28. Which regulariser has the best absorption/forgetting pair, and why is it not used?
29. State {{eq:update-cadence-has-an-interior-optimum}}. What term is usually left out of cadence
    models?
30. At a 14-day drift half-life the best cadence is annual. Explain.
31. State {{eq:self-training-improves-only-the-verifiable-fraction}}. What is the ceiling at
    $v = 0.20$?
32. Why does mixing in real data set the *rate* of diversity contraction rather than a floor?

**World models and embodied AI**

33. Derive the usable horizon from per-step error and amplification. Why is it logarithmic in
    $1/\epsilon$?
34. State {{eq:model-based-gain-is-bounded-by-the-usable-horizon}}. Why do long-horizon tasks get
    decomposed rather than planned?
35. What do the cheap horizon-extending techniques have in common?
36. State {{eq:embodied-data-is-rate-limited-not-cost-limited}}. Why is 50 robots a 51-month
    schedule?
37. Why is human demonstration video the most expensive source per real-equivalent trajectory?
38. Why does system identification beat fine-tuning on real data per unit cost?

**AI for science, and reading the frontier**

39. State {{eq:autonomous-output-shifts-the-bottleneck-to-review}}. Why does 400,000 candidates a
    year produce the same output as 400?
40. Why is thorough verification correct here when cheap-wide gates win elsewhere in this book?
41. State {{eq:a-finding-is-worth-its-verification-probability}}. Why is triage worth more than
    faster review?
42. Why does a bold hypothesis have low information content? Derive it.
43. Compute the waste multiple of an unpublished negative result.
44. State {{eq:confidence-is-a-product-over-independent-evidence}}. Why is four of five factors
    not 80%?
45. Why do citations correlate 0.21 with confidence and a reproduction 0.81?
46. State {{eq:adoption-value-is-tier-times-lead-time}}. Why can two teams adopt differently on
    identical evidence and both be right?

## Assignment: a currency audit

Take a system, a roadmap, or a research programme you are responsible for. Produce a written
audit with six sections.

**1. The currency inventory.** For each major investment currently underway, write two lines: the
resource it improves, and the resource that binds. Mark every row where they differ. This is the
part's organising idea applied to your own plan and it is usually an afternoon.

**2. The serving forecast.** Write down your twelve-month and three-year token or request volume,
with the reasoning. Then compute, from your own numbers: the inference-aware optimal model size;
the break-even volume for any sparse design under consideration; and the training-versus-sampling
crossover for your quality target. Three chapters turn on this one number and it is usually
nobody's job.

**3. The utilisation measurements.** Run a context position sweep and compute effective context.
Measure multi-fact success at $k = 1, 3, 5$. Measure your verifier's selection quality and the
implied test-time ceiling. Each is an afternoon and each is a number almost no team has.

**4. The drift estimate.** Estimate your distribution's half-life from historical accuracy on
freshly labelled samples, compute the cadence optimum, and compare against your current schedule.
Then build one label-free drift signal and measure its correlation with the labelled one.

**5. The evidence audit.** Score every research claim your roadmap depends on against
{{ch:res-frontier}}'s five-factor rubric. Compute your lead-time premium and the break-even it
implies. Report your exposure by tier and the expected rework over your planning horizon.

**6. What you could not measure.** Every quantity this audit needed and could not obtain. As in
the audits for {{part:22}} through {{part:27}}, this is the most valuable section — and here it
will usually be dominated by the sparse-parameter credit, the transfer coefficient, the verifier
quality, and your own drift rate.

Length: ten to fourteen pages and a spreadsheet.

## Challenge problems

**A. The sparse-parameter credit.** {{ch:res-moe}}'s $\rho$ decides where the break-even sits and
is assumed throughout. Design a measurement — a ladder of matched-quality dense and sparse runs —
that fits it directly, and determine whether it varies with corpus diversity as
{{sec:15-advanced-concepts}} predicts.

**B. The two-dimensional memory frontier.** {{ch:res-memory}} models recall as a function of age
alone and notes it also depends on query type. Build the (age, query-type) recall surface for
three memory architectures and find the pair with the most orthogonal failures.

**C. Verifier degradation.** {{ch:res-test-time}} assumes verifier quality is independent of the
generator, and argues it is not: a stronger generator produces more plausible wrong answers.
Measure that degradation and determine the model quality at which test-time compute's value
peaks.

**D. The transfer coefficient's dependence on policy.** {{ch:res-world-models}} treats $\tau$ as a
property of a simulator and argues it is a property of a (simulator, task, policy) triple.
Measure $\tau$ at three points during a training run and determine whether it falls as the policy
improves.

**E. The true external-data fraction.** {{ch:res-continual}}'s $r$ decides whether a production
feedback loop degrades, and deployed outputs contaminate "external" data invisibly. Design a
method for estimating the true $r$ in a live system and run it.

**F. The novelty multiplier.** {{ch:res-ai-for-science}} shows the experiment ranking reverses
only if novelty is worth roughly 30× a replication, and the field behaves as though it is.
Measure the downstream impact of bold results against replications and ablations over a decade of
citation and adoption data, and report the multiplier.

**G. The whole-part residual.** Combine {{ch:res-scaling}}'s deployment optimum,
{{ch:res-moe}}'s break-even, {{ch:res-memory}}'s utilisation and {{ch:res-test-time}}'s ceiling
into a single system design for a specific product and serving forecast. Which term dominates,
and does the ranking change when they are optimised jointly rather than separately?

## Interview preparation

Rehearse these until the answer is a structure rather than a recollection.

1. "We're training compute-optimally." — For which phase, then the 13.7×.
2. "The model has 500 billion parameters and 20 billion active." — Bytes per request, then the
   5.10×.
3. "It supports 128k context." — Effective context, then five-fact success at 0.279.
4. "We sample 1,024 times for accuracy." — The ceiling, then 77.9% at sixteen.
5. "We adapt the model per user." — Adapter size, then the batch, then 591×.
6. "We retrain weekly to stay fresh." — Drift half-life, then the regression term.
7. "Our self-improvement loop is on round twelve." — Diversity, then the peak at round three.
8. "The world model plans twenty steps ahead." — Per-step error, then the tolerance.
9. "We need more robot data." — Robot-days, not dollars, then the transfer coefficient.
10. "The system generates 500 hypotheses a day." — Review capacity, then 17.0.
11. "This paper has four hundred citations." — Correlation 0.21, then ask who ran it.
12. "This result is established." — Five factors, then which one is missing.

The pattern across all twelve: **name the currency the claim is denominated in, ask which
currency actually binds, and price the exchange rate between them.** That is the part in one
sentence, and — since everything specific in it will have moved by the time you need it — it is
the only part worth memorising.

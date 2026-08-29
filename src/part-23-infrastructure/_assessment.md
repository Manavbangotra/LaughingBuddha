---
id: part-23-assessment
status: draft
---

## How to use this

Four sections. The knowledge check takes about ninety minutes. The assignment is a
**regime audit**, because this part's rule was that every performance claim is a claim
about a regime, and the commonest error is quoting one outside its own. The challenge
problems are open-ended. The interview section is what to rehearse.

No answers are provided. Every question is answerable from the chapters.

## Knowledge check

**CPU and GPU fundamentals**

1. Define arithmetic intensity and give its value for decode at batch $m$ with $b$ bytes
   per weight. Why is it independent of model size?
2. A device has 989 TFLOP/s and 3350 GB/s. What is its balance point, and what does that
   number mean?
3. Why does a GPU run decode at $0.3\%$ of peak at batch 1? Is that a defect?
4. State {{eq:batch-is-the-mechanism-not-an-optimisation}} and explain why "optimisation"
   understates what batching does.
5. Weights are read once per pass; KV cache is read per sequence. Derive the crossover
   context $c^\star(m)$ and explain why it *falls* with batch size.
6. Grouped-query attention with four KV heads moves the batch-32 crossover from 772 to
   6176 tokens. Why is that a training-time decision?
7. CPU decode at batch 1 costs $4.3\times$ a datacentre GPU, not the $309\times$ the FLOP
   figures imply. Explain.

**GPU memory and the roofline**

8. Why is one roofline insufficient? Give the balance points for HBM and shared memory.
9. FlashAttention cuts HBM traffic $65\times$ at 8192 tokens and gives a $4.7\times$
   speedup that does not grow. Explain both numbers.
10. State {{eq:tier-crossing-has-a-ceiling}}. What is the ceiling, and what lever remains
    once it is reached?
11. State {{eq:batch-times-context-is-the-budget}}. Why are maximum batch and maximum
    context one setting rather than two?
12. Quantising the cache buys $4.0\times$ the token-slots; quantising the weights buys
    $1.2\times$. Why the difference?
13. Paging's gain is $1/u$ for length utilisation $u$. What does that imply about a
    deployment whose requests use their full context window?

**Batching**

14. Derive static batching's utilisation and explain why it falls as batch size rises.
15. Continuous batching's gain is $\mathbb{E}[\max]/\bar{L}$. What does that predict about
    reproducing a vendor's reported number?
16. A 3200-token prefill run as its own step costs 347 tokens. Whose tokens, and why is
    the victim never the cause?
17. Why is a 263-token prefill chunk free and a 512-token chunk $1.84\times$ a decode
    step? State the mechanism, not the number.
18. Give the formula for the free chunk size and explain why it is computed rather than
    tuned.
19. Chunking wins at 3200-token prompts and disaggregation at 12,000. What changes
    between them?

**Parallelism**

20. Derive $C_{\text{tensor}}/C_{\text{pipe}} = 4L/n$ and evaluate it at $L=80$, $n=8$.
21. Tensor parallelism gives $7.76\times$ on a fast link and $0.79\times$ on 25G ethernet.
    What does $0.79\times$ mean, and what follows for topology?
22. Why does the required interconnect bandwidth rise steeply with tensor-parallel degree?
    Name both terms and their directions.
23. Which parallelism dimension makes a single request faster, and which does not? Give
    the full table.
24. State {{eq:sparsity-erodes-with-batch-size}} and give the half-coverage batch for
    64 experts at top-2.
25. An MoE model costs $0.99\times$ a dense model of its *total* size at batch 128.
    Reconcile that with {{cite:fedus2021switch}}'s $7\times$.

**Distributed inference**

26. Why are prefix affinity and load balancing in conflict? Name the quantity each
    optimises.
27. The optimal affinity moves from 1.0 at 12% utilisation to 0.0 at 44%. What property
    of the queueing term causes that?
28. Why does the affinity curve dip at $\alpha = 0.2$? What is happening there?
29. State {{eq:parallel-group-is-one-failure-domain}}. Why is a 16-way group $256\times$
    worse than one device at two replicas rather than $16\times$?
30. Sixteen devices as one tensor group or as four groups in a pipeline. Which is more
    reliable, by how much, and what does it cost?
31. A failure destroys in-flight KV cache. How many user-visible failures does that
    produce, and what prevents them?

**Serving stacks**

32. Why does the same feature get 19% of the credit in one ordering and 61% in another?
33. State {{eq:overlapping-techniques-are-substitutes}} and say what it implies for a
    roadmap that multiplies published speedups.
34. Launch overhead is $31.0\%$ of a decode step at batch 1 *and* at batch 256. Explain
    why it does not dilute.
35. By how much does the roofline underpredict, and how would you distinguish that from a
    memory-system problem?
36. Why is graph capture's benefit independent of batch size when nothing else in this
    part is?
37. A 70B bf16 model spends $4.3\%$ of its step on launches and a 3B int4 model $80.8\%$.
    What does that imply about quantising small models?

**Kubernetes and autoscaling**

38. Why does GPU utilisation have a dynamic range of $1.0\times$ here?
39. Why is queue depth a good signal in most systems and a bad one in this one?
40. State {{eq:trigger-is-the-reciprocal-of-growth}}. Compute the trigger for a 210-second
    cold start and load doubling every five minutes.
41. Above what ramp rate does reactive autoscaling have no solution? Derive the condition.
42. Where does a cold start's time actually go for a 140 GB model? Give the shares.
43. Why is baking weights into the container image the worst available placement?
44. Weight placement determines idle fleet — 67% against 19%. Trace the causal chain.

**Cloud, edge, and local**

45. State {{eq:self-hosting-is-a-utilisation-bet}}. At what utilisation does self-hosting's
    unit cost cross the API price?
46. Why is the true break-even 44,000 Mtok/month when the fixed-cost calculation gives
    15,124?
47. Under what condition does no break-even exist at any volume?
48. Operations is 62% of fixed cost at a modest estimate. What does that make the
    self-hosting decision?
49. An 8B int4 model fits on every device tested. Why is that not the question?
50. A mid-range phone bursts at 22.7 tokens/second and sustains 9.5. Which number does a
    benchmark report, and which does a user experience?

## Assignment: a regime audit

Take a serving deployment you have access to — yours, your employer's, or a
self-hosted open-source stack you can instrument. Produce a written audit with six
sections.

**1. The four numbers.** Compute, from your own system: arithmetic intensity at your
operating batch, your device's balance point, your token-slot budget, and your kernel
count per token. Each is one division or one profiler run. State which of the four you
already knew.

**2. Regime placement.** For each of the eight chapters, state which regime your
deployment is in — memory- or compute-bound, above or below the KV crossover, above or
below the free chunk size, and so on. Mark every place where a configuration value was
inherited from a benchmark or a default that was measured in a different regime.

**3. The roofline residual.** Measure achieved throughput. Compute the roofline
prediction, add the launch term, and report the remaining gap. If the residual is large,
say what you think it is; if it is small, say so — a confirmed model is a result.

**4. Two corrections.** Pick the two largest discrepancies from sections 2 and 3 and
compute what fixing each would buy, in the units your team cares about. Rank them by
gain per week of work, following {{ch:sd-latency}}'s method.

**5. The economics.** Compute your realised utilisation and your true cost per million
tokens including operations. Compare against a provider price. State whether your
current build-or-buy position is the one the arithmetic supports, and if not, what
non-economic reason justifies it.

**6. What you could not measure.** Every quantity this audit needed that your telemetry
does not retain. As in {{part:22}}'s audit, this is the most valuable section and the
reason to do the exercise before you need it.

Length: six to ten pages and a spreadsheet.

## Challenge problems

**A. The batch is over-determined.** Batch size is constrained by
{{eq:batch-times-context-is-the-budget}}, degraded by
{{eq:kv-traffic-overtakes-weights}}, required by
{{eq:batch-is-the-mechanism-not-an-optimisation}}, and eroded by
{{eq:sparsity-erodes-with-batch-size}}. Write down a single optimisation whose solution
is the batch size, with all four constraints, and solve it for a workload of your choice.
What does the solution depend on most?

**B. Correlation everywhere.** Take the four independence assumptions named in the part
introduction and work out the direction of each error. Which of this part's
recommendations survive under strong correlation, and which reverse?

**C. Dynamic chunk sizing.** {{ch:inf-batching}} notes that the free chunk size is
$I^\star - m_t$ for the *instantaneous* batch, which changes every step under continuous
batching. Design a scheduler that sizes chunks dynamically, estimate what it captures
over a fixed chunk size, and identify what could go wrong.

**D. Weight-residency-aware scheduling.** {{ch:inf-kubernetes}} observes that a node that
recently served a model has its weights in page cache, making restart far cheaper.
Specify the scheduling signal, the placement policy, and the failure mode when the
scheduler prefers warm nodes that are also the busiest.

**E. The two-tier fleet.** {{ch:inf-distributed}} identifies a tension between affinity
routing (wants stable specialised nodes) and autoscaling (wants interchangeable
disposable ones). Design a stable-core-plus-elastic-margin fleet, price it against both
extremes, and say what determines the core size.

**F. Where does the balance point go?** Model the balance point's trajectory over three
hardware generations at historical rates of compute and bandwidth growth. Which of this
part's conclusions change, and which are stable?

## Interview preparation

Rehearse these until the answer is a structure rather than a recollection.

1. "Our GPUs show 100% utilisation." — Occupancy versus percent-of-peak, with the batch-1
   number.
2. "We got a 5× speedup from the new serving stack." — Ask about the baseline, then about
   which inefficiency, then about the regime.
3. "The 8B model fits on the phone." — Bandwidth per gigabyte, then tokens per second,
   then sustained versus burst.
4. "We should self-host, our GPU is cheaper per token." — Utilisation, then operations,
   then the crossover.
5. "Adding GPUs did not reduce latency." — The four dimensions, and which one they added.
6. "Our throughput is 30% below the bandwidth prediction." — Launch overhead, with the
   kernel count.
7. "Scaling is too slow." — Cold start decomposition, then weight placement, then the
   trigger arithmetic.
8. "This MoE gives us a 600B model at 40B cost." — Batch size, expert coverage, and which
   economics the claim is about.

The pattern across all eight: **name the regime, then the binding quantity, then check
whether the claim was measured in the same regime you operate in.** That is the part in
one sentence and the question worth having ready.

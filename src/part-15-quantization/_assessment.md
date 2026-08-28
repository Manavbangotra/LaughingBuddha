---
id: part-15-assessment
status: final
---

## How to use this

Four sections. The knowledge check takes about two hours. The assignment is a
**capacity plan with arithmetic**, because that is what this part is for and it is
the deliverable a real deployment needs. The challenge problems are open-ended.
The interview section is what to rehearse.

No answers are provided. Every question is answerable from the chapters.

## Knowledge check

**Formats**

1. Derive $x_{\max}$, $x_{\min}^{\text{norm}}$ and $x_{\min}^{\text{sub}}$ from
   {{eq:format-is-a-budget}} for E4M3 and E5M2.
2. FP16 gave 8× better relative error on weights and zeroed 9.33% of gradients
   where BF16 zeroed none. Explain why the better number is the wrong number, via
   {{eq:underflow-is-not-error}}.
3. The 16-bit split sweep gave two answers that disagreed at every dynamic range.
   Explain the disagreement using {{eq:energy-concentrates}}, and state the rule
   {{eq:optimal-split}} composes from them.
4. Why is unscaled FP8 E4M3 *worse* than E5M2 on a $\sigma = 0.02$ weight tensor,
   and what does a per-tensor scale change?
5. State {{eq:scale-factor-as-exponent}} and compute the group size at which a
   16-bit scale costs a quarter of a bit per weight.

**Theory**

6. Derive {{eq:errors-add-in-quadrature}} and explain what the measured exponents
   of 0.52, 0.50 and 0.49 confirm.
7. At 80 layers, what is the difference between square-root and linear error
   accumulation, and why does the whole practice depend on it?
8. State {{eq:outlier-inflates-the-step}} and compute {{eq:effective-levels}} at 4
   bits for $M/m = 4, 16, 64$.
9. Outliers cost 16.4× at fixed 8 bits and two bits of width cost 4.0×. What does
   that imply about how the bit-width decision is usually framed?
10. Explain why the fitted exponent drifts to 0.80 in the severe-outlier rows, and
    why that is not a change in accumulation.
11. Relative damage from identical 4-bit quantization grew +3.8% → +4008% across
    checkpoints. Name both mechanisms in
    {{eq:fragility-grows-with-training}} and the two caveats the listing states.

**INT8, INT4, and the four remedies**

12. Derive {{eq:outlier-budget}} and explain why outliers became urgent exactly
    when 4-bit did.
13. Per-channel activation scaling scored best and is not used. Explain via
    {{eq:reduction-axis-constraint}}, and say what SmoothQuant does about it.
14. Verify {{eq:difficulty-migration}} algebraically. What does it change about
    the model, and what does it change about what the quantizer sees?
15. State {{eq:output-error-is-the-target}} and explain why weight magnitude is
    the wrong importance measure.
16. Explain {{eq:rotation-flattens}} and compute the post-rotation outlier ratio
    at $k = 4096$. Why does rotation improve with width?
17. GPTQ had worse weight error and better output error. Prove
    {{eq:weight-error-floor}} and say what it implies about validating an
    implementation.
18. With an isotropic calibration GPTQ scored 0.1603 against round-to-nearest's
    0.1578. State the practical warning that follows.

**Bandwidth, cache and budget**

19. Derive {{eq:arithmetic-intensity}} and {{eq:memory-bound-crossover}}, and
    explain why the crossover is *linear* in bit-width.
20. Six extra operations per weight cost a laptop 61% of its speed and both GPUs
    nothing. Explain with {{eq:dequant-viability}}, and say what that implies about
    format design.
21. Bit allocation beat uniform by 1.67× at 4 bits and lost at 3 bits. Explain both
    results, and state the general lesson about Lagrangian allocation.
22. Explain why keys want per-channel and values per-token quantization, using
    {{eq:group-homogeneity}} *and* {{eq:streaming-forces-the-axis}}. Why is a
    static comparison the wrong experiment?
23. State {{eq:partner-absorbs-the-scale}} and say which of the four axis choices
    need migration.
24. Rank the three levers on KV memory by measured effect and say which is
    available at deployment time.
25. Compute {{eq:inference-budget}} for a 13B GQA-8 model at 32k context and batch
    24, and name the binding term.
26. A 70B GQA-8 model and a 7B MHA model fit the same concurrency. Explain with
    {{eq:capacity-ratio}}.
27. The prefill score matrix at 131k tokens was 2,199 GB. Explain
    {{eq:prefill-is-the-peak}}, and say why no hardware curve rescues it.
28. Why does a batch-48 deployment pass every sequential benchmark and fail in
    production? State both limits in {{eq:admission-control-memory}}.

**Runtimes and regimes**

29. Derive {{eq:static-occupancy}} and explain the measured 1.94×.
30. What does continuous batching make worse, and what does chunked prefill cost?
31. Rejection completed 487 of 500 requests. Why is its throughput number not
    comparable to the others'?
32. State {{eq:swap-versus-recompute}} and compute the threshold for the listing's
    constants.
33. Why is {{eq:wasted-work-fraction}} a better health metric than preemption
    count or utilisation?
34. Explain {{eq:cache-caps-throughput}} and why {{ch:q-gguf}}'s crossover does not
    occur once the cache is included.
35. {{eq:batch-buys-throughput}} contains no parameter count. Explain, and say what
    that implies for model selection in a throughput-bound deployment.
36. Why are speculative decoding and batching substitutes? Use
    {{eq:speculation-spends-idleness}}.
37. Why should a draft model be selected on agreement rather than quality?

## Assignment: a capacity plan, with arithmetic

Pick a model you might deploy and a machine you might deploy it on. **The
deliverable is a two-page capacity plan and the calculations behind it.** The
point is not to be right about the hardware — it is to produce a plan where every
number came from an equation rather than from a benchmark someone else ran.

**Establish the regime**

1. State the **batch size and context length** you expect, and place yourself in
   {{ch:q-throughput-latency}}'s regime table. Everything after this depends on
   it.
2. Compute {{eq:memory-bound-crossover}} and {{eq:cache-caps-throughput}} for your
   machine. Are you memory-bound at your batch? Will you be at your target batch?
3. Compute {{eq:batch-buys-throughput}} — your ceiling. Note that it does not
   depend on which model you chose.

**Choose the precision**

4. Measure $M/m$ per weight tensor on the checkpoint you will actually ship, and
   apply {{eq:outlier-budget}} at your candidate bit-widths.
5. State the **group size and scale precision**, and justify them with
   {{eq:group-size-cost}} against the alternative of one more bit.
6. If you use GPTQ or AWQ: **state the calibration set** and explain why it
   matches your deployment distribution. If it does not,
   {{sec:9-practical-example}} of {{ch:q-int8-int4}} says what that costs.
7. Measure the **decode speed on your target hardware class**, not on a different
   one, and check it against {{eq:dequant-viability}}.

**Size the memory**

8. Compute every term of {{eq:inference-budget}} at your real batch and context,
   and report the **binding term**.
9. Compute the **prefill peak** with the longest prompt you will accept
   ({{eq:fused-prefill-peak}}), confirm fused attention is in use, and set a chunk
   size from {{eq:chunk-size}}.
10. Set **two** admission limits ({{eq:admission-control-memory}}) and say what
    each is.
11. Compute {{eq:capacity-not-size}} and state the concurrency you can serve.

**Choose the configuration**

12. Compute {{eq:slo-to-throughput}} for your latency target and convert it into a
    cost per million tokens.
13. Decide the **preemption policy** from {{eq:swap-versus-recompute}}, measured
    on your machine.
14. If you are at low batch, evaluate speculative decoding: **measure the
    acceptance rate** for a candidate draft model and compute
    {{eq:optimal-depth}}. If you are at high batch, say why you are not evaluating
    it.

**The plan**

State, in order: the regime, the binding term, the three numbers you measured
rather than assumed, the largest lever you did *not* pull and why, and **what you
would re-measure when the checkpoint changes.** The last item exists because
{{eq:fragility-grows-with-training}} makes it necessary and almost nobody plans
for it.

## Challenge problems

**A. The arithmetic that contradicts a benchmark.** Find a published quantization
or serving benchmark, extract the regime it was run in (batch, context, hardware),
and compute what this part's equations predict. Where they disagree, work out
which term the benchmark's setup made dominant. **The interesting output is a
statement about what the benchmark measured, not about whether it was wrong.**

**B. Effective rank of the outlier problem.** For a real model, measure $M/m$ per
tensor and per activation channel, and identify which tensor would fail first at
each bit-width. Does {{eq:outlier-budget}} predict the ordering that
per-layer quantization error actually produces?

**C. Reconstruct the wasted-work fraction.** For a serving stack you use, derive
{{eq:wasted-work-fraction}} from whatever counters it exposes. If it exposes none,
say what would need to be added. Then run it to 95% cache utilisation and see
whether the number is nonzero.

**D. The checkpoint-fragility measurement.** Take two checkpoints of the same
model — different training durations, or a base and its instruction-tuned
descendant — and quantize both identically. Does
{{eq:fragility-grows-with-training}}'s direction hold?

**E. The streaming axis experiment.** Instrument a KV cache and measure
{{eq:group-homogeneity}} on real keys and values. Then measure accuracy **as a
function of position in the sequence** under chunked per-channel key quantization.
{{eq:streaming-forces-the-axis}} predicts position-dependent damage that no
standard evaluation would surface.

**F. Ceiling versus model.** For two models with very different parameter counts
and similar attention architectures, verify {{eq:batch-buys-throughput}}'s
prediction that their maximum decode throughput is the same. **If it is, that is a
model-selection argument worth making to whoever chooses models.**

## Interview preparation

**The questions that separate people who have deployed this from people who have
read about it:**

1. FP16 and BF16 are both 16 bits. What is the difference, and which would you
   use for what?
2. Why does quantization error not explode through an 80-layer model?
3. Which matters more: one more bit, or half the group size?
4. What is "4-bit" missing as a specification?
5. Your INT8 recipe worked at 7B and fails at 70B. What happened?
6. The best-scoring outlier fix is not used in production. Why?
7. When is GPTQ worse than doing nothing?
8. Why does 4-bit quantization roughly quadruple decode speed, and when does it
   not?
9. Why is a 3-bit format sometimes slower than a 4-bit one?
10. A 70B model at 4 bits is 35 GB. Does it fit on a 48 GB card?
11. Why might a 7B model and a 70B model serve the same number of users?
12. Your deployment passes every load test and OOMs in production. What is the
    likely cause?
13. Why is continuous batching faster, and what does it make worse?
14. Two teams disagree about whether an optimisation helped. How do you reconcile
    them without repeating either measurement?
15. What sets the maximum decode throughput of a machine, and why is the model not
    in that answer?

**On the last one**: {{eq:batch-buys-throughput}} is bandwidth divided by cache
bytes per token per unit of context. **Being able to say why the parameter count
is absent — and therefore why model selection for throughput should read
$L$, $h_{\text{kv}}$ and $d_h$ before the number in the model's name — is the
single most transferable thing in this part.**

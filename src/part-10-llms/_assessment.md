---
id: part-10-assessment
status: final
---

## How to use this

Four sections. The knowledge check takes about two hours and tells you what to
re-read. The assignment builds a serving harness — not a model — because
everything this part is about lives *around* the model, and building it is the
only way to see that clearly. The challenge is open-ended. The interview section
is what to rehearse.

No answers are provided. Every question is answerable from the chapters.

## Knowledge check

**The mechanism**

1. Trace a single token through {{eq:embed-step}}, {{eq:attn-sublayer}},
   {{eq:ffn-sublayer}}, {{eq:final-norm}}, and {{eq:unembed-step}}. At which
   step does the residual stream first contain information from another
   position?
2. {{eq:total-params}} and {{eq:block-params}} attribute parameters by
   component. For a 7B model, roughly what fraction is attention and what
   fraction is FFN? Why does that ratio matter for quantization?
3. Explain why weight tying ({{cite:press2017tying}}) is nearly free in
   parameters for a large model and expensive for a small one.
4. State {{eq:cross-entropy-decomposition}}. Which of its terms can training
   reduce and which is a property of the data?
5. A model's loss equals $\log |V|$ after an hour of training. Name the two most
   likely causes and the single measurement that distinguishes them.
6. Why is per-token loss a better training diagnostic than perplexity, given
   they are monotone transforms of each other?

**Decoding**

7. Derive {{eq:entropy-temperature-derivative}} and use it to explain why
   temperature 0.7 is not "70% as random as" temperature 1.0.
8. Contrast {{eq:top-k}} and {{eq:top-p}} on a peaked distribution and on a flat
   one. Which failure does each avoid that the other does not?
9. {{eq:length-normalised-score}} exists because {{eq:beam-search}} has a bias.
   State the bias precisely and explain why length normalisation is a heuristic
   rather than a correction.
10. Give the case where beam search is strictly worse than greedy decoding, and
    say what property of the search space produces it.
11. Why does {{eq:repetition-feedback}} make degeneration self-reinforcing, and
    why does {{cite:holtzman2020}} argue sampling fixes it rather than a better
    search?

**Inference and cost**

12. State {{eq:arithmetic-intensity-phases}}. A batch size of 1 wastes most of a
    GPU during decode — explain why in terms of that equation, not in terms of
    "underutilisation".
13. Decompose a p95 latency target using {{eq:ttft-decomposed}} and
    {{eq:total-latency}}. Which term does adding GPUs improve, and which does it
    make worse?
14. {{eq:max-concurrency}} bounds concurrent requests by KV cache, not by
    compute. Compute the bound for a 70B model with 80GB of memory and an 8k
    context, and say what MQA ({{cite:shazeer2019mqa}}) changes about it.
15. Explain why {{eq:queue-wait}} means a system at 80% utilisation has a worse
    tail than two systems at 40%.

**Prompting and structure**

16. {{eq:prompt-conditioning}} says a prompt is conditioning, not instruction.
    Give one observed behaviour that the "instruction" framing predicts wrongly.
17. {{cite:min2022}} found demonstration *labels* can be randomised with little
    loss. State what that implies about {{eq:demonstration-content}}, and what it
    does *not* imply.
18. Explain {{eq:self-consistency-condition}}: under what condition does
    sampling $k$ chains and taking the majority beat one greedy chain, and when
    does it reliably fail?
19. Why does {{eq:prompt-selection-bias}} mean a prompt tuned on twenty examples
    is likely to be worse in production than a simpler untuned one?
20. Distinguish prompted JSON from constrained decoding using
    {{eq:constrained-distribution}} and {{eq:valid-token-set}}. Which one can
    report a validity *rate* and which one cannot?
21. {{eq:constraint-cost}} shows constrained decoding is not free in quality.
    Explain the mechanism by which forcing a schema can make the content worse.
22. In {{eq:tool-call}} and {{eq:dispatch-loop}}, name every step the model
    performs and every step it does not. Where does the trust boundary sit?
23. {{eq:tool-chain-success}} compounds. For 92% per-call reliability, how many
    calls before end-to-end success drops below one half?

**Failure**

24. Give {{eq:hallucination-taxonomy}}'s categories and one mitigation that
    works for exactly one of them.
25. Explain {{eq:grounded-not-true}}: why is groundedness ({{eq:groundedness}})
    measurable when truth is not, and what does that buy?
26. {{eq:ece}} defines calibration error. Why does {{eq:alignment-confidence-shift}}
    predict that RLHF makes it worse, and what does {{cite:kadavath2022}}
    observe about the base model?
27. Read {{eq:risk-coverage}}. A system abstains on 44% of queries to reach 95%
    precision. Under what business condition is that a good trade, and under
    what condition is it worthless?
28. {{cite:liu2023lost}} reports the U-shape ({{eq:u-shape}}). State what
    {{eq:multi-fact-degradation}} adds that the single-needle result misses.
29. {{eq:usable-context}} distinguishes advertised from usable context. Design
    the measurement that establishes the second number for a given model.
30. {{eq:cascade-cost}} contains an unconditional $c_1$. Derive
    {{eq:cascade-breakeven}} and state the escalation rate above which a cascade
    loses to always-large at a 3× price ratio.

## Practical assignment: a serving harness with a cost model

Build the loop, not the model. Use any small causal LM you can run locally, or
the toy transformer from {{ch:llm-anatomy}} if you have no GPU — the point is
the harness, and a weak model makes the failure modes easier to see.

**Required components.**

1. **A decode loop** implementing greedy, temperature, top-$k$, top-$p$, and
   beam search behind one interface, plus correct incremental detokenization
   ({{eq:incremental-detokenization}}) and stop-string handling
   ({{eq:stop-string}}). Both of the latter are where real harnesses have bugs.
2. **A KV cache** with an explicit memory accounting that reports, per request,
   the bytes held ({{eq:phase-bytes}}) and the resulting concurrency bound
   ({{eq:max-concurrency}}).
3. **Constrained decoding** for a JSON schema of your choice via logit masking.
   Report the validity rate of the prompted baseline and of the constrained
   path. The constrained number must be exactly 1.000 or your mask is wrong.
4. **A tool-calling dispatch loop** ({{eq:dispatch-loop}}) with at least two
   tools, a call-depth limit, and explicit handling of a tool that errors.
5. **An abstention policy** using a confidence signal of your choice, with the
   risk–coverage curve ({{eq:risk-coverage}}) plotted rather than a single
   threshold reported.
6. **A two-model cascade** ({{eq:cascade-cost}}) with the escalation threshold
   stored as a *rate* and re-derived, per {{ch:llm-routing}}.

**Required measurements.** TTFT and ITL separately ({{eq:ttft}},
{{eq:itl}}) at batch sizes 1, 4, and 16; the prefill/decode cost split
({{eq:prefill-flops}}, {{eq:decode-flops}}); the quality/cost frontier of the
cascade against both single-model baselines; and calibration ({{eq:ece}}) of
whatever confidence signal the abstention policy uses.

**The deliverable is the measurement table, not the code.** Anyone can write a
decode loop. The part is about knowing what it costs.

## Advanced challenge

Pick one.

**Reproduce the U-shape and find where it breaks.** Build the needle test
({{eq:needle-test}}) for a model you can run, confirm the position curve, then
extend it to the two-fact and three-fact cases ({{eq:multi-fact-degradation}}).
Report the context length at which multi-fact retrieval falls below 50% and
compare it to the advertised window. Then vary the distractor similarity
({{eq:max-distractor}}) and report which factor — length or distraction —
dominates.

**Measure the cost of constraint.** {{eq:constraint-cost}} claims schema
enforcement can degrade content. Design an experiment that separates *the schema
is wrong for the task* from *the constraint hurt the reasoning*, on a task where
both are plausible. This is harder than it looks and the design is most of the
work.

**Build a router and then break it.** Train a router on a week of traffic,
deploy it against a cascade baseline, then simulate a model update by shifting
the confidence distribution and measure how far the escalation rate drifts
({{ch:llm-routing}}'s threshold-drift result). Report what monitoring would have
caught it, and how long it would have taken.

## Interview preparation

The questions that actually get asked, and what a strong answer contains.

**"Walk me through what happens when I send a prompt."** The answer is
{{ch:llm-prompt-lifecycle}}: template application, tokenization, prefill,
sampling, incremental detokenization, stop conditions. A strong answer names
prefill and decode as different machines without being asked.

**"Why is the second token faster than the first?"** Because prefill processed
$n$ positions and decode processes one, and the KV cache means it stays one.
Follow-up: why does that make decode memory-bound?

**"How do you guarantee valid JSON?"** The word *guarantee* is the test. Any
answer involving prompting, retries, or validation is a rate, not a guarantee.
The guarantee requires masking.

**"The model called a function — did it run it?"** No. It emitted a structured
token sequence; your harness ran it. Candidates who miss this write systems with
no trust boundary.

**"Our model hallucinates. Fix it."** The strong answer refuses the framing
politely: hallucination is a property of the objective, so the question is which
category ({{eq:hallucination-taxonomy}}), what groundedness rate is acceptable,
and whether abstention or retrieval is the cheaper mitigation.

**"When would you not use the biggest model?"** {{eq:cascade-breakeven}}. A
strong answer gives the break-even escalation rate and notes that the saving
scales with the price ratio, so routing between adjacent sizes rarely pays.

**"Your latency p95 regressed and nothing changed in the model."** Queueing
({{eq:queue-wait}}), context length growth, or batch composition. A strong
answer asks for the TTFT/ITL split before speculating.

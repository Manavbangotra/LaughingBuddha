---
id: part-14-assessment
status: final
---

## How to use this

Four sections. The knowledge check takes about two hours and tells you what to
re-read. The assignment runs a real fine-tune end to end, and — as in
{{part:12}} and {{part:13}} — the deliverable is a **decision memo backed by a
measurement table**, because every choice in this part is settled by a number you
either measured or did not. The challenge is open-ended. The interview section is
what to rehearse.

No answers are provided. Every question is answerable from the chapters.

## Knowledge check

**The decision, and what fine-tuning does not fix**

1. State {{eq:fine-tuning-decision}} and name the term in {{eq:adaptation-tco}}
   that decides it in practice and is usually omitted.
2. Fine-tuning teaches format reliably and facts unreliably. Explain the mechanism,
   and say what that implies about the boundary between {{part:12}} and this part.
3. A team reports that their fine-tune "did not work" and asks for more data. List
   the four things you would check first, in order, and which chapter each comes
   from.

**Supervised fine-tuning**

4. Bucketing beat packing at **0.998 against 0.854** token efficiency in
   {{ch:ft-sft}}. Explain both numbers, and explain why bucketing also helps with
   {{ch:ft-qlora-peft}}'s memory-*variance* problem.
5. At `max_len` 2048, 2.8% of examples were truncated and 6.7% of completion tokens
   were lost. Explain the 2.39× amplification, and then explain why the 8.3%
   figure is a worse problem than either.
6. Why is dropping over-long examples better than truncating them, given that both
   lose the same tokens?

**LoRA and PEFT**

7. State the low-rank hypothesis precisely. What is low rank?
8. Derive {{eq:eckart-young}}'s consequence for a rank-$r$ adapter on a rank-8
   update, and explain why {{eq:capacity-floor}} makes "train longer" useless
   below the wall.
9. Explain {{eq:learns-less-forgets-less}} as one mechanism rather than two
   findings, using the measured update norms 1.100 → 1.398.
10. Decompose {{eq:training-memory}} for a 7B model. Which term does LoRA remove,
    which does quantisation remove, and why do they compose?
11. Trainable parameters fall 99.9% and compute falls 33%. Derive the 1.5× ceiling
    from {{eq:compute-ratio}} and say why it holds for every method in
    {{ch:ft-qlora-peft}}.
12. LoRA scored **0.5856** on an output-offset task at rank 1 *and* rank 32.
    Explain via {{eq:peft-expressiveness}}, and state the diagnostic this gives you.
13. Serial and parallel adapters reach the same functions. Why does the serial form
    need **306×** the update norm at $\text{cond}(W_0) = 1000$, and what does
    {{eq:forgetting-quadratic}} do to that penalty?

**Data**

14. 1,000 stratified examples beat 30,000 random ones on macro accuracy and lost on
    the natural distribution. Explain both results with
    {{eq:macro-versus-aggregate}}, and say which is "correct".
15. State {{eq:metric-inherits-bias}} and explain why more evaluation data does not
    fix it.
16. A random split reported 0.811 where a group split reported 0.768. Derive
    {{eq:leakage-inflates}} and explain why the gap grows with model capacity.
17. Decontamination at a threshold moved the reported score from **+0.031** to
    **−0.031** while leaving 50% of the test set leaked. Explain both failures, and
    explain why they partly cancelling is worse than either alone.
18. Why does pushing the decontamination threshold harder *raise* the leaked share
    of the surviving test set?

**Synthetic data**

19. Eight generations of pure self-training left all 8 modes intact; adding a
    quality filter left 1. Explain why the filter is the dangerous component, using
    {{eq:quality-filters-rareness}} and {{eq:filter-contraction}}.
20. Grounding at 30% restored all 8 modes but only 6% of the tail mass against a
    true 20%. What exactly does grounding fix, and what does it not?
21. At 20% corruption the affected region scored 0.393 — below chance. Derive
    {{eq:systematic-noise-is-learnable}} and explain why the accuracy goes below
    0.5 rather than to it.
22. Explain {{eq:self-eval-agreement}} and why random errors are detectable from
    inside a pipeline while systematic ones are not.

**Preference optimisation**

23. Annotators agree 56.9%; the reward model measures 58.9% and is 71.3% accurate.
    Derive {{eq:p-from-agreement}} and {{eq:deconvolved-accuracy}}, and compute the
    correction.
24. Why is unbiased annotator noise a budget multiplier rather than a capability
    ceiling? What would make it a ceiling instead?
25. Binary feedback won at zero bar drift and collapsed at high drift while pairwise
    stayed flat. Prove {{eq:comparison-cancels-the-bar}}.
26. Why are randomly assigned annotators with different bars harmless, while routed
    ones are not? Connect to {{ch:ft-synthetic}}.

**Configuration, forgetting, merging**

27. The exchange rate ran 19.0 → 0.1 along one run. Derive {{eq:rate-collapses}}
    and explain why the first steps are nearly free.
28. Three learning rates across an order of magnitude gave nearly identical damage
    at matched distance. Explain with {{eq:distance-is-the-lever}}, and say what
    the residual divergence at long distances is.
29. Why does {{eq:rehearsal-gradient}} beat {{eq:anchor-penalty}}, and why is the
    gap small enough that nobody implements EWC?
30. State {{eq:lever-ratio}} and explain what it implies about where to spend
    effort.
31. Why does averaging two independently trained networks fail, and why does a
    shared base fix it? Use {{eq:permutation-symmetry}}.
32. A merge's best interpolation weight is 0.95. What does that tell you, and why?
33. Explain why {{eq:soup-variance}} and {{eq:conflict-governs-merging}} describe
    different situations that share a name.
34. Merge quality degraded eightfold as tasks went from aligned to opposed while
    the specialists stayed flat. What does that say about the ceiling of *any*
    merging algorithm?
35. The sign-conflict rate peaked at $\rho = -0.5$ while the worst merge was at
    $\rho = -0.9$. Explain the non-monotonicity.

## Assignment: one fine-tune, measured properly

Take a real task you care about and a base model you can run. **The deliverable is
a decision memo of at most three pages, plus the measurement table behind it.**
The point is not to produce a good model — it is to produce a defensible account
of what you traded.

**Before any training**

1. Write the **skill taxonomy** ({{ch:ft-datasets}}). It does not need to be right;
   it needs to exist. Report the number of strata and the count in the rarest.
2. Record a **provenance key** for every example, and use it for the split
   ({{eq:group-split}}). If you cannot, say so explicitly and state that your
   held-out numbers carry an unknown positive bias.
3. Choose and record the **head/tail allocation**. This is a product decision;
   name who made it.
4. If any data is synthetic, sample **thirty failures and cluster them**
   ({{eq:concentration-test}}). Report the taxonomy of errors, not the rate.
5. Establish the **base-capability benchmark** you will track. Anything that told
   you the base model was good enough will do.

**Configuration**

6. Compute {{eq:training-memory}} and {{eq:activation-memory-unchanged}} for your
   real batch and sequence length, before launching. Report both.
7. Decide the PEFT method by **what kind of change your task needs**
   ({{eq:peft-expressiveness}}), and say why — not by parameter efficiency.
8. Choose the rank, and state whether you chose it from
   {{eq:effective-rank}}, from a sweep, or by convention.
9. Write down $\omega$ or $R_{\min}$ ({{eq:joint-stopping}}, {{eq:stop-on-rate}})
   **before the run**.

**During and after**

10. Log at every checkpoint: target-task loss, **base-capability score**, and
    $\|\theta - \theta_0\|$.
11. Report the **exchange rate curve** and the checkpoint your stated rule selected
    — alongside the checkpoint the naive rule would have selected, and the
    difference between them.
12. Report **macro and aggregate** metrics, with the taxonomy.
13. Run one **mitigation** (rehearsal is the cheapest) and report the improvement
    at matched target-task quality.
14. If you produced more than one fine-tune: run the **sign-agreement screen**, and
    report {{eq:merge-efficiency}} if you merged.

**The memo**

State, in order: what you traded, what it cost, what you did not measure, and
which of your numbers you do not trust and why. **The last two sections are the
ones that make it a memo rather than a report**, and they are the ones this part
has been building toward.

## Challenge problems

**A. The measurement that is not corrupted.** Three chapters here found an
evaluation corrupted by the process it measures — selection bias, generator
beliefs, annotator noise. Find a *fourth* instance in your own pipeline, name the
process and the corruption, and design the external check that breaks it. This is
the highest-value exercise in the part.

**B. Effective rank in the wild.** Fine-tune one layer of a real model without a
rank constraint, take the SVD of the delta, and report $\rho_{\text{eff}}$ at
thresholds 0.9, 0.95 and 0.99. Compare against the rank you would have chosen by
convention. Does {{ch:ft-lora}}'s claim that rank is measurable survive contact
with a real model?

**C. Two collapse mechanisms, separated.** Build a synthetic-data pipeline and
measure diversity every round. Show, in your setting, which of
{{eq:collapse-recursion}} and {{eq:filter-contraction}} dominates, and at what
sample size the crossover sits.

**D. Deconvolve a published result.** Find a reward-model or preference-optimisation
result that reports accuracy without an agreement rate. Estimate the agreement rate
from any source you can, apply {{eq:deconvolved-accuracy}}, and report how the
conclusion changes.

**E. The stopping decision, priced.** For a real fine-tune, produce the exchange
rate curve and present the actual menu of checkpoints to whoever owns the product.
Report which one they chose and why. **The deliverable is the conversation, not the
model.**

**F. Merge screens on real adapters.** Take two adapters over one base. Run the
shared-ancestor and sign-agreement screens, predict whether merging will work,
merge, and report {{eq:merge-efficiency}}. Then break a precondition deliberately —
quantise one base — and confirm the failure mode.

## Interview preparation

**The questions that separate people who have done this from people who have read
about it:**

1. When would you *not* fine-tune? Give three cases and the alternative for each.
2. Decompose fine-tuning memory. Which term binds, and when does that change?
3. Trainable parameters fall 99.9%. What happens to wall-clock time?
4. Your LoRA run plateaus above target. How do you distinguish a capacity floor
   from under-training, and how do you distinguish both from the wrong PEFT method?
5. Is LoRA as good as full fine-tuning?
6. How would you choose the rank without a sweep?
7. Your held-out score is excellent and production is not. Name three mechanisms
   from this part that produce exactly that, and the diagnostic for each.
8. Your synthetic data is 95% accurate. What else do you need to know?
9. Your reward model scores 58%. Is that bad?
10. Which is the bigger lever on catastrophic forgetting: learning rate, rehearsal,
    or the stopping point? Justify with numbers.
11. When can you merge two fine-tuned models, and what bounds the result?
12. Your merge's best interpolation weight is at the edge of the range. What
    happened?
13. What is the single cheapest thing a team can do to improve a fine-tuning
    project, and why does almost nobody do it?

**On the last one**: there are three defensible answers in this part — write the
taxonomy, record the provenance key, log base capability per checkpoint. All three
cost under a day, all three are decided before training starts, and none can be
added afterwards. **Being able to say why that is the shape of the answer matters
more than which one you pick.**

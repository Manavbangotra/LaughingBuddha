---
id: part-14-intro
status: final
---

## What this part is for

{{part:12}} and {{part:13}} moved knowledge *outside* the weights. This part is
about changing the weights, and it opens with a chapter arguing you usually should
not.

**The hazard is that fine-tuning is the most over-prescribed intervention in
applied AI.** It is what people reach for when a model is not doing what they
want, and it is usually the wrong tool: prompting is cheaper, retrieval is more
appropriate for facts, and the actual problem is frequently that nobody has
written down what "doing what they want" means. The literature does not help,
because papers fine-tune by construction — nobody publishes "we tried prompting
and it was fine".

> **The rule adopted for this part: every chapter must state what its technique
> does NOT fix.** {{ch:ft-when}} is a decision chapter with a negative default.
> {{ch:ft-sft}} says fine-tuning teaches format reliably and facts unreliably.
> {{ch:ft-lora}} reports the measured trade rather than the parameter-count
> headline. {{ch:ft-datasets}} says curation is a trade and names what it sells.
> {{ch:ft-merging}} says conflict is a floor no algorithm goes below.

## The organising idea

**A fine-tune moves weights, and every method here is a different answer to: how
far, in which directions, and what does the movement destroy?**

```text
   HOW MUCH MOVEMENT         WHAT IT BUYS               WHAT IT COSTS
   ───────────────────────   ────────────────────────   ────────────────────────
   129 none (prompt)         instant, reversible        a ceiling on behaviour
   130 all weights           maximum capability         forgetting, cost, copies
   131 a low-rank slice      cheap, composable          learns less (measured)
   132 + a quantised base    fits on one GPU            no speed-up, base binding
   133 (selection, not       coverage of the tail       accuracy on the head
        movement)
   134 (generation)          volume                     the tail, again
   135 preference-shaped     what you cannot write      a measurement ceiling
   136 stopped deliberately  a chosen exchange rate     target-task quality
   137 arithmetic on it      one model instead of two   quality on each task
```

The through-line, stated in {{ch:ft-when}} and returned to in {{ch:ft-merging}}:
**a fine-tune produces a weight DELTA, and that delta is an object with
structure.** It has a rank ({{ch:ft-lora}}), a norm that predicts the damage
({{ch:ft-training-config}}), a precision it belongs to ({{ch:ft-qlora-peft}}), and
it can be added, scaled and subtracted ({{ch:ft-merging}}).

**The second through-line is about measurement**, and it caught this part by
surprise three times. {{ch:ft-datasets}}'s evaluation inherits the training
selection's bias; {{ch:ft-synthetic}}'s inherits the generator's beliefs;
{{ch:ft-preference}}'s is capped by the annotators' agreement. **In all three the
number you would use to decide is corrupted by the same process that produced the
thing being measured**, and in all three the fix is to introduce something the
process did not produce.

## Eight things worth knowing before you start

**The parameter-count headline is about the wrong resource.**
{{ch:ft-qlora-peft}} decomposes a 7B fine-tune's memory: **126 GB**, of which the
weights are 14 GB and everything the optimiser needs is **112 GB — 89%.** LoRA
removes the optimiser terms, quantisation removes the weight term, and they
compose because they attack different factors. **Neither touches activations**,
which at 70B are 81.6 GB checkpointed against 35 GB of quantised weights.

**And PEFT is not a speed-up.** Trainable parameters fall 99.9%; compute falls
**33%**, capped at 1.5× whatever you freeze, because the backward pass must
traverse every frozen layer to reach an adapter. A ten-hour run becomes seven
hours, not ten minutes.

**Rank is a capacity limit, not a quality dial.** {{ch:ft-lora}} measures knees
landing exactly at each task's intrinsic rank — 0.4009 → **0.0013** across
$r = 1 \to 2$ for a rank-2 task. Below the wall, more steps and more data cannot
help. **And the rank is measurable**: the effective rank of an unconstrained
delta recovered 2, 8 and 30 against true ranks of 2, 8 and 32.

**"Is LoRA as good as full fine-tuning" has two answers pointing opposite ways.**
Task-B error fell 0.0890 → 0.0155 with rank while task-A error rose 0.1110 →
0.1421, both tracked by the update norm's 1.100 → 1.398. **One mechanism seen
twice**, and it makes the decidable question *how much new capability do I need,
and how much old capability must survive*.

**Allocation beats collection.** {{ch:ft-datasets}} shows 1,000 stratified
examples scoring **0.639** against 30,000 random ones at **0.630** — thirty times
the data, and it loses. **And it is a trade**: on a production-distributed test set
the ranking reverses at every budget.

**A random split lies about clustered data**, by **+0.042** at a 50% duplication
rate against a clean 0% control. And threshold decontamination fails in *two*
directions: it overshoots the truth (+0.031 → −0.031) while leaving **50%** of the
surviving test set still leaked.

**The dangerous component in a synthetic-data pipeline is the filter, not the
recursion.** {{ch:ft-synthetic}} ran eight generations of self-training at 4,000
examples per round with **all 8 modes intact**. Adding the standard quality filter
took it to **1 mode, with the tail gone after a single generation** — because a
rare mode has low likelihood *because* it is rare.

**And forgetting has an exchange rate that collapses.**
{{ch:ft-training-config}} measures **19.0 → 5.5 → 1.9 → 1.0 → 0.3 → 0.1** along
one run: by step 12 the new task had gained 59.1% of its total for 14.6% of the
damage, and the rest of the new task cost the other 85%. **The standard stopping
rule takes that second deal without being asked.**

## What this part deliberately does not cover

**RLHF, DPO's derivation, and reward modelling are {{part:9}}'s.** This part is
their *practice*: where preference data comes from, what agreement does to your
ability to measure anything, and why the format choice depends on your annotators
rather than on the objective.

**Quantisation theory is {{part:15}}'s.** {{ch:ft-qlora-peft}} uses the result and
forward-references it.

**Serving many adapters is {{part:23}}'s.** This part stops at producing them.

**Evaluation infrastructure is {{part:25}}'s** — though three chapters here
contribute to it, because each found a way the standard measurement is wrong.

## How to read it

{{ch:ft-when}} is the most important chapter and the one most likely to be
skipped, because it argues against the thing you came here to do. Read it even if
you have already decided.

{{ch:ft-lora}} and {{ch:ft-training-config}} are the theoretical spine: the first
establishes that movement is what you spend, the second prices it. Everything
between them is a way of spending less or spending it better.

{{ch:ft-datasets}}, {{ch:ft-synthetic}} and {{ch:ft-preference}} are the data
chapters, and they share a structure worth noticing on a second reading. Each
identifies a measurement that is corrupted by the process it is measuring, and
each fix is the same shape: **introduce something the process did not produce** —
a provenance key, an external oracle, a double-labelled calibration set.

{{ch:ft-merging}} is the last chapter because it is what you do when the answer is
"two models", and knowing when that is the answer requires everything before it.

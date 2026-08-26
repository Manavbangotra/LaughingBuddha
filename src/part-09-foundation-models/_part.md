---
id: part-09-intro
status: final
---

## What this part is for

{{part:8}} ended with the encoder era losing pretraining to causal language
models. This part is what the winners did next — and it is the part where the
book's evidentiary standards have to change, because the material does.

Count the venues. Of the sixteen works this part introduces, six have no
peer-reviewed venue at all. PPO, among the most-cited papers in machine
learning, was never formally published. Several are written by the
organisations selling the models they describe. And the numbers that would
settle most disagreements — what is in the training data, what the frontier
recipes are — are not public and will not become public.

> **The rule adopted for this part: when a result comes from a lab reporting on
> its own product, say so in the sentence that reports it.** That is not
> scepticism for its own sake. It is the only way to write nine chapters about
> systems whose construction is largely undisclosed without quietly implying
> more certainty than exists.

## The organising idea

One change in kind drives everything here: **the model stops being trained for a
task and starts being trained for everything, after which the task becomes an
inference-time concept.**

Every stage that follows is a patch for something the stage before it did not
contain.

```text
   THE CAPABILITY                 THE CORRECTIONS              THE COST
   ────────────────────────       ──────────────────────       ─────────────
   79 what foundation models      84 instruction tuning        87 distillation
      are, and the pipeline          — a text continuer           — making it
   80 pretraining as an              does not answer              servable
      engineering process         85 RLHF — "be helpful"
   81 the corpus that feeds it       is not a loss function
   82 scaling laws — how big     86 DPO — the same
   83 emergence — what scaling       objective, without
      does NOT predict               the machinery
```

Pretraining optimises next-token prediction on web text, which is not what
anyone wants — it is what can be optimised at scale. Instruction tuning exists
because such a model does not recognise a request as a request. RLHF exists
because "helpful and harmless" cannot be written as a loss. Distillation exists
because the result is too expensive to serve.

**Read the part as that chain, not as a list of techniques.** The most useful
question to carry through it is *what deficiency is this stage correcting?*

## Three things worth knowing before you start

**Adaptation cannot install what pretraining lacks.**
{{eq:adaptation-information-ratio}} puts fine-tuning about eight orders of
magnitude below pretraining in information supplied. That single inequality
explains why instruction tuning teaches format reliably and facts poorly, why
alignment tilts a distribution rather than creating behaviour, and why
{{part:12}}'s retrieval has to exist at all — supplying knowledge through the
*context* sidesteps the bound entirely.

**A third reported advance dissolved when someone equalised the budget.**
{{cite:levy2015}} found it for word embeddings in {{part:8}};
{{cite:liu2019roberta}} found it for encoder pretraining; and
{{cite:kaplan2020scaling}} versus {{cite:hoffmann2022chinchilla}} is the same
failure again, where a learning-rate schedule shared across runs of different
lengths inverted the field's allocation rule for two years. Both papers fitted
hundreds of models and reported tight confidence intervals. **Statistical rigour
did not protect against a design error**, and the habit this should produce —
asking what was held fixed before believing a comparison — is worth more than
any individual result in these nine chapters.

**Scaling laws predict loss, and loss is not capability.**
{{ch:fm-scaling-laws}} shows the curves are remarkably good.
{{ch:fm-emergence}} shows the step from there to "can it write correct code" is
where the field's ability to plan runs out, and that a measurement choice can
manufacture a qualitative phenomenon out of a quantitative one.

## What is genuinely unsettled

**Whether emergent abilities are real.** {{cite:wei2022emergent}} documented
sharp capability curves; {{cite:schaeffer2023}} showed a discontinuous metric
manufactures sharpness from smooth improvement. Both are partly right, neither
settles it, and a third explanation — contamination correlating with scale —
survives every rescoring and is the least often audited.

**Whether the RL in RLHF was ever load-bearing.** {{cite:rafailov2023}} replaces
a three-stage pipeline with one classification loss and comparable results. If
that holds, the value was always in the preference data. Frontier labs largely
retain online methods, and the best remaining argument is the online/offline
distinction rather than anything about the reward model.

**What data quality is worth.** Neither {{cite:kaplan2020scaling}} nor
{{cite:hoffmann2022chinchilla}} contains a data-quality term at all, while
{{cite:gunasekar2023}} beats their curves at small scale with curated data and
{{cite:lee2022dedup}} shows duplicated tokens are worth less than unique ones.
The laws are missing a variable and nobody has a definition of it that is both
measurable in advance and useful.

## A note on {{ch:fm-dpo}}

{{ch:fm-rlhf}} derives the closed-form optimum of the KL-regularised objective
in full — more fully than the result seems to need. That is deliberate.
{{ch:fm-dpo}} is three algebraic steps from that equation, and read in sequence
the derivation is one of the most satisfying passages in the book: a reward
model, a value model, a sampling loop, and PPO all disappear, with nothing
approximated.

The two chapters should be read together, in order, in one sitting.

## What you should be able to do at the end

Plan a pretraining run from a compute budget, a traffic forecast, and a corpus
audit — and know which of the three usually binds first. Diagnose a training run
from two numbers in its first ten minutes. Implement MinHash deduplication and
explain why it improves quality rather than merely saving compute. Evaluate an
emergence claim by asking what the metric is, how many scale points there were,
and whether anyone checked contamination. Derive DPO from RLHF's optimum. And
compute, rather than assert, whether distillation pays back at your volume.

Above all: **state what kind of evidence a claim rests on, in the sentence that
makes the claim.** Nine chapters of mostly-unrefereed sources is good practice
for the rest of the field.

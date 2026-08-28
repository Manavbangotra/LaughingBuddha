# Part XIV — Fine-Tuning and Model Adaptation: research notes

Research pass run 2026-08-29, before writing. Full tier: 21 sections per chapter,
4,200-word floor, nine chapters. Twelve new bibliography entries, each verified
against an arXiv abstract page on the date above. 231 entries total, none
unverified.

## What this part is, and what it is not

{{part:12}} and {{part:13}} moved knowledge *outside* the weights. This part is
about changing the weights, and it opens with the chapter arguing you usually
should not.

**The hazard here is that fine-tuning is the most over-prescribed intervention in
applied AI.** It is what people reach for when a model is not doing what they
want, and it is usually the wrong tool: prompting is cheaper, retrieval is more
appropriate for facts, and the actual problem is frequently that nobody has
written down what "doing what they want" means. The literature does not help,
because papers fine-tune by construction — nobody publishes "we tried prompting
and it was fine".

> **The rule adopted for this part: every chapter must state what the technique
> does NOT fix.** Concretely — {{ch:ft-when}} is a decision chapter with a
> negative default; {{ch:ft-sft}} says fine-tuning teaches format reliably and
> facts unreliably; {{ch:ft-lora}} reports the measured rank gap rather than the
> parameter-count headline.

## The organising idea

**Fine-tuning moves weights, and every method in this part is a different answer
to: how far, in which directions, and what does the movement destroy?**

```text
   HOW MUCH MOVEMENT        WHAT IT BUYS            WHAT IT COSTS
   ─────────────────────    ────────────────────    ──────────────────────
   129 none (prompt)        instant, reversible     ceiling on behaviour
   130 all weights          maximum capability      forgetting, cost, copies
   131 a low-rank slice     cheap, composable       learns less (measured)
   132 + quantized base     one GPU                 numerical headroom
   135 preference-shaped    what you cannot label   reward hacking
   136 constrained          keeps old capability    slower adaptation
   137 arithmetic on it     compose without training interference
```

The through-line to state in {{ch:ft-when}} and return to in {{ch:ft-merging}}:
**a fine-tune produces a weight DELTA, and that delta is an object with
structure** — it has a rank ({{cite:hu2021lora}}), it can be added and subtracted
({{cite:ilharco2023taskarithmetic}}), it interferes with other deltas
({{cite:yadav2023ties}}), and its size determines how much of the original model
it overwrites ({{cite:kirkpatrick2017ewc}}).

## The genuinely live questions

### 1. Does LoRA match full fine-tuning?

The most-repeated claim in the part and the one with the clearest answer.
{{cite:biderman2024loralearnsless}} is the controlled comparison: **LoRA learns
less AND forgets less**, and the mechanism is a measured rank gap of 10–100×
between what full fine-tuning changes and what a typical LoRA configuration can
represent.

**That is a trade, not a tie**, and the chapters should say so. The right question
is not "is LoRA as good" but "how much new capability do I need, and how much old
capability must survive" — which is answerable per project.

### 2. How much data does fine-tuning need?

{{cite:zhou2023lima}}'s 1,000 examples is the load-bearing result, and its
interpretation matters more than its number: **the knowledge is already in the
pretrained weights, and instruction tuning mostly selects a response format.**

So the scaling behaviour is unlike pretraining — quality dominates quantity, and
past a small number the returns are close to flat. {{ch:ft-datasets}} should
demonstrate that rather than assert it, because "we need more data" is the default
diagnosis for a fine-tune that did not work and is usually wrong.

### 3. Is synthetic data a solution or a trap?

Both, and the honest treatment separates two failure modes.
{{cite:wang2023selfinstruct}} made instruction data a compute problem, which is
real and enormous. The trap is **diversity collapse**: a model generating its own
training data samples from its own distribution, so the data inherits its modes
and its blind spots, and filtering for quality makes this *worse* by removing the
tail.

{{ch:ft-synthetic}} should measure the diversity loss rather than warn about it,
and should state the one thing that reliably fixes it — grounding generation in
real, varied source material rather than in the model's prior.

### 4. What does fine-tuning destroy?

{{cite:kirkpatrick2017ewc}} named it and the effect is routinely under-measured,
because people evaluate the fine-tuned model on the task they fine-tuned for.

**The measurement that matters is the one nobody runs**: general capability before
and after. {{ch:ft-training-config}} should make forgetting a first-class measured
quantity and connect it to {{cite:biderman2024loralearnsless}}'s finding that
parameter-efficient methods forget less *because* they move less.

### 5. Is merging real or folklore?

Real, and {{cite:wortsman2022modelsoups}} is why: models fine-tuned from one base
live in a connected low-loss region, so averaging works. {{cite:ilharco2023taskarithmetic}}
makes the delta an algebraic object and {{cite:yadav2023ties}} names the two
interference mechanisms — redundancy and sign disagreement.

**State the precondition loudly**: merging works between models sharing a base.
It is not a general model-combination technique, and the failures people report
are almost always violations of that precondition.

## Per-chapter findings

### 129 — When to Fine-Tune and When Not To

A decision chapter, and the part's most important. Content: the ladder — prompt,
few-shot, retrieve, fine-tune — and what each rung is *for*. The
{{ch:fm-what-they-are}} adaptation-information argument: fine-tuning teaches
format reliably and facts poorly, which is why {{part:12}} exists.

**Listing:** a cost/latency/quality model across the ladder, with the crossover
computed — including the maintenance cost of a fine-tune, which is the term that
decides it and is always omitted.

### 130 — Supervised Fine-Tuning

Content: the loss, masking prompt tokens, packing, and what SFT actually changes.
The demonstrable claim: **SFT reliably teaches format and unreliably teaches
facts**, and the difference is measurable on the same training run.

### 131 — LoRA and the Low-Rank Hypothesis

The hypothesis stated properly — the *update* is low rank, not the weights — and
then tested. Per live question 1: report the rank gap, and show what rank buys
and what it costs.

**Listing:** approximate a known-rank update at varying LoRA rank and measure
recovery; then show the forgetting side of the same experiment.

### 132 — QLoRA, PEFT, and Adapters

{{cite:dettmers2023qlora}}, {{cite:houlsby2019adapters}},
{{cite:li2021prefixtuning}}. Content: the PEFT family as points on one axis —
where the trainable parameters live — plus the memory arithmetic that makes QLoRA
work. Do not re-teach quantization; {{part:15}} owns it, so forward-reference.

### 133 — Dataset Creation for Fine-Tuning

Per live question 2. The measurement to make: **quality against quantity on the
same budget**, showing curation beating scale.

### 134 — Synthetic Data and Data Quality

Per live question 3. Measure diversity collapse across generations, and measure
what quality filtering does to it.

### 135 — Preference Optimization in Practice

{{cite:rafailov2023}}, {{cite:ethayarajh2024kto}}, with {{part:09}}'s RLHF as
background rather than repetition. The practical content: what preference data
costs, what binary feedback can substitute for, and reward hacking as an expected
condition.

### 136 — Training Configuration, Catastrophic Forgetting, Overfitting

Per live question 4. Learning rate, epochs, and the forgetting measurement.
**Forgetting is the chapter's spine**, not a section.

### 137 — Model Merging and Distillation

Per live question 5, plus {{cite:hinton2015}}. The precondition, the two
interference mechanisms, and merging as the cheap alternative to multi-task
training.

## Cross-part bookkeeping

- **Do not** re-teach RLHF, DPO's derivation, or reward modelling —
  {{part:09}} owns them. This part is their *practice*.
- Quantization theory is {{part:15}}; {{ch:ft-qlora-peft}} uses it and
  forward-references.
- Serving many adapters is {{part:23}}; this part stops at producing them.
- Evaluation of fine-tuned models is {{part:25}}; use its vocabulary, do not
  duplicate its infrastructure.
- Terminology collision check before writing: `adapter`, `rank`, `merge`,
  `distillation`, `alignment`, `forgetting` — `alignment` certainly collides with
  {{part:09}} and with {{ch:emb-what-they-are}}'s geometric sense, and must be
  disambiguated on first use.
- Reuse, do not restate: {{eq:cascade-cost}}, {{eq:risk-coverage}},
  {{eq:identity-embedding}}, {{eq:prior-as-data}}.

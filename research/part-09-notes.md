# Part IX — Foundation Models: research notes

Research pass run 2026-08-25, before writing. Full tier: 21 sections per
chapter, 4,200-word floor, nine chapters. Sixteen new bibliography entries,
every one verified against an arXiv abstract page or proceedings listing on the
date above. Two entries already present — `kaplan2020scaling` and
`hoffmann2022chinchilla` — are load-bearing here and were verified in Part VII.

## The problem this part has to solve

Part VIII ended with the encoder era losing pretraining to causal language
models. This part is what the winners did next, and it has a structural hazard
the earlier parts did not: **most of the primary sources are not peer reviewed,
several are written by the organisations selling the models, and the numbers
that matter most are the ones nobody publishes.**

Count the venues in the new bibliography: `brown2020`, `ouyang2022`,
`rafailov2023`, `schaeffer2023` list no venue on arXiv and were accepted at
NeurIPS; `schulman2017ppo`, `bai2022`, `touvron2023llama`, `gunasekar2023`,
`gao2020pile`, and `bommasani2021` have no peer-reviewed venue at all. PPO is
among the most-cited papers in the field and was never formally published.

That is not a reason to avoid them. It is a reason to be explicit, every time,
about what kind of evidence a claim rests on. **The rule for this part: when a
result comes from a lab reporting on its own product, say so in the sentence
that reports it.**

## The organising idea

Part VI built the network, Part VII built the architecture, Part VIII fed it.
This part is about a single change in kind: **the model stops being trained for
a task and starts being trained for everything, after which the task becomes an
inference-time concept.**

Everything follows from that. In-context learning ({{cite:brown2020}}) exists
because there is no task-specific training step left to put the task in.
Instruction tuning ({{cite:wei2022flan}}) exists because a model trained to
continue text does not recognise a request as a request. RLHF
({{cite:ouyang2022}}) exists because "be helpful" cannot be written as a loss.
Each stage is a patch for something the previous stage's objective did not
contain, and the part is best read as that chain of patches rather than as a
list of techniques.

The through-line to state in {{ch:fm-what-they-are}} and return to at the end:
**every alignment stage exists because the pretraining objective is a proxy.**
Next-token prediction on web text is not what anyone wants; it is what can be
optimised at scale. Everything after it is correcting the gap, and the gap never
fully closes.

## What changes at this tier for this material

The mathematical content is thinner than Parts VI–VII and the *empirical* content
is much thicker. The sections have to be earned differently:

- **§6 Mathematical Foundation.** The genuine derivations are: the scaling-law
  power form and its compute-optimal allocation, the Bradley–Terry preference
  model and its log-likelihood, the KL-regularised RLHF objective and its
  closed-form optimum, the DPO reparameterisation that follows from it, and the
  distillation temperature-gradient relationship. That is enough for nine
  chapters — the DPO derivation in particular is short, complete, and one of
  the most satisfying results in the book.
- **§10 Production Considerations.** Unusually strong here: pretraining is the
  only activity in the book where a single mistake costs seven figures, and
  checkpoint/restart/data-order discipline is real engineering.
- **§12 Failure Modes.** Reward hacking, over-optimisation, alignment tax,
  catastrophic forgetting across stages, contamination.
- **§19 Research Questions.** Genuinely open, and several are settled less than
  the secondary literature suggests — see below.

**Padding risk is highest in {{ch:fm-what-they-are}} and {{ch:fm-datasets}}**,
where the temptation is to enumerate models and corpora. Rule for this part: a
named model earns a paragraph only if it isolated a variable. LLaMA earns one
({{cite:touvron2023llama}} tested over-training past Chinchilla). phi earns one
({{cite:gunasekar2023}} tested data quality at fixed scale). A model that is
"the same recipe, larger" earns a table row.

## The genuinely live questions

### 1. Are emergent abilities real?

Not settled, and the confident answer in either direction is wrong.

{{cite:wei2022emergent}} documented tasks where performance is near chance until
some scale and then rises sharply, and defined emergence as unpredictability
from smaller models. {{cite:schaeffer2023}} argued the sharpness is
manufactured by the metric: exact-match accuracy on a multi-step task is a
step function applied to a smoothly improving per-step accuracy, and under a
continuous metric the curves are smooth.

**Both can be right, and mostly are.** Schaeffer et al. demonstrate that many
published emergence curves dissolve under a continuous metric. They do not
demonstrate that nothing is ever discontinuous, and they do not remove the
practical problem: if the deliverable is exact-match — the program compiles, the
JSON parses, the answer is correct — then the discontinuity is real *for the
user* regardless of what the underlying quantity does.

**What to write:** the distinction between a discontinuity in the model's
capability and a discontinuity in the metric's view of it, and the observation
that product requirements are usually stated in discontinuous metrics. What not
to claim: that emergence has been debunked, or that it is established.

### 2. Do the scaling laws hold, and what do they leave out?

{{cite:kaplan2020scaling}} and {{cite:hoffmann2022chinchilla}} disagreed and the
second won, which is already in Part VII. What matters here is what *both* hold
fixed: **data quality, which appears in neither functional form.**

{{cite:gunasekar2023}} is the direct challenge — a small model on a curated and
synthetic corpus outperforming what the scaling curves predict at that size. The
paper attracted a contamination concern that has not been fully resolved, which
should be stated alongside the result rather than after it.

And {{cite:touvron2023llama}} showed the compute-optimal point is not the
deployment-optimal point: training compute is paid once and inference compute is
paid per request forever, so the right model is usually smaller and
over-trained relative to Chinchilla. That correction is now standard practice
and is the single most useful thing in the chapter.

### 3. Does RLHF work because of RL?

Increasingly doubtful, and {{cite:rafailov2023}} is why.

DPO shows the KL-regularised RLHF objective has a closed-form optimum relating
the optimal policy to the reward, so preference data can train the policy
directly with a classification loss — no reward model, no sampling loop, no PPO.
It performs comparably on most benchmarks.

If a three-stage RL pipeline can be replaced by one supervised loss with
comparable results, then the RL machinery was not where the value was. **The
value is in the preference data.** That is worth stating plainly, because the
secondary literature treats RLHF's complexity as essential rather than
incidental.

The honest qualification: frontier labs largely still use online methods, and
the evidence that DPO closes the gap at the very top is weaker than the evidence
that it closes the gap for open models. Report both.

### 4. What is the alignment tax, and is it real?

{{cite:ouyang2022}} reports capability regressions on some benchmarks after
alignment, mitigated by mixing pretraining gradients into the RL stage. The
size of the tax and whether it is fundamental or an artefact of a particular
recipe is not settled, and it is measured differently by everyone who measures
it.

The related and better-documented failure is **reward over-optimisation**:
{{cite:stiennon2020}} showed that pushing further against a learned reward makes
true quality fall past a point. The KL penalty exists to bound exactly this.

### 5. What is actually in the training data?

Nobody outside the labs knows for the frontier models, and this is a genuine
epistemic limit on the whole part rather than a gap to gloss.

{{cite:gao2020pile}} made composition a studiable variable.
{{cite:lee2022dedup}} showed near-duplicates are pervasive, that removing them
improves models, and — the finding with the widest consequences — that train/test
overlap inflates reported benchmark results. Any benchmark comparison in the
remaining twenty parts inherits that caveat.

## Per-chapter findings

### 79 — What Foundation Models Are

Definitional, and the chapter has to justify a term coined by
{{cite:bommasani2021}} in a report from an institution that also builds them.
Take the definition seriously and interrogate it: trained on broad data at
scale, adaptable to many tasks. The interesting content is the
**homogenisation** argument — when everything is built on a few bases, defects
propagate to everything above — which is the part of that report that has aged
best.

The chapter should establish the pipeline the rest of the part follows:
pretrain → instruction-tune → align → (distil), with each stage introduced as
the correction to a specific deficiency of the one before.

### 80 — Pretraining and Self-Supervised Objectives

Causal language modelling as the objective, and why it won: supervision at every
position ({{ch:nlp-bert}}'s accounting), objective identical to the deployment
task, and generation as a universal interface.

The real content is engineering: what a pretraining run actually is. Data order,
checkpointing, restart-from-failure, loss spikes and what causes them, the fact
that a run is a months-long process with human operators watching curves. This
is the only place in the book where that is the subject.

Tier A code: a complete small causal LM training loop in torch on an inline
corpus, showing the loss falling below the unigram-entropy baseline; a token-
budget calculator; a checkpoint/restart demonstration that proves bit-exact
resumption, which is the property real runs depend on and rarely test.

### 81 — Pretraining Dataset Construction and Curation

The pipeline: source, filter, deduplicate, decontaminate, mix. Each stage with a
measurement.

{{cite:lee2022dedup}} is the anchor: exact and near-duplicate detection, MinHash
and suffix arrays, and the memorisation result. Contamination deserves its own
subsection because it undermines evaluation everywhere else in the book.

Mixture weights are a live design variable — how much code, how much
multilingual, how much of any domain — and the honest statement is that the
frontier weights are unpublished and the public evidence is thin.

Tier A code: a MinHash/LSH near-duplicate detector from scratch, measured for
precision and recall against a known-duplicate set; an n-gram contamination
checker run against a small "benchmark" to show overlap being found.

### 82 — Scaling Laws: Parameters, Data, and Compute

Already introduced in Part VII; developed properly here. The power-law form, the
compute-optimal allocation under a constraint, the Kaplan/Chinchilla
disagreement and why the second is right, and then the two corrections that
matter in practice: inference-aware allocation ({{cite:touvron2023llama}}) and
the missing data-quality term ({{cite:gunasekar2023}}).

Tier A code: fit a power law to synthetic loss-versus-compute data and recover
the exponents; solve the constrained allocation numerically and show the
Chinchilla ratio falling out; compute the lifetime-cost-optimal model size as a
function of expected inference volume, which is the calculation a team actually
has to do.

### 83 — Emergent Capabilities and What Emergence Means

Structured around the disagreement in §1 above. The chapter's job is to make the
reader able to evaluate an emergence claim, not to adjudicate.

Tier A code: **the key demonstration of the part.** Simulate a task with a
smoothly improving per-step accuracy, then score it with exact-match over k
steps and with a continuous metric, and show the same underlying model producing
a sharp curve under one and a smooth curve under the other. That is
{{cite:schaeffer2023}}'s argument as twenty lines of numpy, and it is far more
convincing than the prose.

### 84 — Instruction Tuning

{{cite:wei2022flan}}: the capability/interface distinction. Format, template
diversity, task mixture, and the finding that held-out *task type* generalisation
is what improves.

The chapter should be honest that instruction data quality dominates quantity,
and that this is one of the better-supported claims in the part.

### 85 — Alignment and RLHF

The full three-stage pipeline derived: SFT, the Bradley–Terry reward model with
its log-likelihood, and KL-regularised policy optimisation with PPO
({{cite:schulman2017ppo}}). Lineage from {{cite:christiano2017}} through
{{cite:stiennon2020}} to {{cite:ouyang2022}}.

The number to lead with: a 1.3B aligned model preferred to the 175B base.
Alignment bought more perceived quality than two orders of magnitude of scale.

{{cite:bai2022}} belongs here as the substitution of AI feedback against a
written constitution — and for the argument that an explicit document is a
better place for normative content than implicit annotator preferences.

Tier A code: Bradley–Terry reward model fitted from scratch on synthetic
comparisons, recovering a known latent reward; a KL-penalty demonstration
showing over-optimisation — true quality rising then falling as the policy is
pushed further against a learned proxy — which is {{cite:stiennon2020}}'s result
in miniature.

### 86 — Preference Optimization: DPO and Its Descendants

The derivation is the chapter. From the KL-regularised objective to the
closed-form optimal policy, invert to express reward in terms of policy, and
substitute into the Bradley–Terry likelihood — at which point the reward model
cancels and a classification loss on preference pairs remains.

Then the honest comparison: what DPO gives up, where online methods still win,
and the descendants adjusting its assumptions.

Tier A code: implement DPO's loss and verify numerically that its implicit
reward recovers the ordering of a known latent reward; contrast against the
explicit reward model from chapter 85 on the same synthetic data.

### 87 — Distillation and Model Specialization

{{cite:hinton2015}} derived: soft targets, temperature, and why the teacher's
wrong answers carry information a one-hot label cannot. Then the modern form —
distilling from a large model's generations into a small one — and the honest
statement that most small capable models are made this way.

Connects backwards to {{cite:sanh2019}} in Part VIII and forwards to
{{part:14}} and {{part:15}}.

Tier A code: train a teacher, distil into a student at several temperatures, and
show the student beating an identically-sized model trained on hard labels —
plus the temperature sweep that makes the mechanism visible.

## Cross-part bookkeeping

**Backwards** — anchors that exist: `tf-architectures` (ch068) for decoder-only,
`llm-*` do not exist yet (Part X follows this part), `nlp-bert` (ch076) for the
MLM/CLM comparison and the budget-equalisation lesson, `nlp-similarity` (ch078)
for contrastive training, `dl-losses` (ch052) for cross-entropy and KL,
`dl-optimizers` (ch054), `dl-lr-schedules` (ch055) for warmup and cosine decay,
`ml-metrics` (ch034), `mle-drift` (ch048) for monitoring.

**Forwards** — what this part sets up and must not spend: decoding strategies
and inference mechanics (Part X), parameter-efficient fine-tuning (Part XIV),
quantisation (Part XV), reasoning-specific RL (Part XVI), evaluation as a
discipline (Part XXV), and the fairness and governance treatment (Part XXVII).

**Do not write in this part:** anything about prompting technique (Part X),
agents (Part XVII), or RAG (Part XII).

# Part XVI — Reasoning Models: research notes

Research pass run 2026-08-29, before writing. Full tier: 21 sections per chapter,
4,200-word floor, seven chapters. Twelve new bibliography entries, each verified
against an arXiv abstract page on the date above. 259 entries total, none
unverified.

## What this part is, and what it is not

{{part:15}} made a model run. This part is about making it *think*, and it is the
part of the book where the gap between what is claimed and what is measured is
widest.

**The hazard is that this material is mostly folklore with a thin layer of
evidence on top.** "Chain-of-thought makes models reason" is repeated constantly
and is, as stated, false: {{cite:sprague2024tocot}}'s meta-analysis over 100
papers finds the gains concentrated in maths and symbolic tasks and close to
absent elsewhere. "Models can check their own work" is repeated constantly and
{{cite:huang2024selfcorrect}} finds performance *degrading* without external
feedback. "The chain of thought shows you how the model reasoned" is repeated
constantly and {{cite:turpin2023faithfulness}} shows it can be a plausible story
that never mentions the feature actually driving the answer.

> **The rule adopted for this part: every claim about reasoning must be
> accompanied by the measurement that would falsify it, and the negative results
> get equal space.** This is the part where a careful reader should end up less
> confident than they started, and more able to tell which of the two kinds of
> reasoning they are looking at.

## The organising idea

**Every technique here spends inference compute, and they differ in what they
spend it on and whether anything checks the result.**

```text
   WHAT IS SPENT             ON WHAT                    WHO CHECKS
   ───────────────────────   ────────────────────────   ─────────────────────────
   146 (the distinction)     nothing yet                nobody
   147 tokens, in one line   intermediate steps         nobody
   148 tokens, in parallel   many independent paths     a verifier, or a vote
   149 tokens, in sequence   revision of one path       the model itself (badly)
   150 annotation, upfront   per-STEP reward            humans, expensively
   151 tokens plus calls     externally checkable work  an interpreter or a tool
   152 (the measurement)     finding out if it worked   a benchmark, unreliably
```

The through-line, stated in {{ch:rsn-vs-generation}} and returned to in
{{ch:rsn-benchmarks}}: **generating a plausible chain and performing a reliable
derivation are different things, and almost every practical problem in this part
comes from treating them as the same.** A chain of thought is a SAMPLE from the
model's distribution over plausible-looking reasoning. That it usually correlates
with correctness is an empirical fact about training data, not a property of the
mechanism, and the places where the correlation breaks are exactly where the
failures are.

**The second through-line is that verification, not generation, is the binding
constraint.** {{cite:brown2024monkeys}} separates COVERAGE - does any sample solve
it - from SELECTION - can you tell which one did. Coverage scales log-linearly and
impressively: SWE-bench Lite from 15.9% at one sample to 56% at 250. Selection
plateaus. **Everything that works well in this part works because something
outside the model can check the answer**, and everything that works poorly works
poorly for the same reason.

## The genuinely live questions

### 1. Does chain-of-thought help, and where?

{{cite:wei2022cot}} made it famous. {{cite:sprague2024tocot}}'s meta-analysis
bounded it: **maths and symbolic reasoning, and very little else.** On MMLU
without symbolic operations, close to nothing.

**And where it helps, it is doing symbolic execution that a symbolic solver does
better** - which is the finding that connects this part to
{{ch:rsn-tool-assisted}} and makes tool use the recommended answer rather than
longer chains.

{{ch:rsn-cot}} must carry this. The chapter should be able to say, per task type,
whether to spend the tokens.

### 2. Is the chain of thought the reason for the answer?

{{cite:turpin2023faithfulness}} says not necessarily. Inject a bias - reorder
options so the correct one is always (A) - and accuracy drops by up to 36% while
the explanations confidently justify the biased answers **without ever mentioning
the bias.**

**That is the most important negative result in the part** and it has two
consequences the chapter must state. Reading a chain of thought tells you what a
plausible justification looks like, not what happened. And any safety or
interpretability argument resting on monitoring the trace has to survive this
first.

### 3. Can models correct themselves?

{{cite:huang2024selfcorrect}} distinguishes intrinsic self-correction from
correction with external feedback, and finds the intrinsic kind **does not work and
can make things worse.**

The reason most reported gains disappear: they use the answer key to decide when
to revise. **That oracle is not available at inference time**, and removing it
removes the effect.

{{ch:rsn-self-consistency}} should present reflection honestly: it works when
something external grades it, and the architecture question is what that something
is.

### 4. What does test-time compute actually buy?

{{cite:snell2024testtime}} makes it a budget-allocation problem: adaptive
allocation is worth more than 4x against uniform best-of-N, and in FLOPs-matched
comparisons can beat a 14x larger model **on problems where the smaller model
already partly succeeds** - a qualification that matters enormously and is usually
dropped.

{{cite:brown2024monkeys}} explains the boundary: coverage scales, selection does
not. **So the honest summary is that test-time compute converts a verification
capability into an accuracy gain, and buys little where verification is
unavailable.**

{{ch:rsn-test-time-compute}} should measure the coverage/selection split directly
rather than quoting it.

### 5. Is process supervision worth its cost?

{{cite:lightman2023verify}} says yes on quality - process supervision
substantially beats outcome supervision, reaching 78% on a MATH subset - and the
cost is 800,000 human step-level labels.

**The live question is whether that cost is necessary or whether verifiable
domains can supply the signal automatically**, which is what
{{cite:deepseek2025r1}}'s verifiable-reward RL suggests and what
{{ch:rsn-tool-assisted}} is really about.

{{ch:rsn-supervision}} should make the annotation economics explicit, because it
is what decides whether a team can use the technique.

### 6. Do reasoning benchmarks measure reasoning?

{{cite:mirzadeh2024gsmsymbolic}} regenerates the same problems with different
names and numbers and finds performance declining; adding **one irrelevant
sentence** costs **up to 65%.**

**A benchmark score is a measurement of performance on those exact surface forms.**
{{ch:rsn-benchmarks}} should treat the variance under semantically-null
perturbation as the headline number and the mean as a footnote, which is the
reverse of standard practice.

## Per-chapter findings

### 146 — Reasoning versus Generation

The framing chapter, and the most important. Content: what would distinguish
reasoning from fluent generation, operationally? Candidate criteria - invariance
to irrelevant detail, compositional generalisation, calibrated abstention - and
what each would predict.

**Listing:** construct a task with a known structure, and measure whether a
system that gets it right is invariant to semantically-null perturbations. This
is {{cite:mirzadeh2024gsmsymbolic}}'s design applied to a system we can inspect
completely, so the mechanism is visible rather than inferred.

### 147 — Chain-of-Thought and Its Mechanics

Per live questions 1 and 2. Content: what intermediate tokens actually do -
provide working memory that the forward pass otherwise lacks - and where that
helps.

**Listing:** a task that provably needs serial computation, solved with and
without intermediate tokens, showing the depth/serial-steps trade. Then the
faithfulness experiment: a biased shortcut the model can exploit, and whether the
stated reasoning mentions it.

### 148 — Test-Time Compute and Search

Per live question 4. **Listing:** the coverage/selection decomposition. Measure
pass@k rising log-linearly while majority-vote accuracy plateaus, and show that
the gap IS the verifier's quality. Then the compute-optimal allocation result: a
fixed budget spent uniformly against spent by estimated difficulty.

### 149 — Self-Consistency, Reflection, and Critic Models

Per live question 3. **Listing:** self-consistency working (it does), then
reflection measured with and without an oracle, reproducing the direction of
{{cite:huang2024selfcorrect}}. The chapter's job is to explain why one works and
the other does not, and the answer is that voting aggregates independent samples
while reflection conditions on a possibly-wrong first attempt.

### 150 — Process versus Outcome Supervision

Per live question 5. **Listing:** simulate both supervision signals on a task
where a correct answer can be reached by faulty reasoning, and measure how often
outcome supervision rewards the faulty path. Then price the annotation.

### 151 — Tool-Assisted and Verified Reasoning

The chapter where things work. Content: an external checker changes the problem
from generation to search, and {{cite:brown2024monkeys}}'s coverage becomes
usable. **Listing:** the same task with and without an executable check, measuring
how the accuracy/compute curve changes shape.

### 152 — Reasoning Benchmarks and the Reliability Gap

Per live question 6. **Listing:** build a template-generated benchmark, measure
the variance across surface forms, and show that a single score is an estimate of
a distribution whose spread nobody reports. Then contamination: what a leaked
benchmark looks like from the inside.

## Cross-part bookkeeping

- {{part:9}} owns RLHF and DPO; {{cite:deepseek2025r1}}'s verifiable-reward RL is
  a training method and should be *described* here and not re-derived.
- {{part:14}} owns fine-tuning; {{cite:muennighoff2025s1}}'s 1,000-example recipe
  is an instance of {{ch:ft-datasets}}'s argument and should reference it.
- {{part:15}} owns inference cost; every "spend more compute" claim here must be
  priced against {{ch:q-throughput-latency}}'s frontier, because thinking tokens
  are decode tokens and cost what decode costs.
- {{part:17}} owns agents; ReAct is agent material and appears here only as the
  boundary case.
- {{part:25}} owns evaluation infrastructure; {{ch:rsn-benchmarks}} contributes
  the perturbation-variance argument to it.
- Terminology collision check: `reasoning` (this part's subject versus
  {{part:11}}'s "reasoning over retrieved documents"), `verifier` (a model here, a
  program in {{ch:rsn-tool-assisted}}), `process` and `outcome` supervision versus
  {{part:9}}'s reward modelling vocabulary, `search` (over reasoning states here,
  over documents in {{part:10}}).

## The pattern carried from {{part:15}}

There, the number people quote was repeatedly the less important half of a
specification. Here the analogous hazard is that **the demonstration people quote
is repeatedly the easy case**: chain-of-thought on maths, self-correction with an
oracle, test-time compute on verifiable tasks, benchmark scores on memorised
surface forms.

**The listings are written and run before the prose, and the prose reports what
they found.** Two places are flagged in advance as likely to move: whether
self-consistency's gains survive when the samples are not independent, and whether
the coverage/selection gap is as clean outside code and maths as
{{cite:brown2024monkeys}} suggests.

---
id: part-09-assessment
status: final
---

## How to use this

Four sections. The knowledge check takes about ninety minutes and tells you what
to re-read. The assignment builds the alignment pipeline end to end on a model
small enough to train on one machine — it is the piece of work this part was
written for, and it is deliberately a *measurement* project as much as an
implementation one. The challenge is open-ended. The interview section is what
to rehearse.

No answers are provided. Every question is answerable from the chapters.

## Knowledge check

**The paradigm**

1. State {{cite:bommasani2021}}'s definition and say which of its three clauses
   is a technical claim and which are economic.
2. Explain in terms of {{eq:pretraining-decomposition}} why a proxy objective
   produces general capability, and identify the step where that argument stops
   applying to any actual model.
3. Derive {{eq:adaptation-information-ratio}}'s consequence: why does
   fine-tuning teach format reliably and facts poorly?
4. State the homogenisation argument formally. Give one defect that would
   propagate to every model adapted from a base, and one that would not.
5. Instruction tuning is about $10^{-5}$ of pretraining compute and produces
   most of what a user perceives as quality. What does that ratio establish?

**Pretraining**

6. Give the three reasons causal language modelling beat masked language
   modelling, and say which one the {{cite:clark2020electra}} natural experiment
   identifies as decisive.
7. What two numbers diagnose a language-model training run in its first ten
   minutes? What does each one catch?
8. A run's loss falls convincingly and plateaus at the unigram entropy. Give
   three possible causes.
9. Explain why padding a pretraining batch is a quadratic waste and what
   packing recovers.
10. What must a checkpoint contain beyond weights and optimiser state, and what
    is the symptom when it does not?

**Data**

11. Prove {{eq:minhash-property}}: why does the probability that two documents'
    minimum hashes agree equal their Jaccard similarity?
12. Choose $b$ and $r$ for a 128-element signature giving a threshold near 0.6,
    and verify with {{eq:lsh-threshold}}.
13. Explain {{eq:duplication-reweighting}} and why deduplication improves
    quality rather than only saving compute.
14. Why is every published contamination rate a lower bound? Give two kinds of
    contamination that leave no $n$-gram trace.

**Scaling**

15. Derive {{eq:n-optimal}} from {{eq:chinchilla-form}} under $C = 6ND$.
16. What did {{cite:kaplan2020scaling}} hold fixed that it should not have, and
    which direction did the resulting bias run?
17. Is $D/N = 20$ a law? Answer using the exponents rather than by assertion.
18. Write down {{eq:lifetime-cost-2}} and explain why the optimum moves toward
    smaller models as serving volume rises. What is the floor it approaches?

**Emergence**

19. State the two clauses of the emergence definition and say which is
    falsifiable.
20. Using {{eq:transition-width}}, compute the transition width for $k = 25$.
    With five log-spaced model sizes, what is the chance of observing it?
21. A rescoring with a continuous metric flattens a sharp curve. What does that
    establish, and what does it not?
22. Name the explanation for a sharp curve that survives every rescoring, and
    say why it correlates with scale.

**Alignment**

23. Why can a demonstration not express a preference?
24. Show that adding $c(x)$ to both rewards leaves {{eq:bradley-terry}}
    unchanged, and state what that forbids you from reporting.
25. Derive {{eq:rlhf-optimal-policy}} from {{eq:rlhf-objective}}.
26. Explain {{eq:over-optimisation}}. Why does the KL penalty help, given that
    it does not improve the reward model?
27. Invert {{eq:rlhf-optimal-policy}} and show that $Z(x)$ cancels in the
    Bradley–Terry likelihood. Why is that cancellation the whole method?
28. What is likelihood displacement, and what property of preference pairs makes
    it likely?

**Distillation**

29. Derive the two factors of $1/T$ in {{eq:distillation-gradient}} and explain
    what the $T^2$ correction is for.
30. Why does a soft target carry more supervision per example than a hard
    label? Quantify it.
31. Name three things a student inherits from its teacher that you would rather
    it did not, and say why capacity loss takes rare knowledge first.

## Practical assignment

**Build the alignment pipeline end to end on a small model, and measure every
stage.** The point is not to produce a good model — you cannot, at this scale.
It is to produce a pipeline whose every claim you have verified yourself.

### Part A — pretrain

1. **A causal language model** on a corpus you assemble, small enough to train
   in under an hour. Character or small-BPE vocabulary is fine.
2. **The two diagnostics**: assert the initial loss is within tolerance of
   $\log|V|$, and that the final loss is below the unigram entropy by a stated
   margin.
3. **Bit-exact checkpoint resumption**, including the sampler position. Write
   the test that proves it: resume from step $k$ and confirm the loss at step
   $k{+}1$ matches the uninterrupted run exactly.
4. **Packing rather than padding**, with the efficiency measured against a
   padded baseline on your corpus's actual length distribution.

**Acceptance criteria.** All three assertions pass. The resumption test is the
one that matters and is the one almost nobody writes.

### Part B — the corpus

5. **MinHash + LSH deduplication** over your corpus, with precision and recall
   measured against a labelled duplicate set you construct.
6. **The $b$/$r$ operating curve**: sweep the banding parameters and plot the
   observed S-curve against {{eq:lsh-probability}}'s prediction.
7. **A contamination audit** against whatever you will evaluate on, reported as
   a count per benchmark and with a stated $n$.
8. **Per-stage yields**, logged. Report what fraction of your raw corpus
   survives to training.

### Part C — align

9. **Instruction tuning** with correct loss masking. Assert the mask covers
   every assistant turn and that the first response token is supervised.
10. **A reward model** fitted from synthetic pairwise comparisons, with its
    ordering agreement against the latent reward reported.
11. **DPO**, with the $\log 2$ initial-loss check as an assertion.
12. **The displacement measurement**: track $\log\pi_\theta(y_w)$ throughout
    training, not only the loss, and report whether it fell.

### Part D — distil and decide

13. **Distil** your aligned model into a smaller student, sweeping temperature
    and reporting the curve.
14. **The slice evaluation**: construct a rare subpopulation in your evaluation
    set and report the student's regression on it against the aggregate.
15. **The economics**: compute your break-even volume, and state honestly
    whether distillation would have been worth it at your scale.

### Part E — the report

16. Two pages. For every number you report, state how it was measured and what
    would change it. For every claim you carried over from a chapter rather than
    measuring, say so explicitly.

**The last clause is the point.** This part is nine chapters of results from
sources that mostly could not be checked. Producing a document that distinguishes
what you verified from what you inherited is the transferable skill.

## Advanced challenge

Pick one. Each is a real experiment whose answer is not in the chapters.

**Reproduce the Kaplan bias.** Run a small scaling sweep twice — once with a
shared learning-rate schedule across all runs, once with each schedule correctly
terminated — and fit the exponents both ways. Show the allocation rule flipping.
This is the cheapest available demonstration that a nuisance variable can carry
a field's conclusion for two years.

**Separate the three explanations for emergence.** Construct a task where you
control which mechanism operates — metric composition, sparse sampling, or
contamination — and show that a continuous rescoring distinguishes the first two
and fails on the third.

**Measure DPO against RLHF at matched KL.** Most published comparisons do not
control for divergence from the reference, so they may compare points on one
curve. Implement both, hold KL fixed, and report what remains of the difference.

**Find the data-quality term.** Train small models on corpora differing only in
curation, fit {{eq:chinchilla-form}} to each, and determine whether an
effective-token definition makes the constants transfer across mixtures. A
negative result is publishable.

**Test whether upweighting rare examples recovers distillation's tail loss.**
Section 7 of {{ch:fm-distillation}} argues rare knowledge goes first because the
objective is an expectation. Reweight and measure what it costs on the common
cases.

## Interview preparation

**The seven derivations to do without notes.**

1. Why a proxy objective produces general capability —
   {{eq:pretraining-decomposition}}.
2. The compute-optimal allocation from $C = 6ND$ — {{eq:n-optimal}}.
3. MinHash's defining property — {{eq:minhash-property}}.
4. Exact match as a step function on a smooth quantity —
   {{eq:exact-match-composition}} and {{eq:transition-width}}.
5. The KL-regularised optimum — {{eq:rlhf-optimal-policy}}.
6. DPO from that optimum, including why $Z(x)$ cancels.
7. The $T^2$ factor in distillation — {{eq:distillation-gradient}}.

**The eight numbers.**

- **$\approx 10^{-8}$** — fine-tuning's information relative to pretraining.
- **$\approx 10^{-5}$** — instruction tuning's compute relative to pretraining.
- **$\log|V|$ and $H(X)$** — the two diagnostics for any LM training run.
- **$D/N \approx 20$** — Chinchilla-optimal, at Chinchilla's scale only; it
  drifts as $C^{0.11}$.
- **6.7x** — causal LM's supervision advantage over MLM at equal compute.
- **1.3B beat 175B** — {{cite:ouyang2022}}'s aligned-versus-base result.
- **$\log 2 = 0.693$** — DPO's loss at initialisation, always.
- **4 models against 2** — RLHF's stage 3 versus DPO, and the reason the open
  ecosystem chose DPO.

**The six things people get wrong, and the correction.**

- *"Fine-tune to teach the model facts."* It re-weights a mixture; it does not
  add modes. Use retrieval.
- *"$D/N = 20$ is the law."* It is one fitted setup's value at one scale, and it
  is the wrong target for anything served at volume.
- *"Emergence has been debunked."* The evidence was shown to be weaker than
  claimed. That is a different statement.
- *"RLHF works because of the RL."* One classification loss matches it. The
  value is in the preference data.
- *"DPO has no KL control."* It has, through $\beta$ and the reference; it is
  implicit and therefore easier to loosen by accident.
- *"The student can be better than the teacher."* It is trained on the
  teacher's outputs. The teacher is the ceiling.

**The debugging order for a pipeline that produces a bad model.**

1. **Initial losses.** $\log|V|$ for pretraining, $\log 2$ for DPO. Both are
   free and both catch whole classes of setup bug.
2. **The unigram floor.** Did pretraining learn context, or only frequencies?
3. **Loss masks.** Instruction tuning's mask covering every assistant turn.
4. **Template match** between training and serving — one shared function, one
   golden-string assertion.
5. **Contamination**, on the pretraining corpus *and* on any generated data.
6. **Slice evaluation**, because aggregates hide the regressions that matter.
7. **Absolute log-probabilities**, not only the loss, wherever a preference
   objective is involved.

Steps 1, 2 and 7 are specific to this part, and each catches a failure that
produces a plausible-looking training curve.

**The one disposition to carry forward.** This part's most useful lesson is not
about foundation models. It is that **the field's most confident, most widely
adopted results have repeatedly turned out to rest on an uncontrolled nuisance
variable** — a learning-rate schedule, a metric choice, a set of
hyperparameters — and that each time it was a replication study rather than a
new method that found it. The question to take into {{part:10}} and everything
after is not "is this result impressive" but **"what was held fixed?"**

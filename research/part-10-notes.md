# Part X — Large Language Models: research notes

Research pass run 2026-08-26, before writing. Full tier: 21 sections per
chapter, 4,200-word floor, eleven chapters — the longest part so far. Eight new
bibliography entries, each verified against an arXiv abstract page or
proceedings listing on the date above. 140 entries total, none unverified.

## What this part is, and what it is not

{{part:9}} built the model. This part **follows a single prompt through it**,
from the API boundary to the returned token, and then takes seriously everything
that goes wrong on the way.

The hazard is different from Part IX's. There the problem was unrefereed
sources; here it is that **almost everything in this part is folklore with a
paper attached somewhere behind it.** Prompting in particular is an area where
practice ran far ahead of evidence, where the same technique is described a
dozen ways, and where a great deal of what circulates as advice has never been
tested against a control.

The rule for this part: **for every prompting technique, say what the controlled
evidence is, and where there is none, say that.** {{cite:min2022}} is the model
for what happens when someone actually runs the control — it found that random
labels in few-shot demonstrations barely hurt, which is not what anyone
expected.

## The organising idea

Part IX's chain was *capability, then corrections*. This part's is different:

**Everything a user experiences as a property of "the model" is actually a
property of the model plus a decoding strategy plus a serving configuration.**

Temperature is not in the weights. Top-p is not in the weights. The context
window's usable length is not the number in the datasheet. Structured output is
a masking layer, not a capability. Whether a tool gets called is a template
decision. Nearly every complaint about LLM behaviour resolves to one of these
rather than to the model itself, and the part is organised to make that
attribution possible.

The through-line to state in {{ch:llm-anatomy}} and return to at the end:
**the model is a function from token sequences to a probability distribution.
Everything else is what you do with that distribution.**

## What changes at this tier for this material

The mathematics is thinner than Parts VI–VIII and the *mechanism* is thicker.
Sections have to be earned by tracing actual computation:

- **§6 Mathematical Foundation.** The genuine content: the softmax with
  temperature and its limits, entropy of the truncated distributions, why
  beam search optimises the wrong objective, the arithmetic of KV-cache growth
  and prefill/decode asymmetry (which {{ch:tf-complexity}} set up and this part
  spends), the finite-state-machine construction for constrained decoding, and
  a calibration treatment for hallucination.
- **§7 Internal Mechanics.** Unusually strong throughout — this is the part
  where "what actually happens" is the subject rather than a section.
- **§12 Failure Modes.** The richest in the book so far: prompt sensitivity,
  degenerate repetition, context overflow, position effects, tool-call
  malformation, confident fabrication.
- **§19 Research Questions.** Many, and most are genuinely open because the
  controls have not been run.

**Padding risk is highest in {{ch:llm-prompting}}**, which could easily become a
list of tricks. Rule for that chapter: a technique earns a paragraph only if
there is a controlled result or a mechanism that explains it. Chain-of-thought
earns one ({{cite:wei2022cot}}, {{cite:kojima2022}}). "Tell the model it is an
expert" does not, and saying so is more useful than repeating it.

## The genuinely live questions

### 1. What is in-context learning?

Not settled, and the obvious answer is wrong.

{{cite:brown2020}} presented few-shot prompting as the model learning from the
examples. {{cite:min2022}} replaced the labels with random ones and performance
barely moved — so whatever is happening, it is not learning the input–output
mapping from the demonstrations. What the demonstrations supply is the label
space, the input distribution, and the format.

**What to write:** the finding, its scope (it is strongest for classification
and weaker for generation tasks), and the consequence for prompt design — spend
effort on format and label coverage rather than on getting every exemplar right.
What not to claim: that demonstrations do not matter, or that the mechanism is
understood.

### 2. Does chain-of-thought reflect the model's reasoning?

Open, and consequential for interpretability claims.

{{cite:wei2022cot}} and {{cite:kojima2022}} establish that eliciting
intermediate steps improves answers. They do not establish that the steps are
the process producing the answer. A model can produce a correct answer with
incorrect stated reasoning, and the reverse — which means the trace is not
straightforwardly an explanation.

This matters here because chain-of-thought is routinely presented to users as
transparency. The chapter should separate **the capability claim** (it improves
accuracy, well supported) from **the faithfulness claim** (the trace describes
the computation, not established).

### 3. Is long context usable context?

Settled enough to state and routinely ignored in practice.

{{cite:liu2023lost}} measured a U-shaped curve: information at the beginning and
end of a long context is retrieved well and information in the middle is not.
A 128k window does not mean 128k of usable evidence, and the position of your
retrieved passages is a design variable rather than an implementation detail.

This connects directly to {{part:12}} and should be written knowing that:
{{ch:rag-failures}} inherits it.

### 4. Where does hallucination come from?

Multiple causes, frequently conflated, and the conflation is why mitigations
disappoint.

{{cite:ji2023survey}}'s intrinsic/extrinsic split is the operational one.
Intrinsic hallucination contradicts a provided source and is detectable by
checking against it. Extrinsic hallucination is unverifiable from the source and
requires external knowledge to catch. **They have different detection methods
and different fixes**, and a team measuring one while mitigating the other is
common.

The deeper cause is worth stating plainly: the training objective rewards
plausible continuation, and there is no term anywhere in
{{eq:clm-loss}} for truthfulness. A model that says "I do not know" where the
corpus says something confident is penalised. Hallucination is not a bug that
was introduced; it is the objective working.

### 5. Do structured-output guarantees cost quality?

Genuinely contested and cheaply testable.

{{cite:willard2023}} makes valid JSON a structural guarantee by masking invalid
tokens. The obvious concern is that constraining the distribution degrades
content quality — the model wanted a token it was not allowed. The published
evidence is mixed and mostly not controlled for the fact that unconstrained
baselines *fail to parse* some fraction of the time, which is not a fair
comparison.

## Per-chapter findings

### 88 — Anatomy of an LLM: From Tokens to Logits

The trace. Tokens to embeddings to blocks to final norm to unembedding to
logits, with shapes at every step. Nothing new architecturally — it is
{{part:7}} assembled — and its value is as the reference the rest of the part
points back to.

Tier A code: a complete forward pass in numpy with every intermediate shape
printed and asserted, on a model small enough to inspect by hand.

### 89 — Next-Token Prediction and Cross-Entropy Loss

{{eq:clm-loss}} again, from the *inference* side: what the logits mean, why
cross-entropy is the right loss, and the calibration question — does a
probability of 0.7 mean the token is right 70% of the time? Sets up
{{ch:llm-hallucination}}, because a well-calibrated model that is confidently
wrong is a different problem from a badly calibrated one.

Tier A code: measure calibration of a small trained model with a reliability
diagram computed from scratch.

### 90 — Decoding

The chapter with the most immediately usable content in the part. Greedy,
temperature, top-k ({{cite:fan2018}}), top-p ({{cite:holtzman2020}}), and beam
search — and the argument that beam search optimises sequence likelihood, which
{{cite:holtzman2020}} shows is not what anyone wants for open-ended text.

The degeneration result is the anchor: high-likelihood text is repetitive, and
human text sits in a region of *moderate* surprise. That is a genuinely
counter-intuitive empirical fact and it should be demonstrated rather than
asserted.

Tier A code: implement every sampler from scratch; measure repetition rate and
entropy against temperature and p; reproduce the degeneration curve.

### 91 — Context Windows, KV Cache, and Inference Mechanics

{{ch:tf-masking-kv}} and {{ch:tf-complexity}} did the derivation; this chapter
spends it on serving. Prefill versus decode, why output tokens cost more than
input tokens, cache memory growth, and what happens at the context limit.

Tier A code: the KV-cache size calculator and the prefill/decode cost model,
producing the price asymmetry from first principles.

### 92 — What Actually Happens When You Send a Prompt

End-to-end, including the parts that are not the model: template application,
tokenization, batching, prefill, the sampling loop, stop conditions,
detokenization, streaming. This is the chapter that makes latency legible —
time-to-first-token is prefill, inter-token latency is decode, and they have
different causes and different fixes.

### 93 — Prompting and System Prompts

The chapter most at risk of becoming folklore. Anchor it on
{{cite:min2022}}'s control and {{cite:wei2022cot}}/{{cite:kojima2022}}'s
mechanism, be explicit about prompt sensitivity as a measurable property, and
treat system prompts as what they are — a training-time convention that models
were taught to condition on, not a privileged channel.

Tier A code: prompt-sensitivity measurement — the same task under many
phrasings, reporting the spread. That number is what makes the chapter honest.

### 94 — Structured Output and Constrained Decoding

{{cite:willard2023}}'s FSM construction, derived. This is the engineering answer
to {{ch:fm-emergence}}'s all-or-nothing requirement: rather than hoping the
model emits valid JSON, mask the tokens that would make it invalid.

Tier A code: a working constrained decoder over a small grammar, demonstrating
that invalid outputs are unreachable rather than unlikely — the same property
the CRF gave in {{ch:nlp-extraction}}, twenty years apart.

### 95 — Function Calling and Tool Use

Mechanically, structured output plus a dispatch loop. {{cite:schick2023}} for
the training-time version. The chapter's real content is the failure surface:
malformed arguments, hallucinated tool names, wrong tool selection, and the fact
that the model cannot know whether a call succeeded unless told.

### 96 — Hallucination

Structured on {{cite:ji2023survey}}'s taxonomy, with the objective argument in
§1 above as the mechanism. Mitigations ranked by what they actually address:
retrieval for extrinsic, constrained decoding for format, citation checking for
intrinsic, and calibration for confidence.

### 97 — Long-Context Behavior and Its Limits

{{cite:liu2023lost}}'s U-curve, reproduced. Supported length against usable
length, position effects, and the cost accounting from
{{ch:tf-complexity}} that makes long context expensive as well as unreliable.

Tier A code: reproduce the U-shaped retrieval curve on a synthetic
needle-in-a-haystack task.

### 98 — Model Routing and Model Selection

The economic chapter. Cascades, difficulty estimation, and the same
cheap-stage-then-expensive-stage pattern as {{ch:nlp-similarity}}'s
retrieve-then-rerank and {{ch:nlp-extraction}}'s encoder/LLM cascade. Third
appearance of one architecture, which is worth naming explicitly.

## Cross-part bookkeeping

**Backwards** — anchors that exist: `tf-architectures` (ch068),
`tf-masking-kv` (ch069), `tf-complexity` (ch070), `nlp-subword` (ch073),
`fm-instruction-tuning` (ch084), `fm-rlhf` (ch085), `nlp-extraction` (ch077)
for constrained decoding's ancestor, `ml-metrics` (ch034) for calibration.

**Forwards** — what this part sets up and must not spend: retrieval
({{part:12}}), agents and tool loops beyond a single call ({{part:17}}),
reasoning-specific training ({{part:16}}), serving infrastructure
({{part:23}}), evaluation as a discipline ({{part:25}}), and prompt injection
({{part:26}} — this part covers what a prompt *is*, not how it is attacked).

**Do not write in this part:** RAG, agent loops, or fine-tuning. Every one of
them is a later part and each is easy to drift into.

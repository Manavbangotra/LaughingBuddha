# Part XV — Quantization and Local AI: research notes

Research pass run 2026-08-29, before writing. Full tier: 21 sections per chapter,
4,200-word floor, eight chapters. Sixteen new bibliography entries, each verified
against an arXiv abstract page on the date above. 247 entries total, none
unverified.

## What this part is, and what it is not

{{part:14}} changed the weights. This part changes how they are *stored*, and it
is the most hardware-adjacent material in the book.

**The hazard here is that quantization looks like a menu of methods and is
actually a small number of physical facts.** GPTQ against AWQ against GGUF
Q4_K_M against EXL2 is a comparison that will be stale within two years, and a
chapter organised around it teaches nothing durable. The facts underneath — that
a numeric format is a budget split between range and resolution, that decode is
memory-bound and prefill is compute-bound, that the KV cache is the term growing
with traffic rather than with the model — will still be true.

> **The rule adopted for this part: teach the arithmetic and the failure mode;
> name the formats only as instances.** Every chapter must leave the reader able
> to *compute* whether something fits and *predict* what will break. If a chapter
> would be obsoleted by a new file format, it is written wrong.

## The organising idea

**Every technique here answers one question: which number are you shortening, and
what does the shortening destroy?**

```text
   WHICH NUMBER              WHAT SHORTENING BUYS       WHAT IT DESTROYS
   ───────────────────────   ────────────────────────   ─────────────────────────
   138 the format itself     range or resolution        whichever you did not pick
   139 (the theory)          the error budget           nothing, until the outliers
   140 weights, INT8/INT4    memory, and decode speed   the outlier channels
   141 weights, on CPU       a model that fits at all   arithmetic throughput
   142 activations, KV       batch size, so throughput  keys and values, unequally
   143 (the arithmetic)      an answer before you buy   comfortable assumptions
   144 (the runtimes)        someone else's choices     visibility into them
   145 (the regime)          latency or throughput      the other one
```

The through-line, stated in {{ch:q-formats}} and returned to in
{{ch:q-throughput-latency}}: **precision is a bandwidth decision, not a storage
decision.** Weights are read once per token during decode and never reused, so
bits-per-weight maps almost linearly to tokens-per-second on a memory-bound
device. That single fact explains why 4-bit local inference took over, why
quantization does *not* speed up large-batch serving, and why the same technique
is essential on a laptop and irrelevant on a saturated H100.

## The genuinely live questions

### 1. Is 4-bit still the right default?

{{cite:dettmers2023case4bit}} ran 35,000 experiments and found 4-bit almost
universally optimal for total model bits against zero-shot accuracy. That result
has held up, and it is the reason the local-inference ecosystem converged where it
did.

{{cite:egiazarian2024aqlm}} and {{cite:tseng2024quipsharp}} both argue below it,
reaching 2-3 bits with changed representations rather than better rounding.

**The honest position for {{ch:q-int8-int4}}: 4-bit is the default because
DECODE SPEED, not accuracy, is what binds below it.** Sub-3-bit formats need
codebook lookups or rotations on the critical path, which costs exactly the
bandwidth advantage that motivated going lower. The chapter should measure that
rather than assert it.

### 2. Does quantization robustness depend on the checkpoint?

This is the most consequential recent result in the part and it is barely
absorbed. {{cite:kumar2024precisionscaling}} finds that **post-training
quantization damage INCREASES with the amount of pretraining data**, to the point
where additional pretraining becomes actively harmful if the model will be
quantized.

**That inverts the intuition** that a better-trained model is a more robust one,
and it invalidates a sentence people say constantly: "we validated this
quantization recipe on this architecture." The recipe was validated on a
*checkpoint*. A longer-trained successor of the same architecture may not survive
it.

{{ch:q-theory}} should carry this, and it should be stated as a measurement
obligation rather than a curiosity.

### 3. Why do outliers exist, and does the answer matter?

{{cite:dettmers2022int8}} named the phenomenon: emergent systematic features that
appear above a scale threshold and break per-tensor INT8. Four different responses
followed, and they are genuinely different ideas rather than variations:

| Response | Mechanism | What it assumes |
|---|---|---|
| {{cite:dettmers2022int8}} | keep outlier dimensions in 16-bit | you can afford a second precision path |
| {{cite:xiao2023smoothquant}} | migrate difficulty into the weights | the transformation is exactly equivalent |
| {{cite:lin2023awq}} | protect ~1% of channels, chosen by activations | the calibration set is representative |
| {{cite:tseng2024quipsharp}} | rotate so no coordinate dominates | the rotation is cheap enough at inference |

**{{ch:q-int8-int4}} should present these as four answers to one question**, and
should note what the third assumption implies: AWQ's channel selection depends on
the calibration data, so the calibration set is a hyperparameter that is almost
never reported.

### 4. Is weight-only quantization a speed technique?

**It depends entirely on the regime, and this is the part's most useful practical
distinction.** At batch 1, decode reads every weight once per token and does
almost no arithmetic per byte read, so halving the bits nearly halves the time.
At large batch the same weight read serves many sequences, arithmetic intensity
rises, and the operation becomes compute-bound — where a 4-bit weight has to be
dequantised before it can be multiplied, so quantization can be *slower*.

{{cite:pope2022inference}} is the reference for this framing.
{{ch:q-throughput-latency}} should compute the crossover rather than describe it.

### 5. Where does inference memory actually go?

Three terms: weights, KV cache, activations. **The KV cache is the one that grows
with traffic rather than with the model**, and it is what actually limits batch
size in production.

{{cite:kwon2023pagedattention}} showed the striking part: much of the waste was
not in the tensor but in how it was *managed* — fragmentation and duplication —
and fixing the allocator was worth 2-4x throughput at equal latency. **More than
kernel optimisation delivered.**

{{cite:liu2024kivi}} adds the asymmetry: keys want per-channel quantization and
values want per-token, because their outlier structure differs. "The KV cache" is
two tensors with different statistics, and treating it as one is a measurable
mistake.

## Per-chapter findings

### 138 — Numerical Formats: FP32, FP16, BF16, and FP8

Content: a format is a split of a fixed bit budget between exponent (range) and
mantissa (resolution), and every format in use is a different point on that split.
{{cite:micikevicius2018mixed}} for the mixed-precision pattern and why FP16 needed
loss scaling; {{cite:micikevicius2022fp8}} for E4M3/E5M2 and the explicit division
of labour between forward and backward.

**Listing:** implement the rounding for each format in numpy and measure, on a
real weight-shaped distribution, the representable range, the relative error, and
the fraction of values that underflow. Then show BF16's wider exponent removing
the need for loss scaling that FP16 requires — the clearest demonstration that the
split is the whole design.

### 139 — Why Quantization Works: Theory and Error Analysis

The chapter that has to earn the rest of the part. Content: quantization error is
small relative to what a trained network tolerates, and the reason is worth
proving rather than asserting.

**Listing:** measure whether per-layer quantization errors compound or cancel
through depth. The hypothesis to test is that independent rounding errors partly
cancel, so a deep network tolerates far more per-layer error than a naive
error-propagation bound suggests — and that outliers break the independence, which
is why they matter so much more than their frequency implies.

**Second listing:** the group/block size sweep, per live question 1. Block size is
the single most consequential practical parameter and it is usually left at a
default.

Per live question 2, {{cite:kumar2024precisionscaling}} belongs here.

### 140 — INT8, INT4, GPTQ, and AWQ

Per live question 3. Content: the outlier phenomenon and the four responses to it.

**Listing:** show per-tensor INT8 degrading as outlier magnitude grows, then
per-channel scaling and a SmoothQuant-style migration fixing it — with the point
being that the fix is exactly equivalent as a function and changes only what the
quantizer sees.

**Second listing:** GPTQ's error compensation against round-to-nearest at the same
bit-width, measured. The mechanism — propagate each weight's rounding error into
the weights not yet quantized — is simple enough to implement honestly.

### 141 — GGUF, llama.cpp, and Weight-Only Quantization

Content: why weight-only quantization dominates local inference, which is a
bandwidth argument rather than a memory argument.

**Listing:** the decode roofline. Show that time per token is approximately model
bytes divided by memory bandwidth, so bits-per-weight maps nearly linearly to
tokens-per-second — and then show where that linearity breaks.

**Second listing:** mixed bit allocation in the style of k-quants. Not all tensors
deserve the same width, and allocating by sensitivity rather than uniformly is
measurable.

### 142 — Activation and KV-Cache Quantization

Per live question 5. Content: activations are harder than weights because they are
data-dependent and have outliers; the KV cache is two tensors with different
statistics.

**Listing:** reproduce {{cite:liu2024kivi}}'s asymmetry — measure the outlier
structure of keys and values separately and show that the right quantization axis
differs between them.

**Second listing:** KV-cache memory against context and batch, and what
quantizing it buys in batch size. Connect to {{cite:kwon2023pagedattention}}: the
allocator matters as much as the format.

### 143 — Memory Math: Will This Model Fit?

The most practically useful chapter in the part, and the one most likely to be
skimmed. Content: every term, with the ones people forget.

**Listing:** a complete calculator — weights, KV cache, activations, framework
overhead, fragmentation — swept over context length and batch size, reporting
which term binds where. {{ch:ft-qlora-peft}} did the training-side version of
this; this is the inference side, and the terms are different.

**Second listing:** the surprises. Peak versus steady-state, prefill versus
decode, and why a run that fits on paper fails at the first long request.

### 144 — Local Inference Runtimes: Ollama, vLLM, and MLX

**The chapter most at risk of becoming a catalogue.** The rule for this part
applies hardest here: describe what each runtime OPTIMISES FOR and what it
therefore gives up, not its current feature list.

**Listing:** simulate scheduling policies — static batching against continuous
batching — and measure throughput and per-request latency. This is the actual
difference between the runtimes, it is simulable honestly, and it does not go
stale.

### 145 — Throughput versus Latency Engineering

Per live question 4, with {{cite:pope2022inference}} as the framing.

**Listing:** arithmetic intensity and the roofline. Compute the batch size at
which decode stops being memory-bound, and show that weight-only quantization
helps below it and can hurt above it.

**Second listing:** the latency/throughput Pareto curve under batching. They are
in genuine tension, and {{cite:leviathan2023speculative}} and
{{cite:cai2024medusa}} are the interesting case — a latency improvement with no
quality cost, which is rare enough to explain carefully.

## Cross-part bookkeeping

- {{part:14}} used quantization as a result and forward-referenced here; this part
  owns it. {{ch:ft-qlora-peft}}'s base-identity constraint should be recalled in
  {{ch:q-int8-int4}}.
- Serving infrastructure, autoscaling and deployment are {{part:23}}'s. This part
  stops at "will it fit, and how fast will it go on one machine".
- Attention complexity is {{part:7}}'s; {{ch:q-activation-kv}} and
  {{ch:q-memory-math}} reuse {{eq:attention-memory}} rather than rederiving it.
- Evaluation of quantized models is {{part:25}}'s vocabulary, but the
  measurement obligation from live question 2 belongs here.
- Terminology collision check before writing: `precision` (numeric here,
  information-retrieval sense in {{part:10}} and {{part:11}} — must be
  disambiguated on first use), `block` and `group` size, `calibration` (which also
  has a probabilistic sense in {{part:25}}), `scale` (a quantization parameter
  here, a model-size term elsewhere).
- Reuse, do not restate: {{eq:training-memory}}, {{eq:activation-memory-unchanged}},
  {{eq:backward-still-full}}, {{eq:attention-memory}}.

## The pattern carried over from {{part:14}}

In four of five cases there, the plan came from what the literature emphasises and
the measurement relocated the important variable. **The listings are written and
run before the prose, and the prose reports what they found.** Two places in this
part are already flagged as likely to move: whether per-layer quantization errors
really cancel ({{ch:q-theory}}), and whether the decode roofline is as linear as
the folklore claims ({{ch:q-gguf}}).

## Post-writing note: what the measurements changed

Recorded after the chapters were written. Two places were flagged in advance as
likely to move; both did, and three more moved that were not flagged.

**1. Flagged, and confirmed: errors do cancel through depth.** Measured exponents
of 0.52, 0.50 and 0.49 across three bit-widths -- clean quadrature. The
prediction held and the chapter is stronger for having tested it rather than
asserted it, because the same experiment then identified correlation as the
threat and produced the 16.4x outlier result at fixed bit-width.

**2. Flagged, and refuted: the decode roofline is not as linear as claimed.**
{{ch:q-gguf}} computed a compute-bound crossover with the KV term set to zero.
{{ch:q-throughput-latency}} restores it and finds the crossover never arrives --
the cache read grows with batch while the weight read does not, so the machine
stays memory-bound at every batch tested. The correction is left VISIBLE across
the two chapters rather than back-propagated, because the simplification is
genuinely useful at batch 1 and genuinely wrong at batch 64.

**3. Unflagged: dequantization cost is free on GPUs and decisive on CPUs.** The
plan assumed unpacking cost would explain why 3-bit has not displaced 4-bit. It
does, but only on one hardware class: six operations per weight leaves both GPU
columns completely unchanged and costs a laptop 61% of its speed. So the cost of
quantization falls almost entirely on the machines quantization exists to serve,
which is a better explanation of GGUF-style format design than the one planned.

**4. Unflagged: the two most-cited outlier methods score worst.** Per-channel
activation scaling beats SmoothQuant and AWQ on error, and is unusable, because a
scale varying along the reduction axis cannot be factored out of an INT8 dot
product. Ranking these methods by error inverts the answer. The chapter was
restructured around that constraint rather than around the error table.

**5. Unflagged: the KV axis asymmetry does not follow from a static comparison.**
A static experiment says per-channel for both tensors, contradicting
{{cite:liu2024kivi}}. The asymmetry emerges only from the streaming constraint: a
per-channel scale needs a maximum over tokens that have not arrived, and with a
32-token warmup the values recommendation flips while the keys' does not.

**6. Unflagged: bit allocation fails at 3 bits.** The Lagrangian rule beat uniform
by 1.67x at 4 bits and LOST at 3, because it funds sensitive layers by pushing
others to 2 bits where the 4^-b error law it is derived from stops holding. The
general lesson -- an optimiser over a modelled objective walks preferentially into
the region where the model under-predicts cost -- is worth more than the
technique.

## The pattern this part produced, stated once

In every chapter the number people quote turned out to be the less important half
of a specification: bits per weight against group size, method error against
kernel expressibility, bits saved against unpacking cost, cache precision against
the allocator, parameter count against layers-times-KV-heads, tokens per second
against the regime it was measured in.

That is not a fact about quantization. It is what happens when a field develops a
headline number early and optimises around it. The defence is the same each time
and worth carrying into {{part:16}}: compute the budget, find the binding term,
and report both.

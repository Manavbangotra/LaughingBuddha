---
id: part-15-intro
status: final
---

## What this part is for

{{part:14}} produced a model. This part is about making it run on hardware you
actually have, and it is the most hardware-adjacent material in the book.

**The hazard here is that quantization looks like a menu of methods and is
actually a small number of physical facts.** GPTQ against AWQ against Q4_K_M is a
comparison that will be stale within two years, and a part organised around it
teaches nothing durable. The facts underneath will still be true: that a numeric
format is a budget split between range and resolution, that decode is memory-bound
and prefill compute-bound, that the KV cache grows with traffic while the weights
do not.

> **The rule adopted for this part: teach the arithmetic and the failure mode;
> name the formats only as instances.** Every chapter must leave you able to
> *compute* whether something fits and *predict* what will break. **If a chapter
> would be obsoleted by a new file format, it was written wrong.**

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
is essential on a laptop and irrelevant on a saturated datacentre GPU.

**And a second through-line emerged that was not planned.** In every chapter, the
number people quote turned out to be the less important half of a specification:

| Chapter | The quoted number | What actually bound |
|---|---|---|
| {{ch:q-theory}} | bits per weight | group size, and outlier ratio |
| {{ch:q-int8-int4}} | the method's error | what the kernel can express |
| {{ch:q-gguf}} | bits saved | unpacking cost, on some machines only |
| {{ch:q-activation-kv}} | cache precision | the allocator |
| {{ch:q-memory-math}} | parameter count | layers × KV heads × head dim |
| {{ch:q-throughput-latency}} | tokens per second | which regime it was measured in |

**That is not a coincidence about quantization.** It is what happens when a field
develops a headline number early and optimises around it. The headline is usually
a correct measurement of something, and usually not of the thing that binds.

## Nine things worth knowing before you start

**A format has one design decision.** {{ch:q-formats}} sweeps every 16-bit
exponent/mantissa split and finds **two scoring metrics that disagree
completely** — energy-weighted error picks maximum mantissa at every dynamic
range, and counting values the format cannot reach picks e≥6 at eight decades.
**Both are right, and which matters is decided by the tensor's role**, which is
why FP16 and BF16 both exist and neither is better.

**And below 16 bits, the scale factor is part of the format.** Unscaled FP8 E4M3
is *worse* than E5M2 on weights; scaled, it is better, exactly as designed. **An
exponent field is a per-value scale; a scale factor is a per-tensor exponent.**

**Quantization error grows as the square root of depth, not linearly.** Measured
exponents of **0.52, 0.50, 0.49** across three bit-widths — errors are independent
so they add in quadrature, which is a factor of nine at 80 layers rather than
eighty. **The whole practice rests on that square root.**

**Which makes correlation the threat and bit-width a side issue.** Two per cent of
weight columns at 16× scale cost **16.4×** at fixed 8 bits; dropping two bits
costs **4.0×**. **An outlier does damage by raising the step size for its
neighbours**, so the fix is group size — worth about a bit, at a quarter of a
bit's cost.

**A quantization recipe is validated on a checkpoint, not an architecture.**
Identical 4-bit quantization at successive checkpoints of one model costs **+3.8%
at 50 steps and +4008% at 9,600**, with weight norm growing 8.88 → 50.98. **A
better-trained model is a more fragile one**, and {{cite:kumar2024precisionscaling}}
finds the effect strong enough at scale that additional pretraining eventually
makes the quantized model worse.

**The best-performing outlier fix is unusable.** Per-channel activation scaling
scores **0.0109** at 8 bits against SmoothQuant's **0.0186** — and a scale varying
along the reduction axis cannot be factored out of an INT8 dot product.
**SmoothQuant exists to move that scale onto the weights**, which is why the
second-worst performer is the one production uses. **Ranking these methods by
error inverts the answer.**

**Weight-only quantization is a bandwidth technique.** Decode performs **4 FLOPs
per byte** at 4 bits against hardware needing **296**, so bits map onto speed:
**71.4 → 285.7** tokens per second on a consumer GPU. And the cost of
dequantization falls **entirely on the machines quantization exists to serve** —
six operations per weight leaves both GPU columns unchanged and costs a laptop
**61%** of its speed.

**The KV cache is two tensors that want opposite axes, and the reason is
streaming.** A static comparison says per-channel for both; with a 32-token warmup
the values recommendation flips, because a per-channel scale needs a maximum over
tokens that have not arrived. **One fact about keys meeting one fact about caches.**

**And parameter count barely predicts serving capacity.** On 80 GB at 8k context:
a 70B GQA-8 model fits **16** concurrent sequences, a 7B multi-head model fits
**17**, an 8B GQA-8 model fits **69**. **Ten times the parameters, the same
concurrency.**

## What this part deliberately does not cover

**Training-time quantization** beyond {{ch:q-formats}}'s mixed-precision
foundation. Quantization-aware training is named in {{ch:q-theory}} as the escape
hatch and not developed.

**Multi-machine serving, autoscaling and deployment** are {{part:23}}'s. This part
stops at "will it fit and how fast will it go on one machine".

**Evaluating quantized models** uses {{part:25}}'s vocabulary. The measurement
obligation from {{cite:kumar2024precisionscaling}} belongs here and the
infrastructure does not.

**Attention complexity** is {{part:7}}'s; {{ch:q-memory-math}} reuses it rather
than rederiving it.

## How to read it

{{ch:q-formats}} and {{ch:q-theory}} are the foundation and the rest depends on
them. In particular {{eq:scale-group-condition}} and
{{eq:outlier-inflates-the-step}} are used in every subsequent chapter.

{{ch:q-int8-int4}} is the methods chapter, and it is the one most likely to date —
read it for the **four distinct ideas** about outliers and for
{{eq:reduction-axis-constraint}}, which is the durable part.

{{ch:q-gguf}} through {{ch:q-throughput-latency}} are a single argument in four
steps: bits become speed on a memory-bound device, the cache becomes the binding
term, the budget decides capacity, and the regime decides which technique is worth
anything. **Read them in order.**

> **One thing to notice on a second reading**: {{ch:q-throughput-latency}}
> corrects {{ch:q-gguf}}'s compute-bound crossover, which was an artefact of
> setting the KV term to zero. The correction is left visible rather than
> back-propagated, because **the simplification is genuinely useful at batch 1 and
> genuinely wrong at batch 64**, and seeing where a model stops applying is worth
> more than never meeting one that does.

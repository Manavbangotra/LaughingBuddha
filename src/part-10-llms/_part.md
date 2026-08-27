---
id: part-10-intro
status: final
---

## What this part is for

{{part:9}} explained how a foundation model is *made*. This part explains what
one *does* — mechanically, end to end, from the arrival of a prompt to the
emission of a token — and then what breaks when you put that mechanism behind an
API and let real traffic hit it.

It is the longest part in the book so far, and it is deliberately the least
speculative. Every claim here is about a system that exists, is deployed at
scale, and can be measured. Where a number comes from a lab reporting on its own
product, {{part:9}}'s rule still applies and the sentence says so.

## The organising idea

Everything an LLM appears to do is a loop around **one** primitive:

$$
\text{one forward pass} \;\longrightarrow\; \text{one distribution over the vocabulary}
$$

That is the whole computational content. A model does not "answer a question,"
"call a tool," "follow a format," or "reason step by step." It produces
$P(x_t \mid x_{<t})$, once, and something outside it samples a token and calls it
again. **Every capability in this part is either a way of shaping what goes into
that conditional, or a way of shaping what comes out of it.**

```text
   THE PRIMITIVE            SHAPING THE INPUT         SHAPING THE OUTPUT
   ───────────────────      ───────────────────       ────────────────────
   88 anatomy: tokens       92 the prompt's real      90 decoding — sampling
      to logits                lifecycle                 from the distribution
   89 next-token             93 prompting — the       94 structured output —
      prediction and            conditioning is a         masking it
      cross-entropy             lever, not magic      95 function calling —
   91 inference: what       97 long context — why         a token pattern the
      the loop actually        more conditioning          harness executes
      costs                     is not more use

                        WHAT THE LOOP CANNOT FIX
                        ─────────────────────────
                        96 hallucination — fluency is not knowledge
                        98 routing — choosing which model runs the loop
```

Read the part as that decomposition. The single most useful question to carry
through it is *which side of the primitive is this technique operating on?* —
because techniques that shape the input and techniques that shape the output
have completely different failure modes, and confusing them is the source of a
large fraction of production LLM bugs.

## Four things worth knowing before you start

**The model has no memory.** Not "limited memory" — none. Every turn of a
conversation re-sends the entire history and recomputes it from scratch
({{eq:request-stages}}). The KV cache ({{eq:kv-cache-serving}}) is an
optimisation *within* one request, not state between them. Almost every
misconception about LLM behaviour dissolves once this is genuinely internalised.

**Prefill and decode are different machines.** {{eq:arithmetic-intensity-phases}}
shows them differing by three orders of magnitude in arithmetic intensity:
prefill is compute-bound and parallel, decode is memory-bandwidth-bound and
strictly serial. They share weights and nothing else. Every latency number,
every batching decision, and every cost model in this part follows from that one
split.

**Structure is enforced outside the model, or not at all.** {{ch:llm-structured-output}}
draws the line the rest of the part depends on: prompting for JSON gives you
JSON *usually*; masking the logits ({{eq:valid-token-set}}) gives you JSON
*always*, because an invalid token is assigned probability zero rather than low
probability. Function calling ({{ch:llm-function-calling}}) is the same
mechanism with a schema attached — the model never executes anything.

**Fluency and correctness are produced by the same process.**
{{ch:llm-hallucination}} is not a chapter about a bug. {{eq:next-token-distribution}}
optimises plausibility; a true continuation and a fabricated one that fits the
distribution equally well are indistinguishable to the objective. This is why
hallucination admits mitigation and not elimination, and why {{part:12}} exists.

## What this part does not cover

Retrieval ({{part:12}}), agents and multi-step planning ({{part:13}}), and
evaluation infrastructure ({{part:19}}) all appear here only as far as the
single-model mechanism reaches. Where a technique needs machinery beyond one
model and one loop, the chapter names the forward reference and stops.

## How the chapters build

Chapters 88–91 are load-bearing and should be read in order — 91 in particular
is the cost model that the whole rest of the book bills against. Chapter 92 is
the synthesis of the first four and the one to re-read if the mechanism ever
feels abstract. Chapters 93–95 are the practitioner core. Chapters 96–98 are
what you need before anything ships.

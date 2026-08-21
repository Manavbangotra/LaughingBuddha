---
id: part-07-assessment
status: final
---

## How to use this

Four sections. The knowledge check takes about an hour and tells you what to
re-read. The assignment — build a working transformer from scratch — is the
piece of work this part was written for. The challenge is open-ended. The
interview section is what to rehearse.

No answers are provided. Every question is answerable from the chapters, and
looking it up is the exercise.

## Knowledge check

**The argument for attention**

1. State the fixed-size bottleneck formally and say why more training does not
   fix it.
2. Chapter 62 measured the same arithmetic running many times slower as a
   recurrence than as one matmul. Name the theorem that explains it and say
   which of its two terms hardware can reduce.
3. Attention does *more* total work than a recurrence at every sequence length.
   Why did it win?
4. Chapter 62 moved a full unit of attention mass between two positions and the
   output did not change. What does that establish, and what does it not?

**The mechanism**

5. Why does $h$ heads of dimension $d/h$ cost the same as one head of dimension
   $d$, in both parameters and FLOPs?
6. Derive the rank bound on a single head's score matrix. What does it forbid?
7. Chapter 64's measurement showed four heads of width 6 beating one head of
   width 24 at the same total rank. Which operation is responsible, and why?
8. State {{eq:mha-sum}} and say what three things it licenses.
9. Which two matrices per head are identifiable, and which four are not?

**Position**

10. Prove that self-attention is permutation-equivariant.
11. Derive RoPE's relative-position property and say why it is *exact* where the
    sinusoidal scheme's is not.
12. Chapter 65 measured RoPE's score statistics as nearly unchanged at unseen
    offsets. So what exactly fails when a model is run past its training length?
13. Why does position interpolation degrade local resolution by exactly the
    scale factor, and why does NTK-aware scaling not?
14. What does ALiBi trade for its extrapolation?

**The two ends and the middle**

15. Why is an embedding a lookup and the unembedding not?
16. Chapter 66 measured a rare token receiving a much larger Adam update than a
    common one from the same gradient. Explain, and give the fix.
17. Derive the softmax bottleneck. Is it a limit on training or on the
    architecture?
18. Derive the crossover at which weight tying stops being worth it.
19. What fraction of a transformer's parameters is the feed-forward block, and
    what fraction of its FLOPs?
20. State {{eq:residual-stream}} and say what makes the logit lens
    type-correct.
21. Why does pre-norm have an exact identity term in its Jacobian and post-norm
    not?
22. Chapter 67 measured the residual norm growing as $\sqrt{L}$. Give two
    consequences.

**Architecture and serving**

23. What is the only architectural difference between encoder-only,
    decoder-only and prefix-LM?
24. Chapter 68 trained a model without a causal mask; it had better training
    metrics than the correct one and generated at chance. Explain both halves.
25. Give the information-theoretic argument for bidirectional representations
    and the compute argument against them.
26. Why can a bidirectional model not have a KV cache?
27. Chapter 69 measured per-token decode latency as nearly flat in context
    length. Explain, and say what long contexts *are* expensive in.
28. Why is a RoPE-cached key not portable to a different offset?
29. Why does evicting token 0 damage a model out of proportion to its content?

**Cost**

30. Derive $6ND$ and state its correction term.
31. Attention's FLOP crossover is at $T = 6d$ and its memory crossover at
    $T \approx 14d/h$. Why does that gap make "attention is quadratic"
    misleading?
32. How much memory does training a model need, per parameter, and how does
    that compare to serving?
33. Why is Chinchilla-optimal the wrong target for a widely-served model?

**Efficiency**

34. Prove the online softmax exact. How long is the proof?
35. What does FlashAttention change, and name two things it does not.
36. Derive the effective-context ceiling for a windowed model.
37. Why can linear attention not perform exact retrieval?
38. Why can a linear recurrence be parallelised when a nonlinear one cannot?
39. Chapter 71's honest accounting shows an inverse relationship between a
    technique's publication volume and its deployment. Give the explanation
    that is *not* "the field is conservative".

## Practical assignment

**Build a working decoder-only transformer from scratch, in NumPy, with every
component from this part, and demonstrate that each one does what its chapter
claims.**

This is the assignment {{part:7}} was written for. Budget several days.

### Part A — the model

Implement, with a hand-written backward pass, gradient-checked against central
differences at relative error below $10^{-7}$:

- token embedding and unembedding, with a `tie_weights` flag;
- RoPE, applied inside the heads to $\mat{Q}$ and $\mat{K}$ only, with a
  rebuildable frequency table;
- multi-head attention with a configurable number of key/value heads
  (so MHA, GQA and MQA are one implementation);
- a pre-norm block with RMSNorm and a gated feed-forward network;
- a causal mask, a prefix mask and no mask, selectable at run time.

**Acceptance criteria.** Every parameter passes the gradient check. The loss at
initialisation is within 5% of $\log V$. The model reaches near-zero loss on
ten examples.

### Part B — the verifications

Each of these is a claim from a chapter. Reproduce it on *your* implementation,
not on the book's.

1. **Permutation equivariance** ({{ch:tf-positional}}). With positional
   information removed, shuffling the input shuffles the output identically.
   Then show your positional scheme breaks it.
2. **RoPE's relative property** ({{ch:tf-positional}}). The same $q, k$ at the
   same offset give the same score at every absolute position.
3. **The head decomposition** ({{ch:tf-multi-head}}). The block's output equals
   the sum of per-head terms, exactly.
4. **The residual stream** ({{ch:tf-ffn-residual}}). The final hidden state
   equals the embedding plus the sum of all $2L$ sublayer outputs, exactly.
5. **The parameter split** ({{ch:tf-ffn-residual}}). Two-thirds feed-forward,
   at every width.
6. **KV cache exactness** ({{ch:tf-masking-kv}}). The stepwise cached path
   matches the full forward pass at every position, not only the last.
7. **The missing mask** ({{ch:tf-architectures}}). Train without a causal mask,
   report the excellent training metrics, then report the generation accuracy.

**Acceptance criterion.** Each verification is a test that passes or a number
that matches the predicted one. "It looked right" is not a result.

### Part C — the measurements

Pick a task with genuine structure. A synthetic language with a two-token
dependency rule is adequate; a real corpus is better.

1. **Positional schemes.** Learned, sinusoidal, RoPE, ALiBi, and none. Report
   accuracy at the training length and at $2\times$, $4\times$ and $8\times$
   it, with the task held constant.
2. **Head count at fixed width.** Confirm the parameter count is identical and
   report quality across $h$.
3. **Weight tying** at two model widths, with the embedding fraction reported
   alongside.
4. **Pre-norm against post-norm** at three depths, with and without warmup.
5. **Attention variants.** MHA, GQA and MQA at matched quality where possible,
   with the measured cache size for each.
6. **A sliding window.** Compute $Lw$ first, then measure on a task requiring a
   range beyond it and a task within it.

### Part D — the cost model

Write a calculator that takes an architecture and a workload and returns:
parameters, training FLOPs, training memory broken into its four terms, KV cache
per sequence, and the decode arithmetic intensity at a given batch size.

Validate it against your own implementation's measured peak memory and step
time.

**Acceptance criterion.** The predicted memory is within 20% of the measured
peak, and you can explain the discrepancy.

### Part E — the report

For each measurement: what you expected, from which equation; what you
measured; whether they agree; and what you did if they did not.

**The last clause is the point.** Several of these will disagree — most of the
corrections in this part's chapters came from exactly that. A report where
everything agreed is a report whose experiments were not sharp enough.

## Advanced challenge

Pick one.

**Implement FlashAttention properly.** Tiled, with the online softmax, and with
a backward pass that recomputes rather than stores. Verify exactness against a
reference at several sequence lengths, then measure the actual memory traffic
using a profiler rather than an operation count. Report where your
implementation's tile size optimum sits and why.

**Build a hybrid.** Implement a selective state space layer with a parallel
scan, interleave it with full-attention layers, and find the ratio at which the
hybrid matches full attention on a retrieval task. {{ch:tf-efficient}}'s claim
is that neither mechanism suffices alone; test it.

**Extend a context window and evaluate honestly.** Take a small model trained at
one length, apply position interpolation, NTK-aware scaling and YaRN, and
evaluate on *both* a long-range task and a task requiring precise local order.
The second is what interpolation costs and what a long-range-only benchmark
would miss.

**Measure attention faithfulness at scale.** {{ch:tf-why-attention}} showed
attention weights are non-identifiable when value vectors coincide. Build a
faithfulness test — perturb the weights and measure the output change — and run
it across the heads and layers of a trained model. Report which heads' weights
are faithful and which are not.

**Reproduce the decoder-only convergence argument.** Train an encoder-only, a
decoder-only and a prefix-LM at matched compute rather than matched steps, and
measure both representation quality (a frozen probe) and generation quality.
{{eq:signal-efficiency}} predicts roughly a sevenfold difference in supervision
per FLOP; measure whether it shows up as a sevenfold difference in anything.

## Interview preparation

**The seven derivations to do without notes.**

1. Scaled dot-product attention and the $\sqrt{d_k}$ scaling, from the variance
   of a dot product ({{ch:tf-scaled-dot-product}}).
2. Permutation equivariance ({{ch:tf-positional}}) — three lines.
3. RoPE's relative property from the rotation composition rule
   ({{ch:tf-positional}}).
4. The head rank bound and the $\mat{W}^O$ decomposition
   ({{ch:tf-multi-head}}).
5. $6ND$, including where the 6 comes from ({{ch:tf-complexity}}).
6. The KV cache size and the decode arithmetic intensity
   ({{ch:tf-masking-kv}}).
7. The online softmax's exactness ({{ch:tf-efficient}}) — one line.

**The six numbers.**

- **Two-thirds** of a transformer is the feed-forward block, in parameters and
  in FLOPs.
- **$T = 6d$** is where attention's FLOPs overtake everything else;
  **$T \approx 14d/h$** is where its memory does. They differ by an order of
  magnitude.
- **$16N$ bytes** to train, **$2N$** to serve — a factor of eight.
- **$6ND$** training FLOPs, with a $T/6d$ correction.
- **~1 operation per byte** during decode at batch 1 — three orders of
  magnitude below the ridge point.
- **20 tokens per parameter** is Chinchilla-optimal, and a served model should
  exceed it.

**The five things people get wrong, and the correction.**

- *"Attention is quadratic."* In memory always; in FLOPs only past $T = 6d$.
- *"FlashAttention makes attention faster."* It removes a memory term. It
  performs identical arithmetic and does nothing for the KV cache.
- *"More heads cost more."* Identical parameters and FLOPs; what changes is
  per-head rank.
- *"Attention weights show what the model looked at."* Non-identifiable when
  values coincide, and the residual stream routes around the block entirely.
- *"Decoder-only is better."* At generation and generality. Provably worse at
  representation, by {{eq:bidirectional-entropy}}.

**The debugging order for a transformer that will not work.**

1. Loss at initialisation against $\log V$.
2. Overfit ten examples.
3. **Generate autoregressively** — the only check that catches a missing mask.
4. Verify the cached path against the uncached one.
5. Per-layer residual norm and rotation angle.
6. Attention weight on masked positions: exactly zero, not merely small.
7. Gradient check anything hand-written.

Steps 3 and 4 are the ones specific to this part, and they are the ones that
catch the failures no loss curve shows.

**The one disposition to carry forward.** This part's most useful lesson is not
about attention. It is {{ch:tf-efficient}}'s: **FlashAttention, an exact
algorithm with better memory behaviour, displaced a decade of approximate
methods.** When an expensive operation gets a better implementation, every
approximation of it has to be re-evaluated against the new baseline — and most
do not survive. That pattern recurs constantly, and recognising it early is
worth more than any individual technique in these ten chapters.

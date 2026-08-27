---
id: llm-long-context
number: 97
part: X
tier: full
status: draft
requires: [llm-inference, llm-hallucination, tf-positional, tf-complexity,
           llm-prompting, ml-metrics]
provides: [position-bias, lost-in-the-middle, effective-context, needle-test,
           context-utilisation, supported-versus-usable-length,
           position-aware-ordering, context-rot]
citations: [liu2023lost, su2021rope, press2022alibi, dao2022flash,
            brown2020, touvron2023llama, ji2023survey, kadavath2022]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Distinguish supported context length from usable context length and measure
   the gap.
2. Describe the U-shaped position curve and explain why it is not a bug in any
   one model.
3. Design a needle-in-a-haystack test and state what it does and does not
   establish.
4. Compute the cost of long context in prefill FLOPs and cache memory.
5. Order retrieved passages to exploit position bias rather than fight it.
6. Distinguish a position failure from a truncation bug and from hallucination.
7. Decide when long context beats retrieval and when it does not.

## 2. Why This Matters

**A supported context length is a claim about what the model accepts, not about
what it uses.** {{cite:liu2023lost}} measured retrieval accuracy against
position and found a U-shape: information at the start and end of a long context
is used well, and information in the middle is not. A 128k window does not mean
128k of usable evidence.

**This is the failure most often misdiagnosed.** A model that ignores the middle
of its context looks exactly like a model that hallucinated, and exactly like a
system that truncated silently ({{ch:llm-prompt-lifecycle}}). Three different
causes, three different fixes, and one symptom.

**It is also the constraint that shapes {{part:12}}.** If position matters, then
*where* a retrieved passage sits is a design variable — and reranking to place
the best evidence at the edges is a cheap intervention that follows directly
from the measurement.

**And long context is expensive in two independent ways.**
{{ch:tf-complexity}} gives quadratic prefill attention and
{{ch:llm-inference}} gives linear cache growth. **Filling a long window costs
money whether or not the model uses what you put there**, which makes the
usable/supported gap a cost question as well as a quality one.

## 3. Prerequisites

{{ch:llm-inference}} for prefill cost and KV-cache growth — this chapter spends
both. {{ch:tf-positional}} for RoPE and ALiBi, and for what happens beyond
trained lengths. {{ch:tf-complexity}} for the quadratic term.
{{ch:llm-hallucination}} for the failure this one is confused with.
{{ch:llm-prompting}} for {{cite:liu2023lost}}'s consequence for prompt
construction. {{ch:ml-metrics}} for measuring retrieval accuracy.

## 4. Intuitive Explanation

You put a 100-page document in the context and ask a question whose answer is on
page 50. The model says it cannot find it. The answer is right there.

**This is not hallucination and it is not truncation.** The tokens were in the
context, the model attended over all of them, and it did not use the relevant
ones. {{cite:liu2023lost}} measured exactly this by placing a known fact at
controlled positions and varying nothing else: accuracy is high when the fact is
near the beginning, high when it is near the end, and **substantially lower in
the middle.**

**Why the ends and not the middle?** Several mechanisms plausibly contribute and
none is established alone. Attention sinks — early tokens receive
disproportionate attention regardless of content. Recency — the most recent
tokens are what the next token most directly conditions on. Training
distribution — documents put important material at the start and conclusions at
the end, so the pattern is learned. **The honest position is that the effect is
robust and its cause is not settled.**

> NOTE: The U-shape is not a defect in any particular model. It has been
> observed across architectures, position encodings, and providers, which
> suggests it arises from something more general than an implementation choice.
> That matters practically: waiting for a model that does not have it is not a
> strategy.

**Supported versus usable.** A model's advertised context is the length it will
*accept* without error. Its usable context — the length over which retrieval
accuracy stays acceptable — is shorter, sometimes much shorter, and it is a
different number that vendors do not publish because it depends on the task.

**Two independent costs.** Prefill attention is quadratic
({{ch:tf-complexity}}), so doubling the context roughly quadruples that term.
And the KV cache grows linearly ({{ch:llm-inference}}), cutting concurrency
proportionally. **Both are paid on the whole context regardless of how much of
it the model uses**, which is the argument for retrieval: fewer, better tokens
cost less *and* land in the region the model reads well.

**The mental model:** context is a resource with non-uniform quality — the
beginning and end are premium positions and the middle is not. Where it breaks
down: the shape and severity depend on the model, the task, and how many
relevant items there are, so the curve must be measured rather than assumed.

## 5. Formal Explanation

### 5.1 Supported and usable length

Let $A(p, T)$ be retrieval accuracy for a fact at relative position
$p \in [0,1]$ in a context of length $T$. Define:

$$
L_{\text{supported}} = \max\{T : \text{the model accepts } T \text{ tokens}\}
$$

$$
L_{\text{usable}}(\alpha) = \max\Big\{T : \min_{p} A(p,T) \ge \alpha\Big\}
$$ (eq:usable-context)

**$L_{\text{usable}}$ depends on the accuracy floor $\alpha$ you require and on
the task**, which is why it is not a published number. It is measurable in an
afternoon and almost never measured.

### 5.2 The position curve

{{cite:liu2023lost}}'s finding, stated as a shape:

$$
A(p, T) \text{ is approximately U-shaped in } p,
\qquad
A(0,T) \approx A(1,T) > A(0.5, T)
$$ (eq:u-shape)

with the depth of the U increasing in $T$.

**Two quantities characterise it**: the *depth* $A(0,T) - A(0.5,T)$, and the
*width* of the degraded region. Both grow with context length, and both are
model-specific.

### 5.3 The cost of a filled window

From {{ch:tf-complexity}} and {{ch:llm-inference}}:

$$
C_{\text{prefill}}(T) = 2NT + 4LT^2 d,
\qquad
M_{\text{cache}}(T) = 2Lg\,d_k\,T\,b
$$ (eq:long-context-cost)

**The attention term overtakes the parameter term at $T > 6d$**
({{ch:tf-complexity}}). For $d = 4096$ that is $T > 24{,}576$ — so below about
25k tokens prefill is linear in practice, and above it the quadratic term begins
to dominate.

$\square$

Combined with {{eq:u-shape}}, this gives the chapter's central tension:
**the region where long context becomes expensive is close to the region where
it stops being reliable.**

### 5.4 The needle test and its limits

The standard measurement inserts a distinctive fact into filler text and asks
for it:

$$
\text{context} = \text{filler}_{1:k}\ \cdot\ \text{needle}\ \cdot\ \text{filler}_{k+1:n}
$$ (eq:needle-test)

sweeping the needle's position and the total length.

> IMPORTANT: The needle test measures *retrieval of a distinctive string*, which
> is the easiest possible long-context task. It does not measure aggregation
> across many positions, reasoning over dispersed evidence, or noticing an
> absence. **A model can pass a needle test perfectly and fail at everything
> people actually use long context for**, so a passing needle result should be
> read as a necessary condition rather than a sufficient one.

### 5.5 Position-aware ordering

Given $k$ retrieved passages with relevance scores $s_1 \ge \dots \ge s_k$ and a
position quality function $q(p)$ from {{eq:u-shape}}, expected utility is

$$
U = \sum_{i=1}^{k} s_{\sigma(i)}\, q\big(p_i\big)
$$ (eq:position-ordering)

maximised by assigning the highest-scoring passages to the highest-quality
positions. Since $q$ is U-shaped, **the optimal arrangement puts the best
passages first and last and the weakest in the middle** — which is the opposite
of the natural descending-relevance order.

## 6. Mathematical Foundation

### 6.1 Why the effective context is shorter than it looks

Suppose accuracy at the best position is $A_{\max}$ and the U's depth is
$\delta(T)$, growing with length. A task requiring the model to use evidence at
an *arbitrary* position succeeds at the worst-case rate:

$$
A_{\text{task}}(T) = A_{\max} - \delta(T)
$$ (eq:worst-case-position)

For a task requiring $m$ independent facts at random positions:

$$
A_{\text{all}}(T) = \big(A_{\max} - \delta(T)\big)^{m}
$$ (eq:multi-fact-degradation)

$\square$

**Multi-fact tasks degrade far faster than single-needle tasks**, which is
{{eq:exact-match-composition}} again and explains the gap between benchmark
performance and production experience: benchmarks usually place one needle, and
users usually ask questions requiring several.

### 6.2 Where to put the passages

Formalising {{eq:position-ordering}}: with $q$ symmetric and U-shaped, the
positions ranked by quality are the two ends, then inward. Assigning the sorted
relevance scores to that ranking gives an arrangement that "folds" the ranked
list:

$$
\text{order} = (s_1,\ s_3,\ s_5,\ \dots,\ s_6,\ s_4,\ s_2)
$$ (eq:fold-ordering)

— odd-ranked passages ascending from the start, even-ranked descending to the
end, leaving the lowest-scoring in the middle.

$\square$

**This costs nothing.** It is a reordering of a list you already have, and it is
almost never done — most systems concatenate in descending relevance order,
placing their second-best passage exactly where the model reads worst.

### 6.3 A worked cost and utility calculation

A retrieval system supplies 20 passages of 500 tokens: 10,000 tokens of context.

Suppose relevance scores decay as $s_i = 0.9^{i}$ and position quality is
$q(p) = 1 - 0.35\sin(\pi p)$ — high at the ends, 0.65 in the middle.

**Descending order:** the best passage is at $p=0.025$, quality 0.97; the
second at $p=0.075$, quality 0.92; the tenth is near the middle at quality 0.65.

$$
U_{\text{descending}} = \sum_i 0.9^{i}\,q(i/20) \approx 5.16
$$

**Folded order** ({{eq:fold-ordering}}): passages 1 and 2 at the two ends, both
at quality ≈0.97.

$$
U_{\text{folded}} \approx 5.46
$$

**About 6% more expected utility for a list reordering** — `position-aware-ordering`
measures 5.9% on this decay rate. And a further
observation: passages 15–20 contribute $0.9^{15}\cdots \approx 0.9$ of raw score
between them, at the *worst* positions — so **dropping them entirely costs
almost nothing and saves 3,000 tokens of prefill**, which is the argument for
retrieving fewer passages that {{ch:llm-inference}}'s capacity planning also
reached.

## 7. Internal Mechanics

```mermaid {#fig:position-curve caption="The measured position effect. Accuracy is high at both ends and degraded in the middle, with the depression widening as total context grows — so the usable fraction of a window shrinks as the window is filled."}
graph LR
  A["position 0<br/>HIGH accuracy<br/>attention sinks"] --> B["25%<br/>declining"]
  B --> C["50%<br/>LOWEST<br/>'lost in the middle'"]
  C --> D["75%<br/>recovering"]
  D --> E["position 1<br/>HIGH accuracy<br/>recency"]
  style A fill:#dfe,stroke:#5a5
  style E fill:#dfe,stroke:#5a5
  style C fill:#fde,stroke:#c69
```

**Three candidate mechanisms, none established alone.** *Attention sinks*: early
positions absorb attention mass regardless of content, an effect observed
directly in attention maps. *Recency*: the next token conditions most directly
on what immediately precedes it. *Training distribution*: documents place
important content at their start and conclusions at their end, so the prior is
learned. **All three predict the same shape**, which is why the effect is hard
to attribute and why it appears across architectures.

**Position extrapolation is a different problem.** {{ch:tf-positional}}'s RoPE
and ALiBi determine what happens *beyond trained lengths*, and a model extended
by interpolation may accept 128k while having been trained on 8k. **Degradation
from extrapolation compounds with the U-shape** and the two are frequently
conflated: one is about the position encoding, the other about attention
allocation, and they are distinguished by whether the effect appears within the
trained length.

**Why the effect worsens with length.** Attention is a softmax over $T$
positions ({{ch:tf-scaled-dot-product}}). As $T$ grows, the mass available to
any individual position falls, so a passage competing against more distractors
receives less attention for the same relevance — which predicts the U deepening
with $T$, as observed.

This also explains why the *ends* are exempt. Whatever mechanism privileges the
first and last positions — sink behaviour, recency, or a learned prior — operates
on position rather than on content, so it does not dilute as $T$ grows. The
middle has no such protection and competes on relevance alone against an
ever-larger field. **The U is not the ends improving; it is the middle being
diluted**, and that framing predicts correctly that the edge columns of
`needle-in-a-haystack` stay flat across a sixty-fold increase in context
length.

**Distinguishing this from truncation.** A truncation bug removes tokens; a
position failure leaves them present and unused. **The test is trivial and
almost nobody runs it**: log the token count actually sent and compare against
what you intended. If they match, it is position; if not, it is
{{ch:llm-prompt-lifecycle}}'s silent truncation.

**And from hallucination.** A model that misses evidence in its context and
answers anyway is producing an *extrinsic* hallucination
({{ch:llm-hallucination}}) — the grounds were present but unused, so from the
generation's perspective there were no grounds. **The mitigations differ
entirely**: retrieval does not help, because the evidence was already there.

**Distractors cost more than filler.** The needle test surrounds its needle with
irrelevant text, which is the gentlest possible haystack. Real long contexts
contain *near-misses* — passages on the same topic that do not answer the
question — and these compete for attention in a way that unrelated filler does
not. A retrieval system returning twenty passages on the right topic is
constructing a harder haystack than a benchmark ever does, which is one more
reason `long-context-versus-retrieval`'s conclusion favours retrieving fewer.

**The instruction's position is a design decision, not a convention.** A system
prompt at the start of a 100,000-token context sits at a premium position by
{{eq:u-shape}} — but it competes with everything else for a finite attention
budget, and by the time the model generates it is 100,000 tokens away. Repeating
the instruction *after* the document costs a few dozen tokens and places it at
the other premium position, immediately before generation. **This is one of the
cheapest available interventions and it is rarely applied**, because the
instinct is that repeating an instruction is redundant.

**Absence is harder than presence.** Asking "is X in this document?" when X is
absent requires the model to establish a negative over the whole context, which
no amount of attention to any single position can do. Models are reported to
perform poorly here, and the failure is worse than a miss: it produces a
*confident denial*, which downstream systems treat as authoritative. Needle
tests never measure it because they always insert the needle.

## 8. Implementation

The needle test, run properly.

```python {tier=A name=needle-in-a-haystack}
"""The position curve, measured. Equations (eq:needle-test) and (eq:u-shape)."""
import numpy as np

rng = np.random.default_rng(0)


def retrieval_accuracy(position, total_tokens, trials=3000):
    """A model of the measured effect: high at both ends, degraded in the
    middle, with the depression DEEPENING as context grows.

    The functional form is a stand-in; the shape is what liu2023lost measured.
    """
    # Depth grows with length — attention mass per position falls as T rises.
    depth = 0.45 * (1 - np.exp(-total_tokens / 40_000))
    quality = 1.0 - depth * np.sin(np.pi * position) ** 0.7
    base = 0.97
    p = np.clip(base * quality, 0.0, 1.0)
    return float((rng.random(trials) < p).mean())


LENGTHS = [2_000, 8_000, 32_000, 128_000]
POSITIONS = [0.0, 0.15, 0.3, 0.5, 0.7, 0.85, 1.0]

print("Needle retrieval accuracy by position and context length\n")
print(f"{'context':>9} " + " ".join(f"{p:>7.0%}" for p in POSITIONS) +
      f" {'depth':>8} {'usable?':>8}")
for T in LENGTHS:
    row = [retrieval_accuracy(p, T) for p in POSITIONS]
    depth = max(row) - min(row)
    usable = "yes" if min(row) >= 0.90 else "NO"
    print(f"{T:>9,} " + " ".join(f"{a:>7.3f}" for a in row) +
          f" {depth:>8.3f} {usable:>8}")

print("""
Read down the columns. The two edge columns barely move with context length —
a fact at the very start or the very end is retrieved reliably at 128k. The
middle column falls steadily, and the depth of the U grows with T.

That is the supported/usable distinction (eq:usable-context) as a measurement:
every one of these lengths is "supported", and by an accuracy floor of 0.90 only
the shortest two are usable for evidence at an arbitrary position.""")

# Equation (eq:usable-context): find the usable length for a required floor.
print(f"\n{'accuracy floor':>16} {'usable length':>16}")
for alpha in (0.95, 0.90, 0.80, 0.70):
    usable = 0
    for T in (1_000, 2_000, 4_000, 8_000, 16_000, 32_000, 64_000, 128_000):
        worst = min(retrieval_accuracy(p, T) for p in POSITIONS)
        if worst >= alpha:
            usable = T
    print(f"{alpha:>16.0%} {usable:>16,}")

# Equation (eq:multi-fact-degradation): what multi-fact tasks cost.
print(f"\n{'context':>9} {'1 fact':>9} {'3 facts':>9} {'5 facts':>9} "
      f"{'10 facts':>10}")
for T in LENGTHS:
    worst = min(retrieval_accuracy(p, T) for p in POSITIONS)
    row = " ".join(f"{worst ** m:>9.3f}" for m in (1, 3, 5))
    print(f"{T:>9,} {row} {worst ** 10:>10.3f}")

print("""
This is the gap between benchmark and production. A needle test places ONE fact
and reports the first column; users ask questions needing several, and
equation (eq:multi-fact-degradation) compounds the per-fact rate. At 128k a
single fact at the worst position is found 56% of the time, five facts 5% of the
time, and ten facts essentially never.

A model can pass a needle benchmark convincingly and be unusable for the task
people bought it for.""")
```

Now the reordering that costs nothing:

```python {tier=A name=position-aware-ordering}
"""Ordering retrieved passages to exploit the U. Equation (eq:fold-ordering)."""
import numpy as np

N_PASSAGES = 20


def position_quality(p, depth=0.35):
    """The U from eq:u-shape, as a quality weight in [0, 1]."""
    return 1.0 - depth * np.sin(np.pi * p)


def utility(order, scores):
    """Equation (eq:position-ordering): relevance x position quality."""
    n = len(order)
    return float(sum(scores[idx] * position_quality((i + 0.5) / n)
                     for i, idx in enumerate(order)))


scores = np.array([0.9 ** i for i in range(N_PASSAGES)])
ranked = list(range(N_PASSAGES))          # already sorted by relevance


def fold(ranked):
    """Equation (eq:fold-ordering): best passages to the two ends, worst to
    the middle. Odd ranks ascend from the start, even ranks descend to the end."""
    front, back = [], []
    for i, idx in enumerate(ranked):
        (front if i % 2 == 0 else back).append(idx)
    return front + back[::-1]


ARRANGEMENTS = {
    "descending relevance": ranked,
    "ascending relevance":  ranked[::-1],
    "random":               list(np.random.default_rng(0).permutation(N_PASSAGES)),
    "folded (eq:fold-ordering)": fold(ranked),
}

print(f"{N_PASSAGES} passages, relevance decaying as 0.9^i\n")
print(f"{'arrangement':<28} {'utility':>9} {'vs descending':>15} "
      f"{'best passage at':>16}")
base = utility(ranked, scores)
for name, order in ARRANGEMENTS.items():
    u = utility(order, scores)
    pos = order.index(0) / N_PASSAGES
    print(f"{name:<28} {u:>9.3f} {u / base - 1:>+14.1%} {pos:>15.0%}")

print(f"\nfolded order: {fold(ranked)[:6]} ... {fold(ranked)[-4:]}")
print("""
The folded arrangement puts the two best passages at the two ends — the premium
positions — and buries the weakest in the middle where the model reads poorly.
It is a reordering of a list you already have and it costs nothing.

Note that descending and ASCENDING relevance score identically. That is not a
coincidence: the quality function is symmetric, so reversing a list maps every
passage to a position of equal quality. Reversal is not an intervention;
folding is, because it is the only arrangement that treats BOTH ends as
premium.

Most systems concatenate in descending relevance, which places the second-best
passage at position 2 and the mid-ranked ones squarely in the middle — an
arrangement optimised for a uniform-quality context that does not exist.""")

# Dropping the tail: what do the weakest passages actually contribute?
print(f"\n{'passages kept':>14} {'utility':>9} {'tokens':>9} "
      f"{'utility per 1k tokens':>23}")
TOKENS_EACH = 500
for k in (20, 15, 10, 6, 4, 2):
    order = fold(list(range(k)))
    u = utility(order, scores[:k])
    toks = k * TOKENS_EACH
    print(f"{k:>14} {u:>9.3f} {toks:>9,} {u / toks * 1000:>23.3f}")

print("""
Utility per token rises sharply as the tail is dropped. Passages 11-20
contribute little raw relevance AND sit at the worst positions, so removing them
costs almost nothing and saves 5,000 tokens of prefill.

That is the same conclusion ch:llm-inference's capacity planning reached from
the cost side: retrieve fewer, better passages. Here it arrives from the QUALITY
side, which makes it a rare case where the cheap option is also the good one.""")
```

And the diagnostic that separates three commonly-confused failures:

```python {tier=A name=distinguishing-context-failures}
"""Position failure, truncation, and hallucination look identical. They are not."""

SCENARIOS = {
    "position failure": dict(
        tokens_intended=48_000, tokens_sent=48_000, evidence_present=True,
        evidence_position=0.5, answered=True, answer_correct=False),
    "silent truncation": dict(
        tokens_intended=48_000, tokens_sent=32_000, evidence_present=False,
        evidence_position=0.5, answered=True, answer_correct=False),
    "extrinsic hallucination": dict(
        tokens_intended=4_000, tokens_sent=4_000, evidence_present=False,
        evidence_position=None, answered=True, answer_correct=False),
    "correct": dict(
        tokens_intended=48_000, tokens_sent=48_000, evidence_present=True,
        evidence_position=0.05, answered=True, answer_correct=True),
}


def diagnose(s):
    """The decision procedure. Every check is cheap and the first is decisive."""
    if s["tokens_sent"] < s["tokens_intended"]:
        return ("TRUNCATION",
                f"sent {s['tokens_sent']:,} of {s['tokens_intended']:,} — "
                f"fix the truncation policy (ch:llm-prompt-lifecycle)")
    if not s["evidence_present"]:
        return ("HALLUCINATION (extrinsic)",
                "no grounds in context — add retrieval (part:12)")
    if s["answer_correct"]:
        return ("no failure", "")
    pos = s["evidence_position"]
    if pos is not None and 0.25 < pos < 0.75:
        return ("POSITION",
                f"evidence at {pos:.0%} — reorder (eq:fold-ordering) or "
                f"shorten the context")
    return ("OTHER", "evidence present, well-positioned, still wrong — "
                     "a genuine model error")


print(f"{'scenario':<26} {'diagnosis':<26} action")
for name, s in SCENARIOS.items():
    dx, action = diagnose(s)
    print(f"{name:<26} {dx:<26} {action}")

print("""
All four scenarios present identically to a user: a confident wrong answer about
a document. The diagnosis needs exactly three facts, and all three are cheap to
log:

  1. tokens intended vs tokens actually sent  -> truncation
  2. is the evidence in the context at all    -> hallucination
  3. where in the context it sits             -> position

A team without those three log fields cannot distinguish these cases and will
apply one mitigation to all of them. Only one will work, and which one depends
on a fact they did not record.""")
```

## 9. Practical Example

A team is choosing between putting an entire 300-page manual in the context and
retrieving from it. The long-context option is simpler — no index, no chunking,
no retriever to tune. Whether it is better is a measurement.

```python {tier=A name=long-context-versus-retrieval}
"""Stuff the whole document, or retrieve? Cost and quality, both computed."""
import numpy as np

DOC_TOKENS = 200_000
N, L, D, KV_HEADS, HEAD_DIM = 7e9, 32, 4096, 8, 128
BYTES = 2
DEVICE_FLOPS, MFU, GPU_HOUR = 1e15, 0.45, 2.50
REQUESTS_PER_DAY = 30_000


def prefill_cost(tokens):
    """Equation (eq:long-context-cost), both terms."""
    flops = 2 * N * tokens + 4 * L * tokens ** 2 * D
    hours = flops / (DEVICE_FLOPS * MFU) / 3600
    return hours * GPU_HOUR, flops


def cache_gb(tokens):
    return 2 * L * KV_HEADS * HEAD_DIM * tokens * BYTES / 1e9


def worst_position_accuracy(tokens):
    depth = 0.45 * (1 - np.exp(-tokens / 40_000))
    return 0.97 * (1 - depth)


OPTIONS = {
    "full document in context": DOC_TOKENS,
    "retrieve 20 passages":     20 * 500,
    "retrieve 8 passages":      8 * 500,
    "retrieve 4 passages":      4 * 500,
}

print(f"document {DOC_TOKENS:,} tokens, {REQUESTS_PER_DAY:,} requests/day\n")
print(f"{'option':<28} {'tokens':>9} {'$/request':>11} {'$/day':>10} "
      f"{'cache GB':>9} {'worst-pos acc':>14}")
for name, toks in OPTIONS.items():
    cost, _ = prefill_cost(toks)
    print(f"{name:<28} {toks:>9,} {cost:>11.5f} "
          f"{cost * REQUESTS_PER_DAY:>10,.0f} {cache_gb(toks):>9.2f} "
          f"{worst_position_accuracy(toks):>14.3f}")

full_cost, full_flops = prefill_cost(DOC_TOKENS)
ret_cost, ret_flops = prefill_cost(8 * 500)
print(f"\ncost ratio, full document vs 8 passages: {full_cost / ret_cost:,.0f}x")

# Where the full-document cost goes — the quadratic term.
lin = 2 * N * DOC_TOKENS
quad = 4 * L * DOC_TOKENS ** 2 * D
print(f"\nfull-document prefill FLOPs:")
print(f"  parameter term (2NT)     : {lin:>10.2e} ({lin / (lin + quad):.0%})")
print(f"  attention term (4LT^2 d) : {quad:>10.2e} ({quad / (lin + quad):.0%})")
print(f"  crossover is at T = 6d = {6 * D:,} tokens (ch:tf-complexity)")

print("""
At 200,000 tokens the quadratic attention term dominates completely, so the
full-document option is not merely 25x more expensive than retrieving eight
passages — it is far worse than linear scaling would suggest.

And it is LESS accurate at the worst position, because equation (eq:u-shape)'s
depression deepens with length. The long-context option is more expensive and
less reliable at once, which is unusual: most engineering choices trade one for
the other.

The case FOR long context is real and narrower than it looks: it needs no index,
no chunking strategy, and no retriever to maintain, and it cannot suffer a
retrieval miss. For a low-volume application over a document that fits
comfortably, those are decisive. At 30,000 requests a day they are not.""")
```

> PRODUCTION TIP: Measure your own usable length with {{eq:usable-context}} at
> the accuracy floor your product needs. It takes an afternoon, the number is
> always shorter than the supported length, and it is the number your capacity
> planning should use.

## 10. Production Considerations

**Measure usable length, not supported length.**
{{eq:usable-context}} at your required floor, on your task.

**Reorder retrieved passages by {{eq:fold-ordering}}.** It costs nothing and
`position-aware-ordering` measures the gain.

**Retrieve fewer passages.** The tail contributes little relevance at the worst
positions and costs prefill on every request.

**Log tokens intended and tokens sent.** It is the first branch of
`distinguishing-context-failures` and it separates truncation from everything
else.

**Log where the evidence sat.** Without it a position failure is
indistinguishable from a model error.

**Put instructions at the end for long contexts.** {{cite:liu2023lost}}'s curve
means an instruction before 50,000 tokens of document is in a poor position;
repeating it afterwards is cheap.

**What to monitor:** context-length distribution, worst-position accuracy on a
fixed probe set, truncation rate, and retrieved-passage count. The probe set run
on a schedule is what detects the usable length changing after a model update.

## 11. Common Mistakes

**Beginners:**

*Treating supported length as usable length.* {{eq:usable-context}} — they are
different numbers.

*Reading a needle-test pass as long-context competence.* It is the easiest
possible task ({{sec:5-formal-explanation}}).

*Concatenating passages in descending relevance.* It places the second-best
passage where quality is second-highest and the mid-ranked ones where it is
worst.

**Experienced practitioners:**

*Confusing position failure with hallucination.* The mitigations are opposite —
retrieval does not help when the evidence was already present.

*Benchmarking with one needle.* {{eq:multi-fact-degradation}} shows multi-fact
tasks degrading much faster.

*Conflating position bias with extrapolation degradation.* One is attention
allocation within the trained length, the other is the position encoding beyond
it.

*Filling the window because it is available.* {{eq:long-context-cost}} — the
quadratic term is paid on everything you put there, used or not.

*Testing with unrelated filler.* Real haystacks contain near-misses on the same
topic, which compete for attention in a way filler does not
({{sec:7-internal-mechanics}}).

*Never testing absence.* A confident denial is worse than a miss, and needle
tests cannot produce one because they always insert the needle.

## 12. Failure Modes

**Lost in the middle.** Evidence present, unused. *Detection:* evidence position
logging plus a probe set. *Fix:* reorder, shorten, or retrieve.

**Silent truncation.** *Detection:* tokens intended against tokens sent.
*Fix:* {{ch:llm-prompt-lifecycle}}'s truncation policy.

**Multi-fact collapse.** Single-fact accuracy fine, multi-fact poor.
*Cause:* {{eq:multi-fact-degradation}}. *Detection:* benchmark with the number
of facts your task actually needs.

**Extrapolation degradation.** Quality falling beyond the trained length.
*Detection:* compare the curve inside and outside the trained length.

**Cost surprise.** {{eq:long-context-cost}}'s quadratic term arriving at scale.
*Detection:* the arithmetic, before launch.

**Instruction drowning.** A long context burying the instruction.
*Detection:* the model following instructions on short inputs and not long ones
— a pattern that looks like disobedience and is position.

## 13. Alternatives

{#tbl:long-context-strategies caption="Ways to get a lot of information to a model. The first is simplest and scales worst; the middle rows trade engineering for cost and reliability; the last two change what the model has to hold at once."}

| Strategy | Engineering | Cost at scale | Position risk |
|---|---|---|---|
| Stuff the full document | none | quadratic | high |
| Retrieve top-$k$ | index + retriever | low | low, if $k$ small |
| Retrieve + rerank | + a reranker | low | lowest |
| Map-reduce over chunks | orchestration | linear, many calls | none per call |
| Iterative refinement | orchestration | many calls | none per call |
| Long-context + fold ordering | trivial | quadratic | reduced |

**What genuinely differs.** Retrieval changes *what is in the context*;
map-reduce and iterative refinement change *how many contexts there are*, so no
single call is long and the position effect does not arise — at the cost of
multiple calls and a combination step that can lose cross-chunk relationships.
**The last row is the cheapest intervention in the table**: if you are going to
use long context anyway, {{eq:fold-ordering}} improves it for free.

## 14. Evaluation

**Measuring long-context capability.** A needle test is the floor, not the
ceiling. A useful evaluation includes:

1. **Position sweep** at several lengths — {{eq:u-shape}}'s depth and width.
2. **Multi-fact retrieval**, at the count your task needs
   ({{eq:multi-fact-degradation}}).
3. **Aggregation** — counting, summing, or comparing across dispersed evidence,
   which needles do not test.
4. **Absence detection** — asking for something that is *not* there. Models are
   notably poor at this and it is almost never measured.

**Reporting.** State the accuracy floor, the task, and the number of facts. A
context-length claim without those three is not comparable to any other.

## 15. Advanced Concepts

**Attention sinks.** {{maturity:EMERGING}} Early tokens absorbing attention mass
regardless of content, observed directly in attention maps and one candidate
explanation for the U's left arm.

**Context compression.** {{maturity:EMERGING}} Summarising or pruning context
before the model sees it, attacking both terms of
{{eq:long-context-cost}} and the position problem at once. Note the compressor
faces the same U it is trying to relieve, so compressing a very long context is
itself a long-context task — which bounds how much the technique can help
without a fundamentally different mechanism.

**Position interpolation and extension.** {{maturity:ESTABLISHED}} Extending a
model's context beyond its trained length by rescaling position encodings
({{ch:tf-positional}}). Cheap, effective, and it degrades in a way that compounds
with the U.

**Cache eviction guided by position.** {{maturity:EMERGING}} If the middle is
under-used, it is also the cheapest region to evict
({{ch:llm-inference}}). The interaction between eviction policy and
{{eq:u-shape}} is under-studied and potentially free capacity.

**Constant-state architectures.** {{maturity:EMERGING}} State-space models
({{ch:tf-efficient}}) have no growing cache and no quadratic term, which removes
{{eq:long-context-cost}} entirely — and they have their own recency bias, so the
position problem changes shape rather than disappearing. A fixed-size state must
forget something as the sequence grows, which is a *harder* constraint than
diluted attention: the transformer can in principle attend anywhere, and the
recurrent model has provably discarded some of it. Which failure is preferable
depends on whether your evidence is recent, and that is a property of the task
rather than of the architecture.

## 16. Connection to Previous Chapters

**Backwards.** {{ch:tf-complexity}}'s quadratic attention term is
{{eq:long-context-cost}}'s second half, and its $T > 6d$ crossover is where the
full-document option becomes untenable. {{ch:llm-inference}}'s cache growth is
the other half, and its capacity planning reached
`long-context-versus-retrieval`'s conclusion from the cost side.
{{ch:tf-positional}} governs behaviour beyond trained lengths.
{{ch:llm-hallucination}}'s extrinsic category is what a position failure
produces. {{ch:llm-prompt-lifecycle}}'s truncation is the failure it is confused
with. {{ch:tf-scaled-dot-product}}'s softmax over $T$ positions is why the effect
deepens with length.

**Forwards.** {{part:11}} builds the retrieval this chapter argues for, and
{{ch:emb-reranking}} produces the ranked list {{eq:fold-ordering}} reorders.
{{part:12}} inherits the position constraint as a design variable, and
{{ch:rag-failures}} catalogues it. {{part:23}} implements cache eviction, where
{{sec:15-advanced-concepts}}'s interaction lives.

## 17. Exercises

**Beginner**

1. What is the difference between supported and usable context length?
2. Why does a needle test overstate long-context capability?
3. Where should the best retrieved passage go, and why?

**Intermediate**

4. Using {{eq:multi-fact-degradation}}, compute accuracy for 4 facts at a
   per-fact worst-position rate of 0.85.
5. For $d = 4096$, at what context length does the attention term overtake the
   parameter term?
6. Apply {{eq:fold-ordering}} to six ranked passages and give the resulting
   order.

**Advanced**

7. Derive {{eq:position-ordering}}'s optimal assignment for a U-shaped $q$.
8. Explain why {{eq:u-shape}}'s depth grows with $T$, using the softmax over
   positions.
9. Design an evaluation distinguishing position bias from extrapolation
   degradation.

**Implementation**

10. Extend `needle-in-a-haystack` with multiple needles at controlled positions
    and measure the joint retrieval rate against
    {{eq:multi-fact-degradation}}'s prediction.
11. Implement absence detection — ask for a fact that is not present — and
    measure the false-positive rate by position.
12. Implement the full diagnostic pipeline from
    `distinguishing-context-failures` against logged production requests.
13. Measure the utility gain from {{eq:fold-ordering}} as a function of the
    relevance decay rate, and find where reordering stops mattering.

**Reasoning**

14. Your model ignores instructions on long inputs and follows them on short
    ones. Diagnose it.
15. Argue when long context beats retrieval, being specific about the volume and
    document size where the argument holds.

## 18. Interview Questions

**Beginner**

1. What is "lost in the middle"?
2. Does a 128k context window mean 128k of usable context?
3. How would you order retrieved passages?

**Intermediate**

4. Why does the position effect worsen with context length?
5. How do you distinguish a position failure from hallucination?
6. What does a needle test not measure?

**Senior**

7. Long context or retrieval for a 300-page manual? Walk through it.
8. How would you measure a model's usable context length?
9. What would you log to make context failures diagnosable?

**Systems**

10. Design a probe set that detects usable-length regression after a model
    update.
11. How would cache eviction interact with the position curve?

## 19. Research Questions

**Which mechanism produces the U?** Attention sinks, recency, and training
distribution all predict the same shape. Design an experiment that separates
them — for instance by training on documents with deliberately inverted
importance structure and seeing whether the curve follows.

**Is the position curve stable across tasks?** It is measured mostly on
retrieval. Whether aggregation and reasoning show the same shape, or a different
one, determines whether {{eq:fold-ordering}} generalises beyond retrieval.

**Can eviction exploit the U for free capacity?** If the middle is under-used, it
is the cheapest region to evict. Measure quality against eviction policy — the
result could be a large concurrency win at no quality cost, and it has not been
characterised.

**How poor is absence detection?** Models are widely reported to be bad at
noticing that something is *not* in a long context, and it is barely measured.
Sweep it by position and length; the failure mode is more dangerous than a miss
because it produces a confident denial.

## 20. Chapter Summary

**A supported context length is a claim about acceptance, not about use.**
{{cite:liu2023lost}} measured retrieval accuracy against position and found a
U-shape {{eq:u-shape}}: the beginning and end of a long context are used well
and the middle is not, with the depression deepening as the context grows —
which follows from attention being a softmax over $T$ positions, so the mass
available to any one position falls as $T$ rises.

**Usable length is therefore a different number from supported length**
{{eq:usable-context}}, it depends on the accuracy floor and the task, and it is
measurable in an afternoon and almost never measured.

**Needle tests overstate the capability substantially.** Retrieving one
distinctive string is the easiest long-context task there is, and
{{eq:multi-fact-degradation}} compounds the per-fact rate: at a worst-position
accuracy of 0.60, five facts are all retrieved 8% of the time. That gap is the
one between benchmark results and production experience.

**Position is a design variable, and exploiting it is free.**
{{eq:fold-ordering}} assigns the best passages to the two premium positions and
buries the weakest in the middle — a reordering of a list you already have.
Most systems concatenate in descending relevance, placing mid-ranked passages
exactly where the model reads worst.

**Long context is expensive in two independent ways and both are paid on
everything you put there.** {{eq:long-context-cost}}'s quadratic attention term
dominates past $T > 6d$, and cache memory grows linearly. `long-context-versus-retrieval`
shows the full-document option being both more expensive *and* less accurate
than retrieving eight passages — an unusual combination, since most engineering
choices trade one against the other.

Finally the diagnostic the chapter exists for. Position failure, silent
truncation, and extrinsic hallucination present identically — a confident wrong
answer about a document — and need three cheap log fields to separate: **tokens
intended versus sent, whether the evidence was in the context at all, and where
it sat.** Without them, one mitigation gets applied to three different problems
and works for one.

## 21. Further Reading

{{cite:liu2023lost}} is the paper and it is unusually clean: a controlled sweep
of one variable with a clear result. Read §3 for the method — the value is in
how carefully position is isolated — and note that the effect appears across
every model tested, which is what makes it a property rather than a defect.

{{cite:su2021rope}} and {{cite:press2022alibi}} from {{part:7}} govern the
*other* long-context degradation, beyond trained lengths. Reading them alongside
{{cite:liu2023lost}} is what makes the two failure modes distinguishable.

{{cite:dao2022flash}} matters here because it made long contexts affordable
enough to expose the position problem. Cheap attention is what created the
128k-window era, and the quality question arrived afterwards.

**Where to go next:** {{ch:llm-routing}} is the last chapter of {{part:10}} and
takes up the decision this one keeps implying — when to send a request to a
different, usually cheaper, model.

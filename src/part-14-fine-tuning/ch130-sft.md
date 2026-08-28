---
id: ft-sft
number: 130
part: XIV
tier: full
status: draft
requires: [ft-when, fm-instruction-tuning, dl-optimizers, tf-complexity,
           llm-inference]
provides: [sft-mechanics, token-efficiency, length-bucketing, sequence-packing,
           cross-example-contamination, truncation-as-signal, effective-batch]
citations: [zhou2023lima, ouyang2022, wang2023selfinstruct, hu2021lora,
            vaswani2017, touvron2023llama]
---

## 1. Learning Objectives

By the end of this chapter you will be able to state the SFT objective and say
which tokens carry gradient; compute **token efficiency** for naive padding,
length bucketing and sequence packing, and explain why the ordering of those
three is not the one usually assumed; identify the **cross-example
contamination** that packing introduces and the two flags that prevent it; show
that truncation is not a proportional data loss but an **active training signal**
that teaches the model to stop early; and configure a run whose effective batch
size and epoch count mean what you think they mean.

## 2. Why This Matters

{{ch:ft-when}} decided *whether*. This chapter is *how*, and it is deliberately
mechanical, because the failures here are not subtle modelling errors — they are
configuration defaults that quietly waste money or teach the wrong thing while the
loss curve looks perfect.

**Two measurements make the case.** On a realistic heavy-tailed instruction
dataset, naive padding runs at **0.285** token efficiency: **71% of the compute is
spent on padding** that is masked out of the loss and contributes nothing. That
waste is invisible — the gradients are correct, the curve is clean, and the only
symptom is a bill larger than expected, usually blamed on model size.

The fix people reach for is sequence packing. {{sec:9-practical-example}} finds
that **length bucketing beats packing on every row** — 0.998 against 0.854 — and
bucketing is a sort, needs no attention-mask changes, and has no correctness
hazard. **The sophisticated option is neither the most efficient nor the safest
here**, which is not the usual ordering.

The second measurement is about truncation, and it is the chapter's real finding.
At a 2048-token limit, **2.8% of examples are truncated and 6.7% of completion
tokens are lost** — an amplification of **2.4×**, because the truncated examples
are the long ones. And those examples are not dropped: **8.3% of the completion
tokens the model trains on come from an example that ends mid-sentence at the
token budget.**

**The model is being taught, on a measurable share of its training signal, that a
good answer stops abruptly at around max_len.** The production symptom is
generations that trail off, and the usual diagnosis — "it needs a longer context
window" — is backwards.

{{maturity:ESTABLISHED}} Everything in this chapter. It is here because the
defaults are wrong often enough to be worth a chapter.

## 3. Prerequisites

{{ch:ft-when}} for the decision that got you here; {{ch:fm-instruction-tuning}}
for the objective, loss masking and chat templates — this chapter does not repeat
them and builds on {{eq:instruction-tuning-loss}} and {{eq:loss-masking}};
{{ch:dl-optimizers}} for what a batch means to an optimiser;
{{ch:tf-complexity}} for why sequence length is expensive;
{{ch:llm-inference}} for what the trained model then has to do.

## 4. Intuitive Explanation

### What SFT actually is

Next-token prediction on (prompt, completion) pairs, with the loss computed on the
completion tokens only. That is the whole algorithm —
{{ch:fm-instruction-tuning}} derived it, and nothing in this chapter changes it.

**Everything difficult is logistics**: examples have different lengths, GPUs want
rectangles, and the reconciliation is where the money and the silent failures
live.

### Three ways to make rectangles

```text
   NAIVE PADDING     shuffle, batch, pad each batch to its longest member
                     -> one outlier sets the cost for everything beside it

   BUCKETING         sort by length, then batch
                     -> neighbours are similar, so the pad is tiny

   PACKING           concatenate examples to fill a fixed block
                     -> no padding at all, but the block's tail is wasted
                        and examples now share a sequence
```

The standard telling ranks these in that order, with packing as the sophisticated
answer. {{sec:9-practical-example}} measures **bucketing winning**, because
packing's waste has not disappeared — it has moved from the end of each *batch* to
the end of each *block*. With a 2048-token block and a few-hundred-token median
example, that tail is a meaningful fraction.

**Sorting by length is one line and recovers almost all of the loss.** It is
routinely skipped in favour of the more elaborate option.

### The hazard packing introduces

When several examples share one sequence, a token can attend to tokens from a
*different* example unless you stop it. {{sec:9-practical-example}} measures
**64.8% of attend-able pairs crossing an example boundary** on a realistic length
distribution.

That is a large leak in a specifically unhelpful direction. **The model is trained
to produce a completion while conditioning on an unrelated example that happens to
precede it** — teaching that whatever came before the current instruction is
relevant to it. At inference the symptom is a model dragging irrelevant material
across turn boundaries.

The fix is two flags: a **block-diagonal attention mask** so each packed example
attends only to itself, and **position IDs that reset** per example. Both are
supported everywhere and both are easy to leave unset.

### Truncation teaches something false

Here is the failure worth carrying out of this chapter.

Every run sets a maximum sequence length; every dataset has examples longer than
it. The mental model is "we lose a few per cent of examples". Two things make that
wrong:

> **The cut always lands on the completion.** The prompt comes first, so
> truncation removes the tail of the *answer* — the part the loss is computed on.
> And long examples have long answers, so the loss amplifies: 2.8% of examples
> becomes 6.7% of completion tokens.
>
> **The truncated example is not dropped — it is trained on.** Its last token,
> the one the model learns to follow with an end-of-sequence, sits wherever the
> budget ran out. Mid-sentence.

So the model receives repeated, consistent evidence that a good answer ends at
around max_len. **That is not missing data; it is wrong data**, and it is being
optimised for.

**And the fix is free.** *Drop* the examples you cannot fit instead of truncating
them. You lose exactly the same completion tokens — they were never going to be
trained on — and you remove every fake stopping point. Or split long examples at
natural boundaries so each piece ends where something actually ended.

Truncation is the default in most tooling and is the only one of the three
options that actively teaches something untrue.

## 5. Formal Explanation

### 5.1 The objective

For a dataset of pairs $(x, y)$ with $y$ the completion:

$$ \mathcal{L}_{\text{SFT}} = -\sum_{(x,y)} \sum_{t=1}^{|y|} \log p_\theta\big(y_t \mid x,\, y_{<t}\big) $$ (eq:sft-objective)

which is {{eq:instruction-tuning-loss}} with the masking of
{{eq:loss-masking}} applied. **The only tokens carrying gradient are completion
tokens**, which is the fact every measurement in this chapter is denominated in.

### 5.2 Token efficiency

$$ \eta = \frac{\sum_i \ell_i}{\text{tokens actually processed}} $$ (eq:token-efficiency)

For naive padding with batch size $B$ and lengths $\ell$:

$$ \eta_{\text{pad}} = \frac{\mathbb{E}[\ell]}{\mathbb{E}\big[\max_{j \in \text{batch}} \ell_j\big]} $$ (eq:padding-efficiency)

**The denominator is an expected maximum**, which for a heavy-tailed distribution
grows with $B$ and sits far above the mean. That single fact is the 71% waste.

For bucketing, the batch's members are order statistics that are adjacent in the
sorted list, so $\max_j \ell_j \approx \mathbb{E}[\ell \mid \text{bucket}]$ and
$\eta \to 1$.

For packing with block $L$:

$$ \eta_{\text{pack}} = 1 - \frac{\mathbb{E}[\text{block remainder}]}{L} \approx 1 - \frac{\mathbb{E}[\ell]}{2L} $$ (eq:packing-efficiency)

since the wasted tail is on average about half an example. **{{eq:packing-efficiency}}
is why bucketing wins**: its waste vanishes with sorting, while packing's is a
fixed fraction set by $\mathbb{E}[\ell]/L$.

### 5.3 Cross-example contamination

In a packed block containing examples of lengths $b_1, \dots, b_m$ summing to $n$,
causal attention admits $\binom{n}{2}$ ordered pairs, of which
$\sum_i \binom{b_i}{2}$ are within an example. So the cross-example share is

$$ \gamma = 1 - \frac{\sum_i b_i(b_i - 1)}{n(n-1)} \;\approx\; 1 - \frac{\sum_i b_i^2}{n^2} $$ (eq:cross-contamination)

**$\gamma$ is large whenever the block holds many small examples**, because
$\sum b_i^2 \ll (\sum b_i)^2$ for many similar terms. Measured: 64.8%.

Note the direction of the pathology: this is not noise, it is *systematically*
teaching that preceding unrelated text is relevant, which is the opposite of what
{{ch:llm-long-context}} wants a model to learn.

### 5.4 Truncation, formally

For an example with prompt length $p$ and completion length $c$ under a limit $L$:

$$ c_{\text{kept}} = \min\big(c,\; \max(0,\, L - p)\big), \qquad c_{\text{lost}} = c - c_{\text{kept}} $$ (eq:truncation-hits-completions)

The share of *examples* affected is $\Prob[p + c > L]$; the share of *completion
tokens* lost is $\mathbb{E}[c_{\text{lost}}]/\mathbb{E}[c]$. These differ by

$$ A = \frac{\text{token loss}}{\text{example loss}} = \frac{\mathbb{E}[c_{\text{lost}}]}{\mathbb{E}[c]\,\Prob[\text{truncated}]} > 1 $$ (eq:truncation-amplification)

**$A > 1$ whenever completion length correlates with total length**, which it
always does. Measured: 2.39× at $L = 2048$, rising to 3.72× at 4096.

### 5.5 Truncation as a training signal

The damaging part is not {{eq:truncation-hits-completions}} but what remains. A
truncated example contributes $c_{\text{kept}}$ tokens of gradient, and its final
position is followed by an end-of-sequence that is *false*. The share of trained
completion tokens sitting in such an example:

$$ \phi = \frac{\sum_{\text{truncated}} c_{\text{kept}}}{\sum_{\text{all}} c_{\text{kept}}} $$ (eq:truncation-teaches-stopping)

Measured at $L = 2048$: $\phi = 8.3\%$, which is *larger* than the 2.8% of
examples truncated, because truncated examples contribute the maximum possible
number of tokens each.

$$ \phi > \Prob[\text{truncated}] \quad \text{always} $$ (eq:phi-exceeds-example-share)

**Dropping instead of truncating sets $\phi = 0$ at zero cost in trained tokens**,
since the lost completion tokens were lost either way.

### 5.6 Effective batch size

$$ B_{\text{eff}} = B_{\text{device}} \times n_{\text{accum}} \times n_{\text{devices}} $$ (eq:effective-batch)

which matters because learning rate should scale with $B_{\text{eff}}$, not with
the per-device batch. And under *packing*, $B_{\text{eff}}$ measured in
**examples** is no longer fixed — a block holds a variable number — so

$$ \text{examples per step} = \frac{B_{\text{eff}} \cdot L}{\mathbb{E}[\ell]} \quad \text{(variable)} $$ (eq:packed-examples-per-step)

**An "epoch" under packing is therefore not a fixed number of steps**, and two
runs with the same step count can see different amounts of data.

## 6. Mathematical Foundation

### 6.1 Why the expected maximum is so damaging

For $B$ samples from a lognormal with parameters $(\mu, \sigma)$, the expected
maximum grows roughly as

$$ \mathbb{E}\big[\max_{1..B} \ell\big] \approx e^{\mu + \sigma\sqrt{2\ln B}} $$ (eq:expected-max)

against a mean of $e^{\mu + \sigma^2/2}$. At $\sigma = 0.9$, $B = 16$:

$$ \frac{\mathbb{E}[\max]}{\mathbb{E}[\ell]} \approx e^{0.9(2.35) - 0.405} = e^{1.71} \approx 5.5 $$

predicting $\eta \approx 0.18$ against a measured **0.285** — the same order, with
the gap because the measurement clips lengths at the block size, which truncates
exactly the outliers {{eq:expected-max}} is about.

**The dependence on $B$ is the actionable part.** Efficiency *falls* as batch size
rises under naive padding, which is the opposite of the usual intuition that
bigger batches are more efficient.

### 6.2 Bucketing against packing, decided

From {{eq:packing-efficiency}}, packing's ceiling at $\mathbb{E}[\ell] = 402$,
$L = 2048$:

$$ \eta_{\text{pack}} \approx 1 - \frac{402}{2 \times 2048} = 0.902 $$

against a measured 0.854 (the difference being that the greedy fill leaves more
than half an example on average when lengths are variable). Bucketing's ceiling is
1 minus a within-bucket spread that vanishes with dataset size.

$$ \eta_{\text{bucket}} > \eta_{\text{pack}} \iff L < \frac{\mathbb{E}[\ell]}{2\,(1 - \eta_{\text{bucket}})} $$ (eq:bucket-beats-pack)

**Packing only overtakes bucketing when the block is very large relative to the
examples** — which is exactly the long-context regime where packing is genuinely
right, and not the instruction-tuning regime where it is usually recommended.

> **MATH NOTE:** This comparison is on *token efficiency* alone, which is not the
> only reason to pack. Packing produces fixed-shape blocks, which matters for a
> compiled graph or a static-shape accelerator, and it removes the batch-size
> variance that bucketing introduces. Those are real benefits and this listing
> does not measure them. **The claim here is narrower and still useful: if you are
> packing for efficiency, measure first, because bucketing may already have it.**

### 6.3 The truncation amplification, worked

With completion length proportional to total length, $c \approx (1-\pi)\ell$ for
prompt fraction $\pi$. The truncated set is $\{\ell > L\}$, and for a lognormal
tail the conditional mean of $\ell$ given $\ell > L$ substantially exceeds
$\mathbb{E}[\ell]$:

$$ A = \frac{\mathbb{E}[\ell - L \mid \ell > L]}{\mathbb{E}[\ell]} $$ (eq:amplification-worked)

At $L = 2048$ with a median of 402, the conditional excess is several times the
mean — measured 2.39×. **And $A$ *rises* as $L$ rises** (2.39 → 3.72 from 2048 to
4096), because the surviving truncated examples are drawn from ever further out in
the tail.

That is counter-intuitive and worth stating: **raising max_len reduces the total
loss while increasing the amplification factor.** The examples still being
truncated are increasingly extreme, so each one loses proportionally more.

## 7. Internal Mechanics

```mermaid {#fig:sft-batching caption="Three ways to turn variable-length examples into rectangles, with what each wastes and what each risks. Bucketing's waste vanishes with sorting; packing's is a fixed fraction of the block (eq:packing-efficiency) and it introduces the cross-example attention path that eq:cross-contamination measures. The truncation branch is separate and is the one that teaches something false."}
flowchart TB
    D["examples of<br/>variable length"] --> T{"longer than<br/>max_len?"}
    T -->|"yes: truncate"| BAD["trained on, ending<br/>mid-sentence<br/>(eq:truncation-teaches-stopping)"]
    T -->|"yes: DROP or SPLIT"| OK["no false stopping points"]
    T -->|"no"| B{"batching strategy"}
    B -->|"naive pad"| P1["pad to batch max<br/>eta = 0.285"]
    B -->|"sort, then batch"| P2["pad to bucket max<br/>eta = 0.998"]
    B -->|"pack into blocks"| P3["fill block<br/>eta = 0.854"]
    P3 -.->|"needs block-diagonal mask<br/>+ position ID reset"| P3
```

### 7.1 A configuration checklist that is mostly about defaults

| Setting | Default is often | Should be |
|---|---|---|
| loss on prompt tokens | on | **off** ({{eq:loss-masking}}) |
| over-length examples | truncate | **drop or split** |
| batching | shuffle + pad | **sort into buckets** |
| packing attention mask | full causal | **block-diagonal** |
| packing position IDs | continuous | **reset per example** |
| learning rate | inherited | **scaled to $B_{\text{eff}}$** |

**Five of those six are silent when wrong.** Only the learning rate announces
itself.

### 7.2 How many epochs

Fewer than instinct suggests. {{cite:zhou2023lima}}'s result — a strong model from
1,000 examples — is also a statement about epochs: with high-quality data the
model reaches the format quickly, and further passes memorise rather than
generalise ({{ch:ft-when}}'s {{eq:memorise-not-generalise}}).

Two to three epochs is the usual range, and the diagnostic is
{{ch:ft-when}}'s: **when held-out performance flattens while training loss keeps
falling, additional epochs are buying memorisation.** {{ch:ft-training-config}}
treats this properly.

### 7.3 What to log

- **Token efficiency**, per step. It is one division and it catches the 71% waste
  immediately.
- **Truncation rate and $\phi$** ({{eq:truncation-teaches-stopping}}), once, at
  dataset build time.
- **Completion tokens per step**, not examples per step — under packing they are
  different numbers ({{eq:packed-examples-per-step}}).
- **A generation sample every N steps.** The loss curve cannot show you that the
  model has started stopping early; a sample can.

### 7.4 Why these particular defaults are wrong

It is worth asking why a toolchain would ship settings that waste most of a run
or teach a false stopping point, because the answer says something about where to
look for the next one.

None of these defaults are careless. Truncation is the correct default for
*pretraining*, where the corpus is a continuous stream and a document boundary is
arbitrary — cutting at the budget loses nothing meaningful because nothing was
supposed to end there. Naive shuffled padding is the correct default when
examples are uniform in length, which they are in most classification datasets.
Full causal attention is the correct default when a sequence is one example.

**Each default was right for the setting it came from and wrong for this one**,
and the transfer happened silently because the API did not change. That is the
general shape to watch for whenever a tool is used for something adjacent to its
origin: the defaults encode assumptions about the original setting, and they do
not announce themselves when those assumptions stop holding.

The practical version is a habit rather than a checklist. **For every default you
inherit, ask what workload it was chosen for.** Where that workload differs from
yours in a way the setting depends on, measure rather than assume — which is what
this chapter's two listings are, applied to three settings each.

## 8. Implementation

```python {tier=A name=padding-bucketing-packing}
"""Padding, bucketing and packing: where a fine-tuning budget actually goes.

Training examples have wildly different lengths and a GPU wants rectangles, so
something has to reconcile them. The three standard answers differ by a large
factor in how much of the compute is spent on real tokens
(eq:token-efficiency), and the difference is invisible in the loss curve --
padding tokens are masked out, so they cost money and produce nothing while the
run looks perfectly healthy.

Packing is the efficient answer and it introduces a hazard the other two do not
have: several examples share one sequence, so without a block-diagonal attention
mask, tokens attend ACROSS example boundaries and the model conditions on text
from an unrelated sample (eq:cross-contamination). This listing measures both the
saving and the exposure.
"""
import numpy as np

rng = np.random.default_rng(137)

N_EXAMPLE = 20000
BATCH = 16
BLOCK = 2048               # packed block length, and the max sequence length


def lengths(kind):
    """Realistic instruction-tuning length distributions are heavy-tailed: most
    examples are short and a few are very long."""
    if kind == "uniform-ish":
        L = rng.integers(200, 600, size=N_EXAMPLE).astype(float)
    elif kind == "heavy-tailed":
        L = rng.lognormal(mean=5.4, sigma=0.9, size=N_EXAMPLE)
    else:                                    # bimodal: short chats + long docs
        short = rng.lognormal(mean=4.8, sigma=0.4, size=N_EXAMPLE)
        long_ = rng.lognormal(mean=6.9, sigma=0.5, size=N_EXAMPLE)
        pick = rng.random(N_EXAMPLE) < 0.75
        L = np.where(pick, short, long_)
    return np.clip(L, 24, BLOCK).astype(int)


def naive_padding(L):
    """Shuffle, batch, pad each batch to its own longest member."""
    order = rng.permutation(len(L))
    used = total = 0
    for s in range(0, len(L), BATCH):
        b = L[order[s:s + BATCH]]
        used += b.sum()
        total += len(b) * b.max()
    return used / total


def bucketed(L):
    """Sort by length, then batch -- neighbours have similar lengths so the pad
    to the batch maximum is small."""
    srt = np.sort(L)
    used = total = 0
    for s in range(0, len(srt), BATCH):
        b = srt[s:s + BATCH]
        used += b.sum()
        total += len(b) * b.max()
    return used / total


def packed(L):
    """Concatenate examples into fixed BLOCK-length sequences, starting a new
    block only when the next example does not fit (eq:packing-efficiency)."""
    used = total = 0
    cur = 0
    for x in L:
        if cur + x > BLOCK:
            total += BLOCK
            used += cur
            cur = 0
        cur += x
    total += BLOCK
    used += cur
    return used / total


def contamination(L):
    """In a packed block WITHOUT a block-diagonal mask, every token may attend to
    every earlier token in the block. Report the share of attend-able pairs that
    cross an example boundary -- i.e. the share of attention capacity pointed at
    an unrelated example (eq:cross-contamination)."""
    blocks, cur = [], []
    tot = 0
    for x in L[:6000]:
        if tot + x > BLOCK:
            blocks.append(cur); cur = []; tot = 0
        cur.append(x); tot += x
    if cur:
        blocks.append(cur)
    cross = within = 0
    for b in blocks:
        b = np.asarray(b)
        n = b.sum()
        # Causal pairs inside the block, and those inside each example.
        cross += n * (n - 1) / 2
        within += (b * (b - 1) / 2).sum()
    return float((cross - within) / cross) if cross else 0.0


print(f"{N_EXAMPLE:,} examples, batch {BATCH}, block/max length {BLOCK}\n")
print(f"{'length profile':<18}{'median':>8}{'p99':>8}{'':>3}"
      f"{'naive pad':>12}{'bucketed':>11}{'packed':>9}{'':>3}"
      f"{'cross-example':>15}")
print("-" * 88)

res = {}
for kind in ("uniform-ish", "heavy-tailed", "bimodal"):
    L = lengths(kind)
    n, b, p = naive_padding(L), bucketed(L), packed(L)
    c = contamination(L)
    res[kind] = (n, b, p, c)
    print(f"{kind:<18}{int(np.median(L)):>8}{int(np.percentile(L, 99)):>8}{'':>3}"
          f"{n:>12.3f}{b:>11.3f}{p:>9.3f}{'':>3}{c:>15.1%}")

ht = res["heavy-tailed"]
bm = res["bimodal"]
print(f"""
The three efficiency columns are the fraction of processed tokens that are real
rather than padding. On the heavy-tailed profile -- which is what an instruction
dataset actually looks like, mostly short with a long tail -- naive padding runs
at {ht[0]:.3f}. Around {1 - ht[0]:.0%} of the compute is spent on padding tokens
that are masked out of the loss and contribute nothing.

That waste is invisible. Padding is masked, so the loss curve is clean, the
gradients are correct, and the run looks healthy while most of the bill buys
nothing. It surfaces only as "training is slower and costs more than expected",
which is usually blamed on the model size.

The mechanism is that a batch is padded to its OWN longest member, so one outlier
sets the cost for the fifteen examples beside it (eq:token-efficiency). With a
heavy-tailed distribution most batches contain such an outlier -- which is what
heavy-tailed means. The bimodal row is worse still at {bm[0]:.3f}, because a
mixture of short chats and long documents guarantees the outlier.

Now the result this listing was not built to find. Bucketing -- sort by length,
then batch -- reaches {ht[1]:.3f}, and packing reaches {ht[2]:.3f}. BUCKETING
WINS, on every row, and it wins by a clear margin.

The reason is that packing's waste has simply moved. A block is filled until the
next example does not fit, and the remainder is discarded, so packing pays a
partial block at the end of every block rather than a partial batch at the end of
every batch. With a block of {BLOCK} and a median example of a few hundred
tokens, that tail is a meaningful fraction. Bucketing has no such tail: after
sorting, the pad to the batch maximum is nearly zero because the batch's members
are nearly the same length.

So the usual ordering of these techniques is wrong, at least on token efficiency.
Sorting by length is one line, needs no attention-mask changes, and recovers
almost all of the loss. It is the first thing to try and it is routinely skipped
in favour of the more sophisticated option.

Bucketing does cost something the table does not show: batches are no longer
randomly composed, so examples within a batch are correlated by length, which
correlates them by type too. The standard mitigation is to shuffle the ORDER of
buckets while keeping their contents, which restores randomness across steps
while keeping it out of each step.

And packing carries a hazard bucketing does not, which is the last column. In a
packed block several unrelated examples share one sequence, and unless the
attention mask is block-diagonal, every token may attend to every earlier token
in the block -- including tokens belonging to somebody else's example. The
measured share of attend-able pairs that cross a boundary is {ht[3]:.0%} on the
heavy-tailed profile.

That is a large leak in an unhelpful direction (eq:cross-contamination). The model
is trained to produce a completion while conditioning on an unrelated example
that happens to precede it, which teaches precisely the wrong lesson: that
whatever came before the current instruction is relevant to it. The symptom at
inference is a model that drags irrelevant material across turn boundaries.

The fix is mechanical -- a block-diagonal attention mask so each packed example
attends only to itself, and position IDs that reset per example rather than
running across the block. Every serious training stack supports both, and both
are the kind of flag that is easy to leave unset.

Which gives a clear recommendation. Bucket first: it is simpler, more efficient
here, and has no correctness hazard. Reach for packing when you need fixed-shape
blocks for a compiled graph, or when sequence lengths approach the block size so
the tail waste disappears -- and if you do, verify the mask before the run rather
than after.""")
```

The first listing is about wasted compute. The second is about a default that
does something worse than waste.

```python {tier=A name=truncation-as-signal}
"""Truncation is not a small loss of data. It is a training signal.

Every fine-tuning run sets a maximum sequence length, and every dataset has
examples longer than it. The usual mental model is that a few per cent of
examples get shortened and the effect is proportional and small.

It is neither, for two reasons this listing measures.

First, truncation is not uniform over the example. The prompt comes first, so the
cut always lands on the COMPLETION -- the part being trained on -- and long
examples are exactly the ones with long completions
(eq:truncation-hits-completions).

Second, and worse: a truncated example is not dropped. It is trained on, with an
end-of-sequence position that is not where the answer ended. The model is being
shown, repeatedly, that a good answer stops mid-sentence at exactly the token
budget (eq:truncation-teaches-stopping).
"""
import numpy as np

rng = np.random.default_rng(139)

N = 40000
PROMPT_FRAC = 0.35            # of an example's tokens, roughly, are the prompt


def dataset():
    """Heavy-tailed total lengths; completion length correlates with total, as it
    does in practice -- long questions get long answers."""
    total = np.clip(rng.lognormal(mean=6.0, sigma=0.85, size=N), 48, 32000)
    prompt = total * (PROMPT_FRAC + 0.10 * rng.normal(size=N)).clip(0.15, 0.6)
    completion = total - prompt
    return total.astype(int), prompt.astype(int), completion.astype(int)


TOTAL, PROMPT, COMP = dataset()

print(f"{N:,} examples. median length {int(np.median(TOTAL))}, "
      f"p95 {int(np.percentile(TOTAL, 95))}, p99 {int(np.percentile(TOTAL, 99))}\n")
print(f"{'max_len':>9}{'% examples':>14}{'% completion':>16}"
      f"{'trained tokens from':>22}")
print(f"{'':>9}{'truncated':>14}{'tokens lost':>16}"
      f"{'a truncated example':>22}")
print("-" * 62)

rows = {}
for L in (512, 1024, 2048, 4096, 8192):
    trunc = TOTAL > L
    # Tokens of completion that survive: whatever is left after the prompt.
    kept_comp = np.clip(L - PROMPT, 0, COMP)
    lost_comp = COMP - kept_comp
    ex_frac = float(trunc.mean())
    tok_frac = float(lost_comp.sum() / COMP.sum())
    # Of the completion tokens the model DOES train on, what share come from a
    # truncated example -- i.e. end at a fake stopping point?
    fake_stop = float(kept_comp[trunc].sum() / kept_comp.sum())
    rows[L] = (ex_frac, tok_frac, tok_frac, fake_stop)
    print(f"{L:>9}{ex_frac:>14.1%}{tok_frac:>16.1%}{fake_stop:>22.1%}")

print(f"\n{'max_len':>9}{'amplification: lost tokens / truncated examples':>50}")
print("-" * 60)
for L in (512, 1024, 2048, 4096, 8192):
    e, t, _, _ = rows[L]
    print(f"{L:>9}{(t / e if e else 0):>50.2f}x")

a2, a8 = rows[2048], rows[8192]
print(f"""
Read the first two columns together at max_len 2048: {a2[0]:.1%} of examples are
truncated, and {a2[1]:.1%} of completion tokens are lost. Those are not the same
number, and the ratio in the second table is why -- the examples that get
truncated are the LONG ones, so each truncated example loses far more than an
average example contains (eq:truncation-hits-completions).

The amplification factor makes this concrete. At every budget, the share of
training signal lost is several times the share of examples affected. "We only
truncate a few per cent of examples" is a true statement that describes a much
larger loss than it sounds like, and it is the sentence that usually ends the
conversation.

Note that the amplification RISES with max_len, from 2.39x to 3.72x. That is
counter-intuitive and worth pausing on: raising the limit reduces the total loss
while making each remaining truncation worse, because the examples still being
cut are drawn from ever further out in the tail.

The direction of the loss compounds it. Truncation cuts from the END, and the
prompt is at the start, so the tokens removed are always completion tokens -- the
ones the loss is computed on. A truncated example does not lose 30% of itself
evenly; it keeps its entire prompt and loses the tail of its answer.

Now the last column, which is the part that is not a data-loss problem at all.
Those truncated examples are not discarded. They are trained on, and their final
token -- the one the model learns to follow with an end of sequence -- sits
wherever the budget ran out, mid-sentence. At max_len 2048, {a2[3]:.1%} of the
completion tokens the model trains on come from an example that ends at a fake
stopping point (eq:truncation-teaches-stopping).

The model is not merely missing those answers. It is being taught, on a
meaningful share of its training signal, that a good answer stops abruptly at
around the token budget. The symptom in production is a model that trails off on
long generations, and the diagnosis usually offered is "it needs a longer context
window" -- which is exactly backwards, because the behaviour was installed by
training rather than limited by capacity.

Raising max_len to 8192 improves every column ({a8[0]:.1%} of examples,
{a8[1]:.1%} of tokens, {a8[3]:.1%} fake stops) and costs quadratically in
attention. So the budget is a real constraint and the question is what to do
within it, which has a better answer than picking a number.

DROP the examples you cannot fit, rather than truncating them. Dropping loses
exactly the same completion tokens -- they were never going to be trained on --
and it removes the fake stopping points entirely. The cost is a slightly smaller,
slightly shorter-skewed dataset; the benefit is that nothing in the training data
is a lie about where answers end.

If long examples matter to your task, the answer is not truncation either. Split
them into multiple training examples at natural boundaries, so each one ends where
something actually ended. Both options are cheap. Truncation is the default in
most tooling, and it is the only one of the three that actively teaches something
false.""")
```

## 9. Practical Example

**Most of a naive fine-tuning run is padding.** On a heavy-tailed instruction
dataset, naive padding achieves **0.285** token efficiency — **71% of the compute
spent on masked-out tokens.** {{eq:expected-max}} explains it: a batch is padded to
its own longest member, and for a heavy-tailed distribution the expected maximum
sits far above the mean.

**And efficiency *falls* as batch size rises** under naive padding, which is the
opposite of the usual intuition.

**Bucketing beats packing, on every row.** 0.998 against 0.854, and the reason is
{{eq:packing-efficiency}}: packing's waste has moved rather than vanished, from
the end of each batch to the end of each block. **Sorting by length is one line,
needs no attention-mask changes, and recovers almost everything.**

> **IMPORTANT:** This is a narrower claim than "do not pack". It is about token
> efficiency only, and packing has real benefits this listing does not measure —
> fixed-shape blocks for a compiled graph, and no batch-size variance.
> {{eq:bucket-beats-pack}} says packing overtakes when the block is large relative
> to the examples, **which is the long-context regime where it is genuinely right
> and not the instruction-tuning regime where it is usually recommended.** If you
> are packing for efficiency, measure first.

**Packing also carries a hazard bucketing does not**: **64.8%** of attend-able
pairs cross an example boundary ({{eq:cross-contamination}}). That trains the model
to condition a completion on an unrelated preceding example — the opposite of what
you want — and the fix is two flags that are easy to leave unset.

**Truncation is the chapter's real finding.** At max_len 2048, **2.8% of examples
are truncated** and **6.7% of completion tokens are lost** — an amplification of
**2.39×** ({{eq:truncation-amplification}}), because the truncated examples are the
long ones.

**And the amplification rises with max_len**, 2.39× → 3.72× from 2048 to 4096.
Raising the limit reduces the total loss while making each remaining truncation
worse, since the survivors come from further out in the tail.

**But the data loss is not the damage.** **8.3% of the completion tokens the model
trains on come from an example that ends mid-sentence at the budget**
({{eq:truncation-teaches-stopping}}) — a share *larger* than the fraction of
examples truncated, because each truncated example contributes the maximum
possible tokens.

**The model is receiving repeated, consistent evidence that a good answer stops at
around max_len.** In production that appears as generations trailing off, and the
usual diagnosis — "it needs a longer context window" — is backwards: the behaviour
was installed by training, not limited by capacity.

**And the fix costs nothing.** Drop over-length examples instead of truncating
them: the same completion tokens are lost either way, and $\phi$ goes to zero. Or
split at natural boundaries so each piece ends where something ended. **Truncation
is the tooling default and the only one of the three that teaches something
untrue.**

## 10. Production Considerations

**Log token efficiency per step.** One division, and it catches a 71% waste on the
first run.

**Sort into length buckets before batching**, and shuffle bucket order across
steps to keep randomness where it belongs.

**If you pack, verify the block-diagonal mask and position-ID reset** before the
run, not after. Both are silent when wrong.

**Drop or split over-length examples. Never truncate.**

**Measure the truncation rate and $\phi$ at dataset build time**
({{eq:truncation-teaches-stopping}}), and record them next to the dataset.

**Scale the learning rate to $B_{\text{eff}}$**
({{eq:effective-batch}}), not to the per-device batch.

**Log completion tokens per step, not examples per step**, under packing —
{{eq:packed-examples-per-step}} makes an "epoch" a variable quantity.

**Sample generations during training.** The loss curve cannot show you that the
model has begun stopping early; a sample can.

## 11. Common Mistakes

**Truncating over-length examples.** The chapter's headline.

**Packing without the block-diagonal mask.**

**Reaching for packing before trying bucketing.**

**Computing loss on prompt tokens** ({{eq:loss-masking}}).

**Reporting "we only truncate 3% of examples"** as though it described the loss.

**Raising max_len and assuming the truncation problem is proportionally solved** —
the amplification rises.

**Treating an epoch as a fixed number of steps under packing.**

## 12. Failure Modes

**Early stopping in generations.** Symptom: answers trail off, often near a round
token count. Cause: {{eq:truncation-teaches-stopping}}. Diagnose by histogramming
generation lengths against max_len.

**Cross-turn bleed.** Symptom: the model references content from an unrelated
earlier exchange. Cause: packing without a block-diagonal mask.

**Unexplained training cost.** Symptom: the run costs far more than the token
count suggests. Cause: {{eq:padding-efficiency}}.

**Correlated batches.** Symptom: loss oscillates with a period matching the bucket
structure. Cause: bucketing without shuffling bucket order.

**Learning-rate mismatch after a scaling change.** Symptom: a run that worked at
one device count diverges at another. Cause: {{eq:effective-batch}}.

**Silent prompt-token training.** Symptom: the model completes prompts rather than
answering them. Cause: masking off.

## 13. Alternatives

| Instead of | Consider | When |
|---|---|---|
| naive padding | length bucketing | always — it is a sort |
| packing | bucketing | instruction-length data |
| bucketing | packing | long-context data, or fixed-shape requirements |
| truncation | dropping | always |
| truncation | splitting at boundaries | when long examples matter |
| full fine-tuning | LoRA ({{ch:ft-lora}}) | cost, and forgetting |

**The last row is the next chapter**, and it changes the memory arithmetic of
everything above without changing any of the data mechanics.

## 14. Evaluation

**Report token efficiency** alongside training cost, or the cost number is
uninterpretable.

**Report the truncation rate and $\phi$** with the dataset, not the run.

**Histogram generation lengths** against max_len on the evaluation set. A spike
near the limit is {{eq:truncation-teaches-stopping}} showing up.

**Evaluate on long-completion examples specifically** — they are the ones the
truncation policy affected.

**Compare against the base model on general tasks**, not only on the target — the
subject of {{ch:ft-training-config}}.

## 15. Advanced Concepts

**Curriculum by length.** {{maturity:EMERGING}} Bucketing already orders examples
by length; using that order deliberately — short first — is a curriculum, and
whether it helps is task-dependent and rarely measured.

**Loss weighting by example.** {{maturity:MATURE}} Not all training examples
deserve equal weight, and length is a confound: a long completion contributes more
tokens and therefore more gradient. Normalising loss per *example* rather than per
*token* changes what the run optimises, and neither choice is obviously right.

**Packing as a long-context tool.** {{maturity:MATURE}}
{{eq:bucket-beats-pack}} says packing wins when $L \gg \mathbb{E}[\ell]$, which is
precisely the continued-pretraining and long-context regime. **The technique is
right; the usual justification for it is not.**

**Truncation policy as a data decision.** {{maturity:ESTABLISHED}}
{{eq:truncation-teaches-stopping}} makes max_len a *dataset* parameter rather than
a memory parameter — it changes what the model is taught, not merely how much.
Treating it as a memory knob is the root of the failure.

**Sequence-length scaling laws.** {{maturity:EMERGING}} The right max_len is a
function of the completion-length distribution and the attention cost, and there
is no established formula. In practice it is chosen by memory and then
rationalised, which {{eq:truncation-amplification}} suggests is backwards.

## 16. Connection to Previous Chapters

{{ch:fm-instruction-tuning}}'s {{eq:instruction-tuning-loss}} and
{{eq:loss-masking}} are this chapter's objective, unchanged — everything here is
logistics around them. {{ch:ft-when}}'s {{eq:memorise-not-generalise}} is the
epoch-count diagnostic, and its churn argument is what makes wasted training cost
matter beyond one run. {{ch:tf-complexity}}'s quadratic attention is why max_len
is contested at all, and {{ch:llm-long-context}} is what
{{eq:cross-contamination}} teaches the model to do wrong. Forward:
{{ch:ft-lora}} changes the memory arithmetic without changing any of this;
{{ch:ft-datasets}} decides what goes into the pairs; and
{{ch:ft-training-config}} handles the hyperparameters this chapter deliberately
deferred.

## 17. Exercises

1. Derive {{eq:padding-efficiency}} and explain why efficiency falls as batch size
   rises.
2. Use {{eq:expected-max}} to predict token efficiency at batch 64 on the
   heavy-tailed profile, then check it by changing `BATCH` in the listing.
3. Derive {{eq:packing-efficiency}} and use {{eq:bucket-beats-pack}} to find the
   block size at which packing overtakes bucketing for a 402-token median.
4. In `padding-bucketing-packing`, set `BLOCK = 16384`. Does packing now win, and
   does the crossover match your answer to exercise 3?
5. Derive {{eq:cross-contamination}} and compute $\gamma$ for a block holding four
   equal-length examples.
6. Prove {{eq:phi-exceeds-example-share}}.
7. In `truncation-as-signal`, add a policy that drops over-length examples and
   report the resulting dataset size and $\phi$. What did dropping cost?
8. Take a dataset you fine-tune on. Compute its length distribution, token
   efficiency under your current batching, truncation rate, and $\phi$. Which of
   the four surprises you?

## 18. Interview Questions

1. Which tokens carry gradient in SFT?
2. What fraction of a naive fine-tuning run is padding, and why?
3. Rank padding, bucketing and packing by token efficiency. Justify.
4. What does packing risk, and how do you prevent it?
5. Why is truncation worse than dropping over-length examples?
6. Your fine-tuned model's answers trail off. Diagnose.
7. Why does truncation amplification rise as max_len rises?
8. What is the effective batch size, and what does it govern?
9. Why is an "epoch" ambiguous under packing?
10. Which SFT misconfigurations are silent?

## 19. Research Questions

1. {{eq:bucket-beats-pack}} compares on token efficiency only. What is the right
   comparison once compiled-graph benefits and batch-variance costs are included?
2. Bucketing correlates examples within a batch. How much does that cost in
   optimisation quality, and does bucket-order shuffling fully recover it?
3. {{eq:truncation-teaches-stopping}} predicts a generation-length artefact. How
   large is it in a real fine-tune, and does it persist after further training on
   untruncated data?
4. Loss normalised per token versus per example changes what is optimised. Which
   matches human judgement of answer quality better?
5. Is there a principled way to choose max_len from the completion-length
   distribution and the attention budget, rather than from memory?

## 20. Chapter Summary

SFT is next-token prediction on completions ({{eq:sft-objective}}). **Everything
difficult is logistics**, and the failures are configuration defaults that waste
money or teach the wrong thing while the loss curve looks perfect.

**Naive padding runs at 0.285 token efficiency** — 71% of compute on masked-out
padding — because {{eq:padding-efficiency}}'s denominator is an expected maximum
over a heavy tail. **The waste is invisible and grows with batch size.**

**Bucketing beats packing on every row**, 0.998 against 0.854, because
{{eq:packing-efficiency}} shows packing's waste moved rather than vanished.
Sorting by length is one line and recovers nearly everything;
{{eq:bucket-beats-pack}} says packing overtakes only when the block is large
relative to the examples — **the long-context regime, not the instruction-tuning
regime where it is usually recommended.**

**And packing carries a hazard bucketing does not**: **64.8%** of attend-able pairs
cross an example boundary without a block-diagonal mask
({{eq:cross-contamination}}), training the model to treat unrelated preceding text
as relevant.

**Truncation is the finding to carry.** At 2048 tokens, 2.8% of examples truncated
becomes **6.7% of completion tokens lost** — amplification **2.39×**, rising to
**3.72×** at 4096, because the survivors come from further out in the tail.

**But the data loss is not the damage.** **8.3% of trained completion tokens come
from an example ending mid-sentence at the budget**
({{eq:truncation-teaches-stopping}}), a share larger than the truncated fraction
because each truncated example contributes maximally. **The model is taught,
consistently, that a good answer stops at max_len** — which appears in production
as trailing-off generations, misdiagnosed as a context-window limit.

**The fix costs nothing.** Drop over-length examples instead: the same completion
tokens are lost either way, and the false stopping points disappear entirely. Or
split at natural boundaries.

**Truncation is the tooling default, and it is the only option that actively
teaches something untrue.** Which is the shape of this whole chapter: the
expensive mistakes are not modelling decisions, they are flags nobody set.

## 21. Further Reading

{{cite:ouyang2022}} for what SFT is doing in the wider pipeline, and
{{ch:fm-instruction-tuning}} for the objective this chapter takes as given.
{{cite:zhou2023lima}} for the data-quality argument that decides epoch counts as
much as it decides dataset size.
{{cite:wang2023selfinstruct}} for where the pairs come from, developed in
{{ch:ft-synthetic}}.
{{cite:touvron2023llama}} for a full training recipe reported in enough detail to
read the logistics rather than only the results.
{{cite:vaswani2017}} for why sequence length is quadratic, and
{{ch:tf-complexity}} for what that costs at fine-tuning scale.
{{cite:hu2021lora}} next, which changes the memory arithmetic of everything here.

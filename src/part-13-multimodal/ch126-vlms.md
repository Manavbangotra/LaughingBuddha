---
id: mm-vlms
number: 126
part: XIII
tier: full
status: draft
requires: [mm-vit, mm-clip, mm-ocr, mm-layout, llm-long-context,
           llm-function-calling, fm-instruction-tuning]
provides: [frozen-tower-connector, visual-token-budget, connector-capacity,
           dynamic-resolution, visual-instruction-tuning, vlm-failure-surface]
citations: [alayrac2022flamingo, li2023blip2, liu2023llava, wang2024qwen2vl,
            radford2021clip, mathew2021docvqa, masry2022chartqa]
---

## 1. Learning Objectives

By the end of this chapter you will be able to describe the **frozen-tower plus
connector** architecture every VLM shares and say what each component contributes;
derive the **visual token budget** as the product of a rising legibility factor and
a falling attention factor, and show its optimum *moves with content*, which is
the argument for dynamic resolution; measure a connector's **capacity ceiling** and
explain why it is a uniform tax rather than a cliff; explain why a fixed-query
connector was right for captioning and wrong for documents without anything about
it changing; and locate a VLM failure in the tower, the connector, or the language
model.

## 2. Why This Matters

A vision-language model is the thing a reader in 2026 will actually reach for. It
absorbs {{ch:mm-classification}} through {{ch:mm-layout}} into one call, and it
works well enough that the chapters before it can look like history.

**They are not history, because a VLM's failures are inherited.** It cannot read
small print because of {{ch:mm-vit}}'s {{eq:patch-compression}}. It miscounts and
misplaces because of {{ch:mm-clip}}'s caption supervision. It gets chart
differences wrong because of {{ch:mm-layout}}'s
{{eq:derived-quantity-amplification}}. Every one of those is a component failure
wearing a general model's interface.

This chapter adds the two failures that belong to the VLM itself, and both are
budget problems.

**The visual token budget has an interior optimum that moves.**
{{sec:9-practical-example}} finds it at **N = 256** for a photograph and **N =
1024** for a dense page, because legibility rises with token count while the
chance the model actually *uses* a given token falls — from **0.779 at 256 tokens
to 0.180 at 4096**. A fixed grid is mis-sized for most of its inputs, in one
direction or the other.

**And the connector is lossy always, not lossy eventually.** The measurement
corrected the intuition this chapter was built around: an 8-token connector was
expected to hold up until the image got busy and then collapse. It does not. At
**four facts** — where eight tokens ought to be plentiful — it already recovers
**0.213** against full projection's **0.778**.

**A fixed-length connector does not fail at a threshold. It discards from the
start, and you notice only when a task needs what was discarded** — which is
exactly why the same connector looked excellent at captioning and poor at
documents without changing at all.

{{maturity:ESTABLISHED}} The frozen-tower architecture.
{{maturity:MATURE}} Dynamic resolution ({{cite:wang2024qwen2vl}}).

## 3. Prerequisites

{{ch:mm-vit}} for the tower and {{eq:patch-compression}};
{{ch:mm-clip}} for how it was trained and what its supervision did not reward;
{{ch:llm-long-context}} for why more context is not more usable context;
{{ch:llm-function-calling}} for {{eq:max-distractor}};
{{ch:fm-instruction-tuning}} for the training stage that makes a VLM followable;
{{ch:mm-ocr}} and {{ch:mm-layout}} for the document failures it inherits.

## 4. Intuitive Explanation

### Everyone builds the same three pieces

```text
   image --> [ VISION TOWER ] --> [ CONNECTOR ] --> [ LANGUAGE MODEL ] --> text
              usually frozen       the only        usually frozen
              CLIP/SigLIP          new part        (or lightly tuned)
```

{{cite:alayrac2022flamingo}} established the shape: freeze two strong pretrained
models, train the bridge between them. Every VLM since is a variation on **what
the bridge is** and **how many tokens it emits**.

The economy of it is the point. Both towers cost enormous sums to pretrain and
both already exist. The connector is small, so a VLM is affordable to build — and
{{cite:liu2023llava}} pushed that to its conclusion by showing a **single linear
projection** plus good instruction data is competitive with far more elaborate
bridges.

**Which relocates the expensive part.** If the architecture is a linear layer,
the thing that distinguishes VLMs is the *data*, and
{{cite:liu2023llava}}'s contribution is as much the synthetic instruction-data
recipe as the model.

### The token budget pulls both ways

An image becomes $N$ visual tokens in the language model's context. Two forces:

> **More tokens → smaller patches → finer detail survives.** {{ch:mm-vit}}'s
> {{eq:patch-compression}}: if the patch is bigger than the stroke, the text is
> gone before the model runs.
>
> **More tokens → the relevant one is a smaller share of a longer context.**
> {{ch:llm-long-context}} measured that usable context is well below advertised
> context, and {{eq:max-distractor}} says selection degrades as distractors
> accumulate.

Legible is not the same as used. The product of a rising factor and a falling one
peaks somewhere in the middle, and {{sec:9-practical-example}} finds the peak in
**different places for different content** — 256 tokens for a photograph, 1024 for
a dense page.

**A photograph at 4096 tokens pays the full attention penalty and buys nothing**,
because it was already fully legible at 256. **A page at 256 tokens was never
encoded**, and no amount of language model recovers it.

That is the case for **dynamic resolution**: let the token count follow the image
rather than the architecture. Note what it does *not* do — it does not raise the
ceiling for any one image; it stops the model paying a fixed price regardless of
what it is looking at.

### The connector is a bottleneck with a fixed size

Three designs, and the difference is how many tokens come out:

| Connector | Tokens | Character |
|---|---|---|
| linear projection ({{cite:liu2023llava}}) | one per patch | no ceiling; pays full context |
| fixed queries ({{cite:li2023blip2}}) | a constant | cheap; ceiling independent of content |
| pooling | patches / $k$ | middle, same shape of limit |

{{sec:9-practical-example}} measures what a fixed size costs, and the result is
sharper than the usual framing. **It is not that eight tokens are fine until the
image is busy.** At four facts, eight tokens already recover 0.213 against full
projection's 0.778 — a uniform tax set by the compression ratio, present in the
simplest image.

**Which explains a piece of history that otherwise looks like a reversal.**
{{cite:li2023blip2}}'s Q-Former was efficient and correct for captioning, because
a caption conveys a handful of facts and eight tokens carry them. Point the same
connector at a document — hundreds of distinct facts on a page — and the
information was never there. The connector did not regress; the task started
needing what it had always been throwing away.

### Where a VLM's failures actually come from

A useful habit: when a VLM gets something wrong, ask *which component*.

| Symptom | Component | Chapter |
|---|---|---|
| cannot read small text | tower — patch compression | {{ch:mm-vit}} |
| miscounts objects | tower supervision — captions rarely count | {{ch:mm-clip}} |
| confuses spatial relations | tower supervision, same reason | {{ch:mm-clip}} |
| loses detail in complex images | connector capacity | this chapter |
| ignores part of a long document | token budget / attention | {{ch:llm-long-context}} |
| chart differences wrong | arithmetic conditioning | {{ch:mm-layout}} |
| fluent, confident, fabricated | language model | {{ch:llm-hallucination}} |

**Most of the column on the right is not this chapter**, which is the argument for
having read the others.

## 5. Formal Explanation

### 5.1 The architecture

$$ y = \text{LM}\big(\,[\,c(f_V(x)),\; \text{prompt tokens}\,]\,\big), \qquad c: \mathbb{R}^{P \times d_V} \to \mathbb{R}^{T \times d_L} $$ (eq:vlm-architecture)

with $f_V$ the frozen vision tower producing $P$ patch features and $c$ the
connector emitting $T$ tokens in the language model's embedding space. **Only $c$
is necessarily trained**, and $T$ is the design variable everything else follows
from.

### 5.2 The token budget

Two factors. Legibility: a feature of size $s$ pixels survives when the patch is
no larger, and patch size is $\text{img}/\sqrt{N}$:

$$ \ell(N) = \min\!\left(1,\; \frac{s\sqrt{N}}{\kappa\,\text{img}}\right) \quad \text{— increasing in } N $$ (eq:legibility)

Attention: from {{eq:max-distractor}}, the probability the model uses a given
relevant token falls as the context fills:

$$ a(N) = \frac{1}{1 + N/N_0} \quad \text{— decreasing in } N $$ (eq:attention-dilution)

$$ \text{utility}(N) = \ell(N)\, a(N) $$ (eq:legible-times-attended)

$\ell$ saturates at 1 once patches are smaller than the feature, and $a$ keeps
falling, so **the optimum is at or just past the saturation point of $\ell$** —
and that point depends on $s$, a property of the *image*:

$$ N^{*} \approx \left(\frac{\kappa\,\text{img}}{s}\right)^{2} $$ (eq:optimal-token-count)

**{{eq:optimal-token-count}} is the whole argument for dynamic resolution.**
$N^*$ scales as $1/s^2$, so content four times finer wants sixteen times the
tokens, and no fixed grid serves both.

### 5.3 The cost

$$ C(N) = \underbrace{\alpha N}_{\text{prefill}} + \underbrace{\beta N^2}_{\text{attention}} $$ (eq:vlm-cost)

which is {{ch:mm-vit}}'s {{eq:vit-attention-cost}} arriving in the language
model's context instead of the tower's. Combined with
{{eq:optimal-token-count}}, cost scales as $s^{-2}$ to $s^{-4}$ in feature size —
**halving the text size you need to read multiplies cost by between 4 and 16.**

### 5.4 Connector capacity

The connector maps $P$ patch vectors to $T$ tokens. Whatever its form, the output
occupies at most $T \cdot d_L$ numbers against the input's $P \cdot d_V$, so the
compression ratio

$$ \rho_c = \frac{T\,d_L}{P\,d_V} $$ (eq:connector-capacity)

bounds what can survive. Two consequences, and the measurement establishes the
second:

**(a)** Information beyond the bound is lost, unrecoverably, before the language
model sees anything.

**(b)** The loss is **not conditional on content density**. It applies to the
simplest image, because $\rho_c$ does not depend on what is in the picture:

$$ \frac{\partial\, \text{loss}}{\partial(\text{content complexity})} \approx 0 \quad \text{at fixed } \rho_c $$ (eq:uniform-tax)

**{{eq:uniform-tax}} is the corrected intuition.** A fixed-length connector is
lossy always, and the *task* determines whether you notice.

### 5.5 Two ceilings, compounding

There is a second bound that belongs to the patch representation rather than the
connector: $P$ patches of $d_V$ dimensions hold a bounded number of superposed
facts, so density degrades *every* architecture. {{sec:9-practical-example}}
measures full projection falling 0.778 → 0.669 over a 256-fold density increase.

The two compound unevenly:

$$ \text{relative loss at } T=8: 44\%, \qquad \text{at } T=P: 14\% $$ (eq:compounding-ceilings)

**Compression does not merely subtract a constant — it makes the system more
fragile to exactly the density that made compression tempting.**

### 5.6 Instruction tuning

The connector aligns representations; it does not make the model follow
instructions. {{cite:liu2023llava}}'s second stage is
{{ch:fm-instruction-tuning}} applied multimodally, on data generated by prompting
a text-only model with image annotations.

$$ \text{VLM quality} \approx g(\text{tower},\, \text{LM},\, \underbrace{\text{instruction data}}_{\text{the expensive part}}) $$ (eq:vlm-quality)

With a linear connector competitive against elaborate ones,
{{eq:vlm-quality}}'s third argument is where the differentiation actually is.

## 6. Mathematical Foundation

### 6.1 The optimum, worked

At $\text{img} = 1024$ px, $\kappa = 0.9$:

| content | feature $s$ | $N^*$ from {{eq:optimal-token-count}} | measured |
|---|---|---|---|
| photograph | 90 px | $(0.9\cdot1024/90)^2 = 105$ | **256** |
| slide | 26 px | $(0.9\cdot1024/26)^2 = 1257$ | **1024** |
| dense page | 11 px | $(0.9\cdot1024/11)^2 = 7022$ | **1024** |

The photograph and slide match to within one grid step of the sweep. **The dense
page does not**, and the disagreement is informative: {{eq:optimal-token-count}}
ignores $a(N)$, and for very fine features the attention penalty bites before
legibility saturates. So the true optimum is *below* the legibility-saturating
value, and the model settles for partial legibility rather than paying the
dilution.

$$ \text{when } N^*_{\ell} \text{ is large, } N^*_{\text{true}} < N^*_{\ell} \quad \text{— the page is never fully read} $$ (eq:unreachable-legibility)

> **MATH NOTE:** {{eq:unreachable-legibility}} is the honest statement of why VLMs
> remain imperfect on dense documents even at high resolution. It is not that the
> right budget has not been found; it is that at the budget which would make the
> page legible, dilution has already cost more than legibility gained. The fix is
> not more tokens — it is a better $a(N)$, which is
> {{ch:llm-long-context}}'s problem rather than a vision problem.

### 6.2 What each factor contributes

At $N = 256$ against $N = 4096$:

| content | legible @256 | @4096 | attended @256 | @4096 |
|---|---|---|---|---|
| photograph | **1.000** | 1.000 | 0.779 | **0.180** |
| dense page | **0.191** | 0.764 | 0.779 | **0.180** |

**Read the photograph's row as a pure loss.** Legibility is already 1.000 at 256,
so the sixteenfold increase buys nothing and costs a factor of 4.3 in attention.
**Read the page's row as a genuine trade**: legibility improves fourfold while
attention falls by the same factor, which is why its optimum is interior rather
than at either end.

### 6.3 The compression tax, quantified

From the measurement, recoverability against compression ratio $T/P$:

$$ T/P = 0.125 \to 0.213, \qquad 0.5 \to 0.583, \qquad 1.0 \to 0.778 $$ (eq:compression-tax)

Roughly $R^2 \propto (T/P)^{0.35}$ — **sub-linear, so the first tokens you remove
are the cheapest and the last are not.** Halving from full projection to $T = P/2$
costs a quarter of the recoverable information; going to $T = P/8$ costs three
quarters.

And the *mechanism* is confirmed by the pooling control: learned global queries
and simple neighbour-averaging reach **0.213** and **0.232** at the same $T$.
**Two completely different compressions, essentially the same result** — so the
ceiling is the token count, and the method changes the constant rather than the
shape.

## 7. Internal Mechanics

```mermaid {#fig:vlm-stack caption="The shared architecture, with the two budgets marked. T is the connector's output size and sets eq:connector-capacity's ceiling; it is also what lands in the language model's context, so it drives eq:legible-times-attended's dilution. One number, two independent costs, which is why it is the design decision."}
flowchart LR
    IMG["image"] --> VT["vision tower<br/>(frozen CLIP/SigLIP)"]
    VT --> PF["P patch features"]
    PF --> CN{"connector"}
    CN -->|"linear: T = P"| TK["T visual tokens"]
    CN -->|"queries: T fixed"| TK
    CN -->|"pooling: T = P/k"| TK
    TK --> CTX["language model context"]
    PR["text prompt"] --> CTX
    CTX --> LM["language model<br/>(frozen or lightly tuned)"]
    LM --> OUT["text"]
    CN -.->|"eq:connector-capacity:<br/>hard information ceiling"| TK
    TK -.->|"eq:attention-dilution:<br/>more tokens, less used"| CTX
```

### 7.1 Dynamic resolution, concretely

{{cite:wang2024qwen2vl}}'s mechanism is to let the patch grid follow the image's
actual size rather than resizing everything to a fixed square. A small image
becomes few tokens; a large page becomes many.

**Two things that follow and are easy to miss:**

- Position encoding must handle variable grids, which is why rotary schemes over
  2D positions replace learned per-position embeddings
  ({{ch:mm-vit}}).
- **Cost becomes input-dependent**, so throughput planning needs the *distribution*
  of input sizes rather than a per-image constant. A batch of documents and a
  batch of thumbnails are different workloads on the same endpoint.

### 7.2 Tiling, and its seam

The common way to reach high resolution without a single enormous grid: cut the
image into tiles, encode each at the tower's native resolution, and concatenate
the tokens — usually with a downscaled whole-image thumbnail alongside for global
context.

**The failure it introduces is the seam.** An object or a table row spanning two
tiles is seen by neither tile completely, and the model must reassemble it from
two partial views with no explicit signal that they are adjacent. Symptom: errors
concentrated on content that crosses tile boundaries, which looks random until you
overlay the tile grid.

### 7.3 What is frozen and what is not

| Component | Usually | Why |
|---|---|---|
| vision tower | frozen | pretrained on far more data than you have |
| connector | trained | it is the new part |
| language model | frozen, or LoRA | full fine-tuning degrades text ability |

**Unfreezing the tower is the tempting move and usually the wrong one**, because
the tower's generality came from {{cite:radford2021clip}}-scale data and your
fine-tuning set is small enough to destroy it — {{part:14}}'s catastrophic
forgetting, with the added cost that the damage shows up on inputs you did not
test.

## 8. Implementation

```python {tier=A name=visual-token-budget}
"""The visual token budget, and why one fixed resolution is wrong for everything.

A VLM turns an image into visual tokens and puts them in a language model's
context. How many is the central design decision, and it is pulled in two
directions at once.

More tokens means smaller patches, so finer detail survives the patch embedding
(ch:mm-vit, eq:patch-compression) and more of the page is LEGIBLE. More tokens
also means the relevant token is a smaller share of a longer context, and
ch:llm-long-context measured that usable context is well below advertised
context -- so being legible is not the same as being USED
(eq:legible-times-attended).

The product of a rising factor and a falling one has an interior maximum, and
where it sits depends on how much detail the image actually contains. This listing
finds it, then shows the optimum MOVING with content density -- which is the
argument for dynamic resolution (cite:wang2024qwen2vl).
"""
import numpy as np

IMG_PX = 1024.0            # page rendered at this many pixels on a side
STROKE_RATIO = 0.9         # a feature is legible if patch <= feature / this
ATT_HALF = 900.0           # tokens at which attention to a given token halves
COST_LIN = 1.0             # per-token prefill cost, arbitrary units
COST_QUAD = 1.0 / 1500.0   # attention's quadratic term, relative to the linear


def legible_share(n_tokens, feature_px):
    """eq:legibility -- share of features that survive patchification: a feature
    is resolvable when the patch is no larger than the feature itself."""
    patch = IMG_PX / np.sqrt(n_tokens)
    return float(np.clip(feature_px / (patch * STROKE_RATIO), 0.0, 1.0))


def attended(n_tokens):
    """eq:attention-dilution. ch:llm-long-context and ch:llm-function-calling's
    eq:max-distractor: the chance the model actually uses a given relevant token
    falls as the context fills with others."""
    return 1.0 / (1.0 + n_tokens / ATT_HALF)


def cost(n_tokens):
    """eq:vlm-cost: linear prefill plus quadratic attention."""
    return COST_LIN * n_tokens + COST_QUAD * n_tokens ** 2


CONTENT = [("a photograph (coarse)", 90.0),
           ("a slide (medium text)", 26.0),
           ("a dense page (small print)", 11.0)]
BUDGETS = (64, 256, 576, 1024, 2304, 4096)

print(f"page rendered at {IMG_PX:.0f} px; a feature is legible when the patch "
      f"is no bigger than it\n")
print(f"{'content':<28}" + "".join(f"{'N=' + str(b):>10}" for b in BUDGETS)
      + f"{'best N':>9}{'at cost':>10}")
print("-" * 97)

best = {}
for name, feat in CONTENT:
    scores = [legible_share(b, feat) * attended(b) for b in BUDGETS]
    i = int(np.argmax(scores))
    best[name] = (BUDGETS[i], scores[i], cost(BUDGETS[i]))
    print(f"{name:<28}" + "".join(f"{s:>10.3f}" for s in scores)
          + f"{BUDGETS[i]:>9}{cost(BUDGETS[i]):>10.0f}")

print(f"\n{'content':<28}{'legible@256':>13}{'attended@256':>14}"
      f"{'legible@4096':>14}{'attended@4096':>15}")
print("-" * 84)
for name, feat in CONTENT:
    print(f"{name:<28}{legible_share(256, feat):>13.3f}{attended(256):>14.3f}"
          f"{legible_share(4096, feat):>14.3f}{attended(4096):>15.3f}")

photo, dense = best["a photograph (coarse)"], best["a dense page (small print)"]
print(f"""
Every row has an interior maximum, and the maxima are in different places:
N={photo[0]} for the photograph and N={dense[0]} for the dense page. That is the
whole argument in one line -- there is no single token budget that is right for
both, and a model with a fixed grid is mis-sized for most of its inputs.

The second table shows the two factors separately, which is what makes the shape
non-obvious. At N=256 the photograph is already fully legible (1.000) and the
dense page is not ({legible_share(256, 11.0):.3f}) -- the patch is far larger than
the print, so most of the text never survives the patch embedding
(eq:patch-compression). Going to N=4096 fixes legibility for the dense page,
raising it to {legible_share(4096, 11.0):.3f}.

And it costs something the accuracy-only view misses. Attention to any given
relevant token falls from {attended(256):.3f} to {attended(4096):.3f} across that
same change, because the relevant token is now competing with sixteen times as
many others. For the photograph, which was ALREADY fully legible at 256, that
trade is pure loss: it pays the full attention penalty and buys nothing, which is
why its optimum sits at the small end of the table.

So the two content types want opposite things. Spending 4096 tokens on a
photograph wastes budget and dilutes attention; spending 256 on a dense page means
the text was never encoded and no amount of language modelling recovers it. A
fixed grid has to choose, and it is wrong in one direction or the other for almost
every image.

That is the case for dynamic resolution (cite:wang2024qwen2vl): let the token
count follow the image rather than the architecture, so a photograph gets few
tokens and a page gets many. Note what it is NOT solving -- it does not raise the
ceiling for any single image, it just stops the model paying a fixed price
regardless of what it is looking at. The ceiling is still
eq:legible-times-attended.

The cost column is the reason this cannot simply be solved by always choosing the
largest budget. Cost is linear in tokens plus quadratic in attention
(eq:vlm-cost), so the dense page's optimum costs
{dense[2] / photo[2]:.0f} times the photograph's. A system that sends every image
at document resolution is paying document prices for photographs, at a scale where
that is most of the bill.""")
```

The first listing budgets the tokens. The second asks what the connector does to
them on the way.

```python {tier=A name=connector-capacity}
"""The connector is a bottleneck, and a fixed-length one is a hard ceiling.

Between the vision tower and the language model sits a connector that turns P
patch features into T tokens the LLM will read. The design space is small and the
consequences are not:

  FULL PROJECTION   T = P. Every patch becomes a token (cite:liu2023llava's
                    linear projector). Nothing is discarded and the context bill
                    is the full patch count.
  FIXED QUERIES     T is a constant, independent of the image
                    (cite:li2023blip2's Q-Former). Cheap, and it imposes a
                    capacity ceiling that does not move with content
                    (eq:connector-capacity).
  POOLING           T = P/k. A middle position with the same shape of limit.

The question this listing answers is not "which is more accurate" but "how many
distinct things can an image convey through each", because that is what a fixed T
bounds. Facts are placed in the image, passed through each connector, and then
recovered by a linear decoder -- so what is measured is whether the INFORMATION
survived, not whether some particular model used it.
"""
import numpy as np

rng = np.random.default_rng(97)

P, D = 64, 16          # patches, feature dimension per patch
N_IMG = 3000
NOISE = 0.35


def make_images(n_facts):
    """Each image contains a random subset of n_facts distinct facts. A fact
    lives in one patch and has its own direction."""
    where = rng.integers(0, P, size=n_facts)
    dirs = rng.normal(size=(n_facts, D))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    Y = (rng.random((N_IMG, n_facts)) < 0.5).astype(float)
    X = NOISE * rng.normal(size=(N_IMG, P, D))
    for f in range(n_facts):
        X[:, where[f], :] += Y[:, f:f + 1] * dirs[f]
    return X, Y


def connector(kind, T):
    """Return a (T, P) mixing matrix."""
    if kind == "full":
        return np.eye(P)
    if kind == "pool":
        A = np.zeros((T, P))
        g = P // T
        for t in range(T):
            A[t, t * g:(t + 1) * g] = 1.0 / g
        return A
    # Fixed learned queries, modelled as a dense random mixing over patches --
    # each query reads the whole image and emits one token.
    A = rng.normal(size=(T, P)) / np.sqrt(P)
    return A


def recoverable(X, Y, A):
    """Mean R^2 of a ridge decoder recovering each fact's presence from the
    connector's OUTPUT. This measures information survival, not model skill."""
    Z = np.einsum("tp,npd->ntd", A, X).reshape(len(X), -1)
    Z = np.hstack([Z, np.ones((len(Z), 1))])
    ridge = 1e-3 * np.eye(Z.shape[1])
    W = np.linalg.solve(Z.T @ Z + ridge, Z.T @ Y)
    pred = Z @ W
    ss_res = ((Y - pred) ** 2).sum(0)
    ss_tot = ((Y - Y.mean(0)) ** 2).sum(0)
    return float(np.mean(1.0 - ss_res / ss_tot))


FACTS = (4, 16, 64, 256, 1024)
SETUPS = [("full projection (T=64)", "full", P),
          ("fixed queries, T=8", "query", 8),
          ("fixed queries, T=32", "query", 32),
          ("pooling to T=8", "pool", 8)]

print(f"{P} patches of {D} dims; facts recovered by a linear decoder from the "
      f"connector output\n")
print(f"{'connector':<26}{'tokens':>8}" + "".join(f"{str(m) + ' facts':>12}"
                                                  for m in FACTS))
print("-" * 94)

res = {}
for name, kind, T in SETUPS:
    row = []
    for m in FACTS:
        X, Y = make_images(m)
        row.append(recoverable(X, Y, connector(kind, T)))
    res[name] = row
    print(f"{name:<26}{T:>8}" + "".join(f"{v:>12.3f}" for v in row))

q8, q32 = res["fixed queries, T=8"], res["fixed queries, T=32"]
full, pool8 = res["full projection (T=64)"], res["pooling to T=8"]
print(f"""
The first thing to read is a column, not a row, and it corrects the intuition
this listing was built to test. The expectation was a KNEE -- a fixed-length
connector holding up while the content fits inside T tokens and collapsing once
it does not. There is no knee. At 4 facts, where eight tokens ought to be
plentiful, T=8 already recovers only {q8[0]:.3f} against full projection's
{full[0]:.3f}.

So a fixed-length connector is not "adequate until the image gets busy". It is
LOSSY ALWAYS, by an amount set by the compression ratio T/P, and the loss is
there in the simplest image. The ceiling is a uniform tax rather than a cliff
(eq:uniform-tax).

That distinction changes the diagnosis in a useful way. If it were a cliff you
would expect a connector to work and then fail as documents got denser, and you
would look for the threshold. What actually happens is that the information is
missing from the start and you only NOTICE when a task needs the part that was
discarded. Captioning does not need it -- a caption conveys a handful of facts, so
eight tokens carry enough of them and cite:li2023blip2's efficiency argument was
correct for what it was measured on. Document work needs it, which is where the
same connector looked like it had regressed and had not changed at all.

Read across the rows for the second, separate effect. Content density degrades
every connector -- full projection falls from {full[0]:.3f} to {full[-1]:.3f} as
facts go from 4 to 1024 -- because facts start sharing patches and superpose
within a fixed-dimension feature. That is a SECOND ceiling, belonging to the patch
representation rather than the connector, and it is why more visual tokens
eventually stop helping even with no connector at all.

The two ceilings compound unevenly, which is the practically important part. Over
the same sweep the T=8 connector falls {(1 - q8[-1]/q8[0]) * 100:.0f}% while full
projection falls {(1 - full[-1]/full[0]) * 100:.0f}%. Compression does not merely
subtract a constant; it makes the system more fragile to exactly the density that
makes compression tempting (eq:compounding-ceilings).

The pooling row is the control. It compresses 64 patches to 8 tokens by a
completely different mechanism -- averaging neighbours rather than learned global
queries -- and lands at {pool8[0]:.3f} against the query connector's {q8[0]:.3f}.
Essentially the same. What sets the ceiling is the token count, and how you get
there changes the constant rather than the shape.

So the design question is not which connector is more elegant. It is: how many
distinct things must one image convey, and can I afford that many tokens? A
caption needs few, a spreadsheet needs many, and that number -- not FLOPs and not
architecture -- is what should set T. cite:liu2023llava's plain linear projection
won for documents not because it is cleverer but because it declines to answer
the question in advance.""")
```

## 9. Practical Example

**The token budget's optimum moves with the content.** Every content type has an
interior maximum, and they are in different places: **N = 256** for a photograph,
**N = 1024** for a dense page.

The two factors explain the shape. At **N = 256** the photograph is already fully
legible (**1.000**) while the dense page is at **0.191** — the patch is far larger
than the print, so most of the text never survives {{eq:patch-compression}}. Going
to **N = 4096** raises the page to **0.764** and drops attention to any given token
from **0.779 to 0.180**.

**For the photograph that trade is pure loss** — full legibility already, so the
sixteenfold increase buys nothing and pays the whole dilution penalty. **For the
page it is a genuine trade**, which is why its optimum is interior.

**So a fixed grid is wrong in one direction or the other for almost every image**,
and that is the argument for dynamic resolution. {{eq:optimal-token-count}}
predicted 105 and 1257 tokens for the photograph and slide against measured 256
and 1024 — within a grid step.

> **IMPORTANT:** The dense page is where the closed form breaks, and informatively.
> {{eq:optimal-token-count}} says 7022 tokens; the measured optimum is 1024,
> because {{eq:unreachable-legibility}} — at the budget which would make the page
> legible, dilution has already cost more than legibility gains. **The page is
> never fully read at any budget.** That is not a tuning failure; the fix is a
> better $a(N)$, which is {{ch:llm-long-context}}'s problem rather than a vision
> one.

**And the connector is lossy always, not lossy eventually.** This listing was built
to find a knee and there is none. At **four facts** — where eight tokens should be
plentiful — T=8 recovers **0.213** against full projection's **0.778**.

**A fixed-length connector does not fail at a threshold; it discards from the
start.** Which is a more useful diagnosis than a cliff would be: you do not look
for a density threshold, you ask whether your task needs what was always missing.
Captioning does not — a caption conveys a handful of facts, so
{{cite:li2023blip2}}'s efficiency argument was correct for what it was measured
on. Documents do, which is why the same connector looked like it regressed while
nothing about it changed.

**The second ceiling belongs to the patch representation.** Content density
degrades everything — full projection falls **0.778 → 0.669** over a 256-fold
density increase — and the two ceilings compound unevenly: **T=8 loses 44%** over
that sweep against full projection's **14%**
({{eq:compounding-ceilings}}). **Compression makes the system more fragile to
exactly the density that made compression tempting.**

**And the pooling control confirms the mechanism.** Learned global queries and
neighbour-averaging — completely different compressions — land at **0.213** and
**0.232** at the same $T$. The ceiling is the token count; the method changes the
constant, not the shape.

## 10. Production Considerations

**Size the token budget to your content, not to a default.**
{{eq:optimal-token-count}}: measure the smallest feature you must read, and
compute.

**Use dynamic resolution if available**, and plan capacity from the *distribution*
of input sizes — a document batch and a thumbnail batch are different workloads on
one endpoint.

**Do not send photographs at document resolution.** Pure cost, measured: the dense
page's optimum costs ~6× the photograph's.

**Prefer a full-projection connector for document work** and a compressed one only
where you have measured the task needs few facts per image.

**Check for tile seams** if your provider tiles. Overlay the tile grid on your
error cases; content crossing boundaries is the signature.

**Keep the vision tower frozen** unless you have a large in-domain dataset, and
evaluate general capability after any unfreezing.

**Attribute failures to a component** before trying to fix them
({{sec:4-intuitive-explanation}}'s table). Most VLM complaints are tower or
supervision failures that no prompting reaches.

**Log input resolution and token count per request.** Without them you cannot tell
a budget failure from a model failure.

## 11. Common Mistakes

**Treating the VLM as a black box** when most failures are attributable to a
component.

**Assuming more visual tokens is monotonically better** —
{{eq:legible-times-attended}} says no.

**Using a fixed-query connector for documents.**

**Resizing a document to the tower's default square** and losing the text before
inference.

**Fine-tuning the vision tower on a small dataset.**

**Reading chart differences and counts from a VLM** —
{{ch:mm-layout}} and {{ch:mm-clip}} say those are the two weakest cases.

**Benchmarking on captioning and deploying on documents**, which is the exact
mistake {{eq:uniform-tax}} explains.

## 12. Failure Modes

**Small-text blindness.** Symptom: cannot read fine print at any prompt. Cause:
tower patch compression, or input resize. Not fixable downstream.

**Detail loss in busy images.** Symptom: correct on simple images, misses things
in complex ones. Cause: {{eq:connector-capacity}} plus
{{eq:compounding-ceilings}}.

**Long-document neglect.** Symptom: attends to the first page and ignores later
ones. Cause: {{eq:attention-dilution}}.

**Tile-seam errors.** Symptom: errors on content crossing tile boundaries.

**Counting and spatial errors.** Symptom: wrong object counts, wrong left/right.
Cause: {{ch:mm-clip}}'s caption supervision never rewarded them.

**Confident fabrication.** Symptom: fluent detail not present in the image. Cause:
the language model, and worse when visual evidence was weak — which happens
precisely when the budget was too small.

**Cost blowup on a mixed workload.** Symptom: spend far above forecast. Cause:
dynamic resolution meeting a distribution of large inputs.

## 13. Alternatives

| Alternative | Trades away | When it wins |
|---|---|---|
| specialist model per task | generality | high volume, one task, latency-bound |
| pipeline ({{ch:mm-ocr}}) | implicit layout handling | when you need the text artefact |
| fixed-query connector ({{cite:li2023blip2}}) | capacity | captioning, few facts per image |
| full projection ({{cite:liu2023llava}}) | context budget | documents, dense content |
| tiling | seam errors | high resolution on a fixed-grid tower |
| dynamic resolution ({{cite:wang2024qwen2vl}}) | predictable cost | mixed input sizes |

**The second row is the one most often skipped.** A VLM handles layout implicitly
and correctly and leaves no artefact; {{ch:mm-ocr}}'s
{{eq:different-artefacts}} argues the text layer is the product for most document
systems, and the answer is usually both rather than either.

## 14. Evaluation

**Evaluate at your deployment resolution**, and report it. A VLM number without
its input resolution is not reproducible.

**Break results down by content density** — captioning, natural images, slides,
dense documents. {{eq:uniform-tax}} says a connector that looks fine on one will
not on another.

**Evaluate document tasks specifically** ({{cite:mathew2021docvqa}}) and chart
tasks by question type ({{cite:masry2022chartqa}}), since their failure modes are
unrelated.

**Test counting and spatial relations explicitly.** They are the known weak cases
and general benchmarks under-weight them.

**Report cost per image alongside accuracy.** With dynamic resolution these are
coupled, and an accuracy comparison at different token counts is not a comparison.

**Attribute errors by component** in any error analysis.

## 15. Advanced Concepts

**Any-resolution as the current frontier.** {{maturity:MATURE}}
{{cite:wang2024qwen2vl}}'s dynamic resolution plus 2D rotary positions is the
design most new VLMs adopt, and its cost consequence — input-dependent pricing —
is under-discussed relative to its accuracy consequence.

**Token pruning and merging.** {{maturity:EMERGING}} Many visual tokens are
redundant (blank margins, uniform background). Merging them attacks
{{eq:attention-dilution}} directly, keeping legibility while shortening the
context — the only intervention that improves both terms of
{{eq:legible-times-attended}} at once.

**Interleaved image-text training.** {{maturity:MATURE}}
{{cite:alayrac2022flamingo}}'s sequences of alternating images and text are what
enable multi-image reasoning and in-context learning with images, and a model
trained only on single image-text pairs will not do it however you prompt.

**The connector's ceiling as a design specification.**
{{maturity:EMERGING}} {{eq:connector-capacity}} inverts: given the number of
distinct facts a task needs from one image, it says how many tokens the connector
must emit. Almost nobody sizes it this way, and it is one division.

**Instruction data as the differentiator.** {{maturity:MATURE}}
{{eq:vlm-quality}} with a linear connector says the architecture is nearly
commodity and the data is not — so the transferable skill from
{{cite:liu2023llava}} is the synthetic-data recipe rather than the model.

## 16. Connection to Previous Chapters

{{ch:mm-vit}} is the tower and {{eq:patch-compression}} is
{{eq:legibility}}'s origin; {{ch:mm-clip}} is how the tower was trained and why
counting and spatial relations are weak. {{ch:llm-long-context}} and
{{ch:llm-function-calling}}'s {{eq:max-distractor}} supply
{{eq:attention-dilution}}, so the visual token budget is a context-budget problem
wearing a vision hat. {{ch:mm-ocr}}'s {{eq:different-artefacts}} is the argument
for keeping a pipeline alongside; {{ch:mm-layout}}'s
{{eq:derived-quantity-amplification}} explains the chart failures a VLM inherits.
{{ch:fm-instruction-tuning}} is the second training stage. Forward:
{{ch:mm-multimodal-rag}} indexes what this chapter reads, and
{{ch:mm-video-audio}} multiplies {{eq:legible-times-attended}} by a frame count.

## 17. Exercises

1. Use {{eq:optimal-token-count}} to compute $N^*$ for 8-point text on an A4 page
   rendered at 150 DPI. Compare with a real VLM's default.
2. Derive the shape of {{eq:legible-times-attended}} and show the optimum is at or
   past $\ell$'s saturation point.
3. In `visual-token-budget`, halve `ATT_HALF`. Which optima move and in which
   direction?
4. Add a content type at 4 px features. Does {{eq:unreachable-legibility}} apply,
   and what does the table show?
5. In `connector-capacity`, sweep $T$ from 4 to 64 at 64 facts and fit
   {{eq:compression-tax}}'s exponent.
6. Modify the same listing so facts are spatially clustered rather than uniform.
   Does pooling now beat learned queries?
7. Using {{eq:connector-capacity}}, size a connector for a task needing 200
   distinct facts per image. What $T$, and what does it cost by
   {{eq:vlm-cost}}?
8. Take a VLM you use. Find its default token count per image and its tiling
   scheme, then construct an input that straddles a tile boundary and measure the
   error rate against a centred version.

## 18. Interview Questions

1. Draw a VLM's architecture and say what is trained.
2. Why is more visual tokens not monotonically better?
3. Where does the visual token budget's optimum come from, and what moves it?
4. What is a connector's capacity ceiling, and is it a cliff or a tax?
5. Why did fixed-query connectors lose ground to linear projections?
6. Why can a VLM not read small print, and what would fix it?
7. What is dynamic resolution solving, and what is it not solving?
8. Your VLM is correct on simple images and wrong on busy ones. Diagnose.
9. Would you fine-tune the vision tower? Justify.
10. When would you use an OCR pipeline instead of, or alongside, a VLM?

## 19. Research Questions

1. {{eq:unreachable-legibility}} says dense pages are never fully read because
   dilution outpaces legibility. Does token merging change the conclusion, or only
   the constant?
2. {{eq:compression-tax}}'s sub-linear exponent was measured on random mixings.
   Does a *learned* connector achieve a better exponent, and by how much?
3. {{eq:compounding-ceilings}} shows compression amplifies density sensitivity. Is
   there a connector whose loss is content-adaptive rather than fixed?
4. Tiling introduces seams. Is there an overlap or position-encoding scheme that
   provably removes the boundary penalty?
5. {{eq:vlm-quality}} makes instruction data the differentiator. What is the
   scaling law for multimodal instruction data, and does it saturate?

## 20. Chapter Summary

Every VLM is the same three pieces — **frozen vision tower, trained connector,
frozen language model** ({{eq:vlm-architecture}}) — and
{{cite:liu2023llava}} showed the connector can be a single linear layer, which
moves the differentiation from architecture to **instruction data**
({{eq:vlm-quality}}).

**The visual token budget has an interior optimum and it moves with the content.**
Legibility rises with tokens ({{eq:legibility}}) and attention to any given token
falls ({{eq:attention-dilution}}), so their product peaks in the middle —
measured at **N = 256** for a photograph and **N = 1024** for a dense page. At 256
the photograph is fully legible and the page is at 0.191; at 4096 the page reaches
0.764 and attention drops **0.779 → 0.180**. For the photograph that is pure loss;
for the page it is a real trade. **A fixed grid is mis-sized for almost every
input**, which is the argument for dynamic resolution.

**And there is a budget at which the page still is not fully read.**
{{eq:unreachable-legibility}}: for very fine features, dilution costs more than
legibility gains before saturation is reached. Not a tuning failure — a different
problem, belonging to {{ch:llm-long-context}}.

**The connector's ceiling is a uniform tax, not a cliff**, and this corrected the
intuition the chapter was written to test. At **four facts**, where eight tokens
ought to be plentiful, a fixed-query connector recovers **0.213** against full
projection's **0.778**. It is lossy in the simplest image
({{eq:uniform-tax}}), and the *task* determines whether you notice.

**Which explains a history that looks like a reversal and is not.**
{{cite:li2023blip2}}'s Q-Former was efficient and correct for captioning, where a
handful of facts is the whole payload. Documents need hundreds, so the information
was never there — the connector did not regress, the task started needing what it
had always discarded.

**A second ceiling belongs to the patch representation**, and the two compound
unevenly: over a 256-fold density increase, T=8 lost **44%** and full projection
**14%** ({{eq:compounding-ceilings}}). **Compression makes the system more fragile
to exactly the density that made compression attractive.** And the pooling control
— a completely different compression reaching the same number — confirms that the
ceiling is the token count rather than the method.

Finally, the chapter's operating habit: **most VLM failures belong to a
component**, and the right first question is *which*. Small text is the tower.
Counting and spatial relations are the tower's supervision. Detail in busy images
is the connector. Neglecting page five is the context budget. Fluent fabrication
is the language model. Only two of those five are in this chapter, which is the
argument for the four before it.

## 21. Further Reading

{{cite:alayrac2022flamingo}} for the architecture everything inherited, and for
interleaved image-text sequences, which is the part that enables multi-image
reasoning.
{{cite:li2023blip2}} for the Q-Former and its efficiency argument — correct for
what it measured, and read alongside {{eq:uniform-tax}}.
{{cite:liu2023llava}} for the demonstration that a linear projection plus good
data is enough, which is really a paper about data.
{{cite:wang2024qwen2vl}} for dynamic resolution, and note the cost consequence as
much as the accuracy one.
{{cite:mathew2021docvqa}} and {{cite:masry2022chartqa}} for the two benchmarks
that expose the inherited failures rather than the model's own.
{{cite:radford2021clip}} for what the frozen tower learned and did not.

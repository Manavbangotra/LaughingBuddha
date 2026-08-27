---
id: rag-chunking
number: 108
part: XII
tier: full
status: draft
requires: [rag-ingestion, rag-why, emb-what-they-are, emb-reranking,
           nlp-preprocessing]
provides: [chunking, chunk-size, chunk-overlap, semantic-chunking,
           structure-aware-chunking, boundary-loss, chunk-dilution,
           retrieval-unit]
citations: [sarthi2024raptor, khattab2020colbert, gao2023ragsurvey,
            liu2023lost, lewis2020rag]
---

## 1. Learning Objectives

By the end of this chapter you will be able to explain what chunking *is* — a
choice of retrieval unit forced by the single-vector bottleneck — and why that
framing makes the usual questions answerable; demonstrate that the optimal chunk
size depends on the **query distribution** rather than the corpus, and that two
common query types have incompatible optima; quantify what overlap buys against
what it costs; and choose between fixed, structural, semantic, and hierarchical
strategies on evidence rather than fashion.

## 2. Why This Matters

"What chunk size should I use?" is the most-asked question in RAG and the one
with the worst answers. Every tutorial gives a number, the numbers disagree, and
none of them say what the number depends on.

{{sec:9-practical-example}} answers it properly, and the answer is that **the
question is under-specified**. Chunk size trades two things against each other,
and which side you want depends on what people ask. Fact-lookup queries want
small chunks; synthesis queries want large ones; and the measured optima are far
apart. A single number cannot be right for a mixed workload, which is why every
published recommendation disagrees with every other and all of them are
defensible.

The deeper reason to spend a chapter here is that **chunking is not a
preprocessing detail — it is where the retrieval unit is chosen**, and
{{ch:emb-reranking}} showed that choice is forced. A document with several
meanings cannot be one vector. Chunking is the cheap answer to that, and knowing
what it is an answer *to* is what makes the design decisions follow.

{{maturity:MATURE}} Fixed and structural chunking are stable and universal.
{{maturity:EMERGING}} Hierarchical and contextual approaches are displacing the
single-size question rather than answering it.

## 3. Prerequisites

{{ch:rag-ingestion}} for the text stream and structure markers this chapter
consumes; {{ch:emb-what-they-are}} for what an embedding of a chunk means;
{{ch:emb-reranking}}'s single-vector bottleneck, which chunking exists to work
around; {{ch:rag-why}} for the context budget; {{ch:nlp-preprocessing}} for
sentence segmentation.

## 4. Intuitive Explanation

### What chunking actually is

{{ch:emb-reranking}} established that one vector cannot represent a document
covering several unrelated topics: the vector lands on a bisector far from all of
them, and the document loses to specialists on every query.

There are two responses. Keep many vectors per document — late interaction, at
10–100× the storage. Or **split the document so each piece has one meaning, and
give each piece its own vector.**

The second is chunking. It is the poor practitioner's multi-vector retrieval, it
costs nothing extra in storage, and it works. What it does not do is remove the
problem: it converts *"how do I represent a multi-topic document"* into *"where
do I cut"*, which is a different question with its own failure modes.

### The two forces

Every chunking decision is these two pulling against each other.

**Dilution pushes chunks smaller.** A chunk's embedding is roughly the average of
its content. Put one relevant sentence in a chunk of forty and the relevant
signal is one fortieth of the vector; the same sentence alone dominates its own
vector completely. **Small chunks retrieve precisely.**

**Context pushes chunks larger.** A chunk must contain enough for the answer to
be *in* it. A sentence saying *"this represents a 12% improvement"* is worthless
without the sentence naming what improved. And an answer spanning three
sentences is only retrievable if all three are in one chunk. **Large chunks
answer completely.**

Neither force is about the corpus. Both are about **what people ask**, which is
the chapter's central claim: *the optimal chunk size is a property of your query
distribution.* A corpus of research papers serving fact-lookup queries and the
same corpus serving synthesis queries want different chunk sizes, and no
inspection of the corpus reveals which.

### Why overlap exists and what it is not

An answer that straddles a boundary is lost: no single chunk contains it, and the
retriever returns a fragment.

Overlap hedges this by starting each chunk before the previous one ended, so
material near a boundary appears in two chunks. **It is a hedge, not a fix.** It
costs index size linearly, it duplicates content into the context window, and its
benefit is bounded by how often answers actually straddle boundaries — which is
measurable and usually smaller than the default 10–20% assumes.
{{sec:9-practical-example}} measures both sides.

## 5. Formal Explanation

### 5.1 The retrieval unit

A document $d$ is a sequence of atoms $a_1 \dots a_n$ — sentences, usually. A
chunking is a set of windows:

$$ \mathcal{C}(d) = \big\{\, c_j = (a_{s_j}, \dots, a_{e_j}) \,\big\}, \qquad \bigcup_j [s_j, e_j] = [1, n] $$ (eq:chunking-def)

with chunk size $L = e_j - s_j + 1$ and stride $S = s_{j+1} - s_j$. Overlap is
$L - S$, and the **index multiplier** is

$$ m = \frac{L}{S} = \frac{1}{1 - \text{overlap fraction}} $$ (eq:index-multiplier)

so 50% overlap doubles the index, and 20% overlap costs 25%.

### 5.2 Dilution

Model a chunk's embedding as the mean of its atoms' embeddings — which is
literally true for mean pooling and approximately true otherwise. For a query
matching one atom $a^{*}$:

$$ \hat{f}(c) = \frac{1}{L}\sum_{i \in c} \hat{f}(a_i) \;\Longrightarrow\; \hat{f}(c)\T \hat{f}(q) \approx \frac{1}{L}\,\hat{f}(a^{*})\T\hat{f}(q) + \frac{L-1}{L}\,\bar{s} $$ (eq:chunk-dilution)

where $\bar{s}$ is the average similarity of irrelevant atoms to the query. The
signal term falls as $1/L$ while the background term rises toward $\bar{s}$.

**This is {{ch:emb-reranking}}'s {{eq:bottleneck}} with $L$ made explicit**, and
it gives the precise form of "small chunks retrieve better": the signal-to-
background ratio degrades as $1/L$.

### 5.3 Completeness

Against it, the probability that an answer spanning $w$ consecutive atoms fits
inside a single chunk. For a random start position with stride $S$ and size $L$:

$$ \Prob[\text{contained}] = \begin{cases} 1 & L \geq w + S - 1 \\ \max\!\left(0,\ \dfrac{L - w + 1}{S}\right) & \text{otherwise} \end{cases} $$ (eq:span-containment)

Two things follow. **Containment is 1 whenever $L \geq w + S - 1$**, which for no
overlap ($S = L$) means $L \geq w$ is *not* enough — you need roughly $2w$. And
overlap enters only through $S$: halving the stride at fixed $L$ doubles the
containment probability in the partial regime.

### 5.4 The optimisation

Putting {{eq:chunk-dilution}} and {{eq:span-containment}} together, retrieval
success for a query needing a span of width $w$:

$$ \text{success}(L, S \mid w) \;\approx\; \underbrace{\Prob[\text{contained}]}_{\nearrow \text{ in } L} \times \underbrace{\Prob[\text{retrieved} \mid \text{contained}]}_{\searrow \text{ in } L \text{ via eq:chunk-dilution}} $$ (eq:chunk-success)

**A product of an increasing and a decreasing function has an interior maximum,
and its location depends on $w$.** That is the whole result: $w$ is a property of
the *query*, so the optimum is too.

Over a mixed workload with query types $\{(w_t, \pi_t)\}$:

$$ L^{*} = \argmax_L \sum_t \pi_t \cdot \text{success}(L, S \mid w_t) $$ (eq:mixed-workload-optimum)

which is a weighted compromise satisfying nobody — and is the honest reason to
consider {{sec:13-alternatives}}'s multi-granularity indexes instead of tuning a
single $L$.

## 6. Mathematical Foundation

### 6.1 Where the optima separate

Take two query types: **fact lookup** ($w = 1$) and **synthesis** ($w = 3$).

For $w = 1$, {{eq:span-containment}} is 1 for every $L \geq 1$. Containment is
never binding, so {{eq:chunk-success}} reduces to the dilution term alone and is
**monotonically decreasing in $L$**: the optimum is the smallest chunk that is
still a coherent unit.

For $w = 3$ with no overlap, containment is $\max(0, (L-2)/L)$ — zero at $L \le
2$, and rising toward 1. Multiplied by a decreasing dilution term, this gives an
**interior optimum strictly greater than $w$**.

$$ L^{*}(w=1) = L_{\min}, \qquad L^{*}(w=3) > 3 $$ (eq:optima-separate)

The two do not merely differ in degree; **one is at a boundary and the other is
interior**, so no single number is near-optimal for both. This is why chunk-size
advice is contradictory: each recommendation is correct for the workload its
author had.

### 6.2 What overlap buys, precisely

Fix $L$ and vary $S$. From {{eq:span-containment}} in the partial regime,
containment scales as $1/S$, while {{eq:index-multiplier}} says cost scales as
$L/S$. So per unit of index cost, the marginal return on overlap is

$$ \frac{\partial\,\text{containment}}{\partial\,\text{cost}} \;\propto\; \frac{1}{L} \quad\text{— independent of } S $$ (eq:overlap-return)

**Overlap's cost-effectiveness does not diminish with more overlap**, which is
surprising, and it means the decision is a straight budget question rather than a
search for a knee. It also means overlap is *worse* value at large $L$: the
larger the chunk, the less a given index multiplier buys, because containment was
already high.

The corollary practitioners get wrong: **overlap is nearly worthless when
$L \gg w$**, because {{eq:span-containment}} is already 1. Most systems use large
chunks *and* 20% overlap and are paying 25% of their index for nothing.

### 6.3 Why semantic chunking helps less than it should

Semantic chunking cuts where consecutive sentences become dissimilar, on the
theory that these are natural topic boundaries. It should dominate fixed-size
chunking and in practice the gains are modest. Two reasons worth stating.

**The atoms are already coherent.** Consecutive sentences in written prose are
similar almost everywhere, so the similarity signal is weak and the detected
boundaries are noisy — the method is looking for structure that the writer
already expressed through headings, which {{ch:rag-ingestion}} preserved and a
structural chunker uses directly and reliably.

**It optimises the wrong objective.** Semantic chunking maximises within-chunk
coherence, which is {{eq:chunk-dilution}}'s term only. It has nothing to say
about {{eq:span-containment}}, so it improves precision on $w=1$ queries and can
*hurt* on larger $w$ by cutting exactly at the topic transitions a synthesis
question needs to span.

> **RESEARCH NOTE:** This is a case where a plausible method underperforms for a
> reason visible in the equations. If a technique optimises one factor of
> {{eq:chunk-success}} and ignores the other, expect gains on the query type
> governed by that factor and losses elsewhere — and expect the published
> evaluation to feature the former.

## 7. Internal Mechanics

```mermaid {#fig:chunking-strategies caption="Four strategies over the same document. Fixed and semantic produce one granularity and must choose it; structural inherits boundaries the author wrote; hierarchical refuses the choice and indexes several levels, which is why it sidesteps eq:mixed-workload-optimum rather than solving it."}
flowchart TD
    D["document:<br/>headings, paragraphs, sentences"] --> F["fixed size<br/>L tokens, stride S"]
    D --> ST["structural<br/>cut at headings/paragraphs"]
    D --> SE["semantic<br/>cut where similarity drops"]
    D --> H["hierarchical<br/>index every level"]
    F --> FO["simple; boundaries<br/>ignore meaning"]
    ST --> SO["free structure;<br/>sizes vary wildly"]
    SE --> SEO["coherent chunks;<br/>optimises only dilution"]
    H --> HO["no size to choose;<br/>costs a summarisation pass"]
```

### 7.1 The strategies

**Fixed size.** Split every $S$ tokens. Trivial, predictable sizes, and it cuts
mid-sentence and mid-table. The right baseline and a poor endpoint.

**Recursive.** Try to split on paragraph breaks; if a piece is still too large,
split on sentences; then on words. This is what most libraries do by default and
it is a good default — it respects structure when it can and degrades
predictably.

**Structural.** Cut at the boundaries the author wrote: headings, sections, list
items, table rows. **This is the best default when {{ch:rag-ingestion}} preserved
the structure**, because the boundaries are real rather than inferred, and it is
free. Its problem is size variance — a document with a 20-word section and a
4,000-word section produces chunks that behave completely differently — which is
handled by splitting oversized sections recursively and merging undersized ones.

**Semantic.** Embed each sentence, cut where consecutive similarity drops below a
threshold. Costs an embedding pass at ingest; see {{sec:6-mathematical-foundation}}
for why it disappoints.

**Hierarchical.** Index several granularities at once
({{cite:sarthi2024raptor}}). Sidesteps {{eq:mixed-workload-optimum}} entirely by
declining to choose, at the cost of an LLM summarisation pass over the corpus.
{{ch:rag-advanced-retrieval}} develops this.

### 7.2 What must never be split

Regardless of strategy, some units are atomic because splitting them destroys
{{ch:rag-ingestion}}'s {{eq:table-recoverability}}:

- **A table row**, and often a whole small table with its header.
- **A code block.**
- **A list item**, where the stem is in the preamble.
- **A sentence**, except as a last resort.

The practical rule: **the chunker must consume the structure markers ingestion
produced.** A chunker operating on flat text has no way to know a table row is a
table row, which is why {{ch:rag-ingestion}} insisted on preserving structure and
why the two stages are one design.

### 7.3 What every chunk needs attached

A chunk is retrieved and read *alone*, so anything not attached to it is
unavailable when it matters most:

- **The heading path** — *"Q3 Report → EMEA → Logistics"*. The highest-value
  metadata in the part ({{ch:rag-ingestion}}), because a chunk saying
  "performance improved by 12%" is unusable without it.
- **Document identity and position**, for citation ({{ch:rag-generation}}).
- **Neighbour pointers**, so {{ch:rag-advanced-retrieval}}'s parent–child
  expansion can widen the context after retrieval.
- **Access labels and timestamps**, inherited from ingestion.

**Prepending the heading path to the chunk text before embedding** is a one-line
change that measurably improves retrieval, because it puts the chunk's context
into the vector rather than only into the metadata.

## 8. Implementation

```python {tier=A name=chunk-size-tradeoff}
"""There is no optimal chunk size. There is an optimal chunk size PER QUERY TYPE.

A corpus of documents built from sentences with topic vectors. Two query types:

  fact      -- the answer is in ONE sentence (span w=1)
  synthesis -- the answer needs THREE consecutive sentences (span w=3)

A chunk's embedding is the mean of its sentences' embeddings, which is
eq:chunk-dilution made literal. Retrieval succeeds when a retrieved chunk
CONTAINS the whole answer span -- so the two forces of section 4 are both active
and measurable.
"""
import numpy as np

rng = np.random.default_rng(5)

N_DOC, SENT_PER_DOC, DIM = 150, 48, 48
N_TOPIC, K_RETRIEVE, N_QUERY = 40, 10, 600
CHUNK_SIZES = [1, 2, 3, 4, 6, 8, 12, 16, 24]

topics = rng.normal(size=(N_TOPIC, DIM))
topics /= np.linalg.norm(topics, axis=1, keepdims=True)


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


# Each document drifts slowly through topic space. A sentence is a topic
# component (shared with its neighbours, as in real prose) plus a large
# individual component -- which is what makes a single sentence identifiable,
# and therefore what averaging over a chunk destroys.
sent = np.zeros((N_DOC, SENT_PER_DOC, DIM))
for d in range(N_DOC):
    t = rng.integers(0, N_TOPIC)
    for s in range(SENT_PER_DOC):
        if rng.random() < 0.12:                 # occasional topic shift
            t = rng.integers(0, N_TOPIC)
        sent[d, s] = 0.7 * topics[t] + 0.35 * rng.normal(size=DIM)
sent = unit(sent)


def build_index(chunk_size):
    """Non-overlapping chunks; embedding is the mean of member sentences."""
    vecs, spans = [], []
    for d in range(N_DOC):
        for start in range(0, SENT_PER_DOC, chunk_size):
            end = min(start + chunk_size, SENT_PER_DOC)
            vecs.append(sent[d, start:end].mean(axis=0))
            spans.append((d, start, end))
    return unit(np.array(vecs)), spans


def evaluate(chunk_size, width):
    """Success = some retrieved chunk contains the ENTIRE answer span."""
    vecs, spans = build_index(chunk_size)
    hits = 0
    for _ in range(N_QUERY):
        d = int(rng.integers(0, N_DOC))
        s0 = int(rng.integers(0, SENT_PER_DOC - width + 1))
        target = range(s0, s0 + width)
        # The query looks like the answer span -- a noisy version of its mean.
        q = unit(sent[d, s0:s0 + width].mean(axis=0)
                 + rng.normal(scale=0.12, size=DIM))
        top = np.argpartition(-(vecs @ q), K_RETRIEVE)[:K_RETRIEVE]
        for i in top:
            cd, cs, ce = spans[i]
            if cd == d and cs <= target[0] and target[-1] < ce:
                hits += 1
                break
    return hits / N_QUERY


print(f"{'chunk size':>12}{'chunks':>9}{'fact (w=1)':>13}{'synthesis (w=3)':>18}")
print("-" * 54)
results = {}
for L in CHUNK_SIZES:
    n_chunks = len(build_index(L)[1])
    fact = evaluate(L, 1)
    synth = evaluate(L, 3)
    results[L] = (fact, synth)
    print(f"{L:>12}{n_chunks:>9}{fact:>13.3f}{synth:>18.3f}")

best_fact = max(results, key=lambda L: results[L][0])
best_synth = max(results, key=lambda L: results[L][1])
cost_at_synth_opt = results[best_fact][0] - results[best_synth][0]
cost_at_fact_opt = results[best_synth][1] - results[best_fact][1]
print(f"""
Optimal chunk size for FACT queries:      {best_fact} sentences
Optimal chunk size for SYNTHESIS queries: {best_synth} sentences

Cost of using the synthesis optimum for fact queries:      -{cost_at_synth_opt:.3f}
Cost of using the fact optimum for synthesis queries:      -{cost_at_fact_opt:.3f}

These are the two forces of section 4, separated and measured. Fact queries need
one sentence, so eq:span-containment is satisfied at every size and only dilution
matters -- the curve falls monotonically and the optimum sits at the smallest
usable chunk. Synthesis queries need three consecutive sentences, so small chunks
CANNOT contain the answer at all and the containment term dominates until the
chunk is comfortably larger than the span.

That is eq:optima-separate: one optimum is at a boundary and the other is
interior. They are not close, and no single number is near-optimal for both --
read the two cost lines above. Serving fact queries at the synthesis optimum
gives up roughly half of them; serving synthesis queries at the fact optimum
gives up ALL of them, because a one-sentence chunk cannot contain a
three-sentence answer at any retrieval depth.

That asymmetry is worth noticing on its own. Choosing too LARGE degrades
gracefully; choosing too SMALL fails absolutely for any query whose answer is
wider than the chunk. When you must guess, guess large.

This is why every published chunk-size recommendation disagrees with every other
one, and why all of them are defensible. Each is correct for the query
distribution its author had. Asking "what chunk size should I use" without
stating the query mix is asking an under-specified question, and the answer you
get back will be someone else's workload.""")
```

```python {tier=A name=chunk-overlap}
"""What overlap buys, and what it costs.

Overlap hedges against answers that straddle a chunk boundary. eq:span-containment
predicts the benefit and eq:index-multiplier the cost; here both are measured
against each other so the trade is visible rather than assumed.

The result worth watching is the interaction: overlap's value depends almost
entirely on how large the chunk already is relative to the answer span.
"""
import numpy as np

rng = np.random.default_rng(11)

N_TRIALS = 20_000
SENT_PER_DOC = 48
CONFIGS = [(4, 4), (4, 3), (4, 2), (4, 1),
           (8, 8), (8, 6), (8, 4), (8, 2),
           (16, 16), (16, 12), (16, 8)]


def containment_rate(L, S, w, trials=N_TRIALS):
    """Fraction of random w-spans fully inside at least one chunk (L, stride S)."""
    starts = np.arange(0, SENT_PER_DOC, S)
    hits = 0
    for _ in range(trials):
        s0 = int(rng.integers(0, SENT_PER_DOC - w + 1))
        if any(cs <= s0 and s0 + w - 1 < min(cs + L, SENT_PER_DOC) for cs in starts):
            hits += 1
    return hits / trials


print(f"{'chunk L':>9}{'stride S':>10}{'overlap':>9}{'index x':>9}"
      + "".join(f"{'w=' + str(w):>8}" for w in (1, 3, 6)))
print("-" * 61)
rows = {}
for L, S in CONFIGS:
    overlap = 1 - S / L
    mult = L / S
    rates = [containment_rate(L, S, w) for w in (1, 3, 6)]
    rows[(L, S)] = (mult, rates)
    print(f"{L:>9}{S:>10}{overlap:>8.0%}{mult:>8.2f}x"
          + "".join(f"{r:>8.3f}" for r in rates))

r = {(L, S): rows[(L, S)][1] for L, S in CONFIGS}
print(f"""
Read the w=1 column first: every configuration is 1.000. A single-sentence answer
is inside SOME chunk no matter how you cut, so for fact-lookup queries overlap
buys exactly nothing and costs the index multiplier in full. Systems with a
default 20% overlap serving a mostly fact-lookup workload are paying a fifth of
their index for zero benefit -- and that describes a great many systems.

Now the w=6 column at L=4: every entry is 0.000, including the one with 75%
overlap and a 4x index. OVERLAP CANNOT COMPENSATE FOR A CHUNK SMALLER THAN THE
ANSWER SPAN. No stride makes a four-sentence window contain six sentences. If any
material share of your queries needs a span wider than your chunk, they are not
merely retrieved poorly -- they are unanswerable, and no retrieval depth, no
overlap, and no reranker changes that.

Next, chunk size against overlap as substitutes. At w=3 with NO overlap,
L=4 gives {r[(4, 4)][1]:.3f} and L=16 gives {r[(16, 16)][1]:.3f} -- the larger
chunk contains the span far more often at an index multiplier of 1.00x, i.e. for
free. Buying containment with size is strictly cheaper than buying it with
overlap, whenever the dilution cost of the larger chunk is acceptable.

Finally, when is overlap worth its cost? Compare the same 25% overlap applied at
two sizes, at w=3: it adds {r[(4, 3)][1] - r[(4, 4)][1]:+.3f} at L=4 and only
{r[(16, 12)][1] - r[(16, 16)][1]:+.3f} at L=16, because containment at L=16 was
already {r[(16, 16)][1]:.3f} and there was little left to buy. But at w=6 the
same comparison reverses: {r[(8, 6)][2] - r[(8, 8)][2]:+.3f} at L=8 against
{r[(16, 12)][2] - r[(16, 16)][2]:+.3f} at L=16.

So the rule is not about L. It is about L/w: overlap is worth most when
containment is in the partial regime of eq:span-containment and worth nearly
nothing once L is comfortably above w. Which means you cannot choose an overlap
without knowing w -- and a fixed 20% default, applied without measuring the
answer-span width, is as likely to be wasted as well spent.""")
```

## 9. Practical Example

**Chunk size.** The two query types have different optima and they are not close.
Fact queries ($w=1$) fall monotonically with chunk size — {{eq:span-containment}}
is satisfied everywhere, so only dilution operates, and the best chunk is the
smallest coherent one. Synthesis queries ($w=3$) cannot be answered at all by
chunks too small to contain the span, so containment dominates until the chunk is
comfortably larger than the answer, giving an interior optimum.

This is {{eq:optima-separate}} measured: **one optimum at a boundary, one in the
interior.** Fact retrieval falls monotonically from 1.000 at one sentence to
0.317 at twenty-four; synthesis is **0.000** below three sentences, peaks at 0.588
around six, and declines slowly after.

Note the asymmetry in how the compromise fails. Serving fact queries at the
synthesis optimum costs about half of them. Serving synthesis queries at the fact
optimum costs **all** of them — a one-sentence chunk cannot contain a
three-sentence answer at any retrieval depth, so no amount of $k$ recovers it.
**Too large degrades gracefully; too small fails absolutely.** When you must
guess, guess large.

**That is the answer to "what chunk size should I use": the question is
under-specified.** Every published recommendation is correct for the query
distribution its author had, which is why they disagree and why all of them are
defensible. If you take one thing from this chapter, take the habit of asking
what $w$ looks like in your query log before choosing $L$.

**Overlap.** Four findings, and three of them cut against standard practice.

At $w=1$ every configuration achieves perfect containment, so **overlap buys
literally nothing for fact-lookup queries** while costing the full index
multiplier.

**Overlap cannot compensate for a chunk smaller than the answer span.** At $L=4$
and $w=6$, every configuration scores 0.000 — including 75% overlap at a 4×
index. Those queries are not retrieved poorly; they are *unanswerable*, and no
overlap, retrieval depth, or reranker changes it. This is the sharpest form of
the asymmetry: undersizing is not a degradation, it is a wall.

**Chunk size substitutes for overlap and is free.** At $w=3$ with no overlap,
$L=4$ contains the span 0.527 of the time and $L=16$ contains it 0.913 — a large
gain at an index multiplier of 1.00×.

And **overlap's value depends on $L/w$, not on $L$.** The same 25% overlap adds
+0.144 at $L=4$ and only +0.087 at $L=16$ for $w=3$, because containment at
$L=16$ was already 0.913 — but at $w=6$ the comparison reverses, adding +0.096 at
$L=8$ and +0.159 at $L=16$. Overlap is worth most in {{eq:span-containment}}'s
partial regime and nearly nothing once $L$ is comfortably above $w$.

**Which means you cannot choose an overlap without knowing $w$.** A fixed 20%
default, applied without measuring the answer-span width, is as likely to be
wasted as well spent.

> **PRODUCTION TIP:** Before setting overlap, estimate $w$ — the number of
> consecutive sentences a typical answer needs — from twenty real questions. If
> $w$ is 1 for most of them, set overlap to zero and spend the index budget on
> {{ch:rag-advanced-retrieval}}'s parent–child expansion instead, which achieves
> the same thing at retrieval time rather than index time.

## 10. Production Considerations

**Measure $w$ before choosing $L$.** Twenty real questions, and count the
consecutive sentences each answer needs. It is the parameter both equations turn
on and almost nobody has it.

**Prepend the heading path to the chunk text before embedding.** One line,
measurable improvement, and it fixes the "12% improvement" problem at its source.

**Never split a table row, code block, or list item.** Requires the chunker to
consume {{ch:rag-ingestion}}'s structure markers.

**Handle size variance in structural chunking** — recursively split oversized
sections, merge undersized ones into their neighbours. A 15-token chunk and a
4,000-token chunk in one index behave completely differently under
{{eq:chunk-dilution}}.

**Store chunk provenance**: document, character offsets, neighbours. Offsets are
what let a citation point at a span a human can verify, and neighbours are what
{{ch:rag-advanced-retrieval}} expands.

**Re-chunking invalidates every embedding.** Treat chunk configuration as part of
the index schema alongside the embedding model ({{ch:emb-what-they-are}}), and
version it the same way.

**Log chunk-size distribution and boundary statistics**, not just a count. The
tails are where the failures are.

**Set the chunk size from the widest span you must serve, not the median.**
{{sec:9-practical-example}}'s asymmetry makes this the correct default:
overshooting costs precision smoothly, while undershooting makes a query class
unanswerable at any retrieval depth. If 10% of questions need four sentences and
90% need one, sizing for the 90% does not degrade the 10% — it deletes them. Size
for the wide case and recover precision with {{ch:rag-advanced-retrieval}}'s
parent–child retrieval, which gets both without the compromise.

**Chunk heterogeneous corpora heterogeneously.** Nothing requires one
configuration for one index. A codebase chunked by function, a policy manual by
section, and a ticket archive not at all can share an index perfectly well, and
{{eq:mixed-workload-optimum}}'s compromise loss is entirely avoidable when the
sub-corpora have different natural units. Teams apply one splitter to everything
because the library encourages it, not because it is right.

## 11. Common Mistakes

**Copying a chunk size from a tutorial.** {{eq:mixed-workload-optimum}}: it
encodes someone else's query distribution.

**Overlap by default without measuring $w$.** {{sec:9-practical-example}} shows it
frequently buys nothing.

**Splitting on character count without respecting sentences.** Produces chunks
starting mid-word, which damages both embedding and readability.

**Chunking flat text when structure was available.** {{ch:rag-ingestion}}
preserved it; use it.

**Forgetting the heading path.** The single most common omission and the cheapest
fix.

**One chunk size for a heterogeneous corpus.** A codebase and a policy manual
want different treatment; nothing forces one configuration.

**Assuming semantic chunking is strictly better.**
{{sec:6-mathematical-foundation}} gives the reason it is not.

**Treating re-chunking as a config change.** It is a full re-embed.

## 12. Failure Modes

**Straddled answers.** The answer spans a boundary and no chunk contains it.
Symptom: the retriever returns a chunk adjacent to the answer, repeatedly.
Diagnostic: check whether the gold sentence's *neighbour* was retrieved.

**Dilution misses.** The relevant sentence is buried in a large chunk and never
scores highly. Symptom: recall improves markedly when chunk size is reduced —
which is the test.

**Orphaned chunks.** A chunk that is meaningless alone — "as shown above, this
confirms the hypothesis". Endemic without heading paths, and a
{{ch:rag-generation}} failure that looks like a retrieval failure.

**Table row splitting.** {{ch:rag-ingestion}}'s failure, arriving one stage
later.

**Size explosion in structural chunking.** One document with no headings becomes
one 40,000-token chunk that is retrieved for everything and fits in no context.

**Duplicate crowding from overlap.** With high overlap, the top-$k$ can be five
overlapping windows of the same passage — {{eq:duplicate-slot-loss}} again,
self-inflicted. Deduplicate overlapping retrievals before assembling the context.

**Silent re-chunk drift.** A library upgrade changes the default splitter,
chunk boundaries move, and stored citation offsets become wrong.

## 13. Alternatives

**Hierarchical indexing** ({{cite:sarthi2024raptor}}). Index several
granularities and let retrieval choose. **The principled answer to
{{eq:mixed-workload-optimum}}** — it declines the compromise rather than
optimising it — at the cost of a summarisation pass over the corpus.
{{ch:rag-advanced-retrieval}}.

**Parent–child.** Retrieve small chunks for precision, expand to their parents
for context before generation. **Gets both sides of {{eq:chunk-success}} without
choosing**, and it is the highest-value technique in this part relative to its
complexity.

**Late interaction** ({{cite:khattab2020colbert}}). Skip chunking; keep one
vector per token. Removes the problem at 10–100× storage.

**No chunking.** For short documents — support tickets, product entries, emails —
one document is one chunk and the whole chapter is unnecessary.

**Propositional indexing.** Decompose text into atomic standalone facts with an
LLM at ingest. Excellent for $w=1$ retrieval, expensive, and it loses the
narrative connections synthesis needs.

**Contextual augmentation.** Prepend an LLM-generated summary of the document to
each chunk before embedding. A cheaper cousin of hierarchical indexing;
{{ch:rag-advanced-retrieval}}.

## 14. Evaluation

**Retrieval success by query type**, never in aggregate. Aggregate hides
{{eq:optima-separate}} completely — it is the average of two curves with
different optima and it is a good approximation to neither.

**Answer-span width $w$** as a distribution over your query log. It is the
parameter of the whole chapter.

**Boundary-straddle rate** — how often the gold answer crosses a chunk boundary.
Directly measures what overlap would buy.

**Chunk-size distribution**, including the tails.

**Orphan rate** — the fraction of chunks that are not interpretable alone.
Sample fifty and read them; it is the same hour {{ch:rag-ingestion}} asked for
and it pays twice.

**Index multiplier** ({{eq:index-multiplier}}), reported next to the benefit it
buys, so overlap is a decision rather than a default.

## 15. Advanced Concepts

**The retrieval unit and the generation unit need not be the same**, and
separating them dissolves the chapter's central tension. Retrieve a sentence for
precision, generate from its section for context. Parent–child is the simple
form, hierarchical indexing the general one, and once you see this the search for
a single optimal $L$ looks like a self-imposed constraint.

**Chunking is a lossy compression of document structure**, exactly as an
embedding is a lossy compression of meaning. The same framing applies: what you
discard determines what becomes unanswerable, and no downstream component
recovers it.

**{{eq:chunk-dilution}} is the mean-pooling assumption.** With late interaction
there is no averaging and dilution vanishes — which is the precise sense in which
{{cite:khattab2020colbert}} makes chunk size irrelevant, and the precise reason it
costs so much storage.

**Query-dependent chunking is possible and rare.** Nothing forces one chunking;
an index could hold several and route by predicted $w$. This is
{{ch:llm-routing}}'s argument once more, and it is barely explored.

**The atom matters more than the size.** Sentences are the usual atom because
they are the smallest self-contained unit of prose. For code the atom is a
function; for a table, a row; for a chat log, a turn. **Choosing the wrong atom
cannot be fixed by choosing a good size**, and most chunking discussions argue
about size while assuming the atom.

## 16. Connection to Previous Chapters

{{ch:emb-reranking}}'s single-vector bottleneck is what chunking exists to work
around, and {{eq:chunk-dilution}} is {{eq:bottleneck}} with the chunk length made
explicit. {{ch:rag-ingestion}}'s structure markers are what a good chunker
consumes — the two stages are one design. {{ch:emb-what-they-are}}'s
schema-versioning point applies to chunk configuration, since re-chunking is a
full re-embed. {{ch:rag-why}}'s context budget is what chunk size and $k$ jointly
spend. And {{ch:emb-vector-db}}'s duplicate-crowding equation reappears as
self-inflicted damage when overlap is high.

## 17. Exercises

1. Derive {{eq:span-containment}} for the no-overlap case and confirm that
   $L \geq w$ is insufficient.
2. From {{eq:chunk-dilution}}, at what $L$ does the signal term fall below the
   background term for $\bar{s} = 0.3$ and a gold similarity of 0.8?
3. In `chunk-size-tradeoff`, add a $w=6$ query type. Predict where its optimum
   lies relative to $w=3$, then check.
4. Modify the same listing to use overlapping chunks and confirm
   {{eq:span-containment}}'s stride dependence.
5. Weight the two query types 80/20 and compute {{eq:mixed-workload-optimum}}.
   How much worse is the compromise than each type's own optimum?
6. In `chunk-overlap`, find the configuration maximising $w=3$ containment per
   unit of index multiplier. Is it the highest-overlap one?
7. Implement structural chunking over a Markdown document and report the size
   distribution. What do you do about the tails?
8. Design the boundary-straddle measurement for a live system, given you have
   query logs and gold answers but not sentence-level labels.

## 18. Interview Questions

1. What chunk size should I use? (The correct answer is a question.)
2. Why do small chunks retrieve better?
3. Why do large chunks answer better?
4. What does overlap buy, and when does it buy nothing?
5. Why is semantic chunking less of a win than it sounds?
6. What must never be split, and why?
7. What metadata belongs on a chunk?
8. Your retriever returns chunks adjacent to the answer. Diagnose.
9. How would you avoid choosing a chunk size at all?
10. You change the chunk size. What else has to happen?

## 19. Research Questions

1. Can $w$ be estimated automatically from a query log without gold spans?
   Everything in this chapter turns on it and nobody measures it.
2. Is there a chunking objective that optimises {{eq:chunk-success}} directly,
   rather than one of its factors as semantic chunking does?
3. Query-dependent chunk selection from a multi-granularity index — how much of
   {{eq:mixed-workload-optimum}}'s compromise loss does it recover, and can the
   granularity be predicted before retrieval?
4. Propositional indexing is excellent for $w=1$ and loses narrative structure.
   Is there a decomposition that preserves both?
5. Chunking and ingestion are one design and two stages. What does a joint
   formulation look like, and does it beat the pipeline?

## 20. Chapter Summary

Chunking is **the choice of retrieval unit**, forced by the single-vector
bottleneck: a document with several meanings cannot be one vector, so either keep
many vectors (expensive) or cut the document up (free). Chunking is the second,
and it converts "how do I represent this document" into "where do I cut".

Two forces set the size. **Dilution** ({{eq:chunk-dilution}}) degrades signal as
$1/L$ and pushes chunks smaller. **Containment** ({{eq:span-containment}}) requires
$L \gtrsim 2w$ for an answer spanning $w$ atoms and pushes them larger. Their
product ({{eq:chunk-success}}) has an interior maximum whose location depends on
$w$ — **and $w$ is a property of the query, not the corpus.**

Measured, the optima for fact-lookup and synthesis queries are not close: one
sits at the smallest coherent chunk and the other in the interior
({{eq:optima-separate}}). **So "what chunk size should I use" is
under-specified**, every published recommendation is correct for its author's
workload, and the useful habit is to measure $w$ from twenty real questions
before choosing $L$.

Overlap is a hedge whose value was overstated. It buys **nothing** at $w=1$ while
costing the full index multiplier; it buys nothing at all when $L < w$, where
containment is 0.000 at any stride; chunk size substitutes for it for free; and
its value depends on $L/w$ rather than $L$ — largest in
{{eq:span-containment}}'s partial regime, negligible once $L$ is comfortably
above $w$. **A fixed 20% default chosen without measuring $w$ is as likely to be
wasted as well spent.**

The way out is to stop choosing. Parent–child retrieval and hierarchical indexing
({{cite:sarthi2024raptor}}) decouple the retrieval unit from the generation unit
and decline {{eq:mixed-workload-optimum}}'s compromise instead of optimising it —
which is {{ch:rag-advanced-retrieval}}'s subject and the best answer this part
has.

## 21. Further Reading

{{cite:sarthi2024raptor}} for hierarchical indexing as the principled response to
this chapter's central trade-off — Sections 3 and 4.
{{cite:khattab2020colbert}} for the alternative that removes chunking entirely,
and for the storage cost of doing so.
{{cite:gao2023ragsurvey}} for the standard taxonomy of chunking strategies, which
is a good map and does not address the size question.
{{cite:liu2023lost}} matters here indirectly: chunk size determines how many
chunks fill the context, which determines where in it the good one sits.
{{cite:lewis2020rag}} used 100-word passages throughout, and a surprising amount
of the field's default sizing traces to that choice rather than to any
measurement.

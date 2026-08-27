---
id: rag-advanced-retrieval
number: 112
part: XII
tier: full
status: draft
requires: [rag-chunking, rag-generation, rag-query-understanding, rag-indexing,
           emb-reranking, emb-hybrid]
provides: [parent-child-retrieval, retrieval-generation-decoupling,
           contextual-chunk-augmentation, hierarchical-retrieval,
           sentence-window, orphan-chunks, multi-granularity-index]
citations: [sarthi2024raptor, khattab2020colbert, liu2023lost, gao2023ragsurvey,
            cormack2009rrf, lewis2020rag]
---

## 1. Learning Objectives

By the end of this chapter you will be able to state the one idea that unifies
every technique here — **the retrieval unit and the generation unit need not be
the same object** — and derive parent–child, sentence-window, and hierarchical
retrieval from it; measure a parent–child index against flat chunking on both
query types that {{ch:rag-chunking}} showed have incompatible optima; explain and
fix orphaned chunks with contextual augmentation; and price each technique in
index size, ingest cost, and latency.

## 2. Why This Matters

{{ch:rag-chunking}} ended in a genuine dilemma. Small chunks retrieve precisely
and answer incompletely; large chunks answer completely and retrieve imprecisely;
the two query types have optima that are not close, and
{{eq:mixed-workload-optimum}}'s compromise satisfies nobody.

That dilemma exists **only because one object was doing two jobs.** A chunk is
what gets embedded and searched, and it is also what gets pasted into the prompt.
Nothing requires those to be the same text.

Once separated, the dilemma dissolves:

> **Retrieve the sentence. Generate from the section.**

Precision from the small unit, completeness from the large one, no compromise.
{{sec:9-practical-example}} measures this against both flat baselines and it beats
both on both query types — which is unusual enough to be worth checking, and it
is why this is the highest-value chapter in the part relative to its complexity.

{{maturity:MATURE}} Parent–child and sentence-window retrieval are standard,
simple, and under-used. {{maturity:EMERGING}} Hierarchical summarisation indexes
are effective and carry an ingest cost that papers rarely price.

## 3. Prerequisites

{{ch:rag-chunking}} for the dilemma this chapter resolves and for
{{eq:chunk-dilution}} and {{eq:span-containment}}; {{ch:rag-generation}} for the
context budget the expanded unit spends; {{ch:emb-reranking}}'s single-vector
bottleneck; {{ch:rag-indexing}} for the neighbour pointers that make expansion
possible; {{ch:rag-query-understanding}} for what arrives at retrieval.

## 4. Intuitive Explanation

### One object, two jobs

The retrieval unit must be **small and specific**, because
{{eq:chunk-dilution}} says a chunk's embedding is roughly the average of its
content, so a relevant sentence buried in a long passage contributes little to the
vector.

The generation unit must be **large and complete**, because a sentence read alone
is frequently meaningless — *"this improved throughput by 12%"* — and because
{{eq:span-containment}} says an answer spanning several sentences must arrive
whole.

**These are opposite requirements and there is no reason to satisfy them with one
piece of text.** Index the sentence; when it is retrieved, look up the paragraph
or section it came from and send *that* to the model.

```text
   INDEXED (small)                    SENT TO THE MODEL (large)
   ───────────────────────            ─────────────────────────────
   "throughput improved 12%"    →     [full section: depot consolidation,
                                       the metric definition, the caveat,
                                       and the 12% figure in context]
```

The retrieval decision was made on a precise vector; the generation was given
enough to be right. The only new requirement is a pointer from each chunk to its
parent, which {{ch:rag-chunking}} already asked you to store.

### Orphaned chunks

The second problem this chapter fixes, and the more common one in practice.

A chunk saying *"as noted above, this reduced latency substantially"* is
**unretrievable and unusable**: unretrievable because its embedding contains no
topic, and unusable because a reader cannot tell what "this" refers to. Split any
document at a fixed size and a large fraction of chunks look like this.

The fix is to **give the chunk its context before embedding it**: prepend the
heading path ({{ch:rag-ingestion}}), or a short document-level summary, or both.
The chunk's vector then carries the topic even though its text does not, and
{{sec:9-practical-example}} measures how much that recovers.

**This is the cheapest technique in the chapter and the one most often skipped**,
because the change is one line of string concatenation and does not look like
engineering.

### The unifying frame

Every technique here is one choice of *what to index* against *what to send*:

| Technique | Indexed | Sent |
|---|---|---|
| flat chunking | the chunk | the same chunk |
| sentence-window | a sentence | the sentence ± $n$ neighbours |
| parent–child | a small chunk | its parent section |
| contextual augmentation | chunk + prepended context | the chunk |
| hierarchical | every level | the level that matched |
| late interaction | every token | the document |

Read the last row and notice that {{ch:emb-reranking}}'s ColBERT is the extreme
point of the same axis — index at maximum granularity, generate from the whole —
which is why it costs what it costs.

## 5. Formal Explanation

### 5.1 Decoupling

Let $\text{idx}(\cdot)$ be the text embedded and $\text{gen}(\cdot)$ the text sent
to the model. Flat chunking asserts

$$ \text{idx}(c) = \text{gen}(c) = c $$ (eq:flat-coupling)

and every technique in this chapter relaxes it. Retrieval quality now depends on
$\text{idx}$ and answer quality on $\text{gen}$:

$$ \Prob[\text{answerable}] = \underbrace{\Prob\big[c \text{ retrieved} \mid \text{idx}(c)\big]}_{\text{wants small}} \times \underbrace{\Prob\big[\text{answer} \subseteq \text{gen}(c)\big]}_{\text{wants large}} $$ (eq:decoupled-success)

Compare {{eq:chunk-success}}, where both factors were driven by the same $L$ and
therefore traded against each other. **Here they are independent and can be
maximised separately** — which is the entire content of the chapter, and the
reason the result in {{sec:9-practical-example}} is not merely a better
compromise but a strict improvement.

### 5.2 Parent–child

Index children of size $L_c$, send parents of size $L_p \gg L_c$:

$$ \text{recall} \approx R(L_c) \big|_{\text{small}}, \qquad \text{containment} \approx \Prob[\text{span} \subseteq \text{parent}] \big|_{L_p} $$ (eq:parent-child-factors)

The costs, and they are modest:

- **Index size** is set by $L_c$ — more chunks than a flat index at $L_p$, the
  same as a flat index at $L_c$.
- **Context tokens** are set by $L_p$, so a retrieved set of $k$ children costs
  up to $k L_p$ tokens rather than $k L_c$. **This is the real price**, and it
  interacts with {{ch:rag-generation}}'s budget.
- **Deduplication becomes mandatory**: several retrieved children frequently
  share one parent, and sending it repeatedly wastes the budget on the same text.

$$ \text{effective context} = \sum_{p \in \text{unique parents}} L_p \;\ll\; k L_p \quad \text{when children cluster} $$ (eq:parent-dedup-saving)

Child clustering is *good news* — several retrieved children in one parent is
evidence the parent is relevant — and it makes the token cost far lower than the
naive bound suggests.

### 5.3 Contextual augmentation

Embed the chunk with context prepended, send the chunk alone:

$$ \text{idx}(c) = \text{context}(c) \Vert c, \qquad \text{gen}(c) = c $$ (eq:contextual-augmentation)

The effect on the embedding, under {{eq:chunk-dilution}}'s mean-pooling model with
context of length $L_x$:

$$ \hat{f}(\text{idx}(c)) \approx \frac{L_c}{L_c + L_x}\hat{f}(c) + \frac{L_x}{L_c + L_x}\hat{f}(\text{context}) $$ (eq:augmentation-mix)

**This cuts both ways and the trade is the whole design question.** For an orphan
chunk whose own text carries no topic, the context term supplies one and
retrieval becomes possible at all. For a chunk that was already specific, the
context term *dilutes* it — and if every chunk in a document gets the same
prepended text, every chunk in that document moves toward a common point, which
is {{ch:emb-what-they-are}}'s anisotropy arriving by choice rather than by
accident.

$$ L_x \ll L_c \quad\text{is the design constraint} $$ (eq:context-length-constraint)

A heading path costs a handful of tokens and satisfies it. A 200-token document
summary prepended to a 100-token chunk does not, and is a common way to make
retrieval worse while believing you improved it.

### 5.4 Hierarchical retrieval

{{cite:sarthi2024raptor}} generalises: cluster chunks, summarise each cluster,
cluster the summaries, and index **every level**. A query then matches at
whatever granularity suits it — a detail at the leaves, a theme at the root.

$$ \mathcal{I} = \bigcup_{\ell=0}^{L} \mathcal{I}_\ell, \qquad |\mathcal{I}| \approx |\mathcal{I}_0| \cdot \frac{1}{1 - 1/b} $$ (eq:hierarchical-index-size)

for branching factor $b$ — so a tree over leaves costs about $b/(b-1)$ times the
leaf index, which for $b = 5$ is 25% more. **The index cost is small; the ingest
cost is not**, since building it requires an LLM summarisation call per internal
node:

$$ C_{\text{build}} \approx \frac{N_{\text{chunks}}}{b - 1} \times C_{\text{summarise}} $$ (eq:hierarchical-build-cost)

This is the number papers omit, and it is the number that decides whether the
technique is affordable for a given corpus. It is also a **recurring** cost,
because a document edit invalidates every summary above it.

## 6. Mathematical Foundation

### 6.1 Why parent–child beats both flat baselines

Not obvious, and worth deriving. Flat chunking at size $L$ achieves
{{eq:chunk-success}}'s product with both factors tied to $L$:

$$ S_{\text{flat}}(L) = R(L) \cdot C(L), \qquad R \searrow L,\; C \nearrow L $$ (eq:flat-tradeoff)

so $\max_L S_{\text{flat}}$ is an interior compromise. Parent–child achieves

$$ S_{\text{pc}} = R(L_c) \cdot C(L_p) $$ (eq:parent-child-product)

with $L_c$ and $L_p$ chosen independently. Since $R$ is maximised at small $L$ and
$C$ at large $L$:

$$ S_{\text{pc}} = \max_{L_c} R(L_c) \cdot \max_{L_p} C(L_p) \;\geq\; \max_L \big[R(L) C(L)\big] = S_{\text{flat}} $$ (eq:pc-dominance)

**The product of the maxima is at least the maximum of the product**, with
equality only if both are maximised at the same $L$ — which
{{ch:rag-chunking}} measured and found they are not.

> **MATH NOTE:** {{eq:pc-dominance}} assumes $R$ and $C$ are *separable* — that
> the child size affects only retrieval and the parent size only containment.
> That assumption has a limit, and {{sec:9-practical-example}} finds it. $R(L_c)$
> is not monotonically decreasing in $L_c$ for every query: a query about a
> six-sentence span is itself a *broad* object, and a two-sentence child is a
> poor key for it. **The child must be large enough to be a good key at the
> query's granularity**, which is a coupling {{eq:pc-dominance}} does not model
> — and it is the argument for indexing at several child sizes rather than one
> ({{sec:5-formal-explanation}}'s hierarchical case).

### 6.2 The token cost, honestly

{{eq:pc-dominance}} is a quality argument and ignores the budget. Parent–child
spends more context per retrieved item, so at a fixed budget it retrieves fewer:

$$ k_{\text{pc}} = \frac{B}{\bar{L}_p} \;<\; \frac{B}{L_c} = k_{\text{flat}} $$ (eq:parent-child-budget)

with $\bar{L}_p$ the *deduplicated* parent cost of {{eq:parent-dedup-saving}}.
The comparison is therefore only fair at equal budget — and
{{sec:9-practical-example}} runs it that way, because a technique that wins by
spending more tokens has not been shown to win.

The saving grace is {{ch:rag-generation}}'s {{eq:marginal-chunk-value}}: the
marginal value of the $k$-th chunk is small and eventually negative, so trading
several mediocre chunks for one complete one is usually favourable **on both
axes**.

### 6.3 When augmentation helps and when it hurts

From {{eq:augmentation-mix}}, augmentation improves retrieval of chunk $c$ when
the context's contribution to the query similarity exceeds what it displaces:

$$ \frac{L_x}{L_c + L_x}\big[s(q, \text{context}) - s(q, c)\big] > 0 \iff s(q, \text{context}) > s(q, c) $$ (eq:augmentation-condition)

**So augmentation helps exactly the chunks whose own text is less on-topic than
their document is** — orphans, continuations, tables of numbers, code fragments.
And it *hurts* chunks that are more specific than their document, which are
precisely the highly-informative ones.

The resolution is not to choose but to note that $L_x$ scales the effect: with
$L_x \ll L_c$ the harm is bounded and the benefit to orphans is still large,
because for an orphan $s(q, c) \approx 0$ and any positive context term is a
gain. {{eq:context-length-constraint}}, derived rather than asserted.

## 7. Internal Mechanics

```mermaid {#fig:decoupled-retrieval caption="The one idea, drawn. Everything indexed is small and specific; everything sent is large and complete; a pointer connects them. The dashed path is deduplication, which is not optional once several children can share a parent."}
flowchart LR
    D["document"] --> P["parents:<br/>sections / paragraphs"]
    P --> C["children:<br/>sentences / small chunks"]
    C -->|"embed the CHILD"| I[("index")]
    Q["query"] --> I
    I -->|"top-k children"| L["look up parents<br/>via pointer"]
    L -.->|"dedupe: children<br/>often share a parent<br/>(eq:parent-dedup-saving)"| L
    L -->|"send the PARENT"| G["generation"]
    P -.->|"stored, not indexed"| L
```

### 7.1 Sentence-window: the cheapest version

Index each sentence; on retrieval send the sentence plus $n$ neighbours on each
side. Requires no parent structure at all — just the position pointers
{{ch:rag-chunking}} already asked for — and it is the right first thing to try
because it is roughly ten lines.

The window size $n$ is {{eq:span-containment}}'s $w$ made explicit: set it from
the answer-span width you measured, and note that a **symmetric** window is the
right default because answers may extend in either direction from the sentence
that matched.

### 7.2 Parent selection

Which parent to send is a real choice with a cost profile:

| Parent | Typical size | When |
|---|---|---|
| the paragraph | 100–200 tokens | dense corpora, tight budgets |
| the section | 300–800 tokens | the usual default |
| the document | unbounded | only for short documents |

**Sending the whole document is the failure mode to watch for**: it works in
testing on short documents and blows the budget on the first long one, and the
symptom is a truncation ({{ch:rag-generation}}) rather than an error.

A useful refinement is a **size-capped parent**: expand from the child outward
until a token limit, respecting section boundaries. It bounds the cost without
requiring a uniform parent size, which is what makes it work on heterogeneous
corpora.

### 7.3 What to prepend, in order of value

For {{eq:contextual-augmentation}}, ranked by benefit per token:

1. **The heading path** — a handful of tokens, present from
   {{ch:rag-ingestion}}, and it resolves most orphans. Always do this.
2. **The document title.**
3. **A one-sentence document summary** — an LLM call per *document*, not per
   chunk, so the cost is modest.
4. **An LLM-generated, chunk-specific context sentence** — "This section of the
   Q3 EMEA report discusses depot consolidation." Best quality, and it costs a
   call per *chunk*, which is the expensive tier.

**Stop at the level your measurement justifies.** The first item is nearly free
and captures most of the benefit; the fourth is a large ingest bill and should be
adopted only after the cheaper tiers are shown insufficient.

## 8. Implementation

```python {tier=A name=parent-child-retrieval}
"""Decoupling the retrieval unit from the generation unit.

ch:rag-chunking found that fact-lookup and synthesis queries have incompatible
optimal chunk sizes, so a single flat size is a compromise. Parent-child breaks
the tie: embed SMALL children for precision, send their LARGE parent for
completeness (eq:decoupled-success).

eq:pc-dominance predicts this dominates flat chunking rather than trading against
it. The comparison is run at an EQUAL CONTEXT BUDGET, because a technique that
wins by spending more tokens has not been shown to win.
"""
import numpy as np

rng = np.random.default_rng(5)

N_DOC, SENT_PER_DOC, DIM = 150, 48, 48
N_TOPIC, N_QUERY = 40, 600
BUDGET = 24                     # sentences of context, identical for all strategies

topics = rng.normal(size=(N_TOPIC, DIM))
topics /= np.linalg.norm(topics, axis=1, keepdims=True)


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


sent = np.zeros((N_DOC, SENT_PER_DOC, DIM))
for d in range(N_DOC):
    t = rng.integers(0, N_TOPIC)
    for s in range(SENT_PER_DOC):
        if rng.random() < 0.12:
            t = rng.integers(0, N_TOPIC)
        sent[d, s] = 0.7 * topics[t] + 0.35 * rng.normal(size=DIM)
sent = unit(sent)


def build(chunk_size):
    vecs, spans = [], []
    for d in range(N_DOC):
        for start in range(0, SENT_PER_DOC, chunk_size):
            end = min(start + chunk_size, SENT_PER_DOC)
            vecs.append(sent[d, start:end].mean(axis=0))
            spans.append((d, start, end))
    return unit(np.array(vecs)), spans


INDEXES = {L: build(L) for L in (1, 2, 6, 12)}


def evaluate(strategy, width):
    """Success = the CONTEXT SENT to the model contains the whole answer span,
    within a fixed budget of BUDGET sentences."""
    hits = 0
    for _ in range(N_QUERY):
        d = int(rng.integers(0, N_DOC))
        s0 = int(rng.integers(0, SENT_PER_DOC - width + 1))
        target = (s0, s0 + width)
        q = unit(sent[d, s0:s0 + width].mean(axis=0)
                 + rng.normal(scale=0.12, size=DIM))
        sent_spans = strategy(q)
        for (cd, cs, ce) in sent_spans:
            if cd == d and cs <= target[0] and target[1] <= ce:
                hits += 1
                break
    return hits / N_QUERY


def flat(chunk_size):
    """Retrieve top-k chunks of a single size, k set by the budget."""
    vecs, spans = INDEXES[chunk_size]
    k = max(1, BUDGET // chunk_size)

    def go(q):
        top = np.argpartition(-(vecs @ q), min(k, len(vecs) - 1))[:k]
        return [spans[i] for i in top]
    return go


def parent_child(child_size, parent_size):
    """Embed children; return their PARENTS, deduplicated (eq:parent-dedup-saving),
    taking as many as the budget allows."""
    vecs, spans = INDEXES[child_size]

    def go(q):
        order = np.argsort(-(vecs @ q))
        out, seen, spent = [], set(), 0
        for i in order:
            d, cs, _ = spans[i]
            p_start = (cs // parent_size) * parent_size
            key = (d, p_start)
            if key in seen:
                continue                      # dedupe: children share parents
            if spent + parent_size > BUDGET:
                break
            seen.add(key)
            out.append((d, p_start, min(p_start + parent_size, SENT_PER_DOC)))
            spent += parent_size
        return out
    return go


STRATEGIES = {
    "flat, L=1":              flat(1),
    "flat, L=2":              flat(2),
    "flat, L=6":              flat(6),
    "flat, L=12":             flat(12),
    "parent-child, 1 -> 6":   parent_child(1, 6),
    "parent-child, 1 -> 12":  parent_child(1, 12),
    "parent-child, 2 -> 12":  parent_child(2, 12),
}

print(f"context budget: {BUDGET} sentences for every strategy\n")
print(f"{'strategy':<26}{'fact (w=1)':>13}{'synth (w=3)':>14}"
      f"{'synth (w=6)':>14}{'mean':>8}")
print("-" * 76)
rows = {}
for name, strat in STRATEGIES.items():
    r = [evaluate(strat, w) for w in (1, 3, 6)]
    rows[name] = r
    print(f"{name:<26}{r[0]:>13.3f}{r[1]:>14.3f}{r[2]:>14.3f}"
          f"{np.mean(r):>8.3f}")

best_flat = max((n for n in rows if n.startswith("flat")),
                key=lambda n: np.mean(rows[n]))
best_pc = max((n for n in rows if n.startswith("parent")),
              key=lambda n: np.mean(rows[n]))
print(f"""
best flat:          {best_flat}  (mean {np.mean(rows[best_flat]):.3f})
best parent-child:  {best_pc}  (mean {np.mean(rows[best_pc]):.3f})

Look at the flat rows first and you can see ch:rag-chunking's dilemma laid out:
L=1 is unbeatable on fact queries and scores ZERO on w=6, because a one-sentence
chunk cannot contain a six-sentence answer. L=12 is the reverse. Every flat row
is good at one end of the table and bad at the other, and the best flat
compromise is mediocre everywhere.

The parent-child rows win the mean decisively, and they do it in the way
eq:pc-dominance predicts: they hold the fact column at 1.000 -- which only the
tiny flat chunks manage -- while scoring on synthesis queries, which those tiny
chunks cannot do at all. Two jobs, two units, no compromise between them.

Note that this is at an EQUAL CONTEXT BUDGET. Parent-child sends larger units, so
it sends fewer of them (eq:parent-child-budget), and it still wins. It affords
that through deduplication: when several retrieved children fall in one parent,
that parent is sent once -- and child clustering is itself evidence the parent is
relevant, so the discount arrives exactly when the technique is most likely to be
right.

But read the w=6 column before concluding the technique dominates everywhere,
because it does not. Flat L=12 beats every parent-child configuration there, and
the reason is a coupling eq:pc-dominance does not model. That equation treats
R(L_c) as maximised at the smallest child, which is true for a NARROW query and
false for a wide one: a question whose answer spans six sentences is itself a
broad object, and a two-sentence child is a poor key for it. Retrieval ranks
children by similarity, and a small child simply does not look like a wide query.

So the honest statement is narrower than the clean one. Decoupling removes the
containment constraint entirely -- that part is unconditional, and it is why the
fact and w=3 columns are so strong. It does NOT remove the requirement that the
indexed unit be a good key at the query's granularity, and when queries vary
widely in span, one child size cannot serve all of them.

Which is an argument for indexing at SEVERAL child sizes rather than one, and
that is exactly what hierarchical indexing does (eq:hierarchical-index-size).
This listing is the motivation for the next technique rather than a refutation of
this one: parent-child fixes the completeness half of ch:rag-chunking's dilemma
outright and leaves a residue of the retrieval half, which multi-granularity
indexing addresses at a cost eq:hierarchical-build-cost prices.""")
```

```python {tier=A name=contextual-augmentation}
"""Orphan chunks, and the one-line fix.

Split a document at a fixed size and many chunks are ORPHANS: their text alone
carries no topic -- "as noted above, this reduced latency substantially". They are
unretrievable, because their embedding contains nothing to match, and unusable,
because a reader cannot resolve the reference.

Prepending context before embedding (eq:contextual-augmentation) supplies the
missing topic. eq:augmentation-condition predicts this helps orphans and HURTS
already-specific chunks, and eq:context-length-constraint predicts the harm is
bounded by keeping the prepended text short. We test all three claims.
"""
import numpy as np

rng = np.random.default_rng(29)

N_DOC, CHUNK_PER_DOC, DIM = 400, 12, 48
N_QUERY, K = 800, 10
ORPHAN_RATE = 0.35              # chunks whose own text carries no topic

topics = unit_doc = rng.normal(size=(N_DOC, DIM))
doc_topic = unit_doc / np.linalg.norm(unit_doc, axis=1, keepdims=True)


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


# A chunk's OWN content: specific chunks carry the document topic plus their own
# detail; orphans carry detail only -- no topic at all.
is_orphan = rng.random((N_DOC, CHUNK_PER_DOC)) < ORPHAN_RATE
detail = unit(rng.normal(size=(N_DOC, CHUNK_PER_DOC, DIM)))
own = np.where(is_orphan[..., None],
               detail,
               unit(0.75 * doc_topic[:, None, :] + 0.45 * detail))
own = unit(own)


def augmented(context_weight):
    """eq:augmentation-mix: mix the chunk's own vector with its document's.
    context_weight is L_x / (L_c + L_x) -- the share of the embedded text that
    is prepended context."""
    return unit((1 - context_weight) * own
                + context_weight * doc_topic[:, None, :])


def evaluate(vecs):
    """Retrieve for queries aimed at a specific chunk; report separately for
    orphan and specific targets."""
    flat = vecs.reshape(-1, DIM)
    hits = {"orphan": [0, 0], "specific": [0, 0]}
    for _ in range(N_QUERY):
        d = int(rng.integers(0, N_DOC))
        c = int(rng.integers(0, CHUNK_PER_DOC))
        # The query knows the topic and the detail -- it is a real question about
        # this chunk's content, phrased by someone who knows the document.
        q = unit(0.7 * doc_topic[d] + 0.7 * detail[d, c]
                 + rng.normal(scale=0.20, size=DIM))
        top = np.argpartition(-(flat @ q), K)[:K]
        kind = "orphan" if is_orphan[d, c] else "specific"
        hits[kind][0] += int(d * CHUNK_PER_DOC + c in top)
        hits[kind][1] += 1
    return {k: v[0] / v[1] for k, v in hits.items()}


print(f"{ORPHAN_RATE:.0%} of chunks are orphans (own text carries no topic)\n")
print(f"{'prepended context share':>24}{'orphan chunks':>16}"
      f"{'specific chunks':>18}{'overall':>10}")
print("-" * 70)
for w in [0.0, 0.10, 0.20, 0.35, 0.50, 0.70]:
    r = evaluate(augmented(w))
    overall = ORPHAN_RATE * r["orphan"] + (1 - ORPHAN_RATE) * r["specific"]
    print(f"{w:>24.2f}{r['orphan']:>16.3f}{r['specific']:>18.3f}{overall:>10.3f}")

print("""
The top row is unaugmented chunking, and the orphan column is why this chapter
has a second listing. Those chunks are in the index, they are perfectly good
text, and they are close to unretrievable -- their embedding contains no topic to
match, so a query about their document lands anywhere else.

Adding context rescues them, steeply at first. That is eq:augmentation-condition
in the regime where s(q, context) >> s(q, chunk): for an orphan the chunk's own
similarity is near zero, so any positive context term is a gain.

Now read the specific column, which is the cost nobody prices. It declines
monotonically, because the same mixing that supplies a missing topic DILUTES one
that was already there -- and worse, every chunk in a document is being pulled
toward the same point, which is ch:emb-what-they-are's anisotropy arriving by
choice rather than by accident. At high context share the chunks of a document
become nearly indistinguishable from each other, and retrieval can find the right
document while being unable to find the right chunk within it.

The overall column has an interior optimum, and it sits at a SMALL context share.
That is eq:context-length-constraint derived rather than asserted: a heading path
of a few tokens prepended to a hundred-token chunk lands near the left of this
table, captures most of the orphan gain, and costs the specific chunks almost
nothing. A two-hundred-token document summary prepended to the same chunk lands
on the right, and is a common way to make retrieval worse while believing you
improved it.""")
```

## 9. Practical Example

**Parent–child.** The flat rows lay out {{ch:rag-chunking}}'s dilemma
explicitly: $L=1$ is unbeatable on fact queries and scores **zero** on $w=6$,
because a one-sentence chunk cannot contain a six-sentence answer; $L=12$ is the
reverse; and the best flat compromise is mediocre at both ends.

The parent–child rows win the mean decisively — 0.534 against the best flat
configuration's 0.336 — and they do it the way {{eq:pc-dominance}} predicts:
holding the fact column at **1.000**, which only the one-sentence chunks manage,
while scoring on synthesis queries, which those chunks cannot answer *at all*.
Two jobs, two units, no compromise between them.

The comparison runs at an **equal context budget**, so parent–child sends larger
units and therefore fewer of them ({{eq:parent-child-budget}}) — and still wins.
It affords that through deduplication: when several retrieved children fall in
one parent, the parent is sent once, and **child clustering is itself evidence
the parent is relevant**. The discount arrives exactly when the technique is most
likely to be right.

**But the $w=6$ column is where the clean claim breaks, and it is the more
useful finding.** Flat $L=12$ scores 0.280 there against the best parent–child
configuration's 0.177. The reason is a coupling {{eq:pc-dominance}} does not
model: it treats $R(L_c)$ as maximised at the smallest child, which holds for a
narrow query and fails for a wide one. **A question whose answer spans six
sentences is itself a broad object, and a two-sentence child is a poor key for
it** — retrieval ranks children by similarity, and a small child does not look
like a wide query.

So the honest statement is narrower than the tidy one. **Decoupling removes the
containment constraint outright** — unconditionally, which is why the fact and
$w=3$ columns are so strong. It does *not* remove the requirement that the
indexed unit be a good key at the query's granularity, and one child size cannot
serve queries whose spans vary widely.

> **IMPORTANT:** That residue is the argument for indexing at *several* child
> sizes, which is what hierarchical indexing does
> ({{eq:hierarchical-index-size}}) at a cost {{eq:hierarchical-build-cost}}
> prices. Read this listing as the motivation for that technique rather than as
> a refutation of this one: parent–child fixes the completeness half of
> {{ch:rag-chunking}}'s dilemma outright and leaves a residue of the retrieval
> half.

**Contextual augmentation.** The unaugmented row shows why orphans deserve
attention: those chunks are in the index, they are perfectly good text, and they
are close to unretrievable, because their embedding contains no topic for a query
to match.

Adding context rescues them steeply — {{eq:augmentation-condition}} in the regime
where the chunk's own similarity is near zero, so any positive context term is a
gain.

The specific-chunk column is **the cost nobody prices.** It falls from 0.936 to
0.737 across the sweep, because the same mixing that supplies a missing topic
dilutes one that was already present — every chunk in a document pulled toward
one point, which is {{ch:emb-what-they-are}}'s anisotropy arriving *by choice*.

**The clearest evidence for that mechanism is the last row, where the orphan
column falls too** (0.912, down from 0.973). Augmentation stops helping even the
chunks it exists to help, because once every chunk in a document embeds near the
document centroid, an orphan cannot be told apart from its siblings either. The
pathology is not "specific chunks pay for orphans" — beyond a point,
within-document ranking degrades for everything.

The overall optimum is **broad and sits in the middle** of the sweep, not at
either end — and this corrected what I expected to write. Where it sits depends
on the orphan rate, a property of your corpus and chunker rather than a constant:
at 35% orphans a substantial context share pays, and at 5% it would not. **That
is why {{sec:10-production-considerations}} asks you to measure the orphan rate
before choosing.**

What is robust is the *shape*: a steep gain, a broad plateau, and a decline at
the far end where both columns suffer. A heading path of a few tokens lands
safely on the rising part for almost nothing, which is the recommendation
{{eq:context-length-constraint}} supports — not because small is optimal, but
because small is cheap, safe, and captures the steepest part of the curve.

## 10. Production Considerations

**Store parent pointers at chunk time.** Retrofitting them means re-chunking, and
they cost a few bytes ({{ch:rag-indexing}}).

**Deduplicate parents before assembly.** Not optional
({{eq:parent-dedup-saving}}), and it is where parent–child's budget advantage
comes from.

**Cap parent size and expand outward from the child.** Uniform parents fail on
heterogeneous corpora; a size-capped expansion respecting section boundaries does
not.

**Prepend the heading path and stop there until you have measured.** It is the
first tier of {{sec:7-internal-mechanics}}'s list, nearly free, and captures most
of the benefit.

**Keep prepended context short** ({{eq:context-length-constraint}}). Measure the
specific-chunk column, not just the orphan one — the technique has a cost and it
is invisible if you only look at what it fixes.

**Price the hierarchical build before adopting it**
({{eq:hierarchical-build-cost}}), including the *recurring* cost, since a
document edit invalidates every summary above it.

**Measure the orphan rate.** Sample fifty chunks and count how many are
interpretable alone. It tells you whether augmentation is worth anything at all
for your corpus, and it is the same fifty-chunk read {{ch:rag-ingestion}} asked
for.

## 11. Common Mistakes

**Coupling the retrieval and generation units by default.**
{{eq:flat-coupling}} is a choice nobody made deliberately.

**Sending whole documents as parents.** Works in testing, truncates in
production.

**Not deduplicating parents.** Wastes the budget on repeated text and destroys
the technique's cost advantage.

**Prepending too much context.** {{eq:augmentation-condition}}'s harm term
applies to your best chunks.

**Adopting hierarchical indexing without pricing the ingest.**
{{eq:hierarchical-build-cost}}.

**Comparing at unequal budgets.** A technique that wins by spending more tokens
has not been shown to win.

**Assuming augmentation helps everything.** It helps orphans and hurts specifics;
the net depends on your orphan rate.

## 12. Failure Modes

**Parent explosion.** A retrieved child sits in a 6,000-token section; one result
consumes the budget. Cap parent size.

**Context-window overflow after expansion.** The retrieval stage fits and the
expansion does not, and the truncation is silent
({{ch:rag-generation}}).

**Augmentation-induced document collapse.** Every chunk in a document embeds near
the document centroid, so within-document ranking becomes noise. Symptom: the
right document is retrieved and the wrong chunk within it, consistently.

**Stale hierarchical summaries.** A document is edited; the summaries above it
are not rebuilt; retrieval matches a summary describing content that no longer
exists — and this is {{ch:rag-indexing}}'s {{eq:deletion-asymmetry}} with an
extra level.

**Pointer drift after re-chunking.** Parent references point at the wrong span,
so expansion returns unrelated text — which presents as a retrieval failure and
is not one.

**Orphan rate underestimated** because nobody read the chunks.

## 13. Alternatives

**Late interaction** ({{cite:khattab2020colbert}}). The extreme of the same axis:
index every token, generate from the document. Removes the granularity question
entirely at 10–100× storage.

**Hierarchical indexing** ({{cite:sarthi2024raptor}}). Index every level rather
than two. Better for corpus-level questions and priced by
{{eq:hierarchical-build-cost}}.

**Bigger chunks and a reranker.** Retrieve large, rerank with a cross-encoder
({{ch:emb-reranking}}) to recover precision. A different route to the same place,
paying at query time rather than index time.

**Query decomposition** ({{ch:rag-query-understanding}}) for multi-document
answers, which no chunking scheme reaches.

**Just fix the documents.** Many orphans exist because the source document uses
"this" and "above" heavily. For a corpus you control, editing for
self-containment is unglamorous and permanently effective.

## 14. Evaluation

**By query type**, as {{ch:rag-chunking}} insisted. The whole claim is about
performing well on incompatible query types simultaneously, and an aggregate
cannot show it.

**At equal context budget.** The comparison is meaningless otherwise.

**Orphan-chunk retrieval separately from specific-chunk retrieval.**
{{eq:augmentation-condition}} says they move in opposite directions, so a single
number hides the trade.

**Parent-size distribution and the deduplication rate**, which together determine
the realised token cost.

**Answer completeness given retrieval** — did the sent context contain the whole
answer? This is the factor parent–child is buying and it is not measured by
recall.

**Ingest cost and rebuild time** for hierarchical approaches, as first-class
numbers rather than footnotes.

## 15. Advanced Concepts

**The granularity axis is one design space.** Flat chunking, sentence-window,
parent–child, hierarchical, and late interaction differ only in what is indexed
and what is sent. Seeing them as points on one axis rather than as competing
techniques makes the choice a budget question — index size against ingest cost
against context tokens — which is answerable.

**Decoupling generalises past text.** Index a function signature and send the
function; index a table row and send the table with its header
({{ch:rag-structured}}); index a caption and send the figure
({{part:13}}). The pattern is the same and the payoff is larger where the
retrieval unit and the useful unit differ more sharply.

**{{eq:pc-dominance}} is why this chapter is short on caveats.** Techniques
usually trade; this one dominates, because it removes a constraint rather than
optimising within it. The general lesson — look for the artificial coupling
before tuning the compromise — applies well beyond retrieval.

**Contextual augmentation is anisotropy by choice.** {{ch:emb-what-they-are}}
measured that a shared component across a corpus compresses similarity dynamic
range; augmentation deliberately adds one *per document*. In small doses it is a
feature — within-document chunks *should* be related — and in large doses it is
the same pathology, with the same fix.

**The hierarchical build cost is a recurring cost**, and this is what most
comparisons miss. {{eq:hierarchical-build-cost}} is paid at build time and again
on every material edit, which makes the technique's economics depend on corpus
churn ({{ch:rag-why}}) rather than on corpus size.

## 16. Connection to Previous Chapters

{{ch:rag-chunking}}'s incompatible optima are the problem, and
{{eq:pc-dominance}} is the resolution — the tension existed only because
{{eq:flat-coupling}} tied both factors to one parameter.
{{ch:emb-what-they-are}}'s dilution and anisotropy arguments govern
{{eq:augmentation-mix}} in both directions. {{ch:rag-generation}}'s budget and
{{eq:marginal-chunk-value}} are why trading several mediocre chunks for one
complete one is favourable. {{ch:emb-reranking}}'s late interaction is the
extreme point of this chapter's axis. And {{ch:rag-indexing}}'s neighbour
pointers are the mechanism that makes any of it possible.

## 17. Exercises

1. Prove {{eq:pc-dominance}} and state the condition under which parent–child and
   flat chunking are equivalent.
2. Derive {{eq:augmentation-condition}} from {{eq:augmentation-mix}}.
3. In `parent-child-retrieval`, reduce `BUDGET` to 12. Does parent–child still
   dominate, and which configuration wins?
4. Add a strategy that retrieves with $L=1$ and sends a symmetric ±2 sentence
   window. Compare it to parent–child at equal budget.
5. In `contextual-augmentation`, raise `ORPHAN_RATE` to 0.7. Where does the
   overall optimum move, and why?
6. Modify the same listing so the prepended context is chunk-specific rather than
   document-wide. Does the specific-chunk penalty disappear?
7. Use {{eq:hierarchical-build-cost}} to price a RAPTOR index over 500,000 chunks
   at $b=5$. Now price the monthly rebuild at 10% document churn.
8. Design the orphan-rate measurement. What exactly do you show a human, and what
   do you ask them?

## 18. Interview Questions

1. What is the one idea behind parent–child retrieval?
2. Why does it beat flat chunking rather than trading against it?
3. What does parent–child cost, and how do you keep the cost down?
4. What is an orphan chunk and how do you fix it?
5. Why can prepending context make retrieval worse?
6. How much context should you prepend?
7. When is hierarchical indexing worth its ingest cost?
8. Your system retrieves the right document and the wrong chunk. Diagnose.
9. How would you compare two retrieval architectures fairly?
10. Where else does the retrieval-unit/generation-unit split apply?

## 19. Research Questions

1. Parent size is fixed or capped by heuristic. Can the expansion be *learned* —
   predicting from the child and the query how far to expand?
2. {{eq:augmentation-condition}} implies augmentation should be applied per chunk
   rather than uniformly. Is there a cheap predictor of which chunks are orphans?
3. Is there a principled way to choose the number of levels and the branching
   factor in {{eq:hierarchical-index-size}}, given a query distribution?
4. Hierarchical summaries go stale on edit. Is there an incremental update that
   avoids re-summarising the whole path to the root?
5. The granularity axis has been explored empirically and not characterised. What
   is the optimal indexed/sent pair as a function of the answer-span
   distribution?

## 20. Chapter Summary

{{ch:rag-chunking}}'s dilemma existed only because **one object was doing two
jobs**. A chunk is what gets embedded and what gets sent, and
{{eq:flat-coupling}} is a choice nobody made deliberately. Relax it and the
retrieval factor and the completeness factor become independent
({{eq:decoupled-success}}), maximisable separately.

**Parent–child therefore beats the compromise rather than tuning it.**
{{eq:pc-dominance}} — the product of the maxima is at least the maximum of the
product — and measured at an *equal context budget*, parent–child scores 0.534
against the best flat configuration's 0.336, holding fact retrieval at 1.000
while answering synthesis queries that one-sentence chunks cannot answer at all.
It affords the larger units through deduplication, and child clustering is itself
evidence the parent is relevant.

**With one measured limit.** {{eq:pc-dominance}} assumes the child size affects
only retrieval, and that separability fails for wide queries: a six-sentence
answer is a broad object and a two-sentence child is a poor key for it, so flat
$L=12$ wins that column. Decoupling removes the *containment* constraint
unconditionally; it does not remove the requirement that the indexed unit match
the query's granularity — which is the argument for indexing several child sizes,
priced by {{eq:hierarchical-build-cost}}.

**Contextual augmentation fixes orphans and has a price.** Chunks whose own text
carries no topic are nearly unretrievable, and prepending context rescues them
steeply ({{eq:augmentation-condition}}). But the same mixing dilutes chunks that
were already specific, pulling every chunk in a document toward one point —
{{ch:emb-what-they-are}}'s anisotropy by choice — until retrieval finds the right
document and the wrong chunk — and at the far end of the sweep the orphan column
falls too, because within-document ranking has degraded for everything. The
overall optimum is **broad and moderate**, and where it sits depends on the
orphan rate rather than on any constant, which is why measuring that rate comes
first. What is robust is the shape, and the recommendation
{{eq:context-length-constraint}} supports: **prepend a heading path, not a
summary** — because a few tokens land on the steepest part of the curve for
almost nothing.

The general lesson is worth more than either technique. **Look for the artificial
coupling before optimising the compromise** — the dilemma in
{{ch:rag-chunking}} was not a fact about retrieval, it was a consequence of an
identification nobody had questioned.

## 21. Further Reading

{{cite:sarthi2024raptor}} for hierarchical indexing — Sections 3 and 4, and note
what the paper does not price.
{{cite:khattab2020colbert}} for the extreme point of the granularity axis.
{{cite:liu2023lost}} for why the expanded context's *ordering* matters once
parents are large ({{ch:rag-generation}}).
{{cite:gao2023ragsurvey}} for the standard names — "advanced RAG" covers most of
this chapter and organises it by technique rather than by the axis they share.
{{cite:lewis2020rag}} for the flat 100-word passages this chapter's architecture
departs from.

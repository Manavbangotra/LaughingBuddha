---
id: rag-indexing
number: 109
part: XII
tier: full
status: draft
requires: [rag-chunking, rag-ingestion, emb-vector-db, emb-hybrid, emb-models,
           rag-why]
provides: [metadata-schema, rag-metadata-filtering, access-control-retrieval,
           index-freshness-rag, incremental-indexing, tenant-isolation-rag,
           retrieval-configuration, corpus-versioning]
citations: [gao2023ragsurvey, thakur2021beir, formal2021splade, lewis2020rag,
            liu2023lost]
---

## 1. Learning Objectives

By the end of this chapter you will be able to design a metadata schema that
serves filtering, citation, and access control, and say why each field cannot be
added later; demonstrate why numeric and temporal constraints must be *filters*
rather than embedded text, and measure how badly the alternative fails; implement
permission-aware retrieval and explain why post-filtering is a security
liability rather than a performance one; and reason about index freshness as a
staleness window with a measurable cost.

## 2. Why This Matters

{{part:11}} built the index. This chapter is about the *other* half of a
retrieval request — the half that is not a vector — and it is where RAG systems
acquire their most embarrassing failures.

A user asks for last quarter's figures and gets last year's. A contractor's
question surfaces a board document. A query about pricing in Germany returns the
US price list. **None of these is a retrieval-quality problem**, and every one of
them is routinely attacked by tuning the embedding model, because the team's
mental model of retrieval is "find similar text" and these failures are not about
similarity at all.

The unifying claim: **an embedding is a similarity function, and constraints are
not similarity.** A date range, a permission, a tenant, a document status, a
language — these are *predicates*, they have exact answers, and asking a vector
to approximate them produces a system that is wrong in ways no amount of
retrieval tuning fixes. {{sec:9-practical-example}} measures how wrong.

{{maturity:ESTABLISHED}} Metadata filtering is standard in every vector database
and every search engine. {{maturity:MATURE}} The access-control patterns are
borrowed wholesale from search infrastructure and are well understood — and
routinely reimplemented badly by teams who have not met them before.

## 3. Prerequisites

{{ch:rag-chunking}} for the chunks being indexed and the metadata attached to
them; {{ch:rag-ingestion}} for where that metadata comes from;
{{ch:emb-vector-db}} for pre- and post-filtering, percolation, and the index
mechanics this chapter applies; {{ch:emb-hybrid}} for lexical retrieval;
{{ch:emb-models}} for the embedding schema.

## 4. Intuitive Explanation

### Similarity is not a constraint

The chapter in one idea.

"Find documents about logistics" is a **similarity** question. It has no exact
answer, degrees of relevance are meaningful, and an embedding is the right tool.

"Find documents from Q3 2024" is a **constraint**. It has an exact answer, a
document either satisfies it or does not, and there is no such thing as being
*somewhat* from Q3.

Teams collapse these because both arrive in one sentence — *"show me Q3 logistics
reports"* — and the retrieval stack has one input. The tempting move is to write
the date into the chunk text and let the embedder sort it out:

```text
Date: 2024-07-15. Category: Logistics. Regional throughput improved by 12%...
```

This does not work, and the reason is worth being precise about.
`2024-07-15` and `2024-08-22` are lexically and semantically near-identical to an
embedding model — both are "a date in 2024" — while `2024-07-15` and `2023-07-15`
are *also* near-identical. **The embedding preserves that these are dates and
discards which one is larger.** A query for "after 2023" has no vector that
selects the right half of the corpus, because the ordering the query depends on
was never represented.

{{sec:9-practical-example}} quantifies it. The short version: metadata belongs in
metadata.

### What retrieval actually is here

So a retrieval request has two parts, and they run through different machinery:

$$ \text{request} = \underbrace{\text{"logistics throughput"}}_{\text{similarity → vector index}} \;+\; \underbrace{\text{date} \in \text{Q3 2024},\ \text{tenant} = A}_{\text{constraint → filter}} $$

and {{ch:emb-vector-db}} already told us the hard part: combining them is the
expensive operation, and both strategies for doing so fail at high selectivity.
This chapter's contribution is what the constraints *are* in a RAG system, why
each must exist at index time, and which of them turn a quality bug into a
security incident.

### The permission case is different in kind

Most of this chapter is about correctness. One part is not.

If a date filter is wrong, a user sees a stale figure. If a **permission** filter
is wrong, a user sees a document they are not cleared for — and unlike almost
every other failure in this book, that one is *silent*, *irreversible*, and
*reportable*. The document appears in a result list looking exactly like a
legitimate result; the model summarises it faithfully; nothing errors.

**Permission filtering must therefore be structural rather than a predicate**,
and {{sec:5-formal-explanation}} makes the argument precisely.

## 5. Formal Explanation

### 5.1 The retrieval request

$$ \hat{z} = \text{top-}k \;\big\{\, z \in \mathcal{I} \;:\; \phi(z) = \text{true} \,\big\} \ \text{ordered by} \ s(q, z) $$ (eq:filtered-retrieval)

with $\phi$ a conjunction of predicates over metadata. Everything in this chapter
is a statement about $\phi$: where it comes from, when it is evaluated, and what
happens when it is wrong.

### 5.2 Why constraints cannot be embedded

Let $\phi$ be a range predicate on an attribute $a(z)$, and suppose the attribute
is written into the chunk text and embedded. Retrieval then approximates $\phi$
by similarity to a query mentioning the range. For that to work, the embedding
would need to satisfy

$$ a(z_1) < a(z_2) < a(z_3) \;\Longrightarrow\; \hat{f}(z_2) \text{ lies between } \hat{f}(z_1) \text{ and } \hat{f}(z_3) $$ (eq:ordering-preservation)

— an *order-preserving* embedding of the attribute. Nothing in
{{ch:emb-what-they-are}}'s contrastive objective asks for this. The training
signal makes semantically similar texts near each other, and two dates are
semantically similar *because they are both dates*, regardless of order.

$$ \text{trained for: } \text{sim}(z_1, z_2) \approx \text{semantic relatedness} \;\;\not\Rightarrow\;\; \text{eq:ordering-preservation} $$ (eq:no-order-in-embedding)

So a range query over embedded attributes has **no correct answer available to
it**, and its observed accuracy is whatever the lexical accident of the date
format produces. Change `2024-07-15` to `15 July 2024` and the behaviour changes,
which is the signature of a system relying on something it should not.

**The same argument covers every ordered or exact attribute**: version numbers,
prices, counts, statuses, identifiers. It is {{ch:emb-hybrid}}'s capacity bound
in a different guise — an embedding discards exactly the information a predicate
needs.

### 5.3 Pre- and post-filtering, with the security consequence

{{ch:emb-vector-db}} established the two strategies and their failure modes.
Restated with $\phi$ as a permission predicate, the asymmetry becomes stark:

$$ \text{post-filter: } \text{retrieve top-}B \text{ ignoring } \phi, \text{ then drop } \neg\phi \quad\Longrightarrow\quad \text{correct, but } B = O(k/s) $$ (eq:postfilter-permissions)

$$ \text{pre-filter: } \text{restrict the search to } \phi \quad\Longrightarrow\quad \text{cheap, but recall degrades below } s_c $$ (eq:prefilter-permissions)

Post-filtering is *correct* for permissions — a forbidden document is dropped —
but it has a property that should stop you: **the forbidden documents were
retrieved, scored, and passed through your process before being dropped.** They
appear in logs, in traces, in caches, and in whatever crash dump exists on a bad
day. Correctness of the final response is not the same as never handling the
data, and in a regulated environment those are different obligations.

Pre-filtering does not handle the data — and by {{eq:hnsw-filter-threshold}}
degrades recall silently when the permitted set is small, which for a
per-user permission filter it usually is.

**Neither is right, and the resolution is not to choose.** Partition by tenant so
the permission is part of the *address* ({{ch:emb-vector-db}}), and use $\phi$
only for the finer-grained predicates within a tenant. This is the same
conclusion {{ch:emb-vector-db}} reached from the latency side, arriving here from
the security side — which is usually a sign the conclusion is right.

### 5.4 Freshness

An index is a snapshot. Let documents change at rate $\lambda$ per document per
unit time, over $N_d$ documents, with the index rebuilt every $T$:

$$ \E[\text{stale documents at time } t \text{ into a cycle}] = N_d \lambda t, \qquad \E[\text{average}] = \frac{N_d \lambda T}{2} $$ (eq:index-staleness)

and the probability a query is served at least one stale chunk, retrieving $k$:

$$ \Prob[\text{stale answer}] \approx 1 - \left(1 - \frac{\lambda T}{2}\right)^{k} \approx \frac{k \lambda T}{2} \ \text{ for small } \lambda T $$ (eq:stale-query-rate)

**The $k$ is the part people miss.** Retrieving eight chunks makes a query eight
times as likely to touch stale content as retrieving one, so raising $k$ for
recall reasons raises the staleness rate proportionally.
{{sec:9-practical-example}} measures this.

### 5.5 Deletion is the hard half

Additions are easy: append to the index and they are retrievable. **Deletions and
supersessions are not**, and they are the ones that matter, because a stale
*addition* is a missing answer while a stale *deletion* is a wrong answer served
confidently with a citation to a document that no longer says that.

$$ \text{harm}(\text{missing new}) \ll \text{harm}(\text{serving deleted}) $$ (eq:deletion-asymmetry)

The consequence for design: **deletions must propagate faster than additions.**
A nightly full rebuild plus a real-time deletion tombstone stream is a
substantially better architecture than a nightly rebuild alone, and costs very
little more.

## 6. Mathematical Foundation

### 6.1 The metadata a chunk must carry

Not a mathematical result, but the chapter's load-bearing list, organised by
*why the field cannot be added later*:

| Field | Purpose | Why it must exist at index time |
|---|---|---|
| document id, version | citation, dedup | a chunk with no provenance cannot be cited or invalidated |
| char offsets | verifiable citation | offsets shift on re-chunk; must be captured with the chunk |
| heading path | context, filtering | derivable only during ingestion ({{ch:rag-ingestion}}) |
| timestamp | freshness filter, churn measurement | source dates are lost on export |
| access labels | permissions | retrofitting means re-ingesting |
| tenant | partitioning | determines the physical index |
| language, type, status | filtering | cheap now, a re-embed later |

**Every row is expensive to add retroactively and nearly free to add now.** This
asymmetry is the single most useful thing in the chapter: the cost of an unused
metadata field is a few bytes, and the cost of a missing one is re-ingesting the
corpus.

### 6.2 Selectivity of a conjunction

Filters compose, and the composition is where planners go wrong
({{ch:emb-vector-db}}). Under independence:

$$ s(\phi_1 \wedge \phi_2) = s_1 s_2 $$ (eq:conjunction-selectivity)

which is almost never true in a document corpus. Tenant and language are
correlated; date and status are correlated; a document's type predicts its
access labels. Correlated predicates give

$$ s(\phi_1 \wedge \phi_2) \gg s_1 s_2 $$ (eq:correlated-selectivity)

so the independence estimate is **too small**, the planner believes the filtered
set is tiny, chooses brute force, and scans far more than it expected. That is a
latency failure. The opposite error — over-estimating selectivity and choosing a
filtered graph traversal below $s_c$ — is a *silent recall* failure, which is
worse.

**Under uncertainty, prefer the strategy that fails loudly**, which is
{{ch:emb-vector-db}}'s rule and worth repeating because the consequence here can
be a permission-shaped one.

### 6.3 The cost of a metadata field

For $N_c$ chunks, a field of $b$ bytes costs $N_c b$ in the index plus its
inverted-index or B-tree overhead. For ten million chunks and a 16-byte field
that is 160 MB — against {{ch:emb-vector-db}}'s vector store at

$$ \frac{N_c \cdot d \cdot 4}{N_c \cdot b} = \frac{768 \times 4}{16} = 192 \times $$ (eq:metadata-vs-vector-cost)

**Metadata is roughly two orders of magnitude cheaper than the vectors it sits
beside.** There is no storage argument for omitting a field, and the instinct to
keep the schema lean is misapplied here.

## 7. Internal Mechanics

```mermaid {#fig:rag-retrieval-request caption="A retrieval request has two parts that run through different machinery. The similarity half is approximate and tunable; the constraint half is exact and must not be approximated. Access control is a constraint that has been promoted into the address."}
flowchart TD
    R["request: text + constraints + caller identity"] --> T["tenant / permission<br/>→ selects the index"]
    R --> P["predicates: date, type,<br/>language, status"]
    R --> S["semantic text"]
    T --> IDX[("tenant-partitioned index")]
    P --> F{"filter strategy<br/>(ch:emb-vector-db)"}
    S --> V["embed"]
    V --> F
    IDX --> F
    F --> K["top-k candidates"]
    K --> RR["rerank<br/>(ch:emb-reranking)"]
    RR --> O["context for generation"]
```

### 7.1 Designing the metadata schema

Three rules, in order of how often they are broken.

**Capture everything cheap at ingest.** {{eq:metadata-vs-vector-cost}}: storage is
not the constraint, and the field you did not capture is the one the product
asks for in month three.

**Normalise at index time, not query time.** Dates as timestamps, not strings;
enumerations as canonical values, not free text; languages as codes. A filter on
a field that was never normalised is a filter with a silent false-negative rate,
and it will be blamed on retrieval.

**Distinguish filterable from displayable.** A chunk carries fields used to
*select* it and fields used to *show* it — the heading path is both, the raw HTML
is only the second. Only the first needs an index, and conflating them makes the
index unnecessarily large.

### 7.2 Where the constraints come from

The part teams underestimate: **the user did not state the constraints.** A
question like *"what did we change in the returns policy?"* contains an implicit
"currently", an implicit tenant, and an implicit language. Constraints reach
$\phi$ from four places, and only one of them is the user:

- **The caller's identity** — tenant, permissions, locale. Never from the query
  text, always from the request context.
- **Application context** — the current project, workspace, or document being
  viewed.
- **Explicit query terms** — "last quarter", "in German". Extracting these is
  {{ch:rag-query-understanding}}'s job, and it is a parsing problem, not a
  retrieval one.
- **Defaults** — status is published, language matches the user's, documents are
  current unless asked otherwise. **Defaults are policy** and should be written
  down somewhere a product person can read.

### 7.3 Incremental indexing

The pattern that works, borrowed from search infrastructure and rediscovered by
every RAG team:

- **A content hash per chunk.** Unchanged chunks are skipped, so re-ingesting a
  corpus after editing three documents costs three documents.
- **A deletion stream, applied immediately.** {{eq:deletion-asymmetry}}.
- **Append-only additions**, with a periodic compaction — {{eq:segment-search}}'s
  segment pattern.
- **Atomic swap for full rebuilds**, so a rebuild is never partially visible.

The subtlety is that **chunk identity must be stable under editing.** If chunk
ids are positional, editing paragraph two renumbers every chunk after it and the
whole document appears changed. Content-hash ids make the diff minimal, which is
what makes incremental indexing worth having.

## 8. Implementation

```python {tier=A name=constraints-not-similarity}
"""Why a date range cannot be embedded, measured.

Two ways to serve the request "logistics reports from Q3 2024":

  embedded metadata -- write the date into the chunk text and hope the embedder
                       represents it. This is what teams try first.
  metadata filter   -- filter on a parsed timestamp, then rank the survivors by
                       semantic similarity.

The embedding here is DELIBERATELY generous: dates get their own dimensions and a
smooth representation of year and month, which is far better than a real text
encoder manages. The point is that even a generous embedding fails, because
eq:no-order-in-embedding is about the objective, not the capacity.
"""
import numpy as np

rng = np.random.default_rng(7)

N_CHUNK, DIM, K = 4000, 32, 10
N_TOPIC = 12
TOPIC_DIMS = DIM - 4                 # last 4 dims are reserved for the date

topics = rng.normal(size=(N_TOPIC, TOPIC_DIMS))
topics /= np.linalg.norm(topics, axis=1, keepdims=True)

# Each chunk: a topic, and a date somewhere in 2022-2025.
chunk_topic = rng.integers(0, N_TOPIC, size=N_CHUNK)
year = rng.integers(2022, 2026, size=N_CHUNK)
month = rng.integers(1, 13, size=N_CHUNK)
months_abs = (year - 2022) * 12 + (month - 1)      # the true ordering


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def date_features(year, month):
    """A GENEROUS embedding of a date: year and month, each as a normalised
    scalar plus a cyclic component. A real text encoder does far worse, because
    it sees '2024-07-15' as a token sequence."""
    y = (year - 2022) / 4.0
    m = (month - 1) / 12.0
    return np.stack([y, m, np.sin(2 * np.pi * m), np.cos(2 * np.pi * m)], axis=-1)


emb = np.concatenate([topics[chunk_topic] * 1.0,
                      date_features(year, month) * 0.6], axis=1)
emb = unit(emb)


def query_vector(topic, q_year, q_month):
    v = np.concatenate([topics[topic],
                        date_features(np.array(q_year), np.array(q_month)) * 0.6])
    return unit(v)


def evaluate(window_months):
    """A query asks for one topic within a date WINDOW.

    We score the two halves of the request SEPARATELY, because the chapter's
    claim is that one half works and the other cannot:
      topic ok  -- did the retrieved chunk match the semantic request?
      date ok   -- did it satisfy the constraint?
    """
    out = {"emb_topic": [], "emb_date": [], "emb_both": [],
           "filt_topic": [], "filt_date": [], "filt_both": []}
    for _ in range(400):
        topic = int(rng.integers(0, N_TOPIC))
        end = int(rng.integers(window_months, 48))
        lo, hi = end - window_months, end
        gold = np.flatnonzero((chunk_topic == topic)
                              & (months_abs >= lo) & (months_abs < hi))
        if len(gold) < 3:
            continue

        # Query date is the MIDDLE of the requested window.
        mid = (lo + hi) // 2
        q = query_vector(topic, 2022 + mid // 12, 1 + mid % 12)

        def score(idx):
            in_topic = chunk_topic[idx] == topic
            in_date = (months_abs[idx] >= lo) & (months_abs[idx] < hi)
            return in_topic.mean(), in_date.mean(), (in_topic & in_date).mean()

        # (a) embedded metadata: pure vector search, no filter
        top = np.argpartition(-(emb @ q), K)[:K]
        a, b, c = score(top)
        out["emb_topic"].append(a)
        out["emb_date"].append(b)
        out["emb_both"].append(c)

        # (b) metadata filter: restrict to the window, then rank semantically
        allowed = np.flatnonzero((months_abs >= lo) & (months_abs < hi))
        scores = emb[allowed] @ q
        top_f = allowed[np.argpartition(-scores, min(K, len(allowed) - 1))[:K]]
        a, b, c = score(top_f)
        out["filt_topic"].append(a)
        out["filt_date"].append(b)
        out["filt_both"].append(c)

    return {k: float(np.mean(v)) for k, v in out.items()}


print("fraction of the top-10 that satisfies each half of the request\n")
print(f"{'date window':>13}{'embedded: topic':>17}{'date':>8}{'both':>8}"
      f"{'filtered: topic':>18}{'date':>8}{'both':>8}")
print("-" * 80)
for window in [3, 6, 12, 24]:
    r = evaluate(window)
    print(f"{str(window) + ' months':>13}{r['emb_topic']:>17.3f}"
          f"{r['emb_date']:>8.3f}{r['emb_both']:>8.3f}"
          f"{r['filt_topic']:>18.3f}{r['filt_date']:>8.3f}{r['filt_both']:>8.3f}")

print("""
Read the two halves separately, because that separation is the whole point.

The TOPIC column is high under both strategies. Similarity retrieval works; that
was never in question, and it is what an embedding is for.

The DATE column is where they diverge, and the divergence is total. Filtering
gives 1.000 by construction -- a filter does not approximate a predicate, it
evaluates it. The embedded path sits far below at every window, which means
roughly a third of what it returns VIOLATES the stated constraint. Not ranked
lower: returned, in the top ten, indistinguishable from a correct result.

Now note what the embedded path is not doing: it is not getting worse as the
window narrows, and it is not getting better either. It is roughly CONSTANT,
which is itself the diagnosis. If the embedding had any notion of the interval,
its accuracy would move with the interval's width. It does not, because
similarity in the date subspace is a DISTANCE FROM THE QUERY DATE, not membership
in a range -- a chunk one month outside a narrow window is nearer the query date
than a chunk at the far edge of a wide one. The ranking cannot express "inside or
outside" however the date is encoded, so what you measure is a fixed blend of
near-misses rather than a constraint being applied well or badly.

That is eq:no-order-in-embedding. The predicate needs an ordering relation and a
dot product does not have one.

And this embedding is far MORE generous to the date than any real text encoder:
dedicated dimensions, year and month as smooth scalars, rather than the token
sequence '2024-07-15'. A real encoder does worse, and its behaviour changes when
you reformat the date -- which is the signature of a system relying on a lexical
accident rather than on a represented quantity.""")
```

```python {tier=A name=index-staleness}
"""How stale is the index between rebuilds, and why does k make it worse?

eq:stale-query-rate says the probability a query touches stale content grows with
the retrieval depth k, because each retrieved chunk is an independent chance to
hit something that changed since the last rebuild.

We simulate a corpus with per-document churn against several rebuild cadences,
and separately track the two failure kinds of eq:deletion-asymmetry: serving an
OUTDATED version, and serving a DELETED document.
"""
import numpy as np

rng = np.random.default_rng(23)

N_DOC, N_QUERY, DAYS = 20_000, 4000, 90
CHURN_PER_DOC_PER_DAY = 0.004        # ~11% of the corpus changes per month
DELETE_FRACTION = 0.15               # of changes, this share are deletions
CADENCES = [1, 7, 30]                # rebuild every N days
DEPTHS = [1, 4, 8, 16]

print(f"corpus {N_DOC:,} docs, churn {CHURN_PER_DOC_PER_DAY:.3%}/doc/day "
      f"({N_DOC * CHURN_PER_DOC_PER_DAY:,.0f} docs/day)\n")
print(f"{'rebuild':>9}{'k':>5}{'stale answer':>15}{'  of which deleted':>20}"
      f"{'predicted (eq)':>17}")
print("-" * 66)

for cadence in CADENCES:
    for k in DEPTHS:
        stale_hits, deleted_hits = 0, 0
        for _ in range(N_QUERY):
            # Uniformly random point in the rebuild cycle.
            age_days = rng.random() * cadence
            p_changed = 1 - (1 - CHURN_PER_DOC_PER_DAY) ** age_days
            changed = rng.random(k) < p_changed
            if changed.any():
                stale_hits += 1
                # Of the changed chunks retrieved, were any deletions?
                if (rng.random(int(changed.sum())) < DELETE_FRACTION).any():
                    deleted_hits += 1
        rate = stale_hits / N_QUERY
        del_rate = deleted_hits / N_QUERY
        predicted = 1 - (1 - CHURN_PER_DOC_PER_DAY * cadence / 2) ** k
        print(f"{cadence:>7}d{k:>5}{rate:>15.3f}{del_rate:>20.3f}"
              f"{predicted:>17.3f}")
    print()

print("""
The measured column tracks eq:stale-query-rate closely, which is the point of
printing both -- the model is simple enough to reason with, so use it.

Read across k at a fixed cadence. Retrieval depth multiplies the staleness rate
almost linearly, because every retrieved chunk is another chance to touch
something that has changed. This is a genuine coupling that nobody plans for:
raising k to improve recall (ch:rag-chunking) raises the stale-answer rate
proportionally, so the two decisions are not independent and a recall improvement
can be a freshness regression.

Now read the deleted column. It is a fraction of the stale column, and it is the
one that matters, because of eq:deletion-asymmetry. A stale UPDATE gives a user
an outdated number. A stale DELETION gives them a confident, cited answer from a
document that has been retracted -- and the citation makes it MORE credible, not
less. The harm is not proportional to the rate.

Which gives the architectural conclusion: the rebuild cadence is set by the
update rate and the tolerance for outdated content, but deletions should not wait
for it at all. A tombstone stream applied within seconds costs almost nothing
next to a full rebuild and removes the worst failure mode entirely.""")
```

## 9. Practical Example

**Constraints.** Scoring the two halves of the request separately is what makes
the result legible, and the separation is total.

**The topic half is 1.000 under both strategies.** Similarity retrieval works;
that was never in question, and it is what an embedding is for.

**The date half is 1.000 filtered and about 0.69 embedded**, at every window
width. That second number means roughly **a third of what the embedded path
returns violates the stated constraint** — not ranked lower, but returned in the
top ten, indistinguishable from a correct result, and about to be summarised
confidently.

The most diagnostic feature is that 0.69 barely moves as the window goes from
three months to twenty-four. **If the embedding had any notion of the interval,
its accuracy would vary with the interval's width.** It does not, because
similarity in the date subspace is a *distance from the query date* rather than
membership in a range: a chunk one month outside a narrow window sits nearer the
query date than a chunk at the far edge of a wide one. What the number measures
is a fixed blend of near-misses, not a constraint applied well or badly.

That is {{eq:no-order-in-embedding}} — **the predicate needs an ordering relation
and a dot product does not have one.** And the embedding here is *far more
generous* to the date than any real text encoder: dedicated dimensions, year and
month as smooth scalars, rather than the token sequence `2024-07-15`. A real
encoder does worse, and its behaviour changes when you reformat the date, which
is the signature of a system relying on a lexical accident rather than on a
represented quantity.

**Staleness.** The measured rate tracks {{eq:stale-query-rate}} closely — 0.105
against a predicted 0.107 at a weekly rebuild and $k=8$ — so the model is worth
reasoning with directly.

The $k$ dependence is the finding to carry. At a weekly cadence, staleness runs
**1.4% at $k=1$ and 19.9% at $k=16$**: a fourteenfold increase from retrieval
depth alone, with no change to the corpus, the churn rate, or the rebuild
schedule. Raising $k$ for recall reasons ({{ch:rag-chunking}}) raises the
stale-answer rate proportionally, so **the two decisions are coupled** — and a
recall improvement can ship as a freshness regression with nobody connecting the
two, because they are owned by different people and measured on different
dashboards.

The deleted column is small and is the one that matters.
{{eq:deletion-asymmetry}}: a stale *update* gives a user an outdated number, and
a stale *deletion* gives them a confident, **cited** answer from a retracted
document — where the citation makes it more credible rather than less.

> **PRODUCTION TIP:** Set the rebuild cadence from the update rate and your
> tolerance for outdated content, but **do not let deletions wait for it.** A
> tombstone stream applied within seconds costs almost nothing beside a full
> rebuild and removes the worst failure mode in the chapter.

## 10. Production Considerations

**Capture every cheap metadata field at ingest.**
{{eq:metadata-vs-vector-cost}}: metadata is ~200× cheaper than the vectors beside
it, so there is no storage argument for a lean schema, and the retroactive cost
is a re-ingest.

**Never embed a constraint.** Dates, versions, prices, statuses, identifiers, and
permissions are filters. {{eq:no-order-in-embedding}}.

**Partition by tenant; filter within it.** Permissions are an address, not a
predicate ({{sec:5-formal-explanation}}).

**Apply deletions immediately, additions on a cadence**
({{eq:deletion-asymmetry}}).

**Make chunk ids content-derived**, so editing one paragraph does not invalidate
a whole document's chunks.

**Log the realised selectivity of every filter** alongside the planner's estimate
({{eq:correlated-selectivity}}). The gap is where the latency incidents and the
silent-recall incidents both come from.

**Write down the defaults.** Which status, which language, which recency — these
are product policy expressed as a filter, and if they live only in code nobody
who should review them ever will.

**Test the permission filter adversarially**, with a user who should see nothing.
An empty result is the correct answer and a startling number of systems have
never been asked for one.

## 11. Common Mistakes

**Putting dates and numbers in the chunk text and expecting range queries to
work.** The chapter's central error.

**Filtering for tenant isolation instead of partitioning.** One bug, one breach
({{ch:emb-vector-db}}).

**Adding metadata later.** Every field in {{sec:6-mathematical-foundation}}'s
table requires a re-ingest to backfill.

**Unnormalised filter values.** `"English"`, `"english"`, `"en"`, and `"en-GB"`
in one field is a filter with a silent false-negative rate.

**Estimating conjunction selectivity as a product.**
{{eq:correlated-selectivity}}: real predicates are correlated and the estimate is
too small.

**Positional chunk ids.** Editing one paragraph re-indexes the document.

**Raising $k$ without re-examining freshness.** {{eq:stale-query-rate}}.

**Treating a full rebuild as the deletion mechanism.**

## 12. Failure Modes

**Silent constraint violation.** The user asked for Q3 and got Q2; the answer is
fluent and cited. Undetectable without a filter-correctness test, because
retrieval metrics do not know about the constraint.

**Permission leak via post-filtering residue.** The final answer is correct and
the forbidden documents were still retrieved, scored, logged, and cached.

**Recall collapse under a selective permission filter.**
{{eq:prefilter-permissions}} below $s_c$ — the user sees *fewer* results than
they should, silently, and it looks like a sparse corpus.

**Stale deletion served with a citation.** {{eq:deletion-asymmetry}}, the worst
failure in the chapter.

**Filter-value drift.** An upstream system changes an enumeration; the filter
starts excluding everything; retrieval returns empty and is blamed on the
embedding model.

**Timezone and date-boundary errors.** "Last quarter" computed in the server's
timezone for a user in another. Unglamorous and extremely common.

**Metadata/chunk desynchronisation.** Re-chunking without re-deriving metadata,
so offsets point at the wrong span and citations become wrong in a way that looks
like hallucination.

## 13. Alternatives

**A relational database with a vector column.** Filters are the query planner's
job, which it has been doing well for fifty years, and joins to the data
generating the predicates are free. Slower at vector scale, correct by
construction, and under-chosen ({{ch:emb-vector-db}}).

**A search engine.** Mature filtering, faceting, permissions, and multi-tenancy,
with hybrid retrieval ({{ch:emb-hybrid}}) included. The vector index is usually a
generation behind and the rest is a generation ahead.

**Separate indexes per filter value.** Exact and fast at low cardinality; the
right answer for tenant and language.

**Learned sparse retrieval** ({{cite:formal2021splade}}). Puts exact-match terms
back into the retrievable representation — but note it does *not* solve ranges or
permissions, because those are still not similarity.

**Query planning over a schema.** When the constraints dominate and the semantic
part is thin, this is a database query wearing a RAG costume, and
{{ch:rag-structured}} says so.

## 14. Evaluation

**Filter correctness as a hard test**, not a metric: assert that no returned
chunk violates $\phi$. It should be exactly 100% and a failure is a bug, not a
regression.

**Permission tests with a negative case** — a user entitled to nothing must
receive nothing.

**Recall under realistic filter selectivity**, not unfiltered
({{ch:emb-vector-db}}). The published recall of every index assumes no predicate.

**Staleness rate** ({{eq:stale-query-rate}}), and separately the
deleted-document-served rate, which should be zero.

**Estimated against realised selectivity**, logged per query.

**Metadata completeness** — the fraction of chunks with each field populated.
A field present on 60% of chunks is a filter that silently drops 40% of the
corpus.

## 15. Advanced Concepts

**Retrieval is a query planning problem**, which {{ch:emb-vector-db}} noted and
this chapter makes concrete: a filtered vector search is a join between a
predicate scan and a similarity scan, and the strategy choice is cost-based. The
RAG ecosystem is re-deriving the relational planner, mostly without saying so and
therefore without its accumulated wisdom about correlated predicates and
cardinality estimation.

**Access control has three placements and only one is safe.** Filter at query
time (a predicate — one bug from a breach), partition at index time (an address —
structurally safe), or filter *after* generation (too late; the model has already
seen the content and may have used it). The third is worth naming because it is
what "just tell the model not to mention it" amounts to.

**Metadata can improve ranking, not just filtering.** Recency, authority, and
document type are legitimate ranking *signals* as well as constraints, and
combining them with a similarity score is {{ch:emb-hybrid}}'s fusion problem —
with the same warning, since a raw similarity is not calibrated and cannot be
added to a recency score without a normalisation that assumes something.

**The index is not the source of truth**, and treating it as one causes the
deletion problem. It is a derived, lossy, stale projection of the corpus, and
every property in this chapter follows from taking that seriously.

**Filtering interacts with chunking.** A filter selecting 1% of *documents* may
select 1% of *chunks* or a wildly different fraction if document lengths correlate
with the filtered attribute — long documents produce more chunks, so a filter for
"reports" over a corpus of reports and memos is far less selective at chunk level
than at document level. Selectivity must be measured where the filter is applied.

## 16. Connection to Previous Chapters

{{ch:emb-vector-db}}'s pre/post-filter analysis is applied here with permissions
as the predicate, and its partitioning conclusion is re-derived from the security
side. {{ch:rag-ingestion}} is where every metadata field originates, and
{{ch:rag-chunking}} where it is attached. {{ch:emb-hybrid}}'s capacity bound —
that an embedding cannot hold an identifier — is the same argument as
{{eq:no-order-in-embedding}}, applied to ordering rather than identity.
{{ch:emb-models}}'s schema-versioning point extends to the metadata schema, and
{{eq:segment-search}}'s buffer-and-merge pattern is what makes incremental
indexing work.

## 17. Exercises

1. Derive {{eq:stale-query-rate}} and compute the rate for 5% monthly churn, a
   weekly rebuild, and $k=8$.
2. Using {{eq:metadata-vs-vector-cost}}, compute the storage cost of ten extra
   metadata fields on a 50M-chunk index. Compare to the vectors.
3. In `constraints-not-similarity`, remove the dedicated date dimensions and
   instead append the date as text to the topic vector. Predict what happens to
   the embedded column, then check.
4. Add a second constraint (a category) to the same listing and measure whether
   the embedded path degrades multiplicatively.
5. In `index-staleness`, find the rebuild cadence that keeps the stale-answer
   rate under 2% at $k=8$. Now at $k=16$. What does that say about tuning $k$?
6. A filter selects 1% of documents. Under what condition does it select
   substantially more or less than 1% of chunks?
7. Design the negative permission test: what does the fixture look like, and what
   exactly do you assert?
8. Your filter values were never normalised. Write the migration, and say what
   has to be re-ingested and what does not.

## 18. Interview Questions

1. Why can't you just put the date in the chunk text?
2. How do you do access control in a RAG system?
3. Pre-filter or post-filter for permissions — and what is the argument that is
   not about performance?
4. What metadata does a chunk need, and which fields can be added later?
5. How fresh is your index, and how would you measure it?
6. Why does raising $k$ affect freshness?
7. Which is worse, a stale update or a stale deletion?
8. A filter returns nothing and users complain. Diagnose.
9. How do you re-index a corpus after editing three documents?
10. Your selectivity estimate was wrong. What breaks?

## 19. Research Questions

1. Is there an embedding objective that preserves an ordering
   ({{eq:ordering-preservation}}) for designated attributes without harming
   semantic similarity? It would remove the chapter's central dichotomy.
2. Can filter selectivity be estimated well over *correlated* document
   predicates, where {{eq:conjunction-selectivity}} fails badly?
3. Permission-aware ANN with a guaranteed recall bound is unsolved
   ({{ch:emb-vector-db}}). Does the security setting admit a different
   formulation — for instance, indexes built per permission class?
4. What is the right consistency model for a retrieval index? Databases have a
   vocabulary for this; RAG systems have "we rebuild nightly".
5. Metadata as a ranking signal requires combining an uncalibrated similarity
   with a calibrated attribute. Is there a principled fusion, or is this
   {{ch:emb-similarity}}'s calibration problem again?

## 20. Chapter Summary

**An embedding is a similarity function, and constraints are not similarity.**
Dates, versions, prices, statuses, identifiers, and permissions are predicates
with exact answers, and {{eq:no-order-in-embedding}} shows why a vector cannot
approximate them: the contrastive objective makes semantically related texts
near each other, and two dates are related *because they are dates*, regardless
of order. Measured, an embedding far more generous than any real encoder still
fails on range queries — and fails *worse* as the window narrows, because
similarity in the date subspace is a distance from a point rather than membership
in an interval.

**Metadata must be captured at index time or not at all.** Every field is
retroactively expensive and prospectively free — {{eq:metadata-vs-vector-cost}}
puts metadata around 200× cheaper than the vectors beside it, so a lean schema is
a false economy.

**Permissions are an address, not a predicate.** Post-filtering is correct and
still handles forbidden documents through logs, traces, and caches;
pre-filtering avoids that and silently loses recall below $s_c$. Partitioning by
tenant is the resolution, and it is the same conclusion {{ch:emb-vector-db}}
reached from latency — arrived at here from security.

**Freshness has a $k$ in it.** {{eq:stale-query-rate}}: retrieval depth
multiplies the staleness rate almost linearly, so raising $k$ for recall is
coupled to a freshness regression that nobody attributes correctly. And
{{eq:deletion-asymmetry}} says the two stale cases are not comparable — an
outdated number is a mistake, while a retracted document served *with a citation*
is a confident wrong answer made more credible by its provenance. **Apply
deletions immediately; let additions wait for the rebuild.**

## 21. Further Reading

{{cite:gao2023ragsurvey}} covers indexing as a stage and, characteristically for
the field, treats metadata as a detail.
{{cite:thakur2021beir}} matters here as a reminder that every published retrieval
number is unfiltered, and your system is not.
{{cite:formal2021splade}} for putting exact terms back into the retrievable
representation — and note it addresses identity, not ordering.
{{cite:liu2023lost}} connects: filters change how many chunks compete for context
positions, which changes where the good one lands.
The database literature on cardinality estimation and access-path selection is
the real reference for {{sec:6-mathematical-foundation}}, and it is not cited by
the RAG literature at all.

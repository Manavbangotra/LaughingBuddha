---
id: emb-vector-db
number: 102
part: XI
tier: full
status: draft
requires: [emb-what-they-are, emb-similarity, emb-models, llm-inference,
           ml-metrics]
provides: [vector-database, metadata-filtering, pre-filtering, post-filtering,
           filter-percolation, index-freshness, tombstone-deletion,
           index-segments, vector-index-memory, multi-tenancy-isolation]
citations: [malkov2020hnsw, johnson2019faiss, jegou2011pq, guo2020scann,
            thakur2021beir, karpukhin2020dpr]
---

## 1. Learning Objectives

By the end of this chapter you will be able to say what a vector database
provides beyond an index; compute the over-retrieval budget a post-filtered
query needs and show why it becomes untenable at high selectivity; explain
why pre-filtering breaks graph indexes, predict the selectivity at which it
happens from the graph's degree, and verify the prediction; reason about
deletion, freshness, and segment merging as the operational core of a vector
store; and size an index's memory before deploying it.

## 2. Why This Matters

The index is the part of a vector database that gets written about and the part
that causes the fewest production incidents.

{{ch:emb-ann}} covers the index. This chapter covers everything around it, and
the reason to give it a chapter is that **the hard problem in vector search is
not finding neighbours, it is finding neighbours that also satisfy a
predicate.** Every real query has one — this tenant, this language, this date
range, documents this user may read — and every approximate index is built on
assumptions that a predicate violates.

There are two ways to handle it and both break. Post-filtering degrades
gracefully until it does not, and {{sec:9-practical-example}} measures the
cliff. Pre-filtering breaks the graph the index depends on, at a selectivity
that {{eq:percolation-threshold}} predicts from the graph's degree alone. No
system solves this well; they differ in which failure they choose.

{{maturity:EMERGING}} The index algorithms are established. The *systems* around
them are not: filtered search, deletion, and consistency are handled differently
by every vector database on the market, the differences are material, and the
documentation rarely states which choice was made.

## 3. Prerequisites

{{ch:emb-what-they-are}} for embeddings as a versioned index schema;
{{ch:emb-similarity}} for metric choice and why it is baked in;
{{ch:emb-models}} for the migration cost that makes these decisions sticky;
{{ch:llm-inference}} for latency budgets and serving arithmetic;
{{ch:ml-metrics}} for recall.

## 4. Intuitive Explanation

### What the database adds

Given {{ch:emb-ann}}'s index, you can already answer "which vectors are nearest
this one". A production system needs rather more:

**Filtering.** Nearest *among documents matching a predicate*. This is the hard
one and the rest of the chapter is mostly about it.

**Mutation.** Documents are added, changed, and deleted continuously. An ANN
index is a structure built from a fixed set of points; changing that set is
either expensive or degrading, and often both.

**Freshness.** How long between writing a document and its being retrievable. In
a search index this is seconds; in a naive vector index it is however long a
rebuild takes.

**Durability and replication.** The index is derived data, so it can be rebuilt —
but a rebuild over a hundred million vectors is hours, which makes it a
disaster-recovery plan rather than an operational one.

**Isolation.** Multiple tenants in one index, with a hard guarantee that
tenant A never sees tenant B's documents. This is filtering again, with the
failure mode upgraded from "wrong results" to "data breach".

### Why filtering is hard

The intuition is worth getting exactly right, because it explains both failures.

An ANN index works by **not looking at most of the data**. That is the entire
value proposition: examine a few thousand of a hundred million vectors and be
confident the answer is among them. To do that, it exploits structure built when
the data was *whole*.

A predicate removes part of the data. So either you apply the index first and
filter afterwards — in which case the index, blind to the predicate, spends its
budget on documents you will discard — or you apply the predicate first and
search within the survivors — in which case the structure the index relies on
was built for a set that no longer exists.

```text
   POST-FILTER                          PRE-FILTER
   ───────────────────────────          ───────────────────────────
   search whole index for B             restrict to matching set
   throw away non-matching              search within it
                                        
   cost: B grows as 1/selectivity       cost: index structure broken
   fails when: filter is selective      fails when: filter is selective
   symptom: latency, then empty         symptom: silent recall loss
```

Both fail in the same regime and for opposite reasons, which is why "just use
the other one" is not a fix.

## 5. Formal Explanation

### 5.1 Post-filtering and the over-retrieval budget

Let $s \in (0,1]$ be the **selectivity** of a predicate — the fraction of the
corpus it admits. A post-filtering query retrieves the top $B$ by vector
similarity and keeps those satisfying the predicate.

If predicate satisfaction were independent of similarity rank, the number of
survivors is $\text{Binomial}(B, s)$ with mean $Bs$. To return $k$ results:

$$ \E[\text{survivors}] = Bs \geq k \quad\Longrightarrow\quad B \geq \frac{k}{s} $$ (eq:postfilter-budget)

and to return $k$ results *with high probability* rather than in expectation, a
tail bound adds a further factor. But {{eq:postfilter-budget}} is not the real
requirement, because returning $k$ documents is not the same as returning the
*correct* $k$. Recall of the true filtered top-$k$ requires $B$ large enough that
all $k$ true answers fall inside the unfiltered top-$B$, which is a stronger
condition — and {{sec:9-practical-example}} measures it directly.

**The scaling is the point.** At $s = 0.5$ post-filtering is free. At $s = 0.01$
you must retrieve a hundred times as many candidates. At $s = 10^{-4}$ — a
single tenant among ten thousand — you must scan the corpus, and the index has
bought you nothing.

> **WARNING:** The failure is worse than slow. A post-filtered query that finds
> too few survivors returns a *short list*, not an error. Systems that cap $B$
> for latency reasons therefore return fewer results than requested, silently,
> exactly for the most selective queries — which are usually the most important
> ones.

### 5.2 Pre-filtering and graph percolation

Pre-filtering restricts the search to the matching subset. For an inverted index
this is natural. For a graph index ({{cite:malkov2020hnsw}}) it is a structural
problem, because the search is a walk and the walk may only step on matching
nodes.

Model the filter as independent site percolation: each node is retained with
probability $s$. A random graph with mean degree $\langle \deg \rangle$ has a
giant connected component only above a threshold:

$$ s_c \;\approx\; \frac{1}{\langle \deg \rangle} $$ (eq:percolation-threshold)

Below $s_c$ the retained subgraph shatters into many small components, and a
greedy walk starting anywhere can reach only its own component. **The index does
not report this.** It returns the best node it could reach, which may be
arbitrarily bad, and recall degrades silently.

{{eq:percolation-threshold}} is a genuinely useful engineering result because
$\langle \deg \rangle$ is a parameter you chose: HNSW's $M$ sets it, so

$$ s_c \approx \frac{1}{2M} \quad\text{and}\quad \text{a filter more selective than } s_c \text{ will lose recall} $$ (eq:hnsw-filter-threshold)

With the common $M = 16$, that is a selectivity around 3%. {{sec:9-practical-example}}
builds the graph, measures its degree, predicts the threshold, and confirms it.

> **MATH NOTE:** The percolation model assumes the filter is independent of the
> graph structure. Real filters are usually *correlated* with position — a
> language filter, a tenant filter, a date filter all select regions of the
> embedding space — and correlated filters percolate at lower $s$ than
> {{eq:percolation-threshold}} predicts, because the retained nodes are near
> each other. This is the rare case where the real world is kinder than the
> model, and it is why partitioning by the filter attribute works so well.

### 5.3 The strategies actually deployed

| Strategy | Cost | Fails when | Used by |
|---|---|---|---|
| post-filter | $B = O(k/s)$ | $s$ small | most, as the default |
| pre-filter + brute force | $O(sN)$ | $sN$ large | most, below a size cutoff |
| filtered graph traversal | near-normal | $s < s_c$ | graph indexes, with repairs |
| partition by attribute | near-normal | high-cardinality or multi-attribute | tenant isolation |
| filter-aware index build | build-time | filters not known in advance | specialised |

**The pragmatic composition that most systems converge on**, and worth stating
plainly because the documentation rarely does: estimate $s$ from statistics,
brute-force if $sN$ is small, post-filter if $s$ is large, and traverse with
repairs in between. The estimate is where it goes wrong.

## 6. Mathematical Foundation

### 6.1 Choosing between the strategies

Brute force over the filtered set costs $O(sNd)$. Post-filtered ANN costs
roughly $O\big(\frac{k}{s}\log N \cdot d\big)$ for a graph index. Equating:

$$ sNd \;=\; \frac{k}{s}\,d\log N \quad\Longrightarrow\quad s^{*} = \sqrt{\frac{k \log N}{N}} $$ (eq:strategy-crossover)

For $N = 10^7$, $k = 10$: $s^* \approx \sqrt{161/10^7} \approx 0.004$. **Below
about 0.4% selectivity, brute-forcing the filtered set beats post-filtering** —
which is a surprisingly large crossover and explains why "just scan it" is the
right answer more often than teams expect.

### 6.2 Memory, which decides everything else

$$ \text{bytes} \;=\; \underbrace{N \cdot d \cdot b}_{\text{vectors}} \;+\; \underbrace{N \cdot 2M \cdot 4}_{\text{graph edges}} \;+\; \underbrace{N \cdot (\text{payload})}_{\text{ids, metadata}} $$ (eq:index-memory)

with $b$ bytes per component. For $N = 10^7$, $d = 768$, $b = 4$, $M = 16$:
vectors are 30.7 GB, edges 1.3 GB, so the vectors dominate by more than 20×.

Two consequences follow immediately and they run the rest of {{part:11}}.
**First, dimension reduction and quantization are the only levers that matter**
for index size — hence {{ch:emb-models}}'s nested embeddings and
{{ch:emb-ann}}'s product quantization. **Second, the graph is nearly free**, so
raising $M$ to buy connectivity — and, by {{eq:hnsw-filter-threshold}}, filter
robustness — costs about 1% of the index per unit.

### 6.3 Deletion and the tombstone

Graph indexes cannot cheaply delete: removing a node severs the paths through
it, and repairing those paths means re-running the neighbour selection for every
node that pointed at it. So systems mark deleted nodes and skip them at query
time — a **tombstone**.

Tombstones are traversable but not returnable, so with a deleted fraction $\delta$
the effective search budget shrinks:

$$ \text{effective candidates} \;\approx\; \text{ef} \cdot (1 - \delta) $$ (eq:tombstone-degradation)

Recall decays smoothly as $\delta$ grows, with no error and no metric that moves
unless someone is watching recall specifically. **This is the single most common
"our vector search got worse and nothing changed" cause**, and the remedy is a
rebuild triggered by $\delta$ crossing a threshold — typically 10–20%.

### 6.4 Freshness and the segment pattern

The resolution every mature system reaches, borrowed wholesale from Lucene:
maintain one large immutable index plus a small mutable buffer, query both,
merge periodically.

$$ \text{query cost} = C_{\text{main}} + C_{\text{buffer}}, \qquad \text{freshness lag} = \text{buffer flush interval} $$ (eq:segment-search)

The buffer is brute-forced because it is small, which is why freshness is cheap
to buy. The merge is where the cost lands, and it is why write-heavy vector
workloads are much harder to operate than read-heavy ones.

### 6.5 Sharding, and why vector search scales badly across machines

Relational and inverted-index workloads shard well because a query usually
touches one shard: the predicate names a key, and the router sends the query
there. **A nearest-neighbour query names no key.** The nearest vectors may be
anywhere, so every shard must be searched and the results merged:

$$ \text{latency} = \max_{j=1..S} \text{latency}_j + \text{merge}, \qquad \text{work} = S \cdot \text{work}_j $$ (eq:shard-fanout)

Two consequences that surprise teams arriving from relational systems.

**Adding shards does not reduce tail latency.** The query waits for the slowest
of $S$ responses, so the p99 of the whole is roughly the p99$^{1/S}$ quantile of
each shard — the fan-out *amplifies* tail latency, which is the standard
scatter-gather problem. Doubling the shard count halves the per-shard work and
raises the probability that at least one shard is slow.

**Each shard must return more than $k$.** The global top-$k$ may be
distributed unevenly across shards, so each returns $k' > k$ and the merge picks
the true top-$k$. Setting $k' = k$ is a correctness bug that shows as a small,
persistent recall loss.

The escape is to shard *by the filter attribute* wherever one exists — tenant,
language, region — because then the predicate does name a key and the fan-out
collapses to one shard. This is the same insight as {{sec:5-formal-explanation}}'s
partitioning recommendation, arriving from the latency side rather than the
correctness side, and it is the strongest architectural argument in the chapter:
**a filter you can shard on is worth more than any index tuning.**

## 7. Internal Mechanics

```mermaid {#fig:filter-strategies caption="How a query is planned. The selectivity estimate is the load-bearing part and the usual source of error: a wrong estimate does not merely pick a slower path, it can pick one that silently loses recall."}
flowchart TD
    Q["query + predicate"] --> E["estimate selectivity s"]
    E --> C{"s x N small?"}
    C -->|yes| BF["brute force the<br/>filtered set — exact"]
    C -->|no| D{"s > s_c<br/>(eq:percolation-threshold)"}
    D -->|yes| PF["post-filter with<br/>B = k/s (eq:postfilter-budget)"]
    D -->|no| G["filtered traversal<br/>+ connectivity repair"]
    G -.->|"if estimate was wrong"| L["silent recall loss"]
    PF -.->|"if B capped for latency"| S["short result list"]
```

### 7.1 Repairs for filtered traversal

When $s < s_c$ the graph shatters, and the deployed mitigations all amount to
adding entry points or edges:

- **Multiple random entry points** among matching nodes. Cheap, and it converts
  "one small component" into "a few small components" — a partial fix that
  raises effective recall without restoring connectivity.
- **Traverse through non-matching nodes**, filtering only the returned set. This
  preserves connectivity exactly and is the most common real implementation; it
  costs a larger $\text{ef}$ because most visited nodes are unusable.
- **Higher $M$ at build time**, which lowers $s_c$ directly by
  {{eq:hnsw-filter-threshold}} and costs ~1% of index memory per unit
  ({{eq:index-memory}}).
- **Separate indexes per filter value**, which is exact and only works for low
  cardinality.

The second is the one to reach for by default: it keeps the guarantee and pays in
a knob that is already tunable.

### 7.2 The selectivity estimate

{{fig:filter-strategies}}'s planner turns on one number, and that number is
estimated the way a relational planner estimates one: from statistics gathered
at index build time — histograms per attribute, distinct-value counts,
correlation assumed away.

All three of the classical estimation errors apply, and they matter more here
because the consequence is silent rather than slow:

- **Correlated predicates.** `language = "de" AND region = "EU"` is estimated as
  the product of two marginals and is in fact far larger. The planner picks
  brute force for a set that turns out to be huge.
- **Skew.** A tenant filter averaging 0.01% selectivity has one tenant at 30%.
  Estimating from the mean picks the wrong path for both.
- **Staleness.** Statistics from last quarter's corpus.

The relational lesson transfers directly: **when the estimate is uncertain,
prefer the strategy that degrades gracefully.** Post-filtering with a generous
budget gets slow when the estimate is wrong; filtered traversal loses recall
silently. Given a choice under uncertainty, take the one that shows up in a
latency graph.

### 7.3 Multi-tenancy

Tenant isolation is filtering where a miss is a security incident, and the
options rank differently for that reason.

| Approach | Isolation | Cost |
|---|---|---|
| index per tenant | strongest | poor at high tenant count; per-index overhead dominates |
| partition per tenant, one index | strong | good; the usual answer |
| filter on a tenant field | weakest | depends entirely on correct filtering |

**Filtering for isolation puts a correctness bug one line away from a data
breach**, and the failure is silent — one tenant's document in another's result
list looks like an ordinary result. Partition, and treat the tenant key as part
of the address rather than as a predicate.

## 8. Implementation

```python {tier=A name=post-filter-overretrieval}
"""What a metadata filter costs a post-filtered search.

Post-filtering retrieves the top B by vector similarity and discards
non-matching results. We measure recall of the TRUE filtered top-k -- the k
nearest documents that actually satisfy the predicate -- as a function of the
predicate's selectivity and the retrieval budget B.

The predicate here is independent of position, which is the favourable case.
A predicate correlated with the embedding is worse.
"""
import numpy as np

rng = np.random.default_rng(31)

N, DIM, K = 20_000, 32, 10
N_QUERY = 200
BUDGETS = [100, 500, 2000]
LADDER = [100, 200, 500, 1000, 2000, 5000, 10_000, 20_000]

X = rng.normal(size=(N, DIM))
X /= np.linalg.norm(X, axis=1, keepdims=True)
queries = rng.normal(size=(N_QUERY, DIM))
queries /= np.linalg.norm(queries, axis=1, keepdims=True)
sims = queries @ X.T                       # exact scores; the index is not the point


def recall_at_budget(sims, mask, truth, budget):
    """Retrieve top-`budget` ignoring the filter, then keep matching results."""
    budget = min(budget, sims.shape[1] - 1)
    cand = np.argpartition(-sims, budget, axis=1)[:, :budget]
    hits = []
    for i in range(len(sims)):
        kept = cand[i][mask[cand[i]]]
        hits.append(len(set(kept.tolist()) & truth[i]) / K)
    return float(np.mean(hits))


print(f"{'selectivity':>12}{'matching':>10}"
      + "".join(f"{'B=' + str(b):>10}" for b in BUDGETS)
      + f"{'B for 95%':>12}{'as % corpus':>13}")
print("-" * 79)

for sel in [0.5, 0.2, 0.05, 0.01, 0.002]:
    mask = rng.random(N) < sel
    idx = np.flatnonzero(mask)
    # Ground truth: the k nearest documents that SATISFY the predicate.
    truth_idx = idx[np.argsort(-sims[:, idx], axis=1)[:, :K]]
    truth = [set(row.tolist()) for row in truth_idx]

    row = [recall_at_budget(sims, mask, truth, b) for b in BUDGETS]

    needed = None
    for b in LADDER:
        if recall_at_budget(sims, mask, truth, b) >= 0.95:
            needed = b
            break

    print(f"{sel:>12.3f}{len(idx):>10d}"
          + "".join(f"{r:>10.3f}" for r in row)
          + f"{str(needed):>12}{100 * (needed or N) / N:>12.1f}%")

print("""
Read the B for 95% column. It is very close to k/selectivity -- which is
eq:postfilter-budget, confirmed. The budget a post-filtered query needs scales as
the RECIPROCAL of selectivity, so a filter that admits half the corpus is free
and one that admits a fifth of a percent requires scanning a quarter of it.

At that point the index has bought nothing. This is the arithmetic behind
eq:strategy-crossover: past a certain selectivity, brute-forcing the filtered set
is simply cheaper than post-filtering, and the crossover is at a higher
selectivity than most people guess.

Now note what the low-budget columns do. At selectivity 0.002 and B=100, recall
is 0.029 -- the query returned SOMETHING, ranked plausibly, and it was almost
entirely wrong. A production system with a latency-capped B does exactly this,
and it does it worst on the most selective queries, which are usually the ones a
user has narrowed deliberately.""")
```

```python {tier=A name=pre-filter-connectivity}
"""Why pre-filtering breaks a graph index, and at exactly which selectivity.

A graph index answers queries by walking from node to neighbour. Pre-filtering
means the walk may only step on nodes that satisfy the predicate -- which is
site percolation on the graph, and percolation has a threshold.

We build a k-NN graph, retain each node independently with probability s, and
measure the largest connected component of the retained subgraph. Everything
outside that component is unreachable by a walk that starts inside it.
"""
import numpy as np
from collections import deque

rng = np.random.default_rng(41)

N, DIM, M = 6000, 24, 16
TRIALS = 4

X = rng.normal(size=(N, DIM))
X /= np.linalg.norm(X, axis=1, keepdims=True)
sims = X @ X.T
np.fill_diagonal(sims, -np.inf)

# A k-NN graph, symmetrised -- which is what a graph index maintains.
knn = np.argpartition(-sims, M, axis=1)[:, :M]
neighbours = [set() for _ in range(N)]
for i in range(N):
    for j in knn[i]:
        neighbours[i].add(int(j))
        neighbours[int(j)].add(i)

mean_degree = float(np.mean([len(s) for s in neighbours]))
predicted_sc = 1.0 / mean_degree


def components(mask):
    """Largest component as a fraction of retained nodes, and component count."""
    remaining = set(np.flatnonzero(mask).tolist())
    sizes = []
    while remaining:
        start = remaining.pop()
        seen, queue = {start}, deque([start])
        while queue:
            u = queue.popleft()
            for v in neighbours[u]:
                if v in remaining:
                    remaining.discard(v)
                    seen.add(v)
                    queue.append(v)
        sizes.append(len(seen))
    return max(sizes) / int(mask.sum()), len(sizes)


print(f"mean degree after symmetrisation: {mean_degree:.1f}")
print(f"predicted percolation threshold s_c = 1/degree = {predicted_sc:.3f}"
      f"   (eq:percolation-threshold)\n")
print(f"{'selectivity':>12}{'retained':>10}{'largest component':>19}"
      f"{'components':>12}{'reachable':>11}")
print("-" * 66)

for sel in [1.0, 0.5, 0.3, 0.2, 0.15, 0.10, 0.05, 0.02]:
    fracs, counts, kept = [], [], []
    for _ in range(TRIALS):
        mask = np.ones(N, bool) if sel == 1.0 else rng.random(N) < sel
        f, c = components(mask)
        fracs.append(f)
        counts.append(c)
        kept.append(int(mask.sum()))
    frac = float(np.mean(fracs))
    flag = "ok" if frac > 0.9 else ("degraded" if frac > 0.3 else "SHATTERED")
    print(f"{sel:>12.2f}{int(np.mean(kept)):>10d}{frac:>19.3f}"
          f"{np.mean(counts):>12.1f}{flag:>11}")

print(f"""
The largest-component column IS the recall ceiling for a filtered graph walk. A
greedy search enters at one node and can only reach that node's component, so
whatever fraction of the retained set lies outside it is unreachable -- no matter
how large ef is, and with no error reported.

The collapse is not gradual. Down to selectivity 0.3 the retained subgraph is
essentially intact. By 0.05 the largest component holds under 6% of retained
nodes and the subgraph has shattered into more than a hundred pieces. The
transition sits right around the predicted threshold of {predicted_sc:.3f}, which
is eq:percolation-threshold doing real work: the mean degree is a parameter you
CHOSE at build time, so this number is knowable before deployment.

Two engineering readings follow. First, raising M lowers s_c proportionally and
costs about 1% of index memory per unit (eq:index-memory) -- an unusually cheap
way to buy filter robustness. Second, and more important: the honest fix is to
traverse THROUGH non-matching nodes and filter only the returned set, which
preserves connectivity exactly and pays in a larger ef. Restricting the walk
itself is the version that fails silently.""")
```

## 9. Practical Example

**Post-filtering.** The budget required for 95% recall tracks $k/s$ closely,
confirming {{eq:postfilter-budget}}. At 50% selectivity, 100 candidates suffice;
at 0.2% selectivity, 5,000 are needed — a quarter of the corpus, at which point
the index has bought nothing and {{eq:strategy-crossover}} says to brute-force
instead.

The low-budget columns are the operational lesson. At 0.2% selectivity with a
budget of 100, recall is **0.029**. The query returned a plausible-looking ranked
list that was almost entirely wrong. Any system that caps $B$ for latency does
this, and does it worst precisely on the queries a user has narrowed
deliberately.

**Pre-filtering.** The graph has mean degree 18.1, so
{{eq:percolation-threshold}} predicts a threshold near 0.055. Measured: at
selectivity 0.30 the largest component still holds 99.7% of retained nodes; at
0.15 it holds 92%; at 0.10, 70%; at 0.05 it holds **5.9%**, with the subgraph in
over 160 pieces. The transition lands where predicted.

That largest-component fraction *is* the recall ceiling for a filtered walk,
because a greedy search enters at one node and cannot leave its component. No
value of $\text{ef}$ helps, and nothing reports an error.

> **IMPORTANT:** The two experiments describe the same regime from opposite
> sides. Post-filtering fails at low selectivity by getting slow and then
> returning short lists; pre-filtering fails at low selectivity by silently
> losing recall. **A selective filter is hard for a vector index, full stop** —
> and the correct response is usually neither strategy but partitioning, which
> makes the filter part of the address rather than a predicate.

## 10. Production Considerations

**Instrument selectivity per query.** {{fig:filter-strategies}}'s planner depends
on an estimate, and a wrong estimate silently picks a failing path. Log the
estimate and the realised value; the gap is your bug rate.

**Set $M$ from your filters, not from a benchmark.** {{eq:hnsw-filter-threshold}}
converts the most selective filter you must support into a minimum $M$, and
{{eq:index-memory}} shows the cost is about 1% per unit. Benchmark-tuned $M$
values are tuned for unfiltered recall.

**Monitor the deleted fraction and rebuild on a threshold.**
{{eq:tombstone-degradation}}: recall decays smoothly and nothing alerts. A
rebuild at $\delta = 0.15$ is a reasonable default.

**Partition for tenancy.** Never filter for isolation.

**Size memory before choosing a model.** {{eq:index-memory}} is dominated by
$N \cdot d \cdot b$; run that arithmetic while the dimension is still a choice.

**Shard on a filter attribute if you have one.** {{eq:shard-fanout}}: it turns a
scatter-gather into a point query, and it is worth more than any index tuning.

**Set each shard's $k'$ above $k$.** Otherwise the merge silently loses results
whose distribution across shards is uneven.

**Measure recall against exact search continuously**, on a fixed probe set. It is
the only signal that catches tombstone decay, filter percolation, and index
corruption — none of which move a latency or error-rate metric. This is
{{ch:llm-routing}}'s probe-set argument, in a different system.

## 11. Common Mistakes

**Assuming the filter is free.** It is the most expensive part of the query at
any interesting selectivity.

**Capping the retrieval budget for latency without capping selectivity.** Produces
{{sec:9-practical-example}}'s 0.029-recall query, silently.

**Treating a short result list as "no matches".** It usually means the budget ran
out, which is a different fact with a different fix.

**Filtering for multi-tenant isolation.** One bug, one breach.

**Deleting by tombstone and never rebuilding.**

**Benchmarking unfiltered and deploying filtered.** The published recall numbers
for every index assume no predicate.

**Choosing a database on index benchmarks.** The index is the commodity part;
filtering, deletion, and freshness are where they actually differ, and where the
documentation is thinnest.

## 12. Failure Modes

**Silent recall loss from filter percolation.** Below $s_c$, with no error. Only
a probe set catches it.

**Tombstone decay.** {{eq:tombstone-degradation}}, over weeks.

**Short lists on selective queries.** The system returns three results for $k=10$
and the caller renders three results.

**Selectivity estimate drift.** Statistics computed at index build time; the
corpus's attribute distribution shifts; the planner keeps choosing the path that
was right last quarter.

**Fan-out tail amplification.** {{eq:shard-fanout}}: a p99 that was acceptable on
one shard is not acceptable as the max over sixteen. Adding shards to fix a
latency problem can make the tail worse.

**Merge storms.** {{eq:segment-search}}'s merge is expensive, and a write burst
schedules several at once, spiking latency at the worst moment.

**Cross-tenant leakage.** The filter was right in the query path and absent in
the fallback path.

**Rebuild-window unavailability.** The rebuild takes six hours and the index is
degraded throughout, which is discovered the first time it is needed.

## 13. Alternatives

**pgvector or another in-database index.** The vectors sit beside the data that
generates the predicates, so filtering is the query planner's problem — a
genuinely different and often better answer to this chapter's central question.
Slower at scale, correct by construction.

**A search engine with vector support** (Elasticsearch, OpenSearch). Mature
filtering, mature multi-tenancy, mature operations, and hybrid search
({{ch:emb-hybrid}}) for free. The vector index is usually a generation behind.

**A library, not a database** ({{cite:johnson2019faiss}}). Full control, and you
implement filtering, persistence, replication, and deletion yourself. Correct for
a static corpus; a trap for a mutable one.

**No index at all.** Brute force is exact, trivially filterable, and fast enough
below roughly a million vectors — and {{eq:strategy-crossover}} shows it wins
outright when filters are selective. This is the most under-used option in the
whole part.

## 14. Evaluation

**Recall against exact search, at your filter selectivities.** Unfiltered recall
is not the number you will operate at.

**Latency at the p95 and p99, sliced by selectivity.** The mean hides the
selective queries entirely.

**Result-set completeness.** What fraction of queries returned fewer than $k$?
Nobody measures this and it is where post-filtering's failure lives.

**Freshness lag.** Write-to-retrievable, at the p99.

**Recall after simulated deletion.** Delete 20% and re-measure. This is the
tombstone budget, and it tells you the rebuild cadence.

**Rebuild wall-clock.** It is the recovery time objective, whether or not anyone
has written it down.

## 15. Advanced Concepts

**Filter-aware index construction.** If the filter attributes are known at build
time, build the graph so that each attribute's induced subgraph is itself
connected — adding edges within each attribute class. It defeats
{{eq:percolation-threshold}} by construction, at the cost of a larger graph and a
commitment to a fixed filter set.

**Correlated filters percolate better.** {{sec:5-formal-explanation}}'s note: real
predicates select regions of embedding space, and retained nodes are then
neighbours. This is why partitioning by the filter attribute works so much better
than the independent model suggests, and why synthetic filter benchmarks are
pessimistic.

**Streaming updates and the freshness/recall trade.** {{eq:segment-search}}'s
buffer is brute-forced, so a large buffer is fresh and slow while a small one is
stale and fast. The knob is the flush interval, and it is the same trade-off as
{{ch:llm-inference}}'s batching.

**Disk-based indexes.** {{eq:index-memory}} says vectors dominate; DiskANN-style
designs keep quantized vectors in memory for traversal and full vectors on SSD
for final scoring. It changes the cost model from RAM to IOPS, which is often an
order of magnitude cheaper.

**Vector search as a join.** The framing that makes the whole chapter click: a
filtered vector query is a join between a predicate scan and a similarity scan,
and everything here is the same join-ordering problem a relational planner
solves. Post-filter is similarity-first; pre-filter is predicate-first; the
crossover ({{eq:strategy-crossover}}) is a cost-based decision. Vector databases
are re-deriving query planning, mostly without saying so.

## 16. Connection to Previous Chapters

{{ch:emb-what-they-are}}'s schema-versioning point is why every decision here is
sticky. {{ch:emb-similarity}}'s metric choice is compiled into the index.
{{ch:emb-models}}'s dimension result is what {{eq:index-memory}} bills for, and
its nested embeddings are the lever for shrinking it. {{ch:llm-inference}}'s
latency budgets and replication apply unchanged. {{ch:llm-routing}}'s probe set
reappears as the only instrument that sees silent recall decay — the third time
in the book that a fixed probe set turns out to be the thing that catches what
aggregate metrics cannot.

## 17. Exercises

1. Derive {{eq:strategy-crossover}} and compute $s^*$ for $N = 10^6$ and
   $N = 10^9$ at $k = 10$. Why does the crossover *fall* with $N$?
2. Use {{eq:index-memory}} to size an index for 50M documents at 768 float32
   dimensions with $M = 32$. Now at 256 dimensions. What did the reduction save,
   and what did it cost by {{ch:emb-models}}'s argument?
3. In `post-filter-overretrieval`, make the predicate *correlated* with the
   query direction. Does the budget improve or worsen, and why?
4. In `pre-filter-connectivity`, vary `M` over $\{8, 16, 32\}$ and confirm the
   threshold moves as $1/2M$.
5. Add a repair to `pre-filter-connectivity`: allow the walk to pass through
   non-matching nodes. Measure the reachability restored and the extra nodes
   visited.
6. Simulate tombstones: mark a fraction $\delta$ as deleted, keep them
   traversable, and measure recall against {{eq:tombstone-degradation}}.
7. Design the monitoring for a filtered vector index. What are the three metrics,
   and which failure does each catch?
8. Your corpus is 200M vectors with a mandatory per-user filter averaging 0.01%
   selectivity. Design the system. (There is a right answer and it is not an ANN
   index.)

## 18. Interview Questions

1. Why is filtering hard in a vector database?
2. Pre-filter or post-filter — when each?
3. At what selectivity does post-filtering stop working, and why?
4. What happens to a graph index when you restrict the walk to a subset?
5. How do you delete from an HNSW index?
6. Our vector search got worse over three months, no deploys. Diagnose.
7. Multi-tenant isolation: filter or partition?
8. What dominates a vector index's memory?
9. When would you not use a vector index at all?
10. How do you know your ANN index is still returning good results?

## 19. Research Questions

1. Is there an index with a *guaranteed* recall bound under arbitrary filters,
   rather than one that degrades silently?
2. Can {{eq:percolation-threshold}} be made adaptive — a graph that adds edges on
   demand when a filter shatters it, amortised across queries?
3. What is the right cost model for a filtered vector query, and can a
   relational planner's machinery be applied directly given
   {{sec:15-advanced-concepts}}'s join framing?
4. Deletion in graph indexes is unsolved. Is there a structure with logarithmic
   search and cheap deletion, or is that trade-off fundamental?
5. Real filters are correlated with embedding position. Is there a usable
   *measure* of that correlation that predicts percolation better than $s$?

## 20. Chapter Summary

A vector database is an index plus the things that make it operable: filtering,
mutation, freshness, durability, and isolation. The index is the commodity part.

**Filtering is the hard problem and both strategies fail in the same regime.**
Post-filtering needs a budget scaling as $k/s$ ({{eq:postfilter-budget}}) —
measured, 5,000 candidates from a 20,000-document corpus at 0.2% selectivity —
and when the budget is capped it returns a plausible list with 0.029 recall.
Pre-filtering is site percolation on the index graph, shattering below
$s_c \approx 1/\langle\deg\rangle$ ({{eq:percolation-threshold}}) — measured, the
largest component falls from 99.7% at $s=0.3$ to 5.9% at $s=0.05$ on a graph of
mean degree 18.1, exactly where predicted. That fraction is a hard recall ceiling
no amount of search effort can lift.

The response is rarely to pick a strategy. Brute force wins outright below
$s^* = \sqrt{k\log N / N}$ ({{eq:strategy-crossover}}), which is around 0.4% at
ten million documents; partitioning makes the filter part of the address; and
raising $M$ buys filter robustness for about 1% of index memory per unit.

**Everything else is operational and silent.** Tombstones decay recall smoothly
({{eq:tombstone-degradation}}), freshness comes from a brute-forced buffer
({{eq:segment-search}}), memory is dominated by the vectors themselves
({{eq:index-memory}}), and isolation must be partitioning rather than filtering
because a filtering bug is a breach. None of these move a latency or error-rate
metric, which is why a fixed probe set measured against exact search is the only
monitoring that works.

## 21. Further Reading

{{cite:malkov2020hnsw}} for what the graph assumes — the connectivity argument in
Section 4 is what {{eq:percolation-threshold}} attacks.
{{cite:johnson2019faiss}} for the library view and IVF's natural affinity with
partition-based filtering.
{{cite:jegou2011pq}} and {{cite:guo2020scann}} for the compression that
{{eq:index-memory}} makes necessary.
{{cite:thakur2021beir}} as the reminder that all of this serves retrieval
quality, which is measured elsewhere.

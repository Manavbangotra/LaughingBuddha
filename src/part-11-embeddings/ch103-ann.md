---
id: emb-ann
number: 103
part: XI
tier: full
status: draft
requires: [emb-similarity, emb-vector-db, math-norms, ml-clustering,
           emb-what-they-are]
provides: [approximate-nearest-neighbour, recall-qps-frontier, proximity-graph,
           hnsw-parameters, inverted-file-index, product-quantization,
           asymmetric-distance-computation, anisotropic-quantization,
           locality-sensitive-hashing, ann-rerank-cascade]
citations: [malkov2020hnsw, malkov2014nsw, jegou2011pq, johnson2019faiss,
            guo2020scann, indyk1998lsh, aumuller2020annbench, beyer1999nn]
---

## 1. Learning Objectives

By the end of this chapter you will be able to implement a proximity-graph index
and a product quantizer from scratch; read and report a recall/QPS frontier
rather than a single operating point; explain what HNSW's $M$ and $\text{ef}$
actually control and set them from requirements; derive why asymmetric distance
computation beats symmetric and measure the gap; explain ScaNN's anisotropic
insight and why it generalises beyond ANN; and combine a compressed index with
exact reranking, which is how every large deployment actually works.

## 2. Why This Matters

{{ch:emb-similarity}} established that exact nearest-neighbour search is
meaningful for learned embeddings. It is also $O(Nd)$ per query, which at a
hundred million 768-dimensional vectors is 300 GFLOPs — about a second of a
modern CPU core, for one query.

Approximation is what makes vector search a product rather than a research demo,
and the trade is unusually favourable: {{sec:9-practical-example}} reaches 99.4%
recall while examining 3.8% of the corpus — a 26× speedup. **That ratio —
near-exact answers for a twenty-sixth of the work, and a 56× speedup if 91%
recall suffices — is the whole reason {{part:12}} exists**, because a
retrieval step that costs a second per query cannot sit in front of a language
model.

The second reason for the chapter is that these are the parameters people
actually tune, usually by copying a blog post. $M$, $\text{ef}$, $n_{\text{probe}}$,
and the number of PQ subquantizers each control a specific, derivable trade-off,
and knowing which one to move is the difference between a ten-minute fix and a
week.

{{maturity:ESTABLISHED}} Graph indexes, IVF, and PQ are stable, well-understood,
and shipped everywhere. The frontier is in filtered search ({{ch:emb-vector-db}})
and in disk-resident variants, not in the core algorithms.

## 3. Prerequisites

{{ch:emb-similarity}} for metrics, the concentration result, and why inner
product is not a metric; {{ch:emb-vector-db}} for filtering and the memory
budget; {{ch:ml-clustering}} for $k$-means, which is the whole of IVF and PQ;
{{ch:math-norms}} for distance algebra; {{ch:emb-what-they-are}} for why the
vectors look the way they do.

## 4. Intuitive Explanation

Three families, three different things to give up.

**Graphs give up completeness.** Build a network where each vector links to
some of its near neighbours, then answer a query by starting somewhere and
repeatedly walking to whichever neighbour is closer. You examine a few hundred
vectors instead of a hundred million. What you give up is any guarantee that the
walk found the true nearest — it can get stuck in a local minimum, and the only
defence is to keep more candidates alive.

**Partitions give up coverage.** Cluster the vectors, and at query time search
only the few clusters nearest the query. Cheap, simple, and it fails when the
true nearest neighbour sits just across a cluster boundary. The knob is how many
clusters to probe.

**Quantization gives up precision.** Replace each vector with a short code and
compute distances on codes. Nothing is skipped — every vector is still scored —
but every score is approximate. The knob is the code length.

The three are **composable, not competing**, and the standard large-scale
configuration uses all of them: partition to narrow the candidate set, quantize
to make scoring cheap, then rerank the survivors with exact vectors.

### The one framing to keep

Every ANN index is a **cascade** — the same architecture as {{ch:emb-reranking}}'s
retrieve-then-rerank and {{ch:llm-routing}}'s model cascade. A cheap, imprecise
stage produces candidates; an expensive, precise stage orders them. This is the
fourth time the pattern appears in the book, and by now it should be recognised
on sight rather than re-derived.

{{sec:9-practical-example}} shows why it is not optional: a product quantizer
compressing 32× returns the true top-10 only 39.1% of the time on its own, and
92.0% of the time when its top-100 is rescored with exact vectors — and at 64×
compression, 22.4% becomes 71.6%. **The compression is a candidate generator, not
an answer.**

## 5. Formal Explanation

### 5.1 What "approximate" means

For query $q$, corpus $\mathcal{D}$, and true top-$k$ set $T_k(q)$, an index
returns $\hat{T}_k(q)$. The quality measure is

$$ \text{recall@}k = \frac{|\hat{T}_k(q) \cap T_k(q)|}{k} $$ (eq:ann-recall)

**This is not retrieval quality.** {{ch:emb-what-they-are}} drew the line and it
matters most here: {{eq:ann-recall}} measures fidelity to *exact search over the
same embedding*, so an index at 100% recall over a bad embedding retrieves the
wrong documents perfectly. Index recall is a component metric.

An ANN result is a *curve*, not a number ({{cite:aumuller2020annbench}}):

$$ \mathcal{F} = \big\{\,(\text{recall}(\theta),\ \text{throughput}(\theta))\ :\ \theta \in \Theta\,\big\} $$ (eq:recall-qps-frontier)

over the index's parameters $\theta$. Reporting one point is meaningless, since
any index reaches any recall by working harder. **The question is always: at
your required recall, what does it cost?**

### 5.2 Proximity graphs

{{cite:malkov2014nsw}}'s observation: build a graph by inserting points one at a
time, connecting each to the $M$ nearest already-inserted points. Early
insertions have few candidates, so their links are *long*; later ones are short.
The result has the small-world property — long links for coarse navigation,
short links for precision — from insertion order alone, with no explicit design.

Search is greedy best-first with a candidate list of size $\text{ef}$:

$$ \text{visit } u^{*} = \argmin_{u \in C} d(q, u), \quad C \leftarrow C \cup \mathcal{N}(u^{*}), \quad |C| \leq \text{ef} $$ (eq:greedy-graph-search)

$\text{ef} = 1$ is pure hill-climbing and gets stuck. Larger $\text{ef}$ keeps
alternatives alive and escapes local minima, and

$$ \text{recall} \nearrow \text{ and } \text{cost} \nearrow \quad \text{monotonically in ef} $$ (eq:ef-monotone)

which is what makes $\text{ef}$ the *right* runtime knob: it is monotone, it
requires no rebuild, and it traverses {{eq:recall-qps-frontier}} directly.

{{cite:malkov2020hnsw}} adds a hierarchy — each point appears in layer $\ell$
with probability decaying geometrically, search descends coarse-to-fine — giving

$$ \E[\text{hops}] = O(\log N) $$ (eq:hnsw-complexity)

**Two parameters, two lifecycles**, and conflating them is the usual mistake:

| Parameter | When | Controls | Changing it |
|---|---|---|---|
| $M$ | build | graph degree, memory, filter robustness | requires a rebuild |
| $\text{ef}_{\text{construction}}$ | build | graph quality | requires a rebuild |
| $\text{ef}$ | query | recall/latency | free, per query |

### 5.3 IVF

Cluster $\mathcal{D}$ into $n_{\text{list}}$ Voronoi cells by $k$-means
({{ch:ml-clustering}}); at query time probe the $n_{\text{probe}}$ nearest
centroids:

$$ \text{cost} \approx \underbrace{n_{\text{list}} \cdot d}_{\text{centroid scan}} + \underbrace{\frac{n_{\text{probe}}}{n_{\text{list}}} \cdot N \cdot d}_{\text{cell scan}} $$ (eq:ivf-cost)

Minimising over $n_{\text{list}}$ gives the standard heuristic
$n_{\text{list}} \approx \sqrt{N}$, at which both terms are $O(\sqrt{N} d)$.

The failure is structural: a true neighbour in an unprobed cell is invisible, and
**this happens exactly for queries near a cell boundary** — which, in high
dimension, is most of them. Raising $n_{\text{probe}}$ is the fix and the cost.

IVF's advantage over graphs is not speed; it is that **the partition is
explicit**. It composes naturally with metadata partitioning
({{ch:emb-vector-db}}), it supports deletion cheaply, and it can live on disk.

### 5.4 Product quantization

{{cite:jegou2011pq}}. Split $x \in \R^d$ into $m$ subvectors of dimension $d/m$;
quantize each with its own codebook of $2^b$ centroids learned by $k$-means:

$$ x \approx \big[\,c^{(1)}_{i_1},\, c^{(2)}_{i_2},\, \dots,\, c^{(m)}_{i_m}\,\big], \qquad \text{stored as } (i_1,\dots,i_m) $$ (eq:pq-code)

The effective codebook is the *product* — $2^{bm}$ distinct reconstructions from
$m \cdot 2^b$ stored centroids. With $b=8, m=16$ that is $2^{128}$ from 4,096
centroids, which is the trick.

Storage falls from $4d$ bytes to $mb/8$:

$$ \text{compression} = \frac{32\,d}{m\,b} $$ (eq:pq-compression)

For $d = 768, m = 96, b = 8$: 32×. Given {{ch:emb-vector-db}}'s memory equation,
where vectors dominate 20-to-1, this is the difference between fitting in RAM and
not.

### 5.5 Asymmetric distance computation

The subtle and important part. Two ways to compare a query to a code:

**Symmetric (SDC).** Quantize the query too, compare codes. Both sides are
approximated.

**Asymmetric (ADC).** Keep the query exact. Precompute a table of $q$'s
subvector against every centroid, then score any code by $m$ lookups and adds:

$$ \hat{s}_{\text{ADC}}(q, x) = \sum_{j=1}^{m} \big\langle q^{(j)},\, c^{(j)}_{i_j} \big\rangle, \qquad \text{table cost } m \cdot 2^b \cdot \tfrac{d}{m} $$ (eq:adc)

The table is built once per query and amortised over the whole corpus, so ADC is
**no more expensive than SDC at query time** and strictly more accurate, because
it introduces quantization error on one side instead of two. Roughly:

$$ \E\big[\epsilon^2_{\text{SDC}}\big] \approx 2\,\E\big[\epsilon^2_{\text{ADC}}\big] $$ (eq:adc-vs-sdc)

**ADC is free accuracy** and it is what every real implementation uses.
{{sec:9-practical-example}} measures the gap.

### 5.6 The anisotropic correction

{{cite:guo2020scann}}'s insight, and the deepest idea in the chapter.

$k$-means minimises reconstruction error $\lVert x - \hat{x}\rVert^2$, treating
all error directions alike. But the downstream operation is an inner product with
a query. Decompose the residual $r = x - \hat{x}$ relative to $x$'s own
direction:

$$ r = r_{\parallel} + r_{\perp}, \qquad r_{\parallel} = \frac{\langle r, x\rangle}{\lVert x\rVert^2}\,x $$ (eq:residual-decomposition)

For queries that are *near* $x$ — the ones whose ranking we care about — $q$ is
roughly aligned with $x$, so $\langle q, r_{\perp}\rangle \approx 0$ while
$\langle q, r_{\parallel}\rangle$ passes through undiminished. **The parallel
component corrupts the score and the orthogonal component largely does not.**

So the objective should weight them differently:

$$ \Loss_{\text{aniso}} = \eta \lVert r_{\parallel}\rVert^2 + \lVert r_{\perp}\rVert^2, \qquad \eta > 1 $$ (eq:anisotropic-loss)

> **IMPORTANT:** The general lesson is worth more than the algorithm.
> **Reconstruction error is the wrong objective whenever the downstream operation
> discards part of the error.** The same argument applies to quantizing model
> weights ({{part:14}}), to lossy compression before a classifier, and to any
> approximation whose consumer is a projection.

## 6. Mathematical Foundation

### 6.1 Why greedy graph search works at all

Greedy descent on an arbitrary graph has no guarantee. It works here because a
proximity graph over data with low intrinsic dimension ({{ch:emb-similarity}})
approximates a **navigable small world**: from any node, the neighbour set spans
a range of scales, so the distance to the target contracts by a constant factor
per hop:

$$ d(u_{t+1}, q) \leq \alpha \, d(u_t, q), \quad \alpha < 1 \;\Longrightarrow\; t = O(\log N) \text{ hops} $$ (eq:contraction)

{{eq:contraction}} fails when the neighbour sets are all short-range — no coarse
navigation — or when intrinsic dimension is high enough that
{{cite:beyer1999nn}}'s concentration makes "closer" uninformative. **Both failure
modes are properties of the data, not the index**, which is why an ANN index that
works on one embedding can be poor on another of identical dimension.

### 6.2 What $\text{ef}$ buys

Greedy search fails by terminating at a local minimum. Keeping $\text{ef}$
candidates means failing only if *all* $\text{ef}$ are local minima. Modelling
those as roughly independent with per-candidate failure $p$:

$$ \Prob[\text{miss}] \approx p^{\,\text{ef}} \quad\Longrightarrow\quad \text{recall} \approx 1 - p^{\,\text{ef}} $$ (eq:ef-recall-model)

Recall approaches 1 geometrically in $\text{ef}$ while cost grows roughly
linearly, which is the shape {{sec:9-practical-example}} measures — and it
explains the characteristic knee: early increases in $\text{ef}$ are very cheap
in recall terms and later ones are not.

The practical consequence: **tune $\text{ef}$ to your recall target and stop.**
Past the knee you are paying linearly for exponentially diminishing returns.

### 6.3 Composing the cascade

The standard IVF-PQ-plus-rerank configuration, with its cost:

$$ \underbrace{n_{\text{list}} d}_{\text{centroids}} + \underbrace{\tfrac{n_{\text{probe}}}{n_{\text{list}}} N m}_{\text{ADC scan}} + \underbrace{R \, d}_{\text{exact rerank}} $$ (eq:ivfpq-cost)

with $R$ the rerank depth. Note the middle term costs $m$ per vector rather than
$d$ — for $d=768, m=96$ that is an 8× saving on the dominant term — and the last
term is negligible for $R \sim 100$.

**The rerank stage is what makes the aggressive compression safe.** Without it,
PQ's ranking errors are the final answer; with it, PQ only has to get the true
answers into the top $R$. That is a far weaker requirement, and it is why
production systems compress harder than any PQ-only recall number would justify.

## 7. Internal Mechanics

```mermaid {#fig:ann-cascade caption="The standard large-scale configuration. Each stage narrows the candidate set and raises the precision of scoring; the exact vectors are read only for the final hundred. This is the same cascade as retrieve-then-rerank and model routing."}
flowchart LR
    Q["query"] --> C1["coarse: probe<br/>n_probe of n_list cells"]
    C1 -->|"~N x n_probe/n_list"| C2["ADC scan over<br/>PQ codes (eq:adc)"]
    C2 -->|"top R ~ 100"| C3["exact rescoring<br/>on full vectors"]
    C3 -->|"top k"| O["results"]
    S[("PQ codes<br/>in RAM")] -.-> C2
    F[("full vectors<br/>on SSD")] -.-> C3
```

### 7.1 Choosing $M$

$M$ sets the graph degree, and by {{ch:emb-vector-db}}'s memory equation costs
about 1% of index size per unit — cheap. Higher $M$ gives better recall at the
same $\text{ef}$, slower builds, and — by
{{eq:hnsw-filter-threshold}} — better filter robustness.

The reason not to set it very high is diminishing returns plus build time, not
memory. **Set $M$ from the most selective filter you must support**, then check
that unfiltered recall is adequate; the reverse order produces an $M$ that fails
under filtering.

### 7.2 The three parameters people confuse

- **$\text{ef}_{\text{construction}}$** is how hard the *builder* searches for
  each new node's neighbours. Higher means a better graph, a slower build, and no
  query cost at all.
- **$\text{ef}$** is how hard the *query* searches. Free to change.
- **$M$** is the degree. Rebuild to change.

A recall problem is almost always fixed by raising $\text{ef}$ first, because it
costs nothing to try.

### 7.3 Training the quantizer

PQ's codebooks are $k$-means centroids, and everything {{ch:ml-clustering}} says
about $k$-means applies — including the parts practitioners forget.

**The training sample must match the corpus distribution.** Centroids fitted to
the first hundred thousand documents of a corpus ordered by ingestion date encode
whatever was ingested first. A random sample of 100k–1M vectors is enough;
ordering is not.

**Initialisation matters more than iteration count.** $k$-means++ or
random-points initialisation reaches a usable solution in twenty iterations;
poor initialisation leaves dead centroids — codes that no vector ever uses,
wasting bits. Count the used codes after training; anything below about 95%
utilisation means the initialisation failed, not that the data is unusual.

**Each subspace is trained independently**, which is exactly the assumption that
OPQ ({{sec:15-advanced-concepts}}) relaxes. When adjacent dimensions are strongly
correlated — common in embeddings, where the model has no reason to decorrelate
its output coordinates — the independent split wastes capacity, and a rotation
before splitting recovers it for free at query time.

### 7.4 Reading a frontier

{{eq:recall-qps-frontier}} is a curve, and comparing two curves needs a
convention. The one the field has settled on:

- **Fix a recall target** from the application — usually the recall@$R$ that
  {{ch:emb-reranking}}'s reranker inherits — and compare throughput *there*.
- **Never compare at each index's best point**, which is how vendor benchmarks
  are constructed and why they all appear to win.
- **Plot log-throughput against recall**, since the interesting region is the
  high-recall end where throughput falls off a cliff and a linear axis hides it.
- **State the build parameters**, because an index tuned at build time for one
  recall region will lose in another, and a curve without its $M$ and
  $	ext{ef}_{	ext{construction}}$ is not reproducible.

### 7.5 Where LSH went

{{cite:indyk1998lsh}} is the historical foundation and it is honest to say it
lost. Hash functions that collide with probability related to similarity give
provable sublinear guarantees — the first such result, and the framing every
practical index still lives inside.

It is not competitive empirically. The guarantees are worst-case over adversarial
data, and real embeddings are far from worst-case: graph indexes exploit the
structure that LSH's data-independence deliberately ignores. **The theory
outlasted the algorithm**, which is a common and worth-noting pattern.

## 8. Implementation

```python {tier=A name=graph-ann-frontier}
"""A proximity-graph index from scratch, and its recall/cost frontier.

Build: insert points one at a time, connecting each to the M nearest already
inserted, found by greedy search on the partial graph (eq:greedy-graph-search).
This is Malkov's NSW construction -- the long-range links that make the graph
navigable come from insertion ORDER, not from explicit design.

Search: greedy best-first with a candidate list of size ef. We report recall
against exact search and the number of distance computations, which is the
honest cost unit -- it is what the frontier of eq:recall-qps-frontier is made of.
"""
import heapq
import numpy as np

rng = np.random.default_rng(7)

N, DIM, LATENT = 12_000, 32, 12
M, EF_CONSTRUCTION, K = 12, 40, 10
N_QUERY = 150

# Data with low intrinsic dimension -- the regime where eq:contraction holds.
proj = rng.normal(size=(LATENT, DIM)) / np.sqrt(LATENT)
X = rng.normal(size=(N, LATENT)) @ proj
X /= np.linalg.norm(X, axis=1, keepdims=True)
queries = rng.normal(size=(N_QUERY, LATENT)) @ proj
queries /= np.linalg.norm(queries, axis=1, keepdims=True)

truth = [set(row.tolist())
         for row in np.argsort(-(queries @ X.T), axis=1)[:, :K]]

neighbours = [[] for _ in range(N)]


def search(v, entry, ef, counter):
    """Greedy best-first with a bounded candidate list (eq:greedy-graph-search).

    `cand` is a min-heap on distance (nearest first, to expand next).
    `best`  is a max-heap on distance (farthest first, so it can be trimmed).
    """
    d0 = -float(X[entry] @ v)
    cand, best, seen = [(d0, entry)], [(-d0, entry)], {entry}
    while cand:
        d, u = heapq.heappop(cand)
        if len(best) >= ef and d > -best[0][0]:
            break                       # nothing left that can improve the list
        for w in neighbours[u]:
            if w in seen:
                continue
            seen.add(w)
            counter[0] += 1
            dw = -float(X[w] @ v)
            if len(best) < ef or dw < -best[0][0]:
                heapq.heappush(cand, (dw, w))
                heapq.heappush(best, (-dw, w))
                if len(best) > ef:
                    heapq.heappop(best)
    return sorted((-d, i) for d, i in best)


for i in range(1, N):
    found = search(X[i], int(rng.integers(0, i)), min(EF_CONSTRUCTION, i), [0])
    chosen = [j for _, j in found[:M]]
    neighbours[i] = chosen
    for j in chosen:
        neighbours[j].append(i)
        if len(neighbours[j]) > 2 * M:          # prune to keep degree bounded
            neighbours[j] = sorted(
                neighbours[j], key=lambda w: -float(X[j] @ X[w]))[:2 * M]

mean_degree = float(np.mean([len(s) for s in neighbours]))
print(f"graph built: {N} nodes, mean degree {mean_degree:.1f}\n")
print(f"{'ef':>6}{'recall@10':>12}{'distance comps':>17}{'vs brute force':>16}"
      f"{'speedup':>10}")
print("-" * 61)

for ef in [10, 20, 40, 80, 160, 320]:
    recalls, total = [], 0
    for qi in range(N_QUERY):
        counter = [0]
        found = search(queries[qi], int(rng.integers(0, N)), ef, counter)
        got = {i for _, i in found[:K]}
        recalls.append(len(got & truth[qi]) / K)
        total += counter[0]
    comps = total / N_QUERY
    print(f"{ef:>6}{np.mean(recalls):>12.4f}{comps:>17.1f}"
          f"{comps / N:>15.2%}{N / comps:>10.1f}x")

print("""
This table IS eq:recall-qps-frontier, and it is the only honest way to report an
ANN index. A single recall number means nothing, because any index reaches any
recall by working harder; the question is always what your required recall costs.

Read the knee. Going from ef=10 to ef=40 buys a large jump in recall for roughly
double the work. Going from ef=160 to ef=320 doubles the work again and buys
nothing, because recall has already saturated. That shape is eq:ef-recall-model:
recall approaches 1 geometrically in ef while cost grows linearly, so there is
always a knee and you should stop at it.

The speedup column is the reason approximate search exists. Near-exact answers
while touching a small percentage of the corpus -- and note that this graph was
built by nothing more than inserting points in a random order and linking each to
its nearest predecessors. The long-range links that make it navigable are a free
consequence of the early insertions having few candidates to choose from.""")
```

```python {tier=A name=product-quantization}
"""Product quantization from scratch: compression against recall.

Split each vector into m subvectors and quantize each with its own 256-centroid
codebook (eq:pq-code). Storage falls from 4d bytes to m bytes.

Three scorers compared:
  ADC  -- query exact, table lookup per subspace (eq:adc)
  SDC  -- query quantized too; both sides approximated
  ADC then exact rerank of the top 100, which is what real systems do
"""
import numpy as np
from scipy.cluster.vq import kmeans2

rng = np.random.default_rng(11)

N, DIM, LATENT, K = 20_000, 64, 32, 10
N_QUERY, N_BITS, RERANK = 200, 256, 100

proj = rng.normal(size=(LATENT, DIM)) / np.sqrt(LATENT)
X = rng.normal(size=(N, LATENT)) @ proj
X /= np.linalg.norm(X, axis=1, keepdims=True)
queries = rng.normal(size=(N_QUERY, LATENT)) @ proj
queries /= np.linalg.norm(queries, axis=1, keepdims=True)
truth = [set(r.tolist()) for r in np.argsort(-(queries @ X.T), axis=1)[:, :K]]


def recall(scores, rerank=0):
    hits = []
    for i in range(N_QUERY):
        if rerank:
            cand = np.argpartition(-scores[i], rerank)[:rerank]
            top = cand[np.argsort(-(queries[i] @ X[cand].T))[:K]]
        else:
            top = np.argpartition(-scores[i], K)[:K]
        hits.append(len(set(top.tolist()) & truth[i]) / K)
    return float(np.mean(hits))


print(f"{'m':>4}{'bytes/vec':>11}{'compression':>13}{'ADC':>9}{'SDC':>9}"
      f"{'ADC+rerank':>13}")
print("-" * 59)

for m in [4, 8, 16, 32]:
    sub = DIM // m
    centroids = np.zeros((m, N_BITS, sub))
    codes = np.zeros((N, m), dtype=np.int32)
    for j in range(m):
        block = X[:, j * sub:(j + 1) * sub]
        c, labels = kmeans2(block, N_BITS, minit='points', seed=1, iter=25)
        centroids[j], codes[:, j] = c, labels

    # ADC: build a per-query table against each subspace's centroids, then the
    # score for any code is m lookups. The table cost is paid once per query.
    adc = np.zeros((N_QUERY, N))
    for j in range(m):
        table = queries[:, j * sub:(j + 1) * sub] @ centroids[j].T
        adc += table[:, codes[:, j]]

    # SDC: quantize the query as well, then compare reconstructions.
    q_codes = np.zeros((N_QUERY, m), dtype=np.int32)
    for j in range(m):
        block = queries[:, None, j * sub:(j + 1) * sub]
        q_codes[:, j] = ((block - centroids[j][None]) ** 2).sum(-1).argmin(1)
    q_recon = np.concatenate([centroids[j][q_codes[:, j]] for j in range(m)], 1)
    recon = np.concatenate([centroids[j][codes[:, j]] for j in range(m)], 1)
    sdc = q_recon @ recon.T

    print(f"{m:>4}{m:>11d}{DIM * 4 // m:>12d}x{recall(adc):>9.4f}"
          f"{recall(sdc):>9.4f}{recall(adc, RERANK):>13.4f}")

print(f"""
Compare ADC against SDC first. ADC wins at every compression level, and it costs
NOTHING extra at query time -- the lookup table is built once per query and
amortised over the whole corpus. Keeping the query exact introduces quantization
error on one side instead of two (eq:adc-vs-sdc). This is free accuracy, and it
is why every real implementation is asymmetric.

Now read the ADC column on its own and it looks like bad news: at 32x compression
the quantizer returns the true top-10 well under half the time. A system shipping
that number would be broken.

Then read the rerank column. The SAME codes, rescoring only the top {RERANK}
candidates with exact vectors, recover most of what was lost -- and the harder
the compression, the larger the recovery in absolute terms. That is eq:ivfpq-cost's
third term, and it costs {RERANK} full-precision dot products per query --
negligible beside the scan.

The reframing is the point. PQ is not an answer, it is a CANDIDATE GENERATOR, and
its job is not to rank correctly but to get the true answers somewhere into the
top {RERANK}. That is a far weaker requirement than ranking, which is why
production systems compress far harder than any PQ-only recall number would
justify -- and it is the same cheap-then-exact cascade as retrieve-then-rerank
and model routing, arriving for the fourth time.""")
```

## 9. Practical Example

Both listings are the practical example, and each produces a frontier rather
than a number.

**The graph.** Built by nothing more than inserting points in random order and
linking each to its nearest predecessors, the index reaches 99.4% recall while
examining 3.8% of the corpus, and 90.9% while examining 1.8%. The table's shape
is
{{eq:ef-recall-model}}: a steep early climb, then a knee, then linear cost for
negligible gain. **Tune $\text{ef}$ to the recall you need and stop** — the
region past the knee is pure waste, and it is where hastily tuned indexes usually
sit.

**The quantizer.** ADC beats SDC at every compression level for no extra query
cost — 0.390 against 0.259 at 32×, 0.620 against 0.516 at 16× — exactly as
{{eq:adc-vs-sdc}} predicts.

The rerank column is the result to internalise. At 32× compression the raw ADC
recall is 39.1% — a number that looks like a broken system — and rescoring the
top 100 with exact vectors brings it to 92.0%. At 16× the same move takes 62.0%
to 99.7%. Nothing about the codes changed. **PQ's job is not to rank; it is to get the true answers into the top
$R$**, which is a much weaker requirement, and it is why the aggressive
compression that {{ch:emb-vector-db}}'s memory arithmetic demands is safe in
practice.

> **NOTE:** The two listings compose. A production index at scale is the graph
> or IVF for candidate generation, PQ for cheap scoring, and exact rerank for the
> final order — {{fig:ann-cascade}}, and {{eq:ivfpq-cost}} for the cost. Neither
> stage is adequate alone and neither is trying to be.

## 10. Production Considerations

**Report the frontier, always.** A single recall number is not a result
({{cite:aumuller2020annbench}}). Sweep $\text{ef}$ or $n_{\text{probe}}$ and plot
against latency.

**Measure recall against exact search on a fixed probe set, continuously.** It is
the only signal that catches drift, tombstone decay ({{ch:emb-vector-db}}), or a
parameter regression. Nothing else moves.

**Set $\text{ef}$ per query class, not globally.** A high-value query can afford
more; an autocomplete cannot. It is a free runtime knob and treating it as a
global constant leaves value on the table.

**Always rerank.** {{eq:ivfpq-cost}}'s third term is cheap and it changes the
compression you can afford. Keeping full vectors on SSD purely for reranking is
the standard disk-based design and it is usually the right one.

**Train the quantizer on a sample of the real corpus**, not on the first
$n$ documents and not on a public dataset. PQ codebooks are $k$-means centroids
and inherit every distributional assumption ({{ch:ml-clustering}}).

**Re-train the codebooks when the corpus shifts.** Centroids fitted to last
year's distribution quantize this year's documents badly, and — as with
everything else in this part — the symptom is a quiet recall decline.

**Budget the build.** Graph construction is $O(N \log N)$ distance computations
with a large constant. At a hundred million vectors this is hours, and it is your
recovery time objective.

**Warm the index before serving it.** A freshly loaded index has nothing in page
cache, and graph traversal is close to a random-access pattern over the whole
vector array — so the first thousand queries after a deploy can be an order of
magnitude slower than steady state. Replay a sample of real queries before
putting a replica into rotation, or the deploy shows as a latency incident that
resolves itself and nobody can explain.

## 11. Common Mistakes

**Reporting recall at one operating point.** Meaningless.

**Confusing index recall with retrieval quality.** {{eq:ann-recall}} measures
fidelity to exact search over the same embedding, nothing more.

**Tuning $M$ when $\text{ef}$ was the problem.** $\text{ef}$ is free; $M$ needs a
rebuild. Try the free one first.

**Using PQ without reranking.** The listing's ADC column is what that looks like.

**Using SDC.** Strictly worse, no cheaper.

**Training the quantizer on the wrong distribution.**

**Benchmarking on random Gaussian vectors.** {{ch:emb-similarity}} showed these
are the *hardest* possible case — full intrinsic dimension, maximum
concentration. Results transfer pessimistically at best, and the ranking between
indexes can invert.

**Assuming published parameters transfer.** $M$ and $\text{ef}$ depend on the
data's intrinsic dimension, which differs by embedding model.

## 12. Failure Modes

**Recall collapse on out-of-distribution queries.** The graph was built from the
corpus; a query far from the corpus manifold enters at a poor point and
{{eq:contraction}} does not hold. Symptom: fine on test queries, poor on real
ones.

**Codebook staleness.** Corpus drifts, centroids do not, quantization error
grows, recall declines over months.

**Local-minimum trapping at low $\text{ef}$.** Recall that is *stable and
mediocre*, easily mistaken for an embedding-quality problem.

**Degree starvation after deletion.** Tombstones plus pruning reduce effective
degree, {{eq:contraction}}'s constant worsens, recall declines
({{ch:emb-vector-db}}).

**Build/query parameter mismatch.** An index built with low
$\text{ef}_{\text{construction}}$ has a ceiling no query-time $\text{ef}$ can
raise. Diagnostic: if recall plateaus below 1.0 as $\text{ef} \to$ large, the
graph is the problem.

**Memory blowup from $M$.** Rare, since vectors dominate — but real for
low-dimensional or heavily quantized vectors, where the graph *can* become the
larger term.

## 13. Alternatives

**Brute force.** Exact, trivially filterable, and fast enough below roughly a
million vectors. {{ch:emb-vector-db}}'s crossover shows it also wins outright
under selective filters. Under-used.

**IVF alone.** Simple, deletable, disk-friendly, composes with metadata
partitioning; worse recall/cost than graphs in memory.

**DiskANN-style.** Graph on SSD with quantized vectors in RAM for traversal.
Changes the cost model from RAM to IOPS, often an order of magnitude cheaper.

**ScaNN** ({{cite:guo2020scann}}). Anisotropic quantization plus partitioning;
strongest on MIPS specifically.

**LSH** ({{cite:indyk1998lsh}}). Provable guarantees, empirically displaced.

**GPU brute force** ({{cite:johnson2019faiss}}). At tens of millions of vectors a
GPU scan is exact and fast, and removes every failure mode in this chapter. If
the corpus fits, it is the simplest correct answer.

## 14. Evaluation

**The frontier**: recall@k against queries per second, swept over $\text{ef}$ or
$n_{\text{probe}}$ ({{eq:recall-qps-frontier}}).

**On your embeddings and your query distribution.** Intrinsic dimension governs
everything ({{ch:emb-similarity}}), so published numbers on SIFT or GIST predict
little about a 768-dimensional text embedding.

**Recall at the $k$ your reranker uses.** If a cross-encoder reranks 100, measure
recall@100 — that is the ceiling it inherits ({{ch:emb-reranking}}).

**Tail latency, not mean.** Graph search is variable: unlucky entry points take
many more hops. The p99 can be several times the mean.

**Build time and memory**, since they are the operational constraints.

**Recall under your filters**, which is a different and usually much worse number
({{ch:emb-vector-db}}).

## 15. Advanced Concepts

**Why the hierarchy helps less than expected.** {{cite:malkov2020hnsw}}'s layers
give $O(\log N)$ hops, but a flat NSW with good long links is often within a few
percent. The hierarchy's real contribution is *robustness* — it makes the entry
point irrelevant, which matters most when the query distribution differs from the
corpus.

**OPQ: rotate before quantizing.** PQ assumes the subspaces are independent; a
learned rotation that decorrelates them before splitting reduces error at no
query cost, and is nearly free to add.

**Residual quantization.** Quantize, then quantize the residual, repeatedly. More
accurate than PQ at equal bit budget and slower to decode — the usual
accuracy/complexity trade.

**Anisotropy generalises.** {{eq:anisotropic-loss}} says: weight error by how
much the downstream operation is sensitive to it. That principle applies to
weight quantization, to lossy compression before a classifier, and to any
approximation whose consumer is a projection. It is the most portable idea in the
chapter.

**Graph indexes are learned indexes without the learning.** The graph encodes the
data distribution structurally — a lookup structure fitted to the data rather
than defined a priori, which is exactly the "learned index" framing, arrived at
from a different direction and predating it.

**The entry-point problem.** {{eq:contraction}} assumes you start somewhere
reasonable. HNSW's top layer solves it structurally; flat graphs use multiple
random entries. For out-of-distribution queries neither works well, which is
{{sec:12-failure-modes}}'s first entry.

## 16. Connection to Previous Chapters

{{ch:emb-similarity}}'s intrinsic-dimension result is what makes
{{eq:contraction}} hold and therefore what makes graph search work at all; its
metric analysis is why MIPS needs {{eq:anisotropic-loss}}.
{{ch:ml-clustering}}'s $k$-means is the entirety of IVF and PQ.
{{ch:emb-vector-db}}'s memory equation is why compression is mandatory, and its
percolation result is why $M$ should be set from filters.
{{ch:emb-models}}'s nested embeddings attack the same memory term from the
representation side. And {{ch:llm-routing}}'s cascade equation is
{{eq:ivfpq-cost}} with different letters — the fourth appearance of one
architecture.

## 17. Exercises

1. Derive $n_{\text{list}} \approx \sqrt{N}$ from {{eq:ivf-cost}}.
2. Compute {{eq:pq-compression}} for $d=1536, m=192, b=8$. What is the index
   memory for $10^8$ vectors, with and without PQ?
3. In `graph-ann-frontier`, sweep `M` over $\{6, 12, 24\}$ at fixed `ef`.
   Which changes more, recall or distance computations?
4. Set `EF_CONSTRUCTION` to 5 and rebuild. Show that no query-time `ef` recovers
   full recall, and explain why using {{eq:contraction}}.
5. Replace the low-intrinsic-dimension data with i.i.d. Gaussians and re-measure.
   Relate the degradation to {{ch:emb-similarity}}'s concentration result.
6. In `product-quantization`, sweep `RERANK` over $\{10, 50, 100, 500\}$ at
   $m=8$. Where does it saturate, and what does that say about $R$?
7. Implement OPQ: fit a rotation by SVD of the residuals, apply before
   quantizing, and measure the improvement.
8. Implement {{eq:anisotropic-loss}} as a re-weighted $k$-means and measure
   whether it beats plain PQ on inner-product recall.

## 18. Interview Questions

1. What does "approximate" give up, and how do you control how much?
2. HNSW's $M$ against $\text{ef}$ — which do you tune first and why?
3. Why does product quantization compress so much more than its codebook size
   suggests?
4. Why is asymmetric distance computation better than symmetric, and what does it
   cost?
5. Your ANN recall is 100%. Is retrieval good?
6. Recall plateaus at 0.85 no matter how high $\text{ef}$ goes. Diagnose.
7. When would you not use an ANN index?
8. Why does ScaNN weight quantization error anisotropically?
9. How do you report an ANN benchmark honestly?
10. You have 100M 1536-dimensional vectors and 64 GB of RAM. Design it.

## 19. Research Questions

1. Is there an index with a *distribution-free* recall guarantee that is also
   competitive empirically, or is the theory/practice gap since
   {{cite:indyk1998lsh}} fundamental?
2. {{eq:contraction}}'s $\alpha$ is not measurable a priori. Can it be estimated
   from a corpus cheaply enough to predict an index's frontier before building?
3. Deletion in graph indexes remains unsolved ({{ch:emb-vector-db}}). Is
   logarithmic search with cheap deletion achievable?
4. How far does {{eq:anisotropic-loss}}'s principle generalise? Is there a general
   theory of "quantize for the downstream operator" covering weight quantization
   and ANN together?
5. Entry-point selection for out-of-distribution queries is handled heuristically
   everywhere. Is there a principled method?

## 20. Chapter Summary

Approximate search trades a guarantee for a factor of fifty or more in work, and
the trade is favourable because learned embeddings have low intrinsic dimension —
which is what makes {{eq:contraction}}'s greedy descent converge.

**Three families give up three different things.** Graphs give up completeness
and are controlled by $\text{ef}$, a free runtime knob whose recall follows
{{eq:ef-recall-model}} — geometric gains against linear cost, hence a knee, and
tuning past it is waste. Partitions give up coverage, controlled by
$n_{\text{probe}}$, with $n_{\text{list}} \approx \sqrt{N}$ from
{{eq:ivf-cost}}. Quantization gives up precision, with {{eq:pq-code}}'s product
codebook giving $2^{bm}$ reconstructions from $m2^b$ stored centroids.

**Two results are worth carrying.** Asymmetric distance computation
({{eq:adc}}) is strictly more accurate than symmetric at identical query cost,
because the table is amortised over the corpus — free accuracy, and universal in
practice. And PQ is a candidate generator rather than an answer: measured, 32×
compression gives poor recall alone and recovers most of it when the top 100 are
rescored exactly. That weaker requirement is what licenses the aggressive
compression that memory demands.

**Everything composes into one cascade** ({{fig:ann-cascade}},
{{eq:ivfpq-cost}}) — the fourth appearance of cheap-stage-then-precise-stage in
this book. And {{cite:guo2020scann}}'s anisotropic argument is the most portable
idea here: **reconstruction error is the wrong objective whenever the downstream
operation discards part of the error.**

Finally, report the frontier ({{eq:recall-qps-frontier}}). A single recall number
is not a result, because any index reaches any recall by working harder.

## 21. Further Reading

{{cite:malkov2020hnsw}} for HNSW — Sections 3 and 4 for the construction
heuristic and the parameter analysis.
{{cite:malkov2014nsw}} for the original insight that insertion order creates the
long links.
{{cite:jegou2011pq}} for product quantization; Section 3.2 is the ADC/SDC
analysis behind {{eq:adc-vs-sdc}}.
{{cite:guo2020scann}} for the anisotropic loss — Section 3 has the residual
decomposition.
{{cite:johnson2019faiss}} for how these compose into a system, and for the GPU
brute-force option that makes much of this unnecessary at moderate scale.
{{cite:aumuller2020annbench}} for how to benchmark honestly.

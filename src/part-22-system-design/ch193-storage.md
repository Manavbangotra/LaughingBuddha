---
id: sd-storage
number: 193
part: XXII
tier: full
status: draft
requires: [three-properties-break-the-stack, semantic-failure-has-no-instrument,
           cache-threshold-is-an-error-cost-decision, variance-not-mean-drives-wait]
provides: [access-shape-decides-the-store, cache-is-cheap-access-not-cheap-storage,
           derived-copies-multiply-contradiction, depth-beats-speed-for-staleness]
citations: [malkov2020hnsw, johnson2019faiss, guo2020scann, kwon2023pagedattention]
---

## 1. Learning Objectives

By the end of this chapter you will be able to place a piece of state in the right
store from its access shape rather than from what it contains, and price the
difference; explain why a key-value cache is simultaneously the most expensive and
the cheapest storage in a typical stack, and what decides which; enumerate the
derived copies of a fact that a production AI system holds and compute the window in
which they can contradict each other; show why reducing pipeline *depth* beats
reducing pipeline *latency* for staleness, and why synchronisation barely helps at
all; and recognise every materialised derived copy as a cache subject to
{{ch:sd-routing-caching}}'s error-cost arithmetic.

## 2. Why This Matters

Storage decisions in AI systems are usually made twice and thought about once.
The first decision is real: embeddings need similarity search, so a vector store gets
added. Every subsequent decision is made by default — the state goes wherever the
existing database is, because it is already there and it already works.

That default is correct more often than it deserves to be, which is why it survives.
{{sec:9-practical-example}} prices seven pieces of state a production AI system
actually holds and finds the default correct on **five of seven**. The other two cost
**2.9×** the whole storage bill ({{eq:access-shape-decides-the-store}}), and the
overpayment is concentrated so tightly that finding it is a morning's work rather
than a re-architecture.

The second half is a failure mode that no store can see. An AI system holds the same
fact in several derived forms — source document, extracted text, chunk embeddings,
summary row, cached answer — each rebuilt on its own schedule. When the fact changes
they converge at different times, and in the gap the system can retrieve one version
and quote another. With five copies that gap covers **27.36%** of queries
({{eq:derived-copies-multiply-contradiction}}), and every store involved reports
perfect health throughout.

## 3. Prerequisites

You need {{ch:sd-architecture}}'s three properties
({{eq:three-properties-break-the-stack}}) for the cost arithmetic, and especially
{{eq:semantic-failure-has-no-instrument}} — the second half of this chapter is that
result relocated to the space *between* stores.

{{eq:cache-threshold-is-an-error-cost-decision}} from {{ch:sd-routing-caching}} is
load-bearing in {{sec:15-advanced-concepts}}, where every materialised derived copy
turns out to be a cache.

{{eq:variance-not-mean-drives-wait}} from {{ch:sd-async}} appears as a contrast
rather than an application: the storage layer behaves differently, and the difference
is instructive.

{{ch:emb-vector-db}} and {{ch:emb-ann}} supply the vector-store background assumed in
{{sec:7-internal-mechanics}}.

## 4. Intuitive Explanation

Here is the question that decides where a piece of state belongs, and it is not the
question people ask.

The question people ask is *what is this?* Embeddings, so vector store. Files, so
object storage. Rows with foreign keys, so the database. That taxonomy is about the
data, it is easy to apply, and it is almost irrelevant to the bill.

The question that decides the bill is *how is this touched?* Specifically: how large
is it, and how often is it read? Because storage products price those two things
separately, and they price them in opposite directions.

Object storage is astonishingly cheap per gigabyte and expensive per operation. A
key-value cache is the reverse — it costs many times more per gigabyte than anything
else in the stack, and its reads cost almost nothing. That is not a quirk. It is the
product: **a cache is not expensive storage, it is cheap access sold with expensive
storage attached.**

Which tells you exactly when each wins. Something huge that is rarely read — the
original documents, the raw archive — belongs in object storage, because you are
paying for the gigabytes and barely touching them. Something small that is read
constantly — a prompt cache, a hot lookup table — belongs in a cache, because you are
paying for the reads and there are hardly any gigabytes. Neither decision has
anything to do with what the data contains.

The interesting cases are in the middle, and this is the part worth remembering: the
middle is where capability requirements actually decide. If size and access rate do
not clearly favour one store, the cost difference is small enough that "does it need
transactions?" or "does it need similarity search?" becomes the deciding question.
At the extremes, those requirements are worth paying for only if they are genuinely
required, because the cost gap is an order of magnitude.

The second half of the chapter concerns something subtler. Count how many times your
system stores the same fact. Not the same *row* — the same fact. The document is in
object storage. Its extracted text is somewhere else. Its chunks are embedded in the
vector index. A summary of it is a database row. An answer quoting it is in the
prompt cache. That is five representations of one fact, and each is rebuilt from the
one before it on its own schedule.

Now the fact changes. The extracted text catches up in a few minutes. The embeddings
take longer. The cached answer, which nobody thought about, might take two hours.
During that window the system can retrieve the new chunk and serve the old cached
summary, and produce an answer that contradicts itself.

The instinct is to make everything faster. It works, but it is expensive, and it is
not the best available move. The window is measured from the *source*, which is never
stale, to the *slowest copy* — so it is set almost entirely by the deepest
materialisation. Deleting one copy and computing it at query time instead closes more
of the window than halving every latency in the pipeline.

## 5. Formal Explanation

Let a store $j$ have price components $(\gamma_j, \rho_j, \omega_j)$ for storage per
gigabyte-month, reads per thousand, and writes per thousand, plus a latency floor
$\ell_j$ and capability flags. A piece of state $i$ has size $g_i$, monthly reads
$r_i$, monthly writes $w_i$, a latency budget $b_i$, and capability requirements.

The monthly cost of placing $i$ in $j$ is

$$ K_{ij} \;=\; g_i\gamma_j \;+\; r_i\rho_j \;+\; w_i\omega_j $$ (eq:access-shape-decides-the-store)

defined only where $\ell_j \le b_i$ and $j$ satisfies $i$'s capability requirements.
The placement problem is to choose $\sigma(i) = \arg\min_j K_{ij}$ over feasible $j$,
and the total is $\sum_i K_{i\sigma(i)}$.

The structure worth extracting is which term dominates. Comparing two stores $j$ and
$j'$, the sign of $K_{ij} - K_{ij'}$ is determined by

$$ g_i(\gamma_j - \gamma_{j'}) \;+\; r_i(\rho_j - \rho_{j'}) \;+\; w_i(\omega_j - \omega_{j'}) $$

so the crossover is governed by the **ratios** $r_i/g_i$ and $w_i/g_i$ — reads and
writes per gigabyte — rather than by any absolute quantity. This is why the correct
placement rule is expressible without knowing the scale of the system:

$$ \frac{r_i}{g_i} \;>\; \frac{\gamma_{\text{cache}} - \gamma_{\text{other}}}{\rho_{\text{other}} - \rho_{\text{cache}}} \;\;\Longrightarrow\;\; \text{cache wins} $$ (eq:cache-is-cheap-access-not-cheap-storage)

**A cache is correct exactly when reads per gigabyte exceed the ratio of its storage
premium to its access discount**, and for nothing else. The same inequality run the
other way identifies object storage.

For consistency, let a fact have $n$ derived representations, representation $k$
catching up a random time $L_k \ge 0$ after the source changes, with $L_1 = 0$ for
the source itself. The system can contradict itself whenever some copy has caught up
and another has not, which is the interval $[\min_k L_k, \max_k L_k]$. Its expected
length is

$$ W \;=\; \mathbb{E}\!\left[\max_k L_k\right] - \mathbb{E}\!\left[\min_k L_k\right] \;=\; \int_0^\infty\!\Bigl(1 - \textstyle\prod_k F_k(t)\Bigr)dt \;-\; \int_0^\infty\!\textstyle\prod_k\bigl(1 - F_k(t)\bigr)dt $$ (eq:derived-copies-multiply-contradiction)

and with changes arriving at rate $\lambda_c$, the share of queries landing inside a
window is $\min(1, W\lambda_c)$.

## 6. Mathematical Foundation

{{eq:derived-copies-multiply-contradiction}} has a consequence that determines what
to do about it, and it differs sharply from the queueing result in {{ch:sd-async}}.

Because the source is never stale, $L_1 = 0$ with certainty, so $\min_k L_k = 0$ and
the second integral vanishes. The window collapses to

$$ W \;=\; \mathbb{E}\!\left[\max_k L_k\right] $$

which is dominated by whichever representation has the largest lag. Three
consequences follow.

**Speed helps proportionally.** Scaling every $L_k$ by $\alpha$ scales $W$ by
$\alpha$. Halving all lags halves the window — real, linear, and expensive.

**Synchronisation barely helps.** Setting every derived $L_k$ to a common constant
$L^\star$ gives $W = L^\star$, which is no better than the deepest copy already was.
There is no schedule that closes a gap whose other end is fixed at zero.

**Depth helps superlinearly.** Removing the deepest representation replaces
$\mathbb{E}[\max_k L_k]$ with $\mathbb{E}[\max_{k \ne n} L_k]$, and for
right-skewed lags with increasing depth this is a large drop, because the maximum of
a set is dominated by its most extreme member. Formally, if $L_n$ stochastically
dominates the rest,

$$ \frac{W_{n-1}}{W_n} \;\approx\; \frac{\mathbb{E}[\max_{k<n} L_k]}{\mathbb{E}[L_n]} \;\ll\; 1 $$ (eq:depth-beats-speed-for-staleness)

{{sec:9-practical-example}} measures this: removing one copy takes contradiction from
**27.36%** to **9.64%**, against **13.81%** for halving every lag in the pipeline.

Contrast this with {{eq:variance-not-mean-drives-wait}}, where variance dominated and
the mean barely mattered. The difference is structural: a queue compares every job
against every other, so spread is what costs; a staleness window is measured against
a fixed zero, so the extreme value is what costs. **Both chapters say "not the mean,"
and they mean different things by it.**

## 7. Internal Mechanics

**Why vector stores price the way they do.** An ANN index keeps its graph or
quantised codes resident to serve queries at low latency
({{cite:malkov2020hnsw}}, {{cite:guo2020scann}}), so the storage price reflects
memory rather than disk. That is why the vector store sits between the database and
the cache in per-gigabyte cost, and why "just put everything in the vector store" is
expensive for anything that does not need similarity search.

**Index rebuild is the deepest lag in most pipelines.** Incremental insertion into an
HNSW graph degrades recall over time as the graph's navigability drifts from what a
full build would produce, so periodic rebuilds are required.
{{cite:johnson2019faiss}}'s clustering-based indexes have the same property with a
different mechanism — centroids drift as the distribution moves. The rebuild interval
is therefore a recall-versus-freshness decision, and it lands directly in
{{eq:derived-copies-multiply-contradiction}} as one of the larger $L_k$.

**The prompt cache is a storage decision disguised as an inference optimisation.**
{{cite:kwon2023pagedattention}}'s KV cache management makes prefix reuse cheap enough
to be worth doing, which creates a new, hot, small piece of state with an access shape
unlike anything else in the system — and, by
{{eq:cache-is-cheap-access-not-cheap-storage}}, one whose placement is unambiguous
once anyone computes the ratio.

**Latency floors are capability constraints in disguise.** The reason object
storage cannot serve the prompt cache is not price, it is the 95-millisecond floor
against a 5-millisecond budget. Those floors come from the access path — a network
round trip to a distributed object store versus an in-memory lookup — and they are not
negotiable by spending more. This matters because it makes the placement problem
genuinely constrained rather than merely economic: several of the cells in the cost
table are empty, and for two of the seven items only one store is feasible at all.
When a latency budget and a size profile point at different stores, the budget wins,
and the correct response is to question the budget rather than the placement.

**Write amplification through the derivation chain.** One source change triggers a
write to every derived representation, so the *write* column in
{{eq:access-shape-decides-the-store}} scales with derivation depth. This couples the
two halves of the chapter: reducing depth lowers both the contradiction window and
the write bill, and the write saving is usually the one that gets a project approved.

**Why the audit log belongs where it does.** A tool-call audit log is write-heavy,
read-almost-never, and needs transactional integrity because it is evidence. That
combination has no cheap answer: object storage is cheapest per gigabyte but its
write pricing is punitive and it offers no transactions. The database wins by
default, and this is the case where "it's already there" is genuinely correct
reasoning rather than an absence of reasoning.

## 8. Implementation

The first listing prices seven pieces of state against four stores using access shape
alone, and compares the result against the default of keeping everything in the
primary database.

```python {tier=A name=by1}
"""Where a piece of state belongs, decided by access shape rather than by type.

Storage choices in an AI system are usually made by naming the thing -- embeddings go
in a vector store, rows go in a database, files go in object storage -- or by not
making them at all, and leaving everything in whatever database already exists. Both
taxonomies are about the DATA. Cost is driven by the ACCESS: how often it is read,
how often written, and how much of it there is.

This listing prices seven pieces of state a production AI system actually holds
against four stores, using access shape alone
(eq:access-shape-decides-the-store), and compares that against the default of
keeping everything in the primary database.
"""
# Stores: (name, cost per GB-month, cost per 1k reads, cost per 1k writes,
#          read latency ms, supports similarity search, supports transactions)
STORES = [
    ("relational db",  0.230, 0.180, 0.900,   3.0, False, True),
    ("vector store",   0.850, 0.400, 2.200,  12.0, True,  False),
    ("key-value cache", 3.400, 0.012, 0.030,  0.4, False, False),
    ("object storage", 0.021, 0.420, 5.400, 95.0, False, False),
]

# State: (name, size GB, reads per month in thousands, writes per month in
#         thousands, needs similarity, needs transactions, latency budget ms)
STATE = [
    ("chat transcripts",      940.0,   380.0,   620.0, False, False, 250.0),
    ("document embeddings",   210.0,  4200.0,    18.0, True,  False,  60.0),
    ("document originals",   6400.0,    45.0,    12.0, False, False, 900.0),
    ("user profiles",           4.0,  8800.0,   140.0, False, True,   25.0),
    ("prompt cache",           38.0, 26000.0,  9100.0, False, False,   5.0),
    ("tool call audit log",   150.0,     6.0,  1900.0, False, True,  900.0),
    ("eval label sets",         2.0,    30.0,     4.0, False, True,  900.0),
]


def cost(state, store):
    """Monthly cost, or None if the store cannot serve this state at all."""
    _, gb, r, w, sim, txn, lat = state
    _, per_gb, per_r, per_w, ms, has_sim, has_txn = store
    if sim and not has_sim:
        return None
    if txn and not has_txn:
        return None
    if ms > lat:
        return None
    return gb * per_gb + r * per_r + w * per_w


def default_store(state):
    """What most systems actually do: everything goes in the primary database,
    because it is already there and it already works. A vector store is added
    when embeddings arrive, because the database cannot do similarity search.

    This is not a straw man. It is the path of least resistance, and it is
    correct for several of these rows.
    """
    name, gb, r, w, sim, txn, lat = state
    if sim:
        return "vector store"
    return "relational db"


print("Four stores, priced per GB-month and per thousand operations.")
print()
print(f"{'store':>17}{'$/GB-mo':>10}{'$/1k rd':>10}{'$/1k wr':>10}"
      f"{'latency':>10}{'similarity':>12}{'txns':>7}")
print("-" * 76)
for s in STORES:
    print(f"{s[0]:>17}{s[1]:>10.3f}{s[2]:>10.3f}{s[3]:>10.3f}{s[4]:>9.1f}ms"
          f"{('yes' if s[5] else 'no'):>12}{('yes' if s[6] else 'no'):>7}")

print()
print()
print("Seven pieces of state a production AI system holds, by access shape.")
print()
print(f"{'state':>22}{'size GB':>10}{'k reads':>10}{'k writes':>10}"
      f"{'read:write':>12}{'budget':>9}")
print("-" * 73)
for st in STATE:
    rw = st[2] / st[3] if st[3] else float("inf")
    print(f"{st[0]:>22}{st[1]:>10.1f}{st[2]:>10.1f}{st[3]:>10.1f}"
          f"{rw:>12.2f}{st[6]:>8.0f}ms")

print()
print()
print("Monthly cost in each store. A dash means the store cannot serve it --")
print("wrong access pattern, no transactions, or too slow for the budget.")
print()
print(f"{'state':>22}" + "".join(f"{s[0][:14]:>16}" for s in STORES))
print("-" * 86)
costs = {}
for st in STATE:
    row = {}
    cells = ""
    for store in STORES:
        c = cost(st, store)
        row[store[0]] = c
        cells += f"{('--' if c is None else '%.0f' % c):>16}"
    costs[st[0]] = row
    print(f"{st[0]:>22}{cells}")

print()
print()
print("Cheapest feasible store for each, against the default of keeping it in")
print("the primary database.")
print()
print(f"{'state':>22}{'by access shape':>18}{'by default':>18}"
      f"{'cost of default':>17}")
print("-" * 76)
total_best = 0.0
total_naive = 0.0
picks = {}
for st in STATE:
    feasible = {k: v for k, v in costs[st[0]].items() if v is not None}
    best = min(feasible, key=lambda k: feasible[k])
    nv = default_store(st)
    nvc = costs[st[0]].get(nv)
    if nvc is None:
        # The default store cannot serve this state at all; fall back to the
        # cheapest feasible one and mark it.
        nvc = feasible[best]
        nv = nv + " (X)"
    picks[st[0]] = (best, feasible[best], nv, nvc)
    total_best += feasible[best]
    total_naive += nvc
    print(f"{st[0]:>22}{best:>18}{nv:>18}{nvc - feasible[best]:>17.0f}")

print("-" * 76)
print(f"{'TOTAL':>22}{total_best:>18.0f}{total_naive:>18.0f}"
      f"{total_naive - total_best:>17.0f}")

print()
print()
print("Where the money actually goes, for the two most expensive pieces of state.")
print()
for target in ("prompt cache", "document embeddings"):
    st = [s for s in STATE if s[0] == target][0]
    print(f"{target}:")
    print(f"{'store':>19}{'storage':>11}{'reads':>11}{'writes':>11}{'total':>11}")
    print("  " + "-" * 61)
    for store in STORES:
        c = cost(st, store)
        if c is None:
            continue
        print(f"{store[0]:>19}{st[1] * store[1]:>11.0f}{st[2] * store[2]:>11.0f}"
              f"{st[3] * store[3]:>11.0f}{c:>11.0f}")
    print()

print(f"""
The cost table is what happens when access shape is priced instead of assumed, and
the three rows where it disagrees with the default are the interesting ones.

**The prompt cache is the expensive mistake.** At
{[t for t in STATE if t[0] == 'prompt cache'][0][2]:.0f}k reads and
{[t for t in STATE if t[0] == 'prompt cache'][0][3]:.0f}k writes a month against only
{[t for t in STATE if t[0] == 'prompt cache'][0][1]:.0f} GB, it costs
{costs['prompt cache']['relational db']:.0f} a month in the database and
{costs['prompt cache']['key-value cache']:.0f} in a cache --
**{costs['prompt cache']['relational db'] / costs['prompt cache']['key-value cache']:.0f}
times more** for the identical data.

The reason is worth stating as a rule, because it is the one people get backwards. A
key-value cache is the most expensive storage in the table at
{STORES[2][1]:.2f} per GB -- {STORES[2][1] / STORES[3][1]:.0f} times object storage --
and the cheapest access at {STORES[2][2]:.3f} per thousand reads. **A cache is not
expensive storage; it is cheap access sold with expensive storage attached.** It wins
whenever the read count is large relative to the size, and the prompt cache is read
{[t for t in STATE if t[0] == 'prompt cache'][0][2] / [t for t in STATE if t[0] == 'prompt cache'][0][1]:.0f}
thousand times per gigabyte per month.

**Document originals invert it exactly.**
{[t for t in STATE if t[0] == 'document originals'][0][1]:.0f} GB read only
{[t for t in STATE if t[0] == 'document originals'][0][2]:.0f}k times a month is
large-and-cold, and object storage serves it for
{costs['document originals']['object storage']:.0f} against the database's
{costs['document originals']['relational db']:.0f} --
{costs['document originals']['relational db'] / costs['document originals']['object storage']:.0f}
times cheaper, on the same argument running the other way.

**And chat transcripts are the row where the default is right.** They are large
enough to tempt an object-storage migration and read often enough that it would cost
more: {costs['chat transcripts']['relational db']:.0f} in the database against
{costs['chat transcripts']['object storage']:.0f} in object storage. A rule that
moved everything large out of the database would get this one wrong by
{costs['chat transcripts']['object storage'] - costs['chat transcripts']['relational db']:.0f}
a month, which is why the decision needs the access shape rather than a size
threshold.

Totalling the columns gives the chapter's number. Placing all seven by access shape
costs {total_best:.0f} a month; leaving everything in the primary database except the
embeddings costs {total_naive:.0f} -- **{total_naive / total_best:.1f} times more**
(eq:access-shape-decides-the-store), for identical data, identical durability, and
identical query results.

Two of the seven rows account for essentially all of it.
{(costs['prompt cache']['relational db'] - costs['prompt cache']['key-value cache'] + costs['document originals']['relational db'] - costs['document originals']['object storage']) / (total_naive - total_best):.0%}
of the overpayment is the prompt cache and the document originals, which is the
useful operational form of the result: **you do not need to reclassify your whole
storage layer, you need to find the two pieces of state whose access shape is most
extreme.**

The general rule the table encodes is simple enough to apply without arithmetic.
Storage cost is dominated by whichever of size and access rate is larger relative to
the store's pricing. Large-and-cold belongs in object storage almost regardless of
what it contains; small-and-hot belongs in a cache almost regardless of what it
contains; and the middle is where transactional and similarity requirements actually
decide, because only there is the cost difference small enough for a capability to be
worth paying for.

That is why sorting state by what it IS fails in the specific way it does. The two
dimensions that dominate the bill are both properties of how it is USED, and neither
is visible in the name.""")
```

## 9. Practical Example

Four stores, priced separately for size and access:

```
            store   $/GB-mo   $/1k rd   $/1k wr   latency  similarity   txns
----------------------------------------------------------------------------
    relational db     0.230     0.180     0.900      3.0ms          no    yes
     vector store     0.850     0.400     2.200     12.0ms         yes     no
  key-value cache     3.400     0.012     0.030      0.4ms          no     no
   object storage     0.021     0.420     5.400     95.0ms          no     no
```

The cache costs **162×** object storage per gigabyte and **1/35th** as much per
thousand reads. That spread is the whole chapter in one row.

Pricing the seven pieces of state:

```
                 state   relational db    vector store  key-value cach  object storage
--------------------------------------------------------------------------------------
      chat transcripts             843            2315            3219            3527
   document embeddings              --            1898              --              --
    document originals            1491            5484           21761             218
         user profiles            1711              --              --              --
          prompt cache           12879              --             714              --
   tool call audit log            1746              --              --              --
       eval label sets               9              --              --              --
```

Against the default of leaving everything in the primary database:

```
                 state   by access shape        by default  cost of default
----------------------------------------------------------------------------
      chat transcripts     relational db     relational db                0
   document embeddings      vector store      vector store                0
    document originals    object storage     relational db             1273
         user profiles     relational db     relational db                0
          prompt cache   key-value cache     relational db            12165
   tool call audit log     relational db     relational db                0
       eval label sets     relational db     relational db                0
----------------------------------------------------------------------------
                 TOTAL              7139             20576            13437
```

Placement by access shape costs **7139**; the default costs **20576** — **2.9×**
more ({{eq:access-shape-decides-the-store}}) for identical data, identical
durability, and identical query results.

**All of the overpayment is two rows.** The prompt cache — 38 GB read 26 million
times a month — costs **12879** in the database against **714** in a cache, **18×**
({{eq:cache-is-cheap-access-not-cheap-storage}}). Document originals — 6400 GB read
45 thousand times — cost **1491** in the database against **218** in object storage,
**7×**, on the same argument running the other way.

And chat transcripts are the row where the default is right: **843** in the database
against **3527** in object storage. A rule that moved everything large out of the
database would get this one wrong by 2684 a month, which is why the decision needs
the access shape rather than a size threshold.

The second listing turns to the same fact stored several times over.

```python {tier=A name=by2}
"""Every derived copy of a fact is another chance for the system to contradict itself.

An AI system stores the same fact several times over. The source document sits in
object storage; a chunked copy sits in a vector index; an extracted summary sits in a
database; a cached answer quoting it sits in a cache. Each is derived from the last
and each updates on its own schedule.

When the underlying fact changes, those copies converge at different times. During
the gap the system can retrieve one version and quote another
(eq:derived-copies-multiply-contradiction).

The window in which that is possible is E[max lag] - E[min lag] across the copies.
Since the source itself is never stale, that window is set by the DEEPEST copy --
which makes pipeline depth, not pipeline speed, the parameter that matters.
"""
import math

# Derived representations, in dependency order. Each is rebuilt from the one
# above it. (name, mean lag in minutes after the source changes, lag std dev)
PIPELINE = [
    ("source document",      0.0,   0.0),
    ("extracted text",       4.0,   3.0),
    ("chunk embeddings",    22.0,  26.0),
    ("summary row",         35.0,  40.0),
    ("cached answer",      110.0, 210.0),
]
CHANGE_PER_DAY = 3.2       # how often a given fact changes
QUERIES_PER_DAY = 900.0    # queries touching that fact
MINUTES = 1440.0
GRID = [i * 0.5 for i in range(0, 4801)]      # 0 to 2400 minutes


def cdf(t, mean, sd):
    """P(lag <= t) for a lognormal lag with the given mean and standard deviation.

    Lognormal because a rebuild lag is non-negative and right-skewed: it is
    usually near its typical value and occasionally much longer.
    """
    if mean <= 0.0:
        return 1.0
    if t <= 0.0:
        return 0.0
    if sd <= 0.0:
        return 1.0 if t >= mean else 0.0
    var = math.log(1.0 + (sd * sd) / (mean * mean))
    mu = math.log(mean) - var / 2.0
    return 0.5 * (1.0 + math.erf((math.log(t) - mu) / (math.sqrt(var) * math.sqrt(2.0))))


def disagreement_window(copies):
    """E[max lag] - E[min lag]: the expected minutes, after each change, during
    which at least one copy has caught up and at least one has not."""
    if len(copies) < 2:
        return 0.0
    step = GRID[1] - GRID[0]
    e_max = 0.0
    e_min = 0.0
    for t in GRID:
        all_done = 1.0
        none_done = 1.0
        for _, m, s in copies:
            f = cdf(t, m, s)
            all_done *= f
            none_done *= (1.0 - f)
        e_max += (1.0 - all_done) * step      # E[max] = integral of P(max > t)
        e_min += none_done * step             # E[min] = integral of P(min > t)
    return e_max - e_min


def contradiction_rate(copies):
    """P(a random query lands inside a disagreement window)."""
    w = disagreement_window(copies)
    return min(1.0, w * CHANGE_PER_DAY / MINUTES)


print("A fact, stored five times over. Each copy is rebuilt from the one above.")
print()
print(f"{'representation':>20}{'mean lag':>12}{'std dev':>10}"
      f"{'spread ratio':>15}")
print("-" * 57)
for name, m, sp in PIPELINE:
    ratio = sp / m if m else 0.0
    print(f"{name:>20}{m:>10.0f}m{sp:>9.0f}m{ratio:>15.2f}")

print()
print()
print("Contradiction rate as copies are added, one at a time. The window is")
print("E[slowest copy] - E[fastest copy] after each change to the fact.")
print()
print(f"{'copies kept':>13}{'deepest copy':>20}{'window':>11}"
      f"{'contradiction':>16}{'queries/day':>14}")
print("-" * 74)
rates = {}
for i in range(2, len(PIPELINE) + 1):
    sub = PIPELINE[:i]
    w = disagreement_window(sub)
    r = contradiction_rate(sub)
    rates[i] = (r, w)
    print(f"{i:>13}{sub[-1][0]:>20}{w:>10.0f}m{r:>16.2%}"
          f"{r * QUERIES_PER_DAY:>14.0f}")

print()
print()
print("Now hold the number of copies at five and vary how fast the fact changes.")
print("This is the parameter a product decision moves without noticing.")
print()
print(f"{'changes/day':>13}{'contradiction rate':>21}{'queries/day affected':>22}")
print("-" * 56)
byrate = {}
BASE_WINDOW = disagreement_window(PIPELINE)
for cpd in (0.2, 1.0, 3.2, 12.0, 48.0):
    r = min(1.0, BASE_WINDOW * cpd / MINUTES)
    byrate[cpd] = r
    print(f"{cpd:>13.1f}{r:>21.2%}{r * QUERIES_PER_DAY:>22.0f}")

print()
print()
print("Three ways to attack the window: speed up every stage, tighten the")
print("variance, or synchronise everything onto one schedule.")
print()
print(f"{'strategy':>36}{'worst lag':>12}{'window':>10}{'contradiction':>16}")
print("-" * 74)


def variant(scale_mean, scale_sd):
    return [(n, m * scale_mean, s * scale_sd) for n, m, s in PIPELINE]


OPTIONS = [
    ("as built",                             1.00, 1.00),
    ("all lags halved",                      0.50, 0.50),
    ("all lags quartered",                   0.25, 0.25),
    ("variance halved, means unchanged",     1.00, 0.50),
    ("variance quartered, means unchanged",  1.00, 0.25),
    ("all copies on one 110m schedule",      0.00, 0.00),
]
res = {}
for label, sm, ss in OPTIONS:
    if label.startswith("all copies on one"):
        v = [(n, 0.0 if m == 0 else 110.0, 0.0) for n, m, s in PIPELINE]
    else:
        v = variant(sm, ss)
    r = contradiction_rate(v)
    w = disagreement_window(v)
    worst = max(m for _, m, _ in v)
    res[label] = (r, worst, w)
    print(f"{label:>36}{worst:>10.0f}m{w:>9.0f}m{r:>16.2%}")

print()
print()
print("And the structural alternative: stop deriving. Serve the deep copies from")
print("the shallow ones at query time instead of materialising them.")
print()
print(f"{'design':>32}{'copies':>9}{'contradiction':>16}{'read cost':>12}")
print("-" * 69)
struct = {}
for label, keep, extra in (
        ("materialise all five",        5, 1.00),
        ("materialise four",            4, 1.35),
        ("materialise three",           3, 1.90),
        ("materialise two",             2, 3.10),
        ("source of truth only",        1, 6.40)):
    r = contradiction_rate(PIPELINE[:keep]) if keep >= 2 else 0.0
    struct[keep] = (r, extra)
    print(f"{label:>32}{keep:>9}{r:>16.2%}{extra:>11.2f}x")

print(f"""
The pipeline table looks unremarkable. Every lag is a number somebody chose
deliberately, and the slowest is under two hours. No individual row is alarming.

The second table is what those rows compose into. Two representations disagree for
{rates[2][1]:.0f} minutes after each change, giving a contradiction rate of
{rates[2][0]:.2%}. Five representations disagree for {rates[5][1]:.0f} minutes,
giving {rates[5][0]:.2%} (eq:derived-copies-multiply-contradiction) -- at
{QUERIES_PER_DAY:.0f} queries a day, {rates[5][0] * QUERIES_PER_DAY:.0f} queries
where the system can retrieve one version of a fact and quote another.

Look at where the growth comes from. Adding the cached answer -- ONE copy -- takes
the window from {rates[4][1]:.0f} minutes to {rates[5][1]:.0f}, nearly tripling it,
while adding the summary row before it added only
{rates[4][1] - rates[3][1]:.0f}. The window is the distance from the fastest copy to
the slowest, the source is never stale, and so **the deepest copy sets the window
almost single-handedly.**

**Nothing in any store's monitoring shows this.** Every store is healthy, every write
succeeded, every read returned exactly what it held. The inconsistency exists only
BETWEEN the stores, which is precisely where nobody is looking -- the same gap
ch:sd-architecture identified for semantic failure, relocated to the storage layer.

The change-rate table is why this is a product problem rather than an infrastructure
one. At {0.2:.1f} changes per day the contradiction rate is {byrate[0.2]:.2%}; at
{48.0:.0f} it is {byrate[48.0]:.2%}. A feature that makes documents editable, or a
migration that starts syncing an upstream system hourly, moves this parameter by an
order of magnitude with no storage change at all and no review that would catch it.

The strategy table is where the intuition carried over from ch:sd-async breaks, and
the break is worth being precise about. There, variance did the damage and reducing
the mean barely helped. Here it is the other way round: halving every lag and its
variance takes contradiction from {res['as built'][0]:.2%} to
{res['all lags halved'][0]:.2%}, while halving the variance alone reaches only
{res['variance halved, means unchanged'][0]:.2%} and quartering it
{res['variance quartered, means unchanged'][0]:.2%}.

Synchronising is worth even less. Putting every derived copy on one
{110.0:.0f}-minute schedule gives {res['all copies on one 110m schedule'][0]:.2%} --
almost no improvement -- and the reason is the one thing the two chapters do share.
A queue's wait depends on the spread of service times because every job is compared
against every other. **A staleness window is measured against the source, and the
source is instant.** You cannot synchronise a derived copy to a thing that was never
late; there is no schedule slow enough to close a gap whose other end is zero.

So the lever here is neither speed nor synchronisation. It is DEPTH, and the last
table prices it. Dropping the cached answer -- the single deepest copy -- takes
contradiction from {struct[5][0]:.2%} to {struct[4][0]:.2%}, better than halving
every lag in the pipeline achieved, and it costs {struct[4][1]:.2f} times the read
cost rather than an infrastructure project. Dropping to three copies reaches
{struct[3][0]:.2%} at {struct[3][1]:.2f} times.

**Removing one derived copy beat making the entire pipeline twice as fast.** That is
the chapter's practical result, and it inverts the usual response, which is to keep
every materialisation and buy freshness with engineering.

It is worth naming what a materialised derived copy actually is, because the name
makes the decision obvious. **It is a cache** -- precomputed output kept because
recomputing is expensive -- and every cache over a changing source is a bet that
staleness costs less than recomputation. Here the bet is made implicitly, once, by
whoever built the pipeline, and it is never revisited when the change rate moves,
which eq:cache-threshold-is-an-error-cost-decision says is exactly when it should
be.""")
```

Five derived representations, each rebuilt from the one above:

```
      representation    mean lag   std dev   spread ratio
---------------------------------------------------------
     source document         0m        0m           0.00
      extracted text         4m        3m           0.75
    chunk embeddings        22m       26m           1.18
         summary row        35m       40m           1.14
       cached answer       110m      210m           1.91
```

No individual row is alarming. What they compose into is:

```
  copies kept        deepest copy     window   contradiction   queries/day
--------------------------------------------------------------------------
            2      extracted text         4m           0.94%             9
            3    chunk embeddings        23m           5.01%            45
            4         summary row        43m           9.64%            87
            5       cached answer       123m          27.36%           246
```

Five copies contradict each other on **27.36%** of queries — **246** queries a day
where the system can retrieve one version of a fact and quote another
({{eq:derived-copies-multiply-contradiction}}). Adding the cached answer alone took
the window from 43 to **123** minutes, nearly tripling it.

**Nothing in any store's monitoring shows this.** Every store is healthy, every write
succeeded, every read returned exactly what it held. The inconsistency exists only
*between* the stores.

```mermaid {#fig:derived caption="The contradiction window runs from the source, which is never stale, to the deepest materialised copy. Synchronising the derived copies cannot close a gap whose other end is fixed at zero; removing the deepest copy can."}
flowchart LR
  S["source document<br/>lag 0"] --> A["extracted text<br/>4m"]
  A --> B["chunk embeddings<br/>22m"]
  B --> C["summary row<br/>35m"]
  C --> D["cached answer<br/>110m"]
  S -.->|"window = 123m"| D
```

Three ways to attack it:

```
                            strategy   worst lag    window   contradiction
--------------------------------------------------------------------------
                            as built       110m      123m          27.36%
                     all lags halved        55m       62m          13.81%
                  all lags quartered        28m       31m           6.94%
    variance halved, means unchanged       110m      114m          25.32%
 variance quartered, means unchanged       110m      110m          24.54%
     all copies on one 110m schedule       110m      110m          24.44%
```

Halving every lag and its variance reaches **13.81%**. Halving variance alone reaches
only **25.32%**, and synchronising every copy onto one schedule reaches **24.44%** —
almost nothing. **You cannot synchronise a derived copy to a source that was never
late.**

This is where the intuition from {{ch:sd-async}} breaks, and the break is instructive.
There, spread did the damage because a queue compares every job against every other.
Here the window is measured against a fixed zero, so the *extreme value* does the
damage. Both chapters say "not the mean" and mean different things by it.

The lever is depth:

```
                          design   copies   contradiction   read cost
---------------------------------------------------------------------
            materialise all five        5          27.36%       1.00x
                materialise four        4           9.64%       1.35x
               materialise three        3           5.01%       1.90x
                 materialise two        2           0.94%       3.10x
            source of truth only        1           0.00%       6.40x
```

Dropping the single deepest copy takes contradiction from **27.36%** to **9.64%** at
**1.35×** the read cost — **better than halving every lag in the pipeline achieved**,
and a configuration change rather than an infrastructure project
({{eq:depth-beats-speed-for-staleness}}).

## 10. Production Considerations

Compute reads-per-gigabyte and writes-per-gigabyte for every piece of state you hold.
It is a query against billing data plus a size listing, it takes an afternoon, and by
{{eq:cache-is-cheap-access-not-cheap-storage}} it is sufficient to identify every
misplacement worth fixing.

Look for the two extremes first. The overpayment concentrates in whatever is smallest
and hottest and whatever is largest and coldest; the middle of the distribution is
rarely worth moving.

Enumerate the derived copies of a fact explicitly, as a diagram, and put a lag
estimate on each edge. Most teams have never written this down and are surprised by
the count — five is common and eight is not rare.

Treat the deepest copy as the thing to justify. It sets the contradiction window
almost alone, and it is usually the one added most recently, with the least
consideration, by whoever needed a quick win on latency.

Alert on the *window*, not on individual pipeline lags. A pipeline lag alarm fires
when one stage is slow; the window is what determines whether users see contradictions
and no per-stage alarm computes it.

Version derived copies with the source revision they were built from, and check the
revision at read time. This does not close the window but it makes contradictions
detectable, which converts an invisible failure into a loggable one — the storage
layer's version of {{ch:sd-architecture}}'s second instrument.

Re-examine placement after any change in change-rate. A feature that makes documents
editable moves the contradiction rate by an order of magnitude with no storage change
and no review that would catch it.

## 11. Common Mistakes

**Sorting state by what it is.** The two dimensions that dominate the bill are both
properties of how it is used.

**Treating a cache as expensive.** It is expensive storage and cheap access; whether
it is expensive depends entirely on reads per gigabyte.

**Applying a size threshold to decide object storage.** Chat transcripts are large and
belong in the database; the threshold has to be on the ratio.

**Making the pipeline faster to fix staleness.** Works, linearly, expensively —
and is beaten by removing one copy.

**Synchronising derived copies.** Cannot close a window whose other end is zero.

**Adding a materialised copy for latency without pricing the staleness.** This is how
the deepest copy usually arrives.

## 12. Failure Modes

**Silent contradiction.** The system retrieves one version and quotes another; both
stores report success and no dashboard moves.

**Change-rate drift.** A product change raises the fact change rate and the
contradiction rate rises with it, with no deployment to correlate against.

**Index rebuild starvation.** Under sustained write load the vector index rebuild
never completes, so its lag grows without bound and quietly becomes the deepest copy.

**Cost inversion after growth.** A piece of state correctly placed at one scale
becomes misplaced as its access-to-size ratio moves; nothing re-evaluates it.

**Orphaned derived state.** A copy whose consumer was removed continues to be
maintained, paying write costs and widening the contradiction window for no benefit.

## 13. Alternatives

**Single store for everything.** Operationally simple, and the **2.9×** premium is
sometimes worth paying at small scale where the absolute numbers are trivial. It stops
being worth it exactly when someone notices the bill.

**Compute everything at read time.** Zero contradiction, **6.4×** the read cost in
the listing's model. Correct for low-traffic, high-stakes surfaces — which is
{{eq:cache-threshold-is-an-error-cost-decision}}'s high-$\lambda$ regime arriving in
the storage layer.

**Event-sourced invalidation.** Propagate changes as events so every derived copy
invalidates together rather than rebuilding on a timer. Collapses the window to
propagation latency at the cost of a delivery guarantee you must then operate.

**Read-through with revision check.** Serve the materialised copy but verify its
source revision, falling back to recomputation on mismatch. Bounds contradiction at
the cost of a check per read, and composes well with
{{eq:depth-beats-speed-for-staleness}}.

**Tiered storage with automatic migration.** Let the store move data between hot and
cold tiers by observed access. Captures much of
{{eq:access-shape-decides-the-store}} automatically, and captures none of the
capability reasoning.

## 14. Evaluation

Report cost per piece of state, not cost per store. A per-store bill cannot show that
one item inside it is misplaced, which is exactly the finding.

Measure the contradiction window directly by instrumenting source revisions through
the derivation chain. Estimating it from configured lags is a starting point;
measuring it catches the rebuild that silently stopped completing.

Track derivation depth as an architectural metric alongside testable share from
{{ch:sd-architecture}}. Both are counts that predict a class of failure and neither
appears on any default dashboard.

Sample and diff derived copies against freshly recomputed ones. The disagreement rate
on that sample is the ground truth for
{{eq:derived-copies-multiply-contradiction}} and it needs no labels.

Evaluate placement decisions against the access ratios, not against absolute cost.
Absolute cost changes with scale; the ratios are what determine the correct store.

## 15. Advanced Concepts

The independence assumption in {{eq:derived-copies-multiply-contradiction}} is
optimistic in a useful direction. Real derivation chains are *sequential* — the
summary cannot rebuild until the extraction has — so lags are positively correlated
and $\max_k L_k$ is closer to $\sum_k \ell_k$ than the independent model suggests.
Sequential dependency makes depth matter even more than
{{eq:depth-beats-speed-for-staleness}} indicates, because each stage's lag adds rather
than competing for the maximum.

Every materialised derived copy is a cache, and recognising that makes
{{ch:sd-routing-caching}}'s arithmetic apply directly. The threshold question there
was similarity; here it is staleness tolerance, but
{{eq:cache-threshold-is-an-error-cost-decision}} has the same form: the optimum
depends on what a wrong answer costs on that surface, not on the pipeline's
properties. A summary shown in a list view and a summary quoted in a price
quotation deserve different derivation depths, and almost no system distinguishes
them.

There is a subtlety in how contradiction should be weighted that the model
deliberately ignores. It counts a query as contradictory whenever any two copies
disagree, but not every disagreement reaches the user: a query that retrieves a stale
chunk and never consults the summary row is unaffected by their divergence. The
observable contradiction rate is therefore lower than
{{eq:derived-copies-multiply-contradiction}} suggests, by a factor equal to the
probability that a single query actually touches two divergent copies. That factor is
measurable — it is a property of the query plan, not of the storage — and it varies
enormously between systems. A retrieval-augmented answer that quotes a cached summary
alongside freshly retrieved chunks touches two copies on every request; a system that
reads exactly one representation per query touches two on none. **The count of
derived copies bounds the problem; the query plan determines how much of the bound is
realised**, and a system can lower its exposure by reading fewer representations per
query without removing any of them.

The placement problem in {{eq:access-shape-decides-the-store}} was solved
independently per item, which is correct only because the stores have no shared
capacity constraint in this model. With reserved capacity, minimum commitments, or a
fixed cache size, placement becomes a knapsack problem and the greedy per-item choice
is no longer optimal — though it remains a good starting point, since the two extreme
items dominate the objective.

## 16. Connection to Previous Chapters

{{eq:semantic-failure-has-no-instrument}} from {{ch:sd-architecture}} predicted
failures no instrument catches. {{eq:derived-copies-multiply-contradiction}} is one
that lives between instruments rather than beneath them, which is why it is harder
to retrofit a detector for.

{{eq:variance-not-mean-drives-wait}} from {{ch:sd-async}} is the informative contrast:
spread dominates a queue, extreme value dominates a staleness window, and conflating
the two leads to buying the wrong fix.

{{eq:cache-threshold-is-an-error-cost-decision}} from {{ch:sd-routing-caching}}
governs every materialisation decision here once derived copies are recognised as
caches.

{{eq:scale-buys-redundancy-not-coverage}} from {{ch:sd-retrieval-agents}} interacts
with index rebuild scheduling: a corpus growing fast enough to change coverage is also
growing fast enough to make rebuild lag the deepest copy.

## 17. Exercises

1. Derive the reads-per-gigabyte threshold at which the cache beats the database, for
   the listing's prices. Which of the seven items clear it?

2. Add a fifth store — a columnar analytics warehouse, cheap per gigabyte, expensive
   per point read — and find which items move.

3. Modify the second listing so lags are sequential rather than independent. How much
   worse is the window?

4. Compute the read-cost multiplier at which materialising four copies beats
   materialising three, given a stated cost per contradiction.

5. Diagram the derived copies in a system you work on. How many are there, and which
   is deepest?

## 18. Interview Questions

1. Why is a key-value cache both the most and least expensive storage in a typical
   stack?

2. Our chat transcripts are 940 GB. Should they move to object storage?

3. The system sometimes cites a document version that does not match the summary it
   shows. Where do you look, and what would have caught it?

4. Why does synchronising every derived copy onto one schedule barely reduce
   contradictions?

5. Both {{ch:sd-async}} and this chapter say the mean is the wrong statistic. Do they
   mean the same thing?

## 19. Research Questions

1. Can the contradiction window be estimated from telemetry alone, without
   instrumenting source revisions through every derived copy?

2. What is the right formulation of derivation depth as a cost, so it can be
   optimised jointly with storage placement rather than after it?

3. How does incremental-insertion recall decay in modern ANN indexes actually scale
   with write volume, and what rebuild interval does that imply?

4. Is there a principled way to assign per-surface staleness tolerance, analogous to
   {{ch:sd-routing-caching}}'s per-surface error cost?

## 20. Chapter Summary

Storage placement is decided by access shape, not by data type. Cost is
$g\gamma + r\rho + w\omega$ ({{eq:access-shape-decides-the-store}}), so the crossovers
are governed by reads and writes *per gigabyte*. Pricing seven realistic pieces of
state, placement by access shape costs **7139** a month against **20576** for the
default of leaving everything in the primary database — **2.9×** — with all of the
overpayment in two items.

A cache is not expensive storage; it is cheap access sold with expensive storage
attached ({{eq:cache-is-cheap-access-not-cheap-storage}}), which is why a 38 GB prompt
cache read 26 million times costs **18×** more in a database, and 6400 GB of document
originals read 45 thousand times costs **7×** more.

The same fact is stored in several derived forms, and they converge at different
times. Five copies contradict each other on **27.36%** of queries
({{eq:derived-copies-multiply-contradiction}}) with every store reporting perfect
health, because the inconsistency exists only between them.

The window runs from a source that is never stale to the deepest copy, so
synchronisation is nearly useless (**24.44%**) and halving every lag helps only
linearly (**13.81%**), while removing the single deepest copy reaches **9.64%** at
**1.35×** read cost ({{eq:depth-beats-speed-for-staleness}}).

Both halves of the chapter reward the same move: counting something nobody counts.
Reads per gigabyte is not on any storage dashboard, and derivation depth is not on
any architecture diagram, yet each one predicts a large and specific cost that the
quantities teams do track cannot see at all.

Carry forward: **place state by reads per gigabyte**, and **shorten the derivation
chain before speeding it up**.

## 21. Further Reading

- {{cite:malkov2020hnsw}} — HNSW; why vector-store storage is priced like memory.
- {{cite:johnson2019faiss}} — billion-scale similarity search and index rebuild
  structure.
- {{cite:guo2020scann}} — anisotropic quantisation; the storage-versus-recall trade
  inside the index.
- {{cite:kwon2023pagedattention}} — KV cache management, which creates the small-hot
  state this chapter places.

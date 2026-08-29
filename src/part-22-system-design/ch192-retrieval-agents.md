---
id: sd-retrieval-agents
number: 192
part: XXII
tier: full
status: draft
requires: [variance-not-mean-drives-wait, three-properties-break-the-stack,
           semantic-failure-has-no-instrument, context-is-a-budget]
provides: [fanout-amplifies-the-tail, hedging-beats-optimising-dependencies,
           scale-buys-redundancy-not-coverage, diversity-beats-depth]
citations: [malkov2020hnsw, johnson2019faiss, cemri2025mast, qin2023toolllm]
---

## 1. Learning Objectives

By the end of this chapter you will be able to compute how a fan-out of width $n$
converts a per-dependency tail into a request-level one, and why the conversion is
geometric rather than linear; show why optimising a dependency cannot close the gap
that fan-out opens, and why hedging can; explain why growing a corpus makes retrieval
*more* redundant rather than more complete; distinguish precision and coverage as
retrieval objectives and say which one the answer depends on; and choose between
retrieving more and reordering what you already retrieved, with a number attached to
each.

## 2. Why This Matters

An agent that calls five tools in parallel waits for the slowest. A retrieval layer
that queries eight shards waits for the slowest. Neither waits for the mean, and the
maximum of a sample behaves nothing like its mean.

This is the reason a system built entirely from fast, healthy components can be slow
and unreliable with **nothing in any component's dashboard looking wrong**.
{{sec:9-practical-example}} takes a dependency that returns in 0.05s **60%** of the
time with a **1.3%** slow tail — by any normal standard a healthy dependency — and
fans it out. At width 20, the tail is hit on **23.0%** of requests
({{eq:fanout-amplifies-the-tail}}). The dependency's own dashboard still correctly
reports a 1.3% slow rate. That number now describes something happening to nearly a
quarter of user requests.

The second half concerns retrieval at scale, and contains this part's fourth instance
of a metric that improves while the system degrades. Growing a corpus from 1,000 to
10 million documents drives precision@10 from **4%** to **100%** — an unambiguous
success by the standard retrieval metric — while fact coverage falls from **55.5%**
to **17.2%** ({{eq:scale-buys-redundancy-not-coverage}}). Every returned document is
genuinely on topic. The answers are getting worse.

## 3. Prerequisites

You need {{ch:sd-async}}'s result that variance rather than mean drives waiting
({{eq:variance-not-mean-drives-wait}}). This chapter is the same phenomenon in a
different topology: there, one queue with variable service times; here, many parallel
calls whose maximum is what matters.

{{eq:three-properties-break-the-stack}} from {{ch:sd-architecture}} supplies the cost
asymmetry that decides when hedging is affordable.

{{eq:semantic-failure-has-no-instrument}} is the pattern both halves reproduce.

{{eq:context-is-a-budget}} from {{ch:mcp-schemas}} is the binding constraint in the
retrieval half — the reason retrieval depth cannot be bought indefinitely.

Familiarity with approximate nearest-neighbour indexes ({{ch:emb-ann}}) is assumed
for {{sec:7-internal-mechanics}} but not for the results.

## 4. Intuitive Explanation

Start with a dependency you would be happy with. It answers in 50 milliseconds most
of the time. Occasionally — about one call in eighty — something goes wrong and it
takes 1.6 seconds: a cold cache, a large document, an unlucky shard. Its p95 is 180
milliseconds. Nobody is going to file a ticket about this service.

Now call it five times in parallel, because your agent needs five tools, and you
cannot proceed until all five return. What is the chance that *at least one* of the
five hits the slow path?

Not one in eighty. Roughly one in sixteen. And at twenty parallel calls, roughly one
in four. The rare event has not become more likely — each individual call is exactly
as reliable as before — but you are now taking many draws per request, and a request
is only fast if **every** draw is fast.

This is the same arithmetic as {{ch:ag-loop}}'s chain, and it is worth noticing that
running the steps in parallel does not change it. Parallelism improves latency; it
does nothing at all for reliability, because reliability is a product either way.

The consequence is a conversation that goes wrong in a predictable direction. The
agent is slow. Someone profiles the tools and finds they are all fast. Someone else
proposes optimising the slowest one. But look at what optimisation would have to
achieve: to keep 95% of requests inside budget at fan-out 20, each dependency needs
to be fast **99.74%** of the time, against the **98.70%** it currently manages. That
means cutting the tail by four-fifths — and the tail is where the real work is. A
cold cache is a cold cache.

What does work is refusing to wait for the slow one. Issue a duplicate call for
anything still outstanding after a short delay, and take whichever answer arrives
first. You pay a few percent more load and you attack the maximum directly instead
of trying to improve the distribution it is drawn from.

The retrieval half has a different shape but the same moral. Intuition says a bigger
corpus is better: more documents, more chance the answer is in there. That is true
about the corpus and false about what you retrieve from it.

The reason is that popular things get written about repeatedly and rare things do
not. Double the corpus and you roughly double the number of documents restating the
well-known facts, while the obscure fact is still in one document. Similarity search
ranks by resemblance to the query, and resemblance tracks popularity. So the bigger
the corpus, the more your ten slots fill with ten restatements of the same
well-covered material.

Every one of those ten documents is relevant. Precision is perfect. And the answer
is missing the thing that was only ever mentioned once.

## 5. Formal Explanation

Let a request fan out to $n$ independent calls with latency distribution $X$ and CDF
$F$. The request completes when the last one does, so its latency is
$M_n = \max(X_1, \ldots, X_n)$ with

$$ P(M_n \le t) \;=\; F(t)^n $$ (eq:fanout-amplifies-the-tail)

The consequences follow from the exponent. Writing $\pi = 1 - F(t)$ for the
per-call probability of exceeding $t$, the request-level probability is
$1 - (1 - \pi)^n \approx n\pi$ for small $\pi$ — **linear amplification of the tail
in the fan-out width**. And the $q$-th percentile of $M_n$ is the
$q^{1/n}$-th percentile of $X$, so the p95 of a 20-way fan-out is the p99.74 of a
single call.

That second form is the useful one for design, because it converts a request-level
budget into a per-dependency requirement. To hold $P(M_n \le t) \ge q$ you need

$$ F(t) \;\ge\; q^{1/n} $$

which for $q = 0.95, n = 20$ demands $F(t) \ge 0.9974$. As $n$ grows this requirement
approaches 1 faster than any dependency can be improved, which is the formal content
of "you cannot optimise your way out of fan-out."

Hedging changes the distribution rather than the exponent. Issuing a duplicate at
time $h$ and taking the first response replaces the tail beyond $h$ with $h$ plus a
fresh draw, giving an effective per-call CDF

$$ \tilde{F}(t) \;=\; \begin{cases} F(t) & t \le h \\ F(h) + (1 - F(h))\,F(t - h) & t > h \end{cases} $$ (eq:hedging-beats-optimising-dependencies)

The cost is $n(1 - F(h))$ extra calls per request. Because $\tilde{F}$ is
substantially tighter than $F$ in the tail, and because
{{eq:fanout-amplifies-the-tail}} raises whatever it is given to the $n$-th power, a
modest improvement in $\tilde{F}$ produces a large improvement in $M_n$.

For retrieval, let a question require $F$ distinct facts with prevalence $p_f$
following a Zipf law, and let retrieval at corpus size $N$ return documents whose
fact content is drawn with probability proportional to $p_f^{s(N)}$, where the
selectivity $s(N)$ increases with $N$ because a larger corpus offers more
near-duplicates of popular material. Coverage of the needed facts by the top-$k$ is

$$ C(N, k) \;=\; \frac{1}{F}\sum_{f=1}^{F}\Bigl(1 - (1 - q_f(N))^k\Bigr), \qquad q_f(N) \propto p_f^{s(N)} $$ (eq:scale-buys-redundancy-not-coverage)

Since $s$ is increasing in $N$ and $p_f < 1$, every $q_f$ for rare $f$ decreases with
$N$. **Coverage falls as the corpus grows, at fixed $k$ and fixed retriever.**

## 6. Mathematical Foundation

The retrieval result has a design consequence worth deriving, because it settles a
choice teams make badly.

Given a fixed context budget of $k$ slots, compare two policies. **Depth**: take the
top $k$ by similarity, which is {{eq:scale-buys-redundancy-not-coverage}}.
**Diversity**: choose slots greedily to maximise marginal coverage, so slot $j$ goes
to the document maximising expected gain over what slots $1..j-1$ already cover.

The greedy coverage objective is submodular, so greedy selection is within
$1 - 1/e \approx 63\%$ of optimal. More useful is the comparison against depth. Depth
spends slot $j$ on the $j$-th most similar document, whose marginal contribution is
$\sum_f q_f (1 - q_f)^{j-1}$ — dominated by high-$q_f$ facts already covered.
Diversity spends it on the highest *residual*. The ratio of the two grows with $s(N)$,
giving

$$ \frac{C_{\text{div}}(N,k)}{C_{\text{depth}}(N,k)} \;\; \text{increasing in } N $$ (eq:diversity-beats-depth)

**The larger the corpus, the more reordering is worth relative to retrieving more.**
{{sec:9-practical-example}} measures the gain rising from **+8.6** points at 1,000
documents to **+25.3** points at 10 million — where the reordered ten slots match what
plain depth needs **111** slots to achieve.

That is an eleven-fold effective increase in retrieval depth obtained without
retrieving anything additional, which matters because
{{eq:context-is-a-budget}} says the slots are the scarce resource. Holding coverage
flat by depth alone needs $k = 39$ at 100,000 documents — 16,380 tokens against a
12,000-token budget. **Scale defeats retrieval depth before it defeats the context
window.**

## 7. Internal Mechanics

**Where the fan-out tail comes from.** In an ANN index, query latency is bimodal:
most queries traverse a short path through the graph, and some fall into a region
requiring extensive backtracking. {{cite:malkov2020hnsw}}'s hierarchical structure
makes the common case fast and does not eliminate the tail; sharding multiplies the
number of draws per query, which is {{eq:fanout-amplifies-the-tail}} applied inside
what looks like a single dependency. A "one call" to a sharded vector store is a
fan-out whether or not the caller knows it.

**The shard-count trap.** Sharding a vector index is usually presented as a
scaling decision with a throughput justification: more shards, more parallel
capacity, smaller working set per node. That reasoning is sound and it omits the
cost. Going from four shards to sixteen quadruples the number of draws per query,
and by {{eq:fanout-amplifies-the-tail}} that raises the request-level tail by roughly
the same factor. A team that shards for throughput and measures the result on mean
latency will see an improvement; the p99 moves the other way. The right shard count
balances throughput against fan-out width, and almost nobody writes the second term
down.

**Why hedges must be independent.** {{eq:hedging-beats-optimising-dependencies}}
assumes the retry draws freshly from $F$. A hedge routed to the same overloaded shard
draws from a distribution conditioned on that shard being slow, and buys nothing. The
hedge must go somewhere else — a different replica, a different shard copy — which is
an infrastructure requirement, not a client one.

**Agent fan-out is correlated fan-out.** {{cite:cemri2025mast}}'s failure taxonomy and
{{eq:agent-errors-correlate}} both indicate that parallel agent calls fail together
more often than independence predicts, usually because they share a context, a
retrieved document set, or a rate limit. Correlated fan-out is *worse* than
{{eq:fanout-amplifies-the-tail}} suggests for reliability and *better* for latency,
since correlated slowness at least happens at the same time.

**Tool count and selection.** {{cite:qin2023toolllm}}'s work on large tool
collections runs into the same redundancy problem as document retrieval: many tools
are near-duplicates, and selecting the top-$k$ by description similarity fills the
budget with variants of one capability. The diversity argument in
{{eq:diversity-beats-depth}} applies unchanged.

**Diversity signals that do not require a second pass.** The obvious
implementation of {{eq:diversity-beats-depth}} — maximal marginal relevance over
pairwise embedding similarity — costs $O(k^2)$ similarity computations per query,
which is affordable at $k = 10$ and not at $k = 200$. Cheaper signals usually
suffice: cluster identity from the index itself, source document identity, section
identity within a document, or publication date bucket. Each is a coarse proxy for
"covers different material," and because the coverage objective is submodular, a
coarse diversity signal captures most of the available gain. The failure mode to
avoid is diversifying on a dimension uncorrelated with fact content — diversifying by
document length, for instance, reorders the slots without changing what they cover.

**Index-level deduplication versus selection-level.** Removing near-duplicates at
index time is cheaper but destroys information — two similar documents may differ in
the one detail that matters. Deduplicating at selection time preserves the corpus and
costs a reranking pass. {{cite:johnson2019faiss}}'s clustering structures make the
latter tractable at scale, since cluster identity is a usable diversity signal
without a second embedding pass.

## 8. Implementation

The first listing measures how fan-out width converts a per-dependency tail into a
request-level one, what per-dependency reliability would be required to compensate,
and what hedging buys instead.

```python {tier=A name=bx1}
"""Fan-out converts a good tail into a bad one, and the conversion is arithmetic.

An agent that calls five tools in parallel, or a retrieval layer that queries eight
shards, waits for the SLOWEST of them. That is not the mean latency of a dependency;
it is the maximum of a sample, and the maximum of a sample behaves very differently.

This listing measures how fan-out width turns a per-dependency tail into a
request-level one (eq:fanout-amplifies-the-tail), and what that does to the
p99 a user actually experiences.

The result is why a system composed entirely of fast, reliable components can be
slow and unreliable, with nothing in any component's dashboard looking wrong.
"""
# A dependency whose latency is mostly fast with a thin slow tail.
# (latency in seconds, probability)
DEP = [
    (0.05, 0.60),
    (0.09, 0.25),
    (0.18, 0.10),
    (0.45, 0.037),
    (1.60, 0.013),   # the tail: 1.3% of calls
]
WIDTHS = [1, 2, 3, 5, 8, 12, 20]


def percentile_of_max(width, q):
    """The q-th percentile of the MAXIMUM of `width` independent draws.

    P(max <= t) = P(single <= t)^width, so we walk the support and find the
    smallest t whose cumulative probability raised to `width` reaches q.
    """
    cum = 0.0
    for t, p in DEP:
        cum += p
        if cum ** width >= q:
            return t
    return DEP[-1][0]


def mean_of_max(width):
    """Expected value of the maximum of `width` independent draws."""
    total = 0.0
    prev = 0.0
    cum = 0.0
    for t, p in DEP:
        cum += p
        # P(max == t) = P(all <= t) - P(all <= previous t)
        total += t * (cum ** width - prev)
        prev = cum ** width
    return total


def p_slow(width, threshold):
    """P(at least one of `width` draws exceeds `threshold`)."""
    fast = sum(p for t, p in DEP if t <= threshold)
    return 1.0 - fast ** width


print("One dependency: fast most of the time, with a 1.3% tail at 1.60s.")
print()
print(f"{'latency':>10}{'probability':>14}{'cumulative':>13}")
print("-" * 37)
c = 0.0
for t, p in DEP:
    c += p
    print(f"{t:>9.2f}s{p:>14.1%}{c:>13.1%}")

single_mean = sum(t * p for t, p in DEP)
print()
print(f"mean {single_mean:.3f}s, p50 {percentile_of_max(1, 0.50):.2f}s, "
      f"p95 {percentile_of_max(1, 0.95):.2f}s, "
      f"p99 {percentile_of_max(1, 0.99):.2f}s")

print()
print()
print("Now fan out: call N of them in parallel and wait for all to return.")
print("The request's latency is the MAXIMUM, not the mean.")
print()
print(f"{'fan-out':>9}{'mean':>10}{'p50':>9}{'p95':>9}{'p99':>9}"
      f"{'P(hits tail)':>15}")
print("-" * 61)
tab = {}
for w in WIDTHS:
    m = mean_of_max(w)
    p50 = percentile_of_max(w, 0.50)
    p95 = percentile_of_max(w, 0.95)
    p99 = percentile_of_max(w, 0.99)
    pt = p_slow(w, 0.45)
    tab[w] = (m, p50, p95, p99, pt)
    print(f"{w:>9}{m:>9.3f}s{p50:>8.2f}s{p95:>8.2f}s{p99:>8.2f}s{pt:>15.1%}")

print()
print()
print("The same thing stated as amplification against a single call.")
print()
print(f"{'fan-out':>9}{'mean grows':>13}{'p95 grows':>12}"
      f"{'tail probability':>19}")
print("-" * 53)
for w in WIDTHS:
    m, p50, p95, p99, pt = tab[w]
    print(f"{w:>9}{m / single_mean:>12.2f}x{p95 / percentile_of_max(1, 0.95):>11.2f}x"
          f"{pt:>19.1%}")

print()
print()
print("What it takes to keep a 0.50s budget as fan-out grows: the per-dependency")
print("tail probability you would have to achieve.")
print()
BUDGET_P = 0.95      # we want 95% of requests inside budget
print(f"{'fan-out':>9}{'needed per-dep reliability':>29}"
      f"{'current':>11}{'gap':>10}")
print("-" * 60)
CURRENT_FAST = sum(p for t, p in DEP if t <= 0.45)
need = {}
for w in WIDTHS:
    # need fast**w >= BUDGET_P  ->  fast >= BUDGET_P**(1/w)
    r = BUDGET_P ** (1.0 / w)
    need[w] = r
    print(f"{w:>9}{r:>29.4%}{CURRENT_FAST:>11.2%}"
          f"{(r - CURRENT_FAST):>+10.2%}")

print()
print()
print("And the fix that actually works: hedging. Issue a duplicate request for any")
print("dependency still outstanding at the hedge point, and take the first answer.")
print()
HEDGE_AT = 0.18
print(f"{'fan-out':>9}{'p95 plain':>12}{'p95 hedged':>13}{'extra calls':>14}"
      f"{'improvement':>14}")
print("-" * 63)


def hedged_percentile(width, q, hedge_at):
    """With a hedge, a slow draw is replaced by hedge_at plus a fresh draw, so
    the effective per-call distribution is truncated: anything slower than the
    hedge point becomes hedge_at + (a fresh, usually fast, draw)."""
    eff = []
    slow_mass = 0.0
    for t, p in DEP:
        if t <= hedge_at:
            eff.append((t, p))
        else:
            slow_mass += p
    # The hedged retry lands at hedge_at + a draw from the same distribution.
    for t, p in DEP:
        eff.append((hedge_at + t, slow_mass * p))
    eff.sort()
    cum = 0.0
    for t, p in eff:
        cum += p
        if cum ** width >= q:
            return t
    return eff[-1][0]


for w in WIDTHS:
    plain = tab[w][2]
    hed = hedged_percentile(w, 0.95, HEDGE_AT)
    extra = w * sum(p for t, p in DEP if t > HEDGE_AT)
    print(f"{w:>9}{plain:>11.2f}s{hed:>12.2f}s{extra:>14.2f}"
          f"{(1 - hed / plain):>13.0%}")

print(f"""
A single call to this dependency has a mean of {single_mean:.3f}s and a p95 of
{percentile_of_max(1, 0.95):.2f}s. By any normal standard it is a healthy
dependency: {DEP[0][1]:.0%} of calls return in {DEP[0][0]:.2f}s and only
{DEP[-1][1]:.1%} hit the slow path.

Fan out to {WIDTHS[4]} parallel calls and the picture changes completely. The mean
becomes {tab[8][0]:.3f}s, p95 becomes {tab[8][2]:.2f}s, and the probability that at
least one call lands in the tail rises to {tab[8][4]:.1%}
(eq:fanout-amplifies-the-tail).

At fan-out {WIDTHS[-1]} the tail is hit on {tab[20][4]:.1%} of requests. **The rare
event has become the common case**, and nothing about the dependency changed. Its
own dashboard still shows a {DEP[-1][1]:.1%} slow rate, correctly, and that number
is now describing something that happens to {tab[20][4]:.0%} of user requests.

The mechanism is that P(all fast) is a PRODUCT. Each additional parallel call
multiplies in another chance to be unlucky, so per-request reliability decays
geometrically in the fan-out width -- which is ch:ag-loop's chain
(eq:loop-is-not-a-chain) with the steps running side by side instead of end to
end. Sequential or parallel, the arithmetic is the same; only the latency changes.

The reliability table is the part worth internalising, because it inverts the usual
engineering conversation. To keep {BUDGET_P:.0%} of requests inside a
{0.50:.2f}s budget at fan-out {WIDTHS[-1]}, each dependency needs to stay fast
{need[20]:.2%} of the time. It currently manages {CURRENT_FAST:.2%}.

**That gap is not closeable by optimising the dependency.** Going from
{CURRENT_FAST:.2%} to {need[20]:.2%} means cutting the tail from
{1 - CURRENT_FAST:.2%} to {1 - need[20]:.2%} -- removing four fifths of it -- and
the tail is usually where the real work is -- a cold cache, a large document, a slow
shard. So the answer to "our agent is slow" is rarely "make the tools faster".

Hedging is the lever that does work, and the table shows why. Issuing a duplicate
call for anything still outstanding at {HEDGE_AT:.2f}s cuts p95 at fan-out
{WIDTHS[4]} from {tab[8][2]:.2f}s to {hedged_percentile(8, 0.95, HEDGE_AT):.2f}s --
a {(1 - hedged_percentile(8, 0.95, HEDGE_AT) / tab[8][2]):.0%} improvement -- for
{8 * sum(p for t, p in DEP if t > HEDGE_AT):.2f} extra calls per request, which is
{sum(p for t, p in DEP if t > HEDGE_AT):.0%} more load.

That trade is the chapter's practical result: **a few percent more load buys back
most of the tail amplification**, because a hedge attacks the maximum directly
rather than trying to improve the distribution it is drawn from.

The caveat is that hedging only works when the duplicate is genuinely independent --
a second call to the same overloaded shard hedges nothing. And under ch:sd-async's
cost model, a hedge on an expensive model call is not a few percent of load, it is a
few percent of a large bill, which is why hedging is a retrieval technique far more
often than a generation one.""")
```

## 9. Practical Example

A healthy dependency: 60% of calls in 0.05s, a 1.3% tail at 1.60s, p95 of 0.18s. Fan
it out:

```
  fan-out      mean      p50      p95      p99   P(hits tail)
-------------------------------------------------------------
        1    0.108s    0.05s    0.18s    1.60s           1.3%
        2    0.157s    0.09s    0.45s    1.60s           2.6%
        3    0.199s    0.09s    0.45s    1.60s           3.8%
        5    0.271s    0.18s    1.60s    1.60s           6.3%
        8    0.360s    0.18s    1.60s    1.60s           9.9%
       12    0.458s    0.18s    1.60s    1.60s          14.5%
       20    0.615s    0.45s    1.60s    1.60s          23.0%
```

At fan-out 8 the mean is **0.360s** and the tail is hit on **9.9%** of requests. At
fan-out 20 it is **23.0%** ({{eq:fanout-amplifies-the-tail}}). The dependency has not
changed — its dashboard still correctly reports a 1.3% slow rate, now describing
something that happens to nearly a quarter of user requests.

Note p95 jumping from 0.18s to 1.60s between fan-out 3 and 5: the p95 of a 5-way
fan-out is the p99-ish percentile of one call, and that is where the tail lives.

What would optimising the dependency have to achieve?

```
  fan-out   needed per-dep reliability    current       gap
------------------------------------------------------------
        1                     95.0000%     98.70%    -3.70%
        2                     97.4679%     98.70%    -1.23%
        3                     98.3048%     98.70%    -0.40%
        5                     98.9794%     98.70%    +0.28%
        8                     99.3609%     98.70%    +0.66%
       12                     99.5735%     98.70%    +0.87%
       20                     99.7439%     98.70%    +1.04%
```

To keep 95% of requests inside a 0.50s budget at fan-out 20, each dependency must be
fast **99.74%** of the time against the **98.70%** it manages — cutting the tail from
1.30% to 0.26%. **That gap is not closeable by optimising the dependency**, because
the tail is where the real work is: a cold cache, a large document, a slow shard.

Hedging is the lever that works:

```
  fan-out   p95 plain   p95 hedged   extra calls   improvement
---------------------------------------------------------------
        8       1.60s        0.36s          0.40          78%
```

A duplicate call for anything outstanding at 0.18s cuts p95 at fan-out 8 from
**1.60s** to **0.36s** — a **78%** improvement — for **0.40** extra calls per
request, or 5% more load ({{eq:hedging-beats-optimising-dependencies}}).

```mermaid {#fig:hedge caption="Fan-out raises the per-call CDF to the n-th power, so a request is fast only if every call is. Hedging tightens the CDF being raised, which is why a few percent more load buys back most of the amplification."}
flowchart LR
  A["per-call tail<br/>1.3%"] --> B["fan-out to n"]
  B --> C["request tail<br/>23% at n=20"]
  A --> D["hedge at 0.18s"]
  D --> E["effective tail<br/>much tighter"]
  E --> F["fan-out to n"]
  F --> G["p95 0.36s<br/>at 5% more load"]
```

The second listing turns to retrieval at scale.

```python {tier=A name=bx2}
"""At scale, retrieval gets better by its own metrics and worse at its job.

A question needs several distinct facts to answer. A corpus contains documents, and
as the corpus grows the popular facts get restated many times while the rare ones
stay rare. Similarity search ranks by resemblance to the query, and resemblance
tracks popularity.

So a bigger corpus makes the top-k MORE redundant: the same well-covered facts,
restated. Precision rises, because there are more relevant documents and every slot
fills with genuinely on-topic material. Fact coverage falls, because the slots are
spent on repetition (eq:scale-buys-redundancy-not-coverage).

This is ch:sd-architecture's pattern again -- a metric that is accurate about its own
quantity and silent about the one that decides whether the answer is right.
"""
import math

F = 24                 # distinct facts a full answer needs
FACTS_PER_DOC = 3.0    # facts carried by a typical retrieved document
N0 = 1000              # reference corpus size
CORPORA = [1000, 10000, 100000, 1000000, 10000000]
KS = [3, 5, 10, 20, 40]

# Fact prevalence: Zipf. Fact 1 is discussed everywhere, fact 24 almost nowhere.
PREV = [1.0 / (i + 1) for i in range(F)]
_z = sum(PREV)
PREV = [p / _z for p in PREV]


def selectivity(n):
    """How sharply retrieval concentrates on popular facts, by corpus size.

    A larger corpus gives similarity search more near-duplicates to choose from,
    so the top-k skews further toward whatever is most commonly discussed.
    """
    return 1.0 + 0.42 * math.log10(float(n) / N0)


def per_doc_probs(n):
    """P(a retrieved document carries fact f), for each f."""
    s = selectivity(n)
    w = [p ** s for p in PREV]
    tot = sum(w)
    return [FACTS_PER_DOC * x / tot for x in w]


def coverage(n, k):
    """Expected share of the F needed facts present anywhere in the top-k."""
    q = per_doc_probs(n)
    got = 0.0
    for qi in q:
        qi = min(qi, 1.0)
        got += 1.0 - (1.0 - qi) ** k
    return got / F


def precision_at_k(n, k, relevant_rate=0.0004):
    """Share of the top-k that are topically relevant -- the metric a retrieval
    evaluation usually reports, and the one that improves with scale.

    A larger corpus contains more relevant documents, so the top-k fills up with
    genuinely on-topic material instead of padding.
    """
    relevant = n * relevant_rate
    return min(1.0, relevant / k)


print("A question needing %d distinct facts. Fact prevalence is Zipf: the most" % F)
print("discussed fact appears %.0fx more often than the least."
      % (PREV[0] / PREV[-1]))
print()
print(f"{'corpus size':>14}{'selectivity':>14}{'P(top doc has fact 1)':>24}"
      f"{'P(has fact 24)':>17}")
print("-" * 69)
sel = {}
for n in CORPORA:
    q = per_doc_probs(n)
    sel[n] = (selectivity(n), q[0], q[-1])
    print(f"{n:>14,}{selectivity(n):>14.2f}{min(q[0], 1.0):>24.3f}"
          f"{q[-1]:>17.5f}")

print()
print()
print("Fact coverage at k=10, as the corpus grows. The retriever is unchanged;")
print("only the corpus is bigger.")
print()
K = 10
print(f"{'corpus size':>14}{'fact coverage':>16}{'precision@10':>15}"
      f"{'what a report says':>21}")
print("-" * 69)
main = {}
for n in CORPORA:
    cov = coverage(n, K)
    rec = precision_at_k(n, K)
    main[n] = (cov, rec)
    verdict = "improving" if rec >= main[CORPORA[0]][1] else "degrading"
    print(f"{n:>14,}{cov:>16.1%}{rec:>15.1%}{verdict:>21}")

print()
print()
print("The same across retrieval depth. More slots help, but they are spent on")
print("increasingly redundant documents.")
print()
print(f"{'corpus size':>14}" + "".join(f"{('k=%d' % k):>10}" for k in KS))
print("-" * 64)
grid = {}
for n in CORPORA:
    row = [coverage(n, k) for k in KS]
    grid[n] = row
    print(f"{n:>14,}" + "".join(f"{v:>10.1%}" for v in row))

print()
print()
print("How much retrieval depth it takes to hold coverage flat as the corpus")
print("grows -- and whether a context budget can pay for it.")
print()
TARGET = grid[CORPORA[0]][2]      # coverage achieved at k=10 on the small corpus
print(f"holding coverage at {TARGET:.1%}, the level a 1,000-document corpus")
print("reaches with k=10:")
print()
print(f"{'corpus size':>14}{'k needed':>11}{'vs k=10':>10}"
      f"{'context tokens':>17}{'feasible':>11}")
print("-" * 64)
TOK_PER_DOC = 420
BUDGET_TOK = 12000
need = {}
for n in CORPORA:
    kk = None
    for k in range(1, 4001):
        if coverage(n, k) >= TARGET:
            kk = k
            break
    need[n] = kk
    if kk is None:
        print(f"{n:>14,}{'never':>11}{'--':>10}{'--':>17}{'no':>11}")
    else:
        tok = kk * TOK_PER_DOC
        print(f"{n:>14,}{kk:>11}{kk / 10.0:>9.1f}x{tok:>17,}"
              f"{('yes' if tok <= BUDGET_TOK else 'no'):>11}")

print()
print()
print("And the alternative: deduplicate by fact before filling the context.")
print("Same slots, spent on distinct content instead of the top of the ranking.")
print()
print(f"{'corpus size':>14}{'plain k=10':>13}{'deduped k=10':>15}"
      f"{'gain':>9}{'equivalent plain k':>21}")
print("-" * 72)


def deduped_coverage(n, k):
    """Fill k slots, skipping a document whose facts are already covered.

    Modelled by drawing from the residual: each slot targets the highest-
    prevalence fact not yet covered, which is what a diversity reranker does.
    """
    q = per_doc_probs(n)
    covered = [0.0] * F
    for _ in range(k):
        # The slot goes to the document most likely to add something new.
        gains = [min(q[i], 1.0) * (1.0 - covered[i]) for i in range(F)]
        j = max(range(F), key=lambda i: gains[i])
        # That document carries its target fact plus incidental others.
        for i in range(F):
            add = min(q[i], 1.0) if i != j else 1.0
            covered[i] = covered[i] + (1.0 - covered[i]) * add
    return sum(covered) / F


ded = {}
for n in CORPORA:
    plain = coverage(n, 10)
    dd = deduped_coverage(n, 10)
    eq = None
    for k in range(1, 4001):
        if coverage(n, k) >= dd:
            eq = k
            break
    ded[n] = (plain, dd, eq)
    print(f"{n:>14,}{plain:>13.1%}{dd:>15.1%}{dd - plain:>+9.1%}"
          f"{(str(eq) if eq else 'never'):>21}")

print(f"""
The selectivity table is the mechanism. At {CORPORA[0]:,} documents a retrieved
document carries the rarest fact with probability {sel[CORPORA[0]][2]:.5f}. At
{CORPORA[-1]:,} documents that has fallen to {sel[CORPORA[-1]][2]:.5f}, because a
larger corpus offers similarity search more near-duplicates of the popular material
and it takes them.

Nothing about the retriever changed. The corpus got bigger, which every intuition
says should help, and the rare facts got harder to reach.

The consequence is the second table, and it is the finding. At k={K}, growing the
corpus from {CORPORA[0]:,} to {CORPORA[-1]:,} documents moves fact coverage from
{main[CORPORA[0]][0]:.1%} to {main[CORPORA[-1]][0]:.1%} -- **down
{(main[CORPORA[0]][0] - main[CORPORA[-1]][0]) * 100:.0f} points** -- while
precision@10 rises from {main[CORPORA[0]][1]:.0%} to {main[CORPORA[-1]][1]:.0%}
(eq:scale-buys-redundancy-not-coverage).

**The two metrics move in opposite directions on the same system.** A retrieval
evaluation reporting precision@10 sees a system improving from
{main[CORPORA[0]][1]:.0%} to {main[CORPORA[-1]][1]:.0%} as the corpus grows, which
reads as a clear success, and every document it returns really is on topic. The
answers built from those documents are getting worse, because the top-k has become
{1 - main[CORPORA[-1]][0]:.0%} redundant.

The depth grid shows why buying more slots does not rescue it. On the
{CORPORA[-1]:,}-document corpus, going from k={KS[0]} to k={KS[-1]} -- more than a
tenfold increase in retrieved context -- moves coverage from
{grid[CORPORA[-1]][0]:.1%} to {grid[CORPORA[-1]][-1]:.1%}. The extra slots are
filled with more copies of what was already there.

The feasibility table prices that directly. Holding coverage at {TARGET:.1%} --
the level the small corpus reaches at k=10 -- needs k={need[CORPORA[1]]} at
{CORPORA[1]:,} documents and k={need[CORPORA[2]]} at {CORPORA[2]:,}. At
{CORPORA[2]:,} documents that is {need[CORPORA[2]] * TOK_PER_DOC:,} tokens of
context, against a {BUDGET_TOK:,}-token budget.

**Scale defeats retrieval depth well before it defeats the context window**, and
ch:mcp-schemas's context budget is the binding constraint.

The last table is the lever that works. Spending the same ten slots on distinct
content rather than on the top of the ranking moves coverage at {CORPORA[-1]:,}
documents from {ded[CORPORA[-1]][0]:.1%} to {ded[CORPORA[-1]][1]:.1%} -- a gain of
{(ded[CORPORA[-1]][1] - ded[CORPORA[-1]][0]) * 100:.0f} points from **reordering,
not from retrieving more**.

That is the architectural conclusion, and it is a specific one. As a corpus grows,
the marginal return on a better retriever and on a larger context window both fall,
while the marginal return on **diversity-aware selection** rises. The system's
bottleneck migrates from finding relevant documents to choosing which relevant
documents to spend the budget on -- and those are different problems with different
owners, different metrics, and usually different teams.""")
```

Growing the corpus with the retriever unchanged:

```
   corpus size   fact coverage   precision@10   what a report says
---------------------------------------------------------------------
         1,000           55.5%            4.0%            improving
        10,000           42.0%           40.0%            improving
       100,000           30.6%          100.0%            improving
     1,000,000           22.5%          100.0%            improving
    10,000,000           17.2%          100.0%            improving
```

Fact coverage falls from **55.5%** to **17.2%** — down 38 points — while precision@10
rises from **4%** to **100%** ({{eq:scale-buys-redundancy-not-coverage}}). **The two
metrics move in opposite directions on the same system.** Every returned document is
genuinely on topic; the top-k has become 83% redundant.

Buying more slots does not rescue it:

```
   corpus size       k=3       k=5      k=10      k=20      k=40
----------------------------------------------------------------
         1,000     27.4%     37.8%     55.5%     74.8%     90.6%
        10,000     21.7%     29.0%     42.0%     58.1%     75.2%
       100,000     16.7%     21.7%     30.6%     42.1%     56.2%
     1,000,000     13.1%     16.5%     22.5%     30.3%     40.3%
    10,000,000     10.6%     13.1%     17.2%     22.6%     29.4%
```

On the 10-million-document corpus, a more-than-tenfold increase in retrieved context
(k=3 to k=40) moves coverage from **10.6%** to **29.4%**. Holding coverage at the
small corpus's 55.5% needs k=39 at 100,000 documents — 16,380 tokens against a
12,000-token budget. **Scale defeats retrieval depth before it defeats the context
window.**

Reordering the same ten slots for diversity:

```
   corpus size   plain k=10   deduped k=10     gain   equivalent plain k
------------------------------------------------------------------------
         1,000        55.5%          64.1%    +8.6%                   14
        10,000        42.0%          54.1%   +12.1%                   18
       100,000        30.6%          47.4%   +16.8%                   27
     1,000,000        22.5%          43.9%   +21.5%                   51
    10,000,000        17.2%          42.5%   +25.3%                  111
```

The gain grows with corpus size — **+8.6** points at 1,000 documents, **+25.3** at 10
million, where ten reordered slots match what plain depth needs **111** slots to
achieve ({{eq:diversity-beats-depth}}). That is an eleven-fold effective retrieval
depth from reordering alone, retrieving nothing additional.

## 10. Production Considerations

Measure fan-out width as a first-class metric. It is the exponent in
{{eq:fanout-amplifies-the-tail}} and most systems do not record it, which makes tail
regressions inexplicable when a feature quietly adds a sixth parallel tool call.

Hedge retrieval, not generation. A hedge costs a duplicate call; under
{{ch:sd-architecture}}'s expense property, duplicating a retrieval call is a rounding
error and duplicating a generation call is not.

Set the hedge point at roughly the p90 of the dependency, not at a fixed timeout.
Hedging at the p90 costs 10% extra load by construction and captures most of the tail;
hedging at the p50 doubles load for little additional benefit.

Ensure hedges reach independent replicas. A hedge to the same shard is a no-op that
looks like a mitigation, which is worse than no mitigation because it closes the
investigation.

Put a cap on total hedge budget, expressed as a share of baseline load, and shed
hedges rather than requests when it is exhausted. Without a cap, hedging is a
positive feedback loop: the system slows, more calls cross the hedge point, load
rises, the system slows further. The cap converts a potential collapse into graceful
degradation of tail latency, which is the failure you want.

Instrument which slot each retrieved document occupied and whether it contributed to
the answer. This is cheap — it is a join between the retrieval log and whatever
attribution the generation step already produces — and it is the fastest empirical
route to noticing that slots four through ten are contributing nothing, which is
what {{eq:scale-buys-redundancy-not-coverage}} predicts and what most teams have
never checked.

Report retrieval coverage, not only precision and recall. Coverage requires knowing
what facts an answer needed, which means a labelled evaluation set — expensive, and
the only thing that would have caught the degradation in
{{eq:scale-buys-redundancy-not-coverage}}.

Deploy diversity reranking before increasing $k$. It is cheaper, it works better as
the corpus grows, and it does not consume context budget.

Re-evaluate the retrieval configuration after corpus growth, not only after retriever
changes. The corpus is an input to {{eq:scale-buys-redundancy-not-coverage}} and it
changes continuously without anyone deploying anything.

## 11. Common Mistakes

**Optimising a dependency to fix a fan-out problem.** The required reliability rises
faster than any dependency can be improved.

**Assuming parallelism helps reliability.** It helps latency; reliability is a
product either way.

**Hedging to the same replica.** Draws from a conditioned distribution and buys
nothing.

**Reading precision@k as retrieval health.** It rises as coverage falls.

**Adding retrieval depth to fix redundancy.** The extra slots fill with more of the
same.

**Deduplicating at index time to save reranking cost.** Destroys the distinctions
that made near-duplicates worth keeping.

## 12. Failure Modes

**Silent fan-out growth.** A feature adds one more parallel call; p99 moves and no
component's dashboard changes.

**Correlated tail collapse.** Parallel calls share a rate limit or a shard and all
slow together, defeating both the independence assumption and the hedge.

**Hedge amplification under load.** Hedges fire more often as the system slows,
adding load exactly when it is least available — a positive feedback loop that
requires a hedge budget cap to break.

**Coverage decay by corpus growth.** Answers degrade continuously as the corpus
grows, with every retrieval metric improving.

**Diversity reranking that diversifies noise.** A reranker maximising dissimilarity
rather than marginal relevance fills slots with off-topic material and looks like it
is working by every diversity metric.

## 13. Alternatives

**Sequential rather than parallel calls.** Removes the max-of-sample problem and
replaces it with a sum, which is worse for latency and identical for reliability.
Occasionally right when the calls are dependent anyway.

**Partial results with a deadline.** Return what arrived by time $t$ and proceed
without the rest. Converts a latency failure into a coverage failure, which is the
right trade when the missing dependency was marginal — and requires the answer path
to tolerate incompleteness, which is a design decision made early or not at all.

**Larger context instead of better selection.** Buys coverage linearly against a
problem that degrades geometrically; {{eq:diversity-beats-depth}} shows reordering
dominates as the corpus grows.

**Hierarchical retrieval.** Retrieve summaries, then expand selectively. Effectively
increases $k$ per token and attacks {{eq:context-is-a-budget}} directly.

**Query decomposition.** Issue several targeted retrievals rather than one broad one,
so each sub-query's top-k covers a different fact. Trades a fan-out problem for a
coverage one, and the fan-out cost is now known to be geometric.

## 14. Evaluation

Report tail latency at the request level and at the dependency level, together. The
gap between them is the fan-out amplification and it is the number that explains
otherwise inexplicable p99s.

Measure hedge effectiveness as p95 improvement per percent of extra load. A hedge
that improves p95 by 78% for 5% more load is excellent; the same improvement for 100%
more load is a capacity decision in disguise.

Build a coverage evaluation set — questions with enumerated required facts — even a
small one. Precision and recall cannot detect
{{eq:scale-buys-redundancy-not-coverage}} and nothing else will.

Track corpus size as an evaluation variable. A retrieval configuration validated at
one corpus size is not validated at another, and the direction of the error is
predictable.

Compare diversity reranking against increased $k$ on equal context budget, not on
equal document count. The budget is the constraint.

## 15. Advanced Concepts

The independence assumption in {{eq:fanout-amplifies-the-tail}} is the model's
weakest point. Real parallel calls share infrastructure, so their latencies are
positively correlated. Correlation *reduces* the max relative to independence — if
calls are slow together, the max is not much worse than a single call — so
{{eq:fanout-amplifies-the-tail}} is conservative for latency. It is optimistic for
reliability, since correlated failure means the fan-out fails entirely rather than
partially.

The Zipf assumption in {{eq:scale-buys-redundancy-not-coverage}} matters more than it
looks. Coverage decay depends on the tail index: a heavier-tailed prevalence
distribution decays faster, and a corpus curated to remove near-duplicates has a
lighter one. So **corpus curation is a retrieval intervention**, and one that acts on
the exponent rather than on the retriever.

The selectivity function $s(N)$ is the least principled part of the model. It encodes
the claim that a larger corpus lets similarity search concentrate harder on popular
material, which is empirically plausible and, as far as the author is aware,
unmeasured at scale. {{sec:19-research-questions}} takes this up.

A subtlety in the coverage model is that it treats all $F$ facts as equally
necessary. Real questions have a core the answer fails without and a periphery that
merely enriches it, and the rare facts are not uniformly distributed between those
two categories. If rarity correlates with peripherality — if the obscure fact is
usually the unimportant one — then coverage decay matters far less than the numbers
suggest. If rarity correlates with *importance*, which is the case for questions whose
whole point is to surface something not widely known, then the decay is worse than
modelled, because the facts being lost are precisely the ones the question was asked
to obtain. Which regime a product is in is an empirical question with a large effect
on how much any of this matters, and it is answerable from a coverage evaluation set
at modest cost.

Finally, {{eq:diversity-beats-depth}} and {{eq:hedging-beats-optimising-dependencies}}
have the same structure: in both, the intervention that works acts on the *shape* of
a distribution rather than on its mean, and in both, the intervention that fails is
the one an ordinary optimisation instinct suggests first.

## 16. Connection to Previous Chapters

{{eq:variance-not-mean-drives-wait}} from {{ch:sd-async}} and
{{eq:fanout-amplifies-the-tail}} are the same insight in two topologies: distributional
shape, not central tendency, determines what a user experiences.

{{eq:loop-is-not-a-chain}} from {{ch:ag-loop}} is {{eq:fanout-amplifies-the-tail}}
with the steps side by side rather than end to end. The arithmetic is identical.

{{eq:context-is-a-budget}} from {{ch:mcp-schemas}} is why retrieval depth cannot solve
the coverage problem.

{{eq:semantic-failure-has-no-instrument}} predicted metrics that are accurate and
irrelevant; precision@k is the fourth example this part has produced.

## 17. Exercises

1. For the listing's dependency, compute the fan-out width at which p50 — not p95 —
   first reaches 0.45s. Why does the median move so much later than the tail?

2. Derive the hedge point that minimises p95 subject to at most 10% extra load.

3. Modify the first listing so calls are correlated with coefficient $\rho$. At what
   $\rho$ does hedging stop helping?

4. Compute the corpus size at which diversity reranking on 10 slots beats plain
   retrieval on 50, for the listing's parameters.

5. Design a coverage evaluation set for a domain you know: twenty questions with
   enumerated required facts. How long did it take, and what would it cost to
   maintain?

## 18. Interview Questions

1. Every tool in our agent has a p95 under 200ms, and the agent's p95 is 1.6 seconds.
   Explain.

2. Why does optimising the slowest dependency not fix a fan-out tail?

3. When is hedging free, and when is it unaffordable?

4. Our retrieval precision improved from 40% to 100% after a corpus expansion. Is
   that good news?

5. You have ten context slots and a ten-million-document corpus. Argue for spending
   engineering effort on reranking rather than on a better embedding model.

## 19. Research Questions

1. How does retrieval selectivity actually scale with corpus size on real corpora?
   The $s(N)$ in {{eq:scale-buys-redundancy-not-coverage}} is plausible and
   unmeasured.

2. Can fact coverage be estimated without a labelled evaluation set — for instance
   from the diversity of the retrieved set under some intrinsic measure?

3. What is the right hedging policy under correlated dependencies, where a hedge may
   land on the same congested resource?

4. Does corpus curation for near-duplicate removal improve coverage more than
   diversity reranking does, and at what corpus size does the ordering flip?

## 20. Chapter Summary

Fan-out raises the per-call CDF to the $n$-th power
({{eq:fanout-amplifies-the-tail}}), so a request is fast only if every call is. A
dependency with a **1.3%** tail produces a **23.0%** request-level tail at fan-out 20,
with the dependency's own dashboard correctly unchanged.

Optimising the dependency cannot fix this: fan-out 20 under a 95% budget demands
**99.74%** per-call reliability against **98.70%** achieved. Hedging can — a duplicate
at 0.18s cuts p95 at fan-out 8 from **1.60s** to **0.36s**, a **78%** improvement for
**5%** more load ({{eq:hedging-beats-optimising-dependencies}}).

At scale, retrieval gets more redundant rather than more complete. Growing a corpus
from 1,000 to 10 million documents drives fact coverage from **55.5%** to **17.2%**
while precision@10 rises from **4%** to **100%**
({{eq:scale-buys-redundancy-not-coverage}}).

Depth cannot pay for it — holding coverage flat needs 16,380 tokens against a 12,000
budget at only 100,000 documents. Diversity reranking can: the same ten slots gain
**+25.3** points at 10 million documents, matching what plain depth needs **111**
slots to reach ({{eq:diversity-beats-depth}}).

Both results share a structure worth naming, because it recurs well beyond this
chapter: the intervention that works acts on the shape of a distribution, and the
intervention that fails is the one an ordinary optimisation instinct reaches for
first. Fan-out is not fixed by faster dependencies and redundancy is not fixed by
more retrieval, and in both cases the reason is that the quantity being optimised
was never the one doing the damage.

Carry forward: **hedge the tail rather than optimising the dependency**, and
**reorder the context rather than enlarging it**.

## 21. Further Reading

- {{cite:malkov2020hnsw}} — hierarchical navigable small world graphs; the bimodal
  query latency behind the fan-out tail.
- {{cite:johnson2019faiss}} — billion-scale similarity search; cluster structure as a
  cheap diversity signal.
- {{cite:cemri2025mast}} — multi-agent failure taxonomy; why parallel agent calls fail
  together.
- {{cite:qin2023toolllm}} — large tool collections, where the same redundancy problem
  appears in tool selection.

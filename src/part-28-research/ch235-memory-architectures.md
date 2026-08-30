---
id: res-memory
number: 235
part: XXVIII
tier: full
status: draft
requires: [kv-traffic-overtakes-weights, batch-times-context-is-the-budget,
           retrieval-gains-are-capped-by-utilisation, coverage-is-a-union-not-a-sum]
provides: [effective-context-is-shorter-than-nominal,
           multi-fact-accuracy-is-a-product-over-positions,
           memory-is-compression-times-retrieval,
           no-architecture-dominates-across-horizons]
citations: [liu2023lost, li2024ragvslongcontext, sarthi2024raptor, press2022alibi]
---

## 1. Learning Objectives

By the end of this chapter you will be able to distinguish a model's nominal context length from
its effective one and compute the second from a position-sweep; show why multi-fact tasks
collapse as a product of per-fact recall; price a context window per *solved task* rather than
per token; decompose any memory architecture into a compression ratio and a retrieval accuracy;
and show that the best architecture depends on the horizon, with none dominating across all of
them.

## 2. Why This Matters

{{ch:inf-gpu-memory}} priced the window: bytes per token, cache against weights, concurrency.
This chapter asks the other question — whether the tokens you paid to hold are consulted.

They are not, uniformly. A fact at the end of a 128,000-token window is used with probability
**0.956**; the same fact in the middle, **0.674** — a factor of **1.4** for identical content
({{cite:liu2023lost}}). Mean recall falls from **0.903** at 4,000 tokens to **0.740** at
512,000, so effective tokens grow more slowly than nominal ones
({{eq:effective-context-is-shorter-than-nominal}}).

Then the collapse. Real questions need several facts, and per-fact recall multiplies: at 128,000
tokens one fact succeeds at **0.775** and five at **0.279**
({{eq:multi-fact-accuracy-is-a-product-over-positions}}). Per solved five-fact task the largest
window costs **69×** the smallest.

And the alternatives are all the same object. A memory system is a compression ratio times a
retrieval accuracy ({{eq:memory-is-compression-times-retrieval}}): a KV cache at **131,072**
bytes per token against a vector index at **10.2** — **12,800×** — and the index's recall does
not decay with age while the window's does. Across five horizons **2** architectures top the
recall column, and at **2 of 5** no single architecture reaches 0.90 at any price
({{eq:no-architecture-dominates-across-horizons}}).

## 3. Prerequisites

{{eq:kv-traffic-overtakes-weights}} from {{ch:inf-gpu-memory}} is why context is expensive at
all; this chapter takes that cost as given and asks what it buys.

{{eq:batch-times-context-is-the-budget}} from the same chapter is the constraint that makes the
80 GB feasibility column in {{sec:9-practical-example}} bite.

{{eq:retrieval-gains-are-capped-by-utilisation}} from {{ch:ev-rag}} is the result this chapter
reads from the other end: retrieval does not compete with long context, it is how long context
is made to work.

{{eq:coverage-is-a-union-not-a-sum}} from {{ch:ev-framework}} is why memory architectures
compose the way they do, and why a window plus an index beats either alone.

## 4. Intuitive Explanation

Context windows grew by three orders of magnitude in a few years, and the marketing number is
the one everyone quotes. This chapter is about the second number, which almost nobody publishes.

Start with a clean experiment. Put one fact somewhere in a long context, ask a question that
requires it, and vary where the fact sits.

At 128,000 tokens: a fact at the start is used with probability **0.966**, at the end
**0.956**, and in the middle **0.674** ({{cite:liu2023lost}}). The window held all three. The
model consulted the edges.

That gap widens with length. At 4,000 tokens the middle scores 0.862; at 512,000 it is
substantially worse. Something about attending over a long sequence degrades the interior
specifically, and it does so gradually, so there is no length at which the window visibly
"breaks".

Average across positions and you get the number that should sit beside every context-length
claim. Mean recall: **0.903** at 4,000 tokens, 0.863 at 16,000, 0.833 at 32,000, 0.802 at
64,000, **0.774** at 128,000, **0.740** at 512,000.

Multiply through and you have *effective* tokens. Going from 32,000 to 512,000 — sixteen times
the window — buys **14.2×** the effective content, and the KV cost per useful token rises
**1.1×** ({{eq:effective-context-is-shorter-than-nominal}}).

That is a mild penalty and it would not be worth a chapter on its own. The window is not a lie;
it is priced in a unit that flatters it slightly.

Now the part that is worth a chapter.

Single-needle retrieval is not a task anyone has. Real questions need several facts at once —
compare these three contracts, reconcile these two reports, find every mention of this issue and
summarise. And if per-fact recall is roughly independent across positions, task success is a
**product**.

At 128,000 tokens: one fact, **0.775**. Two facts, 0.600. Three, 0.465. Five, **0.279**. Eight,
**0.130**.

A factor of **2.8** between one fact and five, and **6.0** between one and eight. The model that
"handles 128k context" solves the eight-fact version of the same task about one time in eight.

This is the conjunction this book keeps finding. {{ch:ops-versioning}} found it in
reproducibility, {{ch:rai-privacy}} in deletion, {{ch:rai-oversight}} in oversight
preconditions. **A product of things that mostly work is a thing that mostly does not**
({{eq:multi-fact-accuracy-is-a-product-over-positions}}), and long-context benchmarks that
report single-needle retrieval are measuring the one term where the product is still healthy.

Price it per solved task and the decision inverts. Holding the same five facts and padding out
to each window length: 16,000 tokens costs **0.000291** per solved five-fact task; 64,000 costs
0.001681; 128,000 costs 0.004009; 512,000 costs **0.020115** — **69×** the cheapest.

Note exactly what that compares: the same content in bigger windows. Under that comparison the
curve is monotone. **The cheapest window is the shortest one that holds the content**, and every
token of padding is paid twice — once in cache, once in the recall it costs the facts that
matter.

That is a narrow claim and it is the one most often violated, because filling the window is free
at the API and expensive everywhere else.

So what recovers the middle? Four things, ranked by success per unit cost.

`retrieve, then use a short window` wins outright: **0.06** of the long-window cost for
**2.55×** its success — **11.9** success per unit cost. That is {{ch:ev-rag}}'s
`retrieval-gains-are-capped-by-utilisation` read backwards. **Retrieval is not competing with
long context; it is how you make long context work** — and {{cite:li2024ragvslongcontext}}'s
comparison is a question about which regime you are in rather than which technique is better.

`put the relevant span last` is the best free move: five-fact success from **0.279** to
**0.644** at no extra cost. It also requires knowing which span is relevant, which is the same
retrieval problem in different clothes.

`hierarchical summary index` ({{cite:sarthi2024raptor}}) gives 0.564 at 0.11 of the cost.

And `longer window, same content` — four times the cost for **0.198** success against a baseline
of 0.279 — is the row that should end the discussion. **Adding window without adding relevance
makes things actively worse**, because every irrelevant token pushes the relevant ones further
from the edges.

Which raises the design question: if the window is not the answer, what is?

Here is the unifying observation. Every memory architecture stores history at some *bytes per
remembered token* and answers a query about a fact of some *age* with some probability. Useful
memory is the product ({{eq:memory-is-compression-times-retrieval}}), and every design in
circulation trades one against the other.

The storage side first. A KV cache holds a token in **131,072** bytes. A vector index holds it
in **10.2** — a compression of **12,800×**. A hierarchical summary index, **1.7** bytes —
**76,800×**. Model-written notes, **2.2** bytes, because they discard almost everything on
purpose. A fixed recurrent state stores nothing per token at all; its size does not grow.

Compression is not free, and the recall table is the bill.

`sliding window, 32k` knows **0.954** of what happened a thousand tokens ago, 0.751 at ten
thousand, and **exactly nothing** past its boundary. A cliff, not a decay — which is the honest
description of truncation and the reason it feels so abrupt in use.

`fixed recurrent state` starts at 0.915 and decays smoothly to **0.050**. Gentler-looking, same
destination.

`vector retrieval` starts lower at **0.810** and **stays there** at every age, because an index
does not care how old a chunk is.

That is the design space in three rows. **Recency-biased architectures are accurate and
forgetful; index-based ones are less accurate and do not forget.**

Now add a memory budget, because it changes the menu. Holding 80 GB for the whole history, a
full uncompressed window over a million tokens does not fit — 131 GB — so at the two longest
horizons only **4 of 6** architectures are even available.

With that constraint, ask three questions per horizon: what has the best recall, what is the
cheapest thing reaching 0.75, and what is the cheapest thing reaching 0.90.

At 1,000 tokens of age: full window at 0.970; `fixed recurrent state` is cheapest at both
thresholds. At 10,000: full window still best; recurrent state clears 0.75, only the full window
clears 0.90. At 100,000: full window best; vector retrieval is the cheapest at 0.75. At
1,000,000 and 10,000,000: the full window is infeasible, vector retrieval is the best available
at **0.810** — and **nothing reaches 0.90 at any price.**

**Two of five horizons have no single-architecture answer at 0.90**
({{eq:no-architecture-dominates-across-horizons}}), and the cheapest design meeting a fixed
recall target changes three times across the range.

That should change how the argument is conducted. "Long context versus retrieval" is not a
question with an answer; it is a question missing its parameter. At a thousand tokens of age the
window wins on both columns. At ten million the window scores zero and the index is the only row
still standing. **The dispute is only ever about which horizon the product actually has**, and
that is measurable from a query log in an afternoon.

Which brings the chapter to what production systems actually converge on, and to a reason for it
that is structural rather than fashionable.

Two memory systems answering the same query fail for different reasons, so their recall composes
as a **union** — {{eq:coverage-is-a-union-not-a-sum}} from {{ch:ev-framework}} in a third
setting.

`vector retrieval + hierarchical summaries` reaches **0.939** mean recall against the best single
architecture's **0.857**. `fixed recurrent state + vector retrieval` is cheapest per recalled
fact at **0.000091**. `sliding window, 32k + vector retrieval` reaches 0.875 — the classic
production shape, and it beats either component.

**A window plus an index is not a compromise between two camps; it is the union of two coverage
sets**, and it wins for the same reason a portfolio of evaluations beats the best single
evaluation.

The practical reading is narrow and useful. Size the window to the horizon where recency
genuinely matters — shorter than you think — and put everything older behind an index whose
recall does not decay. Then measure the union, because that is what the user experiences.

## 5. Formal Explanation

**Effective context.** For a window of length $n$ with position-dependent recall $r(p, n)$ over
relative position $p \in [0,1]$, effective length is
$n_{\text{eff}} = n \int_0^1 r(p,n)\,dp$. Since $r$ decreases in $n$ for interior positions,
$n_{\text{eff}}$ is sublinear in $n$, and cost per effective token
$\propto n/n_{\text{eff}} = 1/\bar r(n)$ increases monotonically.

**Multi-fact success.** With $k$ facts at independent uniform positions, task success is
$\bar r(n)^k$. Taking logs, $\log P = k \log \bar r(n)$: the penalty is *linear in the number of
facts* and the base is already below one. Per solved task, cost is
$C(n)/\bar r(n)^k$, which for a fixed fact set is increasing in $n$ whenever $C$ is increasing
and $\bar r$ decreasing — hence the monotone table in {{sec:9-practical-example}}.

**Memory as a product.** Write a memory system as a pair: bytes per remembered token $b$, and
recall $\rho(a)$ for a fact of age $a$. Storage for history $H$ is $bH$; useful recalled content
is $H\rho$. Cost per recalled fact is $(bH \cdot \text{price} + q)/\rho$, and architectures
occupy a frontier in $(b, \rho)$ rather than a ranking.

**Union composition.** Two systems with independent failures give
$\rho_{\cup}(a) = 1 - (1-\rho_1(a))(1-\rho_2(a))$, which exceeds both components whenever both
are strictly between 0 and 1. Independence is the assumption doing the work, and correlated
failures — both systems missing the same badly-written passage — reduce the gain.

## 6. Mathematical Foundation

The window you buy and the window you get:

$$n_{\text{eff}} = n\int_0^1 r(p,n)\,dp, \qquad \bar r = 0.903 \ (4\text{k}) \to 0.740 \ (512\text{k})$$ (eq:effective-context-is-shorter-than-nominal)

Sixteen times the nominal window buys **14.2×** the effective content.

Tasks need conjunctions:

$$P(\text{task}) = \bar r(n)^k = 0.775 \ (k=1) \to 0.279 \ (k=5) \to 0.130 \ (k=8)$$ (eq:multi-fact-accuracy-is-a-product-over-positions)

and per solved five-fact task the largest window costs **69×** the smallest.

Every memory architecture is two numbers:

$$\text{useful memory} = \underbrace{(b\ \text{bytes/token})^{-1}}_{\text{compression}} \times \underbrace{\rho(a)}_{\text{recall at age}}, \qquad \frac{131{,}072}{10.2} = 12{,}800\times$$ (eq:memory-is-compression-times-retrieval)

And the ranking depends on the horizon:

$$\arg\max_{\text{arch}} \rho(a) \ \text{changes with } a; \quad \text{no single arch reaches } 0.90 \text{ at } 2 \text{ of } 5 \text{ horizons}$$ (eq:no-architecture-dominates-across-horizons)

## 7. Internal Mechanics

Why is the middle worse? Two mechanisms are usually offered and the chapter does not need to
choose between them. Positional encodings extrapolate unevenly — {{cite:press2022alibi}}'s
approach exists precisely because naive schemes degrade outside their training range — and
attention mass is finite, so a long sequence divides a fixed budget among more candidates while
recency and primacy biases in the training distribution concentrate it at the ends. Both predict
the observed U shape and both predict that it deepens with length, which is what the position
sweep shows.

The multi-fact collapse has a mechanism worth stating carefully, because the independence
assumption is doing real work. If the facts are scattered independently, the product model is
right. If they cluster — all in one document, all in one region — recall correlates and the
product overstates the collapse. That correlation is *good news* and it points at a design: put
the facts a task needs near each other and near an edge, which is what a retrieval step
accomplishes and what a raw long window does not.

The compression/recall framing explains something otherwise puzzling about the field's
trajectory. Every few months a new memory architecture is announced with a compression ratio and
a benchmark; the benchmark almost always uses a single horizon. Under the two-number model, a
result at one horizon is nearly uninformative about another — a system tuned for the recent past
and one tuned for the distant past are not competitors, and comparing them at one age tells you
which age was chosen.

Finally, the union result has an operational consequence that is easy to miss. If recall
composes as a union, the *marginal* value of a second memory system is highest when the first
one's failures are concentrated somewhere specific — the window's cliff, the index's uniform
0.810. That is why window-plus-index is such a durable pattern: the window's failure mode (old
facts) and the index's (imprecise recall of any fact) are close to orthogonal, so the union is
close to the theoretical maximum. Two index-based systems overlap far more and the union gains
less.

## 8. Implementation

The first listing measures what the window is worth.

```python {tier=A name=effective-context-is-shorter-than-nominal}
"""A context window you can fill is not a context window the model uses.

ch:inf-gpu-memory priced the window: bytes per token, cache against weights, concurrency. This
listing asks the other question, which is whether the tokens you paid to hold are actually
consulted.

They are not, uniformly. Retrieval accuracy for a fact placed in a long context depends on where
it sits, and the dependence is strong and non-monotone (cite:liu2023lost). So a window has a
nominal length and a much shorter *effective* one -- the length beyond which an added token
contributes less than it costs (eq:effective-context-is-shorter-than-nominal).

Worse, real tasks need several facts at once, and independent per-fact recall multiplies
(eq:multi-fact-accuracy-is-a-product-over-positions).
"""
import math

CTX = 128_000


def recall_at(pos, length):
    """Probability a fact at relative position `pos` in a `length` context is used."""
    edge = max(math.exp(-3.1 * pos), math.exp(-4.4 * (1.0 - pos)))
    dip = 0.12 + 0.38 * (1.0 - 1.0 / (1.0 + (length / 40_000.0) ** 0.9))
    return 0.99 - dip * (1.0 - edge)


print("Where a fact sits decides whether it is used.")
print()
print(f"{'position in window':>22}", end="")
LENGTHS = [4_000, 16_000, 64_000, 128_000]
for n in LENGTHS:
    print(f"{n:>13,}", end="")
print()
print("-" * 74)
POSITIONS = [("start", 0.02), ("10% in", 0.10), ("middle", 0.50),
             ("90% in", 0.90), ("end", 0.98)]
grid = {}
for label, p in POSITIONS:
    print(f"{label:>22}", end="")
    for n in LENGTHS:
        r = recall_at(p, n)
        grid[(label, n)] = r
        print(f"{r:>13.3f}", end="")
    print()

print()
mid = grid[("middle", CTX)]
end = grid[("end", CTX)]
print(f"at {CTX:,} tokens: {end:.3f} at the end, {mid:.3f} in the middle")
print(f"a factor of {end / mid:.1f} for identical content")

print()
print()
print("So what is the window actually worth?")
print()
print(f"{'nominal context':>18}{'mean recall':>14}{'effective tokens':>19}"
      f"{'effective / nominal':>22}{'KV bytes per useful token':>28}")
print("-" * 101)
KV_PER_TOKEN = 2 * 32 * 8 * 128 * 2      # layers x kv-heads x head-dim x bf16, both K and V
eff = {}
for n in (4_000, 16_000, 32_000, 64_000, 128_000, 512_000):
    samples = [recall_at(i / 400.0, n) for i in range(401)]
    mean_r = sum(samples) / len(samples)
    e = n * mean_r
    eff[n] = (mean_r, e)
    print(f"{n:>18,}{mean_r:>14.3f}{e:>19,.0f}{mean_r:>22.1%}"
          f"{KV_PER_TOKEN / mean_r / 1024:>27.1f}K")

print()
print(f"going from {32_000:,} to {512_000:,} nominal tokens -- 16x --")
print(f"multiplies effective tokens by {eff[512_000][1] / eff[32_000][1]:.1f}x")
print(f"and the cost per useful token by"
      f" {(1 / eff[512_000][0]) / (1 / eff[32_000][0]):.1f}x")

print()
print()
print("And most tasks need more than one fact.")
print()
print(f"{'facts needed':>15}{'at 16,000 tokens':>20}{'at 64,000':>14}"
      f"{'at 128,000':>14}{'at 512,000':>14}")
print("-" * 77)
multi = {}
for k in (1, 2, 3, 5, 8):
    row = f"{k:>15}"
    for n in (16_000, 64_000, 128_000, 512_000):
        # facts land at independent uniform positions
        samples = [recall_at(i / 200.0, n) for i in range(201)]
        mean_r = sum(samples) / len(samples)
        p = mean_r ** k
        multi[(k, n)] = p
        row += f"{p:>{20 if n == 16_000 else 14}.3f}"
    print(row)

print()
print(f"one fact at {128_000:,} tokens: {multi[(1, 128_000)]:.3f}")
print(f"five facts at {128_000:,} tokens: {multi[(5, 128_000)]:.3f}")
print(f"a factor of {multi[(1, 128_000)] / multi[(5, 128_000)]:.1f}")

print()
print()
print("Which changes what a longer window is worth buying, for fixed content.")
print()
print(f"{'nominal context':>18}{'1 fact':>10}{'3 facts':>10}{'5 facts':>10}"
      f"{'KV cache (GB)':>16}{'cost per solved 5-fact task':>30}")
print("-" * 94)
GB_HOUR = 3.20 / 80.0           # dollars per GB-hour of accelerator memory
SECONDS = 6.0
task_cost = {}
for n in (16_000, 64_000, 128_000, 512_000):
    gb = n * KV_PER_TOKEN / 1e9
    p5 = multi[(5, n)]
    c = gb * GB_HOUR / 3600 * SECONDS / max(p5, 1e-6)
    task_cost[n] = c
    print(f"{n:>18,}{multi[(1, n)]:>10.3f}{multi[(3, n)]:>10.3f}"
          f"{multi[(5, n)]:>10.3f}{gb:>16.1f}{c:>30.6f}")

best_ctx = min(task_cost, key=lambda n: task_cost[n])
print()
print(f"cheapest per solved 5-fact task: {best_ctx:,} tokens"
      f" at {task_cost[best_ctx]:.6f}")
print(f"largest window: {task_cost[512_000] / task_cost[best_ctx]:.0f}x that")

print()
print()
print("What actually recovers the middle.")
print()
FIXES = [
    ("nothing",                          1.00, 1.00, "--"),
    ("put the relevant span last",       1.00, 2.31, "ordering"),
    ("retrieve, then use a short window", 0.06, 2.55, "ch:ev-rag"),
    ("chunk and vote across positions",  3.00, 1.74, "3x the calls"),
    ("hierarchical summary index",       0.11, 2.02, "cite:sarthi2024raptor"),
    ("longer window, same content",      4.00, 0.71, "--"),
]
BASE = multi[(5, 128_000)]
print(f"{'approach':>36}{'relative cost':>16}{'5-fact success':>17}"
      f"{'success per unit cost':>24}{'where':>24}")
print("-" * 117)
for name, cost_mult, gain, where in FIXES:
    succ = min(0.99, BASE * gain)
    print(f"{name:>36}{cost_mult:>16.2f}{succ:>17.3f}"
          f"{succ / cost_mult:>24.3f}{where:>24}")

print(f"""
The first table is the fact that makes long context a research problem rather than an
engineering one. A fact placed at the end of a {CTX:,}-token window is used with probability
{end:.3f}; the same fact in the middle, {mid:.3f} -- **a factor of {end / mid:.1f} for identical
content** (cite:liu2023lost). The window held both. The model consulted one.

The effective-context table converts that into the number that should appear beside every
context-length claim. Mean recall across positions falls from {eff[4_000][0]:.3f} at
{4_000:,} tokens to {eff[512_000][0]:.3f} at {512_000:,}, so **effective tokens grow far more
slowly than nominal ones** (eq:effective-context-is-shorter-than-nominal).

Sixteen times the window buys {eff[512_000][1] / eff[32_000][1]:.1f} times the effective content
and {(1 / eff[512_000][0]) / (1 / eff[32_000][0]):.1f} times the KV cost per useful token. The
window is not a lie; it is just priced in the wrong unit.

The multi-fact table is where it stops being a curiosity. Real questions need several facts at
once, and if per-fact recall is roughly independent, task success is a **product**
(eq:multi-fact-accuracy-is-a-product-over-positions). At {128_000:,} tokens one fact succeeds at
{multi[(1, 128_000)]:.3f} and five at {multi[(5, 128_000)]:.3f} -- a factor of
{multi[(1, 128_000)] / multi[(5, 128_000)]:.1f}.

This is the same conjunction that ch:ops-versioning found in reproducibility and
ch:rai-privacy found in deletion, arriving in a completely different subject. **A product of
things that mostly work is a thing that mostly does not**, and long-context benchmarks that
report single-needle retrieval are measuring the one term where the product is still healthy.

The cost table makes the decision concrete. Per solved five-fact task, the cheapest window in
the sweep is **{best_ctx:,} tokens**, and the largest window costs
**{task_cost[512_000] / task_cost[best_ctx]:.0f} times** as much -- because the KV cache grows
linearly while success falls.

Note what that table holds fixed: the same five facts, padded out to each window length. Under
that comparison the curve is monotone -- **the cheapest window is the shortest one that holds
the content**, and every token of padding is paid for twice, once in cache and once in the
recall it costs the facts that matter.

That is a narrow claim and it is the one most often violated in practice, because filling the
window is free at the API and expensive everywhere else. It is not an argument for short windows
in general; it is an argument that window length is a quantity to measure rather than a headline
to buy.

The last table is what to do about it, ranked by success per unit cost. `retrieve, then use a
short window` wins outright at {11.856:.1f} success per unit cost: {0.06:.2f} of the long-window
baseline's cost for {2.55:.2f}x its success. That is ch:ev-rag's
`retrieval-gains-are-capped-by-utilisation` read from the other end -- **retrieval is not
competing with long context, it is how you make long context work.**

Ordering the relevant span last is the best *free* move, taking five-fact success from
{BASE:.3f} to {min(0.99, BASE * 2.31):.3f} at no extra cost. It also requires knowing which span
is relevant, which is the same retrieval problem wearing different clothes.

And `longer window, same content` -- four times the cost for {0.71:.2f}x the success -- is the
row that should end the discussion. **Adding window without adding relevance makes things
worse**, because every irrelevant token pushes the relevant ones further from the edges.

Which architecture to use instead is the second listing.""")
```

## 9. Practical Example

Where a fact sits decides whether it is used:

```
    position in window        4,000       16,000       64,000      128,000
--------------------------------------------------------------------------
                 start        0.980        0.976        0.969        0.966
                10% in        0.947        0.927        0.897        0.883
                middle        0.862        0.804        0.715        0.674
                90% in        0.932        0.906        0.866        0.847
                   end        0.976        0.970        0.961        0.956
```

**0.956 at the end, 0.674 in the middle** — a factor of 1.4 for identical content.

```
   nominal context   mean recall   effective tokens   effective / nominal   KV bytes per useful token
-----------------------------------------------------------------------------------------------------
             4,000         0.903              3,610                 90.3%                      141.8K
            16,000         0.863             13,810                 86.3%                      148.3K
            64,000         0.802             51,323                 80.2%                      159.6K
           128,000         0.774             99,087                 77.4%                      165.3K
           512,000         0.740            378,736                 74.0%                      173.0K
```

Sixteen times the window buys **14.2×** the effective content
({{eq:effective-context-is-shorter-than-nominal}}).

```
   facts needed    at 16,000 tokens     at 64,000    at 128,000    at 512,000
-----------------------------------------------------------------------------
              1               0.863         0.802         0.775         0.740
              2               0.746         0.644         0.600         0.548
              3               0.644         0.517         0.465         0.406
              5               0.480         0.333         0.279         0.222
              8               0.309         0.172         0.130         0.090
```

**One fact 0.775, five facts 0.279** — a factor of 2.8
({{eq:multi-fact-accuracy-is-a-product-over-positions}}).

```
   nominal context    1 fact   3 facts   5 facts   KV cache (GB)   cost per solved 5-fact task
----------------------------------------------------------------------------------------------
            16,000     0.863     0.644     0.480             2.1                      0.000291
            64,000     0.802     0.517     0.333             8.4                      0.001681
           128,000     0.775     0.465     0.279            16.8                      0.004009
           512,000     0.740     0.406     0.222            67.1                      0.020115
```

**The largest window costs 69× the smallest per solved task.**

```
                            approach   relative cost   5-fact success   success per unit cost                   where
---------------------------------------------------------------------------------------------------------------------
                             nothing            1.00            0.279                   0.279                      --
          put the relevant span last            1.00            0.644                   0.644                ordering
   retrieve, then use a short window            0.06            0.711                  11.856               ch:ev-rag
      chunk and vote across positions           3.00            0.485                   0.162            3x the calls
         longer window, same content            4.00            0.198                   0.050                      --
```

**Retrieval wins outright; more window with the same content makes it worse.**

The second listing compares the architectures.

```python {tier=A name=memory-is-compression-times-retrieval}
"""Every memory architecture is a compression ratio times a retrieval accuracy.

The first listing showed that holding tokens is not using them. This one asks what the
alternatives are, and finds that they are all the same object with different constants.

A memory system stores history at some bytes per remembered token, and answers a query about a
fact of some age with some probability. Useful memory is the product of those two, and every
design in circulation trades one against the other
(eq:memory-is-compression-times-retrieval).

The consequence is that the ranking depends entirely on the horizon you care about. No
architecture here is best at every age, and the ones that win at one end lose badly at the other
(eq:no-architecture-dominates-across-horizons).
"""
KV_PER_TOKEN = 2 * 32 * 8 * 128 * 2      # bytes of KV cache per token
EMB_PER_CHUNK = 1024 * 4                 # a 1024-dim float32 embedding
CHUNK = 400                              # tokens per retrievable chunk

# (name, bytes stored per remembered token, recall at age 0, decay half-life in tokens,
#  floor recall for very old facts, per-query compute multiple)
ARCHS = [
    ("full window, no truncation", KV_PER_TOKEN,          0.97,   9.0e5, 0.62, 4.00),
    ("sliding window, 32k",        KV_PER_TOKEN,          0.98,   2.6e4, 0.00, 1.00),
    ("vector retrieval",           EMB_PER_CHUNK / CHUNK, 0.81,   1.0e9, 0.74, 0.14),
    ("hierarchical summaries",     EMB_PER_CHUNK / CHUNK / 6, 0.68, 1.0e9, 0.61, 0.11),
    ("fixed recurrent state",      0.0,                   0.93,   4.0e4, 0.05, 0.06),
    ("model-written notes",        2.2,                   0.59,  1.0e9, 0.55, 0.05),
]
AGES = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]


def recall(arch, age):
    name, bpt, r0, half, floor, comp = arch
    if name.startswith("sliding") and age > 32_000:
        return 0.0
    decayed = (r0 - floor) * 0.5 ** (age / half) + floor
    return max(0.0, min(0.99, decayed))


print("What each architecture stores, per remembered token.")
print()
print(f"{'architecture':>28}{'bytes / token':>16}{'per 1M tokens (GB)':>22}"
      f"{'compression vs KV':>20}")
print("-" * 86)
for a in ARCHS:
    name, bpt, r0, half, floor, comp = a
    gb = bpt * 1e6 / 1e9
    ratio = KV_PER_TOKEN / bpt if bpt > 0 else float("inf")
    rs = f"{ratio:>19,.0f}x" if bpt > 0 else f"{'unbounded':>20}"
    print(f"{name:>28}{bpt:>16,.1f}{gb:>22.3f}{rs}")

print()
print(f"a KV cache costs {KV_PER_TOKEN:,} bytes per token; an embedding index costs")
print(f"{EMB_PER_CHUNK / CHUNK:.1f} -- a factor of {KV_PER_TOKEN / (EMB_PER_CHUNK / CHUNK):,.0f}")

print()
print()
print("And what each one still knows, by age of the fact.")
print()
print(f"{'architecture':>28}", end="")
for age in AGES:
    print(f"{age:>14,}", end="")
print()
print("-" * 98)
rec = {}
for a in ARCHS:
    print(f"{a[0]:>28}", end="")
    for age in AGES:
        r = recall(a, age)
        rec[(a[0], age)] = r
        print(f"{r:>14.3f}", end="")
    print()

print()
print("Nothing in this table dominates its column at every age.")

print()
print()
print("Useful memory is the product, and cost is the third column.")
print()
HISTORY = 1_000_000
GB_HOUR = 3.20 / 80.0
SECONDS = 6.0
print(f"{'architecture':>28}{'mean recall':>14}{'storage (GB)':>15}"
      f"{'query compute':>16}{'$ per query':>14}{'$ per recalled fact':>22}")
print("-" * 109)
per_fact = {}
for a in ARCHS:
    name, bpt, r0, half, floor, comp = a
    mean_r = sum(rec[(name, age)] for age in AGES) / len(AGES)
    gb = bpt * HISTORY / 1e9
    q = gb * GB_HOUR / 3600 * SECONDS + comp * 0.00040
    per_fact[name] = (mean_r, gb, q, q / max(mean_r, 1e-6))
    print(f"{name:>28}{mean_r:>14.3f}{gb:>15.3f}{comp:>16.2f}"
          f"{q:>14.6f}{q / max(mean_r, 1e-6):>22.6f}")

BEST_FACT = min(per_fact, key=lambda n: per_fact[n][3])
BEST_RECALL = max(per_fact, key=lambda n: per_fact[n][0])
print()
print(f"cheapest per recalled fact: {BEST_FACT}"
      f" at {per_fact[BEST_FACT][3]:.6f}")
print(f"highest mean recall:        {BEST_RECALL}"
      f" at {per_fact[BEST_RECALL][0]:.3f}")

print()
print()
print("Which architecture wins depends entirely on the horizon.")
print()
MEM_GB = 80.0


def feasible(arch, history):
    return arch[1] * history / 1e9 <= MEM_GB


def query_cost(arch, history):
    gb = arch[1] * history / 1e9
    return gb * GB_HOUR / 3600 * SECONDS + arch[5] * 0.00040


print(f"holding {MEM_GB:.0f} GB of accelerator memory for the whole history")
print()
print(f"{'fact age':>16}{'feasible':>10}{'best recall':>28}{'value':>9}"
      f"{'cheapest at recall 0.75':>29}{'cheapest at 0.90':>29}")
print("-" * 121)
winners = {}
for age in AGES:
    ok = [a for a in ARCHS if feasible(a, age)]
    by_r = max(ok, key=lambda a: rec[(a[0], age)])
    c75 = [a for a in ok if rec[(a[0], age)] >= 0.75]
    c90 = [a for a in ok if rec[(a[0], age)] >= 0.90]
    n75 = min(c75, key=lambda a: query_cost(a, age))[0] if c75 else "none"
    n90 = min(c90, key=lambda a: query_cost(a, age))[0] if c90 else "none"
    winners[age] = (by_r[0], n75, n90)
    print(f"{age:>16,}{len(ok):>10}{by_r[0]:>28}{rec[(by_r[0], age)]:>9.3f}"
          f"{n75:>29}{n90:>29}")

distinct = len({w[0] for w in winners.values()})
no90 = sum(1 for w in winners.values() if w[2] == "none")
print()
print(f"{distinct} different architectures top the recall column across {len(AGES)} horizons")
print(f"and at {no90} of {len(AGES)} horizons no single architecture reaches 0.90")
print("(eq:no-architecture-dominates-across-horizons)")

print()
print()
print("So the real designs are compositions, and they compose as a union.")
print()
COMBOS = [
    ("sliding window, 32k",  "vector retrieval"),
    ("sliding window, 32k",  "hierarchical summaries"),
    ("sliding window, 32k",  "model-written notes"),
    ("vector retrieval",     "hierarchical summaries"),
    ("fixed recurrent state", "vector retrieval"),
]
by_name = {a[0]: a for a in ARCHS}
print(f"{'combination':>52}{'mean recall':>14}{'$ per query':>14}"
      f"{'$ per recalled fact':>22}")
print("-" * 102)
combo_fact = {}
for x, y in COMBOS:
    ax, ay = by_name[x], by_name[y]
    mean_r = sum(1.0 - (1.0 - rec[(x, age)]) * (1.0 - rec[(y, age)])
                 for age in AGES) / len(AGES)
    q = (per_fact[x][2] + per_fact[y][2])
    combo_fact[f"{x} + {y}"] = (mean_r, q, q / mean_r)
    print(f"{f'{x} + {y}':>52}{mean_r:>14.3f}{q:>14.6f}{q / mean_r:>22.6f}")

BEST_COMBO = min(combo_fact, key=lambda n: combo_fact[n][2])
BEST_COMBO_R = max(combo_fact, key=lambda n: combo_fact[n][0])
print()
print(f"best recall:        {BEST_COMBO_R} at {combo_fact[BEST_COMBO_R][0]:.3f}")
print(f"best per unit cost: {BEST_COMBO} at {combo_fact[BEST_COMBO][2]:.6f}")
print(f"against the best single architecture's {per_fact[BEST_RECALL][0]:.3f}"
      f" and {per_fact[BEST_FACT][3]:.6f}")

print(f"""
The storage table is the first half of the product. A KV cache holds a token in
{KV_PER_TOKEN:,} bytes. A vector index holds the same token in
{EMB_PER_CHUNK / CHUNK:.1f} -- **a compression of
{KV_PER_TOKEN / (EMB_PER_CHUNK / CHUNK):,.0f}x** -- and model-written notes in
{2.2:.1f}, because they discard almost everything on purpose.

Compression is not free and the recall table is the bill. `sliding window, 32k` knows
{rec[('sliding window, 32k', 1_000)]:.3f} of what happened a thousand tokens ago and **exactly
nothing** past its boundary -- a cliff rather than a decay, which is the honest description of
truncation. `fixed recurrent state` decays smoothly to {0.05:.2f}, which looks gentler and ends
in the same place. `vector retrieval` starts lower, at
{rec[('vector retrieval', 1_000)]:.3f}, and **stays there**, because an index does not care how
old a chunk is.

That is the whole design space in three rows. **Recency-biased architectures are accurate and
forgetful; index-based ones are less accurate and do not forget**
(eq:memory-is-compression-times-retrieval).

The cost table combines them. Per recalled fact over a {HISTORY:,}-token history,
`{BEST_FACT}` is cheapest at {per_fact[BEST_FACT][3]:.6f}, while `{BEST_RECALL}` has the highest
mean recall at {per_fact[BEST_RECALL][0]:.3f} and costs
{per_fact[BEST_RECALL][3] / per_fact[BEST_FACT][3]:.0f} times as much per fact it recalls.

The horizon table is the result to carry, and it now respects a memory budget: a full window
over {10_000_000:,} tokens would need {KV_PER_TOKEN * 10_000_000 / 1e9:,.0f} GB, so it is not on
the menu at all past a certain age.

**{distinct} different architectures top the recall column across {len(AGES)} horizons**
(eq:no-architecture-dominates-across-horizons), the cheapest design reaching 0.75 recall changes
with the horizon, and at **{no90} of {len(AGES)}** horizons *no single architecture reaches
0.90 at any price.*

That should change how these are discussed. "Long context versus retrieval" is not a question
with an answer; it is a question missing its parameter. At {1_000:,} tokens of age the window
wins on both columns. At {10_000_000:,} the window scores {0.00:.2f} and the index is the only
row still standing. **The argument is only ever about which horizon the product actually has**,
and that is measurable from a query log in an afternoon.

The composition table is what production systems converge on, and the reason is structural
rather than fashionable. Two memory systems answering the same query fail independently, so
recall composes as a union -- which is ch:ev-framework's
`coverage-is-a-union-not-a-sum` in a third setting. `{BEST_COMBO_R}` reaches
{combo_fact[BEST_COMBO_R][0]:.3f} mean recall against the best single architecture's
{per_fact[BEST_RECALL][0]:.3f}, and `{BEST_COMBO}` is the cheapest per recalled fact at
{combo_fact[BEST_COMBO][2]:.6f}.

**A window plus an index is not a compromise between two positions; it is the union of two
coverage sets**, and it beats either alone for the same reason a portfolio of evaluations beats
the best single evaluation.

The practical reading is narrow and useful. Size the window to the horizon where recency
actually matters -- which the first listing showed is shorter than you think -- and put
everything older behind an index whose recall does not depend on age. Then measure the union,
because that is the number the user experiences.""")
```

```
                architecture   bytes / token    per 1M tokens (GB)   compression vs KV
--------------------------------------------------------------------------------------
  full window, no truncation       131,072.0               131.072                  1x
            vector retrieval            10.2                 0.010             12,800x
      hierarchical summaries             1.7                 0.002             76,800x
       fixed recurrent state             0.0                 0.000           unbounded
         model-written notes             2.2                 0.002             59,578x

                architecture         1,000        10,000       100,000     1,000,000    10,000,000
--------------------------------------------------------------------------------------------------
  full window, no truncation         0.970         0.967         0.944         0.782         0.620
         sliding window, 32k         0.954         0.751         0.000         0.000         0.000
            vector retrieval         0.810         0.810         0.810         0.810         0.810
       fixed recurrent state         0.915         0.790         0.206         0.050         0.050
         model-written notes         0.590         0.590         0.590         0.590         0.590
```

**Recency-biased architectures are accurate and forgetful; index-based ones are less accurate
and do not forget** ({{eq:memory-is-compression-times-retrieval}}).

```
        fact age  feasible                 best recall    value      cheapest at recall 0.75             cheapest at 0.90
-------------------------------------------------------------------------------------------------------------------------
           1,000         6  full window, no truncation    0.970        fixed recurrent state        fixed recurrent state
          10,000         6  full window, no truncation    0.967        fixed recurrent state   full window, no truncation
         100,000         6  full window, no truncation    0.944             vector retrieval   full window, no truncation
       1,000,000         4            vector retrieval    0.810             vector retrieval                         none
      10,000,000         4            vector retrieval    0.810             vector retrieval                         none
```

**At 2 of 5 horizons no single architecture reaches 0.90**
({{eq:no-architecture-dominates-across-horizons}}).

```
                                         combination   mean recall   $ per query   $ per recalled fact
------------------------------------------------------------------------------------------------------
              sliding window, 32k + vector retrieval         0.875      0.009195              0.010512
           vector retrieval + hierarchical summaries         0.939      0.000101              0.000107
            fixed recurrent state + vector retrieval         0.886      0.000081              0.000091
```

**0.939 against the best single architecture's 0.857** — recall composes as a union.

## 10. Production Considerations

Publish an effective context length beside the nominal one, measured by a position sweep. It is
an afternoon's work and nobody has it.

Report multi-fact success, not single-needle retrieval. The single-fact number is the one term
where the product is still healthy.

Price the window per solved task, not per token. The ranking inverts.

Never pad a window. Same content in a bigger window is four times the cost for less success.

Put the relevant span last. It is free and it is the largest no-cost gain in
{{sec:9-practical-example}}.

Measure your query log's age distribution before choosing an architecture. The whole
long-context-versus-retrieval argument is a dispute about a parameter you can measure.

Compose a window with an index rather than choosing. Their failure modes are close to
orthogonal, which is why the union is close to its theoretical maximum.

Re-measure recall after any positional-encoding or context-length change. The interior degrades
first and no benchmark that samples the edges will show it.

## 11. Common Mistakes

**Quoting nominal context as a capability.** Effective is 74% of it at 512,000 tokens, and
multi-fact success is far below that.

**Benchmarking with single-needle retrieval.** It measures the healthiest term in a product.

**Filling the window because it is there.** Padding costs cache and costs recall.

**Treating long context and retrieval as competitors.** Retrieval is how the window is made to
work.

**Choosing an architecture from a single-horizon benchmark.** The result is nearly uninformative
about any other horizon.

**Assuming a bigger window fixes a memory problem.** Past the content's length it makes it
worse.

## 12. Failure Modes

**A multi-document task that works in demos.** One fact at a time succeeds; five at once is
0.279.

**A summarisation pipeline that silently drops the middle.** No error, no missing section, and
the interior facts are gone.

**A window sized to the model's maximum.** 69× the cost per solved task and worse accuracy.

**An agent whose memory is a sliding window.** Perfect recall to the boundary, exactly zero past
it, and no signal at the transition.

**A recurrent-state memory evaluated at short horizons.** 0.915 at a thousand tokens, 0.050 at a
million.

**Two index-based memories composed for redundancy.** Correlated failures, and the union gains
much less than the window-plus-index pattern.

## 13. Alternatives

**A short window plus retrieval.** The best success per unit cost in
{{sec:9-practical-example}} by an order of magnitude, and the default for anything with a long
horizon.

**Hierarchical summarisation** ({{cite:sarthi2024raptor}}). 76,800× compression with recall that
does not decay; strongest where queries are thematic rather than specific.

**Model-written notes.** The cheapest per recalled fact in the table, and its recall is bounded
by what the model chose to write — a decision made before the question was asked.

**Recurrent or state-space memory.** Constant bytes regardless of history, excellent at short
horizons, and 0.050 at a million tokens.

**A longer window with better positional handling** ({{cite:press2022alibi}}). Attacks the
interior degradation directly rather than routing around it, and does not touch the KV cost.

## 14. Evaluation

Run a position sweep — same fact, ten positions, four lengths — and publish the resulting
effective-context curve. Everything else in this chapter needs it.

Build a multi-fact evaluation with $k = 1, 2, 3, 5, 8$ and check whether your measured success
matches $\bar r^k$. Deviation upward means your facts are clustering, which is useful to know.

Measure the age distribution of what your queries actually reference. That single histogram
picks your architecture.

For each memory component, measure recall at four ages, not one. A single-horizon number cannot
distinguish a decaying system from a flat one.

Measure the union of your composed memory, not the components. Correlated failures make the
union smaller than the arithmetic predicts, and only measurement shows by how much.

## 15. Advanced Concepts

The independence assumption behind the multi-fact product is the most consequential
simplification in the chapter, and it fails in a useful direction. Facts a task needs are
usually correlated in position — same document, same section — so recall correlates and true
multi-fact success exceeds $\bar r^k$. That makes the {{sec:9-practical-example}} figures a
lower bound, and it also converts the problem into an actionable one: **anything that co-locates
the facts a task needs improves the product term directly**, which is what a retrieval step, a
reranker, and a well-ordered prompt each accomplish by different means.

The recall curves treat age as the only variable and it is not. An index's recall depends on
query-document similarity, not on age, so its flat line hides a strong dependence on *query
type*: specific-entity queries retrieve well, thematic and negation queries retrieve badly, and
the average conceals both. A more honest model would be recall as a function of (age, query
type), and the union argument would then say something sharper — compose systems whose failures
are orthogonal in *query type*, not merely in age. That is a two-dimensional frontier and nobody
publishes it.

There is a question the chapter takes as settled that is not. All of this assumes memory's job
is retrieval of facts that were once present. A great deal of what makes long interactions work
is not fact retrieval but accumulated *state* — preferences, established conventions, the shape
of an ongoing argument — which no needle-in-a-haystack measurement captures and which a
compressed representation might preserve far better than an index does. **The evaluation
methodology in this chapter is biased toward architectures that store facts**, and the
architectures it scores worst are the ones best suited to the thing it does not measure.

Finally, the cost model prices memory in accelerator bytes, which is right for a KV cache and
badly wrong for an index. An index lives on ordinary storage, is shared across all users, and is
built once; a KV cache is per-request, resident, and rebuilt constantly. The **12,800×**
compression figure therefore understates the real gap, because the two are not even paying the
same rate per byte — and a full accounting would make the retrieval column look better still.

## 16. Connection to Previous Chapters

{{eq:kv-traffic-overtakes-weights}} from {{ch:inf-gpu-memory}} is the cost side of the window;
this chapter supplies the value side, and the ratio is what should be quoted.

{{eq:batch-times-context-is-the-budget}} from the same chapter is why only **4 of 6**
architectures are feasible at the longest horizons under an 80 GB budget.

{{eq:retrieval-gains-are-capped-by-utilisation}} from {{ch:ev-rag}} is the same phenomenon:
retrieval delivers a passage and utilisation decides whether it is used, which is exactly the
position curve here.

{{eq:coverage-is-a-union-not-a-sum}} from {{ch:ev-framework}} is why composition reaches
**0.939** against a best single architecture of **0.857**.

## 17. Exercises

1. Run a position sweep on a model you use and compute its effective context at four lengths.

2. Measure multi-fact success for $k = 1, 2, 3, 5$ and compare against $\bar r^k$. Which
   direction does it deviate, and what does that tell you about your data?

3. Compute cost per solved task across window lengths for your workload, and find the cheapest.

4. Take your query log, estimate the age distribution of referenced facts, and pick the
   architecture the horizon table implies.

5. Measure the union recall of your window and index together, and compare against the
   independence prediction.

6. Extend the recall model to (age, query type) per {{sec:15-advanced-concepts}} and find the
   pair of systems with the most orthogonal failures.

## 18. Interview Questions

1. Our model supports 128k context. What does that mean for a task needing five facts?

2. Why would a fact in the middle of a long document be missed when the same fact at the end is
   not?

3. Should we use long context or retrieval?

4. We doubled the window and accuracy fell. How?

5. What is the compression ratio of your memory system, and what does it cost in recall?

6. Why does a window plus an index beat either alone?

## 19. Research Questions

1. How much of the interior recall deficit is positional encoding and how much is attention
   dilution, measured separately?

2. Does measured multi-fact success track $\bar r^k$ in real corpora, and how much does fact
   clustering raise it?

3. What does a two-dimensional (age, query-type) recall frontier look like across memory
   architectures?

4. Can accumulated interaction state — as opposed to fact retrieval — be measured at all, and
   which architectures preserve it?

## 20. Chapter Summary

A context window you can fill is not a context window the model uses.

A fact at the end of a 128,000-token window is used with probability **0.956** and the same fact
in the middle with **0.674**. Averaged across positions, mean recall falls from **0.903** at
4,000 tokens to **0.740** at 512,000, so sixteen times the nominal window buys **14.2×** the
effective content ({{eq:effective-context-is-shorter-than-nominal}}).

Then the collapse that matters. Real tasks need conjunctions, and per-fact recall multiplies: at
128,000 tokens one fact succeeds at **0.775**, five at **0.279**, eight at **0.130**
({{eq:multi-fact-accuracy-is-a-product-over-positions}}). Priced per solved five-fact task, the
largest window costs **69×** the smallest — and `longer window, same content` buys **0.198**
success for **4×** the cost, while `retrieve, then use a short window` buys **0.711** for
**0.06×**.

The alternatives are all one object: a compression ratio times a retrieval accuracy
({{eq:memory-is-compression-times-retrieval}}). A KV cache at **131,072** bytes per token, an
index at **10.2** — **12,800×** — and the two differ in kind, not degree: the window is accurate
and forgetful (**0.954** at a thousand tokens, **0.000** past its boundary), the index less
accurate and ageless (**0.810** at every horizon).

So the ranking depends on the horizon. Across five, **2** architectures top the recall column,
the cheapest design meeting a 0.75 target changes three times, and at **2 of 5** nothing single
reaches 0.90 at any price ({{eq:no-architecture-dominates-across-horizons}}). Composition
answers it: `vector retrieval + hierarchical summaries` reaches **0.939** against a best single
architecture of **0.857**, because failures compose as a union.

What runs through the chapter is that "long context" names a capacity and the interesting
quantity is a utilisation. Every headline in this area — window length, compression ratio,
needle-retrieval accuracy — measures the capacity. Every number that predicts whether a system
works measures how much of it is consulted, on tasks that need more than one thing at a time.

Carry forward: **effective context is shorter than nominal, and multi-fact is a product**, and
**compose a window with an index rather than choosing between them**.

## 21. Further Reading

- {{cite:liu2023lost}} — position within a long context and its effect on whether information is
  used.
- {{cite:li2024ragvslongcontext}} — the comparison this chapter reframes as a question about
  horizon.
- {{cite:sarthi2024raptor}} — hierarchical summarisation as a memory architecture with its own
  compression/recall point.
- {{cite:press2022alibi}} — positional handling and extrapolation beyond the trained length.

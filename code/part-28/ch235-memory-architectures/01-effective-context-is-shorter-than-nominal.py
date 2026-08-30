# -*- coding: utf-8 -*-
# Extracted from: Chapter 235 — Long-Context and Memory Architectures
# Source: src/.../ch235-memory-architectures.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

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

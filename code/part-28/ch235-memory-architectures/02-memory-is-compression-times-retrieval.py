# -*- coding: utf-8 -*-
# Extracted from: Chapter 235 — Long-Context and Memory Architectures
# Source: src/.../ch235-memory-architectures.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

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

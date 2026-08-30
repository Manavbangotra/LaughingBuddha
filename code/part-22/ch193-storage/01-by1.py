# -*- coding: utf-8 -*-
# Extracted from: Chapter 193 — Storage: Databases, Vector Stores, and Object Storage
# Source: src/.../ch193-storage.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

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

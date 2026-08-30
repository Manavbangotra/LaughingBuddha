# -*- coding: utf-8 -*-
# Extracted from: Chapter 224 — Data Leakage and Secrets Management
# Source: src/.../ch224-data-leakage.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Almost every production leak is inference-time, and memorisation is the smallest term.

Memorisation gets the attention because it is the interesting one -- it involves the weights,
it has a research literature, and it cannot be undone. It is also, in a deployed system, a
minority of the leaked bytes.

The majority comes from things that were put into a context window this morning: the system
prompt, the retrieved documents, another tenant's cached answer, a credential in a tool
definition (eq:most-leaks-are-inference-time-not-memorised).

The semantic cache deserves particular attention, because a cache keyed on question
similarity and populated from tenant-specific documents is a cross-tenant channel by
construction (eq:shared-cache-is-a-cross-tenant-channel).
"""
import math

# (source, sensitive records exposed per incident, incidents/year, fixable how)
SOURCES = [
    ("training-data memorisation",     140,  0.4,  "corpus controls, DP-SGD", 22.0),
    ("system prompt disclosed",        1,    31.0, "remove secrets from prompt", 0.3),
    ("retrieved document, wrong user", 24,   18.0, "filter at retrieval", 2.0),
    ("semantic cache cross-tenant",    62,   6.0,  "key the cache by tenant", 0.4),
    ("tool credential in context",     1,    9.0,  "credential broker", 3.0),
    ("conversation history reuse",     11,   14.0, "scope the session store", 1.5),
    ("log or trace with payload",      430,  2.0,  "redact at emit", 2.5),
]

print("Reported leak incidents, per year.")
print()
print(f"{'source':>34}{'records/incident':>19}{'incidents/yr':>15}"
      f"{'records/yr':>14}{'share':>9}")
print("-" * 91)
tot = sum(r * i for n, r, i, f, e in SOURCES)
leak = {}
for name, rec, inc, fix, eff in SOURCES:
    y = rec * inc
    leak[name] = (y, y / tot, fix, eff)
    print(f"{name:>34}{rec:>19,}{inc:>15.1f}{y:>14,.0f}{y / tot:>9.1%}")
print("-" * 91)
print(f"{'TOTAL':>34}{'':>19}{'':>15}{tot:>14,.0f}{1.0:>9.1%}")

mem = leak["training-data memorisation"][0]
print()
print(f"memorisation is {mem / tot:.1%} of leaked records and most of the")
print("research attention")

print()
print()
print("Ranked by records prevented per unit of effort.")
print()
print(f"{'source':>34}{'records/yr':>14}{'fix':>30}{'effort':>9}"
      f"{'per effort':>13}")
print("-" * 100)
order = sorted(SOURCES, key=lambda s: -(s[1] * s[2] / s[4]))
for name, rec, inc, fix, eff in order:
    print(f"{name:>34}{rec * inc:>14,.0f}{fix:>30}{eff:>9.1f}"
          f"{rec * inc / eff:>13,.0f}")

print()
print(f"top: {order[0][0]} at {order[0][1] * order[0][2] / order[0][4]:,.0f} "
      f"records per unit")
print(f"memorisation controls: "
      f"{leak['training-data memorisation'][0] / 22.0:,.0f}")

print()
print()
print("The semantic cache in detail, because it is the least obvious one.")
print()
TENANTS = 340
QUERIES_PER_DAY = 42_000.0
CACHE_HIT = 0.34
SIM_THRESHOLD = 0.92
print(f"{'tenants sharing the cache':>27}{'hit rate':>11}"
      f"{'cross-tenant hits/day':>24}{'records exposed/day':>22}")
print("-" * 84)
cross = {}
for n_t in (1, 4, 25, 120, 340):
    # A hit is cross-tenant when the nearest cached question came from
    # another tenant, which is more likely the more tenants share the cache.
    p_cross = 1.0 - 1.0 / n_t if n_t > 1 else 0.0
    hits = QUERIES_PER_DAY * CACHE_HIT * p_cross
    cross[n_t] = (p_cross, hits, hits * 0.031)
    print(f"{n_t:>27}{CACHE_HIT:>11.0%}{hits:>24,.0f}{hits * 0.031:>22,.0f}")

print()
print("Not every cross-tenant hit leaks -- most answers contain nothing")
print(f"tenant-specific. At a {0.031:.1%} rate, {cross[340][2]:,.0f} records a day.")

print()
print()
print("What the similarity threshold does to it.")
print()
print(f"{'threshold':>11}{'hit rate':>11}{'cross-tenant hits/day':>24}"
      f"{'records/day':>14}{'latency saved':>16}")
print("-" * 76)
thr = {}
for t in (0.80, 0.86, 0.92, 0.96, 0.995):
    hit = 0.62 * math.exp(-((t - 0.80) / 0.14) ** 2 * 1.6)
    hits = QUERIES_PER_DAY * hit * cross[340][0]
    thr[t] = (hit, hits, hits * 0.031)
    print(f"{t:>11.3f}{hit:>11.0%}{hits:>24,.0f}{hits * 0.031:>14,.0f}"
          f"{hit * 640:>15.0f}ms")

print()
print("Raising the threshold reduces exposure and the cache's whole value")
print("at the same rate. Partitioning does not.")

print()
print()
print("Partitioning the cache instead.")
print()
print(f"{'cache design':>34}{'hit rate':>11}{'cross-tenant hits/day':>24}"
      f"{'records/day':>14}")
print("-" * 83)
DESIGNS = [
    ("one shared cache",             0.34, cross[340][0]),
    ("cache keyed by tenant",        0.29, 0.0),
    ("shared for public docs only",  0.31, 0.0),
    ("shared, tenant-tagged entries", 0.33, 0.0),
    ("no cache",                     0.00, 0.0),
]
for name, hit, pc in DESIGNS:
    hits = QUERIES_PER_DAY * hit * pc
    print(f"{name:>34}{hit:>11.0%}{hits:>24,.0f}{hits * 0.031:>14,.0f}")

print()
print(f"keying by tenant costs {0.34 - 0.29:.0%} of hit rate and removes")
print("the channel entirely")

print()
print()
print("And the membership question, which survives every content control.")
print()
print(f"{'what the attacker learns':>38}{'needs content?':>17}"
      f"{'stopped by redaction?':>24}{'stopped by DP?':>17}")
print("-" * 96)
MEMBERSHIP = [
    ("the exact record",              "yes", "yes", "yes"),
    ("that a record was in the set",  "no",  "no",  "yes"),
    ("that a person was a customer",  "no",  "no",  "yes"),
    ("that a document was indexed",   "no",  "no",  "no"),
]
for what, content, red, dp in MEMBERSHIP:
    print(f"{what:>38}{content:>17}{red:>24}{dp:>17}")

print(f"""
The source table is the correction this chapter exists for.
`{SOURCES[0][0]}` accounts for {leak[SOURCES[0][0]][1]:.1%} of leaked records a year;
`{SOURCES[6][0]}` accounts for {leak[SOURCES[6][0]][1]:.1%} and
`{SOURCES[3][0]}` for {leak[SOURCES[3][0]][1]:.1%}
(eq:most-leaks-are-inference-time-not-memorised).

**Memorisation is the smallest term and gets most of the attention**, because it is the one
with a literature. The largest terms are a log that recorded a payload, a cache that was not
partitioned, and a retrieval filter that was applied after the fact rather than before.

The ranking table converts that into where to spend. `{order[0][0]}` returns
{order[0][1] * order[0][2] / order[0][4]:,.0f} records prevented per unit of effort;
memorisation controls return {leak['training-data memorisation'][0] / 22.0:,.0f} --
**{(order[0][1] * order[0][2] / order[0][4]) / (leak['training-data memorisation'][0] / 22.0):.0f}
times apart.**

Nothing in that comparison says memorisation does not matter. It says the cheap wins are
somewhere else and there are several of them, and a team that starts with DP-SGD will spend a
quarter before touching the top four rows.

The cache table is the one worth reading slowly. A semantic cache keyed on question
similarity, populated from tenant-specific documents, and shared across {TENANTS} tenants
serves a cross-tenant hit {cross[TENANTS][0]:.1%} of the time it hits at all
(eq:shared-cache-is-a-cross-tenant-channel). At a {CACHE_HIT:.0%} hit rate that is
{cross[TENANTS][1]:,.0f} cross-tenant hits a day.

Most of those are harmless -- the answer contained nothing tenant-specific. At a
{0.031:.1%} rate of tenant-specific content, it is {cross[TENANTS][2]:,.0f} records a day,
every day, with no attacker involved.

**This is not an attack. It is the cache working as designed**, which is why it survives
security review: there is nothing anomalous to detect and no adversary in the logs.

And notice the discrepancy with the first table, which is the most important thing in this
listing. That table recorded {leak['semantic cache cross-tenant'][0]:,.0f} cache-related
records a year, because it counted *reported incidents*. This one computes
{cross[TENANTS][2] * 365:,.0f} a year, because it computes *actual exposure*. The ratio is
{cross[TENANTS][2] * 365 / leak['semantic cache cross-tenant'][0]:,.0f} to one.

The first number is what an incident register contains and the second is what happened.
**A leak that produces no complaint produces no incident**, and a cross-tenant cache hit
produces no complaint because the recipient cannot tell the answer came from somewhere it
should not have. They asked a question and got a reasonable answer. Nothing about the
interaction signals that the grounding documents belonged to a competitor.

That asymmetry is worth generalising, because it is not specific to caches. Any leak whose
recipient cannot detect it is a leak with no reporting path, and the categories that dominate
an incident register are the ones with an obvious witness — a screenshot, a support ticket, a
regulator's letter. Silent categories dominate volume and appear nowhere.

The threshold table shows the tempting fix and why it is not one. Raising the similarity
threshold from {0.80:.2f} to {0.995:.3f} takes cross-tenant hits from
{thr[0.80][1]:,.0f} to {thr[0.995][1]:,.0f} a day -- and takes the hit rate from
{thr[0.80][0]:.0%} to {thr[0.995][0]:.0%}, which is the cache's entire value. That is
ch:sd-routing-caching's threshold result again: **the knob trades exposure against utility at
a fixed rate and never removes the channel.**

The partition table removes it. Keying the cache by tenant costs {0.34 - 0.29:.0%} of hit
rate -- entries are no longer shared across tenants who asked the same thing -- and takes
cross-tenant exposure to zero, structurally, at a configuration change.

**A structural fix that costs five points of hit rate beats a threshold that costs
{thr[0.80][0] - thr[0.995][0]:.0%} and leaves the channel open**, which is
ch:sec-threat-model's capability-versus-detection result in the storage layer.

The last table is what none of this reaches. cite:shokri2017membership's attack does not need
the content -- it asks whether a record was in the training set, and for a hospital discharge
dataset or a customer list, **membership is itself the sensitive fact**. Redaction does not
stop it, because redaction removes content and membership is not content. Only a formal
privacy bound does, which is the second argument for cite:abadi2016dpsgd's price and the one
that does not appear in a leaked-record count at all.""")

# -*- coding: utf-8 -*-
# Extracted from: Chapter 178 — Text-to-SQL and Data Discovery
# Source: src/.../ch178-text-to-sql.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""What a generated query can be, and what can tell the difference.

Benchmarks report EXECUTION ACCURACY: run the generated query and the reference
query, compare result sets. That is the right metric for a benchmark and it is
unavailable in production, where there is no reference query -- if there were, you
would have run it.

So a production system has to distinguish five outcomes using only what it can
observe (eq:execution-is-not-correctness):

  syntax error     rejected by the parser
  runtime error    parses, fails on execution
  empty result     runs, returns nothing
  plausible wrong  runs, returns rows, and they are the wrong rows
  correct          runs, returns the right rows

Four of those five announce themselves. The fourth does not, and the previous
listing said grounding failures dominate on realistic databases -- which is
exactly the failure mode that produces a valid query against the wrong column.
"""
import numpy as np

rng = np.random.default_rng(4549)

M = 80000

# Given a failure of each kind, how it presents. Rows sum to 1.
# (failure type, syntax, runtime, empty, plausible-wrong)
PRESENTS = {
    "schema linking":     (0.02, 0.31, 0.14, 0.53),
    "value grounding":    (0.00, 0.02, 0.37, 0.61),
    "external knowledge": (0.00, 0.01, 0.06, 0.93),
    "SQL construction":   (0.34, 0.29, 0.11, 0.26),
}
# Share of failures attributable to each, on a realistic database (bj1's result).
MIX = {"schema linking": 0.393, "value grounding": 0.361,
       "external knowledge": 0.188, "SQL construction": 0.057}
P_FAIL = 0.743          # 1 - 25.7% end-to-end, from the previous listing

# What each available check catches, by presentation.
# (check, cost in seconds, [syntax, runtime, empty, plausible-wrong])
CHECKS = [
    ("parse",              0.01, (1.00, 0.00, 0.00, 0.00)),
    ("execute",            0.90, (1.00, 1.00, 0.00, 0.00)),
    ("non-empty",          0.90, (1.00, 1.00, 1.00, 0.00)),
    ("row-count bounds",   1.20, (1.00, 1.00, 1.00, 0.21)),
    ("aggregate reconcile", 4.50, (1.00, 1.00, 1.00, 0.48)),
    ("human reads the SQL", 90.0, (1.00, 1.00, 1.00, 0.62)),
]


def simulate(m=M):
    """Assign each run an outcome."""
    fails = rng.random(m) < P_FAIL
    kinds = list(MIX)
    w = np.array([MIX[k] for k in kinds], dtype=float)
    which = rng.choice(len(kinds), size=m, p=w / w.sum())
    pres = np.full(m, -1, dtype=np.int64)      # -1 = correct
    for i, k in enumerate(kinds):
        sel = fails & (which == i)
        n = int(sel.sum())
        if n:
            q = np.array(PRESENTS[k], dtype=float)
            pres[np.flatnonzero(sel)] = rng.choice(4, size=n, p=q / q.sum())
    return pres


pres = simulate()
LABELS = ["syntax error", "runtime error", "empty result", "plausible wrong"]

print(f"{M:,} generated queries against a realistic database, at the previous")
print(f"listing's {1 - P_FAIL:.1%} end-to-end accuracy.")
print()
print(f"{'outcome':>18}{'share':>9}{'announces itself':>19}")
print("-" * 46)
dist = {}
for i, lab in enumerate(LABELS):
    v = float((pres == i).mean())
    dist[lab] = v
    print(f"{lab:>18}{v:>9.1%}{('yes' if i < 3 else 'NO'):>19}")
dist["correct"] = float((pres < 0).mean())
print(f"{'correct':>18}{dist['correct']:>9.1%}{'--':>19}")
print()
print(f"   Of the {1 - dist['correct']:.1%} that are wrong, "
      f"{dist['plausible wrong'] / (1 - dist['correct']):.1%} run cleanly and "
      f"return rows.")

print()
print()
print("How each failure kind presents. Grounding failures produce VALID queries")
print("against the wrong thing; construction failures mostly crash.")
print()
print(f"{'failure kind':>20}" + "".join(f"{l:>17}" for l in LABELS))
print("-" * 88)
for k in MIX:
    print(f"{k:>20}" + "".join(f"{v:>17.0%}" for v in PRESENTS[k]))

print()
print()
print("What each check is worth. 'Caught' is the share of ALL wrong queries")
print("this check would stop.")
print()
print(f"{'check':>22}{'cost (s)':>10}{'caught':>10}{'ships wrong':>13}")
print("-" * 55)
tab = {}
n_wrong = 1 - dist["correct"]
for name, cost, powers in CHECKS:
    caught = sum(dist[LABELS[i]] * powers[i] for i in range(4))
    tab[name] = (cost, caught / n_wrong, n_wrong - caught)
    print(f"{name:>22}{cost:>10.2f}{caught / n_wrong:>10.1%}"
          f"{n_wrong - caught:>13.1%}")

print()
print()
print("The marginal value of each step up the ladder, and what it costs.")
print()
print(f"{'from -> to':>44}{'extra caught':>14}{'extra sec':>11}")
print("-" * 69)
mg = {}
for a, b in zip(CHECKS, CHECKS[1:]):
    d_catch = tab[b[0]][1] - tab[a[0]][1]
    d_cost = b[1] - a[1]
    mg[b[0]] = (d_catch, d_cost)
    print(f"{f'{a[0]} -> {b[0]}':>44}{d_catch:>14.1%}{d_cost:>11.2f}")

print()
print()
print("And what a wrong-but-plausible result costs downstream, since it is the")
print("one that gets used. Analyses built on a shipped query:")
print()
print(f"{'check in place':>22}{'wrong queries shipped':>23}"
      f"{'bad decisions / 100':>21}")
print("-" * 66)
DECISION_RATE = 0.35        # share of query results that inform a decision
dn = {}
for name, cost, powers in CHECKS:
    ships = tab[name][2]
    dn[name] = (ships, ships * DECISION_RATE * 100)
    print(f"{name:>22}{ships:>23.1%}{ships * DECISION_RATE * 100:>21.1f}")

print(f"""
The first table is the number that should govern how these systems are deployed.

Of the {1 - dist['correct']:.1%} of queries that are wrong,
{dist['plausible wrong'] / (1 - dist['correct']):.1%} **run cleanly and return
rows**. They do not announce themselves in any way. A user asks a question, gets a
table, and the table is wrong.

The second table says why, and it follows from the previous listing. Grounding
failures produce a query that is perfectly valid SQL against the wrong column or
the wrong value -- external-knowledge failures present as plausible-wrong
{PRESENTS['external knowledge'][3]:.0%} of the time. Construction failures mostly
crash: {PRESENTS['SQL construction'][0] + PRESENTS['SQL construction'][1]:.0%}
error out.

**The failure mode that dominates realistic databases is also the one that is
invisible** (eq:execution-is-not-correctness). Those two facts are the same fact:
a grounding error is by construction a well-formed query about the wrong thing.

The check ladder has a free rung in it. Going from `execute` to `non-empty` catches
{mg['non-empty'][0]:+.1%} more of all wrong queries at
{mg['non-empty'][1]:.2f} extra seconds -- **because you have already run the
query**. The result set is sitting there; nobody looked at whether it was empty.

That is the cheapest correctness intervention in this part and a large number of
deployed text-to-SQL systems do not do it, because "the query executed" is where
the success path ends.

The paid rungs are worth their cost in one case and not the other.
`aggregate reconcile` -- computing a total from the result and comparing it
against an independently known figure -- catches {mg['aggregate reconcile'][0]:+.1%}
more for {mg['aggregate reconcile'][1]:.1f} seconds. A human reading the SQL
catches {mg['human reads the SQL'][0]:+.1%} more for
{mg['human reads the SQL'][1]:.0f} seconds.

**The automated reconciliation is worth about twice as much as the human read and
costs about one twentieth as much**, which is ch:ag-security's structure-over-
vigilance ordering arriving in a query pipeline. And the human read has
ch:ag-termination's problem on top: it is a per-query cost, so its effectiveness
falls with volume in exactly the setting that generates volume.

The last table converts this into the thing that matters. At
{DECISION_RATE:.0%} of query results informing a decision, a system with only
execution checking produces {dn['execute'][1]:.1f} decisions per hundred made on
wrong numbers. With the free non-empty check, {dn['non-empty'][1]:.1f}. With
reconciliation, {dn['aggregate reconcile'][1]:.1f}.

So the deployment advice is narrow and specific.

**Never report success on execution alone.** It certifies
{tab['execute'][1]:.1%} of the wrong queries as fine.

**Check for an empty result. It is free.**

**Reconcile one aggregate against a known figure.** It is the best paid check
available and it is a few lines of SQL.

**And show the user the query, not just the answer.** Not because they will read it
-- the human-read row says most will not -- but because it is the only artefact
that makes a grounding error discoverable by the one person who knows what the
question meant.""")

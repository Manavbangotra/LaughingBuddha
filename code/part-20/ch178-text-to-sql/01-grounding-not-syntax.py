# -*- coding: utf-8 -*-
# Extracted from: Chapter 178 — Text-to-SQL and Data Discovery
# Source: src/.../ch178-text-to-sql.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Why realistic text-to-SQL scores 40% when the SQL is not the hard part.

cite:yu2018spider made text-to-SQL a generalisation problem: 10,181 questions over
200 multi-table databases in 138 domains, split so that both the queries and the
SCHEMAS in the test set are unseen. cite:li2023bird then made it a realistic one:
95 databases, 33.4 GB, 37 professional domains -- and reported 40.08% execution
accuracy against 92.96% for humans.

A 53-point gap invites the assumption that generating SQL is hard. This listing
decomposes the task into its four sub-problems and asks which one the gap is in
(eq:grounding-not-syntax):

  schema linking     which tables and columns does this question refer to
  value grounding    what do the cells actually contain -- 'CA', 'Calif.', 'California'
  external knowledge facts not in the schema: what counts as 'active', a fiscal year
  SQL construction   joins, aggregation, subqueries, window functions
"""
import numpy as np

rng = np.random.default_rng(4507)

M = 60000

# (name, success on a clean benchmark schema, success on a realistic database)
SUBPROBLEMS = [
    ("schema linking",     0.93, 0.71),
    ("value grounding",    0.97, 0.62),
    ("external knowledge", 0.95, 0.68),
    ("SQL construction",   0.88, 0.86),
]


def run(profile, m=M, subs=None):
    """profile 0 = clean benchmark, 1 = realistic database."""
    subs = subs or SUBPROBLEMS
    ok = np.ones(m, dtype=bool)
    blame = np.full(m, -1, dtype=np.int64)
    for i, row in enumerate(subs):
        p = row[1 + profile]
        good = rng.random(m) < p
        newly = ok & ~good
        blame[newly] = i
        ok &= good
    return float(ok.mean()), blame


print("Four sub-problems, each with a success rate on a clean benchmark schema")
print("and on a realistic production database.")
print()
print(f"{'sub-problem':>20}{'clean':>9}{'realistic':>12}{'drop':>9}")
print("-" * 50)
for name, a, b in SUBPROBLEMS:
    print(f"{name:>20}{a:>9.0%}{b:>12.0%}{b - a:>+9.0%}")

print()
print()
print("End-to-end, and where the failures come from.")
print()
tot = {}
for profile, label in ((0, "clean benchmark"), (1, "realistic database")):
    acc, blame = run(profile)
    tot[label] = (acc, blame)
    print(f"{label:>22}: {acc:.1%} end-to-end")
print()
print(f"{'first failure at':>20}{'clean':>10}{'realistic':>12}")
print("-" * 42)
share = {}
for i, (name, _, _) in enumerate(SUBPROBLEMS):
    a = float((tot['clean benchmark'][1] == i).mean())
    b = float((tot['realistic database'][1] == i).mean())
    share[name] = (a, b)
    print(f"{name:>20}{a:>10.1%}{b:>12.1%}")

print()
print()
print("As a share of the FAILURES, which is the view that says what to work on.")
print()
print(f"{'first failure at':>20}{'clean':>10}{'realistic':>12}")
print("-" * 42)
fs = {}
for name in share:
    a_tot = 1 - tot['clean benchmark'][0]
    b_tot = 1 - tot['realistic database'][0]
    fs[name] = (share[name][0] / a_tot, share[name][1] / b_tot)
    print(f"{name:>20}{fs[name][0]:>10.1%}{fs[name][1]:>12.1%}")

print()
print()
print("The counterfactual that matters for where to spend effort: fix ONE")
print("sub-problem to its clean-benchmark level, leave the rest realistic.")
print()
base = tot['realistic database'][0]
print(f"{'lifted to clean level':>22}{'end-to-end':>13}{'gain':>9}")
print("-" * 45)
cf = {}
for i, (name, a, b) in enumerate(SUBPROBLEMS):
    subs = [list(r) for r in SUBPROBLEMS]
    subs[i][2] = subs[i][1]
    v = run(1, subs=[tuple(r) for r in subs])[0]
    cf[name] = (v, v - base)
    print(f"{name:>22}{v:>13.1%}{v - base:>+9.1%}")

print()
print()
print("And the grounding group together -- schema linking, value grounding and")
print("external knowledge -- against SQL construction alone.")
print()
grounding = [list(r) for r in SUBPROBLEMS]
for i in range(3):
    grounding[i][2] = grounding[i][1]
g_only = run(1, subs=[tuple(r) for r in grounding])[0]
sqlonly = [list(r) for r in SUBPROBLEMS]
sqlonly[3][2] = sqlonly[3][1]
s_only = run(1, subs=[tuple(r) for r in sqlonly])[0]
print(f"{'realistic, as is':>34}{base:>10.1%}")
print(f"{'SQL construction made perfect':>34}{s_only:>10.1%}"
      f"  ({s_only - base:+.1%})")
print(f"{'grounding made perfect':>34}{g_only:>10.1%}"
      f"  ({g_only - base:+.1%})")

print()
print()
print("How the gap moves with database realism -- interpolating each")
print("sub-problem between its clean and realistic rates.")
print()
print(f"{'realism':>9}{'schema':>9}{'values':>9}{'knowledge':>12}{'SQL':>7}"
      f"{'end-to-end':>13}")
print("-" * 59)
rl = {}
for t in (0.0, 0.33, 0.66, 1.0):
    subs = [(n, a, a + (b - a) * t) for n, a, b in SUBPROBLEMS]
    v = run(1, subs=subs)[0]
    rl[t] = (v, [s[2] for s in subs])
    print(f"{t:>9.0%}" + "".join(f"{s[2]:>{w}.0%}" for s, w in
                                 zip(subs, (9, 9, 12, 7)))
          + f"{v:>13.1%}")

print(f"""
The failure-share table is the chapter in two columns.

On a clean benchmark schema, SQL construction is the largest single source of
failure at {fs['SQL construction'][0]:.1%}. On a realistic database it is
{fs['SQL construction'][1]:.1%}, and the three grounding problems together are
{fs['schema linking'][1] + fs['value grounding'][1] + fs['external knowledge'][1]:.1%}.

The counterfactual makes it decisive. Lifting SQL construction to its
clean-benchmark level is worth {cf['SQL construction'][1]:+.1%}. Lifting the
grounding group is worth {g_only - base:+.1%}, from {base:.1%} to {g_only:.1%}.

**The hard part of text-to-SQL is not SQL** (eq:grounding-not-syntax). Generating
correct SQL from a correct understanding is close to solved; working out what the
question refers to in a database nobody documented is not.

That is worth sitting with, because it reframes what cite:li2023bird's
40.08%-against-92.96% actually reports. It is not a measurement of a model's
command of a query language. It is a measurement of how much of a database's
meaning lives outside its schema -- in column values that use three spellings for
the same state, in a `status` field whose codes were documented in a wiki that
moved, in the fact that this organisation's fiscal year starts in April.

A human analyst scoring 92.96% is not better at SQL. They have worked there for
two years.

The last table shows the gradient. At {0.0:.0%} realism the end-to-end figure is
{rl[0.0][0]:.1%} -- roughly what clean-benchmark results look like -- and at
{1.0:.0%} it is {rl[1.0][0]:.1%}. The SQL column barely moves across that entire
range: {rl[0.0][1][3]:.0%} to {rl[1.0][1][3]:.0%}.

**Progress measured on clean schemas transfers to realistic databases only in the
component that was never the bottleneck.** Which is why cite:yu2018spider's
database split -- unseen schemas at test time -- was the right design and still
was not enough: the schemas were unseen but they were clean.

Two practical readings follow.

**Effort spent on the model's SQL ability is nearly wasted.** The counterfactual
puts it at {cf['SQL construction'][1]:+.1%}.

**Effort spent on grounding is not, and most of it is not modelling work.** Column
descriptions, value dictionaries, documented business definitions, a sample of
actual cell contents in the prompt -- these are the interventions that move
{cf['value grounding'][1]:+.1%} and {cf['external knowledge'][1]:+.1%}, and they
are database documentation rather than machine learning.""")

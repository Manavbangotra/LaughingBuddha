---
id: aids-text-to-sql
number: 178
part: XX
tier: full
status: draft
requires: [gradeable-is-not-representative, check-strong-build-weak,
           errors-are-the-interface, decision-cost-versus-dilution]
provides: [grounding-not-syntax, meaning-lives-outside-the-schema,
           execution-is-not-correctness, silent-failure-dominates,
           free-check-before-paid-check]
citations: [yu2018spider, li2023bird, huang2024dacode, testini2025dsautomation,
            greshake2023indirect]
---

## 1. Learning Objectives

By the end of this chapter you will be able to decompose text-to-SQL into its four
sub-problems and say which one the accuracy gap lives in; explain why a human
analyst's $92.96\%$ is not a statement about their SQL ability; distinguish
execution accuracy from correctness and say why the distinction is unavailable in
production; identify which failure modes announce themselves and which do not; and
order the available checks by marginal catch rate per second, including the one
that is free.

## 2. Why This Matters

Text-to-SQL is the best-measured activity in {{ch:aids-stack}}'s stack, which makes
it the right place to see what a benchmark number does and does not say.

{{cite:yu2018spider}} made it a generalisation problem: $10{,}181$ questions over
$200$ multi-table databases across $138$ domains, split so that both the queries
*and the schemas* in the test set are unseen. {{cite:li2023bird}} then made it a
realistic one — $95$ databases, $33.4$ GB, $37$ professional domains — and reported
**$40.08\%$ execution accuracy against $92.96\%$ for humans.**

A fifty-three point gap invites the reading that generating SQL is hard.
{{sec:9-practical-example}} decomposes the task and finds otherwise. On a clean
benchmark schema, SQL construction is the largest source of failure at $41.9\%$;
on a realistic database it is $5.7\%$, and the three *grounding* problems — which
tables and columns, what the cells actually contain, what the business terms mean —
account for $94.2\%$.

The counterfactual is decisive: making SQL construction perfect is worth $+0.4$
points, and making grounding perfect is worth $+47.8$
({{eq:grounding-not-syntax}}). **The hard part of text-to-SQL is not SQL.**

Which reframes the human baseline. A person scoring $92.96\%$ is not better at
writing queries. They have worked there for two years and know that `status = 3`
means cancelled ({{eq:meaning-lives-outside-the-schema}}).

The second half asks what a production system can do about it, and finds a worse
problem. Execution accuracy compares against a reference query, which production
does not have. Using only what can be observed, **$61.4\%$ of wrong queries run
cleanly and return rows** ({{eq:silent-failure-dominates}}) — and that is not a
coincidence, because a grounding error *is* a well-formed query about the wrong
thing.

## 3. Prerequisites

{{ch:aids-stack}}'s {{eq:check-strong-build-weak}} and its gradeability argument —
this chapter examines the most gradeable activity in the stack and finds the metric
still does not mean what it appears to.

{{ch:ag-tool-calling}}'s {{eq:error-message-as-selector}}, since a database error is
a selector and most of them are terrible ones.

{{ch:mcp-primitives}}'s {{eq:decision-cost-versus-dilution}}, because what goes into
the prompt — schema, sample values, documentation — is the same context-budget
question in a new setting.

Working knowledge of SQL and relational schemas is assumed.

## 4. Intuitive Explanation

Someone asks: *how many active enterprise customers did we add in Q3?*

To answer it in SQL you need four different things, and only one of them is SQL.

**Which tables and columns.** Is it `customers`, or `accounts`, or both joined
through `organisations`? Is "added" the `created_at` on the account or the
`activated_on` on the subscription? A benchmark schema with twelve well-named
tables makes this easy. A production warehouse with four hundred tables, six of
which are called some variant of `customer`, does not.

**What the values are.** "Enterprise" is a tier. Is it stored as `'enterprise'`,
`'ENTERPRISE'`, `'ent'`, or `tier_id = 3`? Nothing in the schema says. You have to
look, and looking at four hundred tables' worth of distinct values is not something
that fits in a prompt.

**What the words mean here.** "Active" might mean has a current subscription, or
logged in within thirty days, or is not marked churned — and the organisation has a
definition, written down somewhere or not. "Q3" depends on when the fiscal year
starts.

**The SQL itself.** A join, a date range, a count, maybe a subquery.

Models are good at the fourth. {{sec:9-practical-example}} puts SQL construction at
$86\%$ on realistic databases and $88\%$ on clean ones — barely different, because
the query language does not get harder when the database gets messier.

Everything else falls apart. Value grounding goes from $97\%$ to $62\%$. And
because the four compose multiplicatively, the end-to-end number collapses.

This is why the human baseline is so high. The analyst is not better at SQL. They
know the tier codes, they know what active means here, and they know that the
`customers` table was deprecated eighteen months ago and everyone uses
`dim_customer` now. **The meaning lives outside the schema**, and they have it.

Now the production problem, which is worse than the benchmark problem.

A benchmark grades by *execution accuracy*: run the generated query, run the
reference query, compare results. In production there is no reference query. If
you had one you would have run it.

So the system has to work out whether its answer is right from what it can see. And
{{sec:9-practical-example}} finds that most wrong answers look exactly like right
ones: a grounding error produces a valid query against the wrong column, which runs
without complaint and returns a plausible number.

$61.4\%$ of the failures are of that shape. The user asks a question, gets a table,
and it is wrong.

There is a ladder of checks available and it has one free rung on it. Once the
query has executed you have the result set in hand — and nobody looks at whether it
is empty. Checking that catches twenty-one points of wrong queries at zero
additional cost, and a great many deployed systems do not do it, because "the query
ran" is where the success path ends.

## 5. Formal Explanation

**Decomposition.** Let a correct answer require all four sub-problems solved, with
success probabilities $s_1..s_4$ depending on the database profile:

$$A = \prod_{i=1}^{4} s_i$$

Writing $s_i^{c}$ for clean-benchmark rates and $s_i^{r}$ for realistic ones, the
value of lifting sub-problem $j$ to its clean rate is:

$$\Delta_j = \Big(\prod_{i \ne j} s_i^{r}\Big)\big(s_j^{c} - s_j^{r}\big)$$ (eq:grounding-not-syntax)

which scales with the *gap* $s_j^c - s_j^r$, not with the level. SQL construction
has $s^c - s^r = 0.02$; value grounding has $0.35$. **The sub-problem that is
already reliable contributes nothing to the gap regardless of how central it looks.**

**Why the gap sits where it does.** A schema is a syntactic object: table names,
column names, types, keys. The information required to answer a question includes
the *extension* — what values actually occur — and the *intension* — what the
organisation means by its terms. Neither is in the schema:

$$I_{\text{needed}} = I_{\text{schema}} \cup I_{\text{values}} \cup I_{\text{convention}}$$ (eq:meaning-lives-outside-the-schema)

A benchmark database is constructed so $I_{\text{schema}}$ nearly suffices. A
production database is not, and the shortfall is what a tenured analyst supplies.

**Execution accuracy versus correctness.** A benchmark computes:

$$\text{EA} = \Pr\big[R(q_{\text{gen}}) = R(q_{\text{ref}})\big]$$

which requires $q_{\text{ref}}$. In production the observable is only
$R(q_{\text{gen}})$ and its metadata. Partition the outcomes by what is observable:

$$\Omega = \{\text{syntax}, \text{runtime}, \text{empty}, \text{plausible-wrong}, \text{correct}\}$$

The first three are self-announcing; the last two are observationally identical
without a reference:

$$\Pr[\text{correct} \mid \text{ran and returned rows}] = \frac{\pi_c}{\pi_c + \pi_{pw}}$$ (eq:execution-is-not-correctness)

**The dominance of the silent case follows from the decomposition.** A grounding
error yields a syntactically valid query over existing columns, so it lands in
`plausible-wrong` with high probability; a construction error yields malformed or
type-invalid SQL, so it crashes:

$$\Pr[\text{plausible-wrong} \mid \text{grounding}] \gg \Pr[\text{plausible-wrong} \mid \text{construction}]$$ (eq:silent-failure-dominates)

Combining with {{eq:grounding-not-syntax}}: the failure class that dominates
realistic databases is the class that is invisible. Those are the same fact.

**The check ladder.** A check $k$ catches presentation $p$ with power $\kappa_{kp}$
at cost $c_k$. Ordering checks by marginal catch per unit cost:

$$\rho_k = \frac{\sum_p \pi_p (\kappa_{kp} - \kappa_{k-1,p})}{c_k - c_{k-1}}$$ (eq:free-check-before-paid-check)

For the empty-result check, $c_k - c_{k-1} = 0$ — the query has already run — so
$\rho \to \infty$. **A check whose input you already have should always be
performed**, and it is the only rung on the ladder where the ordering question does
not arise.

## 6. Mathematical Foundation

Three extractions.

**Gaps, not levels, decide where to invest.**
{{eq:grounding-not-syntax}}'s $\Delta_j$ is proportional to $s_j^c - s_j^r$. This is
why "the model is only $86\%$ at SQL construction, we should improve that" is the
wrong inference: it is $86\%$ on both profiles, so it explains none of the
difference between them.

**The invisible failure mode is entailed, not incidental.** From
{{eq:silent-failure-dominates}}, the shift toward grounding failures on realistic
databases *is* a shift toward silent failures, because those are the same events
viewed two ways. A system that improves its handling of realistic data without
changing its verification will therefore see its visible error rate fall and its
true error rate fall less.

**A free check dominates the ordering.**
{{eq:free-check-before-paid-check}} with $\Delta c = 0$ makes the empty-result check
unconditionally correct to perform — no threshold, no trade-off. That such checks
go unimplemented is a fact about where the success path was drawn, not about
economics.

## 7. Internal Mechanics

### 7.1 The four sub-problems, and what fixes each

```mermaid {#fig:sql-subproblems caption="The four sub-problems of text-to-SQL. Only the last is about SQL, and only the last is nearly solved."}
flowchart TD
    Q[natural language question] --> SL["schema linking<br/>which tables, which columns"]
    SL --> VG["value grounding<br/>what do the cells contain"]
    VG --> EK["external knowledge<br/>what do the terms mean here"]
    EK --> SC["SQL construction<br/>joins, aggregation, windows"]
    SC --> R[query]
    D1[(column descriptions)] -.-> SL
    D2[(value dictionaries,<br/>sampled cells)] -.-> VG
    D3[(metric definitions,<br/>business glossary)] -.-> EK
```

The dotted inputs are the interventions that matter, and none of them is model
work. Column descriptions, a dictionary of distinct values for low-cardinality
columns, a written definition of what the organisation means by "active" — these
are **documentation**, and {{sec:9-practical-example}} prices them at $+47.8$ points
against $+0.4$ for a better SQL generator.

### 7.2 Sampled values are the highest-return prompt content

The single most effective grounding intervention is also the most mechanical:
include actual sample values for the columns under consideration.

A schema line saying `tier VARCHAR(32)` tells the model nothing about whether to
write `'enterprise'` or `'ENT'`. A line saying `tier VARCHAR(32) -- e.g. 'ENT',
'PRO', 'FREE'` settles it. For low-cardinality columns the full domain fits; for
high-cardinality ones a handful of examples establishes the format.

This has a context-budget cost and therefore lands in
{{ch:mcp-primitives}}'s trade: sampled values are preloaded content, they dilute,
and they go stale when the data changes. The volatility threshold from that chapter
applies — a `status` enumeration is stable and belongs in the prompt; a set of
current customer names is not and does not.

### 7.3 Schema linking at scale is a retrieval problem

A production warehouse has more tables than fit in a context window, so schema
linking becomes {{ch:mcp-schemas}}'s problem exactly: retrieve the relevant subset,
show only that.

The same results apply. Showing fewer, better-described tables beats showing more
tersely-described ones, and retrieval recall becomes the thing to measure — when
the right table is not retrieved, the model produces a confident query against a
plausible wrong table, which lands in the silent-failure bucket.

**Schema retrieval recall is therefore a direct input to the silent error rate**,
and it is measurable against questions whose correct tables are known.

### 7.4 Database errors as selectors

{{ch:ag-tool-calling}} found error messages to be the cheapest reliability
mechanism available, and database engines are a natural experiment in getting this
wrong.

`ERROR: column "customer_tier" does not exist` is a good error: it names the
missing thing. Some engines add `HINT: Perhaps you meant "customer_tier_id"`, which
is close to ideal — it names the alternative.

`ERROR: syntax error at or near ")"` is not. Nor is a silent empty result, which is
what a filter on a non-existent value produces.

The practical consequence: **wrap the database's errors before returning them to
the model**, adding the schema context the engine has and does not include —
available columns on the referenced table, distinct values for a filtered column
that matched nothing. That converts an uninformative failure into
{{eq:error-message-as-selector}}'s informative one, and it is server-side work of
the kind {{ch:mcp-building}} argued has the best leverage.

### 7.5 The empty result is the most under-used signal

An empty result set is not an error, and it is very often a symptom.
{{sec:9-practical-example}} attributes $37\%$ of value-grounding failures to it:
the query filtered on `tier = 'enterprise'` and the column contains `'ENT'`.

Treating empty as suspicious costs nothing — the query has run — and it converts a
silent failure into an announced one for a substantial fraction of exactly the
error class that dominates. The right handling is not to error but to *investigate*:
report the empty result along with the distinct values actually present in the
filtered columns, which both explains the failure and supplies the grounding the
retry needs.

### 7.6 Aggregate reconciliation, the best paid check

{{sec:9-practical-example}} puts automated aggregate reconciliation at $+16.6$
points for $3.3$ seconds against a human SQL review's $+8.6$ for $85.5$ seconds —
roughly twice the catch at one twenty-fifth the cost.

The mechanism is simple and worth implementing directly. Given a result, compute
one independent quantity that should agree with something already known — a total
row count against a known table size, a sum against a published figure, a
this-period number against last period's. Disagreement beyond a tolerance is a
strong signal that the query addressed the wrong population.

It works because it checks a *property of the answer* rather than a property of the
query, and grounding errors change the population while leaving the query
well-formed. It is the closest thing text-to-SQL has to
{{ch:as-specialized}}'s executable verifier, and it is why this activity scores as
well as it does relative to the ungradeable middle of {{ch:aids-stack}}'s pipeline.

### 7.7 Showing the query is not the same as review

{{sec:9-practical-example}} says a human reading the SQL is poor value per second,
and the recommendation to show the user the query anyway is not a contradiction.

The human-read row prices *systematic review* — a person checking every query. That
is expensive, and {{ch:ag-termination}}'s habituation says it degrades with volume,
which a query interface generates.

Displaying the query is different. It costs nothing, it is not a gate, and it makes
the error *discoverable* by the one person who knows what the question meant — not
on every query, but on the one where the number looks surprising. That is
{{ch:as-roles}}'s advise-rather-than-gate distinction: an advisory artefact has a
floor and a mandatory review does not.

And it has a second effect the listing does not model. A user who sees
`WHERE tier = 'enterprise'` and knows the codes are three-letter can correct the
system's grounding permanently, which is the only mechanism here that improves
$I_{\text{convention}}$ rather than working around its absence.

### 7.8 The semantic layer is the structural answer

{{sec:13-alternatives}} lists a semantic layer as an option and it deserves more
than a line, because it is the only intervention in this chapter that changes the
problem rather than working around it.

{{eq:meaning-lives-outside-the-schema}} decomposes the required information into
schema, values and convention, and says the last two are absent from what the model
sees. Every recommendation so far — sampled values, column descriptions, metric
glossaries — supplies that information *to the prompt*, which means supplying it
again on every query, keeping it fresh against a database that moves, and paying
{{ch:mcp-schemas}}'s rent for it.

A semantic layer supplies it *to the schema*. Metrics are defined once as objects —
`active_customers` is a definition, not a filter someone re-derives — and the model
targets those definitions rather than raw tables. Convention stops being external
information and becomes part of what the query language can express.

That collapses the two largest sub-problem gaps at once. Value grounding mostly
disappears, because the semantic layer's dimensions have declared domains rather
than whatever the underlying column happens to contain. External knowledge mostly
disappears, because the definition *is* the knowledge, written down and versioned.
What remains is schema linking over a much smaller and better-named object set, and
SQL construction, which was never the problem.

The cost is real and is the reason this is not universal advice. Someone has to
build and maintain the layer, which is the same documentation work
{{sec:10-production-considerations}} recommends, done once in an executable form
rather than repeatedly in a prompt. Organisations that already have one — a metrics
store, a well-maintained dbt project, a BI semantic model — are in a substantially
better position for text-to-SQL than their raw warehouse suggests, and often do not
realise it is the relevant asset.

The general form of the point, which recurs in {{ch:aids-agentic-eda}}: **when the
missing information is convention, the durable fix is to write the convention down
somewhere executable**, not to get better at guessing it.

## 8. Implementation

Two listings. The first decomposes the accuracy gap. The second asks what a
production system can observe.

```python {tier=A name=grounding-not-syntax}
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
```

The second listing asks what can tell a wrong query from a right one.

```python {tier=A name=execution-is-not-correctness}
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
```

## 9. Practical Example

The first listing assigns each sub-problem a clean-benchmark and a realistic
success rate:

```
         sub-problem    clean   realistic     drop
--------------------------------------------------
      schema linking      93%         71%     -22%
     value grounding      97%         62%     -35%
  external knowledge      95%         68%     -27%
    SQL construction      88%         86%      -2%
```

End-to-end this gives $75.4\%$ clean and $25.7\%$ realistic. As a share of
failures:

```
    first failure at     clean   realistic
------------------------------------------
      schema linking     28.1%       39.3%
     value grounding     11.9%       36.1%
  external knowledge     18.1%       18.8%
    SQL construction     41.9%        5.7%
```

**SQL construction goes from the largest source of failure to a rounding error.**
The counterfactual:

```
                  realistic, as is     25.7%
     SQL construction made perfect     26.2%  (+0.4%)
            grounding made perfect     73.5%  (+47.8%)
```

$+0.4$ against $+47.8$ ({{eq:grounding-not-syntax}}). And across the realism
gradient the SQL column barely moves — $88\%$ to $86\%$ — while end-to-end falls
from $75.4\%$ to $25.7\%$. **Progress measured on clean schemas transfers only in
the component that was never the bottleneck.**

The second listing takes that failure mix and asks how it presents:

```
           outcome    share   announces itself
----------------------------------------------
      syntax error     2.1%                yes
     runtime error    11.0%                yes
      empty result    15.6%                yes
   plausible wrong    45.6%                 NO
           correct    25.7%                 --
```

**Of the wrong queries, $61.4\%$ run cleanly and return rows**
({{eq:silent-failure-dominates}}). Why:

```
        failure kind     syntax error    runtime error     empty result  plausible wrong
----------------------------------------------------------------------------------------
      schema linking               2%              31%              14%              53%
  external knowledge               0%               1%               6%              93%
    SQL construction              34%              29%              11%              26%
```

Grounding failures produce valid queries against the wrong thing; construction
failures crash. **The failure mode that dominates realistic databases is the one
that is invisible, and those are the same fact.**

The check ladder:

```
                 check  cost (s)    caught  ships wrong
-------------------------------------------------------
                 parse      0.01      2.8%        72.2%
               execute      0.90     17.6%        61.2%
             non-empty      0.90     38.6%        45.6%
      row-count bounds      1.20     51.5%        36.0%
   aggregate reconcile      4.50     68.1%        23.7%
   human reads the SQL     90.00     76.7%        17.3%
```

Marginally:

```
                                  from -> to  extra caught  extra sec
---------------------------------------------------------------------
                        execute -> non-empty         21.0%       0.00
     row-count bounds -> aggregate reconcile         16.6%       3.30
  aggregate reconcile -> human reads the SQL          8.6%      85.50
```

**Checking for an empty result catches $21.0$ points at zero additional cost**
({{eq:free-check-before-paid-check}}) — the query has already run. And automated
reconciliation catches roughly twice what a human SQL review catches at one
twenty-fifth the cost, which is {{ch:ag-security}}'s ordering again.

Downstream, at $35\%$ of results informing a decision:

```
        check in place  wrong queries shipped  bad decisions / 100
------------------------------------------------------------------
               execute                  61.2%                 21.4
             non-empty                  45.6%                 16.0
   aggregate reconcile                  23.7%                  8.3
```

## 10. Production Considerations

Invest in grounding, not in SQL. Column descriptions, value dictionaries and
written metric definitions are worth $+47.8$ points against $+0.4$ for a better
generator — and they are documentation work.

Put sampled values in the prompt for low-cardinality columns, subject to
{{ch:mcp-primitives}}'s volatility threshold: stable enumerations yes, live data
no.

Treat schema linking as retrieval and measure its recall directly. A missed table
becomes a confident query against a plausible wrong one.

Never report success on execution alone — it certifies $61\%$ of wrong queries as
fine.

Check for empty results. It is free and it catches twenty-one points.

On an empty result, return the distinct values actually present in the filtered
columns. That explains the failure and supplies the grounding for the retry.

Wrap database errors with schema context before returning them to the model.

Implement one aggregate reconciliation. It is the best paid check available and it
is a few lines of SQL.

And display the generated query to the user — as an advisory artefact rather than a
gate, since it is the only mechanism that lets a grounding error be corrected
permanently.

## 11. Common Mistakes

**Reading the accuracy gap as a SQL-generation gap.** It is $5.7\%$ of realistic
failures.

**Improving the model to fix grounding.** The information is not in the model's
input; adding capability does not add facts.

**Reporting execution as success.** The dominant failure mode executes.

**Not checking for empty results.** Free, and twenty-one points.

**Returning raw database errors.** An uninformative selector where an informative
one was available.

**Gating on human SQL review.** Poor value per second, and it habituates.

**Preloading volatile sample values.** {{ch:mcp-primitives}}'s staleness, in the
prompt.

## 12. Failure Modes

*Plausible wrong answer.* The characteristic failure — valid query, wrong column,
believable number, no error anywhere.

*Silent empty result.* A filter matching nothing, reported as "no results found"
rather than as a probable grounding error.

*Stale convention.* A metric definition that changed, with the system still using
the old one and nothing indicating it.

*Wrong-table confidence.* Schema retrieval missing the right table, producing a
query against a similar one.

*Injection through data.* {{cite:greshake2023indirect}}'s vector via cell contents
surfaced into the prompt as sampled values — a real cost of the highest-value
grounding intervention.

## 13. Alternatives

**A semantic layer.** Define metrics once, in one place, and let the model target
the semantic layer rather than raw tables. This attacks
{{eq:meaning-lives-outside-the-schema}} directly by moving convention *into* the
schema, and it is the strongest structural answer available.

**Curated query templates.** {{ch:ag-what-is-an-agent}}'s router: enumerate the
common questions, parameterise them, cover the head safely and refuse the tail.

**Retrieval over past queries.** An organisation's query log encodes convention;
retrieving similar answered questions supplies grounding no schema contains.

**Human-in-the-loop on the query.** Correct where consequences are high, subject to
{{ch:ag-termination}}'s volume caveat.

**Not exposing free-form querying.** Defensible when the silent failure rate is
unacceptable and no reconciliation is available.

## 14. Evaluation

Measure your four sub-problems separately. End-to-end accuracy hides which one is
binding, and they need entirely different fixes.

Measure the presentation mix — of your wrong queries, how many executed and
returned rows. That number decides your whole verification strategy and nobody
collects it.

Measure schema retrieval recall against questions with known correct tables.

Report accuracy on *your* database, not on a benchmark. The realism gradient in
{{sec:9-practical-example}} spans $75.4\%$ to $25.7\%$ on identical sub-problem
skills.

Track empty results as a category and audit a sample for grounding errors.

And measure how often a displayed query led a user to correct the system. That is
the only feedback loop here that improves the underlying problem.

## 15. Advanced Concepts

**Automatic value dictionary construction.** Profiling low-cardinality columns and
maintaining a freshness-aware dictionary is mechanical and directly attacks the
largest sub-problem gap. {{maturity:EMERGING}}.

**Convention mining from query logs.** Inferring what "active" means from how
analysts have historically filtered, rather than from documentation nobody wrote.

**Reconciliation target discovery.** Automatically identifying quantities that
should agree, so {{sec:7-internal-mechanics}}'s best paid check does not require a
human to specify it per query.

**Calibrated abstention.** A system that says "I am not confident I have the right
tier codes" converts a silent failure into an announced one, and calibration on
grounding uncertainty specifically is not studied. {{maturity:RESEARCH FRONTIER}}.

## 16. Connection to Previous Chapters

{{ch:aids-stack}}'s check-strong-build-weak rule applies within this chapter:
execution and reconciliation are the strong verifiers, and the weak one — knowing
whether the question was understood — is where the residual sits.

{{ch:mcp-schemas}}'s retrieval results transfer directly to schema linking at
warehouse scale, including the finding that fewer well-described candidates beat
more terse ones.

{{ch:mcp-primitives}}'s volatility threshold decides which sample values belong in
a prompt.

{{ch:ag-tool-calling}}'s error-message result explains why wrapping database errors
is worth doing, and {{ch:ag-security}}'s ordering explains why reconciliation beats
review.

Ahead: {{ch:aids-agentic-eda}} moves into the part of the pipeline with no
verifier at all, where this chapter's reconciliation trick has no analogue.

## 17. Exercises

1. Measure the four sub-problem rates on your own database by constructing
   questions that isolate each.

2. Add sampled values to the prompt for low-cardinality columns and measure the
   change in value-grounding failures specifically.

3. Implement the empty-result investigation — return distinct values from filtered
   columns — and measure recovery on retry.

4. Build one aggregate reconciliation and measure its catch rate against seeded
   grounding errors.

5. Model a semantic layer in the first listing: convention moved into the schema.
   How much of the grounding gap does it close?

6. Measure schema retrieval recall and relate it to your plausible-wrong rate.

## 18. Interview Questions

1. Text-to-SQL scores $40\%$ on realistic databases. Where is the gap?

2. Why does a human analyst score $93\%$, and what would you have to give a model
   to match it?

3. Your system reports $95\%$ of queries execute successfully. What have you
   learned?

4. What is the cheapest check you are probably not doing?

5. A user gets an empty result. What should the system return?

6. Would you have a human review generated SQL?

## 19. Research Questions

1. Can value dictionaries be maintained automatically with adequate freshness?

2. Can organisational convention be mined from query logs reliably enough to
   substitute for documentation?

3. Can reconciliation targets be discovered rather than specified?

4. Is grounding uncertainty separable from other uncertainty well enough to support
   calibrated abstention?

5. How much of the human baseline is tenure, and does it transfer to a new analyst
   faster than to a model?

## 20. Chapter Summary

{{cite:li2023bird}} reported $40.08\%$ execution accuracy against $92.96\%$ human
on realistic databases, and the gap is not where it appears to be. Decomposing the
task, SQL construction accounts for $41.9\%$ of failures on a clean benchmark
schema and $5.7\%$ on a realistic database; the three grounding problems account
for $94.2\%$. Making SQL construction perfect is worth $+0.4$ points and making
grounding perfect is worth $+47.8$ ({{eq:grounding-not-syntax}}).

**The hard part of text-to-SQL is not SQL.** The human baseline reflects tenure
rather than skill: the analyst knows the tier codes, the metric definitions and
which table was deprecated. **Meaning lives outside the schema**
({{eq:meaning-lives-outside-the-schema}}), and the interventions that close the gap
— column descriptions, value dictionaries, written metric definitions — are
documentation rather than modelling.

Production makes it worse, because execution accuracy needs a reference query that
production does not have. Using only observables, **$61.4\%$ of wrong queries run
cleanly and return rows** ({{eq:silent-failure-dominates}}) — and that is entailed
rather than incidental, since a grounding error *is* a well-formed query about the
wrong thing.

The check ladder has a free rung: **checking for an empty result catches $21.0$
points at zero additional cost**, because the query has already run
({{eq:free-check-before-paid-check}}). Automated aggregate reconciliation catches
$+16.6$ more for $3.3$ seconds; a human reading the SQL catches $+8.6$ more for
$85.5$ — roughly half the value at twenty-five times the cost, which is
{{ch:ag-security}}'s structure-over-vigilance ordering in a query pipeline.

At $35\%$ of results informing a decision, the difference between execution-only
checking and reconciliation is $21.4$ against $8.3$ bad decisions per hundred.

## 21. Further Reading

{{cite:yu2018spider}} for the database-split design that made this a generalisation
problem, and {{cite:li2023bird}} for the realism that revealed where the difficulty
actually is — read its dirty-value and external-knowledge discussion directly.

{{cite:huang2024dacode}} for what happens when text-to-SQL is one stage of an agent
task rather than the whole task, and {{cite:testini2025dsautomation}} for why this
activity is so much better measured than its neighbours.

{{ch:mcp-schemas}} for schema linking as retrieval, {{ch:mcp-primitives}} for what
belongs in the prompt, and {{ch:ag-tool-calling}} for why database error messages
are worth rewriting.

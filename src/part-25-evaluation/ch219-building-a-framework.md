---
id: ev-framework
number: 219
part: XXV
tier: full
status: draft
requires: [rework-cost-is-set-by-detection-lateness, format-check-is-the-cheapest-gate,
           most-rag-failures-are-invisible-end-to-end, outcome-evaluation-credits-lucky-trajectories]
provides: [coverage-is-a-union-not-a-sum, optimal-portfolio-is-mostly-reference-free,
           gate-placement-is-set-by-cost-times-escape, a-flaky-gate-has-a-blocking-threshold]
citations: [ribeiro2020checklist, breck2017, card2020power, es2023ragas]
---

## 1. Learning Objectives

By the end of this chapter you will be able to compute an evaluation portfolio's coverage as
a union over overlapping instruments rather than a sum; build the portfolio greedily by
marginal coverage per dollar and read the resulting budget frontier; predict that the optimal
portfolio is dominated by reference-free instruments and explain why; place each gate at the
stage minimising run cost plus escape cost, and explain why "as early as possible" is wrong
for expensive checks; compute the false-positive rate above which a gate should be advisory
rather than blocking; and total a framework's gate hours against the delivery loop's period.

## 2. Why This Matters

Every chapter in this part produced an instrument. This one assembles them, and the assembly
has arithmetic of its own.

Coverage is a union, not a sum. Nine instruments whose individual coverages total **1.827**
cover **0.956** together ({{eq:coverage-is-a-union-not-a-sum}}) — **the overlap exceeds any
single instrument's coverage**, because the cheap instruments all point at the same visible
failures.

Built greedily, the portfolio has a predictable shape. At a budget of **250 per thousand
items** it reaches **0.776** coverage, and **100% of that budget goes to instruments that
need no reference at all** ({{eq:optimal-portfolio-is-mostly-reference-free}}). The frozen
regression suite and the human spot-check — the two things most teams build first — do not
make the cut until much later, and the last **0.180** of coverage costs **39×** what the
first 0.776 did.

Placement is a separate problem with a different answer. A gate on every commit runs **140**
times a release against **4** at pre-deploy, so the same instrument costs **35×** more
early; and escape multipliers run from **1.0** to **17.0**
({{eq:gate-placement-is-set-by-cost-times-escape}}). Cheap gates want to be early and
expensive ones late — a 4,100-per-run instrument on every commit costs **574,000** a release.

And a gate blocks only while its false-positive rate is below its own threshold:
**14.9%** for the judge ensemble, **50.7%** for a schema check, **10.1%** for a human
spot-check ({{eq:a-flaky-gate-has-a-blocking-threshold}}).

Finally, the total: **322 gate hours per release** against {{ch:ops-lifecycle}}'s 847-hour
period — **38%** of the loop.

## 3. Prerequisites

{{eq:rework-cost-is-set-by-detection-lateness}} from {{ch:ops-lifecycle}} supplies the escape
multipliers, and its correction — shorten the return trip rather than blindly detecting
earlier — is what {{sec:9-practical-example}}'s placement result reproduces from the cost
side.

{{eq:format-check-is-the-cheapest-gate}} from {{ch:ops-prompt-versioning}} is the first row of
this chapter's greedy build, and it arrives at the same conclusion by a different route.

{{eq:most-rag-failures-are-invisible-end-to-end}} from {{ch:ev-rag}} and
{{eq:outcome-evaluation-credits-lucky-trajectories}} from {{ch:ev-agents}} are the two
strongest arguments for a portfolio rather than a score: a single verdict cannot distinguish
failures that need different fixes.

{{cite:breck2017}}'s readiness rubric is the closest prior art for a checklist of this kind,
and {{cite:ribeiro2020checklist}}'s behavioural testing is where the invariant style comes
from.

## 4. Intuitive Explanation

By this point in the part there are nine instruments on the table. Schema and format checks.
Invariant suites. Execution and test grading. Faithfulness judges. Utilisation probes. Judge
ensembles run in both orders. Frozen regression suites. Sufficiency annotation. Human
spot-checks.

The question is which ones to build, in what order, and where to put them.

The first mistake is arithmetic. Each instrument covers some share of failures; add them up
and you get 1.827, which is more than all the failures there are. Coverage is a union: a
failure caught by two instruments is caught once.

That would be pedantic except the overlap is enormous — the nine together cover 0.956, so
**the overlap is bigger than any individual instrument's coverage.** And it is not randomly
distributed. The cheap instruments all point at the same easy failures: malformed output,
invalid tool arguments, obvious regressions. The expensive ones point at everything, including
the easy failures. So the marginal value of adding an instrument depends entirely on what is
already there, and the ordering matters more than the selection.

Build greedily — always add the instrument with the best marginal coverage per dollar — and
the order comes out: schema and format check first, at 363 points of coverage per thousand
dollars. Execution grading second. Faithfulness judge third. Invariant suite fourth.
Utilisation probe fifth. Judge ensemble sixth. Then, far behind, the frozen regression suite,
the human spot-check, and sufficiency annotation.

Two things about that ordering are worth sitting with.

**The optimal portfolio at any realistic budget is entirely reference-free.** At 250 per
thousand items, every instrument selected — schema check, execution grading, faithfulness
judge, invariant suite, utilisation probe, judge ensemble — needs no ground-truth answer, no
reference trajectory, no labelled relevance set. That is not an accident.
{{ch:ev-why-hard}}'s escape from the reference-sampling problem was to state a predicate
instead of writing an answer, and a predicate is cheap for exactly the reason it is sound: it
needs nothing external.

**And the things teams build first come last.** A frozen regression suite is what evaluation
looks like inside a codebase, so it gets built. A human review process is what quality looks
like inside an organisation, so it gets built. Neither is wrong — both are late in the
ordering, and a team that builds them first spends most of its budget before touching the
instruments with the best returns.

The budget frontier is the shape of every evaluation programme. At 250 per thousand items,
coverage is 0.776. At 9,000 — thirty-six times the budget — it is 0.956. **The last 0.180 of
coverage costs 39× what the first 0.776 did.** That is worth knowing before the argument
about evaluation budget rather than during it: the framework is cheap and the last tenth is
not.

What remains uncovered at a realistic budget is instructive. The largest residual is
retrieval insufficiency, which needs sufficiency annotation at 4,800 per thousand. The next
two are wrong-but-well-formed facts and wrong specificity, which need a human spot-check at
3,400.

Which is the argument for buying a *slice* of the expensive instruments rather than skipping
them. A few hundred annotated queries a quarter gates nothing and measures the blind spot of
everything else. A portfolio without them has an *unmeasured* residual rather than a small
one, and the difference between those two situations is the whole reason to own the table.

That is what to build. Where to put it is a separate problem with a separate answer.

An instrument at the commit stage runs on every commit — 140 times a release here. The same
instrument at pre-deploy runs four times. So its cost is 35× higher early. Meanwhile a defect
escaping past the commit stage costs the base amount; escaping past canary costs 7.8× base;
reaching production costs 17× — which is {{ch:ops-lifecycle}}'s return-trip result in gate
form.

Gate placement minimises the product of those two terms, and the result is not "as early as
possible." A schema check at 0.9 per run costs 126 a release on every commit — trivially
worth it. A human spot-check at 4,100 per run costs 574,000 a release on every commit, which
is more than every defect it would ever catch. It belongs at pre-deploy.

**Cheap gates want to be early and expensive gates want to be late.** That is a correction to
how "shift left" is usually applied: the principle is right and the implementation — move
every check earlier — is wrong for any check whose per-run cost is not negligible.
{{ch:ops-lifecycle}} made the same correction from the other side, and the two arguments meet
here.

Run the full pipeline with each gate at its cheapest stage and it catches 2.906 of 3.1
defects for 31,990 in run cost and 7,926 in escapes — 39,916 a release against 126,480 with
no gates at all.

Then there is the reason gates get disabled, which is not that they fail to catch things.

A gate with a false-positive rate blocks good changes. Each block costs an investigation, a
rerun, and a delay. The judge ensemble catches 0.300 defects a release at the PR stage,
worth 10,872 against letting them reach production. At a 5% false-positive rate it blocks
1.30 good changes for 3,640 — still worth it. At 20% it blocks 5.20 for 14,560, and the net
turns negative.

The break-even is 14.9%. Above it, the gate costs more in blocked good changes than it saves
in caught defects, and it should be advisory rather than blocking.

That gives a better policy than the usual one. Gates are normally blocking or advisory by
*category* — tests block, lint warns, quality metrics warn. The rule here is uniform and
computed: **a gate blocks if its false-positive rate is below its own threshold**, and the
threshold depends on what it catches and what a block costs. A schema check may block at up
to 50.7%. A human spot-check at up to 10.1%.

Finally, the constraint that overrides both, and the one this part ends on.

Sum the gate hours. Each gate's latency times how often it runs at its stage: 322 hours per
release. Against {{ch:ops-lifecycle}}'s 847-hour loop period, of which 156 was work.

The evaluation framework is 38% of the loop's duration, almost all of it waiting. Every gate
in it is individually justified by the tables above, and the sum is a budget decision nobody
made.

## 5. Formal Explanation

**Coverage as a union.** With failure classes $c$ of share $w_c$ and instruments $i$ with
per-class detection probability $d_{ic}$, the coverage of a set $S$ is

$$C(S) = \sum_c w_c \left[1 - \prod_{i \in S} (1 - d_{ic})\right].$$

This is monotone and submodular in $S$: adding an instrument to a larger set gains no more
than adding it to a smaller one. Submodularity is what makes the greedy build near-optimal —
it guarantees the greedy solution is within $1 - 1/e$ of the best subset at any cardinality —
and it is also what makes coverage sums meaningless.

**The greedy order.** At each step choose $\arg\max_i [C(S \cup \{i\}) - C(S)] / k_i$ for
cost $k_i$. Because the $d_{ic}$ for cheap instruments concentrate on a few classes with high
$d$, and expensive instruments spread moderate $d$ across many, the greedy order is roughly by
cost — but not exactly, and the exceptions are where the interesting choices are.

**Placement.** For gate $g$ at stage $s$ with per-run cost $k_g$, runs per release $n_s$,
detection $d_g$, and escape multiplier $m_s$, the cost per release is

$$L(g, s) = k_g n_s + D(1 - d_g)\, \beta\, m_s,$$

with $D$ defects per release and $\beta$ the base defect cost. Since $n_s$ decreases and
$m_s$ increases along the pipeline, $L$ is a sum of a decreasing and an increasing term and
has an interior minimum whose location depends on $k_g$: large $k_g$ pushes the minimum
later.

**The blocking threshold.** A gate at stage $s$ catching $q_g$ defects is worth
$q_g \beta (m_{\text{prod}} - m_s)$; at false-positive rate $\phi$ over $N$ changes it costs
$N \phi \gamma$ for block cost $\gamma$. Blocking is net positive iff

$$\phi < \phi^\star_g = \frac{q_g \beta (m_{\text{prod}} - m_s)}{N \gamma},$$

which is a per-gate quantity, not a per-category one.

**Loop occupancy.** Total gate hours are $\sum_g \ell_g n_{s(g)}$ for latency $\ell_g$, and
this competes with {{ch:ops-lifecycle}}'s period rather than with the evaluation budget. It
is the constraint that binds first and the one with no line item.

## 6. Mathematical Foundation

Coverage as a submodular union:

$$C(S) = \sum_c w_c\left[1 - \prod_{i \in S}(1 - d_{ic})\right] \;\le\; \sum_{i \in S} C(\{i\})$$ (eq:coverage-is-a-union-not-a-sum)

Individual coverages summing to **1.827** give a union of **0.956**.

The greedy portfolio's composition:

$$S^\star(B) = \text{greedy}\left(\max_i \frac{\Delta C_i}{k_i}\right), \qquad \frac{\sum_{i \in S^\star} k_i \mathbf{1}[\text{reference-free}]}{\sum_{i \in S^\star} k_i} = 100\% \ \text{at}\ B = 250$$ (eq:optimal-portfolio-is-mostly-reference-free)

with $C(S^\star) = 0.776$ and the remaining $0.180$ costing $39\times$ as much.

Placement as a sum of an increasing and a decreasing term:

$$L(g,s) = k_g n_s + D(1-d_g)\beta m_s, \qquad s^\star(g) = \arg\min_s L(g,s), \qquad \frac{\partial s^\star}{\partial k_g} > 0$$ (eq:gate-placement-is-set-by-cost-times-escape)

At $k_g = 0.9$: `every commit`. At $k_g = 4100$: `pre-deploy`, where the same gate on every
commit would cost **574,000**.

And the blocking rule:

$$\phi^\star_g = \frac{q_g \beta (m_{\text{prod}} - m_{s(g)})}{N\gamma}$$ (eq:a-flaky-gate-has-a-blocking-threshold)

**50.7%** for a schema check, **14.9%** for a judge ensemble, **10.1%** for a human
spot-check.

## 7. Internal Mechanics

Why is the overlap between instruments so large? Because instruments are built against
failures somebody has already seen, and the failures everybody has already seen are the same
ones. Malformed output is the first failure any system produces, so every instrument's author
made sure to catch it. The failures with low coverage — retrieval insufficiency, wrong
specificity, unsound-but-correct trajectories — are the ones that are hard to notice, so
nothing was built for them, so they remain hard to notice. **The coverage distribution is a
record of what was easy to observe**, not of what matters.

That is also why the greedy order is roughly by cost and the exceptions are informative. The
utilisation probe is expensive per unit of coverage and enters fifth because it is the only
instrument that sees its class at all; the frozen regression suite is cheaper and enters
seventh because execution grading has already taken most of its class. **An instrument's
value is set by what it sees *uniquely*, and uniqueness is invisible from the instrument's own
documentation.**

The placement result has a mechanism worth stating carefully because it is easy to state
wrongly. The claim is not that early gates are bad — early gates are excellent when they are
cheap. The claim is that the *run count* multiplier is usually forgotten. A team reasoning
about "where should this check go" thinks about the escape cost, which favours early, and does
not multiply by the runs, which favours late. The two terms are of comparable magnitude and
only one of them is salient.

There is a second-order effect on the flake threshold that explains a common organisational
pattern. A gate's threshold depends on how many defects it catches at its stage, which
depends on what the earlier gates already removed. So **adding an early gate lowers the
blocking threshold of every later gate**, because the later gate now catches fewer defects
and its false positives are unchanged. A pipeline that grows over time therefore accumulates
gates that were justified when added and are not any more, and nothing re-evaluates them.
That is the mechanism behind the familiar state of a CI pipeline full of checks nobody
trusts.

Finally, the loop-occupancy result. Gate hours compete with the delivery period rather than
with the evaluation budget, and the two are governed by different people — the budget by
whoever approves spend, the period by whoever complains that shipping is slow. Because no
single decision allocates the period, gates accumulate until someone notices the loop is slow
and starts removing them, usually the newest ones rather than the least valuable ones.
{{ch:ops-lifecycle}}'s finding that **waiting is invisible because nobody is doing anything**
applies here with full force: the 322 hours are real, they are on nobody's report, and the
first evidence of them is a complaint.

## 8. Implementation

The first listing selects the portfolio.

```python {tier=A name=C:/Users/MANAVB~1/AppData/Local/Temp/claude/C--Github-LaughingBuddha/30a87753-43a8-48c3-8378-261faf976dbb/scratchpad/hh1}
"""An evaluation framework is a portfolio of instruments, and coverage is a union.

Every chapter in this part produced an instrument and a failure class it sees. Put them
together and the temptation is to add up coverages, which is wrong twice: instruments
overlap, so the total is a union rather than a sum, and the overlaps are not small because
the cheap instruments all look at the same easy failures
(eq:coverage-is-a-union-not-a-sum).

Selected greedily against cost, the resulting portfolio has a shape worth predicting in
advance: it is dominated by reference-free checks, human evaluation appears as a small
calibration slice, and the benchmark suite -- the thing most teams build first -- does not
make the cut until late (eq:optimal-portfolio-is-mostly-reference-free).
"""
# (failure class, share of production defects)
CLASSES = [
    ("malformed or invalid output",   0.11),
    ("wrong fact, well formed",       0.19),
    ("unsupported claim",             0.14),
    ("retrieval insufficient",        0.13),
    ("wrong tool arguments",          0.09),
    ("unsound but correct outcome",   0.07),
    ("regression on a known case",    0.12),
    ("wrong specificity or length",   0.08),
    ("policy or safety violation",    0.07),
]
shares = dict(CLASSES)

# instrument -> (cost per 1000 items, latency hours, {class: detection prob})
INSTRUMENTS = {
    "schema and format check": (0.4, 0.05, {
        "malformed or invalid output": 0.96, "wrong tool arguments": 0.44}),
    "invariant suite": (31.0, 0.3, {
        "wrong tool arguments": 0.71, "unsound but correct outcome": 0.78,
        "policy or safety violation": 0.42, "malformed or invalid output": 0.35}),
    "execution / test grading": (12.0, 0.6, {
        "wrong fact, well formed": 0.34, "unsound but correct outcome": 0.29,
        "regression on a known case": 0.88}),
    "faithfulness judge": (21.0, 0.4, {
        "unsupported claim": 0.81, "wrong fact, well formed": 0.28}),
    "utilisation probe": (38.0, 0.4, {
        "retrieval insufficient": 0.36, "wrong fact, well formed": 0.22}),
    "judge ensemble, both orders": (114.0, 0.9, {
        "wrong fact, well formed": 0.52, "wrong specificity or length": 0.61,
        "unsupported claim": 0.44, "policy or safety violation": 0.55}),
    "frozen regression suite": (46.0, 0.5, {
        "regression on a known case": 0.94, "wrong fact, well formed": 0.19}),
    "sufficiency annotation": (4800.0, 30.0, {
        "retrieval insufficient": 0.93, "wrong fact, well formed": 0.24}),
    "human spot-check": (3400.0, 26.0, {
        "wrong fact, well formed": 0.78, "wrong specificity or length": 0.83,
        "unsupported claim": 0.71, "policy or safety violation": 0.80,
        "unsound but correct outcome": 0.51, "retrieval insufficient": 0.62}),
}


def coverage(chosen):
    """Weighted share of defects detected by at least one chosen instrument."""
    tot = 0.0
    per = {}
    for cls, sh in CLASSES:
        miss = 1.0
        for name in chosen:
            miss *= (1.0 - INSTRUMENTS[name][2].get(cls, 0.0))
        per[cls] = 1.0 - miss
        tot += sh * (1.0 - miss)
    return tot, per


print("Nine failure classes. What each instrument sees, and what it costs.")
print()
print(f"{'instrument':>30}{'cost/1000':>12}{'latency h':>12}"
      f"{'classes':>10}{'alone covers':>15}")
print("-" * 79)
for name, (c, lat, d) in INSTRUMENTS.items():
    cov, _ = coverage([name])
    print(f"{name:>30}{c:>12,.1f}{lat:>12.2f}{len(d):>10}{cov:>15.3f}")

print()
print(f"sum of individual coverages: "
      f"{sum(coverage([n])[0] for n in INSTRUMENTS):.3f}")
print(f"coverage of all nine together: {coverage(list(INSTRUMENTS))[0]:.3f}")
print("the difference is overlap")

print()
print()
print("Greedy build: add the instrument with the best marginal coverage per")
print("dollar, one at a time.")
print()
print(f"{'add':>30}{'marginal cover':>17}{'cost/1000':>12}"
      f"{'per 1000 dollars':>19}{'total cover':>14}{'total cost':>13}")
print("-" * 105)
chosen = []
spent = 0.0
order = []
while len(chosen) < len(INSTRUMENTS):
    cur, _ = coverage(chosen)
    best, best_ratio, best_gain = None, -1.0, 0.0
    for name in INSTRUMENTS:
        if name in chosen:
            continue
        gain = coverage(chosen + [name])[0] - cur
        ratio = gain / INSTRUMENTS[name][0]
        if ratio > best_ratio:
            best, best_ratio, best_gain = name, ratio, gain
    chosen.append(best)
    spent += INSTRUMENTS[best][0]
    tot, _ = coverage(chosen)
    order.append((best, best_gain, INSTRUMENTS[best][0], best_ratio, tot, spent))
    print(f"{best:>30}{best_gain:>17.4f}{INSTRUMENTS[best][0]:>12,.1f}"
          f"{best_ratio * 1000:>19.2f}{tot:>14.3f}{spent:>13,.1f}")

print()
print()
print("The frontier: coverage available at each budget.")
print()
print(f"{'budget/1000 items':>19}{'instruments':>13}{'coverage':>11}"
      f"{'undetected':>13}{'marginal per 1000':>20}")
print("-" * 76)
front = {}
for b in (1.0, 50.0, 100.0, 250.0, 1000.0, 5000.0, 9000.0):
    sel = []
    s = 0.0
    for name, gain, c, ratio, tot, sp in order:
        if s + c <= b:
            sel.append(name)
            s += c
    cov, _ = coverage(sel)
    front[b] = (len(sel), cov, s)
    prev = front.get(list(front)[max(0, list(front).index(b) - 1)],
                     (0, 0.0, 0.0)) if len(front) > 1 else (0, 0.0, 0.0)
    marg = ((cov - prev[1]) / max(s - prev[2], 1e-9)) * 1000 if s > prev[2] else 0.0
    print(f"{b:>19,.0f}{len(sel):>13}{cov:>11.3f}{1 - cov:>13.3f}"
          f"{marg:>20.4f}")

print()
print()
print("What the portfolio is made of, at a realistic budget.")
print()
BUDGET = 250.0
sel, s = [], 0.0
for name, gain, c, ratio, tot, sp in order:
    if s + c <= BUDGET:
        sel.append(name)
        s += c
cov, per = coverage(sel)
print(f"{'instrument':>30}{'cost/1000':>12}{'share of budget':>18}"
      f"{'needs a reference?':>21}")
print("-" * 81)
NEEDS_REF = {
    "schema and format check": "no", "invariant suite": "no",
    "execution / test grading": "no", "faithfulness judge": "no",
    "utilisation probe": "no", "judge ensemble, both orders": "no",
    "frozen regression suite": "yes", "sufficiency annotation": "yes",
    "human spot-check": "yes",
}
for name in sel:
    print(f"{name:>30}{INSTRUMENTS[name][0]:>12,.1f}"
          f"{INSTRUMENTS[name][0] / s:>18.1%}{NEEDS_REF[name]:>21}")
print("-" * 81)
print(f"{'TOTAL':>30}{s:>12,.1f}{1.0:>18.1%}")
ref_free = sum(INSTRUMENTS[n][0] for n in sel if NEEDS_REF[n] == "no")
print()
print(f"reference-free share of the budget: {ref_free / s:.0%}")
print(f"coverage reached: {cov:.3f}")

print()
print()
print("What is left undetected, by class.")
print()
print(f"{'failure class':>30}{'share':>9}{'detected':>11}"
      f"{'undetected mass':>18}{'what would catch it':>28}")
print("-" * 96)
WOULD = {
    "malformed or invalid output": "already covered",
    "wrong fact, well formed": "human spot-check",
    "unsupported claim": "already covered",
    "retrieval insufficient": "sufficiency annotation",
    "wrong tool arguments": "already covered",
    "unsound but correct outcome": "already covered",
    "regression on a known case": "frozen regression suite",
    "wrong specificity or length": "human spot-check",
    "policy or safety violation": "already covered",
}
resid = sorted(CLASSES, key=lambda c: -(c[1] * (1 - per[c[0]])))
for cls, sh in resid:
    print(f"{cls:>30}{sh:>9.0%}{per[cls]:>11.2f}"
          f"{sh * (1 - per[cls]):>18.4f}{WOULD[cls]:>28}")

print(f"""
The instrument table is the part inventory. Nine instruments, costs spanning four orders of
magnitude from {0.4:.1f} to {4800:,.0f} per thousand items, and each seeing between two and
six of the nine failure classes.

The two lines under it are the arithmetic that matters. Individual coverages sum to
{sum(coverage([n])[0] for n in INSTRUMENTS):.3f}; the nine together cover
{coverage(list(INSTRUMENTS))[0]:.3f} (eq:coverage-is-a-union-not-a-sum). **The overlap is
larger than any single instrument's coverage**, because the cheap instruments all point at
the same visible failures and the expensive ones point at everything.

The greedy build is the design. `{order[0][0]}` goes first at
{order[0][3] * 1000:.2f} points of coverage per thousand dollars; `{order[1][0]}` second;
`{order[2][0]}` third. The last three additions -- the expensive human-referenced ones --
contribute {sum(o[1] for o in order[-3:]):.4f} between them for
{sum(o[2] for o in order[-3:]):,.0f}.

The frontier converts that into a budget decision. At {250:,.0f} per thousand items the
portfolio covers {front[250.0][1]:.3f}; at {9000:,.0f} -- thirty-six times the budget -- it
covers {front[9000.0][1]:.3f}. **The last {front[9000.0][1] - front[250.0][1]:.3f} of
coverage costs {front[9000.0][2] / front[250.0][2]:.0f} times what the first
{front[250.0][1]:.3f} did.**

That is the shape every evaluation budget has, and it is worth knowing before the argument
rather than after: the framework is cheap and the last tenth is not.

The composition table is the prediction to carry into a design review.
**{ref_free / s:.0%} of the optimal budget goes to instruments that need no reference at
all** (eq:optimal-portfolio-is-mostly-reference-free) -- schema checks, invariants,
execution grading, judges. That is not a coincidence: ch:ev-why-hard's escape was to state a
predicate instead of writing an answer, and a predicate is cheap precisely because it needs
nothing external.

Notice which instruments are *absent* at this budget. The frozen regression suite and the
human spot-check both miss the cut, and between them they are what most teams build first --
because a regression suite is what evaluation looks like in a codebase and a human review is
what quality looks like in an organisation. Neither is wrong; both are late in the ordering.

The residual table is the honest closing. `{resid[0][0]}` retains
{resid[0][1] * (1 - per[resid[0][0]]):.4f} of undetected defect mass -- more than the next two
rows combined -- and the instrument that would catch it is `{WOULD[resid[0][0]]}`, which is
the most expensive row in the inventory at {4800:,.0f} per thousand.

The next two residuals both want the human spot-check, at {3400:,.0f}.

Which is the argument for buying a *slice* of the expensive instruments rather than skipping
them. ch:ev-llm-judge's spot-check result applies unchanged: a small human sample gates
nothing and measures the blind spot of everything else, and the same is true of a few hundred
annotated queries a quarter for sufficiency. **A portfolio without them has an unmeasured
residual rather than a small one**, and the difference between those is the whole reason to
own the table.""")
```

## 9. Practical Example

Nine instruments against nine failure classes:

```
                    instrument   cost/1000   latency h   classes   alone covers
-------------------------------------------------------------------------------
       schema and format check         0.4        0.05         2          0.145
               invariant suite        31.0        0.30         4          0.186
      execution / test grading        12.0        0.60         3          0.191
            faithfulness judge        21.0        0.40         2          0.167
   judge ensemble, both orders       114.0        0.90         4          0.248
        sufficiency annotation     4,800.0       30.00         2          0.167
              human spot-check     3,400.0       26.00         6          0.486

sum of individual coverages: 1.827
coverage of all nine together: 0.956
```

**The overlap exceeds any single instrument's coverage**
({{eq:coverage-is-a-union-not-a-sum}}), because the cheap instruments all point at the same
visible failures.

```
                           add   marginal cover   cost/1000   per 1000 dollars   total cover   total cost
---------------------------------------------------------------------------------------------------------
       schema and format check           0.1452         0.4             363.00         0.145          0.4
      execution / test grading           0.1905        12.0              15.88         0.336         12.4
            faithfulness judge           0.1485        21.0               7.07         0.484         33.4
               invariant suite           0.1055        31.0               3.40         0.590         64.4
             utilisation probe           0.0667        38.0               1.75         0.656        102.4
   judge ensemble, both orders           0.1195       114.0               1.05         0.776        216.4
       frozen regression suite           0.0200        46.0               0.43         0.796        262.4
              human spot-check           0.1296     3,400.0               0.04         0.925      3,662.4
        sufficiency annotation           0.0308     4,800.0               0.01         0.956      8,462.4
```

```
  budget/1000 items  instruments   coverage   undetected   marginal per 1000
----------------------------------------------------------------------------
                  1            1      0.145        0.855            363.0000
                100            4      0.590        0.410              3.4029
                250            6      0.776        0.224              1.2245
              5,000            8      0.925        0.075              0.0381
              9,000            9      0.956        0.044              0.0064
```

**The last 0.180 of coverage costs 39× what the first 0.776 did.**

```
                    instrument   cost/1000   share of budget   needs a reference?
---------------------------------------------------------------------------------
       schema and format check         0.4              0.2%                   no
      execution / test grading        12.0              5.5%                   no
            faithfulness judge        21.0              9.7%                   no
               invariant suite        31.0             14.3%                   no
             utilisation probe        38.0             17.6%                   no
   judge ensemble, both orders       114.0             52.7%                   no
```

**100% of the optimal budget is reference-free**
({{eq:optimal-portfolio-is-mostly-reference-free}}) — and the frozen regression suite and
human spot-check, the two things teams build first, are absent.

```
                 failure class    share   detected   undetected mass         what would catch it
------------------------------------------------------------------------------------------------
        retrieval insufficient      13%       0.36            0.0832      sufficiency annotation
       wrong fact, well formed      19%       0.82            0.0338            human spot-check
   wrong specificity or length       8%       0.61            0.0312            human spot-check
   malformed or invalid output      11%       0.97            0.0029             already covered
```

Buy a *slice* of the expensive instruments: they gate nothing and they measure the blind
spot of everything else.

The second listing places the gates.

```python {tier=A name=C:/Users/MANAVB~1/AppData/Local/Temp/claude/C--Github-LaughingBuddha/30a87753-43a8-48c3-8378-261faf976dbb/scratchpad/hh2}
"""Where a gate goes is a cost decision, and "as early as possible" is the wrong rule.

An instrument that runs on every commit runs a hundred times more often than one that runs
per release, so its cost is multiplied by a hundred. A defect that escapes to production
costs ch:ops-lifecycle's full return trip. Gate placement is the product of those two terms
(eq:gate-placement-is-set-by-cost-times-escape), and the optimum puts cheap fast checks
early and expensive slow ones late -- which is not what "shift left" is usually taken to
mean.

The second half is the reason gates get disabled. A gate with a false-positive rate blocks
good changes, and above a computable threshold the blocking costs more than the catching
(eq:a-flaky-gate-has-a-blocking-threshold).
"""
# (stage, runs per release, cost multiplier of a defect escaping past this stage)
STAGES = [
    ("every commit",   140.0,  1.0),
    ("pull request",    32.0,  1.9),
    ("pre-merge",       32.0,  2.4),
    ("pre-deploy",       4.0,  4.1),
    ("canary",           4.0,  7.8),
    ("production",       1.0, 17.0),
]
# (instrument, cost per run, detection probability, latency hours)
GATES = [
    ("schema and format check",       0.9, 0.31, 0.05),
    ("execution / test grading",     14.0, 0.34, 0.60),
    ("faithfulness judge",           26.0, 0.29, 0.40),
    ("invariant suite",              38.0, 0.27, 0.30),
    ("judge ensemble, both orders", 142.0, 0.41, 0.90),
    ("human spot-check",           4100.0, 0.55, 26.0),
]
DEFECTS_PER_RELEASE = 3.1
BASE_DEFECT_COST = 2400.0

print("Six stages, each running a gate a different number of times per release,")
print("each letting a surviving defect cost more.")
print()
print(f"{'stage':>16}{'runs/release':>15}{'escape multiplier':>20}"
      f"{'cost of one escape':>21}")
print("-" * 72)
for name, runs, mult in STAGES:
    print(f"{name:>16}{runs:>15.0f}{mult:>20.1f}"
          f"{BASE_DEFECT_COST * mult:>21,.0f}")

print()
print()
print("Placing one gate: total cost per release at each stage.")
print()


def place(gate, stage):
    gname, gcost, gdet, glat = gate
    sname, runs, mult = stage
    run_cost = gcost * runs
    escaped = DEFECTS_PER_RELEASE * (1.0 - gdet)
    escape_cost = escaped * BASE_DEFECT_COST * mult
    return run_cost, escape_cost, run_cost + escape_cost


print(f"{'gate':>30}", end="")
for sname, runs, mult in STAGES:
    print(f"{sname:>15}", end="")
print()
print("-" * 120)
best_stage = {}
for g in GATES:
    print(f"{g[0]:>30}", end="")
    totals = {}
    for s in STAGES:
        rc, ec, tot = place(g, s)
        totals[s[0]] = tot
        print(f"{tot:>15,.0f}", end="")
    print()
    best_stage[g[0]] = min(totals, key=lambda k: totals[k])

print()
print(f"{'gate':>30}{'cheapest stage':>18}{'run cost there':>17}"
      f"{'escape cost there':>20}")
print("-" * 85)
for g in GATES:
    s = [x for x in STAGES if x[0] == best_stage[g[0]]][0]
    rc, ec, tot = place(g, s)
    print(f"{g[0]:>30}{best_stage[g[0]]:>18}{rc:>17,.0f}{ec:>20,.0f}")

print()
print("Cheap gates want to be early because their run cost is small even at")
print("140 runs; expensive gates want to be late because it is not.")

print()
print()
print("A pipeline: each gate at its cheapest stage, applied in sequence.")
print()
print(f"{'stage':>16}{'gate':>30}{'defects entering':>19}"
      f"{'caught':>9}{'run cost':>11}{'escape cost':>14}")
print("-" * 99)
remaining = DEFECTS_PER_RELEASE
total_run = 0.0
pipeline = []
for sname, runs, mult in STAGES:
    for g in GATES:
        if best_stage[g[0]] != sname:
            continue
        caught = remaining * g[1 + 1]
        rc = g[1] * runs
        total_run += rc
        entering = remaining
        remaining -= caught
        pipeline.append((sname, g[0], entering, caught, rc, remaining))
        print(f"{sname:>16}{g[0]:>30}{entering:>19.3f}"
              f"{caught:>9.3f}{rc:>11,.0f}"
              f"{remaining * BASE_DEFECT_COST * mult:>14,.0f}")

final_escape = remaining * BASE_DEFECT_COST * STAGES[-1][2]
print("-" * 99)
print(f"{'TOTAL':>16}{'':>30}{'':>19}"
      f"{DEFECTS_PER_RELEASE - remaining:>9.3f}{total_run:>11,.0f}"
      f"{final_escape:>14,.0f}")
print()
print(f"total per release: {total_run + final_escape:,.0f}")
print(f"with no gates at all: "
      f"{DEFECTS_PER_RELEASE * BASE_DEFECT_COST * STAGES[-1][2]:,.0f}")

print()
print()
print("Now the reason gates get turned off: false positives block good changes.")
print()
CHANGES_PER_RELEASE = 26.0
BLOCK_COST = 2800.0       # a blocked good change: investigation, rerun, delay
print(f"{'false-positive rate':>21}{'good changes blocked':>23}"
      f"{'blocking cost':>16}{'catching value':>17}{'net':>12}")
print("-" * 89)
caught_at = {gname: (sname, caught)
             for sname, gname, ent, caught, rc, rem in pipeline}
mult_of = {sname: mult for sname, runs, mult in STAGES}
PROD_MULT = STAGES[-1][2]


def gate_value(gname):
    """Defects this gate catches, valued at what reaching production would cost."""
    sname, caught = caught_at[gname]
    return caught * BASE_DEFECT_COST * (PROD_MULT - mult_of[sname])


GATE = GATES[4]
caught_value = gate_value(GATE[0])
flake = {}
for fp in (0.005, 0.02, 0.05, 0.10, 0.20, 0.35):
    blocked = CHANGES_PER_RELEASE * fp
    bcost = blocked * BLOCK_COST
    flake[fp] = (blocked, bcost, caught_value - bcost)
    print(f"{fp:>21.1%}{blocked:>23.2f}{bcost:>16,.0f}"
          f"{caught_value:>17,.0f}{caught_value - bcost:>12,.0f}")

threshold = caught_value / (CHANGES_PER_RELEASE * BLOCK_COST)
print()
print(f"break-even false-positive rate: {threshold:.1%}")
print("above that, the gate should be advisory rather than blocking")

print()
print()
print("The same threshold for each gate, which is where the policy comes from.")
print()
print(f"{'gate':>30}{'catches':>10}{'value caught':>15}"
      f"{'max FP rate to block':>23}")
print("-" * 78)
thr = {}
for g in GATES:
    val = gate_value(g[0])
    tt = val / (CHANGES_PER_RELEASE * BLOCK_COST)
    thr[g[0]] = tt
    print(f"{g[0]:>30}{caught_at[g[0]][1]:>10.3f}{val:>15,.0f}"
          f"{min(tt, 1.0):>23.1%}")

print()
print()
print("And what latency does, independent of cost: how long a gate holds the")
print("loop open.")
print()
runs_of = {sname: runs for sname, runs, mult in STAGES}
print(f"{'gate':>30}{'latency h':>12}{'fits in':>18}"
      f"{'gate hours per release':>25}")
print("-" * 85)
hours = {}
for g in GATES:
    lat = g[3]
    fits = ("a commit hook" if lat < 0.1 else
            "CI" if lat < 1.0 else
            "a nightly run" if lat < 12.0 else
            "a release cycle")
    h = lat * runs_of[best_stage[g[0]]]
    hours[g[0]] = h
    print(f"{g[0]:>30}{lat:>12.2f}{fits:>18}{h:>25.1f}")
print("-" * 85)
print(f"{'TOTAL':>30}{'':>12}{'':>18}{sum(hours.values()):>25.1f}")

print(f"""
The stage table is the two multipliers that decide everything. A gate at `every commit` runs
{STAGES[0][1]:.0f} times a release and a gate at `pre-deploy` runs {STAGES[3][1]:.0f} times,
so the same instrument costs {STAGES[0][1] / STAGES[3][1]:.0f} times more in the first
position. And a defect escaping past `every commit` costs
{STAGES[0][2]:.1f}x base while one escaping past `canary` costs {STAGES[4][2]:.1f}x, which is
ch:ops-lifecycle's return-trip result in gate form.

The placement grid multiplies them out. The `{GATES[0][0]}` is cheapest at
`{best_stage[GATES[0][0]]}`; the `{GATES[5][0]}` is cheapest at
`{best_stage[GATES[5][0]]}` (eq:gate-placement-is-set-by-cost-times-escape).

**Cheap gates want to be early and expensive gates want to be late**, and the reason is
arithmetic rather than philosophy: at {STAGES[0][1]:.0f} runs a release, a
{GATES[5][1]:,.0f}-per-run instrument costs {GATES[5][1] * STAGES[0][1]:,.0f} to sit on every
commit, which is more than every defect it would ever catch.

That is a correction to how "shift left" is usually applied. The principle is right and the
implementation -- move every check earlier -- is wrong for any check whose per-run cost is
not negligible. ch:ops-lifecycle made the same correction from the other side: **shorten the
return trip, do not blindly move detection earlier.**

The pipeline table puts each gate at its cheapest stage and runs them in sequence. Total
{total_run + final_escape:,.0f} per release, against
{DEFECTS_PER_RELEASE * BASE_DEFECT_COST * STAGES[-1][2]:,.0f} with no gates --
{(DEFECTS_PER_RELEASE * BASE_DEFECT_COST * STAGES[-1][2]) / (total_run + final_escape):.1f}
times cheaper -- with {DEFECTS_PER_RELEASE - remaining:.3f} of {DEFECTS_PER_RELEASE:.1f}
defects caught.

Note where the run cost concentrates. The gates at `every commit` are running
{STAGES[0][1]:.0f} times and cost {sum(g[1] * STAGES[0][1] for g in GATES if best_stage[g[0]] == 'every commit'):,.0f}
between them, which is a real line item and is still small against the escapes it prevents.

The flake table is the failure mode that kills gates in practice. The
`{GATE[0]}` catches {caught_at[GATE[0]][1]:.3f} defects a release at its stage, worth
{caught_value:,.0f} against letting them reach production. At a {0.05:.0%} false-positive rate it blocks
{flake[0.05][0]:.2f} good changes for {flake[0.05][1]:,.0f}; at {0.35:.0%} it blocks
{flake[0.35][0]:.2f} for {flake[0.35][1]:,.0f}, and the net is
{flake[0.35][2]:,.0f} (eq:a-flaky-gate-has-a-blocking-threshold).

The break-even is **{threshold:.1%}** -- above that, the gate costs more in blocked good
changes than it saves in caught defects, and it should be advisory rather than blocking.

The per-gate threshold table is the policy this produces, and it is a better policy than the
usual one. Gates are normally blocking or advisory by *category* -- tests block, lint warns,
quality metrics warn. Here the rule is uniform and computed: **a gate blocks if its
false-positive rate is below its own threshold**, which depends on what it catches and what a
block costs. The `{GATES[0][0]}` may block at up to {min(thr[GATES[0][0]], 1.0):.0%}; the
`{GATES[5][0]}` at up to {min(thr[GATES[5][0]], 1.0):.0%}.

The latency table is the constraint that overrides both, and its last column is the number
to check before agreeing to any of this. A gate taking {GATES[5][3]:.0f} hours cannot sit in
CI whatever its economics say. And summed across the pipeline the gates occupy
{sum(hours.values()):.0f} hours per release -- against ch:ops-lifecycle's
{847:.0f}-hour period, of which {156:.0f} was work.

**The evaluation framework is now
{sum(hours.values()) / 847.0:.0%} of the loop's total duration**, and most of it is
waiting rather than working, which is exactly the category that chapter found dominates and
nobody measures. Every gate is individually justified by the table above it and the sum is a
budget decision nobody made.

Which is the last thing to carry out of this part. An evaluation framework is not free even
when every instrument in it is cheap, and the cost that binds is not the one on the invoice.
Run the hours column before the dollars column.""")
```

```
           stage   runs/release   escape multiplier   cost of one escape
------------------------------------------------------------------------
    every commit            140                 1.0                2,400
      pre-deploy              4                 4.1                9,840
          canary              4                 7.8               18,720
      production              1                17.0               40,800

                          gate    cheapest stage   run cost there   escape cost there
-------------------------------------------------------------------------------------
       schema and format check      every commit              126               5,134
   judge ensemble, both orders      pull request            4,544               8,340
              human spot-check        pre-deploy           16,400              13,727
```

**Cheap gates want to be early and expensive gates want to be late**
({{eq:gate-placement-is-set-by-cost-times-escape}}) — a 4,100-per-run gate on every commit
costs **574,000** a release, more than every defect it would catch.

```
           stage                          gate   defects entering   caught   run cost   escape cost
---------------------------------------------------------------------------------------------------
    every commit       schema and format check              3.100    0.961        126         5,134
    every commit      execution / test grading              2.139    0.727      1,960         3,388
    pull request   judge ensemble, both orders              0.732    0.300      4,544         1,969
      pre-deploy              human spot-check              0.432    0.237     16,400         1,912
---------------------------------------------------------------------------------------------------
           TOTAL                                                     2.906     31,990         7,926
```

**39,916 a release against 126,480 with no gates** — 3.2× cheaper, catching 2.906 of 3.1
defects.

```
  false-positive rate   good changes blocked   blocking cost   catching value         net
-----------------------------------------------------------------------------------------
                 5.0%                   1.30           3,640           10,872       7,232
                10.0%                   2.60           7,280           10,872       3,592
                20.0%                   5.20          14,560           10,872      -3,688

                          gate   catches   value caught   max FP rate to block
------------------------------------------------------------------------------
       schema and format check     0.961         36,902                  50.7%
            faithfulness judge     0.409         15,721                  21.6%
   judge ensemble, both orders     0.300         10,872                  14.9%
              human spot-check     0.237          7,351                  10.1%
```

**A gate blocks if its false-positive rate is below its own threshold**
({{eq:a-flaky-gate-has-a-blocking-threshold}}) — a computed per-gate rule, not a per-category
convention.

```
                          gate   latency h           fits in   gate hours per release
-------------------------------------------------------------------------------------
       schema and format check        0.05     a commit hook                      7.0
      execution / test grading        0.60                CI                     84.0
   judge ensemble, both orders        0.90                CI                     28.8
              human spot-check       26.00   a release cycle                    104.0
-------------------------------------------------------------------------------------
                         TOTAL                                                  322.0
```

**322 gate hours per release against an 847-hour loop period — 38%**, almost all of it
waiting, on nobody's report.

## 10. Production Considerations

Compute coverage as a union before agreeing any evaluation roadmap. Summed coverages
overstate by more than the largest instrument contributes.

Build in greedy order, not in the order the instruments were invented. The schema check and
execution grading come before anything with a reference.

Buy a sampled slice of the expensive instruments rather than skipping them. They gate nothing
and they are the only measurement of the residual.

Place each gate by run count times escape multiplier. The run count is the term teams forget
and it is the one that favours late placement.

Compute each gate's blocking threshold from what it catches at its stage. Blocking by
category is a convention; blocking by threshold is a calculation.

Re-evaluate every gate's threshold when an earlier gate is added. Adding an early gate lowers
every later gate's threshold and nothing recomputes it.

Total the gate hours against your loop period before adding anything. It is the constraint
that binds first and it has no line item.

## 11. Common Mistakes

**Summing coverages.** Nine instruments summing to 1.827 cover 0.956.

**Building the regression suite first.** It enters the greedy order seventh of nine.

**Skipping the expensive instruments entirely.** The residual becomes unmeasured rather than
small.

**Placing every gate as early as possible.** Run count multiplies the cost by up to 35×.

**Blocking by category.** Thresholds range from 10.1% to 50.7% across gates on the same
pipeline.

**Ignoring gate hours.** They are 38% of the loop period and appear on no budget.

## 12. Failure Modes

**Portfolio with an unmeasured residual.** Every cheap instrument is green and the largest
failure class has no instrument pointed at it.

**Pipeline of distrusted gates.** Early gates were added over time, later gates' thresholds
fell below their false-positive rates, and nobody recomputed them.

**Expensive gate on every commit.** A per-run cost of 4,100 at 140 runs, justified by an
escape multiplier nobody multiplied out.

**Loop slowed by evaluation.** Shipping is slow, the newest gate is removed, and the gate
removed is not the least valuable one.

**Instruments chosen from documentation.** Each instrument's value is what it sees uniquely,
which its own documentation cannot tell you.

**Coverage reported without a class breakdown.** 0.776 hides that one class is at 0.36 and
another at 0.97.

## 13. Alternatives

**One end-to-end score.** Simple, gate-able, and it cannot distinguish the failure classes
that need different fixes — {{eq:most-rag-failures-are-invisible-end-to-end}}.

**Readiness rubric.** {{cite:breck2017}}'s checklist approach — score the *process* rather
than the outputs. Complementary and it does not detect a defect.

**Behavioural test suites.** {{cite:ribeiro2020checklist}}'s capability matrix. This is the
invariant style generalised, and it is the best-understood member of the portfolio.

**Continuous online evaluation only.** Skip the pre-deploy gates and measure in production.
Removes the loop-occupancy cost and accepts every escape multiplier at 17×.

**Buy an evaluation platform.** Fast to start, and the instruments it ships are the ones with
the largest overlap, so the coverage gain is smaller than the instrument count suggests.

## 14. Evaluation

Draw your own instrument-by-class matrix and compute the union. Most teams have never seen
their coverage as a number.

Compute the greedy order for your costs and compare it against the order you actually built
in.

Report coverage per class, not in aggregate. The aggregate hides which class has nothing
pointed at it.

Measure each gate's false-positive rate and compare against its computed threshold. Turn the
ones above it advisory.

Total your gate hours per release and put the number next to your loop period. Do it before
the complaint rather than after.

## 15. Advanced Concepts

The independence assumed between instruments' detections is the model's weakest joint, and it
fails in the direction that makes the portfolio look better than it is. Two instruments that
both look at the output text will miss the same subtle failures for the same reasons, so
$\prod(1 - d_{ic})$ understates the joint miss probability. The correction is largest exactly
where the portfolio looks strongest — the classes with several instruments pointed at them —
and smallest where it looks weakest. **The realistic coverage curve is flatter than the
greedy build suggests**, which strengthens the argument for buying a slice of an instrument
that works by a genuinely different mechanism, and weakens the argument for adding a fourth
text-inspecting judge.

The greedy build is near-optimal only because coverage is submodular, and that property fails
if instruments *interact* — if, for instance, a judge is more accurate on outputs that have
already passed a schema check, because the malformed cases were confusing it. Positive
interactions of that kind are common and make the true optimum better than greedy rather than
worse, which is a rare direction for a modelling assumption to fail in. It also suggests an
ordering heuristic the model does not capture: **put the instruments that clean the input to
other instruments first**, independent of their own coverage.

The placement result assumes the escape multipliers are exogenous, and {{ch:ops-deployment}}
showed they are not — the multiplier at canary depends on the canary's size and duration, both
of which are design choices. So placement and canary design are one optimisation rather than
two, and solving them separately leaves value on the table. A larger canary raises the escape
multiplier there, which pulls gates earlier; a longer bake lowers it, which pushes them later.
Neither team usually knows the other is setting a parameter in their problem.

Finally, the loop-occupancy result generalises past evaluation and is the right note for this
part to end on. Every quality mechanism in this book — gates, reviews, canaries, spot-checks,
annotation programmes — consumes the delivery loop's period, and the period is the thing
{{ch:ops-lifecycle}} found nobody measures because nobody is working during it. So quality
mechanisms are approved against a budget and paid for out of a resource that has no budget.
**The binding constraint on an evaluation framework is not money and never was**, and a team
that computes the hours column first will build a different and better framework than one that
computes the dollars column first.

## 16. Connection to Previous Chapters

{{eq:rework-cost-is-set-by-detection-lateness}} from {{ch:ops-lifecycle}} supplies the escape
multipliers and the correction to "shift left" that this chapter's placement result derives
independently.

{{eq:format-check-is-the-cheapest-gate}} from {{ch:ops-prompt-versioning}} is the first row of
the greedy build, arrived at there from prompt gating and here from portfolio construction.

{{eq:most-rag-failures-are-invisible-end-to-end}} from {{ch:ev-rag}} and
{{eq:outcome-evaluation-credits-lucky-trajectories}} from {{ch:ev-agents}} are why a portfolio
is required rather than preferred: one verdict cannot distinguish failures needing different
fixes.

{{eq:exposure-is-invariant-to-canary-size}} from {{ch:ops-deployment}} sets the escape
multipliers this chapter treats as given, and {{sec:15-advanced-concepts}} argues the two
should be optimised together.

## 17. Exercises

1. Build your own instrument-by-failure-class matrix and compute the union coverage. How far
   is it from the sum?

2. Run the greedy build on your costs. How does the order compare to what you built?

3. For each existing gate, compute run cost at its current stage and at every other stage.
   How many are misplaced?

4. Measure each gate's false-positive rate and compute its threshold. Which should be
   advisory?

5. Model positively-interacting instruments — a judge that is more accurate after a schema
   check — and find how much the greedy order changes.

## 18. Interview Questions

1. Our three evaluations cover 40%, 35% and 30%. What is our coverage?

2. Where would you put an expensive slow check, and why?

3. Should a quality gate block or warn?

4. We built a regression suite first. Was that right?

5. Our CI has fourteen checks and the team ignores half of them. What happened?

6. What is the constraint on how much evaluation we can afford?

## 19. Research Questions

1. How correlated are detections across instruments in practice, and how much does the true
   coverage curve flatten relative to the independent model?

2. Do instruments interact positively — does cleaning the input to a judge improve its
   accuracy enough to change the greedy order?

3. What are realistic false-positive rates for each instrument class, and how many deployed
   gates are above their own threshold?

4. What share of delivery-loop period is consumed by quality mechanisms across organisations,
   and is it measured anywhere?

## 20. Chapter Summary

An evaluation framework is a portfolio, and portfolios have arithmetic.

**Coverage is a union.** Nine instruments summing to **1.827** cover **0.956**
({{eq:coverage-is-a-union-not-a-sum}}) — the overlap exceeds any single instrument's
contribution, because the cheap ones all point at the same visible failures.

Built greedily, the portfolio reaches **0.776** at **250 per thousand items**, and
**100% of that budget is reference-free** ({{eq:optimal-portfolio-is-mostly-reference-free}})
— schema checks, execution grading, judges, invariants, none needing a ground-truth answer.
The frozen regression suite and the human spot-check, which most teams build first, enter
seventh and eighth. The last **0.180** of coverage costs **39×** the first 0.776, so buy a
*slice* of the expensive instruments: they gate nothing and they are the only measurement of
the residual.

Placement is a different problem. Run counts fall from **140** to **4** along the pipeline
while escape multipliers rise from **1.0** to **17.0**, so cheap gates belong early and
expensive ones late ({{eq:gate-placement-is-set-by-cost-times-escape}}) — a 4,100-per-run gate
on every commit costs **574,000** a release. The assembled pipeline costs **39,916** against
**126,480** ungated.

And a gate should block only below its own false-positive threshold — **50.7%** for a schema
check, **14.9%** for a judge ensemble, **10.1%** for a human spot-check
({{eq:a-flaky-gate-has-a-blocking-threshold}}) — a computed rule rather than a category
convention, and one that must be recomputed whenever an earlier gate is added.

The number this chapter ends on is not a coverage or a cost. It is **322 gate hours per
release against an 847-hour loop period**: the framework occupies 38% of the delivery loop,
almost all of it waiting, and it appears on no budget. Every gate is individually justified
and the total was never decided. Which is the honest summary of how evaluation frameworks come
to exist, and the reason to compute the hours column before the dollars one.

Carry forward: **coverage is a union**, and **the constraint is the loop, not the budget**.

## 21. Further Reading

- {{cite:breck2017}} — the closest prior art for a production-readiness checklist, scoring
  process rather than outputs.
- {{cite:ribeiro2020checklist}} — behavioural testing, which is the invariant style
  generalised and the best-understood member of this portfolio.
- {{cite:card2020power}} — the sizing arithmetic that decides whether any of these
  instruments can detect what it is looking for.
- {{cite:es2023ragas}} — reference-free metric design, the property that turns out to
  dominate the optimal portfolio.

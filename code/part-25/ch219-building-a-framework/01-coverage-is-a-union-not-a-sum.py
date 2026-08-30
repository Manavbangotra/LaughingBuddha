# -*- coding: utf-8 -*-
# Extracted from: Chapter 219 — Building an Evaluation Framework from Scratch
# Source: src/.../ch219-building-a-framework.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

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

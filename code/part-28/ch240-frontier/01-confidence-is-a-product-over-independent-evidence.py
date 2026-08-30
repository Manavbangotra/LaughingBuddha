# -*- coding: utf-8 -*-
# Extracted from: Chapter 240 — Reading the Frontier: Established, Emerging, Speculative
# Source: src/.../ch240-frontier.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A claim's tier is a product over independent evidence, and popularity is not one of the terms.

This book cites 331 papers and refused several dozen more. The rule was
mechanical: a claim gets used if it can be checked. This listing makes that rule explicit and
prices it.

Confidence in a claim is a product over independent kinds of evidence -- replication, effect size
against noise, a held-out or pre-registered test, adversarial probing, deployment experience.
Each is a factor between 0 and 1, and missing any one caps the product
(eq:confidence-is-a-product-over-independent-evidence).

The signals the field actually sorts on -- citations, benchmark position, venue -- correlate with
that product much less than they appear to (cite:singh2025leaderboard)
(eq:popularity-is-a-poor-proxy-for-evidence).
"""
# (evidence kind, weight when present, value when absent, what it rules out)
EVIDENCE = [
    ("independent replication",     0.94, 0.28, "a fluke, or one lab's setup"),
    ("effect large against noise",  0.91, 0.34, "a selection artefact"),
    ("held-out or pre-registered",  0.88, 0.41, "search over the test set"),
    ("adversarial or ablated",      0.83, 0.47, "a confound doing the work"),
    ("deployed and survived",       0.86, 0.55, "a benchmark-only result"),
]

TIERS = [("established", 0.62), ("emerging", 0.28), ("speculative", 0.0)]


MAX_RAW = 1.0
for _n, _y, _no, _r in EVIDENCE:
    MAX_RAW *= _y


def confidence(present):
    """Product over evidence factors, normalised so all-present scores 1."""
    c = 1.0
    for i, (name, yes, no, rules) in enumerate(EVIDENCE):
        c *= yes if present[i] else no
    return c / MAX_RAW


def tier_of(c):
    for name, floor in TIERS:
        if c >= floor:
            return name
    return "speculative"


print("What each kind of evidence rules out.")
print()
print(f"{'evidence':>30}{'factor if present':>20}{'factor if absent':>19}"
      f"{'ratio':>9}{'what it rules out':>32}")
print("-" * 110)
for name, yes, no, rules in EVIDENCE:
    print(f"{name:>30}{yes:>20.2f}{no:>19.2f}{yes / no:>9.2f}{rules:>32}")

ALL_YES = confidence([True] * len(EVIDENCE))
ALL_NO = confidence([False] * len(EVIDENCE))
print()
print(f"all five present: {ALL_YES:.4f}; none present: {ALL_NO:.4f}")
print(f"a range of {ALL_YES / ALL_NO:,.0f}x")

print()
print()
print("Five claims, scored.")
print()
CLAIMS = [
    ("attention beats recurrence at scale",   [1, 1, 1, 1, 1]),
    ("this architecture is 12% better",       [0, 0, 1, 1, 0]),
    ("scaling laws hold in this regime",      [1, 1, 1, 0, 1]),
    ("this prompt format is superior",        [0, 0, 0, 1, 0]),
    ("emergence is a metric artefact",        [1, 1, 0, 1, 0]),
    ("this method will generalise",           [0, 0, 0, 0, 0]),
]
print(f"{'claim':>38}{'repl':>7}{'effect':>8}{'held-out':>10}{'adv':>6}"
      f"{'deployed':>10}{'confidence':>13}{'tier':>14}")
print("-" * 106)
scored = {}
for name, bits in CLAIMS:
    c = confidence([bool(b) for b in bits])
    t = tier_of(c)
    scored[name] = (c, t)
    print(f"{name:>38}"
          f"{('yes' if bits[0] else 'no'):>7}{('yes' if bits[1] else 'no'):>8}"
          f"{('yes' if bits[2] else 'no'):>10}{('yes' if bits[3] else 'no'):>6}"
          f"{('yes' if bits[4] else 'no'):>10}{c:>13.4f}{t:>14}")

print()
print(f"established: {sum(1 for n in scored if scored[n][1] == 'established')}"
      f" of {len(CLAIMS)}")
print(f"speculative: {sum(1 for n in scored if scored[n][1] == 'speculative')}"
      f" of {len(CLAIMS)}")

print()
print()
print("Which evidence moves a claim the most, from a middling starting point.")
print()
START = [0, 1, 1, 0, 0]
BASE_C = confidence([bool(b) for b in START])
print(f"starting confidence {BASE_C:.4f} ({tier_of(BASE_C)})")
print()
print(f"{'add this evidence':>30}{'new confidence':>17}{'gain':>10}"
      f"{'new tier':>14}{'cost to obtain':>22}")
print("-" * 93)
COSTS = {
    "independent replication":    "another lab, months",
    "effect large against noise": "more seeds",
    "held-out or pre-registered": "discipline, free",
    "adversarial or ablated":     "an afternoon",
    "deployed and survived":      "a product and a year",
}
lifts = {}
for i, (name, yes, no, rules) in enumerate(EVIDENCE):
    if START[i]:
        continue
    trial = list(START)
    trial[i] = 1
    c = confidence([bool(b) for b in trial])
    lifts[name] = c - BASE_C
    print(f"{name:>30}{c:>17.4f}{c - BASE_C:>+10.4f}"
          f"{tier_of(c):>14}{COSTS[name]:>22}")

BEST_LIFT = max(lifts, key=lambda n: lifts[n])
print()
print(f"largest single lift: {BEST_LIFT} at {lifts[BEST_LIFT]:+.4f}")

print()
print()
print("And what the field actually sorts on.")
print()
SIGNALS = [
    ("citations in the first year",   0.21, "cite:singh2025leaderboard"),
    ("position on a leaderboard",     0.28, "cite:liang2022helm"),
    ("venue and reviewer scores",     0.34, "--"),
    ("the authors' track record",     0.39, "--"),
    ("a public artefact you can run", 0.66, "--"),
    ("someone you trust reproduced it", 0.81, "--"),
]
print(f"{'signal':>36}{'correlation with confidence':>30}{'cost to check':>18}"
      f"{'where':>28}")
print("-" * 112)
COST_CHECK = {
    "citations in the first year": "free",
    "position on a leaderboard": "free",
    "venue and reviewer scores": "free",
    "the authors' track record": "free",
    "a public artefact you can run": "an afternoon",
    "someone you trust reproduced it": "a phone call",
}
for name, corr, where in SIGNALS:
    print(f"{name:>36}{corr:>30.2f}{COST_CHECK[name]:>18}{where:>28}")

FREE_BEST = max((s for s in SIGNALS if COST_CHECK[s[0]] == "free"),
                key=lambda s: s[1])
PAID_BEST = max((s for s in SIGNALS if COST_CHECK[s[0]] != "free"),
                key=lambda s: s[1])
print()
print(f"best free signal: {FREE_BEST[0]} at {FREE_BEST[1]:.2f}")
print(f"best cheap signal: {PAID_BEST[0]} at {PAID_BEST[1]:.2f}")
print(f"a gain of {PAID_BEST[1] - FREE_BEST[1]:.2f} for a phone call")
print("(eq:popularity-is-a-poor-proxy-for-evidence)")

print()
print()
print("What this book's own citation rule cost and bought.")
print()
RULE = [
    ("cited, arXiv, fetched and read",   331, 1.00, "the rule"),
    ("rejected: not on arXiv",            14, 0.00, "standards, laws, blogs"),
    ("rejected: could not fetch",          9, 0.00, "paywalled or moved"),
    ("rejected: claim not in the paper",   6, 0.00, "the abstract said otherwise"),
]
total = sum(n for t, n, k, w in RULE)
print(f"{'outcome':>36}{'count':>9}{'share':>9}{'note':>28}")
print("-" * 82)
for name, n, keep, note in RULE:
    print(f"{name:>36}{n:>9}{n / total:>9.1%}{note:>28}")

rejected = sum(n for t, n, k, w in RULE if k == 0.0)
print()
print(f"{rejected} of {total} candidate citations were refused"
      f" -- {rejected / total:.1%}")
print("the third category is the one that matters: the paper said something else")

print(f"""
The evidence table is the rubric. Each kind of evidence rules out a specific failure, and each
contributes a factor: {0.94:.2f} if present, {0.28:.2f} if absent for independent replication --
**a ratio of {0.94 / 0.28:.2f}**, the largest in the table.

They multiply, and the scale is normalised so that all five present scores {ALL_YES:.4f}. None
present scores {ALL_NO:.4f} -- a range of {ALL_YES / ALL_NO:,.0f}x
(eq:confidence-is-a-product-over-independent-evidence). The product form
is the point: **a claim with four kinds of evidence and no replication is not 80% established**,
it is capped by the missing factor, exactly as ch:rai-oversight's preconditions were.

The claims table applies it. `attention beats recurrence at scale` scores
{scored['attention beats recurrence at scale'][0]:.4f} -- everything present, and it is the kind
of claim this book builds on without hedging. `this method will generalise` scores
{scored['this method will generalise'][0]:.4f}: no evidence of any kind, which is not a criticism
of the claim but a complete description of its standing.

The middle rows are where judgement lives. `emergence is a metric artefact`
(cite:schaeffer2023mirage) scores {scored['emergence is a metric artefact'][0]:.4f} -- replicated,
large, adversarially probed, and neither pre-registered nor deployed. That is `{scored['emergence is a metric artefact'][1]}`,
and this book used it as such: cited, relied on for a mechanism, not treated as settled.

The lift table says where to spend effort. From a middling start of {BASE_C:.4f},
`{BEST_LIFT}` adds {lifts[BEST_LIFT]:+.4f} -- the largest single move available -- and costs
another lab and several months.

`adversarial or ablated` adds {lifts['adversarial or ablated']:+.4f} for **an afternoon**, and
`held-out or pre-registered` is free if decided before the experiment rather than after.
**Two of the three cheapest factors are procedural rather than empirical**, which means most
claims could be a tier higher for no additional research at all.

The signals table is the uncomfortable one. Citations in the first year correlate {0.21:.2f} with
the confidence rubric; leaderboard position {0.28:.2f} (cite:liang2022helm,
cite:singh2025leaderboard); venue and reviewer scores {0.34:.2f}. Those are the free signals and
they are the ones everyone uses.

`{PAID_BEST[0]}` correlates {PAID_BEST[1]:.2f} and costs a phone call -- a gain of
**{PAID_BEST[1] - FREE_BEST[1]:.2f}** over the best free signal
(eq:popularity-is-a-poor-proxy-for-evidence). `a public artefact you can run` correlates
{0.66:.2f} for an afternoon.

**The cheap signals are free and weak; the strong signals are cheap and unused.** That is not a
difficult trade-off, and it is the single most actionable thing in this chapter.

The last table is this book's own accounting, and it is here because a chapter about reading the
frontier should say what its own rule cost. {total} candidate citations, {rejected} refused --
{rejected / total:.1%}. Fourteen were not on arXiv, which excluded some genuinely load-bearing
practitioner material and is stated wherever it mattered. Nine could not be fetched.

And six were refused because **the paper did not say what it was cited for**. That category is
the reason the rule exists. It is small, it is invisible without checking, and every one of those
six would have entered this book as a fact.""")

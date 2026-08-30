# -*- coding: utf-8 -*-
# Extracted from: Chapter 231 — Regulation and Risk Management
# Source: src/.../ch231-regulation.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Most of a compliance package is engineering you should be doing anyway. Recorded at the time.

A conformity assessment asks for evidence: what the system does, what data it was built on, how
it was evaluated, what it does when it fails, who decided what, and what happened afterwards.

Read that list against the previous twenty-six chapters and most of it is already there --
cite:mitchell2019modelcards' disaggregated evaluation, cite:gebru2021datasheets' dataset
record, ch:ev-framework's coverage, ch:sec-permissions' principal chain, ch:rai-privacy's
deletion completeness (eq:most-compliance-evidence-is-engineering-you-already-do).

The catch is in the second word. Evidence has to be *contemporaneous*: a record written at the
time it describes. Retrofitting produces a document rather than a record, and for several
artefacts the underlying facts are no longer recoverable
(eq:evidence-must-be-contemporaneous).
"""
# (obligation, artefact that satisfies it, where the book built it,
#  share already produced by good engineering, retrofit cost multiple)
OBLIGATIONS = [
    ("describe intended purpose and limits", "model card",
     "ch:rai-bias",       0.71,  1.4),
    ("document the training data",           "datasheet",
     "ch:rai-privacy",    0.34,  9.0),
    ("report disaggregated performance",     "model card",
     "ch:rai-bias",       0.62,  2.1),
    ("evidence of testing and validation",   "evaluation report",
     "ch:ev-framework",   0.88,  1.2),
    ("risk management across the lifecycle", "risk register",
     "ch:ops-lifecycle",  0.44,  2.8),
    ("logging sufficient to trace a decision", "principal chain",
     "ch:sec-permissions", 0.39, 11.0),
    ("human oversight arrangements",         "approval design",
     "ch:sec-permissions", 0.66,  1.6),
    ("accuracy, robustness, cybersecurity",  "eval + threat model",
     "ch:sec-threat-model", 0.72, 1.9),
    ("data governance and deletion",         "deletion completeness",
     "ch:rai-privacy",    0.41,  6.5),
    ("post-market monitoring",               "observability",
     "ch:ops-observability", 0.79, 1.5),
    ("incident reporting",                   "incident register",
     "ch:ops-deployment", 0.83,  1.3),
]

print("What a conformity assessment asks for, and what you already have.")
print()
print(f"{'obligation':>42}{'artefact':>24}{'built in':>22}{'covered':>10}")
print("-" * 98)
cov_total = 0.0
for name, art, where, cov, retro in OBLIGATIONS:
    cov_total += cov
    print(f"{name:>42}{art:>24}{where:>22}{cov:>10.0%}")
print("-" * 98)
mean_cov = cov_total / len(OBLIGATIONS)
print(f"{'MEAN COVERAGE':>42}{'':>24}{'':>22}{mean_cov:>10.0%}")

print()
print(f"{len(OBLIGATIONS)} obligations, {mean_cov:.0%} already produced by")
print("engineering practice this book has already argued for on other grounds")

print()
print()
print("The gap, ranked by what is missing.")
print()
gaps = sorted(OBLIGATIONS, key=lambda o: o[3])
print(f"{'obligation':>42}{'covered':>10}{'missing':>10}"
      f"{'retrofit cost multiple':>25}")
print("-" * 87)
for name, art, where, cov, retro in gaps:
    print(f"{name:>42}{cov:>10.0%}{1 - cov:>10.0%}{retro:>24.1f}x")

print()
print(f"the largest gap is `{gaps[0][0]}` at {1 - gaps[0][3]:.0%} missing")
print(f"the most expensive to retrofit is `{max(OBLIGATIONS, key=lambda o: o[4])[0]}`"
      f" at {max(o[4] for o in OBLIGATIONS):.0f}x")

print()
print()
print("Cost of building each artefact now against retrofitting it later.")
print()
BASE = 20_000.0
print(f"{'obligation':>42}{'build now':>12}{'retrofit later':>17}"
      f"{'saving':>12}{'recoverable at all?':>22}")
print("-" * 105)
RECOVERABLE = {
    "document the training data": "partly",
    "logging sufficient to trace a decision": "no",
    "data governance and deletion": "partly",
    "risk management across the lifecycle": "partly",
}
now_total, later_total = 0.0, 0.0
for name, art, where, cov, retro in OBLIGATIONS:
    now = BASE * (1 - cov)
    later = BASE * (1 - cov) * retro
    now_total += now
    later_total += later
    print(f"{name:>42}{now:>12,.0f}{later:>17,.0f}{later - now:>12,.0f}"
          f"{RECOVERABLE.get(name, 'yes'):>22}")
print("-" * 105)
print(f"{'TOTAL':>42}{now_total:>12,.0f}{later_total:>17,.0f}"
      f"{later_total - now_total:>12,.0f}")

print()
print(f"building now: {now_total:,.0f}; retrofitting: {later_total:,.0f}")
print(f"a factor of {later_total / now_total:.1f}")

print()
print()
print("Why retrofit costs what it does: the fact is gone, not just the document.")
print()
LOST = [
    ("which corpus version trained this model", "unrecoverable if unpinned",
     "ch:ops-versioning"),
    ("who approved this decision, and why",     "unrecoverable if unlogged",
     "ch:sec-permissions"),
    ("what the evaluation set looked like then", "unrecoverable if unversioned",
     "ch:ops-prompt-versioning"),
    ("which licences the training data carried", "unrecoverable if unrecorded",
     "ch:rai-privacy"),
    ("what the model's behaviour was at launch", "unrecoverable without a snapshot",
     "ch:ops-deployment"),
    ("how many users were affected by an incident", "recoverable from logs",
     "ch:ops-observability"),
]
print(f"{'the fact':>44}{'status if not recorded':>36}{'chapter':>26}")
print("-" * 106)
for fact, status, ch in LOST:
    print(f"{fact:>44}{status:>36}{ch:>26}")

unrec = sum(1 for f, s, c in LOST if s.startswith("unrecoverable"))
print()
print(f"{unrec} of {len(LOST)} are unrecoverable rather than expensive")
print("(eq:evidence-must-be-contemporaneous)")

print()
print()
print("What a lead time buys, at a fixed classification.")
print()
print(f"{'when you start':>26}{'artefacts contemporaneous':>28}"
      f"{'cost':>12}{'delay at assessment':>22}")
print("-" * 88)
LEAD = [
    ("at design",           1.00,  now_total,        0),
    ("at first deployment", 0.74,  now_total * 1.9,  3),
    ("when a customer asks", 0.41, now_total * 4.4, 11),
    ("when a regulator asks", 0.19, later_total,     22),
]
for name, contemp, c, delay in LEAD:
    print(f"{name:>26}{contemp:>28.0%}{c:>12,.0f}{delay:>19} wks")

print()
print("The first column is what cannot be bought back, and it falls fastest.")

print()
print()
print("And the residual: what no amount of engineering evidence settles.")
print()
RESIDUAL = [
    ("is the classification correct",     "a legal reading",   "no"),
    ("is the risk acceptable",            "a policy judgement", "no"),
    ("is the human oversight meaningful", "measurable",        "yes"),
    ("is the evaluation adequate",        "measurable",        "yes"),
    ("was consent valid",                 "a legal reading",   "no"),
    ("is the system's purpose as stated", "a governance fact", "partly"),
]
print(f"{'question':>38}{'kind of question':>22}{'engineering settles it?':>26}")
print("-" * 86)
settle = sum(1 for q, k, s in RESIDUAL if s == "yes")
for q, k, s in RESIDUAL:
    print(f"{q:>38}{k:>22}{s:>26}")

print()
print(f"{settle} of {len(RESIDUAL)} are settled by measurement")

print(f"""
The obligation table is the reframing this chapter exists for. Eleven obligations, and
**{mean_cov:.0%} of the evidence is already produced** by practices this book argued for on
entirely other grounds (eq:most-compliance-evidence-is-engineering-you-already-do).

`evidence of testing and validation` is {[o for o in OBLIGATIONS if o[0].startswith('evidence')][0][3]:.0%}
covered by ch:ev-framework's portfolio. `incident reporting` is
{[o for o in OBLIGATIONS if o[0] == 'incident reporting'][0][3]:.0%} covered by an incident
register. `post-market monitoring` is
{[o for o in OBLIGATIONS if o[0] == 'post-market monitoring'][0][3]:.0%} covered by
observability.

None of those was built for a regulator. **A competent engineering practice produces most of a
compliance package as a by-product**, which is a much better position than the usual framing of
compliance as a separate workstream.

The gap table names what is missing, and the ranking is informative.
`{gaps[0][0]}` is {1 - gaps[0][3]:.0%} missing -- ch:rai-privacy found
{0.55:.0%} of a corpus with unresolved licences and no datasheet.
`{gaps[1][0]}` is {1 - gaps[1][3]:.0%} missing, which is ch:sec-permissions' principal chain
recorded {0.19:.0%} of the time.

**The two largest gaps are both records that had to be written when the thing happened**, and
that is the second half of the chapter.

The cost table prices it. Building the missing artefacts now costs {now_total:,.0f}.
Retrofitting them later costs {later_total:,.0f} -- **a factor of
{later_total / now_total:.1f}** -- and the last column is the one that matters more than the
money.

For four of eleven obligations the answer is not "yes, expensively". It is `partly` or `no`,
because the fact itself is gone.

The lost-facts table makes that concrete, and every row points at a chapter that already
recommended recording it for an unrelated reason. Which corpus version trained this model is
ch:ops-versioning's artefact pinning. Who approved a decision is ch:sec-permissions' principal
chain. What the evaluation set looked like is ch:ops-prompt-versioning's coverage.
{unrec} of {len(LOST)} are **unrecoverable rather than expensive**
(eq:evidence-must-be-contemporaneous).

That is the whole argument for doing this early, and it is not a compliance argument. Every one
of those records was already worth having: the corpus version for reproducibility, the principal
chain for incident triage, the evaluation snapshot for regression detection. Compliance is the
second customer for a record you needed anyway.

The lead-time table is how to present the decision. Starting at design gives
{LEAD[0][1]:.0%} contemporaneous artefacts at {LEAD[0][2]:,.0f}. Starting when a regulator asks
gives {LEAD[3][1]:.0%} at {LEAD[3][2]:,.0f} and {LEAD[3][3]} weeks of delay.

**The contemporaneity column is what cannot be bought back**, and it falls faster than the cost
rises.

The residual table is the honest ending. {settle} of {len(RESIDUAL)} questions are settled by
measurement -- is the oversight meaningful, is the evaluation adequate. The rest are legal
readings and policy judgements: whether the classification is right, whether the risk is
acceptable, whether consent was valid.

**Engineering evidence settles the questions it can settle and does not settle the
classification** -- which is ch:rai-regulation's first listing, where the money was. So the
right allocation is: build the evidence early because it is cheap and dual-purpose, and take the
classification question to someone qualified to answer it, early, in writing.""")

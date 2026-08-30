---
id: rai-regulation
number: 231
part: XXVII
tier: full
status: draft
requires: [reproducibility-is-a-product-over-artefacts, audit-completeness-requires-the-principal-chain,
           deletion-is-a-product-over-derived-artefacts, coverage-is-a-union-not-a-sum]
provides: [compliance-cost-is-a-step-function, tier-boundaries-create-design-incentives,
           most-compliance-evidence-is-engineering-you-already-do, evidence-must-be-contemporaneous]
citations: [breck2017, mitchell2019modelcards, gebru2021datasheets, paleyes2020deployment]
---

## 1. Learning Objectives

By the end of this chapter you will be able to describe the risk-tier structure common to AI
regulatory frameworks and price the step between tiers; identify which classification factors
are judgements rather than measurements, and how much the disagreement is worth; compute the
value of designing to stay below a tier boundary and distinguish aligned from unaligned
responses; map conformity-assessment obligations to engineering artefacts you already produce;
and compute the cost and recoverability of retrofitting evidence that must be contemporaneous.

## 2. Why This Matters

> **NOTE** — This chapter treats the *structure* of AI regulation — risk tiers, obligations,
> conformity assessment, evidence — as a design problem. It quotes no regulatory text. The
> instruments in circulation are not arXiv preprints, so under this book's verification rule
> they could not be cited, and nothing here should be read as an interpretation of any specific
> law. Take the classification question to a qualified adviser; take the evidence question to
> your engineering backlog.

Every framework in circulation shares a structure: a system is assigned a risk tier and the
tier determines the obligations. That makes compliance cost a **step function**. Minimal risk
carries 2 obligations and **$36,000** over three years; high risk carries 21 and **$1,640,000**,
plus **22 weeks** before first deployment ({{eq:compliance-cost-is-a-step-function}}).

**The step is $1,473,000 and 19 weeks**, and the classification that decides it turns on
judgements — "materially informs", "easily obtain human review" — where four reasonable readers
reach three different conclusions about the same system.

A step function against an arguable classification creates an incentive. Designing to stay below
the boundary is worth **$717,000** here ({{eq:tier-boundaries-create-design-incentives}}), and
three of the five available moves change a label rather than a risk.

The better news is in the second half. Eleven conformity-assessment obligations are **62%**
already satisfied by engineering this book argued for on other grounds
({{eq:most-compliance-evidence-is-engineering-you-already-do}}). The catch is that evidence must
be contemporaneous: retrofitting costs **3.3×**, and for **5 of 6** critical facts the answer is
not "expensive" but "gone" ({{eq:evidence-must-be-contemporaneous}}).

## 3. Prerequisites

{{eq:reproducibility-is-a-product-over-artefacts}} from {{ch:ops-versioning}} is why "which
corpus version trained this model" is unrecoverable rather than expensive — the artefacts were
not pinned and cannot be reconstructed.

{{eq:audit-completeness-requires-the-principal-chain}} from {{ch:sec-permissions}} is the
obligation with the highest retrofit multiple here, at 11×, because a chain not propagated at
the time cannot be assembled afterwards.

{{eq:deletion-is-a-product-over-derived-artefacts}} from {{ch:rai-privacy}} is the data-governance
obligation, and the completeness number is the evidence.

{{eq:coverage-is-a-union-not-a-sum}} from {{ch:ev-framework}} supplies the testing-and-validation
evidence, which is the best-covered obligation in the table at 88%.

{{cite:breck2017}}'s readiness rubric is the closest engineering analogue to a conformity
assessment and predates all of the regulatory instruments.

## 4. Intuitive Explanation

Regulatory frameworks for AI differ in detail and share a shape. A system is placed in a risk
tier by its purpose and its context; the tier determines a set of obligations; the obligations
determine cost and lead time.

That is a reasonable design. It also produces a specific engineering consequence that nobody
legislates for: **cost is a step function.**

Price the tiers. Minimal risk: two obligations, $36,000 over three years, no delay. Limited
risk: six obligations, $167,000, three weeks. High risk: twenty-one obligations, $1,640,000,
twenty-two weeks before you can deploy at all.

The step from limited to high is $1,473,000 and nineteen weeks. There is no intermediate tier
for a system that is nearly high-risk — that is what a tier structure means.

Now look at what decides which side of the step you are on. Does the system make or materially
inform a decision about a person? Does it operate in a listed domain? Can the person easily
obtain a human review? Is the output advisory?

None of those is a measurement. They are readings, and reasonable people read them differently.
The builder reads their system as advisory: a human decides. The buyer reads it as materially
informing, because that is what they bought it for. The regulator reads the domain. The lawyer
writes down both and adds "depends on deployment".

Four readers, three conclusions, and the disagreement is worth $1,473,000 and nineteen weeks —
more than most teams' annual budget for the feature under discussion.

What does that do to design?

Price the options. Ship as specified: 100% of capability, high risk, $1,640,000, net
$2,560,000. Require a human to confirm every decision: 82% of capability, limited risk,
$167,000, net $3,277,000.

**The boundary is worth $717,000**, attached to a design decision made for reasons that have
nothing to do with what the system does.

Now the honest part. Are the moves that stay below the boundary good moves?

Some are. Adding a human confirmation step is a genuine safety gain. Narrowing the deployment
domain is genuinely reduced coverage and reduced risk.

Some are not. Calling the output advisory changes a label. Splitting one system into two that
each sit below the line changes an org chart. Keeping a human in the loop who always agrees is
{{ch:sec-permissions}}'s rubber stamp — which that chapter showed still produces a decision
record, a human in the causal chain and a delay, while producing no review and no rejection.
It satisfies the obligation's form exactly.

Three of five change a label rather than a risk, and all five sit below the same boundary.
**A step function in cost against an arguable classification will find the argument**, and the
unaligned moves are cheaper than the aligned ones.

It is worth being fair about why. Nobody has to intend this. A product team asks whether the
feature can ship in three weeks instead of twenty-two; someone notices the classification turns
on whether the output is advisory; the specification acquires a confirmation step. Every
individual step in that sequence is reasonable, and the sequence ends with a control whose
purpose is the classification rather than the risk.

One more number before moving on, because it should inform where a team spends its energy.
Across simple, typical and complex systems, the *tier* accounts for about 98% of the cost
variance and the system's own complexity accounts for the rest.

Which is strange and true: **the classification matters more than the system.** A team that
spends a quarter arguing about the tier is allocating effort correctly by the arithmetic and
badly by any other measure.

The way out is not to argue better. It is to notice what the obligations actually ask for.

Take the standard list. Describe the intended purpose and limits. Document the training data.
Report disaggregated performance. Evidence of testing and validation. Risk management across the
lifecycle. Logging sufficient to trace a decision. Human oversight arrangements. Accuracy,
robustness, cybersecurity. Data governance and deletion. Post-market monitoring. Incident
reporting.

Now read that against the last twenty-six chapters.

"Describe intended purpose and limits" is a model card, which {{ch:rai-bias}} recommended for
disaggregated reporting. "Report disaggregated performance" is the same artefact. "Document the
training data" is {{cite:gebru2021datasheets}}'s datasheet, which {{ch:rai-privacy}} needed for
the licence question. "Evidence of testing and validation" is {{ch:ev-framework}}'s portfolio.
"Logging sufficient to trace a decision" is {{ch:sec-permissions}}' principal chain. "Data
governance and deletion" is {{ch:rai-privacy}}'s completeness figure. "Post-market monitoring"
is {{ch:ops-observability}}. "Incident reporting" is {{ch:ops-deployment}}'s register.

Coverage: 88% for testing evidence, 83% for incident reporting, 79% for monitoring, 72% for
robustness. Mean across all eleven: **62%.**

None of that was built for a regulator. **A competent engineering practice produces most of a
compliance package as a by-product.**

The gaps are informative too. Document the training data: 66% missing —
{{ch:rai-privacy}} found 55% of a corpus with unresolved licences and no datasheet. Logging
sufficient to trace a decision: 61% missing — {{ch:sec-permissions}} found the principal chain
recorded 19% of the time. Data governance and deletion: 59% missing.

The two largest gaps are both records that had to be written when the thing happened. Which is
the second half of the chapter and the more important half.

Evidence has to be **contemporaneous**. A conformity assessment does not want a document
describing what you believe happened; it wants a record made at the time. That distinction is
what makes retrofitting expensive, and for some artefacts impossible.

Price it. Building the missing artefacts now: $88,600. Retrofitting later: $421,540. A factor
of 3.3.

But the money is the smaller problem. Look at recoverability. Which corpus version trained this
model — unrecoverable if the artefacts were not pinned. Who approved this decision and why —
unrecoverable if the chain was not logged. What the evaluation set looked like then —
unrecoverable if it was not versioned. Which licences the training data carried — unrecoverable
if not recorded at ingest. What the model's behaviour was at launch — unrecoverable without a
snapshot.

Five of six are **unrecoverable rather than expensive.**

And here is the point that makes this a chapter about engineering rather than about law: every
one of those records was already worth having for a non-regulatory reason. The corpus version
for reproducibility. The principal chain for incident triage. The evaluation snapshot for
regression detection. The launch behaviour for rollback comparison.

**Compliance is the second customer for a record you needed anyway.**

That reframing is worth more than it sounds, because it changes who owns the work and when it
happens. A compliance programme staffed separately, starting when a regulator or a buyer asks,
is building artefacts against a deadline from facts that have partly expired. The same artefacts
built by the engineering team, at design time, for reproducibility and triage and regression
detection, are cheaper, more accurate, and already contemporaneous when the question arrives.

Which gives the lead-time table its shape. Starting at design: 100% of artefacts
contemporaneous, $88,600. Starting when a regulator asks: 19% contemporaneous, $421,540, and
twenty-two weeks of delay.

The contemporaneity column is what cannot be bought back, and it falls faster than the cost
rises.

Finally, the residual — what no amount of engineering evidence settles. Is the human oversight
meaningful? Measurable, and {{ch:sec-permissions}} showed how. Is the evaluation adequate?
Measurable, and {{ch:ev-framework}} showed how. Is the classification correct? A legal reading.
Is the risk acceptable? A policy judgement. Was consent valid? A legal reading.

Two of six are settled by measurement. **Engineering evidence settles the questions it can
settle and does not settle the classification** — which is where the money was.

So the allocation is: build the evidence early, because it is cheap, dual-purpose and
perishable. And take the classification question to someone qualified, early, in writing.

## 5. Formal Explanation

**Step-function cost.** Let a system have a feature vector $x$ and a classifier $T(x)$ into
tiers $t$ with obligation sets $O_t$ and costs $C_t = c^{\text{one}}_t + h \cdot
c^{\text{ann}}_t$. Since $|O_t|$ is defined per tier and not per system, $C$ is piecewise
constant in $x$ with discontinuities at the tier boundaries. The gradient $\partial C/\partial
x$ is zero almost everywhere and unbounded at the boundary, which is exactly the condition that
produces boundary-seeking behaviour under optimisation.

**Boundary value.** With revenue $R$ and capability $\kappa(d)$ under design $d$, net value is
$R\kappa(d) - C_{T(d)}$. The boundary is worth
$\max_d [R\kappa(d) - C_{T(d)}] - [R\kappa(1) - C_{\text{high}}]$, which is positive whenever
some design sacrifices less capability than the tier step costs. Since $\kappa$ is continuous
and $C$ is a step, such a design exists whenever the step exceeds $R \cdot \Delta\kappa$ for
the cheapest tier-reducing modification.

**Evidence coverage.** For obligations $j$ with artefact $a_j$ and pre-existing coverage
$\gamma_j$, the outstanding work is $\sum_j w_j (1 - \gamma_j)$. This is
{{eq:coverage-is-a-union-not-a-sum}}'s structure with a different index set: obligations map
many-to-one onto artefacts, so building one artefact discharges several obligations, and the
correct planning unit is the artefact rather than the obligation.

**Contemporaneity.** An artefact is contemporaneous if it was produced from state available at
the time it describes. Retrofit cost is $\rho_j$ times the build cost when the underlying state
persists, and unbounded when it does not. The distinguishing question is whether the fact was
*recorded* or merely *true*: a fact that was true and unrecorded leaves no residue in a system
that did not retain it.

## 6. Mathematical Foundation

Cost as a step function of a classification:

$$C(x) = C_{T(x)}, \qquad \frac{\partial C}{\partial x} = 0 \ \text{a.e.}, \qquad \Delta C_{\text{limited} \to \text{high}} = \$1{,}473{,}000$$ (eq:compliance-cost-is-a-step-function)

with a 19-week deployment delay attached to the same step.

The incentive that creates:

$$V^\star = \max_d\left[R\kappa(d) - C_{T(d)}\right] - \left[R\kappa(1) - C_{\text{high}}\right] = \$717{,}000$$ (eq:tier-boundaries-create-design-incentives)

Three of five tier-reducing designs change a label rather than a risk.

Evidence already produced:

$$\bar\gamma = \frac{1}{|J|}\sum_j \gamma_j = 62\%, \qquad \text{outstanding} = \sum_j w_j(1 - \gamma_j)$$ (eq:most-compliance-evidence-is-engineering-you-already-do)

with 88% for testing evidence and 34% for the training-data record.

And the perishability of the rest:

$$C^{\text{retro}}_j = \rho_j C^{\text{now}}_j \ \text{if the state persists}, \qquad = \infty \ \text{otherwise}$$ (eq:evidence-must-be-contemporaneous)

**$88,600** now against **$421,540** later — and **5 of 6** critical facts are unrecoverable
rather than expensive.

## 7. Internal Mechanics

Why do tier boundaries attract design? For the same reason any threshold does: a discontinuity
in an objective is a magnet under optimisation, and the optimiser here is a product team with a
budget. Nobody has to intend it. The team asks "can we ship this in three weeks instead of
twenty-two," someone notices the classification turns on whether the output is advisory, and
the specification acquires a confirmation step.

Whether that is good depends entirely on whether the confirmation step is real, and
{{ch:sec-permissions}} gave the test: measure the rejection rate and the time per item. A
confirmation step with a near-zero rejection rate and seconds per item satisfies the
obligation's form and produces none of its substance — and it is cheaper than the alternatives,
which is why it is the equilibrium.

The evidence-coverage result has a mechanism worth naming because it changes how a compliance
programme should be staffed. Obligations map many-to-one onto artefacts: "describe intended
purpose" and "report disaggregated performance" are both a model card;
{{eq:coverage-is-a-union-not-a-sum}}'s union structure applies, and the planning unit is the
*artefact*. A programme organised as a list of obligations produces duplicated work and a
programme organised as a list of artefacts does not.

Contemporaneity has the sharpest mechanism in the chapter. There are two kinds of missing
evidence: the document was not written but the facts persist, and the facts do not persist.
Only the first is a cost. The second is a different situation entirely, and the distinguishing
question is whether anything in the system *retained* the fact.

That maps precisely onto {{ch:ops-versioning}}'s artefact list. A corpus version is recoverable
if it was pinned and gone if it was not; a principal chain is recoverable if a header was
propagated and gone if it was not. **The regulatory obligation and the engineering practice
want the same record for different reasons**, and the engineering reason arrives years earlier.

Finally, why the classification dominates the cost variance. Obligations are defined per tier,
so a simple high-risk system and a complex high-risk system carry the same twenty-one
obligations; complexity scales the cost of satisfying each one, but the count is fixed by the
tier. That makes tier the multiplicative factor and complexity the multiplicand, and a factor
that ranges over 45× dominates one that ranges over 3×.

## 8. Implementation

The first listing prices the tier structure.

```python {tier=A name=C:/Users/MANAVB~1/AppData/Local/Temp/claude/C--Github-LaughingBuddha/30a87753-43a8-48c3-8378-261faf976dbb/scratchpad/jd1}
"""Compliance cost is a step function of a classification, and the classification is arguable.

Every AI regulatory framework in circulation works the same way structurally: a system is
assigned to a risk tier by its purpose and context, and the tier determines the obligations.
That is a sensible design and it has a consequence nobody legislates for -- **cost jumps at the
boundary** (eq:compliance-cost-is-a-step-function).

A step function in cost against a contested classification produces an incentive to sit just
below the step, and that incentive shapes product decisions long before a regulator sees
anything (eq:tier-boundaries-create-design-incentives).

This listing prices the tiers, measures how much the classification actually turns on, and
computes what designing to stay below a boundary is worth.
"""
# (tier, obligations count, one-off cost, annual cost, time to first deploy in weeks)
TIERS = [
    ("minimal risk",     2,     18_000,     6_000,  0),
    ("limited risk",     6,     74_000,    31_000,  3),
    ("high risk",       21,    620_000,   340_000, 22),
    ("prohibited",       0,          0,         0, None),
]

print("Obligations and cost by tier.")
print()
print(f"{'tier':>18}{'obligations':>14}{'one-off':>12}{'annual':>11}"
      f"{'delay (weeks)':>16}{'3-year total':>15}")
print("-" * 86)
cost = {}
for name, obl, one, ann, delay in TIERS:
    if delay is None:
        cost[name] = None
        print(f"{name:>18}{'--':>14}{'--':>12}{'--':>11}{'--':>16}"
              f"{'cannot deploy':>15}")
        continue
    total = one + 3 * ann
    cost[name] = (obl, one, ann, delay, total)
    print(f"{name:>18}{obl:>14}{one:>12,}{ann:>11,}{delay:>16}"
          f"{total:>15,}")

print()
print(f"the step from limited to high risk is "
      f"{cost['high risk'][4] - cost['limited risk'][4]:,} over three years")
print(f"and {cost['high risk'][3] - cost['limited risk'][3]} weeks of delay")

print()
print()
print("What decides the tier: features of the system, each individually arguable.")
print()
FACTORS = [
    ("makes or materially informs a decision about a person", 0.62, "high"),
    ("operates in a listed domain (credit, hiring, health)",  0.71, "high"),
    ("the person cannot easily obtain a human review",        0.44, "high"),
    ("output is advisory and a human decides",                0.55, "limited"),
    ("interacts directly with the public",                    0.83, "limited"),
    ("generates content that could be mistaken for human",    0.77, "limited"),
    ("used only internally by trained staff",                 0.29, "minimal"),
]
print(f"{'factor':>56}{'present?':>10}{'pushes toward':>16}")
print("-" * 82)
for name, p, tier in FACTORS:
    print(f"{name:>56}{p:>10.2f}{tier:>16}")

print()
print("Each row is a judgement, and the judgements are made by different")
print("people at different times.")

print()
print()
print("Two readings of the same system.")
print()
READINGS = [
    ("the builder's reading",   "advisory, human decides", "limited risk"),
    ("the buyer's reading",     "materially informs",      "high risk"),
    ("the regulator's reading", "listed domain",           "high risk"),
    ("the lawyer's reading",    "depends on deployment",   "both, in writing"),
]
print(f"{'who is reading':>26}{'the operative fact':>28}{'conclusion':>20}"
      f"{'3-year cost':>15}")
print("-" * 89)
for who, fact, concl in READINGS:
    c = cost.get(concl)
    cs = f"{c[4]:,}" if c else "contested"
    print(f"{who:>26}{fact:>28}{concl:>20}{cs:>15}")

print()
print(f"the disagreement is worth "
      f"{cost['high risk'][4] - cost['limited risk'][4]:,} and "
      f"{cost['high risk'][3] - cost['limited risk'][3]} weeks")

print()
print()
print("So what is designing to stay below the boundary worth?")
print()
DESIGNS = [
    ("ship as specified",                    1.00, "high risk"),
    ("require a human to confirm every decision", 0.82, "limited risk"),
    ("restrict to internal users only",      0.41, "minimal risk"),
    ("remove the listed-domain use case",    0.68, "limited risk"),
    ("advisory output, no ranking",          0.74, "limited risk"),
]
REVENUE = 4_200_000.0
print(f"{'design choice':>44}{'capability kept':>18}{'tier':>15}"
      f"{'3-year cost':>14}{'net value':>13}")
print("-" * 106)
des = {}
for name, cap, tier in DESIGNS:
    c = cost[tier][4]
    net = REVENUE * cap - c
    des[name] = (cap, tier, c, net)
    print(f"{name:>44}{cap:>18.0%}{tier:>15}{c:>14,}{net:>13,.0f}")

best = max(des, key=lambda n: des[n][3])
print()
print(f"highest net value: {best} at {des[best][3]:,.0f}")
print(f"shipping as specified: {des['ship as specified'][3]:,.0f}")
print(f"the boundary is worth "
      f"{des[best][3] - des['ship as specified'][3]:,.0f} in this model")

print()
print()
print("Which is the incentive the tier structure creates, whether or not")
print("anybody intended it.")
print()
INCENTIVES = [
    ("add a human confirmation step",   "genuine safety gain",  "aligned"),
    ("call the output advisory",        "changes a label",      "not aligned"),
    ("narrow the deployment domain",    "less coverage",        "aligned"),
    ("split into two systems below the line", "same behaviour", "not aligned"),
    ("keep a human who always agrees",  "ch:sec-permissions",   "not aligned"),
]
print(f"{'what the incentive produces':>44}{'what actually changes':>26}"
      f"{'with the regulation?':>23}")
print("-" * 93)
for name, what, aligned in INCENTIVES:
    print(f"{name:>44}{what:>26}{aligned:>23}")

print()
print("Three of five change a label rather than a risk, and all five sit")
print("below the same boundary.")

print()
print()
print("And the sensitivity that matters: how much of the cost is the tier")
print("versus the system.")
print()
print(f"{'system complexity':>22}{'minimal':>12}{'limited':>12}{'high':>12}"
      f"{'tier share of variance':>28}")
print("-" * 86)
for label, mult in (("simple", 0.6), ("typical", 1.0), ("complex", 1.9)):
    row = [cost[t][4] * mult for t in ("minimal risk", "limited risk", "high risk")]
    spread_tier = max(row) - min(row)
    print(f"{label:>22}{row[0]:>12,.0f}{row[1]:>12,.0f}{row[2]:>12,.0f}"
          f"{spread_tier / max(row):>27.0%}")

print(f"""
The tier table is the structure every framework in circulation shares. Minimal risk carries
{cost['minimal risk'][0]} obligations and a three-year cost of
{cost['minimal risk'][4]:,}; high risk carries {cost['high risk'][0]} and
{cost['high risk'][4]:,}, plus {cost['high risk'][3]} weeks before first deployment.

**The step from limited to high is {cost['high risk'][4] - cost['limited risk'][4]:,} and
{cost['high risk'][3] - cost['limited risk'][3]} weeks** (eq:compliance-cost-is-a-step-function),
and it is a step rather than a slope: there is no intermediate obligation set for a system that
is nearly high-risk.

The factor table is what decides which side of the step a system falls on, and every row is a
judgement. Does the system "materially inform" a decision, or is it advisory? Can the person
"easily obtain" human review? Is the domain listed?

Those are not measurements. They are readings, and the readings table shows four reasonable
people reaching three different conclusions about the same system. The builder reads it as
advisory. The buyer reads it as materially informing, because that is why they bought it. The
regulator reads the domain. The lawyer writes down both.

**The disagreement is worth {cost['high risk'][4] - cost['limited risk'][4]:,} and
{cost['high risk'][3] - cost['limited risk'][3]} weeks**, which is more than most teams' annual
engineering budget for the feature under discussion.

The design table is what that produces. Shipping as specified keeps {1.00:.0%} of capability
and costs {des['ship as specified'][2]:,} -- a net of
{des['ship as specified'][3]:,.0f}. `{best}` keeps {des[best][0]:.0%} and costs
{des[best][2]:,}, netting {des[best][3]:,.0f}.

**The boundary is worth {des[best][3] - des['ship as specified'][3]:,.0f}**
(eq:tier-boundaries-create-design-incentives), which is a large number attached to a design
decision that has nothing to do with what the system does.

The incentive table is the honest reading of that. Two of the five moves are genuinely aligned
with the regulation's purpose: adding a human confirmation step is a real safety gain, and
narrowing the deployment domain is real reduced coverage. Three are not. Calling the output
advisory changes a label. Splitting into two systems below the line changes an org chart.
Keeping a human who always agrees is ch:sec-permissions' rubber stamp -- which that chapter
showed still produces a decision record, a human in the causal chain and a delay, while
producing no review and no rejection. It satisfies the obligation's form exactly.

**A step function in cost against an arguable classification will find the argument**, and the
three unaligned rows are cheaper than the two aligned ones.

The last table is the sensitivity that should inform how much energy goes into the
classification argument versus the engineering. Across simple, typical and complex systems, the
tier accounts for {(cost['high risk'][4] - cost['minimal risk'][4]) / cost['high risk'][4]:.0%}
of the cost variance and the system's own complexity accounts for the rest.

Which is a strange thing to be true and it is true: **the classification matters more than the
system**, and a team that spends a quarter arguing about the tier is allocating effort
correctly by the arithmetic and incorrectly by any other measure.

The way out is not to argue better. It is to notice that most of what a high-risk tier requires
is documentation of engineering that a competent team does anyway -- which is
ch:rai-regulation's second listing.""")
```

## 9. Practical Example

Obligations and cost by tier:

```
              tier   obligations     one-off     annual   delay (weeks)   3-year total
--------------------------------------------------------------------------------------
      minimal risk             2      18,000      6,000               0         36,000
      limited risk             6      74,000     31,000               3        167,000
         high risk            21     620,000    340,000              22      1,640,000
        prohibited            --          --         --              --  cannot deploy
```

**The step from limited to high is $1,473,000 and 19 weeks**
({{eq:compliance-cost-is-a-step-function}}) — a step, not a slope.

```
                                                  factor  present?   pushes toward
----------------------------------------------------------------------------------
   makes or materially informs a decision about a person      0.62            high
    operates in a listed domain (credit, hiring, health)      0.71            high
                  output is advisory and a human decides      0.55         limited
                   used only internally by trained staff      0.29         minimal

            who is reading          the operative fact          conclusion    3-year cost
-----------------------------------------------------------------------------------------
     the builder's reading     advisory, human decides        limited risk        167,000
       the buyer's reading          materially informs           high risk      1,640,000
   the regulator's reading               listed domain           high risk      1,640,000
```

**Four readers, three conclusions**, and the disagreement is worth $1,473,000.

```
                               design choice   capability kept           tier   3-year cost    net value
----------------------------------------------------------------------------------------------------------
                           ship as specified              100%      high risk     1,640,000    2,560,000
   require a human to confirm every decision               82%   limited risk       167,000    3,277,000
             restrict to internal users only               41%   minimal risk        36,000    1,686,000
                 advisory output, no ranking               74%   limited risk       167,000    2,941,000
```

**The boundary is worth $717,000** ({{eq:tier-boundaries-create-design-incentives}}).

```
                 what the incentive produces     what actually changes   with the regulation?
---------------------------------------------------------------------------------------------
               add a human confirmation step       genuine safety gain                aligned
                  call the output advisory          changes a label             not aligned
    split into two systems below the line            same behaviour             not aligned
             keep a human who always agrees     ch:sec-permissions              not aligned
```

**Three of five change a label rather than a risk.**

The second listing maps obligations to artefacts.

```python {tier=A name=C:/Users/MANAVB~1/AppData/Local/Temp/claude/C--Github-LaughingBuddha/30a87753-43a8-48c3-8378-261faf976dbb/scratchpad/jd2}
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
```

```
                                obligation                artefact              built in   covered
--------------------------------------------------------------------------------------------------
        evidence of testing and validation       evaluation report       ch:ev-framework       88%
                        incident reporting       incident register     ch:ops-deployment       83%
                    post-market monitoring           observability  ch:ops-observability       79%
       accuracy, robustness, cybersecurity     eval + threat model   ch:sec-threat-model       72%
              data governance and deletion   deletion completeness        ch:rai-privacy       41%
    logging sufficient to trace a decision         principal chain    ch:sec-permissions       39%
                document the training data               datasheet        ch:rai-privacy       34%
--------------------------------------------------------------------------------------------------
                             MEAN COVERAGE                                                     62%
```

**62% is already produced by engineering argued for on other grounds**
({{eq:most-compliance-evidence-is-engineering-you-already-do}}).

```
                                obligation   build now   retrofit later      saving   recoverable at all?
---------------------------------------------------------------------------------------------------------
                document the training data      13,200          118,800     105,600                partly
    logging sufficient to trace a decision      12,200          134,200     122,000                    no
              data governance and deletion      11,800           76,700      64,900                partly
        evidence of testing and validation       2,400            2,880         480                   yes
```

```
                                    the fact              status if not recorded                   chapter
----------------------------------------------------------------------------------------------------------
    which corpus version trained this model           unrecoverable if unpinned         ch:ops-versioning
        who approved this decision, and why           unrecoverable if unlogged        ch:sec-permissions
    which licences the training data carried         unrecoverable if unrecorded            ch:rai-privacy
```

**5 of 6 are unrecoverable rather than expensive**
({{eq:evidence-must-be-contemporaneous}}) — and every row points at a chapter that already
recommended recording it for an unrelated reason.

```
            when you start   artefacts contemporaneous        cost   delay at assessment
-----------------------------------------------------------------------------------------
                 at design                        100%      88,600               0 wks
      at first deployment                          74%     168,340               3 wks
     when a regulator asks                         19%     421,540              22 wks
```

```
                              question      kind of question   engineering settles it?
--------------------------------------------------------------------------------------
        is the classification correct       a legal reading                        no
       is the human oversight meaningful          measurable                       yes
          is the evaluation adequate              measurable                       yes
                     was consent valid       a legal reading                        no
```

**2 of 6 are settled by measurement.**

## 10. Production Considerations

Get the classification in writing, early, from someone qualified. It is 98% of the cost
variance and no amount of engineering settles it.

Plan by artefact, not by obligation. Obligations map many-to-one onto artefacts and an
obligation-shaped plan duplicates work.

Build the perishable records first. Corpus pinning, principal chains, evaluation snapshots — the
ones where the answer to "can we get this later" is no.

Write the datasheet at ingest. It is the largest gap, the second-most expensive retrofit, and
the only moment the information exists.

Do not accept a confirmation step as oversight without measuring it. Rejection rate and time per
item, from {{ch:sec-permissions}}.

Treat compliance evidence as a second customer for records you already need. Every artefact in
the table has a non-regulatory justification that arrives years earlier.

Record when the classification was made and on what facts. If the deployment changes, the
classification may too, and the earlier reading is evidence of good faith rather than of
correctness.

## 11. Common Mistakes

**Staffing compliance as a separate workstream.** 62% of it is engineering you already do.

**Planning by obligation.** They map many-to-one onto artefacts.

**Deferring the perishable records.** Five of six critical facts are unrecoverable, not
expensive.

**Accepting a form-satisfying oversight step.** It produces a record, a delay and no review.

**Arguing the classification internally instead of getting it decided.** It is most of the cost
and none of it is settled by engineering.

**Treating the datasheet as documentation.** It is the only record of a fact that exists once.

## 12. Failure Modes

**Classification changes at the buyer's reading.** Shipped as limited risk, procured as
materially informing, twenty-two weeks of delay discovered in a sales cycle.

**Principal chain requested and absent.** 39% covered, 11× to retrofit, and the answer is that
it cannot be.

**Datasheet written from memory.** A document rather than a record, and the licences were never
knowable after ingest.

**Human oversight that satisfies the form.** {{ch:sec-permissions}}'s rubber stamp, and it is
the cheapest tier-reducing move on the list.

**Two systems split below the line.** Same behaviour, different org chart, and the split has its
own {{ch:sec-tool-abuse}} composition risk.

**Evidence assembled at assessment time.** 19% contemporaneous, 4.8× the cost, and the gaps are
the ones that matter.

## 13. Alternatives

**Design deliberately into the lower tier.** Legitimate when the capability sacrificed is real —
{{sec:9-practical-example}}'s aligned rows — and it should be documented as a deliberate risk
reduction rather than discovered as a classification result.

**Accept the high-risk tier and build the programme.** $1,640,000 and 22 weeks, and 62% of it is
work you would do anyway.

**Deploy in a narrower jurisdiction first.** Buys time and produces the contemporaneous records
under a lighter regime, which then transfer.

**Third-party conformity assessment early.** Expensive, and it converts the classification
argument into a documented finding, which is what the first listing showed the money is.

**{{cite:breck2017}}'s rubric as an internal proxy.** Predates the regulations, covers much of
the same ground, and produces the habit before the obligation.

## 14. Evaluation

Score your system against each classification factor and record who made each judgement and on
what basis. That record is evidence regardless of the answer.

Map each obligation to the artefact that satisfies it and compute your coverage. Most teams have
never seen the 62%.

For each missing artefact, ask whether the underlying facts persist. The answer partitions the
gap into a cost and a loss.

Measure your oversight step's rejection rate. If it is near zero, the obligation is satisfied in
form only and you should know that before someone else does.

Re-run the classification whenever the deployment context changes. It is a function of context
and the context moves.

## 15. Advanced Concepts

The step-function model treats tier assignment as a deterministic function of features, and in
practice it is a *distribution*: the same system may be assessed differently by different
authorities, at different times, or after an incident. That converts the design problem from
"stay below the boundary" to "manage the probability of landing above it", and the correct
object is an expected cost $\mathbb{E}[C] = \sum_t \Pr[T = t] C_t$. Under that framing the
boundary-seeking incentive weakens — a design that is *probably* below the line still carries
much of the high-risk expected cost — and the value of an early written classification rises
sharply, because it collapses the distribution.

The evidence-coverage figure has an optimism worth flagging. Coverage here means the artefact
exists and contains the substance, not that it is in the required format or has been through
the required process. Format conversion is cheap and process is not: an evaluation report that
exists and was never independently reviewed satisfies the substance and not the assessment.
**The 62% is a floor on the work remaining and not a measure of readiness**, and the distance
between substance and process is where most of the remaining time goes.

There is an interaction with {{ch:ops-versioning}} that this chapter inherits and does not
resolve. That chapter argued for pinning every artefact; {{ch:rai-privacy}} argued for deleting
records on request; this chapter argues for contemporaneous records that survive years. All
three want durable artefacts and one of them wants selective destruction, and the standard
resolutions all lose something. The least-bad design is to retain *manifests and decisions*
rather than *data*, which satisfies the evidentiary obligation and the deletion obligation
simultaneously and gives up exact reproducibility.

Finally, a limit on the whole framing. This chapter treats regulation as a cost and a set of
evidence requirements, which is the engineering view and is not the only one. The obligations
exist because deployed systems have harmed people, and the artefacts they require — a stated
purpose, a disaggregated evaluation, a traceable decision — are the ones this book has
independently recommended in eight other chapters for entirely practical reasons. **A framework
that asks for the things good practice already produces is not primarily a cost**, and the
$1,640,000 is mostly the price of doing at a deadline what should have been done at design.

## 16. Connection to Previous Chapters

{{eq:reproducibility-is-a-product-over-artefacts}} from {{ch:ops-versioning}} is why the
corpus-version fact is unrecoverable rather than expensive, and why the pinning that chapter
recommended has a second customer.

{{eq:audit-completeness-requires-the-principal-chain}} from {{ch:sec-permissions}} is the
highest-retrofit obligation here at 11×, for exactly the reason that chapter gave: a chain not
propagated cannot be assembled.

{{eq:deletion-is-a-product-over-derived-artefacts}} from {{ch:rai-privacy}} supplies the
data-governance evidence and, per {{sec:15-advanced-concepts}}, conflicts with the retention
obligation in a way neither chapter resolves.

{{eq:coverage-is-a-union-not-a-sum}} from {{ch:ev-framework}} is the planning structure: many
obligations, fewer artefacts, and the artefact is the correct unit.

## 17. Exercises

1. Score your system against the seven classification factors. Which conclusion does each
   plausible reader reach?

2. Price the tier step for your jurisdiction and compute what a boundary-reducing design would
   be worth.

3. Map your eleven obligations to artefacts and compute coverage. Where are your two largest
   gaps?

4. For each gap, determine whether the underlying facts persist. How much of your gap is a loss
   rather than a cost?

5. Model tier assignment as a distribution per {{sec:15-advanced-concepts}} and recompute the
   value of a boundary-reducing design under expected cost.

## 18. Interview Questions

1. What determines which obligations apply to our system?

2. Two readers classify us differently. What is that disagreement worth?

3. We added a human confirmation step to stay in the lower tier. Is that a real risk reduction?

4. How much of a compliance package do we already have?

5. A regulator asks which corpus version trained the deployed model. Can we answer?

6. Why is evidence built later more expensive than evidence built now?

## 19. Research Questions

1. How much do independent assessors agree on tier classification for the same system?

2. What share of deployed systems' compliance evidence is contemporaneous versus retrospective?

3. Do form-satisfying oversight arrangements measurably differ from substantive ones in
   rejection rate, and is that measured anywhere?

4. Can a manifest-only retention design satisfy both evidentiary and deletion obligations in
   practice?

## 20. Chapter Summary

AI regulation shares a structure, and the structure makes cost a step function.

Minimal risk carries **2 obligations and $36,000**; high risk carries **21 and $1,640,000** plus
**22 weeks** ({{eq:compliance-cost-is-a-step-function}}). **The step is $1,473,000 and 19
weeks**, and it turns on judgements — "materially informs", "easily obtain review" — where four
reasonable readers reach three conclusions about one system.

That creates an incentive: designing below the boundary is worth **$717,000**
({{eq:tier-boundaries-create-design-incentives}}), and **three of five** available moves change
a label rather than a risk — including {{ch:sec-permissions}}' rubber stamp, which satisfies the
form exactly. Across system complexities, the **tier accounts for ~98% of the cost variance**,
so the classification matters more than the system.

The second half is better news. Eleven obligations map onto artefacts this book already argued
for, at **62% mean coverage**
({{eq:most-compliance-evidence-is-engineering-you-already-do}}) — 88% for testing evidence, 83%
for incident reporting, 79% for monitoring. The gaps are the training-data record (66% missing)
and the principal chain (61%), both of which had to be written when the thing happened.

Because evidence must be contemporaneous. Building now costs **$88,600**; retrofitting,
**$421,540** — and for **5 of 6** critical facts the answer is not "expensive" but "gone"
({{eq:evidence-must-be-contemporaneous}}). Every one of those facts was already worth recording:
the corpus version for reproducibility, the chain for triage, the snapshot for regression
detection. **Compliance is the second customer for a record you needed anyway.**

That reframing is worth more than it sounds, because it changes who owns the work and when it
happens. A compliance programme staffed separately, starting when a regulator or a buyer asks,
is building artefacts against a deadline from facts that have partly expired. The same artefacts
built by the engineering team, at design time, for reproducibility and triage and regression
detection, are cheaper, more accurate, and already contemporaneous when the question arrives.

The synthesis is unusual for this book because it is mostly reassuring. The expensive,
contested, unwinnable part — the classification — is not an engineering question and should
leave the engineering team early. The part that is an engineering question is 62% done, and
finishing it produces artefacts with a non-regulatory justification that arrives years sooner.
The failure mode is not doing too little; it is doing it too late, on facts that no longer
exist.

Carry forward: **the classification is where the money is**, and **build the perishable records
first**.

## 21. Further Reading

- {{cite:breck2017}} — a production-readiness rubric that predates the regulatory instruments
  and covers much of the same evidence.
- {{cite:mitchell2019modelcards}} — the artefact satisfying two obligations at once, with
  disaggregated evaluation as its operative content.
- {{cite:gebru2021datasheets}} — the dataset record, which is the largest gap here and the one
  whose facts expire at ingest.
- {{cite:paleyes2020deployment}} — deployment obstacles across the lifecycle, several of which
  are the records this chapter finds missing.

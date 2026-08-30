# -*- coding: utf-8 -*-
# Extracted from: Chapter 231 — Regulation and Risk Management
# Source: src/.../ch231-regulation.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

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

# -*- coding: utf-8 -*-
# Extracted from: Chapter 215 — Human Evaluation and Annotation Design
# Source: src/.../ch215-human-evaluation.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Label noise is not noise once the labels become the reference set.

While you are estimating a quantity, annotator error averages out. The moment those labels
are frozen as an evaluation set, every error becomes a permanent bias in every future
comparison run against it -- and it compresses the gap between models exactly the way
ch:ev-llm-benchmarks' contamination did.

Redundant annotation fixes this, and it is not free: k annotators on the same item means
one k-th of the items for the same budget.

This listing computes the aggregate label's reliability (eq:aggregate-reliability-follows-
spearman-brown), converts it into a label error rate, and finds the k that minimises the
total labelling needed to detect a given difference
(eq:budget-splits-between-items-and-annotators).
"""
import math

SINGLE_REL = 0.61          # correlation between two independent labellings
SINGLE_ERR = 0.14          # P(one annotator labels an item wrongly)
GAP = 0.05                 # the model difference we need to detect
TRUE_ACC = 0.72
POWER_Z = 2.80
LABEL_COST = 3.40          # dollars per item per annotator


def spearman_brown(r, k):
    return k * r / (1.0 + (k - 1) * r)


def majority_error(e, k):
    """P(majority of k independent annotators is wrong). k odd."""
    need = k // 2 + 1
    tot = 0.0
    for j in range(need, k + 1):
        tot += (math.comb(k, j) * (e ** j) * ((1 - e) ** (k - j)))
    return tot


print(f"One annotator: reliability {SINGLE_REL:.2f}, error rate {SINGLE_ERR:.0%}.")
print()
print(f"{'annotators':>12}{'reliability':>14}{'metric ceiling':>17}"
      f"{'label error':>14}{'cost/item':>12}")
print("-" * 69)
rel = {}
for k in (1, 3, 5, 7, 9):
    r = spearman_brown(SINGLE_REL, k)
    e = majority_error(SINGLE_ERR, k)
    rel[k] = (r, math.sqrt(r), e)
    print(f"{k:>12}{r:>14.3f}{math.sqrt(r):>17.3f}{e:>14.3%}"
          f"{k * LABEL_COST:>12.2f}")

print()
print("Reliability has diminishing returns; the error rate falls much faster.")

print()
print()
print("Why the error rate is the column that matters: it biases every future")
print("comparison run against the frozen set.")
print()
print(f"{'annotators':>12}{'label error':>14}{'true gap':>11}"
      f"{'observed gap':>15}{'compression':>14}")
print("-" * 66)
comp = {}
for k in (1, 3, 5, 7, 9):
    e = rel[k][2]
    obs = GAP * (1.0 - 2.0 * e)
    comp[k] = (e, obs, 1.0 - 2.0 * e)
    print(f"{k:>12}{e:>14.3%}{GAP:>11.3f}{obs:>15.4f}"
          f"{1.0 - 2.0 * e:>14.3f}")

print()
print("A model scored against a wrong label loses credit when it is right and")
print("gains it when it is wrong, so the gap shrinks by twice the error rate.")

print()
print()
print("Items needed to detect the gap, and the total labels that costs.")
print()


def n_needed(p, d):
    if d <= 0:
        return float("inf")
    return (POWER_Z ** 2) * 2.0 * p * (1 - p) / (d ** 2)


print(f"{'annotators':>12}{'observed gap':>15}{'items needed':>15}"
      f"{'labels needed':>16}{'cost':>12}")
print("-" * 70)
budget = {}
for k in (1, 3, 5, 7, 9):
    obs = comp[k][1]
    n = n_needed(TRUE_ACC, obs)
    lab = k * n
    budget[k] = (obs, n, lab, lab * LABEL_COST)
    print(f"{k:>12}{obs:>15.4f}{n:>15.0f}{lab:>16.0f}"
          f"{lab * LABEL_COST:>12,.0f}")

best_k = min(budget, key=lambda k: budget[k][2])
print()
print(f"cheapest in labels: k={best_k} at {budget[best_k][2]:,.0f}")
print("and that holds for every error rate -- redundancy never reduces the")
print("labelling needed to detect a difference.")

print()
print()
print("The same check at every annotator quality: the optimum never moves.")
print()
print(f"{'single error':>14}{'best k':>9}{'labels at best k':>19}"
      f"{'labels at k=3':>16}{'penalty':>10}")
print("-" * 68)
sens = {}
for e1 in (0.04, 0.08, 0.14, 0.20, 0.28):
    opts = {}
    for k in (1, 3, 5, 7, 9, 11):
        e = majority_error(e1, k)
        opts[k] = k * n_needed(TRUE_ACC, GAP * (1 - 2 * e))
    bk = min(opts, key=lambda k: opts[k])
    sens[e1] = (bk, opts[bk], opts[1])
    print(f"{e1:>14.0%}{bk:>9}{opts[bk]:>19,.0f}{opts[3]:>16,.0f}"
          f"{opts[3] / opts[bk]:>9.2f}x")

print()
print()
print("So when is redundancy worth buying? When items are scarce and")
print(f"annotators are not. Suppose only {1500} suitable items exist.")
print()
ITEMS_AVAILABLE = 1500
print(f"{'annotators':>12}{'items needed':>15}{'available':>12}"
      f"{'feasible?':>12}{'labels':>10}{'cost':>11}")
print("-" * 72)
feas = {}
for k in (1, 3, 5, 7, 9):
    n = budget[k][1]
    ok = n <= ITEMS_AVAILABLE
    feas[k] = ok
    print(f"{k:>12}{n:>15.0f}{ITEMS_AVAILABLE:>12}"
          f"{('yes' if ok else 'no'):>12}{k * n:>10.0f}"
          f"{k * n * LABEL_COST:>11,.0f}")
min_k = min((k for k in feas if feas[k]), default=None)
print()
print(f"smallest workable redundancy: {min_k} annotators, "
      f"{budget[min_k][3]:,.0f}")

print()
print()
print("How scarce the items have to be before redundancy is forced.")
print()
print(f"{'items available':>17}{'minimum k':>12}{'labels':>11}"
      f"{'cost':>11}{'vs k=1 cost':>14}")
print("-" * 65)
scarce = {}
for avail in (4000, 2500, 1800, 1500, 1200, 900):
    mk = None
    for k in (1, 3, 5, 7, 9, 11, 15):
        e = majority_error(SINGLE_ERR, k)
        n = n_needed(TRUE_ACC, GAP * (1 - 2 * e))
        if n <= avail:
            mk = k
            break
    if mk is None:
        scarce[avail] = None
        print(f"{avail:>17}{'none':>12}{'--':>11}{'--':>11}{'--':>14}")
        continue
    e = majority_error(SINGLE_ERR, mk)
    n = n_needed(TRUE_ACC, GAP * (1 - 2 * e))
    scarce[avail] = (mk, mk * n, mk * n * LABEL_COST)
    print(f"{avail:>17}{mk:>12}{mk * n:>11.0f}{mk * n * LABEL_COST:>11,.0f}"
          f"{mk * n * LABEL_COST / budget[1][3]:>13.1f}x")

print()
print()
print("And the case that changes the answer: the same budget spent on a")
print("question where labels are NOT frozen into a reference set.")
print()
print(f"{'annotators':>12}{'items for 30k labels':>23}{'SE of the mean':>17}"
      f"{'usable?':>10}")
print("-" * 62)
BUDGET_LABELS = 30000
ITEM_SD = 0.45
for k in (1, 3, 5, 7, 9):
    n = BUDGET_LABELS / k
    noise_sd = math.sqrt(max(1e-9, (1 - SINGLE_REL) / SINGLE_REL)) * ITEM_SD
    se = math.sqrt(ITEM_SD ** 2 / n + noise_sd ** 2 / (n * k))
    print(f"{k:>12}{n:>23,.0f}{se:>17.5f}"
          f"{('best' if k == 1 else ''):>10}")

print(f"""
The reliability table is the standard picture and its shape is familiar: aggregating
{3} annotators takes reliability from {SINGLE_REL:.2f} to {rel[3][0]:.3f}, and
{9} takes it to {rel[9][0]:.3f} (eq:aggregate-reliability-follows-spearman-brown). The
metric ceiling from ch:ev-why-hard follows as its square root, from
{rel[1][1]:.3f} to {rel[9][1]:.3f}.

Diminishing returns, which is why most guidance stops at three and calls the question
settled.

The error column is the one that changes the decision, and it falls much faster than
reliability rises: {rel[1][2]:.1%} for one annotator, {rel[3][2]:.1%} for three,
{rel[9][2]:.2%} for nine. Majority voting over independent errors is very effective and
Spearman-Brown does not show it, because reliability and error rate are different
functionals of the same noise.

The compression table says why that error rate is the number to optimise. A frozen label
set is a reference, and a model scored against a wrong label **loses credit when it is
right and gains it when it is wrong** -- so the observed difference between two models is
the true one scaled by {1 - 2 * SINGLE_ERR:.2f} at one annotator.

That is ch:ev-llm-benchmarks' contamination arithmetic arriving from a completely different
direction. Contamination made items uninformative by leaking them; label error makes items
*anti*-informative by mislabelling them, and both compress the gap while leaving the score
looking reasonable.

The budget table is where it becomes a decision, and the decision goes the other way from
what I expected. Detecting a {GAP:.2f} gap needs {budget[1][1]:,.0f} items at single
annotation and {budget[3][1]:,.0f} at three -- fewer items, as redundancy promised -- but
the *labels* go from {budget[1][2]:,.0f} to {budget[3][2]:,.0f}.

**Redundant annotation never reduces the labelling needed to detect a difference**
(eq:budget-splits-between-items-and-annotators). The ratio is
$k[(1-2e_1)/(1-2e_k)]^2$, which exceeds one for every $k>1$ at every error rate, because
tripling the labels per item buys less than a threefold reduction in the squared
compression. The sensitivity table confirms it at every error rate from
{0.04:.0%} to {0.28:.0%}: the optimum is {sens[0.14][0]} annotator throughout.

That is worth stating plainly because redundancy is usually justified on quality grounds
with the implication that it also buys statistical strength. It does not. If your only
constraint is budget, spend it on items.

The scarcity table is where redundancy earns its place, and it is a different constraint
entirely. Suppose only {1500} suitable items exist -- a specialist domain, a rare failure
mode, an expensive elicitation. Single annotation needs {budget[1][1]:,.0f} items and cannot
have them. Three annotators need {budget[3][1]:,.0f} and still cannot. Five need
{budget[5][1]:,.0f}, which is feasible.

**Redundancy converts annotator time into effective items**, and that is the trade it is
actually making. The scarcity table prices it: at {1800} available items the minimum
workable redundancy costs {scarce[1800][2] / budget[1][3]:.1f} times the unconstrained
single-annotation budget, and at {1500} it costs
{scarce[1500][2] / budget[1][3]:.1f} times -- with no alternative, because the items do not
exist to be bought instead.

Below {1200} available items the table reports `none`, and that row is the most useful one
in the listing. **There is a floor below which no amount of annotation makes the comparison
possible**, because the compression from label error is bounded: even perfect aggregation
leaves the observed gap at {GAP:.3f}, and {n_needed(TRUE_ACC, GAP):,.0f} items are needed to
see it. Redundancy buys the difference between {budget[1][1]:,.0f} and
{n_needed(TRUE_ACC, GAP):,.0f} and nothing beyond it.

The last table is the caveat that keeps this honest. When labels are *not* frozen -- when
you are estimating a mean and the noise averages out -- single annotation on more items
wins at every k, because independent errors cancel in an average and do not cancel in a
reference.

Which gives a rule with two conditions rather than one. **Redundancy is for item scarcity,
not for budget efficiency**, and it is never the cheaper way to buy statistical strength. And
label error only bites when the labels are frozen and reused -- in a survey the errors
average out, in an evaluation set they become a permanent scale factor on every comparison
that set will ever support.

So: count your available items first, because that is what decides k, and double-label a
pilot to measure {SINGLE_ERR:.0%} before committing the budget, because every number above
is a function of it.""")

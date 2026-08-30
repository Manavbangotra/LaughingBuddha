---
id: ev-human
number: 215
part: XXV
tier: full
status: draft
requires: [agreement-caps-measurable-quality, reference-scoring-penalises-valid-answers,
           a-score-needs-a-human-baseline, contamination-inflates-and-flattens]
provides: [aggregate-reliability-follows-spearman-brown, budget-splits-between-items-and-annotators,
           guideline-defect-is-the-cheapest-disagreement, pilot-cost-is-recovered-by-avoided-relabelling]
citations: [rein2023gpqa, card2020power, wang2023unfair, ribeiro2020checklist]
---

## 1. Learning Objectives

By the end of this chapter you will be able to compute the reliability of an aggregated
label from single-annotator reliability, and the majority-vote error rate from
single-annotator error; explain why label error compresses model gaps rather than adding
noise once labels are frozen; show that redundant annotation never reduces the labelling
needed to detect a difference, and identify the constraint under which it is nonetheless
required; decompose observed annotator disagreement into its four sources and rank the
remedies by payback; and price a double-labelled pilot against the relabelling it prevents.

## 2. Why This Matters

{{ch:ev-why-hard}} established that annotator agreement caps what any metric can measure.
This chapter is about the process that produces those labels, and it starts with a
distinction that changes every subsequent decision.

**While you are estimating a quantity, annotator error averages out. Once the labels are
frozen as an evaluation set, it becomes a permanent bias.** A model scored against a wrong
label loses credit when it is right and gains it when it is wrong, so the observed gap
between two models is the true gap scaled by $(1 - 2e)$ — **0.720** at a 14%
single-annotator error rate. That is
{{eq:contamination-inflates-and-flattens}}'s compression arriving from a different
direction entirely.

Aggregating three annotators takes reliability from 0.61 to **0.824** and the label error
rate from 14% to **5.3%** ({{eq:aggregate-reliability-follows-spearman-brown}}) — the error
rate falls much faster than reliability rises, and it is the one that matters.

But redundancy is not a budget saving. Detecting a 0.05 gap costs **2,439 labels** at
single annotation and **4,753** at three, and single annotation wins at every annotator
quality from 4% to 28% error ({{eq:budget-splits-between-items-and-annotators}}). Redundancy
is for **item scarcity**: with only 1,500 suitable items available, single annotation needs
2,439 and cannot have them; five annotators need 1,384 and can.

The second half is where agreement actually comes from. Guideline underspecification is
**37%** of observed disagreement against annotator skill's **22%**, and it is **6×** cheaper
to fix ({{eq:guideline-defect-is-the-cheapest-disagreement}}). A 60-item double-labelled
pilot costs **3.8%** of the batch and pays whenever a guideline defect is more likely than
**9.1%** ({{eq:pilot-cost-is-recovered-by-avoided-relabelling}}).

## 3. Prerequisites

{{eq:agreement-caps-measurable-quality}} from {{ch:ev-why-hard}} is the result this chapter
operationalises: agreement bounds the metric ceiling, and this chapter is about raising
agreement without training people to guess each other.

{{eq:reference-scoring-penalises-valid-answers}} from the same chapter explains why human
labels are needed at all for open-ended tasks — there is no reference to compare against, so
the judgement has to come from somewhere.

{{eq:a-score-needs-a-human-baseline}} from {{ch:ev-llm-benchmarks}} is the other reason to
run humans through your evaluation: {{cite:rein2023gpqa}}'s expert and non-expert baselines
are annotation runs, and they supply the units every score in that chapter needed.

{{eq:contamination-inflates-and-flattens}} from the same chapter is the arithmetic
{{sec:9-practical-example}} reproduces: a mechanism that makes items less discriminative
compresses gaps rather than adding noise, and it does so invisibly.

{{cite:card2020power}} supplies the sample-size arithmetic every table in the first listing
rests on.

## 4. Intuitive Explanation

Everyone knows annotators disagree. The interesting questions are what that costs, what it
costs to fix, and which fix.

Start with the cost, because it depends on something that is usually left implicit: what the
labels are *for*.

If you are estimating a quantity — the mean satisfaction score, the share of outputs that
are acceptable — annotator error is noise. Some annotators mark a good answer bad, some mark
a bad answer good, the errors point in both directions, and the average is roughly right.
More items help; redundant annotation is a luxury.

If you are building an evaluation set that will be frozen and reused, the same errors behave
completely differently. Those labels are now the reference. When a model gets an item right
and the label says it is wrong, the model loses a point it earned. When the model gets it
wrong and the label agrees with the mistake, the model gains a point it did not.

So the model's measured accuracy is pulled toward the middle, and — crucially — so is every
*difference* between models. At a 14% label error rate, a true 5-point gap between two
models is observed as 3.6 points. The compression factor is $(1 - 2e)$.

That is exactly {{ch:ev-llm-benchmarks}}'s contamination arithmetic, from a completely
different mechanism. Contamination made items uninformative by leaking them; label error
makes them *anti*-informative by mislabelling them. Both compress the gap. Both leave the
absolute scores looking plausible.

Redundant annotation is the standard remedy, and it works well on the number that matters.
Three annotators voting takes the error rate from 14% to 5.3%; nine takes it to 0.41%.
Reliability, the number people usually quote, moves much less dramatically — 0.61 to 0.82 to
0.93 — which is why the standard advice stops at three and treats the question as settled.
Reliability and error rate are different functionals of the same noise, and majority voting
is far better at one than the other.

Now the budget question, and the answer is not the one I expected when I worked it out.

Detecting a fixed difference needs a number of items that scales as one over the compressed
gap squared. Redundancy shrinks the compression, so it reduces the items needed: 2,439 at
single annotation, 1,584 at three. But three annotators means three labels per item, so the
*labels* go from 2,439 to 4,753.

**Redundant annotation never reduces the labelling needed to detect a difference.** Not at
14% error, not at 28%, not at 4%. The ratio is $k[(1-2e_1)/(1-2e_k)]^2$ and it exceeds one
for every $k > 1$, because tripling the labels per item buys less than a threefold reduction
in the squared compression.

That is worth being blunt about, because redundancy is usually sold with an implication that
it also buys statistical strength. It does not. If your only constraint is money, spend it on
items.

So when is redundancy required? When items are scarce and annotators are not.

Suppose your task is a rare failure mode, or a specialist domain, or requires an expensive
elicitation, and only 1,500 suitable items exist. Single annotation needs 2,439 and you
cannot have them. Three annotators need 1,584 — still too many. Five need 1,384, which is
feasible. **Redundancy converts annotator time into effective items**, and that is the trade
it is actually making.

There is a floor, and it is the useful row in the table. Even perfect labels leave the gap
at 0.05, which needs 1,264 items to detect. Redundancy buys the difference between 2,439 and
1,264 and nothing beyond it. Below about 1,200 available items, no amount of annotation
makes the comparison possible.

That is the first half. The second half is about where disagreement actually comes from, and
it is where most annotation budgets are misspent.

A team measures agreement, finds it low, and concludes the annotators are careless. Hire
better ones. Train them harder. Add a qualification test.

Some of the disagreement is that. Most of it is not.

Disagreement has four sources. **Genuine item ambiguity** — items where careful, expert,
well-briefed people simply differ, because the item is genuinely borderline.
**Guideline underspecification** — the instructions do not say what to do in a case that
occurs often, so each annotator invents a consistent policy and the policies differ.
**Annotator skill variance** — some people are better at this than others.
**Presentation effects** — the same item rated differently depending on where it appeared in
the batch and what preceded it.

In the decomposition used here, the guideline is 37% and annotator skill is 22%. **The
instructions produce more disagreement than the people do**, and fixing instructions is six
times cheaper than fixing people.

Presentation effects are 14% and they are *free* to remove. An item shown first in a batch is
rated 0.21 lower. An item following a very bad one is rated 0.17 higher, by contrast. A
candidate shown on the left is rated 0.11 higher than the same candidate shown on the right —
which is {{cite:wang2023unfair}}'s position bias, established for model judges, present in
humans for the same reason: the first thing read becomes the reference for the second.

Every fix for those is randomisation or a session-length cap. None of it costs anything, and
all of it currently shows up in the agreement statistic as annotator noise — which is how a
free correction turns into a hiring requisition.

The ambiguity floor deserves a note of its own. Twenty-seven percent of the disagreement here
is irreducible: items where careful people genuinely differ. A team pushing agreement past
that floor is not improving the process. It is training annotators to guess each other rather
than to judge, and the resulting agreement is real, measurable, and worthless.

Which raises the obvious question: how do you know which of your disagreement is which?

You double-label a pilot. Sixty items, two annotators, before the main batch. It costs 3.8%
of the batch, and it tells you three things nothing else produces: the single-annotator
error rate that every calculation in the first half depends on, the guideline cases nobody
had thought of, and the irreducible floor.

Skip it, hit a guideline defect, and 55% of the batch has to be redone. The pilot pays
whenever a guideline defect is more likely than 9.1% — and if you have never double-labelled
anything, you have no evidence about whether your guideline clears that bar in either
direction.

## 5. Formal Explanation

Let $r_1$ be the reliability of a single annotation — the correlation between two
independent labellings of the same items. The Spearman–Brown prophecy formula gives the
reliability of an aggregate of $k$:

$$r_k = \frac{k r_1}{1 + (k-1) r_1}.$$

Reliability is a variance ratio, so it saturates. The label *error rate* behaves differently.
If each annotator errs independently with probability $e_1$, the majority vote over odd $k$
errs with probability $\sum_{j > k/2} \binom{k}{j} e_1^j (1-e_1)^{k-j}$, which falls
exponentially in $k$ for $e_1 < 1/2$.

**Frozen labels.** A model with true accuracy $q$ scored against labels with error rate $e$
is measured at $q(1-e) + (1-q)e$. For two models, the observed difference is
$(q_2 - q_1)(1 - 2e)$: the error acts as a multiplicative attenuation on every comparison,
not as additive noise.

**The budget.** Detecting a difference $d$ needs $n \approx z^2 \cdot 2\bar p(1-\bar p) /
[d(1-2e_k)]^2$ items and therefore $k \cdot n$ labels. The ratio of labels at $k$ to labels
at $1$ is $k\left[\frac{1-2e_1}{1-2e_k}\right]^2$, which is minimised at $k=1$ for all
$e_1 \in (0, 1/2)$. Under an item-supply constraint $n \le N_{\text{avail}}$, however, the
feasible set may exclude $k=1$, and the binding choice is the smallest $k$ satisfying it.

**Disagreement decomposition.** Observed disagreement $D$ partitions as $D = D_{\text{amb}} +
D_{\text{guide}} + D_{\text{skill}} + D_{\text{pres}}$, with removable fractions
$\rho_s$ and efforts $c_s$. The build order is by $\rho_s D_s / c_s$, and the achievable
floor is $D_{\text{amb}}$, which no intervention reduces.

**Disagreement to error.** For two independent annotators each erring at $e$, the observed
disagreement is $2e(1-e)$; inverting gives $e = \tfrac12\left(1 - \sqrt{1 - 2D}\right)$,
which is how a measured agreement statistic becomes the $e$ the first half needs.

## 6. Mathematical Foundation

Aggregate reliability and aggregate error, which behave differently:

$$r_k = \frac{k r_1}{1 + (k-1)r_1}, \qquad e_k = \sum_{j > k/2} \binom{k}{j} e_1^{\,j}(1-e_1)^{k-j}$$ (eq:aggregate-reliability-follows-spearman-brown)

At $r_1 = 0.61$, $e_1 = 0.14$: $r_3 = 0.824$ and $e_3 = 5.3\%$; $r_9 = 0.934$ and
$e_9 = 0.41\%$.

The budget, and the constraint that actually decides $k$:

$$\frac{L(k)}{L(1)} = k\left[\frac{1-2e_1}{1-2e_k}\right]^2 > 1 \ \ \forall k>1, \qquad k^\star = \min\{k : n(k) \le N_{\text{avail}}\}$$ (eq:budget-splits-between-items-and-annotators)

Labels needed: **2,439** at $k=1$, **4,753** at $k=3$. Under $N_{\text{avail}} = 1500$,
$k^\star = 5$.

Ranking the sources of disagreement by payback:

$$\text{payback}_s = \frac{\rho_s D_s}{c_s}, \qquad D_{\min} = D_{\text{amb}}$$ (eq:guideline-defect-is-the-cheapest-disagreement)

giving 0.106 for presentation, **0.075 for the guideline**, and 0.005 for annotator skill,
against an irreducible floor of 0.065.

And the pilot's break-even:

$$\Pr[\text{defect}]^\star = \frac{C_{\text{pilot}}}{C_{\text{relabel}}\left(p_d - (1-p_d)\phi\right)} = 9.1\%$$ (eq:pilot-cost-is-recovered-by-avoided-relabelling)

with $C_{\text{pilot}}$ at 3.8% of the batch and $C_{\text{relabel}}$ at 55% of it.

## 7. Internal Mechanics

Why is the guideline the largest component and the last thing anybody fixes?

Because guideline defects are invisible from inside the process. Each annotator encounters an
unspecified case, makes a reasonable decision, and applies it consistently thereafter. From
their side nothing is wrong — they are being careful and consistent. The defect shows up only
in the *comparison* between annotators, which nobody performs unless items are deliberately
double-labelled. A single-annotated batch cannot reveal a guideline defect at all; it can
only carry one.

That is the structural argument for the pilot, and it is stronger than the cost argument.
The pilot is not primarily a quality check. **It is the only instrument in the process
capable of detecting the largest source of error.**

Presentation effects have a different invisibility. They are real, well-documented in
psychology, and produce shifts that are small per item and systematic across a batch. Because
they are systematic they do not average out — an annotator who works through a batch in a
fixed order applies a fatigue gradient to the whole batch, and if two annotators work in the
same order the gradient is *correlated*, which means it does not even show up as
disagreement. It shows up as a bias that both annotators share and neither notices.

Randomising order per annotator converts that shared bias into disagreement, which sounds
like a regression and is an improvement: a bias you can see is one you can correct. This is
worth flagging because a team that randomises order will see agreement *fall*, and the
correct response is to be pleased.

The label-error compression has a mechanical consequence for how evaluation sets age. Because
the compression is multiplicative and constant, it does not decay — an evaluation set with 14%
label error compresses every comparison by 0.72 forever, including comparisons between models
that did not exist when it was built. And because the set's headline numbers look reasonable,
nothing prompts a re-examination. It is the same silent-instrument pattern
{{ch:ops-prompt-versioning}} found for coverage, one level further down: the labels are wrong
at a fixed rate, the rate is never measured, and every result is scaled by it.

Finally, the item-scarcity result explains a pattern that otherwise looks like
inconsistency in the literature. Benchmarks in specialist domains — {{cite:rein2023gpqa}}'s
448 questions, for instance — use heavy redundancy and small item counts, while
general-purpose benchmarks use single annotation and tens of thousands of items. Both are
correct, and they are correct for the same reason under different supply constraints. The
mistake is transferring the practice rather than the reasoning.

## 8. Implementation

The first listing computes what label error costs and what redundancy buys.

```python {tier=A name=budget-splits-between-items-and-annotators}
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
```

## 9. Practical Example

One annotator at reliability 0.61 and error rate 14%:

```
  annotators   reliability   metric ceiling   label error   cost/item
---------------------------------------------------------------------
           1         0.610            0.781       14.000%        3.40
           3         0.824            0.908        5.331%       10.20
           5         0.887            0.942        2.200%       17.00
           9         0.934            0.966        0.414%       30.60
```

Reliability saturates; **the error rate falls much faster**
({{eq:aggregate-reliability-follows-spearman-brown}}), and it is the error rate that biases
every future comparison.

```
  annotators   label error   true gap   observed gap   compression
------------------------------------------------------------------
           1       14.000%      0.050         0.0360         0.720
           3        5.331%      0.050         0.0447         0.893
           5        2.200%      0.050         0.0478         0.956
           9        0.414%      0.050         0.0496         0.992
```

A model scored against a wrong label loses credit when it is right and gains it when it is
wrong: **the gap shrinks by twice the error rate.**

```
  annotators   observed gap   items needed   labels needed        cost
----------------------------------------------------------------------
           1         0.0360           2439            2439       8,293
           3         0.0447           1584            4753      16,160
           5         0.0478           1384            6918      23,520
           9         0.0496           1286           11571      39,340
```

```
  single error   best k   labels at best k   labels at k=3   penalty
--------------------------------------------------------------------
            4%        1              1,494           3,865     2.59x
           14%        1              2,439           4,753     1.95x
           28%        1              6,531           9,951     1.52x
```

**Redundancy never reduces the labelling needed to detect a difference**
({{eq:budget-splits-between-items-and-annotators}}) — not at any annotator quality. If your
only constraint is budget, spend it on items.

```
  annotators   items needed   available   feasible?    labels       cost
------------------------------------------------------------------------
           1           2439        1500          no      2439      8,293
           3           1584        1500          no      4753      16,160
           5           1384        1500         yes      6918      23,520
```

```
  items available   minimum k     labels       cost   vs k=1 cost
-----------------------------------------------------------------
             2500           1       2439      8,293          1.0x
             1800           3       4753     16,160          1.9x
             1500           5       6918     23,520          2.8x
             1200        none         --         --            --
```

**Redundancy converts annotator time into effective items** — that is the trade it makes.
And below 1,200 available items there is no $k$ that works: even perfect labels leave the
gap at 0.050, needing 1,264 items.

The second listing asks where the disagreement comes from.

```python {tier=A name=guideline-defect-is-the-cheapest-disagreement}
"""Annotator disagreement is four different problems and only one of them is the annotator.

Teams read a low agreement number as a staffing problem: the annotators are careless, hire
better ones or train them harder. Some of it is that. Most of it is not.

Disagreement decomposes into genuine item ambiguity, guideline underspecification,
annotator skill variance, and presentation effects -- and they have wildly different
prices. The guideline component is the cheapest to remove and usually the largest
(eq:guideline-defect-is-the-cheapest-disagreement).

Finding out which is which costs a double-labelled pilot, and this listing prices the pilot
against the relabelling it prevents
(eq:pilot-cost-is-recovered-by-avoided-relabelling).
"""
# (source, share of observed disagreement, effort to remove, share removable)
SOURCES = [
    ("guideline underspecification", 0.37, 1.0,  0.85),
    ("presentation and order effects", 0.14, 0.3, 0.95),
    ("annotator skill variance",      0.22, 6.0,  0.60),
    ("genuine item ambiguity",        0.27, 0.0,  0.00),
]
OBSERVED_DISAGREE = 0.24
ITEMS = 3200
LABEL_COST = 3.40
ANNOTATORS = 2

print(f"Observed disagreement between two annotators: {OBSERVED_DISAGREE:.0%}.")
print("Where it comes from, and what removing each part would cost.")
print()
print(f"{'source':>32}{'share':>9}{'of the 24%':>13}{'removable':>12}"
      f"{'effort':>9}{'per effort':>13}")
print("-" * 88)
src = {}
for name, share, eff, rem in SOURCES:
    amount = OBSERVED_DISAGREE * share
    gain = amount * rem
    ratio = gain / eff if eff > 0 else 0.0
    src[name] = (share, amount, gain, eff, ratio)
    print(f"{name:>32}{share:>9.0%}{amount:>13.3f}{rem:>12.0%}"
          f"{eff:>9.1f}{ratio:>13.4f}")

irreducible = OBSERVED_DISAGREE * dict((n, s) for n, s, e, r in SOURCES)["genuine item ambiguity"]
print()
print(f"irreducible floor: {irreducible:.3f} -- items where careful people")
print("genuinely disagree, and no guideline can fix that")

print()
print()
print("Removing them in payback order.")
print()
order = sorted([s for s in SOURCES if s[2] > 0], key=lambda s: -src[s[0]][4])
print(f"{'after removing':>32}{'disagreement':>15}{'effort so far':>16}"
      f"{'vs floor':>11}")
print("-" * 74)
cur = OBSERVED_DISAGREE
eff = 0.0
path = []
for name, share, e, rem in order:
    cur -= src[name][2]
    eff += e
    path.append((name, cur, eff))
    print(f"{name:>32}{cur:>15.3f}{eff:>16.1f}"
          f"{cur / irreducible:>11.2f}x")

print()
print()
print("What that does to the label error rate and to everything downstream.")
print()


def err_from_disagreement(d):
    """If two independent annotators disagree at rate d, each errs at e where
    d = 2e(1-e); invert."""
    return (1.0 - (1.0 - 2.0 * d) ** 0.5) / 2.0


print(f"{'state':>32}{'disagreement':>15}{'implied error':>16}"
      f"{'gap compression':>18}")
print("-" * 81)
states = [("as measured", OBSERVED_DISAGREE)]
states += [(n, c) for n, c, e in path]
comp = {}
for label, d in states:
    e = err_from_disagreement(d)
    comp[label] = (e, 1 - 2 * e)
    print(f"{label:>32}{d:>15.3f}{e:>16.3f}{1 - 2 * e:>18.3f}")

print()
print()
print("Presentation effects deserve their own look, because they are free")
print("to remove and are usually counted as annotator noise.")
print()
print(f"{'effect':>34}{'shifts rating by':>19}{'fix':>26}{'cost':>8}")
print("-" * 87)
PRESENT = [
    ("first item in a batch",       -0.21, "discard or randomise",   "free"),
    ("item after a very bad one",   +0.17, "randomise order",        "free"),
    ("item after a very good one",  -0.14, "randomise order",        "free"),
    ("last hour of a session",      -0.19, "cap session length",     "low"),
    ("candidate shown on the left", +0.11, "balance positions",      "free"),
]
for name, shift, fix, cost in PRESENT:
    print(f"{name:>34}{shift:>+19.2f}{fix:>26}{cost:>8}")

pos = abs([p for p in PRESENT if "left" in p[0]][0][1])
print()
print(f"the position effect alone is {pos:.2f} of a rating point, which is")
print("cite:wang2023unfair's finding for model judges, in humans")

print()
print()
print("The pilot: 60 double-labelled items before the main batch.")
print()
PILOT_ITEMS = 60
P_DETECT = 0.82               # P(a guideline defect shows up in 60 double-labelled items)
RELABEL_SHARE = 0.55          # share of the batch that must be redone if it is missed
pilot_cost = PILOT_ITEMS * ANNOTATORS * LABEL_COST
main_cost = ITEMS * LABEL_COST
relabel_cost = ITEMS * RELABEL_SHARE * LABEL_COST
print(f"{'scenario':>34}{'labelling':>13}{'relabelling':>14}"
      f"{'total':>11}{'vs pilot':>11}")
print("-" * 83)
no_pilot = main_cost + P_DETECT * relabel_cost
with_pilot = pilot_cost + main_cost + (1 - P_DETECT) * relabel_cost * 0.4
print(f"{'no pilot, defect present':>34}{main_cost:>13,.0f}"
      f"{P_DETECT * relabel_cost:>14,.0f}{no_pilot:>11,.0f}"
      f"{no_pilot / with_pilot:>10.2f}x")
print(f"{'pilot, defect found and fixed':>34}"
      f"{pilot_cost + main_cost:>13,.0f}"
      f"{(1 - P_DETECT) * relabel_cost * 0.4:>14,.0f}"
      f"{with_pilot:>11,.0f}{1.0:>10.2f}x")
print(f"{'pilot, no defect present':>34}"
      f"{pilot_cost + main_cost:>13,.0f}{0.0:>14,.0f}"
      f"{pilot_cost + main_cost:>11,.0f}"
      f"{(pilot_cost + main_cost) / main_cost:>10.2f}x")

print()
print(f"pilot cost: {pilot_cost:,.0f} ({pilot_cost / main_cost:.1%} of the batch)")

print()
print()
print("Break-even: how likely a guideline defect has to be for the pilot to pay.")
print()
print(f"{'P(defect present)':>19}{'expected cost, no pilot':>26}"
      f"{'expected cost, pilot':>23}{'better':>9}")
print("-" * 77)
for pd in (0.05, 0.15, 0.30, 0.50, 0.75):
    a = main_cost + pd * P_DETECT * relabel_cost
    b = pilot_cost + main_cost + pd * (1 - P_DETECT) * relabel_cost * 0.4
    print(f"{pd:>19.0%}{a:>26,.0f}{b:>23,.0f}"
          f"{('pilot' if b < a else 'no pilot'):>9}")
breakeven = pilot_cost / (relabel_cost * (P_DETECT - (1 - P_DETECT) * 0.4))
print()
print(f"break-even: the pilot pays whenever a guideline defect is more likely")
print(f"than {breakeven:.1%}")

print(f"""
The decomposition is the first thing to look at and the shares are the point.
`{SOURCES[0][0]}` is {SOURCES[0][1]:.0%} of the observed disagreement and
`{SOURCES[2][0]}` is {SOURCES[2][1]:.0%}, so **the guideline contributes more disagreement
than the annotators do** (eq:guideline-defect-is-the-cheapest-disagreement) -- and it is
{src[SOURCES[2][0]][3] / src[SOURCES[0][0]][3]:.0f} times cheaper to fix.

The payback column makes the ordering unambiguous:
{src[SOURCES[1][0]][4]:.3f} for presentation effects, {src[SOURCES[0][0]][4]:.3f} for the
guideline, {src[SOURCES[2][0]][4]:.3f} for annotator quality. Hiring or retraining is last
by a wide margin and is where the conversation usually starts.

The floor matters too. `{SOURCES[3][0]}` is {SOURCES[3][1]:.0%} of disagreement and
**none of it is removable** -- {irreducible:.3f} of disagreement is items where careful
people genuinely differ. A team chasing agreement past that point is not improving the
process, it is training annotators to guess each other rather than to judge, and the
resulting agreement is real and worthless.

The build-order table takes disagreement from {OBSERVED_DISAGREE:.3f} to
{path[-1][1]:.3f} -- {path[-1][1] / irreducible:.2f} times the floor -- for
{path[-1][2]:.1f} units of effort, of which {order[0][2] + order[1][2]:.1f} buys most of
the movement.

The compression table connects this to ch:ev-human's first listing. At the measured
disagreement the implied per-annotator error is {comp['as measured'][0]:.3f} and every model
comparison run against these labels is scaled by {comp['as measured'][1]:.3f}. After the two
cheap fixes it is {comp[path[1][0]][1]:.3f}.

**A guideline revision is a statistical-power intervention**, which is not how it gets
budgeted. It arrives on the roadmap as documentation.

The presentation table is the free money. A candidate shown on the left is rated
{pos:+.2f} higher than the same candidate shown on the right, which is
cite:wang2023unfair's position bias -- established for model judges, and present in humans
for the same reason: the first thing read becomes the reference for the second.

Every fix in that column is randomisation or a session cap. **None of it costs anything and
all of it is routinely counted as annotator noise**, which is how a free correction becomes
a hiring requisition.

The pilot table prices the discipline that finds all of this. Sixty double-labelled items
cost {pilot_cost:,.0f}, which is {pilot_cost / main_cost:.1%} of the batch. Skipping it and
hitting a guideline defect costs {no_pilot:,.0f} against {with_pilot:,.0f}
(eq:pilot-cost-is-recovered-by-avoided-relabelling), because
{RELABEL_SHARE:.0%} of the batch has to be redone.

The break-even table is the number to argue with. The pilot pays whenever a guideline defect
is more likely than **{breakeven:.1%}** -- and if you have never double-labelled anything,
you have no evidence about whether your guideline clears that bar in either direction. **The pilot is cheap enough that the
prior does not have to be high**, and its second output is the single-annotator error rate,
which ch:ev-human's first listing needs and which nothing else in the process produces.""")
```

```
                          source    share   of the 24%   removable   effort   per effort
----------------------------------------------------------------------------------------
    guideline underspecification      37%        0.089         85%      1.0       0.0755
  presentation and order effects      14%        0.034         95%      0.3       0.1064
        annotator skill variance      22%        0.053         60%      6.0       0.0053
          genuine item ambiguity      27%        0.065          0%      0.0       0.0000
```

**The guideline produces more disagreement than the annotators do** and is **6×** cheaper to
fix ({{eq:guideline-defect-is-the-cheapest-disagreement}}). Hiring is last by a wide margin
and is where the conversation usually starts. The 27% ambiguity row is an **irreducible
floor**: pushing past it trains annotators to guess each other.

```
                  after removing   disagreement   effort so far   vs floor
--------------------------------------------------------------------------
  presentation and order effects          0.208             0.3       3.21x
    guideline underspecification          0.133             1.3       2.05x
        annotator skill variance          0.101             7.3       1.56x

                           state   disagreement   implied error   gap compression
---------------------------------------------------------------------------------
                     as measured          0.240           0.139             0.721
    guideline underspecification          0.133           0.071             0.857
```

The two cheap fixes take compression from **0.721 to 0.857** for 1.3 units of effort.
**A guideline revision is a statistical-power intervention**, and it arrives on the roadmap
as documentation.

```
                            effect   shifts rating by                       fix    cost
---------------------------------------------------------------------------------------
             first item in a batch              -0.21      discard or randomise    free
         item after a very bad one              +0.17           randomise order    free
            last hour of a session              -0.19        cap session length     low
       candidate shown on the left              +0.11         balance positions    free
```

The left-position effect is {{cite:wang2023unfair}}'s judge bias in humans, for the same
reason. **Every fix here is free and all of it currently reads as annotator noise.**

```
                          scenario    labelling   relabelling      total   vs pilot
-----------------------------------------------------------------------------------
          no pilot, defect present       10,880         4,907     15,787      1.35x
     pilot, defect found and fixed       11,288           431     11,719      1.00x
          pilot, no defect present       11,288             0     11,288      1.04x
```

A 60-item pilot costs **3.8%** of the batch and pays whenever a guideline defect is more
likely than **9.1%** ({{eq:pilot-cost-is-recovered-by-avoided-relabelling}}) — and it is the
only instrument in the process capable of detecting one.

## 10. Production Considerations

Double-label a pilot before every labelling programme. It costs a few percent, it produces
the single-annotator error rate every downstream calculation needs, and it is the only way a
guideline defect can surface.

Count your available items before choosing redundancy. Item supply, not budget, is what
decides $k$.

Randomise item order per annotator and balance positions in pairwise tasks. Free, and it
converts a shared invisible bias into visible disagreement.

Cap session length. The last-hour effect is as large as the position effect and it is not
usually measured.

Rewrite the guideline before hiring anyone. It is the largest removable component and the
cheapest.

Report the irreducible ambiguity floor alongside your agreement target, and stop optimising
when you reach it.

Publish the label error rate with the evaluation set, and the implied compression factor.
Every comparison run against that set is scaled by it, permanently.

## 11. Common Mistakes

**Treating label error as noise.** In a frozen set it is a multiplicative bias on every
comparison, forever.

**Buying redundancy for statistical strength.** It never reduces the labelling needed; it
buys effective items when items are scarce.

**Reading low agreement as an annotator problem.** The guideline is the larger component and
is six times cheaper.

**Optimising agreement past the ambiguity floor.** That trains annotators to predict each
other, and the resulting agreement measures nothing.

**Counting presentation effects as annotator noise.** They are free to remove and they turn
into a hiring requisition when misattributed.

**Skipping the pilot because the guideline seems clear.** It seems clear to the person who
wrote it, which is the one perspective that cannot detect underspecification.

## 12. Failure Modes

**Frozen set with unmeasured label error.** Every comparison for the next three years is
scaled by a factor nobody has computed, and the scores look reasonable throughout.

**Agreement improved by convergence.** Annotators discuss borderline cases, agreement rises,
and what improved was their model of each other.

**Correlated fatigue gradient.** Both annotators work the batch in the same order, so the
fatigue bias is shared, does not appear as disagreement, and biases the whole set.

**Redundancy bought instead of items.** Three annotators on a third of the items, at nearly
twice the labelling cost, for a comparison that is now underpowered.

**Guideline revised mid-batch.** Items labelled before and after follow different policies,
and the set contains two populations with no field recording which.

**Expert baseline collected once.** {{cite:rein2023gpqa}}-style baselines are treated as
permanent, but the item set drifts and the baseline does not.

## 13. Alternatives

**Adjudication instead of majority vote.** Disagreements go to a senior annotator. Higher
quality per item, much slower, and it hides the disagreement rate you needed to measure.

**Graded rather than binary labels.** Rate on a scale and threshold later. Preserves
information about ambiguity and makes agreement statistics harder to interpret.

**Pairwise preference collection.** Ask which of two outputs is better rather than how good
one is. Much higher agreement, no absolute level, and the position bias in
{{sec:9-practical-example}} becomes the dominant design concern.

**Model-assisted pre-labelling.** A model proposes, a human confirms. Cheap and it imports
the model's biases into the reference set the model will be evaluated against.

**Behavioural test suites.** {{cite:ribeiro2020checklist}}'s approach substitutes
capability-specific constructed tests for sampled-and-labelled ones, sidestepping annotation
for the cases where the expected behaviour can be stated.

## 14. Evaluation

Measure single-annotator error from a double-labelled sample and publish it with every
evaluation set you release.

Compute the implied compression factor and apply it when interpreting any model comparison
run on that set.

Decompose your disagreement into the four sources by re-labelling a sample under a revised
guideline, with randomised order, using both a strong and a median annotator.

Track agreement before and after order randomisation. A fall is the expected and correct
outcome.

Estimate the ambiguity floor by having your two best annotators independently label fifty
items with the guideline in front of them. Whatever remains is the floor.

## 15. Advanced Concepts

The independence assumption behind majority voting is the weakest part of the first
listing, and it fails in a specific direction. Annotators share a guideline, a training
session, and often a cultural background, so their errors are correlated — and correlated
errors do not cancel under majority voting. With correlation $\rho$ between annotator
errors, the effective $k$ is roughly $k/(1 + (k-1)\rho)$, which saturates quickly: at
$\rho = 0.3$, nine annotators behave like about three. That makes the case for redundancy
*weaker* than the listing suggests and the case for guideline work *stronger*, because the
guideline is precisely the shared cause of the correlation. The two halves of this chapter
therefore reinforce each other more than the tables show.

The compression result also interacts with {{ch:ev-why-hard}}'s ceiling in a way worth
untangling, because the two are easily confused. The ceiling $\sqrt{\rho}$ bounds how well a
metric can *correlate* with true quality when validated against noisy labels. The compression
$(1-2e)$ scales the *difference* between two systems measured against those labels. They come
from the same noise and they constrain different quantities: one limits validation, the other
limits comparison. A team can be at the ceiling on validation and still have plenty of
comparison power, or the reverse, and knowing which is binding decides whether to buy
redundancy or items.

There is a subtler problem with the ambiguity floor that the decomposition hides. "Genuine
ambiguity" is not a fixed property of the items — it is a property of the items *given the
guideline*. Almost every genuinely ambiguous case could be resolved by a guideline that took
a position on it, and the reason it is left ambiguous is usually that the position is
contested rather than unknown. So the floor is partly a record of decisions the organisation
has declined to make, and pushing it down means making them. **The annotation process is
where product policy gets discovered**, and treating the floor as a natural constant hides
that.

Finally, model-assisted pre-labelling deserves a warning proportional to how attractive it
is. Having a model propose labels for humans to confirm roughly halves annotation cost and
raises agreement substantially, because annotators anchor on the proposal. Both effects are
real. The problem is that the resulting reference set encodes the proposing model's decision
boundary, and the primary use of the set is to evaluate models — including that one and its
successors. The bias is in the direction of the thing being measured, which is the worst
possible direction, and it is invisible in every agreement statistic because the annotators
agree *more*.

## 16. Connection to Previous Chapters

{{eq:agreement-caps-measurable-quality}} from {{ch:ev-why-hard}} bounded metric validation;
this chapter supplies the process that raises agreement, and
{{sec:15-advanced-concepts}} separates that ceiling from this chapter's compression factor.

{{eq:contamination-inflates-and-flattens}} from {{ch:ev-llm-benchmarks}} is the same
arithmetic from a different mechanism: contamination makes items uninformative, label error
makes them anti-informative, and both compress every gap.

{{eq:a-score-needs-a-human-baseline}} from the same chapter requires exactly the annotation
programme described here — {{cite:rein2023gpqa}}'s baselines are annotation runs, and their
quality is governed by everything in {{sec:9-practical-example}}.

{{eq:reference-scoring-penalises-valid-answers}} from {{ch:ev-why-hard}} is why human labels
are needed at all: there is no reference for an open-ended task, so the judgement has to be
elicited.

## 17. Exercises

1. Double-label fifty items from your current evaluation set. What is the single-annotator
   error rate, and what compression factor does it imply for every comparison you have run?

2. Compute the labels needed at $k = 1, 3, 5$ for a difference you care about. How many
   suitable items do you actually have?

3. Decompose your observed disagreement into the four sources. Which is largest, and where
   does your budget currently go?

4. Randomise item order for one batch and measure the change in agreement. Explain the sign.

5. Model correlated annotator errors at $\rho = 0.3$ and recompute the effective redundancy
   at $k = 9$. How much of {{sec:9-practical-example}}'s benefit survives?

## 18. Interview Questions

1. Our annotators agree 76% of the time. Is that a problem, and whose?

2. Why does label error compress model gaps rather than adding noise?

3. We have budget for 6,000 labels. Three annotators on 2,000 items, or one on 6,000?

4. When is redundant annotation actually required?

5. Our agreement went down after we randomised presentation order. What happened?

6. How would you decide whether to keep improving agreement?

## 19. Research Questions

1. What is the empirical correlation between annotator errors under a shared guideline, and
   how much does it reduce the effective redundancy?

2. How much of the "genuine ambiguity" floor is removable by guideline decisions the
   organisation has declined to make?

3. How large is the anchoring bias from model-assisted pre-labelling, and can it be measured
   without an unassisted control arm?

4. Do presentation effects in annotation transfer quantitatively to the position biases
   measured for model judges?

## 20. Chapter Summary

Human evaluation is a measurement process, and its errors behave differently depending on
what the labels are for.

While estimating a quantity, annotator error averages out. **Frozen into an evaluation set it
becomes a permanent multiplicative bias**: a true 0.05 gap is observed as **0.0360** at a 14%
label error rate, compressed by $(1-2e) = 0.720$ — the same arithmetic as
{{ch:ev-llm-benchmarks}}'s contamination, from an unrelated mechanism.

Aggregating labels helps the error rate much more than it helps reliability: three annotators
take 0.61 to **0.824** reliability and 14% to **5.3%** error
({{eq:aggregate-reliability-follows-spearman-brown}}). But **redundancy never reduces the
labelling needed to detect a difference** — 2,439 labels at $k=1$ against 4,753 at $k=3$, at
every annotator quality ({{eq:budget-splits-between-items-and-annotators}}). It is for item
scarcity: at 1,500 available items, $k=5$ is the smallest that works, and below 1,200 nothing
does.

Disagreement is four problems. Guideline underspecification is **37%** and annotator skill is
**22%**, with the guideline **6×** cheaper to fix
({{eq:guideline-defect-is-the-cheapest-disagreement}}); presentation effects are **14%** and
free to remove; and **27%** is an irreducible floor that should be reported rather than
optimised against. A 60-item double-labelled pilot costs **3.8%** of the batch and pays above
a **9.1%** chance of a guideline defect
({{eq:pilot-cost-is-recovered-by-avoided-relabelling}}).

What ties the chapter together is that the expensive remedy is almost never the right one.
Hiring better annotators is the intervention with the worst payback in the table and the one
teams reach for, because low agreement reads as a people problem. It is mostly a documentation
problem and partly a randomisation problem, and both are discoverable only by an experiment
nobody runs — which is why the pilot, at four percent of the batch, is the single highest-value
line item in an annotation budget.

Carry forward: **frozen label error is a bias, not noise**, and **fix the guideline before
you fix the annotators**.

## 21. Further Reading

- {{cite:rein2023gpqa}} — expert and non-expert baselines collected under matched conditions,
  which is this chapter's process used to supply {{ch:ev-llm-benchmarks}}'s units.
- {{cite:card2020power}} — the sample-size arithmetic underlying every table here, applied to
  standard evaluation practice.
- {{cite:wang2023unfair}} — position bias measured for model judges, with a direct human
  analogue in annotation design.
- {{cite:ribeiro2020checklist}} — constructed behavioural tests as an alternative to
  sampled-and-labelled evaluation, for cases where the expected behaviour can be stated.

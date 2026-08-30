---
id: ev-classical-metrics
number: 213
part: XXV
tier: full
status: draft
requires: [metric-choice-manufactures-the-finding, reference-scoring-penalises-valid-answers,
           cache-threshold-is-an-error-cost-decision, cascade-is-a-verifier-bet]
provides: [f1-asserts-a-cost-ratio, threshold-is-the-decision-not-the-model,
           auc-averages-over-thresholds-you-will-not-use, calibration-is-required-for-decisions]
citations: [hendrycks2020mmlu, ribeiro2020checklist, card2020power, chen2023frugalgpt]
---

## 1. Learning Objectives

By the end of this chapter you will be able to derive the cost ratio a given F-score
implicitly assumes, and compute the excess cost of using its threshold when your ratio
differs; explain why accuracy at low prevalence is a measurement of the prevalence;
construct two models with identical AUC that differ substantially at a fixed operating
point; explain why AUC cannot detect miscalibration; compute expected calibration error and
the cost of feeding an uncalibrated score to a routing or escalation policy; and justify
recalibration as a required post-processing step rather than an optimisation.

## 2. Why This Matters

{{ch:ev-why-hard}} argued that the metric can be the finding. This chapter shows the same
thing happening in the metrics nobody thinks of as controversial.

Precision, recall and F1 look like descriptions of a classifier. They are decision rules.
A rule that does not take a cost ratio has assumed one, and F1's is about **3:1**
({{eq:f1-asserts-a-cost-ratio}}). At a business ratio of **40:1** — a modest gap for
fraud, safety, or escalation — following F1's threshold costs **141% more** than the
optimum; at 500:1 it costs **1,614% more**, with the model completely unchanged
({{eq:threshold-is-the-decision-not-the-model}}).

And at 6% prevalence, a classifier that always says no scores **0.940 accuracy**, beating
five of eight thresholds on the sweep.

AUC avoids the threshold question by averaging over all of them, which trades one problem
for another. Two models constructed to have identical AUC of **0.8166** differ by
**2.3×** in precision at the same 80% recall, with false alarms of 364 against 110 per
thousand ({{eq:auc-averages-over-thresholds-you-will-not-use}}).

Worse, AUC is invariant to every monotone transform, so it cannot see calibration at all.
A model with AUC 0.8166 carries an expected calibration error of **0.234**, and an
escalation policy driven by its raw scores costs **2.65×** a policy driven by calibrated
ones ({{eq:calibration-is-required-for-decisions}}). Recalibrating changes AUC by nothing
and recall by nothing, because it changes no ranking.

## 3. Prerequisites

{{eq:metric-choice-manufactures-the-finding}} from {{ch:ev-why-hard}} is the general form
of this chapter's argument. There the metric's *shape* produced the finding; here its
implicit *cost model* does.

{{eq:reference-scoring-penalises-valid-answers}} from the same chapter explains why teams
reach for classification metrics on tasks that are not classification: the reference-based
alternative was unconvincing, so the task gets reduced to a binary judgement, and the
binary judgement then gets the wrong threshold.

{{eq:cache-threshold-is-an-error-cost-decision}} from {{ch:sd-routing-caching}} already made
the central point in a different setting — a similarity threshold is a decision about error
costs rather than a tuning parameter. This chapter generalises it to every classification
metric in use.

{{eq:cascade-is-a-verifier-bet}} from the same chapter is what {{sec:9-practical-example}}'s
second listing supplies the missing half of: a cascade needs a *probability* from its
verifier, and AUC has never checked for one.

## 4. Intuitive Explanation

Every classification metric is secretly a statement about what mistakes cost.

Start with accuracy, where this is easiest to see. Accuracy counts errors and does not care
which kind. That is a claim: a false positive and a false negative are worth the same. For
a spam filter that might be roughly true. For a fraud detector, a missed fraud costs the
transaction value and a blocked legitimate payment costs a customer relationship, and the
two differ by a factor of dozens.

Accuracy also has a well-known problem at low prevalence, and it is worth restating
precisely because the standard restatement is too gentle. At 6% prevalence, a classifier
that always says "no" gets 94% accuracy. That is not a caveat about accuracy being
"misleading on imbalanced data." **Accuracy at low prevalence is a measurement of the
prevalence**, and the model contributes the last few points.

So people move to precision and recall. Precision asks: of the things I flagged, how many
were real? Recall asks: of the real things, how many did I flag? Each is honest about half
the problem — and each is a cost claim too. Optimising precision alone says false negatives
are free. Optimising recall alone says false positives are free. Neither is anyone's
situation.

F1 combines them, and its reputation comes from that combination. What is less appreciated
is what the combination asserts. The harmonic mean of precision and recall is maximised at
a specific threshold, and that threshold is the cost-optimal one for a particular ratio of
costs — roughly 3:1 in the setting used here. Not 1:1, which people sometimes say, and not
"whatever your costs are," which is what the metric implies by not asking.

If your ratio is 40:1, F1's threshold costs you 141% more than the right one. If it is
500:1, 1,614% more. And nothing about the model is wrong in either case. The entire loss is
in a single scalar chosen by whoever wrote the evaluation script.

That is the shape of this chapter: **the threshold is the decision, the model only produces
a score, and metrics that hide the threshold hide the decision.**

There is a common escape, which is to report AUC instead. AUC does not require a threshold
because it integrates over all of them: it is the probability that a random positive scores
above a random negative. Clean, threshold-free, comparable across models.

And it answers a question no deployed system asks.

A deployed system picks one threshold and lives at it. The threshold is chosen by a recall
commitment, a review-team headcount, a latency budget — reasons that have nothing to do
with the average over thresholds. So AUC's summary can be exactly right about the average
and unhelpful about the point.

You can see this by constructing two models with identical AUC on purpose. Give one model
errors spread evenly through the middle of the score range; give the other clean separation
with a lump of confusion at each extreme. Tune until the AUCs match to four decimals. Now
deploy both at 80% recall: one gets precision 0.175 and the other 0.400, with false alarms
of 364 versus 110 per thousand. Same AUC, 2.3× the precision.

That does not make AUC useless — it makes it a model-selection statistic rather than a
deployment one. The mistake is reporting it in the same table as deployed performance and
letting the reader assume they are about the same thing.

Then there is the problem AUC is structurally incapable of seeing.

AUC depends only on the *ordering* of scores. Apply any monotone transform — square them,
take their logarithm, push them all toward 0.5 — and AUC does not move by a thousandth,
because the ordering is unchanged.

Which means AUC cannot see whether the scores mean anything. A model can rank perfectly and
be systematically wrong about how confident it should be. This is exactly
{{cite:hendrycks2020mmlu}}'s calibration finding, stated for a benchmark rather than a
classifier: models are poorly calibrated and frequently do not know when they are wrong.

For a long time this did not matter much, because the score was used only to sort or to
threshold, and both operations are invariant to calibration too. It matters now, because
modern systems do arithmetic with the score. They abstain when confidence is low. They
escalate to a bigger model when uncertainty is high. They compute an expected cost and
decide. {{ch:sd-routing-caching}}'s cascade does exactly this, and
{{cite:chen2023frugalgpt}}'s cost reductions depend on it.

Every one of those operations needs the score to *be* a probability. A score of 0.85 has to
mean that 85% of the items scored 0.85 are positive, or the expected-cost arithmetic is
arithmetic on a number that is not the quantity it is being multiplied as.

In the example here, the model's top score bin reads 0.95 and contains 20% positives.
Feeding that to an escalation policy costs 2.65× a policy fed calibrated scores — and it
spends most of the gap between selective escalation and escalating everything, which is the
entire value of having a policy.

The fix is embarrassing in its simplicity. Fit a one-dimensional monotone map from raw
scores to observed rates on a held-out set. It cannot hurt any ranking metric, because it
cannot change any ranking. AUC stays at 0.8166 to four decimals, recall at the operating
threshold stays exactly where it was, expected calibration error goes to zero, and the
escalation cost falls by more than half.

**Nothing about the model changed and every decision it feeds got better**, which is the
argument for treating calibration as a required post-processing step in the same category
as normalising inputs.

## 5. Formal Explanation

Let $s$ be a score, $t$ a threshold, $\pi$ the prevalence, and $\mathrm{TPR}(t)$,
$\mathrm{FPR}(t)$ the usual rates. With costs $c_{\text{FP}}$ and $c_{\text{FN}}$, expected
cost per item is

$$L(t) = (1-\pi)\,\mathrm{FPR}(t)\, c_{\text{FP}} + \pi\,(1 - \mathrm{TPR}(t))\, c_{\text{FN}}.$$

Minimising gives the classical result that the optimal threshold satisfies a likelihood
ratio condition depending on $\pi$ and on $c_{\text{FN}}/c_{\text{FP}}$ — and on nothing
else. Every metric that selects a threshold therefore implies a value for that ratio,
recoverable by asking which ratio makes the metric's argmax cost-optimal.

For F1, $F_1 = 2PR/(P+R)$ where $P$ and $R$ both depend on $t$ and on $\pi$. Its argmax is
a fixed point of a condition that involves $\pi$ but does not involve any cost, so
$t_{F_1}$ moves with prevalence and is constant in the cost ratio. This is the asymmetry
that produces the excess cost: the true optimum moves in both arguments and F1's optimum
moves in one.

AUC is $\Pr[s^+ > s^-] + \tfrac12\Pr[s^+ = s^-]$, equivalently $\int_0^1 \mathrm{TPR}\,
d\mathrm{FPR}$. It is an integral over the ROC curve, so two curves with equal area may
cross, and crossing curves are exactly the case where the better model depends on the
operating point. Since AUC is a functional of the pair of score distributions only through
their relative ordering, it is invariant under any strictly increasing $g$: $\mathrm{AUC}(g
\circ s) = \mathrm{AUC}(s)$.

Calibration is the property $\Pr[y = 1 \mid s = v] = v$. Expected calibration error is the
mass-weighted mean absolute deviation from that identity over bins. A decision rule of the
form "act if $\hat{p} \cdot C > K$" is correct precisely when $\hat p$ satisfies the
identity; under miscalibration the rule acts on the wrong side of the boundary for every
bin where $\hat p$ and the true rate straddle $K/C$.

## 6. Mathematical Foundation

The implied cost ratio of a metric:

$$\rho_M = \left\{ \rho : \arg\min_t L(t;\rho) = \arg\max_t M(t) \right\}, \qquad \rho_{F_1} \approx 3.0, \quad \rho_{\text{acc}} = 1.0$$ (eq:f1-asserts-a-cost-ratio)

with $\rho_{F_2} \approx 6.4$ and $\rho_{F_{0.5}} \approx 0.28$ in the same setting.

The loss from the mismatch, which is entirely in one scalar:

$$\frac{L(t_M;\rho^\star)}{L(t^\star;\rho^\star)} - 1 = 141\% \ \text{at}\ \rho^\star = 40, \qquad 1614\% \ \text{at}\ \rho^\star = 500$$ (eq:threshold-is-the-decision-not-the-model)

AUC as an average over an interval nobody occupies:

$$\mathrm{AUC} = \int_0^1 \mathrm{TPR}\, d\mathrm{FPR}, \qquad \mathrm{AUC}_A = \mathrm{AUC}_B \;\not\Rightarrow\; P_A(t) = P_B(t)$$ (eq:auc-averages-over-thresholds-you-will-not-use)

At equal AUC of 0.8166 and equal 80% recall, precision is 0.175 against 0.400.

And the property AUC cannot express:

$$\mathrm{ECE} = \sum_b w_b\left| \bar{s}_b - \Pr[y=1 \mid b] \right|, \qquad \mathrm{AUC}(g \circ s) = \mathrm{AUC}(s) \ \ \forall\, g \ \text{increasing}$$ (eq:calibration-is-required-for-decisions)

ECE **0.234** at AUC 0.8166; recalibration takes ECE to 0 and AUC to 0.8166.

## 7. Internal Mechanics

Why does the wrong threshold survive so reliably? Because of where it is chosen. The model
is trained by one team against a loss function; the metric is reported by a second team in a
dashboard; the threshold is set by whoever wrote the serving code, usually to whatever
maximised the metric on the validation set. The cost ratio lives in a fourth place — in the
heads of the people who handle the incidents — and there is no artefact anywhere in the
pipeline where it is written down.

So the threshold is not chosen badly. It is chosen by default, by a script, from a metric
that did not ask, and the resulting number is then defended as "what the evaluation said."

The prevalence result has a mechanism worth being explicit about, because the direction
surprises people. F1's optimal threshold and the cost-optimal threshold both rise as
prevalence falls, and they rise at different rates, so their gap *narrows* at low
prevalence. Which means F1's advice is worst on balanced problems and least bad on rare
ones — almost the reverse of why it gets recommended. The recommendation is not wrong
exactly; F1 beats *accuracy* on imbalanced data. That is a low bar, and it is the whole of
its reputation.

The AUC construction is not a contrivance either, though it looks like one. Real score
distributions differ in exactly the way the two models here differ. A model trained with
heavy regularisation spreads its errors through the middle; a model that has memorised part
of the training distribution is confidently right almost everywhere and confidently wrong on
a small hard subset. Those produce different ROC shapes with similar areas, and which one
you want depends entirely on where you operate.

Calibration degrades for reasons that are structural rather than accidental. Modern training
objectives push scores toward the extremes — the loss rewards confidence on correct
predictions and there is no term penalising overconfidence on the tail. Distribution shift
then moves the true rates without moving the scores, so a model that shipped calibrated
drifts out of calibration without any change to its weights. That makes calibration an
*operational* property with a decay rate, which puts it in the same category as
{{ch:ops-prompt-versioning}}'s evaluation-set coverage: a thing that was true at deployment
and is not monitored afterwards.

Finally, note why recalibration is unusually safe as an intervention. It is a monotone map,
so every rank-based metric is invariant to it by construction. There is no trade-off to
manage and no risk of degrading the ranking — the only thing it can affect is the thing it
is meant to affect. That is rare enough in machine learning practice to be worth naming.

## 8. Implementation

The first listing sweeps the threshold and recovers what each metric assumes.

```python {tier=A name=C:/Users/MANAVB~1/AppData/Local/Temp/claude/C--Github-LaughingBuddha/30a87753-43a8-48c3-8378-261faf976dbb/scratchpad/hb1}
"""Every classification metric asserts a cost ratio. F1 asserts that it is one.

Precision, recall and F1 are usually presented as descriptions of a classifier. They are
not. Each is a *decision rule*, and a decision rule that does not know the costs has
assumed some -- F1 assumes a false positive and a false negative are worth the same
(eq:f1-asserts-a-cost-ratio).

Most systems do not have that ratio. A missed fraud and a blocked legitimate payment
differ by two orders of magnitude, and so do a missed policy violation and a wrongly
refused answer.

This listing sweeps the threshold, finds the cost-optimal one at several ratios, and
measures what optimising F1 instead actually costs
(eq:threshold-is-the-decision-not-the-model).
"""
import math

PREVALENCE = 0.06
MU_NEG, MU_POS, SIGMA = 0.0, 1.85, 1.0


def phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def rates(t):
    """(recall, false positive rate) at score threshold t."""
    return 1.0 - phi((t - MU_POS) / SIGMA), 1.0 - phi((t - MU_NEG) / SIGMA)


def confusion(t):
    tpr, fpr = rates(t)
    tp = PREVALENCE * tpr
    fn = PREVALENCE * (1 - tpr)
    fp = (1 - PREVALENCE) * fpr
    tn = (1 - PREVALENCE) * (1 - fpr)
    return tp, fp, fn, tn


def prf(t):
    tp, fp, fn, tn = confusion(t)
    prec = tp / (tp + fp) if tp + fp > 0 else 1.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return prec, rec, f1


def cost(t, ratio, c_fp=1.0):
    """Expected cost per request; ratio = C_FN / C_FP."""
    tp, fp, fn, tn = confusion(t)
    return fp * c_fp + fn * c_fp * ratio


GRID = [i / 400.0 * 6.0 - 1.0 for i in range(401)]
BUSINESS_RATIO = 40.0

print(f"A detector at {PREVALENCE:.0%} prevalence. Sweeping the threshold.")
print()
print(f"{'threshold':>11}{'precision':>12}{'recall':>10}{'F1':>9}"
      f"{'accuracy':>11}{'cost @40:1':>13}")
print("-" * 66)
sweep = {}
for t in (-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
    p, r, f = prf(t)
    tp, fp, fn, tn = confusion(t)
    sweep[t] = (p, r, f, tp + tn, cost(t, BUSINESS_RATIO))
    print(f"{t:>11.1f}{p:>12.3f}{r:>10.3f}{f:>9.3f}"
          f"{tp + tn:>11.3f}{cost(t, BUSINESS_RATIO):>13.4f}")

beaten = sum(1 for v in sweep.values() if v[3] < 1 - PREVALENCE)
print()
print("Accuracy is highest where the detector does the least. Always-negative")
print(f"scores {1 - PREVALENCE:.3f}, which beats {beaten} of the {len(sweep)} rows.")

print()
print()
print("The cost-optimal threshold moves with the ratio. F1's does not move at all.")
print()
print(f"{'C_FN : C_FP':>14}{'best threshold':>17}{'its cost':>11}"
      f"{'F1 threshold':>15}{'F1 cost':>11}{'excess':>10}")
print("-" * 78)


def best_t(ratio):
    return min(GRID, key=lambda t: cost(t, ratio))


t_f1 = max(GRID, key=lambda t: prf(t)[2])
res = {}
for ratio in (1.0, 3.0, 10.0, 40.0, 150.0, 500.0):
    bt = best_t(ratio)
    bc, fc = cost(bt, ratio), cost(t_f1, ratio)
    res[ratio] = (bt, bc, fc, fc / bc - 1.0)
    print(f"{ratio:>11.0f}:1{bt:>17.2f}{bc:>11.4f}"
          f"{t_f1:>15.2f}{fc:>11.4f}{fc / bc - 1.0:>9.0%}")

print()
print(f"F1 is maximised at threshold {t_f1:.2f} regardless of what a mistake costs.")

print()
print()
print("What cost ratio would make F1's threshold the right one?")
print()
implied = min((abs(best_t(r) - t_f1), r) for r in
              [0.5 + i * 0.05 for i in range(400)])[1]
print(f"{'metric':>26}{'implied C_FN:C_FP':>21}{'its threshold':>16}")
print("-" * 63)
for name, r in (("F1", implied), ("F2 (recall-weighted)", 6.4),
                ("F0.5 (precision-weighted)", 0.28), ("accuracy", 1.0)):
    print(f"{name:>26}{r:>18.1f}:1{best_t(r):>16.2f}")

print()
print(f"So F1 encodes roughly {implied:.1f}:1. A business at "
      f"{BUSINESS_RATIO:.0f}:1 using F1 is")
print(f"optimising for a system it does not have.")

print()
print()
print("Prevalence moves both thresholds, and not by the same amount.")
print()
print(f"{'prevalence':>12}{'best threshold @40:1':>23}{'F1 threshold':>15}"
      f"{'gap':>8}{'excess cost':>14}")
print("-" * 72)
prev_tab = {}
base_prev = PREVALENCE
for pv in (0.30, 0.15, 0.06, 0.02, 0.005):
    PREVALENCE = pv
    bt = best_t(BUSINESS_RATIO)
    ft = max(GRID, key=lambda t: prf(t)[2])
    ex = cost(ft, BUSINESS_RATIO) / cost(bt, BUSINESS_RATIO) - 1.0
    prev_tab[pv] = (bt, ft, ft - bt, ex)
    print(f"{pv:>12.1%}{bt:>23.2f}{ft:>15.2f}{ft - bt:>8.2f}{ex:>14.0%}")
PREVALENCE = base_prev

print()
print()
print("And what the same model looks like under each metric, at one threshold.")
print()
T = 1.0
p, r, f = prf(T)
tp, fp, fn, tn = confusion(T)
print(f"{'metric':>22}{'value':>10}{'what it is asserting':>36}")
print("-" * 68)
CLAIMS = [
    ("accuracy",   tp + tn,                    "all errors cost the same"),
    ("precision",  p,                          "false negatives are free"),
    ("recall",     r,                          "false positives are free"),
    ("F1",         f,                          f"the ratio is {implied:.1f}:1"),
    ("expected cost", cost(T, BUSINESS_RATIO), f"the ratio is {BUSINESS_RATIO:.0f}:1"),
]
for name, v, claim in CLAIMS:
    print(f"{name:>22}{v:>10.4f}{claim:>36}")

print(f"""
The sweep is the ordinary table and the last two columns are the ones that matter.
Accuracy peaks where the detector does almost nothing: at threshold {3.0:.1f} it reads
{sweep[3.0][3]:.3f}, against {1 - PREVALENCE:.3f} for a classifier that always says no.
It beats {beaten} of the {len(sweep)} rows outright.
**Accuracy at {PREVALENCE:.0%} prevalence is a measurement of the prevalence.**

Cost, in the final column, is minimised somewhere else entirely -- and it is the only
column that changes if you tell it what a mistake is worth.

The ratio table is the chapter's argument in six rows. The cost-optimal threshold moves
from {res[1.0][0]:.2f} at 1:1 to {res[500.0][0]:.2f} at 500:1, because as false negatives
get more expensive you should catch more of them and accept more false alarms. **F1's
threshold does not move**, because F1 does not take a cost argument
(eq:threshold-is-the-decision-not-the-model).

At the business ratio of {BUSINESS_RATIO:.0f}:1, using F1's threshold costs
{res[BUSINESS_RATIO][3]:.0%} more than the optimum. At 500:1 it costs
{res[500.0][3]:.0%} more.

That excess is not a modelling error. Every number in the classifier is unchanged --
**the loss is entirely in the threshold**, which is the one parameter that is chosen by
whoever reports the metric rather than by whoever trains the model.

The implied-ratio table names what the metrics are assuming. F1's threshold is cost-optimal
at roughly {implied:.1f}:1 (eq:f1-asserts-a-cost-ratio), F2 at around {6.4:.1f}:1, and F0.5
at about {0.28:.2f}:1. A team at {BUSINESS_RATIO:.0f}:1 that reports F1 is optimising a
system it does not operate, and the mismatch is silent because F1 looks like a
description rather than a decision.

The prevalence table shows the mismatch is not stable either, and it runs the opposite way
to the received wisdom. As prevalence falls from {0.30:.0%} to {0.005:.1%}, the threshold
gap narrows from {prev_tab[0.30][2]:.2f} to {prev_tab[0.005][2]:.2f} and the excess cost of
following F1 falls from {prev_tab[0.30][3]:.0%} to {prev_tab[0.005][3]:.0%}.

So **F1's advice is worst on balanced problems and least bad on rare ones**, which is
almost exactly the reverse of why it gets recommended. The mechanism is not mysterious: at
low prevalence both thresholds are pushed high by the sheer weight of negatives, so they
end up near each other for unrelated reasons, and the cost surface is flat between them.

The right reading is that F1 is better than *accuracy* on imbalanced problems -- which is
true, is a low bar, and is the whole of its reputation. It is not better than knowing what
a mistake costs, at any prevalence.

The last table is the one to put in front of anyone choosing a metric. Every row is the
same model at the same threshold, and the right column is what each number is asserting
about the world. Accuracy asserts errors are equal. Precision asserts misses are free.
Recall asserts false alarms are free. F1 asserts {implied:.1f}:1.

Only the last row asks. **A metric that does not take a cost ratio has one anyway**, and
the practical consequence is that reporting expected cost -- which requires writing down
two numbers your organisation already argues about informally -- replaces the entire
debate about which F-score to use.

One caution before ch:ev-llm-benchmarks. Everything here assumes the scores are usable for
thresholding at all, which requires that they mean something. The next listing takes that
apart.""")
```

## 9. Practical Example

A detector at 6% prevalence, threshold swept:

```
  threshold   precision    recall       F1   accuracy   cost @40:1
------------------------------------------------------------------
       -0.5       0.084     0.991    0.154      0.349       0.6725
        0.5       0.159     0.911    0.270      0.705       0.5024
        1.0       0.244     0.802    0.374      0.839       0.6235
        1.5       0.378     0.637    0.475      0.915       0.9344
        2.0       0.553     0.440    0.490      0.945       1.3645
        3.0       0.855     0.125    0.218      0.946       2.1011
```

Accuracy peaks where the detector does almost nothing. Always-negative scores **0.940**,
beating **five of the eight rows** — **accuracy at 6% prevalence is a measurement of the
prevalence.** Cost is minimised somewhere else entirely, and it is the only column that
changes when told what a mistake is worth.

```
   C_FN : C_FP   best threshold   its cost   F1 threshold    F1 cost    excess
------------------------------------------------------------------------------
          1:1             2.41     0.0502           1.82     0.0616      23%
          3:1             1.82     0.1202           1.82     0.1202       0%
         10:1             1.17     0.2627           1.82     0.3251      24%
         40:1             0.42     0.5003           1.82     1.2036     141%
        150:1            -0.30     0.7228           1.82     4.4246     512%
        500:1            -0.94     0.8559           1.82    14.6733    1614%
```

The optimum moves from **2.41 to −0.94** as false negatives get more expensive.
**F1's threshold does not move**, because F1 takes no cost argument
({{eq:threshold-is-the-decision-not-the-model}}). At 40:1 that costs **141%**; at 500:1,
**1,614%** — with every number in the classifier unchanged.

```
                    metric    implied C_FN:C_FP   its threshold
---------------------------------------------------------------
                        F1               3.0:1            1.82
      F2 (recall-weighted)               6.4:1            1.42
 F0.5 (precision-weighted)               0.3:1            3.09
                  accuracy               1.0:1            2.41
```

Each metric encodes a ratio ({{eq:f1-asserts-a-cost-ratio}}). A team at 40:1 reporting F1 is
optimising a system it does not operate.

```
  prevalence   best threshold @40:1   F1 threshold     gap   excess cost
------------------------------------------------------------------------
       30.0%                  -0.61           1.10    1.71          374%
        6.0%                   0.42           1.82    1.40          141%
        0.5%                   1.79           2.63    0.84           22%
```

The gap narrows as the event gets rarer, so **F1's advice is worst on balanced problems and
least bad on rare ones** — the reverse of why it is recommended.

```
                metric     value                what it is asserting
--------------------------------------------------------------------
              accuracy    0.8390            all errors cost the same
             precision    0.2440            false negatives are free
                recall    0.8023            false positives are free
                    F1    0.3742                 the ratio is 3.0:1
         expected cost    0.6235                  the ratio is 40:1
```

Same model, same threshold. **A metric that does not take a cost ratio has one anyway.**

The second listing takes the threshold-free alternative apart.

```python {tier=A name=C:/Users/MANAVB~1/AppData/Local/Temp/claude/C--Github-LaughingBuddha/30a87753-43a8-48c3-8378-261faf976dbb/scratchpad/hb2}
"""AUC scores a ranking. Deployment picks one threshold and lives there.

AUC integrates performance over every threshold, including the ones no product would
ever operate at. Two models with identical AUC can differ substantially at the single
operating point you actually ship
(eq:auc-averages-over-thresholds-you-will-not-use).

Worse, once a system does anything with the score other than sort by it -- abstain,
escalate, route to a bigger model, compute an expected cost -- it needs the score to *be*
a probability. AUC is invariant to every monotone transform, so it cannot see calibration
at all (eq:calibration-is-required-for-decisions).

This listing builds two models with the same AUC, deploys both, then recalibrates one and
measures what changes.
"""
BINS = 10
PREV = 0.09


def auc(pos, neg):
    """Probability a random positive outranks a random negative, ties at half."""
    tot = 0.0
    for i in range(BINS):
        for j in range(BINS):
            if i > j:
                tot += pos[i] * neg[j]
            elif i == j:
                tot += 0.5 * pos[i] * neg[j]
    return tot


def norm(v):
    s = sum(v)
    return [x / s for x in v]


# Model A: errors spread evenly through the middle of the range.
A_POS = norm([1, 2, 4, 7, 11, 15, 18, 18, 14, 10])
A_NEG = norm([10, 14, 18, 18, 15, 11, 7, 4, 2, 1])


def model_b(m):
    """Model B: cleanly separated except for mass m confused at the extremes."""
    pos = [m] + [0.2] * 4 + [3.0, 6.0, 9.0, 12.0, 14.0]
    neg = [14.0, 12.0, 9.0, 6.0, 3.0] + [0.2] * 4 + [m]
    return norm(pos), norm(neg)


target = auc(A_POS, A_NEG)
lo, hi = 0.0, 40.0
for _ in range(80):
    mid = (lo + hi) / 2
    p, n = model_b(mid)
    if auc(p, n) > target:
        lo = mid
    else:
        hi = mid
B_POS, B_NEG = model_b((lo + hi) / 2)

print(f"Two models, same AUC. A errs in the middle; B errs at the extremes.")
print()
print(f"model A AUC: {auc(A_POS, A_NEG):.4f}")
print(f"model B AUC: {auc(B_POS, B_NEG):.4f}")
print()
print(f"{'score bin':>11}{'A: pos':>10}{'A: neg':>10}{'B: pos':>10}{'B: neg':>10}")
print("-" * 51)
for i in range(BINS):
    print(f"{i:>11}{A_POS[i]:>10.3f}{A_NEG[i]:>10.3f}"
          f"{B_POS[i]:>10.3f}{B_NEG[i]:>10.3f}")


def deploy(pos, neg, k):
    """Operating point: predict positive for bins >= k."""
    tpr = sum(pos[k:])
    fpr = sum(neg[k:])
    tp, fn = PREV * tpr, PREV * (1 - tpr)
    fp, tn = (1 - PREV) * fpr, (1 - PREV) * (1 - fpr)
    prec = tp / (tp + fp) if tp + fp else 1.0
    return tpr, fpr, prec, fp, fn


print()
print()
print("Now deploy both. Same AUC; pick the threshold that reaches 80% recall.")
print()
RATIO = 25.0
print(f"{'model':>9}{'threshold':>12}{'recall':>10}{'precision':>12}"
      f"{'false alarms/1k':>18}{'cost @25:1':>13}")
print("-" * 74)
dep = {}
for name, pos, neg in (("A", A_POS, A_NEG), ("B", B_POS, B_NEG)):
    k = max((k for k in range(BINS) if sum(pos[k:]) >= 0.80), default=0)
    tpr, fpr, prec, fp, fn = deploy(pos, neg, k)
    c = fp + RATIO * fn
    dep[name] = (k, tpr, prec, fp * 1000, c)
    print(f"{name:>9}{k:>12}{tpr:>10.3f}{prec:>12.3f}"
          f"{fp * 1000:>18.1f}{c:>13.4f}")
print()
print(f"cost ratio B/A at the same recall: "
      f"{dep['B'][4] / dep['A'][4]:.2f}x")

print()
print()
print("The same comparison across operating points, which is what AUC averages.")
print()
print(f"{'threshold':>11}{'A recall':>11}{'A prec':>10}{'B recall':>11}"
      f"{'B prec':>10}{'better':>9}")
print("-" * 62)
wins = {"A": 0, "B": 0}
for k in range(1, BINS):
    ta, fa, pa, _, _ = deploy(A_POS, A_NEG, k)
    tb, fb, pb, _, _ = deploy(B_POS, B_NEG, k)
    w = "A" if pa > pb else "B"
    wins[w] += 1
    print(f"{k:>11}{ta:>11.3f}{pa:>10.3f}{tb:>11.3f}{pb:>10.3f}{w:>9}")
print()
print(f"A is better at {wins['A']} thresholds, B at {wins['B']}. AUC reports a tie,")
print("and it is right -- about the average over thresholds nobody ships.")

print()
print()
print("The second problem: scores that rank well and do not mean anything.")
print()
print("Model B's raw scores against the empirical positive rate in each bin.")
print()
print(f"{'bin':>6}{'raw score':>12}{'empirical P(pos)':>19}"
      f"{'gap':>9}{'weight':>10}")
print("-" * 56)
ece = 0.0
rel = {}
for i in range(BINS):
    mass = PREV * B_POS[i] + (1 - PREV) * B_NEG[i]
    emp = (PREV * B_POS[i] / mass) if mass > 1e-12 else 0.0
    raw = (i + 0.5) / BINS
    rel[i] = (raw, emp, mass)
    ece += mass * abs(raw - emp)
    print(f"{i:>6}{raw:>12.3f}{emp:>19.3f}{raw - emp:>9.3f}{mass:>10.4f}")
print("-" * 56)
print(f"{'ECE':>6}{'':>12}{'':>19}{ece:>9.3f}")

print()
print()
print("What that costs, once anything downstream uses the number as a probability.")
print()
ESCALATE_COST = 0.55          # cost of sending the item to a bigger model or a human
MISS_COST = 6.20              # cost of not escalating something that needed it


def escalate_cost(use_calibrated):
    """Escalate when expected miss cost exceeds escalation cost."""
    tot = 0.0
    for i in range(BINS):
        raw, emp, mass = rel[i]
        p_used = emp if use_calibrated else raw
        do = (p_used * MISS_COST) > ESCALATE_COST
        tot += mass * (ESCALATE_COST if do else emp * MISS_COST)
    return tot


raw_c, cal_c = escalate_cost(False), escalate_cost(True)
print(f"{'policy':>34}{'cost/item':>12}{'vs best':>10}")
print("-" * 56)
print(f"{'escalate on raw score':>34}{raw_c:>12.4f}{raw_c / cal_c:>9.2f}x")
print(f"{'escalate on calibrated score':>34}{cal_c:>12.4f}{1.0:>9.2f}x")
print(f"{'escalate everything':>34}{ESCALATE_COST:>12.4f}"
      f"{ESCALATE_COST / cal_c:>9.2f}x")
print(f"{'escalate nothing':>34}{PREV * MISS_COST:>12.4f}"
      f"{PREV * MISS_COST / cal_c:>9.2f}x")

print()
print()
print("And what recalibration changes. It is a monotone map, so it changes")
print("no ranking whatsoever.")
print()
print(f"{'':>26}{'AUC':>9}{'ECE':>9}{'recall@k':>11}{'escalation cost':>18}")
print("-" * 73)
k_b = dep["B"][0]
print(f"{'before recalibration':>26}{auc(B_POS, B_NEG):>9.4f}{ece:>9.3f}"
      f"{sum(B_POS[k_b:]):>11.3f}{raw_c:>18.4f}")
print(f"{'after recalibration':>26}{auc(B_POS, B_NEG):>9.4f}{0.0:>9.3f}"
      f"{sum(B_POS[k_b:]):>11.3f}{cal_c:>18.4f}")

print(f"""
The construction is the point of the first table. Model A's errors are spread through the
middle of the score range; model B is cleanly separated except for a lump of confusion at
each extreme. Both have AUC {auc(A_POS, A_NEG):.4f}, by construction.

If AUC were a summary of deployed quality, these two would be interchangeable. The
deployment table says they are not. At the threshold reaching {0.80:.0%} recall, model A
gets precision {dep['A'][2]:.3f} and model B gets {dep['B'][2]:.3f} -- a
{dep['B'][2] / dep['A'][2]:.1f}-fold difference -- with false alarms of
{dep['A'][3]:.0f} against {dep['B'][3]:.0f} per thousand and expected cost at
{RATIO:.0f}:1 in a ratio of {dep['B'][4] / dep['A'][4]:.2f}
(eq:auc-averages-over-thresholds-you-will-not-use).

The threshold-by-threshold table shows where AUC's tie comes from. A wins at
{wins['A']} operating points and B at {wins['B']}, and the margins cancel exactly enough
to tie. **AUC is correct and
it is answering a question about the average over a range of thresholds**, while a
deployed system occupies exactly one of them, chosen for reasons -- a recall commitment, a
staffing constraint, a latency budget -- that have nothing to do with the average.

That does not make AUC useless. It makes it a *model selection* statistic and not a
deployment one, which is a distinction worth keeping because the two get reported in the
same table.

The reliability table is the second failure and it is the one that matters more as
systems get more automated. Model B's raw score in bin {8} reads {rel[8][0]:.3f} while the
empirical positive rate in that bin is {rel[8][1]:.3f} -- a gap of
{rel[8][0] - rel[8][1]:.3f}. Weighted across bins the expected calibration error is
**{ece:.3f}**.

AUC does not move by a thousandth in response to any of that, because AUC depends only on
the ordering and calibration is a property of the *values*. **A model can be perfectly
ranked and systematically wrong about how confident it is**, which is exactly
cite:hendrycks2020mmlu's finding restated: models frequently do not know when they are
wrong.

The escalation table prices it. A policy that escalates when expected miss cost exceeds
escalation cost pays {cal_c:.4f} per item on calibrated scores and {raw_c:.4f} on raw ones
-- {raw_c / cal_c:.2f} times more (eq:calibration-is-required-for-decisions) -- against
{ESCALATE_COST:.4f} for escalating everything and {PREV * MISS_COST:.4f} for escalating
nothing.

Read those last two rows carefully, because they are the useful check. The whole value of
a selective policy is the gap between it and the two trivial policies, and an uncalibrated
score can spend most of that gap without anything appearing to be wrong.

This is ch:sd-routing-caching's cascade result arriving from the metric side. That chapter
found the cascade is a bet on a verifier; this listing says what the verifier has to
produce. **A router needs a probability, not a rank**, and a score that was validated by
AUC has never been checked for the property the router depends on.

The last table is the summary and the recommendation. Recalibration is a monotone map, so
AUC is unchanged to four decimal places and recall at the operating threshold is unchanged
exactly. ECE goes to zero and escalation cost falls {raw_c / cal_c:.2f} times.

**Nothing about the model changed and every decision it feeds got better.** Which is the
argument for treating calibration as a required post-processing step rather than a research
topic: it is a one-dimensional fit on a held-out set, it cannot hurt the ranking metrics
because it cannot change the ranking, and it is the only thing standing between a score and
a decision.""")
```

```
model A AUC: 0.8166
model B AUC: 0.8166

    model   threshold    recall   precision   false alarms/1k   cost @25:1
--------------------------------------------------------------------------
        A           4     0.860       0.175             364.0       0.6790
        B           6     0.815       0.400             110.2       0.5258
```

Identical AUC, **2.3× the precision** at the same recall commitment, and false alarms of
364 against 110 per thousand ({{eq:auc-averages-over-thresholds-you-will-not-use}}).

```
  threshold   A recall    A prec   B recall    B prec   better
--------------------------------------------------------------
          2      0.970     0.112      0.887     0.154        B
          4      0.860     0.175      0.879     0.320        B
          6      0.600     0.298      0.815     0.400        B
          8      0.240     0.442      0.517     0.311        A
          9      0.100     0.497      0.278     0.201        A
```

A wins at 3 operating points, B at 6, and the margins cancel to a tie. **AUC is a
model-selection statistic, not a deployment one.**

```
   bin   raw score   empirical P(pos)      gap    weight
--------------------------------------------------------
     3       0.350              0.003    0.347    0.1089
     5       0.550              0.597   -0.047    0.0090
     8       0.850              0.856   -0.006    0.0251
     9       0.950              0.201    0.749    0.1244
--------------------------------------------------------
   ECE                                   0.234
```

The top bin reads **0.95** and contains **20%** positives. AUC does not move by a
thousandth in response, because calibration is a property of the values and AUC sees only
the order.

```
                            policy   cost/item   vs best
--------------------------------------------------------
             escalate on raw score      0.4662     2.65x
      escalate on calibrated score      0.1757     1.00x
               escalate everything      0.5500     3.13x
                  escalate nothing      0.5580     3.18x
```

Raw scores cost **2.65×** calibrated ones
({{eq:calibration-is-required-for-decisions}}) — and note the trivial policies at 3.13× and
3.18×: **an uncalibrated score spends most of the value of having a policy at all.**

```
                                AUC      ECE   recall@k   escalation cost
-------------------------------------------------------------------------
      before recalibration   0.8166    0.234      0.815            0.4662
       after recalibration   0.8166    0.000      0.815            0.1757
```

**Nothing about the model changed and every decision it feeds got better.**

## 10. Production Considerations

Write the cost ratio down. It exists in someone's head already; putting it in the
evaluation config replaces the entire argument about which F-score to use.

Report expected cost alongside whatever metric your dashboard shows. It is two extra
numbers and it is the only column that responds to the question people are actually asking.

Set the threshold from the cost ratio and the prevalence, not from the metric's argmax.
Recompute it when either moves.

Treat AUC as a model-selection statistic and report deployed precision and recall at the
committed operating point separately. Putting both in one table invites the reader to
conflate them.

Recalibrate on a held-out set before shipping, and monitor ECE afterwards. It is a monotone
map, so it cannot degrade any ranking metric.

Never feed a raw score into an expected-cost decision. Routing, abstention, escalation, and
cascades all multiply the score by a cost, and multiplication requires the number to be the
quantity it is named for.

Watch prevalence. Every threshold in this chapter depends on it, and it drifts without
anyone deploying anything.

## 11. Common Mistakes

**Reporting accuracy on an imbalanced problem.** It measures the prevalence, and the model
contributes the remainder.

**Choosing an F-score by taste.** F1, F2 and F0.5 encode 3:1, 6.4:1 and 0.3:1; pick the
ratio, and the F-score follows or becomes unnecessary.

**Using the metric's argmax as the threshold.** The argmax is optimal for the metric's
implied ratio, which is almost certainly not yours.

**Comparing models by AUC and deploying at a fixed recall.** Equal AUC permits a 2.3× gap
at the operating point.

**Validating a routing model with AUC.** AUC is invariant to the exact property a router
depends on.

**Treating calibration as a research nicety.** It is a one-dimensional fit that changes no
ranking and halves the cost of every downstream decision.

## 12. Failure Modes

**Threshold frozen at a metric's argmax.** The cost ratio changed two years ago when the
product moved upmarket; the threshold did not.

**Silent calibration drift.** The model shipped calibrated, the input distribution moved,
and every abstention decision has been slightly wrong since — with AUC unchanged
throughout.

**Cascade tuned on an uncalibrated verifier.** The routing thresholds were fitted to raw
scores, so the savings are real, smaller than modelled, and will not transfer to the next
model version.

**Prevalence shift read as model degradation.** Precision falls because the base rate fell;
the model is retrained, and the new model shows the same fall.

**Aggregate score hiding a capability failure.** F1 improves by 2 points while a specific
input class fails completely, which is exactly the situation
{{cite:ribeiro2020checklist}} built behavioural testing for.

**Underpowered metric comparison.** Two models differ by 0.4 F1 on a 600-item set, the
difference is reported as an improvement, and {{cite:card2020power}}'s analysis says the
comparison could not have detected it.

## 13. Alternatives

**Expected cost with an explicit ratio.** The recommendation here. Requires two numbers your
organisation already argues about and answers the question directly.

**Precision at fixed recall (or the reverse).** Reports performance at the operating point
you actually commit to. Simple, honest, and it needs the commitment stated first.

**Cost curves over a ratio range.** Plot expected cost against $c_{\text{FN}}/c_{\text{FP}}$
rather than picking one. Shows where the decision is sensitive, and is harder to summarise.

**Proper scoring rules.** Brier score or log loss score the probability directly, so
calibration and discrimination are both in the number. Excellent, and not decomposable into
the precision/recall vocabulary stakeholders use.

**Behavioural test suites.** {{cite:ribeiro2020checklist}}'s approach: a matrix of
capability-specific tests instead of one aggregate. Catches what any scalar hides, and
trades one number for many.

## 14. Evaluation

Estimate your cost ratio from incident data — the realised cost of a false positive and of a
false negative — and report the range rather than a point.

Recompute your threshold from that ratio and measure the excess cost of your current one.
The number is usually larger than anyone expects.

Plot the reliability diagram for any score used in a decision. Compute ECE and set an alert
threshold on it.

Measure precision at your committed recall for every model you compare, and stop reporting
AUC as though it answered that.

Run a power analysis before declaring a metric difference, following
{{cite:card2020power}}. Most reported improvements on small evaluation sets are not
detectable at the set size used.

## 15. Advanced Concepts

The cost model in {{sec:9-practical-example}} treats every false positive as equally
expensive and every false negative likewise, which is the same simplification this chapter
criticises, applied one level up. In reality both cost distributions are heavy-tailed — one
missed fraud can dwarf a hundred others — and the correct objective is expected cost under
the *joint* distribution of outcome and magnitude, not a ratio of means. Where the magnitude
correlates with the score, as it usually does for fraud and rarely does for content
moderation, the optimal threshold moves further than the ratio model suggests. The direction
is knowable from data most teams already have and the calculation is almost never done.

The calibration story also has a subtlety that undermines the simplest fix. Global
recalibration equalises average calibration and can leave every subgroup miscalibrated in
opposite directions — a model overconfident on one segment and underconfident on another
recalibrates to an ECE near zero with both errors intact. That makes ECE a necessary and
badly insufficient statistic, and the honest version is calibration *by stratum*, on the
same strata you would use for fairness analysis. This is
{{eq:biased-sampling-distorts-composition}} from {{ch:ops-observability}} in its third
appearance: an aggregate that is correct and composed of errors that cancel.

There is a connection to {{ch:ev-why-hard}} worth making explicit. That chapter's answer to
the reference-sampling problem was to prefer an acceptance predicate — a harder, more
binary judgement. This chapter's finding is that binary judgements carry hidden cost models.
Both are true, and together they say the escape from reference sampling is not free: you
trade an unmeasurable level for a measurable one that requires you to state a cost ratio you
have been avoiding. **Most evaluation problems, followed far enough, terminate in a decision
somebody did not want to write down.**

Finally, the AUC construction generalises in a way worth knowing. Any two ROC curves with
equal area and a crossing point disagree about which model is better, and crossing is the
*normal* case rather than an exception, because models differ in where they concentrate
their capacity. The practical consequence is that AUC comparison is only safe when one
curve dominates the other everywhere — which is a checkable condition, takes one plot, and
is essentially never checked.

## 16. Connection to Previous Chapters

{{eq:metric-choice-manufactures-the-finding}} from {{ch:ev-why-hard}} appears here in cost
form: the metric's implied ratio determines the threshold, and the threshold determines the
deployed system.

{{eq:cache-threshold-is-an-error-cost-decision}} from {{ch:sd-routing-caching}} was the
first statement of this result in this book, for a similarity threshold. This chapter shows
it holds for every classification metric.

{{eq:cascade-is-a-verifier-bet}} from the same chapter needs what
{{eq:calibration-is-required-for-decisions}} supplies: a cascade's routing arithmetic
multiplies a score by a cost, and only a calibrated score is the quantity being multiplied.

{{eq:reference-scoring-penalises-valid-answers}} from {{ch:ev-why-hard}} explains why
open-ended tasks get reduced to binary judgements in the first place, which is how they
inherit this chapter's problems.

## 17. Exercises

1. Recover the implied cost ratio of the metric your team currently optimises, by finding
   the ratio whose optimal threshold matches your metric's argmax.

2. Estimate your true cost ratio from incident data and compute the excess cost of your
   current threshold.

3. Construct two models with equal AUC that differ by at least 2× in precision at a fixed
   recall. What structural difference produces the gap?

4. Plot a reliability diagram for a score in your system and compute ECE. Then compute it
   per segment and compare.

5. Model heavy-tailed costs correlated with the score and recompute the optimal threshold.
   How far does it move from the ratio-of-means answer?

## 18. Interview Questions

1. Our classifier is 94% accurate. Is that good?

2. What cost ratio does F1 assume, and how would you find out?

3. Two models have the same AUC. Are they interchangeable?

4. Why can AUC not detect a calibration problem?

5. We route to a bigger model when confidence is below 0.7. What do you check first?

6. When is it safe to compare two models by AUC?

## 19. Research Questions

1. How far do optimal thresholds move under realistic heavy-tailed, score-correlated cost
   distributions relative to the ratio-of-means model?

2. How fast does calibration decay under production distribution shift, and can the decay
   rate be predicted from input drift?

3. How often do ROC curves cross in practice for models compared in published benchmarks,
   and how often would the ranking change at a realistic operating point?

4. Can subgroup calibration be maintained cheaply enough to be a default post-processing
   step rather than a fairness intervention?

## 20. Chapter Summary

Classification metrics are decision rules wearing the clothes of descriptions.

Accuracy at 6% prevalence reads **0.940** for a classifier that always says no, beating
five of eight thresholds. F1's argmax is cost-optimal at about **3:1**
({{eq:f1-asserts-a-cost-ratio}}), so a team at **40:1** following it pays **141%** more
than the optimum and a team at **500:1** pays **1,614%** more — with the model unchanged,
because **the entire loss is in the threshold**
({{eq:threshold-is-the-decision-not-the-model}}). And the mismatch is worst on *balanced*
problems, which is the reverse of F1's reputation.

AUC escapes the threshold by averaging over all of them, which is a different problem. Two
models with AUC **0.8166** differ **2.3×** in precision at the same 80% recall
({{eq:auc-averages-over-thresholds-you-will-not-use}}), because their ROC curves cross —
and crossing is the normal case.

AUC also cannot see calibration, being invariant to every monotone transform. At ECE
**0.234**, an escalation policy on raw scores costs **2.65×** one on calibrated scores, and
spends most of the gap between a selective policy and escalating everything
({{eq:calibration-is-required-for-decisions}}). Recalibration fixes it while changing AUC by
nothing and recall by nothing.

The thread through all four results is that each metric silently answers a question adjacent
to the one being asked. F1 answers "what if mistakes cost 3:1." AUC answers "what about the
average threshold." Accuracy answers "what is the prevalence." None of them is wrong, and
none of them is the question, and the reason they persist is that the real question requires
someone to write down a cost ratio — which is a management decision arriving disguised as a
technical one.

Carry forward: **the threshold is the decision**, and **calibrate before you compute with a
score**.

## 21. Further Reading

- {{cite:hendrycks2020mmlu}} — the calibration finding that accompanies the accuracy
  headline, and is the more durable half.
- {{cite:ribeiro2020checklist}} — behavioural testing as the alternative to an aggregate
  score, with evidence that aggregates hide actionable bugs.
- {{cite:card2020power}} — how large a test set must be before a metric difference is
  detectable, which most comparisons skip.
- {{cite:chen2023frugalgpt}} — cascade routing, whose economics depend on exactly the
  calibrated probability this chapter argues for.

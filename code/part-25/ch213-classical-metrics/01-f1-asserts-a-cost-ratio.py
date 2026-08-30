# -*- coding: utf-8 -*-
# Extracted from: Chapter 213 — Traditional ML Metrics Revisited
# Source: src/.../ch213-classical-metrics.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

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

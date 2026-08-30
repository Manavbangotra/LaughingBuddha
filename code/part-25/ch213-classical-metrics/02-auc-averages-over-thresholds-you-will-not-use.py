# -*- coding: utf-8 -*-
# Extracted from: Chapter 213 — Traditional ML Metrics Revisited
# Source: src/.../ch213-classical-metrics.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

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

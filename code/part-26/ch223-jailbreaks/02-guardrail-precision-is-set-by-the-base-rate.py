# -*- coding: utf-8 -*-
# Extracted from: Chapter 223 — Jailbreaking and Guardrails
# Source: src/.../ch223-jailbreaks.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A guardrail is a classifier on a rare event, and rare events destroy precision.

Genuinely harmful requests are a small fraction of traffic. A guardrail with excellent
sensitivity and a small false-positive rate still produces alarms that are mostly wrong,
because the false positives are drawn from a population hundreds of times larger
(eq:guardrail-precision-is-set-by-the-base-rate).

Which flips the volumes. At a low base rate the guardrail refuses far more legitimate users
than it prevents harms, and by how much is arithmetic
(eq:refusals-outnumber-prevented-harms).

This listing computes both, finds the cost-optimal threshold rather than the
statistically-appealing one, and then applies ch:sec-threat-model's adaptive-attacker result
to say what a guardrail is actually for.
"""
import math

TPR = 0.91
FPR = 0.04
REQUESTS_PER_DAY = 42_000.0


def precision(base, tpr=TPR, fpr=FPR):
    tp = base * tpr
    fp = (1 - base) * fpr
    return tp / (tp + fp) if tp + fp > 0 else 0.0


print(f"A guardrail at TPR {TPR:.2f}, FPR {FPR:.2f}. Precision against base rate.")
print()
print(f"{'base rate':>12}{'harmful/day':>14}{'alarms/day':>13}"
      f"{'precision':>12}{'false alarms/day':>19}")
print("-" * 70)
prec = {}
for b in (0.20, 0.05, 0.01, 0.003, 0.0005):
    p = precision(b)
    alarms = REQUESTS_PER_DAY * (b * TPR + (1 - b) * FPR)
    prec[b] = (p, alarms)
    print(f"{b:>12.2%}{REQUESTS_PER_DAY * b:>14,.0f}{alarms:>13,.0f}"
          f"{p:>12.1%}{alarms * (1 - p):>19,.0f}")

BASE = 0.003
print()
print(f"at a {BASE:.1%} base rate, {1 - prec[BASE][0]:.0%} of alarms are wrong")
print(f"and {prec[BASE][1] * (1 - prec[BASE][0]):,.0f} legitimate users are")
print("refused every day")

print()
print()
print("Sweeping the threshold. Both error types have a price.")
print()
HARM_COST = 4200.0
REFUSAL_COST = 26.0


def rates(t):
    """Higher threshold: fewer alarms, both rates fall."""
    tpr = 1.0 / (1.0 + math.exp((t - 0.30) / 0.16))
    fpr = 1.0 / (1.0 + math.exp((t - 0.02) / 0.09))
    return tpr, fpr


print(f"{'threshold':>11}{'TPR':>8}{'FPR':>9}{'precision':>12}"
      f"{'harm cost/day':>16}{'refusal cost/day':>19}{'total':>12}")
print("-" * 87)
sweep = {}
for t in (0.05, 0.15, 0.25, 0.35, 0.50, 0.70):
    tpr, fpr = rates(t)
    harm = REQUESTS_PER_DAY * BASE * (1 - tpr) * HARM_COST
    refuse = REQUESTS_PER_DAY * (1 - BASE) * fpr * REFUSAL_COST
    sweep[t] = (tpr, fpr, precision(BASE, tpr, fpr), harm, refuse, harm + refuse)
    print(f"{t:>11.2f}{tpr:>8.2f}{fpr:>9.3f}{precision(BASE, tpr, fpr):>12.1%}"
          f"{harm:>16,.0f}{refuse:>19,.0f}{harm + refuse:>12,.0f}")

best_t = min(sweep, key=lambda t: sweep[t][5])
print()
print(f"cost-optimal threshold: {best_t:.2f} at {sweep[best_t][5]:,.0f} a day")
print(f"maximum-sensitivity threshold ({0.05:.2f}) costs "
      f"{sweep[0.05][5]:,.0f}")

print()
print()
print("Volumes at the cost-optimal threshold: refusals against prevented harms.")
print()
print(f"{'base rate':>12}{'prevented harms/day':>22}{'refusals/day':>16}"
      f"{'refusals per prevented harm':>30}")
print("-" * 80)
tpr_o, fpr_o = rates(best_t)
vol = {}
for b in (0.20, 0.05, 0.01, 0.003, 0.0005):
    prevented = REQUESTS_PER_DAY * b * tpr_o
    refused = REQUESTS_PER_DAY * (1 - b) * fpr_o
    vol[b] = (prevented, refused, refused / prevented)
    print(f"{b:>12.2%}{prevented:>22,.0f}{refused:>16,.0f}"
          f"{refused / prevented:>30,.0f}")

print()
print(f"at {BASE:.1%} the guardrail refuses {vol[BASE][2]:,.0f} legitimate users")
print("for every harmful request it prevents")

print()
print()
print("Which threshold is right depends on a cost ratio nobody writes down.")
print()
print(f"{'harm : refusal cost':>21}{'best threshold':>17}{'TPR there':>12}"
      f"{'refusals/day':>15}{'prevented/day':>16}")
print("-" * 81)
ratio_tab = {}
for ratio in (2.0, 10.0, 40.0, 160.0, 640.0, 2560.0):
    best, bestc = None, None
    for th in [0.02 * i for i in range(1, 50)]:
        tpr, fpr = rates(th)
        c = (REQUESTS_PER_DAY * BASE * (1 - tpr) * ratio
             + REQUESTS_PER_DAY * (1 - BASE) * fpr * 1.0)
        if bestc is None or c < bestc:
            best, bestc = th, c
    tpr, fpr = rates(best)
    ratio_tab[ratio] = (best, tpr, fpr)
    print(f"{ratio:>18,.0f}:1{best:>17.2f}{tpr:>12.2f}"
          f"{REQUESTS_PER_DAY * (1 - BASE) * fpr:>15,.0f}"
          f"{REQUESTS_PER_DAY * BASE * tpr:>16,.0f}")

print()
print(f"the threshold moves from {ratio_tab[2.0][0]:.2f} to "
      f"{ratio_tab[2560.0][0]:.2f} across that range")
print("and the ratio is the number that is never stated")

print()
print()
print("And what an adaptive attacker does to the sensitivity term.")
print()
print(f"{'attempts':>10}{'P(all blocked)':>17}{'P(one gets through)':>22}"
      f"{'effective TPR':>16}")
print("-" * 65)
t_star = best_t
tpr_star, _ = rates(t_star)
adapt = {}
for k in (1, 3, 10, 30, 100):
    blocked = tpr_star ** k
    adapt[k] = (blocked, 1 - blocked, blocked)
    print(f"{k:>10}{blocked:>17.4f}{1 - blocked:>22.4f}{blocked:>16.4f}")

print()
print(f"at the cost-optimal threshold the guardrail's effective TPR against")
print(f"{30} attempts is {adapt[30][0]:.4f}")

print()
print()
print("So what does it buy? Three things, and only one of them is blocking.")
print()
BUYS = [
    ("blocks a one-shot attempt",   f"{tpr_star:.0%}",           "real, and single-shot"),
    ("blocks a 30-attempt attacker", f"{adapt[30][0]:.1%}",      "essentially nothing"),
    ("raises attempts per success", f"{1 / max(1 - tpr_star, 1e-9):.1f}x", "a rate limiter"),
    ("produces a logged decision",  "every request",             "forensics, liability"),
    ("removes casual misuse",       f"{tpr_star:.0%}",           "most traffic is casual"),
]
print(f"{'what a guardrail buys':>32}{'value':>16}{'reading':>28}")
print("-" * 76)
for name, val, reading in BUYS:
    print(f"{name:>32}{val:>16}{reading:>28}")

print()
print()
print("Refusal cost is not uniform. Where the false positives land.")
print()
SEGMENTS = [
    ("general consumer queries",   0.62, 1.0),
    ("security research",          0.04, 8.4),
    ("medical and clinical",       0.07, 6.1),
    ("legal and compliance",       0.05, 5.2),
    ("creative writing",           0.13, 3.7),
    ("non-English",                0.09, 4.9),
]
tpr_b, fpr_b = rates(best_t)
print(f"{'segment':>28}{'share of traffic':>19}{'relative FPR':>15}"
      f"{'refusals/day':>15}{'share of refusals':>20}")
print("-" * 97)
tot_ref = sum(REQUESTS_PER_DAY * sh * fpr_b * rel for n, sh, rel in SEGMENTS)
for name, sh, rel in SEGMENTS:
    r = REQUESTS_PER_DAY * sh * fpr_b * rel
    print(f"{name:>28}{sh:>19.0%}{rel:>15.1f}x{r:>14,.0f}"
          f"{r / tot_ref:>20.1%}")

print(f"""
The precision table is the arithmetic every guardrail runs into. At a {BASE:.1%} base rate --
generous, for a consumer product -- a guardrail with TPR {TPR:.2f} and FPR {FPR:.2f} produces
{prec[BASE][1]:,.0f} alarms a day of which **{1 - prec[BASE][0]:.0%} are wrong**
(eq:guardrail-precision-is-set-by-the-base-rate).

Nothing is broken. The false positives are drawn from a population
{(1 - BASE) / BASE:.0f} times larger than the true positives, so a
{FPR:.0%} false-positive rate outnumbers a {TPR:.0%} true-positive rate by a wide margin.
This is the same base-rate arithmetic that made accuracy useless in
ch:ev-classical-metrics, arriving in a setting where the consequence is a refused user rather
than a bad metric.

The threshold sweep prices both errors. At maximum sensitivity ({0.05:.2f}) the daily cost is
{sweep[0.05][5]:,.0f}, of which {sweep[0.05][4] / sweep[0.05][5]:.0%} is refusals; the
cost-optimal threshold is {best_t:.2f} at {sweep[best_t][5]:,.0f}.

**The default guardrail configuration is the maximum-sensitivity one**, because that is what
"catch as much as possible" means, and it is
{sweep[0.05][5] / sweep[best_t][5]:.1f} times the cost of the threshold that takes both errors
seriously.

The volume table is the number to sit with. At the cost-optimal threshold and a
{BASE:.1%} base rate, the guardrail prevents {vol[BASE][0]:,.0f} harmful requests a day and
refuses {vol[BASE][1]:,.0f} legitimate ones -- **{vol[BASE][2]:,.0f} refusals per prevented
harm** (eq:refusals-outnumber-prevented-harms). At {0.20:.0%} it is
{vol[0.20][2]:,.0f} to one.

Whether that trade is worth making is a question about the *ratio* of the two unit costs, and
the ratio table shows how much rides on it: the optimal threshold moves from
{ratio_tab[2.0][0]:.2f} at 2:1 to {ratio_tab[2560.0][0]:.2f} at 2,560:1, taking daily refusals
from a number in the tens of thousands to one in the hundreds.

**That ratio is never written down**, which is ch:ev-classical-metrics' finding arriving in a
setting where the cost of the assumed ratio is paid by users rather than by a metric.

The adaptive table is the harder problem and it comes from ch:sec-threat-model. At the
cost-optimal threshold the guardrail blocks {tpr_star:.0%} of a single attempt and
{adapt[30][0]:.1%} of a thirty-attempt attacker. The effective TPR against a determined
adversary is **not the number on the datasheet**, and no threshold choice fixes that -- moving
the threshold up trades refusals for a sensitivity that repetition erases anyway.

Which brings the question the chapter has to answer honestly. If a guardrail cannot stop a
determined attacker, what is it for?

The `buys` table is the answer and it has three real entries. It **removes casual misuse**,
which is most of the traffic that would otherwise reach the model -- people trying something
once because they wondered. It **raises attempts per success** to
{1 / max(1 - tpr_star, 1e-9):.1f}x, which is a rate limiter on the attacker's search and
composes with actual rate limits. And it **produces a logged decision on every request**,
which is the forensic and accountability artefact, and is the entry most often left out of
technical discussions and most often the reason the guardrail was funded.

What it does not do is bound anything. **A guardrail is a cost-raiser and a record, not a
boundary** -- exactly ch:sec-threat-model's classification, now with the numbers.

The segment table is the part to take to a product review, because refusal cost is not spread
evenly. `security research` is {SEGMENTS[1][1]:.0%} of traffic and
{SEGMENTS[1][2]:.1f} times more likely to trip the classifier; `medical and clinical` is
{SEGMENTS[2][1]:.0%} at {SEGMENTS[2][2]:.1f} times.

**The false positives land on the users with the most legitimate need for the capability**,
because proximity to a sensitive topic is what the classifier is measuring. That is not a
tuning problem and it does not improve with a better model -- it is what the feature is. The
remedy is segment-aware thresholds and an appeal path, and both are product work rather than
security work.""")

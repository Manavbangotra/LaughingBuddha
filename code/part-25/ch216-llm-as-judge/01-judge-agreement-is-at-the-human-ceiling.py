# -*- coding: utf-8 -*-
# Extracted from: Chapter 216 — LLM-as-a-Judge
# Source: src/.../ch216-llm-as-judge.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A judge agreeing with humans 80% of the time has matched the humans, not the truth.

cite:zheng2023judge reported that strong LLM judges reach over 80% agreement with human
preferences, which is usually read as "the judge is 80% right". It is not that. Two humans
agree at about the same rate, so the judge has reached the level of another annotator, and
ch:ev-why-hard's ceiling says no instrument validated this way can do better
(eq:judge-agreement-is-at-the-human-ceiling).

The same abstract lists position, verbosity and self-enhancement biases, and
cite:wang2023unfair measured the first: swapping the order of two candidates made a weaker
model beat a stronger one on 66 of 80 queries.

This listing computes what an order advantage is worth in quality-equivalent units, and
therefore which comparisons are decided by presentation rather than by content
(eq:position-advantage-decides-close-pairs).
"""
import math

HUMAN_HUMAN = 0.81            # two annotators on the same pair, from ch:ev-human
JUDGE_HUMAN = 0.81            # cite:zheng2023judge, "over 80%"
JUDGE_SELF = 0.88             # same judge, same pair, resampled
CHANCE = 0.50


def kappa(obs):
    return (obs - CHANCE) / (1.0 - CHANCE)


print("What an agreement rate actually says, once there is something to")
print("compare it against.")
print()
print(f"{'comparison':>28}{'agreement':>12}{'kappa':>9}"
      f"{'implied error':>16}{'reading':>26}")
print("-" * 91)
ROWS = [
    ("two humans",              HUMAN_HUMAN, "the ceiling"),
    ("judge vs human",          JUDGE_HUMAN, "at the ceiling"),
    ("judge vs itself",         JUDGE_SELF,  "more self-consistent"),
    ("coin flip",               CHANCE,      "the floor"),
]
agree = {}
for name, a, reading in ROWS:
    e = (1.0 - math.sqrt(max(0.0, 2.0 * a - 1.0))) / 2.0
    agree[name] = (a, kappa(a), e)
    print(f"{name:>28}{a:>12.0%}{kappa(a):>9.2f}{e:>16.3f}{reading:>26}")

print()
print("The judge is not 80% right. It is as close to a human as another")
print("human is, which is the strongest claim this design can support.")

print()
print()
print("Position advantage, in quality-equivalent units.")
print()
POS_ADV = 0.06                # first-shown candidate's bonus, on a 0-1 quality scale
NOISE = 0.13                  # judge decision noise, same units
GAPS = [0.02, 0.05, 0.10, 0.16, 0.25, 0.40]


def phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def p_pick_better(gap, first_is_better):
    """P(judge picks the genuinely better candidate)."""
    adv = POS_ADV if first_is_better else -POS_ADV
    return phi((gap + adv) / NOISE)


print(f"{'true quality gap':>18}{'better shown first':>21}"
      f"{'better shown second':>22}{'flips on swap':>16}")
print("-" * 77)
flip = {}
for g in GAPS:
    a = p_pick_better(g, True)
    b = p_pick_better(g, False)
    flip[g] = (a, b, a - b)
    print(f"{g:>18.2f}{a:>21.1%}{b:>22.1%}{a - b:>16.1%}")

print()
print(f"a candidate shown first carries a {POS_ADV:.2f} quality-equivalent bonus,")
print("so any pair closer than that is decided by the order")

print()
print()
print("How much of a real comparison set that covers.")
print()
# Realistic distribution of true quality gaps between two candidate models.
GAP_DIST = [(0.02, 0.24), (0.05, 0.21), (0.10, 0.19),
            (0.16, 0.14), (0.25, 0.13), (0.40, 0.09)]
print(f"{'true gap':>10}{'share of pairs':>17}{'decided by order?':>20}"
      f"{'flip rate':>12}{'weighted':>11}")
print("-" * 70)
tot_flip = 0.0
below = 0.0
for g, sh in GAP_DIST:
    f = flip[g][2]
    tot_flip += sh * f
    if g < POS_ADV:
        below += sh
    print(f"{g:>10.2f}{sh:>17.0%}{('yes' if g < POS_ADV else 'no'):>20}"
          f"{f:>12.1%}{sh * f:>11.3f}")
print("-" * 70)
print(f"{'TOTAL':>10}{1.0:>17.0%}{'':>20}{'':>12}{tot_flip:>11.3f}")
print()
print(f"{below:.0%} of pairs are closer than the position advantage;")
print(f"{tot_flip:.0%} of verdicts change when the order is swapped")

print()
print()
print("The fix, and what it costs: judge both orders and keep only the")
print("verdicts that survive.")
print()
print(f"{'protocol':>28}{'judgements':>13}{'decided':>11}"
      f"{'undecided':>12}{'accuracy on decided':>22}")
print("-" * 86)
ACC_SINGLE = sum(sh * (0.5 * (flip[g][0] + flip[g][1])) for g, sh in GAP_DIST)
# Both orders: decided when the two runs agree.
dec, corr = 0.0, 0.0
for g, sh in GAP_DIST:
    a, b = flip[g][0], flip[g][1]
    both_right = a * b
    both_wrong = (1 - a) * (1 - b)
    dec += sh * (both_right + both_wrong)
    corr += sh * both_right
print(f"{'single order':>28}{1:>13}{1.0:>11.0%}{0.0:>12.0%}"
      f"{ACC_SINGLE:>22.1%}")
print(f"{'both orders, agree required':>28}{2:>13}{dec:>11.0%}"
      f"{1 - dec:>12.0%}{corr / dec:>22.1%}")
print(f"{'both orders, tie broken by coin':>28}{2:>13}{1.0:>11.0%}"
      f"{0.0:>12.0%}{(corr + 0.5 * (dec - corr) + 0.5 * (1 - dec)):>22.1%}")

print()
print(f"balancing order raises accuracy on decided pairs from "
      f"{ACC_SINGLE:.1%} to {corr / dec:.1%}")
print(f"and honestly refuses to decide {1 - dec:.0%} of them")

print()
print()
print("Verbosity, the second bias, priced the same way.")
print()
LEN_ADV_PER_50PCT = 0.05      # quality-equivalent bonus per 50% more output
print(f"{'length vs baseline':>20}{'quality-equiv bonus':>22}"
      f"{'win rate at gap 0':>20}{'wins a 0.10 deficit?':>23}")
print("-" * 85)
verb = {}
for mult in (1.0, 1.25, 1.5, 2.0, 3.0):
    bonus = LEN_ADV_PER_50PCT * math.log(mult) / math.log(1.5)
    w0 = phi(bonus / NOISE)
    beats = phi((bonus - 0.10) / NOISE)
    verb[mult] = (bonus, w0, beats)
    print(f"{mult:>19.2f}x{bonus:>22.3f}{w0:>20.1%}{beats:>23.1%}")

print()
print()
print("And what that does over rounds of selecting variants against the judge.")
print()
print(f"{'round':>7}{'length mult':>14}{'judge score':>14}"
      f"{'true quality':>15}{'divergence':>13}")
print("-" * 63)
length = 1.0
true_q = 0.640
drift = {}
for r in range(0, 6):
    bonus = LEN_ADV_PER_50PCT * math.log(length) / math.log(1.5)
    judged = true_q + bonus
    drift[r] = (length, judged, true_q, judged - true_q)
    print(f"{r:>7}{length:>13.2f}x{judged:>14.3f}"
          f"{true_q:>15.3f}{judged - true_q:>13.3f}")
    length *= 1.18
    true_q += 0.004          # real progress, small
print(f"""
The agreement table is the correction most needed and it is the smallest table here. Two
humans agree {HUMAN_HUMAN:.0%} of the time on these pairs; the judge agrees with a human
{JUDGE_HUMAN:.0%} of the time (eq:judge-agreement-is-at-the-human-ceiling).

**Those are the same number**, and cite:zheng2023judge's result should be read as "the judge
performs like another annotator" rather than "the judge is right four times in five." The
second reading is a claim about truth; the data supports only a claim about concordance.

Notice also that the judge is *more consistent with itself* ({JUDGE_SELF:.0%}) than it is
with a human. That gap is where the biases live: a systematic preference is perfectly
self-consistent and reduces agreement with people who do not share it.

The position table converts cite:wang2023unfair's finding into something you can plan with.
An order advantage of {POS_ADV:.2f} on a quality scale means the judge picks the genuinely
better candidate {flip[0.05][0]:.0%} of the time when the better one is shown first, and
{flip[0.05][1]:.0%} of the time when it is shown second, on a pair whose true gap is
{0.05:.2f}.

**That is a {flip[0.05][2]:.0%} swing produced by the order of two items in a prompt**, on
a comparison that has nothing to do with order.

The coverage table says how much of a real comparison set that governs.
{below:.0%} of candidate pairs differ by less than the position advantage, and
{tot_flip:.0%} of all verdicts flip when the order is swapped. cite:wang2023unfair reported
66 of 80 queries flippable, which is a stronger result on a set chosen to demonstrate the
effect; the arithmetic here says a garden-variety comparison set is around
{tot_flip:.0%} exposed.

The protocol table is the fix and it is cheap. Judging both orders and keeping only verdicts
that survive the swap takes accuracy on decided pairs from {ACC_SINGLE:.1%} to
{corr / dec:.1%}, at exactly twice the judging cost.

The important column is the fourth one. **The protocol declines to decide
{1 - dec:.0%} of pairs**, and that refusal is the feature rather than the cost. Those are
the pairs where the position advantage exceeds the quality difference, which is to say the
pairs where the judge has no information -- and a single-order run answers them anyway,
confidently, at slightly better than chance.

The last row shows what happens if you break the ties with a coin: overall accuracy
{(corr + 0.5 * (dec - corr) + 0.5 * (1 - dec)):.1%}, barely above the single-order number.
**The gain is not in the extra judgement, it is in knowing which verdicts to discard.**

The verbosity table prices the second listed bias the same way. Output
{2.0:.1f} times longer carries a {verb[2.0][0]:.3f} quality-equivalent bonus, enough to win
{verb[2.0][1]:.0%} of ties and to overturn a {0.10:.2f} quality deficit
{verb[2.0][2]:.0%} of the time.

And the drift table is why that matters more than it looks. Selecting variants against this
judge for five rounds -- an entirely ordinary development loop -- takes output length to
{drift[5][0]:.2f} times baseline, the judge's score from {drift[0][1]:.3f} to
{drift[5][1]:.3f}, and true quality from {drift[0][2]:.3f} to {drift[5][2]:.3f}.

The measured improvement is {drift[5][1] - drift[0][1]:.3f} and the real one is
{drift[5][2] - drift[0][2]:.3f}. **Most of the reported gain is length.**

Nothing in that loop looks wrong from inside it. Each round genuinely improves the judge's
score, each variant is genuinely selected on merit as the judge sees merit, and the drift is
visible only against a measurement the loop does not contain. Which is
ch:ev-llm-judge's second listing.""")

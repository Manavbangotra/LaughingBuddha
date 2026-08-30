# -*- coding: utf-8 -*-
# Extracted from: Chapter 190 — Model Routing, Caching, and Cost Optimization
# Source: src/.../ch190-routing-caching.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A cascade is a bet on a judge, not a bet on a cheap model.

Cascade routing sends every request to a cheap model first, and escalates to an
expensive one only when the cheap answer looks insufficient. cite:chen2023frugalgpt
reports large savings from this. But the escalation decision is made by something,
and that something is a verifier -- so the cascade inherits every property of the
verifier problem from ch:ag-recovery.

This listing measures what the judge's quality does to the cascade's economics
(eq:cascade-is-a-verifier-bet). The question is not "does a cascade save money" but
"how good does the judge have to be before it saves anything at all".
"""
# Cheap model: right on the easy majority, wrong on the hard tail.
# Expensive model: better everywhere, at 20x the price.
A_CHEAP = 0.72
A_EXP = 0.91
C_CHEAP = 1.0
C_EXP = 20.0


def cascade(judge_tpr, judge_fpr):
    """Every request goes to the cheap model. The judge inspects the cheap answer
    and escalates when it thinks the answer is wrong.

    judge_tpr: P(escalate | cheap answer IS wrong)   -- catching real errors
    judge_fpr: P(escalate | cheap answer is RIGHT)   -- escalating needlessly

    Returns (accuracy, cost, escalation rate).
    """
    wrong = 1 - A_CHEAP
    right = A_CHEAP

    # Four outcomes, by whether the cheap answer was right and whether we escalate.
    esc_from_wrong = wrong * judge_tpr
    esc_from_right = right * judge_fpr
    keep_wrong = wrong * (1 - judge_tpr)
    keep_right = right * (1 - judge_fpr)

    escalated = esc_from_wrong + esc_from_right
    # An escalated request is answered by the expensive model at its own accuracy.
    acc = keep_right + escalated * A_EXP
    cost = C_CHEAP + escalated * C_EXP
    return acc, cost, escalated


print("A two-model cascade. The cheap model is right %.0f%% of the time and costs"
      % (A_CHEAP * 100))
print("%.0f; the expensive model is right %.0f%% and costs %.0f."
      % (C_CHEAP, A_EXP * 100, C_EXP))
print()
print("Baselines, with no cascade at all:")
print()
print(f"{'policy':>22}{'accuracy':>11}{'cost':>9}")
print("-" * 42)
print(f"{'always cheap':>22}{A_CHEAP:>11.1%}{C_CHEAP:>9.2f}")
print(f"{'always expensive':>22}{A_EXP:>11.1%}{C_EXP:>9.2f}")

print()
print()
print("Now the cascade, as the judge gets better at spotting a wrong cheap answer.")
print("A perfect judge escalates every error and nothing else.")
print()
print(f"{'judge recall':>14}{'judge FPR':>12}{'accuracy':>11}{'cost':>9}"
      f"{'escalated':>12}{'vs always-exp':>15}")
print("-" * 73)
rows = {}
for tpr, fpr in ((1.00, 0.00), (0.90, 0.05), (0.75, 0.10),
                 (0.60, 0.20), (0.40, 0.30), (0.20, 0.45)):
    acc, cost, esc = cascade(tpr, fpr)
    rows[tpr] = (acc, cost, esc)
    # Positive means the cascade is cheaper AND at least as accurate.
    verdict = ("saves %.0f%%" % ((1 - cost / C_EXP) * 100)
               if acc >= A_EXP - 0.005 else "loses %.1f pts" % ((A_EXP - acc) * 100))
    print(f"{tpr:>14.0%}{fpr:>12.0%}{acc:>11.1%}{cost:>9.2f}{esc:>12.1%}"
          f"{verdict:>15}")

print()
print()
print("The accuracy the cascade gives up, and the money it saves, side by side.")
print("A cascade is only worth it if the first column is small enough to buy the")
print("second.")
print()
print(f"{'judge recall':>14}{'accuracy given up':>19}{'cost saved':>12}"
      f"{'pts per 1x saved':>18}")
print("-" * 63)
eff = {}
for tpr in sorted(rows, reverse=True):
    acc, cost, esc = rows[tpr]
    lost = A_EXP - acc
    saved = C_EXP - cost
    eff[tpr] = lost / saved if saved > 0 else float("inf")
    print(f"{tpr:>14.0%}{lost:>19.1%}{saved:>12.2f}{lost / saved:>18.4f}")

print()
print()
print("The threshold. Below some judge recall, the cascade is dominated: you could")
print("buy the same accuracy more cheaply by sending a random SHARE of traffic to")
print("the expensive model and skipping the judge entirely.")
print()
print(f"{'judge recall':>14}{'cascade acc':>13}{'cascade cost':>14}"
      f"{'random-split cost':>19}{'verdict':>14}")
print("-" * 74)


def random_split_cost(target_acc):
    """Cheapest way to hit an accuracy with no judge: send share p to expensive."""
    # acc = (1-p)*A_CHEAP + p*A_EXP  ->  p = (acc - A_CHEAP) / (A_EXP - A_CHEAP)
    p = (target_acc - A_CHEAP) / (A_EXP - A_CHEAP)
    p = max(0.0, min(1.0, p))
    return (1 - p) * C_CHEAP + p * C_EXP


beat = {}
for tpr in sorted(rows, reverse=True):
    acc, cost, esc = rows[tpr]
    rc = random_split_cost(acc)
    beat[tpr] = cost < rc
    print(f"{tpr:>14.0%}{acc:>13.1%}{cost:>14.2f}{rc:>19.2f}"
          f"{('cascade wins' if cost < rc else 'random wins'):>14}")

worst_win = min([t for t in beat if beat[t]], default=None)

print()
print()
print("The break-even: the judge recall at which the cascade first matches the")
print("expensive model's accuracy. Below it the cascade is a cost decision that")
print("costs accuracy; above it the cascade dominates on both axes.")
print()
print(f"{'judge recall':>14}{'judge FPR':>12}{'accuracy':>11}{'cost':>9}"
      f"{'vs always-exp':>16}")
print("-" * 62)
be = None
for i in range(101):
    tpr = i / 100.0
    fpr = tpr * 0.10          # a judge that catches more also over-escalates more
    acc, cost, esc = cascade(tpr, fpr)
    if be is None and acc >= A_EXP:
        be = (tpr, fpr, acc, cost)
for tpr in (0.50, 0.65, 0.80, 0.95):
    acc, cost, esc = cascade(tpr, tpr * 0.10)
    print(f"{tpr:>14.0%}{tpr * 0.10:>12.0%}{acc:>11.1%}{cost:>9.2f}"
          f"{(acc - A_EXP):>+16.1%}")
print()
print(f"break-even judge recall: {be[0]:.0%} (FPR {be[1]:.0%}), "
      f"accuracy {be[2]:.1%} at cost {be[3]:.2f}")
print(f"at break-even the cascade costs {be[3] / C_EXP:.0%} of always-expensive")

print(f"""
The two baselines set the range. Always-cheap costs {C_CHEAP:.2f} and is right
{A_CHEAP:.0%} of the time; always-expensive costs {C_EXP:.0f} and is right
{A_EXP:.0%}. Twenty times the money buys {(A_EXP - A_CHEAP) * 100:.0f} points.

Now the finding, which is not the one the cost-saving literature leads with. With a
**perfect** judge the cascade is right {rows[1.0][0]:.1%} of the time -- better than
the expensive model alone -- at a cost of {rows[1.0][1]:.2f}, which is
{rows[1.0][1] / C_EXP:.0%} of always-expensive.

That is not a cheaper way to get the expensive model's answer. It is a **better**
answer for a third of the price, because a perfect judge turns the pair into an
oracle: keep the cheap model's correct answers, escalate exactly its errors.
cite:chen2023frugalgpt's "+4% accuracy at the same cost" is this effect.

But the accuracy column collapses as the judge degrades. At {0.75:.0%} recall the
cascade is {rows[0.75][0]:.1%} -- it has given back the entire oracle bonus and is
now {(A_EXP - rows[0.75][0]) * 100:.1f} points BELOW always-expensive. At {0.40:.0%} recall
it is {rows[0.4][0]:.1%}, and at {0.20:.0%} recall it is {rows[0.2][0]:.1%} --
barely above the cheap model it started from, at {rows[0.2][1]:.2f} times the
cheap model's cost.

**The cascade's entire value lives in the judge** (eq:cascade-is-a-verifier-bet).
The cheap model's accuracy sets the ceiling; the judge decides how much of the gap
to the expensive model you actually capture, and it can capture more than all of it
or almost none.

The break-even table makes the operating requirement explicit. Coupling the false
positive rate to recall at a tenth -- a judge that catches more errors also
escalates more good answers -- the cascade first matches always-expensive at
**{be[0]:.0%} recall**, where it costs {be[3]:.2f}, or {be[3] / C_EXP:.0%} of the
expensive baseline.

So the deployment question has a number attached: **can you build an escalation
judge with better than {be[0]:.0%} recall on your own traffic?** If yes, the cascade
is close to free money. If no, you are paying for a judge to make you worse.

The last table is the check that stops a bad cascade from looking good. A cascade
that loses accuracy can always be compared against the trivial policy of sending a
random share of traffic to the expensive model -- no judge, no infrastructure. The
cascade beats that policy down to {0.40:.0%} recall, and loses at {0.20:.0%}, where
random-split buys the same {rows[0.2][0]:.1%} accuracy for {random_split_cost(rows[0.2][0]):.2f}
against the cascade's {rows[0.2][1]:.2f}.

**A cascade whose judge is worse than a coin flip is beaten by a coin flip**, and it
is beaten while costing more to build and more to run. That comparison belongs in
every routing design review, because it is cheap to compute and it is the one thing
that catches a cascade being justified by its cost column alone.""")

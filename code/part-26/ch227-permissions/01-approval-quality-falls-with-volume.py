# -*- coding: utf-8 -*-
# Extracted from: Chapter 227 — Permission Systems, Approval Flows, and Governance
# Source: src/.../ch227-permissions.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""An approval queue where almost everything is fine trains the reviewer to approve.

Human approval is the strongest control in ch:sec-tool-abuse's table and the one most likely
to be quietly disabled. It is not disabled by a decision; it degrades, because the quantity
that determines how carefully a reviewer reads is how often reading carefully changed the
outcome.

Two mechanisms. Volume: a fixed time budget spread over more items gives less time per item,
so scrutiny falls (eq:approval-quality-falls-with-volume). And base rate: a queue where almost
nothing is rejected provides no reinforcement for care
(eq:a-low-rejection-rate-trains-approval).

A wider gate covers more of the bad actions and destroys the reviewer's ability to see any of
them, so the product has an interior maximum -- and it is far below what a per-call gate
generates.
"""
import math

REVIEW_MINUTES_PER_DAY = 6.0 * 60.0
BAD_PER_DAY = 8.3              # genuinely bad actions attempted per day


def scrutiny(volume):
    """Minutes per item, and the detection that supports."""
    mins = REVIEW_MINUTES_PER_DAY / volume
    return mins, 1.0 - math.exp(-mins / 1.4)


def habituation(reject_rate):
    """Care multiplier: a reviewer who never rejects stops looking."""
    return 0.18 + 0.82 * (1.0 - math.exp(-reject_rate / 0.06))


def coverage(volume):
    """A wider gate intercepts more of the bad actions."""
    return 1.0 - math.exp(-volume / 330.0)


print(f"{REVIEW_MINUTES_PER_DAY:.0f} reviewer-minutes a day, "
      f"{BAD_PER_DAY:.1f} bad actions attempted.")
print()
print(f"{'approvals/day':>15}{'coverage':>10}{'min/item':>11}{'scrutiny':>10}"
      f"{'reject rate':>13}{'habituation':>13}{'catch':>8}{'bad caught':>12}")
print("-" * 92)
tab = {}
for v in (20, 60, 200, 600, 2000, 6000, 20640):
    mins, d = scrutiny(v)
    cov = coverage(v)
    rej = BAD_PER_DAY * cov / v
    h = habituation(rej)
    eff = d * h
    caught = BAD_PER_DAY * cov * eff
    tab[v] = (cov, mins, d, rej, h, eff, caught)
    print(f"{v:>15,}{cov:>10.3f}{mins:>11.2f}{d:>10.3f}{rej:>13.2%}"
          f"{h:>13.3f}{eff:>8.3f}{caught:>12.2f}")

best_v = max(tab, key=lambda v: tab[v][6])
print()
print(f"maximum bad actions caught: {tab[best_v][6]:.2f} a day at "
      f"{best_v:,} approvals")
print(f"at {20640:,} approvals: {tab[20640][6]:.2f} caught, "
      f"{tab[20640][2] * 60:.1f} seconds an item")

print()
print()
print("The two terms separately, at a fixed volume.")
print()
V = 410
mins_v, d_v = scrutiny(V)
print(f"At {V} approvals a day: {mins_v:.2f} minutes an item, "
      f"scrutiny {d_v:.3f}.")
print()
print(f"{'rejection rate':>16}{'habituation':>14}{'effective catch':>18}"
      f"{'vs 0.05%':>11}")
print("-" * 59)
rej_tab = {}
for r in (0.0005, 0.004, 0.02, 0.08, 0.25):
    h = habituation(r)
    rej_tab[r] = (h, d_v * h)
    print(f"{r:>16.2%}{h:>14.3f}{d_v * h:>18.3f}"
          f"{(d_v * h) / (d_v * habituation(0.0005)):>10.1f}x")

print()
print("Same reviewer, same items, same time budget. Only the density changed.")

print()
print()
print("Queue designs from ch:sec-tool-abuse, scored on both terms.")
print()
QUEUES = [
    ("approve every tool call",           20640, 0.99),
    ("approve non-reversible calls",       5250, 0.91),
    ("approve on a taint path only",        190, 0.44),
    ("approve by outcome class",            410, 0.62),
    ("approve outcome class, high-risk",     84, 0.28),
]
print(f"{'queue design':>36}{'items/day':>12}{'coverage':>11}"
      f"{'reject rate':>14}{'catch':>8}{'bad caught/day':>17}")
print("-" * 98)
q = {}
for name, v, cov in QUEUES:
    mins, d = scrutiny(v)
    rej = BAD_PER_DAY * cov / v
    h = habituation(rej)
    eff = d * h
    q[name] = (v, cov, rej, eff, BAD_PER_DAY * cov * eff)
    print(f"{name:>36}{v:>12,}{cov:>11.2f}{rej:>14.2%}{eff:>8.3f}"
          f"{BAD_PER_DAY * cov * eff:>17.2f}")

best_q = max(q, key=lambda n: q[n][4])
print()
print(f"best: {best_q} at {q[best_q][4]:.2f} bad actions caught a day")
print(f"worst: approve every tool call at "
      f"{q['approve every tool call'][4]:.2f}")

print()
print()
print("The alternative nobody proposes: sample deeply instead of reviewing all.")
print()
TOTAL = 20640
print(f"{'policy':>34}{'reviewed':>11}{'min/item':>11}{'catch':>8}"
      f"{'bad-item coverage':>20}{'bad caught/day':>17}")
print("-" * 101)
SAMPLES = [
    ("review everything",         1.000, 0.99),
    ("review 10%, uniform",       0.100, 0.099),
    ("review 2%, uniform",        0.020, 0.020),
    ("review 2%, risk-weighted",  0.020, 0.310),
    ("review 0.5%, risk-weighted", 0.005, 0.140),
]
for name, share, badcov in SAMPLES:
    n = TOTAL * share
    mins, d = scrutiny(n)
    rej = BAD_PER_DAY * badcov / n
    h = habituation(rej)
    eff = d * h
    print(f"{name:>34}{n:>11,.0f}{mins:>11.2f}{eff:>8.3f}"
          f"{badcov:>20.1%}{BAD_PER_DAY * badcov * eff:>17.2f}")

print()
print("Risk-weighted sampling reads 2% of items and reaches 31% of the bad")
print("ones, because the sample is not uniform.")

print()
print()
print("And what a rubber-stamped approval still produces.")
print()
ARTEFACTS = [
    ("a decision record",           "yes", "yes", "compliance, forensics"),
    ("a human in the causal chain", "yes", "yes", "accountability"),
    ("a delay before the action",   "yes", "yes", "a cancellation window"),
    ("an actual review",            "yes", "no",  "the control itself"),
    ("a rejection when warranted",  "yes", "no",  "the control itself"),
]
print(f"{'what approval produces':>32}{'careful review':>17}"
      f"{'rubber stamp':>15}{'why it matters':>25}")
print("-" * 89)
for name, careful, stamp, why in ARTEFACTS:
    print(f"{name:>32}{careful:>17}{stamp:>15}{why:>25}")

print(f"""
The main table is both mechanisms at once and the last column is the finding. A gate
generating {20:,} approvals a day gives {tab[20][1]:.1f} minutes an item and catches
{tab[20][6]:.2f} bad actions -- limited by coverage, since it only intercepts
{tab[20][0]:.1%} of them. A gate generating {20640:,} covers {tab[20640][0]:.1%} and catches
{tab[20640][6]:.2f}, because each item gets {tab[20640][2] * 60:.1f} seconds
(eq:approval-quality-falls-with-volume).

**The maximum is {tab[best_v][6]:.2f} bad actions a day at {best_v:,} approvals**, and both
ends of the table are far worse than the middle. Widening a gate buys coverage and spends
scrutiny, and past a point the second term dominates.

The rejection table isolates the other mechanism, holding volume and time fixed. Same
reviewer, same {mins_v:.2f} minutes an item, same items. At a {0.0005:.2%} rejection rate the
care multiplier is {rej_tab[0.0005][0]:.3f}; at {0.08:.0%} it is {rej_tab[0.08][0]:.3f} --
**{rej_tab[0.08][1] / rej_tab[0.0005][1]:.1f} times the effective catch from density alone**
(eq:a-low-rejection-rate-trains-approval).

That is not a criticism of reviewers. It is what a reinforcement schedule does. A queue in
which four thousand consecutive items were fine has taught, correctly, that the next one
probably is, and no amount of training-day emphasis survives that gradient.

The queue table applies both to ch:sec-tool-abuse's designs. `approve every tool call` has the
best coverage on the list at {q['approve every tool call'][1]:.2f} and catches
{q['approve every tool call'][4]:.2f} bad actions a day. `{best_q}` covers
{q[best_q][1]:.2f} and catches {q[best_q][4]:.2f} --
**{q[best_q][4] / q['approve every tool call'][4]:.0f} times more, from
{q['approve every tool call'][0] / q[best_q][0]:.0f} times fewer approvals.**

Which is the same recommendation ch:sec-tool-abuse reached from the composition side, arriving
here from the human side. It was recommended there because a per-call gate cannot express a
composition; it is recommended here because a per-call gate cannot be read.

The sampling table is the design that sounds like giving up and is not. Reviewing
{0.02:.0%} of items risk-weighted gives {scrutiny(TOTAL * 0.02)[0]:.2f} minutes each, reaches
{0.31:.0%} of the bad actions, and catches more than reviewing everything at
{scrutiny(TOTAL)[0] * 60:.1f} seconds an item.

**A deep review of a biased sample beats a shallow review of everything**, which is
ch:ops-observability's sampling result in the approval queue: uniform coverage of a rare event
is the expensive way to see nothing.

The last table is the honest accounting of a rubber stamp, because the answer is not nothing.
It still produces a decision record, still puts a human in the causal chain, and still imposes
a delay during which the action can be cancelled. Two of those are why the control was funded
and all three survive habituation.

What does not survive is the review and the rejection -- **the control itself**. Which is worth
writing down in the design document, because a queue producing the first three and not the last
two is an accountability mechanism carrying a security mechanism's name, and that confusion is
what prevents anybody from fixing the volume.""")

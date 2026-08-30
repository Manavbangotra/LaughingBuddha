# -*- coding: utf-8 -*-
# Extracted from: Chapter 188 — AI-Assisted versus Autonomous Software Engineering
# Source: src/.../ch188-autonomy.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The autonomy spectrum for software, which is ch:aids-oversight's question in a
second domain.

cite:testini2025dsautomation found data science automation evaluated only at the
extremes -- pure assistance or full autonomy -- with the middle neglected. Software
has the same gap and a sharper way to resolve it, because ch:aise-cicd established
that activities differ enormously in two properties that decide what an agent can
safely do.

This listing crosses six operating modes against four task types and asks which mode
wins where (eq:no-mode-dominates). The answer is that none dominates, and the right
mode is a function of the task's verifiability and reversibility rather than of the
team's appetite for automation.

cite:becker2025devproductivity's measured 19% slowdown is one cell of this table:
experienced developers, mature familiar code, agent-with-review.
"""
# (task type, verifiability, reversibility, human hours unassisted,
#  hours lost when an error reaches the codebase)
# The last column spans two orders of magnitude, which ch:aise-cicd measured
# and which is the reason a uniform policy cannot be right.
TASKS = [
    ("fix a reported bug",   0.92, 0.95,  3.0,   12.0),
    ("implement a feature",  0.58, 0.78,  9.0,   40.0),
    ("refactor legacy code", 0.40, 0.88,  6.0,   55.0),
    ("design a data model",  0.24, 0.19,  5.0,  620.0),
]

# (mode, human hours multiplier, agent share of the work, review depth)
MODES = [
    ("manual",                    1.00, 0.00, 0.00),
    ("completion",                0.88, 0.15, 0.90),
    ("chat-assisted",             0.74, 0.35, 0.85),
    ("agent with full review",    0.46, 0.80, 0.75),
    ("agent with gated review",   0.22, 0.90, 0.35),
    ("fully autonomous",          0.03, 1.00, 0.00),
]

P_HUMAN_WRONG = 0.15
P_AGENT_WRONG = 0.31


def outcome(task, mode):
    """Returns (human hours, defect rate, total hours including rework)."""
    _, ver, rev, base_h, err_h = task
    _, h_mult, agent_share, review = mode
    hours = base_h * h_mult
    p_wrong = P_HUMAN_WRONG * (1 - agent_share) + P_AGENT_WRONG * agent_share
    # Two chances to catch it: the automated verifier, then the human review.
    escaped = p_wrong * (1 - ver) * (1 - review * 0.72)
    # An escaped error costs less where it can be undone.
    rework = escaped * err_h * (1 - rev * 0.80)
    return hours, escaped, hours + rework


print("Six operating modes against four task types. 'Total' includes rework")
print("from errors that reach the codebase.")
print()
print(f"{'mode':>26}" + "".join(f"{t[0][:14]:>16}" for t in TASKS))
print("-" * 90)
tab = {}
for m in MODES:
    row = tuple(outcome(t, m)[2] for t in TASKS)
    tab[m[0]] = row
    print(f"{m[0]:>26}" + "".join(f"{v:>16.2f}" for v in row))

print()
print()
print("The best mode for each task, and what the second best costs.")
print()
print(f"{'task':>24}{'best mode':>26}{'total hours':>13}{'2nd best':>26}")
print("-" * 89)
best = {}
for i, t in enumerate(TASKS):
    ranked = sorted(MODES, key=lambda m: tab[m[0]][i])
    best[t[0]] = (ranked[0][0], tab[ranked[0][0]][i], ranked[1][0])
    print(f"{t[0]:>24}{ranked[0][0]:>26}{tab[ranked[0][0]][i]:>13.2f}"
          f"{ranked[1][0]:>26}")

print()
print()
print("No mode wins everywhere, and the ordering tracks verifiability times")
print("reversibility rather than anything about the mode.")
print()
print(f"{'task':>24}{'verify x reverse':>18}{'best mode':>26}")
print("-" * 68)
for t in TASKS:
    print(f"{t[0]:>24}{t[1] * t[2]:>18.2f}{best[t[0]][0]:>26}")

print()
print()
print("Human hours, separately -- because the mode that minimises total hours")
print("and the mode that minimises HUMAN hours are not the same.")
print()
print(f"{'task':>24}{'min total':>26}{'min human':>26}")
print("-" * 78)
for i, t in enumerate(TASKS):
    by_total = min(MODES, key=lambda m: outcome(t, m)[2])
    by_human = min(MODES, key=lambda m: outcome(t, m)[0])
    print(f"{t[0]:>24}{by_total[0]:>26}{by_human[0]:>26}")

print()
print()
print("Applying one mode everywhere, which is what a policy actually does.")
print()
print(f"{'uniform policy':>26}{'total hours':>14}{'human hours':>14}"
      f"{'vs best-per-task':>18}")
print("-" * 74)
opt = sum(min(outcome(t, m)[2] for m in MODES) for t in TASKS)
uni = {}
for m in MODES:
    tot = sum(outcome(t, m)[2] for t in TASKS)
    hum = sum(outcome(t, m)[0] for t in TASKS)
    uni[m[0]] = (tot, hum)
    print(f"{m[0]:>26}{tot:>14.2f}{hum:>14.2f}{tot - opt:>+18.2f}")

print()
print(f"   best mode chosen per task: {opt:.2f} total hours")
print(f"   best single uniform policy: "
      f"{min(uni.values(), key=lambda x: x[0])[0]:.2f}")

print()
print()
print("And the question the measured evidence actually poses: how much can a")
print("mode add to the surrounding work before it stops paying? This sweeps")
print("the human-hours multiplier for agent-with-review on a feature task.")
print()
print(f"{'hours multiplier':>18}{'total':>10}{'vs manual':>12}{'verdict':>10}")
print("-" * 50)
feature = TASKS[1]
man = outcome(feature, MODES[0])[2]
bk = {}
for h in (0.46, 0.75, 1.00, 1.25, 1.50):
    m = ("agent with full review", h, 0.80, 0.75)
    tot = outcome(feature, m)[2]
    bk[h] = (tot, tot / man - 1)
    print(f"{h:>18.2f}{tot:>10.2f}{tot / man - 1:>+12.0%}"
          f"{('faster' if tot < man else 'slower'):>10}")

lo, hi = 0.1, 3.0
for _ in range(60):
    mid = (lo + hi) / 2
    if outcome(feature, ("x", mid, 0.80, 0.75))[2] < man:
        lo = mid
    else:
        hi = mid
breakeven = (lo + hi) / 2
print()
print(f"   break-even multiplier: {breakeven:.2f}")
print(f"   The mode pays if it leaves the human doing less than "
      f"{breakeven:.0%} of the")
print(f"   unassisted hours. cite:becker2025devproductivity measured a setting")
print(f"   where it did not -- which locates that study above this threshold")
print(f"   rather than contradicting the model.")

print(f"""
The first table has no winning row, and that is the finding
(eq:no-mode-dominates).

Full autonomy is best on three of the four tasks and worst on the fourth by a
margin that swamps its wins: {tab['fully autonomous'][3]:.1f} hours on data-model
design against {tab['completion'][3]:.1f} for completion. The single worst cell in
the table and three of the four best belong to the same mode.

The reason is in the verify-times-reverse column. Bug fixing sits at
{0.92 * 0.95:.2f} and design at {0.24 * 0.19:.2f} -- a factor of nineteen -- and
ch:aise-cicd showed that product entering multiplicatively.

The uniform-policy table is what a real team's decision looks like, because
organisations adopt a stance rather than a per-task rule. **Every uniform policy is
substantially worse than choosing per task**: the best single stance costs
{min(uni.values(), key=lambda x: x[0])[0] - opt:+.1f} hours against per-task
selection, and full autonomy applied uniformly costs
{uni['fully autonomous'][0] - opt:+.1f} -- **worse than doing everything manually.**

That last comparison is the one worth carrying out of this chapter. A team that
adopts autonomy as a policy, rather than as a per-task decision, can end up worse
than a team that adopted nothing, and it will get there while every individual
automation looks like a success.

The human-hours table separates two objectives that are usually conflated. On
three tasks the mode minimising total hours and the mode minimising HUMAN hours
agree. On data-model design they do not: full autonomy minimises human hours and
completion minimises total.

**Minimising human involvement and minimising cost are different objectives**, and
they diverge exactly where the consequences are largest -- which is the worst place
for an organisation to be optimising the wrong one.

The last table addresses the measured evidence directly. Agent-with-review pays as
long as it leaves the human doing less than {breakeven:.0%} of the unassisted hours,
because the review's defect reduction covers a small increase in effort.
cite:becker2025devproductivity measured a setting where hours ROSE {19}%, which is
above that threshold.

So the study and this model agree rather than conflict: **the mode pays until it
adds work, and the study found a setting where it added work.** Which relocates the
question from "do these tools help" to "in which settings does the assistance
exceed the friction" -- and ch:aise-generation's setting table says the answer moves
across the plausible range.""")

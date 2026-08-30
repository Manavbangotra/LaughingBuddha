# -*- coding: utf-8 -*-
# Extracted from: Chapter 232 — Human Oversight in Practice
# Source: src/.../ch232-oversight.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Oversight needs authority, information, time and incentive. Missing one makes it ceremony.

The first listing asked whether a reviewer improves accuracy. This one asks what has to be true
for them to be able to.

Four things: the authority to change the outcome, information sufficient to judge it, time
enough to use that information, and an incentive that does not punish them for saying no. Those
compose as a product, not a sum -- an arrangement with three of the four produces a decision
record and no decision (eq:oversight-is-a-conjunction-of-preconditions).

The fourth is the one nobody designs. The person who rejects absorbs a delay, an escalation and
a chance of being visibly wrong; the person who approves absorbs nothing observable. The
organisation's costs run the other way
(eq:reviewers-bear-the-cost-of-rejecting-not-approving).
"""
# (arrangement, authority, information, time, incentive, share of decisions touched)
ARRANGEMENTS = [
    ("release sign-off by the owner",      0.90, 0.55, 0.70, 0.60, 1.000),
    ("reviewer approves each agent action", 0.85, 0.30, 0.20, 0.35, 1.000),
    ("a nominated accountable executive",  0.99, 0.15, 0.10, 0.45, 1.000),
    ("an ombudsman with escalation power", 0.75, 0.70, 0.65, 0.85, 0.030),
    ("appeal after an adverse decision",   0.95, 0.80, 0.85, 0.90, 0.008),
    ("a quarterly audit sample",           0.40, 0.85, 0.90, 0.75, 0.002),
]
NAMES = ["authority", "information", "time", "incentive"]

print("Four preconditions, and they multiply.")
print()
print(f"{'arrangement':>38}{'auth':>8}{'info':>8}{'time':>8}{'incent':>8}"
      f"{'product':>10}{'binding':>14}")
print("-" * 94)
prod = {}
for name, a, i, t, n, cov in ARRANGEMENTS:
    p = a * i * t * n
    prod[name] = (p, cov)
    binding = NAMES[[a, i, t, n].index(min(a, i, t, n))]
    print(f"{name:>38}{a:>8.2f}{i:>8.2f}{t:>8.2f}{n:>8.2f}{p:>10.4f}{binding:>14}")

print()
print(f"best per-item quality: {max(prod, key=lambda n: prod[n][0])}"
      f" at {max(p for p, c in prod.values()):.4f}")
print("time or information is binding in "
      f"{sum(1 for nm, a, i, t, n, c in ARRANGEMENTS if min(a, i, t, n) in (i, t))}"
      f" of {len(ARRANGEMENTS)} arrangements")

print()
print()
print("And quality is not the whole story, because coverage varies by 500x.")
print()
print(f"{'arrangement':>38}{'per-item quality':>19}{'decisions touched':>20}"
      f"{'effective oversight':>22}")
print("-" * 99)
eff = {}
for name, a, i, t, n, cov in ARRANGEMENTS:
    e = prod[name][0] * cov
    eff[name] = e
    print(f"{name:>38}{prod[name][0]:>19.4f}{cov:>20.3%}{e:>22.5f}")

best_eff = max(eff, key=lambda n: eff[n])
best_qual = max(prod, key=lambda n: prod[n][0])
print()
print(f"highest effective oversight: {best_eff} at {eff[best_eff]:.5f}")
print(f"highest per-item quality:    {best_qual} at {prod[best_qual][0]:.4f}"
      f" but {prod[best_qual][1]:.1%} coverage")

print()
print()
print("Why `time` is binding: what checking one item actually requires.")
print()
# (verification step, seconds, share of catchable errors it would catch)
STEPS = [
    ("read the model's output",                12, 0.06),
    ("read the input the model saw",           25, 0.11),
    ("check the cited source says this",       55, 0.31),
    ("record the reason for the decision",     40, 0.01),
    ("check the applicant's history",         110, 0.03),
    ("check no relevant source was omitted",  180, 0.22),
    ("re-derive the decision from policy",    240, 0.19),
    ("check consistency with past decisions", 300, 0.07),
]
TOTAL_SEC = sum(s for n, s, v in STEPS)
print(f"{'verification step':>40}{'seconds':>10}{'errors it catches':>20}"
      f"{'value per second':>19}")
print("-" * 89)
for name, sec, val in sorted(STEPS, key=lambda s: -s[2] / s[1]):
    print(f"{name:>40}{sec:>10}{val:>20.0%}{val / sec:>19.5f}")

print()
print(f"verifying one item completely: {TOTAL_SEC} seconds ({TOTAL_SEC / 60:.1f} minutes)")


def greedy(budget):
    """Highest error coverage reachable inside a time budget."""
    left, got, done = budget, 0.0, 0
    for name, sec, val in sorted(STEPS, key=lambda s: -s[2] / s[1]):
        if sec <= left:
            left -= sec
            got += val
            done += 1
    return got, done, budget - left


print()
print()
print("What a reviewer covers inside the time they are given.")
print()
print(f"{'time allowed':>16}{'steps completed':>18}{'seconds used':>15}"
      f"{'catchable errors covered':>27}")
print("-" * 76)
BUDGET = 90
cover = {}
for b in (30, 90, 180, 300, 600, TOTAL_SEC):
    got, done, used = greedy(b)
    cover[b] = got
    print(f"{b:>13} s{done:>15} of 8{used:>15}{got:>27.0%}")

print()
print(f"at the usual {BUDGET}-second budget the reviewer covers {cover[BUDGET]:.0%}"
      f" of what is catchable")
print(f"which is {BUDGET / TOTAL_SEC:.0%} of the time the item needs")

print()
print()
print("Now the precondition nobody designs: what it costs to say no.")
print()
# (action, consequence, cost to the reviewer, cost to the organisation, probability)
CONSEQ = [
    ("reject", "delays a colleague's work",          3.0,   1.0, 0.85),
    ("reject", "triggers an escalation meeting",     2.5,   2.0, 0.40),
    ("reject", "the reviewer is overruled",          4.0,   0.5, 0.55),
    ("reject", "the reviewer is visibly wrong",      6.0,   0.5, 0.30),
    ("approve", "nothing observable happens",        0.0,   0.0, 0.988),
    ("approve", "an incident is traced to this",     8.0, 400.0, 0.012),
]
print(f"{'action':>10}{'consequence':>36}{'cost to reviewer':>19}"
      f"{'cost to org':>14}{'probability':>14}")
print("-" * 93)
rev = {"reject": 0.0, "approve": 0.0}
org = {"reject": 0.0, "approve": 0.0}
for act, cons, rc, oc, p in CONSEQ:
    rev[act] += rc * p
    org[act] += oc * p
    print(f"{act:>10}{cons:>36}{rc:>19.1f}{oc:>14.1f}{p:>14.3f}")
print("-" * 93)
print(f"{'EXPECTED':>10}{'reject':>36}{rev['reject']:>19.2f}{org['reject']:>14.2f}")
print(f"{'':>10}{'approve':>36}{rev['approve']:>19.2f}{org['approve']:>14.2f}")

rev_ratio = rev["reject"] / rev["approve"]
org_ratio = org["approve"] / org["reject"]
print()
print(f"to the reviewer, rejecting costs {rev_ratio:.0f}x what approving costs")
print(f"to the organisation, approving costs {org_ratio:.1f}x what rejecting costs")
print(f"the incentives are inverted by a factor of {rev_ratio * org_ratio:.0f}")

INCIDENT = [c for a, cs, c, o, p in CONSEQ if cs.startswith("an incident")][0]
THRESHOLD = rev["reject"] / INCIDENT
print()
print(f"a reviewer minimising their own cost rejects only when they are"
      f" {THRESHOLD:.0%} sure")
print("which is why measured rejection rates sit near zero (ch:sec-permissions)")

print()
print()
print("What actually moves each precondition.")
print()
# (intervention, delta authority, info, time, incentive, reviewer reject-cost multiplier,
#  reviewer approve-cost multiplier)
FIXES = [
    ("give the reviewer the sources",        0.00, 0.45, 0.00, 0.00, 1.00, 1.00),
    ("budget 5 minutes instead of 90 s",     0.00, 0.00, 0.55, 0.00, 1.00, 1.00),
    ("reviewer outside the delivery line",   0.05, 0.00, 0.00, 0.30, 0.35, 1.00),
    ("re-adjudicate 2% of approvals",        0.00, 0.00, 0.00, 0.20, 1.00, 6.00),
    ("publish the rejection rate",           0.00, 0.00, 0.00, 0.15, 1.00, 1.00),
]
BASE = ARRANGEMENTS[1]
print(f"applied to `{BASE[0]}`, product {prod[BASE[0]][0]:.4f}, "
      f"threshold {THRESHOLD:.0%}")
print()
print(f"{'intervention':>38}{'new product':>14}{'improvement':>14}"
      f"{'new threshold':>16}")
print("-" * 82)
for name, da, di, dt, dn, mr, ma in FIXES:
    a = min(1.0, BASE[1] + da)
    i = min(1.0, BASE[2] + di)
    t = min(1.0, BASE[3] + dt)
    n = min(1.0, BASE[4] + dn)
    p = a * i * t * n
    thr = min(1.0, (rev["reject"] * mr) / (INCIDENT * ma))
    print(f"{name:>38}{p:>14.4f}{p / prod[BASE[0]][0]:>13.1f}x{thr:>16.0%}")

a = min(1.0, BASE[1] + sum(f[1] for f in FIXES))
i = min(1.0, BASE[2] + sum(f[2] for f in FIXES))
t = min(1.0, BASE[3] + sum(f[3] for f in FIXES))
n = min(1.0, BASE[4] + sum(f[4] for f in FIXES))
ALL_P = a * i * t * n
ALL_THR = min(1.0, rev["reject"] * 0.35 / (INCIDENT * 6.0))
print("-" * 82)
print(f"{'all five together':>38}{ALL_P:>14.4f}"
      f"{ALL_P / prod[BASE[0]][0]:>13.1f}x{ALL_THR:>16.0%}")

print(f"""
The first table is the audit this chapter exists for. Six oversight arrangements, scored on the
four things a reviewer needs, and the scores **multiply**
(eq:oversight-is-a-conjunction-of-preconditions). The best per-item quality in the table is
{max(p for p, c in prod.values()):.4f}; the worst, `{ARRANGEMENTS[2][0]}`, is
{prod[ARRANGEMENTS[2][0]][0]:.4f} despite near-perfect authority, because authority without
information or time is a signature.

Time or information is the binding constraint in
{sum(1 for nm, a2, i2, t2, n2, c in ARRANGEMENTS if min(a2, i2, t2, n2) in (i2, t2))} of
{len(ARRANGEMENTS)} arrangements, and neither is what organisations discuss when they design
oversight. They discuss authority, which is almost never the constraint.

The second table adds coverage and produces the trade-off that makes this hard. The
high-quality arrangements -- appeal, audit, ombudsman -- touch between {0.002:.1%} and
{0.03:.0%} of decisions. The full-coverage arrangements are the low-quality ones.
`{best_eff}` wins on the product at {eff[best_eff]:.5f}, and `{best_qual}` has the best per-item
quality at {prod[best_qual][0]:.4f} while touching {prod[best_qual][1]:.1%} of decisions.

**An appeal process is excellent oversight of almost nothing.** That is not an argument against
appeals; it is an argument against counting them as the oversight of the system.

The step table is why `time` is binding, and it is the most concrete thing in this chapter.
Verifying one item completely takes {TOTAL_SEC} seconds -- {TOTAL_SEC / 60:.1f} minutes -- and
the checks that catch the most errors are the expensive ones: `check no relevant source was
omitted` at {180} seconds catches {0.22:.0%}, and `re-derive the decision from policy` at
{240} seconds catches {0.19:.0%}.

At the usual {BUDGET}-second budget, spending optimally, the reviewer covers **{cover[BUDGET]:.0%}
of what is catchable** in {BUDGET / TOTAL_SEC:.0%} of the time the item needs. Doubling the
budget to {180} seconds reaches only {cover[180]:.0%}, because the next genuinely useful check
costs {180} seconds on its own and does not fit. {300} seconds reaches {cover[300]:.0%};
{cover[600]:.0%} coverage needs {600} seconds, roughly {600 / BUDGET:.0f} times the usual
budget.

**Review time buys coverage in steps rather than smoothly**, so a small budget increase is
often worth exactly nothing, and the right question is not "how long do reviewers get" but
"which check are we trying to afford".

The consequence table is the precondition nobody designs. To the reviewer, rejecting has an
expected personal cost of {rev['reject']:.2f} and approving {rev['approve']:.2f} -- **rejecting
costs {rev_ratio:.0f} times what approving costs**. To the organisation the ordering reverses:
approving costs {org['approve']:.2f} against {org['reject']:.2f}, a factor of
{org_ratio:.1f}.

**The incentives are inverted by a factor of {rev_ratio * org_ratio:.0f}**
(eq:reviewers-bear-the-cost-of-rejecting-not-approving), and a reviewer minimising their own
cost rejects only when they are **{THRESHOLD:.0%} sure**. That is the mechanism behind
ch:sec-permissions' measured rejection rates, and it is not a statement about the character of
reviewers. It is arithmetic they did not choose.

The last table is what moves it. `give the reviewer the sources` multiplies the arrangement's
product by {(min(1.0, BASE[1]) * min(1.0, BASE[2] + 0.45) * BASE[3] * BASE[4]) / prod[BASE[0]][0]:.1f}
and does nothing to the threshold. `re-adjudicate 2% of approvals` does the reverse: it leaves
the product nearly unchanged and drops the certainty threshold from {THRESHOLD:.0%} to
{min(1.0, rev['reject'] / (INCIDENT * 6.0)):.0%}, because it makes a bad approval observable.

Together the five reach a product of {ALL_P:.4f} -- {ALL_P / prod[BASE[0]][0]:.1f} times the
baseline -- and a threshold of {ALL_THR:.0%}.

**Capability and incentive are separate levers and each fails without the other.** Giving a
reviewer everything they need to catch a problem does not make it rational for them to say so,
and making rejection safe does not give them time to know when to.""")

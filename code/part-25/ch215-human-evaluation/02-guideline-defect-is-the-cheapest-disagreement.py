# -*- coding: utf-8 -*-
# Extracted from: Chapter 215 — Human Evaluation and Annotation Design
# Source: src/.../ch215-human-evaluation.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

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

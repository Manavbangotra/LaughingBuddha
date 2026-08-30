# -*- coding: utf-8 -*-
# Extracted from: Chapter 206 — Data and Model Versioning
# Source: src/.../ch206-versioning.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Un-versioned artefacts are paid for during incidents, at a rate set by search.

The previous listing measured reproducibility as a probability. This one measures what
the missing coverage costs, and the cost is not paid when the artefact changes -- it is
paid weeks later when something is wrong and nobody can say what moved.

Diagnosis is a search over candidate causes. A versioned artefact contributes one known
value; an unversioned one contributes a range that must be explored. So diagnosis time
grows with the size of the candidate space, which grows multiplicatively in the number of
unpinned artefacts (eq:diagnosis-cost-grows-with-unpinned-artefacts).

ch:ops-lifecycle found 15 changes in flight at a 35-day period. This listing shows what
those 15 become when the artefacts underneath them are not pinned.
"""
import math

# (artefact, versioned?, distinct values it could have taken in the window)
ARTEFACTS = [
    ("application code",      True,   14),
    ("model weights",         True,    2),
    ("model version / API",   False,   3),
    ("system prompt",         False,   9),
    ("tool schemas",          False,   4),
    ("retrieval corpus",      False,  31),
    ("retrieval index build", False,   2),
    ("evaluation set",        False,   5),
    ("decoding parameters",   False,   2),
    ("library versions",      True,    6),
]
BISECT_HOURS = 2.6           # cost of testing one hypothesis
INCIDENTS_PER_YEAR = 9.0


def candidates(pinned_set):
    """Size of the space a diagnosis must search."""
    n = 1
    for name, versioned, vals in ARTEFACTS:
        if name in pinned_set or versioned:
            n *= 1          # pinned: exactly one known value
        else:
            n *= vals
    return n


def hours(n_cand):
    """Bisection over an ordered space is log2; unordered is linear in the worst
    case. Real diagnosis is between: assume log2 within an artefact and linear
    across artefacts that must be disambiguated."""
    return BISECT_HOURS * math.log(max(n_cand, 1), 2)


print("Artefacts, and how many distinct values each took during the window a")
print("regression could have been introduced.")
print()
print(f"{'artefact':>24}{'versioned':>12}{'values in window':>19}"
      f"{'contributes':>14}")
print("-" * 72)
for name, versioned, vals in ARTEFACTS:
    print(f"{name:>24}{('yes' if versioned else 'no'):>12}{vals:>19}"
          f"{(1 if versioned else vals):>14}")

base = candidates(set())
print()
print(f"candidate space as-is: {base:,}")
print(f"if everything were pinned: 1")

print()
print()
print("Diagnosis cost, as artefacts are pinned one at a time.")
print()
unpinned = [a for a in ARTEFACTS if not a[1]]
order = sorted(unpinned, key=lambda a: -a[2])
print(f"{'pinned so far':>34}{'candidates':>14}{'diagnosis hrs':>16}"
      f"{'hrs/year':>11}")
print("-" * 78)
pin = set()
path = [(0, "nothing", base, hours(base))]
print(f"{'nothing':>34}{base:>14,}{hours(base):>16.1f}"
      f"{hours(base) * INCIDENTS_PER_YEAR:>11.0f}")
for name, versioned, vals in order:
    pin.add(name)
    c = candidates(pin)
    path.append((len(pin), name, c, hours(c)))
    print(f"{('+ ' + name):>34}{c:>14,}{hours(c):>16.1f}"
          f"{hours(c) * INCIDENTS_PER_YEAR:>11.0f}")

print()
print()
print("What each artefact costs per year, left unpinned.")
print()
print(f"{'artefact':>24}{'values':>9}{'hrs/incident':>15}"
      f"{'hrs/year':>11}{'effort to fix':>16}")
print("-" * 76)
EFFORT = {"model version / API": 1.0, "system prompt": 2.0, "tool schemas": 3.0,
          "retrieval corpus": 8.0, "retrieval index build": 5.0,
          "evaluation set": 2.0, "decoding parameters": 1.0}
cost = {}
for name, versioned, vals in unpinned:
    # Marginal cost: the extra search this artefact alone adds.
    with_it = hours(base)
    without = hours(base / vals)
    cost[name] = (vals, with_it - without,
                  (with_it - without) * INCIDENTS_PER_YEAR, EFFORT[name])
    print(f"{name:>24}{vals:>9}{with_it - without:>15.2f}"
          f"{(with_it - without) * INCIDENTS_PER_YEAR:>11.1f}"
          f"{EFFORT[name]:>16.1f}")

print()
print()
print("Payback: annual hours saved against the effort to pin it.")
print()
rank = sorted(cost, key=lambda k: -(cost[k][2] / cost[k][3]))
print(f"{'rank':>6}{'artefact':>24}{'hrs saved/yr':>15}{'effort':>9}"
      f"{'payback ratio':>16}")
print("-" * 72)
for i, k in enumerate(rank, 1):
    print(f"{i:>6}{k:>24}{cost[k][2]:>15.1f}{cost[k][3]:>9.1f}"
          f"{cost[k][2] / cost[k][3]:>16.1f}")

print()
print()
print("And the interaction with ch:ops-lifecycle's changes-in-flight. A longer")
print("period means more values per artefact, which compounds multiplicatively.")
print()
print(f"{'period days':>13}{'code versions':>16}{'corpus versions':>18}"
      f"{'candidates':>14}{'diagnosis hrs':>16}")
print("-" * 78)
per = {}
for days in (3.0, 7.0, 14.0, 21.0, 35.0):
    code_v = max(1, int(days * 3.0 / 7.0))          # 3 changes a week
    corpus_v = max(1, int(days * 0.9))              # corpus updates daily-ish
    n = 1
    for name, versioned, vals in ARTEFACTS:
        if versioned:
            continue
        if name == "retrieval corpus":
            n *= corpus_v
        else:
            n *= vals
    per[days] = (code_v, corpus_v, n, hours(n))
    print(f"{days:>13.0f}{code_v:>16}{corpus_v:>18}{n:>14,}"
          f"{hours(n):>16.1f}")

print(f"""
The candidate table is the cost of the previous listing's missing coverage, expressed
in the units an incident is measured in. With seven artefacts unpinned and the value
counts shown, a regression could have been introduced by any of **{base:,} distinct
combinations** (eq:diagnosis-cost-grows-with-unpinned-artefacts).

That number is not a search space anyone works through, which is the point. Nobody
enumerates {base:,} hypotheses. What actually happens is that the team
narrows to the two or three they can think of, tries those, and if none is right the
investigation stalls -- so the practical consequence of a large candidate space is not a
long search but an **abandoned** one.

The pinning path prices each step. Pinning `{order[0][0]}` alone takes the space from
{base:,} to {path[1][2]:,} and diagnosis from {path[0][3]:.1f} to {path[1][3]:.1f} hours.
Pinning all seven reaches {path[-1][2]:,} and {path[-1][3]:.1f} hours.

Note the shape, because it is the opposite of the previous listing's. Reproducibility
was a product, and half the effort bought a tenth of the outcome. Diagnosis cost is the
*logarithm* of that product, so **each artefact pinned removes its own log-value from
the total, independently of what else is pinned** -- the reduction is additive rather
than multiplicative.

That difference decides how the work should be justified. If the goal is exact
reproducibility, the programme is all-or-nothing and a half-finished one is nearly
worthless. If the goal is diagnosable incidents, **every step pays its own way**, and
the same list becomes an incrementally-fundable backlog.

Since the second goal is the one that shows up in an incident review, it is usually the
easier case to make -- and it happens to build the same thing.

The per-artefact table ranks by annual cost. `{rank[0]}` is worth
{cost[rank[0]][2]:.1f} hours a year and costs {cost[rank[0]][3]:.1f} to fix, a payback
ratio of {cost[rank[0]][2] / cost[rank[0]][3]:.1f}. `{rank[-1]}` is worth
{cost[rank[-1]][2]:.1f} hours against {cost[rank[-1]][3]:.1f}, a ratio of
{cost[rank[-1]][2] / cost[rank[-1]][3]:.1f}.

**The ranking is not the same as the exposure ranking in the previous listing.** There
the retrieval corpus led because its influence and coverage gap were largest; here the
ordering is driven by value count against effort, and the cheap artefacts with many
versions rise. A team optimising for reproducibility and a team optimising for
diagnosability should build the same things in a different order.

The period table is where this chapter meets ch:ops-lifecycle. The number of distinct
values an artefact took is a function of how long the window is, and the window is the
loop period. At a {3.0:.0f}-day period the candidate space is {per[3.0][2]:,}; at
{35.0:.0f} days it is {per[35.0][2]:,} --
{per[35.0][2] / per[3.0][2]:.0f} times larger.

**The period multiplies the candidate space, and the candidate space is what makes an
incident undiagnosable.** ch:ops-lifecycle argued that a long period destroys
attribution; this table is the mechanism. It is not that fifteen changes are hard to
distinguish. It is that fifteen changes sit on top of thirty-one corpus versions and
nine prompt edits, and the product is what has to be searched.

So the two interventions compose in the direction you would want: shortening the period
reduces the values per artefact, and pinning the artefacts removes them from the product
entirely. Doing both takes the candidate space from {per[35.0][2]:,} to {1} -- there is nothing
to search, because the change that caused it is identified directly.

The practical reading is that **versioning is incident-response tooling that happens to
run continuously.** It is funded as hygiene and it is paid for during outages, which is
why it is chronically under-resourced: the cost lands on a different quarter, a different
metric, and frequently a different team from the one that would have done the work.""")

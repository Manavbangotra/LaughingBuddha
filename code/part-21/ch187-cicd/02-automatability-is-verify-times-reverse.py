# -*- coding: utf-8 -*-
# Extracted from: Chapter 187 — CI/CD and Architecture Agents
# Source: src/.../ch187-cicd.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Architecture, which is the least automatable activity in software and not
because it is the hardest.

ch:as-specialized found two properties deciding whether an agent can work in a
domain: whether it can CHECK its work, and whether it can UNDO a mistake. It found
them complementary -- fixing either alone bought almost nothing and fixing both
bought fifty-four points.

Software engineering activities differ enormously on both, and they differ TOGETHER.
Debugging has a failing test (verifiable) and version control (reversible).
Architecture has neither: no test tells you a service boundary is wrong, and by the
time you find out, three teams have built against it
(eq:automatability-is-verify-times-reverse).

This listing places the activities and prices automating each.
"""
# (activity, share of engineering effort, verifiability, reversibility,
#  cost of a wrong decision, how much cheaper an agent makes the attempt)
ACTIVITIES = [
    ("fix a reported bug",     0.19, 0.92, 0.95,    120.0, 0.45),
    ("write a test",           0.09, 0.70, 0.97,     40.0, 0.35),
    ("refactor a module",      0.11, 0.62, 0.90,    260.0, 0.50),
    ("implement a feature",    0.26, 0.58, 0.78,    380.0, 0.55),
    ("choose a dependency",    0.04, 0.31, 0.34,   2600.0, 0.70),
    ("design a data model",    0.06, 0.24, 0.19,   7000.0, 0.75),
    ("set a service boundary", 0.05, 0.12, 0.09,  21000.0, 0.80),
    ("everything else",        0.20, 0.55, 0.70,    200.0, 0.70),
]

P_AGENT_WRONG = 0.30    # an agent's decision is wrong this often
P_HUMAN_WRONG = 0.17


def expected(activity, who, guarded=False):
    """Expected cost of one decision. A wrong decision is caught by the
    verifier with probability `verifiability`; an uncaught wrong decision is
    undone with probability `reversibility`, at a fraction of its full cost."""
    name, share, ver, rev, cost, _ = activity
    p_wrong = P_AGENT_WRONG if who == "agent" else P_HUMAN_WRONG
    v = min(ver * 1.35, 0.97) if guarded else ver
    caught = p_wrong * v
    escaped = p_wrong * (1 - v)
    # Caught costs a retry; escaped costs the full amount, discounted by how
    # recoverable it is.
    return caught * cost * 0.12 + escaped * cost * (1 - rev * 0.85)


print("Software activities, by whether a mistake can be DETECTED and whether it")
print("can be UNDONE -- ch:as-specialized's two binding properties.")
print()
print(f"{'activity':>24}{'effort':>9}{'verifiable':>12}{'reversible':>12}"
      f"{'cost if wrong':>15}")
print("-" * 72)
for name, share, ver, rev, cost, _ in ACTIVITIES:
    print(f"{name:>24}{share:>9.0%}{ver:>12.0%}{rev:>12.0%}{cost:>15,.0f}")

print()
print()
print("The product of the two is what ch:as-specialized found decisive, and it")
print("orders the activities cleanly.")
print()
print(f"{'activity':>24}{'verify x reverse':>18}{'agent cost':>13}"
      f"{'human cost':>13}{'ratio':>8}")
print("-" * 76)
tab = {}
for a in ACTIVITIES:
    name = a[0]
    ag = expected(a, "agent")
    hu = expected(a, "human")
    tab[name] = (a[2] * a[3], ag, hu, ag / max(hu, 1e-9))
    print(f"{name:>24}{a[2] * a[3]:>18.3f}{ag:>13.1f}{hu:>13.1f}"
          f"{ag / max(hu, 1e-9):>8.2f}")

print()
print()
print("Net effect of automating each activity: the agent is cheaper to run and")
print("more often wrong, so the question is whether the saving covers the risk.")
print()
print(f"{'activity':>24}{'effort saved':>14}{'extra risk':>12}{'net':>10}"
      f"{'verdict':>12}")
print("-" * 72)
net = {}
HOURLY = 95.0
for a in ACTIVITIES:
    name, share, ver, rev, cost, cheaper = a
    # Effort saved, in the same units as risk, per decision.
    saved = cheaper * 4.0 * HOURLY / 10.0
    extra = expected(a, "agent") - expected(a, "human")
    net[name] = (saved, extra, saved - extra)
    print(f"{name:>24}{saved:>14.1f}{extra:>12.1f}{saved - extra:>10.1f}"
          f"{('automate' if saved > extra else 'do not'):>12}")

print()
print()
print("Ranked by net, against the two properties that produced it.")
print()
order = sorted(net, key=lambda k: -net[k][2])
look = {a[0]: a for a in ACTIVITIES}
print(f"{'rank':>6}{'activity':>24}{'net':>10}{'verifiable':>12}"
      f"{'reversible':>12}")
print("-" * 64)
for i, name in enumerate(order, 1):
    a = look[name]
    print(f"{i:>6}{name:>24}{net[name][2]:>10.1f}{a[2]:>12.0%}{a[3]:>12.0%}")

print()
print()
print("What a verifier would buy where one could be built -- ch:aids-stack's")
print("check-strong-build-weak rule, applied here.")
print()
print(f"{'activity':>24}{'as is':>10}{'with a verifier':>18}{'gain':>10}")
print("-" * 62)
gd = {}
for a in ACTIVITIES:
    name = a[0]
    base = expected(a, "agent")
    guard = expected(a, "agent", guarded=True)
    gd[name] = (base, guard, base - guard)
    print(f"{name:>24}{base:>10.1f}{guard:>18.1f}{base - guard:>10.1f}")

print()
print()
print("And the effort-weighted picture, which says how much of software")
print("engineering sits in each regime.")
print()
hi = sum(a[1] for a in ACTIVITIES if a[2] * a[3] >= 0.40)
mid = sum(a[1] for a in ACTIVITIES if 0.10 <= a[2] * a[3] < 0.40)
lo = sum(a[1] for a in ACTIVITIES if a[2] * a[3] < 0.10)
print(f"{'regime':>34}{'share of effort':>18}")
print("-" * 54)
print(f"{'verifiable and reversible':>34}{hi:>18.0%}")
print(f"{'partly one or the other':>34}{mid:>18.0%}")
print(f"{'neither':>34}{lo:>18.0%}")

print(f"""
The ranking table has a cliff in it rather than a slope, and that is the finding.

Implementing a feature nets {net['implement a feature'][2]:+.1f}; choosing a
dependency nets {net['choose a dependency'][2]:+.1f}; setting a service boundary
nets {net['set a service boundary'][2]:+.1f}. **The activities do not shade from
automatable to less automatable. They fall off a cliff**, and the cliff is exactly
where the verify-times-reverse product drops below about
{0.10:.2f} (eq:automatability-is-verify-times-reverse).

That product is ch:as-specialized's finding transplanted. There, fixing observation
alone bought {0.3:+.1f} points and fixing undo alone bought
{12.3:+.1f}, and fixing both bought {54.5:+.1f} -- the properties were
complementary, so a domain weak on both was catastrophically weak. Software
activities span that whole range internally.

Debugging sits at {look['fix a reported bug'][2]:.0%} verifiable and
{look['fix a reported bug'][3]:.0%} reversible: a failing test says whether you
succeeded and version control undoes the attempt. Setting a service boundary sits at
{look['set a service boundary'][2]:.0%} and {look['set a service boundary'][3]:.0%}:
no test says a boundary is wrong, and by the time anyone knows, three teams have
built against it.

**Architecture is the least automatable activity in software, and not because it is
the hardest.** It is because it combines the two properties an agent needs least of
and needs most.

The verifier table is ch:aids-stack's check-strong-build-weak rule, and it points
where that rule always points. A verifier is worth {gd['fix a reported bug'][2]:.1f}
on bug fixing, where one already exists, and {gd['set a service boundary'][2]:.1f} on
service boundaries, where none does.

Which is the constructive reading of this whole chapter, and it is more useful than
"do not automate architecture". **The reason architecture resists automation is a
missing verifier, and verifiers for architectural properties are buildable.** A
layering rule enforced by an import checker. A latency budget asserted in a
contract test. A schema compatibility check that fails a migration that breaks
readers. Each converts an unverifiable decision into a partly verifiable one, and
each is ordinary engineering.

That is also the answer to why some organisations get much more out of coding agents
than others, and it is not model access. **A codebase with executable architectural
constraints has moved several activities up this table**, permanently, for every
agent and every engineer.

The last table sizes the opportunity honestly. About {hi:.0%} of engineering effort
sits in the verifiable-and-reversible regime where agents work well,
{mid:.0%} partly, and {lo:.0%} in the regime where they do not.

So the effort-weighted picture is encouraging and the risk-weighted one is not: the
{lo:.0%} carries the decisions whose costs are measured in
{look['set a service boundary'][4] / look['fix a reported bug'][4]:.0f} times a bug
fix. **Automate the majority of the effort; keep the minority that carries the
consequences** -- which is ch:aids-oversight's divide-by-gradeability rule, arriving
independently in a second domain and with reversibility added as a second
criterion.""")

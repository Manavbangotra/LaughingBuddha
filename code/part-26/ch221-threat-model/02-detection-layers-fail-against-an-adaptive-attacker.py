# -*- coding: utf-8 -*-
# Extracted from: Chapter 221 — The AI Threat Model
# Source: src/.../ch221-threat-model.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Detection layers compose beautifully against an attacker who only tries once.

Defence in depth is the right instinct and the arithmetic is usually done wrong. Stacked
detectors multiply their miss rates, which looks excellent -- until you notice that the
multiplication assumes a *fixed* attack. An attacker who can observe whether an attempt was
blocked is running a search, and search defeats detection at a rate set by how many attempts
they get (eq:detection-layers-fail-against-an-adaptive-attacker).

The layers that survive are the ones whose guarantee does not depend on recognising the
attack: what a successful injection is permitted to reach. Those bound the damage rather
than the probability, and the bound holds at any number of attempts
(eq:only-capability-limits-bound-the-damage).
"""
# (layer, P(blocks a fixed attack), utility cost, depends on detection?)
LAYERS = [
    ("delimiters and warnings",       0.08, 0.01, True),
    ("input injection classifier",    0.62, 0.06, True),
    ("instruction-hierarchy training", 0.44, 0.03, True),
    ("output scanner",                0.37, 0.04, True),
    ("tool allow-list per task",      0.71, 0.18, False),
    ("no privileged sink after untrusted read", 0.93, 0.34, False),
    ("human approval on privileged sinks", 0.88, 0.51, False),
]

print("Each layer alone, against an attacker who submits one fixed attack.")
print()
print(f"{'layer':>44}{'blocks':>9}{'utility cost':>15}"
      f"{'detection-based?':>19}")
print("-" * 87)
for name, p, u, det in LAYERS:
    print(f"{name:>44}{p:>9.0%}{u:>15.0%}"
          f"{('yes' if det else 'no'):>19}")

det_layers = [l for l in LAYERS if l[3]]
cap_layers = [l for l in LAYERS if not l[3]]


def miss(layers):
    m = 1.0
    for name, p, u, det in layers:
        m *= (1.0 - p)
    return m


def utility(layers):
    u = 1.0
    for name, p, uc, det in layers:
        u *= (1.0 - uc)
    return u


print()
print(f"all four detection layers together miss {miss(det_layers):.2%}")
print(f"all three capability layers together miss {miss(cap_layers):.3%}")

print()
print()
print("Now let the attacker try again after each block. Detection layers are")
print("a filter to be searched around; capability limits are not.")
print()
print("(all three columns are P(the attacker succeeds at least once))")
print()
print(f"{'attempts':>10}{'detection stack':>19}{'capability stack':>20}"
      f"{'both':>12}{'vs 1 attempt':>16}")
print("-" * 77)
md, mc = miss(det_layers), miss(cap_layers)
adapt = {}
for k in (1, 3, 10, 30, 100, 1000):
    d = 1.0 - (1.0 - md) ** k
    # A capability limit does not admit search: the sink is unreachable on that
    # path however the request is phrased, so only its own miss rate applies.
    c = mc
    both = c * d
    adapt[k] = (d, c, both)
    print(f"{k:>10}{d:>19.2%}{c:>20.4%}{both:>12.4%}"
          f"{d / md:>15.1f}x")

print()
print("The middle column does not move. That is the whole argument.")

print()
print()
print("What each stack costs in utility, against what it bounds.")
print()
print(f"{'stack':>34}{'utility kept':>15}{'success at 1 try':>19}"
      f"{'success at 100':>17}{'success at 1000':>18}")
print("-" * 103)
STACKS = [
    ("nothing",                       []),
    ("detection only",                det_layers),
    ("capability only",               cap_layers),
    ("detection + capability",        LAYERS),
    ("cheapest detection + tool allow-list",
     [LAYERS[1], LAYERS[4]]),
]
st = {}
for label, ls in STACKS:
    if not ls:
        s1 = s100 = s1000 = 1.0
        u = 1.0
    else:
        dl = [l for l in ls if l[3]]
        cl = [l for l in ls if not l[3]]
        md = miss(dl) if dl else 1.0
        mc = miss(cl) if cl else 1.0
        s1 = md * mc
        s100 = (1.0 - (1.0 - md) ** 100 if dl else 1.0) * mc
        s1000 = (1.0 - (1.0 - md) ** 1000 if dl else 1.0) * mc
        u = utility(ls)
    st[label] = (u, s1, s100, s1000)
    print(f"{label:>34}{u:>15.0%}{s1:>19.3%}{s100:>17.3%}{s1000:>18.3%}")

print()
print("Read the last two columns across. Detection-only degrades toward 1;")
print("anything with a capability limit converges to that limit.")

print()
print()
print("Residual risk = P(success) x blast radius. Only one term responds.")
print()
BLAST = [
    ("no restriction",                 100.0),
    ("read-only tools",                 34.0),
    ("read-only plus rate limit",       21.0),
    ("scoped credentials, one tenant",  12.0),
    ("proposal only, human executes",    3.0),
]
print(f"{'blast-radius control':>34}{'radius':>10}"
      f"{'residual, detection only':>27}{'residual, det + cap':>22}")
print("-" * 93)
res = {}
for name, radius in BLAST:
    r_det = st["detection only"][2] * radius
    r_both = st["detection + capability"][2] * radius
    res[name] = (radius, r_det, r_both)
    print(f"{name:>34}{radius:>10.0f}{r_det:>27.3f}{r_both:>22.4f}")

print()
print(f"detection-only at 100 attempts: P(success) = "
      f"{st['detection only'][2]:.1%}")
print("so the residual is essentially the blast radius")

print()
print()
print("Where a fixed security budget should go.")
print()
# (investment, blocks a fixed attack, detection-based?, utility cost, weeks)
BUDGET_ITEMS = [
    ("tune the injection classifier",     0.62, True,  0.06, 4.0),
    ("add an output scanner",             0.37, True,  0.04, 3.0),
    ("scope tool allow-lists by task",    0.71, False, 0.18, 2.5),
    ("split read and write agents",       0.93, False, 0.34, 8.0),
    ("human approval on the top 3 sinks", 0.88, False, 0.11, 1.5),
]
ATTEMPTS = 100
print(f"{'investment':>38}{'blocks a fixed attack':>24}"
      f"{'blocks at 100 attempts':>25}{'weeks':>8}{'per week':>11}")
print("-" * 106)
inv = {}
for name, fixed, det, u, weeks in BUDGET_ITEMS:
    if det:
        eff = fixed ** ATTEMPTS              # P(all 100 attempts blocked)
    else:
        eff = fixed                          # not searchable
    inv[name] = (fixed, eff, weeks, eff / weeks)
    print(f"{name:>38}{fixed:>24.0%}{eff:>25.1%}{weeks:>8.1f}"
          f"{eff / weeks:>11.3f}")

print()
print("Detection items round to zero against an adaptive attacker with 100")
print("attempts, which is the honest way to score them.")

print(f"""
The single-layer table is the ordinary defence-in-depth picture, and read alone it is
encouraging. Four detection layers together miss only {miss(det_layers):.2%} of a fixed
attack; three capability layers miss {miss(cap_layers):.3%}. Either stack looks adequate.

The adaptive table is what happens when the attacker gets feedback. A blocked injection is
information -- the attacker knows that phrasing failed and tries another. At
{miss(det_layers):.2%} miss per attempt, the detection stack is defeated
{adapt[3][0]:.1%} of the time within {3} attempts and {adapt[100][0]:.1%} within
{100} (eq:detection-layers-fail-against-an-adaptive-attacker).

**The capability column does not move.** {adapt[1][1]:.4%} at one attempt,
{adapt[1000][1]:.4%} at a thousand -- because a tool that is not on the allow-list is not
reachable by a better-phrased request. There is nothing to search around.

That is the structural difference and it is worth stating in one sentence: **detection
bounds a probability and capability bounds a set**, and only one of those survives
repetition.

cite:zou2023universal is the empirical form of the same point. An adversarial suffix found
by optimisation on open models transferred to ChatGPT, Bard and Claude -- which means the
attacker does not even need to query your filter, because they can search against a proxy
and bring the result. An attempt budget of one against your system can still be an attempt
budget of thousands against something that behaves like it.

The stack table prices the choice. Detection alone keeps
{st['detection only'][0]:.0%} of utility and lets {st['detection only'][2]:.1%} through at
100 attempts. Adding capability limits takes utility to
{st['detection + capability'][0]:.0%} and success at 100 attempts to
{st['detection + capability'][2]:.3%}.

The last row is the practical middle. An injection classifier plus per-task tool allow-lists
keeps {st['cheapest detection + tool allow-list'][0]:.0%} of utility and holds success at a
hundred attempts to {st['cheapest detection + tool allow-list'][2]:.0%} -- not a guarantee,
and against detection-only's {st['detection only'][2]:.0%} it is the difference between a
bounded and an unbounded risk.

Note where the bound comes from: {st['cheapest detection + tool allow-list'][2]:.0%} is
exactly the allow-list's own miss rate. **The classifier contributes nothing to the
asymptote** -- it raises the cost per attempt and leaves the ceiling where the allow-list put
it. The classifier is not useless; it is not load-bearing, and an architecture in which it is
load-bearing has no ceiling at all.

The residual table is how to present this to whoever signs off. Residual risk is
probability times blast radius, and against an adaptive attacker the probability term is
close to one under detection-only defences -- {st['detection only'][2]:.0%} at a hundred
attempts. So **the residual is essentially the blast radius**
(eq:only-capability-limits-bound-the-damage), and every row in that table is a product
decision rather than a security control.

Going from unrestricted tools to proposal-only with a human executing takes the radius from
{BLAST[0][1]:.0f} to {BLAST[4][1]:.0f}, and that is the largest single move available in
this chapter.

The budget table is the ranking, and it scores detection items at zero against an adaptive
attacker -- which is harsh and is the right convention, because a control that a hundred
attempts defeat should not be credited with preventing anything at a hundred attempts.
`{BUDGET_ITEMS[4][0]}` returns {inv[BUDGET_ITEMS[4][0]][3]:.3f} per week,
`{BUDGET_ITEMS[2][0]}` returns {inv[BUDGET_ITEMS[2][0]][3]:.3f}, and the two detection items
return {inv[BUDGET_ITEMS[0][0]][3]:.3f} and {inv[BUDGET_ITEMS[1][0]][3]:.3f}.

The honest summary for ch:sec-threat-model is uncomfortable and short. **Assume the model
will be fooled**, spend on what a fooled model can reach, and treat every detector as a
cost-raiser rather than a boundary. cite:beurerkellner2025patterns reaches the same
conclusion from the design-pattern side, and cite:debenedetti2024agentdojo is where you go
to find out what any of it costs in utility on tasks you care about.""")

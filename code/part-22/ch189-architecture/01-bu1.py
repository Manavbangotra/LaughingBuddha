# -*- coding: utf-8 -*-
# Extracted from: Chapter 189 — Architecting Production AI Systems
# Source: src/.../ch189-architecture.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Three properties conventional architecture does not assume together.

A production system's central component is usually deterministic, cheap relative
to the request, and either working or not. A language model is none of those:

  nondeterministic   the same input can produce different output
  expensive          one call can cost more than the rest of the request
  occasionally wrong it returns 200 OK and the wrong answer

Each individually has precedent. Together they break specific classical
techniques, and this listing measures which ones and how badly
(eq:three-properties-break-the-stack).

The technique that breaks worst is the one nobody notices: a 200 response
containing a wrong answer is invisible to every availability instrument built in
the last thirty years.
"""
M = 200000

# (technique, what it assumes, degradation under each property)
# Degradation is the share of the technique's normal value that survives.
TECHNIQUES = [
    ("response caching",     "same input, same answer", 0.35, 1.00, 0.55),
    ("retry on failure",     "failures are transient",  0.90, 0.40, 0.25),
    ("golden-output tests",  "output is a function",    0.15, 1.00, 0.60),
    ("health checks",        "up or down",              0.95, 1.00, 0.05),
    ("capacity planning",    "cost per request is flat", 0.80, 0.20, 0.90),
    ("circuit breakers",     "errors are observable",   0.95, 0.85, 0.15),
    ("load balancing",       "requests are equivalent", 0.85, 0.45, 0.95),
]
PROPS = ["nondeterministic", "expensive", "sometimes wrong"]


def surviving(t):
    """Share of a technique's classical value that survives all three."""
    return t[2] * t[3] * t[4]


print("Classical techniques, and how much of each survives a component that is")
print("nondeterministic, expensive, and occasionally wrong.")
print()
print(f"{'technique':>22}{'assumes':>28}" + "".join(f"{p[:9]:>11}"
                                                    for p in PROPS)
      + f"{'survives':>11}")
print("-" * 106)
tab = {}
for t in TECHNIQUES:
    tab[t[0]] = surviving(t)
    print(f"{t[0]:>22}{t[1]:>28}{t[2]:>11.0%}{t[3]:>11.0%}{t[4]:>11.0%}"
          f"{surviving(t):>11.0%}")

print()
print()
print("Ranked by what is left, which says which parts of a conventional design")
print("have to be replaced rather than tuned.")
print()
order = sorted(tab, key=lambda k: tab[k])
look = {t[0]: t for t in TECHNIQUES}
print(f"{'rank':>6}{'technique':>22}{'survives':>11}{'broken by':>28}")
print("-" * 68)
for i, name in enumerate(order, 1):
    t = look[name]
    worst = PROPS[min(range(3), key=lambda k: t[2 + k])]
    print(f"{i:>6}{name:>22}{tab[name]:>11.0%}{worst:>28}")

print()
print()
print("Which property does the most damage, summed across techniques.")
print()
print(f"{'property':>22}{'mean survival':>16}{'techniques it halves':>23}")
print("-" * 61)
dmg = {}
for k, p in enumerate(PROPS):
    vals = [t[2 + k] for t in TECHNIQUES]
    halved = sum(1 for v in vals if v < 0.5)
    dmg[p] = (sum(vals) / len(vals), halved)
    print(f"{p:>22}{sum(vals) / len(vals):>16.0%}{halved:>23}")

print()
print()
print("The one that matters most is the one with no instrument. Availability")
print("monitoring sees a 200 response; semantic failure is inside it.")
print()
P_UP = 0.999
print(f"{'semantic error rate':>21}{'availability sees':>19}"
      f"{'users experience':>19}{'gap':>10}")
print("-" * 69)
sem = {}
for e in (0.02, 0.06, 0.15, 0.30):
    seen = P_UP
    real = P_UP * (1 - e)
    sem[e] = (seen, real, seen - real)
    print(f"{e:>21.0%}{seen:>19.3%}{real:>19.3%}{seen - real:>10.1%}")

print()
print()
print("And what that does to an error budget. A 99.9% availability target with")
print("a semantic error rate the target cannot see:")
print()
print(f"{'semantic error rate':>21}{'budget nominal':>16}{'budget real':>14}"
      f"{'overspend':>12}")
print("-" * 63)
BUDGET = 1 - P_UP
for e in (0.02, 0.06, 0.15, 0.30):
    real_err = 1 - P_UP * (1 - e)
    print(f"{e:>21.0%}{BUDGET:>16.3%}{real_err:>14.3%}"
          f"{real_err / BUDGET:>12.0f}x")

print(f"""
The survival column is what a conventional design looks like after the central
component acquires three unusual properties. Nothing in the table survives above
{max(tab.values()):.0%}, and the technique that survives worst is the one an
operations team would name first if asked what tells them the system is healthy.

**Health checks survive {tab['health checks']:.0%}**
(eq:three-properties-break-the-stack). A health check answers "is it up", and a
model that is up
returns a confident wrong answer with the same status code as a right one.

The property doing the most damage is the third, and not by the count. Expense
halves as many techniques as sometimes-wrong does -- {dmg['expensive'][1]} each --
but it leaves a mean survival of {dmg['expensive'][0]:.0%} against
sometimes-wrong's {dmg['sometimes wrong'][0]:.0%}. Expense degrades broadly;
being wrong destroys.

That is worth separating from the other two because the industry has vocabulary
for them and not for it. Nondeterminism is familiar from distributed systems.
Expense is familiar from anything with a cloud bill. **A component that succeeds
and is wrong has no established instrument at all**, and the last two tables are
why that matters.

At a {0.06:.0%} semantic error rate, availability monitoring reports
{sem[0.06][0]:.3%} and users experience {sem[0.06][1]:.3%}. The instrument is not
slightly optimistic; it is measuring a different quantity.

The error-budget table converts that into the unit teams actually manage. A
{0.999:.1%} availability target has a budget of {BUDGET:.3%}. At a
{0.06:.0%} semantic error rate the real failure rate is {1 - P_UP * 0.94:.3%} --
**an overspend of {(1 - P_UP * 0.94) / BUDGET:.0f} times the entire budget**,
against an instrument reporting the budget as nearly untouched.

So the architectural consequence is narrow and load-bearing: **a system with a
model in it needs a second reliability instrument**, measuring whether answers are
right rather than whether responses arrived. Nothing in a conventional stack
supplies one, every chapter of this part assumes one exists, and it is the first
thing to build.

The rest of the ranking says what else has to be replaced rather than tuned.
Golden-output tests survive {tab['golden-output tests']:.0%}, which is why
ch:sd-architecture cannot recommend the usual regression suite. Retries survive
{tab['retry on failure']:.0%} -- ch:ag-recovery's finding, arriving as an
architecture constraint rather than an agent one. And caching survives
{tab['response caching']:.0%}, which ch:sd-routing-caching takes up in detail.

Load balancing survives best at {tab['load balancing']:.0%}, and even that is
degraded: requests are no longer equivalent when one costs forty times another,
which is a queueing problem ch:sd-async has to solve.""")

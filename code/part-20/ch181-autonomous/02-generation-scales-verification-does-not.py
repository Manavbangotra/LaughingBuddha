# -*- coding: utf-8 -*-
# Extracted from: Chapter 181 — Autonomous Experimentation and Report Generation
# Source: src/.../ch181-autonomous.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The economics of cheap generation, and what does not get cheaper.

cite:lu2024aiscientist reports under $15 per paper. At that price the natural move
is volume: generate a thousand, keep the good ones. That is
cite:testini2025dsautomation's third gap -- automation as REDESIGN rather than
substitution -- and it is a real argument that this book's earlier listings, which
all price a fixed workload done faster, cannot see.

This listing takes it seriously and finds where it breaks. Generation gets cheap.
Verification does not, because the only verification that works is the kind that
does not share the generator's blind spots -- and that means independent effort,
which is the thing whose cost is set by someone else's time
(eq:generation-scales-verification-does-not).
"""
import numpy as np

rng = np.random.default_rng(4831)

M = 40000
P_GOOD = 0.18
GEN_COST = 15.0             # dollars per generated result
CHEAP_VERIFY = 0.40         # a same-family automated review
CHEAP_PREC = 0.90           # its detection rate on flaws it can see
CHEAP_SHARED = 0.90         # ...but it shares the generator's blind spots
EXPERT_COST = 340.0         # an independent expert review, in dollars of time
EXPERT_DETECT = 0.82


def yield_of(n_generated, use_cheap=True, use_expert=False,
             expert_capacity=None, p_good=P_GOOD):
    """Returns (sound results delivered, unsound delivered, total cost,
    expert reviews consumed)."""
    good = rng.binomial(n_generated, p_good)
    bad = n_generated - good
    cost = n_generated * GEN_COST
    passed_good, passed_bad = good, bad

    if use_cheap:
        cost += n_generated * CHEAP_VERIFY
        # The cheap reviewer only catches flaws outside the shared blind spot.
        eff = CHEAP_PREC * (1.0 - CHEAP_SHARED)
        passed_bad = rng.binomial(passed_bad, 1.0 - eff)
        passed_good = rng.binomial(passed_good, 0.95)

    reviews = 0
    if use_expert:
        submitted = passed_good + passed_bad
        cap = submitted if expert_capacity is None else min(submitted,
                                                            expert_capacity)
        reviews = cap
        frac = cap / submitted if submitted else 0.0
        cost += cap * EXPERT_COST
        # Only the reviewed share is filtered; the rest passes unexamined.
        rev_bad = rng.binomial(passed_bad, frac)
        rev_good = rng.binomial(passed_good, frac)
        kept_bad = (passed_bad - rev_bad) + rng.binomial(
            rev_bad, 1.0 - EXPERT_DETECT)
        kept_good = (passed_good - rev_good) + rng.binomial(rev_good, 0.93)
        passed_good, passed_bad = kept_good, kept_bad

    return int(passed_good), int(passed_bad), float(cost), int(reviews)


print(f"Generation at ${GEN_COST:.0f} each, {P_GOOD:.0%} of results sound.")
print(f"A same-family automated review costs ${CHEAP_VERIFY:.2f} and shares")
print(f"{CHEAP_SHARED:.0%} of the generator's blind spots. An independent expert")
print(f"review costs ${EXPERT_COST:.0f} of someone's time and catches "
      f"{EXPERT_DETECT:.0%}.")
print()
print(f"{'generated':>11}{'delivered sound':>17}{'delivered unsound':>19}"
      f"{'precision':>11}{'cost':>12}")
print("-" * 70)
tab = {}
for n in (10, 100, 1000, 10000):
    g, b, c, _ = yield_of(n)
    tab[n] = (g, b, c, g / (g + b) if g + b else 0)
    print(f"{n:>11,}{g:>17,}{b:>19,}{g / max(g + b, 1):>11.1%}{c:>12,.0f}")

print()
print()
print("Volume delivers more sound results AND more unsound ones, in fixed")
print("proportion, because the filter is correlated with the generator.")
print()
print(f"{'generated':>11}{'sound per $1k':>16}{'unsound per $1k':>18}")
print("-" * 46)
for n in (10, 100, 1000, 10000):
    g, b, c, _ = tab[n]
    print(f"{n:>11,}{g / (c / 1000):>16.2f}{b / (c / 1000):>18.2f}")

print()
print()
print("Now add independent expert review, unlimited. This is the version that")
print("works, and its cost is the reason it is not what gets deployed.")
print()
print(f"{'generated':>11}{'sound':>9}{'unsound':>10}{'precision':>11}"
      f"{'cost':>13}{'$ per sound':>14}")
print("-" * 68)
ex = {}
for n in (10, 100, 1000):
    g, b, c, r = yield_of(n, use_expert=True)
    ex[n] = (g, b, c, r)
    print(f"{n:>11,}{g:>9,}{b:>10,}{g / max(g + b, 1):>11.1%}{c:>13,.0f}"
          f"{c / max(g, 1):>14,.0f}")

print()
print()
print("And the version that actually happens: expert capacity is fixed, so")
print("volume overflows it and the overflow ships unreviewed.")
print()
CAPACITY = 40
print(f"Expert capacity: {CAPACITY} reviews.")
print()
print(f"{'generated':>11}{'reviewed':>10}{'unreviewed':>12}{'precision':>11}"
      f"{'sound delivered':>17}")
print("-" * 61)
cap = {}
for n in (10, 100, 1000, 10000):
    g, b, c, r = yield_of(n, use_expert=True, expert_capacity=CAPACITY)
    submitted = g + b
    cap[n] = (g, b, r, g / max(submitted, 1))
    print(f"{n:>11,}{r:>10,}{max(submitted - r, 0):>12,}"
          f"{g / max(submitted, 1):>11.1%}{g:>17,}")

print()
print()
print("The commons. Every generated result that reaches a reviewer consumes")
print("attention that is not replenished by generating more.")
print()
print(f"{'generated':>11}{'submitted':>12}{'per reviewer':>15}"
      f"{'catch rate':>12}")
print("-" * 50)
REVIEWERS = 4
HALF = 25
cm = {}
for n in (10, 100, 1000, 10000):
    g, b, _, _ = yield_of(n)
    submitted = g + b
    load = submitted / REVIEWERS
    catch = EXPERT_DETECT / (1.0 + load / HALF)
    cm[n] = (submitted, load, catch)
    print(f"{n:>11,}{submitted:>12,}{load:>15,.0f}{catch:>12.1%}")

print(f"""
The first two tables make the volume argument and then undercut it.

Generating {10000:,} results delivers {tab[10000][0]:,} sound ones against
{tab[10][0]:,} from generating {10}. Volume works. But it also delivers
{tab[10000][1]:,} unsound ones, and the precision is
{tab[10000][3]:.1%} at both scales.

**Cheap generation with a correlated filter scales the output and not the
proportion.** You get more of everything, in the ratio the generator produces, and
the ratio is what determines whether the output can be used without checking it.

The third table adds the filter that works, and prices it.

With unlimited independent expert review, precision rises to
{ex[1000][0] / max(ex[1000][0] + ex[1000][1], 1):.1%} -- and the cost per sound
result is ${ex[1000][2] / max(ex[1000][0], 1):,.0f}.

**Generation costs ${GEN_COST:.0f} and verification costs
${ex[1000][2] / max(ex[1000][0], 1):,.0f} per usable result**
(eq:generation-scales-verification-does-not). The famous figure is the cost of the
half that got cheap.

That is not an argument against the pipeline. It is an argument about which number
describes it: a system that produces a sound result for
${ex[1000][2] / max(ex[1000][0], 1):,.0f} may still be excellent value, and it is a
different claim from one that produces a paper for ${GEN_COST:.0f}.

The fourth table is what actually happens, because expert capacity is not
unlimited. With {CAPACITY} reviews available, generating {10} gives
{cap[10][3]:.1%} precision and generating {10000:,} gives {cap[10000][3]:.1%} --
because {cap[10000][1] + cap[10000][0] - cap[10000][2]:,} results ship unreviewed.

**Volume overflows the filter and the overflow is the output.** The system's
quality is set by its verification capacity, and generating more does not add
capacity.

The last table is the part that reaches beyond one team, and it is
ch:mcp-production's review-queue result arriving in a new setting. Submitted volume
of {cm[10][0]:,} leaves each reviewer {cm[10][1]:,.0f} items and a catch rate of
{cm[10][2]:.1%}. Volume of {cm[10000][0]:,} leaves {cm[10000][1]:,.0f} each and
{cm[10000][2]:.1%}.

Reviewer attention is a commons. Generating more consumes it and does not
replenish it, so **a pipeline that floods a review system degrades the filter that
every other participant depends on** -- including its own future runs.

Which gives the honest summary of autonomous research pipelines as they currently
stand.

The generation half is real, cheap, and getting cheaper. The verification half is
the binding constraint, does not get cheaper, and is the only part that determines
whether the output is worth anything. **The correct response to cheap generation is
not more generation. It is independent verification that scales**, and nobody has
one.

Until then the defensible deployment is narrow and useful: generate abundantly,
have a human expert verify a small number, and **never let the unverified output
cross a boundary** -- which is ch:aids-agentic-eda's rule, at a larger scale and
with the same reasoning.""")

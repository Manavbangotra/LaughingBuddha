# -*- coding: utf-8 -*-
# Extracted from: Chapter 183 — Code Generation and Completion
# Source: src/.../ch183-code-generation.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Accepting a suggestion is not the same as having working code.

Code completion is measured by acceptance rate: what share of suggestions a
developer keeps. That is a real number and it is not a quality measure, for the
reason ch:aids-text-to-sql found about queries -- the damaging outcome is the one
that looks fine.

A suggestion can be rejected (costs a glance), accepted and correct (saves the
typing), or accepted and subtly wrong (saves the typing and costs a debugging
session later). The third case is the one acceptance rate cannot see
(eq:acceptance-is-not-correctness).

There is a second effect that makes it worse and that this listing measures
separately: generated code is REVIEWED LESS CAREFULLY than written code, because
reading is cheaper than writing and finished-looking code invites less scrutiny
(eq:generated-code-is-under-reviewed).
"""
import numpy as np

rng = np.random.default_rng(4967)

M = 60000
TYPE_TIME = 42.0        # seconds to write the block by hand
GLANCE = 4.0            # seconds to read and reject a suggestion
REVIEW_FULL = 18.0      # seconds to review a suggestion properly
DEBUG_COST = 900.0      # seconds to find and fix a subtle defect later
P_SUGGEST_OK = 0.72     # a suggestion is correct this often


def run(n_blocks=1, m=M, accept_rate=0.30, review_depth=1.0,
        p_ok=P_SUGGEST_OK, catch_full=0.80, type_time=TYPE_TIME,
        review_full=None):
    """One block of code. `review_depth` scales how thoroughly an accepted
    suggestion is reviewed; 1.0 means as carefully as hand-written code.

    Returns (seconds spent, defects escaped per block).
    """
    offered = np.ones(m, dtype=bool)
    correct = rng.random(m) < p_ok
    # A developer accepts more often when the suggestion looks right, but
    # cannot fully distinguish correct from plausible-wrong.
    look_right = correct | (rng.random(m) < 0.55)
    accepted = look_right & (rng.random(m) < accept_rate / 0.72)

    # Reviewing scales with the size of the suggestion, not with a constant.
    rev = REVIEW_FULL * (type_time / TYPE_TIME) if review_full is None         else review_full
    time = np.zeros(m)
    time[~accepted] = GLANCE + type_time
    time[accepted] = GLANCE + rev * review_depth

    # A wrong-but-accepted suggestion is caught in review with a probability
    # proportional to review depth.
    bad = accepted & ~correct
    caught = bad & (rng.random(m) < catch_full * review_depth)
    time[caught] += type_time          # fix it by hand after catching it
    escaped = bad & ~caught
    time[escaped] += DEBUG_COST * 0.0  # the debug cost lands later, counted apart

    return float(time.mean()), float(escaped.mean()), float(accepted.mean())


print(f"One block of code: {TYPE_TIME:.0f}s to write by hand. A suggestion costs")
print(f"{GLANCE:.0f}s to glance at, {REVIEW_FULL:.0f}s to review properly, and is")
print(f"correct {P_SUGGEST_OK:.0%} of the time. A subtle defect that escapes costs")
print(f"{DEBUG_COST:.0f}s later.")
print()
print(f"{'acceptance rate':>17}{'seconds/block':>15}{'defects escaped':>17}"
      f"{'true cost':>12}")
print("-" * 61)
tab = {}
for a in (0.0, 0.15, 0.30, 0.50, 0.70):
    t, e, acc = run(accept_rate=a)
    tab[a] = (t, e, t + e * DEBUG_COST)
    print(f"{a:>17.0%}{t:>15.1f}{e:>17.3f}{t + e * DEBUG_COST:>12.1f}")

print()
print()
print("The same, with review depth falling as suggestions become routine --")
print("which is what happens, because finished-looking code invites less")
print("scrutiny than a blank line does.")
print()
print(f"{'review depth':>14}{'seconds/block':>15}{'defects escaped':>17}"
      f"{'true cost':>12}")
print("-" * 58)
rd = {}
for d in (1.0, 0.7, 0.45, 0.25, 0.1):
    t, e, acc = run(accept_rate=0.30, review_depth=d)
    rd[d] = (t, e, t + e * DEBUG_COST)
    print(f"{d:>14.0%}{t:>15.1f}{e:>17.3f}{t + e * DEBUG_COST:>12.1f}")

print()
print()
print("Apparent speed against true cost. The apparent saving is what a")
print("developer experiences; the true cost includes the debugging.")
print()
base = run(accept_rate=0.0)[0]
print(f"{'acceptance':>12}{'review depth':>14}{'apparent saving':>17}"
      f"{'true saving':>14}")
print("-" * 57)
ap = {}
for a, d in ((0.30, 1.0), (0.30, 0.45), (0.55, 0.45), (0.70, 0.25)):
    t, e, _ = run(accept_rate=a, review_depth=d)
    true = base - (t + e * DEBUG_COST)
    ap[(a, d)] = (base - t, true)
    print(f"{a:>12.0%}{d:>14.0%}{(base - t) / base:>17.1%}"
          f"{true / base:>14.1%}")

print()
print()
print("The break-even: how good a suggestion must be for shallow review to be")
print("worth it, at a 30% acceptance rate.")
print()
print(f"{'suggestion correct':>20}{'deep review':>14}{'shallow review':>17}"
      f"{'better':>10}")
print("-" * 61)
be = {}
for p in (0.55, 0.72, 0.85, 0.95, 0.99):
    deep = run(accept_rate=0.30, review_depth=1.0, p_ok=p)
    shal = run(accept_rate=0.30, review_depth=0.35, p_ok=p)
    dc = deep[0] + deep[1] * DEBUG_COST
    sc = shal[0] + shal[1] * DEBUG_COST
    be[p] = (dc, sc)
    print(f"{p:>20.0%}{dc:>14.1f}{sc:>17.1f}"
          f"{('shallow' if sc < dc else 'deep'):>10}")

print()
print()
print("And the variable that actually decides it: how much typing a")
print("suggestion saves, against what an escaped defect costs. The ratio is")
print("what matters, so this sweeps the size of the suggestion.")
print()
print(f"{'a suggestion saves':>20}{'defect/save ratio':>19}"
      f"{'best acceptance':>17}{'true cost':>12}")
print("-" * 68)
sz = {}
for tt in (12.0, 42.0, 150.0, 600.0):
    best, bv = None, 1e18
    for a in (0.0, 0.15, 0.30, 0.5, 0.7, 0.9):
        for d in (0.25, 0.45, 0.7, 1.0):
            t, e, _ = run(accept_rate=a, review_depth=d, type_time=tt)
            v = t + e * DEBUG_COST
            if v < bv:
                best, bv = (a, d), v
    sz[tt] = (best, bv)
    print(f"{tt:>20.0f}{DEBUG_COST / tt:>19.0f}{best[0]:>17.0%}{bv:>12.1f}")

print()
print()
print("At each size, the best acceptance rate and the review depth that goes")
print("with it.")
print()
print(f"{'a suggestion saves':>20}{'best acceptance':>17}{'best depth':>13}")
print("-" * 50)
for tt in (12.0, 42.0, 150.0, 600.0):
    (a, d), _ = sz[tt]
    print(f"{tt:>20.0f}{a:>17.0%}{d:>13.0%}")

print(f"""
The first table moves the wrong way. True cost rises from {tab[0.0][2]:.1f}
seconds per block at zero acceptance to {tab[0.7][2]:.1f} at
{0.7:.0%} -- **accepting more suggestions costs more in total**, because the
typing saved is smaller than the debugging added
(eq:acceptance-is-not-correctness).

The seconds-per-block column falls the whole way, from {tab[0.0][0]:.1f} to
{tab[0.7][0]:.1f}. That column is what a developer feels. The true-cost column is
what happens.

The second table adds the effect that makes it worse and that nobody meters.
Holding acceptance at {0.30:.0%} and letting review depth fall from
{1.0:.0%} to {0.1:.0%}, apparent time falls from {rd[1.0][0]:.1f} to
{rd[0.1][0]:.1f} seconds and true cost rises from {rd[1.0][2]:.1f} to
{rd[0.1][2]:.1f}.

**Generated code gets reviewed less carefully than written code**
(eq:generated-code-is-under-reviewed), and the reason is not laziness. Reading is
cheaper than writing, so the same amount of attention goes further and feels like
more; and finished-looking code presents as a completed artefact rather than as a
draft. A blank line demands a decision. A plausible function demands agreement.

The third table is the one to carry, because it is the mechanism behind a real
measurement. At {0.70:.0%} acceptance with {0.25:.0%} review depth, the apparent
saving is {ap[(0.70, 0.25)][0] / base:+.1%} and the true saving is
{ap[(0.70, 0.25)][1] / base:+.1%}.

**The two numbers have opposite signs**, and the developer only has access to the
first one. cite:becker2025devproductivity found experienced developers estimating
they had been made 20% faster while measurement showed them 19% slower -- a
39-point error, by people who had just done the work. This table is what that looks
like block by block.

The break-even table says when shallow review is defensible: at
{0.99:.0%} suggestion correctness and not before. At {0.95:.0%} -- which would be a
remarkable model -- deep review still wins.

And the last two tables give the rule that is actually usable, because acceptance
rate is the wrong thing to tune. What decides it is the RATIO of what a suggestion
saves to what a defect costs.

At a ratio of {DEBUG_COST / 12.0:.0f} -- a small suggestion, an expensive
codebase -- the best acceptance rate is {sz[12.0][0][0]:.0%}. At a ratio of
{DEBUG_COST / 600.0:.0f} -- a large mechanical block -- it is
{sz[600.0][0][0]:.0%}.

**Accept long mechanical suggestions and be sceptical of short subtle ones.** The
saving scales with the size of the suggestion and the risk is roughly constant per
acceptance, so the ratio is what matters and the crossover in this table sits
around a defect costing ten times what the suggestion saves.

That is a rule a developer can apply in the moment, which "your acceptance rate
should be 30%" is not.""")

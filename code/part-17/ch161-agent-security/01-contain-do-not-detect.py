# -*- coding: utf-8 -*-
# Extracted from: Chapter 161 — Agent Security and Excessive Agency
# Source: src/.../ch161-agent-security.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Detection is the wrong place to spend. Containment is the right one.

cite:greshake2023indirect's structural finding is that an LLM-integrated
application blurs the line between data and instructions, so content the agent
RETRIEVES can issue commands -- and its abstract notes that effective mitigations
are lacking. That was 2023 and it has aged well.

The instinct is to build a detector: classify incoming content, block the
injections. This listing prices that against the alternative, which is to accept
that some injections will succeed and bound what they can do
(eq:contain-do-not-detect).

The quantity that matters is not the injection success rate. It is the share of
successful injections that produce an IRREVERSIBLE effect, because a reversible
one is an incident and an irreversible one is a loss.
"""
import numpy as np

rng = np.random.default_rng(2617)

N = 400000              # items the agent reads
P_INJECT = 0.004        # share of retrieved content carrying an injection
FP_COST = 1.0           # a blocked-but-benign item is a failed task


def outcomes(detect_rate, fp_rate, policy, n=N):
    """detect_rate: share of injections the classifier catches.
    fp_rate: share of benign items it wrongly blocks.
    policy: what the agent is permitted to do once an injection has landed."""
    injected = rng.random(n) < P_INJECT
    flagged = np.where(injected, rng.random(n) < detect_rate,
                       rng.random(n) < fp_rate)
    landed = injected & ~flagged
    blocked_benign = (~injected) & flagged

    # An injection that lands attempts an action. What it achieves depends on
    # what the agent is allowed to do.
    want_irrev = rng.random(n) < 0.55      # attackers prefer permanent effects
    if policy == "open":
        irrev = landed & want_irrev
        rev = landed & ~want_irrev
    elif policy == "tool_allowlist":
        # Some tools are removed; the attacker uses whatever is left. Removing
        # tools removes some irreversible paths and not all of them.
        irrev = landed & want_irrev & (rng.random(n) < 0.45)
        rev = landed & ~(irrev)
    elif policy == "confirm_irrev":
        # Irreversible actions require a human, who catches most of them.
        irrev = landed & want_irrev & (rng.random(n) >= 0.85)
        rev = landed & ~want_irrev
    elif policy == "reversible_only":
        # The agent simply has no irreversible capability.
        irrev = np.zeros(n, dtype=bool)
        rev = landed
    else:
        raise ValueError(policy)
    return (float(landed.mean()), float(irrev.mean()), float(rev.mean()),
            float(blocked_benign.mean()))


print(f"{N:,} retrieved items, {P_INJECT:.1%} carrying an injection.")
print("An injection that lands attempts an action; 55% of attackers want a")
print("permanent effect.")
print()
print("First: what a detector alone buys, at an open permission model.")
print()
print(f"{'detector':>10}{'false':>9}{'landed':>10}{'irreversible':>14}"
      f"{'benign blocked':>16}")
print(f"{'recall':>10}{'positives':>9}{'':>10}{'':>14}{'':>16}")
print("-" * 59)
det = {}
for d, f in ((0.0, 0.0), (0.50, 0.005), (0.80, 0.02), (0.95, 0.06),
             (0.99, 0.15)):
    r = outcomes(d, f, "open")
    det[d] = r
    print(f"{d:>10.0%}{f:>9.1%}{r[0]:>10.3%}{r[1]:>14.3%}{r[3]:>16.2%}")

print()
print()
print("Now hold the detector at a realistic 80% and change the permission model")
print("instead. Same injections, same detector, different blast radius.")
print()
print(f"{'permission model':>24}{'landed':>10}{'irreversible':>14}"
      f"{'reversible':>13}{'irrev share':>13}")
print("-" * 74)
pol = {}
for name, p in [("open", "open"), ("tool allowlist", "tool_allowlist"),
                ("confirm irreversible", "confirm_irrev"),
                ("no irreversible tools", "reversible_only")]:
    r = outcomes(0.80, 0.02, p)
    pol[name] = r
    share = r[1] / r[0] if r[0] > 0 else 0.0
    print(f"{name:>24}{r[0]:>10.3%}{r[1]:>14.3%}{r[2]:>13.3%}{share:>13.0%}")

print()
print()
print("Which is the better place to spend? Compare improving the detector")
print("against changing the permission model, on irreversible effects.")
print()
print(f"{'change':>40}{'irreversible':>14}{'vs baseline':>13}"
      f"{'benign blocked':>16}")
print("-" * 83)
base = outcomes(0.80, 0.02, "open")
moves = {}
for name, args in [
        ("baseline: detector 80%, open", (0.80, 0.02, "open")),
        ("detector 80% -> 95%", (0.95, 0.06, "open")),
        ("detector 80% -> 99%", (0.99, 0.15, "open")),
        ("keep 80%, confirm irreversible", (0.80, 0.02, "confirm_irrev")),
        ("keep 80%, remove irreversible tools", (0.80, 0.02, "reversible_only")),
        ("no detector, remove irrev tools", (0.0, 0.0, "reversible_only"))]:
    r = outcomes(*args)
    moves[name] = r
    print(f"{name:>40}{r[1]:>14.3%}{r[1] - base[1]:>+13.3%}{r[3]:>16.2%}")

print()
print()
print("The cost of detection nobody prices: benign items wrongly blocked, which")
print("are failed tasks. Sweep the detector's operating point.")
print()
print(f"{'recall':>9}{'false pos':>11}{'irrev prevented':>18}"
      f"{'tasks broken':>15}{'ratio':>10}")
print("-" * 63)
open0 = outcomes(0.0, 0.0, "open")
for d, f in ((0.50, 0.005), (0.80, 0.02), (0.95, 0.06), (0.99, 0.15)):
    r = outcomes(d, f, "open")
    prevented = open0[1] - r[1]
    print(f"{d:>9.0%}{f:>11.1%}{prevented:>18.3%}{r[3]:>15.2%}"
          f"{r[3] / max(prevented, 1e-12):>10.0f}")

print()
print()
print("And how the two approaches scale as injection prevalence rises.")
print()
print(f"{'injection rate':>16}{'detector 95%':>14}{'no irrev tools':>17}"
      f"{'both':>9}")
print("-" * 56)
PI_SAVE = P_INJECT
prev = {}
for pi in (0.001, 0.004, 0.02, 0.10):
    globals()["P_INJECT"] = pi
    a = outcomes(0.95, 0.06, "open")[1]
    b = outcomes(0.0, 0.0, "reversible_only")[1]
    c = outcomes(0.95, 0.06, "reversible_only")[1]
    prev[pi] = (a, b, c)
    print(f"{pi:>16.1%}{a:>14.3%}{b:>17.3%}{c:>9.3%}")
globals()["P_INJECT"] = PI_SAVE

print(f"""
The first table is the detector on its own, and it works: recall {0.8:.0%} takes
landed injections from {det[0.0][0]:.3%} to {det[0.8][0]:.3%}, and {0.99:.0%}
takes them to {det[0.99][0]:.3%}. Nobody should claim detection is useless.

The last column is what that costs. At {0.99:.0%} recall the classifier is also
blocking {det[0.99][3]:.1%} of BENIGN items -- one task in seven, failed, because a
security control decided the content looked suspicious. Detectors for a rare,
adversarially-chosen signal sit on a steep part of the ROC curve, and the false
positives are not free: they are broken tasks.

The second table holds the detector at a realistic {0.8:.0%} and changes what the
agent is ALLOWED TO DO instead. The landed column barely moves -- the same
injections get through -- and the irreversible column goes
{pol['open'][1]:.3%}, {pol['tool allowlist'][1]:.3%},
{pol['confirm irreversible'][1]:.3%}, {pol['no irreversible tools'][1]:.3%}.

The share of successful injections that achieve something permanent falls from
{pol['open'][1] / pol['open'][0]:.0%} to {0:.0%}. **The attacker still wins the
argument with the model and stops being able to do anything that matters**
(eq:contain-do-not-detect).

The third table is the comparison that decides where to spend, and the last row is
the result.

Taking the detector from {0.8:.0%} to {0.99:.0%} recall reduces irreversible
effects by {base[1] - moves['detector 80% -> 99%'][1]:.3%} and blocks
{moves['detector 80% -> 99%'][3]:.1%} of benign traffic. Keeping the {0.8:.0%}
detector and removing the irreversible capability reduces them by
{base[1] - moves['keep 80%, remove irreversible tools'][1]:.3%} -- more -- and
blocks the same {moves['keep 80%, remove irreversible tools'][3]:.1%} it was
blocking before.

And the bottom row: NO detector at all, with irreversible capabilities removed,
achieves {moves['no detector, remove irrev tools'][1]:.3%} irreversible effects
while blocking {moves['no detector, remove irrev tools'][3]:.1%} of benign
traffic.

**Containment strictly dominates detection here** -- better on the outcome that
matters and free on the cost that detection pays. That is not an argument for
having no detector; it is an argument about which one to build first, and the
usual order is backwards.

The fourth table prices the detector's false positives against what it prevents,
which is the calculation nobody runs. At {0.5:.0%} recall you break {5} tasks per
irreversible action prevented. At {0.99:.0%} you break {69}.

That ratio is the honest cost of a detector-first strategy, and it worsens as you
tighten the detector -- which is the direction a security team under pressure will
always push it. **A control whose cost rises faster than its benefit as you tune it
is a control you should be reluctant to rely on.**

The last table is the argument that settles it, and it is about what happens when
someone is trying.

As injection prevalence rises from {0.001:.1%} to {0.10:.0%} -- an attacker
finding more places to plant content -- the {0.95:.0%} detector's irreversible
effects rise from {prev[0.001][0]:.3%} to {prev[0.10][0]:.3%}, roughly linearly.
The containment column stays at {prev[0.10][1]:.3%} throughout.

**Detection degrades with attacker effort and containment does not.** A detector is
a classifier against an adversary who can iterate on its inputs, which is the worst
situation a classifier can be in -- cite:greshake2023indirect's demonstrations
against production systems are exactly that. Containment does not care how many
injections land, because it changed what landing achieves.

So the design order this listing supports, which is not the usual one:

**First, remove irreversible capabilities the agent does not need.** It costs
nothing, it is invariant to attacker effort, and it is the only control here with
no false-positive cost.

**Second, gate what remains on reversibility**, per ch:ag-termination -- a small,
bounded set that fits inside a reviewer's attention.

**Third, add a detector**, for the reversible-but-costly middle and for telemetry
about what is being attempted. Tune it conservatively, because the fourth table's
ratio is the price of tuning it otherwise.

The framing to carry: an injection is a request the agent will grant. The question
is never whether it will be granted, because the abstract of
cite:greshake2023indirect says mitigations are lacking and nothing since has
changed that. The question is what granting it can accomplish.""")

# -*- coding: utf-8 -*-
# Extracted from: Chapter 183 — Code Generation and Completion
# Source: src/.../ch183-code-generation.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Where a coding task's time goes, and why speeding up typing does so little.

cite:becker2025devproductivity ran a randomised controlled trial: 16 experienced
open-source developers, 246 tasks, mature repositories they averaged five years
on. Developers forecast AI would make them 24% faster. Afterwards they estimated
it had made them 20% faster. It made them 19% SLOWER.

That is three numbers and the interesting one is the gap between the second and
the third: a 39-point error in self-assessment, by people who had just done the
work.

This listing decomposes a coding task and asks which arrangement of effects
reproduces that pattern (eq:writing-is-a-small-share). The structure is
ch:aids-stack's: a stage that gets automated is a stage that was small, and the
automation adds work elsewhere.
"""
import numpy as np

# Minutes per task, by stage, for an experienced developer on a familiar
# codebase. These are this listing's assumptions, stated so they can be checked.
# (stage, baseline minutes, AI multiplier, is the change VISIBLE to the developer)
STAGES = [
    ("understand the issue",   18.0, 0.95, True),
    ("locate the change",      22.0, 0.80, True),
    ("write the code",         16.0, 0.45, True),
    ("review what was written", 9.0, 1.60, False),
    ("get it actually working", 31.0, 1.35, False),
    ("integrate and land it",  14.0, 1.05, False),
]

BASE = sum(s[1] for s in STAGES)

print("An experienced developer's task on a codebase they know, by stage.")
print()
print(f"{'stage':>26}{'minutes':>10}{'share':>8}{'with AI':>10}{'change':>10}"
      f"{'visible?':>11}")
print("-" * 75)
for name, base, mult, vis in STAGES:
    print(f"{name:>26}{base:>10.0f}{base / BASE:>8.0%}{base * mult:>10.1f}"
          f"{base * (mult - 1):>+10.1f}{('yes' if vis else 'no'):>11}")

total_ai = sum(s[1] * s[2] for s in STAGES)
print("-" * 75)
print(f"{'total':>26}{BASE:>10.0f}{1.0:>8.0%}{total_ai:>10.1f}"
      f"{total_ai - BASE:>+10.1f}")
print()
print(f"   measured effect: {total_ai / BASE - 1:+.0%} on task time")

print()
print()
print("What the developer EXPERIENCES: only the visible stages, which are the")
print("ones where the tool is doing something in front of them.")
print()
vis_base = sum(s[1] for s in STAGES if s[3])
vis_ai = sum(s[1] * s[2] for s in STAGES if s[3])
inv_base = sum(s[1] for s in STAGES if not s[3])
inv_ai = sum(s[1] * s[2] for s in STAGES if not s[3])
print(f"{'':>26}{'baseline':>11}{'with AI':>10}{'change':>10}")
print("-" * 57)
print(f"{'visible stages':>26}{vis_base:>11.0f}{vis_ai:>10.1f}"
      f"{vis_ai / vis_base - 1:>+10.0%}")
print(f"{'invisible stages':>26}{inv_base:>11.0f}{inv_ai:>10.1f}"
      f"{inv_ai / inv_base - 1:>+10.0%}")
print(f"{'all stages':>26}{BASE:>11.0f}{total_ai:>10.1f}"
      f"{total_ai / BASE - 1:>+10.0%}")
print()
print(f"   Self-estimate, if only visible stages register: "
      f"{vis_ai / vis_base - 1:+.0%}")
print(f"   Measured:                                       "
      f"{total_ai / BASE - 1:+.0%}")
print(f"   Gap:                                            "
      f"{abs(vis_ai / vis_base - total_ai / BASE):.0%} points")

print()
print()
print("Amdahl on the stage everyone means by 'AI coding'. Perfect automation")
print("of writing, and nothing else changed:")
print()
write = next(s for s in STAGES if s[0] == "write the code")
print(f"{'writing time':>26}{'total':>10}{'speedup':>10}")
print("-" * 46)
for mult, label in ((1.0, "unchanged"), (0.45, "as measured"),
                    (0.10, "near-perfect"), (0.0, "free")):
    t = BASE - write[1] * (1 - mult)
    print(f"{label:>26}{t:>10.1f}{BASE / t:>10.2f}x")

print()
print()
print("The two effects separated. 'Assistance' is the speedup on the stages AI")
print("helps; 'friction' is the slowdown on the ones it does not.")
print()
print(f"{'friction multiplier':>21}{'total':>10}{'effect':>10}{'verdict':>12}")
print("-" * 53)
fr = {}
for f in (1.0, 1.1, 1.25, 1.35, 1.6):
    t = sum(s[1] * (s[2] if s[3] else 1.0 + (s[2] - 1.0) * (f - 1.0) / 0.35)
            for s in STAGES)
    t = sum(s[1] * s[2] if s[3] else s[1] * f for s in STAGES)
    fr[f] = (t, t / BASE - 1)
    print(f"{f:>21.2f}{t:>10.1f}{t / BASE - 1:>+10.0%}"
          f"{('faster' if t < BASE else 'slower'):>12}")

print()
print()
print("And where the effect flips. The break-even friction, and what it implies")
print("about which developers and codebases benefit.")
print()
lo, hi = 1.0, 2.0
for _ in range(60):
    mid = (lo + hi) / 2
    t = sum(s[1] * s[2] if s[3] else s[1] * mid for s in STAGES)
    if t < BASE:
        lo = mid
    else:
        hi = mid
breakeven = (lo + hi) / 2
print(f"   break-even friction multiplier: {breakeven:.2f}")
print(f"   i.e. the tool pays off if it adds less than "
      f"{(breakeven - 1) * 100:.0f}% to review, debugging and integration.")
print()
print(f"{'setting':>34}{'assistance':>12}{'friction':>10}{'effect':>10}")
print("-" * 66)
SETTINGS = [
    ("expert, mature familiar codebase", 0.45, 1.35),
    ("expert, unfamiliar codebase", 0.45, 1.10),
    ("novice, any codebase", 0.35, 1.05),
    ("greenfield, few constraints", 0.25, 1.00),
]
st = {}
for label, wmult, f in SETTINGS:
    t = 0.0
    for name, base, mult, vis in STAGES:
        if name == "write the code":
            t += base * wmult
        elif vis:
            t += base * mult
        else:
            t += base * f
    st[label] = t / BASE - 1
    print(f"{label:>34}{wmult:>12.2f}{f:>10.2f}{t / BASE - 1:>+10.0%}")

print(f"""
The share column is the first finding and it is ch:aids-stack's exactly. **Writing
the code is {16 / BASE:.0%} of the task.** Understanding, locating, debugging and
integrating are the other {1 - 16 / BASE:.0%}.

So the Amdahl table is unsurprising once the share is known: making writing FREE
gives a {BASE / 94.0:.2f}x speedup on the task. Not free -- free. The stage that
"AI coding" means is the stage that was already small.

The second table is the mechanism this listing was built for.

Visible stages -- the ones where the tool is doing something on screen -- improve
by {vis_ai / vis_base - 1:.0%}. Invisible stages -- reviewing, getting it actually
working, landing it -- worsen by {inv_ai / inv_base - 1:+.0%}. The total is
{total_ai / BASE - 1:+.0%}.

**A developer who registers the visible stages would report being
{abs(vis_ai / vis_base - 1):.0%} faster while being {total_ai / BASE - 1:+.0%}
slower**, a gap of {abs(vis_ai / vis_base - total_ai / BASE) * 100:.0f} points.

cite:becker2025devproductivity measured that gap at 39 points -- self-estimate
-20%, measurement +19% -- and this decomposition reproduces its shape from
plausible per-stage assumptions. Note the honest discrepancy: this listing's
parameters produce {total_ai / BASE - 1:+.0%} where the trial measured +19%, which
means the real friction on the invisible stages was **larger** than assumed here,
not smaller.

The reason the gap exists at all is that the two effects land in different places.
The assistance is concentrated, immediate and observed. The friction is diffuse,
delayed, and indistinguishable from the ordinary difficulty of software. A
developer debugging for forty minutes does not experience twelve of those minutes
as attributable to the suggestion they accepted an hour ago.

**The self-report is not dishonest. It is a faithful report of the visible half.**

The friction table locates the boundary. The break-even multiplier is
{breakeven:.2f}: **the tool pays off if it adds less than
{(breakeven - 1) * 100:.0f}% to review, debugging and integration** and costs time
if it adds more.

Which makes the last table the important one, because it says who is on which side.

An expert on a mature codebase they know well: {st['expert, mature familiar codebase']:+.0%}.
The same expert on unfamiliar code: {st['expert, unfamiliar codebase']:+.0%}. A
novice: {st['novice, any codebase']:+.0%}. Greenfield work with few constraints:
{st['greenfield, few constraints']:+.0%}.

**The effect changes sign across settings**, and it does so for a legible reason:
assistance is worth most where the developer's own writing speed is the constraint,
and friction is worst where the code has many constraints the developer knows and
the tool does not.

cite:becker2025devproductivity studied experienced open-source developers on mature
repositories they averaged five years on -- **the least favourable cell in this
table**, and also the cell where the strongest claims are usually made. That is not
a criticism of the study; it is the reason its result is important and the reason
it does not generalise to every setting.

The practical readings are three.

**Do not infer the effect from how it feels.** The felt quantity and the measured
quantity have different signs in some settings and nobody can tell from inside.

**Expect the benefit where writing is the constraint** -- unfamiliar APIs,
boilerplate, greenfield, novices -- and expect friction where the constraint is
knowing what the code must not break.

**And measure it on your own work**, because the parameters that decide this are
per-team and the sign flips inside the plausible range.""")

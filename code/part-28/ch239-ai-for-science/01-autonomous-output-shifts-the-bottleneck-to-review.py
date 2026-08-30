# -*- coding: utf-8 -*-
# Extracted from: Chapter 239 — AI for Science and Autonomous Research
# Source: src/.../ch239-ai-for-science.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Automating research moves the bottleneck to review, and review is capacity-bound.

An automated research system produces candidate findings cheaply (cite:lu2024aiscientist,
cite:chan2024mlebench). Each candidate is worth something only once somebody has established
that it is true, and establishing that costs expert time that does not scale with the generator.

So the throughput of the whole enterprise is not set by how many findings are produced. It is
set by how many can be verified, and automation raises the numerator while leaving the
denominator alone (eq:autonomous-output-shifts-the-bottleneck-to-review).

Which changes what a "finding" is worth. An unverified candidate has a value equal to its
probability of surviving verification, minus the cost of the verification it will consume
(eq:a-finding-is-worth-its-verification-probability).
"""
BASE_TRUE = 0.11          # share of generated candidates that would survive verification
REVIEWERS = 6
REVIEW_HOURS_PER_WEEK = 8.0
WEEKS = 52


# (stage, hours of expert time per candidate, share of false candidates it removes)
GATES = [
    ("automated sanity checks",    0.00,  0.31),
    ("read the claim and method",  0.35,  0.44),
    ("re-run the analysis",        1.80,  0.62),
    ("independent replication",   14.00,  0.88),
]

print("What it costs to establish that a candidate finding is true.")
print()
print(f"{'verification stage':>28}{'expert hours':>15}{'false removed':>16}"
      f"{'candidates / reviewer-year':>29}")
print("-" * 88)
CAPACITY = REVIEWERS * REVIEW_HOURS_PER_WEEK * WEEKS
for name, hours, removed in GATES:
    per_year = CAPACITY / hours if hours > 0 else float("inf")
    ps = f"{per_year:>29,.0f}" if hours > 0 else f"{'unbounded':>29}"
    print(f"{name:>28}{hours:>15.2f}{removed:>16.0%}{ps}")

print()
print(f"{REVIEWERS} reviewers at {REVIEW_HOURS_PER_WEEK:.0f} hours a week is"
      f" {CAPACITY:,.0f} expert-hours a year")

print()
print()
print("Now scale the generator and watch the bottleneck move.")
print()
FULL_STACK = sum(h for n, h, r in GATES)
print(f"{'candidates / year':>19}{'expert-hours needed':>22}{'capacity':>12}"
      f"{'share verifiable':>19}{'verified true findings':>25}")
print("-" * 97)
scaled = {}
for gen in (40, 400, 4_000, 40_000, 400_000):
    need = gen * FULL_STACK
    share = min(1.0, CAPACITY / need)
    true_found = gen * share * BASE_TRUE
    scaled[gen] = (need, share, true_found)
    print(f"{gen:>19,}{need:>22,.0f}{CAPACITY:>12,.0f}"
          f"{share:>19.1%}{true_found:>25,.1f}")

print()
print(f"a {400_000 // 40:,}x increase in generation yields"
      f" {scaled[400_000][2] / scaled[40][2]:.1f}x the verified findings")
print("(eq:autonomous-output-shifts-the-bottleneck-to-review)")

print()
print()
print("So the question is which gate to spend the hours on.")
print()
print(f"{'policy':>34}{'hours / candidate':>20}{'candidates cleared':>21}"
      f"{'true findings':>16}{'false findings':>17}")
print("-" * 108)
GEN = 40_000
POLICIES = [
    ("full stack on everything",       FULL_STACK,  1.00),
    ("read + re-run, replicate 10%",   0.35 + 1.80 + 0.10 * 14.00, 0.10),
    ("read + re-run only",             0.35 + 1.80, 0.00),
    ("read only",                      0.35,        0.00),
    ("automated checks only",          0.00,        0.00),
]
pol = {}
for name, hours, rep_share in POLICIES:
    cleared = GEN if hours == 0 else min(GEN, CAPACITY / hours)
    kept_false_rate = (1 - 0.31) * (1 - 0.44 if hours >= 0.35 else 1.0)
    if hours >= 2.0:
        kept_false_rate *= (1 - 0.62)
    kept_false_rate *= (1 - 0.88 * rep_share)
    true_out = cleared * BASE_TRUE
    false_out = cleared * (1 - BASE_TRUE) * kept_false_rate
    pol[name] = (cleared, true_out, false_out)
    print(f"{name:>34}{hours:>20.2f}{cleared:>21,.0f}"
          f"{true_out:>16,.0f}{false_out:>17,.0f}")

print()
best_p = max(pol, key=lambda n: pol[n][1] - 3.0 * pol[n][2])
print(f"at a 3:1 penalty for a false finding, the best policy is `{best_p}`")
print(f"it yields {pol[best_p][1]:,.0f} true and {pol[best_p][2]:,.0f} false")

print()
print()
print("What a single unverified candidate is actually worth.")
print()
VALUE_TRUE, COST_FALSE = 24_000.0, 9_000.0
HOUR_COST = 190.0
print(f"{'prior that it is true':>24}{'value if true':>16}{'cost if false':>16}"
      f"{'verification cost':>20}{'net worth':>13}")
print("-" * 89)
worth = {}
for prior in (0.02, 0.05, 0.11, 0.25, 0.50, 0.80):
    ver = FULL_STACK * HOUR_COST
    net = prior * VALUE_TRUE - (1 - prior) * COST_FALSE * 0.12 - ver
    worth[prior] = net
    print(f"{prior:>24.2f}{VALUE_TRUE:>16,.0f}{COST_FALSE:>16,.0f}"
          f"{ver:>20,.0f}{net:>13,.0f}")

break_even = None
for prior in sorted(worth):
    if worth[prior] > 0 and break_even is None:
        break_even = prior
print()
print(f"a candidate is worth verifying above a prior of about {break_even:.2f}")
print(f"the generator's base rate is {BASE_TRUE:.2f}")
print("(eq:a-finding-is-worth-its-verification-probability)")

print()
print()
print("Which makes triage worth more than generation.")
print()
TRIAGE = [
    ("no triage",                        BASE_TRUE, 0.000, "--"),
    ("automated plausibility score",     0.19,      0.004, "cheap"),
    ("cross-check against literature",   0.28,      0.031, "retrieval"),
    ("cheap pilot experiment",           0.46,      0.310, "compute"),
    ("expert 10-minute screen",          0.51,      0.310, "the scarce resource"),
]
print(f"{'triage step':>34}{'prior after triage':>21}{'cost / candidate':>19}"
      f"{'net worth after':>18}{'uses':>21}")
print("-" * 113)
for name, prior, cost, uses in TRIAGE:
    ver = FULL_STACK * HOUR_COST
    net = prior * VALUE_TRUE - (1 - prior) * COST_FALSE * 0.12 - ver - cost * HOUR_COST
    print(f"{name:>34}{prior:>21.2f}{cost * HOUR_COST:>19,.0f}"
          f"{net:>18,.0f}{uses:>21}")

print()
print("Three of the four triage steps cost no expert time at all.")

print(f"""
The verification table is the constraint the whole subject runs into. Establishing that one
candidate finding is true costs {FULL_STACK:.2f} expert-hours through the full stack, and
{REVIEWERS} reviewers at {REVIEW_HOURS_PER_WEEK:.0f} hours a week supply
{CAPACITY:,.0f} hours a year -- about {CAPACITY / FULL_STACK:,.0f} candidates.

The scaling table is what happens when the generator gets good
(eq:autonomous-output-shifts-the-bottleneck-to-review). At {40:,} candidates a year everything is
verified and {scaled[40][2]:.1f} findings are established. At {400:,}, capacity binds and
{scaled[400][2]:.1f} are established. At {4_000:,}, {scaled[4000][2]:.1f}. At {400_000:,},
{scaled[400_000][2]:.1f}.

**Beyond a few hundred candidates a year, additional generation produces exactly zero additional
established findings.** The share verified falls from {scaled[400][1]:.1%} to
{scaled[400_000][1]:.1%} and the numerator does not move, because the numerator was never the
generator's.

That is the number that should frame every claim about automated research. **The generator is not
the bottleneck and has not been for some time.** A system that produces ten thousand plausible
hypotheses and a system that produces ten produce the same number of established findings, if the
review capacity is the same.

It is also ch:sec-permissions' approval queue and ch:rai-oversight's review budget arriving in a
third setting, with the same structure: a fixed human capacity meeting a scalable producer, and
the same conclusion -- **the interesting decision is what to spend the capacity on.**

The policy table is that decision. `full stack on everything` clears
{pol['full stack on everything'][0]:,.0f} candidates and produces
{pol['full stack on everything'][1]:,.0f} true findings.
`{best_p}` clears {pol[best_p][0]:,.0f} and produces {pol[best_p][1]:,.0f} true and
{pol[best_p][2]:,.0f} false.

Note which direction that goes, because it inverts the usual advice. Elsewhere in this book --
ch:ev-framework's gate placement, ch:sec-permissions' approval queue -- cheap checks applied
widely beat thorough checks applied narrowly. Here they do not: `read only` clears
{pol['read only'][0]:,.0f} candidates and produces {pol['read only'][1]:,.0f} true findings
alongside **{pol['read only'][2]:,.0f} false ones.**

The difference is the base rate. When {1 - BASE_TRUE:.0%} of candidates are wrong, a gate that
removes most but not all of them still passes more false findings than true ones, and a false
finding entered into the literature costs more than a missed true one. **A low base rate makes
thoroughness the correct policy**, which is the same arithmetic ch:sec-jailbreaks used for
guardrail precision, pointing the other way because the costs are asymmetric in the other
direction.

The worth table is the per-candidate view and it produces the sharpest number here. At a prior of
{BASE_TRUE:.2f} -- the generator's base rate -- a candidate is worth
**{worth[0.11]:,.0f}** to verify. It only becomes worth verifying above a prior of about
{break_even:.2f} (eq:a-finding-is-worth-its-verification-probability).

**The average candidate from an automated generator is not worth the expert time it would
consume.** That is not an argument against automated research; it is an argument that the
generator's output must be triaged before it reaches a human, and that the triage is the valuable
component.

The triage table prices exactly that. `cross-check against literature` raises the prior from
{BASE_TRUE:.2f} to {0.28:.2f} at {0.031 * HOUR_COST:.0f} per candidate and no expert time.
`cheap pilot experiment` reaches {0.46:.2f} for {0.310 * HOUR_COST:.0f} of compute.

An `expert 10-minute screen` reaches {0.51:.2f} -- barely better than the pilot -- and spends the
one resource that is capacity-bound. **Three of the four triage steps cost no expert time at
all**, which is where the leverage in autonomous research actually sits: not in generating more
candidates, and not in reviewing faster, but in raising the prior of what reaches the reviewer.""")

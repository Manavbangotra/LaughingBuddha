# -*- coding: utf-8 -*-
# Extracted from: Chapter 239 — AI for Science and Autonomous Research
# Source: src/.../ch239-ai-for-science.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The experiments worth automating are the ones nobody wants to run.

The first listing found that review capacity, not generation, bounds automated research. This one
asks a different question: given a fixed budget of experiments, which ones are worth running?

Information theory gives a clean answer. An experiment's information content is the entropy of
its outcome, so an experiment whose result is nearly certain teaches almost nothing however
important its subject. Divide by cost and the ranking is not the one the field's incentives
produce (eq:replication-has-higher-information-per-dollar-than-novelty).

And the largest single loss is structural. A negative result that is not published is repeated,
independently, by everyone else who would have had the idea
(eq:an-unpublished-negative-is-repeated-by-everyone-else).
"""
import math


def entropy(p):
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


# (experiment type, cost in dollars, P(the interesting outcome), value multiplier,
#  cost multiple once automated)
EXPERIMENTS = [
    ("a bold novel hypothesis",       24_000.0, 0.11, 6.0, 0.62),
    ("an incremental variation",       9_000.0, 0.62, 2.2, 0.31),
    ("direct replication",             6_000.0, 0.55, 1.0, 0.14),
    ("an ablation of one component",   2_200.0, 0.48, 1.4, 0.09),
    ("reproduction from artefacts",    1_400.0, 0.72, 1.1, 0.05),
    ("a benchmark rerun",                700.0, 0.86, 0.6, 0.03),
]

print("What each kind of experiment teaches, per dollar.")
print()
print(f"{'experiment':>32}{'cost':>11}{'P(interesting)':>17}{'bits':>9}"
      f"{'bits per $1k':>15}{'value per $1k':>16}")
print("-" * 100)
rows = {}
for name, cost, p, mult, auto in EXPERIMENTS:
    h = entropy(p)
    bits_k = h / (cost / 1000.0)
    val_k = h * mult / (cost / 1000.0)
    rows[name] = (cost, p, h, bits_k, val_k, mult, auto)
    print(f"{name:>32}{cost:>11,.0f}{p:>17.2f}{h:>9.4f}"
          f"{bits_k:>15.4f}{val_k:>16.4f}")

BEST_BITS = max(rows, key=lambda n: rows[n][3])
BEST_VALUE = max(rows, key=lambda n: rows[n][4])
NOVEL = "a bold novel hypothesis"
print()
print(f"most bits per dollar: {BEST_BITS} at {rows[BEST_BITS][3]:.4f}")
print(f"most value per dollar: {BEST_VALUE} at {rows[BEST_VALUE][4]:.4f}")
print(f"a bold novel hypothesis: {rows[NOVEL][4]:.4f}"
      f" -- {rows[BEST_VALUE][4] / rows[NOVEL][4]:.1f}x worse")
print("(eq:replication-has-higher-information-per-dollar-than-novelty)")

print()
print()
print("What the field actually spends its effort on.")
print()
ACTUAL = {
    "a bold novel hypothesis":     0.34,
    "an incremental variation":    0.48,
    "direct replication":          0.04,
    "an ablation of one component": 0.11,
    "reproduction from artefacts": 0.02,
    "a benchmark rerun":           0.01,
}
BUDGET = 6_000_000.0
print(f"budget {BUDGET:,.0f}")
print()
print(f"{'experiment':>32}{'actual share':>15}{'experiments run':>18}"
      f"{'value delivered':>18}{'value per $1k':>16}")
print("-" * 99)
actual_value = 0.0
for name, cost, p, mult, auto in EXPERIMENTS:
    share = ACTUAL[name]
    n = BUDGET * share / cost
    v = n * entropy(p) * mult
    actual_value += v
    print(f"{name:>32}{share:>15.0%}{n:>18,.0f}{v:>18,.1f}{rows[name][4]:>16.4f}")
print("-" * 99)
print(f"{'TOTAL':>32}{1.0:>15.0%}{'':>18}{actual_value:>18,.1f}")

opt_n = BUDGET / rows[BEST_VALUE][0]
opt_value = opt_n * rows[BEST_VALUE][2] * rows[BEST_VALUE][5]
print()
print(f"all-in on {BEST_VALUE}: {opt_n:,.0f} experiments, {opt_value:,.1f} value")
print(f"a factor of {opt_value / actual_value:.1f} over the observed allocation")

print()
print()
print("Which is not a recommendation, because value is not fungible.")
print()
CAPS = {
    "a bold novel hypothesis":     1.00,
    "an incremental variation":    0.55,
    "direct replication":          0.30,
    "an ablation of one component": 0.22,
    "reproduction from artefacts": 0.14,
    "a benchmark rerun":           0.08,
}
MIN_NOVEL = 0.25
print(f"reserving {MIN_NOVEL:.0%} for novelty, then filling by value per dollar")
print()
print(f"{'experiment':>32}{'value per $1k':>16}{'share cap':>12}"
      f"{'budget taken':>16}{'value delivered':>18}")
print("-" * 94)
remaining, port_value = 1.0 - MIN_NOVEL, 0.0
n_exp = BUDGET * MIN_NOVEL / rows[NOVEL][0]
nov_value = n_exp * rows[NOVEL][2] * rows[NOVEL][5]
port_value += nov_value
print(f"{NOVEL:>32}{rows[NOVEL][4]:>16.4f}{'reserved':>12}"
      f"{BUDGET * MIN_NOVEL:>16,.0f}{nov_value:>18,.1f}")
for name in sorted(rows, key=lambda n: -rows[n][4]):
    if name == NOVEL:
        continue
    take = min(CAPS[name], remaining)
    remaining -= take
    n_exp = BUDGET * take / rows[name][0]
    v = n_exp * rows[name][2] * rows[name][5]
    port_value += v
    print(f"{name:>32}{rows[name][4]:>16.4f}{CAPS[name]:>12.0%}"
          f"{BUDGET * take:>16,.0f}{v:>18,.1f}")
print("-" * 94)
USED = 1.0 - remaining
print(f"{'TOTAL':>32}{'':>16}{USED:>12.0%}{BUDGET * USED:>16,.0f}"
      f"{port_value:>18,.1f}")

print()
print(f"a capped portfolio delivers {port_value:,.1f}"
      f" against the observed {actual_value:,.1f}")
print(f"a factor of {port_value / actual_value:.1f}, with no new capability required")

print()
print()
print("Now the loss nobody accounts for: the negative that is not published.")
print()
GROUPS = 40
print(f"{'groups who would try it':>26}{'P(each tries)':>16}{'expected repeats':>19}"
      f"{'wasted cost':>15}{'if published':>15}")
print("-" * 91)
waste = {}
for p_try in (0.02, 0.05, 0.12, 0.25, 0.50):
    repeats = GROUPS * p_try
    cost = repeats * rows[NOVEL][0]
    waste[p_try] = cost
    print(f"{GROUPS:>26}{p_try:>16.2f}{repeats:>19.1f}"
          f"{cost:>15,.0f}{rows[NOVEL][0]:>15,.0f}")

print()
print(f"at {0.12:.0%} the field spends {waste[0.12]:,.0f} to learn something")
print(f"one group already knew, and would have shared for {rows[NOVEL][0]:,.0f}")
print(f"a waste multiple of {waste[0.12] / rows[NOVEL][0]:.1f}x")
print("(eq:an-unpublished-negative-is-repeated-by-everyone-else)")

print()
print()
print("And what automation actually changes.")
print()
print(f"{'experiment':>32}{'cost now':>12}{'cost automated':>17}"
      f"{'reduction':>12}{'value per $1k, automated':>27}")
print("-" * 100)
auto_rows = {}
for name, cost, p, mult, auto in EXPERIMENTS:
    new_cost = cost * auto
    v = entropy(p) * mult / (new_cost / 1000.0)
    auto_rows[name] = (new_cost, v)
    print(f"{name:>32}{cost:>12,.0f}{new_cost:>17,.0f}"
          f"{1 / auto:>11.1f}x{v:>27.4f}")

best_auto = max(auto_rows, key=lambda n: auto_rows[n][1])
print()
print(f"automation reduces `{NOVEL}` cost by {1 / rows[NOVEL][6]:.1f}x")
print(f"and `a benchmark rerun` by {1 / rows['a benchmark rerun'][6]:.1f}x")
print(f"the gap in value per dollar widens from"
      f" {rows[BEST_VALUE][4] / rows[NOVEL][4]:.1f}x to"
      f" {auto_rows[best_auto][1] / auto_rows[NOVEL][1]:.1f}x")

print(f"""
The first table is the ranking the field's incentives do not produce. Measured in bits of outcome
entropy per dollar, `{BEST_BITS}` leads at {rows[BEST_BITS][3]:.4f} and `{NOVEL}` trails at
{rows[NOVEL][3]:.4f}.

The obvious objection is that bits are not value -- a novel result matters more than a benchmark
rerun -- so the table carries a value multiplier: {rows[NOVEL][5]:.1f} for novelty against
{rows['a benchmark rerun'][5]:.1f} for a rerun. **It does not change the ranking.**
`{BEST_VALUE}` still leads at {rows[BEST_VALUE][4]:.4f} value per thousand dollars and novelty
is {rows[BEST_VALUE][4] / rows[NOVEL][4]:.1f}x worse
(eq:replication-has-higher-information-per-dollar-than-novelty).

Two things drive that and both are worth naming. Bold hypotheses are *unlikely*, and an unlikely
binary outcome has low entropy -- {rows[NOVEL][2]:.4f} bits against
{rows['an ablation of one component'][2]:.4f} for a coin-flip ablation. And they are expensive,
by a factor of {rows[NOVEL][0] / rows['a benchmark rerun'][0]:.0f} over the cheapest row.

The allocation table shows what is actually done. {ACTUAL[NOVEL]:.0%} of effort on bold
hypotheses, {ACTUAL['an incremental variation']:.0%} on incremental variations, and
{ACTUAL['direct replication']:.0%} on replication -- delivering {actual_value:,.1f} units of
value from {BUDGET:,.0f}.

The portfolio table is the realistic alternative, because value is not fungible and no field
should spend everything on reruns. Reserving {MIN_NOVEL:.0%} for bold hypotheses -- they are the
only source of genuinely new directions, whatever their value per dollar -- and filling the rest
by value per dollar under plausible caps delivers **{port_value:,.1f} against
{actual_value:,.1f}** -- a factor of {port_value / actual_value:.1f}, **with no new capability
required.** The gain is entirely allocative, and novelty's share falls only from
{ACTUAL[NOVEL]:.0%} to {MIN_NOVEL:.0%}.

The repeats table is the largest single loss and the one no budget contains
(eq:an-unpublished-negative-is-repeated-by-everyone-else). If {GROUPS} groups could have the idea
and {0.12:.0%} of them try it, the field spends {waste[0.12]:,.0f} discovering something one
group already knew. Publishing it costs {rows[NOVEL][0]:,.0f} -- a waste multiple of
**{waste[0.12] / rows[NOVEL][0]:.1f}x**.

Nobody bears that cost individually, which is why it persists. It is a commons problem in a field
that measures individuals, and it is the clearest case in this book of a large, computable, and
entirely unaddressed inefficiency.

The last table is what automation changes, and it is the point of the chapter. Automation reduces
the cost of `{NOVEL}` by {1 / rows[NOVEL][6]:.1f}x -- the design, the reasoning and the writing
are hard to automate. It reduces `a benchmark rerun` by
{1 / rows['a benchmark rerun'][6]:.1f}x and `reproduction from artefacts` by
{1 / rows['reproduction from artefacts'][6]:.1f}x, because those are mechanical.

So the value-per-dollar gap **widens** under automation, from
{rows[BEST_VALUE][4] / rows[NOVEL][4]:.1f}x to
{auto_rows[best_auto][1] / auto_rows[NOVEL][1]:.1f}x.

**Automation's comparative advantage is precisely in the experiments the field under-runs.**
Replication, ablation, reproduction from artefacts, negative results -- mechanical, cheap,
high-entropy, and unrewarded. That is a much less exciting claim than an automated scientist
generating novel hypotheses, and on these numbers it is where essentially all of the available
value is.""")

# -*- coding: utf-8 -*-
# Extracted from: Chapter 219 — Building an Evaluation Framework from Scratch
# Source: src/.../ch219-building-a-framework.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Where a gate goes is a cost decision, and "as early as possible" is the wrong rule.

An instrument that runs on every commit runs a hundred times more often than one that runs
per release, so its cost is multiplied by a hundred. A defect that escapes to production
costs ch:ops-lifecycle's full return trip. Gate placement is the product of those two terms
(eq:gate-placement-is-set-by-cost-times-escape), and the optimum puts cheap fast checks
early and expensive slow ones late -- which is not what "shift left" is usually taken to
mean.

The second half is the reason gates get disabled. A gate with a false-positive rate blocks
good changes, and above a computable threshold the blocking costs more than the catching
(eq:a-flaky-gate-has-a-blocking-threshold).
"""
# (stage, runs per release, cost multiplier of a defect escaping past this stage)
STAGES = [
    ("every commit",   140.0,  1.0),
    ("pull request",    32.0,  1.9),
    ("pre-merge",       32.0,  2.4),
    ("pre-deploy",       4.0,  4.1),
    ("canary",           4.0,  7.8),
    ("production",       1.0, 17.0),
]
# (instrument, cost per run, detection probability, latency hours)
GATES = [
    ("schema and format check",       0.9, 0.31, 0.05),
    ("execution / test grading",     14.0, 0.34, 0.60),
    ("faithfulness judge",           26.0, 0.29, 0.40),
    ("invariant suite",              38.0, 0.27, 0.30),
    ("judge ensemble, both orders", 142.0, 0.41, 0.90),
    ("human spot-check",           4100.0, 0.55, 26.0),
]
DEFECTS_PER_RELEASE = 3.1
BASE_DEFECT_COST = 2400.0

print("Six stages, each running a gate a different number of times per release,")
print("each letting a surviving defect cost more.")
print()
print(f"{'stage':>16}{'runs/release':>15}{'escape multiplier':>20}"
      f"{'cost of one escape':>21}")
print("-" * 72)
for name, runs, mult in STAGES:
    print(f"{name:>16}{runs:>15.0f}{mult:>20.1f}"
          f"{BASE_DEFECT_COST * mult:>21,.0f}")

print()
print()
print("Placing one gate: total cost per release at each stage.")
print()


def place(gate, stage):
    gname, gcost, gdet, glat = gate
    sname, runs, mult = stage
    run_cost = gcost * runs
    escaped = DEFECTS_PER_RELEASE * (1.0 - gdet)
    escape_cost = escaped * BASE_DEFECT_COST * mult
    return run_cost, escape_cost, run_cost + escape_cost


print(f"{'gate':>30}", end="")
for sname, runs, mult in STAGES:
    print(f"{sname:>15}", end="")
print()
print("-" * 120)
best_stage = {}
for g in GATES:
    print(f"{g[0]:>30}", end="")
    totals = {}
    for s in STAGES:
        rc, ec, tot = place(g, s)
        totals[s[0]] = tot
        print(f"{tot:>15,.0f}", end="")
    print()
    best_stage[g[0]] = min(totals, key=lambda k: totals[k])

print()
print(f"{'gate':>30}{'cheapest stage':>18}{'run cost there':>17}"
      f"{'escape cost there':>20}")
print("-" * 85)
for g in GATES:
    s = [x for x in STAGES if x[0] == best_stage[g[0]]][0]
    rc, ec, tot = place(g, s)
    print(f"{g[0]:>30}{best_stage[g[0]]:>18}{rc:>17,.0f}{ec:>20,.0f}")

print()
print("Cheap gates want to be early because their run cost is small even at")
print("140 runs; expensive gates want to be late because it is not.")

print()
print()
print("A pipeline: each gate at its cheapest stage, applied in sequence.")
print()
print(f"{'stage':>16}{'gate':>30}{'defects entering':>19}"
      f"{'caught':>9}{'run cost':>11}{'escape cost':>14}")
print("-" * 99)
remaining = DEFECTS_PER_RELEASE
total_run = 0.0
pipeline = []
for sname, runs, mult in STAGES:
    for g in GATES:
        if best_stage[g[0]] != sname:
            continue
        caught = remaining * g[1 + 1]
        rc = g[1] * runs
        total_run += rc
        entering = remaining
        remaining -= caught
        pipeline.append((sname, g[0], entering, caught, rc, remaining))
        print(f"{sname:>16}{g[0]:>30}{entering:>19.3f}"
              f"{caught:>9.3f}{rc:>11,.0f}"
              f"{remaining * BASE_DEFECT_COST * mult:>14,.0f}")

final_escape = remaining * BASE_DEFECT_COST * STAGES[-1][2]
print("-" * 99)
print(f"{'TOTAL':>16}{'':>30}{'':>19}"
      f"{DEFECTS_PER_RELEASE - remaining:>9.3f}{total_run:>11,.0f}"
      f"{final_escape:>14,.0f}")
print()
print(f"total per release: {total_run + final_escape:,.0f}")
print(f"with no gates at all: "
      f"{DEFECTS_PER_RELEASE * BASE_DEFECT_COST * STAGES[-1][2]:,.0f}")

print()
print()
print("Now the reason gates get turned off: false positives block good changes.")
print()
CHANGES_PER_RELEASE = 26.0
BLOCK_COST = 2800.0       # a blocked good change: investigation, rerun, delay
print(f"{'false-positive rate':>21}{'good changes blocked':>23}"
      f"{'blocking cost':>16}{'catching value':>17}{'net':>12}")
print("-" * 89)
caught_at = {gname: (sname, caught)
             for sname, gname, ent, caught, rc, rem in pipeline}
mult_of = {sname: mult for sname, runs, mult in STAGES}
PROD_MULT = STAGES[-1][2]


def gate_value(gname):
    """Defects this gate catches, valued at what reaching production would cost."""
    sname, caught = caught_at[gname]
    return caught * BASE_DEFECT_COST * (PROD_MULT - mult_of[sname])


GATE = GATES[4]
caught_value = gate_value(GATE[0])
flake = {}
for fp in (0.005, 0.02, 0.05, 0.10, 0.20, 0.35):
    blocked = CHANGES_PER_RELEASE * fp
    bcost = blocked * BLOCK_COST
    flake[fp] = (blocked, bcost, caught_value - bcost)
    print(f"{fp:>21.1%}{blocked:>23.2f}{bcost:>16,.0f}"
          f"{caught_value:>17,.0f}{caught_value - bcost:>12,.0f}")

threshold = caught_value / (CHANGES_PER_RELEASE * BLOCK_COST)
print()
print(f"break-even false-positive rate: {threshold:.1%}")
print("above that, the gate should be advisory rather than blocking")

print()
print()
print("The same threshold for each gate, which is where the policy comes from.")
print()
print(f"{'gate':>30}{'catches':>10}{'value caught':>15}"
      f"{'max FP rate to block':>23}")
print("-" * 78)
thr = {}
for g in GATES:
    val = gate_value(g[0])
    tt = val / (CHANGES_PER_RELEASE * BLOCK_COST)
    thr[g[0]] = tt
    print(f"{g[0]:>30}{caught_at[g[0]][1]:>10.3f}{val:>15,.0f}"
          f"{min(tt, 1.0):>23.1%}")

print()
print()
print("And what latency does, independent of cost: how long a gate holds the")
print("loop open.")
print()
runs_of = {sname: runs for sname, runs, mult in STAGES}
print(f"{'gate':>30}{'latency h':>12}{'fits in':>18}"
      f"{'gate hours per release':>25}")
print("-" * 85)
hours = {}
for g in GATES:
    lat = g[3]
    fits = ("a commit hook" if lat < 0.1 else
            "CI" if lat < 1.0 else
            "a nightly run" if lat < 12.0 else
            "a release cycle")
    h = lat * runs_of[best_stage[g[0]]]
    hours[g[0]] = h
    print(f"{g[0]:>30}{lat:>12.2f}{fits:>18}{h:>25.1f}")
print("-" * 85)
print(f"{'TOTAL':>30}{'':>12}{'':>18}{sum(hours.values()):>25.1f}")

print(f"""
The stage table is the two multipliers that decide everything. A gate at `every commit` runs
{STAGES[0][1]:.0f} times a release and a gate at `pre-deploy` runs {STAGES[3][1]:.0f} times,
so the same instrument costs {STAGES[0][1] / STAGES[3][1]:.0f} times more in the first
position. And a defect escaping past `every commit` costs
{STAGES[0][2]:.1f}x base while one escaping past `canary` costs {STAGES[4][2]:.1f}x, which is
ch:ops-lifecycle's return-trip result in gate form.

The placement grid multiplies them out. The `{GATES[0][0]}` is cheapest at
`{best_stage[GATES[0][0]]}`; the `{GATES[5][0]}` is cheapest at
`{best_stage[GATES[5][0]]}` (eq:gate-placement-is-set-by-cost-times-escape).

**Cheap gates want to be early and expensive gates want to be late**, and the reason is
arithmetic rather than philosophy: at {STAGES[0][1]:.0f} runs a release, a
{GATES[5][1]:,.0f}-per-run instrument costs {GATES[5][1] * STAGES[0][1]:,.0f} to sit on every
commit, which is more than every defect it would ever catch.

That is a correction to how "shift left" is usually applied. The principle is right and the
implementation -- move every check earlier -- is wrong for any check whose per-run cost is
not negligible. ch:ops-lifecycle made the same correction from the other side: **shorten the
return trip, do not blindly move detection earlier.**

The pipeline table puts each gate at its cheapest stage and runs them in sequence. Total
{total_run + final_escape:,.0f} per release, against
{DEFECTS_PER_RELEASE * BASE_DEFECT_COST * STAGES[-1][2]:,.0f} with no gates --
{(DEFECTS_PER_RELEASE * BASE_DEFECT_COST * STAGES[-1][2]) / (total_run + final_escape):.1f}
times cheaper -- with {DEFECTS_PER_RELEASE - remaining:.3f} of {DEFECTS_PER_RELEASE:.1f}
defects caught.

Note where the run cost concentrates. The gates at `every commit` are running
{STAGES[0][1]:.0f} times and cost {sum(g[1] * STAGES[0][1] for g in GATES if best_stage[g[0]] == 'every commit'):,.0f}
between them, which is a real line item and is still small against the escapes it prevents.

The flake table is the failure mode that kills gates in practice. The
`{GATE[0]}` catches {caught_at[GATE[0]][1]:.3f} defects a release at its stage, worth
{caught_value:,.0f} against letting them reach production. At a {0.05:.0%} false-positive rate it blocks
{flake[0.05][0]:.2f} good changes for {flake[0.05][1]:,.0f}; at {0.35:.0%} it blocks
{flake[0.35][0]:.2f} for {flake[0.35][1]:,.0f}, and the net is
{flake[0.35][2]:,.0f} (eq:a-flaky-gate-has-a-blocking-threshold).

The break-even is **{threshold:.1%}** -- above that, the gate costs more in blocked good
changes than it saves in caught defects, and it should be advisory rather than blocking.

The per-gate threshold table is the policy this produces, and it is a better policy than the
usual one. Gates are normally blocking or advisory by *category* -- tests block, lint warns,
quality metrics warn. Here the rule is uniform and computed: **a gate blocks if its
false-positive rate is below its own threshold**, which depends on what it catches and what a
block costs. The `{GATES[0][0]}` may block at up to {min(thr[GATES[0][0]], 1.0):.0%}; the
`{GATES[5][0]}` at up to {min(thr[GATES[5][0]], 1.0):.0%}.

The latency table is the constraint that overrides both, and its last column is the number
to check before agreeing to any of this. A gate taking {GATES[5][3]:.0f} hours cannot sit in
CI whatever its economics say. And summed across the pipeline the gates occupy
{sum(hours.values()):.0f} hours per release -- against ch:ops-lifecycle's
{847:.0f}-hour period, of which {156:.0f} was work.

**The evaluation framework is now
{sum(hours.values()) / 847.0:.0%} of the loop's total duration**, and most of it is
waiting rather than working, which is exactly the category that chapter found dominates and
nobody measures. Every gate is individually justified by the table above it and the sum is a
budget decision nobody made.

Which is the last thing to carry out of this part. An evaluation framework is not free even
when every instrument in it is cheap, and the cost that binds is not the one on the invoice.
Run the hours column before the dollars column.""")

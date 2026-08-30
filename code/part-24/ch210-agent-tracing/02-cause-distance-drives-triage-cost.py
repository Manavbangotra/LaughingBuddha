# -*- coding: utf-8 -*-
# Extracted from: Chapter 210 — Agent Tracing and Tool-Call Monitoring
# Source: src/.../ch210-agent-tracing.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""In an agent trace, the failure is a step and the cause is an earlier step.

A single-turn failure has one place to look. An agent failure has a chain: the answer was
wrong because a decision was wrong because a tool returned something unexpected because
the arguments were built from a retrieval that missed.

So localisation is not "which step failed" -- it is "how far back from the visible
failure does the cause sit", and the search cost grows with that distance
(eq:cause-distance-drives-triage-cost).

This listing measures the distance distribution, finds what the trace must record to
close it, and shows why per-step correctness monitoring does not find these at all.
"""
import math

# (cause type, share of failures, steps back from the visible failure,
#  P(a per-step check would have caught it at the step where it happened))
CAUSES = [
    ("tool returned an error",        0.14, 0, 0.94),
    ("tool returned wrong data",      0.19, 2, 0.11),
    ("arguments built wrongly",       0.16, 1, 0.42),
    ("retrieval missed the fact",     0.21, 3, 0.08),
    ("plan was wrong from the start", 0.17, 6, 0.05),
    ("state corrupted mid-run",       0.13, 4, 0.22),
]
STEPS_MEAN = 7.4
BASE_MINUTES_PER_STEP = 3.1

print("Where the cause of an agent failure actually sits, relative to the step")
print("where the failure became visible.")
print()
print(f"{'cause':>32}{'share':>9}{'steps back':>13}"
      f"{'per-step check catches':>25}")
print("-" * 80)
tab = {}
for name, share, back, catch in CAUSES:
    tab[name] = (share, back, catch)
    print(f"{name:>32}{share:>9.0%}{back:>13}{catch:>25.0%}")

mean_back = sum(s * b for n, s, b, c in CAUSES)
caught = sum(s * c for n, s, b, c in CAUSES)
print()
print(f"mean distance from visible failure to cause: {mean_back:.1f} steps")
print(f"share a per-step correctness check would catch: {caught:.0%}")

print()
print()
print("Why per-step checks miss most of it: the steps that CAUSE failures mostly")
print("succeed at the time.")
print()
print(f"{'cause':>32}{'step succeeded?':>18}  {'why the check passes':<40}")
print("-" * 92)
WHY = {
    "tool returned an error":        ("no",  "it did not"),
    "tool returned wrong data":      ("yes", "well-formed, plausible, wrong"),
    "arguments built wrongly":       ("yes", "valid arguments, wrong ones"),
    "retrieval missed the fact":     ("yes", "returned documents, not the right ones"),
    "plan was wrong from the start": ("yes", "each step executed correctly"),
    "state corrupted mid-run":       ("yes", "the write succeeded"),
}
for name, share, back, catch in CAUSES:
    ok, why = WHY[name]
    print(f"{name:>32}{ok:>18}  {why:<40}")

print()
print()
print("Triage cost by distance. Without recorded intermediate state, each step")
print("back must be reconstructed by re-reading and inferring.")
print()
print(f"{'steps back':>12}{'share of failures':>20}{'minutes to localise':>22}"
      f"{'weighted':>11}")
print("-" * 68)
raw_total = 0.0
for back in sorted(set(b for n, s, b, c in CAUSES)):
    share = sum(s for n, s, b, c in CAUSES if b == back)
    mins = BASE_MINUTES_PER_STEP * (back + 1) * (1.0 + 0.35 * back)
    raw_total += share * mins
    print(f"{back:>12}{share:>20.0%}{mins:>22.1f}{share * mins:>11.1f}")
print("-" * 68)
print(f"{'MEAN':>12}{1.0:>20.0%}{'':>22}{raw_total:>11.1f}")

print()
print()
print("What each recorded field removes from that cost.")
print()
FIELDS = [
    ("explicit step boundaries",   0.18, 0.5),
    ("tool arguments as sent",     0.22, 1.0),
    ("tool results as received",   0.26, 1.5),
    ("agent state after each step", 0.31, 4.0),
    ("the plan, and revisions to it", 0.15, 2.0),
    ("causal links (this used that)", 0.34, 7.0),
]
print(f"{'field':>32}{'cuts triage by':>17}{'effort':>9}"
      f"{'minutes after':>16}{'per effort':>13}")
print("-" * 88)
cur = raw_total
per = {}
for name, cut, eff in FIELDS:
    saved = cur * cut
    per[name] = (cut, eff, saved, saved / eff)
    print(f"{name:>32}{cut:>17.0%}{eff:>9.1f}"
          f"{cur * (1 - cut):>16.1f}{saved / eff:>13.2f}")

print()
print()
print("Building them in payback order.")
print()
order = sorted(FIELDS, key=lambda f: -((raw_total * f[1]) / f[2]))
print(f"{'after adding':>32}{'minutes to localise':>22}{'effort so far':>16}"
      f"{'vs raw':>10}")
print("-" * 82)
cur = raw_total
eff = 0.0
path = []
for name, cut, e in order:
    cur *= (1 - cut)
    eff += e
    path.append((name, cur, eff))
    print(f"{name:>32}{cur:>22.1f}{eff:>16.1f}{cur / raw_total:>9.2f}x")

print()
print()
print("What that does to the triage capacity from the previous listing.")
print()
FAILING = 42000.0 * 0.09
HUMAN_MIN_DAY = 6.0 * 60.0
print(f"{'trace structure':>32}{'minutes/trace':>16}"
      f"{'traces/engineer/day':>22}{'engineers for 25%':>20}")
print("-" * 92)
for label, mins in (("raw", raw_total),
                    ("+ top two fields", path[1][1]),
                    ("+ top four fields", path[3][1]),
                    ("everything", path[-1][1])):
    per_eng = HUMAN_MIN_DAY / mins
    print(f"{label:>32}{mins:>16.1f}{per_eng:>22.0f}"
          f"{FAILING * 0.25 / per_eng:>20.1f}")

print()
print()
print("And the alternative to recording: re-run the agent and watch. This is the")
print("only way to recover state that was never written down.")
print()
print(f"{'approach':>32}{'minutes':>10}  {'needs':<36}")
print("-" * 82)
REPLAY = [
    ("read the raw trace",             raw_total, "nothing"),
    ("read a structured trace",        path[-1][1], "instrumentation"),
    ("re-run with full logging",       11.0, "reproducibility, ch:ops-versioning"),
    ("re-run and step through",        34.0, "reproducibility and an engineer"),
]
for label, mins, needs in REPLAY:
    print(f"{label:>32}{mins:>10.1f}  {needs:<36}")

print(f"""
The distance table is the structural difference between an agent failure and any other
kind. The failure becomes visible at one step and the cause sits **{mean_back:.1f} steps
earlier on average** (eq:cause-distance-drives-triage-cost), with the largest single
category -- `{max(CAUSES, key=lambda c: c[1])[0]}` at
{max(CAUSES, key=lambda c: c[1])[1]:.0%} -- sitting
{max(CAUSES, key=lambda c: c[1])[2]} steps back.

The second table is why the obvious instrumentation does not help. Per-step correctness
monitoring -- check each tool call, validate each output -- catches **{caught:.0%}** of
these, and the reason is in the last column: **the causing step succeeded.** A tool that
returns well-formed wrong data has not failed. A retrieval that returns documents has
not failed. A plan whose every step executes correctly has not failed.

Only `{CAUSES[0][0]}` at {CAUSES[0][1]:.0%} is caught reliably, because it is the one
category where something actually errored.

That is ch:sd-architecture's third property arriving in agent form: **the step
succeeded and was wrong**, and per-step checks are health checks by another name.

The cost table converts distance into minutes. A failure whose cause is at the visible
step takes {BASE_MINUTES_PER_STEP * 1 * 1.0:.1f} minutes; one six steps back takes
{BASE_MINUTES_PER_STEP * 7 * (1 + 0.35 * 6):.1f}, because each intervening step has to
be reconstructed by reading and inferring what it must have held. The weighted mean is
**{raw_total:.1f} minutes**.

The field table is the intervention. `{order[0][0]}` cuts triage by {order[0][1]:.0%}
for {order[0][2]:.1f} units of effort -- {raw_total * order[0][1] / order[0][2]:.2f}
minutes saved per unit, the best available. `{order[-1][0]}` cuts
{order[-1][1]:.0%} for {order[-1][2]:.1f}.

Built in payback order, the top two fields take triage from {raw_total:.1f} minutes to
{path[1][1]:.1f} for {path[1][2]:.1f} units of effort. All six reach {path[-1][1]:.1f}.

**The two cheapest fields do most of the work**, and both are things the agent framework
already has in memory at the moment it discards them. Step boundaries and tool arguments
are not derived or inferred -- they are variables that existed and were not written down.

The capacity table closes the loop with the previous listing. At raw traces, an engineer
localises {HUMAN_MIN_DAY / raw_total:.0f} a day and covering a quarter of failures needs
{FAILING * 0.25 / (HUMAN_MIN_DAY / raw_total):.1f} engineers. With the top four fields it
is {HUMAN_MIN_DAY / path[3][1]:.0f} a day and
{FAILING * 0.25 / (HUMAN_MIN_DAY / path[3][1]):.1f} engineers.

**A trace format change is worth more than tripling the team**, which is not how trace
formats are usually justified.

The last table is the honest bound. Re-running the agent with full logging localises in
{11.0:.0f} minutes -- better than any structured trace -- and it requires
ch:ops-versioning's reproducibility, which the same team probably does not have. So the
recording approach is not merely a cheaper alternative to replay; **for most teams it is
the only alternative**, because replay requires an artefact-pinning programme that
chapter found is usually incomplete.

Which gives the ordering: pin the artefacts if you can, and until then record the state,
because a trace you can read is the fallback for a run you cannot reproduce.""")

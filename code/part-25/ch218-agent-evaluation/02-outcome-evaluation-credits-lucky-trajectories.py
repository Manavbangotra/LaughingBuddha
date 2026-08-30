# -*- coding: utf-8 -*-
# Extracted from: Chapter 218 — Agent and Tool-Call Evaluation
# Source: src/.../ch218-agent-evaluation.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""An agent that reaches the right answer by the wrong route has passed your evaluation.

cite:jimenez2023swebench grades by running the repository's tests, which is the strongest
form of outcome evaluation available -- and even there, a patch that passes the tests for
the wrong reason passes. For agents with side effects, the gap is wider: the outcome can be
correct while the trajectory issued a refund twice, read a record it should not have, or
succeeded on a coincidence that will not recur
(eq:outcome-evaluation-credits-lucky-trajectories).

The obvious fix is to evaluate the trajectory. That requires a reference trajectory, which
puts you back in ch:ev-why-hard's position: there are many correct routes and the reference
is one of them (eq:trajectory-matching-inherits-the-answer-space-problem).

This listing measures both errors and finds the instrument that avoids both.
"""
# (outcome correct?, trajectory sound?, share, what it is, generalises?)
CELLS = [
    (True,  True,  0.44, "correct, for the right reason",  True),
    (True,  False, 0.14, "correct, by luck or side effect", False),
    (False, True,  0.11, "sound attempt, task not doable",  True),
    (False, False, 0.31, "wrong, and wrongly",              False),
]

print("Outcome and trajectory are two questions with four answers.")
print()
print(f"{'outcome':>10}{'trajectory':>13}{'share':>9}"
      f"{'what it is':>34}{'will recur?':>13}")
print("-" * 79)
for ok, sound, sh, desc, gen in CELLS:
    print(f"{('pass' if ok else 'fail'):>10}"
          f"{('sound' if sound else 'unsound'):>13}{sh:>9.2f}"
          f"{desc:>34}{('yes' if gen else 'no'):>13}")

outcome_pass = sum(sh for ok, s, sh, d, g in CELLS if ok)
truly_good = sum(sh for ok, s, sh, d, g in CELLS if ok and s)
print("-" * 79)
print(f"{'outcome score':>10}{'':>13}{outcome_pass:>9.2f}")
print(f"{'dependable':>10}{'':>13}{truly_good:>9.2f}")
print()
print(f"outcome evaluation over-credits by "
      f"{outcome_pass - truly_good:.2f} -- "
      f"{(outcome_pass - truly_good) / outcome_pass:.0%} of its passes")

print()
print()
print("What that does to a comparison between two agents.")
print()
AGENTS = [
    ("agent P", 0.44, 0.14),
    ("agent Q", 0.40, 0.21),
]
print(f"{'agent':>10}{'sound passes':>15}{'lucky passes':>15}"
      f"{'outcome score':>16}{'dependable':>13}{'rank flip?':>13}")
print("-" * 82)
by_out = sorted(AGENTS, key=lambda a: -(a[1] + a[2]))
by_dep = sorted(AGENTS, key=lambda a: -a[1])
for name, sound, lucky in AGENTS:
    flip = (by_out.index((name, sound, lucky))
            != by_dep.index((name, sound, lucky)))
    print(f"{name:>10}{sound:>15.2f}{lucky:>15.2f}"
          f"{sound + lucky:>16.2f}{sound:>13.2f}"
          f"{('yes' if flip else 'no'):>13}")
print()
print(f"outcome ranking: {by_out[0][0]} first; dependable ranking: "
      f"{by_dep[0][0]} first")

print()
print()
print("So evaluate the trajectory. How many correct trajectories are there?")
print()
TASKS = [
    ("look up one record",              1.0),
    ("look up and summarise",           3.0),
    ("book a flight to spec",          14.0),
    ("resolve a support ticket",       120.0),
    ("debug and patch a repository", 2600.0),
]
print(f"{'task':>32}{'valid trajectories':>21}"
      f"{'exact-match credit':>21}{'valid marked wrong':>21}")
print("-" * 95)
traj = {}
for name, a in TASKS:
    hit = min(1.0, 1.0 / a)
    traj[name] = (a, hit, 1.0 - hit)
    print(f"{name:>32}{a:>21.0f}{hit:>21.3f}{1.0 - hit:>21.2%}")

print()
print("Trajectory matching is ch:ev-why-hard's reference problem with a")
print("bigger answer space, because a trajectory is a sequence of choices")

print()
print()
print("The instrument that avoids both errors: check invariants, not routes.")
print()
INVARIANTS = [
    ("no tool called with invalid arguments",   0.19, 0.3),
    ("no write repeated with the same effect",  0.22, 0.4),
    ("no read outside the authorised scope",    0.11, 0.5),
    ("every claim traceable to a tool result",  0.26, 1.2),
    ("terminal state matches the request",      0.31, 2.0),
    ("no step contradicts an earlier result",   0.17, 3.0),
]
LUCKY = sum(sh for ok, s, sh, d, g in CELLS if ok and not s)
print(f"{'invariant':>40}{'catches of the lucky':>22}"
      f"{'effort':>9}{'per effort':>13}")
print("-" * 84)
inv = {}
for name, catch, eff in INVARIANTS:
    inv[name] = (catch, eff, catch / eff)
    print(f"{name:>40}{catch:>22.0%}{eff:>9.1f}{catch / eff:>13.3f}")

order = sorted(INVARIANTS, key=lambda i: -(i[1] / i[2]))
print()
print("Applied in payback order:")
print()
print(f"{'after adding':>40}{'lucky passes remaining':>25}"
      f"{'effort so far':>16}{'reported score':>17}")
print("-" * 98)
rem = LUCKY
eff = 0.0
path = []
for name, catch, e in order:
    rem *= (1.0 - catch)
    eff += e
    path.append((name, rem, eff))
    print(f"{name:>40}{rem:>25.4f}{eff:>16.1f}"
          f"{truly_good + rem:>17.4f}")

print()
print(f"the reported score falls from {outcome_pass:.4f} to "
      f"{truly_good + path[-1][1]:.4f} and gets more true")

print()
print()
print("None of these needs a reference trajectory. Which do?")
print()
NEEDS = [
    ("outcome check",              "a correct answer",       "yes"),
    ("trajectory match",           "a reference trajectory", "yes"),
    ("invariant checks",           "nothing",                "no"),
    ("side-effect diff",           "a clean environment",    "no"),
    ("step-level judge",           "a rubric",               "no"),
    ("cause-distance triage",      "recorded state",         "no"),
]
print(f"{'instrument':>26}{'what it needs':>26}{'needs ground truth?':>22}")
print("-" * 74)
for name, needs, gt in NEEDS:
    print(f"{name:>26}{needs:>26}{gt:>22}")

print()
print()
print("Cost per 1000 evaluated trajectories.")
print()
COSTS = [
    ("outcome check (human)",      3400.0, "over-credits by 24%"),
    ("outcome check (tests)",        12.0, "where tests exist"),
    ("trajectory match",           4100.0, "and penalises valid routes"),
    ("invariant checks",             31.0, "catches most lucky passes"),
    ("side-effect diff",             46.0, "catches the dangerous ones"),
    ("step-level judge",            190.0, "ch:ev-llm-judge applies"),
]
print(f"{'instrument':>26}{'cost/1000':>12}{'note':>34}")
print("-" * 72)
cost = {}
for name, c, note in COSTS:
    cost[name] = c
    print(f"{name:>26}{c:>12,.0f}{note:>34}")

combo = cost["outcome check (tests)"] + cost["invariant checks"] + cost["side-effect diff"]
print()
print(f"tests + invariants + side-effect diff: {combo:,.0f} per 1000")
print(f"against human outcome checking alone: "
      f"{cost['outcome check (human)']:,.0f} "
      f"({cost['outcome check (human)'] / combo:.0f}x)")

print(f"""
The quadrant table is the problem in four rows. Outcome evaluation reports
{outcome_pass:.2f} and the dependable share is {truly_good:.2f}: **{(outcome_pass - truly_good) / outcome_pass:.0%} of
its passes are correct for a reason that will not recur**
(eq:outcome-evaluation-credits-lucky-trajectories).

The third row is worth noticing too. {CELLS[2][2]:.0%} of trajectories are sound attempts at
tasks that could not be completed -- the environment was wrong, the record did not exist, the
policy forbade it -- and outcome evaluation scores those as failures. So the outcome score is
wrong in both directions at once, and the errors do not cancel because they are different
tasks.

The comparison table is where this stops being an accounting complaint. Agent P has
{AGENTS[0][1]:.2f} sound passes and {AGENTS[0][2]:.2f} lucky ones; agent Q has
{AGENTS[1][1]:.2f} and {AGENTS[1][2]:.2f}. On the outcome score Q wins
{AGENTS[1][1] + AGENTS[1][2]:.2f} to {AGENTS[0][1] + AGENTS[0][2]:.2f}; on dependable passes
P wins {AGENTS[0][1]:.2f} to {AGENTS[1][1]:.2f}.

**The agent that gets luckier ranks higher**, and luck does not survive contact with a
different distribution of tasks.

The trajectory table is the obvious fix failing. Debugging a repository has on the order of
{2600:.0f} valid trajectories, so exact trajectory matching credits {traj['debug and patch a repository'][1]:.3f}
of correct routes and marks {traj['debug and patch a repository'][2]:.2%} of them wrong
(eq:trajectory-matching-inherits-the-answer-space-problem).

That is ch:ev-why-hard's reference-sampling problem with a much larger answer space, because a
trajectory is a *sequence* of choices and the space multiplies at every step. Trajectory
matching is a worse instance of a problem that was already severe for single answers.

The invariant table is the way out and it is the useful part of this listing. Each row is a
property that can be checked on a trajectory **without any reference at all**: were the tool
arguments valid, was a write repeated, did a read leave the authorised scope, is every claim
traceable to a tool result, does the terminal state match the request, does any step
contradict an earlier one.

`{order[0][0]}` catches {order[0][1]:.0%} of the lucky passes for {order[0][2]:.1f} units of
effort. Applied in payback order, the lucky share falls from {LUCKY:.4f} to
{path[-1][1]:.4f} and **the reported score falls from {outcome_pass:.4f} to
{truly_good + path[-1][1]:.4f}** -- which is the right direction. An evaluation that gets
stricter and lower is an evaluation that started too high.

The dependency table is the structural point. Of six instruments, two need ground truth and
four do not, and the four that do not are the ones that catch the failure modes outcome
scoring cannot see. **Reference-free does not mean weak here**, which is the opposite of the
situation in ch:ev-why-hard, and the reason is that an invariant is a predicate rather than a
comparison -- the same escape that made execution grading work.

The cost table closes it. Test-graded outcomes plus invariant checks plus a side-effect diff
cost {combo:,.0f} per thousand trajectories against {cost['outcome check (human)']:,.0f} for
human outcome checking alone -- {cost['outcome check (human)'] / combo:.0f} times cheaper,
and it measures three things instead of one.

The instrument that is *not* on that list is trajectory matching, at
{cost['trajectory match']:,.0f} and penalising every valid route it did not anticipate. It is
the most expensive option and the only one that is wrong by construction, and it is what
"evaluate the reasoning, not just the answer" turns into when implemented literally.""")

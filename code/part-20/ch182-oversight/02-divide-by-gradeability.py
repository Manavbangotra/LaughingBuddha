# -*- coding: utf-8 -*-
# Extracted from: Chapter 182 — Where Human Oversight Remains Necessary
# Source: src/.../ch182-oversight.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The collaboration spectrum, which is the thing nobody evaluates.

cite:testini2025dsautomation's second finding is that data science automation is
evaluated at the extremes -- pure assistance or full autonomy -- and that the
intermediate regimes are neglected. This listing prices them.

Split any task into two parts, because the whole part has turned on the
distinction:

  judgement   what to ask, whether the answer follows, whether to act. No
              reference answer, so ch:aids-stack's ungradeable region.
  execution   the query, the transformation, the fit. Checkable.

Five ways to divide those between a person and an agent, and the right one depends
on which half is the bottleneck (eq:divide-by-gradeability).
"""
import numpy as np

rng = np.random.default_rng(4919)

M = 50000

# Quality on each half, by who does it.
HUMAN_JUDGE, AGENT_JUDGE = 0.86, 0.58
HUMAN_EXEC, AGENT_EXEC = 0.90, 0.83
# Hours each half costs the human when they do it, and when they review it.
H_JUDGE, H_EXEC = 1.1, 2.4
R_JUDGE, R_EXEC = 0.35, 0.55
# A human reviewing an agent's work catches this share of its errors.
REV_JUDGE, REV_EXEC = 0.70, 0.55
SPOT = 0.25             # share of work a spot-check regime actually inspects


def run(mode, m=M, judge_w=1.0, exec_w=1.0):
    """`judge_w` and `exec_w` scale how much each half matters for this task
    type. Returns (quality, human hours)."""
    def q(base, weight):
        return 1.0 - (1.0 - base) * weight

    if mode == "human only":
        j, e = q(HUMAN_JUDGE, judge_w), q(HUMAN_EXEC, exec_w)
        hrs = H_JUDGE + H_EXEC
    elif mode == "human judges, agent executes":
        j = q(HUMAN_JUDGE, judge_w)
        base = q(AGENT_EXEC, exec_w)
        e = 1 - (1 - base) * (1 - REV_EXEC)
        hrs = H_JUDGE + R_EXEC
    elif mode == "agent proposes, human judges":
        base = q(AGENT_JUDGE, judge_w)
        j = 1 - (1 - base) * (1 - REV_JUDGE)
        e = q(AGENT_EXEC, exec_w)
        hrs = R_JUDGE + R_EXEC
    elif mode == "agent does all, human spot-checks":
        bj, be = q(AGENT_JUDGE, judge_w), q(AGENT_EXEC, exec_w)
        j = 1 - (1 - bj) * (1 - SPOT * REV_JUDGE)
        e = 1 - (1 - be) * (1 - SPOT * REV_EXEC)
        hrs = SPOT * (R_JUDGE + R_EXEC)
    elif mode == "fully autonomous":
        j, e = q(AGENT_JUDGE, judge_w), q(AGENT_EXEC, exec_w)
        hrs = 0.0
    else:
        raise ValueError(mode)
    ok = (rng.random(m) < j) & (rng.random(m) < e)
    return float(ok.mean()), hrs


MODES = ["human only", "human judges, agent executes",
         "agent proposes, human judges", "agent does all, human spot-checks",
         "fully autonomous"]

print(f"{M:,} tasks. Judgement is ungradeable and execution is checkable;")
print("an agent is much weaker at the first and nearly as good at the second.")
print()
print(f"{'':>36}{'judgement':>12}{'execution':>12}")
print("-" * 60)
print(f"{'human':>36}{HUMAN_JUDGE:>12.0%}{HUMAN_EXEC:>12.0%}")
print(f"{'agent':>36}{AGENT_JUDGE:>12.0%}{AGENT_EXEC:>12.0%}")

print()
print()
print("The five modes on a balanced task.")
print()
print(f"{'mode':>36}{'quality':>10}{'human hours':>14}{'per hour':>11}")
print("-" * 71)
tab = {}
for mode in MODES:
    r = run(mode)
    tab[mode] = r
    per = r[0] / r[1] if r[1] else float("inf")
    cell = "--" if r[1] == 0 else f"{per:.3f}"
    print(f"{mode:>36}{r[0]:>10.1%}{r[1]:>14.1f}{cell:>11}")

print()
print()
print("Now vary where the difficulty is. `judgement weight` high means the task")
print("turns on deciding what to ask and whether the answer follows.")
print()
print(f"{'mode':>36}" + "".join(f"{lbl:>13}" for lbl in
                                ("exec-heavy", "balanced", "judge-heavy")))
print("-" * 75)
prof = {}
for mode in MODES:
    row = (run(mode, judge_w=0.4, exec_w=1.6)[0],
           run(mode)[0],
           run(mode, judge_w=1.6, exec_w=0.4)[0])
    prof[mode] = row
    print(f"{mode:>36}" + "".join(f"{v:>13.1%}" for v in row))

print()
print()
print("The best mode in each regime, and the best mode per human hour.")
print()
labels = ("exec-heavy", "balanced", "judge-heavy")
print(f"{'task profile':>16}{'best quality':>36}{'best per hour':>36}")
print("-" * 88)
best = {}
for i, lbl in enumerate(labels):
    bq = max(MODES, key=lambda mo: prof[mo][i])
    bp = max([mo for mo in MODES if tab[mo][1] > 0],
             key=lambda mo: prof[mo][i] / tab[mo][1])
    best[lbl] = (bq, bp)
    print(f"{lbl:>16}{bq:>36}{bp:>36}")

print()
print()
print("What each mode costs the human, which is the axis the extremes are")
print("chosen on and the middle is not evaluated on.")
print()
print(f"{'mode':>36}{'hours':>8}{'vs human only':>16}{'quality kept':>15}")
print("-" * 75)
h0, q0 = tab["human only"][1], tab["human only"][0]
for mode in MODES:
    r = tab[mode]
    print(f"{mode:>36}{r[1]:>8.1f}{r[1] / h0:>16.0%}{r[0] / q0:>15.0%}")

print()
print()
print("And the frontier: quality achievable per human hour spent, which is the")
print("comparison cite:testini2025dsautomation says nobody runs.")
print()
print(f"{'human hours':>13}{'best available mode':>36}{'quality':>10}")
print("-" * 61)
fr = {}
for cap in (0.0, 0.3, 1.0, 1.7, 3.5):
    avail = [mo for mo in MODES if tab[mo][1] <= cap + 1e-9]
    if not avail:
        continue
    b = max(avail, key=lambda mo: tab[mo][0])
    fr[cap] = (b, tab[b][0])
    print(f"{cap:>13.1f}{b:>36}{tab[b][0]:>10.1%}")

print(f"""
The first table's second row is the result.

**"Human judges, agent executes" reaches {tab['human judges, agent executes'][0]:.1%}
against a human doing everything at {tab['human only'][0]:.1%} -- higher quality,
at {tab['human judges, agent executes'][1] / tab['human only'][1]:.0%} of the human
hours.** It is not a trade-off. It is better on both axes, because the agent's
execution is nearly as good as a person's and a person reviewing it adds a second
check the solo human never had.

Fully autonomous reaches {tab['fully autonomous'][0]:.1%}, which is
{tab['fully autonomous'][0] / tab['human only'][0]:.0%} of the human-only quality
for none of the hours. Whether that is a good trade depends entirely on what a
wrong answer costs, and it is the only mode in the table where that question has to
be asked.

The profile table shows how the middle modes shift, and contains something the
listing was not built to show.

On a judgement-heavy task, "agent proposes, human judges" reaches
{prof['agent proposes, human judges'][2]:.1%} against a human doing everything at
{prof['human only'][2]:.1%} -- **a tie**, at
{tab['agent proposes, human judges'][1] / tab['human only'][1]:.0%} of the hours.

That is surprising and the mechanism is worth stating: a reviewed agent proposal
gets TWO passes at the judgement -- the agent's and the reviewer's -- where a solo
human gets one. When the task is hard enough that the human's own judgement is
fallible, the second pass compensates for the agent's weakness. **Delegating
judgement and reviewing it is not obviously worse than making the judgement
yourself**, once you account for the review being an additional check rather than a
substituted one.

The cost table is the one that should change how these systems are evaluated. Every
interior mode keeps most of the quality for a fraction of the hours:
{tab['agent proposes, human judges'][0] / tab['human only'][0]:.0%} of quality at
{tab['agent proposes, human judges'][1] / tab['human only'][1]:.0%} of the cost, and
{tab['agent does all, human spot-checks'][0] / tab['human only'][0]:.0%} at
{tab['agent does all, human spot-checks'][1] / tab['human only'][1]:.0%}.

And the frontier table is cite:testini2025dsautomation's point made arithmetic. At
every budget between zero and {1.7:.1f} hours, **the best available mode is an
intermediate one** -- spot-checking at {0.3:.1f} hours, agent-proposes at
{1.0:.1f}, human-judges-agent-executes at {1.7:.1f}. The two modes the literature
evaluates, full autonomy and pure assistance, are the frontier only at the two
extremes of the budget axis.

So the field measures the two points nobody should operate at, and the useful
region is unmeasured (eq:divide-by-gradeability).

One honest caveat on the "best per hour" column, which picks spot-checking
everywhere: a ratio with a near-zero denominator will do that, and it is not a
recommendation. The frontier table is the right way to read this -- pick your
budget, then pick the mode -- and per-hour ratios are only meaningful between modes
of comparable cost.

The rule that falls out is the part's, restated at the level of a working
arrangement. **Give the agent the half with a verifier and keep the half without
one** -- and where you must delegate the ungradeable half, review it rather than
sampling it, because a review is a second judgement and a sample is a smaller
first one.""")

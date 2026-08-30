# -*- coding: utf-8 -*-
# Extracted from: Chapter 163 — Multi-Agent Architectures and Communication
# Source: src/.../ch163-multi-agent.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Where the decorrelation comes from, and whether it needs a second agent.

ch:as-single-agent found that the entire value of a second agent is decorrelation.
cite:du2023debate is the strongest published case that it is real: several
instances propose and debate over several rounds and factual and mathematical
accuracy improve.

But decorrelation is a property of the SAMPLES, not of the agents. A single agent
that is forced to approach a stuck step differently is decorrelating its own
retries, and it pays no handoff cost to do it (eq:decorrelate-cheaply).

This listing puts four designs on the same call budget: one agent retrying
normally, one agent retrying with forced approach diversity, two agents debating,
and two agents dividing the work. The comparison is at EQUAL COST, which is the
comparison cite:cemri2025mast says is usually missing.
"""
import numpy as np

rng = np.random.default_rng(3001)

M = 40000
WORK = 8
BUDGET = 24             # model calls available per task, for every design
P_ORD = 0.94            # an ordinary step
P_STICKY_FIRST = 0.30   # a sticky step, first approach
P_SHARE_STICKY = 0.22   # share of steps that are sticky
P_HANDOFF = 0.884       # per-handoff factor, measured in the previous listing


def solve_step(sticky, tries, diversity, m):
    """Attempt one step `tries` times. `diversity` is the chance a retry takes a
    genuinely different approach rather than repeating the last one."""
    ok = np.zeros(m, dtype=bool)
    fresh = np.ones(m, dtype=bool)         # is this attempt a new approach?
    for t in range(tries):
        p = np.where(sticky,
                     np.where(fresh, P_STICKY_FIRST, 0.04),
                     P_ORD)
        ok |= (~ok) & (rng.random(m) < p)
        fresh = rng.random(m) < diversity
    return ok


def design(kind, m=M, work=WORK, budget=BUDGET):
    sticky = rng.random((m, work)) < P_SHARE_STICKY
    if kind == "single":
        tries = budget // work
        ok = np.ones(m, dtype=bool)
        for j in range(work):
            ok &= solve_step(sticky[:, j], tries, 0.0, m)
        return float(ok.mean())
    if kind == "single_diverse":
        tries = budget // work
        ok = np.ones(m, dtype=bool)
        for j in range(work):
            ok &= solve_step(sticky[:, j], tries, 0.85, m)
        return float(ok.mean())
    if kind == "debate":
        # Two instances propose independently and reconcile. Half the budget
        # each; a step succeeds if either instance solves it.
        tries = budget // (2 * work)
        ok = np.ones(m, dtype=bool)
        for j in range(work):
            a = solve_step(sticky[:, j], tries, 0.0, m)
            b = solve_step(sticky[:, j], tries, 0.0, m)
            # Independent instances share the sticky blind spot only partly.
            shared = rng.random(m) < 0.45
            ok &= a | (b & ~shared)
        return float(ok.mean())
    if kind == "divide":
        # Two agents take half the work each, with one handoff between them.
        tries = budget // work
        ok = np.ones(m, dtype=bool)
        for j in range(work):
            ok &= solve_step(sticky[:, j], tries, 0.0, m)
        ok &= rng.random(m) < P_HANDOFF
        return float(ok.mean())
    raise ValueError(kind)


DESIGNS = [("one agent, plain retries", "single"),
           ("one agent, forced diversity", "single_diverse"),
           ("two agents, debate", "debate"),
           ("two agents, divided work", "divide")]

print(f"{M:,} tasks, {WORK} steps, {BUDGET} model calls for every design.")
print(f"{P_SHARE_STICKY:.0%} of steps are sticky: {P_STICKY_FIRST:.0%} on a fresh")
print("approach and near-zero on a repeat. Two instances share 45% of the")
print(f"sticky blind spots. A handoff costs a factor of {P_HANDOFF}.")
print()
print(f"{'design':>32}{'completed':>12}{'calls':>9}{'agents':>9}")
print("-" * 62)
res = {}
for name, k in DESIGNS:
    v = design(k)
    res[name] = v
    n_ag = 1 if k.startswith("single") else 2
    print(f"{name:>32}{v:>12.1%}{BUDGET:>9}{n_ag:>9}")

print()
print()
print("How much diversity does a single agent need to match two debating ones?")
print()
print(f"{'diversity':>11}{'one agent':>12}{'two debating':>15}{'gap':>9}")
print("-" * 47)
deb = res["two agents, debate"]
div = {}
for d in (0.0, 0.25, 0.5, 0.75, 0.95):
    ok = np.ones(M, dtype=bool)
    sticky = rng.random((M, WORK)) < P_SHARE_STICKY
    tries = BUDGET // WORK
    for j in range(WORK):
        ok &= solve_step(sticky[:, j], tries, d, M)
    v = float(ok.mean())
    div[d] = v
    print(f"{d:>11.0%}{v:>12.1%}{deb:>15.1%}{v - deb:>+9.1%}")

print()
print()
print("And how the comparison moves with how much the two instances share.")
print("A debate between two copies of the same model shares almost everything.")
print()
print(f"{'shared blind spots':>20}{'two debating':>15}{'one, diverse':>15}"
      f"{'better':>16}")
print("-" * 66)
sh = {}
single_div = res["one agent, forced diversity"]
for s in (0.95, 0.75, 0.45, 0.20, 0.0):
    sticky = rng.random((M, WORK)) < P_SHARE_STICKY
    tries = BUDGET // (2 * WORK)
    ok = np.ones(M, dtype=bool)
    for j in range(WORK):
        a = solve_step(sticky[:, j], tries, 0.0, M)
        b = solve_step(sticky[:, j], tries, 0.0, M)
        shared = rng.random(M) < s
        ok &= a | (b & ~shared)
    v = float(ok.mean())
    sh[s] = v
    best = "debate" if v > single_div else "one agent"
    print(f"{s:>20.0%}{v:>15.1%}{single_div:>15.1%}{best:>16}")

print()
print()
print("The same four designs at several budgets, since halving the budget per")
print("instance is what a two-agent design actually does.")
print()
print(f"{'budget':>8}" + "".join(f"{n.split(',')[0] + ' ' + n.split(',')[1][:8]:>22}"
                                 for n, _ in DESIGNS[:2])
      + f"{'debate':>10}{'divided':>10}")
print("-" * 74)
bd = {}
for b in (16, 24, 40, 64):
    row = [design(k, budget=b) for _, k in DESIGNS]
    bd[b] = row
    print(f"{b:>8}{row[0]:>22.1%}{row[1]:>22.1%}{row[2]:>10.1%}{row[3]:>10.1%}")

print(f"""
The first table is the comparison at equal cost, and the ordering is the result.

One agent with plain retries: {res['one agent, plain retries']:.1%}. Two agents
debating: {res['two agents, debate']:.1%}. Two agents dividing the work:
{res['two agents, divided work']:.1%}. **One agent with forced approach diversity:
{res['one agent, forced diversity']:.1%}.**

The single agent wins by {res['one agent, forced diversity'] - res['two agents, debate']:.1%}
over the debating pair, at the same {BUDGET} calls, with no handoff and no second
system to operate.

The mechanism is arithmetic rather than subtle. **Decorrelation is a property of
the SAMPLES, not of the agents** (eq:decorrelate-cheaply). Two instances
decorrelate because they are different systems; one instance decorrelates because
it was made to try a different approach. Both produce varied attempts, and only
one of them halves its own budget to do it.

The second table quantifies how much diversity a single agent needs to match the
pair. At {0:.0%} forced diversity it already scores {div[0.0]:.1%} against the
debate's {deb:.1%} -- ahead, purely because it kept its whole budget. At
{0.5:.0%} it is {div[0.5]:.1%}, and at {0.95:.0%}, {div[0.95]:.1%}.

The third table is the one that should settle an architecture argument. It sweeps
how much the two debating instances share, from {0.95:.0%} -- two copies of the
same model, which is what a debate between two prompts of one model actually is --
down to {0:.0%}, which is two genuinely independent systems.

At {0.95:.0%} shared blind spots the debate reaches {sh[0.95]:.1%}. At
{0:.0%} -- perfect independence, which no real pair achieves -- it reaches
{sh[0.0]:.1%}. The single diverse agent scores {single_div:.1%}.

**The debate loses at every level of independence**, including at zero, because
the budget halving costs more than the decorrelation buys. That is the honest form
of this listing's finding and it is stronger than expected: it is not that
multi-agent debate is a marginal improvement, it is that at equal cost it is
behind a single agent that varies its own approach.

The fourth table shows the gap is not an artefact of the budget. At {16} calls the
single diverse agent leads by {bd[16][1] - bd[16][2]:.1%}; at {64} by
{bd[64][1] - bd[64][2]:.1%}. **The gap widens with budget**, because the single
agent gets the whole increase and the pair gets half each.

Three honest caveats, and the second is substantial.

This model gives debate no mechanism beyond "either instance solves the step".
cite:du2023debate's actual procedure has instances READ each other's reasoning and
revise, which is a real additional mechanism this listing does not represent --
and their reported gains on mathematical and factual tasks are real. What the
listing establishes is narrower: **the ensembling half of debate is available more
cheaply from one agent**, so a debate's gains have to come from the critique half
to be worth its cost.

Second, "forced diversity" is an assumption here rather than a measurement. The
listing supposes an agent CAN be made to take a genuinely different approach on
demand, at {0.85:.0%} reliability. That is the load-bearing assumption and it is
the one to test on your own system: sample a stuck step twice with a
diversity-forcing prompt and measure how often the two attempts are actually
different. If they are not, the single-agent column in the first table is
optimistic and the argument weakens.

Third, none of this touches the capability term. ch:as-single-agent's
eq:multi-agent-ceiling applies to every column here: a step no instance can do is
not solved by varying the approach or by adding an instance.

So the question to put to any multi-agent proposal is not whether it decorrelates.
It is **whether it decorrelates more per model call than forcing one agent to vary
its approach** -- and this listing says the bar is higher than the architecture
diagrams suggest.""")

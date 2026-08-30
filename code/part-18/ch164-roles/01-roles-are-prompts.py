# -*- coding: utf-8 -*-
# Extracted from: Chapter 164 — Supervisor, Worker, Planner, and Critic Roles
# Source: src/.../ch164-roles.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Do role labels do anything?

Supervisor, worker, planner, critic. The taxonomy is universal and the mechanism
is rarely stated. This listing asks whether a role LABEL buys anything, by
comparing a role-structured multi-agent system against one agent running the same
prompts in sequence (eq:roles-are-prompts).

The comparison is deliberately unfair to the multi-agent version in exactly one
way, which is the way that matters: both spend the same number of model calls.
"""
import numpy as np

rng = np.random.default_rng(3079)

M = 40000
WORK = 8
BUDGET = 24
P_ORD = 0.93
P_SHARE_STICKY = 0.20
P_STICKY_FRESH = 0.32
P_HANDOFF = 0.884       # measured in ch:as-multi-agent
P_CRITIC_TP = 0.72      # a critic notices a real error
P_CRITIC_FP = 0.12      # a critic objects to correct work


def attempt(sticky, tries, diversity, m):
    ok = np.zeros(m, dtype=bool)
    fresh = np.ones(m, dtype=bool)
    for _ in range(tries):
        p = np.where(sticky, np.where(fresh, P_STICKY_FRESH, 0.05), P_ORD)
        ok |= (~ok) & (rng.random(m) < p)
        fresh = rng.random(m) < diversity
    return ok


def system(kind, m=M, work=WORK, budget=BUDGET, corr=0.9, diversity=0.0):
    """kind:
      solo       one agent, one prompt, all the budget
      solo_roles one agent switching prompts (plan, do, check) -- no handoff
      roles      separate planner/worker/critic agents, handoffs between them
      roles_dec  the same, with a critic whose errors are decorrelated (corr)
    """
    sticky = rng.random((m, work)) < P_SHARE_STICKY
    if kind == "solo":
        tries = budget // work
        ok = np.ones(m, dtype=bool)
        for j in range(work):
            ok &= attempt(sticky[:, j], tries, diversity, m)
        return float(ok.mean())

    # Every role-bearing design spends part of its budget on planning and
    # criticism rather than on doing.
    do_budget = int(budget * 0.6)
    tries = max(1, do_budget // work)
    ok = np.ones(m, dtype=bool)
    for j in range(work):
        ok &= attempt(sticky[:, j], tries, diversity, m)

    # The critic reviews the result. Its errors are correlated with the worker's
    # unless it is genuinely a different system.
    shares = rng.random(m) < corr
    caught = (~ok) & (rng.random(m) < P_CRITIC_TP) & ~shares
    objected = ok & (rng.random(m) < P_CRITIC_FP)
    # A caught error gets one more pass; a false objection wastes one.
    ok |= caught & (rng.random(m) < P_ORD)
    ok &= ~(objected & (rng.random(m) < 0.35))     # some rework breaks things

    if kind in ("roles", "roles_dec"):
        # planner -> worker and worker -> critic are two handoffs.
        ok &= rng.random(m) < P_HANDOFF
        ok &= rng.random(m) < P_HANDOFF
    return float(ok.mean())


DESIGNS = [("one agent, one prompt", dict(kind="solo")),
           ("one agent, role prompts", dict(kind="solo_roles")),
           ("three agents, roles", dict(kind="roles")),
           ("three agents, decorrelated critic",
            dict(kind="roles_dec", corr=0.25))]

print(f"{M:,} tasks, {WORK} steps, {BUDGET} model calls for every design.")
print(f"Role-bearing designs spend 40% of the budget on planning and criticism.")
print(f"A critic catches a real error {P_CRITIC_TP:.0%} of the time and objects")
print(f"to correct work {P_CRITIC_FP:.0%} of the time. Each handoff costs")
print(f"{P_HANDOFF}.")
print()
print(f"{'design':>36}{'completed':>12}{'agents':>9}{'handoffs':>11}")
print("-" * 68)
res = {}
for name, kw in DESIGNS:
    v = system(**kw)
    res[name] = v
    n_ag = 1 if kw["kind"].startswith("solo") else 3
    n_ho = 0 if kw["kind"].startswith("solo") else 2
    print(f"{name:>36}{v:>12.1%}{n_ag:>9}{n_ho:>11}")

print()
print()
print("The critic is the role with a mechanism. Sweep how correlated its errors")
print("are with the worker's, holding everything else fixed.")
print()
print(f"{'critic correlation':>20}{'three agents':>15}{'one agent':>13}"
      f"{'best':>13}")
print("-" * 61)
solo = res["one agent, one prompt"]
cc = {}
for c in (1.0, 0.9, 0.6, 0.3, 0.0):
    a = system(kind="roles", corr=c)
    cc[c] = a
    best = "roles" if a > solo else "one agent"
    print(f"{c:>20.1f}{a:>15.1%}{solo:>13.1%}{best:>13}")

print()
print()
print("What the budget split costs. Role designs spend some of it on roles;")
print("sweep how much.")
print()
print(f"{'spent on doing':>16}{'three agents':>15}{'one agent, roles':>19}")
print("-" * 50)
sp = {}
for frac in (0.4, 0.6, 0.8, 1.0):
    a = system(kind="roles", corr=0.25, budget=int(BUDGET))
    # recompute with an explicit do-share by scaling the budget passed through
    tries_budget = int(BUDGET * frac)
    sticky = rng.random((M, WORK)) < P_SHARE_STICKY
    tr = max(1, tries_budget // WORK)
    ok = np.ones(M, dtype=bool)
    for j in range(WORK):
        ok &= attempt(sticky[:, j], tr, 0.0, M)
    shares = rng.random(M) < 0.25
    caught = (~ok) & (rng.random(M) < P_CRITIC_TP) & ~shares
    ok2 = ok | (caught & (rng.random(M) < P_ORD))
    with_ho = ok2 & (rng.random(M) < P_HANDOFF) & (rng.random(M) < P_HANDOFF)
    sp[frac] = (float(with_ho.mean()), float(ok2.mean()))
    print(f"{frac:>16.0%}{sp[frac][0]:>15.1%}{sp[frac][1]:>19.1%}")

print()
print()
print("And what roles buy when they carry different CAPABILITIES rather than")
print("different labels -- ch:ag-security's partition, as an architecture.")
print()
print(f"{'arrangement':>34}{'completed':>12}{'blast radius':>15}")
print("-" * 61)
cap = {}
for name, corr, radius in [("one agent, all capabilities", 0.9, 8),
                           ("three role agents, all capabilities", 0.9, 8),
                           ("three agents, split capabilities", 0.25, 4)]:
    v = system(kind="roles" if "three" in name else "solo", corr=corr)
    cap[name] = (v, radius)
    print(f"{name:>34}{v:>12.1%}{radius:>15}")

print(f"""
The first table is the comparison, and the middle two rows are the finding.

One agent with one prompt: {res['one agent, one prompt']:.1%}. The SAME agent
switching between a planning prompt, a working prompt and a checking prompt:
{res['one agent, role prompts']:.1%}. Three agents with those roles:
{res['three agents, roles']:.1%}.

**Adding roles made it worse, twice.** Once by spending {0.4:.0%} of the budget on
planning and criticism instead of on doing, and again by paying two handoffs to
distribute those roles across agents.

The last row is the exception and it is the whole chapter:
{res['three agents, decorrelated critic']:.1%}, from the same three agents with a
critic whose errors are decorrelated from the worker's.

So a role is not a mechanism. **A role is a prompt, and a prompt does not
decorrelate anything** (eq:roles-are-prompts). What produced the gain in the last
row was not the label "critic"; it was a reviewer that fails in different places,
which is ch:rsn-self-consistency's finding for the fourth time in this book.

The second table isolates that. Sweeping the critic's correlation with the worker
from {1.0} down to {0.0}, the three-agent design goes from {cc[1.0]:.1%} to
{cc[0.0]:.1%}, crossing the single agent's {solo:.1%} somewhere below {0.6}.

**A critic that shares most of the worker's blind spots is worse than no critic**,
because it costs budget and catches little -- and a critic implemented as the same
model with a different prompt shares almost everything. That is the standard
implementation.

The third table prices the budget split directly. Spending {0.4:.0%} of the budget
on doing gives {sp[0.4][1]:.1%} for the single agent and {sp[0.4][0]:.1%} with the
handoffs; spending {1.0:.0%} gives {sp[1.0][1]:.1%} and {sp[1.0][0]:.1%}.

Two costs, cleanly separated: **the budget the roles consume, and the handoffs
they require.** Neither is a property of having roles conceptually; both are
properties of implementing them as separate agents.

The fourth table is where roles genuinely earn their place, and it is not a
performance argument. Three agents with split CAPABILITIES reach
{cap['three agents, split capabilities'][0]:.1%} -- roughly the same as the
decorrelated-critic row, because splitting capabilities also decorrelates -- and
they halve the blast radius, from {cap['one agent, all capabilities'][1]} composed
risks to {cap['three agents, split capabilities'][1]}.

That is ch:ag-security's capability partition arriving as an architecture, and it
is the strongest case for role separation in this book: **a reader that cannot act
and an actor that cannot read private data are different SYSTEMS, not different
prompts** -- so they decorrelate, and they contain.

Which gives the rule. Roles are worth having when they carry different
capabilities, different credentials, or different model lineage. They are worth
nothing when they carry different instructions to the same model, and they cost
budget and handoffs either way.""")

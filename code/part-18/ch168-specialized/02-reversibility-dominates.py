# -*- coding: utf-8 -*-
# Extracted from: Chapter 168 — Specialized Agents: Research, Coding, Data, Browser, Computer-Use
# Source: src/.../ch168-specialized.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Why computer-use is hard, tested against three candidate explanations.

The usual explanation is the action space: a screen has thousands of clickable
points and an API has twelve endpoints, so of course clicking is harder.

This listing was written to test a second explanation against it -- that the real
problem is whether the agent can SEE the state it is acting on, since a screenshot
is a partial, stale, ambiguous observation and a filesystem read is not.

It also includes a third, added mostly for completeness: whether a mistake can be
undone. Coding has version control, so a bad edit costs a step. A sent email, a
deleted row, a submitted form cannot be taken back.

The measurement puts the third one first, which is not what the listing was built
to show (eq:reversibility-dominates), and finds the three strongly complementary
rather than substitutable (eq:domain-properties-are-complementary).
"""
import numpy as np

rng = np.random.default_rng(3767)

M = 40000
GOAL_STEPS = 8
BUDGET = 40
P_FATAL = 0.05     # share of un-undone mistakes that cannot be recovered


def run(n_actions, fidelity, undo, m=M, goal=GOAL_STEPS, budget=BUDGET,
        skill=3.2):
    """The agent needs `goal` correct actions. Each turn it observes the state;
    with probability `fidelity` the observation is faithful, otherwise it is
    misleading. Action choice is skill-weighted over `n_actions` candidates.

    A wrong action is undone with probability `undo`, costing only the turn.
    Otherwise it leaves damage: progress falls back a step, and a small share of
    un-undone mistakes are unrecoverable outright."""
    # Probability of the right action given a faithful observation: a
    # skill-weighted softmax over n candidates.
    p_right_true = skill / (skill + np.log2(n_actions))
    # Given a misleading observation, the agent is choosing at chance.
    p_right_false = 1.0 / n_actions
    prog = np.zeros(m, dtype=np.int64)
    alive = np.ones(m, dtype=bool)
    used = np.zeros(m, dtype=np.int64)
    for _ in range(budget):
        live = np.flatnonzero(alive & (prog < goal))
        if not len(live):
            break
        used[live] += 1
        faithful = rng.random(len(live)) < fidelity
        p = np.where(faithful, p_right_true, p_right_false)
        right = rng.random(len(live)) < p
        prog[live[right]] += 1
        wrong = live[~right]
        if len(wrong):
            stuck = wrong[rng.random(len(wrong)) >= undo]
            prog[stuck] = np.maximum(prog[stuck] - 1, 0)
            fatal = stuck[rng.random(len(stuck)) < P_FATAL]
            alive[fatal] = False
    done = alive & (prog >= goal)
    return float(done.mean()), float(used.mean())


# (name, action space, observation fidelity, undo probability)
PROFILES = [
    ("coding",       40,   0.97, 0.99),
    ("data",         25,   0.93, 0.90),
    ("research",    200,   0.85, 0.97),
    ("browser",     600,   0.70, 0.55),
    ("computer-use", 4000, 0.62, 0.40),
]

print(f"{M:,} tasks needing {GOAL_STEPS} correct actions within {BUDGET} turns.")
print()
print(f"{'domain':>14}{'actions':>9}{'fidelity':>10}{'undo':>7}"
      f"{'success':>10}{'turns':>8}")
print("-" * 58)
prof = {}
for name, n, f, u in PROFILES:
    r = run(n, f, u)
    prof[name] = (n, f, u, r[0], r[1])
    print(f"{name:>14}{n:>9}{f:>10.0%}{u:>7.0%}{r[0]:>10.1%}{r[1]:>8.1f}")

print()
print()
print("One variable at a time, the other two held at the coding profile.")
print()
print(f"{'action space':>14}{'success':>10}   {'fidelity':>10}{'success':>10}"
      f"   {'undo':>8}{'success':>10}")
print("-" * 68)
NS = (40, 200, 600, 1500, 4000)
FS = (0.97, 0.90, 0.80, 0.70, 0.62)
US = (0.99, 0.90, 0.75, 0.55, 0.40)
sw_n, sw_f, sw_u = {}, {}, {}
for n, f, u in zip(NS, FS, US):
    a = run(n, 0.97, 0.99)[0]
    b = run(40, f, 0.99)[0]
    c = run(40, 0.97, u)[0]
    sw_n[n], sw_f[f], sw_u[u] = a, b, c
    print(f"{n:>14}{a:>10.1%}   {f:>10.0%}{b:>10.1%}   {u:>8.0%}{c:>10.1%}")

print()
print()
print("Ranges over each sweep -- how much of the spread each variable explains.")
print()
rng_n = max(sw_n.values()) - min(sw_n.values())
rng_f = max(sw_f.values()) - min(sw_f.values())
rng_u = max(sw_u.values()) - min(sw_u.values())
for label, v in (("action space", rng_n), ("observation fidelity", rng_f),
                 ("undo availability", rng_u)):
    print(f"{label:>24}{v:>10.1%}")

print()
print()
print("The counterfactual that matters for tooling: give the computer-use")
print("profile ONE of the other domains' properties at a time.")
print()
n0, f0, u0 = 4000, 0.62, 0.40
print(f"{'computer-use, plus...':>28}{'success':>10}{'gain':>9}")
print("-" * 47)
cu = {}
base = run(n0, f0, u0)[0]
cu["baseline"] = base
print(f"{'(baseline)':>28}{base:>10.1%}{'--':>9}")
for label, kw in [("a 40-action interface", dict(n_actions=40)),
                  ("faithful observations", dict(fidelity=0.97)),
                  ("reliable undo", dict(undo=0.99)),
                  ("observations + undo", dict(fidelity=0.97, undo=0.99))]:
    args = dict(n_actions=n0, fidelity=f0, undo=u0)
    args.update(kw)
    v = run(**args)[0]
    cu[label] = v
    print(f"{label:>28}{v:>10.1%}{v - base:>+9.1%}")

print(f"""
The first table reproduces the ordering everyone expects, and the sweep underneath
it disagrees about why.

Held one at a time from the coding profile, moving the action space from
{40} to {4000} costs {sw_n[4000] - sw_n[40]:.1%}. Moving observation fidelity across
the full observed range costs {sw_f[0.62] - sw_f[0.97]:.1%}. Moving undo
availability costs {sw_u[0.40] - sw_u[0.99]:.1%}.

**Reversibility explains the largest share of the spread**
(eq:reversibility-dominates), and observation fidelity -- the variable this listing
was written to promote -- explains the smallest. That is worth stating plainly
because the fidelity story is the one usually told about computer-use, and at these
parameters it is the weakest of the three.

The counterfactual table is where it becomes actionable, and it contains a stronger
result than the sweep.

Give the computer-use profile a {40}-action interface and nothing else: it goes
from {cu['baseline']:.1%} to {cu['a 40-action interface']:.1%}. Give it faithful
observations and nothing else: {cu['faithful observations']:.1%}. Give it reliable
undo alone: {cu['reliable undo']:.1%}.

Give it faithful observations AND reliable undo: {cu['observations + undo']:.1%}.

**The properties are complementary rather than substitutable**
(eq:domain-properties-are-complementary). Two interventions worth
{cu['a 40-action interface'] - cu['baseline']:.1%} and
{cu['faithful observations'] - cu['baseline']:.1%} on their own are worth
{cu['observations + undo'] - cu['baseline']:.1%} together, because each one is
useless while another is binding. Seeing the state does not help if a mistaken
action cannot be taken back; being able to take actions back does not help if you
cannot see whether they were mistaken.

That explains something otherwise puzzling about this class of system: individual
improvements to computer-use agents often measure as near-worthless, and then a
combination measures as transformative. It is not that the individual measurements
were wrong. **In a domain with several binding constraints, the marginal value of
relieving one of them is near zero**, which makes incremental progress look like no
progress until the last constraint goes.

The practical reading is the same as the previous listing's and points at the same
place. Specialising for a domain is mostly not model work. It is building the
missing affordances -- a verifier, a faithful observation, an undo -- and building
them together, because relieving one at a time will not show up in the numbers.

And where the domain genuinely cannot offer an undo, ch:ag-termination's answer
stands: that is exactly where the human gate goes, and ch:as-long-running found
placing gates on those steps worth an eightfold review budget.""")

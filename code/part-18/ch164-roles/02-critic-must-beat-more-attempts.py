# -*- coding: utf-8 -*-
# Extracted from: Chapter 164 — Supervisor, Worker, Planner, and Critic Roles
# Source: src/.../ch164-roles.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The critic role, priced against the thing it is competing with.

The critic is the only role in the standard taxonomy with a mechanism rather than
a job description: it is supposed to catch errors the worker missed. That makes it
ch:rsn-self-consistency's critic problem and ch:ag-recovery's gating problem at
the same time, and both said the same thing -- the value is decorrelation, and a
weak signal that GATES is worse than no signal.

This listing puts a critic agent against the things it is competing with, and the
comparison is sharper than it first looks: more attempts are only worth having if
something can SELECT among them, and the critic is that something. So this is
ch:rsn-test-time-compute's coverage/selection decomposition with the critic in the
selector slot (eq:critic-must-beat-more-attempts).
"""
import numpy as np

rng = np.random.default_rng(3163)

M = 60000
P_TASK = 0.55           # a single attempt produces a correct result
BUDGET = 6              # model calls available per task


def outcome(design, budget=BUDGET, tp=0.72, fp=0.12, corr=0.85, check=0.0,
            m=M):
    """design:
      attempts   spend everything on attempts, keep the last
      critic     spend half on attempts, half on a critic that gates rework
      advise     the critic conditions a rework but never blocks a good result
      checker    an executable check of coverage `check` selects among attempts
    """
    if design == "attempts":
        # No selector: you keep the last attempt, so extra attempts buy nothing.
        return float(np.mean(rng.random(m) < P_TASK))
    if design == "coverage":
        # The ceiling: an oracle selector picking any correct attempt.
        ok = np.zeros(m, dtype=bool)
        for _ in range(budget):
            ok |= (~ok) & (rng.random(m) < P_TASK)
        return float(ok.mean())

    half = budget // 2
    ok = np.zeros(m, dtype=bool)
    for _ in range(half):
        ok |= (~ok) & (rng.random(m) < P_TASK)

    shares = rng.random(m) < corr
    flags_bad = (~ok) & (rng.random(m) < tp) & ~shares
    flags_good = ok & (rng.random(m) < fp)

    if design == "critic":
        # A flagged result is reworked; a good result flagged is also reworked
        # and may be broken by the rework.
        rework = flags_bad | flags_good
        redo = np.zeros(m, dtype=bool)
        for _ in range(budget - half):
            redo |= (~redo) & (rng.random(m) < P_TASK)
        out = np.where(rework, redo, ok)
        return float(out.mean())

    if design == "advise":
        # Only genuinely-flagged failures are reworked; good work is never
        # blocked, so a false positive costs nothing.
        redo = np.zeros(m, dtype=bool)
        for _ in range(budget - half):
            redo |= (~redo) & (rng.random(m) < P_TASK)
        return float((ok | (flags_bad & redo)).mean())

    if design == "checker":
        # An executable check of coverage `check` selects a correct attempt if
        # one exists among all `budget` attempts.
        got = np.zeros(m, dtype=bool)
        for _ in range(budget):
            got |= (~got) & (rng.random(m) < P_TASK)
        detected = got & (rng.random(m) < check)
        return float((detected | (got & (rng.random(m) < 0.3))).mean())
    raise ValueError(design)


print(f"{M:,} tasks. One attempt is correct {P_TASK:.0%} of the time;")
print(f"{BUDGET} model calls available. A critic catches a real failure 72% of")
print("the time, objects to good work 12%, and shares 85% of the worker's")
print("blind spots.")
print()
print(f"{'design':>34}{'completed':>12}{'vs no selector':>16}")
print("-" * 60)
base = outcome("attempts")
res = {}
for name, kw in [("more attempts, no selector", dict(design="attempts")),
                 ("half on a gating critic", dict(design="critic")),
                 ("half on an advising critic", dict(design="advise")),
                 ("executable check, 95% coverage",
                  dict(design="checker", check=0.95)),
                 ("oracle selector (the ceiling)", dict(design="coverage"))]:
    v = outcome(**kw)
    res[name] = v
    print(f"{name:>34}{v:>12.1%}{v - base:>+16.1%}")

print()
print()
print("The gating critic, swept over how correlated it is with the worker.")
print()
print(f"{'correlation':>13}{'gating':>10}{'advising':>11}{'no selector':>13}"
      f"{'best':>12}")
print("-" * 57)
cc = {}
for c in (0.95, 0.85, 0.6, 0.3, 0.0):
    g = outcome("critic", corr=c)
    a = outcome("advise", corr=c)
    cc[c] = (g, a)
    best = max([("gating", g), ("advising", a), ("none", base)],
               key=lambda x: x[1])[0]
    print(f"{c:>13.2f}{g:>10.1%}{a:>11.1%}{base:>13.1%}{best:>12}")

print()
print()
print("And swept over the critic's false-positive rate, which is what makes")
print("gating dangerous.")
print()
print(f"{'false positives':>17}{'gating':>10}{'advising':>11}{'gap':>9}")
print("-" * 47)
ff = {}
for f in (0.0, 0.05, 0.12, 0.25, 0.40):
    g = outcome("critic", fp=f, corr=0.3)
    a = outcome("advise", fp=f, corr=0.3)
    ff[f] = (g, a)
    print(f"{f:>17.0%}{g:>10.1%}{a:>11.1%}{a - g:>+9.1%}")

print()
print()
print("How the comparison moves with the budget. Without a selector the budget")
print("buys nothing; every other column is a way of cashing coverage in.")
print()
print(f"{'budget':>8}{'no selector':>13}{'gating':>10}{'advising':>11}"
      f"{'checker':>10}{'oracle':>10}")
print("-" * 50)
bd = {}
for b in (2, 4, 6, 10, 16):
    row = (outcome("attempts", budget=b), outcome("critic", budget=b),
           outcome("advise", budget=b),
           outcome("checker", budget=b, check=0.95),
           outcome("coverage", budget=b))
    bd[b] = row
    print(f"{b:>8}{row[0]:>13.1%}{row[1]:>10.1%}{row[2]:>11.1%}"
          f"{row[3]:>10.1%}{row[4]:>10.1%}")

print(f"""
The first table reframes what a critic is for, and the first row is the reframing.

**Without a selector, more attempts buy nothing.** You keep the last one, so the
budget column is flat at {res['more attempts, no selector']:.1%} no matter how
much you spend. That is ch:rsn-test-time-compute's coverage with nothing to cash
it in, and it is the situation a critic exists to fix.

An oracle selector reaches {res['oracle selector (the ceiling)']:.1%} -- the
ceiling. A gating critic reaches {res['half on a gating critic']:.1%}, an advising
one {res['half on an advising critic']:.1%}, and an executable check
{res['executable check, 95% coverage']:.1%}.

So the critic is doing real work: {res['half on a gating critic'] - res['more attempts, no selector']:+.1%}
over having no selector at all. **The critic role is a selector, and selectors are
the scarce component in this book** -- which is why this is the one role in the
standard taxonomy with a mechanism.

The second table sweeps its correlation with the worker, and the advising column
is consistently ahead. At {0.85} correlation -- which is what a critic implemented
as the same model with a different prompt actually is -- gating gives
{cc[0.85][0]:.1%} and advising {cc[0.85][1]:.1%}. At {0.0}, {cc[0.0][0]:.1%}
against {cc[0.0][1]:.1%}.

The third table says why. Sweeping the critic's false-positive rate, the gap
between gating and advising grows from {ff[0.0][1] - ff[0.0][0]:+.1%} at
{0:.0%} to {ff[0.4][1] - ff[0.4][0]:+.1%} at {0.4:.0%}.

**A gating critic can veto correct work; an advising one cannot.** That is
ch:ag-recovery's asymmetry exactly: a signal that only conditions a rework has a
floor at doing nothing, and one that also blocks has no floor
(eq:critic-must-beat-more-attempts). Since a critic's false-positive rate is the
error it is least often measured on, the advising configuration is the safe
default.

The last table is the one to size a critic against, and it contains a crossover
worth noticing.

At a budget of {2} the checker leads at {bd[2][3]:.1%} against the critic's
{bd[2][1]:.1%}. At {16} the critic leads, {bd[16][1]:.1%} against
{bd[16][3]:.1%} -- because the executable check's coverage is capped at
{0.95:.0%} and the critic gets repeated chances at the same task.

**An imperfect executable check is better at small budgets and worse at large
ones**, which is the opposite of the usual assumption that a real checker always
dominates a model-based one. The reason is that a check with fixed coverage has a
ceiling and a critic with repeated attempts does not.

The practical reading: use the executable check where it exists, keep a critic for
the part it does not cover, and configure the critic to advise rather than to
gate. And note that every column except the first requires the same thing -- a
selector -- which is ch:rsn-test-time-compute's conclusion arriving as an
organisational chart.""")

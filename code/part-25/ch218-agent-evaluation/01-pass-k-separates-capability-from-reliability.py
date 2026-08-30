# -*- coding: utf-8 -*-
# Extracted from: Chapter 218 — Agent and Tool-Call Evaluation
# Source: src/.../ch218-agent-evaluation.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A single-run success rate is a capability measurement. Users experience reliability.

cite:yao2024taubench reported state-of-the-art function-calling agents succeeding on under
50% of tasks, and -- the number that matters more -- pass^8 below 25% in the retail domain:
the probability that all eight independent attempts at the same task succeed.

Under independence those two numbers are irreconcilable. At p = 0.5, pass^8 would be 0.4%.
Observing 25% means the trials are strongly correlated: some tasks succeed every time and
some fail every time, and the single-run average is a mixture of the two
(eq:pass-k-separates-capability-from-reliability).

Which changes what a retry buys. Retrying a coin-flip task helps; retrying a task the agent
cannot do wastes the attempt and the user's patience
(eq:retry-does-not-help-a-deterministic-failure).
"""
# (task class, share, per-attempt success probability)
CLASSES = [
    ("reliably solvable", 0.31, 0.97),
    ("usually solvable",  0.19, 0.78),
    ("coin flip",         0.22, 0.47),
    ("rarely solvable",   0.13, 0.16),
    ("out of reach",      0.15, 0.02),
]

print("A task population with very different per-attempt reliabilities.")
print()
print(f"{'task class':>21}{'share':>9}{'p(success)':>13}"
      f"{'pass^1':>10}{'pass^4':>10}{'pass^8':>10}")
print("-" * 73)
for name, sh, p in CLASSES:
    print(f"{name:>21}{sh:>9.0%}{p:>13.2f}"
          f"{p:>10.2f}{p ** 4:>10.2f}{p ** 8:>10.2f}")


def pass_k(k):
    return sum(sh * p ** k for name, sh, p in CLASSES)


print("-" * 73)
print(f"{'POPULATION':>21}{1.0:>9.0%}{'':>13}"
      f"{pass_k(1):>10.2f}{pass_k(4):>10.2f}{pass_k(8):>10.2f}")

print()
print(f"single-run success {pass_k(1):.0%}, pass^8 {pass_k(8):.0%}")
print(f"under independence, {pass_k(1):.0%}^8 would be "
      f"{pass_k(1) ** 8:.2%} -- the gap is the correlation")

print()
print()
print("Where each level of pass^k gets its mass.")
print()
print(f"{'task class':>21}", end="")
for k in (1, 2, 4, 8, 16):
    print(f"{('share of pass^' + str(k)):>18}", end="")
print()
print("-" * 111)
contrib = {}
for name, sh, p in CLASSES:
    print(f"{name:>21}", end="")
    for k in (1, 2, 4, 8, 16):
        c = sh * p ** k / pass_k(k)
        contrib[(name, k)] = c
        print(f"{c:>18.1%}", end="")
    print()

print()
print(f"at pass^1 the reliably-solvable class is "
      f"{contrib[('reliably solvable', 1)]:.0%} of successes;")
print(f"at pass^8 it is {contrib[('reliably solvable', 8)]:.0%}")

print()
print()
print("Two improvements with the same effect on the headline number.")
print()


def population(classes):
    return {k: sum(sh * p ** k for n, sh, p in classes) for k in (1, 4, 8, 16)}


BASE = list(CLASSES)
# A: make the coin-flip tasks a bit more likely to work.
A = [(n, sh, p + (0.16 if n == "coin flip" else 0.0)) for n, sh, p in CLASSES]
# B: convert some coin-flip tasks into reliably solvable ones.
B = []
moved = 0.07
for n, sh, p in CLASSES:
    if n == "coin flip":
        B.append((n, sh - moved, p))
    elif n == "reliably solvable":
        B.append((n, sh + moved, p))
    else:
        B.append((n, sh, p))

print(f"{'model':>34}{'pass^1':>10}{'pass^4':>10}{'pass^8':>10}"
      f"{'pass^16':>11}")
print("-" * 75)
res = {}
for label, cls in (("baseline", BASE),
                   ("A: coin-flip tasks +0.16", A),
                   ("B: 7% of coin flips made reliable", B)):
    r = population(cls)
    res[label] = r
    print(f"{label:>34}{r[1]:>10.3f}{r[4]:>10.3f}{r[8]:>10.3f}"
          f"{r[16]:>11.3f}")

print()
print(f"A improves pass^1 by {res['A: coin-flip tasks +0.16'][1] - res['baseline'][1]:+.3f} "
      f"and pass^8 by {res['A: coin-flip tasks +0.16'][8] - res['baseline'][8]:+.3f}")
print(f"B improves pass^1 by {res['B: 7% of coin flips made reliable'][1] - res['baseline'][1]:+.3f} "
      f"and pass^8 by {res['B: 7% of coin flips made reliable'][8] - res['baseline'][8]:+.3f}")

print()
print()
print("What a user gets, if the product retries on failure.")
print()
print(f"{'retries allowed':>17}{'task succeeds':>16}{'attempts spent':>17}"
      f"{'attempts per success':>23}")
print("-" * 73)
retry = {}
for r in (1, 2, 3, 5, 8):
    succ = sum(sh * (1.0 - (1.0 - p) ** r) for n, sh, p in CLASSES)
    spent = sum(sh * sum((1.0 - p) ** (j - 1) for j in range(1, r + 1))
                for n, sh, p in CLASSES)
    retry[r] = (succ, spent, spent / succ)
    print(f"{r:>17}{succ:>16.3f}{spent:>17.2f}{spent / succ:>23.2f}")

print()
print()
print("And where those retry attempts go.")
print()
print(f"{'task class':>21}{'share of attempts at r=5':>27}"
       f"{'share of successes':>21}{'ratio':>9}")
print("-" * 78)
r5_spent = {}
for name, sh, p in CLASSES:
    spent = sh * sum((1.0 - p) ** (j - 1) for j in range(1, 6))
    gained = sh * (1.0 - (1.0 - p) ** 5)
    r5_spent[name] = (spent / retry[5][1], gained / retry[5][0])
    print(f"{name:>21}{spent / retry[5][1]:>27.1%}"
          f"{gained / retry[5][0]:>21.1%}"
          f"{(spent / retry[5][1]) / (gained / retry[5][0]):>9.2f}")

print()
print()
print("A retry budget spent where it pays: retry only the classes that")
print("respond to it, if you can tell them apart.")
print()
print(f"{'policy':>34}{'success':>11}{'attempts':>11}"
      f"{'attempts per success':>23}")
print("-" * 79)
POLICIES = [
    ("one attempt, no retries", lambda n, p: 1),
    ("five attempts, everything", lambda n, p: 5),
    ("five attempts if first fails fast", lambda n, p: 5 if p > 0.30 else 1),
    ("retry only the coin flips", lambda n, p: 5 if n == "coin flip" else 1),
]
pol = {}
for label, f in POLICIES:
    succ, spent = 0.0, 0.0
    for name, sh, p in CLASSES:
        r = f(name, p)
        succ += sh * (1.0 - (1.0 - p) ** r)
        spent += sh * sum((1.0 - p) ** (j - 1) for j in range(1, r + 1))
    pol[label] = (succ, spent, spent / succ)
    print(f"{label:>34}{succ:>11.3f}{spent:>11.2f}{spent / succ:>23.2f}")

print(f"""
The population table is cite:yao2024taubench's two numbers reconciled. Single-run success is
{pass_k(1):.0%} and pass^8 is {pass_k(8):.0%}; under independence the second would be
{pass_k(1) ** 8:.2%}. **The gap is entirely correlation between attempts**
(eq:pass-k-separates-capability-from-reliability), and the correlation is not mysterious:
some tasks are within the agent's reach every time and some are outside it every time.

Which means the single-run rate is measuring a mixture and reporting it as a capability. Two
systems with identical {pass_k(1):.0%} single-run rates can have completely different
populations behind them, and the user experience differs entirely.

The contribution table shows what pass^k is actually selecting for. At k=1 the
reliably-solvable class supplies {contrib[('reliably solvable', 1)]:.0%} of successes; at
k=8 it supplies {contrib[('reliably solvable', 8)]:.0%}. **Raising k filters out everything
except the tasks the agent can do consistently**, which is the correct definition of what an
agent can do.

The two-improvements table is the design consequence. Improvement A raises the coin-flip
tasks' success probability by {0.16:.2f} and moves pass^1 by
{res['A: coin-flip tasks +0.16'][1] - res['baseline'][1]:+.3f}; improvement B converts
{moved:.0%} of coin-flip tasks into reliable ones and moves pass^1 by
{res['B: 7% of coin flips made reliable'][1] - res['baseline'][1]:+.3f}.

On the headline number they are indistinguishable. On pass^8, A moves
{res['A: coin-flip tasks +0.16'][8] - res['baseline'][8]:+.3f} and B moves
{res['B: 7% of coin flips made reliable'][8] - res['baseline'][8]:+.3f} --
**B is worth {(res['B: 7% of coin flips made reliable'][8] - res['baseline'][8]) / (res['A: coin-flip tasks +0.16'][8] - res['baseline'][8]):.0f} times as much** on the
metric that describes what users can depend on, and it ties the comparison anyone would
actually run.

The retry table is the other half. Allowing {5} attempts takes success from
{retry[1][0]:.3f} to {retry[5][0]:.3f}, which sounds like a good trade until the last column:
attempts per success rises from {retry[1][2]:.2f} to {retry[5][2]:.2f}.

The allocation table says where they go. `out of reach` tasks are
{r5_spent['out of reach'][0]:.0%} of the attempts spent and
{r5_spent['out of reach'][1]:.0%} of the successes gained --
{r5_spent['out of reach'][0] / r5_spent['out of reach'][1]:.1f} times more consumption than
production. **Retrying a task the agent cannot do burns five attempts to arrive back where it
started** (eq:retry-does-not-help-a-deterministic-failure), and the user waits through all
five.

The policy table prices the fix and its ordering is instructive. Retrying everything reaches
{pol['five attempts, everything'][0]:.3f} at {pol['five attempts, everything'][1]:.2f}
attempts per task. Retrying only when the first attempt failed *fast* -- rather than after a
long doomed trajectory -- reaches {pol['five attempts if first fails fast'][0]:.3f} for
{pol['five attempts if first fails fast'][1]:.2f}, which is
{pol['five attempts, everything'][0] - pol['five attempts if first fails fast'][0]:.3f} less
success for {1 - pol['five attempts if first fails fast'][1] / pol['five attempts, everything'][1]:.0%}
fewer attempts.

Per success it is {pol['five attempts if first fails fast'][2]:.2f} against
{pol['five attempts, everything'][2]:.2f} -- **essentially the same as never retrying at
all**, while capturing most of the retry benefit.

The narrower policy in the last row, retrying only the class that responds best, is worse on
both axes than the fast-failure heuristic. That is the useful lesson: you do not need to
identify the recoverable tasks, only to *stop* on the unrecoverable ones, and a long failing
trajectory is a usable signal for that. A cheap negative filter beats an expensive positive
one.

Two things to carry out of this. **Report pass^k, not pass^1**, because the number users
experience is the one that requires the task to work every time. And **make retry
conditional**, because an unconditional retry budget is spent mostly on tasks that will not
respond to it. The next listing takes up what "succeed" should mean in the first place.""")

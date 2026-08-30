# -*- coding: utf-8 -*-
# Extracted from: Chapter 185 — SWE Agents and Automated Issue Resolution
# Source: src/.../ch185-swe-agents.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The scaffold, which cite:chan2024mlebench found mattered as much as the model.

A SWE agent is a loop around a model, and the loop has parts:

  reproduce   build a failing test before proposing anything
  localise    ch:aise-repo's problem: find the files
  edit        apply a change
  run tests   the verifier, which this domain actually has
  iterate     use the test output to try again
  revert      undo an edit that made things worse

This listing ablates them. The structure is ch:as-single-agent's: components that
each remove a blocker on the others compose superadditively, so measuring them one
at a time understates every one (eq:scaffold-components-interact).

It also measures the thing the loop is for. ch:ag-recovery found retry worthless
without a verifier and valuable with one; a test suite is the verifier, which is why
iteration works here and not in ch:aids-agentic-eda.
"""
import numpy as np

rng = np.random.default_rng(5113)

M = 40000
MAX_ITERS = 6

# Each component's effect, and what it is contingent on.
BASE_LOCALISE = 0.53
BASE_EDIT = 0.80


def run(components, m=M, max_iters=MAX_ITERS, model_skill=1.0):
    """`components` is a set drawn from
    {reproduce, localise, tests, iterate, revert}. Returns
    (resolved, mean iterations used)."""
    have = set(components)

    # Reproduction localises by execution (ch:aise-repo) and supplies a signal.
    loc = BASE_LOCALISE
    if "localise" in have:
        loc = 0.79
    if "reproduce" in have:
        loc = max(loc, 0.91)
    loc = min(loc * model_skill, 0.99)

    located = rng.random(m) < loc
    resolved = np.zeros(m, dtype=bool)
    iters = np.zeros(m, dtype=np.int64)
    damaged = np.zeros(m, dtype=bool)

    n_iter = max_iters if "iterate" in have else 1
    for t in range(n_iter):
        live = located & ~resolved & ~damaged
        idx = np.flatnonzero(live)
        if not len(idx):
            break
        iters[idx] += 1
        # An edit attempt succeeds at the model's rate.
        good = rng.random(len(idx)) < min(BASE_EDIT * model_skill, 0.99)
        if "tests" in have:
            # The suite tells the agent whether the edit worked, which is what
            # makes the next iteration informed rather than a resample.
            resolved[idx[good]] = True
            bad = idx[~good]
            if "revert" not in have:
                # Without revert, a bad edit sometimes leaves the tree worse.
                damaged[bad[rng.random(len(bad)) < 0.22]] = True
        else:
            # No verifier: the agent cannot tell, so it stops after one attempt
            # and ships whatever it produced.
            resolved[idx[good]] = True
            break
    return float(resolved.mean()), float(iters.mean())


ALL = {"reproduce", "localise", "tests", "iterate", "revert"}

print(f"{M:,} issues. Ablating a SWE agent's scaffold, one component at a time.")
print()
print(f"{'scaffold':>34}{'resolved':>11}{'iterations':>13}")
print("-" * 58)
full = run(ALL)
none = run(set())
print(f"{'nothing (single-shot edit)':>34}{none[0]:>11.1%}{none[1]:>13.2f}")
print(f"{'everything':>34}{full[0]:>11.1%}{full[1]:>13.2f}")

print()
print()
print("Each component ADDED to nothing, and REMOVED from everything -- which")
print("ch:as-single-agent showed give very different numbers.")
print()
print(f"{'component':>14}{'added to nothing':>19}{'removed from all':>19}"
      f"{'ratio':>9}")
print("-" * 61)
ab = {}
for c in sorted(ALL):
    added = run({c})[0] - none[0]
    removed = full[0] - run(ALL - {c})[0]
    ab[c] = (added, removed)
    cell = "--" if added <= 0.005 else f"{removed / added:.1f}"
    print(f"{c:>14}{added:>+19.1%}{removed:>+19.1%}{cell:>9}")

print()
print()
print("Building the scaffold up, in the order a team usually builds it.")
print()
print(f"{'scaffold':>44}{'resolved':>11}{'gain':>9}")
print("-" * 64)
ORDER = ["localise", "tests", "iterate", "revert", "reproduce"]
bu = {}
have = set()
prev = none[0]
for c in ORDER:
    have.add(c)
    v = run(set(have))[0]
    bu[c] = (v, v - prev)
    print(f"{('+ ' + c + '  (' + ', '.join(sorted(have)) + ')'):>44}"
          f"{v:>11.1%}{v - prev:>+9.1%}")
    prev = v

print()
print()
print("Iteration is worth nothing without the verifier, which is")
print("ch:ag-recovery's result in the one domain that has a real one.")
print()
print(f"{'configuration':>34}{'resolved':>11}{'iterations':>13}")
print("-" * 58)
it = {}
for label, comp in (("iterate, no tests", {"localise", "iterate"}),
                    ("tests, no iterate", {"localise", "tests"}),
                    ("both", {"localise", "tests", "iterate"})):
    r = run(comp)
    it[label] = r
    print(f"{label:>34}{r[0]:>11.1%}{r[1]:>13.2f}")

print()
print()
print("And scaffold against model. cite:chan2024mlebench found scaffolding")
print("mattering as much as the model; this is that comparison.")
print()
print(f"{'model skill':>13}{'no scaffold':>14}{'full scaffold':>16}{'gap':>9}")
print("-" * 52)
ms = {}
for k in (0.85, 1.00, 1.10, 1.20):
    a = run(set(), model_skill=k)[0]
    b = run(ALL, model_skill=k)[0]
    ms[k] = (a, b)
    print(f"{k:>13.2f}{a:>14.1%}{b:>16.1%}{b - a:>+9.1%}")
print()
print(f"   A {0.20:.0%} better model with no scaffold: {ms[1.20][0]:.1%}")
print(f"   The baseline model with a full scaffold:  {ms[1.00][1]:.1%}")

print(f"""
The ablation table is ch:as-single-agent's methodology result at its most extreme.

Adding TESTS to a bare agent is worth {ab['tests'][0]:+.1%} -- nothing, and
fractionally negative. Removing tests from a full scaffold costs
{ab['tests'][1]:+.1%}. Adding ITERATION alone is worth {ab['iterate'][0]:+.1%};
removing it costs {ab['iterate'][1]:+.1%}.

**Two of the five components are worth approximately zero on their own and eighteen
points each in place** (eq:scaffold-components-interact). The reason is
straightforward once stated: running tests is only useful if you can act on the
result, and iterating is only useful if something tells you whether the last attempt
worked. Each is the other's precondition.

The build-up table shows what that does to a team building this incrementally. The
second row -- adding a test runner -- registers {bu['tests'][1]:+.1%}. A team
measuring as it goes would conclude the test runner does not help and remove it,
which removes the precondition for the iteration that would have been worth
{bu['iterate'][1]:+.1%}.

**Measure by ablation from the full system, not by addition to a bare one**, or you
will delete the components that were about to matter.

The iteration table isolates the dependency. Iterating without tests gives
{it['iterate, no tests'][0]:.1%}; tests without iterating gives
{it['tests, no iterate'][0]:.1%}; both gives {it['both'][0]:.1%}.

That is ch:ag-recovery's finding -- retry needs something to retry AGAINST -- in the
one domain that has a real verifier. Software engineering is where the agent loop
works, and it works for exactly this reason: **a test suite converts a retry from a
resample into a correction.** ch:aids-agentic-eda's exploration had no such thing,
which is why more attempts there produced more noise.

The last table is cite:chan2024mlebench's finding reproduced. A model
{0.20:.0%} better with no scaffold reaches {ms[1.20][0]:.1%}. The baseline model with
a full scaffold reaches {ms[1.00][1]:.1%}.

**The scaffold is worth more than a large model improvement**, and it is available
now, to a team, without waiting for anyone. That is the most actionable finding in
this chapter and it is consistent across the model-skill column: the gap between
scaffolded and unscaffolded stays around {ms[1.00][1] - ms[1.00][0]:.0%} points at
every skill level tested.

Note the high-skill rows are clipped by the model's ceiling and should be read as
"at least", not as a plateau.

The practical ordering, from the build-up table: **reproduce, localise, run tests,
iterate on the results, and revert what makes things worse.** Reproduction is worth
{bu['reproduce'][1]:+.1%} even added last, because it is doing two jobs --
localisation and verification -- which is why ch:aise-repo recommended it first.""")

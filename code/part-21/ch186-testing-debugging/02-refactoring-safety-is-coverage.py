# -*- coding: utf-8 -*-
# Extracted from: Chapter 186 — Test Generation, Debugging, and Refactoring Agents
# Source: src/.../ch186-testing-debugging.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Refactoring, where the verifier has to already exist.

Debugging and refactoring look like the same activity to an agent -- read code,
change code, run tests -- and they have opposite verifier profiles.

Debugging starts from a FAILING test. The verifier is handed to you: the bug is
demonstrated, and the fix is confirmed when the test goes green. That is the
strongest position an agent occupies anywhere in this book.

Refactoring starts from passing tests and must PRESERVE behaviour. The verifier is
the existing suite, so the safety of a refactor is bounded by the coverage of the
code being refactored (eq:refactoring-safety-is-coverage).

Which produces an unfortunate correlation: the code most in need of refactoring --
old, tangled, poorly understood -- is the code least likely to be well tested
(eq:worst-code-is-least-covered).
"""
import numpy as np

rng = np.random.default_rng(5179)

M = 60000
P_REFACTOR_BREAKS = 0.19    # a refactor changes behaviour it should not
P_FIX_WORKS = 0.80


def debug(coverage, m=M, p_fix=P_FIX_WORKS):
    """A failing test exists. Returns (fixed and verified, shipped broken)."""
    fixed = rng.random(m) < p_fix
    # The failing test confirms the fix directly; the rest of the suite catches
    # collateral damage to the extent it covers it.
    collateral = ~fixed | (rng.random(m) < 0.12)
    caught = collateral & (rng.random(m) < coverage)
    ok = fixed & ~(collateral & ~caught)
    return float(ok.mean()), float((collateral & ~caught).mean())


def refactor(coverage, m=M, p_break=P_REFACTOR_BREAKS, extra=0.0):
    """No failing test; behaviour must be preserved. `extra` is characterisation
    tests written before refactoring."""
    cov = coverage + (1.0 - coverage) * extra
    broke = rng.random(m) < p_break
    caught = broke & (rng.random(m) < cov)
    return float((~broke | caught).mean()), float((broke & ~caught).mean()), cov


print(f"{M:,} changes. Debugging starts from a failing test; refactoring starts")
print("from passing ones and must keep them passing for the right reason.")
print()
print(f"{'coverage':>10}{'debug: verified':>17}{'debug: broken':>15}"
      f"{'refactor: safe':>16}{'refactor: broken':>18}")
print("-" * 76)
tab = {}
for c in (0.25, 0.45, 0.70, 0.90):
    d = debug(c)
    r = refactor(c)
    tab[c] = (d, r)
    print(f"{c:>10.0%}{d[0]:>17.1%}{d[1]:>15.1%}{r[0]:>16.1%}{r[1]:>18.1%}")

print()
print()
print("The asymmetry: debugging's outcome barely moves with coverage because")
print("the failing test IS the verifier. Refactoring's moves a lot.")
print()
print(f"{'':>26}{'at 25% coverage':>18}{'at 90% coverage':>18}{'range':>9}")
print("-" * 71)
print(f"{'debugging verified':>26}{tab[0.25][0][0]:>18.1%}"
      f"{tab[0.90][0][0]:>18.1%}{tab[0.90][0][0] - tab[0.25][0][0]:>+9.1%}")
print(f"{'refactoring safe':>26}{tab[0.25][1][0]:>18.1%}"
      f"{tab[0.90][1][0]:>18.1%}{tab[0.90][1][0] - tab[0.25][1][0]:>+9.1%}")

print()
print()
print("Now the correlation that matters. Code is refactored because it is bad,")
print("and bad code is undertested.")
print()
print(f"{'code quality':>16}{'typical coverage':>18}{'refactor safe':>15}"
      f"{'broken silently':>17}")
print("-" * 66)
QUALITY = [("well-maintained", 0.85), ("ordinary", 0.62),
           ("legacy, tangled", 0.34), ("nobody understands it", 0.15)]
q = {}
for label, cov in QUALITY:
    r = refactor(cov)
    q[label] = r
    print(f"{label:>16}{cov:>18.0%}{r[0]:>15.1%}{r[1]:>17.1%}")

print()
print()
print("Characterisation tests: write tests that pin the CURRENT behaviour")
print("before touching anything. They do not need to be right about intent --")
print("only about what the code does now, which is exactly what a model")
print("generating tests from code is good at.")
print()
print(f"{'starting coverage':>19}{'no characterisation':>21}"
      f"{'with them':>12}{'gain':>9}")
print("-" * 61)
ch = {}
for cov in (0.15, 0.34, 0.62, 0.85):
    a = refactor(cov)[0]
    b = refactor(cov, extra=0.75)[0]
    ch[cov] = (a, b)
    print(f"{cov:>19.0%}{a:>21.1%}{b:>12.1%}{b - a:>+9.1%}")

print()
print()
print("Which resolves ch:aise-testing's first listing. Generated tests are a")
print("bad specification and an excellent CHARACTERISATION -- the failure mode")
print("there is the requirement here.")
print()
print(f"{'use':>34}{'what it needs':>22}{'generated tests':>18}")
print("-" * 74)
print(f"{'verify a fix is correct':>34}{'independent intent':>22}{'poor':>18}")
print(f"{'pin current behaviour':>34}{'accurate observation':>22}"
      f"{'excellent':>18}")

print()
print()
print("And the ordering that follows for a refactoring agent.")
print()
print(f"{'procedure':>44}{'safe':>10}{'broken':>10}")
print("-" * 64)
seq = {}
for label, cov, extra in (("refactor legacy code directly", 0.34, 0.0),
                          ("characterise, then refactor", 0.34, 0.75),
                          ("characterise, then refactor, ordinary code",
                           0.62, 0.75)):
    r = refactor(cov, extra=extra)
    seq[label] = r
    print(f"{label:>44}{r[0]:>10.1%}{r[1]:>10.1%}")

print(f"""
The first two tables are the asymmetry, and it is larger than it looks.

Debugging's verified rate moves {tab[0.90][0][0] - tab[0.25][0][0]:+.1%} across the
whole coverage range, from {tab[0.25][0][0]:.1%} to {tab[0.90][0][0]:.1%}.
Refactoring's moves {tab[0.90][1][0] - tab[0.25][1][0]:+.1%}.

The reason is that **debugging is handed its verifier and refactoring has to find
one already there** (eq:refactoring-safety-is-coverage). A failing test demonstrates
the bug and confirms the fix; it exists before the agent starts. A refactor has no
such artefact -- the only evidence that behaviour was preserved is a suite that was
written for other reasons and covers what it happens to cover.

So debugging is the strongest position an agent occupies anywhere in this book, and
refactoring is one of the weakest, and they look identical from the outside: read
code, change code, run tests.

The third table is the correlation that makes this bite. Well-maintained code
refactors safely {q['well-maintained'][0]:.1%} of the time; code nobody understands,
{q['nobody understands it'][0]:.1%}, with {q['nobody understands it'][1]:.1%} broken
silently.

**The code most in need of refactoring is the code least able to verify a refactor**
(eq:worst-code-is-least-covered). That is not a coincidence -- undertested and
tangled are the same history -- and it means the highest-value refactoring targets
are exactly the ones an agent should be least trusted with.

The fourth table is the resolution, and it is why this chapter's two listings belong
together.

**Characterisation tests** pin what the code does now, before anything is touched.
They make no claim about what it SHOULD do; they are a snapshot, and their only
requirement is that they accurately describe current behaviour.

Which is precisely what the previous listing found generated tests to be good at.
Tests written from the code achieved the highest coverage in that table and the
lowest defect detection -- because they describe the implementation rather than the
intent. **The failure mode there is the requirement here.**

Adding characterisation tests to legacy code takes refactoring safety from
{ch[0.34][0]:.1%} to {ch[0.34][1]:.1%}, and on the worst code from
{ch[0.15][0]:.1%} to {ch[0.15][1]:.1%} -- the largest gains exactly where the
starting coverage is worst.

The last table gives the procedure. Refactoring legacy code directly is
{seq['refactor legacy code directly'][0]:.1%} safe; characterising first takes it to
{seq['characterise, then refactor'][0]:.1%}, which is better than refactoring
ORDINARY code without characterisation.

So the recommendations separate cleanly by task, and they are opposite:

**For debugging, use the failing test as the verifier and do not generate tests from
the code** -- you already have the independent artefact you need.

**For refactoring, generate tests from the code deliberately, before touching it.**
The generated suite's dependence on the implementation is the property that makes it
a correct characterisation.

**And never use a characterisation suite to verify a FIX.** It asserts the old
behaviour, which is what the fix is supposed to change -- the two uses require
opposite properties from the same artefact, and confusing them is the failure this
chapter is about.""")

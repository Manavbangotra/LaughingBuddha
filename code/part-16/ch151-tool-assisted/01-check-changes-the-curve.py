# -*- coding: utf-8 -*-
# Extracted from: Chapter 151 — Tool-Assisted and Verified Reasoning
# Source: src/.../ch151-tool-assisted.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""An executable check changes the SHAPE of the accuracy/compute curve.

ch:rsn-test-time-compute measured a verifier of quality q recovering q of the
available coverage, and left the obvious question open: what happens when q is
1? An executable check -- a test suite, a type checker, a solver, an interpreter
-- is the case where it can be, and the interesting part is not that accuracy
goes up. It is that the CURVE changes from one that saturates to one that keeps
climbing with the sample budget (eq:check-turns-coverage-into-accuracy).

The second half is the part that gets skipped. An executable check does not
verify correctness. It verifies the PROPERTY IT CHECKS, and the two coincide only
if the specification is complete. This listing sweeps how complete it is, and the
result is that the same extra compute which is pure profit under a complete check
becomes adversarial pressure under an incomplete one.
"""
import numpy as np

rng = np.random.default_rng(1117)

N_PROB = 5000
NMAX = 128
P_SOLVE = 0.09            # chance one sample is fully correct
N_BUGS = 5                # distinct ways a sample can be wrong


def make_pool(n_prob, ns):
    """Each sample is either correct (0) or carries one of N_BUGS defects.
    Defects are drawn with a skew, so some are far more common than others --
    which is what makes an incomplete specification dangerous rather than merely
    incomplete."""
    correct = rng.random((n_prob, ns)) < P_SOLVE
    w = np.array([0.40, 0.25, 0.18, 0.12, 0.05])
    bug = rng.choice(np.arange(1, N_BUGS + 1), size=(n_prob, ns), p=w)
    return np.where(correct, 0, bug)


POOL = make_pool(N_PROB, NMAX)
ROW = np.arange(N_PROB)


def learned_score(pool, mu):
    """A learned verifier: correct samples score N(mu, 1), wrong ones N(0, 1)."""
    return rng.normal(size=pool.shape) + mu * (pool == 0)


def check_catches(pool, caught):
    """An executable check that detects the defects in `caught` and is silent
    about the rest. A sample PASSES if it is correct, or if its defect is one
    the specification never mentions."""
    passes = pool == 0
    for b in range(1, N_BUGS + 1):
        if b not in caught:
            passes |= (pool == b)
    return passes


BUDGETS = [1, 2, 4, 8, 16, 32, 64, 128]
MU = 1.4                  # a decent learned verifier

print(f"{N_PROB} problems. One sample is fully correct {P_SOLVE:.0%} of the")
print(f"time; otherwise it carries one of {N_BUGS} defects, drawn with a skew.")
print()
print("Selection by a learned verifier, versus by an executable check that")
print("catches every defect.")
print()
print(f"{'samples n':>10}{'coverage':>11}{'learned':>11}{'complete':>11}")
print(f"{'':>10}{'':>11}{'verifier':>11}{'check':>11}")
print("-" * 43)

S_LEARNED = learned_score(POOL, MU)
PASS_ALL = check_catches(POOL, set(range(1, N_BUGS + 1)))
cov, lrn, chk = {}, {}, {}
for n in BUDGETS:
    sub = POOL[:, :n]
    cov[n] = float((sub == 0).any(1).mean())
    idx = S_LEARNED[:, :n].argmax(1)
    lrn[n] = float((sub[ROW, idx] == 0).mean())
    # With a check you keep the FIRST sample that passes; if none passes you are
    # left with the last one you tried.
    p = PASS_ALL[:, :n]
    first = np.where(p.any(1), p.argmax(1), n - 1)
    chk[n] = float((sub[ROW, first] == 0).mean())
    print(f"{n:>10}{cov[n]:>11.1%}{lrn[n]:>11.1%}{chk[n]:>11.1%}")

print()
print()
print("Now make the specification incomplete. The check catches the defects it")
print("knows about and passes the rest. Accuracy at each budget:")
print()
CASES = [("catches all 5", set(range(1, 6))),
         ("catches 4 of 5 (misses the rarest)", {1, 2, 3, 4}),
         ("catches 3 of 5", {1, 2, 3}),
         ("catches 2 of 5 (the two commonest)", {1, 2})]
print(f"{'specification':>36}" + "".join(f"{'n=' + str(n):>9}"
                                         for n in (1, 8, 32, 128)))
print("-" * 72)
inc = {}
for name, caught in CASES:
    p = check_catches(POOL, caught)
    row = {}
    for n in (1, 8, 32, 128):
        pn = p[:, :n]
        first = np.where(pn.any(1), pn.argmax(1), n - 1)
        row[n] = float((POOL[ROW, first][:, None][:, 0] == 0).mean()) \
            if False else float((POOL[:, :n][ROW, first] == 0).mean())
    inc[name] = row
    print(f"{name:>36}" + "".join(f"{row[n]:>9.1%}" for n in (1, 8, 32, 128)))

print()
print()
print("What is the system SHIPPING? Among the samples that passed the check,")
print("what fraction are actually correct, and what does the rest consist of?")
print()
print(f"{'specification':>36}{'shipped':>10}{'shipped':>10}{'and passed':>13}")
print(f"{'':>36}{'at n=1':>10}{'at n=128':>10}{'the check':>13}")
print("-" * 69)
ship = {}
for name, caught in CASES:
    p = check_catches(POOL, caught)
    out = []
    for n in (1, 128):
        pn = p[:, :n]
        first = np.where(pn.any(1), pn.argmax(1), n - 1)
        sel = POOL[:, :n][ROW, first]
        passed = pn[ROW, first]
        out.append(float((sel[passed] == 0).mean()) if passed.any() else 0.0)
    ship[name] = (out[0], out[1], float(p[:, :128].any(1).mean()))
    print(f"{name:>36}{out[0]:>10.1%}{out[1]:>10.1%}{ship[name][2]:>13.1%}")

full = inc["catches all 5"]
four = inc["catches 4 of 5 (misses the rarest)"]
two = inc["catches 2 of 5 (the two commonest)"]
S4 = ship["catches 4 of 5 (misses the rarest)"]
S2 = ship["catches 2 of 5 (the two commonest)"]
print(f"""
The first table is the shape change, and the columns matter more than any single
number in them.

The learned verifier goes {lrn[1]:.1%} -> {lrn[128]:.1%} over a 128x budget. The
complete check goes {chk[1]:.1%} -> {chk[128]:.1%}, tracking coverage
({cov[128]:.1%}) exactly at every budget.

Those are different kinds of curve, not the same curve at different heights. The
learned verifier's gains flatten because it competes against its own false
positives -- ch:rsn-self-consistency's extreme-value problem, where the top score
over a growing pool increasingly belongs to whichever wrong sample scored
luckiest. A complete check has no false positives on the property it checks, so
each extra sample is a clean additional draw and delivered accuracy IS coverage
(eq:check-turns-coverage-into-accuracy).

That is the argument for tool-assisted reasoning stated quantitatively, and it is
not "a checker is a better verifier". It is that a checker moves the binding
constraint OFF the verifier and ONTO the sample budget -- and the sample budget is
the one thing in this whole part that you can simply buy.

It also changes the problem the model is being asked to solve. Without a check
the model must GENERATE a correct answer. With one it must generate a correct
answer somewhere in {NMAX} tries, which is a search problem, and
cite:brown2024monkeys's coverage curve says search is exactly what sampling is
good at.

The second table is what is usually left out, and it changes the conclusion from
"tools fix this" to something more precise.

An executable check does not verify correctness. It verifies the property it
checks. Give it an incomplete specification and the curve does not merely sit
lower -- it stops climbing. Catching four defects out of five reaches
{four[8]:.1%} at n=8 and {four[128]:.1%} at n=128, having gained
{four[128] - four[32]:+.1%} over the last four doublings. Catching two of five
saturates at {two[128]:.1%}.

Compare that with the complete check, which was still gaining
{full[128] - full[32]:+.1%} over the same range and reached {full[128]:.1%}.

**The ceiling is the specification, not the budget.** Once sampling has found a
sample the check accepts, additional compute cannot improve on it, and the check
accepts anything whose defect the specification never mentioned.

The third table says what that means for what actually ships, and it is the
number to take away from this chapter.

At n=128, every specification -- including the ones catching two defects out of
five -- produces a passing sample for {S2[2]:.0%} of problems. The check is green
everywhere. Among those passing samples, the fraction that are actually correct
is {ship['catches all 5'][1]:.1%}, {S4[1]:.1%}, {ship['catches 3 of 5'][1]:.1%}
and {S2[1]:.1%} respectively.

So a system with a two-of-five specification reports a {S2[2]:.0%} pass rate and
ships defective work {1 - S2[1]:.1%} of the time.

Note what did NOT happen, because the usual telling of this story overstates it.
Shipped correctness barely moved with the budget: {S2[0]:.1%} at n=1 against
{S2[1]:.1%} at n=128. Sampling harder did not corrupt the output. What it did was
drive the PASS RATE to {S2[2]:.0%} while correctness stayed flat, which destroys
the check's value as a signal rather than its value as a filter. Before the extra
compute, a failing check told you something. After it, the check passes on
everything and tells you nothing.

That is the quiet failure mode, and it is quieter than the one tools were brought
in to fix. A learned verifier that is wrong gives you a mediocre number. An
incomplete executable check gives you a green run.

Which is the rule this chapter exists to state. **A tool moves the binding
constraint from verification to specification.** That is a large improvement,
because a specification is a thing you can read, version, extend, review and
argue about, and a learned verifier's failure modes are none of those. It is not
the same as removing the constraint. The test suite is now both the evaluation
harness and the optimisation target, and those two roles have different
requirements: a harness needs to be representative, a target needs to be
COMPLETE, and the gap between them is exactly what extra compute will find.""")

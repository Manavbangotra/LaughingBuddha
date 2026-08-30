# -*- coding: utf-8 -*-
# Extracted from: Chapter 148 — Test-Time Compute and Search
# Source: src/.../ch148-test-time-compute.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Sampling more does not mean answering better: coverage versus selection.

cite:brown2024monkeys measured coverage -- the fraction of problems solved by AT
LEAST ONE of n samples -- rising log-linearly in n, from 15.9% at one sample to
56% at 250 on SWE-bench Lite. That is a real and large effect, and it is a
statement about the GENERATOR.

Turning it into accuracy requires picking which sample to keep, which is a
statement about the SELECTOR. This listing separates the two on a task where the
distribution of wrong answers is produced by the actual dynamics rather than
specified by hand (eq:coverage-selection-gap).

The task is ch:rsn-cot's: iterate a fixed permutation for k steps. The generator
is a stepper with two kinds of error, because real generators have two kinds.
On most states it is right with probability p and slips at random. On a minority
of CONFUSED states it applies a consistently wrong rule -- a misconception rather
than noise -- which is what makes some wrong answers far more likely than others.
The distribution of wrong answers is then produced by the task's own dynamics.
"""
import numpy as np
from collections import Counter

rng = np.random.default_rng(419)

N = 48
PERM = rng.permutation(N)
WRONG = rng.permutation(N)          # the consistently-wrong rule
CONFUSED = set(rng.choice(N, size=N // 3, replace=False).tolist())
K = 6                       # steps per chain
P_STEP = 0.93               # per-step accuracy on states it is not confused by
P_CONF = 0.12               # per-step accuracy on states it IS confused by
N_PROB = 3000               # problems
MAXS = 256                  # samples per problem


def truth(x, k=K):
    for _ in range(k):
        x = PERM[x]
    return x


def sample_systematic(x):
    """On a confused state the generator usually applies its own wrong rule, so
    many samples land on the SAME wrong answer."""
    for _ in range(K):
        if x in CONFUSED:
            x = PERM[x] if rng.random() < P_CONF else int(WRONG[x])
        else:
            x = PERM[x] if rng.random() < P_STEP else int(rng.integers(N))
    return int(x)


def sample_unsystematic(x, p):
    """The control. Same task, same chain length, errors go somewhere at
    random -- so wrong answers spread instead of piling up."""
    for _ in range(K):
        x = PERM[x] if rng.random() < p else int(rng.integers(N))
    return int(x)


starts = rng.integers(N, size=N_PROB)
answers = np.array([truth(int(s)) for s in starts])

pool = np.array([[sample_systematic(int(s)) for _ in range(MAXS)]
                 for s in starts])
base_acc = float(np.mean(pool[:, 0] == answers))

# Match the control's single-sample accuracy to the systematic generator's, so
# the comparison is about the SHAPE of the errors and not their rate.
lo, hi = 0.3, 0.999
for _ in range(28):
    mid = (lo + hi) / 2
    a = float(np.mean([sample_unsystematic(int(s), mid) == answers[i]
                       for i, s in enumerate(starts)]))
    if a < base_acc:
        lo = mid
    else:
        hi = mid
P_FLAT = (lo + hi) / 2
pool_flat = np.array([[sample_unsystematic(int(s), P_FLAT) for _ in range(MAXS)]
                      for s in starts])


def majority(row):
    return Counter(row.tolist()).most_common(1)[0][0]


def verifier_pick(row, correct, q):
    """A verifier of quality q. With probability q it recognises a correct
    sample when the pool contains one; otherwise it picks uniformly at random.
    q = 1 is the oracle, q = 0 is single-sample-equivalent selection."""
    good = np.flatnonzero(row == correct)
    if len(good) and rng.random() < q:
        return correct
    return int(row[rng.integers(len(row))])


BUDGETS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
QS = [0.5, 0.8, 1.0]

print(f"{N_PROB} problems, {K} steps each on {N} states. The generator is")
print(f"{P_STEP:.0%} accurate per step on ordinary states and {P_CONF:.0%} on the")
print(f"{len(CONFUSED)} states it is confused by, where its errors are SYSTEMATIC.")
print()
print(f"A control generator makes the SAME single-sample accuracy "
      f"({base_acc:.1%}) with")
print(f"errors that go somewhere at random (per-step {P_FLAT:.1%}), so only the")
print("SHAPE of the errors differs between the two generators.")
print()
print(f"{'':>10}{'SYSTEMATIC errors':>25}{'RANDOM errors':>25}")
print(f"{'samples n':>10}{'coverage':>13}{'majority':>12}"
      f"{'coverage':>13}{'majority':>12}")
print("-" * 60)

cov, maj, covf, majf = {}, {}, {}, {}
for n in BUDGETS:
    sub, subf = pool[:, :n], pool_flat[:, :n]
    cov[n] = float(np.mean([(sub[i] == answers[i]).any() for i in range(N_PROB)]))
    maj[n] = float(np.mean([majority(sub[i]) == answers[i] for i in range(N_PROB)]))
    covf[n] = float(np.mean([(subf[i] == answers[i]).any() for i in range(N_PROB)]))
    majf[n] = float(np.mean([majority(subf[i]) == answers[i] for i in range(N_PROB)]))
    print(f"{n:>10}{cov[n]:>13.1%}{maj[n]:>12.1%}{covf[n]:>13.1%}{majf[n]:>12.1%}")

print()
print()
print("Selection on the SYSTEMATIC pool, by verifier quality. q is the chance")
print("the verifier finds a correct sample when the pool contains one.")
print()
print(f"{'samples n':>10}{'coverage':>11}"
      + "".join(f"{'verifier q=' + str(q):>16}" for q in QS))
print("-" * 69)
ver = {q: {} for q in QS}
for n in BUDGETS:
    sub = pool[:, :n]
    row = f"{n:>10}{cov[n]:>11.1%}"
    for q in QS:
        v = float(np.mean([verifier_pick(sub[i], answers[i], q) == answers[i]
                           for i in range(N_PROB)]))
        ver[q][n] = v
        row += f"{v:>16.1%}"
    print(row)

print()
print()
print("Is coverage log-linear in n, as cite:brown2024monkeys reports? Fit")
print("coverage against log2(n) and look at the residuals.")
print()
xs = np.log2(np.array(BUDGETS, float))
ys = np.array([cov[n] for n in BUDGETS])
A = np.stack([xs, np.ones_like(xs)], 1)
slope, icpt = np.linalg.lstsq(A, ys, rcond=None)[0]
print(f"{'samples n':>10}{'coverage':>11}{'log-linear fit':>17}{'residual':>11}")
print("-" * 49)
for n, y in zip(BUDGETS, ys):
    f = slope * np.log2(n) + icpt
    print(f"{n:>10}{y:>11.1%}{f:>17.1%}{y - f:>+11.1%}")
print()
print(f"  fitted slope: {slope:+.3f} coverage per doubling of n")

print()
print()
print("Where does the majority vote's ceiling come from? For each problem, is")
print("the correct answer the MODAL one in a large pool?")
print()
modal = np.array([majority(pool[i]) for i in range(N_PROB)])
modal_right = float(np.mean(modal == answers))
present = float(np.mean([(pool[i] == answers[i]).any() for i in range(N_PROB)]))
share = np.array([float(np.mean(pool[i] == answers[i])) for i in range(N_PROB)])
modal_share = np.array([float(np.mean(pool[i] == modal[i])) for i in range(N_PROB)])
lost = (modal != answers) & (share > 0)
print(f"{'quantity':>46}{'value':>10}")
print("-" * 56)
print(f"{'a correct sample exists somewhere in the pool':>46}{present:>10.1%}")
print(f"{'the correct answer is the modal one':>46}{modal_right:>10.1%}")
print(f"{'correct answer present but OUTVOTED':>46}{float(np.mean(lost)):>10.1%}")
print()
print(f"{'on those outvoted problems:':>46}")
print(f"{'mean share of samples that are correct':>46}"
      f"{float(share[lost].mean()):>10.1%}")
print(f"{'mean share held by the winning wrong answer':>46}"
      f"{float(modal_share[lost].mean()):>10.1%}")

c1, c256 = cov[1], cov[256]
m1, m256 = maj[1], maj[256]
print(f"""
The first table is the decomposition, and it separates two things that a single
accuracy number fuses.

Coverage is what the GENERATOR can do: the fraction of problems where at least
one sample is right. For the systematic generator it goes from {c1:.1%} at one
sample to {c256:.1%} at 256. That is the effect cite:brown2024monkeys reports,
and the log-linear fit below has residuals under five points across two orders of
magnitude.

The majority vote is one particular SELECTOR, and it goes from {m1:.1%} to
{m256:.1%}. It does not climb slowly. It does not climb.

The control generator is why that is a finding rather than a rigged demo. It has
the same single-sample accuracy, {base_acc:.1%}, on the same task with the same
chain length; the search that set its per-step rate to {P_FLAT:.1%} had matching
that number as its only objective. The one difference is that its errors go
somewhere at random instead of piling onto one wrong answer. Its majority vote
goes from {majf[1]:.1%} to {majf[256]:.1%}.

Identical accuracy, identical task, and the vote is worth
{majf[256] - majf[1]:+.1%} in one case and {m256 - m1:+.1%} in the other.
**Majority voting is not a method for turning samples into accuracy. It is a
method for cancelling UNSYSTEMATIC error, and it has no purchase on the
systematic kind.**

There is a second difference in that table which was not part of the plan, and it
is worth more than the one that was. The two coverage columns are not the same
either. The random-error generator reaches {covf[64]:.1%} coverage by n=64 and
saturates; the systematic one is still at {cov[64]:.1%} and reaches only
{c256:.1%} at 256.

So systematic error does not merely defeat the selector. It caps what sampling
can reach at all, because on a problem the generator is reliably confused about,
every sample fails the same way and there is nothing in the pool to select. A
perfect verifier cannot fix that, and neither can a larger budget. That is a
ceiling on the whole test-time-compute strategy, and it is set by the shape of
the model's errors rather than by anything you can buy.

The modal-answer table says exactly where the vote's failure comes from.

A correct sample is present for {present:.1%} of problems, and the correct answer
is modal for {modal_right:.1%}. The gap -- {float(np.mean(lost)):.1%} of problems
-- is answers that are in the pool and outvoted. On those problems the correct
answer holds {float(share[lost].mean()):.1%} of the samples and the winning wrong
answer holds {float(modal_share[lost].mean()):.1%}. It is not close, and it does
not get closer.

That is the ceiling stated exactly. The vote estimates which answer the generator
produces MOST OFTEN, and it estimates it consistently: more samples make it more
confident. Where the mode is wrong, more samples make it more confidently wrong,
and its limit as n goes to infinity is precisely the fraction of problems whose
mode happens to be correct -- {modal_right:.1%} here.

The verifier table is the same statement from the other side, and it is the
constructive half.

A perfect verifier (q=1.0) turns coverage into accuracy exactly: {ver[1.0][256]:.1%}
at n=256, equal to coverage, because all a perfect verifier needs is for a correct
sample to EXIST. At q=0.8 the same pool yields {ver[0.8][256]:.1%} and at q=0.5,
{ver[0.5][256]:.1%}.

So the gap between coverage and delivered accuracy IS the selector's quality
(eq:coverage-selection-gap), which makes it a budget question with a clear answer.
Doubling the sample budget moves coverage {slope:+.1%}. Moving the verifier from
q=0.5 to q=1.0 moves accuracy {ver[1.0][256] - ver[0.5][256]:+.1%} at n=256 --
and only {ver[1.0][8] - ver[0.5][8]:+.1%} at n=8.

Note the direction of that last comparison, because it is the practical
consequence and it runs against the usual intuition that a better verifier is
worth most when you are sampling least. A verifier can only cash in coverage that
exists. At n=8 there is little to recover; at n=256 there is a great deal.
Sampling and verification are complements, and scaling one without the other buys
half a mechanism -- which is ch:rsn-supervision's argument for why the interesting
work is in the verifier.""")

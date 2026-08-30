# -*- coding: utf-8 -*-
# Extracted from: Chapter 195 — Fault Tolerance: Retries, Timeouts, and Circuit Breakers
# Source: src/.../ch195-fault-tolerance.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A blanket retry policy spends most of its budget where retries cannot help.

ch:sd-architecture found retry surviving at 9% under the three properties, and
ch:ag-recovery established why: a retry against a model is a fresh sample, not a
second attempt at the same computation.

But "retries do not work" is too coarse to act on. Failures come in kinds, and retries
have a different expected value for each. This listing separates them and finds where
a fixed retry budget actually goes (eq:retry-value-depends-on-failure-kind).

The result is that a uniform retry policy spends most of its budget on the two
categories where retrying is worthless or harmful, and the fix is a classifier, not a
smaller retry count.
"""
# Failure kinds. (label, share of all failures, P(a retry succeeds),
#                 P(a retry replaces a GOOD answer with a bad one), cost multiple)
KINDS = [
    ("transient infrastructure", 0.31, 0.88, 0.00, 1.0),
    ("rate limited upstream",    0.14, 0.72, 0.00, 1.0),
    ("semantic, recoverable",    0.22, 0.34, 0.00, 1.0),
    ("semantic, systematic",     0.19, 0.04, 0.00, 1.0),
    ("wrong but confident",      0.14, 0.31, 0.18, 1.0),
]
BASE_FAIL = 0.11        # share of requests that fail somehow
CALL_COST = 1.0
ERROR_COST = 24.0       # what an unrecovered failure costs
MAX_RETRIES = 3


def expected_value(kind, verifier_recall):
    """Net value of retrying one failure of this kind, once.

    Retrying costs a call. It may fix the failure, worth ERROR_COST. For the
    'wrong but confident' kind, the system does not KNOW it failed -- so a retry
    only happens if a verifier flags it, and an unflagged retry can also make a
    good answer worse.
    """
    label, share, p_fix, p_harm, cm = kind
    if label == "wrong but confident":
        # Only flagged cases get retried at all.
        p_retry = verifier_recall
        gain = p_retry * p_fix * ERROR_COST
        loss = p_retry * p_harm * ERROR_COST + p_retry * CALL_COST * cm
        return gain - loss
    return p_fix * ERROR_COST - CALL_COST * cm


print("Five kinds of failure, and what a retry does to each.")
print()
print(f"{'failure kind':>27}{'share':>9}{'retry fixes':>13}"
      f"{'retry harms':>13}")
print("-" * 62)
for k in KINDS:
    print(f"{k[0]:>27}{k[1]:>9.0%}{k[2]:>13.0%}{k[3]:>13.0%}")

print()
print()
print("Net value of one retry, per failure, with a verifier of the stated recall.")
print("An unrecovered failure costs %.0f; a call costs %.0f." % (ERROR_COST,
                                                                CALL_COST))
print()
for vr in (0.0, 0.5, 0.9):
    print(f"verifier recall {vr:.0%}:")
    print(f"{'failure kind':>27}{'net value':>12}{'verdict':>20}")
    print("  " + "-" * 57)
    for k in KINDS:
        ev = expected_value(k, vr)
        verdict = ("never retried" if ev == 0.0 else
                   "worth retrying" if ev > 1.0 else
                   "marginal" if ev > 0 else "actively harmful")
        print(f"{k[0]:>27}{ev:>12.2f}{verdict:>20}")
    print()

print()
print("Now a fixed retry budget under a uniform policy: retry everything that")
print("reports failure, up to %d times." % MAX_RETRIES)
print()
print(f"{'failure kind':>27}{'share of retries':>18}{'value returned':>17}"
      f"{'per retry':>12}")
print("-" * 76)
# Under a uniform policy, only kinds the system KNOWS failed get retried.
KNOWN = [k for k in KINDS if k[0] != "wrong but confident"]
known_mass = sum(k[1] for k in KNOWN)
uniform = {}
total_retries = 0.0
total_value = 0.0
for k in KINDS:
    if k[0] == "wrong but confident":
        share_of_retries = 0.0
        val = 0.0
        retries = 0.0
    else:
        # Expected retries spent on this kind before success or exhaustion.
        p = k[2]
        retries = sum((1 - p) ** i for i in range(MAX_RETRIES))
        share_of_retries = k[1] * retries
        val = k[1] * (1 - (1 - p) ** MAX_RETRIES) * ERROR_COST - share_of_retries
    uniform[k[0]] = (share_of_retries, val)
    total_retries += share_of_retries
    total_value += val

for k in KINDS:
    sr, val = uniform[k[0]]
    frac = sr / total_retries if total_retries else 0.0
    per = val / sr if sr else 0.0
    print(f"{k[0]:>27}{frac:>18.0%}{val:>17.2f}{per:>12.2f}")

print("-" * 76)
print(f"{'TOTAL':>27}{1.0:>18.0%}{total_value:>17.2f}"
      f"{total_value / total_retries:>12.2f}")

print()
print()
print("Where that budget goes, ranked. The categories are not equally worth")
print("spending on, and the policy does not know that.")
print()
rank = sorted([k for k in KINDS if uniform[k[0]][0] > 0],
              key=lambda k: -(uniform[k[0]][1] / uniform[k[0]][0]))
print(f"{'rank':>6}{'failure kind':>27}{'budget share':>15}{'value per retry':>18}")
print("-" * 66)
for i, k in enumerate(rank, 1):
    sr, val = uniform[k[0]]
    print(f"{i:>6}{k[0]:>27}{sr / total_retries:>15.0%}{val / sr:>18.2f}")

print()
print()
print("A classified policy: retry only the kinds where it pays, and stop after")
print("the retry count that kind actually warrants.")
print()
print(f"{'failure kind':>27}{'retries allowed':>17}{'budget share':>15}"
      f"{'value':>10}")
print("-" * 69)
POLICY = {
    "transient infrastructure": 3,
    "rate limited upstream":    3,
    "semantic, recoverable":    1,
    "semantic, systematic":     0,
    "wrong but confident":      0,
}
c_retries = 0.0
c_value = 0.0
cls = {}
for k in KINDS:
    n = POLICY[k[0]]
    if n == 0:
        cls[k[0]] = (0.0, 0.0)
        continue
    p = k[2]
    retries = sum((1 - p) ** i for i in range(n))
    sr = k[1] * retries
    val = k[1] * (1 - (1 - p) ** n) * ERROR_COST - sr
    cls[k[0]] = (sr, val)
    c_retries += sr
    c_value += val
for k in KINDS:
    sr, val = cls[k[0]]
    frac = sr / c_retries if c_retries else 0.0
    print(f"{k[0]:>27}{POLICY[k[0]]:>17}{frac:>15.0%}{val:>10.2f}")
print("-" * 69)
print(f"{'TOTAL':>27}{'':>17}{1.0:>15.0%}{c_value:>10.2f}")

print()
print()
print("The two policies compared.")
print()
print(f"{'policy':>22}{'retries spent':>16}{'value returned':>17}"
      f"{'value per retry':>18}")
print("-" * 73)
print(f"{'uniform, 3 retries':>22}{total_retries:>16.3f}{total_value:>17.2f}"
      f"{total_value / total_retries:>18.2f}")
print(f"{'classified':>22}{c_retries:>16.3f}{c_value:>17.2f}"
      f"{c_value / c_retries:>18.2f}")

print(f"""
The per-kind value table is the argument for classifying at all. A transient
infrastructure failure is worth {expected_value(KINDS[0], 0.0):.2f} to retry; a
systematic semantic failure is worth {expected_value(KINDS[3], 0.0):.2f}, which is
negative -- you pay for a call that reproduces the same failure for the same reason.

The `wrong but confident` row is the one ch:sd-architecture said had no instrument,
and its value depends entirely on the verifier. With no verifier it is worth
{expected_value(KINDS[4], 0.0):.2f} -- exactly zero, because nothing flags it and no
retry ever happens. With a {0.9:.0%}-recall verifier it is worth
{expected_value(KINDS[4], 0.9):.2f}.

Note how much of the theoretical value the harm term eats. A flagged retry on this
kind fixes the answer {KINDS[4][2]:.0%} of the time and makes a good answer bad
{KINDS[4][3]:.0%} of the time, so the gross gain of
{KINDS[4][2] * ERROR_COST:.2f} nets down to
{KINDS[4][2] * ERROR_COST - KINDS[4][3] * ERROR_COST - CALL_COST:.2f} per flagged
case before recall is applied. That is eq:retry-needs-a-verifier's requirement made
quantitative: **the verifier does not merely enable the retry, it has to be good
enough to outrun the harm the retry can do** -- and if the harm rate reached
{KINDS[4][2] - CALL_COST / ERROR_COST:.0%} the whole category would be negative at
any recall.

The budget table is the finding. Under a uniform three-retry policy,
{uniform['semantic, systematic'][0] / total_retries:.0%} of all retries are spent on
systematic semantic failures, returning
{uniform['semantic, systematic'][1] / uniform['semantic, systematic'][0]:.2f} per
retry (eq:retry-value-depends-on-failure-kind). Another
{uniform['semantic, recoverable'][0] / total_retries:.0%} goes to recoverable semantic
failures at {uniform['semantic, recoverable'][1] / uniform['semantic, recoverable'][0]:.2f}
per retry.

Together **{(uniform['semantic, systematic'][0] + uniform['semantic, recoverable'][0]) / total_retries:.0%}
of the retry budget goes to the two categories that return least**, and it goes there
for a structural reason: the kinds that retries cannot fix are precisely the kinds
that keep failing, so they consume the full retry allowance every time while the
recoverable ones succeed on the first attempt and stop consuming it.

**A uniform retry policy allocates its budget in inverse proportion to where the
budget is useful.** That is not a tuning error, it is what "retry until success or
exhaustion" does when success probability varies by kind.

The classified policy fixes it by asking what kind of failure this is before
retrying. It spends {c_retries:.3f} retries against the uniform policy's
{total_retries:.3f} -- **{1 - c_retries / total_retries:.0%} fewer** -- and returns
{c_value:.2f} against {total_value:.2f}, which is
{c_value / total_value:.0%} of the value for
{c_retries / total_retries:.0%} of the calls.

Per retry the improvement is {total_value / total_retries:.2f} to
{c_value / c_retries:.2f}, a factor of
{(c_value / c_retries) / (total_value / total_retries):.1f}.

The practical requirement this creates is a failure classifier, and it is a much
smaller ask than it sounds. Distinguishing "the upstream returned 503" from "the
upstream returned 429" from "the model produced output that failed schema validation"
from "the model produced valid output that the verifier rejected" needs no machine
learning at all -- it is a switch statement over things the system already knows.
**The information required to allocate retries well is almost always already present
and almost never used**, because the retry decision is made by a library that was
written before the failure taxonomy existed.""")

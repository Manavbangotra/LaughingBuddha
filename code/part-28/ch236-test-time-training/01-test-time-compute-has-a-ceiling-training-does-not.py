# -*- coding: utf-8 -*-
# Extracted from: Chapter 236 — Test-Time Training and Test-Time Compute
# Source: src/.../ch236-test-time-training.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Two ways to buy accuracy, and only one of them has no ceiling.

ch:rsn-test-time-compute established the mechanism: sample more, select better, and accuracy
rises with the log of the budget until the verifier runs out (cite:brown2024monkeys,
cite:snell2024testtime). This listing asks the budgeting question that follows -- given a
quality target, is it cheaper to train a better model or to think longer?

They are not interchangeable. Training compute moves a scaling curve with no ceiling in the
range of interest; test-time compute moves along a coverage-selection curve that saturates at
the verifier's quality (eq:test-time-compute-has-a-ceiling-training-does-not).

Above that ceiling no amount of thinking substitutes for a better model. Below it the two are
priced in different units -- one paid once, one paid per request -- and the crossing point is
the decision (eq:the-axes-cross-at-a-quality-target).
"""
import math

A_CEIL, A_SCALE, A_EXP = 0.980, 270.6, 0.130
VERIFIER = 0.86
ACTIVE, TOKENS_OUT = 2.0e10, 700
FLOP_COST = 3.20 / 3600 / 9.9e14
BASE_FLOPS = 3.0e23


def acc_train(flops):
    return A_CEIL - A_SCALE * flops ** -A_EXP


def acc_test(base, samples, verifier=VERIFIER):
    """Coverage rises with log samples; selection caps the gain at the verifier."""
    coverage = 1.0 - (1.0 - base) ** (1.0 + 0.28 * math.log(max(samples, 1), 2))
    return base + (coverage - base) * verifier


def samples_for(base, target, verifier=VERIFIER, cap=100_000):
    s = 1
    while s < cap and acc_test(base, s, verifier) < target:
        s += 1
    return s if acc_test(base, s, verifier) >= target else None


BASE_ACC = acc_train(BASE_FLOPS)
CEIL_TEST = acc_test(BASE_ACC, 10 ** 9)

print("What each axis buys on its own.")
print()
print(f"a model trained with {BASE_FLOPS:.1e} FLOPs scores {BASE_ACC:.4f}")
print()
print(f"{'accuracy target':>18}{'training FLOPs needed':>24}{'multiple of base':>19}"
      f"{'samples needed':>17}{'test FLOPs / request':>23}")
print("-" * 101)
need = {}
TARGETS = (0.78, 0.85, 0.90, 0.93, 0.95, 0.97)
for target in TARGETS:
    tf = (max(A_CEIL - target, 1e-9) / A_SCALE) ** (-1 / A_EXP)
    s = samples_for(BASE_ACC, target)
    need[target] = (tf, s)
    ss = f"{s:>17,}" if s else f"{'unreachable':>17}"
    tt = f"{2 * ACTIVE * TOKENS_OUT * s:>23.2e}" if s else f"{'--':>23}"
    print(f"{target:>18.2f}{tf:>24.2e}{tf / BASE_FLOPS:>18.1f}x{ss}{tt}")

print()
print(f"test-time compute saturates at {CEIL_TEST:.4f}, set by the verifier at {VERIFIER:.2f}")
print(f"training has no ceiling below {A_CEIL:.3f}")

print()
print()
print("Diminishing returns on the test-time axis.")
print()
print(f"{'samples':>10}{'accuracy':>12}{'gain over 1':>14}{'gain per doubling':>21}"
      f"{'share of the ceiling':>23}")
print("-" * 80)
prev = None
for s in (1, 2, 4, 16, 64, 256, 1024, 4096):
    a = acc_test(BASE_ACC, s)
    g = f"{a - prev:>20.4f}" if prev is not None else f"{'--':>21}"
    print(f"{s:>10,}{a:>12.4f}{a - BASE_ACC:>14.4f}{g}"
          f"{(a - BASE_ACC) / (CEIL_TEST - BASE_ACC):>23.1%}")
    prev = a

SHARE16 = (acc_test(BASE_ACC, 16) - BASE_ACC) / (CEIL_TEST - BASE_ACC)
print()
print(f"the first 16 samples deliver {SHARE16:.0%} of everything test-time")
print("compute can ever deliver on this model")

print()
print()
print("The two axes are priced in different units.")
print()
print(f"{'accuracy target':>18}{'train-more $M':>21}{'samples':>10}"
      f"{'think-longer $ / request':>27}{'crossover requests':>21}{'reachable?':>15}")
print("-" * 112)
cross = {}
for target in TARGETS:
    tf, s = need[target]
    train_cost = (tf - BASE_FLOPS) * FLOP_COST
    if s is None:
        print(f"{target:>18.2f}{train_cost / 1e6:>21,.1f}{'--':>10}{'--':>27}"
              f"{'--':>21}{'training only':>15}")
        continue
    per_req = 2 * ACTIVE * TOKENS_OUT * s * FLOP_COST
    cross[target] = (train_cost, s, per_req, train_cost / per_req)
    print(f"{target:>18.2f}{train_cost / 1e6:>21,.1f}{s:>10,}{per_req:>27.6f}"
          f"{train_cost / per_req:>21.2e}{'yes':>15}")

print()
print(f"at {0.90:.2f}: training costs {cross[0.90][0] / 1e6:,.1f}M, thinking costs")
print(f"{cross[0.90][2]:.6f} per request, and they cross at"
      f" {cross[0.90][3]:.1e} requests")

print()
print()
print("Splitting a fixed budget between the two.")
print()
TOTAL, REQS = 4.0e24, 3.0e8
print(f"total budget {TOTAL:.1e} FLOPs, {REQS:.0e} requests")
print()
print(f"{'share spent on training':>26}{'train FLOPs':>15}{'base accuracy':>16}"
      f"{'samples affordable':>21}{'final accuracy':>17}")
print("-" * 95)
best = None
for share in (0.20, 0.50, 0.80, 0.90, 0.95, 0.99, 0.999):
    tf = TOTAL * share
    s = max(1, int(TOTAL * (1 - share) / (2 * ACTIVE * TOKENS_OUT * REQS)))
    b = acc_train(tf)
    a = acc_test(b, s)
    if best is None or a > best[1]:
        best = (share, a, s, b)
    print(f"{share:>26.3f}{tf:>15.2e}{b:>16.4f}{s:>21,}{a:>17.4f}")

ALL_TRAIN = acc_train(TOTAL)
print()
print(f"best split: {best[0]:.1%} on training, {best[2]:,} samples per request,"
      f" reaching {best[1]:.4f}")
print(f"spending it all on training gives {ALL_TRAIN:.4f}")

print()
print()
print("What moves the ceiling.")
print()
print(f"{'verifier quality':>19}{'ceiling accuracy':>19}{'samples to 90% of it':>24}"
      f"{'headroom over base':>21}")
print("-" * 83)
ceilings = {}
for v in (0.55, 0.70, 0.86, 0.95, 1.00):
    ceil = acc_test(BASE_ACC, 10 ** 9, v)
    ceilings[v] = ceil
    s = samples_for(BASE_ACC, BASE_ACC + 0.90 * (ceil - BASE_ACC), v)
    print(f"{v:>19.2f}{ceil:>19.4f}{s:>24,}{ceil - BASE_ACC:>21.4f}")

print(f"""
The first table is the budgeting question nobody sets up explicitly. A model trained with
{BASE_FLOPS:.1e} FLOPs scores {BASE_ACC:.4f}. To reach {0.90:.2f} you can train
{need[0.90][0] / BASE_FLOPS:,.0f} times longer, or sample {need[0.90][1]:,} times per request.
Both work.

To reach {0.97:.2f} only one of them works. Test-time compute saturates at **{CEIL_TEST:.4f}**,
set by the verifier's quality at {VERIFIER:.2f}, and no number of samples crosses it
(eq:test-time-compute-has-a-ceiling-training-does-not). Training has no ceiling below
{A_CEIL:.3f}.

That asymmetry is the most important thing in this listing. Across most of the useful range the
two axes look like substitutes, and they are not the same kind of object: **one moves a curve,
the other moves along one.**

The diminishing-returns table says where the useful range ends. The first 16 samples deliver
**{SHARE16:.0%}** of everything test-time compute can ever deliver on this model, and 1,024
deliver {(acc_test(BASE_ACC, 1024) - BASE_ACC) / (CEIL_TEST - BASE_ACC):.0%}. That is
ch:rsn-test-time-compute's `coverage-log-linear` seen against its own asymptote: the interesting
operating points are all small, and a budget of thousands of samples is buying the last two
percent of a bounded quantity.

The cost table is where the decision gets made, and the numbers are more one-sided than the
framing suggests. Reaching {0.90:.2f} by training costs **${cross[0.90][0] / 1e6:,.0f} million**; reaching it by
sampling costs **{cross[0.90][2]:.6f} per request**. Those cross at
**{cross[0.90][3]:.1e} requests** -- a volume no product reaches
(eq:the-axes-cross-at-a-quality-target).

So for every target test-time compute can reach, thinking longer is cheaper than training
longer, by orders of magnitude, at any realistic volume. **The reason to train a better model is
not that it is cheaper. It is that above {CEIL_TEST:.4f} there is nothing else on the menu.**

That is a genuinely different conclusion from ch:res-scaling's and ch:res-moe's, which both
turned on serving volume. Here volume barely matters and the *ceiling* decides -- which is why
the last table is the one to act on.

The joint-budget table confirms the shape at a planner's level of abstraction. Splitting
{TOTAL:.1e} FLOPs across {REQS:.0e} requests, the best split spends **{best[0]:.1%} on training**
and affords {best[2]:,} samples per request, reaching {best[1]:.4f} -- against {ALL_TRAIN:.4f}
for spending it all on training. The optimum is interior and it is skewed heavily toward
training, because the test-time term is multiplied by the request count and the training term is
not.

The last table is the lever everything rests on. Raising the verifier from {0.55:.2f} to
{0.95:.2f} moves the ceiling from {ceilings[0.55]:.4f} to {ceilings[0.95]:.4f} -- a gain of
{ceilings[0.95] - ceilings[0.55]:.4f}, larger than anything the sampling budget can buy at a
fixed verifier.

**The verifier, not the sampler, sets what test-time compute is worth.** That is
ch:rsn-test-time-compute's `verifier-quality-ceiling` restated as a budgeting instruction:
spending on a better verifier raises a ceiling, and spending on more samples only approaches
one.""")

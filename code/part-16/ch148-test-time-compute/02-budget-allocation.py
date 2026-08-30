# -*- coding: utf-8 -*-
# Extracted from: Chapter 148 — Test-Time Compute and Search
# Source: src/.../ch148-test-time-compute.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A fixed sampling budget, spent seven ways.

cite:snell2024testtime's compute-optimal result says the best way to spend
test-time compute depends on the prompt's difficulty, and that allocating by
difficulty beats a uniform budget substantially. This listing works out what that
means when you cannot see the difficulty and have to estimate it
(eq:marginal-value-of-a-sample), and then what happens when you stop trying to
predict difficulty and simply react to outcomes.

The setup is deliberately favourable to allocation: a population of problems with
widely varying per-sample success rates, so there is a great deal to gain from
spending unevenly.
"""
import numpy as np

rng = np.random.default_rng(523)

M = 4000                  # problems
BUDGET_PER = 16           # mean samples per problem
TOTAL = M * BUDGET_PER

# Per-sample success rate. A realistic spread: many easy, a solid middle, and a
# tail that is effectively hopeless at any budget you can afford.
p = np.concatenate([
    rng.beta(9.0, 1.5, size=M // 3),        # easy
    rng.beta(1.6, 4.0, size=M // 3),        # middle
    rng.beta(0.35, 14.0, size=M - 2 * (M // 3)),   # nearly hopeless
])
rng.shuffle(p)


def solved(alloc, ptrue=p):
    """Expected fraction solved: a problem counts if at least one of its
    allocated samples succeeds."""
    return float(np.mean(1.0 - (1.0 - ptrue) ** alloc))


def marginal(n, ptrue):
    """Increase in P(solved) from one more sample: (1-p)^n * p."""
    return (1.0 - ptrue) ** n * ptrue


def allocate(ptrue, total=TOTAL, cap=4096):
    """Optimal allocation for a concave objective, by water-filling.

    The marginal value of the (n+1)th sample is (1-p)^n * p, which is decreasing
    in n, so the optimum equalises marginal value across problems. Solve
    (1-p)^n * p = lam for n and binary-search lam to hit the budget."""
    ptrue = np.clip(ptrue, 1e-9, 1 - 1e-9)
    lo, hi = 1e-12, float(ptrue.max())

    def alloc_for(lam):
        n = np.log(lam / ptrue) / np.log1p(-ptrue)
        return np.clip(np.round(n), 0, cap).astype(np.int64)

    for _ in range(80):
        mid = (lo + hi) / 2
        if alloc_for(mid).sum() > total:
            lo = mid          # lam too low -> allocating too much
        else:
            hi = mid
    a = alloc_for(hi)
    # Spend any rounding slack on the highest remaining marginal values.
    slack = total - int(a.sum())
    if slack > 0:
        idx = np.argsort(-marginal(a, ptrue))[:slack]
        a[idx] += 1
    return a


def pilot_estimate(ptrue, pilot):
    """What you actually get to see: successes out of `pilot` real samples.
    Laplace-smoothed so a zero-success pilot is not treated as p = 0."""
    s = rng.binomial(pilot, ptrue)
    return (s + 1.0) / (pilot + 2.0)


print(f"{M} problems, a total budget of {TOTAL:,} samples "
      f"({BUDGET_PER} per problem on average).")
print("Per-sample success rates span easy, middle and near-hopeless.")
print()
print(f"{'difficulty band':>22}{'count':>8}{'mean p':>9}"
      f"{'solved at n=16':>16}")
print("-" * 55)
bands = [("p > 0.5 (easy)", p > 0.5),
         ("0.05 < p < 0.5", (p > 0.05) & (p <= 0.5)),
         ("p < 0.05 (hard)", p <= 0.05)]
for name, m in bands:
    print(f"{name:>22}{int(m.sum()):>8}{float(p[m].mean()):>9.3f}"
          f"{solved(np.full(int(m.sum()), BUDGET_PER), p[m]):>16.1%}")

print()
print()
print("Seven ways to spend the same total budget.")
print()

uniform = np.full(M, BUDGET_PER, dtype=np.int64)
oracle = allocate(p)

pilots = {}
for pilot in (2, 4, 8):
    est = pilot_estimate(p, pilot)
    alloc = allocate(est, total=TOTAL - pilot * M)
    # The pilot samples are real attempts and count toward solving.
    pilots[pilot] = alloc + pilot

# Two tempting heuristics, both spending the same total.
order = np.argsort(p)                      # ascending difficulty: hardest first
hardest = np.full(M, BUDGET_PER // 2, dtype=np.int64)
extra = TOTAL - int(hardest.sum())
hardest[order[:M // 4]] += extra // (M // 4)

easiest = np.full(M, BUDGET_PER // 2, dtype=np.int64)
extra = TOTAL - int(easiest.sum())
easiest[order[-(M // 4):]] += extra // (M // 4)


def with_early_stopping(ptrue, total=TOTAL, cap=4096):
    """Sample round-robin over the problems that are still unsolved, and stop
    spending on a problem the moment one of its samples succeeds.

    This needs a VERIFIER -- you cannot stop early unless you can tell that you
    are done -- but it needs no difficulty estimate at all."""
    n = len(ptrue)
    alive = np.ones(n, dtype=bool)
    used = np.zeros(n, dtype=np.int64)
    spent = 0
    while spent < total and alive.any():
        k = int(alive.sum())
        if spent + k > total:
            idx = np.flatnonzero(alive)[: total - spent]
            hit = rng.random(len(idx)) < ptrue[idx]
            used[idx] += 1
            alive[idx[hit]] = False
            break
        idx = np.flatnonzero(alive)
        hit = rng.random(k) < ptrue[idx]
        used[idx] += 1
        alive[idx[hit]] = False
        spent += k
        if used.max() >= cap:
            break
    return float(np.mean(~alive)), used


stop_solved, stop_used = with_early_stopping(p)

print(f"{'strategy':>34}{'solved':>10}{'vs uniform':>13}")
print("-" * 57)
strategies = [
    ("uniform (16 each)", uniform),
    ("oracle allocation (knows p)", oracle),
    ("pilot of 2, then allocate", pilots[2]),
    ("pilot of 4, then allocate", pilots[4]),
    ("pilot of 8, then allocate", pilots[8]),
    ("all spare budget to hardest 25%", hardest),
    ("all spare budget to easiest 25%", easiest),
]
base = solved(uniform)
res = {}
for name, alloc in strategies:
    v = solved(alloc)
    res[name] = v
    print(f"{name:>34}{v:>10.1%}{v - base:>+13.1%}")
res["early stopping (needs a verifier)"] = stop_solved
print(f"{'early stopping (needs a verifier)':>34}{stop_solved:>10.1%}"
      f"{stop_solved - base:>+13.1%}")

print()
print()
print("Where does the budget go? Mean samples per problem by band.")
print()
print(f"{'difficulty band':>22}{'uniform':>10}{'oracle':>10}"
      f"{'pilot of 4':>13}{'early stop':>13}")
print("-" * 68)
for name, m in bands:
    print(f"{name:>22}{BUDGET_PER:>10.1f}{float(oracle[m].mean()):>10.1f}"
          f"{float(pilots[4][m].mean()):>13.1f}"
          f"{float(stop_used[m].mean()):>13.1f}")

o = res["oracle allocation (knows p)"]
p4 = res["pilot of 4, then allocate"]
hd = res["all spare budget to hardest 25%"]
ez = res["all spare budget to easiest 25%"]
easy_m, mid_m, hard_m = bands[0][1], bands[1][1], bands[2][1]
print(f"""
The oracle row is the best a FIXED allocation can do: {o:.1%} against uniform's
{base:.1%}, a gain of {o - base:+.1%} from spending the same {TOTAL:,} samples
differently, with perfect knowledge of every problem's success rate. That is
cite:snell2024testtime's compute-optimal effect in its simplest possible form,
and it is an upper bound only for policies that must commit before seeing any
outcomes.

The allocation table says where the gain comes from, and it is not where the
phrase "spend more on hard problems" points.

The oracle does give the hard band the most -- {float(oracle[hard_m].mean()):.1f}
samples against the middle band's {float(oracle[mid_m].mean()):.1f}. But look at
the easy band: {float(oracle[easy_m].mean()):.1f} samples, down from 16. Those
problems succeed at {float(p[easy_m].mean()):.0%} per sample and are essentially
all solved by the third attempt; the other thirteen samples were buying nothing.
**The gain is almost entirely in not over-sampling the easy problems**, and the
hard band is merely where the freed budget lands.

That distinction is what separates the optimal policy from the heuristic that
sounds like it. Giving all the spare budget to the hardest quarter scores
{hd:.1%}, which is {hd - base:+.1%} against uniform -- it LOSES, because it funds
the hard problems out of the middle band as well as the easy one, and because a
quarter of the problems chosen for being hardest includes the ones that are
hopeless at any budget. Giving it to the easiest quarter is worse still at
{ez:.1%} ({ez - base:+.1%}), which at least fails in the direction you would
expect.

eq:marginal-value-of-a-sample explains both. The value of one more sample is
(1-p)^n * p, which vanishes as p approaches 1 (already solved) and as p
approaches 0 (the sample will not land either). At n={BUDGET_PER} it is maximised
near p = 1/(n+1) = {1/(BUDGET_PER+1):.3f}. Neither end of the difficulty range is
where the money is, and "hard" and "worth sampling" are different properties that
happen to overlap in the middle.

The pilot rows are the practical question, because difficulty is not observable
and estimating it costs samples out of the same budget.

A pilot of 4 gets {p4 - base:+.1%}, which is {(p4 - base) / (o - base):.0%} of the
oracle's {o - base:+.1%}. That is real and it is also a minority of what is
available. Pilots of 2 and 8 give {res['pilot of 2, then allocate'] - base:+.1%}
and {res['pilot of 8, then allocate'] - base:+.1%}, so measuring harder makes
things worse past a point: the pilot spends budget to learn something it then has
less budget to act on. The optimum is interior and shallow, and a third of the
oracle gain is roughly what this approach is worth.

Then there is the last row, which is the one to actually build.

Early stopping scores {stop_solved:.1%}, {stop_solved - base:+.1%} against
uniform. It beats every pilot strategy, and it beats the oracle by
{stop_solved - o:+.1%} while estimating nothing at all.

That is not a bug in the oracle, and the reason is the most useful thing in this
listing. The oracle knows every p and commits its whole allocation up front. Early
stopping knows nothing about p and gets to see OUTCOMES, so it can stop a problem
after one lucky sample and keep feeding one that has failed forty times. An
adaptive policy is allowed to beat the best non-adaptive one, because it is
optimising against information the non-adaptive policy is not permitted to use --
and here it does, by {stop_solved - o:.1%}.

Look at what it does to the bands without being told anything. The easy band
drops to {float(stop_used[easy_m].mean()):.1f} samples on average, the middle to
{float(stop_used[mid_m].mean()):.1f}, and the hard band rises to
{float(stop_used[hard_m].mean()):.1f} -- a more aggressive version of the oracle's
own allocation, arrived at with no model of difficulty whatsoever. A problem that
succeeds immediately consumes one sample whatever its nominal difficulty; a
problem that keeps failing keeps drawing budget until it succeeds or the budget
runs out. The information the pilot spent {4 * M / TOTAL:.0%} of the budget to
buy arrives free as a by-product of doing the work.

The catch is in the parenthesis. Early stopping requires knowing that a sample is
correct, which is a VERIFIER -- the same component whose quality set the ceiling
in the previous listing. So the two halves of this chapter reduce to one
recommendation: the verifier is what converts a sampling budget into answers, and
it is also what tells you when to stop spending. Without it you are choosing
between a uniform budget and a difficulty estimate that costs a third of what it
recovers.""")

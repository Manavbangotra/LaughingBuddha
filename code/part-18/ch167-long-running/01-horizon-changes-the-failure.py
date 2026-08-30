# -*- coding: utf-8 -*-
# Extracted from: Chapter 167 — Long-Running and Autonomous Workflows
# Source: src/.../ch167-long-running.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""What actually fails as the horizon gets longer.

Short-horizon agents fail because a step fails: ch:ag-loop's per-step reliability
compounds and the run dies. That framing does not survive a long horizon, for a
reason worth stating first.

At 98.5% per step, a 300-step run completes 1% of the time. A long-running system
is therefore only possible at all because ch:ag-recovery exists: a failed step is
RETRIED rather than fatal. Recovery does not make step failure go away -- it
converts it into budget consumption (eq:recovery-converts-failure-to-cost).

So the three things that can end a long run are:

  exhaustion   retries and work exceed ch:ag-termination's budget
  drift        an assumption made early stopped being true, and the run continued
               confidently on a stale premise -- every step 'succeeded'
  step failure a step fails in a way retry cannot fix

They scale differently in the horizon, so the dominant one changes
(eq:horizon-changes-the-failure).
"""
import numpy as np

rng = np.random.default_rng(3571)

M = 40000
P_STEP = 0.97           # per-step success; a failure is retried, not fatal
P_HARD = 0.0008         # per-step chance the failure is unrecoverable
P_INVALIDATE = 0.0025   # per-step chance a given standing assumption goes stale
N_ASSUME = 6
BUDGET_MULT = 1.6       # budget as a multiple of the nominal horizon


def run(horizon, m=M, recheck=0, p_step=P_STEP, p_inv=P_INVALIDATE,
        budget_mult=BUDGET_MULT, repair=0.9, recheck_cost=1):
    """Walk `horizon` steps under a budget. A failed step is retried and costs
    budget; a hard failure ends the run. Independently, standing assumptions go
    stale silently -- a run that finishes on a stale assumption produces a wrong
    answer with no error anywhere. A recheck re-validates and mostly repairs."""
    budget = int(horizon * budget_mult)
    alive = np.ones(m, dtype=bool)
    stale = np.zeros(m, dtype=bool)
    pos = np.zeros(m, dtype=np.int64)
    used = np.zeros(m, dtype=np.int64)
    hard = np.zeros(m, dtype=bool)
    for t in range(budget):
        live = alive & (pos < horizon) & (used < budget)
        idx = np.flatnonzero(live)
        if not len(idx):
            break
        used[idx] += 1
        # A hard failure is unrecoverable; a soft one just costs this step.
        h = rng.random(len(idx)) < P_HARD
        alive[idx[h]] = False
        hard[idx[h]] = True
        rest = idx[~h]
        good = rest[rng.random(len(rest)) < p_step]
        pos[good] += 1
        # Assumptions go stale as wall-clock passes, whether or not work advanced.
        went = rng.random(len(rest)) < (1 - (1 - p_inv) ** N_ASSUME)
        stale[rest[went]] = True
        if recheck and ((t + 1) % recheck == 0):
            fixed = rng.random(len(rest)) < repair
            stale[rest[fixed]] = False
            used[rest] += recheck_cost
    finished = alive & (pos >= horizon)
    correct = finished & ~stale
    exhausted = alive & (pos < horizon)
    return (float(correct.mean()), float((finished & stale).mean()),
            float(exhausted.mean()), float(hard.mean()), float(used.mean()))


HORIZONS = [10, 30, 100, 300, 1000]

print(f"{M:,} runs. Per-step success {P_STEP:.0%} with retry, {P_HARD:.2%} chance")
print(f"a failure is unrecoverable, {N_ASSUME} standing assumptions each going")
print(f"stale at {P_INVALIDATE:.2%} per step, budget {BUDGET_MULT:.1f}x horizon.")
print()
print(f"{'horizon':>9}{'correct':>10}{'silent drift':>14}{'exhausted':>11}"
      f"{'hard failure':>14}")
print("-" * 58)
base = {}
for h in HORIZONS:
    r = run(h)
    base[h] = r
    print(f"{h:>9}{r[0]:>10.1%}{r[1]:>14.1%}{r[2]:>11.1%}{r[3]:>14.1%}")

print()
print()
print("As a share of the FAILURES, which is the view that says what to work on.")
print()
print(f"{'horizon':>9}{'silent drift':>15}{'exhausted':>12}{'hard failure':>15}")
print("-" * 51)
share = {}
for h in HORIZONS:
    r = base[h]
    tot = r[1] + r[2] + r[3]
    row = (r[1] / tot, r[2] / tot, r[3] / tot)
    share[h] = row
    print(f"{h:>9}{row[0]:>15.1%}{row[1]:>12.1%}{row[2]:>15.1%}")

print()
print()
print("Rechecking assumptions repairs drift and costs budget. Sweeping the")
print("interval at horizon 300:")
print()
print(f"{'recheck every':>15}{'correct':>10}{'silent drift':>14}"
      f"{'exhausted':>11}{'steps used':>12}")
print("-" * 62)
rc = {}
for k in (0, 100, 50, 25, 10, 5, 2):
    r = run(300, recheck=k)
    rc[k] = r
    label = "never" if k == 0 else str(k)
    print(f"{label:>15}{r[0]:>10.1%}{r[1]:>14.1%}{r[2]:>11.1%}{r[4]:>12.0f}")

print()
print()
print("The optimum moves with the horizon: a longer run has more to go stale and")
print("less budget slack to spend on checking.")
print()
print(f"{'horizon':>9}{'never':>9}{'every 50':>11}{'every 25':>11}"
      f"{'every 10':>11}{'every 5':>10}{'best':>11}")
print("-" * 72)
opt = {}
INTERVALS = (0, 50, 25, 10, 5)
NAMES = ["never", "every 50", "every 25", "every 10", "every 5"]
for h in (30, 100, 300, 1000):
    row = [run(h, recheck=k)[0] for k in INTERVALS]
    best = NAMES[int(np.argmax(row))]
    opt[h] = (row, best)
    print(f"{h:>9}" + "".join(f"{v:>{w}.1%}" for v, w in
                              zip(row, (9, 11, 11, 11, 10))) + f"{best:>11}")

print()
print()
print("And against the rate the world changes, which is a property of the")
print("environment rather than of the agent. Horizon 300:")
print()
print(f"{'staleness rate':>16}{'never':>9}{'every 25':>11}{'every 10':>11}"
      f"{'best gain':>12}")
print("-" * 59)
sw = {}
for pi in (0.0005, 0.0025, 0.008, 0.02):
    row = [run(300, recheck=k, p_inv=pi)[0] for k in (0, 25, 10)]
    sw[pi] = row
    print(f"{pi:>16.2%}{row[0]:>9.1%}{row[1]:>11.1%}{row[2]:>11.1%}"
          f"{max(row[1:]) - row[0]:>+12.1%}")

print(f"""
The first table's most useful column is the one that stays near zero. EXHAUSTED is
{base[300][2]:.1%} at horizon 300 and {base[1000][2]:.1%} at 1000, because a
{BUDGET_MULT:.1f}x budget absorbs the retries comfortably.

That is worth stating plainly, because budget is the thing long-running systems are
usually tuned on. **With recovery in place, the budget is not what binds** -- it is
what converts step failure into a cost you can afford
(eq:recovery-converts-failure-to-cost).

What binds instead is the second column. At horizon {10}, silent drift is
{share[10][0]:.1%} of all failures. At {100} it is {share[100][0]:.1%}. At
{1000} it is {share[1000][0]:.1%}, having handed the lead to unrecoverable step
failure at {share[1000][2]:.1%}.

**The dominant failure mode changes with the horizon** (eq:horizon-changes-the-
failure), and for the range most production workflows live in -- tens to a few
hundred steps -- it is the one that produces no error at all. A drifted run
completes. Every step returned success. The answer is wrong because a premise
stopped being true somewhere around step 40 and nothing looked again.

The third table is the fix, and the size of it is the point. At horizon 300,
never rechecking gives {rc[0][0]:.1%}; rechecking every 2 steps gives
{rc[2][0]:.1%}. Even rechecking every 100 steps -- twice in the whole run -- gets
{rc[100][0]:.1%}.

The cost is small: {rc[0][4]:.0f} steps against {rc[2][4]:.0f}, about
{rc[2][4] / rc[0][4] - 1:+.0%}, for a {rc[2][0] - rc[0][0]:+.1%} swing in
correctness. **Re-validating assumptions is the cheapest intervention in this
chapter by a wide margin**, and almost nothing does it, because there is no error
to prompt it.

The fourth table looked like it would show an interior optimum and does not. The
most frequent interval tested wins at every horizon up to 1000. **At these
parameters there is no rechecking frequency that is too often** -- the budget cost
of a check is simply much smaller than the expected cost of continuing on a stale
premise.

That will not hold if a recheck is expensive; if re-validating means re-running a
query that costs as much as the work itself, the optimum moves inward and has to be
computed. But the default should be to check far more often than feels necessary,
because the failure it prevents is invisible until the end.

The last table shows the value peaking in the middle, which is the least obvious
result here. Rechecking is worth {sw[0.0005][1] - sw[0.0005][0]:+.1%} when the world
changes at {0.0005:.2%} per step, {sw[0.0025][2] - sw[0.0025][0]:+.1%} at
{0.0025:.2%}, and {sw[0.02][2] - sw[0.02][0]:+.1%} at {0.02:.2%}.

**Oversight of drift is worth most in a moderately unstable environment.** In a
stable one there is little to catch; in a violently unstable one the premise goes
stale again immediately after the check, and the right response is to shorten the
horizon rather than to check harder.""")

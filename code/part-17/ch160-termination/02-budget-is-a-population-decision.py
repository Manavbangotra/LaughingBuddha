# -*- coding: utf-8 -*-
# Extracted from: Chapter 160 — Termination, Budgets, and Human-in-the-Loop
# Source: src/.../ch160-termination.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Three reasons to stop, and only one of them is the agent's to decide.

A run ends for one of three reasons, and systems conflate them:

  DONE      -- the task is complete
  EXHAUSTED -- the budget ran out
  ESCALATED -- this needs a person

ch:ag-loop showed the first is a classifier the agent should not be trusted with,
because a false stop is a confident wrong answer. This listing is about the second
and third, and about how the budget should be shared across a population of tasks
rather than fixed per task (eq:budget-is-a-population-decision).

The connection is ch:rsn-test-time-compute's allocation result: a fixed per-task
budget is the uniform allocation, and uniform is not optimal when tasks differ in
difficulty. What is new here is that an agent can OBSERVE its own progress, which
makes adaptive policies available that a one-shot sampler does not have.
"""
import numpy as np

rng = np.random.default_rng(2531)

M = 20000               # tasks
TOTAL_PER = 18          # mean step budget per task
NEED = 6                # productive steps required


def make_tasks():
    """Per-step success rate varies across tasks: some are easy, a tail is not."""
    p = np.concatenate([
        rng.beta(8.0, 1.5, size=M // 3),          # easy
        rng.beta(2.0, 3.0, size=M // 3),          # middling
        rng.beta(0.6, 9.0, size=M - 2 * (M // 3)),  # hard
    ])
    rng.shuffle(p)
    return np.clip(p, 0.01, 0.999)


P = make_tasks()
TOTAL = M * TOTAL_PER


def simulate(alloc, p=P, need=NEED):
    """Run each task until it accumulates `need` productive steps or exhausts
    its allocation. Returns (completed, steps used, steps used on failures)."""
    prog = np.zeros(M, dtype=np.int64)
    used = np.zeros(M, dtype=np.int64)
    alive = np.ones(M, dtype=bool)
    for t in range(int(alloc.max())):
        idx = np.flatnonzero(alive & (used < alloc))
        if not len(idx):
            break
        used[idx] += 1
        prog[idx] += rng.random(len(idx)) < p[idx]
        alive[idx[prog[idx] >= need]] = False
    done = prog >= need
    return (float(done.mean()), float(used.mean()),
            float(used[~done].sum() / M), done, used)


def pooled_adaptive(p=P, need=NEED, total=TOTAL, cap=200):
    """One shared pool. Round-robin over live tasks; a task leaves the pool the
    moment it finishes. Nothing is predicted -- the policy just reacts."""
    prog = np.zeros(M, dtype=np.int64)
    used = np.zeros(M, dtype=np.int64)
    alive = np.ones(M, dtype=bool)
    spent = 0
    while spent < total and alive.any():
        idx = np.flatnonzero(alive & (used < cap))
        if not len(idx):
            break
        k = len(idx)
        if spent + k > total:
            idx = idx[: total - spent]
            k = len(idx)
        used[idx] += 1
        prog[idx] += rng.random(k) < p[idx]
        alive[idx[prog[idx] >= need]] = False
        spent += k
    done = prog >= need
    return (float(done.mean()), float(used.mean()),
            float(used[~done].sum() / M), done, used)


print(f"{M:,} tasks, {NEED} productive steps each, a total budget of")
print(f"{TOTAL:,} steps ({TOTAL_PER} per task on average). Per-step success")
print("rates span easy, middling and hard.")
print()
print(f"{'band':>18}{'count':>8}{'mean p':>9}{'share of budget':>18}")
print("-" * 53)
bands = [("p > 0.5 (easy)", P > 0.5), ("0.1 < p < 0.5", (P > 0.1) & (P <= 0.5)),
         ("p < 0.1 (hard)", P <= 0.1)]
for name, m in bands:
    print(f"{name:>18}{int(m.sum()):>8}{float(P[m].mean()):>9.3f}"
          f"{m.mean():>18.0%}")

print()
print()
print("Four budget policies spending the same total.")
print()
print(f"{'policy':>34}{'completed':>12}{'steps/task':>13}"
      f"{'wasted on failures':>20}")
print("-" * 79)
res = {}
uniform = np.full(M, TOTAL_PER, dtype=np.int64)
res["uniform (18 each)"] = simulate(uniform)

# A fixed cap chosen conservatively, spending the remainder nowhere.
tight = np.full(M, 10, dtype=np.int64)
res["tight cap (10 each)"] = simulate(tight)

# Difficulty-aware, using a pilot of 3 real steps to estimate p.
pilot = 3
s = rng.binomial(pilot, P)
est = (s + 1.0) / (pilot + 2.0)
lam = np.clip(est, 1e-6, 1 - 1e-6)
w = np.log(np.maximum(1e-9, 0.02 / lam)) / np.log1p(-lam)
w = np.clip(np.round(w), 1, 400)
w = np.round(w * (TOTAL - pilot * M) / w.sum()).astype(np.int64)
res["pilot of 3, then allocate"] = simulate(np.maximum(w + pilot, 1))

res["pooled, stop when done"] = pooled_adaptive()

for name, r in res.items():
    print(f"{name:>34}{r[0]:>12.1%}{r[1]:>13.1f}{r[2]:>20.1f}")

print()
print()
print("Where the steps go, by difficulty band, under uniform and pooled.")
print()
print(f"{'band':>18}{'uniform':>22}{'pooled':>22}")
print(f"{'':>18}{'done':>10}{'steps':>12}{'done':>10}{'steps':>12}")
print("-" * 62)
_, _, _, u_done, u_used = simulate(uniform)
_, _, _, p_done, p_used = pooled_adaptive()
band_tab = {}
for name, m in bands:
    band_tab[name] = (float(u_done[m].mean()), float(u_used[m].mean()),
                      float(p_done[m].mean()), float(p_used[m].mean()))
    v = band_tab[name]
    print(f"{name:>18}{v[0]:>10.1%}{v[1]:>12.1f}{v[2]:>10.1%}{v[3]:>12.1f}")

print()
print()
print("How the policies respond to a bigger budget.")
print()
print(f"{'budget/task':>13}{'uniform':>11}{'tight cap 10':>15}{'pooled':>10}")
print("-" * 49)
bd = {}
for b in (8, 12, 18, 30, 50):
    u = simulate(np.full(M, b, dtype=np.int64))[0]
    t = simulate(np.full(M, min(b, 10), dtype=np.int64))[0]
    pl = pooled_adaptive(total=M * b)[0]
    bd[b] = (u, t, pl)
    print(f"{b:>13}{u:>11.1%}{t:>15.1%}{pl:>10.1%}")

print()
print()
print("And what an escalation policy buys, on top of the best budget policy.")
print("A task that exhausts its budget is handed to a person, who resolves it")
print("with probability r at a cost of 20 steps' worth of time.")
print()
print(f"{'human resolves':>16}{'auto-completed':>16}{'escalated':>12}"
      f"{'end to end':>13}{'human load':>13}")
print("-" * 70)
auto = pooled_adaptive()[0]
esc_rate = 1 - auto
esc = {}
for r in (0.0, 0.4, 0.7, 0.95):
    total_done = auto + esc_rate * r
    esc[r] = (auto, esc_rate, total_done, esc_rate * 20)
    print(f"{r:>16.0%}{auto:>16.1%}{esc_rate:>12.1%}{total_done:>13.1%}"
          f"{esc_rate * 20:>13.1f}")

print(f"""
The first table is the population, and the last column is what makes a fixed
per-task budget wrong before any policy is chosen. The hard band is
{bands[2][1].mean():.0%} of tasks and receives {bands[2][1].mean():.0%} of the
budget under a uniform allocation, and it completes almost none of them.

The second table prices four ways of spending the same total.

Uniform -- {TOTAL_PER} steps each, the default in every framework -- completes
{res['uniform (18 each)'][0]:.1%} using {res['uniform (18 each)'][1]:.1f} steps per
task on average.

A tight cap of {10} completes {res['tight cap (10 each)'][0]:.1%}. It saves steps
and gives them back to nobody, which is the failure of a per-task budget: **unused
allowance from an easy task does not become available to a hard one.**

Estimating difficulty with a pilot and allocating accordingly reaches
{res['pilot of 3, then allocate'][0]:.1%} -- WORSE than uniform, at higher cost.
The pilot spends {3 * M / TOTAL:.0%} of the budget on measurement and the estimate
is too noisy at three observations to place the rest well, which is
ch:rsn-test-time-compute's finding about pilots reproduced in an agent setting.

And pooling -- one shared budget, round-robin over live tasks, a task leaves the
pool the moment it finishes -- reaches {res['pooled, stop when done'][0]:.1%}.

**The best policy predicts nothing.** It does not estimate difficulty, it does not
route, and it does not decide in advance how much anything deserves. It reacts:
finished tasks stop consuming, so their unspent allowance flows to tasks still
working. That is ch:rsn-test-time-compute's early-stopping result arriving in the
agent setting, and it beats the difficulty-aware policy by
{res['pooled, stop when done'][0] - res['pilot of 3, then allocate'][0]:.1%}.

The third table shows the reallocation happening. Under uniform, the easy band uses
{band_tab['p > 0.5 (easy)'][1]:.1f} steps and the hard band {band_tab['p < 0.1 (hard)'][1]:.1f}
-- the hard tasks consume their full allowance and complete
{band_tab['p < 0.1 (hard)'][0]:.1%} of the time. Under pooling the easy band still
uses {band_tab['p > 0.5 (easy)'][3]:.1f} and the MIDDLE band rises from
{band_tab['0.1 < p < 0.5'][1]:.1f} to {band_tab['0.1 < p < 0.5'][3]:.1f} steps,
taking its completion from {band_tab['0.1 < p < 0.5'][0]:.1%} to
{band_tab['0.1 < p < 0.5'][2]:.1%}.

Note where the gain is NOT. The hard band goes from
{band_tab['p < 0.1 (hard)'][0]:.1%} to {band_tab['p < 0.1 (hard)'][2]:.1%} while
consuming {band_tab['p < 0.1 (hard)'][3]:.1f} steps -- the most of any band, for
almost nothing. **Pooling reallocates toward the middle and still overspends on
the hopeless**, which is why the cap in the pooling policy matters and why
ch:ag-what-is-an-agent's per-task cap does not go away.

The fourth table shows the policies diverging rather than converging as the budget
grows: at {8} steps per task the three are within
{max(bd[8]) - min(bd[8]):.1%} of each other, and at {50} pooling leads uniform by
{bd[50][2] - bd[50][0]:.1%}. A tight cap flatlines at {bd[50][1]:.1%} no matter
how much budget exists, because it refuses to spend it.

**A per-task cap converts a budget increase into nothing.** That is the specific
harm of expressing the budget in the wrong place, and it is invisible until
somebody raises the budget and observes no improvement.

The last table is the third termination reason, and it is the one that changes the
system's ceiling rather than its efficiency.

Pooling completes {auto:.1%} automatically and exhausts on {esc_rate:.1%}. Those
exhausted runs are a VISIBLE failure -- ch:ag-loop's distinction -- so they can be
escalated. If a person resolves {0.7:.0%} of what is escalated, end-to-end
completion is {esc[0.7][2]:.1%}; at {0.95:.0%}, {esc[0.95][2]:.1%}.

The human load is {esc[0.7][3]:.1f} steps' worth of time per task on average, and
it is spent only on runs that already failed -- which is the crucial difference
from the previous listing's confirmation gate. **A gate spends attention on
everything to catch a few; an escalation spends it only on the failures.** Same
people, same hours available, and the second design puts them where the agent has
already said it could not cope.

That is the argument for building escalation before confirmation. It uses a signal
the system produces for free (the budget was exhausted), it is triggered by a
visible failure rather than a predicted one, and it does not habituate because the
volume is bounded by the failure rate.""")

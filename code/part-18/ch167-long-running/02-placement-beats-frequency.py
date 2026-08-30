# -*- coding: utf-8 -*-
# Extracted from: Chapter 167 — Long-Running and Autonomous Workflows
# Source: src/.../ch167-long-running.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Where to put the human in a run that lasts days.

ch:ag-termination put a human at the approval gate and ch:ag-security found the
gate habituates: a reviewer asked constantly approves reflexively, so the catch
rate per pause FALLS as pauses get more frequent.

A long-running workflow makes that worse in a way a short one does not. Each pause
costs wall-clock -- the human answers in hours, not milliseconds -- so pausing
often can mean the run takes a week. And the drift ch:as-long-running's first
listing measured is exactly what a human WOULD catch, if asked at the right moment.

So there are two questions, and only one of them is the one teams argue about:

  how often   pause every k steps -- the frequency question
  where       pause before consequential steps -- the placement question

This listing measures both (eq:placement-beats-frequency).
"""
import numpy as np

rng = np.random.default_rng(3607)

M = 40000
HORIZON = 200
P_WRONG = 0.010         # per-step chance the run goes off-course
P_CONSEQ = 0.06         # share of steps that are consequential (irreversible)
CATCH_0 = 0.85          # catch rate of an attentive reviewer
HALF = 12               # pauses per run at which attention has halved
HOURS = 4.0             # wall-clock hours a pause costs


def catch_rate(n_pauses):
    """ch:ag-termination's habituation: attention decays with how often you ask."""
    return CATCH_0 / (1.0 + n_pauses / HALF)


def run(every=0, placement="uniform", m=M, horizon=HORIZON, p_wrong=P_WRONG,
        conseq_only=False):
    """Walk the horizon. Off-course states accumulate; a pause may catch and
    repair one. `placement='targeted'` pauses only before consequential steps."""
    conseq = rng.random((m, horizon)) < P_CONSEQ
    if placement == "targeted":
        gate = conseq.copy()
    elif every:
        gate = np.zeros((m, horizon), dtype=bool)
        gate[:, ::every] = True
    else:
        gate = np.zeros((m, horizon), dtype=bool)
    n_pauses = gate.sum(1).mean()
    cr = catch_rate(n_pauses)
    off = np.zeros(m, dtype=bool)
    harm = np.zeros(m, dtype=bool)
    for t in range(horizon):
        went = rng.random(m) < p_wrong
        off |= went
        # A gate fires BEFORE the step, and may catch an off-course run.
        g = gate[:, t] & off
        caught = g & (rng.random(m) < cr)
        off &= ~caught
        # A consequential step taken while still off-course does real damage.
        harm |= off & conseq[:, t]
    return (float((~harm).mean()), float(off.mean()), float(n_pauses),
            float(n_pauses * HOURS), float(cr))


print(f"{M:,} runs of a {HORIZON}-step workflow. Each step has a {P_WRONG:.1%}")
print(f"chance of going off-course; {P_CONSEQ:.0%} of steps are consequential, and")
print(f"an off-course run reaching one does harm. An attentive reviewer catches")
print(f"{CATCH_0:.0%}, halving every {HALF} pauses per run (ch:ag-security).")
print()
print(f"{'pause every':>13}{'no harm':>10}{'pauses':>9}{'catch rate':>13}"
      f"{'delay (h)':>12}")
print("-" * 57)
freq = {}
for k in (0, 50, 25, 10, 5, 2, 1):
    r = run(every=k)
    freq[k] = r
    label = "never" if k == 0 else str(k)
    print(f"{label:>13}{r[0]:>10.1%}{r[2]:>9.1f}{r[4]:>13.1%}{r[3]:>12.0f}")

print()
print()
print("The same budget of human attention, spent on consequential steps only")
print("rather than uniformly. Both rows pause a similar number of times.")
print()
print(f"{'placement':>22}{'no harm':>10}{'pauses':>9}{'catch rate':>13}"
      f"{'delay (h)':>12}")
print("-" * 66)
tgt = run(placement="targeted")
# the uniform interval that produces the closest pause count
k_match = max(1, int(round(HORIZON / tgt[2])))
uni = run(every=k_match)
print(f"{('uniform, every ' + str(k_match)):>22}{uni[0]:>10.1%}{uni[2]:>9.1f}"
      f"{uni[4]:>13.1%}{uni[3]:>12.0f}")
print(f"{'targeted':>22}{tgt[0]:>10.1%}{tgt[2]:>9.1f}{tgt[4]:>13.1%}"
      f"{tgt[3]:>12.0f}")

print()
print()
print("Frequency against placement across horizons, since a longer run has more")
print("chances to drift and more consequential steps to reach.")
print()
print(f"{'horizon':>9}{'never':>9}{'uniform 10':>13}{'uniform 2':>12}"
      f"{'targeted':>11}{'best':>11}")
print("-" * 65)
hz = {}
for h in (50, 200, 600):
    row = (run(every=0, horizon=h)[0], run(every=10, horizon=h)[0],
           run(every=2, horizon=h)[0], run(placement="targeted", horizon=h)[0])
    names = ["never", "uniform 10", "uniform 2", "targeted"]
    hz[h] = (row, names[int(np.argmax(row))])
    print(f"{h:>9}{row[0]:>9.1%}{row[1]:>13.1%}{row[2]:>12.1%}{row[3]:>11.1%}"
          f"{hz[h][1]:>11}")

print()
print()
print("What habituation costs. Same designs, with the decay switched off -- an")
print("idealised reviewer whose attention never falls.")
print()
print(f"{'design':>16}{'with decay':>13}{'no decay':>11}{'loss':>10}")
print("-" * 50)
hab = {}
SAVE = HALF
for name, kw in [("uniform 10", dict(every=10)), ("uniform 2", dict(every=2)),
                 ("uniform 1", dict(every=1)),
                 ("targeted", dict(placement="targeted"))]:
    a = run(**kw)[0]
    globals()["HALF"] = 10 ** 9
    b = run(**kw)[0]
    globals()["HALF"] = SAVE
    hab[name] = (a, b)
    print(f"{name:>16}{a:>13.1%}{b:>11.1%}{a - b:>+10.1%}")

print(f"""
Read the first table as a cost curve rather than a benefit curve.

Going from never pausing to pausing every 50 steps buys
{freq[50][0] - freq[0][0]:+.1%}. Going from every 50 to every 1 -- a
{freq[1][2] / freq[50][2]:.0f}x increase in human interruptions and
{freq[1][3] - freq[50][3]:.0f} extra hours of wall-clock -- buys
{freq[1][0] - freq[50][0]:+.1%} more.

The reason is in the catch-rate column: {freq[50][4]:.1%} at every 50 and
{freq[1][4]:.1%} at every 1. **ch:ag-termination's habituation converts additional
oversight into additional noise**, and past a fairly early point the marginal pause
is close to worthless while its cost is undiminished.

The second table is the chapter's result, and it is about placement rather than
frequency.

Spending the SAME twelve pauses on consequential steps only gives
{tgt[0]:.1%} against the uniform schedule's {uni[0]:.1%}. Identical human cost,
identical habituation, {tgt[0] - uni[0]:+.1%} in harm avoided
(eq:placement-beats-frequency).

Better: targeted's {tgt[0]:.1%} on {tgt[2]:.0f} pauses beats uniform-every-2's
{freq[2][0]:.1%} on {freq[2][2]:.0f} pauses. **The targeted schedule matches an
eightfold larger review budget**, at {tgt[3]:.0f} hours of delay against
{freq[2][3]:.0f}.

The reason is structural rather than statistical. A uniform pause catches an
off-course run at a random moment, and most moments are ones where being off-course
costs nothing yet. A pause before a consequential step is asked exactly when the
answer matters -- it is ch:ag-termination's irreversibility criterion used as a
scheduling rule rather than as a policy.

The third table says the ordering holds across horizons, and it says something less
comfortable too. At horizon {600} every design is at or under
{max(hz[600][0]):.1%}. **Past some horizon, gate-based oversight cannot keep a run
on course at any review budget**, because the number of opportunities to drift
outgrows any schedule of checks. The response there is not more gates; it is the
first listing's re-validation, or a shorter horizon.

The last table prices habituation directly by switching it off. An idealised
reviewer whose attention never decayed would take uniform-every-1 to
{hab['uniform 1'][1]:.1%}; the real one gets {hab['uniform 1'][0]:.1%}, a loss of
{hab['uniform 1'][0] - hab['uniform 1'][1]:.1%}.

**Nearly all of the theoretical value of frequent oversight is destroyed by the
frequency itself.** That is the sharpest way to state why placement is the lever:
you cannot buy attention with volume, so the only remaining move is to spend the
attention you have where it changes an outcome.""")

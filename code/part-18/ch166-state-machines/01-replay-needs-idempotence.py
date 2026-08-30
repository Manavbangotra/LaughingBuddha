# -*- coding: utf-8 -*-
# Extracted from: Chapter 166 — State Machines, Events, and Durable Execution
# Source: src/.../ch166-state-machines.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Durable execution is a checkpoint that survives a crash. Replay is not free.

ch:ag-planning's checkpoint lets a failed segment restart from a known-good state.
Durable execution is the same mechanism with the state written to storage, so the
restart can follow a process crash rather than only a logical failure.

The complication is that an agent step usually has a SIDE EFFECT, and replaying it
does it again. A workflow engine that guarantees at-least-once execution therefore
guarantees at-least-once side effects, and the correctness of a resume depends
entirely on how many of the replayed steps are idempotent
(eq:replay-needs-idempotence).

This listing counts the replays and the duplicate effects they cause.
"""
import numpy as np

rng = np.random.default_rng(3413)

M = 40000
STEPS = 12
P_CRASH = 0.05          # chance of a crash after any given step
MAX_ITERS = 200


def run(idem_frac, ck_every, m=M, steps=STEPS, crash=P_CRASH, keys=False):
    """Walk the workflow. A crash rolls the position back to the last durable
    checkpoint; every step between the checkpoint and the crash point is then
    executed a second time. A replayed non-idempotent step duplicates its side
    effect, unless a deduplication key suppresses it."""
    idem = rng.random((m, steps)) < idem_frac
    pos = np.zeros(m, dtype=np.int64)
    anchor = np.zeros(m, dtype=np.int64)
    ran = np.zeros((m, steps), dtype=bool)
    dupes = np.zeros(m, dtype=np.int64)
    corrupt = np.zeros(m, dtype=bool)
    steps_taken = np.zeros(m, dtype=np.int64)
    rows = np.arange(m)
    for _ in range(MAX_ITERS):
        live = (pos < steps) & ~corrupt
        idx = np.flatnonzero(live)
        if not len(idx):
            break
        j = pos[idx]
        steps_taken[idx] += 1
        replay = ran[idx, j]
        bad = replay & (~idem[idx, j]) & (not keys)
        dupes[idx] += bad
        corrupt[idx[bad]] = True
        ran[idx, j] = True
        pos[idx] += 1
        # A checkpoint durably records progress.
        at_ck = (pos[idx] % ck_every) == 0
        anchor[idx[at_ck]] = pos[idx[at_ck]]
        # A crash rolls back to the last durable point.
        crashed = rng.random(len(idx)) < crash
        pos[idx[crashed]] = anchor[idx[crashed]]
    done = (pos >= steps) & ~corrupt
    return (float(done.mean()), float(corrupt.mean()), float(dupes.mean()),
            float(steps_taken.mean()))


print(f"{M:,} runs of a {STEPS}-step workflow, {P_CRASH:.0%} crash chance after")
print("each step. On resume the run replays from the last durable checkpoint,")
print("so every step between the checkpoint and the crash executes twice.")
print()
print(f"{'idempotent steps':>18}{'completed':>12}{'corrupted':>12}"
      f"{'duplicates':>13}{'steps run':>12}")
print("-" * 67)
tab = {}
for f in (0.0, 0.5, 0.8, 0.95, 1.0):
    r = run(f, 3)
    tab[f] = r
    print(f"{f:>18.0%}{r[0]:>12.1%}{r[1]:>12.1%}{r[2]:>13.2f}{r[3]:>12.1f}")

print()
print()
print("Checkpoint frequency decides how much gets replayed, so it decides how")
print("many duplicate side effects a crash causes.")
print()
print(f"{'checkpoint every':>18}{'completed':>12}{'corrupted':>12}"
      f"{'duplicates':>13}{'steps run':>12}")
print("-" * 67)
ck = {}
for c in (1, 2, 3, 6, 12):
    r = run(0.5, c)
    ck[c] = r
    print(f"{c:>18}{r[0]:>12.1%}{r[1]:>12.1%}{r[2]:>13.2f}{r[3]:>12.1f}")

print()
print()
print("A deduplication key makes a non-idempotent step effectively idempotent:")
print("the engine records that the effect happened and suppresses the repeat.")
print()
print(f"{'idempotent steps':>18}{'no key':>12}{'with key':>12}{'gain':>9}")
print("-" * 51)
dk = {}
for f in (0.0, 0.5, 0.8, 1.0):
    a = run(f, 3)[0]
    b = run(f, 3, keys=True)[0]
    dk[f] = (a, b)
    print(f"{f:>18.0%}{a:>12.1%}{b:>12.1%}{b - a:>+9.1%}")

print()
print()
print("And how it all moves with the crash rate, which is what durability is")
print("bought to survive.")
print()
print(f"{'crash rate':>12}{'ck every 12':>14}{'ck every 3':>13}"
      f"{'ck every 1':>13}{'ck 3 + keys':>14}")
print("-" * 66)
cr = {}
for c in (0.01, 0.05, 0.15, 0.30):
    row = (run(0.5, 12, crash=c)[0], run(0.5, 3, crash=c)[0],
           run(0.5, 1, crash=c)[0], run(0.5, 3, crash=c, keys=True)[0])
    cr[c] = row
    print(f"{c:>12.0%}{row[0]:>14.1%}{row[1]:>13.1%}{row[2]:>13.1%}"
          f"{row[3]:>14.1%}")

print(f"""
The first table is the cost of replay, and the first row is the case a workflow
engine's guarantee does not cover.

With no idempotent steps, {tab[0.0][1]:.1%} of runs end corrupted -- a duplicate
side effect somewhere -- against {tab[1.0][1]:.1%} when every step is idempotent.
The completion column tracks it inversely: {tab[0.0][0]:.1%} to {tab[1.0][0]:.1%}.

**An at-least-once execution guarantee is an at-least-once SIDE EFFECT guarantee**
(eq:replay-needs-idempotence), and the engine cannot tell the difference. It
replays the step; whether that is harmless is a property of the step.

The second table is the knob most teams reach for, and it works. Checkpointing
after every step gives {ck[1][0]:.1%} and checkpointing only at the end gives
{ck[12][0]:.1%}, because the amount replayed after a crash is exactly the distance
back to the last durable point.

It is not free: the steps-run column goes {ck[12][3]:.1f} to {ck[1][3]:.1f}, and
each checkpoint is a durable write. **Checkpoint frequency trades write cost
against replay blast radius**, and it is the same trade ch:ag-planning found
between segment count and verification overhead.

The third table is the fix that actually solves the problem rather than shrinking
it. A deduplication key -- the engine records that a particular effect already
happened and suppresses the repeat -- takes every idempotence level to
{dk[0.0][1]:.1%}.

**A key makes a non-idempotent step idempotent from the engine's point of view**,
and it is the difference between mitigating replay and eliminating it. Note the
size: {dk[0.0][1] - dk[0.0][0]:+.1%} at zero natural idempotence, against the
{ck[1][0] - ck[12][0]:+.1%} that checkpointing every step buys.

The last table is why this matters more as systems get less reliable. At
{0.01:.0%} crash rate the coarse-checkpoint design still reaches {cr[0.01][0]:.1%}
and the difference between designs looks academic. At {0.30:.0%} it is
{cr[0.3][0]:.1%} against {cr[0.3][3]:.1%} with keys.

**Durability is bought to survive crashes and its own correctness degrades with
the crash rate** unless the replay is made safe. A design validated in a stable
environment will fail in an unstable one in a way that looks like the environment's
fault.

Three rules follow.

**Ask of every step: what happens if this runs twice?** That question, per tool,
is the entire content of durable-execution correctness, and it is answerable at
design time rather than discovered in an incident.

**Give every non-idempotent effect a deduplication key**, derived from the run and
the step rather than generated fresh -- a fresh key on replay is not a key.

**Checkpoint often enough that the replay window is small**, and treat the write
cost as the price of a smaller blast radius rather than as overhead.""")

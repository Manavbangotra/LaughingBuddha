# -*- coding: utf-8 -*-
# Extracted from: Chapter 157 — Planning and Plan-and-Execute
# Source: src/.../ch157-planning.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Decomposition: why a plan's structure is worth more than its content.

ch:ag-what-is-an-agent found that task LENGTH hurts more than per-step accuracy,
because success is a per-step base raised to k. That suggests an intervention the
previous listing could not measure: do not make the plan better, make the task
shorter -- by cutting it into segments with a checkable boundary between them
(eq:checkpoints-cap-the-exponent).

A checkpoint does one thing. It converts a failure that loses the whole run into
one that loses the current segment, because the segment can be retried from a
state that is known good. The exponent that governs success stops being the task
length and becomes the SEGMENT length.

The cost is that a checkpoint must be verified, which is a call, and the
verification is itself a classifier with the two asymmetric errors ch:ag-loop
measured.
"""
import numpy as np

rng = np.random.default_rng(2087)

N = 40000
K = 12                  # total steps
P_STEP = 0.90           # a step succeeds
BUDGET = 30             # total step budget including retries


def run(segments, p_step=P_STEP, ck_tpr=1.0, ck_fpr=0.0, budget=BUDGET, k=K):
    """Split k steps into `segments` equal parts. After each segment a
    checkpoint verifies it: ck_tpr is the chance a good segment is passed,
    ck_fpr the chance a bad one is passed anyway (and its error carried on)."""
    seg_len = k // segments
    ok = np.ones(N, dtype=bool)
    spent = np.zeros(N, dtype=np.int64)
    alive = np.ones(N, dtype=bool)
    for _ in range(segments):
        seg_done = np.zeros(N, dtype=bool)
        corrupted = np.zeros(N, dtype=bool)
        for _attempt in range(budget):          # retry the segment until pass
            live = alive & ~seg_done & (spent + seg_len <= budget)
            idx = np.flatnonzero(live)
            if not len(idx):
                break
            spent[idx] += seg_len
            good = (rng.random((len(idx), seg_len)) < p_step).all(1)
            u = rng.random(len(idx))
            passed = np.where(good, u < ck_tpr, u < ck_fpr)
            seg_done[idx[passed]] = True
            corrupted[idx[passed & ~good]] = True
            spent[idx] += 1                      # the checkpoint call itself
        alive &= seg_done
        ok &= seg_done & ~corrupted
    return float(ok.mean()), float(spent.mean())


print(f"A {K}-step task, {P_STEP:.0%} per step, budget {BUDGET} steps. Split into")
print("equal segments with a verified checkpoint between them; a failed segment")
print("is retried from the last good state.")
print()
print(f"{'segments':>10}{'steps each':>13}{'completed':>12}{'steps used':>13}")
print("-" * 48)
seg_tab = {}
for m in (1, 2, 3, 4, 6, 12):
    seg_tab[m] = run(m)
    print(f"{m:>10}{K // m:>13}{seg_tab[m][0]:>12.1%}{seg_tab[m][1]:>13.1f}")

print()
print()
print("The same, holding the checkpoint imperfect. A checkpoint that passes bad")
print("work carries the error forward and the task fails anyway.")
print()
print(f"{'segments':>10}{'perfect':>11}{'fpr 5%':>10}{'fpr 15%':>10}"
      f"{'tpr 85%':>11}")
print("-" * 52)
ck_tab = {}
for m in (1, 2, 3, 4, 6):
    a = run(m)[0]
    b = run(m, ck_fpr=0.05)[0]
    c = run(m, ck_fpr=0.15)[0]
    d = run(m, ck_tpr=0.85)[0]
    ck_tab[m] = (a, b, c, d)
    print(f"{m:>10}{a:>11.1%}{b:>10.1%}{c:>10.1%}{d:>11.1%}")

print()
print()
print("Decomposition against a better model, from the same baseline.")
print()
print(f"{'change':>40}{'completed':>12}{'steps':>9}")
print("-" * 61)
mv = {}
for name, args in [("baseline: 1 segment, step 90%", (1, 0.90)),
                   ("step 90% -> 95%, still 1 segment", (1, 0.95)),
                   ("step 90% -> 99%, still 1 segment", (1, 0.99)),
                   ("keep 90%, split into 3 segments", (3, 0.90)),
                   ("keep 90%, split into 4 segments", (4, 0.90)),
                   ("step 95% AND 4 segments", (4, 0.95))]:
    r = run(args[0], p_step=args[1])
    mv[name] = r
    print(f"{name:>40}{r[0]:>12.1%}{r[1]:>9.1f}")

print()
print()
print("Does the best split depend on the budget? Sweep both.")
print()
print(f"{'budget':>8}" + "".join(f"{str(m) + ' seg':>10}" for m in (1, 2, 3, 4, 6))
      + f"{'best':>8}")
print("-" * 66)
bd = {}
for b in (14, 18, 24, 30, 45):
    row = [run(m, budget=b)[0] for m in (1, 2, 3, 4, 6)]
    bd[b] = row
    best = [1, 2, 3, 4, 6][int(np.argmax(row))]
    print(f"{b:>8}" + "".join(f"{v:>10.1%}" for v in row) + f"{best:>8}")

print()
print()
print("And how it moves with task length, at a budget of 2.5x the task.")
print()
print(f"{'steps k':>9}{'1 segment':>12}{'k/4 segments':>15}{'gain':>9}")
print("-" * 45)
kl = {}
for k in (6, 12, 20, 32):
    a = run(1, budget=int(2.5 * k), k=k)[0]
    b = run(max(k // 3, 1), budget=int(2.5 * k), k=k)[0]
    kl[k] = (a, b)
    print(f"{k:>9}{a:>12.1%}{b:>15.1%}{b - a:>+9.1%}")

print(f"""
The first table is the effect, and the size of it is the point.

The same task, the same model, the same {P_STEP:.0%} per step: {seg_tab[1][0]:.1%}
undivided, {seg_tab[6][0]:.1%} split into six segments of two steps. Nothing about
the agent changed. What changed is the exponent -- an undivided task needs
{K} consecutive successes and a segmented one needs {K // 6}, retried
(eq:checkpoints-cap-the-exponent).

Note that it turns over: {seg_tab[12][0]:.1%} at twelve segments, below the
{seg_tab[6][0]:.1%} at six. A checkpoint costs a call, and at one step per segment
the verification overhead is as large as the work -- {seg_tab[12][1]:.1f} steps
against {seg_tab[6][1]:.1f}. **There is an interior optimum in how finely to cut**,
and it is set by the ratio of checkpoint cost to segment length.

The second table is the thing that makes this harder than it looks, and it is the
reason decomposition is not free.

Every checkpoint is a classifier, with ch:ag-loop's two asymmetric errors. A
checkpoint that PASSES BAD WORK carries the error into the next segment, where it
cannot be repaired -- the retry mechanism only restores to the last state the
checkpoint approved. At six segments, a {0.15:.0%} false-pass rate takes
completion from {ck_tab[6][0]:.1%} to {ck_tab[6][2]:.1%}.

And the damage grows with the number of segments, because more segments means more
checkpoints to fool. Compare the {0.15:.0%} column down the rows: the gap from
perfect widens from {ck_tab[1][0] - ck_tab[1][2]:.1%} at one segment to
{ck_tab[6][0] - ck_tab[6][2]:.1%} at six.

**So decomposition trades one exponent for another.** It removes the task length
from the success exponent and adds the checkpoint count. That is a good trade only
while the checkpoint is more reliable than a step, which is exactly why the
checkpoint should be an executable check rather than a judgement --
ch:rsn-tool-assisted's argument arriving as a structural requirement.

The third table prices it against the alternative everyone reaches for first.

Improving the model from {0.90:.0%} to {0.95:.0%} per step takes completion from
{mv['baseline: 1 segment, step 90%'][0]:.1%} to
{mv['step 90% -> 95%, still 1 segment'][0]:.1%}. Keeping the {0.90:.0%} model and
cutting the task into three takes it to
{mv['keep 90%, split into 3 segments'][0]:.1%}.

**Splitting the task beats a five-point model improvement**, and it is a change to
the prompt and the control flow rather than to the model. The two also compose:
{0.95:.0%} steps with four segments reaches
{mv['step 95% AND 4 segments'][0]:.1%}.

The fourth table is the caveat that matters most in production, and it is a
failure mode rather than a diminishing return.

At a budget of {14} steps, splitting into four segments scores
{bd[14][3]:.1%}. Not "less than one segment" -- zero. The checkpoint calls plus
the segment retries do not fit in the budget at all, so no run ever finishes.
**Decomposition consumes budget before it saves any**, and a system that adds
checkpoints without raising the step budget can go from working to completely
broken in one change.

Above the threshold the ordering reverses hard: at budget {30} the best split is
{6} segments at {bd[30][4]:.1%} against one segment's {bd[30][0]:.1%}. So the
budget and the split have to be chosen together, and neither is meaningful alone.

The last table is the reason this is the most important lever in the part. The
gain from decomposition grows with task length: {kl[6][1] - kl[6][0]:+.1%} at
{6} steps, {kl[12][1] - kl[12][0]:+.1%} at {12}, {kl[32][1] - kl[32][0]:+.1%} at
{32}, where an undivided task completes {kl[32][0]:.1%} of the time and a
segmented one {kl[32][1]:.1%}.

That is the direct consequence of what it does to the exponent, and it says
something specific about cite:liu2024agentbench's finding that long-horizon
consistency is the agent bottleneck. **The bottleneck is not that models cannot
reason over long horizons. It is that nothing was checkpointing them**, so a
twenty-step task was being asked to succeed twenty times consecutively rather than
five times consecutively, four times.

Which is the honest resolution of this chapter's two halves. The previous listing
found planning-as-prediction to be a weak lever, working on the capability models
are worst at. This one finds planning-as-STRUCTURE to be the strongest lever
available -- and the structure that matters is not the sequence of actions. It is
the set of points at which you can verify where you are and retry from there.

**A plan whose steps are checkable is worth far more than a plan whose steps are
correct**, and if you have to choose which property to optimise, the arithmetic
above says which one.""")

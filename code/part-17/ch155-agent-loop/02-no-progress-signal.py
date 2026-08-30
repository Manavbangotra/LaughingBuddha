# -*- coding: utf-8 -*-
# Extracted from: Chapter 155 — The Agent Loop: Perception, Decision, Action
# Source: src/.../ch155-agent-loop.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Where the steps go: progress, repetition, and the cheapest fix for a loop.

The previous listing showed the stopping decision dominating success. This one is
about the other half of the loop's arithmetic: what the steps are SPENT on, and
why an agent that is not making progress usually keeps not making progress
(eq:no-progress-signal).

The mechanism is specific. An agent chooses its next action from the context, and
after a failed action the context is nearly the same as before it -- the failure
added an observation and removed nothing. So the same context produces the same
action, and the loop repeats. That is not a mysterious pathology; it is what a
policy does when its input barely changed.

Three interventions are measured against it, and the cheapest one wins.
"""
import numpy as np

rng = np.random.default_rng(1811)

N = 80000
NEED = 6
HORIZON = 25
P_ACT = 0.82           # a fresh action makes progress
STICK = 0.75           # after a failure, chance of repeating the same action


def run(mode, p_act=P_ACT, stick=STICK, horizon=HORIZON):
    """mode:
       'naive'    -- repeat-prone: a failed action is likely to be retried as is
       'dedupe'   -- an action already tried and failed is not tried again
       'temp'     -- after a failure, sample a different action with prob 1-stick
                     (i.e. raise temperature only when stuck)
       'ideal'    -- never repeats
    """
    prog = np.zeros(N, dtype=np.int64)
    wasted = np.zeros(N, dtype=np.int64)
    repeats = np.zeros(N, dtype=np.int64)
    failed_last = np.zeros(N, dtype=bool)
    tried_bad = np.zeros(N, dtype=np.int64)     # how many distinct duds tried
    alive = np.ones(N, dtype=bool)
    steps = np.zeros(N, dtype=np.int64)
    for _ in range(horizon):
        idx = np.flatnonzero(alive)
        if not len(idx):
            break
        steps[idx] += 1
        if mode == "naive":
            rep = failed_last[idx] & (rng.random(len(idx)) < stick)
        elif mode == "temp":
            rep = failed_last[idx] & (rng.random(len(idx)) < stick * 0.35)
        else:
            rep = np.zeros(len(idx), dtype=bool)
        # A repeated action repeats its outcome: it already failed.
        ok = np.where(rep, False, rng.random(len(idx)) < p_act)
        if mode == "dedupe":
            # Ruling out duds raises the chance the next fresh action works.
            boost = 1.0 + 0.06 * np.minimum(tried_bad[idx], 5)
            ok = rng.random(len(idx)) < np.minimum(p_act * boost, 0.99)
        prog[idx[ok]] += 1
        repeats[idx[rep]] += 1
        wasted[idx[~ok & ~rep]] += 1
        tried_bad[idx[~ok & ~rep]] += 1
        failed_last[idx] = ~ok
        fin = prog[idx] >= NEED
        alive[idx[fin]] = False
    done = prog >= NEED
    return (float(done.mean()), float(steps.mean()), float(repeats.mean()),
            float(wasted.mean()))


MODES = [("naive (repeats on failure)", "naive"),
         ("raise temperature when stuck", "temp"),
         ("do not retry a failed action", "dedupe"),
         ("never repeats (ideal)", "ideal")]

print(f"A task needs {NEED} productive steps in a horizon of {HORIZON}. A fresh")
print(f"action works {P_ACT:.0%} of the time. After a failure the naive agent")
print(f"retries the same action {STICK:.0%} of the time, because its context")
print("barely changed.")
print()
print(f"{'loop policy':>32}{'completed':>12}{'steps':>9}{'repeats':>10}"
      f"{'wasted':>9}")
print("-" * 72)
res = {}
for name, m in MODES:
    r = run(m)
    res[name] = r
    print(f"{name:>32}{r[0]:>12.1%}{r[1]:>9.2f}{r[2]:>10.2f}{r[3]:>9.2f}")

print()
print()
print("How much of the horizon does repetition consume? Sweep the stickiness --")
print("how strongly a failed action pulls the agent to try it again.")
print()
print(f"{'stickiness':>12}{'completed':>12}{'repeats':>10}{'wasted':>17}")
print(f"{'':>12}{'':>12}{'per run':>10}{'share of steps':>17}")
print("-" * 51)
st_tab = {}
for st in (0.0, 0.25, 0.50, 0.75, 0.90):
    r = run("naive", stick=st)
    st_tab[st] = r
    print(f"{st:>12.0%}{r[0]:>12.1%}{r[2]:>10.2f}"
          f"{(r[2] + r[3]) / r[1]:>17.1%}")

print()
print()
print("Does a bigger horizon fix it? Naive against dedupe, horizon swept.")
print()
print(f"{'horizon':>9}{'naive':>22}{'dedupe':>20}")
print(f"{'':>9}{'completed':>12}{'steps':>10}{'completed':>11}{'steps':>9}")
print("-" * 51)
hz, hzd = {}, {}
for h in (8, 10, 15, 25, 40, 60):
    r = run("naive", horizon=h)
    d = run("dedupe", horizon=h)
    hz[h], hzd[h] = r, d
    print(f"{h:>9}{r[0]:>12.1%}{r[1]:>10.2f}{d[0]:>11.1%}{d[1]:>9.2f}")

print()
print()
print("And the comparison that decides where to spend: a better model against")
print("a loop-detection rule, at the same horizon.")
print()
print(f"{'change':>40}{'completed':>12}{'steps':>9}")
print("-" * 61)
base = run("naive")
opts = [("baseline (naive, action 82%)", ("naive", P_ACT)),
        ("action 82% -> 90%", ("naive", 0.90)),
        ("action 82% -> 96%", ("naive", 0.96)),
        ("keep 82%, add dedupe", ("dedupe", P_ACT)),
        ("keep 82%, temperature on failure", ("temp", P_ACT))]
cmp_ = {}
for name, (m, pa) in opts:
    r = run(m, p_act=pa)
    cmp_[name] = r
    print(f"{name:>40}{r[0]:>12.1%}{r[1]:>9.2f}")

nv = res["naive (repeats on failure)"]
dd = res["do not retry a failed action"]
tp = res["raise temperature when stuck"]
idl = res["never repeats (ideal)"]
print(f"""
The first table is where the steps go, and the repeats column is the whole
subject.

The naive agent completes {nv[0]:.1%} of tasks, spending {nv[1]:.2f} steps of
which {nv[2]:.2f} are repeats of an action that already failed. That is
{nv[2] / nv[1]:.0%} of its budget spent re-running something it has already
watched fail.

The mechanism is not exotic. **After a failed action the context is nearly
unchanged** -- the failure appended an observation and removed nothing -- so a
policy conditioned on that context produces nearly the same action. A loop is a
fixed point of the policy, not a bug in it, and describing it as "the agent got
confused" gets the causality backwards.

The second table sweeps how strongly a failure pulls the agent back to the same
action. At {0:.0%} stickiness the agent completes {st_tab[0.0][0]:.1%}; at
{0.9:.0%} it completes {st_tab[0.9][0]:.1%} and wastes
{(st_tab[0.9][2] + st_tab[0.9][3]) / st_tab[0.9][1]:.0%} of its steps.

Note that stickiness is not a property anybody chose. It is the degree to which
a failure changes the context, and a tool that returns a terse error changes it
less than one that returns a specific fault -- which is ch:ag-tool-calling's
error-message result arriving as a loop property. **The error message is also a
loop-breaking mechanism**, and that is a second, independent reason to write it.

The third table is the response most teams reach for first, and the comparison
column is what makes it the wrong one.

Raising the naive agent's horizon does work: {hz[8][0]:.1%} at {8} steps,
{hz[25][0]:.1%} at {25}, {hz[60][0]:.1%} at {60}. So "give it more budget" is not
useless advice.

But look at what dedupe achieves at each horizon. At {8} steps dedupe completes
{hzd[8][0]:.1%} against naive's {hz[8][0]:.1%}; at {10} it is already
{hzd[10][0]:.1%} against {hz[10][0]:.1%}, and it stays there.

**The naive agent needs a horizon of {25} to reach what dedupe reaches at {8}**,
and it spends {hz[25][1]:.2f} steps doing it against dedupe's {hzd[8][1]:.2f} --
{hz[25][1] / hzd[8][1]:.1f} times the cost for a slightly worse result. Dedupe's
step count is flat at {hzd[60][1]:.2f} across every horizon, because it never
needed the extra room.

So the horizon does buy completion, and it buys it inefficiently. **A bigger
budget buys a stuck agent more time to be stuck**, and it is the intervention
that looks free because it requires no code and appears in no diff.

The fourth table is the comparison that matters, and the ordering is decisive.

Improving the action from {P_ACT:.0%} to {0.96:.0%} -- a large model
investment -- takes completion from {cmp_['baseline (naive, action 82%)'][0]:.1%}
to {cmp_['action 82% -> 96%'][0]:.1%}. Keeping the {P_ACT:.0%} model and simply
refusing to re-issue an action that already failed takes it to
{cmp_['keep 82%, add dedupe'][0]:.1%}, in {cmp_['keep 82%, add dedupe'][1]:.2f}
steps against the baseline's {cmp_['baseline (naive, action 82%)'][1]:.2f}.

**A loop-detection rule beats a large model improvement, and it is about fifteen
lines of code.** It is not doing anything clever: it maintains the set of actions
already attempted in this run and removes them from consideration. The reason it
works so well is that it attacks the term that a better model does not -- a more
accurate policy still produces the same action from the same context, so it gets
stuck less often but exits no faster once it is.

Raising the temperature on failure -- a softer version of the same idea, sampling
a different action rather than forbidding the old one -- reaches
{cmp_['keep 82%, temperature on failure'][0]:.1%}. Most of the benefit, none of
the bookkeeping, and it degrades gracefully when the "same action" is hard to
define, which in a real agent it often is.

The general shape is worth naming because it recurs through the rest of this
part. **An agent's failures divide into ones a better policy fixes and ones only
a change of state fixes.** Repetition is the second kind: the policy is behaving
correctly given its input, and the fix is to change the input. Every effective
loop-breaking technique in this part -- deduplication, temperature on failure,
informative errors, replanning, an explicit scratchpad -- is a way of making the
context after a failure genuinely different from the context before it.""")

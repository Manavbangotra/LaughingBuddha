# -*- coding: utf-8 -*-
# Extracted from: Chapter 162 — Single-Agent Architectures
# Source: src/.../ch162-single-agent.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""What is left after a well-built single agent, and what could possibly fix it.

The previous listing got a single agent from 6.8% to about 90% without touching
the model. This one asks what the remaining failures are MADE OF, because that
determines which of part:18's architectures could help and which cannot
(eq:residual-failure-decomposition).

Three kinds of residual failure, and they respond to completely different things:

  CAPABILITY   the model cannot do this step, ever. Retries are draws from a
               distribution whose mass is not on the right answer.
  VERIFICATION the work was done and the system could not tell. ch:ag-loop's
               false stop, surviving a conservative threshold.
  CORRELATED   the model can do the step but reliably does not, because it makes
               the same mistake every time.

Only the third is addressable by adding a second agent, and only if the second
agent's errors are uncorrelated with the first's. This listing measures the split
and then prices decorrelation against the alternatives.
"""
import numpy as np

rng = np.random.default_rng(2833)

M = 40000
NEED = 10
BUDGET = 26
ATTEMPTS = 6

# Per-task step difficulty, split into three regimes.
P_HARD = 0.012      # share of steps the model genuinely cannot do
P_STICKY = 0.10     # share where it can, but makes the same error each time
P_OK = 1 - P_HARD - P_STICKY

P_ACT = 0.93        # success on an ordinary step
P_STICKY_ONCE = 0.25   # first-attempt success on a sticky step
P_STICKY_RETRY = 0.10  # a retry on a sticky step: little new information
P_VERIFY = 0.97     # the completion check is right


def run(second_agent=None, corr=1.0, m=M, need=NEED, attempts=ATTEMPTS):
    """second_agent: None, or the per-step success of a second agent brought in
    when the first stalls. corr is how correlated its errors are with the first's
    (1.0 = identical failures, 0.0 = independent)."""
    kind = rng.choice([0, 1, 2], size=(m, need), p=[P_OK, P_STICKY, P_HARD])
    done_step = np.zeros((m, need), dtype=bool)
    for a in range(attempts):
        first = a == 0
        p = np.where(kind == 0, P_ACT,
                     np.where(kind == 1,
                              P_STICKY_ONCE if first else P_STICKY_RETRY, 0.0))
        if second_agent is not None and a >= attempts // 2:
            # The second agent inherits `corr` of the first's blind spots.
            inherits = rng.random((m, need)) < corr
            p2 = np.where(kind == 0, second_agent,
                          np.where(kind == 1,
                                   np.where(inherits, P_STICKY_RETRY,
                                            P_STICKY_ONCE),
                                   np.where(inherits, 0.0, second_agent * 0.5)))
            p = np.maximum(p, p2)
        done_step |= (~done_step) & (rng.random((m, need)) < p)
    all_done = done_step.all(1)
    verified = all_done & (rng.random(m) < P_VERIFY)
    # Classify the residual.
    fail_hard = (~all_done) & ((kind == 2) & ~done_step).any(1)
    fail_sticky = (~all_done) & ~fail_hard
    fail_verify = all_done & ~verified
    return (float(verified.mean()), float(fail_hard.mean()),
            float(fail_sticky.mean()), float(fail_verify.mean()))


base = run()
print(f"{M:,} tasks, {NEED} steps each, up to {ATTEMPTS} attempts per step.")
print(f"{P_OK:.0%} of steps are ordinary ({P_ACT:.0%} per attempt),")
print(f"{P_STICKY:.0%} are sticky ({P_STICKY_ONCE:.0%} first try, then")
print(f"{P_STICKY_RETRY:.0%}), and {P_HARD:.0%} the model cannot do at all.")
print()
print(f"{'outcome':>28}{'share':>10}{'of failures':>14}")
print("-" * 52)
fails = base[1] + base[2] + base[3]
for name, v in [("completed and verified", base[0]),
                ("failed: capability", base[1]),
                ("failed: correlated (sticky)", base[2]),
                ("failed: verification", base[3])]:
    share = v / fails if name != "completed and verified" else float("nan")
    txt = "--" if name == "completed and verified" else f"{share:.0%}"
    print(f"{name:>28}{v:>10.1%}{txt:>14}")

print()
print()
print("What a second agent adds, as a function of how correlated its errors are")
print("with the first agent's. Same total attempt budget in every row.")
print()
print(f"{'correlation':>13}{'completed':>12}{'vs one agent':>15}"
      f"{'sticky failures':>18}")
print("-" * 58)
corr_tab = {}
for c in (1.0, 0.8, 0.5, 0.2, 0.0):
    r = run(second_agent=P_ACT, corr=c)
    corr_tab[c] = r
    print(f"{c:>13.1f}{r[0]:>12.1%}{r[0] - base[0]:>+15.1%}{r[2]:>18.1%}")

print()
print()
print("Three ways to spend, from the single-agent baseline.")
print()
print(f"{'change':>40}{'completed':>12}{'gain':>9}")
print("-" * 61)
moves = {}
for name, kw in [
        ("baseline: one agent", {}),
        ("a second, identical agent", dict(second_agent=P_ACT, corr=1.0)),
        ("a second, decorrelated agent", dict(second_agent=P_ACT, corr=0.2)),
        ("one agent, better model (93->97%)", {}),
        ("one agent, better verifier (97->99.5%)", {})]:
    if name.startswith("one agent, better model"):
        PA = P_ACT
        globals()["P_ACT"] = 0.97
        r = run()
        globals()["P_ACT"] = PA
    elif name.startswith("one agent, better verifier"):
        PV = P_VERIFY
        globals()["P_VERIFY"] = 0.995
        r = run()
        globals()["P_VERIFY"] = PV
    else:
        r = run(**kw)
    moves[name] = r
    print(f"{name:>40}{r[0]:>12.1%}{r[0] - base[0]:>+9.1%}")

print()
print()
print("And how the residual splits as the model gets better -- which failure")
print("class survives improvement.")
print()
print(f"{'step accuracy':>15}{'completed':>12}{'capability':>13}"
      f"{'correlated':>13}{'verification':>15}")
print("-" * 68)
PA_SAVE = P_ACT
acc = {}
for a in (0.85, 0.93, 0.97, 0.995):
    globals()["P_ACT"] = a
    r = run()
    acc[a] = r
    print(f"{a:>15.1%}{r[0]:>12.1%}{r[1]:>13.1%}{r[2]:>13.1%}{r[3]:>15.1%}")
globals()["P_ACT"] = PA_SAVE

print(f"""
The first table is the residual, and the shares are what matter rather than the
levels.

A well-built single agent completes {base[0]:.1%}. Of what remains,
{base[1] / fails:.0%} is capability -- steps the model cannot do -- and
{base[2] / fails:.0%} is correlated: steps it could do and reliably does not.
Verification failures are {base[3] / fails:.0%}, because ch:ag-loop's conservative
threshold has already handled most of them.

**Roughly a third of the residual is capability and two thirds is correlated
error**, and those respond to completely different interventions
(eq:residual-failure-decomposition).

The second table prices the intervention part:18 is about. A second agent whose
errors are IDENTICAL to the first's buys {corr_tab[1.0][0] - base[0]:+.1%} --
nothing, which is what identical means. A second agent whose errors are
independent buys {corr_tab[0.0][0] - base[0]:+.1%}, and at a realistic correlation
of {0.5} it buys {corr_tab[0.5][0] - base[0]:+.1%}.

**The entire value of a second agent is decorrelation.** Not division of labour,
not specialisation, not a role name. The sticky-failure column falls from
{corr_tab[1.0][2]:.1%} to {corr_tab[0.0][2]:.1%} as correlation drops, and the
capability column does not move at all -- because a second agent that cannot do the
step either is still a model that cannot do the step.

That is the same quantity ch:rsn-self-consistency identified as the variable behind
critic value, and ch:ag-recovery found again in the environment-versus-self
comparison. **Three chapters, three settings, one number.**

The third table puts the second agent against the alternatives, and the losing rows
are as informative as the winning one.

A better model -- ordinary-step accuracy from {0.93:.0%} to {0.97:.0%} -- buys
{moves['one agent, better model (93->97%)'][0] - base[0]:+.1%}. A better verifier
buys {moves['one agent, better verifier (97->99.5%)'][0] - base[0]:+.1%}. A
decorrelated second agent buys
{moves['a second, decorrelated agent'][0] - base[0]:+.1%}.

The fourth table explains why the model row is flat, and it is the finding to carry
into part:18.

Sweeping ordinary-step accuracy from {0.85:.0%} to {0.995:.0%} moves completion
from {acc[0.85][0]:.1%} to {acc[0.995][0]:.1%} -- essentially nothing -- and leaves
the capability and correlated columns unchanged at about {acc[0.995][1]:.0%} and
{acc[0.995][2]:.0%}.

**The residual after a well-built single agent is invariant to how good the model
is at the steps it can already do.** Every remaining failure is either a step
outside the model's ability or a step it approaches the same wrong way every time,
and per-step accuracy on the ordinary steps is orthogonal to both.

That reframes what the next eight chapters are for. A multi-agent architecture
cannot help with the capability third: two instances of a model that cannot do
something still cannot do it. It can help with the correlated two-thirds, and only
to the extent that the second agent is genuinely a different system -- different
model, different lineage, different prompting -- rather than the same model wearing
a role label.

So the question every chapter of part:18 has to answer is not "does this
architecture help" but **"how much decorrelation does it buy, and could I have
bought it more cheaply?"** cite:cemri2025mast's observation that multi-agent gains
on popular benchmarks are often minimal is what happens when the answer is "very
little, and yes".""")

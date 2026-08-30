# -*- coding: utf-8 -*-
# Extracted from: Chapter 163 — Multi-Agent Architectures and Communication
# Source: src/.../ch163-multi-agent.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""What a handoff costs, and why the number of agents should be minimised.

Passing work between agents is a serialisation. The sending agent compresses its
state into a message; the receiving agent reconstructs enough of it to continue.
Both halves are lossy, and the loss is multiplicative in the number of handoffs
(eq:handoff-is-a-bottleneck).

This is ch:rsn-cot's token bottleneck at the level of a whole agent, and
ch:rsn-tool-assisted's boundary-crossing cost with a bigger boundary. The
arithmetic is the same and the constant is worse, because an agent's state is
larger and less structured than a tool call's arguments.

This listing holds the total work fixed and varies only how many times it changes
hands.
"""
import numpy as np

rng = np.random.default_rng(2917)

M = 40000
WORK = 12               # total productive steps the task needs
P_STEP = 0.94           # a step succeeds
ATTEMPTS = 3            # retries available per step


def run(handoffs, p_write=0.93, p_read=0.95, structured=False, m=M,
        work=WORK, p_step=P_STEP, attempts=ATTEMPTS):
    """The work is split into `handoffs + 1` stretches. Between stretches the
    state is serialised and reconstructed. `structured` means the state has a
    schema rather than being prose, which raises both halves."""
    if structured:
        p_write = 1 - (1 - p_write) * 0.35
        p_read = 1 - (1 - p_read) * 0.35
    ok = np.ones(m, dtype=bool)
    stretch = max(1, work // (handoffs + 1))
    for h in range(handoffs + 1):
        n = stretch if h < handoffs else work - stretch * handoffs
        n = max(n, 0)
        for _ in range(n):
            got = np.zeros(m, dtype=bool)
            for _a in range(attempts):
                got |= (~got) & (rng.random(m) < p_step)
            ok &= got
        if h < handoffs:
            ok &= rng.random(m) < p_write
            ok &= rng.random(m) < p_read
    return float(ok.mean())


print(f"A task needing {WORK} productive steps at {P_STEP:.0%} each, with")
print(f"{ATTEMPTS} attempts per step. The work is split across agents; between")
print("stretches the state is written out and read back in.")
print()
print(f"{'handoffs':>10}{'agents':>9}{'steps each':>13}{'prose state':>14}"
      f"{'structured state':>19}")
print("-" * 65)
tab = {}
for h in (0, 1, 2, 3, 5, 11):
    a = run(h)
    b = run(h, structured=True)
    tab[h] = (a, b)
    print(f"{h:>10}{h + 1:>9}{max(1, WORK // (h + 1)):>13}{a:>14.1%}"
          f"{b:>19.1%}")

print()
print()
print("The handoff penalty in isolation: same work, same steps, only the")
print("serialisation quality varies.")
print()
print(f"{'write x read':>14}{'1 handoff':>12}{'3 handoffs':>13}"
      f"{'11 handoffs':>14}")
print("-" * 53)
q_tab = {}
for pw, pr in ((0.99, 0.99), (0.95, 0.97), (0.93, 0.95), (0.88, 0.90),
               (0.80, 0.85)):
    row = [run(h, p_write=pw, p_read=pr) for h in (1, 3, 11)]
    q_tab[(pw, pr)] = row
    print(f"{pw * pr:>14.1%}{row[0]:>12.1%}{row[1]:>13.1%}{row[2]:>14.1%}")

print()
print()
print("Is the handoff cost really multiplicative? Fit success against handoff")
print("count and check the implied per-handoff factor.")
print()
print(f"{'handoffs':>10}{'measured':>11}{'no-handoff x f^h':>19}{'residual':>11}")
print("-" * 51)
f0 = tab[0][0]
fac = (tab[1][0] / f0)
for h in (0, 1, 2, 3, 5, 11):
    pred = f0 * fac ** h
    print(f"{h:>10}{tab[h][0]:>11.1%}{pred:>19.1%}"
          f"{tab[h][0] - pred:>+11.1%}")
print()
print(f"  implied per-handoff factor: {fac:.3f}")

print()
print()
print("What a handoff has to buy to be worth taking. A specialist is better at")
print("its stretch; how much better does it have to be? Shown with retries")
print("available and without, because retries already saturate an easy step.")
print()
print(f"{'specialist edge':>17}{'3 attempts':>26}{'1 attempt':>24}")
print(f"{'':>17}{'no handoff':>13}{'3 handoffs':>13}{'no handoff':>12}"
      f"{'3 handoffs':>12}")
print("-" * 67)
edge = {}
for e in (0.0, 0.02, 0.04, 0.06, 0.10):
    ps = min(P_STEP + e, 0.999)
    a0 = run(0, p_step=P_STEP)
    a3 = run(3, p_step=ps)
    b0 = run(0, p_step=P_STEP, attempts=1)
    b3 = run(3, p_step=ps, attempts=1)
    edge[e] = (a0, a3, b0, b3)
    print(f"{e:>17.0%}{a0:>13.1%}{a3:>13.1%}{b0:>12.1%}{b3:>12.1%}")

print()
print()
print("And how it scales with task size, at a fixed 3 handoffs.")
print()
print(f"{'work steps':>12}{'no handoff':>13}{'3 handoffs':>13}{'loss':>9}")
print("-" * 47)
wk = {}
for w in (6, 12, 24, 40):
    a = run(0, work=w)
    b = run(3, work=w)
    wk[w] = (a, b)
    print(f"{w:>12}{a:>13.1%}{b:>13.1%}{b - a:>+9.1%}")

print(f"""
The first table is the cost, and the shape of the column is the finding.

The same {WORK} steps of work: one agent completes {tab[0][0]:.1%}, two agents
{tab[1][0]:.1%}, four {tab[3][0]:.1%}, twelve {tab[11][0]:.1%}. **Nothing about
the work changed.** The only difference is how many times the state was written
out and read back in.

The structured column is the cheapest available mitigation. Giving the handoff a
schema rather than prose takes twelve agents from {tab[11][0]:.1%} to
{tab[11][1]:.1%}, and three handoffs from {tab[3][0]:.1%} to {tab[3][1]:.1%}. That
is the same intervention ch:ag-tool-calling recommended for tool arguments, at a
larger boundary: **a handoff is a tool call whose argument is an entire working
state, and constraining its shape helps for the same reason.**

The third table checks whether the cost is really multiplicative, and it is
almost exactly so. Predicting from the single-handoff factor of {fac:.3f} raised
to the handoff count reproduces every measured value within
{max(abs(tab[h][0] - f0 * fac ** h) for h in (0, 1, 2, 3, 5, 11)):.1%}
(eq:handoff-is-a-bottleneck).

So the rule is the one ch:rsn-tool-assisted reached about tool boundaries, with a
worse constant: **success falls geometrically in the number of times control
changes hands.** An architecture diagram with six agents in a chain is
{fac:.3f}^5 = {fac ** 5:.2f} of the success of the same work done by one, before
any of them has done anything wrong.

The second table says what the constant depends on, and it is entirely the
serialisation quality. At {0.98:.0%} write-times-read the three-handoff
configuration reaches {q_tab[(0.99, 0.99)][1]:.1%}; at {0.68:.0%} it reaches
{q_tab[(0.80, 0.85)][1]:.1%}. At eleven handoffs the same range is
{q_tab[(0.99, 0.99)][2]:.1%} to {q_tab[(0.80, 0.85)][2]:.1%}.

**Handoff quality is raised to the power of the handoff count**, which means it is
the single most leveraged number in a multi-agent design and the one least often
measured.

The fourth table asks the question that decides whether any of this is worth it:
if the receiving agent is BETTER at its stretch, does the specialisation pay for
the handoff?

With retries available, no. At every specialist edge from {0:.0%} to {0.10:.0%},
three handoffs land at about {edge[0.10][1]:.1%} against a single agent's
{edge[0.10][0]:.1%}. The edge buys nothing because **retries have already
saturated the steps** -- ch:ag-loop's finding that a loop with slack is not a
chain, arriving as an argument against specialisation.

Without retries the picture reverses. A single agent reaches
{edge[0.0][2]:.1%}, three handoffs with no edge reach {edge[0.0][3]:.1%}, and
three handoffs with a {0.04:.0%} edge reach {edge[0.04][3]:.1%} -- ahead.

**Specialisation pays only where retries are unavailable**, and retries are
unavailable when actions have side effects, when the budget is tight, or when the
step is expensive. That is a much narrower condition than the usual case for
splitting work across specialists, and it is checkable: if your agent can retry a
failed step, a specialist's per-step edge is being spent on steps that would have
succeeded on the second try anyway.

The last table confirms the loss is a property of the handoffs rather than of the
task. At {6} steps of work three handoffs cost {wk[6][1] - wk[6][0]:.1%}; at
{40} steps they cost {wk[40][1] - wk[40][0]:.1%}. **The penalty is constant in task
size**, because it depends on how many boundaries there are and not on how much
work sits between them.

Which gives the design rule: **hand off as few times as the work allows, and when
you must, hand off a schema rather than a story.** The number of agents in a
system is a cost, not a feature, and an architecture that adds one should be able
to say what it buys against a factor of {fac:.3f}.""")

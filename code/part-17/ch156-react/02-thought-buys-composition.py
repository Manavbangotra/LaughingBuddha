# -*- coding: utf-8 -*-
# Extracted from: Chapter 156 — ReAct and Interleaved Reasoning and Acting
# Source: src/.../ch156-react.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Does the "reasoning" half of ReAct earn its tokens?

cite:yao2023react's contribution is two things bolted together, and they are
usually evaluated as one. The ACTING half is the interleaving the previous listing
measured. The REASONING half is emitting a thought before each action, and it has
its own justification, its own cost, and its own scope.

ch:rsn-cot established what a thought buys: serial computation the forward pass
does not have. cite:sprague2024tocot established where that helps: tasks whose
difficulty is the DEPTH of a composition, and almost nowhere else. This listing
applies both to the agent setting (eq:thought-buys-composition).

The variable is how many facts already in the context an action's choice has to
combine. Choosing "call search" needs one. Choosing "call transfer with the
account found in step 2, the amount from the user's first message, and the
currency implied by the branch in step 4" needs three, and it is a composition.
"""
import numpy as np

rng = np.random.default_rng(1933)

N = 60000
K = 6
P_LOOKUP = 0.985         # retrieving one fact from context correctly
P_COMPOSE_NOTHOUGHT = 0.90   # combining facts inside one forward pass
P_COMPOSE_THOUGHT = 0.985    # combining them across emitted tokens
THOUGHT_TOKENS = 60      # a thought costs this many output tokens
ACTION_TOKENS = 25


def step_ok(depth, thought):
    """A step succeeds if every fact is retrieved AND they are combined
    correctly. A thought turns one deep composition into a sequence of shallow
    ones (ch:rsn-cot), which is why its benefit scales with depth."""
    look = (rng.random((N, max(depth, 1))) < P_LOOKUP).all(1)
    if depth <= 1:
        comp = np.ones(N, dtype=bool)
    elif thought:
        # Each combination is done in its own emitted step.
        comp = (rng.random((N, depth - 1)) < P_COMPOSE_THOUGHT).all(1)
    else:
        # All combinations must happen in one pass; difficulty compounds.
        comp = rng.random(N) < P_COMPOSE_NOTHOUGHT ** (depth - 1)
    return look & comp


def task(depth, thought, k=K):
    ok = np.ones(N, dtype=bool)
    for _ in range(k):
        ok &= step_ok(depth, thought)
    tok = k * (ACTION_TOKENS + (THOUGHT_TOKENS if thought else 0))
    return float(ok.mean()), float(tok)


DEPTHS = [1, 2, 3, 4, 6]

print(f"A {K}-step task. `depth` is how many facts from the context a step's")
print(f"action has to combine. Retrieving one fact is {P_LOOKUP:.1%} reliable;")
print(f"combining two inside one forward pass is {P_COMPOSE_NOTHOUGHT:.0%},")
print(f"and combining two across emitted tokens is {P_COMPOSE_THOUGHT:.1%}.")
print()
print(f"{'depth':>7}{'no thought':>24}{'with thought':>24}{'gain':>9}")
print(f"{'':>7}{'success':>12}{'tokens':>12}{'success':>12}{'tokens':>12}"
      f"{'':>9}")
print("-" * 76)

tab = {}
for d in DEPTHS:
    a, ta = task(d, False)
    b, tb = task(d, True)
    tab[d] = (a, ta, b, tb)
    print(f"{d:>7}{a:>12.1%}{ta:>12.0f}{b:>12.1%}{tb:>12.0f}{b - a:>+9.1%}")

print()
print()
print("Success per thousand output tokens -- the cost side of the same table.")
print()
print(f"{'depth':>7}{'no thought':>14}{'with thought':>15}{'better':>14}")
print("-" * 50)
eff = {}
for d in DEPTHS:
    a, ta, b, tb = tab[d]
    e = (a / (ta / 1000), b / (tb / 1000))
    eff[d] = e
    print(f"{d:>7}{e[0]:>14.2f}{e[1]:>15.2f}"
          f"{('thought' if e[1] > e[0] else 'no thought'):>14}")

print()
print()
print("A mixed task: most steps are shallow, a few are deep. Sweep the share of")
print("deep steps, and compare always-think against think-only-when-deep.")
print()
print(f"{'deep share':>12}{'never think':>14}{'always think':>15}"
      f"{'think when deep':>18}")
print("-" * 59)
mix = {}
for share in (0.0, 0.10, 0.25, 0.50, 1.0):
    deep = rng.random((N, K)) < share
    never = np.ones(N, dtype=bool)
    always = np.ones(N, dtype=bool)
    sel = np.ones(N, dtype=bool)
    tok_n = tok_a = tok_s = 0.0
    for j in range(K):
        d_deep = 4
        sn = np.where(deep[:, j], step_ok(d_deep, False), step_ok(1, False))
        sa = np.where(deep[:, j], step_ok(d_deep, True), step_ok(1, True))
        ss = np.where(deep[:, j], step_ok(d_deep, True), step_ok(1, False))
        never &= sn
        always &= sa
        sel &= ss
    tok_n = K * ACTION_TOKENS
    tok_a = K * (ACTION_TOKENS + THOUGHT_TOKENS)
    tok_s = K * ACTION_TOKENS + share * K * THOUGHT_TOKENS
    mix[share] = (float(never.mean()), float(always.mean()), float(sel.mean()),
                  tok_n, tok_a, tok_s)
    print(f"{share:>12.0%}{mix[share][0]:>14.1%}{mix[share][1]:>15.1%}"
          f"{mix[share][2]:>18.1%}")

print()
print()
print("And what that costs, at the same three policies.")
print()
print(f"{'deep share':>12}{'never':>10}{'always':>10}{'selective':>12}"
      f"{'selective saves':>18}")
print("-" * 62)
for share in (0.0, 0.10, 0.25, 0.50, 1.0):
    m = mix[share]
    print(f"{share:>12.0%}{m[3]:>10.0f}{m[4]:>10.0f}{m[5]:>12.0f}"
          f"{(m[4] - m[5]) / m[4]:>18.0%}")

print(f"""
The first table is cite:sprague2024tocot's finding transplanted into an agent
loop, and the depth column is the whole result.

At depth {1} -- an action that needs one fact from the context -- thinking first
buys {tab[1][2] - tab[1][0]:+.1%} and costs
{tab[1][3] / tab[1][1]:.1f} times the output tokens. There is nothing to compose,
so there is nothing for the extra serial steps to do, and the tokens are pure
overhead.

At depth {6} it buys {tab[6][2] - tab[6][0]:+.1%}, taking the task from
{tab[6][0]:.1%} to {tab[6][2]:.1%}.

**The benefit of a thought is a function of composition depth and nothing else**,
which is exactly ch:rsn-cot's account: intermediate tokens buy serial steps, and
serial steps are worth something only when the computation needs them. An agent
step that selects a tool by matching a description needs no depth; one that
assembles arguments from four places in the history is a composition, and it is
where the thought pays.

The second table adds the token cost, and it flips the recommendation at the
shallow end. Per thousand output tokens, not thinking scores
{eff[1][0]:.2f} against thinking's {eff[1][1]:.2f} at depth {1} -- roughly
{eff[1][0] / eff[1][1]:.1f} times better -- and the ordering reverses by depth
{[d for d in DEPTHS if eff[d][1] > eff[d][0]][0] if [d for d in DEPTHS if eff[d][1] > eff[d][0]] else 'never'}.

So "always think before acting" is the wrong default and "never think" is also the
wrong default, which is the setup for the third table.

Real tasks are mixed: most steps are shallow and a few are not. At a
{0.25:.0%} share of deep steps, never thinking scores {mix[0.25][0]:.1%}, always
thinking scores {mix[0.25][1]:.1%}, and thinking only on the deep steps scores
{mix[0.25][2]:.1%} -- statistically the same as always thinking, at
{(mix[0.25][4] - mix[0.25][5]) / mix[0.25][4]:.0%} fewer output tokens.

**Selective thinking gets all of the benefit for a quarter of the cost**, and the
saving grows as deep steps get rarer -- which is the direction real distributions
run.

The catch, and it is the same catch as everywhere in this part: selecting requires
knowing which steps are deep, and the thing that would decide is the model, before
it has done the composition. In practice the decision is made structurally rather
than by judgement -- **a step that assembles tool arguments from history gets a
thought; a step that picks a tool from a description does not** -- and the depth
column says how much that structural rule is worth.

One boundary on all of it. This models a thought as buying reliable composition
and nothing else. It does not model the two other things a thought does in a real
agent: it becomes part of the context for later steps, which is
ch:ag-memory's subject and is sometimes the whole point of writing it; and it is
the artefact a human reads when the run goes wrong. **A thought that buys nothing
in accuracy can still be the cheapest observability you have**, and that is a
legitimate reason to emit one that this listing has no way to score.""")

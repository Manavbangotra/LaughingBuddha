# -*- coding: utf-8 -*-
# Extracted from: Chapter 173 — Tool Schemas, Discovery, and Context Budgets
# Source: src/.../ch173-schemas.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Allocating a context budget, which almost nobody does deliberately.

Three things compete for the same context, and each is added by a different part
of the system:

  schemas    tool descriptions, added by whichever servers are connected
  resources  ch:mcp-primitives' preloaded content, added by the host
  history    the conversation so far, added by simply continuing

Each helps, with diminishing returns, and each dilutes the others -- so the total
is capped and the SPLIT is a real decision (eq:context-is-a-budget).

In practice nobody makes it. Schemas take whatever the connected servers happen
to need, resources take what the host was configured to preload, and history
takes the rest. This listing asks what that costs against an allocation chosen on
purpose.
"""
import numpy as np

rng = np.random.default_rng(4241)

M = 30000
STEPS = 6
BUDGET = 24000          # tokens available for schemas + resources + history
BASE = 0.995
DILUTE = 2.0e-6         # per-token degradation of reasoning

# Each component's benefit saturates: the first tokens matter far more than the
# last. `scale` is the token count at which roughly 63% of the benefit is had.
COMPONENTS = {
    "schemas":   dict(scale=3000.0, weight=0.34, floor=0.30),
    "resources": dict(scale=6000.0, weight=0.40, floor=0.35),
    "history":   dict(scale=4000.0, weight=0.26, floor=0.55),
}


def component_quality(name, tokens):
    """Saturating benefit: floor with nothing, approaching 1 with plenty."""
    c = COMPONENTS[name]
    return c["floor"] + (1.0 - c["floor"]) * (1.0 - np.exp(-tokens / c["scale"]))


def run(alloc, m=M, steps=STEPS, budget=BUDGET, dilute=DILUTE):
    """`alloc` maps component -> fraction of budget. Success needs every
    component to do its job, against a reasoning quality that degrades with the
    total tokens present."""
    total = sum(alloc.values()) * budget
    q = 1.0
    for name, frac in alloc.items():
        q *= component_quality(name, frac * budget) ** (
            COMPONENTS[name]["weight"] * 3.0)
    p_step = BASE * (1.0 - dilute * total)
    ok = rng.random(m) < (q * p_step ** steps)
    return float(ok.mean()), total


def sweep_best(budget=BUDGET, grid=11, m=20000):
    """Search the simplex for the best split."""
    best, arg = -1.0, None
    for i in range(grid + 1):
        for j in range(grid + 1 - i):
            k = grid - i - j
            a = {"schemas": i / grid, "resources": j / grid,
                 "history": k / grid}
            v = run(a, m=m, budget=budget)[0]
            if v > best:
                best, arg = v, a
    return arg, best


print(f"{M:,} tasks. {BUDGET:,} tokens to divide between tool schemas,")
print("preloaded resources, and conversation history. Each helps with")
print("diminishing returns; all three dilute.")
print()
print(f"{'allocation':>34}{'schemas':>9}{'resources':>11}{'history':>9}"
      f"{'success':>10}")
print("-" * 73)
PLANS = [
    ("even thirds", {"schemas": 1 / 3, "resources": 1 / 3, "history": 1 / 3}),
    ("schema-heavy (many servers)", {"schemas": 0.6, "resources": 0.25,
                                     "history": 0.15}),
    ("resource-heavy (preload all)", {"schemas": 0.15, "resources": 0.65,
                                      "history": 0.20}),
    ("history-heavy (long chat)", {"schemas": 0.15, "resources": 0.15,
                                   "history": 0.70}),
]
tab = {}
for name, a in PLANS:
    r = run(a)
    tab[name] = r
    print(f"{name:>34}{a['schemas']:>9.0%}{a['resources']:>11.0%}"
          f"{a['history']:>9.0%}{r[0]:>10.1%}")

best_alloc, best_v = sweep_best()
print(f"{'best split found':>34}{best_alloc['schemas']:>9.0%}"
      f"{best_alloc['resources']:>11.0%}{best_alloc['history']:>9.0%}"
      f"{run(best_alloc)[0]:>10.1%}")

print()
print()
print("The cost of allocating by accident. Gap between the best split and each")
print("of the plausible defaults:")
print()
bv = run(best_alloc)[0]
print(f"{'allocation':>34}{'success':>10}{'gap to best':>13}")
print("-" * 57)
for name, _ in PLANS:
    print(f"{name:>34}{tab[name][0]:>10.1%}{tab[name][0] - bv:>+13.1%}")

print()
print()
print("More context is not monotonically better: the dilution term applies")
print("to every token, including the useful ones. Sweeping the total spend,")
print("each row allocated at its own best split:")
print()
print(f"{'tokens spent':>14}{'schemas':>9}{'resources':>11}{'history':>9}"
      f"{'success':>10}")
print("-" * 53)
us = {}
for B in (4000, 12000, 24000, 48000, 96000, 160000):
    a, _ = sweep_best(budget=B)
    v = run(a, budget=B)[0]
    us[B] = (a, v)
    print(f"{B:>14,}{a['schemas']:>9.0%}{a['resources']:>11.0%}"
          f"{a['history']:>9.0%}{v:>10.1%}")
peak = max(us, key=lambda b: us[b][1])

print()
print()
print("The split itself moves with the budget, which is what teams get wrong")
print("when they upgrade to a larger window and change nothing else.")
print()
print(f"{'tokens spent':>14}{'schemas':>9}{'resources':>11}{'history':>9}")
print("-" * 43)
for B in (4000, 12000, 24000, 48000):
    a = us[B][0]
    print(f"{B:>14,}{a['schemas']:>9.0%}{a['resources']:>11.0%}"
          f"{a['history']:>9.0%}")

print(f"""
The first table prices four allocations that all arise by accident, and the
spread between them is the finding.

The best split found is {best_alloc['schemas']:.0%} schemas,
{best_alloc['resources']:.0%} resources, {best_alloc['history']:.0%} history, at
{bv:.1%}. An even three-way split gives {tab['even thirds'][0]:.1%}, which is
respectable -- the surface is flat near its peak. The specific bad defaults are
not respectable: a schema-heavy host that connected many servers loses
{bv - tab['schema-heavy (many servers)'][0]:.1f} points, and a long conversation
that has crowded out everything else loses
{bv - tab['history-heavy (long chat)'][0]:.1f}.

**Nobody chooses these allocations. They are what is left over**
(eq:context-is-a-budget): schemas take what the connected servers need, resources
take what someone configured, and history takes the rest by simply continuing.
The history-heavy row is the one that arrives on its own, in every long session,
without any decision being made.

The second table is the result worth arguing with. Success peaks at
{peak:,} tokens and falls to {us[160000][1]:.1%} at {160000:,}.

**More context is not monotonically better**, because the dilution term applies to
every token including the useful ones. The benefit of each component saturates --
the first few thousand tokens of tool schema carry nearly all of the tool
capability -- while the cost does not saturate at all. Past the peak, additional
context is subtracting.

That is worth stating carefully, because it is easy to over-read. It does not say
large windows are useless; it says filling them is not free, and the filling is
usually automatic. A host that grew its context four times over and put four times
as much in it has moved right along this curve, not up.

The last table is the practical trap. At {4000:,} tokens the best split spends
{us[4000][0]['history']:.0%} on history -- nothing at all -- and at {24000:,} it
spends {us[24000][0]['history']:.0%}.

**The right split depends on the budget**, so a configuration tuned for a small
window is wrong for a large one in a specific direction: it under-weights history
and over-weights schemas. Teams that upgrade to a larger window and change nothing
else keep an allocation that was chosen for scarcity, and then conclude the larger
window did not help.

The general instruction is the one this listing was built to make concrete.
**Allocate the context deliberately, as a budget with named line items**, rather
than letting three subsystems each take what they want. The line items are
knowable, the curve is flat near its peak so precision is not required, and the
defaults that arise by accident are the ones furthest from it.""")

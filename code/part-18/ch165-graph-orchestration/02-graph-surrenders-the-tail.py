# -*- coding: utf-8 -*-
# Extracted from: Chapter 165 — Graph-Based Orchestration
# Source: src/.../ch165-graph-orchestration.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""What a graph gives up, which is the thing that motivated the agent.

The previous listing measured what a graph buys: a path set small enough to test.
This one measures what it costs, and the cost is the same quantity
ch:ag-what-is-an-agent used to justify having an agent at all -- tail coverage
(eq:graph-surrenders-the-tail).

A graph can only execute routes its author drew. A request needing a sequence
nobody anticipated fails, exactly as a router fails on a shape nobody enumerated.
So a graph is a router over paths rather than over flows, and the same arithmetic
applies with a larger enumerable set.
"""
import numpy as np

rng = np.random.default_rng(3331)

M = 60000
P_STEP = 0.93
STEPS_HEAD = 4
STEPS_TAIL = 8
P_EDGE = 0.96           # the graph picks the right edge at a branch
BRANCHES = 3            # branch points on a typical route


def run(tail_mass, shape, m=M, escape=0.0):
    """shape: 'graph' | 'loop' | 'hybrid'. `escape` is the share of tail
    requests a hybrid routes out of the graph and into a free loop."""
    is_tail = rng.random(m) < tail_mass
    steps = np.where(is_tail, STEPS_TAIL, STEPS_HEAD)
    if shape == "loop":
        ok = (rng.random((m, STEPS_TAIL)) < P_STEP)
        ok = np.array([ok[i, :steps[i]].all() for i in range(m)])
        return float(ok.mean()), float(steps.mean())
    if shape == "graph":
        # Head requests follow a drawn route; each branch may be mis-taken.
        edges_ok = (rng.random((m, BRANCHES)) < P_EDGE).all(1)
        body_ok = (rng.random((m, STEPS_HEAD)) < P_STEP).all(1)
        ok = (~is_tail) & edges_ok & body_ok
        return float(ok.mean()), float(np.where(is_tail, 1.0,
                                                STEPS_HEAD).mean())
    if shape == "hybrid":
        edges_ok = (rng.random((m, BRANCHES)) < P_EDGE).all(1)
        body_ok = (rng.random((m, STEPS_HEAD)) < P_STEP).all(1)
        graph_ok = (~is_tail) & edges_ok & body_ok
        routed = is_tail & (rng.random(m) < escape)
        loop_ok = (rng.random((m, STEPS_TAIL)) < P_STEP).all(1)
        ok = graph_ok | (routed & loop_ok)
        cost = np.where(is_tail, np.where(routed, STEPS_TAIL + 1, 1.0),
                        STEPS_HEAD)
        return float(ok.mean()), float(cost.mean())
    raise ValueError(shape)


TAILS = [0.0, 0.05, 0.15, 0.30, 0.50]

print(f"{M:,} requests. Head requests take {STEPS_HEAD} steps on a drawn route")
print(f"with {BRANCHES} branch points ({P_EDGE:.0%} per branch); tail requests")
print(f"take {STEPS_TAIL} steps and no route exists for them. Steps are")
print(f"{P_STEP:.0%} reliable.")
print()
print(f"{'tail mass':>11}{'graph':>20}{'free loop':>20}")
print(f"{'':>11}{'success':>11}{'steps':>9}{'success':>11}{'steps':>9}")
print("-" * 51)
tab = {}
for t in TAILS:
    g = run(t, "graph")
    l = run(t, "loop")
    tab[t] = (g, l)
    print(f"{t:>11.0%}{g[0]:>11.1%}{g[1]:>9.1f}{l[0]:>11.1%}{l[1]:>9.1f}")

print()
print()
print("The hybrid: a graph for the head, an escape hatch to a free loop for")
print("anything the graph cannot route.")
print()
print(f"{'tail mass':>11}{'graph':>9}{'loop':>9}{'hybrid':>10}{'steps':>9}"
      f"{'best':>10}")
print("-" * 58)
hy = {}
for t in TAILS:
    g = tab[t][0][0]
    l = tab[t][1][0]
    h = run(t, "hybrid", escape=1.0)
    hy[t] = h
    best = max([("graph", g), ("loop", l), ("hybrid", h[0])],
               key=lambda x: x[1])[0]
    print(f"{t:>11.0%}{g:>9.1%}{l:>9.1%}{h[0]:>10.1%}{h[1]:>9.1f}{best:>10}")

print()
print()
print("What the hybrid costs in testability, since the escape hatch is a free")
print("loop and therefore unenumerable. Share of RUNS that stay in the graph:")
print()
print(f"{'tail mass':>11}{'runs in graph':>16}{'runs in the loop':>19}"
      f"{'enumerable share':>19}")
print("-" * 65)
for t in TAILS:
    print(f"{t:>11.0%}{1 - t:>16.1%}{t:>19.1%}{1 - t:>19.1%}")

print()
print()
print("And how branch reliability limits the graph, independent of the tail.")
print()
print(f"{'branches':>10}" + "".join(f"{'edge ' + format(p, '.0%'):>13}"
                                    for p in (0.99, 0.96, 0.90))
      + f"{'free loop':>12}")
print("-" * 61)
br = {}
BR_SAVE = BRANCHES
for b in (1, 3, 6, 10):
    globals()["BRANCHES"] = b
    row = []
    for pe in (0.99, 0.96, 0.90):
        PE = P_EDGE
        globals()["P_EDGE"] = pe
        row.append(run(0.0, "graph")[0])
        globals()["P_EDGE"] = PE
    globals()["BRANCHES"] = BR_SAVE
    br[b] = row
    print(f"{b:>10}" + "".join(f"{v:>13.1%}" for v in row)
          + f"{tab[0.0][1][0]:>12.1%}")

print(f"""
The first table contains a result the chapter was not written to expect, and it is
in the first row rather than the last.

At {0:.0%} tail mass -- every request fits a route the author drew, which is the
graph's best case -- the graph scores {tab[0.0][0][0]:.1%} against the free loop's
{tab[0.0][1][0]:.1%}. **The graph is behind where it should be strongest.**

The reason is the branch decisions. A drawn route has branch points, and each is a
decision that can go wrong: {BRANCHES} branches at {P_EDGE:.0%} is
{P_EDGE ** BRANCHES:.1%} before any step executes. **A graph does not remove the
model's uncertainty; it relocates it from "which action" to "which edge"** -- and
an edge choice is a classification over a set the author fixed, which is not
obviously easier.

The last table isolates that. One branch at {0.99:.0%} gives {br[1][0]:.1%},
essentially matching the loop. Ten branches at {0.90:.0%} gives {br[10][2]:.1%}.
**Branch count is an exponent**, exactly as handoff count was in
ch:as-multi-agent, and a richly-branched graph pays it on every request.

Then the tail, which is the cost the chapter was written to measure. At
{0.5:.0%} tail mass the graph scores {tab[0.5][0][0]:.1%} against the loop's
{tab[0.5][1][0]:.1%}, because a request needing a sequence nobody drew has nowhere
to go (eq:graph-surrenders-the-tail). That is ch:ag-what-is-an-agent's router
result with a larger enumerable set: **a graph is a router over paths.**

The second table adds the obvious fix, and it does not rescue the design here. A
hybrid -- graph for the head, escape to a free loop for anything unroutable --
reaches {hy[0.15][0]:.1%} at {0.15:.0%} tail against the pure loop's
{tab[0.15][1][0]:.1%}. It beats the pure graph everywhere and beats the loop
nowhere, because the graph half is itself behind.

The third table is what the hybrid costs in the currency the graph was bought
with. At {0.15:.0%} tail mass, {0.15:.0%} of runs leave the graph and enter an
unenumerable loop, so the enumerable share of behaviour is {0.85:.0%}. **An escape
hatch converts a bounded path set back into an unbounded one at exactly the rate
it is used**, and the runs that use it are the unusual ones -- which
ch:ag-what-is-an-agent noted are the consequential ones.

So the honest summary of these two listings is narrower than the usual case for
graphs, and it is not about reliability.

**A graph buys testability and pays for it twice**: in branch decisions, which cost
{1 - P_EDGE ** BRANCHES:.1%} on every request at this branching factor, and in tail
coverage, which is the thing that motivated using an agent at all.

Where that trade is worth taking is where the path set is genuinely small and the
tail is genuinely thin -- a workflow with a handful of branches, which is
ch:ag-what-is-an-agent's router by another name. Where the tail is thick, the graph
is buying testability with the capability that justified the system.

And there is one thing a graph buys that neither listing measures and that is
probably its strongest justification in practice: **the control flow becomes an
artefact a human can read, review, version and diff.** That is not a reliability
property and it is worth a great deal on a system several people maintain. It
should be argued for on those terms rather than on the numbers above.""")

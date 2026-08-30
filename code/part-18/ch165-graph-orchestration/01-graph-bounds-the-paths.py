# -*- coding: utf-8 -*-
# Extracted from: Chapter 165 — Graph-Based Orchestration
# Source: src/.../ch165-graph-orchestration.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""What a graph buys, in the only currency it can pay in.

A graph orchestration declares the control flow: nodes, edges, and the conditions
on them. The claim is usually about reliability, and the mechanism is about
TESTABILITY -- ch:ag-what-is-an-agent showed a free-running loop has more execution
paths than any test suite can cover, and a graph has as many as its edges allow
(eq:graph-bounds-the-paths).

This listing counts what each shape can reach and what a fixed test budget covers,
and then asks the question that decides whether the trade is worth taking: how much
of the real task distribution does the graph's path set actually contain?
"""
import numpy as np

rng = np.random.default_rng(3251)

TESTS = 400
OUTCOMES = 3            # per-step outcomes in a free loop


def loop_paths(horizon):
    return OUTCOMES ** horizon


def graph_paths(nodes, avg_out):
    """Paths through a DAG of `nodes` with `avg_out` outgoing edges each, up to
    the longest route. This is an upper bound on distinct executions."""
    return int(avg_out ** max(1, nodes - 1))


print(f"A test suite of {TESTS} cases. A free loop has {OUTCOMES} outcomes per")
print("step; a graph has as many paths as its edges permit.")
print()
print(f"{'shape':>30}{'paths':>14}{'covered':>11}{'coverage':>11}")
print("-" * 66)
shapes = [("graph, 6 nodes, 2 edges each", graph_paths(6, 2)),
          ("graph, 10 nodes, 2 edges each", graph_paths(10, 2)),
          ("graph, 10 nodes, 3 edges each", graph_paths(10, 3)),
          ("free loop, horizon 6", loop_paths(6)),
          ("free loop, horizon 10", loop_paths(10)),
          ("free loop, horizon 16", loop_paths(16))]
cov = {}
for name, p in shapes:
    c = min(TESTS, p)
    cov[name] = (p, c / p)
    print(f"{name:>30}{p:>14,}{c:>11,}{c / p:>11.2%}")

print()
print()
print("Coverage is the wrong metric -- ch:ag-what-is-an-agent showed paths are")
print("wildly unequal. What share of RUNS does a test suite cover?")
print()


def mass_covered(paths, tests, skew):
    """Paths ranked by probability with a Zipf-like skew; how much probability
    mass do the commonest `tests` of them hold?"""
    n = min(paths, 2_000_000)
    r = np.arange(1, n + 1, dtype=float)
    w = r ** (-skew)
    w /= w.sum()
    return float(w[:min(tests, n)].sum())


print(f"{'shape':>30}{'paths':>12}{'path coverage':>16}{'mass covered':>15}")
print("-" * 73)
mc = {}
for name, p in shapes:
    m = mass_covered(p, TESTS, 1.2)
    mc[name] = m
    print(f"{name:>30}{p:>12,}{min(TESTS, p) / p:>16.2%}{m:>15.1%}")

print()
print()
print("How much mass a test suite covers, by how concentrated the runs are.")
print()
print(f"{'skew':>8}{'graph, 10 nodes':>18}{'loop, horizon 10':>19}"
      f"{'loop, horizon 16':>19}")
print("-" * 64)
sk = {}
for s in (0.6, 0.9, 1.2, 1.6, 2.2):
    row = (mass_covered(graph_paths(10, 2), TESTS, s),
           mass_covered(loop_paths(10), TESTS, s),
           mass_covered(loop_paths(16), TESTS, s))
    sk[s] = row
    print(f"{s:>8.1f}{row[0]:>18.1%}{row[1]:>19.1%}{row[2]:>19.1%}")

print()
print()
print("And how many tests each shape needs for 99% mass coverage.")
print()
print(f"{'shape':>30}{'tests for 90%':>16}{'for 99%':>11}{'for 99.9%':>12}")
print("-" * 69)
need = {}


def tests_for(paths, target, skew=1.2):
    n = min(paths, 2_000_000)
    r = np.arange(1, n + 1, dtype=float)
    w = r ** (-skew)
    w /= w.sum()
    c = np.cumsum(w)
    idx = int(np.searchsorted(c, target)) + 1
    return min(idx, n)


for name, p in shapes:
    need[name] = tuple(tests_for(p, t) for t in (0.90, 0.99, 0.999))
    print(f"{name:>30}{need[name][0]:>16,}{need[name][1]:>11,}"
          f"{need[name][2]:>12,}")

print(f"""
The first table is the claim, and it holds. A graph with {6} nodes and two edges
each has {cov['graph, 6 nodes, 2 edges each'][0]} distinct paths, and
{TESTS} tests cover all of them. A free loop at horizon {16} has
{cov['free loop, horizon 16'][0]:,}, of which the same suite covers
{cov['free loop, horizon 16'][1]:.2%}.

**A graph bounds the path set** (eq:graph-bounds-the-paths), and that is the one
thing it does that a loop cannot. Everything else about graph orchestration is a
convenience; this is a property.

The second table applies ch:ag-what-is-an-agent's correction, which is that path
coverage is the wrong statistic because paths are wildly unequal. On mass, the
free loop at horizon {10} is at {mc['free loop, horizon 10']:.1%} and the graph at
{mc['graph, 10 nodes, 2 edges each']:.1%}.

That narrows the gap considerably. The loop is not as untestable as its path count
suggests, because most runs take one of a few routes -- which was
ch:ag-what-is-an-agent's finding and it survives here.

The third table says when the gap matters, and it is entirely about how
concentrated the runs are. At skew {2.2} -- runs overwhelmingly following a few
routes -- the loop covers {sk[2.2][1]:.1%} of mass with {TESTS} tests and the graph
{sk[2.2][0]:.1%}. At skew {0.6}, where behaviour is genuinely varied, the loop
covers {sk[0.6][1]:.1%} and the graph {sk[0.6][0]:.1%}.

**A graph's testability advantage is largest exactly where an agent's behaviour is
most varied** -- which is where you wanted an agent. That tension is the chapter,
and the next listing prices the other side of it.

The last table is the practical form. Reaching {0.99:.0%} mass coverage needs
{need['graph, 10 nodes, 2 edges each'][1]:,} tests for a ten-node graph and
{need['free loop, horizon 10'][1]:,} for a horizon-10 loop; at horizon {16} it is
{need['free loop, horizon 16'][1]:,}.

Those are the numbers to put beside a testing budget. A graph makes exhaustive-ish
testing a purchasing decision; a loop makes it impossible and leaves you with
sampling and an explicit uncovered remainder.""")

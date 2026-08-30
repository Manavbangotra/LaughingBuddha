# -*- coding: utf-8 -*-
# Extracted from: Chapter 184 — Repository Understanding and Code Retrieval
# Source: src/.../ch184-repository-understanding.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Retrieving over a repository, where the thing you need is not the thing you
searched for.

ch:mcp-schemas found retrieval necessary once an inventory exceeds a few dozen
items, and a repository is an inventory of thousands. So code agents retrieve, and
they mostly retrieve by text similarity to the issue.

Text similarity finds the file the issue TALKS ABOUT. A change usually also
requires files the issue does not mention: the caller whose contract changes, the
subclass that overrides the method, the test that asserts the old behaviour. Those
are reachable from the first file by the CALL GRAPH and not by any amount of
similarity search (eq:structure-finds-what-text-cannot).

This listing measures how much of a required change set each method reaches, and
what an incomplete change set does to the patch.
"""
import numpy as np

rng = np.random.default_rng(5039)

M = 40000
REPO_FILES = 900
CONTEXT_FILES = 12          # how many files fit in the working context

# A change touches `span` files: one the issue names, and others reachable
# only structurally.
P_TEXT_FINDS = 0.82         # chance text search finds a textually-similar file
P_TEXT_FINDS_STRUCT = 0.21  # ...and a structurally-linked one it does not mention
P_GRAPH_FINDS = 0.86        # chance graph expansion reaches a linked file
FALSE_RATE = 0.30           # share of retrieved files that are irrelevant
DILUTE = 0.010              # per irrelevant file in context, cost to reasoning


def run(method, span, m=M, budget=CONTEXT_FILES, breadth=1.0):
    """Returns (complete change sets, mean files found, mean irrelevant shown,
    patch success)."""
    # File 0 is named by the issue; the rest are structurally linked.
    # Retrieving more candidates raises the chance each needed file is among
    # them, with diminishing returns, and brings more irrelevant ones.
    lift = 1.0 - (1.0 - 0.55) ** breadth
    found = np.zeros((m, span), dtype=bool)
    found[:, 0] = rng.random(m) < min(P_TEXT_FINDS * (0.75 + 0.35 * lift), 0.99)
    for k in range(1, span):
        p_text = min(P_TEXT_FINDS_STRUCT * (0.6 + 1.2 * lift), 0.95)
        p_graph = P_GRAPH_FINDS
        if method == "text":
            found[:, k] = rng.random(m) < p_text
        elif method == "graph":
            # Graph expansion works from the seed file, so it needs the seed.
            found[:, k] = found[:, 0] & (rng.random(m) < p_graph)
        elif method == "both":
            a = rng.random(m) < p_text
            b = found[:, 0] & (rng.random(m) < p_graph)
            found[:, k] = a | b
        else:
            raise ValueError(method)

    complete = found.all(1)
    n_found = found.sum(1)
    # Irrelevant files retrieved alongside, capped by the context budget.
    retrieved = 3.0 * breadth
    shown = np.minimum(np.maximum(retrieved, n_found), budget)
    irrelevant = np.maximum(shown - n_found, 0)
    # A patch succeeds if the change set is complete and the context is not
    # too diluted to reason over.
    ok = complete & (rng.random(m) < np.clip(1 - DILUTE * irrelevant, 0, 1))
    return (float(complete.mean()), float(n_found.mean()),
            float(irrelevant.mean()), float(ok.mean()))


print(f"A repository of {REPO_FILES} files. A change touches several of them: one")
print("the issue names, and others reachable only through the call graph.")
print()
print(f"{'files in the change':>21}" + "".join(f"{m:>12}" for m in
                                               ("text only", "graph only",
                                                "both")))
print("-" * 57)
tab = {}
for span in (1, 2, 3, 5, 8):
    row = tuple(run(m_, span)[0] for m_ in ("text", "graph", "both"))
    tab[span] = row
    print(f"{span:>21}" + "".join(f"{v:>12.1%}" for v in row))

print()
print()
print("The same, as patch success -- complete change set AND a context clean")
print("enough to reason over.")
print()
print(f"{'files in the change':>21}" + "".join(f"{m:>12}" for m in
                                               ("text only", "graph only",
                                                "both")))
print("-" * 57)
ps = {}
for span in (1, 2, 3, 5, 8):
    row = tuple(run(m_, span)[3] for m_ in ("text", "graph", "both"))
    ps[span] = row
    print(f"{span:>21}" + "".join(f"{v:>12.1%}" for v in row))

print()
print()
print("Why graph-only fails on single-file changes and text-only fails on")
print("multi-file ones: they find different things.")
print()
print(f"{'method':>14}{'finds the named file':>22}{'finds a linked file':>21}")
print("-" * 57)
print(f"{'text':>14}{P_TEXT_FINDS:>22.0%}{P_TEXT_FINDS_STRUCT:>21.0%}")
print(f"{'graph':>14}{'via the seed':>22}{P_GRAPH_FINDS:>21.0%}")
print(f"{'both':>14}{P_TEXT_FINDS:>22.0%}"
      f"{1 - (1 - P_TEXT_FINDS_STRUCT) * (1 - P_GRAPH_FINDS):>21.0%}")

print()
print()
print("Graph expansion depends on the seed, so improving text search improves")
print("BOTH -- which is not true in reverse.")
print()
print(f"{'text finds the seed':>21}{'text only':>12}{'both':>10}{'gain':>9}")
print("-" * 52)
sd = {}
for p in (0.50, 0.65, 0.82, 0.95):
    g = globals()
    saved = g["P_TEXT_FINDS"]
    g["P_TEXT_FINDS"] = p
    a = run("text", 3)[3]
    b = run("both", 3)[3]
    g["P_TEXT_FINDS"] = saved
    sd[p] = (a, b)
    print(f"{p:>21.0%}{a:>12.1%}{b:>10.1%}{b - a:>+9.1%}")

print()
print()
print("Retrieval breadth, which is ch:mcp-schemas' trade inside a repository:")
print("showing more candidate files raises recall and dilutes the context.")
print()
print(f"{'files retrieved':>17}{'complete sets':>15}{'irrelevant shown':>18}"
      f"{'patch success':>15}")
print("-" * 65)
cb = {}
for br in (0.7, 1.0, 2.0, 4.0, 8.0):
    r = run("both", 3, breadth=br)
    cb[round(3.0 * br)] = r
    print(f"{3.0 * br:>17.0f}{r[0]:>15.1%}{r[2]:>18.1f}{r[3]:>15.1%}")
best_breadth = max(cb, key=lambda k: cb[k][3])

print()
print()
print("And the reason a failing test is the best localiser of all: it names the")
print("files by executing them, so it needs neither similarity nor a seed.")
print()
print(f"{'localiser':>26}{'span 1':>10}{'span 3':>10}{'span 8':>10}")
print("-" * 56)
for label, m_ in (("text similarity", "text"), ("call-graph expansion", "graph"),
                  ("both", "both")):
    print(f"{label:>26}" + "".join(f"{run(m_, s)[3]:>10.1%}"
                                   for s in (1, 3, 8)))
# A stack trace names every frame it passed through, which is the change set.
trace = {}
for s in (1, 3, 8):
    found_all = rng.random(M) < 0.91 ** 1      # one localisation, not per file
    irr = 2.0
    ok = found_all & (rng.random(M) < (1 - DILUTE * irr))
    trace[s] = float(ok.mean())
print(f"{'a failing test / trace':>26}" + "".join(f"{trace[s]:>10.1%}"
                                                  for s in (1, 3, 8)))

print(f"""
The first table is the finding, and the collapse in the left column is the whole
argument.

Text similarity resolves {tab[1][0]:.1%} of single-file changes and
{tab[8][0]:.1%} of eight-file ones. Call-graph expansion resolves
{tab[8][1]:.1%} of the eight-file ones.

The mechanism is in the third table. **Text search finds the file the issue talks
about; the call graph finds the files that have to change with it**
(eq:structure-finds-what-text-cannot). A caller whose contract changed is not
textually similar to the issue -- the issue never mentions it -- so no amount of
better embedding reaches it.

That matters because cite:jimenez2023swebench's tasks explicitly require
coordinating changes across multiple functions, classes and files. **The benchmark
is hard in precisely the dimension text retrieval cannot address**, and a system
whose retrieval is similarity-only will look adequate on single-file issues and
fail on the rest.

The seed table shows the two methods are not symmetric. Graph expansion starts from
a file, so it needs text search to find that file first: improving the seed rate
from {0.50:.0%} to {0.95:.0%} takes the combined method from {sd[0.50][1]:.1%} to
{sd[0.95][1]:.1%}.

**Better text search improves structural retrieval and not the reverse.** So the
order of investment is settled: get the seed right, then expand from it. A team that
builds graph expansion on top of weak search has built the second half of a
mechanism.

The breadth table is ch:mcp-schemas' trade inside a repository. Retrieving
{cb[2][1]:.0f} files gives {cb[2][3]:.1%}; retrieving {24} gives {cb[24][3]:.1%},
with irrelevant files in context rising from {cb[2][2]:.1f} to {cb[24][2]:.1f}.

Recall rises and dilution offsets it, so the curve flattens rather than turning
over at these parameters -- but note the shape: **most of the benefit is reached by
about six files**, and everything past that is paying dilution for recall that is
nearly exhausted. That is the same early-saturation result ch:mcp-schemas found for
tool schemas.

The last table is the one to act on. A failing test or a stack trace localises at
{trace[1]:.1%} for a one-file change and {trace[8]:.1%} for an eight-file one --
**flat in the span**, where every other method degrades.

The reason is structural rather than a matter of degree. A trace names the files by
having executed them; it does not need the issue to resemble the code, and it does
not need a seed to expand from. It identifies the change set directly.

Which gives this chapter's practical ordering, and it is not the usual one.

**Reproduce the failure first.** A failing test is worth more than any retrieval
system, and building one is often the largest single improvement available to a
code agent.

**Then get the seed right**, because structural expansion is downstream of it.

**Then expand structurally**, not by retrieving more text.

**And retrieve narrowly** -- around six files -- because recall saturates early and
dilution does not.""")

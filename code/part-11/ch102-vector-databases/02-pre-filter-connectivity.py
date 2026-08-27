# -*- coding: utf-8 -*-
# Extracted from: Chapter 102 — Vector Databases and Index Structures
# Source: src/.../ch102-vector-databases.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Why pre-filtering breaks a graph index, and at exactly which selectivity.

A graph index answers queries by walking from node to neighbour. Pre-filtering
means the walk may only step on nodes that satisfy the predicate -- which is
site percolation on the graph, and percolation has a threshold.

We build a k-NN graph, retain each node independently with probability s, and
measure the largest connected component of the retained subgraph. Everything
outside that component is unreachable by a walk that starts inside it.
"""
import numpy as np
from collections import deque

rng = np.random.default_rng(41)

N, DIM, M = 6000, 24, 16
TRIALS = 4

X = rng.normal(size=(N, DIM))
X /= np.linalg.norm(X, axis=1, keepdims=True)
sims = X @ X.T
np.fill_diagonal(sims, -np.inf)

# A k-NN graph, symmetrised -- which is what a graph index maintains.
knn = np.argpartition(-sims, M, axis=1)[:, :M]
neighbours = [set() for _ in range(N)]
for i in range(N):
    for j in knn[i]:
        neighbours[i].add(int(j))
        neighbours[int(j)].add(i)

mean_degree = float(np.mean([len(s) for s in neighbours]))
predicted_sc = 1.0 / mean_degree


def components(mask):
    """Largest component as a fraction of retained nodes, and component count."""
    remaining = set(np.flatnonzero(mask).tolist())
    sizes = []
    while remaining:
        start = remaining.pop()
        seen, queue = {start}, deque([start])
        while queue:
            u = queue.popleft()
            for v in neighbours[u]:
                if v in remaining:
                    remaining.discard(v)
                    seen.add(v)
                    queue.append(v)
        sizes.append(len(seen))
    return max(sizes) / int(mask.sum()), len(sizes)


print(f"mean degree after symmetrisation: {mean_degree:.1f}")
print(f"predicted percolation threshold s_c = 1/degree = {predicted_sc:.3f}"
      f"   (eq:percolation-threshold)\n")
print(f"{'selectivity':>12}{'retained':>10}{'largest component':>19}"
      f"{'components':>12}{'reachable':>11}")
print("-" * 66)

for sel in [1.0, 0.5, 0.3, 0.2, 0.15, 0.10, 0.05, 0.02]:
    fracs, counts, kept = [], [], []
    for _ in range(TRIALS):
        mask = np.ones(N, bool) if sel == 1.0 else rng.random(N) < sel
        f, c = components(mask)
        fracs.append(f)
        counts.append(c)
        kept.append(int(mask.sum()))
    frac = float(np.mean(fracs))
    flag = "ok" if frac > 0.9 else ("degraded" if frac > 0.3 else "SHATTERED")
    print(f"{sel:>12.2f}{int(np.mean(kept)):>10d}{frac:>19.3f}"
          f"{np.mean(counts):>12.1f}{flag:>11}")

print(f"""
The largest-component column IS the recall ceiling for a filtered graph walk. A
greedy search enters at one node and can only reach that node's component, so
whatever fraction of the retained set lies outside it is unreachable -- no matter
how large ef is, and with no error reported.

The collapse is not gradual. Down to selectivity 0.3 the retained subgraph is
essentially intact. By 0.05 the largest component holds under 6% of retained
nodes and the subgraph has shattered into more than a hundred pieces. The
transition sits right around the predicted threshold of {predicted_sc:.3f}, which
is eq:percolation-threshold doing real work: the mean degree is a parameter you
CHOSE at build time, so this number is knowable before deployment.

Two engineering readings follow. First, raising M lowers s_c proportionally and
costs about 1% of index memory per unit (eq:index-memory) -- an unusually cheap
way to buy filter robustness. Second, and more important: the honest fix is to
traverse THROUGH non-matching nodes and filter only the returned set, which
preserves connectivity exactly and pays in a larger ef. Restricting the walk
itself is the version that fails silently.""")

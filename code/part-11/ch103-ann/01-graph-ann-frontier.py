# -*- coding: utf-8 -*-
# Extracted from: Chapter 103 — Approximate Nearest Neighbors: HNSW, IVF, and Product Quantization
# Source: src/.../ch103-ann.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A proximity-graph index from scratch, and its recall/cost frontier.

Build: insert points one at a time, connecting each to the M nearest already
inserted, found by greedy search on the partial graph (eq:greedy-graph-search).
This is Malkov's NSW construction -- the long-range links that make the graph
navigable come from insertion ORDER, not from explicit design.

Search: greedy best-first with a candidate list of size ef. We report recall
against exact search and the number of distance computations, which is the
honest cost unit -- it is what the frontier of eq:recall-qps-frontier is made of.
"""
import heapq
import numpy as np

rng = np.random.default_rng(7)

N, DIM, LATENT = 12_000, 32, 12
M, EF_CONSTRUCTION, K = 12, 40, 10
N_QUERY = 150

# Data with low intrinsic dimension -- the regime where eq:contraction holds.
proj = rng.normal(size=(LATENT, DIM)) / np.sqrt(LATENT)
X = rng.normal(size=(N, LATENT)) @ proj
X /= np.linalg.norm(X, axis=1, keepdims=True)
queries = rng.normal(size=(N_QUERY, LATENT)) @ proj
queries /= np.linalg.norm(queries, axis=1, keepdims=True)

truth = [set(row.tolist())
         for row in np.argsort(-(queries @ X.T), axis=1)[:, :K]]

neighbours = [[] for _ in range(N)]


def search(v, entry, ef, counter):
    """Greedy best-first with a bounded candidate list (eq:greedy-graph-search).

    `cand` is a min-heap on distance (nearest first, to expand next).
    `best`  is a max-heap on distance (farthest first, so it can be trimmed).
    """
    d0 = -float(X[entry] @ v)
    cand, best, seen = [(d0, entry)], [(-d0, entry)], {entry}
    while cand:
        d, u = heapq.heappop(cand)
        if len(best) >= ef and d > -best[0][0]:
            break                       # nothing left that can improve the list
        for w in neighbours[u]:
            if w in seen:
                continue
            seen.add(w)
            counter[0] += 1
            dw = -float(X[w] @ v)
            if len(best) < ef or dw < -best[0][0]:
                heapq.heappush(cand, (dw, w))
                heapq.heappush(best, (-dw, w))
                if len(best) > ef:
                    heapq.heappop(best)
    return sorted((-d, i) for d, i in best)


for i in range(1, N):
    found = search(X[i], int(rng.integers(0, i)), min(EF_CONSTRUCTION, i), [0])
    chosen = [j for _, j in found[:M]]
    neighbours[i] = chosen
    for j in chosen:
        neighbours[j].append(i)
        if len(neighbours[j]) > 2 * M:          # prune to keep degree bounded
            neighbours[j] = sorted(
                neighbours[j], key=lambda w: -float(X[j] @ X[w]))[:2 * M]

mean_degree = float(np.mean([len(s) for s in neighbours]))
print(f"graph built: {N} nodes, mean degree {mean_degree:.1f}\n")
print(f"{'ef':>6}{'recall@10':>12}{'distance comps':>17}{'vs brute force':>16}"
      f"{'speedup':>10}")
print("-" * 61)

for ef in [10, 20, 40, 80, 160, 320]:
    recalls, total = [], 0
    for qi in range(N_QUERY):
        counter = [0]
        found = search(queries[qi], int(rng.integers(0, N)), ef, counter)
        got = {i for _, i in found[:K]}
        recalls.append(len(got & truth[qi]) / K)
        total += counter[0]
    comps = total / N_QUERY
    print(f"{ef:>6}{np.mean(recalls):>12.4f}{comps:>17.1f}"
          f"{comps / N:>15.2%}{N / comps:>10.1f}x")

print("""
This table IS eq:recall-qps-frontier, and it is the only honest way to report an
ANN index. A single recall number means nothing, because any index reaches any
recall by working harder; the question is always what your required recall costs.

Read the knee. Going from ef=10 to ef=40 buys a large jump in recall for roughly
double the work. Going from ef=160 to ef=320 doubles the work again and buys
nothing, because recall has already saturated. That shape is eq:ef-recall-model:
recall approaches 1 geometrically in ef while cost grows linearly, so there is
always a knee and you should stop at it.

The speedup column is the reason approximate search exists. Near-exact answers
while touching a small percentage of the corpus -- and note that this graph was
built by nothing more than inserting points in a random order and linking each to
its nearest predecessors. The long-range links that make it navigable are a free
consequence of the early insertions having few candidates to choose from.""")

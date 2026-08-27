# -*- coding: utf-8 -*-
# Extracted from: Chapter 102 — Vector Databases and Index Structures
# Source: src/.../ch102-vector-databases.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""What a metadata filter costs a post-filtered search.

Post-filtering retrieves the top B by vector similarity and discards
non-matching results. We measure recall of the TRUE filtered top-k -- the k
nearest documents that actually satisfy the predicate -- as a function of the
predicate's selectivity and the retrieval budget B.

The predicate here is independent of position, which is the favourable case.
A predicate correlated with the embedding is worse.
"""
import numpy as np

rng = np.random.default_rng(31)

N, DIM, K = 20_000, 32, 10
N_QUERY = 200
BUDGETS = [100, 500, 2000]
LADDER = [100, 200, 500, 1000, 2000, 5000, 10_000, 20_000]

X = rng.normal(size=(N, DIM))
X /= np.linalg.norm(X, axis=1, keepdims=True)
queries = rng.normal(size=(N_QUERY, DIM))
queries /= np.linalg.norm(queries, axis=1, keepdims=True)
sims = queries @ X.T                       # exact scores; the index is not the point


def recall_at_budget(sims, mask, truth, budget):
    """Retrieve top-`budget` ignoring the filter, then keep matching results."""
    budget = min(budget, sims.shape[1] - 1)
    cand = np.argpartition(-sims, budget, axis=1)[:, :budget]
    hits = []
    for i in range(len(sims)):
        kept = cand[i][mask[cand[i]]]
        hits.append(len(set(kept.tolist()) & truth[i]) / K)
    return float(np.mean(hits))


print(f"{'selectivity':>12}{'matching':>10}"
      + "".join(f"{'B=' + str(b):>10}" for b in BUDGETS)
      + f"{'B for 95%':>12}{'as % corpus':>13}")
print("-" * 79)

for sel in [0.5, 0.2, 0.05, 0.01, 0.002]:
    mask = rng.random(N) < sel
    idx = np.flatnonzero(mask)
    # Ground truth: the k nearest documents that SATISFY the predicate.
    truth_idx = idx[np.argsort(-sims[:, idx], axis=1)[:, :K]]
    truth = [set(row.tolist()) for row in truth_idx]

    row = [recall_at_budget(sims, mask, truth, b) for b in BUDGETS]

    needed = None
    for b in LADDER:
        if recall_at_budget(sims, mask, truth, b) >= 0.95:
            needed = b
            break

    print(f"{sel:>12.3f}{len(idx):>10d}"
          + "".join(f"{r:>10.3f}" for r in row)
          + f"{str(needed):>12}{100 * (needed or N) / N:>12.1f}%")

print("""
Read the B for 95% column. It is very close to k/selectivity -- which is
eq:postfilter-budget, confirmed. The budget a post-filtered query needs scales as
the RECIPROCAL of selectivity, so a filter that admits half the corpus is free
and one that admits a fifth of a percent requires scanning a quarter of it.

At that point the index has bought nothing. This is the arithmetic behind
eq:strategy-crossover: past a certain selectivity, brute-forcing the filtered set
is simply cheaper than post-filtering, and the crossover is at a higher
selectivity than most people guess.

Now note what the low-budget columns do. At selectivity 0.002 and B=100, recall
is 0.029 -- the query returned SOMETHING, ranked plausibly, and it was almost
entirely wrong. A production system with a latency-capped B does exactly this,
and it does it worst on the most selective queries, which are usually the ones a
user has narrowed deliberately.""")

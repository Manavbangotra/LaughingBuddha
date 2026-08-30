# -*- coding: utf-8 -*-
# Extracted from: Chapter 116 — Structured and Multimodal RAG: SQL, Tables, and Images
# Source: src/.../ch116-structured-multimodal.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Serialising rows and embedding them: what it can do, and what it cannot.

The reflex when structured data enters a RAG system is to make it look like the
data the system already handles -- serialise each row to a sentence, embed it,
put it in the vector index. This listing measures what that reflex costs, across
the three query shapes a table actually receives.

Two results are exact rather than empirical. A numeric predicate cannot be
answered by similarity because embeddings do not represent magnitude ORDER
(ch:rag-indexing, eq:no-order-in-embedding). An aggregate cannot be answered by
top-k at all, for the same reason ch:rag-graph's global questions cannot
(eq:aggregate-unreachable) -- except that here the gap is not approximate, and
there is no community-summary trick, because the answer requires arithmetic over
every qualifying row.
"""
import numpy as np

rng = np.random.default_rng(41)

N_ROW, DIM = 6000, 48
N_REGION = 8
N_QUERY = 500
BUDGET = 40                       # rows that fit in the context window


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


region_vec = unit(rng.normal(size=(N_REGION, DIM)))
name_vec = unit(rng.normal(size=(N_ROW, DIM)))
row_region = rng.integers(0, N_REGION, size=N_ROW)
revenue = rng.lognormal(mean=13.0, sigma=1.1, size=N_ROW)

# A serialised row: "Acme Ltd, region EMEA, revenue 1,240,000". The embedding
# carries the name strongly and the region well. The NUMBER contributes a
# direction with no ordering in it -- 1,240,000 and 980,000 are two unrelated
# token sequences as far as the encoder is concerned, which is the whole problem.
rev_token = unit(rng.normal(size=(N_ROW, DIM)))
row_vec = unit(0.75 * name_vec + 0.50 * region_vec[row_region] + 0.18 * rev_token)

THRESHOLD = np.quantile(revenue, 0.80)


def entity_lookup():
    """'Tell me about Acme Ltd.' -- the query shape vector search is built for."""
    hits = 0
    for _ in range(N_QUERY):
        i = int(rng.integers(0, N_ROW))
        q = unit(name_vec[i] + 0.3 * region_vec[row_region[i]]
                 + rng.normal(scale=0.25, size=DIM))
        top = np.argpartition(-(row_vec @ q), BUDGET)[:BUDGET]
        hits += int(i in top)
    return hits / N_QUERY


def numeric_filter():
    """'Which EMEA accounts bill more than the 80th percentile?' -- a predicate
    with an exact answer set. Report recall of that set at the same budget."""
    rec, prec = [], []
    for _ in range(N_QUERY):
        r = int(rng.integers(0, N_REGION))
        target = np.where((row_region == r) & (revenue > THRESHOLD))[0]
        if len(target) == 0:
            continue
        q = unit(region_vec[r] + 0.35 * rng.normal(size=DIM))
        top = np.argpartition(-(row_vec @ q), BUDGET)[:BUDGET]
        got = np.intersect1d(top, target)
        rec.append(len(got) / len(target))
        prec.append(len(got) / BUDGET)
    return float(np.mean(rec)), float(np.mean(prec))


def aggregate(budget):
    """'What is total EMEA revenue?' -- the model can only add up what it sees
    (eq:aggregate-error)."""
    err = []
    for _ in range(N_QUERY):
        r = int(rng.integers(0, N_REGION))
        target = np.where(row_region == r)[0]
        truth = revenue[target].sum()
        q = unit(region_vec[r] + 0.35 * rng.normal(size=DIM))
        top = np.argpartition(-(row_vec @ q), budget)[:budget]
        seen = top[row_region[top] == r]
        err.append(abs(revenue[seen].sum() - truth) / truth)
    return float(np.mean(err))


print(f"{N_ROW:,} rows across {N_REGION} regions; {BUDGET} rows fit in context\n")

print(f"entity lookup   -- recall@{BUDGET}: {entity_lookup():.3f}")
rec, prec = numeric_filter()
print(f"numeric filter  -- recall@{BUDGET}: {rec:.3f}   precision: {prec:.3f}")
print(f"                   (SQL: recall 1.000, precision 1.000)\n")

print(f"{'aggregate: rows retrieved':>26}{'mean relative error':>22}")
print("-" * 48)
for b in (40, 100, 400, 1000, 3000):
    print(f"{b:>26}{aggregate(b):>22.3f}")

print(f"""
The first line is the case the architecture was designed for, and it works.
Looking up a named entity is a similarity problem, the name dominates the
serialised row's embedding, and recall is high. If every question about your
structured data has this shape, serialise the rows and stop reading.

The second line is where it stops working, and the reason is exact rather than
statistical. "Revenue above the 80th percentile" is a predicate over a NUMBER,
and eq:no-order-in-embedding says an embedding does not represent magnitude
order -- 1,240,000 and 980,000 are two unrelated token sequences to the encoder.
So retrieval can find the region and then returns rows drawn essentially at
random with respect to the predicate. SQL answers the same question at recall
1.000 and precision 1.000, by evaluating the predicate, which is what predicates
are for.

The third block is the one to sit with. An aggregate is not a hard retrieval
problem, it is not a retrieval problem: SUM over a set is a function of EVERY
member of that set, so no top-k contains the answer (eq:aggregate-unreachable).
Watch the error fall as the budget grows and notice how it falls -- it only
reaches zero when the budget reaches the size of the whole group. That is not a
retrieval system converging. That is a retrieval system slowly turning into a
full table scan, at LLM prices, to compute something a database does in a
millisecond.

Note what this rules out. ch:rag-graph answered its global questions with
pre-computed community summaries, and there is no equivalent here: you cannot
pre-summarise "total revenue by region" without knowing which aggregate will be
asked, and there are combinatorially many (eq:aggregate-combinatorics). The
answer is not a better index. The answer is to stop indexing the data and start
generating queries against it.""")

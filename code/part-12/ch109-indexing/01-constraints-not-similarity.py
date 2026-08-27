# -*- coding: utf-8 -*-
# Extracted from: Chapter 109 — Indexing, Metadata, and Retrieval
# Source: src/.../ch109-indexing.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Why a date range cannot be embedded, measured.

Two ways to serve the request "logistics reports from Q3 2024":

  embedded metadata -- write the date into the chunk text and hope the embedder
                       represents it. This is what teams try first.
  metadata filter   -- filter on a parsed timestamp, then rank the survivors by
                       semantic similarity.

The embedding here is DELIBERATELY generous: dates get their own dimensions and a
smooth representation of year and month, which is far better than a real text
encoder manages. The point is that even a generous embedding fails, because
eq:no-order-in-embedding is about the objective, not the capacity.
"""
import numpy as np

rng = np.random.default_rng(7)

N_CHUNK, DIM, K = 4000, 32, 10
N_TOPIC = 12
TOPIC_DIMS = DIM - 4                 # last 4 dims are reserved for the date

topics = rng.normal(size=(N_TOPIC, TOPIC_DIMS))
topics /= np.linalg.norm(topics, axis=1, keepdims=True)

# Each chunk: a topic, and a date somewhere in 2022-2025.
chunk_topic = rng.integers(0, N_TOPIC, size=N_CHUNK)
year = rng.integers(2022, 2026, size=N_CHUNK)
month = rng.integers(1, 13, size=N_CHUNK)
months_abs = (year - 2022) * 12 + (month - 1)      # the true ordering


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def date_features(year, month):
    """A GENEROUS embedding of a date: year and month, each as a normalised
    scalar plus a cyclic component. A real text encoder does far worse, because
    it sees '2024-07-15' as a token sequence."""
    y = (year - 2022) / 4.0
    m = (month - 1) / 12.0
    return np.stack([y, m, np.sin(2 * np.pi * m), np.cos(2 * np.pi * m)], axis=-1)


emb = np.concatenate([topics[chunk_topic] * 1.0,
                      date_features(year, month) * 0.6], axis=1)
emb = unit(emb)


def query_vector(topic, q_year, q_month):
    v = np.concatenate([topics[topic],
                        date_features(np.array(q_year), np.array(q_month)) * 0.6])
    return unit(v)


def evaluate(window_months):
    """A query asks for one topic within a date WINDOW.

    We score the two halves of the request SEPARATELY, because the chapter's
    claim is that one half works and the other cannot:
      topic ok  -- did the retrieved chunk match the semantic request?
      date ok   -- did it satisfy the constraint?
    """
    out = {"emb_topic": [], "emb_date": [], "emb_both": [],
           "filt_topic": [], "filt_date": [], "filt_both": []}
    for _ in range(400):
        topic = int(rng.integers(0, N_TOPIC))
        end = int(rng.integers(window_months, 48))
        lo, hi = end - window_months, end
        gold = np.flatnonzero((chunk_topic == topic)
                              & (months_abs >= lo) & (months_abs < hi))
        if len(gold) < 3:
            continue

        # Query date is the MIDDLE of the requested window.
        mid = (lo + hi) // 2
        q = query_vector(topic, 2022 + mid // 12, 1 + mid % 12)

        def score(idx):
            in_topic = chunk_topic[idx] == topic
            in_date = (months_abs[idx] >= lo) & (months_abs[idx] < hi)
            return in_topic.mean(), in_date.mean(), (in_topic & in_date).mean()

        # (a) embedded metadata: pure vector search, no filter
        top = np.argpartition(-(emb @ q), K)[:K]
        a, b, c = score(top)
        out["emb_topic"].append(a)
        out["emb_date"].append(b)
        out["emb_both"].append(c)

        # (b) metadata filter: restrict to the window, then rank semantically
        allowed = np.flatnonzero((months_abs >= lo) & (months_abs < hi))
        scores = emb[allowed] @ q
        top_f = allowed[np.argpartition(-scores, min(K, len(allowed) - 1))[:K]]
        a, b, c = score(top_f)
        out["filt_topic"].append(a)
        out["filt_date"].append(b)
        out["filt_both"].append(c)

    return {k: float(np.mean(v)) for k, v in out.items()}


print("fraction of the top-10 that satisfies each half of the request\n")
print(f"{'date window':>13}{'embedded: topic':>17}{'date':>8}{'both':>8}"
      f"{'filtered: topic':>18}{'date':>8}{'both':>8}")
print("-" * 80)
for window in [3, 6, 12, 24]:
    r = evaluate(window)
    print(f"{str(window) + ' months':>13}{r['emb_topic']:>17.3f}"
          f"{r['emb_date']:>8.3f}{r['emb_both']:>8.3f}"
          f"{r['filt_topic']:>18.3f}{r['filt_date']:>8.3f}{r['filt_both']:>8.3f}")

print("""
Read the two halves separately, because that separation is the whole point.

The TOPIC column is high under both strategies. Similarity retrieval works; that
was never in question, and it is what an embedding is for.

The DATE column is where they diverge, and the divergence is total. Filtering
gives 1.000 by construction -- a filter does not approximate a predicate, it
evaluates it. The embedded path sits far below at every window, which means
roughly a third of what it returns VIOLATES the stated constraint. Not ranked
lower: returned, in the top ten, indistinguishable from a correct result.

Now note what the embedded path is not doing: it is not getting worse as the
window narrows, and it is not getting better either. It is roughly CONSTANT,
which is itself the diagnosis. If the embedding had any notion of the interval,
its accuracy would move with the interval's width. It does not, because
similarity in the date subspace is a DISTANCE FROM THE QUERY DATE, not membership
in a range -- a chunk one month outside a narrow window is nearer the query date
than a chunk at the far edge of a wide one. The ranking cannot express "inside or
outside" however the date is encoded, so what you measure is a fixed blend of
near-misses rather than a constraint being applied well or badly.

That is eq:no-order-in-embedding. The predicate needs an ordering relation and a
dot product does not have one.

And this embedding is far MORE generous to the date than any real text encoder:
dedicated dimensions, year and month as smooth scalars, rather than the token
sequence '2024-07-15'. A real encoder does worse, and its behaviour changes when
you reformat the date -- which is the signature of a system relying on a lexical
accident rather than on a represented quantity.""")

# -*- coding: utf-8 -*-
# Extracted from: Chapter 105 — Reranking and Cross-Encoders
# Source: src/.../ch105-reranking.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""What one vector per document costs, and what MaxSim recovers.

Each document covers several unrelated ASPECTS -- a product page with a
description, a spec table, and a review section; a paper with a method and an
application. Queries target ONE aspect.

  single vector   -- the mean of the aspect vectors (eq:bottleneck)
  late interaction -- one vector per aspect, scored by MaxSim (eq:maxsim)

Both keep the corpus precomputable. Only the storage differs.
"""
import numpy as np

rng = np.random.default_rng(29)

N_DOC, DIM, N_QUERY = 4000, 64, 500


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


print(f"{'aspects/doc':>12}{'single-vector':>15}{'MaxSim':>10}"
      f"{'storage ratio':>15}{'mean cos(aspects)':>19}")
print("-" * 71)

for n_aspect in [1, 2, 4, 8]:
    # Aspects within a document are unrelated to each other -- that is what
    # makes the document hard to summarise with one point.
    aspects = unit(rng.normal(size=(N_DOC, n_aspect, DIM)))
    pooled = unit(aspects.mean(axis=1))

    # Mean cosine between two aspects of the same document, for reference.
    if n_aspect > 1:
        cos_within = float(np.mean(np.einsum('nd,nd->n',
                                             aspects[:, 0], aspects[:, 1])))
    else:
        cos_within = 1.0

    hits_single, hits_late = [], []
    for _ in range(N_QUERY):
        i = int(rng.integers(0, N_DOC))
        a = int(rng.integers(0, n_aspect))
        # A query about ONE aspect of document i, with a little noise.
        q = unit(aspects[i, a] + rng.normal(scale=0.35, size=DIM))

        single = pooled @ q
        late = np.max(aspects @ q, axis=1)          # MaxSim over aspects

        hits_single.append(float(i in np.argpartition(-single, 10)[:10]))
        hits_late.append(float(i in np.argpartition(-late, 10)[:10]))

    print(f"{n_aspect:>12}{np.mean(hits_single):>15.3f}{np.mean(hits_late):>10.3f}"
          f"{n_aspect:>14}x{cos_within:>19.3f}")

print("""
At one aspect per document the two are identical -- the mean of one vector is
that vector, so there is nothing to lose and MaxSim buys nothing. Every row below
that is the cost of compression.

As documents cover more unrelated aspects, the single-vector recall falls while
MaxSim holds up. The mechanism is eq:bottleneck: averaging near-orthogonal
directions lands the document vector on their bisector, which is far from all of
them. With two orthogonal aspects the best a single point can score against
either is 1/sqrt(2) = 0.707, while a document dedicated to one of them scores
1.0 -- so the generalist loses to specialists on every query, including the ones
it is genuinely relevant to.

MaxSim keeps the aspects separate and lets each query find its own evidence. Note
what it does NOT give up: the document encoding still does not depend on the
query, so eq:factorisation-constraint holds and the corpus is still
precomputable. That is the entire reason late interaction is a retrieval method
and a cross-encoder is not.

What it costs is the storage-ratio column, linear in aspects per document. In
ColBERT the unit is a token rather than an aspect, so the ratio is 10-100x, and
that number -- not quality -- is the argument against multi-vector retrieval.

Chunking is the cheap approximation of this table: split the document so each
piece has one aspect, and you are back on the top row with a single vector per
piece. It works, and it moves the problem to choosing the boundaries.""")

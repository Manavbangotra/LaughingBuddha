# -*- coding: utf-8 -*-
# Extracted from: Chapter 100 — Similarity Measures and the Geometry of Embedding Space
# Source: src/.../ch100-similarity.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Cosine, inner product, and Euclidean: identical, then wildly different.

Two corpora over the same 2,000 directions. In the first, every document has
norm 1. In the second, norms are lognormal -- a realistic spread for
un-normalised encoder outputs, where norm tracks length and token frequency.

We rank all documents for 300 queries under all three scorers and measure
agreement, both at the top (which document wins) and over the full ranking
(Kendall tau).
"""
import numpy as np
from scipy.stats import kendalltau, spearmanr

rng = np.random.default_rng(5)
N_Q, N_D, DIM = 300, 2000, 64

directions = rng.normal(size=(N_D, DIM))
directions /= np.linalg.norm(directions, axis=1, keepdims=True)
norms = rng.lognormal(mean=0.0, sigma=0.6, size=(N_D, 1))

queries = rng.normal(size=(N_Q, DIM))
queries /= np.linalg.norm(queries, axis=1, keepdims=True)


def score_all(Q, D):
    """Return the three score matrices of eq:three-scorers, queries x documents."""
    ip = Q @ D.T
    cos = ip / (np.linalg.norm(Q, axis=1, keepdims=True)
                * np.linalg.norm(D, axis=1)[None, :])
    # -||q-d||^2 via eq:l2-ip-identity; negated so that larger is better.
    neg_l2 = -(np.sum(Q ** 2, axis=1)[:, None]
               + np.sum(D ** 2, axis=1)[None, :] - 2.0 * ip)
    return ip, cos, neg_l2


def top1_agreement(a, b):
    return float(np.mean(np.argmax(a, axis=1) == np.argmax(b, axis=1)))


def mean_tau(a, b, n=40):
    """Kendall tau over the FULL ranking, averaged over n queries."""
    return float(np.mean([kendalltau(a[i], b[i]).statistic for i in range(n)]))


for label, docs in [("documents normalised (all norms = 1)", directions),
                    ("documents NOT normalised (lognormal norms)",
                     directions * norms)]:
    ip, cos, neg_l2 = score_all(queries, docs)
    doc_norm = np.linalg.norm(docs, axis=1)

    print(f"\n{label}")
    print(f"  {'pair':<22}{'top-1 agree':>13}{'Kendall tau':>14}")
    for name, a, b in [("inner prod vs cosine", ip, cos),
                       ("inner prod vs -L2", ip, neg_l2),
                       ("cosine vs -L2", cos, neg_l2)]:
        print(f"  {name:<22}{top1_agreement(a, b):>13.4f}{mean_tau(a, b):>+14.4f}")

    # Who does the inner product actually return?
    winners = np.argmax(ip, axis=1)
    times_returned = np.bincount(winners, minlength=N_D)
    rho = spearmanr(doc_norm, times_returned).statistic
    print(f"  spearman(||d||, times returned by inner product) = {rho:+.4f}")
    print(f"  mean ||d|| of inner-product winners = {doc_norm[winners].mean():.3f}"
          f"   (corpus mean {doc_norm.mean():.3f})")

print("""
The first block is eq:rank-equivalence. Not approximately equal -- Kendall tau is
exactly +1.0000 and top-1 agreement exactly 1.0000, because with every norm equal
all three scorers are strictly monotone functions of the same dot product.

The second block is what happens when that assumption is dropped, and it is
worse than "somewhat different". Inner product and negative-L2 pick the same best
document essentially NEVER. That is not noise; it is eq:magnitude-bias. Inner
product rewards long vectors and L2 penalises them, so on a corpus whose only
difference is norm, the two scorers rank in nearly opposite directions.

Note the Kendall tau of inner product against cosine stays high while its top-1
agreement collapses. The bulk ordering is broadly preserved and the TOP of the
ranking is not -- which is the only part anyone retrieves. Rank correlation over
a full list is the wrong diagnostic for a retrieval system.

The last two lines are the practical damage. Inner-product winners have a mean
norm several times the corpus mean: the index has quietly become a popularity
ranker over whatever property drove the norm -- in a real encoder, document
length and token frequency.""")

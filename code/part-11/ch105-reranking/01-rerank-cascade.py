# -*- coding: utf-8 -*-
# Extracted from: Chapter 105 — Reranking and Cross-Encoders
# Source: src/.../ch105-reranking.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The rerank cascade, its cost, and its ceiling.

Ground truth is a graded relevance label per document. Two scorers see it
through different amounts of noise:

  bi-encoder     -- cheap, noisy; scores the whole corpus
  cross-encoder  -- expensive, precise; scores only what it is given

We sweep the rerank depth k and report three things: the first stage's recall@k
of the true top-10 (which is eq:recall-ceiling), the nDCG@10 a PERFECT reranker
would reach on that candidate set (the hard ceiling), and the nDCG@10 the actual
cross-encoder reaches.
"""
import numpy as np

rng = np.random.default_rng(13)

N_DOC, N_QUERY, N_REL = 20_000, 300, 20
BI_NOISE, CROSS_NOISE, LABEL_NOISE = 1.0, 0.45, 0.25
DEPTHS = [10, 20, 50, 100, 200, 500, 1000]


def ndcg(order, rel, k=10):
    gains = rel[order[:k]]
    discount = np.log2(np.arange(2, k + 2))
    dcg = np.sum((2 ** gains - 1) / discount)
    ideal = np.sort(rel)[::-1][:k]
    idcg = np.sum((2 ** ideal - 1) / discount)
    return dcg / idcg if idcg > 0 else 0.0


def make_query():
    """Graded relevance labels, plus the two noisy views of them."""
    rel = np.zeros(N_DOC)
    rel[rng.choice(N_DOC, N_REL, replace=False)] = rng.integers(1, 4, N_REL)
    latent = rel + rng.normal(scale=LABEL_NOISE, size=N_DOC)
    return (rel,
            latent + rng.normal(scale=BI_NOISE, size=N_DOC),
            latent + rng.normal(scale=CROSS_NOISE, size=N_DOC))


print(f"{'k':>6}{'recall@k of top-10':>21}{'nDCG@10 oracle':>17}"
      f"{'nDCG@10 actual':>17}{'CE calls':>10}")
print("-" * 71)

table = {}
for k in DEPTHS:
    recalls, oracle, actual = [], [], []
    for _ in range(N_QUERY):
        rel, bi, cross = make_query()
        cand = np.argpartition(-bi, k)[:k]
        best10 = set(np.argsort(-rel)[:10].tolist())
        recalls.append(len(best10 & set(cand.tolist())) / 10)
        oracle.append(ndcg(cand[np.argsort(-rel[cand])], rel))    # perfect rerank
        actual.append(ndcg(cand[np.argsort(-cross[cand])], rel))  # real rerank
    table[k] = (np.mean(recalls), np.mean(oracle), np.mean(actual))
    print(f"{k:>6}{table[k][0]:>21.4f}{table[k][1]:>17.4f}"
          f"{table[k][2]:>17.4f}{k:>10}")

bi_only, cross_all = [], []
for _ in range(N_QUERY):
    rel, bi, cross = make_query()
    bi_only.append(ndcg(np.argsort(-bi), rel))
    cross_all.append(ndcg(np.argsort(-cross), rel))
bi_only, cross_all = float(np.mean(bi_only)), float(np.mean(cross_all))

print(f"\n{'bi-encoder alone':<24}nDCG@10 {bi_only:.4f}   (0 CE calls)")
print(f"{'cross-encoder on ALL':<24}nDCG@10 {cross_all:.4f}   ({N_DOC} CE calls)")

r100, o100, a100 = table[100]
print(f"""
Start with the two baselines. The bi-encoder alone reaches {bi_only:.4f}. Running
the cross-encoder over the entire corpus reaches {cross_all:.4f} -- and costs
{N_DOC} forward passes PER QUERY, which is why nobody does it. The whole cascade
exists to capture as much of that gap as {N_DOC // 100}x less compute can buy.

At k=100 it captures a lot: nDCG@10 goes from {bi_only:.4f} to {a100:.4f} for one
hundred cross-encoder calls. That is the single largest quality improvement
available to most retrieval systems, and it requires training nothing.

Now the two ceiling columns, which are the point of the experiment. At k=100 the
first stage returns only {r100:.1%} of the true top-10, so a PERFECT reranker
could reach no more than {o100:.4f} on that candidate set -- and the real
cross-encoder reaches {a100:.4f}, within {100 * (o100 - a100) / o100:.0f}% of it.

Read that as a diagnosis, because it is the one eq:two-stage-decomposition
prescribes. The reranker is nearly saturated against its candidate set. Improving
it can buy at most the gap to the oracle column; improving the FIRST STAGE, or
simply raising k, moves the oracle column itself. A team optimising the reranker
here would be working on the smaller half of the problem, and would have no way
of knowing without this table.""")

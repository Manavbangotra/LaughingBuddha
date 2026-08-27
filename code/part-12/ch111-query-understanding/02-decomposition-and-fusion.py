# -*- coding: utf-8 -*-
# Extracted from: Chapter 111 — Query Understanding: Rewriting, Expansion, and Multi-Query
# Source: src/.../ch111-query-understanding.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Multi-query paraphrase against genuine decomposition.

Two ways to turn one question into several retrievals, with very different
justifications:

  paraphrase    -- n rephrasings of the same question, results fused. Hedges
                   against one phrasing missing. eq:multi-query-recall says the
                   gain is governed by how much the variants OVERLAP.
  decomposition -- split a question whose answer lives in several documents.
                   eq:multi-hop-containment says single retrieval cannot succeed
                   at all, so this is structural rather than a recall tweak.

We measure both on multi-hop questions and report the cost alongside.
"""
import numpy as np

rng = np.random.default_rng(13)

N_DOC, N_QUERY, K, DIM = 6000, 600, 10, 48
HOPS = [1, 2, 3, 4]


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


docs = unit(rng.normal(size=(N_DOC, DIM)))


def retrieve(key, k=K):
    return set(np.argpartition(-(docs @ key), k)[:k].tolist())


def run(n_hops, paraphrases):
    """A question whose answer needs facts from `n_hops` distinct documents."""
    single_ok = para_ok = decomp_ok = 0
    for _ in range(N_QUERY):
        targets = rng.choice(N_DOC, n_hops, replace=False)

        # The single query is a blend of all the sub-topics -- which is what a
        # user's one sentence actually is, and it resembles none of them well.
        blended = unit(docs[targets].mean(axis=0)
                       + rng.normal(scale=0.35, size=DIM))
        got = retrieve(blended)
        single_ok += int(all(t in got for t in targets))

        # Paraphrases: the SAME blended query, perturbed. Fused by union.
        union = set()
        for _ in range(paraphrases):
            v = unit(blended + rng.normal(scale=0.12, size=DIM))
            union |= retrieve(v)
        para_ok += int(all(t in union for t in targets))

        # Decomposition: one query per sub-question, each aimed at its own fact.
        # A sub-question is SPECIFIC, so it retrieves its own target far more
        # reliably than the blended query retrieves any of them.
        union_d = set()
        for t in targets:
            v = unit(docs[t] + rng.normal(scale=0.18, size=DIM))
            union_d |= retrieve(v)
        decomp_ok += int(all(t in union_d for t in targets))

    return single_ok / N_QUERY, para_ok / N_QUERY, decomp_ok / N_QUERY


PARAPHRASES = 4
print(f"all {PARAPHRASES} paraphrases and all sub-questions retrieve k={K}\n")
print(f"{'hops':>6}{'single query':>15}{f'{PARAPHRASES} paraphrases':>17}"
      f"{'decomposed':>13}{'retrievals: 1 /':>18}{PARAPHRASES:>3} /  hops")
print("-" * 76)
for h in HOPS:
    s, p, d = run(h, PARAPHRASES)
    print(f"{h:>6}{s:>15.3f}{p:>17.3f}{d:>13.3f}"
          f"{'':>18}{'':>3}    {h}")

print(f"""
Read the single-query column down the hops. It collapses, and it collapses for
the reason eq:multi-hop-containment gives: the answer is not in any one document,
so there is nothing for a single retrieval to find. This is NOT a recall problem
that a larger k would fix -- the required documents are individually unremarkable
and the blended query resembles none of them.

Now compare the paraphrase column against the single-query column. Four
retrievals, four times the cost, and the gain is small. eq:multi-query-recall
explains it: paraphrases of one question overlap almost completely, so the
effective number of independent queries stays near one. FUSION NEEDS DIVERSITY
AND PARAPHRASE IS NOT DIVERSITY -- which is ch:emb-hybrid's complementarity
result arriving in a new setting.

The decomposed column is the one that works, and note that at h hops it costs h
retrievals -- the SAME order of cost as the paraphrases that bought nothing. The
difference is not how many queries were issued but whether they were asking
DIFFERENT THINGS.

Note also what the decomposed column does NOT show. It declines only slightly
across hops, because each sub-question here retrieves its own target with recall
near 0.99, and 0.99^4 is still 0.96. eq:decomposition-success is a product, so
that gentle slope is an artefact of an unrealistically reliable retriever -- at a
more typical per-hop recall of 0.85, four hops would be 0.52 and eight would be
0.27. The wall is real; this listing simply sits well short of it, and
ch:rag-agentic is where a system runs into it.

So the two techniques are not variants of one idea with different budgets. One
addresses a structural gap and the other hedges a phrasing risk, and reaching for
paraphrase when the question was multi-hop is the most common way this stage is
misapplied.""")

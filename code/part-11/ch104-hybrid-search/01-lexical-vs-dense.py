# -*- coding: utf-8 -*-
# Extracted from: Chapter 104 — Sparse Retrieval, BM25, and Hybrid Search
# Source: src/.../ch104-hybrid-search.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Where each retriever wins, and what fusion does about it.

A synthetic corpus with three retrievable signals:
  * topic vocabulary  -- semantic content a dense encoder can represent
  * an identifier     -- one token unique to each document, which a fixed-width
                         embedding structurally cannot preserve
  * a category token  -- shared by about twenty documents, so it NARROWS but does
                         not identify

Three query types exercise them separately, and the point of the experiment is
that the right combiner depends on which type you actually get.
"""
from collections import Counter
import numpy as np

rng = np.random.default_rng(5)

N_DOC, N_TOPIC, VOCAB, DOC_LEN = 4000, 40, 3000, 60
K1, B, RRF_K, DEPTH, K = 1.2, 0.75, 60, 100, 10
N_QUERY = 150

# Topic-conditional word distributions: sparse and overlapping.
topic_w = np.zeros((N_TOPIC, VOCAB))
for t in range(N_TOPIC):
    idx = rng.choice(VOCAB, 120, replace=False)
    topic_w[t, idx] = rng.random(120) ** 2
topic_w /= topic_w.sum(axis=1, keepdims=True)

mixture = rng.dirichlet(np.ones(N_TOPIC) * 0.3, size=N_DOC)
identifier = VOCAB + np.arange(N_DOC)              # unique per document
N_CAT = N_DOC // 20
category = VOCAB + N_DOC + rng.integers(0, N_CAT, size=N_DOC)

docs = []
for i in range(N_DOC):
    p = mixture[i] @ topic_w
    toks = rng.choice(VOCAB, DOC_LEN, p=p).tolist()
    toks += [int(identifier[i]), int(category[i])]
    docs.append(toks)

# ---- BM25 (eq:bm25) --------------------------------------------------------
tf = [Counter(d) for d in docs]
doc_len = np.array([len(d) for d in docs], dtype=float)
avgdl = doc_len.mean()
df = Counter()
for c in tf:
    for w in c:
        df[w] += 1
idf = {w: np.log(1 + (N_DOC - n + 0.5) / (n + 0.5)) for w, n in df.items()}


def bm25(q_tokens):
    scores = np.zeros(N_DOC)
    for w in q_tokens:
        if w not in idf:
            continue
        weight = idf[w]
        for i, counts in enumerate(tf):
            f = counts.get(w, 0)
            if f:
                scores[i] += weight * f * (K1 + 1) / (
                    f + K1 * (1 - B + B * doc_len[i] / avgdl))
    return scores


# ---- Dense: document = its topic mixture; query = topic posterior of its ----
# tokens. Tokens outside the topic vocabulary contribute NOTHING, which is the
# capacity bound of section 4 made literal.
E = mixture + rng.normal(scale=0.005, size=mixture.shape)
E /= np.linalg.norm(E, axis=1, keepdims=True)


def dense(q_tokens):
    q = np.full(N_TOPIC, 1e-6)
    for w in q_tokens:
        if w < VOCAB:
            q += topic_w[:, w]
    return E @ (q / np.linalg.norm(q))


def rrf(*ranked):
    """Fuse the top-DEPTH of each list (eq:rrf). A document outside a
    retriever's top-DEPTH receives nothing from it -- as deployed."""
    s = np.zeros(N_DOC)
    for scores in ranked:
        for rank, d in enumerate(np.argsort(-scores)[:DEPTH]):
            s[d] += 1.0 / (RRF_K + rank + 1)
    return s


def interleave(*ranked):
    """Take alternate results from each list, preserving each one's top hit."""
    lists = [np.argsort(-s)[:DEPTH] for s in ranked]
    out, seen = [], set()
    for pos in range(DEPTH):
        for lst in lists:
            d = int(lst[pos])
            if d not in seen:
                seen.add(d)
                out.append(d)
    return out


def make_query(i, kind):
    p = mixture[i] @ topic_w
    if kind == "semantic":                       # different words, same topic
        return rng.choice(VOCAB, 20, p=p).tolist()
    if kind == "identifier":                     # contains the exact token
        return rng.choice(VOCAB, 3, p=p).tolist() + [int(identifier[i])]
    return rng.choice(VOCAB, 10, p=p).tolist() + [int(category[i])]  # partial


def evaluate(kind):
    hits = {"bm25": [], "dense": [], "rrf": [], "interleave": []}
    overlaps = []
    for _ in range(N_QUERY):
        i = int(rng.integers(0, N_DOC))
        q = make_query(i, kind)
        sb, sd = bm25(q), dense(q)

        top_b = set(np.argsort(-sb)[:K].tolist())
        top_d = set(np.argsort(-sd)[:K].tolist())
        overlaps.append(len(top_b & top_d) / K)

        hits["bm25"].append(float(i in top_b))
        hits["dense"].append(float(i in top_d))
        hits["rrf"].append(float(i in set(np.argsort(-rrf(sb, sd))[:K].tolist())))
        hits["interleave"].append(float(i in set(interleave(sb, sd)[:K])))
    out = {k: float(np.mean(v)) for k, v in hits.items()}
    # An oracle router: send each query type to whichever retriever is better on
    # it. This is the ceiling a perfect query classifier would reach.
    out["oracle"] = max(out["bm25"], out["dense"])
    return out, float(np.mean(overlaps))


results, overlaps = {}, {}
print(f"{'query type':<13}{'BM25':>8}{'dense':>8}{'RRF':>8}{'interleave':>12}"
      f"{'oracle route':>14}{'overlap':>10}")
print("-" * 73)
for kind in ["semantic", "identifier", "partial"]:
    results[kind], overlaps[kind] = evaluate(kind)
    r = results[kind]
    print(f"{kind:<13}{r['bm25']:>8.3f}{r['dense']:>8.3f}{r['rrf']:>8.3f}"
          f"{r['interleave']:>12.3f}{r['oracle']:>14.3f}{overlaps[kind]:>10.3f}")

mixed = {k: float(np.mean([results[c][k] for c in results])) for k in results["semantic"]}
print(f"{'MIXED':<13}{mixed['bm25']:>8.3f}{mixed['dense']:>8.3f}{mixed['rrf']:>8.3f}"
      f"{mixed['interleave']:>12.3f}{mixed['oracle']:>14.3f}"
      f"{np.mean(list(overlaps.values())):>10.3f}")

print(f"""
Read the three query-type rows first. They are the whole argument for keeping a
lexical index: on IDENTIFIER queries BM25 scores {results['identifier']['bm25']:.3f}
and the dense retriever {results['identifier']['dense']:.3f}, and no amount of
training fixes that -- a unique token cannot survive compression into a
fixed-width vector. On SEMANTIC queries the positions reverse just as sharply.

Now read the RRF column against the best single retriever in each row, because
this is where the usual story about hybrid search breaks. On the two rows where
one retriever DOMINATES, fusion is much worse than simply using it. That is
eq:fusion-condition: with k=60, a document at rank 40 in both lists scores
2/101 = 0.0198 and outranks a document at rank 1 in one list and absent from the
other at 1/61 = 0.0164. When the retrievers are complementary, "first in one and
absent from the other" is exactly what a CORRECT answer looks like -- and RRF
buries it.

On the PARTIAL row, where both retrievers have real but incomplete signal, RRF
beats both. That is the case fusion was designed for and it works.

The MIXED row is the honest justification for hybrid search, and note what it
actually says: RRF beats either single retriever ACROSS THE WORKLOAD while losing
to the better one on most individual query types. Fusion is insurance against
query heterogeneity, not a way to make any given query better.

Now the OVERLAP column, which predicted all of this in advance. It sits near
{np.mean(list(overlaps.values())):.2f} -- the two retrievers almost never return
the same documents -- and eq:fusion-decision says that at that value RRF is the
wrong combiner. Compare INTERLEAVE, which is one line of code with no parameters
and simply takes alternate results, preserving each retriever's top hit by
construction. It beats RRF on three of the four rows, including the mixed
workload, and it is the better default whenever overlap is low.

ORACLE ROUTE is the ceiling: send each query type to whichever retriever is
better on it. It beats every combiner, and the gap between it and interleaving is
what a query classifier would actually be worth. That is the same argument as
ch:llm-routing -- when a cheap signal predicts which system should answer,
routing beats blending -- arriving here in a new setting.""")

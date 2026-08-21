# Extracted from: Chapter 78 — Semantic Similarity and Sentence Embeddings
# Source: src/.../ch078-similarity.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Three interventions on a weak bi-encoder, evaluated on recall@k and MRR."""
import numpy as np

rng = np.random.default_rng(3)
D, N_TOPICS, N_ASPECTS, PER_CELL = 64, 25, 8, 5
N_QUERIES, K = 200, 50

# Every document has a TOPIC ("API keys") and an ASPECT ("rotating", "creating").
# A query names both, and the distractors share its topic while differing in
# aspect — which is the 'twelve pages about API keys, none about rotation'
# situation, and the one bi-encoders actually fail at.
topic_vec = rng.normal(size=(N_TOPICS, D))
topic_vec /= np.linalg.norm(topic_vec, axis=1, keepdims=True)
aspect_vec = rng.normal(size=(N_ASPECTS, D))
aspect_vec /= np.linalg.norm(aspect_vec, axis=1, keepdims=True)

meta = np.array([(t_, a) for t_ in range(N_TOPICS) for a in range(N_ASPECTS)
                 for _ in range(PER_CELL)])
N_DOCS = len(meta)
docs = (topic_vec[meta[:, 0]] + 0.9 * aspect_vec[meta[:, 1]]
        + 0.10 * rng.normal(size=(N_DOCS, D)))

gold = rng.choice(N_DOCS, N_QUERIES, replace=False)
queries = (topic_vec[meta[gold, 0]] + 0.9 * aspect_vec[meta[gold, 1]]
           + 0.20 * rng.normal(size=(N_QUERIES, D)))

# The weak encoder: a bottleneck that keeps the topic and attenuates the aspect,
# plus a shared offset direction. That is a summary which preserved the topic
# and discarded the detail — plus anisotropy on top.
shared = rng.normal(size=D)
shared /= np.linalg.norm(shared)
aspect_basis = np.linalg.qr(aspect_vec.T)[0][:, :N_ASPECTS]


def weak_encoder(X):
    aspect_part = (X @ aspect_basis) @ aspect_basis.T
    return (X - aspect_part) + 0.10 * aspect_part + 2.5 * shared


def normalise(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def evaluate(dv, qv, k=10):
    S = normalise(qv) @ normalise(dv).T
    order = np.argsort(-S, axis=1)
    ranks = np.array([np.where(order[i] == gold[i])[0][0] for i in range(N_QUERIES)])
    mean_cos = float(
        (normalise(dv) @ normalise(dv).T)[np.triu_indices(N_DOCS, 1)].mean())
    return dict(r1=float((ranks == 0).mean()), r10=float((ranks < k).mean()),
                mrr=float((1 / (ranks + 1)).mean()), cos=mean_cos), S


results = {}
dv, qv = weak_encoder(docs), weak_encoder(queries)
results["1. as shipped"], _ = evaluate(dv, qv)

# Intervention A: centre the space — equation (eq:anisotropy-fix).
mu = dv.mean(0)
results["2. + centering"], _ = evaluate(dv - mu, qv - mu)

# Intervention B: also remove the top principal component.
Xc = dv - mu
_, _, Vt = np.linalg.svd(Xc, full_matrices=False)
strip = lambda X: X - (X @ Vt[:1].T) @ Vt[:1]
results["3. + remove PC1"], S3 = evaluate(strip(Xc), strip(qv - mu))

# Intervention C: rerank the top K with a scorer that sees both texts jointly
# and can therefore use the aspect dimensions the summary compressed away.
ranks = []
for i in range(N_QUERIES):
    cand = np.argsort(-S3[i])[:K]
    joint = -np.linalg.norm(docs[cand] - queries[i], axis=1)
    reordered = cand[np.argsort(-joint)]
    pos = np.where(reordered == gold[i])[0]
    ranks.append(int(pos[0]) if len(pos) else K + 1)
ranks = np.array(ranks)
results[f"4. + rerank top-{K}"] = dict(
    r1=float((ranks == 0).mean()), r10=float((ranks < 10).mean()),
    mrr=float((1 / (ranks + 1)).mean()), cos=results["3. + remove PC1"]["cos"])

print(f"{N_DOCS} documents, {N_QUERIES} queries, {N_TOPICS} topics x "
      f"{N_ASPECTS} aspects\n")
print(f"{'stage':<22} {'recall@1':>10} {'recall@10':>11} {'MRR':>8} {'mean cos':>10}")
for name, m in results.items():
    print(f"{name:<22} {m['r1']:>10.3f} {m['r10']:>11.3f} {m['mrr']:>8.3f} "
          f"{m['cos']:>10.3f}")

retriever_recall = float(np.mean(
    [gold[i] in np.argsort(-S3[i])[:K] for i in range(N_QUERIES)]))
print(f"\nretriever recall@{K} = {retriever_recall:.3f}   "
      f"<- the ceiling from equation (eq:recall-ceiling)")
print(f"reranked recall@10 = {results[f'4. + rerank top-{K}']['r10']:.3f}, "
      f"which cannot exceed it")

print("""
Three things in this table are worth more than the headline improvement.

  * Centering fixed the geometry completely (mean cosine 0.8 -> 0.0) and did
    NOT fix the retrieval. Removing a principal component made recall@1 very
    slightly worse. The anisotropy was real, and it was not the problem.
  * Reranking moved recall@1 by an order of magnitude, because the failure was
    information the bi-encoder's summary had discarded — which is exactly what
    a joint scorer can recover and a geometric fix cannot.
  * Reranked recall@10 is capped by the retriever's recall@50. The reranker
    reorders the candidate set; it can never add to it.""")

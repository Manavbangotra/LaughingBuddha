# -*- coding: utf-8 -*-
# Extracted from: Chapter 103 — Approximate Nearest Neighbors: HNSW, IVF, and Product Quantization
# Source: src/.../ch103-ann.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Product quantization from scratch: compression against recall.

Split each vector into m subvectors and quantize each with its own 256-centroid
codebook (eq:pq-code). Storage falls from 4d bytes to m bytes.

Three scorers compared:
  ADC  -- query exact, table lookup per subspace (eq:adc)
  SDC  -- query quantized too; both sides approximated
  ADC then exact rerank of the top 100, which is what real systems do
"""
import numpy as np
from scipy.cluster.vq import kmeans2

rng = np.random.default_rng(11)

N, DIM, LATENT, K = 20_000, 64, 32, 10
N_QUERY, N_BITS, RERANK = 200, 256, 100

proj = rng.normal(size=(LATENT, DIM)) / np.sqrt(LATENT)
X = rng.normal(size=(N, LATENT)) @ proj
X /= np.linalg.norm(X, axis=1, keepdims=True)
queries = rng.normal(size=(N_QUERY, LATENT)) @ proj
queries /= np.linalg.norm(queries, axis=1, keepdims=True)
truth = [set(r.tolist()) for r in np.argsort(-(queries @ X.T), axis=1)[:, :K]]


def recall(scores, rerank=0):
    hits = []
    for i in range(N_QUERY):
        if rerank:
            cand = np.argpartition(-scores[i], rerank)[:rerank]
            top = cand[np.argsort(-(queries[i] @ X[cand].T))[:K]]
        else:
            top = np.argpartition(-scores[i], K)[:K]
        hits.append(len(set(top.tolist()) & truth[i]) / K)
    return float(np.mean(hits))


print(f"{'m':>4}{'bytes/vec':>11}{'compression':>13}{'ADC':>9}{'SDC':>9}"
      f"{'ADC+rerank':>13}")
print("-" * 59)

for m in [4, 8, 16, 32]:
    sub = DIM // m
    centroids = np.zeros((m, N_BITS, sub))
    codes = np.zeros((N, m), dtype=np.int32)
    for j in range(m):
        block = X[:, j * sub:(j + 1) * sub]
        c, labels = kmeans2(block, N_BITS, minit='points', seed=1, iter=25)
        centroids[j], codes[:, j] = c, labels

    # ADC: build a per-query table against each subspace's centroids, then the
    # score for any code is m lookups. The table cost is paid once per query.
    adc = np.zeros((N_QUERY, N))
    for j in range(m):
        table = queries[:, j * sub:(j + 1) * sub] @ centroids[j].T
        adc += table[:, codes[:, j]]

    # SDC: quantize the query as well, then compare reconstructions.
    q_codes = np.zeros((N_QUERY, m), dtype=np.int32)
    for j in range(m):
        block = queries[:, None, j * sub:(j + 1) * sub]
        q_codes[:, j] = ((block - centroids[j][None]) ** 2).sum(-1).argmin(1)
    q_recon = np.concatenate([centroids[j][q_codes[:, j]] for j in range(m)], 1)
    recon = np.concatenate([centroids[j][codes[:, j]] for j in range(m)], 1)
    sdc = q_recon @ recon.T

    print(f"{m:>4}{m:>11d}{DIM * 4 // m:>12d}x{recall(adc):>9.4f}"
          f"{recall(sdc):>9.4f}{recall(adc, RERANK):>13.4f}")

print(f"""
Compare ADC against SDC first. ADC wins at every compression level, and it costs
NOTHING extra at query time -- the lookup table is built once per query and
amortised over the whole corpus. Keeping the query exact introduces quantization
error on one side instead of two (eq:adc-vs-sdc). This is free accuracy, and it
is why every real implementation is asymmetric.

Now read the ADC column on its own and it looks like bad news: at 32x compression
the quantizer returns the true top-10 well under half the time. A system shipping
that number would be broken.

Then read the rerank column. The SAME codes, rescoring only the top {RERANK}
candidates with exact vectors, recover most of what was lost -- and the harder
the compression, the larger the recovery in absolute terms. That is eq:ivfpq-cost's
third term, and it costs {RERANK} full-precision dot products per query --
negligible beside the scan.

The reframing is the point. PQ is not an answer, it is a CANDIDATE GENERATOR, and
its job is not to rank correctly but to get the true answers somewhere into the
top {RERANK}. That is a far weaker requirement than ranking, which is why
production systems compress far harder than any PQ-only recall number would
justify -- and it is the same cheap-then-exact cascade as retrieve-then-rerank
and model routing, arriving for the fourth time.""")

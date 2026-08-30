# -*- coding: utf-8 -*-
# Extracted from: Chapter 113 — GraphRAG and Knowledge-Graph Retrieval
# Source: src/.../ch113-graphrag.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Global questions, and the reason top-k retrieval cannot answer them.

ch:rag-indexing's retrieval returns the k chunks most similar to the query. For a
LOCAL question that is exactly the right object: the answer lives in a few spans
and similarity is how you find them. For a GLOBAL question -- "what are the main
themes across these documents" -- the answer is a property of the corpus
(eq:global-aggregate) and no k chunks contain it.

The claim worth testing is sharper than "k is too small". Similarity ranking is a
BIASED sample of the corpus (eq:selection-bias), so the estimate it supports is
wrong in a way that raising k does not fix. This listing measures that against
two alternatives at an EQUAL budget: a uniform random sample, and one summary per
community.
"""
import numpy as np

rng = np.random.default_rng(11)

N_THEME, N_CHUNK, DIM = 24, 4000, 32
N_TRIAL = 40


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


# A corpus with a skewed theme distribution -- a few large topics and a long
# tail of small ones, which is what real corpora look like.
prevalence = 1.0 / (1 + np.arange(N_THEME)) ** 0.9
prevalence /= prevalence.sum()
theme_vec = unit(rng.normal(size=(N_THEME, DIM)))

chunk_theme = rng.choice(N_THEME, size=N_CHUNK, p=prevalence)
chunk_vec = unit(0.80 * theme_vec[chunk_theme] + 0.42 * rng.normal(size=(N_CHUNK, DIM)))

true_dist = np.bincount(chunk_theme, minlength=N_THEME) / N_CHUNK


def kmeans(X, k, iters=25):
    idx = rng.choice(len(X), size=k, replace=False)
    cent = X[idx].copy()
    for _ in range(iters):
        assign = np.argmax(X @ cent.T, axis=1)
        for j in range(k):
            m = assign == j
            if m.any():
                cent[j] = X[m].mean(axis=0)
        cent = unit(cent)
    return np.argmax(X @ cent.T, axis=1), cent


# Communities: clustering stands in for the entity-graph community detection of
# cite:edge2024graphrag. What matters for this measurement is not how the
# partition is found but that a summary REPRESENTS its members and carries their
# count -- which is what makes it a stratified rather than a biased sample.
N_COMM = 28
comm, comm_vec = kmeans(chunk_vec, N_COMM)
comm_size = np.bincount(comm, minlength=N_COMM)

# What a community summary REPORTS is not what the community contains. A summary
# of two hundred chunks names the community's dominant subjects and drops the
# rest, so anything below TAU of its community disappears. Modelling the summary
# as lossless would make this comparison a fiction, and the loss is exactly where
# global search fails in practice (eq:summary-lossiness).
TAU = 0.05
comm_mix = np.zeros((N_COMM, N_THEME))
for j in range(N_COMM):
    m = comm == j
    if m.any():
        counts = np.bincount(chunk_theme[m], minlength=N_THEME).astype(float)
        counts[counts / max(comm_size[j], 1) < TAU] = 0.0
        comm_mix[j] = counts


def tv(p, q):
    """Total variation distance -- half the L1 gap between two distributions."""
    return 0.5 * np.abs(p - q).sum()


def score(est_counts):
    est = est_counts / est_counts.sum() if est_counts.sum() else est_counts
    covered = (est_counts > 0).sum() / N_THEME
    return covered, tv(est, true_dist)


def global_query():
    """A generic corpus-level question. It has to embed SOMEWHERE, and the
    friendliest realistic model is near the corpus centroid -- which is already
    dominated by the head of the theme distribution."""
    return unit(chunk_vec.mean(axis=0) + rng.normal(scale=0.35, size=DIM))


print(f"corpus: {N_CHUNK} chunks, {N_THEME} themes, {N_COMM} communities")
print(f"largest theme {true_dist.max():.1%} of the corpus, "
      f"smallest {true_dist.min():.1%}\n")
print(f"{'budget k':>9}  {'similarity top-k':>26}  {'random k':>22}  "
      f"{'communities, top-k':>26}  {'communities, largest':>26}")
print(f"{'':>9}  {'cover':>12}{'TV err':>14}  {'cover':>10}{'TV err':>12}  "
      f"{'cover':>12}{'TV err':>14}  {'cover':>12}{'TV err':>14}")
print("-" * 118)

for k in (5, 10, 20, 40, 80):
    acc = np.zeros((4, 2))
    for _ in range(N_TRIAL):
        q = global_query()

        top = np.argpartition(-(chunk_vec @ q), k)[:k]
        acc[0] += score(np.bincount(chunk_theme[top], minlength=N_THEME).astype(float))

        rnd = rng.choice(N_CHUNK, size=k, replace=False)
        acc[1] += score(np.bincount(chunk_theme[rnd], minlength=N_THEME).astype(float))

        ck = min(k, N_COMM)
        ctop = np.argpartition(-(comm_vec @ q), ck - 1)[:ck]
        acc[2] += score(comm_mix[ctop].sum(axis=0))

        cbig = np.argsort(-comm_size)[:ck]
        acc[3] += score(comm_mix[cbig].sum(axis=0))
    acc /= N_TRIAL
    print(f"{k:>9}  {acc[0,0]:>12.3f}{acc[0,1]:>14.3f}  "
          f"{acc[1,0]:>10.3f}{acc[1,1]:>12.3f}  "
          f"{acc[2,0]:>12.3f}{acc[2,1]:>14.3f}  "
          f"{acc[3,0]:>12.3f}{acc[3,1]:>14.3f}")

print("""
Read the first two column groups against each other, because that comparison is
the chapter's first result. Similarity top-k is WORSE THAN RANDOM SAMPLING at
every budget, on both metrics. It is not close and it does not close: at k=80 --
2% of the corpus -- similarity has covered 74% of themes with a distribution
error of 0.329, while a random draw of the same size covers 85% with an error of
0.182.

That is eq:selection-bias measured. The top-k set is conditioned on proximity to
the query, so it is a sample of one region rather than of the corpus, and the
error it carries is bias. Random sampling is unbiased and its error falls with k;
similarity sampling is biased and its error falls toward a floor
(eq:bias-floor) it will not cross. For a question whose answer is a property of
the whole corpus, ranking by similarity is optimising for the wrong thing.

The community columns are what pre-computed structure buys. At a budget of FIVE,
community summaries cover 51% of themes against similarity's 16%, because each
summary stands for hundreds of chunks and reports their count -- a stratified
estimator rather than a sample (eq:community-stratification).

But look where the community columns stop: 0.958 coverage and 0.155 error, and
they stay there no matter how much budget you add. That floor is
eq:summary-lossiness, not the budget. Themes that never reach 5% of any community
are not in any summary, so no amount of reading summaries recovers them. A
lossless-summary model would have printed 0.000 here and told you something
false.

Now read the whole table as a buying decision, because the honest conclusion is
uncomfortable for the technique. Community summaries win decisively at small
budgets. At k=80 a plain random sample -- zero build cost, no LLM calls, no
partition to maintain -- reaches 0.853 coverage against the summaries' 0.958 and
an error of 0.182 against 0.155. If your global questions tolerate that gap, you
have just discovered that the correct architecture is `SELECT ... ORDER BY
random() LIMIT 80`.""")

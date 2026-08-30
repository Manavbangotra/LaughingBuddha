# -*- coding: utf-8 -*-
# Extracted from: Chapter 217 — RAG Evaluation
# Source: src/.../ch217-rag-evaluation.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Retrieving the right document is worth what the generator does with it, and no more.

RAG evaluation almost always starts with retrieval metrics -- recall@k, MRR, nDCG -- because
they are the ones with a mature literature and a benchmark (cite:thakur2021beir). They are
also the metrics with the weakest link to what the user receives.

Between "the passage containing the answer was retrieved" and "the answer was correct"
sits a term nobody measures: the probability that the generator actually uses the retrieved
passage rather than its own parametric memory, an adjacent passage, or a plausible
invention (eq:rag-accuracy-is-a-product-with-a-utilisation-term).

That term multiplies every retrieval improvement, which caps the return on retrieval work at
a level set somewhere else entirely
(eq:retrieval-gains-are-capped-by-utilisation).
"""
RECALL = 0.78          # P(a passage containing the answer is in the top-k)
UTILISATION = 0.71     # P(generator grounds on it | it is present)
GEN_CORRECT = 0.91     # P(answer correct | generator grounded on the right passage)
GUESS_RIGHT = 0.19     # P(answer correct | no supporting passage retrieved)
QUERY_OK = 0.94        # P(the query was understood well enough to search on)
RERANK_KEEP = 0.88     # P(the right passage survives reranking into the prompt)

print("A five-stage pipeline, each stage with its own success rate.")
print()
STAGES = [
    ("query understood",          QUERY_OK),
    ("passage retrieved",         RECALL),
    ("survives reranking",        RERANK_KEEP),
    ("generator grounds on it",   UTILISATION),
    ("answer correct given it",   GEN_CORRECT),
]
print(f"{'stage':>28}{'success':>10}{'cumulative':>13}{'lost here':>12}")
print("-" * 63)
cum = 1.0
stage_cum = {}
for name, p in STAGES:
    prev = cum
    cum *= p
    stage_cum[name] = (p, cum, prev - cum)
    print(f"{name:>28}{p:>10.3f}{cum:>13.4f}{prev - cum:>12.4f}")

grounded_path = cum
no_passage = 1.0 - QUERY_OK * RECALL * RERANK_KEEP
e2e = grounded_path + no_passage * GUESS_RIGHT
print("-" * 63)
print(f"{'grounded and correct':>28}{'':>10}{grounded_path:>13.4f}")
print(f"{'plus lucky guesses':>28}{'':>10}{no_passage * GUESS_RIGHT:>13.4f}")
print(f"{'END TO END':>28}{'':>10}{e2e:>13.4f}")

print()
print()
print("Now improve retrieval, which is where the tooling and the papers are.")
print()
print(f"{'recall@k':>10}{'end-to-end':>13}{'gain vs 0.78':>15}"
      f"{'gain per point of recall':>27}")
print("-" * 65)


def end_to_end(recall=RECALL, util=UTILISATION, rerank=RERANK_KEEP,
               gen=GEN_CORRECT):
    reached = QUERY_OK * recall * rerank
    return reached * util * gen + (1.0 - reached) * GUESS_RIGHT


base = end_to_end()
rec_tab = {}
for r in (0.60, 0.70, 0.78, 0.86, 0.92, 0.97):
    v = end_to_end(recall=r)
    rec_tab[r] = v
    per = (v - base) / (r - RECALL) if abs(r - RECALL) > 1e-9 else 0.0
    print(f"{r:>10.2f}{v:>13.4f}{v - base:>15.4f}{per:>27.3f}")

print()
print("Every point of recall is worth utilisation times generation accuracy,")
print(f"which is {UTILISATION * GEN_CORRECT:.3f} -- not 1.")

print()
print()
print("The same sweep at three utilisation levels.")
print()
print(f"{'recall@k':>10}", end="")
for u in (0.45, 0.71, 0.92):
    print(f"{('util ' + format(u, '.2f')):>14}", end="")
print()
print("-" * 52)
grid = {}
for r in (0.60, 0.78, 0.92, 0.97):
    print(f"{r:>10.2f}", end="")
    for u in (0.45, 0.71, 0.92):
        v = end_to_end(recall=r, util=u)
        grid[(r, u)] = v
        print(f"{v:>14.4f}", end="")
    print()

print()
print(f"recall 0.60 at high utilisation ({grid[(0.60, 0.92)]:.4f}) beats")
print(f"recall 0.97 at low utilisation ({grid[(0.97, 0.45)]:.4f})")

print()
print()
print("Interventions, ranked by end-to-end gain per unit of effort.")
print()
INTERVENTIONS = [
    ("swap in a stronger embedding model", dict(recall=0.86), 3.0),
    ("add a cross-encoder reranker",       dict(rerank=0.95), 4.0),
    ("double k, keep the reranker",        dict(recall=0.88, rerank=0.84), 1.5),
    ("cite-your-sources instruction",      dict(util=0.82), 0.5),
    ("put context after the question",     dict(util=0.77), 0.2),
    ("drop passages below a score floor",  dict(util=0.79, recall=0.74), 1.0),
    ("fine-tune the generator on grounding", dict(util=0.90, gen=0.93), 12.0),
]
print(f"{'intervention':>40}{'end-to-end':>13}{'gain':>10}"
      f"{'effort':>9}{'per effort':>13}")
print("-" * 85)
inter = {}
for name, kw, eff in INTERVENTIONS:
    v = end_to_end(**kw)
    inter[name] = (v, v - base, eff, (v - base) / eff)
    print(f"{name:>40}{v:>13.4f}{v - base:>10.4f}{eff:>9.1f}"
          f"{(v - base) / eff:>13.4f}")

best = max(inter, key=lambda n: inter[n][3])
print()
print(f"best payback: {best} at {inter[best][3]:.4f} per unit")

print()
print()
print("Where the failures actually are, for the 100% - end-to-end that fail.")
print()
fail = 1.0 - e2e
buckets = [
    ("query misunderstood", 1.0 - QUERY_OK),
    ("passage not retrieved", QUERY_OK * (1.0 - RECALL)),
    ("lost in reranking", QUERY_OK * RECALL * (1.0 - RERANK_KEEP)),
    ("present but not used", QUERY_OK * RECALL * RERANK_KEEP * (1.0 - UTILISATION)),
    ("used but answered wrong",
     QUERY_OK * RECALL * RERANK_KEEP * UTILISATION * (1.0 - GEN_CORRECT)),
]
lucky = no_passage * GUESS_RIGHT
print(f"{'failure stage':>26}{'share of all queries':>23}"
      f"{'share of failures':>20}{'seen by recall@k?':>20}")
print("-" * 89)
SEEN = {"query misunderstood": "no", "passage not retrieved": "yes",
        "lost in reranking": "partly", "present but not used": "no",
        "used but answered wrong": "no"}
fb = {}
for name, amount in buckets:
    adj = amount * (1.0 if name in ("present but not used",
                                    "used but answered wrong")
                    else (1.0 - GUESS_RIGHT))
    fb[name] = adj
for name, amount in buckets:
    print(f"{name:>26}{fb[name]:>23.4f}{fb[name] / fail:>20.1%}"
          f"{SEEN[name]:>20}")
seen_share = sum(fb[n] for n in fb if SEEN[n] == "yes") / fail
print("-" * 89)
print(f"{'visible to retrieval metrics':>26}{'':>23}{seen_share:>20.1%}")

print(f"""
The pipeline table is the arithmetic and the fourth row is the one that is not in anybody's
dashboard. Query understanding, retrieval and reranking are all measured routinely; **the
probability that the generator actually grounds on the passage it was given is
{UTILISATION:.2f}** and is measured almost nowhere
(eq:rag-accuracy-is-a-product-with-a-utilisation-term).

End to end the system is right {e2e:.4f} of the time, of which
{lucky / e2e:.0%} comes from answering correctly without any supporting passage at all --
parametric memory doing the work and the retrieval system taking the credit.

The recall sweep is the result that should change a roadmap. Improving recall@k from
{0.78:.2f} to {0.92:.2f} moves end-to-end from {base:.4f} to {rec_tab[0.92]:.4f}: a gain of
{rec_tab[0.92] - base:.4f} for fourteen points of recall.

**Each point of recall is worth {UTILISATION * GEN_CORRECT:.3f} points of end-to-end
accuracy**, because it has to survive utilisation and generation
(eq:retrieval-gains-are-capped-by-utilisation). Retrieval work is discounted by a factor
nobody in the retrieval literature measures, and cite:thakur2021beir's benchmark -- which is
excellent for what it does -- measures the undiscounted quantity by construction.

The grid makes the point harder to ignore. Recall {0.60:.2f} with utilisation
{0.92:.2f} produces {grid[(0.60, 0.92)]:.4f}; recall {0.97:.2f} with utilisation
{0.45:.2f} produces {grid[(0.97, 0.45)]:.4f}. **The weaker retriever wins**, and no retrieval
metric can see why.

The intervention table converts that into a ranking. `{best}` returns
{inter[best][3]:.4f} of end-to-end accuracy per unit of effort -- against
{inter['swap in a stronger embedding model'][3]:.4f} for a stronger embedding model and
{inter['add a cross-encoder reranker'][3]:.4f} for a cross-encoder reranker.

The top of that list is prompt-shaped and the bottom is infrastructure-shaped, and the
budget usually goes to the bottom. `put context after the question` is a
{0.2:.1f}-unit change to a template.

Two of the rows are worth reading carefully because they trade off. `double k, keep the
reranker` raises recall and *lowers* the share surviving reranking, netting
{inter['double k, keep the reranker'][1]:.4f}. `drop passages below a score floor` lowers
recall and raises utilisation -- fewer distractors -- for
{inter['drop passages below a score floor'][1]:.4f}. **Both of those are invisible to a
retrieval metric**, and one of them is a retrieval regression that improves the product.

The last table is the attribution and it is the reason this chapter exists. Of everything
that fails, **{seen_share:.0%} is visible to retrieval metrics**. The largest single bucket
is `present but not used` at {fb['present but not used'] / fail:.0%} -- the passage was
retrieved, it survived reranking, it was in the prompt, and the answer did not use it.

A team measuring recall@k sees {seen_share:.0%} of its problem and has a mature toolchain
for improving exactly that part. Which is how a RAG system ends up with excellent retrieval
metrics and an unchanged answer quality, quarter after quarter.""")

---
id: ev-rag
number: 217
part: XXV
tier: full
status: draft
requires: [reference-scoring-penalises-valid-answers, aggregate-hides-which-scenario-moved,
           judge-agreement-is-at-the-human-ceiling, attribution-needs-payload-not-timing]
provides: [rag-accuracy-is-a-product-with-a-utilisation-term, retrieval-gains-are-capped-by-utilisation,
           faithfulness-and-usefulness-are-different-axes, most-rag-failures-are-invisible-end-to-end]
citations: [es2023ragas, thakur2021beir, barnett2024sevenfailures, ji2023survey]
---

## 1. Learning Objectives

By the end of this chapter you will be able to decompose end-to-end RAG accuracy into stage
success rates including the utilisation term; compute the end-to-end return on a point of
retrieval recall and show it is capped by utilisation; identify interventions that reduce a
retrieval metric while improving the product; separate faithfulness from context sufficiency
and explain why optimising the first cannot raise the ceiling set by the second; map
production failure points to the instruments that localise them; and assemble a monitoring
set that covers the largest failure buckets for a fraction of end-to-end human evaluation.

## 2. Why This Matters

RAG evaluation begins with retrieval metrics because that is where the mature literature and
the benchmarks are — {{cite:thakur2021beir}} is excellent at what it measures. Between "the
passage containing the answer was retrieved" and "the answer was correct" sits a term almost
nobody measures: **the probability the generator actually grounds on the passage it was
given**, which is **0.71** here ({{eq:rag-accuracy-is-a-product-with-a-utilisation-term}}).

That term multiplies every retrieval improvement. A point of recall@k is worth **0.377**
points of end-to-end accuracy, not one
({{eq:retrieval-gains-are-capped-by-utilisation}}) — and recall 0.60 at high utilisation
(**0.5112**) beats recall 0.97 at low utilisation (**0.3661**). The weaker retriever wins, and
no retrieval metric can see why.

Of everything that fails, **32.5% is visible to retrieval metrics**. The largest single
bucket — **36.3%** — is `present but not used`.

The generation side has the mirror-image problem. {{cite:es2023ragas}} made faithfulness
computable without ground truth, which is why it is on every dashboard. But faithfulness and
context sufficiency are different axes: faithfulness here is **0.734** and usefulness
**0.627**, and the gap is entirely answers that stay honestly inside a context that did not
contain the answer ({{eq:faithfulness-and-usefulness-are-different-axes}}). Pushing
faithfulness across its whole range raises usefulness **1.15×**; raising sufficiency from
0.63 to 0.88 raises it by **0.190**.

And {{cite:barnett2024sevenfailures}}' seven production failure points need **6 distinct
instruments** to localise, while end-to-end accuracy reports them all identically
({{eq:most-rag-failures-are-invisible-end-to-end}}).

## 3. Prerequisites

{{eq:reference-scoring-penalises-valid-answers}} from {{ch:ev-why-hard}} is why RAG
evaluation cannot simply compare against a reference answer: the acceptable-answer space for
a grounded question is large, and the reference is one draw from it.

{{eq:aggregate-hides-which-scenario-moved}} from {{ch:ev-llm-benchmarks}} is this chapter's
first result in a different guise — an end-to-end score is an aggregate over pipeline stages,
and it has the same null space.

{{eq:judge-agreement-is-at-the-human-ceiling}} from {{ch:ev-llm-judge}} governs every
automated instrument in {{sec:9-practical-example}}: the faithfulness judge, the utilisation
probe and the span-attribution check are all judges, with all of that chapter's properties.

{{eq:attribution-needs-payload-not-timing}} from {{ch:ops-observability}} is the operational
statement of the same idea: you cannot attribute a semantic failure without recording what
each stage held, and here the payload that matters is the retrieved context.

{{cite:ji2023survey}}'s hallucination taxonomy is the background for the faithfulness axis.

## 4. Intuitive Explanation

A RAG system has a pipeline and a scoreboard, and they do not correspond.

The pipeline is: understand the query, retrieve candidate passages, rerank them into a
context window, generate an answer, and — implicitly — have the generator actually use what
it was given.

The scoreboard is: recall@k, maybe MRR or nDCG, and an end-to-end accuracy or judge score.

Between those two lists is a stage with no metric on it. The generator receives a context
containing the answer and produces a response that may or may not be based on it. It might
use its own parametric memory instead. It might latch onto an adjacent passage that looks
more relevant. It might blend the two. In the numbers here, it grounds on the right passage
71% of the time it is present.

Once you write that term down, the arithmetic of RAG improvement changes.

End-to-end accuracy is a product across stages. Improve recall by a point and the gain has to
survive reranking, survive utilisation, and survive the generator being right given the right
passage. In this pipeline a point of recall is worth 0.377 points of end-to-end accuracy.

That is not a small correction. It means the entire retrieval literature — which is deep,
well-benchmarked, and full of real improvements — is optimising a quantity that reaches the
user discounted by a factor nobody in that literature measures.

The grid makes it starker. Recall 0.60 with 92% utilisation produces 0.5112 end-to-end.
Recall 0.97 with 45% utilisation produces 0.3661. **The system with worse retrieval is
substantially better**, and every retrieval metric says the opposite.

Which reorders the intervention list in an uncomfortable way. Swapping in a stronger
embedding model buys 0.0101 end-to-end per unit of effort. Adding a cross-encoder reranker
buys 0.0059. Adding a "cite your sources" instruction to the prompt buys 0.1292. Moving the
context to appear after the question rather than before it buys 0.1761.

The top of the list is prompt-shaped and the bottom is infrastructure-shaped, and the budget
goes to the bottom, because retrieval improvements look like engineering and prompt changes
look like fiddling.

There is a row in that table worth pausing on. `drop passages below a score floor` *lowers*
recall — you are discarding some passages that contained the answer — and raises utilisation,
because the context has fewer distractors. Net, it improves end-to-end accuracy. **It is a
retrieval regression that improves the product**, and a team gated on recall@k would reject
it.

Now the attribution. Of everything that fails, only 32.5% is something a retrieval metric can
see. The largest bucket, 36.3%, is `present but not used`: the passage was retrieved, it
survived reranking, it was in the prompt, and the answer did not use it.

A team measuring recall@k sees a third of its problem and has an excellent toolchain for
improving exactly that third. Which is how a RAG system accumulates quarters of retrieval
improvements and unchanged answer quality.

The generation side has the mirror-image problem, and it starts with a genuine advance.

{{cite:es2023ragas}} made *faithfulness* measurable: is every claim in the answer supported
by the retrieved context? That is checkable by a judge, needs no ground truth, and is cheap.
It went onto every RAG dashboard for good reasons.

But faithfulness is an axis, not a summary. The other axis is whether the context contained
the answer at all — sufficiency — and the two together give four outcomes.

Sufficient context, faithful answer: correct and grounded. The good case.

Sufficient context, unfaithful answer: the model knew the answer and did not use the passage.
Right facts, wrong support. Partially useful, unauditable.

Insufficient context, unfaithful answer: confident invention. The failure everybody worries
about.

Insufficient context, faithful answer: the model stays honestly inside a context that did not
contain the answer. It refuses, or hedges, or answers a nearby question. **Faithful, and
close to useless.**

Faithfulness here is 0.734 and usefulness is 0.627, and the gap is that last quadrant.

So what happens when a team optimises the metric it can measure? Push faithfulness on
insufficient context from 0.52 to 0.97 — refuse rather than invent — and measured
faithfulness rises 1.32×. Usefulness rises 1.15×.

That is a real gain and it is much smaller than the dashboard suggests, because **the ceiling
on usefulness is sufficiency and faithfulness cannot raise it.** What the push genuinely buys
is in a different column: confident inventions fall from 0.178 to 0.011. That is a safety
result, it is worth having, and it should be argued for on those terms rather than as a
quality improvement.

By contrast, taking sufficiency from 0.63 to 0.88 — fixing the corpus, improving chunking,
expanding coverage — moves usefulness from 0.627 to 0.817, against faithfulness
optimisation's move from 0.627 to 0.718 across its entire range. Twice the gain, from the
axis that has no cheap metric attached to it, which is the whole shape of this chapter: the
term with the larger effect is the term with the weaker instrument, in both halves of the
pipeline.

And there is a trap in the third column of that table. Measured faithfulness *rises* as
sufficiency rises, from 0.673 to 0.819, without anyone touching the generator — because a
model given adequate context stays inside it more readily. **A faithfulness improvement can
be a corpus improvement in disguise**, and the dashboard will credit the model.

Finally, {{cite:barnett2024sevenfailures}} catalogued seven distinct failure points from
production RAG deployments: content missing from the corpus, missed top-ranked documents,
documents not making it into the consolidated context, the answer not being extracted from
the context, wrong format, wrong specificity, and incompleteness.

Six different instruments are needed to localise those seven. End-to-end accuracy reports
every one of them identically, as a wrong answer — and for two of them, wrong specificity and
incompleteness, it often does not report anything at all, because the answer is true,
supported, and not what was asked for.

## 5. Formal Explanation

Let $q$ be the probability the query is understood, $r$ = recall@k, $\kappa$ the probability
the relevant passage survives reranking into the context, $u$ the utilisation probability,
$g$ the probability the answer is correct given grounding, and $\gamma$ the probability of a
correct answer with no supporting passage. Then

$$A = q\,r\,\kappa\,u\,g \;+\; (1 - q r \kappa)\,\gamma.$$

Differentiating in $r$:

$$\frac{\partial A}{\partial r} = q\kappa\left(u g - \gamma\right),$$

so the marginal value of retrieval work is scaled by $u g$ and offset by whatever the system
would have got right anyway. Both terms are outside the retrieval system, and neither appears
in any retrieval metric.

The same derivative in $u$ is $q r \kappa g$, which is larger than the derivative in $r$
whenever $u g - \gamma < r g$ — true for essentially any realistic parameterisation. **The
utilisation term has a larger derivative than the recall term and no measurement.**

On the generation side, let $S$ be the event that the context is sufficient and $F$ that the
answer is faithful. Usefulness is $\mathbb{E}[v]$ over the four cells with values
$v_{SF} > v_{S\bar F} > v_{\bar S F} > v_{\bar S \bar F}$. Interventions that raise
$\Pr[F \mid \bar S]$ move mass from the worst cell to the third-worst, which is bounded above
by $\Pr[\bar S] \cdot v_{\bar S F}$; interventions that raise $\Pr[S]$ move mass into the best
cell. Hence

$$\sup_{\text{faithfulness}} \mathbb{E}[v] = \Pr[S]\,\mathbb{E}[v \mid S] + \Pr[\bar S] v_{\bar S F},$$

a ceiling that is a function of $\Pr[S]$ alone.

Finally, measured faithfulness is $\Pr[F] = \Pr[S]\Pr[F\mid S] + \Pr[\bar S]\Pr[F \mid \bar
S]$, which is increasing in $\Pr[S]$ when $\Pr[F\mid S] > \Pr[F \mid \bar S]$ — so a
sufficiency improvement raises the faithfulness metric with no change to the generator.

## 6. Mathematical Foundation

The pipeline as a product with a term nobody measures:

$$A = q\,r\,\kappa\,u\,g + (1 - q r \kappa)\gamma$$ (eq:rag-accuracy-is-a-product-with-a-utilisation-term)

At $q=0.94$, $r=0.78$, $\kappa=0.88$, $u=0.71$, $g=0.91$, $\gamma=0.19$: $A = 0.4843$, of
which **13.9%** comes from answering correctly with no supporting passage.

The return on retrieval work:

$$\frac{\partial A}{\partial r} = q\kappa(ug - \gamma) = 0.377, \qquad \frac{\partial A}{\partial u} = q r \kappa g = 0.587$$ (eq:retrieval-gains-are-capped-by-utilisation)

The unmeasured term has the larger derivative.

Faithfulness and sufficiency as separate axes with a ceiling:

$$\sup_{\Pr[F|\bar S]} \mathbb{E}[v] = \Pr[S]\,\mathbb{E}[v \mid S] + \Pr[\bar S]\,v_{\bar S F}$$ (eq:faithfulness-and-usefulness-are-different-axes)

At $\Pr[S] = 0.63$: usefulness is capped at **0.718** however faithful the model becomes,
against **0.817** at $\Pr[S] = 0.88$.

And the instrument-count result:

$$|\text{failure points}| = 7, \quad |\text{instruments}| = 6, \quad |\text{distinguished by } A| = 1$$ (eq:most-rag-failures-are-invisible-end-to-end)

## 7. Internal Mechanics

Why is utilisation below one at all? Three mechanisms, and they respond to different fixes.

**Competing evidence.** The context contains the right passage and several near-misses. The
generator has no principled way to prefer the correct one and no signal about which the
retriever was most confident in. Adding more passages raises recall and lowers utilisation,
which is why the `double k` row in {{sec:9-practical-example}} nets out so poorly.

**Parametric competition.** The model already has an opinion about the answer. If the
retrieved passage contradicts it, the passage does not automatically win — and if the passage
agrees, the model may answer from memory and the retrieval contributed nothing measurable.
This is the mechanism behind the 13.9% of correct answers that arrive with no supporting
passage: real, useful, and impossible to attribute to the RAG system.

**Positional and formatting effects.** Where the context sits in the prompt, how it is
delimited, and whether the instruction asks for citation all change utilisation by more than
most retrieval improvements change recall. This is the same family of effects
{{ch:ev-llm-judge}} measured for judges, and the magnitudes are comparable.

The last one explains why the prompt-shaped interventions dominate the payback table. They
are not tricks — they are direct manipulations of the term with the largest derivative, and
they are cheap because a prompt template is cheap.

On the faithfulness side, the reason the metric is so popular deserves stating precisely. It
is the only RAG metric that requires *nothing outside the system*: the answer and the context
are both artefacts the pipeline already produced, and a judge can compare them. Every other
metric here needs something external — a labelled relevance set for recall, an annotated
corpus for sufficiency, a ground-truth answer for correctness. **Faithfulness is cheap
because it is self-referential**, and self-referential is exactly why it cannot bound
usefulness.

The sufficiency-raises-faithfulness effect is worth understanding mechanically, because it
produces a specific misdiagnosis. When context is adequate, staying inside it is the path of
least resistance for the generator; when it is not, the model must either refuse or reach
outside. So $\Pr[F \mid S] > \Pr[F \mid \bar S]$ robustly, and any corpus improvement raises
the aggregate faithfulness number. A team that improves its chunking and sees faithfulness
rise will attribute it to the generator or the prompt, and will then defend a change that did
nothing.

## 8. Implementation

The first listing decomposes end-to-end accuracy and finds the unmeasured term.

```python {tier=A name=C:/Users/MANAVB~1/AppData/Local/Temp/claude/C--Github-LaughingBuddha/30a87753-43a8-48c3-8378-261faf976dbb/scratchpad/hf1}
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
```

## 9. Practical Example

The pipeline as a product:

```
                       stage   success   cumulative   lost here
---------------------------------------------------------------
            query understood     0.940       0.9400      0.0600
           passage retrieved     0.780       0.7332      0.2068
          survives reranking     0.880       0.6452      0.0880
     generator grounds on it     0.710       0.4581      0.1871
     answer correct given it     0.910       0.4169      0.0412
---------------------------------------------------------------
        grounded and correct                 0.4169
          plus lucky guesses                 0.0674
                  END TO END                 0.4843
```

The fourth row is on nobody's dashboard: **the generator grounds on the passage it was given
0.71 of the time** ({{eq:rag-accuracy-is-a-product-with-a-utilisation-term}}). And 13.9% of
correct answers arrive with no supporting passage at all — parametric memory doing the work
and the retrieval system taking the credit.

```
  recall@k   end-to-end   gain vs 0.78   gain per point of recall
-----------------------------------------------------------------
      0.60       0.4164        -0.0679                      0.377
      0.78       0.4843         0.0000                      0.000
      0.92       0.5371         0.0528                      0.377
      0.97       0.5560         0.0717                      0.377
```

**Every point of recall is worth 0.377 points of end-to-end accuracy**, not one
({{eq:retrieval-gains-are-capped-by-utilisation}}) — discounted by a factor the retrieval
literature does not measure.

```
  recall@k     util 0.45     util 0.71     util 0.92
----------------------------------------------------
      0.60        0.2989        0.4164        0.5112
      0.78        0.3316        0.4843        0.6076
      0.97        0.3661        0.5560        0.7093
```

Recall **0.60** at high utilisation beats recall **0.97** at low utilisation, **0.5112**
against **0.3661**. The weaker retriever wins and no retrieval metric can see why.

```
                            intervention   end-to-end      gain   effort   per effort
-------------------------------------------------------------------------------------
      swap in a stronger embedding model       0.5145    0.0302      3.0       0.0101
            add a cross-encoder reranker       0.5077    0.0234      4.0       0.0059
             double k, keep the reranker       0.5069    0.0226      1.5       0.0151
           cite-your-sources instruction       0.5489    0.0646      0.5       0.1292
          put context after the question       0.5195    0.0352      0.2       0.1761
       drop passages below a score floor       0.5138    0.0295      1.0       0.0295
    fine-tune the generator on grounding       0.6075    0.1232     12.0       0.0103
```

The top of the list is prompt-shaped; the bottom is infrastructure-shaped; the budget goes to
the bottom. Note `drop passages below a score floor`: it **lowers recall and improves the
product**, and a team gated on recall@k would reject it.

```
             failure stage   share of all queries   share of failures   seen by recall@k?
-----------------------------------------------------------------------------------------
       query misunderstood                 0.0486                9.4%                  no
     passage not retrieved                 0.1675               32.5%                 yes
         lost in reranking                 0.0713               13.8%              partly
      present but not used                 0.1871               36.3%                  no
   used but answered wrong                 0.0412                8.0%                  no
-----------------------------------------------------------------------------------------
visible to retrieval metrics                                      32.5%
```

**Retrieval metrics see 32.5% of the problem**, and the largest bucket at **36.3%** is
`present but not used`.

The second listing takes up the generation side.

```python {tier=A name=C:/Users/MANAVB~1/AppData/Local/Temp/claude/C--Github-LaughingBuddha/30a87753-43a8-48c3-8378-261faf976dbb/scratchpad/hf2}
"""Faithfulness is the cheapest RAG metric to compute and the least related to usefulness.

cite:es2023ragas made faithfulness -- is every claim in the answer supported by the
retrieved context? -- measurable without ground truth, which is a genuine advance and the
reason it is now in every RAG dashboard.

It is also an axis, not a summary. Whether the context *contained* the answer is a separate
question, and the two together define four outcomes with completely different value to a
user (eq:faithfulness-and-usefulness-are-different-axes).

Optimising the cheap axis moves a system into the quadrant where it is faithful and useless,
and cite:barnett2024sevenfailures' catalogue of production failure points is mostly made of
things no end-to-end score can see
(eq:most-rag-failures-are-invisible-end-to-end).
"""
P_SUFFICIENT = 0.63      # context actually contains what is needed
P_FAITHFUL_SUFF = 0.86   # answer stays within the context, when context is sufficient
P_FAITHFUL_INSUFF = 0.52 # answer stays within the context, when it is not

print("Two axes, four outcomes. Only one of them is a good answer.")
print()
QUADRANTS = [
    ("sufficient",   "faithful",     P_SUFFICIENT * P_FAITHFUL_SUFF,
     "correct and grounded",  1.00),
    ("sufficient",   "unfaithful",   P_SUFFICIENT * (1 - P_FAITHFUL_SUFF),
     "right facts, wrong support", 0.35),
    ("insufficient", "faithful",     (1 - P_SUFFICIENT) * P_FAITHFUL_INSUFF,
     "grounded refusal or partial", 0.28),
    ("insufficient", "unfaithful",   (1 - P_SUFFICIENT) * (1 - P_FAITHFUL_INSUFF),
     "confident invention",   0.00),
]
print(f"{'context':>14}{'answer':>13}{'share':>9}"
      f"{'what the user gets':>30}{'value':>8}")
print("-" * 74)
quad = {}
for ctx, ans, share, desc, val in QUADRANTS:
    quad[(ctx, ans)] = (share, val)
    print(f"{ctx:>14}{ans:>13}{share:>9.3f}{desc:>30}{val:>8.2f}")

faith = sum(s for c, a, s, d, v in QUADRANTS if a == "faithful")
useful = sum(s * v for c, a, s, d, v in QUADRANTS)
print("-" * 74)
print(f"{'faithfulness':>14}{'':>13}{faith:>9.3f}")
print(f"{'usefulness':>14}{'':>13}{useful:>9.3f}")

print()
print()
print("Now optimise faithfulness, which is the metric you can measure without")
print("ground truth. Push the model to stay inside the context.")
print()
print(f"{'faithful | insuff':>19}{'faithful | suff':>17}{'measured faith':>16}"
      f"{'usefulness':>13}{'confident inventions':>22}")
print("-" * 87)
opt = {}
for target in (0.52, 0.65, 0.78, 0.90, 0.97):
    # Pushing faithfulness mostly changes behaviour on insufficient context:
    # the model refuses or hedges instead of inventing.
    fs = min(0.97, P_FAITHFUL_SUFF + 0.35 * (target - P_FAITHFUL_INSUFF))
    f_meas = P_SUFFICIENT * fs + (1 - P_SUFFICIENT) * target
    u = (P_SUFFICIENT * fs * 1.00
         + P_SUFFICIENT * (1 - fs) * 0.35
         + (1 - P_SUFFICIENT) * target * 0.28
         + (1 - P_SUFFICIENT) * (1 - target) * 0.00)
    inv = (1 - P_SUFFICIENT) * (1 - target)
    opt[target] = (f_meas, u, inv, fs)
    print(f"{target:>19.2f}{fs:>17.3f}{f_meas:>16.3f}"
          f"{u:>13.3f}{inv:>22.3f}")

print()
print(f"faithfulness rises {opt[0.97][0] / opt[0.52][0]:.2f}x and usefulness rises "
      f"{opt[0.97][1] / opt[0.52][1]:.2f}x")
print("because the ceiling on usefulness is context sufficiency, not faithfulness")

print()
print()
print("What actually moves usefulness: sufficiency.")
print()
print(f"{'sufficiency':>13}{'usefulness':>13}{'measured faith':>16}"
      f"{'gain per point':>17}")
print("-" * 59)
suff = {}
base_u = None
for s in (0.45, 0.55, 0.63, 0.75, 0.88):
    u = (s * P_FAITHFUL_SUFF * 1.00
         + s * (1 - P_FAITHFUL_SUFF) * 0.35
         + (1 - s) * P_FAITHFUL_INSUFF * 0.28)
    f_meas = s * P_FAITHFUL_SUFF + (1 - s) * P_FAITHFUL_INSUFF
    if base_u is None:
        base_u = (s, u)
    suff[s] = (u, f_meas)
    per = (u - base_u[1]) / (s - base_u[0]) if abs(s - base_u[0]) > 1e-9 else 0.0
    print(f"{s:>13.2f}{u:>13.3f}{f_meas:>16.3f}{per:>17.3f}")

print()
print()
print("The seven-failure-point catalogue, and which instrument sees each.")
print()
FAILURES = [
    ("missing content in the corpus",    "sufficiency annotation"),
    ("missed the top-ranked documents",  "recall@k"),
    ("not in the consolidated context",  "reranker audit"),
    ("not extracted from the context",   "utilisation probe"),
    ("wrong output format",              "schema check"),
    ("wrong specificity level",          "human or judge"),
    ("incomplete answer",                "human or judge"),
]
E2E_SEES = {
    "missing content in the corpus": "as a wrong answer",
    "missed the top-ranked documents": "as a wrong answer",
    "not in the consolidated context": "as a wrong answer",
    "not extracted from the context": "as a wrong answer",
    "wrong output format": "as a wrong answer",
    "wrong specificity level": "sometimes not at all",
    "incomplete answer": "sometimes not at all",
}
print(f"{'failure point':>34}{'instrument that localises it':>32}"
      f"{'end-to-end score shows':>24}")
print("-" * 90)
for name, inst in FAILURES:
    print(f"{name:>34}{inst:>32}{E2E_SEES[name]:>24}")

print()
print(f"{len(FAILURES)} failure points, {len(set(i for n, i in FAILURES))} "
      f"distinct instruments, and end-to-end accuracy")
print("distinguishes none of them")

print()
print()
print("Cost and coverage of the instruments, per 1000 queries.")
print()
INSTRUMENTS = [
    ("end-to-end correctness",   1, 3.40, 0.00, "tells you it is broken"),
    ("faithfulness (judge)",     1, 0.021, 0.00, "one axis of two"),
    ("recall@k on labelled set", 1, 0.000, 2.10, "one failure point"),
    ("utilisation probe",        1, 0.038, 0.00, "the largest bucket"),
    ("sufficiency annotation",   1, 4.80, 0.00, "the usefulness ceiling"),
    ("answer-span attribution",  1, 0.055, 0.00, "two failure points"),
]
print(f"{'instrument':>28}{'judge cost':>12}{'human cost':>12}"
      f"{'setup':>9}{'what it buys':>26}")
print("-" * 87)
inst_cost = {}
for name, n, jc, hc, buys in INSTRUMENTS:
    j = 1000 * jc if jc < 1 else 0.0
    h = 1000 * jc if jc >= 1 else 0.0
    setup = hc
    inst_cost[name] = j + h
    print(f"{name:>28}{j:>12,.0f}{h:>12,.0f}{setup:>9.1f}{buys:>26}")

cheap = ["faithfulness (judge)", "utilisation probe", "answer-span attribution"]
print()
print(f"the three automatable instruments together: "
      f"{sum(inst_cost[c] for c in cheap):,.0f} per 1000 queries")
print(f"end-to-end human correctness alone: "
      f"{inst_cost['end-to-end correctness']:,.0f}")
print(f"ratio: {inst_cost['end-to-end correctness'] / sum(inst_cost[c] for c in cheap):.0f}x")

print(f"""
The quadrant table is the whole argument and it takes one reading. Faithfulness is
{faith:.3f} and usefulness is {useful:.3f}, and the gap between them is entirely the
`insufficient context` row: an answer that stays honestly inside a context which did not
contain the answer is faithful and close to useless
(eq:faithfulness-and-usefulness-are-different-axes).

**Faithfulness measures whether the model lied. Sufficiency measures whether it could have
helped.** Only the first is computable without ground truth, which is why only the first
gets computed.

The optimisation table is what happens when a team acts on that. Pushing faithfulness on
insufficient context from {0.52:.2f} to {0.97:.2f} -- refuse rather than invent -- raises
measured faithfulness {opt[0.97][0] / opt[0.52][0]:.2f}x and usefulness
{opt[0.97][1] / opt[0.52][1]:.2f}x.

That is not nothing, and it is much less than the dashboard suggests, because
**the ceiling on usefulness is set by sufficiency and faithfulness cannot raise it**. The
one genuinely good thing it does is in the last column: confident inventions fall from
{opt[0.52][2]:.3f} to {opt[0.97][2]:.3f}, which is a safety result rather than a quality
one, and should be argued for on those terms.

The sufficiency table is the comparison. Taking sufficiency from {0.63:.2f} to
{0.88:.2f} moves usefulness from {suff[0.63][0]:.3f} to {suff[0.88][0]:.3f} --
{suff[0.88][0] - suff[0.63][0]:.3f}, against faithfulness optimisation's
{opt[0.97][1] - opt[0.52][1]:.3f} over its whole range.

Notice the third column while you are there. Measured faithfulness *rises* as sufficiency
rises, from {suff[0.45][1]:.3f} to {suff[0.88][1]:.3f}, without anyone touching the
generator -- because a model given adequate context stays inside it more readily. **A
faithfulness improvement can be a corpus improvement in disguise**, and the dashboard will
credit the model.

The failure-point table is cite:barnett2024sevenfailures' catalogue with a column added.
Seven distinct production failures, {len(set(i for n, i in FAILURES))} different instruments
needed to localise them, and end-to-end accuracy reports every one of them identically as
`a wrong answer` -- when it notices at all
(eq:most-rag-failures-are-invisible-end-to-end).

Two of them it does not notice: wrong specificity and incompleteness produce answers that
are true, supported, and not what was asked for. Those pass a correctness check and fail a
user.

The instrument table is the practical answer and the numbers are friendlier than the
argument so far suggests. A faithfulness judge, a utilisation probe and a span-attribution
check together cost {sum(inst_cost[c] for c in cheap):,.0f} per thousand queries, against
{inst_cost['end-to-end correctness']:,.0f} for human end-to-end correctness --
{inst_cost['end-to-end correctness'] / sum(inst_cost[c] for c in cheap):.0f} times cheaper --
and between them they localise the four largest buckets from
ch:ev-rag's first listing.

The expensive instrument is sufficiency annotation, and it is expensive because somebody has
to read the corpus and decide whether the answer was in there. It is also the only one that
measures the ceiling. **Sample it rather than skipping it**: a few hundred annotated queries
a quarter gives you the number that bounds every other metric on the dashboard, and without
it a RAG programme is optimising terms in a product whose largest factor is unmeasured.""")
```

```
       context       answer    share            what the user gets   value
--------------------------------------------------------------------------
    sufficient     faithful    0.542          correct and grounded    1.00
    sufficient   unfaithful    0.088    right facts, wrong support    0.35
  insufficient     faithful    0.192   grounded refusal or partial    0.28
  insufficient   unfaithful    0.178           confident invention    0.00
--------------------------------------------------------------------------
  faithfulness                 0.734
    usefulness                 0.627
```

Faithfulness **0.734**, usefulness **0.627**, and the gap is the third row: honestly inside a
context that did not contain the answer
({{eq:faithfulness-and-usefulness-are-different-axes}}).

```
  faithful | insuff  faithful | suff  measured faith   usefulness  confident inventions
---------------------------------------------------------------------------------------
               0.52            0.860           0.734        0.627                 0.178
               0.78            0.951           0.888        0.691                 0.081
               0.97            0.970           0.970        0.718                 0.011

  sufficiency   usefulness  measured faith   gain per point
-----------------------------------------------------------
         0.45        0.489           0.673            0.000
         0.63        0.627           0.734            0.763
         0.88        0.817           0.819            0.763
```

Faithfulness across its whole range takes usefulness from **0.627** to **0.718**.
Sufficiency 0.63 → 0.88 takes it from **0.627** to **0.817** — twice as far. What faithfulness genuinely buys is in the last column — confident inventions
from **0.178 to 0.011**, which is a safety result. And note the third column of the second
table: **measured faithfulness rises with sufficiency with no change to the generator.**

```
                     failure point    instrument that localises it  end-to-end score shows
------------------------------------------------------------------------------------------
     missing content in the corpus          sufficiency annotation       as a wrong answer
   missed the top-ranked documents                        recall@k       as a wrong answer
   not in the consolidated context                  reranker audit       as a wrong answer
    not extracted from the context               utilisation probe       as a wrong answer
               wrong output format                    schema check       as a wrong answer
           wrong specificity level                  human or judge    sometimes not at all
                 incomplete answer                  human or judge    sometimes not at all
```

{{cite:barnett2024sevenfailures}}' seven failure points, **6 instruments**, and end-to-end
accuracy distinguishes **none** of them
({{eq:most-rag-failures-are-invisible-end-to-end}}).

```
                  instrument  judge cost  human cost    setup              what it buys
---------------------------------------------------------------------------------------
      end-to-end correctness           0       3,400      0.0    tells you it is broken
        faithfulness (judge)          21           0      0.0           one axis of two
           utilisation probe          38           0      0.0        the largest bucket
      sufficiency annotation           0       4,800      0.0    the usefulness ceiling
     answer-span attribution          55           0      0.0        two failure points
```

Three automatable instruments cost **114 per 1,000 queries** against **3,400** for human
end-to-end correctness — **30× cheaper** — and between them they localise the four largest
buckets.

## 10. Production Considerations

Measure utilisation. It is the term with the largest derivative and the only stage with no
standard metric; a judge asked "is this answer derived from passage $i$?" is enough.

Try the prompt-shaped interventions first. Context position and a citation instruction are
the two highest-payback changes in the table and cost under a day between them.

Stop gating on recall@k alone. At least one improvement in this chapter reduces recall and
improves the product.

Report faithfulness and sufficiency separately, never a blend. They are orthogonal and only
one of them bounds usefulness.

Sample sufficiency annotation quarterly. A few hundred annotated queries gives you the
ceiling every other metric sits under, and it is the only expensive instrument that cannot be
skipped.

Attribute failures to a stage before choosing work. `present but not used` is the largest
bucket in most systems and it is not a retrieval problem.

Argue for faithfulness work as safety, not quality. It removes confident inventions by an
order of magnitude and raises usefulness by a seventh.

## 11. Common Mistakes

**Optimising recall as though it were end-to-end.** It is discounted by utilisation and
generation accuracy.

**Blending faithfulness and correctness into one score.** They are separate axes and only one
has a ground-truth requirement.

**Reading a faithfulness rise as a generator improvement.** A corpus improvement raises it
with the generator untouched.

**Increasing k to improve recall.** It raises recall and lowers utilisation, and can net
negative.

**Treating an end-to-end score as diagnostic.** Seven distinct failure points, one output.

**Crediting RAG for parametric answers.** Here 13.9% of correct answers had no supporting
passage.

## 12. Failure Modes

**Quarters of retrieval gains, flat answer quality.** The team improved 32.5% of the problem
very effectively.

**Faithfulness at 0.97 and users complaining.** The system refuses honestly on insufficient
context, and the dashboard is green.

**Corpus fix credited to a prompt change.** Sufficiency rose, measured faithfulness rose, and
the prompt change is now defended.

**Recall regression blocks a real improvement.** The score floor that improved end-to-end
accuracy fails the retrieval gate.

**Silent specificity failures.** Answers are true, supported, and at the wrong level of
detail; no automated instrument fires.

**Utilisation collapse after a context-length increase.** More passages fit, recall rises,
grounding falls, and nothing in the monitoring set contains the term that moved.

## 13. Alternatives

**End-to-end human evaluation only.** Correct, diagnostic-free, and 30× the cost of the
automatable instrument set.

**Reference-answer scoring.** Compare against a gold answer.
{{eq:reference-scoring-penalises-valid-answers}} applies with full force to grounded
generation.

**Attribution-first evaluation.** Require citations and score the citations. Makes
utilisation directly observable and constrains the generator's output format.

**Synthetic query generation from the corpus.** Generate questions whose answers are known to
be in the corpus, which fixes sufficiency at 1.0 by construction — useful for isolating the
other stages, and it measures a distribution you do not have.

**Counterfactual context ablation.** Run each query with and without the retrieved context
and compare. Directly measures the retrieval system's contribution, at 2× inference cost, and
it is the cleanest available answer to the parametric-credit problem.

## 14. Evaluation

Measure utilisation on a sample: for queries where the answer passage was retrieved, ask a
judge whether the answer used it.

Run the counterfactual ablation on a sample to find how many correct answers survive removing
the context entirely. That is your parametric baseline.

Annotate sufficiency on a few hundred queries per quarter and report it as the ceiling on
every quality metric you publish.

Decompose failures by stage before every planning cycle, and check that the work planned
matches the bucket sizes.

Track faithfulness and sufficiency as two series. If faithfulness moves and sufficiency moves
with it, the generator did nothing.

## 15. Advanced Concepts

The independence assumed between pipeline stages is the model's weakest point and it fails
in a helpful direction and an unhelpful one. Helpfully: queries that are well understood also
tend to retrieve well, so the stages are positively correlated and the true end-to-end
accuracy is *higher* than the product suggests. Unhelpfully: the same correlation means the
failures concentrate — a hard query fails at several stages at once — so the population of
failing queries is smaller and harder than the independent model implies, and the marginal
return on any single-stage improvement is lower. The direction that matters for planning is
the second.

The utilisation term also is not a constant, which the listing treats it as. It depends on
how many distractors are present, on whether the retrieved passage agrees with the model's
prior, and on the context's position — so it is endogenous to every retrieval decision. That
makes $\partial A/\partial r$ an underestimate in some regimes and an overestimate in others:
adding passages raises $r$ and lowers $u$ simultaneously, which the `double k` row captures
crudely and a proper model would capture as a function $u(k)$. Fitting that function from
production data is a day of work and it would replace most of the guesswork in RAG capacity
planning.

There is a measurement problem with utilisation that deserves flagging. Asking a judge "did
this answer use passage $i$?" is itself a judge task, with {{ch:ev-llm-judge}}'s ceiling and
biases, and it is a task where the judge may be systematically wrong: an answer that agrees
with a passage is not necessarily derived from it. The clean instrument is the counterfactual
ablation — remove the passage and see whether the answer changes — which is unambiguous and
costs a second inference. **Where a counterfactual is available, prefer it to a judgement**,
and here one is.

Finally, the faithfulness ceiling result has a consequence for how RAG systems should degrade.
Since usefulness on insufficient context is bounded by $v_{\bar S F}$ — the value of a
grounded refusal — the design question is how large that value can be made. A bare refusal is
worth little; a refusal that says *what* is missing and *where* the user might look is worth
considerably more, and it costs nothing extra to generate because the system already knows
which retrieval scores were low. **Most of the value available in the insufficient-context
quadrant is unclaimed**, and it is claimed by product design rather than by evaluation.

## 16. Connection to Previous Chapters

{{eq:aggregate-hides-which-scenario-moved}} from {{ch:ev-llm-benchmarks}} is this chapter's
attribution result: an end-to-end score is an aggregate over stages, with the same null space
and the same remedy — decompose.

{{eq:judge-agreement-is-at-the-human-ceiling}} from {{ch:ev-llm-judge}} bounds every cheap
instrument here, and {{sec:15-advanced-concepts}} argues the counterfactual ablation escapes
that bound where a judgement cannot.

{{eq:attribution-needs-payload-not-timing}} from {{ch:ops-observability}} is the operational
requirement: none of this decomposition is computable unless the retrieved context was
recorded, which is a payload field and the first thing truncated.

{{eq:reference-scoring-penalises-valid-answers}} from {{ch:ev-why-hard}} is why faithfulness
became the dominant RAG metric in the first place — it was the one thing measurable without a
reference.

## 17. Exercises

1. Estimate utilisation in your own system by asking a judge, for queries where the answer
   passage was retrieved, whether the answer used it.

2. Run the counterfactual ablation on 200 queries. What share of correct answers survive
   removing the context?

3. Compute $\partial A/\partial r$ and $\partial A/\partial u$ for your parameters. Which
   term does your roadmap address?

4. Annotate sufficiency on 200 queries and compute the implied ceiling on usefulness. How far
   below it are you?

5. Model $u(k)$ — utilisation as a function of the number of retrieved passages — and find the
   $k$ that maximises end-to-end accuracy rather than recall.

## 18. Interview Questions

1. Our recall@k went from 0.78 to 0.92 and answer quality barely moved. What happened?

2. Why might reducing recall improve a RAG system?

3. Our faithfulness score is 0.95. What does that tell you about answer quality?

4. What is the cheapest instrument that would tell you whether retrieval is your problem?

5. Our faithfulness improved after a chunking change. What do you suspect?

6. How would you measure whether retrieval is contributing anything at all?

## 19. Research Questions

1. What is the empirical form of $u(k)$ across retrieval configurations, and how much
   end-to-end accuracy is available from optimising $k$ against it rather than against
   recall?

2. How large is the parametric-answer share across domains, and how should RAG systems be
   credited for it?

3. Can utilisation be measured reliably by judgement, or is counterfactual ablation the only
   sound instrument?

4. How much of the insufficient-context quadrant's value is recoverable by refusal design,
   and what refusal formats recover most?

## 20. Chapter Summary

RAG evaluation measures the stages with mature tooling and misses the stage that binds.

End-to-end accuracy is a product, and it contains a term nobody measures: **the generator
grounds on the retrieved passage 0.71 of the time**
({{eq:rag-accuracy-is-a-product-with-a-utilisation-term}}). That term discounts every
retrieval improvement — a point of recall@k is worth **0.377** points of end-to-end accuracy
({{eq:retrieval-gains-are-capped-by-utilisation}}) — and it has a *larger* derivative than
recall does. Recall 0.60 at high utilisation (**0.5112**) beats recall 0.97 at low
utilisation (**0.3661**), and the two highest-payback interventions in the table are prompt
changes. One of them lowers recall.

Retrieval metrics see **32.5%** of failures; the largest bucket, **36.3%**, is `present but
not used`.

On the generation side, faithfulness is cheap because it is self-referential, and
self-referential is why it cannot bound usefulness. Faithfulness **0.734** against usefulness
**0.627**, with the gap in honestly-grounded answers to insufficient contexts
({{eq:faithfulness-and-usefulness-are-different-axes}}). Optimising faithfulness across its
whole range buys **1.15×** usefulness and a tenfold reduction in confident inventions — a
safety result. Sufficiency from 0.63 to 0.88 takes usefulness from **0.627 to 0.817**, twice
as far, and raises measured faithfulness with the generator untouched.

{{cite:barnett2024sevenfailures}}' seven failure points need **6 instruments**; end-to-end
accuracy distinguishes **none** ({{eq:most-rag-failures-are-invisible-end-to-end}}), and
three automatable instruments cover the largest buckets at **30×** less than human
evaluation.

The pattern is one this part keeps producing and this chapter states most plainly: the
measurable quantities and the binding quantities are different sets, and the overlap is
smaller than a dashboard implies. Retrieval is measurable because relevance labels exist.
Faithfulness is measurable because it needs nothing external. Utilisation and sufficiency are
the two terms that bound the system, and they are the two nobody measures — not because they
are hard, but because neither has a benchmark.

Carry forward: **measure utilisation**, and **faithfulness is a safety metric, sufficiency is
the ceiling**.

## 21. Further Reading

- {{cite:es2023ragas}} — reference-free RAG metrics, including the faithfulness axis this
  chapter argues should be read as one of two.
- {{cite:thakur2021beir}} — the retrieval benchmark, excellent at the quantity that reaches
  the user discounted.
- {{cite:barnett2024sevenfailures}} — seven production failure points, which is the
  instrument-count argument in its original form.
- {{cite:ji2023survey}} — hallucination taxonomy, the background for the faithfulness axis and
  the confident-invention quadrant.

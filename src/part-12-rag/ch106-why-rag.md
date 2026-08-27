---
id: rag-why
number: 106
part: XII
tier: full
status: draft
requires: [fm-what-they-are, llm-long-context, llm-hallucination,
           emb-what-they-are, emb-reranking, llm-routing]
provides: [parametric-knowledge, non-parametric-knowledge, rag-architecture,
           knowledge-freshness, attribution, retrieval-cost-model,
           context-stuffing, rag-long-context-tradeoff]
citations: [lewis2020rag, guu2020realm, izacard2022atlas, li2024ragvslongcontext,
            liu2023lost, gao2023ragsurvey, ji2023survey, hoffmann2022chinchilla]
---

## 1. Learning Objectives

By the end of this chapter you will be able to state precisely what RAG changes
about a system — not "adds knowledge" but *moves* it — and derive the four
properties that decide whether to move it; compute the cost crossover between
retrieval and long-context stuffing and say which way it moves with token prices;
explain why fine-tuning is the wrong tool for facts using the information
argument from {{part:9}}; and describe RAG's failure surface as a
*consequence* of the architecture rather than a list of bugs.

## 2. Why This Matters

A foundation model knows what was in its training data, as of its cutoff, with no
way to say where anything came from and no way to forget one thing on request.
For a very large class of applications, every clause in that sentence is a
blocker.

RAG is the standard answer, and it is worth being exact about what it does.
**It does not add knowledge to a model. It moves knowledge out of the model and
puts a retrieval step in front of it.** Everything good about RAG and everything
bad about it follows from that relocation: the knowledge becomes updatable,
attributable, and access-controllable, and it also becomes a distributed system
with four new failure stages that did not exist before.

This chapter is the decision. The eleven that follow are the consequences.

{{maturity:ESTABLISHED}} The architecture is settled and universal.
{{maturity:EMERGING}} Where its boundary sits against long-context models is
genuinely moving, and it moves with token prices rather than with capability.

## 3. Prerequisites

{{ch:fm-what-they-are}} for the adaptation-information argument that makes
fine-tuning the wrong tool for facts; {{ch:llm-long-context}} for usable versus
advertised context; {{ch:llm-hallucination}} for groundedness and why it is
measurable when truth is not; {{ch:emb-what-they-are}} and {{ch:emb-reranking}}
for the retrieval stack this part sits on; {{ch:llm-routing}} for the cascade
arithmetic reused here.

## 4. Intuitive Explanation

### Two places knowledge can live

**In the weights** — *parametric*. Written during pretraining, compressed into
billions of floats, retrieved by a forward pass. Fast, fluent, and inseparable
from everything else the model knows.

**Outside the weights** — *non-parametric*. In documents, in a database, in a
search index. Retrieved by a lookup and inserted into the prompt.

The distinction is not about capability; a model can answer from either. It is
about **four operational properties**, and these are what the decision turns on:

| | parametric | non-parametric |
|---|---|---|
| updating one fact | retrain or fine-tune | edit one row |
| saying where it came from | impossible | trivial |
| hiding a document from one user | impossible | a filter |
| removing a fact on request | not reliably possible | delete it |

Read the second column: **every one of those is a routine operation.** Read the
first: every one is either impossible or a training run. That asymmetry, not
accuracy, is why RAG exists.

### The thing people get wrong

RAG is routinely described as "giving the model access to your data", which
suggests an addition. It is a *relocation*, and relocations have costs.

Once knowledge is external, a question can fail in four new places before the
model ever sees it: the document may never have been ingested correctly, it may
have been chunked so the answer straddles a boundary, retrieval may not have
found it, or the model may have been handed it and ignored it. **A RAG system has
four failure stages where a bare model had one**, and {{ch:rag-failures}} exists
because telling them apart is a skill.

The compensation is that all four are *inspectable*. When a bare model gets a
fact wrong there is nothing to look at. When a RAG system does, you can ask which
stage lost it — and that auditability is worth more in production than the
quality difference.

### Why not just fine-tune?

The most common alternative, and {{part:9}} already answered it.
{{eq:adaptation-information-ratio}} put fine-tuning about eight orders of
magnitude below pretraining in information supplied, and the practical
consequence there was that **fine-tuning teaches format reliably and facts
poorly.**

For facts specifically, three things go wrong. A fact seen a handful of times in
fine-tuning competes against the same fact's contradiction seen thousands of
times in pretraining. There is no mechanism to *remove* a superseded fact — only
to add a competing one. And the model cannot report which of the two it used.

**Fine-tune for behaviour, retrieve for facts.** That is the whole rule, and
{{part:14}} takes the other half.

## 5. Formal Explanation

### 5.1 What RAG is

A bare model computes

$$ P(y \given x) $$ (eq:parametric-generation)

RAG conditions on a retrieved set as well:

$$ P(y \given x) = \sum_{z \in \mathcal{Z}} \underbrace{P(z \given x)}_{\text{retriever}} \; \underbrace{P(y \given x, z)}_{\text{generator}} $$ (eq:rag-marginal)

{{cite:lewis2020rag}} trained both terms jointly and marginalised over the top-$k$
documents. **Almost nothing in production does this.** The standard architecture
takes the arg-max instead:

$$ \hat{z} = \text{top-}k\text{-}\argmax_{z} P(z \given x), \qquad \hat{y} = \argmax_y P\big(y \given x, \hat{z}\big) $$ (eq:rag-practical)

which is worth stating explicitly, because it means **the retriever's errors are
not marginalised away — they are committed to.** {{eq:rag-marginal}}'s tolerance
for an imperfect retriever does not survive the simplification, and that is why
{{ch:emb-reranking}}'s recall ceiling binds so hard here.

### 5.2 The ceiling

The consequence, and the single most important inequality in the part:

$$ \Prob[\text{correct answer}] \;\leq\; \Prob\big[\text{answer} \in \hat{z}\big] \;=\; \text{recall@}k $$ (eq:rag-ceiling)

**A RAG system cannot be more accurate than its retriever's recall.** No prompt,
no larger model, and no reranker recovers a document that was not retrieved —
this is {{eq:recall-ceiling}} from {{ch:emb-reranking}} with the generator in
place of the reranker.

The practical reading is a diagnosis rule that {{ch:rag-failures}} builds on:
measure recall@$k$ *first*. If it is 0.6, the ceiling is 0.6, and every hour
spent on prompts is bounded by a number nobody looked at.

### 5.3 The four properties, formally

**Freshness.** A parametric model's knowledge is fixed at cutoff $T_c$. If facts
decay at rate $\lambda$,

$$ \text{accuracy}_{\text{param}}(t) \approx a_0 e^{-\lambda (t - T_c)}, \qquad \text{accuracy}_{\text{RAG}}(t) \approx a_0 \cdot \text{recall@}k $$ (eq:freshness-decay)

The parametric curve decays and the RAG curve does not. {{sec:9-practical-example}}
computes where they cross, and the answer — months, not years, for
fast-moving domains — is the quantitative form of "the model is out of date".

**Attribution.** {{ch:llm-hallucination}} defined groundedness
({{eq:groundedness}}) and noted it is measurable when truth is not. RAG makes it
*computable*, because the source set is known:

$$ \text{grounded}(y) = \Ind\big[\text{every claim in } y \text{ is supported by some } z \in \hat{z}\big] $$ (eq:rag-groundedness)

A bare model has no $\hat{z}$, so this quantity does not exist for it. **This is
RAG's strongest argument and it is not about accuracy at all.**

**Access control.** Filter $\mathcal{Z}$ by the caller's permissions before
retrieval, and the model cannot reveal what it was never given. In the parametric
case there is no analogous operation. Note this is {{ch:emb-vector-db}}'s
pre-filtering problem with the failure mode upgraded from bad results to a data
breach.

**Cost of update.** Adding a document is an ingest; adding a fact to the weights
is a training run.

### 5.4 The cost comparison

The question everyone asks. Let $N_c$ be corpus tokens, $Q$ queries, $k$
retrieved chunks of $L_c$ tokens, and $p$ the input-token price.

$$ C_{\text{stuff}} = Q \cdot N_c \cdot p, \qquad C_{\text{RAG}} = \underbrace{N_c \cdot p_e}_{\text{embed once}} + Q \cdot k L_c \cdot p + C_{\text{index}} $$ (eq:rag-cost-comparison)

The structural difference is that **stuffing is linear in corpus size *per
query*** while RAG's per-query term is constant in corpus size. So:

$$ \frac{C_{\text{stuff}}}{C_{\text{RAG}}} \approx \frac{N_c}{k L_c} \quad \text{for large } Q $$ (eq:cost-ratio)

For a 10-million-token corpus with $k=8$ chunks of 500 tokens, that is a factor
of 2,500. **The comparison is not close, and it is not close in a way that no
foreseeable price change fixes**, because {{eq:cost-ratio}} has no price in it.

Prices decide only the *small-corpus* boundary, which
{{sec:9-practical-example}} computes.

> **IMPORTANT:** {{cite:li2024ragvslongcontext}} found long context wins on
> *quality* when the corpus fits, and its own conclusion is a router — send each
> query to whichever is better. That is {{ch:llm-routing}}'s architecture
> applied one level up, and it is the honest answer: not "which wins" but "which
> for this query".

## 6. Mathematical Foundation

### 6.1 Why stuffing does not scale even when affordable

{{eq:cost-ratio}} is about money. There is a second argument that is about
attention, and it survives free tokens.

{{ch:llm-long-context}} established that usable context is well below advertised
context ({{eq:usable-context}}), and {{cite:liu2023lost}} that retrieval accuracy
depends on *position* ({{eq:u-shape}}). Let $\rho(n)$ be the probability the
model attends to a relevant fact among $n$ context tokens. Then

$$ \Prob[\text{use the fact}] = \underbrace{\Prob[\text{fact present}]}_{1 \text{ if stuffed}} \times \rho(n) $$ (eq:presence-times-attention)

Stuffing maximises the first factor and degrades the second. Retrieval does the
reverse: it accepts $\text{recall@}k < 1$ in exchange for a small $n$ where
$\rho$ is near 1.

**So RAG and stuffing are two positions on one trade-off, not two techniques.**
And the framing that makes the choice tractable: **stuffing is implicit
selection, performed by attention, unauditable. Retrieval is explicit selection,
performed by a component you can measure.** When the selection is wrong, only one
of the two lets you find out why.

### 6.2 Retrieval as a substitute for parameters

{{cite:izacard2022atlas}}'s claim, in the form worth carrying. A parametric model
stores facts at some bits-per-parameter rate; scaling laws
({{ch:fm-scaling-laws}}) say capability grows as a power of parameter count. A
retrieval-augmented model of size $M$ with a corpus of $N_c$ tokens behaves
roughly like a parametric model of size $M' > M$ on knowledge-intensive tasks:

$$ M'(M, N_c) > M \quad\text{with}\quad \frac{\partial M'}{\partial N_c} > 0 \text{ at negligible cost} $$ (eq:retrieval-substitutes)

The engineering content is the *cost asymmetry*: growing $N_c$ is an ingest job,
growing $M$ is a training run and a permanent serving cost. **Retrieval buys
capability at storage prices instead of compute prices**, and that exchange rate
is the economic case for the entire part.

It does not extend to reasoning. {{ch:fm-emergence}}'s capabilities are not in
the corpus, and no retrieval improves them — which is precisely why
{{part:16}} exists as a separate subject.

### 6.3 The budget, as a constrained optimisation

Given a context budget $B$ tokens, choosing $k$ chunks of length $L_c$:

$$ \max_{k, L_c} \; \underbrace{R(k)}_{\text{recall}} \cdot \underbrace{\rho(k L_c)}_{\text{attention}} \quad \text{s.t.} \quad k L_c \leq B $$ (eq:context-budget)

Both factors move against each other in $k$: more chunks raise recall
({{eq:recall-saturation}}) and lower the probability each is attended to. **There
is an interior optimum, it is usually much smaller than the budget allows, and
almost nobody looks for it.** {{ch:rag-generation}} measures it directly; the
point here is that "use the whole context window" is a choice with a cost, not a
free default.

## 7. Internal Mechanics

```mermaid {#fig:rag-stages caption="The four stages a question passes through, and where each can lose it. A bare model has only the last box; RAG has four, which is the trade the architecture makes — more places to fail, and a place to look when it does."}
flowchart LR
    D["source documents"] -->|"1. ingest<br/>(ch:rag-ingestion)"| P["parsed text"]
    P -->|"2. chunk<br/>(ch:rag-chunking)"| C["chunks + metadata"]
    C -->|"embed, index"| I[("index")]
    Q["question"] -->|"3. retrieve<br/>(ch:rag-indexing)"| I
    I -->|"top-k"| G["4. generate<br/>(ch:rag-generation)"]
    Q --> G
    G --> A["answer + citations"]
    D -.->|"lost in parsing"| X1["never retrievable"]
    C -.->|"answer straddles a boundary"| X2["never retrievable"]
    I -.->|"eq:rag-ceiling"| X3["not retrieved"]
    G -.->|"ignored the context"| X4["wrong despite having it"]
```

### 7.1 The naive baseline, and why it is the right starting point

Parse, split at fixed length, embed, retrieve top-$k$ by cosine, concatenate into
a prompt. {{cite:gao2023ragsurvey}} calls this *naive RAG* and it takes an
afternoon.

**Build it first, always.** Not because it is good, but because it is the only
way to find out which of {{fig:rag-stages}}'s four stages your corpus actually
breaks. The chapters that follow are all *responses to specific measured
failures*, and adopting them without the measurement is how a RAG system acquires
six components and no diagnosis.

### 7.2 What the naive baseline gets wrong, by stage

| Stage | Naive behaviour | Typical damage |
|---|---|---|
| ingest | text extraction only | tables destroyed, 10–30% of PDF content lost |
| chunk | fixed length, no structure | answers straddle boundaries |
| retrieve | dense top-$k$, no filter | identifiers unmatchable ({{ch:emb-hybrid}}) |
| generate | concatenate in rank order | best chunk buried mid-context ({{eq:u-shape}}) |

Every subsequent chapter in this part is a row of this table, expanded.

### 7.3 When not to use RAG

Worth stating plainly, because the default has become reflexive.

- **The corpus fits comfortably in the context window.** Stuffing is simpler,
  has no failure stages, and — per {{cite:li2024ragvslongcontext}} — is usually
  *better*. Note that {{sec:9-practical-example}} finds this is **not** a cost
  argument: retrieval is cheaper almost immediately. It is a complexity and
  quality argument, which is a stronger one.
- **The task needs no external knowledge.** Summarisation, rewriting,
  translation, code transformation. Retrieval adds latency and distractors.
- **The knowledge is structured.** A schema and a query planner beat a vector
  index over serialised rows ({{ch:rag-structured}}).
- **The answer is a property of the whole corpus.** "What are the main themes?"
  is not in any chunk ({{ch:rag-graph}}).
- **Latency is the binding constraint.** Retrieval adds a round trip, and for
  some interactive uses that is decisive.

## 8. Implementation

```python {tier=A name=rag-cost-model}
"""When is retrieval cheaper than stuffing, and by how much?

Compares two architectures across corpus sizes at a fixed query volume:

  stuff -- put the entire corpus in the context on every query
  rag   -- embed the corpus once, retrieve k chunks per query

Prices are set as constants at the top and are the only thing that dates. The
STRUCTURE of the comparison (eq:cost-ratio) does not depend on them, which is the
point the table is meant to make.
"""

PRICE_INPUT = 3.00 / 1_000_000      # $ per input token to the generator
PRICE_EMBED = 0.02 / 1_000_000      # $ per token to embed, one pass
INDEX_GB_MONTH = 0.25               # $ per GB-month for the vector index
BYTES_PER_VECTOR = 768 * 4          # float32, before any compression

CHUNK_TOKENS = 500
K_RETRIEVED = 8
QUERIES_PER_MONTH = 100_000

CORPUS_SIZES = [10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]


def stuffing_cost(corpus_tokens, queries):
    """Every query pays for the whole corpus: linear in corpus size PER QUERY."""
    return queries * corpus_tokens * PRICE_INPUT


def rag_cost(corpus_tokens, queries):
    """Embedding is paid once; the per-query term is constant in corpus size."""
    n_chunks = corpus_tokens / CHUNK_TOKENS
    embed = corpus_tokens * PRICE_EMBED
    index_gb = n_chunks * BYTES_PER_VECTOR / 1e9
    index = index_gb * INDEX_GB_MONTH
    generate = queries * K_RETRIEVED * CHUNK_TOKENS * PRICE_INPUT
    return embed + index + generate, embed, index, generate


print(f"assumptions: {QUERIES_PER_MONTH:,} queries/month, k={K_RETRIEVED} chunks "
      f"of {CHUNK_TOKENS} tokens, ${PRICE_INPUT * 1e6:.2f}/M input tokens\n")
print(f"{'corpus tokens':>15}{'stuff $/mo':>14}{'RAG $/mo':>12}"
      f"{'ratio':>10}{'RAG: embed/index/gen':>28}")
print("-" * 80)

for corpus in CORPUS_SIZES:
    stuff = stuffing_cost(corpus, QUERIES_PER_MONTH)
    total, embed, index, gen = rag_cost(corpus, QUERIES_PER_MONTH)
    ratio = stuff / total
    print(f"{corpus:>15,}{stuff:>14,.0f}{total:>12,.0f}{ratio:>9.1f}x"
          f"   {embed:>7,.1f} /{index:>7,.1f} /{gen:>8,.0f}")

# Where do the two meet?
lo, hi = 1.0, 1e9
for _ in range(200):
    mid = (lo + hi) / 2
    if stuffing_cost(mid, QUERIES_PER_MONTH) < rag_cost(mid, QUERIES_PER_MONTH)[0]:
        lo = mid
    else:
        hi = mid
breakeven = (lo + hi) / 2

print(f"""
Break-even corpus size: {breakeven:,.0f} tokens -- which is exactly
k * chunk_tokens = {K_RETRIEVED} * {CHUNK_TOKENS}, and that is not a coincidence.
Below that size RAG would retrieve the entire corpus anyway, so the two
architectures ARE the same thing and there is nothing to compare.

The honest conclusion is therefore stronger and less interesting than the usual
one: on cost, RAG wins essentially as soon as the corpus exceeds the context you
would retrieve. There is no meaningful cost regime in which stuffing is the
cheaper choice.

So COST IS NOT THE REASON TO STUFF A SMALL CORPUS. The reasons are simplicity and
quality -- no ingestion, no chunking, no retrieval, none of the four failure
stages in fig:rag-stages, and the finding in cite:li2024ragvslongcontext that
long context is BETTER when everything fits. Those are good reasons. "It is
cheaper" is not one, and teams that justify stuffing on cost grounds are usually
right for the wrong reason.

Above the break-even the gap opens fast, because eq:cost-ratio says the ratio
grows as corpus/(k * chunk) and nothing in that expression is a price. At 10M
tokens the factor is {stuffing_cost(10_000_000, QUERIES_PER_MONTH) / rag_cost(10_000_000, QUERIES_PER_MONTH)[0]:,.0f}x.
Halving the price of tokens halves BOTH columns and moves the ratio not at all.

Finally read the RAG breakdown columns and note which term dominates: generation,
by three orders of magnitude over embedding and index at every corpus size here.
The one-time embedding cost teams agonise over is a rounding error, and the
index storage is smaller still. The only lever that matters is k -- the number of
chunks put in the context on every query -- which is a quality decision
(eq:context-budget) that turns out to be the cost decision too.""")
```

```python {tier=A name=knowledge-freshness}
"""How fast does a frozen model go stale, and when does retrieval overtake it?

A MODEL, clearly labelled as one: a corpus of facts, each with a probability per
month of being superseded. A parametric model answers from its training snapshot;
a RAG system answers from the current corpus but only when retrieval finds the
fact.

The question is not whether RAG wins -- it obviously does eventually -- but HOW
SOON, because that is what decides whether staleness is an architectural problem
or a scheduling one.
"""
import numpy as np

rng = np.random.default_rng(19)

N_FACTS, N_TRIALS = 5000, 40
RECALL_AT_K = 0.85           # the retriever's ceiling, eq:rag-ceiling
PARAM_ACCURACY = 0.92        # accuracy on facts the model DID learn correctly
MONTHS = [0, 3, 6, 12, 18, 24, 36]

# Monthly probability that a given fact is superseded, by domain.
CHURN = {"stable (regulations)": 0.005,
         "moderate (product docs)": 0.03,
         "fast (pricing, staffing)": 0.10}

print(f"{'domain / architecture':<30}{'churn':>8}"
      + "".join(f"{'m' + str(m):>8}" for m in MONTHS))
print("-" * (38 + 8 * len(MONTHS)))

for label, churn in CHURN.items():
    param_row, rag_row = [], []
    for months in MONTHS:
        p_current = (1 - churn) ** months        # fact unchanged since cutoff
        acc_param = np.mean([
            np.mean(rng.random(N_FACTS) < p_current * PARAM_ACCURACY)
            for _ in range(N_TRIALS)])
        acc_rag = np.mean([
            np.mean(rng.random(N_FACTS) < RECALL_AT_K * PARAM_ACCURACY)
            for _ in range(N_TRIALS)])
        param_row.append(acc_param)
        rag_row.append(acc_rag)
    print(f"{label:<30}{churn:>8.3f}"
          + "".join(f"{v:>8.3f}" for v in param_row) + "   parametric")
    print(f"{'':<30}{'':>8}"
          + "".join(f"{v:>8.3f}" for v in rag_row) + "   RAG")
    # First month at which RAG overtakes the frozen model.
    cross = next((m for m, p, r in zip(MONTHS, param_row, rag_row) if r > p), None)
    verdict = f"month {cross}" if cross is not None else "not within 36 months"
    print(f"{'  -> RAG overtakes at ' + verdict:<30}\n")

print(f"""
Read the crossover line, which is the only number in the table that changes a
decision.

At month zero the parametric model WINS in every domain: it answers from a
perfect memory of its training snapshot, while RAG pays the eq:rag-ceiling tax of
{RECALL_AT_K:.0%} recall on every query. Retrieval is not free accuracy -- it
trades a retrieval failure mode for a staleness failure mode.

How long that trade takes to pay off is entirely a property of the DOMAIN, not of
the technology. In the fast-churn row the frozen model is overtaken within a
couple of quarters; in the stable row it stays ahead for years, and a RAG system
there is buying attribution and access control rather than accuracy.

That is the useful decomposition. Freshness is one of the four properties in
section 5.3 and it is the only one with a clock on it -- so measure your corpus's
churn rate before assuming staleness is your problem. Teams frequently build RAG
for freshness when what they actually needed was citations.""")
```

## 9. Practical Example

**The cost model**, and it corrected the framing I expected. The break-even
corpus size comes out at 4,000 tokens — exactly $k \cdot L_c$, and not by
coincidence: below that, RAG would retrieve the whole corpus anyway, so the two
architectures are the same thing.

The honest conclusion is therefore blunter than the usual one. **On cost, RAG
wins as soon as the corpus exceeds the context you would retrieve. There is no
meaningful cost regime in which stuffing is cheaper.** So cost is *not* the
reason to stuff a small corpus — the reasons are simplicity, the absence of
{{fig:rag-stages}}'s four failure stages, and
{{cite:li2024ragvslongcontext}}'s finding that long context is *better* when
everything fits. Those are good reasons. Teams that justify stuffing on cost are
usually right for the wrong reason, and will lose the argument to the first
person who opens a spreadsheet.

Above the break-even the gap opens fast and prices do not touch it, because
{{eq:cost-ratio}}'s $N_c/(kL_c)$ contains no price: measured, 2,500× at ten
million tokens and 25,000× at a hundred million.

The breakdown columns carry a second lesson. **Generation dominates embedding and
index by three orders of magnitude** at every corpus size in the table. Teams
agonise over embedding cost and index storage while the actual bill is $k$ chunks
× every query — which means $k$, chosen for quality reasons via
{{eq:context-budget}}, is also the only cost lever that matters.

**The freshness model.** The crossover is the number that changes a decision, and
the first row of the table is the one people do not expect: **at month zero the
frozen model wins in every domain.** RAG pays {{eq:rag-ceiling}}'s recall tax on
every query from day one. It is not free accuracy; it trades a staleness failure
mode for a retrieval failure mode.

How long that trade takes to pay off is a property of the *domain*. Fast-churn
knowledge overtakes within a couple of quarters; slow-churn knowledge may not for
years — and a RAG system built over a stable corpus is buying **attribution and
access control**, not accuracy, which is a perfectly good reason but a different
one that should be stated.

> **PRODUCTION TIP:** Measure your corpus's churn rate before assuming staleness
> is the problem. A surprising number of RAG projects are motivated by
> "the model is out of date" and would have been better served by scheduled
> fine-tuning plus a citation requirement — or, more often, would have been
> served by admitting that citations were the actual requirement all along.

## 10. Production Considerations

**Build the naive baseline first and measure recall@$k$ before anything else.**
{{eq:rag-ceiling}} caps everything downstream, and the number takes an afternoon
to produce.

**Instrument all four stages from the start.** Ingestion loss rate, chunk
boundary statistics, retrieval recall, and groundedness
({{eq:rag-groundedness}}). Retrofitting stage attribution into a running system
is much harder than building it in, and {{ch:rag-failures}} is unusable without
it.

**Decide the access-control model before the retrieval model.** Filtering by
permission is {{ch:emb-vector-db}}'s pre-filtering problem, its cost depends on
selectivity, and retrofitting it into an index built without it is frequently a
rebuild.

**Treat $k$ as a joint cost and quality parameter** ({{eq:context-budget}}), not
as a number copied from a tutorial.

**Version the corpus, not just the index.** When an answer is wrong six weeks
later, you need to know what the retriever could have seen at the time. Store
document versions and retrieval logs; both are cheap and neither can be
reconstructed.

**Separate the corpus's owner from the system's owner, deliberately.** RAG makes
the corpus a production dependency ({{sec:15-advanced-concepts}}), which means
someone must own its coverage and accuracy — and that person is usually not on
the engineering team. Systems where nobody owns the corpus degrade in a
characteristic way: retrieval metrics stay healthy, because the retriever
faithfully returns the best of what exists, while answer quality falls as the
questions drift away from what anyone wrote down. **No technical metric catches
this**, and the fix is editorial rather than architectural.

**Have an abstention path.** {{ch:llm-hallucination}}'s
{{eq:risk-coverage}} applies directly: when retrieval returns nothing good, "I
don't know" is a correct answer and the system needs a way to produce it. A RAG
system without an abstention path converts retrieval failures into confident
fabrications, which is strictly worse than the bare model it replaced.

## 11. Common Mistakes

**Building RAG for a corpus that fits in the context window.** But note the
reason: {{sec:9-practical-example}} shows it is simplicity and quality, not cost.
Arguing it on cost is how the decision gets reversed by the first person to open
a spreadsheet.

**Fine-tuning to add facts.** {{eq:adaptation-information-ratio}}. Fine-tune for
behaviour, retrieve for facts.

**Optimising prompts before measuring recall@$k$.** Bounded by
{{eq:rag-ceiling}}.

**Assuming RAG reduces hallucination by itself.** It supplies material; whether
the model uses it is a separate question, and an unfaithful generator over
perfect retrieval is a documented failure mode ({{ch:rag-failures}}).

**Maximising $k$.** {{eq:context-budget}} has an interior optimum and
{{eq:presence-times-attention}} says why.

**Treating citations as a UI feature.** They are the verification mechanism
({{ch:rag-generation}}), and an unverified citation is decoration.

**Choosing RAG over long context on capability grounds.** It is a cost and
auditability decision ({{cite:li2024ragvslongcontext}}); saying otherwise
invites a rebuttal that will be correct.

## 12. Failure Modes

The four stages, each with its symptom — this is {{ch:rag-failures}}'s taxonomy
in miniature and it is worth carrying from the first chapter.

**Ingestion loss.** The answer is in a table the parser flattened. Symptom:
consistently wrong on a document *type*. Never appears in retrieval metrics
because the text is not in the index to be retrieved.

**Chunk-boundary loss.** The answer straddles a split. Symptom: retrieval returns
a chunk that is adjacent to the answer.

**Retrieval miss.** Symptom: recall@$k$ below expectation on a query slice —
usually identifier-like queries ({{ch:emb-hybrid}}) or paraphrases.

**Generation failure.** The right chunk was in the context and the model answered
from its parametric knowledge anyway. Symptom: the answer is plausible,
uncited, and contradicts the retrieved text.

**And the meta-failure:** no stage attribution, so every one of the above is
reported as "RAG doesn't work" and debugged by changing the prompt.

## 13. Alternatives

**Long context.** Simpler, better when it fits, catastrophically more expensive
when it does not ({{eq:cost-ratio}}).

**Fine-tuning.** The right tool for behaviour, format, and domain style; the
wrong one for facts.

**Continued pretraining.** Genuinely adds knowledge, at pretraining cost, with no
attribution and no removal path.

**Tool use.** {{ch:llm-function-calling}}: rather than retrieving documents, call
an API that knows. **Structurally better whenever the knowledge is in a system
rather than in prose**, and consistently under-used relative to RAG.

**A search box.** Sometimes the user wanted the documents, not an answer about
them, and the honest version of that product is a search engine.

**Model editing.** Direct weight edits to change one fact. Research-stage,
brittle, and interesting for the same reason RAG is: it attacks the update
problem.

## 14. Evaluation

**Retrieval and generation separately, always.** This is the field's
characteristic error and {{cite:es2023ragas}} exists because of it. Retrieval:
recall@$k$ and precision. Generation: groundedness and answer correctness *given*
the retrieved context.

**Stage attribution on every failure.** Which of the four lost it. Without this,
end-to-end accuracy is a number you cannot act on.

**Groundedness against the retrieved set** ({{eq:rag-groundedness}}) — computable
here, unlike for a bare model.

**Abstention behaviour.** When nothing relevant is retrieved, does the system say
so? Test with deliberately unanswerable questions; a shocking number of systems
have never been asked one.

**Cost per query, decomposed.** {{eq:rag-cost-comparison}}: retrieval, context
tokens, generation.

**Against the honest baselines:** the bare model, and stuffing if the corpus
fits. A RAG system that does not beat both is four failure stages for nothing.

## 15. Advanced Concepts

**{{eq:rag-marginal}} versus {{eq:rag-practical}}.** {{cite:lewis2020rag}}
marginalised over retrieved documents and production takes the arg-max, which
throws away the architecture's tolerance for retrieval error. Nothing forces this
— it is a latency decision — and revisiting it is one of the more interesting
under-explored directions in the part.

**{{cite:guu2020realm}}'s question is still open.** Training the retriever
jointly with the generator is clearly better and clearly expensive, and the
industry uses frozen off-the-shelf retrievers for cost reasons. Whether that is a
permanent equilibrium or a temporary one is genuinely unclear.

**Retrieval is a form of conditioning, and conditioning is what
{{ch:llm-prompting}} said a prompt is.** RAG is therefore *automated prompt
construction with a search engine attached* — which sounds reductive and is
actually the most useful way to think about {{ch:rag-generation}}, because it
means every prompting result transfers.

**The knowledge/behaviour split is not clean.** Retrieved examples change
behaviour (few-shot), and fine-tuning does add some facts. The rule in
{{sec:4-intuitive-explanation}} is a strong heuristic about *reliability*, not a
theorem.

**RAG makes the corpus a production dependency.** Its quality, coverage, and
freshness are now system properties, and the largest RAG quality win is
frequently editorial rather than technical: writing the twenty missing documents.
This is unglamorous, it is not in any paper, and it is often correct.

## 16. Connection to Previous Chapters

{{ch:fm-what-they-are}}'s adaptation-information ratio is why fine-tuning cannot
be the answer for facts. {{ch:llm-hallucination}}'s groundedness becomes
computable here because $\hat{z}$ exists. {{ch:llm-long-context}}'s usable-context
result is half of {{eq:presence-times-attention}}, and {{cite:liu2023lost}}'s
U-shape is the other half. {{ch:emb-reranking}}'s recall ceiling reappears as
{{eq:rag-ceiling}} with the generator in the reranker's place. And
{{ch:llm-routing}}'s cascade is what {{cite:li2024ragvslongcontext}} concludes
with — route between RAG and long context rather than choosing.

## 17. Exercises

1. Derive {{eq:cost-ratio}} from {{eq:rag-cost-comparison}} and explain why no
   token price appears in it.
2. Recompute the break-even in `rag-cost-model` at a tenth of the token price.
   By how much does it move, and why so little?
3. Set $k = 30$ in the same listing. What happens to the ratio, and what does
   {{eq:context-budget}} say about whether that $k$ was a good idea anyway?
4. In `knowledge-freshness`, add a domain at 0.5% monthly churn and find its
   crossover. What retriever recall would be needed to bring the crossover under
   twelve months?
5. Prove {{eq:rag-ceiling}} and state the assumption it makes about the
   generator.
6. Your RAG system is 70% accurate. Recall@10 is 0.72. Where is the remaining
   error, and what is the next measurement?
7. Write the argument for using RAG over a corpus that never changes. Which of
   the four properties are you buying?
8. A team proposes fine-tuning weekly on new documents instead of retrieval.
   Give three specific things that will go wrong, referencing {{part:9}}.

## 18. Interview Questions

1. What does RAG actually change about a system?
2. Why not fine-tune the knowledge in?
3. Long context windows are getting huge. Is RAG obsolete?
4. What bounds a RAG system's accuracy?
5. When would you *not* use RAG?
6. How do you know whether a RAG failure is retrieval or generation?
7. What does RAG give you that has nothing to do with accuracy?
8. Your RAG system hallucinates. First three things you check.
9. How does access control work in a RAG system, and where does it break?
10. What is the most expensive part of a RAG system at scale?

## 19. Research Questions

1. Production takes {{eq:rag-practical}}'s arg-max rather than
   {{eq:rag-marginal}}'s marginal. Is there a cheap approximation to the marginal
   that recovers robustness to retrieval error?
2. Is there a principled way to decide *per query* between parametric and
   retrieved knowledge, before paying for retrieval? This is
   {{ch:llm-routing}}'s difficulty-estimation problem in a new setting.
3. Can the four properties be obtained without relocation — attributable,
   updatable, removable parametric knowledge? Model editing is the attack;
   nothing works reliably.
4. {{eq:context-budget}}'s $\rho$ is model-specific and unmeasured for most
   models. Is there a cheap probe that estimates it?
5. Is joint retriever/generator training ({{cite:guu2020realm}}) worth its cost at
   current model scales, and does the answer change as retrievers get cheaper?

## 20. Chapter Summary

RAG does not add knowledge to a model; it **moves knowledge out of the weights**
and puts a retrieval step in front. That relocation buys four things — freshness,
attribution, access control, and cheap updates — and every one of them is
routine in the non-parametric case and impossible or a training run in the
parametric one. **The asymmetry is the argument, and accuracy is not on the
list.**

Fine-tuning is the wrong tool for facts because
{{eq:adaptation-information-ratio}} puts it orders of magnitude below pretraining
in information supplied: it teaches format reliably and facts poorly. Fine-tune
for behaviour, retrieve for facts.

The cost comparison against stuffing is structural rather than a matter of
prices. {{eq:cost-ratio}}'s $N_c/(kL_c)$ contains no price, so cheaper tokens
move both columns and the ratio not at all — measured, thousands of times at a
ten-million-token corpus, with generation rather than embedding or index
dominating the RAG bill. The break-even is just $k \cdot L_c$ — below it the two architectures are the
same thing — so **the case for stuffing a small corpus is simplicity and quality,
never cost.**

And {{eq:presence-times-attention}} is the argument that survives free tokens:
stuffing maximises presence and degrades attention, retrieval does the reverse.
**Stuffing is implicit selection by attention, unauditable; retrieval is explicit
selection by a component you can measure.**

The costs are real. {{eq:rag-ceiling}} caps accuracy at recall@$k$, which is why
measuring it precedes everything, and the freshness model shows RAG *losing* at
month zero in every domain because that tax is paid from day one. Four stages can
lose a question where a bare model had one — but all four are inspectable, and
that auditability is usually worth more than the quality difference.

## 21. Further Reading

{{cite:lewis2020rag}} for the origin and for {{eq:rag-marginal}} — note how much
of the paper's method production discards.
{{cite:guu2020realm}} for retrieval learned during pretraining, which is the
road not taken.
{{cite:izacard2022atlas}} for the quantitative form of "retrieval substitutes for
parameters".
{{cite:li2024ragvslongcontext}} for the long-context comparison; read its routing
conclusion rather than its headline.
{{cite:gao2023ragsurvey}} for the naive/advanced/modular vocabulary the rest of
this part uses.

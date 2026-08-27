---
id: rag-generation
number: 110
part: XII
tier: full
status: draft
requires: [rag-indexing, rag-chunking, llm-prompting, llm-long-context,
           llm-hallucination, rag-why]
provides: [context-assembly, context-ordering, context-budget-allocation,
           citation-verification, grounded-generation, abstention-in-rag,
           answer-attribution, prompt-template-rag]
citations: [liu2023lost, es2023ragas, ji2023survey, lewis2020rag,
            gao2023ragsurvey, kadavath2022]
---

## 1. Learning Objectives

By the end of this chapter you will be able to treat context assembly as a
constrained optimisation rather than a concatenation, and choose an ordering that
accounts for position effects; allocate a token budget between chunks, history,
and instructions with a defensible rule; write generation instructions that make
abstention a reachable outcome; and — the chapter's central claim — implement
**citation verification**, which turns a citation from a UI decoration into a
measurable guarantee.

## 2. Why This Matters

Retrieval hands you a ranked list. Something has to turn that into a prompt, and
that something is usually a `"\n\n".join(chunks)` written in the first hour of
the project and never revisited.

It is the most consequential under-examined stage in the part. The same chunks,
assembled differently, produce measurably different answers:
{{cite:liu2023lost}} showed models use information at the *edges* of a long
context far better than the middle, so **rank order — best chunk first, then
descending — puts the second-best chunk in the worst possible place** and the
worst chunk in a good one. That is a free quality change and almost nobody makes
it.

The larger claim concerns citations. RAG systems display source links, users
trust them, and in most systems **nothing has checked that the cited document
supports the sentence it is attached to.** The model was asked to cite and it
emitted a citation; that is all. An unverified citation is not evidence — it is
*decoration that looks like evidence*, which is worse than none, because it
converts a user's healthy scepticism into misplaced confidence.
{{sec:9-practical-example}} measures what verification catches.

{{maturity:MATURE}} Prompt construction for RAG is well-trodden practice.
{{maturity:EMERGING}} Citation verification is not standard, is cheap, and is the
highest-value thing in this chapter.

## 3. Prerequisites

{{ch:rag-indexing}} for the retrieved set and its metadata;
{{ch:rag-chunking}} for what a chunk contains; {{ch:llm-prompting}} for prompts
as conditioning; {{ch:llm-long-context}} for the position effects this chapter
exploits; {{ch:llm-hallucination}} for groundedness and abstention;
{{ch:rag-why}} for the context budget.

## 4. Intuitive Explanation

### Assembly is a decision, not a formatting step

Given ten retrieved chunks and a budget, you must decide **which to include, in
what order, with what surrounding text.** Each is a real choice with a measurable
effect.

*Which* is a knapsack problem: chunks have values (relevance) and costs (tokens),
and {{ch:rag-why}}'s {{eq:context-budget}} showed adding more is not monotonically
good, because each additional chunk dilutes the model's attention across the
whole context.

*What order* matters because attention is not uniform over position
({{cite:liu2023lost}}). The naive descending-rank order is actively bad: it
places your second-best evidence in the middle, which is where the model is least
likely to use it.

*What surrounding text* determines whether the model treats the chunks as
authority or suggestion, whether it can say "I don't know", and whether it cites
in a form you can check.

### Citations, and what they are for

Three different things get called a citation, and conflating them is the mistake:

**A UI affordance.** A link the user can click. Useful, and it makes no claim
about correctness.

**A provenance record.** "This answer was generated from documents 3, 7, and 12."
True by construction, since those are what you put in the context — and it says
nothing about which sentence came from which document, or whether any of them
supports it.

**A verified attribution.** "This sentence is supported by this span of document
7, and something checked." Only this one is evidence.

Most systems ship the second and present it as the third. The gap is where users
lose trust in a way that is hard to win back, because they discover it by finding
one confident citation that does not say what the answer claimed.

**Verification closes the gap and it is cheap**: for each claim in the answer,
check whether its cited span actually supports it. {{sec:9-practical-example}}
shows how much of the gap that closes and what it costs.

### Abstention has to be reachable

{{ch:llm-hallucination}} established abstention as a first-class outcome. In RAG
it is more important, because the failure is worse: a model handed irrelevant
context will frequently *use* it — producing an answer that is fluent, cited, and
about the wrong thing. **The citation makes it more credible, not less.**

A system with no abstention path converts retrieval failures into confident
fabrications, which is strictly worse than the bare model it replaced. Making
abstention reachable takes an instruction, an example, and a threshold — and it
is the difference between a system that degrades and one that misleads.

## 5. Formal Explanation

### 5.1 The assembly problem

Given candidates $\{(z_i, r_i, t_i)\}$ with relevance $r_i$ and token cost $t_i$,
budget $B$, and a position-dependent utilisation $\rho(p)$, choose a subset $S$
and an ordering $\pi$:

$$ \max_{S, \pi} \sum_{i \in S} r_i \cdot \rho\big(\pi(i)\big) \quad \text{s.t.} \quad \sum_{i \in S} t_i \leq B $$ (eq:context-assembly)

Two structures worth naming. With $\rho$ constant this is a **0/1 knapsack** —
take chunks in descending $r_i/t_i$ until the budget is exhausted, which is
already better than the usual descending-$r_i$. And with $\rho$ non-constant the
ordering is a separate problem that couples to the selection.

### 5.2 Optimal ordering under a non-uniform profile

Fix $S$. The ordering that maximises {{eq:context-assembly}} is the one pairing
the largest $r_i$ with the largest $\rho(p)$ — a **rearrangement inequality**:

$$ \pi^{*} \text{ assigns the } j\text{-th most relevant chunk to the } j\text{-th best position} $$ (eq:optimal-ordering)

Under {{cite:liu2023lost}}'s U-shape, the best positions are the ends, so the
optimal order is not descending rank but an **outside-in interleave**:

$$ \text{positions in decreasing } \rho: \quad 1,\, n,\, 2,\, n-1,\, 3,\, \dots $$ (eq:u-shape-ordering)

placing chunk 1 first, chunk 2 **last**, chunk 3 second, and so on — burying the
*least* relevant chunks in the middle where they will be least used. This is a
reordering of text you were already going to send: **zero cost, no extra tokens,
no extra calls.**

> **NOTE:** {{eq:optimal-ordering}} depends only on $\rho$ being non-uniform, not
> on its being U-shaped specifically. Measure $\rho$ for your model
> ({{ch:llm-long-context}}) and sort accordingly. If $\rho$ turns out to be
> monotone decreasing, descending rank *is* optimal — but that is a finding, not
> an assumption.

### 5.3 Budget allocation

The context holds more than chunks:

$$ B = \underbrace{B_{\text{sys}}}_{\text{instructions}} + \underbrace{B_{\text{hist}}}_{\text{conversation}} + \underbrace{B_{\text{ctx}}}_{\text{retrieved}} + \underbrace{B_{\text{out}}}_{\text{reserved for the answer}} $$ (eq:budget-decomposition)

$B_{\text{out}}$ is the one that is forgotten, and forgetting it produces
truncated answers under exactly the conditions — a long conversation, many chunks
— where the answer needs to be longest.

The competition between $B_{\text{hist}}$ and $B_{\text{ctx}}$ is the live one in
any multi-turn system, and it has no universal answer. **The defensible rule:
reserve $B_{\text{out}}$ first, cap $B_{\text{hist}}$ at a fixed budget with
summarisation beyond it, and give the remainder to retrieval** — because history
degrades gracefully under summarisation and retrieved evidence does not degrade
gracefully under truncation.

### 5.4 Groundedness and citation verification

{{ch:llm-hallucination}} defined groundedness; RAG makes it computable
({{eq:rag-groundedness}}) because the source set is known. Decompose an answer
into claims $c_1 \dots c_m$, each with a cited span $\sigma(c_j)$:

$$ \text{verified}(c_j) = \Ind\big[\, \sigma(c_j) \text{ entails } c_j \,\big], \qquad \text{attribution rate} = \frac{1}{m}\sum_j \text{verified}(c_j) $$ (eq:attribution-rate)

**This is checkable without ground truth**, which is what makes it deployable:
you are not asking whether the claim is *true*, only whether the cited text
supports it. {{ch:llm-hallucination}}'s point that groundedness is measurable
while truth is not, arriving with an implementation.

The verifier can be token overlap, an NLI model, or an LLM judge
({{cite:es2023ragas}}), with the usual cost/accuracy trade. All three are far
cheaper than the generation they check.

### 5.5 What verification buys, precisely

Let $\alpha$ be the fraction of claims that are genuinely unsupported, and let the
verifier have true-positive rate $\tau$ and false-positive rate $\phi$. After
filtering out claims the verifier rejects:

$$ \text{unsupported rate after} = \frac{\alpha(1 - \tau)}{\alpha(1-\tau) + (1-\alpha)(1 - \phi)} $$ (eq:post-verification-rate)

and the fraction of good claims wrongly dropped is $\phi$. **This is
{{eq:risk-coverage}} again** — the abstention trade-off from
{{ch:llm-hallucination}}, applied at claim granularity instead of answer
granularity, which is a strictly better place to apply it because the system can
drop one sentence rather than refuse an entire answer.

## 6. Mathematical Foundation

### 6.1 Why more chunks stop helping

{{eq:context-assembly}} makes {{ch:rag-why}}'s claim precise. Adding chunk $k+1$
changes the objective by

$$ \Delta = \underbrace{r_{k+1}\rho(p_{k+1})}_{\text{new evidence}} \;-\; \underbrace{\sum_{i \leq k} r_i \big[\rho_k(p_i) - \rho_{k+1}(p_i)\big]}_{\text{dilution of existing evidence}} $$ (eq:marginal-chunk-value)

The first term shrinks as $k$ grows, because chunks are retrieved in descending
relevance. The second grows, because a longer context spreads $\rho$ thinner and
pushes existing chunks toward the middle. **There is a $k$ where $\Delta$ turns
negative, and past it retrieving more actively hurts.**

That is the interior optimum {{ch:rag-why}} promised, with a mechanism attached.
It also explains a common and confusing observation: raising $k$ improves
retrieval recall and degrades answer quality *simultaneously*, which looks
contradictory until you notice they are different terms of the same expression.

### 6.2 The verification asymmetry

Why verification is unusually favourable, and worth doing even with a mediocre
verifier. The two errors have very different costs:

$$ \text{false negative (drop a good claim)} \to \text{a less complete answer} $$
$$ \text{false positive (keep a bad claim)} \to \text{a confident cited falsehood} $$ (eq:verification-asymmetry)

Since {{eq:post-verification-rate}}'s numerator falls with $\tau$ while the
penalty for large $\phi$ is only completeness, **the operating point should lean
aggressive**: prefer dropping borderline claims. This is the opposite of the usual
instinct with a classifier, and it follows from the asymmetry rather than from
taste.

**But note the region where no judgement is required at all.** If the verifier's
score distribution for genuinely supported claims has a floor $v_{\min}$, then
every threshold below it satisfies $\phi = 0$:

$$ \theta < v_{\min} \;\Longrightarrow\; \phi = 0 \;\text{ and }\; \tau > 0 $$ (eq:free-verification-region)

catching some fabrications at literally zero cost in completeness.
{{sec:9-practical-example}} measures how much that free region is worth, and the
answer is enough to make verification worth deploying even for a team unwilling
to drop a single correct claim.

### 6.3 Instructions are conditioning, not rules

{{ch:llm-prompting}}'s {{eq:prompt-conditioning}} applies with full force. "Only
use the provided context" does not *constrain* the model — it conditions it, and
{{eq:next-token-distribution}} still assigns non-zero probability to every
continuation, including one drawn from parametric memory.

$$ \Prob[\text{ungrounded claim}] > 0 \quad \text{regardless of the instruction} $$ (eq:instructions-not-guarantees)

**So instructions reduce a rate and verification provides a guarantee.** A system
whose groundedness strategy is entirely in the prompt has a rate it has not
measured. This is the same distinction {{ch:llm-structured-output}} drew between
prompting for JSON and masking for it — a theme of the book by now, and it lands
here on the most consequential property RAG claims to provide.

## 7. Internal Mechanics

```mermaid {#fig:context-assembly caption="From ranked chunks to a verified answer. The two boxes below the main path are the ones most systems omit: the reorder is free, and the verifier is the only step that makes a citation mean anything."}
flowchart TD
    R["ranked chunks + metadata"] --> SEL["select under budget<br/>(eq:context-assembly)"]
    SEL --> DED["deduplicate<br/>near-identical chunks"]
    DED --> ORD["reorder outside-in<br/>(eq:u-shape-ordering)"]
    ORD --> FMT["format: id, source,<br/>heading path, text"]
    FMT --> P["prompt: instructions +<br/>context + history + question"]
    P --> G["generate with<br/>inline citations"]
    G --> V["verify each claim<br/>against its cited span<br/>(eq:attribution-rate)"]
    V --> A["answer + checked citations"]
    V -.->|"attribution rate low"| AB["abstain or flag"]
```

### 7.1 Formatting a chunk

Each chunk needs a **stable, short identifier the model can cite** and enough
provenance for the citation to be checkable:

```text
[3] Q3 Report → EMEA → Logistics (2024-10-02)
Regional throughput improved by 12% following the depot consolidation...
```

Three things earn their tokens. The **bracketed id** gives the model a cheap
citation token — much more reliable than asking it to reproduce a title. The
**heading path** ({{ch:rag-ingestion}}) supplies the context the chunk lacks
alone. The **date** lets the model prefer recent evidence and notice conflicts.

Use numeric ids rather than document titles: they are one token, they are
unambiguous, and they are trivially resolvable back to a source. Asking a model
to cite `"the Q3 EMEA logistics report"` invites paraphrase, and a paraphrased
citation cannot be verified automatically.

### 7.2 The instruction block

What actually changes behaviour, in rough order of effect:

- **Cite inline, per claim**, in a fixed format. This is what makes
  {{eq:attribution-rate}} computable at all — an answer with citations at the end
  cannot be verified per claim.
- **An explicit abstention instruction with a phrase to use.** "If the context
  does not contain the answer, say *I don't have that information.*" Naming the
  output makes it reachable ({{ch:llm-prompting}}).
- **A conflict instruction.** Retrieved chunks disagree more often than teams
  expect — superseded policies, regional variants — and without instruction the
  model silently picks one.
- **Scope.** Whether parametric knowledge may supplement the context. Both
  answers are defensible; leaving it unstated is not.

### 7.3 Deduplication before assembly

{{ch:emb-vector-db}}'s {{eq:duplicate-slot-loss}} arrives here as wasted budget.
Overlapping chunks ({{ch:rag-chunking}}) and near-duplicate documents
({{ch:rag-ingestion}}) both produce retrieved sets where three of eight slots say
the same thing.

Deduplicate *after* retrieval and *before* assembly, by content hash for exact
matches and by high cosine similarity for near ones. **This is the cheapest way
to increase effective $k$ without increasing the budget** — and, by
{{eq:marginal-chunk-value}}, without the dilution that raising real $k$ would
cost.

## 8. Implementation

```python {tier=A name=context-ordering}
"""Reordering the same chunks, for free.

A model does not use context uniformly by position (ch:llm-long-context). Given
a position-utilisation profile rho, eq:optimal-ordering says pair the most
relevant chunk with the best position -- which under a U-shaped rho means an
outside-in interleave, NOT descending rank.

Nothing here changes which chunks are sent or how many tokens are spent. It is a
permutation of text you were already going to send.

The rho profiles below are INPUTS, stated explicitly rather than fitted: a flat
control, a U-shape of the kind reported for long-context models, and a
monotone-decreasing alternative. The conclusion is a function of rho, and the
listing shows how it changes when rho does.
"""
import numpy as np

rng = np.random.default_rng(31)

N_CHUNK, N_TRIAL = 12, 4000

PROFILES = {
    "flat": lambda n: np.ones(n),
    "U-shaped": lambda n: 0.45 + 0.55 * np.abs(np.linspace(-1, 1, n)) ** 1.6,
    "monotone decreasing": lambda n: np.linspace(1.0, 0.35, n),
}


def descending(order_by_relevance):
    """The usual: best chunk first, then descending."""
    return list(order_by_relevance)


def outside_in(order_by_relevance):
    """eq:u-shape-ordering: 1st, last, 2nd, second-last, ... -- the most
    relevant chunks at the ends, the least in the middle."""
    n = len(order_by_relevance)
    slots = [None] * n
    lo, hi = 0, n - 1
    for j, c in enumerate(order_by_relevance):
        if j % 2 == 0:
            slots[lo] = c
            lo += 1
        else:
            slots[hi] = c
            hi -= 1
    return slots


def shuffled(order_by_relevance):
    return list(rng.permutation(order_by_relevance))


STRATEGIES = {"descending rank": descending,
              "outside-in": outside_in,
              "random": shuffled}


# One fixed set of relevance draws, shared by every strategy and profile, so
# that any difference between strategies is attributable to placement alone.
DRAWS = np.sort(rng.random((N_TRIAL, N_CHUNK)), axis=1)[:, ::-1]


def expected_utilisation(strategy, rho):
    """Sum of relevance x position-utilisation over the shared relevance draws --
    the objective of eq:context-assembly with the subset held fixed."""
    totals = []
    for rel in DRAWS:
        placed = strategy(list(range(N_CHUNK)))       # placed[p] = chunk index
        totals.append(sum(rel[c] * rho[p] for p, c in enumerate(placed)))
    return float(np.mean(totals))


for pname, pfn in PROFILES.items():
    rho = pfn(N_CHUNK)
    print(f"\n{pname} profile   rho = "
          + " ".join(f"{v:.2f}" for v in rho))
    base = None
    for sname, sfn in STRATEGIES.items():
        val = expected_utilisation(sfn, rho)
        if base is None:
            base = val
        print(f"   {sname:<20}{val:>9.4f}{(val / base - 1) * 100:>+9.2f}%")

print("""
Under the FLAT profile the deterministic strategies score IDENTICALLY -- not
approximately, exactly -- because with rho constant the objective is just the sum
of relevances and placement cannot change it. That is the control: if position
does not matter, ordering does not matter, and every difference in the blocks
below is attributable to rho alone.

Under the U-SHAPED profile the outside-in order beats descending rank. The
mechanism is eq:optimal-ordering, a rearrangement inequality: descending rank
puts the SECOND most relevant chunk in position two, which under a U-shape is
among the worst places in the context. Outside-in puts it last instead -- one of
the best -- and pushes the least relevant chunks into the middle where their low
utilisation costs least.

Note that random ordering scores close to descending rank. That is worth
knowing on its own: descending rank is not a good order that outside-in improves
slightly, it is a MEDIOCRE order that happens to be the obvious one. Sorting by
relevance feels right and, under a U-shape, buys almost nothing over shuffling.

And the gain is free. No extra tokens, no extra calls, no model change -- a
permutation of text already being sent.

The MONOTONE DECREASING block is the honest control on the recommendation. If
your model's rho falls with position rather than U-shaping, descending rank IS
optimal and outside-in is a mistake. eq:optimal-ordering depends on rho being
non-uniform, not on its shape, so measure rho for your model (ch:llm-long-context)
rather than adopting a conclusion from a paper about a different one.""")
```

```python {tier=A name=citation-verification}
"""An unverified citation is decoration. Verification is what makes it evidence.

Every claim in a generated answer carries a citation. Some claims are genuinely
supported by the cited chunk; some are not -- drawn from parametric memory, from a
different chunk, or fabricated. Nothing in the generation step distinguishes
them, because the model was asked to cite and it emitted a citation.

We simulate answers at a known unsupported rate, run a verifier over each claim,
and measure eq:post-verification-rate: what survives, and at what cost in dropped
good claims. The verifier is a lexical-overlap proxy; a real one is an NLI model
or an LLM judge, both stronger and both far cheaper than the generation.
"""
import numpy as np

rng = np.random.default_rng(17)

N_ANSWER, CLAIMS_PER_ANSWER, VOCAB = 1500, 6, 400
UNSUPPORTED_RATE = 0.22        # fraction of claims not entailed by their citation
CHUNK_LEN, CLAIM_LEN = 60, 12


def make_case():
    """One claim with its cited chunk.

    A SUPPORTED claim is a PARAPHRASE of the chunk, not a copy of it -- so it
    shares most of the chunk's content words and introduces some of its own.
    An UNSUPPORTED claim is about the right topic and says something the chunk
    does not, so it shares fewer.

    The two distributions therefore OVERLAP, which is what makes the threshold
    sweep a real trade-off rather than an exercise. A verifier that separated
    them perfectly would not need a threshold, and would not resemble any
    verifier you can build.
    """
    chunk = rng.choice(VOCAB, CHUNK_LEN, replace=False)
    outside = np.setdiff1d(np.arange(VOCAB), chunk)
    supported = rng.random() > UNSUPPORTED_RATE
    if supported:
        n_shared = int(rng.integers(5, CLAIM_LEN + 1))    # 42%-100% overlap
    else:
        n_shared = int(rng.integers(2, 11))               # 17%-83% overlap
    claim = np.concatenate([
        rng.choice(chunk, n_shared, replace=False),
        rng.choice(outside, CLAIM_LEN - n_shared, replace=False)])
    return set(chunk.tolist()), set(claim.tolist()), supported


def coverage(chunk, claim):
    """Fraction of the claim's content present in the cited span. A crude but
    real verifier: an NLI model or LLM judge does the same job better."""
    return len(claim & chunk) / len(claim)


cases = [make_case() for _ in range(N_ANSWER * CLAIMS_PER_ANSWER)]
scores = np.array([coverage(c, m) for c, m, _ in cases])
truth = np.array([s for _, _, s in cases])

baseline = 1 - truth.mean()
print(f"claims: {len(cases):,}   genuinely unsupported: {baseline:.1%}")
print(f"\n{'threshold':>10}{'claims kept':>13}{'unsupported kept':>18}"
      f"{'good dropped':>14}{'unsupported caught':>20}")
print("-" * 76)

for thr in [0.0, 0.40, 0.50, 0.58, 0.67, 0.75, 0.84, 0.92]:
    keep = scores >= thr
    if keep.sum() == 0:
        continue
    kept_rate = keep.mean()
    unsupported_kept = (~truth[keep]).mean()
    good_dropped = (truth & ~keep).sum() / truth.sum()
    caught = (~truth & ~keep).sum() / (~truth).sum()
    print(f"{thr:>10.2f}{kept_rate:>13.1%}{unsupported_kept:>18.1%}"
          f"{good_dropped:>14.1%}{caught:>20.1%}")

print(f"""
The threshold=0.00 row is the system almost everyone ships: every claim is kept,
every citation is displayed, and {baseline:.0%} of what the user reads is a
confident statement attached to a source that does not support it. Nothing about
the output signals which ones. The citation is not merely uninformative there --
it is actively harmful, because it converts a reader's healthy scepticism into
misplaced confidence.

Now walk down the threshold column, and note that it has two regimes rather than
one.

The first rows are a FREE REGION. Up to the lowest coverage any genuine
paraphrase achieves, the filter catches a third of the fabrications and drops
nothing -- because no supported claim scores that low. Every system should take
this region unconditionally; it is a pure gain and it requires no judgement about
how to value the trade.

Past it the trade turns, and turns UNFAVOURABLY: each further point of
unsupported claims caught costs several points of good claims dropped. That is
worth stating plainly, because the tidy story would be that the curve stays
favourable throughout and it does not. Reaching zero unsupported claims here
means discarding three quarters of the good ones, which is not a system anyone
would ship.

So the recommendation is conditional rather than universal. Take the free region
always. Beyond it, eq:verification-asymmetry is the argument for leaning
aggressive -- a dropped good claim costs COMPLETENESS while a kept bad claim
costs a confident cited falsehood, and those are not comparable harms -- but it
is a genuine trade with a genuine cost, and where you sit on it is a product
decision about how much incompleteness your users will tolerate.

Note also what this measurement does NOT require: no ground truth about whether
the claim is TRUE. It asks only whether the cited text supports it, which is
checkable from material you already have. That is ch:llm-hallucination's point
that groundedness is measurable while truth is not, arriving with an
implementation -- and it is why this is deployable as a runtime filter rather
than only as an offline evaluation.

The verifier here is deliberately crude -- lexical overlap. An NLI model or an LLM
judge does the same job substantially better and still costs a fraction of the
generation being checked. The reason to show the crude version is that even it
separates the two populations well, so the usual objection -- that verification
is too expensive or too unreliable to be worth it -- does not survive contact with
the numbers.""")
```

## 9. Practical Example

**Ordering.** Under a flat profile every strategy scores identically, which is
the control that makes the rest interpretable: if position does not matter,
ordering does not matter.

Under a U-shaped profile the outside-in order beats descending rank, and the
mechanism is {{eq:optimal-ordering}}. Descending rank puts your **second-best
evidence in position two**, which under a U-shape is among the worst places in
the context; outside-in puts it last instead — one of the best — and pushes the
weakest chunks into the middle where low utilisation costs least.

Measured, outside-in scores **14.9% above descending rank** under the U-shaped
profile — and random ordering scores within half a percent of descending. That
second fact is the one worth pausing on: **descending rank is not a good order
that outside-in slightly improves; it is a mediocre order that happens to be the
obvious one.** Sorting by relevance feels right and, under a U-shape, buys almost
nothing over shuffling.

And the gain is free: no extra tokens, no extra calls, no model change, just a
permutation of text already being sent.

The monotone-decreasing block is the honest control on the recommendation. **If
your model's $\rho$ falls with position rather than U-shaping, descending rank is
optimal and outside-in is a mistake** — measured, it loses 12.0% there. {{eq:optimal-ordering}} needs $\rho$ to be
non-uniform, not U-shaped — so measure $\rho$ for your model
({{ch:llm-long-context}}) rather than importing a conclusion about a different
one.

**Citation verification.** The `threshold = 0.00` row is the system almost
everyone ships: every claim kept, every citation displayed, and **22% of what the
user reads is a confident statement attached to a source that does not support
it** — with nothing in the output distinguishing those from the rest.

Walking down the threshold reveals **two regimes, not one** — and this corrected
what I expected to find.

The first is a **free region**: at a threshold below the lowest coverage any
genuine paraphrase achieves, the filter catches **34% of the fabrications while
dropping zero good claims.** No supported claim scores that low, so there is
nothing to lose. Every system should take this region unconditionally.

Past it the trade turns, and turns *unfavourably*: each further point of
unsupported claims caught costs several points of good claims. Reaching zero
unsupported claims means discarding three quarters of the good ones — not a
system anyone would ship. **The tidy story would be that the curve stays
favourable throughout, and it does not.**

So the recommendation is conditional. Take the free region always.
Beyond it, {{eq:verification-asymmetry}} argues for leaning aggressive — a
dropped good claim costs completeness while a kept bad claim costs a confident
cited falsehood, and those are not comparable harms — but it is a real trade with
a real cost, and where you sit on it is a product decision about tolerable
incompleteness, not a fact about verification.

> **IMPORTANT:** This measurement requires **no ground truth about whether the
> claim is true.** It asks only whether the cited text supports it, using
> material already in hand. That is {{ch:llm-hallucination}}'s groundedness point
> arriving with an implementation, and it is why verification is deployable as a
> runtime filter rather than only as an offline evaluation.

The verifier here is deliberately crude — lexical overlap. An NLI model or an LLM
judge does the job substantially better at a fraction of the generation cost. The
reason to show the crude version is that **even it separates the populations
well**, so the standard objection that verification is too expensive or too
unreliable does not survive the numbers.

## 10. Production Considerations

**Reserve output tokens first** ({{eq:budget-decomposition}}). Truncated answers
appear exactly when the answer needed to be longest.

**Deduplicate retrieved chunks before assembly.** The cheapest way to raise
effective $k$ without paying {{eq:marginal-chunk-value}}'s dilution.

**Measure $\rho$ for your model, then order accordingly.** A needle test at your
typical context length ({{ch:llm-long-context}}) is an afternoon and it decides
{{eq:optimal-ordering}}.

**Use short numeric chunk ids.** One token, unambiguous, resolvable, and
verifiable — unlike a paraphrasable title.

**Verify citations at runtime and act on the result** — drop the claim, flag it,
or abstain. Logging the attribution rate without acting on it is measurement
theatre.

**Log the attribution rate as a first-class metric**, per query and in aggregate.
It is the closest thing RAG has to a correctness signal that needs no labels.

**Give abstention a named phrase and test it.** Send deliberately unanswerable
questions in CI; a system that has never been asked one will not handle one.

**Log the assembled prompt, not just the chunk ids.** When an answer is wrong six
weeks later, the exact bytes the model saw are what you need, and they cannot be
reconstructed from ids once chunking or formatting has changed.

## 11. Common Mistakes

**Concatenating in rank order without measuring $\rho$.**

**Displaying citations nobody checked.** {{eq:attribution-rate}}: this is the
chapter's central error, and it is worse than omitting them.

**Filling the context because it is available.**
{{eq:marginal-chunk-value}} turns negative.

**No abstention path.** Converts retrieval failures into confident fabrications.

**Citations at the end of the answer rather than per claim.** Unverifiable by
construction.

**Forgetting $B_{\text{out}}$.**

**Assuming an instruction is a guarantee.**
{{eq:instructions-not-guarantees}} — prompting reduces a rate, verification
provides a guarantee.

**Letting history crowd out retrieval.** History summarises gracefully; evidence
does not truncate gracefully.

## 12. Failure Modes

**Confident ungrounded answer with a citation.** The worst failure in RAG,
because the citation *increases* credibility. Only verification catches it.

**Lost-in-the-middle miss.** The right chunk was in the context, in position
five of ten, and unused. Symptom: the answer improves when $k$ is *reduced*,
which is diagnostic.

**Context overflow silently truncating evidence.** The last chunks are dropped by
the client library and nothing reports it.

**Answering from parametric memory despite good context.** Symptom: the answer is
plausible, uncited, and contradicts a retrieved chunk.

**Conflict resolved silently.** Two chunks disagree; the model picks one and
presents it as settled.

**Citation drift after re-chunking.** Stored offsets point at the wrong span, so
verification fails on correct claims — and the team disables verification rather
than fixing offsets.

**Abstention that never fires** because the instruction exists and the threshold
does not.

## 13. Alternatives

**Generate then retrieve to verify.** Answer first, then retrieve evidence for
each claim and drop the unsupported ones. Slower, and it makes attribution the
*primary* mechanism rather than a check.

**Extractive answering.** Return the retrieved spans rather than a generated
answer. Perfectly grounded by construction, less fluent, and correct more often
than product instinct allows — especially for high-stakes lookups.

**Structured output** ({{ch:llm-structured-output}}). Force a schema of
`{claim, citation}` pairs so {{eq:attribution-rate}} is computable without
parsing prose. Cheap and under-used.

**Per-chunk summarisation before assembly.** Compress each chunk to what is
relevant to the query, fitting more evidence in the budget — at the cost of a
call per chunk and a new place to lose information.

**Long context, no assembly.** Send everything and let attention select
({{cite:li2024ragvslongcontext}}), accepting {{ch:rag-why}}'s cost and the loss of
auditable selection.

**Iterative generation with retrieval mid-answer**
({{ch:rag-agentic}}), which replaces one assembly decision with several.

## 14. Evaluation

**Attribution rate** ({{eq:attribution-rate}}) — the headline metric of this
chapter, needing no labels.

**Groundedness and answer relevance** ({{cite:es2023ragas}}), with the LLM-judge
caveats of {{part:25}}.

**Position sensitivity**: hold the chunk set fixed, permute the order, and
measure the spread. It is the direct measurement of $\rho$'s effect on your
actual workload.

**Abstention rate on unanswerable questions**, which should be high, and on
answerable ones, which should be low. Both, or the metric is meaningless.

**Answer correctness given perfect context** — supply the gold chunk and measure.
This isolates generation from retrieval and is the only way to know which stage
{{ch:rag-failures}} should blame.

**Context utilisation**: how many of the supplied chunks are cited at all.
Consistently low utilisation means $k$ is too high and
{{eq:marginal-chunk-value}} has gone negative.

## 15. Advanced Concepts

**Assembly is prompt engineering with a search engine attached.**
{{ch:rag-why}}'s framing, cashed out: everything {{ch:llm-prompting}} established
about conditioning, ordering effects, and instruction sensitivity applies here,
and the RAG literature largely rediscovers it.

**Claim-level abstention beats answer-level abstention.**
{{eq:post-verification-rate}} operates per claim, so a system can drop one
sentence and keep the rest — a strictly finer instrument than
{{ch:llm-hallucination}}'s all-or-nothing refusal, and it is available for free
once citations are per claim.

**Verification changes what generation is for.** If every claim is checked, the
generator's job is *fluent recombination of supplied evidence* rather than
knowledge, which argues for a smaller and cheaper model than teams typically use
— {{ch:llm-routing}}'s cascade, applied to the generation stage.

**Conflicting evidence is under-specified in almost every system.** Retrieved
chunks disagree regularly and the correct behaviour is a product decision — prefer
recent, prefer authoritative, surface both — that is almost never made
explicitly, so it is made implicitly by the model.

**The citation format is a contract with three parties**: the model must emit it,
the verifier must parse it, and the user must be able to follow it. Optimising
for any one alone breaks the others, and prose citations optimise for the third
while destroying the second.

## 16. Connection to Previous Chapters

{{ch:llm-long-context}}'s position effects are what {{eq:optimal-ordering}}
exploits, and its usable-context result is why {{eq:marginal-chunk-value}} turns
negative. {{ch:llm-prompting}}'s conditioning framing is why
{{eq:instructions-not-guarantees}} holds, and
{{ch:llm-structured-output}}'s prompt-versus-mask distinction is the same shape as
this chapter's instruction-versus-verification one.
{{ch:llm-hallucination}}'s groundedness becomes an implementation here, and its
risk–coverage curve reappears at claim granularity.
{{ch:emb-vector-db}}'s duplicate-crowding equation is why deduplication precedes
assembly.

## 17. Exercises

1. Prove {{eq:optimal-ordering}} from the rearrangement inequality, and state
   what it assumes about $\rho$ being independent of content.
2. Derive the $k$ at which {{eq:marginal-chunk-value}} turns negative for
   $\rho(p) = 1/(1 + \beta n)$ and geometrically decaying relevance.
3. In `context-ordering`, add a profile with a single sharp peak at position one.
   What is the optimal ordering, and does outside-in still help?
4. Modify the same listing so relevance is drawn with heavy ties. Does the gain
   from reordering grow or shrink, and why?
5. In `citation-verification`, raise `UNSUPPORTED_RATE` to 0.4. How does the
   optimal threshold move under {{eq:verification-asymmetry}}?
6. Replace the lexical verifier with one that also requires the claim's numbers
   to appear in the chunk. What does that fix, and what does it miss?
7. Using {{eq:post-verification-rate}}, compute the surviving unsupported rate
   for $\alpha = 0.2$, $\tau = 0.8$, $\phi = 0.1$.
8. Design the position-sensitivity test for a live system: what do you permute,
   what do you hold fixed, and what do you measure?

## 18. Interview Questions

1. How do you order retrieved chunks in the prompt, and why?
2. What is wrong with a citation nobody checked?
3. How would you verify a citation automatically?
4. Why does adding more retrieved chunks eventually hurt?
5. How do you make a RAG system say "I don't know"?
6. Two retrieved chunks contradict each other. What should happen?
7. What is the difference between an instruction and a guarantee?
8. Your answer is right but cites the wrong source. Does that matter?
9. How do you budget the context window in a multi-turn RAG system?
10. How would you tell whether a wrong answer was retrieval's fault or
    generation's?

## 19. Research Questions

1. $\rho$ is model-specific, context-length-dependent, and rarely measured. Is
   there a cheap probe that estimates it well enough to drive
   {{eq:optimal-ordering}} automatically?
2. {{eq:optimal-ordering}} assumes position utilisation is independent of
   content. It is not — related chunks placed together may reinforce. What does
   the coupled problem look like?
3. Can claim-level verification be made differentiable and trained jointly with
   generation, rather than bolted on afterwards?
4. Is there a principled policy for conflicting evidence that does not reduce to
   a hand-written preference order?
5. If verification is reliable, how small can the generator be before quality
   degrades? Nobody has drawn that curve, and it determines the economics of the
   whole architecture.

## 20. Chapter Summary

Context assembly is a **constrained optimisation** ({{eq:context-assembly}}), not
a concatenation: which chunks, in what order, with what framing, under a budget.

**Ordering is free quality.** {{eq:optimal-ordering}} is a rearrangement
inequality — pair the most relevant chunk with the best position — and under a
U-shaped profile that means an outside-in interleave rather than descending rank,
which otherwise buries your second-best evidence in the worst position in the
context. Measured, outside-in beats descending under a U-shape and *loses* under
a monotone profile, so **measure $\rho$ rather than importing a conclusion.**

**More chunks stop helping, with a mechanism.**
{{eq:marginal-chunk-value}}: new evidence shrinks while dilution of existing
evidence grows, so there is a $k$ past which retrieving more hurts — which is why
recall and answer quality can improve and degrade together.

**And the central claim: an unverified citation is decoration.** Most systems
display provenance and present it as attribution. Measured at a 22% unsupported
rate, the shipped configuration hands users confident statements attached to
sources that do not support them, with nothing marking which. Verification
({{eq:attribution-rate}}) closes this, needs **no ground truth about truth** —
only whether the cited span supports the claim — and separates the populations
well even with a crude lexical verifier. The threshold sweep has two regimes: a **free region**
({{eq:free-verification-region}}) below the floor of the supported distribution,
catching 34% of fabrications at zero cost in completeness — which every system
should take unconditionally — and beyond it a genuine trade where each further
point caught costs several points of good claims.
{{eq:verification-asymmetry}} argues for leaning aggressive there, because a
dropped good claim costs completeness while a kept bad one costs a confident
cited falsehood, but it is a product decision rather than a free lunch.

Underneath all of it, {{eq:instructions-not-guarantees}}: **prompting reduces a
rate, verification provides a guarantee** — the same distinction
{{ch:llm-structured-output}} drew for JSON, arriving on the property RAG most
claims to provide.

## 21. Further Reading

{{cite:liu2023lost}} for the position effects behind {{eq:optimal-ordering}} —
and read it as a *measurement of one model family*, which is how this chapter
uses it.
{{cite:es2023ragas}} for reference-free groundedness metrics and for a practical
verifier design.
{{cite:ji2023survey}} for the hallucination taxonomy that citation verification
addresses one branch of.
{{cite:lewis2020rag}} for how thin the original assembly step was — concatenation
in rank order — and how little has changed in most systems since.
{{cite:kadavath2022}} for models' own confidence signals, which are an
alternative route to the abstention decision this chapter reaches by
verification.

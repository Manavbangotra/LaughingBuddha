---
id: rag-query-understanding
number: 111
part: XII
tier: full
status: draft
requires: [rag-generation, rag-indexing, emb-what-they-are, emb-hybrid,
           llm-prompting, rag-why]
provides: [query-rewriting, hypothetical-document-embedding, query-expansion,
           multi-query-retrieval, query-decomposition, query-classification,
           conversational-query-resolution, retrieval-key]
citations: [gao2023hyde, ma2023rewrite, cormack2009rrf, karpukhin2020dpr,
            liu2023lost, gao2023ragsurvey, wang2022e5]
---

## 1. Learning Objectives

By the end of this chapter you will be able to explain why a question is a poor
retrieval key, derived from the asymmetry {{ch:emb-what-they-are}} identified;
implement and measure HyDE, and demonstrate the counterintuitive result that the
hypothetical document's *factual accuracy is irrelevant*; decompose multi-hop
questions and quantify what decomposition buys against what it costs; resolve
conversational references before retrieval; and decide which of these techniques
a given system actually needs.

## 2. Why This Matters

Every chapter so far has taken the query as given. This one asks whether the text
the user typed is the right thing to search with, and the answer is usually no.

The reason is structural rather than incidental. {{ch:emb-what-they-are}}
established that relevance is a relation between a *question* and an *answer*,
and that these look nothing alike:

> *"why is my build slow"* — the document that answers it says
> *"incremental compilation is disabled when the output directory is on a
> network mount"*, and shares almost no words with the question.

An embedding model trained to place similar texts near each other is being asked
to place *dissimilar* texts near each other. That is what the asymmetric training
of {{cite:karpukhin2020dpr}} and the query/passage prefixes of
{{cite:wang2022e5}} exist to patch, and they patch it imperfectly.

Query transformation attacks the same gap from the other side: **change the query
until it looks like the thing you are searching for.** {{cite:gao2023hyde}}'s
version is startling and {{sec:9-practical-example}} confirms it — the
transformed query works *even when it is factually wrong*, because it is being
used as a retrieval key rather than as an answer.

{{maturity:MATURE}} Query rewriting and expansion are decades old in information
retrieval. {{maturity:EMERGING}} LLM-based rewriting and decomposition are recent,
effective, and their cost is routinely omitted from the papers that report them.

## 3. Prerequisites

{{ch:emb-what-they-are}} for the query/document asymmetry and the prefix
convention; {{ch:emb-hybrid}} for what lexical retrieval does with a reformulated
query and for the fusion caveats; {{ch:rag-indexing}} for extracting constraints
from a query; {{ch:llm-prompting}} for the generation step that produces
rewrites; {{ch:rag-generation}} for what happens downstream.

## 4. Intuitive Explanation

### The query is not the target

Four ways the text a user types differs from the text that answers them, in
rough order of how much damage each does:

**Vocabulary.** Users write symptoms; documents describe causes. *"page won't
load"* against *"HTTP 504 gateway timeout"*.

**Length and form.** A five-word question against a two-hundred-word passage.
Even with matched vocabulary, {{ch:emb-what-they-are}}'s geometry places a short
interrogative fragment and a long declarative passage in different regions.

**Under-specification.** *"what's the limit"* — which limit, whose, when. The
constraints of {{ch:rag-indexing}} are implicit in the user's head and absent from
the text.

**Compression.** *"how do I fix the thing we discussed"* — a conversational
reference carrying most of its meaning outside the message.

Each is a different problem needing a different transformation, which is why
"query understanding" is a stage with several components rather than one trick.

### HyDE, and why it is stranger than it sounds

The most counterintuitive idea in the part. Instead of searching with the
question, ask a language model to *write the answer* — then search with **that**.

```text
query:        "why is my build slow"
hypothetical: "Build times increase substantially when incremental compilation
               is disabled. This commonly occurs when the output directory is
               located on a network mount, causing timestamp comparisons to fail..."
```

Now search with the hypothetical. It has answer-shaped vocabulary, answer-shaped
length, and answer-shaped form, so it lands where answers live.

**And here is the part that surprises people: it does not matter whether the
hypothetical is true.** The model may invent a cause that is wrong. The
hypothetical is never shown to the user and never used as an answer — it is
*only a retrieval key*, and a key needs to be shaped like the lock, not correct.

{{sec:9-practical-example}} tests exactly this by generating hypotheticals with
deliberately wrong facts and measuring whether retrieval degrades. **This is the
kind of claim that should be checked rather than repeated**, which is why the
listing checks it.

### Multi-query and decomposition

Two related moves with different justifications.

**Multi-query** issues several paraphrases of one question and fuses the results,
hedging against any single phrasing missing. It buys recall, costs $n$ retrievals
plus a fusion step, and inherits {{ch:emb-hybrid}}'s warning that fusion can
*hurt* when the retrievers are complementary.

**Decomposition** splits a question that no single retrieval can answer.
*"Did revenue grow faster than headcount last year?"* needs two facts that live in
different documents; **no chunk contains the answer**, so retrieval cannot
succeed at any $k$. This is not a recall problem to be tuned — it is a structural
mismatch, and decomposition is the only fix in this chapter.

## 5. Formal Explanation

### 5.1 The retrieval key

Retrieval scores $s(\hat{f}(q), \hat{g}(z))$. Query transformation inserts a step:

$$ q \xrightarrow{\ T\ } q' \longrightarrow \hat{f}(q') $$ (eq:query-transform)

and the transformation is worth applying when

$$ \E_{z^{*}}\big[s(\hat{f}(T(q)), \hat{g}(z^{*}))\big] \;>\; \E_{z^{*}}\big[s(\hat{f}(q), \hat{g}(z^{*}))\big] $$ (eq:transform-worth-it)

for the relevant document $z^{*}$. **Nothing here requires $T(q)$ to be true, or
even meaningful to a human.** It requires only that it be *closer to the target
region of embedding space*. That is the formal content of HyDE's surprise, and
it also explains why the technique cannot be evaluated by reading the
hypotheticals — a good key may look like a bad answer.

### 5.2 Why the asymmetry is not fully trainable away

{{ch:emb-what-they-are}} noted that dual encoders and prefix conventions exist to
handle asymmetry. Why do they not solve it?

A dual encoder learns $\hat{f}$ for queries and $\hat{g}$ for documents from
observed pairs. It therefore handles the asymmetries **present in its training
distribution** and not others:

$$ \text{handled} = \big\{(q, z) \text{ patterns in the training pairs}\big\} \;\subsetneq\; \big\{\text{patterns in your traffic}\big\} $$ (eq:asymmetry-coverage)

Public retrievers are trained largely on web question/answer pairs. A query that
is a log line, a stack trace, a product code, or a half-sentence from a chat
thread is outside that distribution, and the learned asymmetry does not apply.
**Query transformation is a distribution-shift fix**, which is why it helps most
in exactly the domains {{cite:thakur2021beir}} identified as hard: unusual
corpora with unusual queries.

### 5.3 Multi-query recall

For $n$ query variants with individual recall $R$ and pairwise result overlap
$\omega$, fused recall is bounded by

$$ R_n \;\leq\; 1 - (1 - R)^{n_{\text{eff}}}, \qquad n_{\text{eff}} = 1 + (n-1)(1 - \omega) $$ (eq:multi-query-recall)

**The overlap term is what makes this usually disappointing.** Paraphrases of one
question retrieve mostly the same documents — $\omega$ near 1 — so
$n_{\text{eff}} \approx 1$ and $n$ retrievals buy almost nothing. Multi-query pays
off precisely when the variants are *genuinely different* queries, which is
decomposition rather than paraphrase.

This is {{ch:emb-hybrid}}'s complementarity result again: **fusion needs
diversity, and paraphrase is not diversity.**

### 5.4 Decomposition

For a question requiring facts $a_1 \dots a_m$ spread across documents, single
retrieval succeeds only if some chunk contains all of them:

$$ \Prob[\text{single-query success}] = \Prob\big[\exists z : a_1, \dots, a_m \in z\big] $$ (eq:multi-hop-containment)

which is {{ch:rag-chunking}}'s {{eq:span-containment}} across *documents* rather
than within one — and across documents it is typically **zero**, since the facts
were written separately.

Decomposition retrieves per sub-question, so success is the conjunction of
independent retrievals:

$$ \Prob[\text{decomposed success}] = \prod_{i=1}^{m} R_i $$ (eq:decomposition-success)

**A product of numbers below one, which degrades fast**: five sub-questions at
90% each is 59%. This is {{eq:tool-chain-success}} from
{{ch:llm-function-calling}} in a new setting, and it bounds how deep
decomposition can usefully go — a limit {{ch:rag-agentic}} runs into directly.

### 5.5 Cost

Every technique in this chapter costs a generation call before retrieval:

$$ L_{\text{total}} = \underbrace{L_{\text{rewrite}}}_{\text{a model call}} + L_{\text{retrieve}} + L_{\text{generate}}, \qquad C_{\text{total}} = n \cdot C_{\text{retrieve}} + C_{\text{rewrite}} + C_{\text{generate}} $$ (eq:query-transform-cost)

$L_{\text{rewrite}}$ is on the critical path and cannot be overlapped with
retrieval, because retrieval depends on its output. **For an interactive system
this is often the binding objection**, and it is why a small fast model for
rewriting is a much better choice than the one doing the generation — a
{{ch:llm-routing}} decision applied to a pipeline stage.

## 6. Mathematical Foundation

### 6.1 Why a hypothetical answer is a better key

Model an embedding as a topic component plus a *form* component — vocabulary
register, length, declarative versus interrogative:

$$ \hat{f}(x) \approx \alpha\, u_{\text{topic}}(x) + \beta\, u_{\text{form}}(x) $$ (eq:topic-form-decomposition)

Documents share a form (declarative, long, expository). A question does not, so

$$ s(q, z^{*}) = \alpha^2 \langle u_{\text{topic}}\rangle + \beta^2 \langle u^{q}_{\text{form}}, u^{z}_{\text{form}}\rangle $$ (eq:asymmetric-score)

and the second term is *small or negative* for a question against a document,
while it is *large* for a document against a document. A hypothetical answer has
the document form, so it recovers the second term while keeping the first.

**The prediction this makes is testable and is the listing's main point:**
{{eq:asymmetric-score}}'s form term does not depend on the hypothetical's
*content* being correct. Corrupting the facts changes $u_{\text{topic}}$ slightly
and leaves $u_{\text{form}}$ untouched, so retrieval should barely move.

### 6.2 The composition problem

Techniques stack, and stacking is where systems get slow without getting better.
With rewriting, expansion to $n$ variants, and hybrid retrieval
({{ch:emb-hybrid}}), a single user question becomes

$$ 1 \text{ question} \to 1 \text{ rewrite call} \to n \text{ variants} \to 2n \text{ retrievals} \to \text{fusion} \to \text{rerank} $$ (eq:pipeline-explosion)

For $n = 4$ that is eight retrievals, a fusion, and a rerank per question. The
recall gain is bounded by {{eq:multi-query-recall}}'s $n_{\text{eff}}$, which for
paraphrases is close to 1.

**So the composition is frequently 8× the cost for a few points of recall**, and
the discipline this chapter asks for is to measure each stage's marginal
contribution separately rather than adopting the stack.

### 6.3 Which technique for which failure

The decision procedure, derived from what each transformation changes:

| Symptom | Diagnosis | Technique |
|---|---|---|
| retrieval misses on short questions | form mismatch ({{eq:asymmetric-score}}) | HyDE |
| misses on domain jargon | vocabulary gap | expansion, or hybrid ({{ch:emb-hybrid}}) |
| misses on follow-up turns | unresolved reference | conversational resolution |
| answer needs two documents | {{eq:multi-hop-containment}} is zero | decomposition |
| results ignore a stated constraint | not a retrieval problem | filter extraction ({{ch:rag-indexing}}) |
| recall fine, answer poor | not a query problem | {{ch:rag-generation}} |

**The last two rows are the important ones**, because query transformation is
routinely applied to failures it cannot fix — and it makes the pipeline slower
while the actual bug stays where it was.

## 7. Internal Mechanics

```mermaid {#fig:query-pipeline caption="Query understanding as several components, not one. The classify step is what keeps eq:pipeline-explosion under control — most queries need none of the expensive branches, and deciding that cheaply is worth more than any single technique."}
flowchart TD
    Q["user message"] --> RES["resolve references<br/>against history"]
    RES --> EX["extract constraints<br/>→ filters (ch:rag-indexing)"]
    EX --> CL{"classify:<br/>what does this need?"}
    CL -->|"simple lookup"| D["retrieve directly"]
    CL -->|"form mismatch"| H["HyDE: generate a<br/>hypothetical answer"]
    CL -->|"multi-hop"| DEC["decompose into<br/>sub-questions"]
    CL -->|"no retrieval needed"| SKIP["answer directly"]
    H --> D
    DEC --> D2["retrieve per sub-question<br/>(eq:decomposition-success)"]
    D --> F["fuse + rerank"]
    D2 --> F
```

### 7.1 Conversational resolution

The highest-value and least-discussed component. In a multi-turn system, a large
share of messages are not self-contained:

```text
turn 1: "what's the rate limit on the search API?"
turn 2: "and for the write endpoint?"
```

Embedding turn 2 retrieves documents about write endpoints in general and
probably nothing about rate limits. **The retrieval key must be the resolved
question** — *"what is the rate limit on the write endpoint of the search API"* —
which requires the history and a rewriting step.

Two rules that matter more than the technique:

- **Resolve, do not concatenate.** Appending the previous turns adds vocabulary
  from the old topic and drags retrieval toward it, which is worse than the
  unresolved query on a topic change.
- **Detect topic changes.** A resolved query that imports a stale topic is a
  characteristic and confusing failure — the user changed subject and the system
  keeps answering the old one.

### 7.2 Extracting constraints is parsing, not retrieval

{{ch:rag-indexing}} established that constraints must be filters. This is the
stage that finds them: *"our returns policy for EU customers last quarter"*
contains a region filter, a date filter, and a document-type filter.

**This is a parsing problem with an exact answer**, and it should be evaluated as
one — precision and recall of extracted filters against a labelled set — not as
part of end-to-end retrieval quality where its errors are invisible.

### 7.3 Classification is what makes the rest affordable

{{eq:pipeline-explosion}} is only tolerable if the expensive branches run rarely.
A cheap classifier deciding *what this query needs* is therefore worth more than
any individual technique, and it is {{ch:llm-routing}}'s argument applied to a
pipeline stage rather than to models.

The categories worth separating: **needs no retrieval** (greetings, chit-chat,
questions about the conversation itself), **simple lookup**, **needs
transformation**, **multi-hop**. The first category alone is frequently 20–30% of
traffic in a chat product, and retrieving for it wastes latency and injects
distractors into a query the model could answer directly.

## 8. Implementation

```python {tier=A name=hyde-retrieval}
"""Searching with a hypothetical ANSWER instead of the question -- and what the
claim that its factual accuracy is irrelevant actually depends on.

Embeddings carry a topic component, a FACT component (the passage's specific
claims), and a FORM component (register, length, declarative vs interrogative)
-- eq:topic-form-decomposition. Documents share a form; questions do not, so
eq:asymmetric-score's form term works against a raw query.

Four retrieval keys over the same corpus:
  raw question         -- question form, right topic, no facts
  hypothetical, right  -- document form, right topic, right facts
  hypothetical, WRONG  -- document form, right topic, WRONG facts
  the true answer      -- an upper bound, unavailable at query time

HyDE's claim is that the third key works nearly as well as the second. That is
not a free-standing fact: it is a claim about how much of the embedding's mass
sits in the FACT component. So we sweep that weight rather than fixing it, and
report where the claim holds and where it fails.
"""
import numpy as np

rng = np.random.default_rng(41)

N_DOC, N_QUERY, K = 5000, 400, 10
D_TOPIC, D_FACT, D_FORM = 24, 12, 8
W_TOPIC, W_FORM = 1.0, 0.75
FACT_WEIGHTS = [0.10, 0.20, 0.35, 0.55, 0.80]


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


form_doc = unit(rng.normal(size=D_FORM))            # long, declarative
form_question = unit(rng.normal(size=D_FORM))       # short, interrogative

doc_topic = unit(rng.normal(size=(N_DOC, D_TOPIC)))
doc_fact = unit(rng.normal(size=(N_DOC, D_FACT)))
doc_form = unit(np.tile(form_doc, (N_DOC, 1))
                + rng.normal(scale=0.12, size=(N_DOC, D_FORM)))

# Fixed query draws, shared across every fact weight, so the sweep isolates w.
CASES = []
for _ in range(N_QUERY):
    i = int(rng.integers(0, N_DOC))
    CASES.append((
        i,
        unit(doc_topic[i] + rng.normal(scale=0.30, size=D_TOPIC)),   # query topic
        unit(doc_fact[i] + rng.normal(scale=0.45, size=D_FACT)),     # facts guessed right
        unit(rng.normal(size=D_FACT)),                               # facts wrong
        unit(rng.normal(size=D_FACT)) * 0.15,                        # question: no facts
    ))


def block(topic, fact, form, w_fact):
    return unit(np.concatenate([W_TOPIC * topic, w_fact * fact,
                                W_FORM * form], axis=-1))


print(f"{'fact weight':>12}{'raw question':>15}{'hypo (right)':>15}"
      f"{'hypo (WRONG)':>15}{'wrong vs right':>17}")
print("-" * 74)

for w in FACT_WEIGHTS:
    docs = block(doc_topic, doc_fact, doc_form, w)
    hits = {"raw": 0, "right": 0, "wrong": 0}
    for i, q_topic, f_right, f_wrong, f_none in CASES:
        keys = {
            "raw":   block(q_topic, f_none,  form_question, w),
            "right": block(q_topic, f_right, form_doc,      w),
            "wrong": block(q_topic, f_wrong, form_doc,      w),
        }
        for name, key in keys.items():
            if i in np.argpartition(-(docs @ key), K)[:K]:
                hits[name] += 1
    r_raw, r_right, r_wrong = (hits[k] / N_QUERY for k in ("raw", "right", "wrong"))
    print(f"{w:>12.2f}{r_raw:>15.3f}{r_right:>15.3f}{r_wrong:>15.3f}"
          f"{(r_wrong - r_right) * 100:>+16.1f}pp")

print("""
Read the raw-question column first: it is the worst key at every fact weight, and
it is the one every system uses by default. eq:asymmetric-score says why -- its
form component points somewhere documents do not live, so the form term works
AGAINST it however well the topic matches.

Now read the last column, which is HyDE's actual claim under test. At LOW fact
weight the wrong hypothetical performs close to the right one: retrieval is
carried by topic and form, and the invented facts barely register. At HIGH fact
weight the gap opens and the wrong hypothetical degrades sharply -- at the top of
the sweep it can fall below the raw question, because a confidently wrong fact
vector points AWAY from the target while an absent one merely fails to help.

So "the hypothetical does not need to be correct" is not a free-standing fact
about HyDE. It is a claim about where the embedding puts its mass, and it holds
in the regime where passage embeddings actually sit -- topic and register
dominating, specific claims contributing little. That is the regime described in
ch:emb-what-they-are, and it is why the technique works in practice.

But it also says exactly when to be careful: for a corpus where documents are
distinguished mainly by their SPECIFIC CLAIMS rather than their subject -- a
price list, a specification table, a set of near-identical policy variants -- the
fact component carries the discriminative signal, and a hypothetical with
invented specifics will retrieve confidently and wrongly.

Two practical consequences hold across the whole sweep. You cannot evaluate HyDE
by reading the hypotheticals: at low fact weight a factually wrong key is a good
key, so rejecting the technique because the text is wrong rejects it for the
wrong reason. And the rewriting model does not need to be a strong one -- it
needs to produce document-SHAPED text about the right topic, which is far cheaper
(eq:query-transform-cost).""")
```

```python {tier=A name=decomposition-and-fusion}
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
```

## 9. Practical Example

**HyDE.** The raw question is the **worst** key at every setting, scoring around
0.46 while a correct hypothetical reaches 0.58–0.80. {{eq:asymmetric-score}} says
why: the question's form component points somewhere documents do not live, so the
form term works against it however well the topic matches. That much is
unambiguous and it is the case for the technique.

The interesting result is the *conditional* one, and testing it rather than
repeating it changed what this section says. **"The hypothetical does not need to
be correct" is not a free-standing fact about HyDE — it is a claim about where
the embedding puts its mass**, and the sweep shows exactly where it holds:

- At **low fact weight**, a deliberately wrong hypothetical scores 0.562 against
  the correct one's 0.580 — a gap of **1.7 points**. Retrieval is carried by
  topic and form, and the invented specifics barely register. The claim holds.
- At **high fact weight**, the wrong hypothetical collapses to 0.177 against
  0.792 — **61.5 points** — and falls *below the raw question*, because a
  confidently wrong fact vector points away from the target while an absent one
  merely fails to help.

The regime where passage embeddings actually sit is the first one — topic and
register dominating, specific claims contributing little
({{ch:emb-what-they-are}}) — which is why the technique works in practice.

**But the second regime is a real warning and it is not in the paper.** For a
corpus whose documents are distinguished mainly by their *specific claims* rather
than their subject — a price list, a specification table, a set of near-identical
policy variants — the fact component carries the discriminative signal, and a
hypothetical with invented specifics will retrieve confidently and wrongly.

Two consequences hold across the whole sweep. **You cannot evaluate HyDE by
reading the hypotheticals** — in the regime where it works, a factually wrong key
is a good key, so rejecting the technique because the text is wrong rejects it
for the wrong reason. And **the rewriting model does not need to be good**: it
needs document-shaped text about the right topic, which is far cheaper
({{eq:query-transform-cost}}).

**Decomposition.** The single-query column goes from 0.422 at one hop to
**exactly 0.000 at two and beyond** — a total collapse, exactly as
{{eq:multi-hop-containment}} predicts. **This is not a recall problem that a
larger $k$ would fix**: the answer is in no single document, so there is nothing
for one retrieval to find.

Four paraphrases cost four retrievals and buy **nothing** — 0.003 at two hops.
{{eq:multi-query-recall}} explains it: paraphrases overlap almost completely, so
$n_{\text{eff}}$ stays near 1. They are even slightly *worse* than the single
query at one hop, because each perturbed variant is a worse key than the original
and a union of four worse keys need not beat one good one. **Fusion needs
diversity and paraphrase is not diversity** — {{ch:emb-hybrid}}'s complementarity
result in a new setting.

Decomposition holds at 0.96–0.99 across every hop count, and at $h$ hops it costs
$h$ retrievals — **the same order of cost as the paraphrases that bought
nothing.** The difference is not how many queries were issued but whether they
asked *different things*.

> **NOTE:** The decomposed column declines only gently here because each
> sub-question retrieves at recall near 0.99. {{eq:decomposition-success}} is a
> product, so that slope is an artefact of an unrealistically good retriever — at
> a typical 0.85 per hop, four hops is 0.52 and eight is 0.27. **The wall is
> real; this listing sits well short of it**, and {{ch:rag-agentic}} is where a
> system meets it.

> **IMPORTANT:** The two techniques are not variants of one idea at different
> budgets. Decomposition addresses a structural gap; paraphrase hedges a phrasing
> risk. **Reaching for multi-query when the question was multi-hop is the most
> common way this stage is misapplied**, and it produces a system that is 4×
> slower and still cannot answer the question.

## 10. Production Considerations

**Classify before transforming.** {{eq:pipeline-explosion}}: the expensive
branches must run rarely. Detecting the 20–30% of chat traffic needing no
retrieval at all is usually the largest single win here.

**Use a small fast model for rewriting.** It is on the critical path
({{eq:query-transform-cost}}) and, per the HyDE result, does not need to be
good.

**Resolve conversational references; do not concatenate history.** Concatenation
drags retrieval toward stale topics.

**Extract filters as a parsing task with its own metric**
({{ch:rag-indexing}}), not as part of end-to-end retrieval quality.

**Cache aggressively.** Rewrites are deterministic given the query and history,
and query distributions are heavily skewed, so a cache hit rate of 30–50% is
normal and removes the latency objection for repeat traffic.

**Log the transformed query alongside the original.** When retrieval fails, the
first question is which query actually ran, and it is frequently not the one the
user typed.

**Measure each stage's marginal contribution separately.** Adopting the full
stack because a paper reported gains is how {{eq:pipeline-explosion}}'s cost
arrives without its benefit.

**Have a fallback to the raw query.** If the rewriter times out or returns
nonsense, retrieving with the original is a fine degradation and a much better
one than failing.

## 11. Common Mistakes

**Multi-query for a multi-hop question.** {{eq:multi-query-recall}} against
{{eq:multi-hop-containment}}: the chapter's central error.

**Judging HyDE by whether the hypothetical is correct.**

**Concatenating conversation history into the retrieval key.**

**Applying transformation to failures it cannot fix** — the last two rows of
{{sec:6-mathematical-foundation}}'s table.

**Using the generation model to rewrite.** Expensive, slow, and unnecessary.

**Unbounded decomposition depth.** {{eq:decomposition-success}} is a product
below one.

**Not caching rewrites.**

**No fallback path**, turning a rewriter outage into a retrieval outage.

## 12. Failure Modes

**Topic drift on follow-ups.** The resolver imports a stale topic after the user
changes subject; the system confidently answers the previous question.

**Over-specific hypotheticals.** The generated document invents a narrow scenario
and retrieval lands in the wrong neighbourhood. Symptom: HyDE helps on average
and hurts badly on a specific slice — check the tail, not the mean. This is
{{sec:9-practical-example}}'s high-fact-weight regime arriving as a slice rather
than as a corpus: even where facts are lightly weighted overall, the queries that
hinge on a specific number or identifier are exactly the ones an invented
hypothetical will send to the wrong place.

**Decomposition into unanswerable sub-questions.** The split is grammatical
rather than semantic, and each fragment retrieves nothing.

**Fusion burying the good result.** {{ch:emb-hybrid}}'s RRF warning applies to
multi-query fusion identically.

**Latency inflation.** {{eq:pipeline-explosion}} arrives without measurement, and
p99 doubles for a recall gain nobody isolated.

**Rewriter hallucinating a constraint.** The model adds "in 2023" to a query that
had no date, and the filter silently excludes the answer. **This is the most
dangerous failure in the chapter**, because a transformation error becomes an
exact filter.

**Prompt injection through the query.** User text reaches a model that shapes
retrieval; {{part:26}} applies here and this is one of the earliest points in the
pipeline where it does.

## 13. Alternatives

**Hybrid retrieval** ({{ch:emb-hybrid}}). Fixes the vocabulary-gap failure
directly, with no generation call and no latency — and it should be tried
*before* query expansion, which is a more expensive route to a similar effect.

**A better embedding model.** {{eq:asymmetry-coverage}}: a retriever fine-tuned
on your own query/document pairs handles your asymmetries natively and removes
the need for HyDE.

**Query logs as training data.** Clicked results give real query/document pairs,
which is {{ch:emb-models}}'s domain fine-tuning with data you already have.

**Ask the user.** For genuinely under-specified queries, a clarifying question is
more accurate than guessing and is often better product design.

**Retrieve more and rerank.** Raising $k$ with a reranker
({{ch:emb-reranking}}) addresses recall without a pre-retrieval model call, and
compares favourably on latency.

**Do nothing.** If retrieval is already at its ceiling
({{ch:emb-reranking}}'s oracle test), query transformation cannot help and will
make the system slower.

## 14. Evaluation

**Retrieval recall with and without each transformation, separately.** The whole
point is to know which stage earns its cost.

**Sliced by query type.** HyDE helps short questions and can hurt long precise
ones; an aggregate hides both.

**Filter-extraction precision and recall** as its own parsing metric.

**Conversational resolution accuracy** on multi-turn logs, including the
topic-change cases, which is where it fails.

**Latency at p95 and p99**, since {{eq:query-transform-cost}} is on the critical
path.

**Decomposition quality**: are the sub-questions individually answerable? A
decomposition into unanswerable fragments is worse than none.

**Cache hit rate**, which determines whether the measured latency is the one
users see.

## 15. Advanced Concepts

**Query transformation is test-time compute for retrieval.** The same trade
{{part:16}} makes for reasoning — spend more computation at inference to get a
better answer — applied to the retrieval stage. Framing it that way makes the
cost question the right one and connects it to a literature that takes the
trade-off seriously.

**The rewriter can be trained** ({{cite:ma2023rewrite}}) with the reader's output
as the reward, which is the principled version of what prompting does
heuristically. Rare in production because the reward signal requires an
end-to-end evaluation loop most teams do not have.

**HyDE inverts the retrieval direction.** Rather than mapping queries into
document space, it *generates* into document space directly. That framing
suggests generalisations — generate a hypothetical *table row* for structured
retrieval ({{ch:rag-structured}}), a hypothetical *code snippet* for code search
— and they work for the same reason.

**Decomposition depth is bounded by {{eq:decomposition-success}}**, and this is
the same wall {{ch:llm-function-calling}}'s tool chains hit. Any architecture
whose success is a product of per-step reliabilities has a shallow practical
depth, which is the single most important constraint on {{ch:rag-agentic}} and on
{{part:17}}.

**The query distribution is a product artefact.** Users learn what a system
answers well and adapt — so measured query distributions reflect past behaviour,
and a transformation that unlocks a new query type will not show its value until
users discover it. This makes A/B tests of query understanding systematically
pessimistic in the short run.

## 16. Connection to Previous Chapters

{{ch:emb-what-they-are}}'s query/document asymmetry is the problem this chapter
attacks, and {{eq:asymmetry-coverage}} explains why the dual-encoder fix is
partial. {{ch:emb-hybrid}}'s complementarity result reappears as
{{eq:multi-query-recall}} — fusion needs diversity, and paraphrase is not
diversity. {{ch:rag-indexing}}'s constraints are extracted here.
{{ch:llm-function-calling}}'s compounding reliability becomes
{{eq:decomposition-success}} and bounds how deep decomposition can go.
{{ch:llm-routing}}'s routing argument is what makes {{eq:pipeline-explosion}}
affordable.

## 17. Exercises

1. Derive {{eq:multi-query-recall}} and compute $n_{\text{eff}}$ for $n=5$ at
   $\omega = 0.9$ and $\omega = 0.4$.
2. Using {{eq:decomposition-success}}, find the maximum useful hop count for
   per-hop recall of 0.85 and a 60% end-to-end target.
3. In `hyde-retrieval`, vary the form weight from 0 to 1.5 and find where the raw
   question stops being the worst key. What does that say about
   {{eq:asymmetric-score}}?
4. Corrupt the hypothetical's *topic* rather than its facts. Predict the effect
   before running it, then check, and explain the difference.
5. In `decomposition-and-fusion`, reduce the paraphrase perturbation to near
   zero. What happens to the paraphrase column, and why is that the expected
   result?
6. Add a strategy that decomposes into $h$ sub-questions but retrieves only
   $k/h$ for each. Does it beat full decomposition at equal total cost?
7. Design the conversational-resolution test set: what cases must it contain, and
   which one will your system fail?
8. A rewriter adds a date filter the user did not state. Write the guard.

## 18. Interview Questions

1. Why is the user's question a bad thing to search with?
2. What is HyDE, and why does it work when the hypothetical is wrong?
3. When does multi-query help, and when is it a waste?
4. Your system fails on "and what about the write endpoint?". Diagnose.
5. How would you handle "did revenue grow faster than headcount?"
6. What is the latency cost of query rewriting and how do you hide it?
7. How deep can you decompose a question?
8. Would you use hybrid retrieval or query expansion for a vocabulary gap?
9. How do you know query transformation is helping?
10. A rewriter invents a constraint. What breaks and how do you prevent it?

## 19. Research Questions

1. Can the query classifier of {{sec:7-internal-mechanics}} be learned from
   retrieval outcomes rather than hand-specified, so the pipeline routes itself?
2. {{eq:asymmetry-coverage}} suggests query transformation substitutes for
   retriever fine-tuning. Is there a principled account of when each is cheaper,
   given a fixed budget?
3. HyDE works because form dominates. Is there a *direct* way to project a query
   into document form without generating text, and would it be cheaper?
4. {{eq:decomposition-success}}'s product bound applies to every multi-step
   architecture. Is there a decomposition strategy with sub-multiplicative
   degradation — for instance, one where later steps can repair earlier ones?
5. Rewriters can inject constraints that were not stated. Is there a verification
   step for query transformation analogous to {{ch:rag-generation}}'s citation
   verification?

## 20. Chapter Summary

**A question is a poor retrieval key**, and the reason is structural: relevance
relates a question to an answer, and those look nothing alike. Dual encoders and
prefix conventions patch the asymmetries present in their *training*
distribution ({{eq:asymmetry-coverage}}), which is why transformation helps most
in exactly the domains where public retrievers are weakest.

**HyDE's surprise survives testing, conditionally.** Searching with a generated
hypothetical answer beats searching with the question at every setting measured.
Whether the hypothetical needs to be *correct* turns out to depend on how much of
the embedding's mass sits in specific claims rather than topic and register:
measured, a wrong hypothetical costs 1.7 points where facts are lightly weighted
and 61.5 points where they dominate — falling below the raw question there.
Passage embeddings sit in the first regime, which is why the technique works; a
corpus of price lists or near-identical policy variants sits in the second, which
is when not to use it. Two consequences hold throughout: you cannot evaluate the
technique by reading the hypotheticals, and the rewriting model does not need to
be good.

**Multi-query and decomposition are different things.** Paraphrases overlap, so
{{eq:multi-query-recall}}'s $n_{\text{eff}}$ stays near 1 and $n$ retrievals buy
little — fusion needs diversity, and paraphrase is not diversity. Decomposition
addresses a *structural* gap: when the answer lives in several documents,
{{eq:multi-hop-containment}} is zero and no $k$ can succeed. Measured, single-query
success collapses with hop count while decomposition holds, at the same order of
cost as the paraphrases that bought nothing.

**And decomposition has a depth limit**: {{eq:decomposition-success}} is a
product of per-step recalls, so five hops at 90% is 59% — the same wall tool
chains hit, and the binding constraint on everything in {{ch:rag-agentic}}.

The discipline the chapter asks for is to **classify first**
({{eq:pipeline-explosion}}), measure each stage's marginal contribution, and
resist adopting a stack of techniques whose combined cost is certain and whose
combined benefit is not.

## 21. Further Reading

{{cite:gao2023hyde}} for HyDE — short, and its Section 3 makes the
key-not-an-answer argument this chapter tests.
{{cite:ma2023rewrite}} for a *trained* rewriter and the reinforcement signal that
makes it principled.
{{cite:cormack2009rrf}} for the fusion step multi-query depends on, and for why
it is more fragile than it looks.
{{cite:karpukhin2020dpr}} and {{cite:wang2022e5}} for the asymmetry handling built
into the retriever, which is the alternative to fixing it at query time.
{{cite:gao2023ragsurvey}} for the standard taxonomy of these techniques, which is
comprehensive on what exists and quiet on when each is worth its cost.

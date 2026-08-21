---
id: nlp-extraction
number: 77
part: VIII
tier: full
status: draft
requires: [nlp-bert, nlp-contextual, ml-metrics, ml-logistic, dl-losses,
           ds-leakage, mle-splits]
provides: [sequence-labelling, named-entity-recognition, bio-tagging,
           entity-level-f1, span-extraction, conditional-random-field,
           structured-prediction, relation-extraction, annotation-agreement,
           text-classification-head, constrained-decoding-nlp]
citations: [tjongkimsang2003, lample2016, devlin2019bert, rajpurkar2016,
            liu2019roberta, sanh2019, wang2019glue, howard2018]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Formulate classification, sequence labelling, and span extraction as three
   distinct output structures over the same encoder.
2. Encode and decode BIO tags correctly, including the illegal transitions a
   per-token classifier can produce.
3. Explain why entity-level F1 and token-level F1 differ, compute both, and say
   which to report.
4. Derive the CRF's linear-chain objective and explain what structured prediction
   buys over independent per-token decisions.
5. Design an annotation process whose agreement can be measured, and explain why
   agreement bounds achievable accuracy.
6. Compare the encoder and LLM approaches to extraction on cost, latency, schema
   flexibility, and failure mode — and choose between them from evidence.
7. Detect the specific leakage patterns that make extraction results look better
   than they are.

## 2. Why This Matters

**This is the chapter where the encoder does a job.** Everything since
{{ch:nlp-preprocessing}} has been representation; this is the first task with an
output somebody wants — which entities appear, which clauses matter, which
tickets are urgent.

**The evaluation lesson here is the most transferable one in the part.**
{{cite:tjongkimsang2003}} decided that named-entity recognition would be scored
on whole spans, not tokens. That single decision means a system getting three
tokens of a four-token entity right scores **zero** for that entity. The choice
of evaluation unit is a modelling decision, and this is the book's cleanest
example of one.

**The task did not go away; it moved.** Most new extraction systems prompt an LLM
for structured output rather than fine-tuning an encoder ({{ch:llm-structured-output}}).
That is a real shift and it is not total: at high volume the encoder is two
orders of magnitude cheaper, and cost decides more production architectures than
capability does. This chapter gives both sides with the arithmetic.

**And structured prediction generalises far past NER.** The insight — that when
the label space has hard constraints, decoding jointly beats deciding
independently — is the same insight behind constrained decoding for JSON output
in {{ch:llm-structured-output}}. Meeting it first on a small, fully-inspectable
problem makes it legible later.

## 3. Prerequisites

{{ch:nlp-bert}} for the encoder being fine-tuned and the `[CLS]` convention.
{{ch:nlp-contextual}} for feature-based versus fine-tuning transfer.
{{ch:ml-metrics}} for precision, recall, F1, and the micro/macro distinction —
this chapter leans on it heavily. {{ch:ml-logistic}} for the per-token classifier.
{{ch:dl-losses}} for cross-entropy. {{ch:ds-leakage}} for the leakage patterns in
{{sec:12-failure-modes}}. {{ch:mle-splits}} for why entity-level splitting is not
the same as random splitting.

## 4. Intuitive Explanation

Three tasks, one encoder, three different shapes of output.

**Classification** produces one label for the whole text. Is this ticket urgent?
Is this review positive? The encoder produces a vector per token; you need one
vector, so you pool — and then a linear layer.

**Sequence labelling** produces one label per token. Which words are a person's
name? This is where the interesting structure lives, because the labels are not
independent: "the second token of a person's name" cannot appear unless the token
before it was part of a person's name.

**Span extraction** produces a start and an end index. Where in this passage is
the answer to this question? {{cite:rajpurkar2016}} made this the standard
question-answering format, and it is the structure that lets an answer be
*checked* against a source — which is the grounding property RAG inherits.

**The BIO encoding is how sequence labelling represents spans.** Each token gets
one of `B-TYPE` (beginning of an entity), `I-TYPE` (inside one), or `O` (outside
any). "Jane Smith visited Paris" becomes `B-PER I-PER O B-LOC`. The `B`/`I`
distinction exists for one reason: without it, two adjacent entities of the same
type would merge into one.

> NOTE: BIO is a way of squeezing a *span* problem into a *per-token* problem so
> that ordinary classification machinery applies. Everything awkward about it —
> illegal transitions, the entity-versus-token evaluation gap — comes from that
> squeeze. It is a representation choice, not a fact about language.

**And the squeeze leaks.** A per-token classifier can emit `O I-PER O`, which
means "not an entity, continuation of a person, not an entity" — a continuation
of nothing. Nothing in the model forbids it, because each token was classified on
its own. Two fixes exist: repair the output afterwards, or make the model decode
the whole sequence jointly so illegal transitions are impossible. The second is
the CRF, and it is better.

**The mental model:** extraction is classification with a shape constraint, and
almost every practical difficulty comes from the constraint rather than from the
classification. Where it breaks down: some extraction problems have constraints
BIO cannot express at all — nested entities, discontinuous mentions, overlapping
relations — and those need a different output structure, not a better tagger.

## 5. Formal Explanation

### 5.1 Three output structures

Let $f_\theta(\vec{x}) \in \R^{T\times d}$ be the encoder's output for a
$T$-token sequence.

**Classification**: pool to a single vector, then project to $C$ classes.

$$
\hat{y} = \softmax\big(\mat{W}\,\text{pool}(f_\theta(\vec{x})) + \vec{b}\big),
\qquad \mat{W} \in \R^{C\times d}
$$ (eq:classification-head)

**Sequence labelling**: project every position to $|\mathcal{T}|$ tags.

$$
\hat{y}_i = \softmax\big(\mat{W}f_\theta(\vec{x})_i + \vec{b}\big),
\qquad \mat{W}\in\R^{|\mathcal{T}|\times d}
$$ (eq:tagging-head)

**Span extraction**: two projections to scalars, softmaxed over positions.

$$
P_{\text{start}}(i) = \frac{\exp(\vec{w}_s\T f_\theta(\vec{x})_i)}
                           {\sum_j \exp(\vec{w}_s\T f_\theta(\vec{x})_j)},
\qquad \text{likewise } P_{\text{end}}
$$ (eq:span-head)

with the predicted span $\argmax_{i\le j} P_{\text{start}}(i)P_{\text{end}}(j)$,
where the constraint $i \le j$ is enforced during decoding rather than learned.

**Note how little differs.** The encoder is identical; only the head and the loss
change. That is what "pretrain once, fine-tune per task" means concretely.

### 5.2 BIO tagging

For entity types $\mathcal{E}$, the tag set is

$$
\mathcal{T} = \{\texttt{O}\} \cup \{\texttt{B-}e,\ \texttt{I-}e
 \ :\ e \in \mathcal{E}\},
\qquad |\mathcal{T}| = 2|\mathcal{E}| + 1
$$ (eq:bio-tagset)

A tag sequence is **well-formed** if every `I-`$e$ is preceded by `B-`$e$ or
`I-`$e$. The set of well-formed sequences is a strict subset of
$\mathcal{T}^T$, and the fraction that are well-formed shrinks with length:

$$
\frac{|\text{well-formed}|}{|\mathcal{T}|^T} \to 0
 \quad\text{as } T\to\infty
$$ (eq:wellformed-fraction)

**A per-token classifier optimises over the whole space and the constraint is not
part of its hypothesis class.** It will produce ill-formed output at some rate,
and that rate is a property of the decoding scheme, not of how well the model
learned.

### 5.3 Entity-level evaluation

{{cite:tjongkimsang2003}} defines the unit of evaluation as the **span**. Let
$\hat{S}$ be predicted entities and $S$ the gold ones, each a triple (start, end,
type). An entity is correct only on exact match of all three:

$$
\text{P} = \frac{|\hat{S}\cap S|}{|\hat{S}|},\quad
\text{R} = \frac{|\hat{S}\cap S|}{|S|},\quad
\text{F}_1 = \frac{2\text{PR}}{\text{P}+\text{R}}
$$ (eq:entity-f1)

Contrast token-level F1, computed over per-token tag predictions.

> IMPORTANT: These are different numbers and token-level F1 is almost always
> higher. Two reasons compound: `O` dominates the token distribution, so a model
> that predicts `O` everywhere already scores well on tokens; and partial credit
> exists at token level and does not exist at entity level. **Report entity-level
> F1 for any span task.** A paper or dashboard reporting token-level F1 for NER
> is reporting an easier task under the same name.

### 5.4 The linear-chain CRF

Independent per-token classification models

$$
P(\vec{y}\given\vec{x}) = \prod_{i=1}^{T} P(y_i \given \vec{x})
$$ (eq:independent-tagging)

which assigns positive probability to ill-formed sequences. A **conditional
random field** ({{cite:lample2016}}) instead scores the whole sequence, adding a
learned transition term:

$$
s(\vec{x},\vec{y}) = \sum_{i=1}^{T} \underbrace{E_{i,y_i}}_{\text{emission}}
 + \sum_{i=1}^{T+1} \underbrace{A_{y_{i-1},y_i}}_{\text{transition}}
$$ (eq:crf-score)

with $\mat{E}\in\R^{T\times|\mathcal{T}|}$ from {{eq:tagging-head}} and
$\mat{A}\in\R^{(|\mathcal{T}|+2)\times(|\mathcal{T}|+2)}$ a learned transition
matrix including start and stop states. The distribution is

$$
P(\vec{y}\given\vec{x}) = \frac{\exp\,s(\vec{x},\vec{y})}
 {\sum_{\vec{y}'\in\mathcal{T}^T}\exp\,s(\vec{x},\vec{y}')}
$$ (eq:crf-distribution)

**The denominator sums over $|\mathcal{T}|^T$ sequences** and is computed exactly
in $O(T|\mathcal{T}|^2)$ by the forward algorithm — the derivation is in
{{sec:6-mathematical-foundation}}. Decoding uses Viterbi, the same dynamic
program as {{eq:viterbi-recurrence}} in {{ch:nlp-subword}}, and setting
$A_{\texttt{O},\texttt{I-}e} = -\infty$ makes illegal transitions **impossible**
rather than merely unlikely.

### 5.5 Relation extraction

Entities alone are rarely the deliverable. Given entities $e_1,\dots,e_n$, a
relation classifier predicts a label for each ordered pair:

$$
P(r \given e_i, e_j, \vec{x})
 = \softmax\big(\mat{W}[\vec{h}_{e_i};\vec{h}_{e_j};\vec{h}_{\text{ctx}}]\big)
$$ (eq:relation-extraction)

This is $O(n^2)$ classifications per document, and errors compound: a missed
entity removes every relation it participates in. **Pipeline error compounding is
the dominant practical problem in extraction systems**, and it is why joint
models exist.

## 6. Mathematical Foundation

### 6.1 The forward algorithm

The CRF's partition function
$Z = \sum_{\vec{y}}\exp s(\vec{x},\vec{y})$ appears to require summing over
$|\mathcal{T}|^T$ sequences. Define

$$
\alpha_i(t) = \sum_{\substack{\vec{y}_{1:i}\\ y_i = t}}
 \exp\Big(\sum_{k\le i} E_{k,y_k} + \sum_{k\le i} A_{y_{k-1},y_k}\Big)
$$ (eq:forward-alpha)

the total mass of all prefixes ending in tag $t$. Because the score decomposes
over adjacent pairs only, $\alpha$ satisfies

$$
\alpha_i(t) = \exp(E_{i,t})\sum_{t'\in\mathcal{T}}
 \alpha_{i-1}(t')\exp\big(A_{t',t}\big)
$$ (eq:forward-recurrence)

with $\alpha_1(t) = \exp(E_{1,t} + A_{\text{start},t})$ and
$Z = \sum_t \alpha_T(t)\exp(A_{t,\text{stop}})$.

$\square$

**Each of $T$ steps costs $|\mathcal{T}|^2$**, so the exact partition function
over an exponentially large set costs $O(T|\mathcal{T}|^2)$. For $|\mathcal{T}|=9$
and $T=128$ that is about 10,000 operations — negligible next to the encoder's
forward pass.

In practice the recurrence is computed in log space with
$\logsumexp$ to avoid underflow, which is the same numerical-stability move as
the log-sum-exp trick in {{ch:dl-losses}}.

### 6.2 Why entity-level and token-level F1 diverge

Consider $n$ gold entities each of length $L$ tokens, in a document of $T$
tokens. Suppose the model's tags are correct except that it drops the final token
of every entity.

**Token level.** $nL$ entity tokens exist, $n(L-1)$ are tagged correctly, and $n$
are wrongly tagged `O`:

$$
\text{recall}_{\text{token}} = \frac{n(L-1)}{nL} = 1 - \frac{1}{L}
$$ (eq:token-recall)

**Entity level.** Every predicted span has the wrong end index, so no span
matches exactly:

$$
\text{recall}_{\text{entity}} = 0
$$ (eq:entity-recall)

$\square$

With $L = 4$: **token-level recall of 0.75 and entity-level recall of 0.00.**
The same predictions. This is not a pathological construction — systematic
boundary errors are the most common NER failure, especially around titles,
initials, and trailing punctuation.

The gap runs the other way too, though less sharply: a model can score poorly on
tokens by mislabelling a few tokens of a long entity while still getting some
spans exactly right.

### 6.3 A worked evaluation

Gold: `[Jane Smith]_PER visited [Paris]_LOC in [March 2024]_DATE`.

Tokens: `Jane Smith visited Paris in March 2024` → gold tags
`B-PER I-PER O B-LOC O B-DATE I-DATE`.

Prediction: `B-PER O O B-LOC O B-DATE I-DATE`.

**Token level.** 6 of 7 tags correct → accuracy 0.857. Over the six non-`O`
gold tokens, 5 are right: token recall $5/6 = 0.833$.

**Entity level.** Predicted spans: `[Jane]_PER`, `[Paris]_LOC`,
`[March 2024]_DATE`. Gold spans: `[Jane Smith]_PER`, `[Paris]_LOC`,
`[March 2024]_DATE`. Exact matches: 2 of 3.

$$
\text{P} = 2/3 = 0.667,\quad \text{R} = 2/3 = 0.667,\quad \text{F}_1 = 0.667
$$

**0.857 against 0.667 from a single dropped token.** Both are correctly computed;
only one describes what a downstream consumer of the entities experiences,
because a system reading `Jane` where the person is `Jane Smith` has the wrong
person.

## 7. Internal Mechanics

```mermaid {#fig:extraction-heads caption="One encoder, three heads. The pretrained encoder is identical in all three cases; the task is entirely determined by the head, the loss, and — for sequence labelling — the decoder. The CRF is the only component that reasons about the output sequence as a whole."}
graph TD
  A["tokens"] --> B["pretrained encoder<br/>T x d contextual vectors"]
  B --> C["pool → linear → C classes<br/><i>classification</i>"]
  B --> D["linear at every position<br/>→ |T| tags"]
  B --> E["two linear scorers<br/>→ start, end distributions<br/><i>span extraction</i>"]
  D --> F["argmax per token<br/>illegal transitions possible"]
  D --> G["CRF: emissions + transitions<br/>Viterbi over well-formed only"]
  F --> H["BIO decode<br/>+ repair"]
  G --> I["BIO decode<br/>no repair needed"]
  style F fill:#fdd,stroke:#c66
  style G fill:#dfe,stroke:#5a5
```

**Subword alignment is the implementation detail that bites.** Labels are
annotated on words; the encoder consumes subwords ({{ch:nlp-subword}}). `Smith`
may become `Sm`,`##ith`, so a one-word label must map to two positions. The
convention is to label the first subword of each word and mark the rest as
ignored in the loss — and to take the prediction from the first subword at
decode time. Getting this misaligned by one position is the single most common
bug in NER pipelines, and it presents as "the model is bad" rather than as an
error.

**The transition matrix is small and interpretable.** For 4 entity types,
$\mat{A}$ is $11\times 11$ including start and stop. After training, its entries
are readable: the `O`→`I-PER` cell should be strongly negative, and if it is not,
the training data contains ill-formed annotations.

**Where the compute goes.** The encoder dominates completely. For BERT-base at
$T = 128$, the encoder is ~110M parameters of matmul; the CRF's forward algorithm
is $128 \times 81 \approx 10^4$ operations. **The CRF is free**, which makes the
usual argument against it — complexity — the only real one.

## 8. Implementation

BIO encoding and decoding, with the illegal transitions made visible:

```python {tier=A name=bio-encode-decode}
"""BIO tagging: encode spans to tags, decode tags to spans, repair what leaks."""

TOKENS = ["Jane", "Smith", "visited", "Paris", "in", "March", "2024",
          "with", "Mary", "Jones", "and", "Ann", "Lee"]
# Adjacent same-type entities are the reason B- exists at all.
GOLD_SPANS = [(0, 2, "PER"), (3, 4, "LOC"), (5, 7, "DATE"),
              (8, 10, "PER"), (11, 13, "PER")]


def spans_to_bio(spans, n):
    tags = ["O"] * n
    for start, end, typ in spans:
        tags[start] = f"B-{typ}"
        for i in range(start + 1, end):
            tags[i] = f"I-{typ}"
    return tags


def bio_to_spans(tags):
    """Decode. Any I- without a matching open entity is dropped as ill-formed."""
    spans, start, typ = [], None, None
    for i, tag in enumerate(tags + ["O"]):
        if tag.startswith("B-") or tag == "O" or \
                (tag.startswith("I-") and tag[2:] != typ):
            if typ is not None:
                spans.append((start, i, typ))
                start, typ = None, None
        if tag.startswith("B-"):
            start, typ = i, tag[2:]
        elif tag.startswith("I-") and typ is None:
            pass                      # ill-formed: I- with nothing open
    return spans


gold_tags = spans_to_bio(GOLD_SPANS, len(TOKENS))
print(f"{'token':<10} {'gold tag':<10}")
for tok, tag in zip(TOKENS, gold_tags):
    print(f"{tok:<10} {tag:<10}")

assert bio_to_spans(gold_tags) == GOLD_SPANS
print("\nround trip: spans -> BIO -> spans is exact")

# Two adjacent PER entities: without B-, 'Mary Jones and Ann Lee' would decode
# as one entity. Check that they stay separate.
decoded = bio_to_spans(gold_tags)
print(f"entities decoded: {len(decoded)} (gold {len(GOLD_SPANS)})")

# What an independent per-token classifier can emit.
ILLEGAL = ["O", "I-PER", "O", "B-LOC", "I-DATE", "B-DATE", "I-DATE",
           "O", "B-PER", "I-LOC", "O", "I-PER", "I-PER"]


def find_illegal(tags):
    bad = []
    prev = "O"
    for i, tag in enumerate(tags):
        if tag.startswith("I-"):
            typ = tag[2:]
            if not (prev == f"B-{typ}" or prev == f"I-{typ}"):
                bad.append((i, prev, tag))
        prev = tag
    return bad


print(f"\nill-formed sequence: {' '.join(ILLEGAL)}")
for i, prev, tag in find_illegal(ILLEGAL):
    print(f"  position {i}: '{tag}' after '{prev}' — continuation of nothing")
print(f"decoded anyway (repair drops them): {bio_to_spans(ILLEGAL)}")
print("\nRepair is a patch. The CRF makes these sequences unreachable instead.")
```

Now the evaluation that {{sec:6-mathematical-foundation}} argued about, computed
both ways on the same predictions:

```python {tier=A name=entity-vs-token-f1}
"""The same predictions, scored two ways. Equation (eq:entity-f1)."""

GOLD = ["B-PER", "I-PER", "O", "B-LOC", "O", "B-DATE", "I-DATE"]
TOKENS = ["Jane", "Smith", "visited", "Paris", "in", "March", "2024"]

# A single dropped final token on the PER entity — the most common NER error.
PRED = ["B-PER", "O", "O", "B-LOC", "O", "B-DATE", "I-DATE"]


def bio_to_spans(tags):
    spans, start, typ = [], None, None
    for i, tag in enumerate(list(tags) + ["O"]):
        if tag.startswith("B-") or tag == "O" or \
                (tag.startswith("I-") and tag[2:] != typ):
            if typ is not None:
                spans.append((start, i, typ))
                start, typ = None, None
        if tag.startswith("B-"):
            start, typ = i, tag[2:]
    return set(spans)


def prf(tp, n_pred, n_gold):
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


# Token level, over non-O gold tokens (the usual convention).
tp_tok = sum(1 for g, p in zip(GOLD, PRED) if g == p and g != "O")
n_pred_tok = sum(1 for p in PRED if p != "O")
n_gold_tok = sum(1 for g in GOLD if g != "O")

# Entity level: a span counts only on exact start, end, and type.
gold_spans, pred_spans = bio_to_spans(GOLD), bio_to_spans(PRED)
tp_ent = len(gold_spans & pred_spans)

print(f"{'':13} {'precision':>10} {'recall':>8} {'F1':>8}")
for name, (p, r, f) in [
        ("token level", prf(tp_tok, n_pred_tok, n_gold_tok)),
        ("entity level", prf(tp_ent, len(pred_spans), len(gold_spans)))]:
    print(f"{name:<13} {p:>10.3f} {r:>8.3f} {f:>8.3f}")

print(f"\ntoken accuracy over all positions: "
      f"{sum(g == p for g, p in zip(GOLD, PRED)) / len(GOLD):.3f}")
print(f"gold spans: {sorted(gold_spans)}")
print(f"pred spans: {sorted(pred_spans)}")
print(f"exactly matched: {sorted(gold_spans & pred_spans)}")

# Now the systematic case from equation (eq:token-recall): drop the last token
# of every entity, for entities of length L.
print(f"\n{'L':>3} {'token recall':>14} {'entity recall':>15}")
for L in [2, 3, 4, 8]:
    print(f"{L:>3} {1 - 1 / L:>14.3f} {0.0:>15.3f}")
print("\nEquation (eq:token-recall) against (eq:entity-recall): the token metric "
      "rises toward 1.0 as entities get longer while the entity metric stays "
      "at zero. Reporting the first for a span task reports a different task.")
```

Finally the CRF, with the forward algorithm and Viterbi implemented directly:

```python {tier=A name=crf-decoding}
"""A linear-chain CRF: exact partition function, and constrained decoding."""
import numpy as np
from itertools import product

TAGS = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC"]
K = len(TAGS)
rng = np.random.default_rng(0)


def build_transitions(constrain):
    """Transitions, plus start and stop vectors — equation (eq:crf-score).

    The start vector is easy to forget and is exactly where the constraint
    leaks: without start[I-*] = -inf a sequence may BEGIN with a continuation
    tag, which no pairwise transition can forbid.
    """
    A = rng.normal(0, 0.5, (K, K))
    start = rng.normal(0, 0.5, K)
    stop = rng.normal(0, 0.5, K)
    if constrain:
        for j, tag in enumerate(TAGS):
            if tag.startswith("I-"):
                typ = tag[2:]
                start[j] = -np.inf                      # cannot open with I-
                for i, prev in enumerate(TAGS):
                    if prev not in (f"B-{typ}", f"I-{typ}"):
                        A[i, j] = -np.inf               # cannot continue nothing
    return A, start, stop


def logsumexp(a, axis=None):
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)       # an all -inf row contributes nothing
    return np.squeeze(m, axis=axis) + np.log(np.exp(a - m).sum(axis=axis))


def log_partition(E, A, start, stop):
    """The forward algorithm — equation (eq:forward-recurrence), in log space."""
    alpha = E[0] + start
    for t in range(1, len(E)):
        alpha = E[t] + logsumexp(alpha[:, None] + A, axis=0)
    return logsumexp(alpha + stop)


def brute_force_log_partition(E, A, start, stop):
    """Sum over every one of K^T sequences. Tractable only because T is tiny."""
    scores = []
    for seq in product(range(K), repeat=len(E)):
        s = sum(E[t, y] for t, y in enumerate(seq)) + start[seq[0]] + stop[seq[-1]]
        s += sum(A[seq[t - 1], seq[t]] for t in range(1, len(seq)))
        scores.append(s)
    return logsumexp(np.array(scores))


def viterbi(E, A, start, stop):
    """Best path — the same dynamic program as (eq:viterbi-recurrence)."""
    T = len(E)
    delta = E[0] + start
    back = np.zeros((T, K), dtype=int)
    for t in range(1, T):
        scores = delta[:, None] + A
        back[t] = np.argmax(scores, axis=0)
        delta = E[t] + np.max(scores, axis=0)
    delta = delta + stop
    path = [int(np.argmax(delta))]
    for t in range(T - 1, 0, -1):
        path.append(int(back[t, path[-1]]))
    return [TAGS[i] for i in reversed(path)]


def illegal_count(tags):
    n, prev = 0, None
    for tag in tags:
        if tag.startswith("I-") and prev not in (f"B-{tag[2:]}", f"I-{tag[2:]}"):
            n += 1
        prev = tag
    return n


T = 6
E = rng.normal(0, 1.0, (T, K))         # emissions from the encoder head

# 1. The forward algorithm is exact, not an approximation — check it both with
#    and without the -inf entries, since those are where a log-space bug hides.
print(f"log Z over {K ** T:,} sequences")
for label, constrain in [("free transitions", False), ("constrained", True)]:
    A, st, sp = build_transitions(constrain)
    exact = brute_force_log_partition(E, A, st, sp)
    fast = log_partition(E, A, st, sp)
    print(f"  {label:<18} brute force {exact:>10.6f}   "
          f"forward {fast:>10.6f}   ({T} x {K}^2 = {T * K * K} operations)")
    assert abs(exact - fast) < 1e-9

# 2. Independent argmax versus free Viterbi versus constrained Viterbi.
A_free, st_free, sp_free = build_transitions(False)
A_con, st_con, sp_con = build_transitions(True)

print()
for name, tags in [
        ("independent argmax", [TAGS[i] for i in E.argmax(1)]),
        ("CRF, free transitions", viterbi(E, A_free, st_free, sp_free)),
        ("CRF, constrained", viterbi(E, A_con, st_con, sp_con))]:
    print(f"{name:<24} {' '.join(f'{t:<6}' for t in tags)}  "
          f"illegal: {illegal_count(tags)}")

# 3. The property, over many random emission matrices.
n_trials, bad_independent, bad_constrained = 2000, 0, 0
for _ in range(n_trials):
    Ei = rng.normal(0, 1.0, (T, K))
    bad_independent += illegal_count([TAGS[i] for i in Ei.argmax(1)]) > 0
    bad_constrained += illegal_count(viterbi(Ei, A_con, st_con, sp_con)) > 0

print(f"\nover {n_trials:,} random emission matrices:")
print(f"  independent argmax produced ill-formed output "
      f"{bad_independent / n_trials:.1%} of the time")
print(f"  constrained Viterbi produced ill-formed output "
      f"{bad_constrained / n_trials:.1%} of the time")
assert bad_constrained == 0
```

The final assertion is the argument for structured prediction stated as a
property rather than as a benchmark improvement: **the constrained decoder cannot
produce an ill-formed sequence, for any emissions whatsoever.** Repair after the
fact gets you a well-formed sequence too, but a repaired sequence is not the
highest-scoring well-formed sequence — Viterbi's is.

## 9. Practical Example

A contracts team needs party names, dates, and monetary amounts extracted from
50,000 documents per day, with the output feeding an automated obligation
tracker. The question is whether to fine-tune an encoder or prompt an LLM.

Both work. The decision is cost, latency, and how often the schema changes.

```python {tier=A name=encoder-vs-llm-extraction}
"""Encoder fine-tuning versus LLM prompting for extraction, priced."""

DOCS_PER_DAY = 50_000
TOKENS_PER_DOC = 1_200
SCHEMA_CHANGES_PER_YEAR = 6
LABELLED_EXAMPLES_NEEDED = 2_000
COST_PER_LABEL_USD = 1.50          # a domain expert, per annotated document

# Encoder route: fine-tune once, serve cheaply, relabel when the schema changes.
ENCODER = dict(
    gpu_hours_per_finetune=4, gpu_cost_per_hour=2.0,
    ms_per_doc=35, cpu_cost_per_hour=0.10, cores=4,
)
# LLM route: no labelling, no training, priced per token forever.
LLM = dict(input_per_1k=0.003, output_per_1k=0.015, output_tokens=150,
           ms_per_doc=2_500)

# --- encoder ---
label_cost = LABELLED_EXAMPLES_NEEDED * COST_PER_LABEL_USD
train_cost = ENCODER["gpu_hours_per_finetune"] * ENCODER["gpu_cost_per_hour"]
setup_per_change = label_cost * 0.3 + train_cost      # partial relabel each change
encoder_setup_year = label_cost + train_cost + SCHEMA_CHANGES_PER_YEAR * setup_per_change

doc_seconds = ENCODER["ms_per_doc"] / 1000
core_hours_day = DOCS_PER_DAY * doc_seconds / 3600
encoder_serve_day = core_hours_day * ENCODER["cpu_cost_per_hour"]

# --- LLM ---
llm_per_doc = (TOKENS_PER_DOC / 1000 * LLM["input_per_1k"]
               + LLM["output_tokens"] / 1000 * LLM["output_per_1k"])
llm_serve_day = DOCS_PER_DAY * llm_per_doc

print(f"{'':22} {'setup (yr 1)':>14} {'serving/day':>13} {'serving/yr':>13} "
      f"{'total yr 1':>13}")
for name, setup, day in [("fine-tuned encoder", encoder_setup_year, encoder_serve_day),
                         ("LLM prompting", 0.0, llm_serve_day)]:
    print(f"{name:<22} ${setup:>13,.0f} ${day:>12,.2f} ${day * 365:>12,.0f} "
          f"${setup + day * 365:>12,.0f}")

print(f"\nper-document: encoder "
      f"${encoder_serve_day / DOCS_PER_DAY:.6f}, LLM ${llm_per_doc:.4f} "
      f"({llm_per_doc / (encoder_serve_day / DOCS_PER_DAY):,.0f}x)")
print(f"latency:      encoder {ENCODER['ms_per_doc']} ms, "
      f"LLM {LLM['ms_per_doc']:,} ms "
      f"({LLM['ms_per_doc'] / ENCODER['ms_per_doc']:.0f}x)")

# Where the crossover sits.
breakeven = encoder_setup_year / (llm_serve_day - encoder_serve_day)
print(f"\nbreak-even: {breakeven:.0f} days of production volume.")
print("Below that the LLM is cheaper outright; above it the encoder's setup "
      "cost is amortised. The schema-change rate is what moves this number, "
      "because every change re-pays part of the labelling.")
```

**The honest summary is a cascade, not a winner.** Use the LLM to bootstrap
labels and to handle the long tail of rare entity types; fine-tune an encoder for
the high-volume common types; route the low-confidence cases to the LLM. That
structure appears again in {{ch:emb-reranking}} and again in {{part:22}}, always
for the same reason: a cheap high-recall stage in front of an expensive precise
one.

> PRODUCTION TIP: The strongest argument for the LLM route is not cost or
> quality — it is that a schema change is a prompt edit rather than a relabelling
> project. If your schema is still moving, that dominates everything in the
> table above.

## 10. Production Considerations

**Confidence thresholds are the main operating lever.** An extraction system
feeding an automated process needs a precision floor, and the way to get it is to
abstain below a score threshold and route those documents to review. Calibrate
the threshold on a held-out set against the precision the downstream process
requires, and re-calibrate after every model change.

**Annotation quality bounds achievable accuracy.** If two annotators agree 90% of
the time, a model scoring 95% against one of them is measuring noise. Measure
inter-annotator agreement before measuring the model, and treat it as the
ceiling.

**Entity linking is a separate system.** Recognising that `Paris` is a location
does not tell you which Paris. Linking to a knowledge base is its own problem
with its own failure modes, and conflating the two is a common scoping error.

**Schema changes are the operational reality.** New entity types arrive
continuously. Budget for partial relabelling and retraining, or accept the LLM
route's higher per-document cost in exchange for schema agility.

**What to monitor:** entity-level precision and recall by type on a rolling
labelled sample; the abstention rate; the distribution of entities per document
(a sudden drop usually means an upstream parsing change, not a model change); and
the ill-formed-output rate if you are not using constrained decoding.

## 11. Common Mistakes

**Beginners:**

*Reporting token-level F1 for a span task.* {{eq:token-recall}} against
{{eq:entity-recall}} shows how far apart these can be. Entity-level is the
number a consumer of the output experiences.

*Misaligning subword labels.* Labels are on words, the encoder consumes
subwords. Label the first subword and ignore the rest; being off by one presents
as poor model quality rather than as a bug.

*Using accuracy on a tagging task.* `O` is the overwhelming majority class, so
predicting `O` everywhere gives high accuracy and zero utility.

**Experienced practitioners:**

*Splitting documents randomly when entities repeat across them.* The same
company name appearing in train and test makes memorisation look like
generalisation — this is {{ch:ds-leakage}}'s pattern, and the fix is splitting by
document or by entity, whichever the deployment requires.

*Skipping the CRF because it seems dated.* {{sec:7-internal-mechanics}} shows it
costs about $10^4$ operations against the encoder's $10^{11}$. The cost argument
against it does not survive contact with the numbers.

*Constraining the transitions and forgetting the start state.* Setting
$A_{t',\texttt{I-}e} = -\infty$ forbids continuing nothing in the middle of a
sequence and says nothing about position 1, which has no predecessor — so the
decoder happily opens with `I-PER`. The constraint needs
$A_{\text{start},\texttt{I-}e} = -\infty$ too. The symptom is a small, stubborn
rate of ill-formed output that survives "adding a CRF", and it is the reason
{{eq:crf-score}} sums over $T+1$ transitions rather than $T$.

*Evaluating against one annotator.* Without agreement measurement you do not know
whether the remaining error is the model's or the label's.

*Assuming nested entities can be represented.* BIO cannot express an entity
inside another entity. `[Bank of [England]_LOC]_ORG` requires a different output
structure, and no amount of tagger quality substitutes.

## 12. Failure Modes

**Boundary errors.** The most common NER error, and the one entity-level F1
punishes hardest — titles (`Dr.`), initials, trailing possessives, and hyphenated
names all produce off-by-one spans. *Symptom:* a large gap between token-level
and entity-level scores. *Detection:* compute both; the gap is the diagnostic.

**Nested and discontinuous entities.** BIO assigns one tag per token, so nesting
is unrepresentable and discontinuous mentions (`neither X nor Y syndrome`) cannot
be encoded. *Symptom:* a persistent error floor on specific document types.
*Fix:* a span-based or hypergraph output structure, not a better tagger.

**Entity leakage across splits.** The same rare entity in train and test.
*Symptom:* excellent test scores and poor production performance on new
documents. *Detection:* measure the overlap of entity surface forms between
splits; it should be near zero for rare types.

**Annotation drift.** Guidelines change mid-project, so early and late documents
follow different conventions. *Symptom:* accuracy correlated with annotation
date. *Detection:* stratify evaluation by annotation batch.

**Domain shift in entity distribution.** A model trained on news encounters legal
documents where `party` names look nothing like news entities. *Symptom:* recall
collapse on a document subtype with no precision change.

**Pipeline error compounding.** Relation extraction depends on entity extraction,
so entity recall multiplies through. At 0.9 entity recall, a binary relation
requiring both endpoints has a ceiling of $0.9^2 = 0.81$ before the relation
classifier makes a single error. *This ceiling is arithmetic and is frequently
mistaken for a relation-model problem.*

## 13. Alternatives

{#tbl:extraction-approaches caption="Ways to extract structure from text, with the constraint each imposes and the regime where it wins. The first two remain in production not because they are better but because they are auditable and free."}

| Approach | Setup | Per-doc cost | Schema change | Wins when |
|---|---|---|---|---|
| Regex / gazetteer | hours | ~0 | edit a list | formats are fixed, audit matters |
| CRF over hand features | days | ~0 | retrain | tiny data, interpretability required |
| Fine-tuned encoder + CRF | weeks (labelling) | ~$10^{-6}$ | relabel + retrain | high volume, stable schema |
| LLM prompting | hours | ~$10^{-3}$ | edit the prompt | schema moves, volume is low |
| LLM + fine-tuned encoder cascade | weeks | mixed | partial | high volume, long-tail types |

**What is genuinely different here versus merely cheaper.** Regex and gazetteers
compute an exact match against a fixed pattern — a different function, not a
worse approximation, and the correct answer when the target really is a fixed
format like an invoice number. The encoder and the LLM approximate the same
function; the LLM does it zero-shot with a schema in the prompt, the encoder does
it after seeing labelled examples.

**Why the encoder has not been displaced.** {{sec:9-practical-example}} gives the
ratio: about three orders of magnitude on per-document cost and about two on
latency. When neither matters, the LLM wins on flexibility and usually on quality
for rare types. When either matters, it does not.

## 14. Evaluation

**Is the implementation correct?**

1. **Round-trip BIO** — `spans → tags → spans` must be exact, including for
   adjacent same-type entities.
2. **Subword alignment** — assert that every gold entity's first token maps to a
   labelled position, on real data, before training.
3. **Ill-formed output rate** — should be exactly zero with constrained decoding
   and is a measured quantity without it.
4. **Class balance sanity** — the `O` fraction in predictions should resemble the
   `O` fraction in gold.

**Is the model good?**

1. **Entity-level precision, recall, and F1 per type** {{eq:entity-f1}}. Per
   type, because a macro average hides a collapsed rare class and a micro average
   is dominated by the common one ({{ch:ml-metrics}}).
2. **Inter-annotator agreement as the ceiling.** Report the model's score
   alongside it; a model at 0.88 where annotators agree at 0.90 is close to
   saturated, and further modelling work is misdirected.
3. **A boundary-error breakdown** — how many errors are wrong type versus wrong
   boundary. They have different fixes: type errors want more data, boundary
   errors want better tokenization or annotation guidelines.
4. **Held-out documents, not held-out sentences**, when deployment sees whole
   documents.

## 15. Advanced Concepts

**Span-based models.** {{maturity:ESTABLISHED}} Rather than tagging tokens,
enumerate candidate spans and classify each. This handles nesting naturally at
the cost of $O(T^2)$ candidates, usually pruned by a maximum span width.

**Joint entity and relation extraction.** {{maturity:EMERGING}} Predicting both
in one model to avoid the compounding of
{{sec:12-failure-modes}}. It helps, and less than the pipeline arithmetic
suggests it should, because the joint model's own errors correlate.

**Few-shot and zero-shot extraction.** {{maturity:EMERGING}} Prompting a
generative model with a schema and a handful of examples
({{ch:llm-structured-output}}). The dominant approach for new schemas and long-
tail types, and still expensive at volume.

**Weak supervision.** {{maturity:ESTABLISHED}} Generate noisy labels from rules,
gazetteers, and heuristics, model the noise, and train on the result. The
standard answer when labels cost $1.50 each and you need 2,000 of them.

**Constrained decoding beyond BIO.** {{maturity:ESTABLISHED}} The CRF's core idea
— restrict the output space so invalid structures are unreachable — is exactly
what grammar-constrained decoding does for JSON in
{{ch:llm-structured-output}}. Same principle, a different structure, twenty years
apart.

## 16. Connection to Previous Chapters

**Backwards.** {{ch:nlp-bert}} supplied the encoder, and this chapter is the
fine-tuning half of that recipe. {{ch:nlp-subword}} supplied the subword units
whose misalignment with word-level labels is this chapter's most common bug.
{{ch:ml-metrics}} supplied precision, recall, and the micro/macro distinction —
and {{eq:entity-f1}} is that machinery applied to a unit of evaluation the
chapter did not anticipate. {{ch:nlp-subword}}'s Viterbi recurrence
{{eq:viterbi-recurrence}} returns unchanged as the CRF's decoder.
{{ch:ds-leakage}} supplied the entity-overlap leakage pattern.
{{ch:mle-splits}} supplied the reason to split by document.

**Forwards.** {{ch:llm-structured-output}} is this chapter's task performed by a
generative model, with the CRF's constraint idea reappearing as grammar-
constrained decoding. {{ch:nlp-similarity}} builds the bi-encoder that turns
these documents into retrievable vectors. {{part:12}} uses span extraction as the
grounding mechanism for citations, which is {{cite:rajpurkar2016}}'s format doing
a new job. {{part:25}} generalises the "unit of evaluation is a modelling
decision" lesson to generative systems, where the unit is far less obvious than a
span.

## 17. Exercises

**Beginner**

1. Encode `[Marie Curie]_PER won the [Nobel Prize]_MISC in [1903]_DATE` in BIO.
2. Why does BIO need `B-` at all? Give a sentence that breaks without it.
3. Why is accuracy a poor metric for sequence labelling?

**Intermediate**

4. Compute entity-level and token-level P/R/F1 for gold
   `B-PER I-PER O B-ORG I-ORG I-ORG` and prediction
   `B-PER I-PER O B-ORG I-ORG O`.
5. Using {{eq:token-recall}}, find the entity length at which token-level recall
   exceeds 0.9 while entity-level recall is 0.
6. A relation extractor scores 0.95 given gold entities and 0.70 end to end, with
   entity recall 0.85. How much of the drop is arithmetic?

**Advanced**

7. Derive {{eq:forward-recurrence}} from {{eq:forward-alpha}} and state exactly
   which property of {{eq:crf-score}} makes the factorisation valid.
8. Prove that constrained Viterbi returns the highest-scoring **well-formed**
   sequence, and give an emission matrix where repairing the independent argmax
   returns a different, lower-scoring one.
9. Design an output structure for nested entities and state its decoding
   complexity.

**Implementation**

10. Extend `crf-decoding` with CRF training: implement the negative log
    likelihood $-s(\vec{x},\vec{y}) + \log Z$ and fit the transition matrix by
    gradient descent on synthetic well-formed sequences. Verify the learned
    `O`→`I-` entries go strongly negative.
11. Implement subword alignment: given word-level BIO labels and a subword
    tokenizer, produce subword labels, and write a test that catches an
    off-by-one.
12. Implement the span-extraction head {{eq:span-head}} with the $i \le j$
    constraint, and compare against an unconstrained argmax on how often it
    produces an invalid span.
13. Build the entity-leakage detector: given train and test splits, report the
    fraction of test entity surface forms appearing in train, broken down by
    entity frequency.

**Reasoning**

14. A model reports 0.94 entity F1 and annotators agree at 0.89. What can and
    cannot be concluded?
15. Argue for the encoder route and then for the LLM route for a system
    extracting six entity types from 200 documents per day with a schema that
    changes monthly. Name the fact that decides it.

## 18. Interview Questions

**Beginner**

1. What is named-entity recognition and what is BIO tagging?
2. Why is `B-` needed in addition to `I-`?
3. What is the difference between classification and sequence labelling?

**Intermediate**

4. Why is entity-level F1 lower than token-level F1?
5. What does a CRF add over independent per-token classification?
6. How do you handle the mismatch between word-level labels and subword tokens?

**Senior**

7. Encoder or LLM for extraction? Walk through the decision with numbers.
8. Your NER model scores 0.92 offline and performs badly in production.
   Enumerate causes in order of likelihood.
9. How do you set a confidence threshold for an extraction system feeding an
   automated process?

**Systems**

10. Design an extraction pipeline for 50,000 documents a day with a schema that
    changes monthly. Address versioning, re-extraction of old documents, and
    evaluation.
11. How would you build the labelling process, and how would you know the labels
    are good enough?

## 19. Research Questions

**How much of published NER progress is annotation, not modelling?** CoNLL-2003
scores have risen steadily for two decades and the dataset's own annotation is
imperfect. What is the agreement ceiling on that dataset, and how close to it are
the current numbers? If the answer is "at it", the leaderboard has been measuring
noise for years.

**Does joint entity-relation extraction beat the pipeline once error correlation
is accounted for?** The pipeline ceiling is $0.9^2$ arithmetic; joint models
report gains. Design the comparison so the joint model's correlated errors are
visible rather than averaged away.

**Where exactly is the encoder/LLM crossover, per entity type?** Common types
almost certainly favour the encoder and rare ones the LLM. Measure per-type F1
and per-type cost, and produce the routing rule rather than the aggregate
comparison. Directly useful and not published.

**Is constrained decoding still necessary with a strong enough encoder?** A
sufficiently good tagger rarely emits ill-formed sequences even unconstrained.
Measure the rate as a function of encoder quality, and determine whether the CRF
still earns its place or has become a free but useless safety net.

## 20. Chapter Summary

Classification, sequence labelling, and span extraction are three output
structures over one encoder. Only the head and the loss differ
({{eq:classification-head}}, {{eq:tagging-head}}, {{eq:span-head}}), which is
what makes the pretrain-once recipe economically sensible.

BIO tagging squeezes a span problem into a per-token problem, and every awkward
consequence follows from that squeeze. Independent per-token classification can
emit ill-formed sequences — `I-PER` continuing nothing — because the well-formed
sequences are a vanishing fraction of the output space
{{eq:wellformed-fraction}} and the constraint is not in the model's hypothesis
class. The **CRF** {{eq:crf-score}} scores the whole sequence with a learned
transition matrix, computes its partition function exactly in
$O(T|\mathcal{T}|^2)$ by the forward algorithm {{eq:forward-recurrence}}, and
decodes with Viterbi — making illegal transitions unreachable rather than
unlikely, at a cost that is negligible beside the encoder.

**{{cite:tjongkimsang2003}}'s evaluation decision is the chapter's most portable
lesson.** Scoring whole spans rather than tokens means a single dropped token
turns a token-level recall of $1 - 1/L$ into an entity-level recall of zero
{{eq:token-recall}}, {{eq:entity-recall}}. Both numbers are correctly computed
and only one describes what a downstream consumer experiences. The unit of
evaluation is a modelling decision.

Relation extraction compounds errors multiplicatively: at 0.9 entity recall a
binary relation is capped at 0.81 before the relation model errs at all — an
arithmetic ceiling regularly mistaken for a modelling failure.

The task has largely moved to prompting a generative model, and the move is not
total. The encoder is roughly three orders of magnitude cheaper per document and
two orders faster; the LLM handles a schema change with a prompt edit rather than
a relabelling project. Which fact dominates is what decides the architecture, and
the common answer is a cascade that uses both.

## 21. Further Reading

{{cite:tjongkimsang2003}} is five pages and worth reading in full, mostly for §2
and the evaluation definition. It is a shared-task description rather than a
research paper, and the durable content is a set of decisions — four entity
types, span-level scoring, BIO encoding — that outlived every model evaluated
under them.

{{cite:lample2016}} for §2's BiLSTM-CRF. The transformer replaced the BiLSTM and
left the CRF where it was, which is a good reason to read the CRF half carefully
and the architecture half quickly.

{{cite:rajpurkar2016}} for the span-extraction format. Read §2 on the collection
methodology, which is unusually explicit about how the dataset's construction
shaped what the task rewards — and therefore what the models learned to do.

{{cite:devlin2019bert}} §4.3 and §4.4 give the fine-tuning setups for token-level
and span tasks in about a page. It is the concrete counterpart to
{{sec:5-formal-explanation}}'s three heads.

**Where to go next:** {{ch:nlp-similarity}} is the last chapter of this part and
the one that reaches furthest forward — it builds the bi-encoder that
{{part:11}} and {{part:12}} are constructed on.

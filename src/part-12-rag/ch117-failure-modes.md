---
id: rag-failures
number: 117
part: XII
tier: full
status: draft
requires: [rag-why, rag-ingestion, rag-chunking, rag-indexing, rag-generation,
           rag-query-understanding, rag-advanced-retrieval, rag-corrective,
           rag-agentic, rag-structured, llm-hallucination]
provides: [stage-localisation, oracle-substitution, prefix-gap-decomposition,
           symptom-stage-collapse, diagnostic-probes, rag-debugging-procedure]
citations: [barnett2024sevenfailures, es2023ragas, gao2023ragsurvey,
            liu2023lost, lewis2020rag]
---

## 1. Learning Objectives

By the end of this chapter you will be able to write a RAG pipeline's accuracy as
a cascade and show that the marginal value of improving any stage is proportional
to the product of the others — so localisation must precede optimisation;
demonstrate that one-at-a-time oracle substitution is **biased against downstream
stages** and that cumulative prefix substitution decomposes the gap exactly;
prove that the symptoms a user can report do not identify the failing stage, and
measure how badly; design four probes that read the system's *state* rather than
its output and localise a single failure to about 95%; and run the whole
procedure as a repeatable routine rather than as an investigation.

## 2. Why This Matters

This is the chapter the part exists for.

Eleven chapters have each fixed one thing. In production you do not have a
chunking problem or a retrieval problem — you have a **user saying the answer was
wrong**, and four or five stages that could have caused it, with fixes that have
nothing in common. Re-chunking will not help a document that was never ingested.
A better embedding model will not help a generator that ignores its context.

Two properties make this harder than debugging ordinary software, and both are
worth naming.

**The output is fluent regardless of which stage failed.** A missing document, an
unfindable chunk, a failed retrieval, and a generator that ignored good context
all produce a confident paragraph. {{sec:9-practical-example}} measures the
consequence: the best *possible* classifier that sees only the symptom gets the
stage right **51%** of the time — and never once identifies ingestion or indexing
failures at all, because their symptoms are indistinguishable from retrieval's.

**And the stages mask each other.** The obvious way to find the bottleneck —
replace one stage with a perfect version, see how much the score rises — is
**biased**, and biased in the expensive direction. {{sec:9-practical-example}}
finds a stage worth 0.121 measured alone and **0.250** measured in repair order:
its headroom was hidden behind a broken upstream stage. A team following the
naive numbers works the pipeline in an order that keeps under-delivering.

{{cite:barnett2024sevenfailures}} is one of very few sources about the failure
surface rather than the technique, and its central claim is the honest one:
**validation of a RAG system is only feasible during operation.** This chapter
gives you the procedure to run when operation tells you something is wrong.

## 3. Prerequisites

All of {{part:12}}, since the diagnosis is over its stages. Specifically:
{{ch:rag-ingestion}} for {{eq:ingestion-loss}} and {{eq:true-recall}}, the stage
nobody measures; {{ch:rag-indexing}} for {{eq:index-staleness}};
{{ch:rag-generation}} for {{eq:attribution-rate}} and why a wrong answer arrives
with citations; {{ch:rag-corrective}} for graders, which are diagnostic
instruments as well as handlers; {{ch:llm-hallucination}} for
{{eq:groundedness}}.

## 4. Intuitive Explanation

### One symptom, five causes, five unrelated fixes

A user reports: *"It told me our EMEA refund window is 14 days. It's 30."*

Five things could have happened, and they are genuinely different bugs:

```text
   1. INGESTION   the policy PDF's table never parsed; "30 days" is not
                  in the corpus at all                      -> fix the parser
   2. INDEXING    it parsed, but the chunk splitting put "30" in one chunk
                  and "refund window" in another            -> fix chunking
   3. RETRIEVAL   the chunk exists and is findable, and the query "refund
                  window EMEA" did not rank it top-k        -> fix retrieval
   4. GENERATION  the right chunk was in the prompt and the model answered
                  from its own priors                       -> fix the prompt
   5. THE CORPUS  the document really does say 14 days, and it is out of
                  date                                      -> fix the source
```

**The answer looked identical in all five cases.** That is the problem this
chapter solves, and stage 5 is worth keeping on the list because a
correctly-functioning RAG system faithfully reporting a stale document is the one
failure no amount of retrieval engineering touches.

### The two questions, and they are different

**"Which stage should we work on?"** is a question about a *distribution* of
queries, answered by measuring headroom over an evaluation set. Weekly.

**"What went wrong with this query?"** is a question about *one* query, answered
by probing the system's state. Daily, and it is what an on-call engineer needs.

They need different instruments, and confusing them is why RAG debugging so often
consists of re-reading prompts. The rest of the chapter builds one instrument for
each.

### Why localisation has to come first

Stages multiply. If ingestion is at 0.55, then **45% of queries are already lost
before retrieval runs** — so a heroic retrieval improvement moves end-to-end
accuracy by almost nothing, and the team that made it will conclude that
retrieval was not the problem. It was not, *yet*.

The general form: **the return on improving a stage is proportional to the
product of every other stage.** Work on the wrong one and the return is not small,
it is close to zero. This is why "we tried a better embedding model and it did
nothing" is such a common report, and why it is usually not evidence about the
embedding model.

### Why symptoms cannot localise, and states can

Here is the load-bearing observation of the chapter:

> The stages are nearly **indistinguishable in their output** and perfectly
> **distinguishable in their state.**

A chunk is either in the index or it is not — that is a fact you can check in
milliseconds, with no judgement involved. Whether an answer's wrongness "feels
like" a retrieval problem is a judgement, and {{sec:9-practical-example}} shows it
is worth about as much as guessing.

So the diagnostic move is always the same: **stop studying the answer and go and
look at the artefact the stage was supposed to produce.** Four such checks get to
95%.

## 5. Formal Explanation

### 5.1 The pipeline as a cascade

Let $p_1, \dots, p_n$ be the conditional success of each stage, and let $\lambda$
be the **parametric leak**: the probability the generator answers correctly
anyway, from its own weights, when the pipeline failed to deliver.

$$ E = \Big(\prod_{i=1}^{n-1} p_i\Big) p_n \;+\; \Big(1 - \prod_{i=1}^{n-1} p_i\Big) \lambda\, p_n $$ (eq:stage-cascade)

The leak is small and it matters, because it is why {{eq:stage-cascade}} is not
purely multiplicative and why a broken RAG system still appears to work
occasionally. It is also {{ch:rag-corrective}}'s parametric fallback, arriving
uninvited.

Differentiating in $p_j$ for an upstream stage:

$$ \frac{\partial E}{\partial p_j} = (1 - \lambda)\, p_n \prod_{i \ne j,\, i < n} p_i $$ (eq:marginal-stage-value)

**The return on a stage is proportional to the product of all the others.** At
$p_{\text{ingestion}} = 0.55$, every downstream improvement is discounted by
0.55 before it reaches the user. {{eq:marginal-stage-value}} is the formal reason
localisation precedes optimisation, and it is a stronger statement than "find the
bottleneck": there is no bottleneck, there is a *product*, and every factor
scales every other.

### 5.2 Oracle substitution

Define the score with stage set $A$ replaced by perfect versions:

$$ E(A) = E\big(p_i = 1 \;\forall i \in A\big), \qquad E(\varnothing) = E, \qquad E(\text{all}) = \text{ceiling} $$ (eq:oracle-substitution)

The **individual headroom** of stage $j$ is $H_j = E(\{j\}) - E(\varnothing)$, and
it is the number most teams compute. It has two problems.

$$ \sum_j H_j \ne E(\text{all}) - E(\varnothing) $$ (eq:individual-overlap)

The individual headrooms **do not decompose the gap** — they overlap, because two
stages both being broken means fixing either one alone leaves the query lost. And
worse, the error has a direction:

$$ H_j \;\le\; \Delta_j \equiv E\big(\{1..j\}\big) - E\big(\{1..j{-}1\}\big) \quad\text{for downstream } j $$ (eq:downstream-underrating)

**Individual substitution systematically under-rates downstream stages**, because
their headroom is masked by whatever is broken upstream.

### 5.3 Prefix substitution decomposes exactly

Order the stages and substitute cumulatively. The increments telescope:

$$ \sum_{j=1}^{n} \Delta_j = \sum_{j=1}^{n} \Big[E(\{1..j\}) - E(\{1..j{-}1\})\Big] = E(\text{all}) - E(\varnothing) $$ (eq:prefix-decomposition)

**Exactly the gap, with no overlap and no residual.** Each $\Delta_j$ is measured
against a world in which every earlier stage already works — which is precisely
the world in which you will be fixing stage $j$, because repairs happen in
pipeline order.

> **The honest caveat.** {{eq:prefix-decomposition}} telescopes for *any*
> ordering, so the *sum* is order-free but the *attribution* is not. Pipeline
> order is used because it is the order repairs must occur in: you cannot fix
> retrieval for a document that was never ingested. The order-free attribution is
> the Shapley value over stage subsets, at $2^n$ evaluations.

### 5.4 Symptoms do not identify stages

Let $S$ be the observable symptom and $C$ the true cause. Diagnosis from the
symptom alone is bounded by the Bayes-optimal classifier

$$ \hat{C}(s) = \arg\max_c \Prob[C = c]\,\Prob[S = s \mid C = c] $$ (eq:map-triage)

and its accuracy is bounded by how much the likelihoods differ. For RAG they
barely do:

$$ \Prob[S \mid C = \text{ingestion}] \;\approx\; \Prob[S \mid C = \text{indexing}] \;\approx\; \Prob[S \mid C = \text{retrieval}] $$ (eq:symptom-collapse)

because all three deliver *no relevant context* to the generator, which then
behaves identically. {{sec:9-practical-example}} measures the resulting ceiling at
**0.514**, with ingestion and indexing never named at all.

**One symptom escapes {{eq:symptom-collapse}}** and it is worth building a triage
form around: *the answer contradicts the retrieved text*. That symptom requires
the right text to have arrived, so it is evidence about generation specifically —
and it is measurable automatically, as {{ch:rag-generation}}'s
{{eq:post-verification-rate}}.

### 5.5 Probes read state, not output

A probe answers a yes/no question about an artefact:

$$ \begin{aligned} \pi_1 &: \text{is the gold text in the parsed corpus?} \\ \pi_2 &: \text{is it inside an indexed chunk?} \\ \pi_3 &: \text{did that chunk return for this query?} \\ \pi_4 &: \text{given that chunk in the prompt, does the model answer correctly?} \end{aligned} $$ (eq:diagnostic-probes)

Each probe is a *fact*, not a judgement, and the first three cost a string search
and a retrieval call. Run in pipeline order with early exit, they identify the
first stage that failed:

$$ \hat{C} = \min\{\, j : \pi_j = \text{false} \,\} $$ (eq:probe-attribution)

and {{sec:9-practical-example}} measures **0.952** against the symptom's 0.514.

$$ \boxed{\text{stages are indistinguishable in output and distinguishable in state}} $$ (eq:state-versus-output)

### 5.6 The fifth stage

{{eq:diagnostic-probes}} assumes a *gold text exists*. When $\pi_1$ fails, two
different worlds are consistent with it: the document was mis-parsed, or **the
corpus is simply wrong or stale**. {{ch:rag-why}}'s {{eq:rag-ceiling}} bounds the
system by what the corpus contains, so:

$$ \pi_0 : \text{does any document in the source system state the correct answer?} $$ (eq:corpus-probe)

**Run $\pi_0$ first.** It is the cheapest probe and the only one whose failure
means the RAG system is working correctly and the answer is still wrong.

## 6. Mathematical Foundation

### 6.1 Marginal value, worked

Take $p = (0.55, 0.90, 0.80, 0.75)$ and $\lambda = 0.12$. Pipeline delivery is
$0.55 \times 0.90 \times 0.80 = 0.396$, so

$$ E = 0.396 \times 0.75 + 0.604 \times 0.12 \times 0.75 = 0.297 + 0.054 = 0.351 $$ (eq:cascade-worked)

against a simulated 0.349. Now the marginal value of a ten-point gain in each
upstream stage, via {{eq:marginal-stage-value}}:

$$ \Delta_{\text{ingest}} = 0.10 \times 0.88 \times 0.72 \times 0.66 = 0.042, \qquad \Delta_{\text{retr}} = 0.10 \times 0.88 \times 0.495 \times 0.66 = 0.029 $$ (eq:marginal-worked)

using $(1-\lambda)p_n = 0.66$ and the appropriate partial products. **The same
ten-point improvement is worth 45% more at the ingestion stage than at
retrieval**, purely because of what sits around it. Nothing about the difficulty
of the two projects enters this calculation, which is the point: the ordering is
a property of the pipeline, not of the work.

### 6.2 Why individual headroom under-rates the tail

Consider two stages, $p_1$ and $p_2$, ignoring the leak. Then
$E = p_1 p_2$ and

$$ H_1 = p_2(1 - p_1), \qquad H_2 = p_1(1 - p_2), \qquad H_1 + H_2 = p_1 + p_2 - 2p_1p_2 $$ (eq:two-stage-headroom)

while the gap is $1 - p_1 p_2$. The shortfall is

$$ (1 - p_1p_2) - (H_1 + H_2) = (1-p_1)(1-p_2) $$ (eq:headroom-shortfall)

**exactly the probability that both stages fail** — the queries no single fix
recovers, credited to nobody. At $p_1 = 0.55$, $p_2 = 0.75$ that is $0.45 \times
0.25 = 0.113$, an eighth of the whole range, invisible in a one-at-a-time
breakdown.

And it is worse for the downstream stage specifically, because $H_2$ is scaled by
$p_1$ — the broken stage — while $\Delta_2$ is scaled by 1.

$$ \frac{\Delta_2}{H_2} = \frac{1}{p_1} $$ (eq:underrating-factor)

At $p_1 = 0.55$ the second stage is under-rated by a factor of **1.8**, and the
measured ratio in {{sec:9-practical-example}} is 0.250 / 0.121 ≈ 2.1 — the extra
coming from the two stages in between.

> **MATH NOTE:** {{eq:underrating-factor}} says the bias is worst exactly when the
> upstream stage is most broken — i.e. precisely in the situation where a team is
> most likely to be running this diagnosis. The instrument is least trustworthy
> when it is most needed, which is a good reason to prefer
> {{eq:prefix-decomposition}} even though it costs $n$ evaluations instead of
> $n$… which is to say, it costs exactly the same. There is no reason to use the
> biased version.

### 6.3 The information in a symptom

From {{eq:symptom-collapse}}, if three of four causes share a likelihood row, the
symptom carries no information distinguishing them, and the MAP rule assigns all
three to the most probable one. With priors $(0.22, 0.13, 0.40, 0.25)$ the
collapsed mass goes to retrieval, so:

$$ \text{accuracy} \approx \underbrace{0.40}_{\text{retrieval, right by default}} + \underbrace{0.25 \times 0.74}_{\text{generation, genuinely separable}} \approx 0.585 $$ (eq:triage-ceiling)

against a measured 0.514, the difference being the generation symptom leaking
into the collapsed group. **Both numbers say the same thing:** a symptom-based
triage is a prior with extra steps, and it will never name two of your four
stages.

## 7. Internal Mechanics

```mermaid {#fig:rag-diagnosis caption="The procedure. Probes run in pipeline order with early exit, and each reads an artefact rather than judging an answer (eq:state-versus-output). The pi-0 branch is the one most procedures omit: a correctly functioning system faithfully reporting a stale document is not a retrieval bug, and no retrieval work will fix it."}
flowchart TB
    F["a wrong answer"] --> P0{"does any source document<br/>state the correct answer?"}
    P0 -->|"no"| C0["NOT a RAG bug:<br/>the corpus is wrong or stale"]
    P0 -->|"yes"| P1{"is the gold text in<br/>the parsed corpus?"}
    P1 -->|"no"| C1["INGESTION<br/>fix the parser"]
    P1 -->|"yes"| P2{"is it inside an<br/>indexed chunk?"}
    P2 -->|"no"| C2["INDEXING / CHUNKING<br/>fix boundaries or refresh"]
    P2 -->|"yes"| P3{"did that chunk return<br/>for this query?"}
    P3 -->|"no"| C3["RETRIEVAL<br/>query, hybrid, rerank"]
    P3 -->|"yes"| P4{"with that chunk in the prompt,<br/>is the answer right?"}
    P4 -->|"no"| C4["GENERATION<br/>prompt, ordering, budget"]
    P4 -->|"yes"| C5["NON-DETERMINISM<br/>or a stale reproduction"]
```

### 7.1 The four probes, concretely

| Probe | How to run it | Cost |
|---|---|---|
| $\pi_0$ | search the *source system* for the fact | a human minute |
| $\pi_1$ | exact-string grep over the parsed text store | milliseconds |
| $\pi_2$ | look the text up in the chunk store; check the chunk is whole | milliseconds |
| $\pi_3$ | run the query, check whether that chunk id is in the results | one retrieval |
| $\pi_4$ | put the gold chunk in the prompt and re-generate | one LLM call |

**Keeping the parsed text store is what makes $\pi_1$ possible**, and it is the
single highest-value piece of debugging infrastructure in a RAG system. Systems
that parse straight into chunks and discard the intermediate cannot distinguish
ingestion from chunking failures at all, ever.

### 7.2 The last box is not a joke

If all four probes pass and the answer was still wrong, the usual causes are
mundane and worth checking before anything sophisticated: temperature above zero
with a genuinely ambiguous prompt; a different index version between the failing
request and the reproduction; a per-tenant filter ({{ch:rag-indexing}}) that
applied for the user and not for you; or a conversation history that changed the
effective query.

**The fourth of those is the most common and the least often checked.** In a
multi-turn system the retrieval query is not what the user typed — a fact
{{ch:rag-query-understanding}} established and which reproduction scripts
routinely ignore.

### 7.3 Fixes by stage

| Stage | The fixes, in order of usual return |
|---|---|
| corpus | fix the source; add freshness metadata; make staleness visible in the answer |
| ingestion | better parser for the failing format; measure {{eq:ingestion-loss}}; consider visual retrieval ({{ch:rag-structured}}) |
| indexing | chunk boundaries; heading-path augmentation; incremental refresh; check {{eq:index-staleness}} |
| retrieval | hybrid ({{ch:emb-hybrid}}); rerank ({{ch:emb-reranking}}); query rewriting ({{ch:rag-query-understanding}}); parent–child ({{ch:rag-advanced-retrieval}}) |
| generation | ordering ({{eq:u-shape-ordering}}); explicit abstention instructions; smaller $k$; citation verification |

**Notice how little overlap there is between rows.** That is the whole reason
localisation is worth this much effort: a fix from the wrong row does not
partially help, it does nothing.

## 8. Implementation

```python {tier=A name=stage-attribution-ladder}
"""Which stage to fix, and why the obvious way of deciding is biased.

A RAG pipeline is a chain of stages, and end-to-end accuracy tells you nothing
about which one is losing the queries. The standard localisation move is ORACLE
SUBSTITUTION: replace one stage with a perfect version and see how much the score
rises (eq:oracle-substitution).

Done one stage at a time, that measurement is biased in a direction that matters:
a downstream stage's headroom is HIDDEN while an upstream stage is broken, so the
procedure systematically under-rates exactly the stages you will need to fix
second (eq:downstream-underrating). Substituting a cumulative PREFIX of stages
instead gives increments that telescope, and therefore decompose the total gap
exactly (eq:prefix-decomposition).

This listing measures both on the same pipeline and compares the rankings.
"""
import numpy as np

rng = np.random.default_rng(97)

N = 200_000
STAGES = ("ingestion", "indexing", "retrieval", "generation")

# Per-stage conditional success. Ingestion is the badly broken one here, which is
# common and rarely where teams look first (ch:rag-ingestion).
BASE = {"ingestion": 0.55, "indexing": 0.90, "retrieval": 0.80, "generation": 0.75}

# The generator sometimes answers correctly from its own weights even when
# retrieval failed. That leak is what makes the pipeline non-multiplicative, and
# it is why this has to be simulated rather than multiplied out.
PARAMETRIC_LEAK = 0.12


def score(p):
    """End-to-end accuracy of the pipeline with per-stage success rates p."""
    reached = np.ones(N, dtype=bool)
    for s in ("ingestion", "indexing", "retrieval"):
        reached &= rng.random(N) < p[s]
    gen_ok = rng.random(N) < p["generation"]
    leak = (~reached) & (rng.random(N) < PARAMETRIC_LEAK) & gen_ok
    return float(((reached & gen_ok) | leak).mean())


def with_oracle(subset):
    p = dict(BASE)
    for s in subset:
        p[s] = 1.0
    return p


base = score(BASE)
ceiling = score(with_oracle(STAGES))

print(f"pipeline: " + ", ".join(f"{s} {BASE[s]:.2f}" for s in STAGES))
print(f"end-to-end accuracy {base:.3f}; perfect-pipeline ceiling {ceiling:.3f}; "
      f"gap {ceiling - base:.3f}\n")

print(f"{'stage':<14}{'fix it alone':>15}{'fix it in order':>18}{'':>4}"
      f"{'naive rank':>12}{'true rank':>11}")
print("-" * 74)

single = {s: score(with_oracle([s])) - base for s in STAGES}

prefix, prev = {}, base
for i, s in enumerate(STAGES):
    cur = score(with_oracle(STAGES[:i + 1]))
    prefix[s] = cur - prev
    prev = cur

naive_rank = {s: i + 1 for i, s in enumerate(sorted(STAGES, key=lambda x: -single[x]))}
true_rank = {s: i + 1 for i, s in enumerate(sorted(STAGES, key=lambda x: -prefix[x]))}

for s in STAGES:
    print(f"{s:<14}{single[s]:>15.3f}{prefix[s]:>18.3f}{'':>4}"
          f"{naive_rank[s]:>12}{true_rank[s]:>11}")

print("-" * 74)
print(f"{'sum':<14}{sum(single.values()):>15.3f}{sum(prefix.values()):>18.3f}"
      f"     (gap is {ceiling - base:.3f})")

print(f"""
Look at the two sums before anything else. The prefix increments add to
{sum(prefix.values()):.3f}, which is the gap exactly -- they telescope, because
each one is measured against a pipeline in which every earlier stage is already
perfect. The one-at-a-time jumps add to {sum(single.values()):.3f}, which is not
the gap and is not meant to be: those measurements overlap, and reporting them as
a breakdown implies an additivity they do not have.

The bias has a direction, and it is the direction that costs you. Fixing
generation alone is worth {single['generation']:.3f}, which looks like the
smallest project on the board. Fixed IN ORDER -- after the upstream stages are
working -- the same stage is worth {prefix['generation']:.3f}. The headroom was
always there; it was hidden, because a query the ingester dropped fails whether
or not the generator is any good.

So the one-at-a-time procedure systematically under-rates downstream stages while
anything upstream is broken. That is not a small distortion: here it changes the
ranking, and a team following the naive numbers would work through the pipeline
in an order that keeps under-delivering, because each fix unmasks the next.

One honesty note about the right-hand column before using it. Prefix increments
telescope for ANY ordering of the stages, so they always sum to the gap -- but
how much of the gap each stage is credited with DOES depend on the order chosen.
Pipeline order is used here because it is the order in which repairs actually
have to happen: you cannot fix retrieval for a document that was never ingested.
A fully order-free attribution is the Shapley value over stage subsets, which
costs 2^n evaluations and, for four stages, mostly confirms the same ranking.

The rule that follows is simple and worth stating plainly. LOCALISE IN PIPELINE
ORDER. Fix the earliest broken stage first, then re-measure everything -- the
measurements downstream of a repair are stale the moment the repair lands. A
diagnosis that took a day is worth repeating after every fix, and a dashboard
that reports one-at-a-time headroom for four stages simultaneously is reporting
four numbers that were never simultaneously true.""")
```

The first listing chooses the project. The second is the one an on-call engineer
runs on a single complaint.

```python {tier=A name=symptom-versus-probe}
"""Diagnosing ONE failing query: why the symptom does not tell you the stage.

The previous listing decided which stage to work on across a whole evaluation
set. This one is the other job, and the one that actually arrives on a Tuesday:
a user reports a bad answer, and something has to decide which stage lost it.

The obstacle is that a RAG pipeline emits the same few symptoms whatever broke.
A fabricated answer looks identical whether the document was never ingested,
never retrieved, or retrieved and ignored -- because the generator writes fluent
prose in all three cases (eq:symptom-collapse). This listing measures how much a
symptom is worth, then measures four cheap PROBES against it
(eq:diagnostic-probes).
"""
import numpy as np

rng = np.random.default_rng(2024)

N = 60_000
STAGES = ["ingestion", "indexing", "retrieval", "generation"]
SYMPTOMS = ["fabricated", "says-not-found", "contradicts-source", "partial"]

# How often each stage is the true culprit.
PRIOR = np.array([0.22, 0.13, 0.40, 0.25])

# P(symptom | stage). Rows are stages. The first three rows are nearly identical
# in their first two columns, and that is the whole problem: the symptoms a user
# can report do not separate the stages that produce them.
LIKELIHOOD = np.array([
    [0.50, 0.38, 0.02, 0.10],      # ingestion: gone from the corpus entirely
    [0.46, 0.36, 0.04, 0.14],      # indexing: present but unfindable
    [0.48, 0.34, 0.04, 0.14],      # retrieval: findable but not found
    [0.20, 0.06, 0.46, 0.28],      # generation: it had the text and misused it
])

# Probe reliability. Probes read the SYSTEM, not the answer, which is why they
# separate stages that the answer cannot (eq:state-versus-output).
PROBE_ACC = {"in_corpus": 0.97, "in_index": 0.93, "in_topk": 0.98, "used": 0.90}

truth = rng.choice(len(STAGES), size=N, p=PRIOR)
symptom = np.array([rng.choice(len(SYMPTOMS), p=LIKELIHOOD[t]) for t in truth])


def guess_from_symptom():
    """The best possible symptom-only classifier: maximum a posteriori under the
    true generative model (eq:map-triage). No real triage does better."""
    post = PRIOR[:, None] * LIKELIHOOD          # stage x symptom
    best = post.argmax(axis=0)
    return best[symptom]


def guess_from_probes():
    """Four checks, run in pipeline order, each answering one yes/no question
    about the system rather than about the answer:

      in_corpus : does the gold text exist anywhere in the parsed corpus?
      in_index  : is it in a chunk that the index actually holds?
      in_topk   : did that chunk come back for this query?
      used      : given the gold chunk in the prompt, does the model answer?
    """
    out = np.empty(N, dtype=int)
    noisy = {k: rng.random(N) < v for k, v in PROBE_ACC.items()}
    for i in range(N):
        t = truth[i]
        # A probe fails at the broken stage; upstream probes pass. Each probe
        # reports correctly with its own accuracy (eq:probe-attribution).
        if t == 0:
            out[i] = 0 if noisy["in_corpus"][i] else 1
        elif t == 1:
            out[i] = 1 if noisy["in_index"][i] else 2
        elif t == 2:
            out[i] = 2 if noisy["in_topk"][i] else 3
        else:
            out[i] = 3 if noisy["used"][i] else 2
    return out


def confusion(pred, title):
    print(f"\n{title}   accuracy {float((pred == truth).mean()):.3f}")
    print(f"{'true \\ called':<16}" + "".join(f"{s[:9]:>12}" for s in STAGES))
    for i, s in enumerate(STAGES):
        row = [(float(((truth == i) & (pred == j)).sum())
                / max((truth == i).sum(), 1)) for j in range(len(STAGES))]
        print(f"{s:<16}" + "".join(f"{v:>12.2f}" for v in row))


print(f"{N:,} failing queries. True stage prior: "
      + ", ".join(f"{s} {p:.0%}" for s, p in zip(STAGES, PRIOR)))

confusion(guess_from_symptom(), "SYMPTOM ONLY (optimal MAP classifier)")
confusion(guess_from_probes(), "FOUR PROBES, in pipeline order")

print("""
The first matrix is the ceiling on what the reported symptom can tell you, and it
is low. It is not a bad classifier -- it is the OPTIMAL one under the true
generative model, so no amount of prompt engineering on a triage rubric beats it.
Read the first three rows: ingestion, indexing and retrieval failures produce
almost the same symptom distribution, so the classifier gives up and assigns
nearly all of them to whichever is most common. Two entire stages are effectively
invisible -- the ingestion and indexing COLUMNS are empty, which means this
procedure will never once name them, no matter how many tickets it processes.

The one useful signal is in the last row. "The answer contradicts the text that
was retrieved" genuinely does discriminate, because it is the only symptom that
requires the right text to have ARRIVED. That is worth building into a triage
form, and ch:rag-generation's citation verification computes it automatically.
Everything else a user can tell you is close to noise.

The second matrix is what four yes/no probes buy, and the jump is the point of
the chapter: 0.514 to 0.952. Each probe interrogates the SYSTEM rather than the
answer -- is the text in the corpus, is it in the index, did it come back, was it
used. None requires a judgement about quality, all four are cheap, and together
they locate the failure almost exactly.

Note why they work where the symptom does not. The stages are indistinguishable
in their OUTPUT and perfectly distinguishable in their STATE -- a chunk is either
in the index or it is not. So the diagnostic move is always the same: stop
studying the answer and go and look at the artefact the stage was supposed to
produce.

The probes are also cheap in the order given. Run them in pipeline order and stop
at the first failure: most investigations end after one or two, and the last and
most expensive probe -- re-running generation with the gold chunk supplied -- is
needed only for queries that got all the way through.""")
```

## 9. Practical Example

**Choosing the project.** The pipeline scores **0.349** against a perfect-pipeline
ceiling of 1.000, so there is 0.651 of gap to allocate. Two allocations, from the
same simulation:

| Stage | fix it alone | fix it in order |
|---|---|---|
| ingestion | 0.218 | 0.215 |
| indexing | 0.031 | 0.053 |
| retrieval | 0.065 | 0.134 |
| generation | **0.121** | **0.250** |
| **sum** | **0.436** | **0.651** |

**The prefix increments sum to 0.651 — the gap, exactly**, by
{{eq:prefix-decomposition}}'s telescoping. The one-at-a-time jumps sum to 0.436,
which is not the gap and never was: {{eq:individual-overlap}}, with
{{eq:headroom-shortfall}}'s "both stages failed" mass credited to nobody.

**And the rankings disagree at the top.** Generation is worth 0.121 alone — the
smallest-looking project but one — and **0.250** in repair order, the largest.
{{eq:underrating-factor}} predicted a factor of $1/p_1 = 1.8$ from the ingestion
stage alone; the measured 2.1 includes the two stages in between. The headroom was
always there and it was masked, because a query the ingester dropped fails
whether or not the generator is any good.

> **IMPORTANT:** The practical consequence is a rule, not a caveat. **Localise in
> pipeline order, fix the earliest broken stage, then re-measure everything.**
> Every downstream measurement is stale the moment a repair lands upstream, so a
> dashboard showing one-at-a-time headroom for four stages is showing four
> numbers that were never simultaneously true. Both procedures cost $n$
> evaluations, so there is no reason to run the biased one.

**Diagnosing one complaint.** The symptom-only classifier reaches **0.514** — and
it is not a weak classifier, it is the Bayes-optimal one under the true model, so
no triage rubric can beat it. Its confusion matrix is the finding:

**the ingestion and indexing columns are empty.** Not inaccurate — *empty*. The
procedure never names those stages, for any ticket, ever, because
{{eq:symptom-collapse}} makes their symptoms indistinguishable from retrieval's
and the prior sends the collapsed mass to retrieval. A team triaging by symptom
will conclude their system has a retrieval problem, permanently, whatever it
actually has.

The one exception is the generation row, at 0.74. **"The answer contradicts the
retrieved text" is the only symptom that requires the right text to have
arrived**, so it is genuinely diagnostic — and it is computable automatically by
{{ch:rag-generation}}'s citation verification rather than reported by a user.

**Four probes take it to 0.952.** Ingestion 0.97, indexing 0.93, retrieval 0.98,
generation 0.90 on the diagonal. The probes are not cleverer than the symptom
classifier; they are looking somewhere else. **The stages are indistinguishable in
their output and perfectly distinguishable in their state**
({{eq:state-versus-output}}), and a chunk is either in the index or it is not.

Run in pipeline order with early exit, most investigations stop after one or two
checks, and the only expensive probe — re-generating with the gold chunk supplied
— is reached only by queries that survived everything upstream.

## 10. Production Considerations

**Keep the parsed text store.** Without it $\pi_1$ is impossible and ingestion
failures are permanently indistinguishable from retrieval failures. The highest
-value debugging infrastructure in a RAG system, and it is a blob store.

**Log, per query: the retrieval query actually used, chunk ids returned, scores,
the assembled prompt's hash, and the answer.** Without the *actual* query, a
multi-turn failure cannot be reproduced ({{ch:rag-query-understanding}}).

**Run the ladder weekly**, in pipeline order, on a fixed evaluation set. It is the
only measurement that says what to work on.

**Automate $\pi_3$ and $\pi_4$** as a "diagnose this query" endpoint. If a
diagnosis takes an engineer an afternoon it will not happen; if it takes thirty
seconds it happens on every ticket.

**Version the index and record the version with each answer.** Half of
irreproducible failures are reproductions against a different index.

**Measure $\pi_0$ deliberately, at intervals.** A stale corpus is invisible to
every other instrument, and it is the one failure where the RAG system is working
perfectly.

**Track the stage distribution over time**, not just the failure rate. A shift
from retrieval to ingestion failures usually means a new document format arrived.

**Sample and diagnose successes too.** {{eq:stage-cascade}}'s leak means some
correct answers came from the model's weights rather than your corpus; they will
break silently when the model changes.

## 11. Common Mistakes

**Debugging by re-reading the prompt.** The prompt is stage four; three others
are upstream and more often at fault.

**Triaging by symptom.** {{sec:9-practical-example}}: two of four stages are
never named.

**One-at-a-time headroom presented as a breakdown.**
{{eq:individual-overlap}} — the numbers do not add up, literally.

**Optimising before localising.** {{eq:marginal-stage-value}} says the return on
the wrong stage is near zero, and near-zero results get misread as evidence about
the technique.

**Not keeping the parsed text.** Unrecoverable, and it is discovered exactly when
it is needed.

**Reproducing with the user's typed question** rather than the query the system
actually issued.

**Assuming the corpus is right.** $\pi_0$ costs a minute and is skipped
constantly.

**Stopping at the first plausible cause.** Multiple stages fail simultaneously
more often than intuition suggests — {{eq:headroom-shortfall}} puts it at 11% of
the range in the worked example.

## 12. Failure Modes

**The confident fabrication.** Fluent, cited, wrong. Cause: any of stages 1–3.
Distinguish with $\pi_1$–$\pi_3$; the symptom cannot.

**The stale-but-faithful answer.** The system correctly reports what a
superseded document says. $\pi_0$ is the only probe that catches it, and no
retrieval work fixes it. {{ch:rag-why}}'s {{eq:rag-ceiling}}.

**The half-answer.** The retrieved chunk contains part of the answer;
{{ch:rag-chunking}}'s {{eq:span-containment}} failed. Symptom: answers that are
right and incomplete, consistently, on the same document type.

**The right document, wrong chunk.** Retrieval finds the correct document and the
wrong passage within it — {{ch:rag-advanced-retrieval}}'s augmentation collapse.
Detect by scoring document-level and chunk-level recall separately.

**The permission-filtered miss.** The chunk exists, is indexed, is findable, and
this user cannot see it. Looks exactly like a retrieval failure and is a correct
security outcome. {{ch:rag-indexing}}. Reproduce as the user, not as yourself.

**The multi-turn query drift.** The issued query bears little resemblance to the
user's last message, so the failure is unreproducible from the transcript.

**The unmeasurable regression.** Answers get worse after a model upgrade and no
stage metric moves, because the change was in the *leak* — the model used to know
things and now does not, or vice versa. Detect by tracking retrieval-off accuracy
as a control.

**The evaluation set that no longer represents traffic.** Every stage metric is
green and users are unhappy. {{cite:barnett2024sevenfailures}}'s point: offline
validation is not validation.

## 13. Alternatives

| Approach | What it gives | What it misses |
|---|---|---|
| End-to-end accuracy alone | one number, cheap | which stage — the entire problem |
| Reference-free metrics ({{cite:es2023ragas}}) | continuous monitoring, no labels | judge bias; separates retrieval from generation but not ingestion from retrieval |
| Per-stage unit metrics (recall@k, groundedness) | localisation without a labelled end-to-end set | interactions, and {{eq:marginal-stage-value}}'s weighting |
| The probe ladder (this chapter) | per-query attribution at ~95% | needs a gold answer per failing query |
| Full ablation grid ($2^n$) | order-free Shapley attribution | cost, for little ranking change at $n = 4$ |
| Human review of transcripts | finds what you did not think to measure | does not scale; irreplaceable anyway |

**The second and fourth rows are complements, not competitors.** Reference-free
metrics run continuously on everything and tell you *when*; the probe ladder runs
on a handful of failures and tells you *what*. A system with only the first is
blind to cause; a system with only the second is blind until someone complains.

## 14. Evaluation

**Separate retrieval quality from generation quality in every report.** The
field's characteristic error, and {{cite:es2023ragas}} exists because of it.

**Report the stage distribution of failures**, not only their rate.

**Use prefix decomposition for headroom** ({{eq:prefix-decomposition}}), never the
one-at-a-time version.

**Keep a control condition**: the same evaluation with retrieval disabled. It
measures the leak, and it is the only way to detect a model-upgrade regression
that no stage metric shows.

**Include unanswerable questions and stale-answer questions** in the evaluation
set, at their production rates. Otherwise abstention and $\pi_0$ failures are
invisible.

**Re-derive the evaluation set from production traffic quarterly**, and treat
divergence between offline scores and user reports as evidence about the
*evaluation set*.

**Measure diagnosis time.** If localising one failure takes an afternoon, the
procedure in this chapter will not be run, and its value is zero.

## 15. Advanced Concepts

**Continuous probes as monitoring.** {{maturity:EMERGING}} Run $\pi_3$
automatically on every answered query — was the top chunk the one cited? — and
you get retrieval-quality monitoring without labels, at one comparison per query.

**Counterfactual context.** {{maturity:EMERGING}} Re-run generation with each
retrieved chunk removed in turn; the chunk whose removal changes the answer is
the one that was used. Expensive and exact, and it turns
{{ch:rag-generation}}'s attribution question into a measurement.

**Stage attribution as credit assignment.** {{maturity:EXPERIMENTAL}}
{{eq:prefix-decomposition}} is a cooperative-game decomposition, and the
order-dependence is exactly why the Shapley value exists. For $n = 4$ the full
$2^n$ grid is 16 evaluations — affordable, and it removes the last arbitrary
choice in the procedure.

**Diagnosis as a product surface.** {{maturity:EMERGING}} Showing the user which
chunks were retrieved converts an opaque failure into a reportable one: "it used
the wrong document" is a diagnosis a user can give you for free, and it maps
directly onto $\pi_3$.

**The evaluation set is a stage too.** {{maturity:MATURE}}
{{cite:barnett2024sevenfailures}}'s claim that validation is only feasible during
operation is really a claim that the evaluation set has its own recall against the
true query distribution — and nothing in the pipeline measures it. Treat set
staleness the same way you treat index staleness
({{ch:rag-indexing}}'s {{eq:index-staleness}}).

## 16. Connection to Previous Chapters

This chapter is the part's through-line landing. {{ch:rag-why}}'s
{{eq:rag-ceiling}} is $\pi_0$; {{ch:rag-ingestion}}'s {{eq:ingestion-loss}} is
$\pi_1$ and the stage this chapter shows is invisible to symptom-based triage;
{{ch:rag-chunking}}'s {{eq:span-containment}} and {{ch:rag-indexing}}'s
{{eq:index-staleness}} are $\pi_2$; {{ch:rag-query-understanding}} and
{{ch:emb-hybrid}} are the fixes behind $\pi_3$; {{ch:rag-generation}}'s
{{eq:attribution-rate}} and {{eq:post-verification-rate}} are $\pi_4$ and the one
symptom that discriminates. {{ch:rag-corrective}}'s grader is a probe you left
running in production, and {{ch:rag-agentic}}'s observability argument is this
chapter's {{eq:state-versus-output}} applied inside a loop rather than across a
pipeline. {{ch:rag-graph}} and {{ch:rag-structured}} add stages with their own
probes — extraction accuracy and schema recall — and both slot into the same
ladder.

## 17. Exercises

1. Derive {{eq:headroom-shortfall}} and interpret the residual term.
2. Prove {{eq:prefix-decomposition}} telescopes for any ordering, and construct a
   two-stage example where two orderings give different attributions.
3. In `stage-attribution-ladder`, set `PARAMETRIC_LEAK = 0`. Which numbers change
   and what does the leak represent in a real system?
4. Reorder `STAGES` in the same listing. Does the sum still equal the gap? Do the
   ranks change?
5. Compute the full $2^4$ ablation grid and the Shapley value for each stage.
   Does the ranking differ from pipeline-order prefix?
6. In `symptom-versus-probe`, degrade `in_index` accuracy to 0.6. Which
   off-diagonal cell grows, and why that one?
7. Add a fifth stage — a stale corpus — that no probe detects. What does it do to
   both matrices, and what probe would you add?
8. Write the diagnosis runbook for your own system: the exact query for each
   probe, and the expected latency of the whole ladder.

## 18. Interview Questions

1. A user says the answer was wrong. What do you check first, and why that?
2. Why does end-to-end accuracy not tell you which stage to fix?
3. Why is one-at-a-time oracle substitution biased?
4. What does prefix substitution give you that individual substitution does not?
5. Why can a user's description of the failure not identify the stage?
6. Design four probes that localise a RAG failure, in order.
7. What is the one symptom that genuinely discriminates, and why that one?
8. Your retrieval recall@10 is 0.94 and users are unhappy. Where do you look?
9. What infrastructure must exist before RAG is debuggable at all?
10. When is a wrong answer not a RAG bug?

## 19. Research Questions

1. {{eq:prefix-decomposition}}'s attribution is order-dependent. Is pipeline order
   the right convention, or is there a principled argument for Shapley in a
   pipeline where the stages are causally ordered?
2. Probes need a gold answer per failing query. Can $\pi_1$–$\pi_3$ be made
   reference-free, using the answer's own citations as the gold target?
3. {{eq:symptom-collapse}} is measured on a model of symptoms. What are the real
   likelihoods on production tickets, and does any user-reportable signal separate
   ingestion from retrieval?
4. The leak in {{eq:stage-cascade}} moves with model version and is invisible to
   stage metrics. Is there a cheap continuous estimator of it?
5. Evaluation sets go stale against the query distribution. Can staleness be
   detected automatically, by comparing embedding distributions of evaluation and
   production queries?

## 20. Chapter Summary

**A RAG failure is a fluent paragraph, and five different bugs produce the same
one.** That is why this part ends in a procedure rather than a list.

**Localisation must precede optimisation.** {{eq:stage-cascade}} makes the
pipeline a product, so {{eq:marginal-stage-value}} makes the return on any stage
proportional to all the others: at 0.55 ingestion, every downstream improvement is
discounted by 0.55 before a user sees it. Work on the wrong stage and the return
is not merely small — and it will be misread as evidence that the technique does
not work.

**One-at-a-time oracle substitution is the standard method and it is biased.**
The individual headrooms do not sum to the gap ({{eq:individual-overlap}}), and
they under-rate downstream stages by $1/p_{\text{upstream}}$
({{eq:underrating-factor}}) — measured, generation was worth 0.121 alone and
**0.250** in repair order, which changed the ranking at the top. Cumulative
prefix substitution telescopes to the gap **exactly** and costs the same number of
evaluations, so there is no reason to run the biased version.

**Symptoms cannot localise. States can.** The Bayes-optimal symptom classifier
reached **0.514**, and its ingestion and indexing columns were *empty* — a
symptom-based triage will never once name those stages, whatever is actually
wrong. Four probes that read the system's artefacts instead reached **0.952**.
{{eq:state-versus-output}}: **stop studying the answer and go and look at the
artefact the stage was supposed to produce.**

**The one exception is worth building for.** "The answer contradicts the retrieved
text" requires the right text to have arrived, so it discriminates — and
{{ch:rag-generation}}'s citation verification computes it without asking anyone.

**Run $\pi_0$ first.** A correctly-functioning system faithfully reporting a stale
document is the one failure where no retrieval work helps, and it costs a minute
to rule out.

And the part's through-line, stated one last time: **every RAG failure belongs to
exactly one stage, the stages need completely different fixes, and the answer
looks the same either way.** Which makes the ability to tell them apart — cheaply,
repeatably, in thirty seconds rather than an afternoon — the difference between a
RAG system that improves and one that is merely tuned.

## 21. Further Reading

{{cite:barnett2024sevenfailures}} first, and read it as an engineering report
rather than a taxonomy: its claim that validation is only feasible during
operation is the sentence that should shape your evaluation strategy.
{{cite:es2023ragas}} for continuous, label-free monitoring — the *when* to this
chapter's *what*, with {{part:25}}'s judge caveats attached.
{{cite:gao2023ragsurvey}} for the vocabulary to describe whichever stage you find.
{{cite:liu2023lost}} for the generation-stage failure that looks like a retrieval
failure, and which $\pi_4$ is designed to catch.
{{cite:lewis2020rag}} once more, because the original two-stage architecture is
still the object being debugged — everything in this part is a stage inserted into
it, and every stage inserted is a stage that can fail.
Within the book: {{ch:ev-rag}} for evaluation infrastructure at scale,
{{part:24}} for operating this in production, and {{ch:rag-corrective}} for
turning a probe into a runtime handler.

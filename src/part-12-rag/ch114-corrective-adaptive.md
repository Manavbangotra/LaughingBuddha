---
id: rag-corrective
number: 114
part: XII
tier: full
status: draft
requires: [rag-why, rag-generation, rag-query-understanding, rag-graph,
           llm-hallucination, llm-routing, llm-long-context]
provides: [retrieval-grading, corrective-retrieval, terminal-versus-recoverable-handlers,
           retrieval-abstention, adaptive-retrieval-routing, distraction-penalty,
           router-breakeven-accuracy]
citations: [yan2024crag, asai2023selfrag, jeong2024adaptiverag, jiang2023flare,
            es2023ragas, gao2023ragsurvey, liu2023lost]
---

## 1. Learning Objectives

By the end of this chapter you will be able to state why the standard RAG
pipeline has **no error condition** and what that costs in confident wrong
answers; write the value of a retrieval grader as a decision problem in which the
grader's own error rate is a first-class term; explain and demonstrate why a
**recoverable** handler (retry) degrades under grader noise at less than half the
rate of a **terminal** one (abstention), and why that decides the ordering of the
two; compute the accuracy a query router must exceed before it beats the better
fixed policy; and show that the difficulty and the value of routing move in
opposite directions, so the query mix — not the classifier — is the thing to
measure first.

## 2. Why This Matters

Every chapter so far has tried to make retrieval better. This one starts from the
observation that it will still fail, and asks the question the pipeline never
asks: **what happens then?**

The answer, in the architecture everyone ships, is nothing. Retrieval returns
$k$ chunks whether or not any of them is relevant, and the generator writes an
answer from whatever arrived — fluent, cited, and wrong.
{{ch:rag-generation}}'s citations make it *more* convincing, not less. There is
no branch, no threshold, no error path: **a failed retrieval is not an error
condition in a standard RAG system, it is an ordinary input.**

{{sec:9-practical-example}} puts a number on that. With a 30% retrieval failure
rate — unremarkable in production — **38% of everything the system says is a
confident, cited, wrong answer.**

{{cite:yan2024crag}} and {{cite:asai2023selfrag}} insert a judgement between
retrieval and generation, and {{cite:jeong2024adaptiverag}} pushes the judgement
earlier still: should retrieval happen at all? Both are the same move —
**making an implicit policy explicit** — and both are the escalation decision of
{{ch:llm-routing}} in a new setting, which is why this chapter can reuse its
machinery rather than rederive it.

{{maturity:MATURE}} Retrieval grading with a retry path is standard and cheap.
{{maturity:EMERGING}} Learned reflection tokens and query-complexity routers are
effective and the break-even analysis in {{sec:9-practical-example}} is rarely
run before adopting them.

## 3. Prerequisites

{{ch:llm-hallucination}} for {{eq:risk-coverage}} and {{eq:abstention}} — this
chapter is that material with retrieval as the confidence signal;
{{ch:llm-routing}} for {{eq:cascade-cost}} and {{eq:escalation-threshold}};
{{ch:rag-generation}} for why an ungrounded answer arrives with citations
attached; {{ch:rag-query-understanding}} for the rewrite that a corrective loop
performs; {{ch:llm-long-context}} for {{eq:u-shape}} and why irrelevant context
is not free.

## 4. Intuitive Explanation

### The missing branch

Draw the standard pipeline and look for the conditional:

```text
   query -> retrieve k chunks -> stuff into prompt -> generate -> answer
                     |
                     +-- no branch here. none. ever.
```

Every other component in a production system has a failure path. The database
call has a timeout, the payment has a decline, the parser has an exception.
Retrieval has a **top-$k$ list that is never empty**, because similarity search
always returns its $k$ nearest neighbours, however far away they are.

That is the whole problem in one sentence. **Retrieval cannot fail loudly. It can
only return the least-bad garbage available.** The system has no way to
distinguish "here are the three paragraphs that answer this" from "here are the
three least irrelevant paragraphs in a corpus that does not contain the answer",
because both are lists of three chunks.

### Adding the branch

A **grader** scores the retrieved set against the query and decides. Three
handlers are available when the grade fails, and they are not equivalent:

1. **Abstain** — say "I don't know". Cheap, safe, and it also refuses some
   questions it would have answered correctly.
2. **Retry** — rewrite the query ({{ch:rag-query-understanding}}), or search a
   different source such as the web, and try again.
3. **Degrade** — answer from parametric knowledge, flagged as unsourced.

Most treatments present these as alternatives. **They are not alternatives, they
are an ordering**, and {{sec:9-practical-example}} shows the ordering matters more
than the choice: retry first, then abstain if the retry also fails.

### Why the ordering matters — the load-bearing idea

Here is the mental model to keep, and it is worth more than the specific
techniques.

The grader is a classifier and classifiers are wrong sometimes. What does a
grader's mistake cost? **It depends entirely on what the handler does with it.**

> **A terminal handler makes the decider's mistakes permanent. A recoverable
> handler absorbs them.**

Under abstention, a false "bad" grade on a perfectly good retrieval costs the
**entire answer** — you refused a question you could have answered. Under retry,
the same mistake costs **one extra retrieval call**, and then the grader gets a
second look and can prefer the original attempt after all.

Same grader, same error rate, radically different damage. {{sec:9-practical-example}}
measures it: as the grader degrades from perfect to badly noisy, abstention loses
0.102 of accuracy and retry loses 0.047 — **less than half.**

This is why "put the cheap recoverable handler first" is not a stylistic
preference. It is a direct consequence of where the system's unreliability sits.

### Should retrieval happen at all?

The second half of the chapter moves the decision earlier. **Always-retrieve is a
policy, not a default**, and it has a cost that is invisible until measured:
retrieving for a query the model could already answer *lowers* accuracy, because
irrelevant text occupies context ({{ch:llm-long-context}}), carries the authority
of having been retrieved, and invites the model to ground on it.

So there are two fixed policies — always and never — and which wins depends on
the query mix. A router picks per query, and {{sec:9-practical-example}} finds
that the **difficulty of routing and the value of routing move in opposite
directions**, which changes what you should build.

## 5. Formal Explanation

### 5.1 The default pipeline, written out

Let $R(q)$ be the retrieved set and $r \in [0,1]$ its relevance to $q$. The
standard system computes

$$ a = G\big(q, R(q)\big) \qquad \text{for every } q, \text{ unconditionally} $$ (eq:no-handler)

Model the generator as answering correctly with probability increasing in $r$;
taking $\Prob[\text{correct} \mid r] = r$ for concreteness, the outcome split is

$$ \Prob[\text{correct}] = \mathbb{E}[r], \qquad \Prob[\text{confident wrong}] = 1 - \mathbb{E}[r] $$ (eq:harm-rate)

**Every wrong answer is harm**, not a null result: it is fluent, it is cited, and
nothing downstream marks it as ungrounded. At $\mathbb{E}[r] = 0.62$ the system
is wrong-with-citations 38% of the time.

### 5.2 The grading decision

Introduce a grader producing $\hat{r} = r + \varepsilon$, $\varepsilon \sim
\mathcal{N}(0, \sigma^2)$, and a threshold $\tau$. The policy branches on
$\hat{r} \ge \tau$. Its value against the no-handler baseline is

$$ V = \underbrace{\Prob[\hat{r} < \tau,\; r \text{ low}] \cdot \big(v_{\text{handler}} - v_{\text{gen}}\big)}_{\text{true positives: the gain}} \;-\; \underbrace{\Prob[\hat{r} < \tau,\; r \text{ high}] \cdot \big(v_{\text{gen}} - v_{\text{handler}}\big)}_{\text{false positives: the cost}} $$ (eq:grading-value)

Two things follow immediately and both are usually skipped.

**First, $\sigma$ is a parameter of the system, not of the environment.** The
grader is typically an LLM call with a rubric, and its error rate is measurable
and improvable. Papers evaluate at $\sigma \approx 0$ and report the gain term
only.

**Second, the second term is scaled by $v_{\text{gen}} - v_{\text{handler}}$ —
the cost of a false positive — and that quantity is set by which handler you
chose.** Which gives the chapter's central definition.

### 5.3 Terminal and recoverable handlers

$$ \text{terminal: } v_{\text{handler}} = v_{\text{abstain}} = 0 \quad\Longrightarrow\quad \text{false-positive cost} = v_{\text{gen}} $$ (eq:terminal-handler)

$$ \text{recoverable: } v_{\text{handler}} = \mathbb{E}\big[\max(r, r')\big] \ge v_{\text{gen}} \quad\Longrightarrow\quad \text{false-positive cost} \approx c_{\text{retrieval}} $$ (eq:recoverable-handler)

Under a recoverable handler the false-positive cost is **a retrieval call**, not
an answer, because a second attempt cannot be worse than the first if the grader
keeps the better of the two. So

$$ \frac{\partial V_{\text{terminal}}}{\partial \sigma} \;\ll\; \frac{\partial V_{\text{recoverable}}}{\partial \sigma} \;\;(\text{both negative}) $$ (eq:noise-sensitivity)

**{{eq:noise-sensitivity}} is the chapter's main structural result**, and it says
something more general than RAG: when a decision-maker is unreliable, spend your
design effort making its mistakes cheap rather than making it accurate.

### 5.4 What retry actually buys

If the second attempt is *independent* of the first — a different query
formulation, a different source — then

$$ \Prob[\text{both fail}] = p_{\text{bad}} \cdot p'_{\text{bad}} \;\ll\; p_{\text{bad}} $$ (eq:retry-independence)

which is why one retry helps so much and a third helps so little. **Independence
is the load-bearing assumption**, and it is exactly what fails when the retry is
"run the same embedding search again with $k = 20$ instead of $10$". A retry
against the same index with the same query is not a second sample; it is more of
the first one. {{cite:yan2024crag}}'s fallback to web search is a genuinely
independent attempt, and that is why it works.

The cost is a cascade, {{eq:cascade-cost}} with retrieval as the escalation:

$$ \mathbb{E}[\text{retrievals}] = 1 + \Prob[\hat{r} < \tau] $$ (eq:corrective-cost)

### 5.5 Abstention is a risk-coverage choice

Abstention does not improve the system. It **moves it along a curve** —
{{ch:llm-hallucination}}'s {{eq:risk-coverage}}, with retrieval quality as the
confidence signal:

$$ \text{coverage}(\tau) = \Prob[\hat{r} \ge \tau], \qquad \text{precision}(\tau) = \Prob[\text{correct} \mid \hat{r} \ge \tau] $$ (eq:retrieval-risk-coverage)

Raising $\tau$ raises precision and lowers coverage *and lowers accuracy*, because
some refused questions would have been answered correctly. **There is no setting
of $\tau$ at which abstention raises accuracy.** Anyone who reports one has
measured precision and called it accuracy.

### 5.6 Whether to retrieve at all

Partition queries into those the model answers from its weights ("known", share
$s$) and those needing the corpus. Four quantities:

$$ a_{pk},\; a_{pu},\; a_{rc},\; \delta $$ (eq:adaptive-quantities)

— parametric accuracy on known and unknown queries, retrieval accuracy on
corpus queries, and $\delta$, the **distraction penalty**: accuracy lost when
retrieval fires on a query that did not need it. The fixed policies are

$$ V_{\text{never}} = s\,a_{pk} + (1-s)\,a_{pu}, \qquad V_{\text{always}} = s\,(a_{pk} - \delta) + (1-s)\,a_{rc} $$ (eq:fixed-policies)

and they cross at

$$ s^{*} = \frac{a_{rc} - a_{pu}}{(a_{rc} - a_{pu}) + (\delta)} \cdot \frac{1}{1} \quad\text{solved from } V_{\text{never}} = V_{\text{always}} $$ (eq:policy-crossover)

**$\delta = 0$ makes always-retrieve unbeatable at every mix.** That is why teams
who have never measured $\delta$ believe it is a default — and $\delta > 0$ is
{{ch:llm-long-context}}'s {{eq:u-shape}} and {{cite:liu2023lost}} arriving as a
budget item.

A router with per-query accuracy $\rho$ interpolates toward the oracle
$V_{\text{oracle}} = s\,a_{pk} + (1-s)\,a_{rc}$, and it is worth having only when

$$ V_{\text{router}}(\rho) > \max\big(V_{\text{never}},\, V_{\text{always}}\big) $$ (eq:router-breakeven)

{{eq:router-breakeven}} defines a **break-even accuracy** $\rho^{*}$, and
{{sec:9-practical-example}} computes it across mixes. Its shape is the useful
result: $\rho^{*}$ is **lowest where the fixed policies are closest** and rises
toward 1 as either dominates — while the oracle headroom moves the same way.
Routing is easy exactly where it is valuable, and near-impossible exactly where
it is pointless.

## 6. Mathematical Foundation

### 6.1 The false-positive cost, worked

Take $p_{\text{bad}} = 0.30$, $\mathbb{E}[r \mid \text{good}] = 0.85$,
$\mathbb{E}[r \mid \text{bad}] = 0.15$, so
$\mathbb{E}[r] = 0.7(0.85) + 0.3(0.15) = 0.64$.

Give the grader a 10% false-positive rate (it flags a good retrieval as bad 10%
of the time). Under **abstention**, the loss is the correct answers refused:

$$ \Delta_{\text{terminal}} = 0.10 \times 0.70 \times 0.85 = 0.0595 \;\text{accuracy points} $$ (eq:fp-cost-terminal)

Under **retry**, the same false positives trigger a second retrieval whose result
the grader compares against the first. The answer is lost only if the grader also
prefers a worse second attempt — call that 30% of the time — so:

$$ \Delta_{\text{recoverable}} \approx 0.10 \times 0.70 \times 0.30 \times (0.85 - 0.50) = 0.0074 \;\text{accuracy points} $$ (eq:fp-cost-recoverable)

**A factor of eight, from the same grader.** The cost paid instead is $0.10 \times
0.70 = 7\%$ extra retrieval calls, which is a latency and money line item rather
than a correctness one.

> **MATH NOTE:** {{eq:fp-cost-recoverable}} assumes the grader's two judgements are
> independent, which flatters retry slightly — a grader that systematically
> misjudges a *particular* query will misjudge it twice. The measured gap in
> {{sec:9-practical-example}} is a factor of two rather than eight, and the
> difference between the two numbers is precisely this correlation. The ordering
> survives; the magnitude does not. This is the general hazard with independence
> assumptions in retry logic, and it applies again in {{ch:rag-agentic}}.

### 6.2 Break-even router accuracy

Model a router with accuracy $\rho$ as being right with probability $\rho$ and
inverted otherwise. Its value is

$$ V_{\text{router}}(\rho) = \rho\, V_{\text{oracle}} + (1-\rho)\, V_{\text{anti-oracle}} $$ (eq:router-interpolation)

which is linear in $\rho$, so setting it equal to the better fixed policy gives

$$ \rho^{*} = \frac{\max(V_{\text{never}}, V_{\text{always}}) - V_{\text{anti}}}{V_{\text{oracle}} - V_{\text{anti}}} $$ (eq:breakeven-solved)

Try the extremes. With $s = 0.05$ and the listing's constants,
$V_{\text{always}} = 0.787$ while $V_{\text{oracle}} = 0.793$: the numerator is
nearly the denominator, and $\rho^{*} \to 1$. **A router must be near-perfect to
match a policy that is already within six thousandths of optimal.**

With $s = 0.80$: $V_{\text{never}} = 0.712$, $V_{\text{always}} = 0.734$,
$V_{\text{oracle}} = 0.846$. The best fixed policy is 0.112 below the oracle, so
the numerator is small relative to the denominator and $\rho^{*}$ falls to about
0.54 — **barely better than a coin.**

That is {{eq:breakeven-solved}} explaining the shape rather than merely
predicting it: $\rho^{*}$ tracks *how much of the oracle's advantage the better
fixed policy has already captured.* When a fixed policy is nearly optimal,
routing must be flawless and wins nothing.

### 6.3 The order of operations, decided

Combining {{eq:retry-independence}} and {{eq:noise-sensitivity}}: retry first
because it is recoverable and because independence makes it effective, then
abstain on what survives, because abstention is the only handler that bounds
harm. Formally the composed policy answers when

$$ \max(\hat{r}, \hat{r}') \ge \tau $$ (eq:composed-policy)

which has strictly higher coverage than $\hat{r} \ge \tau$ at the same $\tau$, at
the same precision. **Free coverage** — and {{sec:9-practical-example}} confirms
it holds at every noise level tested.

## 7. Internal Mechanics

```mermaid {#fig:corrective-loop caption="The branch the standard pipeline is missing. Note the ordering: the recoverable handler (retry) runs first and absorbs the grader's false positives; the terminal handler (abstain) runs last and only on what survives two independent attempts. The retry arrow must reach a DIFFERENT source or a rewritten query, or eq:retry-independence does not hold and the loop buys nothing."}
flowchart TB
    Q["query"] --> RT{"retrieve<br/>at all?"}
    RT -->|"no: parametric"| G2["answer from weights,<br/>marked unsourced"]
    RT -->|"yes"| R["retrieve top-k"]
    R --> J{"grade the<br/>retrieved set"}
    J -->|"pass"| G["generate + cite"]
    J -->|"fail, attempt 1"| RW["rewrite query<br/>or switch source"]
    RW --> R2["retrieve again<br/>(INDEPENDENT attempt)"]
    R2 --> J2{"grade again;<br/>keep better attempt"}
    J2 -->|"pass"| G
    J2 -->|"fail"| AB["abstain"]
    G --> A["answer"]
    G2 --> A
    AB --> A
```

### 7.1 What the grader actually looks at

In rough order of cost and quality:

| Signal | Cost | What it misses |
|---|---|---|
| top-1 similarity score | free | scores are not calibrated across queries |
| score gap between rank 1 and rank $k$ | free | nothing about relevance, only about decisiveness |
| cross-encoder score ({{ch:emb-reranking}}) | one small model call | domain shift |
| LLM rubric ("does this passage answer the question?") | one LLM call | position bias, self-preference |
| generation-time self-check ({{cite:asai2023selfrag}}) | folded into generation | needs a trained model |

**Raw similarity scores are the tempting choice and the wrong one.** A cosine of
0.72 means different things for different queries, so a fixed threshold on it
grades query difficulty rather than retrieval quality. The score *gap* is better
because it is query-relative, and a cross-encoder is better still because it was
trained on the actual question.

### 7.2 What the retry must change

{{eq:retry-independence}} demands a genuinely different attempt:

- **A rewritten query** ({{ch:rag-query-understanding}}) — different embedding,
  same index. Partially independent.
- **A different retrieval mode** — sparse instead of dense
  ({{ch:emb-hybrid}}). More independent.
- **A different corpus** — web search, a second system. Most independent, and
  {{cite:yan2024crag}}'s choice.
- **The same query with a larger $k$** — *not independent at all*, and the most
  commonly implemented "retry".

### 7.3 The routing signal

{{cite:jeong2024adaptiverag}} trains a small classifier on labels induced from
which strategy actually succeeded — which is the right way to get them, since
nobody can label "did this need retrieval" reliably by inspection. Cheaper
signals that work in practice: named-entity presence, query specificity, whether
the query references corpus-specific vocabulary, and the model's own token-level
confidence on a draft answer.

## 8. Implementation

```python {tier=A name=retrieval-grading-value}
"""Grading retrieval before generating on it, and what the grader's own errors cost.

Every chapter so far has tried to make retrieval better. None has asked what the
system does when retrieval is bad anyway -- and the default answer is the worst
one available: pass the bad context to the generator, which grounds a confident
wrong answer in it and cites the irrelevant documents (ch:rag-generation).

cite:yan2024crag inserts a grader between the two stages. The grader is itself a
classifier with errors, and eq:grading-value says its value depends on those
errors as much as on the retrieval failure rate. This listing measures the whole
decision -- coverage, accuracy, HARM (confident wrong answers), answer precision,
and retrieval cost -- as the grader gets noisier.

Every grading decision here is made on the GRADER's score, never on the true
quality, including the choice between a first and a second attempt. A harness
that peeks at ground truth when picking the better attempt would report a
corrective loop that nobody can build.
"""
import numpy as np

rng = np.random.default_rng(7)

N_QUERY = 60_000
P_BAD = 0.30            # share of queries whose retrieval genuinely fails
TAU = 0.50              # grade threshold


def draw_quality(n):
    """Retrieval quality per query: a bimodal mix, which is what measured
    retrieval looks like -- mostly fine, with a hard tail that is not close."""
    bad = rng.random(n) < P_BAD
    return np.where(bad, rng.beta(1.6, 6.0, n), rng.beta(6.0, 1.6, n))


def grade(q, sigma):
    """The grader observes quality through noise. sigma = 0 is the oracle grader
    the papers implicitly assume."""
    return q if sigma == 0 else q + rng.normal(scale=sigma, size=len(q))


def report(name, q, answer, retrievals):
    """An answered query is correct with probability q. Every wrong answer is
    HARM: fluent, cited, and unmarked as ungrounded."""
    correct = (rng.random(len(q)) < q) & answer
    n_ans = answer.sum()
    print(f"{name:<27}{n_ans / N_QUERY:>10.3f}{correct.sum() / N_QUERY:>11.3f}"
          f"{(n_ans - correct.sum()) / N_QUERY:>10.3f}"
          f"{(correct.sum() / n_ans if n_ans else 0):>12.3f}"
          f"{retrievals / N_QUERY:>12.2f}")


q1 = draw_quality(N_QUERY)
q2 = draw_quality(N_QUERY)          # the second attempt, if one is made
all_yes = np.ones(N_QUERY, dtype=bool)

print(f"{N_QUERY:,} queries; {P_BAD:.0%} with genuinely failed retrieval; "
      f"mean quality {q1.mean():.3f}\n")
print(f"{'policy':<27}{'coverage':>10}{'accuracy':>11}{'harm':>10}"
      f"{'precision':>12}{'retrievals':>12}")
print("-" * 82)

report("generate always", q1, all_yes, N_QUERY)

for sigma in (0.0, 0.10, 0.20, 0.40):
    g1 = grade(q1, sigma)
    passed = g1 >= TAU

    # 1. Abstain on a failing grade. A TERMINAL handler: a false reject costs
    #    the whole answer (eq:terminal-handler).
    report(f"  abstain      s={sigma:.2f}", q1, passed, N_QUERY)

    # 2. Retry on a failing grade, then keep whichever attempt the GRADER
    #    prefers. A RECOVERABLE handler: a false reject costs one retrieval
    #    (eq:recoverable-handler).
    g2 = grade(q2, sigma)
    retried = ~passed
    keep2 = retried & (g2 > g1)
    q_best = np.where(keep2, q2, q1)
    g_best = np.where(keep2, g2, g1)
    n_retr = N_QUERY + retried.sum()
    report(f"  retry        s={sigma:.2f}", q_best, all_yes, n_retr)

    # 3. Retry, then abstain if the surviving attempt still fails the grade
    #    (eq:composed-policy).
    report(f"  retry+abstain s={sigma:.2f}", q_best, g_best >= TAU, n_retr)

print("""
Read the first row as the baseline every RAG tutorial ships. It answers
everything, so coverage is 1.000 and answer precision equals accuracy: about
38% of everything the system says is a confident, cited, wrong answer. There is
no handler, so a failed retrieval is not an error condition -- it is an ordinary
input that produces ordinary-looking output.

Now the oracle rows (s=0.00), which is the regime the papers evaluate in.
Abstention raises answer precision from 0.615 to 0.803 and cuts harm by nearly
two thirds -- and it does that by LOWERING accuracy, from 0.615 to 0.549.
Refusing to answer means refusing some questions you would have got right. That
trade is the entire content of an abstention policy, and which side of it you
want is a product decision, not a modelling one.

Retry improves both axes at once: accuracy 0.750 against 0.615, harm 0.250
against 0.385, for 32% more retrieval calls. It is not a compromise between
answering and abstaining, and it is the same grader doing the deciding. A second
retrieval attempt simply has an independent chance of succeeding where the first
failed, and only the 32% that failed the grade pay for one.

The result worth carrying is what happens as the grader degrades, because the two
handlers degrade at very different rates. From s=0.00 to s=0.40, abstention loses
0.102 of accuracy (0.549 to 0.447) and 0.057 of precision. Retry loses 0.047 of
accuracy (0.750 to 0.703) -- less than half as much.

The reason is structural. Under abstention, a false reject costs the ENTIRE
answer: the grader's mistake is terminal. Under retry, a false reject costs one
extra retrieval, and the grader then gets a second chance to notice it preferred
the wrong attempt. Same grader, same error rate, one-quarter of the damage.

State it as a design rule, because it generalises well past retrieval: WHEN THE
DECIDER IS UNRELIABLE, MAKE ITS MISTAKES RECOVERABLE. A terminal handler inherits
the decider's error rate; a recoverable handler absorbs it.

Finally, compare the third row of each block against the first. At every noise
level, retry-then-abstain dominates abstain-alone on coverage and accuracy at
essentially identical precision -- at s=0.40 it is 0.842 coverage and 0.628
accuracy against 0.599 and 0.447, with precision 0.745 against 0.746. Retrying
BEFORE abstaining buys back most of the coverage abstention gives up and costs
nothing in precision to do it. If you implement one thing from this chapter, it
is that ordering.""")
```

The second listing moves the decision earlier: not *what to do when retrieval
failed*, but *whether to retrieve*.

```python {tier=A name=adaptive-retrieval-routing}
"""Should retrieval happen at all? The break-even accuracy of a router.

ch:rag-why treated retrieval as the architecture. cite:jeong2024adaptiverag treats
it as a DECISION -- always-retrieve is a policy, and like any policy it can be
wrong. Retrieval costs latency and context budget, and on a query the model
already answers well it costs accuracy too, because irrelevant retrieved text
displaces attention (ch:llm-long-context) and invites the model to ground on it.

A router that predicts the query type recovers the best of both fixed policies,
minus its own error rate. eq:router-breakeven says a router must clear a
computable accuracy before it beats simply picking the better fixed policy, and
that threshold depends on the query MIX. This listing computes it.
"""
import numpy as np

rng = np.random.default_rng(31)

N = 120_000

A_PARAM_KNOWN = 0.86    # accuracy answering from weights, on what the model knows
A_PARAM_UNKNOWN = 0.12  # accuracy answering from weights, on corpus-specific facts
A_RETR_CORPUS = 0.79    # accuracy with retrieval, when the corpus has the answer
DISTRACTION = 0.14      # accuracy lost when irrelevant context is retrieved anyway


def evaluate(known, route_retrieve):
    """known: query is answerable from the model's own weights.
    route_retrieve: whether this query is sent to the retriever."""
    a = np.where(
        route_retrieve,
        np.where(known, A_PARAM_KNOWN - DISTRACTION, A_RETR_CORPUS),
        np.where(known, A_PARAM_KNOWN, A_PARAM_UNKNOWN),
    )
    return a.mean(), route_retrieve.mean()


def router(known, accuracy):
    """A classifier that predicts 'needs retrieval' (i.e. not known) with the
    given per-query accuracy. At 0.5 it is a coin; at 1.0 it is the oracle."""
    correct = rng.random(len(known)) < accuracy
    return np.where(correct, ~known, known)


print(f"{N:,} queries. Answering from weights: {A_PARAM_KNOWN:.2f} on what the "
      f"model knows,\n{A_PARAM_UNKNOWN:.2f} on corpus-specific facts. With "
      f"retrieval: {A_RETR_CORPUS:.2f} on corpus facts,\nand a {DISTRACTION:.2f} "
      f"distraction penalty when retrieval fires on a query it was not needed for.\n")

print(f"{'known share':>12}{'never':>9}{'always':>9}{'oracle':>9}"
      f"{'r=0.70':>9}{'r=0.85':>9}{'r=0.95':>9}{'break-even r':>15}")
print("-" * 81)

for share_known in (0.05, 0.20, 0.40, 0.60, 0.80, 0.90):
    known = rng.random(N) < share_known
    never, _ = evaluate(known, np.zeros(N, dtype=bool))
    always, _ = evaluate(known, np.ones(N, dtype=bool))
    oracle, _ = evaluate(known, ~known)
    best_fixed = max(never, always)

    routed = {r: evaluate(known, router(known, r))[0] for r in (0.70, 0.85, 0.95)}

    # The router accuracy at which routing first beats the better fixed policy
    # (eq:breakeven-solved), found by bisection rather than by the formula, so
    # the two can be checked against each other.
    lo, hi = 0.50, 1.00
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if evaluate(known, router(known, mid))[0] >= best_fixed:
            hi = mid
        else:
            lo = mid
    breakeven = hi

    print(f"{share_known:>12.2f}{never:>9.3f}{always:>9.3f}{oracle:>9.3f}"
          f"{routed[0.70]:>9.3f}{routed[0.85]:>9.3f}{routed[0.95]:>9.3f}"
          f"{breakeven:>15.3f}")

print("""
Start with the two fixed policies. At a 5% known share always-retrieve wins by a
mile, 0.787 against 0.156. By a 90% known share the ordering has REVERSED --
never-retrieve scores 0.786 against always-retrieve's 0.727 -- because on a query
the model already answers, firing the retriever spends latency and budget to lose
accuracy. Neither policy is a default. Both are bets on a mix, and the bet flips
at a known share of about 0.83 for these numbers.

The distraction penalty is what makes always-retrieve loseable at all, and it is
the term most pipelines assume is zero. It is not: retrieved text that does not
answer the question still occupies the context window, still carries the
authority of having been retrieved, and still invites the model to ground on it
(ch:llm-hallucination). Set DISTRACTION to 0 and always-retrieve wins at every
mix -- which is precisely why teams who have never measured it believe it does.

Now read the break-even column together with the oracle column, because the two
tell one story and it is not the story routing is usually sold with.

Where one fixed policy dominates, routing is BOTH hard and pointless. At a 5%
known share the router must be right 98.9% of the time merely to match
always-retrieve -- and a PERFECT router would score 0.793 against
always-retrieve's 0.787, so the entire prize is six thousandths of accuracy. You
would be building a classifier that must be near-flawless in order to win almost
nothing.

Where the fixed policies are close, routing is both easy and valuable. At an 80%
known share the two are nearly tied (0.734 and 0.712), break-even falls to 0.542
-- barely better than a coin -- and the oracle scores 0.846, so perfect routing is
worth 0.112 of accuracy, nearly twenty times the prize at the other end.

The difficulty of routing and the value of routing move in OPPOSITE directions,
and both are determined by one measurable quantity: the query mix. So the
adaptive-RAG decision does not start with a classifier. It starts with an
afternoon spent labelling two hundred real queries for whether the corpus was
needed. If that sample comes back 95% corpus-dependent, the correct adaptive
policy is to always retrieve, and a router is a component that can only cost
you.""")
```

## 9. Practical Example

**The cost of having no handler.** The baseline row is the pipeline in every RAG
tutorial: coverage 1.000, accuracy 0.615, and **harm 0.385**. Answer precision
equals accuracy because the system never declines, so **38% of everything it says
is confident, cited, and wrong.** That is not a tail risk; it is the modal
outcome for the 30% of queries whose retrieval failed, multiplied out.

**Abstention buys precision with accuracy.** With a perfect grader, precision
rises 0.615 → **0.803** and harm falls 0.385 → 0.135. Accuracy *falls*, 0.615 →
0.549, because refusing to answer refuses some questions you would have got
right. {{eq:retrieval-risk-coverage}}, and there is no threshold that escapes it.

**Retry improves both axes at once.** Accuracy 0.615 → **0.750**, harm 0.385 →
0.250, for **32% more retrieval calls** ({{eq:corrective-cost}}). This is not a
compromise between answering and abstaining; it is the same grader used
differently, exploiting {{eq:retry-independence}}.

**The main result is the degradation rates.** From $\sigma = 0$ to $\sigma =
0.40$:

| Handler | accuracy at $\sigma=0$ | at $\sigma=0.40$ | lost |
|---|---|---|---|
| abstain (terminal) | 0.549 | 0.447 | **0.102** |
| retry (recoverable) | 0.750 | 0.703 | **0.047** |

**The same grader, with the same error rate, does less than half the damage when
its mistakes are recoverable.** {{eq:noise-sensitivity}} measured. The mechanism
is exactly {{eq:terminal-handler}} against {{eq:recoverable-handler}}: a false
reject costs an answer under abstention and one retrieval call under retry.

> **IMPORTANT:** {{eq:fp-cost-recoverable}} predicted a factor of eight and the
> measurement gives a factor of two. The gap is the independence assumption in
> {{sec:6-mathematical-foundation}}'s MATH NOTE — a grader that misjudges a
> particular query tends to misjudge it twice, so the two attempts are not
> independent draws. **The ordering survives and the magnitude does not**, which
> is the right way to hold every retry argument in this book, including
> {{ch:rag-agentic}}'s.

**And the composed policy is close to free.** At every noise level,
retry-then-abstain dominates abstain-alone on coverage and accuracy at
essentially the same precision. At $\sigma = 0.40$: coverage 0.842 against 0.599,
accuracy 0.628 against 0.447, precision 0.745 against 0.746. **Retrying before
abstaining recovers most of the coverage abstention gives up and pays nothing in
precision for it.** {{eq:composed-policy}}. If one thing from this chapter reaches
production, it should be that ordering.

**Whether to retrieve at all.** The second listing's fixed policies swap places:
always-retrieve wins 0.787 to 0.156 at a 5% known share, and *loses* 0.727 to
0.786 at 90%. The crossover sits near $s = 0.83$ for these constants. **The
distraction penalty is what makes always-retrieve loseable**, and setting it to
zero makes always-retrieve unbeatable at every mix — which is precisely why it
looks like a default to anyone who has not measured it.

**The break-even column is the finding.** At a 5% known share a router must be
right **98.9%** of the time to match always-retrieve, and a *perfect* router
would gain 0.006 accuracy. At an 80% known share break-even falls to **0.542** —
barely better than a coin — and a perfect router gains **0.112**, nearly twenty
times as much.

**Routing is hardest exactly where it is worthless, and easiest exactly where it
is valuable.** {{eq:breakeven-solved}} explains why: $\rho^{*}$ tracks how much
of the oracle's advantage the better fixed policy has already captured. So the
adaptive-RAG decision does not begin with a classifier. It begins with labelling
two hundred real queries for whether the corpus was needed — and if that sample
comes back 95% corpus-dependent, the correct adaptive policy is `always
retrieve`, and a router can only cost you.

## 10. Production Considerations

**Add the branch before tuning anything else.** A system with no handler for
failed retrieval is not a system with a retrieval-quality problem; it is a system
with a missing error path, and no amount of chunking work substitutes.

**Order the handlers: retry, then abstain.** {{eq:composed-policy}} is free
coverage at equal precision.

**Make the retry independent** ({{eq:retry-independence}}). A larger $k$ against
the same index with the same query is not a retry. A different source is.

**Measure grader error rate, not just grader presence.** $\sigma$ is a system
parameter. Label two hundred (query, retrieved-set) pairs and compute the
confusion matrix.

**Measure the distraction penalty $\delta$.** Run your evaluation set with
retrieval forced on and forced off, and compare on the subset the model answers
correctly without retrieval. Almost nobody has this number, and
{{eq:fixed-policies}} cannot be evaluated without it.

**Label the query mix before building a router.** Two hundred queries, one
afternoon, and it decides whether the router is worth writing at all.

**Log the handler that fired, per query.** Grade-pass rate, retry rate, abstain
rate, and the retry's success rate are the four numbers that tell you whether the
loop is working or thrashing.

**Cap the loop at one retry** unless you have measured that a second is
independent of the first. {{eq:retry-independence}} decays fast.

**Budget for the tail.** {{eq:corrective-cost}} makes p99 latency roughly double
p50 for the queries that retry. Set the timeout on the whole loop, not on each
retrieval.

## 11. Common Mistakes

**Shipping a pipeline with no failure path at all** — the default, and the one
this chapter exists for.

**Grading on raw similarity scores.** They are not calibrated across queries, so
a fixed threshold grades query difficulty.

**Abstaining first.** Terminal handler, first in line, maximum exposure to grader
error.

**A "retry" that re-runs the same query** against the same index.
{{eq:retry-independence}} does not hold and the loop buys nothing but latency.

**Reporting precision as accuracy** after adding abstention. The number went up
because the denominator went down.

**Assuming the distraction penalty is zero**, which makes always-retrieve look
free.

**Building a router before measuring the mix**, then finding break-even was 0.99.

**Unbounded corrective loops.** Retry until the grade passes is an unbounded
cost, and on a query whose answer is simply absent it never terminates —
{{ch:rag-agentic}}'s termination problem, arriving early.

## 12. Failure Modes

**Grader collapse.** The grader passes everything (or nothing) because the
threshold was tuned on a different retrieval distribution. Symptom: retry rate at
0% or 100%. Detect by alerting on the pass-rate directly.

**Retry thrashing.** The rewrite produces a query similar to the original, so the
second attempt fails identically. Symptom: high retry rate, near-zero retry
success rate. Detect by logging both.

**Abstention creep.** Threshold set for safety, coverage quietly falls as the
corpus drifts, and users stop asking because the system stopped answering.
Symptom: coverage trending down while grade distribution shifts.

**Distraction on known queries.** The system got *worse* after RAG was added, on
exactly the questions it used to answer. Symptom: regression concentrated on
general-knowledge queries. Diagnose by forcing retrieval off for that subset.

**Router miscalibration after a model upgrade.** The new model knows more, so the
known share moved, so the router's training labels are stale.
{{eq:policy-crossover}} moved and nobody recomputed it.

**Correct abstention read as a bug.** The system correctly declines a question
whose answer is genuinely absent, a user reports it, and someone lowers the
threshold. **Abstentions must be reviewed against whether the answer existed**,
or the feedback loop destroys the handler.

## 13. Alternatives

| Alternative | What it trades | When it wins |
|---|---|---|
| Self-RAG's reflection tokens ({{cite:asai2023selfrag}}) | needs a fine-tuned model | when you control training; folds grading into generation at no extra call |
| FLARE's forward-looking retrieval ({{cite:jiang2023flare}}) | retrieves mid-generation, more calls | long generations where needs appear as writing proceeds |
| Reranker score as the grade ({{ch:emb-reranking}}) | less semantic than an LLM rubric | already have the reranker; near-free |
| Post-hoc verification ({{ch:rag-generation}}) | catches it after generating, wasting the call | when abstention is unacceptable and correction is cheap |
| Query-complexity routing ({{cite:jeong2024adaptiverag}}) | a classifier to train and maintain | balanced query mixes only ({{eq:router-breakeven}}) |
| Just widen $k$ | more context, {{eq:u-shape}} exposure | when the grader would be more expensive than the tokens |

**The last row is a real option and worth stating plainly.** If a grading call
costs more than doubling $k$, and {{eq:marginal-chunk-value}} says the extra
chunks are not harmful, then widening $k$ is the cheaper correction. Grading pays
when context is expensive or when harm is expensive — and harm usually is.

## 14. Evaluation

**Report coverage, accuracy, and precision separately.** They move in different
directions and any one alone is misleading — this is the single most common
reporting error in corrective-RAG results.

**Report harm, not just error.** A confident cited wrong answer and a "don't
know" are both non-answers and they are not equally bad.

**Evaluate the grader as a classifier**: precision, recall, and the confusion
matrix against hand-labelled retrieval quality. The end-to-end number confounds
it with everything else.

**Evaluate the retry's independence** by measuring the second attempt's success
rate *on queries where the first failed*. If it matches the base rate, the retry
is independent; if it is far below, it is not.

**Measure the distraction penalty** as its own experiment.

**Use reference-free groundedness ({{cite:es2023ragas}}) as a monitor, not an
evaluation.** It catches regressions; it does not establish quality, and its
judge inherits the caveats {{part:25}} details.

## 15. Advanced Concepts

**Learned reflection.** {{maturity:EMERGING}} {{cite:asai2023selfrag}} trains the
model to emit its own retrieve/critique tokens, folding
{{eq:grading-value}}'s branch into decoding. This removes the extra call and couples the
grader to the generator — which is efficient, and which means a generator that is
confidently wrong is also confidently *graded* wrong.

**Retrieval during generation.** {{maturity:EMERGING}}
{{cite:jiang2023flare}} triggers retrieval when the next sentence is predicted
with low confidence, making the decision continuous rather than once-per-query.
The natural generalisation of this chapter's branch, and it is where the
corrective loop starts becoming {{ch:rag-agentic}}'s loop.

**The recoverability principle beyond RAG.** {{eq:noise-sensitivity}} says an
unreliable decider paired with a cheap-to-undo action is a good system, and paired
with an irreversible action is a bad one. That is the same argument behind
optimistic concurrency, speculative decoding, and — in {{part:17}} — why an agent
should be allowed to retry a read and never a write.

**Thresholds as a product surface.** {{maturity:MATURE}} $\tau$ is not a tuning
constant. It is where a business decides how it wants to be wrong, and different
users of one system may want different points on
{{eq:retrieval-risk-coverage}}'s curve — a clinician and a marketer should not
share a threshold.

**The distraction penalty is a model property that is improving.**
{{maturity:EMERGING}} As models get better at ignoring irrelevant context,
$\delta$ falls, {{eq:policy-crossover}} moves, and the case for routing weakens.
This is a conclusion with an expiry date, and the right response is to re-measure
$\delta$ on every model upgrade rather than to re-read the literature.

## 16. Connection to Previous Chapters

{{ch:llm-hallucination}}'s {{eq:risk-coverage}} is {{eq:retrieval-risk-coverage}}
with retrieval quality as the confidence signal, and its abstention material is
this chapter's terminal handler. {{ch:llm-routing}}'s {{eq:cascade-cost}} and
escalation threshold are {{eq:corrective-cost}} and {{eq:router-breakeven}} —
the same decision with a retriever instead of a larger model.
{{ch:rag-generation}} explains why an ungrounded answer arrives *with citations*,
which is what makes {{eq:harm-rate}}'s harm harmful.
{{ch:rag-query-understanding}}'s rewrite is the retry's mechanism, and
{{ch:emb-hybrid}}'s second retrieval mode is how to make it independent.
{{ch:llm-long-context}}'s {{eq:u-shape}} is the distraction penalty's origin.
{{ch:rag-graph}}'s local/global router is the same shape as
{{eq:router-breakeven}}, and {{ch:rag-agentic}} is what this chapter's loop
becomes when the number of iterations is not fixed at one.

## 17. Exercises

1. Derive {{eq:breakeven-solved}} from {{eq:router-interpolation}}, and state
   what $V_{\text{anti}}$ is.
2. Using {{eq:fixed-policies}}, compute the crossover $s^{*}$ for the listing's
   constants and check it against the measured table.
3. In `retrieval-grading-value`, make the two grader observations *correlated*
   (share a per-query bias term). How much of the retry advantage survives?
4. Sweep `TAU` in the same listing and plot coverage against precision. Where is
   the knee, and what would move it?
5. Add a third attempt. Quantify the marginal gain and relate it to
   {{eq:retry-independence}}.
6. In `adaptive-retrieval-routing`, set `DISTRACTION = 0`. Which column changes
   most, and what does that say about pipelines that never measured it?
7. Make the router's accuracy depend on query type (better at spotting corpus
   queries than parametric ones). Does break-even rise or fall?
8. Design the distraction-penalty experiment for a real system. What is the
   control, and which subset do you measure on?

## 18. Interview Questions

1. What does a standard RAG pipeline do when retrieval fails?
2. Why is a wrong answer from bad retrieval worse than no answer?
3. What is a retrieval grader and what does it cost?
4. Why should retry come before abstention?
5. Define a terminal and a recoverable handler, and say why the distinction
   matters for an unreliable decider.
6. What makes a retry independent, and why does independence matter?
7. Your abstention rate is 40% and users are unhappy. What do you measure?
8. What is the distraction penalty and how would you measure it?
9. When is a query-complexity router not worth building?
10. Your RAG system got worse on general-knowledge questions after launch.
    Diagnose.

## 19. Research Questions

1. {{eq:retry-independence}} assumes independent attempts. Can retrieval
   independence be *measured* cheaply, so a system knows whether its retry is
   worth making?
2. Grader errors are correlated across attempts on the same query. Is there a
   rewrite strategy that decorrelates them deliberately?
3. {{eq:breakeven-solved}} treats router accuracy as uniform. What is the right
   analysis when the router is accurate on easy queries and poor on the hard ones
   that matter?
4. The distraction penalty falls as models improve. Is there a scaling
   relationship, and does it predict when routing stops paying?
5. Can the grade, the retry decision, and the abstention decision be trained
   jointly against a downstream harm-weighted objective, rather than tuned as
   three thresholds?

## 20. Chapter Summary

**A standard RAG pipeline has no error condition.** Similarity search always
returns $k$ chunks, so a failed retrieval is indistinguishable from a successful
one, and {{eq:no-handler}} generates on it either way. At a 30% failure rate,
**38% of everything the system says is confident, cited, and wrong.**

**Grading adds the missing branch, and the grader's own error rate is a
first-class term** ({{eq:grading-value}}) that papers evaluate at zero. Its cost
is scaled by what the handler does with a false positive — which makes the choice
of handler, not the accuracy of the grader, the dominant design decision.

**Terminal handlers inherit the decider's errors; recoverable handlers absorb
them.** Abstention is terminal: a false reject costs the whole answer. Retry is
recoverable: a false reject costs one retrieval. Measured over the same noise
sweep, abstention loses **0.102** of accuracy and retry **0.047** — less than
half. {{eq:noise-sensitivity}}, and the design rule generalises well past
retrieval: **when the decider is unreliable, make its mistakes cheap rather than
making it accurate.**

**So the ordering is retry, then abstain**, and the composition is nearly free —
at every noise level it beat abstention alone on coverage and accuracy at
identical precision. Retry works because a *different* attempt has an independent
chance ({{eq:retry-independence}}), which is exactly why re-running the same query
with a larger $k$ is not a retry.

**Abstention never raises accuracy.** It moves the system along
{{eq:retrieval-risk-coverage}}'s curve, buying precision with coverage. Reporting
the precision gain as an accuracy gain is the field's most common way of
overstating these results.

**And retrieval itself is a decision.** {{eq:fixed-policies}} shows
always-retrieve winning 0.787 to 0.156 at one query mix and *losing* 0.727 to
0.786 at another, with the distraction penalty $\delta$ as the term that makes
losing possible — and $\delta$ is the number almost nobody measures.

**The break-even analysis is the practical result.** A router needs **98.9%**
accuracy to be worth anything when one fixed policy dominates, and **54.2%** when
they are close — while the prize moves the same way, 0.006 against 0.112.
**Routing is hardest exactly where it is worthless.** Which means the first
artefact of an adaptive-RAG project is not a classifier. It is two hundred
labelled queries.

## 21. Further Reading

{{cite:yan2024crag}} for the corrective loop — note that its fallback to web
search is what makes the retry *independent*, which is the part most
reimplementations drop.
{{cite:asai2023selfrag}} for folding the grade into generation, and for the
trade that comes with coupling grader and generator.
{{cite:jeong2024adaptiverag}} for query-complexity routing, read alongside
{{eq:router-breakeven}} — the paper's gains are real and they are gains on a
particular mix.
{{cite:jiang2023flare}} for making the decision continuous, and as the bridge
into {{ch:rag-agentic}}.
{{cite:liu2023lost}} for the distraction penalty's mechanism.
{{cite:es2023ragas}} for the reference-free metrics that make grading measurable
in production, with {{part:25}}'s judge caveats attached.
{{cite:gao2023ragsurvey}} places corrective and adaptive RAG in the standard
taxonomy, under "modular RAG".

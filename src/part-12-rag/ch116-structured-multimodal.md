---
id: rag-structured
number: 116
part: XII
tier: full
status: draft
requires: [rag-ingestion, rag-chunking, rag-indexing, rag-advanced-retrieval,
           rag-agentic, emb-what-they-are, llm-function-calling,
           llm-structured-output]
provides: [structured-retrieval, query-generation-as-retrieval, schema-retrieval,
           aggregate-unreachability, table-representation, execution-observability,
           visual-document-retrieval]
citations: [yu2018spider, li2023bird, lei2025spider2, faysse2025colpali,
            khattab2020colbert, liu2023lost, gao2023ragsurvey]
---

## 1. Learning Objectives

By the end of this chapter you will be able to prove that a top-$k$ retriever
cannot answer an aggregate query over a table and cannot evaluate a numeric
predicate — and to say which property of embeddings is responsible for each;
recognise that retrieval over structured data means **generating a query**, so
the retrieval problem relocates to the schema rather than disappearing; show that
schema retrieval has an interior optimum because both of its failure modes are
total, and use that to explain the published text-to-SQL benchmark gap; choose a
representation for tables embedded in text corpora; and state what execution
gives you that text retrieval has to pay for.

## 2. Why This Matters

Everything in this part has assumed the knowledge is **text you find by
similarity**. Much of the knowledge an organisation actually has is not: it is
rows in a warehouse, tables inside PDFs, and pages that are pictures.

The reflex is to make it text and embed it. Serialise the row, caption the image,
flatten the table into prose. That reflex is usually wrong, and this chapter is
about why, and what to do instead.

The strongest case is exact rather than empirical. **A top-$k$ retriever cannot
compute a SUM.** An aggregate is a function of *every* qualifying row, so no
selection of $k$ of them contains the answer — the same argument as
{{ch:rag-graph}}'s global questions, except that here it is not an approximation
problem with a stratified-sampling escape. Pre-computed summaries do not save you
because you cannot know in advance which of combinatorially many aggregates will
be asked. {{sec:9-practical-example}} measures a retrieval system converging on
the right total only when it has retrieved *half the table*, which is not a
retrieval system converging — it is a full table scan at LLM prices.

So the question to ask about any knowledge source is not "how do I embed this?"
It is: **does this data already have a query language?** If it does, retrieval
means writing a query, and the vector index gets a different job — finding the
*schema*, not the data. That relocation is the chapter's spine, and
{{sec:9-practical-example}} shows the relocated problem is where text-to-SQL
actually fails: with model quality held constant, growing the schema from a
tutorial database to a warehouse drops end-to-end accuracy by a factor of **ten**.

{{maturity:MATURE}} Text-to-SQL on small schemas. {{maturity:EMERGING}} Schema
retrieval at warehouse scale, and visual document retrieval
({{cite:faysse2025colpali}}), which is the most interesting recent argument in
this part: if the parser is what destroys the tables, one option is to delete the
parser.

## 3. Prerequisites

{{ch:rag-ingestion}} for what parsing does to tables and
{{eq:table-recoverability}}; {{ch:rag-indexing}} for
{{eq:no-order-in-embedding}}, which is half of this chapter's negative results;
{{ch:rag-chunking}}'s {{eq:chunk-dilution}} and
{{ch:rag-advanced-retrieval}}'s {{eq:contextual-augmentation}} for table
representation; {{ch:llm-function-calling}} for
{{eq:max-distractor}} and the tool loop; {{ch:llm-structured-output}} for
generating a query that parses; {{ch:rag-agentic}} for the loop that a failed
query re-enters.

## 4. Intuitive Explanation

### Three kinds of non-text, and only one question

| Source | Has a query language? | So retrieval means |
|---|---|---|
| a warehouse table | **yes** — SQL | generating SQL; retrieving the *schema* |
| a table inside a PDF | no | choosing a representation |
| a page that is an image | no | choosing what to embed |

**The first row is the one people get wrong**, and expensively. Faced with a
database, the instinct is to serialise the rows into sentences and put them in
the vector index, because that is the machinery already built. It produces a
system that can look up a customer by name and cannot answer any question a
database is for.

### Why embedding rows fails, precisely

Two distinct failures, and it is worth keeping them apart because they have
different fixes.

**Numeric predicates.** "Accounts billing more than a million" is a comparison,
and {{ch:rag-indexing}} established that embeddings do not represent magnitude
order. To an encoder, `1,240,000` and `980,000` are two unrelated token
sequences; nothing in the geometry says one is larger. So similarity retrieval
can find *the region* and then returns rows essentially at random with respect to
the predicate. Measured: recall **0.041** against SQL's 1.000.

**Aggregates.** "Total revenue in EMEA" is a function of every qualifying row.
This is not a hard retrieval problem — **it is not a retrieval problem.** No
top-$k$ contains a SUM, and unlike {{ch:rag-graph}}'s global questions there is no
pre-computed-summary escape, because the space of aggregates anyone might ask is
combinatorial.

**The right answer is to stop indexing the data and start querying it.**

### Where the problem goes instead

Generating SQL requires knowing the schema, and a warehouse schema does not fit
in a context window. So something must choose which tables and columns to show
the model — and *that* is a text retrieval problem, over column names and
descriptions.

**The retrieval problem did not disappear. It moved from the rows to the
schema**, and it got harder in a specific way: **both failure modes are total.**
A column that is not retrieved cannot appear in the query, so the query is
unwritable no matter how good the model is. A column that is retrieved and
irrelevant is a distractor, and {{ch:llm-function-calling}} showed selection
accuracy falling as distractors accumulate. Recall and precision both matter
absolutely, so the product has a sharp interior optimum.

This is why text-to-SQL looks solved on a tutorial database and does not survive
contact with a warehouse. The benchmarks say so directly: the same class of
system that scores in the nineties on small cross-domain schemas
({{cite:yu2018spider}}) scores far lower on large realistic ones
({{cite:li2023bird}}) and lower again on enterprise workflows
({{cite:lei2025spider2}}). **Read that progression as a measurement of schema
retrieval, not of SQL generation.**

### The compensation: execution

Structured retrieval gets one thing free that cost {{ch:rag-corrective}} an
entire chapter. **A generated query can be run.** Syntax errors, missing columns,
type errors, and empty results are all observable without a grader — which is
exactly the step observability {{ch:rag-agentic}} identified as the load-bearing
component of a loop.

The catch is that observability is *partial*. A query that runs and returns the
wrong rows is silently wrong, and silent wrongness in SQL is worse than in text,
because a table of numbers reads as authoritative.

## 5. Formal Explanation

### 5.1 Aggregates are unreachable

Let the table be rows $\mathcal{R} = \{r_1, \dots, r_N\}$ and a query ask for
$A = \bigoplus_{r \in \mathcal{R} : \phi(r)} g(r)$ for a predicate $\phi$ and an
associative operator $\oplus$. A retriever returns $S \subseteq \mathcal{R}$,
$|S| = k$. Then

$$ \hat{A}(S) = \bigoplus_{r \in S : \phi(r)} g(r) = A \iff \{r : \phi(r)\} \subseteq S $$ (eq:aggregate-unreachable)

so the retrieval is correct **only when it has retrieved the entire qualifying
set**. For $|\{r : \phi(r)\}| \gg k$ the answer is wrong by construction, and it
is wrong in a *systematic* direction: a SUM computed from a subset is an
underestimate, every time.

Compare {{ch:rag-graph}}'s {{eq:global-aggregate}}. There, community summaries
gave a stratified estimator that worked; here they cannot, because a summary
would have to pre-compute the specific aggregate, and

$$ |\{\text{possible aggregates}\}| = |\text{columns}| \times |\text{operators}| \times |\text{group-bys}| \times |\text{predicates}| $$ (eq:aggregate-combinatorics)

is combinatorial. **Pre-computation is exactly what a database's query engine
declines to do, for the same reason.**

### 5.2 Numeric predicates are unrepresentable

{{ch:rag-indexing}}'s {{eq:no-order-in-embedding}} says an embedding space has no
canonical order encoding a numeric field. Concretely, for a serialised row
containing a value $v$,

$$ \text{sim}\big(q_{>\theta},\, \text{emb}(r_v)\big) \;\not\propto\; \mathbb{1}[v > \theta] $$ (eq:predicate-unrepresentable)

so retrieval for a threshold query returns qualifying rows at their base rate
within whatever *categorical* signal the query does carry:

$$ \text{precision} \approx \Prob\big[\phi(r) \mid r \text{ matches the categorical part}\big] $$ (eq:predicate-base-rate)

{{eq:predicate-base-rate}} is a strong claim and {{sec:9-practical-example}}
tests it: measured precision 0.153 against a base rate of 0.20 — the retriever
does essentially nothing on the predicate.

### 5.3 Retrieval as query generation

$$ q \;\xrightarrow{\;\text{LLM}\;}\; \text{SQL} \;\xrightarrow{\;\text{engine}\;}\; \text{rows} \;\xrightarrow{\;\text{LLM}\;}\; \text{answer} $$ (eq:query-generation-pipeline)

The retrieval step is the first arrow, and it succeeds only if the schema the
model was shown contains what the query needs:

$$ \Prob[\text{correct answer}] = \underbrace{\Prob[\text{needed columns} \subseteq \text{shown}]}_{\text{schema recall}} \times \underbrace{\Prob[\text{correct SQL} \mid \text{shown}]}_{\text{generation}} $$ (eq:text-to-sql-factored)

**The first factor is a hard ceiling.** No improvement in the second recovers a
column that was never shown:

$$ \Prob[\text{needed} \not\subseteq \text{shown}] \;\Longrightarrow\; \Prob[\text{correct}] = 0, \quad \text{regardless of model} $$ (eq:schema-recall-hard-ceiling)

### 5.4 The schema-retrieval optimum

Showing $k$ columns of which $n$ are needed leaves $k - n$ distractors, and
{{eq:max-distractor}} gives a decay in the generation factor. Modelling it as
exponential with half-life $H$:

$$ \Prob[\text{correct}] = R(k) \cdot p_0 \cdot 2^{-(k-n)/H} $$ (eq:schema-budget-optimum)

with $R(k)$ increasing and the second factor decreasing. **Both are total
failures** — an unwritable query and a wrong query score the same — so the
product peaks strictly inside the range, and the peak falls as the schema grows,
because $R(k)$ shifts right while the penalty does not move.

$$ \frac{\partial k^{*}}{\partial |\text{schema}|} > 0, \qquad \frac{\partial\, \Prob[\text{correct}](k^{*})}{\partial |\text{schema}|} < 0 $$ (eq:schema-scaling)

{{eq:schema-scaling}} is the mechanism behind the benchmark progression, and
{{sec:9-practical-example}} exhibits it: the optimum budget rises from 25 to 100
columns while the achievable score falls from 0.821 to 0.084 — **with the model
held fixed.**

### 5.5 Execution observability

$$ o_{\text{syntax}} \approx 1, \qquad o_{\text{empty}} \approx 1, \qquad o_{\text{wrong rows}} \approx 0 $$ (eq:execution-observability)

Substituting into {{ch:rag-agentic}}'s {{eq:loop-degenerates}}: an agentic loop
over SQL has *high* observability for the failures the engine reports and *zero*
for semantic errors. So the loop self-corrects syntax quickly and converges
confidently on the wrong query, which is the characteristic failure of SQL agents
and the reason execution success is not a quality metric.

### 5.6 Tables inside text

A table in a document has no query engine, so the choice is a representation.
Writing $H$ for the header and $r_i$ for a row:

$$ \text{idx}(r_i) = H \Vert r_i \quad\text{(row-wise, header-augmented)}, \qquad \text{idx}(T) = \text{serialise}(T) \quad\text{(whole-table)} $$ (eq:table-representations)

The first is exactly {{ch:rag-advanced-retrieval}}'s
{{eq:contextual-augmentation}}: **a table row without its header is the purest
orphan chunk in the book** — `EMEA | 1,240,000 | 18%` carries no topic at all.
The second hits {{eq:chunk-dilution}}, since a fifty-row table averages to a
vector describing nothing in particular.

The resolution follows the same rule as {{ch:rag-advanced-retrieval}}: **index
the row with its header, send the table.** Retrieval unit small and specific,
generation unit complete.

## 6. Mathematical Foundation

### 6.1 The aggregate error, worked

Take a region with $M = 750$ rows and revenue $\bar{v}$ per row, so $A = 750
\bar{v}$. Retrieve $k$ rows of which $m$ qualify. The naive estimate — the model
adds up what it sees — is

$$ \hat{A} = m \bar{v}, \qquad \frac{|\hat{A} - A|}{A} = 1 - \frac{m}{M} $$ (eq:aggregate-error)

**The relative error is one minus the fraction of the group retrieved.** At $k =
40$, $m \le 40$, so the error is at least $1 - 40/750 = 0.947$ — and
{{sec:9-practical-example}} measures 0.958. Not a bound that improves with a
better retriever; a bound that improves only by retrieving more rows.

To get within 10% you must retrieve 90% of the group. **At that point you have
scanned the table with a language model in the loop**, which costs perhaps five
orders of magnitude more than `SELECT SUM(revenue) WHERE region='EMEA'`.

> **MATH NOTE:** A statistician would object that $\hat{A} = m\bar{v}$ is the
> wrong estimator — scaling by $M/k$ gives an unbiased one. True, and it does not
> rescue the architecture. The scaled estimator requires knowing $M$, which is a
> `COUNT`, which is another aggregate. And it is unbiased with a variance set by
> the sample size, so a confident-sounding total would carry an error bar nobody
> displays. The deeper point stands: {{eq:aggregate-unreachable}} says exactness
> requires the whole qualifying set, and everything short of that is estimation
> dressed as retrieval.

### 6.2 Where the schema optimum sits

Differentiate {{eq:schema-budget-optimum}} in $k$, treating $R$ as smooth:

$$ \frac{R'(k)}{R(k)} = \frac{\ln 2}{H} $$ (eq:schema-first-order)

**The optimum is where schema recall's relative growth rate falls to
$\ln 2 / H$** — a constant set by the model's distractor tolerance. Since $R(k)$
saturates, its relative growth falls monotonically, so there is exactly one
crossing.

At $H = 120$, $\ln 2 / H = 0.0058$: keep adding columns while each 1% increase in
$k$ buys at least a 0.58% increase in recall. That is a rule you can apply to a
measured recall curve with no model of the schema at all.

### 6.3 The factored benchmark gap

Apply {{eq:text-to-sql-factored}} to published results. If a system scores $s_1$
on a small schema and $s_2$ on a large one with the same model, then assuming the
generation factor is unchanged,

$$ \frac{R_{\text{large}}}{R_{\text{small}}} \approx \frac{s_2}{s_1} $$ (eq:gap-attribution)

which attributes the entire gap to schema recall. That is an upper bound on the
attribution — large schemas also have harder queries and more dialects — but it
identifies where to look first, and it predicts something checkable: **giving the
model the gold schema should recover most of the gap.** When it does, the
engineering target is the retriever; when it does not, it is the generator.

## 7. Internal Mechanics

```mermaid {#fig:structured-retrieval caption="One question, three sources, and the decision that separates them. The left branch never embeds the data — the vector index retrieves SCHEMA, and eq:schema-recall-hard-ceiling makes that step a hard ceiling on everything downstream. The execution arrow is free observability (eq:execution-observability) for syntax and emptiness, and none at all for a query that runs and is wrong."}
flowchart TB
    Q["question"] --> K{"does the source have<br/>a query language?"}
    K -->|"yes: warehouse"| SC["retrieve SCHEMA<br/>tables, columns, values"]
    SC --> GEN["generate SQL"]
    GEN --> EX["EXECUTE"]
    EX -->|"syntax / empty:<br/>observable, retry"| GEN
    EX -->|"rows"| ANS["answer"]
    K -->|"no: table in a PDF"| TR["index row + header,<br/>send whole table"]
    TR --> V[("vector index")]
    K -->|"no: page is an image"| IM{"caption it,<br/>or embed the page?"}
    IM -->|"caption"| V
    IM -->|"embed the page<br/>(cite:faysse2025colpali)"| VI[("visual index")]
    V --> ANS
    VI --> ANS
```

### 7.1 What to retrieve as "schema"

Bare column names are the weakest possible representation and the most commonly
used. In increasing order of recall at fixed $k$ — which
{{sec:9-practical-example}} identifies as the only direction that does not pay
the distractor penalty:

1. **Column name.** `cust_rev_ytd`. Nearly opaque.
2. **Name plus description.** The single highest-value addition, and it is a
   documentation task rather than an ML one.
3. **Name, description, and sample values.** Resolves the questions a description
   does not: is `status` `'A'`/`'I'` or `'active'`/`'inactive'`?
   {{cite:li2023bird}} exists largely because of this.
4. **Plus foreign keys**, so retrieving one column pulls in what it joins to.
   Joins are where generated SQL fails, and the join column is frequently not
   mentioned in the question at all.

### 7.2 Hierarchical schema selection

Retrieve *tables* first, then columns within the selected tables. This is
{{ch:rag-advanced-retrieval}}'s parent–child at a different granularity, and it
helps for the same reason: the table name is a better key for a broad question
than any of its columns, and it cuts the candidate set before the expensive
decision.

### 7.3 The execution loop

```text
   generate -> execute -> [error?] -> repair -> execute -> ...
                       -> [empty?] -> is that the answer, or a bad filter?
                       -> [rows]   -> is this what was asked? UNOBSERVABLE
```

Two rules from {{ch:rag-agentic}} apply directly. **Bound the loop** — three
repair attempts, then abstain. And **terminate on a checkable signal**: "the query
executed" is checkable, "the model is satisfied" is not.

One rule is specific to SQL and is a security requirement rather than a quality
one: **execute against a read-only connection with a row limit and a statement
timeout.** A generated query is untrusted input that you are about to run against
a database. {{part:26}} treats this properly.

### 7.4 Documents that are pictures

The conventional pipeline is {{ch:rag-ingestion}}'s: OCR, layout reconstruction,
table extraction, then chunk the text. {{eq:table-recoverability}} says a great
deal is lost, and the loss is concentrated in tables and figures — the parts
most worth retrieving.

{{cite:faysse2025colpali}} inverts it: **embed the page image directly** with a
vision-language model and match with late interaction ({{cite:khattab2020colbert}}).
No OCR, no layout reconstruction, no table extraction — and therefore none of
their failures. The costs are storage (late interaction is many vectors per page,
{{ch:emb-reranking}}), a heavier encoder, and the loss of text you might have
wanted for other purposes.

**The argument is sharper than "it scores better".** {{ch:rag-ingestion}} framed
parsing loss as something to measure and minimise; this says the parser is a
*choice*, and on a corpus where the parser is what fails, deleting it is
available. {{part:13}} covers the encoders; the retrieval question is here.

## 8. Implementation

```python {tier=A name=rows-versus-queries}
"""Serialising rows and embedding them: what it can do, and what it cannot.

The reflex when structured data enters a RAG system is to make it look like the
data the system already handles -- serialise each row to a sentence, embed it,
put it in the vector index. This listing measures what that reflex costs, across
the three query shapes a table actually receives.

Two results are exact rather than empirical. A numeric predicate cannot be
answered by similarity because embeddings do not represent magnitude ORDER
(ch:rag-indexing, eq:no-order-in-embedding). An aggregate cannot be answered by
top-k at all, for the same reason ch:rag-graph's global questions cannot
(eq:aggregate-unreachable) -- except that here the gap is not approximate, and
there is no community-summary trick, because the answer requires arithmetic over
every qualifying row.
"""
import numpy as np

rng = np.random.default_rng(41)

N_ROW, DIM = 6000, 48
N_REGION = 8
N_QUERY = 500
BUDGET = 40                       # rows that fit in the context window


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


region_vec = unit(rng.normal(size=(N_REGION, DIM)))
name_vec = unit(rng.normal(size=(N_ROW, DIM)))
row_region = rng.integers(0, N_REGION, size=N_ROW)
revenue = rng.lognormal(mean=13.0, sigma=1.1, size=N_ROW)

# A serialised row: "Acme Ltd, region EMEA, revenue 1,240,000". The embedding
# carries the name strongly and the region well. The NUMBER contributes a
# direction with no ordering in it -- 1,240,000 and 980,000 are two unrelated
# token sequences as far as the encoder is concerned, which is the whole problem.
rev_token = unit(rng.normal(size=(N_ROW, DIM)))
row_vec = unit(0.75 * name_vec + 0.50 * region_vec[row_region] + 0.18 * rev_token)

THRESHOLD = np.quantile(revenue, 0.80)


def entity_lookup():
    """'Tell me about Acme Ltd.' -- the query shape vector search is built for."""
    hits = 0
    for _ in range(N_QUERY):
        i = int(rng.integers(0, N_ROW))
        q = unit(name_vec[i] + 0.3 * region_vec[row_region[i]]
                 + rng.normal(scale=0.25, size=DIM))
        top = np.argpartition(-(row_vec @ q), BUDGET)[:BUDGET]
        hits += int(i in top)
    return hits / N_QUERY


def numeric_filter():
    """'Which EMEA accounts bill more than the 80th percentile?' -- a predicate
    with an exact answer set. Report recall of that set at the same budget."""
    rec, prec = [], []
    for _ in range(N_QUERY):
        r = int(rng.integers(0, N_REGION))
        target = np.where((row_region == r) & (revenue > THRESHOLD))[0]
        if len(target) == 0:
            continue
        q = unit(region_vec[r] + 0.35 * rng.normal(size=DIM))
        top = np.argpartition(-(row_vec @ q), BUDGET)[:BUDGET]
        got = np.intersect1d(top, target)
        rec.append(len(got) / len(target))
        prec.append(len(got) / BUDGET)
    return float(np.mean(rec)), float(np.mean(prec))


def aggregate(budget):
    """'What is total EMEA revenue?' -- the model can only add up what it sees
    (eq:aggregate-error)."""
    err = []
    for _ in range(N_QUERY):
        r = int(rng.integers(0, N_REGION))
        target = np.where(row_region == r)[0]
        truth = revenue[target].sum()
        q = unit(region_vec[r] + 0.35 * rng.normal(size=DIM))
        top = np.argpartition(-(row_vec @ q), budget)[:budget]
        seen = top[row_region[top] == r]
        err.append(abs(revenue[seen].sum() - truth) / truth)
    return float(np.mean(err))


print(f"{N_ROW:,} rows across {N_REGION} regions; {BUDGET} rows fit in context\n")

print(f"entity lookup   -- recall@{BUDGET}: {entity_lookup():.3f}")
rec, prec = numeric_filter()
print(f"numeric filter  -- recall@{BUDGET}: {rec:.3f}   precision: {prec:.3f}")
print(f"                   (SQL: recall 1.000, precision 1.000)\n")

print(f"{'aggregate: rows retrieved':>26}{'mean relative error':>22}")
print("-" * 48)
for b in (40, 100, 400, 1000, 3000):
    print(f"{b:>26}{aggregate(b):>22.3f}")

print(f"""
The first line is the case the architecture was designed for, and it works.
Looking up a named entity is a similarity problem, the name dominates the
serialised row's embedding, and recall is high. If every question about your
structured data has this shape, serialise the rows and stop reading.

The second line is where it stops working, and the reason is exact rather than
statistical. "Revenue above the 80th percentile" is a predicate over a NUMBER,
and eq:no-order-in-embedding says an embedding does not represent magnitude
order -- 1,240,000 and 980,000 are two unrelated token sequences to the encoder.
So retrieval can find the region and then returns rows drawn essentially at
random with respect to the predicate. SQL answers the same question at recall
1.000 and precision 1.000, by evaluating the predicate, which is what predicates
are for.

The third block is the one to sit with. An aggregate is not a hard retrieval
problem, it is not a retrieval problem: SUM over a set is a function of EVERY
member of that set, so no top-k contains the answer (eq:aggregate-unreachable).
Watch the error fall as the budget grows and notice how it falls -- it only
reaches zero when the budget reaches the size of the whole group. That is not a
retrieval system converging. That is a retrieval system slowly turning into a
full table scan, at LLM prices, to compute something a database does in a
millisecond.

Note what this rules out. ch:rag-graph answered its global questions with
pre-computed community summaries, and there is no equivalent here: you cannot
pre-summarise "total revenue by region" without knowing which aggregate will be
asked, and there are combinatorially many (eq:aggregate-combinatorics). The
answer is not a better index. The answer is to stop indexing the data and start
generating queries against it.""")
```

If the answer is query generation, the retrieval problem moves to the schema.
The second listing measures what happens there.

```python {tier=A name=schema-retrieval-bottleneck}
"""Where text-to-SQL actually fails: retrieving the schema, not writing the SQL.

If structured data is queried rather than embedded, the retrieval problem does
not disappear -- it moves. Something has to decide WHICH tables and columns go
into the prompt, because a warehouse schema does not fit in a context window.
That decision is a text retrieval problem, and it is the one the benchmarks
punish.

It has a shape this book has met repeatedly, with one difference that makes it
harsher: BOTH failure modes are total. A column that is not retrieved cannot be
referenced, so the query is unwritable no matter how good the model is
(eq:schema-recall-hard-ceiling). A column that is retrieved and irrelevant is a
distractor, and ch:llm-function-calling's eq:max-distractor says selection
accuracy falls as they accumulate.

This listing sweeps schema size against retrieval budget and locates the
optimum -- and shows what happens to it as the schema grows from a tutorial
database to an enterprise warehouse.
"""
import numpy as np

rng = np.random.default_rng(67)

N_QUERY = 4000
P_SQL_CLEAN = 0.93          # chance of correct SQL given exactly the right columns
DISTRACTOR_HALFLIFE = 120   # distractors that halve the model's column selection
SIGMA = 0.50                # schema-retriever noise; lower is a better retriever


def trial(n_col, k):
    """One question needing 2-5 columns, against a schema of n_col columns.

    The retriever scores every column: a needed column gets signal 1 plus noise,
    an irrelevant one gets noise alone. SIGMA is therefore exactly 'how good is
    the schema retriever', which is the quantity worth improving.
    """
    recalls, ends = [], []
    kk = min(k, n_col)
    for _ in range(N_QUERY):
        n_need = min(int(rng.integers(2, 6)), n_col)
        score = rng.normal(scale=SIGMA, size=n_col)
        score[:n_need] += 1.0                       # the needed columns
        top = np.argpartition(-score, kk - 1)[:kk]
        got = bool((top < n_need).sum() == n_need)
        recalls.append(got)
        # Distractor penalty: retrieved columns that were not needed
        # (eq:max-distractor). It only matters if the needed ones arrived.
        p_sql = P_SQL_CLEAN * 0.5 ** (max(kk - n_need, 0) / DISTRACTOR_HALFLIFE)
        ends.append(got * p_sql)
    return float(np.mean(recalls)), float(np.mean(ends))


print(f"columns needed per question: 2-5; P(correct SQL | exact schema) = "
      f"{P_SQL_CLEAN}\nschema-retriever noise sigma = {SIGMA}; distractor "
      f"half-life {DISTRACTOR_HALFLIFE} irrelevant columns\n")

BUDGETS = (10, 25, 50, 100, 250, 600)
print(f"{'schema':>9}{'':>3}" + "".join(f"{'k=' + str(b):>19}" for b in BUDGETS))
print(f"{'columns':>9}{'':>3}" + "".join(f"{'recall':>10}{'end':>9}" for b in BUDGETS))
print("-" * (12 + 19 * len(BUDGETS)))

SCHEMAS = (25, 120, 600, 3000)
best = {}
for n_col in SCHEMAS:
    cells = [trial(n_col, b) for b in BUDGETS]
    # Budgets above the schema size are the same experiment; report the smallest
    # budget that achieves the best score rather than an arbitrary tie.
    top = max(e for _, e in cells)
    i = min(j for j, (_, e) in enumerate(cells) if e >= top - 1e-9)
    best[n_col] = (min(BUDGETS[i], n_col), cells[i][1])
    print(f"{n_col:>9}{'':>3}" + "".join(f"{r:>10.3f}{e:>9.3f}" for r, e in cells))

print(f"""
Read the recall pairs first. At a 25-column schema -- a tutorial database, and
roughly what a text-to-SQL demo runs on -- almost any budget retrieves every
needed column, so schema retrieval is not a problem and the end-to-end score is
essentially the model's SQL ability. This is the regime in which text-to-SQL
looks solved.

Now follow one budget down the column as the schema grows. The same k that was
sufficient at 25 columns recalls a fraction of what is needed at 3,000, and the
failure is UNRECOVERABLE: a column that is not in the prompt cannot appear in the
query, so no amount of model quality repairs it (eq:schema-recall-hard-ceiling).
This is the mechanism behind the benchmark collapse -- solved-looking accuracy on
small schemas, and something very different on an enterprise warehouse.

The obvious fix is a bigger budget, and the end-to-end columns show why it only
half works. Raising k raises recall and simultaneously fills the prompt with
irrelevant columns, and eq:max-distractor makes the model's column selection
worse as they accumulate. Both failure modes are TOTAL here -- a missing column
makes the query unwritable, a wrong column makes it wrong -- so the product has
an interior maximum well below either factor alone. At 600 columns the best
budget recalls the full column set only 57% of the time, and widening it further
LOWERS the end-to-end score.

Best budget and best achievable score, by schema size:
""" + "\n".join(f"    {n:>5} columns:  k = {best[n][0]:>3}   end-to-end "
                f"{best[n][1]:.3f}" for n in SCHEMAS) + f"""

The optimum budget rises with schema size and then stops rising, because the
distractor penalty catches up. The score at the optimum falls by a factor of
{best[25][1] / best[3000][1]:.0f} from the tutorial database to the warehouse --
with the SAME model, the same SQL ability, and the same retriever. Nothing about
the language modelling changed. The schema got bigger.

That is the mechanism behind the published benchmark gap: solved-looking
execution accuracy on small cross-domain schemas, materially lower on large
realistic ones, and far lower again on enterprise workflows
(cite:yu2018spider, cite:li2023bird, cite:lei2025spider2). Read those three
numbers as a measurement of schema retrieval, not of SQL generation.

The engineering conclusion is not "retrieve more columns". It is to make the
retrieval better rather than bigger: column descriptions and value samples rather
than bare names, hierarchical selection that picks tables before columns, and
foreign-key expansion so that retrieving one column pulls in the ones it joins
to. Every one of those raises recall at a FIXED k, which is the only direction
that does not pay the distractor penalty.""")
```

## 9. Practical Example

**Rows against queries.** Entity lookup works: recall **0.910** at a 40-row
budget. If every question your users ask has that shape — "tell me about this
customer" — serialising rows into a vector index is a reasonable architecture and
the rest of this section does not apply to you.

**The numeric predicate does not work, and the failure is structural.** Recall
**0.041**, precision **0.153**, against SQL's 1.000 and 1.000.
{{eq:predicate-unrepresentable}}: the encoder has no representation of magnitude
order, so retrieval finds the *region* and then returns rows at their base rate
with respect to the threshold — measured precision 0.153 against a base rate of
0.20, which is to say **the retriever contributes nothing on the predicate**.

**The aggregate is the one to sit with.** Relative error **0.958** at 40 rows,
**0.400** at 1,000, and still **0.090** at 3,000 — which is half the table.
{{eq:aggregate-error}} predicted at least 0.947 at a 40-row budget and the
measurement gives 0.958.

Notice *how* the error falls: only as the budget approaches the size of the
qualifying group. **That is not a retrieval system converging; it is a retrieval
system becoming a full table scan**, at LLM prices, to compute something the
database does in a millisecond. And {{eq:aggregate-combinatorics}} closes the
escape route that worked in {{ch:rag-graph}}: you cannot pre-summarise the
aggregate without knowing which one will be asked.

> **IMPORTANT:** The conclusion is not "vector search is bad at tables". It is
> that **the question "how do I embed this?" was the wrong question.** Structured
> data has a query engine; retrieval means writing a query for it. That decision
> also settles the freshness, permission, and auditability problems
> {{ch:rag-why}} raised, because the database already solved them.

**And then the problem moves.** The second listing shows where it lands. At a
**25-column** schema, any reasonable budget recalls every needed column and the
end-to-end score is essentially the model's SQL ability: **0.821**. At **3,000
columns**, the best achievable score is **0.084** — a factor of **ten**, with the
same model, the same SQL ability, and the same retriever. **Nothing about the
language modelling changed. The schema got bigger.**

The trade is sharp because both failures are total. At 600 columns the optimal
budget recalls the complete column set only **57%** of the time, and widening it
*lowers* the end-to-end score — {{eq:schema-budget-optimum}}'s interior maximum,
with {{eq:schema-scaling}}'s two derivatives visible in the same table: the
optimal budget rises (25 → 50 → 100 → 100) while the value at the optimum falls
(0.821 → 0.618 → 0.304 → 0.084).

**This is the mechanism behind the published benchmark progression.** Systems
scoring in the nineties on small cross-domain schemas
({{cite:yu2018spider}}), materially lower on large realistic ones
({{cite:li2023bird}}), and far lower on enterprise workflows
({{cite:lei2025spider2}}) are, on this account, mostly reporting schema recall.
{{eq:gap-attribution}} makes that checkable: hand the model the gold schema and
see how much of the gap closes.

**Which sets the engineering target.** Not "retrieve more columns" — that path is
capped and then reverses. **Retrieve better at a fixed $k$**: column descriptions
and sample values rather than bare names, tables selected before columns,
foreign-key expansion so a retrieved column brings what it joins to. Those are
the only improvements that do not pay the distractor penalty, and the first of
them is a documentation task rather than a machine-learning one.

## 10. Production Considerations

**Ask whether the source has a query language before designing anything.** It is
the decision that determines every other decision in this chapter.

**Never serialise rows into a vector index for analytical questions.**
{{eq:aggregate-unreachable}} is not a quality issue to tune.

**Write column descriptions.** The highest-return work in a text-to-SQL system,
and it is documentation, not modelling.

**Store sample values with the schema.** {{cite:li2023bird}}'s central finding:
the model needs to know that `status` is `'A'`/`'I'`, and no description
reliably conveys it.

**Retrieve tables before columns**, and expand along foreign keys.

**Measure schema recall separately** from execution accuracy
({{eq:text-to-sql-factored}}), and tune $k$ against the end-to-end product, not
against recall.

**Execute read-only, with a row limit and a statement timeout.** A generated
query is untrusted input ({{part:26}}).

**Log the generated SQL, always.** It is the only artefact that makes a wrong
answer debuggable, and it is the one thing a text RAG system cannot produce.

**Show the query to the user** for anything consequential. Execution
observability ends at "it ran"; a human reading the `WHERE` clause is the only
check on {{eq:execution-observability}}'s zero.

**For tables in documents: index the row with its header, send the whole table.**
{{eq:table-representations}}.

**Measure your parser's table loss before building around it**
({{eq:table-recoverability}}). If it is severe, visual retrieval
({{cite:faysse2025colpali}}) is a real option rather than an exotic one.

## 11. Common Mistakes

**Embedding serialised rows and expecting analytics.** The chapter's headline
mistake.

**Retrieving a numeric predicate.** {{eq:predicate-unrepresentable}}.

**Putting the whole schema in the prompt** — fine at 25 columns, impossible at
3,000, and the failure is a silent truncation.

**Maximising schema recall** instead of the end-to-end product.

**Bare column names as the schema representation.**

**Treating execution success as correctness.** {{eq:execution-observability}}'s
third term is zero.

**Indexing table rows without their headers** — the purest orphan chunks in the
book ({{ch:rag-advanced-retrieval}}).

**Captioning images and discarding the image**, so anything the captioner missed
is unrecoverable — {{ch:rag-ingestion}}'s ingestion loss with a different parser.

## 12. Failure Modes

**Silent underestimate.** An aggregate computed from retrieved rows is *always*
low, never high, and it looks plausible. The most dangerous failure here, because
nothing about the output signals it. Detect by asking a question whose true
answer you know.

**Missing join column.** The question does not mention it, so schema retrieval
does not retrieve it, so the query cannot be written.
{{eq:schema-recall-hard-ceiling}}. Detect by logging which needed columns were
absent from the prompt.

**Value-format mismatch.** `WHERE status = 'active'` against a column storing
`'A'`. Returns zero rows, reads as "no results found", and the user believes it.

**Ambiguous column collision.** Two tables both have `id` or `amount`; the model
picks the wrong one and the query runs perfectly.

**Repair-loop convergence on a wrong query.** Each repair fixes the reported
error and none addresses the semantics. Symptom: high execution success, low
answer accuracy — {{eq:execution-observability}} exactly.

**Table row retrieved without context.** `EMEA | 1,240,000 | 18%` reaches the
generator, which invents plausible column names for it.

**Schema drift.** A column is renamed; the index is stale; every query touching
it fails. {{ch:rag-indexing}}'s {{eq:index-staleness}} with a hard failure
instead of a soft one.

## 13. Alternatives

| Alternative | What it trades | When it wins |
|---|---|---|
| Serialised rows in a vector index | aggregates, predicates | entity lookup only |
| Text-to-SQL | schema-retrieval risk, security surface | analytical questions over a warehouse |
| Pre-built semantic layer / metrics store | flexibility | a stable, known set of business questions |
| Pre-computed aggregate summaries | arbitrary questions ({{eq:aggregate-combinatorics}}) | a small, fixed set of aggregates |
| Row-with-header chunks | exact computation | tables inside documents, where there is no engine |
| Visual page retrieval ({{cite:faysse2025colpali}}) | storage, text availability | scanned or layout-heavy corpora |
| Hand-written query tools ({{ch:llm-function-calling}}) | coverage | high-stakes questions where a wrong `WHERE` is unacceptable |

**The last row is under-used and deserves the emphasis.** For a bounded set of
important questions, five hand-written parameterised queries exposed as tools
beat a text-to-SQL system on every axis that matters: correctness, latency,
security, and auditability. Generated SQL earns its risk when the question space
is genuinely open — and most internal analytics question spaces are not.

## 14. Evaluation

**Evaluate by query shape**: lookup, filter, aggregate, join, multi-step. An
aggregate number over a mixed set hides that one shape is at zero.

**Report schema recall and generation accuracy separately.**
{{eq:text-to-sql-factored}}, and the gold-schema experiment
({{eq:gap-attribution}}) tells you which to work on.

**Use execution accuracy, not string match.** Many correct queries;
{{cite:yu2018spider}} moved the field on this.

**Evaluate on a schema the size of yours.** A system validated on 30 columns
tells you nothing about 3,000 — that is this chapter's whole second half.

**Include unanswerable questions.** A system that always emits SQL will emit SQL
for a question the schema cannot answer.

**For tables in documents, measure retrieval by table and by cell.** Finding the
right table and reading the wrong cell are different failures.

**Measure parser loss before comparing pipelines** ({{eq:table-recoverability}}).
A visual-retrieval comparison is only meaningful against a measured text baseline.

## 15. Advanced Concepts

**The semantic layer as retrieval.** {{maturity:MATURE}} A metrics layer defines
`revenue` once, and generation targets *metrics* rather than raw SQL. It shrinks
the schema-retrieval problem by an order of magnitude and removes the
definitional ambiguity that causes most wrong-but-executing queries. Boring,
effective, and rarely discussed in the RAG literature because it is not new.

**Value indexes.** {{maturity:EMERGING}} Index the distinct *values* of
low-cardinality columns so a question mentioning "Acme Corp" retrieves the column
containing it. A direct attack on {{eq:schema-recall-hard-ceiling}} that raises
recall at fixed $k$.

**Visual document retrieval.** {{maturity:EMERGING}}
{{cite:faysse2025colpali}} makes the parsing stage optional. The interesting
implication for this book is architectural: {{ch:rag-ingestion}} treated parsing
loss as something to minimise, and this makes it something you can decline to
incur.

**The multimodal retrieval question is a representation question.**
{{maturity:EMERGING}} Whether to embed an image, its caption, or both is the same
choice as {{ch:rag-advanced-retrieval}}'s indexed-versus-sent split, one modality
over. Index the caption, send the image; or index the page, send the page.
{{part:13}} covers the encoders.

**Text-to-SQL as an agent.** {{maturity:EMERGING}}
{{cite:lei2025spider2}}'s tasks need multiple queries, intermediate results, and
dialect-specific behaviour — which is {{ch:rag-agentic}}'s loop with
{{eq:execution-observability}}'s asymmetric visibility, and the reason enterprise
scores remain low even with strong models.

## 16. Connection to Previous Chapters

{{ch:rag-indexing}}'s {{eq:no-order-in-embedding}} is why
{{eq:predicate-unrepresentable}} holds, and {{ch:rag-graph}}'s
{{eq:global-aggregate}} is {{eq:aggregate-unreachable}} in a setting where the
stratified escape was available and here is not.
{{ch:rag-advanced-retrieval}}'s {{eq:contextual-augmentation}} is exactly the
right treatment for table rows, which are the book's purest orphan chunks.
{{ch:llm-function-calling}}'s {{eq:max-distractor}} sets the schema budget's
upper limit, and {{ch:llm-structured-output}} is how the generated query parses.
{{ch:rag-agentic}}'s observability argument reappears as
{{eq:execution-observability}}, with the unusual property of being free for some
failures and unavailable for others. {{ch:rag-ingestion}}'s
{{eq:table-recoverability}} is what {{cite:faysse2025colpali}} proposes to route
around entirely.

## 17. Exercises

1. Prove {{eq:aggregate-unreachable}} and state the one class of aggregate for
   which a top-$k$ retrieval *can* be exact.
2. Derive {{eq:aggregate-error}} and use it to compute the budget needed for 5%
   accuracy on a 750-row group.
3. In `rows-versus-queries`, replace the naive sum with the scaled estimator
   $\frac{M}{k}\sum$. Does the error improve, and what does the estimator need
   that you do not have?
4. Raise `THRESHOLD` to the 99th percentile. Does the numeric filter get better
   or worse, and why?
5. In `schema-retrieval-bottleneck`, halve `SIGMA` — a better schema retriever.
   How much of the 3,000-column collapse does it recover?
6. Sweep `DISTRACTOR_HALFLIFE` from 40 to 400. At what value does "show the whole
   schema" become optimal at 600 columns?
7. Use {{eq:schema-first-order}} on the measured recall curve to predict $k^{*}$
   at 600 columns. Compare with the table.
8. Design the gold-schema experiment of {{eq:gap-attribution}} for a real system.
   What exactly do you hold fixed?

## 18. Interview Questions

1. Why can a vector database not answer "what was total revenue last quarter"?
2. Why does embedding a serialised row fail on numeric filters?
3. What is schema retrieval and why is it the bottleneck?
4. Why does showing more of the schema eventually make text-to-SQL worse?
5. Text-to-SQL scores in the nineties on one benchmark and far lower on another.
   Explain.
6. How would you represent a table found inside a PDF for retrieval?
7. What does executing a generated query tell you, and what does it not?
8. When would you hand-write query tools instead of generating SQL?
9. Your SQL agent has 95% execution success and 60% answer accuracy. Diagnose.
10. When is embedding page images better than parsing them?

## 19. Research Questions

1. {{eq:schema-first-order}} gives $k^{*}$ from a recall curve and a distractor
   half-life. Can the half-life be measured cheaply per model, so the budget is
   set rather than tuned?
2. Semantic errors in generated SQL are unobservable
   ({{eq:execution-observability}}). Is there a cheap consistency check —
   result-set cardinality, unit sanity, cross-query agreement — that recovers
   some observability?
3. Value indexes raise recall at fixed $k$. What is the right index for
   high-cardinality columns, where indexing every value is impractical?
4. Visual retrieval avoids parsing loss and pays in storage. Where is the
   crossover as a function of corpus layout complexity?
5. Aggregates are unreachable by retrieval and trivial by query. Is there a
   hybrid representation — sketches, pre-aggregated cubes — that makes a useful
   subset reachable without {{eq:aggregate-combinatorics}}'s explosion?

## 20. Chapter Summary

**The question is not "how do I embed this?" but "does this have a query
language?"** — and for the highest-value structured data the answer is yes, which
changes the architecture rather than the tuning.

**Two results are exact.** A numeric predicate cannot be evaluated by similarity,
because {{eq:no-order-in-embedding}} gives embeddings no representation of
magnitude order — measured recall **0.041**, precision **0.153**, against SQL's
1.000. And an aggregate cannot be answered by top-$k$ at all
({{eq:aggregate-unreachable}}), because it is a function of every qualifying row:
error **0.958** at a 40-row budget, still **0.090** at 3,000 rows. The error falls
only as the budget approaches the whole group, so the system is not converging —
**it is becoming a table scan at LLM prices.** {{ch:rag-graph}}'s
stratified-summary escape is unavailable here, because
{{eq:aggregate-combinatorics}} makes pre-computation infeasible.

**So retrieval becomes query generation, and the retrieval problem moves to the
schema.** {{eq:text-to-sql-factored}} factors accuracy into schema recall and
generation, and {{eq:schema-recall-hard-ceiling}} makes the first a hard ceiling:
a column that was never shown cannot appear in the query, whatever the model.

**Both schema failure modes are total, so the budget has a sharp interior
optimum.** Measured: end-to-end **0.821** at 25 columns and **0.084** at 3,000 —
a factor of ten with the model, its SQL ability, and the retriever all held
fixed. The optimal budget rises (25 → 100) while the value at the optimum falls,
exactly as {{eq:schema-scaling}} says. **This is the mechanism behind the
published benchmark gap**, and {{eq:gap-attribution}} makes it testable.

**The fix is better retrieval at fixed $k$, not more columns**: descriptions,
sample values, table-then-column selection, foreign-key expansion. Widening the
budget is capped and then reverses.

**Execution is free observability, and only partially.** Syntax errors and empty
results are visible without a grader — the thing {{ch:rag-agentic}} showed is
load-bearing. Semantically wrong queries are not visible at all
({{eq:execution-observability}}), which is why execution success is not an
accuracy metric and why the generated SQL must be logged and, for anything
consequential, shown.

**For tables inside documents, the rule is {{ch:rag-advanced-retrieval}}'s:**
index the row with its header, send the table. A bare table row is the purest
orphan chunk in the book.

**And for pages that are pictures, the newest argument is the sharpest**
({{cite:faysse2025colpali}}): if the parser is what destroys the tables, deleting
the parser is an available move. {{ch:rag-ingestion}} treated parsing loss as
something to minimise; it is also something you can decline to incur.

## 21. Further Reading

{{cite:yu2018spider}} for cross-schema generalisation as the measured problem,
and for execution accuracy as the right metric.
{{cite:li2023bird}} next, for why the values in the columns matter as much as the
schema — the finding behind this chapter's sample-values recommendation.
{{cite:lei2025spider2}} for what happens at enterprise scale, and read its
headline gap against {{eq:text-to-sql-factored}} rather than as a statement about
SQL ability.
{{cite:faysse2025colpali}} for visual document retrieval, with
{{cite:khattab2020colbert}} for the late-interaction machinery it uses and the
storage cost that comes with it.
{{cite:liu2023lost}} for why a large retrieved schema is not free.
{{cite:gao2023ragsurvey}} for where structured and multimodal retrieval sit in
the standard taxonomy; {{part:13}} for the encoders, and {{ch:aids-text-to-sql}}
for text-to-SQL as a data-science workflow rather than a retrieval problem.

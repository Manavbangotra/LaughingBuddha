# -*- coding: utf-8 -*-
# Extracted from: Chapter 116 — Structured and Multimodal RAG: SQL, Tables, and Images
# Source: src/.../ch116-structured-multimodal.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

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

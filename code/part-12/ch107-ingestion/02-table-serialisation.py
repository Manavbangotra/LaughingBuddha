# -*- coding: utf-8 -*-
# Extracted from: Chapter 107 — Document Ingestion and Parsing
# Source: src/.../ch107-ingestion.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""What survives when a table becomes text.

A table is a function from (row, column) to a value. Once it is text, that
function is recoverable or it is not (eq:table-recoverability) -- and no
embedding model, chunk size, or reranker can restore what serialisation destroyed.

We serialise the same table four ways, chunk each, and ask a cell-level question:
does a chunk exist from which the correct (row, column) value can be read?
"""
import numpy as np

rng = np.random.default_rng(8)

REGIONS = ["North America", "EMEA", "APAC", "LATAM"]
QUARTERS = ["Q1", "Q2", "Q3", "Q4"]
values = np.round(rng.uniform(2.0, 9.0, size=(len(REGIONS), len(QUARTERS))), 1)


def flattened():
    """What a naive PDF extractor produces: cells in reading order, no structure."""
    out = list(QUARTERS)
    for r, region in enumerate(REGIONS):
        out.append(region)
        out += [f"{v}" for v in values[r]]
    return [" ".join(out)]                       # one undifferentiated blob


def markdown():
    rows = ["| Region | " + " | ".join(QUARTERS) + " |",
            "|---" * (len(QUARTERS) + 1) + "|"]
    for r, region in enumerate(REGIONS):
        rows.append("| " + region + " | "
                    + " | ".join(f"{v}" for v in values[r]) + " |")
    return ["\n".join(rows)]                     # one chunk, structure intact


def row_records():
    return [f"{region}: " + ", ".join(f"{q} {v}" for q, v in zip(QUARTERS, values[r]))
            for r, region in enumerate(REGIONS)]


def cell_sentences():
    return [f"For {region} in {q}, revenue was {values[r][c]} million."
            for r, region in enumerate(REGIONS)
            for c, q in enumerate(QUARTERS)]


def cell_recoverable(chunks, region, quarter, value):
    """Is there a chunk in which this cell's value is UNAMBIGUOUSLY attached to
    both its row key and its column key? That is eq:table-recoverability."""
    val = f"{value}"
    for ch in chunks:
        if region not in ch or val not in ch:
            continue
        if quarter not in ch:
            continue
        # The value must not be ambiguous within the chunk: if the chunk contains
        # several quarters AND several values, position is doing the work, and
        # position is what a flat serialisation destroys.
        n_quarters = sum(1 for q in QUARTERS if q in ch)
        n_values = sum(1 for v in np.ravel(values) if f"{v}" in ch)
        if n_quarters > 1 and n_values > 1 and "|" not in ch and ":" not in ch:
            continue                              # flat blob: not recoverable
        return True
    return False


schemes = {"flattened (naive extractor)": flattened(),
           "markdown table": markdown(),
           "one record per row": row_records(),
           "one sentence per cell": cell_sentences()}

print(f"{'serialisation':<30}{'chunks':>8}{'tokens':>8}{'vs flat':>9}"
      f"{'cells recoverable':>20}")
print("-" * 75)
flat_tokens = sum(len(ch.split()) for ch in schemes["flattened (naive extractor)"])
for name, chunks in schemes.items():
    ok = sum(cell_recoverable(chunks, REGIONS[r], QUARTERS[c], values[r][c])
             for r in range(len(REGIONS)) for c in range(len(QUARTERS)))
    total = values.size
    tokens = sum(len(ch.split()) for ch in chunks)
    print(f"{name:<30}{len(chunks):>8}{tokens:>8}{tokens / flat_tokens:>8.1f}x"
          f"{f'{ok}/{total}':>12}  ({ok / total:>4.0%})")

print(f"""
The flattened row is what a naive PDF extractor produces, and every number is
present in it. The association between a number and its two headers is not,
because in the original that association was carried by POSITION on the page --
and position is exactly what flattening discards. Asked for EMEA in Q3, a perfect
retriever returns this chunk and a perfect model reads some number. There is no
way for it to read the right one.

This is the clearest case in the whole part of a failure that CANNOT be fixed
downstream. No embedding model, no chunk size, no reranker, and no larger
generator recovers information that is not in the text.

The three structured serialisations all preserve it, and they trade differently.
Markdown keeps the whole table in one chunk, which is compact but means the model
must parse a grid at generation time and means one cell cannot be retrieved
without the other {values.size - 1}. One sentence per cell costs about five times
the flattened token count -- read the 'vs flat' column -- and makes every cell
independently retrievable with no structural parsing at all.

For retrieval specifically, verbosity is the right trade far more often than
teams expect: tokens at ingestion are paid once, and ch:rag-why showed the
recurring cost is k chunks per query, not corpus size.""")

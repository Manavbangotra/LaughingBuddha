---
id: rag-ingestion
number: 107
part: XII
tier: full
status: draft
requires: [rag-why, nlp-preprocessing, emb-what-they-are, ds-cleaning]
provides: [document-parsing, layout-analysis, reading-order, table-extraction,
           ingestion-loss-rate, ocr-pipeline, document-deduplication,
           ingestion-observability]
citations: [lewis2020rag, gao2023ragsurvey, lee2022dedup, thakur2021beir]
---

## 1. Learning Objectives

By the end of this chapter you will be able to explain why PDF text extraction is
genuinely hard rather than merely fiddly, and predict which documents will break;
implement and measure reading-order recovery for multi-column layouts; choose a
table serialisation by measuring what survives it rather than by preference;
define and instrument **ingestion loss rate**, the metric that no retrieval
dashboard contains; and design a pipeline whose failures are visible instead of
silent.

## 2. Why This Matters

{{ch:rag-why}} listed four stages where a question can be lost. This chapter is
the first one, it is the least written about, and in document-heavy systems it
loses more answers than the other three combined.

The reason is a structural mismatch. **A PDF does not contain a document. It
contains instructions for drawing one** — glyphs at coordinates, with no notion
of paragraph, column, table, or reading order. Every one of those has to be
*inferred*, and every inference is a place to be wrong. A table becomes a stream
of numbers with no association to their headers. A two-column page interleaves
into alternating half-sentences. A running footer appears in the middle of every
chunk.

None of this shows up in a retrieval metric, because **the content never reached
the index to be retrieved.** Recall@$k$ is computed over what was indexed, so a
document destroyed at ingestion is invisible to every dashboard downstream — and
the team spends its time tuning the retriever.

{{maturity:MATURE}} The techniques are old and well understood in the document
analysis literature. {{maturity:EMERGING}} Vision-language models as parsers are
displacing rule-based extraction and the cost model is still moving.

## 3. Prerequisites

{{ch:rag-why}}'s four stages; {{ch:nlp-preprocessing}} for tokenisation and text
normalisation; {{ch:emb-what-they-are}} for what the embedder will do with the
text you produce; {{ch:ds-cleaning}} for the data-quality discipline this chapter
applies to documents.

## 4. Intuitive Explanation

### What a PDF actually is

The single most useful fact in the chapter. A PDF is a **page description
language**: a sequence of operations that place glyphs at coordinates on a
canvas. It says *"draw the character 'T' at (72, 690) in 11pt Times"*, several
thousand times.

It does not say:

- where a paragraph starts or ends;
- that these two blocks are columns and should be read one after the other;
- that this grid of numbers is a table, or which header each cell belongs to;
- that this line at the bottom of every page is a footer and not content;
- what order any of it should be read in.

**All of that is inferred by the extractor from geometry**, and every extractor
infers it differently. This is why the same document produces different text from
three libraries, and why "PDF parsing" is a research area rather than a
subroutine.

The word processor that produced the PDF knew all of it and threw it away. That
is not a bug in PDF — it was designed to make pages *look* identical everywhere,
and it succeeds completely. Extraction is running the pipeline backwards.

### The failure that looks like a retrieval problem

Here is the shape of the whole chapter in one example.

A financial report contains a table: rows are quarters, columns are business
units, cells are revenue. A naive extractor emits:

```text
Q1 Q2 Q3 Q4 North America 4.2 4.8 5.1 5.6 EMEA 3.1 3.0 3.4 3.9
```

Every number is present. **The association between numbers and their headers is
gone**, and no embedding, chunk size, or reranker recovers it, because the
information is not in the text. Asked "what was EMEA revenue in Q3", a perfect
retriever returns this chunk and a perfect model reads `3.4` — or `3.0`, or
`5.1`, with no way to tell.

The answer is not in the index. The bug is four stages upstream of where it will
be investigated.

### Ingestion loss rate

The metric this chapter exists to introduce, because almost nobody computes it:

> **What fraction of the source content reached the index in a usable form?**

For a clean HTML corpus, near 100%. For scanned PDFs with tables, it is routinely
much lower — and the number is knowable in an afternoon by sampling documents and
checking. **Until you have it, you do not know whether your RAG system has a
retrieval problem or a parsing problem**, and those have nothing in common.

## 5. Formal Explanation

### 5.1 The pipeline

$$ \text{source} \xrightarrow{\ \text{extract}\ } \text{spans} \xrightarrow{\ \text{layout}\ } \text{blocks} \xrightarrow{\ \text{order}\ } \text{reading order} \xrightarrow{\ \text{normalise}\ } \text{text} $$ (eq:ingestion-pipeline)

Four inferences, each of which can fail independently. Most libraries collapse
them into one call, which is convenient and is why the failures are hard to
localise.

### 5.2 Reading order

A page is a set of text spans $s_i = (x_i, y_i, w_i, h_i, \text{text}_i)$. The
extractor must produce a permutation $\pi$ giving reading order.

The naive rule — sort by $y$, then by $x$ — is correct for single-column text and
**wrong for every multi-column layout**, because it interleaves columns line by
line:

$$ \pi_{\text{naive}} = \operatorname{arg\,sort}_i \big(\lfloor y_i / h \rfloor,\; x_i\big) $$ (eq:naive-reading-order)

Correct handling requires *segmentation first*: find the column boundaries, then
order within each. A serviceable detector uses the projection profile — the
histogram of span coverage along $x$ — and cuts at sustained gaps:

$$ g(x) = \sum_i \Ind\big[x \in [x_i, x_i + w_i]\big], \qquad \text{cut where } g(x) = 0 \text{ over a run} > \delta $$ (eq:projection-profile)

This is a 1970s technique and it works. {{sec:9-practical-example}} implements it
and measures what it recovers.

### 5.3 Measuring reading order

Since $\pi$ is a permutation, compare against the true order $\pi^{*}$. Kendall
tau is the obvious choice and it is **the wrong one**, for the reason
{{ch:emb-hybrid}} gave about rank correlation: it is dominated by distant pairs
nobody cares about. What matters locally is whether consecutive text stayed
consecutive:

$$ \text{adjacency}(\pi) = \frac{1}{n-1}\big|\{\,i : \pi \text{ places } \pi^{*}_{i+1} \text{ immediately after } \pi^{*}_i\,\}\big| $$ (eq:reading-order-adjacency)

**This is the right metric because it is the one chunking depends on**: a chunk
is a contiguous window of the extracted stream, so broken adjacency is exactly
what puts unrelated text in one chunk.

### 5.4 Ingestion loss, formally

Let $U(d)$ be the *usable* content of document $d$ — the assertions a reader
could extract — and $\hat{U}(d)$ what survives ingestion:

$$ \text{loss} = 1 - \frac{1}{|D|}\sum_{d \in D} \frac{|\hat{U}(d) \cap U(d)|}{|U(d)|} $$ (eq:ingestion-loss)

Not directly computable, but well approximated by sampling: take 30 documents,
write down the questions each should answer, and check whether the extracted text
contains the answer. **Thirty documents is enough to distinguish 5% loss from
30% loss**, which is the decision you need.

> **PRODUCTION TIP:** Do this before building anything else in the pipeline. It
> takes a morning and it determines whether the next month is spent on parsing or
> on retrieval. Teams that skip it spend the month on retrieval regardless of the
> answer.

### 5.5 Table serialisation

A table is a function from (row key, column key) to a cell. Serialisation to text
must preserve enough for that function to be recoverable:

$$ \text{recoverable} \iff \exists \text{ a span of the text from which } (r, c) \mapsto v \text{ can be read} $$ (eq:table-recoverability)

Naive flattening fails this. The options that do not:

| Serialisation | Preserves | Cost |
|---|---|---|
| flattened text | nothing | smallest |
| markdown / HTML table | full structure | moderate; needs the model to parse it |
| one sentence per cell | full structure, locally | largest, by a lot |
| row-wise records | row-level structure | moderate |

**One sentence per cell** — *"For EMEA in Q3, revenue was 3.4."* — is verbose and
wins on retrieval, because it makes each cell independently retrievable and
requires no structural parsing at generation time. {{sec:9-practical-example}}
measures the difference and it is not small.

## 6. Mathematical Foundation

### 6.1 Why the loss is invisible downstream

Formally, and this is the argument for instrumenting ingestion at all. Retrieval
recall is measured over the index:

$$ \text{recall@}k = \Prob\big[\,z^{*} \in \text{top-}k \given z^{*} \in \mathcal{I}\,\big] $$ (eq:conditional-recall)

conditioned on the answer being *in the index*. The quantity that matters is
unconditional:

$$ \Prob[z^{*} \in \text{top-}k] = \underbrace{\Prob[z^{*} \in \mathcal{I}]}_{1 - \text{ingestion loss}} \times \text{recall@}k $$ (eq:true-recall)

**Every retrieval dashboard reports the second factor and calls it recall.** A
system at 0.90 measured recall and 30% ingestion loss has true recall 0.63, and
no amount of retriever work moves the first factor. {{eq:true-recall}} is the
reason this chapter comes before the retrieval chapters.

### 6.2 The cost of parsing well

Three tiers, with roughly the cost structure that decides between them:

$$ C_{\text{rule}} \approx 10^{-5}\,\text{\$/page}, \qquad C_{\text{specialist}} \approx 10^{-2}\,\text{\$/page}, \qquad C_{\text{VLM}} \approx 10^{-2}\text{--}10^{-1}\,\text{\$/page} $$ (eq:parsing-cost-tiers)

Three to four orders of magnitude, which sounds decisive until you compare
against the alternative. Parsing is a **one-time cost per document**; a wrong
answer is a recurring cost per query. For a corpus of $N_d$ documents serving $Q$
queries:

$$ \text{worth it} \iff N_d \cdot \Delta C_{\text{parse}} \;<\; Q \cdot \Delta(\text{error rate}) \cdot C_{\text{error}} $$ (eq:parsing-worth-it)

**For any corpus that is queried more than a few times per document, expensive
parsing wins easily** — and that is essentially every RAG system. The instinct to
economise on ingestion is close to always wrong, and {{eq:parsing-worth-it}} is
why.

### 6.3 Deduplication

{{cite:lee2022dedup}} showed duplicate training data harms language models; in a
retrieval corpus the damage is different and more immediate. If a fact appears in
$m$ near-identical documents, they occupy $m$ of the $k$ retrieval slots:

$$ \text{effective } k = k - (m - 1) \times \Prob[\text{duplicates co-retrieved}] $$ (eq:duplicate-slot-loss)

and duplicates are *maximally* likely to be co-retrieved, since they are near
each other in embedding space by construction. **Duplicates do not merely waste
context; they crowd out the diverse evidence a good answer needs**, and the
effect is worst exactly when a question requires combining two sources.

Near-duplicate detection by MinHash over shingles is cheap and standard; the
subtlety is that in a document corpus the *right* granularity is often the
section rather than the document, because boilerplate repeats across otherwise
distinct files.

## 7. Internal Mechanics

```mermaid {#fig:ingestion-pipeline caption="The four inferences of eq:ingestion-pipeline, each with its characteristic failure. Most libraries expose only the endpoints, which is why these failures are hard to localise — and why instrumenting each stage separately is worth the effort."}
flowchart TD
    S["source file"] --> E["extract glyphs<br/>+ coordinates"]
    E --> L["layout: group spans<br/>into blocks"]
    L --> O["reading order<br/>(eq:projection-profile)"]
    O --> N["normalise:<br/>dehyphenate, strip<br/>headers/footers"]
    N --> T["text + metadata"]
    E -.->|"scanned page:<br/>no glyphs at all"| F1["OCR required"]
    L -.->|"table read as<br/>flowing text"| F2["structure lost"]
    O -.->|"columns interleaved"| F3["sentences spliced"]
    N -.->|"footer kept"| F4["every chunk polluted"]
```

### 7.1 Format by format

| Format | Difficulty | Characteristic failure |
|---|---|---|
| plain text / Markdown | trivial | none worth naming |
| HTML | easy | boilerplate — navigation, cookie banners, footers |
| DOCX | easy | structure is present; most parsers discard it anyway |
| PDF (digital) | **hard** | reading order, tables, headers |
| PDF (scanned) | **very hard** | OCR errors compound with all of the above |
| slides | hard | reading order is genuinely ambiguous |
| spreadsheets | hard | the meaning is in formulas and layout, not values |
| email | moderate | quoted threads duplicate content endlessly |

**DOCX deserves a note.** It is a zip of XML with headings, lists, and tables
explicitly marked, and most pipelines convert it to PDF or plain text first,
destroying that structure to reuse the PDF path. If a meaningful share of the
corpus is DOCX, parsing it natively is a large and cheap win.

### 7.2 Headers, footers, and boilerplate

The most under-appreciated failure, because the damage is proportional to how
*well* the rest of the pipeline works. A running footer appears on every page, so
after chunking it appears in every chunk — adding a constant term to every
embedding, which by {{ch:emb-what-they-are}}'s anisotropy argument compresses the
dynamic range of every similarity score in the corpus.

Detection is easy and rarely done: **text that repeats at the same vertical
position across most pages is boilerplate.** Two passes over the document and a
frequency threshold.

### 7.3 What to keep besides the text

Ingestion is the only stage that can see this information, so anything not
captured here is lost permanently:

- **Source identity and version** — required for citation ({{ch:rag-generation}})
  and for reproducing what the retriever could have seen.
- **Position** — page and section, so a citation can point somewhere a human can
  check.
- **Section hierarchy** — the heading path a chunk sits under, which
  {{ch:rag-advanced-retrieval}} uses to give chunks context they lack alone.
- **Access-control labels** — attached here or they are attached nowhere, and
  retrofitting them means re-ingesting.
- **Timestamps** — for freshness filtering and for {{ch:rag-why}}'s churn
  measurement.

**The heading path is the highest-value and most-omitted item.** A chunk reading
*"performance improved by 12%"* is nearly useless; the same chunk labelled
*"Q3 Report → EMEA → Logistics"* is retrievable and citable.

## 8. Implementation

```python {tier=A name=reading-order-recovery}
"""Why multi-column PDFs produce spliced sentences, and how to fix it.

A PDF gives you positioned text spans and no reading order. We build a page with
a header, two columns of body text, and a footer -- then compare the naive
extraction rule (sort by y, then x) against a projection-profile column detector
(eq:projection-profile).

The metric is adjacency preservation (eq:reading-order-adjacency): what fraction
of consecutive source sentences remain consecutive after extraction. That is the
quantity chunking depends on.
"""
import numpy as np

rng = np.random.default_rng(3)

PAGE_W, LINE_H = 612.0, 14.0
COL_X = [72.0, 320.0]          # two columns, left edges
COL_W = 220.0
LINES_PER_COL = 24
Y_TOP, Y_TOL = 690.0, 4.0

spans, true_order = [], []
order = 0

# Header and footer: same position on the page, repeated content.
spans.append({"x": 72.0, "y": 730.0, "w": 468.0, "text": "[HEADER] Annual Report"})
header_idx = len(spans) - 1

# Column 1 is read fully before column 2 -- that is the ground truth.
for col, x0 in enumerate(COL_X):
    for line in range(LINES_PER_COL):
        y = Y_TOP - line * LINE_H
        spans.append({"x": x0, "y": y, "w": COL_W,
                      "text": f"c{col}l{line:02d}"})
        true_order.append(len(spans) - 1)

spans.append({"x": 72.0, "y": 40.0, "w": 468.0, "text": "[FOOTER] page 7 of 42"})
footer_idx = len(spans) - 1


def adjacency(extracted, truth):
    """Fraction of consecutive true pairs that remain consecutive (eq:reading-order-adjacency)."""
    pos = {s: i for i, s in enumerate(extracted)}
    good = 0
    for a, b in zip(truth, truth[1:]):
        if a in pos and b in pos and pos[b] == pos[a] + 1:
            good += 1
    return good / (len(truth) - 1)


def naive_order(spans):
    """Sort by line, then x (eq:naive-reading-order). Correct for one column."""
    return sorted(range(len(spans)),
                  key=lambda i: (-round(spans[i]["y"] / Y_TOL), spans[i]["x"]))


def column_order(spans):
    """Detect columns from the projection profile, then read each in turn.

    Full-width spans (headers, footers, figure captions) would bridge every
    vertical gap, so they are excluded from the PROFILE -- but they are still
    content, so they are still emitted, in y order around the columns.
    """
    full = [i for i, s in enumerate(spans) if s["w"] >= 0.6 * PAGE_W]
    body = [i for i, s in enumerate(spans) if s["w"] < 0.6 * PAGE_W]

    # Coverage histogram along x over BODY spans only (eq:projection-profile).
    cover = np.zeros(int(PAGE_W) + 1)
    for i in body:
        s = spans[i]
        cover[int(s["x"]):int(s["x"] + s["w"])] += 1

    gaps, run = [], 0
    for x, c in enumerate(cover):
        if c == 0:
            run += 1
        else:
            if run > 20:                       # a sustained gap is a boundary
                gaps.append((x - run, x))
            run = 0
    boundaries = [0] + [(a + b) // 2 for a, b in gaps] + [int(PAGE_W)]

    above = sorted([i for i in full if spans[i]["y"] > Y_TOP],
                   key=lambda i: -spans[i]["y"])
    below = sorted([i for i in full if spans[i]["y"] <= Y_TOP],
                   key=lambda i: -spans[i]["y"])

    out = list(above)
    for lo, hi in zip(boundaries, boundaries[1:]):
        in_col = [i for i in body if lo <= spans[i]["x"] < hi]
        out += sorted(in_col, key=lambda i: -spans[i]["y"])
    return out + below


def is_boilerplate(spans, i):
    """Full-width text in the header/footer band. In a real corpus the test is
    'repeats at the same y across most pages', which is two passes and a
    frequency threshold."""
    s = spans[i]
    return s["w"] > 0.6 * PAGE_W and (s["y"] > 720 or s["y"] < 60)


def strip_boilerplate(spans, order_fn):
    keep = [i for i in range(len(spans)) if not is_boilerplate(spans, i)]
    sub = [spans[i] for i in keep]
    return [keep[i] for i in order_fn(sub)]


strategies = {
    "naive (y, then x)": naive_order(spans),
    "column-aware": column_order(spans),
    "column-aware + boilerplate strip": strip_boilerplate(spans, column_order),
}

print(f"{'extraction strategy':<36}{'adjacency':>11}{'header/footer kept':>21}")
print("-" * 68)
for name, ext in strategies.items():
    kept = sum(1 for i in ext if i in (header_idx, footer_idx))
    print(f"{name:<36}{adjacency(ext, true_order):>11.3f}{kept:>21d}")

for label in ("naive (y, then x)", "column-aware"):
    seq = [spans[i]["text"] for i in strategies[label]][:6]
    print(f"\nfirst six spans, {label:<18} {' '.join(seq)}")

print("""
Read the naive row's adjacency and then look at the span sequence beneath it.
Sorting by y-then-x alternates between the two columns on every line, so the
extracted stream is column-one-line-one, column-two-line-one, column-one-line-two
-- an interleaving of two unrelated texts. Nearly every consecutive pair in the
source is broken.

That is what produces the spliced half-sentences familiar from any RAG system
built over two-column PDFs, and it happens BEFORE chunking. No chunk size helps:
the stream itself is scrambled, so every contiguous window of it contains
material from both columns.

The column-aware row uses eq:projection-profile -- a coverage histogram along x,
cut at sustained gaps -- which is a technique from the 1970s and takes twenty
lines. Adjacency is restored almost completely.

The last row adds boilerplate removal, and the adjacency number does not move at
ALL -- which is exactly why this failure is neglected. It does not show up in a
reading-order metric, or in any retrieval metric either. Its damage is that the header and footer text lands in every chunk
of the document, adding a constant component to every embedding -- which, by the
anisotropy argument of ch:emb-what-they-are, compresses the dynamic range of
every similarity score in the corpus.""")
```

```python {tier=A name=table-serialisation}
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
```

## 9. Practical Example

**Reading order.** The naive $y$-then-$x$ rule scores near zero on adjacency for
a two-column page, and the span sequence shows why: it alternates between columns
on every line, interleaving two unrelated texts. **This happens before chunking,
so no chunk size can help** — the stream itself is scrambled, and every
contiguous window of it contains material from both columns. Those spliced
half-sentences are the signature failure of RAG over two-column PDFs.

The projection-profile detector ({{eq:projection-profile}}) restores adjacency
almost completely, in about twenty lines, using a technique older than PDF
itself.

The boilerplate row is the instructive one: **stripping headers and footers barely
moves the adjacency metric at all.** That is exactly why the failure is
neglected — it is invisible to the reading-order measurement. Its damage is
elsewhere: the footer lands in every chunk of the document, adding a constant
component to every embedding, which by {{ch:emb-what-they-are}}'s anisotropy
argument compresses similarity dynamic range across the whole corpus.

**Table serialisation.** The flattened row contains every number and recovers no
cells. The association between a value and its two headers was carried by
*position on the page*, and flattening discards position.

This is the cleanest example in the part of **a failure that cannot be fixed
downstream.** Not "hard to fix" — impossible, because the information is not in
the text. A perfect retriever returns the chunk and a perfect model reads a
number, with nothing to distinguish the right one.

All three structured serialisations preserve recoverability and trade
differently. Markdown is compact — about twice the flattened token count — but couples every
cell to every other and requires grid parsing at generation time. One sentence
per cell costs roughly five times the flattened count and makes each cell
independently retrievable with no parsing at all.
**Verbosity is the right trade more often than teams expect**, because ingestion
tokens are paid once while {{ch:rag-why}} showed the recurring cost is $k$ chunks
per query — independent of how large the corpus became.

## 10. Production Considerations

**Measure ingestion loss before anything else** ({{eq:ingestion-loss}}). Thirty
documents, a morning, and it determines whether the next month goes to parsing or
retrieval.

**Instrument each stage of {{eq:ingestion-pipeline}} separately.** Spans
extracted, blocks found, columns detected, boilerplate removed, characters
emitted. A single "documents ingested" counter cannot localise anything.

**Alert on per-document character yield.** A document that produced 200
characters when its peers produce 20,000 is a scanned page, a parse failure, or
an encoding problem. This one check catches most catastrophic ingestion failures
and costs nothing.

**Keep the raw source and make ingestion reproducible.** You will change the
parser, and you need to re-ingest without re-fetching. Store the original bytes
with a content hash.

**Spend on parsing.** {{eq:parsing-worth-it}}: it is a one-time per-document cost
against a recurring per-query error rate, and for any realistic query volume the
expensive parser wins by a wide margin.

**Parse DOCX, HTML, and Markdown natively** rather than converting to PDF. The
structure is already there and the conversion destroys it.

**Attach metadata at ingestion or never** — source, version, page, heading path,
access labels, timestamp. Retrofitting means re-ingesting the corpus.

**Deduplicate at the section level**, not the document level
({{eq:duplicate-slot-loss}}), because boilerplate repeats across otherwise
distinct files.

**Make ingestion incremental and idempotent.** A corpus of any size will be
re-ingested many times — a parser upgrade, a new metadata field, a bug fix — and
a pipeline that only runs whole-corpus takes that from an afternoon to a
scheduling problem. Key each document by a content hash so an unchanged document
is skipped, and make re-running the pipeline over the same input produce
byte-identical output. The second property is what lets you diff two parser
versions and see exactly which documents changed, which is the only practical way
to review a parser upgrade over a corpus too large to read.

**Quarantine rather than drop.** A document that fails to parse should land in a
queue someone looks at, not in a log line. The failures cluster — one scanner
setting, one template, one export tool — so a quarantine of a hundred documents
usually resolves into three fixable causes, while a hundred log lines resolve
into nothing.

## 11. Common Mistakes

**Treating extraction as solved because a library returns a string.** It returns
*a* string.

**Never looking at the extracted text.** Read fifty chunks by hand. It takes an
hour and it is the highest-information hour in the project.

**Flattening tables.** {{eq:table-recoverability}}, and unfixable downstream.

**Keeping headers and footers.** Invisible in reading-order metrics, corrosive to
embeddings.

**Converting DOCX to PDF first.** Discarding structure that was already present.

**Ignoring scanned pages.** A PDF with no extractable text yields an empty
document and, in most pipelines, no error.

**Deduplicating by exact hash.** Near-duplicates are the problem; exact
duplicates are rare.

**Assuming one parser for all formats.** {{sec:7-internal-mechanics}}'s table
exists because the failures differ by format.

## 12. Failure Modes

**Silent empty extraction.** Scanned or protected PDF yields no text; the
pipeline records a successful ingest of an empty document. Catch with the
character-yield check.

**Column interleaving.** Spliced sentences throughout. Symptom: chunks that read
like two documents shuffled.

**Table flattening.** Numbers present, associations gone. Symptom: confidently
wrong numeric answers, and a retriever that returns the right chunk.

**Boilerplate contamination.** Every chunk carries the footer. Symptom:
compressed similarity range, and queries matching the wrong document because they
matched its footer.

**Encoding mojibake.** Ligatures and smart quotes become replacement characters,
breaking tokenisation and exact-match retrieval.

**Hyphenation across line breaks.** `perfor-\nmance` becomes two tokens, neither
of which matches "performance".

**Duplicate crowding.** {{eq:duplicate-slot-loss}}: retrieval returns $k$ copies
of the same document, worst when the question needs two sources.

**Version drift.** The corpus is re-ingested with a new parser and half the
chunks change, silently invalidating every cached embedding and every stored
citation offset.

## 13. Alternatives

**Vision-language model parsers.** Send the page image to a multimodal model and
ask for structured Markdown. Handles layout, tables, and scans in one step,
costs {{eq:parsing-cost-tiers}}'s top tier, and — per
{{eq:parsing-worth-it}} — is frequently worth it. The failure mode is new and
important: **the model can silently hallucinate a plausible table.**

**Commercial document AI services.** Purpose-built extraction with table and form
models. The middle tier, and the pragmatic default for PDF-heavy corpora.

**OCR pipelines.** Required for scans; layout analysis is still needed
afterwards, so OCR is a prerequisite rather than a solution.

**Ask for better source data.** The corpus is often available as HTML, Markdown,
or a database export, and someone converted it to PDF for distribution.
**Finding the pre-PDF source is the single highest-leverage move in this chapter
and it is a conversation, not an engineering task.**

**Manual curation.** For a few hundred high-value documents, structured by hand.
Unfashionable, and frequently the correct answer for a small critical corpus.

## 14. Evaluation

**Ingestion loss rate** ({{eq:ingestion-loss}}) on a sampled set, recomputed
whenever the parser changes.

**Reading-order adjacency** ({{eq:reading-order-adjacency}}) on documents with
known structure.

**Table cell recoverability** ({{eq:table-recoverability}}) — the fraction of
cells answerable from some chunk.

**Character yield per document**, as a distribution. The left tail is your
failure list.

**Boilerplate rate** — the fraction of chunk tokens that are repeated
page furniture.

**Duplicate rate** at the section level.

**And a human read of fifty random chunks.** No metric in this list substitutes
for it, because the failures are diverse and most of them are obvious on sight.

## 15. Advanced Concepts

**Layout analysis is a solved research problem and an unsolved engineering one.**
The document-analysis literature has decades of work on segmentation and reading
order; almost none of it reaches RAG pipelines, which typically call one library
function. The gap is distribution, not knowledge.

**VLM parsing changes the failure distribution rather than removing failures.**
Rule-based extraction fails *loudly* — garbled text is visibly garbled. A VLM
fails *quietly*, producing a clean, plausible, wrong table. **Loud failures are
much cheaper**, so a VLM parser needs verification a rule-based one does not:
round-trip a sample and check numbers against the source.

**Chunking begins at ingestion.** {{ch:rag-chunking}} assumes a clean text
stream with structure markers, and every structural boundary the parser preserves
— heading, section, table, list — is a boundary chunking can use. **The two
stages are one design**, and a parser that emits flat text has already
constrained the chunker to guessing.

**Ingestion is where access control is decided.** Labels attached here propagate;
labels attached later do not exist for already-ingested documents. This is a
security property established by a data-engineering decision, which is exactly
the kind of coupling that produces incidents.

**The corpus is a product, not an input.** {{ch:rag-why}}'s point, arriving with
teeth: the largest available quality win is often writing the twenty missing
documents rather than improving any component. Ingestion is where that becomes
visible, because it is the stage that reveals what the corpus actually contains.

## 16. Connection to Previous Chapters

{{ch:rag-why}}'s stage one, and {{eq:true-recall}} is why it must be measured
before the retrieval chapters. {{ch:emb-what-they-are}}'s anisotropy argument is
what makes boilerplate contamination expensive rather than merely untidy.
{{ch:emb-hybrid}}'s tokenisation warning applies directly — an extractor that
mangles identifiers destroys exactly what the lexical index was kept for.
{{ch:nlp-preprocessing}}'s normalisation decisions happen here.
{{ch:ds-cleaning}}'s discipline — measure the loss, do not assume it — is the
whole chapter in one sentence.

## 17. Exercises

1. Derive {{eq:true-recall}} and compute true recall for a system reporting 0.92
   with 25% ingestion loss.
2. Use {{eq:parsing-worth-it}} to find the query volume at which a $0.05/page
   parser beats a free one, given it reduces the error rate by 8 points and an
   error costs $0.10.
3. In `reading-order-recovery`, add a third column. Does the projection-profile
   detector still work? What about a column of different width?
4. Add a full-width figure caption spanning both columns. Predict what the
   detector does, then check, and propose a fix.
5. In `table-serialisation`, add a table with merged header cells spanning two
   columns. Which serialisations still preserve recoverability?
6. Compute the token cost of one-sentence-per-cell for a corpus of 10,000
   tables at 40 cells each. Using {{ch:rag-why}}'s cost model, is it material?
7. Design the character-yield alert: what threshold, computed how, and what does
   it do about a corpus that legitimately contains short documents?
8. Take ten real PDFs from your own work, extract them with any library, and
   compute {{eq:ingestion-loss}} by hand. Report the number honestly.

## 18. Interview Questions

1. Why is PDF text extraction hard?
2. What is ingestion loss rate and why does no retrieval dashboard show it?
3. A two-column paper produces spliced sentences. Where is the bug and what fixes
   it?
4. How do you get a table into a RAG system?
5. Why do headers and footers matter more than they look like they should?
6. When is a vision model worth it for parsing, and what new risk does it add?
7. What metadata must be captured at ingestion, and why can't it be added later?
8. Your RAG system is wrong about numbers specifically. Diagnose.
9. How would you detect that a document failed to ingest?
10. Would you deduplicate a retrieval corpus? At what granularity?

## 19. Research Questions

1. Is there a reliable *automatic* measure of ingestion loss — one that does not
   require a human to write down what each document should answer?
2. Can a VLM parser be made to fail loudly, flagging low-confidence regions
   rather than emitting a confident wrong table?
3. What is the right serialisation of a table for retrieval, as a function of the
   query distribution? {{sec:9-practical-example}} shows the trade-off; nobody
   has characterised the optimum.
4. Reading order in slides and forms is genuinely ambiguous even for humans. Is
   there a principled formulation, or is it inherently task-dependent?
5. Ingestion and chunking are treated as separate stages and are one design.
   What does a joint formulation look like, and does it beat the pipeline?

## 20. Chapter Summary

**A PDF does not contain a document; it contains instructions for drawing one.**
Paragraphs, columns, tables, headers, and reading order are all *inferred* from
geometry, and each inference in {{eq:ingestion-pipeline}} fails independently.

The failures are invisible downstream, and {{eq:true-recall}} says why: every
retrieval dashboard reports recall *conditioned on the answer being in the
index*, so a system at 0.90 measured recall with 30% ingestion loss has true
recall 0.63 — and no retriever work moves the first factor. **Measure ingestion
loss before touching retrieval.**

Two failures were measured directly. Naive $y$-then-$x$ ordering interleaves
columns line by line, destroying nearly every consecutive pair before chunking
begins, and a projection-profile detector from the 1970s restores it in twenty
lines. And flattened tables contain every number while recovering **no** cells,
because the value/header association was carried by position and flattening
discards position — **the clearest case in this part of a failure that cannot be
fixed downstream by any embedding, chunk size, reranker, or model.**

Structured serialisations trade differently: markdown is compact and couples
cells together; one sentence per cell is verbose and makes each independently
retrievable. Verbosity usually wins, because ingestion tokens are paid once and
the recurring cost is $k$ chunks per query.

Finally, {{eq:parsing-worth-it}}: parsing is a one-time per-document cost against
a recurring per-query error rate, so **the instinct to economise on ingestion is
close to always wrong.** And the highest-leverage move in the chapter is not
technical — it is finding the HTML or database export that existed before someone
made the PDF.

## 21. Further Reading

The document-analysis literature is the right source here and it predates RAG by
decades; projection profiles, XY-cut segmentation, and reading-order recovery are
all standard there and absent from RAG practice.
{{cite:gao2023ragsurvey}} covers ingestion in a paragraph, which is
representative of the field's attention and is the point
{{sec:2-why-this-matters}} makes.
{{cite:lee2022dedup}} for why duplicates matter, in the training setting; the
retrieval consequence in {{eq:duplicate-slot-loss}} is different and sharper.
{{cite:lewis2020rag}} assumes clean Wikipedia passages throughout — worth noting
how much of the field's evaluation inherits that assumption.

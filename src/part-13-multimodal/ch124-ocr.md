---
id: mm-ocr
number: 124
part: XIII
tier: full
status: draft
requires: [mm-cv-fundamentals, mm-vit, mm-segmentation, rag-ingestion,
           llm-structured-output]
provides: [ocr-stages, character-error-rate, field-error-amplification,
           silent-numeric-error, ocr-free-document-understanding,
           pipeline-versus-end-to-end, text-layer-as-artefact]
citations: [kim2022donut, huang2022layoutlmv3, mathew2021docvqa,
            wang2024qwen2vl, faysse2025colpali, ronneberger2015unet]
---

## 1. Learning Objectives

By the end of this chapter you will be able to describe the OCR pipeline's stages
and say which one fails on which kind of document; derive **field-error
amplification** and use it to convert a quoted character error rate into the
number your system actually experiences; distinguish **visible** OCR failures from
**silent** ones and say which validation catches which; compute the crossover at
which an end-to-end model beats a pipeline, and show it is a property of your
fields rather than of the technologies; and state the thing accuracy tables omit —
that the two architectures produce **different artefacts**, not just different
numbers.

## 2. Why This Matters

{{ch:rag-structured}} ended on the observation that a great deal of enterprise
knowledge is pages that are pictures. This chapter is how the text comes out, and
what it costs when it comes out wrong.

**The central problem is a units mismatch.** OCR accuracy is quoted per character;
nothing downstream consumes characters. An extraction pipeline wants a *field*
exactly right, and a field is many characters, so
{{sec:9-practical-example}} measures what "99% accurate" becomes: a 6-character
invoice code survives **0.946** of the time, a 14-digit account number **0.863**,
and a 200-character paragraph **0.138**.

**One number, three wildly different realities**, and the conversion is
exponential. That is why "our OCR is 99% accurate" and "our extraction pipeline is
unusable" are routinely both true statements about the same system.

The second measurement is about which failures you can see. Alphabetic errors
corrupt into things a human or a dictionary flags. Numeric errors are the danger,
and the result is more nuanced than the usual warning: **about a third** of
numeric corruptions survive a type check, because the commonest confusions —
`0`/`O`, `1`/`l`, `5`/`S` — cross the digit/letter boundary and a format check
catches them. **The remaining third parse cleanly as a different, wrong number**,
and no format validation touches them.

{{maturity:ESTABLISHED}} The OCR pipeline. {{maturity:MATURE}} OCR-free document
understanding ({{cite:kim2022donut}}) and general VLMs reading pages directly
({{cite:wang2024qwen2vl}}), which have changed the architecture question from *how
do I improve OCR* to *do I need it*.

## 3. Prerequisites

{{ch:rag-ingestion}} for ingestion loss and reading order — this chapter supplies
the character-level mechanics that chapter's {{eq:ingestion-loss}} summarises;
{{ch:mm-vit}} for {{eq:patch-compression}}, which decides whether small text is
legible to a vision tower at all; {{ch:mm-segmentation}} for text detection as a
segmentation problem; {{ch:llm-structured-output}} for getting a parsed field out
of a model.

## 4. Intuitive Explanation

### The pipeline, and where each stage fails

```text
   image -> [detect] -> [recognise] -> [order] -> [layout] -> [extract]
              boxes       characters    reading    tables      fields
                                        sequence   forms
```

| Stage | Fails on |
|---|---|
| **detect** — find text regions | low contrast, handwriting, text on images, rotation |
| **recognise** — region to characters | unusual fonts, degraded scans, dense small print |
| **order** — sequence the regions | multi-column layouts, sidebars, footnotes |
| **layout** — group into structures | tables, forms, anything where 2D position is the meaning |
| **extract** — populate fields | everything above, compounded |

**The last row is the point.** Each stage's output is the next stage's input, so
the errors compose — and the recognise stage's error is *itself* amplified by
field length before it even reaches the rest.

### Why a character error rate is the wrong number

A field of $L$ characters is correct only if every character is correct. At a
character error rate $\varepsilon$, the field survives with probability
$(1-\varepsilon)^L$.

At $\varepsilon = 0.01$ — the "99% accurate" figure — a 60-character field is
wrong **45%** of the time. The advertised number did not change; the length did.

**So the first thing to do with any quoted OCR accuracy is convert it into your
units**, using your actual field lengths. It usually turns a comfortable number
into an uncomfortable one, and it does so before any modelling work.

### Visible and silent failures

Corrupt a letter in a name and you get a word that is probably not a word. A human
notices; a dictionary notices.

Corrupt a digit in an amount and you get **another amount**. It parses. It passes
the type check. It satisfies the regex. It is simply the wrong number, and nothing
downstream can tell.

{{sec:9-practical-example}} splits this precisely, and the split changes where
effort goes:

- **Two thirds** of numeric corruptions cross into letters — `0`→`O`, `5`→`S` — and
  a format check catches them for free. **Do that first.**
- **One third** stay all-digits. No format validation helps. What helps is
  redundancy OCR cannot corrupt consistently: **check digits, cross-field
  arithmetic** (do the line items sum to the total?), and agreement between two
  independent extractions.

**And this explains a specific production pathology.** A financial-document
pipeline looks excellent on a spot-check: the names and addresses are visibly
fine, and the malformed numbers were already rejected by the format check. The
numbers still wrong are exactly the ones that look right — and they are the reason
the pipeline exists.

### Or: do not parse it at all

{{cite:kim2022donut}} asked whether the text layer is necessary. Feed the page
image to a model and have it emit the structured answer directly — no OCR, no
reading order, no table reconstruction, and therefore none of their failures.
General VLMs now do the same thing ({{cite:wang2024qwen2vl}}), and
{{cite:faysse2025colpali}} applies the argument to retrieval.

{{sec:9-practical-example}} finds the crossover, and the answer is a decision
rule rather than a winner: **for an 8-character field the pipeline holds up to a
0.96% character error rate; for a 60-character field it loses at 0.13%** — seven
times tighter. The crossover is a property of *your fields*, and the same document
can put short fields on one side of the line and long ones on the other.

### The thing the accuracy table cannot show

And here is what usually decides it, which no benchmark reports.

**The pipeline produces an artefact.** A text layer that can be indexed for
retrieval, cited back to a page location, diffed between document versions,
searched by a human who does not trust the extraction, and re-processed later when
the extraction logic changes — without re-reading the images.

**The end-to-end model produces an answer.** Ask a second question and you pay for
a second full inference over the page.

So they are not substitutes even at equal accuracy. **If you need an answer to one
known question, end-to-end is simpler. If you need a searchable, auditable corpus
— which is what most document systems actually are — the intermediate text is the
product**, and its error rate is something to manage rather than a reason to
abandon the architecture.

## 5. Formal Explanation

### 5.1 Character error rate

$$ \text{CER} = \frac{S + D + I}{N} $$ (eq:cer)

for substitutions, deletions and insertions against $N$ reference characters —
edit distance normalised by length. Note it can exceed 1, and note what it does
*not* say: nothing about *which* characters, and OCR errors are far from uniform.

### 5.2 Field error amplification

$$ \Prob[\text{field correct}] = (1 - \varepsilon)^{L} \;\approx\; e^{-\varepsilon L} $$ (eq:field-error-amplification)

so the **effective** error rate at the field level is

$$ \varepsilon_{\text{field}} \approx \varepsilon L \quad \text{for small } \varepsilon L $$ (eq:effective-field-error)

**{{eq:effective-field-error}} is the conversion to memorise**: multiply the
quoted CER by your field length. At $\varepsilon = 0.01$ and $L = 60$ that is 0.6,
and the exact value is $1 - 0.99^{60} = 0.453$.

### 5.3 Pipeline composition

$$ \Prob[\text{extraction correct}] = \underbrace{(1-\varepsilon)^L}_{\text{OCR}} \times \underbrace{p_{\text{layout}}}_{\text{located}} \times \underbrace{p_{\text{parse}}}_{\text{slotted}} $$ (eq:pipeline-composition)

A product of stage successes — {{ch:rag-failures}}'s
{{eq:stage-cascade}} in a new domain, with the same consequence: the marginal
value of improving any stage is proportional to the product of the others, so
**localise before optimising**.

### 5.4 Silent errors

Partition corruptions by whether they survive validation:

$$ \Prob[\text{silent}] = \Prob[\text{corrupted}] \cdot \Prob[\text{output still well-formed} \mid \text{corrupted}] $$ (eq:silent-numeric-error)

For a digit string, "well-formed" means all digits. If a fraction $\phi$ of
confusions map a digit to a letter, the silent share is $1 - \phi$.
{{sec:9-practical-example}} measures the silent share at **≈35%** for numeric
fields under a realistic confusion model, and **0%** for alphabetic fields under a
dictionary check.

**The undetectable third is where validation effort belongs**, and format checks
do not reach it.

### 5.5 The end-to-end comparison

$$ \text{pipeline}(\varepsilon, L) = (1-\varepsilon)^L p_{\text{layout}} p_{\text{parse}}, \qquad \text{end-to-end} = p_{e2e} $$ (eq:e2e-comparison)

Setting them equal and solving for $\varepsilon$:

$$ \varepsilon^{*}(L) = 1 - \left(\frac{p_{e2e}}{p_{\text{layout}}p_{\text{parse}}}\right)^{1/L} $$ (eq:ocr-crossover)

**{{eq:ocr-crossover}} decreases in $L$**, so the longer the field, the lower the
CER a pipeline must achieve to stay competitive. Measured: **0.96% at $L=8$** and
**0.13% at $L=60$**.

### 5.6 What the comparison omits

Let $Q$ be the number of distinct questions asked of a document over its lifetime,
and $C$ the cost of one full model pass:

$$ \text{cost}_{\text{pipeline}} = C_{\text{parse}} + Q\,c_{\text{query on text}}, \qquad \text{cost}_{\text{e2e}} = Q\,C $$ (eq:artefact-economics)

with $c_{\text{query on text}} \lll C$. **The pipeline amortises; the end-to-end
model does not.** And beyond cost, the text layer supports operations the
end-to-end architecture has no equivalent for at all — citation, diffing, keyword
search, and reprocessing under new extraction logic.

$$ \boxed{\text{they produce different artefacts, not merely different accuracies}} $$ (eq:different-artefacts)

## 6. Mathematical Foundation

### 6.1 The conversion, worked

At $\varepsilon = 0.01$:

| $L$ | $(1-\varepsilon)^L$ | measured |
|---|---|---|
| 6 | $0.99^6 = 0.941$ | **0.946** |
| 8 | $0.99^8 = 0.923$ | **0.930** |
| 14 | $0.99^{14} = 0.869$ | **0.863** |
| 40 | $0.99^{40} = 0.669$ | **0.671** |
| 200 | $0.99^{200} = 0.134$ | **0.138** |

{{eq:field-error-amplification}} predicts every row within a few thousandths.
**This is not an empirical finding to be tuned; it is arithmetic**, and it can be
applied to a vendor's quoted number before any pilot.

### 6.2 Inverting it: what CER do you need?

For a target field accuracy $\alpha$:

$$ \varepsilon \le 1 - \alpha^{1/L} $$ (eq:required-cer)

For $\alpha = 0.99$ on a 40-character field: $\varepsilon \le 1 - 0.99^{1/40} =
2.5 \times 10^{-4}$. **A quarter of one tenth of a per cent.**

That is far below what general OCR achieves on real documents, and it is the
honest reason document pipelines have human review steps. Not because the
technology is immature — because {{eq:required-cer}} demands a rate that is not
available.

> **MATH NOTE:** {{eq:field-error-amplification}} assumes independent per-character
> errors, which flatters OCR in one direction and not the other. Real errors are
> *correlated*: a degraded region produces a run of errors, so the variance is
> higher than independent sampling implies. That means fields are more often
> either perfect or badly wrong, and less often wrong by one character — which
> makes the *mean* prediction roughly right and the *distribution* more bimodal
> than modelled. For planning purposes the mean is what you need; for designing a
> review queue, the bimodality is good news, because bad pages cluster.

### 6.3 The crossover's shape

Differentiating {{eq:ocr-crossover}} for small $\varepsilon^*$:

$$ \varepsilon^{*}(L) \approx \frac{1}{L}\ln\frac{p_{\text{layout}}p_{\text{parse}}}{p_{e2e}} \;\propto\; \frac{1}{L} $$ (eq:crossover-inverse-length)

**The required CER falls as $1/L$.** Measured: 0.96% at $L = 8$ and 0.13% at
$L = 60$ — a ratio of 7.4 against a predicted $60/8 = 7.5$.

So a single document with an 8-character reference number and a 60-character
address can genuinely want *different architectures for different fields*, and
{{eq:crossover-inverse-length}} says exactly where the line falls.

## 7. Internal Mechanics

```mermaid {#fig:ocr-architectures caption="Two architectures and what each leaves behind. The pipeline's errors compose (eq:pipeline-composition) and its OCR stage is amplified by field length; its compensation is the text layer, which is an asset the end-to-end path never creates (eq:different-artefacts). The hybrid is what most production systems converge on."}
flowchart TB
    IMG["page image"] --> DET["detect text regions"]
    DET --> REC["recognise characters<br/>CER e"]
    REC --> ORD["reading order"]
    ORD --> LAY["layout / tables"]
    LAY --> TXT[("TEXT LAYER<br/>indexable, citable,<br/>diffable, reusable")]
    TXT --> EXT["extract fields"]
    IMG --> E2E["end-to-end model<br/>reads the page"]
    E2E --> ANS["answer only"]
    EXT --> ANS2["answer"]
    TXT -.->|"amortises over<br/>many questions"| Q["further questions"]
    E2E -.->|"full inference<br/>per question"| Q
```

### 7.1 Detection is a segmentation problem

Finding text regions is {{ch:mm-segmentation}} with one class, and it inherits
that chapter's issues directly: **thin strokes are the structures a coarse stride
erases**, so a detector that downsamples aggressively loses small print before
recognition sees it.

This is also where {{ch:mm-vit}}'s {{eq:patch-compression}} bites for VLM-based
reading: if the text stroke is thin relative to the patch and the projection is
compressive, the characters are gone at tokenisation.

### 7.2 Recognition, and why fonts matter more than they should

Recognition is sequence prediction over a cropped region, and its errors cluster
by *visual* similarity rather than semantic. The standard confusion set —
`0`/`O`, `1`/`l`/`I`, `5`/`S`, `8`/`B`, `rn`/`m` — is the same across engines
because it is a property of the glyphs.

**That is exploitable.** Knowing a field is numeric lets you restrict the output
alphabet, which removes the digit/letter confusions entirely — the two thirds of
numeric errors {{sec:9-practical-example}} shows a format check catches, caught
*before* they are made instead of after.

### 7.3 Where handwriting sits

Printed text recognition is largely solved on clean scans. Handwriting is not, and
the gap is much larger than the benchmarks suggest because handwriting
distributions vary per writer rather than per document class.

The practical consequence: **a system that meets its accuracy target on printed
forms will not meet it on the handwritten fields of the same forms**, and those
fields need either a separate model, a much lower confidence threshold, or routing
to human review.

### 7.4 Confidence scores, and what they are worth

Most engines emit a per-character or per-word confidence. It is the single most
useful signal for building a review queue — **and it is not calibrated**, so treat
it as a ranking rather than a probability ({{ch:emb-similarity}}'s rule again).

The correct use is: sort by confidence, review the bottom $k$%, and choose $k$
from a measured precision/recall curve on a labelled sample. The incorrect use is
a fixed confidence threshold copied from a tutorial.

## 8. Implementation

```python {tier=A name=field-error-amplification}
"""OCR error rates are quoted per character. Nothing downstream consumes characters.

A vendor reports 99% character accuracy and it sounds close to solved. The unit is
the problem: no downstream step cares about characters. An extraction pipeline
cares whether a FIELD came out exactly right, and a field is many characters, so
the per-field error compounds (eq:field-error-amplification).

Worse, the errors are not equally visible. A wrong letter in a name is obvious to
a human reader. A wrong digit in an amount produces a different, perfectly
well-formed number, and nothing downstream can tell (eq:silent-numeric-error).

This listing simulates character-level noise and measures what it does at the
field level, separating the visible failures from the silent ones.
"""
import numpy as np

rng = np.random.default_rng(71)

N_TRIAL = 20000
DIGITS = "0123456789"
ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# Characters OCR actually confuses, rather than uniformly random substitutions.
CONFUSABLE = {"0": "O", "O": "0", "1": "l", "l": "1", "5": "S", "S": "5",
              "8": "B", "B": "8", "2": "Z", "Z": "2", "6": "G", "G": "6",
              "rn": "m"}


def corrupt(s, cer):
    """Apply per-character errors at rate `cer`, preferring realistic
    confusions where one exists."""
    out = []
    for ch in s:
        if rng.random() < cer:
            if ch in CONFUSABLE and len(CONFUSABLE[ch]) == 1:
                out.append(CONFUSABLE[ch])
            else:
                pool = DIGITS if ch.isdigit() else ALPHA
                out.append(pool[int(rng.integers(0, len(pool)))])
        else:
            out.append(ch)
    return "".join(out)


def make_field(kind, length):
    if kind == "numeric":
        return "".join(DIGITS[int(rng.integers(0, 10))] for _ in range(length))
    return "".join(ALPHA[int(rng.integers(0, len(ALPHA)))] for _ in range(length))


FIELDS = [("invoice code", "alpha", 6),
          ("amount", "numeric", 8),
          ("account number", "numeric", 14),
          ("name and address line", "alpha", 40),
          ("paragraph", "alpha", 200)]

print(f"{'field':<24}{'chars':>7}" + "".join(f"{'CER ' + str(c):>12}"
                                             for c in (0.001, 0.005, 0.01, 0.05)))
print(f"{'':<24}{'':>7}" + "".join(f"{'exact %':>12}" for _ in range(4)))
print("-" * 79)

table = {}
for name, kind, L in FIELDS:
    row = []
    for cer in (0.001, 0.005, 0.01, 0.05):
        ok = 0
        for _ in range(N_TRIAL // 4):
            s = make_field(kind, L)
            ok += int(corrupt(s, cer) == s)
        row.append(ok / (N_TRIAL // 4))
    table[name] = row
    print(f"{name:<24}{L:>7}" + "".join(f"{v:>12.3f}" for v in row))

print(f"\n\nSILENT vs VISIBLE failures at CER = 0.01\n")
print(f"{'field':<24}{'wrong':>9}{'still parses':>15}{'silent share':>15}")
print("-" * 63)
for name, kind, L in FIELDS:
    wrong = silent = 0
    for _ in range(N_TRIAL // 4):
        s = make_field(kind, L)
        t = corrupt(s, 0.01)
        if t != s:
            wrong += 1
            # A numeric field that is still all digits parses fine and is wrong.
            if kind == "numeric" and t.isdigit():
                silent += 1
    n = N_TRIAL // 4
    share = silent / wrong if wrong else 0.0
    print(f"{name:<24}{wrong / n:>9.3f}{silent / n:>15.3f}{share:>15.1%}")

acct = table["account number"][2]
para = table["paragraph"][2]
print(f"""
Read the top table across the CER = 0.01 column, which is the "99% accurate"
figure a vendor would quote. A six-character invoice code survives it
{table['invoice code'][2]:.3f} of the time. A fourteen-digit account number
survives {acct:.3f} of the time. A two-hundred-character paragraph survives
{para:.3f} of the time.

That spread comes from one equation and no modelling assumptions:
eq:field-error-amplification says a field of L characters survives with
probability (1 - CER)^L, so the per-field error rate grows with field length
while the advertised number stays fixed. 99% per character is about 86% per
account number and 14% per paragraph. The metric and the requirement are
denominated in different units, and the conversion is exponential.

This is why "our OCR is 99% accurate" and "our extraction pipeline is unusable"
are routinely both true, and why the argument about it goes nowhere: the two
sides are quoting the same system in different units.

The second table asks a different question -- not how often a field is wrong, but
how often being wrong is DETECTABLE. Alphabetic fields corrupt into things a
human or a dictionary can flag. Numeric fields are the interesting case, and the
answer is more nuanced than "you cannot tell".

About a third of numeric corruptions here survive a type check
(eq:silent-numeric-error): {table['account number'][2]:.3f} exact for the account
number, with roughly 35% of its failures still being all-digits and therefore
parsing cleanly as a different, wrong number. The other two thirds cross the
digit/letter boundary -- 0 to O, 1 to l, 5 to S -- and a format check catches
them.

That splits the validation problem into two halves with different answers. The
two thirds that break the format are cheap to catch: a regex, a type cast, a
length check. Do that first, because it is nearly free and it removes most of the
error mass.

The remaining third is the dangerous part and no amount of format validation
touches it, because every corruption of a digit string into another digit string
is a well-formed value. What works there is redundancy OCR cannot corrupt
consistently: check digits, cross-field arithmetic -- do the line items sum to the
stated total? -- and agreement between two independent extractions.

Which explains a specific production pathology. A pipeline reading financial
documents looks excellent on a spot-check, because the names and addresses are
visibly fine and the malformed numbers were already rejected. The numbers that
remain wrong are exactly the ones that look right, and they are the reason anyone
built the pipeline.""")
```

The first listing prices the pipeline's dominant error. The second asks whether to
have a pipeline at all.

```python {tier=A name=pipeline-versus-end-to-end}
"""Parse the document, or look at it? The crossover, and what the table omits.

Two architectures answer a question about a document image.

  PIPELINE     OCR -> text -> layout -> extract. Errors compound along the chain
               (eq:pipeline-composition), and the OCR stage's error is amplified
               by field length (eq:field-error-amplification).

  END-TO-END   a model reads the page image and emits the answer
               (cite:kim2022donut, cite:wang2024qwen2vl). One error rate, no
               amplification, and no intermediate artefact.

The pipeline's disadvantage is compounding; its advantage is that every stage
produces something you can inspect, index, cite and diff. This listing finds where
the accuracy crossover sits, and then prices the thing accuracy does not capture.
"""
import numpy as np

CERS = (0.002, 0.005, 0.01, 0.02, 0.05)
FIELD_LENS = (8, 20, 60)
P_LAYOUT = 0.97          # the field is located correctly on the page
P_PARSE = 0.98           # the located text is parsed into the right slot
E2E = 0.88               # end-to-end accuracy, independent of field length


def pipeline(cer, length):
    """eq:pipeline-composition: every stage must succeed."""
    p_ocr = (1.0 - cer) ** length
    return p_ocr * P_LAYOUT * P_PARSE


print(f"pipeline stages: OCR (length-dependent) x layout {P_LAYOUT} "
      f"x parse {P_PARSE}")
print(f"end-to-end model: flat {E2E}, no length dependence\n")
print(f"{'field chars':>12}{'':>3}" + "".join(f"{'CER ' + str(c):>11}"
                                              for c in CERS) + f"{'e2e wins at':>14}")
print("-" * 84)

cross = {}
for L in FIELD_LENS:
    vals = [pipeline(c, L) for c in CERS]
    # The CER at which the pipeline falls below the end-to-end model
    # (eq:ocr-crossover), found by bisection.
    lo, hi = 0.0, 0.5
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if pipeline(mid, L) > E2E:
            lo = mid
        else:
            hi = mid
    cross[L] = hi
    print(f"{L:>12}{'':>3}" + "".join(f"{v:>11.3f}" for v in vals)
          + f"{hi:>14.4f}")

print(f"""
Read a row across and the pipeline's problem is visible: its accuracy falls with
CER, and how fast depends on the field length, because eq:field-error-amplification
sits inside eq:pipeline-composition. The end-to-end model has no such term -- its
{E2E} does not care how long the field is.

The last column is the decision boundary. For an 8-character field the pipeline
stays ahead until the OCR character error rate reaches {cross[8]:.2%}, which is
comfortable: a decent engine on clean print is well inside that. For a
60-character field the pipeline loses at {cross[60]:.2%} -- an error rate that is
hard to achieve on anything but pristine scans, and a factor of
{cross[8]/cross[60]:.0f} tighter than the requirement for the short field.

So the crossover is not a property of the two technologies. It is a property of
YOUR FIELDS. Short, structured fields favour the pipeline; long free-text fields
favour reading the page directly, and the same system can be on both sides of the
line for different fields on the same document.

That is the useful form of the answer, and it is also why the question "is OCR
obsolete?" has no answer. Ask instead: how long are the fields I need, and what is
my measured CER on MY documents -- not on the vendor's benchmark, which is
almost certainly cleaner.

Now the part this table cannot show, and it is usually the deciding factor.

The pipeline produces an intermediate artefact: a text layer. That text can be
indexed for retrieval (ch:rag-ingestion), cited back to a location on the page,
diffed between two versions of a document, searched by a human who does not trust
the extraction, and re-processed later when the extraction logic changes without
re-reading the images. The end-to-end model produces an answer and nothing else.
Ask it a second question and you pay for a second full inference over the page.

So the two architectures are not substitutes even where their accuracy matches.
They produce different things. If what you need is an answer to one known
question, the end-to-end model is often better and simpler. If what you need is a
searchable, auditable corpus -- which is what almost every document system
actually needs -- the pipeline's intermediate text is the product, and its
accuracy is a property to manage rather than a reason to abandon it.

The hybrid follows from that and is what most serious systems converge on: run
the pipeline to produce the text layer, and use a vision model for the fields the
pipeline is measurably bad at -- which eq:field-error-amplification tells you in
advance are the long ones, and ch:mm-layout tells you are the ones inside tables.""")
```

## 9. Practical Example

**One accuracy figure, three different systems.** At the "99% accurate" character
error rate, a 6-character invoice code survives **0.946** of the time, a 14-digit
account number **0.863**, a 40-character address line **0.671**, and a
200-character paragraph **0.138**.

{{eq:field-error-amplification}} predicts every one of those within a few
thousandths — 0.941, 0.869, 0.669, 0.134. **This is arithmetic, not an empirical
finding**, so it can be applied to a vendor's quoted number before any pilot
starts. And {{eq:required-cer}} inverts it: 99% accuracy on a 40-character field
demands a CER of $2.5 \times 10^{-4}$, which is why document pipelines have human
review steps.

**The visible/silent split changes where validation effort goes.** At CER 0.01,
the account number is wrong **12.5%** of the time — and only **35%** of those
failures are silent. The other 65% cross the digit/letter boundary (`0`→`O`,
`5`→`S`) and a format check catches them for nothing.

> **IMPORTANT:** That leaves a third of numeric errors that **no format validation
> can reach**, because every corruption of a digit string into another digit
> string is a well-formed value. Those need redundancy OCR cannot corrupt
> consistently — check digits, cross-field arithmetic, or two independent
> extractions agreeing. **And they produce the pathology worth naming**: a
> financial pipeline looks excellent on a spot-check, because the names are
> visibly fine and the malformed numbers were already rejected. What remains wrong
> is exactly what looks right.

**The end-to-end crossover is a property of your fields.** The pipeline stays
ahead up to a CER of **0.96% for an 8-character field** and only **0.13% for a
60-character field** — a factor of **7.4**, against
{{eq:crossover-inverse-length}}'s predicted $60/8 = 7.5$.

**So "is OCR obsolete" has no answer.** The same document can want a pipeline for
its reference number and a vision model for its free-text description, and
{{eq:ocr-crossover}} says where the line falls once you supply two numbers you can
measure: your field lengths, and your CER *on your documents* rather than on a
vendor benchmark.

**And the accuracy table omits the deciding factor.** The pipeline leaves a text
layer — indexable, citable, diffable, searchable, and reprocessable when the
extraction logic changes. The end-to-end model leaves an answer, and a second
question costs a second full inference ({{eq:artefact-economics}}).

**They are not substitutes even at equal accuracy** ({{eq:different-artefacts}}).
For one known question, end-to-end is simpler. For a searchable auditable corpus —
what most document systems actually are — the intermediate text *is* the product.

## 10. Production Considerations

**Convert quoted CER into field accuracy before believing it**
({{eq:effective-field-error}}). One multiplication, and it reframes most vendor
conversations.

**Measure CER on your own documents.** Benchmarks are cleaner than production
scans, systematically.

**Restrict the output alphabet per field.** A numeric field with a digits-only
decoder cannot produce the digit/letter confusions that are two thirds of its
error mass.

**Add checksums and cross-field arithmetic** for the silent third. Format
validation does not reach it.

**Never resize away small text.** {{ch:mm-vit}}'s
{{eq:patch-compression}} and {{ch:mm-segmentation}}'s thin-structure erasure both
apply, and both are upstream of anything you can fix later.

**Keep the text layer even if you also run a VLM.** It is the artefact
({{eq:different-artefacts}}), and regenerating it later means re-reading every
page.

**Use confidence as a ranking for a review queue, not as a calibrated
probability.**

**Route handwriting separately.** It will not meet a threshold set on printed
text.

**Split the architecture by field length** ({{eq:ocr-crossover}}) rather than
choosing one for the whole document.

## 11. Common Mistakes

**Quoting CER as though it were the system's accuracy.**

**Believing a format check validates a number.** It catches two thirds.

**Spot-checking a financial pipeline by reading names.**

**Downsampling the page before OCR.**

**Choosing one architecture for a document with fields of very different
lengths.**

**Discarding the text layer** once a VLM is in the pipeline.

**Using a fixed confidence threshold** from a tutorial rather than a measured
operating point.

## 12. Failure Modes

**Silent numeric corruption.** Symptom: none — downstream totals are wrong and
everything validates. Detect with cross-field arithmetic and check digits.

**Long-field collapse.** Symptom: short fields extract well, addresses and
descriptions do not. Cause: {{eq:field-error-amplification}}.

**Small-print erasure.** Symptom: footnotes, disclaimers and dense tables missing
entirely. Cause: resolution, upstream of recognition.

**Handwriting cliff.** Symptom: accuracy fine on printed fields, poor on the
handwritten ones of the same form.

**Reading-order scramble.** Symptom: extracted text is fluent locally and
nonsensical across columns. {{ch:rag-ingestion}}'s
{{eq:naive-reading-order}}.

**Confidence miscalibration.** Symptom: a threshold tuned on one document type
sends the wrong volume to review on another.

**Artefact loss.** Symptom: a new extraction requirement means re-processing every
page from images, because nobody kept the text.

## 13. Alternatives

| Approach | Trades away | When it wins |
|---|---|---|
| classical OCR pipeline | long-field accuracy | short structured fields, and when you need the text layer |
| layout-aware encoder ({{cite:huang2022layoutlmv3}}) | needs OCR first | forms and receipts where 2D position is the meaning |
| OCR-free end-to-end ({{cite:kim2022donut}}) | the text artefact | one known extraction target, fixed schema |
| general VLM ({{cite:wang2024qwen2vl}}) | cost, throughput, the artefact | varied ad-hoc questions, no training budget |
| visual retrieval ({{cite:faysse2025colpali}}) | text availability | retrieval over scanned corpora |
| human review | throughput, cost | the tail that {{eq:required-cer}} says is unreachable |

**The last row is not an admission of failure.** {{eq:required-cer}} shows the CER
needed for 99% field accuracy on long fields is below what general OCR achieves,
so a review step is the *correct* design rather than a stopgap — and the
engineering question is how to route as little as possible into it.

## 14. Evaluation

**Report field-level accuracy, not CER.** CER is a component metric; nothing
consumes it.

**Report per-field-type accuracy** and separate numeric from alphabetic —
{{eq:silent-numeric-error}} makes them different problems.

**Measure the silent-error rate explicitly** by comparing against ground truth on
numeric fields. It is invisible to every other check.

**Evaluate on your document distribution**, including the degraded scans and the
handwritten fields, at their real proportions.

**Report the review-queue volume** at your chosen confidence threshold. It is the
operational cost, and it is not in any accuracy number.

**For end-to-end models, evaluate per question type**, since there is no
intermediate to inspect when one fails.

## 15. Advanced Concepts

**Constrained decoding as error prevention.** {{maturity:MATURE}} Restricting the
output alphabet or applying a field grammar
({{ch:llm-structured-output}}) removes whole classes of confusion *before* they
happen, which is strictly better than detecting them after.

**Cross-field consistency as free supervision.** {{maturity:MATURE}} Documents are
redundant — line items sum to totals, dates are consistent, check digits exist.
That redundancy is the only handle on the silent third, and it is usually left
unused.

**The text layer as the durable asset.** {{maturity:ESTABLISHED}}
{{eq:artefact-economics}} says the pipeline amortises across questions. In a
corpus queried many times over years, the text layer outlives every model that
produced or consumed it — which is an argument for producing it even where
end-to-end accuracy is higher today.

**OCR-free is a claim about the interface, not only accuracy.**
{{maturity:EMERGING}} {{cite:kim2022donut}}'s contribution is that the
intermediate representation is *optional*. Whether it is *desirable* is
{{eq:different-artefacts}}, and that is a systems question rather than a
modelling one.

**Document understanding as the VLM benchmark that matters.**
{{maturity:EMERGING}} {{cite:mathew2021docvqa}} found the hard questions are the
ones needing document *structure*, not reading — which is why
{{ch:mm-layout}} exists and why document VQA became the axis general VLMs are
judged on.

**And the error model is the part that does not transfer.**
{{maturity:EMERGING}} Every number in this chapter is a rate over a corpus, and an
OCR error is not a random draw: it concentrates on the documents that were already
hard — poor scans, unusual layouts, minority scripts — so a corpus-level accuracy
figure describes the easy majority and says almost nothing about the subset that
generates support tickets. **A per-document-class breakdown is the measurement
worth having**, and it is the one that is almost never published, because it
requires labelling the classes before the errors are known.

## 16. Connection to Previous Chapters

{{ch:rag-ingestion}}'s {{eq:ingestion-loss}} is what this chapter supplies the
character-level mechanics for, and its reading-order problem is one of the
pipeline stages here. {{ch:mm-segmentation}}'s thin-structure erasure and
{{ch:mm-vit}}'s {{eq:patch-compression}} are both upstream causes of small-print
failure. {{ch:rag-failures}}'s {{eq:stage-cascade}} is
{{eq:pipeline-composition}} in a different domain, with the same
localise-before-optimising consequence. {{ch:emb-similarity}}'s calibration
warning applies to OCR confidence scores. Forward: {{ch:mm-layout}} takes the
structural half that this chapter's character-level view cannot address, and
{{ch:mm-vlms}} is where end-to-end reading becomes a general capability.

## 17. Exercises

1. Using {{eq:field-error-amplification}}, compute field accuracy at CER 0.005 for
   lengths 10, 30 and 100. Check against the listing.
2. Invert {{eq:required-cer}} for 99.9% accuracy on a 25-character field. Is that
   CER achievable?
3. In `field-error-amplification`, add a field type whose alphabet is digits only
   *and* whose corruptions stay in-alphabet. What is the silent share?
4. Modify the same listing so errors are correlated — a corrupted character makes
   the next one twice as likely to corrupt. How does the field-accuracy
   distribution change, and does the mean move?
5. Derive {{eq:crossover-inverse-length}} and verify the 7.4 ratio measured.
6. In `pipeline-versus-end-to-end`, set `P_LAYOUT` to 0.85 (a hard layout). How far
   does the crossover move, and which stage now dominates?
7. Using {{eq:artefact-economics}}, find the number of questions per document at
   which the pipeline is cheaper, for plausible costs.
8. Take a document type you handle. Measure CER on twenty pages, list your field
   lengths, and compute which fields belong on which side of
   {{eq:ocr-crossover}}.

## 18. Interview Questions

1. Why is character error rate the wrong metric for an extraction pipeline?
2. Convert 99% CER into field accuracy for a 40-character field.
3. Which OCR errors can a format check catch, and which cannot?
4. How would you validate an extracted account number?
5. When would you use an end-to-end document model instead of OCR?
6. What does a pipeline give you that an end-to-end model does not?
7. Your financial extraction looks fine on spot-checks and the totals are wrong.
   Diagnose.
8. What CER would you need for 99% accuracy on a 25-character field?
9. How should OCR confidence scores be used?
10. Why does handwriting break a system that works on printed forms?

## 19. Research Questions

1. {{eq:field-error-amplification}} assumes independent errors and real errors are
   correlated. What is the right model, and does it change the review-queue
   design?
2. The silent third resists format validation. Is there a learned detector of
   "this digit string is implausible in this context" that beats check digits?
3. {{eq:ocr-crossover}} treats end-to-end accuracy as length-independent. Is it,
   or does it degrade with field length too — and if so, more or less steeply?
4. Constrained decoding removes cross-alphabet confusions. How much of the
   remaining error is addressable by field-specific grammars?
5. {{eq:different-artefacts}} argues the text layer has value beyond accuracy. Can
   an end-to-end model be made to emit a *grounded* text layer as a by-product, and
   at what cost?

## 20. Chapter Summary

**OCR accuracy is quoted per character and nothing downstream consumes
characters.** {{eq:field-error-amplification}} converts: at the "99% accurate"
rate, a 6-character code survives **0.946**, a 14-digit account number **0.863**,
and a 200-character paragraph **0.138**. The predictions match measurement within
a few thousandths, because it is arithmetic rather than an empirical finding.

**Which means "our OCR is 99% accurate" and "our pipeline is unusable" are
routinely both true**, and {{eq:required-cer}} shows why the argument goes nowhere:
99% field accuracy on a 40-character field needs a CER of $2.5 \times 10^{-4}$,
which is not available. **The human review step is the correct design, not a
stopgap.**

**Failures split into visible and silent, and the split is more useful than the
usual warning.** Two thirds of numeric corruptions cross the digit/letter boundary
and a format check catches them free. **The remaining third parse cleanly as a
different, wrong number** and need redundancy — check digits, cross-field
arithmetic, independent agreement. Hence the pathology: a financial pipeline looks
excellent on a spot-check because the names are visibly fine and the malformed
numbers were already rejected; **what remains wrong is exactly what looks right.**

**The end-to-end crossover is a property of your fields, not of the
technologies.** The pipeline holds to a CER of **0.96% at 8 characters** and
**0.13% at 60** — a factor of 7.4 against {{eq:crossover-inverse-length}}'s
predicted 7.5. The same document can want different architectures for different
fields.

**And the accuracy comparison omits what usually decides it.** The pipeline leaves
a **text layer** — indexable, citable, diffable, searchable, reprocessable — and
the end-to-end model leaves an answer, with a second question costing a second
full inference ({{eq:artefact-economics}}). **They produce different artefacts,
not merely different accuracies** ({{eq:different-artefacts}}), which is why the
architecture most systems converge on is the hybrid: keep the text layer, and use
a vision model for the fields {{eq:field-error-amplification}} tells you in
advance it will handle badly.

## 21. Further Reading

{{cite:kim2022donut}} for the OCR-free argument — the contribution is that the
intermediate representation is optional, which is a different claim from "better".
{{cite:huang2022layoutlmv3}} for treating text, 2D position and image patches
jointly, developed in {{ch:mm-layout}}.
{{cite:mathew2021docvqa}} for the finding that the hard document questions are
structural rather than textual.
{{cite:wang2024qwen2vl}} for dynamic resolution, which is what made general VLMs
able to read dense pages at all.
{{cite:faysse2025colpali}} for the retrieval version of the same argument, and
{{ch:rag-structured}} for where it lands in a RAG system.
{{cite:ronneberger2015unet}} for text detection as segmentation, and
{{ch:mm-segmentation}} for why thin strokes are the hard case.

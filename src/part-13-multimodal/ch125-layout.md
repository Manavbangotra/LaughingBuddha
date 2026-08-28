---
id: mm-layout
number: 125
part: XIII
tier: full
status: draft
requires: [mm-ocr, mm-vit, rag-ingestion, rag-structured, math-derivatives]
provides: [two-dimensional-position, layout-aware-encoding, table-structure-recovery,
           chart-value-error, derived-quantity-amplification, chart-to-table,
           form-key-value-association]
citations: [huang2022layoutlmv3, masry2022chartqa, mathew2021docvqa,
            kim2022donut, wang2024qwen2vl, li2023bird]
---

## 1. Learning Objectives

By the end of this chapter you will be able to explain why a form's meaning lives
in **2D geometry** and demonstrate that linearising to a reading order destroys
it — not degrades it, destroys it; describe what a layout-aware encoder adds and
why 2D position must be an *input* rather than a post-processing heuristic; state
why table structure recovery is the hardest and least glamorous problem in
document AI; derive **derived-quantity amplification** and use it to explain why a
chart-reading model gets values right and differences wrong; and choose
chart-to-table extraction over direct question answering for the right reason.

## 2. Why This Matters

{{ch:mm-ocr}} got the characters out. This chapter is about everything that was
carried by *where the characters were*, which for forms, tables and charts is most
of the meaning.

**A form is a 2D object and a token sequence is a 1D one**, and the conversion is
lossy in a specific, measurable way. {{sec:9-practical-example}} builds forms
where every character is recognised perfectly and measures label–value association
under both representations: on a single-column stacked form, reading order scores
**1.000**; on the two-column layout most real forms use, it scores exactly
**0.500** while the same task from 2D coordinates scores **1.000**.

**The 0.500 is not noise — it is the mechanism, exactly.** Both columns share
baselines, so band-then-left-to-right puts both labels in one band and both values
in the next: `label_A, label_B, value_A, value_B`. The next value after `label_A`
is right; the next value after `label_B` is the *other column's*. Half correct by
luck, at any number of fields.

The second half of the chapter is about charts, where perception and arithmetic
multiply. {{sec:9-practical-example}} finds a one-pixel reading error gives a
**0.0046** relative error on a single value and **0.089** on a *difference* of two
close values — a factor of **19** from the same reading, because
{{eq:derived-quantity-amplification}} divides by a small number.

**"What is X" is well-conditioned and "how much bigger is X than Y" is not, on the
same chart from the same measurement.**

{{maturity:MATURE}} Layout-aware encoders ({{cite:huang2022layoutlmv3}}).
{{maturity:EMERGING}} General VLMs handling layout implicitly
({{cite:wang2024qwen2vl}}), which works and removes the inspectable intermediate.

## 3. Prerequisites

{{ch:mm-ocr}} for the tokens and their coordinates; {{ch:rag-ingestion}} for
{{eq:naive-reading-order}} and {{eq:table-recoverability}}, which this chapter
makes precise; {{ch:mm-vit}} for {{eq:patch-compression}}, which decides whether
small table text is legible; {{ch:rag-structured}} for what to do once a table is
recovered; {{ch:math-derivatives}} for error propagation.

## 4. Intuitive Explanation

### In a form, position is the meaning

```text
   Invoice Number            Order Date
   INV-2291                  2026-03-04
```

Which value belongs to which label? A human answers instantly, from geometry:
`INV-2291` is *below* `Invoice Number`. There is no linguistic cue — the strings
have no relationship a language model could infer, and swapping them produces
text that reads identically well.

**Now flatten it.** Reading order is `Invoice Number`, `Order Date`, `INV-2291`,
`2026-03-04`. The association is gone, and worse, a plausible-looking wrong
association is available: `Order Date → INV-2291` sits right there as an adjacent
pair.

{{sec:9-practical-example}} measures exactly this at **0.500**, and the number is
structural rather than empirical.

**Note what kind of failure this is.** Not OCR — every character was read
correctly. Not the model — no model has run. It is a **representation failure that
happened during flattening**, before anything intelligent was applied, and nothing
downstream can undo it because the information is no longer present.

### What layout-aware encoding does

{{cite:huang2022layoutlmv3}}'s answer is simple once stated: **stop throwing the
coordinates away.** Each token embedding gets its bounding box added, so the model
sees $(\text{text}, x, y, w, h)$ rather than $(\text{text}, \text{position in
sequence})$.

That is a change of *input*, not of post-processing, and the distinction matters.
A heuristic that reconstructs associations after linearisation is guessing from a
lossy encoding; a model given coordinates is reading the thing itself.

### Tables are harder than they look and get less attention than they deserve

A table is a 2D grid whose semantics come from **alignment**: this cell means what
it means because of the row and column it is in. Recovering that from a page
requires:

1. finding the table's extent,
2. finding row and column boundaries — often with **no ruling lines at all**,
3. handling merged cells, multi-row headers, and cells that wrap,
4. deciding which rows are headers.

Step 2 is where it breaks. A table with no lines is defined by *whitespace
alignment*, and whitespace alignment is exactly what a slightly skewed scan, a
proportional font, or a wrapped cell destroys.

**And a table that is nearly recovered is worse than one that failed**, because a
misaligned cell produces a well-formed row with the wrong values in it —
{{ch:mm-ocr}}'s silent-error problem at the structural level.

### Charts multiply two error sources

A chart encodes numbers as pixels. Reading one is a **measurement**, so it has an
error bar, and {{cite:masry2022chartqa}}'s questions then do arithmetic on the
result.

The measurement is good: half a per cent on a single value. The arithmetic is
where it goes wrong, and only for particular questions:

> **A difference of two close values is ill-conditioned.** The absolute error
> stays the same and the denominator shrinks, so the relative error explodes.
> **A ratio is not**, because its denominator is a bar height rather than a small
> difference.

{{sec:9-practical-example}} measures value error 0.0046, ratio error 0.0066, and
difference error **0.089** — the ratio sits right next to the single value, and
only the difference blows up. That is a property of the *arithmetic*, not of the
chart or the model.

**Which explains the reported behaviour of chart models**: they describe trends
correctly, give plausible values, and get comparisons of similar bars wrong — in
the same confident tone.

### The axis-truncation surprise

A truncated axis (80–100 instead of 0–100) is the classic misleading-chart
device. It also makes the chart **more precise to read** — the same pixels span a
smaller value range, so each pixel is worth less. Measured: **0.0004** relative
error against **0.0046**, a factor of ten.

**It helps a model reading pixels and misleads a human reading impressions,
simultaneously.** Two different consumers, and the chart cannot serve both.

## 5. Formal Explanation

### 5.1 What linearisation discards

A page is a set of tokens with geometry, $\{(t_i, x_i, y_i, w_i, h_i)\}$.
Linearisation is a map

$$ \mathcal{L}: \{(t_i, x_i, y_i, \dots)\} \longmapsto (t_{\sigma(1)}, \dots, t_{\sigma(n)}) $$ (eq:reading-order-loses-2d)

which is **not injective** — many geometries produce the same sequence. So any
function of the geometry that differs between two pages mapping to the same
sequence is *unrecoverable* from the sequence.

Key–value association in a form is such a function, which is why
{{sec:9-practical-example}}'s 1D method cannot exceed chance-plus-luck on the
two-column layout. **This is an information argument, not a modelling one.**

### 5.2 The two-column failure, exactly

For a two-column form with shared baselines, band-then-left-to-right gives, per
row-pair:

$$ \text{seq} = (\ell_A,\, \ell_B,\, v_A,\, v_B) $$

The rule "associate a label with the next value token" yields $\ell_A \to v_A$
(correct) and $\ell_B \to v_A$ (incorrect), so

$$ \text{accuracy} = \tfrac{1}{2} \quad \text{independent of the number of fields} $$ (eq:two-column-half)

{{eq:two-column-half}} is why the measured 4-pair and 10-pair rows are identical.

### 5.3 Layout-aware encoding

$$ e_i = \underbrace{E_{\text{tok}}(t_i)}_{\text{what}} + \underbrace{E_x(x_i) + E_y(y_i) + E_w(w_i) + E_h(h_i)}_{\text{where}} $$ (eq:2d-position-embedding)

Compare {{ch:mm-vit}}'s {{eq:patch-embedding}}: both add position to content, and
the difference is that a document's positions are *continuous and meaningful*
rather than a fixed grid index.

**{{eq:2d-position-embedding}} makes the geometry an input**, so
{{eq:reading-order-loses-2d}}'s non-injectivity never applies — nothing was
collapsed.

### 5.4 Table structure

A recovered table is a function from page tokens to cells:

$$ \tau: t_i \longmapsto (r, c), \qquad \text{meaning}(t_i) = f\big(t_i,\, \text{header}(c),\, \text{header}(r)\big) $$ (eq:table-cell-addressing)

so a token's meaning depends on **two other tokens found by geometry**. An error in
$\tau$ does not corrupt one value; it re-labels it:

$$ \tau(t_i) = (r, c') \text{ instead of } (r, c) \;\Longrightarrow\; \text{a valid-looking value under the wrong header} $$ (eq:table-misalignment-silent)

**{{eq:table-misalignment-silent}} is why table extraction errors are dangerous
rather than merely annoying** — and it is {{ch:mm-ocr}}'s silent-numeric-error
argument one level up.

### 5.5 Chart reading as measurement

For an axis spanning $[y_{\min}, y_{\max}]$ over $P$ pixels, a value read at pixel
$p$ with error $\delta$ pixels:

$$ \hat{v} = y_{\min} + \frac{p + \delta}{P}(y_{\max} - y_{\min}), \qquad \sigma_v = \frac{\delta}{P}(y_{\max} - y_{\min}) $$ (eq:chart-value-error)

The **absolute** error depends on the axis span; the relative error on a value $v$
is $\sigma_v / v$. Truncating the axis shrinks the span, so it shrinks
$\sigma_v$ — measured, a factor of ten.

### 5.6 Derived-quantity amplification

For $d = v_1 - v_2$ with independent reading errors:

$$ \sigma_d = \sqrt{2}\,\sigma_v, \qquad \frac{\sigma_d}{|d|} = \frac{\sqrt{2}\,\sigma_v}{|v_1 - v_2|} $$ (eq:derived-quantity-amplification)

**The numerator grew slightly and the denominator can be arbitrarily small.** For
bars differing by 4% of the axis span, $|d| \approx 0.04\,(y_{\max}-y_{\min})$,
and the relative error is $\sqrt{2}\delta / (0.04 P)$ — at $\delta = 1$, $P = 400$:
$0.088$, against the measurement's **0.089**.

For a ratio $r = v_1/v_2$:

$$ \frac{\sigma_r}{r} = \sqrt{\left(\frac{\sigma_v}{v_1}\right)^2 + \left(\frac{\sigma_v}{v_2}\right)^2} \approx \sqrt{2}\,\frac{\sigma_v}{v} $$ (eq:ratio-conditioning)

**The denominator is a bar height, not a difference**, so a ratio stays
well-conditioned. Measured: 0.0066 against the single value's 0.0046 — a factor of
$\sqrt{2}$, exactly as {{eq:ratio-conditioning}} predicts.

### 5.7 The comparison question

"Which is taller" depends only on $\text{sign}(\hat{d})$, so

$$ \Prob[\text{correct}] = \Phi\!\left(\frac{|d|}{\sqrt{2}\,\sigma_v}\right) $$ (eq:comparison-accuracy)

which depends only on the *pixel* geometry. **Axis truncation does not change it**
— and the measurement confirms this exactly: 0.972 for both axes at one pixel,
0.918 for both at three.

## 6. Mathematical Foundation

### 6.1 The amplification factor, worked

At $P = 400$ px, $\delta = 1$ px, axis $0$–$100$, bars near $v = 60$ differing by
$\Delta = 4$:

$$ \sigma_v = \frac{1}{400}\times 100 = 0.25, \qquad \frac{\sigma_v}{v} = \frac{0.25}{60} = 0.0042 $$

against a measured **0.0046** (the sample averages over bar heights, so the mean
$1/v$ exceeds $1/\bar{v}$).

$$ \frac{\sigma_d}{|d|} = \frac{\sqrt{2}\times 0.25}{4} = 0.088 $$

against a measured **0.089**. And the amplification is the ratio of the two:

$$ \frac{\sigma_d/|d|}{\sigma_v/v} = \sqrt{2}\,\frac{v}{|d|} = \sqrt{2}\times\frac{60}{4} \approx 21 $$ (eq:amplification-factor)

against a measured factor of **19**.

**{{eq:amplification-factor}} is the number to carry**: the amplification is
$\sqrt 2$ times the ratio of the value to the difference. Bars within 5% of each
other amplify by roughly 30.

### 6.2 When is a comparison trustworthy?

Inverting {{eq:comparison-accuracy}} for 99% accuracy needs
$|d| / (\sqrt2 \sigma_v) \ge 2.33$:

$$ |d| \ge 2.33\sqrt{2}\,\frac{\delta}{P}(y_{\max}-y_{\min}) = 3.3\,\frac{\delta}{P}(y_{\max}-y_{\min}) $$ (eq:trustworthy-comparison)

At $\delta = 1$, $P = 400$: the bars must differ by **0.8% of the axis span**. At
$\delta = 3$: **2.5%**.

**That is a usable rule.** Look at the chart; if the two bars differ by less than
a few per cent of the axis height, the model's confident comparison is a coin
weighted only slightly.

> **MATH NOTE:** {{eq:comparison-accuracy}} assumes independent Gaussian reading
> errors, which is optimistic in one specific way: a model reading a chart may have
> *correlated* errors between two bars — a systematic misreading of the axis
> affects both equally. Correlation would help the difference (common error
> cancels) and not change the ratio much. So the measured amplification is a
> worst-case within this model, and the direction of the assumption is stated
> rather than hidden.

### 6.3 Why tables resist the same treatment

{{eq:table-cell-addressing}} makes a cell's meaning depend on its headers, so
table extraction accuracy is a *joint* property:

$$ \Prob[\text{row correct}] = \prod_{c} \Prob[\tau(t_c) = (r,c)] $$ (eq:table-row-accuracy)

which is {{ch:mm-ocr}}'s {{eq:field-error-amplification}} with columns instead of
characters. **A 12-column table needs per-cell alignment accuracy of 0.999 to get
99% of rows right**, and a nearly-aligned table produces
{{eq:table-misalignment-silent}}'s valid-looking wrong rows.

## 7. Internal Mechanics

```mermaid {#fig:layout-paths caption="Three ways to consume a page, and what each keeps. The top path discards geometry at the linearise step (eq:reading-order-loses-2d) and cannot recover it. The middle path keeps coordinates as model input (eq:2d-position-embedding). The bottom path never separates content from layout, and never produces an inspectable intermediate."}
flowchart TB
    P["page image"] --> O["OCR: tokens + boxes"]
    O --> L["linearise to reading order"]
    L --> T["text sequence"]
    T --> M1["language model"]
    O --> LA["keep (token, x, y, w, h)"]
    LA --> M2["layout-aware encoder<br/>(cite:huang2022layoutlmv3)"]
    P --> V["VLM reads pixels<br/>(cite:wang2024qwen2vl)"]
    L -.->|"NOT injective:<br/>geometry is gone"| T
    M1 --> A["answer"]
    M2 --> A
    V --> A
```

### 7.1 Getting a table out, in order of effort

| Approach | Works when | Fails when |
|---|---|---|
| ruling-line detection | the table has lines | most tables do not |
| whitespace projection ({{ch:rag-ingestion}}) | consistent alignment | skew, proportional fonts, wrapped cells |
| learned table structure model | trained on your table style | unusual headers, merged cells |
| VLM emits the table directly | anything it can read | no intermediate to check; silent misalignment |

**Add a validation step whichever you choose**, because
{{eq:table-misalignment-silent}} means the failure is not visible in the output.
The cheapest useful check is arithmetic: **do the columns that should sum, sum?**

### 7.2 Charts: read the data, not the picture

The ordering of preferences, and the reason for each:

1. **Retrieve the underlying data.** If the series exists in a table, a CSV, or a
   database, {{ch:rag-structured}} says query it. No measurement error at all.
2. **Chart-to-table extraction.** Have the model emit the series once, inspect it,
   then compute answers from the numbers. The reading error happens **once and
   visibly**.
3. **Direct question answering on the image.** The reading happens invisibly,
   separately, inside every question, and {{eq:derived-quantity-amplification}}
   applies unbounded.

**Option 2 is under-used and is usually the right default**, because it converts
an unmeasurable error into a measurable artefact — the same argument as
{{ch:mm-ocr}}'s text layer.

### 7.3 What a general VLM changed here

{{cite:wang2024qwen2vl}}'s dynamic resolution made it practical to feed a whole
page at a resolution where small text is legible, which removed the main reason
document work needed a specialist pipeline.

**What it did not remove is the artefact argument.** A VLM handles layout
implicitly and correctly, and it produces no cell map, no coordinates, and nothing
to validate — so {{eq:table-misalignment-silent}}'s silent failures become
undetectable rather than merely hard to detect.

## 8. Implementation

```python {tier=A name=two-dimensional-position}
"""In a form, position IS the meaning -- and reading order destroys it.

ch:rag-ingestion showed that flattening a page to a token sequence scrambles
multi-column text. This listing measures the sharper version of the same problem:
in a form, the ASSOCIATION between a label and its value is carried entirely by
2D geometry, and a 1D reading order does not encode geometry at all
(eq:reading-order-loses-2d).

Two layouts are compared. In a stacked form the value sits directly below its
label, so reading order happens to put them adjacent and 1D works by luck. In a
two-column form -- two independent label/value columns side by side, which is what
most real forms look like -- reading order interleaves the columns and the luck
runs out.
"""
import numpy as np

rng = np.random.default_rng(83)

N_FORM = 3000
PAGE_W, PAGE_H = 600.0, 800.0
LINE_H = 34.0


def stacked_form(n_pairs):
    """Label on one line, value on the next. One column."""
    toks = []
    y = 60.0
    for i in range(n_pairs):
        toks.append({"kind": "label", "id": i, "x": 70.0, "y": y})
        toks.append({"kind": "value", "id": i, "x": 70.0, "y": y + LINE_H})
        y += 2.4 * LINE_H
    return toks


def two_column_form(n_pairs):
    """Two side-by-side columns, each with the value BELOW its label and the
    two columns sharing baselines -- the layout most real forms use.

    This is the case where linearisation breaks. Band-then-left-to-right puts
    both labels in one band and both values in the next, so the sequence reads
    label_A, label_B, value_A, value_B: the token after label_B is not its own
    value (eq:two-column-half).
    """
    toks = []
    per_col = (n_pairs + 1) // 2
    y0 = 60.0
    for k in range(per_col):
        y = y0 + k * 2.6 * LINE_H
        for c, x in ((0, 70.0), (1, 350.0)):
            i = c * per_col + k
            if i >= n_pairs:
                continue
            toks.append({"kind": "label", "id": i, "x": x, "y": y})
            toks.append({"kind": "value", "id": i, "x": x, "y": y + LINE_H})
    return toks


def jitter(toks):
    for t in toks:
        t["x"] += rng.normal(scale=3.0)
        t["y"] += rng.normal(scale=2.0)
    return toks


def reading_order(toks, band=18.0):
    """Group tokens into horizontal bands, then sort left-to-right within each --
    the standard 1D linearisation (ch:rag-ingestion, eq:naive-reading-order)."""
    ts = sorted(toks, key=lambda t: t["y"])
    out, cur, y0 = [], [], None
    for t in ts:
        if y0 is None or abs(t["y"] - y0) <= band:
            cur.append(t)
            y0 = t["y"] if y0 is None else y0
        else:
            out.extend(sorted(cur, key=lambda z: z["x"]))
            cur, y0 = [t], t["y"]
    out.extend(sorted(cur, key=lambda z: z["x"]))
    return out


def assoc_1d(toks):
    """Associate each label with the next VALUE token in reading order."""
    seq = reading_order(toks)
    ok = tot = 0
    for i, t in enumerate(seq):
        if t["kind"] != "label":
            continue
        tot += 1
        for u in seq[i + 1:]:
            if u["kind"] == "value":
                ok += int(u["id"] == t["id"])
                break
    return ok / max(tot, 1)


def assoc_2d(toks):
    """Associate each label with the nearest value to its RIGHT or BELOW,
    using actual page coordinates."""
    labels = [t for t in toks if t["kind"] == "label"]
    values = [t for t in toks if t["kind"] == "value"]
    ok = 0
    for t in labels:
        best, bd = None, 1e18
        for u in values:
            dx, dy = u["x"] - t["x"], u["y"] - t["y"]
            if dx < -20 or dy < -20:                 # not right of, not below
                continue
            # Anisotropic distance: same-line to the right is cheapest, then
            # directly below. This encodes how forms are actually read.
            d = (dx / 3.0) ** 2 + dy ** 2 if abs(dy) < 20 else dx ** 2 + (dy * 1.4) ** 2
            if d < bd:
                best, bd = u, d
        ok += int(best is not None and best["id"] == t["id"])
    return ok / max(len(labels), 1)


print(f"{N_FORM} synthetic forms, positions jittered\n")
print(f"{'layout':<22}{'pairs':>7}{'1D reading order':>20}{'2D coordinates':>18}")
print("-" * 67)

res = {}
for name, builder in (("stacked (1 column)", stacked_form),
                      ("two-column", two_column_form)):
    for n_pairs in (4, 10):
        a1 = a2 = 0.0
        for _ in range(N_FORM // 2):
            toks = jitter(builder(n_pairs))
            a1 += assoc_1d(toks)
            a2 += assoc_2d(toks)
        m = N_FORM // 2
        res[(name, n_pairs)] = (a1 / m, a2 / m)
        print(f"{name:<22}{n_pairs:>7}{a1 / m:>20.3f}{a2 / m:>18.3f}")

s = res[("stacked (1 column)", 10)]
t = res[("two-column", 10)]
print(f"""
The stacked rows are where 1D reading order looks fine: {s[0]:.3f} against the
2D method's {s[1]:.3f}. That is not because the sequence encodes the
relationship. It is because in a single-column stacked form the value happens to
be the very next token, so "next token in reading order" and "the value belonging
to this label" coincide. The 1D method is right for the wrong reason, and a
benchmark built from stacked forms would report it as solved.

The two-column rows are the same task on the layout most real forms use, and 1D
collapses to {t[0]:.3f} while the 2D method holds {t[1]:.3f}. The 0.500 is not
noise; it is the mechanism, exactly.

Both columns share baselines, so band-then-left-to-right puts BOTH labels in one
band and BOTH values in the next. The sequence reads label_A, label_B, value_A,
value_B. The next value token after label_A is value_A, which is right. The next
value token after label_B is also value_A, which is wrong. Half the pairs
associate correctly by luck and half do not, at any number of pairs -- which is
why the 4-pair and 10-pair rows are identical.

Note what this is not. It is not an OCR failure -- every token was recognised
perfectly here. It is not a model failure -- no model has run. It is a
REPRESENTATION failure that happened during flattening, before anything
intelligent was applied, and no downstream component can undo it because the
information is gone (eq:reading-order-loses-2d).

That is the argument for layout-aware models (cite:huang2022layoutlmv3): keep the
2D coordinates as an input rather than discarding them in favour of a sequence.
The token embedding gains x and y, and the model can learn that a value below a
label belongs to it, which is a fact about forms that no amount of language
modelling recovers from a scrambled sequence.

And it explains a specific reported failure. A document pipeline that scores well
on paragraph-shaped text and badly on forms is usually not worse at forms -- it is
using a representation that discarded what forms are made of.""")
```

The first listing is about structure that gets destroyed. The second is about
structure that survives and is then measured badly.

```python {tier=A name=chart-value-amplification}
"""Reading a chart: why the trend is easy and the numbers are not.

A chart encodes numbers as PIXELS, so reading one is a measurement, and every
measurement has an error bar. cite:masry2022chartqa's questions mostly ask for
arithmetic over values that must first be read off the plot, which makes chart
question answering a perception problem and a reasoning problem multiplied
together (eq:chart-value-error).

The interesting part is not the error on a single value -- that is small. It is
what happens when the answer is a DIFFERENCE or a RATIO of two read values, where
the errors do not cancel and the denominator can be small
(eq:derived-quantity-amplification).

This listing simulates reading bar heights to a fixed pixel precision and
measures the error in the value, the difference, the ratio, and in the plain
comparison "which bar is taller".
"""
import numpy as np

rng = np.random.default_rng(89)

N = 200000
PLOT_PX = 400.0                 # pixels of vertical plot area


def read_values(y_min, y_max, true_vals, pixel_err):
    """Convert values to pixels, add a reading error, convert back
    (eq:chart-value-error)."""
    span = y_max - y_min
    px = (true_vals - y_min) / span * PLOT_PX
    px_noisy = px + rng.normal(scale=pixel_err, size=px.shape)
    return y_min + px_noisy / PLOT_PX * span


print(f"plot area {PLOT_PX:.0f} px tall; values read to a given pixel precision\n")
print(f"{'axis':<26}{'px err':>8}{'value':>10}{'diff (med)':>13}"
      f"{'ratio':>10}{'which is taller':>18}")
print(f"{'':<26}{'':>8}{'rel err':>10}{'rel err':>13}{'rel err':>10}{'accuracy':>18}")
print("-" * 85)

rows = {}
for label, y_min, y_max in (("0 to 100 (zero-based)", 0.0, 100.0),
                            ("80 to 100 (truncated)", 80.0, 100.0)):
    for pixel_err in (1.0, 3.0):
        a = rng.uniform(y_min + 0.15 * (y_max - y_min),
                        y_max - 0.05 * (y_max - y_min), size=N)
        # The second bar is CLOSE to the first -- the case the question is
        # usually about, and the case where a difference is ill-conditioned.
        b = a + rng.normal(scale=0.04 * (y_max - y_min), size=N)
        b = np.clip(b, y_min + 0.05 * (y_max - y_min), y_max)

        ra = read_values(y_min, y_max, a, pixel_err)
        rb = read_values(y_min, y_max, b, pixel_err)

        v_err = float(np.mean(np.abs(ra - a) / np.abs(a)))
        d_true, d_read = a - b, ra - rb
        keep = np.abs(d_true) > 1e-9
        # MEDIAN, not mean: the relative error of a difference has a heavy
        # tail because |d_true| can be arbitrarily close to zero, and a mean
        # over that is dominated by a handful of near-ties.
        d_err = float(np.median(np.abs(d_read[keep] - d_true[keep])
                                / np.abs(d_true[keep])))
        r_err = float(np.mean(np.abs((ra / rb) - (a / b)) / np.abs(a / b)))
        cmp_ok = float(np.mean(np.sign(d_read) == np.sign(d_true)))

        rows[(label, pixel_err)] = (v_err, d_err, r_err, cmp_ok)
        print(f"{label:<26}{pixel_err:>8.0f}{v_err:>10.4f}{d_err:>13.3f}"
              f"{r_err:>10.4f}{cmp_ok:>18.3f}")

z1 = rows[("0 to 100 (zero-based)", 1.0)]
z3 = rows[("0 to 100 (zero-based)", 3.0)]
t1 = rows[("80 to 100 (truncated)", 1.0)]
print(f"""
Read the value column first, because it is the reassuring one. A one-pixel
reading error on a 400-pixel plot gives a relative error on a single value of
{z1[0]:.4f} -- half of one per cent. If the question is "roughly how big is the
third bar", perception is not the problem, and this is why a model looks
competent when asked to describe a chart.

Now the difference column, which is the same measurement asked a different
question. The median relative error jumps to {z1[1]:.3f} -- a factor of
{z1[1]/z1[0]:.0f} -- for one reason. The two bars are close, so the true
difference is small, and eq:derived-quantity-amplification says the relative
error of a difference is the absolute error divided by that small number. The
errors also do not cancel: two independent readings contribute independent errors
to their difference.

(Median rather than mean, deliberately. The relative error of a difference has a
heavy tail because the denominator can be arbitrarily close to zero, so a mean
over it is a statement about a handful of near-ties rather than about typical
behaviour.)

This is the most useful single fact about chart question answering. "What is the
value of X" is a well-conditioned question and "how much bigger is X than Y" is
an ill-conditioned one -- on the SAME chart, from the SAME reading. The model has
not become worse at perception between the two; the question has become worse at
tolerating perception error.

The ratio column is the control that confirms the diagnosis. A ratio of two close
values has a denominator that is NOT small -- it is one of the bar heights -- so
it stays well-conditioned at {z1[2]:.4f}, right alongside the single-value error.
Differences are ill-conditioned and ratios are not, which is a property of the
arithmetic rather than of the chart.

The last column turns this into something a user meets. At one pixel of error the
comparison "which bar is taller" is right {z1[3]:.3f} of the time; at three
pixels, {z3[3]:.3f}. Not disasters -- and not the near-certainty that the
confident phrasing of the answer will imply. A chart reader states which of two
close bars is larger in the same tone it uses for everything else.

Finally, compare the two axis blocks, where there is a genuine surprise and a
non-surprise. The surprise: a truncated axis makes the chart MORE precise to
read, {t1[0]:.4f} against {z1[0]:.4f}, a factor of {z1[0]/t1[0]:.0f}, because the
same pixels now span a smaller value range so each pixel is worth less. The
non-surprise: the comparison column is IDENTICAL across the two axes, because the
sign of a difference depends only on which bar is taller in pixels, and
truncation does not change pixel geometry (eq:comparison-accuracy).

So axis truncation helps a model reading pixels and misleads a human reading
impressions, simultaneously. Those are different consumers and the chart cannot
serve both.

The engineering conclusion is to stop asking the model to be a measuring
instrument. Where the underlying data exists -- a table, a CSV, the series behind
the plot -- retrieve it and compute the answer (ch:rag-structured). Where it does
not, prefer chart-to-table extraction over direct question answering, so the
reading step happens ONCE and is inspectable, instead of happening invisibly
inside every arithmetic question.""")
```

## 9. Practical Example

**Linearisation destroys forms, and by exactly half.** On a single-column stacked
form, 1D reading order scores **1.000** — and it is right for the wrong reason,
because the value happens to be the next token. On the two-column layout most real
forms use, it scores **0.500** while 2D coordinates score **1.000**.

**The 0.500 is the mechanism, not noise.** Both columns share baselines, so the
sequence is `label_A, label_B, value_A, value_B`: the next value after `label_A`
is correct, the next value after `label_B` is the other column's.
{{eq:two-column-half}} — and it is why the 4-pair and 10-pair rows are identical.

> **IMPORTANT:** This is not an OCR failure (every token was recognised) and not a
> model failure (no model ran). It is a **representation failure during
> flattening**, and {{eq:reading-order-loses-2d}}'s non-injectivity means nothing
> downstream can undo it. **A pipeline that scores well on paragraphs and badly on
> forms is usually not worse at forms — it is using a representation that
> discarded what forms are made of.**

**Charts: the measurement is fine and the arithmetic is not.** A one-pixel reading
error gives a relative error of **0.0046** on a single value — half a per cent,
which is why models look competent describing charts.

**The same reading gives 0.089 on a difference of two close bars** — a factor of
**19**. {{eq:derived-quantity-amplification}} predicted 0.088 and
{{eq:amplification-factor}} predicted a factor of 21.

**The ratio column confirms the diagnosis rather than merely illustrating it.** A
ratio of the same two values stays at **0.0066**, right beside the single-value
error, exactly $\sqrt2$ times it as {{eq:ratio-conditioning}} says. **Differences
are ill-conditioned; ratios are not.** That is a property of the arithmetic, not
of the chart or the model.

**And the user-facing form**: "which bar is taller" is right **0.972** of the time
at one pixel of error and **0.918** at three — not disasters, and not the
near-certainty the answer's phrasing implies. {{eq:trustworthy-comparison}} gives
the rule: **bars must differ by 0.8% of the axis span for a 99%-reliable
comparison** at one pixel, 2.5% at three.

**The axis-truncation result contains one surprise and one non-surprise.**
Truncating 0–100 to 80–100 makes the chart **ten times more precise to read**
(0.0004 against 0.0046) because the same pixels span a smaller range. And the
comparison accuracy is **identical** across the two axes — 0.972 and 0.972 —
because {{eq:comparison-accuracy}} depends only on pixel geometry, which
truncation does not change.

**So a truncated axis helps a model reading pixels and misleads a human reading
impressions, at the same time.** Different consumers; the chart cannot serve both.

## 10. Production Considerations

**Keep bounding boxes through the whole pipeline.** Once linearised, the geometry
is gone ({{eq:reading-order-loses-2d}}) and no later stage recovers it.

**Use a layout-aware encoder for forms**, or a VLM — but do not use a linearised
text model and then tune it.

**Validate tables arithmetically.** {{eq:table-misalignment-silent}} means a
misaligned table looks correct. Check that columns which should sum, sum.

**Compute {{eq:table-row-accuracy}} for your column count.** A 12-column table
needs 0.999 per-cell accuracy for 99% of rows, which is usually the number that
justifies a review step.

**Never ask a model to compute a difference from a chart.** Extract the series
first ({{sec:7-internal-mechanics}}'s option 2), then do arithmetic on numbers.

**Retrieve the underlying data when it exists.** {{ch:rag-structured}} — a chart
is a rendering of a table that probably still exists somewhere.

**Flag close comparisons.** {{eq:trustworthy-comparison}} gives the threshold;
below it, the answer should be hedged rather than stated.

**Log the resolution the page was read at.** {{ch:mm-vit}}'s
{{eq:patch-compression}} decides whether the table's small text was ever legible.

## 11. Common Mistakes

**Linearising a form and then blaming the model.**

**Treating table extraction as solved because it works on ruled tables.**

**Trusting a table that was nearly recovered** —
{{eq:table-misalignment-silent}}.

**Asking a VLM for differences and percentages read off a chart.**

**Assuming a truncated axis is bad for machines** — it is better, and worse for
humans.

**Reporting chart QA accuracy as one number** across value, comparison and
arithmetic questions, which have very different conditioning.

**Discarding the coordinates once a VLM handles layout implicitly**, losing the
ability to validate anything.

## 12. Failure Modes

**Key–value misassociation.** Symptom: extracted fields are individually correct
and attached to the wrong labels. Cause: {{eq:reading-order-loses-2d}}. Detect by
auditing associations, not values.

**Silent table misalignment.** Symptom: a well-formed table with values under the
wrong headers. Cause: {{eq:table-misalignment-silent}}. Detect arithmetically.

**Confident wrong comparison.** Symptom: "A is larger than B" stated firmly for
near-equal bars. Cause: {{eq:comparison-accuracy}} near its coin-flip region.

**Difference arithmetic garbage.** Symptom: percentage changes and gaps that are
wildly wrong while the underlying values look plausible. Cause:
{{eq:derived-quantity-amplification}}.

**Merged-cell corruption.** Symptom: one row shifts by a column from a merged
header onward, and every subsequent row inherits the shift.

**Header misidentification.** Symptom: the first data row is treated as a header,
so every value is labelled by another value.

**Small-table erasure.** Symptom: dense tables missing entirely. Cause:
resolution, upstream ({{ch:mm-ocr}}).

## 13. Alternatives

| Approach | Trades away | When it wins |
|---|---|---|
| linearised text + LLM | all geometry | flowing prose only |
| layout-aware encoder ({{cite:huang2022layoutlmv3}}) | needs OCR + boxes | forms and receipts |
| OCR-free structured output ({{cite:kim2022donut}}) | the intermediate | fixed schema, known target |
| general VLM ({{cite:wang2024qwen2vl}}) | inspectability, cost | varied questions, complex layout |
| chart-to-table then compute | one extra step | any chart arithmetic — should be the default |
| retrieve the source data ({{cite:li2023bird}}) | needs the data to exist | always preferable when possible |

**The last two rows are the chapter's practical recommendation** and both are
under-used, because reading the picture feels more capable than looking up the
number.

## 14. Evaluation

**Evaluate key–value association separately from value extraction.** They fail for
different reasons and the aggregate hides the association failure entirely.

**Evaluate tables per cell and per row** ({{eq:table-row-accuracy}}), and report
both — per-cell looks fine when per-row does not.

**Break chart QA down by question type**: value lookup, comparison, difference,
ratio, trend. Their conditioning differs by an order of magnitude, so one number
is uninformative.

**Report the axis span and plot resolution** for chart evaluations, since
{{eq:chart-value-error}} depends on both.

**Include unruled tables, merged cells and multi-row headers** at their real
frequency. Benchmarks over-represent clean tables.

**Audit associations by sampling**, not by checking that values look plausible —
misassociated values are plausible by construction.

## 15. Advanced Concepts

**2D position as a first-class modality.** {{maturity:MATURE}}
{{eq:2d-position-embedding}} is the same move as {{ch:mm-vit}}'s patch positions,
and the general principle is that **any structure carried by geometry must be an
input**, because linearisation is not invertible.

**Table structure as graph prediction.** {{maturity:EMERGING}} Predicting
cell adjacency — which cells share a row, which share a column — is more robust
than predicting a grid, because it degrades gracefully on merged cells rather than
shifting every subsequent row.

**Chart derendering.** {{maturity:EMERGING}} Recovering the underlying data series
from a plot turns {{eq:derived-quantity-amplification}} into a one-time,
inspectable error instead of a per-question invisible one. The right default and
still not the common one.

**Conditioning as a general lens.** {{maturity:ESTABLISHED}}
{{eq:derived-quantity-amplification}} is elementary numerical analysis applied to a
perception system, and the lesson generalises: **when a model's output is fed into
arithmetic, ask what the arithmetic does to the error before blaming the model.**
Differences of close quantities are the classic ill-conditioned case, and they are
everywhere in analytics questions.

**Documents as the VLM frontier.** {{maturity:EMERGING}}
{{cite:mathew2021docvqa}} found the hard questions are structural, and
{{cite:masry2022chartqa}} found the hard ones are arithmetic over perceived
values. Both are cases where the *language* half is easy and the interface between
perception and reasoning is where the errors live.

## 16. Connection to Previous Chapters

{{ch:rag-ingestion}}'s {{eq:naive-reading-order}} and
{{eq:table-recoverability}} are what this chapter makes precise and measures.
{{ch:mm-ocr}}'s {{eq:field-error-amplification}} reappears as
{{eq:table-row-accuracy}} with columns replacing characters, and its
silent-numeric-error argument reappears as
{{eq:table-misalignment-silent}} one level up the structure.
{{ch:mm-vit}}'s {{eq:patch-embedding}} is {{eq:2d-position-embedding}} on a fixed
grid rather than continuous coordinates, and its
{{eq:patch-compression}} decides legibility. {{ch:rag-structured}}'s argument —
query the data rather than embedding a rendering of it — is the correct answer to
most chart questions. Forward: {{ch:mm-vlms}} is where layout handling becomes
implicit, with the artefact consequences this chapter names.

## 17. Exercises

1. Prove {{eq:two-column-half}} and state the layout property that makes the
   accuracy exactly one half.
2. In `two-dimensional-position`, add a three-column form. Does 1D accuracy fall
   to 1/3?
3. Modify the same listing so the value is to the RIGHT of its label in a
   two-column layout. Does 1D recover, and why?
4. Derive {{eq:derived-quantity-amplification}} and {{eq:ratio-conditioning}} and
   explain the $\sqrt2$ in each.
5. Verify {{eq:amplification-factor}}'s predicted 21 against the measured 19, and
   account for the gap.
6. In `chart-value-amplification`, make the two bars differ by 20% instead of 4%.
   How does the difference error change, and does it match
   {{eq:amplification-factor}}?
7. Use {{eq:trustworthy-comparison}} to compute the minimum bar gap for 95%
   reliability at 2 pixels of error.
8. Take a table extractor you use. Compute per-cell and per-row accuracy on twenty
   tables, and check {{eq:table-row-accuracy}} against the measurement.

## 18. Interview Questions

1. Why does flattening a form to text lose information that cannot be recovered?
2. What exactly does a layout-aware encoder add?
3. Why is table structure recovery harder than table detection?
4. Why is a nearly-correct table worse than a failed one?
5. A chart model gives good values and bad differences. Explain.
6. Why is a ratio better conditioned than a difference?
7. Does axis truncation help or hurt a model reading a chart?
8. When should you extract a chart to a table rather than answer directly?
9. Your form extractor has 98% field accuracy and users complain about wrong
   values. What do you check?
10. How would you validate an extracted table?

## 19. Research Questions

1. {{eq:reading-order-loses-2d}} is non-injective. Is there a linearisation that
   *is* injective for the layouts that occur in practice, and at what token cost?
2. {{eq:comparison-accuracy}} assumes independent reading errors. Are a VLM's
   errors on two bars in one chart correlated, and does that help or hurt?
3. Table structure as graph prediction degrades better than grid prediction. Is
   there a formulation that makes merged cells a first-class case rather than an
   exception?
4. Chart derendering makes the reading error inspectable. Can a model output
   calibrated *uncertainty* per read value, so downstream arithmetic can propagate
   it?
5. General VLMs handle layout implicitly and produce no artefact. Can a VLM be
   trained to emit grounded coordinates as a by-product, restoring validatability
   without a separate pipeline?

## 20. Chapter Summary

{{ch:mm-ocr}} got the characters out; this chapter is about the meaning carried by
**where they were**.

**Linearisation destroys a form, measurably and exactly.** On the two-column
layout most forms use, reading-order association scores **0.500** against 2D
coordinates' **1.000** — and the half is structural
({{eq:two-column-half}}), identical at 4 fields and at 10. The sequence is
`label_A, label_B, value_A, value_B`, so half the pairs associate correctly by
luck.

**And it is a representation failure, not an OCR or a model failure.** Every token
was read correctly and no model ran. {{eq:reading-order-loses-2d}} is
non-injective, so nothing downstream can recover what flattening removed —
which is why layout-aware encoding
({{eq:2d-position-embedding}}) makes geometry an *input* rather than a heuristic.

**Tables are the same argument one level up.** A cell's meaning depends on two
other tokens found by geometry ({{eq:table-cell-addressing}}), so a misalignment
does not corrupt a value — it re-labels it, producing a well-formed row with the
wrong numbers ({{eq:table-misalignment-silent}}). And
{{eq:table-row-accuracy}} compounds per column, so a 12-column table needs 0.999
per-cell accuracy for 99% of rows.

**Charts multiply perception by arithmetic, and only some questions suffer.** A
one-pixel reading error gives **0.0046** relative error on a value, **0.0066** on
a ratio — and **0.089** on a difference of two close bars, a factor of **19**,
predicted at 21 by {{eq:amplification-factor}}. **Differences are ill-conditioned
and ratios are not**, which is a fact about the arithmetic rather than about the
model. Comparison accuracy is **0.972** at one pixel and **0.918** at three, and
{{eq:trustworthy-comparison}} says bars must differ by 0.8% of the axis span for a
99%-reliable answer.

**Axis truncation contains both a surprise and a non-surprise.** It makes a chart
**ten times more precise to read** (0.0004 against 0.0046) and leaves comparison
accuracy **identical**, because {{eq:comparison-accuracy}} depends only on pixel
geometry. **It helps a model and misleads a human simultaneously.**

The practical conclusion is the same shape in both halves: **stop asking the model
to be a measuring instrument.** Retrieve the underlying data where it exists
({{ch:rag-structured}}); extract a chart to a table where it does not, so the
reading error happens once and visibly; and keep coordinates through the pipeline
so the structure can be validated at all.

## 21. Further Reading

{{cite:huang2022layoutlmv3}} for making 2D position an input — the contribution is
the representation, not the pretraining objective.
{{cite:masry2022chartqa}} for chart reasoning, and read the question taxonomy
rather than the headline accuracy: the categories have very different
conditioning.
{{cite:mathew2021docvqa}} for the finding that the hard document questions are
structural.
{{cite:kim2022donut}} and {{cite:wang2024qwen2vl}} for the two OCR-free routes,
and {{ch:mm-ocr}} for what each gives up.
{{cite:li2023bird}} for the alternative this chapter keeps recommending — if the
data exists as data, query it rather than reading a picture of it.

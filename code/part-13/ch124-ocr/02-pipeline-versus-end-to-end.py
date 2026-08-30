# -*- coding: utf-8 -*-
# Extracted from: Chapter 124 — OCR and Document AI
# Source: src/.../ch124-ocr.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

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

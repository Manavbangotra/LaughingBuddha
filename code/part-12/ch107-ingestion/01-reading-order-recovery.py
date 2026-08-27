# -*- coding: utf-8 -*-
# Extracted from: Chapter 107 — Document Ingestion and Parsing
# Source: src/.../ch107-ingestion.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

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

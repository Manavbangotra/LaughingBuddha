# -*- coding: utf-8 -*-
# Extracted from: Chapter 125 — Layout, Tables, and Chart Understanding
# Source: src/.../ch125-layout.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

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

# -*- coding: utf-8 -*-
# Extracted from: Chapter 125 — Layout, Tables, and Chart Understanding
# Source: src/.../ch125-layout.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

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

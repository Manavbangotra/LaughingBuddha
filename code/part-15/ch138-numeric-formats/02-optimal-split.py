# -*- coding: utf-8 -*-
# Extracted from: Chapter 138 — Numerical Formats: FP32, FP16, BF16, and FP8
# Source: src/.../ch138-numeric-formats.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Where should the line between exponent and mantissa go?

The previous listing took the existing formats as given. This one asks the design
question directly: at a fixed total width, sweep every possible split and measure
which one wins.

The interesting part is that the sweep gives two different answers depending on
how you score it, and the disagreement is not a flaw in the experiment -- it is
the reason more than one format exists (eq:metric-picks-the-format). Then it asks
a second question that matters more in practice: what happens to the whole
picture once a per-tensor scale factor is allowed?
"""
import numpy as np

rng = np.random.default_rng(233)
N = 120000


def fp_round(x, e_bits, m_bits):
    bias = 2 ** (e_bits - 1) - 1
    emin, emax = 1 - bias, (2 ** e_bits - 2) - bias
    maxval = (2.0 - 2.0 ** (-m_bits)) * 2.0 ** emax
    ax = np.abs(x)
    safe = np.where(ax > 0, ax, 1.0)
    e = np.clip(np.floor(np.log2(safe)), emin, emax)
    step = 2.0 ** (e - m_bits)
    q = np.clip(np.round(x / step) * step, -maxval, maxval)
    return np.where(ax > 0, q, 0.0)


def int_round(x, bits):
    qmax = 2 ** (bits - 1) - 1
    s = np.max(np.abs(x)) / qmax
    return np.clip(np.round(x / s), -qmax, qmax) * s


def snr_db(x, q):
    """Weights every value by its ENERGY, so large values dominate and small
    ones are nearly free to lose."""
    err = np.sum((x - q) ** 2)
    return float(10 * np.log10(np.sum(x ** 2) / max(err, 1e-300)))


def dead(x, q):
    """Fraction of nonzero values that rounded to zero -- values the format
    could not reach at all, weighted equally regardless of magnitude."""
    nz = x != 0
    return float(np.mean((q[nz] == 0)))


def dist(decades):
    """Values spanning `decades` orders of magnitude below the largest."""
    mag = 10.0 ** rng.uniform(-decades, 0, size=N)
    return rng.normal(0, 1, size=N) * mag


TOTAL = 16
E_RANGE = (3, 4, 5, 6, 7, 8, 9, 10)
DECADES = (0.5, 1, 2, 4, 8)

print("Fixed budget of 16 bits: 1 sign, e exponent, 15 - e mantissa.")
print("Two scores, because they disagree. SNR in dB (higher better) weights by")
print("energy. 'dead' is the share of values the format cannot reach at all,")
print("and its 'best' column is the SMALLEST exponent that loses nothing.")
print()
print(f"{'decades':>9}{'':>6}" + "".join(f"{'e=' + str(e):>9}" for e in E_RANGE)
      + f"{'best':>7}")
print("-" * 94)

best_snr, best_dead, tab = {}, {}, {}
for d in DECADES:
    x = dist(d)
    qs = [fp_round(x, e, TOTAL - 1 - e) for e in E_RANGE]
    sn = [snr_db(x, q) for q in qs]
    dd = [dead(x, q) for q in qs]
    best_snr[d] = E_RANGE[int(np.argmax(sn))]
    ok = [e for e, v in zip(E_RANGE, dd) if v < 5e-4]
    best_dead[d] = ok[0] if ok else E_RANGE[int(np.argmin(dd))]
    tab[d] = (sn, dd)
    print(f"{d:>9.1f}{'SNR':>6}" + "".join(f"{v:>9.1f}" for v in sn)
          + f"{'e=' + str(best_snr[d]):>7}")
    print(f"{'':>9}{'dead':>6}" + "".join(f"{v:>9.1%}" for v in dd)
          + f"{'e=' + str(best_dead[d]):>7}")

print()
print()
print("Same question at 8 bits, with and without a per-tensor scale factor.")
print()
E8 = (2, 3, 4, 5, 6)
print(f"{'decades':>9}{'':>6}" + "".join(f"{'e=' + str(e):>9}" for e in E8)
      + f"{'INT8':>9}{'best':>7}")
print("-" * 70)

raw8, sc8 = {}, {}
for d in DECADES:
    x = dist(d)
    raw = [snr_db(x, fp_round(x, e, 7 - e)) for e in E8]
    sc = []
    for e in E8:
        hi = (2.0 - 2.0 ** -(7 - e)) * 2.0 ** ((2 ** e - 2) - (2 ** (e - 1) - 1))
        ss = np.max(np.abs(x)) / hi
        sc.append(snr_db(x, fp_round(x / ss, e, 7 - e) * ss))
    i8 = snr_db(x, int_round(x, 8))
    raw8[d], sc8[d] = (raw, i8), sc
    print(f"{d:>9.1f}{'raw':>6}" + "".join(f"{v:>9.1f}" for v in raw)
          + f"{i8:>9.1f}" + f"{'e=' + str(E8[int(np.argmax(raw))]):>7}")
    print(f"{'':>9}{'+sc':>6}" + "".join(f"{v:>9.1f}" for v in sc)
          + f"{i8:>9.1f}" + f"{'e=' + str(E8[int(np.argmax(sc))]):>7}")

s_lo, d_lo = tab[0.5]
s_hi, d_hi = tab[8]
print(f"""
Read the 16-bit table one row-pair at a time and the two scores disagree
completely.

Under SNR, e=3 wins at EVERY dynamic range -- {s_lo[0]:.1f} dB at half a decade
and {s_hi[0]:.1f} dB at eight decades, against e=8's {s_hi[5]:.1f} dB. More
mantissa always wins, no matter how much range the distribution spans.

Under the dead-value count, the picture is a hard constraint rather than a
ranking. At half a decade every split reaches everything, so e={best_dead[0.5]}
suffices. At four decades you need e={best_dead[4]}, and at eight you need
e={best_dead[8]}. Below that threshold the losses are not marginal: e=3 cannot
reach {d_hi[0]:.1%} of the eight-decade values and e=4 cannot reach
{d_hi[1]:.1%}.

Both numbers are correct, and the reason they disagree is worth more than either
of them. SNR is an ENERGY-weighted score, and in a distribution spanning eight
decades essentially all the energy sits in the top decade. Zeroing a value at
1e-7 costs SNR almost nothing, because there was almost nothing there to lose.
Counting dead values weights every element EQUALLY, so the same rounding is a
total loss.

Put the two together and they are not really in conflict -- they compose into a
design rule. Spend the SMALLEST exponent that reaches your distribution, then put
every remaining bit in the mantissa (eq:optimal-split). The dead-value row sets
the floor; the SNR row says never exceed it.

What that rule needs is knowledge of the distribution's dynamic range, and whether
you have it is the actual difference between the formats. FP16's e=5 is the right
answer if you know your values span a few decades. BF16's e=8 is the right answer
if you do not know, or if the range varies between tensors and you want one format
for all of them.

And which score matters downstream decides how much risk the floor carries
(eq:metric-picks-the-format).

For an activation tensor feeding a matmul, the output is a weighted sum, so an
error's effect is roughly proportional to its magnitude -- SNR is the right score,
and more mantissa is the right answer. That is FP16.

For a gradient tensor, every element is the update for one parameter. A gradient
rounded to zero means that parameter does not move, and it does not matter that
the gradient was small -- over many steps, small consistent updates are how most
parameters learn anything. Dead values are the right score, and more exponent is
the right answer. That is BF16.

So the two 16-bit formats are not a historical accident and neither is better.
They are optimal under two different loss functions, and the loss function is
chosen by the tensor's role rather than by its statistics.

The 8-bit table then makes the practical point that governs the rest of
{{part:15}}.

Compare each raw row against the +sc row below it. At eight decades the best raw
split reaches {max(raw8[8][0]):.1f} dB; with a single per-tensor scale factor the
best reaches {max(sc8[8]):.1f} dB. One number, stored once for the whole tensor,
is worth more than any redistribution of the per-value bits.

The reason is that an exponent field is a PER-VALUE scale, carried by every
element. A scale factor is a PER-TENSOR exponent, carried once. If the values in a
tensor share most of their magnitude information -- and weights within one layer
usually do -- then an exponent field per element pays repeatedly for something
nearly constant.

Push that to its conclusion and you get integer quantization: no exponent bits at
all, every bit in the mantissa, one shared scale. The INT8 column is competitive
at low dynamic range ({raw8[0.5][1]:.1f} dB against the best float's
{max(raw8[0.5][0]):.1f}) and falls behind as the range widens, which is exactly
what that reading predicts.

And it reframes the question the next chapters answer. The interesting parameter
is not "how many bits per weight" but "how many weights share a scale factor". A
tensor-wide scale is free and assumes the whole tensor has one dynamic range. A
per-channel or per-64-weight scale costs a little storage and assumes much less.
Every INT4 and INT8 scheme in {{ch:q-int8-int4}} is a position on that axis, and
the bit-width everyone quotes is the less interesting half of the
specification.""")

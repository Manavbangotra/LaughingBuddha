# -*- coding: utf-8 -*-
# Extracted from: Chapter 138 — Numerical Formats: FP32, FP16, BF16, and FP8
# Source: src/.../ch138-numeric-formats.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Every float format is one bit budget split two ways.

A format with B bits spends some on the EXPONENT, which buys dynamic range, and
the rest on the MANTISSA, which buys resolution. Nothing else is available. FP16
and BF16 are both 16 bits and differ only in where the line is drawn, and every
consequence people attribute to them follows from that line
(eq:format-is-a-budget).

This listing implements the rounding for each format directly, so the numbers
come from the arithmetic rather than from a table, and then applies them to two
distributions with very different shapes -- transformer weights, which are tightly
clustered, and gradients, which span many orders of magnitude.
"""
import numpy as np

rng = np.random.default_rng(229)


def fp_round(x, e_bits, m_bits):
    """Round to the nearest value representable with `e_bits` of exponent and
    `m_bits` of mantissa. Subnormals fall out of clamping the exponent at its
    minimum, which is exactly what the hardware does."""
    bias = 2 ** (e_bits - 1) - 1
    emin, emax = 1 - bias, (2 ** e_bits - 2) - bias
    maxval = (2.0 - 2.0 ** (-m_bits)) * 2.0 ** emax
    ax = np.abs(x)
    safe = np.where(ax > 0, ax, 1.0)
    e = np.clip(np.floor(np.log2(safe)), emin, emax)
    step = 2.0 ** (e - m_bits)
    q = np.round(x / step) * step
    q = np.clip(q, -maxval, maxval)
    return np.where(ax > 0, q, 0.0)


def int_round(x, bits):
    """Symmetric per-tensor integer quantization, for contrast: a FIXED step
    everywhere rather than a step that scales with magnitude."""
    qmax = 2 ** (bits - 1) - 1
    s = np.max(np.abs(x)) / qmax
    return np.clip(np.round(x / s), -qmax, qmax) * s


FORMATS = [
    ("FP32", 8, 23), ("FP16", 5, 10), ("BF16", 8, 7),
    ("FP8 E4M3", 4, 3), ("FP8 E5M2", 5, 2),
]


def describe(e_bits, m_bits):
    bias = 2 ** (e_bits - 1) - 1
    emin, emax = 1 - bias, (2 ** e_bits - 2) - bias
    return ((2.0 - 2.0 ** (-m_bits)) * 2.0 ** emax,      # largest normal
            2.0 ** emin,                                  # smallest normal
            2.0 ** (emin - m_bits))                       # smallest subnormal


print("What each format can represent, from the exponent/mantissa split alone.\n")
print(f"{'format':>10}{'bits':>6}{'exp':>5}{'man':>5}{'largest':>12}"
      f"{'smallest':>12}{'smallest':>12}{'decades of':>12}")
print(f"{'':>10}{'':>6}{'':>5}{'':>5}{'normal':>12}{'normal':>12}"
      f"{'subnormal':>12}{'range':>12}")
print("-" * 74)
for name, e, m in FORMATS:
    hi, lo, sub = describe(e, m)
    print(f"{name:>10}{1+e+m:>6}{e:>5}{m:>5}{hi:>12.2e}{lo:>12.2e}"
          f"{sub:>12.2e}{np.log10(hi/sub):>12.1f}")

W = rng.normal(0, 0.02, size=200000)                      # transformer weights
G = rng.normal(0, 1.0, size=200000) * 10.0 ** rng.uniform(-8, 0, size=200000)

print("\n\nApplied to two real distribution shapes.\n")
print(f"{'':>16}{'TRANSFORMER WEIGHTS  N(0, 0.02)':>30}"
      f"{'GRADIENTS  spanning 1e-8 to 1':>30}")
print(f"{'format':>16}{'rel err':>12}{'zeroed':>10}{'over':>8}"
      f"{'rel err':>12}{'zeroed':>10}{'over':>8}")
print("-" * 76)


def report(q, x, maxval=np.inf):
    nz = x != 0
    rel = float(np.mean(np.abs(q[nz] - x[nz]) / np.abs(x[nz])))
    zeroed = float(np.mean((q == 0) & nz))          # fell below the format
    over = float(np.mean(np.abs(x) > maxval))        # rose above it
    return rel, zeroed, over


def scaled(x, e_bits, m_bits):
    """What real FP8 does: divide by a per-tensor scale so the distribution
    uses the format's range, quantize, multiply back. The scale is stored
    alongside in higher precision and is part of the format in practice."""
    hi = describe(e_bits, m_bits)[0]
    s = np.max(np.abs(x)) / hi
    return fp_round(x / s, e_bits, m_bits) * s


rows = {}


def line(name, a, b):
    rows[name] = (a, b)
    print(f"{name:>16}{a[0]:>12.2e}{a[1]:>10.2%}{a[2]:>8.2%}"
          f"{b[0]:>12.2e}{b[1]:>10.2%}{b[2]:>8.2%}")


for name, e, m in FORMATS:
    hi = describe(e, m)[0]
    line(name, report(fp_round(W, e, m), W, hi),
         report(fp_round(G, e, m), G, hi))

print(f"{'':>16}{'--- with a per-tensor scale factor ---':>70}")
for name, e, m in FORMATS[3:]:
    line(name + " +scale", report(scaled(W, e, m), W), report(scaled(G, e, m), G))
line("INT8 +scale", report(int_round(W, 8), W), report(int_round(G, 8), G))

print("\n\nLoss scaling: what multiplying by a constant before rounding buys.\n")
print(f"{'scale':>10}{'FP16 zeroed':>14}{'BF16 zeroed':>14}"
      f"{'FP16 rel err':>15}")
print("-" * 53)
for k in (0, 8, 16, 24):
    s = 2.0 ** k
    f16 = report(fp_round(G * s, 5, 10) / s, G)
    b16 = report(fp_round(G * s, 8, 7) / s, G)
    print(f"{'2^' + str(k):>10}{f16[1]:>14.2%}{b16[1]:>14.2%}{f16[0]:>15.2e}")

w16, wbf = rows["FP16"][0], rows["BF16"][0]
g16, gbf = rows["FP16"][1], rows["BF16"][1]
e4, e5 = rows["FP8 E4M3"], rows["FP8 E5M2"]
e4s, e5s = rows["FP8 E4M3 +scale"], rows["FP8 E5M2 +scale"]
i8s = rows["INT8 +scale"]
print(f"""
The first table is the whole design space in one sentence: at equal total width,
every bit given to the exponent is a bit taken from the mantissa. FP16 and BF16
are both 16 bits. FP16 spends 5 on the exponent and 10 on the mantissa; BF16
spends 8 and 7. Everything else in that table follows -- BF16 reaches
{describe(8,7)[0]/describe(5,10)[0]:.0f}x higher at the top and far lower at the
bottom, and pays {2**(10-7):.0f}x coarser resolution for it
(eq:format-is-a-budget).

The second table shows that the trade is not settled in the abstract. It is
settled by the DISTRIBUTION you intend to store.

On transformer weights -- tightly clustered, spanning a few decades -- FP16's
extra mantissa bits win: {w16[0]:.2e} against BF16's {wbf[0]:.2e}, about
{wbf[0]/w16[0]:.0f}x better, with neither format losing a value. The range BF16
bought is range this distribution never uses.

On gradients spanning eight decades the ranking reverses, and it reverses in the
column that matters rather than the one you were watching. FP16 silently zeroes
{g16[1]:.1%} of them; BF16 zeroes {gbf[1]:.1%}. FP16's relative error on the
survivors is still the better number and it is the wrong number, because a
gradient rounded to zero is not imprecise -- it is ABSENT, and the parameter it
belonged to receives no update at all (eq:underflow-is-not-error).

That distinction is the most useful idea in the chapter and it generalises past
floats. Resolution failures degrade every value slightly and the process usually
absorbs them. Range failures remove some values completely, and nothing
downstream can tell a value that rounded to zero from a value that was zero.

Now the block with the scale factors, which changes the reading of the two FP8
rows above it entirely.

Unscaled, E4M3 is WORSE than E5M2 on weights: {e4[0][0]:.2e} against
{e5[0][0]:.2e}. That looks like a contradiction of its design and it is not.
E4M3's smallest normal value is {describe(4,3)[1]:.2e}, and these weights have a
standard deviation of 0.02, so most of the distribution falls into E4M3's
subnormal region where the step size is fixed and coarse. The format does not lack
resolution; it lacks the RANGE to place this distribution where its resolution
lives.

Give each format a single per-tensor scale factor -- divide by a constant so the
distribution fills the format's range, quantize, multiply back -- and the designed
behaviour appears. E4M3 goes from {e4[0][0]:.2e} to {e4s[0][0]:.2e} on weights,
now beating E5M2's {e5s[0][0]:.2e}. On gradients E5M2 zeroes {e5s[1][1]:.2%}
against E4M3's {e4s[1][1]:.1%}.

So the division of labour cite:micikevicius2022fp8 specifies -- E4M3 for weights
and activations, E5M2 for gradients -- is not a convention. It is this table, and
it only appears once the scale factor is present.

Which is the practical lesson: at 8 bits, the scale factor is not an
implementation detail, it is part of the format. Eight bits is not enough to
carry both the location of a distribution and its shape, so the location is
factored out and stored separately at higher precision. Every scheme in the
chapters that follow is an argument about how FINELY to factor it out -- per
tensor, per channel, per block of 64 weights -- and that granularity choice
matters more than the bit-width people quote.

The INT8 row makes the same point from the other side. Integer quantization has no
exponent field at all, so its steps are uniform rather than log-spaced. With a
scale factor it reaches {i8s[0][0]:.2e} on weights against E4M3's
{e4s[0][0]:.2e}: an 8-bit FLOAT beats an 8-bit INTEGER on a bell-shaped
distribution, because log-spaced steps put resolution where a bell curve puts its
mass. On the eight-decade gradients it zeroes {i8s[1][1]:.1%}, because one uniform
step cannot serve eight decades at any scale.

The last table is the historical footnote that turns out to be the clearest
demonstration in the listing. Loss scaling multiplies gradients by a large
constant before storing them in FP16 and divides afterwards, and underflow falls
from {g16[1]:.1%} toward nothing as the constant grows.

That is a manual exponent adjustment, applied because the format's own exponent
field is too small. BF16 needs none of it at any scale.
cite:micikevicius2018mixed introduced loss scaling as a necessary part of FP16
training; BF16 removed the need by moving the bits; cite:micikevicius2022fp8 then
encoded the same lesson into two formats rather than into the training loop. One
idea, learned three times, each time pushed further down the stack.""")

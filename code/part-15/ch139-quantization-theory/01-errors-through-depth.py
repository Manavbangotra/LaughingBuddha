# -*- coding: utf-8 -*-
# Extracted from: Chapter 139 — Why Quantization Works: Theory and Error Analysis
# Source: src/.../ch139-quantization-theory.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Does quantization error compound through depth, or cancel?

The obvious worry about quantizing a deep network is that each layer adds error
to the layer below it, so a 1% perturbation at every layer of an 80-layer model
should arrive at the output as something enormous. If that were true, quantization
would not work at all, and it plainly does.

The reason it does not is worth measuring rather than asserting. Rounding errors
in different layers are approximately INDEPENDENT, and independent perturbations
add in quadrature rather than linearly -- so the output error grows like the
square root of depth, not like depth (eq:errors-add-in-quadrature).

This listing measures the growth rate directly, and then breaks the independence
on purpose to show what the assumption is worth.
"""
import numpy as np

rng = np.random.default_rng(239)

D, N = 128, 2048
MAX_L = 32


def quantize(W, bits, group=None):
    """Symmetric integer quantization with a scale per group of rows. group=None
    means one scale for the whole tensor."""
    qmax = 2 ** (bits - 1) - 1
    if group is None:
        s = np.max(np.abs(W)) / qmax
    else:
        g = np.max(np.abs(W.reshape(-1, group)), axis=1, keepdims=True) / qmax
        s = np.repeat(g, group, axis=1).reshape(W.shape)
    s = np.where(s == 0, 1e-12, s)
    return np.clip(np.round(W / s), -qmax, qmax) * s


def make_net(L, outlier_frac=0.0, outlier_scale=1.0):
    Ws = []
    for _ in range(L):
        W = rng.normal(size=(D, D)) / np.sqrt(D)
        if outlier_frac > 0:
            k = max(1, int(outlier_frac * D))
            cols = rng.choice(D, size=k, replace=False)
            W[:, cols] *= outlier_scale
        Ws.append(W)
    return Ws


def forward(Ws, X, bits=None, upto=None):
    h = X
    for W in Ws[:upto]:
        Q = W if bits is None else quantize(W, bits)
        h = np.tanh(h @ Q)
    return h


X = rng.normal(size=(N, D))
BITS = (8, 6, 4)

print(f"{D}-wide tanh network. Relative output error against depth, and the")
print("growth exponent p in error ~ depth^p fitted over the last half.")
print()
print(f"{'depth':>7}" + "".join(f"{str(b) + '-bit':>12}" for b in BITS))
print("-" * 43)

DEPTHS = (1, 2, 4, 8, 16, 32)
curves = {b: [] for b in BITS}
nets = {b: make_net(MAX_L) for b in BITS}
for L in DEPTHS:
    row = []
    for b in BITS:
        Ws = nets[b]
        ref = forward(Ws, X, upto=L)
        got = forward(Ws, X, bits=b, upto=L)
        e = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
        curves[b].append(e)
        row.append(e)
    print(f"{L:>7}" + "".join(f"{v:>12.4e}" for v in row))


def exponent(depths, errs):
    """Fit error ~ depth^p on the log-log tail."""
    d = np.log(np.array(depths[2:], float))
    e = np.log(np.array(errs[2:], float))
    return float(np.polyfit(d, e, 1)[0])


print()
print(f"{'':>7}" + "".join(f"{'p = ' + f'{exponent(DEPTHS, curves[b]):.2f}':>12}"
                            for b in BITS))

print()
print()
print("Now break the independence: a few weight columns scaled up, so one")
print("shared scale factor must cover a much wider range.")
print()
print(f"{'outlier':>9}{'outlier':>9}{'8-bit err':>12}{'vs clean':>10}"
      f"{'6-bit err':>12}{'vs clean':>10}{'exponent':>10}")
print(f"{'columns':>9}{'scale':>9}{'at depth 16':>12}{'':>10}"
      f"{'at depth 16':>12}{'':>10}{'p':>10}")
print("-" * 73)

out_rows = {}
DD = (1, 2, 4, 8, 16)
for frac, sc in ((0.0, 1.0), (0.02, 4.0), (0.02, 16.0), (0.10, 16.0)):
    Ws = make_net(16, frac, sc)
    e8, e6 = [], []
    for L in DD:
        ref = forward(Ws, X, upto=L)
        e8.append(float(np.linalg.norm(forward(Ws, X, 8, L) - ref)
                        / np.linalg.norm(ref)))
        e6.append(float(np.linalg.norm(forward(Ws, X, 6, L) - ref)
                        / np.linalg.norm(ref)))
    p8 = exponent(DD, e8)
    out_rows[(frac, sc)] = (e8[-1], e6[-1], p8)
    b = out_rows[(0.0, 1.0)]
    print(f"{frac:>9.0%}{sc:>9.0f}{e8[-1]:>12.4e}{e8[-1]/b[0]:>10.1f}x"
          f"{e6[-1]:>12.4e}{e6[-1]/b[1]:>10.1f}x{p8:>10.2f}")

p8 = exponent(DEPTHS, curves[8])
p4 = exponent(DEPTHS, curves[4])
base = out_rows[(0.0, 1.0)]
mild = out_rows[(0.02, 4.0)]
bad = out_rows[(0.02, 16.0)]
worst = out_rows[(0.10, 16.0)]
print(f"""
The first table answers the question the chapter exists for, and the answer is in
the fitted exponents rather than in the errors themselves.

Error grows as depth to the power {p8:.2f} at 8 bits and {p4:.2f} at 4 bits.
Both are close to one half. That is the signature of INDEPENDENT perturbations
adding in quadrature: L layers each contributing an error of size e produce an
output error of about e times the square root of L, not e times L and certainly
not e compounding multiplicatively (eq:errors-add-in-quadrature).

The difference matters enormously at the depths real models have. At 80 layers,
linear compounding would multiply the per-layer error by 80 and multiplicative
compounding would be far worse still; square-root growth multiplies it by about
nine. That factor of nine, against a per-layer error that is already small, is
the entire reason a quantized 80-layer model still works.

It is worth being precise about why the errors are independent. Each weight's
rounding error is determined by where that weight happens to fall between two
representable values, which is essentially arbitrary and uncorrelated between
weights, between layers, and with the data flowing through. Nothing in the
computation aligns them, so they do not reinforce.

Which immediately identifies what would break the argument: anything that makes
the errors CORRELATED. The second table constructs exactly that.

Scaling up a small number of weight columns forces the shared scale factor to
cover a range it did not have to cover before. The step size grows for every
weight in the tensor, including the ordinary ones, so the errors are no longer
small independent perturbations -- they are a systematic coarsening driven by a
handful of entries.

At 2% of columns scaled by 4, the 8-bit error at depth 16 rises from
{base[0]:.2e} to {mild[0]:.2e} -- {mild[0]/base[0]:.1f}x. At a scale of 16,
{bad[0]:.2e}, which is {bad[0]/base[0]:.1f}x. With 10% of columns scaled by 16,
{worst[0]:.2e}: {worst[0]/base[0]:.1f}x the clean network, with no change to the
bit-width at all.

Put that beside what bit-width buys. Going from 8 bits to 6 on the CLEAN network
costs {base[1]/base[0]:.1f}x -- two whole bits, for a factor of four. Adding 2% of
columns at 16x scale, at 8 bits throughout, costs {bad[0]/base[0]:.1f}x.

So the distribution matters more than the bit-width, by a wide margin, and that
is not how the choice is usually framed. Teams argue about 4-bit against 8-bit
and accept whatever outlier structure the checkpoint happens to have, when the
second factor is the larger one.

One honest note on the exponent column. It sits at {base[2]:.2f} on the clean
network and drifts up to {worst[2]:.2f} in the worst outlier row. That drift is
not a change in how errors accumulate -- it is a ceiling effect. A relative error
near 1 means the output has become uncorrelated with the reference, and a
saturating quantity cannot keep following a power law. Where the errors are still
small, the exponent stays near one half.

Which leaves the practical reading. Outliers do not break the way errors
accumulate; they inflate the per-layer error that then accumulates in the same
square-root fashion. The quadrature argument survives and the constant in front of
it does not.

That distinction is what makes the problem tractable. If outliers changed the
accumulation structure, deep models would be unquantizable and the only remedy
would be fewer layers. Because they inflate a local quantity instead, the remedy
is local -- give the outliers their own scale, or their own precision, or rotate
the basis so no coordinate dominates -- and cite:dettmers2022int8,
cite:xiao2023smoothquant, cite:lin2023awq and cite:tseng2024quipsharp are four
different ways of doing exactly that.""")

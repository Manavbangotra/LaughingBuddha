# -*- coding: utf-8 -*-
# Extracted from: Chapter 140 — INT8, INT4, GPTQ, and AWQ
# Source: src/.../ch140-int8-int4.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Four answers to one problem, measured against each other.

cite:dettmers2022int8 identified the phenomenon that breaks naive INT8 above a
scale threshold: a small number of activation channels carry values far larger
than the rest, and any scale factor shared with them is forced to cover a range
the ordinary values never use.

Four responses followed, and they are genuinely different ideas rather than
variations on one:

  keep outliers in higher precision   cite:dettmers2022int8
  migrate the difficulty to weights   cite:xiao2023smoothquant
  protect the salient channels        cite:lin2023awq
  rotate so no channel dominates      cite:tseng2024quipsharp

This listing implements all four on the same layer and measures the OUTPUT error,
which is what matters, rather than the weight error, which is not
(eq:output-error-is-the-target).
"""
import numpy as np

rng = np.random.default_rng(251)

D_IN, D_OUT, N = 256, 256, 4096
OUTLIER_COLS = 6
OUTLIER_SCALE = 24.0


def quantize(A, bits, axis=None):
    """Symmetric integer quantization. axis=None shares one scale over the whole
    tensor; axis=0 gives each column its own."""
    qmax = 2 ** (bits - 1) - 1
    s = (np.max(np.abs(A)) if axis is None
         else np.max(np.abs(A), axis=axis, keepdims=True)) / qmax
    s = np.maximum(s, 1e-12)
    return np.clip(np.round(A / s), -qmax, qmax) * s


W = rng.normal(size=(D_IN, D_OUT)) / np.sqrt(D_IN)
X = rng.normal(size=(N, D_IN))
hot = rng.choice(D_IN, size=OUTLIER_COLS, replace=False)
X[:, hot] *= OUTLIER_SCALE                      # the emergent outlier features
REF = X @ W


def err(Y):
    return float(np.linalg.norm(Y - REF) / np.linalg.norm(REF))


def hadamard(n):
    """A Hadamard matrix of size n (a power of two), normalised to be
    orthogonal. Multiplying by it mixes every coordinate into every other."""
    H = np.ones((1, 1))
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(n)


def baseline_fp(bits):
    """Weights quantized, activations left alone -- weight-only quantization."""
    return err(X @ quantize(W, bits))


def naive(bits):
    """Both quantized, one scale each. This is the configuration that breaks."""
    return err(quantize(X, bits) @ quantize(W, bits))


def per_channel(bits):
    """A scale per activation channel. The outlier channel gets its own."""
    return err(quantize(X, bits, axis=0) @ quantize(W, bits, axis=0))


def mixed_precision(bits, k=OUTLIER_COLS):
    """cite:dettmers2022int8: keep the k largest-magnitude channels in full
    precision and quantize the rest."""
    mag = np.max(np.abs(X), axis=0)
    keep = np.argsort(-mag)[:k]
    mask = np.zeros(D_IN, bool); mask[keep] = True
    Y = X[:, mask] @ W[mask]
    Xq = quantize(X[:, ~mask], bits)
    Y = Y + Xq @ quantize(W[~mask], bits)
    return err(Y)


def smoothquant(bits):
    """cite:xiao2023smoothquant: divide activations by a per-channel factor and
    multiply the corresponding weight rows by it. Exactly equivalent as a
    function; only what the quantizer sees changes. alpha balances how much
    difficulty is moved, and is searched here as the paper searches it."""
    a = np.maximum(np.max(np.abs(X), axis=0), 1e-12)
    w = np.maximum(np.max(np.abs(W), axis=1), 1e-12)
    best = None
    for alpha in np.linspace(0.1, 0.9, 9):
        s = np.maximum((a ** alpha) / (w ** (1 - alpha)), 1e-12)
        s = s / np.mean(s)
        e = err(quantize(X / s, bits) @ quantize(W * s[:, None], bits))
        if best is None or e < best[0]:
            best = (e, alpha)
    return best[0]


def awq(bits):
    """cite:lin2023awq: importance comes from ACTIVATION statistics, and the
    per-channel scale is s = mean|X_j|^alpha with alpha SEARCHED. The search is
    part of the method, not a refinement of it."""
    imp = np.mean(np.abs(X), axis=0)
    imp = np.maximum(imp / np.mean(imp), 1e-12)
    best = None
    for alpha in np.linspace(0.0, 1.0, 11):
        s = np.maximum(imp ** alpha, 1e-12)
        e = err(quantize(X / s, bits) @ quantize(W * s[:, None], bits))
        if best is None or e < best[0]:
            best = (e, alpha)
    return best[0]


def rotated(bits):
    """cite:tseng2024quipsharp: rotate into a basis where the energy is spread
    over all coordinates, quantize there, and rotate back. Orthogonal, so the
    function is unchanged."""
    H = hadamard(D_IN)
    return err((quantize(X @ H, bits) @ quantize(H.T @ W, bits)))


print(f"A {D_IN}x{D_OUT} layer with {OUTLIER_COLS} activation channels "
      f"{OUTLIER_SCALE:.0f}x larger than the rest.")
print("Relative error of the layer OUTPUT. Weights and activations both "
      "quantized\nexcept where noted.")
print()
print(f"{'method':>34}" + "".join(f"{str(b) + ' bits':>12}" for b in (8, 6, 4)))
print("-" * 70)

METHODS = [
    ("weight-only (activations in fp)", baseline_fp),
    ("naive: one scale each", naive),
    ("per-channel activation scales", per_channel),
    ("mixed precision for outliers", mixed_precision),
    ("SmoothQuant migration", smoothquant),
    ("AWQ salient-channel scaling", awq),
    ("Hadamard rotation", rotated),
]
res = {}
for name, fn in METHODS:
    vals = [fn(b) for b in (8, 6, 4)]
    res[name] = vals
    print(f"{name:>34}" + "".join(f"{v:>12.4f}" for v in vals))

print()
print()
print("How much of the damage is the outliers? Same methods, no outliers.")
print()
print(f"{'method':>34}" + "".join(f"{str(b) + ' bits':>12}" for b in (8, 6, 4)))
print("-" * 70)
X_HOT = X.copy()
X[:, hot] /= OUTLIER_SCALE
REF = X @ W
clean = {}
for name, fn in METHODS:
    vals = [fn(b) for b in (8, 6, 4)]
    clean[name] = vals
    print(f"{name:>34}" + "".join(f"{v:>12.4f}" for v in vals))
X = X_HOT
REF = X @ W

nv, pc = res["naive: one scale each"], res["per-channel activation scales"]
mp, sq = res["mixed precision for outliers"], res["SmoothQuant migration"]
aw, ro = res["AWQ salient-channel scaling"], res["Hadamard rotation"]
wo = res["weight-only (activations in fp)"]
cn = clean["naive: one scale each"]
print(f"""
The naive row is the problem. With both tensors quantized against a single scale
each, the 8-bit output error is {nv[0]:.4f}; the identical configuration without
outliers gives {cn[0]:.4f}. Six channels out of {D_IN} cost a factor of
{nv[0]/cn[0]:.1f}.

The weight-only row is why weight-only quantization dominates local inference.
Leaving activations in full precision sidesteps the problem entirely --
{wo[0]:.4f} at 8 bits, {wo[2]:.4f} at 4 -- because the outliers are an ACTIVATION
phenomenon and a method that never quantizes activations never meets them. It also
never gets INT8 tensor cores, which is the trade ch:q-throughput-latency prices.

Now the four responses. All of them recover most of the loss, and the ranking is
not the one the literature's chronology suggests.

Per-channel activation scales reach {pc[0]:.4f} at 8 bits against naive's
{nv[0]:.4f} -- a factor of {nv[0]/pc[0]:.1f}, from the simplest possible change.
Mixed precision, keeping the {OUTLIER_COLS} largest channels in full precision,
reaches {mp[0]:.4f}, the best number in the table. The Hadamard rotation reaches
{ro[0]:.4f}. SmoothQuant reaches {sq[0]:.4f} and AWQ {aw[0]:.4f}, both with their
scaling exponent searched as the papers search it.

So the two most-cited methods are the two WORST performers here, and that requires
an explanation rather than a shrug.

The explanation is that this listing measures error and the methods were designed
under a constraint it does not model. An INT8 matrix multiply computes a dot
product along the reduction axis, and a single scale factor must apply to that
whole dot product -- so a per-channel activation scale, which varies ALONG the
reduction axis, cannot be folded into an INT8 GEMM. It is expressible in numpy and
not in the kernel that makes INT8 worth using.

That is precisely what SmoothQuant exists to fix. Dividing the activations by a
per-channel constant and multiplying the corresponding weight ROWS by the same
constant leaves the function identical, moves the difficulty into a tensor that
had range to spare, and -- the part that matters -- the surviving per-channel
factor now lives on the weights, where it is a compile-time constant folded in
once rather than a runtime scale varying inside the reduction.

AWQ's contribution is orthogonal to that and survives this listing intact: the
importance of a weight channel is computed from ACTIVATION statistics rather than
from weight magnitudes. A small weight multiplying a large activation matters more
than a large weight multiplying a small one, and only the data can say which is
which (eq:output-error-is-the-target). Note what that implies about the
calibration set -- it is not a formality, it is where the method gets its
information, and it is almost never reported.

The Hadamard row deserves its own note because it attacks the problem from the
furthest away. Rather than protecting the outlier coordinates it changes the basis
so that no coordinate is an outlier: an orthogonal mixing spreads each channel's
energy across all of them, turning a spiky distribution into an approximately
Gaussian one, which is exactly what a uniform quantizer handles well. It costs a
transform on the critical path, which is the trade cite:tseng2024quipsharp makes.

Read the second table against the first and the framing settles. Without outliers,
every method lands within a small factor of every other and naive is competitive
at {cn[0]:.4f}. The entire difference between these techniques is what they do
about a handful of channels.

Which is the chapter's organising claim. These are not four quantization
algorithms. They are four answers to what to do when ch:q-formats's
eq:scale-group-condition is violated by a few coordinates -- isolate them, move
them, protect them, or destroy the basis in which they are special. And the choice
between them is made on kernel cost rather than on the error column, which is why
reading the error column alone gives the wrong ranking.""")

# -*- coding: utf-8 -*-
# Extracted from: Chapter 59 — Convolutional Neural Networks
# Source: src/.../ch059-cnns.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Convolution via im2col, its shape rules, its cost, and the equivariance
that is the whole point of it.
"""
import numpy as np

rng = np.random.default_rng(0)


# --- section 5.1: the operation ---------------------------------------------
def conv_naive(X, K, stride=1, pad=0):
    """Eq. 59.1 written literally. Correct, and far too slow to use."""
    N, C, H, W = X.shape
    F, _, kh, kw = K.shape
    Xp = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    Ho = (H + 2 * pad - kh) // stride + 1
    Wo = (W + 2 * pad - kw) // stride + 1
    Y = np.zeros((N, F, Ho, Wo))
    for n in range(N):
        for f in range(F):
            for i in range(Ho):
                for j in range(Wo):
                    patch = Xp[n, :, i * stride:i * stride + kh,
                               j * stride:j * stride + kw]
                    Y[n, f, i, j] = np.sum(patch * K[f])
    return Y


def im2col(X, kh, kw, stride, pad):
    """Section 7.1: every patch becomes a column, using stride tricks."""
    N, C, H, W = X.shape
    Xp = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    Ho = (H + 2 * pad - kh) // stride + 1
    Wo = (W + 2 * pad - kw) // stride + 1
    s = Xp.strides
    patches = np.lib.stride_tricks.as_strided(
        Xp,
        shape=(N, C, Ho, Wo, kh, kw),
        strides=(s[0], s[1], s[2] * stride, s[3] * stride, s[2], s[3]),
        writeable=False)
    return patches.transpose(0, 2, 3, 1, 4, 5).reshape(N * Ho * Wo, -1), Ho, Wo


def conv_im2col(X, K, stride=1, pad=0):
    """The same operation as ONE matrix multiply."""
    N = X.shape[0]
    F, C, kh, kw = K.shape
    cols, Ho, Wo = im2col(X, kh, kw, stride, pad)
    out = cols @ K.reshape(F, -1).T
    return out.reshape(N, Ho, Wo, F).transpose(0, 3, 1, 2)


print("=" * 72)
print("im2col computes the same thing, much faster (section 7.1)")
print("=" * 72)
X = rng.normal(size=(4, 3, 24, 24))
K = rng.normal(size=(8, 3, 3, 3)) * 0.1
a = conv_naive(X, K, stride=1, pad=1)
b = conv_im2col(X, K, stride=1, pad=1)
print(f"shapes agree: {a.shape == b.shape}   ({a.shape})")
print(f"max |naive - im2col| = {np.abs(a - b).max():.3e}")

import time
for label, fn in (("six nested loops", conv_naive),
                  ("im2col + one matmul", conv_im2col)):
    t0 = time.perf_counter()
    for _ in range(3):
        fn(X, K, 1, 1)
    dt = (time.perf_counter() - t0) / 3
    print(f"{label:<22} {dt * 1e3:>9.2f} ms")

print("\nThe two agree to floating point and differ enormously in speed.")
print("The reason is not a better algorithm — the FLOP count is identical —")
print("but that a matmul is the single most optimised operation available")
print("on any machine, and rewriting the problem as one buys all of that")
print("work for free. That is section 51.6's roofline argument applied to")
print("a different operation.")
print("\nThe cost is memory: im2col materialises every patch, so it uses")
print(f"k^2 = {3 * 3}x the input's memory. That trade is almost always")
print("worth making and it is why frameworks do it.")

# --- section 5.2: output shapes ---------------------------------------------
print("\n" + "=" * 72)
print("output size (eq. 59.2)")
print("=" * 72)


def out_size(H, k, s=1, p=0, d=1):
    return (H + 2 * p - d * (k - 1) - 1) // s + 1


print(f"{'H':>5} {'k':>4} {'stride':>7} {'pad':>5} {'dilation':>9} "
      f"{'H_out':>7}  {'note':<28}")
cases = [
    (32, 3, 1, 0, 1, "valid: shrinks by k-1"),
    (32, 3, 1, 1, 1, "SAME padding, p=(k-1)/2"),
    (32, 5, 1, 2, 1, "SAME padding, k=5"),
    (32, 3, 2, 1, 1, "halving"),
    (32, 1, 1, 0, 1, "1x1: shape unchanged"),
    (32, 3, 1, 2, 2, "dilated d=2, SAME"),
    (32, 3, 1, 4, 4, "dilated d=4, SAME"),
]
for H, k, s, p, d, note in cases:
    print(f"{H:>5} {k:>4} {s:>7} {p:>5} {d:>9} {out_size(H, k, s, p, d):>7}  "
          f"{note:<28}")

print("\nThe SAME-padding rule p = d*(k-1)/2 needs k to be ODD for p to be")
print("an integer, which is why every architecture uses odd kernels. An")
print("even kernel cannot be centred on its output position.")

# --- section 6.1: equivariance, measured ------------------------------------
print("\n" + "=" * 72)
print("convolution is EQUIVARIANT, not invariant (eq. 59.6)")
print("=" * 72)
img = rng.normal(size=(1, 1, 20, 20))
Ke = rng.normal(size=(1, 1, 3, 3))
shift = 3
shifted = np.roll(img, shift, axis=3)

y_then_shift = np.roll(conv_im2col(img, Ke, 1, 1), shift, axis=3)
shift_then_y = conv_im2col(shifted, Ke, 1, 1)

interior = (slice(None), slice(None), slice(1, -1), slice(5, -5))
print(f"max |conv(shift(x)) - shift(conv(x))|, whole map : "
      f"{np.abs(y_then_shift - shift_then_y).max():.3e}")
print(f"max |...|, interior only (away from the border)  : "
      f"{np.abs(y_then_shift[interior] - shift_then_y[interior]).max():.3e}")

print("\nIn the interior the two commute to floating point: shifting the")
print("input and then convolving gives exactly the same answer as")
print("convolving and then shifting. That is eq. 59.6.")
print("\nAt the border they do not, and the reason is padding — the zeros")
print("outside the image are not a translation of anything. So a")
print("'translation equivariant' architecture is exactly equivariant only")
print("in its interior, and a network can and does learn to read the border")
print("as a position signal.")

# --- and the difference from INVARIANCE -------------------------------------
print("\n" + "=" * 72)
print("invariance comes from POOLING, not from the convolution")
print("=" * 72)
feat = conv_im2col(img, Ke, 1, 1)
feat_shift = conv_im2col(shifted, Ke, 1, 1)
print(f"feature map changed by shifting  : "
      f"{np.abs(feat - feat_shift).max():.4f}   (equivariant, so it MOVED)")
print(f"global average pool, original    : {feat.mean():.6f}")
print(f"global average pool, shifted     : {feat_shift.mean():.6f}")
print(f"difference                       : "
      f"{abs(feat.mean() - feat_shift.mean()):.3e}")
print(f"global MAX pool, original        : {feat.max():.6f}")
print(f"global MAX pool, shifted         : {feat_shift.max():.6f}")

print("\nThe feature map moved — that is equivariance. The global pool of")
print("that map barely moved — that is invariance, and it came from the")
print("pooling rather than from the convolution.")
print("\nThe residual difference is again the border: a circular shift moves")
print("content across the edge, where the padding is. On a genuinely")
print("translated scene rather than a rolled array, the pooled value would")
print("be exactly unchanged in the interior.")
print("\nThe design consequence is the ordering: equivariance in the middle,")
print("so the network can locate things, and invariance at the end, so the")
print("label does not depend on where they were.")

# --- section 5.3: parameters vs FLOPs ---------------------------------------
print("\n" + "=" * 72)
print("parameters do not depend on resolution; FLOPs do (eqs. 59.3, 59.4)")
print("=" * 72)


def conv_cost(C_in, C_out, k, H, W, s=1):
    Ho, Wo = out_size(H, k, s, k // 2), out_size(W, k, s, k // 2)
    return C_out * C_in * k * k + C_out, 2 * C_out * C_in * k * k * Ho * Wo


print(f"{'layer':<28} {'resolution':>12} {'parameters':>12} {'MFLOPs':>10}")
for H in (32, 112, 224, 448):
    p_, f_ = conv_cost(64, 128, 3, H, H)
    print(f"{'conv 3x3, 64 -> 128':<28} {f'{H}x{H}':>12} {p_:>12,} "
          f"{f_ / 1e6:>10.1f}")

print("\nThe parameter count is constant and the FLOPs scale with the area.")
print("That is weight sharing stated as an accounting fact, and it is why")
print("convolutional networks handle high resolution more gracefully than")
print("attention does: attention's cost grows with the SQUARE of the number")
print("of positions (Chapter 71), and a convolution's grows linearly.")

# --- compare against a dense layer ------------------------------------------
print("\n" + "=" * 72)
print("the parameter comparison that motivates the whole architecture")
print("=" * 72)
H = W = 224
print(f"a {H}x{W}x3 image -> 1000 units")
dense = H * W * 3 * 1000 + 1000
print(f"  fully connected      : {dense:>15,} parameters")
c_p, c_f = conv_cost(3, 64, 3, H, W)
print(f"  conv 3x3, 3 -> 64    : {c_p:>15,} parameters "
      f"({dense / c_p:,.0f}x fewer)")
print(f"                         {c_f / 1e6:.1f} MFLOPs")

print("\nFive orders of magnitude, and the parameter count is the LESS")
print("important half of it. The dense layer treats every pixel as an")
print("unrelated feature, so a one-pixel shift presents it with a")
print("completely different input that it must learn separately. The")
print("convolution cannot express a position-dependent detector at all.")
print("\nThat inability is the inductive bias. Section 6.3 makes it precise:")
print("a convolution is a dense layer with hard weight tying and hard")
print("zeros, so its hypothesis class is a strict SUBSET. It is not more")
print("powerful — it is less powerful, in the right way.")

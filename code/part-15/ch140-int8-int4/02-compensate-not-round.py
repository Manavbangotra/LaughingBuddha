# -*- coding: utf-8 -*-
# Extracted from: Chapter 140 — INT8, INT4, GPTQ, and AWQ
# Source: src/.../ch140-int8-int4.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Rounding is a choice, and round-to-nearest is the wrong one.

Every quantizer so far has rounded each weight to its nearest representable
value, independently. That is optimal if you want each WEIGHT to be close to its
original. It is not what you want.

What you want is the layer's OUTPUT to be close, and that is a different problem
because the weights are not independent in their effect: an error in one weight
can be partly cancelled by deliberately mis-rounding another
(eq:compensate-not-round).

cite:frantar2023gptq turns that observation into an algorithm. Quantize the
weights in order; after each one, push its rounding error into the weights not yet
quantized, weighted by the input covariance so the compensation is aimed at the
directions the data actually occupies. This listing implements it and measures
what it is worth against plain rounding at the same bit-width.
"""
import numpy as np

rng = np.random.default_rng(257)

D_IN, D_OUT, N = 256, 128, 3000


def rtn(W, bits, group=0):
    """Round to nearest, per group of input dimensions."""
    qmax = 2 ** (bits - 1) - 1
    Q = np.empty_like(W)
    g = group if group > 0 else W.shape[0]
    for a in range(0, W.shape[0], g):
        blk = W[a:a + g]
        s = np.maximum(np.max(np.abs(blk)) / qmax, 1e-12)
        Q[a:a + g] = np.clip(np.round(blk / s), -qmax, qmax) * s
    return Q


def gptq(W, H, bits, group=0, damp=0.01):
    """cite:frantar2023gptq. Quantize input dimensions in order; after each,
    subtract the induced error from the remaining ones through the inverse
    Hessian, so the compensation is aimed where the data has variance."""
    d = W.shape[0]
    qmax = 2 ** (bits - 1) - 1
    Hd = H + damp * np.mean(np.diag(H)) * np.eye(d)
    U = np.linalg.cholesky(np.linalg.inv(Hd)).T      # upper triangular
    Wk = W.copy()
    Q = np.empty_like(W)
    g = group if group > 0 else d
    for a in range(0, d, g):
        blk = Wk[a:a + g]
        s = np.maximum(np.max(np.abs(blk)) / qmax, 1e-12)
        for i in range(a, min(a + g, d)):
            w = Wk[i]
            q = np.clip(np.round(w / s), -qmax, qmax) * s
            Q[i] = q
            e = (w - q) / U[i, i]
            if i + 1 < d:
                Wk[i + 1:] -= np.outer(U[i, i + 1:], e)
    return Q


# A layer whose input covariance is far from isotropic, as real activations are.
A = rng.normal(size=(D_IN, D_IN))
COV = A @ A.T / D_IN + 0.05 * np.eye(D_IN)
L = np.linalg.cholesky(COV)
X = rng.normal(size=(N, D_IN)) @ L.T
hot = rng.choice(D_IN, size=5, replace=False)
X[:, hot] *= 10.0
H = X.T @ X / N

W = rng.normal(size=(D_IN, D_OUT)) / np.sqrt(D_IN)
REF = X @ W


def out_err(Q):
    return float(np.linalg.norm(X @ Q - REF) / np.linalg.norm(REF))


def wgt_err(Q):
    return float(np.linalg.norm(Q - W) / np.linalg.norm(W))


print(f"A {D_IN}x{D_OUT} layer, correlated non-isotropic inputs with 5 outlier")
print("channels. Both metrics reported, because they rank the methods "
      "differently.")
print()
print(f"{'bits':>6}{'group':>8}" + f"{'WEIGHT error':>26}"
      + f"{'OUTPUT error':>28}")
print(f"{'':>6}{'':>8}{'RTN':>12}{'GPTQ':>14}{'RTN':>14}{'GPTQ':>14}")
print("-" * 68)

rows = {}
for bits in (8, 4, 3):
    for group in (0, 64):
        qr, qg = rtn(W, bits, group), gptq(W, H, bits, group)
        r = (wgt_err(qr), wgt_err(qg), out_err(qr), out_err(qg))
        rows[(bits, group)] = r
        lbl = "tensor" if group == 0 else str(group)
        print(f"{bits:>6}{lbl:>8}{r[0]:>12.4f}{r[1]:>14.4f}"
              f"{r[2]:>14.4f}{r[3]:>14.4f}")

print()
print()
print("Does the compensation depend on the calibration data being right?")
print()
print(f"{'calibration set':>28}{'output error':>15}{'vs correct':>13}")
print("-" * 56)

Xw = rng.normal(size=(N, D_IN))                        # isotropic: wrong shape
Xs = X[:64]                                            # too few samples
Xn = X + 0.5 * np.std(X) * rng.normal(size=X.shape)     # noisy, right shape

CAL = [
    ("the true input covariance", out_err(gptq(W, H, 4, 64))),
    ("isotropic (wrong shape)", out_err(gptq(W, Xw.T @ Xw / N, 4, 64))),
    ("64 samples of the real data",
     out_err(gptq(W, Xs.T @ Xs / len(Xs), 4, 64))),
    ("noisy copy of the real data", out_err(gptq(W, Xn.T @ Xn / N, 4, 64))),
    ("RTN, no calibration at all", out_err(rtn(W, 4, 64))),
]
correct = CAL[0][1]
cal = dict(CAL)
for name, v in CAL:
    print(f"{name:>28}{v:>15.4f}{v / correct:>12.2f}x")

t8, t4, t3 = rows[(8, 0)], rows[(4, 0)], rows[(3, 0)]
g4 = rows[(4, 64)]
print(f"""
Read the two metric blocks against each other, because the disagreement is the
whole point.

On WEIGHT error, GPTQ is worse than round-to-nearest at every setting:
{t4[1]:.4f} against {t4[0]:.4f} at 4 bits. That is not a bug and not a close
call. Round-to-nearest MINIMISES weight error by construction, so nothing can
beat it there, and any method that beats it on something else must be worse here.

On OUTPUT error, which is what the layer is for, the ranking reverses:
{t4[3]:.4f} against RTN's {t4[2]:.4f} at 4 bits, better by {t4[2]/t4[3]:.2f}x. At
3 bits, {t3[3]:.4f} against {t3[2]:.4f}. GPTQ is deliberately choosing worse
weights in order to get a better function (eq:compensate-not-round).

The mechanism is easy to describe and easy to misremember. After rounding weight
i downward, the layer's output is slightly too small in a particular direction.
The weights not yet quantized can be nudged upward to put it back -- and the
inverse Hessian says how much of the nudge each should absorb, as a function of
how much variance the input data has in each direction. Weights multiplying
high-variance inputs get small corrections because their effect is large; weights
multiplying directions the data barely occupies get large ones, because there they
are nearly free.

Which makes the calibration data load-bearing rather than a formality, and the
second table shows how much.

With the true input covariance, GPTQ reaches {correct:.4f}. With a noisy copy of
the real data -- substantial noise added, correlation structure preserved --
{cal['noisy copy of the real data']:.4f}, only {cal['noisy copy of the real data']/correct:.2f}x
worse. So the method needs the SHAPE of the input distribution and tolerates a
great deal of noise in it, which matches the practical folklore that a few hundred
calibration sequences suffice.

Now the two failure rows, which are the reason to run this experiment. With an
isotropic covariance -- right size, wrong shape -- GPTQ reaches
{cal['isotropic (wrong shape)']:.4f}. With only 64 samples of the real data,
{cal['64 samples of the real data']:.4f}. And plain round-to-nearest, with no
calibration at all, reaches {cal['RTN, no calibration at all']:.4f}.

Read those three together. **Mis-calibrated GPTQ is worse than no GPTQ.** The
compensation is aimed at directions the data does not occupy, so it is spending
its rounding freedom to fix errors that do not matter while creating ones that
do. An algorithm that improves on RTN by {t4[2]/t4[3]:.2f}x with correct
calibration is {cal['isotropic (wrong shape)']/cal['RTN, no calibration at all']:.2f}x
worse than RTN with the wrong kind.

That is the practical warning the method comes with and rarely carries. A
calibration set drawn from the wrong distribution -- generic web text for a model
serving code, English for a model serving another language, short sequences for a
long-context deployment -- is not a slightly worse calibration set. It can be an
actively harmful one, and nothing in the output of the quantization step says so.

Compare the group column while you are here. Group 64 improves both methods at 8
and 4 bits, and the gains compose: GPTQ at group 64 reaches {g4[3]:.4f} against
tensor-wide GPTQ's {t4[3]:.4f} and group-64 RTN's {rows[(4, 64)][2]:.4f}. Error
compensation and finer scales solve different parts of the problem -- one chooses
better rounding directions, the other reduces the step those roundings work with
-- so applying both is not redundant.

At 3 bits the composition breaks down: GPTQ at group 64 gives
{rows[(3, 64)][3]:.4f} against tensor-wide GPTQ's {t3[3]:.4f}. Resetting the scale
every 64 weights leaves less room for the error propagation to work with, and at
3 bits there is not enough headroom to spare. Worth noting as a reminder that
these techniques interact, and that "apply both" is a default rather than a
theorem.

The general lesson outlasts the algorithm. Every quantizer before this one asked
"what is the closest representable value"; this one asks "what assignment of
representable values minimises the error in the thing I care about". Those
questions have different answers, and the second is the one that was always
meant. The cost is that it needs to know what you care about -- which means
calibration data, which means a hyperparameter that is rarely reported and
occasionally decisive.""")

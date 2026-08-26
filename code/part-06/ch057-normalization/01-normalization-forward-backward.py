# -*- coding: utf-8 -*-
# Extracted from: Chapter 57 — Normalization: Batch, Layer, and RMSNorm
# Source: src/.../ch057-normalization.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Batch, layer and RMS normalisation implemented from their equations, with
the backward pass verified numerically.
"""
import numpy as np

rng = np.random.default_rng(0)


# --- forward passes ---------------------------------------------------------
def batchnorm_forward(x, gamma, beta, eps=1e-5):
    """Eqs. 57.1-57.2. Reduces over the BATCH axis, per feature."""
    mu = x.mean(axis=0)
    var = x.var(axis=0)
    xhat = (x - mu) / np.sqrt(var + eps)
    return gamma * xhat + beta, (xhat, np.sqrt(var + eps), gamma)


def layernorm_forward(x, gamma, beta, eps=1e-5):
    """Eqs. 57.4-57.5. Reduces over the FEATURE axis, per example."""
    mu = x.mean(axis=1, keepdims=True)
    var = x.var(axis=1, keepdims=True)
    xhat = (x - mu) / np.sqrt(var + eps)
    return gamma * xhat + beta, (xhat, np.sqrt(var + eps), gamma)


def rmsnorm_forward(x, gamma, eps=1e-5):
    """Eq. 57.6. No mean, no beta."""
    rms = np.sqrt((x ** 2).mean(axis=1, keepdims=True) + eps)
    return gamma * (x / rms), (x, rms, gamma)


def batchnorm_backward(dy, cache):
    """Eq. 57.9 — the three-term collapse."""
    xhat, s, gamma = cache
    B = len(dy)
    dgamma = (dy * xhat).sum(axis=0)
    dbeta = dy.sum(axis=0)
    dxhat = dy * gamma
    dx = (B * dxhat - dxhat.sum(axis=0)
          - xhat * (dxhat * xhat).sum(axis=0)) / (B * s)
    return dx, dgamma, dbeta


def layernorm_backward(dy, cache):
    xhat, s, gamma = cache
    d = xhat.shape[1]
    dgamma = (dy * xhat).sum(axis=0)
    dbeta = dy.sum(axis=0)
    dxhat = dy * gamma
    dx = (d * dxhat - dxhat.sum(axis=1, keepdims=True)
          - xhat * (dxhat * xhat).sum(axis=1, keepdims=True)) / (d * s)
    return dx, dgamma, dbeta


# --- verify against central differences -------------------------------------
print("=" * 72)
print("the backward passes, verified (eq. 57.9)")
print("=" * 72)
B, D = 8, 5
x = rng.normal(size=(B, D)) * 2 + 1
gamma = rng.normal(1.0, 0.2, D)
beta = rng.normal(0.0, 0.2, D)
w_out = rng.normal(size=(B, D))          # arbitrary downstream gradient


def check(fwd, bwd, label):
    y, cache = fwd(x, gamma, beta)
    dy = w_out
    dx, dgamma, dbeta = bwd(dy, cache)
    num = np.zeros_like(x)
    e = 1e-6
    for i in range(B):
        for j in range(D):
            xp, xm = x.copy(), x.copy()
            xp[i, j] += e
            xm[i, j] -= e
            num[i, j] = ((fwd(xp, gamma, beta)[0] * dy).sum()
                         - (fwd(xm, gamma, beta)[0] * dy).sum()) / (2 * e)
    rel = np.max(np.abs(dx - num) / np.maximum(np.abs(num), 1e-8))
    print(f"{label:<16} max relative error in dx: {rel:.3e}")
    return dx


dx_bn = check(batchnorm_forward, batchnorm_backward, "batchnorm")
dx_ln = check(layernorm_forward, layernorm_backward, "layernorm")

# --- section 6.1: the gradients are forced to sum to zero -------------------
print("\n" + "=" * 72)
print("what the backward pass CONSTRAINS (section 6.1)")
print("=" * 72)
print("Eq. 57.9's second and third terms remove the mean of the gradient")
print("and its component along xhat. So both must vanish exactly.\n")
xhat_bn = batchnorm_forward(x, gamma, beta)[1][0]
xhat_ln = layernorm_forward(x, gamma, beta)[1][0]
print(f"batchnorm: sum of dx over the BATCH axis, per feature")
print(f"  max |sum_i dx_ij|            = {np.abs(dx_bn.sum(axis=0)).max():.3e}")
print(f"  max |sum_i dx_ij * xhat_ij|  = "
      f"{np.abs((dx_bn * xhat_bn).sum(axis=0)).max():.3e}")
print(f"layernorm: sum of dx over the FEATURE axis, per example")
print(f"  max |sum_j dx_ij|            = "
      f"{np.abs(dx_ln.sum(axis=1)).max():.3e}")
print(f"  max |sum_j dx_ij * xhat_ij|  = "
      f"{np.abs((dx_ln * xhat_ln).sum(axis=1)).max():.3e}")

print("\nThe first constraint holds to machine precision. The second holds")
print("to about 1e-5, and the reason is the epsilon: eq. 57.2 divides by")
print("sqrt(var + eps), not by sqrt(var), so the layer is only")
print("scale-invariant up to that additive term. Set eps to zero and the")
print("second column drops to machine precision too:\n")
for label, ee in (("eps = 1e-5", 1e-5), ("eps = 0", 0.0)):
    y2, c2 = batchnorm_forward(x, gamma, beta, eps=ee)
    d2, _, _ = batchnorm_backward(w_out, c2)
    xh2 = c2[0]
    print(f"  {label:<12} max |sum_i dx| = {np.abs(d2.sum(axis=0)).max():.3e}"
          f"   max |sum_i dx*xhat| = "
          f"{np.abs((d2 * xh2).sum(axis=0)).max():.3e}")

print("\nThe constraints are not incidental. A gradient that would push")
print("every example in a batch the same way is projected out entirely,")
print("because it would only move the mean — which the normalisation")
print("immediately removes.")
print("\nThis is the most concrete formal statement available of how")
print("normalisation changes optimisation: it restricts the gradient to a")
print("subspace, and the two removed directions are exactly the two")
print("statistics the layer controls.")

# --- section 6.2: scale invariance ------------------------------------------
print("\n" + "=" * 72)
print("the weights feeding a normalisation are scale-invariant (eq. 57.10)")
print("=" * 72)
h = rng.normal(size=(32, 12))
W = rng.normal(0, 0.5, (12, 6))
g6, b6 = np.ones(6), np.zeros(6)
print(f"{'weight scale a':>15} {'eps = 1e-5':>16} {'eps = 0':>16}")
for a in (0.001, 0.01, 0.5, 1.0, 2.0, 100.0, 10000.0):
    row = []
    for ee in (1e-5, 0.0):
        base = batchnorm_forward(h @ W, g6, b6, eps=ee)[0]
        out = batchnorm_forward(h @ (a * W), g6, b6, eps=ee)[0]
        row.append(np.abs(out - base).max())
    print(f"{a:>15g} {row[0]:>16.3e} {row[1]:>16.3e}")

print("\nWith eps = 0 the invariance of eq. 57.10 is exact to floating")
print("point across seven orders of magnitude of weight scale. That is why")
print("Chapter 56's careful initialisation scale stops mattering in a")
print("normalised network: the scale is removed before the next layer sees")
print("it.")
print("\nWith the standard eps = 1e-5 the invariance is exact for large")
print("scales and BREAKS DOWN for small ones, because eps is an additive")
print("term in the denominator and stops being negligible once the")
print("variance falls to its order. At a = 0.001 the variance is a")
print("millionth of its original value and eps dominates entirely.")
print("\nThat is a real and easily missed limitation. Normalisation")
print("protects you from a badly scaled initialisation in one direction")
print("only: too large is absorbed exactly, and too SMALL runs into the")
print("epsilon floor and is not.")

# --- the effective learning rate consequence --------------------------------
print("\n" + "=" * 72)
print("the surprising consequence: an automatic learning-rate decay (6.2)")
print("=" * 72)
print("If the function is invariant to |W|, the gradient must be")
print("ORTHOGONAL to W. A step then grows |W| by Pythagoras, and since the")
print("effective step scales as 1/|W|^2, it shrinks by itself.\n")


def grad_wrt_W(W, h, target, eps=0.0):
    """Gradient of 0.5*||BN(hW) - target||^2 with respect to W.

    eps = 0 here deliberately: the invariance of eq. 57.10 is exact only
    without the epsilon, and this experiment is about that exact geometry.
    """
    z = h @ W
    y, cache = batchnorm_forward(z, np.ones(W.shape[1]),
                                 np.zeros(W.shape[1]), eps=eps)
    dy = y - target
    dz, _, _ = batchnorm_backward(dy, cache)
    return h.T @ dz


# A FRESH minibatch each step, so the gradient never decays to zero and the
# norm growth is not confounded with convergence.
rs_sgd = np.random.default_rng(4)
H_pool = rng.normal(size=(4096, 12))
T_pool = rng.normal(size=(4096, 6)) * 0.3
Wc = W.copy()
print(f"{'step':>6} {'|W|':>10} {'|grad|':>12} "
      f"{'|cos(W, grad)|':>16} {'eta*|g| / |W|':>15}")
lr = 0.05
for t in range(1, 4001):
    idx = rs_sgd.integers(0, len(H_pool), 32)
    g = grad_wrt_W(Wc, H_pool[idx], T_pool[idx])
    if t in (1, 10, 100, 500, 2000, 4000):
        cos = abs(float((Wc.ravel() @ g.ravel())
                        / (np.linalg.norm(Wc) * np.linalg.norm(g) + 1e-30)))
        print(f"{t:>6} {np.linalg.norm(Wc):>10.4f} "
              f"{np.linalg.norm(g):>12.3e} {cos:>16.3e} "
              f"{lr * np.linalg.norm(g) / np.linalg.norm(Wc):>15.3e}")
    Wc = Wc - lr * g

print("\nThe cosine between W and its gradient is zero to machine precision")
print("at every step. That is not a coincidence — eq. 57.10 says the loss")
print("does not change along the radial direction, so the derivative along")
print("it must vanish identically.")
print("\nEvery step therefore adds to |W| by Pythagoras and nothing ever")
print("subtracts, so the norm grows monotonically. The last column is the")
print("relative step size, and it falls as a direct consequence.")
print("\nNo schedule was applied here and the gradient is a fresh minibatch")
print("every step, so it is not decaying because the problem is being")
print("solved. The decay is produced entirely by the geometry of eq. 57.10.")
print("\nThis is the mechanism behind Chapter 58's account of weight decay")
print("in a normalised network: decay is what stops |W| from growing")
print("forever, and therefore what stops the effective learning rate from")
print("falling to zero on its own.")

# --- RMSNorm vs LayerNorm ---------------------------------------------------
print("\n" + "=" * 72)
print("RMSNorm and LayerNorm agree on centred input and not otherwise")
print("=" * 72)
print("They coincide when each ROW's mean is zero. The distribution's mean")
print("being zero is not enough: with d features the empirical row mean has")
print("standard deviation 1/sqrt(d), so at finite width the rows are not")
print("centred and the two differ.\n")
print(f"{'width d':>9} {'input mean':>12} {'typical row mean':>18} "
      f"{'mean |LN - RMS|':>18}")
for d in (8, 32, 128, 1024):
    for m in (0.0, 1.0):
        xx = rng.normal(size=(256, d)) + m
        lnv = layernorm_forward(xx, np.ones(d), np.zeros(d))[0]
        rnv = rmsnorm_forward(xx, np.ones(d))[0]
        print(f"{d:>9} {m:>12.1f} "
              f"{float(np.abs(xx.mean(axis=1)).mean()):>18.4f} "
              f"{float(np.abs(lnv - rnv).mean()):>18.4f}")

print("\nRead the zero-mean rows down the table: the discrepancy shrinks as")
print("1/sqrt(d), exactly tracking the row-mean column, and by width 1024")
print("it is negligible. That is the regime a transformer operates in.")
print("\nThe rows with an input mean of 1.0 do not shrink with width at")
print("all. A genuine offset in the data is not averaged away by making")
print("the layer wider, and RMSNorm's denominator absorbs it into the")
print("second moment, shrinking every output toward zero.")
print("\nSo the claim that RMSNorm works as well as LayerNorm is a")
print("substantive empirical claim about trained networks: their")
print("activations must be close enough to centred, at a width large")
print("enough, that the mean subtraction has nothing to remove. It is not")
print("a mathematical identity, and it would fail on data with a")
print("systematic offset.")

# -*- coding: utf-8 -*-
# Extracted from: Chapter 119 — Image Classification and the ResNet Lineage
# Source: src/.../ch119-classification.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Why the identity is hard for a plain stack and free for a residual one.

The previous listing showed degradation and showed that gradient magnitude does
not fully explain it. This one isolates the mechanism by giving both
architectures the easiest possible task: reproduce your input.

The identity is the function eq:identity-embedding says a deep network must be
able to express in order for depth to be harmless. Measuring how hard each
architecture finds it separates "can represent" from "can find", which is the
distinction the degradation result turns on.

One detail makes the comparison fair. A ReLU layer computes max(Wh, 0), which
equals h at W = I only when h is non-negative -- so the inputs here are
non-negative, exactly as they would be inside a network after the first
activation. Both architectures can therefore represent the identity EXACTLY, and
any difference in the result is the optimiser's.
"""
import numpy as np

WIDTH = 48
DEPTHS = (2, 4, 8, 16, 32)
N = 384
STEPS, LR = 600, 0.01


def stack_forward(X, Ws, residual, scale):
    h, hs, zs = X, [], []
    for W in Ws:
        hs.append(h)
        z = h @ W
        zs.append(z)
        a = np.maximum(z, 0)
        h = h + scale * a if residual else a
    return h, hs, zs


def init(depth, g):
    return [g.normal(scale=np.sqrt(2 / WIDTH), size=(WIDTH, WIDTH))
            for _ in range(depth)]


def signal_survival(residual, depth, trials=10):
    """At INITIALISATION, how much of the input is still present in the output?

    Mean absolute correlation between an input coordinate and the matching
    output coordinate -- a direct measure of whether the stack has destroyed its
    input before training even starts (eq:signal-decay, eq:signal-survival).
    """
    out = []
    for t in range(trials):
        g = np.random.default_rng(100 + t)
        X = np.abs(g.normal(size=(N, WIDTH)))
        scale = 1.0 / np.sqrt(depth) if residual else 1.0
        Y, _, _ = stack_forward(X, init(depth, g), residual, scale)
        Xc, Yc = X - X.mean(0), Y - Y.mean(0)
        den = np.sqrt((Xc ** 2).sum(0) * (Yc ** 2).sum(0)) + 1e-12
        out.append(float(np.abs((Xc * Yc).sum(0) / den).mean()))
    return float(np.mean(out))


def learn_identity(residual, depth, seed=5):
    """Train the stack to output its input, under an identical budget for both.
    Reported as relative error, so 1.0 means 'no better than predicting zero'.

    Adam rather than plain SGD, deliberately. Gradient magnitudes differ by
    orders of magnitude across these depths, so a single fixed learning rate
    would be testing step-size tuning rather than architecture. An adaptive
    optimiser removes that confound -- and makes the result stronger, since the
    plain stack fails even when the optimiser is choosing its own scale.
    """
    g = np.random.default_rng(seed)
    X = np.abs(g.normal(size=(N, WIDTH)))
    Ws = init(depth, g)
    scale = 1.0 / np.sqrt(depth) if residual else 1.0
    m = [np.zeros_like(W) for W in Ws]
    v = [np.zeros_like(W) for W in Ws]
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, STEPS + 1):
        Y, hs, zs = stack_forward(X, Ws, residual, scale)
        gh = 2.0 * (Y - X) / N
        grads = [None] * depth
        for i in reversed(range(depth)):
            ga = gh * scale if residual else gh
            gz = ga * (zs[i] > 0)
            grads[i] = hs[i].T @ gz
            # Backpropagate through the ORIGINAL weights before updating them.
            gh = gz @ Ws[i].T + (gh if residual else 0)
        for i in range(depth):
            m[i] = b1 * m[i] + (1 - b1) * grads[i]
            v[i] = b2 * v[i] + (1 - b2) * grads[i] ** 2
            mh = m[i] / (1 - b1 ** t)
            vh = v[i] / (1 - b2 ** t)
            Ws[i] -= LR * mh / (np.sqrt(vh) + eps)
    Y, _, _ = stack_forward(X, Ws, residual, scale)
    return float(np.linalg.norm(Y - X) / np.linalg.norm(X))


print(f"width {WIDTH}, {STEPS} gradient steps, identical budget for both.")
print("Inputs are non-negative, so BOTH architectures can represent the "
      "identity exactly.\n")
print(f"{'depth':>7}{'signal at init':>28}{'':>4}"
      f"{'identity error after training':>32}")
print(f"{'':>7}{'plain':>14}{'residual':>14}{'':>4}{'plain':>16}{'residual':>16}")
print("-" * 76)

res = {}
for d in DEPTHS:
    sp, sr = signal_survival(False, d), signal_survival(True, d)
    ip, ir = learn_identity(False, d), learn_identity(True, d)
    res[d] = (sp, sr, ip, ir)
    print(f"{d:>7}{sp:>14.4f}{sr:>14.4f}{'':>4}{ip:>16.4f}{ir:>16.4f}")

d0, dN = DEPTHS[0], DEPTHS[-1]
print(f"""
The left pair is measured before a single gradient step, and it is the cleaner of
the two results. In a plain stack the correlation between an input coordinate and
the matching output coordinate decays with depth -- {res[d0][0]:.4f} at depth
{d0} down to {res[dN][0]:.4f} at depth {dN}. Each layer mixes and rectifies, and
after enough layers the output retains almost no coordinate-wise trace of the
input.

The residual stack starts far higher and decays far more slowly:
{res[d0][1]:.4f} at depth {d0}, {res[dN][1]:.4f} at depth {dN}. The reason is
structural rather than statistical -- the identity path is a TERM IN THE SUM, so
whatever the branches compute, the input is still literally present in the
output. A residual network at initialisation is the identity plus noise; a plain
network at initialisation is noise.

That reframes what depth costs. A plain deep network has to LEARN to preserve
information that its own architecture destroyed by construction. Its optimiser
starts from a function that has thrown the input away, and every useful solution
begins by rebuilding a path back to it.

The right pair puts a number on the consequence, and the comparison is a fair one
because the inputs are non-negative: max(Ih, 0) = h, so a plain stack CAN be the
identity, with W = I at every layer. Asked to find it under a fixed budget, the
plain stack sits at {res[dN][2]:.4f} at depth {dN}, having got worse at every
step down the table, while the residual stack reaches {res[dN][3]:.4f} -- a
factor of {res[dN][2] / res[dN][3]:.1f} between them. Note the plain stack's
depth-2 figure, {res[2][2]:.4f}: it does not find the identity even when there
are only two layers to coordinate.

Nothing separates them but where they start. The residual stack reaches the
identity by driving its branch weights toward ZERO, and zero is where
initialisation already is and where weight decay pulls
(eq:identity-is-default). The plain stack has to find a specific, structured
matrix at every one of {dN} layers simultaneously, through {dN} layers of
rectification, with a gradient that has crossed all of them.

So the two architectures are not competing on capacity -- eq:identity-embedding
holds for both here, by construction. They are competing on where they start and
on what the optimiser must undo. That is why the fix is a wire rather than a
layer, why it costs no parameters, and why it turned out to matter for
transformers and diffusion models too, none of which are convolutional and all of
which are residual.""")

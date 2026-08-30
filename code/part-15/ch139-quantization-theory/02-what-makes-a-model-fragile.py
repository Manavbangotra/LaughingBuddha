# -*- coding: utf-8 -*-
# Extracted from: Chapter 139 — Why Quantization Works: Theory and Error Analysis
# Source: src/.../ch139-quantization-theory.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""What makes a model fragile to quantization? Two answers, and one is a surprise.

The previous listing showed that quantization damage is governed by the per-layer
error rather than by depth. This one asks what governs the per-layer error, and
finds two things.

The first is how finely the scale factors are shared -- the group size, which is
the parameter ch:q-formats identified as more consequential than the bit-width
everyone quotes (eq:group-size-dominates).

The second is not about storage at all. cite:kumar2024precisionscaling reports
that post-training quantization damage INCREASES with the amount of pretraining
data, so a better-trained model is a MORE fragile one. That inverts the usual
intuition, and it is testable directly: train the same model for different lengths
and quantize each checkpoint identically.
"""
import numpy as np

rng = np.random.default_rng(241)


def quantize(W, bits, group=0):
    """Symmetric integer quantization. `group` is how many consecutive weights
    share one scale factor; 0 means the whole tensor shares one."""
    qmax = 2 ** (bits - 1) - 1
    flat = W.reshape(-1)
    if group <= 0 or group >= flat.size:
        s = np.max(np.abs(flat)) / qmax
        s = max(s, 1e-12)
        return (np.clip(np.round(flat / s), -qmax, qmax) * s).reshape(W.shape)
    pad = (-flat.size) % group
    f = np.concatenate([flat, np.zeros(pad)])
    g = f.reshape(-1, group)
    s = np.maximum(np.max(np.abs(g), axis=1, keepdims=True) / qmax, 1e-12)
    q = (np.clip(np.round(g / s), -qmax, qmax) * s).reshape(-1)
    return q[:flat.size].reshape(W.shape)


D = 384
W = rng.normal(size=(D, D)) / np.sqrt(D)
W_OUT = W.copy()
cols = rng.choice(D, size=max(1, D // 50), replace=False)
W_OUT[:, cols] *= 16.0                      # 2% of columns, 16x larger

GROUPS = (0, 1024, 256, 64, 32)
LABEL = {0: "whole tensor", 1024: "1024", 256: "256", 64: "64", 32: "32"}
BITS = (8, 6, 4, 3)


def rel(A, B):
    return float(np.linalg.norm(A - B) / np.linalg.norm(B))


print(f"A {D}x{D} weight matrix. Relative error against group size and bits.")
print("Left block is a clean Gaussian; right block has 2% of columns 16x larger.")
print()
print(f"{'group':>14}" + "".join(f"{str(b) + 'b':>10}" for b in BITS)
      + f"{'':>4}" + "".join(f"{str(b) + 'b':>10}" for b in BITS))
print(f"{'size':>14}" + f"{'CLEAN':>40}" + f"{'':>4}" + f"{'WITH OUTLIERS':>40}")
print("-" * 88)

tab = {}
for g in GROUPS:
    c = [rel(quantize(W, b, g), W) for b in BITS]
    o = [rel(quantize(W_OUT, b, g), W_OUT) for b in BITS]
    tab[g] = (c, o)
    print(f"{LABEL[g]:>14}" + "".join(f"{v:>10.4f}" for v in c) + f"{'':>4}"
          + "".join(f"{v:>10.4f}" for v in o))

print()
print()
print("Now the second question: does training a model LONGER make it more")
print("fragile? Same architecture, same quantization, different checkpoints.")
print()

DI, H, DO_, N = 24, 64, 6, 2000
Wq = rng.normal(size=(DI, DO_))
Xtr = rng.normal(size=(N, DI))
Ytr = (np.tanh(Xtr @ Wq) + 0.4 * Xtr[:, :DO_]
       + 0.10 * rng.normal(size=(N, DO_)))
Xte = rng.normal(size=(2500, DI))
Yte = np.tanh(Xte @ Wq) + 0.4 * Xte[:, :DO_]


def init():
    return [rng.normal(size=(DI, H)) / np.sqrt(DI), np.zeros(H),
            rng.normal(size=(H, DO_)) / np.sqrt(H), np.zeros(DO_)]


def fwd(p, X):
    h = np.tanh(X @ p[0] + p[1])
    return h, h @ p[2] + p[3]


def mse(p, X, Y):
    return float(((fwd(p, X)[1] - Y) ** 2).mean())


def step(p, m, v, t, lr=0.01):
    h, o = fwd(p, Xtr)
    d = 2 * (o - Ytr) / N
    dh = d @ p[2].T * (1 - h ** 2)
    g = [Xtr.T @ dh, dh.sum(0), h.T @ d, d.sum(0)]
    for i in range(4):
        m[i] = 0.9 * m[i] + 0.1 * g[i]
        v[i] = 0.999 * v[i] + 0.001 * g[i] ** 2
        p[i] -= lr * (m[i] / (1 - 0.9 ** t)) / (
            np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)


print(f"{'steps':>8}{'train':>10}{'test':>10}{'||W||':>9}"
      f"{'test after':>12}{'damage':>10}{'damage':>10}")
print(f"{'':>8}{'loss':>10}{'loss':>10}{'':>9}{'4-bit':>12}{'absolute':>10}"
      f"{'relative':>10}")
print("-" * 69)

p = init()
m = [np.zeros_like(w) for w in p]
v = [np.zeros_like(w) for w in p]
rows = {}
t = 0
for target in (50, 200, 800, 3200, 9600):
    while t < target:
        t += 1
        step(p, m, v, t)
    tr, te = mse(p, Xtr, Ytr), mse(p, Xte, Yte)
    nrm = float(np.sqrt(sum((w ** 2).sum() for w in p)))
    pq = [quantize(w, 4, 64) if w.ndim == 2 else w.copy() for w in p]
    teq = mse(pq, Xte, Yte)
    rows[target] = (tr, te, nrm, teq, teq - te, teq / te - 1)
    print(f"{target:>8}{tr:>10.4f}{te:>10.4f}{nrm:>9.2f}{teq:>12.4f}"
          f"{teq - te:>+10.4f}{teq / te - 1:>+10.1%}")

c, o = tab[0]
c64, o64 = tab[64]
c32, o32 = tab[32]
early, late = rows[50], rows[9600]
print(f"""
The first table makes the group-size point concrete, and it is best read by
comparing what a finer group buys against what a bit buys.

On the clean matrix at 4 bits, a whole-tensor scale gives {c[2]:.4f} and a group
of 64 gives {c64[2]:.4f} -- better by {c[2]/c64[2]:.1f}x, from one extra number
per 64 weights. What does a bit buy on the same matrix? Eight bits gives
{c[0]:.4f} against six bits' {c[1]:.4f}: a factor of {c[1]/c[0]:.1f} for two
bits, so roughly {(c[1]/c[0]) ** 0.5:.1f}x per bit.

So on clean weights, going from a tensor-wide scale to groups of 64 is worth
about a bit of width, and it costs a fraction of one -- a 16-bit scale per 64
weights is a quarter of a bit per weight, four times cheaper than the bit it
replaces.

The outlier block is where the argument becomes decisive rather than merely
favourable. At 8 bits the outliers cost {o[0]/c[0]:.1f}x at whole-tensor scale
({o[0]:.4f} against {c[0]:.4f}) and only {o32[0]/c32[0]:.1f}x at groups of 32
({o32[0]:.4f} against {c32[0]:.4f}).

That is the mechanism stated precisely. An outlier does damage by forcing a
shared scale factor to cover a range the other weights never use, which coarsens
the step for every weight sharing that scale. Shrink the group and the damage is
CONTAINED: the outlier still ruins its own group of 32, and every other group is
untouched (eq:group-size-dominates).

So the cost of an outlier is proportional to how many weights share its scale.
That is why every practical 4-bit format has a group size somewhere in its name,
and why quoting "4-bit" without it is an incomplete specification -- the number
people quote is the one that matters less.

The second table inverts an intuition, and it does so sharply.

The model is trained for progressively longer and quantized identically at each
checkpoint. Training works: test loss falls from {early[1]:.4f} at 50 steps to
{rows[800][1]:.4f} at 800. The question is what the same 4-bit quantization costs
at each point.

Absolute damage rises from {early[4]:+.4f} to {late[4]:+.4f} -- a factor of
{late[4]/early[4]:.0f} -- and relative damage from {early[5]:+.1%} to
{late[5]:+.1%}. The longer-trained model is hurt far more by an identical
operation. Both accountings agree, which matters: the relative figure alone could
be an artefact of a shrinking denominator, and the absolute figure rules that out.

The weight-norm column names the mechanism: {early[2]:.1f} at 50 steps rising to
{late[2]:.1f} at {max(rows)}. Training moves weights away from their small random
initialisation, so the tensor's dynamic range grows and a fixed number of levels
has more ground to cover. The same 4 bits are being asked to span a wider
interval, so each step is coarser.

Two honest caveats, because this is a small experiment standing in for a large
result. The weight growth here is unchecked -- there is no weight decay -- and
regularisation would slow it, which is a genuine partial mitigation that real
training runs already apply. And the later checkpoints are overfitting: test loss
bottoms at {rows[800][1]:.4f} around step 800 and rises after, so some of what the
last rows measure is a model that was already getting worse.

Neither caveat removes the effect. cite:kumar2024precisionscaling establishes it
properly at pretraining scale, without either confound, and finds it strong enough
that past some number of tokens additional pretraining makes the post-quantization
model WORSE. This listing shows the direction and the mechanism; the paper shows
the crossover.

The practical consequence is the uncomfortable one. A quantization recipe is
validated on a CHECKPOINT, not on an architecture. The same model family trained
longer may not survive the same recipe, so "we use 4-bit for this model" is a
claim with an expiry date, and the only way to know is to measure again when the
checkpoint changes.""")

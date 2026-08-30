# -*- coding: utf-8 -*-
# Extracted from: Chapter 141 — GGUF, llama.cpp, and Weight-Only Quantization
# Source: src/.../ch141-gguf-llamacpp.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Not every tensor deserves the same number of bits.

"4-bit quantization" almost always means every weight tensor gets four bits. That
is a convention, not a result. Layers differ in how much their errors matter --
by width, by position, by what multiplies them -- and a fixed budget spent
uniformly is a fixed budget spent badly.

The k-quant families in llama.cpp already act on this, giving more bits to some
tensors than others by rule of thumb. This listing does it by MEASUREMENT: derive
each tensor's sensitivity, allocate bits to equalise the marginal damage
(eq:bit-allocation), and compare against uniform at exactly the same average
bits.
"""
import numpy as np

rng = np.random.default_rng(263)

# Layers of deliberately different shapes and scales, as a real network has.
SHAPES = [(64, 256), (256, 256), (256, 512), (512, 256), (256, 256), (256, 64)]
# What actually differs between real weight tensors: outlier content and
# effective rank, not just size. KIND says which structure each layer has.
KIND = ["plain", "outliers", "plain", "lowrank", "outliers", "plain"]
N = 3000


def make_net():
    Ws = []
    for (a, b), kind in zip(SHAPES, KIND):
        W = rng.normal(size=(a, b)) / np.sqrt(a)
        if kind == "outliers":
            cols = rng.choice(b, size=max(1, b // 40), replace=False)
            W[:, cols] *= 20.0
        elif kind == "lowrank":
            r = max(4, min(a, b) // 16)
            U = rng.normal(size=(a, r)); V = rng.normal(size=(r, b))
            W = (U @ V) / np.sqrt(a * r)
        Ws.append(W)
    return Ws


def quantize(W, bits, group=64):
    qmax = 2 ** (bits - 1) - 1
    flat = W.reshape(-1)
    pad = (-flat.size) % group
    f = np.concatenate([flat, np.zeros(pad)]).reshape(-1, group)
    s = np.maximum(np.max(np.abs(f), axis=1, keepdims=True) / qmax, 1e-12)
    q = (np.clip(np.round(f / s), -qmax, qmax) * s).reshape(-1)
    return q[:flat.size].reshape(W.shape)


WS = make_net()
X = rng.normal(size=(N, SHAPES[0][0]))


def forward(Ws):
    h = X
    for i, W in enumerate(Ws):
        h = h @ W
        if i < len(Ws) - 1:
            h = np.tanh(h)
    return h


REF = forward(WS)


def err(Ws):
    return float(np.linalg.norm(forward(Ws) - REF) / np.linalg.norm(REF))


def with_bits(bits_per_layer):
    return [quantize(W, b) for W, b in zip(WS, bits_per_layer)]


SIZES = np.array([a * b for a, b in SHAPES], float)
B_REF = 6

# Sensitivity: quantize ONE layer at the reference width, see what it costs.
sens = []
for i in range(len(WS)):
    Ws = [W.copy() for W in WS]
    Ws[i] = quantize(Ws[i], B_REF)
    sens.append(err(Ws) ** 2)
sens = np.array(sens)

print(f"Per-layer sensitivity, measured by quantizing one layer at {B_REF} bits.")
print()
print(f"{'layer':>7}{'shape':>14}{'params':>10}{'structure':>11}{'error':>12}"
      f"{'relative':>11}")
print("-" * 65)
for i, ((a, b), k) in enumerate(zip(SHAPES, KIND)):
    print(f"{i:>7}{f'{a}x{b}':>14}{int(SIZES[i]):>10,}{k:>11}"
          f"{np.sqrt(sens[i]):>12.5f}{sens[i]/sens.min():>10.1f}x")


def allocate(avg_bits, lo=2, hi=8):
    """Greedy on the marginal return. Raising a layer from b to b+1 cuts its
    error contribution by three quarters and costs SIZES[i] bits, so spend each
    bit where that ratio is largest (eq:bit-allocation). Exact for a separable
    convex objective, which this is."""
    b = np.full(len(WS), float(lo))
    budget = avg_bits * SIZES.sum()
    spent = (b * SIZES).sum()
    while True:
        gain = np.where(b < hi, 0.75 * sens * 4.0 ** (-b) / SIZES, -1.0)
        i = int(np.argmax(gain))
        if gain[i] <= 0 or spent + SIZES[i] > budget:
            break
        b[i] += 1
        spent += SIZES[i]
    return b.astype(int)


def heuristic(avg_bits, lo=2, hi=8):
    """The common rule of thumb: spend extra on the first and last tensors,
    and take it back from the largest ones."""
    b = np.full(len(WS), float(avg_bits))
    b[0] = min(hi, b[0] + 2)
    b[-1] = min(hi, b[-1] + 2)
    budget = avg_bits * SIZES.sum()
    order = np.argsort(-SIZES)
    k = 0
    while (b * SIZES).sum() > budget and k < 10 * len(b):
        i = order[k % len(order)]
        if b[i] > lo:
            b[i] -= 1
        k += 1
    return b.astype(int)


print()
print()
print("Uniform against measured allocation, at exactly the same average bits.")
print()
print(f"{'avg bits':>9}{'uniform':>11}{'heuristic':>12}{'measured':>11}"
      f"{'gain':>8}{'   allocation'}")
print("-" * 78)

for avg in (3, 4, 5, 6):
    bu = np.full(len(WS), avg, int)
    bh = heuristic(avg)
    bm = allocate(avg)
    eu, eh, em = err(with_bits(bu)), err(with_bits(bh)), err(with_bits(bm))
    print(f"{avg:>9}{eu:>11.5f}{eh:>12.5f}{em:>11.5f}{eu/em:>7.2f}x"
          f"   {list(bm)}")

b4 = allocate(4)
e4u, e4m = err(with_bits(np.full(len(WS), 4, int))), err(with_bits(b4))
b3 = allocate(3)
e3u, e3m = err(with_bits(np.full(len(WS), 3, int))), err(with_bits(b3))
b5 = allocate(5)
e5u, e5m = err(with_bits(np.full(len(WS), 5, int))), err(with_bits(b5))
e4h = err(with_bits(heuristic(4)))
print(f"""
The sensitivity table is the first result and the spread is the point. The same
operation -- 6-bit quantization with groups of 64 -- applied to one layer at a
time produces errors differing by a factor of {sens.max()/sens.min():.0f} across
six layers of one small network.

And the reason is legible in the structure column. The two layers with outlier
weights are {sens.max()/sens.min():.0f}x and
{sorted(sens)[-2]/sens.min():.0f}x more sensitive than the least sensitive one.
The low-rank layer is barely above the plain ones. Size hardly matters: the
largest layer here is one of the least sensitive.

That is ch:q-theory's result in a new form. Sensitivity is driven by outlier
content, which sets how coarse the step must be, and not by how big the tensor is
or where it sits. Uniform bit allocation therefore spends the same budget on a
layer whose errors matter eighty times more and one whose errors barely register.

The allocation rule follows from the error model. Quantization error falls like
4^-b, so raising a layer by one bit cuts its contribution by three quarters and
costs its parameter count in storage. Spend each bit where that ratio is largest
and you get the optimum, exactly, because the objective is separable and convex
(eq:bit-allocation).

The comparison table has three things in it and one of them is a failure.

At an average of 4 bits, uniform gives {e4u:.5f} and measured allocation gives
{e4m:.5f} -- a factor of {e4u/e4m:.2f}. At 5 bits, {e5u/e5m:.2f}x. The 4-bit
allocation is {list(b4)}: six bits for each outlier-laden layer, three for the
large plain ones, four for the small plain ones. Not monotone in depth, not
monotone in size, and not something a rule would have produced.

The heuristic column is what actually ships in most quantized model files:
spend extra on the first and last tensors, take it back from the largest. At 4
bits it gives {e4h:.5f} against uniform's {e4u:.5f} -- essentially nothing.

That is worth dwelling on rather than skipping. The rule encodes a positional
correlation: first and last tensors are often smaller and often matter more. Here
the sensitive layers are in the MIDDLE, because sensitivity came from outlier
structure rather than from position, and the rule had no way to know. A heuristic
that tracks a correlate of the thing you care about works exactly as long as the
correlation holds, and fails silently when it does not.

Now the failure row, which is the most useful line in the table. At an average of
3 bits, allocation gives {e3m:.5f} against uniform's {e3u:.5f} -- it is
{e3u/e3m:.2f}x, meaning WORSE. The allocation was {list(b3)}: it took two layers
down to 2 bits to fund the outlier layers, and at 2 bits the error model that
justified the whole procedure has stopped being true.

ch:q-theory flagged this in advance: the 4^-b law comes from treating quantization
error as small uniform noise, and below about 3 bits the error is comparable to
the signal, not uniform, and correlated with the value. The allocation rule is
derived from a model, and it fails precisely where the model does -- which is a
better argument for knowing where a model applies than any amount of warning
about it.

So the practical shape. Bit allocation by measured sensitivity is worth roughly a
factor of two at average widths of four and above, it costs one forward pass per
tensor at quantization time, and it must be floored above the width where the
error model holds. It is a refinement on top of group size and outlier handling
rather than a substitute -- ch:q-theory measured outliers costing 16x at fixed
bit-width, and nothing here approaches that.

And it inherits ch:q-int8-int4's dependency: the sensitivity profile is measured
on data, so a profile from the wrong distribution allocates bits to the layers
that mattered for someone else's workload. The measurement is cheap; using the
right data for it is the part that requires attention.""")

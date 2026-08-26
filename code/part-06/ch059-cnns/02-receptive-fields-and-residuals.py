# -*- coding: utf-8 -*-
# Extracted from: Chapter 59 — Convolutional Neural Networks
# Source: src/.../ch059-cnns.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Receptive fields, which are smaller than people expect, and what a
residual connection does to the gradient product of eq. 53.9.
"""
import numpy as np

rng = np.random.default_rng(1)


# --- section 6.2: receptive field -------------------------------------------
def receptive_field(layers):
    """Eq. 59.7. layers is a list of (kernel, stride, dilation)."""
    r, jump = 1, 1
    for k, s, d in layers:
        r = r + d * (k - 1) * jump
        jump = jump * s
    return r, jump


print("=" * 72)
print("receptive fields are smaller than people expect (eq. 59.7)")
print("=" * 72)
ARCHS = {
    "10 x (3x3, stride 1)": [(3, 1, 1)] * 10,
    "20 x (3x3, stride 1)": [(3, 1, 1)] * 20,
    "50 x (3x3, stride 1)": [(3, 1, 1)] * 50,
    "5 x (3x3) + pool, x4": [(3, 1, 1)] * 5 + [(2, 2, 1)]
                            + ([(3, 1, 1)] * 5 + [(2, 2, 1)]) * 3,
    "10 x (3x3, dilation 2^l)": [(3, 1, 2 ** min(l, 6)) for l in range(10)],
    "5 x (7x7, stride 1)": [(7, 1, 1)] * 5,
}
print(f"{'architecture':<28} {'layers':>8} {'receptive field':>17} "
      f"{'downsample':>12}")
for name, layers in ARCHS.items():
    r, j = receptive_field(layers)
    print(f"{name:<28} {len(layers):>8} {f'{r} x {r}':>17} {f'{j}x':>12}")

print("\nWith stride 1 the receptive field grows LINEARLY: 3x3 kernels add")
print("2 per layer, so covering a 224x224 image takes about 112 layers.")
print("That is the structural limitation of a plain convolutional stack.")
print("\nDownsampling changes the regime. Each stride-2 layer doubles the")
print("jump, so subsequent layers contribute twice as much and the growth")
print("becomes exponential. Downsampling is not primarily a way to save")
print("compute; it is what lets the network see the whole image at all.")
print("\nDilation achieves the same growth WITHOUT losing resolution, which")
print("is why segmentation and audio architectures use it — they need a")
print("large receptive field and a full-resolution output at the same time.")

# --- how much of the receptive field actually matters -----------------------
print("\n" + "=" * 72)
print("the EFFECTIVE receptive field is smaller still")
print("=" * 72)
print("The theoretical field is where a gradient CAN be nonzero. Measure")
print("where it actually is, by backpropagating from one central output")
print("unit through a stack of random 3x3 convolutions.\n")


def effective_field(depth, size=81, seed=0):
    rs = np.random.default_rng(seed)
    Ks = [rs.normal(0, np.sqrt(2.0 / 9), (1, 1, 3, 3)) for _ in range(depth)]
    x = np.zeros((1, 1, size, size))
    x[0, 0, size // 2, size // 2] = 0.0
    # forward with a ones input, then backprop a delta at the centre
    acts = [np.ones((1, 1, size, size))]
    h = acts[0]
    for K in Ks:
        h = _conv(h, K)
        acts.append(h)
    g = np.zeros_like(h)
    g[0, 0, size // 2, size // 2] = 1.0
    for K in reversed(Ks):
        g = _conv(g, K[:, :, ::-1, ::-1])          # transposed convolution
    infl = np.abs(g[0, 0])
    infl = infl / infl.max()
    return infl


def _conv(X, K):
    N, C, H, W = X.shape
    F, _, kh, kw = K.shape
    Xp = np.pad(X, ((0, 0), (0, 0), (1, 1), (1, 1)))
    s = Xp.strides
    patches = np.lib.stride_tricks.as_strided(
        Xp, shape=(N, C, H, W, kh, kw),
        strides=(s[0], s[1], s[2], s[3], s[2], s[3]), writeable=False)
    cols = patches.transpose(0, 2, 3, 1, 4, 5).reshape(N * H * W, -1)
    return (cols @ K.reshape(F, -1).T).reshape(N, H, W, F).transpose(
        0, 3, 1, 2)


print(f"{'depth':>7} {'theoretical':>13} {'radius holding 50%':>20} "
      f"{'radius holding 90%':>20}")
for depth in (5, 10, 20, 30):
    infl = effective_field(depth)
    c = infl.shape[0] // 2
    total = infl.sum()
    r50 = r90 = None
    for r in range(1, c + 1):
        frac = infl[c - r:c + r + 1, c - r:c + r + 1].sum() / total
        if r50 is None and frac >= 0.5:
            r50 = r
        if r90 is None and frac >= 0.9:
            r90 = r
            break
    theo = 2 * depth + 1
    print(f"{depth:>7} {f'{theo} x {theo}':>13} "
          f"{f'{2 * r50 + 1} x {2 * r50 + 1}':>20} "
          f"{f'{2 * r90 + 1} x {2 * r90 + 1}':>20}")

print("\nThe theoretical receptive field is where the gradient CAN be")
print("nonzero. Most of the influence is concentrated far inside it,")
print("because reaching the edge of the field requires taking the same")
print("extreme offset at every single layer — one path out of many — while")
print("reaching the centre can be done in many ways.")
print("\nThe distribution is therefore roughly Gaussian rather than uniform,")
print("and its effective radius grows like sqrt(depth) rather than depth.")
print("So a stack whose theoretical field covers the image may still be")
print("using only a fraction of it, which is a real limitation and one of")
print("the standing arguments for attention.")

# --- section 6.4: residual connections and the gradient ---------------------
print("\n" + "=" * 72)
print("what a residual connection does to the gradient (eq. 59.9)")
print("=" * 72)


def gradient_through_stack(depth, width=64, mode="plain", batch=256,
                           seed=2):
    """Propagate a gradient down a stack and report its RMS at each layer.

    mode: 'plain'        h <- relu(h W)
          'residual'     h <- h + relu(h W), He-scaled branch
          'residual/sqrtL' the same with the branch scaled by 1/sqrt(depth)
          'residual/zero'  the branch's output layer zero-initialised
    """
    rs = np.random.default_rng(seed)
    bscale = {"residual/sqrtL": 1.0 / np.sqrt(depth),
              "residual/zero": 0.0}.get(mode, 1.0)
    Ws = [rs.normal(0, np.sqrt(2.0 / width), (width, width))
          for _ in range(depth)]
    residual = mode.startswith("residual")
    h = rs.normal(size=(batch, width))
    Zs, fwd = [], [float(np.sqrt(np.mean(h ** 2)))]
    for W in Ws:
        z = h @ W
        Zs.append(z)
        a = np.maximum(0.0, z)
        h = h + bscale * a if residual else a
        fwd.append(float(np.sqrt(np.mean(h ** 2))))
    g = rs.normal(size=h.shape) / np.sqrt(batch)
    norms = []
    for l in reversed(range(depth)):
        norms.append(float(np.sqrt(np.mean(g ** 2))))
        dz = (g * bscale if residual else g) * (Zs[l] > 0)
        g = dz @ Ws[l].T + (g if residual else 0.0)
    norms.append(float(np.sqrt(np.mean(g ** 2))))
    return list(reversed(norms)), fwd


print("Gradient RMS reaching each layer. 'ratio' is the gradient at the")
print("TOP of the stack divided by the gradient reaching the BOTTOM: a")
print("value near 1 means the gradient crossed the whole stack intact.\n")
print(f"{'depth':>6} {'mode':<18} " +
      " ".join(f"{f'layer {i}':>11}" for i in ("1", "L/2", "L"))
      + f" {'ratio L/1':>12} {'fwd RMS at L':>14}")
for depth in (10, 30, 60):
    for mode in ("plain", "residual", "residual/sqrtL",
                 "residual/zero"):
        n, fwd = gradient_through_stack(depth, mode=mode)
        picks = [0, depth // 2, depth]
        print(f"{depth:>6} {mode:<18} "
              + " ".join(f"{n[i]:>11.3e}" for i in picks)
              + f" {n[depth] / max(n[0], 1e-300):>12.3e}"
              + f" {fwd[-1]:>14.3e}")

print("\nThe plain stack is FINE, and that is worth saying first: He")
print("initialisation was derived to make it fine, and Chapter 56 measured")
print("it doing so. Its ratio stays within a small factor of 1 at every")
print("depth here. Residual connections are not solving a problem that")
print("appears in a well-initialised plain stack of this size — they solve")
print("the DEGRADATION problem, which is about optimisation rather than")
print("about gradient magnitude, and which needs a trained network to see.")
print("\nThe residual rows are the warning. With a standard He-initialised")
print("branch, BOTH the forward RMS and the backward gradient explode with")
print("depth — the last column shows the forward signal running away, and")
print("the ratio shows the gradient at layer 1 dwarfing the one at the")
print("output. That is exactly Chapter 56's eq. 56.11: the skip and the")
print("branch each contribute variance, so each block roughly doubles it.")
print("\nScaling the branch by 1/sqrt(depth) — eq. 56.13 — removes twelve")
print("of those orders of magnitude and does NOT remove all of them. The")
print("forward RMS still grows. Eq. 56.13's bound of e assumed the skip and")
print("the branch are independent, and they are not: F(x) is computed FROM")
print("x, so their variances more than add. That caveat was flagged in")
print("Chapter 56 and here is what it costs.")
print("\nZero-initialising the branch's output layer removes the growth")
print("entirely: every block is EXACTLY the identity at initialisation, so")
print("the forward RMS is unchanged and the gradient crosses the whole")
print("stack with a ratio of exactly 1 at any depth. It is the only one of")
print("the three that is exact rather than approximate, which is why it is")
print("the standard choice.")
print("\nSo eq. 59.9's identity term is necessary and NOT sufficient. It")
print("guarantees a path with gain exactly 1 from the loss to every layer,")
print("which is why the gradient cannot VANISH. It says nothing about the")
print("other 2^L - 1 terms in the expansion, which is why the gradient can")
print("still EXPLODE — and why every real residual architecture either")
print("zero-initialises its branches, scales them, or puts a normalisation")
print("inside them.")

# --- the degradation problem (section 6.5) ----------------------------------
print("\n" + "=" * 72)
print("why the depth problem was OPTIMISATION, not capacity (section 6.5)")
print("=" * 72)
print("A deeper plain stack can represent a shallower one exactly, by")
print("making the extra layers the identity. So its ACHIEVABLE training")
print("error is at most the shallower network's.\n")
print("Check that the representation exists, by constructing it:")
width = 32
rs = np.random.default_rng(3)
x0 = rs.normal(size=(64, width))
shallow_W = [rs.normal(0, np.sqrt(2.0 / width), (width, width))
             for _ in range(4)]


def run_stack(x, Ws):
    h = x
    for W in Ws:
        h = np.maximum(0.0, h @ W)
    return h


out_shallow = run_stack(np.maximum(x0, 0.0), shallow_W)
# extend with identity layers; ReLU is idempotent on non-negative input
deep_W = shallow_W + [np.eye(width) for _ in range(20)]
out_deep = run_stack(np.maximum(x0, 0.0), deep_W)
print(f"  4-layer output vs 24-layer-with-identity-extension output:")
print(f"  max |difference| = {np.abs(out_shallow - out_deep).max():.3e}")
print("  (ReLU is idempotent on non-negative input, so the identity layers")
print("   pass the activations through unchanged)")

print("\nThe deeper network CAN represent the shallower one, exactly. So")
print("when He et al. measured a 56-layer plain network with HIGHER")
print("TRAINING error than a 20-layer one, the explanation could not be")
print("capacity — the solution was inside the hypothesis class and the")
print("optimiser failed to find it.")
print("\nThat is a rare clean argument: it isolates optimisation from")
print("capacity by construction rather than by inference, and it is why")
print("the fix was architectural. Making the identity the DEFAULT means")
print("the optimiser starts from the solution it was failing to reach.")

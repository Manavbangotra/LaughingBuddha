# -*- coding: utf-8 -*-
# Extracted from: Chapter 56 — Initialization and Signal Propagation
# Source: src/.../ch056-initialization.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The variance calculation of section 6.1, measured layer by layer in
networks 50 deep.
"""
import numpy as np

rng = np.random.default_rng(0)


def propagate(depth, width, scheme, act="relu", batch=512, seed=0):
    """Forward pass through an untrained network, returning per-layer stats."""
    rs = np.random.default_rng(seed)
    h = rs.normal(size=(batch, width))
    stats = [(0, float(np.var(h)), 0.0)]
    for l in range(1, depth + 1):
        if scheme == "lecun":
            sd = np.sqrt(1.0 / width)
        elif scheme == "he":
            sd = np.sqrt(2.0 / width)
        elif scheme == "glorot":
            sd = np.sqrt(2.0 / (width + width))
        elif scheme == "too small":
            sd = np.sqrt(2.0 / width) * 0.7
        elif scheme == "too large":
            sd = np.sqrt(2.0 / width) * 1.4
        elif scheme == "unit":
            sd = 1.0
        W = rs.normal(0, sd, (width, width))
        z = h @ W
        h = np.maximum(0.0, z) if act == "relu" else np.tanh(z)
        stats.append((l, float(np.var(z)),
                      float((h == 0).mean()) if act == "relu" else 0.0))
    return stats


print("=" * 72)
print("forward variance through 50 layers (eq. 56.4)")
print("=" * 72)
print("ReLU network, width 256. Var[z] should stay at its initial value\n"
      "if the scheme is right for the activation.\n")
DEPTH, WIDTH = 50, 256
picks = [0, 1, 5, 10, 25, 50]
print(f"{'scheme':<14} " + " ".join(f"{f'L{i}':>11}" for i in picks)
      + f" {'ratio L50/L0':>14}")
for scheme in ("he", "lecun", "glorot", "too small", "too large"):
    st = propagate(DEPTH, WIDTH, scheme)
    vals = " ".join(f"{st[i][1]:>11.3e}" for i in picks)
    print(f"{scheme:<14} {vals} {st[-1][1] / st[0][1]:>14.3e}")

print("\nRead the last column first. He initialisation holds the variance")
print("within a small factor across fifty layers. Every other scheme moves")
print("it by fifteen orders of magnitude, in one direction or the other.")
print("\nLeCun and Glorot are numerically identical here because the layers")
print("are square — 2/(n+n) is 1/n — and both are missing the factor of two")
print("that eq. 56.5 says a rectifier needs. They are the right answer for a")
print("linear or saturating activation.")
print("\nAgainst section 6.3's prediction: the per-layer gain is")
print("Var[W]/Var[W]*, so LeCun's is 1/2 and after 50 layers that predicts")
print(f"{0.5 ** 50:.3e}, against a measured 3.7e-15 — the right order and a")
print("factor of four out. 'Too large' at 1.4x the standard deviation has")
print(f"gain 1.96 and predicts {1.96 ** 50:.3e}, which the measurement")
print("matches closely.")
print("\nThe He row drifts upward rather than staying flat, and the next")
print("experiment shows why: it is a finite-width effect, not an error in")
print("eq. 56.6.")
print("\nWhat is not in doubt is the sensitivity. A 30 per cent error in")
print("the standard deviation moved the variance by fifteen orders of")
print("magnitude over fifty layers, which is why this scalar is worth")
print("deriving rather than guessing.")

# --- is the He drift a finite-width effect? ---------------------------------
print("\n" + "=" * 72)
print("the residual drift under He is a FINITE-WIDTH effect")
print("=" * 72)
print("Eq. 56.5's E[relu(z)^2] = E[z^2]/2 holds exactly for a symmetric z.")
print("At finite width the units become correlated as depth accumulates and")
print("the empirical ratio drifts above one half. Widening should shrink it.\n")
print(f"{'width':>8} {'batch':>8} {'Var[z] at L50 / L1':>22} "
      f"{'implied per-layer gain':>25}")
for width in (32, 128, 512, 2048):
    st = propagate(50, width, "he", batch=max(512, 2 * width), seed=1)
    ratio = st[-1][1] / st[1][1]
    print(f"{width:>8} {max(512, 2 * width):>8} {ratio:>22.4f} "
          f"{ratio ** (1 / 49):>25.5f}")

print("\nRead the last column as a distance from 1.0. At width 32 the")
print("per-layer gain is off by nine per cent; at width 2048 it is off by")
print("one part in ten thousand. The deviation shrinks by roughly the")
print("expected order as the width grows, and it changes sign along the")
print("way, which is what a finite-size fluctuation does and what a")
print("systematic error in the derivation would not.")
print("\nEq. 56.6 is a statement about expectations over an infinitely wide")
print("layer. A width-256 layer approximates it to within about half a per")
print("cent per layer, which compounds over fifty layers into the factor of")
print("four in the previous table — and is negligible against the fifteen")
print("orders of magnitude a wrong scheme produces.")
print("\nThis is worth knowing because it is the honest status of the whole")
print("mean-field style of argument in this chapter: exact in the limit,")
print("approximate in practice, and the approximation error is small")
print("compared with the effect being predicted.")

# --- the same for tanh, where the factor of two is WRONG --------------------
print("\n" + "=" * 72)
print("the same schemes with tanh, where the factor of two is wrong")
print("=" * 72)
print(f"{'scheme':<14} " + " ".join(f"{f'L{i}':>11}" for i in picks)
      + f" {'ratio L50/L0':>14}")
for scheme in ("he", "lecun", "glorot"):
    st = propagate(DEPTH, WIDTH, scheme, act="tanh")
    vals = " ".join(f"{st[i][1]:>11.3e}" for i in picks)
    print(f"{scheme:<14} {vals} {st[-1][1] / st[0][1]:>14.3e}")

print("\nWith tanh the ordering reverses: Glorot and LeCun hold the scale")
print("and He's factor of two now OVERSHOOTS. Tanh does not zero half its")
print("inputs, so the compensation eq. 56.5 justifies is not needed and")
print("becomes a systematic over-scaling.")
print("\nNote that tanh's own saturation partly rescues the overshoot —")
print("|tanh| <= 1 caps the variance no matter how large the input — which")
print("is why the He row does not explode the way the ReLU 'too large' row")
print("did. Saturation is a bad way to control the scale, because it comes")
print("with the vanishing derivative of Chapter 50.")

# --- dead units -------------------------------------------------------------
print("\n" + "=" * 72)
print("what a bad scale does to the fraction of dead ReLU units")
print("=" * 72)
print(f"{'scheme':<14} " + " ".join(f"{f'L{i}':>9}" for i in [1, 5, 10, 25, 50]))
for scheme in ("he", "too small", "too large"):
    st = propagate(DEPTH, WIDTH, scheme)
    print(f"{scheme:<14} " + " ".join(f"{st[i][2]:>9.4f}"
                                      for i in [1, 5, 10, 25, 50]))
print("\nThe rows are not merely similar — they are IDENTICAL to every")
print("digit. Scaling every weight by a positive constant cannot change")
print("the sign of any pre-activation, so the set of dead units is exactly")
print("the same however wrong the scale is.")
print("\nSo the zero fraction is not a weak diagnostic for a bad scale; it")
print("is provably no diagnostic at all. It is worth knowing because it is")
print("a natural thing to reach for and it measures nothing.")
print("\nWhat goes wrong is the magnitude of the surviving half, and only")
print("the variance table shows that. This is a good example of a plausible")
print("diagnostic that measures nothing.")

# --- backward, and Glorot's compromise --------------------------------------
print("\n" + "=" * 72)
print("Glorot's compromise: the two conditions genuinely conflict (6.2)")
print("=" * 72)
print("A layer with fan_in and fan_out different. The forward pass wants")
print("Var[W] = 1/fan_in and the backward pass wants 1/fan_out.\n")
print(f"{'fan_in':>8} {'fan_out':>8} {'1/fan_in':>11} {'1/fan_out':>11} "
      f"{'Glorot':>11} {'fwd gain':>10} {'bwd gain':>10}")
for fi, fo in ((256, 256), (256, 1024), (1024, 256), (784, 128), (128, 10)):
    v_glorot = 2.0 / (fi + fo)
    print(f"{fi:>8} {fo:>8} {1 / fi:>11.3e} {1 / fo:>11.3e} "
          f"{v_glorot:>11.3e} {fi * v_glorot:>10.3f} {fo * v_glorot:>10.3f}")

print("\nThe two gain columns are what Glorot achieves in each direction: it")
print("hits 1.0 exactly when the layer is square and misses in both")
print("directions otherwise, by factors that are reciprocal. A 784->128")
print("layer amplifies the forward signal by 1.7 and attenuates the")
print("backward one by 0.28.")
print("\nThat is not a flaw in the derivation; it is the derivation's")
print("conclusion. Both conditions cannot hold for a non-square layer, and")
print("eq. 56.2 is the compromise. Networks of roughly constant width — the")
print("usual case — barely notice.")

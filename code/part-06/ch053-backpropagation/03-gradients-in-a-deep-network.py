# Extracted from: Chapter 53 — Backpropagation Derived from Scratch
# Source: src/.../ch053-backpropagation.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The recursion of eq. 53.7 observed in a deep network: what the gradient
looks like at each layer, and what clipping, accumulation and checkpointing
actually do.
"""
import numpy as np

rng = np.random.default_rng(4)


class DeepNet:
    def __init__(self, depth, width, d_in=12, d_out=3, scale=None,
                 act="tanh", seed=0):
        rs = np.random.default_rng(seed)
        sizes = [d_in] + [width] * depth + [d_out]
        self.W, self.b, self.act = [], [], act
        for i in range(len(sizes) - 1):
            s = scale if scale is not None else np.sqrt(2.0 / sizes[i])
            self.W.append(rs.normal(0, s, (sizes[i], sizes[i + 1])))
            self.b.append(np.zeros(sizes[i + 1]))

    def _phi(self, z):
        return np.tanh(z) if self.act == "tanh" else np.maximum(0.0, z)

    def _dphi(self, z, h):
        return 1 - h ** 2 if self.act == "tanh" else (z > 0).astype(float)

    def forward(self, X):
        self.Z, self.H = [], [X]
        h = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            self.Z.append(z)
            h = self._phi(z) if i < len(self.W) - 1 else z
            self.H.append(h)
        return h

    def backward(self, y_idx):
        B = len(y_idx)
        z = self.H[-1]
        m = z.max(axis=1, keepdims=True)
        e = np.exp(z - m)
        p = e / e.sum(axis=1, keepdims=True)
        delta = p.copy()
        delta[np.arange(B), y_idx] -= 1.0
        delta /= B
        gW, gb, prof = [None] * len(self.W), [None] * len(self.W), []
        for l in reversed(range(len(self.W))):
            gW[l] = self.H[l].T @ delta
            gb[l] = delta.sum(axis=0)
            prof.append({"layer": l,
                         "delta": float(np.sqrt(np.mean(delta ** 2))),
                         "gW": float(np.linalg.norm(gW[l])),
                         "h_in": float(np.sqrt(np.mean(self.H[l] ** 2)))})
            if l > 0:
                delta = (delta @ self.W[l].T) * self._dphi(self.Z[l - 1],
                                                           self.H[l])
        return gW, gb, list(reversed(prof))


X = rng.normal(size=(64, 12))
y = rng.integers(0, 3, size=64)

print("=" * 72)
print("the product of eq. 53.9, seen as a gradient norm per layer")
print("=" * 72)
print("A 20-layer tanh network at three initialisation scales. Three")
print("quantities per layer: the forward activation RMS, the error signal")
print("RMS of eq. 53.2, and the weight-gradient norm of eq. 53.5.\n")

picks = [0, 4, 9, 14, 19, 20]
for label, scale in (("small  (sd 0.05)", 0.05),
                     ("He     (sd sqrt(2/n))", None),
                     ("large  (sd 0.35)", 0.35)):
    net = DeepNet(20, 24, scale=scale, seed=2)
    net.forward(X)
    _, _, prof = net.backward(y)
    print(f"{label}")
    print("  " + " ".join(f"{'layer ' + str(i + 1):>11}" for i in picks))
    for key, name in (("h_in", "forward RMS"), ("delta", "delta RMS"),
                      ("gW", "|grad W|")):
        vals = " ".join(f"{prof[i][key]:>11.3e}" for i in picks)
        print(f"  {vals}   <- {name}")
    r_h = prof[0]["h_in"] / max(prof[-1]["h_in"], 1e-300)
    r_g = prof[0]["gW"] / max(prof[-1]["gW"], 1e-300)
    print(f"  layer-1 / layer-21:  forward {r_h:.3e}   "
          f"grad {r_g:.3e}\n")

print("Two distinct failures are visible here and they are worth separating,")
print("because the usual one-line account of 'vanishing gradients' runs them")
print("together.")
print("\nAt sd 0.05 the forward RMS collapses by twelve orders of magnitude")
print("across the depth, and the error signal of eq. 53.2 vanishes by")
print("twelve orders in the other direction — the textbook vanishing")
print("gradient, visible in the delta row exactly as eq. 53.9 predicts.")
print("\nBut look at the weight-gradient row: it is FLAT, at a uniformly")
print("useless 1e-13. That is worth understanding, because it is not what")
print("the delta row alone would suggest. Eq. 53.5 says the weight gradient")
print("is delta times the incoming activation, and here the two decays run")
print("in OPPOSITE directions with depth — a tiny delta meets a large")
print("activation at layer 1, and a large delta meets a tiny activation at")
print("layer 21. The product is roughly constant, and roughly zero.")
print("\nSo this network does not fail by starving its early layers")
print("relative to its late ones. It fails because every layer's gradient")
print("is negligible: a scale catastrophe rather than a tilt.")
print("\nAt sd 0.35 the profile TILTS: the gradient at layer 1 is several")
print("times the gradient at the output, so the lower layers move faster")
print("than the upper ones and the network is unbalanced rather than dead.")
print("\nThe He scale keeps both the forward RMS and the gradient profile")
print("within a small factor across twenty layers, which is exactly what")
print("Chapter 56 derives it to do.")
print("\nThe diagnostic that follows: print BOTH the forward activation")
print("scale and the per-layer gradient norm. The first tells you whether")
print("the signal survives the forward pass; the second tells you whether")
print("the gradient survives the backward one. They fail independently and")
print("the fixes are different.")

# --- gradient clipping ------------------------------------------------------
print("\n" + "=" * 72)
print("gradient clipping: what it does to the direction (section 6.6)")
print("=" * 72)


def clip_global(grads, max_norm):
    total = np.sqrt(sum(float(np.sum(g ** 2)) for g in grads))
    if total <= max_norm:
        return grads, total, 1.0
    s = max_norm / (total + 1e-12)
    return [g * s for g in grads], total, s


net = DeepNet(20, 24, scale=0.35, seed=2)
net.forward(X)
gW, _, _ = net.backward(y)
for mx in (1e9, 10.0, 1.0, 0.1):
    clipped, total, s = clip_global(gW, mx)
    flat_o = np.concatenate([g.ravel() for g in gW])
    flat_c = np.concatenate([g.ravel() for g in clipped])
    cos = float(flat_o @ flat_c / (np.linalg.norm(flat_o)
                                   * np.linalg.norm(flat_c)))
    print(f"max_norm={mx:>8.1e}  original |g|={total:>9.3f}  "
          f"scale={s:>7.4f}  cosine with original={cos:.6f}")
print("\nGlobal-norm clipping rescales every parameter by ONE shared factor,")
print("so the direction is exactly preserved — the cosine is 1 at every")
print("threshold. It is a step-size cap, not a change of direction.")
print("\nThat is what makes it safe. Per-parameter clipping, which clips each")
print("coordinate independently, does NOT preserve the direction:")
for mx in (1.0, 0.1, 0.01):
    per = [np.clip(g, -mx, mx) for g in gW]
    flat_o = np.concatenate([g.ravel() for g in gW])
    flat_p = np.concatenate([g.ravel() for g in per])
    cos = float(flat_o @ flat_p / (np.linalg.norm(flat_o)
                                   * np.linalg.norm(flat_p)))
    print(f"  per-coordinate clip at {mx:>5.2f}: "
          f"cosine with original = {cos:.6f}")
print("\n(The first per-coordinate row is a no-op: no single coordinate")
print("exceeds 1.0, so nothing is clipped and the cosine is trivially 1.")
print("The distortion appears as soon as the threshold actually binds.)")
print("\nUse global-norm clipping. The per-coordinate version is a different")
print("optimiser, not a safety net.")

# --- gradient accumulation --------------------------------------------------
print("\n" + "=" * 72)
print("gradient accumulation is exact (section 7.3)")
print("=" * 72)
Xb = rng.normal(size=(256, 12))
yb = rng.integers(0, 3, size=256)

net = DeepNet(4, 24, seed=5)
net.forward(Xb)
gfull, _, _ = net.backward(yb)

for micro in (256, 64, 32, 8):
    acc = [np.zeros_like(g) for g in gfull]
    nchunks = 256 // micro
    for c in range(nchunks):
        sl = slice(c * micro, (c + 1) * micro)
        net.forward(Xb[sl])
        gm, _, _ = net.backward(yb[sl])
        for a, g in zip(acc, gm):
            a += g / nchunks              # each micro-batch is already a mean
    err = max(float(np.max(np.abs(a - g)))
              for a, g in zip(acc, gfull))
    rel = max(float(np.max(np.abs(a - g)) / max(float(np.max(np.abs(g))), 1e-12))
              for a, g in zip(acc, gfull))
    print(f"micro-batch {micro:>4} ({nchunks:>2} chunks): "
          f"max abs diff {err:.3e}   max rel diff {rel:.3e}")
print("\nAccumulation reproduces the full-batch gradient to floating-point")
print("round-off. The residual difference is summation order (Chapter 46),")
print("not an approximation — the mathematics is identical.")
print("\nThe caveat from section 7.3 is worth repeating: this holds because")
print("everything here is a mean over independent examples. Batch")
print("normalisation is not, since its statistics are computed within a")
print("micro-batch, so accumulation does NOT reproduce a full batch there.")

# --- checkpointing ----------------------------------------------------------
print("\n" + "=" * 72)
print("gradient checkpointing: the memory/compute trade (eq. 53.17)")
print("=" * 72)


def checkpoint_cost(L, every):
    """Stored activations and forward passes for segment length `every`."""
    stored = np.ceil(L / every) + every          # checkpoints + one segment
    recompute = 1.0 + (every - 1) / every        # forwards per backward
    return stored, recompute


print(f"{'depth':>6} {'every':>7} {'stored (units)':>16} "
      f"{'vs storing all':>16} {'fwd passes':>12} {'step cost':>11}")
for L in (16, 64, 256):
    for every in (1, int(np.sqrt(L)), L):
        stored, recomp = checkpoint_cost(L, every)
        # a step is 1 forward + 2 backward-equivalent; recompute adds forwards
        step = (recomp + 2) / 3.0
        print(f"{L:>6} {every:>7} {stored:>16.0f} {stored / L:>15.2f}x "
              f"{recomp:>12.2f} {step:>10.2f}x")
print("\nThe middle row of each group is the interesting one. At segment")
print("length sqrt(L) the stored activations fall to about 2*sqrt(L)")
print("instead of L — a 64-layer network stores 16 units rather than 64,")
print("and a 256-layer one stores 32 rather than 256 — while the step costs")
print("about a third more, because the extra work is one forward pass")
print("against a step that already costs three.")
print("\nNote that BOTH extremes are bad. Checkpointing every layer stores")
print("everything and saves nothing. Checkpointing only the input stores one")
print("checkpoint and then has to hold an entire segment's activations")
print("during recomputation, which is the whole network again — so it pays")
print("the extra forward pass and saves nothing either. The saving comes")
print("from the interior of the range, and sqrt(L) is where the sum of the")
print("two terms is minimised.")
print("\nThat is eq. 53.17. Trading a third of the time for a square-root")
print("reduction in activation memory is why long-context models fit at")
print("all, and it is a decision you make per model rather than once.")

# --- the 3x rule, measured --------------------------------------------------
print("\n" + "=" * 72)
print("a training step costs about three forward passes (eq. 53.14)")
print("=" * 72)
import time

Xt = rng.normal(size=(512, 12))
yt = rng.integers(0, 3, size=512)


def timeit(fn, reps=10):
    fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


print(f"{'network':<22} {'fwd':>8} {'step':>8} {'ratio':>7}   "
      f"{'matmul fwd':>11} {'matmul step':>12} {'ratio':>7}")
for depth, width in ((8, 256), (16, 512)):
    net = DeepNet(depth, width, seed=7)
    tf = timeit(lambda: net.forward(Xt))
    ts = timeit(lambda: (net.forward(Xt), net.backward(yt)))

    # the same FLOPs with the elementwise work removed: forward is one matmul
    # per layer, a step is three (eq. 53.12 adds two)
    Ws = [W for W in net.W]
    H0 = [np.zeros((len(Xt), W.shape[0])) for W in Ws]
    D0 = [np.zeros((len(Xt), W.shape[1])) for W in Ws]

    def mm_fwd():
        for W, h in zip(Ws, H0):
            h @ W

    def mm_step():
        for W, h, d in zip(Ws, H0, D0):
            h @ W
            d @ W.T
            h.T @ d

    tmf = timeit(mm_fwd)
    tms = timeit(mm_step)
    print(f"depth {depth:>2} width {width:>4}     {tf * 1e3:>7.2f}ms "
          f"{ts * 1e3:>7.2f}ms {ts / tf:>6.2f}x   {tmf * 1e3:>10.2f}ms "
          f"{tms * 1e3:>11.2f}ms {tms / tmf:>6.2f}x")

print("\nThe last column is eq. 53.14: strip everything but the matrix")
print("products and a step costs almost exactly three forward passes, which")
print("is what the FLOP count predicts.")
print("\nThe measured ratio for the FULL step is well below three, and the")
print("reason is worth understanding rather than explaining away. The 3x")
print("rule counts matmuls. This forward pass also does a tanh, a bias add")
print("and a list append per layer, and the backward pass does not double")
print("those. If the forward pass costs M of matmul plus E of everything")
print("else, the step costs 3M + E, so the ratio is (3M+E)/(M+E) — which is")
print("3 only when E is negligible and falls toward 1 as E grows.")
print("\nOn a small CPU network E is a large fraction, so the measured")
print("ratio lands near 2. On an accelerator running a large model the")
print("matmuls dominate and fusion removes most of E, so the same")
print("measurement gives close to 3. The rule is right about the")
print("arithmetic and it is a statement about the compute-bound regime,")
print("which is worth remembering before quoting it at a profile that is")
print("not in that regime.")

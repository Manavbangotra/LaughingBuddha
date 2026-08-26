# -*- coding: utf-8 -*-
# Extracted from: Chapter 51 — Forward Propagation and Computational Graphs
# Source: src/.../ch051-forward.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A complete forward pass with shape tracking, FLOP accounting and a
train/eval mode — the three things a real implementation must get right.
"""
import numpy as np

rng = np.random.default_rng(5)


class Layer:
    """A dense layer that reports its own shapes, FLOPs and stored memory."""

    def __init__(self, n_in, n_out, act="relu", name="fc", seed=0):
        rs = np.random.default_rng(seed)
        self.W = rs.normal(0, np.sqrt(2.0 / n_in), (n_in, n_out))
        self.b = np.zeros(n_out)
        self.act, self.name = act, name
        self.cache = None

    def forward(self, X, training=True):
        Z = X @ self.W + self.b
        if self.act == "relu":
            H = np.maximum(0.0, Z)
        elif self.act == "tanh":
            H = np.tanh(Z)
        else:
            H = Z
        # section 4.2: the side effect. Only training keeps this.
        self.cache = (X, Z) if training else None
        return H

    def flops(self, B):
        return 2 * B * self.W.shape[0] * self.W.shape[1]

    def params(self):
        return self.W.size + self.b.size

    def stored_bytes(self, B, bytes_per=4):
        if self.cache is None:
            return 0
        return bytes_per * sum(a.size for a in self.cache)


class Net:
    def __init__(self, sizes, seed=0):
        self.layers = [
            Layer(sizes[i], sizes[i + 1],
                  act="relu" if i < len(sizes) - 2 else "linear",
                  name=f"fc{i + 1}", seed=seed + i)
            for i in range(len(sizes) - 1)]

    def forward(self, X, training=True, trace=False):
        if trace:
            print(f"{'layer':<8} {'input':>16} {'weights':>16} "
                  f"{'output':>16} {'MFLOPs':>9} {'stored MB':>11}")
            print(f"{'input':<8} {str(X.shape):>16} {'-':>16} "
                  f"{str(X.shape):>16} {'-':>9} {'-':>11}")
        h = X
        for L in self.layers:
            prev = h.shape
            h = L.forward(h, training=training)
            if trace:
                print(f"{L.name:<8} {str(prev):>16} {str(L.W.shape):>16} "
                      f"{str(h.shape):>16} {L.flops(len(X)) / 1e6:>9.2f} "
                      f"{L.stored_bytes(len(X)) / 1e6:>11.3f}")
        return h

    def totals(self, B, training=True):
        f = sum(L.flops(B) for L in self.layers)
        p = sum(L.params() for L in self.layers)
        m = sum(L.stored_bytes(B) for L in self.layers) if training else 0
        return f, p, m


# --- a traced forward pass --------------------------------------------------
print("=" * 72)
print("a forward pass, traced (eqs. 51.4, 51.6)")
print("=" * 72)
net = Net([784, 512, 512, 10], seed=1)
X = rng.normal(size=(64, 784))
out = net.forward(X, training=True, trace=True)
f, p, m = net.totals(64)
print(f"\ntotal: {f / 1e6:.1f} MFLOPs forward, {p:,} parameters, "
      f"{m / 1e6:.2f} MB stored")
print(f"a training step is roughly 3x the forward FLOPs (section 7.2): "
      f"{3 * f / 1e6:.1f} MFLOPs")

# --- training vs inference memory -------------------------------------------
print("\n" + "=" * 72)
print("training keeps activations; inference does not (section 5.5)")
print("=" * 72)
print(f"{'batch':>7} {'training MB':>13} {'inference MB':>14} "
      f"{'x more to train':>17}")
for B in (1, 8, 64, 256, 1024):
    Xb = rng.normal(size=(B, 784))
    net.forward(Xb, training=True)
    _, params, m_train = net.totals(B, training=True)
    net.forward(Xb, training=False)
    m_inf = sum(L.stored_bytes(B) for L in net.layers)
    par_mb = params * 4 / 1e6
    print(f"{B:>7} {(par_mb + m_train / 1e6):>13.2f} "
          f"{(par_mb + m_inf / 1e6):>14.2f} "
          f"{(par_mb + m_train / 1e6) / (par_mb + m_inf / 1e6):>17.2f}x")

print("\nAt batch 1 the two are nearly identical — the parameters dominate.")
print("By batch 1024 training needs several times the memory, and every byte")
print("of the difference is activations being held for a backward pass that")
print("inference never performs.")

# --- the train/eval mode bug ------------------------------------------------
print("\n" + "=" * 72)
print("the train/eval mode bug (section 5.5)")
print("=" * 72)


class Dropout:
    """Deliberately written to show what forgetting the mode flag costs."""

    def __init__(self, p=0.5):
        self.p = p

    def forward(self, X, training=True, rs=None):
        if not training:
            return X                                  # identity at eval
        mask = (rs.random(X.shape) > self.p) / (1 - self.p)
        return X * mask


rs = np.random.default_rng(9)
h = rng.normal(size=(2000, 64))
drop = Dropout(0.5)

eval_out = drop.forward(h, training=False)
train_outs = np.array([drop.forward(h, training=True, rs=rs).mean()
                       for _ in range(50)])
print(f"activation mean, eval mode      : {eval_out.mean():>9.5f}")
print(f"activation mean, train mode     : {train_outs.mean():>9.5f} "
      f"(mean over 50 masks)")
print(f"activation mean, train mode SD  : {train_outs.std():>9.5f}")
print(f"\nthe expectations match — inverted dropout rescales by 1/(1-p) so")
print(f"that they do — but a single train-mode forward pass differs from")
print(f"the eval-mode one by a random amount:")
one = drop.forward(h, training=True, rs=rs)
print(f"  max |single train-mode pass - eval-mode pass| = "
      f"{np.abs(one - eval_out).max():.4f}")
print(f"  fraction of activations zeroed = {(one == 0).mean():.4f}")

print("\nServing a model left in training mode gives a DIFFERENT answer for")
print("the same input on every call, with half the activations missing. The")
print("expectation is right, so aggregate metrics computed over a large")
print("evaluation set look almost correct — which is what makes this bug")
print("survive. The symptom is per-request nondeterminism, not a bad score.")

# --- and the FLOP/time gap of section 6.2 -----------------------------------
print("\n" + "=" * 72)
print("FLOPs predict compute-bound time and mislead about the rest")
print("=" * 72)
import time

B = 256
Xb = rng.normal(size=(B, 784))
t0 = time.perf_counter()
for _ in range(20):
    net.forward(Xb, training=True)
dt_full = (time.perf_counter() - t0) / 20

Wc = [L.W for L in net.layers]
t0 = time.perf_counter()
for _ in range(20):
    h = Xb
    for W in Wc:
        h = h @ W                                     # matmuls ONLY
dt_mm = (time.perf_counter() - t0) / 20

f, _, _ = net.totals(B)
print(f"matmuls only          : {dt_mm * 1e3:>7.2f} ms  "
      f"({f / dt_mm / 1e9:>6.1f} GFLOP/s)")
print(f"full forward pass     : {dt_full * 1e3:>7.2f} ms")
print(f"elementwise overhead  : {(dt_full - dt_mm) * 1e3:>7.2f} ms "
      f"({(dt_full - dt_mm) / dt_full:.0%} of the total)")
print("\nThe FLOP count in eq. 51.4 counts only the matmuls, and they are")
print("not the whole time. The bias adds and activations contribute a")
print("negligible number of FLOPs and a non-negligible fraction of the")
print("runtime, because they are bandwidth-bound (section 6.3).")
print("\nThat gap is what operator fusion closes, and it is why two")
print("frameworks running identical mathematics can differ by a factor of")
print("two in wall-clock time.")

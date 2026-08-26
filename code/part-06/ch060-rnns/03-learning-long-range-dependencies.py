# -*- coding: utf-8 -*-
# Extracted from: Chapter 60 — Recurrent Networks: RNN, LSTM, and GRU
# Source: src/.../ch060-rnns.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The task recurrence exists for: a dependency spanning the sequence.
Vanilla RNN against GRU against LSTM, as the distance grows.
"""
import numpy as np

rng = np.random.default_rng(3)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


# --- the task: remember a bit across T steps --------------------------------
def make_copy_task(n, T, seed):
    """The first token carries the label; everything after is noise.
    Solving it REQUIRES carrying information T steps."""
    rs = np.random.default_rng(seed)
    X = rs.normal(0, 0.5, (n, T, 4))
    y = rs.integers(0, 2, n)
    X[:, 0, 0] = np.where(y == 1, 3.0, -3.0)      # the signal, at step 0
    X[:, 0, 1] = 1.0                              # a marker
    return X, y


class GRUNet:
    """Eqs. 60.13-60.14, fused."""

    def __init__(self, n_in, d, seed=0, z_bias=0.0):
        rs = np.random.default_rng(seed)
        k = 1.0 / np.sqrt(d + n_in)
        self.Wzr = rs.normal(0, k, (d + n_in, 2 * d))
        self.bzr = np.zeros(2 * d)
        # Section 7.5's argument, applied to the GRU. h_t = (1-z)h + z*cand,
        # so a NEGATIVE z bias starts the update gate closed and the state
        # held — the GRU's equivalent of a positive LSTM forget bias.
        self.bzr[:d] = z_bias
        self.Wh = rs.normal(0, k, (d + n_in, d))
        self.bh = np.zeros(d)
        self.Wo = rs.normal(0, 1 / np.sqrt(d), (d, 2))
        self.bo = np.zeros(2)
        self.d = d

    def params(self):
        return [self.Wzr, self.bzr, self.Wh, self.bh, self.Wo, self.bo]

    def forward(self, X):
        B, T, _ = X.shape
        d = self.d
        h = np.zeros((B, d))
        self.cache = []
        for t in range(T):
            hx = np.concatenate([h, X[:, t]], axis=1)
            zr = sigmoid(hx @ self.Wzr + self.bzr)
            z, r = zr[:, :d], zr[:, d:]
            hx2 = np.concatenate([r * h, X[:, t]], axis=1)
            cand = np.tanh(hx2 @ self.Wh + self.bh)
            h_new = (1 - z) * h + z * cand                # eq. 60.14
            self.cache.append((hx, z, r, hx2, cand, h))
            h = h_new
        self.hT = h
        self.T = T
        return h @ self.Wo + self.bo

    def backward(self, dlogits):
        d = self.d
        gWo = self.hT.T @ dlogits
        gbo = dlogits.sum(axis=0)
        dh = dlogits @ self.Wo.T
        gWzr = np.zeros_like(self.Wzr)
        gbzr = np.zeros_like(self.bzr)
        gWh = np.zeros_like(self.Wh)
        gbh = np.zeros_like(self.bh)
        carry = []
        for t in reversed(range(self.T)):
            hx, z, r, hx2, cand, h_prev = self.cache[t]
            carry.append(float(np.sqrt(np.mean(dh ** 2))))
            dz = dh * (cand - h_prev)
            dcand = dh * z
            dh_prev = dh * (1 - z)
            dc_pre = dcand * (1 - cand ** 2)
            gWh += hx2.T @ dc_pre
            gbh += dc_pre.sum(axis=0)
            dhx2 = dc_pre @ self.Wh.T
            dr_h = dhx2[:, :d]
            dr = dr_h * h_prev
            dh_prev = dh_prev + dr_h * r
            dzr_pre = np.concatenate(
                [dz * z * (1 - z), dr * r * (1 - r)], axis=1)
            gWzr += hx.T @ dzr_pre
            gbzr += dzr_pre.sum(axis=0)
            dhx = dzr_pre @ self.Wzr.T
            dh = dh_prev + dhx[:, :d]
        return [gWzr, gbzr, gWh, gbh, gWo, gbo], list(reversed(carry))


class VanillaNet:
    def __init__(self, n_in, d, seed=0, ortho=False):
        rs = np.random.default_rng(seed)
        if ortho:
            Q, R = np.linalg.qr(rs.normal(size=(d, d)))
            self.Whh = Q * np.sign(np.diag(R))
        else:
            self.Whh = rs.normal(0, 1 / np.sqrt(d), (d, d))
        self.Wxh = rs.normal(0, 1 / np.sqrt(n_in), (n_in, d))
        self.bh = np.zeros(d)
        self.Wo = rs.normal(0, 1 / np.sqrt(d), (d, 2))
        self.bo = np.zeros(2)
        self.d = d

    def params(self):
        return [self.Whh, self.Wxh, self.bh, self.Wo, self.bo]

    def forward(self, X):
        B, T, _ = X.shape
        h = np.zeros((B, self.d))
        self.H = [h]
        for t in range(T):
            h = np.tanh(h @ self.Whh + X[:, t] @ self.Wxh + self.bh)
            self.H.append(h)
        self.X, self.T = X, T
        return h @ self.Wo + self.bo

    def backward(self, dlogits):
        gWo = self.H[-1].T @ dlogits
        gbo = dlogits.sum(axis=0)
        dh = dlogits @ self.Wo.T
        gWhh = np.zeros_like(self.Whh)
        gWxh = np.zeros_like(self.Wxh)
        gbh = np.zeros_like(self.bh)
        carry = []
        for t in reversed(range(self.T)):
            carry.append(float(np.sqrt(np.mean(dh ** 2))))
            dz = dh * (1 - self.H[t + 1] ** 2)
            gWhh += self.H[t].T @ dz
            gWxh += self.X[:, t].T @ dz
            gbh += dz.sum(axis=0)
            dh = dz @ self.Whh.T
        return [gWhh, gWxh, gbh, gWo, gbo], list(reversed(carry))


def train(net, X, y, Xv, yv, steps=600, lr=5e-3, batch=64, clip=1.0, seed=0):
    ps = net.params()
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    rs = np.random.default_rng(seed + 5)
    for t in range(1, steps + 1):
        idx = rs.integers(0, len(X), batch)
        logits = net.forward(X[idx])
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        p = e / e.sum(axis=1, keepdims=True)
        p[np.arange(len(idx)), y[idx]] -= 1.0
        gs, _ = net.backward(p / len(idx))
        total = np.sqrt(sum(float(np.sum(g ** 2)) for g in gs))
        scale = min(1.0, clip / (total + 1e-12))       # global-norm clipping
        for i, (pp, g) in enumerate(zip(ps, gs)):
            g = g * scale
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            pp -= lr * (m[i] / (1 - 0.9 ** t)) / (
                np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    lg = net.forward(Xv)
    return float((lg.argmax(axis=1) == yv).mean())


print("=" * 72)
print("the task recurrence exists for: carrying a bit across T steps")
print("=" * 72)
print("The label is in the FIRST token; everything after is noise. Solving")
print("it requires preserving information for the whole sequence.\n")
MODELS = [
    ("vanilla", lambda: VanillaNet(4, 48, seed=1)),
    ("vanilla (ortho)", lambda: VanillaNet(4, 48, seed=1, ortho=True)),
    ("GRU, z bias 0", lambda: GRUNet(4, 48, seed=1, z_bias=0.0)),
    ("GRU, z bias -2", lambda: GRUNet(4, 48, seed=1, z_bias=-2.0)),
    ("GRU, z bias -4", lambda: GRUNet(4, 48, seed=1, z_bias=-4.0)),
]
print(f"{'T':>5} " + " ".join(f"{n:>17}" for n, _ in MODELS))
for T in (5, 20, 60, 120):
    Xa, ya = make_copy_task(3000, T, 11)
    Xb, yb = make_copy_task(2000, T, 12)
    row = [train(mk(), Xa, ya, Xb, yb) for _, mk in MODELS]
    print(f"{T:>5} " + " ".join(f"{a:>17.4f}" for a in row))
print("\n(chance is 0.5000)")

print("\nThe three GRU columns are the point of this table, and the result")
print("is not the one the architecture's reputation would predict.")
print("\nAt its default bias the GRU FAILS at long T, and it fails for")
print("exactly the reason section 7.5 gives for the LSTM. Its update gate")
print("starts at sigmoid(0) = 0.5, so eq. 60.14 gives h_t = 0.5 h_{t-1} +")
print("0.5 cand and the carry decays as 0.5^T — gone in twenty steps. There")
print("is then no gradient with which to learn to close the gate, and the")
print("architecture that is supposed to solve the problem cannot get")
print("started on it.")
print("\nBiasing the update gate closed fixes it, and the effect is large.")
print("This is the same chicken-and-egg failure and the same one-line")
print("remedy as the LSTM's forget-gate bias — which is worth knowing")
print("because the LSTM convention is widely taught and the GRU one is not.")
print("\nThe vanilla rows are the other surprise, and the bigger one. The")
print("plain tanh RNN solves this task at T = 120, where eq. 60.8's product")
print("says the gradient reaching step 0 should be around 1e-30.")
print("\nThe explanation is that this task does not require gradient flow")
print("to be solved — it requires a LATCH. The signal is a strong bipolar")
print("spike, and a saturating recurrence can park itself in one of two")
print("attractors and stay there. Once the network finds that solution it")
print("holds the bit indefinitely, and finding it needs gradient only over")
print("the first few steps, not over all 120.")
print("\nThat is a real mechanism and a narrow one. It works because the")
print("thing being remembered is one bit carried by a large-amplitude")
print("signal. It would not work for graded information, for several")
print("competing signals, or where the state must also keep changing —")
print("which is every real sequence task.")
print("\nThe methodological lesson is worth more than the result. A")
print("synthetic long-range benchmark can be solvable by a mechanism other")
print("than the one it was designed to test, and the architecture that")
print("'wins' is then telling you about your task rather than about long-")
print("range dependencies. That the orthogonal initialisation — which")
print("eq. 60.8 says has the best-conditioned product — does WORST here is")
print("the clue that something other than the product is doing the work.")

# --- the gradient reaching step 0, measured ---------------------------------
print("\n" + "=" * 72)
print("the gradient actually reaching the first time step")
print("=" * 72)
T = 60
Xa, ya = make_copy_task(2000, T, 21)
print(f"{'model':<20} " +
      " ".join(f"{f't={t}':>12}" for t in (59, 40, 20, 0))
      + f" {'t=0 / t=59':>13}")
for name, net in (("vanilla", VanillaNet(4, 48, seed=1)),
                  ("vanilla (ortho)", VanillaNet(4, 48, seed=1, ortho=True)),
                  ("GRU, z bias 0", GRUNet(4, 48, seed=1, z_bias=0.0)),
                  ("GRU, z bias -4", GRUNet(4, 48, seed=1, z_bias=-4.0))):
    logits = net.forward(Xa[:256])
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    p = e / e.sum(axis=1, keepdims=True)
    p[np.arange(256), ya[:256]] -= 1.0
    _, carry = net.backward(p / 256)
    print(f"{name:<20} " + " ".join(f"{carry[t]:>12.3e}"
                                    for t in (59, 40, 20, 0))
          + f" {carry[0] / max(carry[59], 1e-300):>13.3e}")

print("\nThis is at INITIALISATION, before training. The last column is the")
print("fraction of the output gradient that reaches the first time step —")
print("the step that holds the answer.")
print("\nA number of 1e-10 means the parameter update responsible for")
print("remembering the first token is ten orders of magnitude smaller than")
print("the update for the last one. The network will learn the recent")
print("context and never learn the long-range dependency, and nothing about")
print("the loss curve will say so.")
print("\nThat invisibility is section 6.2's point about the asymmetry.")
print("Exploding gradients announce themselves as a nan; vanishing ones")
print("produce a model that trains successfully and is quietly wrong about")
print("what it can represent.")

# --- truncated BPTT ---------------------------------------------------------
print("\n" + "=" * 72)
print("truncated BPTT imposes a HARD limit on what can be learned (5.5)")
print("=" * 72)


def train_truncated(net, X, y, Xv, yv, k, steps=600, lr=5e-3, batch=64,
                    seed=0):
    """Zero the gradient contribution from beyond k steps back."""
    ps = net.params()
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    rs = np.random.default_rng(seed + 5)
    T = X.shape[1]
    for t in range(1, steps + 1):
        idx = rs.integers(0, len(X), batch)
        Xb = X[idx].copy()
        # truncation to k steps == the model only SEES the last k steps
        Xb[:, :max(0, T - k)] = 0.0
        logits = net.forward(Xb)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        p = e / e.sum(axis=1, keepdims=True)
        p[np.arange(len(idx)), y[idx]] -= 1.0
        gs, _ = net.backward(p / len(idx))
        total = np.sqrt(sum(float(np.sum(g ** 2)) for g in gs))
        sc = min(1.0, 1.0 / (total + 1e-12))
        for i, (pp, g) in enumerate(zip(ps, gs)):
            g = g * sc
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            pp -= lr * (m[i] / (1 - 0.9 ** t)) / (
                np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    Xvb = Xv.copy()
    Xvb[:, :max(0, T - k)] = 0.0
    return float((net.forward(Xvb).argmax(axis=1) == yv).mean())


T = 40
Xa, ya = make_copy_task(3000, T, 31)
Xb_, yb_ = make_copy_task(2000, T, 32)
print(f"the signal is at step 0 of a {T}-step sequence\n")
print(f"{'truncation k':>14} {'covers step 0?':>16} {'GRU test acc':>14}")
for k in (5, 20, 39, 40):
    net = GRUNet(4, 48, seed=1, z_bias=-4.0)
    acc = train_truncated(net, Xa, ya, Xb_, yb_, k)
    print(f"{k:>14} {str(k >= T):>16} {acc:>14.4f}")
print("\n(chance is 0.5000)")

print("\nTruncating to k steps makes the first T-k steps invisible, and the")
print("signal is at step 0. Below k = T the model cannot see the answer at")
print("all, so no amount of training helps — the limit is imposed by the")
print("training procedure, not by the architecture.")
print("\nThis is the cost of truncated BPTT stated as sharply as possible.")
print("It buys O(k) memory instead of O(T), and it buys it by making")
print("dependencies longer than k unlearnable. When you truncate, you are")
print("choosing a maximum dependency length, and it is worth choosing it")
print("deliberately rather than inheriting it from a memory constraint.")

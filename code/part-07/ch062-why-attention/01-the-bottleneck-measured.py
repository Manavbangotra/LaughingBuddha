# Extracted from: Chapter 62 — Why Recurrence Failed: The Road to Attention
# Source: src/.../ch062-why-attention.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The fixed-size bottleneck, measured: how much can one vector carry?"""
import numpy as np

rng = np.random.default_rng(0)


# --- a copy task: encode a sequence into ONE vector, then decode it ---------
def make_sequences(n, T, V, seed):
    rs = np.random.default_rng(seed)
    return rs.integers(0, V, (n, T))


def onehot(X, V):
    out = np.zeros((*X.shape, V))
    np.put_along_axis(out, X[..., None], 1.0, axis=-1)
    return out


class BottleneckAE:
    """Encode T tokens into ONE d-dimensional vector, decode all T back.

    This is eq. 62.4's constraint in its purest form: the decoder sees
    nothing but c, so whatever it reconstructs must have fitted in d
    numbers.
    """

    def __init__(self, T, V, d, seed=0):
        rs = np.random.default_rng(seed)
        self.T, self.V, self.d = T, V, d
        self.We = rs.normal(0, np.sqrt(2 / (T * V)), (T * V, d))
        self.be = np.zeros(d)
        self.Wd = rs.normal(0, np.sqrt(2 / d), (d, T * V))
        self.bd = np.zeros(T * V)

    def params(self):
        return [self.We, self.be, self.Wd, self.bd]

    def forward(self, X):
        self.flat = onehot(X, self.V).reshape(len(X), -1)
        self.c = np.tanh(self.flat @ self.We + self.be)      # THE bottleneck
        return (self.c @ self.Wd + self.bd).reshape(len(X), self.T, self.V)

    def loss_and_grads(self, X):
        logits = self.forward(X)
        m = logits.max(axis=-1, keepdims=True)
        e = np.exp(logits - m)
        p = e / e.sum(axis=-1, keepdims=True)
        n = len(X) * self.T
        loss = float(-np.log(np.clip(
            np.take_along_axis(p, X[..., None], -1), 1e-12, None)).sum() / n)
        d = p.copy()
        np.put_along_axis(d, X[..., None],
                          np.take_along_axis(d, X[..., None], -1) - 1.0, -1)
        d /= n
        dflat = d.reshape(len(X), -1)
        gWd, gbd = self.c.T @ dflat, dflat.sum(axis=0)
        dc = (dflat @ self.Wd.T) * (1 - self.c ** 2)
        return loss, [self.flat.T @ dc, dc.sum(axis=0), gWd, gbd]

    def accuracy(self, X):
        return float((self.forward(X).argmax(axis=-1) == X).mean())


def train(net, X, steps=4000, lr=3e-3, batch=64, seed=0):
    ps = net.params()
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    rs = np.random.default_rng(seed + 1)
    for t in range(1, steps + 1):
        xb = X[rs.integers(0, len(X), batch)]
        _, gs = net.loss_and_grads(xb)
        for i, (p, g) in enumerate(zip(ps, gs)):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            p -= lr * (m[i] / (1 - 0.9 ** t)) / (
                np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    return net


print("=" * 72)
print("the fixed-size bottleneck (eq. 62.4)")
print("=" * 72)
print("Encode T tokens into ONE d-dimensional vector and decode them back.")
print("Perfect reconstruction requires the sequence to FIT in d numbers.\n")
V = 8
print(f"vocabulary {V}, so a length-T sequence carries "
      f"T * log2({V}) = {np.log2(V):.0f}T bits\n")
print(f"{'d':>5} " + " ".join(f"{f'T={T}':>10}" for T in (2, 4, 8, 16)))
for d in (2, 4, 8, 16, 32):
    row = []
    for T in (2, 4, 8, 16):
        Xtr = make_sequences(3000, T, V, 1)
        net = train(BottleneckAE(T, V, d, seed=2), Xtr, steps=3000)
        row.append(net.accuracy(make_sequences(2000, T, V, 3)))
    print(f"{d:>5} " + " ".join(f"{a:>10.4f}" for a in row))
print(f"\n(chance is 1/{V} = {1 / V:.4f})")

print("\nRead along each row: as the sequence grows at a FIXED bottleneck")
print("width, reconstruction degrades. Read down each column: widening the")
print("bottleneck recovers it. That is eq. 62.4's shape — capacity constant")
print("in T against information linear in T.")
print("\nThis is the failure Bahdanau et al. were looking at in 2014,")
print("reduced to its skeleton. A translation model does not need to")
print("reconstruct its input exactly, so the real curve is gentler than")
print("this one — but it has the same shape, and the observed degradation")
print("of translation quality with source length is what it looks like on")
print("a real task.")

# --- and what attention does to it -----------------------------------------
print("\n" + "=" * 72)
print("what attention changes")
print("=" * 72)
print("The decoder now reads a WEIGHTED AVERAGE of T encoder states rather")
print("than one summary. Capacity available to it grows with the input.\n")
print(f"{'T':>5} {'one vector, d=8':>18} {'T vectors of d=8':>19} "
      f"{'capacity ratio':>16}")
for T in (2, 4, 8, 16, 32):
    print(f"{T:>5} {8:>18} {8 * T:>19} {T:>15}x")
print("\nThat is the entire structural difference and it does not depend on")
print("any detail of how the weights are computed. The bottleneck was a")
print("consequence of summarising into ONE vector, and attention does not.")
print("\nNote what it costs: the decoder must now hold all T encoder states,")
print("so memory grows with the input where before it did not. That is the")
print("first appearance of the trade this whole part is about, and Chapter")
print("69's KV cache is the same bill arriving at serving time.")

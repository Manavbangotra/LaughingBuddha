# Extracted from: Chapter 60 — Recurrent Networks: RNN, LSTM, and GRU
# Source: src/.../ch060-rnns.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""What gating does to the recursion, measured: the LSTM's carry path, the
forget-gate bias, and the horizon each architecture can actually learn.
"""
import numpy as np

rng = np.random.default_rng(0)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


# --- section 6.3: the carry path --------------------------------------------
print("=" * 72)
print("the LSTM carry path is a product of GATE VALUES (eq. 60.15)")
print("=" * 72)
print("A vanilla RNN's carry Jacobian is diag(1-h^2) W^T — a weight matrix")
print("and an activation derivative at every step. An LSTM's is diag(f_t)")
print("— gate values and nothing else.\n")
print(f"{'forget gate f':>15} " +
      " ".join(f"{f'after {t}':>13}" for t in (10, 50, 200, 1000)))
for f in (0.5, 0.9, 0.99, 0.999, 1.0):
    print(f"{f:>15.3f} " + " ".join(f"{f ** t:>13.3e}"
                                    for t in (10, 50, 200, 1000)))

print("\nThis table is the entire argument for gating. At f = 0.5 the")
print("gradient is gone in twenty steps. At f = 0.999 it survives a")
print("thousand. The network chooses f, per unit and per time step.")
print("\nNote what the table also shows: the LSTM does NOT eliminate the")
print("problem. If it learns f = 0.9 the decay is still exponential. What")
print("it provides is the OPTION of a gain near 1 and the gradient signal")
print("to learn when to take it — which a fixed architecture cannot offer.")

# --- section 7.5: the forget-gate bias --------------------------------------
print("\n" + "=" * 72)
print("why the forget-gate bias is initialised positive (section 7.5)")
print("=" * 72)
print(f"{'b_f':>6} {'sigma(b_f)':>12} " +
      " ".join(f"{f'carry @ {t}':>14}" for t in (10, 50, 200)))
for bf in (-1.0, 0.0, 1.0, 2.0, 3.0):
    f = float(sigmoid(bf))
    print(f"{bf:>6.1f} {f:>12.4f} " +
          " ".join(f"{f ** t:>14.3e}" for t in (10, 50, 200)))

print("\nAt b_f = 0 the gate starts at 0.5 and the carry has decayed to")
print("1e-15 after fifty steps — before the network has learned anything at")
print("all, so there is no gradient with which to learn to open the gate.")
print("It is a chicken-and-egg failure.")
print("\nInitialising b_f = 2 starts the gate at 0.88, which survives fifty")
print("steps at 1e-3 — small but present, and enough to learn from. That is")
print("the whole justification for a convention that otherwise looks")
print("arbitrary.")

# --- a full LSTM, and the horizon each architecture learns ------------------
class LSTM:
    """Eqs. 60.9-60.12 with a fused weight matrix (section 7.2)."""

    def __init__(self, n_in, d, n_out, forget_bias=1.0, seed=0):
        rs = np.random.default_rng(seed)
        k = 1.0 / np.sqrt(d + n_in)
        self.W = rs.normal(0, k, (d + n_in, 4 * d))     # f, i, c, o fused
        self.b = np.zeros(4 * d)
        self.b[:d] = forget_bias                        # section 7.5
        self.Why = rs.normal(0, 1.0 / np.sqrt(d), (d, n_out))
        self.by = np.zeros(n_out)
        self.d = d

    def forward(self, X):
        B, T, _ = X.shape
        d = self.d
        h = np.zeros((B, d))
        c = np.zeros((B, d))
        self.cache = []
        for t in range(T):
            z = np.concatenate([h, X[:, t]], axis=1) @ self.W + self.b
            f = sigmoid(z[:, :d])
            i = sigmoid(z[:, d:2 * d])
            g = np.tanh(z[:, 2 * d:3 * d])
            o = sigmoid(z[:, 3 * d:])
            c_prev = c
            c = f * c_prev + i * g                       # eq. 60.11
            tc = np.tanh(c)
            h = o * tc
            self.cache.append((np.concatenate([
                self.h_prev if False else np.zeros(0)], axis=0)
                if False else (f, i, g, o, c_prev, c, tc,
                               np.concatenate([
                                   np.zeros((B, 0)), X[:, t]], axis=1))))
            self.cache[-1] = (f, i, g, o, c_prev, c, tc,
                              np.concatenate([
                                  self.cache[-2][8] if False else np.zeros(
                                      (B, 0)), X[:, t]], axis=1))
        self.hT = h
        self.X = X
        return h @ self.Why + self.by

    def forget_gates(self):
        return np.array([cc[0].mean() for cc in self.cache])


print("\n" + "=" * 72)
print("what an untrained LSTM's carry looks like at each forget bias")
print("=" * 72)
Xs = rng.normal(size=(64, 120, 6))
print(f"{'forget bias':>13} {'mean gate':>11} {'carry over 120 steps':>22}")
for fb in (0.0, 1.0, 2.0, 3.0):
    net = LSTM(6, 32, 3, forget_bias=fb, seed=5)
    net.forward(Xs)
    g = net.forget_gates()
    print(f"{fb:>13.1f} {g.mean():>11.4f} "
          f"{float(np.prod(g)):>22.3e}")

print("\nThe measured mean gate tracks sigmoid(b_f) closely, and the product")
print("over 120 steps is the carry the gradient would see at")
print("initialisation. At b_f = 0 it is numerically zero; at b_f = 3 it")
print("survives.")
print("\nThis is at INITIALISATION, before any learning. The bias does not")
print("decide what the network ends up doing — it decides whether there is")
print("a gradient to learn from at all.")

# --- section 6.5: the parallelism argument ----------------------------------
print("\n" + "=" * 72)
print("the real reason recurrence lost: the critical path (eq. 60.16)")
print("=" * 72)
import time

print("Identical FLOPs both ways. The only difference is whether they are")
print("done in T sequential rounds or one big one.\n")
print(f"{'T':>6} {'batch':>7} {'d':>6} {'per-step MFLOP':>16} "
      f"{'sequential':>13} {'one matmul':>13} {'ratio':>8}")
for T, B, d in ((256, 1, 128), (256, 8, 128), (256, 64, 128),
                (256, 64, 512), (1024, 1, 128), (1024, 64, 512)):
    Wr = rng.normal(0, 1 / np.sqrt(d), (d, d)).astype(np.float32)
    Xb = rng.normal(size=(B, T, d)).astype(np.float32)
    h0 = np.zeros((B, d), dtype=np.float32)
    t0 = time.perf_counter()
    h = h0
    for t in range(T):
        h = np.tanh(h @ Wr + Xb[:, t])
    dt_seq = time.perf_counter() - t0
    flat = Xb.reshape(B * T, d)
    t0 = time.perf_counter()
    _ = np.tanh(flat @ Wr)
    dt_par = time.perf_counter() - t0
    mflop = 2 * B * d * d / 1e6
    print(f"{T:>6} {B:>7} {d:>6} {mflop:>16.3f} "
          f"{dt_seq * 1e3:>11.2f}ms {dt_par * 1e3:>11.2f}ms "
          f"{dt_seq / dt_par:>8.1f}x")

print("\nThe gap reaches more than an order of magnitude, and it appears")
print("where there is real work to serialise. Each sequential round is")
print("small enough that the library never reaches peak throughput on it,")
print("while the fused version does the identical arithmetic in one call")
print("that does. That is Chapter 51's roofline argument in a different")
print("costume.")
print("\nThe two batch-1 rows go the other way, and they are worth reading")
print("rather than dismissing. There the TOTAL work is a few MFLOP and both")
print("versions are dominated by per-call overhead rather than by")
print("arithmetic — the comparison is measuring NumPy's function-call cost,")
print("not the hardware. A ratio below 1 there means the experiment has")
print("left the regime it was designed to probe, not that recurrence wins.")
print("\nBe honest about the scale of all of this. Four CPU cores is not")
print("where the argument bites hardest; on an accelerator with thousands")
print("of parallel units, the per-round shortfall is far larger and the gap")
print("at realistic model sizes dwarfs anything measurable here.")
print("\nThe underlying fact is eq. 60.16, and it does not depend on the")
print("hardware: a recurrence has an O(T) critical path and a transformer")
print("has O(1) in the sequence dimension. The transformer does MORE total")
print("work — O(T^2 d) against O(T d^2) — and wins anyway wherever there")
print("are enough parallel units for the critical path to be the binding")
print("constraint. That is an argument about hardware, not about modelling.")
print("\nWhich is also why the recurrent idea came back. A LINEAR recurrence")
print("is associative, so it can be computed with a parallel scan in")
print("O(log T) depth — recovering the transformer's parallelism at the")
print("recurrence's linear cost. Whether that line displaces attention is")
print("open as of 2026.")

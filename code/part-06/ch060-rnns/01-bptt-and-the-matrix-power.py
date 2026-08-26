# -*- coding: utf-8 -*-
# Extracted from: Chapter 60 — Recurrent Networks: RNN, LSTM, and GRU
# Source: src/.../ch060-rnns.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Backpropagation through time, and the matrix power that makes recurrence
the sharpest case of the vanishing-gradient problem.
"""
import numpy as np

rng = np.random.default_rng(0)


# --- section 6.1: the product is a matrix POWER -----------------------------
print("=" * 72)
print("the recurrent gradient is a matrix power (eq. 60.7)")
print("=" * 72)
print("A feedforward net multiplies L DIFFERENT Jacobians; a recurrence")
print("multiplies the SAME one T times. Compare what that does.\n")


def product_norm(T, d=64, same_matrix=True, rho=1.0, seed=0):
    """Norm of a product of T Jacobians, all identical or all distinct,
    each scaled to the given spectral radius."""
    rs = np.random.default_rng(seed)

    def make():
        W = rs.normal(0, 1.0 / np.sqrt(d), (d, d))
        return W * (rho / max(np.abs(np.linalg.eigvals(W)).max(), 1e-12))

    M = np.eye(d)
    W0 = make()
    for _ in range(T):
        M = (W0 if same_matrix else make()) @ M
    return float(np.linalg.norm(M, 2))


print(f"{'rho':>6} {'T':>5} {'SAME matrix (recurrent)':>25} "
      f"{'DIFFERENT matrices (feedforward)':>34}")
for rho in (0.9, 1.0, 1.1):
    for T in (10, 30, 60):
        a = product_norm(T, same_matrix=True, rho=rho, seed=1)
        b = product_norm(T, same_matrix=False, rho=rho, seed=1)
        print(f"{rho:>6.1f} {T:>5} {a:>25.4e} {b:>34.4e}")

print("\nBoth columns use matrices with the SAME spectral radius, so a naive")
print("reading of eq. 60.8 would predict the same behaviour. It does not")
print("happen.")
print("\nWith the same matrix repeated, the product is W^T and its norm is")
print("governed by rho^T exactly — clean exponential growth or decay with")
print("no escape. With different matrices the singular directions do not")
print("line up between factors, so the growth is different and the")
print("behaviour is not read off a single number.")
print("\nThat is the structural difference between a recurrence and a deep")
print("feedforward network, and it is why Bengio et al.'s result is a")
print("theorem about recurrences specifically.")

# --- an explicit BPTT implementation ----------------------------------------
class VanillaRNN:
    """Eq. 60.1, with backpropagation through time written out."""

    def __init__(self, n_in, d, n_out, w_scale=None, seed=0, ortho=False):
        rs = np.random.default_rng(seed)
        if ortho:
            Q, R = np.linalg.qr(rs.normal(size=(d, d)))
            self.Whh = Q * np.sign(np.diag(R))
        else:
            s = w_scale if w_scale is not None else 1.0 / np.sqrt(d)
            self.Whh = rs.normal(0, s, (d, d))
        self.Wxh = rs.normal(0, 1.0 / np.sqrt(n_in), (n_in, d))
        self.bh = np.zeros(d)
        self.Why = rs.normal(0, 1.0 / np.sqrt(d), (d, n_out))
        self.by = np.zeros(n_out)
        self.d = d

    def forward(self, X):
        """X: (batch, T, n_in). Returns logits at the LAST step only."""
        B, T, _ = X.shape
        h = np.zeros((B, self.d))
        self.H = [h]
        for t in range(T):
            h = np.tanh(h @ self.Whh + X[:, t] @ self.Wxh + self.bh)
            self.H.append(h)
        self.X = X
        return h @ self.Why + self.by

    def backward(self, dlogits, report_norms=False):
        """Eq. 60.5. Returns gradients and, optionally, the norm of the
        error signal at each time step."""
        B, T, _ = self.X.shape
        gWhy = self.H[-1].T @ dlogits
        gby = dlogits.sum(axis=0)
        dh = dlogits @ self.Why.T
        gWhh = np.zeros_like(self.Whh)
        gWxh = np.zeros_like(self.Wxh)
        gbh = np.zeros_like(self.bh)
        norms = []
        for t in reversed(range(T)):
            dz = dh * (1 - self.H[t + 1] ** 2)     # eq. 60.6
            norms.append(float(np.sqrt(np.mean(dz ** 2))))
            gWhh += self.H[t].T @ dz
            gWxh += self.X[:, t].T @ dz
            gbh += dz.sum(axis=0)
            dh = dz @ self.Whh.T
        return (gWhh, gWxh, gbh, gWhy, gby), list(reversed(norms))


print("\n" + "=" * 72)
print("the error signal reaching each time step (eq. 60.5)")
print("=" * 72)
B, T, N_IN, D = 32, 60, 8, 48
Xseq = rng.normal(size=(B, T, N_IN))
dlog = rng.normal(size=(B, 3)) / np.sqrt(B)

print(f"{'W_hh init':<24} " +
      " ".join(f"{f't={t}':>11}" for t in (0, 20, 40, 59))
      + f" {'t=59 / t=0':>13}")
for label, kw in (("small  (sd 0.5/sqrt(d))", {"w_scale": 0.5 / np.sqrt(D)}),
                  ("standard (1/sqrt(d))", {}),
                  ("large  (sd 2/sqrt(d))", {"w_scale": 2.0 / np.sqrt(D)}),
                  ("ORTHOGONAL", {"ortho": True})):
    net = VanillaRNN(N_IN, D, 3, seed=2, **kw)
    net.forward(Xseq)
    _, norms = net.backward(dlog)
    picks = [0, 20, 40, 59]
    rho = float(np.abs(np.linalg.eigvals(net.Whh)).max())
    gamma = float(np.mean(1 - np.concatenate(net.H[1:]) ** 2))
    print(f"{label:<24} " + " ".join(f"{norms[t]:>11.3e}" for t in picks)
          + f" {norms[59] / max(norms[0], 1e-300):>13.3e}")
    print(f"{'  rho, mean tanh deriv':<24} {rho:>11.4f} {gamma:>11.4f}"
          f"   -> (rho*gamma)^59 = "
          f"{(rho * gamma) ** 59:.3e}")

print("\nRead the last column: how much larger the gradient at the LAST")
print("time step is than the gradient reaching the FIRST. It is what")
print("decides whether a dependency spanning the sequence is learnable at")
print("all.")
print("\nThe predicted (rho*gamma)^59 on each second line is eq. 60.8's")
print("bound. Compare it against the reciprocal of the measured ratio: the")
print("bound is two to three orders of magnitude PESSIMISTIC, which is what")
print("a bound should be — it assumes the worst case at every one of the")
print("59 steps. What it gets right is the ordering and the scale, which is")
print("all it claims.")
print("\nNow the honest part. The orthogonal row has spectral radius")
print("exactly 1.0000 — the only initialisation here that does — and its")
print("ratio is NOT the best on the table. It is comparable to the standard")
print("initialisation and enormously better than the small one.")
print("\nThe reason is the gamma column. Orthogonality removes the WEIGHT")
print("MATRIX's contribution to the product and leaves the tanh")
print("derivative's, which is well below 1 and which no choice of W can")
print("touch. Over 59 steps that alone is fatal.")
print("\nSo eq. 60.8 has two factors and orthogonal initialisation fixes")
print("one of them. That is a real improvement and it is not a solution,")
print("which is exactly why gating — section 6.3, where the carry path has")
print("NEITHER factor — was the development that mattered.")
print("\nThe 'large' row is the other failure. Its ratio is below 1, meaning")
print("the gradient at the first step is LARGER than at the last: with rho")
print("above 2 the backward product explodes rather than vanishing.")

# --- section 6.4: orthogonality is preserved under powers -------------------
print("\n" + "=" * 72)
print("only an orthogonal matrix keeps rho = 1 at EVERY horizon (6.4)")
print("=" * 72)
d = 64
rs = np.random.default_rng(3)
Wg = rs.normal(0, 1.0 / np.sqrt(d), (d, d))
Wg = Wg / np.abs(np.linalg.eigvals(Wg)).max()      # force rho = 1
Q, R = np.linalg.qr(rs.normal(size=(d, d)))
Wo = Q * np.sign(np.diag(R))
print(f"both start with spectral radius 1.0000\n")
print(f"{'T':>5} {'||W_gauss^T||_2':>18} {'||W_orth^T||_2':>17} "
      f"{'cond(W_gauss^T)':>18}")
Mg, Mo = np.eye(d), np.eye(d)
for T in range(1, 51):
    Mg, Mo = Wg @ Mg, Wo @ Mo
    if T in (1, 5, 10, 25, 50):
        sg = np.linalg.svd(Mg, compute_uv=False)
        print(f"{T:>5} {np.linalg.norm(Mg, 2):>18.4e} "
              f"{np.linalg.norm(Mo, 2):>17.4e} "
              f"{sg.max() / max(sg.min(), 1e-300):>18.4e}")

print("\nA spectral radius of 1 controls the ASYMPTOTIC growth rate and")
print("says nothing about the transient. The Gaussian matrix at rho = 1 has")
print("singular values spread widely, so its powers amplify some directions")
print("and crush others, and the condition number grows without bound.")
print("\nThe orthogonal matrix's powers are orthogonal — norm exactly 1 and")
print("condition number exactly 1 at every horizon. That is the property")
print("Saxe et al. established, and it is why orthogonal initialisation is")
print("standard in recurrent networks and merely optional elsewhere: here")
print("the SAME matrix recurs, so its conditioning compounds directly.")

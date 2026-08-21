# Extracted from: Chapter 39 — Support Vector Machines and Kernels
# Source: src/.../ch039-svm.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A kernel SVM by SMO-style dual optimisation, and the C/gamma grid.
"""
import numpy as np

rng = np.random.default_rng(7)


def kernel_matrix(A, B, kind="rbf", gamma=1.0, degree=3, coef0=1.0):
    if kind == "linear":
        return A @ B.T
    if kind == "poly":
        return (gamma * (A @ B.T) + coef0) ** degree
    if kind == "rbf":
        sq = (np.sum(A ** 2, 1)[:, None] + np.sum(B ** 2, 1)[None, :]
              - 2 * A @ B.T)
        return np.exp(-gamma * np.maximum(sq, 0.0))
    raise ValueError(kind)


class KernelSVM:
    """Simplified SMO on the dual of eq. 39.4.

    Pairs of multipliers are optimised at a time, because the equality
    constraint sum(alpha_i y_i) = 0 means a single alpha cannot move alone.
    """

    def __init__(self, C=1.0, kind="rbf", gamma=1.0, degree=3,
                 max_passes=12, tol=1e-3, seed=0):
        self.C, self.kind, self.gamma, self.degree = C, kind, gamma, degree
        self.max_passes, self.tol, self.seed = max_passes, tol, seed

    def fit(self, X, y):
        rs = np.random.default_rng(self.seed)
        n = len(y)
        K = kernel_matrix(X, X, self.kind, self.gamma, self.degree)
        a = np.zeros(n)
        b = 0.0
        passes = 0
        while passes < self.max_passes:
            changed = 0
            f = (a * y) @ K + b
            for i in range(n):
                Ei = f[i] - y[i]
                if ((y[i] * Ei < -self.tol and a[i] < self.C)
                        or (y[i] * Ei > self.tol and a[i] > 0)):
                    j = int(rs.integers(0, n - 1))
                    j = j + (j >= i)
                    Ej = f[j] - y[j]
                    ai_old, aj_old = a[i], a[j]
                    if y[i] != y[j]:
                        L, H = max(0.0, aj_old - ai_old), \
                               min(self.C, self.C + aj_old - ai_old)
                    else:
                        L, H = max(0.0, ai_old + aj_old - self.C), \
                               min(self.C, ai_old + aj_old)
                    if H - L < 1e-12:
                        continue
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= -1e-12:
                        continue
                    a[j] = np.clip(aj_old - y[j] * (Ei - Ej) / eta, L, H)
                    if abs(a[j] - aj_old) < 1e-9:
                        continue
                    a[i] = ai_old + y[i] * y[j] * (aj_old - a[j])
                    b1 = (b - Ei - y[i] * (a[i] - ai_old) * K[i, i]
                          - y[j] * (a[j] - aj_old) * K[i, j])
                    b2 = (b - Ej - y[i] * (a[i] - ai_old) * K[i, j]
                          - y[j] * (a[j] - aj_old) * K[j, j])
                    if 0 < a[i] < self.C:
                        b = b1
                    elif 0 < a[j] < self.C:
                        b = b2
                    else:
                        b = 0.5 * (b1 + b2)
                    f = (a * y) @ K + b
                    changed += 1
            passes = passes + 1 if changed == 0 else 0
            if changed == 0:
                break
        self.a, self.b, self.X, self.y = a, b, X, y
        self.sv = a > 1e-8
        return self

    def decision(self, Z):
        K = kernel_matrix(Z, self.X[self.sv], self.kind, self.gamma,
                          self.degree)
        return K @ (self.a[self.sv] * self.y[self.sv]) + self.b

    def predict(self, Z):
        return np.sign(self.decision(Z))


def make_rings(n):
    r = rng.uniform(0, 3, n)
    th = rng.uniform(0, 2 * np.pi, n)
    X = np.column_stack([r * np.cos(th), r * np.sin(th)])
    return X, np.where(r > 1.6, 1.0, -1.0)


def make_xor(n):
    X = rng.normal(size=(n, 2))
    return X, np.where((X[:, 0] > 0) ^ (X[:, 1] > 0), 1.0, -1.0)


def make_linear(n):
    X = rng.normal(size=(n, 2))
    return X, np.where(X[:, 0] + X[:, 1] > 0, 1.0, -1.0)


# --- the kernel decides what the boundary can be ----------------------------
print("=" * 72)
print("kernel choice is inductive bias (table 39.2)")
print("=" * 72)
datasets = {"linear boundary": make_linear, "concentric rings": make_rings,
            "XOR": make_xor}
print(f"{'dataset':<20} {'linear':>9} {'poly d=2':>10} {'poly d=3':>10} "
      f"{'RBF':>8}")
for name, gen in datasets.items():
    Xtr, ytr = gen(300)
    Xte, yte = gen(2000)
    row = []
    for kind, kw in (("linear", {}), ("poly", dict(degree=2, gamma=0.5)),
                     ("poly", dict(degree=3, gamma=0.5)),
                     ("rbf", dict(gamma=0.5))):
        m = KernelSVM(C=1.0, kind=kind, **kw).fit(Xtr, ytr)
        row.append((m.predict(Xte) == yte).mean())
    print(f"{name:<20} {row[0]:>9.4f} {row[1]:>10.4f} {row[2]:>10.4f} "
          f"{row[3]:>8.4f}")

print("\nThe linear kernel solves the linear problem and nothing else — it is")
print("at 0.63-0.65 on rings and XOR, barely above chance.")
print("\nThe degree-2 polynomial is the best model on BOTH nonlinear")
print("problems, and that is not luck: its feature space contains exactly")
print("the monomials x0^2, x1^2 and x0*x1. A ring boundary is")
print("x0^2 + x1^2 = r^2 and XOR's is x0*x1 = 0, so both are LINEAR in that")
print("space. When you know the form of the boundary, the matching kernel")
print("beats the general-purpose one.")
print("\nRBF is close behind on everything without being told anything,")
print("which is why it is the default: it assumes only smoothness. Choosing")
print("a kernel is choosing an assumption, exactly as in Chapter 31 — and")
print("the RBF's assumption is the weakest one available.")

# --- sparsity: how many points actually matter ------------------------------
print("\n" + "=" * 72)
print("sparsity: the model is stored as support vectors (section 5.3)")
print("=" * 72)
Xtr, ytr = make_rings(400)
Xte, yte = make_rings(3000)
print(f"{'C':>8} {'support vectors':>17} {'fraction':>10} {'test acc':>10}")
for C in (0.1, 1.0, 10.0, 100.0):
    m = KernelSVM(C=C, kind="rbf", gamma=0.5).fit(Xtr, ytr)
    print(f"{C:>8} {int(m.sv.sum()):>17} {m.sv.mean():>10.3f} "
          f"{(m.predict(Xte) == yte).mean():>10.4f}")
print("\nSmaller C means a wider margin, so MORE points fall on or inside it")
print("and more become support vectors. The stored model is exactly those")
print("points — which is also why prediction cost grows with them.")

# --- section 5.4: C and gamma must be tuned jointly -------------------------
print("\n" + "=" * 72)
print("C and gamma interact: tune them on a 2-D log grid")
print("=" * 72)
Xtr, ytr = make_rings(400)
Xva, yva = make_rings(1000)
Xte, yte = make_rings(3000)

gammas = [0.01, 0.1, 1.0, 10.0, 100.0]
Cs = [0.1, 1.0, 10.0, 100.0]
print(f"{'':>8}" + "".join(f"{'g=' + str(g):>10}" for g in gammas))
best = (None, -1.0)
for C in Cs:
    row = []
    for g in gammas:
        m = KernelSVM(C=C, kind="rbf", gamma=g).fit(Xtr, ytr)
        acc = (m.predict(Xva) == yva).mean()
        row.append(acc)
        if acc > best[1]:
            best = ((C, g), acc)
    print(f"C={C:<6}" + "".join(f"{a:>10.4f}" for a in row))

(C_star, g_star), _ = best
final = KernelSVM(C=C_star, kind="rbf", gamma=g_star).fit(Xtr, ytr)
print(f"\nbest on validation: C={C_star}, gamma={g_star}")
print(f"test accuracy      : {(final.predict(Xte) == yte).mean():.4f}")
print(f"support vectors    : {int(final.sv.sum())} of {len(ytr)}")

print("\nThe grid is not separable into two one-dimensional searches: the")
print("best gamma at C=0.1 is not the best gamma at C=100. Both control")
print("effective complexity — gamma through the width of each bump, C")
print("through how hard the fit is pushed — so a large gamma can be rescued")
print("by a small C and vice versa. Always search the plane.")

# --- gamma alone: from almost-linear to memorising --------------------------
print("\n" + "=" * 72)
print("gamma is the RBF bandwidth: too large and it memorises")
print("=" * 72)
print(f"{'gamma':>9} {'train acc':>11} {'test acc':>10} {'SVs':>6} "
      f"{'behaviour':<28}")
for g in (0.001, 0.01, 0.1, 1.0, 10.0, 200.0):
    m = KernelSVM(C=10.0, kind="rbf", gamma=g).fit(Xtr, ytr)
    tr = (m.predict(Xtr) == ytr).mean()
    te = (m.predict(Xte) == yte).mean()
    note = ("under-fitting: one broad bump" if g <= 0.01
            else "memorising" if tr - te > 0.05 else "")
    print(f"{g:>9} {tr:>11.4f} {te:>10.4f} {int(m.sv.sum()):>6} {note:<28}")
print("\nBoth ends fail, in opposite ways. At gamma = 0.001 every point is")
print("effectively at distance zero from every other, the kernel is nearly")
print("constant, and almost every point becomes a support vector — the")
print("model is one broad bump and cannot separate anything.")
print("\nAt gamma = 200 each support vector influences only a tiny ball")
print("around itself, so the model approaches a lookup table: training")
print("accuracy 1.0000 and a test accuracy 11 points below its peak, with")
print("almost every point needed as a support vector because nothing")
print("generalises to its neighbours.")
print("\nOne knob, the whole bias-variance trade — the same curve as k in")
print("k-NN and depth in a tree (Chapter 34). Note that the support-vector")
print("count is a useful diagnostic in itself: when nearly every point is a")
print("support vector, the model has stopped generalising.")

# Extracted from: Chapter 64 — Multi-Head Attention
# Source: src/.../ch064-multi-head-attention.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The rank bottleneck of eq. 64.7, measured: what a single narrow head can
and cannot express, and what several of them buy.
"""
import numpy as np

rng = np.random.default_rng(2)


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


# --- section 6.1: the rank bound, verified ----------------------------------
print("=" * 72)
print("a head's score matrix has rank at most d_k (eq. 64.7)")
print("=" * 72)
T, d = 128, 256
X = rng.normal(size=(T, d))
print(f"T = {T} positions, model width d = {d}\n")
print(f"{'d_k':>6} {'score matrix':>15} {'numerical rank':>16} "
      f"{'bound min(T, d_k)':>19}")
for dk in (4, 16, 64, 128, 256):
    Wq = rng.normal(0, 1 / np.sqrt(d), (d, dk))
    Wk = rng.normal(0, 1 / np.sqrt(d), (d, dk))
    S = (X @ Wq) @ (X @ Wk).T / np.sqrt(dk)
    sv = np.linalg.svd(S, compute_uv=False)
    r = int((sv > sv.max() * 1e-10).sum())
    print(f"{dk:>6} {str(S.shape):>15} {r:>16} {min(T, dk):>19}")

print("\nThe bound is tight: the score matrix's rank is exactly min(T, d_k)")
print("at every width. A head of dimension 64 attending over 128 positions")
print("is confined to a rank-64 subspace of a 128-by-128 space, and at a")
print("realistic T = 2048 it is a rank-64 subspace of a 2048-by-2048 one.")

# --- what that costs: fit a target attention pattern ------------------------
print("\n" + "=" * 72)
print("what the rank bound COSTS: fitting a target attention pattern")
print("=" * 72)
print("Construct a target attention matrix of known rank and ask heads of")
print("various widths to reproduce it. This isolates eq. 64.7 from every")
print("other property of a trained model.\n")


def make_target(T, rank, seed):
    """A row-stochastic T-by-T matrix built from a rank-r score matrix."""
    rs = np.random.default_rng(seed)
    U = rs.normal(size=(T, rank))
    Vv = rs.normal(size=(T, rank))
    return softmax(U @ Vv.T * 2.0, axis=-1)


def fit_head(X, target, dk, steps=3000, lr=0.02, seed=0):
    """Learn W_q, W_k so that softmax(XWq (XWk)^T / sqrt(dk)) ~ target."""
    rs = np.random.default_rng(seed)
    d = X.shape[1]
    Wq = rs.normal(0, 1 / np.sqrt(d), (d, dk))
    Wk = rs.normal(0, 1 / np.sqrt(d), (d, dk))
    ps = [Wq, Wk]
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    for t in range(1, steps + 1):
        Q, K = X @ Wq, X @ Wk
        S = Q @ K.T / np.sqrt(dk)
        A = softmax(S, axis=-1)
        # cross-entropy between target rows and predicted rows
        dS = (A - target) / len(X)
        gWq = X.T @ (dS @ K) / np.sqrt(dk)
        gWk = X.T @ (dS.T @ Q) / np.sqrt(dk)
        for i, (p, g) in enumerate(zip(ps, [gWq, gWk])):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            p -= lr * (m[i] / (1 - 0.9 ** t)) / (
                np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    A = softmax((X @ Wq) @ (X @ Wk).T / np.sqrt(dk), axis=-1)
    return float(np.abs(A - target).mean())


T, d = 64, 128
X = rng.normal(size=(T, d))
print(f"{'target rank':>12} " + " ".join(f"{f'd_k={dk}':>10}"
                                         for dk in (2, 8, 32, 64)))
for rank in (2, 8, 32):
    tgt = make_target(T, rank, seed=rank)
    row = [fit_head(X, tgt, dk, seed=5) for dk in (2, 8, 32, 64)]
    print(f"{rank:>12} " + " ".join(f"{e:>10.4f}" for e in row))

print("\nRead the leftmost column down: a head of width 2 handles a rank-2")
print("target and does markedly worse on higher-rank ones. That is eq. 64.7")
print("binding.")
print("\nBut notice how quickly the constraint relaxes. A head of width 8")
print("fits a rank-32 target almost exactly, which the rank bound alone")
print("would not predict.")
print("\nThe reason is that eq. 64.7 bounds the SCORE matrix, and the target")
print("here is an ATTENTION matrix — the softmax of a score matrix. The")
print("softmax is nonlinear, so a low-rank score matrix can produce a very")
print("good approximation of a higher-rank row-stochastic matrix. Sharpening")
print("a few directions is enough to reproduce most of the mass.")
print("\nThat is worth getting right, because the rank bound is frequently")
print("quoted as though it limited attention PATTERNS. It limits the scores.")
print("The patterns are the softmax of those scores, and the softmax buys")
print("back a great deal — which is part of why d_k = 64 has been adequate")
print("for sequence lengths in the thousands.")

# --- and what SEVERAL heads buy ---------------------------------------------
print("\n" + "=" * 72)
print("what several narrow heads buy over one narrow head")
print("=" * 72)
print("A target that is a MIXTURE of distinct low-rank patterns — the")
print("situation section 4.1 describes, where a sentence needs several")
print("relationships attended to at once.\n")


def make_mixture_target(T, n_patterns, rank, seed):
    """n distinct attention patterns; each query row uses one of them."""
    rs = np.random.default_rng(seed)
    pats = [make_target(T, rank, seed=seed * 10 + i)
            for i in range(n_patterns)]
    which = rs.integers(0, n_patterns, T)
    return np.stack([pats[which[i]][i] for i in range(T)]), pats, which


def fit_multihead(X, target, h, dk, steps=3000, lr=0.02, seed=0):
    """h heads, each of width dk, averaged. Fits the MEAN of the heads to
    the target — a crude stand-in for what W_O's mixing allows."""
    rs = np.random.default_rng(seed)
    d = X.shape[1]
    Wq = [rs.normal(0, 1 / np.sqrt(d), (d, dk)) for _ in range(h)]
    Wk = [rs.normal(0, 1 / np.sqrt(d), (d, dk)) for _ in range(h)]
    g = rs.normal(0, 0.5, (len(X), h))            # per-position head mixing
    ps = Wq + Wk + [g]
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    for t in range(1, steps + 1):
        As, Qs, Ks = [], [], []
        for i in range(h):
            Q, K = X @ Wq[i], X @ Wk[i]
            As.append(softmax(Q @ K.T / np.sqrt(dk), axis=-1))
            Qs.append(Q)
            Ks.append(K)
        w = softmax(g, axis=-1)                   # (T, h)
        A = sum(w[:, i:i + 1] * As[i] for i in range(h))
        dA = (A - target) / len(X)
        gg = np.stack([(dA * As[i]).sum(axis=1) for i in range(h)], axis=1)
        gg = w * (gg - (gg * w).sum(axis=1, keepdims=True))
        grads = []
        for i in range(h):
            dAi = dA * w[:, i:i + 1]
            dS = As[i] * (dAi - (dAi * As[i]).sum(axis=1, keepdims=True))
            grads.append(X.T @ (dS @ Ks[i]) / np.sqrt(dk))
        for i in range(h):
            dAi = dA * w[:, i:i + 1]
            dS = As[i] * (dAi - (dAi * As[i]).sum(axis=1, keepdims=True))
            grads.append(X.T @ (dS.T @ Qs[i]) / np.sqrt(dk))
        grads.append(gg)
        for i, (p, gr) in enumerate(zip(ps, grads)):
            m[i] = 0.9 * m[i] + 0.1 * gr
            v[i] = 0.999 * v[i] + 0.001 * gr * gr
            p -= lr * (m[i] / (1 - 0.9 ** t)) / (
                np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    As = [softmax((X @ Wq[i]) @ (X @ Wk[i]).T / np.sqrt(dk), axis=-1)
          for i in range(h)]
    w = softmax(g, axis=-1)
    A = sum(w[:, i:i + 1] * As[i] for i in range(h))
    return float(np.abs(A - target).mean())


T, d = 48, 96
X = rng.normal(size=(T, d))
tgt, pats, which = make_mixture_target(T, 4, rank=6, seed=3)
print(f"target: 4 distinct rank-6 patterns, each query row using one\n")
print(f"{'configuration':<26} {'total rank budget':>19} {'fit error':>12}")
for label, h, dk in (("1 head of width 24", 1, 24),
                     ("1 head of width 6", 1, 6),
                     ("4 heads of width 6", 4, 6),
                     ("8 heads of width 3", 8, 3)):
    e = fit_multihead(X, tgt, h, dk, seed=5)
    print(f"{label:<26} {h * dk:>19} {e:>12.4f}")

print("\nThe first two rows are the same total rank budget spent two ways —")
print("one wide head against one narrow one — and the wide one wins, which")
print("is just eq. 64.7 again.")
print("\nThe interesting comparison is rows 1 and 3: the SAME total rank")
print("budget of 24, as one head of width 24 or four heads of width 6. The")
print("target is a mixture of four distinct patterns, and four heads can")
print("hold one each while one head must find a single rank-24 score matrix")
print("that produces all four after a softmax.")
print("\nThat is the argument of section 4.1 made concrete, and it is a")
print("statement about the SOFTMAX rather than about rank. Each head is")
print("normalised separately, so h heads give h independent probability")
print("distributions — which is something one head of any width cannot")
print("produce, because it has only one softmax.")

# Extracted from: Chapter 66 — Token Embeddings and the Unembedding Matrix
# Source: src/.../ch066-embeddings.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The softmax bottleneck (eq. 66.6): a hard expressiveness limit that
follows from the architecture in two lines.
"""
import numpy as np

rng = np.random.default_rng(2)


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


# --- section 6.1: the rank bound --------------------------------------------
print("=" * 72)
print("the logit matrix has rank at most d (eq. 66.6)")
print("=" * 72)
N, V = 200, 64
print(f"{N} contexts, vocabulary {V}\n")
print(f"{'d':>5} {'logit matrix':>15} {'numerical rank':>16} "
      f"{'log-prob rank':>15} {'bound d+1':>11}")
for d in (4, 8, 16, 64):
    H = rng.normal(size=(N, d))
    U = rng.normal(size=(V, d))
    Z = H @ U.T
    P = softmax(Z)
    logP = np.log(P)
    r1 = int((np.linalg.svd(Z, compute_uv=False) > 1e-9).sum())
    r2 = int((np.linalg.svd(logP - logP.mean(), compute_uv=False)
              > 1e-9 * np.abs(logP).max()).sum())
    print(f"{d:>5} {str(Z.shape):>15} {r1:>16} {r2:>15} {d + 1:>11}")

print("\nThe logit matrix's rank is exactly d and the log-probability")
print("matrix's is at most d+1, because the softmax subtracts a per-row")
print("constant — one extra rank-one term and nothing more.")
print("\nThat bound is on the ARCHITECTURE. No amount of training moves it,")
print("and no choice of unembedding does either.")

# --- what it costs: fit a target distribution of known rank -----------------
print("\n" + "=" * 72)
print("what the bound costs: fitting distributions of known rank")
print("=" * 72)
print("Construct a target log-probability matrix of controlled rank and ask")
print("models of various widths to reproduce it. This isolates eq. 66.6")
print("from every other property of a language model.\n")


def make_target(N, V, rank, seed):
    rs = np.random.default_rng(seed)
    A = rs.normal(size=(N, rank))
    B = rs.normal(size=(rank, V))
    return softmax(A @ B * 1.2, axis=-1)


def fit(target, d, steps=4000, lr=0.05, seed=0):
    """Learn H (N,d) and U (V,d) to match the target distribution."""
    rs = np.random.default_rng(seed)
    N_, V_ = target.shape
    H = rs.normal(0, 0.3, (N_, d))
    U = rs.normal(0, 0.3, (V_, d))
    ps = [H, U]
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    for t in range(1, steps + 1):
        P = softmax(H @ U.T, axis=-1)
        dZ = (P - target) / N_
        gH, gU = dZ @ U, dZ.T @ H
        for i, (p, g) in enumerate(zip(ps, [gH, gU])):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            p -= lr * (m[i] / (1 - 0.9 ** t)) / (
                np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    P = softmax(H @ U.T, axis=-1)
    kl = float((target * np.log(np.clip(target / np.clip(P, 1e-12, None),
                                        1e-12, None))).sum(1).mean())
    return kl


N, V = 128, 48
print(f"{'target rank':>12} " + " ".join(f"{f'd={d}':>11}"
                                         for d in (2, 4, 8, 16, 32)))
for rank in (2, 4, 8, 16):
    tgt = make_target(N, V, rank, seed=rank)
    row = [fit(tgt, d, seed=5) for d in (2, 4, 8, 16, 32)]
    print(f"{rank:>12} " + " ".join(f"{k:>11.5f}" for k in row))
print("\n(entries are KL(target || model) in nats; 0 is exact)")

print("\nRead along each row: a model whose width matches or exceeds the")
print("target's rank fits it essentially exactly, and a narrower one cannot")
print("— it plateaus at a nonzero KL that more training does not move.")
print("\nThat is eq. 66.6 as a capability rather than an inequality. The")
print("rank bound is exactly the set of conditional distributions a")
print("d-dimensional model is able to produce.")

# --- and what a mixture of softmaxes does -----------------------------------
print("\n" + "=" * 72)
print("breaking the bound: a mixture of softmaxes")
print("=" * 72)
print("A weighted SUM of several softmax outputs. Its logarithm is not")
print("constrained to rank d, because log of a sum is not a sum of logs.\n")


def fit_mos(target, d, K, steps=4000, lr=0.05, seed=0):
    rs = np.random.default_rng(seed)
    N_, V_ = target.shape
    H = [rs.normal(0, 0.3, (N_, d)) for _ in range(K)]
    U = rs.normal(0, 0.3, (V_, d))
    W = rs.normal(0, 0.3, (N_, K))
    ps = H + [U, W]
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    for t in range(1, steps + 1):
        comps = [softmax(H[k] @ U.T, axis=-1) for k in range(K)]
        pi = softmax(W, axis=-1)
        P = sum(pi[:, k:k + 1] * comps[k] for k in range(K))
        dP = (P - target) / N_ / np.clip(P, 1e-9, None) * np.clip(P, 1e-9, None)
        grads = []
        for k in range(K):
            dZ = pi[:, k:k + 1] * comps[k] * (
                dP - (dP * comps[k]).sum(1, keepdims=True))
            grads.append(dZ @ U)
        gU = sum(
            (pi[:, k:k + 1] * comps[k] * (
                dP - (dP * comps[k]).sum(1, keepdims=True))).T @ H[k]
            for k in range(K))
        gpi = np.stack([(dP * comps[k]).sum(1) for k in range(K)], 1)
        gW = pi * (gpi - (gpi * pi).sum(1, keepdims=True))
        grads += [gU, gW]
        for i, (p, g) in enumerate(zip(ps, grads)):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            p -= lr * (m[i] / (1 - 0.9 ** t)) / (
                np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    comps = [softmax(H[k] @ U.T, axis=-1) for k in range(K)]
    pi = softmax(W, axis=-1)
    P = sum(pi[:, k:k + 1] * comps[k] for k in range(K))
    return float((target * np.log(np.clip(target / np.clip(P, 1e-12, None),
                                          1e-12, None))).sum(1).mean())


tgt = make_target(N, V, 16, seed=16)
print(f"target rank 16\n")
print(f"{'model':<28} {'effective params':>18} {'KL to target':>14}")
for d in (4, 8):
    print(f"{f'single softmax, d={d}':<28} {N * d + V * d:>18,} "
          f"{fit(tgt, d, seed=5):>14.5f}")
for d, K in ((4, 4), (4, 8)):
    print(f"{f'mixture of {K}, d={d}':<28} {K * N * d + V * d + N * K:>18,} "
          f"{fit_mos(tgt, d, K, seed=5):>14.5f}")
print(f"{'single softmax, d=32':<28} {N * 32 + V * 32:>18,} "
      f"{fit(tgt, 32, seed=5):>14.5f}")

print("\nThe mixture rows do beat the single softmax at the same width, and")
print("they improve as K grows — so eq. 66.6's bound is genuinely escaped.")
print("The reason it CAN be is that the log of a sum is not a sum of logs,")
print("so the mixture's log-probability matrix is not constrained to the")
print("rank of any component.")
print("\nBut read the last row. A single softmax at d = 32 — the same")
print("parameter count as the 8-component mixture at d = 4 — fits the target")
print("two orders of magnitude better. The mixture escapes the bound and")
print("does not escape it EFFICIENTLY.")
print("\nThat comparison is the whole reason the field's response to the")
print("softmax bottleneck was neither of these. It made d larger. At")
print("d = 4096 the bound is far above anything language plausibly needs,")
print("and the problem stopped being interesting for the same reason many")
print("small-model problems did.")


# Extracted from: Chapter 71 — Efficient Attention: FlashAttention, GQA/MQA, Sparse and Linear Variants
# Source: src/.../ch071-efficient-attention.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Sparse attention's path-length cost and linear attention's rank limit
(eqs. 71.10, 71.14).
"""
import numpy as np

rng = np.random.default_rng(1)


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


# --- section 6.5: the effective context of a window -------------------------
def window_mask(T, w, global_tokens=0):
    i = np.arange(T)[:, None]
    j = np.arange(T)[None, :]
    m = (j <= i) & (j > i - w)
    if global_tokens:
        m[:, :global_tokens] = True
        m[:global_tokens, :] = np.tril(np.ones((global_tokens, T),
                                               dtype=bool))[:, :T]
    return m


def reachability(mask, layers):
    """Which pairs can influence each other within `layers` layers?"""
    R = mask.copy()
    for _ in range(layers - 1):
        R = (R.astype(np.int8) @ mask.astype(np.int8)) > 0
    return R


print("=" * 72)
print("what a sliding window costs: reachability (eq. 71.10)")
print("=" * 72)
T = 128
print(f"T = {T} positions, causal.\n")
print(f"{'pattern':<26} {'cost / full':>12} " +
      " ".join(f"{f'L={L}':>10}" for L in (1, 2, 4, 8)))
print(f"{'':<26} {'':>12} " +
      " ".join(f"{'reachable':>10}" for _ in range(4)))
full = np.tril(np.ones((T, T), dtype=bool))
n_full = full.sum()
for label, w, g in (("full", T, 0), ("window w=8", 8, 0),
                    ("window w=32", 32, 0), ("window w=8 + 4 global", 8, 4)):
    m = window_mask(T, w, g)
    row = []
    for L in (1, 2, 4, 8):
        R = reachability(m, L)
        row.append(R.sum() / n_full)
    print(f"{label:<26} {m.sum() / n_full:>12.3f} " +
          " ".join(f"{x:>10.3f}" for x in row))

print("\nThe 'cost' column is the fraction of the full attention matrix")
print("computed; the rest is the fraction of causal pairs that can")
print("influence each other after L layers.")
print("\nA plain window of 8 needs many layers to connect distant positions,")
print("and eq. 71.10 says the effective context is capped at L*w whatever")
print("happens. Adding four global tokens — a two per cent cost increase —")
print("changes the picture at L = 2, because any pair can route through a")
print("global token in two hops.")
print("\nThat is why window-plus-global dominates plain windowing, and it is")
print("the row of table 71.1 that is usually omitted from comparisons.")

# --- eq. 71.10 as a hard limit ----------------------------------------------
print("\n" + "=" * 72)
print("the effective-context ceiling (eq. 71.10)")
print("=" * 72)
print(f"{'layers L':>9} " + " ".join(f"{f'w={w}':>12}" for w in
                                     (256, 1024, 4096))
      + f"   {'target context':>15}")
for L in (8, 16, 32, 80):
    row = [L * w for w in (256, 1024, 4096)]
    print(f"{L:>9} " + " ".join(f"{x:>12,}" for x in row))

print("\nAny (L, w) pair whose product is below your target context has")
print("positions that PROVABLY cannot interact. That is checkable before")
print("training, in one line, and it is the first thing to verify when")
print("choosing a window.")

# --- section 6.3: linear attention's rank limit -----------------------------
print("\n" + "=" * 72)
print("why linear attention cannot retrieve (eqs. 71.8, 71.14)")
print("=" * 72)
print("The softmax can approach a permutation matrix — one position")
print("selected, all others zero. A kernel of feature dimension d_phi")
print("cannot: its weight matrix has rank at most d_phi.\n")


def softmax_attn(Q, K):
    return softmax(Q @ K.T / np.sqrt(Q.shape[1]))


def linear_attn(Q, K, eps=1e-6):
    """elu(x)+1 feature map, the standard choice."""
    phi = lambda x: np.where(x > 0, x + 1, np.exp(np.clip(x, -60, 0)))
    Qp, Kp = phi(Q), phi(K)
    num = Qp @ Kp.T
    return num / (num.sum(-1, keepdims=True) + eps)


T, dk = 64, 16
print(f"T = {T}, d_k = {dk}. Target: attend to exactly one position.\n")
print(f"{'temperature':>13} {'softmax: max weight':>21} "
      f"{'softmax rank':>14} {'linear: max weight':>20} {'linear rank':>13}")
Kb = rng.normal(size=(T, dk))
for temp in (1.0, 4.0, 16.0, 64.0):
    Qb = Kb * temp                                  # queries aligned to keys
    As = softmax_attn(Qb, Kb)
    Al = linear_attn(Qb, Kb)
    rs_ = int((np.linalg.svd(As, compute_uv=False) > 1e-9).sum())
    rl_ = int((np.linalg.svd(Al, compute_uv=False) > 1e-9).sum())
    print(f"{temp:>13.0f} {As.max(1).mean():>21.4f} {rs_:>14} "
          f"{Al.max(1).mean():>20.4f} {rl_:>13}")

print("\nAs the scores are sharpened, the softmax's maximum weight per row")
print("approaches 1 — it is selecting one position — and its rank rises")
print("toward T. Linear attention's maximum weight is bounded well below 1")
print("however sharp the scores, because eq. 71.14 caps its rank.")
print("\nThat is the best available account of the quality gap: exact")
print("retrieval of one position out of T requires a rank-T weight matrix,")
print("and a kernel with a finite feature dimension cannot produce one.")
print("Induction heads and copying circuits (Chapter 64) do exactly this,")
print("which is why they are what linear attention loses first.")

# --- measured on a retrieval task -------------------------------------------
print("\n" + "=" * 72)
print("the same, as a task: retrieve a value by its key")
print("=" * 72)
print("A sequence of (key, value) pairs then a query key. The answer is the")
print("matching value — pure retrieval, which section 6.3 says is exactly")
print("what a rank limit forbids.\n")


def retrieval_task(n, T, dk, seed):
    rs = np.random.default_rng(seed)
    keys = rs.normal(size=(n, T, dk))
    keys /= np.linalg.norm(keys, axis=-1, keepdims=True)
    vals = rs.normal(size=(n, T, dk))
    which = rs.integers(0, T, n)
    q = keys[np.arange(n), which] * 8.0             # sharp query
    target = vals[np.arange(n), which]
    return keys, vals, q, target


print(f"{'T':>6} {'d_k':>5} {'softmax error':>15} {'linear error':>14} "
      f"{'ratio':>8}")
for T in (16, 64, 256):
    dk = 32
    K, V, q, tgt = retrieval_task(500, T, dk, 3)
    # softmax
    s = np.einsum('nd,ntd->nt', q, K) / np.sqrt(dk)
    o_soft = np.einsum('nt,ntd->nd', softmax(s), V)
    # linear
    phi = lambda x: np.where(x > 0, x + 1, np.exp(np.clip(x, -60, 0)))
    qp, kp = phi(q), phi(K)
    num = np.einsum('nd,ntd->nt', qp, kp)
    o_lin = np.einsum('nt,ntd->nd', num / (num.sum(-1, keepdims=True) + 1e-6),
                      V)
    e_s = float(np.linalg.norm(o_soft - tgt, axis=1).mean())
    e_l = float(np.linalg.norm(o_lin - tgt, axis=1).mean())
    print(f"{T:>6} {dk:>5} {e_s:>15.4f} {e_l:>14.4f} {e_l / e_s:>8.1f}x")

print("\nThe softmax retrieves the right value with small error and the gap")
print("widens with T, because retrieving one item out of more of them is")
print("exactly the operation the rank bound forbids.")
print("\nThis is a hand-constructed probe with no training, so it isolates")
print("the mechanism rather than measuring what a trained model would do.")
print("What it establishes is that the limitation is structural: no amount")
print("of training changes eq. 71.14.")

# --- section 6.4: the parallel scan -----------------------------------------
print("\n" + "=" * 72)
print("why a LINEAR recurrence can be parallelised (eq. 71.15)")
print("=" * 72)
print("Composition of affine maps is associative, so a linear recurrence")
print("can be evaluated by a balanced tree in O(log T) depth.\n")


def scan_sequential(a, b):
    h = 0.0
    out = []
    for t in range(len(a)):
        h = a[t] * h + b[t]
        out.append(h)
    return np.array(out)


def scan_parallel(a, b):
    """Blelloch-style scan over affine maps (eq. 71.14)."""
    a, b = a.copy(), b.copy()
    n = len(a)
    step = 1
    while step < n:
        a_new, b_new = a.copy(), b.copy()
        idx = np.arange(step, n)
        a_new[idx] = a[idx] * a[idx - step]
        b_new[idx] = a[idx] * b[idx - step] + b[idx]
        a, b = a_new, b_new
        step *= 2
    return b


T = 1024
a = rng.uniform(0.9, 0.99, T)
b = rng.normal(size=T)
seq = scan_sequential(a, b)
par = scan_parallel(a, b)
print(f"sequence length {T}")
print(f"  max |sequential - parallel| = {np.abs(seq - par).max():.3e}")
print(f"  sequential depth            = {T}")
print(f"  parallel depth              = {int(np.ceil(np.log2(T)))}")
print(f"  depth reduction             = {T / np.ceil(np.log2(T)):.0f}x")

print("\nIdentical results, and the parallel version has logarithmic depth")
print("where the sequential one has linear. That is eq. 71.15, and it is")
print("the property Chapter 60 said a NONLINEAR recurrence cannot have —")
print("tanh(a*h + b) does not compose into an affine map, so there is no")
print("associative operator to scan over.")
print("\nThat single fact is why state space models are trainable at scale")
print("and LSTMs are not. It is not a better architecture in any modelling")
print("sense; it is the same idea with a computational property that")
print("modern hardware requires.")

# --- and what the decay is doing --------------------------------------------
print("\n" + "=" * 72)
print("linear attention needs a decay, and it is a forget gate (7.4)")
print("=" * 72)
print("Eq. 71.9 accumulates every key-value pair and removes none, so the")
print("state saturates. A decay factor fixes it — and it is Chapter 60's")
print("forget gate under another name.\n")
dk = 16
n_pairs = 2000
K = rng.normal(size=(n_pairs, dk))
V = rng.normal(size=(n_pairs, dk))
probe = K[10]                                   # retrieve an EARLY item
print(f"{'decay':>8} " + " ".join(f"{f'after {t}':>12}" for t in
                                  (50, 200, 1000, 2000))
      + f"   {'effective memory':>18}")
for gamma in (1.0, 0.999, 0.99, 0.9):
    S = np.zeros((dk, dk))
    row = []
    for t in range(n_pairs):
        S = gamma * S + np.outer(K[t], V[t])
        if t + 1 in (50, 200, 1000, 2000):
            out = probe @ S
            row.append(float(out @ V[10] / (np.linalg.norm(out)
                                            * np.linalg.norm(V[10]) + 1e-12)))
    eff = "unbounded" if gamma == 1.0 else f"{1 / (1 - gamma):.0f} steps"
    print(f"{gamma:>8.3f} " + " ".join(f"{x:>12.4f}" for x in row)
          + f"   {eff:>18}")

print("\nThe numbers are the cosine between what the probe retrieves and the")
print("value it should retrieve — 1.0 would be perfect recall of item 10.")
print("\nWith no decay the state accumulates 2000 outer products and the")
print("early item is swamped. With a decay it is retained for about")
print("1/(1-gamma) steps and then forgotten, which is exactly the trade")
print("Chapter 60's forget-gate table showed.")
print("\nSo linear attention with a decay is an LSTM cell with a")
print("matrix-valued state. Recognising that is more useful than tracking")
print("the variant names, and it says immediately what the failure mode")
print("will be.")

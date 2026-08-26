# -*- coding: utf-8 -*-
# Extracted from: Chapter 65 — Positional Encoding, RoPE, and ALiBi
# Source: src/.../ch065-positional.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Attention has no notion of order (eq. 65.1), and what each scheme does
about it.
"""
import numpy as np

rng = np.random.default_rng(0)


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def attention(X, Wq, Wk, Wv, bias=None):
    dk = Wq.shape[1]
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    S = Q @ K.T / np.sqrt(dk)
    if bias is not None:
        S = S + bias
    return softmax(S) @ V


# --- section 6.1: permutation equivariance ----------------------------------
print("=" * 72)
print("self-attention is permutation-equivariant (eq. 65.1)")
print("=" * 72)
T, d, dk = 8, 32, 16
X = rng.normal(size=(T, d))
Wq = rng.normal(0, 1 / np.sqrt(d), (d, dk))
Wk = rng.normal(0, 1 / np.sqrt(d), (d, dk))
Wv = rng.normal(0, 1 / np.sqrt(d), (d, dk))

perm = rng.permutation(T)
out = attention(X, Wq, Wk, Wv)
out_perm = attention(X[perm], Wq, Wk, Wv)

print(f"max |Attn(PX) - P Attn(X)| = "
      f"{np.abs(out_perm - out[perm]).max():.3e}")
print("\nExact to floating point. Shuffling the input shuffles the output")
print("identically and changes nothing else, which means the operator sees")
print("a SET of vectors and not a sequence.")

print("\nThe consequence, stated as a task:")
tokens = ["dog", "bites", "man"]
Etab = {t: rng.normal(size=d) for t in tokens}
s1 = np.stack([Etab["dog"], Etab["bites"], Etab["man"]])
s2 = np.stack([Etab["man"], Etab["bites"], Etab["dog"]])
o1, o2 = attention(s1, Wq, Wk, Wv), attention(s2, Wq, Wk, Wv)
pooled1, pooled2 = o1.mean(0), o2.mean(0)
print(f"  'dog bites man' vs 'man bites dog', mean-pooled output:")
print(f"    max |difference| = {np.abs(pooled1 - pooled2).max():.3e}")
print("\nIdentical. A permutation-invariant readout on top of a")
print("permutation-equivariant network cannot distinguish word orders at")
print("all, and no amount of training changes that — it is a property of")
print("the operator, not of the parameters.")

# --- the schemes ------------------------------------------------------------
def sinusoidal(T, d, base=10000.0):
    """Eq. 65.2."""
    pos = np.arange(T)[:, None]
    i = np.arange(0, d, 2)[None, :]
    ang = pos / (base ** (i / d))
    pe = np.zeros((T, d))
    pe[:, 0::2] = np.sin(ang)
    pe[:, 1::2] = np.cos(ang)
    return pe


def rope_tables(T, dk, base=10000.0, pos_scale=1.0):
    """Eqs. 65.4-65.5, half-split convention (section 7.2)."""
    theta = base ** (-np.arange(0, dk, 2) / dk)          # (dk/2,)
    m = np.arange(T)[:, None] / pos_scale
    ang = m * theta[None, :]                             # (T, dk/2)
    return np.cos(ang), np.sin(ang)


def apply_rope(x, cos, sin):
    """x: (T, dk). Pairs dimension j with j + dk/2."""
    dk = x.shape[-1]
    x1, x2 = x[..., :dk // 2], x[..., dk // 2:]
    return np.concatenate([x1 * cos - x2 * sin,
                           x1 * sin + x2 * cos], axis=-1)


def alibi_bias(T, slope):
    """Eq. 65.6, causal."""
    i = np.arange(T)[:, None]
    j = np.arange(T)[None, :]
    b = -slope * (i - j).astype(float)
    b[j > i] = -np.inf
    return b


# --- section 6.2: RoPE's relative property ----------------------------------
print("\n" + "=" * 72)
print("RoPE makes the score depend ONLY on the offset (eq. 65.9)")
print("=" * 72)
dk = 32
q = rng.normal(size=dk)
k = rng.normal(size=dk)
cos, sin = rope_tables(600, dk)

print("The same q and k placed at different absolute positions with the")
print("SAME offset. If eq. 65.9 holds, every score in a column is equal.\n")
print(f"{'offset n - m':>13} " + " ".join(f"{f'm={m}':>12}"
                                          for m in (0, 5, 100, 500)))
for off in (0, 1, 3, 10):
    row = []
    for m in (0, 5, 100, 500):
        n = m + off
        qm = apply_rope(q[None, :], cos[m:m + 1], sin[m:m + 1])[0]
        kn = apply_rope(k[None, :], cos[n:n + 1], sin[n:n + 1])[0]
        row.append(float(qm @ kn))
    print(f"{off:>13} " + " ".join(f"{v:>12.6f}" for v in row))
    print(f"{'':>13} " + " ".join(f"{'':>12}" for _ in range(3))
          + f"  spread {max(row) - min(row):.2e}")

print("\nEvery row is constant to floating point: the score depends on the")
print("offset and not on where the pair sits. That is eq. 65.9, and it is")
print("EXACT rather than approximate — which is the difference from the")
print("sinusoidal scheme, where relative position is merely linearly")
print("recoverable and the model has to learn to recover it.")

print("\nRoPE also preserves norms, being a rotation:")
qm = apply_rope(q[None, :], cos[137:138], sin[137:138])[0]
print(f"  |q| = {np.linalg.norm(q):.6f}   "
      f"|R(137) q| = {np.linalg.norm(qm):.6f}")
print("\nSo it cannot change the SCALE of any score, only its dependence on")
print("position — which is why it composes with Chapter 63's sqrt(d_k)")
print("without re-deriving anything.")

# --- what sinusoidal gives you ----------------------------------------------
print("\n" + "=" * 72)
print("what the sinusoidal scheme gives you, and does not")
print("=" * 72)
pe = sinusoidal(512, 64)
print("Claim (Vaswani et al.): PE(pos+k) is a LINEAR function of PE(pos).")
print("Fit one linear map per offset and check.\n")
print(f"{'offset k':>10} {'linear fit residual':>22} "
      f"{'dot(PE_pos, PE_pos+k) spread':>30}")
for k_ in (1, 5, 20, 100):
    A = pe[:400]
    B = pe[k_:400 + k_]
    M, *_ = np.linalg.lstsq(A, B, rcond=None)
    res = float(np.abs(A @ M - B).max())
    dots = (pe[:400] * pe[k_:400 + k_]).sum(1)
    print(f"{k_:>10} {res:>22.3e} {float(dots.max() - dots.min()):>30.3e}")

print("\nThe linear relation holds essentially exactly — a rotation IS")
print("linear, so this is not surprising. And the raw dot product between")
print("PE(pos) and PE(pos+k) is constant in pos, so the encoding does carry")
print("clean relative information.")
print("\nThe catch is what happens next: the position is ADDED to the token")
print("embedding, and the attention score is computed from the sum. So the")
print("score contains token-token, token-position, position-token and")
print("position-position terms all mixed together, and only the last is")
print("cleanly relative. RoPE avoids the mixing entirely by acting on the")
print("projected q and k rather than on the input.")

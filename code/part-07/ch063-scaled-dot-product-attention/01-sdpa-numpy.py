# Extracted from: Chapter 63 — Scaled Dot-Product Attention
# Source: src/.../ch063-scaled-dot-product-attention.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Scaled dot-product attention from first principles, in NumPy.

Every intermediate is named and shape-checked so that the mapping between the
equation and the code is one-to-one.
"""
import numpy as np


def softmax(z, axis=-1):
    """Numerically stable row-wise softmax.

    Subtracting the row max leaves the result unchanged — softmax is invariant
    to adding a constant to every logit — but it bounds the largest exponent at
    exp(0) = 1, which prevents overflow when scores are large.
    """
    z_max = np.max(z, axis=axis, keepdims=True)
    # Where a whole row is -inf (a fully masked query), z_max is -inf and the
    # subtraction would give nan. Clamp the max to a finite value first.
    z_max = np.where(np.isfinite(z_max), z_max, 0.0)
    e = np.exp(z - z_max)
    return e / np.sum(e, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Compute Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V.

    Q:    (n, d_k)   queries      — what each position is looking for
    K:    (m, d_k)   keys         — what each position advertises
    V:    (m, d_v)   values       — what each position contributes
    mask: (n, m) additive mask, 0 to keep and a large negative to suppress

    Returns (output, attention_weights) with shapes (n, d_v) and (n, m).
    """
    n, d_k = Q.shape
    m, d_k_key = K.shape
    assert d_k == d_k_key, f"query dim {d_k} != key dim {d_k_key}"
    assert V.shape[0] == m, f"K has {m} rows but V has {V.shape[0]}"

    # Step 1: every query against every key. (n, d_k) @ (d_k, m) -> (n, m)
    scores = Q @ K.T
    assert scores.shape == (n, m)

    # Step 2: the scaling of eq. 63.5 — normalise the score standard deviation
    # so the softmax does not start out saturated.
    scores = scores / np.sqrt(d_k)

    # Step 3: suppress forbidden positions BEFORE the softmax, so the surviving
    # weights renormalise over what remains.
    if mask is not None:
        assert mask.shape == (n, m), f"mask {mask.shape} != scores {(n, m)}"
        scores = scores + mask

    # Step 4: each row becomes a distribution over the m key positions.
    attn = softmax(scores, axis=-1)

    # Step 5: convex combination of value rows. (n, m) @ (m, d_v) -> (n, d_v)
    output = attn @ V
    assert output.shape == (n, V.shape[1])
    return output, attn


def causal_mask(n, m=None, neg=-1e9):
    """Upper-triangular additive mask: position i may not see j > i."""
    m = n if m is None else m
    keep = np.tril(np.ones((n, m), dtype=bool))
    return np.where(keep, 0.0, neg)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, d_k, d_v = 5, 8, 16
    Q, K = rng.normal(size=(n, d_k)), rng.normal(size=(n, d_k))
    V = rng.normal(size=(n, d_v))

    out, A = scaled_dot_product_attention(Q, K, V)
    print("output shape         ", out.shape)
    print("attention shape      ", A.shape)

    # Property 1: every row of A is a probability distribution.
    assert np.allclose(A.sum(axis=-1), 1.0), "rows must sum to 1"
    assert (A >= 0).all(), "weights must be non-negative"

    # Property 2: the output is a convex combination of value rows, so its norm
    # cannot exceed the largest value-row norm.
    bound_holds = (np.linalg.norm(out, axis=-1).max()
                   <= np.linalg.norm(V, axis=-1).max() + 1e-9)
    assert bound_holds
    print("convex-combination bound holds:", bound_holds)

    # Property 3: causal masking makes A strictly lower-triangular.
    out_c, A_c = scaled_dot_product_attention(Q, K, V, mask=causal_mask(n))
    assert np.allclose(np.triu(A_c, k=1), 0.0), "causal mask leaked"
    assert np.allclose(A_c.sum(axis=-1), 1.0), "rows must still sum to 1"
    print("causal mask: strictly lower-triangular, rows still normalised")
    print("row 0 attends only to itself:", np.round(A_c[0], 4))

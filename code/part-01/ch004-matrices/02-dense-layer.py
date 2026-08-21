# Extracted from: Chapter 4 — Matrices, Matrix Multiplication, and Linear Maps
# Source: src/.../ch004-matrices.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A fully connected layer, forward pass only, with shapes asserted at every
step and the parameter and FLOP counts computed from the formulas of section 6.4.
"""
import numpy as np

rng = np.random.default_rng(0)

B, d_in, d_out = 8, 128, 64          # batch, input width, output width

X = rng.normal(size=(B, d_in))                       # a batch of examples
W = rng.normal(size=(d_in, d_out)) * np.sqrt(2.0 / d_in)   # He initialisation
b = np.zeros(d_out)


def relu(z):
    return np.maximum(z, 0.0)


def dense_forward(X, W, b, activation=relu):
    """eq. 4.15 : H = phi(XW + b)."""
    assert X.ndim == 2 and W.ndim == 2
    assert X.shape[1] == W.shape[0], (
        f"inner dims must match: X is {X.shape}, W is {W.shape}")
    Z = X @ W + b                     # (B, d_in) @ (d_in, d_out) -> (B, d_out)
    assert Z.shape == (X.shape[0], W.shape[1])
    return activation(Z)


H = dense_forward(X, W, b)
print(f"X {X.shape}  @  W {W.shape}  +  b {b.shape}   ->   H {H.shape}")

params = d_in * d_out + d_out
flops = 2 * B * d_in * d_out
print(f"\nparameters : {params:,}   (d_in*d_out + d_out)")
print(f"FLOPs      : {flops:,}   (2 * B * d_in * d_out)")

# Scale that up to a realistic transformer feed-forward block and the numbers
# stop being abstract.
d_model, d_ff, seq, batch = 4096, 11008, 2048, 1
ffn_params = 2 * d_model * d_ff
ffn_flops = 2 * 2 * batch * seq * d_model * d_ff
print(f"\none transformer FFN block at d_model={d_model}, d_ff={d_ff}:")
print(f"  parameters : {ffn_params/1e6:,.1f} M")
print(f"  FLOPs for {seq} tokens : {ffn_flops/1e9:,.1f} G")
print(f"  across 32 layers : {32*ffn_params/1e9:,.2f} B parameters")

# --- the column view, made visible ------------------------------------------
# Output feature j is a weighted sum of inputs; equivalently the output vector
# is a linear combination of W's columns with the input as coefficients.
x0 = X[0]
combo = sum(x0[i] * W[i] for i in range(d_in))
assert np.allclose(combo, x0 @ W)
print("\ncolumn view confirmed: x @ W == sum_i x_i * W[i, :]")

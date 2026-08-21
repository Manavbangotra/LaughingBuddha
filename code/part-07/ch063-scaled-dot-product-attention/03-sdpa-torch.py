# Extracted from: Chapter 63 — Scaled Dot-Product Attention
# Source: src/.../ch063-scaled-dot-product-attention.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Batched, multi-head-shaped attention in PyTorch, checked against the
built-in fused kernel.
"""
import math

import torch
import torch.nn.functional as F


def sdpa(q, k, v, mask=None):
    """Scaled dot-product attention over (..., seq, dim) tensors.

    Leading dimensions are arbitrary and broadcast — typically (batch, heads).
    q: (..., n, d_k)   k: (..., m, d_k)   v: (..., m, d_v)
    mask: broadcastable to (..., n, m), additive
    """
    d_k = q.size(-1)
    # transpose(-2, -1) swaps only the last two axes, leaving batch/head intact.
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores + mask
    attn = torch.softmax(scores, dim=-1)
    return attn @ v, attn


torch.manual_seed(0)
B, H, N, Dk, Dv = 2, 4, 7, 16, 16
q = torch.randn(B, H, N, Dk, dtype=torch.float64)
k = torch.randn(B, H, N, Dk, dtype=torch.float64)
v = torch.randn(B, H, N, Dv, dtype=torch.float64)

out, attn = sdpa(q, k, v)
print("output shape:", tuple(out.shape), " attention shape:", tuple(attn.shape))

# Agreement with PyTorch's fused implementation. It computes the same function;
# it differs only in how it schedules memory.
ref = F.scaled_dot_product_attention(q, k, v)
print("max abs difference vs F.scaled_dot_product_attention:",
      (out - ref).abs().max().item())
assert torch.allclose(out, ref, atol=1e-10)

# Causal masking, both ways, must also agree.
causal = torch.triu(torch.full((N, N), float("-inf"), dtype=torch.float64),
                    diagonal=1)
out_c, attn_c = sdpa(q, k, v, mask=causal)
ref_c = F.scaled_dot_product_attention(q, k, v, is_causal=True)
assert torch.allclose(out_c, ref_c, atol=1e-10)
assert torch.allclose(attn_c.triu(diagonal=1),
                      torch.zeros_like(attn_c.triu(diagonal=1)))
print("causal path agrees with is_causal=True; upper triangle is exactly zero")

# Permutation equivariance (eq. 63.10): permute the sequence, and the outputs
# are the same vectors in the permuted order.
perm = torch.randperm(N)
out_p, _ = sdpa(q[:, :, perm], k[:, :, perm], v[:, :, perm])
print("permutation equivariance holds:",
      torch.allclose(out_p, out[:, :, perm], atol=1e-10))

# -*- coding: utf-8 -*-
# Extracted from: Chapter 63 — Scaled Dot-Product Attention
# Source: src/.../ch063-scaled-dot-product-attention.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Hand-derived gradients for attention, checked against autograd.

Implements eqs. 63.13-63.15 and compares to torch.autograd on the same inputs.
Run in float64 so that a mismatch means an error in the derivation rather than
accumulated rounding.
"""
import math

import torch

torch.manual_seed(0)
N, M, Dk, Dv = 6, 6, 8, 5
Q = torch.randn(N, Dk, dtype=torch.float64, requires_grad=True)
K = torch.randn(M, Dk, dtype=torch.float64, requires_grad=True)
V = torch.randn(M, Dv, dtype=torch.float64, requires_grad=True)

# --- forward, keeping every intermediate the backward pass needs -------------
S = Q @ K.T / math.sqrt(Dk)
A = torch.softmax(S, dim=-1)
O = A @ V

# An arbitrary scalar loss, so that dL/dO is a fixed known matrix.
G = torch.randn(N, Dv, dtype=torch.float64)
loss = (O * G).sum()
loss.backward()

# --- the same gradients, derived by hand -------------------------------------
with torch.no_grad():
    # eq. 63.13: through O = A V
    dV = A.T @ G
    dA = G @ V.T

    # eq. 63.14: through the row-wise softmax.
    # The row-sum term is what makes each row's gradient sum to zero, which is
    # the differential form of "the row must keep summing to one".
    rowsum = (dA * A).sum(dim=-1, keepdim=True)
    dS = A * (dA - rowsum)

    # eq. 63.15: through the scaled product
    dQ = dS @ K / math.sqrt(Dk)
    dK = dS.T @ Q / math.sqrt(Dk)

for name, mine, auto in (("dQ", dQ, Q.grad), ("dK", dK, K.grad),
                         ("dV", dV, V.grad)):
    err = (mine - auto).abs().max().item()
    print(f"{name}: max abs error vs autograd = {err:.3e}")
    assert err < 1e-12, f"{name} derivation disagrees with autograd"

print("\nAll three hand-derived gradients match autograd to float64 precision.")
print("Note dS rows sum to ~0:", dS.sum(dim=-1).abs().max().item() < 1e-12,
      "— the softmax constraint, differentiated.")

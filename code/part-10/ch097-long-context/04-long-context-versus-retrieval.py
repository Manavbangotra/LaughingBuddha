# -*- coding: utf-8 -*-
# Extracted from: Chapter 97 — Long-Context Behavior and Its Limits
# Source: src/.../ch097-long-context.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Stuff the whole document, or retrieve? Cost and quality, both computed."""
import numpy as np

DOC_TOKENS = 200_000
N, L, D, KV_HEADS, HEAD_DIM = 7e9, 32, 4096, 8, 128
BYTES = 2
DEVICE_FLOPS, MFU, GPU_HOUR = 1e15, 0.45, 2.50
REQUESTS_PER_DAY = 30_000


def prefill_cost(tokens):
    """Equation (eq:long-context-cost), both terms."""
    flops = 2 * N * tokens + 4 * L * tokens ** 2 * D
    hours = flops / (DEVICE_FLOPS * MFU) / 3600
    return hours * GPU_HOUR, flops


def cache_gb(tokens):
    return 2 * L * KV_HEADS * HEAD_DIM * tokens * BYTES / 1e9


def worst_position_accuracy(tokens):
    depth = 0.45 * (1 - np.exp(-tokens / 40_000))
    return 0.97 * (1 - depth)


OPTIONS = {
    "full document in context": DOC_TOKENS,
    "retrieve 20 passages":     20 * 500,
    "retrieve 8 passages":      8 * 500,
    "retrieve 4 passages":      4 * 500,
}

print(f"document {DOC_TOKENS:,} tokens, {REQUESTS_PER_DAY:,} requests/day\n")
print(f"{'option':<28} {'tokens':>9} {'$/request':>11} {'$/day':>10} "
      f"{'cache GB':>9} {'worst-pos acc':>14}")
for name, toks in OPTIONS.items():
    cost, _ = prefill_cost(toks)
    print(f"{name:<28} {toks:>9,} {cost:>11.5f} "
          f"{cost * REQUESTS_PER_DAY:>10,.0f} {cache_gb(toks):>9.2f} "
          f"{worst_position_accuracy(toks):>14.3f}")

full_cost, full_flops = prefill_cost(DOC_TOKENS)
ret_cost, ret_flops = prefill_cost(8 * 500)
print(f"\ncost ratio, full document vs 8 passages: {full_cost / ret_cost:,.0f}x")

# Where the full-document cost goes — the quadratic term.
lin = 2 * N * DOC_TOKENS
quad = 4 * L * DOC_TOKENS ** 2 * D
print(f"\nfull-document prefill FLOPs:")
print(f"  parameter term (2NT)     : {lin:>10.2e} ({lin / (lin + quad):.0%})")
print(f"  attention term (4LT^2 d) : {quad:>10.2e} ({quad / (lin + quad):.0%})")
print(f"  crossover is at T = 6d = {6 * D:,} tokens (ch:tf-complexity)")

print("""
At 200,000 tokens the quadratic attention term dominates completely, so the
full-document option is not merely 25x more expensive than retrieving eight
passages — it is far worse than linear scaling would suggest.

And it is LESS accurate at the worst position, because equation (eq:u-shape)'s
depression deepens with length. The long-context option is more expensive and
less reliable at once, which is unusual: most engineering choices trade one for
the other.

The case FOR long context is real and narrower than it looks: it needs no index,
no chunking strategy, and no retriever to maintain, and it cannot suffer a
retrieval miss. For a low-volume application over a document that fits
comfortably, those are decisive. At 30,000 requests a day they are not.""")

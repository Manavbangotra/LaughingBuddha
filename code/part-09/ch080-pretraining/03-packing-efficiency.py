# -*- coding: utf-8 -*-
# Extracted from: Chapter 80 — Pretraining and Self-Supervised Objectives
# Source: src/.../ch080-pretraining.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Padding versus packing on a realistic document-length distribution."""
import numpy as np

rng = np.random.default_rng(0)
CONTEXT = 2048
N_DOCS = 20_000

# Document lengths in a web corpus are heavy-tailed: mostly short, a long tail.
lengths = np.clip(rng.lognormal(mean=5.8, sigma=1.4, size=N_DOCS).astype(int), 8, 60_000)

useful = int(lengths.sum())
print(f"{N_DOCS:,} documents, {useful:,} useful tokens")
print(f"length percentiles: p50={np.percentile(lengths, 50):,.0f}  "
      f"p90={np.percentile(lengths, 90):,.0f}  "
      f"p99={np.percentile(lengths, 99):,.0f}  max={lengths.max():,}\n")

# --- one document per row, padded to the context length ---------------------
truncated = np.minimum(lengths, CONTEXT)
padded_rows = N_DOCS
padded_processed = padded_rows * CONTEXT
padded_useful = int(truncated.sum())

# --- packed: concatenate, then slice fixed windows --------------------------
packed_rows = int(np.ceil(useful / CONTEXT))
packed_processed = packed_rows * CONTEXT

print(f"{'strategy':<12} {'rows':>9} {'tokens processed':>18} "
      f"{'useful':>12} {'efficiency':>12}")
for name, rows, processed, use in [
        ("padding", padded_rows, padded_processed, padded_useful),
        ("packing", packed_rows, packed_processed, useful)]:
    print(f"{name:<12} {rows:>9,} {processed:>18,} {use:>12,} "
          f"{use / processed:>11.1%}")

print(f"\nrows saved: {(1 - packed_rows / padded_rows):.1%}")
print(f"tokens lost to truncation under padding: "
      f"{useful - padded_useful:,} ({(useful - padded_useful) / useful:.1%})")

# Attention is quadratic in the row length (ch:tf-complexity), and padded rows
# are full-length regardless of content, so the waste compounds.
attn_padded = padded_rows * CONTEXT ** 2
attn_packed = packed_rows * CONTEXT ** 2
print(f"\nattention work, padding : {attn_padded:.3e}")
print(f"attention work, packing : {attn_packed:.3e}")
print(f"ratio                   : {attn_padded / attn_packed:.1f}x")
print("\nPacking is not a micro-optimisation. On this distribution it is a "
      "multiple of the entire training cost, and it also recovers the tokens "
      "that truncation would have discarded.")

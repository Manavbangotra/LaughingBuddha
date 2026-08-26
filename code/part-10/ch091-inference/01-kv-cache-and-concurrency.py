# -*- coding: utf-8 -*-
# Extracted from: Chapter 91 — Context Windows, KV Cache, and Inference Mechanics
# Source: src/.../ch091-inference.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""KV cache memory, concurrency, and what GQA buys. Equation (eq:kv-cache-serving)."""

BYTES = 2                       # bf16


def cache_bytes_per_token(layers, kv_heads, head_dim, bytes_per=BYTES):
    """Equation (eq:kv-cache-serving), per token per sequence."""
    return 2 * layers * kv_heads * head_dim * bytes_per


MODELS = {
    "7B, MHA (g=32)":  dict(params=7e9,  layers=32, kv_heads=32, head_dim=128),
    "7B, GQA (g=8)":   dict(params=7e9,  layers=32, kv_heads=8,  head_dim=128),
    "7B, MQA (g=1)":   dict(params=7e9,  layers=32, kv_heads=1,  head_dim=128),
    "70B, GQA (g=8)":  dict(params=70e9, layers=80, kv_heads=8,  head_dim=128),
}
DEVICE_GB = 80

print(f"device {DEVICE_GB} GB, bf16\n")
print(f"{'model':<18} {'weights':>9} {'KV/token':>10} {'KV @4k':>10} "
      f"{'concurrency @4k':>17} {'@32k':>7}")
for name, m in MODELS.items():
    w = m["params"] * BYTES / 1e9
    per_tok = cache_bytes_per_token(m["layers"], m["kv_heads"], m["head_dim"])
    at_4k = per_tok * 4096 / 1e9
    at_32k = per_tok * 32768 / 1e9
    free = DEVICE_GB - w
    c4 = int(free / at_4k) if free > 0 else 0
    c32 = int(free / at_32k) if free > 0 else 0
    print(f"{name:<18} {w:>8.1f}G {per_tok / 1024:>9.0f}K {at_4k:>9.2f}G "
          f"{c4:>17} {c32:>7}")

mha = cache_bytes_per_token(32, 32, 128)
gqa = cache_bytes_per_token(32, 8, 128)
print(f"\nGQA (g=8) against MHA (g=32): cache per token "
      f"{mha / 1024:.0f}K -> {gqa / 1024:.0f}K, a {mha / gqa:.0f}x reduction")
print("Parameter count is essentially unchanged — the K and V projections "
      "shrink, and they are a small share of the block (ch:tf-ffn-residual).")

# Equation (eq:max-concurrency): concurrency falls linearly with context.
m = MODELS["7B, GQA (g=8)"]
per_tok = cache_bytes_per_token(m["layers"], m["kv_heads"], m["head_dim"])
free_gb = DEVICE_GB - m["params"] * BYTES / 1e9
print(f"\n{'context':>9} {'cache/request':>15} {'max concurrency':>17}")
for ctx in (1024, 4096, 16384, 65536, 131072):
    per_req = per_tok * ctx / 1e9
    print(f"{ctx:>9,} {per_req:>14.2f}G {int(free_gb / per_req):>17}")

print("""
Concurrency falls linearly with context length, and the failure mode is not
gradual: a server sized for 4k conversations holds 128 of them and 8 at 64k. It
does not slow down as it approaches the limit — it rejects requests.

Note also that at 128k context a single request needs more than 16 GB of cache
for a 7B model whose weights are 14 GB. THE CACHE IS LARGER THAN THE MODEL, for
one user. Every long-context serving decision follows from that inversion.""")

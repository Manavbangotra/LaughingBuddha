# Extracted from: Chapter 88 — Anatomy of an LLM: From Tokens to Logits
# Source: src/.../ch088-anatomy.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""What a model's configuration tells you that its parameter count does not."""

MODELS = {
    "A (3B, wide)":   dict(L=26, d=3072, d_ff=8192,  V=32000, h=24, kv_h=24),
    "B (3B, deep)":   dict(L=40, d=2560, d_ff=6912,  V=32000, h=20, kv_h=20),
    "C (7B, GQA)":    dict(L=32, d=4096, d_ff=11008, V=32000, h=32, kv_h=8),
}

BYTES = 2                      # bf16
CONTEXT = 8192
BATCH = 16


def analyse(L, d, d_ff, V, h, kv_h):
    d_head = d // h
    params = L * (4 * d * d + 3 * d * d_ff) + 2 * V * d
    weights_gb = params * BYTES / 1e9
    # KV cache: 2 (K and V) x layers x kv_heads x head_dim x tokens x bytes
    kv_per_token = 2 * L * kv_h * d_head * BYTES
    kv_gb = kv_per_token * CONTEXT * BATCH / 1e9
    return dict(params=params, weights_gb=weights_gb,
                kv_per_token=kv_per_token, kv_gb=kv_gb,
                depth=L, flops_per_token=2 * params)


print(f"context {CONTEXT:,}, batch {BATCH}, bf16\n")
print(f"{'model':<15} {'params':>9} {'weights':>9} {'KV/token':>10} "
      f"{'KV total':>10} {'total GB':>10} {'depth':>7}")
rows = {}
for name, cfg in MODELS.items():
    a = analyse(**cfg)
    rows[name] = a
    total = a["weights_gb"] + a["kv_gb"]
    print(f"{name:<15} {a['params'] / 1e9:>8.2f}B {a['weights_gb']:>8.1f}G "
          f"{a['kv_per_token']:>9,}B {a['kv_gb']:>9.1f}G {total:>9.1f}G "
          f"{a['depth']:>7}")

a, b, c = rows["A (3B, wide)"], rows["B (3B, deep)"], rows["C (7B, GQA)"]
print(f"\nA and B are both '3B' and differ by "
      f"{abs(a['params'] - b['params']) / a['params']:.1%} in parameters.")
print(f"  KV cache per token : {a['kv_per_token']:,} vs {b['kv_per_token']:,} "
      f"({b['kv_per_token'] / a['kv_per_token']:.2f}x)")
print(f"  depth              : {a['depth']} vs {b['depth']} layers "
      f"-> B has {b['depth'] / a['depth']:.2f}x the sequential steps per token")

print(f"\nC is {c['params'] / a['params']:.1f}x A's parameters, but its KV "
      f"cache per token is {c['kv_per_token'] / a['kv_per_token']:.2f}x —")
print(f"grouped-query attention ({MODELS['C (7B, GQA)']['h']} query heads, "
      f"{MODELS['C (7B, GQA)']['kv_h']} KV heads) decouples the two.")

print(f"\ntotal memory at this batch and context:")
for name in rows:
    r = rows[name]
    print(f"  {name:<15} {r['weights_gb'] + r['kv_gb']:>6.1f} GB "
          f"({r['kv_gb'] / (r['weights_gb'] + r['kv_gb']):.0%} of it cache)")

print("""
The parameter count answers one question — how much arithmetic per token — and
is silent on the two that decide serving.

Depth sets the sequential critical path: decoding is one token at a time, so a
40-layer model has 40 sequential dependencies per token against a 26-layer
model's 26, at identical parameter count. That shows up directly in inter-token
latency and cannot be batched away.

And the KV cache is a function of layers, KV heads and head dimension, with no
term in the parameter count at all. C has more than twice A's parameters and a
SMALLER cache per token, because grouped-query attention decoupled them. At
batch 16 and 8k context the cache is a large share of total memory, which is
what actually limits concurrency.

'3B' and '7B' are marketing. The configuration is the specification.""")

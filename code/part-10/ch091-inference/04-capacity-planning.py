# -*- coding: utf-8 -*-
# Extracted from: Chapter 91 — Context Windows, KV Cache, and Inference Mechanics
# Source: src/.../ch091-inference.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Sizing a deployment: what binds, and at what point."""

MODEL = dict(params=7e9, layers=32, kv_heads=8, head_dim=128)
BYTES = 2
DEVICES, DEVICE_GB = 8, 80
DEVICE_FLOPS, BANDWIDTH, MFU = 1e15, 3e12, 0.45

PROMPT, OUTPUT, TARGET_USERS = 12_000, 400, 200

weights_gb = MODEL["params"] * BYTES / 1e9
per_token = 2 * MODEL["layers"] * MODEL["kv_heads"] * MODEL["head_dim"] * BYTES
per_request_gb = per_token * (PROMPT + OUTPUT) / 1e9

print(f"weights            : {weights_gb:.1f} GB")
print(f"cache per request  : {per_request_gb:.2f} GB "
      f"({PROMPT + OUTPUT:,} tokens x {per_token / 1024:.0f} KB)")
print(f"cache for {TARGET_USERS} users: "
      f"{per_request_gb * TARGET_USERS:.0f} GB\n")

total_gb = DEVICES * DEVICE_GB
usable = total_gb - weights_gb * DEVICES     # weights replicated per device
max_users_mem = int(usable / per_request_gb)
print(f"{'total device memory':<28} {total_gb:>8.0f} GB")
print(f"{'weights (replicated x' + str(DEVICES) + ')':<28} "
      f"{weights_gb * DEVICES:>8.0f} GB")
print(f"{'available for cache':<28} {usable:>8.0f} GB")
print(f"{'-> max concurrent users':<28} {max_users_mem:>8}")
print(f"{'target':<28} {TARGET_USERS:>8}")
print(f"{'verdict':<28} "
      f"{('FITS' if max_users_mem >= TARGET_USERS else 'DOES NOT FIT'):>8}\n")

# What binds: memory or compute?
prefill_flops = 2 * MODEL["params"] * PROMPT
decode_flops = 2 * MODEL["params"] * OUTPUT
per_request_flops = prefill_flops + decode_flops
cluster_flops = DEVICES * DEVICE_FLOPS * MFU

print(f"{'per-request FLOPs':<28} {per_request_flops:>10.2e}")
print(f"{'cluster FLOPs/s':<28} {cluster_flops:>10.2e}")
print(f"{'-> requests/second (compute)':<28} "
      f"{cluster_flops / per_request_flops:>10.1f}")

# And the split: how much of the work is prefill?
print(f"\n{'phase':<12} {'FLOPs':>12} {'share':>8}")
for name, f in [("prefill", prefill_flops), ("decode", decode_flops)]:
    print(f"{name:<12} {f:>12.2e} {f / per_request_flops:>7.0%}")

print(f"""
This workload is {prefill_flops / per_request_flops:.0%} PREFILL, which inverts
the usual advice. With a 12,000-token prompt and a 400-token answer, most of the
compute is reading the document, not writing the answer — so batching (which
helps decode) buys much less than it would for a chat workload, and prefill
throughput is what to optimise.

And the binding constraint is memory, not compute: the cluster could serve
{cluster_flops / per_request_flops:.0f} requests/second on arithmetic alone,
while cache memory caps concurrency at {max_users_mem}.""")

# The intervention that actually helps here.
print(f"\n{'intervention':<32} {'cache/request':>15} {'max users':>11}")
options = [
    ("as-is", per_request_gb, ""),
    ("KV cache in fp8", per_request_gb / 2, "halves cache, small quality cost"),
    ("halve the prompt (rerank first)", per_request_gb *
     (PROMPT / 2 + OUTPUT) / (PROMPT + OUTPUT), "ch:emb-reranking"),
    ("both", per_request_gb / 2 *
     (PROMPT / 2 + OUTPUT) / (PROMPT + OUTPUT), ""),
]
for label, cache, note in options:
    print(f"{label:<32} {cache:>14.2f}G {int(usable / cache):>11}")

print("""
Shortening the prompt helps twice over: less cache per request AND less prefill
compute, which is the phase this workload is dominated by. Retrieving fewer,
better passages is therefore not only a quality decision — it is the single
largest lever on both cost and capacity here, which is a good reason to read
Part XI before buying more hardware.""")

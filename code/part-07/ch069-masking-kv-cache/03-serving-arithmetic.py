# -*- coding: utf-8 -*-
# Extracted from: Chapter 69 — Causal Masking and the KV Cache
# Source: src/.../ch069-masking-kv-cache.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The serving decisions this chapter enables: how many users fit, what
prompt caching buys, and what RoPE forbids.
"""
import numpy as np

rng = np.random.default_rng(2)


def kv_bytes(L, g, dk, T, b=2):
    return 2 * b * L * g * dk * T


def weight_bytes(P, b=2):
    return b * P


print("=" * 72)
print("how many users fit on a machine?")
print("=" * 72)
CONFIGS = [
    ("7B  GQA g=8",  7e9,  32,  8, 128),
    ("70B MHA",      7e10, 80, 64, 128),
    ("70B GQA g=8",  7e10, 80,  8, 128),
]
for HBM in (80, 640):
    print(f"\naccelerator memory: {HBM} GB")
    print(f"{'model':<16} {'weights':>9} {'free':>9} " +
          " ".join(f"{f'users @ {T // 1024}k':>16}"
                   for T in (4096, 32768, 131072)))
    for name, P, L, g, dk in CONFIGS:
        w = weight_bytes(P) / 1e9
        free = HBM - w
        if free <= 0:
            print(f"{name:<16} {w:>8.0f}G {'does not fit':>9}")
            continue
        row = [int(free * 1e9 / kv_bytes(L, g, dk, T))
               for T in (4096, 32768, 131072)]
        print(f"{name:<16} {w:>8.0f}G {free:>8.0f}G " +
              " ".join(f"{x:>16,}" for x in row))

print("\nThe 70B rows on one 80 GB device are the whole argument for")
print("multi-device serving: the weights alone do not fit. On a 640 GB")
print("node, the difference between MHA and GQA at a 32k context is the")
print("difference between a handful of users and a useful number.")
print("\nAnd note the trend along each row: doubling the context halves the")
print("users, exactly. Concurrency and context length trade linearly, which")
print("is the single most useful fact for capacity planning.")

# --- prompt caching ---------------------------------------------------------
print("\n" + "=" * 72)
print("what prompt caching buys, and what it requires (section 7.3)")
print("=" * 72)
print("A shared system prompt of S tokens, followed by a per-user query of")
print("Q tokens. Prefill is quadratic in the total, so caching the shared")
print("part saves more than its share.\n")


def prefill_flops(P, L, d, T):
    """Rough: linear term from the weights + quadratic attention term."""
    return 2 * P * T + 4 * L * d * T * T


P, L, d = 7e9, 32, 4096
print(f"{'system S':>10} {'query Q':>9} {'full prefill':>14} "
      f"{'cached prefill':>16} {'saving':>9}")
for S, Q in ((1000, 50), (4000, 50), (16000, 50), (16000, 2000)):
    full = prefill_flops(P, L, d, S + Q)
    # cached: only the Q new tokens are processed, but they attend over S+Q
    cached = 2 * P * Q + 4 * L * d * Q * (S + Q)
    print(f"{S:>10,} {Q:>9,} {full / 1e12:>13.2f}T "
          f"{cached / 1e12:>15.2f}T {1 - cached / full:>8.1%}")

print("\nThe saving grows with the shared fraction, and it is superlinear")
print("because the quadratic attention term over the shared prefix")
print("disappears entirely.")
print("\nThe constraint is section 6.4's. The reused block must be a PREFIX")
print("at the same absolute positions, because a RoPE key is stored after")
print("rotation and carries its position permanently. A shared fragment in")
print("the middle of differing prefixes cannot be reused at all.")
print("\nThat is why prompt design puts the stable part first, and why API")
print("pricing distinguishes cached from uncached input tokens.")

# --- demonstrate the RoPE constraint ----------------------------------------
print("\n" + "=" * 72)
print("why a cached key cannot be moved (section 6.4)")
print("=" * 72)
dk = 32


def rope_tables(T, dk, base=10000.0):
    theta = base ** (-np.arange(0, dk, 2) / dk)
    m = np.arange(T)[:, None]
    ang = m * theta[None, :]
    return np.cos(ang), np.sin(ang)


def apply_rope(x, cos, sin):
    d = x.shape[-1]
    x1, x2 = x[..., :d // 2], x[..., d // 2:]
    return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


cos, sin = rope_tables(256, dk)
k_raw = rng.normal(size=dk)
q_raw = rng.normal(size=dk)

print("A key for the same token, cached at two different positions, scored")
print("against a query at position 100.\n")
print(f"{'key cached at':>14} {'query at':>10} {'offset':>8} "
      f"{'score':>10}")
for kpos in (10, 20, 50):
    kk = apply_rope(k_raw[None, :], cos[kpos:kpos + 1], sin[kpos:kpos + 1])[0]
    qq = apply_rope(q_raw[None, :], cos[100:101], sin[100:101])[0]
    print(f"{kpos:>14} {100:>10} {100 - kpos:>8} {float(qq @ kk):>10.4f}")

print("\nThree different scores for the SAME token, because the rotation")
print("baked its position into the cached key. That is correct behaviour —")
print("eq. 65.9 wants the score to depend on the offset — and it is exactly")
print("why the cached block is not portable.")
print("\nIf you reuse a cache entry at the wrong offset you do not get an")
print("error; you get a score computed for a distance that is not the real")
print("one, and generation that is fluent and wrong.")

# --- the concurrency/context frontier ---------------------------------------
print("\n" + "=" * 72)
print("the frontier a serving system actually operates on")
print("=" * 72)
print("For a 70B GQA model on a 640 GB node, the set of (users, context)")
print("pairs that fit:\n")
P, L, g, dk = 7e10, 80, 8, 128
free = (640 - weight_bytes(P) / 1e9) * 1e9
print(f"{'users':>7} " + " ".join(f"{f'{T // 1024}k':>9}"
                                  for T in (4096, 16384, 65536, 262144)))
for users in (1, 8, 32, 128, 512):
    row = []
    for T in (4096, 16384, 65536, 262144):
        need = users * kv_bytes(L, g, dk, T)
        row.append("yes" if need <= free else "no")
    print(f"{users:>7} " + " ".join(f"{v:>9}" for v in row))

print("\nEvery 'yes' is a deployment configuration and every 'no' is a")
print("capacity failure that a load test at low concurrency will not find.")
print("\nThis table is what capacity planning for an LLM service actually")
print("is, and it is eq. 69.4 with the numbers substituted. It is also why")
print("the techniques of Chapter 71 — cache compression, quantisation,")
print("sharing — are commercially important rather than academically")
print("interesting: each one moves this boundary.")

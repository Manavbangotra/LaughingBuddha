# -*- coding: utf-8 -*-
# Extracted from: Chapter 91 — Context Windows, KV Cache, and Inference Mechanics
# Source: src/.../ch091-inference.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Time-to-first-token and inter-token latency respond to different things."""

N, BYTES, DEVICE_FLOPS, BANDWIDTH, MFU = 7e9, 2, 1e15, 3e12, 0.45


def ttft_ms(prompt_tokens, batch, cached_prefix=0):
    """Equation (eq:ttft). Cached prefix tokens skip prefill compute."""
    new = max(prompt_tokens - cached_prefix, 1)
    flops = 2 * N * new * batch
    t = max(flops / (DEVICE_FLOPS * MFU), BYTES * N / BANDWIDTH)
    return t * 1000


def itl_ms(batch):
    """Equation (eq:itl) — nearly constant in prompt length."""
    t = max(2 * N * batch / (DEVICE_FLOPS * MFU), BYTES * N / BANDWIDTH)
    return t * 1000


print(f"{'prompt':>9} {'TTFT (b=1)':>12} {'ITL (b=1)':>11} "
      f"{'TTFT (b=64)':>13} {'ITL (b=64)':>12}")
for p in (100, 1000, 4000, 16000, 64000):
    print(f"{p:>9,} {ttft_ms(p, 1):>11.1f}m {itl_ms(1):>10.1f}m "
          f"{ttft_ms(p, 64):>12.1f}m {itl_ms(64):>11.1f}m")

print("""
TTFT scales with prompt length and ITL does not. That is why a long prompt
delays the START of streaming rather than slowing it down, and why users
describe long-context requests as "slow to begin" rather than "slow".""")

# The tradeoff: batching improves throughput and ITL, and hurts TTFT via queueing.
print(f"\n{'batch':>7} {'ITL ms':>9} {'tokens/s':>11} {'queue wait ms':>15} "
      f"{'effective TTFT':>16}")
ARRIVAL_RATE = 40                # requests/second
for B in (1, 8, 32, 128, 256):
    itl = itl_ms(B)
    tput = B / (itl / 1000)
    # A larger batch means waiting for it to fill.
    queue = (B / ARRIVAL_RATE) * 1000 / 2
    print(f"{B:>7} {itl:>9.2f} {tput:>11,.0f} {queue:>15.1f} "
          f"{ttft_ms(1000, B) + queue:>15.1f}m")

print("""
The two metrics move in opposite directions. Larger batches raise throughput and
leave ITL unchanged until the crossover, and they lengthen the wait to assemble
a batch — so effective TTFT rises. A system tuned for throughput feels
unresponsive to start and fast once started.

This is why 'latency' is not one number. Decide which one your product is
sensitive to: a chat interface lives on TTFT, a batch summarisation job lives on
throughput, and they want opposite configurations.""")

# Prefix caching, the largest TTFT win for a fixed system prompt.
SYSTEM_PROMPT = 800
print(f"\nprefix caching with an {SYSTEM_PROMPT}-token system prompt:")
print(f"{'user text':>11} {'TTFT uncached':>15} {'TTFT cached':>13} "
      f"{'saving':>9}")
for user in (50, 200, 1000):
    total = SYSTEM_PROMPT + user
    un, ca = ttft_ms(total, 1), ttft_ms(total, 1, cached_prefix=SYSTEM_PROMPT)
    print(f"{user:>11,} {un:>14.1f}m {ca:>12.1f}m "
          f"{(1 - ca / un):>8.0%}")
print("The saving is largest exactly where the system prompt dominates the "
      "request — which is the common case for an assistant with detailed "
      "instructions.")

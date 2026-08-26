# -*- coding: utf-8 -*-
# Extracted from: Chapter 91 — Context Windows, KV Cache, and Inference Mechanics
# Source: src/.../ch091-inference.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Why output tokens cost more than input tokens. Equation (eq:price-ratio)."""

N = 7e9
BYTES = 2
DEVICE_FLOPS = 1e15
BANDWIDTH = 3e12                # bytes/second
MFU = 0.45

PROMPT, OUTPUT = 1000, 200


def prefill_time(T, batch=1):
    flops = 2 * N * T * batch
    bytes_read = BYTES * N                       # weights, once
    return max(flops / (DEVICE_FLOPS * MFU), bytes_read / BANDWIDTH)


def decode_step_time(batch):
    """Equation (eq:decode-step-time)."""
    flops = 2 * N * batch
    bytes_read = BYTES * N                       # weights, once per STEP
    return max(flops / (DEVICE_FLOPS * MFU), bytes_read / BANDWIDTH)


print(f"{N / 1e9:.0f}B model, prompt {PROMPT}, output {OUTPUT} tokens\n")

# Arithmetic intensity, equation (eq:arithmetic-intensity-phases).
ridge = DEVICE_FLOPS / BANDWIDTH
print(f"device ridge point: {ridge:.0f} FLOPs/byte")
print(f"{'phase':<22} {'FLOPs/byte':>12} {'bound by':>12}")
print(f"{'prefill (T=' + str(PROMPT) + ')':<22} {2 * PROMPT / BYTES:>12.0f} "
      f"{'compute':>12}")
for B in (1, 32, 256):
    ai = 2 * B / BYTES
    print(f"{'decode (batch ' + str(B) + ')':<22} {ai:>12.0f} "
          f"{('memory' if ai < ridge else 'compute'):>12}")

pf = prefill_time(PROMPT)
ds = decode_step_time(1)
print(f"\nprefill {PROMPT} tokens : {pf * 1000:>8.1f} ms  "
      f"({PROMPT / pf:>10,.0f} tokens/s)")
print(f"decode  1 token       : {ds * 1000:>8.1f} ms  "
      f"({1 / ds:>10,.0f} tokens/s)")
print(f"per-token ratio       : {(ds) / (pf / PROMPT):>8.0f}x more expensive "
      f"to generate than to read")

# Equation (eq:price-ratio) against real pricing.
print(f"\n{'decode batch':>13} {'predicted price ratio (T/B)':>30}")
for B in (50, 100, 200, 400):
    print(f"{B:>13} {PROMPT / B:>30.1f}")
print("Providers charge 3-5x for output tokens. The arithmetic gives the same "
      "range at realistic batch sizes — the pricing is eq:price-ratio.")

# Batching: free throughput until the crossover of eq:batch-crossover.
crossover = BYTES * DEVICE_FLOPS * MFU / (2 * BANDWIDTH)
print(f"\nbatch crossover (eq:batch-crossover): B* = {crossover:.0f}")
print(f"{'batch':>7} {'step ms':>9} {'tokens/s':>11} {'per-token ms':>14}")
for B in (1, 8, 32, 128, 256, 512, 1024):
    st = decode_step_time(B)
    print(f"{B:>7} {st * 1000:>9.2f} {B / st:>11,.0f} {st * 1000 / B:>14.4f}")

print("""
Step time is FLAT up to the crossover — the same weights serve every sequence in
the batch, so batch 128 costs the same wall-clock per step as batch 1 and
produces 128 times the tokens. That is the largest free win in LLM serving.

Past the crossover decode becomes compute-bound and step time rises linearly, so
further batching buys throughput only at proportional latency cost. Most systems
never get there, because equation (eq:max-concurrency) runs out of cache memory
first.""")

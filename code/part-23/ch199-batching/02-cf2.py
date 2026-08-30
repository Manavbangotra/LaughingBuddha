# -*- coding: utf-8 -*-
# Extracted from: Chapter 199 — Batching and Continuous Batching
# Source: src/.../ch199-batching.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Prefill and decode interfere, and there are two opposite fixes.

Continuous batching solves the unequal-length problem. It does not solve a second one:
a prefill and a decode want completely different things from the same step.

Prefill is compute-bound and long; decode is memory-bound and short. Put a whole
prefill in a step and every decode sharing that step waits for it, so one large prompt
stalls every sequence in flight (eq:prefill-stalls-decode).

There are two known fixes and they are opposites. cite:agrawal2023sarathi CHUNKS the
prefill so each step carries a small piece alongside the decodes -- exploiting the fact
that a decode step has idle compute, which ch:inf-cpu-gpu measured. cite:zhong2024distserve
and cite:patel2023splitwise SEPARATE the phases onto different machines entirely.

This listing measures both against the same workload and finds neither dominates.
"""
import math

# From ch:inf-cpu-gpu: a step is max(weight traffic / bandwidth, FLOPs / peak).
WEIGHT_BYTES = 14.0e9
BANDWIDTH = 3.35e12
PEAK = 9.89e14
PARAMS = 7.0e9
BALANCE = PEAK / BANDWIDTH        # tokens per step to become compute-bound

PROMPTS = [200, 900, 3200, 12000]
BATCH = 32
PREFILL_RATE = 8.0                # prefills arriving per second
DECODE_MACHINES = 8.0


def step_ms(tokens):
    """Milliseconds for one step carrying `tokens` tokens of work."""
    t_mem = WEIGHT_BYTES / BANDWIDTH
    t_flop = 2.0 * PARAMS * tokens / PEAK
    return max(t_mem, t_flop) * 1000.0


DECODE_MS = step_ms(BATCH)
print("A step costs max(weight traffic / bandwidth, FLOPs / peak).")
print("Weights fix the floor at %.1f ms; compute overtakes it past %.0f tokens."
      % (WEIGHT_BYTES / BANDWIDTH * 1000.0, BALANCE))
print()
print(f"{'tokens in step':>16}{'step ms':>10}{'bound by':>12}"
      f"{'headroom to balance':>22}")
print("-" * 62)
for t in (32, 96, 200, 295, 400, 900):
    b = "memory" if t < BALANCE else "compute"
    print(f"{t:>16}{step_ms(t):>10.2f}{b:>12}"
          f"{max(0, BALANCE - t):>22.0f}")

print()
print("A decode step at batch %d carries %d tokens and is memory-bound, so"
      % (BATCH, BATCH))
print("%.0f tokens of compute headroom sit idle every step." % (BALANCE - BATCH))

print()
print()
print("Colocated and unchunked: a prefill runs as its own step.")
print()
print(f"{'prompt tokens':>15}{'prefill ms':>13}{'decode steps lost':>20}"
      f"{'tokens lost':>14}")
print("-" * 64)
stall = {}
for p in PROMPTS:
    ms = step_ms(p)
    steps = ms / DECODE_MS
    stall[p] = (ms, steps, steps * BATCH)
    print(f"{p:>15}{ms:>13.1f}{steps:>20.1f}{steps * BATCH:>14.0f}")

print()
print()
print("Sustained decode throughput at %.1f prefills per second." % PREFILL_RATE)
print()
ideal = BATCH / (DECODE_MS / 1000.0)
print(f"{'prompt tokens':>15}{'prefill duty':>15}{'decode tok/s':>15}"
      f"{'vs ideal':>11}")
print("-" * 58)
colocated = {}
for p in PROMPTS:
    duty = min(0.99, step_ms(p) / 1000.0 * PREFILL_RATE)
    tp = ideal * (1.0 - duty)
    colocated[p] = (duty, tp)
    print(f"{p:>15}{duty:>15.1%}{tp:>15.0f}{tp / ideal:>11.1%}")

print()
print()
print("Chunked prefill: put a chunk of prefill INTO a decode step, using the")
print("idle compute. A step of batch+chunk tokens costs the same as a step of")
print("batch tokens, as long as the total stays under the balance point.")
print()
print(f"{'chunk':>8}{'tokens/step':>14}{'step ms':>10}{'vs decode-only':>17}"
      f"{'prefill tok/step':>19}")
print("-" * 70)
CHUNKS = [64, 128, 256, 263, 512, 1024]
chunkcost = {}
for k in CHUNKS:
    ms = step_ms(BATCH + k)
    chunkcost[k] = ms
    print(f"{k:>8}{BATCH + k:>14}{ms:>10.2f}{ms / DECODE_MS:>16.2f}x"
          f"{k:>19}")

print()
print()
print("Choosing the chunk at the balance point, so prefill is free.")
print()
FREE_CHUNK = int(BALANCE - BATCH)
print(f"free chunk size: {FREE_CHUNK} tokens (batch {BATCH} + chunk = "
      f"{BATCH + FREE_CHUNK} tokens, balance {BALANCE:.0f})")
print()
print(f"{'prompt tokens':>15}{'chunks needed':>16}{'steps to prefill':>19}"
      f"{'decode tok/s':>15}{'vs ideal':>11}")
print("-" * 78)
chunked = {}
for p in PROMPTS:
    n_chunks = int(math.ceil(p / float(FREE_CHUNK)))
    # Each chunk rides a step that was happening anyway, at no extra step time.
    # The only cost is that prefill capacity is bounded by steps per second.
    steps_per_sec = 1000.0 / step_ms(BATCH + FREE_CHUNK)
    chunks_needed_per_sec = PREFILL_RATE * n_chunks
    if chunks_needed_per_sec <= steps_per_sec:
        eff = step_ms(BATCH + FREE_CHUNK)
    else:
        # Demand exceeds what free chunks can carry; the excess costs real time.
        excess = chunks_needed_per_sec - steps_per_sec
        eff = step_ms(BATCH + FREE_CHUNK) * (1.0 + excess / steps_per_sec)
    tp = BATCH / (eff / 1000.0)
    chunked[p] = (n_chunks, eff, tp)
    print(f"{p:>15}{n_chunks:>16}{n_chunks:>19}{tp:>15.0f}{tp / ideal:>11.1%}")

print()
print()
print("Disaggregated: prefill runs on separate machines. Decode machines never")
print("see a prefill, but the KV cache must be shipped between them.")
print()
KV_PER_TOKEN_MB = 0.131
LINK_GB_S = 900.0
print(f"KV per prompt token: {KV_PER_TOKEN_MB:.3f} MB, link {LINK_GB_S:.0f} GB/s")
print()
print(f"{'prompt tokens':>15}{'KV to ship MB':>16}{'ship ms':>10}"
      f"{'decode tok/s':>15}{'vs ideal':>11}")
print("-" * 68)
disagg = {}
for p in PROMPTS:
    kv_mb = p * KV_PER_TOKEN_MB
    ship_ms = kv_mb / (LINK_GB_S * 1000.0) * 1000.0
    disagg[p] = (kv_mb, ship_ms, ideal)
    print(f"{p:>15}{kv_mb:>16.1f}{ship_ms:>10.2f}{ideal:>15.0f}"
          f"{1.0:>11.1%}")

print()
print()
print("Machines required, since disaggregation buys its throughput with hardware.")
print()
print(f"{'prompt tokens':>15}{'prefill machines':>19}{'total machines':>17}"
      f"{'vs colocated':>15}")
print("-" * 68)
fleet = {}
for p in PROMPTS:
    load = step_ms(p) / 1000.0 * PREFILL_RATE
    fleet[p] = load
    print(f"{p:>15}{load:>19.2f}{DECODE_MACHINES + load:>17.2f}"
          f"{(DECODE_MACHINES + load) / DECODE_MACHINES:>14.2f}x")

print()
print()
print("Throughput per machine -- the comparison that decides it.")
print()
print(f"{'prompt tokens':>15}{'colocated':>12}{'chunked':>11}"
      f"{'disaggregated':>16}{'best':>16}")
print("-" * 72)
winner = {}
for p in PROMPTS:
    co = colocated[p][1] / DECODE_MACHINES
    ch = chunked[p][2] / DECODE_MACHINES
    di = disagg[p][2] / (DECODE_MACHINES + fleet[p])
    opts = {"colocated": co, "chunked": ch, "disaggregated": di}
    best = max(opts, key=lambda k: opts[k])
    winner[p] = (co, ch, di, best)
    print(f"{p:>15}{co:>12.0f}{ch:>11.0f}{di:>16.0f}{best:>16}")

print(f"""
The headroom table is the mechanism, and it comes straight from ch:inf-cpu-gpu. A step
is bound by weights until it carries {BALANCE:.0f} tokens. A decode step at batch
{BATCH} carries {BATCH} -- so **{BALANCE - BATCH:.0f} tokens of compute capacity sit
idle in every decode step the system runs.**

That idle capacity is what cite:agrawal2023sarathi spends. A step carrying
{BATCH} decodes plus a {FREE_CHUNK}-token prefill chunk takes
{chunkcost[263] if 263 in chunkcost else step_ms(BATCH + FREE_CHUNK):.2f}ms against a
decode-only step's {DECODE_MS:.2f}ms -- **the same time**, because both are still
memory-bound. The prefill is genuinely free until the balance point, and expensive
immediately after: a {512}-token chunk costs
{chunkcost[512] / DECODE_MS:.2f} times a decode step.

**Chunk size is not a tuning parameter with a smooth curve. It has a cliff at the
balance point**, and the correct value is {FREE_CHUNK} tokens for this batch --
computed, not searched.

The stall table shows what happens without that. A {PROMPTS[2]}-token prompt runs as
its own step costing {stall[PROMPTS[2]][0]:.1f}ms, which is
{stall[PROMPTS[2]][1]:.1f} decode steps during which every one of {BATCH} sequences
produces nothing. **One prompt costs {stall[PROMPTS[2]][2]:.0f} tokens other users were
waiting for** (eq:prefill-stalls-decode), and the victim is never the request that
caused it.

The duty table turns that into sustained throughput: at {PROMPTS[2]}-token prompts,
prefill occupies {colocated[PROMPTS[2]][0]:.1%} of step time and decode falls to
{colocated[PROMPTS[2]][1] / ideal:.1%} of ideal. At {PROMPTS[3]} tokens the device is
{colocated[PROMPTS[3]][0]:.0%} prefill. **A colocated server with long prompts is a
prefill server that occasionally decodes.**

Chunking recovers most of it: {chunked[PROMPTS[2]][2] / ideal:.1%} of ideal at
{PROMPTS[2]} tokens against colocated's {colocated[PROMPTS[2]][1] / ideal:.1%}, and
{chunked[PROMPTS[3]][2] / ideal:.1%} at {PROMPTS[3]} tokens against
{colocated[PROMPTS[3]][1] / ideal:.1%}.

Disaggregation recovers all of it -- decode machines run at {1.0:.0%} by construction --
at the cost of shipping {disagg[PROMPTS[2]][0]:.1f} MB per {PROMPTS[2]}-token prompt,
taking {disagg[PROMPTS[2]][1]:.2f}ms over a fast link. That is cheap, and it is cheap
**only over a fast link**; the same design across a datacentre network is a different
calculation entirely.

The per-machine table is the honest comparison, because disaggregation buys its
{1.0:.0%} by adding hardware. At {PROMPTS[0]} tokens the winner is
{winner[PROMPTS[0]][3]}; at {PROMPTS[3]} tokens it is {winner[PROMPTS[3]][3]}.

**Neither approach dominates**, and the crossover sits inside the range real products
operate in. The choice turns on three things these tables make explicit: prompt length,
interconnect speed, and whether the fleet can be heterogeneous at all --
cite:patel2023splitwise's contribution being precisely that prefill and decode machines
need not be the same generation, which this listing's uniform-machine model cannot
express and which moves the comparison in disaggregation's favour.

A design review that presents either as settled has skipped the measurement.""")

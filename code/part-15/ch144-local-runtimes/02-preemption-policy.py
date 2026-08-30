# -*- coding: utf-8 -*-
# Extracted from: Chapter 144 — Local Inference Runtimes: Ollama, vLLM, and MLX
# Source: src/.../ch144-local-runtimes.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""What a runtime does when it runs out of cache, which is the thing it will do.

ch:q-memory-math computes how many sequences fit. It does not say what happens
when the number is exceeded, and that will happen: sequence lengths are unknown
when a request is admitted, so a scheduler that admits by current size will
eventually find the cache full with everything mid-generation.

There are three answers, and every serving stack implements one of them
(eq:preemption-policy). Refuse to admit until memory frees. Evict a running
sequence and RECOMPUTE its prefill later. Or evict it to host memory and SWAP it
back, paying the transfer instead of the recomputation.

This listing simulates all three and measures the quantity that distinguishes
them: how much of the machine's work was thrown away.
"""
import numpy as np

STEP_BASE_MS = 22.0
CROSSOVER_B = 48
STEP_SLOPE_MS = 0.42
PREFILL_TOK_PER_S = 9000.0
CACHE_TOKENS = 120_000          # total KV slots, from ch:q-memory-math
SWAP_TOK_PER_MS = 60.0          # host transfer rate, tokens of cache per ms


def step_ms(b):
    return STEP_BASE_MS + STEP_SLOPE_MS * max(0, b - CROSSOVER_B)


def workload(n, rate, seed=3):
    r = np.random.default_rng(seed)
    prompt = np.clip(r.lognormal(6.5, 1.2, n).astype(int), 32, 32768)
    out = np.clip(r.lognormal(5.0, 1.0, n).astype(int), 8, 4096)
    return prompt, out, np.cumsum(r.exponential(1000.0 / rate, n))


N = 500
P, O, ARRIVE = workload(N, 5.0)


def simulate(policy):
    now, nxt = 0.0, 0
    done = np.full(N, np.nan)
    active = {}                  # index -> [tokens generated, cache length]
    swapped = {}                 # index -> cache length, held in host memory
    queue = []
    prefill_tokens = 0.0         # total prefill work done, including redone
    useful_tokens = 0.0          # prefill work that was not later discarded
    preempts = 0

    def used():
        return sum(v[1] for v in active.values())

    while nxt < N or active or queue or swapped:
        while nxt < N and ARRIVE[nxt] <= now:
            queue.append(nxt); nxt += 1
        if not active and not queue and not swapped and nxt < N:
            now = max(now, ARRIVE[nxt]); continue

        # Admit if there is room. Swapped-out sequences come back first.
        if swapped and used() + max(swapped.values()) <= CACHE_TOKENS:
            i = min(swapped)
            now += swapped[i] / SWAP_TOK_PER_MS
            active[i] = [O[i] - (O[i] - active.get(i, [O[i], 0])[0]), swapped[i]]
            active[i] = [active[i][0], swapped.pop(i)]
        elif queue and used() + P[queue[0]] <= CACHE_TOKENS:
            i = queue.pop(0)
            now += P[i] / PREFILL_TOK_PER_S * 1000.0
            prefill_tokens += P[i]
            useful_tokens += P[i]
            active[i] = [O[i], int(P[i])]

        if not active:
            if queue or swapped:
                # Nothing fits and nothing is running: the cache is stuck.
                now += STEP_BASE_MS
            continue

        now += step_ms(len(active))
        for i in list(active):
            active[i][0] -= 1
            active[i][1] += 1
            if active[i][0] <= 0:
                done[i] = now - ARRIVE[i]
                del active[i]

        # Over budget? Apply the policy to the most recently admitted sequence.
        while used() > CACHE_TOKENS and active:
            victim = max(active, key=lambda k: active[k][1])
            preempts += 1
            if policy == "recompute":
                useful_tokens -= P[victim]      # that prefill is now wasted
                queue.insert(0, victim)
                del active[victim]
            elif policy == "swap":
                swapped[victim] = active[victim][1]
                now += active[victim][1] / SWAP_TOK_PER_MS
                del active[victim]
            else:                                # "reject": never over budget
                del active[victim]
                done[victim] = np.nan
    return done, now, prefill_tokens, useful_tokens, preempts


print(f"{N} requests at 5/s, cache holds {CACHE_TOKENS:,} tokens.")
print("Prompt median", int(np.median(P)), "tokens, output median",
      int(np.median(O)), "tokens.")
print()
print(f"{'policy':>14}{'completed':>11}{'throughput':>12}{'latency p50':>13}"
      f"{'latency p99':>13}{'preempts':>10}{'wasted work':>13}")
print("-" * 86)

res = {}
for pol in ("reject", "recompute", "swap"):
    d, span, pre, useful, pre_n = simulate(pol)
    ok = np.isfinite(d).sum()
    res[pol] = (ok, ok / (span / 1000.0), np.nanpercentile(d, 50),
                np.nanpercentile(d, 99), pre_n,
                1.0 - useful / max(pre, 1.0))
    print(f"{pol:>14}{ok:>11}{res[pol][1]:>12.2f}{res[pol][2]:>13.0f}"
          f"{res[pol][3]:>13.0f}{pre_n:>10}{res[pol][5]:>12.1%}")

print()
print()
print("How the answer moves with cache size. Recompute policy.")
print()
print(f"{'cache tokens':>14}{'throughput':>12}{'latency p99':>13}"
      f"{'preempts':>10}{'wasted work':>13}")
print("-" * 62)
grid = {}
for cap in (60_000, 120_000, 240_000, 480_000):
    CACHE_TOKENS = cap
    d, span, pre, useful, pre_n = simulate("recompute")
    ok = np.isfinite(d).sum()
    grid[cap] = (ok / (span / 1000.0), np.nanpercentile(d, 99), pre_n,
                 1.0 - useful / max(pre, 1.0))
    print(f"{cap:>14,}{grid[cap][0]:>12.2f}{grid[cap][1]:>13.0f}"
          f"{pre_n:>10}{grid[cap][3]:>12.1%}")

rj, rc, sw = res["reject"], res["recompute"], res["swap"]
print(f"""
The completed column is the first thing to read, because one policy is not
answering the same question as the others.

Rejection completes {rj[0]} of {N} requests. The other two complete
{rc[0]} and {sw[0]}. Rejection does not queue the sequence it evicts -- it drops
it -- so its throughput number is measuring a service that is failing requests,
and comparing it to the others on throughput alone would be a category error. It
is in the table to make that visible, because a system under memory pressure that
reports good latency is often reporting the latency of the requests it did not
drop (eq:preemption-policy).

Between the two policies that keep every request, the difference is what they pay
to free the memory.

Recompute discards the victim's cache and re-runs its prefill later. That is
simple, needs no host transfers, and throws work away: {rc[5]:.1%} of all prefill
tokens processed were later discarded and had to be done again. The machine did
that work and has nothing to show for it.

Swap moves the victim's cache to host memory and brings it back, paying a transfer
in each direction rather than a recomputation. It wastes {sw[5]:.1%} of prefill
work -- none, by construction -- and pays {sw[4]} transfers instead.

Which wins depends on a ratio you can compute rather than guess: the cost of
re-prefilling P tokens against the cost of moving P tokens of cache twice.
Prefill runs at {PREFILL_TOK_PER_S:,.0f} tokens per second and the transfer at
{SWAP_TOK_PER_MS * 1000:,.0f} cache-tokens per second, so swapping is cheaper
here -- and on a machine with a slow host link, or with grouped-query attention
making the cache small relative to the prefill, it would not be.

That is the useful form of the comparison. It is not a question about which
runtime is better designed; it is a hardware ratio, and a stack that supports both
and picks by measurement is doing the right thing.

The second table shows how the whole question dissolves with enough memory. At
{60_000:,} cache tokens the recompute policy wastes {grid[60_000][3]:.1%} of its
prefill work and preempts {grid[60_000][2]} times. At {480_000:,} it preempts
{grid[480_000][2]} times and wastes {grid[480_000][3]:.1%}.

So preemption is not a feature to optimise, it is a symptom of being under-
provisioned, and every ounce of memory recovered by the previous chapters --
paged allocation, grouped-query attention, KV quantization -- shows up here as
preemptions that do not happen. That is the connection worth carrying: the
memory chapters and the scheduling chapters are about the same resource, and
work done in one appears as a different quantity in the other.

Two consequences for choosing and configuring a runtime.

First, ask what it does under pressure, not what it does when comfortable. Every
stack looks similar at 30% cache utilisation. The differences appear at 95%, which
is where any economically-run deployment sits, and a benchmark that never fills
the cache has not tested the behaviour that will define production.

Second, watch the wasted-work fraction rather than the preemption count. Preempts
are cheap under swap and expensive under recompute, so the count alone does not
say whether anything is wrong. The fraction of prefill work discarded is the
quantity that tells you the machine is running to stand still.""")

# -*- coding: utf-8 -*-
# Extracted from: Chapter 144 — Local Inference Runtimes: Ollama, vLLM, and MLX
# Source: src/.../ch144-local-runtimes.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""What actually separates one inference runtime from another: the scheduler.

Feature lists go stale. The scheduling policy does not, and it is where the
throughput differences between serving stacks come from -- not from kernels, which
are largely shared.

This listing simulates three policies on one workload with one set of hardware
constants (eq:scheduling-policy). Static batching collects a batch and runs it to
completion. Continuous batching admits a new request the moment a slot frees.
Chunked prefill additionally refuses to let a long prompt stall everyone else.

The metrics are the two that matter and disagree: aggregate throughput, and the
latency the individual user experiences.
"""
import numpy as np

rng = np.random.default_rng(277)

# Hardware constants, in the shape ch:q-gguf derived: decode is memory-bound and
# nearly batch-independent below the crossover, then compute-bound above it.
STEP_BASE_MS = 22.0          # one decode step, small batch
CROSSOVER_B = 48             # ch:q-gguf's eq:memory-bound-crossover
STEP_SLOPE_MS = 0.42         # per extra sequence above the crossover
PREFILL_TOK_PER_S = 9000.0   # prefill is compute-bound: tokens per second
MAX_SEQS = 64                # what the memory budget allows (ch:q-memory-math)


def step_ms(b):
    return STEP_BASE_MS + STEP_SLOPE_MS * max(0, b - CROSSOVER_B)


def workload(n, rate, seed=0):
    """`rate` is requests per second. Measuring throughput needs an overloaded
    system; measuring latency needs an underloaded one, so the two questions get
    two workloads."""
    r = np.random.default_rng(1000 + seed)
    prompt = np.clip(r.lognormal(6.4, 1.1, n).astype(int), 32, 32768)
    out = np.clip(r.lognormal(4.6, 0.8, n).astype(int), 8, 2048)
    arrive = np.cumsum(r.exponential(1000.0 / rate, n))
    return prompt, out, arrive


P, O, ARRIVE = None, None, None
N = 0


def use(rate, n=600):
    global P, O, ARRIVE, N
    P, O, ARRIVE = workload(n, rate)
    N = len(P)


def simulate(policy, chunk=None):
    """Returns per-request time-to-first-token and total latency, in ms."""
    now = 0.0
    nxt = 0                       # next request not yet admitted
    ttft = np.full(N, np.nan)
    done = np.full(N, np.nan)
    active, remaining = [], {}
    pending_prefill = []          # (index, tokens left to prefill)
    gaps, last_step = [], None    # interval between consecutive decode steps

    while nxt < N or active or pending_prefill:
        if policy == "static":
            # Fill a batch only when the previous one has fully drained.
            if not active and not pending_prefill:
                now = max(now, ARRIVE[nxt])
                take = []
                while nxt < N and len(take) < MAX_SEQS and ARRIVE[nxt] <= now:
                    take.append(nxt); nxt += 1
                if not take:
                    take = [nxt]; nxt += 1
                now += sum(P[i] for i in take) / PREFILL_TOK_PER_S * 1000.0
                for i in take:
                    ttft[i] = now - ARRIVE[i]
                    remaining[i] = O[i]
                active = take
        else:
            # Admit whenever there is room and a request has arrived.
            while (nxt < N and len(active) + len(pending_prefill) < MAX_SEQS
                   and ARRIVE[nxt] <= now):
                pending_prefill.append([nxt, int(P[nxt])]); nxt += 1
            if not active and not pending_prefill and nxt < N:
                now = max(now, ARRIVE[nxt])
                continue
            if pending_prefill:
                if policy == "continuous":
                    # A whole prompt is prefilled in one go, stalling decode.
                    i, tok = pending_prefill.pop(0)
                    now += tok / PREFILL_TOK_PER_S * 1000.0
                    ttft[i] = now - ARRIVE[i]
                    remaining[i] = O[i]
                    active.append(i)
                else:
                    # Chunked: one chunk per scheduler tick, then decode.
                    i, tok = pending_prefill[0]
                    piece = min(chunk, tok)
                    now += piece / PREFILL_TOK_PER_S * 1000.0
                    pending_prefill[0][1] -= piece
                    if pending_prefill[0][1] <= 0:
                        pending_prefill.pop(0)
                        ttft[i] = now - ARRIVE[i]
                        remaining[i] = O[i]
                        active.append(i)

        if active:
            if last_step is not None:
                gaps.append(now - last_step)
            now += step_ms(len(active))
            last_step = now
            finished = []
            for i in active:
                remaining[i] -= 1
                if remaining[i] <= 0:
                    done[i] = now - ARRIVE[i]
                    finished.append(i)
            for i in finished:
                active.remove(i)
        elif not pending_prefill and nxt >= N:
            break
    return ttft, done, now, np.array(gaps) if gaps else np.array([0.0])


use(40)
print(f"SATURATED: {N} requests offered far faster than any policy can serve,")
print(f"so the throughput column is each policy's ceiling. Prompt median "
      f"{int(np.median(P))} tokens, output median {int(np.median(O))}.")
print()
print(f"{'policy':>24}{'throughput':>14}{'vs static':>12}")
print("-" * 50)
sat = {}
for name, pol, ck in (("static batching", "static", None),
                      ("continuous batching", "continuous", None),
                      ("continuous + chunked", "chunked", 512)):
    _, _, span, _ = simulate(pol, ck)
    sat[name] = N / (span / 1000.0)
    print(f"{name:>24}{sat[name]:>14.2f}{sat[name]/sat['static batching']:>11.2f}x")

use(3.0)
print()
print()
print(f"UNDERLOADED at 3 requests per second, so these measure scheduling")
print("rather than queueing. 'Stall' is the gap between consecutive decode")
print("steps -- what a streaming user sees as the output pausing.")
print()
print(f"{'policy':>24}{'TTFT p50':>11}{'TTFT p99':>11}{'latency p50':>13}"
      f"{'stall p99':>13}{'worst stall':>13}")
print(f"{'':>24}{'ms':>11}{'ms':>11}{'ms':>13}{'ms':>13}{'ms':>13}")
print("-" * 85)
res = {}
for name, pol, ck in (("static batching", "static", None),
                      ("continuous batching", "continuous", None),
                      ("continuous + chunked", "chunked", 512)):
    t, d, span, g = simulate(pol, ck)
    res[name] = (np.nanpercentile(t, 50), np.nanpercentile(t, 99),
                 np.nanpercentile(d, 50), np.percentile(g, 99), g.max())
    print(f"{name:>24}{res[name][0]:>11.0f}{res[name][1]:>11.0f}"
          f"{res[name][2]:>13.0f}{res[name][3]:>13.0f}{res[name][4]:>13.0f}")

print()
print()
print("Chunk size is the dial between the prefilling request and everyone else.")
print("Underloaded, so these are scheduling effects.")
print()
print(f"{'chunk':>8}{'TTFT p50':>11}{'TTFT p99':>11}{'stall p99':>13}"
      f"{'worst stall':>13}")
print("-" * 56)
ck_rows = {}
for ck in (128, 512, 2048, 8192):
    t, d, span, g = simulate("chunked", ck)
    ck_rows[ck] = (np.nanpercentile(t, 50), np.nanpercentile(t, 99),
                   np.percentile(g, 99), g.max())
    print(f"{ck:>8}{ck_rows[ck][0]:>11.0f}{ck_rows[ck][1]:>11.0f}"
          f"{ck_rows[ck][2]:>13.0f}{ck_rows[ck][3]:>13.0f}")

st, co, chk = (res["static batching"], res["continuous batching"],
               res["continuous + chunked"])
print(f"""
Three policies, one workload, one set of hardware constants, and the same
kernels. Everything that differs between the rows is when work is scheduled.

Static batching reaches {sat['static batching']:.2f} requests per second.
Continuous batching reaches {sat['continuous batching']:.2f} --
{sat['continuous batching']/sat['static batching']:.2f}x more
(eq:scheduling-policy).

The mechanism is in the latency table. Under static batching a request generating
thirty tokens sits in the batch until the request generating two thousand
finishes, because the batch advances and retires as a unit. Its slot is occupied
and idle for most of that time, so the machine runs a smaller EFFECTIVE batch than
it was configured for. That is where the throughput went -- not into slower steps,
but into steps carrying fewer live sequences than they could have.

The latency consequence is brutal: TTFT p50 of {st[0]:.0f} ms against continuous
batching's {co[0]:.0f} ms, a factor of {st[0]/co[0]:.0f}. A request that arrives
one step after a batch starts waits for the entire batch to drain before it is
even admitted.

So continuous batching wins on both columns, which is why every serving stack
adopted it. The interesting question is what it does NOT fix, and that is the
stall column.

Admitting a request under continuous batching means prefilling its prompt, and
prefill is compute-bound and takes as long as the prompt is long. While it runs,
every sequence already decoding is stopped. The worst stall is {co[4]:.0f} ms --
one and a half seconds during which every streaming user's output freezes,
because somebody else sent a long prompt.

Chunked prefill interleaves: 512 prompt tokens, then a decode step, then the next
512. The long prompt takes the same total time and no longer stops anyone. The
worst stall falls to {chk[4]:.0f} ms, a factor of {co[4]/chk[4]:.0f}.

And it is not free, which is the part worth dwelling on. TTFT p50 rises from
{co[0]:.0f} to {chk[0]:.0f} ms and p99 from {co[1]:.0f} to {chk[1]:.0f}, because
the request being prefilled now has its prefill spread across many scheduler
ticks. Throughput drops slightly, from {sat['continuous batching']:.2f} to
{sat['continuous + chunked']:.2f}.

Chunking trades the prefilling request's latency for everyone else's smoothness.
That is a real trade with no dominant side, and the second table makes it a dial.

At a chunk of 128 the worst stall is {ck_rows[128][3]:.0f} ms and TTFT p50 is
{ck_rows[128][0]:.0f} ms. At 8192 the stall is {ck_rows[8192][3]:.0f} ms and TTFT
p50 is {ck_rows[8192][0]:.0f} ms. There is no setting that is best on both
columns, which is exactly why it is exposed as a configuration parameter rather
than chosen for you -- and why leaving it at a default is a decision about your
users that somebody else made.

Which is the durable way to think about inference runtimes, and it outlasts any
feature comparison. They are not fast or slow. They implement different points in
this space, and the point they implement follows from what they were built for.

A runtime designed for one user on a laptop has a batch size of one. Continuous
batching buys it nothing, because there is never a second sequence to admit.
Chunked prefill only delays its own first token, because there is nobody else to
protect. Every line of scheduler code is overhead against a fixed budget. Such a
runtime should optimise single-stream latency, startup time and memory footprint
-- and the ones that people run locally do exactly that.

A runtime designed to serve many users needs every row here, because at batch 1 it
is wasting the hardware and at batch 64 the scheduling policy IS the performance.
Its complexity is not gratuitous; it is the price of the
{sat['continuous batching']/sat['static batching']:.2f}x, and of the
{co[4]/chk[4]:.0f}x reduction in stall on top of it.

So the question to ask about a runtime is not which is faster. It is which regime
its defaults assume, and whether that is your regime. Running a server stack for a
single local user buys the overhead without the benefit. Running a single-stream
stack behind an API buys a fraction of the hardware you paid for. Both mistakes
are common, and neither shows up in a benchmark run the wrong way round.""")

# -*- coding: utf-8 -*-
# Extracted from: Chapter 202 — Serving Stacks: vLLM, TensorRT-LLM, and Triton
# Source: src/.../ch202-serving-stacks.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The roofline model has no term for the time before any arithmetic happens.

ch:inf-cpu-gpu and ch:inf-gpu-memory modelled a step as max(traffic/bandwidth,
FLOPs/peak). Both terms are proportional to work. Neither describes the fixed cost of
ASKING the device to do the work: each kernel must be launched, its arguments marshalled,
its dependencies resolved.

A transformer decode step launches hundreds of kernels, and that cost is FIXED -- it
does not scale with batch size. Since ch:inf-cpu-gpu showed the step time is also fixed
throughout the memory-bound regime, the two are a constant ratio: launch overhead is the
same share of every decode step a production system runs
(eq:launch-overhead-is-a-floor).

This listing measures where the floor sits, why it explains benchmarks that undershoot
the roofline, and what graph capture recovers.
"""
LAYERS = 32
KERNELS_PER_LAYER = 14        # projections, norms, activations, cache writes
LAUNCH_US = 4.2               # per-kernel launch and dispatch, unbatched
GRAPH_US = 0.35               # per-kernel cost when the graph is pre-captured
WEIGHT_BYTES = 14.0e9
BANDWIDTH = 3.35e12
PEAK = 9.89e14
PARAMS = 7.0e9
BATCHES = [1, 8, 32, 128, 256, 512, 1024, 2048]

KERNELS = LAYERS * KERNELS_PER_LAYER


def roofline_ms(batch):
    t_mem = WEIGHT_BYTES / BANDWIDTH
    t_flop = 2.0 * PARAMS * batch / PEAK
    return max(t_mem, t_flop) * 1000.0


def launch_ms(per_kernel_us):
    return KERNELS * per_kernel_us / 1000.0


print("A %d-layer decode step launches %d kernels." % (LAYERS, KERNELS))
print("Unbatched launch cost: %.2f us each, %.2f ms total."
      % (LAUNCH_US, launch_ms(LAUNCH_US)))
print("Captured in a graph:   %.2f us each, %.2f ms total."
      % (GRAPH_US, launch_ms(GRAPH_US)))
print()
print("Step time: what the roofline predicts, and what it costs with launches.")
print()
print(f"{'batch':>8}{'roofline ms':>14}{'launch ms':>12}{'actual ms':>12}"
      f"{'launch share':>15}{'roofline error':>17}")
print("-" * 80)
tab = {}
L = launch_ms(LAUNCH_US)
for b in BATCHES:
    rf = roofline_ms(b)
    act = rf + L
    tab[b] = (rf, act, L / act)
    print(f"{b:>8}{rf:>14.2f}{L:>12.2f}{act:>12.2f}{L / act:>15.1%}"
          f"{rf / act - 1.0:>16.1%}")

print()
print()
print("Throughput, which is where the error becomes visible as a benchmark that")
print("undershoots.")
print()
print(f"{'batch':>8}{'roofline tok/s':>17}{'actual tok/s':>15}"
      f"{'shortfall':>12}")
print("-" * 54)
short = {}
for b in BATCHES:
    rf, act, _ = tab[b]
    r_t = b / (rf / 1000.0)
    a_t = b / (act / 1000.0)
    short[b] = (r_t, a_t, 1.0 - a_t / r_t)
    print(f"{b:>8}{r_t:>17.0f}{a_t:>15.0f}{1.0 - a_t / r_t:>11.0%}")

print()
print()
print("Graph capture removes most of the per-launch cost by recording the kernel")
print("sequence once and replaying it.")
print()
G = launch_ms(GRAPH_US)
print(f"{'batch':>8}{'plain ms':>11}{'captured ms':>14}{'speedup':>10}"
      f"{'captured tok/s':>17}{'vs roofline':>14}")
print("-" * 76)
cap = {}
for b in BATCHES:
    rf, act, _ = tab[b]
    capt = rf + G
    cap[b] = (capt, b / (capt / 1000.0))
    print(f"{b:>8}{act:>11.2f}{capt:>14.2f}{act / capt:>9.2f}x"
          f"{b / (capt / 1000.0):>17.0f}{(b / (capt / 1000.0)) / short[b][0]:>13.1%}")

print()
print()
print("Where the floor binds: the batch below which launch overhead exceeds the")
print("arithmetic it is launching.")
print()
print(f"{'per-kernel us':>15}{'launch ms':>12}{'crossover batch':>18}"
      f"{'max tok/s at floor':>21}")
print("-" * 68)
floor = {}
for us in (8.0, 4.2, 2.0, 1.0, GRAPH_US):
    lm = launch_ms(us)
    # Crossover: the batch at which roofline time equals launch time.
    b = 1
    while roofline_ms(b) < lm and b < 100000:
        b *= 2
    floor[us] = (lm, b, 1000.0 / lm)
    print(f"{us:>15.2f}{lm:>12.2f}{b:>18}{1000.0 / lm:>21.0f}")
print()
print("(max tok/s at floor is the ceiling from launches alone, at batch 1)")

print()
print()
print("And why this matters more as models get faster: the arithmetic shrinks and")
print("the launch cost does not.")
print()
print(f"{'model':>14}{'weights GB':>13}{'roofline ms':>14}{'launch ms':>12}"
      f"{'launch share':>15}")
print("-" * 70)
share = {}
for label, gb in (("70B bf16", 140.0), ("13B bf16", 26.0), ("7B bf16", 14.0),
                  ("7B fp8", 7.0), ("3B int4", 1.5)):
    rf = gb * 1e9 / BANDWIDTH * 1000.0
    share[label] = L / (rf + L)
    print(f"{label:>14}{gb:>13.1f}{rf:>14.2f}{L:>12.2f}"
          f"{L / (rf + L):>15.1%}")

print(f"""
The first table is the term the roofline omits. A {LAYERS}-layer step launches
{KERNELS} kernels, and at {LAUNCH_US:.1f} microseconds each that is
{L:.2f}ms of pure dispatch before any arithmetic happens
(eq:launch-overhead-is-a-floor).

At batch {1} the roofline predicts {tab[1][0]:.2f}ms and the step actually takes
{tab[1][1]:.2f}ms -- launch overhead is **{tab[1][2]:.1%} of the step**, and the
roofline is wrong by {1.0 - tab[1][0] / tab[1][1]:.0%}.

Now read down that column, because it does something the intuition does not expect. At
batch {8} launch overhead is {tab[8][2]:.1%}. At batch {32}, {tab[32][2]:.1%}. At batch
{256}, {tab[256][2]:.1%}. **It does not shrink at all.**

The reason is ch:inf-cpu-gpu's result: below the balance point the step time is
*constant* in batch size, because the weight read dominates and it happens once
regardless. Launch overhead is also constant. Two constants have a constant ratio, so
**launch overhead is a fixed share of every decode step in the entire memory-bound
regime** -- which is where decode lives.

It falls only once the batch pushes the step into the compute-bound regime: at batch
{1024} it is {tab[1024][2]:.1%} and at batch {2048}, {tab[2048][2]:.1%}. Those are batch
sizes above what ch:inf-gpu-memory's capacity frontier permits at any useful context
length.

The throughput table is how this shows up in practice, and it inherits the same
flatness: a benchmark undershoots the roofline by {short[1][2]:.0%} at batch {1},
{short[8][2]:.0%} at batch {8}, and {short[256][2]:.0%} at batch {256}.

Teams that compute an expected throughput from bandwidth and measure something lower
usually conclude the memory system is underperforming, and go looking for it in the
memory system. It is not there. **The device is waiting to be told what to do**, and no
amount of batching reveals or fixes it.

Graph capture is the fix, and it is a large one at the batch sizes where it matters. It
records the kernel sequence once and replays it as a unit, cutting per-kernel cost from
{LAUNCH_US:.1f} to {GRAPH_US:.2f} microseconds. At batch {1} that is
{tab[1][1] / cap[1][0]:.2f}x, and it is the same {tab[256][1] / cap[256][0]:.2f}x at
batch {256}, for the same reason the overhead share was flat. At batch {2048}, where
compute finally dominates, it is {tab[2048][1] / cap[2048][0]:.2f}x.

**Graph capture is the one technique in this part whose benefit does not depend on
batch size.** ch:inf-batching's continuous batching and ch:inf-cpu-gpu's
arithmetic-intensity argument both need a large batch to help; ch:inf-parallelism's
dimensions need a fast link. Graph capture needs nothing, and delivers the same
{tab[32][1] / cap[32][0]:.2f}x across the whole operating range.

That makes it a complement to everything else rather than a substitute, which is
unusual in this part -- and it is why the previous listing's overlap analysis put
graph capture in a category of its own.

The floor table gives the operating constraint directly. At {LAUNCH_US:.1f} microseconds
per kernel, the launch cost alone caps a batch-1 deployment at
{floor[LAUNCH_US][2]:.0f} tokens a second no matter what the hardware does. Captured, the
same cap is {floor[GRAPH_US][2]:.0f}.

**That is a ceiling the roofline cannot express**, because it does not scale with any
quantity the roofline measures. A faster device does not raise it. More bandwidth does
not raise it. Only launching fewer kernels or launching them more cheaply does.

The last table is why this is getting worse rather than better. Launch cost is fixed;
the arithmetic it launches is not. A {140.0:.0f} GB model spends
{share['70B bf16']:.1%} of its step on launches; a {1.5:.1f} GB one spends
{share['3B int4']:.1%}.

So **every optimisation that makes the model cheaper makes launch overhead a larger
share of what remains.** Quantisation, smaller models, sparsity, better kernels -- each
one shrinks the term the roofline models and leaves the term it does not. A team that
quantises a small model and measures less speedup than the memory arithmetic predicted
has met this, and the missing factor is in the last column rather than in the
quantisation.""")

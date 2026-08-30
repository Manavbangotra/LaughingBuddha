# -*- coding: utf-8 -*-
# Extracted from: Chapter 199 — Batching and Continuous Batching
# Source: src/.../ch199-batching.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Static batching wastes most of its capacity on a queue of unequal-length jobs.

ch:inf-cpu-gpu established that batching is what makes a GPU worth using. This listing
asks what it costs, and the answer depends entirely on how the batch is formed.

A STATIC batch runs to completion together: every sequence occupies a slot until the
LONGEST one finishes. With generation lengths varying by an order of magnitude, most
slots spend most of their time computing padding
(eq:static-batching-pays-for-the-longest).

CONTINUOUS batching lets a finished sequence leave and a waiting one take its slot
immediately. This listing measures the gap, and finds it is larger than the length
variation alone suggests.
"""
import math

# Generation length distribution: mostly short, occasionally very long.
# (length in tokens, share of requests)
LENGTHS = [
    (40, 0.31),
    (110, 0.27),
    (280, 0.21),
    (700, 0.13),
    (1800, 0.06),
    (4200, 0.02),
]
BATCHES = [1, 4, 8, 16, 32, 64]
STEP_MS = 18.0        # time for one decode step across the batch

mean_len = sum(l * p for l, p in LENGTHS)
print("Generation length distribution.")
print()
print(f"{'length':>10}{'share':>9}{'cumulative':>13}")
print("-" * 34)
c = 0.0
for l, p in LENGTHS:
    c += p
    print(f"{l:>10}{p:>9.0%}{c:>13.0%}")
print()
print(f"mean {mean_len:.0f} tokens, max {LENGTHS[-1][0]} tokens, "
      f"ratio {LENGTHS[-1][0] / LENGTHS[0][0]:.0f}x")


def expected_max(n):
    """E[max of n independent draws] from the length distribution."""
    total = 0.0
    prev = 0.0
    cum = 0.0
    for l, p in LENGTHS:
        cum += p
        total += l * (cum ** n - prev)
        prev = cum ** n
    return total


print()
print()
print("Static batching: every slot is held until the LONGEST sequence in the")
print("batch finishes. Useful work is the sum of lengths; paid work is batch")
print("size times the maximum.")
print()
print(f"{'batch':>8}{'E[max len]':>13}{'useful tokens':>16}"
      f"{'paid slots':>13}{'utilisation':>14}")
print("-" * 66)
static = {}
for b in BATCHES:
    emax = expected_max(b)
    useful = b * mean_len
    paid = b * emax
    static[b] = (emax, useful, paid, useful / paid)
    print(f"{b:>8}{emax:>13.0f}{useful:>16.0f}{paid:>13.0f}"
          f"{useful / paid:>14.1%}")

print()
print()
print("What that does to throughput. A step serves the whole batch, so throughput")
print("is batch size over step time -- but only for slots doing real work.")
print()
print(f"{'batch':>8}{'nominal tok/s':>16}{'effective tok/s':>18}"
      f"{'vs batch 1':>13}")
print("-" * 57)
eff_static = {}
for b in BATCHES:
    nominal = b / (STEP_MS / 1000.0)
    effective = nominal * static[b][3]
    eff_static[b] = (nominal, effective)
    print(f"{b:>8}{nominal:>16.0f}{effective:>18.0f}"
          f"{effective / eff_static[1][1]:>12.1f}x")

print()
print()
print("Continuous batching: a finished sequence leaves and a queued one takes")
print("its slot on the next step. Every slot is always doing real work.")
print()
print(f"{'batch':>8}{'static tok/s':>15}{'continuous tok/s':>19}"
      f"{'gain':>9}{'utilisation':>14}")
print("-" * 66)
cont = {}
for b in BATCHES:
    continuous = b / (STEP_MS / 1000.0)
    cont[b] = continuous
    print(f"{b:>8}{eff_static[b][1]:>15.0f}{continuous:>19.0f}"
          f"{continuous / eff_static[b][1]:>8.1f}x{1.0:>14.1%}")

print()
print()
print("Where the gap comes from: it grows with batch size, because E[max] grows")
print("with the number of draws while the mean does not.")
print()
print(f"{'batch':>8}{'E[max]/mean':>15}{'static util':>14}{'gap':>9}")
print("-" * 48)
for b in BATCHES:
    print(f"{b:>8}{static[b][0] / mean_len:>15.2f}{static[b][3]:>14.1%}"
          f"{cont[b] / eff_static[b][1]:>8.1f}x")

print()
print()
print("The same comparison against a tighter length distribution -- what happens")
print("if you cap generation length.")
print()
print(f"{'cap':>8}{'mean len':>11}{'E[max] at b=32':>17}"
      f"{'static util':>14}{'continuous gain':>18}")
print("-" * 70)
caps = {}
for cap in (4200, 1800, 700, 280):
    sub = [(min(l, cap), p) for l, p in LENGTHS]
    m = sum(l * p for l, p in sub)
    total = 0.0
    prev = 0.0
    cum = 0.0
    for l, p in sub:
        cum += p
        total += l * (cum ** 32 - prev)
        prev = cum ** 32
    util = m / total
    caps[cap] = (m, total, util)
    print(f"{cap:>8}{m:>11.0f}{total:>17.0f}{util:>14.1%}{1.0 / util:>17.1f}x")

print()
print()
print("And the latency side, which is what static batching is usually defending.")
print("A request arriving mid-batch must wait for the batch to finish forming.")
print()
print(f"{'batch':>8}{'static wait ms':>17}{'continuous wait ms':>21}"
      f"{'saved':>10}")
print("-" * 58)
ARRIVAL_RATE = 22.0     # requests per second
wait = {}
for b in BATCHES:
    # Static: wait to fill the batch, then wait for the previous batch to drain.
    fill = (b - 1) / (2.0 * ARRIVAL_RATE) * 1000.0
    drain = static[b][0] * STEP_MS
    sw = fill + drain
    # Continuous: wait only for a slot to free, on average one sequence's length
    # divided by the batch size.
    cw = mean_len * STEP_MS / b
    wait[b] = (sw, cw)
    print(f"{b:>8}{sw:>17.0f}{cw:>21.0f}{sw - cw:>10.0f}")

print(f"""
The length distribution is the whole problem. The mean generation is
{mean_len:.0f} tokens and the longest is {LENGTHS[-1][0]} --
{LENGTHS[-1][0] / mean_len:.0f} times the mean, arriving on {LENGTHS[-1][1]:.0%} of
requests.

Under static batching that {LENGTHS[-1][1]:.0%} sets the cost of the whole batch. At
batch {32} the expected maximum length is {static[32][0]:.0f} tokens against a mean of
{mean_len:.0f}, so every one of {32} slots is held for {static[32][0]:.0f} steps while
the average sequence needs {mean_len:.0f} -- a utilisation of
**{static[32][3]:.1%}** (eq:static-batching-pays-for-the-longest).

**Nearly nine tenths of the capacity is computing padding.** Not idle -- computing,
on real silicon, at full power, producing nothing.

The throughput table converts that into the number that matters. Nominal throughput at
batch {32} is {eff_static[32][0]:.0f} tokens a second; effective is
{eff_static[32][1]:.0f}. The gap is exactly the utilisation, and it is why a
benchmark run on equal-length sequences reports numbers a real deployment never sees.

Continuous batching closes it by construction. When a sequence finishes, its slot goes
to a waiting request on the next step, so utilisation is {1.0:.0%} and throughput is
the nominal figure. At batch {32} that is **{cont[32] / eff_static[32][1]:.1f} times**
static batching, for the same hardware and the same requests.

The gap table shows the shape, and it is the uncomfortable one. E[max]/mean rises with
batch size -- {static[4][0] / mean_len:.2f} at batch {4},
{static[64][0] / mean_len:.2f} at batch {64} -- because the maximum of more draws
reaches further into the tail. So **static batching gets worse exactly as you batch
harder**, which is the same shape ch:inf-cpu-gpu found for KV traffic and
ch:sd-retrieval-agents found for fan-out.

At batch {64} static batching achieves {static[64][3]:.1%} utilisation and continuous
batching is {cont[64] / eff_static[64][1]:.1f} times better. The gain is not a
constant factor; it grows with the thing you want to increase.

The cap table is the other lever, and it is worth pricing because teams reach for it.
Capping generation at {280} tokens raises static utilisation from
{caps[4200][2]:.1%} to {caps[280][2]:.1%} -- most of the way to continuous batching's
{1.0:.0%} -- at the cost of truncating every request that needed more.

**Capping length is a way of buying batching efficiency with output quality**, and the
table says how much of each. It is the right trade for some surfaces and a silent
semantic failure for others, which is ch:sd-architecture's missing instrument
appearing in a serving configuration.

The last table addresses the argument static batching usually gets defended with:
predictable latency. It does not survive. At batch {32}, static batching makes a
request wait {wait[32][0]:.0f}ms -- {31 / (2.0 * ARRIVAL_RATE) * 1000.0:.0f}ms to fill
the batch plus {static[32][0] * STEP_MS:.0f}ms to drain it -- against
{wait[32][1]:.0f}ms for continuous batching, a factor of
{wait[32][0] / wait[32][1]:.0f}.

Static batching is worse on throughput AND worse on latency. **The only thing it is
better at is being simple to implement**, which is a real advantage and the reason it
persists, but it should be chosen knowingly rather than inherited.""")

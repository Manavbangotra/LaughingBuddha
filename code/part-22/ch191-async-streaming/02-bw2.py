# -*- coding: utf-8 -*-
# Extracted from: Chapter 191 — Queues, Asynchronous Processing, and Streaming
# Source: src/.../ch191-async-streaming.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Streaming hides latency until it does not, and it stops helping under load.

Streaming is the standard answer to slow generation: show tokens as they arrive and
the user starts reading immediately. The usual claim is that this converts a long
wait into a short one.

It converts a long wait into a short one ONLY while tokens arrive faster than the
user reads them. Below that rate the user catches up and waits at the reader's pace,
and the benefit collapses (eq:streaming-helps-until-the-reader-catches-up).

The sharp part is where that happens. Per-request token rate falls as concurrency
rises, because a shared accelerator divides its throughput among in-flight requests
(cite:kwon2023pagedattention). So streaming stops working precisely under load --
which is when the latency it was hiding actually appears.
"""
READ_RATE = 5.0        # tokens/sec a person reads, ~250 words per minute
TTFT_BASE = 0.35       # seconds to first token, unloaded
AGG_TOKENS = 900.0     # aggregate tokens/sec the server can emit across requests
LENGTHS = [80, 250, 600, 1400]


def rates(concurrency):
    """Per-request token rate and time-to-first-token at a concurrency level."""
    per = AGG_TOKENS / concurrency
    ttft = TTFT_BASE * (1 + 0.06 * concurrency)   # queueing ahead of first token
    return per, ttft


def perceived(length, concurrency, streaming):
    """Seconds the user spends waiting rather than reading."""
    per, ttft = rates(concurrency)
    if not streaming:
        # Nothing appears until the whole answer is generated.
        return ttft + length / per
    # Streaming: the user waits for the first token, then waits only for the
    # amount by which generation lags reading.
    starve = length * (1.0 / per - 1.0 / READ_RATE)
    return ttft + max(0.0, starve)


print("A shared accelerator emitting %.0f tokens/sec in aggregate. Per-request"
      % AGG_TOKENS)
print("rate falls as concurrency rises; a reader consumes %.0f tokens/sec."
      % READ_RATE)
print()
print(f"{'concurrency':>13}{'tokens/sec each':>18}{'time to first token':>21}"
      f"{'vs reader':>12}")
print("-" * 64)
CONC = [1, 8, 30, 90, 180, 360]
info = {}
for c in CONC:
    per, ttft = rates(c)
    info[c] = (per, ttft)
    print(f"{c:>13}{per:>18.1f}{ttft:>20.2f}s{per / READ_RATE:>11.1f}x")

print()
print()
print("Perceived wait for a 600-token answer -- the seconds the user spends")
print("looking at an incomplete screen rather than reading.")
print()
L = 600
print(f"{'concurrency':>13}{'no streaming':>15}{'streaming':>12}"
      f"{'saved':>10}{'saved %':>10}")
print("-" * 60)
save = {}
for c in CONC:
    a = perceived(L, c, False)
    b = perceived(L, c, True)
    save[c] = (a, b, a - b, (a - b) / a)
    print(f"{c:>13}{a:>14.2f}s{b:>11.2f}s{a - b:>9.2f}s{(a - b) / a:>10.0%}")

print()
print()
print("The same sweep across answer lengths. Streaming's benefit is a function of")
print("both length and load, and it disappears in the same corner from both.")
print()
print(f"{'answer length':>15}" + "".join(f"{c:>11}" for c in CONC))
print("-" * 81)
grid = {}
for length in LENGTHS:
    row = []
    for c in CONC:
        a = perceived(length, c, False)
        b = perceived(length, c, True)
        row.append((a - b) / a)
    grid[length] = row
    print(f"{length:>15}" + "".join(f"{v:>10.0%} " for v in row))

print()
print()
print("The threshold: the concurrency at which per-request generation drops to")
print("reading speed. Past it, streaming no longer hides anything.")
print()
CROSS = AGG_TOKENS / READ_RATE
print(f"aggregate throughput      {AGG_TOKENS:>8.0f} tokens/sec")
print(f"reader consumption rate   {READ_RATE:>8.0f} tokens/sec")
print(f"crossover concurrency     {CROSS:>8.0f} concurrent requests")
print()
print(f"{'concurrency':>13}{'tokens/sec each':>18}{'streaming still hides':>24}")
print("-" * 57)
for c in CONC:
    per, _ = info[c]
    verdict = "yes, fully" if per >= READ_RATE else "no, reader starves"
    print(f"{c:>13}{per:>18.1f}{verdict:>24}")

print()
print()
print("And what that does to a latency budget. A 4-second perceived-wait target,")
print("by answer length, with and without streaming:")
print()
TARGET = 4.0
print(f"{'answer length':>15}{'max concurrency (no stream)':>30}"
      f"{'max concurrency (stream)':>28}")
print("-" * 73)
cap = {}
for length in LENGTHS:
    best = {}
    for mode in (False, True):
        ok = 0
        for c in range(1, 2001):
            if perceived(length, c, mode) <= TARGET:
                ok = c
            else:
                break
        best[mode] = ok
    cap[length] = best
    print(f"{length:>15}{best[False]:>30}{best[True]:>28}")

print(f"""
The first table is the mechanism. At concurrency 1 each request gets
{info[1][0]:.0f} tokens/sec -- {info[1][0] / READ_RATE:.0f} times faster than a
person reads. At concurrency 360 each gets {info[360][0]:.1f} tokens/sec, which is
{info[360][0] / READ_RATE:.1f} times the reading rate. Somewhere between those the
reader stops being the bottleneck and the server starts being one.

The crossover is exact: {AGG_TOKENS:.0f} aggregate tokens/sec divided by a
{READ_RATE:.0f} tokens/sec reader gives **{CROSS:.0f} concurrent requests**
(eq:streaming-helps-until-the-reader-catches-up). Below it, streaming hides the
entire generation time. Above it, the reader has caught up and every additional
token is a token the user waits for.

What makes this worth a chapter is the SHAPE of the collapse. It is not a gradual
decay -- the saved-percentage column rises to {save[180][3]:.0%} at the crossover
and only then falls off, to {save[360][3]:.0%} at twice the crossover. A cliff, not
a slope, and nothing in the percentage warns you that you are approaching it.

Which sets up the trap. Read down the streaming column in absolute seconds:
{save[1][1]:.2f}s, {save[8][1]:.2f}s, {save[30][1]:.2f}s, {save[90][1]:.2f}s,
{save[180][1]:.2f}s. The user's wait has grown by a factor of
{save[180][1] / save[1][1]:.0f} while the headline saving improved from
{save[1][3]:.0%} to {save[180][3]:.0%}.

**The percentage saved gets better as the experience gets worse.** Streaming is
doing more work than ever -- it is hiding {save[180][2]:.0f} seconds at the
crossover -- and the user is still waiting {save[180][1]:.1f} seconds, because
time-to-first-token degrades with concurrency and streaming cannot hide the wait
before the first token.

That is this part's recurring failure, in a third form. ch:sd-architecture had an
availability graph that stayed green while answers went wrong; ch:sd-routing-caching
had a hit-rate dashboard that rose while total cost rose with it. Here a
streaming-effectiveness metric climbs to {save[180][3]:.0%} while the thing it
claims to measure gets {save[180][1] / save[1][1]:.0f} times worse. **Measure
perceived wait in seconds, never the percentage streaming saved.**

The length grid shows the benefit is real and large away from the cliff:
{grid[LENGTHS[-1]][2]:.0%} for a {LENGTHS[-1]}-token answer at concurrency 30
against {grid[LENGTHS[0]][2]:.0%} for an {LENGTHS[0]}-token one. Long answers are
where streaming earns its keep, and they are also where the queue cost is highest --
which is the tension the next paragraph turns into a warning.

The capacity table is why this belongs in system design rather than front-end work.
Under a {TARGET:.0f}-second perceived-wait target, a {L}-token answer supports
{cap[L][False]} concurrent requests without streaming and **{cap[L][True]}** with it
-- a factor of {cap[L][True] / max(cap[L][False], 1):.0f}. That is a genuine and
large capacity gain, and it is bounded by the crossover rather than by the
implementation: past {CROSS:.0f} concurrent requests no amount of front-end work
recovers it, and the remaining levers are the ones from this chapter's first
listing -- cut the variance, cap the length, or buy throughput.

One consequence to carry forward. Because streaming makes perceived wait nearly
independent of answer length below the crossover, it removes the user-facing reason
to prefer short answers -- while the queueing cost of length, which scales with the
second moment of service time (eq:variance-not-mean-drives-wait), is untouched.
**Streaming hides the cost of length from the user and not from the system.** A
product tuned on perceived latency alone drifts toward longer answers until the
queue notices, and the queue notices as a cliff.""")

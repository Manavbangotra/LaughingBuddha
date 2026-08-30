# -*- coding: utf-8 -*-
# Extracted from: Chapter 145 — Throughput versus Latency Engineering
# Source: src/.../ch145-throughput-latency.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Speculative decoding spends the idleness, so it competes with batching for it.

ch:q-gguf measured decode at batch 1 running at about one per cent of the
hardware's arithmetic balance point: the multiply-add units are idle almost all
the time, waiting for weights to arrive.

cite:leviathan2023speculative turns that idleness into speed. A cheap draft model
proposes k tokens; the expensive model verifies all of them in ONE forward pass,
because verifying k+1 positions reads the same weights as verifying one and only
costs more arithmetic -- which was free. The sampling rule makes the output
distribution provably identical to the target model's, so it is a latency
improvement with no quality cost, which is rare enough to be worth checking.

The part that is not usually stated is what happens as batch size rises, because
batching spends the same idleness (eq:speculation-spends-idleness).
"""
import numpy as np

P_TARGET = 70e9
P_DRAFT = 1.5e9
BW = 3.35e12
C = 990e12
KV_PER_TOK = 2 * 80 * 8 * 128 * 2
CTX = 4096
BITS = 4


def target_step(batch, positions=1):
    """One forward pass of the big model over `positions` token slots per
    sequence. The weight read is unchanged; only the arithmetic scales."""
    read = P_TARGET * BITS / 8.0 + KV_PER_TOK * CTX * batch
    return max(read / BW, 2.0 * P_TARGET * batch * positions / C)


def draft_step(batch):
    read = P_DRAFT * BITS / 8.0 + KV_PER_TOK * CTX * batch * (P_DRAFT / P_TARGET)
    return max(read / BW, 2.0 * P_DRAFT * batch / C)


def accepted(alpha, k):
    """Expected tokens accepted per verification round, including the bonus
    token the target model always contributes."""
    if alpha >= 1.0:
        return k + 1.0
    return (1.0 - alpha ** (k + 1)) / (1.0 - alpha)


def speculative(batch, k, alpha):
    t = k * draft_step(batch) + target_step(batch, k + 1)
    toks = accepted(alpha, k)
    return t / toks, batch * toks / t          # latency per token, throughput


def plain(batch):
    t = target_step(batch)
    return t, batch / t


ALPHA = 0.72
BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256)

print(f"70B target, 1.5B draft, acceptance rate {ALPHA:.0%}, {CTX:,} context.")
print("Speedup is speculative against plain decoding, at the same batch size.")
print()
print(f"{'batch':>7}" + "".join(f"{'k=' + str(k):>22}" for k in (2, 4, 8))
      + f"{'plain':>12}")
print(f"{'':>7}" + "".join(f"{'lat ms':>11}{'speedup':>11}" for _ in (2, 4, 8))
      + f"{'lat ms':>12}")
print("-" * 85)

rows = {}
for b in BATCHES:
    pl, ptp = plain(b)
    cells = []
    for k in (2, 4, 8):
        lat, tp = speculative(b, k, ALPHA)
        cells.append((lat * 1000, pl / lat))
        rows[(b, k)] = (lat * 1000, pl / lat, tp / ptp)
    print(f"{b:>7}"
          + "".join(f"{c[0]:>11.1f}{c[1]:>10.2f}x" for c in cells)
          + f"{pl*1000:>12.1f}")

print()
print()
print("Where does the idleness go? Arithmetic actually performed as a share of")
print("what the hardware could do in the same wall-clock time.")
print()
print(f"{'batch':>7}{'plain':>12}{'speculative k=4':>18}{'gap closed':>14}")
print("-" * 51)
util = {}
for b in (1, 8, 64, 256):
    tp, _ = plain(b)
    up = (2.0 * P_TARGET * b / C) / tp
    k4 = 4
    t = k4 * draft_step(b) + target_step(b, k4 + 1)
    us = (2.0 * P_TARGET * b * (k4 + 1) / C + k4 * 2.0 * P_DRAFT * b / C) / t
    util[b] = (up, us)
    print(f"{b:>7}{up:>11.1%}{us:>18.1%}{(us - up)/(1 - up):>13.0%}")

print()
print()
print("Acceptance rate is the other variable, and it is a property of the pair.")
print()
print(f"{'alpha':>8}" + "".join(f"{'k=' + str(k):>12}" for k in (2, 4, 8, 16))
      + f"{'best k':>10}")
print("-" * 60)
acc_rows = {}
for a in (0.5, 0.65, 0.8, 0.9):
    lats = []
    for k in (2, 4, 8, 16):
        lat, _ = speculative(1, k, a)
        lats.append(plain(1)[0] / lat)
    best = (2, 4, 8, 16)[int(np.argmax(lats))]
    acc_rows[a] = (lats, best)
    print(f"{a:>8.2f}" + "".join(f"{v:>11.2f}x" for v in lats)
          + f"{'k=' + str(best):>10}")

s1, s64, s256 = rows[(1, 4)], rows[(64, 4)], rows[(256, 4)]
print(f"""
Read the k=4 columns down the page and the effect shrinks as the batch grows.

At batch 1, speculating four tokens ahead gives {s1[1]:.2f}x lower per-token
latency. At batch 64, {s64[1]:.2f}x. At batch 256, {s256[1]:.2f}x.

The mechanism is in the utilisation table. At batch 1 plain decoding uses
{util[1][0]:.1%} of the machine's arithmetic capacity -- the number ch:q-gguf
measured, from the other direction. Speculation raises it to {util[1][1]:.1%},
and everything it gained came out of that gap. At batch 256 plain decoding is
already at {util[256][0]:.1%}, so there is far less gap left, and speculation's
extra arithmetic starts to cost real time rather than filling a hole
(eq:speculation-spends-idleness).

That is the finding, and it is not how the two techniques are usually presented.
**Speculative decoding and batching are substitutes, not complements.** They spend
the same resource -- the arithmetic that decode leaves idle -- and once one of
them has spent it, the other has nothing to work with.

Which resolves a common confusion. A local user at batch 1 measures a large
speculative speedup and reports it. A serving team at batch 128 enables the same
feature, measures almost nothing, and concludes the implementation is broken. Both
measurements are right, and the disagreement is structural rather than a
configuration error.

It also says where speculation belongs. It is a LATENCY technique for the
low-batch regime -- interactive single-user inference, or a serving tier with a
tight per-token SLO that forces small batches. It is not a throughput technique,
and at high batch it is not a technique at all.

The third table adds the variable that decides whether any of this works. The
acceptance rate is a property of the DRAFT AND TARGET PAIR, not of the algorithm,
and it enters as a power: accepting k tokens in a row has probability alpha^k, so
the expected yield saturates quickly.

At alpha={0.5:.2f}, speculating far ahead is pointless -- the best depth is
k={acc_rows[0.5][1]}, and going deeper wastes draft work on tokens that will be
rejected. At alpha={0.9:.2f} the best depth is k={acc_rows[0.9][1]} and the
speedup is {max(acc_rows[0.9][0]):.2f}x.

So the depth is not a tuning parameter to be swept blindly; it follows from the
measured acceptance rate, and the acceptance rate is what a draft model should be
selected on. A draft model that is slightly worse but agrees more often beats a
better one that agrees less, because agreement enters exponentially and quality
does not enter at all -- the target model's distribution is preserved exactly
either way.

Which is the property that makes this technique unusual and worth the chapter's
attention. Almost every other option in {{part:15}} trades quality for speed and
asks you to price the trade. Speculative decoding does not: the output
distribution is provably identical, so the only questions are whether you are in a
regime where it helps and whether you can find a draft model that agrees often
enough.""")

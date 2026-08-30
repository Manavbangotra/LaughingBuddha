# -*- coding: utf-8 -*-
# Extracted from: Chapter 145 — Throughput versus Latency Engineering
# Source: src/.../ch145-throughput-latency.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Latency and throughput are not two names for speed. They are in tension.

cite:pope2022inference established that low-latency generation and high-throughput
batch processing are distinct optimisation regimes with different answers. That is
easy to agree with and hard to act on until the trade is drawn.

This listing draws it. For one model on one machine, it sweeps batch size and
computes both quantities from the roofline, producing the Pareto frontier along
which every serving configuration sits (eq:latency-throughput-pareto). Then it
adds quantization, which does not move the frontier uniformly -- it moves one end
of it and leaves the other where it was.
"""
import numpy as np

P = 70e9                 # parameters
BW = 3.35e12             # bytes per second
C = 990e12               # FLOPs per second
DEQ_OPS = 4.0            # extra ops per weight to unpack a quantized value
KV_BYTES_PER_TOK = 2 * 80 * 8 * 128 * 2      # 80 layers, GQA-8, 16-bit


def step_s(batch, bits, ctx=4096):
    """One decode step: the slower of reading everything and computing."""
    read = P * bits / 8.0 + KV_BYTES_PER_TOK * ctx * batch
    t_mem = read / BW
    flops = 2.0 * P * batch + (DEQ_OPS * P if bits < 16 else 0.0)
    t_cmp = flops / C
    return max(t_mem, t_cmp), t_mem, t_cmp


BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)

print("70B GQA-8, 4k context. Per-token latency is what one user waits between")
print("tokens; throughput is what the machine produces in total.")
print()
print(f"{'batch':>7}" + "".join(f"{h:>13}" for h in
      ("lat 16-bit", "tput 16-bit", "lat 4-bit", "tput 4-bit"))
      + f"{'weights are':>13}{'bound by':>10}")
print(f"{'':>7}{'ms/token':>13}{'tok/s':>13}{'ms/token':>13}{'tok/s':>13}"
      f"{'of the read':>13}{'':>10}")
print("-" * 87)

rows = {}
for b in BATCHES:
    t16, m16, c16 = step_s(b, 16)
    t4, m4, c4 = step_s(b, 4)
    wshare = (P * 0.5) / (P * 0.5 + KV_BYTES_PER_TOK * 4096 * b)
    rows[b] = (t16 * 1000, b / t16, t4 * 1000, b / t4, wshare)
    print(f"{b:>7}{t16*1000:>13.1f}{b/t16:>13.0f}{t4*1000:>13.1f}"
          f"{b/t4:>13.0f}{wshare:>13.0%}"
          f"{('memory' if m4 > c4 else 'compute'):>10}")

ASYMPTOTE = BW / (KV_BYTES_PER_TOK * 4096)
print()
print(f"Asymptotic throughput as batch grows: {ASYMPTOTE:,.0f} tok/s.")
print("Note what is absent from that number: the model.")

print()
print()
print("What does a latency budget cost in throughput? Largest batch that meets")
print("a per-token latency target, and the throughput it yields.")
print()
print(f"{'target':>10}" + "".join(f"{h:>14}" for h in
      ("batch 16-bit", "tput 16-bit", "batch 4-bit", "tput 4-bit"))
      + f"{'4-bit gain':>13}")
print("-" * 79)

targets = {}
for tgt_ms in (25, 40, 60, 100, 200):
    best = {}
    for bits in (16, 4):
        ok = [b for b in BATCHES if step_s(b, bits)[0] * 1000 <= tgt_ms]
        b = max(ok) if ok else 0
        best[bits] = (b, b / step_s(b, bits)[0] if b else 0.0)
    targets[tgt_ms] = best
    g = best[4][1] / best[16][1] if best[16][1] else float("inf")
    print(f"{tgt_ms:>8} ms{best[16][0]:>14}{best[16][1]:>14.0f}"
          f"{best[4][0]:>14}{best[4][1]:>14.0f}"
          + (f"{g:>12.2f}x" if np.isfinite(g) else f"{'--':>13}"))

print()
print()
print("Context length moves the whole picture, because the cache joins the read.")
print()
print(f"{'context':>9}{'batch':>7}{'lat 4-bit':>12}{'tput 4-bit':>13}"
      f"{'KV share of':>14}")
print(f"{'':>9}{'':>7}{'ms/token':>12}{'tok/s':>13}{'bytes read':>14}")
print("-" * 55)
ctx_rows = {}
for ctx in (4096, 32768, 131072):
    for b in (1, 32):
        t, m, c = step_s(b, 4, ctx)
        kvshare = (KV_BYTES_PER_TOK * ctx * b) / (P * 0.5 + KV_BYTES_PER_TOK
                                                  * ctx * b)
        ctx_rows[(ctx, b)] = (t * 1000, b / t, kvshare)
        print(f"{ctx:>9,}{b:>7}{t*1000:>12.1f}{b/t:>13.0f}{kvshare:>13.0%}")

b1, b32, b512 = rows[1], rows[32], rows[512]
print(f"""
The first table is the tension, and it is worth being precise about why it exists
rather than treating it as a slogan.

At batch 1 the machine reads every weight to produce one token, so per-token
latency is as low as it will ever be -- {b1[2]:.1f} ms at 4 bits -- and throughput
is as low as it will ever be too: {b1[3]:.0f} tokens per second from hardware
capable of far more. At batch 512 the same weight read serves 512 sequences, so
throughput reaches {b512[3]:.0f} and each user now waits {b512[2]:.1f} ms between
tokens rather than {b1[2]:.1f}.

Nothing was optimised or mis-optimised between those rows. **Batching converts
latency into throughput at an exchange rate set by the hardware**
(eq:latency-throughput-pareto), and every serving configuration is a choice of
where on that curve to sit. "Make the system faster" is not a well-formed request
until someone names the axis.

Now the column that corrects something ch:q-gguf simplified.

That chapter computed a crossover batch at which decode stops being memory-bound
and becomes compute-bound, and it set the KV cache term to zero to do so. Put the
cache back and the crossover never arrives: every row here is memory-bound, at
every batch size tested.

The reason is in the weights-share column. At batch 1 the weights are
{b1[4]:.0%} of the bytes read per step. At batch 32, {b32[4]:.0%}. At batch 512,
{b512[4]:.0%}. **The cache read grows linearly with batch and the weight read does
not**, so raising the batch does not raise arithmetic intensity the way the
weights-only model predicted -- it just changes what you are reading
(eq:cache-caps-throughput).

Which gives the asymptote printed above the analysis, and it is the most useful
single number in the chapter. As batch grows, throughput approaches memory
bandwidth divided by the cache bytes per token per unit of context:
{ASYMPTOTE:,.0f} tokens per second here.

**The model's parameter count does not appear in that expression.** Maximum decode
throughput at a given context length is a property of the memory bandwidth and the
attention architecture, and the weights -- the thing this entire part has been
quantizing -- have dropped out of it entirely.

That reframes what weight quantization is for, and the 4-bit columns show it. At
batch 1, 4 bits gives {b1[2]:.1f} ms against 16 bits' {b1[0]:.1f} --
{b1[0]/b1[2]:.1f}x. At batch 512, {b512[2]:.1f} against {b512[0]:.1f}: a factor
of {b512[0]/b512[2]:.2f}. **Weight quantization is a low-batch technique, and it
stops working exactly where throughput optimisation starts.**

That is not a caveat. It is the shape of the practice, and it explains why
local-inference practitioners and serving engineers reach opposite conclusions
about the same technique while both measuring correctly.

The second table converts the trade into the form a product decision takes: a
latency budget, and what it costs.

Committing to a 40 ms per-token experience -- comfortable streaming speed --
allows a batch of {targets[40][16][0]} at 16 bits and {targets[40][4][0]} at 4
bits, for {targets[40][16][1]:.0f} and {targets[40][4][1]:.0f} tokens per second.
At 16 bits that target is unreachable at any batch, so the entry is zero: the
configuration cannot meet the SLO at all.

**Which is the most useful way to state quantization's value: it buys back batch
size at a fixed latency.** That is a far more actionable claim than "it makes the
model faster", and it is directly convertible into cost per token.

Relaxing the target to 200 ms allows batch {targets[200][4][0]} and
{targets[200][4][1]:.0f} tokens per second -- so the cost of a tight latency SLO
is a throughput multiple, computable in advance rather than discovered in
production.

The third table adds context length, which moves everything. At {131072:,} tokens
and batch 32 the cache is {ctx_rows[(131072, 32)][2]:.0%} of the bytes read, and
latency is {ctx_rows[(131072, 32)][0]:.1f} ms against
{ctx_rows[(4096, 32)][0]:.1f} ms at 4k.

**No weight format changes that.** The lever that does is ch:q-activation-kv's,
and this is the arithmetic showing that the two chapters are about the same
bottleneck at different context lengths.

So the shape to carry away is one curve with several ends, each raised by a
different technique. **Weight quantization lifts the low-batch end. Cache
quantization and grouped-query attention lift the long-context end. Better
kernels lift whatever compute-bound end remains.** None lifts all of it, and
knowing which end you are on decides which of them is worth anything to you.""")

# -*- coding: utf-8 -*-
# Extracted from: Chapter 198 — GPU Memory, CUDA, and the Roofline Model
# Source: src/.../ch198-gpu-memory.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""What actually fits, and why the answer is a frontier rather than a number.

ch:inf-cpu-gpu found batch size to be the lever that makes a GPU worth using. This
listing asks what bounds it, and the answer is memory capacity: weights are fixed, KV
cache scales with batch TIMES context, and the two trade against each other along a
hyperbola (eq:batch-times-context-is-the-budget).

The practical consequence is that "maximum batch size" and "maximum context length"
are not two configuration values. They are one curve, and a system configured with
both set independently will either waste memory or fail under a load it was told it
could handle.

This listing also measures what fragmentation costs, which is the gap
cite:kwon2023pagedattention closes.
"""
HBM = 80.0e9
LAYERS = 32
D_MODEL = 4096
N_KV_HEADS = 8            # grouped-query
HEAD_DIM = 128
PARAMS = 7.0e9

CONTEXTS = [1024, 4096, 16384, 65536, 262144]
QUANTS = [("bf16", 2.0), ("fp8", 1.0), ("int4", 0.5)]


def kv_per_token(bytes_per=2.0):
    return 2.0 * LAYERS * N_KV_HEADS * HEAD_DIM * bytes_per


def weights(bytes_per):
    return PARAMS * bytes_per


def activation_overhead():
    """Working buffers, CUDA context, allocator reserve. Roughly fixed."""
    return 3.5e9


print("An 80 GB device. Fixed costs first.")
print()
print(f"{'weight format':>16}{'weights GB':>13}{'overhead GB':>14}"
      f"{'left for KV GB':>17}")
print("-" * 60)
free = {}
for name, b in QUANTS:
    w = weights(b)
    f = HBM - w - activation_overhead()
    free[name] = f
    print(f"{name:>16}{w / 1e9:>13.1f}{activation_overhead() / 1e9:>14.1f}"
          f"{f / 1e9:>17.1f}")

print()
print("KV cache per token, bf16 cache, %d KV heads: %.3f MB"
      % (N_KV_HEADS, kv_per_token() / 1e6))

print()
print()
print("The frontier: maximum batch size by context length, for each weight format.")
print("This is batch times context held to a constant.")
print()
print(f"{'context':>10}" + "".join(f"{n:>14}" for n, _ in QUANTS))
print("-" * 52)
front = {}
for c in CONTEXTS:
    row = []
    for name, b in QUANTS:
        m = int(free[name] / (c * kv_per_token()))
        row.append(m)
    front[c] = row
    print(f"{c:>10}" + "".join(f"{v:>14}" for v in row))
print()
print("(maximum concurrent sequences)")

print()
print()
print("The same as a product, which is the quantity actually conserved.")
print()
print(f"{'weight format':>16}{'batch x context':>18}{'vs bf16':>10}")
print("-" * 46)
prod = {}
for name, b in QUANTS:
    p = free[name] / kv_per_token()
    prod[name] = p
    print(f"{name:>16}{p:>18.0f}{p / prod['bf16']:>9.1f}x")

print()
print("Any (batch, context) pair whose product is under that number fits.")

print()
print()
print("Quantising the CACHE as well, which is a separate decision from quantising")
print("the weights.")
print()
print(f"{'weights':>10}{'KV cache':>11}{'KV MB/token':>14}"
      f"{'batch x context':>18}{'vs bf16/bf16':>15}")
print("-" * 70)
base = None
combo = {}
for wn, wb in QUANTS:
    for kn, kb in QUANTS:
        f = HBM - weights(wb) - activation_overhead()
        per = kv_per_token(kb)
        p = f / per
        if base is None:
            base = p
        combo[(wn, kn)] = p
        print(f"{wn:>10}{kn:>11}{per / 1e6:>14.3f}{p:>18.0f}{p / base:>14.1f}x")

print()
print()
print("What fragmentation costs. Without paging, a sequence reserves its MAXIMUM")
print("possible length up front, because the allocation must be contiguous.")
print()
MAXLEN = 8192
print(f"reserved length per sequence: {MAXLEN}")
print()
print(f"{'actual mean length':>20}{'utilisation':>14}{'effective batch':>18}"
      f"{'paged batch':>14}{'gain':>9}")
print("-" * 76)
frag = {}
for actual in (180, 640, 2100, 5400, 8192):
    naive_batch = int(free["bf16"] / (MAXLEN * kv_per_token()))
    paged_batch = int(free["bf16"] / (actual * kv_per_token()))
    util = actual / float(MAXLEN)
    frag[actual] = (util, naive_batch, paged_batch)
    print(f"{actual:>20}{util:>14.1%}{naive_batch:>18}{paged_batch:>14}"
          f"{paged_batch / float(naive_batch):>8.1f}x")

print()
print()
print("And the cost of getting the frontier wrong: configuring a batch and a")
print("context independently, then meeting a request mix that uses both.")
print()
CFG_BATCH = 64
CFG_CTX = 16384
need = CFG_BATCH * CFG_CTX * kv_per_token()
print(f"configured: batch {CFG_BATCH}, context {CFG_CTX}")
print(f"KV needed if both are used at once: {need / 1e9:.1f} GB")
print(f"available at bf16 weights:          {free['bf16'] / 1e9:.1f} GB")
print(f"shortfall:                          {(need - free['bf16']) / 1e9:.1f} GB")
print()
print(f"{'weight format':>16}{'KV available GB':>18}{'KV needed GB':>15}"
      f"{'fits':>8}")
print("-" * 58)
for name, b in QUANTS:
    print(f"{name:>16}{free[name] / 1e9:>18.1f}{need / 1e9:>15.1f}"
          f"{('yes' if free[name] >= need else 'no'):>8}")

print(f"""
The fixed-cost table sets up everything else. At bf16 weights, a
{HBM / 1e9:.0f} GB device has {free['bf16'] / 1e9:.1f} GB left for KV cache after
weights and working buffers. At int4 it has {free['int4'] / 1e9:.1f} GB --
{free['int4'] / free['bf16']:.1f} times more.

That ratio is the first thing worth noticing, because it is much smaller than the
weight saving suggests. Shrinking a {weights(2.0) / 1e9:.0f} GB model to
{weights(0.5) / 1e9:.1f} GB is a {weights(2.0) / weights(0.5):.0f}-fold reduction in
weights and only a {free['int4'] / free['bf16']:.1f}-fold gain in serving capacity,
because the weights were never the thing consuming most of the device.

**On an {HBM / 1e9:.0f} GB card, a {PARAMS / 1e9:.0f}B model's weights are
{weights(2.0) / HBM:.0%} of memory and the cache is most of the rest.** Quantising
weights is therefore a way to fit a model that did not fit, and only marginally a way
to serve more of one that did -- a distinction the memory-footprint framing loses, and
one the cache-quantisation table below makes sharp.

The frontier table is the chapter's main point. At {CONTEXTS[0]} tokens of context a
bf16 deployment serves {front[CONTEXTS[0]][0]} concurrent sequences; at
{CONTEXTS[3]} tokens it serves {front[CONTEXTS[3]][0]}
(eq:batch-times-context-is-the-budget).

**These are not two settings. They are one curve**, and the conserved quantity is the
product: {prod['bf16']:.0f} token-slots at bf16, {prod['int4']:.0f} at int4. Any
(batch, context) pair whose product is under that number fits, and any pair over it
does not, regardless of how the two numbers were arrived at.

The cache-quantisation table separates a decision that is usually made once for both.
Quantising weights to int4 while leaving the cache at bf16 gives
{combo[('int4', 'bf16')] / base:.1f}x the token-slots. Quantising the cache to int4
while leaving weights at bf16 gives {combo[('bf16', 'int4')] / base:.1f}x. Doing both
gives {combo[('int4', 'int4')] / base:.1f}x.

**The cache is the larger lever**, and it is the one usually left alone -- partly
because cache quantisation has a quality cost that is harder to measure than weight
quantisation's, and partly because the tooling makes weights the obvious knob.
ch:q-activation-kv has the quality side; the capacity side is this table.

The fragmentation table is what cite:kwon2023pagedattention addresses. Without
paging, a sequence must reserve its maximum possible length contiguously, so a
deployment allowing {MAXLEN}-token contexts reserves {MAXLEN} tokens for every
sequence -- even the ones that turn out to be {180} tokens long.

At a mean actual length of {640}, that is {frag[640][0]:.1%} utilisation and an
effective batch of {frag[640][1]} against a paged batch of {frag[640][2]} --
**{frag[640][2] / float(frag[640][1]):.1f} times the concurrency for the same
memory**.

The gain is exactly the inverse of the utilisation, which makes it predictable: a
deployment whose requests use a tenth of their allowed context gets roughly ten times
the batch from paging. **Paging does not make memory bigger; it stops a length
distribution from being charged at its maximum.**

The last table is the failure this all exists to prevent. Configuring batch
{CFG_BATCH} and context {CFG_CTX} independently looks reasonable -- each is a
defensible number -- and together they require {need / 1e9:.1f} GB of cache against
{free['bf16'] / 1e9:.1f} GB available. The configuration is
{need / free['bf16']:.1f} times oversubscribed.

It will also work perfectly in testing, because tests rarely produce {CFG_BATCH}
simultaneous {CFG_CTX}-token requests. **The failure arrives as an
out-of-memory error under a load the configuration explicitly permitted**, which is
the most confusing shape an incident can have: nothing exceeded a limit, and the
system still ran out.

The fix is to configure the product and derive the pair, which is one line of
arithmetic and almost never done.""")

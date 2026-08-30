# -*- coding: utf-8 -*-
# Extracted from: Chapter 197 — CPU and GPU Inference Fundamentals
# Source: src/.../ch197-cpu-gpu.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Where the time goes inside a forward pass, and why the answer changes with context.

The previous listing treated a forward pass as one lump of arithmetic against one
lump of weight traffic. Inside it, the work divides into operations whose cost scales
with parameters and operations whose cost scales with SEQUENCE LENGTH, and those two
groups behave completely differently as context grows.

Attention over the KV cache reads state proportional to the tokens already generated.
So at short context it is a rounding error and at long context it dominates, and the
crossover is a property of the model shape rather than the hardware
(eq:kv-traffic-overtakes-weights).

This listing finds the crossover, and shows why the standard optimisations stop
working past it.
"""
# A 7B-class decoder. Shapes chosen so parameter count lands near 7e9.
LAYERS = 32
D_MODEL = 4096
D_FF = 11008
N_HEADS = 32
N_KV_HEADS = 32          # multi-head; grouped-query changes this
HEAD_DIM = D_MODEL // N_HEADS
BYTES = 2.0

CONTEXTS = [128, 512, 2048, 8192, 32768, 131072]
BANDWIDTH = 3.35e12      # bytes/s, current datacentre GPU


def weight_bytes():
    """Bytes of weights read per forward pass, per layer times layers."""
    attn = 4.0 * D_MODEL * D_MODEL           # q, k, v, o projections
    ffn = 3.0 * D_MODEL * D_FF               # gated MLP: up, gate, down
    return LAYERS * (attn + ffn) * BYTES


def kv_bytes(ctx, kv_heads=N_KV_HEADS):
    """Bytes of KV cache read to attend over `ctx` prior tokens, one sequence."""
    per_token_per_layer = 2.0 * kv_heads * HEAD_DIM * BYTES   # K and V
    return LAYERS * ctx * per_token_per_layer


W = weight_bytes()
print("A %d-layer model, d_model %d, d_ff %d, %d heads." % (LAYERS, D_MODEL, D_FF,
                                                            N_HEADS))
print("Weights read per forward pass: %.2f GB" % (W / 1e9))
print("KV cache per token: %.2f MB" % (kv_bytes(1) / 1e6))
print()
print()
print("Memory traffic for one decode step, by context length. Batch 1.")
print()
print(f"{'context':>10}{'weight GB':>12}{'KV GB':>10}{'KV share':>11}"
      f"{'total GB':>11}{'step ms':>10}")
print("-" * 64)
tab = {}
for c in CONTEXTS:
    k = kv_bytes(c)
    tot = W + k
    tab[c] = (k, tot, k / tot, tot / BANDWIDTH * 1000.0)
    print(f"{c:>10}{W / 1e9:>12.2f}{k / 1e9:>10.2f}{k / tot:>11.1%}"
          f"{tot / 1e9:>11.2f}{tot / BANDWIDTH * 1000.0:>10.2f}")

print()
print()
print("The same at batch 32. Weights are read ONCE for the whole batch; KV cache")
print("is per sequence, so it scales with the batch and the weights do not.")
print()
B = 32
print(f"{'context':>10}{'weight GB':>12}{'KV GB':>10}{'KV share':>11}"
      f"{'total GB':>11}{'tokens/s':>11}")
print("-" * 65)
batched = {}
for c in CONTEXTS:
    k = kv_bytes(c) * B
    tot = W + k
    t = tot / BANDWIDTH
    batched[c] = (k, tot, k / tot, B / t)
    print(f"{c:>10}{W / 1e9:>12.2f}{k / 1e9:>10.2f}{k / tot:>11.1%}"
          f"{tot / 1e9:>11.2f}{B / t:>11.0f}")

print()
print()
print("Where the crossover sits: the context at which KV traffic equals weight")
print("traffic, by batch size.")
print()
print(f"{'batch':>8}{'crossover context':>20}{'KV GB there':>14}")
print("-" * 42)
cross = {}
for b in (1, 4, 16, 32, 64, 128):
    # W == b * kv_bytes(c)  ->  c = W / (b * kv_bytes(1))
    c = W / (b * kv_bytes(1))
    cross[b] = c
    print(f"{b:>8}{c:>20.0f}{W / 1e9:>14.2f}")

print()
print()
print("What batching buys, by context. This is the number that decides whether")
print("the previous listing's advice still applies.")
print()
print(f"{'context':>10}" + "".join(f"{('b=%d' % b):>12}" for b in (1, 8, 32, 128)))
print("-" * 58)
gain = {}
for c in CONTEXTS:
    row = []
    for b in (1, 8, 32, 128):
        tot = W + kv_bytes(c) * b
        row.append(b / (tot / BANDWIDTH))
    gain[c] = row
    print(f"{c:>10}" + "".join(f"{v:>12.0f}" for v in row))
print()
print("(decode tokens per second across the whole batch)")

print()
print()
print("Batching efficiency: throughput at batch b divided by b times throughput")
print("at batch 1. Perfect batching is 100%.")
print()
print(f"{'context':>10}" + "".join(f"{('b=%d' % b):>12}" for b in (8, 32, 128)))
print("-" * 46)
eff = {}
for c in CONTEXTS:
    base = 1.0 / ((W + kv_bytes(c)) / BANDWIDTH)
    row = []
    for i, b in enumerate((8, 32, 128)):
        tot = W + kv_bytes(c) * b
        row.append((b / (tot / BANDWIDTH)) / (b * base))
    eff[c] = row
    print(f"{c:>10}" + "".join(f"{v:>11.1%}" for v in row))

print()
print()
print("And what grouped-query attention does to it: fewer KV heads means less")
print("cache to read, which moves the crossover.")
print()
print(f"{'KV heads':>10}{'KV MB/token':>14}{'crossover at b=32':>20}"
      f"{'b=32 eff at 32k':>18}")
print("-" * 62)
gqa = {}
for kvh in (32, 8, 4, 1):
    per_tok = kv_bytes(1, kvh)
    c = W / (32 * per_tok)
    tot = W + kv_bytes(32768, kvh) * 32
    base = 1.0 / ((W + kv_bytes(32768, kvh)) / BANDWIDTH)
    e = (32 / (tot / BANDWIDTH)) / (32 * base)
    gqa[kvh] = (per_tok, c, e)
    print(f"{kvh:>10}{per_tok / 1e6:>14.3f}{c:>20.0f}{e:>18.1%}")

print(f"""
The first table is the shape of the problem. At {CONTEXTS[0]} tokens of context the
KV cache is {tab[CONTEXTS[0]][2]:.1%} of memory traffic and the weights are
everything. At {CONTEXTS[-1]} tokens the KV cache is {tab[CONTEXTS[-1]][2]:.1%} and
the weights are the rounding error (eq:kv-traffic-overtakes-weights).

The step time goes from {tab[CONTEXTS[0]][3]:.2f}ms to
{tab[CONTEXTS[-1]][3]:.2f}ms -- a factor of
{tab[CONTEXTS[-1]][3] / tab[CONTEXTS[0]][3]:.0f} -- for the same model on the same
hardware generating the same single token. **Context length is a latency parameter,
not merely a capability one.**

The batched table is where it turns into an architectural constraint. Weights are
read once per pass no matter how large the batch; **KV cache is read per sequence**,
so it scales with the batch. That means the thing batching amortises is exactly the
part that stops mattering as context grows.

The crossover table gives the boundary directly. At batch 1 the KV cache overtakes
the weights at {cross[1]:.0f} tokens of context. At batch {32} it overtakes at
{cross[32]:.0f} tokens. At batch {128}, {cross[128]:.0f} tokens.

**Every increase in batch size moves the crossover closer**, which is the opposite of
convenient: the harder you batch, the sooner batching stops helping.

The efficiency table prices that. At {CONTEXTS[0]} tokens of context, batch {8}
achieves {eff[CONTEXTS[0]][0]:.1%} of ideal batching -- close to free throughput, and
the regime ch:inf-cpu-gpu's first listing was implicitly describing, since it ignored
the cache entirely.

Hold the batch at {8} and grow the context: {eff[CONTEXTS[2]][0]:.1%} at
{CONTEXTS[2]} tokens, {eff[CONTEXTS[4]][0]:.1%} at {CONTEXTS[4]}. Push to batch
{128} at {CONTEXTS[4]} tokens and it is {eff[CONTEXTS[4]][2]:.1%} -- a hundred and
twenty-eight sequences of memory traffic buying
{gain[CONTEXTS[4]][3] / gain[CONTEXTS[4]][0]:.1f} times the throughput of one.

So the standard advice -- batch harder -- has a domain of validity, and the domain is
**short context**. Past the crossover, adding sequences to a batch adds proportional
memory traffic and buys almost nothing, because each sequence brings its own cache
and no work is shared.

This is the honest form of a claim the serving literature sometimes makes loosely.
cite:kwon2023pagedattention's paged cache and continuous batching raise achievable
batch size by removing fragmentation and letting sequences join and leave freely, and
that is a large real gain. What it cannot do is make KV traffic shared, because it
genuinely is not.

The last table is the intervention that does address it. Grouped-query attention
reduces the number of distinct key-value heads, and the cache shrinks in proportion:
{gqa[32][0] / 1e6:.3f} MB per token at {32} KV heads against
{gqa[4][0] / 1e6:.3f} MB at {4}. That moves the batch-32 crossover from
{gqa[32][1]:.0f} to {gqa[4][1]:.0f} tokens, and lifts batch-32 efficiency at
{32768} tokens of context from {gqa[32][2]:.1%} to {gqa[4][2]:.1%}.

**Grouped-query attention is a serving-throughput decision made at training time**,
and the table is why every model shipped for long context has adopted it. The
quality cost is small and paid once; the traffic saving is paid on every token of
every request forever.""")

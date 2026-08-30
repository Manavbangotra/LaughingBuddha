# -*- coding: utf-8 -*-
# Extracted from: Chapter 142 — Activation and KV-Cache Quantization
# Source: src/.../ch142-activation-kv.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The KV cache is the memory term that grows with traffic, and three things
compete to shrink it.

Weights are a fixed cost: a 70B model occupies the same bytes whether it serves
one request or a thousand. The KV cache is not. It grows with context length, and
it grows again with every concurrent sequence, so past a certain load it is the
term that decides how many requests fit (eq:kv-scales-with-traffic).

Three levers act on it, and they are not the same size. This listing prices all
three against each other: the architectural one (how many KV heads the model has),
the numerical one (how many bits each element takes), and the one that is not
about the tensor at all -- how the memory is ALLOCATED, which
cite:kwon2023pagedattention found was worth more than either.
"""
import numpy as np

# layers, model dim, query heads, head dim -- shapes in the usual proportions.
MODELS = {
    "7B":  dict(P=7e9,  L=32, heads=32, hdim=128),
    "70B": dict(P=70e9, L=80, heads=64, hdim=128),
}


def kv_bytes(m, ctx, batch, bits, kv_heads=None):
    """Two tensors, per layer, per KV head, per token, per sequence."""
    kvh = kv_heads if kv_heads else m["heads"]
    return 2 * m["L"] * kvh * m["hdim"] * ctx * batch * bits / 8.0


def w_bytes(m, bits):
    return m["P"] * bits / 8.0


def gb(x):
    return x / 1e9


print("When does the KV cache overtake the weights? 70B, weights at 4 bits,")
print("cache at 16 bits, full multi-head attention.")
print()
m = MODELS["70B"]
print(f"{'context':>10}" + "".join(f"{'batch ' + str(b):>12}"
                                   for b in (1, 8, 32, 128)))
print(f"{'':>10}{'KV cache GB (weights are ' + f'{gb(w_bytes(m, 4)):.0f} GB)':>48}")
print("-" * 60)
for ctx in (2048, 8192, 32768, 131072):
    row = [gb(kv_bytes(m, ctx, b, 16)) for b in (1, 8, 32, 128)]
    print(f"{ctx:>10,}" + "".join(f"{v:>12.1f}" for v in row))

print()
print()
print("Three levers on the same quantity. 70B, 8k context, batch 32.")
print()
print(f"{'configuration':>34}{'KV cache':>12}{'vs baseline':>14}")
print("-" * 60)
base = kv_bytes(m, 8192, 32, 16)
levers = [
    ("baseline: MHA, 16-bit cache", kv_bytes(m, 8192, 32, 16)),
    ("quantize the cache to 8 bits", kv_bytes(m, 8192, 32, 8)),
    ("quantize the cache to 4 bits", kv_bytes(m, 8192, 32, 4)),
    ("quantize the cache to 2 bits", kv_bytes(m, 8192, 32, 2)),
    ("GQA with 8 KV heads, 16-bit", kv_bytes(m, 8192, 32, 16, 8)),
    ("GQA with 8 KV heads, 4-bit", kv_bytes(m, 8192, 32, 4, 8)),
    ("MQA with 1 KV head, 4-bit", kv_bytes(m, 8192, 32, 4, 1)),
]
for name, v in levers:
    print(f"{name:>34}{gb(v):>10.1f} GB{base/v:>13.0f}x")

print()
print()
print("The third lever: allocation. Reserving the maximum context per sequence")
print("against allocating pages as the sequence actually grows.")
print()
print(f"{'workload':>28}{'mean len':>10}{'max len':>10}{'reserved':>11}"
      f"{'used':>9}{'waste':>9}")
print("-" * 77)

rng = np.random.default_rng(271)
WORKLOADS = [
    ("uniform 1k-2k", lambda n: rng.integers(1024, 2048, n), 2048),
    ("heavy-tailed, 8k cap", lambda n: np.clip(
        rng.lognormal(6.2, 1.0, n).astype(int), 32, 8192), 8192),
    ("heavy-tailed, 32k cap", lambda n: np.clip(
        rng.lognormal(6.2, 1.3, n).astype(int), 32, 32768), 32768),
    ("chat: short with outliers", lambda n: np.where(
        rng.random(n) < 0.05, rng.integers(8000, 32768, n),
        rng.integers(200, 1500, n)), 32768),
]
alloc = {}
for name, gen, cap in WORKLOADS:
    lens = gen(4000)
    reserved = cap * len(lens)
    used = lens.sum()
    alloc[name] = (float(lens.mean()), cap, used / reserved)
    print(f"{name:>28}{lens.mean():>10.0f}{cap:>10,}"
          f"{reserved/1e6:>9.1f}M{used/1e6:>8.1f}M"
          f"{1 - used/reserved:>9.1%}")

print()
print()
print("What each lever buys in concurrent sequences, on one 80 GB card.")
print()
print(f"{'configuration':>40}{'KV budget':>12}{'sequences':>12}")
print("-" * 64)
BUDGET = 80e9 - w_bytes(m, 4)
per_seq = {}
for label, bits, kvh, eff in [
    ("MHA, 16-bit, reserve max context", 16, None, alloc["heavy-tailed, 8k cap"][2]),
    ("MHA, 16-bit, paged", 16, None, 1.0),
    ("GQA-8, 16-bit, paged", 16, 8, 1.0),
    ("GQA-8, 4-bit, paged", 4, 8, 1.0),
]:
    b = kv_bytes(m, 8192, 1, bits, kvh) / eff
    per_seq[label] = BUDGET / b
    print(f"{label:>40}{gb(BUDGET):>10.0f} GB{BUDGET/b:>12.1f}")

q4 = base / kv_bytes(m, 8192, 32, 4)
g8 = base / kv_bytes(m, 8192, 32, 16, 8)
both = base / kv_bytes(m, 8192, 32, 4, 8)
ht = alloc["heavy-tailed, 32k cap"]
chat = alloc["chat: short with outliers"]
print(f"""
The first table is why this chapter exists. A 70B model at 4 bits occupies
{gb(w_bytes(m, 4)):.0f} GB of weights, and that number never changes. At 8k
context and batch 32 the KV cache is {gb(kv_bytes(m, 8192, 32, 16)):.0f} GB --
larger than the model. At 128k context and batch 128 it is
{gb(kv_bytes(m, 131072, 128, 16)):.0f} GB, which is not a number any single
machine has.

That is the structural point. Weights are a fixed cost paid once; the cache is a
variable cost paid per concurrent token. Every capacity question in serving is
really a question about the second (eq:kv-scales-with-traffic), and the model's
parameter count -- the number in its name -- barely enters.

The second table puts the three levers side by side, and the ranking is not the
one the quantization literature would suggest.

Quantizing the cache from 16 bits to 4 is worth {q4:.0f}x. Switching from
multi-head attention to grouped-query attention with 8 KV heads is worth
{g8:.0f}x, and it is an architectural decision made when the model was trained,
not something you apply at deployment. Doing both is worth {both:.0f}x.

So the largest single lever on KV memory is one you do not control at serving
time. That is worth knowing before optimising the one you do -- and it explains
why grouped-query attention became universal so quickly. It bought more than any
amount of numerical cleverness could, at a small and measurable quality cost, and
it composes with quantization rather than competing.

The third table is the lever that is not about the tensor at all, and it is
cite:kwon2023pagedattention's contribution.

A KV cache has to live somewhere contiguous for the attention kernel to read it,
so the obvious implementation reserves the maximum supported context for every
sequence when it starts. Look at what that costs on realistic length
distributions. A heavy-tailed workload with a 32k cap has a mean length of
{ht[0]:.0f} tokens and wastes {1-ht[2]:.0%} of what it reserved. A chat workload
where 5% of conversations are long wastes {1-chat[2]:.0%}.

Those are not inefficiencies at the margin. On the chat row, more than nine tenths
of the most contested memory in the system is reserved for tokens that will never
exist, because a small minority of requests might have needed it.

Paging fixes it the way operating systems fixed the same problem: allocate the
cache in fixed-size blocks that need not be contiguous, and hand out blocks as the
sequence actually grows. The attention kernel takes an indirection through a block
table. Nothing about the tensor's contents changes.

The last table puts all of it together in the unit that matters -- how many
concurrent 8k-context sequences fit alongside the weights on one 80 GB card.

Reserving the maximum with a 16-bit MHA cache fits
{per_seq['MHA, 16-bit, reserve max context']:.1f}. Not one sequence: the
reservation for a single request exceeds the entire remaining budget, which is
the arithmetic form of "this configuration does not work". Paging the same cache
fits {per_seq['MHA, 16-bit, paged']:.1f}. Adding grouped-query attention,
{per_seq['GQA-8, 16-bit, paged']:.1f}. Adding 4-bit quantization on top,
{per_seq['GQA-8, 4-bit, paged']:.1f}.

Read that sequence and the ordering of effort follows. Fix the allocator, then
choose an architecture with fewer KV heads, then quantize. The first two are
larger than the third, and only the third is what "KV cache quantization" refers
to.

None of which makes the quantization worthless -- it is the last multiplier on the
stack and it composes with everything before it. It makes the point that a
throughput problem attributed to the cache's PRECISION is usually a problem with
its ALLOCATION, and that the cheapest fix is not the one this part is about.""")

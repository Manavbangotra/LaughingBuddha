# -*- coding: utf-8 -*-
# Extracted from: Chapter 143 — Memory Math: Will This Model Fit?
# Source: src/.../ch143-memory-math.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Will this model fit? Every term, and which one is binding.

"A 70B model at 4 bits is 35 GB, so it fits on a 48 GB card" is the calculation
everybody does and it is wrong more often than it is right, because the weights
are only one of five terms and usually not the one that binds.

This listing computes all of them -- weights, KV cache, activations, framework
overhead and allocator waste -- across a grid of context lengths and batch sizes,
and reports which term is largest at each point (eq:inference-budget). The useful
output is not a number but a NAME: the thing to fix.
"""
import numpy as np

MODELS = {
    "7B  MHA":   dict(P=7e9,  L=32, h=32, hkv=32, d=128, dm=4096),
    "8B  GQA-8": dict(P=8e9,  L=32, h=32, hkv=8,  d=128, dm=4096),
    "70B GQA-8": dict(P=70e9, L=80, h=64, hkv=8,  d=128, dm=8192),
}

FRAMEWORK_GB = 1.2          # CUDA context, kernels, allocator metadata
ACT_TENSORS = 6             # live intermediates per layer during decode


def weights(m, wbits):
    return m["P"] * wbits / 8.0


def kv(m, ctx, batch, kvbits, util=1.0):
    return (2 * m["L"] * m["hkv"] * m["d"] * ctx * batch * kvbits / 8.0) / util


def activations(m, batch, tokens, abits=2):
    """Live intermediates. During decode `tokens` is 1 per sequence; during
    prefill it is the whole prompt, which is what makes prefill the peak."""
    return ACT_TENSORS * batch * tokens * m["dm"] * abits


def budget(m, ctx, batch, wbits=4, kvbits=16, util=1.0, tokens=1):
    w = weights(m, wbits)
    k = kv(m, ctx, batch, kvbits, util)
    a = activations(m, batch, tokens)
    f = FRAMEWORK_GB * 1e9
    return dict(weights=w, kv=k, activations=a, framework=f)


def gb(x):
    return x / 1e9


CARDS = {"24 GB": 24e9, "48 GB": 48e9, "80 GB": 80e9, "2x80 GB": 160e9}

print("Total inference memory and the BINDING term. Weights 4-bit, cache 16-bit,")
print("paged allocator (no waste), decode only.")
print()
print(f"{'model':>11}{'context':>9}{'batch':>7}" + "".join(f"{c:>10}"
      for c in ("weights", "KV cache", "activ.", "total"))
      + f"{'binding':>12}{'fits on':>10}")
print("-" * 89)

for name, m in MODELS.items():
    for ctx, batch in ((4096, 1), (4096, 32), (32768, 1), (32768, 16),
                       (131072, 4)):
        b = budget(m, ctx, batch)
        tot = sum(b.values())
        binding = max(b, key=b.get)
        card = next((c for c, v in CARDS.items() if tot < v), "too big")
        print(f"{name:>11}{ctx:>9,}{batch:>7}"
              f"{gb(b['weights']):>10.1f}{gb(b['kv']):>10.1f}"
              f"{gb(b['activations']):>10.2f}{gb(tot):>10.1f}"
              f"{binding:>12}{card:>10}")
    print()

print("The same grid, with the levers applied: 4-bit cache and paged allocation.")
print()
print(f"{'model':>11}{'context':>9}{'batch':>7}{'before':>10}{'after':>10}"
      f"{'binding':>12}{'fits on':>10}")
print("-" * 69)
for name, m in MODELS.items():
    for ctx, batch in ((32768, 16), (131072, 4)):
        b0 = budget(m, ctx, batch)
        b1 = budget(m, ctx, batch, kvbits=4)
        t0, t1 = sum(b0.values()), sum(b1.values())
        binding = max(b1, key=b1.get)
        card = next((c for c, v in CARDS.items() if t1 < v), "too big")
        print(f"{name:>11}{ctx:>9,}{batch:>7}{gb(t0):>10.1f}{gb(t1):>10.1f}"
              f"{binding:>12}{card:>10}")

print()
print()
print("How many concurrent sequences fit? Solving the budget for batch.")
print()
print(f"{'model':>11}{'card':>9}{'context':>9}" + "".join(f"{c:>14}" for c in
      ("16-bit cache", "4-bit cache")))
print("-" * 68)


def max_batch(m, cap, ctx, wbits=4, kvbits=16):
    free = cap - weights(m, wbits) - FRAMEWORK_GB * 1e9
    if free <= 0:
        return 0
    per = kv(m, ctx, 1, kvbits) + activations(m, 1, 1)
    return int(free / per)


caps = {}
for name, m in MODELS.items():
    for cardname, cap in (("48 GB", 48e9), ("80 GB", 80e9)):
        for ctx in (8192, 32768):
            a = max_batch(m, cap, ctx, kvbits=16)
            b = max_batch(m, cap, ctx, kvbits=4)
            caps[(name, cardname, ctx)] = a
            print(f"{name:>11}{cardname:>9}{ctx:>9,}{a:>14,}{b:>14,}")

m70 = MODELS["70B GQA-8"]
m7 = MODELS["7B  MHA"]
print(f"""
Read the first table's binding column before any of the numbers, because it is
the output that changes what you do.

At 4k context and batch 1 the weights bind for every model, and that is the case
the folklore describes -- the one where "the model is 35 GB" is the whole
calculation. It is also the case that almost never occurs in production, because
batch 1 at short context is a demo rather than a deployment.

Move one row down and the answer changes. At 4k context and batch 32 the KV cache
binds for the 7B multi-head model: {gb(kv(m7, 4096, 32, 16)):.1f} GB of cache
against {gb(weights(m7, 4)):.1f} GB of weights. The model everybody calls small
has become a cache problem, and quantizing its weights further would not help at
all (eq:inference-budget).

The GQA rows are the control. The 8B model has a nearly identical parameter count
to the 7B one and eight times fewer KV heads, and it stays weight-bound in
situations where the 7B model is cache-bound. That is the architectural lever from
ch:q-activation-kv, seen here as the difference between fitting and not.

The activations column is the one to notice for what it is NOT. During decode it
is negligible -- hundredths of a gigabyte -- because each sequence contributes one
token's worth of intermediates. It is in the table so that its absence is
explicit, and because the next listing shows what happens to it during prefill,
where it is not negligible at all.

The second table applies the cache lever and reports the binding term afterwards,
which is the part worth having. Quantizing the cache to 4 bits moves every "too
big" row onto a card -- 279.6 GB to 73.4, 208.0 to 79.2 -- so the intervention
works.

And the binding column has not changed. The cache still binds in every row after
a fourfold reduction, because it was ten to eighty times the weights before it.
That is the useful negative: a 4x lever applied to a 20x problem leaves a 5x
problem, and the configuration is still cache-limited.

So the next move is not more cache quantization -- 2-bit would buy another factor
of two against a term that needs another factor of five. It is the architectural
lever, or shorter contexts, or fewer concurrent sequences. The binding column is
what says so, and it says so before any of those are tried.

That is the discipline this listing is for. Every optimisation moves the budget
and may or may not move the constraint, and the next thing to do is a function of
where the constraint ended up rather than of what helped last time.

The last table converts the budget into the number a capacity planner actually
needs -- concurrent sequences -- and it contains the most striking row in the
chapter.

On an 80 GB card at 8k context with a 16-bit cache, the 70B GQA-8 model fits
{caps[('70B GQA-8', '80 GB', 8192)]:,} concurrent sequences. The 7B multi-head
model fits {caps[('7B  MHA', '80 GB', 8192)]:,}.

Ten times the parameters, and essentially the same number of concurrent users.

That is not a rounding artefact, it is the arithmetic. Serving capacity is
governed by the cache, and the cache scales with layers times KV heads times head
dimension -- not with parameter count. The 70B model has more layers and the 7B
model has eight times more KV heads, and those nearly cancel. The 8B GQA-8 model
on the same card fits {caps[('8B  GQA-8', '80 GB', 8192)]:,}, four times either
of them, because it has the small model's layer count AND the large model's head
grouping.

So parameter count is close to useless as a predictor of serving capacity, and
the number in a model's name -- the one that determines its price, its
reputation, and the hardware people budget for -- tells you almost nothing about
how many users it can serve at once.

Which gives the sentence this chapter is for. **Model size tells you whether it
loads. Architecture and context tell you whether it serves.** The first is the
question everybody asks; the second is the one that decides the deployment, and
it is answerable in advance with the arithmetic above rather than discovered
during a load test.""")

# -*- coding: utf-8 -*-
# Extracted from: Chapter 122 — Vision Transformers
# Source: src/.../ch122-vision-transformers.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Patch size: the one hyperparameter that sets everything else about a ViT.

A vision transformer's only real decision about images is how to cut them into
tokens. Everything downstream -- how much it can see, what it costs, what it is
structurally unable to resolve -- follows from the patch size, and the
relationships are exact rather than empirical.

Three quantities matter and they pull in opposite directions:
  - token count grows as (H/p)^2, and attention costs its square
    (eq:vit-attention-cost);
  - the patch embedding compresses p*p*C numbers into d, so the compression
    ratio is p^2 C / d (eq:patch-compression);
  - anything finer than a patch has to survive that compression to be
    representable at all.

This listing computes all three across the configurations people actually use.
"""
import numpy as np

C, D = 3, 768                   # input channels, embedding width
HEADS = 12


def config(img, patch):
    n = (img // patch) ** 2                       # tokens (ignoring CLS)
    # Patch embedding: one matmul of (n x p^2 C) by (p^2 C x d).
    embed_flops = n * (patch ** 2 * C) * D
    # Attention: QKV projections, the n x n score matrix, and the value mix.
    proj = 4 * n * D * D
    attn = 2 * n * n * D
    return {
        "tokens": n,
        "compress": (patch ** 2 * C) / D,
        "embed_gf": embed_flops / 1e9,
        "attn_gf": attn / 1e9,
        "proj_gf": proj / 1e9,
        "total_gf": (embed_flops + proj + attn) / 1e9,
        "attn_share": attn / (embed_flops + proj + attn),
    }


print(f"embedding width d = {D}, {C} input channels\n")
print(f"{'image':>7}{'patch':>7}{'tokens':>8}{'p^2C/d':>9}{'embed GF':>10}"
      f"{'proj GF':>9}{'attn GF':>9}{'total GF':>10}{'attn share':>12}")
print("-" * 81)

rows = {}
for img, patch in ((224, 32), (224, 16), (224, 14), (224, 8),
                   (448, 16), (896, 16), (1024, 16)):
    r = config(img, patch)
    rows[(img, patch)] = r
    print(f"{img:>7}{patch:>7}{r['tokens']:>8}{r['compress']:>9.2f}"
          f"{r['embed_gf']:>10.2f}{r['proj_gf']:>9.2f}{r['attn_gf']:>9.2f}"
          f"{r['total_gf']:>10.2f}{r['attn_share']:>12.1%}")

a, b = rows[(224, 16)], rows[(224, 8)]
c, e = rows[(224, 16)], rows[(1024, 16)]
print(f"""
Start with the compression column, because it explains a number everyone treats
as arbitrary. At patch 16 with three channels, a patch holds 16*16*3 = 768 raw
values and the embedding has width 768 -- a ratio of exactly
{rows[(224, 16)]['compress']:.2f}. The standard configuration is the one where
the patch embedding is dimensionally lossless: it is a change of basis, not a
summary. At patch 32 the ratio is {rows[(224, 32)]['compress']:.2f}, so three
quarters of the information inside each patch has to be discarded before a single
attention operation runs.

That is the first thing to know about a vision tower: below the patch grid, it
does not see pixels, it sees whatever survived one linear projection. Text on a
document, a distant road sign, the tick labels on a chart -- if the stroke is thin
relative to the patch and the projection is compressive, the evidence is gone
before the model starts, and no amount of attention recovers it.

Now the cost columns, and the reason patch 16 is not simply replaced by patch 8.
Halving the patch quadruples the token count, {a['tokens']} to {b['tokens']}, and
attention cost goes as the SQUARE of that: {a['attn_gf']:.2f} GFLOPs to
{b['attn_gf']:.2f}, a factor of {b['attn_gf'] / a['attn_gf']:.0f}
(eq:vit-attention-cost). Total cost rises by
{b['total_gf'] / a['total_gf']:.1f}x. Four times the detail for
{b['total_gf'] / a['total_gf']:.0f} times the compute, and the exponent is what
makes fine patches unaffordable rather than merely expensive.

Read the attention-share column down the last three rows, because it settles a
common misconception. At 224 pixels attention is a MINORITY of the cost --
{c['attn_share']:.0%} -- and the projections and patch embedding dominate. People
who profile a ViT at standard resolution correctly conclude that attention is not
the bottleneck, and then generalise it. At 1024 pixels the same architecture
spends {e['attn_share']:.0%} of its FLOPs inside attention, because that term is
the only one growing quadratically while everything else grows linearly in token
count.

So "attention is quadratic" is a statement about a REGIME, not about a model. It
is false at the resolution ViTs are usually benchmarked at and dominant at the
resolution documents need -- which is exactly why high-resolution vision is where
the efficient-attention literature (ch:tf-efficient) actually earns its keep, and
why ch:mm-vlms has to solve the token-budget problem before it can read a page.""")

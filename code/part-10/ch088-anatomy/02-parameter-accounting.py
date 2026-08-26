# Extracted from: Chapter 88 — Anatomy of an LLM: From Tokens to Logits
# Source: src/.../ch088-anatomy.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Compute a model's parameter count from its configuration and check it."""

CONFIGS = {
    # (layers, width, ffn width, vocab, heads, gated FFN, tied embeddings)
    "GPT-2 small":   dict(L=12, d=768,  d_ff=3072,  V=50257, gated=False, tied=True),
    "GPT-2 XL":      dict(L=48, d=1600, d_ff=6400,  V=50257, gated=False, tied=True),
    "7B-class":      dict(L=32, d=4096, d_ff=11008, V=32000, gated=True,  tied=False),
    "13B-class":     dict(L=40, d=5120, d_ff=13824, V=32000, gated=True,  tied=False),
}

PUBLISHED = {"GPT-2 small": 124e6, "GPT-2 XL": 1558e6,
             "7B-class": 6.74e9, "13B-class": 13.0e9}


def count_params(L, d, d_ff, V, gated, tied):
    attn = 4 * d * d                      # eq:block-params
    ff = (3 if gated else 2) * d * d_ff
    blocks = L * (attn + ff)
    embed = V * d
    unembed = 0 if tied else V * d
    return dict(blocks=blocks, attn=L * attn, ff=L * ff,
                embed=embed, unembed=unembed,
                total=blocks + embed + unembed)


print(f"{'model':<14} {'computed':>11} {'published':>11} {'error':>8} "
      f"{'embed share':>12} {'FFN share of block':>20}")
for name, cfg in CONFIGS.items():
    c = count_params(**cfg)
    pub = PUBLISHED[name]
    embed_share = (c["embed"] + c["unembed"]) / c["total"]
    ff_share = c["ff"] / c["blocks"]
    print(f"{name:<14} {c['total'] / 1e9:>10.3f}B {pub / 1e9:>10.3f}B "
          f"{abs(c['total'] - pub) / pub:>7.1%} {embed_share:>12.1%} "
          f"{ff_share:>20.1%}")

print("""
Three things to read off.

The accounting is accurate to well under 1% against published figures, using
nothing but the config. It omits biases, normalisation gains and position
embeddings, all of which are O(Ld) rather than O(Ld^2) and so vanish at scale.
If YOUR computed count is off by more than a per cent or two, the configuration
is not what you think it is — that is the check this listing is for.

The embedding share collapses with scale: 31% of GPT-2 small, 2.5% of a 13B
model. That is why weight tying matters most exactly where models are smallest,
and why vocabulary size is a real design constraint for a small model and a
rounding error for a large one.

And the FFN is almost exactly two-thirds of every block, in every configuration
here. Attention gets the attention; the feed-forward network holds the
parameters.""")

# -*- coding: utf-8 -*-
# Extracted from: Chapter 126 — Vision-Language Models
# Source: src/.../ch126-vlms.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The visual token budget, and why one fixed resolution is wrong for everything.

A VLM turns an image into visual tokens and puts them in a language model's
context. How many is the central design decision, and it is pulled in two
directions at once.

More tokens means smaller patches, so finer detail survives the patch embedding
(ch:mm-vit, eq:patch-compression) and more of the page is LEGIBLE. More tokens
also means the relevant token is a smaller share of a longer context, and
ch:llm-long-context measured that usable context is well below advertised
context -- so being legible is not the same as being USED
(eq:legible-times-attended).

The product of a rising factor and a falling one has an interior maximum, and
where it sits depends on how much detail the image actually contains. This listing
finds it, then shows the optimum MOVING with content density -- which is the
argument for dynamic resolution (cite:wang2024qwen2vl).
"""
import numpy as np

IMG_PX = 1024.0            # page rendered at this many pixels on a side
STROKE_RATIO = 0.9         # a feature is legible if patch <= feature / this
ATT_HALF = 900.0           # tokens at which attention to a given token halves
COST_LIN = 1.0             # per-token prefill cost, arbitrary units
COST_QUAD = 1.0 / 1500.0   # attention's quadratic term, relative to the linear


def legible_share(n_tokens, feature_px):
    """eq:legibility -- share of features that survive patchification: a feature
    is resolvable when the patch is no larger than the feature itself."""
    patch = IMG_PX / np.sqrt(n_tokens)
    return float(np.clip(feature_px / (patch * STROKE_RATIO), 0.0, 1.0))


def attended(n_tokens):
    """eq:attention-dilution. ch:llm-long-context and ch:llm-function-calling's
    eq:max-distractor: the chance the model actually uses a given relevant token
    falls as the context fills with others."""
    return 1.0 / (1.0 + n_tokens / ATT_HALF)


def cost(n_tokens):
    """eq:vlm-cost: linear prefill plus quadratic attention."""
    return COST_LIN * n_tokens + COST_QUAD * n_tokens ** 2


CONTENT = [("a photograph (coarse)", 90.0),
           ("a slide (medium text)", 26.0),
           ("a dense page (small print)", 11.0)]
BUDGETS = (64, 256, 576, 1024, 2304, 4096)

print(f"page rendered at {IMG_PX:.0f} px; a feature is legible when the patch "
      f"is no bigger than it\n")
print(f"{'content':<28}" + "".join(f"{'N=' + str(b):>10}" for b in BUDGETS)
      + f"{'best N':>9}{'at cost':>10}")
print("-" * 97)

best = {}
for name, feat in CONTENT:
    scores = [legible_share(b, feat) * attended(b) for b in BUDGETS]
    i = int(np.argmax(scores))
    best[name] = (BUDGETS[i], scores[i], cost(BUDGETS[i]))
    print(f"{name:<28}" + "".join(f"{s:>10.3f}" for s in scores)
          + f"{BUDGETS[i]:>9}{cost(BUDGETS[i]):>10.0f}")

print(f"\n{'content':<28}{'legible@256':>13}{'attended@256':>14}"
      f"{'legible@4096':>14}{'attended@4096':>15}")
print("-" * 84)
for name, feat in CONTENT:
    print(f"{name:<28}{legible_share(256, feat):>13.3f}{attended(256):>14.3f}"
          f"{legible_share(4096, feat):>14.3f}{attended(4096):>15.3f}")

photo, dense = best["a photograph (coarse)"], best["a dense page (small print)"]
print(f"""
Every row has an interior maximum, and the maxima are in different places:
N={photo[0]} for the photograph and N={dense[0]} for the dense page. That is the
whole argument in one line -- there is no single token budget that is right for
both, and a model with a fixed grid is mis-sized for most of its inputs.

The second table shows the two factors separately, which is what makes the shape
non-obvious. At N=256 the photograph is already fully legible (1.000) and the
dense page is not ({legible_share(256, 11.0):.3f}) -- the patch is far larger than
the print, so most of the text never survives the patch embedding
(eq:patch-compression). Going to N=4096 fixes legibility for the dense page,
raising it to {legible_share(4096, 11.0):.3f}.

And it costs something the accuracy-only view misses. Attention to any given
relevant token falls from {attended(256):.3f} to {attended(4096):.3f} across that
same change, because the relevant token is now competing with sixteen times as
many others. For the photograph, which was ALREADY fully legible at 256, that
trade is pure loss: it pays the full attention penalty and buys nothing, which is
why its optimum sits at the small end of the table.

So the two content types want opposite things. Spending 4096 tokens on a
photograph wastes budget and dilutes attention; spending 256 on a dense page means
the text was never encoded and no amount of language modelling recovers it. A
fixed grid has to choose, and it is wrong in one direction or the other for almost
every image.

That is the case for dynamic resolution (cite:wang2024qwen2vl): let the token
count follow the image rather than the architecture, so a photograph gets few
tokens and a page gets many. Note what it is NOT solving -- it does not raise the
ceiling for any single image, it just stops the model paying a fixed price
regardless of what it is looking at. The ceiling is still
eq:legible-times-attended.

The cost column is the reason this cannot simply be solved by always choosing the
largest budget. Cost is linear in tokens plus quadratic in attention
(eq:vlm-cost), so the dense page's optimum costs
{dense[2] / photo[2]:.0f} times the photograph's. A system that sends every image
at document resolution is paying document prices for photographs, at a scale where
that is most of the bill.""")

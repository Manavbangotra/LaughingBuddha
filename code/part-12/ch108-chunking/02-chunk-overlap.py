# -*- coding: utf-8 -*-
# Extracted from: Chapter 108 — Chunking Strategies
# Source: src/.../ch108-chunking.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""What overlap buys, and what it costs.

Overlap hedges against answers that straddle a chunk boundary. eq:span-containment
predicts the benefit and eq:index-multiplier the cost; here both are measured
against each other so the trade is visible rather than assumed.

The result worth watching is the interaction: overlap's value depends almost
entirely on how large the chunk already is relative to the answer span.
"""
import numpy as np

rng = np.random.default_rng(11)

N_TRIALS = 20_000
SENT_PER_DOC = 48
CONFIGS = [(4, 4), (4, 3), (4, 2), (4, 1),
           (8, 8), (8, 6), (8, 4), (8, 2),
           (16, 16), (16, 12), (16, 8)]


def containment_rate(L, S, w, trials=N_TRIALS):
    """Fraction of random w-spans fully inside at least one chunk (L, stride S)."""
    starts = np.arange(0, SENT_PER_DOC, S)
    hits = 0
    for _ in range(trials):
        s0 = int(rng.integers(0, SENT_PER_DOC - w + 1))
        if any(cs <= s0 and s0 + w - 1 < min(cs + L, SENT_PER_DOC) for cs in starts):
            hits += 1
    return hits / trials


print(f"{'chunk L':>9}{'stride S':>10}{'overlap':>9}{'index x':>9}"
      + "".join(f"{'w=' + str(w):>8}" for w in (1, 3, 6)))
print("-" * 61)
rows = {}
for L, S in CONFIGS:
    overlap = 1 - S / L
    mult = L / S
    rates = [containment_rate(L, S, w) for w in (1, 3, 6)]
    rows[(L, S)] = (mult, rates)
    print(f"{L:>9}{S:>10}{overlap:>8.0%}{mult:>8.2f}x"
          + "".join(f"{r:>8.3f}" for r in rates))

r = {(L, S): rows[(L, S)][1] for L, S in CONFIGS}
print(f"""
Read the w=1 column first: every configuration is 1.000. A single-sentence answer
is inside SOME chunk no matter how you cut, so for fact-lookup queries overlap
buys exactly nothing and costs the index multiplier in full. Systems with a
default 20% overlap serving a mostly fact-lookup workload are paying a fifth of
their index for zero benefit -- and that describes a great many systems.

Now the w=6 column at L=4: every entry is 0.000, including the one with 75%
overlap and a 4x index. OVERLAP CANNOT COMPENSATE FOR A CHUNK SMALLER THAN THE
ANSWER SPAN. No stride makes a four-sentence window contain six sentences. If any
material share of your queries needs a span wider than your chunk, they are not
merely retrieved poorly -- they are unanswerable, and no retrieval depth, no
overlap, and no reranker changes that.

Next, chunk size against overlap as substitutes. At w=3 with NO overlap,
L=4 gives {r[(4, 4)][1]:.3f} and L=16 gives {r[(16, 16)][1]:.3f} -- the larger
chunk contains the span far more often at an index multiplier of 1.00x, i.e. for
free. Buying containment with size is strictly cheaper than buying it with
overlap, whenever the dilution cost of the larger chunk is acceptable.

Finally, when is overlap worth its cost? Compare the same 25% overlap applied at
two sizes, at w=3: it adds {r[(4, 3)][1] - r[(4, 4)][1]:+.3f} at L=4 and only
{r[(16, 12)][1] - r[(16, 16)][1]:+.3f} at L=16, because containment at L=16 was
already {r[(16, 16)][1]:.3f} and there was little left to buy. But at w=6 the
same comparison reverses: {r[(8, 6)][2] - r[(8, 8)][2]:+.3f} at L=8 against
{r[(16, 12)][2] - r[(16, 16)][2]:+.3f} at L=16.

So the rule is not about L. It is about L/w: overlap is worth most when
containment is in the partial regime of eq:span-containment and worth nearly
nothing once L is comfortably above w. Which means you cannot choose an overlap
without knowing w -- and a fixed 20% default, applied without measuring the
answer-span width, is as likely to be wasted as well spent.""")

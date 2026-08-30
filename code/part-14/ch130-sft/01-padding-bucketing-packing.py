# -*- coding: utf-8 -*-
# Extracted from: Chapter 130 — Supervised Fine-Tuning
# Source: src/.../ch130-sft.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Padding, bucketing and packing: where a fine-tuning budget actually goes.

Training examples have wildly different lengths and a GPU wants rectangles, so
something has to reconcile them. The three standard answers differ by a large
factor in how much of the compute is spent on real tokens
(eq:token-efficiency), and the difference is invisible in the loss curve --
padding tokens are masked out, so they cost money and produce nothing while the
run looks perfectly healthy.

Packing is the efficient answer and it introduces a hazard the other two do not
have: several examples share one sequence, so without a block-diagonal attention
mask, tokens attend ACROSS example boundaries and the model conditions on text
from an unrelated sample (eq:cross-contamination). This listing measures both the
saving and the exposure.
"""
import numpy as np

rng = np.random.default_rng(137)

N_EXAMPLE = 20000
BATCH = 16
BLOCK = 2048               # packed block length, and the max sequence length


def lengths(kind):
    """Realistic instruction-tuning length distributions are heavy-tailed: most
    examples are short and a few are very long."""
    if kind == "uniform-ish":
        L = rng.integers(200, 600, size=N_EXAMPLE).astype(float)
    elif kind == "heavy-tailed":
        L = rng.lognormal(mean=5.4, sigma=0.9, size=N_EXAMPLE)
    else:                                    # bimodal: short chats + long docs
        short = rng.lognormal(mean=4.8, sigma=0.4, size=N_EXAMPLE)
        long_ = rng.lognormal(mean=6.9, sigma=0.5, size=N_EXAMPLE)
        pick = rng.random(N_EXAMPLE) < 0.75
        L = np.where(pick, short, long_)
    return np.clip(L, 24, BLOCK).astype(int)


def naive_padding(L):
    """Shuffle, batch, pad each batch to its own longest member."""
    order = rng.permutation(len(L))
    used = total = 0
    for s in range(0, len(L), BATCH):
        b = L[order[s:s + BATCH]]
        used += b.sum()
        total += len(b) * b.max()
    return used / total


def bucketed(L):
    """Sort by length, then batch -- neighbours have similar lengths so the pad
    to the batch maximum is small."""
    srt = np.sort(L)
    used = total = 0
    for s in range(0, len(srt), BATCH):
        b = srt[s:s + BATCH]
        used += b.sum()
        total += len(b) * b.max()
    return used / total


def packed(L):
    """Concatenate examples into fixed BLOCK-length sequences, starting a new
    block only when the next example does not fit (eq:packing-efficiency)."""
    used = total = 0
    cur = 0
    for x in L:
        if cur + x > BLOCK:
            total += BLOCK
            used += cur
            cur = 0
        cur += x
    total += BLOCK
    used += cur
    return used / total


def contamination(L):
    """In a packed block WITHOUT a block-diagonal mask, every token may attend to
    every earlier token in the block. Report the share of attend-able pairs that
    cross an example boundary -- i.e. the share of attention capacity pointed at
    an unrelated example (eq:cross-contamination)."""
    blocks, cur = [], []
    tot = 0
    for x in L[:6000]:
        if tot + x > BLOCK:
            blocks.append(cur); cur = []; tot = 0
        cur.append(x); tot += x
    if cur:
        blocks.append(cur)
    cross = within = 0
    for b in blocks:
        b = np.asarray(b)
        n = b.sum()
        # Causal pairs inside the block, and those inside each example.
        cross += n * (n - 1) / 2
        within += (b * (b - 1) / 2).sum()
    return float((cross - within) / cross) if cross else 0.0


print(f"{N_EXAMPLE:,} examples, batch {BATCH}, block/max length {BLOCK}\n")
print(f"{'length profile':<18}{'median':>8}{'p99':>8}{'':>3}"
      f"{'naive pad':>12}{'bucketed':>11}{'packed':>9}{'':>3}"
      f"{'cross-example':>15}")
print("-" * 88)

res = {}
for kind in ("uniform-ish", "heavy-tailed", "bimodal"):
    L = lengths(kind)
    n, b, p = naive_padding(L), bucketed(L), packed(L)
    c = contamination(L)
    res[kind] = (n, b, p, c)
    print(f"{kind:<18}{int(np.median(L)):>8}{int(np.percentile(L, 99)):>8}{'':>3}"
          f"{n:>12.3f}{b:>11.3f}{p:>9.3f}{'':>3}{c:>15.1%}")

ht = res["heavy-tailed"]
bm = res["bimodal"]
print(f"""
The three efficiency columns are the fraction of processed tokens that are real
rather than padding. On the heavy-tailed profile -- which is what an instruction
dataset actually looks like, mostly short with a long tail -- naive padding runs
at {ht[0]:.3f}. Around {1 - ht[0]:.0%} of the compute is spent on padding tokens
that are masked out of the loss and contribute nothing.

That waste is invisible. Padding is masked, so the loss curve is clean, the
gradients are correct, and the run looks healthy while most of the bill buys
nothing. It surfaces only as "training is slower and costs more than expected",
which is usually blamed on the model size.

The mechanism is that a batch is padded to its OWN longest member, so one outlier
sets the cost for the fifteen examples beside it (eq:token-efficiency). With a
heavy-tailed distribution most batches contain such an outlier -- which is what
heavy-tailed means. The bimodal row is worse still at {bm[0]:.3f}, because a
mixture of short chats and long documents guarantees the outlier.

Now the result this listing was not built to find. Bucketing -- sort by length,
then batch -- reaches {ht[1]:.3f}, and packing reaches {ht[2]:.3f}. BUCKETING
WINS, on every row, and it wins by a clear margin.

The reason is that packing's waste has simply moved. A block is filled until the
next example does not fit, and the remainder is discarded, so packing pays a
partial block at the end of every block rather than a partial batch at the end of
every batch. With a block of {BLOCK} and a median example of a few hundred
tokens, that tail is a meaningful fraction. Bucketing has no such tail: after
sorting, the pad to the batch maximum is nearly zero because the batch's members
are nearly the same length.

So the usual ordering of these techniques is wrong, at least on token efficiency.
Sorting by length is one line, needs no attention-mask changes, and recovers
almost all of the loss. It is the first thing to try and it is routinely skipped
in favour of the more sophisticated option.

Bucketing does cost something the table does not show: batches are no longer
randomly composed, so examples within a batch are correlated by length, which
correlates them by type too. The standard mitigation is to shuffle the ORDER of
buckets while keeping their contents, which restores randomness across steps
while keeping it out of each step.

And packing carries a hazard bucketing does not, which is the last column. In a
packed block several unrelated examples share one sequence, and unless the
attention mask is block-diagonal, every token may attend to every earlier token
in the block -- including tokens belonging to somebody else's example. The
measured share of attend-able pairs that cross a boundary is {ht[3]:.0%} on the
heavy-tailed profile.

That is a large leak in an unhelpful direction (eq:cross-contamination). The model
is trained to produce a completion while conditioning on an unrelated example
that happens to precede it, which teaches precisely the wrong lesson: that
whatever came before the current instruction is relevant to it. The symptom at
inference is a model that drags irrelevant material across turn boundaries.

The fix is mechanical -- a block-diagonal attention mask so each packed example
attends only to itself, and position IDs that reset per example rather than
running across the block. Every serious training stack supports both, and both
are the kind of flag that is easy to leave unset.

Which gives a clear recommendation. Bucket first: it is simpler, more efficient
here, and has no correctness hazard. Reach for packing when you need fixed-shape
blocks for a compiled graph, or when sequence lengths approach the block size so
the tail waste disappears -- and if you do, verify the mask before the run rather
than after.""")

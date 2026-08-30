# -*- coding: utf-8 -*-
# Extracted from: Chapter 143 — Memory Math: Will This Model Fit?
# Source: src/.../ch143-memory-math.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The budget that fits and the run that fails: prefill is a different machine.

The previous listing computed steady-state decode memory, and a deployment sized
by that arithmetic can still die on its first long request. The reason is that a
request has two phases with completely different memory profiles, and only one of
them is in the steady-state number.

Decode processes one token per sequence, so its activations are negligible.
PREFILL processes the entire prompt at once, so its activations scale with the
prompt length -- and, without the right attention kernel, with the prompt length
SQUARED (eq:prefill-is-the-peak).

This listing computes the peak rather than the average, and prices the two
standard remedies.
"""
import numpy as np

M = dict(P=70e9, L=80, h=64, hkv=8, d=128, dm=8192)
FRAMEWORK = 1.2e9
CARD = 80e9


def gb(x):
    return x / 1e9


def weights(wbits=4):
    return M["P"] * wbits / 8.0


def kv(ctx, batch, bits=16):
    return 2 * M["L"] * M["hkv"] * M["d"] * ctx * batch * bits / 8.0


def act_linear(tokens, batch, bytes_per=2, live=6):
    """Intermediates that scale with the number of tokens being processed."""
    return live * batch * tokens * M["dm"] * bytes_per


def act_scores(tokens, batch, bytes_per=2, concurrent_layers=1):
    """The attention score matrix, tokens x tokens per head. A fused kernel
    never materialises this; a naive implementation materialises one layer's
    worth at a time."""
    return concurrent_layers * batch * M["h"] * tokens * tokens * bytes_per


print(f"70B GQA-8, weights at 4 bits ({gb(weights()):.0f} GB), one 80 GB card.")
print("Peak memory during PREFILL of a prompt, batch 1.")
print()
print(f"{'prompt':>9}{'weights':>9}{'KV':>8}{'linear':>9}{'scores':>11}"
      f"{'scores':>11}{'peak, no':>11}{'peak,':>10}")
print(f"{'tokens':>9}{'':>9}{'':>8}{'activ.':>9}{'naive':>11}{'fused':>11}"
      f"{'fusion':>11}{'fused':>10}")
print("-" * 78)

rows = {}
for S in (1024, 4096, 16384, 65536, 131072):
    w, k = weights(), kv(S, 1)
    al = act_linear(S, 1)
    an, af = act_scores(S, 1), 0.0
    pn = w + k + al + an + FRAMEWORK
    pf = w + k + al + af + FRAMEWORK
    rows[S] = (pn, pf, an, al, k)
    print(f"{S:>9,}{gb(w):>9.1f}{gb(k):>8.1f}{gb(al):>9.2f}{gb(an):>11.1f}"
          f"{gb(af):>11.1f}{gb(pn):>11.1f}{gb(pf):>10.1f}")

print()
print()
print("Chunked prefill: process the prompt in pieces of C tokens.")
print("Fused attention, batch 1, 131072-token prompt.")
print()
print(f"{'chunk C':>10}{'linear activ.':>15}{'peak':>10}{'fits 80 GB':>13}"
      f"{'prefill passes':>16}")
print("-" * 64)
S = 131072
chunks = {}
for C in (131072, 32768, 8192, 2048, 512):
    al = act_linear(C, 1)
    peak = weights() + kv(S, 1) + al + FRAMEWORK
    chunks[C] = peak
    print(f"{C:>10,}{gb(al):>13.2f} GB{gb(peak):>8.1f} GB"
          f"{('yes' if peak < CARD else 'NO'):>13}{S // C:>16}")

print()
print()
print("The steady-state trap: a batch sized on decode, then given long prompts.")
print()
print(f"{'batch':>7}{'context':>9}{'decode':>10}{'prefill 1':>12}"
      f"{'prefill all':>13}{'verdict':>24}")
print("-" * 75)

for batch, ctx in ((16, 8192), (16, 32768), (48, 8192), (8, 65536)):
    dec = weights() + kv(ctx, batch, 4) + act_linear(1, batch) + FRAMEWORK
    pre1 = weights() + kv(ctx, batch, 4) + act_linear(ctx, 1) + FRAMEWORK
    preall = weights() + kv(ctx, batch, 4) + act_linear(ctx, batch) + FRAMEWORK
    v = ("fine" if preall < CARD else
         "OOM if prompts overlap" if pre1 < CARD else "OOM on one prompt")
    print(f"{batch:>7}{ctx:>9,}{gb(dec):>8.1f} GB{gb(pre1):>10.1f} GB"
          f"{gb(preall):>11.1f} GB  {v:>22}")

r16, r131 = rows[16384], rows[131072]
print(f"""
The first table is the failure that sizing on decode cannot predict.

Look at the naive-scores column. At a 16k prompt the attention score matrix is
{gb(r16[2]):.1f} GB; at 131k it is {gb(r131[2]):.1f} GB. That is one tensor, for
one layer, for one sequence -- and it is larger than the model, larger than the
card, larger by an amount no other term in the budget approaches. It is quadratic
in the prompt length, and quadratic terms do not stay small.

The fused column is zero, because a fused attention kernel never materialises the
score matrix at all: it computes attention in tiles and keeps only the running
softmax statistics. The entire difference between the last two columns is whether
your kernel does that (eq:prefill-is-the-peak).

That is worth stating in the strongest form the numbers support. Without fused
attention, long-context inference is not slow or memory-hungry -- it is
IMPOSSIBLE, on any hardware, for prompts of the length people now routinely send.
The technique that made long context practical was not a bigger card.

With fusion, the peak at 131k tokens is {gb(r131[1]):.1f} GB -- and the card is
{gb(CARD):.0f} GB, so it STILL does not fit. Fusion removed a 2199 GB term and
left a configuration that is over budget by
{gb(r131[1] - CARD):.0f} GB. The linear activation term is now the problem, and
the second table prices the standard answer to it.

Chunked prefill processes the prompt in pieces, running the model over C tokens at
a time and appending each piece's keys and values to the cache. The linear
activation term becomes proportional to C rather than to the whole prompt, and it
is the only term that changes -- the KV cache still grows to the full prompt,
because that is the point of prefilling.

At {131072:,} tokens in one pass the linear activations are
{gb(act_linear(131072, 1)):.2f} GB; in 2048-token chunks, {gb(act_linear(2048, 1)):.2f} GB.
The cost is {131072 // 2048} sequential passes instead of one, which is slower in
wall-clock but not in total arithmetic, since the same tokens are processed either
way.

The third table is the trap this listing exists for, and it is the one that
produces production incidents rather than benchmark surprises.

A deployment sized on decode memory picks a batch size that fits comfortably.
Then a request arrives with a long prompt, and prefill for that ONE sequence
allocates activations proportional to its whole prompt -- while every other
sequence's cache is still resident and cannot be freed. The decode column and the
prefill column are different numbers for the same configuration, and only the
first appears in a steady-state calculation.

Read the verdict column. Some configurations survive a single long prefill and
fail when two arrive at once, which means the failure is LOAD-DEPENDENT and will
not reproduce under a benchmark that sends one request at a time. That is the
worst kind of capacity bug: it passes every test, fails in production, and the
trigger is a coincidence of arrival times.

Three things follow, and they are the practical content of the chapter.

Size on the PEAK, not the steady state, and compute the peak with the longest
prompt you will accept rather than the average one. If you accept 128k prompts,
128k is the number in the calculation regardless of how rare they are.

Use chunked prefill, and set the chunk size from the memory budget rather than
from a default. It converts an unbounded term into a bounded one, and the
conversion is exact.

And enforce an admission limit on concurrent prefills. The cache term is shared
across sequences and the prefill activation term is not, so the number of
sequences that may be in prefill simultaneously is a separate capacity from the
number that may be resident -- and a scheduler that does not distinguish them will
eventually put too many in the wrong phase at the same moment.""")

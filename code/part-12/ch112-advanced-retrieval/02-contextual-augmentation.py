# -*- coding: utf-8 -*-
# Extracted from: Chapter 112 — Advanced Retrieval: Parent–Child, Contextual, and Hybrid
# Source: src/.../ch112-advanced-retrieval.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Orphan chunks, and the one-line fix.

Split a document at a fixed size and many chunks are ORPHANS: their text alone
carries no topic -- "as noted above, this reduced latency substantially". They are
unretrievable, because their embedding contains nothing to match, and unusable,
because a reader cannot resolve the reference.

Prepending context before embedding (eq:contextual-augmentation) supplies the
missing topic. eq:augmentation-condition predicts this helps orphans and HURTS
already-specific chunks, and eq:context-length-constraint predicts the harm is
bounded by keeping the prepended text short. We test all three claims.
"""
import numpy as np

rng = np.random.default_rng(29)

N_DOC, CHUNK_PER_DOC, DIM = 400, 12, 48
N_QUERY, K = 800, 10
ORPHAN_RATE = 0.35              # chunks whose own text carries no topic

topics = unit_doc = rng.normal(size=(N_DOC, DIM))
doc_topic = unit_doc / np.linalg.norm(unit_doc, axis=1, keepdims=True)


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


# A chunk's OWN content: specific chunks carry the document topic plus their own
# detail; orphans carry detail only -- no topic at all.
is_orphan = rng.random((N_DOC, CHUNK_PER_DOC)) < ORPHAN_RATE
detail = unit(rng.normal(size=(N_DOC, CHUNK_PER_DOC, DIM)))
own = np.where(is_orphan[..., None],
               detail,
               unit(0.75 * doc_topic[:, None, :] + 0.45 * detail))
own = unit(own)


def augmented(context_weight):
    """eq:augmentation-mix: mix the chunk's own vector with its document's.
    context_weight is L_x / (L_c + L_x) -- the share of the embedded text that
    is prepended context."""
    return unit((1 - context_weight) * own
                + context_weight * doc_topic[:, None, :])


def evaluate(vecs):
    """Retrieve for queries aimed at a specific chunk; report separately for
    orphan and specific targets."""
    flat = vecs.reshape(-1, DIM)
    hits = {"orphan": [0, 0], "specific": [0, 0]}
    for _ in range(N_QUERY):
        d = int(rng.integers(0, N_DOC))
        c = int(rng.integers(0, CHUNK_PER_DOC))
        # The query knows the topic and the detail -- it is a real question about
        # this chunk's content, phrased by someone who knows the document.
        q = unit(0.7 * doc_topic[d] + 0.7 * detail[d, c]
                 + rng.normal(scale=0.20, size=DIM))
        top = np.argpartition(-(flat @ q), K)[:K]
        kind = "orphan" if is_orphan[d, c] else "specific"
        hits[kind][0] += int(d * CHUNK_PER_DOC + c in top)
        hits[kind][1] += 1
    return {k: v[0] / v[1] for k, v in hits.items()}


print(f"{ORPHAN_RATE:.0%} of chunks are orphans (own text carries no topic)\n")
print(f"{'prepended context share':>24}{'orphan chunks':>16}"
      f"{'specific chunks':>18}{'overall':>10}")
print("-" * 70)
for w in [0.0, 0.10, 0.20, 0.35, 0.50, 0.70]:
    r = evaluate(augmented(w))
    overall = ORPHAN_RATE * r["orphan"] + (1 - ORPHAN_RATE) * r["specific"]
    print(f"{w:>24.2f}{r['orphan']:>16.3f}{r['specific']:>18.3f}{overall:>10.3f}")

print("""
The top row is unaugmented chunking, and the orphan column is why this chapter
has a second listing. Those chunks are in the index, they are perfectly good
text, and they are close to unretrievable -- their embedding contains no topic to
match, so a query about their document lands anywhere else.

Adding context rescues them, steeply at first. That is eq:augmentation-condition
in the regime where s(q, context) >> s(q, chunk): for an orphan the chunk's own
similarity is near zero, so any positive context term is a gain.

Now read the specific column, which is the cost nobody prices. It declines
monotonically, because the same mixing that supplies a missing topic DILUTES one
that was already there -- and worse, every chunk in a document is being pulled
toward the same point, which is ch:emb-what-they-are's anisotropy arriving by
choice rather than by accident. At high context share the chunks of a document
become nearly indistinguishable from each other, and retrieval can find the right
document while being unable to find the right chunk within it.

The overall column has an interior optimum, and it sits at a SMALL context share.
That is eq:context-length-constraint derived rather than asserted: a heading path
of a few tokens prepended to a hundred-token chunk lands near the left of this
table, captures most of the orphan gain, and costs the specific chunks almost
nothing. A two-hundred-token document summary prepended to the same chunk lands
on the right, and is a common way to make retrieval worse while believing you
improved it.""")

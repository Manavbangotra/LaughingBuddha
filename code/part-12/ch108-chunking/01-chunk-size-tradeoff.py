# -*- coding: utf-8 -*-
# Extracted from: Chapter 108 — Chunking Strategies
# Source: src/.../ch108-chunking.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""There is no optimal chunk size. There is an optimal chunk size PER QUERY TYPE.

A corpus of documents built from sentences with topic vectors. Two query types:

  fact      -- the answer is in ONE sentence (span w=1)
  synthesis -- the answer needs THREE consecutive sentences (span w=3)

A chunk's embedding is the mean of its sentences' embeddings, which is
eq:chunk-dilution made literal. Retrieval succeeds when a retrieved chunk
CONTAINS the whole answer span -- so the two forces of section 4 are both active
and measurable.
"""
import numpy as np

rng = np.random.default_rng(5)

N_DOC, SENT_PER_DOC, DIM = 150, 48, 48
N_TOPIC, K_RETRIEVE, N_QUERY = 40, 10, 600
CHUNK_SIZES = [1, 2, 3, 4, 6, 8, 12, 16, 24]

topics = rng.normal(size=(N_TOPIC, DIM))
topics /= np.linalg.norm(topics, axis=1, keepdims=True)


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


# Each document drifts slowly through topic space. A sentence is a topic
# component (shared with its neighbours, as in real prose) plus a large
# individual component -- which is what makes a single sentence identifiable,
# and therefore what averaging over a chunk destroys.
sent = np.zeros((N_DOC, SENT_PER_DOC, DIM))
for d in range(N_DOC):
    t = rng.integers(0, N_TOPIC)
    for s in range(SENT_PER_DOC):
        if rng.random() < 0.12:                 # occasional topic shift
            t = rng.integers(0, N_TOPIC)
        sent[d, s] = 0.7 * topics[t] + 0.35 * rng.normal(size=DIM)
sent = unit(sent)


def build_index(chunk_size):
    """Non-overlapping chunks; embedding is the mean of member sentences."""
    vecs, spans = [], []
    for d in range(N_DOC):
        for start in range(0, SENT_PER_DOC, chunk_size):
            end = min(start + chunk_size, SENT_PER_DOC)
            vecs.append(sent[d, start:end].mean(axis=0))
            spans.append((d, start, end))
    return unit(np.array(vecs)), spans


def evaluate(chunk_size, width):
    """Success = some retrieved chunk contains the ENTIRE answer span."""
    vecs, spans = build_index(chunk_size)
    hits = 0
    for _ in range(N_QUERY):
        d = int(rng.integers(0, N_DOC))
        s0 = int(rng.integers(0, SENT_PER_DOC - width + 1))
        target = range(s0, s0 + width)
        # The query looks like the answer span -- a noisy version of its mean.
        q = unit(sent[d, s0:s0 + width].mean(axis=0)
                 + rng.normal(scale=0.12, size=DIM))
        top = np.argpartition(-(vecs @ q), K_RETRIEVE)[:K_RETRIEVE]
        for i in top:
            cd, cs, ce = spans[i]
            if cd == d and cs <= target[0] and target[-1] < ce:
                hits += 1
                break
    return hits / N_QUERY


print(f"{'chunk size':>12}{'chunks':>9}{'fact (w=1)':>13}{'synthesis (w=3)':>18}")
print("-" * 54)
results = {}
for L in CHUNK_SIZES:
    n_chunks = len(build_index(L)[1])
    fact = evaluate(L, 1)
    synth = evaluate(L, 3)
    results[L] = (fact, synth)
    print(f"{L:>12}{n_chunks:>9}{fact:>13.3f}{synth:>18.3f}")

best_fact = max(results, key=lambda L: results[L][0])
best_synth = max(results, key=lambda L: results[L][1])
cost_at_synth_opt = results[best_fact][0] - results[best_synth][0]
cost_at_fact_opt = results[best_synth][1] - results[best_fact][1]
print(f"""
Optimal chunk size for FACT queries:      {best_fact} sentences
Optimal chunk size for SYNTHESIS queries: {best_synth} sentences

Cost of using the synthesis optimum for fact queries:      -{cost_at_synth_opt:.3f}
Cost of using the fact optimum for synthesis queries:      -{cost_at_fact_opt:.3f}

These are the two forces of section 4, separated and measured. Fact queries need
one sentence, so eq:span-containment is satisfied at every size and only dilution
matters -- the curve falls monotonically and the optimum sits at the smallest
usable chunk. Synthesis queries need three consecutive sentences, so small chunks
CANNOT contain the answer at all and the containment term dominates until the
chunk is comfortably larger than the span.

That is eq:optima-separate: one optimum is at a boundary and the other is
interior. They are not close, and no single number is near-optimal for both --
read the two cost lines above. Serving fact queries at the synthesis optimum
gives up roughly half of them; serving synthesis queries at the fact optimum
gives up ALL of them, because a one-sentence chunk cannot contain a
three-sentence answer at any retrieval depth.

That asymmetry is worth noticing on its own. Choosing too LARGE degrades
gracefully; choosing too SMALL fails absolutely for any query whose answer is
wider than the chunk. When you must guess, guess large.

This is why every published chunk-size recommendation disagrees with every other
one, and why all of them are defensible. Each is correct for the query
distribution its author had. Asking "what chunk size should I use" without
stating the query mix is asking an under-specified question, and the answer you
get back will be someone else's workload.""")

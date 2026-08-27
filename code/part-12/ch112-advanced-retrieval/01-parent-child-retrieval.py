# -*- coding: utf-8 -*-
# Extracted from: Chapter 112 — Advanced Retrieval: Parent–Child, Contextual, and Hybrid
# Source: src/.../ch112-advanced-retrieval.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Decoupling the retrieval unit from the generation unit.

ch:rag-chunking found that fact-lookup and synthesis queries have incompatible
optimal chunk sizes, so a single flat size is a compromise. Parent-child breaks
the tie: embed SMALL children for precision, send their LARGE parent for
completeness (eq:decoupled-success).

eq:pc-dominance predicts this dominates flat chunking rather than trading against
it. The comparison is run at an EQUAL CONTEXT BUDGET, because a technique that
wins by spending more tokens has not been shown to win.
"""
import numpy as np

rng = np.random.default_rng(5)

N_DOC, SENT_PER_DOC, DIM = 150, 48, 48
N_TOPIC, N_QUERY = 40, 600
BUDGET = 24                     # sentences of context, identical for all strategies

topics = rng.normal(size=(N_TOPIC, DIM))
topics /= np.linalg.norm(topics, axis=1, keepdims=True)


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


sent = np.zeros((N_DOC, SENT_PER_DOC, DIM))
for d in range(N_DOC):
    t = rng.integers(0, N_TOPIC)
    for s in range(SENT_PER_DOC):
        if rng.random() < 0.12:
            t = rng.integers(0, N_TOPIC)
        sent[d, s] = 0.7 * topics[t] + 0.35 * rng.normal(size=DIM)
sent = unit(sent)


def build(chunk_size):
    vecs, spans = [], []
    for d in range(N_DOC):
        for start in range(0, SENT_PER_DOC, chunk_size):
            end = min(start + chunk_size, SENT_PER_DOC)
            vecs.append(sent[d, start:end].mean(axis=0))
            spans.append((d, start, end))
    return unit(np.array(vecs)), spans


INDEXES = {L: build(L) for L in (1, 2, 6, 12)}


def evaluate(strategy, width):
    """Success = the CONTEXT SENT to the model contains the whole answer span,
    within a fixed budget of BUDGET sentences."""
    hits = 0
    for _ in range(N_QUERY):
        d = int(rng.integers(0, N_DOC))
        s0 = int(rng.integers(0, SENT_PER_DOC - width + 1))
        target = (s0, s0 + width)
        q = unit(sent[d, s0:s0 + width].mean(axis=0)
                 + rng.normal(scale=0.12, size=DIM))
        sent_spans = strategy(q)
        for (cd, cs, ce) in sent_spans:
            if cd == d and cs <= target[0] and target[1] <= ce:
                hits += 1
                break
    return hits / N_QUERY


def flat(chunk_size):
    """Retrieve top-k chunks of a single size, k set by the budget."""
    vecs, spans = INDEXES[chunk_size]
    k = max(1, BUDGET // chunk_size)

    def go(q):
        top = np.argpartition(-(vecs @ q), min(k, len(vecs) - 1))[:k]
        return [spans[i] for i in top]
    return go


def parent_child(child_size, parent_size):
    """Embed children; return their PARENTS, deduplicated (eq:parent-dedup-saving),
    taking as many as the budget allows."""
    vecs, spans = INDEXES[child_size]

    def go(q):
        order = np.argsort(-(vecs @ q))
        out, seen, spent = [], set(), 0
        for i in order:
            d, cs, _ = spans[i]
            p_start = (cs // parent_size) * parent_size
            key = (d, p_start)
            if key in seen:
                continue                      # dedupe: children share parents
            if spent + parent_size > BUDGET:
                break
            seen.add(key)
            out.append((d, p_start, min(p_start + parent_size, SENT_PER_DOC)))
            spent += parent_size
        return out
    return go


STRATEGIES = {
    "flat, L=1":              flat(1),
    "flat, L=2":              flat(2),
    "flat, L=6":              flat(6),
    "flat, L=12":             flat(12),
    "parent-child, 1 -> 6":   parent_child(1, 6),
    "parent-child, 1 -> 12":  parent_child(1, 12),
    "parent-child, 2 -> 12":  parent_child(2, 12),
}

print(f"context budget: {BUDGET} sentences for every strategy\n")
print(f"{'strategy':<26}{'fact (w=1)':>13}{'synth (w=3)':>14}"
      f"{'synth (w=6)':>14}{'mean':>8}")
print("-" * 76)
rows = {}
for name, strat in STRATEGIES.items():
    r = [evaluate(strat, w) for w in (1, 3, 6)]
    rows[name] = r
    print(f"{name:<26}{r[0]:>13.3f}{r[1]:>14.3f}{r[2]:>14.3f}"
          f"{np.mean(r):>8.3f}")

best_flat = max((n for n in rows if n.startswith("flat")),
                key=lambda n: np.mean(rows[n]))
best_pc = max((n for n in rows if n.startswith("parent")),
              key=lambda n: np.mean(rows[n]))
print(f"""
best flat:          {best_flat}  (mean {np.mean(rows[best_flat]):.3f})
best parent-child:  {best_pc}  (mean {np.mean(rows[best_pc]):.3f})

Look at the flat rows first and you can see ch:rag-chunking's dilemma laid out:
L=1 is unbeatable on fact queries and scores ZERO on w=6, because a one-sentence
chunk cannot contain a six-sentence answer. L=12 is the reverse. Every flat row
is good at one end of the table and bad at the other, and the best flat
compromise is mediocre everywhere.

The parent-child rows win the mean decisively, and they do it in the way
eq:pc-dominance predicts: they hold the fact column at 1.000 -- which only the
tiny flat chunks manage -- while scoring on synthesis queries, which those tiny
chunks cannot do at all. Two jobs, two units, no compromise between them.

Note that this is at an EQUAL CONTEXT BUDGET. Parent-child sends larger units, so
it sends fewer of them (eq:parent-child-budget), and it still wins. It affords
that through deduplication: when several retrieved children fall in one parent,
that parent is sent once -- and child clustering is itself evidence the parent is
relevant, so the discount arrives exactly when the technique is most likely to be
right.

But read the w=6 column before concluding the technique dominates everywhere,
because it does not. Flat L=12 beats every parent-child configuration there, and
the reason is a coupling eq:pc-dominance does not model. That equation treats
R(L_c) as maximised at the smallest child, which is true for a NARROW query and
false for a wide one: a question whose answer spans six sentences is itself a
broad object, and a two-sentence child is a poor key for it. Retrieval ranks
children by similarity, and a small child simply does not look like a wide query.

So the honest statement is narrower than the clean one. Decoupling removes the
containment constraint entirely -- that part is unconditional, and it is why the
fact and w=3 columns are so strong. It does NOT remove the requirement that the
indexed unit be a good key at the query's granularity, and when queries vary
widely in span, one child size cannot serve all of them.

Which is an argument for indexing at SEVERAL child sizes rather than one, and
that is exactly what hierarchical indexing does (eq:hierarchical-index-size).
This listing is the motivation for the next technique rather than a refutation of
this one: parent-child fixes the completeness half of ch:rag-chunking's dilemma
outright and leaves a residue of the retrieval half, which multi-granularity
indexing addresses at a cost eq:hierarchical-build-cost prices.""")

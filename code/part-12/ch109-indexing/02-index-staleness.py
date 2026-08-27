# -*- coding: utf-8 -*-
# Extracted from: Chapter 109 — Indexing, Metadata, and Retrieval
# Source: src/.../ch109-indexing.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""How stale is the index between rebuilds, and why does k make it worse?

eq:stale-query-rate says the probability a query touches stale content grows with
the retrieval depth k, because each retrieved chunk is an independent chance to
hit something that changed since the last rebuild.

We simulate a corpus with per-document churn against several rebuild cadences,
and separately track the two failure kinds of eq:deletion-asymmetry: serving an
OUTDATED version, and serving a DELETED document.
"""
import numpy as np

rng = np.random.default_rng(23)

N_DOC, N_QUERY, DAYS = 20_000, 4000, 90
CHURN_PER_DOC_PER_DAY = 0.004        # ~11% of the corpus changes per month
DELETE_FRACTION = 0.15               # of changes, this share are deletions
CADENCES = [1, 7, 30]                # rebuild every N days
DEPTHS = [1, 4, 8, 16]

print(f"corpus {N_DOC:,} docs, churn {CHURN_PER_DOC_PER_DAY:.3%}/doc/day "
      f"({N_DOC * CHURN_PER_DOC_PER_DAY:,.0f} docs/day)\n")
print(f"{'rebuild':>9}{'k':>5}{'stale answer':>15}{'  of which deleted':>20}"
      f"{'predicted (eq)':>17}")
print("-" * 66)

for cadence in CADENCES:
    for k in DEPTHS:
        stale_hits, deleted_hits = 0, 0
        for _ in range(N_QUERY):
            # Uniformly random point in the rebuild cycle.
            age_days = rng.random() * cadence
            p_changed = 1 - (1 - CHURN_PER_DOC_PER_DAY) ** age_days
            changed = rng.random(k) < p_changed
            if changed.any():
                stale_hits += 1
                # Of the changed chunks retrieved, were any deletions?
                if (rng.random(int(changed.sum())) < DELETE_FRACTION).any():
                    deleted_hits += 1
        rate = stale_hits / N_QUERY
        del_rate = deleted_hits / N_QUERY
        predicted = 1 - (1 - CHURN_PER_DOC_PER_DAY * cadence / 2) ** k
        print(f"{cadence:>7}d{k:>5}{rate:>15.3f}{del_rate:>20.3f}"
              f"{predicted:>17.3f}")
    print()

print("""
The measured column tracks eq:stale-query-rate closely, which is the point of
printing both -- the model is simple enough to reason with, so use it.

Read across k at a fixed cadence. Retrieval depth multiplies the staleness rate
almost linearly, because every retrieved chunk is another chance to touch
something that has changed. This is a genuine coupling that nobody plans for:
raising k to improve recall (ch:rag-chunking) raises the stale-answer rate
proportionally, so the two decisions are not independent and a recall improvement
can be a freshness regression.

Now read the deleted column. It is a fraction of the stale column, and it is the
one that matters, because of eq:deletion-asymmetry. A stale UPDATE gives a user
an outdated number. A stale DELETION gives them a confident, cited answer from a
document that has been retracted -- and the citation makes it MORE credible, not
less. The harm is not proportional to the rate.

Which gives the architectural conclusion: the rebuild cadence is set by the
update rate and the tolerance for outdated content, but deletions should not wait
for it at all. A tombstone stream applied within seconds costs almost nothing
next to a full rebuild and removes the worst failure mode entirely.""")

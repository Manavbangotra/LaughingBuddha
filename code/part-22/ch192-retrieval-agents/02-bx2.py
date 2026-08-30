# -*- coding: utf-8 -*-
# Extracted from: Chapter 192 — Retrieval and Agent Architecture at Scale
# Source: src/.../ch192-retrieval-agents.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""At scale, retrieval gets better by its own metrics and worse at its job.

A question needs several distinct facts to answer. A corpus contains documents, and
as the corpus grows the popular facts get restated many times while the rare ones
stay rare. Similarity search ranks by resemblance to the query, and resemblance
tracks popularity.

So a bigger corpus makes the top-k MORE redundant: the same well-covered facts,
restated. Precision rises, because there are more relevant documents and every slot
fills with genuinely on-topic material. Fact coverage falls, because the slots are
spent on repetition (eq:scale-buys-redundancy-not-coverage).

This is ch:sd-architecture's pattern again -- a metric that is accurate about its own
quantity and silent about the one that decides whether the answer is right.
"""
import math

F = 24                 # distinct facts a full answer needs
FACTS_PER_DOC = 3.0    # facts carried by a typical retrieved document
N0 = 1000              # reference corpus size
CORPORA = [1000, 10000, 100000, 1000000, 10000000]
KS = [3, 5, 10, 20, 40]

# Fact prevalence: Zipf. Fact 1 is discussed everywhere, fact 24 almost nowhere.
PREV = [1.0 / (i + 1) for i in range(F)]
_z = sum(PREV)
PREV = [p / _z for p in PREV]


def selectivity(n):
    """How sharply retrieval concentrates on popular facts, by corpus size.

    A larger corpus gives similarity search more near-duplicates to choose from,
    so the top-k skews further toward whatever is most commonly discussed.
    """
    return 1.0 + 0.42 * math.log10(float(n) / N0)


def per_doc_probs(n):
    """P(a retrieved document carries fact f), for each f."""
    s = selectivity(n)
    w = [p ** s for p in PREV]
    tot = sum(w)
    return [FACTS_PER_DOC * x / tot for x in w]


def coverage(n, k):
    """Expected share of the F needed facts present anywhere in the top-k."""
    q = per_doc_probs(n)
    got = 0.0
    for qi in q:
        qi = min(qi, 1.0)
        got += 1.0 - (1.0 - qi) ** k
    return got / F


def precision_at_k(n, k, relevant_rate=0.0004):
    """Share of the top-k that are topically relevant -- the metric a retrieval
    evaluation usually reports, and the one that improves with scale.

    A larger corpus contains more relevant documents, so the top-k fills up with
    genuinely on-topic material instead of padding.
    """
    relevant = n * relevant_rate
    return min(1.0, relevant / k)


print("A question needing %d distinct facts. Fact prevalence is Zipf: the most" % F)
print("discussed fact appears %.0fx more often than the least."
      % (PREV[0] / PREV[-1]))
print()
print(f"{'corpus size':>14}{'selectivity':>14}{'P(top doc has fact 1)':>24}"
      f"{'P(has fact 24)':>17}")
print("-" * 69)
sel = {}
for n in CORPORA:
    q = per_doc_probs(n)
    sel[n] = (selectivity(n), q[0], q[-1])
    print(f"{n:>14,}{selectivity(n):>14.2f}{min(q[0], 1.0):>24.3f}"
          f"{q[-1]:>17.5f}")

print()
print()
print("Fact coverage at k=10, as the corpus grows. The retriever is unchanged;")
print("only the corpus is bigger.")
print()
K = 10
print(f"{'corpus size':>14}{'fact coverage':>16}{'precision@10':>15}"
      f"{'what a report says':>21}")
print("-" * 69)
main = {}
for n in CORPORA:
    cov = coverage(n, K)
    rec = precision_at_k(n, K)
    main[n] = (cov, rec)
    verdict = "improving" if rec >= main[CORPORA[0]][1] else "degrading"
    print(f"{n:>14,}{cov:>16.1%}{rec:>15.1%}{verdict:>21}")

print()
print()
print("The same across retrieval depth. More slots help, but they are spent on")
print("increasingly redundant documents.")
print()
print(f"{'corpus size':>14}" + "".join(f"{('k=%d' % k):>10}" for k in KS))
print("-" * 64)
grid = {}
for n in CORPORA:
    row = [coverage(n, k) for k in KS]
    grid[n] = row
    print(f"{n:>14,}" + "".join(f"{v:>10.1%}" for v in row))

print()
print()
print("How much retrieval depth it takes to hold coverage flat as the corpus")
print("grows -- and whether a context budget can pay for it.")
print()
TARGET = grid[CORPORA[0]][2]      # coverage achieved at k=10 on the small corpus
print(f"holding coverage at {TARGET:.1%}, the level a 1,000-document corpus")
print("reaches with k=10:")
print()
print(f"{'corpus size':>14}{'k needed':>11}{'vs k=10':>10}"
      f"{'context tokens':>17}{'feasible':>11}")
print("-" * 64)
TOK_PER_DOC = 420
BUDGET_TOK = 12000
need = {}
for n in CORPORA:
    kk = None
    for k in range(1, 4001):
        if coverage(n, k) >= TARGET:
            kk = k
            break
    need[n] = kk
    if kk is None:
        print(f"{n:>14,}{'never':>11}{'--':>10}{'--':>17}{'no':>11}")
    else:
        tok = kk * TOK_PER_DOC
        print(f"{n:>14,}{kk:>11}{kk / 10.0:>9.1f}x{tok:>17,}"
              f"{('yes' if tok <= BUDGET_TOK else 'no'):>11}")

print()
print()
print("And the alternative: deduplicate by fact before filling the context.")
print("Same slots, spent on distinct content instead of the top of the ranking.")
print()
print(f"{'corpus size':>14}{'plain k=10':>13}{'deduped k=10':>15}"
      f"{'gain':>9}{'equivalent plain k':>21}")
print("-" * 72)


def deduped_coverage(n, k):
    """Fill k slots, skipping a document whose facts are already covered.

    Modelled by drawing from the residual: each slot targets the highest-
    prevalence fact not yet covered, which is what a diversity reranker does.
    """
    q = per_doc_probs(n)
    covered = [0.0] * F
    for _ in range(k):
        # The slot goes to the document most likely to add something new.
        gains = [min(q[i], 1.0) * (1.0 - covered[i]) for i in range(F)]
        j = max(range(F), key=lambda i: gains[i])
        # That document carries its target fact plus incidental others.
        for i in range(F):
            add = min(q[i], 1.0) if i != j else 1.0
            covered[i] = covered[i] + (1.0 - covered[i]) * add
    return sum(covered) / F


ded = {}
for n in CORPORA:
    plain = coverage(n, 10)
    dd = deduped_coverage(n, 10)
    eq = None
    for k in range(1, 4001):
        if coverage(n, k) >= dd:
            eq = k
            break
    ded[n] = (plain, dd, eq)
    print(f"{n:>14,}{plain:>13.1%}{dd:>15.1%}{dd - plain:>+9.1%}"
          f"{(str(eq) if eq else 'never'):>21}")

print(f"""
The selectivity table is the mechanism. At {CORPORA[0]:,} documents a retrieved
document carries the rarest fact with probability {sel[CORPORA[0]][2]:.5f}. At
{CORPORA[-1]:,} documents that has fallen to {sel[CORPORA[-1]][2]:.5f}, because a
larger corpus offers similarity search more near-duplicates of the popular material
and it takes them.

Nothing about the retriever changed. The corpus got bigger, which every intuition
says should help, and the rare facts got harder to reach.

The consequence is the second table, and it is the finding. At k={K}, growing the
corpus from {CORPORA[0]:,} to {CORPORA[-1]:,} documents moves fact coverage from
{main[CORPORA[0]][0]:.1%} to {main[CORPORA[-1]][0]:.1%} -- **down
{(main[CORPORA[0]][0] - main[CORPORA[-1]][0]) * 100:.0f} points** -- while
precision@10 rises from {main[CORPORA[0]][1]:.0%} to {main[CORPORA[-1]][1]:.0%}
(eq:scale-buys-redundancy-not-coverage).

**The two metrics move in opposite directions on the same system.** A retrieval
evaluation reporting precision@10 sees a system improving from
{main[CORPORA[0]][1]:.0%} to {main[CORPORA[-1]][1]:.0%} as the corpus grows, which
reads as a clear success, and every document it returns really is on topic. The
answers built from those documents are getting worse, because the top-k has become
{1 - main[CORPORA[-1]][0]:.0%} redundant.

The depth grid shows why buying more slots does not rescue it. On the
{CORPORA[-1]:,}-document corpus, going from k={KS[0]} to k={KS[-1]} -- more than a
tenfold increase in retrieved context -- moves coverage from
{grid[CORPORA[-1]][0]:.1%} to {grid[CORPORA[-1]][-1]:.1%}. The extra slots are
filled with more copies of what was already there.

The feasibility table prices that directly. Holding coverage at {TARGET:.1%} --
the level the small corpus reaches at k=10 -- needs k={need[CORPORA[1]]} at
{CORPORA[1]:,} documents and k={need[CORPORA[2]]} at {CORPORA[2]:,}. At
{CORPORA[2]:,} documents that is {need[CORPORA[2]] * TOK_PER_DOC:,} tokens of
context, against a {BUDGET_TOK:,}-token budget.

**Scale defeats retrieval depth well before it defeats the context window**, and
ch:mcp-schemas's context budget is the binding constraint.

The last table is the lever that works. Spending the same ten slots on distinct
content rather than on the top of the ranking moves coverage at {CORPORA[-1]:,}
documents from {ded[CORPORA[-1]][0]:.1%} to {ded[CORPORA[-1]][1]:.1%} -- a gain of
{(ded[CORPORA[-1]][1] - ded[CORPORA[-1]][0]) * 100:.0f} points from **reordering,
not from retrieving more**.

That is the architectural conclusion, and it is a specific one. As a corpus grows,
the marginal return on a better retriever and on a larger context window both fall,
while the marginal return on **diversity-aware selection** rises. The system's
bottleneck migrates from finding relevant documents to choosing which relevant
documents to spend the budget on -- and those are different problems with different
owners, different metrics, and usually different teams.""")

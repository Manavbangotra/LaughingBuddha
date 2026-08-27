# -*- coding: utf-8 -*-
# Extracted from: Chapter 106 — Why RAG Exists
# Source: src/.../ch106-why-rag.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""When is retrieval cheaper than stuffing, and by how much?

Compares two architectures across corpus sizes at a fixed query volume:

  stuff -- put the entire corpus in the context on every query
  rag   -- embed the corpus once, retrieve k chunks per query

Prices are set as constants at the top and are the only thing that dates. The
STRUCTURE of the comparison (eq:cost-ratio) does not depend on them, which is the
point the table is meant to make.
"""

PRICE_INPUT = 3.00 / 1_000_000      # $ per input token to the generator
PRICE_EMBED = 0.02 / 1_000_000      # $ per token to embed, one pass
INDEX_GB_MONTH = 0.25               # $ per GB-month for the vector index
BYTES_PER_VECTOR = 768 * 4          # float32, before any compression

CHUNK_TOKENS = 500
K_RETRIEVED = 8
QUERIES_PER_MONTH = 100_000

CORPUS_SIZES = [10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]


def stuffing_cost(corpus_tokens, queries):
    """Every query pays for the whole corpus: linear in corpus size PER QUERY."""
    return queries * corpus_tokens * PRICE_INPUT


def rag_cost(corpus_tokens, queries):
    """Embedding is paid once; the per-query term is constant in corpus size."""
    n_chunks = corpus_tokens / CHUNK_TOKENS
    embed = corpus_tokens * PRICE_EMBED
    index_gb = n_chunks * BYTES_PER_VECTOR / 1e9
    index = index_gb * INDEX_GB_MONTH
    generate = queries * K_RETRIEVED * CHUNK_TOKENS * PRICE_INPUT
    return embed + index + generate, embed, index, generate


print(f"assumptions: {QUERIES_PER_MONTH:,} queries/month, k={K_RETRIEVED} chunks "
      f"of {CHUNK_TOKENS} tokens, ${PRICE_INPUT * 1e6:.2f}/M input tokens\n")
print(f"{'corpus tokens':>15}{'stuff $/mo':>14}{'RAG $/mo':>12}"
      f"{'ratio':>10}{'RAG: embed/index/gen':>28}")
print("-" * 80)

for corpus in CORPUS_SIZES:
    stuff = stuffing_cost(corpus, QUERIES_PER_MONTH)
    total, embed, index, gen = rag_cost(corpus, QUERIES_PER_MONTH)
    ratio = stuff / total
    print(f"{corpus:>15,}{stuff:>14,.0f}{total:>12,.0f}{ratio:>9.1f}x"
          f"   {embed:>7,.1f} /{index:>7,.1f} /{gen:>8,.0f}")

# Where do the two meet?
lo, hi = 1.0, 1e9
for _ in range(200):
    mid = (lo + hi) / 2
    if stuffing_cost(mid, QUERIES_PER_MONTH) < rag_cost(mid, QUERIES_PER_MONTH)[0]:
        lo = mid
    else:
        hi = mid
breakeven = (lo + hi) / 2

print(f"""
Break-even corpus size: {breakeven:,.0f} tokens -- which is exactly
k * chunk_tokens = {K_RETRIEVED} * {CHUNK_TOKENS}, and that is not a coincidence.
Below that size RAG would retrieve the entire corpus anyway, so the two
architectures ARE the same thing and there is nothing to compare.

The honest conclusion is therefore stronger and less interesting than the usual
one: on cost, RAG wins essentially as soon as the corpus exceeds the context you
would retrieve. There is no meaningful cost regime in which stuffing is the
cheaper choice.

So COST IS NOT THE REASON TO STUFF A SMALL CORPUS. The reasons are simplicity and
quality -- no ingestion, no chunking, no retrieval, none of the four failure
stages in fig:rag-stages, and the finding in cite:li2024ragvslongcontext that
long context is BETTER when everything fits. Those are good reasons. "It is
cheaper" is not one, and teams that justify stuffing on cost grounds are usually
right for the wrong reason.

Above the break-even the gap opens fast, because eq:cost-ratio says the ratio
grows as corpus/(k * chunk) and nothing in that expression is a price. At 10M
tokens the factor is {stuffing_cost(10_000_000, QUERIES_PER_MONTH) / rag_cost(10_000_000, QUERIES_PER_MONTH)[0]:,.0f}x.
Halving the price of tokens halves BOTH columns and moves the ratio not at all.

Finally read the RAG breakdown columns and note which term dominates: generation,
by three orders of magnitude over embedding and index at every corpus size here.
The one-time embedding cost teams agonise over is a rounding error, and the
index storage is smaller still. The only lever that matters is k -- the number of
chunks put in the context on every query -- which is a quality decision
(eq:context-budget) that turns out to be the cost decision too.""")

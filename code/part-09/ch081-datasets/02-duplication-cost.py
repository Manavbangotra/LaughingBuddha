# Extracted from: Chapter 81 — Pretraining Dataset Construction and Curation
# Source: src/.../ch081-datasets.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""What duplication does to the training budget and to the objective."""
import numpy as np

rng = np.random.default_rng(1)

N_UNIQUE = 10_000
DUP_FRACTION = 0.30          # share of documents that are duplicated
DUP_COPIES = 10              # how many times each appears

# Build a corpus: most documents appear once, a subset appears many times.
n_dup_docs = int(N_UNIQUE * DUP_FRACTION)
counts = np.ones(N_UNIQUE, dtype=int)
counts[:n_dup_docs] = DUP_COPIES

total_docs = counts.sum()
unique_fraction = N_UNIQUE / total_docs
print(f"{N_UNIQUE:,} unique documents, {n_dup_docs:,} of them duplicated "
      f"{DUP_COPIES}x")
print(f"corpus as stored : {total_docs:,} documents")
print(f"distinct docs / stored docs : {unique_fraction:.3f}")
print(f"reads spent re-reading      : {1 - unique_fraction:.1%} of the budget\n")

# Equation (eq:duplication-reweighting): the effective objective weights.
weights = counts / counts.sum()
dup_share = weights[:n_dup_docs].sum()
print(f"duplicated documents are {n_dup_docs / N_UNIQUE:.0%} of unique content")
print(f"but receive {dup_share:.0%} of the gradient signal")
print(f"over-representation factor: {dup_share / (n_dup_docs / N_UNIQUE):.1f}x\n")

# Memorisation proxy: exposures determine how strongly a document is imprinted.
print(f"{'document class':<22} {'exposures':>10} {'relative imprint':>18}")
print(f"{'unique':<22} {1:>10} {1.0:>18.1f}x")
print(f"{'duplicated':<22} {DUP_COPIES:>10} {float(DUP_COPIES):>18.1f}x")

# What deduplication actually frees. It does not create data.
freed = total_docs - N_UNIQUE
print(f"At a fixed budget of {total_docs:,} document-reads:")
print(f"  with duplicates : {N_UNIQUE:,} distinct documents seen, "
      f"{freed:,} reads re-reading")
print(f"  after dedup     : the same {N_UNIQUE:,} seen in {N_UNIQUE:,} reads, "
      f"{freed:,} freed")
print(f"  those freed reads buy new content only if the corpus HAS more unique "
      f"documents.\n  Deduplication frees budget; it does not create data.")

print("""
Two effects, and the second is the one that matters. Deduplication frees budget
(the first effect, and it is merely a saving). It also removes an unintended
reweighting of the objective — equation (eq:duplication-reweighting) — that was
pushing the model toward whatever the web happens to republish most:
boilerplate, licences, syndicated articles. That is why removing duplicates
improves quality rather than just costing less.""")

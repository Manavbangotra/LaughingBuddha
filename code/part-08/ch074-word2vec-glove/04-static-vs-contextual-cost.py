# Extracted from: Chapter 74 — Static Word Embeddings: Word2Vec and GloVe
# Source: src/.../ch074-word2vec-glove.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The cost side of the static-versus-contextual decision, made explicit."""

QUERIES_PER_DAY = 40_000_000
LATENCY_BUDGET_MS = 5.0

# Averaged static vectors: one lookup and one add per token.
STATIC = dict(dim=300, params=0, flops_per_token=300, ms_per_query=0.02)

# A small transformer encoder: 2·N FLOPs per token (see ch:tf-complexity).
BERT_BASE = dict(dim=768, params=110e6, flops_per_token=2 * 110e6, ms_per_query=8.0)
MINILM = dict(dim=384, params=22e6, flops_per_token=2 * 22e6, ms_per_query=1.6)

TOKENS = 8          # a short query
GPU_COST_PER_HOUR = 2.0

print(f"{'model':<12} {'dim':>5} {'MFLOPs/query':>13} {'ms':>7} "
      f"{'fits 5ms':>9} {'GPU-hours/day':>14}")
for name, m in [("static-avg", STATIC), ("MiniLM", MINILM), ("BERT-base", BERT_BASE)]:
    mflops = m["flops_per_token"] * TOKENS / 1e6
    gpu_hours = QUERIES_PER_DAY * m["ms_per_query"] / 1000 / 3600
    print(f"{name:<12} {m['dim']:>5} {mflops:>13,.1f} {m['ms_per_query']:>7.2f} "
          f"{str(m['ms_per_query'] < LATENCY_BUDGET_MS):>9} {gpu_hours:>14,.0f}")

ratio = BERT_BASE["flops_per_token"] / STATIC["flops_per_token"]
print(f"\nBERT-base does {ratio:,.0f}x the arithmetic per token of a lookup-and-add.")
print(f"At {QUERIES_PER_DAY:,} queries/day the difference between MiniLM and "
      f"BERT-base alone is "
      f"${QUERIES_PER_DAY * (BERT_BASE['ms_per_query'] - MINILM['ms_per_query']) / 1000 / 3600 * GPU_COST_PER_HOUR:,.0f}"
      f"/day of GPU time.")
print("\nStatic embeddings are not the best representation. They are sometimes "
      "the only one that fits the budget — and that is an engineering answer, "
      "not a quality claim.")

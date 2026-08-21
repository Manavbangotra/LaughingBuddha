# Extracted from: Chapter 78 — Semantic Similarity and Sentence Embeddings
# Source: src/.../ch078-similarity.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Bi-encoder, cross-encoder, and the cascade. Equations (eq:cross-cost) onward."""

CORPUS = 1_000_000
QUERIES_PER_DAY = 10_000
ENCODER_MS = 10.0            # one forward pass over a 128-token pair
DOT_NS = 1.0                 # one 768-dimensional dot product
TOP_K = 100

enc_s = ENCODER_MS / 1000
dot_s = DOT_NS / 1e9

cross_query = CORPUS * enc_s
bi_query = enc_s + CORPUS * dot_s
cascade_query = (TOP_K + 1) * enc_s + CORPUS * dot_s

SECONDS_PER_DAY = 86_400


def human(seconds):
    for unit, size in [("years", 365 * 86400), ("days", 86400),
                       ("hours", 3600), ("minutes", 60)]:
        if seconds >= size:
            return f"{seconds / size:,.1f} {unit}"
    return f"{seconds:,.2f} seconds"


print(f"corpus {CORPUS:,}  queries/day {QUERIES_PER_DAY:,}  "
      f"encoder {ENCODER_MS} ms  top-k {TOP_K}\n")
print(f"{'architecture':<18} {'per query':>14} {'compute per day':>20} "
      f"{'index build':>14}")
for name, per_query, build in [
        ("full cross-encoder", cross_query, 0.0),
        ("bi-encoder only", bi_query, CORPUS * enc_s),
        ("cascade", cascade_query, CORPUS * enc_s)]:
    print(f"{name:<18} {human(per_query):>14} "
          f"{human(per_query * QUERIES_PER_DAY):>20} {human(build):>14}")

print(f"\ncost ratio, equation (eq:cost-ratio): "
      f"{enc_s / dot_s:,.0f}x  (a forward pass against a dot product)")
print(f"cascade saving over full cross-encoder: "
      f"{cross_query / cascade_query:,.0f}x")
print(f"real-time feasibility at 1 query: "
      f"cross {cross_query:,.0f} s, cascade {cascade_query:.2f} s, "
      f"bi {bi_query * 1000:.1f} ms")

# The recall ceiling of equation (eq:recall-ceiling).
print(f"\n{'recall@k of the retriever':<28} {'cascade accuracy ceiling':>26}")
for recall in [0.80, 0.90, 0.95, 0.99]:
    print(f"{recall:<28.2f} {recall:>26.2f}")
print("\nNo reranker can exceed the retriever's recall@k — which is why the "
      "first stage is tuned for recall and never for precision@1.")

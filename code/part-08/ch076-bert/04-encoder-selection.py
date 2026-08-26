# -*- coding: utf-8 -*-
# Extracted from: Chapter 76 — BERT, RoBERTa, and Masked Language Modeling
# Source: src/.../ch076-bert.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Encoder selection under a latency budget, with the multilingual tradeoff priced."""

BUDGET_MS, RPS, NON_ENGLISH = 50.0, 30, 0.30

# Representative figures, not verified benchmark results: parameter counts are
# exact, GLUE scores are approximate, and latencies stand in for a 128-token
# sequence on one CPU core. Substitute your own measurements before deciding
# anything — the listing is the shape of the argument, not a source of numbers.
CANDIDATES = {
    "BERT-base":      dict(params=110e6, layers=12, latency_ms=42, glue=79.6, multi=False),
    "RoBERTa-base":   dict(params=125e6, layers=12, latency_ms=44, glue=83.2, multi=False),
    "DistilBERT":     dict(params=66e6,  layers=6,  latency_ms=21, glue=77.0, multi=False),
    "XLM-R-base":     dict(params=270e6, layers=12, latency_ms=48, glue=80.4, multi=True),
}

print(f"{'model':<15} {'params':>9} {'ms':>6} {'GLUE':>6} {'fits 50ms':>10} "
      f"{'cores@30rps':>12} {'non-EN':>8}")
for name, c in CANDIDATES.items():
    fits = c["latency_ms"] < BUDGET_MS
    cores = RPS * c["latency_ms"] / 1000
    print(f"{name:<15} {c['params'] / 1e6:>8.0f}M {c['latency_ms']:>6} "
          f"{c['glue']:>6.1f} {str(fits):>10} {cores:>12.1f} "
          f"{('yes' if c['multi'] else 'no'):>8}")

print()
best_glue = max(CANDIDATES, key=lambda k: CANDIDATES[k]["glue"])
cheapest = min(CANDIDATES, key=lambda k: CANDIDATES[k]["latency_ms"])
print(f"best GLUE:     {best_glue} ({CANDIDATES[best_glue]['glue']})")
print(f"lowest latency:{cheapest} ({CANDIDATES[cheapest]['latency_ms']} ms, "
      f"{CANDIDATES[best_glue]['glue'] - CANDIDATES[cheapest]['glue']:.1f} GLUE points behind)")

# The 30% of non-English traffic cannot be served by a monolingual model at all.
english_only_coverage = 1 - NON_ENGLISH
print(f"\nA monolingual model serves {english_only_coverage:.0%} of traffic. "
      f"The remaining {NON_ENGLISH:.0%} needs either XLM-R or a second model.")

# Two-model cascade: DistilBERT for English, XLM-R for the rest.
cascade_ms = english_only_coverage * CANDIDATES["DistilBERT"]["latency_ms"] \
    + NON_ENGLISH * CANDIDATES["XLM-R-base"]["latency_ms"]
print(f"cascade (DistilBERT for EN, XLM-R otherwise): "
      f"{cascade_ms:.1f} ms weighted mean, "
      f"{RPS * cascade_ms / 1000:.1f} cores — versus "
      f"{RPS * CANDIDATES['XLM-R-base']['latency_ms'] / 1000:.1f} for XLM-R alone.")
print("\nTwo models cost more to operate than one. That is the real decision, "
      "and no benchmark column contains it.")

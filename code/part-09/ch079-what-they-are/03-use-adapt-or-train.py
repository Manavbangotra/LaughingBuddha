# Extracted from: Chapter 79 — What Foundation Models Are
# Source: src/.../ch079-what-they-are.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Choosing an adaptation: what each option can actually deliver."""

REQUIREMENTS = {
    "answer from our current docs":      "knowledge",
    "keep up when docs change weekly":   "knowledge-freshness",
    "reply in our house voice":          "style",
    "always emit our JSON ticket schema": "format",
    "handle our product's jargon":       "vocabulary",
    "cost under $0.01 per conversation": "cost",
}

# What each adaptation is actually good at. Grounded in
# eq:adaptation-information-ratio: adaptation reshapes, retrieval supplies.
CAPABILITY = {
    "prompting only":      {"style": 0.6, "format": 0.7, "vocabulary": 0.4,
                            "knowledge": 0.1, "knowledge-freshness": 0.1, "cost": 0.9},
    "prompt + retrieval":  {"style": 0.6, "format": 0.7, "vocabulary": 0.8,
                            "knowledge": 0.9, "knowledge-freshness": 0.95, "cost": 0.7},
    "fine-tune":           {"style": 0.95, "format": 0.95, "vocabulary": 0.8,
                            "knowledge": 0.3, "knowledge-freshness": 0.0, "cost": 0.8},
    "fine-tune + retrieval": {"style": 0.95, "format": 0.95, "vocabulary": 0.85,
                              "knowledge": 0.9, "knowledge-freshness": 0.95, "cost": 0.6},
}
THRESHOLD = 0.5      # a requirement counts as met above this

SETUP_COST = {"prompting only": 0, "prompt + retrieval": 15_000,
              "fine-tune": 40_000, "fine-tune + retrieval": 55_000}

print(f"{'requirement':<36} {'need':<20}")
for req, need in REQUIREMENTS.items():
    print(f"  {req:<34} {need:<20}")

print(f"\n{'approach':<24} {'worst requirement':<24} {'score':>7} {'setup':>10}")
rows = []
for approach, caps in CAPABILITY.items():
    scores = {need: caps[need] for need in REQUIREMENTS.values()}
    weakest = min(scores, key=scores.get)
    rows.append((approach, weakest, scores[weakest], SETUP_COST[approach]))
    print(f"{approach:<24} {weakest:<24} {scores[weakest]:>7.2f} "
          f"${SETUP_COST[approach]:>9,}")

viable = [r for r in rows if r[2] >= THRESHOLD]
best = min(viable, key=lambda r: r[3]) if viable else None
print(f"\nA system is only as good as its weakest requirement, so the column "
      f"to read is the minimum, not the average.")
if best:
    print(f"Cheapest option clearing every requirement: {best[0]} "
          f"(${best[3]:,} setup)")
else:
    print("No option clears every requirement — the requirements need cutting.")

print("""
Note what the table says about fine-tuning alone: it scores highest on style
and format and fails outright on knowledge freshness, because weights are
frozen at training time and the docs change weekly. No amount of fine-tuning
fixes that — it is a property of where the information lives, which is
equation (eq:adaptation-information-ratio) showing up as a product decision.""")

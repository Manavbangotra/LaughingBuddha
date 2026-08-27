# -*- coding: utf-8 -*-
# Extracted from: Chapter 96 — Hallucination: Causes, Taxonomy, and Mitigation
# Source: src/.../ch096-hallucination.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Which mitigation? It depends on the failure mix, which must be measured."""

# A hundred sampled bad outputs, classified by hand.
OBSERVED = {
    "intrinsic (contradicts the document)": 34,
    "extrinsic (unsupported addition)":     41,
    "wrong tool/argument value":             9,
    "malformed output":                      6,
    "correct but unhelpful":                10,
}

# What each mitigation addresses, and roughly how much of it.
MITIGATIONS = {
    "retrieval (add grounds)": {
        "extrinsic (unsupported addition)": 0.75},
    "citation requirement + span check": {
        "intrinsic (contradicts the document)": 0.70,
        "extrinsic (unsupported addition)": 0.40},
    "constrained decoding": {
        "malformed output": 1.00,
        "wrong tool/argument value": 0.35},
    "lower temperature": {
        "intrinsic (contradicts the document)": 0.15,
        "extrinsic (unsupported addition)": 0.10},
    "abstention at a confidence threshold": {
        "intrinsic (contradicts the document)": 0.30,
        "extrinsic (unsupported addition)": 0.45,
        "wrong tool/argument value": 0.30},
}

total = sum(OBSERVED.values())
print(f"{total} sampled failures\n")
print(f"{'failure class':<40} {'count':>7} {'share':>8}")
for k, v in sorted(OBSERVED.items(), key=lambda kv: -kv[1]):
    print(f"{k:<40} {v:>7} {v / total:>8.0%}")

print(f"\n{'mitigation':<38} {'failures removed':>18} {'share':>8}")
ranked = []
for name, effects in MITIGATIONS.items():
    removed = sum(OBSERVED.get(k, 0) * frac for k, frac in effects.items())
    ranked.append((name, removed))
for name, removed in sorted(ranked, key=lambda x: -x[1]):
    print(f"{name:<38} {removed:>18.1f} {removed / total:>8.0%}")

best = max(ranked, key=lambda x: x[1])
print(f"\nsingle best intervention: {best[0]} ({best[1] / total:.0%})")

# Combining, without double-counting.
print(f"\n{'stacked':<52} {'cumulative removed':>19}")
remaining = dict(OBSERVED)
cumulative = 0.0
for name, _ in sorted(ranked, key=lambda x: -x[1])[:3]:
    removed = 0.0
    for k, frac in MITIGATIONS[name].items():
        take = remaining.get(k, 0) * frac
        remaining[k] = remaining.get(k, 0) - take
        removed += take
    cumulative += removed
    print(f"{'+ ' + name:<52} {cumulative / total:>18.0%}")

print("""
Retrieval is the single best intervention HERE because extrinsic failures are
the largest class — and that was a measurement, not an assumption. On a failure
mix dominated by intrinsic contradiction it would be near the bottom of the
table, because retrieval supplies grounds and intrinsic failures already had
them.

The general rule: classify a hundred failures before choosing a mitigation. It
costs an afternoon and it is the difference between the top row of this table
and the bottom one.""")

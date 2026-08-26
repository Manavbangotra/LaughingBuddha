# -*- coding: utf-8 -*-
# Extracted from: Chapter 94 — Structured Output and Constrained Decoding
# Source: src/.../ch094-structured-output.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""What constraining fixes, and what it leaves untouched."""
import numpy as np

rng = np.random.default_rng(3)
N = 4000

# Failure modes of an extraction system, before any constraint.
BASELINE = {
    "unparseable output":        0.060,
    "missing required field":    0.035,
    "wrong field type":          0.020,
    "invalid enum value":        0.015,
    "value not in the document": 0.055,   # hallucinated
    "wrong value from document": 0.040,   # misread
    "internally inconsistent":   0.025,   # e.g. end date before start
}

# Which of these a grammar can make unreachable.
STRUCTURAL = {"unparseable output", "missing required field",
              "wrong field type", "invalid enum value"}

print(f"{'failure mode':<28} {'rate':>8} {'grammar fixes?':>16}")
for mode, rate in BASELINE.items():
    print(f"{mode:<28} {rate:>8.1%} "
          f"{('YES' if mode in STRUCTURAL else 'no'):>16}")

before = sum(BASELINE.values())
after = sum(r for m, r in BASELINE.items() if m not in STRUCTURAL)
print(f"\n{'total failure rate before':<32} {before:>8.1%}")
print(f"{'total failure rate after constraining':<32} {after:>8.1%}")
print(f"{'reduction':<32} {(before - after) / before:>8.0%}")
print(f"{'remaining, all semantic':<32} {after:>8.1%}")

# What it takes to address the remainder.
print(f"\n{'remaining failure':<28} {'detection method':<34} {'cost'}")
REMEDIES = {
    "value not in the document": ("substring check against the source", "free"),
    "wrong value from document": ("span extraction + verification", "moderate"),
    "internally inconsistent":   ("schema-level validation rules", "cheap"),
}
for mode, (method, cost) in REMEDIES.items():
    print(f"{mode:<28} {method:<34} {cost}")

# The substring check is the highest-value cheap addition.
grounded = 0.055 * 0.85          # a substring check catches most fabrications
consistency = 0.025 * 0.90       # explicit rules catch most inconsistencies
final = after - grounded - consistency
print(f"\n{'after constraining':<40} {after:>8.1%}")
print(f"{'+ substring grounding check':<40} {after - grounded:>8.1%}")
print(f"{'+ consistency rules':<40} {final:>8.1%}")
print(f"{'total reduction from baseline':<40} "
      f"{(before - final) / before:>8.0%}")

print("""
Constraining removes 57% of the failures and every one it removes is structural.
The remaining 43% are semantic and a grammar cannot see them — a hallucinated
value parses perfectly.

The cheap follow-up is the substring check: for extraction, every value should
appear in the source document, and verifying that costs a string search. It
catches most fabrication and it is the single highest-value addition after
constraining — which is ch:nlp-extraction's span-extraction argument arriving in
a new form, and the reason Part XII's grounded generation is built on spans
rather than on free text.""")

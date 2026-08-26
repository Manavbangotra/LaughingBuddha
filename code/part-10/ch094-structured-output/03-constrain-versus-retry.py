# -*- coding: utf-8 -*-
# Extracted from: Chapter 94 — Structured Output and Constrained Decoding
# Source: src/.../ch094-structured-output.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Constrained decoding against validate-and-retry. Equation (eq:retry-cost)."""
import numpy as np

GEN_COST = 1.0                 # cost of one generation, arbitrary units
CONSTRAINT_OVERHEAD = 0.02     # mask lookup and application, per generation

print(f"{'p(valid)':>9} {'k':>3} {'retry success':>15} {'retry E[gens]':>15} "
      f"{'constrained':>13} {'constrained cost':>18}")
for p in (0.99, 0.95, 0.85, 0.70, 0.50):
    for k in (1, 3):
        success = 1 - (1 - p) ** k
        expected = success / p                    # eq:retry-cost
        print(f"{p:>9.2f} {k:>3} {success:>15.4f} {expected:>15.2f} "
              f"{1.0:>13.4f} {1 + CONSTRAINT_OVERHEAD:>18.2f}")

print("""
The 'constrained' column is 1.0000 at every row — the guarantee does not depend
on the model's cooperativeness, which is the whole point. And its cost is one
generation plus a small overhead, regardless.

Retry never reaches 1.0. At p=0.70 with three attempts, 2.7% of requests still
fail after paying for 1.39 generations on average — and those failures arrive as
user-visible errors rather than as degraded output.""")

# When is retry the right choice anyway?
print(f"\n{'situation':<44} {'choose'}")
CASES = [
    ("serving stack supports grammars, p is low", "constrain"),
    ("serving stack supports grammars, p > 0.99", "constrain (free)"),
    ("hosted API with no grammar support", "retry + validate"),
    ("schema changes per request", "retry, or build the index per call"),
    ("output must be semantically checked anyway", "both"),
]
for case, choice in CASES:
    print(f"{case:<44} {choice}")

print("""
The fourth row is the real constraint on adoption. Building the vocabulary index
(eq:vocabulary-index) is O(|Q|.|V|) and worth caching, so a schema that varies
per request either pays that cost per call or falls back to retry. Systems with
a small fixed set of schemas get constrained decoding almost free; systems
generating schemas dynamically do not.

And the last row is the one to internalise: a grammar guarantees STRUCTURE.
Whether the values are correct is a separate check that constrained decoding
does not perform and cannot.""")

# -*- coding: utf-8 -*-
# Extracted from: Chapter 95 — Function Calling and Tool Use
# Source: src/.../ch095-function-calling.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Multi-round reliability, and what error feedback buys. Eq (eq:tool-chain-success)."""
import numpy as np

print(f"{'per-round p':>12} " + " ".join(f"{'k=' + str(k):>8}" for k in
                                          (1, 2, 3, 5, 10, 20)))
for p in (0.99, 0.97, 0.95, 0.90, 0.80):
    row = " ".join(f"{p ** k:>8.3f}" for k in (1, 2, 3, 5, 10, 20))
    print(f"{p:>12.2f} {row}")

print("""
Read across a row. At 95% per-round reliability — which sounds excellent — a
ten-round task succeeds 60% of the time and a twenty-round task 36%. This is
eq:exact-match-composition from the emergence chapter, and it is why long
autonomous chains are hard in a way that per-step benchmarks never reveal.""")

# What error feedback does — equation in section 6.4.
print(f"\n{'rounds':>7} {'no feedback':>13} {'with feedback':>15} {'gain':>8}")
P_BASE, P_RECOVER = 0.93, 0.70
p_effective = P_BASE + (1 - P_BASE) * P_RECOVER
for k in (1, 2, 4, 8, 16):
    a, b = P_BASE ** k, p_effective ** k
    print(f"{k:>7} {a:>13.3f} {b:>15.3f} {b - a:>+8.3f}")

print(f"\nper-round success rises {P_BASE:.2f} -> {p_effective:.3f} with one "
      f"retry at {P_RECOVER:.0%} recovery")
print("""
Error feedback is the highest-leverage intervention available, and it is almost
free: return the error as the tool result instead of raising, and make the
message specific enough to act on. A four-round task goes from 75% to 92%.

Note the gain GROWS with chain length, which is the opposite of most
interventions — the longer the task, the more feedback is worth.""")

# Iteration limits: necessary, and they cost you the tail.
print(f"\n{'limit':>7} {'tasks completed':>18} {'cut off':>10}")
rng = np.random.default_rng(0)
needed = rng.geometric(0.35, size=20000)          # rounds a task actually needs
for limit in (2, 3, 5, 8, 12, 20):
    done = float((needed <= limit).mean())
    print(f"{limit:>7} {done:>18.1%} {1 - done:>10.1%}")

print("""
An iteration limit is mandatory — without one a confused model loops forever —
and it truncates the tail of genuinely long tasks. The limit is a product
decision about which tasks you are willing to fail, not a safety valve that
costs nothing.""")

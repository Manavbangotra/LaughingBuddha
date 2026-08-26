# -*- coding: utf-8 -*-
# Extracted from: Chapter 94 — Structured Output and Constrained Decoding
# Source: src/.../ch094-structured-output.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""How much does constraining cost? Equation (eq:constraint-cost), measured."""
import numpy as np

rng = np.random.default_rng(1)
V = 500


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def step_cost(logits, allowed_mask):
    """Retained mass (eq:retained-mass) and its cost in nats."""
    p = softmax(logits)
    rho = float(p[allowed_mask].sum())
    return rho, -np.log(rho + 1e-12)


SCENARIOS = {
    "schema the model wants":    dict(n_allowed=40,  alignment=3.0),
    "neutral schema":            dict(n_allowed=40,  alignment=0.0),
    "schema fighting the model": dict(n_allowed=40,  alignment=-3.0),
    "very narrow, aligned":      dict(n_allowed=2,   alignment=3.0),
    "very narrow, misaligned":   dict(n_allowed=2,   alignment=-3.0),
}

print(f"{'scenario':<28} {'|A|':>5} {'retained mass':>15} {'cost (nats)':>13}")
for name, cfg in SCENARIOS.items():
    mask = np.zeros(V, dtype=bool)
    mask[rng.choice(V, cfg["n_allowed"], replace=False)] = True
    z = rng.normal(size=V)
    z[mask] += cfg["alignment"]           # does the model like the allowed set?
    rho, cost = step_cost(z, mask)
    print(f"{name:<28} {cfg['n_allowed']:>5} {rho:>15.4f} {cost:>13.4f}")

print("""
The retained mass is the diagnostic. Near 1.0 means the constraint is inactive —
the model wanted a valid token anyway and the guarantee is free. Small values
mean the model is being forced somewhere it did not want to go, and equation
(eq:constraint-cost) prices that in nats.

Crucially this is computable at generation time for nothing: it is one sum over
the mask you already applied.""")

# Where in a generation does the cost concentrate?
print(f"\n{'step':>6} {'retained mass':>15} {'cost':>9}  interpretation")
sequence = [0.99, 0.99, 0.97, 0.12, 0.99, 0.98, 0.06, 0.99]
total = 0.0
for i, rho in enumerate(sequence):
    c = -np.log(rho)
    total += c
    note = "<- schema and model disagree HERE" if rho < 0.3 else ""
    print(f"{i:>6} {rho:>15.2f} {c:>9.4f}  {note}")
print(f"\ntotal constraint cost: {total:.4f} nats over {len(sequence)} steps")
print(f"of which {sum(-np.log(r) for r in sequence if r < 0.3) / total:.0%} "
      f"comes from 2 steps")

print("""
The cost is not spread evenly — it concentrates at the few positions where the
schema and the model genuinely disagree. Those positions are actionable: they
usually indicate a field name the model does not expect, a type it wants to
write differently, or an enum value missing from the schema.

Logging per-step retained mass turns 'constrained output feels worse' into a
list of specific schema decisions to reconsider.""")

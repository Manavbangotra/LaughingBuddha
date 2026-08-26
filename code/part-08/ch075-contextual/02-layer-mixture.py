# -*- coding: utf-8 -*-
# Extracted from: Chapter 75 — Contextual Embeddings and the Encoder Revolution
# Source: src/.../ch075-contextual.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A minimum over the simplex is no larger than a minimum over its vertices."""
import numpy as np

rng = np.random.default_rng(0)
N, D, L = 400, 16, 3          # examples, feature width, layers (0..L-1)

# Two synthetic tasks. Each target is best explained by a different layer,
# which is the empirical situation ELMo reports for syntax versus semantics.
H = [rng.normal(size=(N, D)) for _ in range(L)]
targets = {
    "task A (favours layer 0)": 0.9 * H[0] @ rng.normal(size=(D, 1)),
    "task B (favours layer 2)": 0.9 * H[2] @ rng.normal(size=(D, 1)),
    "task C (needs a mixture)": 0.5 * H[0] @ rng.normal(size=(D, 1))
                               + 0.5 * H[2] @ rng.normal(size=(D, 1)),
}


def residual(features, y):
    """Least-squares residual of the best linear head on these features."""
    coef, *_ = np.linalg.lstsq(features, y, rcond=None)
    return float(np.mean((features @ coef - y) ** 2))


def best_mixture(y, grid=11):
    """Search the simplex for the mixing weights of equation (eq:elmo-mixture)."""
    best, best_s = np.inf, None
    for a in np.linspace(0, 1, grid):
        for b in np.linspace(0, 1 - a, grid):
            s = np.array([a, b, 1 - a - b])
            r = residual(sum(s[j] * H[j] for j in range(L)), y)
            if r < best:
                best, best_s = r, s
    return best, best_s


print(f"{'task':<26} {'layer 0':>9} {'layer 1':>9} {'layer 2':>9} "
      f"{'mixture':>9}  weights")
for name, y in targets.items():
    per_layer = [residual(H[j], y) for j in range(L)]
    mix, s = best_mixture(y)
    print(f"{name:<26} {per_layer[0]:>9.4f} {per_layer[1]:>9.4f} "
          f"{per_layer[2]:>9.4f} {mix:>9.4f}  {np.round(s, 2)}")
    assert mix <= min(per_layer) + 1e-9, "the mixture cannot be worse"

print("\nThe mixture is never worse than the best single layer — it optimises "
      "over the simplex, and each single layer is one of its vertices. The "
      "empirical claim in Peters et al. is that the optimum is not a vertex.")

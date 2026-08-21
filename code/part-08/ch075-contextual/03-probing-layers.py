# Extracted from: Chapter 75 — Contextual Embeddings and the Encoder Revolution
# Source: src/.../ch075-contextual.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Probing: which layer carries the feature this task needs?"""
import numpy as np

rng = np.random.default_rng(7)
N, D = 600, 24

# A stand-in for an encoder's layers. Lower layers carry surface form, middle
# layers carry structure, top layers carry the pretraining objective's target.
surface = rng.normal(size=(N, D))
structure = rng.normal(size=(N, D))
objective = rng.normal(size=(N, D))

layers = {
    "layer 0 (embeddings)": surface,
    "layer 1": 0.7 * surface + 0.3 * structure,
    "layer 2": 0.3 * surface + 0.7 * structure,
    "layer 3": 0.6 * structure + 0.4 * objective,
    "layer 4 (top)": 0.2 * structure + 0.8 * objective,
}

# The downstream task depends mostly on structure — as clause segmentation does.
w = rng.normal(size=(D, 1))
y = (structure @ w > 0).astype(float).ravel()

split = int(0.7 * N)


def probe_accuracy(X, y):
    """Logistic regression by plain gradient descent — the probe must be weak."""
    Xtr, ytr, Xte, yte = X[:split], y[:split], X[split:], y[split:]
    b = np.zeros(X.shape[1])
    for _ in range(600):
        p = 1 / (1 + np.exp(-np.clip(Xtr @ b, -30, 30)))
        b -= 0.05 * (Xtr.T @ (p - ytr)) / len(ytr)
    pred = (Xte @ b > 0).astype(float)
    return float((pred == yte).mean())


print(f"{'layer':<22} {'probe accuracy':>15}")
scores = {}
for name, X in layers.items():
    scores[name] = probe_accuracy(X, y)
    print(f"{name:<22} {scores[name]:>15.3f}")

best = max(scores, key=scores.get)
top = "layer 4 (top)"
print(f"\nbest layer: {best} ({scores[best]:.3f})")
print(f"default choice (top): {scores[top]:.3f}")
print(f"cost of taking the top layer by default: "
      f"{scores[best] - scores[top]:+.3f} accuracy")
print("\nThe probe must be weak — a linear model. A deep probe can recover the "
      "feature from almost any layer and tells you about the probe, not the "
      "representation.")

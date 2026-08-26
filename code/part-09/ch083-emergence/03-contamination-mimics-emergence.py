# -*- coding: utf-8 -*-
# Extracted from: Chapter 83 — Emergent Capabilities and What Emergence Means
# Source: src/.../ch083-emergence.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Memorisation produces the emergence signature, and correlates with scale."""
import numpy as np

rng = np.random.default_rng(0)
scales = np.logspace(7, 11, 12)
log_s = np.log10(scales)
N_TEST = 400

# Genuine capability: smooth, and modest even at the largest scale.
true_skill = 0.15 + 0.35 / (1 + np.exp(-(log_s - 10.2) * 2.0))

# Contamination: larger models are trained on more data, so the chance that a
# given test item was seen rises with scale (ch:fm-datasets).
seen_fraction = np.clip((log_s - 8.5) / 3.0, 0, 1) * 0.55

print(f"{'params':>10} {'true skill':>11} {'% test seen':>12} "
      f"{'observed':>10} {'inflation':>10}")
observed = []
for i, s in enumerate(scales):
    seen = rng.random(N_TEST) < seen_fraction[i]
    correct = np.where(seen, rng.random(N_TEST) < 0.97,       # memorised
                       rng.random(N_TEST) < true_skill[i])    # actually solved
    obs = correct.mean()
    observed.append(obs)
    print(f"{s:>10.1e} {true_skill[i]:>11.3f} {seen_fraction[i]:>11.1%} "
          f"{obs:>10.3f} {obs - true_skill[i]:>10.3f}")

observed = np.array(observed)
print(f"\ntrue skill rises   {true_skill[0]:.3f} -> {true_skill[-1]:.3f} "
      f"({true_skill[-1] / true_skill[0]:.1f}x)")
print(f"observed rises     {observed[0]:.3f} -> {observed[-1]:.3f} "
      f"({observed[-1] / observed[0]:.1f}x)")
print(f"largest jump: true {np.max(np.diff(true_skill)):.3f}, "
      f"observed {np.max(np.diff(observed)):.3f}")

print("""
The observed curve rises far faster than the underlying skill, and it does so
because contamination CORRELATES WITH SCALE — a bigger model saw more data, so
it saw more of the test set. This is a third explanation for a sharp curve,
alongside a real jump and a metric artefact, and it is the one least often
audited.

Note that no rescoring detects it. A continuous metric would show the same
inflation, because the model really is producing the right answers. Only a
contamination audit against the training corpus separates this case, and for
frontier models that corpus is not public.""")

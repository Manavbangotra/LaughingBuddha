# -*- coding: utf-8 -*-
# Extracted from: Chapter 93 — Prompting and System Prompts
# Source: src/.../ch093-prompting.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Where to spend effort on a prompt, ranked by measured effect."""
import numpy as np

rng = np.random.default_rng(5)
N_EVAL = 400

# Effects drawn from what the controlled literature supports, plus the
# techniques that circulate without controls (given ~zero effect).
INTERVENTIONS = {
    "cover the label space in exemplars": (0.055, "min2022: label space dominates"),
    "add chain-of-thought (multi-step task)": (0.048, "wei2022cot, kojima2022"),
    "self-consistency, n=5": (0.041, "eq:self-consistency-condition"),
    "move instruction after the context": (0.022, "liu2023lost: position effects"),
    "vary exemplar surface forms": (0.018, "format is a learned component"),
    "reword the instruction more carefully": (0.006, "within prompt noise"),
    "tell the model it is an expert": (0.002, "no controlled evidence"),
    "offer the model a tip": (0.001, "no controlled evidence"),
}

sigma = np.sqrt(0.7 * 0.3 / N_EVAL)      # binomial noise at m=0.7
print(f"evaluation set {N_EVAL} examples, binomial SE {sigma:.4f}")
print(f"detectable effect at 2 SE: {2 * sigma:.4f}\n")

print(f"{'intervention':<42} {'effect':>8} {'vs noise':>10} {'evidence'}")
for name, (effect, evidence) in sorted(INTERVENTIONS.items(),
                                       key=lambda kv: -kv[1][0]):
    detectable = "yes" if effect > 2 * sigma else "NO"
    print(f"{name:<42} {effect:>+8.3f} {detectable:>10} {evidence}")

print(f"""
The bottom three interventions are below the noise floor of a 400-example
evaluation. That does not prove they do nothing — it means a team measuring on
400 examples CANNOT TELL, and any improvement they report from them is
indistinguishable from the prompt sensitivity measured in the previous listing.

The top three are well above it and all three have controlled evidence behind
them.""")

# What it would take to detect the small effects.
print(f"\n{'to detect an effect of':>24} {'you need n =':>14}")
for effect in (0.05, 0.02, 0.01, 0.005, 0.002):
    n = int(np.ceil(2 * (2 ** 2) * 0.7 * 0.3 / (effect ** 2)))
    print(f"{effect:>24.3f} {n:>14,}")

print("""
Detecting a two-tenths-of-a-point effect needs on the order of a million
examples. Nobody runs that evaluation, which is precisely why advice at that
effect size circulates indefinitely: it can neither be confirmed nor refuted by
any evaluation a team will actually perform.

The discipline this implies is simple. Rank interventions by measured effect
against your evaluation set's noise floor, spend effort on the ones above it,
and treat everything below it as unfalsifiable rather than true.""")

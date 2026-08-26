# -*- coding: utf-8 -*-
# Extracted from: Chapter 25 — Correlation, Causation, and Confounding
# Source: src/.../ch025-causation.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Choosing adjustment variables from causal structure, not from the data.

Four candidate covariates with identical statistical prominence and four
different correct treatments.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(4)
n = 250_000

# --- the true causal structure ----------------------------------------------
# tenure    -> confounder: drives both feature adoption and retention
# adopted   -> the treatment
# engagement-> mediator:   adoption causes engagement causes retention
# support   -> collider:   both adoption and retention affect contacting support
# region    -> predictor of the outcome only

tenure = rng.normal(0, 1, n)
adopted = (rng.random(n) < 1 / (1 + np.exp(-(0.9 * tenure))))
TRUE_DIRECT = 0.04
engagement = 0.5 * adopted + 0.3 * tenure + rng.normal(0, 0.5, n)
region_effect = rng.normal(0, 0.3, n)
retention = (0.30
             + TRUE_DIRECT * adopted            # the direct effect
             + 0.06 * engagement                # the mediated path
             + 0.10 * tenure                    # confounding path
             + region_effect
             + rng.normal(0, 0.1, n))
support = 0.4 * adopted + 0.4 * retention + rng.normal(0, 0.4, n)  # collider

df = pd.DataFrame({"adopted": adopted.astype(int), "tenure": tenure,
                   "engagement": engagement, "support": support,
                   "region_effect": region_effect, "retention": retention})

# The total effect of adoption = direct + through engagement.
TRUE_TOTAL = TRUE_DIRECT + 0.06 * 0.5
print(f"true DIRECT effect of adoption : {TRUE_DIRECT:+.4f}")
print(f"true TOTAL effect (direct + via engagement) : {TRUE_TOTAL:+.4f}\n")


def estimate(adjust_for):
    """Regress retention on adoption, adjusting for the named covariates."""
    cols = ["adopted"] + list(adjust_for)
    X = np.column_stack([np.ones(n)] + [df[c].to_numpy() for c in cols])
    beta, *_ = np.linalg.lstsq(X, df["retention"].to_numpy(), rcond=None)
    return beta[1]


scenarios = [
    ("nothing (naive)",              [],                              TRUE_TOTAL),
    ("tenure (the confounder)",      ["tenure"],                      TRUE_TOTAL),
    ("tenure + engagement",          ["tenure", "engagement"],        TRUE_DIRECT),
    ("tenure + support (collider)",  ["tenure", "support"],           TRUE_TOTAL),
    ("tenure + region",              ["tenure", "region_effect"],     TRUE_TOTAL),
    ("everything",                   ["tenure", "engagement",
                                      "support", "region_effect"],    TRUE_DIRECT),
]

print(f"{'adjusting for':<30} {'estimate':>10} {'target':>9} {'error':>9}")
print("-" * 62)
for label, cols, target in scenarios:
    est = estimate(cols)
    print(f"{label:<30} {est:>+10.4f} {target:>+9.4f} {est - target:>+9.4f}")

print("\ninterpretation:")
print("  nothing            — biased upward: the tenure back-door path is open")
print("  tenure             — correct for the TOTAL effect; the back door is closed")
print("  tenure+engagement  — correct for the DIRECT effect; adjusting for a")
print("                       mediator removes the path you may have wanted")
print("  tenure+support     — the SIGN FLIPS. Support is a collider, and")
print("                       conditioning on it opens a spurious path strong")
print("                       enough to turn a real positive effect negative.")
print("                       An analyst who 'controlled for support contacts'")
print("                       would conclude the feature HARMS retention.")
print("  tenure+region      — unchanged estimate, smaller variance: adjusting")
print("                       for an outcome-only predictor is free precision")
print("  everything         — the 'control for everything' default, which here")
print("                       silently answers a different question AND is")
print("                       contaminated by the collider")

# --- the variance benefit of adjusting for an outcome predictor -------------
print("\n" + "=" * 72)
print("adjusting for an outcome-only predictor reduces variance")
print("=" * 72)
ests_without, ests_with = [], []
for _ in range(300):
    idx = rng.choice(n, 4000, replace=False)
    sub = df.iloc[idx]
    for cols, store in ((["tenure"], ests_without),
                        (["tenure", "region_effect"], ests_with)):
        X = np.column_stack([np.ones(len(sub))]
                            + [sub[c].to_numpy() for c in ["adopted"] + cols])
        b, *_ = np.linalg.lstsq(X, sub["retention"].to_numpy(), rcond=None)
        store.append(b[1])

print(f"{'adjustment':<26} {'mean estimate':>15} {'sd of estimate':>16}")
print(f"{'tenure only':<26} {np.mean(ests_without):>+15.4f} "
      f"{np.std(ests_without):>16.4f}")
print(f"{'tenure + region':<26} {np.mean(ests_with):>+15.4f} "
      f"{np.std(ests_with):>16.4f}")
print(f"\nSame estimate, {np.std(ests_without)/np.std(ests_with):.1f}x tighter. "
      f"Adjusting for a variable that predicts")
print("the outcome but not the treatment costs nothing and buys precision.")

print("\n" + "=" * 72)
print("the point")
print("=" * 72)
print("All four covariates look similar in the data: each correlates with")
print("both adoption and retention. Their correct treatment is opposite in")
print("three of the four cases, and nothing in the table distinguishes them.")
print("The decision comes from a causal model, which comes from knowing how")
print("the system works.")

# -*- coding: utf-8 -*-
# Extracted from: Chapter 93 — Prompting and System Prompts
# Source: src/.../ch093-prompting.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""What do demonstrations actually supply? Ablating the four components."""
import numpy as np

rng = np.random.default_rng(1)

# A model of in-context learning consistent with min2022: performance depends
# mostly on knowing the LABEL SPACE, the INPUT DISTRIBUTION and the FORMAT,
# and only weakly on the demonstrated mapping being correct.
CONTRIBUTION = {
    "label space": 0.18,      # knowing which answers are possible
    "input distribution": 0.09,
    "format": 0.11,
    "correct mapping": 0.03,  # the component people think they are supplying
}
BASE = 0.42                   # zero-shot performance


def performance(components):
    return BASE + sum(CONTRIBUTION[c] for c in components)


ALL = list(CONTRIBUTION)
full = performance(ALL)

print(f"zero-shot baseline            : {BASE:.3f}")
print(f"full demonstrations           : {full:.3f}")
print(f"total gain from demonstrations: {full - BASE:+.3f}\n")

print(f"{'ablation':<34} {'performance':>12} {'cost of removing':>18}")
for c in ALL:
    remaining = [x for x in ALL if x != c]
    p = performance(remaining)
    print(f"{'remove ' + c:<34} {p:>12.3f} {p - full:>+18.3f}")

random_labels = performance([c for c in ALL if c != "correct mapping"])
print(f"\nRANDOM LABELS (min2022's experiment): {random_labels:.3f}")
print(f"  versus correct labels             : {full:.3f}")
print(f"  cost of randomising every label   : "
      f"{random_labels - full:+.3f} "
      f"({abs(random_labels - full) / (full - BASE):.0%} of the total gain)")

print("""
Randomising every label costs a small fraction of what demonstrations buy. The
components that matter are the ones nobody thinks about: which labels exist,
what the inputs look like, and how a response is shaped.

The practical inversion: to improve a few-shot prompt, COVER THE LABEL SPACE and
vary the surface form, rather than perfecting each exemplar's correctness. Most
few-shot prompts are written the other way round.""")

# What that implies for how to spend a fixed exemplar budget.
print(f"\n{'strategy for 6 exemplars':<40} {'label coverage':>16} "
      f"{'est. performance':>18}")
N_LABELS = 6
for label, covered, note in [
        ("6 examples of the majority class", 1, ""),
        ("3 classes, 2 examples each", 3, ""),
        ("6 classes, 1 example each", 6, "<- covers the space"),
        ("6 classes, 1 each, WRONG labels", 6, "<- still covers it")]:
    coverage = covered / N_LABELS
    # Label-space contribution scales with coverage; mapping only matters if right.
    est = (BASE + CONTRIBUTION["label space"] * coverage
           + CONTRIBUTION["input distribution"] + CONTRIBUTION["format"]
           + (CONTRIBUTION["correct mapping"] if "WRONG" not in label else 0))
    print(f"{label:<40} {coverage:>15.0%} {est:>18.3f} {note}")

print("""
The last two rows are the point. Six exemplars with WRONG labels that cover the
label space outperform six correct exemplars that do not — because coverage is
worth six times what correctness is.

That is a strange sentence and it is what the ablation says.""")

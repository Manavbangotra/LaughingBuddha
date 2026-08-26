# -*- coding: utf-8 -*-
# Extracted from: Chapter 93 — Prompting and System Prompts
# Source: src/.../ch093-prompting.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Measuring prompt sensitivity, and the selection bias it enables."""
import numpy as np

rng = np.random.default_rng(0)

# A simulated task: each prompt has a TRUE accuracy, and we observe a noisy
# estimate on a finite evaluation set. That is the whole structure of the
# problem — the observation noise is what selection exploits.
N_PROMPTS, N_EVAL = 20, 200
TRUE_MEAN, TRUE_SPREAD = 0.70, 0.04

true_acc = np.clip(rng.normal(TRUE_MEAN, TRUE_SPREAD, N_PROMPTS), 0.05, 0.95)


def evaluate(acc, n=N_EVAL):
    """Observed accuracy on a finite set — binomial noise."""
    return float(rng.binomial(n, acc) / n)


observed = np.array([evaluate(a) for a in true_acc])

print(f"{N_PROMPTS} semantically equivalent prompts, "
      f"{N_EVAL} evaluation examples each\n")
print(f"{'':<22} {'true':>8} {'observed':>10}")
print(f"{'best prompt':<22} {true_acc.max():>8.3f} {observed.max():>10.3f}")
print(f"{'worst prompt':<22} {true_acc.min():>8.3f} {observed.min():>10.3f}")
print(f"{'mean':<22} {true_acc.mean():>8.3f} {observed.mean():>10.3f}")
print(f"{'sensitivity (max-min)':<22} {true_acc.max() - true_acc.min():>8.3f} "
      f"{observed.max() - observed.min():>10.3f}")

# Equation (eq:prompt-selection-bias): what does picking the best cost you?
selected = int(observed.argmax())
print(f"\nselected prompt {selected}: reported {observed[selected]:.3f}, "
      f"true {true_acc[selected]:.3f}")
print(f"optimism from selection: "
      f"{observed[selected] - true_acc[selected]:+.3f}")

sigma = np.sqrt(TRUE_MEAN * (1 - TRUE_MEAN) / N_EVAL)
predicted_bias = sigma * np.sqrt(2 * np.log(N_PROMPTS))
print(f"\nbinomial standard error       : {sigma:.4f}")
print(f"predicted bias (eq:prompt-selection-bias): {predicted_bias:.4f}")

# Averaged over many trials, so the prediction can be checked.
biases, held_out_gaps = [], []
for _ in range(400):
    ta = np.clip(rng.normal(TRUE_MEAN, TRUE_SPREAD, N_PROMPTS), 0.05, 0.95)
    ob = np.array([evaluate(a) for a in ta])
    s = int(ob.argmax())
    biases.append(ob[s] - ta[s])
    # The fix: re-evaluate the chosen prompt on a FRESH split.
    held_out_gaps.append(evaluate(ta[s]) - ta[s])

print(f"mean observed bias over 400 trials       : {np.mean(biases):.4f}")
print(f"mean bias after re-evaluating on a fresh split: "
      f"{np.mean(held_out_gaps):+.4f}")

print("""
Selecting the best of twenty prompts on a 200-example set inflates the reported
score by several points, and the inflation is pure selection — the chosen
prompt's TRUE accuracy is close to average. Re-evaluating on a fresh split
removes it entirely, and costs one extra evaluation run.

This is ch:mle-hpo's winner's curse with a prompt in place of a hyperparameter,
and it is why a prompt should be reported with the spread across phrasings
rather than as the best number found.""")

# What sensitivity does to a MODEL comparison.
print(f"\n{'comparison':<40} {'verdict':>28}")
model_a = np.clip(rng.normal(0.70, TRUE_SPREAD, N_PROMPTS), 0.05, 0.95)
model_b = np.clip(rng.normal(0.72, TRUE_SPREAD, N_PROMPTS), 0.05, 0.95)
print(f"{'true means (A=0.70, B=0.72)':<40} {'B is better by 0.02':>28}")
print(f"{'A at its best prompt vs B at its worst':<40} "
      f"{f'A wins by {model_a.max() - model_b.min():+.3f}':>28}")
print(f"{'both at a single fixed prompt (#0)':<40} "
      f"{f'{model_b[0] - model_a[0]:+.3f} for B':>28}")
print(f"{'both averaged over all 20 prompts':<40} "
      f"{f'{model_b.mean() - model_a.mean():+.3f} for B':>28}")

print("""
The second row is how model comparisons are frequently reported, and it reverses
the true ordering. Averaging over a prompt distribution recovers it. A single
fixed prompt is better than cherry-picking and is still one draw from a
distribution with a spread larger than the effect being measured.""")

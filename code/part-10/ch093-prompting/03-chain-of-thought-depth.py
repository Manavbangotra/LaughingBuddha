# -*- coding: utf-8 -*-
# Extracted from: Chapter 93 — Prompting and System Prompts
# Source: src/.../ch093-prompting.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Chain-of-thought converts depth into length. Equation (eq:cot-depth)."""
import numpy as np

rng = np.random.default_rng(2)

# A transformer has L layers, so a single forward pass can chain at most L
# sequential operations. Generating intermediate tokens puts partial results
# into the CONTEXT, where later tokens can attend to them.
LAYERS = 32
PER_STEP_ACCURACY = 0.93


def direct_answer_accuracy(required_steps):
    """One forward pass: capped by depth."""
    if required_steps > LAYERS:
        return 0.02                      # cannot be done at all
    return PER_STEP_ACCURACY ** required_steps


def cot_accuracy(required_steps, steps_per_token=4):
    """Intermediate tokens extend the effective depth (eq:cot-depth)."""
    tokens = int(np.ceil(required_steps / steps_per_token))
    # Each generated step is itself a small computation that can fail.
    return PER_STEP_ACCURACY ** required_steps * (0.985 ** tokens)


print(f"{LAYERS}-layer model, {PER_STEP_ACCURACY:.0%} per-step accuracy\n")
print(f"{'task steps':>11} {'direct':>9} {'chain-of-thought':>18} "
      f"{'gain':>9} {'CoT tokens':>12}")
for steps in (2, 5, 10, 20, 40, 80, 160):
    d = direct_answer_accuracy(steps)
    c = cot_accuracy(steps)
    print(f"{steps:>11} {d:>9.3f} {c:>18.3f} {c - d:>+9.3f} "
          f"{int(np.ceil(steps / 4)):>12}")

print("""
Below the layer count the two are comparable — the model can do it in one pass,
and the reasoning tokens add a little risk without adding capability. Past the
layer count the direct answer collapses to chance while chain-of-thought keeps
working, because equation (eq:cot-depth) has removed the depth ceiling.

That is the prediction the mechanism makes, and it matches the empirical
finding: chain-of-thought helps enormously on multi-step arithmetic and
multi-hop reasoning, and barely at all on single-step classification. If a task
fits in one forward pass, asking for reasoning costs tokens and buys nothing.""")

# Self-consistency, equation (eq:self-consistency-condition).
def self_consistency(p_correct, n_wrong_answers, n_samples, trials=4000):
    """Majority vote over n sampled chains."""
    wins = 0
    for _ in range(trials):
        votes = {}
        for _ in range(n_samples):
            if rng.random() < p_correct:
                ans = "correct"
            else:
                ans = f"wrong{rng.integers(n_wrong_answers)}"
            votes[ans] = votes.get(ans, 0) + 1
        if max(votes, key=votes.get) == "correct":
            wins += 1
    return wins / trials


print(f"\nSelf-consistency: majority vote over n sampled chains\n")
print(f"{'p(correct)':>11} {'errors spread over':>19} {'n=1':>7} {'n=5':>7} "
       f"{'n=15':>7} {'threshold 1/(k+1)':>19}")
for p, k in [(0.35, 9), (0.35, 1), (0.55, 9), (0.55, 1), (0.15, 20)]:
    row = [self_consistency(p, k, n) for n in (1, 5, 15)]
    thr = 1 / (k + 1)
    print(f"{p:>11.2f} {k:>19} {row[0]:>7.3f} {row[1]:>7.3f} {row[2]:>7.3f} "
          f"{thr:>19.3f}")

print("""
Read the last column against the first. Voting helps whenever p exceeds
1/(k+1) — equation (eq:self-consistency-condition) — so with errors scattered
over nine wrong answers, 35% per-sample accuracy becomes near-certainty.

The k=1 rows are the failure case. When the model concentrates its errors on ONE
wrong answer, voting needs p > 0.5, and at p = 0.35 sampling more chains makes
things WORSE — it measures the systematic error more precisely. Self-consistency
amplifies whatever the model's mode is, and that is only useful when the mode is
right.""")

# -*- coding: utf-8 -*-
# Extracted from: Chapter 85 — Alignment and RLHF
# Source: src/.../ch085-rlhf.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Planning a preference-data collection, and the reward model's ceiling."""
import numpy as np

rng = np.random.default_rng(5)

PROMPTS = 4_000
K_RESPONSES = 4                    # sampled per prompt
COST_PER_RANKING = 2.20            # one annotator ranking k responses
ANNOTATOR_AGREEMENT = 0.72         # measured on a doubly-labelled subset

pairs_per_prompt = K_RESPONSES * (K_RESPONSES - 1) // 2
total_pairs = PROMPTS * pairs_per_prompt
cost = PROMPTS * COST_PER_RANKING

print(f"{PROMPTS:,} prompts x {K_RESPONSES} responses")
print(f"  pairs per prompt : {pairs_per_prompt}")
print(f"  total pairs      : {total_pairs:,}")
print(f"  annotation cost  : ${cost:,.0f}")
print(f"  cost per pair    : ${cost / total_pairs:.3f}  "
      f"(ranking k responses is much cheaper per pair than collecting pairs)\n")

# The ceiling: a reward model cannot be more accurate than its labels.
# With agreement a, the fraction of pairs where the label matches the true
# preference is a; the rest are noise the model can only fit or ignore.
print(f"annotator agreement: {ANNOTATOR_AGREEMENT:.0%}")
print(f"-> a reward model scoring above {ANNOTATOR_AGREEMENT:.0%} pairwise "
      f"accuracy on this data is fitting label noise\n")

# How does reward-model accuracy scale with the number of comparisons?
D = 16
true_w = rng.normal(size=D)


def fit_rm(n_pairs, agreement):
    items = rng.normal(size=(600, D))
    r = items @ true_w
    i, j = rng.integers(0, 600, n_pairs), rng.integers(0, 600, n_pairs)
    m = i != j
    i, j = i[m], j[m]
    correct = r[i] > r[j]
    # Annotators disagree with the latent preference at rate (1 - agreement).
    flip = rng.random(len(i)) > agreement
    i_wins = np.where(flip, ~correct, correct)
    win, lose = np.where(i_wins, i, j), np.where(i_wins, j, i)

    w = np.zeros(D)
    for _ in range(600):
        diff = items[win] @ w - items[lose] @ w
        sig = 1 / (1 + np.exp(-diff))
        grad = ((sig - 1)[:, None] * (items[win] - items[lose])).mean(0)
        w -= 2.0 * grad

    test_i, test_j = rng.integers(0, 600, 4000), rng.integers(0, 600, 4000)
    m = test_i != test_j
    test_i, test_j = test_i[m], test_j[m]
    pred = (items[test_i] @ w) > (items[test_j] @ w)
    latent = r[test_i] > r[test_j]
    # Held-out LABELS are noisy in exactly the same way the training ones were.
    flip = rng.random(len(test_i)) > agreement
    labels = np.where(flip, ~latent, latent)
    return float(np.mean(pred == latent)), float(np.mean(pred == labels))


print(f"{'comparisons':>13} {'vs LATENT preference':>22} {'vs held-out LABELS':>21}")
for n in (500, 2_000, 8_000, 24_000):
    vs_latent, vs_labels = fit_rm(n, ANNOTATOR_AGREEMENT)
    print(f"{n:>13,} {vs_latent:>22.3f} {vs_labels:>21.3f}")

print(f"""
The two columns are different quantities and confusing them is common.

Accuracy against the LATENT preference keeps climbing toward 1.0. Noisy labels
still identify a consistent underlying ordering given enough of them — noise
that is symmetric averages out, which is why more comparisons help even when
each one is unreliable.

Accuracy against held-out LABELS saturates near the annotator agreement of
{ANNOTATOR_AGREEMENT:.0%}, and cannot exceed it: the held-out labels are wrong
{1 - ANNOTATOR_AGREEMENT:.0%} of the time, so a perfect model disagrees with
them exactly that often.

This matters because the second column is the one you can actually measure. A
reward model scoring {ANNOTATOR_AGREEMENT:.2f} against held-out labels may be
perfect or may be mediocre, and the number alone cannot tell you — which is why
annotator agreement must be measured separately, as the interpretation key for
every reward-model number you will ever report.""")

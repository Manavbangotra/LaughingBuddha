# -*- coding: utf-8 -*-
# Extracted from: Chapter 85 — Alignment and RLHF
# Source: src/.../ch085-rlhf.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Fit a reward model from pairwise comparisons. Equation (eq:reward-model-loss)."""
import numpy as np

rng = np.random.default_rng(0)

N_ITEMS, D, N_PAIRS = 60, 8, 4000

# A latent reward that we will try to recover from comparisons alone.
features = rng.normal(size=(N_ITEMS, D))
true_w = rng.normal(size=D)
true_reward = features @ true_w

# Humans compare pairs and choose stochastically per Bradley-Terry
# (eq:bradley-terry) — they are noisy, not deterministic.
i = rng.integers(0, N_ITEMS, N_PAIRS)
j = rng.integers(0, N_ITEMS, N_PAIRS)
keep = i != j
i, j = i[keep], j[keep]
p_i_wins = 1 / (1 + np.exp(-(true_reward[i] - true_reward[j])))
i_wins = rng.random(len(i)) < p_i_wins
winner = np.where(i_wins, i, j)
loser = np.where(i_wins, j, i)

print(f"{N_ITEMS} responses, {len(winner)} comparisons")
agree = float(np.mean((true_reward[winner] > true_reward[loser])))
print(f"labeller agreement with the latent reward: {agree:.3f} "
      f"(noisy by construction)\n")


def loss_and_grad(w):
    """Negative log likelihood of eq:reward-model-loss, and its gradient."""
    r = features @ w
    diff = r[winner] - r[loser]
    sig = 1 / (1 + np.exp(-diff))
    loss = -np.mean(np.log(sig + 1e-12))
    # d/dw of -log sigma(rw - rl) = -(1 - sigma) * (phi_w - phi_l)
    coef = (sig - 1)[:, None]
    grad = (coef * (features[winner] - features[loser])).mean(0)
    return loss, grad


w = np.zeros(D)
for step in range(1, 3001):
    loss, grad = loss_and_grad(w)
    w -= 1.0 * grad
    if step in (1, 500, 1500, 3000):
        print(f"step {step:>4}: reward-model loss {loss:.4f}")

fitted = features @ w

# Does the fitted reward rank items the way the latent one does?
order_true = np.argsort(true_reward)
order_fit = np.argsort(fitted)
rank_true = np.empty(N_ITEMS); rank_true[order_true] = np.arange(N_ITEMS)
rank_fit = np.empty(N_ITEMS); rank_fit[order_fit] = np.arange(N_ITEMS)
spearman = float(np.corrcoef(rank_true, rank_fit)[0, 1])

pairs = [(a, b) for a in range(N_ITEMS) for b in range(a + 1, N_ITEMS)]
concordant = np.mean([(true_reward[a] > true_reward[b]) ==
                      (fitted[a] > fitted[b]) for a, b in pairs])

print(f"\nrank correlation with the latent reward : {spearman:.4f}")
print(f"pairwise ordering agreement             : {concordant:.4f}")
assert spearman > 0.9, "the reward model should recover the latent ordering"

# The non-identifiability of section 5.1, demonstrated.
shifted = fitted + 7.3
print(f"\nadding a constant to every reward:")
print(f"  mean reward   {fitted.mean():+.3f} -> {shifted.mean():+.3f}")
print(f"  ordering agreement unchanged: "
      f"{np.mean([(fitted[a] > fitted[b]) == (shifted[a] > shifted[b]) for a, b in pairs]):.3f}")
print("Absolute reward values carry no information. Only differences do — so a "
      "dashboard tracking mean reward across prompts is tracking an artefact.")

# -*- coding: utf-8 -*-
# Extracted from: Chapter 86 — Preference Optimization: DPO and Its Descendants
# Source: src/.../ch086-dpo.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""DPO from scratch, on an explicit small distribution where everything is exact."""
import numpy as np

rng = np.random.default_rng(0)

N_PROMPTS, N_RESPONSES, BETA = 6, 12, 0.5

# A reference policy: a proper distribution over responses for each prompt.
ref_logits = rng.normal(size=(N_PROMPTS, N_RESPONSES))
ref = np.exp(ref_logits) / np.exp(ref_logits).sum(1, keepdims=True)

# A latent reward we never give the algorithm. DPO must recover its ORDERING
# from comparisons alone, with no reward model anywhere.
true_reward = rng.normal(size=(N_PROMPTS, N_RESPONSES))


def sample_pairs(n):
    """Comparisons drawn per Bradley-Terry on the latent reward."""
    out = []
    for _ in range(n):
        x = rng.integers(N_PROMPTS)
        a, b = rng.choice(N_RESPONSES, size=2, replace=False)
        p_a = 1 / (1 + np.exp(-(true_reward[x, a] - true_reward[x, b])))
        if rng.random() < p_a:
            out.append((x, a, b))
        else:
            out.append((x, b, a))
    return out


pairs = sample_pairs(6000)
print(f"{N_PROMPTS} prompts x {N_RESPONSES} responses, {len(pairs)} comparisons")
print(f"beta = {BETA}\n")

# The policy is parameterised by logits, initialised AT the reference — which
# is what makes the initial loss exactly log 2.
policy_logits = ref_logits.copy()


def policy_probs(logits):
    e = np.exp(logits - logits.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def implicit_reward(logits):
    """Equation (eq:dpo-implicit-reward)."""
    return BETA * (np.log(policy_probs(logits) + 1e-12) - np.log(ref + 1e-12))


def dpo_loss_and_grad(logits):
    """Equation (eq:dpo-loss) and its gradient with respect to the logits."""
    p = policy_probs(logits)
    logp = np.log(p + 1e-12)
    r = BETA * (logp - np.log(ref + 1e-12))

    total, grad = 0.0, np.zeros_like(logits)
    for x, w, l in pairs:
        margin = r[x, w] - r[x, l]
        total += -np.log(1 / (1 + np.exp(-margin)) + 1e-12)
        # d/d margin of -log sigma(margin) = -(1 - sigma(margin))
        coef = -(1 - 1 / (1 + np.exp(-margin))) * BETA
        # d logp[x,i] / d logits[x,j] = delta_ij - p[x,j]
        grad[x] += coef * ((np.eye(N_RESPONSES)[w] - p[x])
                           - (np.eye(N_RESPONSES)[l] - p[x]))
    return total / len(pairs), grad / len(pairs)


loss0, _ = dpo_loss_and_grad(policy_logits)
print(f"loss at initialisation : {loss0:.6f}")
print(f"log 2                  : {np.log(2):.6f}   <- section 6.4's diagnostic")
assert abs(loss0 - np.log(2)) < 1e-6, \
    "with policy == reference every implicit reward is 0, so the loss is log 2"

for step in range(1, 4001):
    loss, grad = dpo_loss_and_grad(policy_logits)
    policy_logits -= 12.0 * grad
    if step in (1, 500, 2000, 4000):
        print(f"step {step:>4}: DPO loss {loss:.4f}")

# Did the implicit reward recover the latent ORDERING, with no reward model?
r_hat = implicit_reward(policy_logits)
agree, total = 0, 0
for x in range(N_PROMPTS):
    for a in range(N_RESPONSES):
        for b in range(a + 1, N_RESPONSES):
            total += 1
            agree += ((true_reward[x, a] > true_reward[x, b])
                      == (r_hat[x, a] > r_hat[x, b]))
print(f"\nimplicit-reward ordering agreement with the latent reward: "
      f"{agree / total:.4f}")
assert agree / total > 0.85, "the implicit reward should recover the ordering"

# And does the policy match the closed form eq:rlhf-optimal-policy?
analytic = ref * np.exp(true_reward / BETA)
analytic /= analytic.sum(1, keepdims=True)
learned = policy_probs(policy_logits)
corr = float(np.corrcoef(analytic.ravel(), learned.ravel())[0, 1])
print(f"correlation with the closed-form optimal policy: {corr:.4f}")

print("""
No reward model was fitted anywhere in this listing. The ordering was recovered
from the policy's own log-ratio against the reference, which is what
eq:implicit-reward says must be possible — and the learned policy tracks the
closed form of eq:rlhf-optimal-policy that the previous chapter derived.""")

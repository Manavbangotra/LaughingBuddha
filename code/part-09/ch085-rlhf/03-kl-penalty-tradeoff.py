# Extracted from: Chapter 85 — Alignment and RLHF
# Source: src/.../ch085-rlhf.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""What beta buys and what it costs. Equation (eq:rlhf-optimal-policy)."""
import numpy as np

rng = np.random.default_rng(2)
N = 200

# A reference policy over N candidate responses, and a fitted reward whose
# error grows with distance from the reference's mass.
ref_logits = rng.normal(size=N)
ref = np.exp(ref_logits) / np.exp(ref_logits).sum()

true_r = rng.normal(size=N)
# Error is larger for responses the reference rarely produces — exactly the
# responses the reward model has little data on.
rarity = -np.log(ref + 1e-12)
rarity = (rarity - rarity.min()) / (rarity.max() - rarity.min())
fitted_r = true_r + rng.normal(size=N) * rarity * 1.8


def optimal_policy(beta):
    """Equation (eq:rlhf-optimal-policy), computed exactly."""
    logits = np.log(ref + 1e-12) + fitted_r / beta
    logits -= logits.max()
    p = np.exp(logits)
    return p / p.sum()


def kl(p, q):
    return float(np.sum(p * np.log((p + 1e-12) / (q + 1e-12))))


print(f"{'beta':>8} {'KL(pi||ref)':>12} {'E[fitted r]':>13} {'E[TRUE r]':>11} "
      f"{'verdict':<22}")
best_true, best_beta = -np.inf, None
for beta in (10.0, 3.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02):
    p = optimal_policy(beta)
    d = kl(p, ref)
    ef, et = float(p @ fitted_r), float(p @ true_r)
    if et > best_true:
        best_true, best_beta = et, beta
    verdict = "barely moved" if d < 0.1 else ("over-optimised" if et < 0 else "")
    print(f"{beta:>8.2f} {d:>12.3f} {ef:>13.3f} {et:>11.3f} {verdict:<22}")

print(f"\ntrue reward is maximised at beta = {best_beta} "
      f"(E[true r] = {best_true:.3f})")
print(f"at beta = 0.02 the policy has KL {kl(optimal_policy(0.02), ref):.2f} "
      f"from the reference and E[true r] = {optimal_policy(0.02) @ true_r:+.3f}")

print("""
Both ends of the beta range are bad and for different reasons. Large beta keeps
the policy on top of the reference and captures almost none of the available
improvement. Small beta lets the policy chase the reward model into the region
where it is wrong, and true reward falls even as fitted reward rises.

The optimum is interior, and — this is the difficult part — it cannot be found
by watching the fitted reward, which is monotone in 1/beta. Setting beta
requires held-out evaluation with real judges, which is expensive, which is why
in practice it is set conservatively and rarely tuned.""")

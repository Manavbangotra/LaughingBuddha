# -*- coding: utf-8 -*-
# Extracted from: Chapter 85 — Alignment and RLHF
# Source: src/.../ch085-rlhf.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""True quality rises, peaks, then falls, while predicted reward climbs."""
import numpy as np

rng = np.random.default_rng(1)
N_CANDIDATES = 40_000

# The model of the world this listing assumes, stated explicitly:
#
#   1. Moving away from the reference policy buys real quality at first and
#      then costs it — text far from the SFT distribution degrades. So TRUE
#      quality is concave in the distance travelled.
#   2. The reward model's error GROWS with that distance, because it was fitted
#      on samples near the reference and has no evidence further out.
#
# The policy sees only fitted reward and maximises it.

def true_quality_mean(d):
    """Concave in distance: genuine gains, then degradation."""
    return 3.0 * np.sqrt(d) - 0.55 * d


def error_scale(d):
    """The reward model's error, growing with distance from its fitting data."""
    return 0.25 * d


print(f"{'KL budget':>10} {'predicted reward':>18} {'TRUE reward':>13} "
      f"{'E[error]':>10}")
results = []
for kl in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0):
    # Candidate responses available at this distance from the reference.
    true_r = true_quality_mean(kl) + rng.normal(0, 1.0, N_CANDIDATES)
    err = rng.normal(0, error_scale(kl) + 1e-9, N_CANDIDATES)
    fitted_r = true_r + err

    chosen = int(np.argmax(fitted_r))          # the policy maximises FITTED
    results.append((kl, float(fitted_r[chosen]), float(true_r[chosen]),
                    float(err[chosen])))
    print(f"{kl:>10.1f} {fitted_r[chosen]:>18.3f} {true_r[chosen]:>13.3f} "
          f"{err[chosen]:>10.3f}")

kls = [r[0] for r in results]
fits = [r[1] for r in results]
trues = [r[2] for r in results]
peak = int(np.argmax(trues))

print(f"\ntrue reward peaks at KL = {kls[peak]:.1f} "
      f"({trues[peak]:.3f}) and declines to {trues[-1]:.3f}")
print(f"predicted reward rises throughout: {fits[0]:.2f} -> {fits[-1]:.2f}")

assert trues[peak] > trues[-1], "true reward must decline past the peak"
assert fits[-1] > fits[0], "predicted reward must keep rising"
assert peak < len(kls) - 1, "the peak must be interior, not at the boundary"

print("""
This is eq:over-optimisation as a measurement. Two things drive it and both are
necessary: true quality eventually degrades with distance from the reference,
and the reward model's error grows there because it has no data. The policy
selects on the SUM, so past a point it is selecting error rather than quality —
note the E[error] column climbing steadily.

The practical consequence is in the second and third columns. Predicted reward,
the curve visible during training, rises monotonically and gives NO indication
that quality has begun to fall. The only instrument that sees the peak is
held-out human evaluation, which is why the KL penalty is set conservatively
rather than tuned against the reward.""")

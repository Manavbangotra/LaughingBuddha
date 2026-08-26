# -*- coding: utf-8 -*-
# Extracted from: Chapter 86 — Preference Optimization: DPO and Its Descendants
# Source: src/.../ch086-dpo.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Why a fixed preference dataset degrades as the policy moves away from it."""
import numpy as np

rng = np.random.default_rng(7)
D, N_CANDIDATES = 10, 3000

true_w = rng.normal(size=D)


def quality(v):
    return v @ true_w


# Preference data is collected by sampling from SOME policy. Offline DPO uses
# a fixed dataset collected from the SFT model; online methods re-collect from
# the CURRENT policy at every round.
sft_centre = np.zeros(D)


def collect(centre, n=400):
    """Comparisons between responses sampled around `centre`."""
    a = centre + rng.normal(size=(n, D))
    b = centre + rng.normal(size=(n, D))
    return a, b, quality(a) > quality(b)


def informative_fraction(data_centre, policy_centre):
    """What share of a dataset collected at `data_centre` discriminates between
    responses the policy at `policy_centre` would actually produce?"""
    a, b, _ = collect(data_centre, N_CANDIDATES)
    # Responses the current policy plausibly generates: those near its centre.
    near = (np.linalg.norm(a - policy_centre, axis=1) < 3.2) & \
           (np.linalg.norm(b - policy_centre, axis=1) < 3.2)
    return float(near.mean())


print(f"{'policy drift from SFT':>22} {'offline data relevance':>24} "
      f"{'online data relevance':>23}")
for drift in (0.0, 1.0, 2.0, 3.0, 4.0, 6.0):
    direction = true_w / np.linalg.norm(true_w)
    policy_centre = sft_centre + drift * direction
    offline = informative_fraction(sft_centre, policy_centre)
    online = informative_fraction(policy_centre, policy_centre)
    print(f"{drift:>22.1f} {offline:>24.3f} {online:>23.3f}")

print("""
Offline relevance collapses as the policy moves; online relevance does not,
because the data is re-collected where the policy now is.

This is the one thing DPO's derivation does not give it. The algebra is exact,
the reward model is genuinely redundant, and none of that addresses the fact
that a fixed dataset describes preferences over responses the policy has since
stopped producing. It is also why iterated DPO — alternate training and
re-collecting comparisons from the current policy — recovers much of the gap,
and why frontier labs, which can afford continuous collection, have less reason
to abandon online methods than a small team does.""")

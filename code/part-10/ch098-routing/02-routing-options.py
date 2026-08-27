# -*- coding: utf-8 -*-
# Extracted from: Chapter 98 — Model Routing and Model Selection
# Source: src/.../ch098-routing.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Router, cascade, feature-level routing, or nothing. All four costed."""
import numpy as np

rng = np.random.default_rng(1)
N = 20_000
SMALL_COST, LARGE_COST = 1.0, 12.0
LARGE_ACC = 0.94
ROUTER_COST = 0.15                 # an embedding + linear classifier

# Ground truth: each request has a latent difficulty, and the small model
# succeeds on the easy ones. Its overall accuracy is therefore an OUTCOME of
# the difficulty distribution rather than a parameter — which is the point,
# since routing exists precisely because that accuracy is not uniform.
difficulty = rng.random(N)
small_ok = rng.random(N) > difficulty
large_ok = rng.random(N) < LARGE_ACC


def router_prediction(quality):
    """A router that estimates difficulty with a given correlation to truth."""
    noise = rng.normal(0, 1.0, N)
    return quality * difficulty + (1 - quality) * rng.random(N) + 0.15 * noise


STRATEGIES = {}

STRATEGIES["always small"] = (float(small_ok.mean()), SMALL_COST)
STRATEGIES["always large"] = (float(large_ok.mean()), LARGE_COST)

# Feature-level routing: a fixed 60/40 split with no per-request decision.
split = rng.random(N) < 0.6
acc = float(np.where(split, small_ok, large_ok).mean())
cost = float(np.where(split, SMALL_COST, LARGE_COST).mean())
STRATEGIES["feature-level (60/40)"] = (acc, cost)

# Per-request router, at two prediction qualities.
for quality, label in [(0.55, "router (weak)"), (0.85, "router (good)")]:
    pred = router_prediction(quality)
    to_large = pred > np.quantile(pred, 0.6)      # send hardest 40% to large
    acc = float(np.where(to_large, large_ok, small_ok).mean())
    cost = ROUTER_COST + float(np.where(to_large, LARGE_COST, SMALL_COST).mean())
    STRATEGIES[label] = (acc, cost)

# Cascade at a 30% escalation rate, with the small model's confidence.
sep = 1.2
conf = rng.normal(np.where(small_ok, sep, 0.0), 1.0)
k = int(0.3 * N)
esc = np.zeros(N, dtype=bool)
esc[np.argsort(conf)[:k]] = True
acc = float(np.where(esc, large_ok, small_ok).mean())
cost = SMALL_COST + float(esc.mean()) * LARGE_COST
STRATEGIES["cascade (30% escalation)"] = (acc, cost)

print(f"{'strategy':<28} {'accuracy':>10} {'cost':>8} {'quality/cost':>14} "
      f"{'vs always-large':>17}")
for name, (a, c) in sorted(STRATEGIES.items(), key=lambda kv: kv[1][1]):
    print(f"{name:<28} {a:>10.4f} {c:>8.2f} {a / c:>14.4f} "
          f"{f'{a / LARGE_ACC:.0%} qual, {c / LARGE_COST:.0%} cost':>17}")

print("""
Read the quality/cost column. The CASCADE wins it, and by a clear margin — which
is what section 6.2 predicted: the cascade decides after seeing an attempt,
while the router must predict difficulty blind, and its errors concentrate
exactly where the decision matters.

Note also how little separates the weak and good routers on cost: both pay
router overhead and both send 40% of traffic to the large model by construction,
so the prediction quality shows up almost entirely in accuracy. A weak router is
not cheap-and-bad; it is the SAME price as a good one and worse.

Feature-level routing is the baseline worth taking seriously. It has no
overhead, no router errors, and it exploits a real fact — different product
features genuinely have different difficulty distributions. It loses here
because a fixed 60/40 split cannot adapt per request, and it should still be the
first thing tried, because it is a configuration change rather than a system.""")

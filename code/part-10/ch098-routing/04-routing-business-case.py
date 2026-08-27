# -*- coding: utf-8 -*-
# Extracted from: Chapter 98 — Model Routing and Model Selection
# Source: src/.../ch098-routing.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Sizing a routing project: saving, risk, and what has to be built."""

REQUESTS_PER_DAY = 2_000_000
TOKENS_PER_REQUEST = 800

MODELS = {
    "large (current)": dict(params=70e9, accuracy=0.94),
    "small":           dict(params=8e9,  accuracy=0.81),
}
GPU_HOUR, DEVICE_FLOPS, MFU = 2.50, 1e15, 0.45


def daily_cost(params, fraction=1.0):
    flops = 2 * params * REQUESTS_PER_DAY * fraction * TOKENS_PER_REQUEST
    hours = flops / (DEVICE_FLOPS * MFU) / 3600
    return hours * GPU_HOUR


large_only = daily_cost(MODELS["large (current)"]["params"])
small_only = daily_cost(MODELS["small"]["params"])
print(f"{REQUESTS_PER_DAY:,} requests/day at {TOKENS_PER_REQUEST} tokens\n")
print(f"{'baseline':<24} {'$/day':>10} {'$/year':>12} {'accuracy':>10}")
print(f"{'always large':<24} {large_only:>10,.0f} {large_only * 365:>12,.0f} "
      f"{MODELS['large (current)']['accuracy']:>10.2f}")
print(f"{'always small':<24} {small_only:>10,.0f} {small_only * 365:>12,.0f} "
      f"{MODELS['small']['accuracy']:>10.2f}")

ratio = MODELS["large (current)"]["params"] / MODELS["small"]["params"]
print(f"\ncost ratio: {ratio:.1f}x")
print(f"break-even escalation (eq:cascade-breakeven): {1 - 1 / ratio:.0%}")

print(f"\n{'escalation':>11} {'$/day':>10} {'$/year saved':>14} "
      f"{'est. accuracy':>14}")
for e in (0.15, 0.25, 0.40, 0.60):
    cost = small_only + e * large_only
    # Quality: the small model keeps the requests it is confident about, so its
    # accuracy on kept traffic exceeds its overall accuracy.
    kept_acc = min(0.81 + 0.10 * (1 - e), 0.93)
    acc = (1 - e) * kept_acc + e * 0.94
    print(f"{e:>11.0%} {cost:>10,.0f} {(large_only - cost) * 365:>14,.0f} "
          f"{acc:>14.4f}")

# What has to be built, and what it costs to be wrong.
print(f"\n{'requirement':<40} {'note'}")
REQUIREMENTS = [
    ("a confidence signal on the small model", "free if entropy suffices"),
    ("a labelled sample from both models", "one-off; a few thousand requests"),
    ("threshold stored as a RATE, not a value", "threshold-drift"),
    ("escalation-rate monitoring + alert", "the primary operational metric"),
    ("a fixed quality probe set", "detects silent degeneration"),
]
for req, note in REQUIREMENTS:
    print(f"{req:<40} {note}")

at_25 = small_only + 0.25 * large_only
print(f"""
At 25% escalation the saving is ${(large_only - at_25) * 365:,.0f}/year for an
estimated accuracy of about 0.93 against the large model's 0.94.

Whether one point of accuracy is worth that depends entirely on what the
requests are for — equation (eq:escalation-threshold) makes the value of a
correct answer an explicit input, and it is a product number rather than a
modelling one.

The operational requirements matter as much as the arithmetic. Without
escalation-rate monitoring the cascade drifts silently (threshold-drift), and
without a fixed probe set the drift is invisible until users complain — because
the direction that saves money is the same direction that loses quality.""")
